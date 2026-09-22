"""
experiment_structure_interaction_entry_v1
==========================================

Research 1 — Structure Interaction Entry Study V1.

Causal event engine that answers ONE research question:

    After price approaches an already-existing Support / Resistance /
    Buyside-Liquidity / Sellside-Liquidity, which *causal price-interaction
    path* (combined with trend direction, trend slope, ATR deviation and
    structure context) corresponds to a stable, distinct future price
    distribution?

    This is an ``event -> future path`` experiment. It defines NO trading
    rule, NO entry/exit, NO ML, NO parameter search.

Design contract (frozen by the research contract):
  * Reuse canonical owners only; do NOT modify indicator math, Pine
    parameters, or forming-MTF time semantics.
  * Read canonical ``top/bottom`` geometry. SR uses ``sr.channels =
    (top, bottom, strength)``. Liquidity uses ``levels_up/down`` dicts
    with ``top/bottom`` (NOT ``liq_level_price``).
  * Four structure types (SUPPORT / RESISTANCE / BUYSIDE_LIQUIDITY /
    SELLSIDE_LIQUIDITY) are kept independent at the statistics layer.
  * Proximity radius ``R_near = 0.50 * ATR_TF`` (ATR200 of the structure
    timeframe) is a DATA-COLLECTION boundary only.
  * Atomic events are described by a completed 5m bar and may NEVER be
    mutated by a future bar.
  * Production path is single-pass streaming (commit completed HTF bars
    once + preview the forming bar). No per-decision history recompute,
    no slow-reference call on the production path.

This module is the EXECUTOR. It returns an Evidence Packet (event rows
+ counters + performance). It does NOT interpret research significance.

Canonical owners reused (NOT modified):
  * research.export_ob_trigger_execution_v21.load_raw_5m
  * research.phase1_tradability.phase1_contract_v1.discontinuity_flags
  * research.liquidity_oracle_atlas.experiment_structural_reversion_pgm_v1
        PINE_DEFAULT, raw_frame_from_owner, resample_causal
  * research.liquidity_oracle_atlas.build_forming_environment_v1
        precompute_forming_ohlc
  * research.liquidity_oracle_atlas.forming_indicator_state_v1
        IndicatorState (+ DTPState / SRState / LiquidityState)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from research.export_ob_trigger_execution_v21 import load_raw_5m
from research.phase1_tradability.phase1_contract_v1 import discontinuity_flags
from research.liquidity_oracle_atlas.experiment_structural_reversion_pgm_v1 import (
    PINE_DEFAULT,
    raw_frame_from_owner,
    resample_causal,
)
from research.liquidity_oracle_atlas.build_forming_environment_v1 import (
    precompute_forming_ohlc,
)
from research.liquidity_oracle_atlas.forming_indicator_state_v1 import (
    IndicatorState,
)


# --------------------------------------------------------------------------- #
# Frozen constants (from the research contract)                                #
# --------------------------------------------------------------------------- #
TF_ORDER = ["m5", "m15", "h1", "h4"]
TF_MINUTES = {"m5": 5, "m15": 15, "h1": 60, "h4": 240}

ROLES = [
    "SUPPORT",
    "RESISTANCE",
    "BUYSIDE_LIQUIDITY",
    "SELLSIDE_LIQUIDITY",
]

REV_SIGN = {
    "SUPPORT": +1,
    "SELLSIDE_LIQUIDITY": +1,
    "RESISTANCE": -1,
    "BUYSIDE_LIQUIDITY": -1,
}

# Frozen 8-bit mapping for the DP entry mask (contract §7 / §8).
# bit position = tf_index * 2 + family_index  ->  4 TF x {SR, LIQ} = 8 bits.
# NOTE: the DP entry mask deliberately carries NO Support/Resistance/Buyside/
# Sellside semantics; those are re-joined later only to INTERPRET the oracle.
MASK_BIT = {
    (tf, family): ti * 2 + fi
    for ti, tf in enumerate(TF_ORDER)
    for fi, family in enumerate(("SR", "LIQ"))
}

# Data-collection proximity radius (ATR units). Frozen. Never tuned.
NEAR_ATR = 0.50

# Oracle V2 entry-proximity radius (ATR units of the structure timeframe).
# Frozen for R2; the 0.25/0.5/0.75/1.0 sensitivity study is a SEPARATE task.
ENTRY_PROX_ATR = 0.50

# Outcome horizons (5m bars). Frozen.
HORIZONS = (6, 12, 24)

_NAN = float("nan")


# --------------------------------------------------------------------------- #
# Counters (mirrors build_forming_environment_v1.RunStats)                     #
# --------------------------------------------------------------------------- #
@dataclass
class KernelCounters:
    """Production-path instrumentation.

    Hard contract gates (TP / complexity):
      raw_load_count == 1
      resample_count == 4
      full_history_recompute_count == 0
      reference_call_count == 0
      concat_count == 0   (hot-loop concat forbidden)
    """

    raw_load_count: int = 0
    resample_count: int = 0
    state_step_count: int = 0
    preview_count: int = 0
    full_history_recompute_count: int = 0
    reference_call_count: int = 0
    concat_count: int = 0
    # DP-side accounting (incremented by the trade-oracle DP runner, not here)
    dp_state_count: int = 0
    # Event-engine mechanical counters. On the DP mask-only fast path all three
    # MUST stay 0 (proves the episode/event machinery is not executed).
    event_role_iteration_count: int = 0
    event_classifier_call_count: int = 0
    outcome_call_count: int = 0


# --------------------------------------------------------------------------- #
# Episode state                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class Episode:
    """Single interaction episode with one structure instance.

    Geometry (near_edge / far_edge / strength / structure_id / start_atr) is
    FROZEN at start and never retroactively moved by later owner updates.
    """

    symbol: str
    tf: str
    role: str

    segment: int
    structure_id: str

    start_i: int
    start_time: Any

    near_edge: float
    far_edge: float
    strength: float

    rev_sign: int
    start_atr: float

    touched: bool = False
    entered: bool = False
    has_broken: bool = False
    had_partial_reclaim: bool = False

    break_extreme_u: float = _NAN

    phase: str = "APPROACH"

    bars_alive: int = 0
    visit_count: int = 1

    prev_close_u: float = _NAN

    # --- lifecycle / audit extras ---
    termination_reason: Optional[str] = None
    gap_into_episode: bool = False
    approach_dists: List[float] = field(default_factory=list)
    first_event_emitted: bool = False
    events_emitted: int = 0
    approach_bars: int = 0
    approach_velocity: float = 0.0
    approach_monotone: float = 0.0
    approach_max_retrace: float = 0.0

    # --- structure metadata (frozen at start) ---
    is_liq: bool = False
    liq_left: Optional[int] = None
    liq_level: Optional[float] = None
    top: float = _NAN
    bottom: float = _NAN


# --------------------------------------------------------------------------- #
# Reference classifier (contract §33) — used by T0 / differential core check  #
# --------------------------------------------------------------------------- #
def classify_reference(prev, O, H, L, C, ep: Episode) -> Optional[str]:
    """Independent, SIMPLER reference classifier.

    Core boundary semantics only. Production ``classify_bar`` adds the
    break-phase annotations (EXTENSION / RETURNING / HOLD / REBREAK) but
    must agree with this on the core event set.
    """
    s = ep.rev_sign

    close_u = s * C
    low_u = min(s * H, s * L)

    near_u = s * ep.near_edge
    far_u = s * ep.far_edge

    touched = low_u <= near_u
    broke = low_u < far_u

    if broke:
        if close_u > near_u:
            return "SAME_BAR_FULL_RECLAIM"
        if close_u >= far_u:
            return "SAME_BAR_PARTIAL_RECLAIM"
        if not ep.has_broken:
            return "FIRST_BREAK"

    if not ep.has_broken:
        if touched and close_u > near_u:
            return "TOUCH_REJECT"
        if touched and far_u <= close_u <= near_u:
            return "ENTER_ZONE"

    if ep.has_broken:
        if close_u > near_u:
            return "DELAYED_FULL_RECLAIM"
        if close_u >= far_u:
            return "DELAYED_PARTIAL_RECLAIM"

    return None


# --------------------------------------------------------------------------- #
# Production classifier (contract §35) — single source of events on prod path  #
# --------------------------------------------------------------------------- #
def classify_bar(ep: Episode, O, H, L, C) -> Optional[str]:
    """Production atomic-event classifier. DO NOT change these semantics."""
    s = ep.rev_sign

    open_u = s * O
    high_u = max(s * H, s * L)
    low_u = min(s * H, s * L)
    close_u = s * C

    near_u = s * ep.near_edge
    far_u = s * ep.far_edge

    touched = low_u <= near_u
    broke_now = low_u < far_u

    # --- break / reclaim on this bar ---
    if broke_now:
        ep.touched = True
        ep.entered = True

        if close_u > near_u:
            if ep.has_broken:
                event = "DELAYED_FULL_RECLAIM"
            else:
                event = "SAME_BAR_FULL_RECLAIM"

            ep.has_broken = True
            ep.phase = "FULL_RECLAIM"
            ep.prev_close_u = close_u
            return event

        if close_u >= far_u:
            if ep.has_broken:
                event = "DELAYED_PARTIAL_RECLAIM"
            else:
                event = "SAME_BAR_PARTIAL_RECLAIM"

            ep.has_broken = True
            ep.had_partial_reclaim = True
            ep.phase = "PARTIAL_RECLAIM"
            ep.prev_close_u = close_u
            return event

        # closes beyond far edge
        if not ep.has_broken:
            ep.has_broken = True
            ep.break_extreme_u = low_u
            ep.phase = "BREAK"
            ep.prev_close_u = close_u
            return "FIRST_BREAK"

    # --- pre-break ---
    if not ep.has_broken:
        if touched:
            ep.touched = True

            if close_u > near_u:
                ep.phase = "REJECTED"
                ep.prev_close_u = close_u
                return "ZONE_REJECT" if ep.entered else "TOUCH_REJECT"

            if close_u >= far_u:
                first_enter = not ep.entered
                ep.entered = True
                ep.phase = "ZONE"
                ep.prev_close_u = close_u

                if first_enter:
                    return "ENTER_ZONE"

        ep.prev_close_u = close_u
        return None

    # --- already broken, still on break side ---
    if close_u < far_u:
        if low_u < ep.break_extreme_u:
            ep.break_extreme_u = low_u

            if ep.phase != "BREAK_EXTENSION":
                ep.phase = "BREAK_EXTENSION"
                ep.prev_close_u = close_u
                return "BREAK_EXTENSION"

        if math.isfinite(ep.prev_close_u) and close_u > ep.prev_close_u:
            if ep.phase != "BREAK_RETURNING":
                ep.phase = "BREAK_RETURNING"
                ep.prev_close_u = close_u
                return "BREAK_RETURNING"

        if ep.phase != "BREAK_HOLD":
            ep.phase = "BREAK_HOLD"
            ep.prev_close_u = close_u
            return "BREAK_HOLD"

        ep.prev_close_u = close_u
        return None

    # --- rebreak after partial reclaim ---
    if ep.had_partial_reclaim and close_u < far_u:
        ep.phase = "REBREAK"
        ep.prev_close_u = close_u
        return "REBREAK_AFTER_PARTIAL"

    ep.prev_close_u = close_u
    return None


# --------------------------------------------------------------------------- #
# Geometry / candidate selection (contract §3 / §7 / §14)                     #
# --------------------------------------------------------------------------- #
def select_target(
    role: str,
    channels: List[Tuple[float, float, float]],
    liq_up: List[Dict[str, Any]],
    liq_down: List[Dict[str, Any]],
    close: float,
    atr_tf: float,
    tf: str,
    seg: int,
    i: int,
    sr_first_seen: Dict[Tuple, int],
) -> Optional[Dict[str, Any]]:
    """Pick the NEAREST candidate structure for ``role`` at this decision.

    Returns frozen geometry (near_edge / far_edge) with correct orientation:
      SUPPORT / SELLSIDE_LIQUIDITY : near_edge = top,   far_edge = bottom
      RESISTANCE / BUYSIDE_LIQUIDITY: near_edge = bottom, far_edge = top
    so that u = s * P always satisfies u_near > u_far.
    """
    if role == "SUPPORT":
        cands = [z for z in channels if z[0] < close]
        if not cands:
            return None
        z = max(cands, key=lambda q: q[0])
        top, bottom, strength = z
        near_edge, far_edge = top, bottom
    elif role == "RESISTANCE":
        cands = [z for z in channels if z[1] > close]
        if not cands:
            return None
        z = min(cands, key=lambda q: q[1])
        top, bottom, strength = z
        near_edge, far_edge = bottom, top
    elif role == "BUYSIDE_LIQUIDITY":
        cands = [d for d in liq_up if (not d["broken"]) and d["bottom"] > close]
        if not cands:
            return None
        z = min(cands, key=lambda q: q["bottom"])
        top, bottom = z["top"], z["bottom"]
        strength = 1.0
        near_edge, far_edge = bottom, top
        left, level = z["left"], z["level"]
    elif role == "SELLSIDE_LIQUIDITY":
        cands = [d for d in liq_down if (not d["broken"]) and d["top"] < close]
        if not cands:
            return None
        z = max(cands, key=lambda q: q["top"])
        top, bottom = z["top"], z["bottom"]
        strength = 1.0
        near_edge, far_edge = top, bottom
        left, level = z["left"], z["level"]
    else:
        return None

    if role in ("SUPPORT", "RESISTANCE"):
        key = (
            tf,
            seg,
            round(top, 6),
            round(bottom, 6),
            round(strength, 4),
        )
        if key not in sr_first_seen:
            sr_first_seen[key] = i
        fs = sr_first_seen[key]
        sid = f"SR|{tf}|{seg}|{fs}|{top}|{bottom}|{strength}"
        is_liq = False
        liq_left = None
        liq_level = None
    else:
        sid = f"LIQ|{tf}|{role}|{seg}|{left}|{level}"
        is_liq = True
        liq_left = int(left)
        liq_level = float(level)

    return {
        "role": role,
        "structure_id": sid,
        "near_edge": near_edge,
        "far_edge": far_edge,
        "strength": strength,
        "start_atr": atr_tf,
        "is_liq": is_liq,
        "liq_left": liq_left,
        "liq_level": liq_level,
        "top": top,
        "bottom": bottom,
    }


def in_proximity(cand: Dict[str, Any], O, H, L, C, atr: float) -> bool:
    """Contract §8: start episode when distance <= 0.5 ATR, or the bar
    touches / is already inside the structure (gap into zone).

    NOTE: this governs the EVENT/episode machine only. The DP entry mask uses
    ``entry_bits_from_prev_geometry`` (true touch against pre-existing zones,
    delta=0), NOT this proximity radius.
    """
    s = REV_SIGN[cand["role"]]
    close = C
    u = s * close
    u_near = s * cand["near_edge"]
    low_u = min(s * H, s * L)
    touch = low_u <= u_near
    if u >= u_near:
        dist = (u - u_near) / atr
        approach_ok = dist <= NEAR_ATR
    else:
        approach_ok = True  # inside / through the structure
    return touch or approach_ok


# --------------------------------------------------------------------------- #
# DP entry mask (contract §7 / §8): 8-bit TF x {SR, LIQ}, delta = 0 touch      #
# against structures that were ALREADY KNOWN at the previous 5m close.         #
# --------------------------------------------------------------------------- #
def bar_hits_zone(low5: float, high5: float, bottom: float, top: float) -> bool:
    """True iff the current 5m bar range [low5, high5] intersects [bottom, top]."""
    return high5 >= bottom and low5 <= top


def entry_bits_from_prev_geometry(
    low5: float,
    high5: float,
    prev_geom_by_tf: Dict[str, Tuple],
) -> int:
    """8-bit entry mask: current 5m range touches a PRE-EXISTING SR / Liquidity.

    ``prev_geom_by_tf[tf] = (channels, liq_up, liq_down)`` where ``channels`` is
    the canonical ``sr.channels`` list of ``(top, bottom, strength)`` and
    ``liq_up/liq_down`` are canonical liquidity level dicts with ``top`` /
    ``bottom`` / ``broken``.

    ONLY the geometry known at the PREVIOUS 5m close may qualify (causal): a
    structure formed by the current bar can never retroactively make the
    current bar eligible. ``broken`` liquidity is excluded.
    """
    bits = 0
    for tf in TF_ORDER:
        g = prev_geom_by_tf.get(tf)
        if g is None:
            continue
        channels, liq_up, liq_down, *_ = g

        hit_sr = any(
            bar_hits_zone(low5, high5, float(bottom), float(top))
            for top, bottom, _strength in channels
        )
        if hit_sr:
            bits |= 1 << MASK_BIT[(tf, "SR")]

        hit_liq = any(
            (not bool(z["broken"]))
            and bar_hits_zone(low5, high5, float(z["bottom"]), float(z["top"]))
            for z in (*liq_up, *liq_down)
        )
        if hit_liq:
            bits |= 1 << MASK_BIT[(tf, "LIQ")]

    return int(bits)


# --------------------------------------------------------------------------- #
# Oracle V2 proximity kernel (contract FUTURE-INTRADAY-DP-ORACLE-R2-*)         #
# Distance-based proximity: a bar is "near" a structure when its range lies     #
# within ENTRY_PROX_ATR of ANY pre-existing SR / Liquidity zone. This is the   #
# DP entry gate for V2 (NOT the true-touch entry_bits_from_prev_geometry).      #
# --------------------------------------------------------------------------- #
def bar_zone_distance(
    low5: float,
    high5: float,
    bottom: float,
    top: float,
) -> float:
    """Distance between the current 5m price interval and a structure zone.

    d = 0  when the range truly touches / is inside the zone,
    d = L - top  when the price is ABOVE the zone,
    d = bottom - H  when the price is BELOW the zone.
    """
    return max(
        float(bottom) - float(high5),
        float(low5) - float(top),
        0.0,
    )


def proximity_bits_from_prev_geometry(
    low5: float,
    high5: float,
    prev_geom_by_tf: dict,
    *,
    alpha: float = ENTRY_PROX_ATR,
) -> int:
    """8-bit (TF x {SR, LIQ}) proximity mask computed from the geometry known at
    the PREVIOUS 5m close.

    A bit is set iff the current 5m range [low5, high5] lies within
    ``alpha * ATR_tf`` of the corresponding pre-existing SR / Liquidity zone.
    A structure formed by the CURRENT bar can never retroactively qualify the
    current bar because only ``prev_geom_by_tf`` (previous close) is consulted.
    Broken liquidity is excluded. ``prev_geom_by_tf[tf]`` is the 4-tuple
    ``(channels, liq_up, liq_down, atr_tf)``.
    """
    bits = 0
    for tf in TF_ORDER:
        g = prev_geom_by_tf.get(tf)
        if g is None:
            continue

        channels, liq_up, liq_down, atr_tf = g

        if not (np.isfinite(atr_tf) and atr_tf > 0):
            continue

        radius = float(alpha) * float(atr_tf)

        hit_sr = False
        for top, bottom, _strength in channels:
            if bar_zone_distance(low5, high5, bottom, top) <= radius:
                hit_sr = True
                break
        if hit_sr:
            bits |= 1 << MASK_BIT[(tf, "SR")]

        hit_liq = False
        for z in liq_up:
            if bool(z.get("broken")):
                continue
            if bar_zone_distance(low5, high5, z["bottom"], z["top"]) <= radius:
                hit_liq = True
                break
        if not hit_liq:
            for z in liq_down:
                if bool(z.get("broken")):
                    continue
                if bar_zone_distance(low5, high5, z["bottom"], z["top"]) <= radius:
                    hit_liq = True
                    break
        if hit_liq:
            bits |= 1 << MASK_BIT[(tf, "LIQ")]

    return int(bits)


# --------------------------------------------------------------------------- #
# Slow reference structure snapshot (contract §32) — T0 / T1 ONLY             #
# --------------------------------------------------------------------------- #
def slow_preview_state_reference(
    base: pd.DataFrame, i: int, minutes: int
) -> Dict[str, Any]:
    """Independent SLOW structure snapshot for decision ``i``.

    Rebuilds the IndicatorState from ``base.iloc[:i+1]`` (current segment
    only) by prefix recompute. Used ONLY by T0 / T1 differential. The
    production path MUST NOT call this.
    """
    prefix = base.iloc[: i + 1].copy()

    time_idx = pd.DatetimeIndex(prefix["time"])
    bucket = time_idx.floor(f"{minutes}min")
    cur_bucket = bucket[-1]

    seg = int(prefix["segment"].iloc[-1])
    x = prefix[prefix["segment"] == seg].copy()

    tf = resample_causal(x, minutes)
    tf = tf[tf["time"] < cur_bucket]

    cur = x[
        pd.DatetimeIndex(x["time"]).floor(f"{minutes}min") == cur_bucket
    ]
    if cur.empty:
        forming = dict(open=_NAN, high=_NAN, low=_NAN, close=_NAN)
    else:
        forming = dict(
            open=float(cur["open"].iloc[0]),
            high=float(cur["high"].max()),
            low=float(cur["low"].min()),
            close=float(cur["close"].iloc[-1]),
        )

    st = IndicatorState(PINE_DEFAULT, include_sr=True)
    k = 0
    for row in tf.itertuples():
        st.step(k, row.open, row.high, row.low, row.close)
        k += 1
    feats = st.step(k, forming["open"], forming["high"], forming["low"], forming["close"])

    return {
        "sr_channels": list(st.sr.channels),
        "liq_up": [dict(x) for x in st.liq.levels_up],
        "liq_down": [dict(x) for x in st.liq.levels_down],
        "atr": feats["atr"],
        "dev": feats["dev"],
        "slope_atr": feats["slope_atr"],
        "trend_state": feats["trend_state"],
    }


# --------------------------------------------------------------------------- #
# Data pipeline (contract §2) — reuse canonical owners                        #
# --------------------------------------------------------------------------- #
def build_base_from_arrays(
    time_arr,
    trading_day_arr,
    o,
    h,
    l,
    c,
    disc,
    counters: KernelCounters,
) -> Dict[str, Any]:
    """Build base frame + per-TF completed/forming bars from raw arrays.

    Reuses canonical ``raw_frame_from_owner``, ``resample_causal`` and
    ``precompute_forming_ohlc``. Increments ``resample_count`` by len(TF_ORDER).
    Used by both the real-data loader and synthetic T0 tests.
    """
    bars = dict(
        n=len(o),
        t=np.asarray(time_arr),
        day=np.asarray(trading_day_arr),
        disc=np.asarray(disc, dtype=bool),
        o=np.asarray(o, dtype=float),
        h=np.asarray(h, dtype=float),
        l=np.asarray(l, dtype=float),
        c=np.asarray(c, dtype=float),
    )
    base = raw_frame_from_owner(bars)
    counters.resample_count += len(TF_ORDER)

    form: Dict[str, Dict[str, np.ndarray]] = {}
    seg_completed: Dict[str, Dict[int, List[Tuple[int, Dict[str, float]]]]] = {}
    for tf in TF_ORDER:
        minutes = TF_MINUTES[tf]
        completed = resample_causal(base, minutes)
        form[tf] = precompute_forming_ohlc(base, minutes)
        seg_completed[tf] = _build_seg_completed(base, completed, form[tf])

    return {"base": base, "form": form, "seg_completed": seg_completed}


def build_base_frame(symbol: str, counters: KernelCounters) -> Dict[str, Any]:
    """Load once (canonical owner), then build frames via build_base_from_arrays.

    Counters: raw_load_count += 1 (build_base_from_arrays adds resample_count).
    """
    counters.raw_load_count += 1

    raw = load_raw_5m(symbol).sort_values("bar_start_time").reset_index(drop=True)
    disc = np.asarray(discontinuity_flags(symbol), dtype=bool)

    if len(raw) != len(disc):
        raise SystemExit("STOP_RAW_DISC_LENGTH_MISMATCH")

    return build_base_from_arrays(
        pd.to_datetime(raw["bar_start_time"]).to_numpy(),
        pd.to_datetime(raw["trading_day"]).to_numpy(),
        raw["open"].to_numpy(float),
        raw["high"].to_numpy(float),
        raw["low"].to_numpy(float),
        raw["close"].to_numpy(float),
        disc,
        counters,
    )


def _build_seg_completed(
    base: pd.DataFrame,
    completed: pd.DataFrame,
    form: Dict[str, np.ndarray],
) -> Dict[int, List[Tuple[int, Dict[str, float]]]]:
    """Per-segment list of (last_base_index, ohlc_dict) for completed TF bars.

    Mirrors build_forming_environment_v1.FormingEnvironmentBuilder.prepare:
    a completed TF bar's membership is keyed by (trading_day, segment,
    bucket); its last base index is the commit trigger (committed when
    ``last_base_index < decision_i``).
    """
    n = len(base)
    seg_arr = base["segment"].to_numpy(np.int64)
    td_arr = base["trading_day"].to_numpy()
    bucket_arr = form["bucket_start"]

    li: Dict[Tuple[int, np.datetime64, np.datetime64], int] = {}
    for j in range(n):
        key = (int(seg_arr[j]), td_arr[j], bucket_arr[j])
        li[key] = j

    out: Dict[int, List[Tuple[int, Dict[str, float]]]] = {}
    for _, row in completed.iterrows():
        key = (
            int(row["segment"]),
            np.datetime64(row["trading_day"]),
            np.datetime64(row["time"]),
        )
        lj = li.get(key)
        if lj is None:
            continue
        out.setdefault(int(row["segment"]), []).append(
            (
                int(lj),
                dict(
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                ),
            )
        )
    for s in out:
        out[s].sort(key=lambda x: x[0])
    return out


# --------------------------------------------------------------------------- #
# Outcome (contract §20-§22) — causal isolation, outcome_* namespace only      #
# --------------------------------------------------------------------------- #
def compute_outcome(
    s: int,
    i: int,
    O5: np.ndarray,
    H5: np.ndarray,
    L5: np.ndarray,
    C5: np.ndarray,
    atr5m: np.ndarray,
) -> Dict[str, float]:
    """Reversal-normalized future path from O_{t+1} over H in {6,12,24}.

    ret_rev_H = s * (C_{t+H} - O_{t+1}) / ATR5m_t
    MFE/MAE in reversal direction, normalized by ATR5m_t.
    First-passage of +/-0.5 and +/-1.0 ATR (rev direction); same-bar
    double-touch -> AMBIGUOUS.
    """
    out: Dict[str, float] = {}
    a0 = atr5m[i] if (i < len(atr5m) and np.isfinite(atr5m[i])) else _NAN

    for H in HORIZONS:
        h = f"ret_rev_{H}"
        mfe = f"mfe_rev_{H}"
        mae = f"mae_rev_{H}"
        fp05 = f"first_passage_05_{H}"
        fp10 = f"first_passage_10_{H}"
        lat = f"label_available_time_{H}"
        if not np.isfinite(a0) or i + 1 >= len(C5) or i + H >= len(C5):
            out[h] = _NAN
            out[mfe] = _NAN
            out[mae] = _NAN
            out[fp05] = _NAN
            out[fp10] = _NAN
            out[lat] = _NAN
            continue

        entry = O5[i + 1]
        end = C5[i + H]
        ret = s * (end - entry) / a0
        out[h] = ret

        # MFE / MAE over [i+1, i+H]
        hi = H5[i + 1 : i + H + 1]
        lo = L5[i + 1 : i + H + 1]
        mfe_v = s * (np.max(hi) - entry) / a0
        mae_v = s * (np.min(lo) - entry) / a0
        out[mfe] = mfe_v
        out[mae] = mae_v

        # first passage
        p05 = _first_passage(s, entry, hi, lo, a0, 0.5)
        p10 = _first_passage(s, entry, hi, lo, a0, 1.0)
        out[fp05] = p05
        out[fp10] = p10
        out[lat] = float(H * 5)  # minutes until label is knowable

    return out


def _first_passage(
    s: int, entry: float, hi: np.ndarray, lo: np.ndarray, a0: float, lvl: float
) -> float:
    """Return +1 (target side, favorable), -1 (stop side, adverse), or 0
    (AMBIGUOUS: both +lvl and -lvl touched in the same 5m bar)."""
    for k in range(len(hi)):
        high_rev = s * (hi[k] - entry) / a0
        low_rev = s * (lo[k] - entry) / a0
        hit_target = high_rev >= lvl
        hit_stop = low_rev <= -lvl
        if hit_target and hit_stop:
            return 0.0  # AMBIGUOUS
        if hit_target:
            return 1.0
        if hit_stop:
            return -1.0
    return _NAN


# --------------------------------------------------------------------------- #
# Confluence (contract §19) — recorded only, never defines events             #
# --------------------------------------------------------------------------- #
def _nearest_same_side_dist(
    role: str,
    channels: List[Tuple[float, float, float]],
    liq_up: List[Dict[str, Any]],
    liq_down: List[Dict[str, Any]],
    close: float,
    atr_o: float,
) -> Optional[float]:
    s = REV_SIGN[role]
    best = None
    if role in ("SUPPORT", "RESISTANCE"):
        if role == "SUPPORT":
            cands = [z for z in channels if z[0] < close]
            edges = [z[0] for z in cands]
        else:
            cands = [z for z in channels if z[1] > close]
            edges = [z[1] for z in cands]
    else:
        if role == "BUYSIDE_LIQUIDITY":
            cands = [d for d in liq_up if (not d["broken"]) and d["bottom"] > close]
            edges = [d["bottom"] for d in cands]
        else:
            cands = [d for d in liq_down if (not d["broken"]) and d["top"] < close]
            edges = [d["top"] for d in cands]
    for e in edges:
        d = abs(close - e) / atr_o
        if best is None or d < best:
            best = d
    return best


def confluence_for(
    tf: str,
    role: str,
    geom_by_tf: Dict[str, Tuple],
    close: float,
) -> Dict[str, float]:
    same_025 = 0
    same_050 = 0
    nearest_same = _NAN
    nearest_opp = _NAN
    for oth in TF_ORDER:
        if oth == tf:
            continue
        channels, liq_up, liq_down, atr_o = geom_by_tf[oth]
        if not (np.isfinite(atr_o) and atr_o > 0):
            continue
        d = _nearest_same_side_dist(role, channels, liq_up, liq_down, close, atr_o)
        if d is not None:
            if not np.isfinite(nearest_same) or d < nearest_same:
                nearest_same = d
            if d <= 0.25:
                same_025 += 1
            if d <= 0.50:
                same_050 += 1
        # opposing side: take the opposite-role nearest distance
        opp_role = _opposing_role(role)
        dd = _nearest_same_side_dist(
            opp_role, channels, liq_up, liq_down, close, atr_o
        )
        if dd is not None and (not np.isfinite(nearest_opp) or dd < nearest_opp):
            nearest_opp = dd
    return {
        "nearest_same_side_structure_dist_atr": nearest_same,
        "nearest_opposing_structure_dist_atr": nearest_opp,
        "same_side_structure_count_025atr": float(same_025),
        "same_side_structure_count_050atr": float(same_050),
    }


def _opposing_role(role: str) -> str:
    return {
        "SUPPORT": "RESISTANCE",
        "RESISTANCE": "SUPPORT",
        "BUYSIDE_LIQUIDITY": "SELLSIDE_LIQUIDITY",
        "SELLSIDE_LIQUIDITY": "BUYSIDE_LIQUIDITY",
    }[role]


# --------------------------------------------------------------------------- #
# Episode helpers                                                              #
# --------------------------------------------------------------------------- #
def start_episode(
    symbol: str,
    tf: str,
    role: str,
    cand: Dict[str, Any],
    i: int,
    seg: int,
    t,
    C: float,
) -> Episode:
    s = REV_SIGN[role]
    u = s * C
    u_near = s * cand["near_edge"]
    gap = bool(u <= u_near)  # started already at/inside/through, not from far side
    return Episode(
        symbol=symbol,
        tf=tf,
        role=role,
        segment=seg,
        structure_id=cand["structure_id"],
        start_i=i,
        start_time=t,
        near_edge=cand["near_edge"],
        far_edge=cand["far_edge"],
        strength=cand["strength"],
        rev_sign=s,
        start_atr=cand["start_atr"],
        touched=False,
        entered=False,
        has_broken=False,
        had_partial_reclaim=False,
        break_extreme_u=_NAN,
        phase="APPROACH",
        bars_alive=0,
        visit_count=1,
        prev_close_u=_NAN,
        termination_reason=None,
        gap_into_episode=gap,
        approach_dists=[],
        first_event_emitted=False,
        is_liq=cand["is_liq"],
        liq_left=cand["liq_left"],
        liq_level=cand["liq_level"],
        top=cand["top"],
        bottom=cand["bottom"],
    )


def recedes_without_touch(ep: Episode, C: float, atr_tf: float) -> bool:
    """Contract §9 / §13: an episode that never interacted (no event emitted,
    never touched, never entered) and whose price has receded beyond the
    proximity radius terminates as ``APPROACH_NO_TOUCH_REJECT`` (a CENSORED
    REJECTED_AWAY). Geometry uses the FROZEN ``start_atr``.

    Returns True iff this is exactly that terminal condition.
    """
    atr = ep.start_atr
    if not (np.isfinite(atr) and atr > 0):
        atr = atr_tf
    if not (np.isfinite(atr) and atr > 0):
        return False
    s = ep.rev_sign
    u = s * C
    u_near = s * ep.near_edge
    receded = u - u_near > NEAR_ATR * atr
    return bool(
        receded
        and ep.events_emitted == 0
        and not ep.touched
        and not ep.entered
    )


def track_approach(ep: Episode, C: float, atr_tf: float) -> None:
    if ep.first_event_emitted:
        return
    s = ep.rev_sign
    u = s * C
    u_near = s * ep.near_edge
    atr = ep.start_atr if (np.isfinite(ep.start_atr) and ep.start_atr > 0) else atr_tf
    if atr > 0:
        dist = (u - u_near) / atr if u >= u_near else 0.0
    else:
        dist = 0.0
    ep.approach_dists.append(dist)


def finalize_approach(ep: Episode) -> None:
    d = ep.approach_dists
    if not d:
        ep.approach_bars = 0
        ep.approach_velocity = 0.0
        ep.approach_monotone = 0.0
        ep.approach_max_retrace = 0.0
        return
    ep.approach_bars = len(d)
    d0 = d[0]
    dlast = d[-1]
    elapsed = max(1, len(d) - 1)
    ep.approach_velocity = (d0 - dlast) / elapsed
    dec = 0
    for k in range(1, len(d)):
        if d[k] < d[k - 1]:
            dec += 1
    ep.approach_monotone = dec / (len(d) - 1) if len(d) > 1 else 0.0
    mr = 0.0
    for k in range(1, len(d)):
        mr = max(mr, d[k] - d[k - 1])
    ep.approach_max_retrace = mr


def event_bar_geometry(ep: Episode, O, H, L, C) -> Dict[str, float]:
    s = ep.rev_sign
    atr = ep.start_atr
    u_near = s * ep.near_edge
    u_far = s * ep.far_edge
    u_min = min(s * H, s * L)  # oriented extreme toward break side
    width = (u_near - u_far)
    if atr > 0 and width > 0:
        pen_frac = (u_near - u_min) / width
        pen_atr = (u_near - u_min) / atr
        wick = max(0.0, u_near - u_min) / atr
    else:
        pen_frac = _NAN
        pen_atr = _NAN
        wick = _NAN
    close_loc = (s * C - u_near) / width if width > 0 else _NAN
    return {
        "penetration_atr": pen_atr,
        "penetration_zone_fraction": pen_frac,
        "body_atr": abs(C - O) / atr if atr > 0 else _NAN,
        "range_atr": (H - L) / atr if atr > 0 else _NAN,
        "close_location_oriented": close_loc,
        "wick_into_structure_atr": wick,
    }


# --------------------------------------------------------------------------- #
# Production streaming kernel (contract §34 / §35 / §36)                       #
# --------------------------------------------------------------------------- #
def run_symbol_streaming(
    symbol: str,
    counters: KernelCounters,
    max_bars: Optional[int] = None,
    capture_geom: bool = False,
    capture_entry_mask: bool = False,
    emit_events: bool = True,
    mask_only: bool = False,
    capture_proximity: bool = False,
) -> Dict[str, Any]:
    """Single-pass causal event engine for one symbol (real data path).

    O(N * TF * bounded_state) + O(E * H). No per-decision history recompute,
    no slow-reference call, no hot-loop concat.

    ``capture_geom`` records the per-decision production geometry for the T1
    differential harness. ``capture_entry_mask`` records the frozen 8-bit
    entry mask (contract §7 / §8): current 5m range touching a PRE-EXISTING
    SR / Liquidity zone. ``capture_proximity`` records the Oracle V2 distance-
    based proximity mask (within ENTRY_PROX_ATR of a pre-existing SR / Liquidity
    zone) used as the V2 DP entry gate. ``emit_events=False`` skips event-row
    construction (no ``compute_outcome``). ``mask_only=True`` is the true DP
    fast path: the (tf, role) episode/event lifecycle is NOT executed at all and
    no DTP context arrays are stored. None of these call the slow reference.
    """
    info = build_base_frame(symbol, counters)
    return _stream_from_base(
        info["base"], info["form"], info["seg_completed"], counters,
        max_bars, symbol, capture_geom, capture_entry_mask, emit_events,
        mask_only, capture_proximity,
    )


def stream_from_base(
    base: pd.DataFrame,
    counters: KernelCounters,
    max_bars: Optional[int] = None,
    symbol: str = "SYNTH",
    capture_geom: bool = False,
    capture_entry_mask: bool = False,
    emit_events: bool = True,
    mask_only: bool = False,
    capture_proximity: bool = False,
) -> Dict[str, Any]:
    """Streaming entry point for a prebuilt base frame (synthetic T0 tests).

    Reuses canonical precompute_forming_ohlc / resample_causal and increments
    resample_count by len(TF_ORDER).
    """
    form = {tf: precompute_forming_ohlc(base, TF_MINUTES[tf]) for tf in TF_ORDER}
    seg_completed = {
        tf: _build_seg_completed(
            base, resample_causal(base, TF_MINUTES[tf]), form[tf]
        )
        for tf in TF_ORDER
    }
    counters.resample_count += len(TF_ORDER)
    return _stream_from_base(
        base, form, seg_completed, counters, max_bars, symbol, capture_geom,
        capture_entry_mask, emit_events, mask_only, capture_proximity,
    )


def _stream_from_base(
    base: pd.DataFrame,
    form: Dict[str, Dict[str, np.ndarray]],
    seg_completed: Dict[str, Dict[int, List[Tuple[int, Dict[str, float]]]]],
    counters: KernelCounters,
    max_bars: Optional[int] = None,
    symbol: str = "SYNTH",
    capture_geom: bool = False,
    capture_entry_mask: bool = False,
    emit_events: bool = True,
    mask_only: bool = False,
    capture_proximity: bool = False,
    capture_atr5m: bool = False,
) -> Dict[str, Any]:
    """Core single-pass streaming loop (shared by all entry points)."""
    n = len(base)
    if max_bars is not None:
        n = min(n, int(max_bars))

    seg_arr = base["segment"].to_numpy(np.int64)
    time_arr = base["time"].to_numpy()
    O5 = base["open"].to_numpy(float)[:n]
    H5 = base["high"].to_numpy(float)[:n]
    L5 = base["low"].to_numpy(float)[:n]
    C5 = base["close"].to_numpy(float)[:n]

    # per-TF committed state + preview bookkeeping
    tf_states = {tf: IndicatorState(PINE_DEFAULT, include_sr=True) for tf in TF_ORDER}
    cur_seg_per_tf = {tf: None for tf in TF_ORDER}
    ci_per_tf = {tf: 0 for tf in TF_ORDER}
    seg_list_per_tf = {tf: [] for tf in TF_ORDER}

    # per-decision DTP context per TF + m5 atr for outcomes. These are ONLY
    # needed by the event experiment; the DP mask-only path neither allocates
    # nor stores them (contract FIX1).
    if mask_only:
        dtp_ctx = None
        # NEW (cost robustness): allow capturing the already-computed m5 ATR even
        # on the mask-only fast path. Default False preserves old behaviour
        # (atr5m stays None, no extra array).
        atr5m = np.full(n, _NAN) if capture_atr5m else None
        active: Dict[Tuple[str, str], Optional[Episode]] = {}
    else:
        dtp_ctx = {
            tf: {
                "dev": np.full(n, _NAN),
                "slope_atr": np.full(n, _NAN),
                "trend_state": np.full(n, _NAN),
                "atr": np.full(n, _NAN),
            }
            for tf in TF_ORDER
        }
        atr5m = np.full(n, _NAN)
        active = {(tf, role): None for tf in TF_ORDER for role in ROLES}

    sr_first_seen: Dict[Tuple, int] = {}

    events: List[Dict[str, Any]] = []
    decision_geom: List[Dict[str, Any]] = [] if capture_geom else None
    entry_mask = np.zeros(n, dtype=np.uint16) if capture_entry_mask else None
    proximity_bits = np.zeros(n, dtype=np.uint16) if capture_proximity else None
    proximity_any = np.zeros(n, dtype=bool) if capture_proximity else None
    # geometry known at the PREVIOUS 5m close (causal entry mask input)
    prev_geom_by_tf: Optional[Dict[str, Tuple]] = None
    prev_seg = None

    for i in range(n):
        seg = int(seg_arr[i])

        # ---- segment boundary: terminate episodes (CENSORED) + reset states
        if seg != prev_seg:
            for key, ep in active.items():
                if ep is not None:
                    ep.termination_reason = "SEGMENT_END"
                    active[key] = None
            for tf in TF_ORDER:
                tf_states[tf].reset()
                cur_seg_per_tf[tf] = seg
                ci_per_tf[tf] = 0
                seg_list_per_tf[tf] = seg_completed[tf].get(seg, [])
            # Canonical IndicatorState reset -> the pre-existing structure set
            # must reset too (the first bar of a new segment can never touch a
            # previous segment's structures).
            prev_geom_by_tf = None
            prev_seg = seg

        # ---- per-TF commit + preview (geometry snapshot)
        geom_by_tf: Dict[str, Tuple] = {}
        for tf in TF_ORDER:
            minutes = TF_MINUTES[tf]
            f = form[tf]
            seg_list = seg_list_per_tf[tf]
            ci = ci_per_tf[tf]
            while ci < len(seg_list) and seg_list[ci][0] < i:
                bar = seg_list[ci][1]
                tf_states[tf].step(ci, bar["open"], bar["high"], bar["low"], bar["close"])
                counters.state_step_count += 1
                ci += 1
            ci_per_tf[tf] = ci

            fo = float(f["open"][i])
            fh = float(f["high"][i])
            fl = float(f["low"][i])
            fc = float(f["close"][i])
            pv = tf_states[tf].snapshot()
            feats = pv.step(ci, fo, fh, fl, fc)
            counters.preview_count += 1

            atr_tf = feats["atr"]
            if not mask_only:
                dtp_ctx[tf]["dev"][i] = feats["dev"]
                dtp_ctx[tf]["slope_atr"][i] = feats["slope_atr"]
                dtp_ctx[tf]["trend_state"][i] = feats["trend_state"]
                dtp_ctx[tf]["atr"][i] = atr_tf
            # NEW (cost robustness): capture the m5 ATR whenever atr5m is
            # allocated (both full and mask_only+capture_atr5m paths). This is
            # the causal ATR5m[t] used by the cost kernel; it is a byproduct of
            # the canonical preview, so it adds no extra computation.
            if atr5m is not None and tf == "m5":
                atr5m[i] = atr_tf

            channels = list(pv.sr.channels)
            liq_up = [dict(x) for x in pv.liq.levels_up]
            liq_down = [dict(x) for x in pv.liq.levels_down]
            geom_by_tf[tf] = (channels, liq_up, liq_down, atr_tf)

        if capture_geom:
            decision_geom.append({tf: geom_by_tf[tf] for tf in TF_ORDER})

        capture_geom_prev = capture_entry_mask or capture_proximity
        if capture_geom_prev:
            # current 5m bar range vs geometry known at the PREVIOUS close
            if capture_entry_mask:
                entry_mask[i] = np.uint16(
                    entry_bits_from_prev_geometry(
                        float(L5[i]), float(H5[i]), prev_geom_by_tf or {}
                    )
                )
            if capture_proximity:
                pb = int(
                    proximity_bits_from_prev_geometry(
                        float(L5[i]), float(H5[i]), prev_geom_by_tf or {},
                        alpha=ENTRY_PROX_ATR,
                    )
                )
                proximity_bits[i] = np.uint16(pb)
                proximity_any[i] = bool(pb != 0)
            # geometry computed once -> the NEXT 5m bar consumes it
            # (now carries ATR so the V2 proximity gate can scale by TF ATR)
            prev_geom_by_tf = {
                tf: (
                    geom_by_tf[tf][0],
                    geom_by_tf[tf][1],
                    geom_by_tf[tf][2],
                    geom_by_tf[tf][3],
                )
                for tf in TF_ORDER
            }

        # DP mask-only fast path: geometry + mask only. The (tf, role) episode /
        # event lifecycle is deliberately NOT executed and no DTP context is
        # stored (contract FIX1).
        if mask_only:
            continue

        # ---- per (tf, role) episode lifecycle + event labeling
        for tf in TF_ORDER:
            channels, liq_up, liq_down, atr_tf = geom_by_tf[tf]
            for role in ROLES:
                counters.event_role_iteration_count += 1
                key = (tf, role)
                ep = active[key]

                if not (np.isfinite(atr_tf) and atr_tf > 0):
                    continue

                O = O5[i]
                H = H5[i]
                L = L5[i]
                C = C5[i]

                cand = select_target(
                    role, channels, liq_up, liq_down, C, atr_tf, tf, seg, i, sr_first_seen
                )
                # Proximity is evaluated exactly ONCE per (tf, role) and reused
                # for the episode state machine (start / owner-changed). This is
                # a behaviour-preserving refactor; it must not change any event
                # row. NOTE: it no longer feeds the DP entry mask (that mask is
                # the 8-bit touch mask computed above, contract §7 / §8).
                prox = cand is not None and in_proximity(cand, O, H, L, C, atr_tf)

                if ep is None:
                    if prox:
                        ep = start_episode(symbol, tf, role, cand, i, seg, time_arr[i], C)
                        active[key] = ep
                    else:
                        continue

                # structure owner changed -> CENSORED termination
                if cand is not None and cand["structure_id"] != ep.structure_id:
                    ep.termination_reason = "STRUCTURE_OWNER_CHANGED"
                    active[key] = None
                    if prox:
                        ep = start_episode(
                            symbol, tf, role, cand, i, seg, time_arr[i], C
                        )
                        active[key] = ep
                    else:
                        continue

                # classify this bar
                counters.event_classifier_call_count += 1
                ev = classify_bar(ep, O, H, L, C)
                track_approach(ep, C, atr_tf)
                if ev is not None:
                    if not ep.first_event_emitted:
                        finalize_approach(ep)
                        ep.first_event_emitted = True
                    if emit_events:
                        counters.outcome_call_count += 1
                        row = build_event_row(
                            ep, tf, role, f"{role}_{ev}", i, time_arr[i], dtp_ctx, geom_by_tf,
                            C, atr5m, O5, H5, L5, C5,
                        )
                        events.append(row)
                    ep.events_emitted += 1

                # termination (never on the start bar)
                if i != ep.start_i:
                    s = ep.rev_sign
                    u = s * C
                    u_near = s * ep.near_edge
                    u_far = s * ep.far_edge
                    atr = ep.start_atr
                    if i == n - 1:
                        ep.termination_reason = "DATA_END"
                        active[key] = None
                    elif recedes_without_touch(ep, C, atr):
                        # no-touch approach that receded without any interaction
                        if emit_events:
                            counters.outcome_call_count += 1
                            row = build_event_row(
                                ep, tf, role, f"{role}_APPROACH_NO_TOUCH_REJECT", i,
                                time_arr[i], dtp_ctx, geom_by_tf, C, atr5m,
                                O5, H5, L5, C5,
                            )
                            events.append(row)
                        ep.termination_reason = "REJECTED_AWAY"
                        active[key] = None
                    elif u - u_near > NEAR_ATR * atr:
                        ep.termination_reason = "REJECTED_AWAY"
                        active[key] = None
                    elif u_far - u > NEAR_ATR * atr:
                        ep.termination_reason = "BROKE_AWAY"
                        active[key] = None

    result: Dict[str, Any] = {
        "symbol": symbol,
        "n": n,
        "events": events,
        "dtp_ctx": dtp_ctx,
        "atr5m": atr5m,
        "mask_only": mask_only,
    }
    if capture_geom:
        result["decision_geom"] = decision_geom
    if capture_entry_mask:
        result["entry_mask"] = entry_mask
        result["entry_eligible"] = entry_mask != 0
    if capture_proximity:
        result["proximity_bits"] = proximity_bits
        result["proximity_any"] = proximity_any
    return result


def build_event_row(
    ep: Episode,
    tf: str,
    role: str,
    ev: str,
    i: int,
    t,
    dtp_ctx: Dict[str, Any],
    geom_by_tf: Dict[str, Tuple],
    C: float,
    atr5m: np.ndarray,
    O5: np.ndarray,
    H5: np.ndarray,
    L5: np.ndarray,
    C5: np.ndarray,
) -> Dict[str, Any]:
    s = ep.rev_sign
    atr = ep.start_atr

    row: Dict[str, Any] = {
        "symbol": ep.symbol,
        "structure_tf": tf,
        "structure_type": role,
        "structure_id": ep.structure_id,
        "episode_id": f"{ep.symbol}_{tf}_{role}_{ep.start_i}",
        "event_seq": None,  # assigned later if needed
        "decision_bar_index": i,
        "decision_time": pd.Timestamp(t) + pd.Timedelta(minutes=5),
        "event_type": ev,
        "near_edge": ep.near_edge,
        "far_edge": ep.far_edge,
        "structure_strength": ep.strength,
        "visit_count": ep.visit_count,
        "gap_into_episode": int(ep.gap_into_episode),
        "had_partial_reclaim_before": int(ep.had_partial_reclaim),
        # approach geometry (frozen at first event)
        "bars_from_proximity_to_event": ep.approach_bars,
        "approach_velocity": ep.approach_velocity,
        "approach_monotone_fraction": ep.approach_monotone,
        "max_retrace_atr": ep.approach_max_retrace,
        # structure context
        "structure_top": ep.top,
        "structure_bottom": ep.bottom,
        "structure_mid": 0.5 * (ep.top + ep.bottom) if np.isfinite(ep.top) else _NAN,
        "structure_width_atr": (ep.top - ep.bottom) / atr if atr > 0 else _NAN,
        "liq_level": ep.liq_level,
        "liq_left": ep.liq_left,
        "broken_before_episode": 0,
    }

    # 4-TF DTP context at decision i
    for tfx in TF_ORDER:
        row[f"{tfx}_dev"] = dtp_ctx[tfx]["dev"][i]
        row[f"{tfx}_slope_atr"] = dtp_ctx[tfx]["slope_atr"][i]
        row[f"{tfx}_trend_state"] = dtp_ctx[tfx]["trend_state"][i]
        row[f"{tfx}_atr"] = dtp_ctx[tfx]["atr"][i]

    # event-bar geometry
    eg = event_bar_geometry(ep, O5[i], H5[i], L5[i], C5[i])
    row.update(eg)

    # confluence (recorded only)
    conf = confluence_for(tf, role, geom_by_tf, C)
    row.update(conf)

    # outcome (causal isolation: only future, outcome_* namespace)
    oc = compute_outcome(s, i, O5, H5, L5, C5, atr5m)
    out_row = {f"outcome_{k}": v for k, v in oc.items()}
    row.update(out_row)

    return row


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def run_stage_kernel(
    symbols: List[str],
    max_bars: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the production kernel over a list of symbols.

    Returns aggregated events + the production-path counters for the
    Evidence Packet. The slow reference is NEVER called here.
    """
    counters = KernelCounters()
    all_events: List[Dict[str, Any]] = []
    per_symbol: Dict[str, int] = {}
    for sym in symbols:
        res = run_symbol_streaming(sym, counters, max_bars=max_bars)
        all_events.extend(res["events"])
        per_symbol[sym] = len(res["events"])
    return {
        "events": all_events,
        "counters": counters,
        "per_symbol": per_symbol,
    }


def events_to_dataframe(events: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(events)
