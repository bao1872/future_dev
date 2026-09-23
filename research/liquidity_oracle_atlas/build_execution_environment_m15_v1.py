"""build_execution_environment_m15_v1
=====================================

Canonical R4 15m Execution Environment owner — the SINGLE module that runs the
indicator math on the 15m execution axis for R4.

At every 15m decision bar close C_t (decision_time = 15m bar END):
  * m15    is COMMITTED  (the 15m bar is complete at its close)
  * h1, h4 are FORMING  (preview of the still-in-progress HTF bar)

This is what distinguishes R4 from R3: R3 made every TF (including 5m/m15)
a PREVIEW at the 5m close. Here the execution TF (m15) is a completed bar, so
its SR / Liquidity / DTP are the committed (post-close) state, and only the
higher HTFs are forming. The 5m axis does not appear at all.

One pass produces, per 15m decision:
  * per-TF DTP9 + SR/Liquidity features  (Viewer + later STRUCT33 dataset)
  * per-TF geometry (SR channels, Liquidity levels) = the EXACT structure the
    touch owner judges a bar against (the single source of candidate truth)
  * the 6-bit true-touch mask  (3TF x {SR, LIQ})
  * the per-trigger touch proof (exact hit zones)
  * the next-bar Candidate gate (via derive_m15_candidate_gate)

5m appears ONLY as the resample input and is never a decision / state / feature.

Reused canonical kernels (READ ONLY, no math copied):
  IndicatorState, precompute_forming_ohlc, resample_causal, bar_hits_zone.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.experiment_structural_reversion_pgm_v1 import (
    PINE_DEFAULT,
)
from research.liquidity_oracle_atlas.forming_indicator_state_v1 import (
    DISCRETE_COLS,
    FEATURE_COLS,
    IndicatorState,
)
from research.liquidity_oracle_atlas.build_forming_environment_v1 import (
    precompute_forming_ohlc,
)
from research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1 import (
    _build_seg_completed,
    bar_hits_zone,
    resample_causal,
)
from research.liquidity_oracle_atlas.build_execution_frame_m15_v1 import (
    build_execution_frame_m15,
)

# --------------------------------------------------------------------------- #
# R4 research-axis contract (no m5 anywhere)                                    #
# --------------------------------------------------------------------------- #
EXEC_TF = "m15"
TF_ORDER_R4 = ("m15", "h1", "h4")
TF_MINUTES_R4 = {"m15": 15, "h1": 60, "h4": 240}

# 3TF x {SR, LIQ} = 6 bits. No m5 bit exists.
MASK_BIT_R4 = {
    (tf, fam): ti * 2 + fi
    for ti, tf in enumerate(TF_ORDER_R4)
    for fi, fam in enumerate(("SR", "LIQ"))
}

# Candidate eligibility comes ONLY from the m15 SR / LIQ bit (hard-derived from
# MASK_BIT_R4 so it cannot drift).
M15_GATE_MASK = np.uint8(
    (1 << MASK_BIT_R4[("m15", "SR")]) | (1 << MASK_BIT_R4[("m15", "LIQ")])
)


# --------------------------------------------------------------------------- #
# Touch owner (bit + proof in the SAME bar_hits_zone pass)                      #
# --------------------------------------------------------------------------- #
def touch_from_prev_geometry_r4(
    low15: float,
    high15: float,
    prev_geom: Dict[str, Tuple],
    capture_proof: bool = False,
) -> Tuple[int, List[Dict[str, Any]]]:
    """R4 6-bit true-touch owner.

    ``prev_geom[tf] = (channels, liq_up, liq_down, atr)`` as known at the
    PREVIOUS 15m close (causal). ``channels`` is the canonical SR channel list
    ``(top, bottom, strength)``; ``liq_up/liq_down`` are liquidity level dicts
    with ``top`` / ``bottom`` / ``level`` / ``broken``.

    The bit and every provenance record are produced by the SAME ``bar_hits_zone``
    call inside the SAME loop — never a second divergent touch computation.
    """
    bits = 0
    matches: List[Dict[str, Any]] = []
    for tf in TF_ORDER_R4:
        g = prev_geom.get(tf)
        if g is None:
            continue
        channels, liq_up, liq_down = g[:3]

        for slot, (top, bottom, strength) in enumerate(channels):
            if bar_hits_zone(low15, high15, float(bottom), float(top)):
                bits |= 1 << MASK_BIT_R4[(tf, "SR")]
                if capture_proof:
                    matches.append({
                        "tf": tf,
                        "family": "SR",
                        "side": None,
                        "slot": int(slot),
                        "top": float(top),
                        "bottom": float(bottom),
                        "level": None,
                        "strength": float(strength),
                        "intersects": True,
                    })

        for side, levels in (("BUY", liq_up), ("SELL", liq_down)):
            for slot, z in enumerate(levels):
                if bool(z["broken"]):
                    continue
                if bar_hits_zone(low15, high15, float(z["bottom"]), float(z["top"])):
                    bits |= 1 << MASK_BIT_R4[(tf, "LIQ")]
                    if capture_proof:
                        matches.append({
                            "tf": tf,
                            "family": "LIQ",
                            "side": side,
                            "slot": int(slot),
                            "top": float(z["top"]),
                            "bottom": float(z["bottom"]),
                            "level": float(z["level"]),
                            "strength": None,
                            "intersects": True,
                        })

    return int(bits), matches


# --------------------------------------------------------------------------- #
# Candidate derivation (ONLY the m15 SR/LIQ bit grants eligibility)              #
# --------------------------------------------------------------------------- #
def derive_m15_candidate_gate(
    touch_bits: np.ndarray,
    segment: np.ndarray,
    trading_day: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Derive next-bar candidate membership + retained trigger context.

    Candidate[t+1] = same_unit(t,t+1) AND
                     (touch_bits[t] has 15m SR or 15m LIQ).
    Higher-TF touch never independently creates a Candidate but is preserved
    verbatim as candidate_trigger_bits (causal confluence / context).
    """
    touch_bits = np.asarray(touch_bits, dtype=np.uint8)
    segment = np.asarray(segment)
    trading_day = np.asarray(trading_day)

    n = len(touch_bits)

    prev_touch = np.zeros(n, dtype=np.uint8)
    if n > 1:
        prev_touch[1:] = touch_bits[:-1]

    same_unit = np.zeros(n, dtype=bool)
    if n > 1:
        same_unit[1:] = (
            (segment[1:] == segment[:-1])
            & (trading_day[1:] == trading_day[:-1])
        )

    m15_trigger = (prev_touch & M15_GATE_MASK) != 0
    candidate_any = same_unit & m15_trigger

    candidate_trigger_bits = np.where(
        candidate_any, prev_touch, 0
    ).astype(np.uint8)

    return {
        "candidate_any": candidate_any,
        "candidate_trigger_bits": candidate_trigger_bits,
        "prev_touch_bits": prev_touch,
        "same_unit": same_unit,
    }


def compute_candidate_episode_id(
    candidate_any: np.ndarray,
    same_unit: np.ndarray,
) -> np.ndarray:
    """Single canonical Candidate episode id (computed ONCE, shared everywhere).

    Mirrors the R3 rule: a new episode starts at a candidate bar that is the first
    candidate in the series or follows a non-candidate bar or a unit boundary.
    Non-candidate bars get -1.
    """
    c = np.asarray(candidate_any, dtype=bool)
    n = len(c)
    prev_c = np.r_[False, c[:-1]] if n > 1 else np.zeros(0, dtype=bool)
    episode_start = c & (~prev_c | ~np.asarray(same_unit, dtype=bool))
    eid = np.cumsum(episode_start, dtype=np.int64)
    out = np.full(n, -1, dtype=np.int64)
    out[c] = eid[c]
    return out


# --------------------------------------------------------------------------- #
# Single-pass streaming kernel (15m decision axis)                             #
# --------------------------------------------------------------------------- #
def run_environment_m15(
    symbol: str,
    max_bars: Optional[int] = None,
    capture_provenance: bool = True,
) -> Dict[str, Any]:
    """Run the canonical R4 15m environment for one symbol.

    Returns a dict with:
      exec_frame      : the 15m execution frame (DataFrame)
      features        : per-15m-decision DTP9 + SR/Liq features (DataFrame)
      geom_by_decision: list[tf->(channels, liq_up, liq_down, atr)] per bar
      touch_bits      : uint8 6-bit true-touch mask per 15m bar
      entry_matches   : list[list[dict]] of hit zones per bar (proof)
    """
    exec_frame = build_execution_frame_m15(symbol, max_bars)
    n = len(exec_frame)

    base = pd.DataFrame({
        "time": exec_frame["bar_start_time"].to_numpy(),
        "trading_day": exec_frame["trading_day"].to_numpy(),
        "segment": exec_frame["segment"].to_numpy(np.int64),
        "open": exec_frame["open"].to_numpy(float),
        "high": exec_frame["high"].to_numpy(float),
        "low": exec_frame["low"].to_numpy(float),
        "close": exec_frame["close"].to_numpy(float),
    })

    seg_arr = base["segment"].to_numpy(np.int64)
    O = base["open"].to_numpy(float)
    H = base["high"].to_numpy(float)
    L = base["low"].to_numpy(float)
    C = base["close"].to_numpy(float)

    # Per-TF forming-OHLC precompute + completed-bar bookkeeping (canonical).
    form = {
        tf: precompute_forming_ohlc(base, TF_MINUTES_R4[tf]) for tf in TF_ORDER_R4
    }
    seg_completed = {
        tf: _build_seg_completed(
            base, resample_causal(base, TF_MINUTES_R4[tf]), form[tf]
        )
        for tf in TF_ORDER_R4
    }

    tf_states = {
        tf: IndicatorState(PINE_DEFAULT, include_sr=True) for tf in TF_ORDER_R4
    }
    ci_per_tf = {tf: 0 for tf in TF_ORDER_R4}
    seg_list_per_tf = {tf: [] for tf in TF_ORDER_R4}

    arr: Dict[str, np.ndarray] = {}
    for tf in TF_ORDER_R4:
        for c in FEATURE_COLS:
            arr[f"{tf}_{c}"] = np.full(n, np.nan, dtype=float)

    touch_bits = np.zeros(n, dtype=np.uint8)
    entry_matches: Optional[List[List[Dict[str, Any]]]] = (
        [[] for _ in range(n)] if capture_provenance else None
    )
    geom_by_decision: List[Optional[Dict[str, Any]]] = [None] * n

    prev_geom: Optional[Dict[str, Tuple]] = None
    prev_seg: Optional[int] = None

    for i in range(n):
        seg = int(seg_arr[i])

        # ---- segment boundary: reset all TF states ---------------------------
        if seg != prev_seg:
            for tf in TF_ORDER_R4:
                tf_states[tf].reset()
                ci_per_tf[tf] = 0
                seg_list_per_tf[tf] = seg_completed[tf].get(seg, [])
            # canonical reset -> pre-existing structures do not survive a gap
            prev_geom = None
            prev_seg = seg

        # ---- per-TF commit (m15) / preview (h1, h4) -------------------------
        geom_by_tf: Dict[str, Tuple] = {}
        for tf in TF_ORDER_R4:
            f = form[tf]
            seg_list = seg_list_per_tf[tf]
            ci = ci_per_tf[tf]
            # advance commits for completed HTF bars strictly before i
            while ci < len(seg_list) and seg_list[ci][0] < i:
                bar = seg_list[ci][1]
                tf_states[tf].step(
                    ci, bar["open"], bar["high"], bar["low"], bar["close"]
                )
                ci += 1
            ci_per_tf[tf] = ci

            fo = float(f["open"][i])
            fh = float(f["high"][i])
            fl = float(f["low"][i])
            fc = float(f["close"][i])

            if tf == EXEC_TF:
                # m15 bar is COMPLETE at this close -> commit it. The committed
                # state after step(i) is the post-close m15 environment.
                feats = tf_states[tf].step(i, fo, fh, fl, fc)
                channels = list(tf_states[tf].sr.channels)
                liq_up = [dict(x) for x in tf_states[tf].liq.levels_up]
                liq_down = [dict(x) for x in tf_states[tf].liq.levels_down]
                # already committed; do not let the next iteration re-commit it
                ci_per_tf[tf] = i + 1
            else:
                # h1 / h4 still forming -> preview (never mutates committed state)
                pv = tf_states[tf].snapshot()
                feats = pv.step(i, fo, fh, fl, fc)
                channels = list(pv.sr.channels)
                liq_up = [dict(x) for x in pv.liq.levels_up]
                liq_down = [dict(x) for x in pv.liq.levels_down]

            atr = feats["atr"]
            for c in FEATURE_COLS:
                arr[f"{tf}_{c}"][i] = feats[c]
            geom_by_tf[tf] = (channels, liq_up, liq_down, atr)

        geom_by_decision[i] = geom_by_tf

        # ---- touch: current 15m bar vs geometry known at PREVIOUS 15m close --
        if prev_geom is not None:
            b, m = touch_from_prev_geometry_r4(
                float(L[i]), float(H[i]), prev_geom, capture_provenance
            )
            touch_bits[i] = np.uint8(b)
            if entry_matches is not None:
                entry_matches[i] = m
        # geometry computed once -> the NEXT 15m bar consumes it
        prev_geom = {
            tf: (
                geom_by_tf[tf][0],
                geom_by_tf[tf][1],
                geom_by_tf[tf][2],
                geom_by_tf[tf][3],
            )
            for tf in TF_ORDER_R4
        }

    # feature frame
    fdata = {"execution_bar_index": np.arange(n, dtype=np.int64)}
    for col, a in arr.items():
        fdata[col] = a
    features = pd.DataFrame(fdata)
    for tf in TF_ORDER_R4:
        for c in DISCRETE_COLS:
            col = f"{tf}_{c}"
            features[col] = features[col].astype("int64")

    return {
        "exec_frame": exec_frame,
        "features": features,
        "geom_by_decision": geom_by_decision,
        "touch_bits": touch_bits,
        "entry_matches": entry_matches,
    }
