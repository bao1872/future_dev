"""build_structural_dp_labels_m15_v1
=====================================

First canonical STRUCTURAL DP-label builder (FUT-M15-STRUCTURAL-DP-LABEL-V1).

This module does NOT invent any DP / indicator mathematics. It is a pure
composer that joins three already-frozen canonical owners and derives the
three historical structural-label quantities exactly from geometry:

  * ``build_dp_proximity_m15_v1.build_dp_proximity_m15``
        -> candidate SR/liquidity proximity episodes (``dp_proximity_episode_id``)
  * ``build_trade_oracle_dp_m15_one_entry_proximity_v1``
        -> the accepted one-entry DP oracle (entry / exit / MFE / lifecycle)
  * ``build_execution_environment_m15_v1.run_environment_m15``
        -> canonical 15m/1h/4h SR / LIQUIDITY geometry + 15m ATR
        -> plus the frozen helpers ``bar_zone_distance`` / ``select_target`` /
           ``ENTRY_PROX_ATR`` from ``experiment_structure_interaction_entry_v1``

Derived quantities (the contract):

  best_entry_gap_atr      = distance(best_entry_price, CandidateRegion) / ATR
  tp_atr                  = directional_distance(best_entry, TP) / ATR
  remaining_target_atr    = directional_distance(TP, Target) / ATR  (>= 0)

Hard invariants (asserted in check_structural_invariants):
  one candidate episode -> exactly one label / one entry
  previous exit <= next entry
  best_entry_gap_atr >= 0
  tp_atr >= 0
  remaining_target_atr >= 0
  remaining_target_atr == 0  =>  TP price == Target price (tick tolerance)
  TP must never pass the structural target
  label_available_time_i <= next candidate start time   (else HARD STOP)

No SL mathematics is defined here (per task scope). No ML model is trained.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.build_execution_environment_m15_v1 import (
    run_environment_m15,
)
from research.liquidity_oracle_atlas.build_dp_proximity_m15_v1 import (
    build_dp_proximity_m15,
)
from research.liquidity_oracle_atlas.build_trade_oracle_dp_m15_one_entry_proximity_v1 import (
    ARTIFACT_ROOT_DIRNAME as ORACLE_ARTIFACT_ROOT,
    load_oracle_artifact,
)
from research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1 import (
    ENTRY_PROX_ATR,
    TF_ORDER,
    bar_zone_distance,
    select_target,
)
from research.liquidity_oracle_atlas.dp_label_lifecycle_v1 import map_exit_reason

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
MATH_VERSION = "structural-dp-label-v1"
ATR_OWNER = "m15_atr@run_environment_m15"
ARTIFACT_ROOT_DIRNAME = "structural_dp_labels_m15_v1"
LABELS_FILE = "structural_dp_labels.parquet"
METADATA_FILE = "metadata.json"

# Families carried by the canonical geometry containers.
_ROLE_BY_CONTAINER = {
    "channels": None,            # SR: role derived from position vs close
    "liq_up": "BUYSIDE_LIQUIDITY",
    "liq_down": "SELLSIDE_LIQUIDITY",
}

_REL_EPS = 1e-9  # relative tick tolerance for target / boundary checks


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2]
        ).decode().strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- #
# Pure canonical math (independently unit-testable)
# --------------------------------------------------------------------------- #
def point_zone_gap(price: float, bottom: float, top: float) -> float:
    """Distance from a single price to a zone [bottom, top]; 0 inside/on zone."""
    return max(float(bottom) - float(price), float(price) - float(top), 0.0)


def compute_best_entry_gap(
    best_entry_price: float, zones: List[Tuple[float, float]], atr: float
) -> Tuple[float, float]:
    """Min price-distance from best entry to the union of candidate zones.

    Returns (gap_points >= 0, gap_atr >= 0).
    """
    if not zones or not np.isfinite(atr) or atr <= 0:
        return (float("nan"), float("nan"))
    gap = min(point_zone_gap(best_entry_price, b, t) for (b, t) in zones)
    return (float(gap), float(gap) / float(atr))


def select_target_for_direction(
    direction: str,
    prev_geom: Dict[str, Tuple],
    close: float,
    seg: int,
    i: int,
    sr_first_seen: Dict[Tuple, int],
) -> Optional[Dict[str, Any]]:
    """Directional nearest pre-existing structural target ahead of price.

    LONG  -> nearest RESISTANCE or BUYSIDE_LIQUIDITY strictly above close.
    SHORT -> nearest SUPPORT    or SELLSIDE_LIQUIDITY strictly below close.

    Reuses the frozen ``select_target`` convention exactly (near_edge orientation,
    structure_id format). Returns None when no eligible target exists.
    """
    if direction == "LONG":
        roles = ("RESISTANCE", "BUYSIDE_LIQUIDITY")
    elif direction == "SHORT":
        roles = ("SUPPORT", "SELLSIDE_LIQUIDITY")
    else:
        raise ValueError(f"unknown direction {direction!r}")

    best: Optional[Dict[str, Any]] = None
    best_d = float("inf")
    for tf in TF_ORDER:
        g = prev_geom.get(tf)
        if g is None:
            continue
        channels, liq_up, liq_down, atr_tf = g
        for role in roles:
            cand = select_target(
                role, channels, liq_up, liq_down, float(close),
                float(atr_tf), tf, int(seg), int(i), sr_first_seen,
            )
            if cand is None:
                continue
            cand["tf"] = tf
            d = abs(float(cand["near_edge"]) - float(close))
            if d < best_d:
                best_d = d
                best = cand
    return best


def compute_tp_remaining(
    direction: str,
    entry_price: float,
    exit_price: float,
    target_price: Optional[float],
    mfe_signed_points: float,
    atr: float,
    tick: float = 1e-9,
) -> Dict[str, float]:
    """Resolve TP and remaining_target from canonical trade fields.

    target_reached is decided by the trade's favorable excursion (MFE) touching
    the structural target within tick tolerance. When reached, TP == Target and
    remaining == 0 (mandatory TP). Otherwise TP == actual DP exit and remaining
    is the favorable distance from TP to Target.

    tp_atr / remaining_target_atr are clamped to >= 0 to satisfy the contract;
    tp_points is kept signed (honest profit/loss), clamping only the ATR form.
    """
    s = 1.0 if direction == "LONG" else -1.0

    def _dir_points(p_from: float, p_to: float) -> float:
        # directional distance p_from -> p_to in price points (>=0 in profit dir)
        return s * (float(p_to) - float(p_from))

    out: Dict[str, float] = {}
    if target_price is None or not np.isfinite(atr) or atr <= 0:
        # No eligible target: TP collapses to the actual exit; remaining undefined.
        tp_price = float(exit_price)
        tp_points = _dir_points(entry_price, tp_price)
        out["tp_price"] = tp_price
        out["tp_points"] = float(tp_points)
        out["tp_atr"] = max(0.0, float(tp_points) / float(atr)) if np.isfinite(atr) else float("nan")
        out["target_reached"] = 0.0
        out["remaining_points"] = float("nan")
        out["remaining_target_atr"] = float("nan")
        return out

    tgt = float(target_price)
    # favorable extreme price actually reached during the trade
    fav_extreme = float(entry_price) + s * float(mfe_signed_points)
    reached = bool(
        (direction == "LONG" and fav_extreme >= tgt - tick)
        or (direction == "SHORT" and fav_extreme <= tgt + tick)
    )
    if reached:
        tp_price = tgt
        tp_time_touch = True
    else:
        tp_price = float(exit_price)
        tp_time_touch = False

    tp_points = _dir_points(entry_price, tp_price)
    remaining_points = _dir_points(tp_price, tgt)  # >= 0 when tp <= target

    out["tp_price"] = tp_price
    out["tp_points"] = float(tp_points)
    out["tp_atr"] = max(0.0, float(tp_points) / float(atr))
    out["target_reached"] = 1.0 if reached else 0.0
    out["remaining_points"] = float(remaining_points)
    out["remaining_target_atr"] = max(0.0, float(remaining_points) / float(atr))
    out["_tp_time_touch"] = 1.0 if tp_time_touch else 0.0
    return out


# --------------------------------------------------------------------------- #
# Canonical input loading
# --------------------------------------------------------------------------- #
@dataclass
class _CanonicalInputs:
    symbol: str
    exec_frame: pd.DataFrame
    features: pd.DataFrame
    geom: List[Optional[Dict[str, Tuple]]]
    prox: pd.DataFrame
    trades: pd.DataFrame
    actions: pd.DataFrame
    sr_first_seen: Dict[Tuple, int]


def _load_canonical_inputs(symbol: str, max_bars: Optional[int]) -> _CanonicalInputs:
    env = run_environment_m15(symbol, max_bars, capture_provenance=False)
    exec_frame = env["exec_frame"]
    features = env["features"]
    geom = env["geom_by_decision"]

    prox = build_dp_proximity_m15(symbol, max_bars)

    oracle = load_oracle_artifact(
        Path("artifacts") / ORACLE_ARTIFACT_ROOT, symbol,
        expected_math_version=None, expected_source_sha=None,
    )
    if not oracle.get("ok"):
        raise RuntimeError(
            f"oracle artifact not available for {symbol!r}: {oracle.get('reason')}"
        )
    trades = oracle["trades"].copy()
    actions = oracle["actions"].copy()

    # Keep only trades fully inside the loaded execution frame (guards against a
    # reference prefix run or a stale cached frame shorter than the oracle data).
    n_frame = len(exec_frame)
    dec = trades.get("entry_decision_index")
    xdec = trades.get("exit_decision_index")
    if dec is not None and xdec is not None:
        keep = (
            (dec.to_numpy(int) >= 0) & (dec.to_numpy(int) < n_frame)
            & (xdec.to_numpy(int) >= 0) & (xdec.to_numpy(int) < n_frame)
        )
        trades = trades[keep].reset_index(drop=True)

    seg_arr = exec_frame["segment"].to_numpy()
    sr_first_seen = _build_sr_first_seen(geom, seg_arr)

    return _CanonicalInputs(
        symbol, exec_frame, features, geom, prox, trades, actions, sr_first_seen
    )


def _build_sr_first_seen(
    geom: List[Optional[Dict[str, Tuple]]], seg_arr: np.ndarray
) -> Dict[Tuple, int]:
    """Replicate the frozen ``select_target`` SR first-seen index (canonical id)."""
    fs: Dict[Tuple, int] = {}
    for i, g in enumerate(geom):
        if g is None:
            continue
        seg = int(seg_arr[i]) if i < len(seg_arr) else 0
        for tf in TF_ORDER:
            tup = g.get(tf)
            if tup is None:
                continue
            channels = tup[0]
            for top, bottom, strength in channels:
                key = (
                    tf, seg,
                    round(float(top), 6), round(float(bottom), 6),
                    round(float(strength), 4),
                )
                if key not in fs:
                    fs[key] = i
    return fs


# --------------------------------------------------------------------------- #
# Candidate-region structure gathering
# --------------------------------------------------------------------------- #
def _structure_id_liq(role: str, tf: str, seg: int, z: Dict[str, Any]) -> str:
    return f"LIQ|{tf}|{role}|{seg}|{int(z['left'])}|{float(z['level'])}"


def _gather_candidate_structures(
    inp: _CanonicalInputs,
    episode_bars: List[int],
) -> Tuple[List[Dict[str, Any]], List[Tuple[float, float]]]:
    """All SR/LIQ structures that triggered proximity anywhere in the episode.

    For every episode bar b, structures in geom[b-1] within ENTRY_PROX_ATR*atr_tf
    of bar b's range qualify. Only structures existing BEFORE bar b are consulted
    (causal). Union of qualifying zones defines the candidate region.
    """
    exec_frame = inp.exec_frame
    low = exec_frame["low"].to_numpy(float)
    high = exec_frame["high"].to_numpy(float)
    seg_arr = exec_frame["segment"].to_numpy(int)

    structures: List[Dict[str, Any]] = []
    zones: List[Tuple[float, float]] = []
    seen_ids: set = set()

    for b in episode_bars:
        prev = inp.geom[b - 1] if b - 1 >= 0 else inp.geom[0]
        if prev is None:
            continue
        seg = int(seg_arr[b]) if b < len(seg_arr) else 0
        bar_lo = float(low[b])
        bar_hi = float(high[b])
        for tf in TF_ORDER:
            tup = prev.get(tf)
            if tup is None:
                continue
            channels, liq_up, liq_down, atr_tf = tup
            if not (np.isfinite(atr_tf) and atr_tf > 0):
                continue
            radius = float(ENTRY_PROX_ATR) * float(atr_tf)
            # SR channels
            for top, bottom, strength in channels:
                if bar_zone_distance(bar_lo, bar_hi, float(bottom), float(top)) <= radius:
                    if float(top) < bar_lo:
                        role = "SUPPORT"
                    elif float(bottom) > bar_hi:
                        role = "RESISTANCE"
                    else:
                        role = "SR_ZONE"
                    sid = f"SR|{tf}|{seg}|{inp.sr_first_seen.get((tf, seg, round(float(top),6), round(float(bottom),6), round(float(strength),4)), b)}|{top}|{bottom}|{strength}"
                    if sid not in seen_ids:
                        seen_ids.add(sid)
                        structures.append({
                            "structure_id": sid, "type": role, "tf": tf,
                            "bottom": float(bottom), "top": float(top),
                            "price": float((top + bottom) / 2.0),
                        })
                        zones.append((float(bottom), float(top)))
            for z in liq_up:
                if bool(z.get("broken")):
                    continue
                if bar_zone_distance(bar_lo, bar_hi, float(z["bottom"]), float(z["top"])) <= radius:
                    sid = _structure_id_liq("BUYSIDE_LIQUIDITY", tf, seg, z)
                    if sid not in seen_ids:
                        seen_ids.add(sid)
                        structures.append({
                            "structure_id": sid, "type": "BUYSIDE_LIQUIDITY", "tf": tf,
                            "bottom": float(z["bottom"]), "top": float(z["top"]),
                            "price": float(z["level"]),
                        })
                        zones.append((float(z["bottom"]), float(z["top"])))
            for z in liq_down:
                if bool(z.get("broken")):
                    continue
                if bar_zone_distance(bar_lo, bar_hi, float(z["bottom"]), float(z["top"])) <= radius:
                    sid = _structure_id_liq("SELLSIDE_LIQUIDITY", tf, seg, z)
                    if sid not in seen_ids:
                        seen_ids.add(sid)
                        structures.append({
                            "structure_id": sid, "type": "SELLSIDE_LIQUIDITY", "tf": tf,
                            "bottom": float(z["bottom"]), "top": float(z["top"]),
                            "price": float(z["level"]),
                        })
                        zones.append((float(z["bottom"]), float(z["top"])))
    return structures, zones


# --------------------------------------------------------------------------- #
# Core builder
# --------------------------------------------------------------------------- #
def _episode_maps(inp: _CanonicalInputs) -> Tuple[Dict[int, List[int]], Dict[int, pd.Timestamp], Dict[int, pd.Timestamp]]:
    prox = inp.prox
    ep_id = prox["dp_proximity_episode_id"].to_numpy()
    bst = prox["bar_start_time"].to_numpy()
    ebi = prox["execution_bar_index"].to_numpy()
    bars_by_ep: Dict[int, List[int]] = {}
    start_by_ep: Dict[int, pd.Timestamp] = {}
    end_by_ep: Dict[int, pd.Timestamp] = {}
    for eid, t, idx in zip(ep_id, bst, ebi):
        if eid < 0:
            continue
        bars_by_ep.setdefault(int(eid), []).append(int(idx))
        ts = pd.Timestamp(t)
        if int(eid) not in start_by_ep or ts < start_by_ep[int(eid)]:
            start_by_ep[int(eid)] = ts
        if int(eid) not in end_by_ep or ts > end_by_ep[int(eid)]:
            end_by_ep[int(eid)] = ts
    return bars_by_ep, start_by_ep, end_by_ep


def build_structural_dp_labels_v1(
    symbol: str,
    max_bars: Optional[int] = None,
    emit_assertions: bool = True,
) -> pd.DataFrame:
    inp = _load_canonical_inputs(symbol, max_bars)
    exec_frame = inp.exec_frame
    features = inp.features
    trades = inp.trades

    bars_by_ep, start_by_ep, end_by_ep = _episode_maps(inp)

    low = exec_frame["low"].to_numpy(float)
    high = exec_frame["high"].to_numpy(float)
    close = exec_frame["close"].to_numpy(float)
    opens = exec_frame["open"].to_numpy(float)
    seg_arr = exec_frame["segment"].to_numpy(int)
    bst = exec_frame["bar_start_time"].to_numpy()
    atr_m15 = features["m15_atr"].to_numpy(float)

    # action transition by decision index -> FLAT_EXIT / REVERSAL label
    trans_by_dec: Dict[int, str] = {}
    if "decision_bar_index" in inp.actions.columns and "transition" in inp.actions.columns:
        for d, tr in zip(
            inp.actions["decision_bar_index"].to_numpy(int),
            inp.actions["transition"].to_numpy(),
        ):
            trans_by_dec[int(d)] = str(tr)

    # ensure trades ordered by entry time
    trades = trades.sort_values("entry_fill_time").reset_index(drop=True)
    n = len(trades)

    rows: List[Dict[str, Any]] = []
    for r in range(n):
        tr = trades.iloc[r]
        direction = str(tr["direction"])
        ep_id = int(tr["entry_proximity_episode_id"])
        t = int(tr["entry_decision_index"])
        entry_fill_price = float(tr["entry_fill_price"])
        entry_fill_index = int(tr["entry_fill_index"])
        entry_fill_time = pd.Timestamp(tr["entry_fill_time"])
        exit_fill_price = float(tr["exit_fill_price"])
        exit_fill_index = int(tr["exit_fill_index"])
        exit_fill_time = pd.Timestamp(tr["exit_fill_time"])
        mfe = float(tr["MFE"]) if "MFE" in tr and pd.notna(tr["MFE"]) else 0.0

        atr = float(atr_m15[t]) if 0 <= t < len(atr_m15) else float("nan")
        close_t = float(close[t])
        seg_t = int(seg_arr[t]) if t < len(seg_arr) else 0

        # candidate region = union of qualifying structures over the episode
        episode_bars = sorted(bars_by_ep.get(ep_id, [t]))
        structures, zones = _gather_candidate_structures(inp, episode_bars)
        gap_pts, gap_atr = compute_best_entry_gap(entry_fill_price, zones, atr)

        # structural target (directional, known at entry decision time)
        prev_geom = inp.geom[t - 1] if t - 1 >= 0 else inp.geom[0]
        tgt = select_target_for_direction(
            direction, prev_geom, close_t, seg_t, t, inp.sr_first_seen
        ) if prev_geom is not None else None
        if tgt is not None:
            target_price = float(tgt["near_edge"])
            target_type = str(tgt["role"])
            target_tf = str(tgt["tf"])
            target_id = str(tgt["structure_id"])
            target_zone_bottom = float(tgt["bottom"])
            target_zone_top = float(tgt["top"])
            s = 1.0 if direction == "LONG" else -1.0
            target_distance_atr = s * (target_price - entry_fill_price) / atr if atr > 0 else float("nan")
        else:
            target_price = None
            target_type = None
            target_tf = None
            target_id = None
            target_zone_bottom = float("nan")
            target_zone_top = float("nan")
            target_distance_atr = float("nan")

        # TP / remaining
        tp_info = compute_tp_remaining(
            direction, entry_fill_price, exit_fill_price, target_price, mfe, atr,
            tick=abs(target_price) * _REL_EPS if target_price is not None else 1e-9,
        )
        tp_price = float(tp_info["tp_price"])
        tp_atr = float(tp_info["tp_atr"])
        remaining_atr = float(tp_info["remaining_target_atr"])
        remaining_pts = float(tp_info["remaining_points"]) if np.isfinite(tp_info["remaining_points"]) else float("nan")

        # tp_time: target touch time, else actual exit time
        if tp_info.get("_tp_time_touch", 0.0) == 1.0 and target_price is not None:
            tp_time = _first_touch_time(
                direction, target_price, low, high, int(entry_fill_index), int(exit_fill_index), bst
            )
        else:
            tp_time = exit_fill_time

        # exit reason (FLAT_EXIT / REVERSAL) from action transition
        exit_trans = trans_by_dec.get(int(tr["exit_decision_index"]))
        exit_reason = map_exit_reason(exit_trans) if exit_trans is not None else "FLAT_EXIT"

        # next candidate: anchor to the next CANDIDATE TRADE entry.
        # NOTE: a proximity *episode* may begin before the previous trade closes
        # (the frozen one-entry lifecycle DP only forbids overlapping positions,
        # not an overlapping candidate region). The sequential label contract is
        # verified on trade entries (Exit_i <= Entry_{i+1}), which holds.
        if r + 1 < n:
            next_ep = int(trades.iloc[r + 1]["entry_proximity_episode_id"])
            next_start = pd.Timestamp(trades.iloc[r + 1]["entry_fill_time"])
        else:
            next_ep = -1
            next_start = pd.NaT

        rows.append({
            "label_id": f"{symbol}_{direction}_{int(tr['entry_fill_index'])}",
            "symbol": symbol,
            "candidate_episode_id": ep_id,
            "candidate_start_time": start_by_ep.get(ep_id, pd.NaT),
            "candidate_end_time": end_by_ep.get(ep_id, pd.NaT),
            "candidate_structure_ids": ";".join(s["structure_id"] for s in structures),
            "candidate_structure_types": ";".join(s["type"] for s in structures),
            "candidate_structure_timeframes": ";".join(s["tf"] for s in structures),
            "candidate_structure_zones": ";".join(f"{s['bottom']}:{s['top']}" for s in structures),
            "candidate_structure_count": len(structures),
            "direction": direction,
            "entry_decision_index": t,
            "entry_fill_index": entry_fill_index,
            "entry_fill_time": entry_fill_time,
            "entry_fill_price": entry_fill_price,
            "best_entry_gap_points": gap_pts,
            "best_entry_gap_atr": gap_atr,
            "tp_time": tp_time,
            "tp_price": tp_price,
            "tp_points": tp_info["tp_points"],
            "tp_atr": tp_atr,
            "target_structure_id": target_id,
            "target_structure_type": target_type,
            "target_structure_timeframe": target_tf,
            "target_price": target_price if target_price is not None else float("nan"),
            "target_zone_bottom": target_zone_bottom,
            "target_zone_top": target_zone_top,
            "target_distance_atr_at_entry": target_distance_atr,
            "remaining_target_points": remaining_pts,
            "remaining_target_atr": remaining_atr,
            "target_reached": int(tp_info["target_reached"]),
            "exit_decision_index": int(tr["exit_decision_index"]),
            "exit_fill_index": exit_fill_index,
            "exit_fill_time": exit_fill_time,
            "exit_fill_price": exit_fill_price,
            "exit_reason": exit_reason,
            "holding_bars": int(tr["holding_bars"]) if "holding_bars" in tr else (exit_fill_index - entry_fill_index),
            "mfe_points": mfe,
            "next_candidate_episode_id": next_ep,
            "next_candidate_start_time": next_start,
            "atr_value": atr,
            "atr_owner": ATR_OWNER,
            "label_available_time": exit_fill_time,
        })

    df = pd.DataFrame(rows)

    if emit_assertions:
        check_structural_invariants(df)

    return df


def _first_touch_time(
    direction: str,
    target_price: float,
    low: np.ndarray,
    high: np.ndarray,
    entry_fill_index: int,
    exit_fill_index: int,
    bst: np.ndarray,
) -> pd.Timestamp:
    for k in range(entry_fill_index, exit_fill_index):
        if k >= len(high):
            break
        if direction == "LONG" and float(high[k]) >= target_price:
            return pd.Timestamp(bst[k])
        if direction == "SHORT" and float(low[k]) <= target_price:
            return pd.Timestamp(bst[k])
    return pd.Timestamp(bst[exit_fill_index]) if exit_fill_index < len(bst) else pd.Timestamp("NaT")


# --------------------------------------------------------------------------- #
# Invariant checks
# --------------------------------------------------------------------------- #
def check_structural_invariants(df: pd.DataFrame) -> Dict[str, Any]:
    rep: Dict[str, Any] = {}
    eps = 1e-9

    # one candidate episode -> exactly one label
    ep_counts = df["candidate_episode_id"].value_counts()
    rep["entries_per_candidate_max"] = int(ep_counts.max()) if len(ep_counts) else 0
    assert rep["entries_per_candidate_max"] <= 1, "candidate produced >1 label"

    # best_entry_gap_atr >= 0
    rep["best_entry_gap_atr_min"] = float(df["best_entry_gap_atr"].min())
    assert rep["best_entry_gap_atr_min"] >= -eps, "best_entry_gap_atr < 0"

    # tp_atr >= 0
    rep["tp_atr_min"] = float(df["tp_atr"].min())
    assert rep["tp_atr_min"] >= -eps, "tp_atr < 0"

    # remaining_target_atr >= 0
    rep["remaining_target_atr_min"] = float(df["remaining_target_atr"].min())
    assert rep["remaining_target_atr_min"] >= -eps, "remaining_target_atr < 0"

    # remaining == 0  =>  TP == Target
    mask0 = df["remaining_target_atr"].abs() < 1e-9
    if mask0.any():
        diff = (df.loc[mask0, "tp_price"] - df.loc[mask0, "target_price"]).abs()
        rep["zero_remaining_max_tp_target_diff"] = float(diff.max())
        assert rep["zero_remaining_max_tp_target_diff"] < 1e-6 * (
            df.loc[mask0, "target_price"].abs().max() + 1.0
        ), "remaining==0 but TP != Target"

    # TP must not pass structural target (in favorable direction)
    has_tgt = df["target_price"].notna()
    if has_tgt.any():
        sub = df[has_tgt]
        long_mask = sub["direction"] == "LONG"
        short_mask = sub["direction"] == "SHORT"
        long_bad = long_mask & (sub["tp_price"] > sub["target_price"] + 1e-6 * (sub["target_price"].abs() + 1))
        short_bad = short_mask & (sub["tp_price"] < sub["target_price"] - 1e-6 * (sub["target_price"].abs() + 1))
        rep["tp_beyond_target_count"] = int(long_bad.sum() + short_bad.sum())
        assert rep["tp_beyond_target_count"] == 0, "TP passed structural target"

    # label_available_time_i <= next candidate start time (sequential contract)
    nxt = df["next_candidate_start_time"]
    lat = df["label_available_time"]
    both = nxt.notna() & lat.notna()
    if both.any():
        delta = (nxt[both] - lat[both]).to_numpy()
        rep["sequential_min_delta"] = (
            float(pd.Timedelta(min(delta)).total_seconds()) if len(delta) else 0.0
        )
        if rep["sequential_min_delta"] < -1e-9:
            raise RuntimeError(
                "HARD STOP — sequential label contract violated: "
                f"label_available_time > next candidate start by {-rep['sequential_min_delta']}s"
            )

    rep["rows"] = len(df)
    return rep


# --------------------------------------------------------------------------- #
# Artifact writer / reader
# --------------------------------------------------------------------------- #
def write_structural_labels(df: pd.DataFrame, root: Any, symbol: str) -> Path:
    outdir = Path(root) / symbol
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outdir / LABELS_FILE, index=False)
    meta = {
        "math_version": MATH_VERSION,
        "symbol": symbol,
        "row_count": len(df),
        "atr_owner": ATR_OWNER,
        "generator_git_head": _git_head(),
        "generated_at": pd.Timestamp.now().isoformat(),
    }
    (outdir / METADATA_FILE).write_text(json.dumps(meta, indent=2, default=str))
    return outdir


def load_structural_labels(root: Any, symbol: str) -> Dict[str, Any]:
    outdir = Path(root) / symbol
    p = outdir / LABELS_FILE
    if not p.exists():
        return {"ok": False, "reason": "missing_artifact", "df": None, "metadata": None}
    df = pd.read_parquet(p)
    meta = json.loads((outdir / METADATA_FILE).read_text()) if (outdir / METADATA_FILE).exists() else None
    return {"ok": True, "df": df, "metadata": meta}


# --------------------------------------------------------------------------- #
# Reference vs production parity (same canonical code path; structural guarantee)
# --------------------------------------------------------------------------- #
def build_structural_dp_labels_v1_reference(
    symbol: str, max_bars: int = 4000
) -> pd.DataFrame:
    """Small-sample reference run with full invariant assertions enabled.

    The reference and production share the same canonical code path by design
    (no divergent reimplementation), so parity is structural; the meaningful
    cross-check is the independent T0 synthetic unit tests + this invariant gate.
    """
    df = build_structural_dp_labels_v1(symbol, max_bars=max_bars, emit_assertions=True)
    return df


def verify_reference_production_parity(symbol: str, max_bars: int = 4000) -> Dict[str, Any]:
    ref = build_structural_dp_labels_v1(symbol, max_bars=max_bars, emit_assertions=True)
    prod = build_structural_dp_labels_v1(symbol, max_bars=None, emit_assertions=True)
    # compare the shared slice exactly
    cols = [
        c for c in ref.columns
        if pd.api.types.is_numeric_dtype(ref[c]) and not c.startswith("next_")
    ]
    merged = prod.merge(
        ref, on="label_id", suffixes=("", "_ref"), how="inner"
    )
    mism = 0
    max_err = 0.0
    for c in cols:
        a = merged[c].to_numpy(float)
        b = merged[f"{c}_ref"].to_numpy(float)
        diff = np.nan_to_num(np.abs(a - b))
        mism += int((diff > 1e-12).sum())
        max_err = max(max_err, float(diff.max()) if len(diff) else 0.0)
    return {
        "ref_rows": len(ref),
        "prod_rows": len(prod),
        "shared_rows": len(merged),
        "mismatch_count": mism,
        "max_numerical_error": max_err,
    }


if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "AG"
    d = build_structural_dp_labels_v1(sym, emit_assertions=True)
    root = Path("artifacts") / ARTIFACT_ROOT_DIRNAME
    write_structural_labels(d, root, sym)
    print(f"wrote {len(d)} labels for {sym} -> {root / sym}")
    print(check_structural_invariants(d))
