"""build_structural_dp_labels_m15_v2
====================================

FUT-M15-STRUCTURAL-DP-LABEL-V2 — first canonical structural DP-label builder
with CORRECT structural-event semantics (V1 was rejected as a prototype).

Key corrections vs V1 (rejected):

1. Candidate Event identity is STRUCTURAL, not proximity-episode identity.
   The fundamental unit is ONE distinct SR / liquidity structural location.
   A temporary loss of proximity followed by renewed proximity to the SAME
   structure does NOT create a new event — it remains the same event. We
   recover the canonical structure id per proximity bar from `geom[t-1]`
   (reusing `bar_zone_distance` + `ENTRY_PROX_ATR` + the `SR|...`/`LIQ|...`
   id format) and group contacts by structure id.

2. The candidate geometry is FROZEN when the event begins (the primary
   structure's zone). We do NOT union in additional structures discovered
   during later bars of the same event.

3. Target touch is a REAL DP terminal condition, solved in the structural
   event's own DP. If price reaches the frozen target structure before any
   other legal DP exit, the trade FORCES exit there (TP == Target,
   remaining == 0) and the label completes. This is not post-hoc relabeling:
   the trade genuinely terminates at the target touch.

4. `tp_atr` is only populated when the DP-selected exit is a POSITIVE TP.
   Losing exits are NOT clamped to 0; instead `positive_tp_exists = False`
   and `tp_atr = NA`. The exit price is still recorded honestly.

5. The next-structural-event boundary drives the segment. We persist BOTH
   `next_structural_event_start_time` (next different structure) and
   `next_trade_entry_time` (next actual entry) as distinct concepts, and the
   label-availability contract is verified against the structural-event
   boundary, not the next trade entry.

The DP reuses the FROZEN oracle objective exactly:
    objective = gross open-to-open pnl,  cost_mode = zero_cost
    (the accepted one-entry DP oracle's default),
    execution semantics: decision = close(i), fill = open(i+1).

The exit solver adds the frozen target as a HARD terminal on top of the same
objective — it does not redesign the objective.

Phase 1A single-position lifecycle is reused (the oracle artifact); this module
does not modify it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1 import (
    bar_zone_distance,
    ENTRY_PROX_ATR,
    select_target,
)
from research.liquidity_oracle_atlas.build_dp_proximity_m15_v1 import (
    build_dp_proximity_m15,
)
from research.liquidity_oracle_atlas.build_execution_environment_m15_v1 import (
    run_environment_m15,
)
from research.liquidity_oracle_atlas.build_trade_oracle_dp_m15_one_entry_proximity_v1 import (
    load_oracle_artifact,
    ARTIFACT_ROOT_DIRNAME as ORACLE_ARTIFACT_ROOT,
)

ARTIFACT_ROOT_DIRNAME = "structural_dp_labels_m15_v2"
MATH_VERSION = "structural-dp-label-v2"
ATR_OWNER = "m15_atr@run_environment_m15"


# --------------------------------------------------------------------------- #
# Low-level structural helpers (reuse canonical proximity math verbatim)
# --------------------------------------------------------------------------- #
def _structures_in_proximity(
    prev_geom: Optional[Dict[str, Any]],
    low: float,
    high: float,
    seg: int,
    i: int,
    sr_first_seen: Dict[Any, int],
) -> List[Tuple[str, str, float, float]]:
    """Return the list of canonical structures in proximity at bar `i`.

    Mirrors `proximity_bits_from_prev_geometry` exactly (same radius, same
    `bar_zone_distance` test) but additionally returns the canonical
    `SR|tf|seg|fs|top|bottom|strength` / `LIQ|tf|role|seg|left|level` ids so the
    event can be keyed to a structural location. `prev_geom` is `geom[i-1]`.
    Returns list of (structure_id, type, bottom, top).
    """
    out: List[Tuple[str, str, float, float]] = []
    if not prev_geom:
        return out
    for tf, g in prev_geom.items():
        channels, liq_up, liq_down, atr_tf = g
        if not (np.isfinite(atr_tf) and atr_tf > 0):
            continue
        radius = float(ENTRY_PROX_ATR) * float(atr_tf)
        for top, bottom, strength in channels:
            if bar_zone_distance(low, high, bottom, top) <= radius:
                key = (tf, seg, round(top, 6), round(bottom, 6), round(strength, 4))
                if key not in sr_first_seen:
                    sr_first_seen[key] = i
                fs = sr_first_seen[key]
                sid = f"SR|{tf}|{seg}|{fs}|{top}|{bottom}|{strength}"
                out.append((sid, "SR", float(bottom), float(top)))
        for z in liq_up:
            if z.get("broken"):
                continue
            if bar_zone_distance(low, high, z["bottom"], z["top"]) <= radius:
                sid = f"LIQ|{tf}|BUYSIDE_LIQUIDITY|{seg}|{z['left']}|{z['level']}"
                out.append((sid, "LIQ", float(z["bottom"]), float(z["top"])))
        for z in liq_down:
            if z.get("broken"):
                continue
            if bar_zone_distance(low, high, z["bottom"], z["top"]) <= radius:
                sid = f"LIQ|{tf}|SELLSIDE_LIQUIDITY|{seg}|{z['left']}|{z['level']}"
                out.append((sid, "LIQ", float(z["bottom"]), float(z["top"])))
    return out


def _primary_structure(
    structs: List[Tuple[str, str, float, float]], low: float, high: float
) -> Optional[Tuple[str, str, float, float]]:
    best = None
    best_d = None
    for sid, typ, b, t in structs:
        d = bar_zone_distance(low, high, b, t)
        if best_d is None or d < best_d:
            best_d = d
            best = (sid, typ, b, t)
    return best


def _pick_target(
    geom_prev: Optional[Dict[str, Any]],
    ref_price: float,
    seg: int,
    i: int,
    sr_first_seen: Dict[Any, int],
    direction: str,
) -> Optional[Dict[str, Any]]:
    """Directional nearest eligible structural target ahead of `ref_price`."""
    roles = (
        ["RESISTANCE", "BUYSIDE_LIQUIDITY"]
        if direction == "LONG"
        else ["SUPPORT", "SELLSIDE_LIQUIDITY"]
    )
    cands: List[Dict[str, Any]] = []
    ref = float(ref_price)
    for tf, g in (geom_prev or {}).items():
        channels, liq_up, liq_down, atr_tf = g
        for role in roles:
            c = select_target(
                role, channels, liq_up, liq_down, ref, float(atr_tf),
                tf, int(seg), int(i), sr_first_seen,
            )
            if c is None:
                continue
            c["tf"] = tf
            cands.append(c)
    if not cands:
        return None
    return min(cands, key=lambda x: abs(float(x["near_edge"]) - ref))


def _unit_starts(td: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """First bar index of each (trading_day, segment) intraday unit."""
    n = len(td)
    starts = np.zeros(n, dtype=bool)
    for i in range(n):
        if i == 0 or td[i] != td[i - 1] or seg[i] != seg[i - 1]:
            starts[i] = True
    return np.nonzero(starts)[0]


# --------------------------------------------------------------------------- #
# Structural event construction
# --------------------------------------------------------------------------- #
def _group_structural_events(
    per_bar: List[Optional[List[Tuple[str, str, float, float]]]],
    low: np.ndarray,
    high: np.ndarray,
    n: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Pure structural-event grouping (no env dependency).

    Groups proximity contacts by canonical structure id. The SAME structure
    across a temporary loss of proximity remains ONE event. A DIFFERENT
    structure appearing while the current one is absent starts a new event.
    Returns (events, raw_proximity_runs).
    """
    events: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    raw_runs = 0
    proximity_runs = 0
    same_structure_merges = 0
    for t in range(n):
        structs = per_bar[t]
        if not structs:
            continue
        raw_runs += 1
        run_start = (t == 0 or per_bar[t - 1] is None)
        if run_start:
            proximity_runs += 1
        prim = _primary_structure(structs, float(low[t]), float(high[t]))
        assert prim is not None
        sid = prim[0]
        if cur is None:
            cur = {
                "event_id": len(events),
                "structure_id": sid,
                "structure_type": prim[1],
                "timeframe": sid.split("|")[1],
                "zone_bottom": prim[2],
                "zone_top": prim[3],
                "start_bar": t,
                "end_bar": n,
            }
        else:
            if cur["structure_id"] in {s[0] for s in structs}:
                # current event's structure still in proximity -> continue
                if run_start:
                    same_structure_merges += 1  # a repeated same-structure run merged
                continue
            # a DIFFERENT structure is now the nearest -> close current, open new
            cur["end_bar"] = t
            events.append(cur)
            cur = {
                "event_id": len(events),
                "structure_id": sid,
                "structure_type": prim[1],
                "timeframe": sid.split("|")[1],
                "zone_bottom": prim[2],
                "zone_top": prim[3],
                "start_bar": t,
                "end_bar": n,
            }
    if cur is not None:
        events.append(cur)
    return events, raw_runs, proximity_runs, same_structure_merges


def build_structural_events_v2(
    symbol: str,
    max_bars: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the structural-event sequence keyed by canonical structure id.

    Returns a dict with:
        events: list of event dicts
            {event_id, structure_id, structure_type, timeframe,
             zone_bottom, zone_top, start_bar, end_bar}
        meta: counts (raw_proximity_runs, unique_events, collapsed_runs)
    """
    prox = build_dp_proximity_m15(symbol, max_bars)
    n = len(prox)
    prox_any = np.asarray(prox["dp_proximity_any"].to_numpy(), dtype=bool)
    low = prox["low"].to_numpy(float)
    high = prox["high"].to_numpy(float)
    seg_arr = prox["segment"].to_numpy(np.int64)

    env = run_environment_m15(symbol, max_bars, capture_provenance=False)
    geom = env["geom_by_decision"]

    sr_first_seen: Dict[Any, int] = {}
    per_bar: List[Optional[List[Tuple[str, str, float, float]]]] = [None] * n
    for t in range(n):
        if not prox_any[t]:
            continue
        prev = geom[t - 1] if t >= 1 else None
        per_bar[t] = _structures_in_proximity(
            prev, low[t], high[t], int(seg_arr[t]), t, sr_first_seen
        )

    events, raw_runs, proximity_runs, same_structure_merges = _group_structural_events(
        per_bar, low, high, n
    )
    return {
        "events": events,
        "meta": {
            "raw_proximity_runs": raw_runs,
            "proximity_runs": proximity_runs,
            "unique_structural_events": len(events),
            "collapsed_same_structure_runs": same_structure_merges,
        },
    }


# --------------------------------------------------------------------------- #
# Exit DP solver (reuses the frozen open-to-open objective; target = hard terminal)
# --------------------------------------------------------------------------- #
def solve_event_exit_v2(
    direction: str,
    entry_decision_index: int,
    entry_fill_index: int,
    entry_price: float,
    seg_end: int,
    target_price: Optional[float],
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    atr_value: float,
) -> Dict[str, Any]:
    """Solve the exit for one structural event under the frozen objective.

    - If the frozen target is reached within [entry_fill_index, seg_end],
      it is a HARD terminal: exit at the target (TP == Target, remaining 0).
    - Otherwise pick the best FLAT exit (argmax gross open-to-open pnl).
    - If the best exit is a loss, `positive_tp_exists = False`, `tp_atr = NA`.
    """
    s = 1.0 if direction == "LONG" else -1.0

    def _dir(a: float, b: float) -> float:
        return (b - a) if direction == "LONG" else (a - b)

    # 1) target touch (hard terminal)
    target_reached_bar = None
    if target_price is not None:
        tp = float(target_price)
        for t in range(entry_fill_index, int(seg_end) + 1):
            if direction == "LONG":
                if highs[t] >= tp:
                    target_reached_bar = t
                    break
            else:
                if lows[t] <= tp:
                    target_reached_bar = t
                    break

    if target_reached_bar is not None:
        tp_price = float(target_price)
        tp_bar = target_reached_bar
        tp_points = s * (tp_price - entry_price)
        tp_atr = _dir(entry_price, tp_price) / atr_value
        remaining_atr = 0.0
        positive = True
        exit_reason = "TARGET_TOUCH"
    else:
        # 2) best FLAT exit (argmax gross open-to-open pnl, zero cost),
        #    strictly bounded to [entry_fill_index+1, seg_end].
        seg_lo = entry_fill_index + 1
        seg_hi = int(seg_end)
        if seg_hi < seg_lo:
            # Next structural event arrives immediately: the previous trade is
            # forced flat with no intra-segment room -> boundary close, no TP.
            best_k = entry_fill_index
            best_pnl = 0.0
        else:
            best_k = seg_lo
            best_pnl = s * (opens[seg_lo] - entry_price)
            for k in range(seg_lo + 1, seg_hi + 1):
                pnl = s * (opens[k] - entry_price)
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_k = k
        tp_price = float(opens[best_k])
        tp_bar = best_k
        tp_points = s * (tp_price - entry_price)
        if best_pnl > 1e-9:
            positive = True
            tp_atr = _dir(entry_price, tp_price) / atr_value
            exit_reason = "DP_EARLY_EXIT"
        else:
            positive = False
            tp_atr = float("nan")
            exit_reason = "DP_LOSS_EXIT"
        if target_price is None:
            remaining_atr = float("nan")
        else:
            remaining_atr = _dir(tp_price, float(target_price)) / atr_value

    return {
        "tp_price": tp_price,
        "tp_fill_index": int(tp_bar),
        "tp_points": float(tp_points),
        "tp_atr": float(tp_atr),
        "positive_tp_exists": bool(positive),
        "remaining_target_atr": float(remaining_atr),
        "exit_reason": exit_reason,
        "exit_fill_price": tp_price,
        "exit_fill_index": int(tp_bar),
        "exit_decision_index": int(tp_bar) - 1,
    }


# --------------------------------------------------------------------------- #
# Production orchestrator
# --------------------------------------------------------------------------- #
def build_structural_dp_labels_v2(
    symbol: str,
    max_bars: Optional[int] = None,
    emit_assertions: bool = True,
) -> pd.DataFrame:
    ev = build_structural_events_v2(symbol, max_bars)
    events = ev["events"]
    meta_events = ev["meta"]

    prox = build_dp_proximity_m15(symbol, max_bars)
    n = len(prox)
    opens = prox["open"].to_numpy(float)
    highs = prox["high"].to_numpy(float)
    lows = prox["low"].to_numpy(float)
    bar_start = pd.to_datetime(prox["bar_start_time"]).to_numpy()
    seg_arr = prox["segment"].to_numpy(np.int64)
    td_arr = pd.to_datetime(prox["trading_day"]).to_numpy()

    env = run_environment_m15(symbol, max_bars, capture_provenance=False)
    atr_series = np.asarray(env["features"]["m15_atr"].to_numpy(float))
    geom = env["geom_by_decision"]

    oracle = load_oracle_artifact(str(Path("artifacts") / ORACLE_ARTIFACT_ROOT), symbol)
    if not oracle["ok"]:
        raise RuntimeError(
            "Structural DP Label V2 requires the frozen one-entry DP oracle "
            f"artifact for {symbol}. Generate it first."
        )
    trades = oracle["trades"]

    unit_starts = _unit_starts(td_arr, seg_arr)

    def _next_unit_start_after(bar: int) -> Optional[int]:
        later = unit_starts[unit_starts > bar]
        return int(later[0]) if len(later) else None

    sr_first_seen: Dict[Any, int] = {}
    rows: List[Dict[str, Any]] = []
    invalid_count = 0
    collapsed_entries = 0

    for e in events:
        sid = e["structure_id"]
        s_bar = e["start_bar"]
        e_bar = e["end_bar"]

        # next structural event (same for valid/invalid rows)
        next_ev = None
        for othere in events:
            if othere["event_id"] > e["event_id"]:
                next_ev = othere
                break
        nxt_struct_start = (
            pd.Timestamp(bar_start[next_ev["start_bar"]]) if next_ev else pd.NaT
        )

        # oracle entry: first trade whose fill is within [s_bar, e_bar)
        mask = (
            (trades["entry_fill_index"].to_numpy(int) >= s_bar)
            & (trades["entry_fill_index"].to_numpy(int) < e_bar)
        )
        sub = trades[mask]

        zb, zt = e["zone_bottom"], e["zone_top"]

        if sub.empty:
            # No legal single entry under the frozen DP rules -> INVALID label,
            # but kept in the sequence (not silently dropped) per spec.
            invalid_count += 1
            ref_price = float(prox["close"].iloc[s_bar])
            direction = "LONG"  # placeholder; no trade occurred
            target_price = None
            target_distance_atr = float("nan")
            tgt = None
            rows.append({
                "label_id": f"{symbol}_{sid}_{s_bar}_NOENTRY",
                "symbol": symbol,
                "event_id": int(e["event_id"]),
                "structure_id": sid,
                "structure_type": e["structure_type"],
                "timeframe": e["timeframe"],
                "zone_bottom": zb,
                "zone_top": zt,
                "candidate_start_bar": int(s_bar),
                "candidate_end_bar": int(e_bar),
                "candidate_start_time": pd.Timestamp(bar_start[s_bar]),
                "candidate_end_time": pd.Timestamp(bar_start[e_bar - 1]) if e_bar <= n else pd.Timestamp(bar_start[-1]),
                "direction": "NONE",
                "entry_present": False,
                "entry_decision_index": -1,
                "entry_fill_index": -1,
                "entry_fill_time": pd.NaT,
                "entry_fill_price": float("nan"),
                "best_entry_gap_points": float("nan"),
                "best_entry_gap_atr": float("nan"),
                "tp_fill_index": -1,
                "tp_time": pd.NaT,
                "tp_price": float("nan"),
                "tp_points": float("nan"),
                "tp_atr": float("nan"),
                "positive_tp_exists": False,
                "target_structure_id": None,
                "target_structure_type": None,
                "target_timeframe": None,
                "target_price": target_price,
                "target_zone_bottom": None,
                "target_zone_top": None,
                "target_distance_atr": target_distance_atr,
                "remaining_target_atr": float("nan"),
                "exit_decision_index": -1,
                "exit_fill_index": -1,
                "exit_fill_time": pd.NaT,
                "exit_fill_price": float("nan"),
                "exit_reason": "NO_ENTRY",
                "next_structural_event_id": int(next_ev["event_id"]) if next_ev else -1,
                "next_structural_event_start_time": nxt_struct_start,
                "next_trade_entry_time": pd.NaT,
                "atr_value": float("nan"),
                "atr_owner": ATR_OWNER,
                "label_available_time": pd.NaT,
            })
            continue

        sub = sub.sort_values("entry_fill_index")
        ent = sub.iloc[0]
        collapsed_entries += len(sub) - 1  # extra oracle entries on same structure

        direction = ent["direction"]
        entry_decision_index = int(ent["entry_decision_index"])
        entry_fill_index = int(ent["entry_fill_index"])
        entry_fill_price = float(ent["entry_fill_price"])
        entry_fill_time = pd.Timestamp(ent["entry_fill_time"])

        atr_value = float(atr_series[entry_decision_index])
        gap_points = bar_zone_distance(entry_fill_price, entry_fill_price, zb, zt)
        best_entry_gap_atr = gap_points / atr_value

        # target (frozen at event start, using structures known before entry)
        geom_prev = geom[entry_decision_index - 1] if entry_decision_index >= 1 else None
        tgt = _pick_target(
            geom_prev, entry_fill_price, int(seg_arr[entry_decision_index]),
            entry_decision_index, sr_first_seen, direction,
        )
        if tgt is not None:
            target_price = float(tgt["near_edge"])
            target_distance_atr = (
                ((target_price - entry_fill_price) if direction == "LONG"
                 else (entry_fill_price - target_price)) / atr_value
            )
        else:
            target_price = None
            target_distance_atr = float("nan")

        # segment end: next different structural event OR next unit boundary
        seg_end = e_bar - 1
        nus = _next_unit_start_after(entry_fill_index)
        if nus is not None:
            seg_end = min(seg_end, nus - 1)
        seg_end = min(seg_end, n - 1)

        exit_d = solve_event_exit_v2(
            direction, entry_decision_index, entry_fill_index, entry_fill_price,
            seg_end, target_price, opens, highs, lows, atr_value,
        )

        later_trades = trades[trades["entry_fill_index"].to_numpy(int) > entry_fill_index]
        next_trade_entry_time = (
            pd.Timestamp(later_trades.iloc[0]["entry_fill_time"])
            if not later_trades.empty else pd.NaT
        )
        label_avail = pd.Timestamp(prox["bar_start_time"].iloc[exit_d["exit_fill_index"]]) \
            if exit_d["exit_fill_index"] < n else pd.Timestamp(bar_start[-1])

        rows.append({
            "label_id": f"{symbol}_{sid}_{entry_fill_index}",
            "symbol": symbol,
            "event_id": int(e["event_id"]),
            "structure_id": sid,
            "structure_type": e["structure_type"],
            "timeframe": e["timeframe"],
            "zone_bottom": zb,
            "zone_top": zt,
            "candidate_start_bar": int(s_bar),
            "candidate_end_bar": int(e_bar),
            "candidate_start_time": pd.Timestamp(bar_start[s_bar]),
            "candidate_end_time": pd.Timestamp(bar_start[e_bar - 1]) if e_bar <= n else pd.Timestamp(bar_start[-1]),
            "direction": direction,
            "entry_present": True,
            "entry_decision_index": entry_decision_index,
            "entry_fill_index": entry_fill_index,
            "entry_fill_time": entry_fill_time,
            "entry_fill_price": entry_fill_price,
            "best_entry_gap_points": float(gap_points),
            "best_entry_gap_atr": float(best_entry_gap_atr),
            "tp_fill_index": exit_d["tp_fill_index"],
            "tp_time": pd.Timestamp(bar_start[exit_d["tp_fill_index"]]),
            "tp_price": exit_d["tp_price"],
            "tp_points": exit_d["tp_points"],
            "tp_atr": exit_d["tp_atr"],
            "positive_tp_exists": exit_d["positive_tp_exists"],
            "target_structure_id": tgt["structure_id"] if tgt else None,
            "target_structure_type": tgt["role"] if tgt else None,
            "target_timeframe": tgt.get("tf") if tgt else None,
            "target_price": target_price,
            "target_zone_bottom": tgt["bottom"] if tgt else None,
            "target_zone_top": tgt["top"] if tgt else None,
            "target_distance_atr": target_distance_atr,
            "remaining_target_atr": exit_d["remaining_target_atr"],
            "exit_decision_index": exit_d["exit_decision_index"],
            "exit_fill_index": exit_d["exit_fill_index"],
            "exit_fill_time": pd.Timestamp(bar_start[exit_d["exit_fill_index"]]),
            "exit_fill_price": exit_d["exit_fill_price"],
            "exit_reason": exit_d["exit_reason"],
            "next_structural_event_id": int(next_ev["event_id"]) if next_ev else -1,
            "next_structural_event_start_time": nxt_struct_start,
            "next_trade_entry_time": next_trade_entry_time,
            "atr_value": atr_value,
            "atr_owner": ATR_OWNER,
            "label_available_time": label_avail,
        })

    df = pd.DataFrame(rows)
    if emit_assertions:
        rep = check_structural_invariants_v2(df)
        if rep.get("hard_fail"):
            raise RuntimeError(f"STRUCTURAL V2 HARD FAIL: {rep['hard_fail']}")

    df.attrs["meta"] = {
        "math_version": MATH_VERSION,
        "raw_proximity_runs": meta_events["raw_proximity_runs"],
        "unique_structural_events": meta_events["unique_structural_events"],
        "collapsed_same_structure_runs": meta_events["collapsed_same_structure_runs"],
        "total_structural_event_rows": len(df),
        "valid_labels_entry_present": int(df["entry_present"].sum()) if "entry_present" in df else len(df),
        "invalid_no_entry_count": invalid_count,
        "collapsed_same_structure_entries": collapsed_entries,
    }
    return df


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #
def check_structural_invariants_v2(df: pd.DataFrame) -> Dict[str, Any]:
    rep: Dict[str, Any] = {"rows": len(df), "hard_fail": None}

    def _fail(msg):
        rep["hard_fail"] = msg

    if df.empty:
        return rep

    # one entry per structural event (by construction); verify label count == events
    rep["entries_per_structural_event_max"] = int(
        df.groupby("event_id").size().max()
    )
    if rep["entries_per_structural_event_max"] > 1:
        _fail("entries_per_structural_event > 1")

    # best_entry_gap_atr >= 0
    _gap = df["best_entry_gap_atr"].dropna()
    rep["best_entry_gap_atr_min"] = float(_gap.min()) if len(_gap) else float("nan")
    if rep["best_entry_gap_atr_min"] < -1e-9:
        _fail("best_entry_gap_atr < 0")

    # tp_atr >= 0 OR NaN (positive_tp_exists False)
    tp = df["tp_atr"].to_numpy(float)
    rep["tp_atr_min_finite"] = float(np.nanmin(tp)) if np.isfinite(tp).any() else float("nan")
    bad_tp = df[df["positive_tp_exists"] & (df["tp_atr"].to_numpy(float) < -1e-9)]
    if len(bad_tp):
        _fail("positive TP with tp_atr < 0")

    # remaining_target_atr >= 0 when target exists
    rem = df["remaining_target_atr"].to_numpy(float)
    rep["remaining_target_atr_min_finite"] = (
        float(np.nanmin(rem)) if np.isfinite(rem).any() else float("nan")
    )
    bad_rem = df[df["target_price"].notna() & (df["remaining_target_atr"].to_numpy(float) < -1e-9)]
    if len(bad_rem):
        _fail("remaining_target_atr < 0 (negative geometry)")

    # remaining == 0 => TP == Target
    reached = df[df["remaining_target_atr"].to_numpy(float) == 0.0]
    diff = 0.0
    for _, r in reached.iterrows():
        if r["target_price"] is None:
            continue
        diff = max(diff, abs(float(r["tp_price"]) - float(r["target_price"])))
    rep["zero_remaining_max_tp_target_diff"] = float(diff)
    if diff > 1e-6:
        _fail("remaining==0 but TP != Target")

    # TP must not pass the structural target
    beyond = 0
    for _, r in df[df["target_price"].notna() & df["positive_tp_exists"]].iterrows():
        dir_dist_tp = (r["tp_price"] - r["entry_fill_price"]) if r["direction"] == "LONG" \
            else (r["entry_fill_price"] - r["tp_price"])
        dir_dist_tgt = (r["target_price"] - r["entry_fill_price"]) if r["direction"] == "LONG" \
            else (r["entry_fill_price"] - r["target_price"])
        if dir_dist_tp > dir_dist_tgt + 1e-6:
            beyond += 1
    rep["tp_beyond_target_count"] = int(beyond)
    if beyond:
        _fail("TP beyond structural target")

    # positive_tp_exists consistency
    incon = df[
        (df["positive_tp_exists"]) & (~df["target_price"].isna()) &
        (df["tp_atr"].to_numpy(float) > df["target_distance_atr"].to_numpy(float) + 1e-6)
    ]
    rep["tp_exceeds_target_distance_count"] = int(len(incon))
    if len(incon):
        _fail("positive TP atr exceeds target distance atr")

    # sequential contract: label_available <= next structural event start
    both = df["label_available_time"].notna() & df["next_structural_event_start_time"].notna()
    if both.any():
        lat = df.loc[both, "label_available_time"]
        nxt = df.loc[both, "next_structural_event_start_time"]
        delta = (nxt.to_numpy() - lat.to_numpy()).astype("timedelta64[s]").astype(float)
        rep["sequential_min_delta_s"] = float(delta.min())
        if delta.min() < -1e-6:
            _fail("label_available_time > next_structural_event_start_time")
    else:
        rep["sequential_min_delta_s"] = 0.0

    # no overlapping trades (valid trades only): exit_fill_index <= next entry
    sd = df[df["entry_present"]].sort_values("entry_fill_index").reset_index(drop=True)
    overlaps = 0
    for i in range(1, len(sd)):
        if sd.loc[i, "entry_fill_index"] < sd.loc[i - 1, "exit_fill_index"]:
            overlaps += 1
    rep["overlapping_trades"] = int(overlaps)
    if overlaps:
        _fail("overlapping trades")

    return rep


# --------------------------------------------------------------------------- #
# Artifact IO
# --------------------------------------------------------------------------- #
def load_structural_labels_v2(root: Any, symbol: str) -> Dict[str, Any]:
    root = Path(root) / ARTIFACT_ROOT_DIRNAME
    p = root / symbol / "structural_dp_labels.parquet"
    if not p.exists():
        return {"ok": False, "reason": f"missing {p}"}
    df = pd.read_parquet(p)
    meta = {}
    mp = root / symbol / "metadata.json"
    if mp.exists():
        meta = json.loads(mp.read_text())
    return {"ok": True, "df": df, "metadata": meta}


def _save(df: pd.DataFrame, symbol: str, root: Any, generator_git_head: str) -> None:
    from research.liquidity_oracle_atlas.git_head import git_head  # noqa: F401 (validates availability)
    out_root = Path(root) / ARTIFACT_ROOT_DIRNAME / symbol
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_root / "structural_dp_labels.parquet", index=False)
    meta = {
        "math_version": MATH_VERSION,
        "symbol": symbol,
        "row_count": int(len(df)),
        "atr_owner": ATR_OWNER,
        "generator_git_head": generator_git_head,
        "event_meta": df.attrs.get("meta", {}),
    }
    (out_root / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))


def main(symbol: str = "AG") -> None:
    from research.liquidity_oracle_atlas.git_head import git_head
    df = build_structural_dp_labels_v2(symbol, emit_assertions=True)
    _save(df, symbol, Path("artifacts"), git_head())
    rep = check_structural_invariants_v2(df)
    print(f"wrote {len(df)} labels for {symbol} -> "
          f"artifacts/{ARTIFACT_ROOT_DIRNAME}/{symbol}")
    em = df.attrs["meta"]
    print("event meta:", em)
    print("invariants:", {k: v for k, v in rep.items() if k != "hard_fail"})


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "AG")
