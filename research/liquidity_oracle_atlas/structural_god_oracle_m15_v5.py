"""structural_god_oracle_m15_v5
===============================

FUT-M15-STRUCTURAL-GOD-ORACLE-V5

God-mode oracle -- V5 REWRITE (first-passage direction + global non-overlap DP).

What changed vs V4 (frozen here)
--------------------------------
V4 decided direction by "which side eventually reaches its frozen target with
the larger profit" and a single greedy sequential cursor. That let a trade be
labelled SHORT merely because the lower target was *eventually* touched, even
when price first completed a structural move to the upside. V5 fixes the atomic
direction definition first:

    From the real Entry (next-bar open after the decision close), the market
    either touches the frozen UPPER structure first or the frozen LOWER
    structure first. That race -- NOT eventual profit -- is the direction.

Atomic trade semantics (the ONLY correct layer before scheduling):

  * Every structure co-present at a decision bar is a Candidate A (no
    primary-only gating). ``per_bar[d]`` is the candidate set.
  * Each decision bar ``d`` freezes its OWN two competing targets from
    ``geom[d-1]`` (already-known geometry), referenced to ``close[d]``:
        upper = nearest still-untouched structure ABOVE close[d]
        lower = nearest still-untouched structure BELOW close[d]
    A target already touched during decision bar d is NOT a future target.
  * Entry = open[d+1] (decision close -> fill next open). The d -> d+1 fill
    must stay in the same trading_day / segment / intraday execution unit.
  * Gap-through: if the next open already skips a frozen target, that branch
    is invalid for this decision (never silently promoted to the next target).
  * Direction = competing first-passage from the Entry:
        tau_upper = first bar >= fill with high >= upper.near_edge
        tau_lower = first bar >= fill with low  <= lower.near_edge
        LONG  if tau_upper <  tau_lower
        SHORT if tau_lower <  tau_upper
        AMBIGUOUS (no label) if tau_upper == tau_lower (same 15m bar touches
        both -- intrabar order is unknowable)
  * Exit = the winning target's first touch; TARGET_TOUCH only; no holding cap.

Scheduling (the SECOND, separate layer):
  Enumerate every atomic TradeOption across all decision bars, then select the
  maximum-total-utility set of NON-OVERLAPPING options via weighted-interval
  DP. The scheduler NEVER decides direction; it only resolves "one position at
  a time".

This module is PRODUCTION. It reuses V4's O(log N) segment tree
(``_build_target_tree`` / ``_first_target_touch``) so the per-option target
query is O(log N); total work is O(N + Q log N + K) where Q = number of atomic
options and K = number of selected trades. No full-window target scan occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Reuse the proven helpers / data adapters from V4 (identical math, no change):
#   * target identity + self-target exclusion
#   * execution-unit helpers + O(log N) target-touch tree
#   * geom / proximity / event builders + market view
#   * meta aggregation (record schema stays compatible with the viewer/builder)
from research.liquidity_oracle_atlas.structural_god_oracle_m15_v4 import (
    ATR_OWNER,
    MATH_VERSION,
    ORACLE_LONG,
    ORACLE_SHORT,
    PNL_EPS,
    R_TARGET,
    _build_meta,
    _build_target_tree,
    _first_target_touch,
    _unit_of,
    _unit_starts_from_arrays,
    build_per_bar_proximity,
    hotpath_counters,
    reset_hotpath_counters,
    target_is_self,
)
from research.liquidity_oracle_atlas.build_structural_dp_labels_m15_v2 import (
    build_structural_events_v2,
)
from research.liquidity_oracle_atlas.build_dp_proximity_m15_v1 import (
    build_dp_proximity_m15,
)
from research.liquidity_oracle_atlas.build_execution_environment_m15_v1 import (
    run_environment_m15,
)
from research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1 import (
    bar_zone_distance,
)
from research.liquidity_oracle_atlas.structural_event_dp_kernel_v3 import (
    market_view_from_arrays,
)

EPS = 1e-9


@dataclass(frozen=True)
class FrozenTarget:
    structure_id: str
    structure_type: str
    timeframe: str
    role: str

    near_edge: float
    far_edge: float


@dataclass(frozen=True)
class TradeOption:
    decision: int
    fill: int
    exit: int

    direction: str

    candidate_structure_id: str
    candidate_structure_type: str
    zone_bottom: float
    zone_top: float

    target_structure_id: str
    target_structure_type: str
    target_price: float

    entry_price: float
    exit_price: float

    utility_points: float
    utility_atr: float
    entry_gap_atr: float

    event_id: int = -1


# ============================================================
# 1. Execution
# ============================================================

def same_execution_unit(
    d: int,
    f: int,
    trading_day: np.ndarray,
    segment: np.ndarray,
    mv,
) -> bool:
    if d < 0 or f >= mv.n:
        return False

    return (
        int(trading_day[d]) == int(trading_day[f])
        and int(segment[d]) == int(segment[f])
        and _unit_of(mv, d) == _unit_of(mv, f)
    )


# ============================================================
# 2. Stable SR identity
# ============================================================

def make_sr_sid(
    *,
    tf: str,
    seg: int,
    i: int,
    top: float,
    bottom: float,
    strength: float,
    sr_first_seen: Dict[Any, int],
) -> str:

    key = (
        str(tf),
        int(seg),
        round(float(top), 6),
        round(float(bottom), 6),
        round(float(strength), 4),
    )

    if key not in sr_first_seen:
        sr_first_seen[key] = int(i)

    fs = int(sr_first_seen[key])

    return (
        f"SR|{tf}|{int(seg)}|{fs}|"
        f"{float(top)}|{float(bottom)}|{float(strength)}"
    )


# ============================================================
# 3. Enumerate ALL structural targets known at decision d
# ============================================================

def enumerate_frozen_targets(
    geom_prev: Optional[Dict[str, Any]],
    *,
    seg: int,
    d: int,
    sr_first_seen: Dict[Any, int],
    candidate_sid: str,
) -> List[FrozenTarget]:

    if not geom_prev:
        return []

    out: List[FrozenTarget] = []

    for tf, g in geom_prev.items():

        channels, liq_up, liq_down, _atr_tf = g

        # ---------- SR ----------
        for top, bottom, strength in channels:

            top = float(top)
            bottom = float(bottom)

            sid = make_sr_sid(
                tf=tf,
                seg=seg,
                i=d,
                top=top,
                bottom=bottom,
                strength=float(strength),
                sr_first_seen=sr_first_seen,
            )

            if target_is_self(sid, candidate_sid):
                continue

            # The same SR zone may act as resistance if above price,
            # or support if below price. Direction is resolved later.
            out.append(
                FrozenTarget(
                    structure_id=sid,
                    structure_type="SR",
                    timeframe=str(tf),
                    role="SR",
                    near_edge=np.nan,  # resolved by side
                    far_edge=np.nan,
                )
            )

        # ---------- Buyside liquidity ----------
        for z in liq_up:

            if bool(z.get("broken")):
                continue

            sid = (
                f"LIQ|{tf}|BUYSIDE_LIQUIDITY|{int(seg)}|"
                f"{int(z['left'])}|{float(z['level'])}"
            )

            if target_is_self(sid, candidate_sid):
                continue

            out.append(
                FrozenTarget(
                    structure_id=sid,
                    structure_type="LIQ",
                    timeframe=str(tf),
                    role="BUYSIDE_LIQUIDITY",
                    near_edge=float(z["bottom"]),
                    far_edge=float(z["top"]),
                )
            )

        # ---------- Sellside liquidity ----------
        for z in liq_down:

            if bool(z.get("broken")):
                continue

            sid = (
                f"LIQ|{tf}|SELLSIDE_LIQUIDITY|{int(seg)}|"
                f"{int(z['left'])}|{float(z['level'])}"
            )

            if target_is_self(sid, candidate_sid):
                continue

            out.append(
                FrozenTarget(
                    structure_id=sid,
                    structure_type="LIQ",
                    timeframe=str(tf),
                    role="SELLSIDE_LIQUIDITY",
                    near_edge=float(z["top"]),
                    far_edge=float(z["bottom"]),
                )
            )

    return out


def nearest_upper_and_lower_target(
    geom_prev: Optional[Dict[str, Any]],
    *,
    close_d: float,
    high_d: float,
    low_d: float,
    seg: int,
    d: int,
    sr_first_seen: Dict[Any, int],
    candidate_sid: str,
) -> Tuple[Optional[FrozenTarget], Optional[FrozenTarget]]:
    """
    Freeze TWO competing targets at decision close:

        upper = nearest still-untouched structure above close[d]
        lower = nearest still-untouched structure below close[d]

    A target touched during decision bar d is NOT future.
    """

    if not geom_prev:
        return None, None

    upper: List[FrozenTarget] = []
    lower: List[FrozenTarget] = []

    for tf, g in geom_prev.items():

        channels, liq_up, liq_down, _atr_tf = g

        # ---------- SR ----------
        for top, bottom, strength in channels:

            top = float(top)
            bottom = float(bottom)

            sid = make_sr_sid(
                tf=tf,
                seg=seg,
                i=d,
                top=top,
                bottom=bottom,
                strength=float(strength),
                sr_first_seen=sr_first_seen,
            )

            if target_is_self(sid, candidate_sid):
                continue

            # Entire zone ahead above close
            if bottom > float(close_d) + EPS:
                # already touched before decision close => not future
                if float(high_d) < bottom - EPS:
                    upper.append(
                        FrozenTarget(
                            structure_id=sid,
                            structure_type="SR",
                            timeframe=str(tf),
                            role="RESISTANCE",
                            near_edge=bottom,
                            far_edge=top,
                        )
                    )

            # Entire zone ahead below close
            if top < float(close_d) - EPS:
                if float(low_d) > top + EPS:
                    lower.append(
                        FrozenTarget(
                            structure_id=sid,
                            structure_type="SR",
                            timeframe=str(tf),
                            role="SUPPORT",
                            near_edge=top,
                            far_edge=bottom,
                        )
                    )

        # ---------- Buy liquidity = upper ----------
        for z in liq_up:

            if bool(z.get("broken")):
                continue

            bottom = float(z["bottom"])
            top = float(z["top"])

            if bottom <= float(close_d) + EPS:
                continue

            if float(high_d) >= bottom - EPS:
                continue

            sid = (
                f"LIQ|{tf}|BUYSIDE_LIQUIDITY|{int(seg)}|"
                f"{int(z['left'])}|{float(z['level'])}"
            )

            if target_is_self(sid, candidate_sid):
                continue

            upper.append(
                FrozenTarget(
                    structure_id=sid,
                    structure_type="LIQ",
                    timeframe=str(tf),
                    role="BUYSIDE_LIQUIDITY",
                    near_edge=bottom,
                    far_edge=top,
                )
            )

        # ---------- Sell liquidity = lower ----------
        for z in liq_down:

            if bool(z.get("broken")):
                continue

            bottom = float(z["bottom"])
            top = float(z["top"])

            if top >= float(close_d) - EPS:
                continue

            if float(low_d) <= top + EPS:
                continue

            sid = (
                f"LIQ|{tf}|SELLSIDE_LIQUIDITY|{int(seg)}|"
                f"{int(z['left'])}|{float(z['level'])}"
            )

            if target_is_self(sid, candidate_sid):
                continue

            lower.append(
                FrozenTarget(
                    structure_id=sid,
                    structure_type="LIQ",
                    timeframe=str(tf),
                    role="SELLSIDE_LIQUIDITY",
                    near_edge=top,
                    far_edge=bottom,
                )
            )

    upper.sort(
        key=lambda x: float(x.near_edge) - float(close_d)
    )

    lower.sort(
        key=lambda x: float(close_d) - float(x.near_edge)
    )

    return (
        upper[0] if upper else None,
        lower[0] if lower else None,
    )


# ============================================================
# 4. One decision -> ONE direction via competing first passage
# ============================================================

def evaluate_candidate_at_decision(
    *,
    d: int,
    candidate_sid: str,
    candidate_type: str,
    zone_bottom: float,
    zone_top: float,

    geom_prev,
    sr_first_seen,

    opens,
    highs,
    lows,
    closes,
    atr_series,
    trading_day,
    segment,

    mv,
    target_tree,

    event_id: int = -1,
) -> Optional[TradeOption]:

    n = int(mv.n)

    if d < 0 or d + 1 >= n:
        return None

    f = int(d) + 1

    # Entry execution must be legal.
    if not same_execution_unit(
        d,
        f,
        trading_day,
        segment,
        mv,
    ):
        return None

    seg_d = int(segment[d])

    upper, lower = nearest_upper_and_lower_target(
        geom_prev,
        close_d=float(closes[d]),
        high_d=float(highs[d]),
        low_d=float(lows[d]),
        seg=seg_d,
        d=int(d),
        sr_first_seen=sr_first_seen,
        candidate_sid=candidate_sid,
    )

    entry = float(opens[f])

    # --------------------------------------------------------
    # Gap-through handling
    # --------------------------------------------------------

    upper_valid = upper is not None
    lower_valid = lower is not None

    if upper_valid and entry >= float(upper.near_edge) - EPS:
        upper_valid = False

    if lower_valid and entry <= float(lower.near_edge) + EPS:
        lower_valid = False

    if not upper_valid and not lower_valid:
        return None

    # --------------------------------------------------------
    # Competing first passage AFTER fill
    # --------------------------------------------------------

    tau_up = -1
    tau_dn = -1

    if upper_valid:
        tau_up = _first_target_touch(
            target_tree,
            f,
            "LONG",
            float(upper.near_edge),
        )

    if lower_valid:
        tau_dn = _first_target_touch(
            target_tree,
            f,
            "SHORT",
            float(lower.near_edge),
        )

    if tau_up < 0 and tau_dn < 0:
        return None

    # Same 15m bar touches both targets:
    # intrabar order is unknowable => no canonical direction.
    if tau_up >= 0 and tau_dn >= 0 and tau_up == tau_dn:
        return None

    if tau_up >= 0 and (tau_dn < 0 or tau_up < tau_dn):

        direction = "LONG"
        target = upper
        exit_idx = int(tau_up)
        utility = float(target.near_edge) - entry

    else:

        direction = "SHORT"
        target = lower
        exit_idx = int(tau_dn)
        utility = entry - float(target.near_edge)

    if utility <= EPS:
        return None

    atr = float(atr_series[d])

    if np.isfinite(atr) and atr > 0:
        utility_atr = utility / atr

        gap_points = bar_zone_distance(
            entry,
            entry,
            float(zone_bottom),
            float(zone_top),
        )
        gap_atr = gap_points / atr

    else:
        utility_atr = float("nan")
        gap_atr = float("nan")

    return TradeOption(
        decision=int(d),
        fill=int(f),
        exit=int(exit_idx),

        direction=direction,

        candidate_structure_id=str(candidate_sid),
        candidate_structure_type=str(candidate_type),
        zone_bottom=float(zone_bottom),
        zone_top=float(zone_top),

        target_structure_id=str(target.structure_id),
        target_structure_type=str(target.structure_type),
        target_price=float(target.near_edge),

        entry_price=float(entry),
        exit_price=float(target.near_edge),

        utility_points=float(utility),
        utility_atr=float(utility_atr),
        entry_gap_atr=float(gap_atr),

        event_id=int(event_id),
    )


# ============================================================
# 5. Enumerate ALL atomic TradeOptions
# ============================================================

def enumerate_trade_options(
    *,
    per_bar,
    geom,
    opens,
    highs,
    lows,
    closes,
    atr_series,
    trading_day,
    segment,
    mv,
    sr_first_seen,
    event_id_by_sid,
) -> List[List[TradeOption]]:

    n = int(mv.n)

    target_tree = _build_target_tree(mv)

    options: List[List[TradeOption]] = [
        [] for _ in range(n)
    ]

    for d in range(n - 1):

        pb = per_bar[d]

        if not pb:
            continue

        geom_prev = geom[d - 1] if d >= 1 else None

        if not geom_prev:
            continue

        seen = set()

        # ALL structures in proximity.
        # No primary-only filter.
        for sid, stype, zb, zt in pb:

            sid = str(sid)

            if sid in seen:
                continue

            seen.add(sid)

            tr = evaluate_candidate_at_decision(
                d=d,

                candidate_sid=sid,
                candidate_type=str(stype),
                zone_bottom=float(zb),
                zone_top=float(zt),

                geom_prev=geom_prev,
                sr_first_seen=sr_first_seen,

                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                atr_series=atr_series,
                trading_day=trading_day,
                segment=segment,

                mv=mv,
                target_tree=target_tree,

                event_id=int(
                    event_id_by_sid.get(sid, -1)
                ),
            )

            if tr is not None:
                options[d].append(tr)

    return options


# ============================================================
# 6. Global one-position God-mode scheduler
# ============================================================

def better_state(
    value_a: float,
    count_a: int,
    hold_a: int,
    value_b: float,
    count_b: int,
    hold_b: int,
) -> bool:

    # Primary objective = total gross points.
    if value_a > value_b + EPS:
        return True

    if value_b > value_a + EPS:
        return False

    # Tie: more completed labels.
    if count_a != count_b:
        return count_a > count_b

    # Tie again: shorter total occupation.
    return hold_a < hold_b


def solve_nonoverlap_god_dp(
    options: List[List[TradeOption]],
    n: int,
) -> List[TradeOption]:

    V = np.zeros(n + 1, dtype=float)
    C = np.zeros(n + 1, dtype=np.int64)
    H = np.zeros(n + 1, dtype=np.int64)

    choice: List[Optional[TradeOption]] = [
        None for _ in range(n)
    ]

    for t in range(n - 1, -1, -1):

        # Skip t.
        best_v = float(V[t + 1])
        best_c = int(C[t + 1])
        best_h = int(H[t + 1])
        best_trade = None

        for tr in options[t]:

            nxt = min(int(tr.exit) + 1, n)

            cand_v = (
                float(tr.utility_points)
                + float(V[nxt])
            )

            cand_c = 1 + int(C[nxt])

            cand_h = (
                int(tr.exit)
                - int(tr.fill)
                + 1
                + int(H[nxt])
            )

            if better_state(
                cand_v,
                cand_c,
                cand_h,
                best_v,
                best_c,
                best_h,
            ):
                best_v = cand_v
                best_c = cand_c
                best_h = cand_h
                best_trade = tr

        V[t] = best_v
        C[t] = best_c
        H[t] = best_h
        choice[t] = best_trade

    selected: List[TradeOption] = []

    t = 0

    while t < n:

        tr = choice[t]

        if tr is None:
            t += 1
            continue

        selected.append(tr)

        t = int(tr.exit) + 1

    for i in range(len(selected) - 1):
        assert (
            int(selected[i].exit)
            < int(selected[i + 1].decision)
        )

    return selected


# ============================================================
# 7. Oracle entry point: build data -> enumerate -> DP -> records
# ============================================================

def _to_record(tr: TradeOption, mv, atr_series) -> Dict[str, Any]:
    d = int(tr.decision)
    atr = float(atr_series[d]) if d < len(atr_series) else float("nan")
    is_long = tr.direction == "LONG"
    return {
        "event_id": int(tr.event_id),
        "structure_id": str(tr.candidate_structure_id),
        "structure_type": str(tr.candidate_structure_type),
        "timeframe": (
            str(tr.candidate_structure_id).split("|")[1]
            if str(tr.candidate_structure_id).startswith("SR|") else None
        ),
        "zone_bottom": float(tr.zone_bottom),
        "zone_top": float(tr.zone_top),
        "candidate_start_bar": int(d),
        "candidate_end_bar": int(tr.exit) + 1,
        "candidate_start_time": pd.Timestamp(mv.times[d]),
        "candidate_end_time": pd.Timestamp(mv.times[tr.exit]),
        "atr_value": float(atr),
        "atr_owner": ATR_OWNER,
        "best_entry_decision_index": int(d),
        "best_entry_fill_index": int(tr.fill),
        "best_entry_fill_time": pd.Timestamp(mv.times[tr.fill]),
        "best_entry_price": float(tr.entry_price),
        "best_entry_gap_atr": float(tr.entry_gap_atr),
        "exit_fill_index": int(tr.exit),
        "exit_fill_time": pd.Timestamp(mv.times[tr.exit]),
        "exit_price": float(tr.exit_price),
        "exit_reason": "TARGET_TOUCH",
        "oracle_direction": str(tr.direction),
        "oracle_decision": str(tr.direction),
        "canonical_oracle_trade": True,
        "long_branch_valid": is_long,
        "short_branch_valid": (not is_long),
        "long_branch_invalid_reason": (
            None if is_long else "NO_VALID_TARGET_BRANCH"
        ),
        "short_branch_invalid_reason": (
            None if (not is_long) else "NO_VALID_TARGET_BRANCH"
        ),
        "long_value_points": (float(tr.utility_points) if is_long else None),
        "short_value_points": (float(tr.utility_points) if (not is_long) else None),
        "long_value_atr": (
            float(tr.utility_atr) if (is_long and np.isfinite(atr)) else None
        ),
        "short_value_atr": (
            float(tr.utility_atr) if ((not is_long) and np.isfinite(atr)) else None
        ),
        "long_target_price": (float(tr.target_price) if is_long else None),
        "short_target_price": (float(tr.target_price) if (not is_long) else None),
        "long_target_sid": (str(tr.target_structure_id) if is_long else None),
        "short_target_sid": (str(tr.target_structure_id) if (not is_long) else None),
        "target_structure_id": str(tr.target_structure_id),
        "target_structure_type": str(tr.target_structure_type),
        "target_price": float(tr.target_price),
        "target_timeframe": (
            str(tr.target_structure_id).split("|")[1]
            if str(tr.target_structure_id).startswith("SR|") else None
        ),
        "tp_atr": float(tr.utility_atr),
        "utility": float(tr.utility_points),
        "utility_atr": float(tr.utility_atr),
        "positive_tp_exists": True,
        "remaining_target_atr": 0.0,
        "target_distance_atr": float(tr.utility_atr),
        "direction_margin_atr": float(tr.utility_atr),
        "label_available_time": (
            pd.Timestamp(mv.times[tr.exit]) + pd.Timedelta(minutes=15)
        ),
        "next_structural_event_id": -1,
        "next_structural_event_start_time": pd.NaT,
        "next_structural_event_structure_id": None,
    }


def _make_mv_helper(opens, highs, lows, closes, times, td, seg):
    """Small wrapper used by tests to build an EventMarketView from arrays."""
    unit_starts = _unit_starts_from_arrays(
        np.asarray(td, np.int64), np.asarray(seg, np.int64)
    )
    return market_view_from_arrays(
        np.asarray(opens, float),
        np.asarray(highs, float),
        np.asarray(lows, float),
        np.asarray(closes, float),
        times=np.asarray(times),
        unit_starts=unit_starts,
        symbol="TEST",
    )


def _load_inputs(
    symbol: str,
    max_bars: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the shared V5 inputs (price arrays, geometry, market view).

    Exposed so tests / audits can read ``geom`` and the raw series without
    re-running the full oracle.
    """
    ev = build_structural_events_v2(symbol, max_bars)
    events = ev["events"]
    meta_events = ev["meta"]

    prox = build_dp_proximity_m15(symbol, max_bars)
    n = len(prox)
    opens = prox["open"].to_numpy(float)
    highs = prox["high"].to_numpy(float)
    lows = prox["low"].to_numpy(float)
    closes = prox["close"].to_numpy(float)
    times = prox["bar_start_time"].to_numpy()
    seg_arr = prox["segment"].to_numpy(np.int64)
    td_arr = prox["trading_day"].to_numpy(np.int64)

    mv = market_view_from_arrays(
        opens, highs, lows, closes, times=times,
        unit_starts=_unit_starts_from_arrays(td_arr, seg_arr), symbol=symbol,
    )

    env = run_environment_m15(symbol, max_bars, capture_provenance=False)
    geom = env["geom_by_decision"]
    atr_series = np.asarray(env["features"]["m15_atr"].to_numpy(float))

    sr_fs: Dict[Any, int] = {}
    per_bar = build_per_bar_proximity(prox, geom, sr_fs)

    event_id_by_sid = {e["structure_id"]: int(e["event_id"]) for e in events}

    return {
        "events": events,
        "meta_events": meta_events,
        "n": n,
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "times": times,
        "seg_arr": seg_arr,
        "td_arr": td_arr,
        "mv": mv,
        "geom": geom,
        "atr_series": atr_series,
        "per_bar": per_bar,
        "sr_fs": sr_fs,
        "event_id_by_sid": event_id_by_sid,
    }


def run_god_oracle_v5(
    symbol: str,
    max_bars: Optional[int] = None,
) -> Dict[str, Any]:
    """Production V5 God-mode oracle.

    Pipeline: build environment + proximity -> enumerate ALL atomic
    TradeOptions (competing-first-passage direction) -> global non-overlap
    weighted-interval DP -> canonical records.
    """
    inp = _load_inputs(symbol, max_bars)
    n = inp["n"]
    opens = inp["opens"]
    highs = inp["highs"]
    lows = inp["lows"]
    closes = inp["closes"]
    atr_series = inp["atr_series"]
    td_arr = inp["td_arr"]
    seg_arr = inp["seg_arr"]
    mv = inp["mv"]
    geom = inp["geom"]
    per_bar = inp["per_bar"]
    sr_fs = inp["sr_fs"]
    event_id_by_sid = inp["event_id_by_sid"]
    meta_events = inp["meta_events"]

    options = enumerate_trade_options(
        per_bar=per_bar,
        geom=geom,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        atr_series=atr_series,
        trading_day=td_arr,
        segment=seg_arr,
        mv=mv,
        sr_first_seen=sr_fs,
        event_id_by_sid=event_id_by_sid,
    )

    selected = solve_nonoverlap_god_dp(options, n)

    records = [_to_record(tr, mv, atr_series) for tr in selected]

    meta = _build_meta(
        records, meta_events, self_target_total=0,
        cross_day=0, cross_seg=0, cross_unit=0,
    )
    meta["symbol"] = symbol
    meta["math_version"] = "structural-god-oracle-v5"
    return {"records": records, "meta": meta, "market_view": mv, "options": options}


def main(symbol: str = "AG") -> None:
    res = run_god_oracle_v5(symbol)
    records = res["records"]
    meta = res["meta"]
    canon = [r for r in records if r["canonical_oracle_trade"]]
    print(f"=== structural-god-oracle-v5 ({symbol}) ===")
    print("META:", {k: v for k, v in meta.items()})
    print(f"canonical_trades={len(canon)}")
    for r in canon[:20]:
        print(
            f"  d={r['best_entry_decision_index']} "
            f"dir={r['oracle_direction']} "
            f"sid={r['structure_id']} "
            f"entry={r['best_entry_price']:.1f} "
            f"target={r['target_price']:.1f} "
            f"exit={r['exit_fill_index']} "
            f"util={r['utility']:.1f}"
        )
    counters = hotpath_counters()
    print("TP counters:", counters)
    assert meta["canonical_loss_count"] == 0
    assert meta["self_target_count"] == 0
    assert meta["cross_day_fill_count"] == 0
    assert meta["cross_segment_fill_count"] == 0
    assert meta["cross_unit_fill_count"] == 0
    assert meta["canonical_oracle_trades"] > 0
    assert meta["target_touch"] == meta["canonical_oracle_trades"]
    print("ALL HARD SANITY CHECKS PASSED")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "AG")
