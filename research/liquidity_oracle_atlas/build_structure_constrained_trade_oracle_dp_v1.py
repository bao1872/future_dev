"""
build_structure_constrained_trade_oracle_dp_v1
==============================================

Intraday Structure-Constrained Trade Oracle DP V1.

Task ID : FUTURE-INTRADAY-DP-ORACLE-R1
Base SHA: 956c7957d1051d23f203355be853f48d8a7851f0

The model is deliberately "clean": the DP only answers

    "during one trading day (further split by discontinuity), which executable
     Long/Short/Flat position path maximizes that day's gross PnL?"

SR / Liquidity only tell the DP WHERE a new position may be opened; penetration,
reclaim, trend, DTP, event taxonomy never enter the DP (they are re-joined later
only to INTERPRET the oracle labels).

Frozen contract:
  * Solve separately per (trading_day, segment) block ("unit"); every boundary
    must be Flat. Decision i = close(bar i); a change fills at Open_{i+1}; the
    last decision of a unit is forced Flat at Open[end] (no last-5-minute carry).
  * position p in {-1,0,+1} (Short/Flat/Long); single unit, no add, no partial.
  * Entry mask M_i = 1{ current 5m range [L_i,H_i] intersects a PRE-EXISTING
    canonical SR / Liquidity zone } (delta = 0, true touch). 8-bit TF x {SR,LIQ}.
    The 0.5*ATR proximity is NOT used for the DP entry.
  * Allowed actions:
        p=0,  M=0 -> {0}
        p=0,  M=1 -> {-1,0,+1}
        p!=0, M=0 -> {p,0}
        p!=0, M=1 -> {p,0,-p}
    Exit is FREE anywhere; a direct reversal is a new entry (requires M=1).
  * Objective V1 = max GROSS PnL: r_i = a_i (O_{i+2} - O_{i+1}) - c_i |a_i - p_i|
    with cost in PRICE POINTS, c_i = cost_points[i], V1 c_i = 0 (gross oracle).
    No 6/12/24 horizon, no MAE penalty / RR / slope reward / DTP / event term.
  * Bellman:  V_i(p) = max_{a in A_i(p)} [ a(O_{i+2}-O_{i+1}) - c_i|a-p| + V_{i+1}(a) ]
    with V_{end}(0)=0 and p_start = p_end = 0.
  * Outputs TWO answers: (A) the global optimal position path, and (B) the full
    3x3 Q_i(p,a) counterfactual table (best action / edge / ambiguous) so that a
    future Entry model has labels even when the oracle was not flat.

Complexity: DP O(N * S * A) = O(N); structure streaming O(N * 4TF * bounded).
Forbidden: per-decision history rebuild, future-exit scans, per-day raw reload,
per-parameter indicator rebuild, hot-loop dict append.

Canonical owner reused (READ ONLY):
  research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1
    KernelCounters, build_base_frame, stream_from_base, _stream_from_base
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1 import (
    KernelCounters,
    _stream_from_base,
    build_base_frame,
    build_base_from_arrays,
    stream_from_base,
)

try:  # optional; used only for the TP peak-memory profile
    import tracemalloc
except ImportError:  # pragma: no cover
    tracemalloc = None


# --------------------------------------------------------------------------- #
# Frozen DP constants (contract §10 — do NOT redesign)                         #
# --------------------------------------------------------------------------- #
POS = np.array([-1, 0, 1], dtype=np.int8)
P2I = {-1: 0, 0: 1, 1: 2}
INVALID_ACTION = np.int8(127)
EPS = 1e-9

# rows = current p, cols = next a  (order: Short, Flat, Long)
TURNOVER = np.abs(POS[:, None] - POS[None, :]).astype(float)

ALLOW_NO_ENTRY = np.array(
    [
        [True, True, False],  # Short -> Short / Flat
        [False, True, False],  # Flat  -> Flat only
        [False, True, True],  # Long  -> Flat / Long
    ],
    dtype=bool,
)

ALLOW_ENTRY = np.ones((3, 3), dtype=bool)

TERMINAL = np.array(
    [
        [False, True, False],
        [False, True, False],
        [False, True, False],
    ],
    dtype=bool,
)

MATH_VERSION = "intraday_dp_oracle_r1"
TASK_ID = "FUTURE-INTRADAY-DP-ORACLE-R1"

# unit terminal vocabulary (contract §14 / §15)
DATA_END = "DATA_END"  # censored: last bars available
DISCONTINUITY = "DISCONTINUITY"  # unit ended by a segment break
TRADING_DAY_END = "TRADING_DAY_END"  # normal intraday terminal


# --------------------------------------------------------------------------- #
# Core DP (contract §10 — verbatim, frozen)                                    #
# --------------------------------------------------------------------------- #
def choose_actions(Q: np.ndarray, eps: float = EPS):
    """Q: [3 current states, 3 next actions].

    Tie preference: 1. keep same position  2. Flat  3. deterministic fallback.
    A true value tie is still separately marked ambiguous (never forced).
    """
    vmax = np.max(Q, axis=1)
    best = np.abs(Q - vmax[:, None]) <= eps
    ambiguous = best.sum(axis=1) > 1

    chosen = np.empty(3, dtype=np.int8)
    for pi in range(3):
        if best[pi, pi]:
            chosen[pi] = POS[pi]
        elif best[pi, 1]:
            chosen[pi] = 0
        else:
            j = int(np.flatnonzero(best[pi])[0])
            chosen[pi] = POS[j]

    sortq = np.sort(Q, axis=1)
    edge = sortq[:, -1] - sortq[:, -2]

    return chosen, ambiguous, edge, vmax


def solve_day_dp(
    opens: np.ndarray,
    entry_eligible: np.ndarray,
    cost_points: np.ndarray,
    start: int,
    end: int,
) -> Dict[str, np.ndarray]:
    """Bars in unit: [start, end] inclusive; decisions start ... end-1.

    Decision end-1 is forced Flat at Open[end]. Returns the full per-decision
    Q[3,3] table plus best action / edge / ambiguity for every state.
    """
    n = len(opens)

    actions = np.full((n, 3), INVALID_ACTION, dtype=np.int8)
    ambiguous = np.zeros((n, 3), dtype=bool)
    edges = np.full((n, 3), np.nan, dtype=float)
    Q_all = np.full((n, 3, 3), np.nan, dtype=float)

    V_next = np.zeros(3, dtype=float)

    delta = np.zeros(n, dtype=float)
    idx = np.arange(start, end - 1)
    if len(idx):
        delta[idx] = opens[idx + 2] - opens[idx + 1]

    for t in range(end - 1, start - 1, -1):
        terminal = t == end - 1

        pnl = POS[None, :] * delta[t]
        cost = float(cost_points[t]) * TURNOVER
        Q = pnl - cost + V_next[None, :]

        if terminal:
            allowed = TERMINAL
        elif entry_eligible[t]:
            allowed = ALLOW_ENTRY
        else:
            allowed = ALLOW_NO_ENTRY

        Q = np.where(allowed, Q, -np.inf)

        chosen, amb, edge, V_cur = choose_actions(Q)

        actions[t] = chosen
        ambiguous[t] = amb
        edges[t] = edge
        Q_all[t] = Q

        V_next = V_cur

    return {
        "actions": actions,
        "ambiguous": ambiguous,
        "edges": edges,
        "Q": Q_all,
    }


# --------------------------------------------------------------------------- #
# Intraday unit boundaries (contract §11 — vectorized, no pandas groupby)      #
# --------------------------------------------------------------------------- #
def build_intraday_units(
    trading_day: np.ndarray, segment: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Contiguous (trading_day, segment) blocks -> (starts, ends) inclusive."""
    trading_day = np.asarray(trading_day)
    segment = np.asarray(segment)

    boundary = np.empty(len(segment), dtype=bool)
    boundary[0] = True
    if len(segment) > 1:
        boundary[1:] = (trading_day[1:] != trading_day[:-1]) | (
            segment[1:] != segment[:-1]
        )

    starts = np.flatnonzero(boundary)
    ends = np.r_[starts[1:] - 1, len(segment) - 1]
    return starts, ends


# --------------------------------------------------------------------------- #
# Independent exhaustive reference (contract §13 — T0 / T1 ONLY)               #
# --------------------------------------------------------------------------- #
def exhaustive_reference(
    opens: np.ndarray,
    entry_ok: np.ndarray,
    cost: np.ndarray,
    start: int,
    end: int,
    start_pos: int = 0,
) -> Tuple[float, set]:
    """True DFS over ALL legal paths (NO memo, NO production helper).

    ``start_pos`` defaults to 0 (flat start, the frozen contract). It is exposed
    only so tests can validate the counterfactual value V_t(p) for p != 0.
    Returns (best_value, set(best_action_paths)).
    """
    best = -np.inf
    best_paths: List[tuple] = []

    def dfs(t: int, p: int, pnl: float, path: List[int]) -> None:
        nonlocal best, best_paths
        if t == end:
            if p != 0:
                return
            if pnl > best + 1e-9:
                best = pnl
                best_paths = [tuple(path)]
            elif abs(pnl - best) <= 1e-9:
                best_paths.append(tuple(path))
            return

        terminal = t == end - 1
        if terminal:
            acts = (0,)
        elif p == 0:
            acts = (-1, 0, 1) if entry_ok[t] else (0,)
        elif entry_ok[t]:
            acts = (p, 0, -p)
        else:
            acts = (p, 0)

        for a in acts:
            move = 0.0 if terminal else a * (opens[t + 2] - opens[t + 1])
            fee = float(cost[t]) * abs(a - p)
            path.append(int(a))
            dfs(t + 1, a, pnl + move - fee, path)
            path.pop()

    dfs(start, int(start_pos), 0.0, [])
    return best, set(best_paths)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _transition_label(p: int, a: int) -> str:
    if p == a:
        return "HOLD" if p != 0 else "FLAT"
    if p == 0 and a == 1:
        return "LONG_ENTRY"
    if p == 0 and a == -1:
        return "SHORT_ENTRY"
    if p == 1 and a == 0:
        return "LONG_EXIT"
    if p == -1 and a == 0:
        return "SHORT_EXIT"
    if p == 1 and a == -1:
        return "LONG_TO_SHORT"
    if p == -1 and a == 1:
        return "SHORT_TO_LONG"
    raise ValueError(f"impossible transition {p} -> {a}")


def _unit_terminal_reason(end: int, n: int, seg_arr: np.ndarray) -> str:
    if end >= n - 1:
        return DATA_END
    if seg_arr[end + 1] != seg_arr[end]:
        return DISCONTINUITY
    return TRADING_DAY_END


def _solve_unit(
    opens: np.ndarray,
    entry_ok: np.ndarray,
    cost: np.ndarray,
    start: int,
    end: int,
) -> Tuple[Dict[str, np.ndarray], int]:
    """Solve one unit on a slice so the total DP cost stays O(N).

    The frozen ``solve_day_dp`` allocates O(len(opens)); calling it on the unit
    slice [start, end+2] keeps the sum over units O(N).
    """
    length = end - start
    core = solve_day_dp(
        opens[start : end + 2],
        entry_ok[start:end],
        cost[start:end],
        0,
        length,
    )
    return core, length


def _trade_excursion(
    direction_sign: int,
    entry_price: float,
    exit_price: float,
    entry_fill: int,
    exit_fill: int,
    highs: np.ndarray,
    lows: np.ndarray,
) -> Tuple[float, float]:
    """MFE / MAE in PRICE POINTS over [entry_fill, exit_fill-1] + entry/exit pts."""
    lo_i = int(entry_fill)
    hi_i = int(exit_fill)
    if hi_i > lo_i:
        pts_hi = max(float(np.max(highs[lo_i:hi_i])), exit_price, entry_price)
        pts_lo = min(float(np.min(lows[lo_i:hi_i])), exit_price, entry_price)
    else:
        pts_hi = max(exit_price, entry_price)
        pts_lo = min(exit_price, entry_price)
    if direction_sign == 1:
        return pts_hi - entry_price, pts_lo - entry_price
    return entry_price - pts_lo, entry_price - pts_hi


def _append_unit_trades(
    trades: List[Dict[str, Any]],
    symbol: str,
    td_arr: np.ndarray,
    times: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    entry_mask: np.ndarray,
    cost: np.ndarray,
    path: List[Tuple[int, int, int]],
    terminal_reason: str,
    training_eligible: bool,
) -> None:
    """Reconstruct complete trades from the unit's optimal (t, p, a) path."""
    open_trade: Optional[Dict[str, Any]] = None

    def start(t: int, a: int) -> Dict[str, Any]:
        fill = t + 1
        direction = "LONG" if a == 1 else "SHORT"
        return {
            "trade_id": f"{symbol}_{direction}_{fill}",
            "symbol": symbol,
            "trading_day": td_arr[t],
            "direction": direction,
            "entry_decision_index": int(t),
            "entry_fill_index": fill,
            "entry_fill_time": times[fill],
            "entry_fill_price": float(opens[fill]),
            "entry_source_bits": int(entry_mask[t]),
            "terminal_reason": terminal_reason,
            "training_eligible": bool(training_eligible),
        }

    def close(tr: Dict[str, Any], t: int) -> Dict[str, Any]:
        fill = t + 1
        tr = dict(tr)
        tr.update(
            {
                "exit_decision_index": int(t),
                "exit_fill_index": fill,
                "exit_fill_time": times[fill],
                "exit_fill_price": float(opens[fill]),
            }
        )
        s = 1 if tr["direction"] == "LONG" else -1
        entry = float(tr["entry_fill_price"])
        exit_p = float(tr["exit_fill_price"])
        tr["holding_bars"] = int(fill - int(tr["entry_fill_index"]))
        tr["gross_points"] = s * (exit_p - entry)
        # one leg of turnover is attributable to this trade at entry and one at
        # exit (a reversal's 2c is split across the two adjacent trades).
        tr["cost_points"] = float(cost[int(tr["entry_decision_index"])]) + float(
            cost[int(tr["exit_decision_index"])]
        )
        tr["net_points"] = tr["gross_points"] - tr["cost_points"]
        mfe, mae = _trade_excursion(
            s,
            entry,
            exit_p,
            tr["entry_fill_index"],
            tr["exit_fill_index"],
            highs,
            lows,
        )
        tr["MFE"] = mfe
        tr["MAE"] = mae
        return tr

    for t, p, a in path:
        if open_trade is None:
            if p == 0 and a != 0:
                open_trade = start(t, a)
        else:
            if a == 0:
                trades.append(close(open_trade, t))
                open_trade = None
            elif a == -p:  # reversal: close old + open new at the same fill
                trades.append(close(open_trade, t))
                open_trade = start(t, a)

    if open_trade is not None:
        raise AssertionError("unit DP left an unclosed trade")


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #
def _dp_from_stream(
    symbol: str,
    base: pd.DataFrame,
    res: Dict[str, Any],
    counters: KernelCounters,
    *,
    cost_points: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """DP stage: base frame + structure-stream result -> per-unit DP + artifact."""
    n = int(res["n"])
    opens = base["open"].to_numpy(float)[:n]
    highs = base["high"].to_numpy(float)[:n]
    lows = base["low"].to_numpy(float)[:n]
    times = base["time"].to_numpy()[:n]
    seg_arr = base["segment"].to_numpy(np.int64)[:n]
    td_arr = pd.to_datetime(base["trading_day"]).to_numpy()[:n]
    entry_mask = res["entry_mask"]
    entry_ok = np.asarray(res["entry_eligible"], dtype=bool)
    if cost_points is None:
        cost = np.zeros(n, dtype=float)
    else:
        cost = np.asarray(cost_points, dtype=float)

    starts, ends = build_intraday_units(td_arr, seg_arr)

    # preallocated per-decision output arrays (no hot-loop dict append)
    valid = np.zeros(n, dtype=bool)
    dec: Dict[str, np.ndarray] = {
        "pos_before": np.zeros(n, dtype=np.int8),
        "pos_after": np.zeros(n, dtype=np.int8),
        "transition": np.empty(n, dtype=object),
        "q_f_s": np.full(n, np.nan),
        "q_f_f": np.full(n, np.nan),
        "q_f_l": np.full(n, np.nan),
        "best_flat_action": np.zeros(n, dtype=np.int8),
        "flat_edge": np.full(n, np.nan),
        "flat_ambiguous": np.zeros(n, dtype=bool),
        "q_l_s": np.full(n, np.nan),
        "q_l_f": np.full(n, np.nan),
        "q_l_l": np.full(n, np.nan),
        "best_long_action": np.zeros(n, dtype=np.int8),
        "long_edge": np.full(n, np.nan),
        "long_ambiguous": np.zeros(n, dtype=bool),
        "q_s_s": np.full(n, np.nan),
        "q_s_f": np.full(n, np.nan),
        "q_s_l": np.full(n, np.nan),
        "best_short_action": np.zeros(n, dtype=np.int8),
        "short_edge": np.full(n, np.nan),
        "short_ambiguous": np.zeros(n, dtype=bool),
        "ambiguous": np.zeros(n, dtype=bool),
        "terminal_reason": np.empty(n, dtype=object),
        "training_eligible": np.zeros(n, dtype=bool),
        "label_available_time": np.full(
            n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
        ),
    }

    trades: List[Dict[str, Any]] = []
    units: List[Dict[str, Any]] = []
    unit_values: List[float] = []

    for s, e in zip((int(x) for x in starts), (int(x) for x in ends)):
        term = _unit_terminal_reason(e, n, seg_arr)
        eligible = term != DATA_END
        lav = times[e] + np.timedelta64(5, "m")
        units.append(
            {
                "seg_start": s,
                "seg_end": e,
                "terminal_reason": term,
                "training_eligible": eligible,
            }
        )
        if e - s < 2:
            continue

        core, length = _solve_unit(opens, entry_ok, cost, s, e)
        counters.dp_state_count += (e - s) * len(POS)

        actions = core["actions"]
        Q = core["Q"]
        edges = core["edges"]
        ambg = core["ambiguous"]
        unit_values.append(float(np.nanmax(Q[0, P2I[0], :])))

        # oracle path from flat start (local index -> global t = s + local)
        p = 0
        path: List[Tuple[int, int, int]] = []
        for local in range(length):
            pi = P2I[p]
            a = int(actions[local, pi])
            if a == int(INVALID_ACTION):
                raise RuntimeError(f"missing DP action at t={s + local}, p={p}")
            path.append((s + local, p, a))
            p = a
        if p != 0:
            raise AssertionError("unit did not terminate flat")

        for t, pb, pa in path:
            valid[t] = True
            dec["pos_before"][t] = pb
            dec["pos_after"][t] = pa
            dec["transition"][t] = _transition_label(pb, pa)
            dec["q_f_s"][t] = Q[t - s, 1, 0]
            dec["q_f_f"][t] = Q[t - s, 1, 1]
            dec["q_f_l"][t] = Q[t - s, 1, 2]
            dec["best_flat_action"][t] = actions[t - s, 1]
            dec["flat_edge"][t] = edges[t - s, 1]
            dec["flat_ambiguous"][t] = ambg[t - s, 1]
            dec["q_l_s"][t] = Q[t - s, 2, 0]
            dec["q_l_f"][t] = Q[t - s, 2, 1]
            dec["q_l_l"][t] = Q[t - s, 2, 2]
            dec["best_long_action"][t] = actions[t - s, 2]
            dec["long_edge"][t] = edges[t - s, 2]
            dec["long_ambiguous"][t] = ambg[t - s, 2]
            dec["q_s_s"][t] = Q[t - s, 0, 0]
            dec["q_s_f"][t] = Q[t - s, 0, 1]
            dec["q_s_l"][t] = Q[t - s, 0, 2]
            dec["best_short_action"][t] = actions[t - s, 0]
            dec["short_edge"][t] = edges[t - s, 0]
            dec["short_ambiguous"][t] = ambg[t - s, 0]
            dec["ambiguous"][t] = ambg[t - s, P2I[pb]]
            dec["terminal_reason"][t] = term
            dec["training_eligible"][t] = eligible
            dec["label_available_time"][t] = lav

        _append_unit_trades(
            trades,
            symbol,
            td_arr,
            times,
            opens,
            highs,
            lows,
            entry_mask,
            cost,
            path,
            term,
            eligible,
        )

    sel = np.flatnonzero(valid)
    return {
        "symbol": symbol,
        "n": n,
        "open": opens,
        "high": highs,
        "low": lows,
        "time": times,
        "trading_day": td_arr,
        "segment": seg_arr,
        "entry_mask": entry_mask,
        "entry_eligible": entry_ok,
        "cost_points": cost,
        "sel": sel,
        "decision": dec,
        "starts": starts,
        "ends": ends,
        "units": units,
        "unit_values": unit_values,
        "trades": trades,
    }


def run_symbol_dp(
    symbol: str,
    counters: KernelCounters,
    max_bars: Optional[int] = None,
    *,
    cost_points: Optional[np.ndarray] = None,
    profile_memory: bool = False,
) -> Dict[str, Any]:
    """Single-pass production runner for one symbol.

    raw load once -> build_base_frame once -> structure streaming once
    (capture_entry_mask=True, emit_events=False, mask_only=True) -> DP once per
    intraday unit.

    ``profile_memory`` wraps ONLY the DP stage with ``tracemalloc`` (the artifact
    arrays are the memory that scales with N; the full-history base load is a
    constant and tracing it would dominate the runtime).
    """
    t0 = time.perf_counter()
    info = build_base_frame(symbol, counters)
    base = info["base"]
    res = _stream_from_base(
        info["base"],
        info["form"],
        info["seg_completed"],
        counters,
        max_bars,
        symbol,
        capture_entry_mask=True,
        emit_events=False,
        mask_only=True,
    )
    t1 = time.perf_counter()

    if profile_memory and tracemalloc is not None:
        tracemalloc.start()
    out = _dp_from_stream(symbol, base, res, counters, cost_points=cost_points)
    if profile_memory and tracemalloc is not None:
        _cur, peak = tracemalloc.get_traced_memory()
        out["peak_tracemalloc_mb"] = peak / (1024.0 * 1024.0)
        tracemalloc.stop()

    t2 = time.perf_counter()
    out["runtime_candidate_sec"] = t1 - t0
    out["runtime_dp_sec"] = t2 - t1
    out["runtime_total_sec"] = t2 - t0
    return out


def run_base_dp(
    base: pd.DataFrame,
    counters: KernelCounters,
    symbol: str = "SYNTH",
    max_bars: Optional[int] = None,
    *,
    cost_points: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Synthetic/testing entry: prebuilt base frame -> streaming -> intraday DP."""
    res = stream_from_base(
        base,
        counters,
        max_bars=max_bars,
        symbol=symbol,
        capture_entry_mask=True,
        emit_events=False,
        mask_only=True,
    )
    return _dp_from_stream(symbol, base, res, counters, cost_points=cost_points)


def run_arrays_dp(
    time_arr,
    trading_day_arr,
    o,
    h,
    l,
    c,
    disc,
    counters: KernelCounters,
    symbol: str = "PREFIX",
    *,
    cost_points: Optional[np.ndarray] = None,
    profile_memory: bool = False,
) -> Dict[str, Any]:
    """TRUE prefix runner for the N/2N/4N TP benchmark.

    Given already-sliced (aligned, causal) raw arrays, this times the WHOLE
    prefix pipeline:

        canonical raw_frame_from_owner + resample/forming precompute
          -> mask-only structure streaming
          -> intraday DP

    There is NO full-history precompute: the caller slices the prefix and the
    raw load happens outside the timed section, so the measured input is exactly
    the prefix length. ``profile_memory`` wraps the whole prefix pipeline.
    """
    if profile_memory and tracemalloc is not None:
        tracemalloc.start()

    t0 = time.perf_counter()
    info = build_base_from_arrays(
        time_arr, trading_day_arr, o, h, l, c, disc, counters
    )
    base = info["base"]
    res = _stream_from_base(
        info["base"],
        info["form"],
        info["seg_completed"],
        counters,
        None,
        symbol,
        capture_entry_mask=True,
        emit_events=False,
        mask_only=True,
    )
    t1 = time.perf_counter()
    out = _dp_from_stream(symbol, base, res, counters, cost_points=cost_points)
    t2 = time.perf_counter()

    if profile_memory and tracemalloc is not None:
        _cur, peak = tracemalloc.get_traced_memory()
        out["peak_tracemalloc_mb"] = peak / (1024.0 * 1024.0)
        tracemalloc.stop()

    out["runtime_candidate_sec"] = t1 - t0
    out["runtime_dp_sec"] = t2 - t1
    out["runtime_total_sec"] = t2 - t0
    return out


def run_stage_dp(
    symbols: Sequence[str],
    max_bars: Optional[int] = None,
    *,
    cost_points: Optional[np.ndarray] = None,
    profile_memory: bool = False,
) -> Dict[str, Any]:
    """Run the intraday DP oracle over symbols; one DataFrame build at the end."""
    counters = KernelCounters()
    results: List[Dict[str, Any]] = []
    for sym in symbols:
        results.append(
            run_symbol_dp(
                sym,
                counters,
                max_bars,
                cost_points=cost_points,
                profile_memory=profile_memory,
            )
        )
    frames = [build_artifact_frames(r) for r in results]
    oracle_actions = (
        pd.concat([f["oracle_actions"] for f in frames], ignore_index=True)
        if frames
        else pd.DataFrame()
    )
    oracle_trades = (
        pd.concat([f["oracle_trades"] for f in frames], ignore_index=True)
        if frames
        else pd.DataFrame()
    )
    runtime = {
        "candidate_sec": sum(r["runtime_candidate_sec"] for r in results),
        "dp_sec": sum(r["runtime_dp_sec"] for r in results),
        "total_sec": sum(r["runtime_total_sec"] for r in results),
    }
    peak_mb = max((r.get("peak_tracemalloc_mb", 0.0) for r in results), default=0.0)
    return {
        "oracle_actions": oracle_actions,
        "oracle_trades": oracle_trades,
        "counters": counters,
        "runtime": runtime,
        "peak_tracemalloc_mb": peak_mb,
        "per_symbol": {
            r["symbol"]: {
                "decisions": int(len(r["sel"])),  # noqa: RUF046 (len() is an int)
                "trades": len(r["trades"]),
                "units": len(r["units"]),
            }
            for r in results
        },
        "_results": results,
    }


# --------------------------------------------------------------------------- #
# Artifact construction (contract §14 — capability only, NO disk write here)   #
# --------------------------------------------------------------------------- #
def build_artifact_frames(result: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Build the two oracle tables from preallocated arrays (one gather, no per-row dict)."""
    sel = result["sel"]
    times = result["time"]
    dec = result["decision"]
    symbol = result["symbol"]

    oracle_actions = pd.DataFrame(
        {
            "symbol": symbol,
            "trading_day": pd.to_datetime(result["trading_day"][sel]),
            "decision_bar_index": sel,
            "decision_time": pd.to_datetime(times[sel]) + pd.Timedelta(minutes=5),
            "entry_eligible": result["entry_eligible"][sel],
            "entry_source_bits": result["entry_mask"][sel],
            "Q_F_S": dec["q_f_s"][sel],
            "Q_F_F": dec["q_f_f"][sel],
            "Q_F_L": dec["q_f_l"][sel],
            "best_flat_action": dec["best_flat_action"][sel],
            "flat_edge": dec["flat_edge"][sel],
            "flat_ambiguous": dec["flat_ambiguous"][sel],
            "Q_L_S": dec["q_l_s"][sel],
            "Q_L_F": dec["q_l_f"][sel],
            "Q_L_L": dec["q_l_l"][sel],
            "best_long_action": dec["best_long_action"][sel],
            "long_edge": dec["long_edge"][sel],
            "long_ambiguous": dec["long_ambiguous"][sel],
            "Q_S_S": dec["q_s_s"][sel],
            "Q_S_F": dec["q_s_f"][sel],
            "Q_S_L": dec["q_s_l"][sel],
            "best_short_action": dec["best_short_action"][sel],
            "short_edge": dec["short_edge"][sel],
            "short_ambiguous": dec["short_ambiguous"][sel],
            "position_before": dec["pos_before"][sel],
            "position_after": dec["pos_after"][sel],
            "transition": dec["transition"][sel],
            "ambiguous": dec["ambiguous"][sel],
            "label_available_time": pd.to_datetime(dec["label_available_time"][sel]),
            "terminal_reason": dec["terminal_reason"][sel],
            "training_eligible": dec["training_eligible"][sel],
        }
    )

    trade_cols = [
        "trade_id",
        "symbol",
        "trading_day",
        "direction",
        "entry_decision_index",
        "entry_fill_index",
        "entry_fill_time",
        "entry_fill_price",
        "entry_source_bits",
        "exit_decision_index",
        "exit_fill_index",
        "exit_fill_time",
        "exit_fill_price",
        "holding_bars",
        "gross_points",
        "cost_points",
        "net_points",
        "MFE",
        "MAE",
        "terminal_reason",
        "training_eligible",
    ]
    if result["trades"]:
        oracle_trades = pd.DataFrame(result["trades"])
        for c in trade_cols:
            if c not in oracle_trades.columns:
                oracle_trades[c] = np.nan
        oracle_trades = oracle_trades[trade_cols]
    else:
        oracle_trades = pd.DataFrame(columns=trade_cols)

    return {"oracle_actions": oracle_actions, "oracle_trades": oracle_trades}


def artifact_metadata(
    source_sha: str,
    symbol: str,
    data_start: Any = None,
    data_end: Any = None,
    cost_mode: str = "gross_points_zero_cost",
) -> Dict[str, Any]:
    """Artifact provenance (contract §14 / §19) — consumed by the future UI."""
    return {
        "source_sha": source_sha,
        "task_id": TASK_ID,
        "math_version": MATH_VERSION,
        "entry_semantics": "current_5m_range_touch_pre_existing_SR_LIQ_zone_delta0",
        "execution_semantics": "decision=close(i); fill=open(i+1); unit_flat_both_ends",
        "objective": "gross_open_to_open_pnl",
        "cost_mode": cost_mode,
        "symbol": symbol,
        "data_start": str(data_start),
        "data_end": str(data_end),
    }
