"""
build_structure_constrained_trade_oracle_dp_v1
==============================================

Research 1 (continuation) — Structure-Constrained Trade Oracle DP V1.

Task ID : FUTURE-STRATEGY-DP-ORACLE-V1
Base SHA: 096a659171d461ccea683214c6dbe55ab2ddd8df

This module answers ONE research question:

    Under the causal constraint that new positions may only be initiated when
    price is near an as-of SR or Liquidity structure, what is the executable
    hindsight-optimal Long/Short/Flat position path within each continuous 5m
    segment, and what entry/exit points does that path imply?

It is the EXECUTOR. It returns an Evidence Packet (oracle labels + counters
+ performance). It defines NO trading rule and draws NO research conclusion.

Frozen contract (must not be changed by the executor):
  * Decision clock  = close(5m bar t);  a position change fills at O_{t+1}.
  * position p_t in {-1,0,+1};  action a_t in {-1,0,+1} (chosen at t, executes
    at O_{t+1}).
  * Entry mask M_t = 1{any TF in {5m,15m,1H,4H} has an SR or Liquidity within
    proximity} — reuses the frozen structure kernel (`select_target` +
    `in_proximity`, R_near = 0.5 x ATR_structureTF). It never "reinvents"
    proximity.
  * Allowed actions:
        p=0,  M=1 -> {-1,0,+1}
        p=0,  M=0 -> {0}
        p!=0, M=1 -> {p,0,-p}      (direct reversal is a NEW entry)
        p!=0, M=0 -> {p,0}
    i.e. Exits may happen anywhere; new Entry / Reversal only near structure.
  * Primary objective = gross open-to-open PnL  a_t (O_{t+2} - O_{t+1}).
    c_roundtrip_ATR = 0 for V1 (gross hindsight upper-bound oracle).
  * No fixed H=6/12/24; natural terminal per continuous segment (Flat at both
    ends, no crossing a discontinuity).
  * Single position, 1 unit, no add, no partial.
  * No DTP/SR/Liquidity term inside the reward.

Complexity contract:  Production DP = O(N * S * A) = O(N)  (S=3, A<=3).
Forbidden: per-decision history rerun, future-exit scans, SR/Liquidity rebuild,
per-trade raw reload.

Canonical owner reused (READ ONLY):
  research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1
    KernelCounters, build_base_frame, _stream_from_base
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1 import (
    KernelCounters,
    TF_ORDER,
    build_base_frame,
    stream_from_base,
    _stream_from_base,
)

try:  # optional metric (kept cheap / dependency-free)
    import tracemalloc  # noqa: F401
except Exception:  # pragma: no cover
    tracemalloc = None


# --------------------------------------------------------------------------- #
# Frozen DP constants (contract §11 — do NOT change)                           #
# --------------------------------------------------------------------------- #
POS = np.array([-1, 0, 1], dtype=np.int8)
P2I = {-1: 0, 0: 1, 1: 2}
INVALID_ACTION = np.int8(127)
EPS = 1e-6

MATH_VERSION = "structure_constrained_trade_oracle_dp_v1"
TASK_ID = "FUTURE-STRATEGY-DP-ORACLE-V1"
NEAR_ATR_CONTRACT = 0.50

DATA_END = "DATA_END"
DISCONTINUITY = "DISCONTINUITY"


# --------------------------------------------------------------------------- #
# Allowed actions + tie-aware selection (contract §11 — verbatim, frozen)      #
# --------------------------------------------------------------------------- #
def _allowed_actions(p: int, entry_ok: bool, terminal: bool):
    if terminal:
        return (0,)

    if p == 0:
        return (-1, 0, 1) if entry_ok else (0,)

    # existing position may always hold or exit.
    # direct reversal is a new entry, so only allowed near structure.
    if entry_ok:
        return (p, 0, -p)
    return (p, 0)


def _choose_tie_aware(actions, values, current_pos, eps=EPS):
    values = np.asarray(values, float)
    vmax = float(np.max(values))

    best = [
        int(a)
        for a, v in zip(actions, values)
        if abs(float(v) - vmax) <= eps
    ]

    ambiguous = len(best) > 1

    # Among true value ties prefer no unnecessary turnover.
    if current_pos in best:
        chosen = current_pos
    elif 0 in best:
        chosen = 0
    else:
        # Extremely rare symmetric Long/Short tie.
        # Deterministic only for path reconstruction; label remains ambiguous.
        chosen = min(best)

    sv = np.sort(values)[::-1]
    second = float(sv[1]) if len(sv) > 1 else -np.inf
    edge = vmax - second if np.isfinite(second) else np.nan

    return int(chosen), bool(ambiguous), float(edge), vmax


def _switch_cost(p: int, q: int, atr_t: float, c_roundtrip_atr: float) -> float:
    """Transition friction in PRICE POINTS (fail-closed if scale missing)."""
    if c_roundtrip_atr == 0.0:
        return 0.0
    if not np.isfinite(atr_t) or atr_t <= 0:
        return np.inf if int(q) != int(p) else 0.0
    return 0.5 * c_roundtrip_atr * float(atr_t) * abs(int(q) - int(p))


# --------------------------------------------------------------------------- #
# Production DP (contract §11 — verbatim core, frozen)                         #
# --------------------------------------------------------------------------- #
def solve_segment_dp(
    opens: np.ndarray,
    atr5: np.ndarray,
    entry_ok: np.ndarray,
    seg_start: int,
    seg_end: int,
    *,
    c_roundtrip_atr: float = 0.0,
):
    """
    Segment bars are [seg_start, seg_end], inclusive.

    Decision t occurs at close(t).
    A position change fills at open(t+1).

    Last decision is t = seg_end - 1 and is forced Flat,
    so every trade exits inside the same segment.

    Complexity:
        O(N_segment * 3 states * <=3 actions)
    """

    n = len(opens)

    action = np.full((n, 3), INVALID_ACTION, dtype=np.int8)
    ambiguous = np.zeros((n, 3), dtype=bool)
    edge = np.full((n, 3), np.nan, dtype=float)
    value = np.full((n, 3), np.nan, dtype=float)

    # V_{t+1}(position)
    V_next = np.zeros(3, dtype=float)

    if seg_end - seg_start < 2:
        return {
            "action": action,
            "ambiguous": ambiguous,
            "edge": edge,
            "value": value,
        }

    for t in range(seg_end - 1, seg_start - 1, -1):
        terminal = (t == seg_end - 1)
        V_cur = np.full(3, -np.inf, dtype=float)

        for pi, p in enumerate(POS):
            acts = _allowed_actions(
                int(p),
                bool(entry_ok[t]),
                terminal,
            )

            qvals = []

            for q in acts:
                qi = P2I[int(q)]

                switch_cost = _switch_cost(
                    int(p), int(q), atr5[t], c_roundtrip_atr
                )

                if terminal:
                    price_move = 0.0
                else:
                    # action q fills at O[t+1] and is held to O[t+2]
                    price_move = (
                        float(q)
                        * (float(opens[t + 2]) - float(opens[t + 1]))
                    )

                qv = (
                    price_move
                    - switch_cost
                    + V_next[qi]
                )
                qvals.append(qv)

            chosen, amb, ed, best = _choose_tie_aware(
                acts, qvals, int(p)
            )

            action[t, pi] = np.int8(chosen)
            ambiguous[t, pi] = amb
            edge[t, pi] = ed
            value[t, pi] = best
            V_cur[pi] = best

        V_next = V_cur

    return {
        "action": action,
        "ambiguous": ambiguous,
        "edge": edge,
        "value": value,
    }


# --------------------------------------------------------------------------- #
# Backtrack (contract §11 — verbatim, frozen)                                  #
# --------------------------------------------------------------------------- #
def backtrack_segment(
    core,
    opens,
    atr5,
    times,
    seg_start,
    seg_end,
):
    p = 0
    transitions = []

    for t in range(seg_start, seg_end):
        pi = P2I[p]
        q = int(core["action"][t, pi])

        if q == int(INVALID_ACTION):
            raise RuntimeError(f"missing DP action at t={t}, p={p}")

        if q != p:
            transitions.append({
                "decision_bar_index": int(t),
                "decision_time": times[t] + np.timedelta64(5, "m"),
                "fill_bar_index": int(t + 1),
                "fill_time": times[t + 1],
                "fill_price": float(opens[t + 1]),
                "position_before": int(p),
                "position_after": int(q),
                "transition": (
                    "LONG_ENTRY" if p == 0 and q == 1 else
                    "SHORT_ENTRY" if p == 0 and q == -1 else
                    "LONG_EXIT" if p == 1 and q == 0 else
                    "SHORT_EXIT" if p == -1 and q == 0 else
                    "LONG_TO_SHORT" if p == 1 and q == -1 else
                    "SHORT_TO_LONG"
                ),
                "oracle_ambiguous": bool(
                    core["ambiguous"][t, pi]
                ),
                "oracle_edge_points": float(core["edge"][t, pi]),
                "oracle_edge_atr": (
                    float(core["edge"][t, pi]) / float(atr5[t])
                    if np.isfinite(atr5[t]) and atr5[t] > 0
                    else np.nan
                ),
            })

        p = q

    if p != 0:
        raise AssertionError("segment DP did not terminate flat")

    return transitions


def _transition_label(p: int, q: int) -> str:
    """Full-sweep transition label (includes Hold/Flat)."""
    if p == q:
        return "HOLD" if p != 0 else "FLAT"
    if p == 0 and q == 1:
        return "LONG_ENTRY"
    if p == 0 and q == -1:
        return "SHORT_ENTRY"
    if p == 1 and q == 0:
        return "LONG_EXIT"
    if p == -1 and q == 0:
        return "SHORT_EXIT"
    if p == 1 and q == -1:
        return "LONG_TO_SHORT"
    if p == -1 and q == 1:
        return "SHORT_TO_LONG"
    raise ValueError(f"impossible transition {p} -> {q}")


# --------------------------------------------------------------------------- #
# Independent brute-force Reference (T0 / T1 ONLY — never on production path)  #
# --------------------------------------------------------------------------- #
def _ref_allowed(p: int, entry_ok: bool, terminal: bool) -> Tuple[int, ...]:
    """Independent re-implementation of the allowed-action contract.

    Deliberately does NOT reuse production `_allowed_actions` so the reference
    is a genuinely independent oracle.
    """
    if terminal:
        return (0,)
    if p == 0:
        if entry_ok:
            return (-1, 0, 1)
        return (0,)
    if entry_ok:
        return (p, 0, -p)
    return (p, 0)


def _ref_reward(
    t: int,
    p: int,
    q: int,
    terminal: bool,
    opens: np.ndarray,
    atr5: np.ndarray,
    c_roundtrip_atr: float,
) -> float:
    if terminal:
        price_move = 0.0
    else:
        price_move = float(q) * (float(opens[t + 2]) - float(opens[t + 1]))
    return price_move - _switch_cost(int(p), int(q), atr5[t], c_roundtrip_atr)


def reference_solve_segment(
    opens: np.ndarray,
    atr5: np.ndarray,
    entry_ok: np.ndarray,
    seg_start: int,
    seg_end: int,
    *,
    c_roundtrip_atr: float = 0.0,
) -> Dict[str, Any]:
    """Exhaustive enumeration over the legal action tree (with memoization).

    Independent of the production downward-array DP. Returns, for every
    (t, p) in the segment:

        value[(t, p)]        = V_t(p)
        best_actions[(t, p)] = frozenset of actions attaining V_t(p)
        chosen[(t, p)]       = deterministic tie-aware pick (shared rule)
        ambiguous[(t, p)]    = len(best_actions) > 1

    plus:
        value_start : [V_{seg_start}(-1), V_{seg_start}(0), V_{seg_start}(+1)]
        best_value  : V_{seg_start}(0)
        optimal_paths : all flat-start position paths attaining best_value
    """
    # decisions are t in [seg_start, seg_end - 1]; terminal decision = seg_end - 1
    memo: Dict[Tuple[int, int], Tuple[float, frozenset]] = {}

    def enum(t: int, p: int) -> Tuple[float, frozenset]:
        if t == seg_end:
            # past last decision: only a Flat position is a legal terminal
            if p == 0:
                return 0.0, frozenset()
            return -np.inf, frozenset()

        key = (t, p)
        if key in memo:
            return memo[key]

        terminal = (t == seg_end - 1)
        acts = _ref_allowed(p, bool(entry_ok[t]), terminal)

        best_v = -np.inf
        best_acts = set()
        for q in acts:
            reward = _ref_reward(
                t, p, q, terminal, opens, atr5, c_roundtrip_atr
            )
            sub_v, _ = enum(t + 1, q)
            total = reward + sub_v
            if total > best_v + EPS:
                best_v = total
                best_acts = {int(q)}
            elif abs(total - best_v) <= EPS:
                best_acts.add(int(q))

        res = (float(best_v), frozenset(best_acts))
        memo[key] = res
        return res

    value: Dict[Tuple[int, int], float] = {}
    best_actions: Dict[Tuple[int, int], frozenset] = {}
    chosen: Dict[Tuple[int, int], int] = {}
    ambiguous: Dict[Tuple[int, int], bool] = {}

    if seg_end - seg_start < 2:
        return {
            "value": value,
            "best_actions": best_actions,
            "chosen": chosen,
            "ambiguous": ambiguous,
            "value_start": [np.nan, np.nan, np.nan],
            "best_value": np.nan,
            "optimal_paths": [],
        }

    def _value_at(t: int, p: int) -> float:
        if t == seg_end:
            return 0.0 if p == 0 else -np.inf
        return value[(t, p)]

    # pass 1: exhaustively enumerate V_t(p) for all (t, p)
    for t in range(seg_start, seg_end):
        for p in POS:
            v, acts = enum(t, int(p))
            value[(t, int(p))] = float(v)
            best_actions[(t, int(p))] = acts

    # pass 2: derive the deterministic tie-aware choice from real Q values
    for t in range(seg_start, seg_end):
        for p in POS:
            acts = best_actions[(t, int(p))]
            ordered = sorted(acts) if acts else []
            if not ordered:
                chosen[(t, int(p))] = int(INVALID_ACTION)
                ambiguous[(t, int(p))] = False
                continue
            terminal = (t == seg_end - 1)
            qvals = [
                _ref_reward(t, int(p), int(q), terminal, opens, atr5, c_roundtrip_atr)
                + _value_at(t + 1, int(q))
                for q in ordered
            ]
            pick, amb, _edge, _vmax = _choose_tie_aware(
                ordered, qvals, int(p)
            )
            chosen[(t, int(p))] = int(pick)
            ambiguous[(t, int(p))] = bool(amb)

    # all flat-start optimal position paths
    best_value = value[(seg_start, 0)]
    optimal_paths: List[List[int]] = []

    def collect(t: int, p: int, acc: List[int]) -> None:
        if t == seg_end:
            if p == 0:
                optimal_paths.append(list(acc))
            return
        for q in sorted(best_actions[(t, p)]):
            reward = _ref_reward(
                t, p, q, (t == seg_end - 1), opens, atr5, c_roundtrip_atr
            )
            if abs(reward + _value_at(t + 1, q) - value[(t, p)]) <= 1e-6:
                acc.append(int(q))
                collect(t + 1, q, acc)
                acc.pop()

    if math.isfinite(best_value):
        collect(seg_start, 0, [])

    return {
        "value": value,
        "best_actions": best_actions,
        "chosen": chosen,
        "ambiguous": ambiguous,
        "value_start": [value[(seg_start, int(p))] for p in POS],
        "best_value": float(best_value),
        "optimal_paths": optimal_paths,
    }


# --------------------------------------------------------------------------- #
# Segment enumeration + production runner                                      #
# --------------------------------------------------------------------------- #
def segment_bounds(seg_arr: np.ndarray, n: int) -> List[Tuple[int, int]]:
    """Contiguous inclusive [start, end] runs of equal segment id."""
    if n <= 0:
        return []
    bounds: List[Tuple[int, int]] = []
    s = 0
    for i in range(1, n):
        if seg_arr[i] != seg_arr[i - 1]:
            bounds.append((s, i - 1))
            s = i
    bounds.append((s, n - 1))
    return bounds


def _dp_from_stream(
    symbol: str,
    base: pd.DataFrame,
    res: Dict[str, Any],
    counters: KernelCounters,
    *,
    c_roundtrip_atr: float = 0.0,
) -> Dict[str, Any]:
    """DP stage only: given a base frame + a structure-stream result, run the
    per-segment DP and reconstruct action rows / trades.

    Shared by the real-data runner and the synthetic/testing runner so the DP
    logic is never duplicated.
    """
    n = int(res["n"])
    opens = base["open"].to_numpy(float)[:n]
    highs = base["high"].to_numpy(float)[:n]
    lows = base["low"].to_numpy(float)[:n]
    times = base["time"].to_numpy()[:n]
    seg_arr = base["segment"].to_numpy(np.int64)[:n]
    atr5 = res["atr5m"]
    entry_bits = res["entry_candidate_bits"]
    entry_ok = res["entry_eligible"]

    bounds = segment_bounds(seg_arr, n)
    segments: List[Dict[str, Any]] = []
    cores: Dict[Tuple[int, int], Any] = {}
    for (s, e) in bounds:
        terminal_reason = DATA_END if e == n - 1 else DISCONTINUITY
        segments.append({
            "seg_start": int(s),
            "seg_end": int(e),
            "terminal_reason": terminal_reason,
            "training_eligible": terminal_reason != DATA_END,
        })
        if e - s < 2:
            continue
        core = solve_segment_dp(
            opens, atr5, entry_ok, s, e, c_roundtrip_atr=c_roundtrip_atr
        )
        cores[(s, e)] = core
        counters.dp_state_count += (e - s) * len(POS)

    action_rows = build_oracle_action_rows(
        symbol, times, seg_arr, atr5, entry_bits, cores, segments
    )
    trades = build_oracle_trades(
        symbol, times, opens, highs, lows, atr5, action_rows
    )
    return {
        "symbol": symbol,
        "n": n,
        "open": opens,
        "high": highs,
        "low": lows,
        "time": times,
        "segment": seg_arr,
        "atr5": atr5,
        "entry_candidate_bits": entry_bits,
        "entry_eligible": entry_ok,
        "segments": segments,
        "cores": cores,
        "action_rows": action_rows,
        "trades": trades,
        "c_roundtrip_atr": c_roundtrip_atr,
    }


def run_symbol_dp(
    symbol: str,
    counters: KernelCounters,
    max_bars: Optional[int] = None,
    *,
    c_roundtrip_atr: float = 0.0,
) -> Dict[str, Any]:
    """Single-pass production runner for one symbol.

    raw load once -> build_base_frame once -> structure streaming once
    (capture_entry_bits=True, emit_events=False) -> DP once per segment.

    Returns the oracle inputs, per-decision action rows and trades. Nothing is
    written to disk here (artifact sampling is a later stage).
    """
    t0 = time.perf_counter()
    info = build_base_frame(symbol, counters)
    base = info["base"]
    res = _stream_from_base(
        info["base"], info["form"], info["seg_completed"], counters,
        max_bars, symbol,
        capture_entry_bits=True, emit_events=False,
    )
    t1 = time.perf_counter()
    out = _dp_from_stream(symbol, base, res, counters,
                          c_roundtrip_atr=c_roundtrip_atr)
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
    c_roundtrip_atr: float = 0.0,
) -> Dict[str, Any]:
    """Synthetic/testing entry point: prebuilt base frame -> streaming -> DP.

    No disk load. Reuses the canonical streaming path so synthetic tests exercise
    exactly the same DP contract as the real-data runner.
    """
    res = stream_from_base(
        base, counters, max_bars=max_bars, symbol=symbol,
        capture_entry_bits=True, emit_events=False,
    )
    return _dp_from_stream(symbol, base, res, counters,
                           c_roundtrip_atr=c_roundtrip_atr)


def run_stage_dp(
    symbols: Sequence[str],
    max_bars: Optional[int] = None,
    *,
    c_roundtrip_atr: float = 0.0,
) -> Dict[str, Any]:
    """Run the DP oracle over a list of symbols; aggregate rows + counters."""
    counters = KernelCounters()
    action_rows: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []
    per_symbol: Dict[str, Dict[str, int]] = {}
    runtime = {"candidate_sec": 0.0, "dp_sec": 0.0, "total_sec": 0.0}
    for sym in symbols:
        res = run_symbol_dp(sym, counters, max_bars, c_roundtrip_atr=c_roundtrip_atr)
        action_rows.extend(res["action_rows"])
        trades.extend(res["trades"])
        runtime["candidate_sec"] += res["runtime_candidate_sec"]
        runtime["dp_sec"] += res["runtime_dp_sec"]
        runtime["total_sec"] += res["runtime_total_sec"]
        per_symbol[sym] = {
            "decisions": len(res["action_rows"]),
            "trades": len(res["trades"]),
            "segments": len(res["segments"]),
        }
    return {
        "action_rows": action_rows,
        "trades": trades,
        "counters": counters,
        "per_symbol": per_symbol,
        "runtime": runtime,
    }


# --------------------------------------------------------------------------- #
# Oracle action rows + trades (contract §14)                                   #
# --------------------------------------------------------------------------- #
def _edge_atr(edge_points: float, atr_t: float) -> float:
    if np.isfinite(atr_t) and atr_t > 0 and np.isfinite(edge_points):
        return float(edge_points) / float(atr_t)
    return float("inf") if np.isinf(edge_points) else float("nan")


def build_oracle_action_rows(
    symbol: str,
    times: np.ndarray,
    seg_arr: np.ndarray,
    atr5: np.ndarray,
    entry_bits: Optional[np.ndarray],
    cores: Dict[Tuple[int, int], Any],
    segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """One row per 5m decision: the optimal action + its edge (contract §14)."""
    rows: List[Dict[str, Any]] = []
    for seg in segments:
        s, e = int(seg["seg_start"]), int(seg["seg_end"])
        if (s, e) not in cores:
            continue
        core = cores[(s, e)]
        label_available_time = pd.Timestamp(times[e]) + pd.Timedelta(minutes=5)
        p = 0
        for t in range(s, e):
            pi = P2I[p]
            q = int(core["action"][t, pi])
            if q == int(INVALID_ACTION):
                raise RuntimeError(f"missing DP action at t={t}, p={p}")

            bits = int(entry_bits[t]) if entry_bits is not None else 0
            edge_points = float(core["edge"][t, pi])
            rows.append({
                "symbol": symbol,
                "decision_bar_index": int(t),
                "decision_time": pd.Timestamp(times[t]) + pd.Timedelta(minutes=5),
                "segment": int(seg_arr[t]),
                "entry_candidate_bits": bits,
                "entry_eligible": bool(bits != 0),
                "oracle_position_before": int(p),
                "oracle_position_after": int(q),
                "oracle_transition": _transition_label(int(p), int(q)),
                "oracle_edge_points": edge_points,
                "oracle_edge_ATR": _edge_atr(edge_points, float(atr5[t])),
                "oracle_ambiguous": bool(core["ambiguous"][t, pi]),
                "label_available_time": label_available_time,
                "terminal_reason": seg["terminal_reason"],
                "training_eligible": bool(seg["training_eligible"]),
            })
            p = q
    return rows


def build_oracle_trades(
    symbol: str,
    times: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    atr5: np.ndarray,
    action_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Pair entries/exits (incl. reversals) into complete trade rows (§14)."""
    trades: List[Dict[str, Any]] = []
    open_trade: Optional[Dict[str, Any]] = None

    def start(r: Dict[str, Any]) -> Dict[str, Any]:
        fill = int(r["decision_bar_index"]) + 1
        direction = "LONG" if int(r["oracle_position_after"]) == 1 else "SHORT"
        return {
            "trade_id": f"{symbol}_{direction}_{fill}",
            "symbol": symbol,
            "direction": direction,
            "entry_decision_index": int(r["decision_bar_index"]),
            "entry_fill_index": fill,
            "entry_fill_time": pd.Timestamp(times[fill]),
            "entry_fill_price": float(opens[fill]),
            "entry_candidate_bits": int(r["entry_candidate_bits"]),
            "entry_edge_ATR": float(r["oracle_edge_ATR"]),
            "terminal_reason": r["terminal_reason"],
            "training_eligible": bool(r["training_eligible"]),
        }

    def close(tr: Dict[str, Any], r: Dict[str, Any]) -> Dict[str, Any]:
        fill = int(r["decision_bar_index"]) + 1
        tr = dict(tr)
        tr.update({
            "exit_decision_index": int(r["decision_bar_index"]),
            "exit_fill_index": fill,
            "exit_fill_time": pd.Timestamp(times[fill]),
            "exit_fill_price": float(opens[fill]),
            "exit_edge_ATR": float(r["oracle_edge_ATR"]),
        })
        s = 1 if tr["direction"] == "LONG" else -1
        entry = float(tr["entry_fill_price"])
        exit_p = float(tr["exit_fill_price"])
        a0 = float(atr5[int(tr["entry_decision_index"])])
        tr["holding_bars"] = int(fill - int(tr["entry_fill_index"]))
        tr["gross_points"] = s * (exit_p - entry)
        tr["gross_ATR"] = (
            tr["gross_points"] / a0 if np.isfinite(a0) and a0 > 0 else np.nan
        )
        # MFE / MAE over the holding path: bars [entry_fill_index, exit_fill_index-1]
        # PLUS the entry point and the exit fill point (both are reached, so the
        # realized gross return always lies inside [MAE, MFE]).
        lo_i = int(tr["entry_fill_index"])
        hi_i = int(tr["exit_fill_index"])
        if hi_i > lo_i:
            pts_hi = max(float(np.max(highs[lo_i:hi_i])), exit_p, entry)
            pts_lo = min(float(np.min(lows[lo_i:hi_i])), exit_p, entry)
            if s == 1:
                mfe_pts, mae_pts = pts_hi - entry, pts_lo - entry
            else:
                mfe_pts, mae_pts = entry - pts_lo, entry - pts_hi
            tr["MFE_ATR"] = mfe_pts / a0 if np.isfinite(a0) and a0 > 0 else np.nan
            tr["MAE_ATR"] = mae_pts / a0 if np.isfinite(a0) and a0 > 0 else np.nan
        else:
            tr["MFE_ATR"] = 0.0
            tr["MAE_ATR"] = 0.0
        return tr

    for r in action_rows:
        pb = int(r["oracle_position_before"])
        pa = int(r["oracle_position_after"])
        if open_trade is None:
            if pb == 0 and pa != 0:
                open_trade = start(r)
        else:
            if pa == 0:
                trades.append(close(open_trade, r))
                open_trade = None
            elif pa == -pb:  # reversal: close old + open new at same fill
                trades.append(close(open_trade, r))
                open_trade = start(r)
            # else: hold, nothing to do

    if open_trade is not None:
        raise AssertionError("segment DP left an unclosed trade")
    return trades


# --------------------------------------------------------------------------- #
# Artifact construction (capability only — NO disk write in this stage)        #
# --------------------------------------------------------------------------- #
def build_artifact_frames(result: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Build the two oracle tables WITHOUT writing them to disk (contract §14)."""
    actions = pd.DataFrame(result["action_rows"])
    trades = pd.DataFrame(result["trades"])
    return {"oracle_actions": actions, "oracle_trades": trades}


def artifact_metadata(
    source_sha: str,
    symbol: str,
    data_start: Any,
    data_end: Any,
    c_roundtrip_atr: float = 0.0,
) -> Dict[str, Any]:
    """Artifact provenance (contract §19) — fail-closed for the future UI."""
    return {
        "source_sha": source_sha,
        "task_id": TASK_ID,
        "math_version": MATH_VERSION,
        "NEAR_ATR": NEAR_ATR_CONTRACT,
        "execution_semantics": "decision=close(t); fill=open(t+1)",
        "objective": "gross_open_to_open_pnl",
        "c_roundtrip_atr": c_roundtrip_atr,
        "symbol": symbol,
        "data_start": str(data_start),
        "data_end": str(data_end),
        "tf_order": list(TF_ORDER),
    }
