"""build_trade_oracle_dp_m15_one_entry_proximity_v1
====================================================

15m Oracle — mechanical timeframe port of the frozen 5m R2
``intraday_dp_oracle_r2_one_entry_proximity`` (SHA cc7891723beb7298aa5225275b0697969d8e19bb).

The ONLY semantic substitutions vs the frozen 5m R2:
  * execution clock 5m            -> 15m
  * TF universe 5m/15m/1h/4h      -> 15m/1h/4h
  * ATR5m                         -> ATR15m
  * availability / fill timing    -> +15min (15m clock)
  * 5m opens                      -> 15m opens

Everything else is carried over EXACTLY:
  * 6-state (position p in {-1,0,+1}, entry_right armed q in {0,1}) Bellman
  * 0.50-ATR continuous proximity episode (distance to pre-existing 15m/1h/4h
    SR/LIQ, <= 0.5 ATR); NO gap<=3 merging, NO re-arm countdown
  * at most ONE new entry per proximity episode; re-arm only after price leaves
    ALL proximity (P_{t+1} == 0)
  * free exit / hold; reversal counts as a new entry
  * (trading_day, segment) units; forced-flat unit terminals (no overnight)
  * turnover/cost interface; tie-breaking keep -> flat -> fallback
  * full 6x3 Bellman Q persistence for every decision row
  * trade reconstruction

The DP's proximity is OWNED by ``build_dp_proximity_m15_v1`` (dp_proximity_*).
The R4 Candidate Trading Zones (candidate_gate_r4, candidate_any, touch_proof,
merged_episode_id, quota_reset_after, gap<=3) are a SEPARATE subsystem and are
NOT imported or referenced by this module.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.build_dp_proximity_m15_v1 import (
    build_dp_proximity_m15,
    compute_proximity_episode_id,
)
from research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1 import (
    ENTRY_PROX_ATR,
    KernelCounters,
)


# --------------------------------------------------------------------------- #
# Frozen DP constants (contract)                                                #
# --------------------------------------------------------------------------- #
POS = np.array([-1, 0, 1], dtype=np.int8)
ACT = POS.copy()
INVALID_ACTION = np.int8(127)
EPS = 1e-9

# 6-state ordering: (Short,0),(Short,1),(Flat,0),(Flat,1),(Long,0),(Long,1)
STATE_POS = np.repeat(POS, 2)
STATE_ARMED = np.tile(np.array([0, 1], dtype=np.int8), 3)
S2I = {(int(p), int(q)): i for i, (p, q) in enumerate(zip(STATE_POS, STATE_ARMED))}
STATES = [(-1, 0), (-1, 1), (0, 0), (0, 1), (1, 0), (1, 1)]
STATE_LABELS = ["S0", "S1", "F0", "F1", "L0", "L1"]
ACTION_LETTER = {0: "s", 1: "f", 2: "l"}

# [6 states, 3 actions]
TURNOVER = np.abs(STATE_POS[:, None] - ACT[None, :]).astype(float)

# New-entry actions: Flat->±1, or reversal p -> -p.
NEW_ENTRY = (
    ((STATE_POS[:, None] == 0) & (ACT[None, :] != 0))
    | ((STATE_POS[:, None] != 0) & (ACT[None, :] == -STATE_POS[:, None]))
)
# Hold / Exit / Flat do NOT require entry permission.
BASE_ALLOWED = ~NEW_ENTRY

MATH_VERSION = "intraday_dp_oracle_r2_one_entry_proximity_15m_port"
TASK_ID = "FUTURE-INTRADAY-DP-ORACLE-R2-ONE-ENTRY-PROXIMITY-15M"

# unit terminal vocabulary
DATA_END = "DATA_END"
DISCONTINUITY = "DISCONTINUITY"
TRADING_DAY_END = "TRADING_DAY_END"


# --------------------------------------------------------------------------- #
# New-entry predicate + 6-state action selection                                #
# --------------------------------------------------------------------------- #
def is_new_entry(p: int, a: int) -> bool:
    return (p == 0 and a != 0) or (p != 0 and a == -p)


def choose_actions_v2(Q: np.ndarray, eps: float = EPS):
    """Q: [6 states, 3 actions].

    Tie preference: 1. keep same position  2. Flat  3. deterministic fallback.
    A true value tie is still separately marked ambiguous (never forced).
    """
    vmax = np.max(Q, axis=1)
    best = np.abs(Q - vmax[:, None]) <= eps
    ambiguous = best.sum(axis=1) > 1

    chosen = np.empty(len(STATE_POS), dtype=np.int8)
    for si in range(len(STATE_POS)):
        p = int(STATE_POS[si])
        keep_idx = int(p + 1)
        if best[si, keep_idx]:
            chosen[si] = p
        elif best[si, 1]:
            chosen[si] = 0
        else:
            j = int(np.flatnonzero(best[si])[0])
            chosen[si] = ACT[j]

    sortq = np.sort(Q, axis=1)
    edge = sortq[:, -1] - sortq[:, -2]
    return chosen, ambiguous, edge, vmax


# --------------------------------------------------------------------------- #
# 6-state Bellman DP                                                           #
# --------------------------------------------------------------------------- #
def solve_day_dp_v2(
    opens: np.ndarray,
    proximity_any: np.ndarray,
    cost_points: np.ndarray,
    start: int,
    end: int,
) -> Dict[str, np.ndarray]:
    """Bars in unit: [start, end] inclusive; decisions start ... end-1.

    Decision end-1 is forced Flat at Open[end]. Returns the full per-decision
    Q[6,3] table plus best action / edge / ambiguity for every one of the 6
    states.
    """
    n = len(opens)

    actions = np.full((n, 6), INVALID_ACTION, dtype=np.int8)
    ambiguous = np.zeros((n, 6), dtype=bool)
    edges = np.full((n, 6), np.nan, dtype=float)
    Q_all = np.full((n, 6, 3), np.nan, dtype=float)

    V_next = np.zeros(6, dtype=float)

    delta = np.zeros(n, dtype=float)
    idx = np.arange(start, end - 1)
    if len(idx):
        delta[idx] = opens[idx + 2] - opens[idx + 1]

    for t in range(end - 1, start - 1, -1):
        terminal = t == end - 1

        pnl = ACT[None, :] * delta[t]
        cost = float(cost_points[t]) * TURNOVER

        if terminal:
            allowed = ACT[None, :] == 0
            future = 0.0
        else:
            # New directional exposure requires proximity AND an unused right.
            can_open = bool(proximity_any[t]) & STATE_ARMED.astype(bool)
            allowed = BASE_ALLOWED | (NEW_ENTRY & can_open[:, None])

            armed_after = np.where(NEW_ENTRY, 0, STATE_ARMED[:, None]).astype(np.int8)
            # leaving ALL proximity zones rearms the next decision.
            if not bool(proximity_any[t + 1]):
                next_armed = np.ones_like(armed_after, dtype=np.int8)
            else:
                next_armed = armed_after
            # state index: (-1,q)->0/1, (0,q)->2/3, (+1,q)->4/5
            next_state_idx = (ACT[None, :] + 1) * 2 + next_armed
            future = V_next[next_state_idx]

        Q = pnl - cost + future
        Q = np.where(allowed, Q, -np.inf)

        chosen, amb, edge, V_cur = choose_actions_v2(Q)

        actions[t] = chosen
        ambiguous[t] = amb
        edges[t] = edge
        Q_all[t] = Q
        V_next = V_cur

    return {"actions": actions, "ambiguous": ambiguous, "edges": edges, "Q": Q_all}


# --------------------------------------------------------------------------- #
# Independent exhaustive reference (NO memo, NO production helper)              #
# --------------------------------------------------------------------------- #
def exhaustive_reference_v2(
    opens: np.ndarray,
    proximity: np.ndarray,
    cost: np.ndarray,
    start: int,
    end: int,
    start_pos: int = 0,
    start_armed: int = 1,
) -> Tuple[float, set]:
    """True DFS over ALL legal paths (NO memo, NO production helper).

    ``start_pos`` / ``start_armed`` default to the frozen contract start
    (Flat, armed=1). Exposed fully so tests can validate all 6 counterfactual
    state values. Returns (best_value, set(best_action_paths)).
    """
    best = -np.inf
    best_paths: List[tuple] = []

    def dfs(t: int, p: int, q: int, pnl: float, path: List[int]) -> None:
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
        else:
            acts = []
            if p == 0:
                acts.append(0)
            else:
                acts.extend((p, 0))
            if proximity[t] and q == 1:
                if p == 0:
                    acts.extend((-1, 1))
                else:
                    acts.append(-p)

        for a in acts:
            new_entry = (p == 0 and a != 0) or (p != 0 and a == -p)
            move = 0.0 if terminal else a * (opens[t + 2] - opens[t + 1])
            fee = float(cost[t]) * abs(a - p)
            q_after = 0 if new_entry else q
            if terminal:
                q_next = q_after  # unused: dfs(t+1) returns immediately
            else:
                q_next = 1 if not proximity[t + 1] else q_after
            dfs(t + 1, a, q_next, pnl + move - fee, path + [a])

    dfs(start, int(start_pos), int(start_armed), 0.0, [])
    return best, set(best_paths)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
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


def _solve_unit_v2(
    opens: np.ndarray,
    proximity_any: np.ndarray,
    cost: np.ndarray,
    start: int,
    end: int,
) -> Tuple[Dict[str, np.ndarray], int]:
    """Solve one unit on a slice so the total DP cost stays O(N)."""
    length = end - start
    core = solve_day_dp_v2(
        opens[start : end + 2],
        proximity_any[start:end],
        cost[start:end],
        0,
        length,
    )
    return core, length


def _walk_unit_path(
    core: Dict[str, np.ndarray], proximity_any: np.ndarray, s: int, e: int
):
    """Backtrack the optimal (p, q) path from the frozen (Flat, armed=1) start.

    ``proximity_any`` is the GLOBAL array; ``s`` / ``e`` are the unit bounds so
    we can rewire ``q`` from ``proximity_any[s + local + 1]`` (the next
    decision's proximity), exactly as the DP kernel did.
    """
    actions = core["actions"]
    p = 0
    q = 1
    path = []
    for local in range(e - s):
        si = S2I[(p, q)]
        a = int(actions[local, si])
        if a == int(INVALID_ACTION):
            raise RuntimeError(f"missing DP action at t={s + local}, state {(p, q)}")
        new_entry = is_new_entry(p, a)
        q_after = 0 if new_entry else q
        path.append(
            {
                "t": s + local,
                "pb": p,
                "pa": a,
                "qb": q,
                "qa": q_after,
                "ne": new_entry,
                "si": si,
                "local": local,
            }
        )
        p = a
        if (s + local) < e - 1:
            q = 1 if not bool(proximity_any[s + local + 1]) else q_after
    if p != 0:
        raise AssertionError("unit did not terminate flat")
    return path


# --------------------------------------------------------------------------- #
# Runner (production)                                                          #
# --------------------------------------------------------------------------- #
def _dp_from_proximity(
    symbol: str,
    prox: pd.DataFrame,
    counters: KernelCounters,
    *,
    cost_points: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    n = int(len(prox))
    opens = prox["open"].to_numpy(float)[:n]
    highs = prox["high"].to_numpy(float)[:n]
    lows = prox["low"].to_numpy(float)[:n]
    times = prox["bar_start_time"].to_numpy()[:n]
    seg_arr = prox["segment"].to_numpy(np.int64)[:n]
    td_arr = pd.to_datetime(prox["trading_day"]).to_numpy()[:n]
    proximity_bits = np.asarray(prox["dp_proximity_bits"], dtype=np.int64)[:n]
    proximity_any = np.asarray(prox["dp_proximity_any"], dtype=bool)[:n]
    if cost_points is None:
        cost = np.zeros(n, dtype=float)
    else:
        cost = np.asarray(cost_points, dtype=float)

    starts, ends = build_intraday_units(td_arr, seg_arr)
    prox_ep_id = compute_proximity_episode_id(proximity_any, starts, n)

    dec: Dict[str, np.ndarray] = {
        "proximity_bits": np.zeros(n, dtype=np.int64),
        "proximity_any": np.zeros(n, dtype=bool),
        "proximity_episode_id": np.full(n, -1, dtype=np.int64),
        "position_before": np.zeros(n, dtype=np.int8),
        "position_after": np.zeros(n, dtype=np.int8),
        "entry_right_before": np.zeros(n, dtype=np.int8),
        "entry_right_after": np.zeros(n, dtype=np.int8),
        "new_entry_consumed": np.zeros(n, dtype=bool),
        "transition": np.empty(n, dtype=object),
    }
    for label in STATE_LABELS:
        for a_idx, a_letter in enumerate(("s", "f", "l")):
            dec[f"q_{label}_{a_letter}"] = np.full(n, np.nan, dtype=float)
        dec[f"best_{label}"] = np.zeros(n, dtype=np.int8)
        dec[f"edge_{label}"] = np.full(n, np.nan, dtype=float)
        dec[f"amb_{label}"] = np.zeros(n, dtype=bool)
    dec["ambiguous"] = np.zeros(n, dtype=bool)
    dec["terminal_reason"] = np.empty(n, dtype=object)
    dec["training_eligible"] = np.zeros(n, dtype=bool)
    dec["label_available_time"] = np.full(
        n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
    )

    trades: List[Dict[str, Any]] = []
    units: List[Dict[str, Any]] = []
    unit_values: List[float] = []
    valid = np.zeros(n, dtype=bool)

    for s, e in zip((int(x) for x in starts), (int(x) for x in ends)):
        term = _unit_terminal_reason(e, n, seg_arr)
        eligible = term != DATA_END
        # 15m clock availability: decision close i -> known at i close (+15min)
        lav = times[e] + np.timedelta64(15, "m")
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

        core, length = _solve_unit_v2(opens, proximity_any, cost, s, e)
        counters.dp_state_count += (e - s) * len(STATE_POS)
        actions = core["actions"]
        Q = core["Q"]
        edges = core["edges"]
        ambg = core["ambiguous"]
        unit_values.append(float(np.nanmax(Q[0, S2I[(0, 1)], :])))

        # optimal (p, q) path from (Flat, armed=1); rewire q via global prox
        p = 0
        q = 1
        path: List[Dict[str, Any]] = []
        for local in range(length):
            si = S2I[(p, q)]
            a = int(actions[local, si])
            if a == int(INVALID_ACTION):
                raise RuntimeError(f"missing DP action at t={s + local}, state {(p, q)}")
            new_entry = is_new_entry(p, a)
            q_after = 0 if new_entry else q
            path.append(
                {
                    "t": s + local,
                    "pb": p,
                    "pa": a,
                    "qb": q,
                    "qa": q_after,
                    "ne": new_entry,
                    "si": si,
                    "local": local,
                }
            )
            p = a
            if (s + local) < e - 1:
                q = 1 if not bool(proximity_any[s + local + 1]) else q_after
        if p != 0:
            raise AssertionError("unit did not terminate flat")

        for d in path:
            t = int(d["t"])
            si = int(d["si"])
            local = int(d["local"])
            valid[t] = True
            dec["proximity_bits"][t] = int(proximity_bits[t])
            dec["proximity_any"][t] = bool(proximity_any[t])
            dec["proximity_episode_id"][t] = int(prox_ep_id[t])
            dec["position_before"][t] = d["pb"]
            dec["position_after"][t] = d["pa"]
            dec["entry_right_before"][t] = d["qb"]
            dec["entry_right_after"][t] = d["qa"]
            dec["new_entry_consumed"][t] = d["ne"]
            dec["transition"][t] = _transition_label(d["pb"], d["pa"])
            # FULL 6-state x 3-action Bellman output persisted for every decision
            # row, not only the path state. Illegal actions stay -inf (never NaN).
            for sj, label in enumerate(STATE_LABELS):
                for a_idx, a_letter in enumerate(("s", "f", "l")):
                    dec[f"q_{label}_{a_letter}"][t] = float(Q[local, sj, a_idx])
                dec[f"best_{label}"][t] = actions[local, sj]
                dec[f"edge_{label}"][t] = edges[local, sj]
                dec[f"amb_{label}"][t] = bool(ambg[local, sj])
            # ambiguous marker reflects the ACTUAL path state only
            dec["ambiguous"][t] = bool(ambg[local, si])
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
            proximity_bits,
            prox_ep_id,
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
        "proximity_bits": proximity_bits,
        "proximity_any": proximity_any,
        "proximity_episode_id": prox_ep_id,
        "cost_points": cost,
        "sel": sel,
        "decision": dec,
        "starts": starts,
        "ends": ends,
        "units": units,
        "unit_values": unit_values,
        "trades": trades,
    }


def _append_unit_trades(
    trades: List[Dict[str, Any]],
    symbol: str,
    td_arr: np.ndarray,
    times: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    proximity_bits: np.ndarray,
    prox_ep_id: np.ndarray,
    cost: np.ndarray,
    path: List[Dict[str, Any]],
    terminal_reason: str,
    training_eligible: bool,
) -> None:
    """Reconstruct complete trades from the unit's optimal (t, p, a, q) path."""
    open_trade: Optional[Dict[str, Any]] = None

    def start(d: Dict[str, Any]) -> Dict[str, Any]:
        fill = int(d["t"]) + 1
        direction = "LONG" if d["pa"] == 1 else "SHORT"
        return {
            "trade_id": f"{symbol}_{direction}_{fill}",
            "symbol": symbol,
            "trading_day": td_arr[d["t"]],
            "direction": direction,
            "entry_decision_index": int(d["t"]),
            "entry_fill_index": fill,
            "entry_fill_time": times[fill],
            "entry_fill_price": float(opens[fill]),
            "entry_proximity_bits": int(proximity_bits[d["t"]]),
            "entry_proximity_episode_id": int(prox_ep_id[d["t"]]),
            "entry_armed_before": int(d["qb"]),
            "terminal_reason": terminal_reason,
            "training_eligible": bool(training_eligible),
        }

    def close(tr: Dict[str, Any], d: Dict[str, Any]) -> Dict[str, Any]:
        fill = int(d["t"]) + 1
        tr = dict(tr)
        tr.update(
            {
                "exit_decision_index": int(d["t"]),
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
        tr["cost_points"] = float(cost[int(tr["entry_decision_index"])]) + float(
            cost[int(d["t"])]
        )
        tr["net_points"] = tr["gross_points"] - tr["cost_points"]
        mfe, mae = _trade_excursion(
            s, entry, exit_p, tr["entry_fill_index"], tr["exit_fill_index"],
            highs, lows,
        )
        tr["MFE"] = mfe
        tr["MAE"] = mae
        return tr

    for d in path:
        if open_trade is None:
            if d["ne"]:
                open_trade = start(d)
        else:
            if d["pa"] == 0:
                trades.append(close(open_trade, d))
                open_trade = None
            elif d["ne"]:
                trades.append(close(open_trade, d))
                open_trade = start(d)

    if open_trade is not None:
        raise AssertionError("unit DP left an unclosed trade")


# --------------------------------------------------------------------------- #
# Runners                                                                      #
# --------------------------------------------------------------------------- #
def run_dp_m15_one_entry_proximity(
    symbol: str,
    *,
    cost_points: Optional[np.ndarray] = None,
    max_bars: Optional[int] = None,
    counters: Optional[KernelCounters] = None,
) -> Dict[str, Any]:
    """Single-pass production runner for one symbol (15m port of R2)."""
    counters = counters or KernelCounters()
    prox = build_dp_proximity_m15(symbol, max_bars)
    return _dp_from_proximity(symbol, prox, counters, cost_points=cost_points)


# --------------------------------------------------------------------------- #
# Artifact construction                                                        #
# --------------------------------------------------------------------------- #
def build_artifact_frames_v1(result: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    sel = result["sel"]
    times = result["time"]
    dec = result["decision"]
    symbol = result["symbol"]

    action_cols = {
        "symbol": symbol,
        "trading_day": pd.to_datetime(result["trading_day"][sel]),
        "decision_bar_index": sel,
        "decision_time": pd.to_datetime(times[sel]) + pd.Timedelta(minutes=15),
        "dp_proximity_bits": dec["proximity_bits"][sel],
        "dp_proximity_any": dec["proximity_any"][sel],
        "dp_proximity_episode_id": dec["proximity_episode_id"][sel],
        "position_before": dec["position_before"][sel],
        "position_after": dec["position_after"][sel],
        "entry_right_before": dec["entry_right_before"][sel],
        "entry_right_after": dec["entry_right_after"][sel],
        "new_entry_consumed": dec["new_entry_consumed"][sel],
        "transition": dec["transition"][sel],
    }
    for label in STATE_LABELS:
        for a_idx, a_letter in enumerate(("s", "f", "l")):
            action_cols[f"Q_{label}_{a_letter.upper()}"] = dec[f"q_{label}_{a_letter}"][sel]
        action_cols[f"best_{label}"] = dec[f"best_{label}"][sel]
        action_cols[f"edge_{label}"] = dec[f"edge_{label}"][sel]
        action_cols[f"amb_{label}"] = dec[f"amb_{label}"][sel]
    action_cols["ambiguous"] = dec["ambiguous"][sel]
    action_cols["label_available_time"] = pd.to_datetime(dec["label_available_time"][sel])
    action_cols["terminal_reason"] = dec["terminal_reason"][sel]
    action_cols["training_eligible"] = dec["training_eligible"][sel]
    oracle_actions = pd.DataFrame(action_cols)

    trade_cols = [
        "trade_id", "symbol", "trading_day", "direction",
        "entry_decision_index", "entry_fill_index", "entry_fill_time", "entry_fill_price",
        "entry_proximity_bits", "entry_proximity_episode_id", "entry_armed_before",
        "exit_decision_index", "exit_fill_index", "exit_fill_time", "exit_fill_price",
        "holding_bars", "gross_points", "cost_points", "net_points", "MFE", "MAE",
        "terminal_reason", "training_eligible",
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


def artifact_metadata_v1(
    oracle_source_sha: str,
    symbol: str,
    data_start: Any = None,
    data_end: Any = None,
    *,
    cost_mode: str = "zero_cost",
    generated_at: Optional[str] = None,
    row_count_actions: int = 0,
    row_count_trades: int = 0,
) -> Dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "math_version": MATH_VERSION,
        "oracle_source_sha": oracle_source_sha,
        "symbol": symbol,
        "data_start": str(data_start),
        "data_end": str(data_end),
        "objective": "gross_open_to_open_pnl",
        "execution_clock": "15m",
        "cost_mode": cost_mode,
        "entry_proximity_atr": ENTRY_PROX_ATR,
        "tf_universe": ["m15", "h1", "h4"],
        "entry_semantics": (
            "pre_existing_SR_LIQ_within_alpha_ATR; "
            "one_new_entry_per_continuous_proximity_episode"
        ),
        "quota_reset": "only_when_outside_all_SR_LIQ_proximity",
        "execution_semantics": "decision=close(i); fill=open(i+1); unit_flat_both_ends",
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "row_count_actions": int(row_count_actions),
        "row_count_trades": int(row_count_trades),
    }


# --------------------------------------------------------------------------- #
# Artifact writer / reader (fail-closed)                                       #
# --------------------------------------------------------------------------- #
ARTIFACT_ROOT_DIRNAME = "trade_oracle_dp_m15_one_entry_proximity_v1"
ORACLE_ACTIONS_FILE = "oracle_actions.parquet"
ORACLE_TRADES_FILE = "oracle_trades.parquet"
ORACLE_METADATA_FILE = "metadata.json"

# columns the Viewer overlay requires on the trades table (V2 superset)
ORACLE_VIEWER_TRADE_COLUMNS = (
    "direction",
    "entry_fill_time",
    "entry_fill_price",
    "exit_fill_time",
    "exit_fill_price",
    "trade_id",
    "entry_proximity_bits",
    "entry_proximity_episode_id",
    "entry_armed_before",
)


def write_oracle_artifact(
    result: Dict[str, Any],
    root: Any,
    *,
    oracle_source_sha: str,
    generated_at: Optional[str] = None,
) -> Path:
    symbol = result["symbol"]
    outdir = Path(root) / symbol
    outdir.mkdir(parents=True, exist_ok=True)

    frames = build_artifact_frames_v1(result)
    actions = frames["oracle_actions"]
    trades = frames["oracle_trades"]
    actions.to_parquet(outdir / ORACLE_ACTIONS_FILE, index=False)
    trades.to_parquet(outdir / ORACLE_TRADES_FILE, index=False)

    times = result["time"]
    meta = artifact_metadata_v1(
        oracle_source_sha, symbol,
        data_start=pd.Timestamp(times[0]), data_end=pd.Timestamp(times[-1]),
        generated_at=generated_at,
        row_count_actions=len(actions), row_count_trades=len(trades),
    )
    (outdir / ORACLE_METADATA_FILE).write_text(json.dumps(meta, indent=2, default=str))
    return outdir


def load_oracle_artifact(
    root: Any,
    symbol: str,
    *,
    expected_math_version: Optional[str] = MATH_VERSION,
    expected_source_sha: Optional[str] = None,
) -> Dict[str, Any]:
    def _fail(reason: str) -> Dict[str, Any]:
        return {"ok": False, "reason": reason, "actions": None, "trades": None,
                "metadata": None}

    outdir = Path(root) / symbol
    ap = outdir / ORACLE_ACTIONS_FILE
    tp = outdir / ORACLE_TRADES_FILE
    mp = outdir / ORACLE_METADATA_FILE
    if not (ap.exists() and tp.exists() and mp.exists()):
        return _fail("missing_artifact")

    try:
        meta = json.loads(mp.read_text())
    except (OSError, ValueError):
        return _fail("missing_metadata")
    if not isinstance(meta, dict) or "math_version" not in meta:
        return _fail("missing_metadata")
    if expected_math_version is not None and meta.get("math_version") != expected_math_version:
        return _fail("math_version_mismatch")
    if meta.get("symbol") != symbol:
        return _fail("symbol_mismatch")
    if expected_source_sha is not None and meta.get("oracle_source_sha") != expected_source_sha:
        return _fail("source_sha_mismatch")

    try:
        actions = pd.read_parquet(ap)
        trades = pd.read_parquet(tp)
    except (OSError, ValueError):
        return _fail("unreadable_artifact")

    rc_actions = meta.get("row_count_actions")
    rc_trades = meta.get("row_count_trades")
    if rc_actions is not None and len(actions) != int(rc_actions):
        return _fail("row_count_mismatch")
    if rc_trades is not None and len(trades) != int(rc_trades):
        return _fail("row_count_mismatch")

    missing = [c for c in ORACLE_VIEWER_TRADE_COLUMNS if c not in trades.columns]
    if missing:
        return _fail("missing_trade_columns")

    return {"ok": True, "reason": None, "actions": actions, "trades": trades,
            "metadata": meta}


# --------------------------------------------------------------------------- #
# Mechanical invariant checker (used by tests + audit)                          #
# --------------------------------------------------------------------------- #
def check_oracle_invariants(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return counts of invariant violations (all should be 0)."""
    dec = result["decision"]
    prox = result["proximity_any"]
    ep = dec["proximity_episode_id"]
    trades = result["trades"]

    out: Dict[str, Any] = {k: 0 for k in (
        "new_entry_outside_proximity",
        "new_entry_armed_zero",
        "illegal_reversal",
        "cross_day",
        "cross_segment",
        "nonflat_terminal",
        "pnl_mismatch_flag",
    )}

    sel = result["sel"]
    per_ep: Dict[int, int] = {}
    for t in sel:
        t = int(t)
        ne = bool(dec["new_entry_consumed"][t])
        if not ne:
            continue
        if not bool(prox[t]):
            out["new_entry_outside_proximity"] += 1
        if int(dec["entry_right_before"][t]) != 1:
            out["new_entry_armed_zero"] += 1
        eid = int(ep[t])
        per_ep[eid] = per_ep.get(eid, 0) + 1
        pb = int(dec["position_before"][t])
        pa = int(dec["position_after"][t])
        if pb != 0 and pa == -pb:
            # reversal consumes the right -> needs proximity + armed=1
            if not (bool(prox[t]) and int(dec["entry_right_before"][t]) == 1):
                out["illegal_reversal"] += 1

    # max new entries per non-trivial proximity episode (continuous P=1 runs)
    out["max_new_entries_per_episode"] = max(per_ep.values()) if per_ep else 0

    # cross-day / cross-segment trades
    if trades:
        td = pd.to_datetime(result["trading_day"]).to_numpy()
        seg = result["segment"]
        for tr in trades:
            ei, xi = int(tr["entry_fill_index"]), int(tr["exit_fill_index"])
            if td[ei] != td[xi]:
                out["cross_day"] += 1
            if seg[ei] != seg[xi]:
                out["cross_segment"] += 1

    # non-flat unit terminals
    for u in result["units"]:
        e = int(u["seg_end"])
        if e - int(u["seg_start"]) < 2:
            continue
        if int(dec["position_after"][e - 1]) != 0:
            out["nonflat_terminal"] += 1

    # PnL reconstruction
    total_val = sum(result["unit_values"])
    total_gross = sum(float(t["gross_points"]) for t in trades)
    out["pnl_abs_diff"] = abs(total_val - total_gross)
    if out["pnl_abs_diff"] > 1e-6:
        out["pnl_mismatch_flag"] = 1

    return out


# --------------------------------------------------------------------------- #
# CLI builder                                                                   #
# --------------------------------------------------------------------------- #
def build_oracle_m15_for_symbol(symbol: str, root: Optional[str] = None) -> Dict[str, Any]:
    from research.liquidity_oracle_atlas.git_head import git_head

    result = run_dp_m15_one_entry_proximity(symbol)
    out_root = root or str(Path("artifacts") / ARTIFACT_ROOT_DIRNAME)
    write_oracle_artifact(result, out_root, oracle_source_sha=git_head())
    inv = check_oracle_invariants(result)
    frames = build_artifact_frames_v1(result)
    return {
        "symbol": symbol,
        "root": out_root,
        "decisions": int(len(result["sel"])),
        "trades": len(result["trades"]),
        "units": len(result["units"]),
        "invariants": inv,
        "actions": frames["oracle_actions"],
        "trades_df": frames["oracle_trades"],
    }


def event_sort_key(e: Tuple[int, str]) -> Tuple[int, int]:
    """Chronological order; at one fill an EXIT must precede the ENTRY.

    A naive ``sorted(events)`` would order by the KIND string, putting
    "LONG_ENTRY" before "SHORT_EXIT" and fabricating a sequence error out of a
    perfectly legal reversal.
    """
    return (int(e[0]), 0 if "EXIT" in e[1] else 1)


def validate_execution_events(events: List[Tuple[int, str]]) -> List[str]:
    """Independent FSM over execution events. Returns a list of STOP codes."""
    errs: List[str] = []
    pos = 0
    for idx, kind in events:
        if kind in ("LONG_ENTRY", "SHORT_ENTRY"):
            if pos != 0:
                errs.append(f"STOP_DOUBLE_OR_OVERLAP_ENTRY@{idx}:{kind}:pos={pos}")
            pos = 1 if kind == "LONG_ENTRY" else -1
        elif kind == "LONG_EXIT":
            if pos != 1:
                errs.append(f"STOP_EXIT_WITHOUT_LONG@{idx}")
            pos = 0
        elif kind == "SHORT_EXIT":
            if pos != -1:
                errs.append(f"STOP_EXIT_WITHOUT_SHORT@{idx}")
            pos = 0
        else:
            errs.append(f"STOP_UNKNOWN_EVENT@{idx}:{kind}")
    if pos != 0:
        errs.append(f"STOP_UNCLOSED_POSITION:pos={pos}")
    return errs


if __name__ == "__main__":
    import sys

    syms = sys.argv[1:] or ["AG"]
    for sym in syms:
        s = build_oracle_m15_for_symbol(sym)
        print(f"[R2-15m] {sym}: decisions={s['decisions']} trades={s['trades']} "
              f"units={s['units']}")
        print(f"     invariants: {s['invariants']}")
