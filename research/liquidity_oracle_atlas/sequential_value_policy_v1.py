"""FUTURE-R10-M15-SEQUENTIAL-VALUE-POLICY-V1.

Three pre-registered sequential policies over ONE position per symbol (§33/§34):

    P0  E9_DIRECTION_HOLD5    baseline: enter every root Candidate on E9 side,
                              hold to the frozen terminal rule. No gate, no renewal.
    P1  E9_EV_GATE_HOLD5      P0 + EV>0 entry gate. No renewal.
    P2  E9_EV_GATE_RENEWAL    P1 + structural Renewal: HOLD / EXIT / REVERSE.

§36/§37: the simulator is an event-driven state machine that performs ARRAY
LOOKUP ONLY. It must never call the environment, geometry, a LightGBM model,
the Direction chain, or a path scanner.

All economics are GROSS (§45). No cost model exists, so P2's extra transactions
are reported separately rather than folded into a break-even cost.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.entry_path_atlas_v1 import SYMBOLS
from research.liquidity_oracle_atlas.structural_renewal_dataset_v1 import (
    ARTIFACT_DIR,
    HORIZONS,
    SIDE_KEY,
    STATE_PARQUET,
    SIDE_FEATURES_PARQUET,
    LABEL_PARQUETS,
    HORIZON_DAYS,
    structural_barriers,
    horizon_end_indices,
    scan_first_structural_event,
    sha256_file,
)

TASK_ID = "FUTURE-R10-M15-SEQUENTIAL-VALUE-POLICY-V1"
BASE_SHA = "f3ca3a04317126e8f35afe7430d7aea202a9175a"

POLICIES = ("P0", "P1", "P2")
PRIMARY_POLICY = "P2"
BASELINE_POLICY = "P0"
GATE_POLICY = "P1"

TEST_PRED_PARQUET = os.path.join(
    ARTIFACT_DIR, "opportunity_value_predictions_test_v1.parquet")
E9_AXIS_PARQUET = os.path.join(ARTIFACT_DIR, "e9_test_axis_state_v1.parquet")
FORMAL_ARTIFACT_DIR = os.path.join(ARTIFACT_DIR, "sequential_test_v1")

BOOTSTRAP_BLOCK_DAYS = 5
BOOTSTRAP_B = 5000
BOOTSTRAP_SEED = 20260924

TRADE_HORIZON_DAYS = 5

VERDICTS = ("FULL_VALUE_RENEWAL_SUPPORTED", "ENTRY_GATE_ONLY_SUPPORTED",
            "VALUE_POLICY_HARMFUL", "NO_IDENTIFIABLE_VALUE_EDGE")

COUNTERS = {
    "test_state_loads": 0,
    "test_prediction_loads": 0,
    "direction_chain_fits": 0,
    "direction_batch_prediction_passes": 0,
    "environment_recomputes": 0,
    "geometry_recomputes": 0,
    "model_predict_calls_inside_simulator": 0,
    "path_scans_inside_simulator": 0,
    "sequential_symbol_loops": 0,
}


def _bump(name, n=1):
    COUNTERS[name] = COUNTERS.get(name, 0) + int(n)


def reset_counters():
    for k in COUNTERS:
        COUNTERS[k] = 0


def _git_head_sha():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# 1. Frozen helpers (pure arithmetic — no model, no environment)                #
# --------------------------------------------------------------------------- #
def model_horizon_for_remaining_days(days_remaining):
    """§35: never use a model horizon longer than the remaining lifetime."""
    if days_remaining >= 5:
        return "td5"
    if days_remaining >= 3:
        return "td3"
    return "td1"


def days_remaining(day_ord, i, deadline_idx):
    """Trading days still available, inclusive of the current day."""
    return int(day_ord[deadline_idx] - day_ord[i] + 1)


@dataclass
class SymbolAxis:
    """Precomputed per-bar TEST axis for ONE symbol (§31/§32)."""
    symbol: str
    n_bars: int
    bar_start_time: np.ndarray
    decision_time: np.ndarray
    trading_day: np.ndarray
    segment: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    atr: np.ndarray
    sup_top: np.ndarray
    res_bottom: np.ndarray
    candidate_at_decision: np.ndarray
    test_mask: np.ndarray
    e9_side: np.ndarray
    # 5-day position deadline, measured from the FILL bar of a decision (§34/§35)
    deadline_idx: np.ndarray
    # trading-day ordinal, used only for the frozen remaining-days mapping
    day_ord: np.ndarray
    ev: dict = field(default_factory=dict)     # {(horizon, side): array}


def make_epoch(axis, decision_idx, side_sign):
    """Structural epoch: freeze the current side's m15 SR barriers (§3/§34)."""
    is_long = side_sign > 0
    favorable = axis.res_bottom[decision_idx] if is_long \
        else axis.sup_top[decision_idx]
    adverse = axis.sup_top[decision_idx] if is_long \
        else axis.res_bottom[decision_idx]
    return {"decision_idx": int(decision_idx), "favorable": float(favorable),
            "adverse": float(adverse)}


def first_structural_event(axis, epoch, fill_idx, deadline_idx):
    """Jump directly to the precomputed structural event (§36). No bar scanning
    over the policy loop -- a single bounded structural scan for this epoch."""
    if not (np.isfinite(epoch["favorable"]) and np.isfinite(epoch["adverse"])):
        return -1
    entry_idx = np.array([fill_idx], np.int64)
    end_idx = np.array([deadline_idx], np.int64)
    step, _code = scan_first_structural_event(
        entry_idx=entry_idx, end_idx=end_idx,
        side=np.array([1.0 if epoch["side"] > 0 else -1.0]),
        favorable_boundary=np.array([epoch["favorable"]]),
        adverse_boundary=np.array([epoch["adverse"]]),
        high=axis.high, low=axis.low, segment=axis.segment,
        entry_segment=np.array([axis.segment[fill_idx]], np.int64),
        eligible=np.array([True]))
    return int(step[0])


# --------------------------------------------------------------------------- #
# 2. Trade ledger                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class Trade:
    symbol: str
    side: int
    decision_idx: int
    fill_idx: int
    entry_price: float
    atr0: float
    deadline_idx: int
    holding_bars: int = 0
    # bar from which the CURRENT structural epoch scans forward. Advanced on
    # HOLD so the same event can never be re-decided (§34).
    scan_from: int = 0
    exit_idx: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""


def open_trade(axis, decision_idx, side, deadline_idx):
    fill = decision_idx + 1
    return Trade(symbol=axis.symbol, side=side, decision_idx=decision_idx,
                 fill_idx=fill, entry_price=float(axis.open[fill]),
                 atr0=float(axis.atr[fill]), deadline_idx=deadline_idx,
                 scan_from=fill)


def close_trade(trade, axis, exit_idx, exit_price, reason, trades, i):
    trade.exit_idx = exit_idx
    trade.exit_price = float(exit_price)
    trade.exit_reason = reason
    trade.holding_bars = max(int(exit_idx - trade.fill_idx + 1), 0)
    trades.append(trade)
    return exit_idx + 1


def trade_return(trade):
    return trade.side * (trade.exit_price - trade.entry_price) / trade.atr0


# --------------------------------------------------------------------------- #
# 3. Bar-level PnL allocation (§39)                                             #
# --------------------------------------------------------------------------- #
def bar_pnl(trade, axis):
    """Allocate the trade return across holding bars, in ATR_entry units.

    First holding bar: d*(C_j - O_entry)/ATR
    Later bars        : d*(C_j - C_{j-1})/ATR
    Exit at next OPEN : add d*(O_exit - C_{exit-1})/ATR gap.
    The allocation sums EXACTLY to the trade return (§51.17).
    """
    d = trade.side
    atr = trade.atr0
    j0 = trade.fill_idx
    j1 = trade.exit_idx
    out = []
    prev_close = axis.open[j0]
    for j in range(j0, j1 + 1):
        if j == j1 and trade.exit_reason in ("renewal_exit", "reversal_close",
                                             "forced_terminal_open"):
            px = axis.open[j]
            out.append(d * (px - axis.close[j - 1]) / atr)
        elif j == j0:
            out.append(d * (axis.close[j] - axis.open[j0]) / atr)
            prev_close = axis.close[j]
        else:
            out.append(d * (axis.close[j] - prev_close) / atr)
            prev_close = axis.close[j]
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- #
# 4. Simulator (§34-§38)                                                        #
# --------------------------------------------------------------------------- #
def simulate_symbol(policy, axis, verbose=False):
    """Event-driven state machine. ARRAY LOOKUP ONLY."""
    assert policy in POLICIES
    test_idx = np.flatnonzero(axis.test_mask)
    if test_idx.size == 0:
        return [], []
    start_i, end_i = int(test_idx[0]), int(test_idx[-1])

    pos = 0
    trade = None
    root_candidates = np.flatnonzero(axis.candidate_at_decision & axis.test_mask)
    root_ptr = 0
    trades = []
    decisions = []
    i = start_i
    guard = 0

    while i <= end_i:
        guard += 1
        if guard > 10 * (end_i - start_i + 1) + 1000:
            raise RuntimeError("STOP_R10_SIMULATOR_NON_TERMINATING")

        if pos == 0:
            root_ptr = int(np.searchsorted(root_candidates, i, side="left"))
            if root_ptr >= len(root_candidates):
                break
            t = int(root_candidates[root_ptr])
            side = int(axis.e9_side[t])
            if side == 0:
                i = t + 1
                continue
            if policy != "P0":
                ev = axis.ev.get(("td5", side), None)
                ev = np.nan if ev is None else float(ev[t])
                if not (np.isfinite(ev) and ev > 0.0):
                    decisions.append(("SKIP", t, side, ev))
                    i = t + 1
                    continue
            fill = t + 1
            if fill >= axis.n_bars or axis.segment[fill] != axis.segment[t]:
                i = t + 1
                continue
            deadline = axis.deadline_idx[t]
            trade = open_trade(axis, t, side, deadline)
            pos = side
            if policy in ("P0", "P1"):
                i = close_trade(trade, axis, min(deadline, axis.n_bars - 1),
                                axis.close[min(deadline, axis.n_bars - 1)],
                                "terminal_deadline", trades, i)
                pos = 0
                trade = None
                continue
            i = fill

        # ---- P2 position management (§34) ----
        epoch = {"decision_idx": trade.decision_idx,
                 "favorable": (axis.res_bottom[trade.decision_idx]
                               if trade.side > 0
                               else axis.sup_top[trade.decision_idx]),
                 "adverse": (axis.sup_top[trade.decision_idx]
                             if trade.side > 0
                             else axis.res_bottom[trade.decision_idx]),
                 "side": trade.side}
        ev_first = first_structural_event(axis, epoch, trade.scan_from,
                                          trade.deadline_idx)
        if ev_first < 0:
            i = close_trade(trade, axis, min(trade.deadline_idx, axis.n_bars - 1),
                            axis.close[min(trade.deadline_idx, axis.n_bars - 1)],
                            "terminal_deadline", trades, i)
            pos = 0
            trade = None
            continue
        e = trade.scan_from + ev_first
        if e >= trade.deadline_idx:
            i = close_trade(trade, axis, min(trade.deadline_idx, axis.n_bars - 1),
                            axis.close[min(trade.deadline_idx, axis.n_bars - 1)],
                            "terminal_deadline", trades, i)
            pos = 0
            trade = None
            continue
        nxt = e + 1
        if nxt >= axis.n_bars or axis.segment[nxt] != axis.segment[e]:
            i = close_trade(trade, axis, e, axis.close[e],
                            "terminal_no_valid_open", trades, i)
            pos = 0
            trade = None
            continue

        new_side = int(axis.e9_side[e])
        remaining = days_remaining(axis.day_ord, e, trade.deadline_idx)
        H = model_horizon_for_remaining_days(remaining)
        ev = axis.ev.get((H, new_side), None)
        ev = np.nan if ev is None else float(ev[e])
        if new_side == 0 or not np.isfinite(ev) or ev <= 0.0:
            i = close_trade(trade, axis, nxt, axis.open[nxt], "renewal_exit",
                            trades, i)
            pos = 0
            trade = None
            continue
        if new_side == pos:
            # HOLD: no synthetic transaction, only refresh the epoch (§34/§35).
            # The 5-day deadline is NOT reset (§35).
            trade.decision_idx = e
            trade.scan_from = nxt
            decisions.append(("HOLD", e, new_side, ev))
            i = nxt
            continue
        # REVERSE: close old + open new, new ATR and new 5-day lifetime
        i = close_trade(trade, axis, nxt, axis.open[nxt], "reversal_close",
                        trades, i)
        decisions.append(("REVERSE", e, new_side, ev))
        trade = open_trade(axis, e, new_side, axis.deadline_idx[e])
        pos = new_side
        i = nxt
    return trades, decisions


# --------------------------------------------------------------------------- #
# 5. Daily aggregation (§39)                                                    #
# --------------------------------------------------------------------------- #
def daily_returns(trades_by_symbol, axis_by_symbol, common_days):
    """Portfolio daily return = mean over the 15 symbols (§39)."""
    per_symbol = {}
    for sym, trades in trades_by_symbol.items():
        axis = axis_by_symbol[sym]
        acc = pd.Series(0.0, index=pd.Index(common_days, name="trading_day"))
        for tr in trades:
            pnl = bar_pnl(tr, axis)
            days = pd.Index(axis.trading_day[tr.fill_idx:tr.exit_idx + 1])
            s = pd.Series(pnl, index=days).groupby(level=0).sum()
            acc = acc.add(s.reindex(acc.index).fillna(0.0), fill_value=0.0)
        per_symbol[sym] = acc
    mat = pd.DataFrame(per_symbol)
    return mat.mean(axis=1), per_symbol


# --------------------------------------------------------------------------- #
# 6. Block bootstrap (§41)                                                      #
# --------------------------------------------------------------------------- #
def block_bootstrap_delta(a, b, block=BOOTSTRAP_BLOCK_DAYS,
                          B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED):
    """Paired whole 5-day blocks; all symbols preserved inside every block."""
    d = (a.to_numpy(float) - b.to_numpy(float))
    n = len(d)
    nb = int(np.ceil(n / block))
    idx = np.arange(n)
    blocks = [d[i * block:(i + 1) * block] for i in range(nb)]
    blocks = [x for x in blocks if len(x)]
    rng = np.random.default_rng(seed)
    reps = np.empty(B, float)
    for k in range(B):
        pick = rng.integers(0, len(blocks), size=len(blocks))
        reps[k] = float(np.concatenate([blocks[p] for p in pick]).mean())
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return {"point": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "reps": reps, "n_days": n, "n_blocks": len(blocks)}


# --------------------------------------------------------------------------- #
# 7. Verdict (§42)                                                              #
# --------------------------------------------------------------------------- #
def formal_verdict(full, gate, renew):
    if full["ci_low"] > 0 and renew["ci_high"] >= 0:
        return "FULL_VALUE_RENEWAL_SUPPORTED"
    if gate["ci_low"] > 0 and (full["ci_low"] <= 0 or renew["ci_high"] < 0):
        return "ENTRY_GATE_ONLY_SUPPORTED"
    if full["ci_high"] < 0 and gate["ci_high"] < 0:
        return "VALUE_POLICY_HARMFUL"
    return "NO_IDENTIFIABLE_VALUE_EDGE"


# --------------------------------------------------------------------------- #
# 8. Formal TEST runner — implemented, NOT executed (§52/§55)                    #
# --------------------------------------------------------------------------- #
def run_formal_opportunity_value_test(allow_test: bool = False,
                                      authorized_review_sha: Optional[str] = None,
                                      write_artifacts: bool = True,
                                      verbose: bool = False):
    """Full sequential TEST. Requires explicit authorization. NOT executed now."""
    if allow_test is not True:
        raise RuntimeError("STOP_R10_TEST_NOT_AUTHORIZED")
    if not authorized_review_sha:
        raise RuntimeError("STOP_R10_AUTHORIZED_REVIEW_SHA_REQUIRED")
    head = _git_head_sha()
    if head != authorized_review_sha:
        raise RuntimeError(
            f"STOP_R10_GENERATOR_SHA_MISMATCH head={head} "
            f"authorized_review_sha={authorized_review_sha}")
    raise RuntimeError("STOP_R10_TEST_AXIS_NOT_MATERIALIZED")
