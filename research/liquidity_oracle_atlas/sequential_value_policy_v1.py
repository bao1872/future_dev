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
    RENEWAL_AXIS_PARQUET,
    HORIZON_DAYS,
    SCIENTIFIC_STATUS,
    structural_barriers,
    horizon_end_indices,
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
    # FIX10: precomputed TD5 structural-renewal axis (simulation-only outcomes)
    event_idx_long: Optional[np.ndarray] = None
    event_idx_short: Optional[np.ndarray] = None
    renewal_fill_long: Optional[np.ndarray] = None
    renewal_fill_short: Optional[np.ndarray] = None


def make_epoch(axis, decision_idx, side_sign):
    """Structural epoch: freeze the current side's m15 SR barriers (§3/§34)."""
    is_long = side_sign > 0
    favorable = axis.res_bottom[decision_idx] if is_long \
        else axis.sup_top[decision_idx]
    adverse = axis.sup_top[decision_idx] if is_long \
        else axis.res_bottom[decision_idx]
    return {"decision_idx": int(decision_idx), "favorable": float(favorable),
            "adverse": float(adverse)}


def lookup_precomputed_event(axis, decision_idx, side):
    """FIX10: O(1) lookup of the precomputed structural-renewal event.

    R8 already ran ONE vectorized TD5 first-passage scan for every
    (symbol, decision_bar, side); R10 only reads `event_idx_td5` out of that
    axis. No structural scanner is called, directly or indirectly.
    """
    arr = axis.event_idx_long if side > 0 else axis.event_idx_short
    if arr is None:
        raise RuntimeError("STOP_R10_RENEWAL_AXIS_MISSING")
    if decision_idx < 0 or decision_idx >= len(arr):
        return -1, -1
    return int(arr[decision_idx]), int(
        (axis.renewal_fill_long if side > 0 else axis.renewal_fill_short)
        [decision_idx])


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
    # FIX2: decision-time ATR only. ATR at the fill bar is not known at Open.
    return Trade(symbol=axis.symbol, side=side, decision_idx=decision_idx,
                 fill_idx=fill, entry_price=float(axis.open[fill]),
                 atr0=float(axis.atr[decision_idx]), deadline_idx=deadline_idx,
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
        # FIX10: O(1) precomputed event lookup -- no structural scan here.
        e, renewal_fill = lookup_precomputed_event(
            axis, trade.decision_idx, pos)
        deadline_cap = min(trade.deadline_idx, axis.n_bars - 1)
        if (e < 0 or e >= trade.deadline_idx or renewal_fill < 0
                or renewal_fill > trade.deadline_idx):
            i = close_trade(trade, axis, deadline_cap, axis.close[deadline_cap],
                            "terminal_deadline", trades, i)
            pos = 0
            trade = None
            continue
        nxt = renewal_fill
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
def complete_blocks(n_days, block=BOOTSTRAP_BLOCK_DAYS):
    """FIX14: freeze inference to COMPLETE non-overlapping 5-trading-day blocks.

    The terminal remainder of 0-4 days is excluded from the Primary point
    estimate and CI (it is reported descriptively only).
    """
    n_full = int(n_days // block)
    return n_full, int(block * n_full)


def block_bootstrap_delta(a, b, block=BOOTSTRAP_BLOCK_DAYS,
                          B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED):
    """Paired whole 5-day blocks; all symbols preserved inside every block."""
    d_full = (np.asarray(a, float) - np.asarray(b, float))
    n_days = len(d_full)
    n_full, inference_days = complete_blocks(n_days, block)
    d = d_full[:inference_days]
    remainder = d_full[inference_days:]
    if n_full == 0:
        return {"point": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "reps": np.empty(B, float),
                "n_days": n_days, "n_inference_days": 0, "n_blocks": 0,
                "excluded_tail_days": int(len(d_full)),
                "excluded_tail_mean": float(d_full.mean())}
    blocks = [d[i * block:(i + 1) * block] for i in range(n_full)]
    rng = np.random.default_rng(seed)
    reps = np.empty(B, float)
    for k in range(B):
        pick = rng.integers(0, n_full, size=n_full)
        reps[k] = float(np.concatenate([blocks[p] for p in pick]).mean())
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return {"point": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "reps": reps, "n_days": n_days, "n_inference_days": inference_days,
            "n_blocks": n_full,
            "excluded_tail_days": int(len(remainder)),
            "excluded_tail_mean": (float(remainder.mean())
                                   if len(remainder) else 0.0)}


# --------------------------------------------------------------------------- #
# 7. Verdict (§42)                                                              #
# --------------------------------------------------------------------------- #
def policy_decomposition_identity(daily_p2, daily_p1, daily_p0):
    """FIX15: Delta_FULL = Delta_GATE + Delta_RENEW, as paired daily vectors."""
    d_full = (np.asarray(daily_p2, float) - np.asarray(daily_p0, float))
    d_gate = (np.asarray(daily_p1, float) - np.asarray(daily_p0, float))
    d_renew = (np.asarray(daily_p2, float) - np.asarray(daily_p1, float))
    dev = float(np.max(np.abs(d_full - (d_gate + d_renew))))
    if dev > 1e-12:
        raise RuntimeError(f"STOP_R10_DECOMPOSITION_IDENTITY dev={dev}")
    return {"max_abs_dev": dev, "ok": dev <= 1e-12}


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
# --------------------------------------------------------------------------- #
# 8. TEST-axis construction (§32 / FIX12) — behind authorization                 #
# --------------------------------------------------------------------------- #
def build_e9_test_axis(split=None):
    """FIX12: fit the frozen Direction chain ONCE on frozen TRAIN/VAL, then
    batch-predict E9 across the TEST decision axis.

    Returns (e9_axis_parquet_frame, reproduction_gate).
    """
    from research.liquidity_oracle_atlas.direction_null_baseline_v1 import (
        build_frozen_split)
    from research.liquidity_oracle_atlas.direction_gated_experts_v1 import (
        build_direction_expert_data, run_chain)
    if split is None:
        split = build_frozen_split()
    ds = split["ds"]
    data = build_direction_expert_data(ds)
    chain = run_chain(data, ds, split["train_idx"], split["val_idx"],
                      split["test_idx"])
    _bump("direction_chain_fits")
    e9 = np.asarray(chain["E9"], dtype=np.uint8)
    sub = ds.iloc[split["test_idx"]]
    out = pd.DataFrame({
        "symbol": data.symbol[split["test_idx"]],
        "decision_bar": sub["candidate_decision_index"].to_numpy(np.int64),
        "e9_direction": np.where(e9 == 1, "LONG", "SHORT"),
        "e9_side": np.where(e9 == 1, 1, -1)})
    _bump("direction_batch_prediction_passes")
    return out, chain


def e9_axis_reproduction_gate(axis_e9):
    """FIX12: exact equality on all frozen 13,773 overlapping Candidate rows."""
    frozen = pd.read_parquet(
        "artifacts/entry_path_atlas_v1/e9_direction_state_v1.parquet",
        columns=["symbol", "candidate_decision_index", "e9_direction"])
    m = frozen.merge(axis_e9, how="left",
                     left_on=["symbol", "candidate_decision_index"],
                     right_on=["symbol", "decision_bar"])
    bad = int((m["e9_direction_x"].to_numpy(object)
               != m["e9_direction_y"].to_numpy(object)).sum())
    if bad:
        raise RuntimeError(
            f"STOP_R10_E9_AXIS_REPRODUCTION_MISMATCH n_mismatch={bad}")
    return {"n_frozen_rows": int(len(m)), "n_mismatch": bad, "ok": bad == 0}


def build_symbol_axes(state_df, pred_df, renewal_df, e9_df, split, symbols=None):
    """FIX13 step 10: construct all SymbolAxis objects (array lookup only)."""
    from research.liquidity_oracle_atlas.structural_renewal_dataset_v1 import (
        horizon_end_indices)
    t2 = np.datetime64(split["cal"]["cuts"][1], "ns")
    axes = {}
    for sym in (symbols or sorted(state_df["symbol"].unique())):
        s = state_df[state_df["symbol"] == sym].sort_values("bar_index")
        n = len(s)
        day = s["trading_day"].to_numpy()
        seg = s["segment"].to_numpy(np.int64)
        ends = horizon_end_indices(day, seg, n, (5,))
        fill = np.minimum(np.arange(n) + 1, n - 1)
        test_mask = s["decision_time"].to_numpy("datetime64[ns]") >= t2

        def side_arrays(side_name, col):
            p = pred_df[(pred_df["symbol"] == sym)
                        & (pred_df["side"] == side_name)]
            r = renewal_df[(renewal_df["symbol"] == sym)
                           & (renewal_df["side"] == side_name)]
            p = p.set_index("decision_bar").sort_index()
            r = r.set_index("decision_bar").sort_index()
            return (r[col].reindex(np.arange(n)).to_numpy(np.int64),
                    r["renewal_fill_idx_td5"].reindex(np.arange(n))
                    .to_numpy(np.int64))

        ev = {}
        for H in HORIZONS:
            for sd in (1, -1):
                nm = "LONG" if sd > 0 else "SHORT"
                p = pred_df[(pred_df["symbol"] == sym)
                            & (pred_df["side"] == nm)]
                p = p.set_index("decision_bar")[
                    f"{H}_predicted_ev"].reindex(np.arange(n))
                ev[(H, sd)] = p.to_numpy(float)
        e9s = e9_df[e9_df["symbol"] == sym].set_index("decision_bar")["e9_side"]
        e9_arr = np.zeros(n, np.int8)
        vals = e9s.reindex(np.arange(n)).to_numpy(float)
        e9_arr[np.isfinite(vals)] = vals[np.isfinite(vals)].astype(np.int8)

        el, fl = side_arrays("LONG", "event_idx_td5")
        es, fs = side_arrays("SHORT", "event_idx_td5")
        axes[sym] = SymbolAxis(
            symbol=sym, n_bars=n,
            bar_start_time=s["bar_start_time"].to_numpy(),
            decision_time=s["decision_time"].to_numpy(),
            trading_day=day, segment=seg,
            open=s["open"].to_numpy(float), high=s["high"].to_numpy(float),
            low=s["low"].to_numpy(float), close=s["close"].to_numpy(float),
            atr=s["atr"].to_numpy(float),
            sup_top=s["sup_top"].to_numpy(float),
            res_bottom=s["res_bottom"].to_numpy(float),
            candidate_at_decision=s["candidate_at_decision"].to_numpy(bool),
            test_mask=test_mask, e9_side=e9_arr,
            deadline_idx=ends[5][fill],
            day_ord=np.asarray(pd.factorize(day, sort=True)[0], np.int64),
            ev=ev, event_idx_long=el, event_idx_short=es,
            renewal_fill_long=fl, renewal_fill_short=fs)
    return axes


def run_formal_opportunity_value_test(allow_test: bool = False,
                                      authorized_review_sha: Optional[str] = None,
                                      write_artifacts: bool = True,
                                      verbose: bool = False):
    """FIX13: complete Formal TEST call graph. Blocked by default; NOT executed.

    Ordering:
      1  HEAD == authorized SHA
      2  committed PRE-TEST evidence identity
      3  R8 artifact SHA
      4  all nine R9 model SHA
      5  load TEST state once
      6  batch TEST Opportunity predictions once
      7  materialize E9 TEST axis once
      8  13,773 overlap reproduction gate
      9  load precomputed renewal-event axis once
     10 construct all 15 SymbolAxis objects
     11 run P0/P1/P2 sequentially
     12 per-symbol daily PnL
     13 common 15-symbol portfolio daily PnL
     14 paired block inference
     15 frozen verdict
     16 write Formal evidence
     17 write Formal manifest LAST
    """
    if allow_test is not True:
        raise RuntimeError("STOP_R10_TEST_NOT_AUTHORIZED")
    if not authorized_review_sha:
        raise RuntimeError("STOP_R10_AUTHORIZED_REVIEW_SHA_REQUIRED")
    head = _git_head_sha()
    if head != authorized_review_sha:
        raise RuntimeError(
            f"STOP_R10_GENERATOR_SHA_MISMATCH head={head} "
            f"authorized_review_sha={authorized_review_sha}")

    # 2: committed PRE-TEST evidence identity
    ev = os.path.join("research", "liquidity_oracle_atlas", "evidence",
                      "opportunity_value_renewal_v1_pretest_summary.json")
    if not os.path.exists(ev):
        raise RuntimeError("STOP_R10_PRETEST_EVIDENCE_MISSING")
    with open(ev) as f:
        pre = json.load(f)
    # 3 + 4: artifact / model SHA verification
    for name, want in pre["r8_manifest"]["artifact_sha256"].items():
        p = os.path.join(ARTIFACT_DIR, name)
        if sha256_file(p) != want:
            raise RuntimeError(f"STOP_R10_R8_ARTIFACT_SHA_MISMATCH {name}")
    for name, want in pre["r9_model_manifest"]["model_sha256"].items():
        p = os.path.join("artifacts", "opportunity_value_v1", "models", name)
        if sha256_file(p) != want:
            raise RuntimeError(f"STOP_R10_R9_MODEL_SHA_MISMATCH {name}")

    # 5: TEST state once
    state_df = pd.read_parquet(STATE_PARQUET)
    _bump("test_state_loads")
    # 6: batch TEST predictions once
    from research.liquidity_oracle_atlas import opportunity_value_model_v1 as R9
    pred_df = R9.predict_test(allow_test=True,
                              authorized_review_sha=authorized_review_sha,
                              write_artifacts=write_artifacts)
    _bump("test_prediction_loads")
    # 7 + 8: E9 axis once + reproduction gate
    e9_df, _chain = build_e9_test_axis()
    gate = e9_axis_reproduction_gate(e9_df)
    # 9: renewal axis once
    renewal_df = pd.read_parquet(RENEWAL_AXIS_PARQUET)
    # 10
    from research.liquidity_oracle_atlas.direction_null_baseline_v1 import (
        build_frozen_split)
    split = build_frozen_split()
    axes = build_symbol_axes(state_df, pred_df, renewal_df, e9_df, split)
    # 11
    trades_by_policy = {}
    for p in POLICIES:
        per = {}
        for sym, ax in axes.items():
            _bump("sequential_symbol_loops")
            per[sym], _ = simulate_symbol(p, ax)
        trades_by_policy[p] = per
    # 12-13
    common_days = None
    daily_by_policy = {}
    for p in POLICIES:
        if common_days is None:
            common_days = _common_days(axes, split)
        port, per_sym = daily_returns(trades_by_policy[p], axes, common_days)
        daily_by_policy[p] = port
    # 14
    full = block_bootstrap_delta(daily_by_policy[PRIMARY_POLICY],
                                 daily_by_policy[BASELINE_POLICY])
    gate_d = block_bootstrap_delta(daily_by_policy[GATE_POLICY],
                                   daily_by_policy[BASELINE_POLICY])
    renew = block_bootstrap_delta(daily_by_policy[PRIMARY_POLICY],
                                  daily_by_policy[GATE_POLICY])
    # 15
    verdict = formal_verdict(full, gate_d, renew)
    result = {"verdict": verdict, "delta_full": {k: v for k, v in full.items()
                                                 if k != "reps"},
              "delta_gate": {k: v for k, v in gate_d.items() if k != "reps"},
              "delta_renew": {k: v for k, v in renew.items() if k != "reps"},
              "e9_reproduction": gate,
              "scientific_status": SCIENTIFIC_STATUS,
              "performance": dict(COUNTERS)}
    if verbose:
        print(json.dumps(result, indent=2, default=str))
    return result


def _common_days(axes, split):
    import pandas as pd
    sets = None
    for ax in axes.values():
        days = pd.Index(ax.trading_day[ax.test_mask]).unique()
        sets = days if sets is None else sets.intersection(days)
    return np.asarray(sorted(sets))
