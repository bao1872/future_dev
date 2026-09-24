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
# FP3: E9 exists ONLY on canonical ROOT Candidate decision bars.
E9_ROOT_AXIS_PARQUET = os.path.join(ARTIFACT_DIR, "e9_root_axis_v1.parquet")
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
    # FP10: the strategy path must never read TEST labels.
    "test_label_reads_during_strategy": 0,
    # FP11: TEST labels may be read exactly ONCE, only after the verdict.
    "post_verdict_test_label_reads": 0,
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
    # FP9: TEST mask is the CLOSED window [T2, COMMON_END].
    test_mask: np.ndarray
    # FP3: E9 exists ONLY on canonical ROOT Candidate decision bars (0 elsewhere).
    e9_root_side: np.ndarray
    # 5-day position deadline, measured from the FILL bar of a decision, and
    # capped at the common TEST end (FP9).
    deadline_idx: np.ndarray
    # trading-day ordinal, used only for the frozen remaining-days mapping
    day_ord: np.ndarray
    ev: dict = field(default_factory=dict)     # {(horizon, side): array}
    # FIX10: precomputed TD5 structural-renewal axis (simulation-only outcomes)
    event_idx_long: Optional[np.ndarray] = None
    event_idx_short: Optional[np.ndarray] = None
    renewal_fill_long: Optional[np.ndarray] = None
    renewal_fill_short: Optional[np.ndarray] = None
    # FP7: per-bar structural bracket eligibility from the same R8 axis
    bracket_eligible_long: Optional[np.ndarray] = None
    bracket_eligible_short: Optional[np.ndarray] = None


def make_epoch(axis, decision_idx, side_sign):
    """Structural epoch: freeze the current side's m15 SR barriers (§3/§34)."""
    is_long = side_sign > 0
    favorable = axis.res_bottom[decision_idx] if is_long \
        else axis.sup_top[decision_idx]
    adverse = axis.sup_top[decision_idx] if is_long \
        else axis.res_bottom[decision_idx]
    return {"decision_idx": int(decision_idx), "favorable": float(favorable),
            "adverse": float(adverse)}


def bracket_eligible(axis, decision_idx, side) -> bool:
    """FP7: structural bracket eligibility is taken from the R8 renewal axis."""
    arr = (axis.bracket_eligible_long if side > 0
           else axis.bracket_eligible_short)
    if arr is None:
        raise RuntimeError("STOP_R10_BRACKET_AXIS_MISSING")
    return bool(arr[decision_idx])


def lookup_ev(axis, decision_idx, side, horizon):
    if not bracket_eligible(axis, decision_idx, side):
        return -np.inf
    arr = axis.ev.get((horizon, side))
    if arr is None:
        return np.nan
    return float(arr[decision_idx])


def action_value(axis, decision_idx, horizon, side):
    """FP5: an ineligible side has action value -inf (unselectable)."""
    if not bracket_eligible(axis, decision_idx, side):
        return -np.inf
    arr = axis.ev.get((horizon, side))
    if arr is None:
        return -np.inf
    v = float(arr[decision_idx])
    return -np.inf if not np.isfinite(v) else v


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
            # FP6: ROOT entry is E9-gated for all three policies.
            root_ptr = int(np.searchsorted(root_candidates, i, side="left"))
            if root_ptr >= len(root_candidates):
                break
            t = int(root_candidates[root_ptr])
            side = int(axis.e9_root_side[t])
            if side == 0:
                decisions.append(("SKIP_NO_E9", t, side, np.nan))
                i = t + 1
                continue
            if policy != "P0":
                # FP6: P1/P2 additionally require structural bracket eligibility
                # AND EV>0. P0 does NOT (it stays the frozen Direction baseline).
                if not bracket_eligible(axis, t, side):
                    decisions.append(("SKIP_INELIGIBLE", t, side, np.nan))
                    i = t + 1
                    continue
                ev = lookup_ev(axis, t, side, "td5")
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

        # ---- FP5 / FP17: Opportunity-native renewal decision ----
        # A Renewal event is a NEW decision epoch. E9 is NOT queried here; the
        # action value is the Opportunity EV of each side versus flat (0).
        remaining = days_remaining(axis.day_ord, e, trade.deadline_idx)
        H = model_horizon_for_remaining_days(remaining)
        ev_long = action_value(axis, e, H, +1)
        ev_short = action_value(axis, e, H, -1)
        ev_cur = ev_long if pos > 0 else ev_short
        ev_opp = ev_short if pos > 0 else ev_long

        if max(ev_cur, ev_opp) <= 0.0:
            # neither side has positive action value
            i = close_trade(trade, axis, nxt, axis.open[nxt], "renewal_exit",
                            trades, i)
            decisions.append(("EXIT", e, 0, max(ev_cur, ev_opp)))
            pos = 0
            trade = None
            continue
        if ev_opp > ev_cur:
            # REVERSE: close old + open new, new ATR and new 5-day lifetime
            new_side = -pos
            i = close_trade(trade, axis, nxt, axis.open[nxt], "reversal_close",
                            trades, i)
            decisions.append(("REVERSE", e, new_side, ev_opp))
            trade = open_trade(axis, e, new_side, axis.deadline_idx[e])
            pos = new_side
            i = nxt
            continue
        # HOLD (includes the exact-tie rule: equal positive EV -> keep current).
        # No synthetic transaction; the 5-day deadline is NOT reset (§35).
        trade.decision_idx = e
        trade.scan_from = nxt
        decisions.append(("HOLD", e, pos, ev_cur))
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
def build_e9_root_axis(state_df, split=None, test_mask=None):
    """FP3: E9 ROOT axis -- E9 on every canonical TEST ROOT Candidate bar.

    The frozen Direction chain is fitted EXACTLY ONCE from frozen TRAIN/VAL.
    The fitted router and the fitted E9 LongExpert/ShortExpert are then applied
    in ONE batch pass to the raw DTP9 state persisted by R8 (FP2).

    E9 is NOT required on non-Candidate bars: at a structural Renewal the
    policy is Opportunity-native (FP5) and never queries E9.
    """
    from research.liquidity_oracle_atlas.direction_null_baseline_v1 import (
        build_frozen_split)
    from research.liquidity_oracle_atlas.direction_gated_experts_v1 import (
        build_direction_expert_data, run_chain, orient_dtp9_router_side,
        router_confidence, predict_experts)
    from research.liquidity_oracle_atlas.structural_renewal_dataset_v1 import (
        DTP9, split_bounds)
    if split is None:
        split = build_frozen_split()
    ds = split["ds"]
    data = build_direction_expert_data(ds)
    chain = run_chain(data, ds, split["train_idx"], split["val_idx"],
                      split["test_idx"])
    _bump("direction_chain_fits")

    _t1, t2, end_t = split_bounds(split)
    if test_mask is None:
        dt_all = state_df["decision_time"].to_numpy("datetime64[ns]")
        test_mask = (dt_all >= t2) & (dt_all <= end_t)
    s = state_df[state_df["decision_time"].to_numpy("datetime64[ns]") <= end_t]
    s = s[s["decision_time"].to_numpy("datetime64[ns]") >= t2]
    root = s[s["candidate_at_decision"].to_numpy(bool)]
    idx = root["bar_index"].to_numpy(np.int64)

    # FP2 contract: exact DTP9 DataFrame representation the router was fitted on.
    X9_df = pd.DataFrame({c: root[c].to_numpy() for c in DTP9},
                         columns=list(DTP9))
    router = chain["models"]["router"]
    p_long = router.predict_proba(X9_df)[:, 1]
    router_pred = (p_long >= 0.5).astype(np.int8)

    # META10 uses ONLY the frozen orientation + frozen router-confidence helpers.
    router_long = router_pred >= 1
    X9o = orient_dtp9_router_side(
        X9_df.to_numpy(np.float32), router_long)
    ps = router_confidence(p_long, router_long)
    m10 = np.hstack([X9o, ps.reshape(-1, 1)])

    long_m, short_m = chain["models"]["e9"]
    fin_e9, p_correct = predict_experts(long_m, short_m, m10, router_pred)
    _bump("direction_batch_prediction_passes")
    out = pd.DataFrame({
        "symbol": root["symbol"].to_numpy(object),
        "decision_bar": idx,
        "e9_direction": np.where(np.asarray(fin_e9) == 1, "LONG", "SHORT"),
        "e9_side": np.where(np.asarray(fin_e9) == 1, 1, -1),
        "router_direction": np.where(router_pred == 1, "LONG", "SHORT"),
        "router_p_long": np.asarray(p_long, float),
        "router_p_side": np.asarray(ps, float)})
    return out, chain


def e9_axis_reproduction_gate(axis_e9, state_df=None, split=None):
    """FP4: exact equality on all frozen 13,773 overlapping Candidate rows.

    Also reports the root Candidate universe split into
      * overlap with the frozen 13,773 research population,
      * root Candidates OUTSIDE that population (deployment-only, no Teacher
        mapping required).
    The two governance gates are: all 13,773 present, and zero mismatch.
    """
    frozen = pd.read_parquet(
        "artifacts/entry_path_atlas_v1/e9_direction_state_v1.parquet",
        columns=["symbol", "candidate_decision_index", "e9_direction"])
    m = frozen.merge(axis_e9, how="left",
                     left_on=["symbol", "candidate_decision_index"],
                     right_on=["symbol", "decision_bar"])
    missing = int(m["e9_direction_y"].isna().sum())
    bad = int((m["e9_direction_x"].to_numpy(object)
               != m["e9_direction_y"].to_numpy(object)).sum())
    if missing or bad:
        raise RuntimeError(
            "STOP_R10_E9_AXIS_REPRODUCTION_MISMATCH "
            f"n_missing={missing} n_mismatch={bad}")

    rep = {"n_frozen_rows": int(len(m)), "n_missing": missing,
           "n_mismatch": bad, "ok": (missing == 0 and bad == 0)}
    if state_df is not None and split is not None:
        from research.liquidity_oracle_atlas.structural_renewal_dataset_v1 import (
            split_bounds)
        _t1, t2, end_t = split_bounds(split)
        dt = state_df["decision_time"].to_numpy("datetime64[ns]")
        root = state_df[(dt >= t2) & (dt <= end_t)
                        & state_df["candidate_at_decision"].to_numpy(bool)]
        rk = set(zip(root["symbol"].to_numpy(object),
                     root["bar_index"].to_numpy(np.int64)))
        fk = set(zip(frozen["symbol"].to_numpy(object),
                     frozen["candidate_decision_index"].to_numpy(np.int64)))
        rep["root_candidates_total"] = len(rk)
        rep["root_candidates_overlapping_frozen"] = len(fk & rk)
        rep["root_candidates_outside_frozen_research_population"] = len(rk - fk)
    return rep


def build_symbol_axes(state_df, pred_df, renewal_df, e9_df, split, symbols=None):
    """FIX13 step 10: construct all SymbolAxis objects (array lookup only)."""
    from research.liquidity_oracle_atlas.structural_renewal_dataset_v1 import (
        horizon_end_indices)
    # FP9: TEST is the CLOSED window [T2, COMMON_END].
    from research.liquidity_oracle_atlas.structural_renewal_dataset_v1 import (
        split_bounds)
    _t1, t2, end_t = split_bounds(split)
    axes = {}
    common_end_idx = {}
    for sym in (symbols or sorted(state_df["symbol"].unique())):
        s = state_df[state_df["symbol"] == sym].sort_values("bar_index")
        n = len(s)
        day = s["trading_day"].to_numpy()
        seg = s["segment"].to_numpy(np.int64)
        ends = horizon_end_indices(day, seg, n, (5,))
        fill = np.minimum(np.arange(n) + 1, n - 1)
        dt = s["decision_time"].to_numpy("datetime64[ns]")
        test_mask = (dt >= t2) & (dt <= end_t)
        # last canonical bar whose decision time is still inside TEST
        inside = np.flatnonzero(test_mask)
        common_end_idx[sym] = int(inside[-1]) if inside.size else n - 1

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
        # FP3: E9 is defined ONLY on canonical ROOT Candidate decision bars.
        e9_arr = np.zeros(n, np.int8)
        if e9_df is not None and len(e9_df):
            e9s = e9_df[e9_df["symbol"] == sym].set_index("decision_bar")["e9_side"]
            vals = e9s.reindex(np.arange(n)).to_numpy(float)
            e9_arr[np.isfinite(vals)] = vals[np.isfinite(vals)].astype(np.int8)

        el, fl = side_arrays("LONG", "event_idx_td5")
        es, fs = side_arrays("SHORT", "event_idx_td5")
        # FP7: bracket eligibility from the SAME R8 renewal axis.
        bel = renewal_df[(renewal_df["symbol"] == sym)
                         & (renewal_df["side"] == "LONG")].set_index(
            "decision_bar")["bracket_eligible"].reindex(np.arange(n))
        bes = renewal_df[(renewal_df["symbol"] == sym)
                         & (renewal_df["side"] == "SHORT")].set_index(
            "decision_bar")["bracket_eligible"].reindex(np.arange(n))
        # FP9: no position deadline may exceed the common evaluation end.
        deadline = np.minimum(ends[5][fill], int(common_end_idx.get(sym, n - 1)))
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
            test_mask=test_mask, e9_root_side=e9_arr,
            deadline_idx=deadline,
            day_ord=np.asarray(pd.factorize(day, sort=True)[0], np.int64),
            ev=ev, event_idx_long=el, event_idx_short=es,
            renewal_fill_long=fl, renewal_fill_short=fs,
            bracket_eligible_long=bel.to_numpy(bool),
            bracket_eligible_short=bes.to_numpy(bool))
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
    # FP12: counters are reset at the start of the real Formal run so the
    # performance gate measures THIS run only.
    reset_counters()

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
    # 7 + 8: E9 ROOT axis once + reproduction gate
    e9_df, _chain = build_e9_root_axis(state_df)
    gate = e9_axis_reproduction_gate(e9_df, state_df, split=None)
    # 9: renewal axis once
    renewal_df = pd.read_parquet(RENEWAL_AXIS_PARQUET)
    # 10
    from research.liquidity_oracle_atlas.direction_null_baseline_v1 import (
        build_frozen_split)
    split = build_frozen_split()
    axes = build_symbol_axes(state_df, pred_df, renewal_df, e9_df, split)
    # 11
    trades_by_policy = {}
    decision_by_policy = {}
    for p in POLICIES:
        per = {}
        dec = {}
        for sym, ax in axes.items():
            _bump("sequential_symbol_loops")
            per[sym], dec[sym] = simulate_symbol(p, ax)
        trades_by_policy[p] = per
        decision_by_policy[p] = dec
    # 12-13
    common_days = _common_days(axes, split)
    daily_by_policy = {}
    per_symbol_daily = {}
    for p in POLICIES:
        port, per_sym = daily_returns(trades_by_policy[p], axes, common_days)
        daily_by_policy[p] = port
        per_symbol_daily[p] = per_sym
    # 14
    full = block_bootstrap_delta(daily_by_policy[PRIMARY_POLICY],
                                 daily_by_policy[BASELINE_POLICY])
    gate_d = block_bootstrap_delta(daily_by_policy[GATE_POLICY],
                                   daily_by_policy[BASELINE_POLICY])
    renew = block_bootstrap_delta(daily_by_policy[PRIMARY_POLICY],
                                  daily_by_policy[GATE_POLICY])
    ident = policy_decomposition_identity(daily_by_policy[PRIMARY_POLICY],
                                          daily_by_policy[GATE_POLICY],
                                          daily_by_policy[BASELINE_POLICY])
    # 15
    verdict = formal_verdict(full, gate_d, renew)

    # FP14 strategy diagnostics, FP15 per-symbol + LOSO (cached, no resim)
    diag = strategy_diagnostics(trades_by_policy, axes, common_days,
                                daily_by_policy, full)
    per_sym_rows = per_symbol_deltas(per_symbol_daily, common_days)
    loso_rows = loso_deltas(per_symbol_daily, common_days)

    # FP12: performance gate BEFORE Formal evidence acceptance
    perf = dict(COUNTERS)
    perf_mismatch = {k: (perf.get(k), v) for k, v in FORMAL_PERF_EXPECTED.items()
                     if perf.get(k) != v}
    if perf_mismatch:
        raise RuntimeError(
            f"STOP_R10_FORMAL_PERFORMANCE_GATE {perf_mismatch}")

    result = {"verdict": verdict,
              "delta_full": {k: v for k, v in full.items() if k != "reps"},
              "delta_gate": {k: v for k, v in gate_d.items() if k != "reps"},
              "delta_renew": {k: v for k, v in renew.items() if k != "reps"},
              "decomposition_identity": ident,
              "e9_reproduction": gate,
              "strategy_diagnostics": diag,
              "scientific_status": SCIENTIFIC_STATUS,
              "performance": perf}

    # FP11: only AFTER the verdict is frozen may TEST labels be read, once.
    deciles = post_verdict_test_diagnostics(pred_df)
    result["post_verdict_test_ev_deciles"] = deciles

    if write_artifacts:
        # 16 write Formal evidence, 17 manifest LAST
        write_formal_evidence(result, trades_by_policy, decision_by_policy,
                              daily_by_policy, per_sym_rows, loso_rows, deciles,
                              full, authorized_review_sha, gate)
    if verbose:
        print(json.dumps({k: v for k, v in result.items()
                          if k != "post_verdict_test_ev_deciles"},
                         indent=2, default=str))
    return result


# --------------------------------------------------------------------------- #
# 9. FP12 frozen Formal performance budget                                      #
# --------------------------------------------------------------------------- #
FORMAL_PERF_EXPECTED = {
    "test_state_loads": 1,
    "test_prediction_loads": 1,
    "direction_chain_fits": 1,
    "direction_batch_prediction_passes": 1,
    "environment_recomputes": 0,
    "geometry_recomputes": 0,
    "model_predict_calls_inside_simulator": 0,
    "path_scans_inside_simulator": 0,
    "test_label_reads_during_strategy": 0,
    # 15 symbols x 3 policies
    "sequential_symbol_loops": 45,
}


# --------------------------------------------------------------------------- #
# 10. FP14 strategy diagnostics                                                 #
# --------------------------------------------------------------------------- #
def strategy_diagnostics(trades_by_policy, axes, common_days, daily_by_policy,
                         full):
    out = {}
    n_days = int(full.get("n_inference_days", len(common_days)))
    for p, per in trades_by_policy.items():
        trades = [t for sym in sorted(per) for t in per[sym]]
        dec = {}
        n = len(trades)
        rets = (np.array([trade_return(t) for t in trades], float) if n
                else np.zeros(0))
        hb = (np.array([t.holding_bars for t in trades], float) if n
              else np.zeros(0))
        # holding trading days per trade
        hd = []
        for t in trades:
            ax = axes.get(t.symbol)
            if ax is None:
                hd.append(0)
                continue
            days = pd.Index(ax.trading_day[t.fill_idx:t.exit_idx + 1]).unique()
            hd.append(len(days))
        hd = np.array(hd, float) if n else np.zeros(0)
        total_bars = sum(int(np.flatnonzero(ax.test_mask).size)
                         for ax in axes.values())
        port = daily_by_policy.get(p)
        cum = (np.cumsum(port.to_numpy(float)) if port is not None
               else np.zeros(0))
        peak = np.maximum.accumulate(cum) if len(cum) else np.zeros(0)
        dd = float(np.min(cum - peak)) if len(cum) else 0.0
        wins = rets[rets > 0]
        losses = rets[rets <= 0]
        avg_win = float(wins.mean()) if wins.size else float("nan")
        avg_loss = float((-losses).mean()) if losses.size else float("nan")
        out[p] = {
            "root_candidates_observed": dec.get("root", 0),
            "n_trades": int(n),
            "n_long_trades": int(sum(1 for t in trades if t.side > 0)),
            "n_short_trades": int(sum(1 for t in trades if t.side < 0)),
            "n_renewal_exits": int(sum(1 for t in trades
                                       if t.exit_reason == "renewal_exit")),
            "n_reversals": int(sum(1 for t in trades
                                   if t.exit_reason == "reversal_close")),
            "n_terminal": int(sum(1 for t in trades
                                  if t.exit_reason == "terminal_deadline")),
            "mean_holding_bars": float(hb.mean()) if n else 0.0,
            "median_holding_bars": float(np.median(hb)) if n else 0.0,
            "mean_holding_trading_days": float(hd.mean()) if n else 0.0,
            "exposure_fraction": (float(hb.sum() / total_bars)
                                  if total_bars else 0.0),
            "gross_total_R": float(rets.sum()) if n else 0.0,
            "gross_R_per_inference_day": (float(rets.sum() / n_days)
                                          if n_days else 0.0),
            "mean_trade_R": float(rets.mean()) if n else 0.0,
            "median_trade_R": float(np.median(rets)) if n else 0.0,
            "trade_win_rate": float((rets > 0).mean()) if n else 0.0,
            "average_win": avg_win, "average_loss": avg_loss,
            "realized_payoff_ratio": (float(avg_win / avg_loss)
                                      if avg_loss and avg_loss > 0 else np.nan),
            "max_drawdown_daily_R": dd,
        }
    return out


# --------------------------------------------------------------------------- #
# 11. FP15 per-symbol + LOSO (cached daily series, no resimulation)             #
# --------------------------------------------------------------------------- #
def per_symbol_deltas(per_symbol_daily, common_days):
    rows = []
    syms = sorted(per_symbol_daily[BASELINE_POLICY].keys())
    for sym in syms:
        p0 = per_symbol_daily[BASELINE_POLICY][sym].to_numpy(float)
        p1 = per_symbol_daily[GATE_POLICY][sym].to_numpy(float)
        p2 = per_symbol_daily[PRIMARY_POLICY][sym].to_numpy(float)
        rows.append({
            "symbol": sym,
            "delta_gate_point": float((p1 - p0).mean()),
            "delta_renew_point": float((p2 - p1).mean()),
            "delta_full_point": float((p2 - p0).mean()),
            "n_inference_days": int(complete_blocks(len(p0))[1]),
        })
    return rows


def loso_deltas(per_symbol_daily, common_days):
    """FP15: leave-one-symbol-out, averaging the REMAINING 14 daily series."""
    rows = []
    syms = sorted(per_symbol_daily[BASELINE_POLICY].keys())
    for drop in syms:
        keep = [s for s in syms if s != drop]
        ser = {}
        for p in POLICIES:
            mat = pd.DataFrame({s: per_symbol_daily[p][s].to_numpy(float)
                                for s in keep})
            ser[p] = mat.mean(axis=1)
        d_full = block_bootstrap_delta(ser[PRIMARY_POLICY],
                                       ser[BASELINE_POLICY], B=200)
        d_gate = block_bootstrap_delta(ser[GATE_POLICY],
                                       ser[BASELINE_POLICY], B=200)
        d_ren = block_bootstrap_delta(ser[PRIMARY_POLICY],
                                      ser[GATE_POLICY], B=200)
        rows.append({"removed_symbol": drop,
                     "delta_full_point": d_full["point"],
                     "delta_full_ci_low": d_full["ci_low"],
                     "delta_full_ci_high": d_full["ci_high"],
                     "delta_gate_point": d_gate["point"],
                     "delta_renew_point": d_ren["point"]})
    return rows


# --------------------------------------------------------------------------- #
# 12. FP11 post-verdict TEST Opportunity diagnostics (labels read ONCE)         #
# --------------------------------------------------------------------------- #
def post_verdict_test_diagnostics(pred_df, n_bins=10):
    """FP11: read labels_test EXACTLY ONCE, after the verdict is frozen.

    Descriptive only: it may not change the EV threshold, policy, verdict,
    features or models.
    """
    lab = pd.read_parquet(LABEL_PARQUETS["test"], columns=[
        "symbol", "decision_bar", "side", "horizon", "episode_return_atr",
        "sample_weight", "win"])
    _bump("post_verdict_test_label_reads")
    out = {}
    for H in HORIZONS:
        col = f"{H}_predicted_ev"
        if col not in pred_df.columns:
            continue
        m = lab[lab["horizon"] == H].merge(
            pred_df[["symbol", "decision_bar", "side", col]],
            on=["symbol", "decision_bar", "side"], how="inner")
        if len(m) == 0:
            continue
        w = m["sample_weight"].to_numpy(float)
        y = m["episode_return_atr"].to_numpy(float)
        pe = m[col].to_numpy(float)
        dec = pd.qcut(pd.Series(pe).rank(method="first"), n_bins,
                      labels=False)
        dec = np.asarray(dec, dtype=int)
        rows = []
        for d in range(n_bins):
            s = dec == d
            if not s.any():
                continue
            ww, yy = w[s], y[s]
            win = yy > 0
            p_act = float(np.average(win.astype(float), weights=ww))
            avg_win = (float(np.average(yy[win], weights=ww[win]))
                       if win.any() else np.nan)
            avg_loss = (float(np.average((-yy)[~win], weights=ww[~win]))
                        if (~win).any() else np.nan)
            mean_ret = float(np.average(yy, weights=ww))
            ident = p_act * avg_win - (1.0 - p_act) * avg_loss
            dev = abs(ident - mean_ret)
            if np.isfinite(dev) and dev > 1e-12:
                raise RuntimeError(
                    f"STOP_R10_TEST_DECILE_IDENTITY horizon={H} decile={d+1}")
            denom = avg_win + avg_loss
            rows.append({
                "horizon": H, "decile": d + 1, "n_rows": int(s.sum()),
                "mean_predicted_ev": float(np.average(pe[s], weights=ww)),
                "actual_win_rate": p_act, "actual_avg_win": avg_win,
                "actual_avg_loss": avg_loss,
                "actual_payoff_ratio": (float(avg_win / avg_loss)
                                        if avg_loss and avg_loss > 0 else np.nan),
                "actual_break_even_win_rate": (float(avg_loss / denom)
                                               if denom > 0 else np.nan),
                "actual_mean_return_atr": mean_ret,
                "actual_ev_identity_abs_dev": float(dev)})
        out[H] = rows
    return out


# --------------------------------------------------------------------------- #
# 13. FP13 Formal evidence writer (manifest LAST)                               #
# --------------------------------------------------------------------------- #
FORMAL_EVIDENCE_DIR = os.path.join("research", "liquidity_oracle_atlas",
                                   "evidence")
F_POLICY_SUMMARY = os.path.join(FORMAL_EVIDENCE_DIR,
                                "opportunity_value_renewal_v1_policy_summary.csv")
F_DAILY = os.path.join(FORMAL_EVIDENCE_DIR,
                       "opportunity_value_renewal_v1_daily_returns.csv")
F_PER_SYMBOL = os.path.join(FORMAL_EVIDENCE_DIR,
                            "opportunity_value_renewal_v1_per_symbol.csv")
F_TRADE_LEDGER = os.path.join(FORMAL_EVIDENCE_DIR,
                              "opportunity_value_renewal_v1_trade_ledger.csv")
F_DECISION_LEDGER = os.path.join(
    FORMAL_EVIDENCE_DIR, "opportunity_value_renewal_v1_decision_ledger.csv")
F_EV_DECILES = os.path.join(FORMAL_EVIDENCE_DIR,
                            "opportunity_value_renewal_v1_test_ev_deciles.csv")
F_SUMMARY = os.path.join(FORMAL_EVIDENCE_DIR,
                         "opportunity_value_renewal_v1_summary.json")
F_MANIFEST = os.path.join(FORMAL_EVIDENCE_DIR,
                          "opportunity_value_renewal_v1_manifest.json")


def _write_csv(path, df):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def write_formal_evidence(result, trades_by_policy, decision_by_policy,
                          daily_by_policy, per_sym_rows, loso_rows, deciles,
                          full, authorized_review_sha, gate):
    """FP13 step 16-17: write Formal evidence, then the manifest LAST."""
    os.makedirs(FORMAL_EVIDENCE_DIR, exist_ok=True)

    pol = pd.DataFrame([dict(policy=p, **v)
                        for p, v in result["strategy_diagnostics"].items()])
    _write_csv(F_POLICY_SUMMARY, pol)

    daily = pd.DataFrame({p: daily_by_policy[p].to_numpy(float)
                          for p in POLICIES})
    daily.insert(0, "trading_day",
                 list(daily_by_policy[BASELINE_POLICY].index))
    for a, b, name in ((PRIMARY_POLICY, BASELINE_POLICY, "delta_full"),
                       (GATE_POLICY, BASELINE_POLICY, "delta_gate"),
                       (PRIMARY_POLICY, GATE_POLICY, "delta_renew")):
        daily[name] = daily[a] - daily[b]
    _write_csv(F_DAILY, daily)

    _write_csv(F_PER_SYMBOL, pd.DataFrame(per_sym_rows + [
        dict(symbol="LOSO:" + r["removed_symbol"],
             delta_full_point=r["delta_full_point"],
             delta_gate_point=r["delta_gate_point"],
             delta_renew_point=r["delta_renew_point"],
             n_inference_days=0) for r in loso_rows]))

    led = []
    for p in POLICIES:
        for sym in sorted(trades_by_policy[p]):
            for t in trades_by_policy[p][sym]:
                led.append({"policy": p, "symbol": t.symbol,
                            "side": "LONG" if t.side > 0 else "SHORT",
                            "decision_idx": t.decision_idx,
                            "fill_idx": t.fill_idx, "exit_idx": t.exit_idx,
                            "entry_price": t.entry_price, "atr0": t.atr0,
                            "exit_price": t.exit_price,
                            "exit_reason": t.exit_reason,
                            "holding_bars": t.holding_bars,
                            "trade_return_atr": trade_return(t)})
    _write_csv(F_TRADE_LEDGER, pd.DataFrame(led))

    dled = []
    for p in POLICIES:
        for sym in sorted(decision_by_policy[p]):
            for d in decision_by_policy[p][sym]:
                dled.append({"policy": p, "symbol": sym, "action": d[0],
                             "bar": d[1], "side": d[2], "ev": d[3]})
    _write_csv(F_DECISION_LEDGER, pd.DataFrame(dled))

    drows = [r for hs in deciles.values() for r in hs]
    _write_csv(F_EV_DECILES, pd.DataFrame(drows))

    with open(F_SUMMARY, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # ---- manifest LAST ----
    art = {
        "opportunity_value_predictions_test_v1.parquet":
            sha256_file(TEST_PRED_PARQUET) if os.path.exists(TEST_PRED_PARQUET)
            else None,
        "e9_root_axis_v1.parquet":
            sha256_file(E9_ROOT_AXIS_PARQUET)
            if os.path.exists(E9_ROOT_AXIS_PARQUET) else None,
    }
    for p in (F_POLICY_SUMMARY, F_DAILY, F_PER_SYMBOL, F_TRADE_LEDGER,
              F_DECISION_LEDGER, F_EV_DECILES, F_SUMMARY):
        art[os.path.basename(p)] = sha256_file(p)
    manifest = {
        "task_id": TASK_ID,
        "authorized_review_sha": authorized_review_sha,
        "generator_code_sha": _git_head_sha(),
        "reviewed_parent_sha": authorized_review_sha,
        "scientific_status": SCIENTIFIC_STATUS,
        "verdict": result["verdict"],
        "e9_reproduction": gate,
        "decomposition_identity": result["decomposition_identity"],
        "performance_gates": {
            "expected": FORMAL_PERF_EXPECTED,
            "actual": {k: dict(COUNTERS).get(k) for k in FORMAL_PERF_EXPECTED},
            "mismatch": {k: (dict(COUNTERS).get(k), v)
                         for k, v in FORMAL_PERF_EXPECTED.items()
                         if dict(COUNTERS).get(k) != v},
            "pass": all(dict(COUNTERS).get(k) == v
                        for k, v in FORMAL_PERF_EXPECTED.items())},
        "artifact_sha256": art,
        "serialization_manifest_last": True,
    }
    with open(F_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


def _common_days(axes, split):
    import pandas as pd
    sets = None
    for ax in axes.values():
        days = pd.Index(ax.trading_day[ax.test_mask]).unique()
        sets = days if sets is None else sets.intersection(days)
    return np.asarray(sorted(sets))
