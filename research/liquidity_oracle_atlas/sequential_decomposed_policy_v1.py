"""FUTURE-R10-M15-SEQUENTIAL-DECOMPOSED-POLICY-V1.

Five pre-registered sequential policies over ONE position per symbol:

    P0  E9_DIRECTION_HOLD5          baseline: enter every root Candidate on the
                                    E9 side, hold to the frozen 5-day terminal.
                                    No value gate, no renewal.
    PW  WIN_ONLY_GATE_HOLD5         P0 root + EV_W > 0 entry gate. No renewal.
    PR  PAYOFF_ONLY_GATE_HOLD5      P0 root + EV_R > 0 entry gate. No renewal.
    PC  COMBINED_GATE_HOLD5         P0 root + EV_C > 0 entry gate. No renewal.
    PN  COMBINED_GATE_RENEWAL       PC root + structural Renewal: at a renewal
                                    event compare EV_C(LONG) vs EV_C(SHORT) vs 0,
                                    HOLD / EXIT / REVERSE. E9 is NOT queried at
                                    renewal (Opportunity-native).

The simulator is an event-driven state machine performing ARRAY LOOKUP ONLY.
It never calls the environment, geometry, a LightGBM model, the Direction
chain (except the frozen E9 ROOT axis), or a path scanner.

The Formal TEST runner is fully implemented but BLOCKED by default: it is NOT
executed until explicitly authorized (see run_formal_opportunity_value_test).
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
    RENEWAL_AXIS_PARQUET,
    HORIZON_DAYS,
    SCIENTIFIC_STATUS,
    horizon_end_indices,
    sha256_file,
    split_bounds,
    DTP9,
)
from research.liquidity_oracle_atlas.build_decomposed_value_dataset_v1 import (
    DEC_ARTIFACT_DIR,
)
from research.liquidity_oracle_atlas import win_probability_model_v1 as R9A
from research.liquidity_oracle_atlas import payoff_ratio_model_v1 as R9B
from research.liquidity_oracle_atlas import decomposed_value_composer_v1 as R9C

TASK_ID = "FUTURE-R10-M15-SEQUENTIAL-DECOMPOSED-POLICY-V1"
BASE_SHA = "f3ca3a04317126e8f35afe7430d7aea202a9175a"

POLICIES = ("P0", "PW", "PR", "PC", "PN")
PRIMARY_POLICY = "PN"
BASELINE_POLICY = "P0"
GATE_POLICY = "PC"          # combined entry gate
WIN_POLICY = "PW"           # win-only mechanism
PAYOFF_POLICY = "PR"        # payoff-only mechanism

TEST_PRED_PARQUET = os.path.join(
    DEC_ARTIFACT_DIR, "decomposed_predictions_test_v1.parquet")
E9_ROOT_AXIS_PARQUET = os.path.join(DEC_ARTIFACT_DIR, "e9_root_axis_v1.parquet")
FORMAL_ARTIFACT_DIR = os.path.join(DEC_ARTIFACT_DIR, "sequential_test_v1")
PRETEST_SUMMARY_PATH = os.path.join(
    "research", "liquidity_oracle_atlas", "evidence",
    "decomposed_value_renewal_v1_pretest_summary.json")


def _formal_paths(art_root: Optional[str] = None) -> dict:
    """G21: resolve Formal artifact paths. When ``art_root`` is given, every
    Formal artifact (TEST predictions, E9 root axis, all CSV/JSON) is redirected
    under ``art_root`` so the canonical ``DEC_ARTIFACT_DIR`` / ``FORMAL_EVIDENCE_DIR``
    are never touched by a mocked run."""
    if art_root is None:
        return {
            "evidence_dir": FORMAL_EVIDENCE_DIR,
            "test_pred": TEST_PRED_PARQUET,
            "e9_root": E9_ROOT_AXIS_PARQUET,
        }
    return {
        "evidence_dir": os.path.join(art_root, "evidence"),
        "test_pred": os.path.join(art_root, os.path.basename(TEST_PRED_PARQUET)),
        "e9_root": os.path.join(art_root, os.path.basename(E9_ROOT_AXIS_PARQUET)),
    }

BOOTSTRAP_BLOCK_DAYS = 5
BOOTSTRAP_B = 5000
BOOTSTRAP_SEED = 20260924

VERDICTS = (
    "FULL_DECOMPOSED_RENEWAL_SUPPORTED",
    "COMBINED_ENTRY_GATE_ONLY_SUPPORTED",
    "DECOMPOSED_VALUE_POLICY_HARMFUL",
    "NO_IDENTIFIABLE_DECOMPOSED_VALUE_EDGE",
)

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
    "test_label_reads_during_strategy": 0,
    "post_verdict_test_label_reads": 0,
}

# F2: only genuine next-OPEN exits belong here.
# `terminal_no_valid_open` closes at the EVENT bar's CLOSE (axis.close[e]), not
# at the next bar's OPEN. Keeping it here made bar_pnl() mark the event bar to
# OPEN, so sum(bar_pnl) != trade_return and daily PnL was corrupted whenever
# that path occurred. It is a CLOSE exit, so the event bar is held.
_NEXT_OPEN_EXIT_REASONS = (
    "renewal_exit", "reversal_close")


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
# 1. Frozen helpers (pure arithmetic)                                           #
# --------------------------------------------------------------------------- #
def model_horizon_for_remaining_days(days_remaining):
    if days_remaining >= 5:
        return "td5"
    if days_remaining >= 3:
        return "td3"
    return "td1"


def days_remaining(day_ord, i, deadline_idx):
    return int(day_ord[deadline_idx] - day_ord[i] + 1)


@dataclass
class SymbolAxis:
    """Precomputed per-bar TEST axis for ONE symbol (array lookup only)."""
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
    e9_root_side: np.ndarray
    deadline_idx: np.ndarray
    day_ord: np.ndarray
    evw: dict = field(default_factory=dict)     # {(horizon, side): array}
    evr: dict = field(default_factory=dict)
    evc: dict = field(default_factory=dict)
    event_idx_long: Optional[np.ndarray] = None
    event_idx_short: Optional[np.ndarray] = None
    renewal_fill_long: Optional[np.ndarray] = None
    renewal_fill_short: Optional[np.ndarray] = None
    bracket_eligible_long: Optional[np.ndarray] = None
    bracket_eligible_short: Optional[np.ndarray] = None


def make_epoch(axis, decision_idx, side_sign):
    is_long = side_sign > 0
    favorable = axis.res_bottom[decision_idx] if is_long \
        else axis.sup_top[decision_idx]
    adverse = axis.sup_top[decision_idx] if is_long \
        else axis.res_bottom[decision_idx]
    return {"decision_idx": int(decision_idx), "favorable": float(favorable),
            "adverse": float(adverse)}


def _eligible(axis, decision_idx, side) -> bool:
    arr = (axis.bracket_eligible_long if side > 0
           else axis.bracket_eligible_short)
    if arr is None:
        raise RuntimeError("STOP_R10_BRACKET_AXIS_MISSING")
    return bool(arr[decision_idx])


def _lookup(arr_by_side, axis, decision_idx, side):
    arr = (arr_by_side["long"] if side > 0 else arr_by_side["short"])
    return float(arr[decision_idx])


def evc_side(axis, decision_idx, side, horizon):
    key = (horizon, side)
    if key not in axis.evc:
        return -np.inf
    v = float(axis.evc[key][decision_idx])
    if not (_eligible(axis, decision_idx, side) and np.isfinite(v)):
        return -np.inf
    return v


def evw_side(axis, decision_idx, side, horizon):
    key = (horizon, side)
    if key not in axis.evw:
        return -np.inf
    v = float(axis.evw[key][decision_idx])
    if not (_eligible(axis, decision_idx, side) and np.isfinite(v)):
        return -np.inf
    return v


def evr_side(axis, decision_idx, side, horizon):
    key = (horizon, side)
    if key not in axis.evr:
        return -np.inf
    v = float(axis.evr[key][decision_idx])
    if not (_eligible(axis, decision_idx, side) and np.isfinite(v)):
        return -np.inf
    return v


def lookup_precomputed_event(axis, decision_idx, side):
    arr = axis.event_idx_long if side > 0 else axis.event_idx_short
    if arr is None:
        raise RuntimeError("STOP_R10_RENEWAL_AXIS_MISSING")
    if decision_idx < 0 or decision_idx >= len(arr):
        return -1, -1
    return int(arr[decision_idx]), int(
        (axis.renewal_fill_long if side > 0 else axis.renewal_fill_short)
        [decision_idx])


# --------------------------------------------------------------------------- #
# 2. Trade ledger                                                              #
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
    scan_from: int = 0
    exit_idx: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    # F3: `decision_idx` is IMMUTABLE — it is always the decision that CREATED
    # this trade. `epoch_decision_idx` is the mutable renewal cursor: it starts
    # equal to `decision_idx` and advances to the latest renewal decision on HOLD.
    epoch_decision_idx: int = -1


def open_trade(axis, decision_idx, side, deadline_idx):
    fill = decision_idx + 1
    return Trade(symbol=axis.symbol, side=side, decision_idx=decision_idx,
                 fill_idx=fill, entry_price=float(axis.open[fill]),
                 atr0=float(axis.atr[decision_idx]), deadline_idx=deadline_idx,
                 scan_from=fill, epoch_decision_idx=decision_idx)


def close_trade(trade, axis, exit_idx, exit_price, reason, trades, i):
    trade.exit_idx = exit_idx
    trade.exit_price = float(exit_price)
    trade.exit_reason = reason
    # FG9: a next-OPEN exit does NOT hold the exit bar after its OPEN.
    if reason in _NEXT_OPEN_EXIT_REASONS:
        trade.holding_bars = max(int(exit_idx - trade.fill_idx), 0)
    else:
        trade.holding_bars = max(int(exit_idx - trade.fill_idx + 1), 0)
    trades.append(trade)
    return exit_idx + 1


def trade_return(trade):
    return trade.side * (trade.exit_price - trade.entry_price) / trade.atr0


# --------------------------------------------------------------------------- #
# 3. Bar-level PnL allocation                                                   #
# --------------------------------------------------------------------------- #
def bar_pnl(trade, axis):
    d = trade.side
    atr = trade.atr0
    j0 = trade.fill_idx
    j1 = trade.exit_idx
    out = []
    prev_close = axis.open[j0]
    for j in range(j0, j1 + 1):
        if j == j1 and trade.exit_reason in _NEXT_OPEN_EXIT_REASONS:
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
# 4. Simulator                                                                 #
# --------------------------------------------------------------------------- #
def _entry_ev(policy, axis, t, side):
    """Return the entry-gate EV for `policy`, or None for the no-gate baseline."""
    if policy == "P0":
        return None
    if policy == "PW":
        return evw_side(axis, t, side, "td5")
    if policy == "PR":
        return evr_side(axis, t, side, "td5")
    # PC and PN share the same combined-gate entry.
    return evc_side(axis, t, side, "td5")


def simulate_symbol(policy, axis, verbose=False):
    assert policy in POLICIES
    _bump("sequential_symbol_loops")
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
            side = int(axis.e9_root_side[t])
            ev = None
            if side == 0:
                decisions.append(("SKIP_NO_E9", t, side, np.nan))
                i = t + 1
                continue
            if policy != "P0":
                if not _eligible(axis, t, side):
                    decisions.append(("SKIP_INELIGIBLE", t, side, np.nan))
                    i = t + 1
                    continue
                ev = _entry_ev(policy, axis, t, side)
                if ev is None:
                    pass
                elif not (np.isfinite(ev) and ev > 0.0):
                    decisions.append(("SKIP", t, side, ev))
                    i = t + 1
                    continue
            fill = t + 1
            if fill >= axis.n_bars or axis.segment[fill] != axis.segment[t]:
                decisions.append(("ROOT_NO_VALID_OPEN", t, side, np.nan))
                i = t + 1
                continue
            deadline = axis.deadline_idx[t]
            trade = open_trade(axis, t, side, deadline)
            pos = side
            decisions.append(("ENTER", t, side, ev if ev is not None else np.nan))
            if policy in ("P0", "PW", "PR", "PC"):
                i = close_trade(trade, axis, min(deadline, axis.n_bars - 1),
                                axis.close[min(deadline, axis.n_bars - 1)],
                                "terminal_deadline", trades, i)
                pos = 0
                trade = None
                continue
            i = fill

        # ---- PN position management (renewal) ----
        # F3: renewal progression is driven by the epoch cursor, never by the
        # immutable entry decision.
        e, renewal_fill = lookup_precomputed_event(
            axis, trade.epoch_decision_idx, pos)
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

        remaining = days_remaining(axis.day_ord, e, trade.deadline_idx)
        H = model_horizon_for_remaining_days(remaining)
        evc_cur = evc_side(axis, e, pos, H)
        evc_opp = evc_side(axis, e, -pos, H)

        if max(evc_cur, evc_opp) <= 0.0:
            i = close_trade(trade, axis, nxt, axis.open[nxt], "renewal_exit",
                            trades, i)
            decisions.append(("EXIT", e, 0, max(evc_cur, evc_opp)))
            pos = 0
            trade = None
            continue
        if evc_opp > evc_cur:
            new_side = -pos
            i = close_trade(trade, axis, nxt, axis.open[nxt], "reversal_close",
                            trades, i)
            decisions.append(("REVERSE", e, new_side, evc_opp))
            trade = open_trade(axis, e, new_side, axis.deadline_idx[e])
            pos = new_side
            i = nxt
            continue
        # HOLD: exact tie (evc_opp == evc_cur > 0) also keeps current side.
        trade.epoch_decision_idx = e
        trade.scan_from = nxt
        decisions.append(("HOLD", e, pos, evc_cur))
        i = nxt
    return trades, decisions


# --------------------------------------------------------------------------- #
# 5. Daily aggregation                                                         #
# --------------------------------------------------------------------------- #
def daily_returns(trades_by_symbol, axis_by_symbol, common_days):
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
# 6. Block bootstrap                                                           #
# --------------------------------------------------------------------------- #
def complete_blocks(n_days, block=BOOTSTRAP_BLOCK_DAYS):
    n_full = int(n_days // block)
    return n_full, int(block * n_full)


def block_bootstrap_delta(a, b, block=BOOTSTRAP_BLOCK_DAYS,
                          B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED):
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
# 7. Verdict                                                                   #
# --------------------------------------------------------------------------- #
def policy_decomposition_identity(daily_full, daily_gate, daily_base):
    d_full = (np.asarray(daily_full, float) - np.asarray(daily_base, float))
    d_gate = (np.asarray(daily_gate, float) - np.asarray(daily_base, float))
    d_renew = (np.asarray(daily_full, float) - np.asarray(daily_gate, float))
    dev = float(np.max(np.abs(d_full - (d_gate + d_renew))))
    if dev > 1e-12:
        raise RuntimeError(f"STOP_R10_DECOMPOSITION_IDENTITY dev={dev}")
    return {"max_abs_dev": dev, "ok": dev <= 1e-12}


def formal_verdict(full, combo, renew):
    if full["ci_low"] > 0 and renew["ci_high"] >= 0:
        return "FULL_DECOMPOSED_RENEWAL_SUPPORTED"
    if combo["ci_low"] > 0 and (full["ci_low"] <= 0 or renew["ci_high"] < 0):
        return "COMBINED_ENTRY_GATE_ONLY_SUPPORTED"
    if full["ci_high"] < 0 and combo["ci_high"] < 0:
        return "DECOMPOSED_VALUE_POLICY_HARMFUL"
    return "NO_IDENTIFIABLE_DECOMPOSED_VALUE_EDGE"


# --------------------------------------------------------------------------- #
# 8. TEST-axis construction (behind authorization)                              #
# --------------------------------------------------------------------------- #
def build_e9_root_axis(state_df, split=None, test_mask=None):
    """FP3: E9 ROOT axis — frozen Direction chain fitted once, applied in batch."""
    from research.liquidity_oracle_atlas.direction_null_baseline_v1 import (
        build_frozen_split)
    from research.liquidity_oracle_atlas.direction_gated_experts_v1 import (
        build_direction_expert_data, run_chain, orient_dtp9_router_side,
        router_confidence, predict_experts)
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

    X9_df = pd.DataFrame({c: root[c].to_numpy() for c in DTP9},
                         columns=list(DTP9))
    router = chain["models"]["router"]
    p_long = router.predict_proba(X9_df)[:, 1]
    router_pred = (p_long >= 0.5).astype(np.int8)

    router_long = router_pred >= 1
    X9o = orient_dtp9_router_side(X9_df.to_numpy(np.float32), router_long)
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
    from research.liquidity_oracle_atlas.structural_renewal_dataset_v1 import (
        horizon_end_indices, split_bounds)
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
        inside = np.flatnonzero(test_mask)
        common_end_idx[sym] = int(inside[-1]) if inside.size else n - 1

        def side_arrays(side_name, col):
            r = renewal_df[(renewal_df["symbol"] == sym)
                           & (renewal_df["side"] == side_name)]
            r = r.set_index("decision_bar").sort_index()
            return (r[col].reindex(np.arange(n)).to_numpy(np.int64),
                    r["renewal_fill_idx_td5"].reindex(np.arange(n))
                    .to_numpy(np.int64))

        evw, evr, evc = {}, {}, {}
        for H in HORIZONS:
            for sd, nm in ((1, "LONG"), (-1, "SHORT")):
                for dst, src in ((evw, "ev_w"), (evr, "ev_r"), (evc, "ev_c")):
                    p = pred_df[(pred_df["symbol"] == sym)
                                & (pred_df["side"] == nm)]
                    p = p.set_index("decision_bar")[f"{H}_{src}"] \
                        .reindex(np.arange(n))
                    dst[(H, sd)] = p.to_numpy(float)
        e9_arr = np.zeros(n, np.int8)
        if e9_df is not None and len(e9_df):
            e9s = e9_df[e9_df["symbol"] == sym].set_index("decision_bar")[
                "e9_side"]
            vals = e9s.reindex(np.arange(n)).to_numpy(float)
            e9_arr[np.isfinite(vals)] = vals[np.isfinite(vals)].astype(np.int8)

        el, fl = side_arrays("LONG", "event_idx_td5")
        es, fs = side_arrays("SHORT", "event_idx_td5")
        bel = renewal_df[(renewal_df["symbol"] == sym)
                         & (renewal_df["side"] == "LONG")].set_index(
            "decision_bar")["bracket_eligible"].reindex(np.arange(n))
        bes = renewal_df[(renewal_df["symbol"] == sym)
                         & (renewal_df["side"] == "SHORT")].set_index(
            "decision_bar")["bracket_eligible"].reindex(np.arange(n))
        deadline = np.minimum(ends[5][fill],
                              int(common_end_idx.get(sym, n - 1)))
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
            evw=evw, evr=evr, evc=evc,
            event_idx_long=el, event_idx_short=es,
            renewal_fill_long=fl, renewal_fill_short=fs,
            bracket_eligible_long=bel.to_numpy(bool),
            bracket_eligible_short=bes.to_numpy(bool))
    return axes


# --------------------------------------------------------------------------- #
# 9. Unified TEST prediction (gated; NOT executed)                             #
# --------------------------------------------------------------------------- #
def predict_test(allow_test: bool = False, authorized_review_sha: Optional[str] = None,
                 write_artifacts: bool = True, art_root: Optional[str] = None):
    """Batch-predict Pwin/mu_W/mu_L and compose EV_W/EV_R/EV_C for TEST, one
    pass per horizon. Implemented but NOT executed until authorized."""
    if allow_test is not True:
        raise RuntimeError("STOP_R10_TEST_PREDICTION_NOT_AUTHORIZED")
    if not authorized_review_sha:
        raise RuntimeError("STOP_R10_AUTHORIZED_REVIEW_SHA_REQUIRED")
    head = _git_head_sha()
    if head != authorized_review_sha:
        raise RuntimeError(
            f"STOP_R10_GENERATOR_SHA_MISMATCH head={head} "
            f"authorized_review_sha={authorized_review_sha}")

    from research.liquidity_oracle_atlas.direction_null_baseline_v1 import (
        build_frozen_split)
    split = build_frozen_split()
    t2 = np.datetime64(split["cal"]["cuts"][1], "ns")
    end_t = np.datetime64(pd.Timestamp(split["cal"]["end"]).to_datetime64(), "ns")

    win_feats = R9A.load_win_features()
    pay_feats = R9B.load_payoff_features()
    state = pd.read_parquet(
        os.path.join(DEC_ARTIFACT_DIR, "state_v1.parquet"),
        columns=["bar_index", "decision_time"])
    dt = state["decision_time"].to_numpy("datetime64[ns]")
    # G11: closed TEST window [T2, COMMON_END]; no prediction row after end_t.
    in_window = (dt >= t2) & (dt <= end_t)
    test_bars = set(state["bar_index"].to_numpy(np.int64)[in_window])
    if in_window.any() and dt[in_window].max() > end_t:
        raise RuntimeError("STOP_R10_TEST_WINDOW_EXCEEDS_COMMON_END")
    mask = np.fromiter(
        (b in test_bars for b in win_feats["decision_bar"].to_numpy(np.int64)),
        dtype=bool, count=len(win_feats))
    win_sub = win_feats[mask].reset_index(drop=True)
    pay_sub = pay_feats[mask].reset_index(drop=True)
    Xw = win_sub[list(R9A.WIN33_COLS)].to_numpy(np.float32)
    Xp = pay_sub[list(R9B.PAY8_COLS)].to_numpy(np.float32)

    priors = R9C.compute_train_priors()
    out = win_sub[list(SIDE_KEY)].copy()
    for H in HORIZONS:
        p_hat = R9A.predict_win_probability(R9A.load_bundle(H), Xw)
        mu_w, mu_l, rr = R9B.predict_payoff(R9B.load_bundle(H), Xp)
        ev_w, ev_r, ev_c = R9C.compose_scores(
            p_hat, mu_w, mu_l,
            p_train_prior=priors[H]["p0"],
            mu_w_train_prior=priors[H]["muW0"],
            mu_l_train_prior=priors[H]["muL0"])
        out[f"{H}_p_win"] = p_hat
        out[f"{H}_mu_win"] = mu_w
        out[f"{H}_mu_loss"] = mu_l
        out[f"{H}_predicted_rr"] = rr
        out[f"{H}_ev_w"] = ev_w
        out[f"{H}_ev_r"] = ev_r
        out[f"{H}_ev_c"] = ev_c
    if write_artifacts:
        # G21: redirect TEST predictions into art_root when mocking.
        out.to_parquet(_formal_paths(art_root)["test_pred"], index=False)
    return out


# --------------------------------------------------------------------------- #
# 10. Formal TEST runner — implemented, NOT executed                           #
# --------------------------------------------------------------------------- #
# G14: the E9 frozen root-candidate universe has exactly this many rows; the
# reproduction gate must reproduce all of them (n_missing == 0, n_mismatch == 0)
# and the frozen universe size must match this value.
E9_FROZEN_UNIVERSE_ROWS = 13773

# F1: (tag, pretest key, model subdir, schema-SHA key, upstream-feature-SHA key)
_R9_BINDING_SPECS = (
    ("R9A", "r9a", "win", "win33_schema_sha256", "win_features_sha256"),
    ("R9B", "r9b", "payoff", "pay8_schema_sha256", "payoff_features_sha256"),
)


def _models_dir(sub: str, art_root: Optional[str] = None) -> str:
    """Frozen R9 model directory (``sub`` = "win" or "payoff").

    G21: when ``art_root`` is given the model tree is redirected there so a
    mocked run never reads or writes the canonical bundle.
    """
    if art_root is None:
        return os.path.join("artifacts", "decomposed_value_v1", "models", sub)
    return os.path.join(art_root, "models", sub)


def _r9_binding_fields(manifest: dict, schema_key: str, feature_key: str) -> dict:
    """F1: the manifest subset that MUST equal the frozen PRE-TEST manifest."""
    up = manifest.get("upstream", {}) or {}
    return {
        "task_id": manifest.get("task_id"),
        "generator_code_sha": manifest.get("generator_code_sha"),
        "feature_schema_sha256": manifest.get(schema_key),
        "upstream_state_sha256": up.get("state_sha256"),
        "upstream_feature_sha256": up.get(feature_key),
        "labels_train_sha256": up.get("labels_train_sha256"),
        "labels_val_sha256": up.get("labels_val_sha256"),
        "model_sha256": manifest.get("model_sha256"),
        "best_iteration": manifest.get("best_iteration"),
        "model_count": manifest.get("model_count"),
    }


def verify_r9_pretest_binding(pre: dict, art_root: Optional[str] = None) -> dict:
    """F1: three-layer binding of the frozen R9 model bundle.

        PRE-TEST evidence -> local manifest -> model bytes

    A locally self-consistent bundle (local manifest matching local files) that
    differs from the committed PRE-TEST evidence is REJECTED.
    """
    audit = {}
    for tag, key, sub, schema_key, feat_key in _R9_BINDING_SPECS:
        stop = f"STOP_R10_{tag}_PRETEST_BINDING_MISMATCH"
        pre_man = ((pre.get(key) or {}).get("model_manifest") or {})
        if not pre_man:
            raise RuntimeError(f"{stop} pretest_manifest_missing")
        mdir = _models_dir(sub, art_root)
        mp = os.path.join(mdir, "model_manifest.json")
        if not os.path.exists(mp):
            raise RuntimeError(f"STOP_R10_{tag}_MODEL_MANIFEST_MISSING")
        with open(mp) as f:
            loc_man = json.load(f)
        want = _r9_binding_fields(pre_man, schema_key, feat_key)
        got = _r9_binding_fields(loc_man, schema_key, feat_key)
        for fname, w in want.items():
            if got.get(fname) != w:
                raise RuntimeError(
                    f"{stop} field={fname} pretest={w!r} local={got.get(fname)!r}")
        # Third layer: actual model bytes must equal the PRE-TEST frozen map.
        frozen = pre_man.get("model_sha256") or {}
        for name, want_sha in frozen.items():
            p = os.path.join(mdir, name)
            if not os.path.exists(p):
                raise RuntimeError(f"{stop} model_missing {name}")
            got_sha = sha256_file(p)
            if got_sha != want_sha:
                raise RuntimeError(
                    f"{stop} model_bytes {name} pretest={want_sha} local={got_sha}")
        audit[tag] = {
            "model_dir": mdir,
            "fields_verified": sorted(want),
            "models_verified": {n: frozen[n] for n in sorted(frozen)},
            "model_count": len(frozen),
        }
    return audit


def run_formal_opportunity_value_test(allow_test: bool = False,
                                      authorized_review_sha: Optional[str] = None,
                                      write_artifacts: bool = True,
                                      verbose: bool = False,
                                      art_root: Optional[str] = None):
    """Complete Formal TEST call graph. Blocked by default; NOT executed.

    Hard gate (FG4): the one real TEST run MUST write evidence.
    """
    if allow_test is not True:
        raise RuntimeError("STOP_R10_TEST_NOT_AUTHORIZED")
    if not authorized_review_sha:
        raise RuntimeError("STOP_R10_AUTHORIZED_REVIEW_SHA_REQUIRED")
    if write_artifacts is not True:
        raise RuntimeError("STOP_R10_FORMAL_ARTIFACT_WRITE_REQUIRED")
    head = _git_head_sha()
    if head != authorized_review_sha:
        raise RuntimeError(
            f"STOP_R10_GENERATOR_SHA_MISMATCH head={head} "
            f"authorized_review_sha={authorized_review_sha}")
    reset_counters()

    # PRE-TEST evidence identity (lineage gate).
    ev = PRETEST_SUMMARY_PATH
    if not os.path.exists(ev):
        raise RuntimeError("STOP_R10_PRETEST_EVIDENCE_MISSING")
    with open(ev) as f:
        pre = json.load(f)

    # R8 / R9A / R9B / R9C upstream SHA verification.
    dec_manifest_path = os.path.join(DEC_ARTIFACT_DIR, "r8_manifest_v1.json")
    if not os.path.exists(dec_manifest_path):
        raise RuntimeError("STOP_R10_R8_MANIFEST_MISSING")
    with open(dec_manifest_path) as f:
        dec_manifest = json.load(f)
    pre_dec = pre.get("r8_manifest", {})
    # G12: every upstream R8 artifact MUST exist, then match exact SHA.
    for name, want in pre_dec.get("artifact_sha256", {}).items():
        p = os.path.join(DEC_ARTIFACT_DIR, name)
        if not os.path.exists(p):
            raise RuntimeError(f"STOP_R10_R8_ARTIFACT_MISSING {name}")
        if sha256_file(p) != want:
            raise RuntimeError(f"STOP_R10_R8_ARTIFACT_SHA_MISMATCH {name}")
    # F1: PRE-TEST evidence -> local manifest -> model bytes (three layers).
    # Runs BEFORE TEST state load / TEST prediction / direction-chain fit /
    # strategy simulation, so a locally self-consistent but PRE-TEST-different
    # bundle can never reach the simulation.
    binding_audit = verify_r9_pretest_binding(pre, art_root=art_root)

    # Frozen split (used by E9 reproduction gate + downstream diagnostics).
    from research.liquidity_oracle_atlas.direction_null_baseline_v1 import (
        build_frozen_split)
    split = build_frozen_split()

    # 5: TEST state once
    state_df = pd.read_parquet(STATE_PARQUET)
    _bump("test_state_loads")
    # 6: unified TEST predictions once
    pred_df = predict_test(allow_test=True,
                            authorized_review_sha=authorized_review_sha,
                            write_artifacts=write_artifacts,
                            art_root=art_root)
    _bump("test_prediction_loads")
    # 7 + 8: E9 ROOT axis once + reproduction gate (FG6). G14: gate BEFORE write.
    e9_df, _chain = build_e9_root_axis(state_df)
    gate = e9_axis_reproduction_gate(e9_df, state_df, split=split)
    if not (gate.get("n_missing") == 0
            and gate.get("n_mismatch") == 0
            and gate.get("n_frozen_rows") == E9_FROZEN_UNIVERSE_ROWS):
        raise RuntimeError(
            f"STOP_R10_E9_AXIS_REPRODUCTION_FAIL "
            f"missing={gate.get('n_missing')} "
            f"mismatch={gate.get('n_mismatch')} "
            f"frozen_rows={gate.get('n_frozen_rows')}")
    if write_artifacts:
        # G21: redirect E9 root axis into art_root when mocking.
        e9_path = _formal_paths(art_root)["e9_root"]
        e9_df.to_parquet(e9_path, index=False)
    # 9: renewal axis once
    renewal_df = pd.read_parquet(RENEWAL_AXIS_PARQUET)
    # 10: split already built above (used by the E9 reproduction gate).
    axes = build_symbol_axes(state_df, pred_df, renewal_df, e9_df, split)
    # 11
    trades_by_policy, decision_by_policy = {}, {}
    for p in POLICIES:
        per, dec = {}, {}
        for sym, ax in axes.items():
            per[sym], dec[sym] = simulate_symbol(p, ax)
        trades_by_policy[p] = per
        decision_by_policy[p] = dec
    # 12-13
    common_days = _common_days(axes, split)
    daily_by_policy, per_symbol_daily = {}, {}
    for p in POLICIES:
        port, per_sym = daily_returns(trades_by_policy[p], axes, common_days)
        daily_by_policy[p] = port
        per_symbol_daily[p] = per_sym
    # 14
    full = block_bootstrap_delta(daily_by_policy[PRIMARY_POLICY],
                                 daily_by_policy[BASELINE_POLICY])
    combo = block_bootstrap_delta(daily_by_policy[GATE_POLICY],
                                  daily_by_policy[BASELINE_POLICY])
    renew = block_bootstrap_delta(daily_by_policy[PRIMARY_POLICY],
                                  daily_by_policy[GATE_POLICY])
    d_win = block_bootstrap_delta(daily_by_policy[WIN_POLICY],
                                  daily_by_policy[BASELINE_POLICY])
    d_payoff = block_bootstrap_delta(daily_by_policy[PAYOFF_POLICY],
                                     daily_by_policy[BASELINE_POLICY])
    ident = policy_decomposition_identity(daily_by_policy[PRIMARY_POLICY],
                                          daily_by_policy[GATE_POLICY],
                                          daily_by_policy[BASELINE_POLICY])
    # 15
    verdict = formal_verdict(full, combo, renew)

    diag = strategy_diagnostics(trades_by_policy, axes, common_days,
                                daily_by_policy, full, decision_by_policy)
    per_sym_rows = per_symbol_deltas(per_symbol_daily, common_days)
    loso_rows = loso_deltas(per_symbol_daily, common_days)

    # FG13: performance gate BEFORE evidence acceptance.
    perf = dict(COUNTERS)
    perf_mismatch = {k: (perf.get(k), v) for k, v in FORMAL_PERF_EXPECTED.items()
                     if perf.get(k) != v}
    if perf_mismatch:
        raise RuntimeError(
            f"STOP_R10_FORMAL_PERFORMANCE_GATE {perf_mismatch}")

    result = {"verdict": verdict,
              "delta_full": {k: v for k, v in full.items() if k != "reps"},
              "delta_combo": {k: v for k, v in combo.items() if k != "reps"},
              "delta_renew": {k: v for k, v in renew.items() if k != "reps"},
              "delta_win": {k: v for k, v in d_win.items() if k != "reps"},
              "delta_payoff": {k: v for k, v in d_payoff.items() if k != "reps"},
              "decomposition_identity": ident,
              "e9_reproduction": gate,
              "strategy_diagnostics": diag,
              "scientific_status": SCIENTIFIC_STATUS,
              "performance": perf}

    # FG10: only AFTER verdict may TEST labels be read, exactly once.
    deciles = post_verdict_test_diagnostics(pred_df)
    result["post_verdict_test_diagnostics"] = deciles

    if write_artifacts:
        write_formal_evidence(result, trades_by_policy, decision_by_policy,
                              daily_by_policy, per_sym_rows, loso_rows, deciles,
                              full, authorized_review_sha, gate, art_root=art_root,
                              binding_audit=binding_audit)
    if verbose:
        print(json.dumps({k: v for k, v in result.items()
                          if k != "post_verdict_test_diagnostics"},
                         indent=2, default=str))
    return result


# --------------------------------------------------------------------------- #
# 11. Frozen Formal performance budget (FG13)                                   #
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
    # 15 symbols x 5 policies
    "sequential_symbol_loops": 75,
}


# --------------------------------------------------------------------------- #
# 12. Strategy diagnostics                                                     #
# --------------------------------------------------------------------------- #
def strategy_diagnostics(trades_by_policy, axes, common_days, daily_by_policy,
                         full, decision_by_policy=None):
    out = {}
    n_days = int(full.get("n_inference_days", len(common_days)))
    for p, per in trades_by_policy.items():
        trades = [t for sym in sorted(per) for t in per[sym]]
        dec = {}
        if decision_by_policy is not None:
            for sym in sorted(decision_by_policy.get(p, {})):
                for d in decision_by_policy[p][sym]:
                    dec[d[0]] = dec.get(d[0], 0) + 1
        n = len(trades)
        rets = (np.array([trade_return(t) for t in trades], float) if n
                else np.zeros(0))
        hb = (np.array([t.holding_bars for t in trades], float) if n
              else np.zeros(0))
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
            "root_candidates_observed": dec.get("SKIP_NO_E9", 0)
            + dec.get("SKIP_INELIGIBLE", 0) + dec.get("SKIP", 0)
            + dec.get("ROOT_NO_VALID_OPEN", 0) + dec.get("ENTER", 0),
            "root_entries": dec.get("ENTER", 0),
            "root_skips": dec.get("SKIP", 0),
            "root_skips_ineligible": dec.get("SKIP_INELIGIBLE", 0),
            "root_skips_no_e9": dec.get("SKIP_NO_E9", 0),
            "root_no_valid_open": dec.get("ROOT_NO_VALID_OPEN", 0),
            "holds": dec.get("HOLD", 0),
            "renewal_exits": dec.get("EXIT", 0),
            "reversals": dec.get("REVERSE", 0),
            "n_trades": int(n),
            "n_long_trades": int(sum(1 for t in trades if t.side > 0)),
            "n_short_trades": int(sum(1 for t in trades if t.side < 0)),
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
# 13. Per-symbol + LOSO (cached daily series, no resimulation)                  #
# --------------------------------------------------------------------------- #
def per_symbol_deltas(per_symbol_daily, common_days):
    rows = []
    syms = sorted(per_symbol_daily[BASELINE_POLICY].keys())
    for sym in syms:
        p0 = per_symbol_daily[BASELINE_POLICY][sym].to_numpy(float)
        p1 = per_symbol_daily[GATE_POLICY][sym].to_numpy(float)
        p2 = per_symbol_daily[PRIMARY_POLICY][sym].to_numpy(float)
        # G17: point estimates use ONLY the complete 5-day inference blocks,
        # exactly like the Primary bootstrap inference (not the terminal remainder).
        n_inf = int(complete_blocks(len(p0))[1])
        d0, d1, d2 = p0[:n_inf], p1[:n_inf], p2[:n_inf]
        rows.append({
            "symbol": sym,
            "delta_combo_point": float((d1 - d0).mean()),
            "delta_renew_point": float((d2 - d1).mean()),
            "delta_full_point": float((d2 - d0).mean()),
            "n_inference_days": n_inf,
        })
    return rows


def loso_deltas(per_symbol_daily, common_days):
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
                                       ser[BASELINE_POLICY], B=BOOTSTRAP_B)
        d_combo = block_bootstrap_delta(ser[GATE_POLICY],
                                        ser[BASELINE_POLICY], B=BOOTSTRAP_B)
        d_ren = block_bootstrap_delta(ser[PRIMARY_POLICY],
                                      ser[GATE_POLICY], B=BOOTSTRAP_B)
        rows.append({"removed_symbol": drop,
                     "delta_full_point": d_full["point"],
                     "delta_full_ci_low": d_full["ci_low"],
                     "delta_full_ci_high": d_full["ci_high"],
                     "delta_combo_point": d_combo["point"],
                     "delta_combo_ci_low": d_combo["ci_low"],
                     "delta_combo_ci_high": d_combo["ci_high"],
                     "delta_renew_point": d_ren["point"],
                     "delta_renew_ci_low": d_ren["ci_low"],
                     "delta_renew_ci_high": d_ren["ci_high"]})
    return rows


# --------------------------------------------------------------------------- #
# 14. Post-verdict TEST diagnostics (labels read ONCE)                         #
# --------------------------------------------------------------------------- #
def post_verdict_test_diagnostics(pred_df, n_bins=10):
    """FG10: read TEST labels exactly once, after the verdict is frozen.

    Descriptive only: Win deciles, Payoff (RR) deciles, and the 5x5 Pwin x RR
    grid. Must NOT change thresholds, models, features or the verdict.
    """
    from research.liquidity_oracle_atlas.build_decomposed_value_dataset_v1 import (
        LABEL_PARQUETS_D)
    lab = pd.read_parquet(LABEL_PARQUETS_D["test"], columns=[
        "symbol", "decision_bar", "side", "horizon", "episode_return_atr",
        "sample_weight", "win"])
    _bump("post_verdict_test_label_reads")
    out = {"pwin_deciles": {}, "rr_deciles": {}, "map_5x5": {}}
    for H in HORIZONS:
        pw = pred_df[["symbol", "decision_bar", "side", f"{H}_p_win",
                      f"{H}_mu_win", f"{H}_mu_loss"]]
        m = lab[lab["horizon"] == H].merge(
            pw, on=["symbol", "decision_bar", "side"], how="inner")
        if len(m) == 0:
            continue
        w = m["sample_weight"].to_numpy(float)
        y = m["episode_return_atr"].to_numpy(float)
        pwin = m[f"{H}_p_win"].to_numpy(float)
        rr = (m[f"{H}_mu_win"] / m[f"{H}_mu_loss"]).to_numpy(float)
        # 10 Pwin deciles
        out["pwin_deciles"][H] = _decile_table(pwin, y, w, "pwin")
        # 10 RR deciles
        out["rr_deciles"][H] = _decile_table(rr, y, w, "rr")
        # 5x5 grid
        out["map_5x5"][H] = _grid_5x5(pwin, rr, y, w)
    return out


def _decile_table(x, y, w, kind):
    dec = np.asarray(pd.qcut(pd.Series(x).rank(method="first"), 10,
                             labels=False), dtype=int)
    rows = []
    for d in range(10):
        s = dec == d
        if not s.any():
            continue
        ww, yy = w[s], y[s]
        win = yy > 0
        avg_win = (float(np.average(yy[win], weights=ww[win]))
                   if win.any() else np.nan)
        avg_loss = (float(np.average((-yy)[~win], weights=ww[~win]))
                    if (~win).any() else np.nan)
        p_act = float(np.average(win.astype(float), weights=ww))
        mean_ret = float(np.average(yy, weights=ww))
        ident = p_act * avg_win - (1.0 - p_act) * avg_loss
        dev = abs(ident - mean_ret)
        if np.isfinite(dev) and dev > 1e-12:
            raise RuntimeError(
                f"STOP_R10_TEST_{kind.upper()}_IDENTITY decile={d + 1} dev={dev}")
        denom = avg_win + avg_loss
        rows.append({
            "decile": d + 1, "n_rows": int(s.sum()),
            "mean_predicted": float(np.average(x[s], weights=ww)),
            "actual_win_rate": p_act, "actual_avg_win": avg_win,
            "actual_avg_loss": avg_loss,
            "actual_payoff_ratio": (float(avg_win / avg_loss)
                                    if avg_loss and avg_loss > 0 else np.nan),
            "actual_mean_return_atr": mean_ret,
            "actual_ev_identity_abs_dev": float(dev)})
    return rows


def _grid_5x5(pwin, rr, y, w):
    qp = np.asarray(pd.qcut(pd.Series(pwin).rank(method="first"), 5,
                            labels=False), dtype=int)
    qr = np.asarray(pd.qcut(pd.Series(rr).rank(method="first"), 5,
                            labels=False), dtype=int)
    cells = []
    for a in range(5):
        for b in range(5):
            m = (qp == a) & (qr == b)
            if not m.any():
                continue
            ww, yy = w[m], y[m]
            win = yy > 0
            avg_win = (float(np.average(yy[win], weights=ww[win]))
                       if win.any() else np.nan)
            avg_loss = (float(np.average((-yy)[~win], weights=ww[~win]))
                        if (~win).any() else np.nan)
            p_act = float(np.average(win.astype(float), weights=ww))
            cells.append({
                "pwin_quintile": a + 1, "rr_quintile": b + 1,
                "n_rows": int(m.sum()),
                "mean_predicted_pwin": float(np.average(pwin[m], weights=ww)),
                "mean_predicted_rr": float(np.average(rr[m], weights=ww)),
                "actual_win_rate": p_act, "actual_avg_win": avg_win,
                "actual_avg_loss": avg_loss,
                "actual_payoff_ratio": (float(avg_win / avg_loss)
                                        if avg_loss and avg_loss > 0
                                        else np.nan),
                "actual_mean_return_atr": float(np.average(yy, weights=ww))})
    return cells


# --------------------------------------------------------------------------- #
# 15. Formal evidence writer (manifest LAST)                                    #
# --------------------------------------------------------------------------- #
FORMAL_EVIDENCE_DIR = os.path.join("research", "liquidity_oracle_atlas",
                                    "evidence")
F_POLICY_SUMMARY = os.path.join(FORMAL_EVIDENCE_DIR,
                                "decomposed_value_policy_summary.csv")
F_DAILY = os.path.join(FORMAL_EVIDENCE_DIR, "decomposed_value_daily_returns.csv")
F_PER_SYMBOL = os.path.join(FORMAL_EVIDENCE_DIR, "decomposed_value_per_symbol.csv")
F_TRADE_LEDGER = os.path.join(FORMAL_EVIDENCE_DIR,
                              "decomposed_value_trade_ledger.csv")
F_DECISION_LEDGER = os.path.join(
    FORMAL_EVIDENCE_DIR, "decomposed_value_decision_ledger.csv")
F_TEST_DIAG = os.path.join(FORMAL_EVIDENCE_DIR,
                           "decomposed_value_test_diagnostics.csv")
F_SUMMARY = os.path.join(FORMAL_EVIDENCE_DIR, "decomposed_value_summary.json")
F_MANIFEST = os.path.join(FORMAL_EVIDENCE_DIR, "decomposed_value_manifest.json")


def _write_csv(path, df):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def _load_decision_ledger(trades_by_policy, decision_by_policy):
    led = []
    for p in POLICIES:
        for sym in sorted(decision_by_policy[p]):
            for d in decision_by_policy[p][sym]:
                led.append({"policy": p, "symbol": sym, "action": d[0],
                            "bar": d[1], "side": d[2], "ev": d[3]})
    return led


def write_formal_evidence(result, trades_by_policy, decision_by_policy,
                          daily_by_policy, per_sym_rows, loso_rows, deciles,
                          full, authorized_review_sha, gate, art_root=None,
                          binding_audit=None):
    # G21: redirect every Formal artifact into art_root when mocking.
    paths = _formal_paths(art_root)
    fdir = paths["evidence_dir"]
    test_pred = paths["test_pred"]
    e9_root = paths["e9_root"]
    os.makedirs(fdir, exist_ok=True)
    F = {
        "policy": os.path.join(fdir, "decomposed_value_policy_summary.csv"),
        "daily": os.path.join(fdir, "decomposed_value_daily_returns.csv"),
        "persymbol": os.path.join(fdir, "decomposed_value_per_symbol.csv"),
        "trade": os.path.join(fdir, "decomposed_value_trade_ledger.csv"),
        "decision": os.path.join(fdir, "decomposed_value_decision_ledger.csv"),
        "testdiag": os.path.join(fdir, "decomposed_value_test_diagnostics.csv"),
        "summary": os.path.join(fdir, "decomposed_value_summary.json"),
        "manifest": os.path.join(fdir, "decomposed_value_manifest.json"),
    }

    def _load_json(p):
        if not os.path.exists(p):
            return {}
        with open(p) as f:
            return json.load(f)

    pol = pd.DataFrame([dict(policy=p, **v)
                        for p, v in result["strategy_diagnostics"].items()])
    _write_csv(F["policy"], pol)

    daily = pd.DataFrame({p: daily_by_policy[p].to_numpy(float)
                          for p in POLICIES})
    daily.insert(0, "trading_day",
                 list(daily_by_policy[BASELINE_POLICY].index))
    for a, b, name in ((PRIMARY_POLICY, BASELINE_POLICY, "delta_full"),
                       (GATE_POLICY, BASELINE_POLICY, "delta_combo"),
                       (PRIMARY_POLICY, GATE_POLICY, "delta_renew"),
                       (WIN_POLICY, BASELINE_POLICY, "delta_win"),
                       (PAYOFF_POLICY, BASELINE_POLICY, "delta_payoff")):
        daily[name] = daily[a] - daily[b]
    _write_csv(F["daily"], daily)

    _write_csv(F["persymbol"], pd.DataFrame(per_sym_rows + [
        dict(symbol="LOSO:" + r["removed_symbol"],
             delta_combo_point=r["delta_combo_point"],
             delta_renew_point=r["delta_renew_point"],
             delta_full_point=r["delta_full_point"],
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
    _write_csv(F["trade"], pd.DataFrame(led))

    _write_csv(F["decision"],
               pd.DataFrame(_load_decision_ledger(
                   trades_by_policy, decision_by_policy)))

    drows = []
    for H, rows in deciles.get("pwin_deciles", {}).items():
        for r in rows:
            drows.append({"kind": "pwin", "horizon": H, **r})
    for H, rows in deciles.get("rr_deciles", {}).items():
        for r in rows:
            drows.append({"kind": "rr", "horizon": H, **r})
    for H, cells in deciles.get("map_5x5", {}).items():
        for c in cells:
            drows.append({"kind": "map_5x5", "horizon": H, **c})
    _write_csv(F["testdiag"], pd.DataFrame(drows))

    with open(F["summary"], "w") as f:
        json.dump(result, f, indent=2, default=str)

    # G20: full lineage binding.
    pretest_path = os.path.join("research", "liquidity_oracle_atlas", "evidence",
                                "decomposed_value_renewal_v1_pretest_summary.json")
    r8_man = _load_json(os.path.join(DEC_ARTIFACT_DIR, "r8_manifest_v1.json"))
    r9a_man = _load_json(os.path.join("artifacts", "decomposed_value_v1", "models",
                                      "win", "model_manifest.json"))
    r9b_man = _load_json(os.path.join("artifacts", "decomposed_value_v1", "models",
                                      "payoff", "model_manifest.json"))

    # G19: TEST-label governance — verified BEFORE any verdict/strategy output.
    test_label_governance = {
        "test_label_reads_during_fit": {
            "r9a": int(R9A.COUNTERS.get("test_label_reads_during_fit", 0)),
            "r9b": int(R9B.COUNTERS.get("test_label_reads_during_fit", 0)),
        },
        "test_label_reads_during_strategy":
            int(COUNTERS.get("test_label_reads_during_strategy", 0)),
        "post_verdict_test_label_reads":
            int(COUNTERS.get("post_verdict_test_label_reads", 0)),
    }
    if (test_label_governance["test_label_reads_during_fit"]["r9a"] != 0
            or test_label_governance["test_label_reads_during_fit"]["r9b"] != 0
            or test_label_governance["test_label_reads_during_strategy"] != 0
            or test_label_governance["post_verdict_test_label_reads"] != 1):
        raise RuntimeError("STOP_R10_TEST_LABEL_GOVERNANCE")

    art = {
        "decomposed_predictions_test_v1.parquet":
            sha256_file(test_pred) if os.path.exists(test_pred) else None,
        "e9_root_axis_v1.parquet":
            sha256_file(e9_root) if os.path.exists(e9_root) else None,
    }
    for p in (F["policy"], F["daily"], F["persymbol"], F["trade"],
              F["decision"], F["testdiag"], F["summary"]):
        art[os.path.basename(p)] = sha256_file(p)
    if any(v is None for v in art.values()):
        raise RuntimeError("STOP_R10_FORMAL_ARTIFACT_NULL_SHA")
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
        "test_label_governance": test_label_governance,
        "lineage": {
            "pretest_evidence_sha256":
                sha256_file(pretest_path) if os.path.exists(pretest_path) else None,
            "r8_generator_code_sha": r8_man.get("generator_code_sha"),
            "r8_artifact_sha256": r8_man.get("artifact_sha256"),
            "win33_schema_sha256": R9A.win33_schema_sha256(),
            "pay8_schema_sha256": R9B.pay8_schema_sha256(),
            "r9a_generator_code_sha": r9a_man.get("generator_code_sha"),
            "r9a_model_sha256": r9a_man.get("model_sha256"),
            "r9b_generator_code_sha": r9b_man.get("generator_code_sha"),
            "r9b_model_sha256": r9b_man.get("model_sha256"),
            "train_priors": R9C.compute_train_priors(),
            # F1: proof that local models are bound to the PRE-TEST frozen map.
            "r9_pretest_binding": binding_audit,
        },
        "artifact_sha256": art,
        "serialization_manifest_last": True,
    }
    if manifest["lineage"]["pretest_evidence_sha256"] is None:
        raise RuntimeError("STOP_R10_FORMAL_PRETEST_SHA_MISSING")
    with open(F["manifest"], "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


def _common_days(axes, split):
    sets = None
    for ax in axes.values():
        days = pd.Index(ax.trading_day[ax.test_mask]).unique()
        sets = days if sets is None else sets.intersection(days)
    return np.asarray(sorted(sets))


# --------------------------------------------------------------------------- #
# 16. PRE-TEST synthetic evidence (no real TEST)                               #
# --------------------------------------------------------------------------- #
def run_pretest_synthetic_evidence(verbose: bool = False):
    """Generate PRE-TEST evidence from synthetic axes (no real TEST, no labels).

    Produces the policy summary / ledgers / decomposition identity on a small
    deterministic synthetic dataset so the governance surfaces (decision ledger,
    holding bars, performance counters, decomposition identity) are exercised
    without reading TEST labels or training models.
    """
    reset_counters()
    axes = _synthetic_axes()
    trades_by_policy, decision_by_policy = {}, {}
    for p in POLICIES:
        per, dec = {}, {}
        for sym, ax in axes.items():
            per[sym], dec[sym] = simulate_symbol(p, ax)
        trades_by_policy[p] = per
        decision_by_policy[p] = dec
    common_days = np.arange(20)
    daily_by_policy, per_symbol_daily = {}, {}
    for p in POLICIES:
        port, per_sym = daily_returns(trades_by_policy[p], axes, common_days)
        daily_by_policy[p] = port
        per_symbol_daily[p] = per_sym
    full = block_bootstrap_delta(daily_by_policy[PRIMARY_POLICY],
                                 daily_by_policy[BASELINE_POLICY])
    combo = block_bootstrap_delta(daily_by_policy[GATE_POLICY],
                                  daily_by_policy[BASELINE_POLICY])
    renew = block_bootstrap_delta(daily_by_policy[PRIMARY_POLICY],
                                  daily_by_policy[GATE_POLICY])
    ident = policy_decomposition_identity(daily_by_policy[PRIMARY_POLICY],
                                          daily_by_policy[GATE_POLICY],
                                          daily_by_policy[BASELINE_POLICY])
    diag = strategy_diagnostics(trades_by_policy, axes, common_days,
                                daily_by_policy, full, decision_by_policy)
    perf = dict(COUNTERS)
    result = {
        "mode": "PRETEST_SYNTHETIC",
        "policies": list(POLICIES),
        "delta_full": {k: v for k, v in full.items() if k != "reps"},
        "delta_combo": {k: v for k, v in combo.items() if k != "reps"},
        "delta_renew": {k: v for k, v in renew.items() if k != "reps"},
        "decomposition_identity": ident,
        "strategy_diagnostics": diag,
        "performance": perf,
        "performance_gate_pass": all(perf.get(k) == v
                                     for k, v in FORMAL_PERF_EXPECTED.items()),
    }
    if verbose:
        print(json.dumps({k: v for k, v in result.items()
                          if k not in ("strategy_diagnostics",)}, indent=2,
                         default=str))
    return result, trades_by_policy, decision_by_policy


def _synthetic_axes():
    """Deterministic 2-symbol synthetic axis to exercise the simulator."""
    np.random.seed(0)
    axes = {}
    for sym in ("SYN_A", "SYN_B"):
        n = 60
        arr = lambda: np.zeros(n)
        day = np.repeat(np.arange(n // 3), 3)[:n]
        seg = np.ones(n, np.int64)
        decision_time = pd.date_range("2024-01-01", periods=n, freq="15min") \
            .to_numpy("datetime64[ns]")
        axis = SymbolAxis(
            symbol=sym, n_bars=n,
            bar_start_time=decision_time, decision_time=decision_time,
            trading_day=day, segment=seg,
            open=np.linspace(100, 110, n), high=np.linspace(101, 111, n),
            low=np.linspace(99, 109, n), close=np.linspace(100, 110, n),
            atr=np.full(n, 1.0), sup_top=np.full(n, 112.0),
            res_bottom=np.full(n, 98.0),
            candidate_at_decision=np.ones(n, bool),
            test_mask=np.ones(n, bool), e9_root_side=np.tile([1, -1], n // 2 + 1)[:n],
            deadline_idx=np.minimum(np.arange(n) + 15, n - 1),
            day_ord=np.asarray(pd.factorize(day, sort=True)[0], np.int64),
            event_idx_long=np.full(n, -1),
            event_idx_short=np.full(n, -1),
            renewal_fill_long=np.full(n, -1),
            renewal_fill_short=np.full(n, -1),
            bracket_eligible_long=np.ones(n, bool),
            bracket_eligible_short=np.ones(n, bool))
        # Give every (horizon, side) a constant positive EV so PW/PR/PC/PN
        # enter and (for PN) the renewal keeps current side (no event -> terminal).
        evw, evr, evc = {}, {}, {}
        for H in HORIZONS:
            for sd in (1, -1):
                evw[(H, sd)] = np.full(n, 0.05)
                evr[(H, sd)] = np.full(n, 0.05)
                evc[(H, sd)] = np.full(n, 0.10)
        axis.evw, axis.evr, axis.evc = evw, evr, evc
        axes[sym] = axis
    return axes


if __name__ == "__main__":
    t0 = time.time()
    res, _, _ = run_pretest_synthetic_evidence(verbose=True)
    print("runtime_sec", time.time() - t0)
