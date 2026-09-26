"""FUTURE-R13.9-V2 TARGET-POLICY ALIGNMENT + SEQUENTIAL PRODUCTION AUDIT.

Supersedes the prior R13.9 contract. The frozen TD5 training target
``episode_return_atr`` is RENEWAL-AWARE (exit at the structural-event OPEN when
executable, else at the TD5 terminal CLOSE). The true production question is a
fixed 5-day HOLD5 (exit at TD5 terminal CLOSE regardless of structure).

This experiment evaluates BOTH exit semantics side by side:

  * X0 EPISODE_EXIT  -- reproduces the frozen model target exactly (HARD parity A)
  * X1 TRUE_HOLD5    -- independent terminal-close return   (HARD parity B)

No model is trained or retuned. No PGM / EV_C / p_BE / RR / renewal-decision
model is used. The only gate is the frozen R13.8 causal-q80 ``trade20``.

This module is ARRAY-LOOKUP ONLY. It never calls the environment, geometry, a
model, or a path scanner.
"""

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.structural_renewal_dataset_v1 import (
    horizon_end_indices,
    sha256_file,
)
import research.liquidity_oracle_atlas.run_decomposed_v2_research as R

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
ART = os.path.join(PROJECT_ROOT, "artifacts", "decomposed_value_v2")
EVID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidence")

TASK_ID = "FUTURE-R13.9-V2-TARGET-POLICY-ALIGNMENT-AUDIT"
REVIEWED_PARENT_SHA = "e63d1e9120c4d5c875dbf21190409f8ab9e4b6b9"

DIRECTION_AXIS = os.path.join(
    ART, "a9_e9_train_oof_axis_v2.parquet")
FACTORIAL_OOF = os.path.join(
    ART, "direction_value_factorial_oof_v2.parquet")
STATE_PARQUET = R.ALLOWED_V1_STATE
RENEWAL_AXIS = os.path.join(
    PROJECT_ROOT, "artifacts", "decomposed_value_v1",
    "renewal_event_axis_v1.parquet")

EXPECTED_DIRECTION_SHA = (
    "026b20755ff625420d39167a18c98209c00e91fd5df92930d8e58306f751d6ac")
EXPECTED_FACTORIAL_SHA = (
    "70fa060f4f42b239f64ed28025461ae310516d22f7f7adcd41b18975cef81406")

EVAL_FOLDS = [1, 2, 3, 4]
BOOTSTRAP_BLOCK_DAYS = 5
BOOTSTRAP_B = 5000
BOOTSTRAP_SEED = 20260926

SIDE_MAP = {"LONG": 1.0, "SHORT": -1.0}


def _lookup(df, key):
    """Index lookup on a MultiIndex DataFrame; return row Series or None."""
    try:
        return df.loc[key]
    except KeyError:
        return None

# Exit reasons that close at the NEXT bar OPEN (renewal episode exit). These are
# treated as a partial bar: the exit bar itself contributes open[x]-close[x-1].
_NEXT_OPEN_EXIT_REASONS = ("STRUCTURAL_EPISODE_END",)


# --------------------------------------------------------------------------- #
# Frozen input validation
# --------------------------------------------------------------------------- #
def verify_frozen_inputs():
    """Hard check: R13.8 prediction artifacts must be byte-identical."""
    d_sha = sha256_file(DIRECTION_AXIS)
    f_sha = sha256_file(FACTORIAL_OOF)
    if d_sha != EXPECTED_DIRECTION_SHA:
        raise RuntimeError(
            f"STOP_R13_9_V2_FROZEN_INPUT_DRIFT direction_axis {d_sha}")
    if f_sha != EXPECTED_FACTORIAL_SHA:
        raise RuntimeError(
            f"STOP_R13_9_V2_FROZEN_INPUT_DRIFT factorial_oof {f_sha}")
    return {"direction_axis_sha256": d_sha, "factorial_oof_sha256": f_sha}


# --------------------------------------------------------------------------- #
# Pure arithmetic helpers (self-contained; do NOT import PGM/EV_C/win/payoff)
# --------------------------------------------------------------------------- #
def trade_return(side, entry_price, exit_price, atr0):
    return side * (exit_price - entry_price) / atr0


def bar_pnl(side, atr0, fill_idx, exit_idx, exit_reason, open_arr, close_arr):
    """Per-bar PnL allocation consistent with trade_return within 1e-10.

    Entry at open[fill_idx]. Exit semantics:
      * next-open exit (renewal episode): position closed at open[exit_idx];
        the exit bar contributes open[exit_idx]-close[exit_idx-1].
      * close exit (TD5 terminal): position closed at close[exit_idx];
        the exit bar contributes close[exit_idx]-close[exit_idx-1].
    """
    d = float(side)
    a = float(atr0)
    j0 = int(fill_idx)
    j1 = int(exit_idx)
    out = []
    prev_close = open_arr[j0]
    for j in range(j0, j1 + 1):
        if j == j1 and exit_reason in _NEXT_OPEN_EXIT_REASONS:
            px = open_arr[j]
            out.append(d * (px - prev_close) / a)
        elif j == j0:
            out.append(d * (close_arr[j] - open_arr[j0]) / a)
            prev_close = close_arr[j]
        else:
            out.append(d * (close_arr[j] - prev_close) / a)
            prev_close = close_arr[j]
    return np.asarray(out, dtype=float)


def complete_blocks(n_days, block=BOOTSTRAP_BLOCK_DAYS):
    n_full = int(n_days // block)
    return n_full, int(block * n_full)


def build_block_draws(n_full, b=BOOTSTRAP_B, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n_full, size=n_full) for _ in range(b)]


def block_bootstrap_mean(daily, draws):
    d = np.asarray(daily, dtype=float)
    n_days = len(d)
    n_full, n_inf = complete_blocks(n_days)
    if n_full == 0:
        return {"point": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "n_days": n_days,
                "n_inference_days": 0}
    x = d[:n_inf]
    blocks = [x[i * BOOTSTRAP_BLOCK_DAYS:(i + 1) * BOOTSTRAP_BLOCK_DAYS]
              for i in range(n_full)]
    reps = np.empty(len(draws), dtype=float)
    for k, pick in enumerate(draws):
        reps[k] = float(np.concatenate([blocks[p] for p in pick]).mean())
    return {
        "point": float(np.mean(d)),
        "ci_lo": float(np.quantile(reps, 0.025)),
        "ci_hi": float(np.quantile(reps, 0.975)),
        "n_days": int(n_days),
        "n_inference_days": int(n_inf),
    }


def block_bootstrap_delta(a, b, draws):
    da = np.asarray(a, dtype=float)
    db = np.asarray(b, dtype=float)
    return block_bootstrap_mean(da - db, draws)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
class SymbolState:
    """Per-symbol frozen-state view, indexed by bar_index (0..n-1)."""

    def __init__(self, symbol, df):
        df = df.sort_values("bar_index").reset_index(drop=True)
        assert (df["bar_index"].to_numpy() == np.arange(len(df))).all(), \
            f"{symbol} bar_index not 0..n-1"
        self.symbol = symbol
        self.n_bars = len(df)
        self.open = df["open"].to_numpy(float)
        self.close = df["close"].to_numpy(float)
        self.atr = df["atr"].to_numpy(float)
        self.segment = df["segment"].to_numpy(int)
        self.trading_day = df["trading_day"].to_numpy(object)
        self.decision_time = df["decision_time"].to_numpy(object)
        self.ends5 = horizon_end_indices(
            self.trading_day, self.segment, self.n_bars, (5,))[5]


def load_states():
    state = pd.read_parquet(STATE_PARQUET)
    out = {}
    for s, g in state.groupby("symbol"):
        out[s] = SymbolState(s, g)
    return out


def load_candidate_universe():
    """Return per (direction, value_arch) candidate DataFrames (folds 1-4)."""
    oo = pd.read_parquet(FACTORIAL_OOF)
    oo = oo[oo["fold"].isin(EVAL_FOLDS)].copy()
    out = {}
    for (d, va), g in oo.groupby(["direction", "value_arch"]):
        base = g.drop_duplicates(["symbol", "decision_bar"]).copy()
        base = base[["symbol", "decision_bar", "decision_time",
                     "trading_day", "fold", "selected_side", "score"]]
        p20 = (g[g["gate"] == "P"]
               .set_index(["symbol", "decision_bar"])["trade20"])
        idx = base.set_index(["symbol", "decision_bar"]).index
        base["gate_selected_P20"] = idx.map(p20).fillna(False).astype(bool)
        base["gate_selected_ALL"] = True
        base["direction"] = d
        base["value_arch"] = va
        out[(d, va)] = base.reset_index(drop=True)
    return out


def load_episode_lookup():
    """Frozen TD5 label, keyed by (symbol, decision_bar, side).

    Provides renewal_executable, renewal_fill_idx, end_idx,
    episode_exit_price, episode_return_atr.
    """
    lab = R.read_train_labels()
    lab = lab[lab["horizon"] == "td5"].copy()
    lab["decision_bar"] = lab["decision_bar"].astype(int)
    lab["side_int"] = lab["side"].map(SIDE_MAP)
    return lab.set_index(["symbol", "decision_bar", "side"])


def load_renewal_axis():
    ra = pd.read_parquet(RENEWAL_AXIS)
    ra["decision_bar"] = ra["decision_bar"].astype(int)
    ra["side_int"] = ra["side"].map(SIDE_MAP)
    return ra.set_index(["symbol", "decision_bar", "side"])


# --------------------------------------------------------------------------- #
# Policy matrix
# --------------------------------------------------------------------------- #
def policy_definitions():
    """Exactly 12 policies: 8 primary (A0) + 4 replication (A1)."""
    pols = []
    for direction in ("A9", "E9"):
        for value_arch in ("A0",):
            for gate in ("ALL", "P20"):
                for exit in ("EPISODE", "HOLD5"):
                    pols.append({
                        "policy": f"{direction}_{value_arch}_{gate}_{exit}",
                        "direction": direction, "value_arch": value_arch,
                        "gate": gate, "exit": exit, "primary": True,
                    })
        for value_arch in ("A1",):
            for gate in ("P20",):
                for exit in ("EPISODE", "HOLD5"):
                    pols.append({
                        "policy": f"{direction}_{value_arch}_{gate}_{exit}",
                        "direction": direction, "value_arch": value_arch,
                        "gate": gate, "exit": exit, "primary": False,
                    })
    return pols


# --------------------------------------------------------------------------- #
# Sequential simulator
# --------------------------------------------------------------------------- #
def _episode_exit(decision_bar, side_int, st, episode_lookup):
    t = int(decision_bar)
    fill = t + 1
    rec = _lookup(episode_lookup, (st.symbol, decision_bar,
                              "LONG" if side_int > 0 else "SHORT"))
    if rec is not None and bool(rec["renewal_executable"]):
        rf = int(rec["renewal_fill_idx"])
        return rf, float(st.open[rf]), "STRUCTURAL_EPISODE_END"
    terminal = int(st.ends5[fill])
    return terminal, float(st.close[terminal]), "TD5_TERMINAL"


def _hold5_exit(decision_bar, st):
    t = int(decision_bar)
    fill = t + 1
    terminal = int(st.ends5[fill])
    return terminal, float(st.close[terminal]), "TD5_TERMINAL"


def simulate_stream(states, cand, exit_mode, episode_lookup=None):
    """Run one-position-per-symbol sequential simulation for one policy.

    ``cand`` must carry columns: symbol, decision_bar, selected_side,
    gate_selected (bool), fold. Returns (trades, audit).
    """
    trades = []
    audit = []
    by_sym = {s: g for s, g in cand.groupby("symbol")}
    for sym, rows in by_sym.items():
        st = states[sym]
        rows = rows.sort_values(["decision_time", "decision_bar"])
        occupied_until = -1
        for r in rows.itertuples(index=False):
            t = int(r.decision_bar)
            sel = bool(r.gate_selected)
            if t <= occupied_until:
                audit.append({
                    "symbol": sym, "decision_bar": t, "policy": None,
                    "selected_side": r.selected_side,
                    "status": "BLOCKED_BY_OCCUPANCY",
                    "would_pass_gate": sel,
                })
                continue
            if not sel:
                audit.append({
                    "symbol": sym, "decision_bar": t, "policy": None,
                    "selected_side": r.selected_side,
                    "status": "SKIP_GATE", "would_pass_gate": False,
                })
                continue
            side_int = SIDE_MAP[r.selected_side]
            fill = t + 1
            if fill >= st.n_bars or st.segment[fill] != st.segment[t]:
                audit.append({
                    "symbol": sym, "decision_bar": t, "policy": None,
                    "selected_side": r.selected_side,
                    "status": "SKIP_NO_VALID_OPEN", "would_pass_gate": True,
                })
                continue
            if exit_mode == "EPISODE":
                exit_bar, exit_px, reason = _episode_exit(
                    t, side_int, st, episode_lookup)
            elif exit_mode == "HOLD5":
                exit_bar, exit_px, reason = _hold5_exit(t, st)
            else:
                raise RuntimeError("UNKNOWN_EXIT_MODE")
            entry_px = float(st.open[fill])
            atr0 = float(st.atr[t])
            ret = trade_return(side_int, entry_px, exit_px, atr0)
            trades.append({
                "symbol": sym, "decision_bar": t, "fold": int(r.fold),
                "side": int(side_int), "fill_idx": fill,
                "exit_idx": int(exit_bar), "entry_price": entry_px,
                "exit_price": exit_px, "atr0": atr0,
                "exit_mode": exit_mode, "exit_reason": reason,
                "return_atr": ret,
            })
            occupied_until = int(exit_bar)
    return trades, audit


# --------------------------------------------------------------------------- #
# Daily portfolio aggregation
# --------------------------------------------------------------------------- #
def daily_portfolio_returns(trades_by_symbol, states, common_days):
    day_pos = {d: i for i, d in enumerate(common_days)}
    per_symbol = {}
    for sym, trades in trades_by_symbol.items():
        st = states[sym]
        acc = np.zeros(len(common_days), dtype=float)
        for tr in trades:
            f = int(tr["fill_idx"])
            x = int(tr["exit_idx"])
            pnl = bar_pnl(tr["side"], tr["atr0"], f, x,
                         tr["exit_reason"], st.open, st.close)
            days = st.trading_day[f:x + 1]
            s = pd.Series(pnl, index=pd.Index(days))
            grp = s.groupby(level=0).sum()
            for day, val in grp.items():
                if day in day_pos:
                    acc[day_pos[day]] += val
        per_symbol[sym] = acc
    mat = pd.DataFrame(per_symbol)
    portfolio = mat.mean(axis=1).to_numpy(float)
    return portfolio, per_symbol


def build_common_days(states):
    days = set()
    for st in states.values():
        days.update(st.trading_day.tolist())
    return sorted(days)


# --------------------------------------------------------------------------- #
# HOLD5 evaluation-only outcome table (independent code path)
# --------------------------------------------------------------------------- #
def derive_hold5_table(states, universe):
    """For every (symbol, decision_bar, side) in the common universe compute
    TRUE_HOLD5 return at the TD5 terminal CLOSE. EVALUATION ONLY."""
    rows = []
    for (d, va), cand in universe.items():
        for r in cand.itertuples(index=False):
            sym = r.symbol
            st = states[sym]
            t = int(r.decision_bar)
            side_int = SIDE_MAP[r.selected_side]
            fill = t + 1
            if fill >= st.n_bars or st.segment[fill] != st.segment[t]:
                continue
            terminal = int(st.ends5[fill])
            entry_open = float(st.open[fill])
            terminal_close = float(st.close[terminal])
            atr0 = float(st.atr[t])
            ret = trade_return(side_int, entry_open, terminal_close, atr0)
            rows.append({
                "symbol": sym, "decision_bar": t,
                "side": r.selected_side, "entry_bar": fill,
                "terminal_bar": terminal, "entry_open": entry_open,
                "terminal_close": terminal_close, "atr0": atr0,
                "hold5_return_atr": ret,
            })
    df = pd.DataFrame(rows)
    # A (symbol, decision_bar, side) appears once per value_arch (A0/A1) but the
    # HOLD5 return depends only on state bars, so rows are identical -> dedupe.
    df = df.drop_duplicates(["symbol", "decision_bar", "side"]).reset_index(
        drop=True)
    return df.set_index(["symbol", "decision_bar", "side"])


# --------------------------------------------------------------------------- #
# Parity gates
# --------------------------------------------------------------------------- #
def parity_gate_episode(trades, episode_lookup):
    max_ret_err = 0.0
    max_px_err = 0.0
    for tr in trades:
        rec = _lookup(episode_lookup, (tr["symbol"], tr["decision_bar"],
                                  "LONG" if tr["side"] > 0 else "SHORT"))
        if rec is None:
            raise RuntimeError(
                f"STOP_R13_9_V2_EPISODE_PARITY missing label "
                f"{tr['symbol']} {tr['decision_bar']}")
        exp_ret = float(rec["episode_return_atr"])
        exp_px = float(rec["episode_exit_price"])
        max_ret_err = max(max_ret_err, abs(tr["return_atr"] - exp_ret))
        max_px_err = max(max_px_err, abs(tr["exit_price"] - exp_px))
    if max_ret_err > 1e-8 or max_px_err > 1e-10:
        raise RuntimeError(
            f"STOP_R13_9_V2_EPISODE_PARITY max_ret_err={max_ret_err} "
            f"max_px_err={max_px_err}")
    return {"max_return_error": max_ret_err, "max_exit_price_error": max_px_err}


def parity_gate_hold5(trades, hold5_table):
    max_err = 0.0
    for tr in trades:
        key = (tr["symbol"], tr["decision_bar"],
               "LONG" if tr["side"] > 0 else "SHORT")
        rec = _lookup(hold5_table,key)
        if rec is None:
            raise RuntimeError(
                f"STOP_R13_9_V2_HOLD5_PARITY missing hold5 "
                f"{tr['symbol']} {tr['decision_bar']}")
        exp = float(rec["hold5_return_atr"])
        max_err = max(max_err, abs(tr["return_atr"] - exp))
    if max_err > 1e-10:
        raise RuntimeError(
            f"STOP_R13_9_V2_HOLD5_PARITY max_err={max_err}")
    return {"max_return_error": max_err}


# --------------------------------------------------------------------------- #
# Phase B: outcome-free entry ledger + pre-eval lock
# --------------------------------------------------------------------------- #
def build_entry_ledger(universe, policies):
    rows = []
    for pol in policies:
        d, va = pol["direction"], pol["value_arch"]
        cand = universe[(d, va)].copy()
        gate_col = "gate_selected_ALL" if pol["gate"] == "ALL" \
            else "gate_selected_P20"
        for r in cand.itertuples(index=False):
            rows.append({
                "policy": pol["policy"],
                "symbol": r.symbol,
                "decision_bar": int(r.decision_bar),
                "decision_time": r.decision_time,
                "trading_day": r.trading_day,
                "fold": int(r.fold),
                "direction": d,
                "value_arch": va,
                "selected_side": r.selected_side,
                "gate": pol["gate"],
                "exit_mode": pol["exit"],
                "gate_score": float(r.score),
                "gate_selected": bool(getattr(r, gate_col)),
            })
    return pd.DataFrame(rows)


def policy_candidates(universe, pol):
    cand = universe[(pol["direction"], pol["value_arch"])].copy()
    gate_col = ("gate_selected_ALL" if pol["gate"] == "ALL"
                else "gate_selected_P20")
    cand["gate_selected"] = cand[gate_col].astype(bool)
    return cand


def _git_head_sha():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Phase B: outcome-free entry ledger + pre-eval lock
# --------------------------------------------------------------------------- #
def run_phase_b():
    shas = verify_frozen_inputs()
    universe = load_candidate_universe()
    policies = policy_definitions()
    ledger = build_entry_ledger(universe, policies)

    forbidden = {"episode_return_atr", "actual_return", "future_close",
                 "win", "loss", "pnl"}
    assert not (set(ledger.columns) & forbidden), \
        f"entry ledger leaked outcome columns: {set(ledger.columns) & forbidden}"

    ledger_path = os.path.join(ART, "r13_9_v2_entry_decisions.parquet")
    ledger.to_parquet(ledger_path)
    ledger_sha = sha256_file(ledger_path)

    lock = {
        "task_id": TASK_ID,
        "reviewed_parent_sha": REVIEWED_PARENT_SHA,
        "generated_by_commit": _git_head_sha(),
        "direction_axis_sha256": shas["direction_axis_sha256"],
        "factorial_oof_sha256": shas["factorial_oof_sha256"],
        "state_sha256": sha256_file(STATE_PARQUET),
        "renewal_axis_sha256": sha256_file(RENEWAL_AXIS),
        "train_labels_sha256": sha256_file(R.ALLOWED_V1_LABELS_TRAIN),
        "entry_decision_ledger_sha256": ledger_sha,
        "policy_count": len(policies),
        "policy_definitions": policies,
        "evaluation_folds": EVAL_FOLDS,
        "hold": "TD5",
        "entry": "next valid open",
        "exit": "terminal deadline close",
        "renewal": False,
        "PGM": False,
        "EV_C": False,
        "threshold_recompute": False,
        "PRIMARY_EXIT": "TRUE_HOLD5",
        "DIAGNOSTIC_EXIT": "EPISODE_EXIT",
        "A0_primary": True,
        "A1_replication_only": True,
        "A1_cannot_set_primary_verdict": True,
    }
    lock_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "evidence",
        "r13_9_v2_pre_eval_lock.json")
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "w") as f:
        json.dump(lock, f, indent=2)
    print("R13_9_V2_POLICY_LOCK_SHA", ledger_sha)
    print("entry ledger rows:", len(ledger))
    print("policies:", len(policies))
    return lock


# --------------------------------------------------------------------------- #
# Frozen renewal-axis consistency
# --------------------------------------------------------------------------- #
def _assert_renewal_axis_consistency(episode_lookup, renewal_axis):
    mism = 0
    checked = 0
    for key, rec in renewal_axis.iterrows():
        lab = _lookup(episode_lookup, key)
        if lab is None:
            continue
        checked += 1
        rf_axis = int(rec["renewal_fill_idx_td5"])
        rf_lab = int(lab["renewal_fill_idx"])
        if rf_axis != rf_lab:
            mism += 1
            continue
        renew_axis = rf_axis >= 0
        if renew_axis != bool(lab["renewal_executable"]):
            mism += 1
    if mism:
        raise RuntimeError(
            f"STOP_R13_9_V2_FROZEN_INPUT_DRIFT renewal_axis vs label "
            f"mismatches={mism}/{checked}")
    return {"checked": checked, "mismatches": mism}


# --------------------------------------------------------------------------- #
# Phase C: sequential simulation + BOTH hard parity gates
# --------------------------------------------------------------------------- #
def run_phase_c():
    verify_frozen_inputs()
    states = load_states()
    universe = load_candidate_universe()
    policies = policy_definitions()
    episode_lookup = load_episode_lookup()
    renewal_axis = load_renewal_axis()
    _assert_renewal_axis_consistency(episode_lookup, renewal_axis)
    common_days = build_common_days(states)

    hold5 = derive_hold5_table(states, universe)
    hold5_path = os.path.join(ART, "hold5_terminal_outcome_v1.parquet")
    hold5.reset_index().to_parquet(hold5_path)

    trade_rows = []
    audit_rows = []
    daily_by_policy = {}
    metrics = {}
    parity = {}

    for pol in policies:
        cand = policy_candidates(universe, pol)
        trades, audit = simulate_stream(
            states, cand, pol["exit"], episode_lookup)
        for tr in trades:
            tr["policy"] = pol["policy"]
            trade_rows.append(tr)
        for a in audit:
            a["policy"] = pol["policy"]
            audit_rows.append(a)

        # ---- HARD parity gate FIRST ----
        if pol["exit"] == "EPISODE":
            parity[pol["policy"]] = parity_gate_episode(trades, episode_lookup)
        else:
            parity[pol["policy"]] = parity_gate_hold5(trades, hold5)

        by_sym = {}
        for tr in trades:
            by_sym.setdefault(tr["symbol"], []).append(tr)
        daily, _ = daily_portfolio_returns(by_sym, states, common_days)
        daily_by_policy[pol["policy"]] = daily
        metrics[pol["policy"]] = summarize_policy(
            trades, audit, len(cand), int(cand["gate_selected"].sum()),
            daily, common_days)

    trades_df = pd.DataFrame(trade_rows)
    audit_df = pd.DataFrame(audit_rows)
    daily_df = pd.DataFrame(
        daily_by_policy, index=pd.Index(common_days, name="trading_day"))

    trades_df.to_parquet(os.path.join(ART, "r13_9_v2_trade_ledger.parquet"))
    audit_df.to_parquet(os.path.join(ART, "r13_9_v2_audit.parquet"))
    daily_df.to_parquet(
        os.path.join(ART, "r13_9_v2_daily_policy_returns_v1.parquet"))
    with open(os.path.join(ART, "r13_9_v2_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(EVID, "r13_9_v2_parity.json"), "w") as f:
        json.dump(parity, f, indent=2)

    max_epi_ret = max((p["max_return_error"] for k, p in parity.items()
                       if "EPISODE" in k), default=0.0)
    max_epi_px = max((p["max_exit_price_error"] for k, p in parity.items()
                      if "EPISODE" in k), default=0.0)
    max_h5 = max((p["max_return_error"] for k, p in parity.items()
                  if "HOLD5" in k), default=0.0)
    print("Episode parity max ret err", max_epi_ret, "max px err", max_epi_px)
    print("Hold5 parity max ret err", max_h5)
    print("trades:", len(trades_df), "audit:", len(audit_df))
    return {"parity": parity, "n_trades": len(trades_df)}


# --------------------------------------------------------------------------- #
# Phase D: bootstrap contrasts, diagnostics, evidence
# --------------------------------------------------------------------------- #
def _boot_abs(daily_df, draws):
    return {p: block_bootstrap_mean(
        daily_df[p].to_numpy(float), draws) for p in daily_df.columns}


def _contrast(daily_df, draws, name, a, b):
    return {name: block_bootstrap_delta(
        daily_df[a].to_numpy(float), daily_df[b].to_numpy(float), draws)}


def run_phase_d():
    daily_df = pd.read_parquet(
        os.path.join(ART, "r13_9_v2_daily_policy_returns_v1.parquet"))
    metrics = json.load(open(os.path.join(ART, "r13_9_v2_metrics.json")))
    audit_df = pd.read_parquet(os.path.join(ART, "r13_9_v2_audit.parquet"))
    trades_df = pd.read_parquet(
        os.path.join(ART, "r13_9_v2_trade_ledger.parquet"))
    hold5 = pd.read_parquet(
        os.path.join(ART, "hold5_terminal_outcome_v1.parquet"))
    hold5_idx = hold5.set_index(["symbol", "decision_bar", "side"])
    episode_lookup = load_episode_lookup()
    universe = load_candidate_universe()
    common_days = list(daily_df.index)

    draws = build_block_draws(len(common_days) // BOOTSTRAP_BLOCK_DAYS)

    abs_boot = _boot_abs(daily_df, draws)

    # ---- contrasts ----
    contrasts = {}
    for d in ("A9", "E9"):
        contrasts.update(_contrast(
            daily_df, draws, f"{d}_A0_P20_EPISODE_minus_ALL_EPISODE",
            f"{d}_A0_P20_EPISODE", f"{d}_A0_ALL_EPISODE"))
        contrasts.update(_contrast(
            daily_df, draws, f"{d}_A0_P20_HOLD5_minus_ALL_HOLD5",
            f"{d}_A0_P20_HOLD5", f"{d}_A0_ALL_HOLD5"))
    contrasts.update(_contrast(
        daily_df, draws, "E9_A0_ALL_HOLD5_minus_A9_A0_ALL_HOLD5",
        "E9_A0_ALL_HOLD5", "A9_A0_ALL_HOLD5"))
    contrasts.update(_contrast(
        daily_df, draws, "E9_A0_P20_HOLD5_minus_A9_A0_P20_HOLD5",
        "E9_A0_P20_HOLD5", "A9_A0_P20_HOLD5"))
    # A1 replication vs A0
    for d in ("A9", "E9"):
        for ex in ("EPISODE", "HOLD5"):
            contrasts.update(_contrast(
                daily_df, draws, f"{d}_A1_P20_{ex}_minus_A0_P20_{ex}",
                f"{d}_A1_P20_{ex}", f"{d}_A0_P20_{ex}"))

    # ---- classification per direction (A0) ----
    def classify(direction):
        eg = contrasts[f"{direction}_A0_P20_EPISODE_minus_ALL_EPISODE"]
        hg = contrasts[f"{direction}_A0_P20_HOLD5_minus_ALL_HOLD5"]
        h_abs = abs_boot[f"{direction}_A0_P20_HOLD5"]
        if (eg["ci_lo"] > 0 and hg["ci_lo"] > 0 and h_abs["ci_lo"] > 0):
            return "TARGET_AND_HOLD5_SUPPORTED"
        if (eg["point"] > 0 and hg["point"] > 0 and h_abs["point"] > 0):
            return "TARGET_SUPPORTED_HOLD5_PROMISING"
        if (eg["point"] > 0 and hg["point"] <= 0):
            return "TARGET_POLICY_MISMATCH"
        return "NO_SEQUENTIAL_P_EDGE"

    a9_status = classify("A9")
    e9_status = classify("E9")

    # DEV eligibility
    dev_candidates = []
    if a9_status in ("TARGET_AND_HOLD5_SUPPORTED",
                     "TARGET_SUPPORTED_HOLD5_PROMISING"):
        dev_candidates.append("A9_A0_P20")
    if e9_status in ("TARGET_AND_HOLD5_SUPPORTED",
                     "TARGET_SUPPORTED_HOLD5_PROMISING"):
        dev_candidates.append("E9_A0_P20")
    if not dev_candidates:
        dev_status = "NO_HOLD5_POLICY_ELIGIBLE"
    else:
        dev_status = "ELIGIBLE:" + ",".join(dev_candidates)

    # sequential direction verdict (HOLD5)
    dd = contrasts["E9_A0_P20_HOLD5_minus_A9_A0_P20_HOLD5"]
    if dd["ci_lo"] > 0:
        dir_status = "E9_HOLD5_SUPPORTED_OVER_A9"
    elif dd["ci_hi"] < 0:
        dir_status = "A9_HOLD5_SUPPORTED_OVER_E9"
    else:
        dir_status = "A9_E9_HOLD5_UNRESOLVED"

    # ---- target semantics diagnostic ----
    epi_vals, h5_vals = [], []
    for (d, va), cand in universe.items():
        for r in cand.itertuples(index=False):
            key = (r.symbol, int(r.decision_bar), r.selected_side)
            lab = _lookup(episode_lookup, key)
            h = _lookup(hold5_idx, key)
            if lab is None or h is None:
                continue
            epi_vals.append(float(lab["episode_return_atr"]))
            h5_vals.append(float(h["hold5_return_atr"]))
    epi_vals = np.asarray(epi_vals)
    h5_vals = np.asarray(h5_vals)
    delta = h5_vals - epi_vals
    tg = np.sign(epi_vals)
    th = np.sign(h5_vals)
    target_sem = {
        "n": int(len(epi_vals)),
        "mean_delta_hold5_minus_episode": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "corr_episode_hold5": float(np.corrcoef(epi_vals, h5_vals)[0, 1]),
        "sign_agreement": float(np.mean(tg == th)),
        "win_label_agreement": float(np.mean((epi_vals > 0) == (h5_vals > 0))),
        "episode_pos_hold5_neg": int(np.sum((epi_vals > 0) & (h5_vals <= 0))),
        "episode_neg_hold5_pos": int(np.sum((epi_vals <= 0) & (h5_vals > 0))),
    }

    # ---- candidate-level transfer diagnostic (P20 selected) ----
    transfer_rows = []
    for d in ("A9", "E9"):
        cand = universe[(d, "A0")].copy()
        sel = cand[cand["gate_selected_P20"]]
        epi_mean = []
        h5_mean = []
        for r in sel.itertuples(index=False):
            key = (r.symbol, int(r.decision_bar), r.selected_side)
            lab = _lookup(episode_lookup, key)
            h = _lookup(hold5_idx, key)
            if lab is None or h is None:
                continue
            epi_mean.append(float(lab["episode_return_atr"]))
            h5_mean.append(float(h["hold5_return_atr"]))
        transfer_rows.append({
            "direction": d, "policy": f"{d}_A0_P20",
            "n_selected": len(epi_mean),
            "mean_episode_return": float(np.mean(epi_mean)) if epi_mean else float("nan"),
            "mean_hold5_return": float(np.mean(h5_mean)) if h5_mean else float("nan"),
        })
    transfer_df = pd.DataFrame(transfer_rows)

    # ---- occupancy diagnostic (P20 A0, HOLD5 policy audit) ----
    occ_rows = []
    for d in ("A9", "E9"):
        pol = f"{d}_A0_P20_HOLD5"
        pa = audit_df[audit_df["policy"] == pol]
        pt = trades_df[trades_df["policy"] == pol]
        exec_keys = set(zip(pt["symbol"], pt["decision_bar"],
                            pt["side"].map({1: "LONG", -1: "SHORT"})))
        exec_epi, exec_h5, blk_epi, blk_h5 = [], [], [], []
        for r in pa.itertuples(index=False):
            key = (r.symbol, int(r.decision_bar), r.selected_side)
            lab = _lookup(episode_lookup, key)
            h = _lookup(hold5_idx, key)
            if lab is None or h is None:
                continue
            er = float(lab["episode_return_atr"])
            hr = float(h["hold5_return_atr"])
            if key in exec_keys:
                exec_epi.append(er); exec_h5.append(hr)
            elif r.status == "BLOCKED_BY_OCCUPANCY" and r.would_pass_gate:
                blk_epi.append(er); blk_h5.append(hr)
        occ_rows.append({
            "direction": d, "policy": pol,
            "n_executed": len(exec_epi),
            "executed_mean_episode": float(np.mean(exec_epi)) if exec_epi else float("nan"),
            "executed_mean_hold5": float(np.mean(exec_h5)) if exec_h5 else float("nan"),
            "n_blocked_selected": len(blk_epi),
            "blocked_mean_episode": float(np.mean(blk_epi)) if blk_epi else float("nan"),
            "blocked_mean_hold5": float(np.mean(blk_h5)) if blk_h5 else float("nan"),
        })
    occ_df = pd.DataFrame(occ_rows)

    # ---- policy summary + fold ----
    summary_rows = []
    for p, m in metrics.items():
        b = abs_boot[p]
        row = {"policy": p}
        row.update(m)
        row["daily_point"] = b["point"]
        row["daily_ci_lo"] = b["ci_lo"]
        row["daily_ci_hi"] = b["ci_hi"]
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    fold_rows = []
    for (p, grp) in trades_df.groupby("policy"):
        for fold, g in grp.groupby("fold"):
            ret = g["return_atr"].to_numpy(float)
            fold_rows.append({
                "policy": p, "fold": int(fold), "entries": len(g),
                "mean_trade_R": float(ret.mean()) if len(ret) else float("nan"),
                "win_rate": float(np.mean(ret > 0)) if len(ret) else float("nan"),
                "total_trade_R": float(ret.sum()),
            })
    fold_df = pd.DataFrame(fold_rows)

    contrasts_df = pd.DataFrame(
        [{"contrast": k, **v} for k, v in contrasts.items()])

    # ---- write evidence (per contract §39: evidence/ directory) ----
    os.makedirs(EVID, exist_ok=True)
    summary_df.to_csv(os.path.join(EVID, "r13_9_v2_policy_summary.csv"),
                      index=False)
    fold_df.to_csv(os.path.join(EVID, "r13_9_v2_policy_fold.csv"), index=False)
    contrasts_df.to_csv(os.path.join(EVID, "r13_9_v2_primary_contrasts.csv"),
                        index=False)
    replication_df = contrasts_df[
        contrasts_df["contrast"].str.contains("_A1_P20_")].copy()
    replication_df.to_csv(os.path.join(EVID, "r13_9_v2_replication.csv"),
                         index=False)
    transfer_df.to_csv(os.path.join(EVID, "r13_9_v2_candidate_transfer.csv"),
                       index=False)
    occ_df.to_csv(os.path.join(EVID, "r13_9_v2_occupancy.csv"), index=False)
    pd.DataFrame([target_sem]).to_csv(
        os.path.join(EVID, "r13_9_v2_target_semantics.csv"), index=False)

    manifest = {
        "task_id": TASK_ID,
        "reviewed_parent_sha": REVIEWED_PARENT_SHA,
        "generated_by_commit": _git_head_sha(),
        "model_fits": 0,
        "direction_fits": 0,
        "win_fits": 0,
        "payoff_fits": 0,
        "PGM_fits": 0,
        "prediction_refits": 0,
        "threshold_recomputes": 0,
        "environment_recomputes": 0,
        "geometry_recomputes": 0,
        "DEV_reads": 0,
        "old_TEST_reads": 0,
        "episode_parity_max_return_error": max(
            (p["max_return_error"] for k, p in
             {p: v for p, v in
              __import__("json").load(open(os.path.join(
                  EVID, "r13_9_v2_parity.json"))).items()
              if "EPISODE" in p}.items()), default=0.0),
        "hold5_parity_max_return_error": max(
            (p["max_return_error"] for k, p in
             __import__("json").load(open(os.path.join(
                  EVID, "r13_9_v2_parity.json"))).items()
              if "HOLD5" in p), default=0.0),
        "A9_target_policy_status": a9_status,
        "E9_target_policy_status": e9_status,
        "HOLD5_DIRECTION_STATUS": dir_status,
        "DEV_PRIMARY_POLICY_STATUS": dev_status,
        "DEV_PRIMARY_POLICY_CANDIDATES": dev_candidates,
        "target_semantics": target_sem,
    }
    with open(os.path.join(EVID, "r13_9_v2_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("A9 status:", a9_status, "| E9 status:", e9_status)
    print("HOLD5 direction:", dir_status)
    print("DEV status:", dev_status)
    print("target-semantics:", target_sem)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=TASK_ID)
    ap.add_argument("phase", choices=["b", "c", "d", "hold5", "all"])
    args = ap.parse_args()
    if args.phase == "b":
        run_phase_b()
    elif args.phase == "c":
        run_phase_c()
    elif args.phase == "d":
        run_phase_d()
    elif args.phase == "hold5":
        verify_frozen_inputs()
        states = load_states()
        universe = load_candidate_universe()
        hold5 = derive_hold5_table(states, universe)
        hold5.reset_index().to_parquet(
            os.path.join(ART, "hold5_terminal_outcome_v1.parquet"))
        print("hold5 table rows:", len(hold5))
    elif args.phase == "all":
        run_phase_b()
        run_phase_c()
        run_phase_d()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def summarize_policy(trades, audit, root_candidates, n_gate_selected,
                     daily, common_days):
    entries = len(trades)
    status_counts = {}
    blocked_selected = 0
    if audit:
        for a in audit:
            status_counts[a["status"]] = status_counts.get(a["status"], 0) + 1
            if (a["status"] == "BLOCKED_BY_OCCUPANCY"
                    and a["would_pass_gate"]):
                blocked_selected += 1
    returns = np.asarray([t["return_atr"] for t in trades], dtype=float)
    longs = sum(1 for t in trades if t["side"] > 0)
    shorts = entries - longs
    win_rate = float(np.mean(returns > 0)) if entries else float("nan")
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    avg_win = float(wins.mean()) if wins.size else float("nan")
    avg_loss = float(losses.mean()) if losses.size else float("nan")
    payoff_ratio = (abs(avg_win / avg_loss)
                    if (losses.size and avg_loss != 0) else float("nan"))
    mean_trade_R = float(returns.mean()) if entries else float("nan")
    total_trade_R = float(returns.sum())
    mean_daily_R = float(np.mean(daily)) if daily.size else float("nan")
    cum = np.cumsum(daily)
    peak = np.maximum.accumulate(cum)
    max_dd = float((cum - peak).min()) if cum.size else float("nan")
    holding = np.asarray([
        t["exit_idx"] - t["fill_idx"]
        + (0 if t["exit_reason"] in _NEXT_OPEN_EXIT_REASONS else 1)
        for t in trades], dtype=float)
    # episode structural-exit share
    epi = [t for t in trades if t["exit_reason"] in _NEXT_OPEN_EXIT_REASONS]
    pct_struct = (len(epi) / entries) if entries else float("nan")
    return {
        "root_candidates": int(root_candidates),
        "gate_selected": int(n_gate_selected),
        "entries": entries,
        "SKIP_GATE": status_counts.get("SKIP_GATE", 0),
        "BLOCKED_BY_OCCUPANCY": status_counts.get("BLOCKED_BY_OCCUPANCY", 0),
        "BLOCKED_SELECTED": blocked_selected,
        "SKIP_NO_VALID_OPEN": status_counts.get("SKIP_NO_VALID_OPEN", 0),
        "execution_rate": (entries / n_gate_selected)
        if n_gate_selected else float("nan"),
        "LONG": longs, "SHORT": shorts,
        "mean_trade_R": mean_trade_R, "total_trade_R": total_trade_R,
        "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss,
        "payoff_ratio": payoff_ratio,
        "mean_daily_R": mean_daily_R,
        "total_portfolio_R": float(cum[-1]) if cum.size else float("nan"),
        "max_drawdown_R": max_dd,
        "mean_holding_bars": (float(holding.mean())
                              if holding.size else float("nan")),
        "median_holding_bars": (float(np.median(holding))
                                if holding.size else float("nan")),
        "pct_structural_episode_exits": pct_struct,
        "active_symbol_days": entries,
    }


if __name__ == "__main__":
    main()
