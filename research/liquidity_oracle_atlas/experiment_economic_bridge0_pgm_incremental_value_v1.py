"""
experiment_economic_bridge0_pgm_incremental_value_v1.py
=======================================================

ECONOMIC-BRIDGE-0A -- PGM Incremental Economic Value on Frozen Opportunity Universe

PURPOSE
-------
Answer ONE narrow question: in an already-frozen trading-opportunity universe
(R1-R4 regions, discovered historically across multiple development windows),
does adding the *current* DYNAMIC-PGM (MARKET-STATE / conditional-support) representation
increase action-selection economic value, compared with using the PGM's source
market state alone?

This experiment does NOT:
  * build a new backtester,
  * re-scan OHLC to manufacture reward labels,
  * re-design stop / target / RR / actions,
  * train any non-Linear model,
  * run RL,
  * tune thresholds / horizons,
  * prune symbols,
  * use TB3 to select ANY parameter.

It MAXIMALLY reuses the existing economic / execution / PGM stack:
  * reward table (SKIP / MARKET / LIMIT_RR3 / REASSESS_RR3)  -> run_v2_ml_recency_multi_action_v1
  * execution / purge / cost grid                          -> run_fixed_execution_* / run_enter_skip_selection_v1
  * Q fit / predict / policy                                -> run_v2_ml_recency_multi_action_v1 (frozen Linear contract)
  * PGM fit + deterministic conditional-support scores       -> experiment_dynamic_pgm1c_free_run_rollout_v1

EXPERIMENT SCOPE (hard-coded, never escalated)
------------------------------------------------
This is NOT a final-strategy OOS validation. The opportunity regions were
themselves discovered with development windows, so this experiment can only claim:

    "In a frozen opportunity universe, does PGM add action-selection value?"

It must NEVER emit PROFITABLE_STRATEGY / LIVE_READY / FINAL_STRATEGY.

LEAKAGE DESIGN (critical)
-------------------------
PGM Window A : fit on TB1  -> score TB2  (OOS for TB2)
PGM Window B : fit on TB1+TB2 -> score TB3 (OOS for TB3)
Economic map : CALIBRATION = TB2,  TEST = TB3.
TB1 in-sample PGM scores are NEVER used to train the economic mapper.
TB3 reward is NEVER used for feature / model / threshold / hyperparameter selection.

VERDICT PRE-REGISTRATION (only emitted by --formal, NOT run this round)
----------------------------------------------------------------------
If on TB3:  mean(STATE_PGM - STATE) > 0  AND  trading-day bootstrap 95% CI lower > 0
  -> PGM_INCREMENTAL_ECONOMIC_VALUE_SUPPORTED_ON_FROZEN_UNIVERSE
else
  -> PGM_INCREMENTAL_ECONOMIC_VALUE_NOT_SUPPORTED_ON_FROZEN_UNIVERSE

This is NOT a final-strategy-profitability verdict.
"""

from __future__ import annotations

import sys
import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

# Make the repository root importable when run as a script (python research/.../experiment_*.py)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

import research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 as ex0
import research.liquidity_oracle_atlas.run_v2_ml_recency_multi_action_v1 as v2
import research.liquidity_oracle_atlas.run_enter_skip_selection_v1 as ess
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1c_free_run_rollout_v1 as pgm
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1b_terminal_reset_closure_v1 as exp1b
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a2c_representation_control_v1 as rep


# ---------------------------------------------------------------------------
# Constants & governance
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "ECONOMIC-BRIDGE-0A -- PGM Incremental Economic Value on Frozen Opportunity Universe"
EXPERIMENT_SCOPE = "CONDITIONAL_INCREMENTAL_VALUE_ON_FROZEN_OPPORTUNITY_UNIVERSE"

# Verdicts this experiment is forbidden from ever emitting.
FORBIDDEN_VERDICTS = {"PROFITABLE_STRATEGY", "LIVE_READY", "FINAL_STRATEGY"}

ACTIONS = ["SKIP", "MARKET", "LIMIT_RR3", "REASSESS_RR3"]
TRADE_ACTIONS = ["MARKET", "LIMIT_RR3", "REASSESS_RR3"]
REWARD_COLS = ["reward_SKIP", "reward_MARKET", "reward_LIMIT_RR3", "reward_REASSESS_RR3"]

CALIB_BLOCK = "TB2"   # economic CALIBRATION
TEST_BLOCK = "TB3"    # economic TEST

# STATE view = current PGM raw numeric state (frozen lists only; no new features)
STATE_COLS: List[str] = list(dict.fromkeys(list(exp1b.OBS_NUM) + list(rep.MC_EXTRA)))

# PGM view = deterministic conditional-support scores from the already-fitted PGM (no MC rollout)
PGM_SCORE_COLS: List[str] = [
    "z_d_up_mu", "upper_p_active", "lower_p_active", "upper_mu_logv", "lower_mu_logv",
    "p_up_distance_negative", "p_down_distance_negative", "p_geometry_invalid",
    "p_upper_age_invalid", "p_lower_age_invalid", "p_any_age_invalid", "p_physical_invalid_approx",
]

COST_R_GRID = ess.COST_R_GRID
BOOTSTRAP_N = v2.BOOTSTRAP_N
SEED = v2.SEED
HARDENING_VERSION = v2.HARDENING_VERSION

REPO_ROOT = v2.REPO_ROOT
TRADES_PATH = REPO_ROOT / "research/analysis_results/execution_frontier_v1/execution_lag1_trades.parquet"
OUT_DIR = REPO_ROOT / "research/analysis_results/economic_bridge0a"

REUSE_MAP: Dict[str, str] = {
    "economic_environment": "run_fixed_execution_baseline_v1.load_env",
    "raw_bars": "run_fixed_execution_baseline_v1.load_env -> D['bars_by_sym']",
    "execution": "run_v2_ml_recency_multi_action_v1.build_multi_action_dataset (frozen next-open / no-cross-discontinuity / single-position)",
    "reward": "run_v2_ml_recency_multi_action_v1.build_multi_action_dataset -> reward_SKIP/MARKET/LIMIT_RR3/REASSESS_RR3",
    "purge": "run_v2_ml_recency_multi_action_v1 (reward_end_time = BAR END, hardening 'bar-index-purge-v2')",
    "cost_grid": "run_enter_skip_selection_v1.COST_R_GRID",
    "Q_fit": "run_v2_ml_recency_multi_action_v1.fit_action_q_models (Linear / Ridge alpha=1, no tuning)",
    "Q_predict": "run_v2_ml_recency_multi_action_v1.predict_action_q",
    "Q_policy": "run_v2_ml_recency_multi_action_v1.apply_action_policy (argmax over [0,Q_M,Q_L,Q_R], all<=0 -> SKIP)",
    "PGM_fit": "experiment_dynamic_pgm1c_free_run_rollout_v1.fit_samplers_for_window",
    "PGM_scores": "experiment_dynamic_pgm1c_free_run_rollout_v1.compute_state_conditional_support_probs + FittedTransitionSampler.analytic_conditional_support",
    "PGM_state_cols": "experiment_dynamic_pgm1b_terminal_reset_closure_v1.OBS_NUM + experiment_dynamic_pgm1a2c_representation_control_v1.MC_EXTRA",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return "UNKNOWN"


def state_matrix(df: pd.DataFrame, state_cols: List[str]) -> np.ndarray:
    """Build STATE feature matrix directly from frozen lists.

    Implements the frozen zt_* -> phi_* substitution exactly as the PGM sampler does:
    a missing zt_* column is filled from its phi_* counterpart; if neither exists, NaN.
    No interaction / polynomial / new feature engineering.
    """
    arrs = []
    for c in state_cols:
        if c in df.columns:
            arrs.append(df[c].to_numpy(dtype=np.float64))
        else:
            pc = c.replace("zt_", "phi_") if c.startswith("zt_") else None
            if pc is not None and pc in df.columns:
                arrs.append(df[pc].to_numpy(dtype=np.float64))
            else:
                arrs.append(np.full(len(df), np.nan, dtype=np.float64))
    return np.column_stack(arrs)


def load_economic() -> pd.DataFrame:
    D, master_by_sym, bars_by_sym = ex0.load_env()
    trades = pd.read_parquet(TRADES_PATH)
    economic, _ = v2.build_multi_action_dataset(D, master_by_sym, bars_by_sym, trades)
    return economic


def load_pgm_sample() -> pd.DataFrame:
    s = pd.read_parquet(pgm.SAMPLE_PATH)
    # one row per (symbol, bar_t); the sample carries no 'window' column -> dedupe defensively
    s = s.drop_duplicates(subset=["symbol", "bar_t"], keep="first").reset_index(drop=True)
    return s


def exact_join(economic: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """EXACT join on symbol + (signal_bar_index == bar_t). No asof / ffill / bfill.

    Returns the economic rows for CALIB/TEST blocks with sample columns merged and a
    `matched` boolean (True iff the bar matched a PGM sample row).
    """
    ec = economic[economic["block"].isin([CALIB_BLOCK, TEST_BLOCK])].copy()
    ec["bar_t"] = ec["signal_bar_index"].astype(int)
    # The PGM sample also carries a 'block' column -> drop it to avoid a _x/_y collision
    # with economic's block (which is the one we use downstream).
    s = sample.drop(columns=[c for c in ["block"] if c in sample.columns])
    j = ec.merge(
        s,
        on=["symbol", "bar_t"],
        how="left",
        validate="many_to_one",
        indicator="_merge",
    )
    j["matched"] = j["_merge"] == "both"
    return j


def reward_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[REWARD_COLS].to_numpy(dtype=np.float64)


def compute_pgm_scores(df: pd.DataFrame, fit_A: Dict[str, Any], fit_B: Dict[str, Any]) -> np.ndarray:
    """Score each row with the window-appropriate OOS fitted transition sampler.

    TB2 rows -> fit_A (trained on TB1).  TB3 rows -> fit_B (trained on TB1+TB2).
    """
    mcA = fit_A["trans_samplers"]["MC_STATE_CURREENCODING"]
    mcB = fit_B["trans_samplers"]["MC_STATE_CURREENCODING"]
    out = np.full((len(df), len(PGM_SCORE_COLS)), np.nan, dtype=np.float64)
    for block, mc in [(CALIB_BLOCK, mcA), (TEST_BLOCK, mcB)]:
        m = (df["block"] == block).to_numpy()
        if not m.any():
            continue
        sub = df[m].reset_index(drop=True)
        moms = mc.analytic_conditional_support(sub)
        cs = pgm.compute_state_conditional_support_probs(sub, mc)
        arr = np.column_stack([
            np.asarray(moms["z_d_up_mu"], np.float64),
            np.asarray(moms["upper_p_active"], np.float64),
            np.asarray(moms["lower_p_active"], np.float64),
            np.asarray(moms["upper_mu_logv"], np.float64),
            np.asarray(moms["lower_mu_logv"], np.float64),
            np.asarray(cs["p_up_distance_negative"], np.float64),
            np.asarray(cs["p_down_distance_negative"], np.float64),
            np.asarray(cs["p_geometry_invalid"], np.float64),
            np.asarray(cs["p_upper_age_invalid"], np.float64),
            np.asarray(cs["p_lower_age_invalid"], np.float64),
            np.asarray(cs["p_any_age_invalid"], np.float64),
            np.asarray(cs["p_physical_invalid_approx"], np.float64),
        ])
        out[m] = arr
    return out


def train_only_impute(X_tr: np.ndarray, X_te: np.ndarray):
    med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    X_tr2 = np.where(np.isnan(X_tr), med, X_tr)
    X_te2 = np.where(np.isnan(X_te), med, X_te)
    return np.asarray(X_tr2, np.float64), np.asarray(X_te2, np.float64), med


def fit_q(X_tr: np.ndarray, R_tr: np.ndarray, feature_names: List[str]):
    rewards = {a: R_tr[:, ACTIONS.index(a)] for a in TRADE_ACTIONS}
    return v2.fit_action_q_models("Linear", X_tr, rewards, np.ones(len(X_tr)), feature_names)


def policy_rewards(X_te: np.ndarray, models: Dict[str, Any], R_te: np.ndarray):
    q = v2.predict_action_q(models, X_te)
    a = v2.apply_action_policy(q)
    chosen = R_te[np.arange(len(a)), a]
    return a, chosen


def _cost_stress(chosen: np.ndarray, non_skip: np.ndarray):
    out = []
    for c in COST_R_GRID:
        net = chosen - c * non_skip.astype(float)
        out.append(dict(
            cost_R=c,
            net_EV_per_signal_R=float(np.mean(net)),
            net_EV_per_trade_R=float(np.mean(net[non_skip])) if non_skip.any() else 0.0,
            total_net_R=float(np.sum(net)),
        ))
    return out


def route_metrics(chosen: np.ndarray, action: np.ndarray, R_te: np.ndarray, trading_day: np.ndarray) -> Dict[str, Any]:
    n = len(chosen)
    non_skip = action != 0
    n_ns = int(non_skip.sum())
    trade_rate = n_ns / n if n else 0.0
    gross_ev_ps = float(np.mean(chosen)) if n else 0.0
    gross_ev_pt = float(np.mean(chosen[non_skip])) if n_ns else 0.0
    total_R = float(np.sum(chosen))
    wins = chosen[non_skip & (chosen > 0)]
    losses = chosen[non_skip & (chosen < 0)]
    win_rate = len(wins) / n_ns if n_ns else 0.0
    mean_win = float(np.mean(wins)) if len(wins) else 0.0
    mean_loss = float(np.mean(losses)) if len(losses) else 0.0
    payoff = mean_win / abs(mean_loss) if len(losses) else float("nan")
    pf = float(np.sum(wins) / abs(np.sum(losses))) if len(losses) and np.sum(losses) != 0 else float("nan")
    cum = np.cumsum(chosen)
    max_dd = float(np.min(cum - np.maximum.accumulate(cum)))
    share = {a: int((action == i).sum()) for i, a in enumerate(ACTIONS)}
    cost_stress = _cost_stress(chosen, non_skip)
    break_even = gross_ev_ps / trade_rate if trade_rate > 0 else float("nan")
    return dict(
        n_signals=n, n_non_skip=n_ns, trade_rate=trade_rate,
        gross_EV_per_signal_R=gross_ev_ps, gross_EV_per_trade_R=gross_ev_pt, total_R=total_R,
        win_rate=win_rate, mean_win_R=mean_win, mean_loss_R=mean_loss,
        payoff_ratio=payoff, profit_factor=pf, max_drawdown_R=max_dd,
        action_SHARE=share, cost_stress=cost_stress, break_even_cost_R=break_even,
    )


def incremental_bootstrap(r_spg: np.ndarray, r_state: np.ndarray,
                          trading_day: np.ndarray, n: int = BOOTSTRAP_N, seed: int = SEED) -> Dict[str, float]:
    delta = r_spg - r_state
    days = np.unique(trading_day)
    rng = np.random.default_rng(seed)
    day_idx = {d: np.where(trading_day == d)[0] for d in days}
    means = np.empty(n)
    for b in range(n):
        sel = rng.choice(days, size=len(days), replace=True)
        idx = np.concatenate([day_idx[d] for d in sel])
        means[b] = delta[idx].mean()
    return dict(
        mean_delta_R=float(delta.mean()),
        ci_lo=float(np.percentile(means, 2.5)),
        ci_hi=float(np.percentile(means, 97.5)),
        p_delta_positive=float((means > 0).mean()),
    )


def symbol_breadth(routes: Dict[str, Any], symbol_test: np.ndarray) -> Dict[str, Any]:
    syms = np.unique(symbol_test)
    rows = []
    for s in syms:
        m = symbol_test == s
        r_state = routes["STATE"][1][m]
        r_pgm = routes["PGM"][1][m]
        r_spg = routes["STATE_PGM"][1][m]
        ev_s = float(r_state.mean()) if m.any() else 0.0
        ev_p = float(r_pgm.mean()) if m.any() else 0.0
        ev_sp = float(r_spg.mean()) if m.any() else 0.0
        delta = ev_sp - ev_s
        rows.append(dict(
            symbol=s, n=int(m.sum()),
            EV_STATE=ev_s, EV_PGM=ev_p, EV_STATE_PGM=ev_sp,
            delta_STATE_PGM_vs_STATE=delta,
            total_delta_R=float((r_spg - r_state).sum()),
        ))
    n_pos = sum(1 for r in rows if r["delta_STATE_PGM_vs_STATE"] > 0)
    return dict(per_symbol=rows, n_symbols_positive_delta=n_pos)


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------
def run_audit_only() -> Dict[str, Any]:
    """Provenance / contract audit. NO PGM fit, NO economic Q fit, NO formal verdict."""
    head = git_head()
    economic = load_economic()
    sample = load_pgm_sample()

    # hard reward assertions
    assert (economic["reward_SKIP"] == 0).all(), "reward_SKIP must be exactly 0"
    assert np.isfinite(economic[REWARD_COLS].to_numpy()).all(), "reward_* must be finite"
    assert (economic["reward_end_semantics"] == HARDENING_VERSION).all(), "reward_end_semantics mismatch"

    j = exact_join(economic, sample)

    # PGM-side duplicate bar keys (the thing that actually matters for the many_to_one join).
    # Reported from the RAW sample (before the dedup applied inside load_pgm_sample).
    raw_sample = pd.read_parquet(pgm.SAMPLE_PATH)
    pgm_dup_per_sym = {s: int(g.duplicated(subset=["symbol", "bar_t"]).sum())
                       for s, g in raw_sample.groupby("symbol")}

    coverage = {}
    for block in [CALIB_BLOCK, TEST_BLOCK]:
        sub = j[j["block"] == block]
        per_sym = {}
        for s, g in sub.groupby("symbol"):
            n_ec = len(g)
            n_match = int(g["matched"].sum())
            dup = pgm_dup_per_sym.get(s, 0)
            per_sym[s] = dict(
                n_economic=n_ec, n_matched=n_match,
                coverage=round(n_match / n_ec, 4) if n_ec else 0.0,
                duplicate_count=dup, missing_count=n_ec - n_match,
            )
        coverage[block] = per_sym

    forbidden = set(REWARD_COLS) | {"target_price", "stop_price", "target_atr",
                                    "entry_price", "reference_entry"}
    forbidden_violation = (set(STATE_COLS) | set(PGM_SCORE_COLS)) & forbidden

    tb3_start = pd.to_datetime(economic[economic["block"] == TEST_BLOCK]["entry_time"]).min()
    tb2 = economic[economic["block"] == CALIB_BLOCK]
    tb2_end = pd.to_datetime(tb2["reward_end_time"])
    leak = int((tb2_end >= tb3_start).sum())
    max_tb2_end = tb2_end.max()

    audit = dict(
        base_head=head,
        experiment_scope=EXPERIMENT_SCOPE,
        reuse_map=REUSE_MAP,
        reward_artifact=dict(
            reward_cols=REWARD_COLS,
            n_rows=len(economic),
            row_counts_by_block={b: int((economic["block"] == b).sum())
                                 for b in ["TB1", "TB2", "TB3", "TB4"]},
            reward_end_semantics=HARDENING_VERSION,
            reward_SKIP_all_zero=bool((economic["reward_SKIP"] == 0).all()),
        ),
        economic_row_counts_by_TB={b: int((economic["block"] == b).sum())
                                  for b in ["TB1", "TB2", "TB3", "TB4"]},
        pgm_sample_schema=dict(n_rows=len(sample),
                               n_symbols=int(sample["symbol"].nunique()),
                               columns=list(sample.columns)),
        exact_join_semantics="symbol + (signal_bar_index == bar_t); exact merge (validate=many_to_one); no asof/ffill/bfill; matched via _merge=='both'",
        tb2_tb3_match_coverage=coverage,
        feature_view_dimensions=dict(
            STATE=len(STATE_COLS), PGM=len(PGM_SCORE_COLS),
            STATE_PGM=len(STATE_COLS) + len(PGM_SCORE_COLS),
        ),
        forbidden_future_columns_check=dict(
            forbidden=sorted(forbidden), violations=sorted(forbidden_violation)),
        reward_end_purge_check=dict(
            test_start=str(tb3_start),
            tb2_leak_rows=leak,
            max_tb2_reward_end_time=str(max_tb2_end),
            note="rows with reward_end_time >= test_start are purged from CALIBRATION",
        ),
        note="audit-only: NO PGM fit, NO economic Q fit, NO formal verdict",
    )
    print(json.dumps(audit, indent=2, default=str))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "eb0a_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    return audit


def _build_routes(j: pd.DataFrame, n_cap: int | None = None) -> Dict[str, Any]:
    """Fit PGM + Q and produce chosen-reward arrays for every route on the TB3 test set.

    `j` must already be filtered to matched CALIB/TEST rows. If n_cap is given,
    TB2/TB3 are deterministically subsampled to <= n_cap rows each (smoke only).
    """
    if n_cap is not None:
        parts = []
        for block in [CALIB_BLOCK, TEST_BLOCK]:
            sub = j[j["block"] == block]
            if len(sub) > n_cap:
                sub = sub.sample(n=n_cap, random_state=SEED).reset_index(drop=True)
            parts.append(sub)
        j = pd.concat(parts, ignore_index=True)

    X_state = state_matrix(j, STATE_COLS)
    fit_A = pgm.fit_samplers_for_window(pgm.WINDOWS[0], pgm.SAMPLE_PATH, pgm.TRANSITION_SAMPLE_PATH)
    fit_B = pgm.fit_samplers_for_window(pgm.WINDOWS[1], pgm.SAMPLE_PATH, pgm.TRANSITION_SAMPLE_PATH)
    X_pgm = compute_pgm_scores(j, fit_A, fit_B)
    X_state_pgm = np.column_stack([X_state, X_pgm])

    R = reward_matrix(j)
    trading_day = pd.to_datetime(j["entry_time"]).dt.normalize().to_numpy()

    tr_mask = (j["block"] == CALIB_BLOCK).to_numpy()
    te_mask = (j["block"] == TEST_BLOCK).to_numpy()
    test_start = pd.to_datetime(j[te_mask]["entry_time"]).min()
    tr_keep = tr_mask & (pd.to_datetime(j["reward_end_time"]) < test_start).to_numpy()

    routes: Dict[str, Any] = {}
    n_te = int(te_mask.sum())
    for name, fixed in [("ALWAYS_SKIP", 0), ("ALWAYS_MARKET", 1), ("ALWAYS_REASSESS", 3)]:
        a = np.full(n_te, fixed, dtype=int)
        chosen = R[te_mask][np.arange(n_te), a]
        routes[name] = (a, chosen)

    for vname, X in [("STATE", X_state), ("PGM", X_pgm), ("STATE_PGM", X_state_pgm)]:
        Xt, Xte, _ = train_only_impute(X[tr_keep], X[te_mask])
        models = fit_q(Xt, R[tr_keep], [f"f{i}" for i in range(X.shape[1])])
        a, chosen = policy_rewards(Xte, models, R[te_mask])
        routes[vname] = (a, chosen)

    return dict(
        j=j, R=R, trading_day=trading_day, tr_mask=tr_mask, te_mask=te_mask,
        tr_keep=tr_keep, routes=routes, n_train=int(tr_keep.sum()), n_test=n_te,
    )


def run_smoke(n_cap: int = 512) -> Dict[str, Any]:
    """Small-subset pipeline smoke. Executes fit + Q + metrics but emits NO formal verdict."""
    head = git_head()
    economic = load_economic()
    sample = load_pgm_sample()
    j = exact_join(economic, sample)
    j = j[j["matched"]].reset_index(drop=True)

    built = _build_routes(j, n_cap=n_cap)
    routes = built["routes"]
    te_mask = built["te_mask"]
    R_te = built["R"][te_mask]
    td_te = built["trading_day"][te_mask]
    sym_te = built["j"]["symbol"].to_numpy()[te_mask]

    metrics = {}
    for name, (a, chosen) in routes.items():
        metrics[name] = route_metrics(chosen, a, R_te, td_te)

    r_spg = routes["STATE_PGM"][1]
    r_state = routes["STATE"][1]
    boot = incremental_bootstrap(r_spg, r_state, td_te)
    breadth = symbol_breadth(routes, sym_te)

    out = dict(
        scope=EXPERIMENT_SCOPE,
        mode="SMOKE",
        base_head=head,
        n_train=built["n_train"],
        n_test=built["n_test"],
        routes=metrics,
        PRIMARY_comparison_STATE_PGM_vs_STATE=boot,
        SECONDARY_comparison_PGM_vs_STATE=incremental_bootstrap(
            routes["PGM"][1], r_state, td_te),
        SECONDARY_comparison_STATE_PGM_vs_ALWAYS_REASSESS=incremental_bootstrap(
            r_spg, routes["ALWAYS_REASSESS"][1], td_te),
        symbol_breadth=breadth,
        note=("SMOKE: pipeline executed on <=%d/block subset; NO formal verdict; "
              "not scientifically conclusive" % n_cap),
    )
    assert "verdict" not in out, "smoke must not emit a formal verdict"
    print(json.dumps(out, indent=2, default=str))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "eb0a_smoke.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def run_formal() -> Dict[str, Any]:
    """Full-data run + pre-registered verdict. NOT executed in the current round."""
    head = git_head()
    economic = load_economic()
    sample = load_pgm_sample()
    j = exact_join(economic, sample)
    j = j[j["matched"]].reset_index(drop=True)

    built = _build_routes(j, n_cap=None)
    routes = built["routes"]
    te_mask = built["te_mask"]
    R_te = built["R"][te_mask]
    td_te = built["trading_day"][te_mask]
    sym_te = built["j"]["symbol"].to_numpy()[te_mask]

    metrics = {}
    for name, (a, chosen) in routes.items():
        metrics[name] = route_metrics(chosen, a, R_te, td_te)

    r_spg = routes["STATE_PGM"][1]
    r_state = routes["STATE"][1]
    boot = incremental_bootstrap(r_spg, r_state, td_te)
    breadth = symbol_breadth(routes, sym_te)

    mean_delta = boot["mean_delta_R"]
    ci_lo = boot["ci_lo"]
    if mean_delta > 0 and ci_lo > 0:
        verdict = "PGM_INCREMENTAL_ECONOMIC_VALUE_SUPPORTED_ON_FROZEN_UNIVERSE"
    else:
        verdict = "PGM_INCREMENTAL_ECONOMIC_VALUE_NOT_SUPPORTED_ON_FROZEN_UNIVERSE"
    assert verdict not in FORBIDDEN_VERDICTS, "formal verdict must not be a forbidden strategy verdict"

    out = dict(
        scope=EXPERIMENT_SCOPE,
        mode="FORMAL",
        base_head=head,
        n_train=built["n_train"],
        n_test=built["n_test"],
        routes=metrics,
        PRIMARY_comparison_STATE_PGM_vs_STATE=boot,
        SECONDARY_comparison_PGM_vs_STATE=incremental_bootstrap(routes["PGM"][1], r_state, td_te),
        SECONDARY_comparison_STATE_PGM_vs_ALWAYS_REASSESS=incremental_bootstrap(
            r_spg, routes["ALWAYS_REASSESS"][1], td_te),
        symbol_breadth=breadth,
        verdict=verdict,
    )
    print(json.dumps(out, indent=2, default=str))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "eb0a_formal.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def main():
    ap = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    ap.add_argument("--audit-only", action="store_true", help="Provenance/contract audit only (no fit, no verdict).")
    ap.add_argument("--smoke", action="store_true", help="Small-subset pipeline smoke (no formal verdict).")
    ap.add_argument("--formal", action="store_true", help="Full run + pre-registered verdict (NOT used this round).")
    args = ap.parse_args()

    if args.formal:
        run_formal()
    elif args.smoke:
        run_smoke()
    else:
        # default == audit-only
        run_audit_only()


if __name__ == "__main__":
    main()
