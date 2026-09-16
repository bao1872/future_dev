"""
experiment_pgm_native0d_acceleration_terminal_outcome_v1.py
===========================================================

PGM-NATIVE-0D -- Acceleration Terminal-Outcome & Economic Gate

Question: while the frozen PGM is still emitting a continuation signal (d_t = sign(m_t)),
which acceleration states imply THIS continuation will lose money?

Three layers (single pre-registration):
  0D-A  H1 harm classifier   : P(pi < 0 | H=1, F, A)      -> information
  0D-B  H1 payoff regression : E[pi   | H=1, F, A]        -> economic magnitude
  0D-C  branch-value gate    : GateA = d_t * 1[V_hat > c] -> money

Mixture value (all inputs known at bar t):
    mu0  = beta0 * |m_t|                    (single-parameter H0 calibration)
    mu1  = payoff head evaluated on H1 branch
    p_h  = P(H=1 | T2 + U + E)              (0C hazard owner, frozen)
    V    = (1 - p_h) * mu0 + p_h * mu1

Frozen economic rule: execute iff V > cost. The threshold IS the cost -- no search.

SCOPE
-----
EXPERIMENT_SCOPE = EXPLORATORY_STRATEGY_DEVELOPMENT_ON_FROZEN_TB1_TB2_TB3_ONLY
TB4 is FORBIDDEN. TB3 was previously inspected -> exploratory only.

ROUND 1 (this file): architecture + audit + smoke ONLY. `--full-exploratory` is hard-blocked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

for _bt in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_bt] = "1"

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import scipy.stats
import sklearn.metrics
from sklearn.linear_model import Ridge

import research.liquidity_oracle_atlas.experiment_pgm_native0a_one_step_alpha_v1 as n0a
import research.liquidity_oracle_atlas.experiment_pgm_native0c_state_augmentation_v1 as n0c
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1c_free_run_rollout_v1 as pgm
import research.liquidity_oracle_atlas.experiment_path0_episode_path_memory_v1 as pm
import research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 as ex0


# ===========================================================================
# Governance constants
# ===========================================================================
BASE_SHA = "3399f20b5497572dd44d29b48f2f340b34d29599"
EXPERIMENT_NAME = "PGM-NATIVE-0D -- Acceleration Terminal-Outcome & Economic Gate"
EXPERIMENT_SCOPE = "EXPLORATORY_STRATEGY_DEVELOPMENT_ON_FROZEN_TB1_TB2_TB3_ONLY"
PREFIX = "pgm_native0d1"

ALLOWED_BLOCKS = ["TB1", "TB2", "TB3"]
TB2_BLOCK = "TB2"
TB3_BLOCK = "TB3"

EPOCH_ACK = "__d0_ack"

A_COLS: List[str] = [
    "a_dir_velocity",
    "a_dir_accel_1",
    "a_dir_jerk_1",
    "a_speed_accel_1",
    "a_range_accel_2",
    "a_eff_slope_1",
    "a_burst_exhaustion",
    "a_conviction_burst",
]

PRIMARY_COST_ATR0 = 0.01
COST_GRID = [0.0, 0.01, 0.02, 0.03, 0.05, 0.10]
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20260916
SMOKE_BOOTSTRAP_N = 200
SMOKE_EVAL_CAP = 1024
CLUSTER_OWNER = "entry_day"

QUINTILE_FEATURES = ["a_burst_exhaustion", "a_conviction_burst"]

ARTIFACT_FILES: List[str] = [
    f"{PREFIX}_outcome_metrics.csv",
    f"{PREFIX}_information_bootstrap.csv",
    f"{PREFIX}_strategy_metrics.csv",
    f"{PREFIX}_economic_bootstrap.csv",
    f"{PREFIX}_cost_grid.csv",
    f"{PREFIX}_symbol_metrics.csv",
    f"{PREFIX}_acceleration_quintiles.csv",
    f"{PREFIX}_formal_summary.json",
]
FULL_RESULT_DIR = _REPO_ROOT / "research" / "analysis_results" / "local_liquidity_transition_v0"

STRATEGY_FIELDS = [
    "n_decisions", "n_trades", "trade_rate",
    "gross_total_ATR0", "net_total_ATR0",
    "gross_EV_per_decision", "net_EV_per_decision", "net_EV_per_trade",
    "win_rate", "mean_win", "mean_loss", "payoff_ratio", "profit_factor",
    "break_even_cost", "daily_sharpe_annualized", "max_drawdown_ATR0",
    "positive_symbol_count", "top3_profit_share",
]

VERDICT_STRINGS = {
    "INFO_JOINT": "PGM_ACCELERATION_TERMINAL_OUTCOME_SUPPORTED_JOINTLY_EXPLORATORY",
    "INFO_HARM_ONLY": "PGM_ACCELERATION_HARM_ONLY_EXPLORATORY",
    "INFO_PAYOFF_ONLY": "PGM_ACCELERATION_PAYOFF_ONLY_EXPLORATORY",
    "INFO_NONE": "PGM_ACCELERATION_TERMINAL_OUTCOME_NOT_SUPPORTED_EXPLORATORY",
    "ECON_SUPPORTED": "PGM_ACCELERATION_ECONOMIC_GATE_SUPPORTED_EXPLORATORY",
    "ECON_NOT_INCREMENTAL": "PGM_TERMINAL_VALUE_GATE_PROFITABLE_BUT_ACCELERATION_NOT_INCREMENTAL_EXPLORATORY",
    "ECON_NOT_SUPPORTED": "PGM_ACCELERATION_ECONOMIC_GATE_NOT_SUPPORTED_EXPLORATORY",
}

FUTURE_TOKENS = ["shift(-1)", "future_", "next_", "remaining", "final_"]


# ===========================================================================
# Acceleration block (exactly 8 features, causal)
# ===========================================================================
def add_acceleration_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the frozen 8-variable acceleration block. Causal: no forward shift."""
    x = df.sort_values(["symbol", "episode_id", "bar_t"], kind="stable").copy()
    g = x.groupby(["symbol", "episode_id"], sort=False, group_keys=False)

    r0 = x["path_last_return_R"].to_numpy(np.float64)

    r1 = g["path_last_return_R"].shift(1)
    r1 = r1.fillna(x["path_last_return_R"]).to_numpy(np.float64)

    r2 = g["path_last_return_R"].shift(2)
    r2 = r2.fillna(pd.Series(r1, index=x.index)).to_numpy(np.float64)

    q0 = x["path_current_bar_range_R"].to_numpy(np.float64)
    q1 = g["path_current_bar_range_R"].shift(1)
    q1 = q1.fillna(x["path_current_bar_range_R"]).to_numpy(np.float64)
    q2 = g["path_current_bar_range_R"].shift(2)
    q2 = q2.fillna(pd.Series(q1, index=x.index)).to_numpy(np.float64)

    eff0 = x["e_local_eff_3"].to_numpy(np.float64)
    eff1 = g["e_local_eff_3"].shift(1)
    eff1 = eff1.fillna(x["e_local_eff_3"]).to_numpy(np.float64)

    m = x["score_mu"].to_numpy(np.float64)
    d = np.sign(m)

    x["a_dir_velocity"] = d * r0
    x["a_dir_accel_1"] = d * (r0 - r1)
    x["a_dir_jerk_1"] = d * (r0 - 2.0 * r1 + r2)
    x["a_speed_accel_1"] = np.abs(r0) - np.abs(r1)
    x["a_range_accel_2"] = q0 - 2.0 * q1 + q2
    x["a_eff_slope_1"] = eff0 - eff1
    x["a_burst_exhaustion"] = (np.maximum(x["a_speed_accel_1"], 0.0)
                               * np.maximum(-x["a_eff_slope_1"], 0.0))
    x["a_conviction_burst"] = np.abs(m) * np.maximum(x["a_dir_accel_1"], 0.0)
    return x


def audit_acceleration_finite(df: pd.DataFrame) -> Dict[str, Any]:
    bad = {}
    for c in A_COLS:
        v = df[c].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(v)):
            bad[c] = int(np.sum(~np.isfinite(v)))
    return dict(all_finite=len(bad) == 0, non_finite_counts=bad)


def _prefix_invariance_check() -> bool:
    n = 12

    def mk(perturb: bool) -> pd.DataFrame:
        rng = np.random.default_rng(5)
        df = pd.DataFrame(dict(
            symbol=["X"] * n, episode_id=["E1"] * n, block=["TB1"] * n,
            bar_t=np.arange(100, 100 + n), start_bar=[100] * n,
            score_mu=rng.normal(0, 1, n),
            path_last_return_R=rng.normal(0, 0.5, n),
            path_current_bar_range_R=np.abs(rng.normal(0.4, 0.1, n)),
            e_local_eff_3=rng.uniform(0, 1, n),
        ))
        if perturb:
            for c in ["path_last_return_R", "path_current_bar_range_R", "e_local_eff_3"]:
                df.loc[6:, c] = df.loc[6:, c] * 4.0 + 3.0
            df.loc[6:, "bar_t"] = df.loc[6:, "bar_t"] + 30
        return df

    a = add_acceleration_features(mk(False))
    b = add_acceleration_features(mk(True))
    for c in A_COLS:
        if not np.array_equal(a[c].to_numpy(np.float64)[:6], b[c].to_numpy(np.float64)[:6]):
            return False
    return True


# ===========================================================================
# Scored frame + same-block entry validity
# ===========================================================================
def attach_baseline_score(df: pd.DataFrame, mc_sampler: Any) -> pd.DataFrame:
    out = df.copy()
    mom = mc_sampler.analytic_conditional_support(out)
    out["score_mu"] = -np.asarray(mom["z_d_up_mu"], dtype=np.float64)
    return out


def finalize_scored_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["base_action"] = np.sign(out["score_mu"].to_numpy(np.float64))
    out["abs_score_mu"] = np.abs(out["score_mu"].to_numpy(np.float64))
    out["pi"] = out["base_action"].to_numpy(np.float64) * out["r_trad_OC_ATR0"].to_numpy(np.float64)
    out["harm_flag"] = (out["pi"].to_numpy(np.float64) < 0.0).astype(np.int64)
    return out


def compute_same_block_entry_valid(df: pd.DataFrame) -> np.ndarray:
    """entry_day must lie in the decision_day day-set of the row's own block."""
    valid = np.zeros(len(df), dtype=bool)
    entry = pd.to_datetime(df["entry_day"]).to_numpy()
    day = pd.to_datetime(df["decision_day"]).to_numpy()
    block = df["block"].to_numpy()
    for b in np.unique(block):
        m = block == b
        day_set = set(day[m])
        valid[m] = np.isin(entry[m], list(day_set))
    return valid


def assert_allowed_blocks(df: pd.DataFrame) -> None:
    bad = set(df["block"].unique()) - set(ALLOWED_BLOCKS)
    if bad:
        raise SystemExit(f"STOP_PGM_NATIVE0D_FORBIDDEN_BLOCK: {sorted(bad)}")


# ===========================================================================
# Trigger ONCE per block (efficiency contract)
# ===========================================================================
_ACC_DONE: Dict[str, bool] = {}


def build_acceleration_once(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    if _ACC_DONE.get(tag):
        raise SystemExit("STOP_PGM_NATIVE0D_ACCELERATION_BUILT_TWICE")
    out = add_acceleration_features(df)
    _ACC_DONE[tag] = True
    return out


# ===========================================================================
# Outcome heads
# ===========================================================================
def outcome_base_num() -> List[str]:
    return list(pgm.T2_NUM) + list(n0c.U_COLS) + list(n0c.E_COLS) + ["score_mu", "abs_score_mu"]


def outcome_cat() -> List[str]:
    return list(pgm.CAT)


def make_ridge_pipeline(num_cols: Sequence[str], cat_cols: Sequence[str]):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    parts = [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                               ("sc", StandardScaler())]), list(num_cols))]
    if cat_cols:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), list(cat_cols)))
    return Pipeline([("pre", ColumnTransformer(parts)),
                     ("reg", Ridge(alpha=1.0, solver="lsqr"))])


def fit_harm_model(train_h1: pd.DataFrame, eval_h1: pd.DataFrame,
                   extra_cols: Sequence[str]) -> Dict[str, Any]:
    num_cols = outcome_base_num() + list(extra_cols)
    cat_cols = outcome_cat()
    cols = num_cols + cat_cols
    pipe = pm.make_pipeline(num_cols, cat_cols)
    pipe.fit(train_h1[cols], train_h1["harm_flag"].to_numpy(np.int64))
    p = np.asarray(pipe.predict_proba(eval_h1[cols])[:, 1], dtype=np.float64)
    y = eval_h1["harm_flag"].to_numpy(np.int64)
    pc = np.clip(p, 1e-15, 1.0 - 1e-15)
    row_logloss = -(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc))
    row_brier = (p - y) ** 2
    return dict(p_harm=p, row_logloss=row_logloss, row_brier=row_brier, pipeline=pipe,
                num_cols=num_cols)


def evaluate_harm(fit: Dict[str, Any], y: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=np.int64)
    p = fit["p_harm"]
    out = dict(log_loss=float(np.mean(fit["row_logloss"])),
               brier=float(np.mean(fit["row_brier"])))
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(sklearn.metrics.roc_auc_score(y, p))
        out["pr_auc"] = float(sklearn.metrics.average_precision_score(y, p))
    else:
        out["roc_auc"] = 0.5
        out["pr_auc"] = float(np.mean(y))
    return out


def fit_payoff_model(train_h1: pd.DataFrame, eval_h1: pd.DataFrame,
                     extra_cols: Sequence[str]) -> Dict[str, Any]:
    num_cols = outcome_base_num() + list(extra_cols)
    cat_cols = outcome_cat()
    cols = num_cols + cat_cols
    pipe = make_ridge_pipeline(num_cols, cat_cols)
    pipe.fit(train_h1[cols], train_h1["pi"].to_numpy(np.float64))
    pred = np.asarray(pipe.predict(eval_h1[cols]), dtype=np.float64)
    y = eval_h1["pi"].to_numpy(np.float64)
    return dict(pred_pi=pred, sqerr=(pred - y) ** 2, abserr=np.abs(pred - y),
                pipeline=pipe, num_cols=num_cols)


def evaluate_payoff(fit: Dict[str, Any], y: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    pred = fit["pred_pi"]
    mse = float(np.mean(fit["sqerr"]))
    mae = float(np.mean(fit["abserr"]))
    sp = float(scipy.stats.spearmanr(pred, y).statistic) if len(pred) > 2 else 0.0
    return dict(mse=mse, mae=mae, spearman=sp)


# ===========================================================================
# Branch value
# ===========================================================================
def calibrate_mu0(train_h0: pd.DataFrame) -> float:
    x = np.abs(train_h0["score_mu"].to_numpy(np.float64))
    y = train_h0["pi"].to_numpy(np.float64)
    return float(np.dot(x, y) / (np.dot(x, x) + 1e-12))


def compose_branch_value(p_h: np.ndarray, score_mu: np.ndarray,
                         beta0: float, mu1: np.ndarray) -> np.ndarray:
    mu0 = beta0 * np.abs(np.asarray(score_mu, dtype=np.float64))
    return (1.0 - np.asarray(p_h, dtype=np.float64)) * mu0 + \
        np.asarray(p_h, dtype=np.float64) * np.asarray(mu1, dtype=np.float64)


# ===========================================================================
# Fixed policies
# ===========================================================================
def apply_value_gate(score_mu: np.ndarray, value_hat: np.ndarray, cost: float) -> np.ndarray:
    base = np.sign(np.asarray(score_mu, dtype=np.float64))
    execute = np.asarray(value_hat, dtype=np.float64) > cost
    return base * execute.astype(np.float64)


def apply_value_flip(score_mu: np.ndarray, value_hat: np.ndarray, cost: float) -> np.ndarray:
    base = np.sign(np.asarray(score_mu, dtype=np.float64))
    v = np.asarray(value_hat, dtype=np.float64)
    side = np.where(v > cost, 1.0, np.where(v < -cost, -1.0, 0.0))
    return base * side


def net_return(action: np.ndarray, r_trad: np.ndarray, cost: float) -> np.ndarray:
    a = np.asarray(action, dtype=np.float64)
    r = np.asarray(r_trad, dtype=np.float64)
    return a * r - cost * (a != 0).astype(np.float64)


# ===========================================================================
# Strategy metrics
# ===========================================================================
def strategy_metrics(action: np.ndarray, r_trad: np.ndarray, cost: float,
                     entry_day: np.ndarray, symbol: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(action, dtype=np.float64)
    r = np.asarray(r_trad, dtype=np.float64)
    executed = a != 0
    gross = a * r
    net = gross - cost * executed.astype(np.float64)
    n_dec = int(len(a))
    n_tr = int(executed.sum())
    trade_rate = n_tr / n_dec if n_dec else 0.0

    wins = net[executed & (net > 0)]
    losses = net[executed & (net < 0)]
    mean_win = float(np.mean(wins)) if len(wins) else 0.0
    mean_loss = float(np.mean(losses)) if len(losses) else 0.0
    pf = float(np.sum(wins) / abs(np.sum(losses))) if len(losses) and np.sum(losses) != 0 else float("nan")

    daily = pd.Series(net).groupby(pd.Series(np.asarray(entry_day))).sum()
    dvals = daily.to_numpy(dtype=np.float64)
    sd = float(np.std(dvals, ddof=1)) if len(dvals) > 1 else 0.0
    sharpe = float(np.sqrt(252.0) * np.mean(dvals) / sd) if sd > 1e-15 else 0.0
    cum = np.cumsum(dvals)
    max_dd = float(np.min(cum - np.maximum.accumulate(cum))) if len(cum) else 0.0

    gross_ev_dec = float(np.mean(gross)) if n_dec else 0.0
    break_even = gross_ev_dec / trade_rate if trade_rate > 0 else float("nan")

    sym = np.asarray(symbol)
    by_sym = {}
    for s in np.unique(sym):
        m = sym == s
        by_sym[s] = dict(net_total=float(np.sum(net[m])), trade_count=int(executed[m].sum()))
    pos_syms = sum(1 for v in by_sym.values() if v["net_total"] > 0)
    pos_vals = sorted([v["net_total"] for v in by_sym.values() if v["net_total"] > 0], reverse=True)
    total_pos = sum(pos_vals)
    top3_share = float(sum(pos_vals[:3]) / total_pos) if total_pos > 0 else 0.0

    return dict(
        n_decisions=n_dec, n_trades=n_tr, trade_rate=trade_rate,
        gross_total_ATR0=float(np.sum(gross)), net_total_ATR0=float(np.sum(net)),
        gross_EV_per_decision=float(np.mean(gross)) if n_dec else 0.0,
        net_EV_per_decision=float(np.mean(net)) if n_dec else 0.0,
        net_EV_per_trade=float(np.mean(net[executed])) if n_tr else 0.0,
        win_rate=float(len(wins) / n_tr) if n_tr else 0.0,
        mean_win=mean_win, mean_loss=mean_loss,
        payoff_ratio=(mean_win / abs(mean_loss)) if mean_loss < 0 else float("nan"),
        profit_factor=pf, break_even_cost=break_even,
        daily_sharpe_annualized=sharpe, max_drawdown_ATR0=max_dd,
        by_symbol=by_sym, positive_symbol_count=int(pos_syms),
        top3_profit_share=top3_share,
    )


# ===========================================================================
# Day-clustered bootstrap (economic, paired across policies)
# ===========================================================================
def economic_bootstrap(entry_day: np.ndarray, net_by_policy: Dict[str, np.ndarray],
                       n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> Dict[str, Any]:
    """Same resampled trading days drive every policy and every paired difference."""
    day = np.asarray(entry_day)
    days = np.unique(day)
    D = len(days)
    didx = {d: i for i, d in enumerate(days)}
    pos = np.array([didx[x] for x in day])

    names = list(net_by_policy.keys())
    S = {k: np.zeros(D) for k in names}
    N = np.zeros(D)
    for i, p in enumerate(pos):
        N[p] += 1.0
        for k in names:
            S[k][p] += float(net_by_policy[k][i])

    rng = np.random.default_rng(seed)
    counts = rng.multinomial(D, np.full(D, 1.0 / D), size=n_boot)
    denom = counts @ N
    ev = {k: (counts @ S[k]) / denom for k in names}

    def _summ(point: float, dist: np.ndarray) -> Dict[str, float]:
        return dict(point=float(point),
                    ci95_lower=float(np.percentile(dist, 2.5)),
                    ci95_upper=float(np.percentile(dist, 97.5)),
                    p_pos=float(np.mean(dist > 0)))

    out: Dict[str, Any] = {}
    for k in names:
        out[k] = _summ(float(np.sum(net_by_policy[k]) / len(net_by_policy[k])), ev[k])
    for a, b in [("GATEA", "BASE"), ("GATEA", "GATE0")]:
        if a in ev and b in ev:
            d = ev[a] - ev[b]
            out[f"{a}-{b}"] = _summ(
                float(np.mean(net_by_policy[a]) - np.mean(net_by_policy[b])), d)
    out["n_days"] = int(D)
    out["n_boot"] = int(n_boot)
    return out


# ===========================================================================
# Verdicts
# ===========================================================================
def determine_information_verdict(dll: Dict[str, float], dmse: Dict[str, float]) -> str:
    a = dll["ci95_lower"] > 0
    b = dmse["ci95_lower"] > 0
    if a and b:
        return VERDICT_STRINGS["INFO_JOINT"]
    if a and not b:
        return VERDICT_STRINGS["INFO_HARM_ONLY"]
    if (not a) and b:
        return VERDICT_STRINGS["INFO_PAYOFF_ONLY"]
    return VERDICT_STRINGS["INFO_NONE"]


def determine_economic_verdict(boot_tb2: Dict[str, Any], boot_tb3: Dict[str, Any]) -> str:
    p1 = boot_tb2["GATEA"]["ci95_lower"] > 0
    p2 = boot_tb3["GATEA"]["ci95_lower"] > 0
    p3 = boot_tb3["GATEA-GATE0"]["ci95_lower"] > 0
    if p1 and p2 and p3:
        return VERDICT_STRINGS["ECON_SUPPORTED"]
    if p1 and p2 and (not p3):
        return VERDICT_STRINGS["ECON_NOT_INCREMENTAL"]
    return VERDICT_STRINGS["ECON_NOT_SUPPORTED"]


# ===========================================================================
# Quintile diagnostics (train-frozen edges)
# ===========================================================================
def quintile_edges(train: pd.DataFrame, col: str) -> np.ndarray:
    q = np.quantile(train[col].to_numpy(np.float64), [0.2, 0.4, 0.6, 0.8])
    return np.concatenate([[-np.inf], q, [np.inf]])


def quintile_diagnostic(eval_df: pd.DataFrame, col: str, edges: np.ndarray,
                        block: str) -> List[Dict[str, Any]]:
    bins = np.digitize(eval_df[col].to_numpy(np.float64), np.asarray(edges)[1:-1])
    hz = eval_df["hazard"].to_numpy(np.int64)
    pi = eval_df["pi"].to_numpy(np.float64)
    harm = (pi < 0)
    rows = []
    for b in range(5):
        m = bins == b
        if not m.any():
            continue
        rows.append(dict(block=block, feature=col, bin=b, n=int(m.sum()),
                         H1_prevalence=float(hz[m].mean()),
                         harm_rate=float(harm[m].mean()),
                         mean_pi=float(pi[m].mean())))
    return rows


# ===========================================================================
# Shared helpers
# ===========================================================================
def compute_artifact_hashes() -> Dict[str, str]:
    return n0c.compute_artifact_hashes()


def print_reuse_map() -> None:
    print("PGM-NATIVE-0D REUSE MAP:", flush=True)
    for k, v in REUSE_MAP.items():
        print(f"  REUSE {k}: {v}", flush=True)


REUSE_MAP = {
    "universe": "n0a.load_observed_decision_universe",
    "incremental_state": "n0c.add_incremental_state_features (U/E)",
    "returns_alignment": "n0a.align_raw_bars_and_returns",
    "decision_day": "n0c.attach_decision_day",
    "pgm_fit": "pgm.fit_samplers_for_window",
    "hazard_owner": "n0c.fit_terminal_hazard_variant",
    "logistic": "pm.make_pipeline (fixed L2, C=1)",
    "day_bootstrap": "n0c.fast_cluster_bootstrap_delta / economic_bootstrap",
}


def require_full_authorization() -> None:
    token = os.environ.get("AUTHORIZE_PGM_NATIVE0D_FULL_EXPLORATORY", "").strip()
    if token != "1":
        raise SystemExit("STOP_PGM_NATIVE0D_FULL_EXPLORATORY_NOT_AUTHORIZED")


def load_prepared_frame() -> Dict[str, Any]:
    obs = n0a.load_observed_decision_universe()
    assert_allowed_blocks(obs)
    feat = n0c.add_incremental_state_features(obs)
    _, _, bars_by_sym = ex0.load_env()
    obs_day = n0c.attach_decision_day(feat, bars_by_sym)
    aligned = n0a.align_raw_bars_and_returns(obs_day, bars_by_sym,
                                             cur_truth=n0a.load_transition_truth_audit()["cur"])[0]
    return dict(obs_day=obs_day, aligned=aligned, bars_by_sym=bars_by_sym)


def prepare_window_windowframe(aligned: pd.DataFrame, fit: Dict[str, Any],
                               tag: str) -> pd.DataFrame:
    """Attach the WINDOW-OWNED baseline PGM score, acceleration block and economic labels/scope.

    p_h_UE (hazard) is NOT computed here -- it is fitted/scored per window inside _run_window.
    """
    mc = fit["trans_samplers"][n0c.PRIMARY_TRANSITION_HEAD]
    scored = attach_baseline_score(aligned, mc)
    scored = build_acceleration_once(scored, tag=tag)
    scored = finalize_scored_frame(scored)
    scored["same_block_entry_valid"] = compute_same_block_entry_valid(scored)
    return scored


def verify_window_score_owner(scored: pd.DataFrame, fit: Dict[str, Any], block: str) -> float:
    """Prove scored.score_mu on `block` is owned by this window's MC sampler (fail-closed)."""
    sub = scored[scored["block"] == block]
    if len(sub) == 0:
        raise SystemExit(f"STOP_PGM_NATIVE0D_SCORE_OWNER_PARITY_FAIL: empty block {block}")
    mc = fit["trans_samplers"][n0c.PRIMARY_TRANSITION_HEAD]
    actual = sub["score_mu"].to_numpy(np.float64)
    expected = -np.asarray(mc.analytic_conditional_support(sub)["z_d_up_mu"], dtype=np.float64)
    diff = actual - expected
    if not (np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
            and np.all(np.isfinite(diff))):
        raise SystemExit("STOP_PGM_NATIVE0D_SCORE_OWNER_NONFINITE")
    d = float(np.max(np.abs(diff)))
    if d > 1e-12:
        raise SystemExit(f"STOP_PGM_NATIVE0D_SCORE_OWNER_PARITY_FAIL: {block} {d}")
    return d


# ===========================================================================
# Modes
# ===========================================================================
def run_audit_only() -> None:
    print("=" * 50, flush=True)
    print("PGM-NATIVE-0D: AUDIT-ONLY", flush=True)
    print("=" * 50, flush=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT),
                                   text=True).strip()
    print(f"[AUDIT] HEAD={head}")
    print(f"[AUDIT] BASE_SHA={BASE_SHA}")
    if subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"],
                      cwd=str(_REPO_ROOT), capture_output=True).returncode != 0:
        raise SystemExit("STOP_PGM_NATIVE0D_FREEZE_CHECK_FAIL")
    hashes = compute_artifact_hashes()
    print(f"[AUDIT] sample_sha={hashes['sample_artifact_sha256']}")
    print(f"[AUDIT] transition_sha={hashes['transition_artifact_sha256']}")

    prep = load_prepared_frame()
    obs = n0a.load_observed_decision_universe()
    aud = n0a.audit_decision_universe(obs)
    print(f"[AUDIT] n_all_obs={aud['n_all_obs']} H0={aud['n_hazard0']} H1={aud['n_hazard1']}")
    print(f"[AUDIT] symbols={len(aud['symbols'])}")
    print(f"[AUDIT] blocks={sorted(obs['block'].unique().tolist())}")
    err = n0a.audit_atr0_owner_parity(obs, n0a.load_transition_truth_audit()["cur"])
    print(f"[AUDIT] max_abs_atr0_owner_error={err:.2e}")

    aligned = prep["aligned"]
    print(f"[AUDIT] A_COLS n={len(A_COLS)} {A_COLS}")
    if len(A_COLS) != 8:
        raise SystemExit("STOP_PGM_NATIVE0D_A_COLS_NOT_EIGHT")
    collision = set(A_COLS) & set(outcome_base_num())
    print(f"[AUDIT] A_COLS vs OUTCOME_BASE_NUM collision={len(collision)}")
    if collision:
        raise SystemExit(f"STOP_PGM_NATIVE0D_INCREMENTAL_COLUMN_COLLISION: {collision}")

    print("[AUDIT] Fitting Window A samplers...", flush=True)
    fit_A = pgm.fit_samplers_for_window(pgm.WINDOWS[0], pgm.SAMPLE_PATH,
                                        pgm.TRANSITION_SAMPLE_PATH)
    scored_A = prepare_window_windowframe(aligned, fit_A, "scored_A")
    print("[AUDIT] Fitting Window B samplers...", flush=True)
    fit_B = pgm.fit_samplers_for_window(pgm.WINDOWS[1], pgm.SAMPLE_PATH,
                                        pgm.TRANSITION_SAMPLE_PATH)
    scored_B = prepare_window_windowframe(aligned, fit_B, "scored_B")

    dA = verify_window_score_owner(scored_A, fit_A, TB2_BLOCK)
    dB = verify_window_score_owner(scored_B, fit_B, TB3_BLOCK)
    print(f"[AUDIT] Window A score-owner parity (TB2) max_abs_diff={dA:.3e}")
    print(f"[AUDIT] Window B score-owner parity (TB3) max_abs_diff={dB:.3e}")

    fin = audit_acceleration_finite(scored_A)
    finB = audit_acceleration_finite(scored_B)
    print(f"[AUDIT] WindowA acceleration finite={fin['all_finite']} {fin['non_finite_counts']}")
    print(f"[AUDIT] WindowB acceleration finite={finB['all_finite']} {finB['non_finite_counts']}")
    if not fin["all_finite"] or not finB["all_finite"]:
        raise SystemExit("STOP_PGM_NATIVE0D_ACCELERATION_NON_FINITE")
    if not _prefix_invariance_check():
        raise SystemExit("STOP_PGM_NATIVE0D_FEATURE_FUTURE_DEPENDENCE")
    print("[AUDIT] prefix invariance: PASS")
    import inspect as _inspect
    builder_src = _inspect.getsource(add_acceleration_features)
    for tok in FUTURE_TOKENS:
        if tok in builder_src:
            raise SystemExit(f"STOP_PGM_NATIVE0D_FUTURE_TOKEN: {tok}")
    print("[AUDIT] no future token: PASS")

    first_mask = scored_A.index.isin(
        scored_A.groupby(["symbol", "episode_id"], sort=False).head(1).index)
    first_rows = scored_A[first_mask]
    if not np.allclose(first_rows["a_dir_jerk_1"].to_numpy(float), 0.0, atol=1e-12):
        raise SystemExit("STOP_PGM_NATIVE0D_FIRST_ROW_JERK_NONZERO")
    print("[AUDIT] first-row acceleration zero: PASS")

    # same-block entry exclusion counts (row/scope identical across score frames)
    for b in ALLOWED_BLOCKS:
        m = scored_A["block"] == b
        raw = int(m.sum())
        ev = int((m & scored_A["is_entry_valid"]).sum())
        sb = int((m & scored_A["is_entry_valid"] & scored_A["same_block_entry_valid"]).sum())
        print(f"[AUDIT] {b}: raw={raw} entry_valid={ev} same_block_valid={sb} excluded={ev - sb}")

    # H1 sample sizes: TB2 uses Window A score frame, TB3 uses Window B score frame
    for blk, w, sf in [(TB2_BLOCK, pgm.WINDOWS[0], scored_A), (TB3_BLOCK, pgm.WINDOWS[1], scored_B)]:
        tr = sf[(sf["block"].isin(w["train"])) & sf["same_block_entry_valid"]
                & (sf["hazard"] == 1) & (sf["base_action"] != 0)]
        ev = sf[(sf["block"] == w["eval"]) & sf["same_block_entry_valid"]
                & (sf["hazard"] == 1) & (sf["base_action"] != 0)]
        print(f"[AUDIT] window {blk}: H1 train={len(tr)} H1 eval={len(ev)}")
    print("[AUDIT] NO SCIENTIFIC VERDICT EMITTED", flush=True)


def _run_window(scored: pd.DataFrame, w: Dict[str, Any], n_boot: int,
                eval_cap: Optional[int] = None) -> Dict[str, Any]:
    """Fit all 0D heads for one window and evaluate the fixed policies on the eval block."""
    tr_all = scored[scored["block"].isin(w["train"])].reset_index(drop=True)
    ev_all = scored[scored["block"] == w["eval"]].reset_index(drop=True)
    econ_tr = tr_all[tr_all["same_block_entry_valid"]].reset_index(drop=True)
    econ_ev = ev_all[ev_all["same_block_entry_valid"]].reset_index(drop=True)
    if eval_cap is not None:
        econ_ev = econ_ev.head(eval_cap).reset_index(drop=True)
        ev_all = ev_all.head(eval_cap).reset_index(drop=True)

    h1_tr = econ_tr[(econ_tr["hazard"] == 1) & (econ_tr["base_action"] != 0)].reset_index(drop=True)
    h1_ev = econ_ev[(econ_ev["hazard"] == 1) & (econ_ev["base_action"] != 0)].reset_index(drop=True)
    h0_tr = econ_tr[(econ_tr["hazard"] == 0) & (econ_tr["base_action"] != 0)].reset_index(drop=True)

    harm0 = fit_harm_model(h1_tr, h1_ev, [])
    harmA = fit_harm_model(h1_tr, h1_ev, A_COLS)
    payoff0 = fit_payoff_model(h1_tr, h1_ev, [])
    payoffA = fit_payoff_model(h1_tr, h1_ev, A_COLS)
    y_harm = h1_ev["harm_flag"].to_numpy(np.int64)
    y_pi = h1_ev["pi"].to_numpy(np.float64)

    m0 = evaluate_harm(harm0, y_harm)
    mA = evaluate_harm(harmA, y_harm)
    p0 = evaluate_payoff(payoff0, y_pi)
    pA = evaluate_payoff(payoffA, y_pi)

    day_h1 = h1_ev["entry_day"].to_numpy()
    d_ll = n0c.fast_cluster_bootstrap_delta(day_h1, harm0["row_logloss"], harmA["row_logloss"],
                                            n_boot=n_boot)
    d_mse = n0c.fast_cluster_bootstrap_delta(day_h1, payoff0["sqerr"], payoffA["sqerr"],
                                             n_boot=n_boot)

    beta0 = calibrate_mu0(h0_tr)
    # hazard owner (frozen 0C structure: T2 + U + E) scored on the economic eval rows
    p_h_UE = n0c.fit_terminal_hazard_variant(tr_all, econ_ev,
                                             n0c.U_COLS + n0c.E_COLS)["p_eval"]
    mu1_O0 = np.asarray(payoff0["pipeline"].predict(econ_ev[outcome_base_num() + outcome_cat()]))
    mu1_OA = np.asarray(payoffA["pipeline"].predict(econ_ev[outcome_base_num() + A_COLS + outcome_cat()]))
    V0 = compose_branch_value(p_h_UE, econ_ev["score_mu"].to_numpy(np.float64), beta0, mu1_O0)
    VA = compose_branch_value(p_h_UE, econ_ev["score_mu"].to_numpy(np.float64), beta0, mu1_OA)

    base_action = econ_ev["base_action"].to_numpy(np.float64)
    score_mu = econ_ev["score_mu"].to_numpy(np.float64)
    r_trad = econ_ev["r_trad_OC_ATR0"].to_numpy(np.float64)
    day = econ_ev["entry_day"].to_numpy()
    sym = econ_ev["symbol"].to_numpy()

    def policies_at(cost: float) -> Dict[str, np.ndarray]:
        return {
            "BASE": base_action,
            "GATE0": apply_value_gate(score_mu, V0, cost),
            "GATEA": apply_value_gate(score_mu, VA, cost),
            "FLIPA": apply_value_flip(score_mu, VA, cost),
        }

    policies = policies_at(PRIMARY_COST_ATR0)
    net = {k: net_return(a, r_trad, PRIMARY_COST_ATR0) for k, a in policies.items()}
    metrics = {k: strategy_metrics(a, r_trad, PRIMARY_COST_ATR0, day, sym)
               for k, a in policies.items()}
    boot = economic_bootstrap(day, net, n_boot=n_boot)

    # Cost grid: gates MUST be recomputed at each cost (the threshold IS the cost).
    cost_grid = {}
    for c in COST_GRID:
        pol_c = policies_at(c)
        cost_grid[str(c)] = {k: strategy_metrics(a, r_trad, c, day, sym)
                             for k, a in pol_c.items()}

    grid_primary = cost_grid[str(PRIMARY_COST_ATR0)]
    for k in policies:
        for f in ["n_trades", "trade_rate", "gross_total_ATR0", "net_total_ATR0",
                  "net_EV_per_decision", "profit_factor"]:
            if not np.isclose(grid_primary[k][f], metrics[k][f], rtol=0.0, atol=1e-12,
                              equal_nan=True):
                raise SystemExit("STOP_PGM_NATIVE0D_PRIMARY_COST_GRID_PARITY_FAIL")

    for k in ["GATE0", "GATEA", "FLIPA"]:
        counts = [cost_grid[str(c)][k]["n_trades"] for c in COST_GRID]
        if any(counts[i + 1] > counts[i] for i in range(len(counts) - 1)):
            raise SystemExit(f"STOP_PGM_NATIVE0D_COST_GRID_MONOTONICITY_FAIL: {k} {counts}")
    base_counts = [cost_grid[str(c)]["BASE"]["n_trades"] for c in COST_GRID]
    if len(set(base_counts)) != 1:
        raise SystemExit(f"STOP_PGM_NATIVE0D_COST_GRID_MONOTONICITY_FAIL: BASE {base_counts}")

    q_edges = {}
    q_rows: List[Dict[str, Any]] = []
    for feat in QUINTILE_FEATURES:
        e = quintile_edges(econ_tr, feat)
        q_edges[feat] = e.tolist()
        q_rows += quintile_diagnostic(econ_ev, feat, e, w["eval"])

    return dict(harm_metrics=dict(O0=m0, OA=mA), payoff_metrics=dict(O0=p0, OA=pA),
                delta_harm_logloss=d_ll, delta_payoff_mse=d_mse,
                beta0=beta0, n_h1_train=len(h1_tr), n_h1_eval=len(h1_ev),
                n_h0_train=len(h0_tr), n_h0_eval=len(econ_ev),
                p_h_mean=float(np.mean(p_h_UE)),
                V0_mean=float(np.mean(V0)), VA_mean=float(np.mean(VA)),
                metrics=metrics, bootstrap=boot, cost_grid=cost_grid,
                quintile_edges=q_edges, quintiles=q_rows)


def run_smoke_test() -> None:
    print("=" * 50, flush=True)
    print("PGM-NATIVE-0D: SMOKE (wiring only)", flush=True)
    print("=" * 50, flush=True)
    t0 = time.perf_counter()
    prep = load_prepared_frame()
    aligned = prep["aligned"]
    print_reuse_map()
    print("[SMOKE] Fitting Window A samplers...", flush=True)
    fit_A = pgm.fit_samplers_for_window(pgm.WINDOWS[0], pgm.SAMPLE_PATH,
                                        pgm.TRANSITION_SAMPLE_PATH)
    scored_A = prepare_window_windowframe(aligned, fit_A, "scored_A")
    print("[SMOKE] Fitting Window B samplers...", flush=True)
    fit_B = pgm.fit_samplers_for_window(pgm.WINDOWS[1], pgm.SAMPLE_PATH,
                                        pgm.TRANSITION_SAMPLE_PATH)
    scored_B = prepare_window_windowframe(aligned, fit_B, "scored_B")

    dA = verify_window_score_owner(scored_A, fit_A, TB2_BLOCK)
    dB = verify_window_score_owner(scored_B, fit_B, TB3_BLOCK)
    print(f"[SMOKE] score-owner parity: WindowA(TB2)={dA:.3e} WindowB(TB3)={dB:.3e}")

    for blk, w, sf in [(TB2_BLOCK, pgm.WINDOWS[0], scored_A),
                       (TB3_BLOCK, pgm.WINDOWS[1], scored_B)]:
        print(f"[SMOKE] --- window {blk} ---")
        r = _run_window(sf, w, n_boot=SMOKE_BOOTSTRAP_N, eval_cap=SMOKE_EVAL_CAP)
        print(f"  H1 train={r['n_h1_train']} eval={r['n_h1_eval']} H0 train={r['n_h0_train']}")
        print(f"  harm O0 LL={r['harm_metrics']['O0']['log_loss']:.6f} "
              f"OA LL={r['harm_metrics']['OA']['log_loss']:.6f}")
        print(f"  payoff O0 MSE={r['payoff_metrics']['O0']['mse']:.6f} "
              f"OA MSE={r['payoff_metrics']['OA']['mse']:.6f}")
        print(f"  Delta_Harm_LogLoss={r['delta_harm_logloss']['point']:.6f} "
              f"CI=[{r['delta_harm_logloss']['ci95_lower']:.6f}, "
              f"{r['delta_harm_logloss']['ci95_upper']:.6f}]")
        print(f"  Delta_Payoff_MSE={r['delta_payoff_mse']['point']:.6f} "
              f"CI=[{r['delta_payoff_mse']['ci95_lower']:.6f}, "
              f"{r['delta_payoff_mse']['ci95_upper']:.6f}]")
        print(f"  beta0={r['beta0']:.6f} p_h_mean={r['p_h_mean']:.4f} "
              f"V0_mean={r['V0_mean']:.6f} VA_mean={r['VA_mean']:.6f}")
        for k in ["BASE", "GATE0", "GATEA", "FLIPA"]:
            mm = r["metrics"][k]
            bb = r["bootstrap"][k]
            print(f"    {k}: net_total={mm['net_total_ATR0']:.3f} "
                  f"netEV/dec={mm['net_EV_per_decision']:.6f} trades={mm['n_trades']} "
                  f"rate={mm['trade_rate']:.3f} boot_CI=[{bb['ci95_lower']:.6f}, {bb['ci95_upper']:.6f}]")
        for k in ["GATEA-BASE", "GATEA-GATE0"]:
            print(f"    {k}: point={r['bootstrap'][k]['point']:.6f} "
                  f"CI=[{r['bootstrap'][k]['ci95_lower']:.6f}, {r['bootstrap'][k]['ci95_upper']:.6f}]")
        for p in ["BASE", "GATE0", "GATEA", "FLIPA"]:
            row = " ".join(f"{c:g}={r['cost_grid'][str(c)][p]['n_trades']}" for c in COST_GRID)
            print(f"    cost-grid n_trades {p}: {row}")
    print("[SMOKE] SMOKE ONLY / NO SCIENTIFIC VERDICT", flush=True)
    print(f"[SMOKE COMPLETE] {time.perf_counter() - t0:.2f}s", flush=True)


# ===========================================================================
# Pre-fit integrity gates (NO production model fitting)
# ===========================================================================
def run_pre_fit_integrity_gates(obs: pd.DataFrame, aligned: pd.DataFrame) -> Dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT),
                                   text=True).strip()
    if subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"],
                      cwd=str(_REPO_ROOT), capture_output=True).returncode != 0:
        raise SystemExit("STOP_PGM_NATIVE0D_FREEZE_CHECK_FAIL")
    assert_allowed_blocks(obs)
    if "TB4" in set(aligned["block"].unique()) or "TB4" in set(obs["block"].unique()):
        raise SystemExit("STOP_PGM_NATIVE0D_FORBIDDEN_BLOCK: TB4")

    hashes = compute_artifact_hashes()
    aud = n0a.audit_decision_universe(obs)
    err = n0a.audit_atr0_owner_parity(obs, n0a.load_transition_truth_audit()["cur"])

    if len(A_COLS) != 8:
        raise SystemExit("STOP_PGM_NATIVE0D_A_COLS_NOT_EIGHT")
    if set(A_COLS) & set(outcome_base_num()):
        raise SystemExit("STOP_PGM_NATIVE0D_INCREMENTAL_COLUMN_COLLISION")
    if not _prefix_invariance_check():
        raise SystemExit("STOP_PGM_NATIVE0D_FEATURE_FUTURE_DEPENDENCE")
    for bad in ["hazard", "harm_flag", "pi", "r_trad_OC_ATR0", "target_mask"]:
        if bad in outcome_base_num() or bad in A_COLS:
            raise SystemExit(f"STOP_PGM_NATIVE0D_PREDICTOR_LEAK: {bad}")
    if len(aligned) != len(obs):
        raise SystemExit("STOP_PGM_NATIVE0D_ALIGN_ROWCOUNT_CHANGED")

    blocks = obs["block"].to_numpy()
    raw_counts = {b: int((blocks == b).sum()) for b in ALLOWED_BLOCKS}
    sb = compute_same_block_entry_valid(aligned)
    sb_counts = {b: int(((aligned["block"].to_numpy() == b) & sb).sum()) for b in ALLOWED_BLOCKS}
    return dict(head=head, hashes=hashes, aud=aud, atr0=err,
                raw_counts=raw_counts, same_block_counts=sb_counts)


# ===========================================================================
# Artifact assembly + writers
# ===========================================================================
COST_FIELDS = ["n_decisions", "n_trades", "trade_rate", "gross_total_ATR0", "net_total_ATR0",
               "net_EV_per_decision", "net_EV_per_trade", "profit_factor", "win_rate",
               "daily_sharpe_annualized", "max_drawdown_ATR0"]


def _outcome_rows(block: str, r: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for task, variants in [("harm", r["harm_metrics"]), ("payoff", r["payoff_metrics"])]:
        for v in ["O0", "OA"]:
            m = variants[v]
            row = dict(block=block, task=task, variant=v, n=int(r["n_h1_eval"]),
                       log_loss=np.nan, brier=np.nan, roc_auc=np.nan, pr_auc=np.nan,
                       mse=np.nan, mae=np.nan, spearman=np.nan)
            if task == "harm":
                row.update(log_loss=m["log_loss"], brier=m["brier"],
                           roc_auc=m["roc_auc"], pr_auc=m["pr_auc"])
            else:
                row.update(mse=m["mse"], mae=m["mae"], spearman=m["spearman"])
            rows.append(row)
    return rows


def _info_rows(block: str, r: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for metric, key in [("Delta_Harm_LogLoss", "delta_harm_logloss"),
                        ("Delta_Payoff_MSE", "delta_payoff_mse")]:
        b = r[key]
        out.append(dict(block=block, metric=metric, point=b["point"],
                        ci95_lower=b["ci95_lower"], ci95_upper=b["ci95_upper"],
                        p_pos=b["p_pos"]))
    return out


def _strategy_rows(block: str, r: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [dict(block=block, policy=p, **{f: r["metrics"][p][f] for f in STRATEGY_FIELDS})
            for p in ["BASE", "GATE0", "GATEA", "FLIPA"]]


def _econ_rows(block: str, r: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for it in ["BASE", "GATE0", "GATEA", "FLIPA", "GATEA-BASE", "GATEA-GATE0"]:
        b = r["bootstrap"][it]
        out.append(dict(block=block, item=it, point=b["point"],
                        ci95_lower=b["ci95_lower"], ci95_upper=b["ci95_upper"],
                        p_pos=b["p_pos"]))
    return out


def _cost_rows(block: str, r: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for c in COST_GRID:
        for p in ["BASE", "GATE0", "GATEA", "FLIPA"]:
            m = r["cost_grid"][str(c)][p]
            rows.append(dict(block=block, cost=c, policy=p, **{f: m[f] for f in COST_FIELDS}))
    return rows


def _symbol_rows(block: str, r: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for p in ["BASE", "GATE0", "GATEA", "FLIPA"]:
        by_sym = r["metrics"][p]["by_symbol"]
        for s in n0a.EXPECTED_SYMBOLS:
            v = by_sym.get(s)
            rows.append(dict(block=block, policy=p, symbol=s,
                             trade_count=int(v["trade_count"]) if v else 0,
                             net_total_ATR0=float(v["net_total"]) if v else 0.0))
    return rows


def _quintile_rows(block: str, r: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [dict(block=row["block"], feature=row["feature"], bin=row["bin"], n=row["n"],
                 H1_prevalence=row["H1_prevalence"], harm_rate=row["harm_rate"],
                 mean_pi=row["mean_pi"]) for row in r["quintiles"]]


def _write_full_artifacts(summary: Dict[str, Any], out_dir: Path) -> None:
    out = Path(out_dir)
    r2, r3 = summary["TB2"], summary["TB3"]
    outcome = _outcome_rows("TB2", r2) + _outcome_rows("TB3", r3)
    info = _info_rows("TB2", r2) + _info_rows("TB3", r3)
    strat = _strategy_rows("TB2", r2) + _strategy_rows("TB3", r3)
    econ = _econ_rows("TB2", r2) + _econ_rows("TB3", r3)
    cost = _cost_rows("TB2", r2) + _cost_rows("TB3", r3)
    sym = _symbol_rows("TB2", r2) + _symbol_rows("TB3", r3)
    quint = _quintile_rows("TB2", r2) + _quintile_rows("TB3", r3)

    pd.DataFrame(outcome)[["block", "task", "variant", "n", "log_loss", "brier",
                           "roc_auc", "pr_auc", "mse", "mae", "spearman"]].to_csv(
        out / f"{PREFIX}_outcome_metrics.csv", index=False)
    pd.DataFrame(info)[["block", "metric", "point", "ci95_lower", "ci95_upper", "p_pos"]].to_csv(
        out / f"{PREFIX}_information_bootstrap.csv", index=False)
    pd.DataFrame(strat)[["block", "policy"] + STRATEGY_FIELDS].to_csv(
        out / f"{PREFIX}_strategy_metrics.csv", index=False)
    pd.DataFrame(econ)[["block", "item", "point", "ci95_lower", "ci95_upper", "p_pos"]].to_csv(
        out / f"{PREFIX}_economic_bootstrap.csv", index=False)
    pd.DataFrame(cost)[["block", "cost", "policy"] + COST_FIELDS].to_csv(
        out / f"{PREFIX}_cost_grid.csv", index=False)
    pd.DataFrame(sym)[["block", "policy", "symbol", "trade_count", "net_total_ATR0"]].to_csv(
        out / f"{PREFIX}_symbol_metrics.csv", index=False)
    pd.DataFrame(quint)[["block", "feature", "bin", "n", "H1_prevalence", "harm_rate",
                         "mean_pi"]].to_csv(
        out / f"{PREFIX}_acceleration_quintiles.csv", index=False)


# ===========================================================================
# Final artifact parity (fail-closed)
# ===========================================================================
def validate_output_artifacts(summary: Dict[str, Any], out_dir: Path) -> bool:
    out = Path(out_dir)
    ATOL = 1e-12

    def _fail(msg: str):
        raise SystemExit(f"STOP_PGM_NATIVE0D_OUTPUT_PARITY_FAIL: {msg}")

    def _close(a, b, atol=ATOL) -> bool:
        return bool(np.isclose(float(a), float(b), rtol=0.0, atol=atol, equal_nan=True))

    for fn in ARTIFACT_FILES:
        p = out / fn
        if (not p.exists()) or p.stat().st_size == 0:
            _fail(f"missing/empty {fn}")

    # governance JSON parity
    js = json.loads((out / f"{PREFIX}_formal_summary.json").read_text())
    for k in ["EXPERIMENT_NAME", "EXPERIMENT_SCOPE", "base_sha", "run_head",
              "sample_artifact_sha256", "transition_artifact_sha256",
              "n_all_obs", "n_H0", "n_H1", "symbols", "blocks",
              "max_abs_atr0_owner_error", "A_COLS", "OUTCOME_BASE_NUM", "OUTCOME_CAT",
              "PRIMARY_COST_ATR0", "COST_GRID", "BOOTSTRAP_N", "BOOTSTRAP_SEED",
              "information_verdict", "economic_verdict"]:
        if k not in summary:
            _fail(f"summary missing {k}")
        if k not in js or js[k] != summary[k]:
            _fail(f"formal_summary mismatch on {k}")

    # outcome metrics: 8 rows
    om = pd.read_csv(out / f"{PREFIX}_outcome_metrics.csv")
    if len(om) != 8:
        _fail(f"outcome_metrics rows={len(om)} != 8")
    for blk in ["TB2", "TB3"]:
        for task, key in [("harm", "harm_metrics"), ("payoff", "payoff_metrics")]:
            for v in ["O0", "OA"]:
                row = om[(om["block"] == blk) & (om["task"] == task) & (om["variant"] == v)]
                if len(row) != 1:
                    _fail(f"outcome row {blk}/{task}/{v}")
                row = row.iloc[0]
                m = summary[blk][key][v]
                if int(row["n"]) != int(summary[blk]["n_h1_eval"]):
                    _fail(f"outcome n mismatch {blk}/{task}/{v}")
                fields = (["log_loss", "brier", "roc_auc", "pr_auc"] if task == "harm"
                          else ["mse", "mae", "spearman"])
                for f in fields:
                    if not _close(row[f], m[f]):
                        _fail(f"outcome {blk}/{task}/{v}.{f} mismatch")

    # information bootstrap: 4 rows + verdict re-derivation
    ib = pd.read_csv(out / f"{PREFIX}_information_bootstrap.csv")
    if len(ib) != 4:
        _fail(f"information_bootstrap rows={len(ib)} != 4")
    for blk in ["TB2", "TB3"]:
        for metric, key in [("Delta_Harm_LogLoss", "delta_harm_logloss"),
                            ("Delta_Payoff_MSE", "delta_payoff_mse")]:
            row = ib[(ib["block"] == blk) & (ib["metric"] == metric)]
            if len(row) != 1:
                _fail(f"info row {blk}/{metric}")
            row = row.iloc[0]
            b = summary[blk][key]
            for f in ["point", "ci95_lower", "ci95_upper", "p_pos"]:
                if not _close(row[f], b[f]):
                    _fail(f"info {blk}/{metric}.{f} mismatch")
    v_info = determine_information_verdict(
        summary["TB3"]["delta_harm_logloss"], summary["TB3"]["delta_payoff_mse"])
    if v_info != summary["information_verdict"]:
        _fail("information verdict not reproducible from TB3 rows")

    # strategy metrics: 8 rows
    sm = pd.read_csv(out / f"{PREFIX}_strategy_metrics.csv")
    if len(sm) != 8:
        _fail(f"strategy_metrics rows={len(sm)} != 8")
    for blk in ["TB2", "TB3"]:
        for p in ["BASE", "GATE0", "GATEA", "FLIPA"]:
            row = sm[(sm["block"] == blk) & (sm["policy"] == p)]
            if len(row) != 1:
                _fail(f"strategy row {blk}/{p}")
            row = row.iloc[0]
            for f in STRATEGY_FIELDS:
                if not _close(row[f], summary[blk]["metrics"][p][f]):
                    _fail(f"strategy {blk}/{p}.{f} mismatch")

    # economic bootstrap: 12 rows + verdict re-derivation
    eb = pd.read_csv(out / f"{PREFIX}_economic_bootstrap.csv")
    if len(eb) != 12:
        _fail(f"economic_bootstrap rows={len(eb)} != 12")
    for blk in ["TB2", "TB3"]:
        for it in ["BASE", "GATE0", "GATEA", "FLIPA", "GATEA-BASE", "GATEA-GATE0"]:
            row = eb[(eb["block"] == blk) & (eb["item"] == it)]
            if len(row) != 1:
                _fail(f"econ row {blk}/{it}")
            row = row.iloc[0]
            b = summary[blk]["bootstrap"][it]
            for f in ["point", "ci95_lower", "ci95_upper", "p_pos"]:
                if not _close(row[f], b[f]):
                    _fail(f"econ {blk}/{it}.{f} mismatch")
    v_econ = determine_economic_verdict(summary["TB2"]["bootstrap"], summary["TB3"]["bootstrap"])
    if v_econ != summary["economic_verdict"]:
        _fail("economic verdict not reproducible from bootstrap rows")

    # cost grid: 48 rows + monotonic + .01 parity
    cg = pd.read_csv(out / f"{PREFIX}_cost_grid.csv")
    if len(cg) != 48:
        _fail(f"cost_grid rows={len(cg)} != 48")
    for blk in ["TB2", "TB3"]:
        for p in ["GATE0", "GATEA", "FLIPA"]:
            counts = [int(cg[(cg["block"] == blk) & (cg["policy"] == p)
                             & (cg["cost"] == c)].iloc[0]["n_trades"]) for c in COST_GRID]
            if any(counts[i + 1] > counts[i] for i in range(len(counts) - 1)):
                _fail(f"cost_grid monotonic {blk}/{p} {counts}")
        bc = [int(cg[(cg["block"] == blk) & (cg["policy"] == "BASE")
                     & (cg["cost"] == c)].iloc[0]["n_trades"]) for c in COST_GRID]
        if len(set(bc)) != 1:
            _fail(f"cost_grid BASE invariant {blk} {bc}")
        for p in ["BASE", "GATE0", "GATEA", "FLIPA"]:
            row = cg[(cg["block"] == blk) & (cg["policy"] == p)
                     & (cg["cost"] == PRIMARY_COST_ATR0)]
            if len(row) != 1:
                _fail(f"cost_grid .01 row {blk}/{p}")
            row = row.iloc[0]
            for f in COST_FIELDS:
                if not _close(row[f], summary[blk]["metrics"][p][f]):
                    _fail(f"cost_grid .01 {blk}/{p}.{f} mismatch")

    # symbol metrics: 120 rows + totals closure
    sy = pd.read_csv(out / f"{PREFIX}_symbol_metrics.csv")
    if len(sy) != 120:
        _fail(f"symbol_metrics rows={len(sy)} != 120")
    for blk in ["TB2", "TB3"]:
        for p in ["BASE", "GATE0", "GATEA", "FLIPA"]:
            g = sy[(sy["block"] == blk) & (sy["policy"] == p)]
            if len(g) != 15:
                _fail(f"symbol group {blk}/{p} n={len(g)} != 15")
            if not np.isclose(float(g["net_total_ATR0"].sum()),
                              summary[blk]["metrics"][p]["net_total_ATR0"], atol=1e-10, rtol=0):
                _fail(f"symbol net_total closure {blk}/{p}")
            if int(g["trade_count"].sum()) != int(summary[blk]["metrics"][p]["n_trades"]):
                _fail(f"symbol trade_count closure {blk}/{p}")

    # quintiles
    q = pd.read_csv(out / f"{PREFIX}_acceleration_quintiles.csv")
    if set(q["feature"].unique()) - set(QUINTILE_FEATURES):
        _fail("quintile unexpected feature")
    if not set(q["bin"].astype(int)) <= set(range(5)):
        _fail("quintile unexpected bin")
    if set(q["block"].unique()) - {"TB2", "TB3"}:
        _fail("quintile unexpected block")
    return True


def run_full_exploratory(output_dir: Optional[Path] = None) -> Dict[str, Any]:
    require_full_authorization()

    out_dir = Path(output_dir) if output_dir is not None else FULL_RESULT_DIR
    for fn in ARTIFACT_FILES:
        if (out_dir / fn).exists():
            raise SystemExit("STOP_PGM_NATIVE0D_FORMAL_ARTIFACT_ALREADY_EXISTS")

    prep = load_prepared_frame()
    obs = n0a.load_observed_decision_universe()
    aligned = prep["aligned"]
    gates = run_pre_fit_integrity_gates(obs, aligned)

    # Window A: fit exactly once
    fit_A = pgm.fit_samplers_for_window(pgm.WINDOWS[0], pgm.SAMPLE_PATH,
                                        pgm.TRANSITION_SAMPLE_PATH)
    scored_A = prepare_window_windowframe(aligned, fit_A, "formal_scored_A")
    dA = verify_window_score_owner(scored_A, fit_A, TB2_BLOCK)
    finA = audit_acceleration_finite(scored_A)
    if not finA["all_finite"]:
        raise SystemExit("STOP_PGM_NATIVE0D_ACCELERATION_NON_FINITE")

    # Window B: fit exactly once
    fit_B = pgm.fit_samplers_for_window(pgm.WINDOWS[1], pgm.SAMPLE_PATH,
                                        pgm.TRANSITION_SAMPLE_PATH)
    scored_B = prepare_window_windowframe(aligned, fit_B, "formal_scored_B")
    dB = verify_window_score_owner(scored_B, fit_B, TB3_BLOCK)
    finB = audit_acceleration_finite(scored_B)
    if not finB["all_finite"]:
        raise SystemExit("STOP_PGM_NATIVE0D_ACCELERATION_NON_FINITE")

    r2 = _run_window(scored_A, pgm.WINDOWS[0], n_boot=BOOTSTRAP_N, eval_cap=None)
    r3 = _run_window(scored_B, pgm.WINDOWS[1], n_boot=BOOTSTRAP_N, eval_cap=None)

    info_verdict = determine_information_verdict(r3["delta_harm_logloss"], r3["delta_payoff_mse"])
    econ_verdict = determine_economic_verdict(r2["bootstrap"], r3["bootstrap"])

    aud = gates["aud"]
    summary = {
        "EXPERIMENT_NAME": EXPERIMENT_NAME,
        "EXPERIMENT_SCOPE": EXPERIMENT_SCOPE,
        "base_sha": BASE_SHA,
        "run_head": gates["head"],
        "sample_artifact_sha256": gates["hashes"]["sample_artifact_sha256"],
        "transition_artifact_sha256": gates["hashes"]["transition_artifact_sha256"],
        "n_all_obs": int(aud["n_all_obs"]),
        "n_H0": int(aud["n_hazard0"]),
        "n_H1": int(aud["n_hazard1"]),
        "symbols": sorted(aud["symbols"]),
        "blocks": sorted(obs["block"].unique().tolist()),
        "same_block_entry_counts": gates["same_block_counts"],
        "raw_block_counts": gates["raw_counts"],
        "max_abs_atr0_owner_error": float(gates["atr0"]),
        "A_COLS": A_COLS,
        "OUTCOME_BASE_NUM": outcome_base_num(),
        "OUTCOME_CAT": outcome_cat(),
        "PRIMARY_COST_ATR0": PRIMARY_COST_ATR0,
        "COST_GRID": COST_GRID,
        "BOOTSTRAP_N": BOOTSTRAP_N,
        "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
        "cluster_owner": CLUSTER_OWNER,
        "WindowA_score_owner_max_abs_diff": float(dA),
        "WindowB_score_owner_max_abs_diff": float(dB),
        "WindowA_acceleration_finite": bool(finA["all_finite"]),
        "WindowB_acceleration_finite": bool(finB["all_finite"]),
        "TB2": r2,
        "TB3": r3,
        "information_verdict": info_verdict,
        "economic_verdict": econ_verdict,
        "quintile_frozen_edges": {"TB2": r2["quintile_edges"], "TB3": r3["quintile_edges"]},
        "artifact_files": ARTIFACT_FILES,
        "known_limitations": [
            "TB2/TB3 were previously inspected across multiple research rounds; this is exploratory strategy development, not a pristine holdout.",
            "TB4 is forbidden; this experiment never touches TB4.",
            "The frozen PGM sample excludes cross-block episodes.",
            "event_mask==0 censored episodes are absent from the frozen sample.",
            "Economic return is a single 5m next-open -> next-close bar.",
            "Cost is ATR0-normalized friction, not a real contract commission/bid-ask/slippage model.",
            "The H1 outcome head is fit only on realized H1 rows and then used as a conditional branch on all current states.",
            "score_mu used by the outcome models comes from the same training-window production MC fit, not out-of-fold stacking.",
            "Models stay fixed Logistic / Ridge; no nonlinear model search.",
            "No threshold / symbol / holding / cost / feature tuning.",
        ],
    }

    _write_full_artifacts(summary, out_dir)
    # FINAL summary must be written BEFORE the final parity check
    (out_dir / f"{PREFIX}_formal_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    validate_output_artifacts(summary, out_dir)
    print(f"[FULL] information_verdict={info_verdict}", flush=True)
    print(f"[FULL] economic_verdict={econ_verdict}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full-exploratory", action="store_true")
    args = ap.parse_args()
    if args.full_exploratory:
        run_full_exploratory()
    elif args.smoke:
        run_smoke_test()
    else:
        run_audit_only()


if __name__ == "__main__":
    main()
