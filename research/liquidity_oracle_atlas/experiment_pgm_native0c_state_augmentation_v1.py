"""
experiment_pgm_native0c_state_augmentation_v1.py
================================================

PGM-NATIVE-0C: Incremental State Representation Audit.

Tests whether adding two CAUSAL incremental state blocks to the FROZEN production PGM
provides genuine conditional incremental information:

    S_t          -> PGM0     (baseline)
    S_t + U_t    -> PGM_U    (ablation)
    S_t + E_t    -> PGM_E    (ablation)
    S_t + U_t + E_t -> PGM_UE (SOLE PRIMARY augmentation)

U_t = SURVIVAL_PATH_STATE  (duration / per-bar path motion / excursion / efficiency / net speed)
E_t = EXHAUSTION_STATE     (speed decay / range decay / local efficiency / direction-change drift)

Information is measured as strict OOS improvement in:
    Delta_LogLoss = LogLoss_PGM0 - LogLoss_variant      (terminal hazard)
    Delta_JNLL    = JNLL_PGM0    - JNLL_variant         (H0 conditional transition)
Positive = the new state adds information beyond the frozen PGM.

This is NOT a new trading strategy. No economic policy is produced from augmented models.

SCOPE
-----
EXPERIMENT_SCOPE = EXPLORATORY_PGM_STATE_AUGMENTATION_ON_PREVIOUSLY_INSPECTED_TB3
TB3 is temporal OOS relative to fit, but was previously inspected in 0A/0B.
Never label PRISTINE_HOLDOUT / FINAL_CONFIRMATION / FINAL_ALPHA / PROFITABLE_STRATEGY /
LIVE_READY / FINAL_STRATEGY.

REUSE CONTRACT
--------------
All model logic is owned by production: pgm.fit_samplers_for_window, FittedTerminalSampler,
FittedTransitionSampler, safe_transform, lag._make_ct, base.fit_state_heads,
base.fit_state_count_head, base.fit_constant_count_head, rep._node_eval_nll,
pgm.fit_count_occurrence_models, pm.make_pipeline. Research code only ADDS feature
columns and calls the production owners again.
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

import research.liquidity_oracle_atlas.experiment_dynamic_pgm1c_free_run_rollout_v1 as pgm
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a1b_count_magnitude_closure_v1 as base
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a2_lag_closure_v1 as lag
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a2c_representation_control_v1 as rep
import research.liquidity_oracle_atlas.experiment_path0_episode_path_memory_v1 as pm
import research.liquidity_oracle_atlas.experiment_pgm_bar0_path_smc_v1 as pbar
import research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 as ex0
import research.liquidity_oracle_atlas.experiment_pgm_native0a_one_step_alpha_v1 as n0a
import research.liquidity_oracle_atlas.experiment_pgm_native0b_hazard_reliability_v1 as n0b


# ===========================================================================
# Governance constants
# ===========================================================================
BASE_SHA = "fc3c34f27ff4efa061a268162af709a34d74d0f1"
EXPERIMENT_NAME = "PGM-NATIVE-0C -- Incremental State Representation Audit"
EXPERIMENT_SCOPE = "EXPLORATORY_PGM_STATE_AUGMENTATION_ON_PREVIOUSLY_INSPECTED_TB3"
PREFIX = "pgm_native0c1"

FORBIDDEN_VERDICTS = {
    "PRISTINE_HOLDOUT", "FINAL_CONFIRMATION", "FINAL_ALPHA",
    "PROFITABLE_STRATEGY", "LIVE_READY", "FINAL_STRATEGY",
}

VERDICT_STRINGS = {
    "JOINTLY": "PGM_INCREMENTAL_STATE_SUPPORTED_JOINTLY_EXPLORATORY",
    "TERMINATION_ONLY": "PGM_INCREMENTAL_STATE_SUPPORTED_FOR_TERMINATION_ONLY_EXPLORATORY",
    "TRANSITION_ONLY": "PGM_INCREMENTAL_STATE_SUPPORTED_FOR_TRANSITION_ONLY_EXPLORATORY",
    "NOT_SUPPORTED": "PGM_INCREMENTAL_STATE_NOT_SUPPORTED_EXPLORATORY",
}

BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20260915
EPS = 1e-9
P_CLIP = 1e-15

PRIMARY_TERMINAL_HEAD = "T2_STATE_PHI_MEM"
PRIMARY_TRANSITION_HEAD = "MC_STATE_CURREENCODING"

TB2_BLOCK = "TB2"
TB3_BLOCK = "TB3"

U_COLS: List[str] = [
    "u_log_age", "u_tv_per_bar", "u_excursion_per_bar",
    "u_path_efficiency", "u_net_speed",
]
E_COLS: List[str] = [
    "e_speed_decay_3", "e_speed_ratio_5", "e_range_decay_3",
    "e_local_eff_3", "e_local_eff_5", "e_dcr_change",
]
INCREMENTAL_COLS: List[str] = U_COLS + E_COLS

# Row-level identity parity for the (symbol, episode_id, seq) transition join.
# These current-state primitives exist on BOTH the H0 observation rows and the frozen
# transition artifact, so they prove each seq really corresponds to the same bar --
# not merely that row counts happen to match.
JOIN_PARITY_CORE_COLS: List[str] = [
    "cur_up_distance_R", "cur_down_distance_R", "path_total_variation_R",
    "path_max_up_excursion_R", "path_max_down_excursion_R",
    "path_direction_change_rate", "path_last_return_R",
    "path_current_bar_range_R",
]
MIN_JOIN_PARITY_COLS = 6
JOIN_PARITY_RTOL = 1e-6
JOIN_PARITY_ATOL = 2e-6
JOIN_PARITY_SUFFIX = "__obs"

# The four pre-registered variants ONLY. PGM_UE is the sole PRIMARY.
VARIANTS: Dict[str, List[str]] = {
    "PGM0": [],
    "PGM_U": list(U_COLS),
    "PGM_E": list(E_COLS),
    "PGM_UE": list(U_COLS) + list(E_COLS),
}
PRIMARY_VARIANT = "PGM_UE"

U_PRIMITIVES = [
    "symbol", "episode_id", "block", "bar_t", "start_bar",
    "cur_up_distance_R", "path_total_variation_R",
    "path_max_up_excursion_R", "path_max_down_excursion_R",
]
E_PRIMITIVES = [
    "path_last_return_R", "path_current_bar_range_R",
    "path_direction_change_rate",
]
FUTURE_TOKENS = ["shift(-1)", "next_", "future_", "final_episode", "remaining_bars",
                 "terminal_distance"]

EXPECTED_TRANSITION_ROWS = 321727
EXPECTED_SYMBOLS = n0a.EXPECTED_SYMBOLS

# The 9 planned full-run artifacts.
ARTIFACT_FILES: List[str] = [
    f"{PREFIX}_terminal_metrics.csv",
    f"{PREFIX}_terminal_bootstrap.csv",
    f"{PREFIX}_transition_metrics.csv",
    f"{PREFIX}_transition_bootstrap.csv",
    f"{PREFIX}_mechanism_grid.csv",
    f"{PREFIX}_conditional_contrast.csv",
    f"{PREFIX}_age_hazard.csv",
    f"{PREFIX}_h1_harm.csv",
    f"{PREFIX}_formal_summary.json",
]
FULL_RESULT_DIR = _REPO_ROOT / "research" / "analysis_results" / "local_liquidity_transition_v0"


# ===========================================================================
# Block U / E feature construction
# ===========================================================================
def add_incremental_state_features(obs: pd.DataFrame) -> pd.DataFrame:
    """Add causal U (survival/path) and E (exhaustion) state blocks.

    Uses ONLY information available at or before t. Sorted by (symbol, episode_id, bar_t).
    """
    x = obs.sort_values(["symbol", "episode_id", "bar_t"], kind="stable").copy()

    g = x.groupby(["symbol", "episode_id"], sort=False, group_keys=False)

    age = x["bar_t"].to_numpy(np.int64) - x["start_bar"].to_numpy(np.int64)
    if np.any(age < 0):
        raise SystemExit("STOP_PGM_NATIVE0C_NEGATIVE_EPISODE_AGE")
    denom_age = age.astype(np.float64) + 1.0

    x["u_log_age"] = np.log1p(age.astype(np.float64))

    start_up = g["cur_up_distance_R"].transform("first")
    net_up = x["cur_up_distance_R"] - start_up

    x["u_tv_per_bar"] = x["path_total_variation_R"] / denom_age
    x["u_excursion_per_bar"] = (
        x["path_max_up_excursion_R"] + x["path_max_down_excursion_R"]
    ) / denom_age
    x["u_path_efficiency"] = np.abs(net_up) / (x["path_total_variation_R"] + EPS)
    x["u_net_speed"] = net_up / denom_age

    x["_absret"] = np.abs(x["path_last_return_R"].to_numpy(dtype=np.float64))

    prev_abs3 = g["_absret"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    prev_abs5 = g["_absret"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    prev_abs3 = prev_abs3.fillna(x["_absret"])
    prev_abs5 = prev_abs5.fillna(x["_absret"])

    x["e_speed_decay_3"] = x["_absret"] - prev_abs3
    x["e_speed_ratio_5"] = x["_absret"] / (prev_abs5 + EPS)

    prev_rng3 = g["path_current_bar_range_R"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    prev_rng3 = prev_rng3.fillna(x["path_current_bar_range_R"])
    x["e_range_decay_3"] = x["path_current_bar_range_R"] - prev_rng3

    signed3 = g["path_last_return_R"].transform(
        lambda s: s.rolling(3, min_periods=1).sum())
    abs3 = g["_absret"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    x["e_local_eff_3"] = np.abs(signed3) / (abs3 + EPS)

    signed5 = g["path_last_return_R"].transform(
        lambda s: s.rolling(5, min_periods=1).sum())
    abs5 = g["_absret"].transform(lambda s: s.rolling(5, min_periods=1).sum())
    x["e_local_eff_5"] = np.abs(signed5) / (abs5 + EPS)

    prev_dcr = g["path_direction_change_rate"].shift(1)
    prev_dcr = prev_dcr.fillna(x["path_direction_change_rate"])
    x["e_dcr_change"] = x["path_direction_change_rate"] - prev_dcr

    x = x.drop(columns=["_absret"])
    return x


def audit_incremental_features_finite(df: pd.DataFrame) -> Dict[str, Any]:
    bad = {}
    for c in INCREMENTAL_COLS:
        v = df[c].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(v)):
            bad[c] = int(np.sum(~np.isfinite(v)))
    return dict(non_finite_counts=bad, all_finite=len(bad) == 0)


def attach_decision_day(obs: pd.DataFrame, bars_by_sym: Dict[str, Any]) -> pd.DataFrame:
    """Attach decision_day = raw-bar trading day of the current state bar (bar_t).

    Cluster identifier ONLY. Must never enter T2 predictors, MC predictors, or U/E.
    """
    out = obs.copy()
    dd = np.full(len(out), np.datetime64("NaT"), dtype="datetime64[us]")
    bar_t_all = out["bar_t"].to_numpy(dtype=np.int64)
    for sym, idx in out.groupby("symbol", sort=False).indices.items():
        bars = bars_by_sym[sym]
        n = int(bars["n"])
        t = bar_t_all[idx]
        if np.any(t < 0) or np.any(t >= n):
            raise SystemExit("STOP_PGM_NATIVE0C_DECISION_DAY_OOB")
        dd[idx] = bars["day"][t]
    out["decision_day"] = dd
    return out


def merge_incremental_features_into_transition(
    trans: pd.DataFrame, feat: pd.DataFrame
) -> pd.DataFrame:
    """Attach U/E to the H0 conditional transition sample.

    The frozen transition artifact has NO bar_t column (verified), so the join uses
    (symbol, episode_id, within-episode sequence order) -- the within-episode cumcount
    of obs rows sorted by bar_t and of transition rows in frozen file order.
    Row count is hard-preserved and unmatched is hard-zero.
    """
    h0 = feat[feat["hazard"] == 0].copy()
    h0["seq"] = h0.groupby(["symbol", "episode_id"], sort=False).cumcount()
    tr = trans.copy()
    tr["seq"] = tr.groupby(["symbol", "episode_id"], sort=False).cumcount()

    parity_cols = [c for c in JOIN_PARITY_CORE_COLS if c in tr.columns and c in h0.columns]
    if len(parity_cols) < MIN_JOIN_PARITY_COLS:
        raise SystemExit(
            f"STOP_PGM_NATIVE0C_TRANSITION_JOIN_PARITY_COLUMNS_MISSING: "
            f"found {len(parity_cols)} < {MIN_JOIN_PARITY_COLS}")

    carry = ["symbol", "episode_id", "seq"] + INCREMENTAL_COLS + ["bar_t", "start_bar"]
    if "decision_day" in h0.columns:
        carry = carry + ["decision_day"]
    right = h0[carry + parity_cols].copy()
    right = right.rename(columns={c: c + JOIN_PARITY_SUFFIX for c in parity_cols})

    merged = tr.merge(
        right,
        on=["symbol", "episode_id", "seq"],
        how="left",
        validate="one_to_one",
        indicator="_merge",
    )
    n_unmatched = int((merged["_merge"] != "both").sum())
    if len(merged) != len(trans):
        raise SystemExit("STOP_PGM_NATIVE0C_TRANSITION_JOIN_ROWCOUNT_CHANGED")
    if n_unmatched != 0:
        raise SystemExit(f"STOP_PGM_NATIVE0C_TRANSITION_JOIN_UNMATCHED: {n_unmatched}")
    if len(merged) != EXPECTED_TRANSITION_ROWS:
        raise SystemExit(
            f"STOP_PGM_NATIVE0C_TRANSITION_ROWCOUNT_MISMATCH: {len(merged)} != {EXPECTED_TRANSITION_ROWS}")

    # Row-level identity parity: the obs-side payload must equal the transition payload
    # for every shared current-state primitive. This upgrades the join from
    # "row counts match" to "each seq is the same current state".
    per_col = {}
    ok = True
    for c in parity_cols:
        a = merged[c].to_numpy(dtype=np.float64)
        b = merged[c + JOIN_PARITY_SUFFIX].to_numpy(dtype=np.float64)
        per_col[c] = float(np.max(np.abs(a - b)))
        if not np.allclose(a, b, rtol=JOIN_PARITY_RTOL, atol=JOIN_PARITY_ATOL, equal_nan=True):
            ok = False
    overall = float(max(per_col.values())) if per_col else 0.0
    if not ok:
        raise SystemExit(
            f"STOP_PGM_NATIVE0C_TRANSITION_JOIN_STATE_PARITY_FAIL: {per_col}")

    out = merged.drop(columns=["_merge", "seq"] + [c + JOIN_PARITY_SUFFIX for c in parity_cols])
    out.attrs["join_parity"] = dict(
        parity_columns=parity_cols, per_column_max_abs_diff=per_col,
        overall_max_abs_diff=overall,
        rtol=JOIN_PARITY_RTOL, atol=JOIN_PARITY_ATOL, passed=True,
    )
    return out


# ===========================================================================
# Terminal variant fitter (production owner only)
# ===========================================================================
def fit_terminal_hazard_variant(
    train_obs: pd.DataFrame, eval_obs: pd.DataFrame, extra_cols: Sequence[str]
) -> Dict[str, Any]:
    num_cols = list(pgm.T2_NUM) + list(extra_cols)
    cat_cols = list(pgm.CAT)
    cols = num_cols + cat_cols
    pipe = pm.make_pipeline(num_cols, cat_cols)
    pipe.fit(train_obs[cols], train_obs["hazard"].to_numpy(np.int64))
    p_eval = np.asarray(pipe.predict_proba(eval_obs[cols])[:, 1], dtype=np.float64)

    y = eval_obs["hazard"].to_numpy(np.int64)
    p = np.clip(p_eval, P_CLIP, 1.0 - P_CLIP)
    row_logloss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    row_brier = (p_eval - y) ** 2
    return dict(p_eval=p_eval, row_logloss=row_logloss, row_brier=row_brier,
                pipeline=pipe, num_cols=num_cols)


def evaluate_terminal_variant(fit: Dict[str, Any], hazard: np.ndarray) -> Dict[str, float]:
    y = np.asarray(hazard, dtype=np.int64)
    p = fit["p_eval"]
    ll = float(np.mean(fit["row_logloss"]))
    brier = float(np.mean(fit["row_brier"]))
    auc = float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y, p)) \
        if len(np.unique(y)) > 1 else 0.5
    pr_auc = float(__import__("sklearn.metrics", fromlist=["average_precision_score"])
                   .average_precision_score(y, p)) if len(np.unique(y)) > 1 else float(np.mean(y))
    return dict(log_loss=ll, brier=brier, roc_auc=auc, pr_auc=pr_auc)


# ===========================================================================
# Transition variant fitter (production owners only)
# ===========================================================================
def fit_transition_variant(
    train_trans: pd.DataFrame,
    eval_trans: pd.DataFrame,
    extra_cols: Sequence[str],
    baseline_design_cols: Sequence[str],
    tag: str = "AUG",
) -> Dict[str, Any]:
    num_cols = list(baseline_design_cols) + list(extra_cols)

    Zc_tr = train_trans[base.ALL_Z_COLS].to_numpy(np.float32)
    Zc_ev = eval_trans[base.ALL_Z_COLS].to_numpy(np.float32)
    yd_tr = train_trans[base.DISC_Z].to_numpy(np.int64)
    yd_ev = eval_trans[base.DISC_Z].to_numpy(np.int64)
    Yc_tr = train_trans[base.COUNT_Z].to_numpy(np.int64)
    Yc_ev = eval_trans[base.COUNT_Z].to_numpy(np.int64)

    base.configure_child_semantics(agezero_deterministic=True)
    k0_count = base.fit_constant_count_head(Yc_tr, Yc_ev)

    ct = lag._make_ct(num_cols)
    Xtr_sparse = ct.fit_transform(train_trans)
    Xtr = Xtr_sparse.astype(np.float32)
    Xev_sparse = ct.transform(eval_trans)
    Xev = Xev_sparse.astype(np.float32)

    k = base.fit_state_heads(Xtr, Xev, Zc_tr, Zc_ev, yd_tr, yd_ev)
    kc = base.fit_state_count_head(Xtr, Yc_tr, Xev, Yc_ev,
                                   constant_rates=k0_count["constant_rates"])
    occ_models = pgm.fit_count_occurrence_models(Xtr, Yc_tr, Xev, Yc_ev, tag)

    cont = rep._node_eval_nll(k["nodes"])
    disc = np.asarray(k["disc_ev"], dtype=np.float64)
    cnt = np.asarray(kc["nll_ev"], dtype=np.float64).sum(axis=1)
    joint_nll_row = cont + disc + cnt

    Xtr_d = np.asarray(pbar.densify(Xtr_sparse), dtype=np.float64)
    sampler = pgm.FittedTransitionSampler(
        tag, ct, k, kc, occ_models,
        train_min=Xtr_d.min(axis=0), train_max=Xtr_d.max(axis=0),
        train_absmax=np.abs(Xtr_d).max(axis=0),
        feature_names=list(ct.get_feature_names_out()),
        design_cols=num_cols,
    )
    return dict(sampler=sampler, joint_nll_row=np.asarray(joint_nll_row, dtype=np.float64),
                mean_joint_nll=float(np.mean(joint_nll_row)), num_cols=num_cols)


def evaluate_transition_variant(fit: Dict[str, Any], eval_trans: pd.DataFrame) -> Dict[str, float]:
    mu = np.asarray(
        fit["sampler"].analytic_conditional_support(eval_trans)["z_d_up_mu"], dtype=np.float64)
    real = eval_trans["z_d_up"].to_numpy(dtype=np.float64)
    rho = float(scipy.stats.spearmanr(mu, real).statistic) if len(mu) > 2 else 0.0
    return dict(mean_joint_nll=float(fit["mean_joint_nll"]), rho_zdup=rho)


# ===========================================================================
# Fast day-cluster paired bootstrap
# ===========================================================================
def fast_cluster_bootstrap_delta(
    day: np.ndarray, loss_base: np.ndarray, loss_aug: np.ndarray,
    n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED,
) -> Dict[str, float]:
    d = pd.DataFrame({"day": day, "base": loss_base, "aug": loss_aug})
    daily = (d.groupby("day", sort=False)
              .agg(base_sum=("base", "sum"), aug_sum=("aug", "sum"), n=("base", "size"))
              .reset_index(drop=True))
    B = daily["base_sum"].to_numpy(dtype=np.float64)
    A = daily["aug_sum"].to_numpy(dtype=np.float64)
    N = daily["n"].to_numpy(dtype=np.float64)
    D = len(daily)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(D, np.full(D, 1.0 / D), size=n_boot)
    denom = counts @ N
    base_loss = (counts @ B) / denom
    aug_loss = (counts @ A) / denom
    delta = base_loss - aug_loss  # positive = augmentation better
    return dict(
        point=float(np.mean(loss_base) - np.mean(loss_aug)),
        ci95_lower=float(np.percentile(delta, 2.5)),
        ci95_upper=float(np.percentile(delta, 97.5)),
        p_pos=float(np.mean(delta > 0)),
    )


# ===========================================================================
# Frozen 0B mechanism grid (baseline model only; EX-POST diagnostic)
# ===========================================================================
def build_baseline_mechanism_grid(
    df_tb2: pd.DataFrame, df_tb3: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """5x5 hazard x conviction grid. TB2 frozen edges applied to TB3.

    pi = sign(m_t) * r_trad_OC_ATR0 (frozen 0A continuation return).
    p_star is EX_POST diagnostic only and never enters any policy.
    """
    p2 = df_tb2["p_h"].to_numpy(dtype=np.float64)
    m2 = np.abs(df_tb2["score_mu"].to_numpy(dtype=np.float64))
    pe = np.quantile(p2, [0.2, 0.4, 0.6, 0.8])
    me = np.quantile(m2, [0.2, 0.4, 0.6, 0.8])
    p_edges = np.concatenate([[-np.inf], pe, [np.inf]])
    m_edges = np.concatenate([[-np.inf], me, [np.inf]])

    out = []
    for tag, df in [("TB2", df_tb2), ("TB3", df_tb3)]:
        p_b = np.digitize(df["p_h"].to_numpy(dtype=np.float64), p_edges[1:-1], right=False)
        m_b = np.digitize(np.abs(df["score_mu"].to_numpy(dtype=np.float64)), m_edges[1:-1], right=False)
        pi = df["pi"].to_numpy(dtype=np.float64)
        hz = df["hazard"].to_numpy(dtype=np.int64)
        for hb in range(5):
            for cb in range(5):
                mask = (p_b == hb) & (m_b == cb)
                n = int(mask.sum())
                if n == 0:
                    continue
                mu0 = float(pi[mask & (hz == 0)].mean()) if np.any(mask & (hz == 0)) else float("nan")
                mu1 = float(pi[mask & (hz == 1)].mean()) if np.any(mask & (hz == 1)) else float("nan")
                ev = float(pi[mask].mean())
                p_star = float("nan")
                if np.isfinite(mu0) and np.isfinite(mu1) and mu0 > 0 and mu1 < 0:
                    p_star = mu0 / (mu0 - mu1)
                out.append(dict(
                    block=tag, hazard_bin=hb, conviction_bin=cb, n=n,
                    mean_p_h=float(df["p_h"].to_numpy(dtype=np.float64)[mask].mean()),
                    mean_abs_m=float(np.abs(df["score_mu"].to_numpy(dtype=np.float64))[mask].mean()),
                    observed_H1_rate=float(hz[mask].mean()),
                    mu0=mu0, mu1=mu1, EV=ev,
                    H1_loss_rate=float((pi[mask & (hz == 1)] < 0).mean())
                    if np.any(mask & (hz == 1)) else float("nan"),
                    H1_mean_pi=mu1, p_star=p_star,
                ))
    meta = dict(p_edges=p_edges.tolist(), m_edges=m_edges.tolist(),
                note="EX_POST_MECHANISM_DIAGNOSTIC_ONLY; p_star never enters policy")
    return out, meta


def conditional_hazard_contrast(
    df_tb3: pd.DataFrame,
    p_edges: Sequence[float],
    m_edges: Sequence[float],
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Conditional hazard contrast: within each conviction quintile, top vs bottom hazard quintile.

    Every bootstrap replicate RE-COMPUTES the contrast from day-resampled data
    (no fixed point estimate is recycled). Cluster owner is `entry_day`, consistent
    with the 0A/0B day-cluster bootstraps. The five conviction quintiles are equally
    weighted -- never re-weighted by cell sample size.

    df_tb3 must contain: entry_day, p_h, score_mu, hazard, pi.
    Bins come from TB2-frozen edges (never TB3 quantiles).
    """
    p_h = df_tb3["p_h"].to_numpy(dtype=np.float64)
    abs_m = np.abs(df_tb3["score_mu"].to_numpy(dtype=np.float64))
    hz = df_tb3["hazard"].to_numpy(dtype=np.int64)
    pi = df_tb3["pi"].to_numpy(dtype=np.float64)
    day = df_tb3["entry_day"].to_numpy()

    p_b = np.digitize(p_h, np.asarray(p_edges, dtype=np.float64)[1:-1])
    m_b = np.digitize(abs_m, np.asarray(m_edges, dtype=np.float64)[1:-1])

    keep = (p_b == 0) | (p_b == 4)
    sub = pd.DataFrame(dict(day=day[keep], cb=m_b[keep], hb=p_b[keep],
                            h1=hz[keep].astype(np.float64), pi=pi[keep]))
    agg = (sub.groupby(["day", "cb", "hb"], sort=False)
              .agg(n=("h1", "size"), h1s=("h1", "sum"), pis=("pi", "sum"))
              .reset_index())

    days = np.unique(day)
    D = int(len(days))
    didx = {d: i for i, d in enumerate(days)}

    def mat(hb: int, col: str) -> np.ndarray:
        M = np.zeros((D, 5), dtype=np.float64)
        a = agg[agg["hb"] == hb]
        for d, cb, v in zip(a["day"].to_numpy(), a["cb"].to_numpy(), a[col].to_numpy()):
            ci = int(cb)
            if 0 <= ci < 5:
                M[didx[d], ci] = float(v)
        return M

    top_n, bot_n = mat(4, "n"), mat(0, "n")
    top_h1, bot_h1 = mat(4, "h1s"), mat(0, "h1s")
    top_pi, bot_pi = mat(4, "pis"), mat(0, "pis")

    # ---- point estimate: computed directly from the full TB3 sample ----
    Nt = top_n.sum(axis=0)
    Nb = bot_n.sum(axis=0)
    if np.any(Nt == 0) or np.any(Nb == 0):
        raise SystemExit("STOP_PGM_NATIVE0C_CONDITIONAL_CONTRAST_EMPTY_CELL")
    rate_t = top_h1.sum(axis=0) / Nt
    rate_b = bot_h1.sum(axis=0) / Nb
    ev_t = top_pi.sum(axis=0) / Nt
    ev_b = bot_pi.sum(axis=0) / Nb
    dH1_q = rate_t - rate_b
    dEV_q = ev_t - ev_b
    point_h1 = float(np.mean(dH1_q))
    point_ev = float(np.mean(dEV_q))

    # ---- bootstrap: day multiplicity, recomputed per replicate ----
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(D, np.full(D, 1.0 / D), size=n_boot)
    Nt_b = counts @ top_n
    Nb_b = counts @ bot_n
    Ht_b = counts @ top_h1
    Hb_b = counts @ bot_h1
    Pt_b = counts @ top_pi
    Pb_b = counts @ bot_pi

    valid = np.all(Nt_b > 0, axis=1) & np.all(Nb_b > 0, axis=1)
    n_invalid = int((~valid).sum())
    if n_invalid > 0:
        raise SystemExit(
            f"STOP_PGM_NATIVE0C_CONDITIONAL_BOOTSTRAP_EMPTY_CELL: {n_invalid}")

    boot_h1 = np.mean((Ht_b / Nt_b) - (Hb_b / Nb_b), axis=1)
    boot_ev = np.mean((Pt_b / Nt_b) - (Pb_b / Nb_b), axis=1)

    def _s(pt: float, boot: np.ndarray) -> Dict[str, float]:
        return dict(point=float(pt),
                    ci95_lower=float(np.percentile(boot, 2.5)),
                    ci95_upper=float(np.percentile(boot, 97.5)),
                    p_pos=float(np.mean(boot > 0)))

    out_ev = _s(point_ev, boot_ev)
    out_ev["p_neg"] = float(np.mean(boot_ev < 0))

    per_q = [dict(conviction_bin=q, n_top=float(Nt[q]), n_bottom=float(Nb[q]),
                  H1_top=float(rate_t[q]), H1_bottom=float(rate_b[q]),
                  Delta_H1=float(dH1_q[q]),
                  EV_top=float(ev_t[q]), EV_bottom=float(ev_b[q]),
                  Delta_EV=float(dEV_q[q])) for q in range(5)]

    return dict(Delta_H1_cond=_s(point_h1, boot_h1),
                Delta_EV_cond=out_ev,
                per_conviction=per_q,
                n_days=D, n_boot=int(n_boot),
                n_invalid_replicates=n_invalid,
                cluster_owner="entry_day")


def compute_age_hazard_curve(df: pd.DataFrame, block: str) -> List[Dict[str, Any]]:
    age = (df["bar_t"].to_numpy(np.int64) - df["start_bar"].to_numpy(np.int64))
    age = np.clip(age, 0, 21)
    hz = df["hazard"].to_numpy(dtype=np.int64)
    ph = df["p_h"].to_numpy(dtype=np.float64)
    rows = []
    for b in range(22):
        mask = age == b
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append(dict(block=block, age_bucket=("21+" if b == 21 else str(b)),
                         age_numeric=float(b), n=n,
                         H1_rate=float(hz[mask].mean()), mean_p_h=float(ph[mask].mean())))
    if rows:
        # Real numeric bucket values (0..20, 21) -- never a compressed rank index.
        xs = np.array([r["age_numeric"] for r in rows], dtype=np.float64)
        ys = np.array([r["H1_rate"] for r in rows], dtype=np.float64)
        sp = float(scipy.stats.spearmanr(xs, ys).statistic) if len(rows) > 2 else 0.0
    else:
        sp = 0.0
    for r in rows:
        r["spearman_age_vs_H1"] = sp
    return rows


def compute_h1_harm_diagnostics(df: pd.DataFrame, block: str,
                                p_edges: Sequence[float],
                                m_edges: Sequence[float]) -> Dict[str, Any]:
    """H1 rows only. harm_flag/harm_magnitude are diagnostic labels; no model is trained.

    p_edges / m_edges MUST be the TB2-frozen baseline quintile edges, so TB2 and TB3
    use the SAME risk ruler. No internal re-quantiling is permitted.
    """
    h1 = df[df["hazard"] == 1].copy()
    pi = h1["pi"].to_numpy(dtype=np.float64)
    harm = np.maximum(-pi, 0.0)

    def _group(bins: np.ndarray, name: str) -> List[Dict[str, Any]]:
        out = []
        for b in np.unique(bins):
            m = bins == b
            if int(m.sum()) == 0:
                continue
            out.append(dict(block=block, group=name, bin=int(b), n=int(m.sum()),
                            harm_rate=float((pi[m] < 0).mean()),
                            mean_harm=float(harm[m].mean()), mean_pi=float(pi[m].mean())))
        return out

    pb = np.digitize(h1["p_h"].to_numpy(dtype=np.float64),
                     np.asarray(p_edges, dtype=np.float64)[1:-1])
    mb = np.digitize(np.abs(h1["score_mu"].to_numpy(dtype=np.float64)),
                     np.asarray(m_edges, dtype=np.float64)[1:-1])
    age = np.clip(h1["bar_t"].to_numpy(np.int64) - h1["start_bar"].to_numpy(np.int64), 0, 21)
    return dict(by_hazard_quintile=_group(pb, "hazard_quintile"),
                by_conviction_quintile=_group(mb, "conviction_quintile"),
                by_age_bucket=_group(age, "age_bucket"))


# ===========================================================================
# Verdict
# ===========================================================================
def determine_state_augmentation_verdict(delta_ll: Dict[str, float],
                                         delta_jnll: Dict[str, float]) -> str:
    a_ok = delta_ll["ci95_lower"] > 0
    b_ok = delta_jnll["ci95_lower"] > 0
    if a_ok and b_ok:
        return VERDICT_STRINGS["JOINTLY"]
    if a_ok and not b_ok:
        return VERDICT_STRINGS["TERMINATION_ONLY"]
    if (not a_ok) and b_ok:
        return VERDICT_STRINGS["TRANSITION_ONLY"]
    return VERDICT_STRINGS["NOT_SUPPORTED"]


# ===========================================================================
# Shared loaders
# ===========================================================================
def compute_artifact_hashes() -> Dict[str, str]:
    if not pgm.SAMPLE_PATH.exists():
        raise SystemExit("STOP_PGM_NATIVE_SAMPLE_ARTIFACT_MISSING")
    if not pgm.TRANSITION_SAMPLE_PATH.exists():
        raise SystemExit("STOP_PGM_NATIVE_TRANSITION_ARTIFACT_MISSING")
    return dict(
        sample_artifact_sha256=hashlib.sha256(pgm.SAMPLE_PATH.read_bytes()).hexdigest(),
        transition_artifact_sha256=hashlib.sha256(
            pgm.TRANSITION_SAMPLE_PATH.read_bytes()).hexdigest(),
    )


def load_obs_with_features() -> pd.DataFrame:
    obs = n0a.load_observed_decision_universe()
    return add_incremental_state_features(obs)


def load_aligned_with_features() -> pd.DataFrame:
    """Observation rows + incremental features + frozen 0A returns alignment.

    n0a.align_raw_bars_and_returns preserves the incremental feature columns and adds
    r_trad_OC_ATR0 / is_entry_valid (needed for the frozen 0B mechanism grid).
    """
    feat = load_obs_with_features()
    _, _, bars_by_sym = ex0.load_env()
    cur_truth = n0a.load_transition_truth_audit()["cur"]
    aligned, _ = n0a.align_raw_bars_and_returns(feat, bars_by_sym, cur_truth=cur_truth)
    return aligned


def build_baseline_scored_frame(aligned: pd.DataFrame, fit_A: Dict[str, Any],
                                fit_B: Dict[str, Any]) -> pd.DataFrame:
    """Baseline (non-augmented) p_h / score_mu / pi for the frozen 0B mechanism grid."""
    aligned = aligned[aligned["is_entry_valid"]].copy()
    frames = []
    for block, fit in [(TB2_BLOCK, fit_A), (TB3_BLOCK, fit_B)]:
        sub = aligned[aligned["block"] == block].copy()
        mc = fit["trans_samplers"][PRIMARY_TRANSITION_HEAD]
        term = fit["term_samplers"][PRIMARY_TERMINAL_HEAD]
        mom = mc.analytic_conditional_support(sub)
        sub["score_mu"] = -np.asarray(mom["z_d_up_mu"], dtype=np.float64)
        sub["p_h"] = n0b.predict_hazard_probability(term, sub)
        frames.append(sub)
    out = pd.concat(frames, ignore_index=True)
    out["pi"] = np.sign(out["score_mu"].to_numpy(dtype=np.float64)) * \
        out["r_trad_OC_ATR0"].to_numpy(dtype=np.float64)
    return out


# ===========================================================================
# Modes
# ===========================================================================
def run_audit_only() -> None:
    print("=" * 50, flush=True)
    print("PGM-NATIVE-0C: AUDIT-ONLY", flush=True)
    print("=" * 50, flush=True)

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT),
                                   text=True).strip()
    print(f"[AUDIT] current HEAD: {head}")
    print(f"[AUDIT] BASE_SHA: {BASE_SHA}")
    res = subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"],
                         cwd=str(_REPO_ROOT), capture_output=True)
    if res.returncode != 0:
        raise SystemExit("STOP_PGM_NATIVE0C_WRONG_BASE_HEAD")
    print("[AUDIT] BASE SHA ancestry: PASS")

    hashes = compute_artifact_hashes()
    print(f"[AUDIT] sample_artifact_sha256: {hashes['sample_artifact_sha256']}")
    print(f"[AUDIT] transition_artifact_sha256: {hashes['transition_artifact_sha256']}")

    obs = n0a.load_observed_decision_universe()
    aud = n0a.audit_decision_universe(obs)
    print(f"[AUDIT] n_all_obs={aud['n_all_obs']} H0={aud['n_hazard0']} H1={aud['n_hazard1']}")
    print(f"[AUDIT] 15 symbols complete: {sorted(aud['symbols']) == EXPECTED_SYMBOLS}")

    trans_aud = n0a.load_transition_truth_audit()
    err = n0a.audit_atr0_owner_parity(obs, trans_aud["cur"])
    print(f"[AUDIT] max_abs_atr0_owner_error: {err:.2e}")

    # U / E primitive audit
    for c in U_PRIMITIVES:
        if c not in obs.columns:
            raise SystemExit(f"STOP_PGM_NATIVE0C_U_PRIMITIVE_MISSING: {c}")
    for c in E_PRIMITIVES:
        if c not in obs.columns:
            raise SystemExit(f"STOP_PGM_NATIVE0C_E_PRIMITIVE_MISSING: {c}")
    print("[AUDIT] U primitives: OK | E primitives: OK")

    feat = add_incremental_state_features(obs)
    fin = audit_incremental_features_finite(feat)
    print(f"[AUDIT] U_COLS={U_COLS}")
    print(f"[AUDIT] E_COLS={E_COLS}")
    print(f"[AUDIT] all incremental features finite: {fin['all_finite']} {fin['non_finite_counts']}")
    if not fin["all_finite"]:
        raise SystemExit("STOP_PGM_NATIVE0C_INCREMENTAL_NON_FINITE")

    age_first = (feat.groupby(["symbol", "episode_id"], sort=False)["bar_t"].first()
                 - feat.groupby(["symbol", "episode_id"], sort=False)["start_bar"].first())
    if not bool((age_first == 0).all()):
        raise SystemExit("STOP_PGM_NATIVE0C_EPISODE_FIRST_AGE_NOT_ZERO")
    print("[AUDIT] episode first-row age == 0: PASS")

    # prefix invariance
    pi_ok = _prefix_invariance_check()
    print(f"[AUDIT] prefix invariance: {'PASS' if pi_ok else 'FAIL'}")
    if not pi_ok:
        raise SystemExit("STOP_PGM_NATIVE0C_FEATURE_FUTURE_DEPENDENCE")

    fit_A = pgm.fit_samplers_for_window(pgm.WINDOWS[0], pgm.SAMPLE_PATH,
                                        pgm.TRANSITION_SAMPLE_PATH)
    t2_cols = fit_A["term_samplers"][PRIMARY_TERMINAL_HEAD].design_cols
    mc_cols = fit_A["trans_samplers"][PRIMARY_TRANSITION_HEAD].design_cols
    print(f"[AUDIT] actual T2 terminal design cols: {len(t2_cols)}")
    print(f"[AUDIT] actual MC transition design cols: {len(mc_cols)}")

    collision = set(INCREMENTAL_COLS) & (set(t2_cols) | set(mc_cols))
    print(f"[AUDIT] incremental-column collision count: {len(collision)}")
    if collision:
        raise SystemExit(f"STOP_PGM_NATIVE0C_INCREMENTAL_COLUMN_COLLISION: {collision}")

    # transition join audit
    trans = pd.read_parquet(pgm.TRANSITION_SAMPLE_PATH)
    merged = merge_incremental_features_into_transition(trans, feat)
    print(f"[AUDIT] transition merged rows: {len(merged)} (expected {EXPECTED_TRANSITION_ROWS})")
    print("[AUDIT] transition unmatched: 0")
    jp = merged.attrs.get("join_parity", {})
    print(f"[AUDIT] join parity columns ({len(jp.get('parity_columns', []))}): "
          f"{jp.get('parity_columns', [])}")
    for c, v in (jp.get("per_column_max_abs_diff") or {}).items():
        print(f"[AUDIT]   join parity max_abs_diff {c}: {v:.3e}")
    print(f"[AUDIT] join parity OVERALL max_abs_diff: {jp.get('overall_max_abs_diff')}")
    print("[AUDIT] join identity parity: PASS")

    # baseline parity: BOTH windows must close before full is authorized
    par_A = verify_window_baseline_parity(pgm.WINDOWS[0], fit_A, feat, merged, "WindowA")
    _print_window_parity(par_A)

    print("[AUDIT] Fitting Window B samplers for baseline parity...", flush=True)
    fit_B = pgm.fit_samplers_for_window(pgm.WINDOWS[1], pgm.SAMPLE_PATH,
                                        pgm.TRANSITION_SAMPLE_PATH)
    par_B = verify_window_baseline_parity(pgm.WINDOWS[1], fit_B, feat, merged, "WindowB")
    _print_window_parity(par_B)

    print("[AUDIT] NO ECONOMIC / SCIENTIFIC VERDICT EMITTED", flush=True)


def _print_window_parity(p: Dict[str, Any]) -> None:
    lab = p["label"]
    print(f"[AUDIT] {lab} terminal baseline max_abs(p_wrapper - p_production): "
          f"{p['terminal_p_max_abs_diff']:.3e}")
    print(f"[AUDIT] {lab} transition max_abs(z_d_up_mu wrapper - production): "
          f"{p['transition_mu_max_abs_diff']:.3e}")
    print(f"[AUDIT] {lab} mean_joint_nll production={p['production_mean_joint_nll']:.16f}")
    print(f"[AUDIT] {lab} mean_joint_nll wrapper ={p['wrapper_mean_joint_nll']:.16f}")
    print(f"[AUDIT] {lab} mean_joint_nll abs diff ={p['transition_mean_jnll_abs_diff']:.3e}")


def verify_window_baseline_parity(w: Dict[str, Any], fit: Dict[str, Any],
                                  feat: pd.DataFrame, merged: pd.DataFrame,
                                  label: str) -> Dict[str, Any]:
    """Research wrapper (extra_cols=[]) must reproduce production exactly for one window."""
    ev_obs = feat[feat["block"] == w["eval"]].reset_index(drop=True)
    tr_obs = feat[feat["block"].isin(w["train"])].reset_index(drop=True)
    wrap = fit_terminal_hazard_variant(tr_obs, ev_obs, [])
    prod_p = n0b.predict_hazard_probability(
        fit["term_samplers"][PRIMARY_TERMINAL_HEAD], ev_obs)
    d_p = float(np.max(np.abs(wrap["p_eval"] - prod_p)))
    if d_p > 1e-12:
        raise SystemExit(f"STOP_PGM_NATIVE0C_TERMINAL_BASELINE_PARITY_FAIL: {label} {d_p}")

    tr_t = merged[merged["block"].isin(w["train"])].reset_index(drop=True)
    ev_t = merged[merged["block"] == w["eval"]].reset_index(drop=True)
    mc_cols = fit["trans_samplers"][PRIMARY_TRANSITION_HEAD].design_cols
    var = fit_transition_variant(tr_t, ev_t, [], mc_cols, tag=f"{label}_PARITY")
    mu_prod = np.asarray(fit["trans_samplers"][PRIMARY_TRANSITION_HEAD]
                         .analytic_conditional_support(ev_t)["z_d_up_mu"], dtype=np.float64)
    mu_wrap = np.asarray(var["sampler"].analytic_conditional_support(ev_t)["z_d_up_mu"],
                         dtype=np.float64)
    d_mu = float(np.max(np.abs(mu_wrap - mu_prod)))
    if d_mu > 1e-12:
        raise SystemExit(f"STOP_PGM_NATIVE0C_TRANSITION_BASELINE_PARITY_FAIL: {label} {d_mu}")

    prod_jnll = float(fit["trans_parity"][PRIMARY_TRANSITION_HEAD]["mean_joint_nll"])
    d_jnll = abs(float(var["mean_joint_nll"]) - prod_jnll)
    if d_jnll > 1e-12:
        raise SystemExit(f"STOP_PGM_NATIVE0C_TRANSITION_BASELINE_PARITY_FAIL: {label} {d_jnll}")

    return dict(label=label,
                terminal_p_max_abs_diff=d_p,
                transition_mu_max_abs_diff=d_mu,
                transition_mean_jnll_abs_diff=d_jnll,
                production_mean_joint_nll=prod_jnll,
                wrapper_mean_joint_nll=float(var["mean_joint_nll"]))


def _prefix_invariance_check() -> bool:
    n = 12
    def mk(seed_tail: int) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        base_rows = dict(
            symbol=["X"] * n, episode_id=["E1"] * n,
            block=["TB2"] * n, bar_t=np.arange(100, 100 + n),
            start_bar=[100] * n,
            cur_up_distance_R=rng.normal(0, 1, n),
            path_total_variation_R=np.abs(rng.normal(0.5, 0.2, n)),
            path_max_up_excursion_R=np.abs(rng.normal(0.3, 0.1, n)),
            path_max_down_excursion_R=np.abs(rng.normal(0.3, 0.1, n)),
            path_last_return_R=rng.normal(0, 0.5, n),
            path_current_bar_range_R=np.abs(rng.normal(0.4, 0.1, n)),
            path_direction_change_rate=rng.uniform(0, 1, n),
            hazard=[0] * n,
        )
        df = pd.DataFrame(base_rows)
        if seed_tail == 1:
            for c in ["path_last_return_R", "path_current_bar_range_R",
                      "path_direction_change_rate", "cur_up_distance_R",
                      "path_total_variation_R", "path_max_up_excursion_R",
                      "path_max_down_excursion_R"]:
                df.loc[6:, c] = df.loc[6:, c] * 3.0 + 5.0
            df.loc[6:, "bar_t"] = df.loc[6:, "bar_t"] + 50
        return df

    a = add_incremental_state_features(mk(0))
    b = add_incremental_state_features(mk(1))
    k = 6
    for c in INCREMENTAL_COLS:
        da = a[c].to_numpy(dtype=np.float64)[:k]
        db = b[c].to_numpy(dtype=np.float64)[:k]
        if not np.array_equal(da, db):
            return False
    return True


def run_smoke_test() -> None:
    print("=" * 50, flush=True)
    print("PGM-NATIVE-0C: SMOKE (wiring only)", flush=True)
    print("=" * 50, flush=True)
    t0 = time.perf_counter()

    obs_aug = load_obs_with_features()
    trans = pd.read_parquet(pgm.TRANSITION_SAMPLE_PATH)
    merged = merge_incremental_features_into_transition(trans, obs_aug)

    wA, wB = pgm.WINDOWS[0], pgm.WINDOWS[1]
    fit_A = pgm.fit_samplers_for_window(wA, pgm.SAMPLE_PATH, pgm.TRANSITION_SAMPLE_PATH)
    mc_cols = fit_A["trans_samplers"][PRIMARY_TRANSITION_HEAD].design_cols

    def sub(df, w, train: bool, cap: int):
        """Train side uses the FULL block (head-slicing can degenerate a head to a
        single class and is not the production contract). Eval side is capped."""
        if train:
            s = df[df["block"].isin(w["train"])]
        else:
            s = df[df["block"] == w["eval"]].head(cap)
        return s.reset_index(drop=True)

    print("[SMOKE] --- terminal variants ---")
    for vname, extra in VARIANTS.items():
        tr_o = sub(obs_aug, wA, True, 4096)
        ev_o = sub(obs_aug, wA, False, 512)
        f = fit_terminal_hazard_variant(tr_o, ev_o, extra)
        m = evaluate_terminal_variant(f, ev_o["hazard"].to_numpy(np.int64))
        print(f"  {vname}: log_loss={m['log_loss']:.6f} brier={m['brier']:.6f} "
              f"auc={m['roc_auc']:.4f}")

    print("[SMOKE] --- transition variants ---")
    for vname, extra in VARIANTS.items():
        tr_t = sub(merged, wA, True, 4096)
        ev_t = sub(merged, wA, False, 512)
        f = fit_transition_variant(tr_t, ev_t, extra, mc_cols, tag=f"A_{vname}")
        m = evaluate_transition_variant(f, ev_t)
        print(f"  {vname}: mean_joint_nll={m['mean_joint_nll']:.6f} rho_zdup={m['rho_zdup']:.4f}")

    print("[SMOKE] --- 0B mechanism grid wiring ---")
    fit_B = pgm.fit_samplers_for_window(wB, pgm.SAMPLE_PATH, pgm.TRANSITION_SAMPLE_PATH)
    aligned = load_aligned_with_features()
    scored = build_baseline_scored_frame(aligned, fit_A, fit_B)
    # Mechanism/contrast diagnostics use a wider subsample than the 512-row model eval
    # so that every 5x5 cell (and thus every top/bottom contrast denominator) is populated.
    g2 = scored[scored["block"] == TB2_BLOCK].head(8192).reset_index(drop=True)
    g3 = scored[scored["block"] == TB3_BLOCK].head(4096).reset_index(drop=True)
    grid, meta = build_baseline_mechanism_grid(g2, g3)
    print(f"  grid cells={len(grid)} (TB2 frozen edges -> TB3) p_star diagnostic only")

    p_edges = np.asarray(meta["p_edges"], dtype=np.float64)
    m_edges = np.asarray(meta["m_edges"], dtype=np.float64)

    print("[SMOKE] --- conditional hazard contrast wiring ---")
    cc = conditional_hazard_contrast(g3, p_edges, m_edges, n_boot=200)
    for k in ["Delta_H1_cond", "Delta_EV_cond"]:
        s = cc[k]
        extra = f" p_neg={s['p_neg']:.4f}" if "p_neg" in s else ""
        print(f"  {k}: point={s['point']:.6f} "
              f"CI95=[{s['ci95_lower']:.6f}, {s['ci95_upper']:.6f}] "
              f"p_pos={s['p_pos']:.4f}{extra}")
    print(f"  n_days={cc['n_days']} n_boot={cc['n_boot']} "
          f"n_invalid_replicates={cc['n_invalid_replicates']} "
          f"cluster_owner={cc['cluster_owner']}")

    print("[SMOKE] --- age hazard wiring ---")
    age_rows = compute_age_hazard_curve(g3, TB3_BLOCK)
    print(f"  age buckets={len(age_rows)}")

    print("[SMOKE] --- H1 harm wiring (frozen TB2 edges) ---")
    harm = compute_h1_harm_diagnostics(g3, TB3_BLOCK, p_edges, m_edges)
    print(f"  by_hazard={len(harm['by_hazard_quintile'])} "
          f"by_conviction={len(harm['by_conviction_quintile'])} "
          f"by_age={len(harm['by_age_bucket'])}")

    print("[SMOKE] SMOKE ONLY / NO SCIENTIFIC VERDICT", flush=True)
    print(f"[SMOKE COMPLETE] {time.perf_counter() - t0:.2f}s -- NO SCIENTIFIC VERDICT", flush=True)


# ===========================================================================
# Full exploratory runner (0C.2)
# ===========================================================================
def require_full_authorization() -> None:
    token = os.environ.get("AUTHORIZE_PGM_NATIVE0C_FULL_EXPLORATORY", "").strip()
    if token != "1":
        raise SystemExit("STOP_PGM_NATIVE0C_FULL_EXPLORATORY_NOT_AUTHORIZED")


def _terminal_metrics_row(block: str, variant: str, fit: Dict[str, Any],
                          hazard: np.ndarray) -> Dict[str, Any]:
    m = evaluate_terminal_variant(fit, hazard)
    return dict(block=block, variant=variant, n=int(len(hazard)),
                log_loss=m["log_loss"], brier=m["brier"],
                roc_auc=m["roc_auc"], pr_auc=m["pr_auc"])


def execute_full_pipeline(
    obs_aug: pd.DataFrame,
    merged_trans: pd.DataFrame,
    aligned_econ: pd.DataFrame,
    fit_A: Dict[str, Any],
    fit_B: Dict[str, Any],
    bars_by_sym: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Full 0C evaluation. Only ADDS features / calls production owners; no strategy logic.

    Predictive-loss bootstraps cluster on `decision_day` (state bar t).
    Economic mechanism contrast clusters on `entry_day` (next-open entry).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = [(TB2_BLOCK, pgm.WINDOWS[0], fit_A), (TB3_BLOCK, pgm.WINDOWS[1], fit_B)]

    terminal_rows: List[Dict[str, Any]] = []
    terminal_boot: List[Dict[str, Any]] = []
    transition_rows: List[Dict[str, Any]] = []
    transition_boot: List[Dict[str, Any]] = []
    term_fits: Dict[str, Dict[str, Any]] = {}
    trans_fits: Dict[str, Dict[str, Any]] = {}
    eval_obs_by_block: Dict[str, pd.DataFrame] = {}

    for block, w, fit in windows:
        tr_o = obs_aug[obs_aug["block"].isin(w["train"])].reset_index(drop=True)
        ev_o = obs_aug[obs_aug["block"] == w["eval"]].reset_index(drop=True)
        eval_obs_by_block[block] = ev_o
        haz = ev_o["hazard"].to_numpy(np.int64)
        day_o = ev_o["decision_day"].to_numpy()

        term_fits[block] = {}
        for v, extra in VARIANTS.items():
            term_fits[block][v] = fit_terminal_hazard_variant(tr_o, ev_o, extra)
        for v in VARIANTS:
            terminal_rows.append(_terminal_metrics_row(block, v, term_fits[block][v], haz))
        b_ll = next(r["log_loss"] for r in terminal_rows
                    if r["block"] == block and r["variant"] == "PGM0")
        b_br = next(r["brier"] for r in terminal_rows
                    if r["block"] == block and r["variant"] == "PGM0")
        for r in terminal_rows:
            if r["block"] == block:
                r["Delta_LogLoss_vs_PGM0"] = b_ll - r["log_loss"]
                r["Delta_Brier_vs_PGM0"] = b_br - r["brier"]
        for v in ["PGM_U", "PGM_E", "PGM_UE"]:
            for metric, key in [("Delta_LogLoss", "row_logloss"), ("Delta_Brier", "row_brier")]:
                s = fast_cluster_bootstrap_delta(
                    day_o, term_fits[block]["PGM0"][key], term_fits[block][v][key])
                terminal_boot.append(dict(block=block, variant=v, metric=metric, **s))

        tr_t = merged_trans[merged_trans["block"].isin(w["train"])].reset_index(drop=True)
        ev_t = merged_trans[merged_trans["block"] == w["eval"]].reset_index(drop=True)
        mc_cols = fit["trans_samplers"][PRIMARY_TRANSITION_HEAD].design_cols
        day_t = ev_t["decision_day"].to_numpy()
        trans_fits[block] = {}
        for v, extra in VARIANTS.items():
            trans_fits[block][v] = fit_transition_variant(
                tr_t, ev_t, extra, mc_cols, tag=f"{block}_{v}")
            m = evaluate_transition_variant(trans_fits[block][v], ev_t)
            transition_rows.append(dict(block=block, variant=v, n=int(len(ev_t)),
                                        mean_joint_nll=m["mean_joint_nll"],
                                        rho_zdup=m["rho_zdup"]))
        b_j = next(r["mean_joint_nll"] for r in transition_rows
                   if r["block"] == block and r["variant"] == "PGM0")
        for r in transition_rows:
            if r["block"] == block:
                r["Delta_JNLL_vs_PGM0"] = b_j - r["mean_joint_nll"]
        for v in ["PGM_U", "PGM_E", "PGM_UE"]:
            s = fast_cluster_bootstrap_delta(
                day_t, trans_fits[block]["PGM0"]["joint_nll_row"],
                trans_fits[block][v]["joint_nll_row"])
            transition_boot.append(dict(block=block, variant=v, metric="Delta_JNLL", **s))

    # ---- baseline economic mechanism frame (frozen baseline models only) ----
    scored = build_baseline_scored_frame(aligned_econ, fit_A, fit_B)
    s2 = scored[scored["block"] == TB2_BLOCK].reset_index(drop=True)
    s3 = scored[scored["block"] == TB3_BLOCK].reset_index(drop=True)
    grid, meta = build_baseline_mechanism_grid(s2, s3)
    p_edges = np.asarray(meta["p_edges"], dtype=np.float64)
    m_edges = np.asarray(meta["m_edges"], dtype=np.float64)
    contrast = conditional_hazard_contrast(s3, p_edges, m_edges,
                                           n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)

    # ---- age hazard on FULL observation eval rows (not is_entry_valid filtered) ----
    age_rows: List[Dict[str, Any]] = []
    for block, w, fit in windows:
        tmp = eval_obs_by_block[block][["block", "bar_t", "start_bar", "hazard"]].copy()
        tmp["p_h"] = term_fits[block]["PGM0"]["p_eval"]
        age_rows += compute_age_hazard_curve(tmp, block)

    # ---- H1 economic harm (frozen TB2 edges) ----
    harm2 = compute_h1_harm_diagnostics(s2, TB2_BLOCK, p_edges, m_edges)
    harm3 = compute_h1_harm_diagnostics(s3, TB3_BLOCK, p_edges, m_edges)
    harm_rows = (harm2["by_hazard_quintile"] + harm2["by_conviction_quintile"]
                 + harm2["by_age_bucket"] + harm3["by_hazard_quintile"]
                 + harm3["by_conviction_quintile"] + harm3["by_age_bucket"])

    # ---- PRIMARY verdict (frozen) ----
    ll_boot = next(r for r in terminal_boot
                   if r["block"] == TB3_BLOCK and r["variant"] == "PGM_UE"
                   and r["metric"] == "Delta_LogLoss")
    jnll_boot = next(r for r in transition_boot
                     if r["block"] == TB3_BLOCK and r["variant"] == "PGM_UE")
    verdict = determine_state_augmentation_verdict(ll_boot, jnll_boot)

    summary = {
        "EXPERIMENT_NAME": EXPERIMENT_NAME,
        "EXPERIMENT_SCOPE": EXPERIMENT_SCOPE,
        "base_sha": BASE_SHA,
        "run_head": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                            cwd=str(_REPO_ROOT), text=True).strip(),
        **compute_artifact_hashes(),
        "U_COLS": U_COLS, "E_COLS": E_COLS,
        "VARIANTS": {k: list(v) for k, v in VARIANTS.items()},
        "PRIMARY_VARIANT": PRIMARY_VARIANT,
        "join_parity": merged_trans.attrs.get("join_parity", {}),
        "TB2": dict(terminal_metrics=[r for r in terminal_rows if r["block"] == TB2_BLOCK],
                    terminal_bootstrap=[r for r in terminal_boot if r["block"] == TB2_BLOCK],
                    transition_metrics=[r for r in transition_rows if r["block"] == TB2_BLOCK],
                    transition_bootstrap=[r for r in transition_boot if r["block"] == TB2_BLOCK]),
        "TB3": dict(terminal_metrics=[r for r in terminal_rows if r["block"] == TB3_BLOCK],
                    terminal_bootstrap=[r for r in terminal_boot if r["block"] == TB3_BLOCK],
                    transition_metrics=[r for r in transition_rows if r["block"] == TB3_BLOCK],
                    transition_bootstrap=[r for r in transition_boot if r["block"] == TB3_BLOCK]),
        "primary": dict(TB3_PGM_UE_Delta_LogLoss_bootstrap=ll_boot,
                        TB3_PGM_UE_Delta_JNLL_bootstrap=jnll_boot),
        "mechanism": dict(p_edges=p_edges.tolist(), m_edges=m_edges.tolist(),
                          conditional_contrast=contrast),
        "age": dict(TB2_spearman=(age_rows[0]["spearman_age_vs_H1"] if age_rows else 0.0),
                    TB3_spearman=(next((r["spearman_age_vs_H1"] for r in age_rows
                                        if r["block"] == TB3_BLOCK), 0.0))),
        "formal_verdict": verdict,
        "known_scope_limitations": [
            "previously inspected TB3 (exploratory mechanism validation, not pristine holdout)",
            "cross-block episodes excluded from frozen PGM sample",
            "event_mask==0 censored episodes excluded from frozen PGM sample",
            "frozen PGM-bar sample",
            "no trading-policy optimization",
            "economic mechanism diagnostics are ex-post",
            "augmentation uses transforms of existing state primitives, not new external data",
        ],
    }

    _write_artifacts(summary, terminal_rows, terminal_boot, transition_rows,
                     transition_boot, grid, contrast, age_rows, harm_rows, out_dir)
    validate_output_artifacts(summary, out_dir)
    print("[FULL] verdict:", verdict, flush=True)
    return summary


def _write_artifacts(summary, terminal_rows, terminal_boot, transition_rows,
                     transition_boot, grid, contrast, age_rows, harm_rows, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    terminal_cols = ["block", "variant", "n", "log_loss", "brier", "roc_auc", "pr_auc",
                     "Delta_LogLoss_vs_PGM0", "Delta_Brier_vs_PGM0"]
    pd.DataFrame(terminal_rows)[terminal_cols].to_csv(
        out_dir / f"{PREFIX}_terminal_metrics.csv", index=False)
    pd.DataFrame(terminal_boot)[["block", "variant", "metric", "point",
                                 "ci95_lower", "ci95_upper", "p_pos"]].to_csv(
        out_dir / f"{PREFIX}_terminal_bootstrap.csv", index=False)
    pd.DataFrame(transition_rows)[["block", "variant", "n", "mean_joint_nll",
                                   "rho_zdup", "Delta_JNLL_vs_PGM0"]].to_csv(
        out_dir / f"{PREFIX}_transition_metrics.csv", index=False)
    pd.DataFrame(transition_boot)[["block", "variant", "metric", "point",
                                   "ci95_lower", "ci95_upper", "p_pos"]].to_csv(
        out_dir / f"{PREFIX}_transition_bootstrap.csv", index=False)

    grid_cols = ["block", "hazard_bin", "conviction_bin", "n", "mean_p_h", "mean_abs_m",
                 "observed_H1_rate", "mu0", "mu1", "EV", "H1_loss_rate", "H1_mean_pi", "p_star"]
    pd.DataFrame(grid)[grid_cols].to_csv(out_dir / f"{PREFIX}_mechanism_grid.csv", index=False)

    # conditional contrast: fixed schema, row_type in {summary, conviction_bin}
    cc = contrast
    cc_rows = [
        dict(row_type="summary", metric="Delta_H1_cond", conviction_bin=-1,
             point=cc["Delta_H1_cond"]["point"],
             ci95_lower=cc["Delta_H1_cond"]["ci95_lower"],
             ci95_upper=cc["Delta_H1_cond"]["ci95_upper"],
             p_pos=cc["Delta_H1_cond"]["p_pos"], p_neg=float("nan")),
        dict(row_type="summary", metric="Delta_EV_cond", conviction_bin=-1,
             point=cc["Delta_EV_cond"]["point"],
             ci95_lower=cc["Delta_EV_cond"]["ci95_lower"],
             ci95_upper=cc["Delta_EV_cond"]["ci95_upper"],
             p_pos=cc["Delta_EV_cond"]["p_pos"],
             p_neg=cc["Delta_EV_cond"]["p_neg"]),
    ]
    for q in cc["per_conviction"]:
        cc_rows.append(dict(row_type="conviction_bin", metric="", conviction_bin=q["conviction_bin"],
                            point=float("nan"), ci95_lower=float("nan"),
                            ci95_upper=float("nan"), p_pos=float("nan"), p_neg=float("nan"),
                            n_top=q["n_top"], n_bottom=q["n_bottom"],
                            H1_top=q["H1_top"], H1_bottom=q["H1_bottom"], Delta_H1=q["Delta_H1"],
                            EV_top=q["EV_top"], EV_bottom=q["EV_bottom"], Delta_EV=q["Delta_EV"]))
    cc_cols = ["row_type", "metric", "conviction_bin", "point", "ci95_lower", "ci95_upper",
               "p_pos", "p_neg", "n_top", "n_bottom", "H1_top", "H1_bottom", "Delta_H1",
               "EV_top", "EV_bottom", "Delta_EV"]
    pd.DataFrame(cc_rows).reindex(columns=cc_cols).to_csv(
        out_dir / f"{PREFIX}_conditional_contrast.csv", index=False)

    age_cols = ["block", "age_bucket", "age_numeric", "n", "H1_rate", "mean_p_h",
                "spearman_age_vs_H1"]
    pd.DataFrame(age_rows)[age_cols].to_csv(out_dir / f"{PREFIX}_age_hazard.csv", index=False)
    pd.DataFrame(harm_rows)[["block", "group", "bin", "n", "harm_rate",
                             "mean_harm", "mean_pi"]].to_csv(
        out_dir / f"{PREFIX}_h1_harm.csv", index=False)
    (out_dir / f"{PREFIX}_formal_summary.json").write_text(json.dumps(summary, indent=2, default=str))


def validate_output_artifacts(summary: Dict[str, Any], out_dir: Path) -> bool:
    """Re-read artifacts and cross-check against the in-memory summary."""
    out = Path(out_dir)

    def _fail(msg: str):
        raise SystemExit(f"STOP_PGM_NATIVE0C_OUTPUT_PARITY_FAIL: {msg}")

    for fn in ARTIFACT_FILES:
        p = out / fn
        if (not p.exists()) or p.stat().st_size == 0:
            _fail(f"missing/empty {fn}")

    tm = pd.read_csv(out / f"{PREFIX}_terminal_metrics.csv")
    for blk in ["TB2", "TB3"]:
        for r in summary[blk]["terminal_metrics"]:
            row = tm[(tm["block"] == blk) & (tm["variant"] == r["variant"])]
            if len(row) != 1:
                _fail(f"terminal_metrics row {blk}/{r['variant']}")
            if abs(float(row.iloc[0]["log_loss"]) - r["log_loss"]) > 1e-12:
                _fail(f"terminal log_loss mismatch {blk}/{r['variant']}")

    trm = pd.read_csv(out / f"{PREFIX}_transition_metrics.csv")
    for blk in ["TB2", "TB3"]:
        for r in summary[blk]["transition_metrics"]:
            row = trm[(trm["block"] == blk) & (trm["variant"] == r["variant"])]
            if len(row) != 1:
                _fail(f"transition_metrics row {blk}/{r['variant']}")
            if abs(float(row.iloc[0]["mean_joint_nll"]) - r["mean_joint_nll"]) > 1e-12:
                _fail(f"transition mean_joint_nll mismatch {blk}/{r['variant']}")

    tb = pd.read_csv(out / f"{PREFIX}_terminal_bootstrap.csv")
    p = summary["primary"]["TB3_PGM_UE_Delta_LogLoss_bootstrap"]
    row = tb[(tb["block"] == "TB3") & (tb["variant"] == "PGM_UE")
             & (tb["metric"] == "Delta_LogLoss")]
    if len(row) != 1 or abs(float(row.iloc[0]["ci95_lower"]) - p["ci95_lower"]) > 1e-12:
        _fail("primary Delta_LogLoss bootstrap mismatch")

    jb = pd.read_csv(out / f"{PREFIX}_transition_bootstrap.csv")
    pj = summary["primary"]["TB3_PGM_UE_Delta_JNLL_bootstrap"]
    row = jb[(jb["block"] == "TB3") & (jb["variant"] == "PGM_UE")]
    if len(row) != 1 or abs(float(row.iloc[0]["ci95_lower"]) - pj["ci95_lower"]) > 1e-12:
        _fail("primary Delta_JNLL bootstrap mismatch")

    grid = pd.read_csv(out / f"{PREFIX}_mechanism_grid.csv")
    for blk in ["TB2", "TB3"]:
        if len(grid[grid["block"] == blk]) > 25:
            _fail(f"mechanism_grid {blk} has >25 cells")
        if len(grid[grid["block"] == blk]) == 0:
            _fail(f"mechanism_grid {blk} empty")

    age = pd.read_csv(out / f"{PREFIX}_age_hazard.csv")
    for blk in ["TB2", "TB3"]:
        if len(age[age["block"] == blk]) == 0:
            _fail(f"age_hazard {blk} empty")

    harm = pd.read_csv(out / f"{PREFIX}_h1_harm.csv")
    if set(harm["group"].unique()) - {"hazard_quintile", "conviction_quintile", "age_bucket"}:
        _fail("h1_harm has unexpected group values")

    return True


def run_full_exploratory(output_dir: Optional[Path] = None) -> Dict[str, Any]:
    require_full_authorization()

    # ---- governance + causal gates BEFORE any model fit ----
    head = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   cwd=str(_REPO_ROOT), text=True).strip()
    res = subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"],
                         cwd=str(_REPO_ROOT), capture_output=True)
    if res.returncode != 0:
        raise SystemExit("STOP_PGM_NATIVE0C_WRONG_BASE_HEAD")
    hashes = compute_artifact_hashes()

    obs = n0a.load_observed_decision_universe()
    aud = n0a.audit_decision_universe(obs)
    trans_aud = n0a.load_transition_truth_audit()
    max_atr0_err = n0a.audit_atr0_owner_parity(obs, trans_aud["cur"])

    feat = add_incremental_state_features(obs)
    fin = audit_incremental_features_finite(feat)
    if not fin["all_finite"]:
        raise SystemExit("STOP_PGM_NATIVE0C_INCREMENTAL_NON_FINITE")
    if not _prefix_invariance_check():
        raise SystemExit("STOP_PGM_NATIVE0C_FEATURE_FUTURE_DEPENDENCE")

    _, _, bars_by_sym = ex0.load_env()
    obs_day = attach_decision_day(feat, bars_by_sym)
    trans = pd.read_parquet(pgm.TRANSITION_SAMPLE_PATH)
    merged = merge_incremental_features_into_transition(trans, obs_day)

    fit_A = pgm.fit_samplers_for_window(pgm.WINDOWS[0], pgm.SAMPLE_PATH,
                                        pgm.TRANSITION_SAMPLE_PATH)
    fit_B = pgm.fit_samplers_for_window(pgm.WINDOWS[1], pgm.SAMPLE_PATH,
                                        pgm.TRANSITION_SAMPLE_PATH)
    par_A = verify_window_baseline_parity(pgm.WINDOWS[0], fit_A, feat, merged, "WindowA")
    par_B = verify_window_baseline_parity(pgm.WINDOWS[1], fit_B, feat, merged, "WindowB")

    aligned = n0a.align_raw_bars_and_returns(obs_day, bars_by_sym, cur_truth=trans_aud["cur"])[0]

    if output_dir is None:
        output_dir = FULL_RESULT_DIR

    summary = execute_full_pipeline(obs_day, merged, aligned, fit_A, fit_B,
                                    bars_by_sym, output_dir)
    summary["n_all_obs"] = int(aud["n_all_obs"])
    summary["n_H0"] = int(aud["n_hazard0"])
    summary["n_H1"] = int(aud["n_hazard1"])
    summary["symbols"] = sorted(aud["symbols"])
    summary["max_abs_atr0_owner_error"] = float(max_atr0_err)
    summary["sample_artifact_sha256"] = hashes["sample_artifact_sha256"]
    summary["transition_artifact_sha256"] = hashes["transition_artifact_sha256"]
    summary["WindowA_baseline_parity"] = par_A
    summary["WindowB_baseline_parity"] = par_B
    (Path(output_dir) / f"{PREFIX}_formal_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
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
