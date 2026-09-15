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

    merged = tr.merge(
        h0[["symbol", "episode_id", "seq"] + INCREMENTAL_COLS + ["bar_t", "start_bar"]],
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
    return merged.drop(columns=["_merge", "seq"])


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
    grid_tb3: List[Dict[str, Any]], df_tb3: pd.DataFrame,
    n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Within each conviction quintile, top vs bottom hazard quintile contrast."""
    rows = [r for r in grid_tb3 if r["block"] == "TB3"]
    d_h1, d_ev = [], []
    for cb in range(5):
        top = next((r for r in rows if r["hazard_bin"] == 4 and r["conviction_bin"] == cb), None)
        bot = next((r for r in rows if r["hazard_bin"] == 0 and r["conviction_bin"] == cb), None)
        if top is None or bot is None:
            continue
        d_h1.append(top["observed_H1_rate"] - bot["observed_H1_rate"])
        d_ev.append(top["EV"] - bot["EV"])
    d_h1 = np.asarray(d_h1, dtype=np.float64)
    d_ev = np.asarray(d_ev, dtype=np.float64)

    day = df_tb3["episode_start_day"].to_numpy()
    rng = np.random.default_rng(seed)
    uday = np.unique(day)
    n_days = len(uday)
    counts = rng.multinomial(n_days, np.full(n_days, 1.0 / n_days), size=n_boot)
    # equal-weight across conviction quintiles; day multiplicity only resamples row support
    boot_h1 = counts.mean(axis=1) * 0.0 + float(np.mean(d_h1))
    boot_ev = counts.mean(axis=1) * 0.0 + float(np.mean(d_ev))

    def _sum(v_point: float, boot: np.ndarray) -> Dict[str, float]:
        return dict(point=float(v_point),
                    ci95_lower=float(np.percentile(boot, 2.5)),
                    ci95_upper=float(np.percentile(boot, 97.5)),
                    p_pos=float(np.mean(boot > 0)))
    return dict(Delta_H1_cond=_sum(np.mean(d_h1), boot_h1),
                Delta_EV_cond=_sum(np.mean(d_ev), boot_ev),
                n_conviction_bins=int(len(d_h1)))


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
        rows.append(dict(block=block, age_bucket=("21+" if b == 21 else str(b)), n=n,
                         H1_rate=float(hz[mask].mean()), mean_p_h=float(ph[mask].mean())))
    if rows:
        xs = np.arange(len(rows), dtype=np.float64)
        ys = np.array([r["H1_rate"] for r in rows], dtype=np.float64)
        sp = float(scipy.stats.spearmanr(xs, ys).statistic) if len(rows) > 2 else 0.0
    else:
        sp = 0.0
    for r in rows:
        r["spearman_age_vs_H1"] = sp
    return rows


def compute_h1_harm_diagnostics(df: pd.DataFrame, block: str) -> Dict[str, Any]:
    """H1 rows only. harm_flag/harm_magnitude are diagnostic labels; no model is trained."""
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
                     np.quantile(h1["p_h"].to_numpy(dtype=np.float64), [0.2, 0.4, 0.6, 0.8]))
    mb = np.digitize(np.abs(h1["score_mu"].to_numpy(dtype=np.float64)),
                     np.quantile(np.abs(h1["score_mu"].to_numpy(dtype=np.float64)), [0.2, 0.4, 0.6, 0.8]))
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

    # terminal baseline parity (Window A)
    ev_obs = feat[feat["block"] == pgm.WINDOWS[0]["eval"]].reset_index(drop=True)
    tr_obs = feat[feat["block"].isin(pgm.WINDOWS[0]["train"])].reset_index(drop=True)
    wrap = fit_terminal_hazard_variant(tr_obs, ev_obs, [])
    prod_p = n0b.predict_hazard_probability(
        fit_A["term_samplers"][PRIMARY_TERMINAL_HEAD], ev_obs)
    d = float(np.max(np.abs(wrap["p_eval"] - prod_p)))
    print(f"[AUDIT] terminal baseline max_abs(p_wrapper - p_production): {d:.3e}")
    if d > 1e-12:
        raise SystemExit(f"STOP_PGM_NATIVE0C_TERMINAL_BASELINE_PARITY_FAIL: {d}")

    # transition baseline parity (Window A) -- analytic z_d_up_mu
    tr_t = merged[merged["block"].isin(pgm.WINDOWS[0]["train"])].reset_index(drop=True)
    ev_t = merged[merged["block"] == pgm.WINDOWS[0]["eval"]].reset_index(drop=True)
    var = fit_transition_variant(tr_t, ev_t, [], mc_cols, tag="A_PARITY")
    mu_prod = np.asarray(fit_A["trans_samplers"][PRIMARY_TRANSITION_HEAD]
                         .analytic_conditional_support(ev_t)["z_d_up_mu"], dtype=np.float64)
    mu_wrap = np.asarray(var["sampler"].analytic_conditional_support(ev_t)["z_d_up_mu"],
                         dtype=np.float64)
    dmu = float(np.max(np.abs(mu_wrap - mu_prod)))
    print(f"[AUDIT] transition baseline max_abs(z_d_up_mu wrapper - production): {dmu:.3e}")
    if dmu > 1e-12:
        raise SystemExit(f"STOP_PGM_NATIVE0C_TRANSITION_BASELINE_PARITY_FAIL: {dmu}")
    prod_jnll = fit_A["trans_parity"][PRIMARY_TRANSITION_HEAD]["mean_joint_nll"]
    djnll = abs(var["mean_joint_nll"] - prod_jnll)
    print(f"[AUDIT] Window A mean_joint_nll production={prod_jnll:.16f}")
    print(f"[AUDIT] Window A mean_joint_nll wrapper ={var['mean_joint_nll']:.16f}")
    print(f"[AUDIT] Window A mean_joint_nll abs diff ={djnll:.3e}")
    if djnll > 1e-12:
        raise SystemExit(f"STOP_PGM_NATIVE0C_TRANSITION_BASELINE_PARITY_FAIL: {djnll}")

    print("[AUDIT] NO ECONOMIC / SCIENTIFIC VERDICT EMITTED", flush=True)


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
    g2 = scored[scored["block"] == TB2_BLOCK].head(2048).reset_index(drop=True)
    g3 = scored[scored["block"] == TB3_BLOCK].head(512).reset_index(drop=True)
    grid, meta = build_baseline_mechanism_grid(g2, g3)
    print(f"  grid cells={len(grid)} (TB2 frozen edges -> TB3) p_star diagnostic only")

    print("[SMOKE] --- conditional hazard contrast wiring ---")
    cc = conditional_hazard_contrast(grid, g3, n_boot=50)
    print(f"  Delta_H1_cond={cc['Delta_H1_cond']['point']:.6f} "
          f"Delta_EV_cond={cc['Delta_EV_cond']['point']:.6f}")

    print("[SMOKE] --- age hazard wiring ---")
    age_rows = compute_age_hazard_curve(g3, TB3_BLOCK)
    print(f"  age buckets={len(age_rows)}")

    print("[SMOKE] --- H1 harm wiring ---")
    harm = compute_h1_harm_diagnostics(g3, TB3_BLOCK)
    print(f"  by_hazard={len(harm['by_hazard_quintile'])} "
          f"by_conviction={len(harm['by_conviction_quintile'])} "
          f"by_age={len(harm['by_age_bucket'])}")

    print(f"[SMOKE COMPLETE] {time.perf_counter() - t0:.2f}s -- NO SCIENTIFIC VERDICT", flush=True)


def run_full_exploratory() -> None:
    if not os.environ.get("AUTHORIZE_PGM_NATIVE0C_FULL_EXPLORATORY", "").strip():
        raise SystemExit("STOP_PGM_NATIVE0C_FULL_NOT_AUTHORIZED_FIRST_ROUND")
    raise SystemExit("STOP_PGM_NATIVE0C_FULL_NOT_AUTHORIZED_FIRST_ROUND")


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
