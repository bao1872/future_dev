"""
experiment_pgm_native0b_hazard_reliability_v1.py
================================================

PGM-NATIVE-0B: Predicted Hazard Reliability Probe
Evaluating Whether PGM Predicted Terminal Hazard Probability Explains and Attenuates
Transition Signal Reliability Failure on Frozen PGM Bar Sample.

RESEARCH BACKGROUND & CAUSAL HYPOTHESIS
---------------------------------------
In PGM-NATIVE-0A formal evaluation, the conditional nonterminal transition model
m(S_t) = E[-z_{d, t+1} | S_t, H_{t+1}=0] demonstrated robust predictive structure
(rho_nonterminal = 0.1258, CI95 = [0.1187, 0.1331] in TB3, 15/15 symbols positive),
but raw directional trading (action = sign(m_t)) suffered from severe terminal mismatch:
    TB3 H0 (89.4%): EV = +0.03936 ATR0, win_rate = 48.24%, PF = 1.2355
    TB3 H1 (10.6%): EV = -0.31873 ATR0, win_rate = 32.12%, PF = 0.4663
    ALL   (100%):   EV = +0.00139 ATR0, CI95 = [-0.00475, +0.00725] (crosses zero)

The 10.6% terminal rows eroded 96.5% of the nonterminal edge. This is not ordinary noise,
but an intrinsic model-regime mismatch: a model conditioned on survival (H_{t+1}=0) was
forced to act on terminal transitions (H_{t+1}=1).

The falsifiable hypothesis of PGM-NATIVE-0B is:
    "If the frozen PGM terminal hazard head (T2_STATE_PHI_MEM) can causally predict
     terminal transition risk at bar t close (p_h(t) = P(H_{t+1}=1 | F_t)), then:
     1. Higher predicted hazard p_h will correlate with higher realized H1 failure rate
        and systematic degradation in 0A transition strategy EV.
     2. A survival-weighted policy w_t = (1 - p_h(t)) * sign(m_t), without any threshold tuning,
        will attenuate terminal losses and improve exposure-normalized EV (ENEV)."

RESEARCH SCOPE & METHODOLOGICAL DISCLOSURE
------------------------------------------
EXPERIMENT_SCOPE = "EXPLORATORY_MECHANISM_VALIDATION_ON_PREVIOUSLY_INSPECTED_TB3"

Methodological Notice:
TB3 was previously inspected for H0/H1 diagnostics in PGM-NATIVE-0A. While model fitting
(Window A on TB1, Window B on TB1+TB2) preserves strictly causal forward temporal splits,
the 0B survival-weighting hypothesis itself was formed after observing 0A TB3 H0/H1 breakdown.
Therefore, TB3 results here constitute EXPLORATORY mechanism validation, NOT a pristine,
untouched out-of-sample holdout or final trading strategy confirmation.
Prospective validation would require fresh time blocks (e.g. TB4 or live holdout).

FORBIDDEN ELEMENTS:
    - No V2 indicators, R1-R4 rules, Q-learning, RL, Ridge/HGB meta-models
    - No true hazard filtering or future lookahead in trading decisions
    - No threshold search or holding-period tuning on TB3
    - No symbol pruning or cherry-picking
    - Forbidden verdicts: {"PROFITABLE_STRATEGY", "FINAL_ALPHA", "LIVE_READY", "FULL_MARKET_OOS", "FINAL_STRATEGY"}
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
from typing import Any, Dict, List, Optional, Tuple

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

import research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 as ex0
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1c_free_run_rollout_v1 as pgm
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a1b_count_magnitude_closure_v1 as base
import research.liquidity_oracle_atlas.experiment_pgm_native0a_one_step_alpha_v1 as n0a


# ===========================================================================
# Governance Constants
# ===========================================================================
BASE_SHA = "dfb2bba5edeccd731b84610754798537ab3ef674"
EXPERIMENT_NAME = "PGM-NATIVE-0B -- Predicted Hazard Reliability Probe"
EXPERIMENT_SCOPE = "EXPLORATORY_MECHANISM_VALIDATION_ON_PREVIOUSLY_INSPECTED_TB3"
PREFIX = "pgm_native0b1"

FORBIDDEN_VERDICTS = {
    "PROFITABLE_STRATEGY",
    "FINAL_ALPHA",
    "LIVE_READY",
    "FULL_MARKET_OOS",
    "FINAL_STRATEGY",
    "PRISTINE_HOLDOUT",
    "FINAL_CONFIRMATION",
}

VERDICT_STRINGS = {
    "HAZARD_NOT_SUPPORTED": "PGM_HAZARD_RELIABILITY_NOT_SUPPORTED",
    "HAZARD_PREDICTIVE_BUT_VALUE_NOT_SUPPORTED": "PGM_HAZARD_PREDICTIVE_BUT_TRANSITION_RELIABILITY_VALUE_NOT_SUPPORTED",
    "RELIABILITY_SUPPORTED_EXPLORATORY": "PGM_HAZARD_CONDITIONED_RELIABILITY_SUPPORTED_EXPLORATORY",
}

BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20260915
COST_ATR0_GRID = n0a.COST_ATR0_GRID
EXPECTED_SYMBOLS = n0a.EXPECTED_SYMBOLS

PRIMARY_TRANSITION_HEAD = "MC_STATE_CURREENCODING"
PRIMARY_TERMINAL_HEAD = "T2_STATE_PHI_MEM"
DIAGNOSTIC_TERMINAL_HEAD = "T0_STATE_AVAIL"


# ===========================================================================
# Reuse Map
# ===========================================================================
REUSE_MAP = {
    "decision_universe": "n0a.load_observed_decision_universe() -> pgm.SAMPLE_PATH (ALL observation rows)",
    "universe_audit": "n0a.audit_decision_universe()",
    "transition_truth": "n0a.load_transition_truth_audit() -> cur_truth",
    "atr0_owner_parity": "n0a.audit_atr0_owner_parity(obs, cur_truth)",
    "raw_bars_and_returns": "n0a.align_raw_bars_and_returns(obs, bars_by_sym, cur_truth)",
    "symbol_breadth_helper": "n0a.compute_symbol_breadth",
    "cost_grid": "n0a.COST_ATR0_GRID",
    "expected_symbols": "n0a.EXPECTED_SYMBOLS",
    "pgm_fit": "pgm.fit_samplers_for_window(pgm.WINDOWS[0/1], pgm.SAMPLE_PATH, pgm.TRANSITION_SAMPLE_PATH)",
    "safe_transform": "pgm.safe_transform",
}


def print_reuse_map() -> None:
    print("==================================================", flush=True)
    print("PGM-NATIVE-0B REUSE MAP:", flush=True)
    for k, v in REUSE_MAP.items():
        print(f"  REUSE {k}: {v}", flush=True)
    print("==================================================", flush=True)


# ===========================================================================
# Deterministic Hazard Probability Wrapper
# ===========================================================================
def predict_hazard_probability(term_sampler: Any, df: pd.DataFrame) -> np.ndarray:
    """Deterministic hazard probability prediction using frozen terminal sampler internals.

    Must achieve exact parity with:
        term_sampler.sample_hazard(df, FixedRNG(u)) == (u < predict_hazard_probability(term_sampler, df))
    """
    X = pgm.safe_transform(
        term_sampler.pre,
        term_sampler._design_df(df),
        "TERMINAL_INPUT_NONFINITE",
        "TERMINAL_DESIGN_NONFINITE",
        term_sampler.num_cols,
        term_sampler.allowed_nan_cols,
    )
    p_h = np.asarray(term_sampler.clf.predict_proba(X)[:, 1], dtype=np.float64)
    if not np.all(np.isfinite(p_h)):
        raise SystemExit("STOP_PGM_NATIVE0B_HAZARD_PROBABILITY_NONFINITE")
    if np.any((p_h < 0.0) | (p_h > 1.0)):
        raise SystemExit("STOP_PGM_NATIVE0B_HAZARD_PROBABILITY_OUT_OF_BOUNDS")
    return p_h


class FixedRNG:
    """Deterministic pseudo-RNG for validating sampler probability parity."""
    def __init__(self, u: np.ndarray):
        self.u = np.asarray(u, dtype=np.float64)

    def random(self, n: int) -> np.ndarray:
        return self.u[:n]


def audit_hazard_probability_parity(term_sampler: Any, df_sample: pd.DataFrame) -> bool:
    """Prove deterministic p_h wrapper has exact parity with FittedTerminalSampler.sample_hazard."""
    p_h = predict_hazard_probability(term_sampler, df_sample)
    n = len(df_sample)
    # Test multiple arbitrary threshold levels
    for u_val in [0.05, 0.10, 0.20, 0.50, 0.80]:
        u_arr = np.full(n, u_val, dtype=np.float64)
        rng = FixedRNG(u_arr)
        sample_res = term_sampler.sample_hazard(df_sample, rng)
        direct_res = u_arr < p_h
        if not np.array_equal(sample_res, direct_res):
            raise SystemExit("STOP_PGM_NATIVE0B_HAZARD_PROBABILITY_PARITY_FAIL")
    return True


# ===========================================================================
# Score and Policy Evaluation
# ===========================================================================
def score_and_evaluate_policies(
    df_block: pd.DataFrame,
    mc_sampler: Any,
    term_sampler_t2: Any,
    term_sampler_t0: Optional[Any] = None,
) -> pd.DataFrame:
    """Compute directional score, hazard probabilities, and 0A/0B policies on block rows.

    0A Baseline:
        m_t = score_mu = -E[z_{d, t+1} | S_t, H_{t+1}=0]
        action_0a = sign(m_t) in {-1, 0, +1}
        ret_0a = action_0a * r_trad_OC_ATR0
        exp_0a = abs(action_0a)

    0B Primary (Survival-Weighted, No Threshold):
        p_h(t) = P(H_{t+1}=1 | F_t) via T2_STATE_PHI_MEM
        survival_prob = 1.0 - p_h(t) in [0, 1]
        position_0b = survival_prob * sign(m_t) in [-1, +1]
        ret_0b = position_0b * r_trad_OC_ATR0
        exp_0b = abs(position_0b)
    """
    out = df_block.copy()

    # 1. PGM Transition Score
    mom = mc_sampler.analytic_conditional_support(out)
    out["score_mu"] = -np.asarray(mom["z_d_up_mu"], dtype=np.float64)

    # 2. 0A Baseline Policy
    s = out["score_mu"].to_numpy(float)
    act_0a = np.where(s > 0, 1, np.where(s < 0, -1, 0))
    out["action_0a"] = act_0a
    out["exposure_0a"] = np.abs(act_0a)
    out["ret_0a"] = act_0a * out["r_trad_OC_ATR0"].to_numpy(float)

    # 3. Deterministic Predicted Hazard (T2 Primary)
    p_h_t2 = predict_hazard_probability(term_sampler_t2, out)
    out["p_h"] = p_h_t2
    out["survival_prob"] = 1.0 - p_h_t2

    # 4. 0B Survival-Weighted Policy
    pos_0b = out["survival_prob"].to_numpy(float) * act_0a
    # Hard assert bounds [-1, +1]
    if np.any((pos_0b < -1.0 - 1e-12) | (pos_0b > 1.0 + 1e-12)):
        raise SystemExit("STOP_PGM_NATIVE0B_POSITION_OUT_OF_BOUNDS")
    pos_0b = np.clip(pos_0b, -1.0, 1.0)
    out["position_0b"] = pos_0b
    out["exposure_0b"] = np.abs(pos_0b)
    out["ret_0b"] = pos_0b * out["r_trad_OC_ATR0"].to_numpy(float)

    # 5. Optional T0 Diagnostic Sensitivity
    if term_sampler_t0 is not None:
        out["p_h_t0"] = predict_hazard_probability(term_sampler_t0, out)

    return out


# ===========================================================================
# Hazard Head Verification Metrics
# ===========================================================================
def compute_hazard_metrics(hazard: np.ndarray, p_h: np.ndarray) -> Dict[str, float]:
    """Evaluate statistical quality of predicted hazard probabilities."""
    y_true = np.asarray(hazard, dtype=int)
    y_prob = np.asarray(p_h, dtype=np.float64)

    prev = float(np.mean(y_true))
    roc_auc = float(sklearn.metrics.roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(sklearn.metrics.average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else prev

    # Clip probabilities for robust log loss
    p_clipped = np.clip(y_prob, 1e-15, 1.0 - 1e-15)
    ll = float(sklearn.metrics.log_loss(y_true, p_clipped))

    brier = float(np.mean((y_prob - y_true) ** 2))
    brier_const = float(np.mean((prev - y_true) ** 2))
    brier_skill = float(1.0 - brier / brier_const) if brier_const > 1e-15 else 0.0

    return dict(
        hazard_prevalence=prev,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        log_loss=ll,
        brier_score=brier,
        brier_constant=brier_const,
        brier_skill=brier_skill,
    )


# ===========================================================================
# Hazard Decile Reliability Analysis
# ===========================================================================
def compute_hazard_decile_edges(p_h_tb2: np.ndarray) -> np.ndarray:
    """Compute 10 quantile edges on TB2 predicted hazard."""
    quantiles = np.linspace(0.1, 0.9, 9)
    inner_edges = np.quantile(p_h_tb2, quantiles)
    return np.concatenate([[-np.inf], inner_edges, [np.inf]])


def evaluate_hazard_deciles(df: pd.DataFrame, edges: np.ndarray) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Evaluate observed failure rate and policy EV across predicted hazard deciles."""
    p_h = df["p_h"].to_numpy(float)
    haz = df["hazard"].to_numpy(int)
    score = df["score_mu"].to_numpy(float)
    ret_0a = df["ret_0a"].to_numpy(float)
    ret_0b = df["ret_0b"].to_numpy(float)
    r_trad = df["r_trad_OC_ATR0"].to_numpy(float)

    total_h1 = max(1, int(np.sum(haz == 1)))
    n_bins = len(edges) - 1
    bins = pd.cut(p_h, bins=edges, labels=False, include_lowest=True, duplicates="drop")

    rows = []
    bin_p_h = []
    bin_obs_h1 = []
    bin_ev_0a = []

    for b in range(n_bins):
        mask = bins == b
        n_b = int(np.sum(mask))
        if n_b == 0:
            continue
        m_ph = float(np.mean(p_h[mask]))
        obs_h1 = float(np.mean(haz[mask]))
        m_abs_score = float(np.mean(np.abs(score[mask])))
        ev_0a = float(np.mean(ret_0a[mask]))
        ev_0b = float(np.mean(ret_0b[mask]))
        h1_cnt = int(np.sum(haz[mask] == 1))
        h1_sh = float(h1_cnt / total_h1)

        rho_tr = float(scipy.stats.spearmanr(score[mask], r_trad[mask]).statistic) if n_b > 2 else 0.0

        rows.append(dict(
            bin_idx=b,
            n=n_b,
            mean_p_h=m_ph,
            observed_H1_rate=obs_h1,
            mean_abs_score=m_abs_score,
            EV_0a=ev_0a,
            EV_0b=ev_0b,
            rho_trad_0a=rho_tr,
            H1_share=h1_sh,
        ))
        bin_p_h.append(m_ph)
        bin_obs_h1.append(obs_h1)
        bin_ev_0a.append(ev_0a)

    # Monotonicity rank correlations
    sp_h1 = float(scipy.stats.spearmanr(bin_p_h, bin_obs_h1).statistic) if len(bin_p_h) > 2 else 0.0
    sp_ev = float(scipy.stats.spearmanr(bin_p_h, bin_ev_0a).statistic) if len(bin_p_h) > 2 else 0.0

    stats = dict(
        spearman_p_h_vs_observed_H1=sp_h1,
        spearman_p_h_vs_0a_EV=sp_ev,
    )
    return rows, stats


# ===========================================================================
# Ex-Post Subgroup Diagnostics (H0 vs H1)
# ===========================================================================
def compute_hazard_subgroups(df: pd.DataFrame) -> Dict[str, Any]:
    """Ex-post diagnostic metrics on H0 vs H1 rows (forbidden for policy tuning)."""
    h0_mask = df["hazard"] == 0
    h1_mask = df["hazard"] == 1

    ev_0a_h0 = float(np.mean(df.loc[h0_mask, "ret_0a"])) if h0_mask.any() else 0.0
    ev_0b_h0 = float(np.mean(df.loc[h0_mask, "ret_0b"])) if h0_mask.any() else 0.0
    exp_0b_h0 = float(np.mean(df.loc[h0_mask, "exposure_0b"])) if h0_mask.any() else 0.0
    p_h_h0 = float(np.mean(df.loc[h0_mask, "p_h"])) if h0_mask.any() else 0.0

    ev_0a_h1 = float(np.mean(df.loc[h1_mask, "ret_0a"])) if h1_mask.any() else 0.0
    ev_0b_h1 = float(np.mean(df.loc[h1_mask, "ret_0b"])) if h1_mask.any() else 0.0
    exp_0b_h1 = float(np.mean(df.loc[h1_mask, "exposure_0b"])) if h1_mask.any() else 0.0
    p_h_h1 = float(np.mean(df.loc[h1_mask, "p_h"])) if h1_mask.any() else 0.0

    # Attenuation of terminal loss: 1 - |EV_0B_H1| / |EV_0A_H1|
    if abs(ev_0a_h1) > 1e-12:
        attenuation = float(1.0 - abs(ev_0b_h1) / abs(ev_0a_h1))
    else:
        attenuation = 0.0

    return dict(
        H0=dict(
            n=int(h0_mask.sum()),
            mean_p_h=p_h_h0,
            EV_0a=ev_0a_h0,
            EV_0b=ev_0b_h0,
            mean_exposure_0b=exp_0b_h0,
        ),
        H1=dict(
            n=int(h1_mask.sum()),
            mean_p_h=p_h_h1,
            EV_0a=ev_0a_h1,
            EV_0b=ev_0b_h1,
            mean_exposure_0b=exp_0b_h1,
        ),
        terminal_loss_attenuation=attenuation,
    )


# ===========================================================================
# Paired Day-Clustered Bootstrap
# ===========================================================================
def run_paired_day_clustered_bootstrap(
    df_scored: pd.DataFrame,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Dict[str, float]]:
    """Trading-day clustered paired bootstrap comparing 0A and 0B policies.

    CRITICAL: 0A and 0B evaluate on the EXACT SAME resampled trading days per replicate.
    """
    days = df_scored["entry_day"].to_numpy()
    ret_0a = df_scored["ret_0a"].to_numpy(float)
    ret_0b = df_scored["ret_0b"].to_numpy(float)
    exp_0a = df_scored["exposure_0a"].to_numpy(float)
    exp_0b = df_scored["exposure_0b"].to_numpy(float)
    haz = df_scored["hazard"].to_numpy(int)
    p_h = df_scored["p_h"].to_numpy(float)

    # Point estimates on full sample
    pt_ev_0a = float(np.mean(ret_0a))
    pt_ev_0b = float(np.mean(ret_0b))
    pt_delta_ev = pt_ev_0b - pt_ev_0a

    sum_exp_0a = float(np.sum(exp_0a))
    sum_exp_0b = float(np.sum(exp_0b))
    pt_enev_0a = float(np.sum(ret_0a) / sum_exp_0a) if sum_exp_0a > 0 else 0.0
    pt_enev_0b = float(np.sum(ret_0b) / sum_exp_0b) if sum_exp_0b > 0 else 0.0
    pt_delta_enev = pt_enev_0b - pt_enev_0a

    pt_auc = float(sklearn.metrics.roc_auc_score(haz, p_h)) if len(np.unique(haz)) > 1 else 0.5

    # Pre-index days
    unique_days, inverse = np.unique(days, return_inverse=True)
    n_days = len(unique_days)
    day_indices = [np.flatnonzero(inverse == d) for d in range(n_days)]

    rng = np.random.default_rng(seed)

    boot_ev_0a = np.empty(n_boot, dtype=np.float64)
    boot_ev_0b = np.empty(n_boot, dtype=np.float64)
    boot_delta_ev = np.empty(n_boot, dtype=np.float64)

    boot_enev_0a = np.empty(n_boot, dtype=np.float64)
    boot_enev_0b = np.empty(n_boot, dtype=np.float64)
    boot_delta_enev = np.empty(n_boot, dtype=np.float64)

    boot_auc = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        sampled_d = rng.choice(n_days, size=n_days, replace=True)
        idx_b = np.concatenate([day_indices[d] for d in sampled_d])

        r0a_b = ret_0a[idx_b]
        r0b_b = ret_0b[idx_b]
        e0a_b = exp_0a[idx_b]
        e0b_b = exp_0b[idx_b]
        haz_b = haz[idx_b]
        ph_b = p_h[idx_b]

        m_ev_0a = np.mean(r0a_b)
        m_ev_0b = np.mean(r0b_b)
        boot_ev_0a[b] = m_ev_0a
        boot_ev_0b[b] = m_ev_0b
        boot_delta_ev[b] = m_ev_0b - m_ev_0a

        s_e0a = np.sum(e0a_b)
        s_e0b = np.sum(e0b_b)
        m_enev_0a = np.sum(r0a_b) / s_e0a if s_e0a > 0 else 0.0
        m_enev_0b = np.sum(r0b_b) / s_e0b if s_e0b > 0 else 0.0
        boot_enev_0a[b] = m_enev_0a
        boot_enev_0b[b] = m_enev_0b
        boot_delta_enev[b] = m_enev_0b - m_enev_0a

        if len(np.unique(haz_b)) > 1:
            boot_auc[b] = sklearn.metrics.roc_auc_score(haz_b, ph_b)
        else:
            boot_auc[b] = 0.5

    def _summarize(pt: float, dist: np.ndarray) -> Dict[str, float]:
        lo = float(np.percentile(dist, 2.5))
        hi = float(np.percentile(dist, 97.5))
        p_pos = float(np.mean(dist > 0))
        return dict(point=pt, ci95_lower=lo, ci95_upper=hi, p_pos=p_pos)

    return dict(
        hazard_auc=_summarize(pt_auc, boot_auc),
        EV_0A=_summarize(pt_ev_0a, boot_ev_0a),
        EV_0B=_summarize(pt_ev_0b, boot_ev_0b),
        Delta_EV=_summarize(pt_delta_ev, boot_delta_ev),
        ENEV_0A=_summarize(pt_enev_0a, boot_enev_0a),
        ENEV_0B=_summarize(pt_enev_0b, boot_enev_0b),
        Delta_ENEV=_summarize(pt_delta_enev, boot_delta_enev),
    )


# ===========================================================================
# Cost Stress Grid
# ===========================================================================
def run_cost_stress_grid_0b(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Cost stress grid testing for 0A (discrete) vs 0B (fractional exposure).

    Friction formula:
        net_ret_0a = ret_0a - c * exposure_0a
        net_ret_0b = ret_0b - c * exposure_0b
    """
    ret_0a = df["ret_0a"].to_numpy(float)
    ret_0b = df["ret_0b"].to_numpy(float)
    exp_0a = df["exposure_0a"].to_numpy(float)
    exp_0b = df["exposure_0b"].to_numpy(float)

    mean_exp_0a = float(np.mean(exp_0a))
    mean_exp_0b = float(np.mean(exp_0b))
    sum_exp_0a = float(np.sum(exp_0a))
    sum_exp_0b = float(np.sum(exp_0b))

    raw_ev_0a = float(np.mean(ret_0a))
    raw_ev_0b = float(np.mean(ret_0b))
    be_0a = float(raw_ev_0a / mean_exp_0a) if mean_exp_0a > 0 else 0.0
    be_0b = float(raw_ev_0b / mean_exp_0b) if mean_exp_0b > 0 else 0.0

    rows = []
    for c in COST_ATR0_GRID:
        net_0a = ret_0a - c * exp_0a
        net_0b = ret_0b - c * exp_0b

        net_ev_0a = float(np.mean(net_0a))
        net_ev_0b = float(np.mean(net_0b))
        net_enev_0a = float(np.sum(net_0a) / sum_exp_0a) if sum_exp_0a > 0 else 0.0
        net_enev_0b = float(np.sum(net_0b) / sum_exp_0b) if sum_exp_0b > 0 else 0.0
        tot_net_0a = float(np.sum(net_0a))
        tot_net_0b = float(np.sum(net_0b))

        rows.append(dict(
            cost_ATR0=c,
            net_EV_0A=net_ev_0a,
            net_EV_0B=net_ev_0b,
            net_ENEV_0A=net_enev_0a,
            net_ENEV_0B=net_enev_0b,
            total_net_0A=tot_net_0a,
            total_net_0B=tot_net_0b,
            break_even_cost_0A=be_0a,
            break_even_cost_0B=be_0b,
        ))
    return rows


# ===========================================================================
# 15-Symbol Breadth Analysis
# ===========================================================================
def compute_symbol_breadth_0b(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Per-symbol evaluation across all 15 symbols in the continuous main universe."""
    rows = []
    n_delta_ev_pos = 0
    n_delta_enev_pos = 0

    for sym in EXPECTED_SYMBOLS:
        sub = df[df["symbol"] == sym]
        n_sym = len(sub)
        if n_sym == 0:
            continue

        haz_sym = sub["hazard"].to_numpy(int)
        ph_sym = sub["p_h"].to_numpy(float)
        ret_0a = sub["ret_0a"].to_numpy(float)
        ret_0b = sub["ret_0b"].to_numpy(float)
        exp_0a = sub["exposure_0a"].to_numpy(float)
        exp_0b = sub["exposure_0b"].to_numpy(float)

        auc = float(sklearn.metrics.roc_auc_score(haz_sym, ph_sym)) if len(np.unique(haz_sym)) > 1 else 0.5
        m_ph = float(np.mean(ph_sym))

        ev_0a = float(np.mean(ret_0a))
        ev_0b = float(np.mean(ret_0b))
        d_ev = ev_0b - ev_0a

        s_e0a = float(np.sum(exp_0a))
        s_e0b = float(np.sum(exp_0b))
        enev_0a = float(np.sum(ret_0a) / s_e0a) if s_e0a > 0 else 0.0
        enev_0b = float(np.sum(ret_0b) / s_e0b) if s_e0b > 0 else 0.0
        d_enev = enev_0b - enev_0a

        if d_ev > 0:
            n_delta_ev_pos += 1
        if d_enev > 0:
            n_delta_enev_pos += 1

        # Terminal loss attenuation on symbol
        h1_mask = haz_sym == 1
        if h1_mask.any():
            ev_0a_h1 = float(np.mean(ret_0a[h1_mask]))
            ev_0b_h1 = float(np.mean(ret_0b[h1_mask]))
            att = float(1.0 - abs(ev_0b_h1) / abs(ev_0a_h1)) if abs(ev_0a_h1) > 1e-12 else 0.0
        else:
            att = 0.0

        rows.append(dict(
            symbol=sym,
            n=n_sym,
            hazard_auc=auc,
            mean_p_h=m_ph,
            EV_0a=ev_0a,
            EV_0b=ev_0b,
            Delta_EV=d_ev,
            ENEV_0a=enev_0a,
            ENEV_0b=enev_0b,
            Delta_ENEV=d_enev,
            terminal_loss_attenuation=att,
        ))

    counts = dict(
        n_symbols_total=len(rows),
        n_symbols_Delta_EV_positive=n_delta_ev_pos,
        n_symbols_Delta_ENEV_positive=n_delta_enev_pos,
    )
    return rows, counts


# ===========================================================================
# Pre-Registered Verdict Logic
# ===========================================================================
def determine_formal_verdict_0b(boot_tb3: Dict[str, Any]) -> str:
    """Pre-registered verdict logic based on TB3 paired bootstrap results.

    1. If hazard ROC-AUC day-cluster CI95 lower <= 0.5:
       -> PGM_HAZARD_RELIABILITY_NOT_SUPPORTED
    2. Else if Delta_ENEV CI95 lower <= 0:
       -> PGM_HAZARD_PREDICTIVE_BUT_TRANSITION_RELIABILITY_VALUE_NOT_SUPPORTED
    3. Else:
       -> PGM_HAZARD_CONDITIONED_RELIABILITY_SUPPORTED_EXPLORATORY
    """
    auc_ci_lower = boot_tb3["hazard_auc"]["ci95_lower"]
    delta_enev_ci_lower = boot_tb3["Delta_ENEV"]["ci95_lower"]

    if auc_ci_lower <= 0.5:
        return VERDICT_STRINGS["HAZARD_NOT_SUPPORTED"]
    elif delta_enev_ci_lower <= 0.0:
        return VERDICT_STRINGS["HAZARD_PREDICTIVE_BUT_VALUE_NOT_SUPPORTED"]
    else:
        return VERDICT_STRINGS["RELIABILITY_SUPPORTED_EXPLORATORY"]


# ===========================================================================
# Full Exploratory Pipeline Executor
# ===========================================================================
def execute_exploratory_pipeline(
    obs: pd.DataFrame,
    bars_by_sym: Dict[str, Any],
    cur_truth: pd.DataFrame,
    fit_A: Dict[str, Any],
    fit_B: Dict[str, Any],
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    output_dir: Optional[Path] = None,
    max_abs_atr0_owner_error: Optional[float] = None,
) -> Dict[str, Any]:
    """Execute exploratory pipeline comparing 0A and 0B across TB2 and TB3."""
    aligned, align_aud = n0a.align_raw_bars_and_returns(obs, bars_by_sym, cur_truth=cur_truth)
    df_valid = aligned[aligned["is_entry_valid"]].copy()

    # Split TB2 and TB3
    df_tb2 = df_valid[df_valid["block"] == n0a.TB2_BLOCK].copy()
    df_tb3 = df_valid[df_valid["block"] == n0a.TB3_BLOCK].copy()

    # Route: TB2 evaluated with Window A models, TB3 with Window B models
    mc_A = fit_A["trans_samplers"][PRIMARY_TRANSITION_HEAD]
    term_A_t2 = fit_A["term_samplers"][PRIMARY_TERMINAL_HEAD]
    term_A_t0 = fit_A["term_samplers"][DIAGNOSTIC_TERMINAL_HEAD]

    mc_B = fit_B["trans_samplers"][PRIMARY_TRANSITION_HEAD]
    term_B_t2 = fit_B["term_samplers"][PRIMARY_TERMINAL_HEAD]
    term_B_t0 = fit_B["term_samplers"][DIAGNOSTIC_TERMINAL_HEAD]

    tb2_scored = score_and_evaluate_policies(df_tb2, mc_A, term_A_t2, term_A_t0)
    tb3_scored = score_and_evaluate_policies(df_tb3, mc_B, term_B_t2, term_B_t0)

    # 1. Hazard quality metrics
    haz_met_tb2 = compute_hazard_metrics(tb2_scored["hazard"].to_numpy(int), tb2_scored["p_h"].to_numpy(float))
    haz_met_tb3 = compute_hazard_metrics(tb3_scored["hazard"].to_numpy(int), tb3_scored["p_h"].to_numpy(float))

    # T0 baseline metrics (diagnostic)
    t0_met_tb2 = compute_hazard_metrics(tb2_scored["hazard"].to_numpy(int), tb2_scored["p_h_t0"].to_numpy(float))
    t0_met_tb3 = compute_hazard_metrics(tb3_scored["hazard"].to_numpy(int), tb3_scored["p_h_t0"].to_numpy(float))

    # 2. Hazard deciles (frozen on TB2 edges)
    edges_p_h = compute_hazard_decile_edges(tb2_scored["p_h"].to_numpy(float))
    dec_tb2, dec_stats_tb2 = evaluate_hazard_deciles(tb2_scored, edges_p_h)
    dec_tb3, dec_stats_tb3 = evaluate_hazard_deciles(tb3_scored, edges_p_h)

    # 3. Subgroup diagnostics (H0 vs H1)
    sub_tb2 = compute_hazard_subgroups(tb2_scored)
    sub_tb3 = compute_hazard_subgroups(tb3_scored)

    # 4. Paired Day-clustered bootstrap
    boot_tb2 = run_paired_day_clustered_bootstrap(tb2_scored, n_boot=n_boot, seed=seed)
    boot_tb3 = run_paired_day_clustered_bootstrap(tb3_scored, n_boot=n_boot, seed=seed)

    # 5. Cost stress
    stress_tb2 = run_cost_stress_grid_0b(tb2_scored)
    stress_tb3 = run_cost_stress_grid_0b(tb3_scored)

    # 6. Symbol breadth
    breadth_tb2_rows, breadth_tb2_counts = compute_symbol_breadth_0b(tb2_scored)
    breadth_tb3_rows, breadth_tb3_counts = compute_symbol_breadth_0b(tb3_scored)

    # 7. Pre-registered verdict on TB3
    verdict = determine_formal_verdict_0b(boot_tb3)

    # Artifact hashes
    if not pgm.SAMPLE_PATH.exists():
        raise SystemExit("STOP_PGM_NATIVE_SAMPLE_ARTIFACT_MISSING")
    if not pgm.TRANSITION_SAMPLE_PATH.exists():
        raise SystemExit("STOP_PGM_NATIVE_TRANSITION_ARTIFACT_MISSING")

    summary = {
        "EXPERIMENT_NAME": EXPERIMENT_NAME,
        "EXPERIMENT_SCOPE": EXPERIMENT_SCOPE,
        "base_sha": BASE_SHA,
        "run_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), text=True).strip(),
        "sample_artifact_sha256": hashlib.sha256(pgm.SAMPLE_PATH.read_bytes()).hexdigest(),
        "transition_artifact_sha256": hashlib.sha256(pgm.TRANSITION_SAMPLE_PATH.read_bytes()).hexdigest(),
        "max_abs_atr0_owner_error": float(max_abs_atr0_owner_error) if max_abs_atr0_owner_error is not None else 0.0,
        "n_all_obs": len(obs),
        "tb2_row_count": len(tb2_scored),
        "tb3_row_count": len(tb3_scored),
        "TB2": {
            "hazard_metrics_T2": haz_met_tb2,
            "hazard_metrics_T0_diagnostic": t0_met_tb2,
            "EV_0a": float(np.mean(tb2_scored["ret_0a"])),
            "EV_0b": float(np.mean(tb2_scored["ret_0b"])),
            "Delta_EV": float(np.mean(tb2_scored["ret_0b"]) - np.mean(tb2_scored["ret_0a"])),
            "ENEV_0a": float(np.sum(tb2_scored["ret_0a"]) / np.sum(tb2_scored["exposure_0a"])),
            "ENEV_0b": float(np.sum(tb2_scored["ret_0b"]) / np.sum(tb2_scored["exposure_0b"])),
            "Delta_ENEV": float(np.sum(tb2_scored["ret_0b"]) / np.sum(tb2_scored["exposure_0b"]) - np.sum(tb2_scored["ret_0a"]) / np.sum(tb2_scored["exposure_0a"])),
            "mean_exposure_0b": float(np.mean(tb2_scored["exposure_0b"])),
            "decile_monotonicity": dec_stats_tb2,
            "subgroups": sub_tb2,
        },
        "TB3": {
            "hazard_metrics_T2": haz_met_tb3,
            "hazard_metrics_T0_diagnostic": t0_met_tb3,
            "EV_0a": float(np.mean(tb3_scored["ret_0a"])),
            "EV_0b": float(np.mean(tb3_scored["ret_0b"])),
            "Delta_EV": float(np.mean(tb3_scored["ret_0b"]) - np.mean(tb3_scored["ret_0a"])),
            "ENEV_0a": float(np.sum(tb3_scored["ret_0a"]) / np.sum(tb3_scored["exposure_0a"])),
            "ENEV_0b": float(np.sum(tb3_scored["ret_0b"]) / np.sum(tb3_scored["exposure_0b"])),
            "Delta_ENEV": float(np.sum(tb3_scored["ret_0b"]) / np.sum(tb3_scored["exposure_0b"]) - np.sum(tb3_scored["ret_0a"]) / np.sum(tb3_scored["exposure_0a"])),
            "mean_exposure_0b": float(np.mean(tb3_scored["exposure_0b"])),
            "decile_monotonicity": dec_stats_tb3,
            "subgroups": sub_tb3,
        },
        "TB2_bootstrap": boot_tb2,
        "TB3_bootstrap": boot_tb3,
        "TB2_symbol_breadth": breadth_tb2_counts,
        "TB3_symbol_breadth": breadth_tb3_counts,
        "formal_verdict": verdict,
        "known_scope_limitations": [
            "cross-block episodes excluded from frozen PGM sample",
            "event_mask==0 censored episodes excluded from frozen PGM sample",
            "normalized friction stress only; not realistic net PnL",
            "one-step open-to-close probe; not portfolio replay",
            "exploratory mechanism validation on previously inspected TB3",
        ],
    }

    if output_dir is not None:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        (out_p / f"{PREFIX}_formal_summary.json").write_text(json.dumps(summary, indent=2))

        # Save deciles
        dec_all = [dict(block="TB2", **r) for r in dec_tb2] + [dict(block="TB3", **r) for r in dec_tb3]
        pd.DataFrame(dec_all).to_csv(out_p / f"{PREFIX}_hazard_deciles.csv", index=False)

        # Save cost stress
        stress_all = [dict(block="TB2", **r) for r in stress_tb2] + [dict(block="TB3", **r) for r in stress_tb3]
        pd.DataFrame(stress_all).to_csv(out_p / f"{PREFIX}_cost_stress.csv", index=False)

        # Save symbol breadth
        breadth_all = [dict(block="TB2", **r) for r in breadth_tb2_rows] + [dict(block="TB3", **r) for r in breadth_tb3_rows]
        pd.DataFrame(breadth_all).to_csv(out_p / f"{PREFIX}_symbol_breadth.csv", index=False)

        # Save bootstrap
        boot_rows = []
        for blk_name, b_dict in [("TB2", boot_tb2), ("TB3", boot_tb3)]:
            for k, v in b_dict.items():
                boot_rows.append({"block": blk_name, "metric": k, **v})
        pd.DataFrame(boot_rows).to_csv(out_p / f"{PREFIX}_bootstrap.csv", index=False)

    return summary


# ===========================================================================
# Execution Modes: Audit-Only, Smoke, Full-Exploratory
# ===========================================================================
def run_audit_only() -> None:
    """Execute static & structural audit without fitting models or calculating economic policies."""
    print("==================================================", flush=True)
    print("PGM-NATIVE-0B: AUDIT-ONLY EXECUTION", flush=True)
    print("==================================================", flush=True)

    # 1. Base HEAD ancestor verification
    res = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
    )
    if res.returncode != 0:
        raise SystemExit(f"STOP_PGM_NATIVE_BASE_SHA_NOT_ANCESTOR: BASE_SHA {BASE_SHA} is not an ancestor of HEAD")
    print(f"[AUDIT] BASE SHA ANCESTRY CHECK: PASS (BASE_SHA {BASE_SHA} is ancestor of HEAD)")

    # 2. Print REUSE MAP
    print_reuse_map()

    # 3. Decision Universe Audit
    print("[AUDIT] Loading observation decision universe from pgm.SAMPLE_PATH...", flush=True)
    obs = n0a.load_observed_decision_universe()
    aud_res = n0a.audit_decision_universe(obs)
    print(f"[AUDIT] Decision universe total observation rows: {aud_res['n_all_obs']}")
    print(f"[AUDIT] Hazard == 0 count: {aud_res['n_hazard0']}, Hazard == 1 count: {aud_res['n_hazard1']}")
    print(f"[AUDIT] 15 Symbols complete: {sorted(aud_res['symbols']) == EXPECTED_SYMBOLS}")

    # 4. Transition Truth & ATR0 Parity
    print("[AUDIT] Auditing transition truth and ATR0 owner parity...", flush=True)
    trans_aud = n0a.load_transition_truth_audit()
    max_atr0_err = n0a.audit_atr0_owner_parity(obs, trans_aud["cur"])
    print(f"[AUDIT] max_abs_atr0_owner_error: {max_atr0_err:.2e} (target <= 1e-12)")

    # 5. Fit Window A to verify actual Sampler Schema and T2 design columns
    print("[AUDIT] Fitting Window A samplers to verify schema & design columns...", flush=True)
    fit_A = pgm.fit_samplers_for_window(pgm.WINDOWS[0], pgm.SAMPLE_PATH, pgm.TRANSITION_SAMPLE_PATH)

    # Check terminal samplers schema
    term_keys = sorted(fit_A["term_samplers"].keys())
    print(f"[AUDIT] Actual terminal samplers keys: {term_keys}")
    if PRIMARY_TERMINAL_HEAD not in term_keys or DIAGNOSTIC_TERMINAL_HEAD not in term_keys:
        raise SystemExit(f"STOP_PGM_NATIVE0B_TERMINAL_SAMPLER_KEY_MISMATCH: {term_keys}")

    # Check transition samplers schema
    trans_keys = sorted(fit_A["trans_samplers"].keys())
    print(f"[AUDIT] Actual transition samplers keys: {trans_keys}")
    if PRIMARY_TRANSITION_HEAD not in trans_keys:
        raise SystemExit(f"STOP_PGM_NATIVE0B_TRANSITION_SAMPLER_KEY_MISMATCH: {trans_keys}")

    # Check T2 design columns for future/target leakage
    t2_sampler = fit_A["term_samplers"][PRIMARY_TERMINAL_HEAD]
    t2_design = t2_sampler.design_cols
    forbidden_target_cols = set(base.ALL_Z_COLS + base.COUNT_Z + [
        "hazard", "target_mask",
        "reward_SKIP", "reward_MARKET", "reward_LIMIT_RR3", "reward_REASSESS_RR3",
        "r_CC_ATR0", "gap_ATR0", "r_trad_OC_ATR0",
    ])
    intersection = set(t2_design).intersection(forbidden_target_cols)
    print(f"[AUDIT] T2 design future/target overlap count: {len(intersection)}")
    if intersection:
        raise SystemExit(f"STOP_PGM_NATIVE0B_T2_DESIGN_FUTURE_LEAKAGE: {intersection}")

    # 6. Check deterministic hazard probability wrapper parity
    print("[AUDIT] Verifying deterministic hazard probability wrapper parity against sample_hazard...", flush=True)
    obs_head = obs.head(50)
    audit_hazard_probability_parity(t2_sampler, obs_head)
    print("[AUDIT] Hazard probability wrapper parity check: PASS (exact parity verified)")

    print("[AUDIT] Audit-only completed successfully. Stopping before economic evaluation.", flush=True)


def run_smoke_test() -> None:
    """Execute lightweight end-to-end smoke test on <= 512 rows per block sampled from ALL rows."""
    print("==================================================", flush=True)
    print("PGM-NATIVE-0B: SMOKE TEST EXECUTION (<=512 rows/block on ALL rows)", flush=True)
    print("==================================================", flush=True)
    t0 = time.perf_counter()

    # 1. Load universe & raw bars
    obs = n0a.load_observed_decision_universe()
    trans_aud = n0a.load_transition_truth_audit()
    _, _, bars_by_sym = ex0.load_env()
    aligned_df, _ = n0a.align_raw_bars_and_returns(obs, bars_by_sym, cur_truth=trans_aud["cur"])

    # 2. Subsample <= 512 rows per block (ensuring both H0 and H1 present)
    df_valid = aligned_df[aligned_df["is_entry_valid"]].copy()
    tb2_all = df_valid[df_valid["block"] == n0a.TB2_BLOCK]
    tb3_all = df_valid[df_valid["block"] == n0a.TB3_BLOCK]

    def _sample_balanced(df_b: pd.DataFrame, n: int = 512) -> pd.DataFrame:
        h0 = df_b[df_b["hazard"] == 0]
        h1 = df_b[df_b["hazard"] == 1]
        n1 = min(len(h1), max(20, int(n * 0.10)))
        n0 = n - n1
        s0 = h0.sample(n=n0, random_state=42)
        s1 = h1.sample(n=n1, random_state=42)
        return pd.concat([s0, s1]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    tb2_sub = _sample_balanced(tb2_all, 512)
    tb3_sub = _sample_balanced(tb3_all, 512)
    print(f"[SMOKE] Subsampled TB2: {len(tb2_sub)} rows (H0={(tb2_sub['hazard']==0).sum()}, H1={(tb2_sub['hazard']==1).sum()})")
    print(f"[SMOKE] Subsampled TB3: {len(tb3_sub)} rows (H0={(tb3_sub['hazard']==0).sum()}, H1={(tb3_sub['hazard']==1).sum()})")

    # 3. Fit Window A and Window B samplers
    print("[SMOKE] Fitting Window A samplers...", flush=True)
    fit_A = pgm.fit_samplers_for_window(pgm.WINDOWS[0], pgm.SAMPLE_PATH, pgm.TRANSITION_SAMPLE_PATH)
    print("[SMOKE] Fitting Window B samplers...", flush=True)
    fit_B = pgm.fit_samplers_for_window(pgm.WINDOWS[1], pgm.SAMPLE_PATH, pgm.TRANSITION_SAMPLE_PATH)

    # 4. Score and evaluate policies
    mc_A = fit_A["trans_samplers"][PRIMARY_TRANSITION_HEAD]
    term_A_t2 = fit_A["term_samplers"][PRIMARY_TERMINAL_HEAD]
    term_A_t0 = fit_A["term_samplers"][DIAGNOSTIC_TERMINAL_HEAD]
    tb2_scored = score_and_evaluate_policies(tb2_sub, mc_A, term_A_t2, term_A_t0)

    mc_B = fit_B["trans_samplers"][PRIMARY_TRANSITION_HEAD]
    term_B_t2 = fit_B["term_samplers"][PRIMARY_TERMINAL_HEAD]
    term_B_t0 = fit_B["term_samplers"][DIAGNOSTIC_TERMINAL_HEAD]
    tb3_scored = score_and_evaluate_policies(tb3_sub, mc_B, term_B_t2, term_B_t0)

    # 5. Output smoke metrics
    haz_tb3 = compute_hazard_metrics(tb3_scored["hazard"].to_numpy(int), tb3_scored["p_h"].to_numpy(float))
    print(f"[SMOKE TB3 Hazard] AUC={haz_tb3['roc_auc']:.4f}, PR-AUC={haz_tb3['pr_auc']:.4f}, "
          f"Brier={haz_tb3['brier_score']:.4f}, Brier_skill={haz_tb3['brier_skill']:.4f}")

    ev_0a_tb3 = float(np.mean(tb3_scored["ret_0a"]))
    ev_0b_tb3 = float(np.mean(tb3_scored["ret_0b"]))
    d_ev_tb3 = ev_0b_tb3 - ev_0a_tb3
    enev_0a_tb3 = float(np.sum(tb3_scored["ret_0a"]) / np.sum(tb3_scored["exposure_0a"]))
    enev_0b_tb3 = float(np.sum(tb3_scored["ret_0b"]) / np.sum(tb3_scored["exposure_0b"]))
    d_enev_tb3 = enev_0b_tb3 - enev_0a_tb3
    print(f"[SMOKE TB3 EV] 0A EV={ev_0a_tb3:.4f}, 0B EV={ev_0b_tb3:.4f}, Delta_EV={d_ev_tb3:.4f}")
    print(f"[SMOKE TB3 ENEV] 0A ENEV={enev_0a_tb3:.4f}, 0B ENEV={enev_0b_tb3:.4f}, Delta_ENEV={d_enev_tb3:.4f}")

    # Subgroups
    sub_tb3 = compute_hazard_subgroups(tb3_scored)
    print(f"[SMOKE TB3 Subgroups] H0: mean_p_h={sub_tb3['H0']['mean_p_h']:.4f}, EV_0A={sub_tb3['H0']['EV_0a']:.4f}, EV_0B={sub_tb3['H0']['EV_0b']:.4f}")
    print(f"[SMOKE TB3 Subgroups] H1: mean_p_h={sub_tb3['H1']['mean_p_h']:.4f}, EV_0A={sub_tb3['H1']['EV_0a']:.4f}, EV_0B={sub_tb3['H1']['EV_0b']:.4f}")
    print(f"[SMOKE TB3 Subgroups] terminal_loss_attenuation={sub_tb3['terminal_loss_attenuation']:.4f}")

    elapsed = time.perf_counter() - t0
    print(f"[SMOKE COMPLETE] Successfully executed in {elapsed:.2f}s! (NO SCIENTIFIC VERDICT EMITTED)", flush=True)


def run_full_exploratory(output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Execute full exploratory evaluation across all TB2 and TB3 rows (BLOCKED THIS ROUND)."""
    if not os.environ.get("AUTHORIZE_PGM_NATIVE0B_FULL_EXPLORATORY", "").strip():
        raise SystemExit(
            "STOP_PGM_NATIVE0B_FULL_EXPLORATORY_NOT_AUTHORIZED_THIS_ROUND:\n"
            "本轮未授权运行 --full-exploratory。必须先提交代码与测试由用户完成独立审计，获得明确授权后再运行。"
        )

    # Base SHA ancestry check
    res = subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"], cwd=str(_REPO_ROOT), capture_output=True)
    if res.returncode != 0:
        raise SystemExit(f"STOP_PGM_NATIVE_BASE_SHA_NOT_ANCESTOR: BASE_SHA {BASE_SHA} is not an ancestor of HEAD")

    # Load universe & audit
    obs = n0a.load_observed_decision_universe()
    n0a.audit_decision_universe(obs)

    # Transition truth & ATR0 parity gate (must run BEFORE any model fit)
    trans_aud = n0a.load_transition_truth_audit()
    max_atr0_err = n0a.audit_atr0_owner_parity(obs, trans_aud["cur"])

    # Load raw bars
    _, _, bars_by_sym = ex0.load_env()

    # Fit Window A and Window B models
    print("[EXPLORATORY] Fitting Window A samplers...", flush=True)
    fit_A = pgm.fit_samplers_for_window(pgm.WINDOWS[0], pgm.SAMPLE_PATH, pgm.TRANSITION_SAMPLE_PATH)

    print("[EXPLORATORY] Fitting Window B samplers...", flush=True)
    fit_B = pgm.fit_samplers_for_window(pgm.WINDOWS[1], pgm.SAMPLE_PATH, pgm.TRANSITION_SAMPLE_PATH)

    if output_dir is None:
        output_dir = _REPO_ROOT / "research" / "analysis_results" / "local_liquidity_transition_v0"

    return execute_exploratory_pipeline(
        obs=obs,
        bars_by_sym=bars_by_sym,
        cur_truth=trans_aud["cur"],
        fit_A=fit_A,
        fit_B=fit_B,
        n_boot=BOOTSTRAP_N,
        seed=BOOTSTRAP_SEED,
        output_dir=output_dir,
        max_abs_atr0_owner_error=max_atr0_err,
    )


# ===========================================================================
# CLI Interface
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--audit-only", action="store_true", help="Run static and structural audit without fitting models.")
    parser.add_argument("--smoke", action="store_true", help="Run lightweight end-to-end smoke test.")
    parser.add_argument("--full-exploratory", action="store_true", help="Execute full exploratory evaluation (BLOCKED THIS ROUND).")
    args = parser.parse_args()

    if args.full_exploratory:
        run_full_exploratory()
        return

    if args.audit_only:
        run_audit_only()
        return

    if args.smoke:
        run_smoke_test()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
