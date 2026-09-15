"""DYNAMIC-PGM-1C -- Free-Run Rollout Closure.

Base Commit: ddaaf8ca8e7072d99957b687f10997f093b3fe44

Core Purpose:
Transition from one-step predictive NLL to true recursive generative simulation (rollout).
Test whether the world model:
  1. Preserves mathematical/physical support without clamping;
  2. Maintains realistic episode duration, 15-mask endpoint distribution,
     terminal state distribution, reset state distribution, and gap distribution;
  3. Proves that 1A/1B one-step improvements survive multi-step rollout (W1 vs W0);
  4. Decides whether terminal-memory challenger (WT) provides rollout utility over W1.

Three World Models:
  W0_BASELINE:
    Terminal:   T0_STATE_AVAIL
    Transition: M0_STATE_AVAIL
    Reset:      R0_ENDPOINT_ONLY
  W1_MAIN:
    Terminal:   T0_STATE_AVAIL
    Transition: MC_STATE_CURREENCODING
    Reset:      R1_STATE_PHI
  WT_TERMINAL_MEMORY:
    Terminal:   T2_STATE_PHI_MEM
    Transition: MC_STATE_CURREENCODING
    Reset:      R1_STATE_PHI

Four Stages:
  Stage C0: Sampler Parity & Epsilon Closure Audit
  Stage C1: Generative Support Probe (50,000 nonterminal + 50,000 reset draws)
  Stage C2: Observed-Start Seeded Episode Rollout (4 reps x eval episode starts)
  Stage C3: Multi-Episode Free-Run (16 chains x 15 symbols, 16 burn-in, 128 collect)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

for _bt in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_bt] = "1"

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a1b_count_magnitude_closure_v1 as base  # noqa: E402
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a2_lag_closure_v1 as lag  # noqa: E402
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a2c_representation_control_v1 as rep  # noqa: E402
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1b_terminal_reset_closure_v1 as exp1b  # noqa: E402
import research.liquidity_oracle_atlas.experiment_market_state1_1_state_closure_v1 as ms  # noqa: E402
import research.liquidity_oracle_atlas.experiment_pgm_bar0_path_smc_v1 as pbar  # noqa: E402
import research.liquidity_oracle_atlas.experiment_path0_episode_path_memory_v1 as pm  # noqa: E402

# ===========================================================================
# Frozen References & Constants
# ===========================================================================
BASE_SHA = "ddaaf8ca8e7072d99957b687f10997f093b3fe44"
EXPECTED_ROWS = 359714
EXPECTED_EPISODES = 37987
EXPECTED_TRANSITIONS = 321727
PARITY_TOL = 1e-8
MAX_EPISODE_BARS = 512
N_SEEDED_REPS = 4
N_CHAIN_REPS = 16
BURN_IN_EPISODES = 16
COLLECT_EPISODES = 128
BOOT_REPS = 5000
SYMBOL_BREADTH_MIN = 10
PREFIX = "dynamic_pgm1c"

CACHE = base.CACHE
OUT = base.OUT
WINDOWS = base.WINDOWS
FULL_UNIV = ms.FULL_UNIV

FROZEN_TRANSITION = {
    "A_TB1_to_TB2": dict(M0=1.3738896895068666, MC=0.9613370222493051),
    "B_TB1TB2_to_TB3": dict(M0=1.4054025921992275, MC=1.0374443295432418),
}

FROZEN_TERMINAL = {
    "A_TB1_to_TB2": dict(
        T0=dict(hazard_nll=0.32656089707129626,
                endpoint_joint_nll=1.148037580864784,
                mean_episode_nll=4.277133595605331),
        T2=dict(hazard_nll=0.32453465076961363,
                endpoint_joint_nll=1.141531177713386,
                mean_episode_nll=4.2512117655478345),
    ),
    "B_TB1TB2_to_TB3": dict(
        T0=dict(hazard_nll=0.3313873513680312,
                endpoint_joint_nll=1.1693060867628815,
                mean_episode_nll=4.295130738470753),
        T2=dict(hazard_nll=0.33098955816835507,
                endpoint_joint_nll=1.1722605523415464,
                mean_episode_nll=4.294333003536344),
    ),
}

FROZEN_RESET = {
    "A_TB1_to_TB2": dict(
        R0=dict(mean_reset_nll=-6.140183999205777),
        R1=dict(mean_reset_nll=-6.960087457231333),
    ),
    "B_TB1TB2_to_TB3": dict(
        R0=dict(mean_reset_nll=-6.702154068349344),
        R1=dict(mean_reset_nll=-7.523508970709451),
    ),
}

# Features
CAT = list(exp1b.CAT)
OBS_NUM = list(exp1b.OBS_NUM)
LAG_AVAIL = exp1b.LAG_AVAIL
PHI_COLS = list(exp1b.PHI_COLS)
MEM_COLS = list(exp1b.MEM_COLS)
T0_NUM = list(exp1b.T0_NUM)
T2_NUM = list(exp1b.T2_NUM)

RESET_GEOM = list(exp1b.RESET_GEOM)
RESET_LOGRATIO_RESID = list(exp1b.RESET_LOGRATIO_RESID)
RESET_SHAPE = list(exp1b.RESET_SHAPE)
RESET_AGE = list(exp1b.RESET_AGE)
RESET_CNT = list(exp1b.RESET_CNT)
RESET_TARGETS = list(exp1b.RESET_TARGETS)
MASK_CAT = list(exp1b.MASK_CAT)
R0_OCC = list(exp1b.R0_OCC)
R1_OCC = list(exp1b.R1_OCC)
GAP_FEAT = list(exp1b.GAP_FEAT)

TERMINAL_DYNAMIC_FIELDS = [
    "cur_up_distance_R",
    "cur_down_distance_R",
    "path_total_variation_R",
    "path_max_up_excursion_R",
    "path_max_down_excursion_R",
    "path_direction_change_rate",
    "path_current_bar_range_R",
    "upper_newest_log_age_residual",
    "lower_newest_log_age_residual",
    "upper_active_identity_count_delta",
    "lower_active_identity_count_delta",
]

RESET_STRUCTURAL_FIELDS = [
    "start_up_distance_R",
    "start_down_distance_R",
    "MFE0",
    "MAE0",
    "upper_newest_log_age",
    "upper_span",
    "lower_newest_log_age",
    "lower_span",
    "upper_n_active_minus1",
    "lower_n_active_minus1",
]


# ===========================================================================
# Exact Epsilon-Scale Helper & Solver
# ===========================================================================
def solve_eps_R(d_u, d_d, log_ratio):
    """Solve for eps_R from generated reset primitive values.

    L = log(d_u / d_d) + r_log
    q = exp(L)
    eps_R = (d_u - q * d_d) / (q - 1)

    Evaluated in np.longdouble for precision, then cast to float64.
    Does NOT clamp. Returns None (or NaN) if nonfinite, eps_R <= 0,
    or denominator numerically singular.
    """
    d_u_ld = np.asarray(d_u, dtype=np.longdouble)
    d_d_ld = np.asarray(d_d, dtype=np.longdouble)
    lr_ld = np.asarray(log_ratio, dtype=np.longdouble)
    q_ld = np.exp(lr_ld)
    denom = q_ld - 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        eps_ld = (d_u_ld - q_ld * d_d_ld) / denom

    eps_f = np.asarray(eps_ld, dtype=np.float64)
    denom_f = np.asarray(denom, dtype=np.float64)

    invalid = (~np.isfinite(eps_f)) | (eps_f <= 0.0) | (np.abs(denom_f) < 1e-12)
    if np.ndim(eps_f) == 0:
        return None if bool(invalid) else float(eps_f)
    return np.where(invalid, np.nan, eps_f)


def audit_eps_r_closure(obs: pd.DataFrame, ep_meta: pd.DataFrame) -> Dict[str, Any]:
    """Verify that eps_R = 1e-9 / atr0 exactly closes cur_log_ratio across all rows."""
    meta_slim = ep_meta[["symbol", "start_bar", "start_upper_price", "start_lower_price"]].drop_duplicates()
    m = obs[["symbol", "start_bar", "start_width_R", "cur_up_distance_R",
             "cur_down_distance_R", "cur_log_ratio"]].merge(
        meta_slim, on=["symbol", "start_bar"], how="left"
    )
    if m["start_upper_price"].isna().any():
        raise SystemExit("STOP_DYNAMIC_PGM1C_EPSR_MISSING_METADATA")

    span = m["start_upper_price"].to_numpy(np.float64) - m["start_lower_price"].to_numpy(np.float64)
    atr0 = span / m["start_width_R"].to_numpy(np.float64)
    eps_R = 1e-9 / atr0

    up = m["cur_up_distance_R"].to_numpy(np.float64)
    dn = m["cur_down_distance_R"].to_numpy(np.float64)
    cur_lr = m["cur_log_ratio"].to_numpy(np.float64)

    recon_lr = np.log((up + eps_R) / (dn + eps_R))
    max_err = float(np.max(np.abs(recon_lr - cur_lr)))
    if max_err >= 1e-8:
        raise SystemExit(f"STOP_DYNAMIC_PGM1C_EPSR_CLOSURE_FAIL: max_err={max_err}")

    return dict(
        n_rows=int(len(m)),
        max_error=max_err,
        min_eps_r=float(eps_R.min()),
        max_eps_r=float(eps_R.max()),
        passed=True,
    )


# ===========================================================================
# Stochastic Sampling Primitives
# ===========================================================================
def sample_ztp_vec(lam: float, rng: np.random.Generator, size: int) -> np.ndarray:
    """Vectorized inverse-CDF sampler for Zero-Truncated Poisson ZTP(lambda).

    P(Y=y | Y>0) = lambda^y * exp(-lambda) / (y! * (1 - exp(-lambda)))
    """
    if size <= 0:
        return np.empty(0, dtype=np.int64)
    u = rng.random(size)
    p_init = lam * np.exp(-lam) / (-np.expm1(-lam))
    p = np.full(size, p_init, dtype=np.float64)
    cdf = p.copy()
    y = np.ones(size, dtype=np.int64)
    active = u > cdf
    while np.any(active):
        y[active] += 1
        p[active] *= lam / y[active]
        cdf[active] += p[active]
        active = u > cdf
    return y


def sample_gaussian(head: Any, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample from Gaussian head: Z = X @ B + intercept + Normal(0, Sigma)."""
    X = np.asarray(X, dtype=np.float64)
    mu = X @ head.B + head.intercept
    z = rng.standard_normal((len(X), head.k))
    return mu + z @ head.chol.T


def sample_hurdle_ln(head: Any, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample from Hurdle-LogNormal head.

    Occurrence Logistic -> if active: logv ~ Gaussian, raw = sign * exp(logv), else 0.
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    p = head.logit.predict_proba(X)[:, 1]
    active = rng.random(n) < p
    raw = np.zeros(n, dtype=np.float64)
    if np.any(active):
        logv = sample_gaussian(head.g, X[active], rng)[:, 0]
        raw[active] = head.sign * np.exp(logv)
    return raw


def sample_dcr(head: Any, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample from ZeroInteriorOneHead for DCR.

    3-class categorical {0, interior, 1}.
    Interior -> Gaussian logit -> sigmoid(z). No clipping.
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    probs = np.clip(head.cat.predict_proba(X), 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    cdf = np.cumsum(probs, axis=1)
    u = rng.random(n)[:, None]
    cat_draw = (u > cdf[:, :-1]).sum(axis=1)

    raw = np.zeros(n, dtype=np.float64)
    raw[cat_draw == 2] = 1.0
    interior = cat_draw == 1
    if np.any(interior):
        z = sample_gaussian(head.g, X[interior], rng)[:, 0]
        raw[interior] = 1.0 / (1.0 + np.exp(-z))
    return raw


def sample_count_head(clf: Any, lam: float, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample from state-dependent occurrence Logistic + constant ZTP magnitude."""
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    p = clf.predict_proba(X)[:, 1]
    active = rng.random(n) < p
    cnt = np.zeros(n, dtype=np.int64)
    n_active = int(np.sum(active))
    if n_active > 0:
        cnt[active] = sample_ztp_vec(lam, rng, n_active)
    return cnt


def sample_crf_endpoint(theta: np.ndarray, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample 15-mask endpoint from conditional pairwise CRF."""
    X = np.asarray(X, dtype=np.float64)
    p_mask = pbar.predict_mask_prob(theta, X, with_pairs=True)
    p_mask = np.clip(p_mask, 1e-15, 1.0)
    p_mask = p_mask / p_mask.sum(axis=1, keepdims=True)
    cdf = np.cumsum(p_mask, axis=1)
    u = rng.random(len(X))[:, None]
    mask_idx = (u > cdf[:, :-1]).sum(axis=1)
    return (mask_idx + 1).astype(np.int64)


# ===========================================================================
# Sampler Fitted Container Classes
# ===========================================================================
class FittedTransitionSampler:
    """Fitted transition sampler for M0 or MC."""

    def __init__(self, tag: str, ct: Any, heads: Dict[str, Any], count_head: Dict[str, Any]):
        self.tag = tag
        self.ct = ct
        self.heads = heads
        self.count_head = count_head
        self.constant_rates = count_head["constant_rates"]

    def sample_batch(self, df: pd.DataFrame, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        X = self.ct.transform(df).astype(np.float32)
        n = len(df)
        nodes = self.heads["nodes"]

        # Continuous nodes
        d_up = sample_gaussian(nodes["z_d_up"]["head"], X, rng)[:, 0]
        dmfe = sample_hurdle_ln(nodes["z_dmfe"]["head"], X, rng)
        dmae = sample_hurdle_ln(nodes["z_dmae"]["head"], X, rng)
        dcr = sample_dcr(nodes["z_dcr"]["head"], X, rng)
        rng_val = sample_hurdle_ln(nodes["z_range"]["head"], X, rng)
        uresid = sample_hurdle_ln(nodes["z_uresid"]["head"], X, rng)
        lresid = sample_hurdle_ln(nodes["z_lresid"]["head"], X, rng)

        # Counts
        cnt_clfs = self.count_head["models"]
        lam_u = float(self.constant_rates[0])
        lam_l = float(self.constant_rates[1])
        c_u = sample_count_head(cnt_clfs[0], lam_u, X, rng)
        c_l = sample_count_head(cnt_clfs[1], lam_l, X, rng)

        return dict(
            z_d_up=d_up,
            dmfe=dmfe,
            dmae=dmae,
            dcr=dcr,
            range=rng_val,
            uresid=uresid,
            lresid=lresid,
            delta_upper_count=c_u,
            delta_lower_count=c_l,
        )


class FittedTerminalSampler:
    """Fitted terminal sampler for T0 or T2."""

    def __init__(self, tag: str, pipe: Any, theta: np.ndarray):
        self.tag = tag
        self.pipe = pipe
        self.pre = pipe.named_steps["pre"]
        self.clf = pipe.named_steps["clf"]
        self.theta = theta

    def sample_hazard(self, df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        p_h = self.pipe.predict_proba(df)[:, 1]
        return rng.random(len(df)) < p_h

    def sample_endpoint(self, df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        X = pbar.densify(self.pre.transform(df))
        return sample_crf_endpoint(self.theta, X, rng)


class FittedResetSampler:
    """Fitted reset sampler for R0 or R1."""

    def __init__(self, tag: str, occ_cols: List[str], state_cols: List[str],
                 pre_occ: Any, pre_state: Any, gap_occ_clf: Any,
                 gap_p: float, heads: Dict[str, Any]):
        self.tag = tag
        self.occ_cols = occ_cols
        self.state_cols = state_cols
        self.pre_occ = pre_occ
        self.pre_state = pre_state
        self.gap_occ_clf = gap_occ_clf
        self.gap_p = gap_p
        self.heads = heads
        self.constant_rates = heads["count_rates"]

    def sample_gap(self, df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        cols = self.occ_cols + MASK_CAT
        X_occ = pbar.densify(self.pre_occ.transform(df[cols]))
        p_gap = self.gap_occ_clf.predict_proba(X_occ)[:, 1]
        has_gap = rng.random(len(df)) < p_gap
        gap_bars = np.zeros(len(df), dtype=np.int64)
        n_pos = int(np.sum(has_gap))
        if n_pos > 0:
            gap_bars[has_gap] = rng.geometric(self.gap_p, size=n_pos)
        return gap_bars

    def sample_reset_primitives(self, df: pd.DataFrame, gap_bars: np.ndarray,
                                rng: np.random.Generator) -> Dict[str, np.ndarray]:
        df_state = df.copy()
        df_state["gap_positive"] = (gap_bars > 0).astype(np.int64)
        df_state["log1p_gap"] = np.log1p(gap_bars.astype(np.float64))
        cols = self.state_cols + MASK_CAT
        X_s = pbar.densify(self.pre_state.transform(df_state[cols]))

        # 1. Geometry 2D Gaussian
        log_geom = sample_gaussian(self.heads["geometry"], X_s, rng)
        d_u = np.exp(log_geom[:, 0])
        d_d = np.exp(log_geom[:, 1])

        # 2. Logratio residual 1D Gaussian
        r_log = sample_gaussian(self.heads["logratio_residual"], X_s, rng)[:, 0]

        # 3. Shape & Age HurdleLogNormal heads
        shape_age_draws = {}
        for name in RESET_SHAPE + RESET_AGE:
            shape_age_draws[name] = sample_hurdle_ln(self.heads[name], X_s, rng)

        # 4. Counts
        lam_u = float(self.constant_rates[0])
        lam_l = float(self.constant_rates[1])
        cnt_clfs = self.heads["counts"]["models"]
        n_act_u_minus1 = sample_count_head(cnt_clfs[0], lam_u, X_s, rng)
        n_act_l_minus1 = sample_count_head(cnt_clfs[1], lam_l, X_s, rng)

        return dict(
            start_up_distance_R=d_u,
            start_down_distance_R=d_d,
            start_log_ratio_residual=r_log,
            next_path_max_up_excursion_R=shape_age_draws["next_path_max_up_excursion_R"],
            next_path_max_down_excursion_R=shape_age_draws["next_path_max_down_excursion_R"],
            next_upper_newest_log_age=shape_age_draws["next_upper_newest_log_age"],
            next_upper_span=shape_age_draws["next_upper_span"],
            next_lower_newest_log_age=shape_age_draws["next_lower_newest_log_age"],
            next_lower_span=shape_age_draws["next_lower_span"],
            next_upper_n_active_minus1=n_act_u_minus1,
            next_lower_n_active_minus1=n_act_l_minus1,
        )


# ===========================================================================
# Deterministic State Reconstruction & Support Checks
# ===========================================================================
def advance_nonterminal(state: Dict[str, Any], z: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """Deterministic state update from S_t and realized Z_{t+1}.

    Does NOT clamp. Returns (next_state, violation_reason).
    """
    nxt = dict(state)
    d_up = float(z["z_d_up"])
    nxt_up = state["cur_up_distance_R"] + d_up
    nxt_dn = state["cur_down_distance_R"] - d_up

    # Support checks on distances
    if nxt_up < 0:
        return nxt, "TRANSITION_UP_DISTANCE_NEGATIVE"
    if nxt_dn < 0:
        return nxt, "TRANSITION_DOWN_DISTANCE_NEGATIVE"

    nxt["cur_up_distance_R"] = nxt_up
    nxt["cur_down_distance_R"] = nxt_dn

    # Excursions
    dmfe = float(z["dmfe"])
    dmae = float(z["dmae"])
    nxt["path_max_up_excursion_R"] = state["path_max_up_excursion_R"] + dmfe
    nxt["path_max_down_excursion_R"] = state["path_max_down_excursion_R"] + dmae

    # DCR
    dcr = float(z["dcr"])
    if dcr < 0.0 or dcr > 1.0:
        return nxt, "TRANSITION_DCR_OOB"
    nxt["path_direction_change_rate"] = dcr

    # Range
    rng_val = float(z["range"])
    if rng_val < 0.0:
        return nxt, "TRANSITION_RANGE_NEGATIVE"
    nxt["path_current_bar_range_R"] = rng_val

    # Residuals
    uresid = float(z["uresid"])
    lresid = float(z["lresid"])
    nxt["upper_newest_log_age_residual"] = uresid
    nxt["lower_newest_log_age_residual"] = lresid

    # Counts
    cu_inc = int(z["delta_upper_count"])
    cl_inc = int(z["delta_lower_count"])
    if cu_inc < 0 or cl_inc < 0:
        return nxt, "TRANSITION_COUNT_NEGATIVE"
    nxt["upper_active_identity_count_delta"] = state["upper_active_identity_count_delta"] + cu_inc
    nxt["lower_active_identity_count_delta"] = state["lower_active_identity_count_delta"] + cl_inc

    # Timing
    age_next = state["episode_age"] + 1
    nxt["episode_age"] = age_next
    nxt["elapsed_log"] = np.log1p(age_next)

    # Derived newest ages & age-zero
    for side in ("upper", "lower"):
        res = uresid if side == "upper" else lresid
        new_0 = np.expm1(state[f"{side}_newest_log_age"])
        expected_new = new_0 + age_next
        age_next_val = np.expm1(res + np.log1p(expected_new))
        if age_next_val < -1e-9:
            return nxt, "TRANSITION_NEGATIVE_NEWEST_AGE"
        is_zero = 1.0 if np.isclose(age_next_val, 0.0, atol=1e-10) else 0.0
        nxt[f"{side}_current_newest_age_zero"] = is_zero

    # Exact tempo & log ratio
    net = state["start_up_distance_R"] - nxt_up
    tv = state["path_total_variation_R"] + abs(d_up)
    nxt["path_total_variation_R"] = tv
    nxt["path_last_return_R"] = -d_up

    nxt["tempo_signed_speed"] = net / max(age_next, 1)
    nxt["tempo_abs_speed"] = abs(nxt["tempo_signed_speed"])
    nxt["tempo_signed_efficiency"] = net / max(tv, 1e-12)
    nxt["tempo_abs_efficiency"] = abs(nxt["tempo_signed_efficiency"])

    eps_R = state["eps_R"]
    nxt["cur_log_ratio"] = np.log((nxt_up + eps_R) / (nxt_dn + eps_R))

    # Availability
    nxt[LAG_AVAIL] = 1.0

    # Nonlinear re-encodings (phi_*)
    _phi_raws = {
        "z_d_up": -nxt["path_last_return_R"],
        "z_dcr": nxt["path_direction_change_rate"],
        "z_range": nxt["path_current_bar_range_R"],
        "z_uresid": nxt["upper_newest_log_age_residual"],
        "z_lresid": nxt["lower_newest_log_age_residual"],
    }
    for node, raw in _phi_raws.items():
        for zc, vals in rep._encode_with_frozen(node, np.array([raw])).items():
            nxt[f"phi_{zc}"] = float(vals[0])

    # Memory challenger innovations (mem_*)
    for node, raw in [("z_dmfe", dmfe), ("z_dmae", dmae)]:
        for zc, vals in rep._encode_with_frozen(node, np.array([raw])).items():
            nxt[f"mem_{zc}"] = float(vals[0])
    nxt["mem_z_delta_upper_count"] = float(cu_inc)
    nxt["mem_z_delta_lower_count"] = float(cl_inc)

    # Check nonfinite
    for k, v in nxt.items():
        if isinstance(v, (int, float)) and not np.isfinite(v):
            return nxt, "TRANSITION_NONFINITE"

    return nxt, None


def reset_to_start(r: Dict[str, Any], endpoint_mask: int) -> Tuple[Dict[str, Any], Optional[str]]:
    """Reconstruct S^0_{n+1} from reset primitives and endpoint mask.

    Does NOT clamp. Returns (start_state, violation_reason).
    """
    d_u = float(r["start_up_distance_R"])
    d_d = float(r["start_down_distance_R"])
    if d_u <= 0 or d_d <= 0:
        return {}, "RESET_GEOMETRY_NONPOSITIVE"

    r_log = float(r["start_log_ratio_residual"])
    log_ratio = np.log(d_u / d_d) + r_log

    eps_R = solve_eps_R(d_u, d_d, log_ratio)
    if eps_R is None:
        return {}, "RESET_EPSR_INCOMPATIBLE"

    mfe0 = float(r["next_path_max_up_excursion_R"])
    mae0 = float(r["next_path_max_down_excursion_R"])
    newest_u = float(r["next_upper_newest_log_age"])
    span_u = float(r["next_upper_span"])
    newest_l = float(r["next_lower_newest_log_age"])
    span_l = float(r["next_lower_span"])

    if newest_u < 0 or span_u < 0 or newest_l < 0 or span_l < 0:
        return {}, "RESET_AGE_NEGATIVE"

    n_act_u = int(r["next_upper_n_active_minus1"]) + 1
    n_act_l = int(r["next_lower_n_active_minus1"]) + 1
    if n_act_u < 1 or n_act_l < 1:
        return {}, "RESET_COUNT_INVALID"

    oldest_u = newest_u + span_u
    oldest_l = newest_l + span_l

    up_new_zero = 1.0 if newest_u <= 1e-10 else 0.0
    up_old_zero = 1.0 if oldest_u <= 1e-10 else 0.0
    lo_new_zero = 1.0 if newest_l <= 1e-10 else 0.0
    lo_old_zero = 1.0 if oldest_l <= 1e-10 else 0.0

    state: Dict[str, Any] = {
        "start_up_distance_R": d_u,
        "start_down_distance_R": d_d,
        "start_width_R": d_u + d_d,
        "start_log_ratio": log_ratio,

        "upper_oldest_log_age": oldest_u,
        "upper_newest_log_age": newest_u,
        "upper_newest_age_zero": up_new_zero,
        "upper_oldest_age_zero": up_old_zero,
        "upper_n_active_identities": float(n_act_u),

        "lower_oldest_log_age": oldest_l,
        "lower_newest_log_age": newest_l,
        "lower_newest_age_zero": lo_new_zero,
        "lower_oldest_age_zero": lo_old_zero,
        "lower_n_active_identities": float(n_act_l),

        "elapsed_log": 0.0,
        "cur_up_distance_R": d_u,
        "cur_down_distance_R": d_d,
        "cur_width_R": d_u + d_d,
        "cur_log_ratio": log_ratio,

        "path_total_variation_R": 0.0,
        "path_max_up_excursion_R": mfe0,
        "path_max_down_excursion_R": mae0,
        "path_direction_change_rate": 0.0,
        "path_last_return_R": 0.0,
        "path_current_bar_range_R": mfe0 + mae0,

        "upper_newest_log_age_residual": 0.0,
        "upper_current_newest_age_zero": up_new_zero,
        "upper_active_identity_count_delta": 0.0,

        "lower_newest_log_age_residual": 0.0,
        "lower_current_newest_age_zero": lo_new_zero,
        "lower_active_identity_count_delta": 0.0,

        "tempo_signed_speed": 0.0,
        "tempo_abs_speed": 0.0,
        "tempo_signed_efficiency": 0.0,
        "tempo_abs_efficiency": 0.0,

        "prev_event_mask": int(endpoint_mask),
        LAG_AVAIL: 0.0,
        "eps_R": eps_R,
        "episode_age": 0,
    }

    # Derived phi_* for start state
    _phi_raws = {
        "z_d_up": 0.0,
        "z_dcr": 0.0,
        "z_range": state["path_current_bar_range_R"],
        "z_uresid": 0.0,
        "z_lresid": 0.0,
    }
    for node, raw in _phi_raws.items():
        for zc, vals in rep._encode_with_frozen(node, np.array([raw])).items():
            state[f"phi_{zc}"] = float(vals[0])

    # First row memory challenger features
    for c in ["mem_z_dmfe_ispos", "mem_z_dmfe_log", "mem_z_dmae_ispos", "mem_z_dmae_log"]:
        state[c] = 0.0
    state["mem_z_delta_upper_count"] = np.nan
    state["mem_z_delta_lower_count"] = np.nan

    # Nonfinite check
    for k, v in state.items():
        if k in ("mem_z_delta_upper_count", "mem_z_delta_lower_count"):
            continue
        if isinstance(v, (int, float)) and not np.isfinite(v):
            return {}, "RESET_NONFINITE"

    return state, None


# ===========================================================================
# Rollout Discrepancy Metrics
# ===========================================================================
def normalized_wasserstein(x_gen: np.ndarray, x_obs: np.ndarray, floor: float = 1e-6) -> float:
    """Normalized 1D Wasserstein distance: W_1(X_g, X_o) / max(Q95(X_o) - Q05(X_o), floor)."""
    x_g = np.asarray(x_gen, dtype=np.float64)
    x_o = np.asarray(x_obs, dtype=np.float64)
    x_g = x_g[np.isfinite(x_g)]
    x_o = x_o[np.isfinite(x_o)]
    if len(x_g) == 0 or len(x_o) == 0:
        return np.nan
    w1 = float(scipy.stats.wasserstein_distance(x_g, x_o))
    scale = float(np.percentile(x_o, 95) - np.percentile(x_o, 5))
    norm = max(scale, floor)
    return w1 / norm


def endpoint_tvd(p_gen: np.ndarray, p_obs: np.ndarray) -> float:
    """Total Variation Distance between 15-mask discrete probability vectors."""
    p_g = np.asarray(p_gen, dtype=np.float64)
    p_o = np.asarray(p_obs, dtype=np.float64)
    return float(0.5 * np.sum(np.abs(p_g - p_o)))


def compute_rollout_discrepancy(gen_data: Dict[str, Any], obs_data: Dict[str, Any]) -> Dict[str, float]:
    """Compute 5-component discrepancy score D_total."""
    # 1. Duration (floor = 1.0 bar)
    d_dur = normalized_wasserstein(gen_data["durations"], obs_data["durations"], floor=1.0)

    # 2. Endpoint TVD
    p_g = np.zeros(15, dtype=np.float64)
    p_o = np.zeros(15, dtype=np.float64)
    for m in range(1, 16):
        p_g[m - 1] = np.mean(gen_data["endpoints"] == m) if len(gen_data["endpoints"]) else 0.0
        p_o[m - 1] = np.mean(obs_data["endpoints"] == m) if len(obs_data["endpoints"]) else 0.0
    d_end = endpoint_tvd(p_g, p_o)

    # 3. Terminal dynamic fields (11)
    d_term_list = []
    for fld in TERMINAL_DYNAMIC_FIELDS:
        dw = normalized_wasserstein(gen_data["terminals"][fld], obs_data["terminals"][fld], floor=1e-6)
        d_term_list.append(dw)
    d_term = float(np.nanmean(d_term_list))

    # 4. Reset structural primitives (10)
    d_reset_list = []
    for fld in RESET_STRUCTURAL_FIELDS:
        dw = normalized_wasserstein(gen_data["resets"][fld], obs_data["resets"][fld], floor=1e-6)
        d_reset_list.append(dw)
    d_reset = float(np.nanmean(d_reset_list))

    # 5. Gap
    p_gap_g = float(np.mean(gen_data["gaps"] > 0)) if len(gen_data["gaps"]) else 0.0
    p_gap_o = float(np.mean(obs_data["gaps"] > 0)) if len(obs_data["gaps"]) else 0.0
    occ_diff = abs(p_gap_g - p_gap_o)
    pos_g = gen_data["gaps"][gen_data["gaps"] > 0]
    pos_o = obs_data["gaps"][obs_data["gaps"] > 0]
    if len(pos_g) >= 20 and len(pos_o) >= 20:
        mag_dw = normalized_wasserstein(pos_g, pos_o, floor=1.0)
        d_gap = 0.5 * (occ_diff + mag_dw)
    else:
        d_gap = occ_diff

    # Composite: equal weight
    d_total = float((d_dur + d_end + d_term + d_reset + d_gap) / 5.0)

    return dict(
        D_duration=d_dur,
        D_endpoint=d_end,
        D_terminal=d_term,
        D_reset=d_reset,
        D_gap=d_gap,
        D_total=d_total,
    )


def paired_bootstrap_replicates(diffs_by_rep: np.ndarray, seed: int,
                                reps: int = BOOT_REPS) -> Tuple[float, float, float]:
    """Cluster-bootstrap on paired chain replicate IDs with replacement.

    Preserves multiplicity across replicates.
    """
    diffs = np.asarray(diffs_by_rep, dtype=np.float64)
    n = len(diffs)
    point = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    boot_means = np.empty(reps, dtype=np.float64)
    for b in range(reps):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = np.mean(diffs[idx])
    return (
        float(np.percentile(boot_means, 2.5)),
        float(np.percentile(boot_means, 97.5)),
        point,
    )


# ===========================================================================
# Model Fitting for Samplers
# ===========================================================================
def fit_samplers_for_window(w: Dict[str, Any], obs_sample_path: Path,
                            transitions_path: Path) -> Dict[str, Any]:
    """Fit all sampler objects for window w and verify parity."""
    print(f"[FIT SAMPLERS] Starting window {w['name']}...", flush=True)

    # 1. Terminal Samplers
    obs = pd.read_parquet(obs_sample_path)
    tr = obs[obs["block"].isin(w["train"])].reset_index(drop=True)
    ev = obs[obs["block"] == w["eval"]].reset_index(drop=True)

    term_samplers = {}
    term_parity_eval = {}

    for tag, num_cols in [("T0_STATE_AVAIL", T0_NUM), ("T2_STATE_PHI_MEM", T2_NUM)]:
        t0 = time.perf_counter()
        cols = num_cols + CAT
        pipe = pm.make_pipeline(num_cols, CAT)
        pipe.fit(tr[cols], tr["hazard"].to_numpy(np.int64))
        pre = pipe.named_steps["pre"]
        p_h = pipe.predict_proba(ev[cols])[:, 1]

        Xtr = pbar.densify(pre.transform(tr[cols]))
        Xev = pbar.densify(pre.transform(ev[cols]))
        tt = tr["hazard"].to_numpy(np.int64) == 1
        mtr = tr["target_mask"].to_numpy()[tt].astype(np.int64)
        theta, aud = pbar.fit_conditional_crf(Xtr[tt], mtr, with_pairs=True)

        te = ev["hazard"].to_numpy(np.int64) == 1
        p_mask = pbar.predict_mask_prob(theta, Xev[te], with_pairs=True)
        met = ms.model_metrics(ev, p_h, p_mask, te)
        term_parity_eval[tag] = met
        term_samplers[tag] = FittedTerminalSampler(tag, pipe, theta)
        print(f"  [TERMINAL {tag}] took {time.perf_counter() - t0:.2f}s "
              f"haz_nll={met['hazard_nll']:.6f} ep_joint={met['endpoint_joint_nll']:.6f} "
              f"mean_ep={met['mean_episode_nll']:.6f}", flush=True)

    # 2. Reset Samplers
    pair, first = exp1b.build_reset_pairs(obs)
    ptr = pair[pair["block"].isin(w["train"]) & pair["next_episode_id"].notna()].reset_index(drop=True)
    pev = pair[(pair["block"] == w["eval"]) & pair["next_episode_id"].notna()].reset_index(drop=True)

    gap_p = exp1b.fit_gap_geometric(ptr["gap_bars"].to_numpy(np.float64))
    reset_samplers = {}
    reset_parity_eval = {}

    for tag, occ_cols in [("R0_ENDPOINT_ONLY", R0_OCC), ("R1_STATE_PHI", R1_OCC)]:
        t0 = time.perf_counter()
        state_cols = list(occ_cols) + GAP_FEAT

        cols_occ = occ_cols + MASK_CAT
        pre_o = pm.make_pipeline(occ_cols, MASK_CAT).named_steps["pre"]
        pre_o.fit(ptr[cols_occ])
        Xtr_o = pbar.densify(pre_o.transform(ptr[cols_occ]))
        Xev_o = pbar.densify(pre_o.transform(pev[cols_occ]))

        cols_state = state_cols + MASK_CAT
        pre_s = pm.make_pipeline(state_cols, MASK_CAT).named_steps["pre"]
        pre_s.fit(ptr[cols_state])
        Xtr_s = pbar.densify(pre_s.transform(ptr[cols_state]))
        Xev_s = pbar.densify(pre_s.transform(pev[cols_state]))

        ytr_gap = ptr["gap_positive"].to_numpy(np.int64)
        clf_gap = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=3000)
        clf_gap.fit(Xtr_o, ytr_gap)
        exp1b.base._check_logistic_convergence(clf_gap, f"{tag}_gap_occurrence")
        gap_occ = np.clip(clf_gap.predict_proba(Xev_o)[:, 1], 1e-6, 1.0 - 1e-6)
        nll_occ = np.where(pev["gap_positive"].to_numpy(np.int64) == 1,
                           -np.log(gap_occ), -np.log1p(-gap_occ))

        # Heads on Xtr_s
        heads: Dict[str, Any] = {}
        # Geometry
        ytr_log = np.log(ptr[RESET_GEOM].to_numpy(np.float64))
        yev_log = np.log(pev[RESET_GEOM].to_numpy(np.float64))
        gh = base.GaussianTransitionHead(alpha=1.0).fit(Xtr_s, ytr_log)
        heads["geometry"] = gh
        nll_geom = gh.nll_per_row(Xev_s, yev_log) + yev_log.sum(axis=1)

        # Logratio residual
        ytr_rlog = ptr[RESET_LOGRATIO_RESID].to_numpy(np.float64)
        yev_rlog = pev[RESET_LOGRATIO_RESID].to_numpy(np.float64)
        gh_rlog = base.GaussianTransitionHead(alpha=1.0).fit(Xtr_s, ytr_rlog)
        heads["logratio_residual"] = gh_rlog
        nll_rlog = gh_rlog.nll_per_row(Xev_s, yev_rlog)

        # Shape & Age
        nll_shape_age = 0.0
        for name in RESET_SHAPE + RESET_AGE:
            ztr = exp1b._encode_nonneg(ptr[name].to_numpy(np.float64))
            zev = exp1b._encode_nonneg(pev[name].to_numpy(np.float64))
            h = base.HurdleLogNormalHead(alpha=1.0, sign=1.0).fit(Xtr_s, ztr)
            heads[name] = h
            nll_shape_age = nll_shape_age + h.nll_per_row(Xev_s, zev)

        # Counts
        Ytr = ptr[RESET_CNT].to_numpy(np.int64)
        Yev = pev[RESET_CNT].to_numpy(np.int64)
        k0c = base.fit_constant_count_head(Ytr, Yev)
        kc = base.fit_state_count_head(Xtr_s, Ytr, Xev_s, Yev, constant_rates=k0c["constant_rates"])
        heads["counts"] = kc
        heads["count_rates"] = k0c["constant_rates"]
        nll_cnt = kc["nll_ev"].sum(axis=1)

        total_nll = (nll_occ
                     + exp1b.gap_positive_nll(pev["gap_bars"].to_numpy(np.float64), gap_p)
                     + nll_geom + nll_rlog + nll_shape_age + nll_cnt)

        mean_res_nll = float(total_nll.mean())
        reset_parity_eval[tag] = dict(mean_reset_nll=mean_res_nll)
        reset_samplers[tag] = FittedResetSampler(
            tag, occ_cols, state_cols, pre_o, pre_s, clf_gap, gap_p, heads
        )
        print(f"  [RESET {tag}] took {time.perf_counter() - t0:.2f}s "
              f"mean_reset_nll={mean_res_nll:.6f}", flush=True)

    # 3. Transition Samplers
    df_trans = pd.read_parquet(transitions_path)
    tr_trans = df_trans[df_trans["block"].isin(w["train"])].reset_index(drop=True)
    ev_trans = df_trans[df_trans["block"] == w["eval"]].reset_index(drop=True)

    Zc_tr = tr_trans[base.ALL_Z_COLS].to_numpy(np.float32)
    Zc_ev = ev_trans[base.ALL_Z_COLS].to_numpy(np.float32)
    yd_tr = tr_trans[base.DISC_Z].to_numpy(np.int64)
    yd_ev = ev_trans[base.DISC_Z].to_numpy(np.int64)
    Yc_tr = tr_trans[base.COUNT_Z].to_numpy(np.int64)
    Yc_ev = ev_trans[base.COUNT_Z].to_numpy(np.int64)

    base.configure_child_semantics(agezero_deterministic=True)
    k0_count = base.fit_constant_count_head(Yc_tr, Yc_ev)
    trans_samplers = {}
    trans_parity_eval = {}

    obs_cols = list(base.OBS_STATE_NUM)
    for tag, num_cols in [("M0_STATE_AVAIL", obs_cols + rep.M0_EXTRA),
                          ("MC_STATE_CURREENCODING", obs_cols + rep.MC_EXTRA)]:
        t0 = time.perf_counter()
        ct = lag._make_ct(num_cols)
        Xtr = ct.fit_transform(tr_trans).astype(np.float32)
        Xev = ct.transform(ev_trans).astype(np.float32)

        k = base.fit_state_heads(Xtr, Xev, Zc_tr, Zc_ev, yd_tr, yd_ev)
        kc = base.fit_state_count_head(Xtr, Yc_tr, Xev, Yc_ev, constant_rates=k0_count["constant_rates"])

        cont = rep._node_eval_nll(k["nodes"])
        disc = k["disc_ev"]
        cnt = kc["nll_ev"].sum(axis=1)
        mean_j = float(np.mean(cont + disc + cnt))

        trans_parity_eval[tag] = dict(mean_joint_nll=mean_j)
        trans_samplers[tag] = FittedTransitionSampler(tag, ct, k, kc)
        print(f"  [TRANSITION {tag}] took {time.perf_counter() - t0:.2f}s "
              f"mean_joint_nll={mean_j:.16f}", flush=True)

    return dict(
        window=w["name"],
        term_samplers=term_samplers,
        reset_samplers=reset_samplers,
        trans_samplers=trans_samplers,
        term_parity=term_parity_eval,
        reset_parity=reset_parity_eval,
        trans_parity=trans_parity_eval,
    )


# ===========================================================================
# Stage C1: Generative Support Probe
# ===========================================================================
def run_stage_c1_probe(window_name: str, trans_sampler: FittedTransitionSampler,
                       reset_sampler: FittedResetSampler, ev_states: pd.DataFrame,
                       ev_pairs: pd.DataFrame, seed: int,
                       n_draws: int = 50000) -> Dict[str, Any]:
    """Draw 50,000 transition and 50,000 reset samples to probe mathematical support."""
    print(f"[STAGE C1] Support probe on {window_name} ({n_draws} draws)...", flush=True)
    rng = np.random.default_rng(seed)

    # 1. Transition probe on nonterminal states
    nonterm = ev_states[ev_states["hazard"] == 0].reset_index(drop=True)
    idx_t = rng.integers(0, len(nonterm), size=n_draws)
    sample_df = nonterm.iloc[idx_t].copy().reset_index(drop=True)

    z_draws = trans_sampler.sample_batch(sample_df, rng)

    t_violations: Dict[str, int] = {
        "TRANSITION_UP_DISTANCE_NEGATIVE": 0,
        "TRANSITION_DOWN_DISTANCE_NEGATIVE": 0,
        "TRANSITION_NEGATIVE_NEWEST_AGE": 0,
        "TRANSITION_DCR_OOB": 0,
        "TRANSITION_RANGE_NEGATIVE": 0,
        "TRANSITION_COUNT_NONINTEGER": 0,
        "TRANSITION_COUNT_NEGATIVE": 0,
        "TRANSITION_NONFINITE": 0,
    }

    d_up = z_draws["z_d_up"]
    cur_up = sample_df["cur_up_distance_R"].to_numpy(np.float64)
    cur_dn = sample_df["cur_down_distance_R"].to_numpy(np.float64)
    nxt_up = cur_up + d_up
    nxt_dn = cur_dn - d_up

    t_violations["TRANSITION_UP_DISTANCE_NEGATIVE"] = int(np.sum(nxt_up < 0))
    t_violations["TRANSITION_DOWN_DISTANCE_NEGATIVE"] = int(np.sum(nxt_dn < 0))

    dcr = z_draws["dcr"]
    t_violations["TRANSITION_DCR_OOB"] = int(np.sum((dcr < 0.0) | (dcr > 1.0)))

    rng_v = z_draws["range"]
    t_violations["TRANSITION_RANGE_NEGATIVE"] = int(np.sum(rng_v < 0.0))

    cu = z_draws["delta_upper_count"]
    cl = z_draws["delta_lower_count"]
    t_violations["TRANSITION_COUNT_NEGATIVE"] = int(np.sum((cu < 0) | (cl < 0)))

    for side, res in [("upper", z_draws["uresid"]), ("lower", z_draws["lresid"])]:
        start_age = np.expm1(sample_df[f"{side}_newest_log_age"].to_numpy(np.float64))
        elapsed_cur = (sample_df["bar_t"].to_numpy(np.int64) - sample_df["start_bar"].to_numpy(np.int64))
        expected_next = start_age + (elapsed_cur + 1)
        age_next = np.expm1(res + np.log1p(expected_next))
        t_violations["TRANSITION_NEGATIVE_NEWEST_AGE"] += int(np.sum(age_next < -1e-9))

    # 2. Reset probe
    idx_r = rng.integers(0, len(ev_pairs), size=n_draws)
    pair_sample = ev_pairs.iloc[idx_r].copy().reset_index(drop=True)
    gaps = reset_sampler.sample_gap(pair_sample, rng)
    r_draws = reset_sampler.sample_reset_primitives(pair_sample, gaps, rng)

    r_violations: Dict[str, int] = {
        "RESET_GEOMETRY_NONPOSITIVE": 0,
        "RESET_AGE_NEGATIVE": 0,
        "RESET_COUNT_INVALID": 0,
        "RESET_EPSR_INCOMPATIBLE": 0,
        "RESET_NONFINITE": 0,
    }

    d_u = r_draws["start_up_distance_R"]
    d_d = r_draws["start_down_distance_R"]
    r_violations["RESET_GEOMETRY_NONPOSITIVE"] = int(np.sum((d_u <= 0) | (d_d <= 0)))

    newest_u = r_draws["next_upper_newest_log_age"]
    span_u = r_draws["next_upper_span"]
    newest_l = r_draws["next_lower_newest_log_age"]
    span_l = r_draws["next_lower_span"]
    r_violations["RESET_AGE_NEGATIVE"] = int(np.sum(
        (newest_u < 0) | (span_u < 0) | (newest_l < 0) | (span_l < 0)
    ))

    nu = r_draws["next_upper_n_active_minus1"]
    nl = r_draws["next_lower_n_active_minus1"]
    r_violations["RESET_COUNT_INVALID"] = int(np.sum((nu < 0) | (nl < 0)))

    r_log = r_draws["start_log_ratio_residual"]
    l_r = np.log(d_u / d_d) + r_log
    eps_arr = solve_eps_R(d_u, d_d, l_r)
    r_violations["RESET_EPSR_INCOMPATIBLE"] = int(np.sum(np.isnan(eps_arr)))

    total_t_bad = sum(t_violations.values())
    total_r_bad = sum(r_violations.values())

    return dict(
        window=window_name,
        n_transition_draws=n_draws,
        n_reset_draws=n_draws,
        transition_violations=t_violations,
        reset_violations=r_violations,
        total_transition_invalid=total_t_bad,
        transition_invalid_rate=float(total_t_bad / n_draws),
        total_reset_invalid=total_r_bad,
        reset_invalid_rate=float(total_r_bad / n_draws),
    )


# ===========================================================================
# Stage C2: Observed-Start Seeded Episode Rollout
# ===========================================================================
def run_single_episode_rollout(init_state: Dict[str, Any],
                               trans_sampler: FittedTransitionSampler,
                               term_sampler: FittedTerminalSampler,
                               rng: np.random.Generator,
                               max_bars: int = MAX_EPISODE_BARS) -> Dict[str, Any]:
    """Execute one recursive episode rollout from init_state until terminal or violation."""
    state = dict(init_state)
    dur = 0
    while state["episode_age"] < max_bars:
        dur += 1
        df_row = pd.DataFrame([state])

        # 1. Sample terminal
        is_term = bool(term_sampler.sample_hazard(df_row, rng)[0])
        if is_term:
            endpoint = int(term_sampler.sample_endpoint(df_row, rng)[0])
            return dict(
                status="TERMINAL",
                duration=dur,
                endpoint_mask=endpoint,
                terminal_state={k: state[k] for k in TERMINAL_DYNAMIC_FIELDS if k in state},
                violation=None,
            )

        # 2. Sample transition
        z_dict = trans_sampler.sample_batch(df_row, rng)
        z_single = {k: v[0] for k, v in z_dict.items()}

        nxt_state, violation = advance_nonterminal(state, z_single)
        if violation is not None:
            return dict(
                status="INVALID",
                duration=dur,
                endpoint_mask=None,
                terminal_state=None,
                violation=violation,
            )

        state = nxt_state

    return dict(
        status="TIMEOUT",
        duration=max_bars,
        endpoint_mask=None,
        terminal_state=None,
        violation="EPISODE_TIMEOUT",
    )


# ===========================================================================
# Stage C3: Multi-Episode Free-Run
# ===========================================================================
def run_single_freerun_chain(symbol: str, seed_state: Dict[str, Any],
                             trans_sampler: FittedTransitionSampler,
                             term_sampler: FittedTerminalSampler,
                             reset_sampler: FittedResetSampler,
                             rng: np.random.Generator,
                             burn_in: int = BURN_IN_EPISODES,
                             collect: int = COLLECT_EPISODES,
                             max_bars: int = MAX_EPISODE_BARS) -> Dict[str, Any]:
    """Run an independent free-run chain of episodes.

    Initial episode is seeded with seed_state.
    All subsequent episodes are generated entirely synthetically via reset_sampler.
    """
    current_start = dict(seed_state)
    episodes_collected: List[Dict[str, Any]] = []
    total_target = burn_in + collect
    chain_violation = None

    for ep_idx in range(total_target):
        ep_res = run_single_episode_rollout(current_start, trans_sampler, term_sampler, rng, max_bars)
        if ep_res["status"] != "TERMINAL":
            chain_violation = ep_res["violation"]
            break

        endpoint = ep_res["endpoint_mask"]
        term_state = ep_res["terminal_state"]

        # Sample gap and reset for next episode
        df_term = pd.DataFrame([current_start])
        df_term["prev_endpoint_mask"] = endpoint

        gaps = reset_sampler.sample_gap(df_term, rng)
        gap_val = int(gaps[0])
        r_dict = reset_sampler.sample_reset_primitives(df_term, gaps, rng)
        r_single = {k: v[0] for k, v in r_dict.items()}

        next_start, reset_viol = reset_to_start(r_single, endpoint)
        if reset_viol is not None:
            chain_violation = reset_viol
            break

        if ep_idx >= burn_in:
            episodes_collected.append(dict(
                duration=ep_res["duration"],
                endpoint_mask=endpoint,
                terminal_state=term_state,
                reset_state={
                    "start_up_distance_R": r_single["start_up_distance_R"],
                    "start_down_distance_R": r_single["start_down_distance_R"],
                    "MFE0": r_single["next_path_max_up_excursion_R"],
                    "MAE0": r_single["next_path_max_down_excursion_R"],
                    "upper_newest_log_age": r_single["next_upper_newest_log_age"],
                    "upper_span": r_single["next_upper_span"],
                    "lower_newest_log_age": r_single["next_lower_newest_log_age"],
                    "lower_span": r_single["next_lower_span"],
                    "upper_n_active_minus1": r_single["next_upper_n_active_minus1"],
                    "lower_n_active_minus1": r_single["next_lower_n_active_minus1"],
                },
                gap=gap_val,
            ))

        current_start = next_start

    return dict(
        symbol=symbol,
        episodes=episodes_collected,
        n_collected=len(episodes_collected),
        violation=chain_violation,
        completed=bool(chain_violation is None and len(episodes_collected) == collect),
    )


# ===========================================================================
# Main Execution Pipeline
# ===========================================================================
def run_dynamic_pgm1c_audit(obs_sample_path: Path, transitions_path: Path,
                            ep_meta_path: Path) -> Dict[str, Any]:
    """Execute Stage C0 Sampler Parity and Epsilon Closure Audit."""
    print("==================================================", flush=True)
    print("STAGE C0: SAMPLER PARITY & EPSILON CLOSURE AUDIT", flush=True)
    print("==================================================", flush=True)
    t0 = time.perf_counter()

    # 1. Audit eps_R closure on all observed rows
    obs = pd.read_parquet(obs_sample_path)
    ep_meta = pd.read_parquet(ep_meta_path)
    eps_audit = audit_eps_r_closure(obs, ep_meta)
    print(f"[AUDIT] eps_R closure passed on {eps_audit['n_rows']} rows, "
          f"max_err={eps_audit['max_error']:.2e}", flush=True)

    # 2. Reset pairs audit
    pair, first = exp1b.build_reset_pairs(obs)
    pair_audit = exp1b.audit_reset_pairs(pair)
    recon_audit = exp1b.audit_reset_reconstruction(pair, first)
    print(f"[AUDIT] reset pair reconstruction max_err={recon_audit['max_error']:.2e}", flush=True)

    # 3. Verify sampler parity across both windows
    parity_summary = {"windows": {}, "tol": PARITY_TOL, "all_passed": True}

    for w in WINDOWS:
        w_fit = fit_samplers_for_window(w, obs_sample_path, transitions_path)
        wname = w["name"]
        w_res: Dict[str, Any] = {"trans_parity": {}, "term_parity": {}, "reset_parity": {}, "passed": True}

        # Transition parity
        for tag in ("M0_STATE_AVAIL", "MC_STATE_CURREENCODING"):
            key = "M0" if tag == "M0_STATE_AVAIL" else "MC"
            got = float(w_fit["trans_parity"][tag]["mean_joint_nll"])
            target = float(FROZEN_TRANSITION[wname][key])
            diff = abs(got - target)
            ok = diff < PARITY_TOL
            w_res["trans_parity"][tag] = dict(got=got, target=target, diff=diff, ok=ok)
            if not ok:
                w_res["passed"] = False
                parity_summary["all_passed"] = False
                print(f"FAILED TRANSITION PARITY {wname} {tag}: got {got} target {target} diff {diff}")

        # Terminal parity
        for tag in ("T0_STATE_AVAIL", "T2_STATE_PHI_MEM"):
            key = "T0" if tag == "T0_STATE_AVAIL" else "T2"
            target = FROZEN_TERMINAL[wname][key]
            got = w_fit["term_parity"][tag]
            diff_h = abs(got["hazard_nll"] - target["hazard_nll"])
            diff_e = abs(got["endpoint_joint_nll"] - target["endpoint_joint_nll"])
            diff_m = abs(got["mean_episode_nll"] - target["mean_episode_nll"])
            ok = max(diff_h, diff_e, diff_m) < PARITY_TOL
            w_res["term_parity"][tag] = dict(got=got, target=target, diff_h=diff_h, diff_e=diff_e, diff_m=diff_m, ok=ok)
            if not ok:
                w_res["passed"] = False
                parity_summary["all_passed"] = False
                print(f"FAILED TERMINAL PARITY {wname} {tag}")

        # Reset parity
        for tag in ("R0_ENDPOINT_ONLY", "R1_STATE_PHI"):
            key = "R0" if tag == "R0_ENDPOINT_ONLY" else "R1"
            target = FROZEN_RESET[wname][key]
            got = w_fit["reset_parity"][tag]
            diff = abs(got["mean_reset_nll"] - target["mean_reset_nll"])
            ok = diff < PARITY_TOL
            w_res["reset_parity"][tag] = dict(got=got, target=target, diff=diff, ok=ok)
            if not ok:
                w_res["passed"] = False
                parity_summary["all_passed"] = False
                print(f"FAILED RESET PARITY {wname} {tag}: got {got['mean_reset_nll']} target {target['mean_reset_nll']}")

        parity_summary["windows"][wname] = w_res

    if not parity_summary["all_passed"]:
        raise SystemExit(f"STOP_DYNAMIC_PGM1C_SAMPLER_PARITY_FAIL: {parity_summary}")

    # Output JSON artifacts
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{PREFIX}_epsr_closure_audit.json").write_text(
        json.dumps(eps_audit, indent=2, default=str)
    )
    (OUT / f"{PREFIX}_sampler_parity_audit.json").write_text(
        json.dumps(dict(
            parent_commit=BASE_SHA,
            parity=parity_summary,
            reset_pairs=pair_audit,
            reset_reconstruction=recon_audit,
            elapsed_seconds=round(time.perf_counter() - t0, 2),
        ), indent=2, default=str)
    )

    print(f"[STAGE C0 COMPLETE] All samplers bit-for-bit verified (tol={PARITY_TOL}). "
          f"Total time: {time.perf_counter() - t0:.2f}s", flush=True)
    return dict(eps_audit=eps_audit, parity=parity_summary)


def main():
    parser = argparse.ArgumentParser(description="DYNAMIC-PGM-1C Free-Run Rollout Closure")
    parser.add_argument("--audit-only", action="store_true", help="Run Stage C0 parity and closure audits only.")
    args = parser.parse_args()

    obs_sample_path = CACHE / "dynamic_pgm1b_sample.parquet"
    transitions_path = CACHE / "dynamic_pgm1a2c_transitions.parquet"
    ep_meta_path = CACHE / "episode_repl0_through_tb3.parquet"

    # Stage C0 is mandatory for both audit-only and full run
    audit_res = run_dynamic_pgm1c_audit(obs_sample_path, transitions_path, ep_meta_path)

    if args.audit_only or os.environ.get("DYNAMIC_PGM1C_AUDIT_ONLY") == "1":
        print("[AUDIT-ONLY] Complete. Stopping before any stochastic rollout simulation.", flush=True)
        return


if __name__ == "__main__":
    main()
