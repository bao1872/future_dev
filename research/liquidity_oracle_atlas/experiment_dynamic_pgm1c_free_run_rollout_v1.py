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
# Rollout Support Error & Fail-Closed Design Validation
# ===========================================================================
class RolloutSupportError(RuntimeError):
    """Raised when the generative model enters an illegal / numerically unrepresentable state.

    Used ONLY for rollout numerical-support failures (so that the failure becomes a
    scientific result, not a Python crash). Real programming bugs (ValueError,
    KeyError, etc.) must NOT be swallowed by this class -- they must still crash.
    """

    def __init__(self, reason: str, diagnostics: Optional[Dict[str, Any]] = None):
        super().__init__(reason)
        self.reason = reason
        self.diagnostics = diagnostics or {}


def transition_design_check(X64: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fail-closed two-stage float32 design validation for transition sampling.

    X64: dense float64 design matrix (the frozen float32 design densified).
    Returns (X32, diagnostics). Raises RolloutSupportError on:
      - nonfinite X64              -> TRANSITION_DESIGN_NONFINITE_FLOAT64
      - float32 numerical overflow -> TRANSITION_DESIGN_FLOAT32_OVERFLOW
    Does NOT change the frozen float32 sampling contract: the caller still uses the
    original `.astype(np.float32)` design for the actual head sampling.
    """
    X64 = np.asarray(X64, dtype=np.float64)
    if not np.all(np.isfinite(X64)):
        idx = int(np.argmax(~np.isfinite(X64)))
        raise RolloutSupportError("TRANSITION_DESIGN_NONFINITE_FLOAT64", {
            "n_nonfinite": int(np.sum(~np.isfinite(X64))),
            "first_bad_index": idx,
        })
    with np.errstate(over="ignore", invalid="ignore"):
        X32 = X64.astype(np.float32)
    if not np.all(np.isfinite(X32)):
        idx = int(np.argmax(~np.isfinite(X32)))
        raise RolloutSupportError("TRANSITION_DESIGN_FLOAT32_OVERFLOW", {
            "n_nonfinite": int(np.sum(~np.isfinite(X32))),
            "first_bad_index": idx,
        })
    return X32, {}


def safe_transform(pre: Any, df: pd.DataFrame, input_tag: str, design_tag: str) -> np.ndarray:
    """Fail-closed preprocessing for terminal / reset samplers.

    - raw numeric conditioning columns must be finite (else `input_tag`)
    - preprocessor output must be finite (else `design_tag`)
    Returns the dense float64 design matrix (unchanged semantics for predict_proba).
    """
    num = df.select_dtypes(include=[np.number])
    arr = num.to_numpy(dtype=np.float64)
    if arr.size and not np.all(np.isfinite(arr)):
        raise RolloutSupportError(input_tag, {"n_nonfinite": int(np.sum(~np.isfinite(arr)))})
    X = np.asarray(pbar.densify(pre.transform(df)), dtype=np.float64)
    if not np.all(np.isfinite(X)):
        raise RolloutSupportError(design_tag, {"n_nonfinite": int(np.sum(~np.isfinite(X)))})
    return X


# Tokens used to recognise numerical-support failures for gate/counter bookkeeping.
_NUMERIC_SUPPORT_TOKENS = (
    "INPUT_NONFINITE", "DESIGN_NONFINITE_FLOAT64", "DESIGN_FLOAT32_OVERFLOW",
    "GAUSSIAN_NONFINITE", "HURDLE_LOGMAG_OVERFLOW", "DCR_NONFINITE",
    "TERMINAL_INPUT", "TERMINAL_DESIGN", "RESET_INPUT", "RESET_DESIGN",
)


def is_numeric_support_error(violation: Optional[str]) -> bool:
    """True if `violation` is a rollout numerical-support failure (vs a physical one)."""
    if not isinstance(violation, str):
        return False
    return any(tok in violation for tok in _NUMERIC_SUPPORT_TOKENS)


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

WORLD_MODELS = {
    "W0": dict(terminal="T0_STATE_AVAIL", transition="M0_STATE_AVAIL", reset="R0_ENDPOINT_ONLY"),
    "W1": dict(terminal="T0_STATE_AVAIL", transition="MC_STATE_CURREENCODING", reset="R1_STATE_PHI"),
    "WT": dict(terminal="T2_STATE_PHI_MEM", transition="MC_STATE_CURREENCODING", reset="R1_STATE_PHI"),
}

CACHE = base.CACHE
OUT = base.OUT
SAMPLE_PATH = CACHE / "dynamic_pgm1b_sample.parquet"
TRANSITION_SAMPLE_PATH = CACHE / "dynamic_pgm1a2c_transitions.parquet"
EP_META_PATH = CACHE / "episode_repl0_through_tb3.parquet"
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

RESET_PAIR_TO_STRUCTURAL = {
    "start_up_distance_R": "next_start_up_distance_R",
    "start_down_distance_R": "next_start_down_distance_R",
    "MFE0": "next_path_max_up_excursion_R",
    "MAE0": "next_path_max_down_excursion_R",
    "upper_newest_log_age": "next_upper_newest_log_age",
    "upper_span": "next_upper_span",
    "lower_newest_log_age": "next_lower_newest_log_age",
    "lower_span": "next_lower_span",
    "upper_n_active_minus1": "next_upper_n_active_minus1",
    "lower_n_active_minus1": "next_lower_n_active_minus1",
}


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


def sample_gaussian(head: Any, X: np.ndarray, rng: np.random.Generator, node: str = "gaussian") -> np.ndarray:
    """Sample from Gaussian head: Z = X @ B + intercept + Normal(0, Sigma).

    Raises RolloutSupportError("<node>_GAUSSIAN_NONFINITE" if the conditional mean or the
    realized sample is nonfinite (numerical-support failure), instead of propagating inf.
    """
    X = np.asarray(X, dtype=np.float64)
    mu = X @ head.B + head.intercept
    if not np.all(np.isfinite(mu)):
        raise RolloutSupportError(f"{node}_GAUSSIAN_NONFINITE", {"stage": "conditional_mean"})
    z = rng.standard_normal((len(X), head.k))
    val = mu + z @ head.chol.T
    if not np.all(np.isfinite(val)):
        raise RolloutSupportError(f"{node}_GAUSSIAN_NONFINITE", {"stage": "sampled_value"})
    return val


def sample_hurdle_ln(head: Any, X: np.ndarray, rng: np.random.Generator, node: str = "hurdle") -> np.ndarray:
    """Sample from Hurdle-LogNormal head.

    Occurrence Logistic -> if active: logv ~ Gaussian, raw = sign * exp(logv), else 0.
    Raises RolloutSupportError("<node>_HURDLE_LOGMAG_OVERFLOW" before exp if logv is
    nonfinite or would overflow exp (no clipping -- overflow is a scientific result).
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    p = head.logit.predict_proba(X)[:, 1]
    active = rng.random(n) < p
    raw = np.zeros(n, dtype=np.float64)
    if np.any(active):
        logv = sample_gaussian(head.g, X[active], rng, node=node)[:, 0]
        if not np.all(np.isfinite(logv)):
            raise RolloutSupportError(f"{node}_HURDLE_LOGMAG_OVERFLOW", {"stage": "logv_nonfinite"})
        max_log = np.log(np.finfo(np.float64).max)
        if np.any(logv > max_log):
            raise RolloutSupportError(f"{node}_HURDLE_LOGMAG_OVERFLOW", {
                "stage": "logv_overflow", "max_logv": float(np.max(logv)),
            })
        raw[active] = head.sign * np.exp(logv)
    return raw


def sample_dcr(head: Any, X: np.ndarray, rng: np.random.Generator, node: str = "dcr") -> np.ndarray:
    """Sample from ZeroInteriorOneHead for DCR.

    3-class categorical {0, interior, 1}.
    Interior -> Gaussian logit -> sigmoid(z). No clipping.
    Raises RolloutSupportError("<node>_DCR_NONFINITE" if the interior value is nonfinite.
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
        z = sample_gaussian(head.g, X[interior], rng, node=node)[:, 0]
        sig = 1.0 / (1.0 + np.exp(-z))
        if not np.all(np.isfinite(sig)):
            raise RolloutSupportError(f"{node}_DCR_NONFINITE", {})
        raw[interior] = sig
    return raw


class ConstantOccurrenceModel:
    """Fallback constant occurrence probability model for single-class train data."""

    def __init__(self, p: float):
        self.p = float(p)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = np.full(len(X), self.p, dtype=np.float64)
        return np.column_stack([1.0 - p, p])


def fit_count_occurrence_models(Xtr: np.ndarray, Ytr: np.ndarray,
                                Xev: np.ndarray, Yev: np.ndarray,
                                tag: str) -> List[Any]:
    """Fit local count occurrence models matching frozen base.fit_state_count_head exactly."""
    models: List[Any] = []
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Ytr = np.asarray(Ytr, dtype=np.int64)

    for j in range(Ytr.shape[1]):
        yt = Ytr[:, j]
        zt = (yt > 0).astype(np.int64)
        if len(np.unique(zt)) < 2:
            p_const = 1.0 - 1e-6 if zt[0] == 1 else 1e-6
            models.append(ConstantOccurrenceModel(p_const))
        else:
            logit = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=3000)
            logit.fit(Xtr, zt)
            base._check_logistic_convergence(logit, f"{tag}_count_{j}_occurrence")
            models.append(logit)

    return models


def sample_count_head(model: Any, lam: float, X: np.ndarray, rng: np.random.Generator,
                     node: str = "count") -> np.ndarray:
    """Sample from state-dependent occurrence Logistic + constant ZTP magnitude."""
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    p = np.clip(model.predict_proba(X)[:, 1], 1e-6, 1.0 - 1e-6)
    active = rng.random(n) < p
    cnt = np.zeros(n, dtype=np.int64)
    n_active = int(np.sum(active))
    if n_active > 0:
        cnt[active] = sample_ztp_vec(lam, rng, n_active)
    if not np.all(cnt >= 0):
        raise RolloutSupportError(f"{node}_COUNT_INVALID", {"stage": "negative_count"})
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

    def __init__(self, tag: str, ct: Any, heads: Dict[str, Any],
                 count_head: Dict[str, Any], count_occ_models: List[Any],
                 train_min: Optional[np.ndarray] = None,
                 train_max: Optional[np.ndarray] = None,
                 train_absmax: Optional[np.ndarray] = None,
                 feature_names: Optional[List[str]] = None,
                 design_cols: Optional[List[str]] = None):
        self.tag = tag
        self.ct = ct
        self.heads = heads
        self.count_head = count_head
        self.constant_rates = count_head["constant_rates"]
        self.count_occ_models = count_occ_models
        # Training-domain support statistics (diagnostic ONLY, never a gate).
        self.train_min = np.asarray(train_min if train_min is not None else np.array([]), dtype=np.float64)
        self.train_max = np.asarray(train_max if train_max is not None else np.array([]), dtype=np.float64)
        self.train_absmax = np.asarray(train_absmax if train_absmax is not None else np.array([]), dtype=np.float64)
        self.train_feature_names = list(feature_names) if feature_names is not None else []
        # Design-input columns actually consumed by ct (NOT feature_names_in_, which also
        # lists target columns that are absent from rollout/observed-state inputs).
        self.design_cols = list(design_cols) if design_cols is not None else None
        # Last rollout-step design diagnostics (set by sample_batch).
        self.last_design_diag: Optional[Dict[str, Any]] = None

    def _design_diagnostics(self, X64: np.ndarray) -> Dict[str, Any]:
        """Training-domain extrapolation diagnostics. NOT a gate -- pure observation."""
        X64 = np.asarray(X64, dtype=np.float64)
        if X64.size == 0:
            return dict(max_abs_design=0.0, max_extrapolation_ratio=0.0,
                        n_features_outside_train_minmax=0, worst_feature_name=None,
                        worst_feature_value=0.0, train_feature_min=None, train_feature_max=None)
        absX = np.abs(X64)
        max_abs = float(np.max(absX))
        denom = np.maximum(self.train_absmax, 1e-12)
        ratios = absX / denom
        max_ratio = float(np.max(ratios))
        outside = int(np.sum(~((X64 >= self.train_min) & (X64 <= self.train_max))))
        fi = int(np.argmax(ratios))
        widx = fi % ratios.shape[1]
        worst_val = float(X64.reshape(-1)[fi])
        names = self.train_feature_names
        worst_name = names[widx] if 0 <= widx < len(names) else None
        train_fmin = float(self.train_min[widx]) if 0 <= widx < len(self.train_min) else None
        train_fmax = float(self.train_max[widx]) if 0 <= widx < len(self.train_max) else None
        return dict(
            max_abs_design=max_abs,
            max_extrapolation_ratio=max_ratio,
            n_features_outside_train_minmax=outside,
            worst_feature_name=worst_name,
            worst_feature_value=worst_val,
            train_feature_min=train_fmin,
            train_feature_max=train_fmax,
        )

    def sample_batch(self, df: pd.DataFrame, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        df_in = df
        needed_zt = [c for c in rep.MC_EXTRA if c.startswith("zt_")]
        # zt_* encodes the previous step's sampled z. In the rollout it is always finite:
        # at episode start it is derived from the current embedding (phi_*), and after each
        # step it equals the freshly-drawn z. So derive/repair zt_* from phi_* BEFORE the
        # finite guard -- a NaN zt_* (first-step / no-prior row in a transition dataset) is not
        # a real support failure, it is the legitimate "no previous step" encoding.
        if needed_zt:
            df_in = df.copy()
            for zc in needed_zt:
                pc = zc.replace("zt_", "phi_")
                if zc not in df_in.columns:
                    df_in[zc] = df_in[pc] if pc in df_in.columns else 0.0
                else:
                    df_in[zc] = df_in[zc].fillna(df_in[pc]) if pc in df_in.columns else df_in[zc].fillna(0.0)

        # 0. raw input finite guard (only the design-input columns actually consumed by ct;
        #     NOT feature_names_in_, which also lists target columns absent from observed-state inputs)
        feat_cols = self.design_cols
        if feat_cols:
            sub = df_in[[c for c in feat_cols if c in df_in.columns]]
            num = sub.select_dtypes(include=[np.number])
            if num.size:
                arr = num.to_numpy(dtype=np.float64)
                if not np.all(np.isfinite(arr)):
                    raise RolloutSupportError("TRANSITION_INPUT_NONFINITE", {
                        "n_nonfinite": int(np.sum(~np.isfinite(arr))),
                    })

        # 1. transform (frozen float32 contract starts here)
        raw = self.ct.transform(df_in)
        X64 = np.asarray(pbar.densify(raw), dtype=np.float64)

        # 2. fail-closed design validation (raises RolloutSupportError before sampling)
        transition_design_check(X64)

        # 3. training-domain extrapolation diagnostics (NOT a gate)
        self.last_design_diag = self._design_diagnostics(X64)

        # 4. frozen float32 design matrix for actual head sampling (unchanged semantics)
        X = raw.astype(np.float32)
        n = len(df)
        nodes = self.heads["nodes"]

        # Continuous nodes
        d_up = sample_gaussian(nodes["z_d_up"]["head"], X, rng, node="z_d_up")[:, 0]
        dmfe = sample_hurdle_ln(nodes["z_dmfe"]["head"], X, rng, node="z_dmfe")
        dmae = sample_hurdle_ln(nodes["z_dmae"]["head"], X, rng, node="z_dmae")
        dcr = sample_dcr(nodes["z_dcr"]["head"], X, rng, node="z_dcr")
        rng_val = sample_hurdle_ln(nodes["z_range"]["head"], X, rng, node="z_range")
        uresid = sample_hurdle_ln(nodes["z_uresid"]["head"], X, rng, node="z_uresid")
        lresid = sample_hurdle_ln(nodes["z_lresid"]["head"], X, rng, node="z_lresid")

        # Counts
        lam_u = float(self.constant_rates[0])
        lam_l = float(self.constant_rates[1])
        X_64 = np.asarray(X, dtype=np.float64)
        c_u = sample_count_head(self.count_occ_models[0], lam_u, X_64, rng, node="z_delta_upper_count")
        c_l = sample_count_head(self.count_occ_models[1], lam_l, X_64, rng, node="z_delta_lower_count")

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
        X = safe_transform(self.pre, df, "TERMINAL_INPUT_NONFINITE", "TERMINAL_DESIGN_NONFINITE")
        p_h = self.clf.predict_proba(X)[:, 1]
        return rng.random(len(df)) < p_h

    def sample_endpoint(self, df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        X = safe_transform(self.pre, df, "TERMINAL_INPUT_NONFINITE", "TERMINAL_DESIGN_NONFINITE")
        return sample_crf_endpoint(self.theta, X, rng)


class FittedResetSampler:
    """Fitted reset sampler for R0 or R1."""

    def __init__(self, tag: str, occ_cols: List[str], state_cols: List[str],
                 pre_occ: Any, pre_state: Any, gap_occ_clf: Any,
                 gap_p: float, heads: Dict[str, Any], count_occ_models: List[Any]):
        self.tag = tag
        self.occ_cols = occ_cols
        self.state_cols = state_cols
        self.pre_occ = pre_occ
        self.pre_state = pre_state
        self.gap_occ_clf = gap_occ_clf
        self.gap_p = gap_p
        self.heads = heads
        self.constant_rates = heads["count_rates"]
        self.count_occ_models = count_occ_models

    def sample_gap(self, df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        for c in self.occ_cols + MASK_CAT:
            if c not in df.columns:
                raise SystemExit(f"STOP_DYNAMIC_PGM1C_RESET_INPUT_MISSING_COL: {c}")
        cols = self.occ_cols + MASK_CAT
        X_occ = safe_transform(self.pre_occ, df[cols], "RESET_INPUT_NONFINITE", "RESET_DESIGN_NONFINITE")
        p_gap = self.gap_occ_clf.predict_proba(X_occ)[:, 1]
        has_gap = rng.random(len(df)) < p_gap
        gap_bars = np.zeros(len(df), dtype=np.int64)
        n_pos = int(np.sum(has_gap))
        if n_pos > 0:
            gap_bars[has_gap] = rng.geometric(self.gap_p, size=n_pos)
        return gap_bars

    def sample_reset_primitives(self, df: pd.DataFrame, gap_bars: np.ndarray,
                                rng: np.random.Generator) -> Dict[str, np.ndarray]:
        for c in self.occ_cols + MASK_CAT:
            if c not in df.columns:
                raise SystemExit(f"STOP_DYNAMIC_PGM1C_RESET_INPUT_MISSING_COL: {c}")
        df_state = df.copy()
        df_state["gap_positive"] = (gap_bars > 0).astype(np.int64)
        df_state["log1p_gap"] = np.log1p(gap_bars.astype(np.float64))
        for c in self.state_cols + MASK_CAT:
            if c not in df_state.columns:
                raise SystemExit(f"STOP_DYNAMIC_PGM1C_RESET_INPUT_MISSING_COL: {c}")
        cols = self.state_cols + MASK_CAT
        X_s = safe_transform(self.pre_state, df_state[cols], "RESET_INPUT_NONFINITE", "RESET_DESIGN_NONFINITE")

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
        X_s_64 = np.asarray(X_s, dtype=np.float64)
        n_act_u_minus1 = sample_count_head(self.count_occ_models[0], lam_u, X_s_64, rng)
        n_act_l_minus1 = sample_count_head(self.count_occ_models[1], lam_l, X_s_64, rng)

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
    nxt["cur_width_R"] = nxt_up + nxt_dn

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


def build_observed_start_state(first_row: Any, eps_R: float) -> Dict[str, Any]:
    """Construct initial state dictionary S^0 from the observed first row of an episode."""
    row = dict(first_row)
    d_u = float(row["start_up_distance_R"])
    d_d = float(row["start_down_distance_R"])
    st: Dict[str, Any] = {
        "start_up_distance_R": d_u,
        "start_down_distance_R": d_d,
        "start_width_R": float(row["start_width_R"]),
        "start_log_ratio": float(row["start_log_ratio"]),
        "upper_oldest_log_age": float(row["upper_oldest_log_age"]),
        "upper_newest_log_age": float(row["upper_newest_log_age"]),
        "upper_newest_age_zero": float(row["upper_newest_age_zero"]),
        "upper_oldest_age_zero": float(row["upper_oldest_age_zero"]),
        "upper_n_active_identities": float(row["upper_n_active_identities"]),
        "lower_oldest_log_age": float(row["lower_oldest_log_age"]),
        "lower_newest_log_age": float(row["lower_newest_log_age"]),
        "lower_newest_age_zero": float(row["lower_newest_age_zero"]),
        "lower_oldest_age_zero": float(row["lower_oldest_age_zero"]),
        "lower_n_active_identities": float(row["lower_n_active_identities"]),
        "elapsed_log": 0.0,
        "cur_up_distance_R": float(row["cur_up_distance_R"]),
        "cur_down_distance_R": float(row["cur_down_distance_R"]),
        "cur_width_R": float(row["cur_width_R"]),
        "cur_log_ratio": float(row["cur_log_ratio"]),
        "path_total_variation_R": 0.0,
        "path_max_up_excursion_R": float(row["path_max_up_excursion_R"]),
        "path_max_down_excursion_R": float(row["path_max_down_excursion_R"]),
        "path_direction_change_rate": 0.0,
        "path_last_return_R": 0.0,
        "path_current_bar_range_R": float(row["path_current_bar_range_R"]),
        "upper_newest_log_age_residual": 0.0,
        "upper_current_newest_age_zero": float(row["upper_current_newest_age_zero"]),
        "upper_active_identity_count_delta": 0.0,
        "lower_newest_log_age_residual": 0.0,
        "lower_current_newest_age_zero": float(row["lower_current_newest_age_zero"]),
        "lower_active_identity_count_delta": 0.0,
        "tempo_signed_speed": 0.0,
        "tempo_abs_speed": 0.0,
        "tempo_signed_efficiency": 0.0,
        "tempo_abs_efficiency": 0.0,
        "prev_event_mask": int(row["prev_event_mask"]),
        LAG_AVAIL: 0.0,
        "eps_R": float(eps_R),
        "episode_age": 0,
        "symbol": str(row.get("symbol", "")),
        "block": str(row.get("block", "")),
        "episode_id": str(row.get("episode_id", "")),
    }
    _phi_raws = {
        "z_d_up": 0.0,
        "z_dcr": 0.0,
        "z_range": st["path_current_bar_range_R"],
        "z_uresid": 0.0,
        "z_lresid": 0.0,
    }
    for node, raw in _phi_raws.items():
        for zc, vals in rep._encode_with_frozen(node, np.array([raw])).items():
            st[f"phi_{zc}"] = float(vals[0])

    for c in ["mem_z_dmfe_ispos", "mem_z_dmfe_log", "mem_z_dmae_ispos", "mem_z_dmae_log"]:
        st[c] = 0.0
    st["mem_z_delta_upper_count"] = np.nan
    st["mem_z_delta_lower_count"] = np.nan
    return st


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

        reset_count_occ_models = fit_count_occurrence_models(Xtr_s, Ytr, Xev_s, Yev, f"{w['name']}_{tag}_reset")
        for j, m in enumerate(reset_count_occ_models):
            p_eval = np.clip(m.predict_proba(Xev_s.astype(np.float64))[:, 1], 1e-6, 1.0 - 1e-6)
            diff = float(np.max(np.abs(p_eval - kc["p0_ev"][:, j])))
            if diff >= 1e-12:
                raise SystemExit(f"STOP_DYNAMIC_PGM1C_RESET_COUNT_OCC_MISMATCH: {w['name']} {tag} col {j} diff {diff}")

        total_nll = (nll_occ
                     + exp1b.gap_positive_nll(pev["gap_bars"].to_numpy(np.float64), gap_p)
                     + nll_geom + nll_rlog + nll_shape_age + nll_cnt)

        mean_res_nll = float(total_nll.mean())
        reset_parity_eval[tag] = dict(mean_reset_nll=mean_res_nll)
        reset_samplers[tag] = FittedResetSampler(
            tag, occ_cols, state_cols, pre_o, pre_s, clf_gap, gap_p, heads, reset_count_occ_models
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
        Xtr_sparse = ct.fit_transform(tr_trans)
        Xtr = Xtr_sparse.astype(np.float32)
        Xev_sparse = ct.transform(ev_trans)
        Xev = Xev_sparse.astype(np.float32)

        # Training-domain support statistics (diagnostic only; never gates sampling).
        Xtr_d = np.asarray(pbar.densify(Xtr_sparse), dtype=np.float64)
        train_min = Xtr_d.min(axis=0)
        train_max = Xtr_d.max(axis=0)
        train_absmax = np.abs(Xtr_d).max(axis=0)
        feature_names = list(ct.get_feature_names_out())

        k = base.fit_state_heads(Xtr, Xev, Zc_tr, Zc_ev, yd_tr, yd_ev)
        kc = base.fit_state_count_head(Xtr, Yc_tr, Xev, Yc_ev, constant_rates=k0_count["constant_rates"])

        trans_count_occ_models = fit_count_occurrence_models(Xtr, Yc_tr, Xev, Yc_ev, f"{w['name']}_{tag}_trans")
        for j, m in enumerate(trans_count_occ_models):
            p_eval = np.clip(m.predict_proba(Xev.astype(np.float64))[:, 1], 1e-6, 1.0 - 1e-6)
            diff = float(np.max(np.abs(p_eval - kc["p0_ev"][:, j])))
            if diff >= 1e-12:
                raise SystemExit(f"STOP_DYNAMIC_PGM1C_TRANS_COUNT_OCC_MISMATCH: {w['name']} {tag} col {j} diff {diff}")

        cont = rep._node_eval_nll(k["nodes"])
        disc = k["disc_ev"]
        cnt = kc["nll_ev"].sum(axis=1)
        mean_j = float(np.mean(cont + disc + cnt))

        trans_parity_eval[tag] = dict(mean_joint_nll=mean_j)
        trans_samplers[tag] = FittedTransitionSampler(
            tag, ct, k, kc, trans_count_occ_models,
            train_min=train_min, train_max=train_max,
            train_absmax=train_absmax, feature_names=feature_names,
            design_cols=num_cols,
        )
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

    t_invalid_mask = np.zeros(n_draws, dtype=bool)
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

    v_up_neg = (nxt_up < 0)
    t_violations["TRANSITION_UP_DISTANCE_NEGATIVE"] = int(np.sum(v_up_neg))
    t_invalid_mask |= v_up_neg

    v_dn_neg = (nxt_dn < 0)
    t_violations["TRANSITION_DOWN_DISTANCE_NEGATIVE"] = int(np.sum(v_dn_neg))
    t_invalid_mask |= v_dn_neg

    v_age_neg = np.zeros(n_draws, dtype=bool)
    for side, res in [("upper", z_draws["uresid"]), ("lower", z_draws["lresid"])]:
        start_age = np.expm1(sample_df[f"{side}_newest_log_age"].to_numpy(np.float64))
        elapsed_cur = (sample_df["bar_t"].to_numpy(np.int64) - sample_df["start_bar"].to_numpy(np.int64))
        expected_next = start_age + (elapsed_cur + 1)
        age_next = np.expm1(res + np.log1p(expected_next))
        v_age_neg |= (age_next < -1e-9)
    t_violations["TRANSITION_NEGATIVE_NEWEST_AGE"] = int(np.sum(v_age_neg))
    t_invalid_mask |= v_age_neg

    dcr = z_draws["dcr"]
    v_dcr_oob = (dcr < 0.0) | (dcr > 1.0)
    t_violations["TRANSITION_DCR_OOB"] = int(np.sum(v_dcr_oob))
    t_invalid_mask |= v_dcr_oob

    rng_v = z_draws["range"]
    v_rng_neg = (rng_v < 0.0)
    t_violations["TRANSITION_RANGE_NEGATIVE"] = int(np.sum(v_rng_neg))
    t_invalid_mask |= v_rng_neg

    cu = z_draws["delta_upper_count"]
    cl = z_draws["delta_lower_count"]
    v_cnt_nonint = (~np.equal(np.mod(cu, 1), 0)) | (~np.equal(np.mod(cl, 1), 0))
    t_violations["TRANSITION_COUNT_NONINTEGER"] = int(np.sum(v_cnt_nonint))
    t_invalid_mask |= v_cnt_nonint

    v_cnt_neg = (cu < 0) | (cl < 0)
    t_violations["TRANSITION_COUNT_NEGATIVE"] = int(np.sum(v_cnt_neg))
    t_invalid_mask |= v_cnt_neg

    v_t_nonfin = np.zeros(n_draws, dtype=bool)
    for col_arr in [d_up, nxt_up, nxt_dn, dcr, rng_v, cu, cl, z_draws["uresid"], z_draws["lresid"]]:
        v_t_nonfin |= (~np.isfinite(col_arr))
    t_violations["TRANSITION_NONFINITE"] = int(np.sum(v_t_nonfin))
    t_invalid_mask |= v_t_nonfin

    total_t_invalid = int(np.sum(t_invalid_mask))

    # 2. Reset probe
    idx_r = rng.integers(0, len(ev_pairs), size=n_draws)
    pair_sample = ev_pairs.iloc[idx_r].copy().reset_index(drop=True)
    gaps = reset_sampler.sample_gap(pair_sample, rng)
    r_draws = reset_sampler.sample_reset_primitives(pair_sample, gaps, rng)

    r_invalid_mask = np.zeros(n_draws, dtype=bool)
    r_violations: Dict[str, int] = {
        "RESET_GEOMETRY_NONPOSITIVE": 0,
        "RESET_AGE_NEGATIVE": 0,
        "RESET_COUNT_INVALID": 0,
        "RESET_EPSR_INCOMPATIBLE": 0,
        "RESET_NONFINITE": 0,
    }

    d_u = r_draws["start_up_distance_R"]
    d_d = r_draws["start_down_distance_R"]
    v_geom_nonpos = (d_u <= 0) | (d_d <= 0)
    r_violations["RESET_GEOMETRY_NONPOSITIVE"] = int(np.sum(v_geom_nonpos))
    r_invalid_mask |= v_geom_nonpos

    newest_u = r_draws["next_upper_newest_log_age"]
    span_u = r_draws["next_upper_span"]
    newest_l = r_draws["next_lower_newest_log_age"]
    span_l = r_draws["next_lower_span"]
    v_age_neg_r = (newest_u < 0) | (span_u < 0) | (newest_l < 0) | (span_l < 0)
    r_violations["RESET_AGE_NEGATIVE"] = int(np.sum(v_age_neg_r))
    r_invalid_mask |= v_age_neg_r

    nu = r_draws["next_upper_n_active_minus1"]
    nl = r_draws["next_lower_n_active_minus1"]
    v_cnt_inv = (nu < 0) | (nl < 0) | (~np.equal(np.mod(nu, 1), 0)) | (~np.equal(np.mod(nl, 1), 0))
    r_violations["RESET_COUNT_INVALID"] = int(np.sum(v_cnt_inv))
    r_invalid_mask |= v_cnt_inv

    r_log = r_draws["start_log_ratio_residual"]
    l_r = np.log(d_u / d_d) + r_log
    eps_arr = solve_eps_R(d_u, d_d, l_r)
    v_eps_incomp = np.isnan(eps_arr) | (eps_arr <= 0)
    r_violations["RESET_EPSR_INCOMPATIBLE"] = int(np.sum(v_eps_incomp))
    r_invalid_mask |= v_eps_incomp

    v_r_nonfin = np.zeros(n_draws, dtype=bool)
    for k, arr in r_draws.items():
        v_r_nonfin |= (~np.isfinite(arr))
    r_violations["RESET_NONFINITE"] = int(np.sum(v_r_nonfin))
    r_invalid_mask |= v_r_nonfin

    total_r_invalid = int(np.sum(r_invalid_mask))

    return dict(
        window=window_name,
        n_draws=n_draws,
        n_transition_draws=n_draws,
        n_reset_draws=n_draws,
        transition_violations=t_violations,
        reset_violations=r_violations,
        invalid_draw_count=total_t_invalid + total_r_invalid,
        invalid_draw_rate=float((total_t_invalid + total_r_invalid) / (2 * n_draws)),
        total_transition_invalid=total_t_invalid,
        transition_invalid_rate=float(total_t_invalid / n_draws),
        total_reset_invalid=total_r_invalid,
        reset_invalid_rate=float(total_r_invalid / n_draws),
    )


# ===========================================================================
# Stage C2: Observed-Start Seeded Episode Rollout
# ===========================================================================
def run_single_episode_rollout(init_state: Dict[str, Any],
                               trans_sampler: FittedTransitionSampler,
                               term_sampler: FittedTerminalSampler,
                               rng: np.random.Generator,
                               max_bars: int = MAX_EPISODE_BARS) -> Dict[str, Any]:
    """Execute one recursive episode rollout from init_state until terminal or violation.

    A RolloutSupportError raised by any sampler is caught and turned into a structured
    INVALID result (failure_step / failure_diagnostics) -- it does NOT kill the process.
    Physical-support violations from advance_nonterminal/reset_to_start still return their
    own violation strings (they are not wrapped). Real programmer errors propagate.
    """
    state = dict(init_state)
    dur = 0
    max_extr = 0.0
    first_outside_step = None
    max_abs_design = 0.0
    worst: Dict[str, Any] = dict(
        worst_feature_name=None, worst_feature_value=0.0,
        train_feature_min=None, train_feature_max=None,
    )

    def _update_diag(dg: Optional[Dict[str, Any]]) -> None:
        if not dg:
            return
        nonlocal max_extr, first_outside_step, max_abs_design
        max_extr = max(max_extr, float(dg.get("max_extrapolation_ratio", 0.0)))
        max_abs_design = max(max_abs_design, float(dg.get("max_abs_design", 0.0)))
        if first_outside_step is None and int(dg.get("n_features_outside_train_minmax", 0)) > 0:
            first_outside_step = dur
        wn = dg.get("worst_feature_name")
        if wn is not None:
            worst["worst_feature_name"] = wn
            worst["worst_feature_value"] = dg.get("worst_feature_value", 0.0)
            worst["train_feature_min"] = dg.get("train_feature_min")
            worst["train_feature_max"] = dg.get("train_feature_max")

    def _result(status, violation, duration, endpoint_mask=None):
        return dict(
            status=status,
            duration=duration,
            endpoint_mask=endpoint_mask,
            terminal_state=({k: state[k] for k in TERMINAL_DYNAMIC_FIELDS if k in state}
                            if status == "TERMINAL" else None),
            terminal_full_state=(dict(state) if status == "TERMINAL" else None),
            violation=violation,
            failure_step=None,
            failure_diagnostics=None,
            max_extrapolation_ratio_seen=max_extr,
            first_outside_train_step=first_outside_step,
            max_abs_design_seen=max_abs_design,
            worst_feature_name=worst["worst_feature_name"],
            worst_feature_value=worst["worst_feature_value"],
            train_feature_min=worst["train_feature_min"],
            train_feature_max=worst["train_feature_max"],
        )

    try:
        while state["episode_age"] < max_bars:
            dur += 1
            df_row = pd.DataFrame([state])

            # 1. Sample terminal
            is_term = bool(term_sampler.sample_hazard(df_row, rng)[0])
            if is_term:
                endpoint = int(term_sampler.sample_endpoint(df_row, rng)[0])
                return _result("TERMINAL", None, dur, endpoint)

            # 2. Sample transition
            z_dict = trans_sampler.sample_batch(df_row, rng)
            _update_diag(getattr(trans_sampler, "last_design_diag", None))
            z_single = {k: v[0] for k, v in z_dict.items()}

            nxt_state, violation = advance_nonterminal(state, z_single)
            if violation is not None:
                return _result("INVALID", violation, dur)
            state = nxt_state
    except RolloutSupportError as exc:
        return dict(
            status="INVALID",
            duration=dur,
            endpoint_mask=None,
            terminal_state=None,
            terminal_full_state=None,
            violation=exc.reason,
            failure_step=dur,
            failure_diagnostics=exc.diagnostics,
            max_extrapolation_ratio_seen=max_extr,
            first_outside_train_step=first_outside_step,
            max_abs_design_seen=max_abs_design,
            worst_feature_name=worst["worst_feature_name"],
            worst_feature_value=worst["worst_feature_value"],
            train_feature_min=worst["train_feature_min"],
            train_feature_max=worst["train_feature_max"],
        )

    return _result("TIMEOUT", "EPISODE_TIMEOUT", max_bars)


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
    failure_episode = None
    failure_step = None
    failure_diagnostics = None

    def _incomplete_chain(violation, ep_idx, step=None, diag=None):
        return dict(
            symbol=symbol,
            episodes=episodes_collected,
            n_collected=len(episodes_collected),
            violation=violation,
            failure_episode=ep_idx,
            failure_step=step,
            failure_diagnostics=diag,
            completed=False,
        )

    for ep_idx in range(total_target):
        try:
            ep_res = run_single_episode_rollout(current_start, trans_sampler, term_sampler, rng, max_bars)
        except RolloutSupportError as exc:
            return _incomplete_chain(exc.reason, ep_idx, exc.diagnostics.get("failure_step"), exc.diagnostics)
        if ep_res["status"] != "TERMINAL":
            return _incomplete_chain(ep_res["violation"], ep_idx,
                                     ep_res.get("failure_step"), ep_res.get("failure_diagnostics"))

        endpoint = ep_res["endpoint_mask"]
        term_state = ep_res["terminal_state"]
        term_full = ep_res["terminal_full_state"]

        # Sample gap and reset for next episode using terminal_full_state
        df_term = pd.DataFrame([term_full])
        df_term["prev_endpoint_mask"] = endpoint

        # R1 reset input contract check
        if hasattr(reset_sampler, "occ_cols"):
            required_cols = list(reset_sampler.occ_cols) + MASK_CAT
            for c in required_cols:
                if c not in df_term.columns:
                    raise SystemExit(f"STOP_DYNAMIC_PGM1C_RESET_INPUT_MISSING_COL: {c}")

        try:
            gaps = reset_sampler.sample_gap(df_term, rng)
            gap_val = int(gaps[0])
            r_dict = reset_sampler.sample_reset_primitives(df_term, gaps, rng)
        except RolloutSupportError as exc:
            return _incomplete_chain(exc.reason, ep_idx, exc.diagnostics.get("failure_step"), exc.diagnostics)
        r_single = {k: v[0] for k, v in r_dict.items()}

        next_start, reset_viol = reset_to_start(r_single, endpoint)
        if reset_viol is not None:
            return _incomplete_chain(reset_viol, ep_idx, None, None)

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
        failure_episode=failure_episode,
        failure_step=failure_step,
        failure_diagnostics=failure_diagnostics,
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


def build_observed_comparators(ev_obs: pd.DataFrame, ev_pairs: pd.DataFrame) -> Dict[str, Any]:
    """Construct pooled and per-symbol empirical comparators for eval window."""
    ev_terms = ev_obs[ev_obs["hazard"] == 1].copy().reset_index(drop=True)
    ev_terms["duration"] = ev_terms["bar_t"] - ev_terms["start_bar"] + 1
    ev_valid_pairs = ev_pairs[ev_pairs["next_episode_id"].notna()].copy().reset_index(drop=True)

    def _extract_pack(df_t: pd.DataFrame, df_p: pd.DataFrame) -> Dict[str, Any]:
        term_dict = {fld: df_t[fld].to_numpy(np.float64) for fld in TERMINAL_DYNAMIC_FIELDS if fld in df_t}
        reset_dict = {
            fld: df_p[RESET_PAIR_TO_STRUCTURAL[fld]].to_numpy(np.float64)
            for fld in RESET_STRUCTURAL_FIELDS if RESET_PAIR_TO_STRUCTURAL[fld] in df_p
        }
        return dict(
            durations=df_t["duration"].to_numpy(np.float64) if len(df_t) else np.empty(0, dtype=np.float64),
            endpoints=df_t["target_mask"].to_numpy(np.int64) if len(df_t) else np.empty(0, dtype=np.int64),
            terminals=term_dict,
            resets=reset_dict,
            gaps=df_p["gap_bars"].to_numpy(np.float64) if len(df_p) else np.empty(0, dtype=np.float64),
        )

    pooled = _extract_pack(ev_terms, ev_valid_pairs)
    by_symbol = {}
    symbols = sorted(ev_obs["symbol"].unique())
    for s in symbols:
        st = ev_terms[ev_terms["symbol"] == s]
        sp = ev_valid_pairs[ev_valid_pairs["symbol"] == s]
        by_symbol[s] = _extract_pack(st, sp)

    return dict(pooled=pooled, by_symbol=by_symbol)


def run_stage_c2_rollout(window_name: str,
                         eval_first_rows: pd.DataFrame,
                         fitted_samplers: Dict[str, Any],
                         seed: int,
                         n_reps: int = N_SEEDED_REPS,
                         max_bars: int = MAX_EPISODE_BARS) -> Dict[str, Any]:
    """Execute Stage C2: Observed-Start Seeded Episode Rollout."""
    print(f"[STAGE C2] Seeded episode rollout on {window_name} ({len(eval_first_rows)} episodes x {n_reps} reps)...", flush=True)
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    records: List[Dict[str, Any]] = []
    w1_invalid = 0
    w1_timeout = 0
    w1_nonfinite = 0

    n_done = 0
    term_ct = inv_ct = to_ct = 0
    last_viol = None
    n_starts = len(eval_first_rows)

    for model_key in ["W0", "W1", "WT"]:
        m_spec = WORLD_MODELS[model_key]
        term_sampler = fitted_samplers["term_samplers"][m_spec["terminal"]]
        trans_sampler = fitted_samplers["trans_samplers"][m_spec["transition"]]

        for rep in range(n_reps):
            for i in range(n_starts):
                n_done += 1
                fr = eval_first_rows.iloc[i]
                st = build_observed_start_state(fr, fr["eps_R"])
                ep_res = run_single_episode_rollout(st, trans_sampler, term_sampler, rng, max_bars)
                v = ep_res["violation"]

                if ep_res["status"] == "TERMINAL":
                    term_ct += 1
                elif ep_res["status"] == "INVALID":
                    inv_ct += 1
                    last_viol = v
                elif ep_res["status"] == "TIMEOUT":
                    to_ct += 1
                    last_viol = v

                rec = {
                    "window": window_name,
                    "model": model_key,
                    "symbol": str(fr["symbol"]),
                    "episode_id": str(fr["episode_id"]),
                    "rep": rep,
                    "duration": ep_res["duration"],
                    "endpoint": ep_res["endpoint_mask"],
                    "status": ep_res["status"],
                    "violation": v,
                    "failure_step": ep_res.get("failure_step"),
                    "max_extrapolation_ratio_seen": ep_res.get("max_extrapolation_ratio_seen"),
                    "first_outside_train_step": ep_res.get("first_outside_train_step"),
                    "worst_feature_name": ep_res.get("worst_feature_name"),
                }
                records.append(rec)

                if model_key == "W1":
                    if ep_res["status"] == "INVALID":
                        w1_invalid += 1
                        if is_numeric_support_error(v):
                            w1_nonfinite += 1
                    elif ep_res["status"] == "TIMEOUT":
                        w1_timeout += 1

                if n_done % 250 == 0:
                    print(f"[C2 PROGRESS] {window_name} model={model_key} rep={rep} "
                          f"i={i}/{n_starts} elapsed={time.perf_counter() - t0:.1f}s "
                          f"term={term_ct} invalid={inv_ct} timeout={to_ct} last_violation={last_viol}",
                          flush=True)

    df_records = pd.DataFrame(records)
    print(f"[STAGE C2 COMPLETE] {window_name} took {time.perf_counter() - t0:.2f}s. "
          f"W1 invalid={w1_invalid}, timeout={w1_timeout}, nonfinite={w1_nonfinite}", flush=True)
    return dict(
        records=df_records,
        w1_invalid=w1_invalid,
        w1_timeout=w1_timeout,
        w1_nonfinite=w1_nonfinite,
    )


def run_stage_c3_freerun(window_name: str,
                         eval_first_rows: pd.DataFrame,
                         fitted_samplers: Dict[str, Any],
                         seed: int,
                         n_chains: int = N_CHAIN_REPS,
                         burn_in: int = BURN_IN_EPISODES,
                         collect: int = COLLECT_EPISODES,
                         max_bars: int = MAX_EPISODE_BARS,
                         symbols: Optional[List[str]] = None) -> Dict[str, Any]:
    """Execute Stage C3: Multi-Episode Free-Run."""
    if symbols is None:
        symbols = sorted(eval_first_rows["symbol"].unique())
    print(f"[STAGE C3] Multi-episode free-run on {window_name} "
          f"({len(symbols)} symbols x {n_chains} chains x {burn_in + collect} episodes)...", flush=True)
    t0 = time.perf_counter()

    chains_by_model: Dict[str, List[Dict[str, Any]]] = {"W0": [], "W1": [], "WT": []}
    w1_c3_invalid = 0
    w1_c3_timeout = 0
    w1_c3_nonfinite = 0

    for sym in symbols:
        sym_firsts = eval_first_rows[eval_first_rows["symbol"] == sym].reset_index(drop=True)
        n_sym = len(sym_firsts)
        if n_sym == 0:
            continue

        for rep_id in range(n_chains):
            seed_row = sym_firsts.iloc[rep_id % n_sym]
            seed_ep_id = seed_row["episode_id"]
            seed_state = build_observed_start_state(seed_row, seed_row["eps_R"])

            for model_key in ["W0", "W1", "WT"]:
                m_spec = WORLD_MODELS[model_key]
                term_s = fitted_samplers["term_samplers"][m_spec["terminal"]]
                trans_s = fitted_samplers["trans_samplers"][m_spec["transition"]]
                reset_s = fitted_samplers["reset_samplers"][m_spec["reset"]]

                chain_seed = int(hashlib.md5(f"{seed}_{window_name}_{sym}_{rep_id}_{model_key}".encode()).hexdigest()[:8], 16)
                chain_rng = np.random.default_rng(chain_seed)

                res = run_single_freerun_chain(
                    sym, seed_state, trans_s, term_s, reset_s, chain_rng,
                    burn_in=burn_in, collect=collect, max_bars=max_bars
                )
                res["window"] = window_name
                res["model"] = model_key
                res["rep_id"] = rep_id
                res["seed_episode_id"] = seed_ep_id
                res.setdefault("failure_episode", None)
                res.setdefault("failure_step", None)
                res.setdefault("failure_diagnostics", None)
                chains_by_model[model_key].append(res)

                if model_key == "W1":
                    if not res["completed"]:
                        w1_c3_invalid += 1
                    if res["violation"] == "EPISODE_TIMEOUT":
                        w1_c3_timeout += 1
                    if is_numeric_support_error(res["violation"]):
                        w1_c3_nonfinite += 1

            # Progress: every 4 chain reps per symbol
            if rep_id % 4 == 0:
                print(f"[C3 PROGRESS] {window_name} sym={sym} rep={rep_id}/{n_chains} "
                      f"elapsed={time.perf_counter() - t0:.1f}s", flush=True)

    print(f"[STAGE C3 COMPLETE] {window_name} took {time.perf_counter() - t0:.2f}s. "
          f"W1 incomplete_chains={w1_c3_invalid}, timeouts={w1_c3_timeout}, nonfinite={w1_c3_nonfinite}", flush=True)
    return dict(
        chains=chains_by_model,
        w1_invalid=w1_c3_invalid,
        w1_timeout=w1_c3_timeout,
        w1_nonfinite=w1_c3_nonfinite,
    )


def compute_replicate_discrepancies(chains_by_model: Dict[str, List[Dict[str, Any]]],
                                    obs_comparator: Dict[str, Any],
                                    n_chains: int = N_CHAIN_REPS) -> Dict[str, Any]:
    """Compute pooled replicate D_total and by-symbol D_total.

    Hardening (H): a replicate's D_total only counts toward the formal gate if ALL symbol
    chains for that (model, rep_id) completed. Incomplete chains' already-collected episodes
    are recorded (partial_D_total_diagnostic) but EXCLUDED from the gate and from the paired
    bootstrap (no survivor bias). By-symbol comparison only includes symbols whose chains are
    complete for BOTH compared models; the formal breadth denominator is unchanged.
    """
    symbols = sorted(obs_comparator["by_symbol"].keys())
    expected = len(symbols)

    rep_valid: Dict[str, List[bool]] = {"W0": [], "W1": [], "WT": []}
    by_symbol_complete: Dict[str, Dict[str, bool]] = {s: {"W0": False, "W1": False, "WT": False} for s in symbols}

    def _pack(all_eps):
        if len(all_eps) == 0:
            return dict(
                durations=np.empty(0, dtype=np.float64),
                endpoints=np.empty(0, dtype=np.int64),
                terminals={f: np.empty(0, dtype=np.float64) for f in TERMINAL_DYNAMIC_FIELDS},
                resets={f: np.empty(0, dtype=np.float64) for f in RESET_STRUCTURAL_FIELDS},
                gaps=np.empty(0, dtype=np.float64),
            )
        durs = np.array([ep["duration"] for ep in all_eps], dtype=np.float64)
        ends = np.array([ep["endpoint_mask"] for ep in all_eps], dtype=np.int64)
        terms = {f: np.array([ep["terminal_state"].get(f, np.nan) for ep in all_eps], dtype=np.float64)
                 for f in TERMINAL_DYNAMIC_FIELDS}
        resets = {f: np.array([ep["reset_state"].get(f, np.nan) for ep in all_eps], dtype=np.float64)
                  for f in RESET_STRUCTURAL_FIELDS}
        gaps = np.array([ep["gap"] for ep in all_eps], dtype=np.float64)
        return dict(durations=durs, endpoints=ends, terminals=terms, resets=resets, gaps=gaps)

    for model_key in ["W0", "W1", "WT"]:
        m_chains = chains_by_model[model_key]
        for rep_id in range(n_chains):
            rc = [c for c in m_chains if c["rep_id"] == rep_id]
            valid = (len(rc) == expected) and all(c["completed"] for c in rc)
            rep_valid[model_key].append(valid)
        for s in symbols:
            sc = [c for c in m_chains if c["symbol"] == s]
            by_symbol_complete[s][model_key] = (len(sc) == n_chains) and all(c["completed"] for c in sc)

    rep_metrics: Dict[str, List[Dict[str, Any]]] = {"W0": [], "W1": [], "WT": []}
    for model_key in ["W0", "W1", "WT"]:
        m_chains = chains_by_model[model_key]
        for rep_id in range(n_chains):
            rep_c = [c for c in m_chains if c["rep_id"] == rep_id]
            all_eps = [ep for c in rep_c for ep in c["episodes"]]
            gen_pack = _pack(all_eps)
            disc = compute_rollout_discrepancy(gen_pack, obs_comparator["pooled"])
            valid = rep_valid[model_key][rep_id]
            disc["valid_for_gate"] = bool(valid)
            disc["D_total_gate"] = float(disc["D_total"]) if valid else float("nan")
            disc["partial_D_total_diagnostic"] = float(disc["D_total"])
            rep_metrics[model_key].append(disc)

    by_symbol_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for s in symbols:
        by_symbol_metrics[s] = {}
        for model_key in ["W0", "W1", "WT"]:
            m_chains = [c for c in chains_by_model[model_key] if c["symbol"] == s]
            all_eps = [ep for c in m_chains for ep in c["episodes"]]
            gen_pack = _pack(all_eps)
            disc = compute_rollout_discrepancy(gen_pack, obs_comparator["by_symbol"][s])
            if model_key in ("W0", "W1"):
                cmp_valid = bool(by_symbol_complete[s]["W0"] and by_symbol_complete[s]["W1"])
            else:
                cmp_valid = bool(by_symbol_complete[s][model_key])
            disc["comparison_valid"] = cmp_valid
            by_symbol_metrics[s][model_key] = disc

    return dict(
        rep_metrics=rep_metrics,
        by_symbol_metrics=by_symbol_metrics,
        rep_valid=rep_valid,
        by_symbol_complete=by_symbol_complete,
        expected_symbols=expected,
    )


def evaluate_and_write_outputs(results_by_window: Dict[str, Any],
                               probe_by_window: Dict[str, Any],
                               comparators_by_window: Dict[str, Any],
                               boot_reps: int = BOOT_REPS) -> Dict[str, Any]:
    """Aggregate discrepancies, perform paired bootstrap, evaluate gates, and write 13 CSV/JSON outputs."""
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # 1. dynamic_pgm1c_support_probe.csv
    probe_rows = []
    for wname, pr in probe_by_window.items():
        probe_rows.append({
            "window": wname,
            "n_draws": pr["n_draws"],
            "n_transition_draws": pr["n_transition_draws"],
            "n_reset_draws": pr["n_reset_draws"],
            "total_transition_invalid": pr["total_transition_invalid"],
            "transition_invalid_rate": pr["transition_invalid_rate"],
            "total_reset_invalid": pr["total_reset_invalid"],
            "reset_invalid_rate": pr["reset_invalid_rate"],
            "invalid_draw_count": pr["invalid_draw_count"],
            "invalid_draw_rate": pr["invalid_draw_rate"],
        })
    pd.DataFrame(probe_rows).to_csv(OUT / f"{PREFIX}_support_probe.csv", index=False)

    # 2. dynamic_pgm1c_violation_counts.csv
    viol_rows = []
    for wname, pr in probe_by_window.items():
        for k, v in pr["transition_violations"].items():
            viol_rows.append({"window": wname, "stage": "C1_TRANSITION", "violation_type": k, "count": v})
        for k, v in pr["reset_violations"].items():
            viol_rows.append({"window": wname, "stage": "C1_RESET", "violation_type": k, "count": v})
    for wname, wdata in results_by_window.items():
        c2_recs = wdata["c2"]["records"]
        for viol, cnt in c2_recs["violation"].value_counts().items():
            if viol is not None:
                viol_rows.append({"window": wname, "stage": "C2_ROLLOUT", "violation_type": viol, "count": int(cnt)})
        for mkey, mchains in wdata["c3"]["chains"].items():
            for c in mchains:
                if c["violation"] is not None:
                    viol_rows.append({"window": wname, "stage": f"C3_CHAIN_{mkey}", "violation_type": c["violation"], "count": 1})
    pd.DataFrame(viol_rows).to_csv(OUT / f"{PREFIX}_violation_counts.csv", index=False)

    # 3. dynamic_pgm1c_seeded_episode_metrics.csv
    c2_rows = []
    for wname, wdata in results_by_window.items():
        c2_df = wdata["c2"]["records"]
        for mkey in ["W0", "W1", "WT"]:
            sub = c2_df[c2_df["model"] == mkey]
            n_tot = len(sub)
            n_term = int(np.sum(sub["status"] == "TERMINAL"))
            n_inv = int(np.sum(sub["status"] == "INVALID"))
            n_to = int(np.sum(sub["status"] == "TIMEOUT"))
            durs = sub["duration"].to_numpy(np.float64)
            c2_rows.append({
                "window": wname,
                "model": mkey,
                "n_episodes": len(sub["episode_id"].unique()),
                "n_reps": len(sub["rep"].unique()) if len(sub) else 0,
                "mean_duration": float(np.mean(durs)) if n_tot else np.nan,
                "std_duration": float(np.std(durs)) if n_tot else np.nan,
                "terminal_rate": float(n_term / n_tot) if n_tot else np.nan,
                "invalid_rate": float(n_inv / n_tot) if n_tot else np.nan,
                "timeout_rate": float(n_to / n_tot) if n_tot else np.nan,
            })
    pd.DataFrame(c2_rows).to_csv(OUT / f"{PREFIX}_seeded_episode_metrics.csv", index=False)

    # 4. dynamic_pgm1c_chain_metrics.csv
    c3_rows = []
    for wname, wdata in results_by_window.items():
        for mkey in ["W0", "W1", "WT"]:
            mchains = wdata["c3"]["chains"][mkey]
            n_tot = len(mchains)
            n_comp = int(sum(1 for c in mchains if c["completed"]))
            n_inv = int(sum(1 for c in mchains if not c["completed"] and c["violation"] != "EPISODE_TIMEOUT"))
            n_to = int(sum(1 for c in mchains if c["violation"] == "EPISODE_TIMEOUT"))
            n_nonfin = int(sum(1 for c in mchains if is_numeric_support_error(c["violation"])))
            tot_eps = sum(len(c["episodes"]) for c in mchains)
            c3_rows.append({
                "window": wname,
                "model": mkey,
                "n_chains": n_tot,
                "burn_in": BURN_IN_EPISODES,
                "collect": COLLECT_EPISODES,
                "completed_chains": n_comp,
                "total_episodes": tot_eps,
                "invalid_chains": n_inv,
                "timeout_chains": n_to,
                "nonfinite_chains": n_nonfin,
            })
    pd.DataFrame(c3_rows).to_csv(OUT / f"{PREFIX}_chain_metrics.csv", index=False)

    # 5. dynamic_pgm1c_chain_rep_metrics.csv
    rep_rows = []
    for wname, wdata in results_by_window.items():
        disc_data = wdata["discrepancies"]["rep_metrics"]
        for mkey in ["W0", "W1", "WT"]:
            for ridx, d in enumerate(disc_data[mkey]):
                row = {"window": wname, "model": mkey, "rep_id": ridx}
                row.update(d)
                rep_rows.append(row)
    pd.DataFrame(rep_rows).to_csv(OUT / f"{PREFIX}_chain_rep_metrics.csv", index=False)

    # 6. dynamic_pgm1c_discrepancy_modules.csv
    comp_rows = []
    w1_w0_passed_all = True
    wt_w1_passed_all = True
    w1_w0_support_complete_all = True
    wt_w1_support_complete_all = True

    for wname, wdata in results_by_window.items():
        disc_data = wdata["discrepancies"]["rep_metrics"]
        rep_valid = wdata["discrepancies"]["rep_valid"]
        by_symbol_complete = wdata["discrepancies"]["by_symbol_complete"]
        # Use the actual number of replicates produced by compute_replicate_discrepancies
        # (which respects the n_chains it was called with), not the module default.
        n_ch = len(rep_valid["W0"])

        # Paired-valid replicates: both models' rep must be complete.
        paired_10 = [r for r in range(n_ch) if rep_valid["W0"][r] and rep_valid["W1"][r]]
        paired_t1 = [r for r in range(n_ch) if rep_valid["W1"][r] and rep_valid["WT"][r]]
        support_10_complete = (len(paired_10) == n_ch)
        support_t1_complete = (len(paired_t1) == n_ch)
        if not support_10_complete:
            w1_w0_support_complete_all = False
        if not support_t1_complete:
            wt_w1_support_complete_all = False

        if support_10_complete:
            d_w0 = np.array([disc_data["W0"][r]["D_total"] for r in paired_10])
            d_w1 = np.array([disc_data["W1"][r]["D_total"] for r in paired_10])
            diff_10 = d_w1 - d_w0
            lo_10, hi_10, pt_10 = paired_bootstrap_replicates(diff_10, seed=20260915, reps=boot_reps)
        else:
            lo_10 = hi_10 = pt_10 = float("nan")

        if support_t1_complete:
            d_w1 = np.array([disc_data["W1"][r]["D_total"] for r in paired_t1])
            d_wt = np.array([disc_data["WT"][r]["D_total"] for r in paired_t1])
            diff_t1 = d_wt - d_w1
            lo_t1, hi_t1, pt_t1 = paired_bootstrap_replicates(diff_t1, seed=20260915, reps=boot_reps)
        else:
            lo_t1 = hi_t1 = pt_t1 = float("nan")

        # By-symbol breadth counts ONLY fully-complete symbols (denominator stays total).
        sym_disc = wdata["discrepancies"]["by_symbol_metrics"]
        syms = sorted(sym_disc.keys())
        fav_10 = sum(1 for s in syms if by_symbol_complete[s]["W0"] and by_symbol_complete[s]["W1"]
                     and sym_disc[s]["W1"]["D_total"] < sym_disc[s]["W0"]["D_total"])
        fav_t1 = sum(1 for s in syms if by_symbol_complete[s]["W1"] and by_symbol_complete[s]["WT"]
                     and sym_disc[s]["WT"]["D_total"] < sym_disc[s]["W1"]["D_total"])

        comp_rows.append({
            "window": wname, "comparison": "W1 - W0", "point": pt_10,
            "ci_lo": lo_10, "ci_hi": hi_10, "symbols_favored": fav_10,
            "total_symbols": len(syms),
            "p_support": bool(support_10_complete and hi_10 < 0 and fav_10 >= SYMBOL_BREADTH_MIN),
            "support_complete": support_10_complete, "paired_valid_reps": len(paired_10),
        })
        comp_rows.append({
            "window": wname, "comparison": "WT - W1", "point": pt_t1,
            "ci_lo": lo_t1, "ci_hi": hi_t1, "symbols_favored": fav_t1,
            "total_symbols": len(syms),
            "p_support": bool(support_t1_complete and hi_t1 < 0 and fav_t1 >= SYMBOL_BREADTH_MIN),
            "support_complete": support_t1_complete, "paired_valid_reps": len(paired_t1),
        })

        if not (support_10_complete and hi_10 < 0 and fav_10 >= SYMBOL_BREADTH_MIN):
            w1_w0_passed_all = False
        if not (support_t1_complete and hi_t1 < 0 and fav_t1 >= SYMBOL_BREADTH_MIN):
            wt_w1_passed_all = False

    pd.DataFrame(comp_rows).to_csv(OUT / f"{PREFIX}_discrepancy_modules.csv", index=False)

    # 7. dynamic_pgm1c_by_symbol.csv
    by_sym_rows = []
    for wname, wdata in results_by_window.items():
        sym_disc = wdata["discrepancies"]["by_symbol_metrics"]
        for s in sorted(sym_disc.keys()):
            for mkey in ["W0", "W1", "WT"]:
                r = {"window": wname, "symbol": s, "model": mkey}
                r.update(sym_disc[s][mkey])
                by_sym_rows.append(r)
    pd.DataFrame(by_sym_rows).to_csv(OUT / f"{PREFIX}_by_symbol.csv", index=False)

    # 8. dynamic_pgm1c_duration_diagnostics.csv
    dur_rows = []
    for wname, wdata in results_by_window.items():
        obs_dur = comparators_by_window[wname]["pooled"]["durations"]
        dur_rows.append({
            "window": wname, "source": "OBSERVED",
            "mean": float(np.mean(obs_dur)) if len(obs_dur) else np.nan,
            "std": float(np.std(obs_dur)) if len(obs_dur) else np.nan,
            "median": float(np.median(obs_dur)) if len(obs_dur) else np.nan,
            "q05": float(np.percentile(obs_dur, 5)) if len(obs_dur) else np.nan,
            "q95": float(np.percentile(obs_dur, 95)) if len(obs_dur) else np.nan,
            "max": float(np.max(obs_dur)) if len(obs_dur) else np.nan,
        })
        for mkey in ["W0", "W1", "WT"]:
            mchains = wdata["c3"]["chains"][mkey]
            durs = np.array([ep["duration"] for c in mchains for ep in c["episodes"]], dtype=np.float64)
            if len(durs):
                dur_rows.append({
                    "window": wname, "source": mkey,
                    "mean": float(np.mean(durs)), "std": float(np.std(durs)),
                    "median": float(np.median(durs)), "q05": float(np.percentile(durs, 5)),
                    "q95": float(np.percentile(durs, 95)), "max": float(np.max(durs)),
                })
    pd.DataFrame(dur_rows).to_csv(OUT / f"{PREFIX}_duration_diagnostics.csv", index=False)

    # 9. dynamic_pgm1c_endpoint_diagnostics.csv
    end_rows = []
    for wname, wdata in results_by_window.items():
        obs_end = comparators_by_window[wname]["pooled"]["endpoints"]
        for m in range(1, 16):
            p_obs = float(np.mean(obs_end == m)) if len(obs_end) else 0.0
            end_rows.append({"window": wname, "source": "OBSERVED", "mask": m, "probability": p_obs})
        for mkey in ["W0", "W1", "WT"]:
            mchains = wdata["c3"]["chains"][mkey]
            ends = np.array([ep["endpoint_mask"] for c in mchains for ep in c["episodes"]], dtype=np.int64)
            for m in range(1, 16):
                p_m = float(np.mean(ends == m)) if len(ends) else 0.0
                end_rows.append({"window": wname, "source": mkey, "mask": m, "probability": p_m})
    pd.DataFrame(end_rows).to_csv(OUT / f"{PREFIX}_endpoint_diagnostics.csv", index=False)

    # 10. dynamic_pgm1c_state_diagnostics.csv
    state_rows = []
    for wname, wdata in results_by_window.items():
        obs_term = comparators_by_window[wname]["pooled"]["terminals"]
        for fld in TERMINAL_DYNAMIC_FIELDS:
            arr = obs_term.get(fld, np.empty(0))
            state_rows.append({
                "window": wname, "source": "OBSERVED", "field": fld,
                "mean": float(np.mean(arr)) if len(arr) else np.nan,
                "std": float(np.std(arr)) if len(arr) else np.nan,
            })
        for mkey in ["W0", "W1", "WT"]:
            mchains = wdata["c3"]["chains"][mkey]
            for fld in TERMINAL_DYNAMIC_FIELDS:
                arr = np.array([ep["terminal_state"].get(fld, np.nan) for c in mchains for ep in c["episodes"]], dtype=np.float64)
                arr = arr[np.isfinite(arr)]
                state_rows.append({
                    "window": wname, "source": mkey, "field": fld,
                    "mean": float(np.mean(arr)) if len(arr) else np.nan,
                    "std": float(np.std(arr)) if len(arr) else np.nan,
                })
    pd.DataFrame(state_rows).to_csv(OUT / f"{PREFIX}_state_diagnostics.csv", index=False)

    # 11. dynamic_pgm1c_reset_diagnostics.csv
    reset_diag_rows = []
    for wname, wdata in results_by_window.items():
        obs_rst = comparators_by_window[wname]["pooled"]["resets"]
        for fld in RESET_STRUCTURAL_FIELDS:
            arr = obs_rst.get(fld, np.empty(0))
            reset_diag_rows.append({
                "window": wname, "source": "OBSERVED", "field": fld,
                "mean": float(np.mean(arr)) if len(arr) else np.nan,
                "std": float(np.std(arr)) if len(arr) else np.nan,
            })
        for mkey in ["W0", "W1", "WT"]:
            mchains = wdata["c3"]["chains"][mkey]
            for fld in RESET_STRUCTURAL_FIELDS:
                arr = np.array([ep["reset_state"].get(fld, np.nan) for c in mchains for ep in c["episodes"]], dtype=np.float64)
                arr = arr[np.isfinite(arr)]
                reset_diag_rows.append({
                    "window": wname, "source": mkey, "field": fld,
                    "mean": float(np.mean(arr)) if len(arr) else np.nan,
                    "std": float(np.std(arr)) if len(arr) else np.nan,
                })
    pd.DataFrame(reset_diag_rows).to_csv(OUT / f"{PREFIX}_reset_diagnostics.csv", index=False)

    # 12. dynamic_pgm1c_gap_diagnostics.csv
    gap_rows = []
    for wname, wdata in results_by_window.items():
        obs_gap = comparators_by_window[wname]["pooled"]["gaps"]
        p_occ = float(np.mean(obs_gap > 0)) if len(obs_gap) else 0.0
        pos = obs_gap[obs_gap > 0]
        gap_rows.append({
            "window": wname, "source": "OBSERVED",
            "gap_occurrence_prob": p_occ,
            "mean_positive_gap": float(np.mean(pos)) if len(pos) else 0.0,
            "max_gap": float(np.max(obs_gap)) if len(obs_gap) else 0.0,
        })
        for mkey in ["W0", "W1", "WT"]:
            mchains = wdata["c3"]["chains"][mkey]
            gaps = np.array([ep["gap"] for c in mchains for ep in c["episodes"]], dtype=np.float64)
            p_g = float(np.mean(gaps > 0)) if len(gaps) else 0.0
            pos_g = gaps[gaps > 0]
            gap_rows.append({
                "window": wname, "source": mkey,
                "gap_occurrence_prob": p_g,
                "mean_positive_gap": float(np.mean(pos_g)) if len(pos_g) else 0.0,
                "max_gap": float(np.max(gaps)) if len(gaps) else 0.0,
            })
    pd.DataFrame(gap_rows).to_csv(OUT / f"{PREFIX}_gap_diagnostics.csv", index=False)

    # 13. Summary JSON and Gates
    gate0_pass = True
    for wname, wdata in results_by_window.items():
        if wdata["c2"]["w1_invalid"] > 0 or wdata["c2"]["w1_timeout"] > 0 or wdata["c2"]["w1_nonfinite"] > 0:
            gate0_pass = False
        if wdata["c3"]["w1_invalid"] > 0 or wdata["c3"]["w1_timeout"] > 0 or wdata["c3"]["w1_nonfinite"] > 0:
            gate0_pass = False

    gate0_dec = "FREE_RUN_SUPPORT_CLOSED" if gate0_pass else "FREE_RUN_SUPPORT_NOT_CLOSED"
    gate1_dec = ("ROLLOUT_COMPARISON_SUPPORT_INCOMPLETE" if not w1_w0_support_complete_all
                 else "ROLLOUT_PHI_RESET_SUPPORTED" if w1_w0_passed_all
                 else "ONE_STEP_GAINS_DO_NOT_SURVIVE_ROLLOUT")
    gate_mem_dec = ("TERMINAL_MEMORY_COMPARISON_SUPPORT_INCOMPLETE" if not wt_w1_support_complete_all
                   else "TERMINAL_MEMORY_ROLLOUT_USEFUL" if wt_w1_passed_all
                   else "TERMINAL_MEMORY_NOT_PROMOTED")

    summary = dict(
        experiment="DYNAMIC-PGM-1C Free-Run Rollout Closure",
        parent_commit=BASE_SHA,
        gate0=gate0_dec,
        gate1=gate1_dec,
        gate_terminal_memory=gate_mem_dec,
        support_closed=gate0_pass,
        comparisons=comp_rows,
        probe_summary=probe_rows,
        elapsed_seconds=round(time.perf_counter() - t0, 2),
    )
    (OUT / f"{PREFIX}_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"[EVALUATION COMPLETE] 13 output files generated in {OUT}. "
          f"Gate0={gate0_dec}, Gate1={gate1_dec}, GateMem={gate_mem_dec}", flush=True)
    return summary


def run_dynamic_pgm1c_smoke_test(obs_sample_path: Optional[Path] = None,
                                 transitions_path: Optional[Path] = None,
                                 ep_meta_path: Optional[Path] = None) -> Dict[str, Any]:
    """Execute tiny end-to-end stochastic smoke test through full execution chain."""
    if obs_sample_path is None:
        obs_sample_path = SAMPLE_PATH
    if transitions_path is None:
        transitions_path = TRANSITION_SAMPLE_PATH
    if ep_meta_path is None:
        ep_meta_path = EP_META_PATH

    print("==================================================", flush=True)
    print("STAGE C0-C3: TINY END-TO-END STOCHASTIC SMOKE TEST", flush=True)
    print("==================================================", flush=True)
    t0 = time.perf_counter()

    obs = pd.read_parquet(obs_sample_path)
    ep_meta = pd.read_parquet(ep_meta_path)
    meta_slim = ep_meta[["symbol", "start_bar", "start_upper_price", "start_lower_price"]].drop_duplicates()
    obs = obs.merge(meta_slim, on=["symbol", "start_bar"], how="left")
    span = obs["start_upper_price"].to_numpy(np.float64) - obs["start_lower_price"].to_numpy(np.float64)
    atr0 = span / obs["start_width_R"].to_numpy(np.float64)
    obs["eps_R"] = 1e-9 / atr0

    pair, first = exp1b.build_reset_pairs(obs)

    w = WINDOWS[0]  # Window A
    wname = w["name"]
    fitted = fit_samplers_for_window(w, obs_sample_path, transitions_path)

    ev_obs = obs[obs["block"] == w["eval"]].reset_index(drop=True)
    ev_pairs = pair[pair["block"] == w["eval"]].reset_index(drop=True)
    ev_firsts = ev_obs[ev_obs["bar_t"] == ev_obs["start_bar"]].reset_index(drop=True)

    # 1. C1 probe (100 draws)
    c1_probe = run_stage_c1_probe(
        wname,
        fitted["trans_samplers"]["MC_STATE_CURREENCODING"],
        fitted["reset_samplers"]["R1_STATE_PHI"],
        ev_obs, ev_pairs, seed=20260915, n_draws=100
    )
    print(f"[SMOKE C1] 100 draws: invalid={c1_probe['invalid_draw_count']}", flush=True)

    # 2. C2 rollout (2 starts x 2 reps)
    c2_starts = ev_firsts.head(2).copy().reset_index(drop=True)
    c2_res = run_stage_c2_rollout(wname, c2_starts, fitted, seed=20260915, n_reps=2, max_bars=64)
    print(f"[SMOKE C2] {len(c2_res['records'])} episode rollouts completed", flush=True)

    # 3. C3 free-run (1 symbol x 2 chains x burn1 x collect3)
    sym = "AG"
    c3_res = run_stage_c3_freerun(
        wname, ev_firsts, fitted, seed=20260915,
        n_chains=2, burn_in=1, collect=3, max_bars=64, symbols=[sym]
    )
    print(f"[SMOKE C3] {len(c3_res['chains']['W1'])} chains completed for W1", flush=True)

    # 4. Discrepancy & Evaluation
    obs_comps = {wname: build_observed_comparators(ev_obs, ev_pairs)}
    disc_res = compute_replicate_discrepancies(c3_res["chains"], obs_comps[wname], n_chains=2)

    results_by_window = {
        wname: dict(c2=c2_res, c3=c3_res, discrepancies=disc_res)
    }
    probe_by_window = {wname: c1_probe}

    summary = evaluate_and_write_outputs(
        results_by_window, probe_by_window, obs_comps, boot_reps=100
    )
    elapsed = time.perf_counter() - t0
    print(f"[SMOKE COMPLETE] Entire end-to-end chain ran successfully in {elapsed:.2f}s!", flush=True)
    return summary


def run_dynamic_pgm1c_full(obs_sample_path: Path, transitions_path: Path,
                           ep_meta_path: Path) -> Dict[str, Any]:
    """Execute formal Stage C1..C3 simulation, discrepancy evaluation, and gate resolution."""
    print("==================================================", flush=True)
    print("STAGE C1-C3: FORMAL FREE-RUN ROLLOUT CLOSURE", flush=True)
    print("==================================================", flush=True)
    t0 = time.perf_counter()

    obs = pd.read_parquet(obs_sample_path)
    ep_meta = pd.read_parquet(ep_meta_path)
    meta_slim = ep_meta[["symbol", "start_bar", "start_upper_price", "start_lower_price"]].drop_duplicates()
    obs = obs.merge(meta_slim, on=["symbol", "start_bar"], how="left")
    span = obs["start_upper_price"].to_numpy(np.float64) - obs["start_lower_price"].to_numpy(np.float64)
    atr0 = span / obs["start_width_R"].to_numpy(np.float64)
    obs["eps_R"] = 1e-9 / atr0

    pair, first = exp1b.build_reset_pairs(obs)

    results_by_window: Dict[str, Any] = {}
    probe_by_window: Dict[str, Any] = {}
    comparators_by_window: Dict[str, Any] = {}

    for w in WINDOWS:
        wname = w["name"]
        fitted = fit_samplers_for_window(w, obs_sample_path, transitions_path)

        ev_obs = obs[obs["block"] == w["eval"]].reset_index(drop=True)
        ev_pairs = pair[pair["block"] == w["eval"]].reset_index(drop=True)
        ev_firsts = ev_obs[ev_obs["bar_t"] == ev_obs["start_bar"]].reset_index(drop=True)

        obs_comp = build_observed_comparators(ev_obs, ev_pairs)
        comparators_by_window[wname] = obs_comp

        # Stage C1
        c1_probe = run_stage_c1_probe(
            wname,
            fitted["trans_samplers"]["MC_STATE_CURREENCODING"],
            fitted["reset_samplers"]["R1_STATE_PHI"],
            ev_obs, ev_pairs, seed=20260915
        )
        probe_by_window[wname] = c1_probe

        # Stage C2
        c2_res = run_stage_c2_rollout(wname, ev_firsts, fitted, seed=20260915)

        # Stage C3
        c3_res = run_stage_c3_freerun(wname, ev_firsts, fitted, seed=20260915)

        # Discrepancies
        disc_res = compute_replicate_discrepancies(c3_res["chains"], obs_comp)

        results_by_window[wname] = dict(
            c2=c2_res, c3=c3_res, discrepancies=disc_res
        )

    summary = evaluate_and_write_outputs(
        results_by_window, probe_by_window, comparators_by_window
    )
    print(f"[DYNAMIC-PGM-1C COMPLETE] Full experiment completed in {time.perf_counter() - t0:.2f}s", flush=True)
    return summary


def run_stability_probe(obs_sample_path: Path, transitions_path: Path, ep_meta_path: Path) -> Dict[str, Any]:
    """Closed-loop stability probe (K/L). Diagnostic ONLY -- does NOT produce a formal 1C verdict.

    Uses REAL recursive rollout (terminal -> transition -> state update) for W1 only, over both
    windows, with up to 64 observed starts per symbol, 1 rep, and MAX_EPISODE_BARS=512. Each
    rollout records its failure mode via the `violation` field and the training-domain
    extrapolation diagnostics, so we can later decide whether the issue is numeric
    representation, conditional-head extrapolation, or state-dynamics support closure.
    """
    print("==================================================", flush=True)
    print("CLOSED-LOOP STABILITY PROBE (diagnostic only)", flush=True)
    print("==================================================", flush=True)
    PROBE_REPS = 1
    PROBE_STARTS_PER_SYMBOL = 64
    probe_rows: List[Dict[str, Any]] = []

    for w in WINDOWS:
        wname = w["name"]
        print(f"[STABILITY PROBE] window={wname} ...", flush=True)
        fitted = fit_samplers_for_window(w, obs_sample_path, transitions_path)
        term_s = fitted["term_samplers"]["T0_STATE_AVAIL"]
        trans_s = fitted["trans_samplers"]["MC_STATE_CURREENCODING"]
        rng = np.random.default_rng(20260915)

        obs = pd.read_parquet(obs_sample_path)
        ep_meta = pd.read_parquet(ep_meta_path)
        meta_slim = ep_meta[["symbol", "start_bar", "start_upper_price", "start_lower_price"]].drop_duplicates()
        obs = obs.merge(meta_slim, on=["symbol", "start_bar"], how="left")
        span = obs["start_upper_price"].to_numpy(np.float64) - obs["start_lower_price"].to_numpy(np.float64)
        atr0 = span / obs["start_width_R"].to_numpy(np.float64)
        obs["eps_R"] = 1e-9 / atr0
        pair, first = exp1b.build_reset_pairs(obs)
        ev_obs = obs[obs["block"] == w["eval"]].reset_index(drop=True)
        ev_firsts = ev_obs[ev_obs["bar_t"] == ev_obs["start_bar"]].reset_index(drop=True)

        syms = sorted(ev_firsts["symbol"].unique())
        for sym in syms:
            sym_firsts = ev_firsts[ev_firsts["symbol"] == sym].reset_index(drop=True).head(PROBE_STARTS_PER_SYMBOL)
            for fr_idx in range(len(sym_firsts)):
                fr = sym_firsts.iloc[fr_idx]
                st = build_observed_start_state(fr, fr["eps_R"])
                ep_res = run_single_episode_rollout(st, trans_s, term_s, rng, MAX_EPISODE_BARS)
                probe_rows.append(dict(
                    window=wname,
                    symbol=sym,
                    episode_id=str(fr["episode_id"]),
                    status=ep_res["status"],
                    duration=ep_res["duration"],
                    failure_step=ep_res.get("failure_step"),
                    violation=ep_res["violation"],
                    max_abs_design_seen=ep_res.get("max_abs_design_seen"),
                    max_extrapolation_ratio_seen=ep_res.get("max_extrapolation_ratio_seen"),
                    first_outside_train_step=ep_res.get("first_outside_train_step"),
                    worst_feature_name=ep_res.get("worst_feature_name"),
                    worst_feature_value=ep_res.get("worst_feature_value"),
                    train_feature_min=ep_res.get("train_feature_min"),
                    train_feature_max=ep_res.get("train_feature_max"),
                ))

    probe_df = pd.DataFrame(probe_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    probe_df.to_csv(OUT / f"{PREFIX}_stability_probe.csv", index=False)

    def _p(arr: "pd.Series", q: float):
        return float(np.percentile(arr, q)) if len(arr) else None

    summ: Dict[str, Any] = {}
    for w in WINDOWS:
        wname = w["name"]
        sub = probe_df[probe_df["window"] == wname]
        n = len(sub)
        invalid = sub[sub["status"] == "INVALID"]
        timeout = sub[sub["status"] == "TIMEOUT"]
        terminal = sub[sub["status"] == "TERMINAL"]
        viol_breakdown = {k: int(v) for k, v in invalid["violation"].value_counts().to_dict().items()} if len(invalid) else {}
        fs = invalid["failure_step"].dropna().astype(float)
        mer = sub["max_extrapolation_ratio_seen"].dropna().astype(float)
        fos = sub["first_outside_train_step"].dropna().astype(float)
        summ[wname] = dict(
            n_rollouts=n,
            terminal=int(len(terminal)),
            timeout=int(len(timeout)),
            invalid=int(len(invalid)),
            invalid_rate=(float(len(invalid) / n) if n else 0.0),
            violation_breakdown=viol_breakdown,
            failure_step=dict(median=_p(fs, 50), p10=_p(fs, 10), p50=_p(fs, 50), p90=_p(fs, 90)),
            max_extrapolation_ratio=dict(
                median=_p(mer, 50), p90=_p(mer, 90), p99=_p(mer, 99),
                max=float(mer.max()) if len(mer) else None,
            ),
            first_outside_train_step_distribution=dict(
                median=_p(fos, 50), p10=_p(fos, 10), p90=_p(fos, 90),
            ),
        )
    (OUT / f"{PREFIX}_stability_probe_summary.json").write_text(json.dumps(summ, indent=2, default=str))
    print(f"[STABILITY PROBE COMPLETE] wrote {len(probe_df)} rows to {OUT}", flush=True)
    return summ


def main():
    parser = argparse.ArgumentParser(description="DYNAMIC-PGM-1C Free-Run Rollout Closure")
    parser.add_argument("--audit-only", action="store_true", help="Run Stage C0 parity and closure audits only.")
    parser.add_argument("--smoke-test", action="store_true", help="Run tiny end-to-end stochastic smoke test through full pipeline.")
    parser.add_argument("--stability-probe", action="store_true",
                        help="Run closed-loop stability probe (diagnostic only, no formal gate).")
    args = parser.parse_args()

    obs_sample_path = CACHE / "dynamic_pgm1b_sample.parquet"
    transitions_path = CACHE / "dynamic_pgm1a2c_transitions.parquet"
    ep_meta_path = CACHE / "episode_repl0_through_tb3.parquet"

    # Stage C0 is mandatory for both audit-only and full run
    audit_res = run_dynamic_pgm1c_audit(obs_sample_path, transitions_path, ep_meta_path)

    if args.audit_only or os.environ.get("DYNAMIC_PGM1C_AUDIT_ONLY") == "1":
        print("[AUDIT-ONLY] Complete. Stopping before any stochastic rollout simulation.", flush=True)
        return

    if args.stability_probe or os.environ.get("DYNAMIC_PGM1C_STABILITY_PROBE") == "1":
        run_stability_probe(obs_sample_path, transitions_path, ep_meta_path)
        print("[STABILITY-PROBE] Complete. Diagnostic only -- no formal 1C verdict produced.", flush=True)
        return

    if args.smoke_test or os.environ.get("DYNAMIC_PGM1C_SMOKE_TEST") == "1":
        smoke_res = run_dynamic_pgm1c_smoke_test(obs_sample_path, transitions_path, ep_meta_path)
        print("[SMOKE-TEST] Complete. Stopping before formal full simulation.", flush=True)
        return

    # Formal full experiment execution
    run_dynamic_pgm1c_full(obs_sample_path, transitions_path, ep_meta_path)


if __name__ == "__main__":
    main()
