"""
experiment_pgm_native0e_consensus_acceleration_v1.py
====================================================

PGM-NATIVE-0E -- Directional Consensus x Acceleration Interaction

SINGLE PRE-REGISTERED QUESTION
------------------------------
Does the economic effect of directional acceleration change sign as the
pre-existing directional consensus state (C_{t-1}) rises?

    "When the market has just reversed and participants still doubt, a fresh
     acceleration means 'the market is starting to believe' -> continuation is
     more likely good. When most participants already fully believe, another
     acceleration may be the last players piling in -> exhaustion."

We do NOT name it fear/greed. We define an observable PGM structural proxy:

    C_{t-1} = Directional Consensus State  (how accepted the current direction is)

STRICT TIME CONTRACT (causal)
-----------------------------
    C_{t-1} -> A_t -> pi_{t+1}

  - t-1 : what psychological / structural environment players were already in
  - t   : did the market suddenly accelerate along the current direction
  - t+1 : is this acceleration good or bad for the next continuation trade

This avoids the circular "the current bar rose a lot -> the position is late ->
therefore it is also high-consensus" explanation.

REUSE
-----
  d0  = experiment_pgm_native0d_acceleration_terminal_outcome_v1  (0D acceleration, owner, bootstrap, metrics)
  n0c = experiment_pgm_native0c_state_augmentation_v1            (U/E columns, hashes, day bootstrap)
  pgm = experiment_dynamic_pgm1c_free_run_rollout_v1            (WINDOWS, samplers, T2_NUM, CAT)
  pm  = experiment_path0_episode_path_memory_v1                (fixed L2 Logistic pipeline)
  n0a = experiment_pgm_native0a_one_step_alpha_v1              (universe, audit, atr0 parity)

ROUND 1 (this file): architecture + audit + smoke ONLY.
`--full-exploratory` is HARD-BLOCKED.

ROUND 2: the formal runner (`run_full_exploratory`), the authorization token gate,
the 8 frozen artifacts (CSV/JSON), the in-memory validator and the disk-parity
validator are implemented. The actual `--full-exploratory` run is still gated by
AUTHORIZE_PGM_NATIVE0E_FULL_EXPLORATORY=1 and is NOT executed during Round 2.
"""

from __future__ import annotations

import argparse
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

import research.liquidity_oracle_atlas.experiment_pgm_native0a_one_step_alpha_v1 as n0a
import research.liquidity_oracle_atlas.experiment_pgm_native0c_state_augmentation_v1 as n0c
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1c_free_run_rollout_v1 as pgm
import research.liquidity_oracle_atlas.experiment_path0_episode_path_memory_v1 as pm
import research.liquidity_oracle_atlas.experiment_pgm_native0d_acceleration_terminal_outcome_v1 as d0


# ===========================================================================
# Governance constants
# ===========================================================================
BASE_SHA = "f76b10e724848bfd34076652bf3c76f4cd1ef696"
EXPERIMENT_NAME = "PGM-NATIVE-0E -- Directional Consensus x Acceleration Interaction"
EXPERIMENT_SCOPE = "EXPLORATORY_PGM_CONSENSUS_ACCELERATION_INTERACTION_ON_TB1_TB2_TB3"
PREFIX = "pgm_native0e1"

ALLOWED_BLOCKS = ["TB1", "TB2", "TB3"]
TB2_BLOCK = "TB2"
TB3_BLOCK = "TB3"
# Formal evaluation windows (TB3 only decides the verdict).
EVAL_BLOCKS = [TB2_BLOCK, TB3_BLOCK]

EPS = 1e-9
CONSENSUS_LOOKBACK = 5
CONSENSUS_MIN_PRIOR = 3
MIN_CELL_N = 500
PRIMARY_CELLS = ["LOW_ACCEL", "LOW_OFF", "HIGH_ACCEL", "HIGH_OFF"]

# a_dir_accel_1 is inherited verbatim from 0D; it must NOT be redefined here.
PRIMARY_ACCELERATION_COL = "a_dir_accel_1"
A_COLS = d0.A_COLS                       # exact 8-feature frozen block, reused

PRIMARY_COST_ATR0 = 0.01
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20260916
SMOKE_BOOTSTRAP_N = 200
SMOKE_EVAL_CAP = 2048
# Smoke trains predictive models on a fixed deterministic evenly-spaced subset of the
# train set (wiring-only; rank-map + tercile still use the FULL train). Formal path
# keeps model_train_cap=None (full train).
SMOKE_MODEL_TRAIN_CAP = 8192
CLUSTER_OWNER = "entry_day"

# ---------------------------------------------------------------------------
# Round 2: formal artifact contract
# ---------------------------------------------------------------------------
# Environment token that authorizes the ONE-SHOT formal full run. Only the exact
# value "1" authorizes; anything else (unset / "0" / other) hard-stops.
AUTHORIZE_ENV = "AUTHORIZE_PGM_NATIVE0E_FULL_EXPLORATORY"

# The 8 frozen formal artifacts (no more, no fewer).
ARTIFACT_FILES = [
    "pgm_native0e1_primary_cells.csv",
    "pgm_native0e1_primary_bootstrap.csv",
    "pgm_native0e1_predictive_metrics.csv",
    "pgm_native0e1_predictive_bootstrap.csv",
    "pgm_native0e1_psych_gate.csv",
    "pgm_native0e1_component_diagnostics.csv",
    "pgm_native0e1_age_zero_diagnostics.csv",
    "pgm_native0e1_formal_summary.json",
]

OUT_DIR = _REPO_ROOT / "research" / "analysis_results" / "local_liquidity_transition_v0"

# The 4 interpretable psychology primitives (each from t-1 or earlier only)
CONSENSUS_RAW = [
    "c_position",
    "c_path_agreement",
    "c_boundary_freshness",
    "c_boundary_density",
]

VERDICT = {
    "SIGN_SWITCH": "PGM_CONSENSUS_ACCELERATION_SIGN_SWITCH_SUPPORTED_EXPLORATORY",
    "STATE_DEP": "PGM_CONSENSUS_ACCELERATION_STATE_DEPENDENCE_SUPPORTED_EXPLORATORY",
    "NOT_SUPPORTED": "PGM_CONSENSUS_ACCELERATION_NOT_SUPPORTED_EXPLORATORY",
}

# Forbidden tokens that would indicate a future-leaking implementation.
FUTURE_TOKENS = ["shift(-1)", "future_", "next_", "remaining", "final_"]

# Predictive falsification model spec (fixed, no tuning)
PRED_NUM_BASE = (list(pgm.T2_NUM) + list(n0c.U_COLS) + list(n0c.E_COLS)
                 + ["score_mu", "abs_score_mu", "consensus_score", "accel_plus"])
PRED_CAT = ["prev_event_mask", "symbol"]


# ===========================================================================
# Consensus primitives -- STRICTLY t-1 (or earlier) PGM state
# ===========================================================================
def add_consensus_primitives(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the 4 interpretable consensus primitives, all using t-1 PGM state.

    C_{t-1} must be a function of bar t-1 (or earlier) ONLY. The current bar's
    own state is never used to define the psychological environment.
    """
    x = df.sort_values(["symbol", "episode_id", "bar_t"], kind="stable").copy()
    g = x.groupby(["symbol", "episode_id"], sort=False, group_keys=False)

    # All t-1 features in ONE group-aware shift (groupby().shift is C-vectorized and
    # restarts at each (symbol, episode_id) boundary -- no per-group Python callback).
    _lagcols = [
        "score_mu", "cur_up_distance_R", "cur_down_distance_R",
        "upper_newest_log_age", "lower_newest_log_age",
        "upper_n_active_identities", "lower_n_active_identities",
    ]
    _lag1 = g[_lagcols].shift(1)
    x["lag1_score_mu"] = _lag1["score_mu"].to_numpy(float)
    d_now = np.sign(x["score_mu"].to_numpy(float))
    d_prev = np.sign(x["lag1_score_mu"].to_numpy(float))

    # direction_stable: same direction now as at t-1, and t-1 was not undecided
    x["direction_stable"] = (
        np.isfinite(d_prev) & (d_now != 0) & (d_prev == d_now)
    )

    up = _lag1["cur_up_distance_R"].to_numpy(float)
    dn = _lag1["cur_down_distance_R"].to_numpy(float)
    up_age = _lag1["upper_newest_log_age"].to_numpy(float)
    dn_age = _lag1["lower_newest_log_age"].to_numpy(float)
    up_n = _lag1["upper_n_active_identities"].to_numpy(float)
    dn_n = _lag1["lower_n_active_identities"].to_numpy(float)

    front = np.where(d_prev > 0, up, dn)
    back = np.where(d_prev > 0, dn, up)
    front_age = np.where(d_prev > 0, up_age, dn_age)
    back_age = np.where(d_prev > 0, dn_age, up_age)
    front_n = np.where(d_prev > 0, up_n, dn_n)
    back_n = np.where(d_prev > 0, dn_n, up_n)

    # 1. Liquidity position: how far the trend has walked toward the front liquidity
    x["c_position"] = (back - front) / (back + front + EPS)

    # 2. Path agreement: did the recent path mostly walk along the current direction?
    # Vectorized fixed 5-lag sum. groupby().shift(k) restarts at group boundaries;
    # np.nansum over the 5 lags with min_periods==CONSENSUS_MIN_PRIOR replicates
    # shift(1).rolling(CONSENSUS_LOOKBACK, min_periods=CONSENSUS_MIN_PRIOR).sum()
    # EXACTLY (verified parity), but with zero per-group Python callbacks.
    gshift = g["path_last_return_R"].shift
    _lags = np.column_stack([
        gshift(1).to_numpy(float), gshift(2).to_numpy(float), gshift(3).to_numpy(float),
        gshift(4).to_numpy(float), gshift(5).to_numpy(float),
    ])
    _valid = np.isfinite(_lags).sum(axis=1)
    prior_sum = np.nansum(_lags, axis=1)
    prior_abs = np.nansum(np.abs(_lags), axis=1)
    prior_sum[_valid < CONSENSUS_MIN_PRIOR] = np.nan
    prior_abs[_valid < CONSENSUS_MIN_PRIOR] = np.nan
    x["c_path_agreement"] = d_prev * prior_sum / (prior_abs + EPS)

    # 3. Boundary freshness: is the back boundary newer than the front boundary?
    x["c_boundary_freshness"] = front_age - back_age

    # 4. Boundary density: structural memory accumulated behind the current direction
    x["c_boundary_density"] = back_n - front_n

    x["consensus_eligible"] = x["direction_stable"] & x[CONSENSUS_RAW].notna().all(axis=1)
    return x


def fit_rank_maps(train: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Train-only empirical CDF references (no supervision on outcome)."""
    maps = {}
    for col in CONSENSUS_RAW:
        v = train[col].to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            raise SystemExit(f"STOP_PGM_NATIVE0E_EMPTY_RANK_MAP:{col}")
        maps[col] = np.sort(v)
    return maps


def empirical_rank(values, ref: np.ndarray) -> np.ndarray:
    """Empirical rank in (-1, 1). NaN input -> NaN output (never mapped to a high rank)."""
    values = np.asarray(values, float)
    out = np.full(len(values), np.nan)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    k = np.searchsorted(ref, values[finite], side="right")
    u = (k + 0.5) / (len(ref) + 1.0)
    out[finite] = 2.0 * u - 1.0
    return out


def attach_consensus_score(df: pd.DataFrame, rank_maps: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Mean of the 4 empirical ranks -> consensus_score in [-1, 1]. Equal weight.

    NaN contract: any component rank NaN -> consensus_score NaN (no skipna).
    """
    x = df.copy()
    rank_cols = []
    for col in CONSENSUS_RAW:
        rc = f"{col}_rank"
        x[rc] = empirical_rank(x[col].to_numpy(float), rank_maps[col])
        rank_cols.append(rc)
    rank_matrix = x[rank_cols].to_numpy(float)
    x["consensus_score"] = np.mean(rank_matrix, axis=1)
    return x


def compute_consensus_terciles(train_score: np.ndarray) -> Tuple[float, float]:
    """Train-only terciles; degenerate (q_low >= q_high) is a hard STOP."""
    v = np.asarray(train_score, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        raise SystemExit("STOP_PGM_NATIVE0E_EMPTY_TRAIN_SCORE")
    ql = float(np.quantile(v, 1.0 / 3.0))
    qh = float(np.quantile(v, 2.0 / 3.0))
    if ql >= qh:
        raise SystemExit("STOP_PGM_NATIVE0E_CONSENSUS_TERCILE_DEGENERATE")
    return ql, qh


def attach_primary_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """Primary acceleration = a_dir_accel_1 (0D frozen). No other accel enters."""
    x = df.copy()
    x["accel_raw"] = x[PRIMARY_ACCELERATION_COL].to_numpy(float)
    x["accel_positive"] = x["accel_raw"] > 0.0
    x["accel_plus"] = np.maximum(x["accel_raw"].to_numpy(float), 0.0)
    x["consensus_x_accel"] = x["consensus_score"].to_numpy(float) * x["accel_plus"].to_numpy(float)
    return x


def primary_eligibility_mask(x: pd.DataFrame) -> np.ndarray:
    """Exact primary-sample eligibility, reused for rank-map training, tercile
    training, predictive training and primary-sample construction.

    same_block_entry_valid & direction_stable & consensus_eligible &
    (base_action != 0) & isfinite(pi)
    """
    return (
        x["same_block_entry_valid"].to_numpy(bool)
        & x["direction_stable"].to_numpy(bool)
        & x["consensus_eligible"].to_numpy(bool)
        & (x["base_action"].to_numpy(float) != 0)
        & np.isfinite(x["pi"].to_numpy(float))
    )


def prepare_consensus(scored: pd.DataFrame, train_blocks: List[str]) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    x = add_consensus_primitives(scored)
    # rank-map reference must come ONLY from train-block, PRIMARY-ELIGIBLE rows.
    # Ineligible rows (no complete C_{t-1}, zero base action, non-finite pi) must
    # NOT move the empirical ranks / terciles.
    train_block_mask = x["block"].isin(train_blocks).to_numpy(bool)
    elig = primary_eligibility_mask(x)
    train_ref = x[train_block_mask & elig]
    maps = fit_rank_maps(train_ref)
    x = attach_consensus_score(x, maps)
    x = attach_primary_acceleration(x)
    return x, maps


def build_primary_sample_from_scored(x: pd.DataFrame, terciles: Tuple[float, float]) -> pd.DataFrame:
    pi = x["pi"].to_numpy(float)
    elig = (
        x["same_block_entry_valid"].to_numpy(bool)
        & x["direction_stable"].to_numpy(bool)
        & x["consensus_eligible"].to_numpy(bool)
        & (x["base_action"].to_numpy(float) != 0)
        & np.isfinite(pi)
    )
    sub = x[elig].copy()
    cs = sub["consensus_score"].to_numpy(float)
    ql, qh = terciles
    sub["consensus_group"] = np.where(cs >= qh, "HIGH",
                                      np.where(cs <= ql, "LOW", "MID"))
    return sub


# ===========================================================================
# Four-cell statistics
# ===========================================================================
def _cells_with_group(sub: pd.DataFrame, group_col: str) -> Dict[str, Dict[str, float]]:
    gv = sub[group_col].to_numpy()
    acc = sub["accel_positive"].to_numpy(bool)
    pi = sub["pi"].to_numpy(float)
    harm = sub["harm_flag"].to_numpy(int)
    haz = sub["hazard"].to_numpy(int)
    out: Dict[str, Dict[str, float]] = {}
    for grp in ["LOW", "HIGH"]:
        for a in [True, False]:
            m = (gv == grp) & (acc == a)
            key = f"{grp}_{'ACCEL' if a else 'OFF'}"
            n = int(m.sum())
            mp = float(np.mean(pi[m])) if n else float("nan")
            out[key] = dict(
                n=n,
                mean_pi=mp,
                # reporting-only economic fields; do NOT enter DID / verdict
                gross_EV=mp,
                net_EV_at_0p01=(mp - PRIMARY_COST_ATR0) if n else float("nan"),
                harm_rate=float(np.mean(harm[m])) if n else float("nan"),
                H1_prevalence=float(np.mean(haz[m])) if n else float("nan"),
            )
    return out


def primary_effects(cells: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    def _d(metric: str, ga: str, go: str) -> float:
        return cells[ga][metric] - cells[go][metric]

    dlow_pi = _d("mean_pi", "LOW_ACCEL", "LOW_OFF")
    dhigh_pi = _d("mean_pi", "HIGH_ACCEL", "HIGH_OFF")
    dlow_h = _d("harm_rate", "LOW_ACCEL", "LOW_OFF")
    dhigh_h = _d("harm_rate", "HIGH_ACCEL", "HIGH_OFF")
    dlow_z = _d("H1_prevalence", "LOW_ACCEL", "LOW_OFF")
    dhigh_z = _d("H1_prevalence", "HIGH_ACCEL", "HIGH_OFF")
    return dict(
        Delta_LOW_pi=dlow_pi, Delta_HIGH_pi=dhigh_pi, DID_pi=dhigh_pi - dlow_pi,
        Delta_LOW_harm=dlow_h, Delta_HIGH_harm=dhigh_h, DID_harm=dhigh_h - dlow_h,
        Delta_LOW_H1=dlow_z, Delta_HIGH_H1=dhigh_z, DID_H1=dhigh_z - dlow_z,
    )


def primary_cell_counts(sub: pd.DataFrame, group_col: str = "consensus_group") -> Dict[str, int]:
    """Pure STRUCTURE count of the four primary cells.

    Reads ONLY consensus_group and accel_positive. Never touches pi / harm_flag /
    hazard / r_trad_OC_ATR0, so cell-count auditing does not depend on outcomes.
    """
    gv = sub[group_col].to_numpy()
    acc = sub["accel_positive"].to_numpy(bool)
    out: Dict[str, int] = {}
    for grp in ["LOW", "HIGH"]:
        for a in [True, False]:
            key = f"{grp}_{'ACCEL' if a else 'OFF'}"
            out[key] = int(((gv == grp) & (acc == a)).sum())
    return out


def assert_min_cells(ev_full: pd.DataFrame) -> None:
    # count-only gate: must NOT call _cells_with_group (which would compute outcomes)
    cells = primary_cell_counts(ev_full)
    for k in PRIMARY_CELLS:
        if cells[k] < MIN_CELL_N:
            raise SystemExit(f"STOP_PGM_NATIVE0E_PRIMARY_CELL_TOO_SMALL: {k}={cells[k]}")


# ===========================================================================
# Day-cluster (paired) bootstrap of the DID
# ===========================================================================
def _summ(point: float, dist: np.ndarray) -> Dict[str, float]:
    dist = np.asarray(dist, float)
    dist = dist[np.isfinite(dist)]
    if len(dist) == 0:
        return dict(point=float("nan"), ci95_lower=float("nan"),
                    ci95_upper=float("nan"), p_pos=float("nan"))
    return dict(point=float(point),
                ci95_lower=float(np.percentile(dist, 2.5)),
                ci95_upper=float(np.percentile(dist, 97.5)),
                p_pos=float(np.mean(dist > 0)))


def bootstrap_did(eval_sub: pd.DataFrame, n_boot: int = BOOTSTRAP_N,
                  seed: int = BOOTSTRAP_SEED) -> Dict[str, Dict[str, float]]:
    """Pre-aggregate day x cell, then multinomial resample trading DAYS only.

    The same resampled day weights drive every cell, so Delta_LOW / Delta_HIGH /
    DID are jointly paired across the bootstrap.
    """
    sub = eval_sub[eval_sub["consensus_group"].isin(["LOW", "HIGH"])].copy()
    day = sub["entry_day"].to_numpy()
    days, pos = np.unique(day, return_inverse=True)
    D = len(days)

    grp = sub["consensus_group"].to_numpy()
    acc = sub["accel_positive"].to_numpy(bool)
    pi = sub["pi"].to_numpy(float)
    harm = sub["harm_flag"].to_numpy(int)
    haz = sub["hazard"].to_numpy(int)

    CELLS = ["LOW_OFF", "LOW_ACCEL", "HIGH_OFF", "HIGH_ACCEL"]
    cidx = {c: i for i, c in enumerate(CELLS)}
    # Vectorized day x cell aggregation. bincount accumulates in row order, so this is
    # bit-identical to the old per-row loop. cell encodes (HIGH?, ACCEL?) in CELLS order.
    cell = 2 * (grp == "HIGH").astype(np.int8) + acc.astype(np.int8)
    flat = pos * 4 + cell
    count = np.bincount(flat, minlength=D * 4).reshape(D, 4).astype(float)
    spi = np.bincount(flat, weights=pi, minlength=D * 4).reshape(D, 4)
    sharm = np.bincount(flat, weights=harm.astype(float), minlength=D * 4).reshape(D, 4)
    shaz = np.bincount(flat, weights=haz.astype(float), minlength=D * 4).reshape(D, 4)

    # point estimate: row-weighted full-sample means
    om = _overall_cell_means(sub)
    point = dict(
        dl=om["LOW_ACCEL"]["pi"] - om["LOW_OFF"]["pi"],
        dh=om["HIGH_ACCEL"]["pi"] - om["HIGH_OFF"]["pi"],
        did=(om["HIGH_ACCEL"]["pi"] - om["HIGH_OFF"]["pi"])
        - (om["LOW_ACCEL"]["pi"] - om["LOW_OFF"]["pi"]),
        dlh=om["LOW_ACCEL"]["harm"] - om["LOW_OFF"]["harm"],
        dhh=om["HIGH_ACCEL"]["harm"] - om["HIGH_OFF"]["harm"],
        didh=(om["HIGH_ACCEL"]["harm"] - om["HIGH_OFF"]["harm"])
        - (om["LOW_ACCEL"]["harm"] - om["LOW_OFF"]["harm"]),
        dlz=om["LOW_ACCEL"]["hz"] - om["LOW_OFF"]["hz"],
        dhz=om["HIGH_ACCEL"]["hz"] - om["HIGH_OFF"]["hz"],
        didz=(om["HIGH_ACCEL"]["hz"] - om["HIGH_OFF"]["hz"])
        - (om["LOW_ACCEL"]["hz"] - om["LOW_OFF"]["hz"]),
    )

    rng = np.random.default_rng(seed)
    # Vectorized multinomial: rng.multinomial(..., size=n_boot) is bit-identical to
    # n_boot sequential draws, so the per-bootstrap cell means match the old loop exactly.
    W = rng.multinomial(D, np.full(D, 1.0 / D), size=n_boot)  # (n_boot, D)
    cnt = W @ count
    mp = W @ spi
    mh = W @ sharm
    mz = W @ shaz
    mpi = np.where(cnt > 0, mp / cnt, np.nan)
    mh_ = np.where(cnt > 0, mh / cnt, np.nan)
    mz_ = np.where(cnt > 0, mz / cnt, np.nan)
    LOFF, LACC, HOFF, HACC = (cidx["LOW_OFF"], cidx["LOW_ACCEL"],
                              cidx["HIGH_OFF"], cidx["HIGH_ACCEL"])
    dl = mpi[:, LACC] - mpi[:, LOFF]
    dh = mpi[:, HACC] - mpi[:, HOFF]
    did = dh - dl
    dlh = mh_[:, LACC] - mh_[:, LOFF]
    dhh = mh_[:, HACC] - mh_[:, HOFF]
    didh = dhh - dlh
    dlz = mz_[:, LACC] - mz_[:, LOFF]
    dhz = mz_[:, HACC] - mz_[:, HOFF]
    didz = dhz - dlz
    dist = {
        "dl": dl, "dh": dh, "did": did,
        "dlh": dlh, "dhh": dhh, "didh": didh,
        "dlz": dlz, "dhz": dhz, "didz": didz,
    }
    res = {k: _summ(float(point[k]), np.asarray(dist[k], float)) for k in point}
    return {
        "Delta_LOW_pi": res["dl"], "Delta_HIGH_pi": res["dh"], "DID_pi": res["did"],
        "Delta_LOW_harm": res["dlh"], "Delta_HIGH_harm": res["dhh"], "DID_harm": res["didh"],
        "Delta_LOW_H1": res["dlz"], "Delta_HIGH_H1": res["dhz"], "DID_H1": res["didz"],
    }


def _overall_cell_means(sub: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    gv = sub["consensus_group"].to_numpy()
    acc = sub["accel_positive"].to_numpy(bool)
    pi = sub["pi"].to_numpy(float)
    harm = sub["harm_flag"].to_numpy(int)
    haz = sub["hazard"].to_numpy(int)
    out: Dict[str, Dict[str, float]] = {}
    for grp in ["LOW", "HIGH"]:
        for a in [True, False]:
            m = (gv == grp) & (acc == a)
            key = f"{grp}_{'ACCEL' if a else 'OFF'}"
            n = int(m.sum())
            out[key] = dict(
                pi=float(np.mean(pi[m])) if n else float("nan"),
                harm=float(np.mean(harm[m])) if n else float("nan"),
                hz=float(np.mean(haz[m])) if n else float("nan"),
            )
    return out


# ===========================================================================
# Predictive falsification (fixed models, no tuning)
# ===========================================================================
def _predict_ridge(train: pd.DataFrame, eval_: pd.DataFrame, num_cols, cat_cols, target: str):
    pipe = d0.make_ridge_pipeline(num_cols, cat_cols)
    pipe.fit(train[num_cols + cat_cols], train[target].to_numpy(float))
    pred = np.asarray(pipe.predict(eval_[num_cols + cat_cols]), float)
    y = eval_[target].to_numpy(float)
    sqerr = (pred - y) ** 2                     # row-level, same order as eval_
    return pred, dict(
        mse=float(np.mean(sqerr)),
        mae=float(np.mean(np.abs(pred - y))),
        spearman=float(scipy.stats.spearmanr(pred, y).statistic) if len(pred) > 2 else 0.0,
    ), sqerr


def _predict_logistic(train: pd.DataFrame, eval_: pd.DataFrame, num_cols, cat_cols, target: str):
    pipe = pm.make_pipeline(num_cols, cat_cols)
    pipe.fit(train[num_cols + cat_cols], train[target].to_numpy(int))
    p = np.asarray(pipe.predict_proba(eval_[num_cols + cat_cols])[:, 1], float)
    y = eval_[target].to_numpy(int)
    pc = np.clip(p, 1e-15, 1.0 - 1e-15)
    row_logloss = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))   # row-level
    brier = float(np.mean((p - y) ** 2))
    roc = float(sklearn.metrics.roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5
    prauc = float(sklearn.metrics.average_precision_score(y, p)) if len(np.unique(y)) > 1 else float(np.mean(y))
    return p, dict(log_loss=float(np.mean(row_logloss)), brier=brier, roc_auc=roc, pr_auc=prauc), row_logloss


def paired_day_mean_bootstrap(entry_day, values, n_boot: int = BOOTSTRAP_N,
                             seed: int = BOOTSTRAP_SEED) -> Dict[str, float]:
    """Generic entry-day (day-cluster) mean bootstrap — the SINGLE owner for every
    day-resampled statistic in this module (predictive delta, psych-gate policy diff, ...).

    Aggregates ``values`` per entry_day via bincount (row-order accumulation, bit-identical
    to a per-row loop), then multinomial-resamples DAYS only. ``rng.multinomial(...,
    size=n_boot)`` is bit-identical to ``n_boot`` sequential draws, so the result matches
    the old per-row + per-bootstrap loop exactly.

    point = mean(values); returns point / ci95_lower / ci95_upper / p_pos.
    """
    day = np.asarray(entry_day)
    days, pos = np.unique(day, return_inverse=True)
    D = len(days)
    values = np.asarray(values, float)

    # Vectorized per-day aggregation (bincount accumulates in row order -> bit-identical
    # to the old per-row loop).
    S = np.bincount(pos, weights=values, minlength=D)
    C = np.bincount(pos, minlength=D).astype(float)

    point = float(np.mean(values))
    rng = np.random.default_rng(seed)
    # Vectorized multinomial: bit-identical to n_boot sequential draws.
    W = rng.multinomial(D, np.full(D, 1.0 / D), size=n_boot)  # (n_boot, D)
    denom = W @ C
    dist = np.where(denom > 0, (W @ S) / denom, float("nan"))
    return _summ(point, dist)


def paired_day_loss_bootstrap(entry_day, loss0, loss1, n_boot: int = BOOTSTRAP_N,
                             seed: int = BOOTSTRAP_SEED) -> Dict[str, float]:
    """Entry-day paired bootstrap of the predictive increment.

    delta = loss0 - loss1  (positive => M1 with interaction IMPROVES over M0).
    Delegates the day-aggregation + multinomial bootstrap to ``paired_day_mean_bootstrap``
    so there is exactly ONE owner for every day-cluster bootstrap in this module.

    point = mean(delta); returns point / ci95 / P(delta > 0).
    """
    delta = np.asarray(loss0, float) - np.asarray(loss1, float)
    return paired_day_mean_bootstrap(entry_day, delta, n_boot, seed)


def m0_num() -> List[str]:
    return list(PRED_NUM_BASE)


def m1_num() -> List[str]:
    return list(PRED_NUM_BASE) + ["consensus_x_accel"]


# ===========================================================================
# Component interaction diagnostics (cannot change verdict)
# ===========================================================================
def component_diagnostics(eval_sub: pd.DataFrame,
                          rank_maps: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for comp in CONSENSUS_RAW:
        ref = rank_maps[comp]
        ql = float(np.quantile(ref, 1.0 / 3.0))
        qh = float(np.quantile(ref, 2.0 / 3.0))
        cv = eval_sub[comp].to_numpy(float)
        grp = np.where(cv >= qh, "HIGH", np.where(cv <= ql, "LOW", "MID"))
        s2 = eval_sub.copy()
        s2["_cg"] = grp
        cells = _cells_with_group(s2, "_cg")
        out[comp] = primary_effects(cells)
    return out


# ===========================================================================
# Age-zero / simplification diagnostic (cannot enter primary score)
# ===========================================================================
def age_zero_diagnostic(sub: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    cs = sub["consensus_score"].to_numpy(float)
    for c in ["abs_score_mu", "u_log_age", "upper_current_newest_age_zero",
              "lower_current_newest_age_zero"]:
        if c in sub.columns:
            v = sub[c].to_numpy(float)
            m = np.isfinite(v) & np.isfinite(cs)
            if m.sum() > 2:
                out[c] = float(scipy.stats.spearmanr(cs[m], v[m]).statistic)
    return out


# ===========================================================================
# Fixed psychology economic diagnostic (does NOT drive verdict)
# ===========================================================================
def psych_gate_diagnostic(eval_sub: pd.DataFrame, cost: float = PRIMARY_COST_ATR0,
                          n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> Dict[str, Any]:
    base = eval_sub["base_action"].to_numpy(float)
    high = eval_sub["consensus_group"].to_numpy() == "HIGH"
    accel = eval_sub["accel_positive"].to_numpy(bool)
    gate = base.copy()
    gate[high & accel] = 0.0  # ONLY avoid "high consensus + another acceleration"

    r = eval_sub["r_trad_OC_ATR0"].to_numpy(float)
    day = eval_sub["entry_day"].to_numpy()
    sym = eval_sub["symbol"].to_numpy()

    base_m = d0.strategy_metrics(base, r, cost, day, sym)
    gate_m = d0.strategy_metrics(gate, r, cost, day, sym)

    net = {"BASE": d0.net_return(base, r, cost), "PSYCH_GATE": d0.net_return(gate, r, cost)}
    # n_boot is owned by the caller (200 smoke, 2000 formal future); never fixed here.
    boot = d0.economic_bootstrap(day, net, n_boot=n_boot, seed=seed)

    # explicit paired day bootstrap for the policy difference (single owner:
    # paired_day_mean_bootstrap — bit-identical to the old row-aggregation + sequential
    # multinomial version).
    ndiff = net["PSYCH_GATE"] - net["BASE"]
    boot["PSYCH_GATE-BASE"] = paired_day_mean_bootstrap(day, ndiff, n_boot, seed)
    return dict(BASE=base_m, PSYCH_GATE=gate_m, bootstrap=boot)


# ===========================================================================
# Verdict (TB3 only)
# ===========================================================================
def determine_psych_verdict(did_pi: Dict[str, float], dlow_pi: Dict[str, float],
                            dhigh_pi: Dict[str, float]) -> str:
    if (did_pi["ci95_upper"] < 0) and (dlow_pi["ci95_lower"] > 0) and (dhigh_pi["ci95_upper"] < 0):
        return VERDICT["SIGN_SWITCH"]
    if did_pi["ci95_upper"] < 0:
        return VERDICT["STATE_DEP"]
    return VERDICT["NOT_SUPPORTED"]


# ===========================================================================
# Full window runner (Round 1: no artifacts; audit/smoke only)
# ===========================================================================
def run_window_complete(scored: pd.DataFrame, w: Dict[str, Any], n_boot: int,
                         eval_cap: Optional[int] = None,
                         model_train_cap: Optional[int] = None) -> Tuple[Dict[str, Any], Tuple[float, float], Dict[str, np.ndarray]]:
    t_prep = time.perf_counter()
    x, maps = prepare_consensus(scored, w["train"])

    # terciles trained ONLY on train-block PRIMARY-ELIGIBLE rows (same universe
    # as the rank-map reference). n_rank_train == n_tercile_train == n_train_primary.
    train_block_mask = x["block"].isin(w["train"]).to_numpy(bool)
    elig = primary_eligibility_mask(x)
    train_primary = x[train_block_mask & elig]
    n_rank_train = int(len(train_primary))
    terciles = compute_consensus_terciles(train_primary["consensus_score"].to_numpy(float))
    sub = build_primary_sample_from_scored(x, terciles)

    ev_full = sub[sub["block"] == w["eval"]].copy()
    assert_min_cells(ev_full)  # hard gate on the REAL eval (pre-cap)

    ev = ev_full
    if eval_cap is not None:
        ev = ev_full.head(eval_cap).copy()

    cells = _cells_with_group(ev, "consensus_group")
    effects = primary_effects(cells)
    boot = bootstrap_did(ev, n_boot=n_boot, seed=BOOTSTRAP_SEED)
    consensus_s = time.perf_counter() - t_prep

    # Predictive training uses the FULL train set by default (model_train_cap=None).
    # Smoke passes a fixed deterministic evenly-spaced subset (wiring-only) to cut cost.
    tr = sub[sub["block"].isin(w["train"])].copy()
    if model_train_cap is not None:
        n_tr = len(tr)
        if n_tr > model_train_cap:
            idx = np.linspace(0, n_tr - 1, model_train_cap, dtype=np.int64)
            tr = tr.iloc[idx].copy()

    t_model = time.perf_counter()
    pay0, m0m, sq0 = _predict_ridge(tr, ev, m0_num(), PRED_CAT, "pi")
    pay1, m1m, sq1 = _predict_ridge(tr, ev, m1_num(), PRED_CAT, "pi")
    h0p, h0m, ll0 = _predict_logistic(tr, ev, m0_num(), PRED_CAT, "harm_flag")
    h1p, h1m, ll1 = _predict_logistic(tr, ev, m1_num(), PRED_CAT, "harm_flag")
    model_s = time.perf_counter() - t_model

    # predictive falsification via ENTRY-DAY PAIRED bootstrap (not point estimate alone)
    t_boot = time.perf_counter()
    payoff_boot = paired_day_loss_bootstrap(ev["entry_day"].to_numpy(), sq0, sq1,
                                           n_boot=n_boot, seed=BOOTSTRAP_SEED)
    harm_boot = paired_day_loss_bootstrap(ev["entry_day"].to_numpy(), ll0, ll1,
                                         n_boot=n_boot, seed=BOOTSTRAP_SEED)

    # n_boot ownership: smoke=200, formal future=2000 -- passed through, never fixed here.
    psych = psych_gate_diagnostic(ev, PRIMARY_COST_ATR0, n_boot, BOOTSTRAP_SEED)
    comp = component_diagnostics(ev, maps)
    age = age_zero_diagnostic(ev)
    bootstrap_s = time.perf_counter() - t_boot

    return dict(
        cells=cells, effects=effects, bootstrap=boot,
        payoff_m0=m0m, payoff_m1=m1m, delta_mse=m0m["mse"] - m1m["mse"],
        harm_m0=h0m, harm_m1=h1m, delta_logloss=h0m["log_loss"] - h1m["log_loss"],
        payoff_bootstrap=payoff_boot, harm_bootstrap=harm_boot,
        psych=psych, component=comp, age_zero=age,
        n_rank_train=n_rank_train, n_train=len(tr), n_eval=len(ev),
        timing=dict(consensus_s=consensus_s, model_s=model_s, bootstrap_s=bootstrap_s),
    ), terciles, maps


# ===========================================================================
# Loading / scoring
# ===========================================================================
_load_counter = 0


def _load_and_score():
    # Unique tags per pipeline run: d0.build_acceleration_once allows each tag ONCE,
    # so re-running audit/smoke in the same process must use fresh tags.
    global _load_counter
    _load_counter += 1
    tag = f"0e_r{_load_counter}"
    prep = d0.load_prepared_frame()
    aligned = prep["aligned"]
    fit_A = pgm.fit_samplers_for_window(pgm.WINDOWS[0], pgm.SAMPLE_PATH, pgm.TRANSITION_SAMPLE_PATH)
    scored_A = d0.prepare_window_windowframe(aligned, fit_A, f"{tag}_A")
    fit_B = pgm.fit_samplers_for_window(pgm.WINDOWS[1], pgm.SAMPLE_PATH, pgm.TRANSITION_SAMPLE_PATH)
    scored_B = d0.prepare_window_windowframe(aligned, fit_B, f"{tag}_B")
    return prep, fit_A, scored_A, fit_B, scored_B


def require_full_authorization() -> None:
    """Token gate for the one-shot formal full run.

    Only the exact environment value "1" authorizes; anything else (unset, "0",
    or any other string) hard-stops BEFORE any load / sampler fit / recompute.
    """
    if os.environ.get(AUTHORIZE_ENV) != "1":
        raise SystemExit("STOP_PGM_NATIVE0E_FULL_NOT_AUTHORIZED")


# ===========================================================================
# Modes
# ===========================================================================
def run_audit_only() -> None:
    # Round-1.1 ACK: prior audit path at SHA 33a338f2df8148a835a7d43215ce296cd76407cd
    # prematurely evaluated full TB2/TB3 exploratory metrics (DID / payoff / harm /
    # PSYCH_GATE). Those numbers are NOT used to alter CONSENSUS_RAW, the 4 primitive
    # formulas, acceleration, the threshold rule, or the verdict rule. 0E remains
    # EXPLORATORY. Audit-only is GOVERNANCE ONLY: it never calls run_window_complete,
    # never computes DID / Ridge / Logistic / PSYCH_GATE, and emits no scientific metrics.
    # Round-1.1a ACK: SHA c37dc3e790ec2619a5bce007bf196220e5eb92dd still exposed the
    # full-eval four-cell mean_pi / gross_EV / net_EV_0p01 under the "NO SCIENTIFIC
    # METRICS" label (the cell helper computed pi/harm/H1 under the hood). Those
    # observed numbers are ALSO not used to change CONSENSUS_RAW / formula /
    # acceleration / threshold / verdict. Audit-only now uses primary_cell_counts
    # (structure-only; reads neither pi, harm_flag, hazard nor r_trad_OC_ATR0) for the
    # MIN_CELL_N gate and prints only row counts.
    print("=" * 60, flush=True)
    print("PGM-NATIVE-0E: AUDIT-ONLY", flush=True)
    print("=" * 60, flush=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), text=True).strip()
    print(f"[AUDIT] HEAD={head}")
    print(f"[AUDIT] BASE_SHA={BASE_SHA}")
    if subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"],
                      cwd=str(_REPO_ROOT), capture_output=True).returncode != 0:
        raise SystemExit("STOP_PGM_NATIVE0E_FREEZE_CHECK_FAIL")

    obs = n0a.load_observed_decision_universe()
    d0.assert_allowed_blocks(obs)
    if "TB4" in set(obs["block"].unique()):
        raise SystemExit("STOP_PGM_NATIVE0E_FORBIDDEN_BLOCK: TB4")

    hashes = n0c.compute_artifact_hashes()
    aud = n0a.audit_decision_universe(obs)
    err = n0a.audit_atr0_owner_parity(obs, n0a.load_transition_truth_audit()["cur"])
    print(f"[AUDIT] sample_sha={hashes['sample_artifact_sha256']}")
    print(f"[AUDIT] transition_sha={hashes['transition_artifact_sha256']}")
    print(f"[AUDIT] n_all_obs={aud['n_all_obs']} H0={aud['n_hazard0']} H1={aud['n_hazard1']}")
    print(f"[AUDIT] symbols={len(aud['symbols'])} blocks={sorted(obs['block'].unique().tolist())}")
    print(f"[AUDIT] max_abs_atr0_owner_error={err:.2e}")

    prep, fit_A, scored_A, fit_B, scored_B = _load_and_score()
    dA = d0.verify_window_score_owner(scored_A, fit_A, TB2_BLOCK)
    dB = d0.verify_window_score_owner(scored_B, fit_B, TB3_BLOCK)
    print(f"[AUDIT] WindowA score-owner parity (TB2) max_abs_diff={dA:.3e}")
    print(f"[AUDIT] WindowB score-owner parity (TB3) max_abs_diff={dB:.3e}")

    finA = d0.audit_acceleration_finite(scored_A)
    finB = d0.audit_acceleration_finite(scored_B)
    if not finA["all_finite"] or not finB["all_finite"]:
        raise SystemExit("STOP_PGM_NATIVE0E_ACCELERATION_NON_FINITE")
    print(f"[AUDIT] WindowA/B acceleration finite={finA['all_finite']}/{finB['all_finite']}")

    # consensus finite / eligibility / rank / tercile / sample-count audit (governance only)
    for blk, scored, tb in [(TB2_BLOCK, scored_A, pgm.WINDOWS[0]),
                            (TB3_BLOCK, scored_B, pgm.WINDOWS[1])]:
        x, _ = prepare_consensus(scored, tb["train"])
        elig = primary_eligibility_mask(x)
        for c in CONSENSUS_RAW + ["consensus_score"]:
            v = x[c].to_numpy(float)
            if np.any(np.isinf(v)):
                raise SystemExit(f"STOP_PGM_NATIVE0E_CONSENSUS_INF:{c}")
            # ineligible rows are NaN by design; only PRIMARY-eligible rows must be finite
            if not np.all(np.isfinite(v[elig])):
                raise SystemExit(f"STOP_PGM_NATIVE0E_CONSENSUS_NON_FINITE:{c}")
        tbm = x["block"].isin(tb["train"]).to_numpy(bool)
        train_primary = x[tbm & elig]
        n_rank_train = int(len(train_primary))
        ql, qh = compute_consensus_terciles(train_primary["consensus_score"].to_numpy(float))
        sub = build_primary_sample_from_scored(x, (ql, qh))
        tr_primary = sub[sub["block"].isin(tb["train"])]
        ev_primary = sub[sub["block"] == tb["eval"]]
        print(f"[AUDIT] {blk}: n_rank_train={n_rank_train} n_tercile_train={n_rank_train} "
              f"q_low={ql:.4f} q_high={qh:.4f}")
        print(f"[AUDIT] {blk}: n_train_primary={len(tr_primary)} n_eval_primary={len(ev_primary)}")
        # structure-only counts: NO outcome (pi / gross_EV / net_EV / harm / H1) is read or printed
        cells = primary_cell_counts(ev_primary)
        for k in PRIMARY_CELLS:
            print(f"[AUDIT]   {k}: n={cells[k]}")
        assert_min_cells(ev_primary)  # MIN_CELL_N gate on the real eval (pre-cap)
        print(f"[AUDIT] {blk}: MIN_CELL_N gate PASS (>= {MIN_CELL_N})")

    # consensus causal audit: current-row mutation must NOT change C(t-1)
    if not _causal_prefix_invariant():
        raise SystemExit("STOP_PGM_NATIVE0E_FEATURE_FUTURE_DEPENDENCE")
    print("[AUDIT] consensus causal (lag-1) audit: PASS")

    print("[AUDIT] AUDIT ONLY / NO SCIENTIFIC METRICS / NO SCIENTIFIC VERDICT EMITTED", flush=True)


def run_smoke_test() -> None:
    print("=" * 60, flush=True)
    print("PGM-NATIVE-0E: SMOKE (wiring only)", flush=True)
    print("=" * 60, flush=True)
    t0 = time.perf_counter()
    prep, fit_A, scored_A, fit_B, scored_B = _load_and_score()
    load_s = time.perf_counter() - t0
    dA = d0.verify_window_score_owner(scored_A, fit_A, TB2_BLOCK)
    dB = d0.verify_window_score_owner(scored_B, fit_B, TB3_BLOCK)
    print(f"[SMOKE] score-owner parity: WindowA(TB2)={dA:.3e} WindowB(TB3)={dB:.3e}")
    print(f"[SMOKE] load_and_score_seconds={load_s:.2f}")

    cons_a = cons_b = model_a = model_b = boot_total = 0.0
    for blk, w, sc in [(TB2_BLOCK, pgm.WINDOWS[0], scored_A),
                       (TB3_BLOCK, pgm.WINDOWS[1], scored_B)]:
        print(f"[SMOKE] --- window {blk} ---")
        r, _, _ = run_window_complete(sc, w, n_boot=SMOKE_BOOTSTRAP_N,
                                     eval_cap=SMOKE_EVAL_CAP,
                                     model_train_cap=SMOKE_MODEL_TRAIN_CAP)
        t = r.get("timing", {})
        print(f"  n_train_primary={r['n_train']} n_eval_primary(cap)={r['n_eval']}")
        print(f"  [timing] consensus={t.get('consensus_s', 0):.2f}s "
              f"model={t.get('model_s', 0):.2f}s bootstrap={t.get('bootstrap_s', 0):.2f}s")
        if blk == TB2_BLOCK:
            cons_a += t.get("consensus_s", 0); model_a += t.get("model_s", 0)
        else:
            cons_b += t.get("consensus_s", 0); model_b += t.get("model_s", 0)
        boot_total += t.get("bootstrap_s", 0)
        for k in PRIMARY_CELLS:
            print(f"    {k}: n={r['cells'][k]['n']}")
        e = r["effects"]
        print(f"  Delta_LOW_pi={e['Delta_LOW_pi']:.6f} Delta_HIGH_pi={e['Delta_HIGH_pi']:.6f} "
              f"DID_pi={e['DID_pi']:.6f}")
        print(f"  DID_harm={e['DID_harm']:.6f} DID_H1={e['DID_H1']:.6f}")
        b = r["bootstrap"]
        print(f"  DID_pi point={b['DID_pi']['point']:.6f} "
              f"CI=[{b['DID_pi']['ci95_lower']:.6f},{b['DID_pi']['ci95_upper']:.6f}]")
        print(f"  Delta_LOW_pi CI=[{b['Delta_LOW_pi']['ci95_lower']:.6f},{b['Delta_LOW_pi']['ci95_upper']:.6f}]")
        print(f"  Delta_HIGH_pi CI=[{b['Delta_HIGH_pi']['ci95_lower']:.6f},{b['Delta_HIGH_pi']['ci95_upper']:.6f}]")
        print(f"  payoff M0 MSE={r['payoff_m0']['mse']:.5f} M1={r['payoff_m1']['mse']:.5f} "
              f"delta_MSE={r['delta_mse']:.6f}")
        print(f"  Delta_MSE point={r['delta_mse']:.6f} "
              f"CI=[{r['payoff_bootstrap']['ci95_lower']:.6f},{r['payoff_bootstrap']['ci95_upper']:.6f}] "
              f"P>0={r['payoff_bootstrap']['p_pos']:.3f}")
        print(f"  harm M0 LL={r['harm_m0']['log_loss']:.5f} M1 LL={r['harm_m1']['log_loss']:.5f} "
              f"delta_LL={r['delta_logloss']:.6f}")
        print(f"  Delta_LogLoss point={r['delta_logloss']:.6f} "
              f"CI=[{r['harm_bootstrap']['ci95_lower']:.6f},{r['harm_bootstrap']['ci95_upper']:.6f}] "
              f"P>0={r['harm_bootstrap']['p_pos']:.3f}")
        for comp in CONSENSUS_RAW:
            ce = r["component"][comp]
            print(f"  COMPONENT {comp}: DID_pi={ce['DID_pi']:.6f} "
                  f"Delta_LOW_pi={ce['Delta_LOW_pi']:.6f} Delta_HIGH_pi={ce['Delta_HIGH_pi']:.6f}")
        print(f"  PSYCH_GATE netEV/dec={r['psych']['PSYCH_GATE']['net_EV_per_decision']:.6f} "
              f"BASE={r['psych']['BASE']['net_EV_per_decision']:.6f}")
        pg = r['psych']['bootstrap'].get('PSYCH_GATE-BASE')
        if pg:
            print(f"  PSYCH_GATE-BASE netEV point={pg['point']:.6f} "
                  f"CI=[{pg['ci95_lower']:.6f},{pg['ci95_upper']:.6f}]")
    total_s = time.perf_counter() - t0
    print("[SMOKE] SMOKE ONLY / NO SCIENTIFIC VERDICT", flush=True)
    print(f"[SMOKE] consensus_A_seconds={cons_a:.2f} consensus_B_seconds={cons_b:.2f} "
          f"model_A_seconds={model_a:.2f} model_B_seconds={model_b:.2f} "
          f"bootstrap_seconds={boot_total:.2f} total_smoke_seconds={total_s:.2f}", flush=True)
    print(f"[SMOKE COMPLETE] {total_s:.2f}s", flush=True)


def _causal_prefix_invariant() -> bool:
    """Consensus C_{t-1} at row k must be unchanged by edits to rows > k."""
    rng = np.random.default_rng(7)
    n = 12
    df = pd.DataFrame(dict(
        symbol=["X"] * n, episode_id=["E1"] * n, block=["TB1"] * n,
        bar_t=np.arange(100, 100 + n), start_bar=[100] * n,
        score_mu=rng.normal(0, 1, n),
        path_last_return_R=rng.normal(0, 0.5, n),
        cur_up_distance_R=np.abs(rng.normal(1.0, 0.3, n)),
        cur_down_distance_R=np.abs(rng.normal(1.0, 0.3, n)),
        upper_newest_log_age=rng.uniform(0.5, 5, n),
        lower_newest_log_age=rng.uniform(0.5, 5, n),
        upper_n_active_identities=rng.integers(1, 8, n).astype(float),
        lower_n_active_identities=rng.integers(1, 8, n).astype(float),
    ))
    a = add_consensus_primitives(df)
    b = add_consensus_primitives(df.copy())
    # perturb rows 6..end
    for c in ["cur_up_distance_R", "cur_down_distance_R", "upper_newest_log_age",
              "lower_newest_log_age", "upper_n_active_identities",
              "lower_n_active_identities", "path_last_return_R", "score_mu"]:
        b.loc[6:, c] = b.loc[6:, c] * 4.0 + 3.0
    for c in CONSENSUS_RAW:
        if not np.allclose(a[c].to_numpy(float)[:6], b[c].to_numpy(float)[:6], atol=1e-12, equal_nan=True):
            return False
    return True


# ===========================================================================
# Round 2: formal artifact contract (pre-run closure / assembly / validators)
# ===========================================================================
def assert_no_existing_prefixed_artifacts(out_dir: Path = OUT_DIR) -> None:
    """Pre-run closure: refuse if ANY pgm_native0e1_* artifact already exists.

    Catches known artifacts, unknown extra artifacts, temp files, and partial runs.
    Never deletes anything automatically.
    """
    existing = {p.name for p in out_dir.glob(f"{PREFIX}_*")}
    if existing:
        raise SystemExit(
            f"STOP_PGM_NATIVE0E_FORMAL_ARTIFACT_ALREADY_EXISTS: {sorted(existing)}")


def _collect_formal_meta(prep, scored_A, scored_B, fit_A, fit_B, head: str) -> Dict[str, Any]:
    """Governance metadata for the summary (universe stats, hashes, window parity)."""
    universe = n0a.load_observed_decision_universe()
    aud = n0a.audit_decision_universe(universe)
    hashes = n0c.compute_artifact_hashes()
    return dict(
        n_all_obs=aud["n_all_obs"], n_hazard0=aud["n_hazard0"], n_hazard1=aud["n_hazard1"],
        symbols=list(aud["symbols"]),
        blocks=sorted(universe["block"].unique().tolist()),
        sample_sha=hashes["sample_artifact_sha256"],
        transition_sha=hashes["transition_artifact_sha256"],
        winA_owner=d0.verify_window_score_owner(scored_A, fit_A, TB2_BLOCK),
        winB_owner=d0.verify_window_score_owner(scored_B, fit_B, TB3_BLOCK),
        winA_finite=bool(d0.audit_acceleration_finite(scored_A)["all_finite"]),
        winB_finite=bool(d0.audit_acceleration_finite(scored_B)["all_finite"]),
    )


def _formal_run_windows(scored_A: pd.DataFrame, scored_B: pd.DataFrame):
    """Window A -> TB2, Window B -> TB3. Each window runs exactly ONCE with the
    frozen formal contract (n_boot=2000, eval_cap=None, model_train_cap=None)."""
    res_A, terr_A, maps_A = run_window_complete(
        scored_A, pgm.WINDOWS[0], n_boot=BOOTSTRAP_N, eval_cap=None, model_train_cap=None)
    res_B, terr_B, maps_B = run_window_complete(
        scored_B, pgm.WINDOWS[1], n_boot=BOOTSTRAP_N, eval_cap=None, model_train_cap=None)
    return res_A, res_B, terr_A, terr_B, maps_A, maps_B


PRIMARY_CELL_COLS = ["block", "cell", "n", "mean_pi", "gross_EV", "net_EV_at_0p01",
                     "harm_rate", "H1_prevalence"]
PRIMARY_BOOT_COLS = ["block", "metric", "point", "ci95_lower", "ci95_upper", "p_pos"]
PRED_METRIC_COLS = ["block", "target", "model", "mse", "mae", "spearman",
                    "log_loss", "brier", "roc_auc", "pr_auc"]
PRED_BOOT_COLS = ["block", "metric", "point", "ci95_lower", "ci95_upper", "p_pos"]
PSYCH_COLS = ["n_decisions", "n_trades", "trade_rate", "gross_total_ATR0", "net_total_ATR0",
              "gross_EV_per_decision", "net_EV_per_decision", "net_EV_per_trade",
              "win_rate", "mean_win", "mean_loss", "payoff_ratio", "profit_factor",
              "break_even_cost", "daily_sharpe_annualized", "max_drawdown_ATR0",
              "positive_symbol_count", "top3_profit_share"]
PSYCH_COLS_FULL = ["block", "policy"] + PSYCH_COLS + ["point", "ci95_lower", "ci95_upper", "p_pos"]
COMP_COLS = ["Delta_LOW_pi", "Delta_HIGH_pi", "DID_pi", "Delta_LOW_harm", "Delta_HIGH_harm",
             "DID_harm", "Delta_LOW_H1", "Delta_HIGH_H1", "DID_H1"]
COMP_DIAG_COLS = ["block", "component"] + COMP_COLS
AGE_DIAG_COLS = ["block", "variable", "spearman_with_consensus"]
BOOT_METRICS = ["Delta_LOW_pi", "Delta_HIGH_pi", "DID_pi", "Delta_LOW_harm", "Delta_HIGH_harm",
                "DID_harm", "Delta_LOW_H1", "Delta_HIGH_H1", "DID_H1"]


def build_known_limitations() -> List[str]:
    return [
        "TB3 is exploratory / previously inspected, not pristine OOS.",
        "TB4 untouched.",
        "Frozen PGM sample exclusions remain.",
        "Psychology is a structural proxy, not observed human sentiment.",
        "H1 means episode termination, not structural reversal.",
        "harm means one-bar continuation harm, not structural reversal.",
        "Consensus score is equal-weight, train-ranked structural proxy.",
        "Formal outcome is one-bar next-open to next-close payoff.",
        "Primary economic cost is normalized 0.01 ATR0, not a market-specific execution model.",
        "No new data added.",
        "No threshold / feature / symbol tuning.",
        "Earlier exploratory audit leakage at 33a338f and c37dc3e was observed, "
        "but not used to alter the frozen science.",
    ]


def _assemble_artifacts(res_A, res_B, terr_A, terr_B, meta, head: str):
    """Build the 7 CSV DataFrames + summary dict from run_window_complete outputs.

    Pure record-to-DataFrame assembly (no per-row DataFrame.append / iterrows /
    concat-in-loop). Every artifact is expanded directly from results already
    produced by run_window_complete — no repeated fit or recompute.
    """
    pair = [(TB2_BLOCK, res_A, terr_A), (TB3_BLOCK, res_B, terr_B)]

    # 1. primary cells (exact 8 rows)
    cell_rows = []
    for blk, res, _ in pair:
        for cell in PRIMARY_CELLS:
            c = res["cells"][cell]
            cell_rows.append(dict(
                block=blk, cell=cell, n=int(c["n"]), mean_pi=float(c["mean_pi"]),
                gross_EV=float(c["gross_EV"]), net_EV_at_0p01=float(c["net_EV_at_0p01"]),
                harm_rate=float(c["harm_rate"]), H1_prevalence=float(c["H1_prevalence"])))
    primary_cells = pd.DataFrame(cell_rows, columns=PRIMARY_CELL_COLS)

    # 2. primary bootstrap (exact 18 rows)
    boot_rows = []
    for blk, res, _ in pair:
        for m in BOOT_METRICS:
            b = res["bootstrap"][m]
            boot_rows.append(dict(
                block=blk, metric=m, point=float(b["point"]),
                ci95_lower=float(b["ci95_lower"]), ci95_upper=float(b["ci95_upper"]),
                p_pos=float(b["p_pos"])))
    primary_bootstrap = pd.DataFrame(boot_rows, columns=PRIMARY_BOOT_COLS)

    # 3. predictive metrics (exact 12 rows: PAYOFF_M0/M1, HARM_M0/M1, +2 delta rows)
    nan = float("nan")
    pred_rows = []
    for blk, res, _ in pair:
        m0, m1 = res["payoff_m0"], res["payoff_m1"]
        h0, h1 = res["harm_m0"], res["harm_m1"]
        pred_rows += [
            dict(block=blk, target="pi", model="PAYOFF_M0", mse=m0["mse"], mae=m0["mae"],
                 spearman=m0["spearman"], log_loss=nan, brier=nan, roc_auc=nan, pr_auc=nan),
            dict(block=blk, target="pi", model="PAYOFF_M1", mse=m1["mse"], mae=m1["mae"],
                 spearman=m1["spearman"], log_loss=nan, brier=nan, roc_auc=nan, pr_auc=nan),
            dict(block=blk, target="harm_flag", model="HARM_M0", mse=nan, mae=nan, spearman=nan,
                 log_loss=h0["log_loss"], brier=h0["brier"], roc_auc=h0["roc_auc"], pr_auc=h0["pr_auc"]),
            dict(block=blk, target="harm_flag", model="HARM_M1", mse=nan, mae=nan, spearman=nan,
                 log_loss=h1["log_loss"], brier=h1["brier"], roc_auc=h1["roc_auc"], pr_auc=h1["pr_auc"]),
            dict(block=blk, target="pi", model="PAYOFF_DELTA", mse=res["delta_mse"], mae=nan,
                 spearman=nan, log_loss=nan, brier=nan, roc_auc=nan, pr_auc=nan),
            dict(block=blk, target="harm_flag", model="HARM_DELTA", mse=nan, mae=nan, spearman=nan,
                 log_loss=res["delta_logloss"], brier=nan, roc_auc=nan, pr_auc=nan),
        ]
    predictive_metrics = pd.DataFrame(pred_rows, columns=PRED_METRIC_COLS)

    # 4. predictive bootstrap (exact 4 rows)
    pb_rows = []
    for blk, res, _ in pair:
        p, h = res["payoff_bootstrap"], res["harm_bootstrap"]
        pb_rows += [
            dict(block=blk, metric="Delta_MSE", point=float(p["point"]),
                 ci95_lower=float(p["ci95_lower"]), ci95_upper=float(p["ci95_upper"]), p_pos=float(p["p_pos"])),
            dict(block=blk, metric="Delta_LogLoss", point=float(h["point"]),
                 ci95_lower=float(h["ci95_lower"]), ci95_upper=float(h["ci95_upper"]), p_pos=float(h["p_pos"])),
        ]
    predictive_bootstrap = pd.DataFrame(pb_rows, columns=PRED_BOOT_COLS)

    # 5. psych gate (exact 6 rows: BASE / PSYCH_GATE / PSYCH_GATE_MINUS_BASE)
    pg_rows = []
    for blk, res, _ in pair:
        psy = res["psych"]
        for pol in ["BASE", "PSYCH_GATE"]:
            sm = psy[pol]
            row = {"block": blk, "policy": pol}
            for c in PSYCH_COLS:
                row[c] = sm[c]
            row.update(point=nan, ci95_lower=nan, ci95_upper=nan, p_pos=nan)
            pg_rows.append(row)
        diff = psy["bootstrap"]["PSYCH_GATE-BASE"]
        row = {"block": blk, "policy": "PSYCH_GATE_MINUS_BASE"}
        for c in PSYCH_COLS:
            row[c] = nan
        row.update(point=float(diff["point"]), ci95_lower=float(diff["ci95_lower"]),
                   ci95_upper=float(diff["ci95_upper"]), p_pos=float(diff["p_pos"]))
        pg_rows.append(row)
    psych_gate = pd.DataFrame(pg_rows, columns=PSYCH_COLS_FULL)

    # 6. component diagnostics (exact 8 rows; secondary only, never enters verdict)
    comp_rows = []
    for blk, res, _ in pair:
        for comp in CONSENSUS_RAW:
            eff = res["component"][comp]
            row = {"block": blk, "component": comp}
            for c in COMP_COLS:
                row[c] = float(eff[c])
            comp_rows.append(row)
    component_diagnostics = pd.DataFrame(comp_rows, columns=COMP_DIAG_COLS)

    # 7. age-zero diagnostics (one row per block x variable present in the frame)
    az_rows = []
    age_vars: List[str] = []
    for blk, res, _ in pair:
        az = res["age_zero"]
        for var, val in az.items():
            az_rows.append(dict(block=blk, variable=var, spearman_with_consensus=float(val)))
            if var not in age_vars:
                age_vars.append(var)
    age_zero_diagnostics = pd.DataFrame(az_rows, columns=AGE_DIAG_COLS)

    dfs = {
        ARTIFACT_FILES[0]: primary_cells,
        ARTIFACT_FILES[1]: primary_bootstrap,
        ARTIFACT_FILES[2]: predictive_metrics,
        ARTIFACT_FILES[3]: predictive_bootstrap,
        ARTIFACT_FILES[4]: psych_gate,
        ARTIFACT_FILES[5]: component_diagnostics,
        ARTIFACT_FILES[6]: age_zero_diagnostics,
    }

    # TB3-only verdict from the TB3 primary bootstrap.
    tb3 = res_B["bootstrap"]
    verdict = determine_psych_verdict(tb3["DID_pi"], tb3["Delta_LOW_pi"], tb3["Delta_HIGH_pi"])

    def _block_stat(blk, res, terr):
        return dict(
            n_rank_train=int(res["n_rank_train"]),
            n_train_primary=int(res["n_rank_train"]),
            n_eval_primary=int(res["n_eval"]),
            q_low=float(terr[0]), q_high=float(terr[1]),
            cell_counts={k: int(res["cells"][k]["n"]) for k in PRIMARY_CELLS},
        )

    summary = dict(
        experiment_name=EXPERIMENT_NAME,
        experiment_scope=EXPERIMENT_SCOPE,
        run_head=head,
        base_sha=BASE_SHA,
        sample_artifact_sha256=meta["sample_sha"],
        transition_artifact_sha256=meta["transition_sha"],
        n_all_obs=int(meta["n_all_obs"]), n_hazard0=int(meta["n_hazard0"]), n_hazard1=int(meta["n_hazard1"]),
        symbols=list(meta["symbols"]), blocks=list(EVAL_BLOCKS),
        bootstrap_n=int(BOOTSTRAP_N), bootstrap_seed=int(BOOTSTRAP_SEED), cluster_owner=CLUSTER_OWNER,
        primary_cost_atr0=float(PRIMARY_COST_ATR0),
        consensus_raw=list(CONSENSUS_RAW),
        primary_acceleration_col=PRIMARY_ACCELERATION_COL,
        consensus_time_contract="C_tminus1_to_A_t_to_pi_tplus1",
        windowA_score_owner_max_abs_diff=float(meta["winA_owner"]),
        windowB_score_owner_max_abs_diff=float(meta["winB_owner"]),
        windowA_acceleration_finite=bool(meta["winA_finite"]),
        windowB_acceleration_finite=bool(meta["winB_finite"]),
        TB2=_block_stat(TB2_BLOCK, res_A, terr_A),
        TB3=_block_stat(TB3_BLOCK, res_B, terr_B),
        psychology_verdict=verdict,
        artifact_files=list(ARTIFACT_FILES),
        known_limitations=build_known_limitations(),
        run_meta=dict(n_boot=int(BOOTSTRAP_N), model_train_cap=None, eval_cap=None),
        timing=dict(windowA=res_A["timing"], windowB=res_B["timing"]),
        actual_age_zero_variables=sorted(age_vars),
    )
    return dfs, summary


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_artifacts(dfs, summary, out_dir: Path = OUT_DIR) -> None:
    """Write 7 CSVs + 1 JSON to disk. The JSON carries the exact in-memory summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in dfs.items():
        df.to_csv(out_dir / name, index=False)
    (out_dir / ARTIFACT_FILES[7]).write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")


def validate_in_memory_results(dfs, summary) -> None:
    """Pre-disk structural / scientific sanity on the in-memory results."""
    if set(dfs.keys()) != set(ARTIFACT_FILES[:7]):
        raise SystemExit("STOP_PGM_NATIVE0E_INMEM_RESULT_KEYS")
    if set(summary["blocks"]) != {"TB2", "TB3"}:
        raise SystemExit("STOP_PGM_NATIVE0E_BLOCKS_NOT_TB2_TB3")
    if len(dfs[ARTIFACT_FILES[0]]) != 8:
        raise SystemExit("STOP_PGM_NATIVE0E_PRIMARY_CELLS_ROWS")
    if len(dfs[ARTIFACT_FILES[1]]) != 18:
        raise SystemExit("STOP_PGM_NATIVE0E_PRIMARY_BOOTSTRAP_ROWS")
    if len(dfs[ARTIFACT_FILES[2]]) != 12:
        raise SystemExit("STOP_PGM_NATIVE0E_PREDICTIVE_METRICS_ROWS")
    if len(dfs[ARTIFACT_FILES[3]]) != 4:
        raise SystemExit("STOP_PGM_NATIVE0E_PREDICTIVE_BOOTSTRAP_ROWS")
    if len(dfs[ARTIFACT_FILES[4]]) != 6:
        raise SystemExit("STOP_PGM_NATIVE0E_PSYCH_GATE_ROWS")
    if len(dfs[ARTIFACT_FILES[5]]) != 8:
        raise SystemExit("STOP_PGM_NATIVE0E_COMPONENT_ROWS")
    # cell n >= MIN_CELL_N
    if int(dfs[ARTIFACT_FILES[0]]["n"].min()) < MIN_CELL_N:
        raise SystemExit("STOP_PGM_NATIVE0E_CELL_TOO_SMALL")
    # all formal CI ordered
    for nm in (ARTIFACT_FILES[1], ARTIFACT_FILES[3]):
        df = dfs[nm]
        if (df["ci95_lower"] > df["ci95_upper"]).any():
            raise SystemExit(f"STOP_PGM_NATIVE0E_CI_UNORDERED:{nm}")
    diff = dfs[ARTIFACT_FILES[4]]
    diff = diff[diff["policy"] == "PSYCH_GATE_MINUS_BASE"]
    if (diff["ci95_lower"] > diff["ci95_upper"]).any():
        raise SystemExit("STOP_PGM_NATIVE0E_PSYCH_CI_UNORDERED")
    # all P>0 in [0,1]
    for nm in (ARTIFACT_FILES[1], ARTIFACT_FILES[3]):
        p = dfs[nm]["p_pos"].to_numpy(float)
        if (p < 0).any() or (p > 1).any():
            raise SystemExit(f"STOP_PGM_NATIVE0E_P_POS_RANGE:{nm}")
    # window owner <= 1e-12
    if summary["windowA_score_owner_max_abs_diff"] > 1e-12:
        raise SystemExit("STOP_PGM_NATIVE0E_WINDOWA_OWNER_PARITY")
    if summary["windowB_score_owner_max_abs_diff"] > 1e-12:
        raise SystemExit("STOP_PGM_NATIVE0E_WINDOWB_OWNER_PARITY")
    # acceleration finite
    if not summary["windowA_acceleration_finite"] or not summary["windowB_acceleration_finite"]:
        raise SystemExit("STOP_PGM_NATIVE0E_ACCELERATION_NON_FINITE")
    # TB3-only verdict equals determine_psych_verdict recomputed from disk-ready data
    pb = dfs[ARTIFACT_FILES[1]]
    tb3 = pb[pb["block"] == TB3_BLOCK].set_index("metric")
    def _bd(m):
        return dict(point=tb3.loc[m, "point"], ci95_lower=tb3.loc[m, "ci95_lower"],
                    ci95_upper=tb3.loc[m, "ci95_upper"], p_pos=tb3.loc[m, "p_pos"])
    v = determine_psych_verdict(_bd("DID_pi"), _bd("Delta_LOW_pi"), _bd("Delta_HIGH_pi"))
    if v != summary["psychology_verdict"]:
        raise SystemExit("STOP_PGM_NATIVE0E_VERDICT_INCONSISTENT")
    # formal full-fit contract
    rm = summary["run_meta"]
    if rm["n_boot"] != 2000 or rm["model_train_cap"] is not None or rm["eval_cap"] is not None:
        raise SystemExit("STOP_PGM_NATIVE0E_RUN_META_CONTRACT")


def _assert_csv_parity(disk: pd.DataFrame, exp: pd.DataFrame, name: str) -> None:
    if list(disk.columns) != list(exp.columns):
        raise SystemExit(f"STOP_PGM_NATIVE0E_CSV_COLS:{name}")
    if len(disk) != len(exp):
        raise SystemExit(f"STOP_PGM_NATIVE0E_CSV_ROWS:{name}:{len(disk)}!={len(exp)}")
    sortkeys = {
        ARTIFACT_FILES[0]: ["block", "cell"],
        ARTIFACT_FILES[1]: ["block", "metric"],
        ARTIFACT_FILES[2]: ["block", "target", "model"],
        ARTIFACT_FILES[3]: ["block", "metric"],
        ARTIFACT_FILES[4]: ["block", "policy"],
        ARTIFACT_FILES[5]: ["block", "component"],
        ARTIFACT_FILES[6]: ["block", "variable"],
    }
    sk = sortkeys[name]
    d = disk.sort_values(sk).reset_index(drop=True)
    e = exp.sort_values(sk).reset_index(drop=True)
    for col in e.columns:
        ev = e[col].to_numpy()
        dv = d[col].to_numpy()
        if pd.api.types.is_numeric_dtype(e[col]) and pd.api.types.is_numeric_dtype(d[col]):
            if not np.allclose(ev, dv, rtol=0, atol=1e-12, equal_nan=True):
                raise SystemExit(f"STOP_PGM_NATIVE0E_CSV_NUMERIC:{name}:{col}")
        else:
            if not (pd.Series(ev).astype(str).to_numpy() == pd.Series(dv).astype(str).to_numpy()).all():
                raise SystemExit(f"STOP_PGM_NATIVE0E_CSV_STRING:{name}:{col}")


def _json_equal(a, b, atol: float = 1e-12) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_json_equal(a[k], b[k], atol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_json_equal(x, y, atol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= atol
    return a == b


def _assert_disk_content_sanity(out_dir: Path) -> None:
    """Independent scientific sanity on the RE-READ disk data (not the in-memory dict)."""
    pc = pd.read_csv(out_dir / ARTIFACT_FILES[0])
    if int(pc["n"].min()) < MIN_CELL_N:
        raise SystemExit("STOP_PGM_NATIVE0E_CELL_TOO_SMALL_DISK")
    for nm in (ARTIFACT_FILES[1], ARTIFACT_FILES[3]):
        df = pd.read_csv(out_dir / nm)
        if (df["ci95_lower"] > df["ci95_upper"]).any():
            raise SystemExit(f"STOP_PGM_NATIVE0E_CI_UNORDERED_DISK:{nm}")
    pg = pd.read_csv(out_dir / ARTIFACT_FILES[4])
    diff = pg[pg["policy"] == "PSYCH_GATE_MINUS_BASE"]
    if (diff["ci95_lower"] > diff["ci95_upper"]).any():
        raise SystemExit("STOP_PGM_NATIVE0E_PSYCH_CI_UNORDERED_DISK")
    for nm in (ARTIFACT_FILES[1], ARTIFACT_FILES[3]):
        p = pd.read_csv(out_dir / nm)["p_pos"].to_numpy(float)
        if (p < 0).any() or (p > 1).any():
            raise SystemExit(f"STOP_PGM_NATIVE0E_P_POS_RANGE_DISK:{nm}")
    summary = json.loads((out_dir / ARTIFACT_FILES[7]).read_text(encoding="utf-8"))
    if summary["windowA_score_owner_max_abs_diff"] > 1e-12:
        raise SystemExit("STOP_PGM_NATIVE0E_WINDOWA_OWNER_DISK")
    if summary["windowB_score_owner_max_abs_diff"] > 1e-12:
        raise SystemExit("STOP_PGM_NATIVE0E_WINDOWB_OWNER_DISK")
    if not (summary["windowA_acceleration_finite"] and summary["windowB_acceleration_finite"]):
        raise SystemExit("STOP_PGM_NATIVE0E_ACCELERATION_DISK")
    pb = pd.read_csv(out_dir / ARTIFACT_FILES[1])
    tb3 = pb[pb["block"] == TB3_BLOCK].set_index("metric")
    def _bd(m):
        return dict(point=tb3.loc[m, "point"], ci95_lower=tb3.loc[m, "ci95_lower"],
                    ci95_upper=tb3.loc[m, "ci95_upper"], p_pos=tb3.loc[m, "p_pos"])
    v = determine_psych_verdict(_bd("DID_pi"), _bd("Delta_LOW_pi"), _bd("Delta_HIGH_pi"))
    if v != summary["psychology_verdict"]:
        raise SystemExit("STOP_PGM_NATIVE0E_VERDICT_DISK")
    rm = summary["run_meta"]
    if rm["n_boot"] != 2000 or rm["model_train_cap"] is not None or rm["eval_cap"] is not None:
        raise SystemExit("STOP_PGM_NATIVE0E_RUN_META_DISK")


def validate_output_artifacts(out_dir: Path, dfs, summary) -> None:
    """Re-read ALL 8 artifacts from disk and compare to the in-memory truth.

    Checks: (1) exact artifact set, (2) per-CSV row/column/value parity (rtol=0,
    atol<=1e-12, strings exact), (3) JSON item-by-item, (4) independent scientific
    sanity on the re-read disk data.
    """
    actual = {p.name for p in out_dir.glob(f"{PREFIX}_*")}
    expected = set(ARTIFACT_FILES)
    if actual != expected:
        raise SystemExit(
            f"STOP_PGM_NATIVE0E_OUTPUT_PARITY_FAIL: actual={sorted(actual)} expected={sorted(expected)}")
    for name in ARTIFACT_FILES[:7]:
        disk = pd.read_csv(out_dir / name)
        _assert_csv_parity(disk, dfs[name], name)
    loaded = json.loads((out_dir / ARTIFACT_FILES[7]).read_text(encoding="utf-8"))
    if not _json_equal(loaded, summary):
        raise SystemExit("STOP_PGM_NATIVE0E_SUMMARY_JSON_PARITY_FAIL")
    _assert_disk_content_sanity(out_dir)


def run_full_exploratory(out_dir: Path = OUT_DIR) -> None:
    """One-shot formal full run. GATED by AUTHORIZE_PGM_NATIVE0E_FULL_EXPLORATORY=1.

    Pre-flight: auth -> clean tree -> clean index -> no existing artifacts -> HEAD.
    Then a SINGLE _load_and_score(), windows A(TB2)/B(TB3) each once (n_boot=2000,
    eval_cap=None, model_train_cap=None), assemble artifacts, validate in-memory,
    write, then disk-parity validate.
    """
    # ---- pre-flight ----
    require_full_authorization()
    if subprocess.run(["git", "diff", "--exit-code"], cwd=str(_REPO_ROOT),
                      capture_output=True).returncode != 0:
        raise SystemExit("STOP_PGM_NATIVE0E_FORMAL_TREE_NOT_CLEAN")
    if subprocess.run(["git", "diff", "--cached", "--exit-code"], cwd=str(_REPO_ROOT),
                      capture_output=True).returncode != 0:
        raise SystemExit("STOP_PGM_NATIVE0E_FORMAL_INDEX_NOT_CLEAN")
    assert_no_existing_prefixed_artifacts(out_dir)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), text=True).strip()

    # ---- single load + score ----
    prep, fit_A, scored_A, fit_B, scored_B = _load_and_score()

    # ---- windows (each exactly once) ----
    res_A, res_B, terr_A, terr_B, maps_A, maps_B = _formal_run_windows(scored_A, scored_B)

    # ---- metadata ----
    meta = _collect_formal_meta(prep, scored_A, scored_B, fit_A, fit_B, head)

    # ---- assemble ----
    dfs, summary = _assemble_artifacts(res_A, res_B, terr_A, terr_B, meta, head)

    # ---- validate in-memory, write, then disk parity ----
    validate_in_memory_results(dfs, summary)
    write_artifacts(dfs, summary, out_dir)
    validate_output_artifacts(out_dir, dfs, summary)


def main() -> None:
    ap = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full-exploratory", action="store_true")
    args = ap.parse_args()
    if args.full_exploratory:
        require_full_authorization()
        run_full_exploratory()
    elif args.smoke:
        run_smoke_test()
    else:
        run_audit_only()


if __name__ == "__main__":
    main()
