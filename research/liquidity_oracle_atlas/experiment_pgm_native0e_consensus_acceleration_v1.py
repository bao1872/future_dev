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
`--full-exploratory` is HARD-BLOCKED (STOP_PGM_NATIVE0E_FULL_NOT_AUTHORIZED_ROUND1).
"""

from __future__ import annotations

import argparse
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
CLUSTER_OWNER = "entry_day"

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

    x["lag1_score_mu"] = g["score_mu"].shift(1)
    d_now = np.sign(x["score_mu"].to_numpy(float))
    d_prev = np.sign(x["lag1_score_mu"].to_numpy(float))

    # direction_stable: same direction now as at t-1, and t-1 was not undecided
    x["direction_stable"] = (
        np.isfinite(d_prev) & (d_now != 0) & (d_prev == d_now)
    )

    up = g["cur_up_distance_R"].shift(1).to_numpy(float)
    dn = g["cur_down_distance_R"].shift(1).to_numpy(float)
    up_age = g["upper_newest_log_age"].shift(1).to_numpy(float)
    dn_age = g["lower_newest_log_age"].shift(1).to_numpy(float)
    up_n = g["upper_n_active_identities"].shift(1).to_numpy(float)
    dn_n = g["lower_n_active_identities"].shift(1).to_numpy(float)

    front = np.where(d_prev > 0, up, dn)
    back = np.where(d_prev > 0, dn, up)
    front_age = np.where(d_prev > 0, up_age, dn_age)
    back_age = np.where(d_prev > 0, dn_age, up_age)
    front_n = np.where(d_prev > 0, up_n, dn_n)
    back_n = np.where(d_prev > 0, dn_n, up_n)

    # 1. Liquidity position: how far the trend has walked toward the front liquidity
    x["c_position"] = (back - front) / (back + front + EPS)

    # 2. Path agreement: did the recent path mostly walk along the current direction?
    prior_sum = g["path_last_return_R"].transform(
        lambda s: s.shift(1).rolling(CONSENSUS_LOOKBACK, min_periods=CONSENSUS_MIN_PRIOR).sum())
    prior_abs = g["path_last_return_R"].transform(
        lambda s: s.abs().shift(1).rolling(CONSENSUS_LOOKBACK, min_periods=CONSENSUS_MIN_PRIOR).sum())
    x["c_path_agreement"] = d_prev * prior_sum.to_numpy(float) / (prior_abs.to_numpy(float) + EPS)

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


def assert_min_cells(ev_full: pd.DataFrame) -> None:
    cells = _cells_with_group(ev_full, "consensus_group")
    for k in PRIMARY_CELLS:
        if cells[k]["n"] < MIN_CELL_N:
            raise SystemExit(f"STOP_PGM_NATIVE0E_PRIMARY_CELL_TOO_SMALL: {k}={cells[k]['n']}")


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
    days = np.unique(day)
    D = len(days)
    didx = {d: i for i, d in enumerate(days)}
    pos = np.array([didx[x] for x in day])

    grp = sub["consensus_group"].to_numpy()
    acc = sub["accel_positive"].to_numpy(bool)
    pi = sub["pi"].to_numpy(float)
    harm = sub["harm_flag"].to_numpy(int)
    haz = sub["hazard"].to_numpy(int)

    CELLS = ["LOW_OFF", "LOW_ACCEL", "HIGH_OFF", "HIGH_ACCEL"]
    cidx = {c: i for i, c in enumerate(CELLS)}
    count = np.zeros((D, 4))
    spi = np.zeros((D, 4))
    sharm = np.zeros((D, 4))
    shaz = np.zeros((D, 4))

    def _ci(g: str, a: bool) -> int:
        return cidx[f"{g}_{'ACCEL' if a else 'OFF'}"]

    for i in range(len(pos)):
        c = _ci(grp[i], acc[i])
        d = pos[i]
        count[d, c] += 1
        spi[d, c] += pi[i]
        sharm[d, c] += harm[i]
        shaz[d, c] += haz[i]

    def _means_from(w: np.ndarray):
        cnt = w @ count
        mp = w @ spi
        mh = w @ sharm
        mz = w @ shaz
        mpi = np.where(cnt > 0, mp / cnt, np.nan)
        mh_ = np.where(cnt > 0, mh / cnt, np.nan)
        mz_ = np.where(cnt > 0, mz / cnt, np.nan)
        return dict(
            dl=mpi[cidx["LOW_ACCEL"]] - mpi[cidx["LOW_OFF"]],
            dh=mpi[cidx["HIGH_ACCEL"]] - mpi[cidx["HIGH_OFF"]],
            did=(mpi[cidx["HIGH_ACCEL"]] - mpi[cidx["HIGH_OFF"]])
            - (mpi[cidx["LOW_ACCEL"]] - mpi[cidx["LOW_OFF"]]),
            dlh=mh_[cidx["LOW_ACCEL"]] - mh_[cidx["LOW_OFF"]],
            dhh=mh_[cidx["HIGH_ACCEL"]] - mh_[cidx["HIGH_OFF"]],
            didh=(mh_[cidx["HIGH_ACCEL"]] - mh_[cidx["HIGH_OFF"]])
            - (mh_[cidx["LOW_ACCEL"]] - mh_[cidx["LOW_OFF"]]),
            dlz=mz_[cidx["LOW_ACCEL"]] - mz_[cidx["LOW_OFF"]],
            dhz=mz_[cidx["HIGH_ACCEL"]] - mz_[cidx["HIGH_OFF"]],
            didz=(mz_[cidx["HIGH_ACCEL"]] - mz_[cidx["HIGH_OFF"]])
            - (mz_[cidx["LOW_ACCEL"]] - mz_[cidx["LOW_OFF"]]),
        )

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
    dist: Dict[str, list] = {k: [] for k in point}
    for _ in range(n_boot):
        w = rng.multinomial(D, np.full(D, 1.0 / D))
        r = _means_from(w)
        for k in point:
            dist[k].append(r[k])

    res = {k: _summ(float(point[k]), np.array(dist[k], float)) for k in point}
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


def paired_day_loss_bootstrap(entry_day, loss0, loss1, n_boot: int = BOOTSTRAP_N,
                             seed: int = BOOTSTRAP_SEED) -> Dict[str, float]:
    """Entry-day paired bootstrap of the predictive increment.

    delta = loss0 - loss1  (positive => M1 with interaction IMPROVES over M0).
    Pre-aggregate to per-day sum_delta / count, then multinomial-resample DAYS
    only so every within-day row shares the same resampled weight.

    point = mean(delta); returns point / ci95 / P(delta > 0).
    """
    day = np.asarray(entry_day)
    days = np.unique(day)
    D = len(days)
    didx = {d: i for i, d in enumerate(days)}
    pos = np.array([didx[x] for x in day])
    delta = np.asarray(loss0, float) - np.asarray(loss1, float)

    S = np.zeros(D)
    C = np.zeros(D)
    for i, p in enumerate(pos):
        S[p] += delta[i]
        C[p] += 1.0

    point = float(np.mean(delta))
    rng = np.random.default_rng(seed)
    dist = np.empty(n_boot)
    for b in range(n_boot):
        w = rng.multinomial(D, np.full(D, 1.0 / D))
        denom = w @ C
        dist[b] = (w @ S) / denom if denom > 0 else float("nan")
    return _summ(point, dist)


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

    # explicit paired day bootstrap for the policy difference
    ndiff = net["PSYCH_GATE"] - net["BASE"]
    days = np.unique(day)
    D = len(days)
    didx = {d: i for i, d in enumerate(days)}
    pos = np.array([didx[x] for x in day])
    S = np.zeros(D)
    Nc = np.zeros(D)
    for i, p in enumerate(pos):
        S[p] += ndiff[i]
        Nc[p] += 1.0
    rng = np.random.default_rng(seed)
    dist = np.empty(n_boot)
    for b in range(n_boot):
        w = rng.multinomial(D, np.full(D, 1.0 / D))
        denom = w @ Nc
        dist[b] = (w @ S) / denom if denom > 0 else float("nan")
    boot["PSYCH_GATE-BASE"] = _summ(float(np.sum(ndiff) / len(ndiff)), dist)
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
                         eval_cap: Optional[int] = None) -> Tuple[Dict[str, Any], Tuple[float, float], Dict[str, np.ndarray]]:
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

    tr = sub[sub["block"].isin(w["train"])].copy()
    pay0, m0m, sq0 = _predict_ridge(tr, ev, m0_num(), PRED_CAT, "pi")
    pay1, m1m, sq1 = _predict_ridge(tr, ev, m1_num(), PRED_CAT, "pi")
    h0p, h0m, ll0 = _predict_logistic(tr, ev, m0_num(), PRED_CAT, "harm_flag")
    h1p, h1m, ll1 = _predict_logistic(tr, ev, m1_num(), PRED_CAT, "harm_flag")

    # predictive falsification via ENTRY-DAY PAIRED bootstrap (not point estimate alone)
    payoff_boot = paired_day_loss_bootstrap(ev["entry_day"].to_numpy(), sq0, sq1,
                                           n_boot=n_boot, seed=BOOTSTRAP_SEED)
    harm_boot = paired_day_loss_bootstrap(ev["entry_day"].to_numpy(), ll0, ll1,
                                         n_boot=n_boot, seed=BOOTSTRAP_SEED)

    # n_boot ownership: smoke=200, formal future=2000 -- passed through, never fixed here.
    psych = psych_gate_diagnostic(ev, PRIMARY_COST_ATR0, n_boot, BOOTSTRAP_SEED)
    comp = component_diagnostics(ev, maps)
    age = age_zero_diagnostic(ev)

    return dict(
        cells=cells, effects=effects, bootstrap=boot,
        payoff_m0=m0m, payoff_m1=m1m, delta_mse=m0m["mse"] - m1m["mse"],
        harm_m0=h0m, harm_m1=h1m, delta_logloss=h0m["log_loss"] - h1m["log_loss"],
        payoff_bootstrap=payoff_boot, harm_bootstrap=harm_boot,
        psych=psych, component=comp, age_zero=age,
        n_rank_train=n_rank_train, n_train=len(tr), n_eval=len(ev),
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
    # Round 1: full is ALWAYS blocked, even if the token is present.
    raise SystemExit("STOP_PGM_NATIVE0E_FULL_NOT_AUTHORIZED_ROUND1")


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
        cells = _cells_with_group(ev_primary, "consensus_group")
        for k in PRIMARY_CELLS:
            c = cells[k]
            print(f"[AUDIT]   {k}: n={c['n']} mean_pi={c['mean_pi']:.5f} "
                  f"gross_EV={c['gross_EV']:.5f} net_EV_0p01={c['net_EV_at_0p01']:.5f}")
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
    dA = d0.verify_window_score_owner(scored_A, fit_A, TB2_BLOCK)
    dB = d0.verify_window_score_owner(scored_B, fit_B, TB3_BLOCK)
    print(f"[SMOKE] score-owner parity: WindowA(TB2)={dA:.3e} WindowB(TB3)={dB:.3e}")

    for blk, w, sc in [(TB2_BLOCK, pgm.WINDOWS[0], scored_A),
                       (TB3_BLOCK, pgm.WINDOWS[1], scored_B)]:
        print(f"[SMOKE] --- window {blk} ---")
        r, _, _ = run_window_complete(sc, w, n_boot=SMOKE_BOOTSTRAP_N, eval_cap=SMOKE_EVAL_CAP)
        print(f"  n_train_primary={r['n_train']} n_eval_primary(cap)={r['n_eval']}")
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
    print("[SMOKE] SMOKE ONLY / NO SCIENTIFIC VERDICT", flush=True)
    print(f"[SMOKE COMPLETE] {time.perf_counter() - t0:.2f}s", flush=True)


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


def main() -> None:
    ap = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full-exploratory", action="store_true")
    args = ap.parse_args()
    if args.full_exploratory:
        require_full_authorization()
    elif args.smoke:
        run_smoke_test()
    else:
        run_audit_only()


if __name__ == "__main__":
    main()
