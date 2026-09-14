"""DYNAMIC-PGM-1A — Within-Episode One-Step Transition Kernel（基于 0d3db40）

===========================================================
只研究 episode 内部的 5m 状态转移，不碰 terminal reset。
===========================================================
已有：
    P(Y_{t+1} | S_t)   —— endpoint / hazard 模型（PGM-BAR-0 / MARKET-STATE-1）
这一轮研究：
    P(S_{t+1} | S_t, H_{t+1}=0)   —— 非终止时，当前 Market State 能否预测
                                    下一根之后 Market State 怎么变。

完整分解留待 1B/1C：
    P(Y_{t+1}, S_{t+1} | S_t) = P(Y_{t+1}|S_t) P(S_{t+1}|S_t, Y_{t+1})

不预测 35 个 state feature（很多是确定性关系），只预测真正的
state innovations Z_{t+1}，然后用确定公式 S_{t+1}=F(S_t, Z_{t+1})：

连续（9）：
    z_d_up           = d_U,t+1 - d_U,t            (=> r = -Δd_U)
    z_log1p_dmfe     = log1p(MFE_{t+1}-MFE_t)
    z_log1p_dmae     = log1p(MAE_{t+1}-MAE_t)
    z_delta_dcr      = DCR_{t+1}-DCR_t
    z_log1p_next_range = log1p(Range_{t+1})
    z_delta_upper_newest_resid / z_delta_upper_count
    z_delta_lower_newest_resid / z_delta_lower_count

离散（4 类）：
    z_agezero_code = 1[upper newest age=0] + 2[lower newest age=0]

模型：
    K0 UNCONDITIONAL  不看 state，只学平均 transition 分布
    K1 STATE          OBSERVED-STATE-v1
    K2 STATE+LAG1     K1 + 上一根 dynamic state（15 个 lag1_*）

主检验：K1-K0 是否有稳定增量（state 能否预测状态演化）
次检验：K2-K1 是否还有 lag1 增量（当前 state 是否仍遗留一阶历史依赖）

连续：Z^c|S_t ~ N(mu(S_t), Sigma) 多输出 Ridge + LedoitWolf 完整 9x9 协方差
离散：Softmax(W phi(S_t)+b) 4-class Logistic
第一版先 conditional independence Z^c ⊥ Z^d | S_t。

不进入 terminal reset / recursive rollout / Dynamic PGM-1B / 1C。
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

# Pin numerical BLAS threads to 1 BEFORE numpy/sklearn import. Linear dynamic
# PGM fits are tiny (tens of MB of sufficient statistics); multi-threaded
# Accelerate/OpenBLAS only inflates transient peak RSS via per-thread workspaces
# and never helps here. Single thread keeps the memory envelope predictable.
for _bt in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_bt, "1")


def _peak(tag):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"[MEM] {tag} peak_rss={rss/1e6:.1f}MB", flush=True)

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.liquidity_oracle_atlas.experiment_pgm_bar0_path_smc_v1 import (  # noqa: E402
    M2_NUM, CAT,
)
from research.liquidity_oracle_atlas.experiment_market_state1_1_state_closure_v1 import (  # noqa: E402
    COMPACT_PROV, TEMPO, WINDOWS,
)
from research.liquidity_oracle_atlas.experiment_local_liquidity_transition_v0 import (  # noqa: E402
    CACHE, OUT,
)

BASE_SHA = "0d3db4057f40607fe16f463d9be78112437bb29e"
EXPECTED_TRANSITIONS = 321727
BOOTSTRAP_REPS = 1000

OBS_STATE_NUM = list(M2_NUM) + list(COMPACT_PROV) + list(TEMPO)   # 35
OBS_STATE_CAT = list(CAT)                                         # ['prev_event_mask']

# K2 lag block（上一根 dynamic state，不含 tempo）
LAG_BASE = [
    "cur_up_distance_R", "cur_down_distance_R", "cur_log_ratio",
    "path_total_variation_R", "path_max_up_excursion_R",
    "path_max_down_excursion_R", "path_direction_change_rate",
    "path_last_return_R", "path_current_bar_range_R",
] + list(COMPACT_PROV)   # 9 + 6 = 15

CONT_Z = [
    "z_d_up", "z_log1p_dmfe", "z_log1p_dmae", "z_delta_dcr",
    "z_log1p_next_range", "z_delta_upper_newest_resid", "z_delta_upper_count",
    "z_delta_lower_newest_resid", "z_delta_lower_count",
]
DISC_Z = "z_agezero_code"

# Source columns consumed from the enriched frame by build_transition_sample
# (both `cur` and the shifted `nxt`). Kept at float64 so the exact deterministic
# invariants (TV update / last-return / monotonic excursion) retain precision.
BUILD_SRC_COLS = [
    "cur_width_R", "cur_up_distance_R", "path_max_up_excursion_R",
    "path_max_down_excursion_R", "path_direction_change_rate",
    "path_current_bar_range_R", "upper_newest_log_age_residual",
    "lower_newest_log_age_residual", "upper_active_identity_count_delta",
    "lower_active_identity_count_delta", "upper_current_newest_age_zero",
    "lower_current_newest_age_zero", "path_total_variation_R",
    "path_last_return_R",
]

BLOCKS = {
    "Location": dict(cont=["z_d_up"], disc=[]),
    "Path": dict(cont=["z_log1p_dmfe", "z_log1p_dmae", "z_delta_dcr",
                       "z_log1p_next_range"], disc=[]),
    "LiquidityComposition": dict(
        cont=["z_delta_upper_newest_resid", "z_delta_upper_count",
              "z_delta_lower_newest_resid", "z_delta_lower_count"],
        disc=["z_agezero_code"]),
}


# ===========================================================================
# transition sample construction + deterministic invariants
# ===========================================================================
def build_transition_sample(df):
    df = df.sort_values(["episode_id", "bar_t"], kind="stable").reset_index(
        drop=True)
    g = df.groupby("episode_id", sort=False)
    next_ep = g["episode_id"].shift(-1)
    next_bar = g["bar_t"].shift(-1)
    keep = (next_ep.eq(df["episode_id"])
            & next_bar.eq(df["bar_t"] + 1)
            & df["hazard"].eq(0))

    cur = df.loc[keep].copy()
    nxt = g.shift(-1).loc[keep].copy()

    if len(cur) != EXPECTED_TRANSITIONS:
        raise SystemExit(
            f"STOP_DYNAMIC_PGM1A_TRANSITION_COUNT_FAIL: {len(cur)}")

    # ---- row-level invariants
    if not (next_ep.loc[keep].to_numpy() == cur["episode_id"].to_numpy()).all():
        raise SystemExit("STOP_DYNAMIC_PGM1A_NEXT_EPISODE_MISMATCH")
    if not (next_bar.loc[keep].to_numpy() == cur["bar_t"].to_numpy() + 1).all():
        raise SystemExit("STOP_DYNAMIC_PGM1A_NEXT_BAR_MISMATCH")

    # ---- width exact constant within episode
    if not np.allclose(nxt["cur_width_R"].to_numpy(float),
                       cur["cur_width_R"].to_numpy(float), atol=1e-10):
        raise SystemExit("STOP_DYNAMIC_PGM1A_WIDTH_NONCONSTANT")

    cur["z_d_up"] = (nxt["cur_up_distance_R"].to_numpy(float)
                     - cur["cur_up_distance_R"].to_numpy(float))

    dmfe = (nxt["path_max_up_excursion_R"].to_numpy(float)
            - cur["path_max_up_excursion_R"].to_numpy(float))
    dmae = (nxt["path_max_down_excursion_R"].to_numpy(float)
            - cur["path_max_down_excursion_R"].to_numpy(float))
    if dmfe.min() < -1e-9 or dmae.min() < -1e-9:
        raise SystemExit("STOP_DYNAMIC_PGM1A_NONMONOTONE_EXCURSION")
    cur["z_log1p_dmfe"] = np.log1p(np.maximum(dmfe, 0.0))
    cur["z_log1p_dmae"] = np.log1p(np.maximum(dmae, 0.0))
    cur["z_delta_dcr"] = (nxt["path_direction_change_rate"].to_numpy(float)
                          - cur["path_direction_change_rate"].to_numpy(float))
    cur["z_log1p_next_range"] = np.log1p(
        np.maximum(nxt["path_current_bar_range_R"].to_numpy(float), 0.0))

    for side in ("upper", "lower"):
        cur[f"z_delta_{side}_newest_resid"] = (
            nxt[f"{side}_newest_log_age_residual"].to_numpy(float)
            - cur[f"{side}_newest_log_age_residual"].to_numpy(float))
        cur[f"z_delta_{side}_count"] = (
            nxt[f"{side}_active_identity_count_delta"].to_numpy(float)
            - cur[f"{side}_active_identity_count_delta"].to_numpy(float))

    up0 = nxt["upper_current_newest_age_zero"].to_numpy(np.int64)
    lo0 = nxt["lower_current_newest_age_zero"].to_numpy(np.int64)
    cur["z_agezero_code"] = up0 + 2 * lo0

    # ---- TV update exact: TV_{t+1}-TV_t = |Δd_U|
    tv_delta = (nxt["path_total_variation_R"].to_numpy(float)
                - cur["path_total_variation_R"].to_numpy(float))
    if not np.allclose(tv_delta, np.abs(cur["z_d_up"].to_numpy(float)),
                       atol=1e-8):
        raise SystemExit("STOP_DYNAMIC_PGM1A_TV_UPDATE_FAIL")

    # ---- last_return = -Δd_U
    if not np.allclose(nxt["path_last_return_R"].to_numpy(float),
                       -cur["z_d_up"].to_numpy(float), atol=1e-8):
        raise SystemExit("STOP_DYNAMIC_PGM1A_LASTRETURN_FAIL")

    # ---- age-zero code range
    if not set(np.unique(cur["z_agezero_code"])).issubset({0, 1, 2, 3}):
        raise SystemExit("STOP_DYNAMIC_PGM1A_AGEZERO_CODE_RANGE")

    # slim the shifted `nxt` to only the source columns actually consumed by the
    # invariants / z_* computations -> avoids holding the full (86-col) shift
    # copy in memory through the rest of the pipeline.
    nxt = nxt[BUILD_SRC_COLS].copy()
    return cur, nxt


def add_lag1(df):
    df = df.sort_values(["episode_id", "bar_t"], kind="stable").copy()
    g = df.groupby("episode_id", sort=False)
    for c in LAG_BASE:
        df[f"lag1_{c}"] = g[c].shift(1)
    df["lag1_available"] = (g.cumcount() > 0).astype(float)
    return df


# ===========================================================================
# transition heads
# ===========================================================================
class GaussianTransitionHead:
    """Multi-output Gaussian transition head.

    Continuous transition:  Z_{t+1} = X B + epsilon,  epsilon ~ N(0, Sigma).
    Fitted via sufficient statistics rather than a full sklearn Ridge object so
    that the whole pipeline never has to retain the (n x p) standardized design
    matrix beyond what NLL evaluation needs. The solve is exactly Ridge(alpha)
    with fit_intercept=True (sklearn centers X and Z, solves the normal
    equations, then restores the intercept), so the point predictions, residual
    covariance (LedoitWolf) and NLL are numerically identical to the original.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, Z):
        X = np.asarray(X, dtype=np.float64)
        Z = np.asarray(Z, dtype=np.float64)
        p, q = X.shape[1], Z.shape[1]
        # sufficient statistics (p x p and p x q) -- a few thousand floats even
        # for p=50; no need to keep the full n x p matrix around.
        xm = X.mean(axis=0)
        zm = Z.mean(axis=0)
        Xc = X - xm
        Zc = Z - zm
        xtx = Xc.T @ Xc
        xtz = Xc.T @ Zc
        A = xtx + self.alpha * np.eye(p)
        B = np.linalg.solve(A, xtz)
        self.B = B
        self.intercept = zm - xm @ B
        self.mean = X @ B + self.intercept
        resid = Z - self.mean
        lw = LedoitWolf().fit(resid)
        self.cov = lw.covariance_
        self.chol = np.linalg.cholesky(self.cov)
        self.logdet = 2.0 * np.log(np.diag(self.chol)).sum()
        self.k = q
        return self

    def nll_per_row(self, X, Z):
        X = np.asarray(X, dtype=np.float64)
        Z = np.asarray(Z, dtype=np.float64)
        err = Z - (X @ self.B + self.intercept)
        solved = np.linalg.solve(self.chol, err.T)
        quad = np.sum(solved * solved, axis=0)
        return 0.5 * (self.k * np.log(2.0 * np.pi) + self.logdet + quad)


class ConstantGaussianHead:
    def fit(self, Z):
        self.mean = Z.mean(axis=0)
        resid = Z - self.mean
        lw = LedoitWolf().fit(resid)
        self.cov = lw.covariance_
        self.chol = np.linalg.cholesky(self.cov)
        self.logdet = 2.0 * np.log(np.diag(self.chol)).sum()
        self.k = Z.shape[1]
        return self

    def nll_per_row(self, Z):
        err = Z - self.mean
        solved = np.linalg.solve(self.chol, err.T)
        quad = np.sum(solved * solved, axis=0)
        return 0.5 * (self.k * np.log(2.0 * np.pi) + self.logdet + quad)


def make_transformer(num_cols, cat_cols):
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler())])
    cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])


def fit_state_heads(Xtr, Xev, Zc_tr, Zc_ev, yd_tr, yd_ev):
    """K1/K2: continuous Gaussian head + discrete 4-class Logistic."""
    head = GaussianTransitionHead(alpha=1.0).fit(Xtr, Zc_tr)
    cont_tr = head.nll_per_row(Xtr, Zc_tr)
    cont_ev = head.nll_per_row(Xev, Zc_ev)

    if len(np.unique(yd_tr)) != 4:
        raise SystemExit("STOP_DYNAMIC_PGM1A_DISCRETE_CLASS_MISSING")

    disc = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                             max_iter=3000)
    disc.fit(Xtr, yd_tr)
    p_tr = disc.predict_proba(Xtr)[np.arange(len(yd_tr)), yd_tr]
    p_ev = disc.predict_proba(Xev)[np.arange(len(yd_ev)), yd_ev]
    disc_tr = -np.log(np.clip(p_tr, 1e-12, 1.0))
    disc_ev = -np.log(np.clip(p_ev, 1e-12, 1.0))
    return dict(cont_tr=cont_tr, cont_ev=cont_ev, disc_tr=disc_tr,
                disc_ev=disc_ev, head=head, disc=disc,
                n_params=int(Xtr.shape[1] * Zc_tr.shape[1] + Zc_tr.shape[1]
                             + Xtr.shape[1] * 4 + 4))


def fit_constant_heads(Zc_tr, Zc_ev, yd_tr, yd_ev):
    """K0: constant Gaussian + Laplace 4-class."""
    head = ConstantGaussianHead().fit(Zc_tr)
    cont_tr = head.nll_per_row(Zc_tr)
    cont_ev = head.nll_per_row(Zc_ev)
    counts = np.bincount(yd_tr, minlength=4).astype(float)
    prob = (counts + 0.5) / (len(yd_tr) + 2.0)
    disc_tr = -np.log(prob[yd_tr])
    disc_ev = -np.log(prob[yd_ev])
    return dict(cont_tr=cont_tr, cont_ev=cont_ev, disc_tr=disc_tr,
                disc_ev=disc_ev, head=head,
                n_params=int(Zc_tr.shape[1] + 4))


# ===========================================================================
# cluster bootstrap (by episode_start_trading_day)
# ===========================================================================
def boot_cluster(delta_per_row, day_per_row, eid_per_row, seed):
    df = pd.DataFrame({"d": delta_per_row, "day": day_per_row,
                       "eid": eid_per_row})
    ep = df.groupby("eid").agg(d=("d", "mean"), day=("day", "first"))
    rng = np.random.default_rng(seed)
    days = ep["day"].to_numpy()
    uniq_days = np.unique(days)
    ests = np.empty(BOOTSTRAP_REPS)
    for b in range(BOOTSTRAP_REPS):
        sel = rng.choice(uniq_days, size=len(uniq_days), replace=True)
        mask = ep["day"].isin(sel).to_numpy()
        ests[b] = ep.loc[mask, "d"].mean()
    point = float(ep["d"].mean())
    return float(np.percentile(ests, 2.5)), float(np.percentile(ests, 97.5)), point


def run_single_window(w, data_path):
    """Fit K0/K1/K2 for one window and return all metrics as a plain dict.

    Designed to run inside its own subprocess so that, after it returns, the OS
    (not just the Python GC) reclaims the memory -- pandas / sklearn allocations
    do not reliably return RSS to macOS even after `del` + gc.collect().

    Execution engine only; the mathematical model is identical to the original:
      * K1 / K2 share ONE preprocessing pass (combined ColumnTransformer) and
        K1 is a contiguous prefix view of the K2 design matrix -> no second
        transform, no duplicated n x p matrix.
      * Gaussian heads use sufficient-statistics solves (== Ridge(alpha=1.0,
        fit_intercept=True)); residual covariance stays LedoitWolf on the small
        n x q residual matrix.
    """
    t_w = time.perf_counter()
    cur = pd.read_parquet(data_path)
    _peak(f"{w['name']} loaded cur")

    tr = cur[cur["block"].isin(w["train"])].reset_index(drop=True)
    ev = cur[cur["block"] == w["eval"]].reset_index(drop=True)
    if not len(tr) or not len(ev):
        raise SystemExit(f"STOP_DYNAMIC_PGM1A_EMPTY_SPLIT: {w['name']}")

    Zc_tr = tr[CONT_Z].to_numpy(np.float64)
    Zc_ev = ev[CONT_Z].to_numpy(np.float64)
    yd_tr = tr[DISC_Z].to_numpy(np.int64)
    yd_ev = ev[DISC_Z].to_numpy(np.int64)
    sym_ev = ev["symbol"].to_numpy()
    day_ev = ev["episode_start_day"].to_numpy()
    eid_ev = ev["episode_id"].to_numpy()

    model_metrics, opt_rows, block_rows, target_rows = [], [], [], []

    # ---------------- K0: constant Gaussian + Laplace 4-class
    t_k0 = time.perf_counter()
    k0 = fit_constant_heads(Zc_tr, Zc_ev, yd_tr, yd_ev)
    opt_rows.append(dict(window=w["name"], model="K0_UNCONDITIONAL",
                         n_params=k0["n_params"], success=True,
                         elapsed_seconds=round(time.perf_counter() - t_k0, 3)))
    j_k0_tr = k0["cont_tr"] + k0["disc_tr"]
    j_k0_ev = k0["cont_ev"] + k0["disc_ev"]
    model_metrics.append(dict(window=w["name"], model="K0_UNCONDITIONAL",
                             mean_joint_nll=float(np.mean(j_k0_ev)),
                             mean_cont_nll=float(np.mean(k0["cont_ev"])),
                             mean_disc_nll=float(np.mean(k0["disc_ev"])),
                             n_rows=int(len(ev))))

    # ---------------- shared preprocessing (K1 + K2 in one pass)
    # Order: [cat | OBS_STATE_NUM | lag1_* + lag1_available]. K1 is then the
    # contiguous prefix [cat | OBS_STATE_NUM]; K2 is the full matrix. Column
    # permutation is mathematically irrelevant for Ridge / Logistic.
    t_k1 = time.perf_counter()
    lag_cols = [f"lag1_{c}" for c in LAG_BASE] + ["lag1_available"]
    ct = ColumnTransformer([
        ("cat", Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))]),
         list(OBS_STATE_CAT)),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]),
         list(OBS_STATE_NUM) + lag_cols),
    ])
    Xtr2 = ct.fit_transform(tr).astype(np.float32)
    Xev2 = ct.transform(ev).astype(np.float32)
    k1_dim = Xtr2.shape[1] - len(lag_cols)
    Xtr1 = Xtr2[:, :k1_dim]      # contiguous VIEW, no copy
    Xev1 = Xev2[:, :k1_dim]
    _peak(f"{w['name']} after shared transform")

    # ---------------- K1: OBSERVED-STATE-v1
    k1 = fit_state_heads(Xtr1, Xev1, Zc_tr, Zc_ev, yd_tr, yd_ev)
    _peak(f"{w['name']} after K1 transform+fit")
    opt_rows.append(dict(window=w["name"], model="K1_STATE",
                         n_params=k1["n_params"], success=True,
                         elapsed_seconds=round(time.perf_counter() - t_k1, 3)))
    j_k1_tr = k1["cont_tr"] + k1["disc_tr"]
    j_k1_ev = k1["cont_ev"] + k1["disc_ev"]
    model_metrics.append(dict(window=w["name"], model="K1_STATE",
                             mean_joint_nll=float(np.mean(j_k1_ev)),
                             mean_cont_nll=float(np.mean(k1["cont_ev"])),
                             mean_disc_nll=float(np.mean(k1["disc_ev"])),
                             n_rows=int(len(ev))))

    # ---------------- covariance PD audit (reuse K1 design)
    head_cov = GaussianTransitionHead(alpha=1.0).fit(Xtr1, Zc_tr)
    e_cov = np.linalg.eigvalsh(head_cov.cov)
    cov_audit = {w["name"]: dict(min_eig=float(e_cov.min()),
                                 pd=bool(e_cov.min() > 0))}

    # ---------------- K2: K1 + lag1
    t_k2 = time.perf_counter()
    k2 = fit_state_heads(Xtr2, Xev2, Zc_tr, Zc_ev, yd_tr, yd_ev)
    _peak(f"{w['name']} after K2 transform+fit")
    opt_rows.append(dict(window=w["name"], model="K2_STATE_LAG1",
                         n_params=k2["n_params"], success=True,
                         elapsed_seconds=round(time.perf_counter() - t_k2, 3)))
    j_k2_tr = k2["cont_tr"] + k2["disc_tr"]
    j_k2_ev = k2["cont_ev"] + k2["disc_ev"]
    model_metrics.append(dict(window=w["name"], model="K2_STATE_LAG1",
                             mean_joint_nll=float(np.mean(j_k2_ev)),
                             mean_cont_nll=float(np.mean(k2["cont_ev"])),
                             mean_disc_nll=float(np.mean(k2["disc_ev"])),
                             n_rows=int(len(ev))))
    del k2, ct

    if not (len(j_k0_ev) == len(j_k1_ev) == len(j_k2_ev)):
        raise SystemExit("STOP_DYNAMIC_PGM1A_SAMPLE_MISMATCH")

    # ---------------- bootstrap K1-K0, K2-K1 (vectorized by-symbol)
    boots, bysym = [], []
    for label, hi, lo in [("K1-K0", j_k1_ev, j_k0_ev),
                          ("K2-K1", j_k2_ev, j_k1_ev)]:
        delta = hi - lo
        t_b = time.perf_counter()
        print(f"[STAGE] window {w['name']} bootstrap {label} start "
              f"elapsed={round(t_b - t_w, 1)}", flush=True)
        lo_ci, hi_ci, point = boot_cluster(
            delta, day_ev, eid_ev,
            w["seed"] + (0 if label == "K1-K0" else 1))
        print(f"[STAGE] window {w['name']} bootstrap {label} done "
              f"took={round(time.perf_counter() - t_b, 2)}", flush=True)
        codes, sym_names = pd.factorize(sym_ev)
        cnt = np.bincount(codes, minlength=len(sym_names))
        for s in range(len(sym_names)):
            if cnt[s] == 0:
                continue
            d = delta[codes == s]
            bysym.append(dict(window=w["name"], comparison=label,
                              symbol=sym_names[s], n=int(cnt[s]),
                              mean_delta=float(d.mean()),
                              n_negative=int((d < 0).sum()),
                              n_positive=int((d > 0).sum())))
        boots.append(dict(window=w["name"], comparison=label,
                          delta_sample_mean=point, ci_lo=lo_ci, ci_hi=hi_ci,
                          verdict=("CI_below_zero" if hi_ci < 0
                                   else "CI_above_zero" if lo_ci > 0
                                   else "CI_contains_zero")))

    # ---------------- block contribution (eval, reuse K1 design)
    for bname, bdef in BLOCKS.items():
        bc = bdef["cont"]
        cols = [CONT_Z.index(c) for c in bc]
        bhead = GaussianTransitionHead(alpha=1.0).fit(Xtr1, Zc_tr[:, cols])
        bcont_k1_ev = bhead.nll_per_row(Xev1, Zc_ev[:, cols])
        bhead0 = ConstantGaussianHead().fit(Zc_tr[:, cols])
        bcont_k0_ev = bhead0.nll_per_row(Zc_ev[:, cols])
        bjoint_k1 = bcont_k1_ev
        bjoint_k0 = bcont_k0_ev
        if bdef["disc"]:
            bjoint_k1 = bcont_k1_ev + k1["disc_ev"]
            bjoint_k0 = bcont_k0_ev + k0["disc_ev"]
        block_rows.append(dict(
            window=w["name"], block=bname,
            mean_joint_k0=float(np.mean(bjoint_k0)),
            mean_joint_k1=float(np.mean(bjoint_k1)),
            delta_k1_minus_k0=float(np.mean(bjoint_k1) - np.mean(bjoint_k0))))

    # ---------------- per-target (eval, reuse K1 design)
    for i, c in enumerate(CONT_Z):
        h1 = GaussianTransitionHead(alpha=1.0).fit(Xtr1, Zc_tr[:, i:i + 1])
        n1 = h1.nll_per_row(Xev1, Zc_ev[:, i:i + 1])
        h0 = ConstantGaussianHead().fit(Zc_tr[:, i:i + 1])
        n0 = h0.nll_per_row(Zc_ev[:, i:i + 1])
        target_rows.append(dict(window=w["name"], target=c, kind="continuous",
                                mean_nll_k0=float(np.mean(n0)),
                                mean_nll_k1=float(np.mean(n1)),
                                delta_k1_minus_k0=float(np.mean(n1) - np.mean(n0))))
    target_rows.append(dict(window=w["name"], target=DISC_Z, kind="discrete_4class",
                            mean_nll_k0=float(np.mean(k0["disc_ev"])),
                            mean_nll_k1=float(np.mean(k1["disc_ev"])),
                            delta_k1_minus_k0=float(np.mean(k1["disc_ev"])
                                                    - np.mean(k0["disc_ev"]))))

    # free everything before process exit -> OS reclaims
    del Xtr1, Xev1, Xtr2, Xev2, k1, k0, tr, ev
    del Zc_tr, Zc_ev, yd_tr, yd_ev
    del j_k0_tr, j_k0_ev, j_k1_tr, j_k1_ev, j_k2_tr, j_k2_ev

    print(f"[STAGE] window {w['name']} done "
          f"took={round(time.perf_counter() - t_w, 2)}s", flush=True)
    return dict(window=w["name"], model_metrics=model_metrics,
                opt_rows=opt_rows, boots=boots, bysym=bysym,
                block_rows=block_rows, target_rows=target_rows,
                cov_audit=cov_audit)


def main():
    t_total = time.perf_counter()
    timing = {}

    t0 = time.perf_counter()
    keep_cols = (list(OBS_STATE_NUM) + list(OBS_STATE_CAT) + list(LAG_BASE)
                 + BUILD_SRC_COLS
                 + ["hazard", "block", "symbol", "episode_id",
                    "episode_start_day", "bar_t"])
    # dedupe while preserving order (LAG_BASE/COMPACT_PROV overlap OBS_STATE_NUM)
    keep_cols = list(dict.fromkeys(keep_cols))
    # read ONLY the columns the transition kernel consumes (avoids loading the
    # full ~700MB enriched frame) -> cuts the read-phase peak RSS by ~700MB.
    # Build-source columns stay float64 so the exact deterministic invariants
    # (TV update, last-return, monotonicity) keep their required precision;
    # float32 halving happens later in the slim step after invariants pass.
    df = pd.read_parquet(
        CACHE / "market_state1_samples.parquet", columns=keep_cols)
    _peak("after read+prune (slim, native dtype)")
    # keep build-source columns float64 (exact invariants need their precision);
    # halve the rest (OBS_STATE_NUM / LAG_BASE / etc.) to float32 up front.
    build_set = set(BUILD_SRC_COLS)
    for c in df.columns:
        if c not in build_set and str(df[c].dtype).startswith("float64"):
            df[c] = df[c].astype(np.float32)
    cur, nxt = build_transition_sample(df)
    _peak("after build_transition_sample")
    cur = add_lag1(cur)
    _peak("after add_lag1")
    # df is no longer needed once the transition sample is built -> free it to
    # keep steady RSS low through the per-window fits.
    del df
    timing["build_transition_seconds"] = round(time.perf_counter() - t0, 2)
    print(f"[STAGE] build done n={len(cur)}", flush=True)

    # ---------------- sample audit / hash
    hash_cols = CONT_Z + [DISC_Z]
    h = hashlib.sha256(
        pd.util.hash_pandas_object(cur[hash_cols], index=False)
        .values.tobytes()).hexdigest()
    sample_audit = dict(
        n_transition_rows=int(len(cur)),
        expected_transitions=EXPECTED_TRANSITIONS,
        count_ok=bool(len(cur) == EXPECTED_TRANSITIONS),
        z_feature_hash=h,
        blocks={b: int((cur["block"] == b).sum())
                for b in ["TB1", "TB2", "TB3"]},
        n_episodes=int(cur["episode_id"].nunique()),
        agezero_code_distribution={int(k): int(v) for k, v in
                                  cur[DISC_Z].value_counts().sort_index().items()},
    )

    # ---------------- transition invariants
    invariants = dict(
        width_exact_constant=bool(np.allclose(
            nxt["cur_width_R"].to_numpy(float),
            cur["cur_width_R"].to_numpy(float), atol=1e-10)),
        tv_update_exact=bool(np.allclose(
            (nxt["path_total_variation_R"].to_numpy(float)
             - cur["path_total_variation_R"].to_numpy(float)),
            np.abs(cur["z_d_up"].to_numpy(float)), atol=1e-8)),
        last_return_exact=bool(np.allclose(
            nxt["path_last_return_R"].to_numpy(float),
            -cur["z_d_up"].to_numpy(float), atol=1e-8)),
        mfe_nonnegative=bool((nxt["path_max_up_excursion_R"].to_numpy(float)
                              - cur["path_max_up_excursion_R"].to_numpy(float)
                              ).min() >= -1e-9),
        mae_nonnegative=bool((nxt["path_max_down_excursion_R"].to_numpy(float)
                              - cur["path_max_down_excursion_R"].to_numpy(float)
                              ).min() >= -1e-9),
        agezero_code_in_0_3=bool(set(np.unique(cur[DISC_Z])).issubset(
            {0, 1, 2, 3})),
        lag1_from_past_only=bool(
            (cur.loc[cur["lag1_available"] == 0,
                     [f"lag1_{c}" for c in LAG_BASE]].isna().all(axis=None))
            and (cur.loc[cur["lag1_available"] == 1,
                         [f"lag1_{c}" for c in LAG_BASE]].notna().any(axis=1)
                 .all())),
        no_tb4=bool("TB4" not in set(cur["block"].unique())),
    )
    if not all(invariants.values()):
        raise SystemExit(
            f"STOP_DYNAMIC_PGM1A_INVARIANT_FAIL: {invariants}")

    # nxt is only consumed by the invariants above -> free it now (it is a
    # full-width shifted copy) to keep steady RSS low before the slim step.
    del nxt

    # ---------------- write cache (gitignored) + free heavy temporaries
    needed_cols = (list(OBS_STATE_NUM) + list(OBS_STATE_CAT)
                   + [f"lag1_{c}" for c in LAG_BASE] + ["lag1_available"]
                   + list(CONT_Z) + [DISC_Z]
                   + ["episode_id", "symbol", "block", "episode_start_day"])
    # halve float memory IN PLACE first (each float64 column buffer is released
    # as it is rewritten), then select the slim subset. This avoids the old
    # float64 cur and the new float32 cur coexisting at once.
    for c in cur.columns:
        if str(cur[c].dtype).startswith("float64"):
            cur[c] = cur[c].astype(np.float32)
    cur = cur[needed_cols].copy()
    cur.to_parquet(CACHE / "dynamic_pgm1a_transitions.parquet", index=False)
    _peak("after slim cur")

    # ---------------- Z matrices
    Zc = cur[CONT_Z].to_numpy(np.float32)
    yd = cur[DISC_Z].to_numpy(np.int64)
    day = cur["episode_start_day"].to_numpy()
    eid = cur["episode_id"].to_numpy()

    # K2 numeric columns
    k2_num = OBS_STATE_NUM + [f"lag1_{c}" for c in LAG_BASE] + ["lag1_available"]
    print(f"[STAGE] slimmed cur n={len(cur)} cols={cur.shape[1]} "
          f"elapsed={round(time.perf_counter()-t_total,1)}s", flush=True)

    # ---------------- per-window fit (one subprocess per window)
    # Each window is fitted in its own process so the OS -- not just the Python
    # GC -- reclaims memory between windows (pandas/sklearn allocations do not
    # reliably return RSS to macOS even after del + gc). With BLAS pinned to 1
    # thread, per-window peak stays in the low hundreds of MB; the parent only
    # orchestrates and assembles the final summary.
    data_path = CACHE / "dynamic_pgm1a_transitions.parquet"
    win_env = dict(os.environ)
    for _bt in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        win_env[_bt] = "1"

    model_metrics, opt_rows, boots, bysym = [], [], [], []
    block_rows, target_rows = [], []
    cov_audit = {}
    for w in WINDOWS:
        result_path = CACHE / f"_window_{w['name']}.json"
        cmd = [sys.executable, str(Path(__file__)),
               "--window-json", json.dumps(w),
               "--data", str(data_path), "--result", str(result_path)]
        print(f"[STAGE] window {w['name']} subprocess start", flush=True)
        t_w = time.perf_counter()
        r = subprocess.run(cmd, env=win_env, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout, file=sys.stderr)
            print(r.stderr, file=sys.stderr)
            raise SystemExit(
                f"STOP_DYNAMIC_PGM1A_WINDOW_FAIL: {w['name']} rc={r.returncode}")
        res = json.loads(Path(result_path).read_text())
        model_metrics += res["model_metrics"]
        opt_rows += res["opt_rows"]
        boots += res["boots"]
        bysym += res["bysym"]
        block_rows += res["block_rows"]
        target_rows += res["target_rows"]
        cov_audit.update(res["cov_audit"])
        print(f"[STAGE] window {w['name']} subprocess done "
              f"took={round(time.perf_counter() - t_w, 2)}s", flush=True)

    # ---------------- verdict
    def sym_neg(wname, label):
        sub = pd.DataFrame(bysym)
        sub = sub[(sub["window"] == wname) & (sub["comparison"] == label)]
        return int((sub["mean_delta"] < 0).sum())

    verdict = {}
    # Gate A: transition signal
    wsA = {w["name"]: {} for w in WINDOWS}
    for w in WINDOWS:
        row = next(b for b in boots if b["window"] == w["name"]
                   and b["comparison"] == "K1-K0")
        wsA[w["name"]] = dict(ci_hi=row["ci_hi"],
                              n_sym_neg=sym_neg(w["name"], "K1-K0"))
    gateA = (all(wsA[w["name"]]["ci_hi"] < 0 for w in WINDOWS)
             and all(wsA[w["name"]]["n_sym_neg"] >= 10 for w in WINDOWS))
    verdict["WITHIN_EPISODE_TRANSITION_SIGNAL_SUPPORTED"] = dict(
        supported=bool(gateA), windows=wsA)

    # Markov closure: lag1
    wsL = {w["name"]: {} for w in WINDOWS}
    for w in WINDOWS:
        row = next(b for b in boots if b["window"] == w["name"]
                   and b["comparison"] == "K2-K1")
        wsL[w["name"]] = dict(ci_hi=row["ci_hi"],
                              n_sym_neg=sym_neg(w["name"], "K2-K1"))
    lag_stable = (all(wsL[w["name"]]["ci_hi"] < 0 for w in WINDOWS)
                  and all(wsL[w["name"]]["n_sym_neg"] >= 10 for w in WINDOWS))
    verdict["LAG1_RESIDUAL_DEPENDENCE_SUPPORTED"] = dict(
        supported=bool(lag_stable), windows=wsL,
        note=("Stable lag1 increment detected; do NOT interpret as latent "
              "state without checking functional form / interaction / "
              "state omission first." if lag_stable else
              "No stable lag1 increment detected at current resolution."))

    # ---------------- outputs
    print("[STAGE] writing outputs", flush=True)
    pd.DataFrame(model_metrics).to_csv(
        OUT / "dynamic_pgm1a_model_metrics.csv", index=False)
    pd.DataFrame(boots).to_csv(OUT / "dynamic_pgm1a_bootstrap.csv", index=False)
    pd.DataFrame(bysym).to_csv(OUT / "dynamic_pgm1a_by_symbol.csv", index=False)
    pd.DataFrame(target_rows).to_csv(
        OUT / "dynamic_pgm1a_target_metrics.csv", index=False)
    pd.DataFrame(opt_rows).to_csv(
        OUT / "dynamic_pgm1a_optimizer_audit.csv", index=False)
    (OUT / "dynamic_pgm1a_sample_audit.json").write_text(
        json.dumps(sample_audit, indent=2))
    (OUT / "dynamic_pgm1a_transition_invariants.json").write_text(
        json.dumps(invariants, indent=2))

    summary = dict(
        experiment="DYNAMIC-PGM-1A Within-Episode One-Step Transition Kernel",
        parent_commit=BASE_SHA,
        design=dict(
            scope="within-episode 5m state transition, terminal reset excluded",
            target="P(S_{t+1} | S_t, H_{t+1}=0)",
            predicted_innovations=dict(
                continuous=CONT_Z, discrete=DISC_Z,
                note="S_{t+1}=F(S_t,Z_{t+1}); price/location has 1 dof "
                     "(z_d_up), TV/last_return deterministic from z_d_up"),
            models=dict(
                K0_UNCONDITIONAL="no state; constant Gaussian + Laplace 4-class",
                K1_STATE="OBSERVED-STATE-v1 (35 numeric + prev_event_mask)",
                K2_STATE_LAG1="K1 + 15 lag1_* + lag1_available"),
            kernel=dict(
                continuous="Ridge mean + LedoitWolf full 9x9 covariance "
                           "(learns cross-innovation coupling)",
                discrete="4-class Logistic; Z^c perp Z^d | S_t (v1)"),
            obs_state_numeric=OBS_STATE_NUM,
            obs_state_categorical=OBS_STATE_CAT,
            lag_base=LAG_BASE,
            blocks={b: dict(cont=v["cont"], disc=v["disc"])
                    for b, v in BLOCKS.items()},
            tempo_classification="DERIVED_REPRESENTATION (not new state info)",
        ),
        windows=[dict(name=w["name"], train=w["train"], eval=w["eval"],
                      seed=w["seed"]) for w in WINDOWS],
        n_transition_rows=EXPECTED_TRANSITIONS,
        primary_metric="joint transition NLL (continuous + discrete)",
        model_metrics=model_metrics,
        bootstrap=boots,
        by_symbol_negative_counts={w["name"]: {
            "K1-K0": sym_neg(w["name"], "K1-K0"),
            "K2-K1": sym_neg(w["name"], "K2-K1")} for w in WINDOWS},
        block_contribution=block_rows,
        per_target=target_rows,
        covariance_audit=cov_audit,
        verdict=verdict,
        bootstrap_reps=BOOTSTRAP_REPS,
    )
    (OUT / "dynamic_pgm1a_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    timing["total_seconds"] = round(time.perf_counter() - t_total, 2)
    s2 = json.loads((OUT / "dynamic_pgm1a_summary.json").read_text())
    s2["timing"] = timing
    (OUT / "dynamic_pgm1a_summary.json").write_text(
        json.dumps(s2, indent=2, default=str))

    print(f"[MODEL METRICS]\n{pd.DataFrame(model_metrics).to_string(index=False)}")
    print(f"[BOOTSTRAP]\n{pd.DataFrame(boots).to_string(index=False)}")
    print(f"[BLOCK CONTRIBUTION]\n{pd.DataFrame(block_rows).to_string(index=False)}")
    print(f"[VERDICT] {verdict}")
    print(f"[COV AUDIT] {cov_audit}")
    print(f"[TIMING] {timing}")
    print(f"[DONE] -> {OUT}")


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--window-json")
    _ap.add_argument("--data")
    _ap.add_argument("--result")
    _args = _ap.parse_args()
    if _args.window_json:
        # child mode: fit a single window and emit its metrics JSON, then exit so
        # the OS reclaims all memory before the next window's subprocess starts.
        _w = json.loads(_args.window_json)
        _res = run_single_window(_w, _args.data)
        Path(_args.result).write_text(json.dumps(_res, default=str))
    else:
        main()
