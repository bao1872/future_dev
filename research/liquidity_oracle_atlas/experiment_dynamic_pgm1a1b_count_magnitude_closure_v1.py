"""DYNAMIC-PGM-1A.1b — Count Magnitude Closure Transition Kernel

===========================================================
只研究 episode 内部的 5m 状态转移，不碰 terminal reset。
===========================================================

NOTE: DYNAMIC-PGM-1A（Gaussian-count，9 维连续）已 FROZEN 于 commit
20fbd2e487dbe5cc9ce22ec1205cc7a84227b8fb，必须继续复现其既有 dynamic_pgm1a_*
结果。本 1A.1 是其独立的 support-correct 变体：count 增量改用 Hurdle-Poisson，
且正数支是为 EXACT zero-truncated Poisson（见 ZeroTruncatedPoissonRegressor 与
_ztnp_nll），取代 1A 的旧 Gaussian count。禁止把 1A.1 折回 1A。
已有：
    P(Y_{t+1} | S_t)   —— endpoint / hazard 模型（PGM-BAR-0 / MARKET-STATE-1）
这一轮研究：
    P(S_{t+1} | S_t, H_{t+1}=0)   —— 非终止时，当前 Market State 能否预测
                                    下一根之后 Market State 怎么变。

完整分解留待 1B/1C：
    P(Y_{t+1}, S_{t+1} | S_t) = P(Y_{t+1}|S_t) P(S_{t+1}|S_t, Y_{t+1})

不预测 35 个 state feature（很多是确定性关系），只预测真正的
state innovations Z_{t+1}，然后用确定公式 S_{t+1}=F(S_t, Z_{t+1})：

连续（7）：
    z_d_up           = d_U,t+1 - d_U,t            (=> r = -Δd_U)
    z_log1p_dmfe     = log1p(MFE_{t+1}-MFE_t)
    z_log1p_dmae     = log1p(MAE_{t+1}-MAE_t)
    z_delta_dcr      = DCR_{t+1}-DCR_t
    z_log1p_next_range = log1p(Range_{t+1})
    z_delta_upper_newest_resid / z_delta_lower_newest_resid   (residual update)

离散（4 类）：
    z_agezero_code = 1[upper newest age=0] + 2[lower newest age=0]

计数增量（2，离散非负整数 -> Hurdle-Poisson）：
    z_delta_upper_count = next.upper_active_identity_count_delta - cur.upper_active_identity_count_delta
    z_delta_lower_count = next.lower_active_identity_count_delta - cur.lower_active_identity_count_delta
    # 注意：stored *_active_identity_count_delta 是 episode 内【累计 activation 计数】
    # (cumulative activation counter)，NOT 当前 active 数 − 起点。故 count 节点预测单步
    # activation 增量，重建时 current_counter + increment == next_counter 必须逐行精确。

模型：
    K0 UNCONDITIONAL  不看 state，只学平均 transition 分布
    K1 STATE          OBSERVED-STATE-v1
    K2 STATE+LAG1     K1 + 上一根 dynamic state（15 个 lag1_*）

主检验：K1-K0 是否有稳定增量（state 能否预测状态演化）
次检验：K2-K1 是否还有 lag1 增量（当前 state 是否仍遗留一阶历史依赖）

连续（7）：Z^c|S_t ~ N(mu(S_t), Sigma) 多输出 Ridge + LedoitWolf 完整 7x7 协方差
离散（4 类）：Softmax(W phi(S_t)+b) 4-class Logistic
计数增量（2，非负整数）：Hurdle-Poisson —— 每侧独立 Logistic P(>0) + zero-truncated Poisson（正数子集）
第一版先 conditional independence Z^c ⊥ Z^d | S_t；count 增量与连续/离散条件独立。

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
from sklearn.metrics import log_loss, brier_score_loss, average_precision_score
from scipy.optimize import minimize, brentq
from scipy.special import gammaln

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

BASE_SHA = "3fcac2ee0875e90256e6c6b3def4c6b72b768399"
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

# ===========================================================================
# Support-correct transition node specification
# Each Z_{t+1}^{(k)} is modeled with a distribution whose support matches the
# physical support of that state increment / next-state component. Reconstruction
# F(S_t, Z_{t+1}) = S_{t+1} must hold exactly (<=1e-8) using the realized Z.
#   kind:
#     gaussian_delta      : Z = S_{t+1}-S_t (any real)            -> Gaussian
#     hurdle_ln_delta     : delta>=0; P(=0)+log(delta) on +      -> Hurdle-LogNormal
#     ln_value           : S_{t+1}>0; log(S_{t+1})               -> LogNormal
#     hurdle_ln_value     : S_{t+1}>=0; P(=0)+log(S_{t+1}) on +  -> Hurdle-LogNormal
#     hurdle_ln_neg       : delta<=0; model q=-delta>=0 via P(=0)+log(q) on +
#                           -> Hurdle-LogNormal (sign flips recon to -exp(log))
#     signed_hurdle       : delta in R; sign{0,-,+} multinomial + log|delta| Gaussian
#     zero_interior_one   : S_{t+1} in {0}U(0,1)U{1}            -> 3-class + logit-Normal
#
# Residual / range support is DATA-DRIVEN: a real support audit (audit_support_*)
# selects the kind from the realized sign distribution (pre-registered rules). The
# baked NODE_SPECS below encode the audit outcome for the current 321,727-transition
# dataset (residuals are all <= 0 -> hurdle_ln_neg; range has zeros -> hurdle_ln_value).
# main() re-runs the audit on real data and RAISES STOP if the selection disagrees
# with the baked kind (i.e. the data support moved and the transform must be
# re-decided). This prevents "force positive because the code is convenient".
# ===========================================================================
NODE_SPECS = [
    ("z_d_up",   "gaussian_delta",    "cur_up_distance_R",            "cur_up_distance_R",            "Location"),
    ("z_dmfe",   "hurdle_ln_delta",   "path_max_up_excursion_R",      "path_max_up_excursion_R",      "Path"),
    ("z_dmae",   "hurdle_ln_delta",   "path_max_down_excursion_R",    "path_max_down_excursion_R",    "Path"),
    ("z_dcr",    "zero_interior_one", "path_direction_change_rate",   "path_direction_change_rate",   "Path"),
    ("z_range",  "hurdle_ln_value",   "path_current_bar_range_R",     "path_current_bar_range_R",     "Path"),
    ("z_uresid", "hurdle_ln_neg",     "upper_newest_log_age_residual", "upper_newest_log_age_residual", "LiquidityComposition"),
    ("z_lresid", "hurdle_ln_neg",     "lower_newest_log_age_residual", "lower_newest_log_age_residual", "LiquidityComposition"),
]
# count 节点：episode 内累计 activation 计数的单步增量（非负整数 -> Hurdle-Poisson）。
# stored *_active_identity_count_delta 是 cumulative activation counter，不是当前 active 数差。
COUNT_Z = ["z_delta_upper_count", "z_delta_lower_count"]
DISC_Z = "z_agezero_code"

# Set True by main() after the real age-zero reconstruction audit proves the
# 4-class age-zero node is a deterministic derived state (mismatch == 0). When
# True the stochastic 4-class node is dropped from the model (BLOCKS disc emptied).
AGEZERO_DETERMINISTIC = False


def _node_zcols(name, kind):
    if kind == "gaussian_delta":
        return [name]
    if kind in ("hurdle_ln_delta", "hurdle_ln_value", "hurdle_ln_neg"):
        return [f"{name}_ispos", f"{name}_log"]
    if kind == "ln_value":
        return [f"{name}_log"]
    if kind == "signed_hurdle":
        return [f"{name}_is0", f"{name}_isneg", f"{name}_ispos", f"{name}_logabs"]
    if kind == "zero_interior_one":
        return [f"{name}_is0", f"{name}_is1", f"{name}_logit"]
    raise ValueError(f"unknown kind {kind}")


# ---- support audit (data-driven kind selection; pre-registered rules) ----
def audit_support_residual(res, name="residual"):
    """residual sign distribution -> recommended model kind + stats.

    Pre-registered rule:
      only (<=0):  negatives + zeros  -> hurdle_ln_neg  (model q = -res >= 0)
      only (>=0):  positives + zeros  -> hurdle_ln_value(model q = res  >= 0)
      both signs:                      -> signed_hurdle (sign{0,-,+} + log|res|)
    """
    res = np.asarray(res, dtype=np.float64)
    n_neg = int((res < -1e-12).sum())
    n_zero = int((np.abs(res) <= 1e-12).sum())
    n_pos = int((res > 1e-12).sum())
    if n_neg > 0 and n_pos == 0:
        kind = "hurdle_ln_neg"
    elif n_pos > 0 and n_neg == 0:
        kind = "hurdle_ln_value"
    else:
        kind = "signed_hurdle"
    return dict(name=name, min=float(res.min()), max=float(res.max()),
                n_neg=n_neg, n_zero=n_zero, n_pos=n_pos,
                recommended_kind=kind)


def audit_support_range(rng, name="range"):
    """range support -> recommended model kind + stats.

    Pre-registered rule: zero_count == 0 -> LogNormal; zero_count > 0 -> Hurdle-LogNormal.
    """
    rng = np.asarray(rng, dtype=np.float64)
    n_zero = int((rng <= 0).sum())
    n_pos = int((rng > 0).sum())
    kind = "hurdle_ln_value" if n_zero > 0 else "ln_value"
    return dict(name=name, min=float(rng.min()), max=float(rng.max()),
                n_zero=n_zero, n_pos=n_pos, recommended_kind=kind)


def _kind_of(name):
    for n, k, _, _, _ in NODE_SPECS:
        if n == name:
            return k
    raise KeyError(name)


NODE_ZCOLS = {name: _node_zcols(name, kind) for name, kind, _, _, _ in NODE_SPECS}
NODE_KIND = {name: kind for name, kind, _, _, _ in NODE_SPECS}

# Zc column layout = concatenation of NODE_ZCOLS in NODE_SPECS order.
# Z_LAYOUT stores (node_name, integer offsets into the concatenated Z matrix) for
# slicing numpy arrays; ALL_Z_COLS is the flat list of actual z *column names* used
# to select from the pandas frame.
Z_LAYOUT = []
_OFF = 0
for _n, _c in NODE_ZCOLS.items():
    Z_LAYOUT.append((_n, list(range(_OFF, _OFF + len(_c)))))
    _OFF += len(_c)
ALL_Z_COLS = [c for _cols in NODE_ZCOLS.values() for c in _cols]


def build_z_columns(node, kind, raw):
    """Transform a raw next-value (or delta) array into its Z columns.

    Returns dict {z_col_name: (n,) array} ready to assign onto the transition frame.
    """
    raw = np.asarray(raw, dtype=np.float64)
    cols = _node_zcols(node, kind)
    if kind == "gaussian_delta":
        return {cols[0]: raw}
    if kind in ("hurdle_ln_value", "hurdle_ln_neg"):
        q = raw if kind == "hurdle_ln_value" else -raw
        ispos = (q > 0).astype(float)
        logv = np.where(ispos > 0.5, np.log(np.maximum(q, 1e-12)), 0.0)
        return {cols[0]: ispos, cols[1]: logv}
    if kind == "ln_value":
        return {cols[0]: np.log(np.maximum(raw, 1e-12))}
    if kind == "signed_hurdle":
        is0 = (np.abs(raw) <= 1e-12).astype(float)
        isneg = (raw < -1e-12).astype(float)
        ispos = (raw > 1e-12).astype(float)
        logabs = np.where(is0 < 0.5, np.log(np.maximum(np.abs(raw), 1e-12)), 0.0)
        return {cols[0]: is0, cols[1]: isneg, cols[2]: ispos, cols[3]: logabs}
    raise ValueError(f"build_z_columns unhandled kind {kind}")


def reconstruct_value(kind, z):
    """Reconstruct the original-space value from a node's Z columns.

    `z` is a list/array of the node's Z columns in _node_zcols order.
    """
    z = [np.asarray(c, dtype=np.float64) for c in z]
    if kind == "gaussian_delta":
        return z[0]
    if kind in ("hurdle_ln_value", "hurdle_ln_delta"):
        ispos, logv = z[0], z[1]
        return np.where(ispos > 0.5, np.exp(logv), 0.0)
    if kind == "hurdle_ln_neg":
        ispos, logv = z[0], z[1]
        return np.where(ispos > 0.5, -np.exp(logv), 0.0)
    if kind == "ln_value":
        return np.exp(z[0])
    if kind == "zero_interior_one":
        is0, is1, logit = z[0], z[1], z[2]
        return np.where(is0 > 0.5, 0.0,
                       np.where(is1 > 0.5, 1.0, 1.0 / (1.0 + np.exp(-logit))))
    if kind == "signed_hurdle":
        is0, isneg, ispos, logabs = z
        sign = np.where(isneg > 0.5, -1.0,
                       np.where(ispos > 0.5, 1.0, 0.0))
        mag = np.where(is0 < 0.5, np.exp(logabs), 0.0)
        return sign * mag
    raise ValueError(f"reconstruct_value unhandled kind {kind}")


# Source columns consumed from the enriched frame by build_transition_sample
# (both `cur` and the shifted `nxt`). Kept at float64 so the exact deterministic
# invariants (TV update / last-return / monotonic excursion / age-zero recon) retain
# precision. upper/lower *_newest_log_age are needed to deterministically reconstruct
# the next age-zero from the residual + current age (see reconstruct_agezero).
BUILD_SRC_COLS = [
    "cur_width_R", "cur_up_distance_R", "cur_down_distance_R", "cur_log_ratio",
    "path_max_up_excursion_R", "path_max_down_excursion_R",
    "path_direction_change_rate", "path_current_bar_range_R",
    "upper_newest_log_age_residual", "lower_newest_log_age_residual",
    "upper_newest_log_age", "lower_newest_log_age",
    "upper_active_identity_count_delta", "lower_active_identity_count_delta",
    "upper_current_newest_age_zero", "lower_current_newest_age_zero",
    "path_total_variation_R", "path_last_return_R",
    "start_bar", "bar_t",
]

BLOCKS = {
    "Location": dict(nodes=["z_d_up"], disc=[], count=[]),
    "Path": dict(nodes=["z_dmfe", "z_dmae", "z_dcr", "z_range"], disc=[], count=[]),
    "LiquidityComposition": dict(nodes=["z_uresid", "z_lresid"],
                                disc=["z_agezero_code"], count=COUNT_Z),
}


def disc_spec():
    """Current set of stochastic discrete (4-class) nodes from BLOCKS.

    Empty once the age-zero node is proven deterministic and removed (see main()).
    """
    return [d for b in BLOCKS.values() for d in b["disc"]]


def configure_child_semantics(agezero_deterministic: bool = False):
    """Configure model semantics for parent or child window processes.

    When agezero_deterministic is True:
      - Marks AGEZERO_DETERMINISTIC = True
      - Empties BLOCKS["LiquidityComposition"]["disc"]
    When False:
      - Marks AGEZERO_DETERMINISTIC = False
      - Restores BLOCKS["LiquidityComposition"]["disc"] = [DISC_Z]
    """
    global AGEZERO_DETERMINISTIC
    if agezero_deterministic:
        AGEZERO_DETERMINISTIC = True
        BLOCKS["LiquidityComposition"]["disc"] = []
    else:
        AGEZERO_DETERMINISTIC = False
        BLOCKS["LiquidityComposition"]["disc"] = [DISC_Z]


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
    # Hurdle-LogNormal delta:  P(Δ=0)  +  log(Δ) on positives  (Δ>=0 exact)
    cur["z_dmfe_ispos"] = (dmfe > 0).astype(float)
    cur["z_dmfe_log"] = np.where(dmfe > 0, np.log(np.maximum(dmfe, 1e-12)), 0.0)
    cur["z_dmae_ispos"] = (dmae > 0).astype(float)
    cur["z_dmae_log"] = np.where(dmae > 0, np.log(np.maximum(dmae, 1e-12)), 0.0)

    # DCR next value: 0 / interior / 1.  interior modeled via logit-Normal.
    dcr = nxt["path_direction_change_rate"].to_numpy(float)
    if dcr.min() < -1e-9 or dcr.max() > 1 + 1e-9:
        raise SystemExit("STOP_DYNAMIC_PGM1A_DCR_OUT_OF_RANGE")
    cur["z_dcr_is0"] = (dcr <= 1e-9).astype(float)
    cur["z_dcr_is1"] = (dcr >= 1 - 1e-9).astype(float)
    interior = (dcr > 1e-9) & (dcr < 1 - 1e-9)
    cur["z_dcr_logit"] = np.where(interior, np.log(dcr / (1.0 - dcr)), 0.0)

    # Range next value: LogNormal if no zeros, else Hurdle-LogNormal. The support is
    # selected by the real-data audit (audit_support_range); the baked NODE_KIND is
    # "hurdle_ln_value" because the data contains zeros. Do NOT force Range > 0.
    rng = nxt["path_current_bar_range_R"].to_numpy(float)
    _zc_range = build_z_columns("z_range", NODE_KIND["z_range"], rng)
    for _c, _v in _zc_range.items():
        cur[_c] = _v

    for side in ("upper", "lower"):
        res = nxt[f"{side}_newest_log_age_residual"].to_numpy(float)
        # Residual support is all <= 0 in the data -> hurdle_ln_neg (q = -res >= 0).
        # Kind is data-selected by the real audit (audit_support_residual); the baked
        # NODE_KIND must match (else STOP_DYNAMIC_PGM1A_SUPPORT_SELECTION_MISMATCH in
        # main). Do NOT force residual >= 0; a negative residual is the dominant
        # provenance signal (new liquidity activation => newest age < expected => r<0).
        node = "z_uresid" if side == "upper" else "z_lresid"
        _zc = build_z_columns(node, NODE_KIND[node], res)
        for _c, _v in _zc.items():
            cur[_c] = _v
        cur[f"z_delta_{side}_count"] = (
            nxt[f"{side}_active_identity_count_delta"].to_numpy(float)
            - cur[f"{side}_active_identity_count_delta"].to_numpy(float))

    # ---- count-increment hard guards (cumulative activation counter)
    # stored *_active_identity_count_delta 是 episode 内累计 activation 计数；
    # 单步增量必须非负整数，且 cur + inc == next 精确（否则累计解释有误）。
    for side in ("upper", "lower"):
        inc = cur[f"z_delta_{side}_count"].to_numpy(float)
        nxtc = nxt[f"{side}_active_identity_count_delta"].to_numpy(float)
        curc = cur[f"{side}_active_identity_count_delta"].to_numpy(float)
        if not np.all(np.isfinite(inc)):
            raise SystemExit("STOP_DYNAMIC_PGM1A_ACTIVATION_INCREMENT_NAN")
        if not np.all(np.equal(np.mod(inc, 1), 0)):
            raise SystemExit("STOP_DYNAMIC_PGM1A_ACTIVATION_INCREMENT_NONINTEGER")
        if inc.min() < 0:
            raise SystemExit("STOP_DYNAMIC_PGM1A_ACTIVATION_COUNTER_NONMONOTONE")
        if not np.allclose(curc + inc, nxtc, atol=1e-9):
            raise SystemExit("STOP_DYNAMIC_PGM1A_ACTIVATION_RECON_FAIL")

    # 4-class age-zero code from the current-newest-age-zero columns, which the
    # provenance residual + current state deterministically reconstruct.
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

    # Map episode start upper/lower prices to compute exact atr0 = (U - D) / cur_width_R
    if "symbol" in cur.columns and "start_bar" in cur.columns and "cur_width_R" in cur.columns:
        ep0_path = CACHE / "episode0_episodes.parquet"
        ep3_path = CACHE / "episode_repl0_through_tb3.parquet"
        if ep0_path.exists() and ep3_path.exists():
            ep0 = pd.read_parquet(ep0_path, columns=["symbol", "start_bar", "start_upper_price", "start_lower_price"])
            ep3 = pd.read_parquet(ep3_path, columns=["symbol", "start_bar", "start_upper_price", "start_lower_price"])
            ep_all = pd.concat([ep0, ep3]).drop_duplicates(subset=["symbol", "start_bar"])
            ep_all["span_price"] = ep_all["start_upper_price"].astype(float) - ep_all["start_lower_price"].astype(float)
            ep_map = dict(zip(zip(ep_all["symbol"], ep_all["start_bar"].astype(int)), ep_all["span_price"]))
            keys = list(zip(cur["symbol"], cur["start_bar"].astype(int)))
            spans = np.array([ep_map.get(k, 1.0) for k in keys], dtype=float)
            cur["atr0"] = spans / cur["cur_width_R"].to_numpy(float)
            nxt["atr0"] = cur["atr0"].to_numpy(float)
        else:
            cur["atr0"] = 1.0
            nxt["atr0"] = 1.0
    else:
        cur["atr0"] = 1.0
        nxt["atr0"] = 1.0

    # slim the shifted `nxt` to only the source columns actually consumed by the
    # invariants / z_* computations -> avoids holding the full (86-col) shift
    # copy in memory through the rest of the pipeline.
    nxt_cols = [c for c in BUILD_SRC_COLS if c in nxt.columns]
    if "atr0" in nxt.columns and "atr0" not in nxt_cols:
        nxt_cols.append("atr0")
    nxt = nxt[nxt_cols].copy()
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
def _check_logistic_convergence(clf, name="logistic"):
    if hasattr(clf, "n_iter_"):
        max_iter = int(np.max(clf.n_iter_))
        if max_iter >= 3000:
            raise SystemExit(
                f"STOP_DYNAMIC_PGM1A1_LOGISTIC_MAXITER: {name} reached max_iter={max_iter}")


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
        self.n_params = p * q + q
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


class HurdleLogNormalHead:
    """Hurdle-LogNormal node: P(Δ=0) via Logistic + log(Δ) ~ Gaussian on Δ>0.

    Consumes z columns [ispos, log]; ispos in {0,1}, log = log(|Δ|) for Δ!=0 else 0.
    `sign` (+1 / -1) selects the original-space reconstruction:
        sign=+1  ->  Δ = 0 if ispos==0 else +exp(log)   (Δ >= 0)
        sign=-1  ->  Δ = 0 if ispos==0 else -exp(log)   (Δ <= 0, used for residual)
    The Jacobian of y = sign*exp(z) is |dy/dz| = exp(z) = |y|, so the original-space
    NLL gains +log|z| on positive rows: NLL_orig = NLL_log + log|z|.
    """

    def __init__(self, alpha=1.0, sign=1.0):
        self.alpha = alpha
        self.sign = float(sign)

    def fit(self, X, Z):
        Z = np.asarray(Z, dtype=np.float64)
        ispos = Z[:, 0]
        logv = Z[:, 1]
        pos = ispos > 0.5
        self.logit = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                        max_iter=3000).fit(X, ispos.astype(int))
        _check_logistic_convergence(self.logit, "hurdle_ln_logit")
        self.g = GaussianTransitionHead(alpha=self.alpha).fit(
            X[pos], logv[pos].reshape(-1, 1))
        self.n_params = (X.shape[1] + 1) + (X.shape[1] * 1 + 1)
        return self

    def nll_per_row(self, X, Z):
        Z = np.asarray(Z, dtype=np.float64)
        ispos = Z[:, 0]
        logv = Z[:, 1]
        pos = ispos > 0.5
        p0 = np.clip(self.logit.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        nll_ispos = np.where(pos, -np.log(p0), -np.log1p(-p0))
        nll_log = self.g.nll_per_row(X, logv.reshape(-1, 1))
        nll_log = np.where(pos, nll_log, 0.0)
        # change-of-variable: y = sign*exp(z) -> |J| = exp(z) -> log|J| = z = logv
        jac = np.where(pos, logv, 0.0)
        return nll_ispos + nll_log + jac


class LogNormalHead:
    """LogNormal node: log(S_{t+1}) ~ Gaussian. Consumes [log] z column.

    Original-space NLL gains +log(S) = +log(z_column) (since S = exp(z)).
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, Z):
        Z = np.asarray(Z, dtype=np.float64)
        self.g = GaussianTransitionHead(alpha=self.alpha).fit(
            X, Z.reshape(-1, 1))
        self.n_params = X.shape[1] * 1 + 1
        return self

    def nll_per_row(self, X, Z):
        Z = np.asarray(Z, dtype=np.float64)
        # Jacobian: S = exp(z) -> log|J| = z
        return self.g.nll_per_row(X, Z.reshape(-1, 1)) + Z.reshape(-1)


class ZeroInteriorOneHead:
    """DCR node: S_{t+1} in {0} U (0,1) U {1}.

    3-class Logistic for {0, interior, 1} + logit(S_{t+1}) ~ Gaussian on interior.
    Consumes z columns [is0, is1, logit]; interior = not is0 and not is1.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, Z):
        Z = np.asarray(Z, dtype=np.float64)
        is0 = Z[:, 0]
        is1 = Z[:, 1]
        logit = Z[:, 2]
        interior = (is0 < 0.5) & (is1 < 0.5)
        y3 = np.where(is0 > 0.5, 0, np.where(is1 > 0.5, 2, 1)).astype(int)
        self.cat = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                     max_iter=3000).fit(X, y3)
        _check_logistic_convergence(self.cat, "dcr_cat")
        self.g = GaussianTransitionHead(alpha=self.alpha).fit(
            X[interior], logit[interior].reshape(-1, 1))
        self.n_params = (X.shape[1] * 3 + 3) + (X.shape[1] * 1 + 1)
        return self

    def nll_per_row(self, X, Z):
        Z = np.asarray(Z, dtype=np.float64)
        is0 = Z[:, 0]
        is1 = Z[:, 1]
        logit = Z[:, 2]
        interior = (is0 < 0.5) & (is1 < 0.5)
        y3 = np.where(is0 > 0.5, 0, np.where(is1 > 0.5, 2, 1)).astype(int)
        p = np.clip(self.cat.predict_proba(X), 1e-12, 1.0)
        nll_cat = -np.log(p[np.arange(len(y3)), y3])
        nll_logit = self.g.nll_per_row(X, logit.reshape(-1, 1))
        nll_logit = np.where(interior, nll_logit, 0.0)
        # change-of-variable for y = sigmoid(z): |J| = y(1-y); log|J| = z - 2*log1p(e^z)
        # (only on interior rows; boundaries are deterministic point masses).
        jac = np.where(interior, _logit_jacobian(logit), 0.0)
        return nll_cat + nll_logit + jac


class SignedHurdleHead:
    """Signed-Hurdle node: sign in {0,-,+} multinomial Logistic + log|Δ| ~ Gaussian on Δ!=0.

    Consumes z columns [is0, isneg, ispos, logabs]; logabs = log(|Δ|) for Δ!=0 else 0.
    Used when a residual/value has BOTH signs in the real data (pre-registered fallback).
    Recon: Δ = 0 if is0==1 else sign*exp(logabs). Jacobian: |Δ| = exp(logabs) -> +logabs.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, Z):
        Z = np.asarray(Z, dtype=np.float64)
        is0 = Z[:, 0]
        isneg = Z[:, 1]
        ispos = Z[:, 2]
        logabs = Z[:, 3]
        nonzero = (is0 < 0.5)
        y3 = np.where(isneg > 0.5, 0, np.where(ispos > 0.5, 2, 1)).astype(int)
        self.cat = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                     max_iter=3000).fit(X, y3)
        _check_logistic_convergence(self.cat, "signed_hurdle_cat")
        self.g = GaussianTransitionHead(alpha=self.alpha).fit(
            X[nonzero], logabs[nonzero].reshape(-1, 1))
        self.n_params = (X.shape[1] * 3 + 3) + (X.shape[1] * 1 + 1)
        return self

    def nll_per_row(self, X, Z):
        Z = np.asarray(Z, dtype=np.float64)
        is0 = Z[:, 0]
        isneg = Z[:, 1]
        ispos = Z[:, 2]
        logabs = Z[:, 3]
        nonzero = (is0 < 0.5)
        y3 = np.where(isneg > 0.5, 0, np.where(ispos > 0.5, 2, 1)).astype(int)
        p = np.clip(self.cat.predict_proba(X), 1e-12, 1.0)
        nll_cat = -np.log(p[np.arange(len(y3)), y3])
        nll_log = self.g.nll_per_row(X, logabs.reshape(-1, 1))
        nll_log = np.where(nonzero, nll_log, 0.0)
        jac = np.where(nonzero, logabs, 0.0)
        return nll_cat + nll_log + jac


def _logit_jacobian(z):
    """log|J| for y = sigmoid(z): log[y(1-y)] = z - 2*log1p(exp(z)), numerically stable."""
    z = np.asarray(z, dtype=np.float64)
    # log1p(exp(z)) stable: for z>=0 -> z + log1p(exp(-z)); for z<0 -> log1p(exp(z))
    l1e = np.where(z >= 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))
    return z - 2.0 * l1e


def _head_for(kind):
    if kind == "gaussian_delta":
        return GaussianTransitionHead(alpha=1.0)
    if kind == "hurdle_ln_delta":
        return HurdleLogNormalHead(alpha=1.0, sign=1.0)
    if kind == "hurdle_ln_value":
        return HurdleLogNormalHead(alpha=1.0, sign=1.0)
    if kind == "hurdle_ln_neg":
        return HurdleLogNormalHead(alpha=1.0, sign=-1.0)
    if kind == "ln_value":
        return LogNormalHead(alpha=1.0)
    if kind == "signed_hurdle":
        return SignedHurdleHead(alpha=1.0)
    if kind == "zero_interior_one":
        return ZeroInteriorOneHead(alpha=1.0)
    raise ValueError(f"unknown kind {kind}")


def make_transformer(num_cols, cat_cols):
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler())])
    cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])


def _fit_nodes(Xtr, Xev, Zc_tr, Zc_ev):
    """Fit one support-correct head per node; return per-node NLL dicts + n_params."""
    nodes = {}
    n_params = 0
    kind_of = dict((n, k) for n, k, _, _, _ in NODE_SPECS)
    for name, cols in Z_LAYOUT:
        head = _head_for(kind_of[name])
        head.fit(Xtr, Zc_tr[:, cols])
        nodes[name] = dict(
            tr=head.nll_per_row(Xtr, Zc_tr[:, cols]),
            ev=head.nll_per_row(Xev, Zc_ev[:, cols]),
            head=head)
        n_params += head.n_params
    return nodes, n_params


def _fit_disc(Xtr, Xev, yd_tr, yd_ev):
    if len(np.unique(yd_tr)) != 4:
        raise SystemExit("STOP_DYNAMIC_PGM1A_DISCRETE_CLASS_MISSING")
    disc = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                             max_iter=3000)
    disc.fit(Xtr, yd_tr)
    _check_logistic_convergence(disc, "disc_4class")
    p_tr = disc.predict_proba(Xtr)[np.arange(len(yd_tr)), yd_tr]
    p_ev = disc.predict_proba(Xev)[np.arange(len(yd_ev)), yd_ev]
    disc_tr = -np.log(np.clip(p_tr, 1e-12, 1.0))
    disc_ev = -np.log(np.clip(p_ev, 1e-12, 1.0))
    return disc, disc_tr, disc_ev, Xtr.shape[1] * 4 + 4


def fit_state_heads(Xtr, Xev, Zc_tr, Zc_ev, yd_tr, yd_ev):
    """K1/K2: per-node support-correct heads + 4-class Logistic for age-zero.

    If the 4-class age-zero node has been removed (disc_spec() empty), disc is
    skipped and the discrete NLL terms are zero (so joint NLL == continuous NLL).
    """
    Zc_tr = np.asarray(Zc_tr, dtype=np.float64)
    Zc_ev = np.asarray(Zc_ev, dtype=np.float64)
    nodes, np_cont = _fit_nodes(Xtr, Xev, Zc_tr, Zc_ev)
    if disc_spec():
        disc, disc_tr, disc_ev, np_disc = _fit_disc(Xtr, Xev, yd_tr, yd_ev)
    else:
        n_tr, n_ev = len(Xtr), len(Xev)
        disc, disc_tr, disc_ev, np_disc = None, np.zeros(n_tr), np.zeros(n_ev), 0
    return dict(nodes=nodes, disc=disc, disc_tr=disc_tr, disc_ev=disc_ev,
                n_params=np_cont + np_disc)


def fit_constant_heads(Zc_tr, Zc_ev, yd_tr, yd_ev):
    """K0: per-node marginal heads (X = constant) + marginal 4-class.

    Skips the discrete node when disc_spec() is empty (see fit_state_heads).
    """
    Zc_tr = np.asarray(Zc_tr, dtype=np.float64)
    Zc_ev = np.asarray(Zc_ev, dtype=np.float64)
    Xtr = np.ones((len(Zc_tr), 1))
    Xev = np.ones((len(Zc_ev), 1))
    nodes, np_cont = _fit_nodes(Xtr, Xev, Zc_tr, Zc_ev)
    if disc_spec():
        counts = np.bincount(yd_tr, minlength=4).astype(float)
        prob = (counts + 0.5) / (len(yd_tr) + 2.0)
        disc_tr = -np.log(prob[yd_tr])
        disc_ev = -np.log(prob[yd_ev])
        np_disc = 4
    else:
        disc_tr = np.zeros(len(Xtr))
        disc_ev = np.zeros(len(Xev))
        np_disc = 0
    return dict(nodes=nodes, disc_tr=disc_tr, disc_ev=disc_ev,
                n_params=np_cont + np_disc)


# ===========================================================================
# count head: per-side Hurdle-Poisson for the activation increment
# ===========================================================================
def _log_factorial(y):
    y = np.asarray(y, dtype=np.float64)
    return gammaln(y + 1.0)


def _ztnp_nll(y, rate):
    """Exact zero-truncated Poisson NLL for y >= 1.

    P(Y=y | Y>0, lam) = e^{-lam} lam^y / y!  /  (1 - e^{-lam})
    => NLL = lam - y*log(lam) + log(y!) + log(1 - e^{-lam})

    Numerically stable: log(1 - e^{-lam}) = log(-expm1(-lam)).
    """
    rate = np.maximum(np.asarray(rate, dtype=np.float64), 1e-12)
    log_trunc = np.log(-np.expm1(-rate))
    return rate - y * np.log(rate) + _log_factorial(y) + log_trunc


def _hurdle_nll(y, p0, rate):
    """Per-row Hurdle-Poisson NLL.

    P(A|S) = P(A>0|S) * P(A|A>0,S),  A>=0 integer.
    y    : (n,) int increment (>=0)
    p0   : (n,) P(increment > 0)   (Logistic)
    rate : (n,) zero-truncated Poisson rate (positive), only used where y>0
    """
    y = np.asarray(y, dtype=np.float64)
    p0 = np.asarray(p0, dtype=np.float64)
    rate = np.maximum(np.asarray(rate, dtype=np.float64), 1e-6)
    pos = y > 0
    nll_pos = -np.log(np.clip(p0, 1e-12, 1.0)) + _ztnp_nll(y, rate)
    nll_zero = -np.log1p(-p0)
    return np.where(pos, nll_pos, nll_zero)


def _fit_constant_ztp_rate(yt):
    """K0/K1 constant ZTP rate: solve E[Y|Y>0] = lam / (1 - e^{-lam}) = mean(Y>0) via Brent's method.

    E[Y|Y>0] = lam / (1 - e^{-lam}); for mean near 1.0, implied lam -> 0.
    """
    pos = yt[yt > 0]
    if len(pos) == 0:
        return 1e-8
    m = float(np.mean(pos))
    if (not np.isfinite(m)) or m <= 1.0 + 1e-12:
        return 1e-8
    f = lambda lam: lam / (-np.expm1(-lam)) - m
    try:
        return float(brentq(f, 1e-8, 100.0))
    except (ValueError, RuntimeError):
        return 1e-8


def fit_constant_count_head(Ytr, Yev):
    """K0 count head: marginal Hurdle-Poisson per column (constant occurrence + constant ZTP)."""
    Ytr = np.asarray(Ytr, dtype=np.int64)
    Yev = np.asarray(Yev, dtype=np.int64)
    nll_tr = np.zeros((len(Ytr), Ytr.shape[1]))
    nll_ev = np.zeros((len(Yev), Yev.shape[1]))
    p0_ev_all = np.zeros((len(Yev), Ytr.shape[1]))
    rate_ev_all = np.zeros((len(Yev), Ytr.shape[1]))
    rates = []
    for j in range(Ytr.shape[1]):
        yt, ye = Ytr[:, j], Yev[:, j]
        p0 = float(np.mean(yt > 0))
        rate = _fit_constant_ztp_rate(yt)
        rates.append(rate)
        p0_ev_all[:, j] = p0
        rate_ev_all[:, j] = rate
        nll_tr[:, j] = _hurdle_nll(yt, p0, rate)
        nll_ev[:, j] = _hurdle_nll(ye, p0, rate)
    return dict(nll_tr=nll_tr, nll_ev=nll_ev,
                p0_ev=p0_ev_all, rate_ev=rate_ev_all,
                constant_rates=rates,
                n_params=int(2 * Ytr.shape[1]))


def fit_state_count_head(Xtr, Ytr, Xev, Yev, constant_rates):
    """K1 count head under Count Magnitude Closure:
    Occurrence: Logistic regression P(A > 0 | X).
    Magnitude: state-independent exact ZTP using train-fitted constant rates (shared with K0).
    """
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Xev = np.asarray(Xev, dtype=np.float64)
    Ytr = np.asarray(Ytr, dtype=np.int64)
    Yev = np.asarray(Yev, dtype=np.int64)
    n_tr, n_ev = len(Ytr), len(Yev)
    nll_tr = np.zeros((n_tr, Ytr.shape[1]))
    nll_ev = np.zeros((n_ev, Yev.shape[1]))
    p0_ev_all = np.zeros((n_ev, Ytr.shape[1]))
    rate_ev_all = np.zeros((n_ev, Ytr.shape[1]))

    for j in range(Ytr.shape[1]):
        yt, ye = Ytr[:, j], Yev[:, j]
        zt = (yt > 0).astype(np.int64)
        if len(np.unique(zt)) < 2:
            p_const = 1.0 - 1e-6 if zt[0] == 1 else 1e-6
            p0_tr = np.full(n_tr, p_const)
            p0_ev = np.full(n_ev, p_const)
        else:
            logit = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                       max_iter=3000).fit(Xtr, zt)
            _check_logistic_convergence(logit, f"count_{COUNT_Z[j]}_occurrence")
            p0_tr = np.clip(logit.predict_proba(Xtr)[:, 1], 1e-6, 1.0 - 1e-6)
            p0_ev = np.clip(logit.predict_proba(Xev)[:, 1], 1e-6, 1.0 - 1e-6)
        p0_ev_all[:, j] = p0_ev

        rate = constant_rates[j]
        rate_ev_all[:, j] = rate

        nll_tr[:, j] = _hurdle_nll(yt, p0_tr, rate)
        nll_ev[:, j] = _hurdle_nll(ye, p0_ev, rate)

    return dict(
        nll_tr=nll_tr, nll_ev=nll_ev,
        p0_ev=p0_ev_all, rate_ev=rate_ev_all,
        constant_rates=constant_rates,
        n_params=int((Xtr.shape[1] + 1 + 1) * Ytr.shape[1]),
    )


def compute_count_closure_diagnostics(w_name, Ytr, Yev, k0_count, k1_count):
    """Compute detailed count magnitude closure diagnostics for upper and lower count increments."""
    diag_rows = []
    Ytr = np.asarray(Ytr, dtype=np.int64)
    Yev = np.asarray(Yev, dtype=np.int64)
    for j, c in enumerate(COUNT_Z):
        yt, ye = Ytr[:, j], Yev[:, j]
        pos_tr = yt > 0
        pos_ev = ye > 0
        n_pos_tr = int(pos_tr.sum())
        n_pos_ev = int(pos_ev.sum())
        train_p_occ = float(np.mean(pos_tr))
        eval_p_occ = float(np.mean(pos_ev))
        lam = float(k0_count["constant_rates"][j])
        denom = -np.expm1(-lam)
        expected_mag = float(lam / denom) if denom > 0 else 1.0

        actual_tr_pos_mean = float(np.mean(yt[pos_tr])) if n_pos_tr > 0 else float("nan")
        actual_ev_pos_mean = float(np.mean(ye[pos_ev])) if n_pos_ev > 0 else float("nan")

        pred_p_y1 = float(lam * np.exp(-lam) / denom) if denom > 0 else 1.0
        pred_p_y2 = float(0.5 * (lam ** 2) * np.exp(-lam) / denom) if denom > 0 else 0.0
        pred_p_y_ge3 = float(max(0.0, 1.0 - pred_p_y1 - pred_p_y2))

        actual_tr_p_y1 = float(np.mean(yt[pos_tr] == 1)) if n_pos_tr > 0 else float("nan")
        actual_tr_p_y2 = float(np.mean(yt[pos_tr] == 2)) if n_pos_tr > 0 else float("nan")
        actual_tr_p_y_ge3 = float(np.mean(yt[pos_tr] >= 3)) if n_pos_tr > 0 else float("nan")

        actual_ev_p_y1 = float(np.mean(ye[pos_ev] == 1)) if n_pos_ev > 0 else float("nan")
        actual_ev_p_y2 = float(np.mean(ye[pos_ev] == 2)) if n_pos_ev > 0 else float("nan")
        actual_ev_p_y_ge3 = float(np.mean(ye[pos_ev] >= 3)) if n_pos_ev > 0 else float("nan")

        ze = pos_ev.astype(int)
        p0_ev_k1 = k1_count["p0_ev"][:, j]
        p0_ev_k0 = k0_count["p0_ev"][:, j]
        k1_ll = float(log_loss(ze, p0_ev_k1, labels=[0, 1]))
        k1_brier = float(brier_score_loss(ze, p0_ev_k1))
        k1_prauc = float(average_precision_score(ze, p0_ev_k1)) if len(np.unique(ze)) > 1 else float("nan")
        k0_ll = float(log_loss(ze, p0_ev_k0, labels=[0, 1]))
        k0_brier = float(brier_score_loss(ze, p0_ev_k0))

        k0_mean_nll = float(np.mean(k0_count["nll_ev"][:, j]))
        k1_mean_nll = float(np.mean(k1_count["nll_ev"][:, j]))

        nll_occ_k0 = np.where(pos_ev, -np.log(np.clip(p0_ev_k0, 1e-12, 1.0)), -np.log1p(-p0_ev_k0))
        nll_occ_k1 = np.where(pos_ev, -np.log(np.clip(p0_ev_k1, 1e-12, 1.0)), -np.log1p(-p0_ev_k1))
        k0_mean_occ_nll = float(np.mean(nll_occ_k0))
        k1_mean_occ_nll = float(np.mean(nll_occ_k1))

        shared_mag_nll = np.where(pos_ev, _ztnp_nll(ye, lam), 0.0)
        shared_mean_mag_nll = float(np.mean(shared_mag_nll))

        delta_nll = k1_mean_nll - k0_mean_nll

        diag_rows.append(dict(
            window=w_name,
            target=c,
            train_positive_n=n_pos_tr,
            eval_positive_n=n_pos_ev,
            train_p_occ=train_p_occ,
            eval_p_occ=eval_p_occ,
            constant_ztp_lambda=lam,
            expected_magnitude=expected_mag,
            actual_train_positive_mean=actual_tr_pos_mean,
            actual_eval_positive_mean=actual_ev_pos_mean,
            predicted_p_y1=pred_p_y1,
            predicted_p_y2=pred_p_y2,
            predicted_p_y_ge3=pred_p_y_ge3,
            actual_train_p_y1=actual_tr_p_y1,
            actual_train_p_y2=actual_tr_p_y2,
            actual_train_p_y_ge3=actual_tr_p_y_ge3,
            actual_eval_p_y1=actual_ev_p_y1,
            actual_eval_p_y2=actual_ev_p_y2,
            actual_eval_p_y_ge3=actual_ev_p_y_ge3,
            k1_occ_logloss=k1_ll,
            k1_occ_brier=k1_brier,
            k1_occ_pr_auc=k1_prauc,
            k0_occ_logloss=k0_ll,
            k0_occ_brier=k0_brier,
            k0_mean_nll=k0_mean_nll,
            k1_mean_nll=k1_mean_nll,
            k0_mean_occ_nll=k0_mean_occ_nll,
            k1_mean_occ_nll=k1_mean_occ_nll,
            shared_mean_mag_nll=shared_mean_mag_nll,
            delta_nll=delta_nll,
        ))
    return diag_rows
# ===========================================================================
# cluster bootstrap (by episode_start_trading_day, exact cluster multiplicity)
# ===========================================================================
def boot_cluster(delta_per_row, day_per_row, eid_per_row, seed):
    df = pd.DataFrame({"d": delta_per_row, "day": day_per_row,
                       "eid": eid_per_row})
    ep = df.groupby("eid", as_index=False).agg(d=("d", "mean"), day=("day", "first"))
    day_stats = ep.groupby("day")["d"].agg(["sum", "count"]).reset_index()
    day_sums = day_stats["sum"].to_numpy(dtype=np.float64)
    day_counts = day_stats["count"].to_numpy(dtype=np.float64)
    n_days = len(day_stats)
    rng = np.random.default_rng(seed)
    ests = np.empty(BOOTSTRAP_REPS)
    for b in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, n_days, size=n_days)
        ests[b] = day_sums[idx].sum() / day_counts[idx].sum()
    point = float(ep["d"].mean())
    return float(np.percentile(ests, 2.5)), float(np.percentile(ests, 97.5)), point


def reconstruct_agezero(cur_df, z_df, side):
    """Deterministically reconstruct the next newest age-zero flag.

    Canonical provenance definition:
        expected_new_{t+1} = new_0 + elapsed_{t+1}
        where new_0 = expm1(start_newest_log_age)
              elapsed_{t+1} = bar_{t+1} - start_bar = bar_t + 1 - start_bar
        r_{t+1} = log1p(newest_{t+1}) - log1p(expected_new_{t+1})

    Therefore:
        newest_{t+1} = exp(r_{t+1} + log1p(new_0 + elapsed_{t+1})) - 1
        age_zero_{t+1} = 1[newest_{t+1} == 0]
    """
    node = "z_uresid" if side == "upper" else "z_lresid"
    cols = NODE_ZCOLS[node]
    resid_next = reconstruct_value(NODE_KIND[node],
                                   [z_df[col].to_numpy(float) for col in cols])
    start_age = np.expm1(cur_df[f"{side}_newest_log_age"].to_numpy(float))
    elapsed_next = (cur_df["bar_t"].to_numpy(np.int64) + 1 - cur_df["start_bar"].to_numpy(np.int64))
    expected_next = start_age + elapsed_next
    age_next = np.expm1(resid_next + np.log1p(expected_next))
    return np.isclose(age_next, 0.0, atol=1e-10).astype(int)


def reconstruct_next_state(cur_df, z_df):
    """Reconstruct S_{t+1} from S_t and the realized Z_{t+1} innovations.

    Returns a dict mapping each state column to a float64 numpy array. Must hold
    F(S_t, Z_{t+1}) == S_{t+1} exactly (<=1e-8) when Z is the realized draw, i.e.
    Z is *sufficient* to drive the within-episode transition. This is the core of
    the support-correct closure. Includes the 7 continuous nodes, the 2 count nodes,
    the deterministic companions (down-distance, width, ratio, TV, last-return) and
    the deterministically-reconstructed age-zero flags.
    """
    c = cur_df
    z = z_df
    out = {}
    out["cur_up_distance_R"] = (
        c["cur_up_distance_R"].to_numpy(float) + z["z_d_up"].to_numpy(float))
    for node, src in [("z_dmfe", "path_max_up_excursion_R"),
                      ("z_dmae", "path_max_down_excursion_R")]:
        cols = NODE_ZCOLS[node]
        val = reconstruct_value(NODE_KIND[node],
                                [z[col].to_numpy(float) for col in cols])
        out[src] = c[src].to_numpy(float) + val
    for node, src in [("z_dcr", "path_direction_change_rate"),
                      ("z_range", "path_current_bar_range_R"),
                      ("z_uresid", "upper_newest_log_age_residual"),
                      ("z_lresid", "lower_newest_log_age_residual")]:
        cols = NODE_ZCOLS[node]
        out[src] = reconstruct_value(NODE_KIND[node],
                                     [z[col].to_numpy(float) for col in cols])
    # deterministic companions (exact by construction)
    out["cur_down_distance_R"] = (
        c["cur_down_distance_R"].to_numpy(float) - z["z_d_up"].to_numpy(float))
    out["cur_width_R"] = c["cur_width_R"].to_numpy(float)  # constant within episode

    up = out["cur_up_distance_R"]
    dn = out["cur_down_distance_R"]
    # snap floating point subtraction noise near zero (< 1e-12)
    up = np.where(np.abs(up) < 1e-12, 0.0, up)
    dn = np.where(np.abs(dn) < 1e-12, 0.0, dn)
    out["cur_up_distance_R"] = up
    out["cur_down_distance_R"] = dn

    atr0 = c["atr0"].to_numpy(float) if "atr0" in c.columns else 1.0
    out["cur_log_ratio"] = np.log((up * atr0 + 1e-9) / (dn * atr0 + 1e-9))

    out["path_total_variation_R"] = (
        c["path_total_variation_R"].to_numpy(float)
        + np.abs(z["z_d_up"].to_numpy(float)))
    out["path_last_return_R"] = -z["z_d_up"].to_numpy(float)
    for side in ("upper", "lower"):
        out[f"{side}_active_identity_count_delta"] = (
            c[f"{side}_active_identity_count_delta"].to_numpy(float)
            + z[f"z_delta_{side}_count"].to_numpy(float))
    azo_u = reconstruct_agezero(c, z, "upper")
    azo_l = reconstruct_agezero(c, z, "lower")
    out["upper_current_newest_age_zero"] = azo_u.astype(float)
    out["lower_current_newest_age_zero"] = azo_l.astype(float)
    out["z_agezero_code"] = (azo_u + 2 * azo_l).astype(float)
    out["bar_t_next"] = c["bar_t"].to_numpy(float) + 1.0
    return out


def audit_agezero_reconstruction(cur_df, z_df, nxt_df=None):
    """Hard audit: can the next age-zero be reconstructed deterministically from the
    realized residual Z_{t+1} + current age (see reconstruct_agezero)?

    If `nxt_df` is given, the reconstruction is compared against the *true* next
    current_newest_age_zero columns of the shifted frame. Also audits the actual
    newest age trajectory: delta = age_{t+1} - age_t (<0, =0, =1, >1, min, max).

    Gate (per governance): if mismatch == 0 -> the 4-class stochastic age-zero node
    is a deterministic derived state and MUST be deleted from the model. If mismatch
    > 0 -> STOP (investigate why the canonical residual cannot reconstruct age).
    """
    results = {}
    for side in ("upper", "lower"):
        recon = reconstruct_agezero(cur_df, z_df, side)
        if nxt_df is not None:
            stored = nxt_df[f"{side}_current_newest_age_zero"].to_numpy(int)
            start_age = np.expm1(cur_df[f"{side}_newest_log_age"].to_numpy(float))
            elapsed_cur = (cur_df["bar_t"].to_numpy(np.int64) - cur_df["start_bar"].to_numpy(np.int64))
            elapsed_next = elapsed_cur + 1

            resid_cur = cur_df[f"{side}_newest_log_age_residual"].to_numpy(float)
            resid_next = nxt_df[f"{side}_newest_log_age_residual"].to_numpy(float)

            age_cur = np.expm1(resid_cur + np.log1p(start_age + elapsed_cur))
            age_next = np.expm1(resid_next + np.log1p(start_age + elapsed_next))
            delta = age_next - age_cur
            delta_clean = np.round(delta, 6)

            n_lt_0 = int(np.sum(delta_clean < 0))
            n_eq_0 = int(np.sum(delta_clean == 0))
            n_eq_1 = int(np.sum(delta_clean == 1))
            n_gt_1 = int(np.sum(delta_clean > 1))
            min_val = float(np.min(delta))
            max_val = float(np.max(delta))

            delta_audit = dict(
                n_lt_0=n_lt_0,
                p_lt_0=float(n_lt_0 / len(cur_df)),
                n_eq_0=n_eq_0,
                p_eq_0=float(n_eq_0 / len(cur_df)),
                n_eq_1=n_eq_1,
                p_eq_1=float(n_eq_1 / len(cur_df)),
                n_gt_1=n_gt_1,
                p_gt_1=float(n_gt_1 / len(cur_df)),
                min=min_val,
                max=max_val,
            )
        else:
            stored = recon
            delta_audit = {}
        mismatch = int((recon != stored).sum())
        results[side] = dict(exact=(mismatch == 0), n_mismatch=mismatch,
                             n=len(recon), delta_newest_age=delta_audit)
    return results


def audit_full_reconstruction(cur_df, z_df, nxt_df):
    """Full state-reconstruction hard gate on REAL transitions.

    Reconstructs S_{t+1} = F(S_t, Z_{t+1}) for every transition and compares to the
    true next state.
    Two tiers:
      * state closure (HARD gate): exact-deterministic columns whose value is fully
        fixed by (S_t, Z_{t+1}) -- distances, width, cur_log_ratio, TV, MFE/MAE,
        last-return, range, DCR, newest residuals, activation counters.
        Requires state_max_error < 1e-8.
      * derived representations (HARD gate for ratio, reported for tempo):
        cur_log_ratio error must be < 1e-8 (matches canonical formula with EPS=1e-9).
        Tempo columns (derived return-speed representations) are reported for transparency.

    discrete_mismatch (age-zero flags + code) must be 0. If any gate fails, training
    must not start (STOP_DYNAMIC_PGM1A1_STATE_RECONSTRUCTION_FAIL).
    """
    recon = reconstruct_next_state(cur_df, z_df)
    state_cols = [
        "cur_up_distance_R", "cur_down_distance_R", "cur_width_R", "cur_log_ratio",
        "path_total_variation_R", "path_max_up_excursion_R", "path_max_down_excursion_R",
        "path_last_return_R", "path_current_bar_range_R", "path_direction_change_rate",
        "upper_newest_log_age_residual", "lower_newest_log_age_residual",
        "upper_active_identity_count_delta", "lower_active_identity_count_delta",
    ]
    rep_cols = ["cur_log_ratio"]
    rep_cols += [c for c in nxt_df.columns if "tempo" in c.lower()]
    per_state = {}
    state_max_err = 0.0
    for col in state_cols:
        a = np.asarray(recon[col], dtype=np.float64)
        b = np.asarray(nxt_df[col].to_numpy(float), dtype=np.float64)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() == 0:
            per_state[col] = None
            continue
        err = float(np.max(np.abs(a[m] - b[m])))
        per_state[col] = err
        state_max_err = max(state_max_err, err)
    per_rep = {}
    for col in rep_cols:
        a = np.asarray(recon.get(col), dtype=np.float64) \
            if col in recon else np.asarray(nxt_df[col].to_numpy(float), dtype=np.float64)
        b = np.asarray(nxt_df[col].to_numpy(float), dtype=np.float64)
        m = np.isfinite(a) & np.isfinite(b)
        per_rep[col] = (None if m.sum() == 0 else float(np.max(np.abs(a[m] - b[m]))))
    disc_mismatch = 0
    for side in ("upper", "lower"):
        a = recon[f"{side}_current_newest_age_zero"].astype(int)
        b = nxt_df[f"{side}_current_newest_age_zero"].to_numpy(int)
        disc_mismatch += int((a != b).sum())
    a_code = recon["z_agezero_code"].astype(int)
    b_code = (nxt_df["upper_current_newest_age_zero"].to_numpy(int)
              + 2 * nxt_df["lower_current_newest_age_zero"].to_numpy(int))
    disc_mismatch += int((a_code != b_code).sum())
    return dict(state_max_error=state_max_err,
                derived_representation_max_error=float(per_rep.get("cur_log_ratio", 0.0)),
                per_column_max_error=per_state,
                representation_errors=per_rep,
                discrete_mismatch=disc_mismatch, n=len(nxt_df))


def run_single_window(w, data_path):
    t_w = time.perf_counter()
    cur = pd.read_parquet(data_path)
    _peak(f"{w['name']} loaded cur")

    tr = cur[cur["block"].isin(w["train"])].reset_index(drop=True)
    ev = cur[cur["block"] == w["eval"]].reset_index(drop=True)
    if not len(tr) or not len(ev):
        raise SystemExit(f"STOP_DYNAMIC_PGM1A_EMPTY_SPLIT: {w['name']}")
    del cur
    _peak(f"{w['name']} after split train/eval")
    print(f"[STAGE] window {w['name']} n_train={len(tr)} n_eval={len(ev)}", flush=True)

    Zc_tr = tr[ALL_Z_COLS].to_numpy(np.float32)
    Zc_ev = ev[ALL_Z_COLS].to_numpy(np.float32)
    yd_tr = tr[DISC_Z].to_numpy(np.int64)
    yd_ev = ev[DISC_Z].to_numpy(np.int64)
    Yc_tr = tr[COUNT_Z].to_numpy(np.int64)
    Yc_ev = ev[COUNT_Z].to_numpy(np.int64)
    sym_ev = ev["symbol"].to_numpy()
    day_ev = ev["episode_start_day"].to_numpy()
    eid_ev = ev["episode_id"].to_numpy()

    model_metrics, opt_rows, block_rows, target_rows = [], [], [], []
    node_diag_rows = []

    def _node_eval_nll(nodes):
        return np.sum([nodes[n]["ev"] for n, _ in Z_LAYOUT], axis=0)

    # ---------------- K0: constant per-node heads + marginal 4-class + constant count
    t_k0 = time.perf_counter()
    k0 = fit_constant_heads(Zc_tr, Zc_ev, yd_tr, yd_ev)
    k0_count = fit_constant_count_head(Yc_tr, Yc_ev)
    opt_rows.append(dict(window=w["name"], model="K0_UNCONDITIONAL",
                        n_params=k0["n_params"] + k0_count["n_params"], success=True,
                        elapsed_seconds=round(time.perf_counter() - t_k0, 3)))
    cont_ev_k0 = _node_eval_nll(k0["nodes"])
    count_ev_k0 = k0_count["nll_ev"].sum(axis=1)
    disc_ev_k0 = k0["disc_ev"]
    j_k0_ev = cont_ev_k0 + disc_ev_k0 + count_ev_k0

    mean_cont_k0 = float(np.mean(cont_ev_k0))
    mean_count_k0 = float(np.mean(count_ev_k0))
    mean_disc_k0 = float(np.mean(disc_ev_k0))
    mean_joint_k0 = float(np.mean(j_k0_ev))
    assert abs(mean_joint_k0 - mean_cont_k0 - mean_disc_k0 - mean_count_k0) < 1e-10

    model_metrics.append(dict(window=w["name"], model="K0_UNCONDITIONAL",
                            mean_joint_nll=mean_joint_k0,
                            mean_cont_nll=mean_cont_k0,
                            mean_disc_nll=mean_disc_k0,
                            mean_count_nll=mean_count_k0,
                            n_rows=int(len(ev))))

    # ---------------- Preprocessing for K1 (OBS_STATE_CAT + OBS_STATE_NUM)
    t_k1 = time.perf_counter()
    ct = ColumnTransformer([
        ("cat", Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))]),
         list(OBS_STATE_CAT)),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]),
         list(OBS_STATE_NUM)),
    ])
    Xtr1 = ct.fit_transform(tr).astype(np.float32)
    Xev1 = ct.transform(ev).astype(np.float32)
    _peak(f"{w['name']} after transform")

    # ---------------- K1: OBSERVED-STATE-v1 + Count Magnitude Closure
    k1 = fit_state_heads(Xtr1, Xev1, Zc_tr, Zc_ev, yd_tr, yd_ev)
    k1_count = fit_state_count_head(Xtr1, Yc_tr, Xev1, Yc_ev, constant_rates=k0_count["constant_rates"])
    _peak(f"{w['name']} after K1 transform+fit")
    opt_rows.append(dict(window=w["name"], model="K1_STATE",
                        n_params=k1["n_params"] + k1_count["n_params"], success=True,
                        elapsed_seconds=round(time.perf_counter() - t_k1, 3)))
    cont_ev_k1 = _node_eval_nll(k1["nodes"])
    count_ev_k1 = k1_count["nll_ev"].sum(axis=1)
    disc_ev_k1 = k1["disc_ev"]
    j_k1_ev = cont_ev_k1 + disc_ev_k1 + count_ev_k1

    mean_cont_k1 = float(np.mean(cont_ev_k1))
    mean_count_k1 = float(np.mean(count_ev_k1))
    mean_disc_k1 = float(np.mean(disc_ev_k1))
    mean_joint_k1 = float(np.mean(j_k1_ev))
    assert abs(mean_joint_k1 - mean_cont_k1 - mean_disc_k1 - mean_count_k1) < 1e-10

    model_metrics.append(dict(window=w["name"], model="K1_STATE",
                            mean_joint_nll=mean_joint_k1,
                            mean_cont_nll=mean_cont_k1,
                            mean_disc_nll=mean_disc_k1,
                            mean_count_nll=mean_count_k1,
                            n_rows=int(len(ev))))

    # ---------------- Covariance PD audit
    def _head_min_eig(h):
        for cand in (h, getattr(h, "g", None)):
            if cand is not None and hasattr(cand, "cov"):
                return float(np.linalg.eigvalsh(
                    np.asarray(cand.cov, dtype=np.float64)).min())
        return None
    cov_audit = {w["name"]: {}}
    for nm, d in k1["nodes"].items():
        e = _head_min_eig(d["head"])
        cov_audit[w["name"]][nm] = dict(
            min_eig=(e if e is not None else "n/a"),
            pd=bool(e is not None and e > 0))

    # ---------------- Count magnitude closure diagnostics
    count_diag_rows = compute_count_closure_diagnostics(w["name"], Yc_tr, Yc_ev, k0_count, k1_count)

    # ---------------- Bootstrap K1-K0
    boots, bysym = [], []
    delta = j_k1_ev - j_k0_ev
    t_b = time.perf_counter()
    print(f"[STAGE] window {w['name']} bootstrap K1-K0 start", flush=True)
    lo_ci, hi_ci, point = boot_cluster(delta, day_ev, eid_ev, w["seed"])
    print(f"[STAGE] window {w['name']} bootstrap K1-K0 done "
          f"took={round(time.perf_counter() - t_b, 2)}", flush=True)

    df_d = pd.DataFrame({"symbol": sym_ev, "eid": eid_ev, "delta": delta})
    ep_d = df_d.groupby(["symbol", "eid"], as_index=False)["delta"].mean()
    for sym, grp in ep_d.groupby("symbol"):
        m_ep = float(grp["delta"].mean())
        raw_grp = df_d[df_d["symbol"] == sym]
        m_row = float(raw_grp["delta"].mean())
        d_raw = raw_grp["delta"].to_numpy()
        bysym.append(dict(
            window=w["name"], comparison="K1-K0", symbol=sym,
            n_episodes=int(len(grp)),
            n_rows=int(len(raw_grp)),
            mean_delta_episode=m_ep,
            mean_delta_row=m_row,
            mean_delta=m_ep,
            n_negative=int((d_raw < 0).sum()),
            n_positive=int((d_raw > 0).sum()),
        ))
    boots.append(dict(window=w["name"], comparison="K1-K0",
                      delta_sample_mean=point, ci_lo=lo_ci, ci_hi=hi_ci,
                      verdict=("CI_below_zero" if hi_ci < 0
                               else "CI_above_zero" if lo_ci > 0
                               else "CI_contains_zero")))

    # ---------------- Block contribution (eval)
    def _block_nll(nodes, disc_ev, count_ev, bdef):
        if bdef["nodes"]:
            s = np.sum([nodes[n]["ev"] for n in bdef["nodes"]], axis=0)
        else:
            s = np.zeros(len(nodes[list(nodes)[0]]["ev"]))
        if bdef["disc"]:
            s = s + disc_ev
        if bdef.get("count"):
            idx = [COUNT_Z.index(c) for c in bdef["count"]]
            s = s + count_ev[:, idx].sum(axis=1)
        return s

    for bname, bdef in BLOCKS.items():
        bjoint_k0 = _block_nll(k0["nodes"], k0["disc_ev"], k0_count["nll_ev"], bdef)
        bjoint_k1 = _block_nll(k1["nodes"], k1["disc_ev"], k1_count["nll_ev"], bdef)
        m0 = float(np.mean(bjoint_k0))
        m1 = float(np.mean(bjoint_k1))
        block_rows.append(dict(
            window=w["name"], block=bname,
            mean_joint_k0=m0,
            mean_joint_k1=m1,
            delta_k1_minus_k0=m1 - m0))

    # ---------------- Per-target (eval)
    for nm, cols in Z_LAYOUT:
        n0 = k0["nodes"][nm]["ev"]
        n1 = k1["nodes"][nm]["ev"]
        m0, m1 = float(np.mean(n0)), float(np.mean(n1))
        target_rows.append(dict(window=w["name"], target=nm, kind="node",
                                mean_nll_k0=m0,
                                mean_nll_k1=m1,
                                delta_k1_minus_k0=m1 - m0))
    for j, c in enumerate(COUNT_Z):
        n0 = k0_count["nll_ev"][:, j]
        n1 = k1_count["nll_ev"][:, j]
        m0, m1 = float(np.mean(n0)), float(np.mean(n1))
        target_rows.append(dict(window=w["name"], target=c, kind="discrete_count",
                                mean_nll_k0=m0,
                                mean_nll_k1=m1,
                                delta_k1_minus_k0=m1 - m0))
    if disc_spec():
        m0 = float(np.mean(k0["disc_ev"]))
        m1 = float(np.mean(k1["disc_ev"]))
        target_rows.append(dict(window=w["name"], target=DISC_Z, kind="discrete_4class",
                                mean_nll_k0=m0,
                                mean_nll_k1=m1,
                                delta_k1_minus_k0=m1 - m0))

    # ---------------- Node diagnostics (K1 eval metrics)
    z_layout_map = dict(Z_LAYOUT)
    for nm in ["z_dmfe", "z_dmae", "z_range", "z_uresid", "z_lresid"]:
        if nm in k1["nodes"]:
            h = k1["nodes"][nm]["head"]
            if hasattr(h, "logit"):
                cols_idx = z_layout_map[nm]
                ispos_col_idx = cols_idx[0]
                y_true = (Zc_ev[:, ispos_col_idx] > 0.5).astype(int)
                prob = np.clip(h.logit.predict_proba(Xev1)[:, 1], 1e-6, 1.0 - 1e-6)
                prev = float(np.mean(y_true))
                ll = float(log_loss(y_true, prob, labels=[0, 1]))
                brier = float(brier_score_loss(y_true, prob))
                prauc = float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan")
                node_diag_rows.append(dict(
                    window=w["name"], model="K1_STATE", target=nm, kind="hurdle_occurrence",
                    prevalence=prev, logloss=ll, brier=brier, pr_auc=prauc,
                    p_zero=None, p_interior=None, p_one=None, cat_nll=None, interior_nll=None,
                    actual_mean_increment=None, predicted_mean_increment=None, mean_ztp_lambda=None,
                ))

    if "z_dcr" in k1["nodes"]:
        hdcr = k1["nodes"]["z_dcr"]["head"]
        if hasattr(hdcr, "cat") and hasattr(hdcr, "g"):
            cols_idx = z_layout_map["z_dcr"]
            is0_ev = Zc_ev[:, cols_idx[0]] > 0.5
            is1_ev = Zc_ev[:, cols_idx[1]] > 0.5
            int_ev = (~is0_ev) & (~is1_ev)
            y3_ev = np.where(is0_ev, 0, np.where(is1_ev, 2, 1)).astype(int)
            p3 = np.clip(hdcr.cat.predict_proba(Xev1), 1e-12, 1.0)
            p3 = p3 / p3.sum(axis=1, keepdims=True)
            cat_nll = float(log_loss(y3_ev, p3, labels=[0, 1, 2]))
            if int_ev.sum() > 0:
                logit_ev = Zc_ev[int_ev, cols_idx[2]]
                nll_logit = hdcr.g.nll_per_row(Xev1[int_ev], logit_ev.reshape(-1, 1))
                jac = _logit_jacobian(logit_ev)
                int_nll = float(np.mean(nll_logit + jac))
            else:
                int_nll = 0.0
            node_diag_rows.append(dict(
                window=w["name"], model="K1_STATE", target="z_dcr", kind="dcr",
                prevalence=None, logloss=None, brier=None, pr_auc=None,
                p_zero=float(np.mean(is0_ev)),
                p_interior=float(np.mean(int_ev)),
                p_one=float(np.mean(is1_ev)),
                cat_nll=cat_nll, interior_nll=int_nll,
                actual_mean_increment=None, predicted_mean_increment=None, mean_ztp_lambda=None,
            ))

    for j, c in enumerate(COUNT_Z):
        ye = Yc_ev[:, j]
        ze = (ye > 0).astype(int)
        p0_ev = k1_count["p0_ev"][:, j]
        rate_ev = k1_count["rate_ev"][:, j]
        prev = float(np.mean(ze))
        ll = float(log_loss(ze, p0_ev, labels=[0, 1]))
        brier = float(brier_score_loss(ze, p0_ev))
        prauc = float(average_precision_score(ze, p0_ev)) if len(np.unique(ze)) > 1 else float("nan")
        act_inc = float(np.mean(ye))
        denom = np.maximum(-np.expm1(-rate_ev), 1e-12)
        pred_inc = float(np.mean(p0_ev * (rate_ev / denom)))
        mean_lam = float(np.mean(rate_ev))
        node_diag_rows.append(dict(
            window=w["name"], model="K1_STATE", target=c, kind="hurdle_count_closure",
            prevalence=prev, logloss=ll, brier=brier, pr_auc=prauc,
            p_zero=None, p_interior=None, p_one=None, cat_nll=None, interior_nll=None,
            actual_mean_increment=act_inc, predicted_mean_increment=pred_inc, mean_ztp_lambda=mean_lam,
        ))

    del Xtr1, Xev1, k1, k0, tr, ev, ct
    del Zc_tr, Zc_ev, yd_tr, yd_ev
    del j_k0_ev, j_k1_ev

    print(f"[STAGE] window {w['name']} done "
          f"took={round(time.perf_counter() - t_w, 2)}s", flush=True)
    return dict(window=w["name"],
                model_metrics=model_metrics,
                opt_rows=opt_rows, boots=boots, bysym=bysym,
                block_rows=block_rows, target_rows=target_rows,
                node_diag_rows=node_diag_rows,
                count_diag_rows=count_diag_rows,
                cov_audit=cov_audit)


def main():
    t_total = time.perf_counter()
    timing = {}

    t0 = time.perf_counter()
    keep_cols = (list(OBS_STATE_NUM) + list(OBS_STATE_CAT) + list(LAG_BASE)
                 + BUILD_SRC_COLS
                 + ["hazard", "block", "symbol", "episode_id",
                    "episode_start_day", "bar_t"])
    keep_cols = list(dict.fromkeys(keep_cols))
    df = pd.read_parquet(
        CACHE / "market_state1_samples.parquet", columns=keep_cols)
    _peak("after read+prune (slim, native dtype)")

    build_set = set(BUILD_SRC_COLS)
    for c in df.columns:
        if c not in build_set and str(df[c].dtype).startswith("float64"):
            df[c] = df[c].astype(np.float32)
    cur, nxt = build_transition_sample(df)
    _peak("after build_transition_sample")
    cur = add_lag1(cur)
    _peak("after add_lag1")
    del df
    timing["build_transition_seconds"] = round(time.perf_counter() - t0, 2)
    print(f"[STAGE] build done n={len(cur)}", flush=True)

    # ---------------- sample audit / hash
    hash_cols = ALL_Z_COLS + [DISC_Z] + COUNT_Z
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
        count_increment_integer=bool(all(
            np.all(np.equal(np.mod(cur[f"z_delta_{s}_count"].to_numpy(float), 1), 0))
            for s in ("upper", "lower"))),
        count_increment_nonneg=bool(all(
            cur[f"z_delta_{s}_count"].to_numpy(float).min() >= -1e-9
            for s in ("upper", "lower"))),
        count_reconstruction_exact=bool(all(
            np.allclose(
                cur[f"z_delta_{s}_count"].to_numpy(float)
                + cur[f"{s}_active_identity_count_delta"].to_numpy(float),
                nxt[f"{s}_active_identity_count_delta"].to_numpy(float), atol=1e-9)
            for s in ("upper", "lower"))),
    )
    if not all(invariants.values()):
        raise SystemExit(
            f"STOP_DYNAMIC_PGM1A_INVARIANT_FAIL: {invariants}")

    support_audit = dict(
        upper_residual=audit_support_residual(
            nxt["upper_newest_log_age_residual"].to_numpy(float), name="upper_residual"),
        lower_residual=audit_support_residual(
            nxt["lower_newest_log_age_residual"].to_numpy(float), name="lower_residual"),
        range=audit_support_range(
            nxt["path_current_bar_range_R"].to_numpy(float), name="range"),
    )
    for key, rep in [("z_uresid", support_audit["upper_residual"]),
                     ("z_lresid", support_audit["lower_residual"]),
                     ("z_range", support_audit["range"])]:
        if rep["recommended_kind"] != NODE_KIND[key]:
            raise SystemExit(
                f"STOP_DYNAMIC_PGM1A_SUPPORT_SELECTION_MISMATCH: {key} "
                f"audit={rep['recommended_kind']} baked={NODE_KIND[key]}")

    full_recon = audit_full_reconstruction(cur, cur, nxt)
    agezero_audit = audit_agezero_reconstruction(cur, cur, nxt)
    if (full_recon["state_max_error"] >= 1e-8
            or full_recon["derived_representation_max_error"] >= 1e-8
            or full_recon["discrete_mismatch"] != 0):
        raise SystemExit(
            f"STOP_DYNAMIC_PGM1A1_STATE_RECONSTRUCTION_FAIL: {full_recon}")
    az_u = agezero_audit["upper"]["n_mismatch"]
    az_l = agezero_audit["lower"]["n_mismatch"]
    if az_u != 0 or az_l != 0:
        raise SystemExit(
            f"STOP_DYNAMIC_PGM1A1_AGEZERO_RECON_MISMATCH: {agezero_audit}")
    configure_child_semantics(agezero_deterministic=True)
    if disc_spec() != []:
        raise SystemExit(
            f"STOP_DYNAMIC_PGM1A1_DISC_SPEC_NONEMPTY: {disc_spec()}")

    audit_report = dict(
        support=support_audit,
        full_reconstruction=full_recon,
        agezero=agezero_audit,
        agezero_node_removed=AGEZERO_DETERMINISTIC,
    )
    (OUT / "dynamic_pgm1a1b_reconstruction_audit.json").write_text(
        json.dumps(audit_report, indent=2, default=str))
    print(f"[AUDIT] support={support_audit} "
          f"recon_maxerr={full_recon['state_max_error']:.3e} "
          f"derived_rep_maxerr={full_recon['derived_representation_max_error']:.3e} "
          f"disc_mismatch={full_recon['discrete_mismatch']} "
          f"agezero_mismatch(u/l)={az_u}/{az_l}", flush=True)

    del nxt

    if os.environ.get("DYNAMIC_PGM1A_AUDIT_ONLY") == "1" or "--audit-only" in sys.argv:
        print("[AUDIT-ONLY] Complete. dynamic_pgm1a1b_reconstruction_audit.json written. Stopping.", flush=True)
        return audit_report

    needed_cols = (list(OBS_STATE_NUM) + list(OBS_STATE_CAT)
                   + ALL_Z_COLS + [DISC_Z] + COUNT_Z
                   + ["episode_id", "symbol", "block", "episode_start_day"])
    for c in cur.columns:
        if str(cur[c].dtype).startswith("float64"):
            cur[c] = cur[c].astype(np.float32)
    cur = cur[needed_cols].copy()
    cur.to_parquet(CACHE / "dynamic_pgm1a1b_transitions.parquet", index=False)
    _peak("after slim cur")

    data_path = CACHE / "dynamic_pgm1a1b_transitions.parquet"
    win_env = dict(os.environ)
    for _bt in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        win_env[_bt] = "1"

    model_metrics, opt_rows, boots, bysym = [], [], [], []
    block_rows, target_rows, node_diag_rows, count_diag_rows = [], [], [], []
    cov_audit = {}
    for w in WINDOWS:
        result_path = CACHE / f"_dynamic_pgm1a1b_window_{w['name']}.json"
        cmd = [sys.executable, str(Path(__file__)),
               "--window-json", json.dumps(w),
               "--data", str(data_path), "--result", str(result_path)]
        if AGEZERO_DETERMINISTIC:
            cmd.append("--agezero-deterministic")
        else:
            raise SystemExit(
                "STOP_DYNAMIC_PGM1A1_AGEZERO_NOT_DETERMINISTIC_BEFORE_WINDOWS")
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
        node_diag_rows += res.get("node_diag_rows", [])
        count_diag_rows += res.get("count_diag_rows", [])
        cov_audit.update(res["cov_audit"])
        print(f"[STAGE] window {w['name']} subprocess done "
              f"took={round(time.perf_counter() - t_w, 2)}s", flush=True)

    def sym_neg(wname, label):
        sub = pd.DataFrame(bysym)
        if len(sub) == 0:
            return 0
        sub = sub[(sub["window"] == wname) & (sub["comparison"] == label)]
        return int((sub["mean_delta_episode"] < 0).sum())

    verdict = {}
    wsA = {w["name"]: {} for w in WINDOWS}
    for w in WINDOWS:
        row = next(b for b in boots if b["window"] == w["name"]
                   and b["comparison"] == "K1-K0")
        wsA[w["name"]] = dict(ci_hi=row["ci_hi"],
                              n_sym_neg=sym_neg(w["name"], "K1-K0"))
    gateA = (all(wsA[w["name"]]["ci_hi"] is not None and wsA[w["name"]]["ci_hi"] < 0 for w in WINDOWS)
             and all(wsA[w["name"]]["n_sym_neg"] >= 10 for w in WINDOWS))
    transition_verdict = (
        "SUPPORT_CORRECT_TRANSITION_SIGNAL_SUPPORTED" if gateA
        else "SUPPORT_CORRECT_TRANSITION_SIGNAL_NOT_SUPPORTED"
    )
    verdict["SUPPORT_CORRECT_TRANSITION_SIGNAL_SUPPORTED"] = dict(
        supported=bool(gateA), verdict=transition_verdict, windows=wsA)

    print("[STAGE] writing outputs", flush=True)
    pd.DataFrame(model_metrics).to_csv(
        OUT / "dynamic_pgm1a1b_model_metrics.csv", index=False)
    pd.DataFrame(boots).to_csv(OUT / "dynamic_pgm1a1b_bootstrap.csv", index=False)
    pd.DataFrame(bysym).to_csv(OUT / "dynamic_pgm1a1b_by_symbol.csv", index=False)
    pd.DataFrame(target_rows).to_csv(
        OUT / "dynamic_pgm1a1b_target_metrics.csv", index=False)
    pd.DataFrame(opt_rows).to_csv(
        OUT / "dynamic_pgm1a1b_optimizer_audit.csv", index=False)
    pd.DataFrame(node_diag_rows).to_csv(
        OUT / "dynamic_pgm1a1b_node_diagnostics.csv", index=False)
    pd.DataFrame(count_diag_rows).to_csv(
        OUT / "dynamic_pgm1a1b_count_closure_diagnostics.csv", index=False)
    (OUT / "dynamic_pgm1a1b_sample_audit.json").write_text(
        json.dumps(sample_audit, indent=2))
    (OUT / "dynamic_pgm1a1b_transition_invariants.json").write_text(
        json.dumps(invariants, indent=2))

    summary = dict(
        experiment="DYNAMIC-PGM-1A.1b Count Magnitude Closure Transition Kernel",
        parent_commit=BASE_SHA,
        transition_verdict=transition_verdict,
        k2_status="POSTPONED_TO_1A2_LAG_CLOSURE",
        k2_note="K2 (lag1) is excluded from 1A.1b; reserved for DYNAMIC-PGM-1A.2 Lag Closure.",
        design=dict(
            scope="within-episode 5m state transition, terminal reset excluded",
            target="P(S_{t+1} | S_t, H_{t+1}=0)",
            count_magnitude_closure="P(A|S) = P(A>0|S) * P(A|A>0); occurrence state-dependent Logistic; magnitude state-independent train-constant exact ZTP",
            predicted_innovations=dict(
                nodes=[n for n, _ in NODE_SPECS],
                z_columns=ALL_Z_COLS,
                discrete="deterministic derived state; stochastic node removed",
                count_increments=COUNT_Z,
                note="S_{t+1}=F(S_t,Z_{t+1}); count increments use Count Magnitude Closure: "
                     "occurrence P(A>0|S) is state-dependent Logistic (K1) vs constant (K0); "
                     "magnitude P(A|A>0) is state-independent train-constant exact ZTP, shared identically across K0 and K1."),
            models=dict(
                K0_UNCONDITIONAL="no state; per-node constant heads + marginal occurrence + constant ZTP",
                K1_STATE="OBSERVED-STATE-v1 (35 numeric + prev_event_mask) + Logistic occurrence + shared constant ZTP"),
            kernel=dict(
                z_d_up="Gaussian delta (any real)",
                z_dmfe="Hurdle-LogNormal delta (P(=0) Logistic + log(Δ) Gaussian)",
                z_dmae="Hurdle-LogNormal delta (P(=0) Logistic + log(Δ) Gaussian)",
                z_range="Hurdle-LogNormal value (P(=0) Logistic + log(S) Gaussian)",
                z_uresid="Hurdle-LogNormal on q = -residual >= 0 (point mass at 0 + log(q) Gaussian)",
                z_lresid="Hurdle-LogNormal on q = -residual >= 0 (point mass at 0 + log(q) Gaussian)",
                z_dcr="0/interior/1 (3-class Logistic + logit(S) Gaussian on interior)",
                z_delta_upper_count="Logistic occurrence + constant exact ZTP",
                z_delta_lower_count="Logistic occurrence + constant exact ZTP",
                age_zero="deterministic derived state; stochastic node removed",
                discrete="deterministic derived state; stochastic node removed"),
            obs_state_numeric=OBS_STATE_NUM,
            obs_state_categorical=OBS_STATE_CAT,
            blocks={b: dict(nodes=v["nodes"], disc=v["disc"], count=v.get("count", []))
                    for b, v in BLOCKS.items()},
            tempo_classification="DERIVED_REPRESENTATION (not new state info)",
        ),
        windows=[dict(name=w["name"], train=w["train"], eval=w["eval"],
                      seed=w["seed"]) for w in WINDOWS],
        n_transition_rows=EXPECTED_TRANSITIONS,
        primary_metric="joint transition NLL (continuous + discrete + count)",
        model_metrics=model_metrics,
        bootstrap=boots,
        by_symbol_negative_counts={w["name"]: {
            "K1-K0": sym_neg(w["name"], "K1-K0")} for w in WINDOWS},
        block_contribution=block_rows,
        per_target=target_rows,
        node_diagnostics_file="dynamic_pgm1a1b_node_diagnostics.csv",
        count_closure_diagnostics_file="dynamic_pgm1a1b_count_closure_diagnostics.csv",
        covariance_audit=cov_audit,
        verdict=verdict,
        bootstrap_reps=BOOTSTRAP_REPS,
    )
    (OUT / "dynamic_pgm1a1b_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    timing["total_seconds"] = round(time.perf_counter() - t_total, 2)
    s2 = json.loads((OUT / "dynamic_pgm1a1b_summary.json").read_text())
    s2["timing"] = timing
    (OUT / "dynamic_pgm1a1b_summary.json").write_text(
        json.dumps(s2, indent=2, default=str))

    print(f"[MODEL METRICS]\n{pd.DataFrame(model_metrics).to_string(index=False)}")
    print(f"[BOOTSTRAP]\n{pd.DataFrame(boots).to_string(index=False)}")
    print(f"[BLOCK CONTRIBUTION]\n{pd.DataFrame(block_rows).to_string(index=False)}")
    print(f"[COUNT CLOSURE DIAG]\n{pd.DataFrame(count_diag_rows).to_string(index=False)}")
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
    _ap.add_argument("--agezero-deterministic", action="store_true",
                     help="Child subprocess mode: age-zero is deterministic derived state, drop 4-class discrete node.")
    _ap.add_argument("--audit-only", action="store_true",
                     help="Run real-data support and reconstruction audits only, then stop.")
    _args = _ap.parse_args()
    if _args.audit_only:
        os.environ["DYNAMIC_PGM1A_AUDIT_ONLY"] = "1"
    if _args.window_json:
        if not _args.agezero_deterministic:
            raise SystemExit(
                "STOP_DYNAMIC_PGM1A1_CHILD_AGEZERO_STATE_NOT_PROPAGATED"
            )
        configure_child_semantics(agezero_deterministic=True)
        if disc_spec() != []:
            raise SystemExit(
                "STOP_DYNAMIC_PGM1A1_CHILD_DISC_SPEC_NONEMPTY"
            )
        _w = json.loads(_args.window_json)
        _res = run_single_window(_w, _args.data)
        Path(_args.result).write_text(json.dumps(_res, default=str))
    else:
        main()
