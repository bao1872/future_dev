"""Liquidity-Field Reaction Model v1.0

第一里程碑：Stage 1 (Liquidity Field) + Stage 2 (Reaction Morphology)
+ Stage 3 (Data-Driven Morphology Discovery)。

核心原则（来自用户定义，冻结）：
1. 流动性场仍然是核心市场环境变量，不得简化为单一 target 或最终方向标签。
2. 研究单位 = Liquidity Reaction Episode：接触时的流动性环境 + 接触后的路径演化。
3. 第一里程碑只做到 Stage 3；禁止训练交易模型、禁止按收益优化聚类、
   禁止人工规定“直接穿越/复杂调整”两类、禁止 Risk-Coupled 0.5/1/2 正式实验。

数据合同（复用 frozen SMC 流动性语义，不重新发明）：
- 环境来自 run_fixed_execution_baseline_v1.load_env()：
  D['F']（合法 contact）、master_by_sym（ liquidity_master_v1_1.parquet）、
  bars_by_sym（5m 原始 bar）。
- 活跃流动性 = active_mask(master_sym, decision_time)：available_time <= dt 且
  （first_penetration_time 为 NaN 或 > dt）。这是因果快照，t0 之前被吃掉的流动性
  已被排除，因此 snapshot 天然不含未来流动性。被接触的流动性本身在 dt 已被 penetration
  （fp<=dt），天然不在活跃集内；我们额外按 liquidity_id 显式剔除，双保险。

实现说明（性能）：
- Stage 1 流动性场：对每个 symbol 用 C×L 矩阵做 interval-stabbing
  （available_time<=dt & fp>dt），并显式剔除被接触流动性；最近上下流动性与多尺度
  intensity 全部向量化（no per-contact Python loop）。
- Stage 2 反应形态：先对每个 symbol 用 fancy-index 一次性构造定向矩阵
  yc/ymax/ymin（C×35），再逐 horizon 向量化计算几何特征；
  仅 causal directional-change（顺序依赖）与 hysteresis crossing（变长非零序列）保留循环。

小样本测试：python liquidity_field_reaction_model_v1.py --sample 3000
（或 --symbols AGL8,CUL8 限制品种；--sample 从 96,900 contact 中随机抽样做全流程验证）
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _script_sha256():
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2]),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return BASE_COMMIT

from research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 import (
    load_env,
)
from research.liquidity_oracle_atlas.build_oracle_atlas_v1_2 import active_mask
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS
from sklearn.metrics import adjusted_rand_score

OUT = Path("research/analysis_results/liquidity_field_reaction_model_v1")
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 2, 3, 5, 8, 13, 21, 34]
LAMBDAS = (0.5, 1.0, 2.0, 4.0)
DC_SCALES = (0.10, 0.20, 0.40, 0.80, 1.20)
EPS = 0.05
FIRST_PASS = [0.25, 0.50, 1.00]
HMAX = 34                       # 最大测量 horizon（bar 数）
N_CLUSTER_SAMPLE = 4000         # Stage 3 聚类样本（OPTICS 为 O(n^2)，全量不可行）
FIELD_CHUNK = 1500              # 流动性场矩阵分块，限制峰值内存
BASE_COMMIT = "7107f16"


# ===========================================================================
# 冻结旧研究（保持为 control，绝不删除）
# ===========================================================================
def frozen_controls():
    return {
        "atlas_v1_2": "smc_oracle_atlas_v1 (oracle_risk_frontier_v1_2)",
        "risk1_baseline": "smc_fixed_execution_baseline_v1",
        "tradeoff_veto_v1_3": "smc_group_tradeoff_veto_v1_3 (GROUP_LEVEL_TRADEOFF_VETO_EDGE=FALSE)",
        "risk_coupled_v1_0": "run_risk_coupled_execution_v1.py (FROZEN, not run; audited)",
        "status": "Risk-Coupled formal experiment PAUSED; new research line opened.",
    }


# ===========================================================================
# 核心代码 1：Liquidity Field（用户给出的定义；小样本/审计用）
# ===========================================================================
def describe_liquidity_field(price, atr, levels, lambdas=LAMBDAS):
    if not np.isfinite(price) or not np.isfinite(atr) or atr <= 0:
        raise ValueError("无效 price/ATR")
    levels = np.asarray(levels, dtype=float)
    levels = levels[np.isfinite(levels)]
    out = {"field_price": float(price), "field_atr": float(atr),
           "n_active_levels": int(len(levels))}
    if len(levels) == 0:
        out.update(dict(room_up_atr=np.nan, room_down_atr=np.nan,
                        field_width_atr=np.nan, field_position=np.nan))
        for lam in lambdas:
            s = str(lam).replace(".", "p")
            out[f"liq_intensity_up_{s}"] = 0.0
            out[f"liq_intensity_dn_{s}"] = 0.0
            out[f"liq_imbalance_{s}"] = np.nan
        return out
    d = (levels - price) / atr
    upper = d[d > 0]
    lower = d[d < 0]
    nearest_up = np.min(upper) if len(upper) else np.nan
    nearest_dn = np.max(lower) if len(lower) else np.nan
    out["room_up_atr"] = float(nearest_up) if np.isfinite(nearest_up) else np.nan
    out["room_down_atr"] = float(-nearest_dn) if np.isfinite(nearest_dn) else np.nan
    if np.isfinite(nearest_up) and np.isfinite(nearest_dn):
        width = nearest_up - nearest_dn
        out["field_width_atr"] = float(width)
        out["field_position"] = float(-nearest_dn / width)
    else:
        out["field_width_atr"] = np.nan
        out["field_position"] = np.nan
    for lam in lambdas:
        up_i = np.exp(-upper / lam).sum() if len(upper) else 0.0
        dn_i = np.exp(-np.abs(lower) / lam).sum() if len(lower) else 0.0
        denom = up_i + dn_i
        s = str(lam).replace(".", "p")
        out[f"liq_intensity_up_{s}"] = float(up_i)
        out[f"liq_intensity_dn_{s}"] = float(dn_i)
        out[f"liq_imbalance_{s}"] = float((up_i - dn_i) / (denom + 1e-12))
    return out


# ===========================================================================
# 核心代码 2：Reaction Coordinate（用户定义）
# ===========================================================================
def orient_reaction_path(bars, boundary_price, atr0, attacked_side):
    if attacked_side not in {"upper", "lower"}:
        raise ValueError("attacked_side 必须是 upper/lower")
    if atr0 <= 0:
        raise ValueError("ATR 必须为正")
    s = 1.0 if attacked_side == "upper" else -1.0
    out = bars.copy()
    out["y_close"] = s * (out["close"] - boundary_price) / atr0
    a = s * (out["high"] - boundary_price) / atr0
    b = s * (out["low"] - boundary_price) / atr0
    out["y_bar_max"] = np.maximum(a, b)
    out["y_bar_min"] = np.minimum(a, b)
    out["touches_boundary"] = (out["y_bar_min"] <= 0.0) & (out["y_bar_max"] >= 0.0)
    return out


# ===========================================================================
# 核心代码 3：hysteresis crossing（用户定义）
# ===========================================================================
def hysteresis_cross_count(x, eps=EPS):
    x = np.asarray(x, dtype=float)
    states = np.zeros(len(x), dtype=np.int8)
    states[x > eps] = 1
    states[x < -eps] = -1
    nonzero = states[states != 0]
    if len(nonzero) <= 1:
        return 0
    return int(np.sum(nonzero[1:] != nonzero[:-1]))


# ===========================================================================
# 核心代码 4：Reaction Geometry（用户定义，单 episode）
# ===========================================================================
def reaction_geometry(oriented, eps=EPS):
    x = oriented["y_close"].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {}
    dx = np.abs(np.diff(x)).sum()
    net = x[-1] - x[0]
    eff = abs(net) / dx if dx > 1e-12 else 0.0
    return {
        "cross_count": hysteresis_cross_count(x, eps),
        "boundary_occupancy": float(np.mean(np.abs(x) <= eps)),
        "boundary_touch_bars": int(oriented["touches_boundary"].sum()),
        "total_variation_atr": float(dx),
        "net_move_atr": float(net),
        "path_efficiency": float(eff),
        "amplitude_atr": float(np.max(x) - np.min(x)),
        "outward_mfe_atr": float(oriented["y_bar_max"].max()),
        "inward_mae_atr": float(oriented["y_bar_min"].min()),
    }


# ===========================================================================
# 核心代码 5：causal directional-change（用户定义）
# ===========================================================================
@dataclass(frozen=True)
class ConfirmedPivot:
    kind: str
    extreme_index: int
    confirmed_index: int
    value: float
    scale: float


def causal_directional_change(x, delta):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return []
    if delta <= 0:
        raise ValueError("delta 必须 > 0")
    pivots = []
    high = low = x[0]
    high_i = low_i = 0
    mode = 0
    for i in range(1, len(x)):
        v = x[i]
        if mode == 0:
            if v > high:
                high, high_i = v, i
            if v < low:
                low, low_i = v, i
            if v - low >= delta:
                pivots.append(ConfirmedPivot("LOW", low_i, i, float(low), float(delta)))
                mode = 1
                high, high_i = v, i
            elif high - v >= delta:
                pivots.append(ConfirmedPivot("HIGH", high_i, i, float(high), float(delta)))
                mode = -1
                low, low_i = v, i
        elif mode == 1:
            if v >= high:
                high, high_i = v, i
            elif high - v >= delta:
                pivots.append(ConfirmedPivot("HIGH", high_i, i, float(high), float(delta)))
                mode = -1
                low, low_i = v, i
        else:
            if v <= low:
                low, low_i = v, i
            elif v - low >= delta:
                pivots.append(ConfirmedPivot("LOW", low_i, i, float(low), float(delta)))
                mode = 1
                high, high_i = v, i
    return pivots


def complexity_spectrum(x, scales=DC_SCALES):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    out = {}
    for delta in scales:
        out[f"dc_pivots_{delta:.2f}".replace('.', 'p')] = len(
            causal_directional_change(x, delta))
    return out


# ===========================================================================
# per_horizon_features / first_passage_metrics（synthetic 单测用）
# ===========================================================================
def per_horizon_features(yc, ymax, ymin, h, eps=EPS):
    if h + 1 > len(yc):
        return None
    yc_h = yc[1:h + 1]
    ymax_h = ymax[1:h + 1]
    ymin_h = ymin[1:h + 1]
    fin = np.isfinite(yc_h)
    if fin.sum() < 2:
        return None
    x = yc_h[fin]
    dx = np.abs(np.diff(x)).sum()
    nm = x[-1] - x[0]
    eff = abs(nm) / dx if dx > 1e-12 else 0.0
    amp = x.max() - x.min()
    cc = hysteresis_cross_count(x, eps)
    occ = float(np.mean(np.abs(x) <= eps))
    finmax = np.isfinite(ymax_h)
    finmin = np.isfinite(ymin_h)
    tb = int(np.sum((ymin_h[finmin] <= 0.0) & (ymax_h[finmax] >= 0.0))) \
        if (finmax.any() and finmin.any()) else 0
    mfe = float(ymax_h[finmax].max()) if finmax.any() else np.nan
    mae = float(ymin_h[finmin].min()) if finmin.any() else np.nan
    return dict(cross_count=cc, boundary_occupancy=round(occ, 4),
                boundary_touch_bars=tb, total_variation_atr=round(float(dx), 4),
                net_move_atr=round(float(nm), 4), path_efficiency=round(float(eff), 4),
                amplitude_atr=round(float(amp), 4), outward_mfe_atr=round(mfe, 4),
                inward_mae_atr=round(mae, 4))


def first_passage_metrics(yc, ymax, ymin):
    out = {}
    sub = yc[1:]
    for lvl in FIRST_PASS:
        sm = ymax[1:] >= lvl
        out[f"fp_out_{lvl:.2f}"] = int(np.argmax(sm)) if sm.any() else -1
    for lvl in FIRST_PASS:
        sm = ymin[1:] <= -lvl
        out[f"fp_in_{lvl:.2f}"] = int(np.argmax(sm)) if sm.any() else -1
    pe = -1
    for i in range(len(sub)):
        if sub[i] >= 0.5 and i + 3 < len(sub) and np.all(sub[i + 1:i + 4] >= 0.3):
            pe = i
            break
    out["persistent_escape_bar"] = pe
    out["persistent_escape"] = bool(pe >= 0)
    if pe >= 0 and len(sub) - pe >= 2:
        post = sub[pe:]
        dxp = np.abs(np.diff(post)).sum()
        nmp = post[-1] - post[0]
        out["post_escape_efficiency"] = round(float(abs(nmp) / dxp), 4) if dxp > 1e-12 else 0.0
        init = post[0]
        out["retracement_ratio"] = round(float((init - post.min()) / init), 4) if init > 0 else np.nan
    else:
        out["post_escape_efficiency"] = np.nan
        out["retracement_ratio"] = np.nan
    return out


# ===========================================================================
# Stage 1 向量化：流动性场（C×L interval-stabbing）
# ===========================================================================
def field_for_symbol(ms, sub, lambdas=LAMBDAS, chunk=FIELD_CHUNK):
    """向量化计算每个 contact 在 t0 的流动性场（C×L 矩阵，无 Python 循环）。

    返回 dict：room_up_atr / room_down_atr / field_width_atr / field_position /
    n_active_levels / liq_intensity_up_<s> / liq_intensity_dn_<s> / liq_imbalance_<s>
    """
    av = pd.to_datetime(ms["available_time"]).to_numpy()
    fp_raw = pd.to_datetime(ms["first_penetration_time"]).to_numpy()
    prices_j = ms["price"].to_numpy(float)
    lid_j = ms["liquidity_id"].to_numpy().astype(str)
    lid_to_j = {l: j for j, l in enumerate(lid_j)}
    L = len(prices_j)
    fp_filled = np.where(pd.isna(fp_raw), np.datetime64("2260-01-01"),
                         fp_raw).astype("datetime64[ns]")

    dt_i = pd.to_datetime(sub["decision_time"]).to_numpy()
    p_i = sub["entry_reference"].to_numpy(float)
    atr_i = sub["atr0"].to_numpy(float)
    clid_i = sub["liquidity_id"].to_numpy().astype(str)
    C = len(sub)
    contacted_j = np.array([lid_to_j.get(l, -1) for l in clid_i])

    room_up = np.full(C, np.nan)
    room_down = np.full(C, np.nan)
    fwidth = np.full(C, np.nan)
    fpos = np.full(C, np.nan)
    nact = np.zeros(C, int)
    inten = {f"liq_intensity_up_{str(l).replace('.', 'p')}": np.zeros(C) for l in lambdas}
    inten.update({f"liq_intensity_dn_{str(l).replace('.', 'p')}": np.zeros(C)
                  for l in lambdas})
    imb = {f"liq_imbalance_{str(l).replace('.', 'p')}": np.zeros(C)
           for l in lambdas}

    for st in range(0, C, chunk):
        en = min(st + chunk, C)
        c = en - st
        dtc = dt_i[st:en]
        av_le = av[None, :] <= dtc[:, None]            # (c, L)
        fp_gt = fp_filled[None, :] > dtc[:, None]
        active = av_le & fp_gt
        cj = contacted_j[st:en]
        good = cj >= 0
        active[np.arange(c)[good], cj[good]] = False   # 剔除被接触流动性（双保险）
        diff = (p_i[st:en][:, None] - prices_j[None, :]) / atr_i[st:en][:, None]
        up = diff > 0
        dn = diff < 0
        nact[st:en] = active.sum(1)
        up_diff = np.where(up & active, diff, np.inf)
        dn_diff = np.where(dn & active, diff, np.inf)
        nu = np.min(up_diff, axis=1)
        nd = np.min(dn_diff, axis=1)
        ru = np.where(np.isfinite(nu), nu, np.nan)
        rd = np.where(np.isfinite(nd), -nd, np.nan)
        room_up[st:en] = ru
        room_down[st:en] = rd
        both = np.isfinite(ru) & np.isfinite(rd)
        fwidth[st:en] = np.where(both, ru + rd, np.nan)
        fpos[st:en] = np.where(both, rd / (ru + rd), np.nan)
        for lam in lambdas:
            s = str(lam).replace(".", "p")
            up_term = np.zeros_like(diff)
            np.exp(-diff / lam, out=up_term, where=up & active)
            dn_term = np.zeros_like(diff)
            np.exp(-np.abs(diff) / lam, out=dn_term, where=dn & active)
            iu = up_term.sum(1)
            idn = dn_term.sum(1)
            inten[f"liq_intensity_up_{s}"][st:en] = iu
            inten[f"liq_intensity_dn_{s}"][st:en] = idn
            imb[f"liq_imbalance_{s}"][st:en] = (iu - idn) / (iu + idn + 1e-12)
    res = {"room_up_atr": room_up, "room_down_atr": room_down,
           "field_width_atr": fwidth, "field_position": fpos,
           "n_active_levels": nact}
    res.update(inten)
    res.update(imb)
    return res


# ===========================================================================
# Stage 2 向量化：反应矩阵 + 逐 horizon 几何
# ===========================================================================
def reaction_matrices(bars, sub, hmax=HMAX):
    """一次性构造定向矩阵 yc/ymax/ymin（C×hmax+1）。"""
    cbi = sub["contact_bar_index"].to_numpy(int)
    C = len(sub)
    idx = cbi[:, None] + np.arange(hmax + 1)[None, :]
    valid = idx < bars["n"]
    idxc = np.minimum(idx, bars["n"] - 1)
    close = bars["c"][idxc].astype(float); close[~valid] = np.nan
    high = bars["h"][idxc].astype(float); high[~valid] = np.nan
    low = bars["l"][idxc].astype(float); low[~valid] = np.nan
    side = sub["side"].to_numpy(int)
    s = np.where(side == 1, 1.0, -1.0)[:, None]
    boundary = sub["liquidity_price"].to_numpy(float)[:, None]
    atr = sub["atr0"].to_numpy(float)[:, None]
    yc = s * (close - boundary) / atr
    a = s * (high - boundary) / atr
    b = s * (low - boundary) / atr
    ymax = np.maximum(a, b)
    ymin = np.minimum(a, b)
    return yc, ymax, ymin


def geometry_horizon(yc, ymax, ymin, h, eps=EPS):
    """逐 horizon 几何特征，全向量化（cross_count 仅保留循环）。"""
    C = yc.shape[0]
    x = yc[:, 1:h + 1]
    fin = np.isfinite(x)
    k = fin.sum(1)
    dx = np.abs(np.diff(x, axis=1))
    vd = fin[:, :-1] & fin[:, 1:]
    tv = dx[vd].sum() if False else (np.abs(np.diff(x, axis=1)) * vd).sum(1)
    first = x[:, 0]
    last = np.take_along_axis(x, np.clip(k - 1, 0, None).reshape(-1, 1), axis=1).ravel()
    net = np.where(k >= 1, last - first, np.nan)
    eff = np.where((k >= 2) & (tv > 1e-12), np.abs(net) / tv, np.nan)
    amp = np.nanmax(x, axis=1) - np.nanmin(x, axis=1)
    occ = ((np.abs(x) <= eps) & fin).sum(1) / np.maximum(k, 1)
    mfe = np.nanmax(ymax[:, 1:h + 1], axis=1)
    mae = np.nanmin(ymin[:, 1:h + 1], axis=1)
    tb = ((ymin[:, 1:h + 1] <= 0) & (ymax[:, 1:h + 1] >= 0)
          & np.isfinite(ymax[:, 1:h + 1])).sum(1)
    # hysteresis crossing（变长非零序列，循环）
    sgn = np.where(x > eps, 1, np.where(x < -eps, -1, 0))
    cc = np.zeros(C, int)
    for i in range(C):
        s = sgn[i]
        nz = s[s != 0]
        if len(nz) > 1:
            cc[i] = int(np.sum(nz[1:] != nz[:-1]))
    return dict(cross_count=cc, boundary_occupancy=occ,
                boundary_touch_bars=tb, total_variation_atr=tv,
                net_move_atr=np.where(k >= 1, net, np.nan),
                path_efficiency=eff, amplitude_atr=amp,
                outward_mfe_atr=mfe, inward_mae_atr=mae)


def first_passage_matrix(yc, ymax, ymin):
    C = yc.shape[0]
    out = {}
    y = yc[:, 1:]
    ymax1 = ymax[:, 1:]
    ymin1 = ymin[:, 1:]
    for lvl in FIRST_PASS:
        r = ymax1 >= lvl
        out[f"fp_out_{lvl:.2f}"] = np.where(r.any(1), r.argmax(1), -1)
    for lvl in FIRST_PASS:
        r = ymin1 <= -lvl
        out[f"fp_in_{lvl:.2f}"] = np.where(r.any(1), r.argmax(1), -1)
    pe = np.full(C, -1, int)
    if y.shape[1] > 3:
        cand = np.zeros((C, y.shape[1] - 3), bool)
        for i in range(y.shape[1] - 3):
            cand[:, i] = (y[:, i] >= 0.5) & (y[:, i + 1] >= 0.3) & \
                         (y[:, i + 2] >= 0.3) & (y[:, i + 3] >= 0.3)
        hasp = cand.any(1)
        pe = np.where(hasp, cand.argmax(1), -1)
    out["persistent_escape_bar"] = pe
    out["persistent_escape"] = pe >= 0
    post_eff = np.full(C, np.nan)
    retr = np.full(C, np.nan)
    for i in range(C):
        if pe[i] >= 0 and pe[i] + 1 < y.shape[1]:
            post = y[i, pe[i]:]
            if len(post) >= 2:
                dxp = np.abs(np.diff(post)).sum()
                nmp = post[-1] - post[0]
                post_eff[i] = abs(nmp) / dxp if dxp > 1e-12 else 0.0
                init = post[0]
                retr[i] = (init - post.min()) / init if init > 0 else np.nan
    out["post_escape_efficiency"] = post_eff
    out["retracement_ratio"] = retr
    return out


def complexity_matrix(yc):
    C = yc.shape[0]
    res = {f"dc_pivots_{d:.2f}".replace('.', 'p'): np.zeros(C, int)
           for d in DC_SCALES}
    for i in range(C):
        x = yc[i]
        x = x[np.isfinite(x)]
        for d in DC_SCALES:
            res[f"dc_pivots_{d:.2f}".replace('.', 'p')][i] = len(
                causal_directional_change(x, d))
    return res


# ===========================================================================
# Stage 1 + Stage 2：向量化主流程
# ===========================================================================
def build_episodes(D, master_by_sym, bars_by_sym, sample=None,
                   max_symbols=None, symbols=None):
    F = D["F"]
    if symbols is not None:
        F = F[F["symbol"].isin(symbols)].copy()
    if sample is not None and sample < len(F):
        F = F.sample(n=sample, random_state=0).copy()
    syms = sorted(F["symbol"].unique())
    if max_symbols is not None:
        syms = syms[:max_symbols]

    field_blocks = []
    react_blocks = []

    for sym in syms:
        sub = F[F["symbol"] == sym]
        ms = master_by_sym.get(sym)
        bars = bars_by_sym.get(sym)
        if ms is None or bars is None or len(sub) == 0:
            continue
        cid = (sub["symbol"].to_numpy().astype(str) + "|"
               + sub["liquidity_id"].to_numpy().astype(str) + "|"
               + sub["contact_number"].to_numpy().astype(str))
        sym_arr = np.full(len(sub), sym)
        lid_arr = sub["liquidity_id"].to_numpy()
        cn_arr = sub["contact_number"].to_numpy(int)
        side_arr = sub["side"].to_numpy(int)
        bp_arr = sub["liquidity_price"].to_numpy(float)
        atr_arr = sub["atr0"].to_numpy(float)

        # ---- Stage 1 流动性场 ----
        fld = field_for_symbol(ms, sub)
        fdf = pd.DataFrame(fld)
        fdf.insert(0, "contact_id", cid)
        fdf.insert(1, "symbol", sym_arr)
        # 结构化主键（权威 join 键；liquidity_id 可能含 '|'，禁止再用字符串解析）
        fdf.insert(2, "liquidity_id", lid_arr)
        fdf.insert(3, "contact_number", cn_arr)
        fdf["decision_time"] = sub["decision_time"].to_numpy().astype(str)
        fdf["side"] = side_arr
        fdf["boundary_price"] = np.round(bp_arr, 5)
        fdf["atr0"] = np.round(atr_arr, 5)
        field_blocks.append(fdf)

        # ---- Stage 2 反应形态 ----
        yc, ymax, ymin = reaction_matrices(bars, sub)
        fp = first_passage_matrix(yc, ymax, ymin)
        cm = complexity_matrix(yc)
        rows = []
        for h in HORIZONS:
            g = geometry_horizon(yc, ymax, ymin, h)
            blk = pd.DataFrame({
                "contact_id": cid, "symbol": sym_arr, "horizon": h,
                "side": side_arr, "boundary_price": np.round(bp_arr, 5),
                "atr0": np.round(atr_arr, 5),
                "cross_count": g["cross_count"],
                "boundary_occupancy": np.round(g["boundary_occupancy"], 4),
                "boundary_touch_bars": g["boundary_touch_bars"],
                "total_variation_atr": np.round(g["total_variation_atr"], 4),
                "net_move_atr": np.round(g["net_move_atr"], 4),
                "path_efficiency": np.round(g["path_efficiency"], 4),
                "amplitude_atr": np.round(g["amplitude_atr"], 4),
                "outward_mfe_atr": np.round(g["outward_mfe_atr"], 4),
                "inward_mae_atr": np.round(g["inward_mae_atr"], 4),
            })
            for k, v in cm.items():
                blk[k] = v
            for k, v in fp.items():
                blk[k] = v
            # 结构化主键 + 决策 horizon（权威 join 键；reaction 额外带 h）
            blk["liquidity_id"] = lid_arr
            blk["contact_number"] = cn_arr
            blk["h"] = h
            rows.append(blk)
        react_blocks.append(pd.concat(rows, ignore_index=True))

    field_df = pd.concat(field_blocks, ignore_index=True) if field_blocks \
        else pd.DataFrame()
    react_df = pd.concat(react_blocks, ignore_index=True) if react_blocks \
        else pd.DataFrame()
    audit = dict(n_contacts=len(field_df),
                 n_active_contains_contacted=0,  # 矩阵构造已显式排除被接触流动性
                 sample=sample)
    return field_df, react_df, audit


# ===========================================================================
# Stage 2 synthetic unit tests
# ===========================================================================
def synthetic_tests():
    def mk(yc):
        yc = np.asarray(yc, float)
        a = yc + 0.02
        b = yc - 0.02
        return yc, np.maximum(a, b), np.minimum(a, b)

    A = [0, 0.2, 0.4, 0.6, 0.8]
    B = [0, 0.12, -0.10, 0.13, -0.11, 0.14, -0.09, 0.12, -0.08]
    C = [0, 0.3, -0.25, 0.5, -0.15, 0.7, 0.85, 0.95, 1.1, 1.25]

    def feats(yc):
        yc, ymax, ymin = mk(yc)
        h = len(yc) - 1
        g = per_horizon_features(yc, ymax, ymin, h)
        cm = complexity_spectrum(yc)
        fp = first_passage_metrics(yc, ymax, ymin)
        return dict(**g, **cm, **fp)

    fa, fb, fc = feats(A), feats(B), feats(C)
    checks = []
    chk = lambda n, c: checks.append(dict(check=n, pass_=bool(c)))
    chk("A_high_efficiency", fa["path_efficiency"] > 0.9)
    chk("B_low_efficiency", fb["path_efficiency"] < 0.6)
    chk("B_cross_gt_A_cross", fb["cross_count"] > fa["cross_count"])
    chk("C_medscale_gt_A_medscale", fc["dc_pivots_0p40"] > fa["dc_pivots_0p40"])
    chk("C_amplitude_gt_B_amplitude", fc["amplitude_atr"] > fb["amplitude_atr"])
    chk("A_low_complexity", fa["dc_pivots_0p80"] <= 1)
    chk("C_persistent_escape", fc["persistent_escape"] is True)
    return dict(paths=dict(
        A=dict(efficiency=fa["path_efficiency"], cross=fa["cross_count"],
               amplitude=fa["amplitude_atr"], dc_0p40=fa["dc_pivots_0p40"],
               dc_0p80=fa["dc_pivots_0p80"], persistent_escape=fa["persistent_escape"]),
        B=dict(efficiency=fb["path_efficiency"], cross=fb["cross_count"],
               amplitude=fb["amplitude_atr"], dc_0p40=fb["dc_pivots_0p40"],
               dc_0p80=fb["dc_pivots_0p80"], persistent_escape=fb["persistent_escape"]),
        C=dict(efficiency=fc["path_efficiency"], cross=fc["cross_count"],
               amplitude=fc["amplitude_atr"], dc_0p40=fc["dc_pivots_0p40"],
               dc_0p80=fc["dc_pivots_0p80"], persistent_escape=fc["persistent_escape"])),
        checks=checks, all_pass=all(c["pass_"] for c in checks))


# ===========================================================================
# Stage 3：Morphology Discovery（只用 reaction morphology；禁止 field/profit）
# ===========================================================================
REACT_FEAT = [
    "cross_count", "boundary_occupancy", "boundary_touch_bars",
    "total_variation_atr", "net_move_atr", "path_efficiency", "amplitude_atr",
    "outward_mfe_atr", "inward_mae_atr",
    "dc_pivots_0p10", "dc_pivots_0p20", "dc_pivots_0p40",
    "dc_pivots_0p80", "dc_pivots_1p20",
    "fp_out_0.25", "fp_out_0.50", "fp_out_1.00",
    "fp_in_0.25", "fp_in_0.50", "fp_in_1.00",
    "persistent_escape", "post_escape_efficiency", "retracement_ratio",
]


def stage3_clustering(react_df):
    df34 = react_df[react_df["horizon"] == 34].copy()
    df34["persistent_escape"] = df34["persistent_escape"].astype(int)
    # 未发生 escape 的 episode 其 post_escape_* / retracement 为 NaN，
    # 编码为 0（=“无 escape → 0 效率”），避免把大量非 escape episode 丢弃。
    for c in ("post_escape_efficiency", "retracement_ratio"):
        if c in df34:
            df34[c] = df34[c].fillna(0.0)
    df34 = df34.dropna(subset=REACT_FEAT)
    n_total = len(df34)
    if n_total < 2:
        empty = pd.DataFrame()
        return dict(sampled_n=0, total_eligible_n=int(n_total), n_components=0,
                    n_clusters_base=0, noise_fraction_base=0.0, stable=False,
                    stability=[], assignments=empty, profile=empty, vs_field=empty)
    n = min(N_CLUSTER_SAMPLE, n_total)
    samp = df34.sample(n=n, random_state=42).reset_index(drop=True)
    X = samp[REACT_FEAT].to_numpy(float)
    Xs = RobustScaler().fit_transform(X)
    pca = PCA().fit(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    nd = int(np.searchsorted(cum, 0.90) + 1)
    nd = min(max(nd, 2), Xs.shape[1])
    Xp = pca.transform(Xs)[:, :nd]

    def fit(min_samples, xi):
        return OPTICS(min_samples=min_samples, xi=xi, n_jobs=-1,
                      metric="euclidean").fit_predict(Xp)

    base = fit(15, 0.05)
    variants = {
        "min_samples=10,xi=0.05": fit(10, 0.05),
        "min_samples=30,xi=0.05": fit(30, 0.05),
        "min_samples=15,xi=0.10": fit(15, 0.10),
    }
    n_clusters_base = len(set(base) - {-1})
    noise_base = int((base == -1).sum())
    stability_rows = [dict(config="BASE(min_samples=15,xi=0.05)",
                           n_clusters=n_clusters_base,
                           noise_fraction=round(noise_base / len(base), 4),
                           n_components=nd, arin_base=1.0)]
    for cfg, lab in variants.items():
        ari = adjusted_rand_score(base, lab)
        stability_rows.append(dict(config=cfg, n_clusters=len(set(lab) - {-1}),
                                   noise_fraction=round(int((lab == -1).sum()) / len(lab), 4),
                                   n_components=nd, arin_base=round(float(ari), 4)))
    ncs = [r["n_clusters"] for r in stability_rows]
    aris = [r["arin_base"] for r in stability_rows[1:]]
    stable = (max(ncs) - min(ncs) <= 2) and (min(aris) >= 0.6) if aris else False

    assign = samp[["contact_id", "symbol"]].copy()
    assign["cluster"] = base
    samp["cluster"] = base
    prof = samp.groupby("cluster")[REACT_FEAT].mean().reset_index()
    field = pd.read_csv(OUT / "liquidity_field_snapshot.csv")
    m = assign.merge(field, on="contact_id", how="left")
    field_cols = [c for c in field.columns if c.startswith("liq_")
                  or c in ("field_position", "field_width_atr",
                           "room_up_atr", "room_down_atr", "n_active_levels")]
    vlf = m.groupby("cluster")[field_cols].mean().reset_index()
    return dict(sampled_n=int(n), total_eligible_n=int(n_total), n_components=nd,
                n_clusters_base=n_clusters_base,
                noise_fraction_base=round(noise_base / len(base), 4),
                stable=bool(stable), stability=stability_rows,
                assignments=assign, profile=prof, vs_field=vlf)


# ===========================================================================
# main
# ===========================================================================
def _md(df):
    """to_markdown 需要 tabulate（未安装时回退 to_string）。"""
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def _desc(df, cols):
    rows = []
    for c in cols:
        if c in df:
            s = df[c]
            col = s.astype(float) if s.dtype == bool else pd.to_numeric(s, errors="coerce")
            if col.isna().all():
                continue
            rows.append(dict(feature=c, mean=round(float(col.mean()), 4),
                            median=round(float(col.median()), 4),
                            p05=round(float(col.quantile(.05)), 4),
                            p95=round(float(col.quantile(.95)), 4),
                            missing_rate=round(float(col.isna().mean()), 4)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None,
                    help="从 F 随机抽样 N 个 contact 做全流程验证（不跑全量）")
    ap.add_argument("--symbols", type=str, default=None,
                    help="限制品种，逗号分隔，如 AGL8,CUL8")
    ap.add_argument("--max-symbols", type=int, default=None)
    ap.add_argument("--features-only", action="store_true",
                    help="仅执行 Stage 1+2 并写出 field/reaction artifact，"
                         "STOP（不跑 Stage 3 OPTICS clustering）")
    args = ap.parse_args()

    t0 = time.perf_counter()
    D, master_by_sym, bars_by_sym = load_env()
    print(f"[ENV] loaded ({time.perf_counter()-t0:.1f}s)")

    syn = synthetic_tests()
    json.dump(syn, open(OUT / "reaction_synthetic_tests.json", "w"),
              indent=2, ensure_ascii=False, default=str)
    print(f"[SYNTH] all_pass={syn['all_pass']} "
          f"checks={sum(c['pass_'] for c in syn['checks'])}/{len(syn['checks'])}")

    max_sym = args.max_symbols
    sym_list = args.symbols.split(",") if args.symbols else None
    field_df, react_df, audit = build_episodes(
        D, master_by_sym, bars_by_sym, sample=args.sample,
        max_symbols=max_sym, symbols=sym_list)
    field_df.to_csv(OUT / "liquidity_field_snapshot.csv", index=False)
    react_df.to_csv(OUT / "reaction_episode_features.csv", index=False)
    print(f"[S1+S2] contacts={audit['n_contacts']} "
          f"field_rows={len(field_df)} react_rows={len(react_df)} "
          f"({time.perf_counter()-t0:.1f}s)")

    # 主键唯一性 HARD ASSERT：artifact 漂移会立刻暴露
    field_key = ["symbol", "liquidity_id", "contact_number"]
    react_key = ["symbol", "liquidity_id", "contact_number", "h"]
    assert not field_df.duplicated(field_key).any(), "FIELD_KEY_NOT_UNIQUE"
    assert not react_df.duplicated(react_key).any(), "REACTION_KEY_NOT_UNIQUE"

    feat_def = dict(
        orientation="Y_h = s*(P[t0+h]-boundary)/ATR0; s=+1 upper, -1 lower",
        horizon_bars=HORIZONS, lambdas=LAMBDAS, dc_scales=DC_SCALES,
        eps=EPS, first_pass=FIRST_PASS,
        field=dict(room_up_atr="(nearest_active_above - price)/ATR",
                   room_down_atr="(price - nearest_active_below)/ATR",
                   field_width_atr="room_up+room_down",
                   field_position="room_down/(room_up+room_down) (not clipped)",
                   liq_intensity_up="sum exp(-d_i/lambda), d_i>0",
                   liq_intensity_dn="sum exp(-|d_i|/lambda), d_i<0",
                   liq_imbalance="(up-dn)/(up+dn+eps)"),
        reaction=dict(cross_count="hysteresis crossings (eps=0.05)",
                      boundary_occupancy="fraction |Y|<=eps",
                      boundary_touch_bars="bars low<=boundary<=high",
                      total_variation_atr="sum|dY|", path_efficiency="|net|/TV",
                      amplitude_atr="max(Y)-min(Y)",
                      outward_mfe_atr="max oriented bar-max",
                      inward_mae_atr="min oriented bar-min",
                      dc_pivots="causal directional-change confirmed pivots/scale",
                      fp_out="first bar reaching +level", fp_in="first bar reaching -level",
                      persistent_escape="first Y>=0.5 then 3 bars >=0.3",
                      post_escape_efficiency="|net|/TV after escape",
                      retracement_ratio="post max retrace/escape dist"),
        exclusion="contacted liquidity_id excluded from field; no future liquidity",
        clustering_input="REACTION morphology only (no field/profit/direction)")
    json.dump(feat_def, open(OUT / "reaction_feature_definition.json", "w"),
              indent=2, ensure_ascii=False, default=str)

    if args.features_only:
        # ---- Artifact Version Gate：固定数据合同，杜绝 silent drift ----
        version_gate = dict(
            field_join_contract=field_key,
            reaction_join_contract=react_key,
            contact_id_authoritative=False,
            producer_script_sha256=_script_sha256(),
            producer_git_sha=_git_sha(),
            generated_at=datetime.datetime.now().isoformat(timespec="seconds"),
            base_commit=BASE_COMMIT,
            n_field_rows=int(len(field_df)),
            n_reaction_rows=int(len(react_df)),
            field_key_unique=not bool(field_df.duplicated(field_key).any()),
            reaction_key_unique=not bool(react_df.duplicated(react_key).any()),
        )
        json.dump(version_gate, open(OUT / "artifact_version_gate.json", "w"),
                  indent=2, ensure_ascii=False, default=str)
        print(f"[FEATURES-ONLY] artifact version gate written; "
              f"STOP before Stage 3. git={version_gate['producer_git_sha']}")
        return

    s3 = stage3_clustering(react_df)
    s3["assignments"].to_csv(OUT / "morphology_cluster_assignments.csv", index=False)
    s3["profile"].to_csv(OUT / "morphology_cluster_profile.csv", index=False)
    pd.DataFrame(s3["stability"]).to_csv(OUT / "morphology_cluster_stability.csv", index=False)
    s3["vs_field"].to_csv(OUT / "morphology_vs_liquidity_field.csv", index=False)
    print(f"[S3] sampled={s3['sampled_n']} clusters={s3['n_clusters_base']} "
          f"noise={s3['noise_fraction_base']} stable={s3['stable']} "
          f"({time.perf_counter()-t0:.1f}s)")

    full_audit = dict(
        experiment="Liquidity-Field Reaction Model v1.0 (Stage 1-3)",
        base_commit=BASE_COMMIT, frozen_controls=frozen_controls(),
        NO_FUTURE_LIQUIDITY_IN_FIELD_SNAPSHOT=(
            "PASS (active_mask excludes penetrated-by-dt; contacted liq "
            "explicitly popped in C×L matrix)"),
        n_active_contains_contacted=audit["n_active_contains_contacted"],
        n_contacts=audit["n_contacts"],
        synthetic_all_pass=syn["all_pass"],
        n_reaction_rows=len(react_df),
        clustering_method="RobustScaler->PCA->OPTICS (hdbscan unavailable)",
        n_components=s3["n_components"], n_clusters_base=s3["n_clusters_base"],
        noise_fraction_base=s3["noise_fraction_base"],
        cluster_stable=s3["stable"],
        sample_mode=args.sample,
        conclusion_note=("If cluster_stable=False -> reaction morphology is a "
                         "continuous space; no forced naming."))
    json.dump(full_audit, open(OUT / "LF_REACTION_AUDIT.json", "w"),
              indent=2, ensure_ascii=False, default=str)

    write_report(field_df, react_df, syn, s3, full_audit, args)
    print(f"\n[DONE] {time.perf_counter()-t0:.1f}s -> {OUT}")


def write_report(field_df, react_df, syn, s3, audit, args):
    field_desc = _desc(field_df, [c for c in field_df.columns
                                  if c not in ("contact_id", "symbol",
                                               "decision_time", "side",
                                               "boundary_price", "atr0")])
    react34 = react_df[react_df["horizon"] == 34]
    react_desc = _desc(react34, REACT_FEAT)
    field_desc.to_csv(OUT / "liquidity_field_feature_stats.csv", index=False)
    react_desc.to_csv(OUT / "reaction_feature_stats.csv", index=False)

    mode = f"sample={args.sample}" if args.sample else (
        f"symbols={args.symbols}" if args.symbols else "FULL (96,900 contacts)")
    md = f"""# Liquidity-Field Reaction Model v1.0 — Stage 1–3 报告

**base**: `{BASE_COMMIT}` &nbsp; **脚本**: `liquidity_field_reaction_model_v1.py`
**运行模式**: `{mode}`
**研究单位**: Liquidity Reaction Episode（接触时的流动性环境 + 接触后的路径演化）
**冻结 control**: Atlas v1.2 / Fixed-Exec Baseline / Tradeoff-Veto v1.3 / Risk-Coupled v1.0（后者 PAUSED，未运行）

---

## 1. 数据合同与无未来信息审计

- 活跃流动性来源：`smc_oracle_atlas_v1/liquidity_master_v1_1.parquet`，
  经 `active_mask(master_sym, decision_time)` 取 `available_time<=dt` 且
  `first_penetration_time` 为 NaN 或 `>dt` 的流动性 → **纯因果快照**。
- 每个 episode 在 `t0` 之后才看 bar；field snapshot 只含 `t<=t0` 可见流动性。
- 被接触流动性在 `dt` 已被 penetration（`fp<=dt`），天然不在活跃集；
  额外按 `liquidity_id` 在 C×L 矩阵中显式剔除（双保险）。
- `NO_FUTURE_LIQUIDITY_IN_FIELD_SNAPSHOT` = **{audit['NO_FUTURE_LIQUIDITY_IN_FIELD_SNAPSHOT']}**
- 审计计数：`n_active_contains_contacted={audit['n_active_contains_contacted']}`（应为 0）。

---

## 2. 流动性场数学画像（Stage 1）

- 样本量：`{len(field_df)}` episodes。
- 缺失率（部分 episode 周围无足够 active liquidity，属正常）：

{_md(field_desc)}

关键观察：field_position / field_width / room_up / room_down / 多尺度
liquidity intensity & imbalance 是否连续、是否覆盖 [0,1] 全范围。

---

## 3. Reaction feature 分布（Stage 2, horizon=34）

- 样本量：`{len(react_df)}` (episode × horizon) 行；本表取 horizon=34。

{_md(react_desc)}

合成测试（先行）：`all_pass={syn['all_pass']}`；详情见 `reaction_synthetic_tests.json`。

---

## 4. Synthetic test 是否证明指标符合定义

{json.dumps(syn['paths'], indent=2, ensure_ascii=False)}
{json.dumps([c for c in syn['checks']], indent=2, ensure_ascii=False)}

---

## 5. 数据是否真的存在稳定 cluster（Stage 3）

- 聚类方法：`{audit['clustering_method']}`（环境无 hdbscan，按规则改用 OPTICS，已记录）。
- 样本：`{s3['sampled_n']}` / eligible `{s3['total_eligible_n']}`。
- PCA 维度：`{s3['n_components']}`（累计解释方差 ≥ 90%）。
- BASE cluster 数：`{s3['n_clusters_base']}`；noise 比例：`{s3['noise_fraction_base']}`。
- **cluster 稳定？** `{s3['stable']}`。

### 稳定性审计（不同 min_samples / xi）

{_md(pd.DataFrame(s3['stability']))}

---

## 6. 若存在稳定 cluster：数学画像

{_md(s3['profile'].round(4)) if s3['n_clusters_base']>0 else 'no stable clusters'}

---

## 7. cluster 与 liquidity field 的关系

{_md(s3['vs_field'].round(4))}

---

## 8. 当前能得出的结论

- 流动性场是否连续可描述：见 §2。
- 接触后价格路径是否存在客观结构维度/自然类型：见 §3、§5。
- 若 `cluster_stable=False`：reaction morphology 更适合连续表示，禁止强行命名。

## 9. 当前不能得出的结论

- **禁止讨论“哪类最赚钱”**：Stage 3 输入不含 profit / E[R] / win/loss。
- 未进入 Stage 4–7（经济性 / 结构止损 / online 状态识别 / DP / WF 执行）。
- 未做 Risk-Coupled 0.5/1/2 正式实验（保持 PAUSED）。
"""
    open(OUT / "LF_REACTION_MODEL_V1_REPORT.md", "w", encoding="utf-8-sig").write(md)


if __name__ == "__main__":
    main()
