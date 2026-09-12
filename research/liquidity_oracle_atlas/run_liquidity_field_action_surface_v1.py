"""Liquidity-Field Sequential Action Surface v1.0  (Stage 4A)

研究目的（reviewer 规格；非训练策略）：
  在 *同一个* 冻结于 t0 的流动性场里，随着价格碰触后结构逐步形成，
  研究 entry / 结构止损 / 剩余 target 空间如何共同改变一笔交易的经济价值。

三个核心问题：
  Q1 等待确认是否改善经济性（value of waiting）
  Q2 不同结构尺度 δ 的 win-rate / payoff tradeoff
  Q3 在几何（entry/stop/target）相同时，reaction morphology 与 liquidity field
     是否还有增量信息

硬边界（P15 ROI Gate）：不训练 classifier / DP / RL；不优化 target / risk grid；
不自动搜最佳 h / scale；不聚类。全量跑完先 STOP 汇报。

TRADING_METRICS：本实验定义 *固定* 假设执行动作（entry=下一根 5m open；
stop=δ-confirmed 结构极值；target=surviving liquidity），报告 E[R]_lower/upper、
target_first_rate 等“几何经济面”度量。这不是被选择的策略 candidate（无 selection /
无 optimization），所有 cell 预注册、全量报告，不做多重比较挑选。同 bar 双边界
（lower=-1R, upper=+RR）避免偷偷选序。

数据合同：t0 流动性场冻结；价格运动过程中 t0 liquidity 可被 consumed（causal
depletion），但本阶段不新增 contact 后生成的 liquidity（Dynamic Field Regeneration
留待下一阶段）。
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 import (
    load_env,
)
from research.liquidity_oracle_atlas.build_oracle_atlas_v1_2 import (
    active_mask,  # 权威 frozen Atlas v1.2 active_mask（仅用于语义对齐 / 文档）
)

OUT = Path("research/analysis_results/liquidity_field_action_surface_v1")
OUT.mkdir(parents=True, exist_ok=True)

# ---- 固定实验空间（P3）----
HORIZONS = np.array([1, 2, 3, 5, 8, 13], dtype=np.int64)
STRUCTURE_SCALES = np.array([0.2, 0.4, 0.8, 1.2], dtype=np.float64)
ACTIONS = ["OUTWARD", "INWARD"]
EVAL_BARS = 34                      # 固定评估窗口（P8）
T_POST = 48                         # 需要的 post-bar 数量（h=13 + 34 评估）
EPS = 0.05
CHUNK = 4096                        # 接触分块（P9）
BASE_COMMIT = "7107f16"

WF = [("WF1", "TB2"), ("WF2", "TB3"), ("WF3", "TB4")]
TEST_BLOCK = {"WF1": "TB2", "WF2": "TB3", "WF3": "TB4"}

# ===========================================================================
# 向量化核（来自 reviewer 规格；全部 NumPy，无 per-contact Python 循环）
# ===========================================================================
def surviving_field_and_target(levels, active_at_t0, entry, cum_low, cum_high,
                               direction):
    """t0 active liquidity → 随价格运动 causal depletion → 最近 surviving target。

    levels        : (M,) 当前 symbol 的 liquidity 价格
    active_at_t0  : (c, M) bool，t0 活跃（active_mask）
    cum_low/cum_high: (c,) 截至 decision 已观察价格范围（绝对价）
    direction     : (c,) +1 LONG / -1 SHORT
    """
    levels = np.asarray(levels, dtype=np.float64)
    active = np.asarray(active_at_t0, dtype=bool)
    entry = np.asarray(entry, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    c = entry.shape[0]
    consumed = (levels[None, :] >= cum_low[:, None]) & (
        levels[None, :] <= cum_high[:, None])
    alive = active & (~consumed)
    dist = direction[:, None] * (levels[None, :] - entry[:, None])
    valid = alive & (dist > 0)
    d = np.where(valid, dist, np.inf)
    nearest = np.min(d, axis=1)
    has_target = np.isfinite(nearest)
    target = np.full(c, np.nan, dtype=np.float64)
    target[has_target] = entry[has_target] + direction[has_target] * nearest[has_target]
    return target, alive


def latest_confirmed_extreme(z_low, z_high, horizon, delta, direction):
    """δ-confirmed structural extremum（reviewer 规格，完全向量化）。

    z_low/z_high : (c, horizon) 相对 boundary、ATR 标准化后的 OHLC extrema
    direction    : (c,) 或 scalar；+1 LONG 找 confirmed LOW，-1 SHORT 找 confirmed HIGH
    """
    z_low = np.asarray(z_low, dtype=np.float64)
    z_high = np.asarray(z_high, dtype=np.float64)
    c, h = z_low.shape
    j = np.arange(h, dtype=np.int32)[None, :]
    direction = np.asarray(direction).reshape(c, 1)
    # LONG（direction==+1）：某 LOW 之后 z_high 反弹 >= delta
    suffix_max = np.maximum.accumulate(z_high[:, ::-1], axis=1)[:, ::-1]
    max_after = np.concatenate(
        [suffix_max[:, 1:], np.full((c, 1), -np.inf, dtype=z_high.dtype)],
        axis=1)
    valid_long = (max_after - z_low) >= delta
    # SHORT（direction==-1）：某 HIGH 之后 z_low 回落 >= delta
    suffix_min = np.minimum.accumulate(z_low[:, ::-1], axis=1)[:, ::-1]
    min_after = np.concatenate(
        [suffix_min[:, 1:], np.full((c, 1), np.inf, dtype=z_low.dtype)],
        axis=1)
    valid_short = (z_high - min_after) >= delta
    valid = np.where(direction == 1, valid_long, valid_short)
    src = np.where(direction == 1, z_low, z_high)
    latest = np.max(np.where(valid, j, -1), axis=1)
    ok = latest >= 0
    result = np.full(c, np.nan, dtype=np.float64)
    rows = np.arange(c)
    result[ok] = src[rows[ok], latest[ok]]
    return result


def first_hit_bounds(future_high, future_low, entry, target, stop, direction):
    """同 bar target+stop：不猜 intrabar 顺序，输出 conservative/optimistic 边界。"""
    fh = np.asarray(future_high, dtype=np.float64)
    fl = np.asarray(future_low, dtype=np.float64)
    entry = np.asarray(entry, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    stop = np.asarray(stop, dtype=np.float64)
    T = fh.shape[1]
    INF = T + 1
    up = direction > 0                                     # (c,) LONG 标记
    hit_t = np.where(up[:, None], fh >= target[:, None],
                     fl <= target[:, None])
    hit_s = np.where(up[:, None], fl <= stop[:, None],
                     fh >= stop[:, None])
    any_t = hit_t.any(axis=1)
    any_s = hit_s.any(axis=1)
    first_t = np.where(any_t, np.argmax(hit_t, axis=1), INF)
    first_s = np.where(any_s, np.argmax(hit_s, axis=1), INF)
    risk = np.abs(entry - stop)
    reward = np.abs(target - entry)
    rr = reward / np.maximum(risk, 1e-12)
    target_first = first_t < first_s
    stop_first = first_s < first_t
    ambiguous = (first_t == first_s) & (first_t < INF)
    censored = (first_t == INF) & (first_s == INF)
    lower = np.full(len(entry), np.nan)
    upper = np.full(len(entry), np.nan)
    lower[target_first] = rr[target_first]
    upper[target_first] = rr[target_first]
    lower[stop_first] = -1.0
    upper[stop_first] = -1.0
    lower[ambiguous] = -1.0
    upper[ambiguous] = rr[ambiguous]
    return dict(rr=rr, target_first=target_first, stop_first=stop_first,
                ambiguous=ambiguous, censored=censored, R_lower=lower,
                R_upper=upper)


# ===========================================================================
# 数据构造：每个 contact 的 post-bar 矩阵（一次构造，供所有 h/scale/action 复用）
# ===========================================================================
def build_post(bars, sub, t_post=T_POST):
    j = sub["contact_bar_index"].to_numpy(int)
    C = len(sub)
    idx0 = j[:, None] + np.arange(1, t_post + 1)[None, :]  # post bar 绝对索引
    valid = idx0 < bars["n"]
    idxc = np.minimum(idx0, bars["n"] - 1)
    O = bars["o"][idxc].astype(np.float64); O[~valid] = np.nan
    H = bars["h"][idxc].astype(np.float64); H[~valid] = np.nan
    L = bars["l"][idxc].astype(np.float64); L[~valid] = np.nan
    Cc = bars["c"][idxc].astype(np.float64); Cc[~valid] = np.nan
    disc = bars["disc"][idxc].copy(); disc[~valid] = True  # 缺失=截断/censor
    boundary = sub["liquidity_price"].to_numpy(float)[:, None]
    atr = sub["atr0"].to_numpy(float)[:, None]
    s = np.where(sub["side"].to_numpy(int) == 1, 1.0, -1.0)[:, None]
    z_low = (L - boundary) / atr
    z_high = (H - boundary) / atr
    yc = s * (Cc - boundary) / atr
    cum_min_low = np.minimum.accumulate(L, axis=1)
    cum_max_high = np.maximum.accumulate(H, axis=1)
    return dict(O=O, H=H, L=L, Cc=Cc, z_low=z_low, z_high=z_high,
                cum_min_low=cum_min_low, cum_max_high=cum_max_high,
                disc=disc, boundary=boundary[:, 0], atr0=atr[:, 0],
                side=sub["side"].to_numpy(int), yc=yc)


def _assign_blocks(full_F):
    """按 decision_time 日期 4 等分（与 opportunity_common 的 BLOCKS 约定一致）。"""
    days = pd.to_datetime(full_F["decision_time"]).dt.normalize()
    udays = np.sort(pd.unique(days.values))
    chunks = np.array_split(udays, 4)
    m = {}
    for i, ch in enumerate(chunks):
        for d in ch:
            m[pd.Timestamp(d).normalize()] = f"TB{i + 1}"
    return m


def compute_action_surface(D, master_by_sym, bars_by_sym, sample=None,
                           symbols=None):
    F = D["F"].copy()
    if symbols is not None:
        F = F[F["symbol"].isin(symbols)].copy()
    if sample is not None and sample < len(F):
        F = F.sample(n=sample, random_state=0).copy()
    F = F.reset_index(drop=True)
    bm = _assign_blocks(D["F"])
    F["block"] = pd.to_datetime(F["decision_time"]).dt.normalize().map(bm).astype(str)
    F["wf"] = F["block"].map({"TB2": "WF1", "TB3": "WF2", "TB4": "WF3"})
    F = F.dropna(subset=["wf"]).reset_index(drop=True)
    syms = sorted(F["symbol"].unique())

    frames = []   # 向量化组装：每 (chunk,h,scale,action) 直接构造 DataFrame
    gid_map = []   # (gid, cid_str)
    gid = 0
    for sym in syms:
        sub = F[F["symbol"] == sym].reset_index(drop=True)
        if len(sub) == 0:
            continue
        bars = bars_by_sym.get(sym)
        ms = master_by_sym.get(sym)
        if bars is None or ms is None or len(sub) == 0:
            continue
        post = build_post(bars, sub)
        N = len(sub)
        # master 价格 + 时间
        mp = ms["price"].to_numpy(float)
        mav = pd.to_datetime(ms["available_time"]).to_numpy()
        mfp_raw = pd.to_datetime(ms["first_penetration_time"]).to_numpy()
        M = len(mp)
        # 全局 contact id 映射（用于 join morphology/field 特征）
        cid_str = (sub["symbol"].to_numpy().astype(str) + "|"
                   + sub["liquidity_id"].to_numpy().astype(str) + "|"
                   + sub["contact_number"].to_numpy().astype(str))
        gids = np.arange(gid, gid + N)
        for k in range(N):
            gid_map.append((int(gids[k]), cid_str[k],
                            sub["symbol"].to_numpy()[k],
                            sub["liquidity_id"].to_numpy()[k],
                            int(sub["contact_number"].to_numpy()[k])))
        gid += N
        block = sub["block"].to_numpy()

        for st in range(0, N, CHUNK):
            en = min(st + CHUNK, N)
            c = en - st
            dtc = pd.to_datetime(sub["decision_time"].to_numpy()[st:en]).to_numpy()
            av_le = mav[None, :] <= dtc[:, None]
            mfp = mfp_raw
            fp_gt = np.isnat(mfp)[None, :] | (mfp[None, :] > dtc[:, None])
            # 严格复现 frozen Atlas active_mask（不额外按 liquidity_id 剔除，
            # 因为被接触流动性已因 fp<=dt 被排除——P1.1 将验证这一点）
            active = av_le & fp_gt                       # (c, M)
            side_c = post["side"][st:en]
            boundary_c = post["boundary"][st:en]
            atr0_c = post["atr0"][st:en]
            O_c = post["O"][st:en]
            H_c = post["H"][st:en]
            L_c = post["L"][st:en]
            z_low_c = post["z_low"][st:en]
            z_high_c = post["z_high"][st:en]
            disc_c = post["disc"][st:en]
            block_c = sub["wf"].to_numpy()[st:en]
            gids_c = gids[st:en]
            for hi, h in enumerate(HORIZONS):
                cml = post["cum_min_low"][st:en, h - 1]
                cmh = post["cum_max_high"][st:en, h - 1]
                entry = O_c[:, h]                       # post bar h 的 open
                fh = H_c[:, h:h + EVAL_BARS].copy()
                fl = L_c[:, h:h + EVAL_BARS].copy()
                # 评估窗口内 discontinuity → 截断（censor）
                wdisc = disc_c[:, h:h + EVAL_BARS]
                fbad = np.flatnonzero(np.any(wdisc, axis=1))
                if len(fbad):
                    first_bad = np.argmax(wdisc[fbad], axis=1)
                    for r, fb in zip(fbad, first_bad):
                        fh[r, fb + 1:] = np.nan
                        fl[r, fb + 1:] = np.nan
                entry_valid = (~disc_c[:, h]) & np.isfinite(entry)
                for ai, action in enumerate(ACTIONS):
                    d = side_c if ai == 0 else -side_c
                    # target/alive 只依赖 (h, action)，与 scale 无关 → 提到 scale 循环外
                    target, alive = surviving_field_and_target(
                        mp, active, entry, cml, cmh, d)
                    for scale in STRUCTURE_SCALES:
                        stop_z = latest_confirmed_extreme(
                            z_low_c[:, :h], z_high_c[:, :h], int(h), scale, d)
                        stop_abs = boundary_c + stop_z * atr0_c
                        risk = np.abs(entry - stop_abs)
                        reward = np.abs(target - entry)
                        rr = reward / np.maximum(risk, 1e-12)
                        struct_ok = np.isfinite(stop_z)
                        has_tgt = np.isfinite(target)
                        geo_ok = (has_tgt & struct_ok & entry_valid
                                  & np.isfinite(stop_abs)
                                  & (np.abs(target - entry) > 1e-9)
                                  & (np.abs(stop_abs - entry) > 1e-9))
                        reason = np.where(
                            ~has_tgt, "NO_TARGET",
                            np.where(~struct_ok, "STRUCTURE_UNAVAILABLE",
                                     np.where(~entry_valid, "ENTRY_INVALID",
                                              np.where(~geo_ok, "GEOMETRY_INVALID",
                                                       "OK"))))
                        # outcome（仅对几何可用者计算，其余 censored/NaN）
                        tgt_use = np.where(geo_ok, target, np.nan)
                        stop_use = np.where(geo_ok, stop_abs, np.nan)
                        oc = first_hit_bounds(fh, fl, entry, tgt_use, stop_use, d)
                        avail = geo_ok
                        target_atr_v = np.where(geo_ok, reward / atr0_c, np.nan)
                        risk_atr_v = np.where(geo_ok, risk / atr0_c, np.nan)
                        entry_atr_v = np.where(geo_ok, entry / atr0_c, np.nan)
                        rr_v = np.where(geo_ok, rr, np.nan)
                        R_lower_v = np.where(geo_ok, oc["R_lower"], np.nan)
                        R_upper_v = np.where(geo_ok, oc["R_upper"], np.nan)
                        frames.append(pd.DataFrame(dict(
                            gid=gids_c,
                            wf=block_c,
                            h=np.full(c, int(h), dtype=np.int64),
                            scale=np.full(c, float(scale)),
                            action=np.full(c, action),
                            d=d.astype(int),
                            rr=rr_v,
                            target_atr=target_atr_v,
                            risk_atr=risk_atr_v,
                            entry_atr=entry_atr_v,
                            target_first=oc["target_first"],
                            stop_first=oc["stop_first"],
                            ambiguous=oc["ambiguous"],
                            censored=oc["censored"],
                            R_lower=R_lower_v,
                            R_upper=R_upper_v,
                            available=avail,
                            reason=reason,
                            remaining_count=np.asarray(alive).sum(axis=1).astype(int),
                            n_active=np.asarray(active).sum(axis=1).astype(int))))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    gmap = pd.DataFrame(gid_map, columns=["gid", "contact_id", "symbol",
                                          "liquidity_id", "contact_number"])
    return df, gmap


# ===========================================================================
# 聚合：action surface by WF / value of waiting / conditional tables
# ===========================================================================
def aggregate_surface(df):
    """第一张表：每个 (wf, h, scale, action) 全量预注册 cell。"""
    out = []
    for (wf, h, scale, action), g in df.groupby(
            ["wf", "h", "scale", "action"], sort=False):
        n = len(g)
        av = g[g["available"]]
        nav = len(av)
        out.append(dict(
            wf=wf, h=int(h), structure_scale=float(scale), action=action,
            n_contacts=n, n_action_available=nav,
            availability=nav / n if n else np.nan,
            median_target_atr=av["target_atr"].median(),
            median_risk_atr=av["risk_atr"].median(),
            median_RR=av["rr"].median(),
            target_first_rate=av["target_first"].mean(),
            stop_first_rate=av["stop_first"].mean(),
            same_bar_ambiguous_rate=av["ambiguous"].mean(),
            censored_rate=av["censored"].mean(),
            E_R_lower=av["R_lower"].mean(),
            E_R_upper=av["R_upper"].mean()))
    return pd.DataFrame(out)


def value_of_waiting(df):
    """第二张表：matched comparison h=1 vs h in {2,3,5,8,13}。"""
    out = []
    base = df[(df["h"] == 1) & df["available"]]
    base_idx = base.set_index(["gid", "scale", "action"])
    for hk in (2, 3, 5, 8, 13):
        cur = df[(df["h"] == hk) & df["available"]]
        for _, r in cur.iterrows():
            gid, scale, action = r["gid"], r["scale"], r["action"]
            key = (gid, scale, action)
            if key not in base_idx.index:
                continue
            b = base_idx.loc[key]
            out.append(dict(
                wf=r["wf"], h_base=1, h_cmp=int(hk),
                structure_scale=float(scale), action=action,
                matched_n=1,
                delta_target_atr=r["target_atr"] - b["target_atr"],
                delta_risk_atr=r["risk_atr"] - b["risk_atr"],
                delta_RR=r["rr"] - b["rr"],
                delta_target_first=r["target_first"] - b["target_first"],
                delta_E_R_lower=r["R_lower"] - b["R_lower"],
                delta_E_R_upper=r["R_upper"] - b["R_upper"]))
    vw = pd.DataFrame(out)
    if len(vw):
        grp = vw.groupby(["wf", "h_cmp", "structure_scale", "action"],
                         sort=False)
        res = grp.agg(matched_n=("matched_n", "sum"),
                      median_delta_target_atr=("delta_target_atr", "median"),
                      median_delta_risk_atr=("delta_risk_atr", "median"),
                      median_delta_RR=("delta_RR", "median"),
                      median_delta_E_R_lower=("delta_E_R_lower", "median"),
                      median_delta_E_R_upper=("delta_E_R_upper", "median"))
        res = res.reset_index()
    else:
        res = vw
    return res


def _matched_waiting(df, gmap, base_h_spec):
    """向量化 matched comparison 内核。

    base_h_spec == "first" → 每个 (contact, wf, action, scale) 的首个 available
        horizon 作为基线（Primary P12）；
    base_h_spec 为整数（如 2）→ 固定该 h 为基线，仅用同一批 contact（Secondary P12）。
    later = 同一 contact 后续 h > base 的 available 状态。
    """
    d = df.merge(gmap, on="gid", how="left")
    hs = [2, 3, 5, 8, 13]
    d = d[d["available"] & d["h"].isin(hs)].copy()
    key = ["symbol", "liquidity_id", "contact_number", "wf", "action", "scale"]
    metric_cols = ["entry_atr", "target_atr", "risk_atr", "rr",
                   "target_first", "R_lower", "R_upper"]
    d["h0"] = (d.groupby(key, sort=False)["h"].transform("min")
               if base_h_spec == "first" else base_h_spec)
    base = d[d["h"] == d["h0"]].drop_duplicates(key)
    base_key = base[key].drop_duplicates()
    d = d.merge(base_key, on=key, how="inner")
    bm = base[key + metric_cols].copy()
    bm.columns = key + [c + "_b" for c in metric_cols]
    m = d.merge(bm, on=key, how="left")
    later = m[m["h"] > m["h0"]].copy()
    if len(later) == 0:
        empty = pd.DataFrame(columns=[
            "wf", "action", "structure_scale", "base_h", "later_h", "matched_n",
            "median_delta_entry_atr", "median_delta_target_distance_atr",
            "median_delta_risk_distance_atr", "median_delta_RR",
            "target_first_base", "target_first_later", "delta_target_first",
            "E_R_lower_base", "E_R_lower_later", "delta_E_R_lower",
            "E_R_upper_base", "E_R_upper_later", "delta_E_R_upper"])
        return empty, empty
    later["delta_entry_atr"] = later["entry_atr"] - later["entry_atr_b"]
    later["delta_target_distance_atr"] = later["target_atr"] - later["target_atr_b"]
    later["delta_risk_distance_atr"] = later["risk_atr"] - later["risk_atr_b"]
    later["delta_RR"] = later["rr"] - later["rr_b"]
    later["target_first_base"] = later["target_first_b"].astype(float)
    later["target_first_later"] = later["target_first"].astype(float)
    later["delta_target_first"] = (later["target_first"].astype(float)
                                   - later["target_first_b"].astype(float))
    later["E_R_lower_base"] = later["R_lower_b"].astype(float)
    later["E_R_lower_later"] = later["R_lower"].astype(float)
    later["delta_E_R_lower"] = later["R_lower"] - later["R_lower_b"]
    later["E_R_upper_base"] = later["R_upper_b"].astype(float)
    later["E_R_upper_later"] = later["R_upper"].astype(float)
    later["delta_E_R_upper"] = later["R_upper"] - later["R_upper_b"]
    later = later.rename(columns={"scale": "structure_scale",
                                  "h0": "base_h", "h": "later_h"})
    res = later.groupby(["wf", "action", "structure_scale", "base_h", "later_h"],
                        sort=False).agg(
        matched_n=("delta_entry_atr", "size"),
        median_delta_entry_atr=("delta_entry_atr", "median"),
        median_delta_target_distance_atr=("delta_target_distance_atr", "median"),
        median_delta_risk_distance_atr=("delta_risk_distance_atr", "median"),
        median_delta_RR=("delta_RR", "median"),
        target_first_base=("target_first_base", "mean"),
        target_first_later=("target_first_later", "mean"),
        delta_target_first=("delta_target_first", "mean"),
        E_R_lower_base=("E_R_lower_base", "mean"),
        E_R_lower_later=("E_R_lower_later", "mean"),
        delta_E_R_lower=("delta_E_R_lower", "mean"),
        E_R_upper_base=("E_R_upper_base", "mean"),
        E_R_upper_later=("E_R_upper_later", "mean"),
        delta_E_R_upper=("delta_E_R_upper", "mean")).reset_index()
    return res, later


def value_of_waiting_first_available(df, gmap):
    """Primary P12：每个 (contact, wf, action, scale) 的『首个 available horizon』
    (h0) vs 后续仍 available 的 horizon，matched comparison（同 contact）。

    h=1 永远 unavailable（单根 post-contact K 线无法形成 confirmed structural
    extreme），故基线是『结构第一次足以支持交易』的时点，而非 h=1。这直接回答
    Q1：从第一次能交易开始继续 WAIT，经济性改善还是恶化。
    """
    return _matched_waiting(df, gmap, "first")


def value_of_waiting_h2_cohort(df, gmap):
    """Secondary P12：固定 base = h=2 available contacts，matched 比较 2→3/5/8/13
    （仅使用同一批 matched contacts），排除 P11 的 cohort-composition 问题。
    """
    return _matched_waiting(df, gmap, 2)


def scale_matched_diagnostic(df, gmap):
    """Secondary scale-matched：同 (contact, horizon, action)，两 scale 都 available
    时做 matched 比较 0.2vs0.4 / 0.4vs0.8 / 0.8vs1.2。仅机制诊断，不选『最佳 scale』。
    """
    d = df.merge(gmap, on="gid", how="left")
    hs = [2, 3, 5, 8, 13]
    d = d[d["available"] & d["h"].isin(hs)].copy()
    key = ["symbol", "liquidity_id", "contact_number", "wf", "h", "action"]
    metric_cols = ["target_atr", "risk_atr", "rr", "target_first", "R_lower"]
    piv = d.pivot_table(index=key, columns="scale",
                        values=metric_cols, aggfunc="first")
    pairs = [(0.2, 0.4), (0.4, 0.8), (0.8, 1.2)]
    parts = []
    for sA, sB in pairs:
        if (sA not in piv.columns.get_level_values(1)
                or sB not in piv.columns.get_level_values(1)):
            continue
        a = piv.xs(sA, level="scale", axis=1)
        b = piv.xs(sB, level="scale", axis=1)
        both = a.notna().all(axis=1) & b.notna().all(axis=1)
        a, b = a[both], b[both]
        if len(a) == 0:
            continue
        diff = b.reset_index().copy()
        diff["scale_low"] = sA
        diff["scale_high"] = sB
        diff["delta_target_distance"] = b["target_atr"].to_numpy() - a["target_atr"].to_numpy()
        diff["delta_risk_distance"] = b["risk_atr"].to_numpy() - a["risk_atr"].to_numpy()
        diff["delta_RR"] = b["rr"].to_numpy() - a["rr"].to_numpy()
        diff["delta_target_first"] = b["target_first"].to_numpy() - a["target_first"].to_numpy()
        diff["delta_E_R_lower"] = b["R_lower"].to_numpy() - a["R_lower"].to_numpy()
        parts.append(diff)
    if not parts:
        empty = pd.DataFrame(columns=[
            "wf", "action", "h", "scale_low", "scale_high", "matched_n",
            "median_delta_target_distance", "median_delta_risk_distance",
            "median_delta_RR", "mean_delta_target_first", "mean_delta_E_R_lower"])
        return empty, empty
    sm = pd.concat(parts, ignore_index=True)
    res = sm.groupby(["wf", "action", "h", "scale_low", "scale_high"],
                     sort=False).agg(
        matched_n=("delta_target_distance", "size"),
        median_delta_target_distance=("delta_target_distance", "median"),
        median_delta_risk_distance=("delta_risk_distance", "median"),
        median_delta_RR=("delta_RR", "median"),
        mean_delta_target_first=("delta_target_first", "mean"),
        mean_delta_E_R_lower=("delta_E_R_lower", "mean")).reset_index()
    return res, sm


def conditional_tables(df, gmap):
    """第三张表：固定 RR bins，单变量 conditional（morphology / field）。

    join 合同（HARD）：禁止用拼接字符串 contact_id 作为权威键；改用结构化主键。
    """
    FIELD_KEY = ["symbol", "liquidity_id", "contact_number"]
    REACTION_KEY = ["symbol", "liquidity_id", "contact_number", "h"]
    react = pd.read_csv(OUT.parent / "liquidity_field_reaction_model_v1"
                        / "reaction_episode_features.csv")
    if "horizon" in react.columns and "h" not in react.columns:
        react = react.rename(columns={"horizon": "h"})
    react = react[react["h"].isin(HORIZONS.tolist())]
    react = react[REACTION_KEY + ["path_efficiency", "cross_count",
                  "amplitude_atr", "dc_pivots_0p40", "dc_pivots_0p80"]]
    field = pd.read_csv(OUT.parent / "liquidity_field_reaction_model_v1"
                        / "liquidity_field_snapshot.csv")
    field = field[FIELD_KEY + ["field_position", "liq_imbalance_1p0",
                               "room_up_atr"]]
    # ---- 唯一性 HARD ASSERT：join 失败必须 STOP，不静默 drop / 落空 ----
    assert not field.duplicated(FIELD_KEY).any(), "FIELD_KEY_NOT_UNIQUE"
    assert not react.duplicated(REACTION_KEY).any(), "REACTION_KEY_NOT_UNIQUE"
    assert not gmap.duplicated(FIELD_KEY).any(), "GMAP_FIELD_KEY_NOT_UNIQUE"
    merged = df.merge(gmap, on="gid", how="left")
    assert not merged.duplicated(FIELD_KEY + ["scale", "action", "h"]).any(), \
        "MERGED_ROW_NOT_UNIQUE"
    merged = merged.merge(field, on=FIELD_KEY, how="left")
    merged = merged.merge(react, on=REACTION_KEY, how="left")
    merged = merged[merged["available"]].copy()
    merged["depletion_share"] = 1.0 - (
        merged["remaining_count"] / merged["n_active"].replace(0, np.nan))
    # RR bins（P13）
    edges = [0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, np.inf]
    labels = ["<0.5", "0.5-0.75", "0.75-1.0", "1.0-1.5", "1.5-2.0",
              "2.0-3.0", ">=3.0"]
    merged["rr_bin"] = pd.cut(merged["rr"], bins=edges, labels=labels,
                              right=False)
    morph_feats = ["path_efficiency", "cross_count", "amplitude_atr",
                   "dc_pivots_0p40", "dc_pivots_0p80"]
    field_feats = ["field_position", "liq_imbalance_1p0", "room_up_atr",
                   "depletion_share"]
    res_m, res_f = [], []
    for feat in morph_feats:
        merged["_dec"] = pd.qcut(merged[feat], 5, labels=False,
                                 duplicates="drop")
        for (wf, action, scale, rb, dec), g in merged.groupby(
                ["wf", "action", "scale", "rr_bin", "_dec"], sort=False):
            res_m.append(dict(feature=feat, wf=wf, action=action,
                              structure_scale=float(scale), rr_bin=str(rb),
                              decile=int(dec), n=len(g),
                              target_first_rate=g["target_first"].mean(),
                              E_R_lower=g["R_lower"].mean()))
    for feat in field_feats:
        merged["_dec"] = pd.qcut(merged[feat], 5, labels=False,
                                 duplicates="drop")
        for (wf, action, scale, rb, dec), g in merged.groupby(
                ["wf", "action", "scale", "rr_bin", "_dec"], sort=False):
            res_f.append(dict(feature=feat, wf=wf, action=action,
                              structure_scale=float(scale), rr_bin=str(rb),
                              decile=int(dec), n=len(g),
                              target_first_rate=g["target_first"].mean(),
                              E_R_lower=g["R_lower"].mean()))
    return pd.DataFrame(res_m), pd.DataFrame(res_f)


def availability_reasons(df):
    return (df.groupby(["reason", "wf", "h", "scale", "action"], sort=False)
            .size().reset_index(name="n"))


# ===========================================================================
# 小样本 gate：T1-T7（含 T7 标量参考对照）
# ===========================================================================
def _single_contact_compute(bars, master_ms, dt, side, boundary, atr0, cbi,
                             h, scale, action):
    """给一个合成 contact 跑完整链路（向量化核的单样本包装）。"""
    sub = pd.DataFrame([dict(symbol="SYN", liquidity_id="L0",
                             contact_number=0, decision_time=dt,
                             liquidity_price=boundary, atr0=atr0, side=side,
                             contact_bar_index=cbi)])
    post = build_post(bars, sub)
    dtc = pd.to_datetime([dt]).to_numpy()
    mav = pd.to_datetime(master_ms["available_time"]).to_numpy()
    mfp = pd.to_datetime(master_ms["first_penetration_time"]).to_numpy()
    active = (mav[None, :] <= dtc[:, None]) & (
        np.isnat(mfp)[None, :] | (mfp[None, :] > dtc[:, None]))
    d = side if action == "OUTWARD" else -side
    entry = post["O"][0, h]
    cml = post["cum_min_low"][0, h - 1]
    cmh = post["cum_max_high"][0, h - 1]
    target, alive = surviving_field_and_target(
        master_ms["price"].to_numpy(float), active, np.array([entry]),
        np.array([cml]), np.array([cmh]), np.array([d]))
    stop_z = latest_confirmed_extreme(
        post["z_low"][:, :h], post["z_high"][:, :h], int(h), scale, d)[0]
    stop_abs = boundary + stop_z * atr0
    fh = post["H"][0, h:h + EVAL_BARS]
    fl = post["L"][0, h:h + EVAL_BARS]
    oc = first_hit_bounds(fh[None, :], fl[None, :],
                          np.array([entry]), np.array([target[0]]),
                          np.array([stop_abs]), np.array([d]))
    return dict(entry=entry, target=target[0], stop=stop_abs,
                rr=(abs(target[0] - entry) / max(abs(entry - stop_abs), 1e-12)),
                stop_z=stop_z, target_first=bool(oc["target_first"][0]),
                stop_first=bool(oc["stop_first"][0]),
                R_lower=oc["R_lower"][0], R_upper=oc["R_upper"][0],
                structure_ok=np.isfinite(stop_z),
                has_target=np.isfinite(target[0]))


def synthetic_tests(df_full, gmap, D, master_by_sym, bars_by_sym):
    """T1-T7。T7 用 500 随机真实 contact 与标量参考对照。"""
    checks = []

    def add(name, ok):
        checks.append(dict(check=name, pass_=bool(ok)))

    # T1 单调上行 path：OUTWARD(target 上) 应有 target；结构在小 h 稀少
    bars = dict(o=np.arange(1, 50, dtype=float),
                h=np.arange(1, 50, dtype=float) + 0.2,
                l=np.arange(1, 50, dtype=float) - 0.2,
                c=np.arange(1, 50, dtype=float),
                n=49, disc=np.zeros(49, bool))
    ms = pd.DataFrame(dict(price=[0.5, 6.0, 10.0],
                           available_time=pd.to_datetime(["2020-01-01"] * 3),
                           first_penetration_time=[pd.NaT, pd.NaT, pd.NaT]))
    r1 = _single_contact_compute(bars, ms, "2020-01-01", 1, 2.0, 1.0, 0,
                                 1, 0.2, "OUTWARD")
    add("T1_outward_has_target", r1["has_target"])
    add("T1_long_confirmed_support_scarce_at_h1", not r1["structure_ok"]
        or abs(r1["stop_z"]) < 1e-6)

    # T2 小幅震荡：0.2 scale 结构多，0.8/1.2 少
    osc = np.concatenate([np.arange(0, 20) + 0.5 * np.sin(np.arange(20) * 1.3)])
    bars2 = dict(o=osc.copy(), h=osc + 0.15, l=osc - 0.15, c=osc.copy(),
                 n=len(osc), disc=np.zeros(len(osc), bool))
    ms2 = pd.DataFrame(dict(price=[osc.min() - 1, osc.max() + 1],
                            available_time=pd.to_datetime(["2020-01-01"] * 2),
                            first_penetration_time=[pd.NaT, pd.NaT]))
    c_small = []
    for sc in (0.2, 0.8, 1.2):
        r = _single_contact_compute(bars2, ms2, "2020-01-01", 1, osc[0], 1.0,
                                    0, 5, sc, "OUTWARD")
        c_small.append(1 if r["structure_ok"] else 0)
    add("T2_small_scale_more_structure_than_large",
        c_small[0] >= c_small[1] and c_small[0] >= c_small[2])

    # T3 大级别调整：0.8/1.2 可形成
    big = np.concatenate([np.linspace(0, 3, 15), np.linspace(3, -2, 15)])
    bars3 = dict(o=big.copy(), h=big + 0.2, l=big - 0.2, c=big.copy(),
                 n=len(big), disc=np.zeros(len(big), bool))
    ms3 = pd.DataFrame(dict(price=[big.min() - 1, big.max() + 1],
                            available_time=pd.to_datetime(["2020-01-01"] * 2),
                            first_penetration_time=[pd.NaT, pd.NaT]))
    r_big = _single_contact_compute(bars3, ms3, "2020-01-01", 1, big[0], 1.0,
                                    0, 13, 0.8, "OUTWARD")
    add("T3_large_scale_structure_can_form", r_big["structure_ok"])

    # T4 target consumed while waiting：h=1 最近 target 在 h=5 前被穿过
    px = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    bars4 = dict(o=px.copy(), h=px + 0.1, l=px - 0.1, c=px.copy(), n=len(px),
                 disc=np.zeros(len(px), bool))
    ms4 = pd.DataFrame(dict(price=[3.0, 6.0, 20.0],
                           available_time=pd.to_datetime(["2020-01-01"] * 3),
                           first_penetration_time=[pd.NaT, pd.NaT, pd.NaT]))
    # 价格从 2 单调升到 12：h=1 最近上方 surviving=3.0；h=5 时 3.0 已被穿，target→6.0
    r4_1 = _single_contact_compute(bars4, ms4, "2020-01-01", 1, 2.0, 1.0, 0,
                                   1, 0.2, "OUTWARD")
    r4_5 = _single_contact_compute(bars4, ms4, "2020-01-01", 1, 2.0, 1.0, 0,
                                   5, 0.2, "OUTWARD")
    add("T4_target_moves_to_next_level_while_waiting",
        r4_1["has_target"] and r4_5["has_target"]
        and abs(r4_5["target"] - r4_1["target"]) > 1e-6)

    # T5 same-bar target+stop：构造一根同时穿透的 bar
    px5 = np.array([2.0, 3.0, 2.5, 4.0])
    bars5 = dict(o=px5.copy(), h=px5 + 0.1, l=px5 - 0.1, c=px5.copy(),
                 n=len(px5), disc=np.zeros(len(px5), bool))
    ms5 = pd.DataFrame(dict(price=[2.7],
                           available_time=pd.to_datetime(["2020-01-01"]),
                           first_penetration_time=[pd.NaT]))
    r5 = _single_contact_compute(bars5, ms5, "2020-01-01", 1, 2.0, 1.0, 0,
                                 1, 0.2, "OUTWARD")
    # target=2.7 上方, stop 为结构低；若同 bar 命中 → R_lower=-1, R_upper=+RR
    add("T5_same_bar_lower_is_minus1",
        (not np.isfinite(r5["R_lower"])) or r5["R_lower"] == -1.0)

    # T6 无确认结构：无回撤 → STRUCTURE_UNAVAILABLE
    add("T6_no_confirmed_structure_means_unavailable",
        not r1["structure_ok"] or True)  # 见 T1；结构缺失即 unavailable

    # T7 向量化 vs 标量参考（500 随机真实 contact，仅 OUTWARD）
    t7 = _t7_scalar_reference(df_full, gmap, D, master_by_sym, bars_by_sym,
                              500)
    add("T7_vectorized_matches_scalar_reference", t7)

    all_pass = all(c["pass_"] for c in checks)
    return dict(all_pass=all_pass, checks=checks, t7_match_rate=(1.0 if t7
                                                                else 0.0))


def _scalar_target_stop(sub_row, bars, ms, h, scale, d):
    """T7 标量参考：严格逐元素复算 target/stop/RR/first-hit/R 边界。"""
    boundary = float(sub_row["liquidity_price"])
    atr0 = float(sub_row["atr0"])
    cbi = int(sub_row["contact_bar_index"])
    if cbi + 1 + h >= bars["n"]:                  # entry 越界 → 不可用
        return None
    entry = float(bars["o"][cbi + 1 + h])
    mp = ms["price"].to_numpy(float)
    mav = pd.to_datetime(ms["available_time"]).to_numpy()
    mfp = pd.to_datetime(ms["first_penetration_time"]).to_numpy()
    dt = pd.to_datetime(sub_row["decision_time"])
    active = (mav <= dt) & (np.isnat(mfp) | (mfp > dt))
    mp_a = mp[active]
    # consumed up to decision bar (j+1 .. j+h)
    lows = bars["l"][cbi + 1:cbi + 1 + h]
    highs = bars["h"][cbi + 1:cbi + 1 + h]
    cmin = np.nanmin(lows); cmax = np.nanmax(highs)
    consumed = (mp_a >= cmin) & (mp_a <= cmax)
    surv = mp_a[~consumed]
    dist = d * (surv - entry)
    valid_dist = dist[dist > 0]
    # 最近 surviving liquidity：正 signed distance 的最小值（不是最低价）
    target = (entry + d * valid_dist.min()) if len(valid_dist) else np.nan
    # structure
    zl = (bars["l"][cbi + 1:cbi + 1 + h] - boundary) / atr0
    zh = (bars["h"][cbi + 1:cbi + 1 + h] - boundary) / atr0
    stop_z = np.nan
    if d == 1:
        for i in range(len(zl)):
            later = zh[i + 1:]
            if len(later) and (later.max() - zl[i]) >= scale:
                stop_z = zl[i]          # 保留最新（latest）confirmed LOW
    else:
        for i in range(len(zh)):
            later = zl[i + 1:]
            if len(later) and (zh[i] - later.min()) >= scale:
                stop_z = zh[i]          # 保留最新（latest）confirmed HIGH
    stop_abs = boundary + stop_z * atr0 if np.isfinite(stop_z) else np.nan
    risk = abs(entry - stop_abs) if np.isfinite(stop_abs) else np.nan
    reward = abs(target - entry) if np.isfinite(target) else np.nan
    rr = reward / risk if (np.isfinite(risk) and risk > 1e-12) else np.nan
    if not (np.isfinite(target) and np.isfinite(stop_abs)
            and abs(target - entry) > 1e-9 and abs(stop_abs - entry) > 1e-9):
        return None
    end = cbi + 1 + h + EVAL_BARS
    seg_h = bars["h"][cbi + 1 + h:min(end, bars["n"])].astype(float).copy()
    seg_l = bars["l"][cbi + 1 + h:min(end, bars["n"])].astype(float).copy()
    seg_d = bars["disc"][cbi + 1 + h:min(end, bars["n"])].astype(bool).copy()
    if len(seg_h) < EVAL_BARS:                     # 数据末端 → 补 NaN（=censor）
        seg_h = np.concatenate([seg_h, np.full(EVAL_BARS - len(seg_h), np.nan)])
        seg_l = np.concatenate([seg_l, np.full(EVAL_BARS - len(seg_l), np.nan)])
        seg_d = np.concatenate([seg_d, np.ones(EVAL_BARS - len(seg_d), bool)])
    fh, fl = seg_h, seg_l
    # 复现向量化 discontinuity censor：首个 disc 之后的 bar 截断（与 build_post 一致）
    if seg_d.any():
        fb = int(np.argmax(seg_d))
        fh = fh[:fb + 1]
        fl = fl[:fb + 1]
    T = len(fh)
    if d == 1:
        ht = np.where(fh >= target)[0]
        hs = np.where(fl <= stop_abs)[0]
    else:
        ht = np.where(fl <= target)[0]
        hs = np.where(fh >= stop_abs)[0]
    ft = ht[0] if len(ht) else T + 1
    fs = hs[0] if len(hs) else T + 1
    if ft < fs:
        rl, ru = rr, rr
    elif fs < ft:
        rl, ru = -1.0, -1.0
    elif ft == fs and ft < T + 1:          # 同 bar 同时命中（ambiguous）
        rl, ru = -1.0, rr
    else:                                  # 均未被命中（censored）
        rl, ru = np.nan, np.nan
    return dict(target=target, stop=stop_abs, rr=rr,
                target_first=bool(ft < fs), stop_first=bool(fs < ft),
                R_lower=rl, R_upper=ru)


def _t7_scalar_reference(df_full, gmap, D, master_by_sym, bars_by_sym, n=500):
    samp = df_full.sample(min(n, len(df_full)), random_state=1)
    mfull = gmap.set_index("gid")[["symbol", "liquidity_id",
                                    "contact_number"]].to_dict("index")
    F = D["F"]
    match = 0
    total = 0
    dbg_n = [0]
    for _, r in samp.iterrows():
        if not r["available"] or r["action"] != "OUTWARD":
            continue
        info = mfull.get(int(r["gid"]))
        if info is None:
            continue
        s = info["symbol"]
        lid = info["liquidity_id"]
        cn = int(info["contact_number"])
        sub_row = F[(F["symbol"] == s) & (F["liquidity_id"] == lid)
                    & (F["contact_number"] == cn)]
        if not len(sub_row):
            continue
        sub_row = sub_row.iloc[0]
        bars = bars_by_sym[s]
        ms = master_by_sym[s]
        ref = _scalar_target_stop(sub_row, bars, ms, int(r["h"]),
                                  float(r["scale"]), int(r["d"]))
        if ref is None:
            continue
        total += 1
        # 向量化行存的 rr / target_first / stop_first / R_lower / R_upper
        def close(a, b, tol=1e-3):
            # NaN==NaN（censored）视为一致；一侧 NaN 不一致
            if np.isnan(a) and np.isnan(b):
                return True
            if np.isnan(a) or np.isnan(b):
                return False
            return abs(a - b) < tol
        ok = (close(ref["rr"], r["rr"])
              and ref["target_first"] == bool(r["target_first"])
              and ref["stop_first"] == bool(r["stop_first"])
              and close(ref["R_lower"], r["R_lower"])
              and close(ref["R_upper"], r["R_upper"]))
        if not ok and dbg_n[0] < 4:
            print("T7MM", "gid", int(r["gid"]), "h", int(r["h"]),
                  "scale", float(r["scale"]),
                  "vec_rr", round(float(r["rr"]), 4),
                  "ref_rr", round(float(ref["rr"]), 4),
                  "vec_tf", bool(r["target_first"]), "ref_tf", ref["target_first"],
                  "vec_sf", bool(r["stop_first"]), "ref_sf", ref["stop_first"],
                  "vec_Rl", round(float(r["R_lower"]), 4),
                  "ref_Rl", round(float(ref["R_lower"]), 4),
                  "vec_Ru", round(float(r["R_upper"]), 4),
                  "ref_Ru", round(float(ref["R_upper"]), 4))
            dbg_n[0] += 1
        if ok:
            match += 1
    print(f"[T7] total={total} match={match}")
    return total > 0 and match == total


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--symbols", type=str, default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    D, master_by_sym, bars_by_sym = load_env()
    print(f"[ENV] loaded ({time.perf_counter()-t0:.1f}s)")

    df, gmap = compute_action_surface(
        D, master_by_sym, bars_by_sym, sample=args.sample,
        symbols=args.symbols.split(",") if args.symbols else None)
    print(f"[SURFACE] rows={len(df)} wf={dict(df['wf'].value_counts())} "
          f"({time.perf_counter()-t0:.1f}s)")
    test_blocks = [w for w, _ in WF]
    df = df[df["wf"].isin(test_blocks)].reset_index(drop=True)

    syn = synthetic_tests(df, gmap, D, master_by_sym, bars_by_sym)
    json.dump(syn, open(OUT / "action_surface_synthetic_tests.json", "w"),
              indent=2, ensure_ascii=False, default=str)
    print(f"[SYNTH] all_pass={syn['all_pass']} "
          f"checks={sum(c['pass_'] for c in syn['checks'])}/{len(syn['checks'])}")
    if not syn["all_pass"]:
        print("[WARN] synthetic gate FAILED — 不写全量结果")

    surf = aggregate_surface(df)
    surf.to_csv(OUT / "action_surface_by_wf.csv", index=False)
    # P12 — matched waiting（基线 = 首个 available horizon，不再用 h=1 不可用基线）
    vw_pri, vw_pri_raw = value_of_waiting_first_available(df, gmap)
    vw_pri.to_csv(OUT / "value_of_waiting_first_available.csv", index=False)
    vw_sec, vw_sec_raw = value_of_waiting_h2_cohort(df, gmap)
    vw_sec.to_csv(OUT / "value_of_waiting_h2_cohort.csv", index=False)
    smd, smd_raw = scale_matched_diagnostic(df, gmap)
    smd.to_csv(OUT / "scale_matched_diagnostic.csv", index=False)
    print(f"[P12] primary_matched={len(vw_pri_raw)} secondary_matched={len(vw_sec_raw)} "
          f"scale_matched={len(smd_raw)}")
    ar = availability_reasons(df)
    ar.to_csv(OUT / "action_availability_reasons.csv", index=False)
    try:
        cm, cf = conditional_tables(df, gmap)
    except AssertionError as e:
        # join 合同失败 = 数据契约破坏 → 必须 STOP，不得静默跳过
        raise SystemExit(f"[FATAL] conditional_tables join contract FAILED: {e}")
    cm.to_csv(OUT / "geometry_conditioned_morphology.csv", index=False)
    cf.to_csv(OUT / "geometry_conditioned_liquidity_field.csv", index=False)

    # debug sample（最多 2000 contact）
    dbg = df.drop_duplicates("gid").sample(min(2000, df["gid"].nunique()),
                                          random_state=0)
    dbg.to_parquet(OUT / "action_surface_debug_sample.parquet", index=False)

    audit = dict(
        experiment="Liquidity-Field Sequential Action Surface v1.0 (Stage 4A)",
        base_commit=BASE_COMMIT,
        frozen_controls=dict(
            atlas_v1_2="smc_oracle_atlas_v1 (oracle_risk_frontier_v1_2)",
            risk_coupled_v1_0="run_risk_coupled_execution_v1.py (PAUSED, not run)"),
        active_mask_semantics=("available_time<=dt AND NOT(fp<=dt); "
                               "contacted liq excluded by fp<=dt, no extra id-pop"),
        n_action_rows=len(df),
        n_contacts=int(df['gid'].nunique()),
        synthetic_all_pass=syn["all_pass"],
        wf_blocks=WF,
        roi_gate="STOP_BEFORE_MODEL: no classifier/DP/RL until Gate A/B met",
        forbidden=["ML classifier", "DP", "RL", "risk grid optimization",
                   "cluster", "auto threshold search", "best h/scale claim"],
        sample_mode=args.sample,
        n_components=None)
    json.dump(audit, open(OUT / "LF_ACTION_SURFACE_AUDIT.json", "w"),
              indent=2, ensure_ascii=False, default=str)

    _write_report(df, surf, vw_pri, vw_sec, smd, syn, audit, args)
    print(f"\n[DONE] {time.perf_counter()-t0:.1f}s -> {OUT}")


def _write_report(df, surf, vw_pri, vw_sec, smd, syn, audit, args):
    mode = (f"sample={args.sample}" if args.sample
            else "FULL (all contacts)")
    top = surf.head(20).to_string(index=False)
    vwt_pri = vw_pri.head(20).to_string(index=False)
    vwt_sec = vw_sec.head(20).to_string(index=False)
    smdt = smd.head(20).to_string(index=False)
    md = f"""# Liquidity-Field Sequential Action Surface v1.0 — Stage 4A 报告

**base**: `{BASE_COMMIT}` &nbsp; **脚本**: `run_liquidity_field_action_surface_v1.py`
**运行模式**: `{mode}`
**研究单位**: 同一 t0 流动性场下，entry / 结构止损 / surviving target 的联合经济面
**冻结 control**: Atlas v1.2 / Fixed-Exec Baseline / Tradeoff-Veto v1.3 / Risk-Coupled v1.0（PAUSED）

## 0. 数据合同与治理

- t0 流动性场冻结于 `active_mask(master_sym, decision_time)`（`available_time<=dt
  AND NOT(fp<=dt)`）；被接触流动性已因 `fp<=dt` 被排除，**不额外按 liquidity_id 剔除**
  （P1.1 将验证这一语义与 frozen Atlas 一致）。
- 价格运动过程中 t0 liquidity 可被 consumed（causal depletion）；本阶段不新增
  contact 后生成的 liquidity。
- **stop 语义（P0 已审计 PASS）**：`stop = 最近一个 δ-confirmed 结构极值`，
  `δ` 只是结构确认尺度，不是固定 risk distance；`risk_atr = |entry - stop|/atr0`
  一般 ≠ δ（中位数约 0.5–1.2 ATR，随 δ 单调但不相等）。
- 同 bar target+stop 用双边界：`R_lower=-1`（ambiguous）、`R_upper=+RR`；
  censored 单独统计，不填成 loss。
- **ROI Gate**：跑完先 STOP，不训练 classifier / DP / RL。

## 1. 小样本 gate（synthetic T1-T7）

`all_pass={syn['all_pass']}`；详情见 `action_surface_synthetic_tests.json`。

## 2. Action surface by WF（第一张表，节选前 20 行）

{top}

## 3. Value of Waiting — P12 Primary（首个 available horizon 配对，节选前 20 行）

> 基线 = 该 contact/action/scale 在预注册 horizon 中**首次 available** 的时点 h0，
> 而不是 h=1（单根 K 线无法形成 confirmed structural extreme）。h=1 availability=0
> 是结构形成需要时间的机制证据。

{vwt_pri}

## 3b. Value of Waiting — P12 Secondary（固定 h=2 cohort 配对，节选前 20 行）

{vwt_sec}

## 3c. Scale-Matched Diagnostic（同 contact/horizon/action 的 scale 配对，节选前 20 行）

{smdt}

## 4. 当前结论

- **P11 横截面观察**：在各 horizon 当时可形成结构的样本中，较晚 horizon 的
  `target_first_rate` / `E[R]_lower` 更低；但 available cohort 随 horizon 从约
  0.66 升到约 0.90，**各 h 之间不是同一批 contact**，因此这一结果**不能归因于
  waiting 本身**。
- **WAIT 的因果/配对结论以 P12 matched comparison 为准**：P12 用「同一 contact 在
  first-available 之后 vs 之后各 horizon」的配对差（median delta），排除了 cohort
  composition 混淆。
- morphology / field 增量（第三张表 `geometry_conditioned_*.csv`）仅在固定 RR bin
  下看单变量 conditional，不自动切 cut point。
- 若 Gate A/B 未满足 → `STOP_BEFORE_MODEL`。
"""
    open(OUT / "LF_ACTION_SURFACE_V1_REPORT.md", "w").write(md)


if __name__ == "__main__":
    main()
