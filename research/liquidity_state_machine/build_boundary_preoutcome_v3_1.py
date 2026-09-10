"""订单块边界局部反应实验 v3.1 —— 阶段一：完整市场状态快照 + Gate。

本阶段**不加载、不计算任何 outcome**。

核心识别（保持不变）：
    running = ob_margin_R = penetration_depth_R - nearest_ob_distance_R
    ob_margin_R < 0  -> 还没进入最近反向 OB
    ob_margin_R >= 0 -> 已进入最近反向 OB

v3.1 升级（Amendment）：
  A. 多周期趋势快照 4h / 1h / 15m / 5m + 派生方向关系
  B. 当前被扫 liquidity 自身的 id / type / source_tf / scope / side
  C. 同价位（严格数值相等）多周期 liquidity 叠加
  D. sweep 前方下一个 / 反方向最近 liquidity 距离
  E. OB 上下文（周期 / 宽度 / freshness / 进入次数）
  F. 跨周期组合 liquidity_tf_x_ob_tf

所有 context 一律满足：available_time <= penetration_bar_start。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import research.m2_nondeep_temporal_v1 as m2
from research.build_pytdx_panel import aggregate_15m
from research.liquidity_state_machine.liquidity_specificity_placebo_v2 import (
    SymbolData,
)
from research.liquidity_state_machine.match_liquidity_placebo_v2 import rel

RESULTS = Path("research/analysis_results/liquidity_ob_boundary_v3_1")
RESULTS.mkdir(parents=True, exist_ok=True)

BANDWIDTHS_R = [0.10, 0.20, 0.30, 0.50]
N_MIN_TOTAL = 2000
N_MIN_SIDE = 750
N_MIN_SYMBOL_SIDE = 20
N_MIN_SYMBOL_OK = 10

# Outcome Lock
FORBIDDEN = {"later_reclaim", "stage2", "stage2_state",
             "structural_acceptance"}

# 与研究变量结构相关，不做平衡要求
NO_BALANCE = {"nearest_ob_distance_R", "ob_margin_R"}

NUMERIC_AUDIT = ["penetration_depth_R", "close_beyond_R", "bar_range_R",
                 "abs_return_R", "pre_range_12_R", "pre_rv_12_R",
                 "pre_ret_3_R", "pre_ret_12_R", "volume_z_t0",
                 "level_age_log1p", "atr_rel_pre", "volume_z_20_pre",
                 "minute_from_session_open",
                 "next_liquidity_distance_R",
                 "opposite_liquidity_distance_R",
                 "nearest_ob_width_R"]
KEY_NUMERIC = ["penetration_depth_R", "close_beyond_R", "pre_range_12_R"]

CATEG_AUDIT = ["symbol", "liquidity_type", "liquidity_source_tf",
               "side", "session_type", "same_price_has_htf_liquidity",
               "nearest_ob_source_tf", "nearest_ob_freshness",
               "sweep_vs_1h", "sweep_vs_15m",
               "env4h_vs_1h", "trend_1h_vs_15m"]

CAT_CANON_MAX_PP = 8.0


def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.nanmean(a) - np.nanmean(b)
    p = np.sqrt(.5 * (np.nanvar(a, ddof=1) + np.nanvar(b, ddof=1)))
    return 0.0 if (not np.isfinite(p) or p < 1e-12) else float(m / p)


def session_minute(t) -> np.ndarray:
    """按 5 分钟断口分段，返回段内分钟偏移。"""
    ts = pd.to_datetime(pd.Series(t))
    brk = ts.diff() > pd.Timedelta(minutes=5)
    seg = brk.cumsum().to_numpy()
    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    seg_start = np.empty(len(seg), dtype=np.int64)
    cur = 0
    for i in range(len(seg)):
        while cur + 1 < len(starts) and starts[cur + 1] <= i:
            cur += 1
        seg_start[i] = starts[cur]
    dt = ts.to_numpy()
    return ((dt - dt[seg_start]) / np.timedelta64(1, "m")).astype(np.int64)


def attach_trend_multi(df, sd):
    """as-of join：4h 环境 + 1h / 15m / 5m 结构趋势（全部 canonical）。

    只使用 available_time <= interaction_time 的已确认状态。
    """
    if len(df) == 0:
        return df
    if not hasattr(sd, "trend5m"):
        sd.trend5m = sd._trend(sd.five, "5m")
    if not hasattr(sd, "trend15m"):
        sd.trend15m = sd._trend(aggregate_15m(sd.five), "15m")
    has = df["interaction_time"].notna().to_numpy()
    o = df[has].sort_values("interaction_time")
    for frame in (sd.trend1h, sd.trend15m, sd.trend5m, sd.env):
        o = pd.merge_asof(o.sort_values("interaction_time"),
                          frame.sort_values("available_time"),
                          left_on="interaction_time",
                          right_on="available_time", direction="backward",
                          allow_exact_matches=True)
        o = o.drop(columns=["available_time"], errors="ignore")
    return pd.concat([o, df[~has]], ignore_index=True)


def build_liquidity_context(d: pd.DataFrame, lmap: pd.DataFrame
                            ) -> pd.DataFrame:
    """为每个事件构建 t0 时刻可见的多周期 liquidity context。

    严格：liquidity_available_time <= penetration_bar_start。
    同价位使用 exchange/tick 规范化后的严格数值相等（无 ATR 邻近阈值）。
    """
    recs = []
    for sym, g in d.groupby("symbol"):
        m = lmap[lmap["symbol"] == sym]
        order = np.argsort(m["available_time"].to_numpy())
        mp = m["price"].to_numpy(float)[order]
        ma = m["available_time"].to_numpy()[order]
        mtype = m["liquidity_type"].astype(str).to_numpy()[order]
        mscope = m["liquidity_scope"].astype(str).to_numpy()[order]
        mkey = m["level_key"].astype(str).to_numpy()[order]
        mside = m["side"].to_numpy(float)[order]

        t0s = pd.to_datetime(g["penetration_bar_start"]).to_numpy()
        cps = pd.to_numeric(g["price"], errors="coerce").to_numpy(float)
        csides = g["side"].to_numpy(float)
        ckeys = g["level_key"].astype(str).to_numpy()
        r0s = pd.to_numeric(g["R_at_t0"], errors="coerce").to_numpy(float)

        for t0, cp, cs, ck, r0 in zip(t0s, cps, csides, ckeys, r0s):
            k = int(np.searchsorted(ma, np.datetime64(t0), side="right"))
            vis_p, vis_type = mp[:k], mtype[:k]
            vis_scope, vis_key = mscope[:k], mkey[:k]
            vis_side = mside[:k]
            if len(vis_p) == 0:
                recs.append(dict(level_key=ck))
                continue

            # 只考虑同侧（同方向）liquidity 作为前后关系参照
            same_side = vis_side == cs
            # 1. 同价位重合（严格数值相等，含自身）
            same = np.isclose(vis_p, cp, rtol=0.0, atol=1e-12) & same_side
            n_same = int(same.sum())
            htf = any(x in ("15m", "1h", "TRADING_DAY", "TRADING_WEEK")
                      for x in vis_scope[same])

            # 2. 沿 sweep 方向下一个 liquidity（排除自身）
            dd = cs * (vis_p - cp)
            fwd = (dd > 0) & (vis_key != ck) & same_side
            if fwd.any():
                j = int(np.flatnonzero(fwd)[np.argmin(dd[fwd])])
                nx_d = float(dd[j] / r0) if np.isfinite(r0) and r0 > 0 \
                    else np.nan
                nx_t, nx_s = vis_type[j], vis_scope[j]
            else:
                nx_d, nx_t, nx_s = np.nan, None, None

            # 3. 反方向最近 liquidity
            bwd = (dd < 0) & same_side
            if bwd.any():
                jj = int(np.flatnonzero(bwd)[np.argmin(-dd[bwd])])
                op_d = float(-dd[jj] / r0) if np.isfinite(r0) and r0 > 0 \
                    else np.nan
                op_t, op_s = vis_type[jj], vis_scope[jj]
            else:
                op_d, op_t, op_s = np.nan, None, None

            recs.append(dict(
                level_key=ck,
                same_price_liquidity_count=n_same,
                same_price_liquidity_types="|".join(
                    sorted(set(vis_type[same]))),
                same_price_liquidity_tfs="|".join(
                    sorted(set(vis_scope[same]))),
                same_price_has_htf_liquidity=bool(htf),
                next_liquidity_distance_R=nx_d,
                next_liquidity_type=nx_t,
                next_liquidity_source_tf=nx_s,
                opposite_liquidity_distance_R=op_d,
                opposite_liquidity_type=op_t,
                opposite_liquidity_source_tf=op_s,
            ))
        print(f"  [liqctx] {sym} ({time.perf_counter():.0f}s)", flush=True)
    return pd.DataFrame(recs)


def main():
    ob = pd.read_parquet(
        "research/analysis_results/liquidity_ob_confluence_v3/"
        "ob_confluence_interactions.parquet")
    tr = pd.read_parquet(
        "research/analysis_results/liquidity_specificity_v2/"
        "true_interactions.parquet")

    # 注意：symbol / liquidity_type / side 在 ob 中已持有，
    # 再合并会产生 _x/_y 后缀，故此处排除。
    keep = ["level_key", "interaction_time", "trading_day", "R_at_t0",
            "level_age_log1p", "atr_rel_pre", "pre_ret_3_R", "pre_ret_12_R",
            "pre_rv_12_R", "pre_range_12_R", "volume_z_20_pre",
            "bar_range_R", "abs_return_R", "gap_R", "volume_z_t0",
            "session_type", "time_bucket_30m", "price", "available_time",
            "liquidity_scope"]
    d = ob.merge(tr[keep], on="level_key", how="left", validate="one_to_one")

    d = d[(d["stage1"] == "CLOSE_BEYOND")
          & (d["ob_geometry_relation"] == "OB_BEYOND_LIQUIDITY")].copy()

    # ---- T1/T2：margin 方向与 hit 一致 ----
    d["margin"] = pd.to_numeric(d["ob_margin_R"], errors="coerce")
    d["hit"] = d["has_opposing_ob_hit"].astype(bool)
    assert bool((~d.loc[d["margin"] < 0, "hit"]).all()), "margin<0 却 hit"
    assert bool(d.loc[d["margin"] >= 0, "hit"].all()), "margin>=0 却未 hit"
    print("[T1/T2] margin 方向与 hit 一致: OK")

    # ---- T3/T11：只含 CLOSE_BEYOND 且 interaction 唯一 ----
    assert (d["stage1"] == "CLOSE_BEYOND").all()
    assert d["level_key"].is_unique

    # ---- B. 当前被扫 liquidity 自身 ----
    d = d.rename(columns={"available_time": "liquidity_available_time"})
    d["liquidity_id"] = d["level_key"]
    d["liquidity_source_tf"] = d["liquidity_scope"]
    d["liquidity_side"] = pd.to_numeric(d["side"], errors="coerce")

    # ---- A. 多周期趋势 as-of join ----
    parts = []
    for sym, g in d.groupby("symbol"):
        sd = SymbolData(sym)
        a = attach_trend_multi(g, sd)
        mfs = session_minute(sd.t)
        a["minute_from_session_open"] = mfs[[int(x) for x in a["interaction_i"]]]
        parts.append(a)
        print(f"  [state] {sym} ({time.perf_counter():.0f}s)", flush=True)
    d = pd.concat(parts, ignore_index=True)
    for c, tag in (("1h", "sweep_vs_1h"), ("15m", "sweep_vs_15m"),
                   ("5m", "sweep_vs_5m")):
        col = f"trend_struct_{c}"
        d[tag] = ([rel(int(r.side), getattr(r, col, np.nan))
                   for r in d.itertuples()] if col in d.columns
                  else ["UNKNOWN"] * len(d))
    d["env4h_vs_1h"] = [rel(getattr(r, "env_direction_4h", np.nan),
                            getattr(r, "trend_struct_1h", np.nan))
                        for r in d.itertuples()]
    d["trend_1h_vs_15m"] = [rel(getattr(r, "trend_struct_1h", np.nan),
                                getattr(r, "trend_struct_15m", np.nan))
                            for r in d.itertuples()]
    d["trend_15m_vs_5m"] = [rel(getattr(r, "trend_struct_15m", np.nan),
                                getattr(r, "trend_struct_5m", np.nan))
                            for r in d.itertuples()]
    d["trend_tuple_4h_1h_15m_5m"] = [
        f"{_s(getattr(r, 'env_direction_4h', np.nan))}|"
        f"{_s(getattr(r, 'trend_struct_1h', np.nan))}|"
        f"{_s(getattr(r, 'trend_struct_15m', np.nan))}|"
        f"{_s(getattr(r, 'trend_struct_5m', np.nan))}"
        for r in d.itertuples()]

    # ---- C/D. liquidity context ----
    lmap = tr[["level_key", "symbol", "liquidity_type", "liquidity_scope",
               "side", "price", "available_time"]].copy()
    ctx = build_liquidity_context(d, lmap)
    d = d.merge(ctx, on="level_key", how="left", validate="one_to_one")

    # ---- F. 跨周期组合 ----
    d["liquidity_tf_x_ob_tf"] = (d["liquidity_source_tf"].astype(str)
                                 + "->" + d["nearest_ob_source_tf"]
                                 .astype(str))
    d["liquidity_overlap"] = np.where(
        pd.to_numeric(d["same_price_liquidity_count"], errors="coerce") >= 2,
        "MULTI_IDENTITY", "SINGLE")
    # 时序 liquidity 归并
    d["liquidity_source_tf_group"] = np.where(
        d["liquidity_source_tf"].isin(["5m", "15m", "1h"]),
        d["liquidity_source_tf"], "TIME")

    # ---- F1-F4 ----
    # build_folds 返回 4 个 (train, selection, test) 交易日集合，
    # 用各折的 test 段给事件打 F1-F4 标签。
    folds, _ = m2.build_folds(d["trading_day"].to_numpy(), len(d))
    td = d["trading_day"].to_numpy()
    fid = np.full(len(d), -1, dtype=int)
    for i, (_tr, _se, te) in enumerate(folds):
        fid[np.isin(td, list(te))] = i
    d["fold"] = [f"F{int(x)+1}" if x >= 0 else "NA" for x in fid]

    # ---- T5：Outcome Lock ----
    d = d.drop(columns=[c for c in d.columns if c in FORBIDDEN])
    assert not (FORBIDDEN & set(d.columns)), "outcome 泄漏"

    # ---- T9/T10：future / inactive OB 不进入 nearest OB（建构时保证） ----
    assert d["nearest_ob_id"].notna().all()

    d.to_parquet(RESULTS / "boundary_preoutcome.parquet", index=False)
    print(f"\nboundary preoutcome = {len(d)}  列数 = {len(d.columns)}")

    # ---- 边界密度审计 ----
    edges = np.arange(-0.50, 0.501, 0.05)
    dens = (d.groupby(pd.cut(d["margin"], edges, include_lowest=True),
                      observed=True).size().rename("n").reset_index())
    dens["share"] = (dens["n"] / dens["n"].sum()).round(5)
    dens.to_csv(RESULTS / "margin_density_audit.csv", index=False,
                encoding="utf-8-sig")
    zc = int((d["margin"] == 0).sum())
    meta = dict(exact_zero_count=zc,
                exact_zero_share=round(zc / len(d), 6),
                unique_margin_values_left=int(
                    d.loc[d["margin"] < 0, "margin"].nunique()),
                unique_margin_values_right=int(
                    d.loc[d["margin"] >= 0, "margin"].nunique()))
    print("\n=== 边界密度 ===")
    print(dens.to_string(index=False))
    near_l = int(d["margin"].between(-0.05, 0, inclusive="left").sum())
    near_r = int(d["margin"].between(0, 0.05, inclusive="right").sum())
    meta["n_left_005"] = near_l
    meta["n_right_005"] = near_r
    print(meta)
    print(f"[-0.05,0)={near_l}  [0,0.05]={near_r}")

    # ---- bandwidth 支持度 ----
    sup = []
    for h in BANDWIDTHS_R:
        g = d[d["margin"].abs() <= h]
        L = int((g["margin"] < 0).sum())
        R = int((g["margin"] >= 0).sum())
        sym_ok = sum(
            1 for _, gs in g.groupby("symbol")
            if (gs["margin"] < 0).sum() >= N_MIN_SYMBOL_SIDE
            and (gs["margin"] >= 0).sum() >= N_MIN_SYMBOL_SIDE)
        sup.append(dict(bandwidth=h, n=len(g), n_left=L, n_right=R,
                        n_symbol_ok=sym_ok,
                        support_ok=bool(len(g) >= N_MIN_TOTAL
                                        and L >= N_MIN_SIDE
                                        and R >= N_MIN_SIDE
                                        and sym_ok >= N_MIN_SYMBOL_OK)))
    S = pd.DataFrame(sup)
    S.to_csv(RESULTS / "bandwidth_support.csv", index=False,
             encoding="utf-8-sig")
    print("\n=== bandwidth 支持度 ===")
    print(S.to_string(index=False))

    prim = S[S["support_ok"]]
    if not len(prim):
        json.dump(dict(primary_bandwidth=None,
                       status="LOCAL_SUPPORT_FAIL"),
                  open(RESULTS / "primary_bandwidth_decision.json", "w"),
                  indent=2)
        print("\nLOCAL_SUPPORT_FAIL")
        return
    h = float(prim["bandwidth"].iloc[0])
    json.dump(dict(primary_bandwidth=h,
                   rule="从小到大第一个满足支持度 Gate 的 h",
                   n=int(prim["n"].iloc[0]),
                   n_left=int(prim["n_left"].iloc[0]),
                   n_right=int(prim["n_right"].iloc[0]), status="OK"),
              open(RESULTS / "primary_bandwidth_decision.json", "w"),
              indent=2)
    print(f"\nPRIMARY_BANDWIDTH = {h}")

    # ---- 连续性 Gate ----
    g = d[d["margin"].abs() <= h].copy()
    g["D"] = (g["margin"] >= 0).astype(int)
    L, R = g[g["D"] == 0], g[g["D"] == 1]

    rows = []
    for c in NUMERIC_AUDIT:
        if c not in g.columns or c in NO_BALANCE:
            continue
        a = pd.to_numeric(L[c], errors="coerce")
        b = pd.to_numeric(R[c], errors="coerce")
        if a.notna().sum() < 30 or b.notna().sum() < 30:
            continue
        rows.append(dict(variable=c, left_mean=round(float(a.mean()), 4),
                         right_mean=round(float(b.mean()), 4),
                         left_median=round(float(a.median()), 4),
                         right_median=round(float(b.median()), 4),
                         smd=round(smd(a, b), 4),
                         key=bool(c in KEY_NUMERIC)))
    NC = pd.DataFrame(rows)
    NC.to_csv(RESULTS / "numeric_continuity_audit.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== numeric 连续性 ===")
    print(NC.to_string(index=False))

    mx_all = float(NC["smd"].abs().max())
    mx_key = float(NC.loc[NC["key"], "smd"].abs().max())
    print(f"\nmax|SMD| all={mx_all:.4f}(<=0.15)  key={mx_key:.4f}(<=0.10)")

    crows = []
    for c in CATEG_AUDIT:
        if c not in g.columns:
            continue
        pl = L[c].astype(str).value_counts(normalize=True)
        pr = R[c].astype(str).value_counts(normalize=True)
        for k in sorted(set(pl.index) | set(pr.index)):
            crows.append(dict(variable=c, level=k,
                              left_share=round(float(pl.get(k, 0)), 4),
                              right_share=round(float(pr.get(k, 0)), 4),
                              abs_diff_pp=round(
                                  abs(float(pl.get(k, 0))
                                      - float(pr.get(k, 0))) * 100, 2)))
    CC = pd.DataFrame(crows)
    CC.to_csv(RESULTS / "categorical_continuity_audit.csv", index=False,
              encoding="utf-8-sig")
    mx_cat = float(CC["abs_diff_pp"].max())
    print(f"max categorical diff = {mx_cat:.2f} pp (<= {CAT_CANON_MAX_PP})")
    print(CC.sort_values("abs_diff_pp", ascending=False).head(8)
          .to_string(index=False))

    gate = dict(primary_bandwidth=h, max_smd_all=round(mx_all, 4),
                max_smd_key=round(mx_key, 4),
                max_categorical_pp=round(mx_cat, 2),
                gate_numeric=bool(mx_all <= 0.15 and mx_key <= 0.10),
                gate_categorical=bool(mx_cat <= CAT_CANON_MAX_PP), **meta)
    gate["gate_pass"] = bool(gate["gate_numeric"] and gate["gate_categorical"])
    json.dump(gate, open(RESULTS / "AUDIT_V3_1.json", "w"), indent=2)
    print(f"\nGATE = {'PASS' if gate['gate_pass'] else 'FAIL'}")
    print("PREOUTCOME_DONE")


def _s(v):
    try:
        f = float(v)
        return "?" if not np.isfinite(f) else str(int(f))
    except (TypeError, ValueError):
        return "?"


if __name__ == "__main__":
    main()
