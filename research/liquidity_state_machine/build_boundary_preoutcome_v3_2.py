"""订单块边界实验 v3.2 —— 阶段一：语义修复版 preoutcome。

修复 v3.1 的两个实质语义问题：
  1. 多周期趋势未真正接入（_trend 恒返回 trend_struct_1h，
     导致 trend_struct_5m/15m 全 UNKNOWN，1h×15m 有效样本为 0）
     -> 新增 build_trend_series(bars, tf)，直接从 canonical
        state_timeline 的 swing_bias / internal_bias 构建
  2. ob_margin_R == 0 被错误归入"进入 OB"
     -> 改为价格空间三态：NEAR_MISS / EDGE_TOUCH / ENTERED_OB
        EDGE_TOUCH 不进左侧、不进右侧、不进局部模型

Outcome Lock：本文件不产生任何 outcome 字段。
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
from research.ob_trigger_snapshot import (aggregate_1h_from_15m,
                                          build_full_ob_smc_tf)

RESULTS = Path("research/analysis_results/liquidity_ob_boundary_v3_2")
RESULTS.mkdir(parents=True, exist_ok=True)

BANDWIDTHS_R = [0.10, 0.20, 0.30, 0.50]
N_MIN_TOTAL, N_MIN_SIDE = 2000, 750
N_MIN_SYMBOL_SIDE, N_MIN_SYMBOL_OK = 20, 10

FORBIDDEN = {"later_reclaim", "stage2", "stage2_state",
             "structural_acceptance"}
NO_BALANCE = {"nearest_ob_distance_R", "ob_margin_R"}

TREND_COLS = ["trend_struct_5m", "trend_struct_15m", "trend_struct_1h"]

NUMERIC_AUDIT = ["penetration_depth_R", "close_beyond_R", "bar_range_R",
                 "abs_return_R", "atr_rel_pre", "level_age_log1p",
                 "pre_ret_3_R", "pre_ret_12_R", "pre_rv_12_R",
                 "pre_range_12_R", "volume_z_20_pre", "volume_z_t0",
                 "minute_from_session_open", "next_liquidity_distance_R",
                 "opposite_liquidity_distance_R", "nearest_ob_width_R"]
KEY_NUMERIC = ["penetration_depth_R", "close_beyond_R", "pre_range_12_R"]

CATEG_AUDIT = ["symbol", "liquidity_type", "liquidity_source_tf",
               "same_price_has_htf_liquidity", "nearest_ob_source_tf",
               "nearest_ob_freshness", "trend_struct_5m",
               "trend_struct_15m", "trend_struct_1h", "env_direction_4h",
               "sweep_vs_5m", "sweep_vs_15m", "sweep_vs_1h",
               "env4h_vs_1h", "trend_1h_vs_15m", "trend_15m_vs_5m"]

CTX_COLS = ["same_price_liquidity_count", "same_price_liquidity_types",
            "same_price_liquidity_tfs", "same_price_has_htf_liquidity",
            "next_liquidity_distance_R", "next_liquidity_type",
            "next_liquidity_source_tf", "opposite_liquidity_distance_R",
            "opposite_liquidity_type", "opposite_liquidity_source_tf"]


def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.nanmean(a) - np.nanmean(b)
    p = np.sqrt(.5 * (np.nanvar(a, ddof=1) + np.nanvar(b, ddof=1)))
    return 0.0 if (not np.isfinite(p) or p < 1e-12) else float(m / p)


def session_minute(t) -> np.ndarray:
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


# ---------------- 修复 1：多周期趋势 ----------------
def build_trend_series(bars, tf):
    """用 canonical SMC state_timeline 为指定周期生成 causal 结构趋势。

    直接取 state_timeline 的 swing_bias / internal_bias，
    不依赖任何会恒返回 trend_struct_1h 的旧封装。
    """
    smc = build_full_ob_smc_tf(bars.copy())
    st = pd.DataFrame(smc["state_timeline"])
    bar_time = pd.to_datetime(bars["bar_start_time"])
    per = {"5m": pd.Timedelta(minutes=5),
           "15m": pd.Timedelta(minutes=15),
           "1h": pd.Timedelta(hours=1)}[tf]
    idx = st["bar_index"].to_numpy(int)
    return pd.DataFrame({
        "available_time": (bar_time.iloc[idx] + per).to_numpy(),
        f"trend_struct_{tf}": st["swing_bias"].to_numpy(int),
        f"internal_bias_{tf}": st["internal_bias"].to_numpy(int),
    })


def attach_trend_multi(df, sd):
    """as-of join 5m / 15m / 1h 结构趋势 + 4h 环境方向。"""
    if len(df) == 0:
        return df
    fif = aggregate_15m(sd.five)
    oneh = aggregate_1h_from_15m(fif)
    frames = [("5m", build_trend_series(sd.five, "5m")),
              ("15m", build_trend_series(fif, "15m")),
              ("1h", build_trend_series(oneh, "1h")),
              ("env", sd.env)]
    has = df["interaction_time"].notna().to_numpy()
    o = df[has].sort_values("interaction_time")
    for tag, fr in frames:
        o = pd.merge_asof(o.sort_values("interaction_time"),
                          fr.sort_values("available_time"),
                          left_on="interaction_time",
                          right_on="available_time", direction="backward",
                          allow_exact_matches=True)
        o = o.rename(columns={"available_time": f"trend_avail_{tag}"})
    return pd.concat([o, df[~has]], ignore_index=True)


# ---------------- 修复 2：边界三态 ----------------
def boundary_state(side, penetration_extreme, ob_near_edge, r0):
    """价格空间三态：NEAR_MISS / EDGE_TOUCH / ENTERED_OB。

    price_delta == 0 使用原始价格值的严格边界相等，
    不引入任何 ATR touch threshold。
    """
    if side == +1:
        delta = penetration_extreme - ob_near_edge
    else:
        delta = ob_near_edge - penetration_extreme
    if r0 is None or not np.isfinite(r0) or r0 <= 0:
        margin = np.nan
    else:
        margin = delta / r0
    if delta < 0:
        st = "NEAR_MISS"
    elif delta == 0:
        st = "EDGE_TOUCH"
    else:
        st = "ENTERED_OB"
    return st, float(margin) if margin == margin else np.nan


def main():
    ob = pd.read_parquet(
        "research/analysis_results/liquidity_ob_confluence_v3/"
        "ob_confluence_interactions.parquet")
    tr = pd.read_parquet(
        "research/analysis_results/liquidity_specificity_v2/"
        "true_interactions.parquet")
    # 复用 v3.1 已构建的 liquidity context（语义未变，避免重算）
    v31 = pd.read_parquet(
        "research/analysis_results/liquidity_ob_boundary_v3_1/"
        "boundary_preoutcome.parquet")

    d = ob.merge(v31[["level_key"] + CTX_COLS], on="level_key", how="left",
                 validate="one_to_one")
    keep = ["level_key", "interaction_time", "trading_day", "R_at_t0",
            "level_age_log1p", "atr_rel_pre", "pre_ret_3_R", "pre_ret_12_R",
            "pre_rv_12_R", "pre_range_12_R", "volume_z_20_pre",
            "bar_range_R", "abs_return_R", "gap_R", "volume_z_t0",
            "session_type", "time_bucket_30m", "price", "available_time",
            "liquidity_scope"]
    d = d.merge(tr[keep], on="level_key", how="left", validate="one_to_one")

    d = d[(d["stage1"] == "CLOSE_BEYOND")
          & (d["ob_geometry_relation"] == "OB_BEYOND_LIQUIDITY")].copy()
    d = d.rename(columns={"available_time": "liquidity_available_time"})
    d["liquidity_id"] = d["level_key"]
    d["liquidity_source_tf"] = d["liquidity_scope"]
    d["liquidity_side"] = pd.to_numeric(d["side"], errors="coerce")

    # ---- 趋势 + 极值 + session ----
    parts = []
    for sym, g in d.groupby("symbol"):
        sd = SymbolData(sym)
        a = attach_trend_multi(g, sd)
        ii = [int(x) for x in a["interaction_i"]]
        sd_side = a["side"].to_numpy(int)
        hi, lo = sd.hi[ii], sd.lo[ii]
        a["penetration_extreme"] = np.where(sd_side == +1, hi, lo)
        mfs = session_minute(sd.t)
        a["minute_from_session_open"] = mfs[ii]
        parts.append(a)
        print(f"  [state] {sym} ({time.perf_counter():.0f}s)", flush=True)
    d = pd.concat(parts, ignore_index=True)

    # ---- 三态 ----
    st_m = [boundary_state(int(r.side), float(r.penetration_extreme),
                           float(r.nearest_ob_near_edge), float(r.R_at_t0))
            for r in d.itertuples()]
    d["ob_boundary_state"] = [s for s, _ in st_m]
    d["ob_margin_R"] = [m for _, m in st_m]

    # 三态硬断言（R0 无效时 margin 为 NaN，不参与符号断言）
    n_nan = int(d["ob_margin_R"].isna().sum())
    for state, cond in (("NEAR_MISS", lambda v: v < 0),
                        ("EDGE_TOUCH", lambda v: v == 0),
                        ("ENTERED_OB", lambda v: v > 0)):
        sub = d.loc[d["ob_boundary_state"] == state,
                    "ob_margin_R"].dropna()
        assert bool(cond(sub).all()), f"{state} 的 ob_margin_R 符号不一致"
    print(f"[三态] 符号断言通过（margin NaN {n_nan} 条不参与）")

    # ---- 派生关系 ----
    for c, tag in (("1h", "sweep_vs_1h"), ("15m", "sweep_vs_15m"),
                   ("5m", "sweep_vs_5m")):
        d[tag] = [rel(int(r.side), getattr(r, f"trend_struct_{c}", np.nan))
                  for r in d.itertuples()]
    d["env4h_vs_1h"] = [rel(getattr(r, "env_direction_4h", np.nan),
                            getattr(r, "trend_struct_1h", np.nan))
                        for r in d.itertuples()]
    d["trend_1h_vs_15m"] = [rel(getattr(r, "trend_struct_1h", np.nan),
                                getattr(r, "trend_struct_15m", np.nan))
                            for r in d.itertuples()]
    d["trend_15m_vs_5m"] = [rel(getattr(r, "trend_struct_15m", np.nan),
                                getattr(r, "trend_struct_5m", np.nan))
                            for r in d.itertuples()]
    d["liquidity_overlap"] = np.where(
        pd.to_numeric(d["same_price_liquidity_count"], errors="coerce") >= 2,
        "MULTI_IDENTITY", "SINGLE")
    d["liquidity_source_tf_group"] = np.where(
        d["liquidity_source_tf"].isin(["5m", "15m", "1h"]),
        d["liquidity_source_tf"], "TIME")
    d["liquidity_tf_x_ob_tf"] = (d["liquidity_source_tf"].astype(str) + "->"
                                 + d["nearest_ob_source_tf"].astype(str))

    # ---- 趋势硬审计 ----
    trows = []
    for c in TREND_COLS:
        assert c in d.columns, f"{c} 缺失"
        v = pd.to_numeric(d[c], errors="coerce")
        valid = v.isin([-1, 0, 1])
        share = float(valid.mean())
        assert share >= 0.95, f"{c} 有效率 {share:.3f} < 0.95"
        trows.append(dict(field=c, valid_share=round(share, 4),
                          n_plus=int((v == 1).sum()), n_zero=int((v == 0).sum()),
                          n_minus=int((v == -1).sum()),
                          n_null=int(v.isna().sum())))
    # 因果抽查：随机 100 个事件确认 trend available_time <= penetration_bar_start
    rng = np.random.default_rng(7)
    pb = pd.to_datetime(d["penetration_bar_start"])
    for tag in ("5m", "15m", "1h"):
        col = f"trend_avail_{tag}"
        if col not in d.columns:
            continue
        s = pd.to_datetime(d[col])
        idx = rng.choice(len(d), size=min(100, len(d)), replace=False)
        ok = bool((s.iloc[idx] <= pb.iloc[idx]).all())
        trows.append(dict(field=f"causal_check_{tag}", valid_share=1.0 if ok else 0.0,
                          n_plus=int(ok), n_zero=0, n_minus=0, n_null=0))
        assert ok, f"{tag} 趋势 available_time 晚于 penetration_bar_start"
    TA = pd.DataFrame(trows)
    TA.to_csv(RESULTS / "trend_context_audit.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== 趋势上下文审计 ===")
    print(TA.to_string(index=False))

    # ---- 1h×15m 必须有有效样本 ----
    rel_rows = []
    for c in ("trend_1h_vs_15m", "trend_15m_vs_5m", "env4h_vs_1h",
              "sweep_vs_1h", "sweep_vs_15m", "sweep_vs_5m"):
        vc = d[c].astype(str).value_counts()
        for k in ("WITH_TREND", "AGAINST_TREND", "UNKNOWN"):
            rel_rows.append(dict(relation=c, value=k,
                                 n=int(vc.get(k, 0))))
    RC = pd.DataFrame(rel_rows)
    RC.to_csv(RESULTS / "trend_relation_counts.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== 趋势关系计数 ===")
    print(RC.pivot_table(index="relation", columns="value", values="n",
                         aggfunc="first").to_string())
    eff = RC[(RC.relation == "trend_1h_vs_15m")
             & (RC.value.isin(["WITH_TREND", "AGAINST_TREND"]))]["n"].sum()
    if eff == 0:
        json.dump(dict(status="TREND_CONTEXT_BUILD_FAIL"),
                  open(RESULTS / "AUDIT_V3_2.json", "w"), indent=2)
        print("\nTREND_CONTEXT_BUILD_FAIL —— 立即停止")
        return
    print(f"1h×15m 有效样本 = {eff}")

    # ---- 三态计数 ----
    bs = (d["ob_boundary_state"].value_counts().rename("n").reset_index())
    bs.columns = ["state", "n"]
    bs["share"] = (bs["n"] / bs["n"].sum()).round(4)
    bs.to_csv(RESULTS / "boundary_state_counts.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== 边界三态 ===")
    print(bs.to_string(index=False))

    # ---- F1-F4 ----
    folds, _ = m2.build_folds(d["trading_day"].to_numpy(), len(d))
    td = d["trading_day"].to_numpy()
    fid = np.full(len(d), -1, dtype=int)
    for i, (_a, _b, te) in enumerate(folds):
        fid[np.isin(td, list(te))] = i
    d["fold"] = [f"F{int(x)+1}" if x >= 0 else "NA" for x in fid]

    d = d.drop(columns=[c for c in d.columns if c in FORBIDDEN])
    assert not (FORBIDDEN & set(d.columns))
    d.to_parquet(RESULTS / "boundary_preoutcome_v3_2.parquet", index=False)
    print(f"\npreoutcome v3.2 = {len(d)}")

    # ---- 支持度（只算 NEAR_MISS + ENTERED_OB） ----
    sup = []
    for h in BANDWIDTHS_R:
        g = d[(d["ob_margin_R"].abs() <= h)
              & (d["ob_boundary_state"] != "EDGE_TOUCH")]
        L = int((g["ob_boundary_state"] == "NEAR_MISS").sum())
        R = int((g["ob_boundary_state"] == "ENTERED_OB").sum())
        nz = int(((d["ob_margin_R"].abs() <= h)
                  & (d["ob_boundary_state"] == "EDGE_TOUCH")).sum())
        sym_ok = sum(1 for _, gs in g.groupby("symbol")
                     if (gs["ob_boundary_state"] == "NEAR_MISS").sum()
                     >= N_MIN_SYMBOL_SIDE
                     and (gs["ob_boundary_state"] == "ENTERED_OB").sum()
                     >= N_MIN_SYMBOL_SIDE)
        sup.append(dict(bandwidth=h, n=len(g), n_left=L, n_right=R,
                        n_edge_touch=nz, n_symbol_ok=sym_ok,
                        support_ok=bool(len(g) >= N_MIN_TOTAL
                                        and L >= N_MIN_SIDE
                                        and R >= N_MIN_SIDE
                                        and sym_ok >= N_MIN_SYMBOL_OK)))
    S = pd.DataFrame(sup)
    S.to_csv(RESULTS / "bandwidth_support.csv", index=False,
             encoding="utf-8-sig")
    print("\n=== bandwidth 支持度（排除 EDGE_TOUCH） ===")
    print(S.to_string(index=False))

    prim = S[S["support_ok"]]
    if not len(prim):
        json.dump(dict(primary_bandwidth=None, status="LOCAL_SUPPORT_FAIL"),
                  open(RESULTS / "primary_bandwidth_decision.json", "w"),
                  indent=2)
        print("\nLOCAL_SUPPORT_FAIL")
        return
    h = float(prim["bandwidth"].iloc[0])
    json.dump(dict(primary_bandwidth=h,
                   rule="从小到大第一个满足支持度 Gate 的 h（排除 EDGE_TOUCH）",
                   n=int(prim["n"].iloc[0]), n_left=int(prim["n_left"].iloc[0]),
                   n_right=int(prim["n_right"].iloc[0]),
                   n_edge_touch=int(prim["n_edge_touch"].iloc[0]),
                   status="OK"),
              open(RESULTS / "primary_bandwidth_decision.json", "w"),
              indent=2)
    print(f"\nPRIMARY_BANDWIDTH = {h}")

    # ---- 连续性 Gate ----
    g = d[(d["ob_margin_R"].abs() <= h)
          & (d["ob_boundary_state"] != "EDGE_TOUCH")].copy()
    L = g[g["ob_boundary_state"] == "NEAR_MISS"]
    R = g[g["ob_boundary_state"] == "ENTERED_OB"]

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
    print(f"max categorical diff = {mx_cat:.2f} pp (<=8.00)")
    print(CC.sort_values("abs_diff_pp", ascending=False).head(10)
          .to_string(index=False))

    gate = dict(primary_bandwidth=h, max_smd_all=round(mx_all, 4),
                max_smd_key=round(mx_key, 4),
                max_categorical_pp=round(mx_cat, 2),
                gate_numeric=bool(mx_all <= 0.15 and mx_key <= 0.10),
                gate_categorical=bool(mx_cat <= 8.0),
                n_left=int(len(L)), n_right=int(len(R)))
    gate["gate_pass"] = bool(gate["gate_numeric"] and gate["gate_categorical"])
    json.dump(gate, open(RESULTS / "AUDIT_V3_2.json", "w"), indent=2)
    print(f"\nGATE = {'PASS' if gate['gate_pass'] else 'FAIL'}")
    print("PREOUTCOME_V3_2_DONE")


if __name__ == "__main__":
    main()
