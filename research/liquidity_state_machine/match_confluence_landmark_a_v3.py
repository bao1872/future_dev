"""Landmark A：条件预测价值匹配实验。

问题：在价格已经完成相似程度的 liquidity sweep 后，如果这个 sweep 恰好
触及一个事前存在的反向 OB，是否比相似 sweep 但尚未触及 OB，更容易随后
发生 reclaim？

**不是因果效应声明**，是条件预测价值比较。

OUTCOME LOCK：coverage / balance gate 全部通过之前，禁止计算任何
later_reclaim / BOS acceptance。
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from research.liquidity_state_machine.liquidity_specificity_placebo_v2 import (
    MATCH_COLS_NUM, RESULTS as R2, SymbolData, session_coords,
)
from research.liquidity_state_machine.match_liquidity_placebo_v2 import (
    attach_trend, rel, resolve_stage2,
)

RESULTS = "research/analysis_results/liquidity_ob_confluence_v3"
K = 3
DAY_WINDOW = 120
CB_CALIPER = 0.50
SESS_CALIPER_MIN = 60
CB_IDX = MATCH_COLS_NUM.index("close_beyond_R")

# 明确排除：这两个变量是本实验研究的结构差异，不得进入匹配
FORBIDDEN = ("nearest_ob_distance_R", "ob_margin_R")


def robust_params(frame):
    med = frame[MATCH_COLS_NUM].median()
    s = (frame[MATCH_COLS_NUM].quantile(.75)
         - frame[MATCH_COLS_NUM].quantile(.25)).clip(lower=1e-8)
    return med.to_numpy(float), s.to_numpy(float)


def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.nanmean(a) - np.nanmean(b)
    p = np.sqrt(.5 * (np.nanvar(a, ddof=1) + np.nanvar(b, ddof=1)))
    return 0.0 if (not np.isfinite(p) or p < 1e-12) else float(m / p)


def main():
    ob = pd.read_parquet(f"{RESULTS}/ob_confluence_interactions.parquet")
    tr = pd.read_parquet(f"{R2}/true_interactions.parquet")

    # 合并数值协变量（ob 只有几何与身份）
    # 注意：penetration_depth_R / close_beyond_R 在 ob 中已持有，
    # 再合并会产生 _x/_y 后缀，故此处排除。
    keep = ["level_key", "interaction_time", "trading_day", "R_at_t0",
            "level_age_log1p", "atr_rel_pre", "pre_ret_3_R", "pre_ret_12_R",
            "pre_rv_12_R", "pre_range_12_R", "volume_z_20_pre",
            "bar_range_R", "abs_return_R", "gap_R", "volume_z_t0",
            "session_type", "time_bucket_30m"]
    d = ob.merge(tr[keep], on="level_key", how="left", validate="one_to_one")

    # Landmark A universe
    d = d[(d["stage1"] == "CLOSE_BEYOND")
          & (d["ob_geometry_relation"] == "OB_BEYOND_LIQUIDITY")].copy()
    d = d[d[MATCH_COLS_NUM].notna().all(axis=1)].copy()
    print(f"Landmark A universe = {len(d)}")

    # trend + session position（逐品种）
    parts = []
    for sym, g in d.groupby("symbol"):
        sd = SymbolData(sym)
        a = attach_trend(g, sd)
        st, tb = session_coords(sd.t)
        a["minute_from_session_open"] = tb[np.asarray(
            [int(x) for x in a["interaction_i"]])] if False else np.nan
        parts.append(a)
        print(f"  [trend] {sym} ({time.perf_counter():.0f}s)", flush=True)
    d = pd.concat(parts, ignore_index=True)

    # minute_from_session_open：逐品种重算（session_coords 返回 bucket，
    # 这里用 30min bucket 的中点近似会导致 caliper 失真，故按品种重算真实值）
    mins = {}
    for sym in d["symbol"].unique():
        sd = SymbolData(sym)
        ts = pd.to_datetime(pd.Series(sd.t))
        brk = ts.diff() > pd.Timedelta(minutes=5)
        seg = brk.cumsum().to_numpy()
        starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
        seg_start_idx = np.empty(len(seg), dtype=np.int64)
        cur = 0
        for i in range(len(seg)):
            while cur + 1 < len(starts) and starts[cur + 1] <= i:
                cur += 1
            seg_start_idx[i] = starts[cur]
        dt = ts.to_numpy()
        mfs = (dt - dt[seg_start_idx]) / np.timedelta64(1, "m")
        mins[sym] = mfs.astype(np.int64)
    d["minute_from_session_open"] = [
        mins[r.symbol][int(r.interaction_i)] for r in d.itertuples()
    ]

    d["sweep_vs_1h"] = [
        rel(int(r.side), r.trend_struct_1h)
        for r in d.itertuples()]

    treat = d[d["has_opposing_ob_hit"]].copy()
    ctl = d[~d["has_opposing_ob_hit"]].copy()
    print(f"Treatment = {len(treat)}   Control = {len(ctl)}")

    # 断言：结构变量不得进入匹配
    assert not any(f in MATCH_COLS_NUM for f in FORBIDDEN)

    num_cols = list(MATCH_COLS_NUM) + ["minute_from_session_open"]
    med, sc = robust_params(pd.concat([treat[MATCH_COLS_NUM],
                                       ctl[MATCH_COLS_NUM]],
                                      ignore_index=True))
    # close_beyond caliper 用 MATCH_COLS_NUM 的 scale；session 用分钟原值
    med_f = np.append(med, 0.0)
    sc_f = np.append(sc, 1.0)

    uday = {x: i for i, x in enumerate(
        sorted(set(d["trading_day"].dropna().tolist())))}
    for x in (treat, ctl):
        x["day_ord"] = [uday.get(y, -1) for y in x["trading_day"]]

    # exact buckets
    buckets = {}
    for key, g in ctl.groupby(["symbol", "liquidity_type", "side",
                               "session_type", "sweep_vs_1h"]):
        g = g.copy()
        X = np.column_stack([
            (g[MATCH_COLS_NUM].to_numpy(float) - med) / sc,
            g["minute_from_session_open"].to_numpy(float)])
        buckets[key] = (X, g["day_ord"].to_numpy(),
                        g["level_key"].to_numpy())

    pairs, unmatched = [], 0
    for r in treat.itertuples(index=False):
        key = (r.symbol, r.liquidity_type, r.side, r.session_type,
               r.sweep_vs_1h)
        if key not in buckets:
            unmatched += 1
            continue
        Xc, dord, cid = buckets[key]
        xt = np.append(
            (pd.Series(r._asdict())[MATCH_COLS_NUM].to_numpy(float)
             - med) / sc,
            float(r.minute_from_session_open))
        m = ((dord != r.day_ord)
             & (np.abs(dord - r.day_ord) <= DAY_WINDOW)
             & (np.abs(Xc[:, CB_IDX] - xt[CB_IDX]) <= CB_CALIPER)
             & (np.abs(Xc[:, -1] - xt[-1]) <= SESS_CALIPER_MIN))
        if m.sum() < K:
            unmatched += 1
            continue
        Xs = Xc[m]
        dist = np.sqrt(np.sum((Xs - xt[None, :]) ** 2, axis=1))
        take = np.argpartition(dist, K - 1)[:K]
        take = take[np.argsort(dist[take])]
        for t_ in take:
            pairs.append(dict(treat_key=r.level_key,
                              control_key=str(cid[m][t_]),
                              distance=float(dist[t_])))
    P = pd.DataFrame(pairs)
    if len(P) == 0:
        print("LANDMARK_A_MATCHING_FAIL: no pairs")
        return
    P.to_parquet(f"{RESULTS}/landmark_a_pairs.parquet", index=False)

    rate = P["treat_key"].nunique() / len(treat)
    print(f"\ncoverage = {rate:.4f} "
          f"({P['treat_key'].nunique()}/{len(treat)})  "
          f"unique controls = {P['control_key'].nunique()}")

    cov = [dict(维度="ALL", n=len(treat),
                matched=P["treat_key"].nunique(),
                coverage=round(rate, 4))]
    for c, name in (("liquidity_type", "type"), ("symbol", "symbol")):
        for k, g in treat.groupby(c):
            if len(g) < 500:
                continue
            mm = P["treat_key"].nunique() and \
                P[P["treat_key"].isin(set(g["level_key"]))]
            cov.append(dict(维度=f"{name}={k}", n=len(g),
                            matched=int(mm["treat_key"].nunique()),
                            coverage=round(
                                mm["treat_key"].nunique() / len(g), 4)))
    covd = pd.DataFrame(cov)
    covd.to_csv(f"{RESULTS}/landmark_a_matching_coverage.csv", index=False,
                encoding="utf-8-sig")
    print(covd.to_string(index=False))

    gate_cov = bool(rate >= 0.80) and bool(
        (covd[covd["维度"] != "ALL"]["coverage"] >= 0.70).all())

    # balance
    pr = P.merge(d, left_on="treat_key", right_on="level_key",
                 how="left", suffixes=("", "_t"))
    pr = pr.merge(d, left_on="control_key", right_on="level_key",
                  how="left", suffixes=("_t", "_c"))
    bal = []
    for c in MATCH_COLS_NUM + ["minute_from_session_open"]:
        bal.append(dict(variable=c, dimension="ALL",
                        smd=round(smd(pr[f"{c}_t"], pr[f"{c}_c"]), 4)))
    for c in ("liquidity_type", "symbol"):
        for k, g in pr.groupby(f"{c}_t"):
            for cc in MATCH_COLS_NUM + ["minute_from_session_open"]:
                bal.append(dict(variable=cc, dimension=f"{k}",
                                smd=round(smd(g[f"{cc}_t"],
                                              g[f"{cc}_c"]), 4)))
    bd = pd.DataFrame(bal)
    bd.to_csv(f"{RESULTS}/landmark_a_matching_balance.csv", index=False,
              encoding="utf-8-sig")
    mx_all = bd[bd.dimension == "ALL"]["smd"].abs().max()
    mx_sub = (bd[bd.dimension != "ALL"].groupby("dimension")["smd"]
              .apply(lambda s: s.abs().max()))
    print(f"\nmax |SMD| overall = {mx_all:.4f}")
    print(f"max |SMD| by group = {mx_sub.max():.4f}")
    gate_bal = bool(mx_all <= 0.10) and bool((mx_sub <= 0.15).all())

    rc = (P.groupby("control_key").size().rename("times").reset_index())
    rc.groupby("times").size().rename("n_controls").reset_index().to_csv(
        f"{RESULTS}/landmark_a_control_reuse.csv", index=False,
        encoding="utf-8-sig")
    print(f"control reuse max = {int(rc['times'].max())}")

    with open(f"{RESULTS}/landmark_a_gate.json", "w", encoding="utf-8") as f:
        json.dump(dict(coverage=round(rate, 4), gate_coverage=bool(gate_cov),
                       max_smd_all=round(float(mx_all), 4),
                       max_smd_group=round(float(mx_sub.max()), 4),
                       gate_balance=bool(gate_bal)), f, indent=2)
    print(f"\nGATE coverage={gate_cov} balance={gate_bal}")
    if not (gate_cov and gate_bal):
        print("LANDMARK_A_MATCHING_FAIL")
        return

    # ---------- 解锁 outcome ----------
    print("\nGATE PASS —— 解锁 future outcome")
    sd_map = {}
    res = {}
    for r in P.itertuples(index=False):
        sym = r.treat_key.split("|")[0]
        if sym not in sd_map:
            sd_map[sym] = SymbolData(sym)
    # 批量解析
    tinfo = d.set_index("level_key")
    out_t, out_c = [], []
    for r in P.itertuples(index=False):
        sym = r.treat_key.split("|")[0]
        sd = sd_map[sym]
        ti = int(tinfo.loc[r.treat_key, "interaction_i"])
        side = int(tinfo.loc[r.treat_key, "side"])
        lvl = float(tinfo.loc[r.treat_key, "price"])
        s, _ = resolve_stage2(sd, ti, side, lvl)
        out_t.append(int(s == "LATER_RECLAIM"))
        ci = int(tinfo.loc[r.control_key, "interaction_i"])
        cside = int(tinfo.loc[r.control_key, "side"])
        clvl = float(tinfo.loc[r.control_key, "price"])
        cs, _ = resolve_stage2(sd, ci, cside, clvl)
        out_c.append(int(cs == "LATER_RECLAIM"))
    P["t_reclaim"] = out_t
    P["c_reclaim"] = out_c
    P.to_parquet(f"{RESULTS}/landmark_a_pairs.parquet", index=False)
    print(f"OB HIT  later reclaim = {np.mean(out_t):.4f}")
    print(f"OB MISS later reclaim = {np.mean(out_c):.4f}")
    print("LANDMARK_A_DONE")


if __name__ == "__main__":
    main()
