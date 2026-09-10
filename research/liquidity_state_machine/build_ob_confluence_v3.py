"""多周期流动性 × 订单块共振实验 v3.0 —— 第一步：订单块几何补全。

本步骤**不看任何 future outcome**。只做：
  1. 修对照组定义（Primary control = 外侧有反向 OB 但没扫到）
  2. 所有 OBBEYOND 事件都保存"最近的反向 active OB"（hit 与 miss 都保存）
  3. 建立连续边界变量 ob_margin_R
  4. 建立进入方式 5 分类

订单块完全复用 canonical：
    build_full_ob_smc_tf / replay_ob_lifetimes / spatial_ob_bounds
来源周期仅 5m / 15m / 1h（仓库无合法 canonical 4h OB，禁止伪造）。

硬因果边界：
    ob_available_time <= penetration_bar_start
    ob 在 t0 仍 active（inactive_time 为空或 > t0）
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.build_ob_candidate_universe_v3 import (
    _ob_identity, replay_ob_lifetimes,
)
from research.build_pytdx_panel import aggregate_15m
from research.export_ob_trigger_execution_v21 import load_raw_5m
from research.ob_trigger_snapshot import (aggregate_1h_from_15m,
                                          build_full_ob_smc_tf,
                                          spatial_ob_bounds)

RESULTS = Path("research/analysis_results/liquidity_ob_confluence_v3")
RESULTS.mkdir(parents=True, exist_ok=True)
TF_PERIOD = {"5m": pd.Timedelta(minutes=5), "15m": pd.Timedelta(minutes=15),
             "1h": pd.Timedelta(hours=1)}

# 统一极小数值容忍（不使用 ATR 阈值）
TOL = 1e-9


def build_ob_map(sym: str) -> pd.DataFrame:
    """返回每个 (tf, OB) 的生命周期与空间边界（全部来自 canonical）。"""
    five = load_raw_5m(sym).sort_values("bar_start_time").reset_index(
        drop=True)
    five["volume"] = five["trade"].astype(float)
    fifteen = aggregate_15m(five)
    oneh = aggregate_1h_from_15m(fifteen)

    rows = []
    for tf, bars in (("5m", five), ("15m", fifteen), ("1h", oneh)):
        per = TF_PERIOD[tf]
        t = pd.to_datetime(bars["bar_start_time"])
        smc = build_full_ob_smc_tf(bars.copy())
        lives = replay_ob_lifetimes(smc, len(bars))

        life_by_key = {}
        for lv in lives:
            key = lv["key"]
            ci = int(lv["confirmed_index"])
            ii = lv.get("inactive_index")
            life_by_key[key] = dict(
                available_time=(t.iloc[ci] + per if ci < len(t) else None),
                inactive_time=(t.iloc[int(ii)] + per
                               if (ii is not None and int(ii) < len(t))
                               else None),
                inactive_reason=lv.get("inactive_reason"),
            )

        # OB_ENTERED 次数（用于 freshness，只统计 t0 前的）
        enter_cnt = {}
        for e in smc.get("ob_lifecycle_events", []):
            if e.get("type") != "OB_ENTERED":
                continue
            k = _ob_identity(e)
            ci = int(e["enter_index"])
            ts = t.iloc[ci] + per if ci < len(t) else None
            enter_cnt.setdefault(k, []).append(ts)

        for ob in smc.get("order_blocks", []):
            key = _ob_identity(ob)
            lf = life_by_key.get(key)
            if lf is None or lf["available_time"] is None:
                continue
            zl, zh, swapped = spatial_ob_bounds(ob["bar_low"], ob["bar_high"])
            rows.append(dict(
                ob_id=(f"{sym}|{tf}|{int(bool(ob['internal']))}|"
                       f"{int(ob['bias'])}|{int(ob['anchor_index'])}|"
                       f"{int(ob['confirmed_index'])}"),
                symbol=sym, ob_source_tf=tf,
                ob_internal=bool(ob["internal"]),
                ob_bias=int(ob["bias"]),
                ob_available_time=lf["available_time"],
                ob_inactive_time=lf["inactive_time"],
                ob_inactive_reason=lf["inactive_reason"],
                zone_low=float(zl), zone_high=float(zh),
                endpoints_swapped=bool(swapped),
                ob_enter_times=sorted(
                    [x for x in enter_cnt.get(key, []) if x is not None]),
            ))
    return pd.DataFrame(rows)


def enter_mode(side: int, extreme: float, close_t0: float,
               near_edge: float, far_edge: float) -> str:
    """进入方式 5 分类（按 liquidity side 对称）。"""
    if side == +1:
        # 向上扫：near=zone_low, far=zone_high, far > near
        if close_t0 > far_edge:
            return "CLOSE_BEYOND_OB"
        if extreme > far_edge:
            return "TRAVERSE_OB"
        if close_t0 >= near_edge:
            return "CLOSE_INSIDE_OB"
        if extreme >= near_edge:
            return "WICK_ENTER_OB"
        return "NEAR_MISS"
    else:
        # 向下扫：near=zone_high, far=zone_low, far < near
        if close_t0 < far_edge:
            return "CLOSE_BEYOND_OB"
        if extreme < far_edge:
            return "TRAVERSE_OB"
        if close_t0 <= near_edge:
            return "CLOSE_INSIDE_OB"
        if extreme <= near_edge:
            return "WICK_ENTER_OB"
        return "NEAR_MISS"


def classify(liq_level: float, side: int, extreme: float, close_t0: float,
             t0: pd.Timestamp, obm: pd.DataFrame, r0: float) -> dict:
    """判断穿透路径与"事前存在的反向订单块"的几何关系。

    无论 hit / miss，都会记录 liquidity 外侧**最近**的反向 active OB。
    """
    rev = -side
    out = dict(has_opposing_ob_hit=False, opposing_ob_count=0,
               same_direction_ob_hit=False, same_direction_ob_count=0,
               ob_geometry_relation="NO_RELEVANT_OB")

    def near_of(r):
        # 向上扫 BSL：反应区在上方，第一次遇到的是 zone_low
        # 向下扫 SSL：反应区在下方，第一次遇到的是 zone_high
        return float(r.zone_low) if side == +1 else float(r.zone_high)

    def far_of(r):
        return float(r.zone_high) if side == +1 else float(r.zone_low)

    act = []
    for r in obm.itertuples(index=False):
        # 因果边界：事前存在 + t0 时仍 active
        if r.ob_available_time > t0:
            continue
        if r.ob_inactive_time is not None and r.ob_inactive_time <= t0:
            continue

        ne, fe = near_of(r), far_of(r)
        zl, zh = float(r.zone_low), float(r.zone_high)

        if side == +1:
            if zl > liq_level:
                rel = "BEYOND"
            elif zh < liq_level:
                rel = "BEFORE"
            else:
                rel = "CONTAINS"
            reached = extreme >= ne
        else:
            if zh < liq_level:
                rel = "BEYOND"
            elif zl > liq_level:
                rel = "BEFORE"
            else:
                rel = "CONTAINS"
            reached = extreme <= ne

        if int(r.ob_bias) == rev:
            if rel == "BEYOND":
                act.append(dict(
                    ob_id=r.ob_id, source_tf=r.ob_source_tf,
                    bias=int(r.ob_bias), zone_low=zl, zone_high=zh,
                    near_edge=ne, far_edge=fe,
                    distance=abs(ne - liq_level), reached=bool(reached),
                    enters=[x for x in (r.ob_enter_times or []) if x <= t0]))
            elif rel == "CONTAINS":
                out["ob_geometry_relation"] = "OB_CONTAINS_LIQUIDITY"
            elif out["ob_geometry_relation"] == "NO_RELEVANT_OB":
                out["ob_geometry_relation"] = "OB_BEFORE_LIQUIDITY"
        elif int(r.ob_bias) == side:
            out["same_direction_ob_count"] += 1
            if reached:
                out["same_direction_ob_hit"] = True

    if not act:
        return out

    # 最近的反向 OB —— 无论 hit / miss 都记录
    act.sort(key=lambda c: c["distance"])
    nearest = act[0]
    out["opposing_ob_count"] = len(act)
    out["all_ob_ids"] = ",".join(c["ob_id"] for c in act)
    out["all_ob_source_tfs"] = ",".join(
        sorted({c["source_tf"] for c in act}))

    if np.isfinite(r0) and r0 > 0:
        out["nearest_ob_distance_R"] = nearest["distance"] / r0
        out["nearest_ob_width_R"] = (abs(nearest["zone_high"]
                                         - nearest["zone_low"]) / r0)
        # ob_margin_R = penetration_depth_R - nearest_ob_distance_R
        #            = side * (extreme - near_edge) / r0
        out["ob_margin_R"] = side * (extreme - nearest["near_edge"]) / r0
    out["nearest_ob_id"] = nearest["ob_id"]
    out["nearest_ob_source_tf"] = nearest["source_tf"]
    out["nearest_ob_near_edge"] = nearest["near_edge"]
    out["nearest_ob_far_edge"] = nearest["far_edge"]
    out["nearest_ob_prior_enter_count"] = len(nearest["enters"])
    out["nearest_ob_freshness"] = ("FRESH" if len(nearest["enters"]) == 0
                                   else "RETESTED")

    out["ob_geometry_relation"] = "OB_BEYOND_LIQUIDITY"
    out["ob_enter_mode"] = enter_mode(side, extreme, close_t0,
                                      nearest["near_edge"],
                                      nearest["far_edge"])

    beyond = [c for c in act if c["reached"]]
    if beyond:
        p = beyond[0]
        out.update(dict(
            has_opposing_ob_hit=True,
            primary_ob_id=p["ob_id"], primary_ob_source_tf=p["source_tf"],
            primary_ob_bias=p["bias"], ob_zone_low=p["zone_low"],
            ob_zone_high=p["zone_high"], ob_near_edge=p["near_edge"],
            prior_ob_enter_count=len(p["enters"]),
        ))
    return out


def main():
    tr = pd.read_parquet(
        "research/analysis_results/liquidity_specificity_v2/"
        "true_interactions.parquet")
    m = ((tr["activation_state"] == "VALID_AHEAD")
         & (tr["interaction_path"] == "CONTINUOUS_CROSS")
         & (tr["penetrated"] == True))  # noqa: E712
    T = tr[m].copy()
    print(f"fresh penetrated interactions = {len(T)}")

    audits, rows = [], []
    t0 = time.perf_counter()
    for sym, g in T.groupby("symbol"):
        obm = build_ob_map(sym)
        audits.append(dict(symbol=sym, n_ob=len(obm),
                           by_tf=obm["ob_source_tf"]
                           .value_counts().to_dict()))
        five = load_raw_5m(sym).sort_values("bar_start_time").reset_index(
            drop=True)
        hi = five["high"].to_numpy(float)
        lo = five["low"].to_numpy(float)
        cl = five["close"].to_numpy(float)
        for r in g.itertuples(index=False):
            i = int(r.interaction_i)
            ext = hi[i] if int(r.side) == +1 else lo[i]
            c = classify(float(r.price), int(r.side), float(ext),
                         float(cl[i]),
                         pd.Timestamp(r.penetration_bar_start), obm,
                         float(getattr(r, "R_at_t0", np.nan)))
            rec = dict(level_key=r.level_key, symbol=sym,
                       liquidity_type=r.liquidity_type, side=int(r.side),
                       interaction_i=i,
                       penetration_bar_start=r.penetration_bar_start,
                       stage1=r.stage1,
                       penetration_depth_R=getattr(r, "penetration_depth_R",
                                                   np.nan),
                       close_beyond_R=getattr(r, "close_beyond_R", np.nan))
            rec.update(c)
            rows.append(rec)
        print(f"  {sym}: ob={len(obm)} ({time.perf_counter()-t0:.0f}s)",
              flush=True)

    out = pd.DataFrame(rows)
    out.to_parquet(RESULTS / "ob_confluence_interactions.parquet",
                   index=False)
    pd.DataFrame(audits).to_csv(RESULTS / "ob_active_map_audit.csv",
                                index=False, encoding="utf-8-sig")
    (out["ob_geometry_relation"].value_counts().rename("n")
     .reset_index()).rename(
        columns={"index": "relation"}).to_csv(
        RESULTS / "ob_geometry_relation.csv", index=False,
        encoding="utf-8-sig")

    # ---------- 对照组修复 ----------
    treat = out[(out["ob_geometry_relation"] == "OB_BEYOND_LIQUIDITY")
                & (out["has_opposing_ob_hit"])]
    control = out[(out["ob_geometry_relation"] == "OB_BEYOND_LIQUIDITY")
                  & (~out["has_opposing_ob_hit"])
                  & (out["opposing_ob_count"] > 0)]
    print(f"\nOB_BEYOND_LIQUIDITY 总计 = "
          f"{int((out['ob_geometry_relation']=='OB_BEYOND_LIQUIDITY').sum())}"
          f"  (应为 48655)")
    print(f"Treatment (扫进OB)  = {len(treat)}   (应 ~15177)")
    print(f"Control   (未扫到)  = {len(control)} (应 ~33478)")
    print(f"合计               = {len(treat)+len(control)}")

    # ---------- ob_margin 断言 ----------
    if "ob_margin_R" in out.columns:
        tm = treat["ob_margin_R"].dropna()
        cm = control["ob_margin_R"].dropna()
        print(f"\n断言 treatment ob_margin_R >= -TOL: "
              f"{bool((tm >= -TOL).all())} (min={tm.min():.6f})")
        print(f"断言 control   ob_margin_R <  TOL: "
              f"{bool((cm < TOL).all())} (max={cm.max():.6f})")

    # ---------- 共同支撑画像 ----------
    cols = ["penetration_depth_R", "close_beyond_R",
            "nearest_ob_distance_R", "ob_margin_R", "nearest_ob_width_R"]
    prof = []
    for tag, g in (("TREAT(hit)", treat), ("CONTROL(miss)", control),
                   ("ALL_BEYOND", out[out["ob_geometry_relation"]
                                      == "OB_BEYOND_LIQUIDITY"])):
        for c in cols:
            if c not in g.columns:
                continue
            v = pd.to_numeric(g[c], errors="coerce").dropna()
            if not len(v):
                continue
            prof.append(dict(组=tag, 变量=c, n=len(v),
                             p10=round(float(v.quantile(.10)), 4),
                             p25=round(float(v.quantile(.25)), 4),
                             median=round(float(v.median()), 4),
                             p75=round(float(v.quantile(.75)), 4),
                             p90=round(float(v.quantile(.90)), 4)))
    pd.DataFrame(prof).to_csv(RESULTS / "ob_geometry_support_profile.csv",
                              index=False, encoding="utf-8-sig")

    # 关键：同样 penetration_depth_R 区间内 hit/miss 是否都有样本
    bey = out[out["ob_geometry_relation"] == "OB_BEYOND_LIQUIDITY"].copy()
    bey["hit"] = bey["has_opposing_ob_hit"].astype(int)
    bins = [-np.inf, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, np.inf]
    bey["depth_bin"] = pd.cut(pd.to_numeric(bey["penetration_depth_R"],
                                            errors="coerce"), bins)
    sup = (bey.groupby("depth_bin", observed=True)
           .agg(n=("hit", "size"), n_hit=("hit", "sum"))
           .reset_index())
    sup["n_miss"] = sup["n"] - sup["n_hit"]
    sup["hit_rate"] = (sup["n_hit"] / sup["n"]).round(4)
    sup.to_csv(RESULTS / "landmark_a_support_by_depth.csv", index=False,
               encoding="utf-8-sig")
    print("\n=== 共同支撑：penetration_depth_R 分箱 ===")
    print(sup.to_string(index=False))
    ok = bool(((sup["n_hit"] >= 200) & (sup["n_miss"] >= 200)).sum() >= 3)
    print(f"\n共同支撑（>=3 个分箱 hit/miss 各 >=200）: "
          f"{'PASS' if ok else 'LANDMARK_A_SUPPORT_FAIL'}")

    print("\nV3_GEOMETRY_DONE")


if __name__ == "__main__":
    main()
