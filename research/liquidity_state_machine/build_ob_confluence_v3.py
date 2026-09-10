"""多周期流动性 × 订单块共振实验 v3.0 —— 订单块层与共振分类。

订单块完全复用 canonical：
    build_full_ob_smc_tf / replay_ob_lifetimes / spatial_ob_bounds
来源周期仅 5m / 15m / 1h（仓库无合法 canonical 4h OB，禁止伪造）。

硬因果边界：
    ob_available_time   <= penetration_bar_start
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


def build_ob_map(sym: str) -> pd.DataFrame:
    """返回每个 (tf, OB) 的生命周期与空间边界。"""
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

        # 每个 OB 身份 -> 生命周期
        life_by_key = {}
        for lv in lives:
            key = lv["key"]
            ci = int(lv["confirmed_index"])
            ii = lv.get("inactive_index")
            life_by_key[key] = dict(
                available_time=t.iloc[ci] + per if ci < len(t) else None,
                inactive_time=(t.iloc[int(ii)] + per
                               if (ii is not None and int(ii) < len(t))
                               else None),
                inactive_reason=lv.get("inactive_reason"),
            )

        # OB_ENTERED 次数（用于 freshness）
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
                ob_id=f"{sym}|{tf}|{int(bool(ob['internal']))}|"
                      f"{int(ob['bias'])}|{int(ob['anchor_index'])}|"
                      f"{int(ob['confirmed_index'])}",
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


def classify(liq_level: float, side: int, extreme: float,
             t0: pd.Timestamp, obm: pd.DataFrame) -> dict:
    """判断穿透路径是否进入事前存在的反向订单块。"""
    rev = -side
    out = dict(has_opposing_ob_hit=False, opposing_ob_count=0,
               same_direction_ob_hit=False,
               same_direction_ob_count=0,
               ob_geometry_relation="NO_RELEVANT_OB")

    def near_of(r):
        # 向上扫 BSL：反应区在上方，第一次遇到的是 zone_low
        # 向下扫 SSL：反应区在下方，第一次遇到的是 zone_high
        return float(r.zone_low) if side == +1 else float(r.zone_high)

    act = []
    for r in obm.itertuples(index=False):
        if r.ob_available_time > t0:
            continue
        if r.ob_inactive_time is not None and r.ob_inactive_time <= t0:
            continue
        ne = near_of(r)
        zl, zh = float(r.zone_low), float(r.zone_high)
        if side == +1:
            if zl > liq_level:
                rel, geom = "BEYOND", "OUTSIDE"
            elif zh < liq_level:
                rel, geom = "BEFORE", "INSIDE"
            else:
                rel, geom = "CONTAINS", "CONTAINS"
            reached = extreme >= ne
        else:
            if zh < liq_level:
                rel, geom = "BEYOND", "OUTSIDE"
            elif zl > liq_level:
                rel, geom = "BEFORE", "INSIDE"
            else:
                rel, geom = "CONTAINS", "CONTAINS"
            reached = extreme <= ne

        if int(r.ob_bias) == rev:
            if rel == "BEYOND":
                act.append(dict(ob_id=r.ob_id, source_tf=r.ob_source_tf,
                                bias=int(r.ob_bias), zone_low=zl,
                                zone_high=zh, near_edge=ne,
                                distance=abs(ne - liq_level),
                                reached=bool(reached),
                                available_time=r.ob_available_time,
                                enters=[x for x in (r.ob_enter_times or [])
                                        if x <= t0]))
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

    beyond = [c for c in act if c["reached"]]
    out["opposing_ob_count"] = len(act)
    out["all_ob_ids"] = ",".join(c["ob_id"] for c in act)
    out["all_ob_source_tfs"] = ",".join(sorted({c["source_tf"]
                                                for c in act}))
    if beyond:
        beyond.sort(key=lambda c: c["distance"])
        p = beyond[0]
        out.update(dict(
            has_opposing_ob_hit=True,
            ob_geometry_relation="OB_BEYOND_LIQUIDITY",
            primary_ob_id=p["ob_id"], primary_ob_source_tf=p["source_tf"],
            primary_ob_bias=p["bias"], ob_zone_low=p["zone_low"],
            ob_zone_high=p["zone_high"], ob_near_edge=p["near_edge"],
            prior_ob_enter_count=len(p["enters"]),
        ))
    else:
        out["ob_geometry_relation"] = "OB_BEYOND_LIQUIDITY"
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
        atr = None
        for r in g.itertuples(index=False):
            i = int(r.interaction_i)
            ext = hi[i] if int(r.side) == +1 else lo[i]
            c = classify(float(r.price), int(r.side), float(ext),
                         pd.Timestamp(r.penetration_bar_start), obm)
            rec = dict(level_key=r.level_key, symbol=sym,
                       liquidity_type=r.liquidity_type, side=int(r.side),
                       interaction_i=i,
                       penetration_bar_start=r.penetration_bar_start,
                       stage1=r.stage1, penetration_depth_R=getattr(
                           r, "penetration_depth_R", np.nan))
            rec.update(c)
            rows.append(rec)
        print(f"  {sym}: ob={len(obm)} ({time.perf_counter()-t0:.0f}s)",
              flush=True)

    out = pd.DataFrame(rows)
    out.to_parquet(RESULTS / "ob_confluence_interactions.parquet",
                   index=False)
    pd.DataFrame(audits).to_csv(RESULTS / "ob_active_map_audit.csv",
                                index=False, encoding="utf-8-sig")
    gm = (out["ob_geometry_relation"].value_counts().rename("n")
          .reset_index())
    gm.columns = ["relation", "n"]
    gm.to_csv(RESULTS / "ob_geometry_relation.csv", index=False,
              encoding="utf-8-sig")

    print("\n几何关系分布："); print(gm.to_string(index=False))
    hit = out[out["has_opposing_ob_hit"]]
    print(f"\nOB_CONFLUENCE=1 : {len(hit)}")
    nob = out[(~out["has_opposing_ob_hit"])
              & (out["opposing_ob_count"] == 0)]
    print(f"OB_CONFLUENCE=0 : {len(nob)}")
    print("\n主 OB 来源周期：")
    print(hit["primary_ob_source_tf"].value_counts().to_string())
    print("\nfreshness：")
    print(pd.cut(hit["prior_ob_enter_count"], [-1, 0, 100],
                 labels=["FRESH", "RETESTED"]).value_counts().to_string())
    print("\nV3_CONFLUENCE_DONE")


if __name__ == "__main__":
    main()
