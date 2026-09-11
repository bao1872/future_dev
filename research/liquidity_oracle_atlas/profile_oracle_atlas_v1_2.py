"""SMC Oracle Atlas v1.2 —— 方向标签 / 不确定性 / 稳定性画像。

三层标签：
  第一层 每个 (contact × direction × risk_ATR) 的区间价值
  第二层 同 risk_ATR 下 LONG vs SHORT（RR 比较 + RR×时间比较）
  第三层 跨 7 档 risk 的方向稳定性

risk_ATR 是**条件轴**，不作为"越小越优"的效用目标。
不训练模型、不做 PnL、不筛 subgroup。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.build_liquidity_lifecycle_v1_1 import (
    RESULTS,
)
from research.liquidity_oracle_atlas.build_oracle_atlas_v1 import (
    BIN_LABELS, RISK_ATR_GRID, SCOPES,
)

MIN_RESOLVED = 4


def _cmp_dir(lrow, srow):
    """同 risk 的 RR 方向判定（保守：censor 上界未知则不猜）。"""
    if (lrow["resolution_class"] == "NO_ACTIVE_TARGET"
            or srow["resolution_class"] == "NO_ACTIVE_TARGET"):
        return "NO_COMPARABLE_TARGET"
    lu, su = lrow["best_R_upper"], srow["best_R_upper"]
    if pd.isna(lu) or pd.isna(su):
        return "UNRESOLVED_CENSOR"
    if lrow["best_R_lower"] > su:
        return "LONG_DOMINATES"
    if srow["best_R_lower"] > lu:
        return "SHORT_DOMINATES"
    return "TRADEOFF_OR_OVERLAP"


def _time_dir(lrow, srow):
    """同 risk 的 RR × holding bars 比较。"""
    if (lrow["resolution_class"] == "NO_ACTIVE_TARGET"
            or srow["resolution_class"] == "NO_ACTIVE_TARGET"):
        return "NO_COMPARABLE_TARGET"
    if pd.isna(lrow["best_R_upper"]) or pd.isna(srow["best_R_upper"]):
        return "UNRESOLVED"
    lR, sR = lrow["best_R_lower"] or 0.0, srow["best_R_lower"] or 0.0
    lb, sb = lrow["bars_to_best_lower"], srow["bars_to_best_lower"]
    if lR > 0 and sR > 0 and pd.notna(lb) and pd.notna(sb):
        if lR >= sR and lb <= sb and (lR > sR or lb < sb):
            return "LONG_TIME_RR_DOMINATES"
        if sR >= lR and sb <= lb and (sR > lR or sb < lb):
            return "SHORT_TIME_RR_DOMINATES"
        return "TRADEOFF"
    if lR > 0:
        return "LONG_TIME_RR_DOMINATES"
    if sR > 0:
        return "SHORT_TIME_RR_DOMINATES"
    return "NO_MOVE_BOTH"


def _stability(dirs):
    res = [d for d in dirs if d in ("LONG_DOMINATES", "SHORT_DOMINATES",
                                    "TRADEOFF_OR_OVERLAP")]
    nl = dirs.count("LONG_DOMINATES")
    ns = dirs.count("SHORT_DOMINATES")
    if len(res) < MIN_RESOLVED:
        return "UNRESOLVED"
    if nl >= 1 and ns >= 1:
        return "RISK_DEPENDENT"
    if nl == len(res):
        return "ROBUST_LONG"
    if ns == len(res):
        return "ROBUST_SHORT"
    return "NO_DIRECTION"


def main():
    M = pd.read_parquet(RESULTS / "liquidity_master_v1_1.parquet")
    C = pd.read_parquet(RESULTS / "liquidity_contacts_v1_1.parquet")
    S = pd.read_parquet(RESULTS / "liquidity_state_snapshot_v1_2.parquet")
    O = pd.read_parquet(RESULTS / "oracle_risk_frontier_v1_2.parquet")
    out = {}

    # ---------- 1. active density / overlap ----------
    rows = []
    for sc in SCOPES:
        p, n_ = f"{sc}_bin_(0,0.5]", f"{sc}_bin_(-0.5,0]"
        if p in S.columns:
            v = S[p].to_numpy(float) + S[n_].to_numpy(float)
            rows.append(dict(scope=sc, pm05_mean_v1_2=round(float(v.mean()), 4)))
    AD = pd.DataFrame(rows)
    try:
        v11 = pd.read_csv(RESULTS / "active_density_vs_v10.csv")
        AD = AD.merge(v11[["scope", "pm05_mean_active"]].rename(
            columns={"pm05_mean_active": "pm05_mean_v1_1"}),
            on="scope", how="left")
        AD["ratio_v12_over_v11"] = (AD["pm05_mean_v1_2"]
                                    / AD["pm05_mean_v1_1"]).round(4)
    except Exception:
        pass
    AD.to_csv(RESULTS / "active_density_v1_2.csv", index=False,
              encoding="utf-8-sig")
    print("=== active density (+-0.5R) ===")
    print(AD.to_string(index=False))
    out["active_density"] = AD.to_dict("records")

    a12 = pd.to_numeric(S["same_price_identity_count"], errors="coerce")
    a11 = pd.to_numeric(S["same_price_identity_count_v11"], errors="coerce")
    out["pct_multi_v1_2"] = round(float((a12 >= 2).mean()), 4)
    out["pct_multi_v1_1"] = round(float((a11 >= 2).mean()), 4)
    print(f"\nmulti-identity: v1.2={out['pct_multi_v1_2']:.4f} "
          f"v1.1={out['pct_multi_v1_1']:.4f}")

    # ---------- 2. resolution / censor ----------
    rc = (O["resolution_class"].value_counts().rename("n").reset_index())
    rc.columns = ["resolution_class", "n"]
    rc["share"] = (rc["n"] / rc["n"].sum()).round(4)
    rc.to_csv(RESULTS / "censor_resolution_profile_v1_2.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== resolution_class ===")
    print(rc.to_string(index=False))
    out["resolution_class"] = {k: int(v) for k, v in
                               O["resolution_class"].value_counts().items()}

    # ---------- 3/4. risk-specific direction ----------
    # liquidity_id 本身含 "|"，故用多列索引做 join，不拼字符串
    idx = ["symbol", "liquidity_id", "contact_number", "risk_ATR"]
    cols = ["best_R_lower", "best_R_upper", "resolution_class",
            "bars_to_best_lower"]
    Lg = O[O["direction"] == "LONG"].set_index(idx)[cols]
    Sg = O[O["direction"] == "SHORT"].set_index(idx)[cols]
    B = Lg.join(Sg, lsuffix="_l", rsuffix="_s", how="inner").reset_index()
    ll, lu = B["best_R_lower_l"], B["best_R_upper_l"]
    sl, su = B["best_R_lower_s"], B["best_R_upper_s"]
    lb, sb = B["bars_to_best_lower_l"], B["bars_to_best_lower_s"]
    notg = ((B["resolution_class_l"] == "NO_ACTIVE_TARGET")
            | (B["resolution_class_s"] == "NO_ACTIVE_TARGET"))
    unres = (~notg) & (lu.isna() | su.isna())
    cmp_ok = (~notg) & (~unres)
    B["rr_direction"] = np.select(
        [notg, unres, cmp_ok & (ll > su), cmp_ok & (sl > lu)],
        ["NO_COMPARABLE_TARGET", "UNRESOLVED_CENSOR",
         "LONG_DOMINATES", "SHORT_DOMINATES"],
        default="TRADEOFF_OR_OVERLAP")
    both_pos = cmp_ok & (ll > 0) & (sl > 0) & lb.notna() & sb.notna()
    ldom = both_pos & (ll >= sl) & (lb <= sb) & ((ll > sl) | (lb < sb))
    sdom = both_pos & (sl >= ll) & (sb <= lb) & ((sl > ll) | (sb < lb))
    B["time_direction"] = np.select(
        [notg, unres, ldom, sdom,
         cmp_ok & (ll > 0) & (sl <= 0), cmp_ok & (sl > 0) & (ll <= 0),
         both_pos],
        ["NO_COMPARABLE_TARGET", "UNRESOLVED",
         "LONG_TIME_RR_DOMINATES", "SHORT_TIME_RR_DOMINATES",
         "LONG_TIME_RR_DOMINATES", "SHORT_TIME_RR_DOMINATES",
         "TRADEOFF"], default="NO_MOVE_BOTH")
    RD = B[idx + ["rr_direction", "best_R_lower_l", "best_R_upper_l",
                  "best_R_lower_s", "best_R_upper_s"]].rename(
        columns={"best_R_lower_l": "long_R_lower",
                 "best_R_upper_l": "long_R_upper",
                 "best_R_lower_s": "short_R_lower",
                 "best_R_upper_s": "short_R_upper"})
    TD = B[idx + ["time_direction", "bars_to_best_lower_l",
                  "bars_to_best_lower_s"]].rename(
        columns={"bars_to_best_lower_l": "long_bars",
                 "bars_to_best_lower_s": "short_bars"})
    RD.to_parquet(RESULTS / "oracle_risk_direction_v1_2.parquet", index=False)
    TD.to_parquet(RESULTS / "oracle_time_direction_v1_2.parquet", index=False)

    prof = []
    for risk in RISK_ATR_GRID:
        g = RD[RD["risk_ATR"] == risk]
        vc = g["rr_direction"].value_counts()
        prof.append(dict(risk_ATR=risk, n=len(g),
                         **{k: int(vc.get(k, 0)) for k in
                            ("LONG_DOMINATES", "SHORT_DOMINATES",
                             "TRADEOFF_OR_OVERLAP", "UNRESOLVED_CENSOR",
                             "NO_COMPARABLE_TARGET")},
                         pct_resolved=round(float(
                             (~g["rr_direction"].isin(
                                 ["UNRESOLVED_CENSOR",
                                  "NO_COMPARABLE_TARGET"])).mean()), 4)))
    RDP = pd.DataFrame(prof)
    RDP.to_csv(RESULTS / "risk_direction_profile_v1_2.csv", index=False,
               encoding="utf-8-sig")
    print("\n=== risk-specific RR direction ===")
    print(RDP.to_string(index=False))
    out["risk_direction"] = RDP.to_dict("records")

    tprof = []
    for risk in RISK_ATR_GRID:
        g = TD[TD["risk_ATR"] == risk]
        vc = g["time_direction"].value_counts()
        tprof.append(dict(risk_ATR=risk, n=len(g),
                          **{k: int(vc.get(k, 0)) for k in
                             ("LONG_TIME_RR_DOMINATES",
                              "SHORT_TIME_RR_DOMINATES", "TRADEOFF",
                              "UNRESOLVED", "NO_COMPARABLE_TARGET",
                              "NO_MOVE_BOTH")}))
    TDP = pd.DataFrame(tprof)
    TDP.to_csv(RESULTS / "time_direction_profile_v1_2.csv", index=False,
               encoding="utf-8-sig")
    print("\n=== risk-specific time-aware direction ===")
    print(TDP.to_string(index=False))

    # ---------- 5. cross-risk stability ----------
    srows = []
    for key, g in RD.groupby(["symbol", "liquidity_id", "contact_number"]):
        dirs = list(g.sort_values("risk_ATR")["rr_direction"])
        srows.append(dict(
            symbol=key[0], liquidity_id=key[1], contact_number=key[2],
            n_risk=len(dirs),
            n_long_dom=dirs.count("LONG_DOMINATES"),
            n_short_dom=dirs.count("SHORT_DOMINATES"),
            n_tradeoff=dirs.count("TRADEOFF_OR_OVERLAP"),
            n_unresolved=dirs.count("UNRESOLVED_CENSOR"),
            n_no_target=dirs.count("NO_COMPARABLE_TARGET"),
            direction_stability=_stability(dirs)))
    SD = pd.DataFrame(srows)
    SD.to_parquet(RESULTS / "oracle_direction_stability_v1_2.parquet",
                  index=False)
    sp = (SD["direction_stability"].value_counts().rename("n")
          .reset_index())
    sp.columns = ["direction_stability", "n"]
    sp["share"] = (sp["n"] / sp["n"].sum()).round(4)
    sp.to_csv(RESULTS / "direction_stability_profile_v1_2.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== cross-risk direction stability ===")
    print(sp.to_string(index=False))
    out["direction_stability"] = {k: int(v) for k, v in
                                  SD["direction_stability"]
                                  .value_counts().items()}

    # v1.1 rr_direction 对比
    try:
        o11 = pd.read_parquet(RESULTS / "liquidity_oracle_labels_v1_1.parquet")
        cmp_ = SD.merge(o11[["symbol", "liquidity_id", "contact_number",
                             "rr_direction"]],
                        on=["symbol", "liquidity_id", "contact_number"],
                        how="inner", suffixes=("_v12", "_v11"))
        ct = pd.crosstab(cmp_["rr_direction_v11"], cmp_["direction_stability"])
        ct.to_csv(RESULTS / "v11_vs_v12_direction_crosstab.csv",
                  encoding="utf-8-sig")
        print("\n=== v1.1 rr_direction × v1.2 stability ===")
        print(ct.to_string())
    except Exception as e:
        print(f"\n(v1.1 对比跳过: {e})")

    # ---------- 8. target cluster ----------
    tb = O[O["best_target_price"].notna()]
    if len(tb):
        comb = (tb["best_target_scopes"].value_counts().rename("n")
                .reset_index())
        comb.columns = ["scope_combination", "n"]
        comb["share"] = (comb["n"] / comb["n"].sum()).round(4)
        comb.head(20).to_csv(RESULTS / "target_cluster_profile_v1_2.csv",
                             index=False, encoding="utf-8-sig")
        print("\n=== best target scope 组合 top12 ===")
        print(comb.head(12).to_string(index=False))
        out["target_scope_top"] = comb.head(12).to_dict("records")
        # 95%+ day 是否仍成立
        day_cols = [c for c in ("best_target_has_trading_day",
                                "best_target_has_trading_week",
                                "best_target_has_contig_session")
                    if c in tb.columns]
        if day_cols:
            out["pct_best_target_has_day_or_above"] = round(float(
                tb[day_cols].any(axis=1).mean()), 4)
            print(f"\n含 day/session/week 占比 = "
                  f"{out['pct_best_target_has_day_or_above']:.4f}")

    # ---------- 10. hazard（v1.1 口径 + 分层） ----------
    hz = []
    for k in (1, 2, 3, 4):
        g = C[C["contact_number"] == k]
        if not len(g):
            continue
        hz.append(dict(contact_number=k, n_at_risk=len(g),
                       penetration_hazard=round(
                           float(g["is_penetration"].mean()), 4)))
    HZ = pd.DataFrame(hz)
    # 按 scope
    for k in (1, 2, 3):
        g = C[C["contact_number"] == k]
        for sc, gs in g.groupby("liquidity_scope"):
            if len(gs) >= 100:
                hz.append(dict(contact_number=f"{k}/{sc}",
                               n_at_risk=len(gs),
                               penetration_hazard=round(
                                   float(gs["is_penetration"].mean()), 4)))
    HZ = pd.DataFrame(hz)
    HZ.to_csv(RESULTS / "contact_hazard_v1_2.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== contact hazard ===")
    print(HZ.to_string(index=False))
    out["hazard"] = HZ.to_dict("records")

    json.dump(out, open(RESULTS / "AUDIT_ORACLE_ATLAS_V1_2.json", "w"),
              indent=2, ensure_ascii=False)
    print("\nPROFILE_V1_2_DONE")


if __name__ == "__main__":
    main()
