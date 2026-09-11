"""SMC Oracle Atlas v1.1 —— 第三层：修复后画像与语义审计。

本轮只做审计与画像：不训练模型、不筛选状态、不做 PnL。
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
from research.liquidity_oracle_atlas.build_oracle_atlas_v1_1 import (
    AUDIT_DIST_ATR, AUDIT_MAX_N, AUDIT_MAXB,
)


def q(x, p):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return round(float(np.percentile(x, p)), 4) if len(x) else np.nan


def main():
    M = pd.read_parquet(RESULTS / "liquidity_master_v1_1.parquet")
    C = pd.read_parquet(RESULTS / "liquidity_contacts_v1_1.parquet")
    S = pd.read_parquet(RESULTS / "liquidity_state_snapshot_v1_1.parquet")
    O = pd.read_parquet(RESULTS / "oracle_risk_frontier_v1_1.parquet")
    A = pd.read_parquet(RESULTS / "oracle_path_audit_v1_1.parquet")

    out = {}

    # ---------- 1. active density vs v1.0 ----------
    rows = []
    for sc in SCOPES:
        pos = f"{sc}_bin_(0,0.5]"
        neg = f"{sc}_bin_(-0.5,0]"
        if pos in S.columns and neg in S.columns:
            v = S[pos].to_numpy(float) + S[neg].to_numpy(float)
            rows.append(dict(scope=sc, pm05_mean_active=round(float(v.mean()), 4),
                             pm05_median_active=round(float(np.median(v)), 4)))
    AD = pd.DataFrame(rows)
    try:
        v10 = pd.read_csv(RESULTS / "liquidity_field_profile.csv")
        v10 = v10[v10["bin"].isin(["(0,0.5]", "(-0.5,0]"])]
        g10 = v10.groupby("scope")["mean_count"].sum().rename("pm05_mean_v10")
        AD = AD.merge(g10, left_on="scope", right_index=True, how="left")
        AD["ratio_v11_over_v10"] = (AD["pm05_mean_active"]
                                    / AD["pm05_mean_v10"]).round(4)
    except Exception:
        pass
    AD.to_csv(RESULTS / "active_density_vs_v10.csv", index=False,
              encoding="utf-8-sig")
    print("=== active density (+-0.5R) ===")
    print(AD.to_string(index=False))
    out["active_density_pm05"] = AD.to_dict("records")

    # ---------- 2. same-price overlap ----------
    act = pd.to_numeric(S["same_price_identity_count"], errors="coerce")
    v10c = pd.to_numeric(S["same_price_identity_count_v10"], errors="coerce")
    out["pct_multi_active"] = round(float((act >= 2).mean()), 4)
    out["pct_multi_v10_historical"] = round(float((v10c >= 2).mean()), 4)
    out["mean_active_visible"] = round(
        float(pd.to_numeric(S["active_visible_count"]).mean()), 2)
    out["mean_historical_visible"] = round(
        float(pd.to_numeric(S["historical_visible_count"]).mean()), 2)
    print(f"\nmulti-identity: active={out['pct_multi_active']:.4f} "
          f"v1.0历史口径={out['pct_multi_v10_historical']:.4f}")
    print(f"visible: active={out['mean_active_visible']} "
          f"historical={out['mean_historical_visible']}")

    # ---------- 3. duplicate identity audit ----------
    mk = ["symbol", "price", "side", "liquidity_type", "liquidity_scope",
          "available_time"]
    dup = (M.groupby(mk).size().rename("n").reset_index())
    n_dup_rows = int((dup["n"] > 1).sum())
    n_dup_extra = int((dup["n"] - 1).clip(lower=0).sum())
    out["duplicate_identity_groups"] = n_dup_rows
    out["duplicate_identity_extra_rows"] = n_dup_extra
    pd.DataFrame([dict(metric="完全重复 identity 组数", value=n_dup_rows),
                  dict(metric="多出的重复行", value=n_dup_extra),
                  dict(metric="总 identity", value=int(len(M)))]) \
        .to_csv(RESULTS / "duplicate_identity_audit.csv", index=False,
                encoding="utf-8-sig")
    print(f"\n完全重复 identity 组数 = {n_dup_rows}（多出 {n_dup_extra} 行）")

    # ---------- 4. contact hazard ----------
    hz = []
    for k in (1, 2, 3, 4):
        g = C[C["contact_number"] == k]
        if not len(g):
            continue
        hz.append(dict(contact_number=k, n_at_risk=len(g),
                       penetration_hazard=round(float(g["is_penetration"].mean()), 4),
                       touch_only_probability=round(
                           float(1 - g["is_penetration"].mean()), 4)))
    HZ = pd.DataFrame(hz)
    HZ.to_csv(RESULTS / "contact_hazard.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== contact hazard ===")
    print(HZ.to_string(index=False))
    out["contact_hazard"] = HZ.to_dict("records")

    # ---------- 5/6. 截断审计 ----------
    o5 = O[O["path_len"] > AUDIT_MAXB]
    cens500 = O[(O["path_len"] > AUDIT_MAXB)
                & ((~O["stop_hit"]) | (O["bars_to_stop"] > AUDIT_MAXB))]
    ca = [dict(metric="path_len > 500 的 oracle 行占比",
               value=round(float(len(o5) / len(O)), 4)),
          dict(metric="第500根仍未 stop 且路径未结束（v1.0 被截断）占比",
               value=round(float(len(cens500) / len(O)), 4)),
          dict(metric="20ATR 距离截断会丢 target 的 contact 占比",
               value=round(float((O["n_target_clipped_dist"] > 0).mean()), 4)),
          dict(metric="400 target 数量截断生效的 contact 占比",
               value=round(float((O["n_target_clipped_count"] > 0).mean()), 4)),
          dict(metric="median path_len", value=q(O["path_len"], 50)),
          dict(metric="p95 path_len", value=q(O["path_len"], 95))]
    CA = pd.DataFrame(ca)
    CA.to_csv(RESULTS / "oracle_truncation_audit.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== 截断审计 ===")
    print(CA.to_string(index=False))
    out["truncation_audit"] = {r["metric"]: r["value"]
                               for r in CA.to_dict("records")}

    # ---------- 7. ATR-RR 前沿（修复后） ----------
    rr = []
    for d in ("LONG", "SHORT"):
        for rk in RISK_ATR_GRID:
            v = O[(O.direction == d) & (O.risk_ATR == rk)]
            r = v["conservative_best_R"].to_numpy(float)
            b = v["bars_to_best_conservative"].dropna().to_numpy(float)
            rr.append(dict(direction=d, risk_ATR=rk, n=len(v),
                           mean_R=round(float(r.mean()), 4),
                           median_R=q(r, 50), p75_R=q(r, 75), p90_R=q(r, 90),
                           pct_zero_R=round(float((r <= 0).mean()), 4),
                           pct_stopped=round(float(v["stop_hit"].mean()), 4),
                           median_bars_to_best=q(b, 50),
                           p90_bars_to_best=q(b, 90)))
    RR = pd.DataFrame(rr)
    RR.to_csv(RESULTS / "risk_atr_rr_profile_v1_1.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== ATR-RR 前沿（v1.1） ===")
    print(RR.to_string(index=False))

    # ---------- 8. censor 分布 ----------
    cs = (O.groupby(["direction", "status"]).size().rename("n")
          .reset_index())
    cs.to_csv(RESULTS / "oracle_status_profile.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== Oracle status ===")
    print(cs.pivot_table(index="status", columns="direction", values="n",
                         aggfunc="first").to_string())
    out["status_counts"] = {k: int(v) for k, v in
                            O["status"].value_counts().items()}
    out["path_censor"] = {k: int(v) for k, v in
                          O["path_censor"].value_counts().items()}

    # ---------- 9. RR dominance ----------
    p = O.pivot_table(index=["symbol", "liquidity_id", "contact_number",
                             "risk_ATR"], columns="direction",
                      values="conservative_best_R", aggfunc="first")
    p = p.reset_index()
    p.columns = [str(c) for c in p.columns]
    p = p.rename(columns={"LONG": "long_best_R", "SHORT": "short_best_R"})

    def dom(g):
        L = g["long_best_R"].to_numpy(float)
        Sh = g["short_best_R"].to_numpy(float)
        if (L <= 0).all() and (Sh <= 0).all():
            return "NO_CLEAR_RR"
        if (L >= Sh).all() and (L > Sh).any():
            return "LONG_RR_DOMINATES"
        if (Sh >= L).all() and (Sh > L).any():
            return "SHORT_RR_DOMINATES"
        return "MIXED_RR"

    lab = (p.groupby(["symbol", "liquidity_id", "contact_number"])
           .apply(dom).rename("rr_direction").reset_index())
    C2 = C.merge(lab, on=["symbol", "liquidity_id", "contact_number"],
                 how="left")

    # ---------- 10. time-aware Pareto dominance ----------
    def pareto_dir(g):
        d = g[["risk_ATR", "conservative_best_R",
               "bars_to_best_conservative"]].dropna()
        if not len(d):
            return []
        pts = d.to_numpy(float)
        keep = []
        for i, pt in enumerate(pts):
            bad = False
            for j, qt in enumerate(pts):
                if i == j:
                    continue
                if (qt[1] >= pt[1] and qt[2] <= pt[2] and qt[0] <= pt[0]
                        and (qt[1] > pt[1] or qt[2] < pt[2] or qt[0] < pt[0])):
                    bad = True
                    break
            if not bad:
                keep.append((pt[0], pt[1], pt[2], g["direction"].iloc[0]))
        return keep

    prows, pdom = [], []
    for (sym, lid, cno), g in O.groupby(["symbol", "liquidity_id",
                                         "contact_number"]):
        front = []
        for d in ("LONG", "SHORT"):
            front.extend(pareto_dir(g[g["direction"] == d]))
        for r, rr_, b, d in front:
            prows.append(dict(symbol=sym, liquidity_id=lid,
                              contact_number=cno, direction=d, risk_ATR=r,
                              best_R=round(float(rr_), 4),
                              bars_to_best=int(b)))
        if front:
            dirs = {x[3] for x in front}
            if dirs == {"LONG"}:
                pdom.append(dict(symbol=sym, liquidity_id=lid,
                                 contact_number=cno,
                                 pareto_direction="LONG_PARETO_DOMINATES"))
            elif dirs == {"SHORT"}:
                pdom.append(dict(symbol=sym, liquidity_id=lid,
                                 contact_number=cno,
                                 pareto_direction="SHORT_PARETO_DOMINATES"))
            else:
                pdom.append(dict(symbol=sym, liquidity_id=lid,
                                 contact_number=cno,
                                 pareto_direction="MIXED_PARETO"))
        else:
            pdom.append(dict(symbol=sym, liquidity_id=lid,
                             contact_number=cno,
                             pareto_direction="NO_PARETO"))
    PF = pd.DataFrame(prows)
    PF.to_parquet(RESULTS / "oracle_pareto_frontier_v1_1.parquet",
                  index=False)
    PD = pd.DataFrame(pdom)
    C2 = C2.merge(PD, on=["symbol", "liquidity_id", "contact_number"],
                  how="left")
    C2.to_parquet(RESULTS / "liquidity_oracle_labels_v1_1.parquet",
                  index=False)

    for col in ("rr_direction", "pareto_direction"):
        t = (C2[col].value_counts().rename("n").reset_index())
        t.columns = [col, "n"]
        t["share"] = (t["n"] / t["n"].sum()).round(4)
        t.to_csv(RESULTS / f"oracle_{col}_profile.csv", index=False,
                 encoding="utf-8-sig")
        print(f"\n=== {col} ===")
        print(t.to_string(index=False))
        out[col] = {k: int(v) for k, v in
                    C2[col].value_counts().items()}

    # ---------- 11. target profile（最佳目标 scope/type） ----------
    tb = O[O["best_target_price"].notna()]
    if len(tb):
        tg = pd.DataFrame({
            "has_1h": tb.groupby("risk_ATR")["best_target_has_1h"].mean(),
            "has_day": tb.groupby("risk_ATR")["best_target_has_day"].mean(),
            "cluster_size": tb.groupby("risk_ATR")[
                "best_target_cluster_size"].mean()}).round(4).reset_index()
        tg.to_csv(RESULTS / "best_target_profile.csv", index=False,
                  encoding="utf-8-sig")
        print("\n=== best target profile ===")
        print(tg.to_string(index=False))

    json.dump(out, open(RESULTS / "AUDIT_ORACLE_ATLAS_V1_1.json", "w"),
              indent=2, ensure_ascii=False)
    print("\nPROFILE_V1_1_DONE")


if __name__ == "__main__":
    main()
