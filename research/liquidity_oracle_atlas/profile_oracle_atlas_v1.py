"""SMC Oracle Atlas v1.0 —— 第三层：画像 / 支配关系 / 帕累托前沿。

只做画像，不做显著性挖掘、不训练模型、不做 PnL。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.build_liquidity_lifecycle_v1 import (
    RESULTS,
)
from research.liquidity_oracle_atlas.build_oracle_atlas_v1 import (
    BIN_LABELS, RISK_ATR_GRID, SCOPES,
)

TWO_D = RESULTS / "two_dimensional_profiles"
TWO_D.mkdir(parents=True, exist_ok=True)


def dominance(g):
    L = g["long_best_R"].to_numpy(float)
    S = g["short_best_R"].to_numpy(float)
    if (L <= 0).all() and (S <= 0).all():
        return "NO_CLEAR_ACTION"
    if (L >= S).all() and (L > S).any():
        return "LONG_DOMINATES"
    if (S >= L).all() and (S > L).any():
        return "SHORT_DOMINATES"
    return "MIXED"


def pareto_of(g):
    """RR × holding bars 帕累托前沿：RR 更高且 bars 不长于，且至少一项严格。"""
    d = g[["risk_ATR", "conservative_best_R",
           "bars_to_best_conservative"]].dropna()
    if not len(d):
        return []
    pts = d.to_numpy(float)
    keep = []
    for i, p in enumerate(pts):
        dominated = False
        for j, q in enumerate(pts):
            if i == j:
                continue
            if (q[1] >= p[1] and q[2] <= p[2]
                    and (q[1] > p[1] or q[2] < p[2])):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return [tuple(pts[i]) for i in keep]


def q(x, p):
    return round(float(np.nanpercentile(x, p)), 4) if len(x) else np.nan


def main():
    M = pd.read_parquet(RESULTS / "liquidity_master_v1.parquet")
    C = pd.read_parquet(RESULTS / "liquidity_contacts_v1.parquet")
    S = pd.read_parquet(RESULTS / "liquidity_state_snapshot.parquet")
    O = pd.read_parquet(RESULTS / "oracle_risk_frontier.parquet")

    # ---------- sample funnel ----------
    funnel = [
        dict(stage="liquidity identity (VALID_AHEAD)", n=len(M)),
        dict(stage="liquidity with >=1 contact",
             n=int((M["n_contacts"] >= 1).sum())),
        dict(stage="liquidity consumed (首次严格穿透)",
             n=int(M["consumed"].sum())),
        dict(stage="contact events", n=len(C)),
    ]
    for ct in ["TOUCH_ONLY", "PENETRATE_RECLAIM", "PENETRATE_CLOSE_AT",
               "PENETRATE_CLOSE_BEYOND", "GAP_CROSS"]:
        funnel.append(dict(stage=f"contact_type={ct}",
                           n=int((C["contact_type"] == ct).sum())))
    pd.DataFrame(funnel).to_csv(RESULTS / "sample_funnel.csv", index=False,
                                encoding="utf-8-sig")

    # ---------- contact type / number ----------
    ct = (C.groupby("contact_type")
          .agg(n=("contact_number", "size"),
               pct_first_contact=("is_first_contact", "mean"))
          .reset_index())
    ct["pct_first_contact"] = ct["pct_first_contact"].round(4)
    ct.to_csv(RESULTS / "contact_type_profile.csv", index=False,
              encoding="utf-8-sig")

    cn = (C.groupby("contact_number")
          .agg(n=("liquidity_id", "size"))
          .reset_index().head(10))
    cn.to_csv(RESULTS / "contact_number_profile.csv", index=False,
              encoding="utf-8-sig")

    # 接触类型 × 接触次数
    a = (C.groupby(["contact_number", "contact_type"]).size()
         .rename("n").reset_index())
    a.to_csv(TWO_D / "contact_type_x_contact_number.csv", index=False,
             encoding="utf-8-sig")

    # ---------- Oracle 支配关系 ----------
    p = O.pivot_table(index=["symbol", "liquidity_id", "contact_number",
                             "risk_ATR"], columns="direction",
                      values="conservative_best_R", aggfunc="first")
    p = p.reset_index()
    p.columns = [str(c) for c in p.columns]
    p = p.rename(columns={"LONG": "long_best_R", "SHORT": "short_best_R"})
    lab = (p.groupby(["symbol", "liquidity_id", "contact_number"])
           .apply(dominance).rename("oracle_direction").reset_index())
    C2 = C.merge(lab, on=["symbol", "liquidity_id", "contact_number"],
                 how="left")
    C2.to_parquet(RESULTS / "liquidity_oracle_labels.parquet", index=False)

    # ---------- 帕累托前沿 ----------
    prows = []
    for (sym, lid, cno), g in O.groupby(["symbol", "liquidity_id",
                                         "contact_number"]):
        for d in ("LONG", "SHORT"):
            gd = g[g["direction"] == d]
            for r, rr, bars in pareto_of(gd):
                prows.append(dict(symbol=sym, liquidity_id=lid,
                                  contact_number=cno, direction=d,
                                  risk_ATR=r, best_R=round(float(rr), 4),
                                  bars_to_best=int(bars)))
    PF = pd.DataFrame(prows)
    PF.to_parquet(RESULTS / "oracle_pareto_frontier.parquet", index=False)

    # ---------- risk_ATR × RR / holding ----------
    rr_rows = []
    for d in ("LONG", "SHORT"):
        for r in RISK_ATR_GRID:
            v = O[(O.direction == d) & (O.risk_ATR == r)][
                "conservative_best_R"].to_numpy(float)
            b = O[(O.direction == d) & (O.risk_ATR == r)][
                "bars_to_best_conservative"].dropna().to_numpy(float)
            st = O[(O.direction == d) & (O.risk_ATR == r)]["stop_hit"]
            rr_rows.append(dict(
                direction=d, risk_ATR=r, n=len(v),
                mean_R=round(float(np.mean(v)), 4),
                p25_R=q(v, 25), median_R=q(v, 50), p75_R=q(v, 75),
                p90_R=q(v, 90),
                pct_zero_R=round(float((v <= 0).mean()), 4),
                pct_stopped=round(float(st.mean()), 4),
                median_bars_to_best=q(b, 50),
                p90_bars_to_best=q(b, 90)))
    RR = pd.DataFrame(rr_rows)
    RR.to_csv(RESULTS / "risk_atr_rr_profile.csv", index=False,
              encoding="utf-8-sig")
    RR.to_csv(RESULTS / "risk_atr_holding_profile.csv", index=False,
              encoding="utf-8-sig")

    # ---------- oracle direction profile ----------
    od = (C2["oracle_direction"].value_counts().rename("n").reset_index())
    od.columns = ["oracle_direction", "n"]
    od["share"] = (od["n"] / od["n"].sum()).round(4)
    od.to_csv(RESULTS / "oracle_direction_profile.csv", index=False,
              encoding="utf-8-sig")

    # ---------- trend joint ----------
    tj = []
    for a_, b_ in (("trend_struct_1h", "trend_struct_15m"),
                   ("trend_struct_15m", "trend_struct_5m")):
        t = (S.groupby([a_, b_]).size().rename("n").reset_index())
        t.columns = ["var_a", "var_b", "n"]
        t.insert(0, "pair", f"{a_} x {b_}")
        tj.append(t)
    TJ = pd.concat(tj, ignore_index=True)
    TJ.to_csv(RESULTS / "trend_joint_profile.csv", index=False,
              encoding="utf-8-sig")
    TJ.to_csv(TWO_D / "trend_1h_x_15m_and_15m_x_5m.csv", index=False,
              encoding="utf-8-sig")

    # ---------- liquidity field ----------
    F = pd.read_parquet(RESULTS / "liquidity_field_snapshot.parquet")
    fr = []
    for sc in SCOPES:
        for b in BIN_LABELS:
            c = f"{sc}_bin_{b}"
            if c in F.columns:
                fr.append(dict(scope=sc, bin=b,
                               mean_count=round(float(F[c].mean()), 4),
                               pct_nonzero=round(float((F[c] > 0).mean()), 4)))
    pd.DataFrame(fr).to_csv(RESULTS / "liquidity_field_profile.csv",
                            index=False, encoding="utf-8-sig")
    for v in ("nearest_above_R", "nearest_below_R", "nearest_ahead_R",
              "nearest_behind_R"):
        if v in F.columns:
            x = F[v].dropna().to_numpy(float)
            fr.append(dict(scope="ALL", bin=v, mean_count=q(x, 50),
                           pct_nonzero=round(float(F[v].notna().mean()), 4)))
    pd.DataFrame(fr).to_csv(RESULTS / "liquidity_field_profile.csv",
                            index=False, encoding="utf-8-sig")

    # overlap
    ov = (F["same_price_identity_count"].value_counts().rename("n")
          .reset_index())
    ov.columns = ["same_price_identity_count", "n"]
    ov["share"] = (ov["n"] / ov["n"].sum()).round(4)
    ov.to_csv(RESULTS / "liquidity_overlap_profile.csv", index=False,
              encoding="utf-8-sig")

    # ---------- OB field ----------
    obr = []
    for v in ("nearest_opposing_ob_distance_R", "nearest_opposing_ob_width_R",
              "nearest_same_direction_ob_distance_R"):
        if v in S.columns:
            x = S[v].dropna().to_numpy(float)
            obr.append(dict(variable=v, n=len(x), p10=q(x, 10), p25=q(x, 25),
                            median=q(x, 50), p75=q(x, 75), p90=q(x, 90),
                            missing_rate=round(float(S[v].isna().mean()), 4)))
    for v in ("nearest_opposing_ob_source_tf", "nearest_opposing_ob_freshness"):
        if v in S.columns:
            t = S[v].astype(str).value_counts(normalize=True).round(4)
            for k, val in t.items():
                obr.append(dict(variable=f"{v}={k}", n=None, p10=None,
                                p25=None, median=None, p75=None, p90=None,
                                missing_rate=float(val)))
    pd.DataFrame(obr).to_csv(RESULTS / "ob_field_profile.csv", index=False,
                             encoding="utf-8-sig")

    # ---------- 二维画像 ----------
    S2 = S.merge(C2[["liquidity_id", "contact_number", "oracle_direction"]],
                 on=["liquidity_id", "contact_number"], how="left")
    pairs = [("liquidity_scope", "contact_type"),
             ("liquidity_scope", "oracle_direction"),
             ("same_price_has_multi", "oracle_direction"),
             ("penetration_depth_bin", "oracle_direction"),
             ("nearest_ahead_R_bin", "oracle_direction"),
             ("nearest_opposing_ob_distance_R_bin", "oracle_direction")]
    S2["same_price_has_multi"] = np.where(
        pd.to_numeric(F["same_price_identity_count"], errors="coerce").values
        >= 2, "MULTI_IDENTITY", "SINGLE")
    S2["penetration_depth_bin"] = pd.cut(
        pd.to_numeric(S2["penetration_depth_R"], errors="coerce"),
        [-0.01, 0, 0.25, 0.5, 1.0, 2.0, np.inf])
    S2["nearest_ahead_R_bin"] = pd.cut(
        pd.to_numeric(S2["nearest_ahead_R"], errors="coerce"),
        [0, 0.5, 1, 2, 4, np.inf])
    S2["nearest_opposing_ob_distance_R_bin"] = pd.cut(
        pd.to_numeric(S2["nearest_opposing_ob_distance_R"], errors="coerce"),
        [0, 0.25, 0.5, 1, 2, 4, np.inf])
    for a_, b_ in pairs:
        if a_ not in S2.columns or b_ not in S2.columns:
            continue
        t = (S2.groupby([a_, b_], observed=True).size().rename("n")
             .reset_index())
        t.to_csv(TWO_D / f"{a_}_x_{b_}.csv", index=False,
                 encoding="utf-8-sig")

    # ---------- AUDIT ----------
    audit = dict(
        n_liquidity=int(len(M)),
        n_contacts=int(len(C)),
        n_consumed=int(M["consumed"].sum()),
        contacts_per_liquidity=round(float(len(C) / max(len(M), 1)), 4),
        contact_type_counts={k: int(v) for k, v in
                             C["contact_type"].value_counts().items()},
        contact_number_top5={int(k): int(v) for k, v in
                             C["contact_number"].value_counts()
                             .sort_index().head(5).items()},
        oracle_direction_counts={k: int(v) for k, v in
                                 C2["oracle_direction"].value_counts()
                                 .items()},
        risk_atr_grid=RISK_ATR_GRID,
        n_oracle_rows=int(len(O)),
        n_pareto_points=int(len(PF)),
        max_scan_bars=500,
    )
    json.dump(audit, open(RESULTS / "AUDIT_ORACLE_ATLAS_V1.json", "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print("\n=== risk_ATR × RR ===")
    print(RR.to_string(index=False))
    print("\nPROFILE_DONE")


if __name__ == "__main__":
    main()
