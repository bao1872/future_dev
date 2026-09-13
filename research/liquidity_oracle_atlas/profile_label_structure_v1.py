"""SMC Oracle 标签结构画像 v1.0 —— 读懂冻结后的 Oracle 标签内部结构。

不是修底座，不训练模型，不做 PnL。
输入：oracle_direction_stability_v1_2 / oracle_risk_direction_v1_2 /
     oracle_risk_frontier_v1_2 / liquidity_state_snapshot_v1_2。

设计原则（来自 IDE 指令）：
- NO_DIRECTION 拆成 NO_REACH / BALANCED_BIDIRECTIONAL / WEAK_OR_TIED
  （描述性拆分，非正式训练标签）
- ROBUST_LONG/SHORT 与趋势/liquidity side 做 native alignment
  （reversal = oracle 与 sweep 反向）
- RISK_DEPENDENT 看成风险方向序列，找单次切换与切换 ATR
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

RISK = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00]
RESULTS = Path(
    "research/analysis_results/smc_oracle_atlas_v1")
OUT = Path(
    "research/analysis_results/smc_oracle_label_structure_v1")
OUT.mkdir(parents=True, exist_ok=True)
TD = OUT / "two_dimensional_profiles"
TD.mkdir(exist_ok=True)


def _key(df):
    return df.set_index(["symbol", "liquidity_id", "contact_number"])


def main():
    stab = pd.read_parquet(
        RESULTS / "oracle_direction_stability_v1_2.parquet")
    rd = pd.read_parquet(
        RESULTS / "oracle_risk_direction_v1_2.parquet")
    fr = pd.read_parquet(
        RESULTS / "oracle_risk_frontier_v1_2.parquet")
    st = pd.read_parquet(
        RESULTS / "liquidity_state_snapshot_v1_2.parquet")
    out = {}

    # ---------- 0. 主表 join ----------
    M = stab.merge(st, on=["symbol", "liquidity_id", "contact_number"],
                   how="left")
    M["oracle_sign"] = M["direction_stability"].map(
        {"ROBUST_LONG": 1, "ROBUST_SHORT": -1}).fillna(0).astype(int)

    # ---------- 1. NO_DIRECTION 拆解 ----------
    # 从 frontier 聚合每个 contact 的 long/short best_R_lower（保守下界）
    L = fr[fr["direction"] == "LONG"][
        ["symbol", "liquidity_id", "contact_number", "risk_ATR",
         "best_R_lower"]].rename(columns={"best_R_lower": "lR"})
    S = fr[fr["direction"] == "SHORT"][
        ["symbol", "liquidity_id", "contact_number", "risk_ATR",
         "best_R_lower"]].rename(columns={"best_R_lower": "sR"})
    P = L.merge(S, on=["symbol", "liquidity_id", "contact_number",
                       "risk_ATR"], how="inner")
    P["lpos"] = P["lR"].gt(0).fillna(False)
    P["spos"] = P["sR"].gt(0).fillna(False)
    agg = P.groupby(["symbol", "liquidity_id", "contact_number"]).agg(
        n_long_pos=("lpos", "sum"),
        n_short_pos=("spos", "sum"),
        n_both_pos=("lpos", lambda x: (x & P.loc[x.index, "spos"]).sum()),
        n_both_zero=("lpos", lambda x: (~x & ~P.loc[x.index, "spos"]).sum()),
        n_risk=("lpos", "size")).reset_index()
    nd = stab[stab["direction_stability"] == "NO_DIRECTION"].merge(
        agg, on=["symbol", "liquidity_id", "contact_number"], how="inner")
    # 描述性拆分（非训练标签）
    def _nd_class(r):
        if r["n_both_pos"] == 0:
            return "NO_REACH"
        if r["n_long_dom"] == 0 and r["n_short_dom"] == 0:
            return "BALANCED_BIDIRECTIONAL"
        return "WEAK_OR_TIED"
    nd["nd_subtype"] = nd.apply(_nd_class, axis=1)
    nd_sub = nd["nd_subtype"].value_counts().rename("n").reset_index()
    nd_sub["share"] = (nd_sub["n"] / nd_sub["n"].sum()).round(4)
    nd_sub.to_csv(OUT / "no_direction_decomposition.csv", index=False,
                  encoding="utf-8-sig")
    print("=== NO_DIRECTION 拆解（n=%d）===" % len(nd))
    print(nd_sub.to_string(index=False))
    out["no_direction_subtype"] = nd_sub.to_dict("records")
    out["no_direction_total"] = int(len(nd))

    # ---------- 2. ROBUST 画像 ----------
    RB = M[M["direction_stability"].isin(
        ["ROBUST_LONG", "ROBUST_SHORT"])].copy()
    print("\n=== ROBUST 方向分布 ===")
    print(RB["direction_stability"].value_counts().to_string())

    def _dist(col, by):
        g = RB.groupby(by)[col]
        return pd.DataFrame({"mean": g.mean(), "median": g.median()}).round(4)

    prof = []
    for col in ["penetration_depth_R", "nearest_above_R", "nearest_below_R",
                "nearest_ahead_R", "nearest_behind_R",
                "nearest_opposing_ob_distance_R", "nearest_opposing_ob_width_R",
                "same_price_identity_count", "active_visible_count"]:
        if col in RB.columns:
            t = _dist(col, "direction_stability").reset_index()
            t["variable"] = col
            prof.append(t)
    # 类别型
    for col in ["contact_type", "liquidity_scope", "side",
                "trend_struct_1h", "trend_struct_15m", "trend_struct_5m",
                "env_direction_4h", "sweep_vs_1h", "sweep_vs_15m",
                "sweep_vs_5m", "env4h_vs_1h", "trend_1h_vs_15m",
                "trend_15m_vs_5m", "nearest_opposing_ob_source_tf",
                "nearest_opposing_ob_freshness"]:
        if col in RB.columns:
            t = (RB.groupby(["direction_stability", col]).size()
                 .rename("n").reset_index())
            t["variable"] = col
            prof.append(t)
    pd.concat(prof, ignore_index=True).to_csv(
        OUT / "robust_direction_profile.csv", index=False,
        encoding="utf-8-sig")
    print("\n[robust_direction_profile 已写]")

    # ---------- 3. native alignment ----------
    for tf in ("1h", "15m", "5m", "4h"):
        c = f"trend_struct_{tf}" if tf != "4h" else "env_direction_4h"
        if c in RB.columns:
            RB[f"align_{tf}"] = (
                (RB["oracle_sign"] == RB[c]).astype(int))
    RB["align_liquidity_side"] = (
        (RB["oracle_sign"] == RB["side"]).astype(int))
    RB["oracle_relation"] = np.where(
        RB["oracle_sign"] == RB["side"], "CONTINUATION",
        "REVERSAL")
    al = []
    for col in ["align_1h", "align_15m", "align_5m", "align_4h",
                "align_liquidity_side"]:
        al.append(dict(metric=col,
                       share_aligned=round(float(RB[col].mean()), 4),
                       n=len(RB)))
    pd.DataFrame(al).to_csv(
        OUT / "oracle_alignment_profile.csv", index=False,
        encoding="utf-8-sig")
    print("\n=== Oracle alignment（ROBUST 子集）===")
    print(pd.DataFrame(al).to_string(index=False))
    out["alignment"] = al

    # ---------- 4. reversal / continuation ----------
    rc = (RB.groupby(["oracle_relation", "direction_stability"]).size()
          .rename("n").reset_index())
    rc.to_csv(OUT / "reversal_continuation_profile.csv", index=False,
              encoding="utf-8-sig")
    # 分层
    rows = []
    for strat in ["trend_struct_1h", "trend_struct_15m", "contact_type",
                  "liquidity_scope", "contact_number", "side"]:
        for v, g in RB.groupby(strat):
            n = len(g)
            if n < 200:
                continue
            rows.append(dict(stratum=strat, value=str(v), n=n,
                            reversal_share=round(
                                float((g["oracle_relation"] ==
                                       "REVERSAL").mean()), 4),
                            long_share=round(
                                float((g["direction_stability"] ==
                                       "ROBUST_LONG").mean()), 4)))
    pd.DataFrame(rows).to_csv(
        OUT / "reversal_continuation_by_stratum.csv", index=False,
        encoding="utf-8-sig")
    print("\n=== REVERSAL vs CONTINUATION（ROBUST）===")
    print(rc.to_string(index=False))
    out["reversal_continuation"] = rc.to_dict("records")

    # ---------- 5. RISK_DEPENDENT 序列 ----------
    rd2 = rd.copy()
    mp = {"LONG_DOMINATES": "L", "SHORT_DOMINATES": "S",
          "TRADEOFF_OR_OVERLAP": "T", "UNRESOLVED_CENSOR": "U",
          "NO_COMPARABLE_TARGET": "N"}
    rd2["code"] = rd2["rr_direction"].map(mp)
    seq = (rd2.sort_values("risk_ATR").groupby(
        ["symbol", "liquidity_id", "contact_number"])["code"]
        .apply(lambda s: "".join(s)).reset_index())
    RDc = stab[stab["direction_stability"] == "RISK_DEPENDENT"].merge(
        seq, on=["symbol", "liquidity_id", "contact_number"], how="inner")
    seq_counts = RDc["code"].value_counts().rename("n").reset_index()
    seq_counts.columns = ["sequence", "n"]
    seq_counts.to_csv(OUT / "risk_direction_sequences.csv", index=False,
                      encoding="utf-8-sig")
    top = seq_counts.head(30)
    top.to_csv(OUT / "risk_direction_sequence_top.csv", index=False,
               encoding="utf-8-sig")
    print("\n=== RISK_DEPENDENT top 序列 ===")
    print(top.to_string(index=False))
    out["risk_dependent_sequences_top"] = top.to_dict("records")
    out["risk_dependent_n"] = int(len(RDc))

    # ---------- 6. 切换类型 ----------
    def _switch(seq):
        s = [c for c in seq if c in ("L", "S")]
        if len(s) < 2:
            return "INSUFFICIENT"
        switches = sum(1 for i in range(1, len(s))
                       if s[i] != s[i - 1])
        if switches == 0:
            return "NO_SWITCH"
        if switches > 1:
            return "MULTI_SWITCH"
        # 单次切换
        i = next(i for i in range(1, len(s)) if s[i] != s[i - 1])
        return ("SHORT_TO_LONG" if s[i - 1] == "S" else "LONG_TO_SHORT",
                i)
    sw_rows = []
    for _, r in RDc.iterrows():
        res = _switch(r["code"])
        if isinstance(res, tuple):
            typ, i = res
            lo, hi = RISK[i - 1], RISK[i]
            sw_rows.append(dict(
                symbol=r["symbol"], liquidity_id=r["liquidity_id"],
                contact_number=r["contact_number"], switch_type=typ,
                switch_ATR_low=lo, switch_ATR_high=hi,
                switch_ATR_midpoint=(lo + hi) / 2))
        else:
            sw_rows.append(dict(
                symbol=r["symbol"], liquidity_id=r["liquidity_id"],
                contact_number=r["contact_number"], switch_type=res,
                switch_ATR_low=np.nan, switch_ATR_high=np.nan,
                switch_ATR_midpoint=np.nan))
    SW = pd.DataFrame(sw_rows)
    SW.to_csv(OUT / "risk_switch_profile.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== 切换类型 ===")
    print(SW["switch_type"].value_counts().to_string())
    out["switch_type"] = SW["switch_type"].value_counts().to_dict()

    # ---------- 7. 切换 ATR vs 市场尺度 ----------
    SWm = SW[SW["switch_ATR_midpoint"].notna()].merge(
        M[["symbol", "liquidity_id", "contact_number",
           "penetration_depth_R", "nearest_ahead_R", "nearest_behind_R",
           "nearest_opposing_ob_distance_R"]],
        on=["symbol", "liquidity_id", "contact_number"], how="left")
    sc_rows = []
    for var in ["penetration_depth_R", "nearest_ahead_R",
                "nearest_behind_R", "nearest_opposing_ob_distance_R"]:
        for lab, g in SWm.groupby("switch_type"):
            if len(g) < 50:
                continue
            q = g[var].quantile([0.25, 0.5, 0.75]).round(3)
            sc_rows.append(dict(switch_type=lab, market_scale=var,
                                p25=q[0.25], median=q[0.5], p75=q[0.75],
                                n=len(g)))
    sc = pd.DataFrame(sc_rows)
    sc.to_csv(OUT / "risk_switch_scale_profile.csv", index=False,
               encoding="utf-8-sig")
    print("\n[risk_switch_scale_profile 已写]")

    # ---------- 8. 方向信息曲线 by risk ----------
    g = rd.groupby("risk_ATR")["rr_direction"].value_counts().unstack(
        fill_value=0)
    g = g.reindex(columns=["LONG_DOMINATES", "SHORT_DOMINATES",
                           "TRADEOFF_OR_OVERLAP", "UNRESOLVED_CENSOR",
                           "NO_COMPARABLE_TARGET"], fill_value=0)
    g["comparable"] = g["LONG_DOMINATES"] + g["SHORT_DOMINATES"] + \
        g["TRADEOFF_OR_OVERLAP"]
    g["directional_share"] = (
        (g["LONG_DOMINATES"] + g["SHORT_DOMINATES"]) /
        g["comparable"]).round(4)
    g = g.reset_index()
    g.to_csv(OUT / "directional_share_by_risk.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== 方向信息曲线 by risk ===")
    print(g[["risk_ATR", "LONG_DOMINATES", "SHORT_DOMINATES",
             "TRADEOFF_OR_OVERLAP", "UNRESOLVED_CENSOR",
             "NO_COMPARABLE_TARGET", "directional_share"]].to_string(
                 index=False))
    out["directional_share_by_risk"] = g.to_dict("records")

    # ---------- 9. 持有时间 by label ----------
    fr2 = fr.dropna(subset=["bars_to_best_lower"]).copy()
    fr2["bars_to_best_lower"] = fr2["bars_to_best_lower"].astype(float)
    ht = fr2.merge(
        M[["symbol", "liquidity_id", "contact_number",
           "direction_stability"]],
        on=["symbol", "liquidity_id", "contact_number"], how="left")
    rows = []
    for lab in ["ROBUST_LONG", "ROBUST_SHORT", "RISK_DEPENDENT",
                "NO_DIRECTION"]:
        sub = ht[ht["direction_stability"] == lab]
        for rk, g in sub.groupby("risk_ATR"):
            rows.append(dict(label=lab, risk_ATR=rk, n=len(g),
                             median_bars=float(g["bars_to_best_lower"].median()),
                             p75_bars=float(g["bars_to_best_lower"].quantile(0.75)),
                             p90_bars=float(g["bars_to_best_lower"].quantile(0.90))))
    pd.DataFrame(rows).to_csv(OUT / "holding_time_by_label.csv", index=False,
                              encoding="utf-8-sig")
    print("\n[holding_time_by_label 已写]")

    # ---------- 10. 二维画像 ----------
    RB_rel = RB.copy()
    two_d = []
    # oracle relation 二维表（ROBUST 子集）
    two_d.append(("trend1h_x_trend15m_x_relation",
                  RB_rel.groupby(["trend_struct_1h", "trend_struct_15m",
                                  "oracle_relation"]).size()))
    two_d.append(("side_x_trend1h_x_direction",
                  RB_rel.groupby(["side", "trend_struct_1h",
                                  "direction_stability"]).size()))
    two_d.append(("contact_type_x_relation",
                  RB_rel.groupby(["contact_type", "oracle_relation"]).size()))
    two_d.append(("contact_number_x_relation",
                  RB_rel.groupby(["contact_number", "oracle_relation"]).size()))
    two_d.append(("liquidity_scope_x_relation",
                  RB_rel.groupby(["liquidity_scope",
                                  "oracle_relation"]).size()))
    two_d.append(("ob_tf_x_relation",
                  RB_rel.groupby(["nearest_opposing_ob_source_tf",
                                  "oracle_relation"]).size()))
    # OB distance bin
    obd = pd.cut(RB_rel["nearest_opposing_ob_distance_R"],
                 [-np.inf, 0.25, 0.5, 1, 2, np.inf])
    two_d.append(("ob_distance_bin_x_relation",
                  RB_rel.groupby([obd, "oracle_relation"]).size()))
    # nearest ahead liquidity bin
    nh = pd.cut(RB_rel["nearest_ahead_R"],
                [-np.inf, 0.5, 1, 2, 4, np.inf])
    two_d.append(("nearest_ahead_bin_x_relation",
                  RB_rel.groupby([nh, "oracle_relation"]).size()))
    # raw overlap count
    ov = pd.cut(RB_rel["same_price_identity_count"],
                [-np.inf, 1, 2, 3, np.inf])
    two_d.append(("overlap_count_x_relation",
                  RB_rel.groupby([ov, "oracle_relation"]).size()))
    for name, ser in two_d:
        ser.reset_index().to_csv(
            TD / f"{name}.csv", index=False, encoding="utf-8-sig")

    # ---------- 11. 跨品种 / F1-F4 ----------
    # F1-F4：从 contacts 取 fold（如不可用仅做 symbol 层）
    try:
        con = pd.read_parquet(
            RESULTS / "liquidity_contacts_v1_1.parquet")
        M = M.merge(con[["symbol", "liquidity_id", "contact_number", "fold"]],
                    on=["symbol", "liquidity_id", "contact_number"],
                    how="left")
        RB2 = M[M["direction_stability"].isin(
            ["ROBUST_LONG", "ROBUST_SHORT"])]
        fr_tab = (RB2.groupby("fold")["direction_stability"]
                  .apply(lambda s: (s == "ROBUST_LONG").mean()).round(4))
        fr_tab.to_csv(OUT / "robust_direction_by_fold.csv")
        out["robust_by_fold"] = fr_tab.to_dict()
    except Exception as e:
        print("fold 层跳过：", e)

    sym_tab = (RB.groupby("symbol")["direction_stability"]
               .apply(lambda s: (s == "ROBUST_LONG").mean()).round(4))
    out["robust_long_share_by_symbol"] = {
        k: float(v) for k, v in sym_tab.items()}
    out["robust_long_share_symbol_median"] = float(sym_tab.median())
    out["robust_long_share_positive_symbols"] = int((sym_tab > 0.5).sum())

    json.dump(out, open(OUT / "PROFILE_AUDIT.json", "w"), indent=2,
              ensure_ascii=False)
    print("\nPROFILE_V1_DONE")


if __name__ == "__main__":
    main()
