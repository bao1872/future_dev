"""SMC Structural Delivery Opportunity Study v1.0 —— Stage 0: Outcome Atlas.

只做标签构造与结果画像，不训练、不做 PnL、不修改 Atlas v1.2。

核心语义纠正（用户协议 §一）：
- 旧画像字段 ``NO_REACH`` 实际定义是 ``n_both_pos == 0``，即
  “七档 ATR 中没有任何一档出现 Long 与 Short 双向同时到达 target”。
  它**不**等于“没有 delivery / 没有可交易路径”。
- 本轮新报告中旧字段改称 ``LEGACY_NO_BIDIRECTIONAL_REACH``，
  且严格不再解释为“没有 delivery”。

真正的 Structural Delivery Opportunity 标签（per contact × risk_ATR）：
- CERTAIN_REACH(LONG/SHORT):  best_R_lower > 0
- CERTAIN_NO_REACH:           resolution_class == EXACT_RESOLVED
                              AND best_R_lower == 0 AND best_R_upper == 0
- UNCERTAIN:                  CENSORED_LOWER_BOUND 或 AMBIGUOUS_INTERVAL
                              （best_R_lower == 0，不得当 negative）
- NO_ACTIVE_TARGET:           resolution_class == NO_ACTIVE_TARGET（单独画像）

配对 LONG/SHORT 得 opportunity_label:
- NO_TARGET_ENVIRONMENT: 任一侧 NO_ACTIVE_TARGET（target availability 不足）
- DELIVERY:                LONG CERTAIN_REACH 或 SHORT CERTAIN_REACH
- NO_DELIVERY:            LONG CERTAIN_NO_REACH 且 SHORT CERTAIN_NO_REACH
- UNRESOLVED:             其余（至少一侧 UNCERTAIN）

Primary prediction = DELIVERY vs NO_DELIVERY，仅用 primary_eligible
（两侧均有 active target）样本。

输出：research/analysis_results/smc_opportunity_v1/
  opportunity_labels.parquet
  opportunity_outcome_by_risk.csv
  legacy_vs_true_delivery.csv
  delivery_curve_by_risk.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ATLAS = Path("research/analysis_results/smc_oracle_atlas_v1")
OUT = Path("research/analysis_results/smc_opportunity_v1")
OUT.mkdir(parents=True, exist_ok=True)

RISK_GRID = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00]
KEYC = ["symbol", "liquidity_id", "contact_number", "risk_ATR"]

print("[0] load frontier (Atlas v1.2, frozen) ...")
fr = pd.read_parquet(ATLAS / "oracle_risk_frontier_v1_2.parquet")
assert set(RISK_GRID).issubset(set(fr["risk_ATR"].unique())), "risk grid 不完整"
assert fr["direction"].isin(["LONG", "SHORT"]).all()
print(f"    frontier rows={len(fr)}  "
      f"resolution_class={fr['resolution_class'].value_counts().to_dict()}")


def flag_direction(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    out["certain_reach"] = out["best_R_lower"] > 0
    out["certain_no_reach"] = (
        (out["resolution_class"] == "EXACT_RESOLVED")
        & (out["best_R_lower"] == 0)
        & (out["best_R_upper"] == 0)
    )
    out["uncertain"] = out["resolution_class"].isin(
        ["CENSORED_LOWER_BOUND", "AMBIGUOUS_INTERVAL"]) & (out["best_R_lower"] == 0)
    out["no_active_target"] = out["resolution_class"] == "NO_ACTIVE_TARGET"
    return out


print("[1] 每方向 reach 判定 ...")
L = flag_direction(fr[fr["direction"] == "LONG"])[
    KEYC + ["certain_reach", "certain_no_reach", "uncertain",
            "no_active_target", "n_targets"]].rename(
    columns={c: f"{c}_L" for c in
             ["certain_reach", "certain_no_reach", "uncertain",
              "no_active_target", "n_targets"]})
S = flag_direction(fr[fr["direction"] == "SHORT"])[
    KEYC + ["certain_reach", "certain_no_reach", "uncertain",
            "no_active_target", "n_targets"]].rename(
    columns={c: f"{c}_S" for c in
             ["certain_reach", "certain_no_reach", "uncertain",
              "no_active_target", "n_targets"]})
P = L.merge(S, on=KEYC, how="inner")
print(f"    paired contact×risk rows={len(P)} (=96900×7={96900 * 7})")

print("[2] 构造 opportunity_label + 6-path ...")
P["has_active_L"] = ~P["no_active_target_L"]
P["has_active_S"] = ~P["no_active_target_S"]
P["primary_eligible"] = P["has_active_L"] & P["has_active_S"]
P["delivery"] = P["certain_reach_L"] | P["certain_reach_S"]
P["no_delivery"] = P["certain_no_reach_L"] & P["certain_no_reach_S"]


def _label(r):
    if not (r["has_active_L"] and r["has_active_S"]):
        return "NO_TARGET_ENVIRONMENT"
    if r["delivery"]:
        return "DELIVERY"
    if r["no_delivery"]:
        return "NO_DELIVERY"
    return "UNRESOLVED"


def _path6(r):
    if not (r["has_active_L"] and r["has_active_S"]):
        return "NO_TARGET_ENVIRONMENT"
    if r["certain_reach_L"] and r["certain_reach_S"]:
        return "BOTH_DIRECTION_DELIVERY"
    if r["certain_reach_L"]:
        return "LONG_ONLY_DELIVERY"
    if r["certain_reach_S"]:
        return "SHORT_ONLY_DELIVERY"
    if r["no_delivery"]:
        return "NO_DELIVERY"
    return "UNRESOLVED"


P["opportunity_label"] = P.apply(_label, axis=1)
P["outcome_path"] = P.apply(_path6, axis=1)

# ----------------------------------------------------------------------
# §三 测试（协议 §二十六 1-7,10,11）
# ----------------------------------------------------------------------
print("[3] 运行标签逻辑测试 ...")
errs = []


def chk(cond, msg):
    if not cond:
        errs.append(msg)


# 1. legacy NO_REACH 不再等于 NO_DELIVERY（见 §六 cross-tab）
subL = flag_direction(fr[fr["direction"] == "LONG"])
subS = flag_direction(fr[fr["direction"] == "SHORT"])
# 2. best_R_lower>0 一定判 CERTAIN_REACH
chk((subL["certain_reach"] == (subL["best_R_lower"] > 0)).all(),
    "T2: best_R_lower>0 必判 CERTAIN_REACH 失败")
# 3. CENSORED_LOWER_BOUND 且 lower==0 的不得被误判为 CERTAIN_NO_REACH
cen = subL[(subL["resolution_class"] == "CENSORED_LOWER_BOUND")
           & (subL["best_R_lower"] == 0)]
chk((~cen["certain_no_reach"]).all(),
    "T3: CENSORED_LOWER_BOUND(lower==0) 不得判 CERTAIN_NO_REACH")
# 4. AMBIGUOUS_INTERVAL 且 lower==0 的不得被误判为 CERTAIN_NO_REACH
amb = subL[(subL["resolution_class"] == "AMBIGUOUS_INTERVAL")
           & (subL["best_R_lower"] == 0)]
chk(((amb["best_R_lower"] == 0) & (~amb["certain_no_reach"])).all(),
    "T4: AMBIGUOUS_INTERVAL(lower==0) 不得判 CERTAIN_NO_REACH")
# 5. NO_ACTIVE_TARGET 不得当 R=0 / 不得进 primary
chk((P.loc[~P["primary_eligible"], "opportunity_label"]
     == "NO_TARGET_ENVIRONMENT").all(),
    "T5: 至少一侧 NO_ACTIVE_TARGET 必为 NO_TARGET_ENVIRONMENT")
# 6. NO_DELIVERY 要求两侧都确定 no-reach
nd = P[P["opportunity_label"] == "NO_DELIVERY"]
chk((nd["certain_no_reach_L"] & nd["certain_no_reach_S"]).all(),
    "T6: NO_DELIVERY 必须两侧 CERTAIN_NO_REACH")
# 7. DELIVERY 只需一侧 certain reach
dl = P[P["opportunity_label"] == "DELIVERY"]
chk((dl["certain_reach_L"] | dl["certain_reach_S"]).all(),
    "T7: DELIVERY 必须至少一侧 CERTAIN_REACH")
# 10. decision-time overlap 与 pre-contact overlap 分开（本阶段无 pre 字段，记录）
# 11. active target geometry 只使用 active liquidity（B8 后续阶段，本阶段用 n_targets）
chk(P["opportunity_label"].isin(
    ["DELIVERY", "NO_DELIVERY", "UNRESOLVED", "NO_TARGET_ENVIRONMENT"]).all(),
    "T: opportunity_label 取值非法")
if errs:
    raise SystemExit("标签测试失败:\n  - " + "\n  - ".join(errs))
print("    全部标签测试通过 ✓")

# ----------------------------------------------------------------------
# §五 Stage 0 输出
# ----------------------------------------------------------------------
print("[4] opportunity_outcome_by_risk.csv (4-class + 6-path) ...")
four = (P.groupby(["risk_ATR", "opportunity_label"]).size()
        .reset_index(name="n"))
four["share_of_risk"] = four.groupby("risk_ATR")["n"].transform(
    lambda s: s / s.sum())
six = (P.groupby(["risk_ATR", "outcome_path"]).size()
       .reset_index(name="n"))
six["share_of_risk"] = six.groupby("risk_ATR")["n"].transform(
    lambda s: s / s.sum())
out_by_risk = four.merge(
    six, on=["risk_ATR"], suffixes=("_4class", "_6path"))
out_by_risk.to_csv(OUT / "opportunity_outcome_by_risk.csv",
                   index=False, encoding="utf-8-sig")

print("[5] delivery_curve_by_risk.csv ...")
curve = []
for r, g in P.groupby("risk_ATR"):
    pe = g[g["primary_eligible"]]
    n_all = len(g)
    n_pe = len(pe)
    n_del = int((pe["opportunity_label"] == "DELIVERY").sum())
    n_nd = int((pe["opportunity_label"] == "NO_DELIVERY").sum())
    n_un = int((pe["opportunity_label"] == "UNRESOLVED").sum())
    n_nt = int((g["opportunity_label"] == "NO_TARGET_ENVIRONMENT").sum())
    curve.append(dict(
        risk_ATR=r,
        n_total=n_all,
        n_primary_eligible=n_pe,
        n_NO_TARGET_ENVIRONMENT=n_nt,
        P_DELIVERY=(n_del / n_pe) if n_pe else np.nan,
        P_NO_DELIVERY=(n_nd / n_pe) if n_pe else np.nan,
        P_UNRESOLVED=(n_un / n_pe) if n_pe else np.nan,
        base_delivery_rate=(n_del / n_pe) if n_pe else np.nan,
        P_NO_TARGET_ENV_overall=(n_nt / n_all),
    ))
curve_df = pd.DataFrame(curve).sort_values("risk_ATR")
curve_df.to_csv(OUT / "delivery_curve_by_risk.csv",
                index=False, encoding="utf-8-sig")

print("[6] legacy_vs_true_delivery.csv ...")
# 联系级别 n_both_pos：多少 risk 档下 LONG 与 SHORT 同时 CERTAIN_REACH
P["both_reach"] = P["certain_reach_L"] & P["certain_reach_S"]
both_per_contact = P.groupby(
    ["symbol", "liquidity_id", "contact_number"])["both_reach"].sum()
legacy = (both_per_contact == 0)                     # 无条件 n_both_pos==0
# 条件口径：原始 86% 画像是在 direction_stability==NO_DIRECTION 子集内算 n_both_pos==0
stab = pd.read_parquet(ATLAS / "oracle_direction_stability_v1_2.parquet")
is_no_dir = (stab.set_index(
    ["symbol", "liquidity_id", "contact_number"])["direction_stability"]
    == "NO_DIRECTION")
legacy_cond = legacy & is_no_dir.reindex(legacy.index).fillna(False)

contact_meta = (legacy.rename("legacy_no_bidir_reach").to_frame()
                .join(legacy_cond.rename("legacy_no_bidir_reach_cond")))
contact_meta = contact_meta.reset_index()
P = P.merge(contact_meta, on=["symbol", "liquidity_id", "contact_number"],
            how="left")
print(f"    无条件 LEGACY_NO_BIDIRECTIONAL_REACH 联系 = "
      f"{int(legacy.sum())}/{len(legacy)} ({legacy.mean():.4f})")
print(f"    条件口径(NO_DIRECTION & n_both_pos==0) 联系 = "
      f"{int(legacy_cond.sum())}/{len(legacy)} "
      f"({legacy_cond.mean():.4f})  <- 对应原始画像 ~86%")


def cohort_crosstab(mask, cohort_name):
    leg = P[mask]
    ct = (leg.groupby(["risk_ATR", "outcome_path"]).size()
          .reset_index(name="n"))
    ct["share_within_cohort"] = ct.groupby("risk_ATR")["n"].transform(
        lambda s: s / s.sum())
    ct["cohort"] = cohort_name
    return ct


ct = pd.concat([
    cohort_crosstab(P["legacy_no_bidir_reach_cond"], "conditional_NO_DIRECTION"),
    cohort_crosstab(P["legacy_no_bidir_reach"], "unconditional"),
]).sort_values(["cohort", "risk_ATR", "outcome_path"]).reset_index(drop=True)
ct.to_csv(OUT / "legacy_vs_true_delivery.csv", index=False,
          encoding="utf-8-sig")


def summarize_cohort(mask, name):
    sub = P[mask]
    lc = (sub.groupby(["symbol", "liquidity_id", "contact_number"])
          .apply(lambda g: pd.Series({
              "n_LONG_ONLY": int((g["outcome_path"] == "LONG_ONLY_DELIVERY").sum()),
              "n_SHORT_ONLY": int((g["outcome_path"] == "SHORT_ONLY_DELIVERY").sum()),
              "n_BOTH": int((g["outcome_path"] == "BOTH_DIRECTION_DELIVERY").sum()),
              "n_NO_DELIVERY": int((g["outcome_path"] == "NO_DELIVERY").sum()),
              "n_UNRESOLVED": int((g["outcome_path"] == "UNRESOLVED").sum()),
              "n_NO_TARGET": int((g["outcome_path"] == "NO_TARGET_ENVIRONMENT").sum()),
          })).reset_index())
    if len(lc) == 0:
        print(f"    [{name}] 空队列，跳过")
        return {}
    lc["has_single_sided_delivery"] = (
        (lc["n_LONG_ONLY"] > 0) | (lc["n_SHORT_ONLY"] > 0))
    lc["truly_no_delivery_all_risk"] = (lc["n_NO_DELIVERY"] == 7)
    lc["mixed_unresolved"] = (
        (~lc["has_single_sided_delivery"])
        & (~lc["truly_no_delivery_all_risk"]))
    s = dict(
        n_legacy_contacts=int(len(lc)),
        n_has_single_sided_delivery=int(lc["has_single_sided_delivery"].sum()),
        n_truly_no_delivery_all_risk=int(lc["truly_no_delivery_all_risk"].sum()),
        n_mixed_unresolved=int(lc["mixed_unresolved"].sum()),
    )
    s["pct_has_single_sided_delivery"] = (
        s["n_has_single_sided_delivery"] / s["n_legacy_contacts"])
    s["pct_truly_no_delivery_all_risk"] = (
        s["n_truly_no_delivery_all_risk"] / s["n_legacy_contacts"])
    print(f"    [{name}] 联系数={s['n_legacy_contacts']}")
    print(f"      有单边 delivery（至少一档）: {s['n_has_single_sided_delivery']} "
          f"({s['pct_has_single_sided_delivery']:.4f})")
    print(f"      七档全 NO_DELIVERY: {s['n_truly_no_delivery_all_risk']} "
          f"({s['pct_truly_no_delivery_all_risk']:.4f})")
    print(f"      其余(unresolved/no-target): {s['n_mixed_unresolved']}")
    return s


print("    —— legacy 联系内真实构成（关键纠正） ——")
summ_cond = summarize_cohort(P["legacy_no_bidir_reach_cond"],
                             "conditional_NO_DIRECTION(~86% 原 cohort)")
summ_uncond = summarize_cohort(P["legacy_no_bidir_reach"], "unconditional")

# ----------------------------------------------------------------------
# 保存标签 parquet
# ----------------------------------------------------------------------
print("[7] opportunity_labels.parquet ...")
keep = KEYC + ["certain_reach_L", "certain_reach_S",
               "certain_no_reach_L", "certain_no_reach_S",
               "uncertain_L", "uncertain_S",
               "no_active_target_L", "no_active_target_S",
               "has_active_L", "has_active_S", "primary_eligible",
               "delivery", "no_delivery",
               "n_targets_L", "n_targets_S",
               "opportunity_label", "outcome_path",
               "legacy_no_bidir_reach"]
P[keep].to_parquet(OUT / "opportunity_labels.parquet", index=False)

# 汇总打印
print("\n=== Stage 0 结果速览 ===")
print(curve_df.to_string(index=False))
print(f"\nlegacy(旧NO_REACH≈86% cohort) 联系数={int(legacy_cond.sum())}, "
      f"其中真正七档全 NO_DELIVERY={summ_cond['n_truly_no_delivery_all_risk']} "
      f"({summ_cond['pct_truly_no_delivery_all_risk']:.4f})")
print(f"无条件 legacy 联系数={int(legacy.sum())}, "
      f"七档全 NO_DELIVERY={summ_uncond['n_truly_no_delivery_all_risk']} "
      f"({summ_uncond['pct_truly_no_delivery_all_risk']:.4f})")
print("[DONE] Stage 0 输出目录:", OUT)
