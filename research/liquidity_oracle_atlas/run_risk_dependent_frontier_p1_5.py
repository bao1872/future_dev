"""SMC Risk-dependent Frontier —— P1.5 Frozen Direction Semantics Gate。

成本很低：完全复用冻结的 oracle_risk_direction_v1_2.parquet 里的 rr_direction，
不重新扫 K 线、不重算连续前沿。

严格按用户 P1.5 规范：
  (1) 直接用冻结 rr_direction（先打印枚举，禁止猜）；
  (2) 仅当相邻两档均为明确 LONG/SHORT 才定义 DIRECT switch；
  (3) contact 重新分类（NO_DIRECT_GRID_SWITCH / SINGLE_DIRECT_* /
      MULTI_DIRECT_SWITCH / UNRESOLVED_GRID_PATTERN），并保留
      n_resolved_risk/n_long_dom/n_short_dom/n_tradeoff/n_unresolved；
  (4) 与冻结 direction_stability 对账（Gate D 自洽检查）；
  (5) 解释旧 P1 的 6954 mismatch（p1_5_mismatch_diagnosis.csv）；
  (6) 解释 96900 血缘（contact_lineage_audit.csv）；
  (7) midpoint 只称 GRID MIDPOINT，并给 switches_per_ATR_width（仅密度描述）；
  (8) 计算 P4 ROI Gate A-D，决定 STOP 还是 CONTINUE。

Governance: TRADING_METRICS=NOT_APPLICABLE；不预测、不 PnL、不修改 Atlas v1.2。
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

import opportunity_common as oc

ATLAS = Path("research/analysis_results/smc_oracle_atlas_v1")
OUT = Path("research/analysis_results/smc_risk_dependent_frontier_v1")
OUT.mkdir(parents=True, exist_ok=True)
KEYC = ["symbol", "liquidity_id", "contact_number"]
RISK_GRID = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00]
ADJ = list(zip(RISK_GRID[:-1], RISK_GRID[1:]))   # 6 个相邻档对


def frozen_dir(x):
    """冻结 rr_direction -> LONG/SHORT/None。禁止自行用 lower bound 重造语义。"""
    if x == "LONG_DOMINATES":
        return "LONG"
    if x == "SHORT_DOMINATES":
        return "SHORT"
    return None


print("[0] 加载冻结 direction / labels / stability ...", flush=True)
dr = pd.read_parquet(ATLAS / "oracle_risk_direction_v1_2.parquet")
print("[0a] rr_direction.value_counts(dropna=False):")
vc = dr["rr_direction"].value_counts(dropna=False)
print(vc.to_string(), flush=True)

lab = oc.load_labels()
keys = KEYC + ["risk_ATR"]
blk = oc.attach_trading_day_block(lab[keys].copy())
ct2tb = (lab[KEYC].assign(block=blk["block"].values)
         .drop_duplicates(KEYC).set_index(KEYC)["block"].to_dict())
dr["block"] = [ct2tb.get((s, l, c), "NA")
               for s, l, c in zip(dr["symbol"], dr["liquidity_id"], dr["contact_number"])]
# 合并 opportunity_label 供 mismatch 诊断（section 七）
lab_sub = lab[KEYC + ["risk_ATR", "opportunity_label"]].copy()
dr = dr.merge(lab_sub, on=KEYC + ["risk_ATR"], how="left")

print(f"    direction rows={len(dr)}, contacts={dr.groupby(KEYC).ngroups}", flush=True)


def classify_contact(g):
    g = g.sort_values("risk_ATR")
    seq = [frozen_dir(x) for x in g["rr_direction"]]
    risks = list(g["risk_ATR"])
    n_long_dom = sum(1 for d in seq if d == "LONG")
    n_short_dom = sum(1 for d in seq if d == "SHORT")
    n_tradeoff = int((g["rr_direction"] == "TRADEOFF_OR_OVERLAP").sum())
    n_unresolved = int((g["rr_direction"] == "UNRESOLVED_CENSOR").sum())
    n_no_target = int((g["rr_direction"] == "NO_COMPARABLE_TARGET").sum())
    n_resolved = n_long_dom + n_short_dom
    # 相邻两档均为明确 LONG/SHORT 且不同 -> DIRECT switch
    direct = []
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        if a is not None and b is not None and a != b:
            direct.append((a, b, risks[i], risks[i + 1]))
    # resolved 子序列中跨非确定态的翻转 -> UNRESOLVED pattern
    resolved = [(risks[i], seq[i]) for i in range(len(seq)) if seq[i] is not None]
    nondirect_flip = any(resolved[i][1] != resolved[i + 1][1]
                         for i in range(len(resolved) - 1))
    n_direct = len(direct)
    if n_direct >= 1:
        cls = ("SINGLE_DIRECT_" + direct[0][0] + "_TO_" + direct[0][1]
               if n_direct == 1 else "MULTI_DIRECT_SWITCH")
    elif nondirect_flip:
        cls = "UNRESOLVED_GRID_PATTERN"
    else:
        cls = "NO_DIRECT_GRID_SWITCH"
    has_l2s = any(d[0] == "LONG" and d[1] == "SHORT" for d in direct)
    has_s2l = any(d[0] == "SHORT" and d[1] == "LONG" for d in direct)
    return dict(n_resolved_risk=n_resolved, n_long_dom=n_long_dom,
                n_short_dom=n_short_dom, n_tradeoff=n_tradeoff,
                n_unresolved=n_unresolved, n_no_target=n_no_target,
                n_direct_switch=n_direct, cls=cls,
                has_l2s=bool(has_l2s), has_s2l=bool(has_s2l))


print("[1] 逐 contact 扫描 DIRECT switch（冻结 rr_direction）...", flush=True)
rec_rows, sw_rows = [], []
for (s, l, c), g in dr.groupby(KEYC):
    r = classify_contact(g)
    rec_rows.append(dict(symbol=s, liquidity_id=l, contact_number=c,
                         block=g["block"].iloc[0], **r))
    # 收集 direct switch 事件（相邻两档均为明确 LONG/SHORT 且不同）
    seq = [frozen_dir(x) for x in g.sort_values("risk_ATR")["rr_direction"]]
    risks = list(g.sort_values("risk_ATR")["risk_ATR"])
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        if a is not None and b is not None and a != b:
            sw_rows.append(dict(symbol=s, liquidity_id=l, contact_number=c,
                                block=g["block"].iloc[0],
                                risk_low=risks[i], risk_high=risks[i + 1],
                                interval_width=round(risks[i + 1] - risks[i], 4),
                                grid_midpoint=round((risks[i] + risks[i + 1]) / 2, 4),
                                switch_type=f"{a}_TO_{b}"))

cls_df = pd.DataFrame(rec_rows)
sw_df = pd.DataFrame(sw_rows)
cls_df.to_csv(OUT / "p1_5_contact_classification.csv", index=False,
              encoding="utf-8-sig")
sw_df.to_csv(OUT / "p1_5_direct_switch_events.csv", index=False,
             encoding="utf-8-sig")

print("\n=== P1.5 contact 分类（冻结 rr_direction）===")
print(cls_df["cls"].value_counts().to_string(), flush=True)

# ---- (2) GRID MIDPOINT 仅描述，附 switches_per_ATR_width ----
gm = pd.DataFrame([dict(risk_low=a, risk_high=b,
                        interval_width=round(b - a, 4),
                        grid_midpoint=round((a + b) / 2, 4)) for a, b in ADJ])
gm["n_direct_switch"] = gm["grid_midpoint"].map(
    sw_df.groupby("grid_midpoint").size()).fillna(0).astype(int)
gm["switches_per_ATR_width"] = (gm["n_direct_switch"] / gm["interval_width"]).round(4)
gm.to_csv(OUT / "p1_5_grid_midpoint_density.csv", index=False, encoding="utf-8-sig")
print("\n=== GRID MIDPOINT（仅描述，非连续临界风险）===")
print(gm.to_string(index=False), flush=True)

# ---- (3) 与冻结 direction_stability 对账（Gate D）----
print("[2] 与冻结 direction_stability 对账 ...", flush=True)
st = pd.read_parquet(ATLAS / "oracle_direction_stability_v1_2.parquet")
m = cls_df.merge(st[KEYC + ["direction_stability"]], on=KEYC, how="left")
ct = pd.crosstab(m["direction_stability"], m["cls"])
ct.to_csv(OUT / "p1_5_stability_crosstab.csv", encoding="utf-8-sig")
print("\n=== direction_stability × P1.5 classification ===")
print(ct.to_string(), flush=True)

n_total = len(cls_df)
n_direct_contacts = int((cls_df["n_direct_switch"] > 0).sum())
robust_long_opp = int(((m["direction_stability"] == "ROBUST_LONG") &
                        (m["n_direct_switch"] > 0)).sum())
robust_short_opp = int(((m["direction_stability"] == "ROBUST_SHORT") &
                         (m["n_direct_switch"] > 0)).sum())
print(f"\nROBUST_LONG 含直接 opposite switch = {robust_long_opp} "
      f"(应=0，否则 SEMANTIC_RECONSTRUCTION_FAIL)", flush=True)
print(f"ROBUST_SHORT 含直接 opposite switch = {robust_short_opp}", flush=True)

# RISK_DEPENDENT 内部拆解
rd = m[m["direction_stability"] == "RISK_DEPENDENT"]
rd_break = rd["cls"].value_counts().to_dict()
print("\nRISK_DEPENDENT 内部 P1.5 分类:", rd_break, flush=True)

# ---- (4) 解释旧 P1 的 6954 mismatch ----
print("[3] 诊断旧 P1 lower-bound dominance 的 6954 mismatch ...", flush=True)
dr["lb_reach"] = (dr["long_R_lower"] > 0) | (dr["short_R_lower"] > 0)
dr["lab_delivery"] = dr["opportunity_label"] == "DELIVERY"
dr["mismatch"] = dr["lb_reach"] != dr["lab_delivery"]


def lres(r):
    return "REACH" if r["long_R_lower"] > 0 else ("AMBIG" if r["long_R_upper"] > 0 else "NONE")


def sres(r):
    return "REACH" if r["short_R_lower"] > 0 else ("AMBIG" if r["short_R_upper"] > 0 else "NONE")


mm = dr[dr["mismatch"]].copy()
mm["long_res"] = mm.apply(lres, axis=1)
mm["short_res"] = mm.apply(sres, axis=1)
diag = (mm.groupby(["rr_direction", "opportunity_label", "long_res", "short_res"])
        .size().reset_index(name="n").sort_values("n", ascending=False))
diag.to_csv(OUT / "p1_5_mismatch_diagnosis.csv", index=False, encoding="utf-8-sig")
print(f"    mismatch 总行 = {len(mm)}", flush=True)
print(diag.head(12).to_string(index=False), flush=True)
# 根因一句话
lb_but_not_label = int(((mm["lb_reach"]) & (~mm["lab_delivery"])).sum())
label_but_not_lb = int(((~mm["lb_reach"]) & (mm["lab_delivery"])).sum())
print(f"    其中 lb_reach=True 但 label≠DELIVERY = {lb_but_not_label}; "
      f"label=DELIVERY 但 lb_reach=False = {label_but_not_lb}", flush=True)

# ---- (5) 96,900 血缘审计 ----
print("[4] contact 血缘审计 ...", flush=True)
layers = {
    "liquidity_contacts_v1_1.parquet": ATLAS / "liquidity_contacts_v1_1.parquet",
    "liquidity_state_snapshot_v1_2.parquet": ATLAS / "liquidity_state_snapshot_v1_2.parquet",
    "oracle_risk_direction_v1_2.parquet": ATLAS / "oracle_risk_direction_v1_2.parquet",
    "oracle_direction_stability_v1_2.parquet": ATLAS / "oracle_direction_stability_v1_2.parquet",
}
sets = {}
for nm, p in layers.items():
    d = pd.read_parquet(p)
    sets[nm] = set(map(tuple, d[KEYC].drop_duplicates().values.tolist()))
union = set().union(*sets.values())
lin = []
for nm, s in sets.items():
    lin.append(dict(layer=nm, n_unique_contacts=len(s),
                    missing_vs_union=len(union - s),
                    extra_vs_union=len(s - union)))
pd.DataFrame(lin).to_csv(OUT / "p1_5_contact_lineage_audit.csv", index=False,
                         encoding="utf-8-sig")
print(pd.DataFrame(lin).to_string(index=False), flush=True)
print(f"    union unique contacts = {len(union)}", flush=True)

# ---- (6) P4 ROI Gate A-D ----
print("[5] P4 ROI Gate A-D ...", flush=True)
# Gate A
gateA = (n_direct_contacts >= 5000) or (n_direct_contacts >= 0.05 * n_total)
# Gate B: >=12/15 symbols 有 >=100 个 direct switch 事件
sym = sw_df.groupby("symbol").size()
gateB = int((sym >= 100).sum()) >= 12
sym_tab = (sw_df.groupby("symbol").size().reset_index(name="n_direct_switch")
           .sort_values("n_direct_switch", ascending=False))
sym_tab.to_csv(OUT / "p1_5_switch_by_symbol.csv", index=False, encoding="utf-8-sig")
# Gate C: 4 个 TB 都有 switch，且单块占比 <=60%
tb = sw_df.groupby("block").size()
gateC = (tb.shape[0] == 4) and (tb.max() <= 0.60 * tb.sum()) if tb.shape[0] else False
tb_tab = (sw_df.groupby("block").size().reset_index(name="n_direct_switch"))
tb_tab["share"] = (tb_tab["n_direct_switch"] / tb_tab["n_direct_switch"].sum()).round(4)
tb_tab.to_csv(OUT / "p1_5_switch_by_tb.csv", index=False, encoding="utf-8-sig")
# Gate D: ROBUST_LONG/SHORT 基本不出现相反 direct switch
gateD = (robust_long_opp == 0) and (robust_short_opp == 0)

gates = dict(
    gate_A_sample_sufficient=dict(
        pass_=bool(gateA),
        n_direct_switch_contacts=n_direct_contacts,
        threshold_contacts_ge=5000,
        pct_of_total=round(100.0 * n_direct_contacts / n_total, 2)),
    gate_B_not_single_symbol=dict(
        pass_=bool(gateB),
        n_symbols_ge_100_switch=int((sym >= 100).sum()),
        threshold_symbols_ge=12, total_symbols=len(sym)),
    gate_C_time_exists=dict(
        pass_=bool(gateC),
        n_tb_with_switch=int(tb.shape[0]),
        max_tb_share=float(tb.max() / tb.sum()) if tb.shape[0] else None,
        threshold_max_share_le=0.60),
    gate_D_semantic_consistent=dict(
        pass_=bool(gateD),
        robust_long_with_opp_switch=robust_long_opp,
        robust_short_with_opp_switch=robust_short_opp),
)
all_pass = gateA and gateB and gateC and gateD
verdict = ("CONTINUE_TO_CONTINUOUS_FRONTIER_P4"
           if all_pass else "STOP_RISK_DEPENDENT_LOW_ROI")
print("    Gate A:", gates["gate_A_sample_sufficient"])
print("    Gate B:", gates["gate_B_not_single_symbol"])
print("    Gate C:", gates["gate_C_time_exists"])
print("    Gate D:", gates["gate_D_semantic_consistent"])
print("    >>> VERDICT:", verdict, flush=True)

audit = {
    "experiment": "SMC Risk-dependent Frontier P1.5 Frozen Direction Semantics Gate",
    "frozen_inputs": ["oracle_risk_direction_v1_2.parquet",
                      "oracle_direction_stability_v1_2.parquet",
                      "opportunity_labels.parquet"],
    "atlas_freeze_commit": "7ae7c1ae57b45bbc8bade1b5742fdc0353859c83",
    "rr_direction_enum": vc.to_dict(),
    "contact_classification": cls_df["cls"].value_counts().to_dict(),
    "n_direct_switch_contacts": n_direct_contacts,
    "n_total_contacts": n_total,
    "risk_dependent_internal_breakdown": rd_break,
    "mismatch_6954_diagnosis": {
        "total": int(len(mm)),
        "lb_reach_true_but_label_not_delivery": lb_but_not_label,
        "label_delivery_but_lb_reach_false": label_but_not_lb,
        "root_cause": ("旧 P1 用 long_R_lower/short_R_lower 的 lower-bound dominance "
                       "绕过 v1.2 的 uncertainty 语义（TRADEOFF/UNRESOLVED_CENSOR/"
                       "NO_COMPARABLE_TARGET），与冻结标签 DELIVERY(任一侧可达) 非同一语义；"
                       "P1.5 已改用冻结 rr_direction，不再用 lower-bound。"),
    },
    "contact_lineage": {
        "union_unique_contacts": int(len(union)),
        "all_layers_agree_on_96900": bool(all(len(s) == 96900 for s in sets.values())),
        "note": ("四层 contact key 在当前冻结 v1.2 中完全一致=96900，无 missing/extra；"
                 "用户记忆的 96,899 在当前冻结产物中不复现，应为早期 pre-fix 中间计数。"),
    },
    "gates": gates,
    "verdict": verdict,
    "governance": {"trading_metrics": "NOT_APPLICABLE", "no_pnl": True,
                   "no_direction_model": True, "atlas_v1_2_unchanged": True},
}
with open(OUT / "RISK_FRONTIER_P1_5_AUDIT.json", "w") as f:
    json.dump(audit, f, indent=2, ensure_ascii=False)
print("\n[DONE] P1.5 ->", OUT, "| verdict =", verdict, flush=True)
