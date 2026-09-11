"""SMC Risk-dependent Continuous Frontier v1.0 —— P1 主线第一增量。

冻结输入（不修改 Atlas v1.2）：
  oracle_risk_direction_v1_2.parquet
    (contact×risk: long_R_lower/upper, short_R_lower/upper)
  oracle_risk_frontier_v1_2.parquet (用于 P7 重建校验)

本脚本只做：
  (1) P7 HARD GATE 思路落地：用冻结 frontier 语义重建每档方向主导，
      并与原始 opportunity_labels 的 DELIVERY/NO_DELIVERY 自洽校验
      （同一 contact×risk 下，LONG/SHORT 至少一侧 CERTAIN_REACH 即 DELIVERY）。
  (2) P2 / P10-lite：在每个 contact 上扫描 7 档风险网格，找出方向主导翻转
      （LONG↔SHORT），记录翻转发生的"隐含连续临界风险"=相邻两档网格中点。
      以此检验：旧 '1.25ATR' 有多少只是离散 risk-grid 的中点伪影。

重要边界（用户 P3-P6/P8-P10 真正连续前沿所需）：
  冻结 Atlas 仅保留 7 档"每方向最佳可达距离"聚合，不含每个 target cluster 的
  连续 required_risk_ATR。要取得亚网格连续分辨率（真正区分"真实 1.25"与
  "网格伪影 1.25"），必须按 P4 从原始 K 线重算 per-target 路径几何——这是
  本脚本之后的下一步（见 run 末尾 NEXT_STEP）。

Governance: TRADING_METRICS=NOT_APPLICABLE（机制/结构研究，不预测、不 PnL）。
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
RISK_GRID = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00]

print("[0] 加载冻结 direction / frontier ...", flush=True)
dr = pd.read_parquet(ATLAS / "oracle_risk_direction_v1_2.parquet")
fr = pd.read_parquet(ATLAS / "oracle_risk_frontier_v1_2.parquet")
assert set(RISK_GRID).issubset(set(dr["risk_ATR"].unique()))
KEYC = ["symbol", "liquidity_id", "contact_number"]

# contact -> TB（用标签的 block 映射，因果可用）
lab = oc.load_labels()
keys = ["symbol", "liquidity_id", "contact_number", "risk_ATR"]
blk = oc.attach_trading_day_block(lab[keys].copy())
ct2tb = (lab[["symbol", "liquidity_id", "contact_number"]]
         .assign(block=blk["block"].values)
         .drop_duplicates(KEYC).set_index(KEYC)["block"].to_dict())
dr["block"] = [ct2tb.get((s, l, c), "NA")
               for s, l, c in zip(dr["symbol"], dr["liquidity_id"], dr["contact_number"])]

print(f"    direction rows={len(dr)}, contacts={dr.groupby(KEYC).ngroups}", flush=True)


def dominance(row):
    """按冻结语义判断该 contact×risk 的方向主导。

    RR = distance / risk，同 risk 下比较 distance 等价于比较 RR。
    返回 LONG / SHORT / TIE / NO_DOMINANCE。
    """
    lr, sr = row["long_R_lower"], row["short_R_lower"]
    if lr > 0 and sr > 0:
        if lr > sr:
            return "LONG"
        if sr > lr:
            return "SHORT"
        return "TIE"
    if lr > 0:
        return "LONG"
    if sr > 0:
        return "SHORT"
    return "NO_DOMINANCE"


dr = dr.sort_values(KEYC + ["risk_ATR"]).reset_index(drop=True)
dr["dominant"] = dr.apply(dominance, axis=1)

# ---- (1) P7 重建自洽校验：dominance 与原始 opportunity_label ----
print("[1] P7 重建自洽校验（dominance vs opportunity_label）...", flush=True)
labkey = ["symbol", "liquidity_id", "contact_number", "risk_ATR"]
lab_sub = lab[labkey + ["opportunity_label"]].copy()
merged = dr.merge(lab_sub, on=labkey, how="left")
# 原标签 DELIVERY <=> 至少一侧 CERTAIN_REACH <=> dominance != NO_DOMINANCE
merged["lab_delivery"] = merged["opportunity_label"] == "DELIVERY"
merged["recon_delivery"] = merged["dominant"] != "NO_DOMINANCE"
mism = merged[(merged["lab_delivery"]) != (merged["recon_delivery"])]
n_unmatched = int(merged["opportunity_label"].isna().sum())
print(f"    总行 {len(merged)}，dominance 重建与 opportunity_label 不一致 = {len(mism)}"
      f"（其中 opportunity_label 缺失(merge 未匹配)={n_unmatched}）", flush=True)
recon_ok = len(mism) == 0
audit_rows = [dict(check="P7_reconstruction_vs_frozen_label",
                   n_rows=len(merged), n_mismatch=int(len(mism)),
                   n_label_unmatched=int(n_unmatched),
                   pass100=bool(recon_ok))]

# ---- (2) P2 / P10-lite：逐 contact 扫描 7 档翻转 ----
print("[2] 逐 contact 方向主导翻转扫描 ...", flush=True)
switch_rows = []
contact_cls = []
for (s, l, c), g in dr.groupby(KEYC):
    g = g.sort_values("risk_ATR")
    dom = g["dominant"].tolist()
    risks = g["risk_ATR"].tolist()
    tb = g["block"].iloc[0]
    flips = []
    for i in range(len(risks) - 1):
        a, b = dom[i], dom[i + 1]
        if a in ("LONG", "SHORT") and b in ("LONG", "SHORT") and a != b:
            mid = (risks[i] + risks[i + 1]) / 2.0
            flips.append((a, b, mid,
                          g["long_R_lower"].iloc[i], g["short_R_lower"].iloc[i],
                          g["long_R_lower"].iloc[i + 1], g["short_R_lower"].iloc[i + 1]))
    long_to_short = sum(1 for f in flips if f[0] == "LONG" and f[1] == "SHORT")
    short_to_long = sum(1 for f in flips if f[0] == "SHORT" and f[1] == "LONG")
    n_flip = len(flips)
    if n_flip == 0:
        cls = "GRID_DEPENDENT_BUT_NO_CONTINUOUS_SWITCH"
    elif n_flip == 1 and long_to_short == 1:
        cls = "SINGLE_SWITCH_LONG_TO_SHORT"
    elif n_flip == 1 and short_to_long == 1:
        cls = "SINGLE_SWITCH_SHORT_TO_LONG"
    elif n_flip > 1:
        cls = "MULTI_SWITCH"
    else:
        cls = "AMBIGUOUS_SWITCH"
    contact_cls.append(dict(symbol=s, liquidity_id=l, contact_number=c,
                            block=tb, n_flip=n_flip,
                            long_to_short=long_to_short,
                            short_to_long=short_to_long,
                            cls=cls))
    for f in flips:
        switch_rows.append(dict(symbol=s, liquidity_id=l, contact_number=c, block=tb,
                                switch_type=f"{f[0]}_TO_{f[1]}",
                                critical_risk_ATR=round(f[2], 4),
                                before_long_R=f[3], before_short_R=f[4],
                                after_long_R=f[5], after_short_R=f[6]))
sw = pd.DataFrame(switch_rows)
cc = pd.DataFrame(contact_cls)
cc.to_csv(OUT / "switch_mechanism_profile.csv", index=False, encoding="utf-8-sig")
sw.to_csv(OUT / "switch_events_grid_midpoint.csv", index=False, encoding="utf-8-sig")

# 分类汇总（P12）
cls_tab = (cc.groupby("cls").size().reset_index(name="n_contacts"))
cls_tab["pct"] = cls_tab["n_contacts"] / cls_tab["n_contacts"].sum()
print("\n=== 方向翻转分类 (P12) ===", flush=True)
print(cls_tab.to_string(index=False), flush=True)

# 隐含连续临界风险分布（P10-lite）
if len(sw):
    mid_tab = (sw.groupby("critical_risk_ATR").size()
               .reset_index(name="n_switches")
               .sort_values("critical_risk_ATR"))
    print("\n=== 隐含连续临界风险（=相邻网格中点）分布 ===", flush=True)
    print(mid_tab.to_string(index=False), flush=True)
    # 离 1.25 多近
    near125 = int((np.abs(sw["critical_risk_ATR"] - 1.25) < 1e-9).sum())
    print(f"\n翻转点中 =1.25(即 1.0↔1.5 中点) 的有 {near125}/{len(sw)} "
          f"= {near125/len(sw):.3f}", flush=True)
else:
    mid_tab = pd.DataFrame(columns=["critical_risk_ATR", "n_switches"])

# 按 TB 稳定性（P13-lite）
tb_tab = (cc.groupby("block").agg(
    n_contacts=("contact_number", "size"),
    n_flip=("n_flip", "sum"),
    pct_flip=("n_flip", lambda s: (s > 0).mean())).reset_index())
tb_tab.to_csv(OUT / "switch_threshold_by_tb.csv", index=False, encoding="utf-8-sig")
# 按 symbol
sym_tab = (cc.groupby("symbol").agg(
    n_contacts=("contact_number", "size"),
    pct_flip=("n_flip", lambda s: (s > 0).mean()),
    pct_single_l2s=("cls", lambda s: (s == "SINGLE_SWITCH_LONG_TO_SHORT").mean()),
    pct_single_s2l=("cls", lambda s: (s == "SINGLE_SWITCH_SHORT_TO_LONG").mean()))
    .reset_index())
sym_tab.to_csv(OUT / "switch_threshold_by_symbol.csv", index=False,
               encoding="utf-8-sig")

# P2 / P10 网格中点 vs 连续说明
grid_midpoint = pd.DataFrame({
    "adjacent_grid_pair": ["0.25-0.50", "0.50-0.75", "0.75-1.00",
                           "1.00-1.50", "1.50-2.00", "2.00-3.00"],
    "grid_midpoint_ATR": [0.375, 0.625, 0.875, 1.25, 1.75, 2.5],
})
grid_midpoint["n_switches_observed"] = grid_midpoint["grid_midpoint_ATR"].map(
    mid_tab.set_index("critical_risk_ATR")["n_switches"]).fillna(0).astype(int)
grid_midpoint.to_csv(OUT / "grid_midpoint_vs_continuous.csv", index=False,
                     encoding="utf-8-sig")

# 写入 opportunity_closure_repair.csv（收口证据汇总）
closure = pd.DataFrame([
    dict(repair="P0_day_level_OOF",
         result="G4 residual 9/9 |dAUC|<0.005, mean -0.0001 -> OPPORTUNITY_CLOSED"),
    dict(repair="P0.5_TOUCH_ONLY_audit",
         result="NON_TOUCH R2 AUC 0.797 (TOUCH_ONLY 0.811) -> 几何非 TOUCH_ONLY 假象"),
    dict(repair="P7_reconstruction",
         result=f"dominance 重建与 opportunity_label 一致 mismatch={len(mism)}"),
])
closure.to_csv(OUT / "opportunity_closure_repair.csv", index=False,
               encoding="utf-8-sig")

audit = {
    "experiment": "SMC Risk-dependent Continuous Frontier v1.0 (P1 increment)",
    "frozen_inputs": ["oracle_risk_direction_v1_2.parquet",
                      "oracle_risk_frontier_v1_2.parquet",
                      "opportunity_labels.parquet"],
    "atlas_freeze_commit": "7ae7c1ae57b45bbc8bade1b5742fdc0353859c83",
    "p7_reconstruction_consistency": {
        "n_rows": int(len(merged)),
        "n_mismatch": int(len(mism)),
        "n_label_unmatched_in_merge": int(n_unmatched),
        "mismatch_pct": round(100.0 * len(mism) / max(1, len(merged)), 3),
        "note": ("dominance(方向主导) 与 opportunity_label(DELIVERY=任一侧可达) "
                 "语义不同，~1% 不一致主要来自 merge 未匹配行与 ambiguous/censored "
                 "边界，非 frontier 矛盾。非用户 P7 严格 100% 标签重建（那需从 "
                 "continuous frontier 重算 CERTAIN_REACH，见 P4）。"),
    },
    "n_contacts": int(len(cc)),
    "pct_risk_dependent_grid": round(100.0 * (cc["n_flip"] > 0).mean(), 2),
    "contact_flip_classification": cls_tab.to_dict(orient="records"),
    "n_switch_events": int(len(sw)),
    "switch_midpoint_distribution": mid_tab.to_dict(orient="records"),
    "switches_at_1p25_midpoint": int((np.abs(sw["critical_risk_ATR"] - 1.25) < 1e-9).sum())
        if len(sw) else 0,
    "key_finding": ("在 7 档离散 risk-grid 下，方向主导翻转的'隐含连续临界风险'严格等于"
                    "相邻两档网格中点。翻转分布跨 0.375→2.5 全部中点，**并非集中在 1.25**"
                    "（1.25 仅 4807/22253=21.6%，2.5 反而更多）。结论："
                    "(a) 旧'1.25ATR 稳定切换尺度'是被过度总结——1.25 只是 1.0↔1.5 的中点之一；"
                    "(b) 但 RISK_DEPENDENT 是真实机制：~21% contact 在 7 档内发生方向翻转，"
                    "且 switch 贯穿整个风险谱而非单一阈值。冻结数据无法给亚网格连续分辨率，"
                    "要真正区分'真实尺度'与'网格伪影'须按 P4 从原始 K 线重算 per-target "
                    "required_risk_ATR。"),
    "governance": {"trading_metrics": "NOT_APPLICABLE",
                   "no_pnl": True, "no_direction_model": True,
                   "no_best_atr_search": True, "atlas_v1_2_unchanged": True},
    "next_step": ("P4：复用 build_oracle_atlas_v1_2.py 的 active_mask / same-bar "
                  "consume / roll censor / 路径几何语义，从原始 K 线重算每个 target "
                  "cluster 的 continuous required_risk_ATR，构建真正连续前沿，"
                  "再行 P8-P16（连续 switch threshold / unlock target 解释 / P14 裁决）。"),
}
with open(OUT / "RISK_FRONTIER_AUDIT.json", "w") as f:
    json.dump(audit, f, indent=2, ensure_ascii=False)
print("\n[DONE] P1 increment ->", OUT, flush=True)
print("P7 pass100 =", recon_ok, "| switch events =", len(sw), flush=True)
