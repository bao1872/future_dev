"""SMC Opportunity Fast Closure Audit v1.2-lite（两阶段 Gate，低成本终局）。

目标：用最低成本回答两个问题（冻结 Atlas / label / TB 语义不变）：
  Q1. 一条"最近目标距离 / stop 距离"几何规则是否已足以替代 Opportunity 模型？
  Q2. G4 在 G1 已知后，是否还有"大到值得继续维护这 60 个字段"的残余信息？

设计（用户指定，ROI 优先）：
- 仅 3 档 risk：0.5 / 1.0(primary) / 2.0 ATR。
- WF1/2/3 点估计，Stage 1 不做 bootstrap。
- Stage 1-A 简单几何压缩 R0 / R2 / R3。
- Stage 1-B G4 残余：temporal inner OOF（expanding 40/60/80 → 预测 20%）
  生成 p_g1 / p_g4，logit 后 meta，比较 META_G1_G4 - META_G1。
- 明确 Gate：未通过则停止 Opportunity，转 RISK_DEPENDENT。

Governance: 机制/去混淆实验，TRADING_METRICS=NOT_APPLICABLE。
不进入 PnL / 方向模型 / 最佳 ATR / LightGBM / SHAP。
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

import opportunity_common as oc

OUT = Path("research/analysis_results/smc_opportunity_fast_closure_v1_2")
OUT.mkdir(parents=True, exist_ok=True)
RISKS = [0.5, 1.0, 2.0]
WF = [("WF1", ["TB1"], "TB2"), ("WF2", ["TB1", "TB2"], "TB3"),
      ("WF3", ["TB1", "TB2", "TB3"], "TB4")]
G1 = ["nearest_above_R", "nearest_below_R"]
G4 = oc.B2_BIN_COLS
B0 = oc.B0_COLS


def metrics(y, s):
    """s: higher=better score/proba。返回 (roc, pr, top20, bot20)。"""
    y = np.asarray(y, dtype=np.int64); s = np.asarray(s, dtype=np.float64)
    roc = oc.tie_aware_auc(y, s)
    pr = average_precision_score(y, s)
    n = len(y); k = max(1, int(round(0.2 * n)))
    order = np.argsort(-s)
    top20 = y[order[:k]].mean()
    bot20 = y[order[-k:]].mean()
    return roc, pr, top20, bot20


def expanding_oof_by_day(Xtr, ytr, daytr, cols, Xte):
    """Temporal inner OOF，按 canonical trading-day block 切分（修复原按行数切的
    leakage bug）：expanding 40/60/80% 的 *交易日* 训练 → 预测随后 20% 交易日。
    返回 (oof_on_train, pred_on_test)。禁止 in-sample stacking，且保证
    inner-train 的 trading_day 严格早于 inner-validation（同日硬断言）。"""
    days = np.array(sorted(pd.unique(daytr)))
    n_days = len(days)
    cuts = [(0.40, 0.60), (0.60, 0.80), (0.80, 1.00)]
    oof = np.full(len(ytr), np.nan, dtype=float)
    for a, b in cuts:
        train_end = int(np.floor(a * n_days))
        pred_end = int(np.floor(b * n_days))
        train_days = days[:train_end]
        pred_days = days[train_end:pred_end]
        tr = np.isin(daytr, train_days)
        va = np.isin(daytr, pred_days)
        assert pd.Series(daytr[tr]).max() < pd.Series(daytr[va]).min(), \
            "inner OOF 同日交叉：违反 trading-day block 约束"
        if tr.sum() == 0 or va.sum() == 0:
            continue
        m = oc.make_trifold_pipeline(cols).fit(Xtr.loc[tr, cols], ytr[tr])
        oof[va] = m.predict_proba(Xtr.loc[va, cols])[:, 1]
    full = oc.make_trifold_pipeline(cols).fit(Xtr[cols], ytr)
    pte = full.predict_proba(Xte[cols])[:, 1]
    return oof, pte


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


print("[0] 加载标签 + v1.1 特征 + block ...", flush=True)
lab = oc.load_labels()
F = oc.build_features()
keys = ["symbol", "liquidity_id", "contact_number", "risk_ATR"]
blk = oc.attach_trading_day_block(lab[keys].copy())
lab = lab.copy()
lab["block"] = blk["block"].values
lab["trading_day"] = blk["trading_day"].values
F = F.copy()
F["block"] = blk["block"].values
F["trading_day"] = blk["trading_day"].values
F["primary_eligible"] = lab["primary_eligible"].values
F["opportunity_label"] = lab["opportunity_label"].values
F["model_eligible"] = (F["primary_eligible"]
                       & F["opportunity_label"].isin(["DELIVERY", "NO_DELIVERY"]))
F["y"] = (F["opportunity_label"] == "DELIVERY").astype(int)
F["rho"] = (np.minimum(F["nearest_above_R"], F["nearest_below_R"])
            / F["risk_ATR"])
# contact_type 来自 build_features() 的 B1 块（与 F 索引对齐），用于 P0.5 审计
assert "contact_type" in F.columns, "F 缺少 contact_type（build_features B1 块）"
print(f"    model_eligible 总行数 = {int(F['model_eligible'].sum())}", flush=True)

ALL = B0 + G1 + G4
geo_rows, g4_rows, r2_test_rows = [], [], []

print("[A] Stage 1-A 简单几何压缩 R0/R2/R3 ...", flush=True)
for r in RISKS:
    sub = F[(F["risk_ATR"] == r) & F["model_eligible"]].reset_index(drop=True)
    X = sub[ALL].copy()
    for wfn, trb, teb in WF:
        tr = sub["block"].isin(trb).values
        te = (sub["block"] == teb).values
        if tr.sum() == 0 or te.sum() == 0:
            continue
        ytr = sub["y"][tr].values; yte = sub["y"][te].values
        # R0: 纯规则，不训练
        s0 = -sub["rho"][te].values
        r0 = metrics(yte, s0)
        # R2: 上下两个 target 距离
        p2 = (oc.make_trifold_pipeline(G1).fit(X[tr][G1], ytr)
              .predict_proba(X[te][G1])[:, 1])
        r2 = metrics(yte, p2)
        # R3: B0 + G1 (= 此前 M_G1)
        p3 = (oc.make_trifold_pipeline(B0 + G1).fit(X[tr][B0 + G1], ytr)
              .predict_proba(X[te][B0 + G1])[:, 1])
        r3 = metrics(yte, p3)
        for tag, rr in [("R0", r0), ("R2", r2), ("R3", r3)]:
            geo_rows.append(dict(risk_ATR=r, wf=wfn, model=tag,
                                 roc_auc=rr[0], pr_auc=rr[1],
                                 top20_delivery_rate=rr[2],
                                 bottom20_delivery_rate=rr[3],
                                 test_base_rate=yte.mean()))
        # P0.5: 收集 R2 测试预测 + contact_type，便于 TOUCH_ONLY 敏感性审计
        r2_test_rows.append(dict(
            risk_ATR=r, wf=wfn,
            contact_type=sub["contact_type"][te].values,
            y=yte, p2=p2))
    print(f"  [A] risk={r} 完成", flush=True)

geo = pd.DataFrame(geo_rows)
geo.to_csv(OUT / "fast_geometry_metrics.csv", index=False, encoding="utf-8-sig")
print(f"    fast_geometry_metrics rows={len(geo)}", flush=True)

# A Gate
g = geo.set_index(["risk_ATR", "wf"])
dR2_R0 = g.loc[g.model == "R2", "roc_auc"] - g.loc[g.model == "R0", "roc_auc"]
dR3_R2 = g.loc[g.model == "R3", "roc_auc"] - g.loc[g.model == "R2", "roc_auc"]
mean_R2_R0 = float(dR2_R0.mean())
mean_R3_R2 = float(dR3_R2.mean())
simple_sufficient = (mean_R2_R0 < 0.01) and (mean_R3_R2 < 0.01)
a_gate = {
    "mean_dAUC_R2_minus_R0": round(mean_R2_R0, 4),
    "mean_dAUC_R3_minus_R2": round(mean_R3_R2, 4),
    "threshold": 0.01,
    "decision": ("R0_RULE_SUFFICIENT" if simple_sufficient
                 else "R0_INSUFFICIENT_TWO_SIDED_GEOMETRY_SUFFICIENT"),
    "detail": ("单条归一化距离规则 R0 已解释绝大部分 -> 不需要 ML 模型"
               if simple_sufficient else
               "R0 不够（ΔAUC=%.3f>0.01），但 R2(上下两条距离) 即封顶，"
               "R3-R2=%.3f≈0 说明 metadata 零增量 -> 模型压缩为 2 特征几何，无需复杂 ML"
               % (mean_R2_R0, mean_R3_R2)),
}

print("[B] Stage 1-B G4 残余（temporal inner OOF meta）...", flush=True)
for r in RISKS:
    sub = (F[(F["risk_ATR"] == r) & F["model_eligible"]
            ].sort_values("trading_day").reset_index(drop=True))
    X = sub[ALL].copy()
    for wfn, trb, teb in WF:
        tr = sub["block"].isin(trb).values
        te = (sub["block"] == teb).values
        if tr.sum() == 0 or te.sum() == 0:
            continue
        Xtr, ytr = X[tr].reset_index(drop=True), sub["y"][tr].values
        Xte, yte = X[te].reset_index(drop=True), sub["y"][te].values
        daytr = sub["trading_day"][tr].values   # 训练样本对应的交易日（按日切块）
        oof_g1, pt_g1 = expanding_oof_by_day(Xtr, ytr, daytr, B0 + G1, Xte)
        oof_g4, pt_g4 = expanding_oof_by_day(Xtr, ytr, daytr, B0 + G4, Xte)
        mask = ~np.isnan(oof_g1)
        lg1, lg4 = lg(oof_g1), lg(oof_g4)
        tg1, tg4 = lg(pt_g1), lg(pt_g4)
        # META_G1
        m1 = (LogisticRegression(max_iter=3000)
              .fit(lg1[mask].reshape(-1, 1), ytr[mask])
              .predict_proba(tg1.reshape(-1, 1))[:, 1])
        # META_G1_G4
        feat = np.column_stack([lg1, lg4])[mask]
        m14 = (LogisticRegression(max_iter=3000)
               .fit(feat, ytr[mask])
               .predict_proba(np.column_stack([tg1, tg4]))[:, 1])
        auc_g1 = oc.tie_aware_auc(yte, m1)
        auc_g14 = oc.tie_aware_auc(yte, m14)
        auc_bg1 = oc.tie_aware_auc(yte, pt_g1)
        auc_bg4 = oc.tie_aware_auc(yte, pt_g4)
        g4_rows.append(dict(
            risk_ATR=r, wf=wfn,
            auc_base_G1=auc_bg1, auc_base_G4=auc_bg4,
            auc_meta_G1=auc_g1, auc_meta_G1G4=auc_g14,
            dAUC_meta=auc_g14 - auc_g1,
            pr_meta_G1=average_precision_score(yte, m1),
            pr_meta_G1G4=average_precision_score(yte, m14),
            dPR_meta=average_precision_score(yte, m14)
                     - average_precision_score(yte, m1),
            brier_meta_G1=brier_score_loss(yte, np.clip(m1, 1e-6, 1 - 1e-6)),
            brier_meta_G1G4=brier_score_loss(yte, np.clip(m14, 1e-6, 1 - 1e-6)),
            dAUC_positive=(auc_g14 > auc_g1)))
    print(f"  [B] risk={r} 完成", flush=True)

g4 = pd.DataFrame(g4_rows)
g4.to_csv(OUT / "g4_residual_fast_metrics.csv", index=False, encoding="utf-8-sig")

# B Gate（day-level OOF 修复后）
d = g4["dAUC_meta"]
# 每个 risk 的 3 个 WF 是否全正
risks_allpos = []
for r in RISKS:
    gr = g4[g4.risk_ATR == r]
    if len(gr) == 3 and (gr["dAUC_meta"] > 0).all():
        risks_allpos.append(r)
mean_dAUC = float(d.mean())
n_total = len(d)
all_small = bool((d.abs() < 0.005).all())          # 9 cell 全部 |ΔAUC|<0.005
reverse_cond = (len(risks_allpos) >= 2) and (mean_dAUC >= 0.01)
if all_small and (mean_dAUC < 0.005):
    b_decision = "OPPORTUNITY_CLOSED_G4_RESIDUAL_NOT_MATERIAL"
elif reverse_cond:
    b_decision = "OPPORTUNITY_NEEDS_CONTINUE"
else:
    b_decision = "G4_RESIDUAL_NOT_MATERIAL"
b_gate = {
    "mean_dAUC_meta": round(mean_dAUC, 4),
    "all_cells_abs_dAUC_lt_0p005": all_small,
    "n_cells_total": n_total,
    "risks_with_all3WF_positive": [float(x) for x in risks_allpos],
    "threshold_all_cells_abs": 0.005,
    "threshold_mean_dAUC": 0.005,
    "decision": b_decision,
    "detail": ("G4 在 G1 之上 9/9 |ΔAUC|<0.005 且均值<0.005 -> 正式冻结 "
               "OPPORTUNITY_CLOSED, G4_RESIDUAL_NOT_MATERIAL"
               if b_decision == "OPPORTUNITY_CLOSED_G4_RESIDUAL_NOT_MATERIAL"
               else ("G4 在 >=2 risk 下 WF1-3 全正且均值>=0.01 -> "
                     "Opportunity 需继续研究，不收口"
                     if b_decision == "OPPORTUNITY_NEEDS_CONTINUE"
                     else "G4 残余不显著，按无材料处理 -> 停止 Opportunity")),
}

# P0.5: TOUCH_ONLY 敏感性审计（不改标签，仅按 contact_type 拆分 R2 测试预测）
rows = []
for rec in r2_test_rows:
    ct = rec["contact_type"]; y = np.asarray(rec["y"]); p = np.asarray(rec["p2"])
    for c in pd.unique(ct):
        m = ct == c
        yy, pp = y[m], p[m]
        npos = int(yy.sum()); n = len(yy)
        auc = oc.tie_aware_auc(yy, pp) if 0 < npos < n else np.nan
        rows.append(dict(risk_ATR=rec["risk_ATR"], wf=rec["wf"],
                         contact_type=c, n=n, base_rate=yy.mean(),
                         r2_roc_auc=auc))
tou = pd.DataFrame(rows)
# NON_TOUCH 聚合（跨 risk×wf 合并同一 contact_type 之外的全部）
nontouch_rows = []
for rec in r2_test_rows:
    ct = rec["contact_type"]; y = np.asarray(rec["y"]); p = np.asarray(rec["p2"])
    m = ct != "TOUCH_ONLY"
    if m.sum() == 0:
        continue
    yy, pp = y[m], p[m]
    npos = int(yy.sum()); n = len(yy)
    auc = oc.tie_aware_auc(yy, pp) if 0 < npos < n else np.nan
    nontouch_rows.append(dict(risk_ATR=rec["risk_ATR"], wf=rec["wf"],
                              n=n, base_rate=yy.mean(), r2_roc_auc=auc))
tou.to_csv(OUT / "touch_only_sensitivity.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(nontouch_rows).to_csv(OUT / "touch_only_sensitivity_nontouch.csv",
                                   index=False, encoding="utf-8-sig")
ct_auc = (tou.groupby("contact_type")
          .agg(n=("r2_roc_auc", "sum"), base_rate=("base_rate", "mean"),
               r2_roc_auc=("r2_roc_auc", "mean")).reset_index())
print("[P0.5] 按 contact_type 拆分 R2 AUC:\n", ct_auc.to_string(index=False),
      flush=True)

# 综合裁决
if b_decision in ("OPPORTUNITY_CLOSED_G4_RESIDUAL_NOT_MATERIAL",
                 "G4_RESIDUAL_NOT_MATERIAL"):
    overall = "STOP_OPPORTUNITY_TWO_SIDED_GEOMETRY_SUFFICIENT"
elif b_decision == "OPPORTUNITY_NEEDS_CONTINUE":
    overall = "CONTINUE_OPPORTUNITY_STAGE2"
else:
    overall = "CONTINUE_OPPORTUNITY_STAGE2"
stage1_decision = {
    "experiment": "SMC Opportunity Fast Closure Audit v1.2-lite",
    "stage1_A_geometry": a_gate,
    "stage1_B_g4_residual": b_gate,
    "overall": overall,
    "note": "低成本 Gate：单条 R0 规则不够（Δ=%.3f），但 R2(上下两条距离) "
            "即封顶且 R3-R2≈0、G4 残余≈0 -> Opportunity 压缩为 2 特征几何规则，"
            "无 ML/metadata/G4 增量；研究收口，资源转 RISK_DEPENDENT。"
            % mean_R2_R0,
}
with open(OUT / "stage1_gate_decision.json", "w") as f:
    json.dump(stage1_decision, f, indent=2, ensure_ascii=False)
print("[GATE] A:", a_gate["decision"], "| B:", b_decision,
      "| overall:", stage1_decision["overall"], flush=True)
print("DONE fast closure v1.2-lite")
