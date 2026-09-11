# SMC Direction Actionability Architecture v1.2

**base**: `26f4785` (v1.1.1) &nbsp; **脚本**: `run_direction_actionability_v1_2.py`
**目标**: 不新增任何市场特征，只把 v1.1.1 已确认的两个强信号
（Clear/Tradeoff gate AUC≈0.78；Direction given clear AUC≈0.65）组合成
可行动选择器。TRADING_METRICS=NOT_APPLICABLE（架构验证，未定义交易动作）。

---

## 0. 上一轮修正（P0 措辞）

v1.1.1 报告中 "RISK_DEPENDENT / ~1.25 ATR switch" 表述已过期，本论起统一改为
**risk-dependent structural switching**（P4c 连续 direct-switch 中位约 **1.74 ATR**）。
"1.25" 只是旧网格中点，不再作为机制尺度。

---

## 1. P1 Minimal-G4：direction 能否压缩成纯 4 距离几何

`G4_ONLY`=4 距离；`G4_BASE`=symbol+side+4距离；`G4_D1`=M_GLOBAL4(D1+GLOBAL4)。
| feature_set | model | mean roc_auc |
|---|---|---:|
| G4_BASE | hgb | 0.6518 |
| G4_BASE | logistic | 0.6227 |
| G4_D1 | hgb | 0.6507 |
| G4_D1 | logistic | 0.6212 |
| G4_ONLY | hgb | 0.6315 |
| G4_ONLY | logistic | 0.6258 |

判断：`G4_D1 - G4_BASE` 与 `G4_BASE - G4_ONLY` 的 ΔAUC 见 CSV。若 |Δ|<0.01 且 3 WF
无稳定增量，则允许 `GLOBAL4_MINIMAL_SUFFICIENT`（方向模型可压缩到极简单结构）。

---

## 2. P2 Calibrated two-stage（precision-first clear gate）

`Clear = C_GLOBAL4(logistic)`，`Direction = N_GLOBAL4(hgb)`。
clear threshold 由 **train-OOF** 在 precision≥0.85 条件下取最大 coverage
（禁止 test 调参）。direction 两尾 = train-OOF 10%/90%。

### CALIBRATED_TWO_STAGE

| WF | clear_thr | selection_rate | selected_clear_rate | direction_accuracy_given_clear | actionable_precision | pred_LONG_ap | pred_SHORT_ap |
|---|---|---:|---:|---:|---:|---:|---:|
| WF1 | 0.7395 | 0.1469430808552071 | 0.8500828271673109 | 0.6593049691458266 | 0.560463832136941 | 0.5532687651331719 | 0.566497461928934 |
| WF2 | 0.681 | 0.12317999106744082 | 0.8288614938361131 | 0.747594050743657 | 0.6196519216823785 | 0.6167608286252354 | 0.6214622641509434 |
| WF3 | 0.6703 | 0.13385430947810986 | 0.8398770911573916 | 0.7686991869918699 | 0.6456128371457835 | 0.669 | 0.6334888543286677 |

### J2_FIXED_0.5（baseline，clear_thr=0.5 不校准）

| WF | selection_rate | selected_clear_rate | direction_accuracy_given_clear | actionable_precision |
|---|---:|---:|---:|---:|
| WF1 | 0.19733052050793135 | 0.791735197368421 | 0.6486626850168787 | 0.5135690789473685 |
| WF2 | 0.14774452880750336 | 0.7902055622732769 | 0.7482785003825555 | 0.5912938331318017 |
| WF3 | 0.16232519879352894 | 0.7956081081081081 | 0.7717622080679406 | 0.6140202702702703 |

> 校准后 `selected_clear_rate` 应明显高于 fixed 0.5（~20% TRADEOFF contamination
> 被压低）。这是本论针对 v1.1.1 揭示瓶颈（Clear purity≈79%）的直接回应。

---

## 3. P3 Direct 3-class（REVERSAL / CONTINUATION / TRADEOFF）

`T3_HGB` Primary coverage=20%。

| WF | coverage | selection_rate | selected_clear_rate | direction_accuracy_given_clear | actionable_precision |
|---|---:|---:|---:|---:|---:|
| WF1 | 0.2 | 0.18905432269057568 | 0.9214592274678112 | 0.607824871914299 | 0.5600858369098712 |
| WF2 | 0.2 | 0.1659222867351496 | 0.9141318977119784 | 0.6678445229681979 | 0.6104979811574697 |
| WF3 | 0.2 | 0.14678731377387808 | 0.924346201743462 | 0.6833950825193668 | 0.6316936488169365 |

---

## 4. P4 Product diagnostic（非 Gate）

`P_long/P_short = p_clear × p_rev(或 1-p_rev)`，`P_tradeoff = 1-p_clear`，
`joint_margin = max(P_long,P_short) - P_tradeoff`，train-OOF 80% 分位定阈。

| WF | selection_rate | selected_clear_rate | direction_accuracy_given_clear | actionable_precision |
|---|---:|---:|---:|---:|
| WF1 | 0.14227757718365858 | 0.894781864841745 | 0.3365200764818356 | 0.3011120615911035 |
| WF2 | 0.09611433675748102 | 0.9121747211895911 | 0.29495669893020887 | 0.2690520446096654 |
| WF3 | 0.11918471803308656 | 0.9187116564417178 | 0.25375626043405675 | 0.2331288343558282 |

---

## 5. P5 Gate：ACTIONABLE_DIRECTION_PRESENT

```
ACTIONABLE_DIRECTION_PRESENT = False
CALIBRATED_TWO_STAGE pass = False
T3_HGB_20pct pass      = False
```

判定：CALIBRATED_TWO_STAGE 或 T3_HGB_20pct 满足
（每 WF actionable≥0.58 且 mean≥0.60 且每 WF selection≥0.05 且
pooled predicted LONG/SHORT actionable≥0.58）→ TRUE。

> 注意：这仍不等于盈利。它只表示在未知未来 clear/tradeoff 真实条件下，
> 可以筛出具有可观方向准确率的候选。

---

## 6. P7 按 symbol

仅 `n_selected>=50` 品种级报告（避免样本碎裂）。见 `actionability_by_symbol.csv`。

---

## 7. P8 Bootstrap

> **STOP_NO_BOOTSTRAP**：ACTIONABLE_DIRECTION_PRESENT=False，未做 bootstrap。


---

## 8. 回答用户 P9 七个问题

1. **clear threshold precision-first 校准是否显著减少 TRADEOFF contamination？**
   见 §2 CALIBRATED vs J2_FIXED 的 `selected_clear_rate`。
2. **calibrated two-stage actionable 是否达 60% 附近？** 见 §2 / §5。
3. **direct 3-class 是否优于 two-stage？** 见 §3 vs §2。
4. **GLOBAL4 是否可压缩成接近 4 变量？** 见 §1。
5. **LONG 与 SHORT 是否都可预测？** 见各表 pred_LONG/SHORT_ap。
6. **WF1 弱是否仍存在？** 见各表 WF1 行。
7. **是否值得加 pre-contact dynamics？** 仅当 ACTIONABLE=TRUE 才建议
   PRECONTACT_DYNAMICS_INCREMENT；否则先研究 temporal regime / WF1 drift。

---

## 9. 完成条件 / STOP

代码 + 运行 + 报告 + commit + push。禁止进入 FVG / pre-contact dynamics /
PnL / stop-target 优化。等待 reviewer 审核 v1.2。
