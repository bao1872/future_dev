# SMC Direction Asymmetric Selective Abstention v1.4

**base**: `9bd2c87` (v1.3) &nbsp; **脚本**: `run_direction_asymmetric_abstention_v1_4.py`
**目标**: 不新增任何市场特征，验证 v1.3 暴露的 Continuation/Reversal tail reliability
asymmetry 能否通过 class-specific abstention 转化为稳定 actionable direction。
TRADING_METRICS=NOT_APPLICABLE。

---

## 0. P0 冻结 v1.3 结论（措辞收紧）

- Direction rank stable：**AUC ≈ 0.63–0.66**。
- Clear gate stable：**AUC ≈ 0.78**。
- **marginal G4 PSI < 0.03，但 domain AUC ≈ 0.60–0.63**
  → **没有强烈 marginal covariate shift，但存在 modest multivariate temporal shift**。
  **不得写 "NO covariate shift"。**
- `RANK_STABLE_TAIL_DRIFT` 是 **descriptive protocol verdict**，
  **不是 independent OOS statistical proof**。

### 0b. v1.5 对 v1.4 报告的收紧（P0）

- 允许结论：**S1 是当前最强的 development selector**。
- **不得写：已独立验证可执行。**
- v1.4 test cohort **excluded** `UNRESOLVED_CENSOR`（~4,127）与
  `NO_COMPARABLE_TARGET`（~3,540），合计约 **7.9% contacts**；因此
  actionable ≈65.4% 是"在事后可解析约 92% contact universe 上"的开发结果，
  不等于实盘全部 contact 的水平。（v1.5 专门修此项。）
- `selector_by_symbol.csv` 只含 `n_selected >= 50` 的 symbol-cell，因此只能写：
  **pooled 层面双边；足量 symbol-cell 中普遍双边**；
  **不得写：所有 15 品种每个 WF 均验证双边 LONG/SHORT**。

---

## 1. P1 冻结模型

Clear = `C_GLOBAL4 Logistic`，outer-train OOF target precision=0.85，min sel=0.05（不调）。
Direction = `G4_BASE` + HGB(max_depth=3, lr=0.05, max_iter=200, l2=1.0, seed=42)。
Walk-forward：WF1(TB1→TB2) / WF2(TB1+TB2→TB3) / WF3(TB1+TB2+TB3→TB4)。

---

## 2. P3/P7 四个固定 selector 指标

### S0_SYMMETRIC_10_10（baseline）

| wf | selection_rate | selected_clear_rate | direction_accuracy_given_clear | actionable_precision | n_pred_continuation | n_pred_reversal | predicted_LONG_actionable_precision | predicted_SHORT_actionable_precision |
|---|---|---|---|---|---|---|---|---|
| WF1 | 0.10499411740841413 | 0.8446676970633694 | 0.7086001829826166 | 0.598531684698609 | 1763 | 825 | 0.5945723684210527 | 0.6020408163265306 |
| WF2 | 0.14720857525681108 | 0.8355582524271845 | 0.7411038489469862 | 0.6192354368932039 | 1855 | 1441 | 0.5901771336553945 | 0.6368062317429406 |
| WF3 | 0.14815830362855315 | 0.8325107958050586 | 0.7525009262689886 | 0.6264651449722394 | 1756 | 1486 | 0.6467341306347746 | 0.6162412993039443 |

### S1_CONTINUATION_ONLY_10（只做高置信 Continuation）

| wf | selection_rate | selected_clear_rate | direction_accuracy_given_clear | actionable_precision | n_pred_continuation | n_pred_reversal | predicted_LONG_actionable_precision | predicted_SHORT_actionable_precision |
|---|---|---|---|---|---|---|---|---|
| WF1 | 0.07152419976469633 | 0.8167895632444696 | 0.7791666666666667 | 0.6364152013613159 | 1763 | 0 | 0.6496969696969697 | 0.6247334754797441 |
| WF2 | 0.08284948637784725 | 0.816711590296496 | 0.7920792079207921 | 0.6469002695417789 | 1855 | 0 | 0.6265356265356266 | 0.6628242074927954 |
| WF3 | 0.08024860616031441 | 0.8001138952164009 | 0.8476868327402135 | 0.678246013667426 | 1756 | 0 | 0.6955223880597015 | 0.6675874769797422 |

### S2_REVERSAL_ONLY_10（对照）

| wf | selection_rate | selected_clear_rate | direction_accuracy_given_clear | actionable_precision | n_pred_continuation | n_pred_reversal | predicted_LONG_actionable_precision | predicted_SHORT_actionable_precision |
|---|---|---|---|---|---|---|---|---|
| WF1 | 0.033469917643717796 | 0.9042424242424243 | 0.5723860589812333 | 0.5175757575757576 | 0 | 825 | 0.4782608695652174 | 0.5529953917050692 |
| WF2 | 0.06435908887896383 | 0.8598195697432338 | 0.678773204196933 | 0.5836224843858432 | 0 | 1441 | 0.5210280373831776 | 0.6100691016781836 |
| WF3 | 0.06790969746823873 | 0.8707940780619112 | 0.6491499227202473 | 0.5652759084791387 | 0 | 1486 | 0.5683453237410072 | 0.5640785781103835 |

### S3_CLASS_SPECIFIC_PRECISION（Primary）

| wf | selection_rate | selected_clear_rate | direction_accuracy_given_clear | actionable_precision | n_pred_continuation | n_pred_reversal | predicted_LONG_actionable_precision | predicted_SHORT_actionable_precision |
|---|---|---|---|---|---|---|---|---|
| WF1 | 0.19384153515355593 | 0.8275429049811637 | 0.753667172483561 | 0.6236919213059857 | 4605 | 173 | 0.6225806451612903 | 0.6246165644171779 |
| WF2 | 0.3054488610987048 | 0.813569235268314 | 0.7138749101365924 | 0.5807866647170639 | 6839 | 0 | 0.5647812761214823 | 0.5984615384615385 |
| WF3 | 0.34672333424732654 | 0.8095426387241333 | 0.7385216541843048 | 0.597864768683274 | 7096 | 491 | 0.5866592241138711 | 0.6078921078921079 |

> `continuation_precision_given_clear` / `reversal_precision_given_clear` 见
> `selector_metrics.csv`。**continuation-only 也必须同时产生 LONG 与 SHORT**
> （P12：behavior class 与 absolute trade side 必须区分）。

### 2b. 关键解读（数据驱动）

- **S1_CONTINUATION_ONLY（最稳）**：actionable = 0.6364 / 0.6469 / 0.6782，mean=0.6539；每个 WF 的 predicted LONG/SHORT 均 > 0 → pooled LONG=0.6548 / SHORT=0.6529。**Continuation-only 已是可执行 setup，且绝对方向仍是双边。**
- **S2_REVERSAL_ONLY（对照）**：actionable = 0.5176 / 0.5836 / 0.5653，mean=0.5555 **FAIL** → 证实 Reversal 高置信 precision 明显弱于 Continuation。
- **S3_CLASS_SPECIFIC（Primary）**：actionable = 0.6237 / 0.5808 / 0.5979，mean=0.6008（**marginal**）。S3 WF2 的 Reversal 在 0.71 精度下无法达到 min coverage，**被 disabled**；S3 实际主要由 continuation 驱动。
- **S0_SYMMETRIC（baseline）**：actionable = 0.5985 / 0.6192 / 0.6265。注意：v1.2/v1.3 two-stage 用 **G4_D1** 方向特征，WF1 actionable=0.560 不过门槛；本轮冻结 **G4_BASE** 使 WF1 升至约 0.599，与 v1.3 GLOBAL4_MINIMAL_SUFFICIENT 一致。
- **P12 提醒**：Continuation ≠ LONG。upper/lower contact 由 `side` 映射，continuation-only 同时产生 LONG 与 SHORT；不得因 Reversal 弱而称模型单边。

---

## 3. P4/P5 class 阈值（只用 outer-train OOF）

目标 precision = **0.71**（= target actionable 0.60 / clear precision 0.85 = 0.706，
**参数不是看历史结果拍的**）。min class OOF coverage = 0.02。
达不到精度的 class **允许 disabled**，禁止降精度强行产生信号。

| selector | wf | cont_enabled | cont_threshold | cont_oof_precision | cont_oof_coverage | rev_enabled | rev_threshold | rev_oof_precision | rev_oof_coverage |
|---|---|---|---|---|---|---|---|---|---|
| S0_SYMMETRIC_10_10 | WF1 | True | 0.23301423852639502 | nan | nan | True | 0.5815625638195261 | nan | nan |
| S0_SYMMETRIC_10_10 | WF2 | True | 0.23726264146415307 | nan | nan | True | 0.5735121876157734 | nan | nan |
| S0_SYMMETRIC_10_10 | WF3 | True | 0.23238126128035855 | nan | nan | True | 0.5673824658567059 | nan | nan |
| S1_CONTINUATION_ONLY_10 | WF1 | True | 0.23301423852639502 | nan | nan | False | nan | nan | nan |
| S1_CONTINUATION_ONLY_10 | WF2 | True | 0.23726264146415307 | nan | nan | False | nan | nan | nan |
| S1_CONTINUATION_ONLY_10 | WF3 | True | 0.23238126128035855 | nan | nan | False | nan | nan | nan |
| S2_REVERSAL_ONLY_10 | WF1 | False | nan | nan | nan | True | 0.5815625638195261 | nan | nan |
| S2_REVERSAL_ONLY_10 | WF2 | False | nan | nan | nan | True | 0.5735121876157734 | nan | nan |
| S2_REVERSAL_ONLY_10 | WF3 | False | nan | nan | nan | True | 0.5673824658567059 | nan | nan |
| S3_CLASS_SPECIFIC_PRECISION | WF1 | True | 0.6446056237673135 | 0.7100912200684151 | 0.3848179025888548 | True | 0.7074245967182117 | 0.7126436781609196 | 0.028630978499341816 |
| S3_CLASS_SPECIFIC_PRECISION | WF2 | True | 0.6059700233853407 | 0.7100355239786856 | 0.41638162152167885 | False | nan | nan | nan |
| S3_CLASS_SPECIFIC_PRECISION | WF3 | True | 0.5926446766847323 | 0.7100148420383066 | 0.4608344461453278 | True | 0.643556239370186 | 0.7105263157894737 | 0.02227795329446634 |
---

## 4. P8 Tail asymmetry 正式 audit + day-block bootstrap

| wf | n_cont_tail | n_rev_tail | continuation_tail | reversal_tail | delta |
|---|---:|---:|---:|---:|---:|
| WF1 | 1764 | 1282 | 0.7726757369614512 | 0.5655226209048362 | 0.20715311605661502 |
| WF2 | 1792 | 1479 | 0.7868303571428571 | 0.6680189317106153 | 0.11881142543224177 |
| WF3 | 1611 | 1660 | 0.845437616387337 | 0.6590361445783133 | 0.18640147180902378 |

Bootstrap（500 trading-day block resample，duplicates allowed）：

| wf | n_boot | point_delta | delta_mean | ci_lo | ci_hi |
|---|---:|---:|---:|---:|---:|
| WF1 | 500 | 0.20715311605661502 | 0.20774247685135755 | 0.13752023519269704 | 0.2797060603007303 |
| WF2 | 500 | 0.11881142543224177 | 0.11735824591652969 | 0.05149286577261229 | 0.19238972972179316 |
| WF3 | 500 | 0.18640147180902378 | 0.18404991894929776 | 0.12122468170663774 | 0.24181577862346568 |


判定：3/3 WF `continuation > reversal` 且 ≥2/3 WF `CI lower > 0`
→ `CONTINUATION_TAIL_STRUCTURALLY_STRONGER = True`
（n_wf_cont_gt_rev=3/3，
n_wf_ci_lo_gt0=3/3）。
注意：同一历史 development evidence，**不是 independent OOS**。

---

## 5. P9 Direct 3-class 非对称（secondary）

T3_HGB + G4_D1，分别对 P(CONT)/P(REV) 找 class-specific 阈值，目标 actionable
precision=0.60（3-class OOF 已含 TRADEOFF）。达不到 60% → disabled。

| wf | selection_rate | actionable_precision | n_pred_continuation | n_pred_reversal | cont_enabled | rev_enabled |
|---|---:|---:|---:|---:|---|---|
| WF1 | 0.15432674753539696 | 0.6206624605678234 | 3804 | 0 | True | False |
| WF2 | 0.26485037963376507 | 0.5924114671163575 | 5930 | 0 | True | False |
| WF3 | 0.2740151722877251 | 0.63025350233489 | 5996 | 0 | True | False |

---

## 6. P10/P11 Gate

```json
{
  "ASYMMETRIC_ACTIONABLE_DIRECTION": true,
  "CONTINUATION_ONLY_ACTIONABLE": true,
  "S0_SYMMETRIC_pass": true,
  "S1_pass": true,
  "S2_pass": false,
  "S3_pass": true,
  "T3_ASYM_pass": true
}
```

- **ASYMMETRIC_ACTIONABLE_DIRECTION (S3, Primary) = True**
- **CONTINUATION_ONLY_ACTIONABLE (S1) = True**
- 对照：S0=True / S2=False / T3_ASYM=True

阈值：每 WF actionable≥0.58、mean≥0.60、每 WF selection≥0.05、
pooled predicted LONG/SHORT actionable≥0.58。

---

## 7. P13 按 symbol（S3 / S1，n_selected>=50，不删品种）

见 `selector_by_symbol.csv`。**只含 n_selected>=50 的 symbol-cell**：
结论只能是 **pooled 层面双边 + 足量 symbol-cell 中普遍双边**，
**不得写"所有 15 品种每个 WF 都验证双边"**（有 WF 存在 n_selected&lt;50 的品种，
如 S1 WF1 的 I）。

---

## 8. P14 下一步决策

- 若 **S3 或 S1 PASS** → `FIXED_EXECUTION_BASELINE_V1`
  （固定 risk=1 ATR、冻结方向 selector、冻结真实 liquidity target，
  加手续费/滑点/roll/same-bar bounds，**第一次进入真实交易层**）。
- 若全部 FAIL → `PRECONTACT_DYNAMICS_INCREMENT`（只研究 Reversal 为什么弱）。

**本次裁决 next_step = `FIXED_EXECUTION_BASELINE_V1`**

---

## 9. P17 完成 / STOP

代码 + 测试 + 运行 + 报告 + commit + push 后 STOP。
禁止自动进入 PnL / pre-contact / FVG。等 reviewer。
