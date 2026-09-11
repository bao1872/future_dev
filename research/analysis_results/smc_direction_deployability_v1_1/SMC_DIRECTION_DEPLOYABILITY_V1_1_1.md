# SMC Direction Deployability v1.1.1 — Audit Repair

**base**: `c3030c8` (v1.1) &nbsp; **修复脚本**: `run_direction_deployability_v1_1_1_repair.py`
**目标**: 只修 v1.1 审计错误，不扩实验、不重跑已确认的方向分解结论。

---

## 0. 明确撤回 v1.1 的结论

以下 v1.1 结论 **全部撤回**（原因：P9 把 clear-label 模型误当成 direction 模型）：

- `DEPLOYABLE_DIRECTION_SELECTOR = FALSE` —— 无效，基于错误实现。
- `joint actionable_precision = 0.4485` —— 伪值。
- "direction collapses on clear subset" —— 从未被真正测试。
- v1.1 `ROBUST diagnostic`（用 ad-hoc `robust_dir`，非冻结语义）—— 撤回重算。
- by-symbol gate `45/45` —— 实为 45 个 WF-cell 误当 15 个 symbol。
- risk sensitivity 在 r0.5/r2.0 的 `abs_direction_accuracy` —— 误用 risk=1 的 rr。

**v1.1 仍站得住的结论（未重跑，直接复用既有 CSV 用于 Gate）**：
GLOBAL geometry 有方向信息；canonical liquidity identity（scope/type）无稳定增量；
clear gate 强（AUC≈0.78）；direction 可学（N_MAP_CORE≈0.642）；
N_GLOBAL4（≈0.651）略优于 N_MAP_CORE（≈0.642）。

---

## 1. P9 联合选择器修复（J0 / J1 / J2）

修复核心：clear 用 `y_clear`（LONG/SHORT/TRADEOFF 三态）独立训练；
direction 用 `y_rev`（LONG_DOMINATES/SHORT_DOMINATES 的 reversal）独立训练；
测试集两个预测**断言不相等**（`assert not np.allclose`），防止 v1.1 同分 bug 回归。
阈值固定：clear_thr=0.5；direction 取 train-OOF 两尾各 10%（合计 20%），test 不调阈值。

### J0_EXACT（v1.1 原意修复：C_MAP_CORE + M_MAP_CORE，均 logistic）

| WF | selection_rate | selected_clear_rate | direction_auc_inside_clear_gate | actionable_precision | pred_LONG_ap | pred_SHORT_ap |
|---|---:|---:|---:|---:|---:|---:|
| WF1 | 0.2866 | 0.7967 | 0.551 | 0.398 | 0.3846 | 0.4105 |
| WF2 | 0.105 | 0.7963 | 0.6268 | 0.5483 | 0.5234 | 0.5619 |
| WF3 | 0.1012 | 0.8266 | 0.6352 | 0.5422 | 0.5435 | 0.5415 |

**J0 actionable_precision（mean / min-wf）**: 0.4962 / 0.398

### J1_SIMPLE_LINEAR（C_GLOBAL4 + M_GLOBAL4，logistic）

| WF | selection_rate | direction_auc_inside_clear_gate | actionable_precision | pred_LONG_ap | pred_SHORT_ap |
|---|---:|---:|---:|---:|---:|
| WF1 | 0.1501 | 0.5936 | 0.4515 | 0.4472 | 0.4542 |
| WF2 | 0.1368 | 0.6382 | 0.5132 | 0.5134 | 0.5131 |
| WF3 | 0.1737 | 0.6381 | 0.5097 | 0.5046 | 0.5132 |

### J2_SIMPLE_NONLINEAR（C_GLOBAL4 logistic + N_GLOBAL4 HGB）

| WF | selection_rate | direction_auc_inside_clear_gate | actionable_precision | pred_LONG_ap | pred_SHORT_ap |
|---|---:|---:|---:|---:|---:|
| WF1 | 0.1973 | 0.634 | 0.5136 | 0.5133 | 0.5138 |
| WF2 | 0.1477 | 0.6658 | 0.5913 | 0.589 | 0.5928 |
| WF3 | 0.1623 | 0.6629 | 0.614 | 0.6218 | 0.6095 |

> J1/J2 来自 v1.1 已观察结果（GLOBAL4 更简洁更强），属 **post-hoc 机制诊断**，
> 非独立确认。J0 才是 v1.1 原意图的严格修复。

### 失败来源诊断（关键，修正 v1.1 的"direction collapses"误读）

修正后 **direction 并未在 clear 子集上塌缩**。以 conditional direction accuracy
（selected AND clear 上的方向准确率）看：

- J0：WF2=0.689, WF3=0.656（强），WF1=0.500（弱）
- J2：WF2=0.748, WF3=0.772（很强），WF1=0.649

`actionable_precision` 明显低于 conditional accuracy，原因有三：

1. **clear gate 在 thr=0.5 漏过 ~20% TRADEOFF**（tradeoff_selected_rate 0.17–0.21）。
   TRADEOFF 无确定方向，必然计为 actionable 失败，拉低 actionable_precision。
2. **WF1（最早时段）direction inside clear gate 较弱**（J0 AUC≈0.55, J2≈0.63），
   WF2/WF3 稳定 0.63–0.66，存在时间漂移。
3. **J2（C_GLOBAL4 + N_GLOBAL4）优于 J0**（C/M_MAP_CORE）：mean actionable 0.573 vs 0.496，
   conditional accuracy WF3 达 0.772。支持"更简单 GLOBAL4 geometry 比 identity map 更好"。

结论：可部署性未达 0.60 **不是因为 direction 失效**，而是
(a) clear gate 的 tradeoff 误选 与 (b) WF1 早期时段方向漂移。
这正是 **情况 B**：P(clear) 与 P(direction|clear) 分别可学，但联合选择器的
actionable 受 clear-gate 假阳性 + 早期时段漂移限制。

---

## 2. ROBUST diagnostic（冻结语义重算）

使用冻结 Atlas v1.2 的 `oracle_direction_stability_v1_2.parquet`：
`direction_stability ∈ {ROBUST_LONG, ROBUST_SHORT, RISK_DEPENDENT, NO_DIRECTION, UNRESOLVED}`。
仅 ROBUST_LONG/SHORT 计算 `y_rev_robust`。

**Count audit**：冻结 ROBUST_LONG+SHORT = 18902；
重算 `y_rev_robust` notna = 18902；
**PASS = True**（不一致则 FATAL_SEMANTIC_MISMATCH 停止）。

冻结分布：{'NO_DIRECTION': 51836, 'RISK_DEPENDENT': 18495, 'ROBUST_SHORT': 9477, 'ROBUST_LONG': 9425, 'UNRESOLVED': 7667}

### ROBUST direction AUC（eligible = ROBUST_LONG/SHORT，insample）

| WF | model | roc_auc | balanced_accuracy |
|---|---|---:|---:|
| WF1 | M_GLOBAL4 | 0.5626 | 0.539 |
| WF1 | M_IDENTITY_MAP | 0.5399 | 0.531 |
| WF1 | M_MAP_CORE | 0.5579 | 0.550 |
| WF2 | M_GLOBAL4 | 0.6051 | 0.561 |
| WF2 | M_IDENTITY_MAP | 0.5905 | 0.559 |
| WF2 | M_MAP_CORE | 0.5910 | 0.557 |
| WF3 | M_GLOBAL4 | 0.5962 | 0.541 |
| WF3 | M_IDENTITY_MAP | 0.5900 | 0.545 |
| WF3 | M_MAP_CORE | 0.6101 | 0.572 |

---

## 3. Risk sensitivity（仅 AUC；已删 abs_direction_accuracy）

### risk=1.0 (N_MAP_CORE / M_MAP_CORE)

| WF | model | roc_auc | balanced_accuracy |
|---|---|---:|---:|
| WF1 | M_MAP_CORE_r10 | 0.5551 | 0.538 |
| WF1 | N_MAP_CORE_r10 | 0.6209 | 0.576 |
| WF2 | M_MAP_CORE_r10 | 0.6239 | 0.563 |
| WF2 | N_MAP_CORE_r10 | 0.6454 | 0.578 |
| WF3 | M_MAP_CORE_r10 | 0.6333 | 0.582 |
| WF3 | N_MAP_CORE_r10 | 0.6585 | 0.589 |

### risk=0.5

| WF | model | roc_auc | balanced_accuracy |
|---|---|---:|---:|
| WF1 | M_MAP_CORE_r05 | 0.5783 | 0.557 |
| WF1 | N_MAP_CORE_r05 | 0.6498 | 0.599 |
| WF2 | M_MAP_CORE_r05 | 0.6357 | 0.592 |
| WF2 | N_MAP_CORE_r05 | 0.6781 | 0.603 |
| WF3 | M_MAP_CORE_r05 | 0.6625 | 0.611 |
| WF3 | N_MAP_CORE_r05 | 0.6940 | 0.617 |

### risk=2.0

| WF | model | roc_auc | balanced_accuracy |
|---|---|---:|---:|
| WF1 | M_MAP_CORE_r20 | 0.5412 | 0.530 |
| WF1 | N_MAP_CORE_r20 | 0.5735 | 0.545 |
| WF2 | M_MAP_CORE_r20 | 0.5707 | 0.541 |
| WF2 | N_MAP_CORE_r20 | 0.5782 | 0.537 |
| WF3 | M_MAP_CORE_r20 | 0.5986 | 0.555 |
| WF3 | N_MAP_CORE_r20 | 0.5980 | 0.543 |

> AUC 跨 risk 单调：risk 越紧 → geometry 对最终方向越可预测（r0.5>r1.0>r2.0），
> 与 RISK_DEPENDENT/~1.25 ATR switch 发现相容。absolute accuracy 已删除（旧实现语义错误）。

---

## 4. 修正后 Gate

```json
{
  "DEPLOYABLE_DIRECTION_SELECTOR": false,
  "DIRECTION_SIGNAL_CONFIRMED": true,
  "LIQUIDITY_IDENTITY_INCREMENTAL": false,
  "RETRACTED_v1_1_DEPLOYABLE_FALSE": true,
  "n_map_core_mean_auc": 0.6416,
  "n_map_core_per_wf": {
    "WF1": 0.6209,
    "WF2": 0.6454,
    "WF3": 0.6585
  },
  "cov20_macro_tail_accuracy": 0.6566,
  "n_symbols_auc_gt_05": 15,
  "n_symbols_total": 15,
  "n_symbol_wf_cells_auc_gt05": 45,
  "clear_gate_mean_auc": 0.7708,
  "joint_actionable_precision_J0": 0.4962,
  "joint_actionable_precision_J0_min_wf": 0.398,
  "identity_incremental_point_delta": -0.0201,
  "identity_incremental_wf_ci_lower_gt0": 0,
  "verdict": "DIRECTION_SIGNAL_CONFIRMED"
}
```

**verdict**: `DIRECTION_SIGNAL_CONFIRMED`

> 注：`DEPLOYABLE_DIRECTION_SELECTOR` 现由 J0 的 `actionable_precision` 真实计算。
> 若仍 < 0.60 / 任一 WF < 0.55，则区分失败来源：
> clear gate 不行（selected_clear_rate 低）还是 direction conditional 不行
> （direction_auc_inside_clear_gate 低），或两者交集冲突。

---

## 5. 当前项目真实进展（修正版）

| 问题 | 当前结论 |
|---|---|
| 有无 structural delivery opportunity | **已确认** |
| Opportunity 主要来自 | **简单 liquidity geometry** |
| canonical liquidity identity 是否有额外价值 | **当前否** |
| clear vs tradeoff 能否预测 | **强，可学，AUC≈0.78** |
| clear 条件下方向能否预测 | **是，AUC≈0.65** |
| nonlinear 是否有价值 | **是，GLOBAL4 上稳定提升** |
| 联合 clear→direction selector 是否可部署 | **见上方 Gate（已由正确实现重测）** |
| ROBUST direction | **已用冻结语义重算（见 §2）** |
| PnL | **尚未测试** |

---

## 6. 测试

`run_tests` 内置 7 项断言（全部 PASS 方可写出结果）：
clear≠direction 分数、direction_accuracy 排除 TRADEOFF、tradeoff 计入 actionable 失败、
predicted LONG/SHORT≠contact side、by-symbol gate 计 15 symbol 非 45 cell、
robust 标签与冻结计数一致。

---

## 7. 完成条件 / STOP

代码修复 + 测试 + 重跑受影响实验 + 报告 + commit + push 后 STOP。
**禁止**自动进入：pre-contact dynamics / FVG / PnL / 新模型 / 参数搜索。
等待 reviewer 审核 v1.1.1。
