# SMC Direction Pre-Execution Integrity Gate v1.5

**base**: `2117cb1` (v1.4) &nbsp; **脚本**: `run_direction_preexec_integrity_v1_5.py`
**目标**: 审计并消除 (1) label-availability leakage、(2) 事后 cohort 推理
（改为 full live universe）、(3) ex-ante target availability、(4) duplicate/conflict。
**不做 PnL**。TRADING_METRICS=NOT_APPLICABLE。

---

## 0. P0 收紧 v1.4 措辞

- v1.4 允许结论：**S1 是当前最强的 development selector**；**不得写已独立验证可执行**。
- v1.4 test cohort **excluded** `UNRESOLVED_CENSOR` 与 `NO_COMPARABLE_TARGET`
  （合计约 7.9%），故 65.4% 是"事后可解析约 92% contact universe"的开发结果。
- `selector_by_symbol.csv` 只含 `n_selected>=50`：只能写 **pooled 双边 +
  足量 symbol-cell 普遍双边**，不得写"所有 15 品种每个 WF 均双边"。

---

## 1. P1 Label availability

`label_available_time = bar_start_time[contact_bar_index +
max(long_bars_to_stop, short_bars_to_stop)] + 5min`（risk=1.0）。

**语义检查**（冻结 clear/tradeoff 必须两方向 resolve）：

```json
{
  "frozen_clear_tradeoff_n": 89233,
  "frozen_rows_missing_bars_to_stop": 0,
  "frozen_rows_censored_lower_bound": 0,
  "frozen_rows_label_avail_nat": 0,
  "availability_index_out_of_range": 0,
  "FATAL_LABEL_AVAILABILITY_SEMANTIC_FAIL": false,
  "all_label_avail_ge_decision_time": true
}
```

| rr_direction | n | label_avail NaT | long_bts null | short_bts null | median_lag_days |
|---|---:|---:|---:|---:|---:|
| LONG_DOMINATES | 31847 | 0 | 0 | 0 | 0.9618 |
| SHORT_DOMINATES | 33161 | 0 | 0 | 0 | 0.9583 |
| TRADEOFF_OR_OVERLAP | 24225 | 0 | 0 | 0 | 0.184 |
| UNRESOLVED_CENSOR | 4127 | 4127 | 3325 | 816 | nan |
| NO_COMPARABLE_TARGET | 3540 | 3540 | 2676 | 978 | nan |
若 `FATAL_LABEL_AVAILABILITY_SEMANTIC_FAIL=true` 则立即停止（本次为
`False`）。

---

## 2. P2/P3 外/内层 availability 过滤

外层：train 只用 `label_available_time < test_start_time`。
内层：`expanding_oof_pred_available()` 每个 fold 断言
`max(train label available) < min(val decision_time)`。

见 `inner_oof_availability_audit.csv`（每 WF×fold 的 n_before/n_after/removed_share/skipped）。

---

## 3. P4 availability-safe S1

| wf | clear_tr before→after | dir_tr before→after | dir_removed | clear_thr | cont_thr | dir_auc_eval | sel_rate | evaluable_act |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WF1 | 20312 | 19050 | 14779 | 13556 | 0.0828 | 0.7587 | 0.2011 | 0.63 | 0.0447 | 0.654 |
| WF2 | 44961 | 43661 | 33168 | 31881 | 0.0388 | 0.6805 | 0.218 | 0.6601 | 0.0742 | 0.6526 |
| WF3 | 67351 | 66461 | 49431 | 48568 | 0.0175 | 0.6661 | 0.2258 | 0.6553 | 0.0721 | 0.685 |
Pooled predicted LONG/SHORT（evaluable）：
**LONG=0.6643 / SHORT=0.6647**。

Threshold **不变**：clear OOF precision=0.85；continuation = direction OOF bottom 10%。

---

## 4. P5/P6/P7 Full live universe + 两个 actionability 指标

Test inference 在 **ALL test contacts** 上运行（不要求 `y_clear` notna）。

| wf | n_test_all | n_selected | selected_LONG_DOM | selected_SHORT_DOM | selected_TRADEOFF | selected_UNRESOLVED | selected_NO_COMPARABLE |
|---|---:|---:|---:|---:|---:|---:|---:|
| WF1 | 25903 | 1159 | 504 | 413 | 213 | 29 | 0 |
| WF2 | 24440 | 1813 | 729 | 719 | 308 | 54 | 3 |
| WF3 | 23869 | 1720 | 557 | 743 | 303 | 117 | 0 |

| wf | evaluable_actionable | live_conservative_lower_bound | selected_unknown_rate |
|---|---:|---:|---:|
| WF1 | 0.654 | 0.6376 | 0.025 |
| WF2 | 0.6526 | 0.6321 | 0.0314 |
| WF3 | 0.685 | 0.6384 | 0.068 |

> `evaluable` 用于与 v1.4 比较；`live_conservative_lower_bound` 把
> TRADEOFF/UNRESOLVED/NO_COMPARABLE 全部计为 failure（完整 selected 分母），
> 是保守标签边界，**不代表真实 execution precision**。

---

## 5. P8/P9 Ex-ante target availability

只用 decision-time active liquidity（`active_mask`），`trade_direction = side`
（Continuation）。报告 availability，**不做 outcome，不做最小 RR 筛选**。

> **P0 target caveat**：这里的 `nearest_exante_target` **只验证 active target
> availability，不是冻结 execution target 语义**。TOUCH_ONLY 的 contact 不会产生
> `first_penetration_time`，当前 attacked level 在 decision_time 仍 active，
> 因此大量 `target == attack/contact price`（见 §5 `target==contact` 列）。
> 真正的 execution target 必须是 **beyond attacked boundary**（v1.0 冻结）。

| wf | selected | no_target | no_target_rate | exec_candidate | exec_rate | target==contact |
|---|---:|---:|---:|---:|---:|---:|
| WF1 | 1159 | 0 | 0.0 | 1159 | 1.0 | 646 |
| WF2 | 1813 | 0 | 0.0 | 1813 | 1.0 | 686 |
| WF3 | 1720 | 0 | 0.0 | 1720 | 1.0 | 641 |

---

## 6. P10 Duplicate / Conflict audit

| wf | n_exec | dup_group | dup_contact | conflict_group | conflict_contact |
|---|---:|---:|---:|---:|---:|
| WF1 | 1159 | 0.3636 | 0.6437 | 0.0 | 0.0 |
| WF2 | 1813 | 0.3711 | 0.6784 | 0.0 | 0.0 |
| WF3 | 1720 | 0.4011 | 0.6884 | 0.0 | 0.0 |

未来执行策略：same-direction duplicates → collapse 为一个 signal；
direction conflict → abstain。本轮只报告。

---

## 7. P11/P12 Gate

```json
{
  "LABEL_AVAILABILITY_SAFE_DIRECTION": true,
  "LIVE_UNIVERSE_SELECTOR_AUDITED": true,
  "per_wf_evaluable_actionable": [
    0.6539823008849558,
    0.6526195899772209,
    0.6849656893325016
  ],
  "mean_evaluable_actionable": 0.6638558600648927,
  "pooled_LONG_actionable": 0.6643421664342166,
  "pooled_SHORT_actionable": 0.6646706586826348,
  "per_wf_selection_rate": [
    0.04474385206346755,
    0.07418166939443535,
    0.07205999413465164
  ],
  "selection_ge_5pct_per_wf": false,
  "LABEL_AVAILABILITY_SAFE_DIRECTION_STRICT_WITH_COVERAGE": false,
  "coverage_caveat": "WF1 full-universe selection_rate=4.47% < 5%：P11 未把 coverage 列入 gate，故主 gate 仍 TRUE；若沿用 v1.2-v1.4 的 >=5% 规则则为 FALSE。请 reviewer 裁决。"
}
```

- **LABEL_AVAILABILITY_SAFE_DIRECTION = True**
  （P11 原文门槛：WF1/WF2/WF3 evaluable actionable≥0.58、mean≥0.60、
  pooled predicted LONG/SHORT≥0.58）
- **LABEL_AVAILABILITY_SAFE_DIRECTION_STRICT_WITH_COVERAGE = False**
- **LIVE_UNIVERSE_SELECTOR_AUDITED = True**

> **Coverage caveat**：P11 未把 coverage 列入 gate。本轮 full-universe 分母下
> **WF1 selection_rate = 4.47% < 5%**（WF2/WF3 分别 7.42%/7.21%）。因此：
> 按 P11 原文 → 主 gate = `True`；
> 若沿用 v1.2–v1.4 的 `selection≥0.05` 规则 → `False`。
> **请 reviewer 裁决用哪一个。**

若 `LABEL_AVAILABILITY_SAFE_DIRECTION=FALSE` → 立即 STOP（过去方向结果受
label availability 影响），不得 PnL。若 TRUE → next step `FIXED_EXECUTION_BASELINE_V1`。

---

## 8. P15 完成 / STOP

代码 + 测试 + 运行 + 报告 + commit + push 后 STOP。
禁止自动进入 PnL / 手续费假设 / pre-contact / FVG。等 reviewer。
