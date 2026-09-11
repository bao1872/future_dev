# SMC Direction Temporal Stability & Tail Drift Audit v1.3

**base**: `3da5e68` (v1.2) &nbsp; **脚本**: `run_direction_temporal_stability_v1_3.py`
**目标**: 不新增任何市场特征，纯时间稳定性诊断，回答"WF1 为什么比 WF2/WF3 弱"。
TRADING_METRICS=NOT_APPLICABLE（架构/机制验证，未定义交易动作）。

---

## 0. v1.2 审计修复（P0 / P0.5 / P0.6）

**P0 Gate aggregation bug（已修）**：v1.2 `gate_eval` 用 `extra["side"]`（存 `+1/-1`）
去比较字符串 `"LONG_DOMINATES"`，导致 `pooled_LONG_actionable=0.0`、
`pooled_SHORT_actionable=0.8518`（错误）。修正为用 `selected+actual_clear+
correct_direction` 计算。修正后：

| candidate | ACTIONABLE | pooled_LONG | pooled_SHORT |
|---|---|---:|---:|
| CALIBRATED_TWO_STAGE | False | 0.6025848142164781 | 0.6062555853440572 |
| T3_HGB_20pct | False | 0.5772357723577236 | 0.6133509583608724 |

> Gate 仍 **FALSE**：因 WF1 per-WF actionable=0.560<0.58。但理由改为
> "WF1 per-WF actionable 没过门槛"，不再引用错误的 pooled LONG=0。
> 修正后 pooled LONG/SHORT 均约 **0.60**，证明是双边模型，方向能力不偏 LONG 或 SHORT。

**P0.5 Product diagnostic（正式作废）**：`TWO_STAGE_PRODUCT` 的 OOF 对齐
`oidx_c`/`oidx_d` 属于不同坐标系（相对 `trc` vs `tr_dir`），直接 `set()&` 无效。
标记 `TWO_STAGE_PRODUCT_DIAGNOSTIC=RETRACTED_INVALID_OOF_ALIGNMENT`；
旧 `product_diagnostic_metrics.csv` 保留作 bug evidence，本论不引用其 0.23–0.30 结果，也不修复（ROI 低）。

**P0.6 解释收紧**：TRADEOFF contamination **降低**（~20%→~15–17%），非 "solved"；
WF1 高置信方向更弱，未证明 regime drift。

---

## 1. P1 冻结 cohort

方向任务 = `LONG_DOMINATES` vs `SHORT_DOMINATES`，镜像为 `CONTINUATION=0 / REVERSAL=1`，
仅 clear rows。本论所有方向模型只看 clear subset，不让 clear gate 混淆原因。
Primary 特征固定 `G4_BASE = {symbol, side, nearest_above_R, nearest_below_R,
nearest_ahead_R, nearest_behind_R}`；Primary 模型固定 HGB(max_depth=3, lr=0.05,
max_iter=200, l2=1.0, seed=42)。

---

## 2. P2 训练历史长度（rolling vs expanding vs recent-two）

| config | window_type | test_auc | macro_tail | tail_cov |
|---|---|---:|---:|---:|
| TB1_TB2 | ONE_BLOCK_ROLLING | 0.6336228113288576 | 0.6690991789331437 | 0.16564250367067268 |
| TB2_TB3 | ONE_BLOCK_ROLLING | 0.653356340637075 | 0.7233805018791812 | 0.1939986472360573 |
| TB3_TB4 | ONE_BLOCK_ROLLING | 0.6399925541488447 | 0.6792722547108512 | 0.15908069589779802 |
| E123 | EXPANDING | 0.6591039918664263 | 0.7274246444267363 | 0.2011314025702515 |
| E1234 | EXPANDING | 0.6626762290939371 | 0.7522368804828252 | 0.20998908647364706 |
| R23_4 | RECENT_TWO_BLOCK | 0.6540073547117531 | 0.7383881708900051 | 0.16594979777877641 |

**解读**：ONE_BLOCK_ROLLING（R12/R23/R34）对比 EXPANDING（E123/E1234）与
RECENT_TWO_BLOCK（R23_4），判断训练历史长度是否是 WF1 弱的主因。
若 expanding 明显优于 rolling-1-block → TRAIN_HISTORY_LIMITED。

---

## 3. P3 历史模型跨远期测试（6 个 causal pair）

| config | train→test | test_auc | macro_tail |
|---|---|---:|---:|
| TB1_TB2 | TB1 | TB2 | 0.6336228113288576 | 0.6690991789331437 |
| TB1_TB3 | TB1 | TB3 | 0.6451233762729894 | 0.6926000534556248 |
| TB1_TB4 | TB1 | TB4 | 0.6391870956826523 | 0.7147232431228422 |
| TB2_TB3 | TB2 | TB3 | 0.653356340637075 | 0.7233805018791812 |
| TB2_TB4 | TB2 | TB4 | 0.6461765935815461 | 0.7164016047816693 |
| TB3_TB4 | TB3 | TB4 | 0.6399925541488447 | 0.6792722547108512 |

全部满足 train time < test time（无反向训练、无未来信息）。
若 `TB1→TB2` 弱但 `TB1→TB3/TB4` 恢复 → TB2 是特殊测试期（TB2_TARGET_REGIME_SHIFT）；
若 TB1 模型对所有未来都弱 → SOURCE_TB1_RELATION_DRIFT。

---

## 4. P4 Train-OOF tail transfer（仅信 train quantile）

| config | test_auc | rev_tail | cont_tail | macro_tail | cov |
|---|---|---:|---:|---:|---:|
| TB1_TB2 | 0.6336228113288576 | 0.5655226209048362 | 0.7726757369614512 | 0.6690991789331437 | 0.16564250367067268 |
| TB2_TB3 | 0.653356340637075 | 0.6855926188786373 | 0.761168384879725 | 0.7233805018791812 | 0.1939986472360573 |
| TB3_TB4 | 0.6399925541488447 | 0.5684210526315789 | 0.7901234567901234 | 0.6792722547108512 | 0.15908069589779802 |
| TB1_TB3 | 0.6451233762729894 | 0.6013195098963242 | 0.7838805970149254 | 0.6926000534556248 | 0.16823464305478694 |
| TB1_TB4 | 0.6391870956826523 | 0.6164383561643836 | 0.8130081300813008 | 0.7147232431228422 | 0.16505103678500352 |
| TB2_TB4 | 0.6461765935815461 | 0.6125852918877938 | 0.8202179176755447 | 0.7164016047816693 | 0.19072992232137126 |
| E123 | 0.6591039918664263 | 0.6680189317106153 | 0.7868303571428571 | 0.7274246444267363 | 0.2011314025702515 |
| E1234 | 0.6626762290939371 | 0.6590361445783133 | 0.845437616387337 | 0.7522368804828252 | 0.20998908647364706 |
| R23_4 | 0.6540073547117531 | 0.6338958180484225 | 0.8428805237315876 | 0.7383881708900051 | 0.16594979777877641 |

尾阈值（`lo`/`hi`）**只来自 train-OOF 10%/90% quantile**，test 不调。
若 AUC 稳定但 tail macro 明显低于后续 block → RANK_STABLE_TAIL_DRIFT。

---

## 5. P5 机制裁决（rank drift vs tail drift）

**VERDICT**: `RANK_STABLE_TAIL_DRIFT`
**NEXT_STEP**: rank-based adaptive abstention（非 FVG）

裁决证据（protocol verdict，非统计显著性声明）：

```json
{
  "TB1_model_weak_to_all_future": false,
  "TB2_target_regime_shift": false,
  "train_history_helps": false,
  "rank_stable_tail_drift": true,
  "broad_nonstationarity": false,
  "aucs": {
    "TB1_TB2": 0.6336,
    "TB2_TB3": 0.6534,
    "TB3_TB4": 0.64,
    "TB1_TB3": 0.6451,
    "TB1_TB4": 0.6392,
    "TB2_TB4": 0.6462,
    "E123": 0.6591,
    "E1234": 0.6627,
    "R23_4": 0.654
  },
  "tail_macro": {
    "TB1_TB2": 0.6691,
    "TB2_TB3": 0.7234,
    "TB3_TB4": 0.6793,
    "TB1_TB3": 0.6926,
    "TB1_TB4": 0.7147,
    "TB2_TB4": 0.7164,
    "E123": 0.7274,
    "E1234": 0.7522,
    "R23_4": 0.7384
  },
  "best_tail_macro": 0.7522
}
```

硬性规则：禁止只因 "WF1 actionable 低" 就写 regime drift；必须同时引用
rolling vs expanding、cross-block AUC、tail transfer、G4 分布、score-decile 稳定性、symbol macro。

---

## 6. P6 Geometry 分布漂移（4 距离）

每 TB 分布统计见 `g4_distribution_by_tb.csv`（missing_rate / p10 / p25 / median /
p75 / p90）。train→test PSI（bin 由 train 定义）见 `g4_psi_train_test.csv`。

PSI 汇总（每个 test_block 上，所有 train→test 配对、所有 field 的最大 PSI）：

- →TB2: max PSI = 0.027
- →TB3: max PSI = 0.0137
- →TB4: max PSI = 0.0221


PSI<0.1 可忽略；0.1–0.25 中等；>0.25 明显偏移。missingness 单独在分布表报告，不进 PSI。

---

## 7. P7 Domain classifier（covariate-shift 诊断，不进交易）

| pair | model | domain_auc |
|---|---|---:|
| TB1_vs_TB2 | DOMAIN_G4_ONLY | 0.6251 |
| TB1_vs_TB2 | DOMAIN_G4_BASE | 0.6316 |
| TB2_vs_TB3 | DOMAIN_G4_ONLY | 0.602 |
| TB2_vs_TB3 | DOMAIN_G4_BASE | 0.6198 |
| TB3_vs_TB4 | DOMAIN_G4_ONLY | 0.607 |
| TB3_vs_TB4 | DOMAIN_G4_BASE | 0.6273 |

`DOMAIN_G4_ONLY` 只看 4 距离（纯 geometry drift 信号）；`DOMAIN_G4_BASE` 含
symbol+side（可能反映品种构成变化）。domain_auc>>0.5 表示特征分布确实变化。
注意：这是分布可区分性诊断，不是方向能力证据。

---

## 8. P8 Score-decile 稳定性（train OOF 边界 → 跨期）

见 `score_decile_stability.csv`：每个 (train_source, test_block, score_decile) 的
`n / mean_score / actual_reversal_rate`。重点看最高/最低 decile 的 reversal 纯度是否随时间改变。
decile 边界**只来自 train OOF**，禁止用 test quantile 重新分箱。

---

## 9. P9 Symbol macro drift（不删品种）

见 `direction_symbol_temporal.csv` + 聚合 `direction_symbol_temporal_agg.csv`：
每个 causal pair 的 eligible symbol（n>=100 且两类都存在）AUC / tail macro。

| config | eligible_sym | median_auc | auc_iqr | n_auc>0.5 |
|---|---:|---:|---:|---:|
| TB1_TB2 | 15 | 0.6405 | 0.0327 | 15 |
| TB1_TB3 | 15 | 0.6559 | 0.0285 | 15 |
| TB1_TB4 | 15 | 0.6427 | 0.0483 | 15 |
| TB2_TB3 | 15 | 0.6539 | 0.0473 | 15 |
| TB2_TB4 | 15 | 0.6508 | 0.0625 | 15 |
| TB3_TB4 | 15 | 0.6427 | 0.0824 | 15 |

仅用于判断 WF1 弱是 15 品种普遍现象还是少数品种拖累。**不根据结果删品种**。

---

## 10. P10 Clear gate 时间稳定性（secondary）

见 `clear_gate_temporal_secondary.csv`：C_GLOBAL4 logistic，固定 target precision 0.85。
若 clear AUC / selected_clear_rate 在各窗口稳定，则 WF1 弱**不是 clear gate 造成**，
主疑点确在 direction。

---

## 11. 当前项目进展

| 问题 | 当前结论 |
|---|---|
| 有没有方向信息 | **有** |
| Clear/Tradeoff 可学 | **很强，AUC≈0.78** |
| Long/Short 方向可学 | **有，AUC≈0.65** |
| identity 是否重要 | **没有稳定增量** |
| 最小状态 | **symbol + side + GLOBAL4** |
| WF1 actionable 是否过预注册 Gate | **没有，0.560** |
| WF1 弱的原因 | 见 §5 裁决：`RANK_STABLE_TAIL_DRIFT` |
| 是否已证明 regime drift | **没有**（除非裁决为 *_REGIME_SHIFT / NONSTATIONARITY） |
| PnL | **仍未进入** |

方向不是死路：已从"Long/Short 能不能猜"推进到"方向信号在不同历史阶段为何强弱不同、应如何稳定使用"。

---

## 12. 完成条件 / STOP

代码 + 测试 + 运行 + 报告 + commit + push 后 STOP。禁止自动进入下一实验
（尤其禁止因 WF1 弱就加 FVG/pre-contact dynamics）。下一步决策见 §5 NEXT_STEP，
需你授权。
