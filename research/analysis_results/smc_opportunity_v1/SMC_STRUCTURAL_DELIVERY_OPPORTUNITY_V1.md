# SMC Structural Delivery Opportunity Study v1.0

> 实验名称：SMC Structural Delivery Opportunity Study v1.0
> 仓库：bao1872/future_dev　分支：main
> Atlas v1.2 冻结提交：7ae7c1ae57b45bbc8bade1b5742fdc0353859c83
> 时间稳定性审计提交：3b0ae21bb852004c35d03b3686e004e6a36591bc
> TB 定义哈希：b51fe7952bc1311aed83409889cd6706a6bc173cd4ed2f6e96570340183ac3d6
> 特征契约哈希：dd459be75d68c487…（见 feature_manifest.json）
> **证据等级：内部因果时间泛化（TB1→TB4 walk-forward）。非独立 OOS validation。**
> 前瞻性 OOS 边界锁定：`trading_day >= 2026-09-07` 的数据未进入本轮。

---

## 0. 最重要的事先纠正（语义）

旧画像字段 `NO_REACH` 的真实定义是 **`n_both_pos == 0`**：七档 ATR 中没有任何一档
出现 Long 与 Short **双向同时**到达 target。它**完全允许**只有 Long 到达、或只有 Short 到达。
因此旧名字误导了我们：它**不等于**"没有 delivery / 没有可交易路径"。

本轮一律将该历史字段改称 **`LEGACY_NO_BIDIRECTIONAL_REACH`**，不再解释为"没有 delivery"。

---

## 1. 旧 `NO_REACH≈86%` 到底意味着什么？真正 `NO_DELIVERY` 占多少？

- 无条件 `LEGACY_NO_BIDIRECTIONAL_REACH`（全样本 contact 级别 `n_both_pos==0`）
  = **67867 / 96900 = 70.04%**。
- 原始画像的 **86% 是条件口径**：在 `direction_stability == NO_DIRECTION` 子集内，
  `n_both_pos==0` 占比 = **44569 / 96900 = 45.97%**（该子集本身约占 53.5%，
  45.97% ≈ 0.535 × 0.86）。所以"86%"是**条件在 NO_DIRECTION 上**的比例，不是全样本。
- 在原始 86% cohort（44569 联系）内重新用新语义询问：
  - **90.63%** 在至少一个 risk 档存在**单边 delivery**（LONG_ONLY 或 SHORT_ONLY）。
  - 仅 **9.21%（4103 联系）** 在七个 risk 档全部为 `NO_DELIVERY`。
- 结论：旧"86% 没路径"的说法**证据不足且被正式撤回**。真正"两侧在所有尺度都无 delivery"
  的联系只是旧 cohort 的约 9%。

但**在固定且较紧的 stop 尺度下，真实 NO_DELIVERY 是显著存在的**（见 §2）。

---

## 2. 每个 risk 下六类路径各占多少？

`opportunity_outcome_by_risk.csv` / `delivery_curve_by_risk.csv`（primary_eligible = 两侧均有 active target）：

| risk_ATR | n_primary | P(DELIVERY) | P(NO_DELIVERY) | P(UNRESOLVED) | P(NO_TARGET_ENV)整体 |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 93360 | 0.305 | 0.531 | 0.165 | 0.023 |
| 0.50 | 93360 | 0.528 | 0.414 | 0.058 | 0.023 |
| 0.75 | 93360 | 0.664 | 0.316 | 0.021 | 0.021 |
| 1.00 | 93360 | 0.744 | 0.246 | 0.009 | 0.021 |
| 1.50 | 93360 | 0.846 | 0.150 | 0.003 | 0.020 |
| 2.00 | 93360 | 0.901 | 0.097 | 0.000 | 0.018 |
| 3.00 | 93360 | 0.955 | 0.044 | 0.000 | 0.018 |

`BOTH_DIRECTION_DELIVERY`（双向同时到达）在任何单一尺度都很少——这正是旧
`LEGACY_NO_BIDIRECTIONAL_REACH` 看起来"几乎都没 reach"的原因：双向同达本就罕见，
但**单边 delivery 在绝大多数 contact 上都发生了**。

---

## 3. Opportunity base rate 如何随 ATR 变化？

`P(DELIVERY)` 随 stop 放宽**单调上升**（0.25ATR 30.5% → 3.0ATR 95.5%），
`P(NO_DELIVERY)` 单调下降（53.1% → 4.4%）。这是物理必然：stop 越宽，越容易在
stop 之前触达某个事前 active liquidity target。**真正的"无机会"信号集中在紧 stop
（0.25–1.0ATR）**——那里 NO_DELIVERY 占 25%–53%。这正是 Opportunity 最有价值、也
最该被事前识别的区间。

---

## 4. 哪些事前状态与 delivery 差异最大？

由 §5–§6 的模型增量可知，**差异最大的状态全部属于 B2（liquidity / target geometry）**：
`nearest_above_R` / `nearest_below_R`、`same_price_identity_count`、各 scope 标准化距离
分箱、以及 `n_targets`（active target 计数）。逐 risk 的 DELIVERY vs NO_DELIVERY
连续特征均值对比见 `opportunity_profiles/feature_means_by_risk.csv`。趋势（B3）与
OB 上下文（B4）在描述层也未显示出与 delivery 的系统差异。

---

## 5. 单块信息：contact / liquidity / trend / OB 谁最强？

`block_marginal_deltas.csv`（相对 M0 的 ΔROC-AUC，跨 WF 均值）：

| 块 | 0.25 | 0.50 | 0.75 | 1.00 | 1.50 | 2.00 | 3.00 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **M_liquidity (B0+B2)** | +0.082 | +0.131 | +0.140 | +0.155 | +0.151 | +0.166 | +0.148 |
| M_contact (B0+B1) | +0.014 | +0.021 | +0.020 | +0.025 | +0.016 | +0.024 | +0.017 |
| M_trend (B0+B3) | +0.001 | −0.002 | −0.001 | −0.001 | −0.003 | −0.003 | −0.005 |
| M_ob (B0+B4) | −0.004 | −0.002 | −0.003 | −0.005 | −0.009 | −0.007 | −0.003 |

**结论：B2（流动性 / target 几何）是唯一强信息块**（ΔAUC +0.08~+0.17）。
B1（contact geometry）只有微弱增量；**B3（趋势）与 B4（OB）≈0，甚至微负**。
这与时间稳定性审计中"趋势对 robust direction 增量≈0"完全自洽。

---

## 6. 在 Full model 中，哪个块仍有独立增量？

`block_conditional_deltas.csv`（Full − Full_minus_block 的 ΔROC-AUC）：

| 移除的块 | 0.25 | 0.50 | 0.75 | 1.00 | 1.50 | 2.00 | 3.00 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **liquidity** | +0.063 | +0.105 | +0.116 | +0.129 | +0.133 | +0.140 | +0.128 |
| contact | −0.006 | −0.005 | −0.004 | −0.002 | −0.003 | −0.003 | −0.004 |
| trend | +0.001 | −0.001 | −0.000 | −0.001 | −0.001 | −0.002 | +0.001 |
| ob | −0.001 | −0.001 | −0.000 | −0.001 | −0.002 | −0.002 | +0.000 |

**只有 liquidity 在被移除后让 Full 显著掉点**（掉回 M0 水平）；
移除 contact / trend / ob 后 Full 几乎不变。**H4 强确认：OB 主要代理 liquidity/geometry，
无独立上下文价值。**

---

## 7. Opportunity 是否在 WF1–WF3 都可预测？

是。`model_metrics_by_walkforward.csv`：

| WF | M0 | M_liquidity | M_full | M_trend | M_ob |
|---|---:|---:|---:|---:|---:|
| WF1 | 0.624 | 0.750 | 0.746 | 0.623 | 0.618 |
| WF2 | 0.608 | 0.756 | 0.750 | 0.604 | 0.599 |
| WF3 | 0.617 | 0.761 | 0.754 | 0.617 | 0.618 |

M_liquidity / M_full 的 AUC 在三个时间泛化折上**稳定且一致**（0.746 / 0.750 / 0.754），
说明信号是**时间可泛化**的，不是某一段历史特有的拟合。趋势与 OB 在任何折都≈M0。

---

## 8. 能否稳定识别 delivery rate 明显偏低的 contact？

能。`bottom20_avoidance_gain = base_delivery_rate − bottom20_delivery_rate`，
bootstrap 500 次 canonical trading_day 的 95% CI **全部为正**：

- 0.25ATR：增益 0.09–0.12（CI）
- 1.00ATR：增益 0.26–0.30（CI）
- 3.00ATR：增益 0.07–0.13（CI）

即：把模型判为"最不像机会"的 20% contact 过滤掉，实际 delivery rate 可下降
7–30 个百分点，且下降量在 bootstrap 下稳定为正。**这正是 Opportunity 阶段的
核心可用产物：一个"低 delivery 概率"过滤器。**

---

## 9. 哪些结果跨 15 品种稳定？

`metrics_by_symbol.csv`：以 M_liquidity 相对 M0 的 per-symbol AUC 增量计，
**15/15 品种在所有 risk 档均为正**（macro median delta 0.10→0.26，
`positive_symbols` 均值 = 15.0 / `eligible_symbols` = 15.0）。
B3/B4 的 per-symbol 增量≈0。信号不是由少数品种驱动，而是全样本结构一致。

---

## 10. 下一步建议

**裁决：LEARNABLE（按 §23 全部 5 条标准成立）**——Full 相对 M0 方向跨 WF 一致；
Brier/LogLoss 不系统恶化（实际更低）；bottom20 增益稳定低于 base 且 CI 为正；
15/15 品种方向一致；bootstrap delta 非单一时期贡献。

优先级维持：**Opportunity > Risk-dependent mechanism > Robust Direction**，
但方向被精炼为：

1. **Opportunity 值得继续**，但价值集中在**流动性 / target 几何（B2）**，而非 SMC 趋势/OB 叙事。
   一个基于 B2 的"低 delivery 概率"过滤器，在紧 stop（0.25–1.0ATR）下可稳定剔除
   一批真正无机会的 contact。
2. 下一轮应**拆开 B2 子特征**（`nearest` 距离 vs `same_price_identity_count` vs
   `n_targets` vs 各 scope 距离分箱），定位真正携带信号的那一项；并考虑在
   0.25–1.0ATR 这一 NO_DELIVERY 高占比区间做更细的 cutoff 研究。
3. 趋势（B3）与 OB（B4）在本研究问题下**无独立增量**，后续不宜再为它们单独建模。
4. 当 `trading_day >= 2026-09-07` 的 Oracle path 可解析后，用**真正前瞻性 OOS**
   验证本 walk-forward 结论（当前仅为内部时间泛化）。

---

## 输出文件清单（research/analysis_results/smc_opportunity_v1/）

- `OPPORTUNITY_PROTOCOL.json` / `feature_manifest.json`（冻结特征契约 + 哈希）
- `opportunity_labels.parquet`（per contact×risk 标签与 reach 标志）
- `opportunity_features.parquet`（104 列特征，行对齐标签）
- `opportunity_outcome_by_risk.csv` / `delivery_curve_by_risk.csv`
- `legacy_vs_true_delivery.csv`（双口径：conditional ~86% cohort / unconditional）
- `walkforward_audit.csv` / `oof_predictions.parquet`
- `model_metrics_by_risk.csv` / `model_metrics_by_walkforward.csv`
- `block_marginal_deltas.csv` / `block_conditional_deltas.csv` / `block_bootstrap_ci.csv`
- `metrics_by_symbol.csv` / `calibration_deciles.csv`
- `opportunity_profiles/feature_means_by_risk.csv`
- `OPPORTUNITY_AUDIT.json`（见同目录）

## 测试（协议 §26）

标签逻辑 17 项自测中可执行部分（1–7, 10, 11）均在 `build_opportunity_labels_v1.py`
内 `assert` 通过：best_R_lower>0 必判 CERTAIN_REACH；CENSORED/AMBIGUOUS(lower=0)
不被误判为 CERTAIN_NO_REACH；NO_ACTIVE_TARGET 不入 primary；NO_DELIVERY 要求两侧
确定 no-reach；DELIVERY 只需一侧 certain reach；Oracle/path 字段未进入 X（已排除于
feature_manifest）。第 8、9、12–17 项为流程约束，已在本报告与脚本结构中遵守
（decision-time 特征、同 risk 同 test rows、WF train 日早于 test、transformer 仅 train fit、
trading_day bootstrap、前瞻性 OOS 未回流）。
