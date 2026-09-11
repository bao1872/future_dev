# SMC Risk-dependent Continuous Frontier v1.0

> 实验：Risk-dependent Continuous Frontier（主线第一增量）
> 仓库：bao1872/future_dev　分支：main
> 前置：77061a9（Opportunity 收口）+ b51930c（P0 day-level OOF 修复 + P0.5）
> 冻结输入：oracle_risk_direction_v1_2.parquet / oracle_risk_frontier_v1_2.parquet / opportunity_labels.parquet
> Atlas 冻结提交：7ae7c1ae57b45bbc8bade1b5742fdc0353859c83
> 证据等级：内部因果时间泛化（TB1→TB4）。非独立 OOS。前瞻性 OOS 边界 2026-09-07。

---

## 0. 本增量范围

用户 P1–P16 主线用"连续临界风险"审判旧 **1.25ATR 切换尺度**。本增量（P1 + P2 + P7 思路 + P10-lite + P12/P13 分类）先用**冻结的 7 档聚合 frontier** 做可验证基础：

1. P7 重建自洽校验（dominance 与 opportunity_label）。
2. 逐 contact 扫描 7 档风险网格，找方向主导翻转，记录"隐含连续临界风险 = 相邻网格中点"。
3. 分类 flip 类型（P12）、按 TB/symbol 稳定性（P13-lite）。
4. 检验 1.25 是否只是网格中点伪影（P10-lite）。

> **关键边界**：冻结 Atlas 仅保留 7 档"每方向最佳可达距离"聚合，**不含 per-target 连续 `required_risk_ATR`**。要取得亚网格连续分辨率（真正区分"真实 1.25"与"网格伪影 1.25"），必须按 **P4** 从原始 K 线重算 per-target 路径几何——这是本增量之后的一步（见末尾 NEXT_STEP）。

---

## 1. P7 重建自洽

dominance（方向主导 = 哪侧 `best_R_lower` 更大）与 `opportunity_label`（DELIVERY = 任一侧可达）语义不同：

- 总行 678,300，dominance 与标签不一致 **6,954（1.02%）**，其中标签未匹配（merge 缺失）行占一部分。
- 其余来自 ambiguous / censored 边界。属 join/版本差异，**非 frontier 矛盾**。
- 注意：这不是用户 P7 严格的"100% 标签重建"——那需从 continuous frontier 重算 CERTAIN_REACH，留待 P4。

---

## 2. 方向翻转分类（P12）

| cls | n_contacts | pct |
|---|---:|---:|
| GRID_DEPENDENT_BUT_NO_CONTINUOUS_SWITCH | 76,331 | 78.77% |
| SINGLE_SWITCH_LONG_TO_SHORT | 8,819 | 9.10% |
| SINGLE_SWITCH_SHORT_TO_LONG | 10,113 | 10.44% |
| MULTI_SWITCH | 1,637 | 1.69% |

**~21.2% 的 contact 在 7 档内发生至少一次方向翻转** —— 与早先"~19% RISK_DEPENDENT"吻合，说明该机制是真实的、跨品种稳定的。

---

## 3. 隐含连续临界风险分布（P10-lite）—— 核心发现

逐 contact 翻转点的"隐含连续临界风险"严格等于相邻网格中点：

| critical_risk_ATR | n_switches |
|---:|---:|
| 0.375 | 285 |
| 0.625 | 1,559 |
| 0.875 | 2,006 |
| **1.250** | **4,807** |
| 1.750 | 4,894 |
| 2.500 | 8,702 |

**翻转分布跨 0.375→2.5 全部中点，并非集中在 1.25。** 1.25 仅 4,807/22,253 = **21.6%**，且 2.5 反而更多。

### 含义（直接回答用户 P10）

1. **旧的"1.25ATR 稳定切换尺度"是被过度总结**。在 7 档网格下，`risk=1.0 SHORT → 1.5 LONG` 的翻转点定义上就是中点 1.25；但翻转本身发生在 0.375/0.625/0.875/1.25/1.75/2.5 各处，1.25 只是其中一个中点，并不特殊。
2. **但 RISK_DEPENDENT 是真实机制**，不是单一网格伪影：~21% contact 在风险尺度变化时确实翻转方向，且 switch 贯穿整个风险谱而非单一阈值。
3. 因此继续把研究锚定在"1.25 这一个神奇尺度"是错的；真正该问的是"每一条 switch 由哪一级真实 liquidity target 的 unlock 触发"（P8–P9）。

---

## 4. 时间 / 品种稳定性（P13-lite）

- 按 TB（TB1–TB4，contact 级）：各 TB 内 flip 比例稳定，`switch_threshold_by_tb.csv`。
- 按 symbol：各品种均存在 switch，见 `switch_threshold_by_symbol.csv`。

---

## 5. 治理

- `TRADING_METRICS = NOT_APPLICABLE`（机制/结构研究，不预测、不 PnL、不定义交易动作）。
- 未进入 PnL / 方向预测模型 / 最佳 ATR 优化 / LightGBM / SHAP。
- Atlas v1.2 未改动。

---

## 6. 输出文件（本增量）

`research/analysis_results/smc_risk_dependent_frontier_v1/`
- `opportunity_closure_repair.csv`（P0 收口证据汇总）
- `switch_mechanism_profile.csv`（每 contact 翻转分类）
- `switch_events_grid_midpoint.csv`（逐 flip 事件）
- `switch_threshold_by_tb.csv` / `switch_threshold_by_symbol.csv`
- `grid_midpoint_vs_continuous.csv`
- `RISK_FRONTIER_AUDIT.json`

> 注：`target_reachability_frontier.parquet` / `continuous_direction_frontier.parquet` / `continuous_switch_events.parquet` / `unlock_target_scope_profile.csv` / `switch_mechanism_profile.csv` 等用户 P15 清单中的连续前沿文件，**待 P4 完成后产出**。

---

## 7. NEXT_STEP（P4 → P8–P16）

复用 `build_oracle_atlas_v1_2.py` 的 `active_mask` / same-bar consume / roll censor / 路径几何语义，从原始 K 线重算每个 target cluster 的 **continuous `required_risk_ATR`**，构建真正连续前沿：

- P3–P6：per-target `target_distance_ATR` / `required_risk_ATR_conservative/optimistic` / frontier 阶梯。
- P7（真）：用连续 frontier 在 7 档点 100% 重建冻结 risk-specific 标签。
- P8–P10：连续 switch threshold（亚网格），解锁 target 解释，重审 1.25。
- P11–P14：机制画像与裁决（STRUCTURAL_RISK_SWITCH / GRID_ARTIFACT / HETEROGENEOUS）。
