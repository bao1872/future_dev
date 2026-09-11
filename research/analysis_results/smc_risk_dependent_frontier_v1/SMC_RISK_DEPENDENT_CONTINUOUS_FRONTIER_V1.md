# SMC Risk-dependent Continuous Frontier v1.0

> ⚠️ **PROVISIONAL（提交 `0b8c85d` 部分措辞暂标）**：以下第 2–3 节中"21.23% 是真实 RISK_DEPENDENT 机制""1.25 不是特殊尺度因为 2.5 更多""SINGLE_SWITCH 约 19.5%"三处结论，系用 **lower-bound dominance**（`long_R_lower`/`short_R_lower` 直接比较）得出，**绕过了 v1.2 的 uncertainty 语义**，已被 P1.5 推翻/修正。正式结论以 **P1.5**（`run_risk_dependent_frontier_p1_5.py` + `RISK_FRONTIER_P1_5_AUDIT.json`）为准。

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

> **上述第 2–3 节为 `0b8c85d` 的 lower-bound dominance 结果，已标 PROVISIONAL；P1.5 见下。**

---

## 3.5 P1.5：Frozen Direction Semantics Gate（修正 `0b8c85d`）

直接复用冻结 `oracle_risk_direction_v1_2.parquet` 的 `rr_direction`（枚举：`LONG_DOMINATES / SHORT_DOMINATES / TRADEOFF_OR_OVERLAP / UNRESOLVED_CENSOR / NO_COMPARABLE_TARGET`），**仅当相邻两档均为明确 LONG/SHORT 才定义 DIRECT switch**，不重扫 K 线。

**Contact 分类（冻结语义，N=96,900）：**

| cls | n | pct |
|---|---:|---:|
| NO_DIRECT_GRID_SWITCH | 78,390 | 80.9% |
| SINGLE_DIRECT_LONG_TO_SHORT | 8,621 | 8.9% |
| SINGLE_DIRECT_SHORT_TO_LONG | 8,425 | 8.7% |
| MULTI_DIRECT_SWITCH | 1,360 | 1.4% |
| UNRESOLVED_GRID_PATTERN | 104 | 0.1% |

**直接 L↔S 翻转 contact = 18,406（18.99%）**，与冻结 `direction_stability=RISK_DEPENDENT`（18,495）**高度一致、但非逐 contact 完全等价**：

- 冻结 `RISK_DEPENDENT` 中 direct switch = **18,391**（8613+8418+1360），另有 **104** 个 `UNRESOLVED_GRID_PATTERN`；
- 冻结 `UNRESOLVED` 类中还有 **15** 个 direct switch（8+7）；
- 故 P1.5 的 18,406 = 18,391（RD direct）+ 15（UNRESOLVED direct）。

两套定义都基于冻结 `rr_direction`，给出几乎相同的约 19% 规模，但边界处有差异（UNRESOLVED_GRID_PATTERN vs 跨类的 direct switch）。这不影响 P1.5 Gate。

> **P1.5 正式结论**：在冻结七档 Oracle 标签中，**约 19% 的 contact 表现出明确的 risk-dependent direction change**。此结论仅限离散七档标签上的结构事实，**尚未证明**是连续风险空间中的真实市场机制——那正是 P4 要验证的。

**Grid midpoint（仅描述，非连续临界风险）：** 各相邻档中点 direct switch 数随区间宽度增加（0.375→265、0.625→1376、0.875→1806、1.25→4314、1.75→4341、2.5→7702）；`switches_per_ATR_width`（n/区间宽）反而前段更高（0.375 段密度 1060/ATR，2.5 段 7702/ATR 因区间宽 1.0 而密度最低）。→ 旧"2.5 比 1.25 多"**不能**证明市场偏好 2.5（区间更宽），反过来也不能否定 1.25；正确结论仍是"旧 1.25 是网格中点，连续阈值待 P4 求"。

**对账 / 诊断：**
- Gate D 自洽：`ROBUST_LONG`/`ROBUST_SHORT` 含相反 direct switch = **0**（语义一致）。
- 旧 P1 的 6954 mismatch **全部**来自 `NO_COMPARABLE_TARGET × NO_TARGET_ENVIRONMENT` 且 `lb_reach=True 但 label≠DELIVERY`（6954/6954）→ 证实 lower-bound dominance 在"无可见可比较 target"时仍按正 lower-bound 误判可达，**绕过 uncertainty 语义**，非数据损坏。
- 血缘：四层 contact key 在当前冻结 v1.2 均为 **96,900**（union=96,900），用户记忆的 96,899 不复现（早期 pre-fix 中间计数）。

**P4 ROI Gate A–D 全 PASS → `CONTINUE_TO_CONTINUOUS_FRONTIER_P4`：**
- A：18,406 direct-switch contacts ≥ 5000（且 ≥5%）✓
- B：15/15 品种均 ≥100 direct switch ✓
- C：TB1–TB4 均有 switch，最大块占比 29.7% ≤ 60% ✓
- D：ROBUST_LONG/SHORT 无相反 switch ✓

> **结论**：按冻结的 uncertainty-aware 七档语义，约 **19% 的 contact 在离散风险标签中表现出明确且大规模的 risk-dependent direction change**——但这仍是离散七档 Oracle 标签上的结构事实，尚未证明是连续市场机制。下一步按 P4a（3 品种语义复现试跑）→ P4b（15 品种连续前沿）从原始 K 线重算 per-target `required_risk_ATR`，回答"这些翻转在连续 risk 空间里还剩多少、由哪级 target unlock 触发"。

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
