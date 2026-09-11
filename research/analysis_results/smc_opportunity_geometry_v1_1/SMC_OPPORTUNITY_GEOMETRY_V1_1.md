# SMC Opportunity Geometry Decomposition v1.1 — REPAIRED

> 实验：SMC Opportunity Geometry Decomposition v1.1
> 仓库：bao1872/future_dev　分支：main
> 基准提交：a746e3e5402e71a44cf57dbf1b718f4610f8e2bd
> Atlas 冻结提交：7ae7c1ae57b45bbc8bade1b5742fdc0353859c83
> 证据等级：内部因果时间泛化（TB1→TB4 walk-forward）。非独立 OOS。
> 前瞻性 OOS 边界：2026-09-07（未进入）。

> **状态：REPAIRED。** 提交 `275db5f` 的部分结论已被审计要求撤销并重跑。
> 先说最重要的更正：上一轮"G4 不存在 / UNAVAILABLE"是**错误结论**（扫描错误 + 模型脚本里把 `G4=[]` 硬编码）。G4（多周期 liquidity field）**真实存在于冻结 Atlas**，且本次完整接入模型后**测出了真实信号**。

---

## 0. 审计发现与本轮修复（对应 reviewer 的 P0–P7）

| reviewer 质疑 | 核查结果 | 处置 |
|---|---|---|
| G4 在冻结 Atlas 不存在 | **错误**：当前 `liquidity_state_snapshot_v1_2.parquet` 有 60 个 `_bin_` 列且非零；P0.2 重算 active density 与冻结 `active_density_v1_2.csv` **逐项 diff=0.0** | G4 真实接入；无需 worktree 重建 |
| UNRESOLVED 被当 NO_DELIVERY 训练 | **成立**：`primary_eligible` 含 UNRESOLVED 24244 行（0.25ATR 16.5%、0.50ATR 5.8%） | P1：`model_eligible = primary_eligible & label∈{DELIVERY,NO_DELIVERY}`；硬断言训练/测试仅 {0,1} |
| 数值变量被当分类 | **成立**：n_targets/same_price/prior_enter 等被 OneHot | P2：三路管线（categorical / spline / ordinary-numeric） |
| block bootstrap 实现错（np.isin 丢权重） | **成立** | P4：按天索引整块拼接 |
| Bottom20 用全期 base rate | **成立** | P5：每个 test fold / 每次 bootstrap 用自己的 base rate |
| AUC ties 不正确 | **成立** | P3：tie-aware Mann-Whitney AUC（单测误差 <1e-16） |
| 只报分别 CI、未报 paired Δ | **成立** | P6：paired Δ 直接 bootstrap |

---

## 1. P0 产物血缘审计

- **P0.1**：当前 snapshot 有 60 个 `*_bin_*` 列，全部非零（如 `5m_bin_(0,0.5]` 和=7747）。
- **P0.2**：用冻结公式 `density = mean(bin_(0,0.5] + bin_(-0.5,0])` 从当前 snapshot 重算，与冻结 `active_density_v1_2.csv` 逐项比较 **abs diff 全部 = 0.0**（5m/15m/1h/CONTIG_SESSION/TRADING_DAY/TRADING_WEEK）。
- **结论**：本地 artifact 与冻结 producer code **一致，无漂移**，不需要 P0.3 重建。
- **说明**：上一轮"G4 UNAVAILABLE / 60 列全零"是我（IDE）的扫描错误 + 模型脚本硬编码 `G4=[]` 共同造成的误报。本轮已纠正，G4 = `B2_BIN_COLS`（60 列，经 `_build_bin_features` 归并为 `5m_pos_*`/`5m_neg_*`）。

---

## 2. 修复后的分解结果（ROC-AUC，跨 WF 均值）

| 模型 | 0.25 | 0.50 | 0.75 | 1.00 | 1.50 | 2.00 | 3.00 | 内容 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| M0 | 0.569 | 0.598 | 0.620 | 0.638 | 0.658 | 0.673 | 0.702 | 仅元数据 |
| **M_G1** | **0.764** | **0.816** | **0.845** | **0.861** | **0.876** | **0.869** | **0.884** | 最近上/下 target 距离 |
| M_G2 | 0.570 | 0.602 | 0.620 | 0.636 | 0.657 | 0.672 | 0.691 | target 数量（中性） |
| M_G3 | 0.691 | 0.749 | 0.790 | 0.819 | 0.844 | 0.856 | 0.869 | event 前/后最近 liquidity |
| M_G4 | 0.707 | 0.755 | 0.780 | 0.799 | 0.819 | 0.829 | 0.832 | **多周期 liquidity field** |
| M_G5 | 0.604 | 0.645 | 0.665 | 0.677 | 0.695 | 0.709 | 0.717 | 同价重合 identity |
| M_simple | 0.764 | 0.816 | 0.845 | 0.861 | 0.875 | 0.868 | 0.882 | G1+G2 |
| M_field | 0.745 | 0.798 | 0.829 | 0.851 | 0.868 | 0.860 | 0.864 | G1+G2+G4 |
| M_B2_full | 0.745 | 0.798 | 0.829 | 0.851 | 0.868 | 0.861 | 0.868 | 全部 |
| M_ob | 0.563 | 0.596 | 0.614 | 0.632 | 0.656 | 0.673 | 0.699 | ≈ M0 |
| M_full | 0.743 | 0.796 | 0.827 | 0.850 | 0.867 | 0.860 | 0.864 | 全特征 |

**两个决定性观察：**
1. **M_G1 单独（0.764→0.884）几乎是最强模型**；在它之上加 G3/G4/G5 → 不提升，反而略降（M_field、M_B2_full 都比 M_G1 低）。
2. **M_G4 单独（0.707→0.832）其实很强** —— 证明多周期 liquidity field **不是没数据、也不是无效**。

---

## 3. Paired Δ（block bootstrap 95% CI，同一批事件上直接算 Δ）

| pair | 含义 | ΔROC（点估计，区间跨 risk×WF） | 判定 |
|---|---|---|---|
| G1_vs_M0 | 最近距离增量 | **+0.16 ~ +0.20，CI 全 > 0** | 主导信号 |
| G3_vs_M0 | event-relative 增量 | +0.11 ~ +0.17，CI 全 > 0 | 真实但冗余 |
| **G4_vs_M0** | **多周期 field 独立增量** | **+0.10 ~ +0.15，CI 全 > 0** | **真实独立信号** |
| G5_vs_M0 | 同价增量 | +0.014 ~ +0.039（部分 WF3 CI 含 0） | 弱 |
| G2_vs_M0 | target 数量增量 | ~0（中位 +0.017，WF2 略负） | 中性 |
| **field_vs_simple** | **在 G1+G2 之上再加 G4** | **−0.009 ~ −0.034，CI 多为负** | **无增量（冗余）** |
| B2full_vs_field | 在 G1+G2+G4 之上加 G3+G5 | ~0（±0.005，CI 含 0） | 无增量 |
| B2full_vs_simple | 全部 − 简单几何 | −0.003 ~ −0.035 | 略负（过拟合） |
| OB_vs_M0 | OB 边际增量 | ~0（CI 含 0） | **OB 无增量** |
| full_vs_fullminusob | OB 条件增量 | ~0（±0.003，CI 含 0） | **OB 无增量** |

---

## 4. 裁决：SIMPLE_GEOMETRY（带关于 G4 的关键限定）

满足 SIMPLE_GEOMETRY 的核心判据（按 reviewer P11）：`M_field − M_simple` 不为正（实际为**负**），paired ΔROC CI 不支持正增量。跨 WF 稳定（M_G1 = 0.825/0.810/0.810）、跨品种 15/15 正向（中位 +0.275）。

**但必须加关键限定（这是本轮最重要的发现）：**

> **G4（多周期 liquidity field）有真实、广泛的独立信号**——`G4_vs_M0` ΔROC +0.10~+0.15，15/15 品种正向（中位 +0.215），跨 WF 稳定。它**不是空的、也不是无效的**。
>
> 然而，一旦 G1（最近 target 距离）已经在模型里，G4 **不再提供额外信息**（`field_vs_simple` 为负）。也就是说，多周期 liquidity 分布确实承载信息，但这些信息**已被"最近 target 有多近"这一最简单的几何量几乎完全吸收**。

所以回答你最早的核心问题：

- **主要是 case A（简单目标距离几何）**：主导、非冗余的信号是 G1。
- **case B 在绝对意义上部分成立**（G4 对 baseline 有真实信号），**但在你关心的条件意义（控制 G1+G2 后 G4 仍增量）上不成立** → 不是"多周期 landscape 在简单几何之外还有稳定结构增量"。
- **结论倾向**：不要过度工程化；简单几何（尤其 G1）足够。G4 真实但冗余，可作为稳健性佐证而非增量特征。

---

## 5. OB freshness 修复 + 收口

- 修复：`nearest_*_ob_freshness` 改为 categorical（FRESH/RETESTED/missing），进入 OneHot。
- 结果：`M_ob − M0` 与 `M_full − M_full_minus_ob` 的 paired ΔROC 均 ~0、CI 含 0。
- **→ OB（B4）仍无稳定独立增量，正式收口为"无独立价值"。**

---

## 6. 机械基准（修复标签后）

`geometry_ratio = min(nearest_above_R, nearest_below_R) / risk_ATR`，**仅 RESOLVED 标签**。强单调：

| risk_ATR | ≤0.25 | (0.25,0.5] | (0.5,1] | (1,2] | >2 |
|---|---:|---:|---:|---:|---:|
| 0.25 | 0.832 | 0.896 | 0.856 | 0.641 | 0.265 |
| 1.00 | 0.964 | 0.925 | 0.838 | 0.677 | 0.433 |
| 3.00 | 0.993 | 0.973 | 0.939 | 0.793 | 0.496 |

单一维几何比值复现模型大部分效果。模型主要重新发现"目标近→易在止损前触及"的物理关系，而非复杂 SMC 理论。

---

## 7. 对 275db5f 结论的更正清单

| 275db5f 结论 | 状态 | 更正 |
|---|---|---|
| G4 UNAVAILABLE | **撤销** | G4 真实存在且接入；独立信号 +0.10~+0.15 |
| G2 强负贡献 | **撤销** | 正确管线下 G2 中性（~0） |
| SIMPLE_GEOMETRY 最终成立 | **降级** | 改为 SIMPLE_GEOMETRY（G4 真实但被 G1 吸收） |
| bootstrap CI 支持 | **撤销** | 旧 bootstrap 实现错误；本轮重做 |
| OB 无增量 | **维持**（且更干净） | 修复后 paired Δ 仍 ~0 |
| G1 强候选信号 | **维持** | 复跑后更强、更稳 |
| geometry_ratio 单调 | **维持**（标签修复后更干净） | — |

---

## 8. 输出文件

`research/analysis_results/smc_opportunity_geometry_v1_1/`
- `GEOMETRY_AUDIT.json` / `feature_manifest_v1_1.json`（G4 已标真实存在）
- `feature_type_audit.csv`（每列管线路由）+ `opportunity_features_v1_1.parquet`
- `walkforward_audit.csv` / `subblock_metrics_by_risk.csv` / `subblock_metrics_by_wf.csv`
- `subblock_marginal_deltas.csv` / `subblock_conditional_deltas.csv`
- `subblock_bootstrap_ci.csv` / **`paired_delta_bootstrap_ci.csv`**（paired Δ CI）
- `subblock_by_symbol.csv`
- `geometry_ratio_summary_repaired.csv` / `geometry_ratio_profile.csv`
- `ob_freshness_fix_audit.csv`（本轮内部 paired 比较）
- `SMC_OPPORTUNITY_GEOMETRY_V1_1.md`

## 9. 治理

- 未修改 Atlas v1.2；无 PnL / 方向模型 / 最佳 ATR 选择 / LightGBM / SHAP。
- 标签污染已修正；管线三路分离；block bootstrap 正确；AUC tie-aware。
- 不进入 Risk-dependent / Long-Short。
