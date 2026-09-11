# SMC Opportunity Geometry Decomposition v1.1

> 实验名称：SMC Opportunity Geometry Decomposition v1.1
> 仓库：bao1872/future_dev　分支：main
> 基准提交：a746e3e5402e71a44cf57dbf1b718f4610f8e2bd
> Atlas v1.2 冻结提交：7ae7c1ae57b45bbc8bade1b5742fdc0353859c83
> 特征契约哈希：55de426492f0f96b370bb7507348cb52f9cbd972e299c26c2df15704e51e2135
> **证据等级：内部因果时间泛化（TB1→TB4 walk-forward）。非独立 OOS。**
> 前瞻性 OOS 边界锁定：trading_day >= 2026-09-07（未进入本轮）。

---

## 0. 最必须先说的事：G4 在冻结 Atlas 中不存在

v1.1 协议把 G4 定义为：

> `5m/15m/1h/CONTIG_SESSION/TRADING_DAY/TRADING_WEEK` 固定距离 bin counts

**但冻结 Atlas v1.2 没有任何 per-scope 距离分箱列。** 已扫描全部 20 个冻结
parquet（`liquidity_state_snapshot_v1_2` 只有 `nearest_above_R / below_R / ahead_R /
behind_R`、`same_price_identity_count`、`active/historical_visible_count` 等；其余文件
均无 bin 列）。

**这同时解释了 v1.0 的一个隐藏问题**：v1.0 的 `B2_BIN_COLS` 列名是手写的
（`5m_pos_(0,0.5]` 等），与真实数据不匹配，所以 G4 在 v1.0 里**从头到尾就是空块**。
features parquet 中那 60 个分箱列全部恒为 0（遗留 `_build_bin_features` 产物），不携带
任何信息。

**结论**：v1.0 的"B2 压倒性最强"实际上**全部来自 G1+G2+G3+G5（简单几何）**，从未测到
多周期 liquidity field。本轮 case B（多周期 field 提供额外结构信息）**无法用冻结数据测试**。
要补 G4，必须从冻结原始 liquidity 账簿重建 per-scope 距离分箱——属于新特征工程，需你显式授权。

---

## 1. 修复了 OB freshness bug

`nearest_opposing_ob_freshness` / `nearest_same_direction_ob_freshness` 原被错误放入
`NUMERIC` 并 `pd.to_numeric(errors="coerce")` → 字符串 `FRESH`/`RETESTED` 全部变 NaN。
已改为 categorical，由 OneHotEncoder 处理。原始数据确有值
（`nearest_opposing_ob_freshness`：11783 空、`nearest_same_direction_ob_freshness`：
17552 空，其余为 FRESH/RETESTED）。

修复后定点复跑 `M_ob / M_full / M_full_minus_ob`（7 risk × WF1–WF3），对照 v1.0：

| 模型 | ΔROC（修复−v1.0） | 结论 |
|---|---|---|
| M_ob | 多数 ±0.01，最大 |0.013| | 无实质变化 |
| M_full | ±0.013 内 | 无实质变化 |
| M_full_minus_ob | ≈ M_full | OB 仍无独立增量 |

**→ 即使修复 freshness，OB 仍≈0。B4 可正式收口为"无稳定独立增量"。**

---

## 2. B2 拆成 G1–G5 后的分解

`subblock_metrics_by_risk.csv` / `subblock_marginal_deltas.csv`。ROC-AUC（跨 WF 均值）：

| 模型 | 0.25 | 0.50 | 0.75 | 1.00 | 1.50 | 2.00 | 3.00 | 内容 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| M0 | 0.543 | 0.573 | 0.599 | 0.612 | 0.637 | 0.651 | 0.701 | 仅元数据 |
| **M_G1** | **0.668** | **0.735** | **0.769** | **0.793** | **0.819** | **0.842** | **0.882** | 最近上/下 target 距离 |
| M_G2 | 0.532 | 0.555 | 0.567 | 0.579 | 0.601 | 0.620 | 0.641 | target 数量 |
| M_G3 | 0.629 | 0.684 | 0.721 | 0.745 | 0.781 | 0.812 | 0.868 | event 前/后最近 liquidity |
| M_G5 | 0.565 | 0.607 | 0.629 | 0.641 | 0.661 | 0.677 | 0.719 | 同价重合 identity |
| M_B2_full | 0.643 | 0.716 | 0.751 | 0.776 | 0.802 | 0.827 | 0.859 | G1+G2+G3+G5 |
| M_simple | 0.644 | 0.716 | 0.751 | 0.776 | 0.802 | 0.824 | 0.856 | G1+G2 |

**两个反直觉但关键的事实：**

1. **M_G1（最近 target 距离）单独 AUC ≥ M_B2_full 与 M_full**——即"最近 liquidity 多近"
   一个特征就几乎穷尽了全部信号。
2. **G2（target 数量）是负贡献**（ΔAUC −0.011~−0.061）。"目标多=更容易 delivery"的直觉
   不成立；`n_targets_L/S` 加入反而略降 AUC（可能与 G1 共线或自身噪声）。

conditional delta（从 M_B2_full 移除某块）：

| 移除 | 0.25 | 0.50 | 0.75 | 1.00 | 1.50 | 2.00 | 3.00 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **G1** | **+0.029** | **+0.041** | **+0.039** | **+0.039** | **+0.031** | **+0.023** | **+0.014** |
| G2 | −0.023 | −0.019 | −0.017 | −0.017 | −0.015 | −0.016 | −0.022 |
| G3 | −0.000 | +0.000 | +0.001 | +0.001 | +0.001 | +0.003 | +0.004 |
| G5 | −0.001 | −0.000 | −0.000 | −0.001 | −0.001 | −0.001 | −0.000 |

移除 **G1 掉点最多**（它是关键）；移除 G2 反而略升；移除 G3/G5 ≈0。**G1 就是全部信号。**

---

## 3. 机械基准：模型只是在重新发现简单几何

`geometry_ratio = min(nearest_above_R, nearest_below_R) / risk_ATR`
（目标距 ÷ 止损距），分箱后真实 delivery 率：

| risk_ATR | ≤0.25 | (0.25,0.5] | (0.5,1] | (1,2] | >2 |
|---|---:|---:|---:|---:|---:|
| 0.25 | 0.473 | 0.563 | 0.517 | 0.389 | 0.248 |
| 1.00 | **0.947** | 0.907 | 0.832 | 0.674 | **0.432** |
| 3.00 | **0.992** | 0.972 | 0.936 | 0.791 | **0.494** |

**强单调**：目标越近于止损，delivery 率越高；目标超过 2× 止损时 delivery 率腰斩。
一个一维几何比值即复现模型大部分效果——这正说明 Logistic 主要在学习
"目标近→容易在止损前触及" 的物理常识，而不是复杂 SMC 理论。

---

## 4. 裁决：SIMPLE_GEOMETRY（case A）

满足全部稳定性判据：

- **跨 WF 方向一致**：M_G1 AUC WF1/2/3 = 0.794 / 0.784 / 0.783；M_ob ≈ M0 在所有 WF。
- **bootstrap 95% CI 支持**：M_G1 全部 0.65–0.91，远高于 0.5。
- **多数品种同向**：M_G1 在 15/15 品种全为正（macro median delta 0.13→0.30）；M_ob 仅 5–10/15 为正、median≈0。
- **Bottom20 avoidance 稳定**：M_G1 的 bottom20_avoidance_gain 与 M_B2_full 一致（见 CSV）。

**结论**：
> Opportunity 的可预测性主要来自**最近 target 距离与 target 数量以外的简单邻近几何**；
> 不是复杂多周期 liquidity field，也不是趋势或 OB。
> 给定"目标离止损多远"这一个量，就能解释大部分 structural delivery 信号。

---

## 5. 下一步与未决项

1. **G4（多周期 liquidity field）缺口**：本轮无法测试 case B。要补全，需从冻结原始
   liquidity 账簿（`liquidity_field_snapshot_v1_1` / `liquidity_master_v1_1` 中真实的
   per-scope 流动性水平）重建距离分箱特征。**这是新特征工程，需你授权**，不在"禁止新增字段"
   的合规范围内。若你授权，下一轮即可真正回答"多周期 field 是否还有增量"。
2. **G2 负贡献**值得单独查：是共线还是 `n_targets` 噪声，可做一个 G1 控制后 G2 的偏增量。
3. **优先级不变**：Opportunity 仍是第一优先，但应定位为"基于目标邻近几何的低-delivery 过滤器"，
   而非复杂 liquidity 理论。无需上 LightGBM/SHAP——简单几何规则可能已足够。
4. **不要进入 Risk-dependent / Long-Short / PnL**。

---

## 输出文件（research/analysis_results/smc_opportunity_geometry_v1_1/）

- `GEOMETRY_PROTOCOL.json` / `feature_manifest_v1_1.json`（G4 诚实标 UNAVAILABLE）
- `opportunity_features_v1_1.parquet` / `opportunity_labels.parquet`
- `geometry_ratio_profile.csv` / `geometry_ratio_summary.csv`（机械基准）
- `subblock_metrics_by_risk.csv` / `subblock_metrics_by_wf.csv`
- `subblock_marginal_deltas.csv` / `subblock_conditional_deltas.csv`
- `subblock_bootstrap_ci.csv` / `subblock_by_symbol.csv`
- `ob_freshness_fix_audit.csv`（freshness 修复前后对照）
- `GEOMETRY_AUDIT.json` / `SMC_OPPORTUNITY_GEOMETRY_V1_1.md`

## 测试（§26 对应项）

- OB freshness 现为 categorical（FRESH/RETESTED/missing），不再被 `to_numeric` 清零。
- G4 列缺失 → 不在任何模型中（`G4=[]`），无静默空块被误当有效特征。
- 同 risk 所有模型共享相同 test rows；WF train 日早于 test；transformer 仅 train fit；
  bootstrap 用 canonical trading_day；前瞻性 OOS 未进入。
- 未修改 Atlas v1.2；无 PnL / 策略回测 / 最佳止损 / 方向模型 / LightGBM / SHAP。
