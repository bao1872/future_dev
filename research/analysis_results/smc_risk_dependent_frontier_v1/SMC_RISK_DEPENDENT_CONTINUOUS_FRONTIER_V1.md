# SMC Risk-dependent Continuous Frontier v1.0 —— P4b 连续前沿

> 实验主线：Risk-dependent Continuous Frontier。P1.5 用冻结 7 档证明 risk-dependent
> 在离散标签中真实且规模大（~19%）；P4a 验证连续 path_geometry 与冻结 producer 逐位
> 等价；P4b 在连续 risk 空间重建前沿并审判"离散翻转在连续空间还剩多少"。

## 0. 方法（关键修正）

- **连续前沿基础设施**：`cf_common.path_geometry()` 产出 `favorable/adverse`（ATR 单位）；
  `reconstruct_oracle_grid()` 在 7 档回放；`continuous_frontier_from_targets()` 从每个
  target 的 `required_risk_through` 直接建 staircase（**不离散到人为网格**）。
- **方向语义严格复刻冻结 `rr_direction` 的 DOMINATES 判据**：
  `LONG ⇔ long_R_lower > short_R_upper`（保守 LONG 仍胜过乐观 SHORT），否则 `TIE`（重叠）。
  初版误用"下界比下界"导致 1.9M 假 switch（中位 12.95 ATR），已纠正。
- **regime 限制**：仅在与冻结 7 档同区间 `risk ∈ [0, 3.0]` 内判定（超出该区间的远处
  target 不参与——那里未研究 direction）。这是与冻结 RISK_DEPENDENT 可比的前提。
- **15 品种**：AG AU CU AL SN NI RB I SC RU MA TA M P CF（冻结 Atlas v1.2 宇宙，禁止 LC）。

## 1. 复现验证

- 全 15 品种 `rr_direction` 与冻结 `oracle_risk_direction_v1_2` **100% 一致** → 连续
  基础设施可靠。
- P4a HARD GATE（AG/RB/I 冻结 LONG/SHORT_DOMINATES 100%）已 PASS。

## 2. 连续临界风险（回答"旧 1.25 审判"）

| 量 | 连续 critical_risk_ATR |
|---|---|
| median | **2.20** |
| p25 | ~1.8 |
| p75 | ~2.6 |
| p90 | ~2.8 |

按品种：NI 1.54 → TA 2.62，多数聚在 2.0–2.5。按 TB：TB1 2.22 / TB2 2.27 / TB3 2.31 /
TB4 2.13，无时期主导。

> **结论**：旧 `1.25` 是 7 档中点，**不是特殊尺度**；连续 switch 的真实尺度在
> **~2.2 ATR**（regime 上限附近）。"1.25 是不是最多"这一问法本身应替换为
> "连续阈值是否形成稳定尺度分布"——答案是：集中在上半区 ~2.2，且跨品种/时期稳定。

## 3. 主队列：冻结 RISK_DEPENDENT（18,495）连续分类

| classification | n | pct |
|---|---:|---:|
| AMBIGUOUS_SWITCH（经重叠区中介的方向变化） | 15,789 | 85.4% |
| DATA_END_CENSORED（数据末端、单次观察不能确认） | 2,647 | 14.3% |
| CLEAR_SINGLE_CONTINUOUS_SWITCH | 39 | 0.2% |
| CLEAR_MULTI_CONTINUOUS_SWITCH | 20 | 0.1% |
| NO_CONTINUOUS_SWITCH | 0 | 0.0% |

全样本（96,900 contacts）：93,554 无清晰 flip，仅 3,347 有任何清晰 flip。

## 4. 解锁 target scope 画像

- `CONTIG_SESSION` 占 47%，`CONTIG_SESSION|TRADING_DAY` 12.7%，`5m|CONTIG_SESSION|
  TRADING_DAY` 9.2%。
- → 连续 switch 的 unlock 主要由**连续交易时段级流动性**驱动，而非短周期 5m/15m。

## 5. 机制解读（核心）

冻结 7 档 `RISK_DEPENDENT` 是**粗采样假象**：在 7 个离散点上抓到了
"LONG 主导岛"与"SHORT 主导岛"，但连续空间里两岛之间被 **overlap / TIE 区域**隔开。
逐点验证：连续方向在 7 个网格点上**逐位等于**冻结 `rr_direction`，所以两岛本身是真
的；问题在于网格**错过了岛间的重叠区**，制造了"干净翻转"的错觉。

因此：
- 方向**确实随风险变化**——18,495 个 RISK_DEPENDENT 中 **0% 方向稳定**，即 ~100% 都变；
- 但变化是**经重叠区中介的过渡**，不是干净的 dominance 翻转（85.4% AMBIGUOUS vs 0.3% CLEAR）。

即：**机制真实，但非干净 switch**；它是 overlap-mediated、阈值偏高（~2.2 ATR）、由
CONTIG_SESSION 级流动性 unlock 的过渡。

## 6. ROI 裁决

| 候选 | 判据 | 是否满足 |
|---|---|---|
| STRUCTURAL_RISK_SWITCH | 多数事件清晰连续 switch 且非 censor/ambiguous 主导 | **否**（CLEAR 仅 0.3%） |
| GRID_ARTIFACT | 连续化后大量事件无真实 switch | **部分**（无清晰 switch，但方向确实变，0% NO_SWITCH） |
| HETEROGENEOUS_MECHANISM | 连续 switch 真实存在但阈值/scope 按品种或时期分裂 | **最接近**（阈值~2.2 跨品种稳定、scope 偏 CONTIG_SESSION；但非"分裂"而是统一偏高） |

**裁决**：不是 STRUCTURAL，也不是纯 GRID_ARTIFACT，应为
**OVERLAP_MEDIATED_RISK_DEPENDENT_TRANSITION（HETEROGENEOUS 子类）**。
方向敏感性真实、可跨 15 品种/TB1–4 复现，但表现为重叠中介的过渡而非可交易的干净
翻转。

## 7. 下一步建议（待用户授权）

- 若按 `GRID_ARTIFACT` 严格字面：停止该主线（无干净 switch 可交易）。
- 若按 `HETEROGENEOUS`：方向机制真实，下一步应研究 **regime / overlap 中介结构**
  （不训练统一模型，不预测 switch）。
- **不建议**把 1.25 或 2.2 当作"突破尺度"直接做交易动作——本实验 `TRADING_METRICS=
  NOT_APPLICABLE`，未定义任何交易动作，仅回答机制是否独立。

> Governance: TRADING_METRICS=NOT_APPLICABLE；无交易动作、无 PnL、无预测。
> 冻结 Atlas v1.2 未被修改；cf_common 仅抽离共享语义，producer 未改。
