# SMC Risk-dependent Frontier — P4c 连续语义修复终稿

> 用户代码审计结论：P4a 接受；P4b 的连续 evaluator **未通过自己的 HARD GATE**
> （只存 `req_through`、没用 `req_before`；"100% 复现"实际验证的是 7 档
> `reconstruct_oracle_grid`，不是 `continuous_frontier_from_targets`；censor 未按
> risk 区间应用；2.2 统计被 unlock target 数加权；scope 没做 enrichment）。
> P4c 修复后重新裁决。

## 0. P4b 结论状态：PROVISIONAL / 已撤回

| P4b 暂标结论 | 状态 | P4c 修正后 |
|---|---|---|
| CLEAR = 0.32% | **撤回** | DIRECT = 99.26% |
| AMBIGUOUS = 85.4% | **撤回** | OVERLAP_MEDIATED = 0.74% |
| 连续尺度 2.2 ATR | **撤回** | 真实 direct median = 1.74 ATR |
| CONTIG_SESSION 47% = 重要 | **撤回** | enrichment ≈ 1.0，非特异 |
| OVERLAP_MEDIATED 过渡带 | **撤回** | overlap 罕见且窄（184 个，中位宽 0.54 ATR） |

P4b 的"85% ambiguous / overlap-mediated"完全是其 evaluator bug（只用 `req_through`
当 upper，丢了 `req_before` 这个独立阈值）的假象，不是真实机制。

## 1. P4c HARD GATE（连续 evaluator 自己过 Gate）

`continuous_side_bounds`（lower=max{dist:`required_through<r`}；upper=max{dist:
`required_before<r`}；`r>max_adverse`⇒`CENSORED_LOWER_BOUND`）+ `continuous_pair_direction`
（复刻冻结 `np.select`）直接在 7 档点与冻结 `oracle_risk_direction_v1_2.rr_direction` 比较。

- 15 品种 × 7 档 = **678,300 行，100.000% 一致**（每品种 100%，无 mismatch）。
- 因 `adv` 单调非减，`required_through<r ⇔ reach<stop_idx`、`required_before<r ⇔
  reach<=stop_idx`，与冻结 `oracle_direction` 逐位等价。

> 结论：连续 evaluator 现在**逐位复刻**冻结 v1.2，可独立信任。

## 2. 主队列：冻结 RISK_DEPENDENT（18,495）连续分类

| contact_class | n | share |
|---|---:|---:|
| **DIRECT_CONTINUOUS_SWITCH** | 18,358 | **99.26%** |
| OVERLAP_MEDIATED_TRANSITION | 137 | 0.74% |
| CENSOR_MEDIATED_UNRESOLVED | 0 | 0.00% |
| NO_CONTINUOUS_DIRECTION_CHANGE | 0 | 0.00% |
| SINGLE_SIDE_DOMINANCE | 0 | 0.00% |

**跨品种 / 时期普遍成立（结构性，非集中）：**

- 按品种 DIRECT 比例：SN 100.0% / AU 99.8% / RU 99.9% / SC 99.9% / CU 99.7% / P 99.7% /
  TA 99.7% / I 99.5% / RB 99.3% / AG 99.2% / M 99.1% / CF 99.2% / NI 98.4% /
  AL 98.5% / MA 97.9% —— 全部 97.9%–100%。
- 按 TB：TB1 99.21% / TB2 99.10% / TB3 99.37% / TB4 99.40%。

## 3. 真实连续尺度（修正统计单位）

统计单位改为 **transition_event（每 contact×转变一行）**，不再被 unlock target 数加权；
只用 primary RD cohort（18,495），不与全样本混算。

DIRECT switch 的 critical risk（RD）：

| n | median | p25 | p75 | p90 |
|---:|---:|---:|---:|---:|
| 19,846 | **1.7391** | 1.1538 | 2.3333 | 2.7143 |

- 真实连续尺度 **中位 ~1.74 ATR**，集中在 1.15–2.33，落在研究 regime（≤3.0）内。
- **不是 1.25**（旧网格中点，仅 Geometric 中点估计，无资格称真实尺度）。
- **不是 P4b 的 2.2**（那是 unlock-target 加权 + 全样本污染的假数）。

## 4. overlap 过渡带？——罕见且窄

- 处于两个 dominance 之间的 TRADEOFF overlap band：RD 中仅 **184** 个，中位
  `width = 0.54 ATR`、`center = 1.32 ATR`。
- 用户"很可能是多空 RR 区间重叠的过渡带"的倾向假设**未被支持**：方向在临界 risk
  处是**锐利的直接翻转**，不是宽过渡带。

## 5. scope enrichment（baseline-normalized）

背景 = RD cohort 内所有 reached target breakpoint 按 scope 的 indicator 出现率；
transition = RD 内 unlock target 的 scope indicator。

| scope | background | transition | enrichment |
|---|---:|---:|---:|
| 5m | 0.1601 | 0.1637 | 1.022 |
| 15m | 0.0798 | 0.0710 | 0.890 |
| 1h | 0.0373 | 0.0299 | 0.802 |
| CONTIG_SESSION | 0.4465 | 0.4751 | 1.064 |
| TRADING_DAY | 0.1987 | 0.1958 | 0.985 |
| TRADING_WEEK | 0.0777 | 0.0645 | 0.830 |

- 所有 scope enrichment ≈ 1.0。**switch 不特异于任何 scope**。
- CONTIG_SESSION 的 47% 只是其背景出现率（44.6%）的忠实反映，并非机制重要性。
- P4b "CONTIG_SESSION 47% = 重要" 的解释**被证伪**。

## 6. 裁决：DIRECT_STRUCTURAL_SWITCH

满足用户 P4b-8 / P4c-14 的 STRUCTURAL 条件：

- 多数 RD 事件（99.26%）为**清晰直接**的连续 LONG↔SHORT 翻转；
- 不是 censor / overlap 中介主导（两者合占 0.74%）；
- **15 品种、TB1–4 广泛存在**（97.9%–100% / 99.1%–99.4%）；
- 真实尺度稳定（中位 1.74 ATR）。

> **冻结 RISK_DEPENDENT 是一条真实、锐利、结构性的风险阈值翻转机制**：
> 在某一特定 critical risk（中位 ~1.74 ATR）处，某一侧的流动性交付确定性地
> 压过另一侧，方向由 LONG 主导翻为 SHORT 主导（或反之）。它不是网格采样假象，
> 也不是宽 overlap 过渡带，也不特定于某 scope。这正是 P4 要验证的机制问题的最终答案。

## 7. 与 P1.5 / 用户原始假设的对接

- P1.5 的"~19% 离散 Risk-dependent"现在被连续空间**坐实**：这 18,495 个 contact 在
  连续 risk 中 99.26% 仍表现为清晰的直接方向翻转，且每次翻转都能定位到具体
  liquidity target 的解锁（见 `continuous_transition_unlock_targets.parquet`）。
- 用户原始直觉"值得继续"得到支持；但机制形态比"找神奇 stop 阈值"更精确：
  是**风险尺度上的一个 sharp switch regime**，而非某个点突然多翻空。

## 8. 下一步（待用户授权）

按用户 P4c 指令：**本轮只修语义/分类/统计单位，不预测、不加模型、不 PnL、不 regime 模型、不转 regime 研究**。本实验 `TRADING_METRICS=NOT_APPLICABLE`，未定义任何交易动作。

给出修复后的机制结论后，下一步方向（需用户明确授权）包括：
1. 研究 critical risk ~1.74 ATR 是否可被具体 liquidity 结构（cluster size / 相对距离）预测；
2. 验证该 switch 在 out-of-sample / 不同品种组上的稳健性；
3. 评估是否构成可交易的"风险尺度区间"语义（仅在定义交易动作后才进入策略治理）。

> Governance: TRADING_METRICS=NOT_APPLICABLE；无交易动作、无 PnL、无预测。冻结
> Atlas v1.2 未被修改；cf_common 仅抽离共享语义，producer 未改。

## 9. 配套产出

- `P4C_CONTINUOUS_SEMANTIC_AUDIT.json` — Gate + 裁决 + 统计
- `p4c_frozen_grid_reconstruction.csv` — 逐品种 Gate 匹配率
- `rd_transition_classification.csv` — 每 contact 分类（primary + 全样本）
- `continuous_transition_events.parquet` — transition 事件（不入 git）
- `continuous_transition_unlock_targets.parquet` — 每事件 unlock target（不入 git）
- `overlap_band_profile.parquet` / `overlap_band_profile.csv` — overlap band
- `direct_switch_threshold_profile.csv` / `censor_transition_profile.csv`
- `scope_background_vs_transition.csv` — enrichment
