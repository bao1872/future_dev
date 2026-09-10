# Phase 1 — OB Event Tradability（结果说明）

## 研究问题

> 一个 canonical OB 触发事件，在不知道未来交易方向和具体买卖点的前提下，
> 是否值得进入后续交易研究？

标签：**是否存在至少一侧 1:2.5 的参考机会**（方向中性）。
不使用 12 根固定 horizon，采用 first-passage barrier resolution。
**本阶段不进入 Phase 2（方向 / 买卖点 / 止损 / 退出）。**

## 继承的前提（不重新讨论）

正常休市不是缺K线；未来路径用 next valid bars；真实缺失与 rollover 单独处理；
`candidate_id` 显式关联（禁止 positional concat）；同事件派生行不得跨
train/selection/test；特征只能来自决策时刻已知信息；62 维中大量字段为
action-relative，Phase 1 禁用；TEMPORAL 与 FOLLOW/FADE 方向模型本阶段不继续；
target 选择 / 动态退出 / 强化学习全部暂停；当前 407 个交易日属于 **discovery**，
不得称为最终独立 OOS；连续主力 rollover 仍为研究限制。

## 决策时点与参考尺

    decision_bar_index = touch_5m_bar_index
    reference_price    = close[decision_bar_index]
    R_ref              = ATR5（真实波幅 5 根滚动均值，decision bar 及之前）
    start_idx          = decision_bar_index + 1

R_ref 只是判断「有没有足够机会」的统一尺子，**不是最终止损 / 仓位风险 / 入场方案**。

## 候选事件

21,481 个 canonical entered 事件，已覆盖全部 source_tf：

| source_tf | 事件数 |
|---|---:|
| 5m | 12,939 |
| 15m | 6,043 |
| 1h | 2,499 |

按品种：AG 6,365 / CU 5,632 / RB 4,851 / M 4,633。
原 `event_index_v2` 的 20,008 是其真子集，因此直接复用 `candidate_id`。

## 标签

| status | 数量 | 占比 |
|---|---:|---:|
| RESOLVED | 21,421 | 99.72% |
| AMBIGUOUS_INTRABAR | 56 | 0.26% |
| END_OF_DATA_CENSORED | 4 | 0.02% |
| ROLL_CENSORED / BAD_INDEX / NO_ATR | 0 | 0% |

- tradable = 1：**12,613**
- tradable = 0：**8,808**
- **base rate = 58.88%**

AMBIGUOUS 比例 0.26%，远低于需要讨论更细周期数据的阈值，继续。

## Resolution（bars）

| 维度 | p25 | p50 | p75 | p90 | p95 | p99 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 全部 | 5 | **10** | 19 | 35 | 51 | 103 | 791 | 16.8 |
| 5m | 5 | 10 | 19 | 33 | 48 | 93 | 791 | 16.1 |
| 15m | 5 | 10 | 20 | 37 | 53 | 103 | 791 | 17.2 |
| 1h | 5 | 11 | 21 | 39 | 62 | 129 | 662 | 19.3 |

中位数 10 根 = **50 分钟**，即这是一个**短周期机会标签**（不是长期持有标签）。

## Label availability purge

| 折 | TRAIN 可用/原始 | TRAIN 删除 | SELECTION 可用/原始 | SELECTION 删除 |
|---|---:|---:|---:|---:|
| F1 | 10,520 / 10,527 | 7 | 1,990 / 1,996 | 6 |
| F2 | 12,517 / 12,523 | 6 | 2,043 / 2,053 | 10 |
| F3 | 14,566 / 14,576 | 10 | 2,160 / 2,165 | 5 |
| F4 | 16,736 / 16,741 | 5 | 2,121 / 2,133 | 12 |

因 resolution 很快（中位数 50 分钟），purge 只删掉个位数事件，未造成样本不足。

## 特征

`PHASE1_FEATURES_V1` = **105** 维（详见 `phase1_feature_contract_v1.csv`）：
风险几何 30、OB属性 19、结构 18、趋势 12、DSA 9、动量 9、多周期背景 4、其他 2。

排除 79 项，含：全部 62 维 action-relative、`target_fit_*`、`target_R`、
`trade_direction`、`trade_mode`、所有 `*_rel_*`、`stop_structure_*`、
`*_zone_low/high`、`*_level_`；`symbol` / `source_tf` 作为分组变量仅由
Baseline1 使用，不进入全模型。

硬断言全部通过：candidate_id 唯一、tradable ∈ {0,1}、resolution_time >
decision_time、特征集与动作字段无交集、`_rel_` 计数 0、无 `target_fit_*`。

## 模型结果（合并口径，n=8,898，base rate 0.5901）

| 模型 | AUC | PR-AUC | Brier | Lift@10 | Lift@20 | Lift@30 | Lift@50 | 十分位Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline0 base rate | — | — | — | — | — | — | — | — |
| Baseline1 symbol+source_tf | 0.4945 | 0.5803 | 0.2419 | 0.9492 | 0.9487 | 0.9790 | 0.9838 | −0.109 |
| **Model1 Logistic** | **0.5282** | **0.6095** | 0.2468 | **1.0846** | **1.0516** | **1.0565** | **1.0310** | **0.806** |
| Model2 LightGBM | 0.4980 | 0.5845 | 0.2596 | 0.9836 | 1.0001 | 0.9974 | 0.9952 | −0.188 |

**关键：信号由线性模型捕获，LightGBM 过拟合**（105 维 / ~1 万样本）。

## 裁决对照（预注册 7 条）

| # | 标准 | 结果 |
|---|---|---|
| 1 | LightGBM 明显超过 symbol+source_tf | ⚠️ **未达成**（0.498 vs 0.4945，仅微弱；Logistic 0.5282 明显超过，达成） |
| 2 | Lift@20 > 1 且有幅度 | ✅ 1.0516 |
| 3 | 多数折 Lift@20 > 1 | ✅ 4/4（1.003 / 1.011 / 1.111 / 1.110） |
| 4 | 非单一品种驱动 | ✅ AG 1.133 / M 1.109 / RB 1.026 为正；CU 0.940 |
| 5 | 非单一 source_tf 驱动 | ✅ 1h 1.118 / 5m 1.037 / 15m 1.035 全为正 |
| 6 | 十分位有梯度 | ✅ Spearman 0.806 |
| 7 | DEDUP 不完全消失 | ✅ 1.0516 → 1.0393 |

## 结论

**存在可学习的弱信号，但幅度有限且仅由线性模型稳定捕获。**
事件发生时的绝对状态能把「是否存在至少一侧 2.5R 机会」的识别率从 base 59%
提升到 Top20% 约 62%（Lift 1.05）。这**不足以**说明知道怎么买、知道方向、
可以盈利。

按第 11 条前提：本结果属于同一段 407 日数据上的 discovery，
**不得作为最终独立 OOS 证据**。

## Phase 1A — Tradability Signal Sanity Check

不训练新模型，只用现有标签 + 确定性折叠拟合取回逐事件分数。

### A. decile × 累计成功发生率（Logistic）

| decile | by 6 | by 12 | by 24 | by 48 | by 96 | 最终(无限) |
|---|---:|---:|---:|---:|---:|---:|
| D1 | 0.1753 | 0.2921 | 0.4034 | 0.4652 | 0.5124 | 0.5360 |
| D10 | 0.2539 | 0.4258 | 0.5427 | 0.6112 | 0.6303 | 0.6393 |
| **差** | **+7.9pp** | **+13.4pp** | **+13.9pp** | **+14.6pp** | +11.8pp | +10.3pp |

D10 的同期 Lift：by6 **1.153**、by12 1.149、by24 1.099、by48 1.094、
by96 1.080、最终 1.085。

**关键：区分在 12–24 根 bar 内就已基本形成；无限延长反而稀释信号**
（48bar 时 +14.6pp → 无限时 +10.3pp；Lift 1.15 → 1.08）。
即模型抓到的是「触发后早期就会出现机会」的状态，不是靠长尾等待。

### B. 成功速度（Logistic）

| decile | 成功中位 bars | 成功均值 bars |
|---|---:|---:|
| D1 | 11.0 | 23.49 |
| D10 | **8.0** | **15.18 |

Spearman(decile, 成功中位bars) = **−0.430**；
Spearman(decile, 成功均值bars) = **−0.321**。
高分事件不仅更容易成功，而且**明显更快**成功。

### C. source_tf × decile（success_by_24bar）

| source_tf | D1 | D10 |
|---|---:|---:|
| 5m | 0.4045 | 0.5388 |
| 15m | 0.3891 | 0.5175 |
| 1h | 0.3238 | **0.6190** |

三个周期 D10 均高于 D1；1h 最强。且三者 resolution 中位数接近
（5m 10 / 15m 10 / 1h 11 根），**没有数量级差异** → 不需要按 source_tf
人为设计不同 horizon。

### D. rollover 阈值敏感性

| 阈值 | RESOLVED | AMBIGUOUS | ROLL_CENSORED | base rate | 标签与10ATR不同 |
|---|---:|---:|---:|---:|---:|
| 3 ATR | 21,019 | 56 | 402 | 0.5888 | **0** |
| 5 ATR | 21,421 | 56 | 0 | 0.5888 | 0 |
| 10 ATR | 21,421 | 56 | 0 | 0.5888 | — |

**三个阈值下标签零变化。** 即便收紧到 3 ATR，仅 402 个事件（1.9%）被
ROLL_CENSORED，且剩余样本 base rate 完全不变（58.88%）。
rollover 未污染标签，此前 ROLL_CENSORED=0 是可信的，不是检测失效。

### Phase 1A 结论

命中用户设定的**情况 A**：区分早期形成、长尾稀释。
因此**不启动 canonical OB lifecycle / horizon 重建**
（未满足三个启动条件中的任何一个）。

## 可再生中间产物（不入库，已 .gitignore）

`candidates_v1.parquet`、`labels_v1.parquet`、`features_v1.parquet`、
`predictions_v1.parquet`、`phase1a_predictions.parquet`，
均由 `research/phase1_tradability/` 下脚本重建。
