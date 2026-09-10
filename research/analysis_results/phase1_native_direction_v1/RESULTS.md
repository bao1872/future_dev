# Phase 1 — Native-Direction OB Tradability

**结论：FAIL**

本实验与旧 `research/phase1_tradability` 完全独立。旧结论（58.88% base rate、
direction-agnostic UP-or-DOWN 标签、12bar primary、103 维冻结模型、
Phase 1 discovery PASS）在本实验中一律视为 INVALID，未被引用。

---

## 0. 定义（本实验锁死）

| 项 | 值 |
|---|---|
| candidate | canonical `OB_ENTERED` 事件（V3 Source Owner 直接产生） |
| native_direction | `source_ob_bias`，+1 = Bullish/LONG，−1 = Bearish/SHORT |
| reference_price | trigger bar **close**（测量坐标原点，不是 entry_price） |
| R_ref | trigger bar 的 causal ATR5（统计测量尺，不是最终止损） |
| scan start | `touch_5m_bar_index + 1` |
| target / stop | ±2.5R / ∓1R，沿 native_direction |
| horizon | **无**——让 target 与 stop 自然分出胜负 |

native_direction 语义由 Source Owner 代码三处独立确认：
`build_ob_candidate_universe_v3.py:1187`（bias==1 记为 "bull"）、
`:386`（bias==1 时 far edge = zone_low，即止损在下方 → LONG）、
`ob_trigger_snapshot.py:436`（突破 pivot_high 的事件 bias=+1）。

---

## 1. 实际 candidate universe

| 项 | 值 |
|---|---:|
| 事件总数 | **79,896** |
| 品种 | **15** |
| source_tf | 5m / 15m / 1h |
| 排除 | **LC**（`quantile train < 600`，不降低门槛凑数） |

按 source_tf：5m 47,889 / 15m 22,285 / 1h 9,722。

## 2. LONG / SHORT

| | 数量 | 占比 |
|---|---:|---:|
| LONG (native=+1) | 40,605 | 50.8% |
| SHORT (native=−1) | 39,291 | 49.2% |
| LONG/SHORT | 1.0334 | |

## 3-5. native vs flipped placebo

RESOLVED 79,728；两者都成功解析 79,541。

| | native | flipped | 差 | 95% CI |
|---|---:|---:|---:|---|
| **ALL** | **0.2907** | **0.2959** | **−0.0051** | **[−0.0167, 0.0069]** |
| LONG | 0.2960 | 0.2890 | +0.0071 | [−0.0139, 0.0274] |
| SHORT | 0.2853 | 0.3030 | −0.0177 | [−0.0374, 0.0023] |

**所有区间均含 0。** native 与 flipped 都紧贴 1/3.5 = 28.57% 的几何基准，
差异不可分辨。**canonical OB 的原生方向在本测量口径下不携带可用的方向信息。**

（native 单独：23,126 target / 56,602 stop，base rate **0.2901**）

## 6. Resolution 自然耗时

| | p10 | p25 | median | p75 | p90 | p95 | p99 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bars | 1 | 2 | **5** | 13 | 25 | 39 | 87 | 1407 | 11.17 |

median 5 根 ≈ 0.42 小时。LONG 5 / SHORT 5；5m 5 / 15m 5 / 1h 6。
**本分布只用于描述，不用于选择任何 horizon。**

## 7-8. 模型 vs baseline

| universe | 模型 | AUC | PR-AUC | Lift@20 | Top20 uplift |
|---|---|---:|---:|---:|---:|
| ALL15_refit | Baseline0 constant | 0.5000 | 0.2846 | 1.0000 | 0.0000 |
| ALL15_refit | **Baseline1 (dir+tf+symbol)** | **0.5146** | 0.2967 | **1.1042** | 0.0297 |
| ALL15_refit | BaselineZS (dir+tf) | 0.5134 | 0.2962 | 1.0897 | 0.0255 |
| ALL15_refit | **Logistic** | 0.5061 | 0.2862 | 1.0088 | 0.0025 |
| ALL15_refit | LightGBM | 0.5067 | 0.2903 | 1.0174 | 0.0049 |

**Full Logistic 输给 metadata baseline**（AUC 0.5061 < 0.5146，
Lift@20 1.0088 < 1.1042）。LightGBM 未提供额外可泛化信息。

## 9. 十分位梯度（Logistic，pooled）

| D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2773 | 0.2871 | 0.2746 | 0.2758 | 0.2798 | 0.2868 | 0.2926 | 0.2975 | 0.2959 | **0.2782** |

**无单调梯度**：最高组 D10 反而低于 D9，D1 与 D10 几乎相同。

## 10-13. 分层诊断

| 折 | Lift@20 | | source_tf | Lift@20 |
|---|---:|---|---|---:|
| F1 | 1.0500 | | 5m | 0.9914 |
| F2 | 1.0257 | | 15m | 1.0360 |
| F3 | 0.9761 | | 1h | 1.0290 |
| F4 | 1.0047 | | | |

**0/4 折 > 1.05。**

方向：LONG **0.9929** / SHORT 1.0085 → LONG 为负。
品种：8/15 为正，macro 中位 **1.0014**，mean 1.0065。

## 14. DEDUP

| | Logistic | LightGBM |
|---|---:|---:|
| ALL_EVENTS | 1.0088 | 1.0174 |
| DEDUP_EVENTS | **1.0007** | 1.0290 |

Logistic 在去重后完全消失。

## 15. Zero-shot（DEV4 训练 → 未见品种）

| 测试集 | 模型 | AUC | Lift@20 | D10−D1 |
|---|---|---:|---:|---:|
| DEV4（开发品种） | Logistic | 0.5218 | 1.0418 | 0.0662 |
| **NEW11（未见）** | **Logistic** | **0.4985** | **1.0008** | 0.0026 |
| NEW11 | LightGBM | 0.5091 | 1.0149 | 0.0126 |

**DEV4 上看到的微弱信号（AUC 0.522）在未见品种上完全消失（0.4985）。**

## 16. Bootstrap（trading_day block，500 次）

| 模型 | Top20 uplift CI | D10−D1 CI |
|---|---|---|
| Logistic | [−0.0126, 0.0169] | [−0.0321, 0.0321] |
| LightGBM | [−0.0100, 0.0197] | [−0.0025, 0.0767] |

均含 0。

---

## 裁决（对照预注册标准）

| 标准 | 要求 | 实测 | 判定 |
|---|---|---:|:--:|
| A | Logistic AUC & Lift > metadata baseline | 0.5061 / 1.0088 vs 0.5146 / 1.1042 | ❌ |
| B | pooled Lift@20 ≥ 1.10 | 1.0088 | ❌ |
| C | ≥3/4 折 Lift@20 > 1.05 | 0/4 | ❌ |
| D | 多数 symbol >1 且 macro 中位 > 1.05 | 8/15，中位 1.0014 | ❌ |
| E | LONG 与 SHORT Lift@20 均 > 1 | LONG 0.9929 | ❌ |
| F | DEDUP Lift@20 > 1.05 | 1.0007 | ❌ |
| G | Top20 uplift 95% CI 下界 > 0 | −0.0126 | ❌ |

### **Phase 1 native-direction tradability：FAIL**

## SYNTHETIC REFERENCE METRIC ONLY

    reference_mean_R = 3.5 * 0.2846 - 1.0 = -0.0039 R

**不是策略收益**：未考虑真实 entry、滑点、手续费、Phase 2 止损放置与执行。
不计算 Sharpe。

---

## 已知限制

1. 只用 5m OHLC，同 bar 内 target/stop 先后不可知（150 例记 AMBIGUOUS 并排除）。
2. R_ref 用 ATR5 统一尺度，可能低估/高估个别品种的真实波动；本实验禁止同时搜索其它 R 定义。
3. 未设 time-stop；少数事件直到 1407 根才 resolution（p99 = 87）。
4. 15 品种而非 16（LC 数据合同不满足）。
5. 结论只对"触发时沿 OB 原生方向、2.5R/1R 参考交易"这一口径成立。

## 含义

按预注册停止规则：**不进入 Phase 2、不做 entry/exit 优化、不做强化学习、
不回到 FOLLOW/FADE 框架。**

本轮最硬的负面事实是两条：

1. **native 与 flipped 的成功率都停在 28.57% 几何基准上**（29.07% vs 29.59%，
   差 −0.0051，CI 含 0）。OB 原生方向在本口径下没有可测的方向优势。
2. **metadata baseline（direction+source_tf+symbol，Lift 1.104）明显强于
   179 维全模型（Lift 1.009）**，且 DEV4 的微弱信号在未见品种上归零。

即：不是"模型不够好"，而是**事件时的 causal state 在此标签下几乎不携带
关于 native-direction 成功的排序信息**。
