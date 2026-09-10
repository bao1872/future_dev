# Phase 1 — Native-Direction OB Tradability

**结论：FAIL**（经 nested metadata+state 审计确认）

前一轮结论曾改为 `PROVISIONAL FAIL — pending nested metadata+state audit`，
原因是原始比较为 **metadata-only vs state-only**（非嵌套）。
本轮补做嵌套比较 `M1 metadata` vs `M3 metadata+state`（见 §7b），
结论确认为 **FAIL**。

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

> ⚠️ 此处的 1.1042 来自把 `symbol` 当作**整数编码**的 baseline。
> 对名义变量这是不当编码（强加了不存在的序关系）。
> 改用 train-only one-hot 后，metadata baseline 的 Lift@20 = **1.0361**
> （见 §7b M1_metadata）。因此 metadata 的条件基准率信息是**存在但温和**的，
> 不是 1.10 那个量级。

## 7b. Nested 审计：M1 metadata vs M3 metadata+state

标签 / candidate / feature contract / split 全部冻结，未调参。
M3 与 M1 使用完全相同的 Logistic 参数、train-only imputer/scaler/one-hot。

| universe | 模型 | AUC | PR-AUC | Brier | Lift@20 | Top20 uplift |
|---|---|---:|---:|---:|---:|---:|
| ALL15 | M0 constant | 0.5000 | 0.2846 | 0.203586 | 1.0000 | 0.0000 |
| ALL15 | **M1_metadata** | **0.5072** | 0.2890 | 0.203636 | 1.0361 | 0.0103 |
| ALL15 | M2_state | 0.5061 | 0.2862 | 0.206004 | 1.0088 | 0.0025 |
| ALL15 | **M3_metadata_state** | **0.5039** | 0.2862 | 0.206071 | 1.0356 | 0.0101 |
| NEW11 | ZS_M1 (tf+dir) | 0.5080 | 0.2880 | 0.203248 | 1.0233 | 0.0066 |
| NEW11 | **ZS_M3 (tf+dir+state)** | **0.4995** | 0.2836 | 0.224177 | 1.0074 | 0.0021 |

### 增量（M3 − M1）

| universe | 比较 | ΔAUC | ΔPR-AUC | ΔBrier↓ | ΔTop20 uplift |
|---|---|---:|---:|---:|---:|
| ALL15 | M3 − M1 | **−0.0033** | −0.0028 | +0.002435 | −0.0002 |
| NEW11 zero-shot | M3 − M1 | **−0.0059** | −0.0048 | +0.003030 | −0.0090 |
| NEW11 zero-shot | ZS_M3 − ZS_M1 | **−0.0085** | −0.0044 | +0.020929 | −0.0045 |
| DEV4 | M3 − M1 | +0.0042 | +0.0067 | +0.000840 | +0.0157 |

**唯一的正增量出现在 DEV4**——即训练时见过其 symbol 的那一组。
在未见品种上增量全部为负。Brier 一律变差（越高越差）。

### 每折增量

| 折 | ALL15 ΔAUC | ALL15 Δuplift | NEW11_ZS ΔAUC | NEW11_ZS Δuplift |
|---|---:|---:|---:|---:|
| F1 | +0.0118 | +0.0151 | −0.0308 | −0.0367 |
| F2 | −0.0094 | +0.0080 | −0.0085 | +0.0175 |
| F3 | +0.0096 | −0.0067 | +0.0098 | −0.0064 |
| F4 | −0.0203 | −0.0363 | −0.0160 | −0.0029 |

ALL15 2/4 折 ΔAUC 为正；NEW11_ZS **1/4**。方向不稳定。

### Within-stratum（symbol × source_tf × native_direction 组内排序）

| 数据集 | 可用 strata | 事件加权 uplift | 事件加权 Lift | macro 中位 Lift | 正 strata |
|---|---:|---:|---:|---:|---:|
| ALL15 M3 | 86 | +0.0042 | 1.0105 | 1.0450 | 46/86 |
| NEW11 ZS_M3 | 63 | **−0.0038** | **0.9925** | 0.9784 | 28/63 |
| DEDUP M3 | 62 | +0.0006 | **0.9971** | 0.9949 | 31/62 |

严格控制品种 / 周期 / 多空后，事件状态在未见品种上的组内排序能力为负，
去重后基本归零（Lift 0.9971）。

### DEDUP

| | M1 | M3 | Δ |
|---|---:|---:|---:|
| AUC | 0.5041 | 0.5028 | **−0.0013** |
| Lift@20 | 1.0286 | 1.0313 | +0.0027 |
| Top20 uplift | 0.0082 | 0.0090 | +0.0008 |

### Paired day-block bootstrap（同一 resample 内同时算 M3 与 M1，500 次）

| 数据集 | ΔAUC 95%CI | ΔTop20 uplift 95%CI |
|---|---|---|
| ALL15 | [−0.0192, 0.0127] | [−0.0229, 0.0179] |
| NEW11 zero-shot | [−0.0301, 0.0146] | [−0.0321, 0.0276] |

**全部含 0。**

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

## 裁决（嵌套审计后的最终判据）

| 判据 | 实测 | 判定 |
|---|---|:--:|
| ALL15：M3 稳定超过 M1 | ΔAUC −0.0033，Δuplift −0.0002 | ❌ |
| NEW11 zero-shot：ZS_M3 稳定超过 ZS_M1 | ΔAUC −0.0085，Δuplift −0.0045 | ❌ |
| within-stratum uplift ≈ 0 | ALL15 +0.0042 / NEW11 −0.0038 / DEDUP 0.9971 | ❌ |
| paired bootstrap 增量 CI 含 0 | 全部含 0 | ❌ |
| DEDUP 后无增量 | ΔAUC −0.0013 | ❌ |

### **Phase 1 native-direction tradability：FAIL**

    Metadata contains modest conditional base-rate information,
    but event-time causal market state provides no reproducible
    incremental ability to rank native-direction OB quality.

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

本轮最硬的负面事实是三条：

1. **native 与 flipped 的成功率几乎相同**（29.07% vs 29.59%，差 −0.0051，
   paired CI 含 0）。最可靠的证据是这条 paired placebo 本身，
   而不是"29% ≈ 28.57%"——28.57% 来自理想化无漂移连续随机过程，
   真实市场有 drift / volatility clustering / gaps / 交易时段结构，
   不应把它当成必须精确命中的零假设。
2. **事件状态无法在 metadata 之上提供增量**：M3 − M1 的 ΔAUC 在 ALL15 为
   −0.0033、在未见品种为 −0.0059（ZS −0.0085），Brier 一律变差；
   唯一的正增量出现在训练时见过其 symbol 的 DEV4。
3. **组内排序（严格控制 symbol × source_tf × direction 后）在未见品种上为负
   （Lift 0.9925）、去重后归零（0.9971）**，paired bootstrap 增量 CI 全部含 0。

即：不是"模型不够好"，而是**事件时的 causal market state 在此标签下不携带
可复现的、能区分 native-direction OB 质量的排序信息**。

需要区分的是：这**不等于**"OB 什么都没有"。metadata 含有**很弱的**条件
基准率差异（one-hot M1 Lift@20 = 1.0361，AUC 0.5072）——
即不同 品种 × 周期 × 多空 组合的成功率存在小幅差异；
但单个 OB 触发时的状态无法进一步区分同类 OB 之间的质量。

（措辞已按裁决收紧：原表述"存在**稳定**差异"证据不足，改为"**很弱的**
条件基准率差异"。）

## 测量定义已审计（Reference Semantics Audit）

结论 `trigger-close reference is adequate`，见
`RESULTS_REFERENCE_AUDIT.md`：

- 中位事件在触发 bar 收盘时相对 canonical 进入边界仅位移 **+0.026R**；
- 按滞后程度分组的 base rate 平坦（0.2855–0.2961，极差 1.06pp），
  FAIL 不集中在"close 已滞后"的事件；
- near/far edge 定义与 canonical `touch_close_beyond_far_edge` 一致率
  **1.000000**（n=79,896）。

→ 因此本 FAIL 应被理解为：

    "以 trigger close + ATR5 作为统一参考交易，OB native direction
     的触发时状态不可筛选。"

而不是"数学上证明 OB 无效"。

## 最后一次 native-direction 实验：Structural OB label

用 OB 自己的失效边 `far_edge` 定义风险（不再用固定 1 ATR5）：

    R_struct = |close − far_edge|
    target   = close + d × 2.5 × R_struct
    stop     = close − d × 1.0 × R_struct ≡ far_edge

门控通过（`R_struct/ATR5` median **1.406**，仅 15.72% 落在 [0.8,1.2]；
1h 中位达 3.409 —— 原 ATR5 止损对 1h OB 确实过紧），故该实验有理由执行。

**结果同样 FAIL，且更差**（详见 `RESULTS_STRUCTURAL.md`）：

- native 0.2804 vs flipped 0.2936，diff **−0.0132**，
  CI **[−0.0254, −0.0002] 不含 0 → 反向显著更优**；
- M3 − M1 ΔAUC −0.0140（ALL15）/ **−0.0314**（NEW11 zero-shot）；
- within-stratum 组内 Lift 0.9929（ALL15）/ **0.9767**（NEW11）；
- NEW11 paired bootstrap ΔAUC CI **[−0.0494, −0.0104] 完全在 0 以下**。

    Native-direction OB tradability: CLOSED.
    换用更有理论依据的 OB 自身结构边界后结果未改善反而更差，
    按路线图彻底结束 native direction，不再给第三次机会。

下一步研究问题改为 **OB Event Value Test（OB vs matched non-OB）**：
不再问"哪个 OB 值得做多/做空"，而问"OB 出现后市场是否比普通时刻
更容易发动行情"。该研究本轮未启动。
