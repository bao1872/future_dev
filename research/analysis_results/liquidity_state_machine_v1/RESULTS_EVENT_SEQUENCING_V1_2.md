# Event Sequencing Audit v1.2

Base: `1cd96283f2d35fea9a1f93d95db5ef7e50ed5073`

```
P(RECLAIM|PENETRATION)=81.43%  →  PROVISIONAL — ASYMMETRIC EVENT TIMING
```

不再称 "strong state-transition evidence"。本轮修正事件时间语义后重估。

---

## 1. 修正清单

| # | 问题 | 修正 |
|---|---|---|
| 1 | trading_day 用自然日 `strftime` | 改用 canonical `five.iloc[ti]["trading_day"]` |
| 2 | first touch 可跨 rollover | 触碰前遇 `disc` → `ROLL_CENSORED_PRE_INTERACTION` |
| 3 | penetration 由 inferred tick 定义 | 改为严格空间穿透 `high > level` / `low < level`；tick 仅描述 |
| 4 | `resolution_time` = bar_start 冒充可知时间 | 分离 `penetration_bar_start` / `penetration_available_time` |
| 5 | reclaim 与 acceptance 非对称竞争 | Stage 1 同 bar 分类；Stage 2 仅 CLOSE_BEYOND 进入对称竞争 |
| 6 | post-reclaim 用 `>=` bar start | 严格 `available_time > reclaim_available_time` |

### trading_day 修正的实质影响

```
calendar_date != canonical trading_day : 30,104 / 77,174 = 39.01%
```

示例（夜盘）：`interaction_time = 2025-01-02 21:00` → `trading_day = 2025-01-03`。
因此 v1.1 的 bootstrap CI 与 F1–F4 **都不是 trading-day based**，已全部重跑。

### Pre-interaction roll censor

被剔除 **82** 个 level（占 81,306 的 **0.10%**）。影响很小，但语义已修正。

---

## 2. Stage 1 — penetration bar 自身（对称、无参数）

PENETRATION = **64,098**

| | n | % of penetration |
|---|---:|---:|
| **SAME_BAR_RECLAIM** | 23,169 | **36.15%** |
| **CLOSE_BEYOND** | 36,193 | **56.47%** |
| CLOSE_AT_LEVEL | 4,736 | 7.39% |

**关键解释量**：

```
old RECLAIM = 52,221
same-bar 占 old reclaim = 44.37%
```

即 **v1.1 的 81.43% 中，有 44% 来自穿透当根 bar 自己收回来的长影线**；
其余 56% 是真正收盘站到 level 之外（CLOSE_BEYOND）后的后续行为。

## 3. Stage 2 — 真正 structural competing risks

只有 CLOSE_BEYOND（36,193）进入，从 `penetration_available_time` 严格之后开始：

| | n | % of CLOSE_BEYOND |
|---|---:|---:|
| **LATER_RECLAIM** | 25,172 | **69.55%** |
| **STRUCTURAL_ACCEPTANCE** | 10,988 | **30.36%** |
| ROLL_CENSORED | 27 | 0.07% |
| END_OF_DATA_CENSORED | 6 | 0.02% |

`bars_to_stage2`：p25 **2** / median **8** / p75 **23** / p90 **49**（未设 horizon）。

→ **修正后 reclaim 仍以约 70 : 30 明显占优。**

---

## 4. HTF 检验（canonical trading_day block bootstrap，500 次）

### H1 — immediate wick rejection

| | n | P(SAME_BAR_RECLAIM\|pen) |
|---|---:|---:|
| AGAINST 1h | 29,536 | 0.3688 |
| WITH 1h | 30,705 | 0.3582 |

pp **+1.07**，RR 1.0298，CI **[−0.28pp, +2.45pp]** → **含 0**

### H2 — held penetration subsequently fails

| | n | P(LATER_RECLAIM\|CLOSE_BEYOND) |
|---|---:|---:|
| AGAINST 1h | 16,524 | 0.7040 |
| WITH 1h | 17,451 | 0.6890 |

pp **+1.50**，RR 1.0218，CI **[−0.55pp, +3.72pp]** → **含 0**

→ **逆 1h 趋势的流动性穿透，在两个阶段都没有显著提高回收概率。**

### 4h environment × 1h

| env4h vs 1h | n_pen | sbr against | sbr with | n_cb | later_reclaim against | later_reclaim with |
|---|---:|---:|---:|---:|---:|---:|
| ALIGNED | 25,965 | 0.3692 | 0.3654 | 14,769 | 0.7141 | 0.6881 |
| CONFLICT | 38,133 | 0.3685 | 0.3529 | 21,424 | 0.6960 | 0.6896 |

ALIGNED 组的 against−with 差（+2.60pp）大于 CONFLICT 组（+0.64pp），
但两组 CI 均未单独给出且幅度仍小；**4h environment 未提供可靠增量**。

### Post-reclaim（严格时间顺序）

仅对 SAME_BAR_RECLAIM 计算（LATER_RECLAIM 未设 post 段，见限制）。

| | n |
|---|---:|
| REJECTION_FAILED_REACCEPTED | 9,394 |
| REVERSAL_MSS_CONFIRMED | 8,126 |
| OTHER | 5,642 |
| NO_LATER_EVENT | 7 |

AGAINST 0.3508 vs WITH 0.3532，pp **−0.24**，CI **[−2.69pp, +2.40pp]** → 含 0。

**不使用 50/50 作为零假设**（BOS 与 CHoCH detector 的无条件 base hazard 不同）。
只报告 AGAINST vs WITH 的条件差：**无差异**。

---

## 5. 稳定性

### by symbol（later_reclaim \| CLOSE_BEYOND）

**15/15 > 0.5**，macro median **0.7116**，范围 0.6301（SC）– 0.7522（I）。

### by fold（canonical trading_day）

| fold | n_pen | p_same_bar | n_cb | p_later_reclaim |
|---|---:|---:|---:|---:|
| F1 | 6,558 | 0.3599 | 3,753 | 0.6944 |
| F2 | 6,647 | 0.3841 | 3,768 | 0.7123 |
| F3 | 6,630 | 0.3774 | 3,698 | 0.6842 |
| F4 | 7,761 | 0.3788 | 4,320 | 0.7081 |

四折 0.684–0.712，**稳定**。

### cluster dedup

| | n_pen | p_same_bar | n_cb | p_later_reclaim |
|---|---:|---:|---:|---:|
| CLUSTER_DEDUP | 41,582 | 0.3630 | 23,222 | **0.7012** |

与主结果（0.6955）一致 → **保留**。

---

## 6. 回答 9 个问题

1. **原 81.43% 中多少是 same-bar？** **44.37%**（23,169 / 52,221）。
2. **收盘仍站 level 外的比例？** **56.47%**（CLOSE_BEYOND 36,193；另有 7.39% CLOSE_AT_LEVEL）。
3. **CLOSE_BEYOND 后 later reclaim vs BOS acceptance？** **69.55% vs 30.36%**。
4. **修正 canonical trading_day 后 CI / F1–F4 是否变化？** 点估计基本不变
   （later_reclaim 69.55%），但 **39.01% 的样本此前被归错交易日**，
   CI 与 fold 已按正确口径重跑；结论不变。
5. **多少 level 因 pre-interaction roll 被剔除？** **82（0.10%）**。
6. **AGAINST 1h 是否提高？** 否 —— same-bar **+1.07pp（CI 含 0）**；
   later reclaim **+1.50pp（CI 含 0）**。
7. **4h alignment 是否仍无增量？** **是**（无可靠增量）。
8. **严格时间顺序后 reversal-MSS 结果？** AGAINST vs WITH **pp −0.24，CI 含 0**，
   且不再用 50/50 解释 BOS/CHoCH 比例。
9. **by symbol / fold / cluster dedup 是否保留？** **全部保留**
   （15/15 品种 > 0.5；四折 0.684–0.712；cluster dedup 0.7012）。

---

## 7. Decision rule 判定

Case A 需要**同时**满足：
- old reclaim **mostly** SAME_BAR_RECLAIM → **否**（44.37%，非多数）；
- P(LATER_RECLAIM\|CLOSE_BEYOND) 不再明显占优 → **否**（69.55% vs 30.36%，
  15/15 品种、四折稳定、cluster dedup 保留）。

### → **Case B**

```
structural reclaim phenomenon survives event-timing correction
```

**v1.1 的 81% 中约 44% 确实是同 bar 长影线现象，但剩下 56% 的
CLOSE_BEYOND 之后仍然有约 70% 会 later reclaim，只有约 30% 形成结构性 BOS
接受。该现象在剔除同 bar 偏差、修正 canonical trading_day、修正 roll censor、
修正 tick 真值判定之后仍然成立，且跨 15/15 品种、跨四折、cluster dedup 后保留。**

## 8. 修正后的三层命题状态

| 命题 | 当前判断 |
|---|---|
| HTF"逆趋势扫流动性后恢复趋势" | **仍不支持**（两阶段 CI 均含 0；post-reclaim 无差异） |
| 穿透 liquidity 后经常收回 | **在 CLOSE_BEYOND 条件下仍成立**（69.6% vs 30.4%），已通过事件时序修正 |
| 该现象是否 liquidity 特有 | **完全未检验** ← 下一轮 |

## 9. 下一步（本轮不做）

按 Case B 规定：**STOP**。下一轮由 reviewer 设计

```
MATCHED NON-LIQUIDITY PLACEBO
```

回答：这种 reclaim 行为是 liquidity level 特有，
还是任意水平价位被穿透后都如此？

## 10. 已知限制

1. post-reclaim 仅对 SAME_BAR_RECLAIM 计算；LATER_RECLAIM 未设 post 段
   （本轮口径下 later reclaim 自身即为 Stage-2 终点）。
2. `CLOSE_AT_LEVEL`（4,736，7.39%）单独报告，未强行归类。
3. tick 仍由数据推断，仅用于 `penetration_depth_inferred_ticks` 描述，
   **不再决定事件真值**。
4. 4h 仍为 `CLOCK_4H_ENVIRONMENT`，不是 canonical structure。
5. `dsa_direction_5m/15m/1h` 仍未计算（同 v1.1 限制）。
6. 未做多重检验校正。
