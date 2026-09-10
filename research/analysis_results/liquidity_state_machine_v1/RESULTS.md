# SMC Multi-Timeframe Liquidity State Machine v1.1

Base: `cd1a339`。合同修订见
`research/liquidity_state_machine/REVIEWER_DECISION_V1_1.md`。

> ⚠️ **本文件的 `P(RECLAIM|PENETRATION)=81.43%` 已标记为
> `PROVISIONAL — ASYMMETRIC EVENT TIMING`。**
> v1.1 中 reclaim 可在 penetration bar 自身成立，而 acceptance 必须等待
> penetration bar 之后的新 BOS，两者不是对称 competing risks。
> 且 trading_day 曾误用自然日（39.01% 样本归错）。
> **已由 `RESULTS_EVENT_SEQUENCING_V1_2.md` 修正重估，请以该文件为准。**
> 修正后：CLOSE_BEYOND 条件下 later reclaim 69.55% vs 结构性接受 30.36%。

**v1.1 分类（已被 v1.2 取代）：B. PARTIAL / HETEROGENEOUS STATE TRANSITION**

```
NO ML / NO model / NO PnL / NO Sharpe / NO entry-stop-target optimization
DELIVERY_STAGE = NOT_STARTED
Displacement = NOT TESTED — NO CANONICAL SEMANTICS
```

---

## 1. GATES

### GATE A — 5m/15m/1h structural trend causality：**PASS**

黄金标准**截断测试**：在 `bars[:i+1]` 上重建 SMC，bar i 的
`swing_bias` / `internal_bias` 必须与全序列结果一致。

| | 检查数 | 不一致 |
|---|---:|---:|
| 6 symbols × {5m,15m,1h} | **718** | **0** |

→ `swing_bias` 是 per-bar `state_timeline` 值，**非 retrospective backfill**。

### GATE B — 4h environment direction causality：**PASS**

- 只用 `aggregate_4h_from_1h` 输出，命名 `env_4h` / `CLOCK_4H_ENVIRONMENT`，
  **未生成任何 4h swing / BOS / MSS / liquidity / OB**。
- `env_direction_4h` 由 canonical `compute_dsa_canonical(env4)` 原样计算
  （timeframe-agnostic，未修改算法/窗口/阈值）：

  ```
  canonical_algorithm = yes
  canonical_4h_state  = no
  research_derived_environment = yes
  ```

- `env_available_time` = bucket 最后一根成分 1h bar 的结束时间；
  `merge_asof(direction="backward")` + 断言
  `env_available_time <= interaction_time` 全部通过。
- **未制造** `trend_struct_4h`。

### Causality gate
`level_available_time <= interaction_time` 与
`state_available_time <= interaction_time` 全部由 `merge_asof(backward)`
加断言保证。confirmed pivot 使用 `confirmed_time + 一个 bar 周期`
（保守：确认 bar 收盘后才可用），**未把 pivot_time 当 available_time**。

---

## 2. Liquidity universe

**81,306** 个 level；source_tf / scope 覆盖
`5m / 15m / 1h`（canonical structural）
与 `TRADING_DAY / CONTIG_SESSION / TRADING_WEEK`
（`RESEARCH_DERIVED_TIME_LIQUIDITY`）。

类型：`CONFIRMED_SWING_HIGH/LOW`、`CANONICAL_EQH/EQL`、
`PREV_TRADING_DAY_HIGH/LOW`、`PREV_CONTIG_SESSION_HIGH/LOW`、
`PREV_TRADING_WEEK_HIGH/LOW`。
同价位多个 identity **未 collapse**，另建 `interaction_cluster_id`
用于 sensitivity。

## 3. Primary funnel（FIRST VALID INTERACTION ONLY）

| stage | raw n | % of prev |
|---|---:|---:|
| LIQUIDITY LEVELS | 81,306 | — |
| FIRST INTERACTIONS (touch) | 77,232 | 94.99% |
| **PENETRATION** | **64,126** | 83.03% |
| **RECLAIM** | **52,221** | **81.43%** |
| **ACCEPTANCE** | **11,872** | **18.51%** |
| post-reclaim: REVERSAL_MSS | 24,377 | 46.68% of reclaim |
| post-reclaim: RE-ACCEPTANCE | 27,806 | 53.24% of reclaim |

（另有 ROLL_CENSORED 27、END_OF_DATA_CENSORED 6；
post-reclaim AMBIGUOUS_SAME_TIMESTAMP 1、END_OR_ROLL_CENSORED 19。）

---

## 4. 核心统计

### A. 第一分叉 —— **强结构**

```
P(RECLAIM    | PENETRATION) = 0.8143
P(ACCEPTANCE | PENETRATION) = 0.1851
```

穿透后**回收显著压倒结构性接受**（约 4.4 : 1）。
这是本轮最稳健的发现。

### B. Sweep 方向 × 1h structural trend —— **不支持**

| | n | P(reclaim\|pen) |
|---|---:|---:|
| AGAINST 1h | 29,528 | 0.8222 |
| WITH 1h | 30,750 | 0.8089 |

pp 差 **+1.33pp**，relative risk 1.0164，
bootstrap 95% CI **[−0.09pp, +2.67pp]**、RR CI [0.9988, 1.0334]
→ **CI 含 0**。

### C. Reclaim 后反转确认 —— **不支持**

| | n | P(REVERSAL_MSS_5M\|reclaim) |
|---|---:|---:|
| AGAINST 1h | 24,278 | 0.4738 |
| WITH 1h | 24,874 | 0.4560 |

pp 差 **+1.79pp**，CI **[−0.56pp, +3.95pp]** → 含 0。
反向指标（RE-ACCEPTANCE）：against 0.5257 vs with 0.5429，
pp −1.72pp，CI [−3.89pp, +0.66pp] → 含 0。

### D. Trend resumption —— **与假设相反**

在 **AGAINST 1h + RECLAIM**（理论最有利的组合）下：

```
REVERSAL_MSS_CONFIRMED        0.4738
REJECTION_FAILED_REACCEPTED   0.5257   <-- 更高
```

即：即便逆 1h 趋势扫流动性并被回收，随后**更常见的是沿穿透方向被重新接受**，
而不是出现恢复 1h 趋势的反转 MSS。这与
"先杀逆趋势流动性、再恢复主趋势"的叙述**方向相反**。

### E. 4h environment × 1h —— **无增强**

| env4h vs 1h | n_pen | P(reclaim\|against) | P(reclaim\|with) | P(rev_mss\|against) | P(rev_mss\|with) |
|---|---:|---:|---:|---:|---:|
| ALIGNED | 26,010 | 0.8278 | 0.8102 | 0.4723 | 0.4491 |
| CONFLICT | 38,116 | 0.8178 | 0.8080 | 0.4750 | 0.4610 |

两组几乎完全重合：4h environment 与 1h 同向**没有**进一步增强任何效应。

---

## 5. 分层

### by liquidity type

| type | n_pen | P(reclaim) | P(accept) | P(rev_mss\|reclaim) |
|---|---:|---:|---:|---:|
| PREV_CONTIG_SESSION_HIGH | 19,655 | 0.8041 | 0.1952 | 0.4547 |
| PREV_CONTIG_SESSION_LOW | 18,674 | 0.8308 | 0.1691 | 0.4980 |
| PREV_TRADING_DAY_HIGH | 5,152 | 0.7991 | 0.1995 | 0.4202 |
| PREV_TRADING_DAY_LOW | 4,651 | 0.8194 | 0.1806 | 0.4700 |
| CONFIRMED_SWING_HIGH | 4,418 | 0.7938 | 0.2046 | 0.3949 |
| CONFIRMED_SWING_LOW | 3,836 | 0.8147 | 0.1853 | 0.4698 |
| CANONICAL_EQH | 3,059 | 0.8215 | 0.1785 | 0.4767 |
| CANONICAL_EQL | 2,696 | 0.8457 | 0.1543 | 0.5202 |
| PREV_TRADING_WEEK_HIGH | 1,092 | 0.7628 | 0.2344 | 0.4214 |
| PREV_TRADING_WEEK_LOW | 893 | 0.8007 | 0.1993 | 0.4965 |

各类 reclaim 率集中在 **0.76–0.85**，无某一类显著突出。
（注：`PREV_CONTIG_SESSION_*` 样本量最大，因为 5m 断口分段最细。）

### by liquidity scope

| scope | n_pen | P(reclaim) | P(accept) |
|---|---:|---:|---:|
| CONTIG_SESSION | 38,329 | 0.8171 | 0.1824 |
| 5m | 9,948 | 0.8219 | 0.1777 |
| TRADING_DAY | 9,803 | 0.8087 | 0.1906 |
| 15m | 3,194 | 0.8034 | 0.1960 |
| TRADING_WEEK | 1,985 | 0.7798 | 0.2186 |
| 1h | 867 | 0.7878 | 0.2111 |

**未按 source timeframe 出现明显梯度** —— 4h/1h/15m/5m 流动性
（此处最高只到 1h）行为相似。

### by symbol —— **15/15 全部一致**

P(reclaim) 范围 **0.7544（SC）– 0.8472（SN）**，
macro median **0.8314**。方向完全一致。

### by fold —— **非常稳定**

| fold | n_pen | P(reclaim) | P(accept) | P(rev_mss) |
|---|---:|---:|---:|---:|
| F1 | 6,514 | 0.8129 | 0.1830 | 0.4568 |
| F2 | 6,592 | 0.8260 | 0.1740 | 0.4905 |
| F3 | 6,739 | 0.8098 | 0.1902 | 0.4422 |
| F4 | 7,699 | 0.8245 | 0.1747 | 0.4751 |

### cluster dedup —— **保留**

| | n_pen | P(reclaim) | P(accept) | P(rev_mss) |
|---|---:|---:|---:|---:|
| CLUSTER_DEDUP | 41,588 | 0.8188 | 0.1808 | 0.4789 |

与主结果（0.8143 / 0.1851 / 0.4738）几乎一致 →
**不是同价位多重标签重复计数造成的**。

---

## 6. 明确的研究缺口（不隐藏）

```
The current repository has no canonical displacement semantics.

Therefore V1.1 does NOT test the full
Sweep→Reclaim→Displacement→MSS SMC chain.

It tests: Penetration→Reclaim/Acceptance→MSS.
```

另缺 canonical：FVG/imbalance、range/consolidation boundary
（本轮按 Decision 3 排除，未临时构造）。

---

## 7. 回答 10 个科学问题

1. **穿透后更常 reclaim 还是 acceptance？** **reclaim**（0.814 vs 0.185）。
2. **单独 penetration 是否有方向信息？** **基本没有** ——
   post-reclaim 后 reversal MSS 与 re-acceptance 接近 47/53，
   穿透方向本身几乎不预测后续方向。
3. **逆 1h 的穿透更容易 reclaim？** **否**（+1.33pp，CI 含 0）。
4. **reclaim 后 reversal MSS 是否明显增加？** **否**
   （against vs with 仅 +1.79pp，CI 含 0）。
5. **反转恢复 1h 时概率是否高于 countertrend reversal？** **否，且相反**
   —— against 组 re-acceptance（0.526）高于 reversal MSS（0.474）。
6. **env4h 与 1h 同向时 resumption 是否增强？** **否**（几乎完全重合）。
7. **4h/1h 冲突时发生什么？** 与同向时无实质差异（保留未删除）。
8. **哪类 liquidity 最明显？** 无显著突出者，集中在 0.76–0.85；
   `CANONICAL_EQL` 最高（0.846）、`PREV_TRADING_WEEK_HIGH` 最低（0.763）。
9. **跨 symbol 与 F1–F4 稳定？** **是** —— 15/15 品种一致，
   四折 reclaim 率 0.810–0.826。
10. **cluster dedup 后仍存在？** **是**（0.8188）。

---

## 8. 结论

```
A clear and highly reproducible first-branch asymmetry exists:
liquidity penetration is overwhelmingly reclaimed (~81%) rather than
structurally accepted (~19%), stable across 15/15 symbols and all four
folds, and retained after cluster dedup.

However, the theoretically motivated HTF layer is NOT supported:
against-trend sweeps are not meaningfully more likely to reclaim
(+1.3pp, CI includes 0), and after reclaim the market is slightly MORE
likely to be re-accepted in the penetration direction (52.6%) than to
produce a trend-resuming reversal MSS (47.4%). 4h environment alignment
adds nothing.

=> B. PARTIAL / HETEROGENEOUS STATE TRANSITION
```

**不得**解读为"SMC 被证明"或"主力在猎杀止损"。
只能说：price-action sequence 支持
**"穿透 → 回收"这一层状态转移**，但**不支持**
"逆趋势扫流动性 → 回收 → 恢复主趋势"的完整叙述。

## 9. 已知限制

1. Acceptance 用"穿透方向出现新 BOS"代理（无更明确 canonical 定义）。
2. `env_direction_4h` 是 research-derived environment，
   不是 canonical 4h structure；1h 仍是最高的 canonical 结构周期。
3. `dsa_direction_5m/15m/1h` 本轮**未计算**
   （canonical 函数在 5m 上约 159s/品种，超出本轮预算），
   仅 `env_direction_4h` 使用了 canonical DSA。已如实标记，未用替代品冒充。
4. tick 由数据推断（最小非零价格变动），非交易所 tick 表。
5. 5m OHLC 无法恢复 bar 内顺序；
   本轮 competing-state 未出现大量 AMBIGUOUS（0 例 intrabar、
   1 例 post-reclaim 同时间戳）。
6. 未做多重检验校正。
