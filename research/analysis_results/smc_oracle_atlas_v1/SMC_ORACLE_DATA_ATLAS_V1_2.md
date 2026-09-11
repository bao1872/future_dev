# SMC Oracle Atlas v1.2 —— 方向标签、不确定性与同 bar 消费语义冻结

Base: `baa76f320ca4154feaea2e52d595fdbc4c867297`

本轮**不训练模型、不做 PnL、不筛 subgroup**。目标是冻结未来所有预测
研究使用的 Oracle 标签语义。

---

## 修复清单

| # | 问题 | 修复 |
|---|---|---|
| 1 | `first_penetration_time < decision_time` 使同 bar 被扫的 liquidity 仍 active | 改为 `<=` |
| 2 | Oracle 只给点估计，censor/ambiguous 被压成一个数 | 改为区间 `lower/upper/is_exact/resolution_class` |
| 3 | `NO_ACTIVE_TARGET` 被当 R=0 | 改为 `lower=upper=NaN` |
| 4 | 方向标签跨 risk 逐档比较，未处理 censor | 改为**同 risk** 成对比较，不可证明即 `UNRESOLVED_CENSOR` |
| 5 | Pareto 在 LONG/SHORT **内部**分别求，未真正跨方向比较 | 废弃为正式标签 |
| 6 | `risk_ATR` 被当作"越小越优"的 Pareto 目标 | 改为**条件轴**，只在同 risk 内比较 |
| 7 | 只有单一方向标签 | 增加跨 risk 稳定性 + 时间稳定性 |
| 8 | target 只保留 `has_day` 布尔 | 保留完整 scope 组成（min/max/count/各周期） |

---

## 一、same-bar consume 修复后的变化

### active density（±0.5R 平均个数）

| scope | v1.1 | **v1.2** | 比值 |
|---|---:|---:|---:|
| 5m | 0.4609 | **0.1360** | 0.295 |
| 15m | 0.1794 | **0.0451** | 0.251 |
| 1h | 0.0611 | **0.0152** | 0.249 |
| CONTIG_SESSION | 1.2481 | **0.4396** | 0.352 |
| TRADING_DAY | 0.4721 | **0.1460** | 0.309 |
| TRADING_WEEK | 0.1261 | **0.0343** | 0.272 |

### same-price multi-identity

| | v1.1 | **v1.2** |
|---|---:|---:|
| multi-identity（同价 ≥2） | 56.93% | **13.62%** |

**为什么降这么多？** 因为同价的多个 identity 会在同一根 bar 被同时
触及并同时消费。修复后它们在收盘时全部退出 active field。

**语义后果（重要）**：v1.2 的 `same_price_identity_count` 衡量的是
"**当前价位还剩多少未被消费的 identity**"，而不是"这个价位原本叠加了
多少"。对 PENETRATE 类 contact，叠加度在决策时点已被消耗；
对 TOUCH_ONLY 则保留。

因此 v1.1 的 56.93% 描述的是"未被同 bar 消费的叠加"，
v1.2 的 13.62% 是严格 active 口径。两个都保留在 state 表中
（`same_price_identity_count` 与 `same_price_identity_count_v11`），
研究者可按问题选择。

## 二、Oracle 行分辨率构成

| resolution_class | n | share | lower | upper |
|---|---:|---:|---|---|
| **EXACT_RESOLVED** | 1,260,826 | **92.94%** | = upper | 确定 |
| CENSORED_LOWER_BOUND | 37,060 | 2.73% | 已实现下界 | **NaN（未知）** |
| AMBIGUOUS_INTERVAL | 33,934 | 2.50% | conservative | optimistic |
| NO_ACTIVE_TARGET | 24,780 | 1.83% | **NaN** | **NaN** |

92.94% 的 Oracle 行是完整解析的确定值；7.06% 存在不同程度不确定性，
已按合同显式保存，**不再压成点估计**。

## 三、每个 ATR risk 的 LONG/SHORT 方向（同 risk 成对比较）

| risk_ATR | LONG_DOM | SHORT_DOM | **TRADEOFF** | UNRESOLVED | NO_TARGET | **可解析率** |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 13,774 | 13,490 | **64,965** | 1,131 | 3,540 | **95.18%** |
| 0.50 | 23,243 | 23,571 | 44,398 | 2,148 | 3,540 | 94.13% |
| 0.75 | 28,984 | 29,406 | 31,783 | 3,187 | 3,540 | 93.06% |
| 1.00 | 31,847 | 33,161 | 24,225 | 4,127 | 3,540 | 92.09% |
| 1.50 | 35,253 | 37,452 | 14,584 | 6,071 | 3,540 | 90.08% |
| 2.00 | 37,206 | 38,908 | 9,409 | 7,837 | 3,540 | 88.26% |
| 3.00 | **37,867** | **40,104** | **4,311** | **11,078** | 3,540 | 84.91% |

**TRADEOFF 随 risk 单调减少**（64,965 → 4,311），可解析的支配随之增多。
原因是小止损下大量方向都"不到达任何 target"（R=0 对 R=0），
无法区分；大止损下更多 target 被到达，差异显现。

## 四、哪个 ATR 区间方向最容易被解析（只描述）

**小止损最易解析**（0.25 ATR 95.18%），但解析结果中 67% 是 TRADEOFF
（双方都没到 target）。**大止损解析率低**（3.00 ATR 84.91%，
因 censor 增多），但一旦解析，**支配结论更多**（76,971 vs 小止损 27,264）。

即：**"能不能判定方向"与"判定出有意义方向"是两件事。**
本报告不宣布最佳参数。

## 五、跨 7 档的方向稳定性

| direction_stability | n | share |
|---|---:|---:|
| **NO_DIRECTION** | 51,836 | **53.49%** |
| **RISK_DEPENDENT** | 18,495 | **19.09%** |
| ROBUST_SHORT | 9,477 | 9.78% |
| ROBUST_LONG | 9,425 | 9.73% |
| UNRESOLVED | 7,667 | 7.91% |

- **ROBUST_LONG + ROBUST_SHORT = 19.51%**：约 1/5 的 contact 在全部
  可解析风险档上方向一致。
- **RISK_DEPENDENT 19.09%**：小止损适合一边、大止损适合另一边 ——
  这正是你预判的"MIXED 不等于没机会"。
- 所有 contact 均输出连续计数 `n_long_dom / n_short_dom / n_tradeoff /
  n_unresolved / n_no_target`，后续研究可不做分类直接用计数。

## 六、加入时间维度后

| risk_ATR | LONG_TIME | SHORT_TIME | **TRADEOFF** | NO_MOVE_BOTH |
|---:|---:|---:|---:|---:|
| 0.25 | 13,758 | 13,500 | 66 | 64,905 |
| 1.00 | 28,881 | 29,746 | 6,771 | 23,835 |
| 3.00 | 25,902 | 27,508 | **24,737** | 4,135 |

加入 bars 后，TRADEOFF 随 risk 显著上升（66 → 24,737）：
**大止损下"方向更明确但更慢"与"方向弱但更快"经常同时出现**，
不能强行判优劣。这与 RR 口径（TRADEOFF 随 risk 下降）**相反**，
说明 RR 与时间确实是两个不同的轴。

## 七、v1.1 RR dominance → v1.2 stability 改判情况

| v1.1 ↓ / v1.2 → | NO_DIRECTION | RISK_DEPENDENT | ROBUST_LONG | ROBUST_SHORT | UNRESOLVED |
|---|---:|---:|---:|---:|---:|
| LONG_RR_DOMINATES (33,255) | **20,258** | 107 | **9,129** | 0 | 3,761 |
| MIXED_RR (29,013) | 9,016 | **18,232** | 296 | 684 | 785 |
| NO_CLEAR_RR (2,998) | 2,084 | 0 | 0 | 0 | 914 |
| SHORT_RR_DOMINATES (31,634) | **20,478** | 156 | 0 | **8,793** | 2,207 |

**关键改判：**

1. **v1.1 的 LONG/SHORT_RR_DOMINATES 有 60.9% 被改判为 NO_DIRECTION。**
   原因：v1.1 逐档比较 `conservative_best_R`，当双方大量档位都是 0
   （都没到达 target）时，只要有一档略高就判支配。v1.2 要求
   `lower > upper` 才判支配，0 vs 0 只能是 TRADEOFF。
2. **v1.1 的 MIXED_RR 有 62.8%（18,232）被改判为 RISK_DEPENDENT。**
   这完全印证了你的判断：这些不是"没机会"，而是**小止损与大止损
   指向相反方向**，本身就是市场状态信息。

→ **v1.1 的 `rr_direction` 比例作废**，v1.2 的 stability 才是正式标签。

## 八、v1.1 Pareto direction 正式废弃

```text
LONG_PARETO_DOMINATES / SHORT_PARETO_DOMINATES / MIXED_PARETO
= PROVISIONAL / SEMANTICALLY_REPLACED
```

废弃理由：
1. 实现是在 LONG、SHORT **各自内部**求前沿，再按"前沿是否含两个方向"
   判定 —— 这并没有比较"LONG 前沿是否支配 SHORT 前沿"。只要两边各有
   一个可达 target，就必然落入 MIXED，因此 44.97% 的 MIXED_PARETO
   **不能解释为"两方向互不支配"**。
2. `risk_ATR` 被当作"越小越优"的目标，隐含了固定手数的经济假设。
   在固定账户风险、按止损距离调整仓位的口径下该假设不成立。

v1.2 改用：同 risk 的 RR 比较 + 同 risk 的 RR×时间比较 + 跨 risk 稳定性。

## 九、best target cluster 的 scope 组成

| scope 组合 | n | share |
|---|---:|---:|
| CONTIG_SESSION（单独） | 256,370 | **47.04%** |
| CONTIG_SESSION\|TRADING_DAY | 60,928 | 11.18% |
| 15m\|1h\|5m\|CONTIG_SESSION\|TRADING_DAY\|TRADING_WEEK | 51,298 | 9.41% |
| 5m\|CONTIG_SESSION\|TRADING_DAY | 44,635 | 8.19% |
| 5m\|CONTIG_SESSION | 27,638 | 5.07% |
| 15m\|5m\|CONTIG_SESSION\|TRADING_DAY | 20,869 | 3.83% |
| 15m\|5m\|CONTIG_SESSION\|TRADING_DAY\|TRADING_WEEK | 20,390 | 3.74% |
| 5m（单独） | 15,971 | 2.93% |
| CONTIG_SESSION\|TRADING_DAY\|TRADING_WEEK | 9,193 | 1.69% |
| 5m\|CONTIG_SESSION\|TRADING_DAY\|TRADING_WEEK | 8,746 | 1.60% |
| TRADING_DAY（单独） | 7,311 | 1.34% |
| 15m\|CONTIG_SESSION | 3,705 | 0.68% |

**"95%+ 含 day/session/week" 仍然成立（96.63%）**，但你的提醒是对的：
这个数字主要说明 CONTIG_SESSION 类 liquidity 数量多、存活久。
按组合看，**近半数最佳目标是"纯 CONTIG_SESSION"单一身份**，
只有 9.41% 是六周期全叠加。两者显然不是同一种目标，后续研究
必须按 `best_target_scopes` 细分，不能只用 `has_day`。

## 十、contact hazard（v1.1 口径，未改）

| contact_number | n at risk | penetration hazard |
|---|---:|---:|
| 1 | 76,099 | 0.8326 |
| 2 | 12,415 | 0.6244 |
| 3 | 4,541 | 0.5461 |
| 4 | 2,009 | 0.5296 |

只描述，不推因果；选择效应未消除。

## 十一、Atlas 是否可以正式冻结

```text
可以冻结为后续研究底座。
```

**标签分层已明确：**

| 层 | 内容 | 可否进特征 |
|---|---|:--:|
| 事实状态表 | liquidity / contact / trend / field / OB | ✅ 事前可见 |
| 路径事实 | target reach / stop / bars / censor | ❌ 事后 |
| Oracle lower-bound | `best_R_lower`（允许 censor） | ❌ 事后 |
| Oracle exact label | 仅 `EXACT_RESOLVED` | ❌ 事后 |
| Direction stability | ROBUST_* / RISK_DEPENDENT / … | ❌ **绝不能进事前特征** |

**后续建模时的强制约束：**
1. 只能用 `EXACT_RESOLVED`（92.94%）做精确标签；
2. `CENSORED_LOWER_BOUND`（2.73%）只能当下界，需 survival/censored 处理；
3. `AMBIGUOUS_INTERVAL`（2.50%）按区间建模或分保守/乐观两档；
4. `NO_ACTIVE_TARGET`（1.83%）应单独剔除，不是 R=0。

## 十二、测试

`research/tests/test_oracle_atlas_v1_2.py`，**16 项全部通过**：
同 bar consume 排除 · v1.2 ≤ v1.1 同价计数 · censor upper 为 NaN ·
NO_ACTIVE_TARGET 为 NaN 且不参与比较 · ambiguous 区间合法 ·
exact long/short 判定 · 区间重叠不强判 · censor 无法证明即 unresolved ·
stability 五类规则 · 计数守恒 · 时间维度只比同 risk 且 TRADEOFF 不强判 ·
Oracle 未进状态表 · roll/data-end censor 保持 · target cluster 字段完整。

## 十三、已知限制

1. same-bar consume 使 PENETRATE 类 contact 的叠加度归零，
   研究"价位重要性"时应改用 v1.1 口径字段。
2. `NO_DIRECTION` 中大量是"双方都没到达 target"，与"双方等效"
   语义不同，后续需细分。
3. `RISK_DEPENDENT` 定义依赖"至少一档 LONG + 至少一档 SHORT"，
   未对档位间距加权。
4. 时间稳定性标签为次级，未与主 RR 标签合并为单一真值。
5. 未训练任何模型，未验证任何状态的可预测性。
