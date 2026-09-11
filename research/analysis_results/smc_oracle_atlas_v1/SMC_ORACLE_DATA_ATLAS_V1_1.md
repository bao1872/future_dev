# SMC 市场状态—流动性场—路径 Oracle Atlas v1.1 —— 数据语义审计与修复

Base: `3b7d1be9ea26d802ba209d1e81f42f135bf5e56d`

本轮**不训练模型、不筛选状态、不做 PnL**。目标是确认 v1.0 的
事实层是否可信。结论：**v1.0 的核心画像大部分作废**，见第二节。

---

## 一、修复清单

| # | 问题 | 修复 |
|---|---|---|
| 1 | liquidity field 与 target 使用"历史曾出现"而非"决策时仍 active" | 改为 `available AND NOT consumed_before` |
| 2 | 生命周期与 Oracle 路径未做换月/断点 censor | 生命周期搜索上界 = 首个 discontinuity；Oracle 路径终点 = min(roll, data end) |
| 3 | `decision_time` 存的是 bar **开始**时间 | 改为 bar **结束**（+5min） |
| 4 | `first_1R/2R/3R` 未受 stop 约束 | 改为相对 stop first passage 五态判定 |
| 5 | `MAXB=500` 作为主 Oracle horizon | 取消；500 仅用于审计 |
| 6 | `20ATR / 400 targets` 主结果截断 | 取消；仅审计 |
| 7 | target 只剩裸价格数组 | 升级为**价格簇**，保留 identity/scope/type |
| 8 | Oracle status 未区分 censor | 显式 6 类 |
| 9 | "重复接触更容易 TOUCH_ONLY"用组成比例 | 改为**条件风险率 hazard** |
| 10 | 82.5% 同价重叠未审计是否重复生成 | 增加 duplicate identity 审计 |
| 11 | 只有 RR dominance 一套方向标签 | 增加 time-aware Pareto dominance |

---

## 二、回答 13 个问题

### 1. v1.0 contact 中有多少跨 roll 污染？

| 项 | 值 |
|---|---:|
| v1.1 被 roll censor 的 liquidity | **3,806 / 80,289（4.74%）** |
| v1.1 被 roll censor 的 Oracle 行 | **60,102 / 1,356,600（4.43%）** |
| contact 总数变化 | 96,959 → **96,899**（−60） |

contact 数量变化很小，因为多数 liquidity 在首次 roll 之前就已被消费。
但**Oracle 路径层面有 4.43% 的行在 v1.0 中可能跨 roll 计算** ——
这些行的 target hit / stop hit / RR 不可信。

### 2. 修复后 identity / contact 数

| | v1.0 | **v1.1** |
|---|---:|---:|
| liquidity identity | 80,289 | **80,289** |
| contact | 96,959 | **96,899** |
| Oracle 行 | 1,357,426 | **1,356,600** |

### 3. v1.0 vs v1.1 active density（±0.5R 平均个数）

| scope | v1.0（历史可见） | **v1.1（active）** | 比值 |
|---|---:|---:|---:|
| 5m | 4.3619 | **0.4609** | 0.106 |
| 15m | 1.3043 | **0.1794** | 0.138 |
| 1h | 0.3626 | **0.0611** | 0.169 |
| CONTIG_SESSION | 17.0686 | **1.2481** | 0.073 |
| TRADING_DAY | 4.3249 | **0.4721** | 0.109 |
| TRADING_WEEK | 0.8930 | **0.1261** | 0.141 |

**v1.0 高估 6–14 倍。** 每个 contact 决策时点：

| | 平均可见 liquidity 数 |
|---|---:|
| 历史可见（v1.0 口径） | **2,678.76** |
| active（v1.1 口径） | **198.92** |

**v1.0 把约 93% 的已消费历史"尸体"算进了 liquidity field。**
→ **v1.0 density 画像作废。**

### 4. 82.5% overlap 修复后是多少？疑似重复多少？

| 指标 | v1.0 | **v1.1** |
|---|---:|---:|
| multi-identity（同价 ≥2）占比 | 82.54% | **56.93%** |
| **完全重复 identity 组数** | — | **0（多出 0 行）** |

**没有任何完全重复 identity**（同 symbol/price/side/type/scope/
available_time）。所以重叠不是生成链重复造成的。

- v1.0 的 82.5% 含大量已消费历史 level → **作废**
- v1.1 的 **56.93%** 是当前 active 口径下的真实多周期重合

### 5. 500-bar 截断影响多少 Oracle path？

| 指标 | 值 |
|---|---:|
| `path_len > 500` 的 Oracle 行占比 | **98.38%** |
| **第 500 根仍未 stop 且路径未结束（v1.0 被截断）** | **10.59%** |
| median path_len | 15,209 |
| p95 path_len | 33,773 |

**v1.0 有 10.59% 的 Oracle 路径被 500 bar 人为截断**，这些行的
best_R / 支配标签不完整。→ v1.0 Oracle 数值作废。

### 6. 20ATR / 400 target 截断影响多少？

| 指标 | 值 |
|---|---:|
| 20ATR 距离截断会丢 target 的 contact 占比 | **80.09%** |
| 400 target 数量截断生效的 contact 占比 | **0.00%** |

- **20ATR 截断影响 80% 的 contact** —— v1.0 系统性丢掉了远距离
  target，因此低估了 best_R。
- 400 数量截断**从未生效**（active target 数远低于 400），无害。

### 7. 修复后 ATR-risk / RR 前沿（LONG）

| risk_ATR | mean R | median | p75 | p90 | **归零率** | 止损率 | bars 中位 | bars p90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 4.517 | 0.000 | 0.000 | 4.706 | **79.6%** | 97.1% | 2 | 179 |
| 0.50 | 4.149 | 0.000 | 1.085 | 7.273 | 65.0% | 96.1% | 3 | 235 |
| 0.75 | 4.000 | 0.000 | 1.449 | 7.974 | 55.0% | 95.3% | 4 | 289 |
| 1.00 | 3.789 | 0.109 | 1.667 | 8.148 | 48.8% | 94.5% | 6 | 375 |
| 1.50 | 3.525 | 0.278 | 2.273 | 8.485 | 40.3% | 92.7% | 12 | 546 |
| 2.00 | 3.299 | 0.360 | 2.500 | 8.447 | 34.8% | 91.3% | 20 | 690 |
| 3.00 | 2.944 | 0.517 | 2.628 | 7.976 | 28.6% | 88.5% | 42 | 1001 |

SHORT：0.25→3.231 / 1.00→2.680 / 3.00→2.038（形态一致）。

**与 v1.0 的差异：mean R 全面上升**（取消 20ATR 截断后，远距离
target 被计入），但**归零率也全面上升**（active target 更少，
很多方向找不到可到达目标）。

定性结论不变：**小止损 RR 期望高但归零率高；大止损反之。**
中位 R 仍随 risk 单调上升（0.000 → 0.517），未在 1.0 ATR 走平
—— 这点**与 v1.0 不同**（v1.0 在 1.0 后走平），原因是 v1.1
路径更长且无 20ATR 截断。

### 8. 修复后 censor 比例

| status | LONG | SHORT | 合计占比 |
|---|---:|---:|---:|
| TARGET_REACHED_THEN_STOPPED | 306,843 | 324,772 | 46.55% |
| STOPPED | 295,336 | 307,288 | 44.42% |
| **AMBIGUOUS_INTRABAR_ORDER** | 32,942 | 33,021 | **4.86%** |
| TARGET_REACHED_CENSORED | 29,621 | 7,325 | 2.72% |
| NO_ACTIVE_TARGET | 13,405 | 5,642 | 1.40% |
| CENSORED_NO_TARGET | 153 | 252 | 0.03% |

路径 censor：`DATA_END_CENSORED` 1,296,498（95.57%）/
`ROLL_CENSORED` 60,102（4.43%）。

**4.86% 的路径存在同 bar 同时触发 stop 与 target**，已按合同
同时给出保守/乐观两档，未自行规定先后。

### 9. RR dominance

| rr_direction | n | share |
|---|---:|---:|
| LONG_RR_DOMINATES | 33,255 | **34.32%** |
| SHORT_RR_DOMINATES | 31,634 | **32.65%** |
| MIXED_RR | 29,013 | **29.94%** |
| NO_CLEAR_RR | 2,998 | 3.09% |

**与 v1.0 差异巨大**：v1.0 是 MIXED 51.93% / LONG 24.12% /
SHORT 23.24%。v1.1 的 MIXED 降到 29.94%。
→ **v1.0 的 Oracle direction 比例作废。**

### 10. Time-aware Pareto dominance

| pareto_direction | n | share |
|---|---:|---:|
| MIXED_PARETO | 43,578 | **44.97%** |
| LONG_PARETO_DOMINATES | 25,614 | 26.43% |
| SHORT_PARETO_DOMINATES | 24,710 | 25.50% |
| NO_PARETO | 2,998 | 3.09% |

三维度（RR × holding bars × ATR risk）帕累托下 MIXED 回升到 44.97%
—— 因为加入时间维度后，更多方案互不支配。两套标签**同时保存**，
本轮不合并为单一标签。

### 11. contact #1–#4 的 penetration hazard

| contact_number | n at risk | **penetration hazard** | touch-only 概率 |
|---|---:|---:|---:|
| 1 | 76,099 | **0.8326** | 0.1674 |
| 2 | 12,415 | **0.6244** | 0.3756 |
| 3 | 4,541 | **0.5461** | 0.4539 |
| 4 | 2,009 | **0.5296** | 0.4704 |

**条件风险率随接触次数单调下降**（0.833 → 0.624 → 0.546 → 0.530）。

这是 hazard 口径，不再是组成比例。它说明：
**在"已经活到第 k 次接触"的条件下，第 k 次真正穿透的概率随 k 下降。**

需要注意：这仍然可能受"能活到第 k 次的 liquidity 本身更受尊重"
的选择效应影响 —— 本轮**只描述，不推因果**。

### 12. 修复前后哪些保留、哪些作废

**保留（v1.0 有效）**
- 80,289 liquidity / ~96,900 contact 的规模框架
- contact 三级数据结构与接触类型分类定义
- 多周期趋势快照框架、ATR 风险网格设计
- Long/Short 双向 Oracle 框架与标签隔离原则
- 定性结论："小止损 RR 期望高但归零率高，大止损反之"

**作废（v1.0 不可引用）**
- liquidity field density（高估 6–14 倍）
- 82.5% 同价重叠（真实 active 口径 56.93%）
- Oracle RR 前沿的**精确数值**
- Oracle direction 比例（MIXED 51.93% → 29.94%）
- target 相关画像
- "重复接触更容易 TOUCH_ONLY"这一**表述方式**
  （现象方向在 hazard 口径下仍成立，但原统计方式无效）

### 13. Atlas 是否已具备下一阶段预测研究资格？

```text
具备，但附有条件。
```

**具备**：三级表结构、active liquidity 定义、roll censor、
路径 censor 状态、hazard 口径、两套方向标签均已就位，
13 项语义测试全部通过。

**条件 / 仍需注意**
1. `TARGET_REACHED_CENSORED`（2.72%）与 `CENSORED_NO_TARGET`
   是右删失路径，建模时必须显式处理，不能当"更好的路径"。
2. `AMBIGUOUS_INTRABAR_ORDER`（4.86%）需按保守/乐观两档分别建模，
   或单独剔除。
3. `NO_ACTIVE_TARGET`（1.40%）表示该方向根本没有 active 目标，
   不是"R=0"。
4. best target 中 **95%+ 是 day/session 级别**，1h 仅 8–13%
   —— 若后续研究 HTF target，需注意样本结构。
5. hazard 的选择效应未消除，不能推因果。
6. 本轮**未做**任何预测模型，也未验证任何状态的可预测性。

---

## 三、best target 画像

| risk_ATR | 含 1h | 含 day/session/week | 簇大小 |
|---:|---:|---:|---:|
| 0.25 | 7.6% | **95.6%** | 2.25 |
| 1.00 | 9.0% | 95.8% | 2.31 |
| 3.00 | 13.3% | 96.4% | 2.53 |

最佳目标平均是 **2.25–2.53 个 identity 组成的价格簇**，
且绝大多数包含 day/session/week 级别 liquidity。

## 四、测试

`research/tests/test_oracle_atlas_v1_1.py`，**13 项全部通过**：
decision_time = bar 结束 · 生命周期 roll censor · find_contacts 尊重
discontinuity 上界 · Oracle 路径 censor · active_mask 正确排除已消费 ·
active 数 < 历史可见数 · first_mR 状态合法且与 stop 一致 ·
Oracle status 显式 6 类 · 无 500/400 主截断 · 无完全重复 identity ·
Oracle 字段未进入事前特征表 · hazard 已计算。

## 五、已知限制

1. `TARGET_REACHED_CENSORED` 路径未观察到 stop，best_R 可能随更长
   数据变化。
2. 4h 仍只作 environment direction。
3. hazard 未做协变量调整，存在选择效应。
4. Pareto dominance 用三维（RR × bars × risk）合并跨方向非支配集，
   定义是本轮新增，尚未与其他口径比较。
5. 未做多重检验、未训练任何模型。
