# SMC 市场状态—流动性场—路径 Oracle Atlas v1.0

Base: `06d45de39897f73a5243b080d3d5de47e020503c`

本轮**不训练任何预测模型、不做 PnL、不优化交易规则**。
目标是建立未来所有 SMC 研究使用的统一事件数据库。

研究单位不再是"一条 liquidity 一行"，而是三级表：

```text
liquidity_master        : 一个事前形成的 liquidity identity 一行
liquidity_contacts      : liquidity_id × contact_number 一行
liquidity_oracle_labels : 每个 contact 时点的事后 Long/Short Oracle 标签
```

Oracle 标签**可以使用未来**，因为它本来就是事后 Oracle；
但**绝不能回到事前特征表**。

---

## 一、数据规模

| 层 | 数量 |
|---|---:|
| liquidity identity（VALID_AHEAD） | **80,289** |
| 有 ≥1 次接触 | 76,598 |
| 被消费（首次严格穿透） | 75,598 |
| **contact 事件** | **96,959** |
| Oracle 行（contact × 2 方向 × 7 风险） | 1,357,426 |
| 帕累托前沿点 | 240,118 |

**平均每个 liquidity 被接触 1.208 次。**

## 二、一条 liquidity 通常被接触几次

| contact_number | n | 占比 |
|---|---:|---:|
| 1 | 76,147 | 78.5% |
| 2 | 12,425 | 12.8% |
| 3 | 4,541 | 4.7% |
| 4 | 2,010 | 2.1% |
| 5 | 912 | 0.9% |
| 6–10 | 880 | 0.9% |

**绝大多数（78.5%）liquidity 只被接触一次就结束**（要么穿透被消费，
要么再没回来）。有第 2 次及以上接触的共 20,812 个 —— 样本足够做
"第一次 vs 第三次"比较，但第 4 次以后（<2,000）要谨慎。

## 三、接触类型分布

| contact_type | n | 占比 | 其中属于首次接触 |
|---|---:|---:|---:|
| PENETRATE_CLOSE_BEYOND | 30,272 | 31.2% | 81.6% |
| PENETRATE_RECLAIM | 22,816 | 23.5% | 84.0% |
| **TOUCH_ONLY** | **21,361** | **22.0%** | **59.7%** |
| GAP_CROSS | 16,186 | 16.7% | 93.9% |
| PENETRATE_CLOSE_AT | 6,324 | 6.5% | 68.6% |

**值得注意**：TOUCH_ONLY（碰而不穿）有 21,361 次，其中只有 59.7% 是
首次接触 —— 也就是说 **40% 的"碰而不穿"发生在重复接触时**。
而穿透类（reclaim / close beyond）80% 以上是首次接触。

这暗示：**第一次接触更容易直接穿透；已经被碰过的 level 更容易再次
被"碰一下但不穿"**。GLP_CROSS（开盘跳空越过）93.9% 是首次接触，
符合"跳空通常一次性完成"的直觉。

## 四、Oracle 方向分布

| oracle_direction | n | share |
|---|---:|---:|
| **MIXED** | **50,348** | **51.93%** |
| LONG_DOMINATES | 23,382 | 24.12% |
| SHORT_DOMINATES | 22,535 | 23.24% |
| NO_CLEAR_ACTION | 694 | 0.72% |

**超过一半的接触时点，Long 与 Short 不存在支配关系。**
LONG 与 SHORT 占比几乎相同（24.1% vs 23.2%）。

支配判定基于 7 档 ATR 风险下保守 best_R 的逐档比较：
只有在所有档位都不劣、且至少一档严格更优时才判支配。

## 五、ATR 风险—可实现 RR 前沿（本轮最重要的一张表）

### LONG

| risk_ATR | mean R | p25 | median | p75 | p90 | **归零率** | **被止损率** | bars 中位 | bars p90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 3.512 | 0.000 | 0.000 | 2.182 | 8.802 | **68.5%** | 95.9% | 2 | 69 |
| 0.50 | 3.109 | 0.000 | 0.545 | 2.727 | 8.684 | 44.7% | 93.6% | 2 | 74 |
| 0.75 | 2.833 | 0.000 | 0.824 | 2.813 | 8.406 | 29.2% | 91.6% | 3 | 90 |
| **1.00** | **2.575** | 0.166 | **0.840** | 2.806 | 7.662 | **21.1%** | 89.6% | 4 | 109 |
| 1.50 | 2.241 | 0.238 | 0.877 | 2.807 | 6.667 | 13.2% | 85.5% | 8 | 159 |
| 2.00 | 1.989 | 0.250 | 0.896 | 2.727 | 5.778 | 10.0% | 81.9% | 13 | 201 |
| 3.00 | 1.634 | 0.265 | 0.909 | 2.377 | 4.575 | 7.3% | 74.7% | 26 | 268 |

### SHORT

| risk_ATR | mean R | median | p90 | 归零率 | 被止损率 | bars 中位 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 3.173 | 0.000 | 8.000 | 69.0% | 97.3% | 2 |
| 1.00 | 2.378 | 0.833 | 6.801 | 21.4% | 92.8% | 4 |
| 3.00 | 1.498 | 0.833 | 4.087 | 7.3% | 80.4% | 23 |

### 读法（只描述，不宣布最佳参数）

- **RR 期望随风险单调下降**：0.25 ATR 均值 3.5R，3.0 ATR 均值 1.6R。
  这与"止损越小、单位风险的倍数自然越大"一致。
- **但归零率同时单调下降**：0.25 ATR 有 **68.5%** 的接触最终一无所获
  （止损先被触发），3.0 ATR 只有 7.3%。
- 因此"最佳 RR = 8"这类说法会严重误导 —— 它对应的是 68.5% 归零率。
- **中位 R 在 1.0 ATR 附近达到拐点**（0.84），之后基本走平
  （1.5/2.0/3.0 ATR 分别为 0.88/0.90/0.91），但所需 bar 数从 4 根
  涨到 26 根。**1.0 ATR 之后继续放宽止损，几乎换不到更多中位收益，
  只会显著拉长持有时间。**
- 达到 best target 的中位 bar 数很短（0.25 ATR：2 根；1.0 ATR：4 根），
  但 p90 很长（69 / 109 根）—— 分布极度右偏。

## 六、多周期 liquidity 场有多密

以决策价格为 0、ATR0 为单位，可见 liquidity 的分箱平均数量：

| scope | (−4,−2] | (−2,−1] | (−1,−0.5] | (−0.5,0] | (0,0.5] | (0.5,1] | (1,2] | (2,4] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **5m** | 8.72 | 4.29 | 2.20 | 1.90 | 2.46 | 2.07 | 4.11 | 8.12 |
| **1h** | 0.69 | 0.34 | 0.18 | 0.15 | 0.21 | 0.16 | 0.32 | 0.64 |
| TRADING_DAY | 8.58 | 4.26 | 2.18 | 1.86 | 2.46 | 2.06 | 4.08 | 7.98 |

- **5m liquidity 极密**：±0.5R 内平均约 4.4 个，±2R 内约 21 个。
- **1h liquidity 很稀疏**：±0.5R 内平均仅 **0.36 个**，
  即多数接触时点附近根本没有 1h 级别的 liquidity。
- 5m 与 TRADING_DAY 的密度高度重合（数值几乎相同），
  说明大量 TRADING_DAY level 与 5m level 同价或近价 —— 这直接对应下一节。

## 七、liquidity 是否成簇出现

| same_price_identity_count | n | share |
|---|---:|---:|
| 1（单一） | 16,929 | 17.46% |
| 2 | 12,853 | 13.26% |
| 3 | 11,556 | 11.92% |
| 4 | 9,633 | 9.94% |
| 5 | 7,777 | 8.02% |
| ≥6 | 39,211 | 40.4% |

**82.5% 的接触位置存在多重 liquidity 叠加**（≥2 个 identity 同价）。
只有 17.5% 是"干净的单一 liquidity"。

这是本 Atlas 最重要的结构性事实之一：
以后研究"某个 liquidity 被扫"时必须意识到，**该价位通常同时是多个
周期、多个类型的 liquidity**。

## 八、OB 场

| 变量 | n | p10 | p25 | 中位 | p75 | p90 | 缺失率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| nearest_opposing_ob_distance_R | 85,162 | 0.094 | 0.200 | **0.435** | 1.500 | 41.72 | 12.2% |
| nearest_opposing_ob_width_R | 85,162 | 0.357 | 0.556 | 0.882 | 1.346 | 2.000 | 12.2% |
| nearest_same_direction_ob_distance_R | 79,390 | 0.101 | 0.208 | 0.500 | 1.875 | 43.22 | 18.1% |

反向 OB 距离中位数 **0.435R**，但 p90 高达 41.7R —— 分布极端右偏：
多数接触点外侧确实有近的反向 OB，少数则远得离谱。
**12.2% 的接触找不到外侧反向 OB。**

## 九、Oracle 方向与状态变量的关系（二维画像）

### liquidity scope × Oracle 方向

| scope | LONG_DOM | MIXED | SHORT_DOM |
|---|---:|---:|---:|
| 5m | 3,748 | 7,834 | 3,706 |
| 15m | 1,075 | 2,248 | 1,032 |
| 1h | 294 | 565 | 273 |
| CONTIG_SESSION | 14,224 | 31,363 | 13,638 |
| TRADING_DAY | 3,381 | 7,096 | 3,000+ |

**各 scope 的 LONG/SHORT 比例几乎一致**，MIXED 始终占约 55%。
→ **被扫 liquidity 的级别不决定事后方向优势。**

### 外侧最近 liquidity 距离 × Oracle 方向

| nearest_ahead_R | LONG_DOM | MIXED | SHORT_DOM |
|---|---:|---:|---:|
| (0, 0.5] | 20,199 | 45,543 | 18,816 |
| (0.5, 1.0] | 1,530 | 3,064 | 1,385 |
| (1.0, 2.0] | 673 | 940 | 585 |
| (2.0, 4.0] | 272 | — | — |

同样 **LONG/SHORT 高度对称**，前方 liquidity 距离不改变方向优势。

### 反向 OB 距离 × Oracle 方向

| OB distance (R) | LONG_DOM | MIXED | SHORT_DOM |
|---|---:|---:|---:|
| (0, 0.25] | 6,251 | 15,333 | 6,446 |
| (0.25, 0.5] | 4,283 | 9,478 | 4,114 |
| (0.5, 1.0] | 3,186 | 6,713 | 2,945 |
| (1.0, 2.0] | 1,891 | — | — |

**依然对称。** OB 距离不改变 LONG/SHORT 支配比例。

## 十、回答 15 个问题

1. **多少独立 liquidity、多少真实 contact？**
   80,289 个 liquidity identity，96,959 次真实接触。

2. **一条 liquidity 通常被接触几次？**
   平均 1.208 次；78.5% 只被接触一次。

3. **第一/二/三次接触有什么不同？**
   首次接触 80% 以上是穿透类（reclaim/close beyond/gap）；
   重复接触中 TOUCH_ONLY 占比显著上升（首次接触中 TOUCH_ONLY 只占 59.7%）。
   **越靠后的接触越倾向于"碰一下不穿"。**

4. **touch / sweep-reclaim / close-beyond 比例？**
   TOUCH_ONLY 22.0% / PENETRATE_RECLAIM 23.5% /
   PENETRATE_CLOSE_BEYOND 31.2% / GAP_CROSS 16.7% /
   PENETRATE_CLOSE_AT 6.5%。

5. **接触发生时最常见的多周期状态？**
   见 `trend_joint_profile.csv` 与
   `two_dimensional_profiles/trend_1h_x_15m_and_15m_x_5m.csv`。

6. **多周期 liquidity 在价格附近有多密？**
   5m：±0.5R 内约 4.4 个；1h：±0.5R 内仅 0.36 个。5m 极密、1h 稀疏。

7. **不同级别 liquidity 是否成簇？**
   是，**82.5%** 的接触位置有 ≥2 个 identity 同价叠加。

8. **ATR stop 从 0.25 到 3.0，可实现 RR 如何变化？**
   均值从 3.51R 单调降到 1.63R；归零率从 68.5% 单调降到 7.3%。

9. **RR 越高是否明显需要更多 bar？**
   是。达到 best target 的中位 bar 数：0.25 ATR→2 根，1.0→4 根，
   3.0→26 根。p90 从 69 根涨到 268 根。

10. **哪个 ATR 风险区间的 RR/时间前沿最好？（只描述）**
    中位 R 在 **1.0 ATR** 附近达到 0.84 后基本走平（3.0 ATR 也仅 0.91），
    但持有 bar 数从 4 根增至 26 根。
    按"每单位时间收益"看，0.75–1.5 ATR 区间看起来最经济；
    **但这只是画像描述，不构成策略参数建议。**

11. **LONG/SHORT/MIXED 各占多少？**
    MIXED 51.93% / LONG_DOMINATES 24.12% / SHORT_DOMINATES 23.24% /
    NO_CLEAR_ACTION 0.72%。

12. **Oracle 方向与多周期趋势的关系？**
    见 `two_dimensional_profiles/` 与 `trend_joint_profile.csv`；
    需在下一阶段用足够样本的组合正式检验，本轮不做结论。

13. **Oracle 方向与 liquidity 分布的关系？**
    与 liquidity scope、外侧最近 liquidity 距离均**未观察到方向偏移**
    —— LONG/SHORT 占比在各档位高度对称。

14. **Oracle 方向与 OB 距离/freshness 的关系？**
    同样**未观察到方向偏移**。OB 距离从 0.25R 到 2R，
    LONG/SHORT 支配比例保持稳定。

15. **哪些状态有足够样本值得下一阶段正式检验？**
    - contact_number 1–3（≥4,541）：可做"第几次接触"比较
    - 各 contact_type（≥6,324）：可做接触类型比较
    - liquidity scope：5m / 15m / 1h / CONTIG_SESSION / TRADING_DAY 均 ≥884
    - OB 距离 (0,0.25] / (0.25,0.5] / (0.5,1.0]：均 ≥2,900
    - 前身 liquidity 距离 (0,0.5]：≥1,385
    - **LONG/SHORT 支配作为目标变量**：24,000 / 22,500，样本充足
    - 需谨慎：contact_number ≥5（<912）、1h liquidity（1,182）、
      NO_CLEAR_ACTION（694）

## 十一、已知限制

1. Oracle 向后扫描上限 500 根 5m bar（约 42 小时）；
   超过仍未止损记为 `CENSORED_NO_STOP`。
2. target 只取决策时点已存在、且距离 ≤20 ATR 的 liquidity，
   最多 400 个（按距离取最近）。
3. 同 bar 同时触发 stop 与 target 时标记为
   `AMBIGUOUS_INTRABAR_ORDER`，同时给出保守 / 乐观两档 RR，
   未自行规定先后顺序。
4. `GAP_CROSS`（16.7%）的"接触"是开盘跳空，其路径与普通 sweep
   不可混同，使用时需单独分层。
5. OB 场缺失率 12.2%（找不到外侧反向 OB）。
6. 4h 仍只作为 environment direction，未做结构趋势。
7. 未做多重检验校正；本轮所有二维画像**只描述不检验**。

## 十二、输出文件

```text
liquidity_master_v1.parquet          80,289 行
liquidity_contacts_v1.parquet        96,959 行
liquidity_state_snapshot.parquet     96,959 行（多周期趋势 + OB 场）
liquidity_field_snapshot.parquet     96,959 行（标准化多周期流动性场）
oracle_risk_frontier.parquet      1,357,426 行（contact × 2 方向 × 7 风险）
liquidity_oracle_labels.parquet      96,959 行（含 oracle_direction）
oracle_pareto_frontier.parquet      240,118 行（RR × bars × risk 前沿）

sample_funnel.csv / contact_type_profile.csv /
contact_number_profile.csv / trend_joint_profile.csv /
liquidity_field_profile.csv / liquidity_overlap_profile.csv /
ob_field_profile.csv / risk_atr_rr_profile.csv /
risk_atr_holding_profile.csv / oracle_direction_profile.csv

two_dimensional_profiles/
  contact_type_x_contact_number.csv
  trend_1h_x_15m_and_15m_x_5m.csv
  liquidity_scope_x_contact_type.csv
  liquidity_scope_x_oracle_direction.csv
  same_price_has_multi_x_oracle_direction.csv
  penetration_depth_bin_x_oracle_direction.csv
  nearest_ahead_R_bin_x_oracle_direction.csv
  nearest_opposing_ob_distance_R_bin_x_oracle_direction.csv

AUDIT_ORACLE_ATLAS_V1.json
```

本轮未做：机器学习、特征重要性、SHAP、PnL、策略回测、
筛最佳参数、挑最好 subgroup。
