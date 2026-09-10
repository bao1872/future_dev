# 多周期流动性 × 订单块共振实验 v3.0

Base: `193d04f847b9c7a748ba616991facad75839645b`

```text
不训练机器学习 / 不计算夏普 / 不计算策略收益 / 不寻找入场点 / 不优化止损
不使用固定未来 N 根 K 线标签 / 不研究 2.5R
不重新定义订单块 / 不重新定义 displacement
```

**本轮状态：几何补全完成；Landmark A 匹配 Gate 未过；outcome 保持 LOCKED。**

---

## 一、对照组修复（第一步）

原代码用 `opposing_ob_count == 0` 定义"无 OB 共振"，实际代表
**根本没有外侧反向 OB**，与报告口径不一致。已修正为：

```python
treatment = OB_BEYOND_LIQUIDITY &  has_opposing_ob_hit
control   = OB_BEYOND_LIQUIDITY & ~has_opposing_ob_hit
                                &  opposing_ob_count > 0
```

验证结果（完全吻合预期）：

| | n |
|---|---:|
| OB_BEYOND_LIQUIDITY 总计 | **48,655** |
| Treatment（扫进 OB） | **15,177** |
| Control（外侧有 OB 但未扫到） | **33,478** |
| 合计 | **48,655** ✓ |

## 二、订单块几何补全（无论 hit / miss 都保存）

所有 `OB_BEYOND_LIQUIDITY` 事件现在都记录 liquidity 外侧**最近的反向
active OB**（此前只有 hit 组有 OB 信息，miss 组全空）：

`nearest_ob_id` / `nearest_ob_source_tf` / `nearest_ob_near_edge` /
`nearest_ob_far_edge` / `nearest_ob_distance_R` / `nearest_ob_width_R` /
`nearest_ob_prior_enter_count` / `nearest_ob_freshness`

### ob_margin_R 断言

```python
ob_margin_R = penetration_depth_R - nearest_ob_distance_R
            = side * (extreme - near_edge) / R0
```

| 断言 | 结果 |
|---|---|
| treatment `ob_margin_R >= -TOL` | **True**（min = −0.000000） |
| control `ob_margin_R < TOL` | **True**（max = −0.002999） |

容忍度 `TOL = 1e-9`（统一极小数值容忍，非 ATR 阈值）。

### 几何画像（R 单位，`R0 = ATR5[t0-1]`）

| 变量 | 组 | p10 | p25 | median | p75 | p90 |
|---|---|---:|---:|---:|---:|---:|
| nearest_ob_distance_R | HIT | 0.133 | 0.238 | **0.455** | 0.900 | 1.579 |
| nearest_ob_distance_R | MISS | 0.710 | 1.270 | **2.500** | 5.000 | 8.929 |
| nearest_ob_width_R | HIT | 0.625 | 0.952 | 1.486 | 2.432 | 3.998 |
| nearest_ob_width_R | MISS | 0.676 | 1.027 | 1.667 | 2.791 | 4.521 |
| ob_margin_R | HIT | 0.000 | 0.156 | 0.485 | 1.146 | 2.356 |
| ob_margin_R | MISS | −7.917 | −4.054 | **−1.750** | −0.675 | −0.278 |

**关键读数：HIT 组最近 OB 距离中位数 0.455R，MISS 组 2.500R。**
即"是否扫进 OB"主要由 **OB 离 liquidity 多远** 决定，
其次才是扫破深度。

### 最近 OB 周期分布

| hit | 5m | 15m | 1h |
|---|---:|---:|---:|
| False (MISS) | 16,553 | 10,830 | 6,095 |
| True (HIT) | 8,702 | 4,480 | 1,995 |

### 新鲜度分布

| hit | FRESH | RETESTED |
|---|---:|---:|
| False (MISS) | 28,486 | 4,992 |
| True (HIT) | 10,686 | 4,491 |

### 进入方式（5 分类）

| mode | n |
|---|---:|
| NEAR_MISS | 33,478 |
| WICK_ENTER_OB | 7,344 |
| CLOSE_INSIDE_OB | 4,926 |
| TRAVERSE_OB | 1,518 |
| CLOSE_BEYOND_OB | 1,389 |

## 三、共同支撑画像

`penetration_depth_R` 分箱下 hit / miss 双侧样本量：

| depth 区间(R) | n | n_hit | n_miss | hit_rate |
|---|---:|---:|---:|---:|
| (−inf, 0.25] | 9,031 | 880 | 8,151 | 0.0974 |
| (0.25, 0.5] | 11,271 | 2,141 | 9,130 | 0.1900 |
| (0.5, 0.75] | 7,279 | 1,979 | 5,300 | 0.2719 |
| (0.75, 1.0] | 5,467 | 1,951 | 3,516 | 0.3569 |
| (1.0, 1.5] | 6,189 | 2,624 | 3,565 | 0.4240 |
| (1.5, 2.0] | 3,265 | 1,634 | 1,631 | 0.5005 |
| (2.0, inf] | 6,099 | 3,926 | 2,173 | 0.6437 |

**共同支撑判定：PASS**（≥3 个分箱 hit/miss 各 ≥200；实际 7/7 分箱满足）。

但必须注意：**hit_rate 随扫破深度从 9.7% 单调升到 64.4%**。
这说明 `penetration_depth_R` 与 `has_opposing_ob_hit` 高度共线。

## 四、Landmark A 匹配 —— **GATE FAIL**

Landmark A universe（CLOSE_BEYOND + OB_BEYOND_LIQUIDITY）= 24,986
→ Treatment 10,237 / Control 14,749。

匹配设置（严格按合同，未调参）：
- Exact：symbol × liquidity_type × side × session_type × sweep_vs_1h
- Numeric（symbol 内 median/IQR robust scaling）：14 个变量
- **明确排除** `nearest_ob_distance_R` 与 `ob_margin_R`
- K=3 with replacement；不同 canonical trading_day；±120 日
- caliper：`|z(close_beyond_R)| ≤ 0.50`；session 位置 ≤ 60 分钟

### Coverage（要求 overall ≥ 0.80，主要 type/symbol ≥ 0.70）

| 维度 | coverage |
|---|---:|
| **ALL** | **0.4498** ❌ |
| PREV_CONTIG_SESSION_HIGH | 0.6138 ❌ |
| PREV_CONTIG_SESSION_LOW | 0.6410 ❌ |
| CONFIRMED_SWING_HIGH | 0.2119 ❌ |
| CONFIRMED_SWING_LOW | 0.1827 ❌ |
| PREV_TRADING_DAY_HIGH | 0.2674 ❌ |
| PREV_TRADING_DAY_LOW | 0.2358 ❌ |
| symbol 范围 | 0.3795 – 0.5513 ❌ |

全部未达标。

### Balance（要求 overall ≤ 0.10，主要组 ≤ 0.15）

`max |SMD| overall = 0.2927`，`max |SMD| by group = 0.4781`。❌

最差变量：

| variable | SMD |
|---|---:|
| **penetration_depth_R** | **0.2927** |
| pre_range_12_R | 0.2668 |
| volume_z_t0 | 0.1943 |
| level_age_log1p | 0.1750 |
| bar_range_R | 0.1619 |

control reuse max = 14（可接受，非主要失败原因）。

### 判定

```text
LANDMARK_A_MATCHING_FAIL
```

**outcome 保持 LOCKED** —— 未计算任何 `later_reclaim` /
`STRUCTURAL_ACCEPTANCE`。按合同未调 K / caliper / exact 约束 / 距离度量。

## 五、失败原因诊断（这是本轮最重要的发现）

**不是实现 bug，是结构性的可识别性问题，而且它精确验证了第三节的洞察。**

`has_opposing_ob_hit` 在数学上等价于：

```text
penetration_depth_R >= nearest_ob_distance_R
```

因此：

1. `penetration_depth_R` 与 hit 天然高度共线（hit_rate 9.7% → 64.4%）。
2. 在同一个 exact cell（symbol × type × side × session × sweep_vs_1h）内，
   同时满足"depth 接近"和"hit 状态不同"的 control 极其稀少
   → 覆盖率只有 45%。
3. 强行匹配 depth 后，残差仍不平衡（SMD 0.2927），
   因为 depth 相近时 hit 状态几乎被确定。

这正对应你第三节给出的两个案例：

```text
案例 A：OB 距 liquidity 0.3R，价格扫 0.5R → 扫进 OB
案例 B：OB 距 liquidity 0.8R，价格扫 0.5R → 没扫进 OB
```

在"同样扫了 0.5R"的条件下，A/B 的真正差别就是 **OB 距离**。
而 OB 距离是我们要研究的结构变量，不能匹配掉；
但 depth 匹配不掉又会留下混淆。

**结论：简单的"匹配 sweep 强度 → 比较 hit/miss"走不通。**
这不是可以靠调 K 或 caliper 解决的。

## 六、下一步建议（未执行，供裁定）

数据本身指向一个更干净的识别策略，即你第九节的思路：

### 建议 A（推荐）：`ob_margin_R` 断点设计

在 `ob_margin_R ≈ 0` 附近比较紧邻两侧：

```text
ob_margin_R ∈ [-δ, 0)  差一点没到 OB near edge
ob_margin_R ∈ [0, +δ]  刚刚进入 OB
```

这里 depth 与 distance 之和被局部固定，断点两侧只在"是否跨过 OB 边界"
上不同，识别最干净。可直接回答：

> 价格真正跨入 OB 边界时，市场状态是否发生变化？

这也自然避免了对 depth 的全局匹配。

### 建议 B：在共同支撑最好的 depth 区间内分层比较

`(1.5, 2.0]` 区间 hit_rate ≈ 0.50（1,634 vs 1,631，近乎完美平衡），
`(1.0, 2.0]` 也是不错的区间。可在该区间内直接比较 hit / miss。

### 建议 C：改用 `nearest_ob_distance_R` 作为连续暴露变量

不做二元 hit/miss，直接看 reclaim 概率随 OB 距离 / margin 的形状。

## 七、本轮可确认的事实（不依赖 outcome，均可信）

1. **94.6%（48,655/51,424）** 的真实 liquidity 穿透，其外侧存在事前活动的
   反向订单块 —— 说明"外面有没有 OB"几乎没有筛选能力。
2. 但**只有 29.5%（15,177）真正扫进**该 OB —— "是否真正进入"仍有筛选价值。
3. 扫进的 OB 以 **5m 为主（8,702）**，15m 次之（4,480），1h 最少（1,995）；
   **70.4% 是 FRESH**。
4. **HIT 组最近 OB 距离中位数 0.455R，MISS 组 2.500R** ——
   hit/miss 主要由 OB 距离驱动。
5. 进入方式分布：NEAR_MISS 33,478 / WICK_ENTER 7,344 /
   CLOSE_INSIDE 4,926 / TRAVERSE 1,518 / CLOSE_BEYOND_OB 1,389。

**"扫进订单块是否提供额外预测信息"仍未回答** —— outcome 保持锁定。

## 八、未执行

- Landmark A outcome（later_reclaim 比较）
- Landmark B（reclaim 后 reversal MSS）
- 按 OB 周期 / freshness / 品种 / F1-F4 / AGAINST-WITH 1h / 4h 分层的效应
- 同方向 OB sensitivity
- pytest

## 九、已知限制

1. Landmark A 匹配失败，未做后续分析。
2. `minute_from_session_open` 由 session 分段重算（5 分钟断口），
   非交易所官方 session 时钟。
3. 未做多重检验校正。
4. 共同支撑 PASS 是"双侧都有样本"的意义，不等于"匹配后可比"。
