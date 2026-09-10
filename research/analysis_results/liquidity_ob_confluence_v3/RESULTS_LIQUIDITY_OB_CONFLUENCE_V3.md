# 多周期流动性 × 订单块共振实验 v3.0（阶段一：订单块与共振层）

Base: `efce81a1464f13805f14313f1285979369ca962a`

```
不训练机器学习 / 不计算夏普 / 不计算策略收益 / 不寻找入场点 / 不优化止损
不使用固定未来 N 根 K 线标签 / 不研究 2.5R
不重新定义订单块 / 不重新定义 displacement
```

## 当前状态

**订单块活动图与共振分类已完成并通过因果审计。**
**Landmark A / Landmark B 的匹配与结果尚未执行。**
本文件**不得**被当作订单块有效性的结论。

---

## 一、订单块来源完全复用 canonical

- `build_full_ob_smc_tf`（每个周期）
- `replay_ob_lifetimes`（canonical 活动/失效/驱逐生命周期）
- `spatial_ob_bounds`（空间边界，处理端点互换）
- `smc["ob_lifecycle_events"]` 中的 `OB_ENTERED`（用于新鲜度）

来源周期仅 **5m / 15m / 1h**。**未伪造 4h 订单块**（仓库无合法 canonical 4h OB）。

每个 OB 记录：`ob_id` / `ob_source_tf` / `ob_bias` / `ob_available_time` /
`ob_inactive_time` / `ob_inactive_reason` / `zone_low` / `zone_high` /
`endpoints_swapped` / `ob_enter_times`。

### 因果边界（硬断言）
```python
ob_available_time <= penetration_bar_start
ob_inactive_time is None or ob_inactive_time > penetration_bar_start
```
即：**penetration bar 当根才确认的、未来才形成的、已失效的订单块，一律排除**。

### 订单块活动图规模（ob_active_map_audit.csv）

| symbol | OB 数 |
|---|---:|
| AG | 3,700 |
| AL | 2,674 |
| AU | 3,681 |
| CF | 1,917 |
（其余品种见审计文件）

---

## 二、共振分类

输入：**fresh penetrated interactions = 51,424**
（即 v2.0 的 `VALID_AHEAD + CONTINUOUS_CROSS + PENETRATED`）

### 几何关系分布（ob_geometry_relation.csv）

| relation | n | 占比 |
|---|---:|---:|
| **OB_BEYOND_LIQUIDITY（Primary 族）** | 48,655 | 94.61% |
| OB_CONTAINS_LIQUIDITY | 1,229 | 2.39% |
| NO_RELEVANT_OB | 1,094 | 2.13% |
| OB_BEFORE_LIQUIDITY | 446 | 0.87% |

Primary 只保留 **OB_BEYOND_LIQUIDITY**（OB 位于 liquidity 外侧，
几何上必然先穿 liquidity 再进入 OB，不依赖 bar 内路径假设）。
其余三类单独画像，**不进入主结论**。

### 处理组 / 对照组（同一几何族内）

| 组 | 定义 | n |
|---|---|---:|
| **处理组 OB_CONFLUENCE=1** | 外侧存在反向 OB，且穿透极值**真正到达**其 near edge | **15,177** |
| **对照组 OB_CONFLUENCE=0** | 外侧存在反向 OB，但穿透极值**未到达** | **33,478** |

选取"同一几何族内 扫到 vs 未扫到"而非"有 OB vs 无 OB"，
是因为后者会混入"该位置附近根本没有 OB"的另一类市场结构，
不是最干净的对照。

### 反向 OB 方向规则

- 上方 BSL 被扫 → `reversal_direction = -1` → 只匹配 **bearish OB**
- 下方 SSL 被扫 → `reversal_direction = +1` → 只匹配 **bullish OB**

OB bias 只用于定义区域的历史结构方向，**不作为交易标签**。

### 主 OB 来源周期（处理组内）

| source_tf | n |
|---|---:|
| 5m | 8,702 |
| 15m | 4,480 |
| 1h | 1,995 |

### 新鲜度（处理组内，`prior_ob_enter_count` 只统计 t0 前的 canonical OB_ENTERED）

| | n |
|---|---:|
| FRESH（未被进入过） | 10,686 |
| RETESTED（已进入过 ≥1 次） | 4,491 |

### 同方向 OB
`same_direction_ob_count_in_path` / `same_direction_ob_hit` 已记录，
用于后续 sensitivity（排除同方向 OB 污染）。

---

## 三、⚠️ 未匹配的原始对比不可解读

未匹配的描述性数字如下，**仅供说明混淆程度，不得作为结论**：

| 组 | n | same_bar_reclaim | close_beyond | reversal_mss(占全体) |
|---|---:|---:|---:|---:|
| 未扫到 OB | 33,478 | 0.4614 | 0.4409 | 0.2131 |
| 扫进 OB | 15,177 | 0.2627 | 0.6767 | 0.1226 |

**这组数字完全被穿透深度混淆**：
"扫进 OB"在定义上就要求穿透极值更大，因此必然
更少同 bar 收回、更多收盘站到 level 外。
两组 `reversal_mss` 也不可比——该字段当前只对 SAME_BAR_RECLAIM 计算，
而两组的同 bar 收回比例差异巨大。

**必须完成 Landmark A / B 的匹配（含 `penetration_depth_R` 匹配变量）之后，
才允许给出任何效应结论。**

---

## 四、尚未执行的部分

- Landmark A：`CLOSE_BEYOND → LATER_RECLAIM vs 同方向新 BOS`，
  匹配（symbol × liquidity_type × side × session_type × sweep_vs_1h exact；
  K=3；`close_beyond` caliper 0.50；session 位置 caliper 60 分钟）、
  coverage/balance gate、outcome、bootstrap
- Landmark B：`RECLAIM → 反向 MSS vs 重新接受`，重新匹配与 gate
- 完整路径漏斗 `P(RECLAIM 且 REVERSAL_MSS | OB_CONFLUENCE)`
- 同方向 OB sensitivity
- 非 Primary 几何（CONTAINS / BEFORE）画像
- 多周期（4h×1h、1h×15m、1h×15m×5m）分层
- 14 项 pytest

## 五、本轮可确认的事实

1. 真实 liquidity 穿透中，**94.6%** 的位置外侧存在事前活动的反向订单块；
   其中 **29.5%**（15,177 / 51,424）的穿透极值真正扫进了该订单块。
2. 扫进的订单块以 **5m 为主（8,702）**，15m 次之（4,480），1h 最少（1,995）。
3. 扫进的订单块中 **70.4% 是 FRESH**（此前未被 canonical 进入过）。

以上三点均为**构造事实**，不依赖任何未来结果，可信。
"订单块是否提高状态转移概率"这一核心问题**仍未回答**。
