# Reference Semantics Audit

**目的**：确认以 `trigger-bar close` 作为 Phase 1 `reference_price`，是否真正代表
"OB entered 后尚未明显走掉行情" 的合理测量时点。

**不训练任何模型、不修改标签、不修改 ATR、不进入 Phase 2。**
`PHASE 1 NATIVE-DIRECTION = FAIL` 在本审计期间保持不变。

---

## 0. 字段与坐标约定

Source Owner 未提供 `touch_price` / `first_touch_price` / `touch_reference`
（已 grep 确认 `build_ob_candidate_universe_v3.py` 无此类字段）：

    INTRABAR_TOUCH_PRICE_UNRESOLVED

因此**精确的 intrabar 首次触碰价与时间不可恢复**。本审计改用
**canonical 进入边界 `near_edge`** 作为可无歧义确定的参考水平。

| | LONG (d=+1) | SHORT (d=−1) |
|---|---|---|
| near_edge（价格进入时先触及的边） | `zone_high` | `zone_low` |
| far_edge（失效侧） | `zone_low` | `zone_high` |

依据 `build_ob_candidate_universe_v3.touch_bar_state`：bias=1 时 far edge
= `zone_low`；bias=−1 时 far edge = `zone_high`。

**交叉校验**：本审计计算的 `close_beyond_far_edge` 与 canonical
`touch_close_beyond_far_edge` 一致率 = **1.000000（n=79,896）**。
→ near/far edge 与方向坐标约定与 Source Owner 完全一致。

---

## 1. trigger close 相对 OB zone 的位置

| 位置 | 占比 |
|---|---:|
| close 在 zone **内** | **42.72%** |
| close 回到 **进入侧之外**（未真正收进 zone，边界被拒） | 50.27% |
| close 已越过 **失效边** | 7.01% |

按品种 38.9%–47.9% 收在 zone 内；按 source_tf：5m 40.7% / 15m 44.9% / 1h 47.6%；
LONG 42.4% / SHORT 43.0%。**各层一致。**

> 说明：canonical `OB_ENTERED` 由"价格触及 zone 边界"触发，并不要求收盘落在
> zone 内。因此约一半事件在触发 bar 收盘时价格又回到进入侧之外——这是
> canonical 语义本身，不是本审计引入的偏差。

## 2. entered → close 已走掉多少 R（核心量）

`touch_to_close_R = d × (close − near_edge) / ATR5`

| | p10 | p25 | **median** | p75 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ALL** | −0.7500 | −0.2778 | **+0.0256** | +0.3846 | +0.7500 | +1.0000 | +1.5626 |
| LONG | — | — | +0.0403 | — | 0.7500 | — | — |
| SHORT | — | — | 0.0000 | — | 0.7500 | — | — |
| 5m | — | — | +0.0458 | — | 0.7500 | — | — |
| 15m / 1h | — | — | 0.0000 | — | 0.7143–0.7500 | — | — |

- 恰好 = 0 的事件占 11.60%（收盘正好落在 canonical 边界上）
- \|z\| ≤ 0.05R 占 13.61%
- z < 0 占 38.13%，z > 0 占 50.27%

**中位事件在触发 bar 收盘时，相对 canonical 进入水平仅位移 +0.026R。**
分品种 median 全部落在 0.0000–0.1667R 之间。

## 3. trigger bar 内的 excursion（上界，含重要 caveat）

| 指标 | 占比 |
|---|---:|
| native MFE ≥ 0.5R | 85.38% |
| native MFE ≥ 1.0R | 43.60% |
| native MFE ≥ 1.5R | 13.15% |
| native MFE ≥ 2.5R | 1.19% |
| native MAE ≥ 0.5R | 33.12% |
| native MAE ≥ 1.0R | 14.17% |
| **bar 内方向与顺序不可分辨**（同时 MFE≥0.5R 且 MAE≥0.5R） | **22.04%** |

> ⚠️ **MFE 必须从 near_edge 起算，因而包含了"进入之前的接近段"。**
> 对 LONG 而言价格在进入前本就位于 zone 上方，所以 bar 的 high 天然高于
> near_edge——`MFE ≥ 1R` 的 43.6% 主要反映的是**进场前的价格位置**，
> 而非"进入后已经顺势走了 1R"。由于 5m OHLC 无法区分 bar 内先后，
> 该值是**上界**，不能解读为"已实现的顺势行情"。
>
> 真正确定的量是 §2 的 `touch_to_close_R`（median +0.026R）。

## 4. 按滞后程度分组的 base rate（仅描述）

| touch_to_close_R | n | 占比 | native base rate |
|---|---:|---:|---:|
| ≤ 0R | 39,730 | 49.73% | 0.2916 |
| 0 ~ 0.25R | 12,262 | 15.35% | 0.2855 |
| 0.25 ~ 0.5R | 13,041 | 16.32% | 0.2889 |
| 0.5 ~ 1R | 11,193 | 14.01% | 0.2889 |
| > 1R | 3,670 | 4.59% | 0.2961 |

**极差仅 1.06 个百分点，无单调关系。** 最滞后的组（>1R）base rate 0.2961
甚至略高于最不滞后的组。

→ **当前 FAIL 并不集中在"close 已经严重滞后"的事件上。**

---

## 裁决

| 检查项 | 结果 | 判断 |
|---|---|:--:|
| 大多数 close 仍接近 OB？ | 中位位移 +0.026R；\|z\|≤0.05R 占 13.6% | ✅ |
| entered→close 已走掉的 R？ | median +0.026R，p75 +0.385R，p90 +0.75R | ✅ 小 |
| bar 内已实现重大顺势 excursion？ | 唯一可确定的量 median 0.026R；MFE 上界被进场前位置污染 | ✅ 低 |
| FAIL 集中在滞后事件？ | base rate 在滞后分组间平坦（极差 1.06pp） | ❌ 不集中 |

### 结论：**trigger-close reference is adequate**

当前 `reference_price = trigger bar close` **在结构上并不滞后**：
中位事件在收盘时相对 canonical 进入边界仅位移约 0.03R，
且 FAIL 结果与滞后程度无关（分组 base rate 平坦）。

因此：

    PHASE 1 NATIVE-DIRECTION = FAIL   （维持，正式接受）

---

## 结论措辞修正

按裁决，将此前表述中过强的部分收紧：

- ❌ "不同品种×周期×多空组合的成功率存在**稳定**差异"
- ✅ "存在**很弱的**条件基准率差异"（M1 one-hot：AUC 0.5072 / Lift@20 1.0361）

---

## 输出文件

    reference_semantics_profile.csv     总体画像
    reference_by_symbol.csv             分品种
    reference_by_source_tf.csv          分周期
    reference_by_direction.csv          分多空
    trigger_bar_excursion.csv           滞后分组 × base rate
    reference_semantics_events.csv      逐事件明细

## 已知限制

1. `INTRABAR_TOUCH_PRICE_UNRESOLVED`：无 canonical 触碰价，
   以 `near_edge` 作为进入水平；bar 内顺序不可知，MFE/MAE 仅为上界。
2. 22.04% 的 trigger bar 同时存在 ≥0.5R 的顺向与逆向波动，其先后不可判定。
3. 7.01% 的事件在触发 bar 收盘时已越过失效边——这些事件在参考点已处于
   "尚未开始扫描即已失效"的状态，但比例不足以解释整体结论。
4. 本审计只评价 reference price 语义；ATR5 作为 R 的合理性未在本轮检验
   （按设计一次只改一个变量）。
