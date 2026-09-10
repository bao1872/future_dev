# Structural-OB Label（Native Direction 最后一次实验）

**结论：FAIL —— 且比 ATR5 版本更干净地失败。**

按路线图，这是 native-direction 标签最后一次有理论依据的修改：
把失败边界从"固定 1 ATR5"改为"OB 自己的失效边 far_edge"。
本轮之后不再修改 native-direction 标签。

---

## 0. 定义

    R_struct = |close − far_edge|         （方向化后 > 0 表示 OB 仍有效）
    target   = close + d × 2.5 × R_struct
    stop     = close − d × 1.0 × R_struct ≡ far_edge

因为 stop 恰等于 `far_edge`，本标签在数学上与既有 first-passage 状态机**完全同构**，
只是把 `r_ref` 从 ATR5 换成 `R_struct`，因此直接复用
`label_native_direction_event`，未另写一套语义。

`R_struct ≤ 0`（trigger close 时 OB 已失效）→ `INVALID_AT_DECISION`，
label=None，**不进入模型 universe**（避免制造机械可识别的 0 标签）。

---

## Step 1 门控：R_struct 与 ATR5 是否本质不同？

`R_struct / ATR5` 分布（n=79,896）：

| | p5 | p10 | p25 | **median** | p75 | p90 | p95 | p99 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | −0.226 | 0.170 | 0.758 | **1.406** | 2.270 | 3.500 | 4.615 | 7.778 | 1.703 |

| 区间 | 占比 |
|---|---:|
| 在 [0.8, 1.2] 内（≈ 与 ATR5 等价） | **15.72%** |
| < 0.5 ATR | 16.92% |
| > 1.5 ATR | 46.38% |
| > 2.0 ATR | 30.66% |

按 source_tf：

| source_tf | median | p25 | p75 | p90 | 落在 [0.8,1.2] |
|---|---:|---:|---:|---:|---:|
| 5m | 1.129 | 0.600 | 1.667 | 2.273 | 20.2% |
| 15m | 1.838 | 1.071 | 2.727 | 3.704 | 11.3% |
| **1h** | **3.409** | 2.000 | 5.000 | 7.105 | **3.9%** |

**门控结论：明显不同 → 有理由构造 structural label。**
1h OB 的中位结构风险是 ATR5 的 3.4 倍，说明原 ATR5 止损对 1h OB
确实过紧；5m 的中位 1.13 相对接近。

`INVALID_AT_DECISION` = **6,462（8.09%）**，已单独排除并报告。

---

## Step 2：Structural label 结果

### 标签画像

| status | n |
|---|---:|
| RESOLVED | 72,309 |
| INVALID_AT_DECISION | 6,462 |
| AMBIGUOUS_INTRABAR | 1,084 |
| END_OF_DATA_CENSORED | 41 |

**native structural base rate = 0.2777**（target 20,079 / stop 52,230）

### native vs flipped placebo（同 R_struct，方向相反）

| | n | native | flipped | diff | 95% CI |
|---|---:|---:|---:|---:|---|
| **ALL** | 71,611 | **0.2804** | **0.2936** | **−0.0132** | **[−0.0254, −0.0002]** |
| LONG | 36,449 | 0.2964 | 0.2743 | +0.0222 | — |
| SHORT | 35,162 | 0.2638 | 0.3136 | −0.0499 | — |

**CI 不含 0，且为负** —— 结构口径下 native 方向**显著略差于**反向 placebo。
LONG 略优（+0.0222）、SHORT 明显更差（−0.0499），多空不对称。

（对照 ATR5 版本：native 0.2907 / flipped 0.2959 / diff −0.0051 /
CI [−0.0167, 0.0069] 含 0。结构口径把差异放大且推向负值。）

### Nested：M1 metadata vs M3 metadata+state（只 Logistic）

| universe | 模型 | AUC | PR-AUC | Brier | Lift@20 | Top20 uplift |
|---|---|---:|---:|---:|---:|---:|
| ALL15 | M0 constant | 0.5000 | 0.2775 | — | 1.0000 | 0.0000 |
| ALL15 | **M1_metadata** | **0.5224** | 0.2821 | 0.199205 | 1.0342 | 0.0094 |
| ALL15 | M3_metadata_state | 0.5084 | 0.2796 | 0.202355 | 1.0239 | 0.0066 |
| NEW11 | ZS_M1 | **0.5300** | 0.2940 | 0.198307 | 1.0979 | 0.0268 |
| NEW11 | ZS_M3 | 0.4986 | 0.2761 | 0.217252 | 1.0194 | 0.0053 |

| universe | 比较 | ΔAUC | ΔPR-AUC | ΔBrier↓ | ΔTop20 uplift |
|---|---|---:|---:|---:|---:|
| ALL15 | M3 − M1 | **−0.0140** | −0.0025 | +0.003150 | −0.0028 |
| ALL15 | ZS_M3 − ZS_M1 | **−0.0209** | −0.0104 | +0.015729 | −0.0094 |
| NEW11 | M3 − M1 | **−0.0211** | −0.0053 | +0.004002 | −0.0100 |
| NEW11 | ZS_M3 − ZS_M1 | **−0.0314** | −0.0179 | +0.018945 | −0.0215 |

**全部为负，Brier 全部变差。**

### Within-stratum（symbol × source_tf × direction 组内）

| 数据集 | 可用 strata | 事件加权 uplift | 事件加权 Lift | macro 中位 Lift | 正 strata |
|---|---:|---:|---:|---:|---:|
| ALL15 M3 | 85 | **−0.0025** | 0.9929 | 0.9909 | 42/85 |
| NEW11 ZS_M3 | 63 | **−0.0097** | **0.9767** | 0.9311 | 23/63 |

组内排序能力为负。

### Paired block bootstrap（500 次，同一 resample 内算两者）

| 数据集 | ΔAUC 95%CI | ΔTop20 uplift 95%CI |
|---|---|---|
| ALL15 | [−0.0300, 0.0006] | [−0.0231, 0.0206] |
| **NEW11 zero-shot** | **[−0.0494, −0.0104]** | [−0.0468, 0.0024] |

**NEW11 的 ΔAUC 区间完全位于 0 以下** —— 在未见品种上，加入事件状态
不只是无助，而是**显著损害**。

### Resolution（结构口径）

median **10** 根（ATR5 口径为 5）；p75 29，p90 72，p95 124，p99 345。
按周期：5m 6 / 15m 15 / **1h 48**。结构风险更大 → 自然耗时更长。

---

## 裁决

| 判据 | 结果 | 判定 |
|---|---|:--:|
| native 明显优于 flipped？ | −0.0132，CI [−0.0254, −0.0002] 不含 0 | ❌ 反向更优 |
| M3 稳定超过 M1（ALL15）？ | ΔAUC −0.0140 | ❌ |
| M3 稳定超过 M1（NEW11 ZS）？ | ΔAUC −0.0314，CI 全负 | ❌ |
| within-stratum uplift > 0？ | ALL15 −0.0025 / NEW11 −0.0097 | ❌ |
| DEDUP / bootstrap 支持？ | NEW11 ΔAUC CI 全负 | ❌ |

### **Structural OB native-direction：FAIL**

---

## 综合结论（native-direction 路线关闭）

两条互相独立的失败：

1. **ATR5 口径**：native ≈ flipped（−0.0051，CI 含 0）；
   M3 − M1 ΔAUC −0.0033；组内 Lift 0.9925（NEW11）/ 0.9971（DEDUP）。
2. **结构口径**：native **显著差于** flipped（−0.0132，CI 不含 0）；
   M3 − M1 ΔAUC −0.0140（ALL15）/ −0.0314（NEW11）；
   组内 Lift 0.9929 / 0.9767；NEW11 ΔAUC bootstrap CI 全负。

而且换用"OB 自己的失效边界"这一更有理论依据的定义后，结果**没有改善反而更差**。

    Native-direction OB tradability: CLOSED.
    在 trigger close 观察点下，无论用 ATR5 还是 OB 自身结构边界定义风险，
    canonical OB 的原生方向既无平均优势，其触发时因果状态也无法在
    metadata 之上提供可复现的增量排序能力。

按路线图：**彻底结束 native direction，不再给第三次机会。**

## 下一步（尚未开始，仅记录路线）

研究问题改为：

    OB entered 本身是否是一个特殊的市场事件？
    —— OB Event Value Test（OB vs matched non-OB 对照）

不再问"哪个 OB 值得做多/做空"，而问
"OB 出现后，市场是否比普通时刻更容易发动行情"。
该研究需要先设计 matched-control 与标签定义，本轮未启动。
