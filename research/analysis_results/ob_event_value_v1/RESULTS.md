# OB Event Value Test

**NO MODEL / NO ML / NO feature selection / NO parameter tuning.**
过去所有 Phase 1 结论（58.88%、12bar direction-agnostic、opportunity model）
仅作历史探索，未用于证明本实验。

**问题**：在事件发生前市场状态相似的情况下，出现 `OB_ENTERED` 的时点，
后续是否比没有 OB 的普通时点更容易发生价格扩张？

**最终分类：WEAK EVENT EFFECT**
（`OB event effect exists but is weak / heterogeneous`）

---

## 1. Treatment universe

| | 数量 |
|---|---:|
| candidate rows | 79,896 |
| **event_bar**（symbol × source_tf × bar） | **68,512** |
| ANY_TF dedup（symbol × bar） | 55,671 |
| collapse 掉的伪重复 | 11,384 |
| ANY_TF 再 collapse | 12,841 |

by source_tf：**5m 39,656 / 15m 19,823 / 1h 9,033**
bias composition：BULLISH 34,755 / BEARISH 33,615 / MIXED 142

## 2. Control pool

**448,125** 合法普通时点（`CLEAN_CONTROL_3`：360,129）。
条件：t0 无任何 source_tf OB_ENTERED、pre-history ≥ 21、完整未来 24 根、
未来 24 根不跨 discontinuity、`R0 = ATR5[t0-1] > 0`。
**未**要求"未来 24 根不得出现 OB"（post-treatment selection bias）。

## 3. Matching v1.1（唯一一次设计修订）

v1.0 的 BALANCE GATE 唯一超标项是 `pre_range_12_R`（0.1081 > 0.10），
故只增加一个定向 caliper：`|z_pre_range_t − z_pre_range_c| ≤ 0.50`
（symbol-level robust IQR 单位），在算最近邻**之前**过滤。
其余全部不变：symbol / session_type / 30min bucket / 不同 trading_day /
±120 交易日 / 6 个事前变量 / K=3 / with replacement。

| Gate | 结果 |
|---|---|
| **COVERAGE** | **PASS**：overall **0.9608**（≥0.85）；5m 0.9691 / 15m 0.9522 / 1h 0.9438（≥0.80）；unmatched 2,683 |
| **BALANCE** | **PASS**：6 个变量 overall \|SMD\| 全部 ≤ **0.0441**；by source_tf max 0.0485（≤0.15） |

```
MATCHING DESIGN FROZEN
```

## 4. 必须保留的事前事实（treatment assignment mechanism）

| variable | before-matching SMD | after |
|---|---:|---:|
| **pre_range_12_R** | **0.3678** | 0.0441 |
| **pre_rv_12_R** | **0.2835** | 0.0434 |
| volume_z_20_pre | 0.1078 | 0.0420 |
| atr_rel_pre | −0.0471 | 0.0348 |
| pre_ret_12_R | −0.0185 | −0.0075 |
| pre_ret_3_R | −0.0090 | −0.0078 |

> **OB_ENTERED events occur in materially more active pre-event market
> states than ordinary market bars.**

这不是小差异：OB 明显偏向发生在**此前已有较大振幅、较高实现波动**的环境。
因此若不做匹配，任何"OB 后波动更大"的发现都会被
"过去波动大 → 更容易产生 OB → 未来波动具有持续性"这条路径混淆。
**匹配后平衡不得从报告中删除此项。**

## 5. PRIMARY RESPONSE CURVE — max_directional_excursion_R

| H | OB | Control | Δ R | relative uplift |
|---:|---:|---:|---:|---:|
| 1 | 1.0543 | 0.9472 | **+0.1070** | **+11.30%** |
| 3 | 1.8014 | 1.6681 | +0.1333 | +7.99% |
| 6 | 2.5482 | 2.4212 | +0.1270 | +5.25% |
| **12** | **3.6503** | **3.5202** | **+0.1301** | **+3.70%** |
| 24 | 5.2358 | 5.1542 | +0.0816 | +1.58% |

**效应在第一根 bar 就已经出现（+0.107R），此后绝对差基本持平在 +0.13R 左右，
不再累积；相对 uplift 则随时间单调衰减（11.3% → 1.6%）。**

即：OB 之后不是"行情逐渐展开得更多"，而是**紧接着就多出约 0.1R 的一次性
扩张偏移**，随后与对照以相同速度继续扩张。

## 6. H=12 anchor（固定汇报点，bootstrap 500）

| outcome | OB | Control | Δ | 95% CI | relative | rel CI |
|---|---:|---:|---:|---|---:|---:|
| **max_directional_excursion_R** | 3.6503 | 3.5202 | **+0.1301** | [0.0712, 0.1911] | **+3.70%** | [+2.02%, +5.44%] |
| forward_range_R | 4.4726 | 4.3215 | +0.1511 | [0.0942, 0.2101] | +3.50% | [+2.17%, +4.88%] |
| realized_vol_R | 2.8179 | 2.7211 | +0.0969 | [0.0535, 0.1402] | +3.56% | [+1.96%, +5.16%] |
| abs_close_move_R | 2.2828 | 2.1966 | +0.0863 | [0.0395, 0.1368] | +3.93% | [+1.79%, +6.25%] |

所有连续指标 CI 下界均 > 0，且三个独立指标（range / realized vol /
close move）同方向——**不是单一指标偶然**。

### Barrier（secondary）

| | OB | Control | risk diff | 95% CI | relative risk |
|---|---:|---:|---:|---|---:|
| hit_abs_1p5R | 0.8518 | 0.8362 | +0.0156 | [0.0119, 0.0203] | +1.87% |
| hit_abs_2p5R | 0.5711 | 0.5494 | +0.0217 | [0.0157, 0.0284] | +3.95% |

## 7. 分层

### by source_tf

| tf | n | OB | Control | Δ | rel |
|---|---:|---:|---:|---:|---:|
| 5m | 38,429 | 3.7425 | 3.5863 | +0.1562 | +4.36% |
| 15m | 18,875 | 3.5707 | 3.4502 | +0.1205 | +3.49% |
| **1h** | 8,525 | 3.4107 | 3.3771 | **+0.0336** | **+1.00%** |

3/3 为正，但 **1h 几乎无效**。

### by symbol（macro median Δ = +0.1314）

14/15 为正（最强 M +0.2567 / AU +0.2168 / SC +0.2004；
最弱 **MA −0.0015**、CF +0.0117、CU +0.0640）。

### by bias composition

| | n | Δ | rel |
|---|---:|---:|---:|
| BULLISH | 33,502 | +0.1464 | +4.20% |
| BEARISH | 32,192 | +0.1101 | +3.10% |
| MIXED | 135 | +0.8611 | +19.58% |

（MIXED 仅 135 例，不解释。）

### by fold —— **时间不稳定**

| fold | n | Δ | rel |
|---|---:|---:|---:|
| F1 | 6,352 | +0.3104 | +9.11% |
| F2 | 6,258 | +0.3468 | +9.98% |
| F3 | 6,319 | +0.1013 | +2.81% |
| **F4** | 7,729 | **−0.0888** | **−2.38%** |

**3/4 折为正，但最近一折为负。**

## 8. 敏感性

| 检查 | Δ max_exc | rel | 结论 |
|---|---:|---:|---|
| ANY_TF dedup（54,297） | +0.1258 | +3.56% | **保留** |
| CLEAN_CONTROL_3（65,647，match 0.9583，max SMD 0.0801） | +0.1035 | +2.92% | **保留** |

CLEAN_CONTROL_3 的 forward_range +0.1174（+2.70%）、hit_2p5R +0.0171（+3.10%）。
效应不是由"对照点附近也有 OB"或伪重复放大出来的。

## 9. Native-direction falsification

| | OB | Control | Δ |
|---|---:|---:|---:|
| native_signed close move (H=12) | −0.0193 | +0.0004 | **−0.0198** |

**把 OB 自己的方向赋予其匹配对照后，OB 的顺向收益几乎为 0（且略负）。**
结合第 5–6 节：

```
unsigned expansion:  OB > control   (显著)
native signed:       OB ≈ control   (≈0，略负)
```

→ **OB 是扩张事件，不是方向事件。** 这与已关闭的 native-direction 主线
结论一致，且互相独立地再次得到验证。

---

## 10. 分类判定

| A（Strong）条件 | 实测 | 判定 |
|---|---|:--:|
| H12 Δ > 0 | +0.1301 | ✅ |
| bootstrap CI lower > 0 | 0.0712 | ✅ |
| **relative uplift ≥ 5%** | **+3.70%** | ❌ |
| range 或 rvol 独立同向且 CI > 0 | 两者皆是 | ✅ |
| ≥2/3 source_tf 为正 | 3/3 | ✅ |
| 多数 symbol 为正 | 14/15 | ✅ |
| ANY_TF dedup 保留 | +3.56% | ✅ |
| CLEAN_CONTROL_3 保留 | +2.92% | ✅ |

**未达 A（唯一卡在 relative uplift 3.70% < 5%），落入 B。**
且存在明确异质性：1h 仅 +1.00%，F4 为 −2.38%，MA 为 −0.05%。

### **WEAK EVENT EFFECT**

```
OB_ENTERED has a statistically robust but small unsigned expansion effect:
about +0.13R (~+3.7%) at H=12, present from the first bar, surviving
dedup and clean-control, but below the 5% pre-registered materiality
threshold, heterogeneous across timeframes, and negative in the most
recent fold.
```

---

## 11. 回答 7 个科学问题

1. **OB 是否比匹配对照更容易产生未来扩张？** 是，但小——H12 +0.13R（+3.7%），
   全部 CI 下界 > 0。
2. **效应有多大？** 约 **+0.13 ATR 单位**（对照 3.52R → OB 3.65R），相对 +3.7%。
   远小于"OB 是重要事件"应有的量级。
3. **从第几根出现、是否衰减？** **第 1 根就出现**（+0.107R，+11.3%），
   绝对差此后持平，**相对 uplift 单调衰减**到 H24 的 +1.6%。
4. **5m/15m/1h 一致？** 方向一致（3/3 正），但 1h 仅 +1.0%，实质无效。
5. **跨品种/时间稳定？** 品种 14/15 为正（稳）；
   时间折 **3/4，最近一折为负**（不稳）。
6. **仍然没有 native direction？** 是。native-signed Δ = −0.0198 ≈ 0。
7. **OB 主线值得继续吗？** 见下。

## 12. 结论与下一步

按预注册：落入 B 档 → **不要马上训练模型，先研究异质性来源。**

值得追的两个异质性信号：
- **1h OB 几乎无效应**（+1.0%）vs 5m（+4.4%）——可能与 1h OB 结构风险远大于
  ATR5（前一轮已测：中位 3.4×）有关，即 1h OB 的"扩张"在时间上更分散。
- **F4（最近一折）为负**——效应可能随时间衰减，需确认。

不建议的方向：换 outcome 定义、调 caliper/K、训练预测模型、回到 native trading。

## 13. 已知限制

1. 匹配只用了 6 个事前变量；未观测混杂（如订单流、持仓量）无法排除。
2. Control 允许未来出现 OB（避免 post-treatment selection），
   因此本设计测的是"OB 时点 vs 普通时点"，不是"有 OB 的整段 vs 无 OB 的整段"。
3. 未匹配"距上次 OB 的时间"等 OB 密度类变量。
4. 1h 样本量最小（8,525），分层结论需谨慎。
5. MIXED bias 仅 135 例，未解释。
6. 本轮未对多个 outcome 做多重检验校正；primary endpoint 为
   `max_directional_excursion_R`，其余为辅助/secondary。
