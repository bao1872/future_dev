# Phase 1 Candidate v1 — FROZEN

状态：**Phase 1 discovery PASS**
冻结提交：`c1c4d1c`（实验）→ 本 freeze 提交
冻结日期：2026-09-10

> 这是 **discovery 层面**的通过，不是策略通过，也不是 alpha 最终确认。
> Phase 1 只回答"这个 OB 触发事件值不值得关注"，不涉及任何执行。

---

## 1. 冻结的业务定义

### Candidate

全部 **canonical OB entered**，全部 source_tf（`5m` / `15m` / `1h`）。
不做任何 source_tf / OB 宽度 / 方向 / 历史收益的提前过滤。

### Reference price

```text
trigger 5m bar close
```

### Risk unit

```text
R_ref = decision-time ATR5（真实波幅 5 根滚动均值，仅含当前及之前数据）
```

### Label（PRIMARY）

```text
horizon = 12 valid 5m bars（约 1 小时有效交易时间）

若在未来 12 根有效 5m bar 内，至少一个方向满足
    2.5R target 先于 1R adverse barrier 被触及
→ Tradable = 1
否则
→ Tradable = 0
```

- 方向中性，无 FOLLOW / FADE / target tier / action-relative 概念。
- 12 根内未形成机会 = **明确 0**，不是 censored。
- 同 bar 同时触及 target 与 stop → `AMBIGUOUS_INTRABAR`，排除。
- horizon 内遇到不可信边界 → `ROLL_CENSORED`，排除。
- 数据不足以覆盖完整 horizon → `END_OF_DATA_CENSORED`，排除。

### 历史 sensitivity（保留，不用于任何结论）

`24` / `48` / `unbounded` bar。仅作 horizon 衰减的记录。

### Features

冻结 **103 维 event-level causal features**（`PHASE1_FEATURES_V1`）。
硬性排除：动作相关、未来、标识、`*_rel_*`、`_zone_low/high`、`stop_structure_*`。
**`symbol` 与 `source_tf` 不进入全模型特征矩阵**（只作为 baseline 分组变量）。

### Model

```text
Logistic Regression
  SimpleImputer(median) → StandardScaler → LogisticRegression(max_iter=2000, random_state=SEED)
```

预处理 / 参数 / rolling folds / coverage 全部沿用，不得调整。

### Output

输出命名为 **Tradability Score**，**不是概率**。
理由：Brier 未证明优于 constant baseline，未做校准。

---

## 2. 品种 universe

Canonical 16（`run_direction_v3r.INSTRUMENTS`）。

| 项 | 值 |
|---|---|
| raw 5m 可用 | 16/16 |
| canonical OB 可构建 | **15/16** |
| DEV4（开发组） | AG, CU, M, RB |
| NEW11（零样本外推组） | AL, AU, CF, I, MA, NI, P, RU, SC, SN, TA |
| 排除 | **LC** |

**LC 排除原因**：`quantile_state_contiguous_oos` 要求 `len(train) >= 600`，
LC 仅 18,316 根 5m bar（AG 44,452），无法满足。
**不为了凑 16/16 放宽该规则。** 数据长度足够后自然纳入。

---

## 3. 权威结果（12bar primary）

### 3.1 核心对照

| universe | n | base rate | AUC | PR-AUC | D10−D1 | Lift@10 | Lift@20 | Lift@30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DEV4（历史基准） | 8,898 | 0.3708 | 0.5835 | 0.4374 | 0.2337 | 1.3153 | **1.2501** | 1.2264 |
| **NEW11 zero-shot** | 23,877 | 0.3686 | **0.5730** | 0.4160 | 0.1696 | 1.1984 | **1.2075** | 1.1963 |
| Baseline-ZS（仅 source_tf） | 23,877 | 0.3686 | 0.5152 | 0.3784 | 0.0297 | 1.0431 | 1.0415 | 1.0415 |
| **ALL15 refit** | 32,775 | 0.3692 | **0.6069** | 0.4559 | 0.2932 | 1.3831 | **1.3239** | 1.2781 |
| Baseline（symbol+source_tf） | 32,775 | 0.3692 | 0.5109 | 0.3746 | 0.0247 | 1.0162 | 1.0062 | 1.0380 |

DEV4 → NEW11 zero-shot 的 Lift@20 从 1.2501 到 1.2075，**衰减极小**。

### 3.2 NEW11 zero-shot 分品种（11/11 全部 ≥1.03）

| symbol | n | base | AUC | D10−D1 | Lift@20 | 提升 | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| P | 1,998 | 0.3954 | 0.6245 | 0.3900 | **1.4100** | +0.1621 | 0.2108 |
| AU | 2,558 | 0.3815 | 0.5929 | 0.2070 | 1.3309 | +0.1263 | 0.1563 |
| RU | 2,106 | 0.3718 | 0.5719 | 0.2322 | 1.2875 | +0.1069 | 0.1204 |
| AL | 2,416 | 0.3506 | 0.5845 | 0.2603 | 1.2435 | +0.0854 | 0.1397 |
| SC | 2,643 | 0.3693 | 0.5771 | 0.1925 | 1.2388 | +0.0882 | 0.1289 |
| I | 2,137 | 0.3107 | 0.5788 | 0.2150 | 1.2182 | +0.0678 | 0.1264 |
| TA | 1,616 | 0.3465 | 0.5601 | 0.2346 | 1.1846 | +0.0640 | 0.0991 |
| MA | 1,741 | 0.3900 | 0.6028 | 0.2941 | 1.1682 | +0.0656 | 0.1737 |
| NI | 2,529 | 0.3780 | 0.5752 | 0.2372 | 1.1136 | +0.0429 | 0.1263 |
| CF | 1,754 | 0.3700 | 0.5767 | 0.1534 | 1.0857 | +0.0317 | 0.1283 |
| SN | 2,379 | 0.3876 | 0.5256 | 0.0714 | 1.0462 | +0.0179 | 0.0431 |

macro：mean AUC 0.5791 / median 0.5771；median Lift@20 **1.2182**；
p25 1.1409 / p75 1.2655；positive **11/11**；meaningful(≥1.03) **11/11**。

### 3.3 NEW11 分折（4/4 为正）

| 折 | n | AUC | D10−D1 | Lift@20 | 提升 |
|---|---:|---:|---:|---:|---:|
| F1 | 5,726 | 0.5772 | 0.1693 | 1.2242 | +0.0825 |
| F2 | 5,478 | 0.5724 | 0.1533 | 1.1667 | +0.0601 |
| F3 | 5,724 | 0.5653 | 0.1658 | 1.1733 | +0.0653 |
| F4 | 6,949 | 0.5775 | 0.1755 | 1.2318 | +0.0856 |

### 3.4 NEW11 分 source_tf（3/3 为正）

| source_tf | n | AUC | D10−D1 | Lift@20 |
|---|---:|---:|---:|---:|
| 5m | 14,205 | 0.5718 | 0.1710 | 1.1960 |
| 15m | 6,664 | 0.5686 | 0.1394 | 1.1856 |
| 1h | 3,008 | **0.5919** | **0.2159** | **1.2842** |

### 3.5 ALL15 refit 分品种（15/15 全部 ≥1.03）

AG 1.3993 / AU 1.3924 / NI 1.3907 / P 1.3847 / RB 1.3654 / AL 1.3496 /
SC 1.3463 / CU 1.2899 / RU 1.2811 / CF 1.2628 / M 1.2605 / I 1.2482 /
MA 1.2049 / SN 1.2034 / TA 1.1222

macro：mean AUC 0.6064 / median 0.6037；median Lift@20 **1.2899**；
p25 1.2544 / p75 1.3750；positive **15/15**；meaningful **15/15**。

### 3.6 DEDUP

| | n | AUC | D10−D1 | Lift@20 |
|---|---:|---:|---:|---:|
| DEDUP NEW11 | 16,475 | 0.5644 | 0.1456 | 1.1815 |
| DEDUP ALL15 | 22,677 | 0.5969 | 0.2672 | 1.2939 |

信号在去除重复 OB 后保持。

### 3.7 Block bootstrap（trading_day，500 次）

| | D10−D1 95%CI | Top20 提升 95%CI |
|---|---|---|
| NEW11 zero-shot | [0.1304, 0.2050] | [0.0582, 0.0936] |
| ALL15 refit | [0.2600, 0.3228] | [0.1063, 0.1319] |

**均不含 0。**

### 3.8 symbol × fold 矩阵（NEW11）

**39/44 单元格 Lift@20 > 1（88.6%）**。按折中位数：F1 1.1995、F2 1.1731、
F3 1.2045、F4 1.2030。SN 是唯一偏弱品种（4 格中 3 格 < 1）。

---

## 4. Horizon 衰减（记录，非选择依据的复用）

| horizon | DEV4 AUC | NEW11 AUC | NEW11 D10−D1 | NEW11 Lift@20 |
|---|---:|---:|---:|---:|
| **12（primary）** | **0.5835** | **0.5730** | **0.1696** | **1.2075** |
| 24 | 0.5624 | 0.5556 | 0.1542 | 1.1032 |
| 48 | 0.5456 | 0.5274 | 0.0562 | 1.0398 |
| unbounded | 0.5296 | 0.5079 | 0.0256 | 1.0219 |

单调衰减。NEW11 从未参与 horizon discovery 仍完整重现该顺序。

---

## 5. 已排除的解释

| 备择解释 | 结论 |
|---|---|
| symbol base rate | 排除（NEW11 zero-shot，symbol 不进特征） |
| source_tf base rate | 排除（Baseline-ZS Lift@20 仅 1.0415） |
| 重复 OB 触发 | 排除（DEDUP 后 1.1815） |
| 单一品种驱动 | 排除（macro median ≈ pooled，11/11 品种为正） |
| 单一时间折 | 排除（4/4 折为正） |
| 无限等待 | 排除（unbounded 最弱，Lift 1.0219） |
| rollover 假象 | 排除（3/5/10 ATR 标签零变化） |
| 分布外塌方 | 排除（missing diff 0.0，\|z\|>5 比例 0.0） |

---

## 6. 冻结后纪律

1. primary label = **12 valid 5m bars**。
2. 24 / 48 / unbounded 仅作历史 sensitivity 保留，**不得用于结论**。
3. 103 维 feature contract 冻结，不加特征、不做特征选择。
4. Logistic 参数与 preprocessing 冻结，不调 C、不换模型族。
5. rolling folds / coverage 冻结。
6. **LC 保持 excluded**，不修改 quantile minimum 600。
7. **不再优化 Phase 1。** 输出继续称 Tradability Score，不称概率。
8. 引用 Phase 1 结果时一律使用本文件 §3 的 12bar 数值。

---

## 7. 下一步（不在本冻结范围内）

Phase 2 重新定义为 **买卖点识别**：

```text
OB trigger → Phase 1 Tradability Score → 筛选高质量事件
→ 等待市场给出新的价格信息 → Phase 2 Execution / Entry
```

Phase 2 不恢复 FOLLOW/FADE / 1.5R/2R/2.5R / 六动作框架。
M8 已证明 t=0 静态方向预测不可行；待研究的是
**t+1、t+2 观察到价格行为后能否识别方向**。

Phase 2 的标签与决策时点需单独设计，本冻结不包含。
