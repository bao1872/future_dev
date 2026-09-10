# Liquidity Specificity Placebo Test v2.0

Base: `18b300db96f316eae76a62eb56637b9046c643d6`

```
NO ML / NO PnL / NO displacement / NO fixed horizon / NO outcome-driven tuning
```

**状态：`MATCHING_GATE_FAIL` — 特异性检验未完成，STOP**

P0（资格修正 + post-reclaim 修复）与 P1（placebo 生成）**已完成并通过内部一致性检查**；
P2 的 coverage gate **未通过**，因此按合同 **未计算任何 Stage-2 outcome**
（TRUE 与 PLACEBO 的 later-reclaim 比较**未执行**）。

---

## P0.1 — Activation eligibility

| activation_state | n |
|---|---:|
| **VALID_AHEAD** | **80,289** |
| AT_LEVEL | 868 |
| STALE_ALREADY_BEYOND | 0 |

判定使用 activation 前最后一根已完成 5m close（`c_ref`）：
BSL 要求 `c_ref < level`；SSL 要求 `c_ref > level`；且 `d0_R > 0`。
**868 个 AT_LEVEL 与所有已越过的 level 被排除**，不进入 primary universe。

## P0.3 — Interaction path

| interaction_path | n |
|---|---:|
| **CONTINUOUS_CROSS（primary）** | **64,290** |
| GAP_CROSS（单独画像，不删除） | 11,889 |

GAP_CROSS 定义：prior close 在旧侧，而 interaction bar 的 open 已在新侧
（session gap 直接跳过 level）。

## P0.4 — Fresh TRUE anchor（VALID_AHEAD + CONTINUOUS_CROSS）

```
FRESH PENETRATION = 51,424
  SAME_BAR_RECLAIM  20,483  (39.83%)
  CLOSE_BEYOND      26,543  (51.62%)
  CLOSE_AT_LEVEL     4,398  ( 8.55%)

Stage-2 TRUE pool = 26,498

P(LATER_RECLAIM         | CLOSE_BEYOND) = 0.7624  (20,202)
P(STRUCTURAL_ACCEPTANCE | CLOSE_BEYOND) = 0.2372  ( 6,285)
```

**与 v1.2 的 69.55% 相比，新的 fresh anchor 升至 76.24%。**
原因：排除了 activation 时已失效/已在 level 另一侧的 level，
以及排除了 gap 直接跳过 level 的样本。

by symbol：**15/15 > 0.5**，macro median **0.7635**，
范围 0.7185（AG）– 0.7950（P）。

## P0.5 — Post-reclaim competing events 修复

v1.2 取"第一条任意 structure event"，无关事件即落成 `OTHER`（5,642 例）。
本轮改为**扫描至第一个相关 competing event**：

| post_reclaim_state | n |
|---|---:|
| REJECTION_FAILED_REACCEPTED | 12,229 |
| REVERSAL_MSS_CONFIRMED | 10,694 |
| NO_RELEVANT_EVENT_BEFORE_CENSOR | 12 |

`OTHER` 已消失。**该组数字仍为 secondary，且为 provisional。**

---

## P1 — Placebo generation（distance-permuted）

- 3 个预注册 replica：`20260910 / 20260911 / 20260912`
- **240,133** 个 placebo level
- 置换 strata：`symbol × liquidity_type × side`
- derangement：`donor ≠ recipient`
- `pseudo_price = C0 + side × donor_d0_R × R0`
  （只使用 activation 时刻的 `C0` 与 `R0`，**不含任何未来信息**）

每个 replica 的 placebo interaction pool：

| replica | pool |
|---|---:|
| 20260910 | 30,227 |
| 20260911 | 29,901 |
| 20260912 | 30,140 |

`d0_R` 分布在置换 strata 内保持不变（置换即保持边缘分布）。

> placebo 不声称该价位"绝对没有流动性"；它只是**没有当前 detector 所赋予的
> 显著性**。

---

## P2 — MATCHING_GATE_FAIL

### Coverage（要求 overall ≥ 85%，每个 n≥500 的 type ≥ 75%）

| | replica 1 | replica 2 | replica 3 |
|---|---:|---:|---:|
| **overall** | **0.4771** | **0.4792** | **0.4780** |

per liquidity_type（replica 1）：

| liquidity_type | n | matched | coverage |
|---|---:|---:|---:|
| CANONICAL_EQH | 1,320 | 140 | **0.1061** |
| CANONICAL_EQL | 1,166 | 128 | **0.1098** |
| CONFIRMED_SWING_HIGH | 2,048 | 442 | **0.2158** |
| CONFIRMED_SWING_LOW | 1,700 | 277 | **0.1629** |
| PREV_CONTIG_SESSION_HIGH | 7,915 | 5,430 | **0.6860** |
| PREV_CONTIG_SESSION_LOW | 7,493 | 5,073 | **0.6770** |
| PREV_TRADING_DAY_HIGH | 2,118 | 604 | **0.2852** |
| PREV_TRADING_DAY_LOW | 1,911 | 529 | **0.2768** |
| PREV_TRADING_WEEK_HIGH | 464 | 12 | LOW_N |
| PREV_TRADING_WEEK_LOW | 363 | 6 | LOW_N |

**全部未达标。**

### 失败诊断

Exact-match 条件为
`symbol × liquidity_type × side × sweep_vs_1h × session_type × time_bucket_30m`，
理论上约 15 × 10 × 2 × 3 × ~6 × ~8 ≈ **43,200** 个单元，
而每个 replica 的 placebo control pool 只有约 **30,000** 条
→ 绝大多数单元为空或不足 K=3，导致覆盖率约 48%。

**这是匹配设计问题，不是 placebo 构造问题**：placebo 生成、距离置换、
derangement、资格判定均正常。

### 按合同处置

合同明确：

> 如果 FAIL：`MATCHING_GATE_FAIL`，STOP。本轮不调 K/caliper。

因此**未**采取任何以下动作：
未放宽 caliper、未改 K、未合并 session bucket、未去掉 sweep_vs_1h
exact-match、未扩大日期窗口、未改距离度量。

**Stage-2 outcome 保持 OUTCOME LOCKED，未计算。**

---

## P3 / P4 — 未执行

- TRUE vs PLACEBO later-reclaim 差值：**未计算**
- bootstrap CI：**未计算**
- 3 replica 符号一致性：**未计算**
- by symbol / type / scope / fold / trend：**未计算**
- Stage-1 same-bar specificity：**未计算**

---

## 结论

```
MATCHING_GATE_FAIL

P0 / P1 completed and internally consistent.
Stage-2 specificity test NOT executed because matched-control
coverage (≈48%) fell below the pre-registered gate (≥85% overall,
≥75% per liquidity type). Outcome remains LOCKED.
No tuning of K / caliper / exact-match constraints was performed.
```

### 已可确认（不依赖 placebo）

在**事前定义、activation 时仍位于价格前方、且被连续穿越**的水平位置上，
5m 收盘真正站到 level 外侧后，约 **76.2%** 会在后续重新收回该 level，
且发生在同向新 BOS 之前。15/15 品种一致。

按 reviewer 的措辞修正，这应称为

> **horizontal-level re-cross before structural continuation phenomenon**

**不能**称为 "structural liquidity reclaim phenomenon"——
`LATER_RECLAIM`（close 再越回 level）与 `STRUCTURAL_ACCEPTANCE`
（canonical BOS）仍是两类不同的结构事件。

### 未回答的核心问题

> 这个 76.2% 是显著流动性位置特有，还是普通水平线也如此？

**仍然未检验。** 需要 reviewer 重新设计匹配（例如：去掉 session/time bucket
exact-match 改为数值协变量、或先按 liquidity_type 各自匹配、
或扩大 placebo pool 到每 replica 更多 donor），再进入 P3。

## 已知限制

1. 未做多重检验校正。
2. `sweep_vs_1h` / `env_align_1h` 分层在 TRUE anchor 上未输出（列未持久化）。
3. post-reclaim 仅对 SAME_BAR_RECLAIM 计算。
4. tests 未覆盖（本轮未新增 pytest）。
5. placebo 的 `d0_R` 分布保持，但具体价位可能偶然落在真实 level 附近
   （已用 `|pseudo - true| < 1e-12` 排除完全相同的情形）。
