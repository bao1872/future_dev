# REVIEWER DECISION V1.1 — 合同修订记录

Base: `cd1a339ca6c4330f1d8ab33c329f690d4fa9f1c1`
不覆盖 L0 审计（`CANONICAL_SEMANTICS_AUDIT.md` 保持原样）。

## 修订总原则

> **"canonical 不存在"不等于该研究变量不能构造；**
> **真正不能接受的是为了结果临时发明带参数的语义。**

因此：
- **确定性、无参数、事前完全可知**的变量（前日高低点等）→ 允许构造，
  但必须标记 `RESEARCH_DERIVED_*`。
- **需要人为阈值**的变量（displacement、consolidation 边界）→ 本轮不构造。

---

## DECISION 1 — 4H：Environment 可以，Structure 不可以

现有 `aggregate_4h_from_1h` 继续使用，但**只能命名**：

    CLOCK_4H_ENVIRONMENT / env_4h

**严禁**称：canonical 4h structure / trend_struct_4h / 4h swing /
4h BOS / 4h MSS / 4h liquidity / 4h OB。

理由：Source Owner 已明确它是 `"4h aggregation (environment only)"`，
V1.1 尊重该边界。它只回答"当前处于什么粗粒度高周期市场环境"，
不冒充 SMC structural timeframe。

分层改为：

    4h  = context (environment)
    1h  = highest canonical SMC structural timeframe
    15m = intermediate / local structure
    5m  = interaction / confirmation

4H causal hard rule：只能使用在 `t0` 之前**已完整结束并可知**的 4h
environment bucket；禁止使用包含 t0 的未完成 bucket 最终 OHLC。
必须建立 `env_available_time` 并断言 `env_available_time <= t0`。

## DECISION 2 — Trend Causality Gate（进入 L1 前必须过）

`swing_bias_*` 作为 PRIMARY structural trend；
`internal_bias_*` 与 `dsa_direction_*` 保留为**独立 context**，不合并成 score。

必须验证 `trend_available_time <= interaction_time`；
confirmed pivot 必须用 `confirmed_index / available_time`，
**不能把 pivot_time 当 available_time**。
若 `swing_bias` 是 retrospective backfill → `TREND_CAUSALITY_GATE_FAIL`，停止。

字段：`trend_struct_5m/15m/1h`、`internal_bias_5m/15m/1h`、
`dsa_direction_5m/15m/1h`、`env_direction_4h`。
**不制造** `trend_struct_4h`。

## DECISION 3 — Liquidity Universe

### CANONICAL STRUCTURAL
`CONFIRMED_SWING_HIGH/LOW`、`CANONICAL_EQH/EQL`，source_tf ∈ {5m,15m,1h}。
使用 `available_time` 而非 origin/pivot time。**无 4h structural liquidity。**

### RESEARCH-DERIVED TIME LIQUIDITY（新增，无参数、事前可知）
`PREV_TRADING_DAY_HIGH/LOW`、`PREV_CONTIG_SESSION_HIGH/LOW`、
`PREV_TRADING_WEEK_HIGH/LOW`，标记 `RESEARCH_DERIVED_TIME_LIQUIDITY`。

命名用 `PREV_CONTIG_SESSION_*` 而非 `PREV_SESSION_*` —— 因为现有
`build_session_masks` 是基于**交易断口**的连续 segment，
不是人为定义的 Asian/London/NY session。

available_time 规则：prev day H/L 从 **D+1 trading_day 第一根 valid bar**
开始可用（不直接用 `source_last_time`）；prev segment 从 segment 结束后
的 next valid bar 开始；prev week 基于 `trading_day` 的
**ISO year + ISO week**（不能只用 week number），从下一 trading week
第一根 valid bar 开始。

### 继续排除
range high/low、consolidation boundary、opening-range、
hand-built S/R、4h structural liquidity —— 均需新分段或 threshold 语义。

## DECISION 4 — Displacement 从 V1.1 删除

审计已证明 `NO CANONICAL DISPLACEMENT`。本轮**不检验**
`Sweep→Reclaim→Displacement→MSS`，**不留全 NULL 字段假装检验过**。

V1.1 正式链条：

    LIQUIDITY EXISTS → TOUCH → PENETRATION
        → {RECLAIM | ACCEPTANCE}
        → REVERSAL MSS

若此链有结构，下一实验**单独预注册** `DISPLACEMENT SEMANTICS STUDY`。

## DECISION 5 — Acceptance 定义

penetration 方向的 canonical BOS 先于 reclaim 出现 → ACCEPTANCE。
BSL(+1) 需 bullish BOS；SSL(−1) 对称。竞态含 AMBIGUOUS / ROLL_CENSORED /
END_OF_DATA_CENSORED。事件 schema 适配 Source Owner，不发明事件名。

## DECISION 6 — Reclaim 后也用 competing structural events（重要修正）

原合同允许 reclaim 后无限等待 MSS —— 会让概率失去意义。
修正为 `POST_RECLAIM_PENDING` 竞态：

- **Event R**：`reversal_direction` 的 canonical CHoCH/MSS
  → `REVERSAL_MSS_CONFIRMED`
- **Event F**：`penetration_direction` 的 canonical BOS
  → `REJECTION_FAILED_REACCEPTED`
- 同时间戳两者皆有 → `AMBIGUOUS_SAME_TIMESTAMP`
- 其余 → `ROLL_CENSORED` / `END_OF_DATA_CENSORED`

roll censor 必须按真实 market path 插入 event ordering。

## DECISION 7 — Primary confirmation 用 5m

第一层（reclaim vs acceptance）与 post-reclaim 竞态**均只在 5m 解析**，
避免高周期确认速度造成时间语义混乱。5m `REVERSAL_MSS_CONFIRMED` 之后
**另外记录** next 15m / next 1h structural event 属于哪一类，
仍采用 first competing structural event，不固定 bar 数。

## DECISION 8 — 检验命题（不使用"主力"措辞）

> 当高周期趋势明确时，逆高周期趋势方向的 liquidity penetration
> 是否更容易被 reclaim，并随后产生恢复高周期趋势的 LTF MSS？

**PRIMARY HTF TREND = `trend_struct_1h`**；4h 仅作增强 context。

- Q1：单看 1h，`against-trend sweep` 是否有 trend-resumption 倾向？
- Q2：若 `env_direction_4h` 与 1h 同向，该效应是否进一步增强？

## DECISION 9 — 固定核心统计

A. `P(RECLAIM|pen)` vs `P(ACCEPTANCE|pen)`
B. `P(RECLAIM|against 1h)` vs `P(RECLAIM|with 1h)`（pp 差 + RR + bootstrap CI）
C. `P(REVERSAL_MSS_5M|reclaim, against 1h)` vs `(…|with 1h)`
D. `P(REVERSAL_MSS_5M aligns 1h | against 1h, reclaim)` ← 核心数字
E. 上述 A–D 在 `env4h == 1h` 与 `env4h != 1h` 下分别计算（**保留冲突组**）

## DECISION 10 — 不打 liquidity_score

保留 `liquidity_type / liquidity_scope / overlap_count / overlap_types`，
但**不建立总分**。第一轮直接对 5 类画像。

## DECISION 11 — Overlap 不 collapse

同价位多个 identity 全部保留；另建 `interaction_cluster_id`
（symbol + interaction bar + tick-normalized level）。
Primary = 每个 identity 独立；Sensitivity = 每 cluster 一条。
必须报告 cluster-dedup 后主效应是否保留。

## DECISION 12-16 — 输出与边界

- L1 输出：`liquidity_levels.parquet`、`interactions_all.parquet`、
  `interactions_primary.parquet`、`mtf_state_snapshot.parquet`、
  `trend_causality_audit.csv`、`env4h_causality_audit.csv`、
  `time_liquidity_audit.csv`。Primary = **FIRST VALID INTERACTION ONLY**。
- L2 分层：liquidity type / scope / symbol / F1-F4 / trend_struct_1h /
  env4h×1h / 1h×15m×5m。低样本只展示不解释。
- Bootstrap：trading_day block，500 次，报 pp 差 + 95%CI + RR，不以 p 值主导。
- **Displacement 必须显式写入研究缺口**，不得藏缺项。

## HARD GATES（进入 L1 前只剩两个）

- **GATE A**：5m/15m/1h structural trend causality → FAIL 则
  `RESEARCH_VALIDITY_BLOCKED`，STOP。
- **GATE B**：4h environment direction causality（必须来自
  last completed CLOCK_4H_ENVIRONMENT only）→ FAIL 则
  `4H_DIRECTION_GATE_FAIL`，STOP，**不偷偷降级**。
