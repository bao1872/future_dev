# SMC Multi-Timeframe Liquidity State Machine v1 — STOPPED at L0

**状态：`TF4H_SEMANTICS_GATE_FAIL`**
**分类：D. RESEARCH VALIDITY BLOCKED**

本轮**未进入 L1 / L2**，未构建流动性地图、未构建交互数据集、未运行状态机、
未产出任何转移概率。

---

## 做了什么

完成 **Phase L0 — Canonical Semantic Audit**（只读）。

产出：
- `research/liquidity_state_machine/CANONICAL_SEMANTICS_AUDIT.md`
- `research/liquidity_state_machine/canonical_audit.py`（可重跑）
- `research/liquidity_state_machine/canonical_probe_raw.csv`

## 为什么停在这里

合同第 3 节把 4h 定为 **Hard Gate**，并规定：

> 如果既没有 canonical 4h，也没有可治理的 HTF builder：
> `TF4H_SEMANTICS_GATE_FAIL`，立即停止。不要自己发明 4h convention。

审计结果是**两者都没有**：

1. **没有 canonical 4h 结构。** 唯一的 4h 入口
   `build_ob_candidate_universe_v3.aggregate_4h_from_1h` 注释写明
   *"4h aggregation (environment only)"*，且 canonical 构建器显式禁止
   4h 进入结构：

   ```python
   if any(r["source_tf"] == "4h" for r in cand_records):
       raise RuntimeError("4h candidate is forbidden")
   ```

   实测 canonical OB universe 的 `source_tf` 只有 `['15m', '1h', '5m']`。

2. **没有 session-aware HTF builder。** 现有三个 HTF 聚合器全部是
   时钟 / epoch 锚定：
   - `aggregate_15m`（15 分钟时钟格，要求恰好 3 根连续 5m）
   - `aggregate_1h_from_15m`（整点 epoch 桶）
   - `aggregate_4h_from_1h`（4 小时 epoch 桶
     `start_ns // FOUR_HOUR_NS` → UTC 00/04/08/12/16/20
     = 北京 08/12/16/20/00/04）

   第三个正是合同明令禁止的"自然时钟切中国期货夜盘"。

因此 IDE 不自行发明 4h，按合同停止。

## 已确认可用的 canonical 语义（供下一轮使用）

| concept | 可用 | 来源 |
|---|:--:|---|
| 5m / 15m / 1h bars | ✅ | `load_raw_5m`、`aggregate_15m`、`aggregate_1h_from_15m` |
| session / trading-day | ✅ | `trading_day`、`build_session_masks` |
| roll / discontinuity | ✅ | `discontinuity_flags`（`ROLL_GAP_ATR_THRESHOLD`） |
| confirmed pivots | ✅ | `current_pivots_at` / `confirmed_index` |
| BOS / CHoCH(MSS) | ✅ | `ob_trigger_snapshot` SMC events |
| OB lifecycle | ✅ | `OB_CREATED` / `OB_ENTERED` / `OB_MITIGATED` |
| **equal highs / lows** | ✅ | `smc["equal_highs_lows"]` + `confirmed_equal_levels_at` |
| structure / trend bias | ⚠️ 待确认因果 | `swing_bias_*` / `internal_bias_*` |
| DSA direction | ✅ | `dsa_direction_*` |
| **displacement** | ❌ | 全仓库无 canonical displacement |
| FVG / imbalance | ❌ | 无 canonical（仅 HF 论文脚本提及） |
| prev day / session / week H/L | ❌ | 无 canonical |
| range / consolidation boundary | ❌ | 无 canonical |
| **4h structure** | ❌ | 被 Source Owner 显式禁止 |

即：即便 4h 问题解决，**displacement、FVG、前日/前 session/前周高低点、
 consolidation 边界在 V1 也都没有 canonical 定义**。合同第 18/20/21 节要求
 acceptance 与 displacement 必须复用 canonical，因此这些缺口同样需要在
 L1 之前裁定。

## 需要 reviewer 先裁定的三个问题

### Q1 — 4h 语义（阻塞项）

任选其一：

1. **定义合法 4h**：例如"每个 trading session 内连续 4 根 valid 1h bar，
   不跨 session 断点"，并显式写出 anchor / `available_time` / causal close。
2. **移除 4h**，HTF 栈降级为 1h / 15m / 5m，研究改称 "MTF (1h) stack"。
3. **仅把现有 epoch-4h 当 environment 上下文**，不参与结构确认与流动性定义。

### Q2 — acceptance / displacement 的 canonical 来源

合同要求 acceptance 与 displacement 必须来自 Source Owner，但：
- 没有任何 canonical displacement；
- acceptance 只能用 BOS/CHoCH 近似。

需要确认：是否接受"penetration 方向上的 canonical BOS 先于 reclaim 出现"
作为 acceptance 的 V1 定义，以及 displacement 是否**留空**而不是新定义。

### Q3 — liquidity pool 实际能建哪些

当前只有两类能因果、无阈值地构建：
- **confirmed swing high/low**（5m/15m/1h，+4h 若 Q1 解决）
- **canonical equal highs/lows**（已有 `equal_highs_lows`）

而 prev day / prev session / prev week H/L、range/consolidation extreme
**都没有 canonical**，按合同第 9 节应记 `MISSING_CANONICAL_*` 而不是临时创造。
需确认是否接受 V1 只跑 swing + EQH/EQL 这两类。

---

## 未做的事（明确声明）

- 未构建 `liquidity_levels.parquet`
- 未构建 `interactions_primary.parquet`
- 未运行任何状态转移、未产出任何条件概率
- 未做 bootstrap、未做分品种/分折画像
- **未训练任何模型**（合同禁止 ML / LightGBM / Logistic / NN / RL）
- **未计算任何 Sharpe / PnL / entry / stop / target 优化**

```
DELIVERY_STAGE = NOT_STARTED
```
