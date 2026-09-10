# CANONICAL SEMANTICS AUDIT (Phase L0)

只读审计。本文件由 `research/liquidity_state_machine/canonical_audit.py` 生成。

## 1. 概念可用性探测

| concept | exists | n_hits | examples |
|---|---|---|---|
| 5m bars | True | 11 | build_ob_candidate_universe_v3.py:81; download_v3r_5m.py:15; download_v3r_5m.py:16; download_v3r_5m.py:50 |
| 15m bars | True | 1 | build_pytdx_panel.py:105 |
| 1h bars | True | 1 | ob_trigger_snapshot.py:108 |
| 4h bars | True | 1 | build_ob_candidate_universe_v3.py:410 |
| session / trading-day mapping | True | 178 | analyze_ob_candidate_v3_phase1_1.py:219; analyze_ob_candidate_v3_phase1_1.py:230; analyze_ob_candidate_v3_phase1_1.py:231; analyze_ob_candidate_v3_phase1_1.py:482 |
| continuous-futures roll / discontinuity | True | 12 | build_rl_exit_trajectory_v1.py:57; build_rl_exit_trajectory_v1.py:107; m2_nondeep_temporal_v1.py:37; m2_nondeep_temporal_v1.py:89 |
| confirmed swings / pivots | True | 66 | audit_ob_candidate_v3_posthoc.py:123; audit_ob_candidate_v3_posthoc.py:123; audit_ob_candidate_v3_posthoc.py:389; audit_ob_candidate_v3_posthoc.py:412 |
| BOS | True | 12 | charting.py:165; charting.py:201; environment_atlas_spec.py:18; ob_rl_model_view_v0_spec.py:196 |
| CHoCH / MSS | True | 13 | charting.py:165; charting.py:201; environment_atlas_spec.py:18; ob_rl_model_view_v0_spec.py:196 |
| structure / trend direction | True | 66 | analyze_ob_candidate_v3_phase1.py:432; analyze_ob_candidate_v3_phase1.py:432; analyze_ob_candidate_v3_phase1.py:893; analyze_ob_candidate_v3_phase1.py:1016 |
| DSA direction | True | 69 | analyze_ob_candidate_v3_phase1.py:432; analyze_ob_candidate_v3_phase1.py:905; analyze_ob_candidate_v3_phase1.py:1022; analyze_ob_candidate_v3_phase1_1.py:495 |
| displacement | False | 0 |  |
| FVG / imbalance | True | 2 | run_paper_a1_hf_predictability.py:1107; run_paper_a1_hf_predictability.py:1110 |
| OB lifecycle | True | 59 | analyze_ob_candidate_v3_phase1.py:735; audit_ob_candidate_v3_posthoc.py:152; audit_ob_candidate_v3_posthoc.py:166; audit_ob_candidate_v3_posthoc.py:424 |
| equal highs / lows | True | 4 | charting.py:327; ob_trigger_snapshot.py:403; ob_trigger_snapshot.py:422; ob_trigger_snapshot.py:820 |
| previous day H/L | False | 0 |  |
| previous session H/L | True | 1 | run_strategy_s2_session_transition.py:352 |
| previous week H/L | False | 0 |  |
| range / consolidation boundary | False | 0 |  |

## 2. 4H GATE（Hard Gate）

用户合同明确要求 5m / 15m / 1h / 4h 四周期。因此 4h 不能偷偷缺失。

**判定：`TF4H_SEMANTICS_GATE_FAIL`**

事实（逐条来自 Source Owner 代码，不是推测）：

1. 存在 `build_ob_candidate_universe_v3.aggregate_4h_from_1h`，但其注释明确写着 **“4h aggregation (environment only)”**。
2. 它的分桶方式是**epoch 锚定的自然时钟 4 小时桶**：
   `x["_bucket"] = (start_ns // FOUR_HOUR_NS) * FOUR_HOUR_NS`，
   等价于 `resample("4H")`，切出的边界是 UTC 00/04/08/12/16/20，
   对应北京时间 08/12/16/20/00/04 —— 正好是合同禁止的“自然时钟切中国期货夜盘”。
3. 更关键的是，canonical 构建器**显式禁止 4h 结构**：
   ```python
   if any(r["source_tf"] == "4h" for r in cand_records):
       raise RuntimeError("4h candidate is forbidden")
   ```
   以及全局校验：
   ```python
   if (candidates["source_tf"] == "4h").any():
       raise RuntimeError("4h candidate count > 0")
   ```
4. 实测 canonical OB universe 的 `source_tf` 取值只有 `['15m', '1h', '5m']`，**没有任何 4h 结构事件**。

因此：

- **不存在 canonical 4h 结构**（只有 environment 用途的epoch 聚合，且被禁止用于 candidate）。
- **不存在项目统一的 session-aware HTF bar builder**：现有 HTF 构建器只有 `aggregate_15m`（15 分钟时钟格）、`aggregate_1h_from_15m`（整点 epoch 桶）、`aggregate_4h_from_1h`（4 小时 epoch 桶），全部是时钟/epoch 锚定，没有基于 valid trading bar 计数或session 边界的 HTF 构建器。

按合同第 3 节：**既没有 canonical 4h，也没有可治理的 HTF builder → `TF4H_SEMANTICS_GATE_FAIL`，立即停止，不自己发明 4h convention。**

本轮因此**不进入 L1 / L2**，不构建流动性地图与状态机。

### 若要解除该 gate，需要 reviewer 先裁定以下之一：

1. **定义 4h 的合法语义**：采用“每个交易 session 内的第 N 个1h bar 组合”或“连续 4 根 valid 1h bar 且不跨 session 断点”，并显式写出 anchor / available_time / causal close 规则；或
2. **正式把 4h 从研究范围移除**，把 HTF 栈降级为 1h / 15m / 5m，并在本研究中改称 “MTF (1h) stack”；或
3. 复用现有 epoch-anchored 4h 但**只作为 environment 上下文**，明确声明它不参与结构确认与流动性定义。

这三种选择会改变 HTF_STACK 的语义，必须由 reviewer 决定，IDE 不自行选择。
