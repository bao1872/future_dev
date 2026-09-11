# SMC Fixed Execution Baseline v1.0

**base**: `2006fcb` (v1.5) &nbsp; **脚本**: `run_fixed_execution_baseline_v1.py`
**第一次真实执行层实验**：把通过 label-availability + full-live-universe 审计的
方向 selector 转为严格事前可执行的固定规则，测真实 win rate / payoff / expectancy。

Primary = **S1 Continuation-only**（direction = side，risk = 1 ATR）
Secondary = **S2 Reversal-only**（diagnostic，不参与 PASS/FAIL）

---

## 0. P0 v1.5 文档语义修正

- `gate.per_wf_selection = 0.05` **移出 Gate** → 改名 `coverage_reference`。
  原因：contact 不是执行计量单位；collapse 后应以 unique signal / trade frequency 报告。
- v1.5 `nearest_exante_target` 只验证 **active target availability**，
  **不是冻结 execution target 语义**（TOUCH_ONLY 下 attacked level 仍 active）。

---

## 1. 执行合同（冻结）

| 环节 | 定义 |
|---|---|
| collapse | `(symbol, decision_time, setup)`；方向冲突 → abstain |
| attack_boundary | 同组 contacted liquidity 沿 direction 的**最外沿** |
| CONT target | decision-time active **AND** ahead_of_entry **AND** beyond_attack |
| REV target | decision-time active **AND** ahead_of_entry（离开 attacked level） |
| entry | **下一根有效 5m bar open**（禁止 contact close 成交） |
| stop | `decision_close - direction*atr0`（冻结 risk=1，**不重新 anchor**） |
| path | entry_bar .. 第一个 discontinuity 前（不跨 roll） |
| same-bar | Primary **STOP_FIRST**，Secondary TARGET_FIRST |
| position | 一品种同时最多一个 position |

---

## 2. Execution funnel

| wf | setup | raw contacts | collapsed | conflict | after target | executed | skip_open |
|---|---|---:|---:|---:|---:|---:|---:|
| WF1 | CONT | 1159 | 649 | 0 | 644 | 614 | 18 |
| WF1 | REV | 749 | 404 | 0 | 404 | 364 | 12 |
| WF2 | CONT | 1813 | 927 | 0 | 917 | 856 | 35 |
| WF2 | REV | 1773 | 768 | 0 | 768 | 725 | 15 |
| WF3 | CONT | 1720 | 895 | 0 | 891 | 848 | 29 |
| WF3 | REV | 1266 | 604 | 0 | 604 | 572 | 9 |

Coverage（P18 新单位，不再用 contact selection≥5% 当 gate）：

| wf | setup | raw_contact_sel | collapsed_signal_rate | executed_trade_rate |
|---|---|---:|---:|---:|
| WF1 | CONT | 0.0447 | 0.0251 | 0.0237 |
| WF1 | REV | 0.0289 | 0.0156 | 0.0141 |
| WF2 | CONT | 0.0742 | 0.0379 | 0.035 |
| WF2 | REV | 0.0725 | 0.0314 | 0.0297 |
| WF3 | CONT | 0.0721 | 0.0375 | 0.0355 |
| WF3 | REV | 0.053 | 0.0253 | 0.024 |

---

## 3. P12 Primary Continuation 执行指标

| wf | trades | /day | target_hit | stop_hit | roll | same_bar | med_tgt_R | avg_win_R | avg_loss_R | payoff | **gross_exp_R** | PF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WF1 | 614 | 6.0196 | 0.5065 | 0.4935 | 0.0 | 0.0212 | 0.8889 | 0.9525 | -1.0629 | 0.8962 | -0.042 | 0.9199 |
| WF2 | 856 | 8.3922 | 0.5666 | 0.4334 | 0.0 | 0.0187 | 0.7487 | 0.8291 | -1.0255 | 0.8085 | 0.0253 | 1.0569 |
| WF3 | 848 | 8.396 | 0.5401 | 0.4575 | 0.0 | 0.0189 | 0.75 | 0.8349 | -1.065 | 0.7839 | -0.0376 | 0.9229 |

> `target_hit_rate` 才是真正交易胜率；此前的 65% `actionable` **不是 win rate**。

---

## 4. P13 Continuation vs Reversal

| wf | metric | Continuation | Reversal |
|---|---|---:|---:|
| WF1 | n_executed_trades | 614.0 | 364.0 |
| WF1 | target_hit_rate | 0.5065 | 0.6126 |
| WF1 | median_target_R_exec | 0.8889 | 0.6522 |
| WF1 | mean_win_R | 0.9525 | 0.7162 |
| WF1 | mean_loss_R | -1.0629 | -1.0203 |
| WF1 | payoff_ratio | 0.8962 | 0.7019 |
| WF1 | gross_expectancy_R | -0.042 | 0.0435 |
| WF1 | profit_factor_R | 0.9199 | 1.1101 |
| WF2 | n_executed_trades | 856.0 | 725.0 |
| WF2 | target_hit_rate | 0.5666 | 0.6207 |
| WF2 | median_target_R_exec | 0.7487 | 0.6481 |
| WF2 | mean_win_R | 0.8291 | 0.6955 |
| WF2 | mean_loss_R | -1.0255 | -1.0112 |
| WF2 | payoff_ratio | 0.8085 | 0.6878 |
| WF2 | gross_expectancy_R | 0.0253 | 0.0481 |
| WF2 | profit_factor_R | 1.0569 | 1.1255 |
| WF3 | n_executed_trades | 848.0 | 572.0 |
| WF3 | target_hit_rate | 0.5401 | 0.6101 |
| WF3 | median_target_R_exec | 0.75 | 0.6885 |
| WF3 | mean_win_R | 0.8349 | 0.6562 |
| WF3 | mean_loss_R | -1.065 | -1.035 |
| WF3 | payoff_ratio | 0.7839 | 0.6339 |
| WF3 | gross_expectancy_R | -0.0376 | -0.0032 |
| WF3 | profit_factor_R | 0.9229 | 0.9921 |

Reversal 仅 `SECONDARY_DIAGNOSTIC`，不得据此加入 Primary。

---

## 5. P17 Target semantics audit（OLD vs NEW）

| wf | definition | no_target_rate | median ATR | p25 | p75 | p90 | target==attack |
|---|---|---:|---:|---:|---:|---:|---:|
| WF1 | NEW_BEYOND_ATTACK | 0.0077 | 0.8696 | 0.4762 | 1.7102 | 2.7035 | 0.0 |
| WF1 | OLD_ENTRY_NEAREST | 0.0 | 0.3333 | 0.2381 | 0.5 | 0.625 | 0.6333 |
| WF2 | NEW_BEYOND_ATTACK | 0.0108 | 0.7317 | 0.4327 | 1.5 | 2.5 | 0.0 |
| WF2 | OLD_ENTRY_NEAREST | 0.0 | 0.3846 | 0.2273 | 0.5556 | 0.7223 | 0.4995 |
| WF3 | NEW_BEYOND_ATTACK | 0.0045 | 0.7143 | 0.4545 | 1.4583 | 3.0556 | 0.0 |
| WF3 | OLD_ENTRY_NEAREST | 0.0 | 0.3913 | 0.2381 | 0.5827 | 0.7348 | 0.4626 |

**HARD ASSERTION**：`NEW_BEYOND_ATTACK` 的 `target_equals_attack_rate` 必须为 **0**
（实测 `0.0`）。
OLD 定义下该比例很高，正是 v1.5 median target≈0.32–0.40 ATR 的原因。

---

## 6. P14 Cost metadata

```json
{
  "scanned": [
    "tick_size",
    "contract_multiplier",
    "commission",
    "exchange_fee",
    "broker_fee",
    "slippage"
  ],
  "canonical_table_found": false,
  "evidence": [
    "research/exports/strategy_s1/s1_config.json: 'simple gross returns, no fees, no slippage'",
    "research/analysis_results/phase1_tradability_v1/symbol_universe.csv: no tick_size/multiplier/commission"
  ],
  "REALISTIC_NET_PNL": "UNAVAILABLE_COST_METADATA",
  "rule": "禁止凭记忆填写手续费/滑点；只报 GROSS R 与 break-even cost",
  "break_even_round_trip_cost_R": -0.0156,
  "interpretation": "每笔交易总往返成本（R）超过该值则 gross edge 消失；不得声称 net profitable。",
  "break_even_note": "gross expectancy<=0 → break-even cost<=0：在扣除任何成本之前 edge 已为负，没有任何成本承受空间。"
}
```

只报 **GROSS R**；`break_even_round_trip_cost_R = gross_expectancy_R` 表示
每笔往返成本超过该值则 gross edge 消失。**不得声称 net profitable。**

---

## 7. P15 Gate

```json
{
  "GROSS_EXECUTION_EDGE_PRESENT": false,
  "per_wf_gross_expectancy_R": [
    -0.042,
    0.0253,
    -0.0376
  ],
  "pooled_gross_expectancy_R": -0.0156
}
```

**GROSS_EXECUTION_EDGE_PRESENT = False**
（要求 3/3 WF expectancy_R>0 且 pooled>0）。若 FALSE → 停止进一步成本/参数优化，
**不得为过 gate 改 target/stop/threshold**。

---

## 8. P16 Bootstrap（entry trading-day block）

| scope | n_boot | p2.5 | p50 | p97.5 |
|---|---:|---:|---:|---:|
| - | STOP_NO_BOOTSTRAP | | | |

---

## 9. 按 symbol（P13 secondary，不筛品种）

| symbol | trades | hit_rate | avg_win_R | avg_loss_R | gross_exp_R |
|---|---:|---:|---:|---:|---:|
| AG | 98 | 0.6122 | 0.754 | -1.0133 | 0.0687 |
| AL | 192 | 0.4792 | 0.9095 | -1.0685 | -0.1207 |
| AU | 139 | 0.6691 | 0.6381 | -0.987 | 0.1003 |
| CF | 163 | 0.4172 | 0.8692 | -1.0093 | -0.2195 |
| CU | 147 | 0.5306 | 0.9101 | -1.2161 | -0.0879 |
| I | 57 | 0.4737 | 0.946 | -1.1095 | -0.1359 |
| M | 162 | 0.5494 | 1.0283 | -1.0 | 0.1143 |
| MA | 168 | 0.5238 | 0.8843 | -1.0108 | -0.0181 |
| NI | 139 | 0.5899 | 0.73 | -1.1321 | -0.0336 |
| P | 198 | 0.5909 | 0.9102 | -1.0172 | 0.1217 |
| RB | 248 | 0.4919 | 1.012 | -1.0253 | -0.0231 |
| RU | 177 | 0.5593 | 1.0115 | -1.0264 | 0.1135 |
| SC | 133 | 0.4887 | 0.9311 | -1.1627 | -0.1394 |
| SN | 119 | 0.6807 | 0.5686 | -1.0439 | 0.0537 |
| TA | 178 | 0.5225 | 0.7629 | -1.0026 | -0.0802 |

---

## 10. P21 完成 / STOP

代码 + 测试 + 运行 + 报告 + commit + push 后 STOP。
禁止据本轮结果优化 target/stop、筛 symbol、改 threshold、把 Reversal 加入主策略。
