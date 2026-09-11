# SMC Fixed Execution Baseline v1.0.1 — Integrity Repair + Geometry Attribution

**base**: `4c37cb8` (v1.0) &nbsp; **脚本**: `run_fixed_execution_integrity_v1_0_1.py`

> v1.0 状态：**`PROVISIONAL_PENDING_OOS_PATH_AUDIT`**。
> v1.0 的持仓路径没有截断 `OOS_START=2026-09-07`；本轮修复 + 审计。

本轮 **不优化** target / stop / risk / threshold / symbol。

---

## 1. P3 OOS path audit

| wf | setup | old_exec | old_paths_touch_oos | old_exit>=OOS | new_exec | new_exit>=OOS |
|---|---|---:|---:|---:|---:|---:|
| WF1 | CONT | 614 | 0 | 0 | 614 | 0 |
| WF1 | REV | 364 | 0 | 0 | 364 | 0 |
| WF2 | CONT | 856 | 0 | 0 | 856 | 0 |
| WF2 | REV | 725 | 0 | 0 | 725 | 0 |
| WF3 | CONT | 848 | 0 | 0 | 848 | 0 |
| WF3 | REV | 572 | 0 | 0 | 572 | 0 |

**状态：`OOS_PATH_BUG_NO_NUMERICAL_IMPACT`**（old=0，
new=0，new 必须为 0，HARD ASSERTION PASS）。

---

## 2. P6 Repaired execution（唯一变化 = OOS 硬截）

### Continuation（Primary）

| wf | trades | hit | stop | med_tgt_R | avg_win | avg_loss | payoff | **exp_R** | PF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WF1 | 614 | 0.5065 | 0.4935 | 0.8889 | 0.9525 | -1.0629 | 0.8962 | -0.042 | 0.9199 |
| WF2 | 856 | 0.5666 | 0.4334 | 0.7487 | 0.8291 | -1.0255 | 0.8085 | 0.0253 | 1.0569 |
| WF3 | 848 | 0.5401 | 0.4575 | 0.75 | 0.8349 | -1.065 | 0.7839 | -0.0376 | 0.9229 |

### Reversal（SECONDARY_DIAGNOSTIC）

| wf | trades | hit | med_tgt_R | payoff | exp_R |
|---|---:|---:|---:|---:|---:|
| WF1 | 364 | 0.6126 | 0.6522 | 0.7019 | 0.0435 |
| WF2 | 725 | 0.6207 | 0.6481 | 0.6878 | 0.0481 |
| WF3 | 572 | 0.6101 | 0.6885 | 0.6339 | -0.0032 |

---

## 3. P4 All-contact attack boundary audit

| wf | signals | same_boundary_rate | all_further_rate | median_gap | p75 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| WF1 | 649 | 0.9291 | 0.0709 | 0.0 | 0.0 | 0.0 |
| WF2 | 927 | 0.9666 | 0.0334 | 0.0 | 0.0 | 0.0 |
| WF3 | 895 | 0.962 | 0.038 | 0.0 | 0.0 | -0.0 |

HARD ASSERTION：`boundary_gap_ATR >= 0`（all-contact boundary 必不比 selected 更近）。

---

## 4. P5 Target boundary semantics（descriptive，不改 Primary）

| wf | definition | no_target | median | p25 | p75 | p90 |
|---|---|---:|---:|---:|---:|---:|
| WF1 | TARGET_SELECTED_BOUNDARY | 0.0077 | 0.8696 | 0.4762 | 1.7102 | 2.7035 |
| WF1 | TARGET_ALL_ATTACKED_BOUNDARY | 0.0077 | 0.9498 | 0.5263 | 1.8421 | 2.8976 |
| WF1 | OLD_ENTRY_NEAREST | 0.0 | 0.3333 | 0.2381 | 0.5 | 0.625 |
| WF2 | TARGET_SELECTED_BOUNDARY | 0.0108 | 0.7317 | 0.4327 | 1.5 | 2.5 |
| WF2 | TARGET_ALL_ATTACKED_BOUNDARY | 0.0108 | 0.75 | 0.4545 | 1.5909 | 2.8456 |
| WF2 | OLD_ENTRY_NEAREST | 0.0 | 0.3846 | 0.2273 | 0.5556 | 0.7223 |
| WF3 | TARGET_SELECTED_BOUNDARY | 0.0045 | 0.7143 | 0.4545 | 1.4583 | 3.0556 |
| WF3 | TARGET_ALL_ATTACKED_BOUNDARY | 0.0045 | 0.7143 | 0.464 | 1.5625 | 3.125 |
| WF3 | OLD_ENTRY_NEAREST | 0.0 | 0.3913 | 0.2381 | 0.5827 | 0.7348 |

---

## 5. P7 Geometry buckets（固定经济分桶，非 quantile）

| wf | setup | bucket | n | share | hit | avg_win | avg_loss | payoff | exp_R |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| WF1 | CONT | <0.50 | 150 | 0.2443 | 0.7533 | 0.2986 | -1.0 | 0.2986 | -0.0217 |
| WF1 | CONT | 0.50-0.75 | 109 | 0.1775 | 0.6422 | 0.6163 | -1.0356 | 0.5951 | 0.0252 |
| WF1 | CONT | 0.75-1.00 | 76 | 0.1238 | 0.5263 | 0.8684 | -1.0 | 0.8684 | -0.0166 |
| WF1 | CONT | 1.00-1.50 | 89 | 0.145 | 0.3933 | 1.2312 | -1.0185 | 1.2088 | -0.1338 |
| WF1 | CONT | >=1.50 | 190 | 0.3094 | 0.2789 | 2.6702 | -1.1216 | 2.3808 | -0.0639 |
| WF1 | REV | <0.50 | 127 | 0.3489 | 0.7638 | 0.2937 | -1.0 | 0.2937 | -0.0119 |
| WF1 | REV | 0.50-0.75 | 75 | 0.206 | 0.6533 | 0.6137 | -1.0 | 0.6137 | 0.0543 |
| WF1 | REV | 0.75-1.00 | 49 | 0.1346 | 0.5714 | 0.8587 | -1.0 | 0.8587 | 0.0621 |
| WF1 | REV | 1.00-1.50 | 53 | 0.1456 | 0.566 | 1.1511 | -1.0 | 1.1511 | 0.2176 |
| WF1 | REV | >=1.50 | 60 | 0.1648 | 0.3167 | 2.2407 | -1.0699 | 2.0943 | -0.0215 |
| WF2 | CONT | <0.50 | 258 | 0.3014 | 0.7403 | 0.307 | -1.0078 | 0.3046 | -0.0345 |
| WF2 | CONT | 0.50-0.75 | 170 | 0.1986 | 0.6706 | 0.6218 | -1.0015 | 0.6208 | 0.087 |
| WF2 | CONT | 0.75-1.00 | 110 | 0.1285 | 0.6364 | 0.8632 | -1.0079 | 0.8564 | 0.1828 |
| WF2 | CONT | 1.00-1.50 | 105 | 0.1227 | 0.4286 | 1.2331 | -1.0877 | 1.1337 | -0.0931 |
| WF2 | CONT | >=1.50 | 213 | 0.2488 | 0.3052 | 2.4107 | -1.0222 | 2.3583 | 0.0254 |
| WF2 | REV | <0.50 | 250 | 0.3448 | 0.744 | 0.3042 | -1.0 | 0.3042 | -0.0297 |
| WF2 | REV | 0.50-0.75 | 167 | 0.2303 | 0.6946 | 0.5974 | -1.0603 | 0.5634 | 0.0912 |
| WF2 | REV | 0.75-1.00 | 111 | 0.1531 | 0.5225 | 0.858 | -1.0 | 0.858 | -0.0292 |
| WF2 | REV | 1.00-1.50 | 140 | 0.1931 | 0.4286 | 1.209 | -1.0 | 1.209 | -0.0533 |
| WF2 | REV | >=1.50 | 57 | 0.0786 | 0.5263 | 2.1595 | -1.0 | 2.1595 | 0.6629 |
| WF3 | CONT | <0.50 | 239 | 0.2818 | 0.7573 | 0.2959 | -1.1158 | 0.2652 | -0.042 |
| WF3 | CONT | 0.50-0.75 | 183 | 0.2158 | 0.6448 | 0.6447 | -1.1004 | 0.5858 | 0.0248 |
| WF3 | CONT | 0.75-1.00 | 119 | 0.1403 | 0.4874 | 0.8837 | -1.1173 | 0.791 | -0.142 |
| WF3 | CONT | 1.00-1.50 | 88 | 0.1038 | 0.5 | 1.1712 | -1.0199 | 1.1483 | 0.0756 |
| WF3 | CONT | >=1.50 | 219 | 0.2583 | 0.2603 | 2.6307 | -1.0255 | 2.5653 | -0.0739 |
| WF3 | REV | <0.50 | 195 | 0.3409 | 0.7641 | 0.3038 | -1.0 | 0.3038 | -0.0038 |
| WF3 | REV | 0.50-0.75 | 120 | 0.2098 | 0.65 | 0.6274 | -1.0 | 0.6274 | 0.0578 |
| WF3 | REV | 0.75-1.00 | 109 | 0.1906 | 0.4954 | 0.8624 | -1.1342 | 0.7603 | -0.1451 |
| WF3 | REV | 1.00-1.50 | 118 | 0.2063 | 0.4576 | 1.1709 | -1.0057 | 1.1643 | -0.0096 |
| WF3 | REV | >=1.50 | 30 | 0.0524 | 0.4667 | 1.7861 | -1.0045 | 1.7781 | 0.2978 |

**禁止据结果挑最佳 bucket**；只看低 RR 是否系统性拖累。

---

## 6. P8 Direction-label × Execution attribution（仅 post-outcome）

| wf | setup | labeled | A corr+hit | B corr+loss | C wrong+hit | D wrong+loss | P(hit|corr) | P(hit|wrong) | E[R|corr] | E[R|wrong] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WF1 | CONT | 485 | 284 | 110 | 17 | 74 | 0.7208 | 0.1868 | 0.3624 | -0.74 |
| WF1 | REV | 315 | 168 | 0 | 38 | 109 | 1.0 | 0.2585 | 0.7871 | -0.6596 |
| WF2 | CONT | 688 | 438 | 111 | 28 | 111 | 0.7978 | 0.2014 | 0.4687 | -0.7117 |
| WF2 | REV | 570 | 365 | 0 | 55 | 150 | 1.0 | 0.2683 | 0.7208 | -0.6129 |
| WF3 | CONT | 640 | 409 | 119 | 19 | 93 | 0.7746 | 0.1696 | 0.4322 | -0.8891 |
| WF3 | REV | 441 | 278 | 0 | 39 | 124 | 1.0 | 0.2393 | 0.6868 | -0.6639 |

若 **B（方向正确但交易亏）** 很大 → 瓶颈在 execution mapping，不在 direction model。

---

## 7. P9 Entry degradation & P10 Same-bar

| wf | setup | next_open_R | ideal_close_R | delta |
|---|---|---:|---:|---:|
| WF1 | CONT | -0.042 | -0.0376 | -0.0044 |
| WF1 | REV | 0.0435 | 0.0132 | 0.0304 |
| WF2 | CONT | 0.0253 | 0.0287 | -0.0034 |
| WF2 | REV | 0.0481 | 0.0144 | 0.0338 |
| WF3 | CONT | -0.0376 | -0.0282 | -0.0095 |
| WF3 | REV | -0.0032 | -0.0166 | 0.0134 |

| wf | setup | ambiguous | stop_first_exp | target_first_exp | delta |
|---|---|---:|---:|---:|---:|
| WF1 | CONT | 0.0212 | -0.042 | -0.0085 | 0.0335 |
| WF1 | REV | 0.0275 | 0.0435 | 0.1081 | 0.0646 |
| WF2 | CONT | 0.0187 | 0.0253 | 0.068 | 0.0427 |
| WF2 | REV | 0.0179 | 0.0481 | 0.087 | 0.0389 |
| WF3 | CONT | 0.0189 | -0.0376 | 0.0009 | 0.0385 |
| WF3 | REV | 0.021 | -0.0032 | 0.0302 | 0.0334 |

---

## 8. P11 Reversal robustness（不 Promote）

| wf | trades | actual_hit | breakeven_hit | hit−breakeven | exp_R |
|---|---:|---:|---:|---:|---:|
| WF1 | 364 | 0.6126 | 0.5876 | 0.025 | 0.0435 |
| WF2 | 725 | 0.6207 | 0.5925 | 0.0282 | 0.0481 |
| WF3 | 572 | 0.6101 | 0.612 | -0.0019 | -0.0032 |

pooled gross expectancy = **0.0294**；
3/3 above breakeven = **False** →
保持 `SECONDARY_DIAGNOSTIC`。

---

## 9. P12 Cost

`COST_METADATA_DEFERRED_UNTIL_GROSS_EDGE`：Continuation gross point edge ≤0，
补成本的 ROI≈0。本轮无任何手续费/滑点假设。

---

## 10. P13 裁决

```json
{
  "EXECUTION_BASELINE_VALID_NEGATIVE": true,
  "TARGET_GEOMETRY_PRIMARY_BOTTLENECK": false,
  "verdict": "NO_SINGLE_EXECUTION_BOTTLENECK",
  "components": {
    "entry_degradation_small": true,
    "samebar_optimistic_not_fixed": true,
    "direction_correct_still_loss": false,
    "geometry_structure_present": false
  },
  "note": "VALID_NEGATIVE 不等同于 DIRECTION_EDGE_FALSE：只证明现有方向 edge 未被当前固定 execution mapping 转化为正 expectancy。"
}
```

- **EXECUTION_BASELINE_VALID_NEGATIVE = True**
  → fixed 1ATR + selected-boundary nearest target **没有稳定 gross edge**。
  **这不等同于 `DIRECTION_EDGE_FALSE`。**
- **NO_SINGLE_EXECUTION_BOTTLENECK**

---

## 11. P14 下一步 / P17 STOP

next_step = `RISK_COUPLED_EXECUTION (0.5/1.0/2.0, own frozen direction label & model per risk)`

代码 + 测试 + 运行 + 报告 + commit + push 后 STOP。
不自动执行 1:1 RR filter / risk 0.5–2.0 / target 优化 / Reversal promote /
symbol 筛选 / 成本建模。等 reviewer。
