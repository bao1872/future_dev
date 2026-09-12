# SMC Group-Level Conditional Tradeoff Veto v1.3

**base**: `7107f16` (v1.2) &nbsp; **脚本**: `run_group_tradeoff_veto_v1_3.py`

v1.2 接受了 contact-level veto 失效，但裁决 **`NO_TRADEOFF_VETO_EDGE` 下得过头**
→ 改为 **`CONDITIONAL_SIGNAL_LEVEL_TRADEOFF_VETO = UNTESTED`**。本轮直接修 v1.2 的
两个 architecture mismatch：

1. **deployment-cohort mismatch**：toxicity 训练 cohort 现在**包含 CLEAR85 gate**
   —— nested-causal cross-fitted baseline group OOF（每个 outer-train fold 内部先
   用 train history 生成 clear/direction 阈值，再给 validation 的全部 contacts 打分、
   CLEAR85 AND q10、collapse），使毒性训练分布与真正 deployment 一致。
2. **decision-unit mismatch**：veto 单位从 contact 改为 **signal/group**。
   先 collapse 成 signal，再对**整个 signal** KEEP 或 VETO；veto 之后**绝不重新
   从剩余 contacts 计算 attack boundary**。

不新增任何 market feature：G0 = outermost selected contact 的 G4_BASE；G1 = G0 +
冻结 selector 的 decision-time 聚合元数据。

---

## 0. P9 Baseline reproduction（HARD）

| wf | baseline groups | baseline trades | baseline E[R] | v1.0.1 E[R] |
|---|---:|---:|---:|---:|
| WF1 | 649 | 614 | -0.042 | -0.042 |
| WF2 | 927 | 856 | 0.0253 | 0.0253 |
| WF3 | 895 | 848 | -0.0376 | -0.0376 |

> `FATAL_BASELINE_REPRODUCTION_FAIL = False`；per-WF E[R] 与 trades 逐位复现 v1.0.1。

---

## 1. P1 Group label 冻结（HARD ASSERT）

group key = `(symbol, decision_time)`；`rr_direction` 在同一 group 全部 contacts 上
**唯一**（已断言 `n_unique_rr <= 1`）。

```python
group_rr = sub.groupby(["symbol","decision_time"])["rr_direction"].nunique()
assert (group_rr <= 1).all()
```

group label：`TOXIC = TRADEOFF_OR_OVERLAP`，`CLEAR = LONG/SHORT_DOMINATES`；
训练排除 `UNRESOLVED_CENSOR / NO_COMPARABLE_TARGET`，但 test execution 不排除。
`group_label_available_time = max(同 group 全部 contacts 的 label_available_time)`（max, 保守）。

---

## 2. P2 cross-fitted baseline group cohort

| wf | fold | n_val | n_groups | n_toxic | group base rate | clear_thr | cont_thr |
|---|---|---:|---:|---:|---:|---:|---:|
| WF1 | 0.4-0.6 | 5112 | 61 | 6 | 0.0984 | 0.7031 | 0.1658 |
| WF1 | 0.6-0.8 | 4450 | 153 | 29 | 0.1895 | 0.6891 | 0.2013 |
| WF1 | 0.8-1.0 | 4546 | 165 | 30 | 0.1818 | 0.6943 | 0.1982 |
| WF2 | 0.4-0.6 | 9801 | 346 | 64 | 0.185 | 0.6943 | 0.1982 |
| WF2 | 0.6-0.8 | 10276 | 271 | 51 | 0.1882 | 0.6933 | 0.2044 |
| WF2 | 0.8-1.0 | 10372 | 390 | 67 | 0.1718 | 0.6863 | 0.2111 |
| WF3 | 0.4-0.6 | 15783 | 404 | 72 | 0.1782 | 0.6933 | 0.2044 |
| WF3 | 0.6-0.8 | 15198 | 498 | 83 | 0.1667 | 0.6778 | 0.2134 |
| WF3 | 0.8-1.0 | 14107 | 465 | 91 | 0.1957 | 0.6555 | 0.2183 |

> 与 v1.2 的关键差异：cohort base rate 现在落在 **0.10–0.20** 区间（≈ deployment 的
> 15–18%），不再是 v1.2 的 0.21–0.25。

---

## 3. P5 / P6 Group toxicity model（cross-fitted cohort）

| wf | block | model | cohort OOF n | toxic n | base rate | ROC-AUC | PR-AUC | Brier | LogLoss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| WF1 | G0_OUTERMOST | LOGIT | 287 | 54 | 0.1882 | 0.4932 | 0.1885 | 0.1703 | 0.5819 |
| WF1 | G0_OUTERMOST | HGB | 287 | 54 | 0.1882 | 0.5151 | 0.2237 | 0.1755 | 0.6116 |
| WF2 | G0_OUTERMOST | LOGIT | 606 | 110 | 0.1815 | 0.5166 | 0.1837 | 0.1559 | 0.4925 |
| WF2 | G0_OUTERMOST | HGB | 606 | 110 | 0.1815 | 0.556 | 0.1998 | 0.1617 | 0.5123 |
| WF3 | G0_OUTERMOST | LOGIT | 848 | 160 | 0.1887 | 0.603 | 0.2348 | 0.1517 | 0.4775 |
| WF3 | G0_OUTERMOST | HGB | 848 | 160 | 0.1887 | 0.5511 | 0.2076 | 0.1648 | 0.5222 |
| WF1 | G1_ARCH_AGG | LOGIT | 287 | 54 | 0.1882 | 0.4777 | 0.1804 | 0.1775 | 0.5964 |
| WF1 | G1_ARCH_AGG | HGB | 287 | 54 | 0.1882 | 0.5045 | 0.2119 | 0.1773 | 0.5842 |
| WF2 | G1_ARCH_AGG | LOGIT | 606 | 110 | 0.1815 | 0.5199 | 0.1863 | 0.1562 | 0.4928 |
| WF2 | G1_ARCH_AGG | HGB | 606 | 110 | 0.1815 | 0.5893 | 0.216 | 0.1595 | 0.4989 |
| WF3 | G1_ARCH_AGG | LOGIT | 848 | 160 | 0.1887 | 0.5918 | 0.2397 | 0.1529 | 0.4815 |
| WF3 | G1_ARCH_AGG | HGB | 848 | 160 | 0.1887 | 0.5775 | 0.2266 | 0.161 | 0.508 |

---

## 4. P7 Group veto threshold（denominator = baseline GROUPS）

| wf | block | model | available | threshold | OOF precision | OOF coverage | GROUP_TOXIC_VETO_UNAVAILABLE |
|---|---|---|---:|---:|---:|---:|---:|
| WF1 | G0_OUTERMOST | LOGIT | False | nan | nan | nan | True |
| WF1 | G0_OUTERMOST | HGB | False | nan | nan | nan | True |
| WF2 | G0_OUTERMOST | LOGIT | False | nan | nan | nan | True |
| WF2 | G0_OUTERMOST | HGB | False | nan | nan | nan | True |
| WF3 | G0_OUTERMOST | LOGIT | False | nan | nan | nan | True |
| WF3 | G0_OUTERMOST | HGB | False | nan | nan | nan | True |
| WF1 | G1_ARCH_AGG | LOGIT | False | nan | nan | nan | True |
| WF1 | G1_ARCH_AGG | HGB | False | nan | nan | nan | True |
| WF2 | G1_ARCH_AGG | LOGIT | False | nan | nan | nan | True |
| WF2 | G1_ARCH_AGG | HGB | False | nan | nan | nan | True |
| WF3 | G1_ARCH_AGG | LOGIT | False | nan | nan | nan | True |
| WF3 | G1_ARCH_AGG | HGB | False | nan | nan | nan | True |

---

## 5. P10 / P12 G0 mechanism + gate

| wf | model | base groups | base toxic | vetoed | post-veto toxic | toxic prec | toxic recall | clear ret | post-veto share |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WF1 | LOGIT | 649 | 118 | 0 | 118 | nan | 0.0 | 1.0 | nan |
| WF1 | HGB | 649 | 118 | 0 | 118 | nan | 0.0 | 1.0 | nan |
| WF2 | LOGIT | 927 | 147 | 0 | 147 | nan | 0.0 | 1.0 | nan |
| WF2 | HGB | 927 | 147 | 0 | 147 | nan | 0.0 | 1.0 | nan |
| WF3 | LOGIT | 895 | 164 | 0 | 164 | nan | 0.0 | 1.0 | nan |
| WF3 | HGB | 895 | 164 | 0 | 164 | nan | 0.0 | 1.0 | nan |

### G0 execution

| wf | model | base trades | post-veto trades | base TRADEOFF share | post-veto share | Δshare | clear ret | base E[R] | post-veto E[R] | ΔE[R] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WF1 | LOGIT | 614 | 614 | 0.184 | 0.184 | 0.0 | 1.0 | -0.042 | -0.042 | 0.0 |
| WF1 | HGB | 614 | 614 | 0.184 | 0.184 | 0.0 | 1.0 | -0.042 | -0.042 | 0.0 |
| WF2 | LOGIT | 856 | 856 | 0.16 | 0.16 | 0.0 | 1.0 | 0.0253 | 0.0253 | 0.0 |
| WF2 | HGB | 856 | 856 | 0.16 | 0.16 | 0.0 | 1.0 | 0.0253 | 0.0253 | 0.0 |
| WF3 | LOGIT | 848 | 848 | 0.1792 | 0.1792 | 0.0 | 1.0 | -0.0376 | -0.0376 | 0.0 |
| WF3 | HGB | 848 | 848 | 0.1792 | 0.1792 | 0.0 | 1.0 | -0.0376 | -0.0376 | 0.0 |

G0 mechanism gate = False；G0 execution gate (3/3 E[R]>0) = False。

---

## 6. P13 G1（仅 G0 execution 不过时运行）

### G1 mechanism + execution

| WF1 | LOGIT | 649 | 118 | 0 | 118 | 0.0 | 1.0 | nan |
| WF1 | HGB | 649 | 118 | 0 | 118 | 0.0 | 1.0 | nan |
| WF2 | LOGIT | 927 | 147 | 0 | 147 | 0.0 | 1.0 | nan |
| WF2 | HGB | 927 | 147 | 0 | 147 | 0.0 | 1.0 | nan |
| WF3 | LOGIT | 895 | 164 | 0 | 164 | 0.0 | 1.0 | nan |
| WF3 | HGB | 895 | 164 | 0 | 164 | 0.0 | 1.0 | nan |

| WF1 | LOGIT | 614 | 614 | 0.184 | 0.184 | 0.0 | 1.0 | -0.042 | -0.042 | 0.0 |
| WF1 | HGB | 614 | 614 | 0.184 | 0.184 | 0.0 | 1.0 | -0.042 | -0.042 | 0.0 |
| WF2 | LOGIT | 856 | 856 | 0.16 | 0.16 | 0.0 | 1.0 | 0.0253 | 0.0253 | 0.0 |
| WF2 | HGB | 856 | 856 | 0.16 | 0.16 | 0.0 | 1.0 | 0.0253 | 0.0253 | 0.0 |
| WF3 | LOGIT | 848 | 848 | 0.1792 | 0.1792 | 0.0 | 1.0 | -0.0376 | -0.0376 | 0.0 |
| WF3 | HGB | 848 | 848 | 0.1792 | 0.1792 | 0.0 | 1.0 | -0.0376 | -0.0376 | 0.0 |

---

## 7. P14 Final gate

```json
{
  "pre_registered_architecture": "NO_GROUP_TRADEOFF_VETO_EDGE",
  "rule": "G0 execution PASS -> G0_HGB; else G1 exec PASS -> G1_HGB; else NO_GROUP_TRADEOFF_VETO_EDGE",
  "GROUP_LEVEL_TRADEOFF_VETO_EDGE": false,
  "next_step": "RISK_COUPLED_EXECUTION (0.5/1.0/2.0, each with own frozen direction label/model)"
}
```

**GROUP_LEVEL_TRADEOFF_VETO_EDGE = False**

---

## 8. P15 Unknown diagnostic（报告，不用于选择）

见 `group_unknown_diagnostic.csv`。veto 只作用于 resolved groups；unknown groups 始终
KEEP，因此 baseline 与 post-veto 的 unknown 计数一致（这是预期）。

---

## 9. OOS guard

max `n_exit_on_or_after_oos` = 0
（必须为 0，HARD；见 `oos_guard_audit.csv`）。

---

## 10. P16 Bootstrap

| scope | n_boot | p2.5 | p50 | p97.5 |
|---|---:|---:|---:|---:|
| - | STOP_NO_BOOTSTRAP | | | |

---

## 11. Reversal / STOP

`REVERSAL_CLEAR90_CANDIDATE` 保持冻结，不 promote。代码+测试+运行+报告+commit+push 后
**STOP**；不自动进入 risk coupling / cost / RR filter / 新 SMC taxonomy。
