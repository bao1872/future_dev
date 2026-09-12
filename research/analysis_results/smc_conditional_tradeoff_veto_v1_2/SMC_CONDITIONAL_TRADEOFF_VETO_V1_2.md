# SMC Conditional Tradeoff Veto v1.2

**base**: `1c16a08` (v1.1) &nbsp; **脚本**: `run_conditional_tradeoff_veto_v1_2.py`

核心问题：**在下单之前，能不能事前认出 15–18% 的 TRADEOFF toxic events 并 veto？**

不优化 risk / target / stop / selector threshold，不加新 SMC taxonomy（Stage B 只用
严格 contact-前 price path）。toxicity 模型**只能 veto**，不能新增 baseline 未选中的交易。

---

## 0. P0 v1.1 结论冻结与措辞修正

接受 v1.1：

- `CLEAR85 Continuation` per-WF gross E[R] =
  [-0.042, 0.0253, -0.0376]
- `TRADEOFF_OR_OVERLAP` = 当前 fixed execution 下 **empirical toxic class**
  （per-WF n = [113, 137, 152]，
  target_hit = 0/0/0，E[R] ≈
  [-1.0182, -1.0026, -1.0518]）。

**措辞修正（必须）**：禁止写"TRADEOFF 定义上必然打不到 target"。
`TRADEOFF_OR_OVERLAP` 的定义只是"Long/Short 两侧 Oracle R 区间不存在严格支配关系"，
它**不构成**"Continuation 的 beyond-attack target 必不可能被击中"的数学恒等式。
正确表述：**跨 3/3 WF 稳定的经验事实（empirical toxic class）**。

### 0b. P0 结构性发现：P3 字面伪代码不可实现

`y_reversal` 在**全部 24,225 条 `TRADEOFF_OR_OVERLAP` 行上都是 NaN**
（LONG 31,847 / SHORT 33,161 / TRADEOFF 24,225 → `y_rev` notna = 0）。
因此 P3 的字面写法

```python
cont_candidate_oof = (p_rev_oof <= np.quantile(p_rev_oof, 0.10))
y_toxic = (rr_direction == "TRADEOFF_OR_OVERLAP").astype(int)
```

**结构上不可能产生任何 toxic 正样本**——direction OOF cohort 只含 clear 行。

本实现改为**忠实还原冻结 selector 的候选定义**：fold-wise 用 availability-safe
训练得到的 direction 模型给**全部** validation 行（含 TRADEOFF）打分；阈值仍取
clear 行 OOF 分布的 q10；cohort 再限定 frozen resolved 类。

> 已记录的分布差异：本 cohort **不含 clear gate**，因此其 toxic base rate 高于
> baseline-selected 分布。

---

## 1. P1 Oracle ceiling（仅 descriptive）

| wf | n_executed | n_tradeoff | original E[R] | oracle_no_tradeoff E[R] | delta |
|---|---:|---:|---:|---:|---:|
| WF1 | 614 | 113 | -0.042 | 0.1781 | 0.2202 |
| WF2 | 856 | 137 | 0.0253 | 0.2211 | 0.1959 |
| WF3 | 848 | 152 | -0.0376 | 0.1838 | 0.2215 |

> **`ORACLE_DIAGNOSTIC_ONLY / NOT_DEPLOYABLE`** —— 事前不知道谁是 TRADEOFF，
> 这不是策略结果。它只说明：如果事前识别能做明显更好，经济价值远大于微调
> stop 0.5/1/2。

---

## 2. P2 冻结 baseline（逐位复现 v1.0.1）

baseline = `CLEAR85 AND Continuation q10`，execution 完全复用 v1.0.1 repaired 路径。
上表 `original E[R]` 即 baseline per-WF gross E[R]，与 v1.0.1
`execution_metrics_repaired.csv` 一致（见测试 `test_clear85_q10_baseline_unchanged`）。

---

## 3. P5 Stage A：toxicity 模型（G4_BASE only）

| wf | model | cohort OOF n | toxic n | base rate | ROC-AUC | PR-AUC | Brier | LogLoss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| WF1 | T0_LOGIT | 751 | 159 | 0.2117 | 0.6677 | 0.4042 | 0.1649 | 0.5147 |
| WF1 | T0_HGB | 751 | 159 | 0.2117 | 0.6553 | 0.3728 | 0.1689 | 0.5503 |
| WF2 | T0_LOGIT | 1446 | 305 | 0.2109 | 0.5772 | 0.265 | 0.1732 | 0.535 |
| WF2 | T0_HGB | 1446 | 305 | 0.2109 | 0.6224 | 0.2886 | 0.1716 | 0.5274 |
| WF3 | T0_LOGIT | 2498 | 631 | 0.2526 | 0.6684 | 0.4322 | 0.1753 | 0.5352 |
| WF3 | T0_HGB | 2498 | 631 | 0.2526 | 0.6504 | 0.4045 | 0.1818 | 0.5527 |

指标在 **Continuation OOF candidate cohort** 上报告（非全 contacts）。

---

## 4. P6 toxic veto threshold（train-OOF only，不扫）

预注册：`TARGET_TOXIC_PRECISION = 0.5`、
`MIN_VETO_COVERAGE = 0.03`。达不到则 `TOXIC_VETO_UNAVAILABLE`，
**不得降低 precision**。

| wf | model | available | threshold | OOF precision | OOF veto coverage | vetoed contacts | veto rate of selected |
|---|---|---|---:|---:|---:|---:|---:|
| WF1 | T0_LOGIT | True | 0.559215 | 0.5 | 0.1065 | 0 | 0.0 |
| WF1 | T0_HGB | True | 0.494206 | 0.5 | 0.1332 | 16 | 0.0138 |
| WF2 | T0_LOGIT | False | nan | nan | nan | 0 | 0.0 |
| WF2 | T0_HGB | False | nan | nan | nan | 0 | 0.0 |
| WF3 | T0_LOGIT | True | 0.388888 | 0.5 | 0.1673 | 15 | 0.0087 |
| WF3 | T0_HGB | True | 0.448738 | 0.5 | 0.1425 | 58 | 0.0337 |

---

## 5. P8 Stage A veto execution

| wf | model | base trades | post-veto trades | base TRADEOFF share | post-veto TRADEOFF share | Δshare | clear retention | TRADEOFF removal | base E[R] | post-veto E[R] | ΔE[R] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WF1 | T0_LOGIT | 614 | 614 | 0.184 | 0.184 | 0.0 | 1.0 | 0.0 | -0.042 | -0.042 | 0.0 |
| WF1 | T0_HGB | 614 | 602 | 0.184 | 0.1827 | 0.0013 | 0.9814 | 0.0265 | -0.042 | -0.0303 | 0.0117 |
| WF2 | T0_LOGIT | 856 | 856 | 0.16 | 0.16 | 0.0 | 1.0 | 0.0 | 0.0253 | 0.0253 | 0.0 |
| WF2 | T0_HGB | 856 | 856 | 0.16 | 0.16 | 0.0 | 1.0 | 0.0 | 0.0253 | 0.0253 | 0.0 |
| WF3 | T0_LOGIT | 848 | 840 | 0.1792 | 0.1786 | 0.0007 | 0.9906 | 0.0132 | -0.0376 | -0.037 | 0.0006 |
| WF3 | T0_HGB | 848 | 824 | 0.1792 | 0.1808 | -0.0016 | 0.9703 | 0.0197 | -0.0376 | -0.0369 | 0.0007 |

Stage A 机制门（3/3 TRADEOFF share 下降 + 3/3 E[R] 改善 + clear retention≥
0.65）：

```json
{
  "T0_LOGIT": {
    "passed": false,
    "detail": {
      "tradeoff_share_down": false,
      "expectancy_up": false,
      "clear_retention_ok": true
    }
  },
  "T0_HGB": {
    "passed": false,
    "detail": {
      "tradeoff_share_down": false,
      "expectancy_up": false,
      "clear_retention_ok": true
    }
  }
}
```

Stage A execution 门（post-veto E[R]>0 3/3、pooled>0、≥200 trades/WF）：

```json
{
  "T0_LOGIT": {
    "passed": false,
    "detail": {
      "per_wf_e_R": [
        -0.042,
        0.0253,
        -0.037
      ],
      "pooled_E_R": -0.015267620281868267,
      "per_wf_trades": [
        614.0,
        856.0,
        840.0
      ]
    }
  },
  "T0_HGB": {
    "passed": false,
    "detail": {
      "per_wf_e_R": [
        -0.0303,
        0.0253,
        -0.0369
      ],
      "pooled_E_R": -0.01184605355518335,
      "per_wf_trades": [
        602.0,
        856.0,
        824.0
      ]
    }
  }
}
```

**`STAGE_A_EXECUTION_PASS = False`**

### 5b. 机制归因：接触级 veto 为什么（不）能消除信号级 toxicity

toxic outcome 在 **collapsed group（signal）层面**确定；contact 级 veto 只有把某组的
**最外沿 attacked contact** 也 veto 掉、并且该组全部 selected contact 都被 veto 时，
才可能真正移除该 toxic signal。

| wf | model | base groups | base toxic groups | post-veto groups | post-veto toxic groups | toxic groups deactivated | newly toxic | vetoed contacts | vetoed TRADEOFF | vetoed clear | vetoed-toxic precision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WF1 | T0_LOGIT | 649 | 118 | 649 | 118 | 0 | 0 | 0 | 0 | 0 | nan |
| WF1 | T0_HGB | 649 | 118 | 637 | 115 | 3 | 0 | 16 | 3 | 13 | 0.1875 |
| WF2 | T0_LOGIT | 927 | 147 | 927 | 147 | 0 | 0 | 0 | 0 | 0 | nan |
| WF2 | T0_HGB | 927 | 147 | 927 | 147 | 0 | 0 | 0 | 0 | 0 | nan |
| WF3 | T0_LOGIT | 895 | 164 | 887 | 162 | 2 | 0 | 15 | 4 | 11 | 0.2667 |
| WF3 | T0_HGB | 895 | 164 | 870 | 160 | 4 | 0 | 58 | 10 | 42 | 0.1724 |
| WF1 | T1_HGB | 649 | 118 | 643 | 117 | 1 | 0 | 12 | 3 | 6 | 0.25 |
| WF2 | T1_HGB | 927 | 147 | 927 | 147 | 0 | 0 | 0 | 0 | 0 | nan |
| WF3 | T1_HGB | 895 | 164 | 882 | 161 | 3 | 0 | 37 | 7 | 29 | 0.1892 |

**三条机制结论：**

1. **veto 力度本身微乎其微**：被 veto 的 contact 只占 selected 的 0–3.4%
   （`veto_rate_of_selected`），因为 50% toxic precision 在 cohort
   （base rate 0.21–0.25）上只对应 10–17% OOF coverage，且阈值迁移到
   test-selected 分布后实际 precision 掉到 0.17–0.27；**WF2 两个 Stage A 模型
   连 3% coverage / 50% precision 都达不到 → `TOXIC_VETO_UNAVAILABLE`，0 veto**。
2. **信号级 toxicity 几乎没被消除**：`n_baseline_toxic_groups_deactivated`
   仅 0/0/2–4（baseline toxic groups = 118/147/164），
   `n_groups_newly_toxic = 0`。原因：toxic outcome 在 collapsed group 层面确定，
   contact 级 veto 只有在 veto 掉该组**最外沿 attacked contact**、且该组全部
   selected contact 均被 veto 时才可能移除该 signal。
3. **甚至可能反向**：WF3 `T0_HGB` 的 post-veto TRADEOFF **share**
   （0.1792→0.1808）上升，尽管 TRADEOFF 绝对笔数下降（152→149）——
   因为总笔数下降更快（848→824）；同时 veto 可能把 clear 的外沿 contact 去掉，
   使 TRADEOFF contact 变成新的 attack boundary。

> 结论：**接触级 veto 在本设计下结构上无法消除信号级毒单**。这是本轮最重要的
> 机制结论之一，且它独立于 toxicity 模型质量。

---

## 6. P9–P12 Stage B（严格 pre-contact dynamics）

`RUN_STAGE_B = True`（仅 Stage A execution 不通过时运行）。

只用 `bar <= contact_bar_index - 1`，horizons 3/6/12；
禁止 FVG / OB / trend taxonomy；不含 Volume。

| feature | group | n_valid | nan_rate | p10 | median | p90 | note |
|---|---|---:|---:|---:|---:|---:|---|
| approach_return_R_3 | 3 | 96900 | 0.0 | -0.2703 | 0.6977 | 2.0 |  |
| approach_return_R_6 | 6 | 96900 | 0.0 | -0.2966 | 1.0 | 2.5543 |  |
| approach_return_R_12 | 12 | 96900 | 0.0 | -0.3125 | 1.3257 | 3.3824 |  |
| path_length_R_3 | 3 | 96900 | 0.0 | 0.4545 | 1.2663 | 2.3529 |  |
| path_length_R_6 | 6 | 96900 | 0.0 | 1.1321 | 2.4634 | 3.6667 |  |
| path_length_R_12 | 12 | 96900 | 0.0 | 2.4286 | 4.5833 | 6.9231 |  |
| efficiency_3 | 3 | 96900 | 0.0 | 0.1429 | 0.7377 | 1.0 | unsigned |net|/plen；与 approach_efficiency_3 互补（符号 vs 幅度） |
| efficiency_6 | 6 | 96900 | 0.0 | 0.0952 | 0.5 | 0.9149 |  |
| efficiency_12 | 12 | 96900 | 0.0 | 0.0667 | 0.3333 | 0.6842 |  |
| toward_fraction_3 | 3 | 96900 | 0.0 | 0.3333 | 0.6667 | 1.0 |  |
| toward_fraction_6 | 6 | 96900 | 0.0 | 0.3333 | 0.5 | 0.8333 |  |
| toward_fraction_12 | 12 | 96900 | 0.0 | 0.3333 | 0.5 | 0.6667 |  |
| mean_range_R_3 | 3 | 96900 | 0.0 | 0.463 | 0.8621 | 1.0897 |  |
| mean_range_R_6 | 6 | 96900 | 0.0 | 0.4444 | 0.8205 | 1.0526 |  |
| mean_range_R_12 | 12 | 96900 | 0.0 | 0.4167 | 0.7738 | 1.1047 |  |
| range_ratio_3_12 | derived | 96900 | 0.0 | 0.7522 | 1.0526 | 1.5833 |  |
| approach_efficiency_3 | derived | 96900 | 0.0 | -0.3333 | 0.6471 | 1.0 | SIGNED counterpart of efficiency_3: equals side*net/plen (= approach_return_R_3 / path_length_R_3), NOT identical to unsigned efficiency_3 |

### Stage B incremental

| wf | model | PR-AUC | ROC-AUC | ΔPR-AUC vs T0_HGB | ΔROC-AUC vs T0_HGB |
|---|---|---:|---:|---:|---:|
| WF1 | T1_HGB | 0.3582 | 0.619 | -0.014600000000000002 | -0.0363 |
| WF2 | T1_HGB | 0.2693 | 0.5964 | -0.01930000000000004 | -0.025999999999999912 |
| WF3 | T1_HGB | 0.3804 | 0.6435 | -0.02410000000000001 | -0.006900000000000017 |

**`PRECONTACT_INCREMENT_WEAK`**

### Stage B veto execution

| wf | model | base trades | post-veto trades | base TRADEOFF share | post-veto TRADEOFF share | clear retention | base E[R] | post-veto E[R] | ΔE[R] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WF1 | T1_HGB | 614 | 608 | 0.184 | 0.1842 | 0.9918 | -0.042 | -0.0432 | -0.0012 |
| WF2 | T1_HGB | 856 | 856 | 0.16 | 0.16 | 1.0 | 0.0253 | 0.0253 | 0.0 |
| WF3 | T1_HGB | 848 | 838 | 0.1792 | 0.1802 | 0.9859 | -0.0376 | -0.0388 | -0.0011 |

---

## 7. 按 frozen class 拆 execution（baseline / post-veto）

| wf | model | variant | frozen_class | n | share | target_hit | E[R] |
|---|---|---|---|---:|---:|---:|---:|
| WF1 | BASELINE | baseline | LONG_DOMINATES | 283 | 0.4609 | 0.6325 | 0.1997 |
| WF1 | BASELINE | baseline | SHORT_DOMINATES | 202 | 0.329 | 0.604 | 0.0937 |
| WF1 | BASELINE | baseline | TRADEOFF_OR_OVERLAP | 113 | 0.184 | 0.0 | -1.0182 |
| WF1 | BASELINE | baseline | UNRESOLVED_CENSOR | 16 | 0.0261 | 0.625 | 0.8634 |
| WF1 | BASELINE | baseline | NO_COMPARABLE_TARGET | 0 | 0.0 | nan | nan |
| WF2 | BASELINE | baseline | LONG_DOMINATES | 341 | 0.3984 | 0.6686 | 0.2432 |
| WF2 | BASELINE | baseline | SHORT_DOMINATES | 347 | 0.4054 | 0.6859 | 0.2174 |
| WF2 | BASELINE | baseline | TRADEOFF_OR_OVERLAP | 137 | 0.16 | 0.0 | -1.0026 |
| WF2 | BASELINE | baseline | UNRESOLVED_CENSOR | 29 | 0.0339 | 0.5862 | -0.0002 |
| WF2 | BASELINE | baseline | NO_COMPARABLE_TARGET | 2 | 0.0023 | 1.0 | 0.3226 |
| WF3 | BASELINE | baseline | LONG_DOMINATES | 271 | 0.3196 | 0.5351 | 0.0405 |
| WF3 | BASELINE | baseline | SHORT_DOMINATES | 369 | 0.4351 | 0.7669 | 0.3189 |
| WF3 | BASELINE | baseline | TRADEOFF_OR_OVERLAP | 152 | 0.1792 | 0.0 | -1.0518 |
| WF3 | BASELINE | baseline | UNRESOLVED_CENSOR | 56 | 0.066 | 0.5357 | -0.0126 |
| WF3 | BASELINE | baseline | NO_COMPARABLE_TARGET | 0 | 0.0 | nan | nan |
| WF1 | T0_LOGIT | post_veto | LONG_DOMINATES | 283 | 0.4609 | 0.6325 | 0.1997 |
| WF1 | T0_LOGIT | post_veto | SHORT_DOMINATES | 202 | 0.329 | 0.604 | 0.0937 |
| WF1 | T0_LOGIT | post_veto | TRADEOFF_OR_OVERLAP | 113 | 0.184 | 0.0 | -1.0182 |
| WF1 | T0_LOGIT | post_veto | UNRESOLVED_CENSOR | 16 | 0.0261 | 0.625 | 0.8634 |
| WF1 | T0_LOGIT | post_veto | NO_COMPARABLE_TARGET | 0 | 0.0 | nan | nan |
| WF1 | T0_HGB | post_veto | LONG_DOMINATES | 277 | 0.4601 | 0.639 | 0.2171 |
| WF1 | T0_HGB | post_veto | SHORT_DOMINATES | 199 | 0.3306 | 0.608 | 0.0997 |
| WF1 | T0_HGB | post_veto | TRADEOFF_OR_OVERLAP | 110 | 0.1827 | 0.0 | -1.0187 |
| WF1 | T0_HGB | post_veto | UNRESOLVED_CENSOR | 16 | 0.0266 | 0.625 | 0.8634 |
| WF1 | T0_HGB | post_veto | NO_COMPARABLE_TARGET | 0 | 0.0 | nan | nan |
| WF2 | T0_LOGIT | post_veto | LONG_DOMINATES | 341 | 0.3984 | 0.6686 | 0.2432 |
| WF2 | T0_LOGIT | post_veto | SHORT_DOMINATES | 347 | 0.4054 | 0.6859 | 0.2174 |
| WF2 | T0_LOGIT | post_veto | TRADEOFF_OR_OVERLAP | 137 | 0.16 | 0.0 | -1.0026 |
| WF2 | T0_LOGIT | post_veto | UNRESOLVED_CENSOR | 29 | 0.0339 | 0.5862 | -0.0002 |
| WF2 | T0_LOGIT | post_veto | NO_COMPARABLE_TARGET | 2 | 0.0023 | 1.0 | 0.3226 |
| WF2 | T0_HGB | post_veto | LONG_DOMINATES | 341 | 0.3984 | 0.6686 | 0.2432 |
| WF2 | T0_HGB | post_veto | SHORT_DOMINATES | 347 | 0.4054 | 0.6859 | 0.2174 |
| WF2 | T0_HGB | post_veto | TRADEOFF_OR_OVERLAP | 137 | 0.16 | 0.0 | -1.0026 |
| WF2 | T0_HGB | post_veto | UNRESOLVED_CENSOR | 29 | 0.0339 | 0.5862 | -0.0002 |
| WF2 | T0_HGB | post_veto | NO_COMPARABLE_TARGET | 2 | 0.0023 | 1.0 | 0.3226 |
| WF3 | T0_LOGIT | post_veto | LONG_DOMINATES | 270 | 0.3214 | 0.537 | 0.0443 |
| WF3 | T0_LOGIT | post_veto | SHORT_DOMINATES | 364 | 0.4333 | 0.7665 | 0.3173 |
| WF3 | T0_LOGIT | post_veto | TRADEOFF_OR_OVERLAP | 150 | 0.1786 | 0.0 | -1.0525 |
| WF3 | T0_LOGIT | post_veto | UNRESOLVED_CENSOR | 56 | 0.0667 | 0.5357 | -0.0126 |
| WF3 | T0_LOGIT | post_veto | NO_COMPARABLE_TARGET | 0 | 0.0 | nan | nan |
| WF3 | T0_HGB | post_veto | LONG_DOMINATES | 266 | 0.3228 | 0.5338 | 0.0378 |
| WF3 | T0_HGB | post_veto | SHORT_DOMINATES | 355 | 0.4308 | 0.769 | 0.3242 |
| WF3 | T0_HGB | post_veto | TRADEOFF_OR_OVERLAP | 149 | 0.1808 | 0.0 | -1.0528 |
| WF3 | T0_HGB | post_veto | UNRESOLVED_CENSOR | 54 | 0.0655 | 0.5556 | 0.024 |
| WF3 | T0_HGB | post_veto | NO_COMPARABLE_TARGET | 0 | 0.0 | nan | nan |
| WF1 | T1_HGB | post_veto | LONG_DOMINATES | 281 | 0.4622 | 0.637 | 0.2089 |
| WF1 | T1_HGB | post_veto | SHORT_DOMINATES | 200 | 0.3289 | 0.605 | 0.0896 |
| WF1 | T1_HGB | post_veto | TRADEOFF_OR_OVERLAP | 112 | 0.1842 | 0.0 | -1.0184 |
| WF1 | T1_HGB | post_veto | UNRESOLVED_CENSOR | 15 | 0.0247 | 0.6 | 0.7444 |
| WF1 | T1_HGB | post_veto | NO_COMPARABLE_TARGET | 0 | 0.0 | nan | nan |
| WF2 | T1_HGB | post_veto | LONG_DOMINATES | 341 | 0.3984 | 0.6686 | 0.2432 |
| WF2 | T1_HGB | post_veto | SHORT_DOMINATES | 347 | 0.4054 | 0.6859 | 0.2174 |
| WF2 | T1_HGB | post_veto | TRADEOFF_OR_OVERLAP | 137 | 0.16 | 0.0 | -1.0026 |
| WF2 | T1_HGB | post_veto | UNRESOLVED_CENSOR | 29 | 0.0339 | 0.5862 | -0.0002 |
| WF2 | T1_HGB | post_veto | NO_COMPARABLE_TARGET | 2 | 0.0023 | 1.0 | 0.3226 |
| WF3 | T1_HGB | post_veto | LONG_DOMINATES | 267 | 0.3186 | 0.5356 | 0.0435 |
| WF3 | T1_HGB | post_veto | SHORT_DOMINATES | 364 | 0.4344 | 0.7692 | 0.3172 |
| WF3 | T1_HGB | post_veto | TRADEOFF_OR_OVERLAP | 151 | 0.1802 | 0.0 | -1.0521 |
| WF3 | T1_HGB | post_veto | UNRESOLVED_CENSOR | 56 | 0.0668 | 0.5357 | -0.0126 |
| WF3 | T1_HGB | post_veto | NO_COMPARABLE_TARGET | 0 | 0.0 | nan | nan |

---

## 8. P13 Final economic gate（预注册架构，非 test 后挑选）

规则：`StageA PASS → T0_HGB`；否则 `StageB incremental PASS → T1_HGB`；
否则 `NO_TRADEOFF_VETO_EDGE`。

```json
{
  "pre_registered_architecture": "NO_TRADEOFF_VETO_EDGE",
  "architecture_rule": "StageA PASS->T0_HGB; else StageB incr PASS->T1_HGB; else NO_TRADEOFF_VETO_EDGE",
  "detail": {
    "per_wf_e_R": [],
    "pooled_E_R": NaN,
    "per_wf_trades": []
  },
  "TRADEOFF_VETO_GROSS_EDGE_PRESENT": false
}
```

**`TRADEOFF_VETO_GROSS_EDGE_PRESENT = False`**

---

## 9. P14 Reversal

保持冻结 `REVERSAL_CLEAR90_CANDIDATE frozen`：本轮不重新调、不 promote。

---

## 10. OOS guard

`max n_exit_on_or_after_oos = 0`
（必须为 0，HARD；见 `oos_guard_audit.csv`）。

---

## 11. P15 / P18 下一步 / STOP

next_step = `RISK_COUPLED_EXECUTION (0.5/1.0/2.0，各自独立 frozen direction label/model)`

代码 + 测试 + 运行 + 报告 + commit + push 后 **STOP**。禁止自动进入
risk coupling / cost metadata / Reversal promotion / RR filter / 新 SMC taxonomy。
等 reviewer。
