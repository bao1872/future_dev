# SMC Execution Selection Purity Gate v1.1

**base**: `7c7858a` (v1.0.1) &nbsp; **脚本**: `run_execution_selection_purity_v1_1.py`

冻结 v1.0.1 的 target / risk / entry / stop / direction model。
**唯一新增变量**：一个预注册的更严格 clear gate `CLEAR90`（train-OOF precision=0.90）。

---

## 1. P1 Signal-level `rr_direction` consistency（HARD GATE）

| wf | groups | unique_rr==1 | unique_rr>1 | max_unique_rr |
|---|---:|---:|---:|---:|
| WF1 | 12692 | 12692 | 0 | 1 |
| WF2 | 11886 | 11886 | 0 | 1 |
| WF3 | 11693 | 11693 | 0 | 1 |

`n_groups_unique_rr_gt1 = 0`
→ signal-level frozen class 语义成立（同一 signal 的 simultaneous contacts
共享同一 Oracle direction）。

---

## 2. P4 CLEAR85 复现 v1.0.1

`CLEAR85 reproduces v1.0.1 = True`（三 WF 逐位一致）。

---

## 3. P2 按 frozen class 拆 execution

| wf | setup | frozen_class | n | share | hit | avg_win | avg_loss | exp_R |
|---|---|---|---:|---:|---:|---:|---:|---:|
| WF1 | CLEAR85_CONT | LONG_DOMINATES | 283 | 0.4609 | 0.6325 | 0.9486 | -1.0893 | 0.1997 |
| WF1 | CLEAR85_CONT | SHORT_DOMINATES | 202 | 0.329 | 0.604 | 0.874 | -1.0963 | 0.0937 |
| WF1 | CLEAR85_CONT | TRADEOFF_OR_OVERLAP | 113 | 0.184 | 0.0 | nan | -1.0182 | -1.0182 |
| WF1 | CLEAR85_CONT | UNRESOLVED_CENSOR | 16 | 0.0261 | 0.625 | 1.9814 | -1.0 | 0.8634 |
| WF1 | CLEAR85_CONT | NO_COMPARABLE_TARGET | 0 | 0.0 | nan | nan | nan | nan |
| WF1 | CLEAR85_REV | LONG_DOMINATES | 169 | 0.4643 | 0.7574 | 0.7104 | -1.0699 | 0.2785 |
| WF1 | CLEAR85_REV | SHORT_DOMINATES | 146 | 0.4011 | 0.5342 | 0.7205 | -1.0 | -0.0808 |
| WF1 | CLEAR85_REV | TRADEOFF_OR_OVERLAP | 36 | 0.0989 | 0.1667 | 0.5048 | -1.0 | -0.7492 |
| WF1 | CLEAR85_REV | UNRESOLVED_CENSOR | 13 | 0.0357 | 0.8462 | 0.8677 | -1.0 | 0.5803 |
| WF1 | CLEAR85_REV | NO_COMPARABLE_TARGET | 0 | 0.0 | nan | nan | nan | nan |
| WF1 | CLEAR90_CONT | LONG_DOMINATES | 146 | 0.4983 | 0.637 | 0.839 | -1.0951 | 0.1369 |
| WF1 | CLEAR90_CONT | SHORT_DOMINATES | 96 | 0.3276 | 0.6458 | 0.7562 | -1.2266 | 0.054 |
| WF1 | CLEAR90_CONT | TRADEOFF_OR_OVERLAP | 44 | 0.1502 | 0.0 | nan | -1.0 | -1.0 |
| WF1 | CLEAR90_CONT | UNRESOLVED_CENSOR | 7 | 0.0239 | 0.7143 | 2.9417 | -1.0 | 1.8155 |
| WF1 | CLEAR90_CONT | NO_COMPARABLE_TARGET | 0 | 0.0 | nan | nan | nan | nan |
| WF1 | CLEAR90_REV | LONG_DOMINATES | 124 | 0.4644 | 0.7823 | 0.7008 | -1.0 | 0.3305 |
| WF1 | CLEAR90_REV | SHORT_DOMINATES | 117 | 0.4382 | 0.5214 | 0.6157 | -1.0 | -0.1576 |
| WF1 | CLEAR90_REV | TRADEOFF_OR_OVERLAP | 20 | 0.0749 | 0.3 | 0.5048 | -1.0 | -0.5486 |
| WF1 | CLEAR90_REV | UNRESOLVED_CENSOR | 6 | 0.0225 | 1.0 | 0.7686 | nan | 0.7686 |
| WF1 | CLEAR90_REV | NO_COMPARABLE_TARGET | 0 | 0.0 | nan | nan | nan | nan |
| WF2 | CLEAR85_CONT | LONG_DOMINATES | 341 | 0.3984 | 0.6686 | 0.8612 | -1.0038 | 0.2432 |
| WF2 | CLEAR85_CONT | SHORT_DOMINATES | 347 | 0.4054 | 0.6859 | 0.8114 | -1.0797 | 0.2174 |
| WF2 | CLEAR85_CONT | TRADEOFF_OR_OVERLAP | 137 | 0.16 | 0.0 | nan | -1.0026 | -1.0026 |
| WF2 | CLEAR85_CONT | UNRESOLVED_CENSOR | 29 | 0.0339 | 0.5862 | 0.7055 | -1.0 | -0.0002 |
| WF2 | CLEAR85_CONT | NO_COMPARABLE_TARGET | 2 | 0.0023 | 1.0 | 0.3226 | nan | 0.3226 |
| WF2 | CLEAR85_REV | LONG_DOMINATES | 278 | 0.3834 | 0.6763 | 0.732 | -1.0 | 0.1713 |
| WF2 | CLEAR85_REV | SHORT_DOMINATES | 292 | 0.4028 | 0.7945 | 0.6591 | -1.0512 | 0.3077 |
| WF2 | CLEAR85_REV | TRADEOFF_OR_OVERLAP | 115 | 0.1586 | 0.0261 | 0.5837 | -1.0 | -0.9587 |
| WF2 | CLEAR85_REV | UNRESOLVED_CENSOR | 35 | 0.0483 | 0.7143 | 0.7932 | -1.0 | 0.2809 |
| WF2 | CLEAR85_REV | NO_COMPARABLE_TARGET | 5 | 0.0069 | 0.4 | 0.4258 | -1.0 | -0.4297 |
| WF2 | CLEAR90_CONT | LONG_DOMINATES | 181 | 0.4219 | 0.6796 | 0.8788 | -1.0074 | 0.2744 |
| WF2 | CLEAR90_CONT | SHORT_DOMINATES | 182 | 0.4242 | 0.6593 | 0.7369 | -1.0567 | 0.1259 |
| WF2 | CLEAR90_CONT | TRADEOFF_OR_OVERLAP | 48 | 0.1119 | 0.0 | nan | -1.0003 | -1.0003 |
| WF2 | CLEAR90_CONT | UNRESOLVED_CENSOR | 16 | 0.0373 | 0.625 | 0.7688 | -1.0 | 0.1055 |
| WF2 | CLEAR90_CONT | NO_COMPARABLE_TARGET | 2 | 0.0047 | 1.0 | 0.3226 | nan | 0.3226 |
| WF2 | CLEAR90_REV | LONG_DOMINATES | 206 | 0.4145 | 0.6796 | 0.5569 | -1.0 | 0.0581 |
| WF2 | CLEAR90_REV | SHORT_DOMINATES | 207 | 0.4165 | 0.8309 | 0.5407 | -1.0878 | 0.2653 |
| WF2 | CLEAR90_REV | TRADEOFF_OR_OVERLAP | 52 | 0.1046 | 0.0577 | 0.5837 | -1.0 | -0.9086 |
| WF2 | CLEAR90_REV | UNRESOLVED_CENSOR | 27 | 0.0543 | 0.6667 | 0.5549 | -1.0 | 0.0366 |
| WF2 | CLEAR90_REV | NO_COMPARABLE_TARGET | 5 | 0.0101 | 0.4 | 0.4258 | -1.0 | -0.4297 |
| WF3 | CLEAR85_CONT | LONG_DOMINATES | 271 | 0.3196 | 0.5351 | 1.0073 | -1.0722 | 0.0405 |
| WF3 | CLEAR85_CONT | SHORT_DOMINATES | 369 | 0.4351 | 0.7669 | 0.7509 | -1.1024 | 0.3189 |
| WF3 | CLEAR85_CONT | TRADEOFF_OR_OVERLAP | 152 | 0.1792 | 0.0 | nan | -1.0518 | -1.0518 |
| WF3 | CLEAR85_CONT | UNRESOLVED_CENSOR | 56 | 0.066 | 0.5357 | 0.7938 | -0.9807 | -0.0126 |
| WF3 | CLEAR85_CONT | NO_COMPARABLE_TARGET | 0 | 0.0 | nan | nan | nan | nan |
| WF3 | CLEAR85_REV | LONG_DOMINATES | 190 | 0.3322 | 0.5316 | 0.6506 | -1.0 | -0.1226 |
| WF3 | CLEAR85_REV | SHORT_DOMINATES | 251 | 0.4388 | 0.8606 | 0.6591 | -1.0387 | 0.4223 |
| WF3 | CLEAR85_REV | TRADEOFF_OR_OVERLAP | 84 | 0.1469 | 0.0119 | 0.3846 | -1.0288 | -1.012 |
| WF3 | CLEAR85_REV | UNRESOLVED_CENSOR | 44 | 0.0769 | 0.6364 | 0.6914 | -1.2546 | -0.0162 |
| WF3 | CLEAR85_REV | NO_COMPARABLE_TARGET | 3 | 0.0052 | 1.0 | 0.3972 | nan | 0.3972 |
| WF3 | CLEAR90_CONT | LONG_DOMINATES | 171 | 0.3497 | 0.5205 | 0.9791 | -1.0322 | 0.0146 |
| WF3 | CLEAR90_CONT | SHORT_DOMINATES | 222 | 0.454 | 0.7297 | 0.6806 | -1.1447 | 0.1873 |
| WF3 | CLEAR90_CONT | TRADEOFF_OR_OVERLAP | 68 | 0.1391 | 0.0 | nan | -1.0 | -1.0 |
| WF3 | CLEAR90_CONT | UNRESOLVED_CENSOR | 28 | 0.0573 | 0.5714 | 0.81 | -0.9561 | 0.0872 |
| WF3 | CLEAR90_CONT | NO_COMPARABLE_TARGET | 0 | 0.0 | nan | nan | nan | nan |
| WF3 | CLEAR90_REV | LONG_DOMINATES | 135 | 0.3453 | 0.4593 | 0.5019 | -1.0 | -0.3102 |
| WF3 | CLEAR90_REV | SHORT_DOMINATES | 189 | 0.4834 | 0.9048 | 0.5525 | -1.0752 | 0.3975 |
| WF3 | CLEAR90_REV | TRADEOFF_OR_OVERLAP | 37 | 0.0946 | 0.027 | 0.3846 | -1.0101 | -0.9724 |
| WF3 | CLEAR90_REV | UNRESOLVED_CENSOR | 27 | 0.0691 | 0.7407 | 0.5663 | -1.0 | 0.1602 |
| WF3 | CLEAR90_REV | NO_COMPARABLE_TARGET | 3 | 0.0077 | 1.0 | 0.3972 | nan | 0.3972 |

---

## 4. P3 Clear / Non-clear mixture

| wf | setup | variant | n | n_clear | n_nonclear | purity | E_R_clear | E_R_nonclear | E_R_total | required_purity | purity−req |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WF1 | CONT | CLEAR85 | 614 | 485 | 129 | 0.7899 | 0.1555 | -0.7848 | -0.042 | 0.8346 | -0.0447 |
| WF1 | REV | CLEAR85 | 364 | 315 | 49 | 0.8654 | 0.112 | -0.3965 | 0.0435 | 0.7798 | 0.0856 |
| WF1 | CONT | CLEAR90 | 293 | 242 | 51 | 0.8259 | 0.104 | -0.6136 | -0.0209 | 0.8551 | -0.0291 |
| WF1 | REV | CLEAR90 | 267 | 241 | 26 | 0.9026 | 0.0935 | -0.2446 | 0.0606 | 0.7234 | 0.1792 |
| WF2 | CONT | CLEAR85 | 856 | 688 | 168 | 0.8037 | 0.2302 | -0.8138 | 0.0253 | 0.7795 | 0.0242 |
| WF2 | REV | CLEAR85 | 725 | 570 | 155 | 0.7862 | 0.2411 | -0.6617 | 0.0481 | 0.7329 | 0.0533 |
| WF2 | CONT | CLEAR90 | 429 | 363 | 66 | 0.8462 | 0.1999 | -0.6921 | 0.0627 | 0.7759 | 0.0703 |
| WF2 | REV | CLEAR90 | 497 | 413 | 84 | 0.831 | 0.162 | -0.5763 | 0.0372 | 0.7806 | 0.0504 |
| WF3 | CONT | CLEAR85 | 848 | 640 | 208 | 0.7547 | 0.201 | -0.772 | -0.0376 | 0.7934 | -0.0387 |
| WF3 | REV | CLEAR85 | 572 | 441 | 131 | 0.771 | 0.1876 | -0.6452 | -0.0032 | 0.7748 | -0.0038 |
| WF3 | CONT | CLEAR90 | 489 | 393 | 96 | 0.8037 | 0.1121 | -0.6829 | -0.0439 | 0.8589 | -0.0553 |
| WF3 | REV | CLEAR90 | 391 | 324 | 67 | 0.8286 | 0.1026 | -0.4546 | 0.0071 | 0.8159 | 0.0128 |

---

## 5. Selector thresholds

| wf | clear_thr_85 | clear_thr_90 | cont_thr | rev_thr | clear90_available |
|---|---:|---:|---:|---:|---|
| WF1 | 0.7587 | 0.85 | 0.2011 | 0.5946 | True |
| WF2 | 0.6805 | 0.8207 | 0.218 | 0.5782 | True |
| WF3 | 0.6661 | 0.8039 | 0.2258 | 0.5758 | True |

Direction q10/q90 **未为 CLEAR90 重新优化**（与 CLEAR85 同一 availability-safe OOF）。

---

## 6. P7 CLEAR85 vs CLEAR90 execution

| wf | setup | variant | raw | collapsed | executed | purity | TRADEOFF | UNRES | NOCOMP | hit | **exp_R** | PF | /day | LONG | SHORT |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WF1 | CONT | CLEAR85 | 1159 | 649 | 614 | 0.7899 | 0.184 | 0.0261 | 0.0 | 0.5065 | -0.042 | 0.9199 | 6.0196 | 369 | 245 |
| WF1 | REV | CLEAR85 | 749 | 404 | 364 | 0.8654 | 0.0989 | 0.0357 | 0.0 | 0.6126 | 0.0435 | 1.1101 | 3.5686 | 225 | 139 |
| WF1 | CONT | CLEAR90 | 528 | 309 | 293 | 0.8259 | 0.1502 | 0.0239 | 0.0 | 0.5461 | -0.0209 | 0.958 | 2.8725 | 174 | 119 |
| WF1 | REV | CLEAR90 | 547 | 302 | 267 | 0.9026 | 0.0749 | 0.0225 | 0.0 | 0.6367 | 0.0606 | 1.1668 | 2.6176 | 168 | 99 |
| WF2 | CONT | CLEAR85 | 1813 | 927 | 856 | 0.8037 | 0.16 | 0.0339 | 0.0023 | 0.5666 | 0.0253 | 1.0569 | 8.3922 | 434 | 422 |
| WF2 | REV | CLEAR85 | 1773 | 768 | 725 | 0.7862 | 0.1586 | 0.0483 | 0.0069 | 0.6207 | 0.0481 | 1.1255 | 7.1078 | 290 | 435 |
| WF2 | CONT | CLEAR90 | 881 | 458 | 429 | 0.8462 | 0.1119 | 0.0373 | 0.0047 | 0.5944 | 0.0627 | 1.1511 | 4.2059 | 229 | 200 |
| WF2 | REV | CLEAR90 | 1258 | 529 | 497 | 0.831 | 0.1046 | 0.0543 | 0.0101 | 0.674 | 0.0372 | 1.112 | 4.8725 | 175 | 322 |
| WF3 | CONT | CLEAR85 | 1720 | 895 | 848 | 0.7547 | 0.1792 | 0.066 | 0.0 | 0.5401 | -0.0376 | 0.9229 | 8.396 | 302 | 546 |
| WF3 | REV | CLEAR85 | 1266 | 604 | 572 | 0.771 | 0.1469 | 0.0769 | 0.0052 | 0.6101 | -0.0032 | 0.9921 | 5.6634 | 144 | 428 |
| WF3 | CONT | CLEAR90 | 898 | 514 | 489 | 0.8037 | 0.1391 | 0.0573 | 0.0 | 0.546 | -0.0439 | 0.9073 | 4.8416 | 188 | 301 |
| WF3 | REV | CLEAR90 | 878 | 416 | 391 | 0.8286 | 0.0946 | 0.0691 | 0.0077 | 0.6573 | 0.0071 | 1.0205 | 3.8713 | 69 | 322 |

---

## 7. P11 Geometry by clear status（CLEAR85，固定 bins）

| wf | status | bucket | n | hit | exp_R |
|---|---|---|---:|---:|---:|
| WF1 | CLEAR | <0.50 | 123 | 0.8862 | 0.1524 |
| WF1 | CLEAR | 0.50-0.75 | 90 | 0.7667 | 0.2244 |
| WF1 | CLEAR | 0.75-1.00 | 61 | 0.6557 | 0.2252 |
| WF1 | CLEAR | 1.00-1.50 | 66 | 0.5 | 0.1016 |
| WF1 | CLEAR | >=1.50 | 145 | 0.3448 | 0.1107 |
| WF1 | NONCLEAR | <0.50 | 27 | 0.1481 | -0.815 |
| WF1 | NONCLEAR | 0.50-0.75 | 19 | 0.0526 | -0.918 |
| WF1 | NONCLEAR | 0.75-1.00 | 15 | 0.0 | -1.0 |
| WF1 | NONCLEAR | 1.00-1.50 | 23 | 0.087 | -0.8092 |
| WF1 | NONCLEAR | >=1.50 | 45 | 0.0667 | -0.6263 |
| WF2 | CLEAR | <0.50 | 217 | 0.8479 | 0.1091 |
| WF2 | CLEAR | 0.50-0.75 | 130 | 0.8154 | 0.3225 |
| WF2 | CLEAR | 0.75-1.00 | 90 | 0.7667 | 0.4236 |
| WF2 | CLEAR | 1.00-1.50 | 78 | 0.5513 | 0.166 |
| WF2 | CLEAR | >=1.50 | 173 | 0.3699 | 0.241 |
| WF2 | NONCLEAR | <0.50 | 41 | 0.1707 | -0.7941 |
| WF2 | NONCLEAR | 0.50-0.75 | 40 | 0.2 | -0.6782 |
| WF2 | NONCLEAR | 0.75-1.00 | 20 | 0.05 | -0.9008 |
| WF2 | NONCLEAR | 1.00-1.50 | 27 | 0.0741 | -0.8415 |
| WF2 | NONCLEAR | >=1.50 | 40 | 0.025 | -0.9073 |
| WF3 | CLEAR | <0.50 | 189 | 0.8995 | 0.1678 |
| WF3 | CLEAR | 0.50-0.75 | 127 | 0.8504 | 0.3393 |
| WF3 | CLEAR | 0.75-1.00 | 85 | 0.6353 | 0.1169 |
| WF3 | CLEAR | 1.00-1.50 | 70 | 0.6 | 0.3058 |
| WF3 | CLEAR | >=1.50 | 169 | 0.3195 | 0.1331 |
| WF3 | NONCLEAR | <0.50 | 50 | 0.22 | -0.8353 |
| WF3 | NONCLEAR | 0.50-0.75 | 56 | 0.1786 | -0.6883 |
| WF3 | NONCLEAR | 0.75-1.00 | 34 | 0.1176 | -0.7893 |
| WF3 | NONCLEAR | 1.00-1.50 | 18 | 0.1111 | -0.8193 |
| WF3 | NONCLEAR | >=1.50 | 50 | 0.06 | -0.7736 |

判断 v1.0.1 的"RR 无结构"是否只是 non-clear 污染混合造成。**不得据此新增 RR filter。**

---

## 8. P10 Reversal CLEAR90（secondary）

| wf | trades | purity | exp_R |
|---|---:|---:|---:|
| WF1 | 267 | 0.9026 | 0.0606 |
| WF2 | 497 | 0.831 | 0.0372 |
| WF3 | 391 | 0.8286 | 0.0071 |

`REVERSAL_CLEAR90_CANDIDATE = True`（candidate，不 promote）。

---

## 9. P9 Bootstrap

| scope | n_boot | p2.5 | p50 | p97.5 |
|---|---:|---:|---:|---:|
| - | STOP_NO_BOOTSTRAP | | | |

---

## 10. P8/P9 Gates

```json
{
  "SELECTION_PURITY_MECHANISM_CONFIRMED": false,
  "CLEAR90_GROSS_EXECUTION_EDGE_PRESENT": false,
  "components": {
    "A_baseline_mechanism": true,
    "B_purity_improved": true,
    "C_expectancy_improved": false
  },
  "per_wf_clear85": [
    -0.042,
    0.0253,
    -0.0376
  ],
  "per_wf_clear90": [
    -0.0209,
    0.0627,
    -0.0439
  ],
  "pooled_clear90": -0.0006
}
```

- **SELECTION_PURITY_MECHANISM_CONFIRMED = False**
- **CLEAR90_GROSS_EXECUTION_EDGE_PRESENT = False**

> CLEAR90 是看过 v1.0.1 后提出的 development hypothesis，
> **不是 independent OOS confirmation**。

---

## 11. P12/P13 下一步 / P16 STOP

next_step = `边界案例（reviewer 裁决）：purity 3/3 提高 (B=TRUE)，但 expectancy 未 3/3 改善 (C=FALSE, WF3 恶化)；pooled 由 -0.0156 改善到 -0.0006（约到 breakeven 但未转正）。按 P12，'purity 提高但 expectancy 未稳定改善' 使 RISK_COUPLED_EXECUTION 变为 eligible，但需 reviewer 决定。`

只有 CLEAR90 未能提高 purity、或 purity 提高但 expectancy 无改善时，
才批准 `RISK_COUPLED_EXECUTION`。若 CLEAR90 3/3 正 → 下一步
`CONTRACT_COST_METADATA_AUDIT`（先不做 stop/target 优化）。

代码 + 测试 + 运行 + 报告 + commit + push 后 STOP。等 reviewer。
