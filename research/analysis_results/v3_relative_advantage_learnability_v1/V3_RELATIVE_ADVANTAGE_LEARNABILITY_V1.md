# V3-A2 — Relative Execution Advantage Learnability Audit

**不构造交易 policy。** 只回答：相对固定 RR3 Strict Reassess 的动作优势 ΔR 是否可被在线预测。

- base `c905b3c`；n=9015；equal weight only；无 time decay；T1–T15 全 PASS；P1_read=`False`
- LIMIT redundancy：`reduced oracle == full oracle` = True，`unique_LIMIT_RR3` = 0 → **LIMIT 不进入学习动作集**

## 0. Target summary

```
task  wf    n  mean_delta  share_delta_gt_0  share_delta_gt_0p05  share_delta_gt_0p10
   E WF1 3160   -0.023850          0.325316             0.325000             0.320886
   E WF2 2847   -0.011251          0.344573             0.343168             0.336846
   E WF3 3008   -0.060340          0.346742             0.344747             0.340758
   V WF1 3160   -0.034226          0.206329             0.206329             0.206329
   V WF2 2847   -0.044369          0.203372             0.203372             0.203372
   V WF3 3008   -0.104135          0.211104             0.211104             0.211104
```

**Task E = execution-style**（Market vs Reassess）；**Task V = post-selection veto**（重新打开 Enter/Skip，不是 execution-style）。两者不得合并成同一经济结论。

## 1. Selected models + outer metrics

```
task  wf model_type feature_block  inner_mean_spearman
   E WF1    HistGBR            S1             0.020007
   V WF1    HistGBR            S2             0.084864
   E WF2    HistGBR            S1             0.027694
   V WF2    HistGBR            S2             0.016183
   E WF3      Ridge            S1             0.027171
   V WF3    HistGBR            S1             0.046914
```

```
 wf task model_type feature_block  inner_mean_spearman  n_train  n_test        r2  spearman      mae   pearson
WF1    E    HistGBR            S1             0.020007     2603    3160 -0.309978  0.014044 0.632981  0.021613
WF1    V    HistGBR            S2             0.084864     2603    3160 -0.111281 -0.008071 0.606834 -0.008681
WF2    E    HistGBR            S1             0.027694     5753    2847 -0.117424 -0.043375 0.612184 -0.026508
WF2    V    HistGBR            S2             0.016183     5753    2847 -0.053492  0.041039 0.589829  0.029191
WF3    E      Ridge            S1             0.027171     8607    3008 -0.013143  0.063566 0.546368  0.037679
WF3    V    HistGBR            S1             0.046914     8607    3008 -0.027072  0.044402 0.646648  0.055924
```

Spearman 为 Primary；R² 仅 secondary。

## 2. Quintile diagnostic（predicted Δ 五等分）

```
 wf task  quintile   n  mean_predicted  mean_realized  median_realized  share_realized_gt_0  q5_minus_q1  spearman_quintile_vs_realized
WF1    E         1 632       -0.595106      -0.064967              0.0             0.313291     0.015791                            0.4
WF1    E         2 632       -0.177076      -0.041687              0.0             0.310127     0.015791                            0.4
WF1    E         3 632        0.051583      -0.031218              0.0             0.351266     0.015791                            0.4
WF1    E         4 632        0.243316       0.067797              0.0             0.359177     0.015791                            0.4
WF1    E         5 632        0.564255      -0.049176              0.0             0.292722     0.015791                            0.4
WF1    V         1 632       -0.389658      -0.002260              0.0             0.295886    -0.025772                            0.1
WF1    V         2 632       -0.039031      -0.092563              0.0             0.156646    -0.025772                            0.1
WF1    V         3 632        0.092670      -0.097109              0.0             0.136076    -0.025772                            0.1
WF1    V         4 632        0.229076       0.048833              0.0             0.199367    -0.025772                            0.1
WF1    V         5 632        0.496874      -0.028032              0.0             0.243671    -0.025772                            0.1
WF2    E         1 570       -0.387009      -0.008251              0.0             0.354386    -0.090412                           -0.7
WF2    E         2 569       -0.135090       0.039957              0.0             0.383128    -0.090412                           -0.7
WF2    E         3 570       -0.008921       0.019043              0.0             0.347368    -0.090412                           -0.7
WF2    E         4 569        0.110551      -0.008398              0.0             0.344464    -0.090412                           -0.7
WF2    E         5 569        0.308572      -0.098663              0.0             0.293497    -0.090412                           -0.7
WF2    V         1 570       -0.381879      -0.049348              0.0             0.217544     0.074198                            0.9
WF2    V         2 569       -0.106578      -0.141091              0.0             0.168717     0.074198                            0.9
WF2    V         3 570        0.026106      -0.029240              0.0             0.168421     0.074198                            0.9
WF2    V         4 569        0.142927      -0.027031              0.0             0.212654     0.074198                            0.9
WF2    V         5 569        0.396319       0.024850              0.0             0.249561     0.074198                            0.9
WF3    E         1 602       -0.182075      -0.074733              0.0             0.284053     0.019038                            0.8
WF3    E         2 602       -0.084120      -0.095882              0.0             0.318937     0.019038                            0.8
WF3    E         3 601       -0.021317      -0.069607              0.0             0.327787     0.019038                            0.8
WF3    E         4 602        0.034390      -0.005793              0.0             0.440199     0.019038                            0.8
WF3    E         5 601        0.136996      -0.055694              0.0             0.362729     0.019038                            0.8
WF3    V         1 602       -0.407437      -0.230210              0.0             0.202658     0.236764                            0.0
WF3    V         2 602       -0.119474       0.007007              0.0             0.192691     0.236764                            0.0
WF3    V         3 601       -0.001416      -0.064659              0.0             0.186356     0.236764                            0.0
WF3    V         4 602        0.104655      -0.239119              0.0             0.199336     0.236764                            0.0
WF3    V         5 601        0.291244       0.006554              0.0             0.274542     0.236764                            0.0
```

## 3. Sign diagnostic

```
 wf task    n  predicted_positive_share  precision   recall  mean_realized_delta_given_pred_positive  median_realized_delta_given_pred_positive  baseline_positive_rate  overall_mean_delta  positive_precision_lift  positive_mean_delta_lift
WF1    E 3160                  0.554430   0.335046 0.571012                                -0.007548                                        0.0                0.325316           -0.023850                 0.009729                  0.016302
WF1    V 3160                  0.653797   0.186350 0.590491                                -0.023101                                        0.0                0.206329           -0.034226                -0.019979                  0.011125
WF2    E 2847                  0.484370   0.320522 0.450561                                -0.052173                                        0.0                0.344573           -0.011251                -0.024051                 -0.040923
WF2    V 2847                  0.543730   0.212532 0.568221                                -0.011568                                        0.0                0.203372           -0.044369                 0.009160                  0.032800
WF3    E 3008                  0.436503   0.398324 0.501438                                -0.024050                                        0.0                0.346742           -0.060340                 0.051582                  0.036290
WF3    V 3008                  0.500332   0.222591 0.527559                                -0.084060                                        0.0                0.211104           -0.104135                 0.011488                  0.020075
```

## 4. Bootstrap（1997+/2000，prediction 固定后重采样）

```
 wf task  bootstrap_n  spearman_median  spearman_ci_2p5  spearman_ci_97p5  predpos_mean_delta_median  predpos_mean_delta_ci_2p5  predpos_mean_delta_ci_97p5
WF1    E         2000         0.013262        -0.019809          0.048804                  -0.008060                  -0.041022                    0.025623
WF1    V         2000        -0.007569        -0.046395          0.029429                  -0.023454                  -0.065896                    0.019394
WF2    E         2000        -0.043232        -0.078189         -0.005044                  -0.050829                  -0.095196                   -0.009739
WF2    V         2000         0.041453         0.002782          0.078319                  -0.011494                  -0.063265                    0.036049
WF3    E         2000         0.063720         0.028098          0.099492                  -0.023856                  -0.065180                    0.017190
WF3    V         2000         0.044584         0.007171          0.081809                  -0.082407                  -0.144222                   -0.023462
```

## 5. LIMIT historical diagnostic（不训练、不进入 policy）

```
 wf  mean_delta_limit  share_delta_limit_gt_0  unique_oracle_count
WF1         -0.040190                0.003481                    0
WF2         -0.060414                0.006674                    0
WF3         -0.045213                0.006649                    0
```

## 6. Gates

```
task  A_ranking  B_positive_subset  C_precision_lift  passed
   E      False              False              True   False
   V      False              False              True   False
```

- `MARKET_ADVANTAGE_LEARNABLE` = **False**
- `VETO_ADVANTAGE_LEARNABLE` = **False**

## 7. Verdict
**NO_LEARNABLE_RELATIVE_ADVANTAGE**

## 8. Tests

```
 id                                           desc  passed
 T1                       n=9015 / WF counts exact    True
 T2                  frozen Market/Reassess parity    True
 T3 LIMIT redundant: reduced oracle == full oracle    True
 T4                       unique LIMIT oracle == 0    True
 T5                  delta_market exact arithmetic    True
 T6                 delta_skip == -reward_reassess    True
 T7                train/validation temporal order    True
 T8           purge exact (reward_end < val_start)    True
 T9                    train/test zero gid overlap    True
T10                   preprocessing fit train only    True
T11              no time decay (equal weight only)    True
T12          only 4 registered candidates per task    True
T13                     prediction reproducibility    True
T14                               P1_read == false    True
T15                   no policy outcome calculated    True
```


**STOP**：不进入 V3-A3，不构造 override policy。
