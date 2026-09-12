# V3-A1 — Execution Oracle / Regret Map

**No model is trained in this experiment.** 目的：测量在 frozen 四动作集合下，完美事后动作选择相对固定 RR3 Strict Reassess 还有多少理论收益空间。

- signals = `9015` (WF1/WF2/WF3 = 3160/2847/3008)
- primary reward = frozen **censor-worst** R
- T1–T12 全部 PASS；P1_read = `False`；sklearn = `False`

## 1. Headroom（核心表）

```
 wf  n_signals  EV_reassess  EV_oracle  oracle_headroom_per_signal  oracle_total_extra_R  reassess_capture_ratio  share_headroom_positive  median_headroom  p90_headroom  oracle_tie_rate
WF1       3160     0.034226   0.458748                    0.424522           1341.489668                0.074608                 0.530696         0.200000           1.0         0.471519
WF2       2847     0.044369   0.498699                    0.454331           1293.480059                0.088968                 0.545135         0.250000           1.0         0.459431
WF3       3008     0.104135   0.535571                    0.431436           1297.758175                0.194438                 0.554521         0.267857           1.0         0.445479
```

`reassess_capture_ratio` = EV_reassess / EV_oracle：固定 Reassess 已经拿走多少 theoretical action-selection value。

## 2. Reassess 相对 Oracle 的差距

```
 wf  EV_reassess  EV_oracle  headroom  captured_pct
WF1     0.034226   0.458748  0.424522      7.460800
WF2     0.044369   0.498699  0.454331      8.896848
WF3     0.104135   0.535571  0.431436     19.443762
```

## 3. Best-alternative advantage margin

`share(best_alt_advantage > margin)`：至少有一个 alternative 比 Reassess 好多少。

```
 wf  margin_R  share_best_alt_gt_margin  n_best_alt_gt_margin
WF1      0.00                  0.530696                  1677
WF1      0.05                  0.530380                  1676
WF1      0.10                  0.526266                  1663
WF1      0.25                  0.482595                  1525
WF1      0.50                  0.395570                  1250
WF1      1.00                  0.049684                   157
WF2      0.00                  0.545135                  1552
WF2      0.05                  0.544784                  1551
WF2      0.10                  0.538110                  1532
WF2      0.25                  0.495609                  1411
WF2      0.50                  0.395504                  1126
WF2      1.00                  0.070601                   201
WF3      0.00                  0.554521                  1668
WF3      0.05                  0.552527                  1662
WF3      0.10                  0.548537                  1650
WF3      0.25                  0.505652                  1521
WF3      0.50                  0.378657                  1139
WF3      1.00                  0.055851                   168
```

margin=0 即「至少有一个 alternative 严格优于 Reassess」的比例。

## 4. Per-action advantage vs Reassess

```
 wf    action  mean_delta  median_delta  share_delta_gt_0  share_delta_gt_0p05  share_delta_gt_0p10  share_delta_gt_0p25  share_delta_gt_0p50
WF1    MARKET   -0.023850           0.0          0.325316             0.325000             0.320886             0.277215             0.190190
WF1 LIMIT_RR3   -0.040190           0.0          0.003481             0.003481             0.003481             0.003481             0.003481
WF1      SKIP   -0.034226           0.0          0.206329             0.206329             0.206329             0.206329             0.206329
WF2    MARKET   -0.011251           0.0          0.343519             0.343168             0.336846             0.298209             0.194591
WF2 LIMIT_RR3   -0.060414           0.0          0.006674             0.006674             0.006674             0.006674             0.006674
WF2      SKIP   -0.044369           0.0          0.203372             0.203372             0.203372             0.203372             0.203372
WF3    MARKET   -0.060340           0.0          0.346742             0.344747             0.340758             0.298205             0.170878
WF3 LIMIT_RR3   -0.045213           0.0          0.006649             0.006649             0.006649             0.006649             0.006649
WF3      SKIP   -0.104135           0.0          0.211104             0.211104             0.211104             0.211104             0.211104
```

## 5. Oracle action membership（tie-aware）

```
 wf  n_signals  oracle_tie_rate  oracle_unique_rate  oracle_contains_SKIP_share  oracle_unique_SKIP_share  oracle_contains_MARKET_share  oracle_unique_MARKET_share  oracle_contains_LIMIT_RR3_share  oracle_unique_LIMIT_RR3_share  oracle_contains_REASSESS_RR3_share  oracle_unique_REASSESS_RR3_share  n_best_eq_1_share  n_best_eq_2_share  n_best_eq_3_share  n_best_eq_4_share
WF1       3160         0.471519            0.528481                    0.605380                  0.202848                      0.583544                    0.324684                         0.430696                            0.0                            0.469304                          0.000949           0.528481           0.092089           0.141456           0.237975
WF2       2847         0.459431            0.540569                    0.587285                  0.196698                      0.573586                    0.342466                         0.396558                            0.0                            0.454865                          0.001405           0.540569           0.118019           0.129961           0.211451
WF3       3008         0.445479            0.554521                    0.564827                  0.204455                      0.555186                    0.346410                         0.404588                            0.0                            0.445479                          0.003657           0.554521           0.105718           0.154920           0.184840
```

`contains_*` = 属于最优动作集合；`unique_*` = 唯一严格最优。未做 tie-break。

## 6. Bootstrap（paired, Oracle − Reassess）

```
 wf     mean  median_bootstrap   ci_2p5  ci_97p5
WF1 0.424522          0.424494 0.407122 0.441533
WF2 0.454331          0.454658 0.434569 0.474039
WF3 0.431436          0.431355 0.414051 0.448271
```

## 7. Region diagnostic（仅解释，不得据此造 region policy）

```
group    n  EV_reassess  EV_oracle  headroom  share_headroom_positive  oracle_contains_SKIP_share  oracle_contains_MARKET_share  oracle_contains_LIMIT_RR3_share  oracle_contains_REASSESS_RR3_share
   R1 1258    -0.000653   0.704719  0.705372                 0.655803                    0.739269                      0.282194                         0.319555                            0.344197
   R2 2919     0.059106   0.434764  0.375657                 0.500514                    0.589243                      0.623501                         0.440562                            0.499486
   R3 2164     0.047828   0.471198  0.423370                 0.552680                    0.539279                      0.633549                         0.398336                            0.447320
   R4 2674     0.101907   0.488090  0.386183                 0.529170                    0.548616                      0.598728                         0.432685                            0.470830
```

## 8. Symbol diagnostic（n ≥ 100，仅解释）

```
group    n  EV_reassess  EV_oracle  headroom  share_headroom_positive  oracle_contains_SKIP_share  oracle_contains_MARKET_share  oracle_contains_LIMIT_RR3_share  oracle_contains_REASSESS_RR3_share
   AG  418     0.060090   0.489176  0.429086                 0.535885                    0.622010                      0.495215                         0.430622                            0.464115
   AL  548     0.009854   0.486800  0.476946                 0.571168                    0.565693                      0.589416                         0.397810                            0.428832
   AU  336     0.207849   0.733837  0.525988                 0.589286                    0.562500                      0.511905                         0.395833                            0.410714
   CF  805     0.142547   0.555775  0.413228                 0.486957                    0.586335                      0.643478                         0.454658                            0.513043
   CU  507     0.010041   0.408531  0.398490                 0.548323                    0.552268                      0.585799                         0.428008                            0.451677
    I 1093     0.042543   0.415032  0.372488                 0.477585                    0.634035                      0.560842                         0.447392                            0.522415
    M  795     0.032013   0.464798  0.432786                 0.548428                    0.592453                      0.586164                         0.430189                            0.451572
   MA  587    -0.092277   0.350319  0.442596                 0.567291                    0.599659                      0.558773                         0.383305                            0.432709
   NI  381     0.123876   0.549702  0.425826                 0.538058                    0.595801                      0.530184                         0.422572                            0.461942
    P  473     0.230528   0.731623  0.501095                 0.583510                    0.475687                      0.632135                         0.389006                            0.416490
   RB 1043     0.033557   0.475136  0.441579                 0.592522                    0.536913                      0.593480                         0.344199                            0.407478
   RU  532     0.227444   0.663619  0.436175                 0.526316                    0.616541                      0.528195                         0.437970                            0.473684
   SC  412     0.024671   0.388084  0.363413                 0.475728                    0.669903                      0.487864                         0.468447                            0.524272
   SN  401     0.051747   0.510402  0.458655                 0.608479                    0.541147                      0.566085                         0.371571                            0.391521
   TA  684    -0.048859   0.454172  0.503031                 0.558480                    0.619883                      0.576023                         0.377193                            0.441520
```

## 9. Secondary：R_lower oracle（机制参考，非 Primary）

```
 wf  EV_reassess  EV_oracle_R_lower  headroom_R_lower_vs_reassess
WF1     0.034226           0.458748                      0.409649
WF2     0.044369           0.498699                      0.449062
WF3     0.104135           0.535571                      0.423124
```

## 10. Verdict

**ORACLE_HEADROOM_MATERIAL**

```
                        rule                                                              meaning
ORACLE_HEADROOM_INCONSISTENT max_wf_headroom / min_wf_headroom >= 2.0 (WFs disagree in magnitude)
         ORACLE_HEADROOM_LOW              max over WF of share(best_alt_advantage > 0.10R) < 0.10
    ORACLE_HEADROOM_MATERIAL           otherwise (all WF headroom > 0 and material share >= 0.10)
                        note           thresholds declared a priori; no economic tuning performed
       headroom_spread_ratio                                                   1.0702174330760044
```

## 11. 结论要点

- Q1/Q2/Q3 每 WF Reassess / Oracle / Headroom：见 §1、§2。
- Q4 Reassess 捕获比例：WF1=7.5%；WF2=8.9%；WF3=19.4%
- Q5 alternative > Reassess 的比例（margin=0）：WF1=0.531；WF2=0.545；WF3=0.555
- Q6 > 0.05 / 0.10 / 0.25 / 0.50R：见 §3 表。
- Q7 经常属于 oracle 的动作：见 §5。
- Q8 unique winner 分布：见 §5 的 `oracle_unique_*_share`。
- Q9 region / symbol：仅解释性异质性，禁止据此形成 policy。
- Q10 是否值得继续 Relative Advantage ML：由 §10 verdict 决定。

**STOP**：不自动进入 Relative Advantage ML。
