# TF15 — 15m Execution-Cycle Exploration v1

**不训练模型。** 目标：在 frozen liquidity/contact universe 上，只把 observation / reaction / geometry / execution clock 换成 15m，看是否存在独立稳定的 execution edge。

- 15m clean parents：`168,397`（disc parent：`1`）
- contacts：`96,900`，映射覆盖 `1.0000`
- action surface rows：`3,876,000`
- geometry cells：`4,474`；跨 WF regions：`1,120`；robust positive：`29`

## 1. Market frontier（4 个 scale 全量报告）

```
 scale  wf  signals  executed_trades  execution_rate  EV_censor_worst_per_signal  EV_censor_worst_per_executed_trade  EV_R_lower_filled  win_rate  profit_factor  mean_R_executed  median_R_executed     total_R  max_drawdown_R  max_drawdown_R_per_signal_sequence  ambiguity_rate  gap_invalid_rate
   0.2 WF1     4159             4095        0.984612                   -0.008184                           -0.008312           0.026533  0.638339       0.977017        -0.008312           0.375000  -34.037297      211.036699                          211.036699        0.004088          0.015388
   0.2 WF2     3584             3526        0.983817                    0.060420                            0.061414           0.080723  0.675553       1.189287         0.061414           0.416667  216.544023       44.287847                           44.287847        0.003069          0.016183
   0.2 WF3     3739             3692        0.987430                   -0.017681                           -0.017906           0.021090  0.631636       0.951390        -0.017906           0.391304  -66.109360      187.586419                          187.586419        0.001605          0.012570
   0.2 WF0     3255             3190        0.980031                   -0.059033                           -0.060236          -0.023184  0.608777       0.846032        -0.060236           0.362853 -192.151952      230.734420                          230.734420        0.006452          0.019969
   0.4 WF1     5614             5525        0.984147                   -0.027613                           -0.028058           0.018972  0.598552       0.930109        -0.028058           0.337838 -155.018435      287.362667                          287.362667        0.003206          0.015853
   0.4 WF2     5026             4913        0.977517                    0.014431                            0.014763           0.060977  0.622430       1.039099         0.014763           0.388889   72.529243      184.817222                          184.817222        0.002189          0.022483
   0.4 WF3     4948             4855        0.981205                   -0.043714                           -0.044551           0.011713  0.586818       0.892176        -0.044551           0.375000 -216.295808      261.035773                          261.035773        0.003234          0.018795
   0.4 WF0     4543             4452        0.979969                   -0.080500                           -0.082146          -0.031456  0.564016       0.811585        -0.082146           0.307692 -365.713321      397.964397                          397.964397        0.004622          0.020031
   0.8 WF1     5442             5331        0.979603                   -0.001094                           -0.001117           0.060133  0.546802       0.997536        -0.001117           0.316667   -5.953046      339.178377                          339.178377        0.004410          0.020397
   0.8 WF2     4999             4851        0.970394                    0.060676                            0.062527           0.082158  0.573078       1.146461         0.062527           0.375000  303.320556      126.161533                          126.161533        0.003801          0.029606
   0.8 WF3     4870             4783        0.982136                    0.026847                            0.027335           0.070765  0.546519       1.060278         0.027335           0.333333  130.742566      101.790902                          101.790902        0.002053          0.017864
   0.8 WF0     4308             4202        0.975395                   -0.028150                           -0.028860           0.017385  0.526416       0.939061        -0.028860           0.294118 -121.268918      177.828871                          177.828871        0.003714          0.024605
   1.2 WF1     6827             6678        0.978175                   -0.055681                           -0.056923           0.029400  0.515124       0.882603        -0.056923           0.269697 -380.132311      600.977308                          600.977308        0.001758          0.021825
   1.2 WF2     6626             6486        0.978871                   -0.027667                           -0.028264           0.048524  0.519118       0.941225        -0.028264           0.285714 -183.319620      341.186605                          341.186605        0.002113          0.021129
   1.2 WF3     6564             6417        0.977605                   -0.009458                           -0.009675           0.088917  0.510363       0.980241        -0.009675           0.243243  -62.081938      241.081565                          241.081565        0.000914          0.022395
   1.2 WF0     5832             5703        0.977881                   -0.141333                           -0.144530          -0.051012  0.464142       0.730284        -0.144530          -1.000000 -824.252025      850.591039                          850.591039        0.001715          0.022119
```

## 2. Gate

**STOP_15M_EXECUTION_LINE_FINAL**    passing scales = `[]`

## 3. Robust regions（完整）

```
 action  h  scale target_bin risk_bin  WF1_n  WF1_E_R_lower  WF1_ambiguity_gap_R  WF2_n  WF2_E_R_lower  WF2_ambiguity_gap_R  WF3_n  WF3_E_R_lower  WF3_ambiguity_gap_R  sample_gate  positive_3_of_3  ambiguity_gate  robust_positive_geometry
OUTWARD  2    0.2      0.5-1      1-2   1037       0.039136             0.003303    905       0.010691             0.001575   1063       0.033298             0.000000         True             True            True                      True
OUTWARD  2    0.4      0.5-1      1-2   1037       0.039136             0.003303    905       0.010691             0.001575   1063       0.033298             0.000000         True             True            True                      True
OUTWARD  2    0.8      0.5-1      1-2   1037       0.039136             0.003303    905       0.010691             0.001575   1063       0.033298             0.000000         True             True            True                      True
OUTWARD  2    1.2      0.5-1      1-2    975       0.036319             0.003510    823       0.014353             0.001733    943       0.071045             0.000000         True             True            True                      True
 INWARD  2    0.2      0.5-1      1-2    885       0.030913             0.013393    904       0.088994             0.015900    821       0.015764             0.002217         True             True            True                      True
 INWARD  2    0.4      0.5-1      1-2    885       0.030913             0.013393    904       0.088994             0.015900    821       0.015764             0.002217         True             True            True                      True
 INWARD  2    0.8      0.5-1      1-2    885       0.030913             0.013393    904       0.088994             0.015900    821       0.015764             0.002217         True             True            True                      True
 INWARD  2    0.8      0.5-1    0.5-1    764       0.103227             0.042967    681       0.125112             0.002054    662       0.052496             0.017291         True             True            True                      True
 INWARD  2    1.2      0.5-1      1-2    808       0.028351             0.006767    816       0.095143             0.017667    737       0.046511             0.002466         True             True            True                      True
OUTWARD  3    0.2       <0.5    0.5-1    735       0.029501             0.013592    455       0.074653             0.000000    569       0.026987             0.018171         True             True            True                      True
OUTWARD  3    0.2      0.5-1      1-2    965       0.028415             0.002904    901       0.075556             0.008534    907       0.009133             0.001907         True             True            True                      True
OUTWARD  3    0.4       <0.5    0.5-1    756       0.033721             0.013207    498       0.087455             0.000000    582       0.011341             0.035391         True             True            True                      True
OUTWARD  3    0.4      0.5-1      1-2    972       0.031782             0.002881    908       0.068880             0.008466    909       0.010602             0.001903         True             True            True                      True
OUTWARD  3    0.8      0.5-1      1-2   1115       0.011718             0.002538   1005       0.059069             0.009210    991       0.015227             0.001747         True             True            True                      True
OUTWARD  3    1.2        1-2 0.25-0.5    330       0.167239             0.000000    280       0.142245             0.000000    264       0.576268             0.025549         True             True            True                      True
OUTWARD  3    1.2      0.5-1      1-2   1253       0.008798             0.002227   1152       0.041024             0.006368   1216       0.028695             0.003613         True             True            True                      True
 INWARD  3    0.2      0.5-1      1-2    785       0.013839             0.002013    701       0.163800             0.000000    664       0.037195             0.000000         True             True            True                      True
 INWARD  3    0.4      0.5-1      1-2    799       0.018179             0.001977    701       0.163800             0.000000    671       0.041202             0.000000         True             True            True                      True
 INWARD  3    0.8      0.5-1      1-2    937       0.011649             0.008319    857       0.175373             0.001893    806       0.054433             0.002092         True             True            True                      True
 INWARD  3    0.8        1-2      2-3    294       0.183313             0.000000    221       0.024756             0.000000    212       0.153676             0.000000         True             True            True                      True
 INWARD  3    1.2      0.5-1    0.5-1    354       0.115500             0.005434    321       0.241563             0.006472    282       0.086452             0.000000         True             True            True                      True
 INWARD  5    0.4        1-2      1-2   1403       0.010732             0.001630   1325       0.014558             0.000000   1194       0.014213             0.001957         True             True            True                      True
 INWARD  5    0.8        1-2 0.25-0.5   1085       0.012216             0.006116   1122       0.141230             0.013939    955       0.306501             0.016781         True             True            True                      True
 INWARD  5    1.2      0.5-1    0.5-1    495       0.010876             0.004691    593       0.090603             0.010714    464       0.043445             0.000000         True             True            True                      True
 INWARD  5    1.2        1-2 0.25-0.5    433       0.318389             0.014503    425       0.073691             0.000000    402       0.124622             0.000000         True             True            True                      True
 INWARD  5    1.2        1-2      1-2   2601       0.011442             0.000903   2696       0.008736             0.000000   2611       0.054033             0.000935         True             True            True                      True
OUTWARD  8    0.4       <0.5    0.5-1    325       0.023684             0.000000    356       0.098062             0.000000    295       0.018113             0.018588         True             True            True                      True
OUTWARD  8    1.2        1-2      2-3    329       0.161838             0.000000    234       0.111402             0.000000    344       0.182292             0.000000         True             True            True                      True
 INWARD  8    1.2        1-2      2-3    361       0.013660             0.000000    321       0.046821             0.000000    296       0.179411             0.000000         True             True            True                      True
```

## 4. Tests

```
                                       test  pass
                  T2_ohlc_aggregation_exact  True
                    T3_trade_position_exact  True
              T4_availability_eq_parent_end  True
            T5_no_cross_session_aggregation  True
                     T6_atr15_prefix_causal  True
                T1_every_15m_bar_3_children  True
            T9_geometry_from_decision_close  True
             T12_actual_entry_next_15m_open  True
                   T13_gap_invalid_detected  True
            T15_same_child_ambiguous_bounds  True
                 T16_geometry_bins_complete  True
                    T17_no_5m_R1R4_imported  True
 T18_dual_conflict_skips_stage_allows_later  True
T19_no_duplicate_attempt_per_h_scale_action  True
                      T26_no_model_training  True
                            T27_p1_not_read  True
                  T20_one_trade_per_contact  True
                    T28_5m_freeze_unchanged  True
        T_disc_outcome_convention_fb_plus_1  True
          T_disc_activation_uses_bar_itself  True
               T8_decision_prefix_causality  True
              T10_structural_stop_causality  True
      T11_liquidity_target_depletion_parity  True
 T7_contact_visible_only_after_parent_close  True
       T14_outcome_window_le_36_child_slots  True
```


**STOP**：不自动挑 best scale；不进入 ML。
