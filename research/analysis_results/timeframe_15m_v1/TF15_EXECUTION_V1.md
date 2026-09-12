# TF15 — 15m Execution-Cycle Exploration v1

**不训练模型。** 目标：在 frozen liquidity/contact universe 上，只把 observation / reaction / geometry / execution clock 换成 15m，看是否存在独立稳定的 execution edge。

- 15m clean parents：`168,397`（disc parent：`1`）
- contacts：`96,900`，映射覆盖 `1.0000`
- action surface rows：`3,876,000`
- geometry cells：`4,474`；跨 WF regions：`1,120`；robust positive：`29`

## 1. Market frontier（4 个 scale 全量报告）

```
 scale  wf  signals  executed_trades  execution_rate  EV_censor_worst_per_signal  EV_R_lower_filled  win_rate  profit_factor    mean_R  median_R     total_R  max_drawdown_R  ambiguity_rate  gap_invalid_rate
   0.2 WF1     4096             4096             1.0                   -0.008554           0.026273  0.638184       0.976358 -0.008554  0.375000  -35.037297       83.354598        0.004150               0.0
   0.2 WF2     3528             3528             1.0                    0.061202           0.080496  0.675454       1.188576  0.061202  0.416667  215.919023       29.141675        0.003118               0.0
   0.2 WF3     3692             3692             1.0                   -0.017906           0.021090  0.631636       0.951390 -0.017906  0.391304  -66.109360      108.903230        0.001625               0.0
   0.2 WF0     3193             3193             1.0                   -0.060582          -0.023580  0.608519       0.845250 -0.060582  0.362069 -193.437667      196.586189        0.006577               0.0
   0.4 WF1     5531             5531             1.0                   -0.027765           0.019225  0.598626       0.930825 -0.027765  0.337838 -153.568435      214.644818        0.003254               0.0
   0.4 WF2     4923             4923             1.0                    0.014555           0.060661  0.622182       1.038524  0.014555  0.388889   71.654243      108.754044        0.002234               0.0
   0.4 WF3     4855             4855             1.0                   -0.044551           0.011713  0.586818       0.892176 -0.044551  0.375000 -216.295808      217.249573        0.003296               0.0
   0.4 WF0     4468             4468             1.0                   -0.078910          -0.028234  0.564682       0.818730 -0.078910  0.310172 -352.570464      357.540858        0.004700               0.0
   0.8 WF1     5335             5335             1.0                   -0.001303           0.060097  0.546767       0.997124 -0.001303  0.316667   -6.953046      170.225789        0.004499               0.0
   0.8 WF2     4859             4859             1.0                    0.061678           0.081261  0.572752       1.144362  0.061678  0.375000  299.695556       73.582979        0.003910               0.0
   0.8 WF3     4784             4784             1.0                    0.027451           0.070877  0.546614       1.060548  0.027451  0.333333  131.327668       94.591766        0.002090               0.0
   0.8 WF0     4205             4205             1.0                   -0.029145           0.017052  0.526278       0.938477 -0.029145  0.294118 -122.554633      242.944963        0.003805               0.0
   1.2 WF1     6695             6695             1.0                   -0.055758           0.030768  0.515758       0.884855 -0.055758  0.272727 -373.301416      376.572342        0.001792               0.0
   1.2 WF2     6501             6501             1.0                   -0.028827           0.047726  0.518843       0.940088 -0.028827  0.285714 -187.403767      240.782639        0.002154               0.0
   1.2 WF3     6428             6428             1.0                   -0.008671           0.089835  0.511201       0.982261 -0.008671  0.249996  -55.736090      146.447773        0.000933               0.0
   1.2 WF0     5735             5735             1.0                   -0.139989          -0.046556  0.466085       0.737807 -0.139989 -1.000000 -802.836100      809.475748        0.001744               0.0
```

## 2. Gate

**STOP_15M_EXECUTION_LINE**    passing scales = `[]`

## 3. Robust regions（完整）

```
 action  h  scale target_bin risk_bin  WF1_n  WF1_E_R_lower  WF1_ambiguity_gap_R  WF2_n  WF2_E_R_lower  WF2_ambiguity_gap_R  WF3_n  WF3_E_R_lower  WF3_ambiguity_gap_R  sample_gate  positive_3_of_3  ambiguity_gate  robust_positive_geometry
OUTWARD  2    0.2      0.5-1      1-2   1022       0.039136             0.003303    897       0.010691             0.001575   1048       0.033298             0.000000         True             True            True                      True
OUTWARD  2    0.4      0.5-1      1-2   1022       0.039136             0.003303    897       0.010691             0.001575   1048       0.033298             0.000000         True             True            True                      True
OUTWARD  2    0.8      0.5-1      1-2   1022       0.039136             0.003303    897       0.010691             0.001575   1048       0.033298             0.000000         True             True            True                      True
OUTWARD  2    1.2      0.5-1      1-2    961       0.036319             0.003510    815       0.014353             0.001733    929       0.071045             0.000000         True             True            True                      True
 INWARD  2    0.2      0.5-1      1-2    874       0.030913             0.013393    886       0.088994             0.015900    817       0.015764             0.002217         True             True            True                      True
 INWARD  2    0.4      0.5-1      1-2    874       0.030913             0.013393    886       0.088994             0.015900    817       0.015764             0.002217         True             True            True                      True
 INWARD  2    0.8      0.5-1      1-2    874       0.030913             0.013393    886       0.088994             0.015900    817       0.015764             0.002217         True             True            True                      True
 INWARD  2    0.8      0.5-1    0.5-1    755       0.103227             0.042967    653       0.125112             0.002054    654       0.052496             0.017291         True             True            True                      True
 INWARD  2    1.2      0.5-1      1-2    798       0.028351             0.006767    799       0.095143             0.017667    733       0.046511             0.002466         True             True            True                      True
OUTWARD  3    0.2       <0.5    0.5-1    721       0.029501             0.013592    440       0.074653             0.000000    559       0.026987             0.018171         True             True            True                      True
OUTWARD  3    0.2      0.5-1      1-2    950       0.028415             0.002904    886       0.075556             0.008534    895       0.009133             0.001907         True             True            True                      True
OUTWARD  3    0.4       <0.5    0.5-1    742       0.033721             0.013207    483       0.087455             0.000000    572       0.011341             0.035391         True             True            True                      True
OUTWARD  3    0.4      0.5-1      1-2    957       0.031782             0.002881    893       0.068880             0.008466    897       0.010602             0.001903         True             True            True                      True
OUTWARD  3    0.8      0.5-1      1-2   1099       0.011718             0.002538    988       0.059069             0.009210    979       0.015227             0.001747         True             True            True                      True
OUTWARD  3    1.2        1-2 0.25-0.5    323       0.167239             0.000000    279       0.142245             0.000000    260       0.576268             0.025549         True             True            True                      True
OUTWARD  3    1.2      0.5-1      1-2   1234       0.008798             0.002227   1139       0.041024             0.006368   1204       0.028695             0.003613         True             True            True                      True
 INWARD  3    0.2      0.5-1      1-2    773       0.013839             0.002013    695       0.163800             0.000000    656       0.037195             0.000000         True             True            True                      True
 INWARD  3    0.4      0.5-1      1-2    787       0.018179             0.001977    695       0.163800             0.000000    662       0.041202             0.000000         True             True            True                      True
 INWARD  3    0.8      0.5-1      1-2    925       0.011649             0.008319    851       0.175373             0.001893    796       0.054433             0.002092         True             True            True                      True
 INWARD  3    0.8        1-2      2-3    289       0.183313             0.000000    217       0.024756             0.000000    210       0.153676             0.000000         True             True            True                      True
 INWARD  3    1.2      0.5-1    0.5-1    350       0.115500             0.005434    312       0.241563             0.006472    270       0.086452             0.000000         True             True            True                      True
 INWARD  5    0.4        1-2      1-2   1373       0.010732             0.001630   1286       0.014558             0.000000   1168       0.014213             0.001957         True             True            True                      True
 INWARD  5    0.8        1-2 0.25-0.5   1030       0.012216             0.006116   1036       0.141230             0.013939    886       0.306501             0.016781         True             True            True                      True
 INWARD  5    1.2      0.5-1    0.5-1    473       0.010876             0.004691    573       0.090603             0.010714    434       0.043445             0.000000         True             True            True                      True
 INWARD  5    1.2        1-2 0.25-0.5    396       0.318389             0.014503    408       0.073691             0.000000    383       0.124622             0.000000         True             True            True                      True
 INWARD  5    1.2        1-2      1-2   2543       0.011442             0.000903   2631       0.008736             0.000000   2535       0.054033             0.000935         True             True            True                      True
OUTWARD  8    0.4       <0.5    0.5-1    318       0.023684             0.000000    333       0.098062             0.000000    269       0.018113             0.018588         True             True            True                      True
OUTWARD  8    1.2        1-2      2-3    324       0.161838             0.000000    233       0.111402             0.000000    344       0.182292             0.000000         True             True            True                      True
 INWARD  8    1.2        1-2      2-3    352       0.013660             0.000000    321       0.046821             0.000000    296       0.179411             0.000000         True             True            True                      True
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
       T14_outcome_window_le_36_child_slots  True
            T15_same_child_ambiguous_bounds  True
                 T16_geometry_bins_complete  True
                    T17_no_5m_R1R4_imported  True
 T18_dual_conflict_skips_stage_allows_later  True
T19_no_duplicate_attempt_per_h_scale_action  True
                      T26_no_model_training  True
                            T27_p1_not_read  True
                  T20_one_trade_per_contact  True
                    T28_5m_freeze_unchanged  True
```


**STOP**：不自动挑 best scale；不进入 ML。
