# Execution Semantics Repair v1

E0 is reclassified as **reference geometry**, not a live-executable backtest. R1-R4
membership observes the next open; E1 enters at the following full 5m bar open.

## Gate

**LAG1_HARDENED_EXECUTION_EDGE_SURVIVES** — `E2_LIMIT_FRONTIER_ALLOWED`

```
     execution        scope  wf  n_signals  n_trades  nonentry_n  resolved_n  censored_n  E_R_lower  E_R_upper  EV_R_lower_per_signal  EV_R_lower_censor_worst_per_signal  win_rate  profit_factor    total_R  max_drawdown_R  ambiguous_rate
E1_LAG1_MARKET WHOLE_POLICY WF1       3160      2406         754        2306         100   0.057584   0.065650               0.042022                            0.010376  0.540763       1.125391 132.788702      -83.519379        0.002532
E1_LAG1_MARKET WHOLE_POLICY WF2       2847      2245         602        2204          41   0.061382   0.065183               0.047519                            0.033118  0.534483       1.131858 135.286334      -85.288917        0.002107
E1_LAG1_MARKET WHOLE_POLICY WF3       3008      2446         562        2348          98   0.097843   0.108008               0.076375                            0.043795  0.555366       1.220053 229.735402      -62.558226        0.003989
E1_LAG1_MARKET           R1 WF1        465       429          36         365          64   0.012946   0.012946               0.010162                           -0.127472  0.323288       1.019131   4.725377      -51.808228        0.000000
E1_LAG1_MARKET           R1 WF2        380       351          29         324          27   0.180368   0.180368               0.153788                            0.082735  0.330247       1.269306  58.439330      -31.028788        0.000000
E1_LAG1_MARKET           R1 WF3        413       383          30         356          27   0.077889   0.077889               0.067139                            0.001763  0.289326       1.109598  27.728323      -80.942172        0.000000
E1_LAG1_MARKET           R2 WF1       1044       726         318         704          22  -0.051685  -0.045151              -0.034853                           -0.055926  0.544034       0.886647 -36.386454      -79.880767        0.000958
E1_LAG1_MARKET           R2 WF2        952       698         254         697           1   0.064222   0.066661               0.047020                            0.045969  0.575323       1.151226  44.762749      -60.674241        0.001050
E1_LAG1_MARKET           R2 WF3        923       687         236         681           6   0.049111   0.053516               0.036234                            0.029734  0.604993       1.124329  33.444368      -57.885508        0.002167
E1_LAG1_MARKET           R3 WF1        720       548         172         540           8   0.218917   0.226325               0.164188                            0.153077  0.601852       1.549839 118.215357      -39.428571        0.002778
E1_LAG1_MARKET           R3 WF2        759       604         155         604           0   0.115500   0.117570               0.091913                            0.091913  0.617550       1.302000  69.762035      -43.505417        0.001318
E1_LAG1_MARKET           R3 WF3        685       539         146         532           7  -0.034059  -0.031239              -0.026451                           -0.036670  0.562030       0.922235 -18.119186      -38.659598        0.001460
E1_LAG1_MARKET           R4 WF1        931       703         228         697           6   0.066333   0.080681               0.049661                            0.043216  0.604017       1.167516  46.234421      -39.274553        0.005371
E1_LAG1_MARKET           R4 WF2        756       592         164         579          13  -0.065074  -0.055702              -0.049838                           -0.067034  0.512953       0.866391 -37.677780      -71.969554        0.005291
E1_LAG1_MARKET           R4 WF3        987       837         150         779          58   0.239643   0.264504               0.189141                            0.130377  0.629012       1.645958 186.681897      -47.880752        0.009119
```

P1 was not read. No signal threshold, region, feature, symbol, or time rule changed.
