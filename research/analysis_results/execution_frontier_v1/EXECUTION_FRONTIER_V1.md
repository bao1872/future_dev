# Execution Semantics Repair v1

E0 is reclassified as **reference geometry**, not a live-executable backtest. R1-R4
membership observes the next open; E1 enters at the following full 5m bar open.

## Gate

**LAG1_EXECUTION_EDGE_SURVIVES** — `E2_LIMIT_FRONTIER_ALLOWED`

```
     execution        scope  wf  n_signals  n_trades  entry_invalid_n  resolved_n  censored_n  E_R_lower  E_R_upper  EV_R_lower_per_signal  EV_R_lower_censor_worst_per_signal  win_rate  profit_factor    total_R  max_drawdown_R  ambiguous_rate
E1_LAG1_MARKET WHOLE_POLICY WF1       3160      2792              368        2691         101   0.039563   0.047125               0.033691                            0.001729  0.564846       1.090917 106.463266      -87.935973        0.002848
E1_LAG1_MARKET WHOLE_POLICY WF2       2847      2565              282        2524          41   0.067116   0.070434               0.059501                            0.045100  0.543978       1.147177 169.400212      -92.444388        0.002107
E1_LAG1_MARKET WHOLE_POLICY WF3       3008      2687              321        2589          98   0.083372   0.096775               0.071759                            0.039179  0.571263       1.194460 215.851109      -68.385812        0.004987
E1_LAG1_MARKET           R1 WF1        465       445               20         380          65  -0.013750  -0.013750              -0.011236                           -0.151021  0.318421       0.979827  -5.224813      -55.608228        0.000000
E1_LAG1_MARKET           R1 WF2        380       364               16         337          27   0.387060   0.387060               0.343261                            0.272209  0.332344       1.579730 130.439330      -31.406255        0.000000
E1_LAG1_MARKET           R1 WF3        413       395               18         368          27   0.042740   0.042740               0.038083                           -0.027292  0.279891       1.059352  15.728323      -92.942172        0.000000
E1_LAG1_MARKET           R2 WF1       1044       864              180         842          22  -0.044309  -0.036767              -0.035736                           -0.056809  0.572447       0.896366 -37.308140      -77.646246        0.001916
E1_LAG1_MARKET           R2 WF2        952       832              120         831           1   0.028873   0.030919               0.025203                            0.024153  0.580024       1.068749  23.993316      -77.883869        0.001050
E1_LAG1_MARKET           R2 WF3        923       773              150         767           6   0.040636   0.055412               0.033768                            0.027268  0.619296       1.106740  31.167957      -55.240411        0.004334
E1_LAG1_MARKET           R3 WF1        720       649               71         641           8   0.155336   0.161577               0.138292                            0.127181  0.605304       1.393559  99.570548      -41.161905        0.002778
E1_LAG1_MARKET           R3 WF2        759       687               72         687           0   0.078964   0.080784               0.071474                            0.071474  0.611354       1.203178  54.248488      -44.065785        0.001318
E1_LAG1_MARKET           R3 WF3        685       614               71         607           7  -0.021193  -0.018722              -0.018780                           -0.028999  0.596376       0.947493 -12.864162      -39.599614        0.001460
E1_LAG1_MARKET           R4 WF1        931       834               97         828           6   0.059693   0.071770               0.053089                            0.046644  0.638889       1.165303  49.425672      -39.269205        0.005371
E1_LAG1_MARKET           R4 WF2        756       682               74         669          13  -0.058716  -0.050605              -0.051959                           -0.069155  0.536622       0.873287 -39.280922      -73.319925        0.005291
E1_LAG1_MARKET           R4 WF3        987       905               82         847          58   0.214662   0.240479               0.184214                            0.125450  0.636364       1.590321 181.818991      -46.476475        0.010132
```

P1 was not read. No signal threshold, region, feature, symbol, or time rule changed.
