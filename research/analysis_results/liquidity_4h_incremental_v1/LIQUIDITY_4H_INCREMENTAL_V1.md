# V2-A 4h Liquidity Incremental Audit

**NO_MATERIAL_4H_INCREMENT**

4h levels: 484; preexisting exact-price share: 99.79%; active contact coverage: 95.55%.

## Action M3-M2

```
 wf   M2_AUC   M3_AUC  delta_AUC  delta_LogLoss  delta_Brier  secondary_delta_AUC   M0_AUC
WF1 0.826369 0.824075  -0.002293       0.003765     0.000390            -0.001736 0.824154
WF2 0.819583 0.819871   0.000288      -0.000226    -0.000092             0.000099 0.818233
WF3 0.825100 0.824951  -0.000150       0.000096     0.000083            -0.000237 0.822429
```

## WAIT W1-W0

```
 wf  n_train  n_test    W0_R2     W1_R2  delta_R2  W0_Spearman  W1_Spearman  delta_Spearman   W0_MAE   W1_MAE
WF1   464766  543295 0.000917 -0.030154 -0.031071     0.097504     0.083468       -0.014036 0.922868 0.943797
WF2  1008061  527629 0.002377  0.001564 -0.000813     0.116324     0.099872       -0.016452 0.930945 0.933358
WF3  1535690  514289 0.001866  0.001668 -0.000198     0.125357     0.119155       -0.006202 0.957851 0.955776
```

P1 and all policies were untouched.
