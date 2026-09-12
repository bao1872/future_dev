# Stage 4B — Latent-State Compression Gate v1.0

样本单位：**contact × decision_horizon**（不含 action/scale 重复）。Geometry 永不参与 PCA。

- state rows：`581,400`；join match rate = `1.0`（reaction/field/depletion 均须 1.0）
- Stage 4A 复现：cells=`144`，max|diff|=`0.0`

## Q1/Q2 — 有效维度（k80/k90/k95）

Reaction：

```
   block  wf  k80  k90  k95  k_primary    pc1    pc2    pc3
reaction WF1    5    7    8          6 0.3098 0.1864 0.1344
reaction WF2    5    7    8          6 0.3065 0.1841 0.1319
reaction WF3    5    7    8          6 0.3074 0.1829 0.1336
```

Liquidity：

```
    block  wf  k80  k90  k95  k_primary    pc1    pc2    pc3
liquidity WF1    4    5    6          5 0.3689 0.3347 0.0891
liquidity WF2    4    5    6          5 0.3689 0.3410 0.0882
liquidity WF3    3    5    6          5 0.3611 0.3508 0.0897
```

## Q3 — 跨 WF 稳定性（Hungarian 匹配，k=3）

```
   block wf_a wf_b  k  component_a  component_b  abs_cosine  max_principal_angle_deg  mean_principal_angle_deg
reaction  WF1  WF2  3            1            1    0.998822                 4.957166                  2.221612
reaction  WF1  WF2  3            2            2    0.994693                 4.957166                  2.221612
reaction  WF1  WF2  3            3            3    0.994762                 4.957166                  2.221612
reaction  WF1  WF2  3            0            0    0.996092                 4.957166                  2.221612
reaction  WF1  WF3  3            1            1    0.997170                 4.820841                  2.518380
reaction  WF1  WF3  3            2            2    0.990018                 4.820841                  2.518380
reaction  WF1  WF3  3            3            3    0.991560                 4.820841                  2.518380
reaction  WF1  WF3  3            0            0    0.992916                 4.820841                  2.518380
reaction  WF2  WF3  3            1            1    0.999572                 1.487979                  0.780457
reaction  WF2  WF3  3            2            2    0.999213                 1.487979                  0.780457
reaction  WF2  WF3  3            3            3    0.998919                 1.487979                  0.780457
reaction  WF2  WF3  3            0            0    0.999234                 1.487979                  0.780457
```

```
    block wf_a wf_b  k  component_a  component_b  abs_cosine  max_principal_angle_deg  mean_principal_angle_deg
liquidity  WF1  WF2  3            1            1    0.999620                59.056370                 20.249720
liquidity  WF1  WF2  3            2            2    0.999566                59.056370                 20.249720
liquidity  WF1  WF2  3            3            3    0.514012                59.056370                 20.249720
liquidity  WF1  WF2  3            0            0    0.837733                59.056370                 20.249720
liquidity  WF1  WF3  3            1            1    0.994530                57.189322                 19.690424
liquidity  WF1  WF3  3            2            2    0.994974                57.189322                 19.690424
liquidity  WF1  WF3  3            3            3    0.541872                57.189322                 19.690424
liquidity  WF1  WF3  3            0            0    0.843792                57.189322                 19.690424
liquidity  WF2  WF3  3            1            1    0.996567                 4.403752                  1.605500
liquidity  WF2  WF3  3            2            2    0.996456                 4.403752                  1.605500
liquidity  WF2  WF3  3            3            3    0.997044                 4.403752                  1.605500
liquidity  WF2  WF3  3            0            0    0.996689                 4.403752                  1.605500
```

## Q4/Q5 — 当前 action outcome：压缩后是否保留信息（target_first）

```
 wf       M0       M1       M2       M3       M4       M5
WF1 0.824154 0.824528 0.824335 0.825013 0.825445 0.826046
WF2 0.818233 0.818395 0.818474 0.819124 0.818798 0.819276
WF3 0.822429 0.823253 0.823014 0.823576 0.823670 0.824739
```

ΔAUC：

```
 wf  block  d_M1_M0  d_M2_M0   d_M1_M2  d_M3_M0  d_M4_M0   d_M3_M4  d_M5_M0
WF1 action 0.000374 0.000181  0.000193 0.000859 0.001291 -0.000431 0.001892
WF2 action 0.000162 0.000241 -0.000079 0.000891 0.000565  0.000326 0.001043
WF3 action 0.000824 0.000585  0.000239 0.001147 0.001241 -0.000094 0.002310
```

## Q6 — base state 能否解释 WAIT value

primary target = `delta_E_R_lower`（连续，未二值化）：

```
 wf       W0       W1       W2       W3       W4       W5
WF1 0.000731 0.001616 0.001295 0.000026 0.000727 0.001297
WF2 0.001221 0.002355 0.002038 0.001203 0.001116 0.001924
WF3 0.000798 0.001960 0.001492 0.000668 0.000739 0.001448
```

Spearman（primary target）：

```
 wf       W0       W1       W2       W3       W4       W5
WF1 0.140174 0.114028 0.119458 0.096914 0.121914 0.110434
WF2 0.127964 0.123598 0.125250 0.111402 0.117878 0.117404
WF3 0.138202 0.134822 0.131109 0.123930 0.131700 0.126213
```

Δ（R² / Spearman）：

```
 wf block  d_W1_W0_r2  d_W2_W0_r2  d_W1_W2_r2  d_W3_W0_r2  d_W4_W0_r2  d_W3_W4_r2  d_W5_W0_r2  d_W1_W0_spearman  d_W1_W2_spearman  d_W3_W0_spearman  d_W3_W4_spearman
WF1  wait    0.000885    0.000564    0.000321   -0.000704   -0.000003   -0.000701    0.000567         -0.026146         -0.005429         -0.043261         -0.025000
WF2  wait    0.001134    0.000816    0.000317   -0.000019   -0.000105    0.000087    0.000703         -0.004366         -0.001652         -0.016563         -0.006477
WF3  wait    0.001162    0.000695    0.000467   -0.000129   -0.000058   -0.000071    0.000650         -0.003380          0.003713         -0.014272         -0.007770
```

`delta_target_first` 仅作 group-level 汇总（不建模为单样本概率差）：

```
 wf      n  mean_delta_target_first  mean_delta_E_R_lower
WF1 598325                -0.067396             -0.125213
WF2 577549                -0.062400             -0.096958
WF3 567740                -0.061713             -0.106971
```

## Q7 — nonlinear compression gate

判定：**STOP_BEFORE_NONLINEAR_COMPRESSION**

```
         channel  raw_gain_wf  pca_loss_wf
 action_reaction            0            0
action_liquidity            0            0
   wait_reaction            0            0
  wait_liquidity            0            0
```

### 结论

- Q1 Reaction：k90 = [7, 7, 7]，k_primary = [6, 6, 6]。
- Q2 Liquidity：k90 = [5, 5, 5]，k_primary = [5, 5, 5]。
- Q3 稳定性：reaction mean matched |cos| = 0.996；liquidity = 0.893。
- Q4 reaction 压缩保留：raw ΔAUC 均值 +0.0005 vs PCA ΔAUC 均值 +0.0003。
- Q5 liquidity 压缩保留：raw ΔAUC 均值 +0.0010 vs PCA ΔAUC 均值 +0.0010。
- Q6 base state → WAIT value：delta_E_R_lower raw R² 增益均值 +0.00106，PCA 版 +0.00069。
- Q7 判定 STOP_BEFORE_NONLINEAR_COMPRESSION。

**DEFERRED**：`remaining_liq_imbalance_1p0` 未纳入 primary liquidity block——它需要新的 surviving-field imbalance 定义（decision-h 参考价 + 带权强度），不在 frozen snapshot 合同内，不在本轮自行发明公式。

**STOP**：本轮不进入 Autoencoder / Kernel PCA / UMAP / t-SNE / DP / RL。
