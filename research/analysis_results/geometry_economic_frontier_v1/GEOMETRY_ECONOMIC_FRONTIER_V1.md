# Geometry Economic Frontier v1.0
## 判定
**ROBUST_POSITIVE_GEOMETRY**
研究对象是每个 contact/action/scale 在预注册 horizon 中的第一次可交易 geometry。本实验没有训练模型、搜索阈值或加入 morphology/liquidity 特征。
## 预注册合同
- 每个 WF 最低样本量：`200`
- ambiguity gap 上限：`0.25R`
- 正期望要求：WF1、WF2、WF3 的 `E[R]lower` 均严格大于 0
- 主平面：target distance × risk distance；辅助平面：RR × target distance
- region 保留 action、first-available h、structure scale，不跨执行语义合并
## 样本与结果
- first-available rows：`559,841`
- 全量 WF cells：`6,637`
- 跨 WF regions：`2,459`
- 同时通过样本量与 ambiguity gate：`425`
- ROBUST_POSITIVE_GEOMETRY：主平面 `4`；辅助平面 `6`
两个平面的 region 会重叠，数量不可相加解释为独立 edge。结果是 gross、cell-level 描述，不含成本，也尚未构成可执行 selection policy。
## 通过的 regions（完整）
```
        plane  action  first_h  scale target_bin risk_bin   rr_bin  min_n_across_wf  min_E_R_lower_across_wf  max_ambiguity_gap_R_across_wf
target_x_risk OUTWARD        2    1.2        3-5      1-2      NaN              380                 0.035879                       0.000000
target_x_risk OUTWARD        5    1.2      0.5-1      1-2      NaN              372                 0.025776                       0.010731
target_x_risk  INWARD        5    1.2      0.5-1      1-2      NaN              227                 0.048212                       0.006667
target_x_risk OUTWARD        8    1.2      0.5-1      1-2      NaN              254                 0.019482                       0.020950
  rr_x_target  INWARD        2    0.2        2-3      NaN    1-1.5              204                 0.002302                       0.000000
  rr_x_target  INWARD        2    0.2      0.5-1      NaN 0.5-0.75              322                 0.023092                       0.019472
  rr_x_target  INWARD        2    0.4        2-3      NaN    1-1.5              200                 0.002302                       0.000000
  rr_x_target  INWARD        2    0.4      0.5-1      NaN 0.5-0.75              322                 0.023092                       0.019472
  rr_x_target  INWARD        2    0.8      0.5-1      NaN 0.5-0.75              312                 0.029093                       0.015158
  rr_x_target  INWARD        2    1.2      0.5-1      NaN 0.5-0.75              240                 0.020307                       0.012760
```
下一步允许进入预注册的 ENTER/SKIP selection experiment；本报告本身不做 selection。
