# Liquidity-Field Reaction Model v1.0 — Stage 1–3 报告

**base**: `7107f16` &nbsp; **脚本**: `liquidity_field_reaction_model_v1.py`
**运行模式**: `FULL (96,900 contacts)`
**研究单位**: Liquidity Reaction Episode（接触时的流动性环境 + 接触后的路径演化）
**冻结 control**: Atlas v1.2 / Fixed-Exec Baseline / Tradeoff-Veto v1.3 / Risk-Coupled v1.0（后者 PAUSED，未运行）

---

## 1. 数据合同与无未来信息审计

- 活跃流动性来源：`smc_oracle_atlas_v1/liquidity_master_v1_1.parquet`，
  经 `active_mask(master_sym, decision_time)` 取 `available_time<=dt` 且
  `first_penetration_time` 为 NaN 或 `>dt` 的流动性 → **纯因果快照**。
- 每个 episode 在 `t0` 之后才看 bar；field snapshot 只含 `t<=t0` 可见流动性。
- 被接触流动性在 `dt` 已被 penetration（`fp<=dt`），天然不在活跃集；
  额外按 `liquidity_id` 在 C×L 矩阵中显式剔除（双保险）。
- `NO_FUTURE_LIQUIDITY_IN_FIELD_SNAPSHOT` = **PASS (active_mask excludes penetrated-by-dt; contacted liq explicitly popped in C×L matrix)**
- 审计计数：`n_active_contains_contacted=0`（应为 0）。

---

## 2. 流动性场数学画像（Stage 1）

- 样本量：`96900` episodes。
- 缺失率（部分 episode 周围无足够 active liquidity，属正常）：

             feature     mean   median     p05      p95  missing_rate
         room_up_atr   2.5747   1.9048  0.2632   6.9722        0.0101
       room_down_atr  59.7398  44.1935  5.3030 163.2603        0.0270
     field_width_atr  62.4372  46.7857  7.5556 166.6667        0.0371
      field_position   0.9155   0.9605  0.6607   0.9957        0.0371
     n_active_levels 194.9107 173.0000 53.0000 435.0000        0.0000
liq_intensity_up_0p5   0.3231   0.0377  0.0000   1.5675        0.0000
liq_intensity_up_1p0   0.8081   0.3324  0.0011   3.2537        0.0000
liq_intensity_up_2p0   2.0070   1.2727  0.0612   6.3938        0.0000
liq_intensity_up_4p0   4.9065   3.7475  0.5483  12.9601        0.0000
liq_intensity_dn_0p5   0.3108   0.0341  0.0000   1.5039        0.0000
liq_intensity_dn_1p0   0.7788   0.3112  0.0002   3.1358        0.0000
liq_intensity_dn_2p0   1.8888   1.1941  0.0258   6.0376        0.0000
liq_intensity_dn_4p0   4.3485   3.3421  0.3284  11.6300        0.0000
   liq_imbalance_0p5   0.0280   0.0775 -1.0000   1.0000        0.0000
   liq_imbalance_1p0   0.0283   0.0533 -0.9930   0.9984        0.0000
   liq_imbalance_2p0   0.0365   0.0472 -0.9111   0.9603        0.0000
   liq_imbalance_4p0   0.0597   0.0624 -0.7163   0.8415        0.0000

关键观察：field_position / field_width / room_up / room_down / 多尺度
liquidity intensity & imbalance 是否连续、是否覆盖 [0,1] 全范围。

**field_position 解读约束（P14）**：中位数 ≈0.96 主要反映 surviving
nearest-liquidity field 高度不对称——`room_down` 中位 44 ATR 远大于
`room_up` 中位 1.9 ATR，且该 ratio 被极远的 opposite-side liquidity 强烈
拉高。它**不等于**“价格已接近 target / RR 很差”，也**不能代替 RR**。
真实入场几何必须由 Stage 4 的 entry / structural stop / surviving target
联合计算；`liq_intensity λ=0.5/1/2/4` 的指数衰减（远端流动性快速衰减）
正是为了不让 44 ATR 外的 liquidity 污染局部场。

---

## 3. Reaction feature 分布（Stage 2, horizon=34）

- 样本量：`775200` (episode × horizon) 行；本表取 horizon=34。

               feature    mean  median     p05     p95  missing_rate
           cross_count  1.9132  1.0000  0.0000  6.0000        0.0000
    boundary_occupancy  0.0332  0.0000  0.0000  0.1471        0.0000
   boundary_touch_bars  6.2376  5.0000  0.0000 17.0000        0.0000
   total_variation_atr 14.3655 13.0620  5.7674 27.0004        0.0008
          net_move_atr  0.0324 -0.0641 -6.1111  6.4706        0.0000
       path_efficiency  0.1861  0.1600  0.0154  0.4485        0.0017
         amplitude_atr  4.7910  3.8636  1.4773 11.0417        0.0000
       outward_mfe_atr  3.0814  2.2500 -0.0943  8.9286        0.0000
        inward_mae_atr -2.6682 -1.9388 -8.3752  0.6897        0.0000
        dc_pivots_0p10 14.9281 15.0000 10.0000 20.0000        0.0000
        dc_pivots_0p20 12.6400 13.0000  7.0000 18.0000        0.0000
        dc_pivots_0p40  8.6536  9.0000  3.0000 14.0000        0.0000
        dc_pivots_0p80  4.8286  4.0000  1.0000  9.0000        0.0000
        dc_pivots_1p20  3.1460  3.0000  1.0000  7.0000        0.0000
           fp_out_0.25  1.8806  0.0000 -1.0000 14.0000        0.0000
           fp_out_0.50  2.6395  0.0000 -1.0000 17.0000        0.0000
           fp_out_1.00  3.8585  0.0000 -1.0000 22.0000        0.0000
            fp_in_0.25  1.7450  0.0000 -1.0000 14.0000        0.0000
            fp_in_0.50  2.4213  0.0000 -1.0000 17.0000        0.0000
            fp_in_1.00  3.8381  0.0000 -1.0000 22.0000        0.0000
     persistent_escape  0.6465  1.0000  0.0000  1.0000        0.0000
post_escape_efficiency  0.2144  0.1765  0.0156  0.5402        0.3535
     retracement_ratio  1.5401  0.7301  0.0000  5.6667        0.3537

> **escape 编码说明**：`persistent_escape` 为 0/1 的真实标记；
> `post_escape_efficiency` / `retracement_ratio` 仅在 `persistent_escape=1`
> 时有定义（未 escape 者为 NaN，缺失率 35.4%）。**仅 Stage 3 聚类输入**为
> 避免丢弃非 escape episode，将 NaN 填为 0（语义：“无 escape → 0 效率”），
> 不影响 `reaction_episode_features.csv` 原始列。下一阶段须严格保持
> `escaped=0/1` 且 `post_escape_efficiency=NaN if escaped==0`。

合成测试（先行）：`all_pass=True`；详情见 `reaction_synthetic_tests.json`。

---

## 4. Synthetic test 是否证明指标符合定义

{
  "A": {
    "efficiency": 1.0,
    "cross": 0,
    "amplitude": 0.6,
    "dc_0p40": 1,
    "dc_0p80": 1,
    "persistent_escape": false
  },
  "B": {
    "efficiency": 0.1266,
    "cross": 7,
    "amplitude": 0.25,
    "dc_0p40": 0,
    "dc_0p80": 0,
    "persistent_escape": false
  },
  "C": {
    "efficiency": 0.2836,
    "cross": 4,
    "amplitude": 1.5,
    "dc_0p40": 4,
    "dc_0p80": 1,
    "persistent_escape": true
  }
}
[
  {
    "check": "A_high_efficiency",
    "pass_": true
  },
  {
    "check": "B_low_efficiency",
    "pass_": true
  },
  {
    "check": "B_cross_gt_A_cross",
    "pass_": true
  },
  {
    "check": "C_medscale_gt_A_medscale",
    "pass_": true
  },
  {
    "check": "C_amplitude_gt_B_amplitude",
    "pass_": true
  },
  {
    "check": "A_low_complexity",
    "pass_": true
  },
  {
    "check": "C_persistent_escape",
    "pass_": true
  }
]

---

## 5. 数据是否真的存在稳定 cluster（Stage 3）

- 聚类方法：`RobustScaler->PCA->OPTICS (hdbscan unavailable)`（环境无 hdbscan，按规则改用 OPTICS，已记录）。
- 样本：`4000` / eligible `96739`。
- PCA 维度：`6`（累计解释方差 ≥ 90%）。
- BASE cluster 数：`9`；noise 比例：`0.939`。
- **cluster 稳定？** `False`。

### 稳定性审计（不同 min_samples / xi）

                      config  n_clusters  noise_fraction  n_components  arin_base
BASE(min_samples=15,xi=0.05)           9          0.9390             6     1.0000
      min_samples=10,xi=0.05          28          0.8845             6     0.3647
      min_samples=30,xi=0.05           2          0.7635             6     0.0241
      min_samples=15,xi=0.10           1          0.9940             6     0.1658

---

## 6. 若存在稳定 cluster：数学画像

 cluster  cross_count  boundary_occupancy  boundary_touch_bars  total_variation_atr  net_move_atr  path_efficiency  amplitude_atr  outward_mfe_atr  inward_mae_atr  dc_pivots_0p10  dc_pivots_0p20  dc_pivots_0p40  dc_pivots_0p80  dc_pivots_1p20  fp_out_0.25  fp_out_0.50  fp_out_1.00  fp_in_0.25  fp_in_0.50  fp_in_1.00  persistent_escape  post_escape_efficiency  retracement_ratio
      -1       1.9020              0.0336               6.2630              14.5095       -0.1190           0.1814         4.7999           3.0297         -2.7472         14.9478         12.6954          8.7396          4.9366          3.1720       1.4241       2.3594       3.5349       1.717      2.4728      3.8890             0.6464                  0.1324             1.0194
       0       2.2727              0.0267               7.2727              11.2923       -2.5185           0.2323         3.6372           1.4228         -3.0825         14.5909         11.3636          8.0909          4.5000          2.4545       0.0000      -0.0455      -0.3182       2.000      2.0909      3.5455             0.1364                  0.0174             0.1352
       1       1.4118              0.0086               4.1176              10.3667       -2.1151           0.2068         3.1149           0.6796         -3.1180         16.1176         13.1176          7.6471          3.2941          2.1176       0.0000      -0.2353      -0.8824       1.000      1.1765      3.1176             0.0000                  0.0000             0.0000
       2       0.0000              0.0000               0.4167               9.4683       -2.1661           0.2290         3.0507          -0.3029         -4.0722         13.7083         11.8333          6.6667          3.1667          2.1250      -1.0000      -1.0000      -1.0000       0.000      0.0000      0.3750             0.0000                  0.0000             0.0000
       3       0.8235              0.0086               2.7647              18.4076        5.2916           0.3052         6.8925           7.1228         -0.8697         15.2941         13.3529         10.4118          6.1176          3.1765       1.0000       1.2941       2.5294       0.000      0.0588     -0.7647             1.0000                  0.2501             0.0906
       4       0.0000              0.0000               0.0667               9.2543        2.2381           0.2492         2.8345           4.6576          0.7265         15.1333         11.6000          5.8667          2.6667          1.9333       0.0000       0.0000       0.0667      -1.000     -1.0000     -1.0000             1.0000                  0.2449             0.0301
       5       0.0000              0.0000               0.3143              17.0940        4.4221           0.2740         6.4408           8.4010          0.4800         15.9143         13.3714          9.2857          5.5143          3.6000       0.0000       0.0000       0.0857      -1.000     -1.0000     -1.0000             1.0000                  0.2729             0.1596
       6       2.1607              0.0331               6.6071              12.8943        0.9860           0.1359         3.9579           2.3683         -2.4911         14.2679         11.7679          7.9821          4.5714          3.2321      14.8750      15.4821      12.6429       0.000      0.0714      1.1786             0.5536                  0.1343             0.4614
       7       2.1111              0.0327               6.7500              13.0782        1.6735           0.1514         4.3839           2.5124         -2.8837         14.5556         11.8056          8.1389          4.7778          3.2222      21.0556      21.6389      17.4167       0.000      0.0556      1.2500             0.5556                  0.1874             0.3810
       8       1.6364              0.0214               5.4545              14.9562        2.3290           0.1841         4.9261           2.2986         -3.6893         16.0455         14.0455          9.7727          5.0455          3.1818      27.4545      27.8636      26.2727       0.000      0.0000      1.9545             0.5455                  0.2072             0.0782

---

## 7. cluster 与 liquidity field 的关系

 cluster  room_up_atr  room_down_atr  field_width_atr  field_position  n_active_levels  liq_intensity_up_0p5  liq_intensity_up_1p0  liq_intensity_up_2p0  liq_intensity_up_4p0  liq_intensity_dn_0p5  liq_intensity_dn_1p0  liq_intensity_dn_2p0  liq_intensity_dn_4p0  liq_imbalance_0p5  liq_imbalance_1p0  liq_imbalance_2p0  liq_imbalance_4p0
      -1       2.6107        59.4983          62.3399          0.9171         195.2705                0.3366                0.8315                2.0345                4.9078                0.2840                0.7227                1.7914                4.2184             0.0550             0.0545             0.0581             0.0728
       0       2.7486        53.5027          56.2326          0.9244         191.5000                0.3351                0.9650                2.2571                5.2054                0.5023                1.0302                2.1372                4.5569            -0.0660            -0.0189             0.0395             0.1145
       1       2.3791        56.1435          58.3627          0.9208         239.7647                0.1694                0.5737                1.7364                4.7297                0.1736                0.5610                1.6162                4.1711             0.0438             0.0751             0.1116             0.1228
       2       2.2912        53.5360          55.7683          0.8969         221.7083                0.2727                0.7150                1.8174                4.5996                0.2631                0.6701                1.6115                3.7759             0.1738             0.1280             0.1303             0.1547
       3       3.0348        77.0385          80.0732          0.9258         199.0000                0.2130                0.6101                1.5728                4.1373                0.2800                0.6859                1.5894                3.7942            -0.0463            -0.0210             0.0103             0.0615
       4       2.2178        33.8361          35.9604          0.8687         177.8000                0.2105                0.6373                1.9336                5.2621                0.1794                0.7461                2.1222                5.0846            -0.0058             0.0058             0.0063             0.0422
       5       2.3833        90.9838          92.6624          0.9163         215.3714                0.2608                0.7882                1.9820                4.6764                0.2376                0.6912                1.7876                4.0598            -0.0143            -0.0106            -0.0108             0.0045
       6       2.0359        47.6763          48.8941          0.9205         196.8393                0.4297                1.1739                2.8720                6.3797                0.1845                0.5871                1.8809                4.9349             0.1376             0.1281             0.1084             0.0945
       7       2.0334        50.0023          52.0358          0.9166         196.6667                0.3151                0.7939                1.9751                4.8604                0.4329                0.9823                2.0940                4.4101             0.0218             0.0019             0.0256             0.0700
       8       3.2327        58.6212          61.8539          0.9023         152.4091                0.3696                0.9924                2.2952                4.6647                0.2805                0.6539                1.5585                3.6364             0.0736             0.1166             0.1209             0.0900

> **cluster × field 关联约束（P14）**：本聚类结果本身不稳定（94% noise、
> 簇数在 1–28 间剧烈波动），因此 `cluster×field` 关联**没有稳定的左变量**。
> 仅能说“在该不稳定聚类下未观察到明显 field separation”，**不能**据此得出
> “reaction morphology 与 liquidity field 无关”。此问题仍开放——历史
> Opportunity 实验已证明两侧 liquidity geometry 信息很强。

---

## 8. 当前能得出的结论

- 流动性场是否连续可描述：见 §2。
- 接触后价格路径是否存在客观结构维度/自然类型：见 §3、§5。
- `cluster_stable=False` → 结论标记为 **NO_STABLE_DISCRETE_MORPHOLOGY_FOUND**。
  严格表述：在当前这些特征、样本与 OPTICS 参数下，**没有证据支持稳定的离散
  形态**；从研究 ROI 角度应改为按连续状态空间处理，而非继续寻找“5 种 /
  8 种形态”。这**不等于已经证明 morphology 在本质上是连续空间**（这是过头
  结论，禁止写）。

## 9. 当前不能得出的结论

- **禁止讨论“哪类最赚钱”**：Stage 3 输入不含 profit / E[R] / win/loss。
- 未进入 Stage 4–7（经济性 / 结构止损 / online 状态识别 / DP / WF 执行）。
- 未做 Risk-Coupled 0.5/1/2 正式实验（保持 PAUSED）。
