# Liquidity-Field Sequential Action Surface v1.0 — Stage 4A 报告

**base**: `7107f16` &nbsp; **脚本**: `run_liquidity_field_action_surface_v1.py`
**运行模式**: `FULL (all contacts)`
**研究单位**: 同一 t0 流动性场下，entry / 结构止损 / surviving target 的联合经济面
**冻结 control**: Atlas v1.2 / Fixed-Exec Baseline / Tradeoff-Veto v1.3 / Risk-Coupled v1.0（PAUSED）

## 0. 数据合同与治理

- t0 流动性场冻结于 `active_mask(master_sym, decision_time)`（`available_time<=dt
  AND NOT(fp<=dt)`）；被接触流动性已因 `fp<=dt` 被排除，**不额外按 liquidity_id 剔除**
  （P1.1 将验证这一语义与 frozen Atlas 一致）。
- 价格运动过程中 t0 liquidity 可被 consumed（causal depletion）；本阶段不新增
  contact 后生成的 liquidity。
- **stop 语义（P0 已审计 PASS）**：`stop = 最近一个 δ-confirmed 结构极值`，
  `δ` 只是结构确认尺度，不是固定 risk distance；`risk_atr = |entry - stop|/atr0`
  一般 ≠ δ（中位数约 0.5–1.2 ATR，随 δ 单调但不相等）。
- 同 bar target+stop 用双边界：`R_lower=-1`（ambiguous）、`R_upper=+RR`；
  censored 单独统计，不填成 loss。
- **ROI Gate**：跑完先 STOP，不训练 classifier / DP / RL。

## 1. 小样本 gate（synthetic T1-T7）

`all_pass=True`；详情见 `action_surface_synthetic_tests.json`。

## 2. Action surface by WF（第一张表，节选前 20 行）

 wf  h  structure_scale  action  n_contacts  n_action_available  availability  median_target_atr  median_risk_atr  median_RR  target_first_rate  stop_first_rate  same_bar_ambiguous_rate  censored_rate  E_R_lower  E_R_upper
WF1  1              0.2 OUTWARD       25383                   0           0.0                NaN              NaN        NaN                NaN              NaN                      NaN            NaN        NaN        NaN
WF2  1              0.2 OUTWARD       24692                   0           0.0                NaN              NaN        NaN                NaN              NaN                      NaN            NaN        NaN        NaN
WF3  1              0.2 OUTWARD       23810                   0           0.0                NaN              NaN        NaN                NaN              NaN                      NaN            NaN        NaN        NaN
WF1  1              0.4 OUTWARD       25383                   0           0.0                NaN              NaN        NaN                NaN              NaN                      NaN            NaN        NaN        NaN
WF2  1              0.4 OUTWARD       24692                   0           0.0                NaN              NaN        NaN                NaN              NaN                      NaN            NaN        NaN        NaN
WF3  1              0.4 OUTWARD       23810                   0           0.0                NaN              NaN        NaN                NaN              NaN                      NaN            NaN        NaN        NaN
WF1  1              0.8 OUTWARD       25383                   0           0.0                NaN              NaN        NaN                NaN              NaN                      NaN            NaN        NaN        NaN
WF2  1              0.8 OUTWARD       24692                   0           0.0                NaN              NaN        NaN                NaN              NaN                      NaN            NaN        NaN        NaN
WF3  1              0.8 OUTWARD       23810                   0           0.0                NaN              NaN        NaN                NaN              NaN                      NaN            NaN        NaN        NaN
WF1  1              1.2 OUTWARD       25383                   0           0.0                NaN              NaN        NaN                NaN              NaN                      NaN            NaN        NaN        NaN
WF2  1              1.2 OUTWARD       24692                   0           0.0                NaN              NaN        NaN                NaN              NaN                      NaN            NaN        NaN        NaN
WF3  1              1.2 OUTWARD       23810                   0           0.0                NaN              NaN        NaN                NaN              NaN                      NaN            NaN        NaN        NaN
WF1  1              0.2  INWARD       25383                   0           0.0                NaN              NaN        NaN                NaN              NaN                      NaN            NaN        NaN        NaN
WF2  1              0.2  INWARD       24692                   0           0.0                NaN              NaN        NaN                NaN              NaN                      NaN            NaN        NaN        NaN
WF3  1              0.2  INWARD       23810                   0           0.0                NaN              NaN        NaN                NaN              NaN                      NaN            NaN        NaN        NaN
WF1  1              0.4  INWARD       25383                   0           0.0                NaN              NaN        NaN                NaN              NaN                      NaN            NaN        NaN        NaN
WF2  1              0.4  INWARD       24692                   0           0.0                NaN              NaN        NaN                NaN              NaN                      NaN            NaN        NaN        NaN
WF3  1              0.4  INWARD       23810                   0           0.0                NaN              NaN        NaN                NaN              NaN                      NaN            NaN        NaN        NaN
WF1  1              0.8  INWARD       25383                   0           0.0                NaN              NaN        NaN                NaN              NaN                      NaN            NaN        NaN        NaN
WF2  1              0.8  INWARD       24692                   0           0.0                NaN              NaN        NaN                NaN              NaN                      NaN            NaN        NaN        NaN

## 3. Value of Waiting — P12 Primary（首个 available horizon 配对，节选前 20 行）

> 基线 = 该 contact/action/scale 在预注册 horizon 中**首次 available** 的时点 h0，
> 而不是 h=1（单根 K 线无法形成 confirmed structural extreme）。h=1 availability=0
> 是结构形成需要时间的机制证据。

 wf  action  structure_scale  base_h  later_h  matched_n  median_delta_entry_atr  median_delta_target_distance_atr  median_delta_risk_distance_atr  median_delta_RR  target_first_base  target_first_later  delta_target_first  E_R_lower_base  E_R_lower_later  delta_E_R_lower  E_R_upper_base  E_R_upper_later  delta_E_R_upper
WF1 OUTWARD              0.2       2        3      20182                     0.0                          0.000000                        0.000000         0.206349           0.241056            0.207611           -0.033446       -0.295255        -0.377887        -0.072875       -0.247165        -0.334595        -0.073948
WF2 OUTWARD              0.2       2        3      19571                     0.0                          0.000000                        0.000000         0.194444           0.239436            0.217158           -0.022278       -0.287564        -0.344043        -0.047807       -0.226842        -0.317973        -0.059890
WF3 OUTWARD              0.2       2        3      19778                     0.0                          0.059173                        0.000000         0.250000           0.231773            0.209273           -0.022500       -0.278699        -0.296265        -0.010947       -0.240167        -0.262183        -0.013274
WF1 OUTWARD              0.4       2        3      18201                     0.0                          0.056818                        0.000000         0.238095           0.264051            0.219384           -0.044668       -0.242886        -0.337803        -0.083161       -0.201911        -0.297843        -0.080205
WF2 OUTWARD              0.4       2        3      17408                     0.0                          0.000000                        0.000000         0.208333           0.262810            0.229952           -0.032858       -0.263316        -0.315071        -0.039916       -0.207219        -0.288345        -0.045935
WF3 OUTWARD              0.4       2        3      17400                     0.0                          0.064935                        0.000000         0.225083           0.259828            0.225460           -0.034368       -0.213937        -0.253264        -0.030572       -0.177146        -0.227581        -0.041047
WF1 OUTWARD              0.8       2        3      12101                     0.0                          0.142863                       -0.119048         0.392857           0.329477            0.264854           -0.064623       -0.162636        -0.248813        -0.064744       -0.141849        -0.230013        -0.066264
WF2 OUTWARD              0.8       2        3      11180                     0.0                          0.083333                       -0.101010         0.323684           0.343470            0.275760           -0.067710       -0.111873        -0.214965        -0.092662       -0.087822        -0.194929        -0.095285
WF3 OUTWARD              0.8       2        3      11095                     0.0                          0.107527                       -0.132898         0.430769           0.334205            0.271924           -0.062280       -0.105251        -0.157892        -0.038787       -0.091214        -0.145501        -0.039443
WF1 OUTWARD              1.2       2        3       6820                     0.0                          0.204082                       -0.200000         0.422876           0.397361            0.307918           -0.089443       -0.096554        -0.198520        -0.076333       -0.091687        -0.189101        -0.071649
WF2 OUTWARD              1.2       2        3       6712                     0.0                          0.147059                       -0.156250         0.324108           0.386770            0.314213           -0.072557       -0.113951        -0.184116        -0.057127       -0.086546        -0.172957        -0.072222
WF3 OUTWARD              1.2       2        3       6224                     0.0                          0.191815                       -0.226840         0.400000           0.385443            0.320855           -0.064589       -0.086618        -0.090459         0.017278       -0.074513        -0.081779         0.013887
WF1  INWARD              0.2       2        3      21032                     0.0                          0.000000                        0.000000         0.116429           0.183577            0.162847           -0.020730       -0.321403        -0.354559        -0.021885       -0.303750        -0.329675        -0.013985
WF2  INWARD              0.2       2        3      20839                     0.0                          0.000000                        0.000000         0.234698           0.177024            0.157397           -0.019627       -0.335936        -0.368722        -0.028484       -0.315967        -0.359244        -0.039148
WF3  INWARD              0.2       2        3      20476                     0.0                          0.000000                        0.000000         0.050000           0.171811            0.164143           -0.007668       -0.301498        -0.325121        -0.016266       -0.275529        -0.310943        -0.014949
WF1  INWARD              0.4       2        3      19346                     0.0                          0.000000                        0.000000         0.100000           0.198387            0.170061           -0.028326       -0.272434        -0.328987        -0.041819       -0.255782        -0.302377        -0.031018
WF2  INWARD              0.4       2        3      18867                     0.0                          0.000000                        0.000000         0.166667           0.194891            0.171357           -0.023533       -0.271595        -0.313746        -0.035233       -0.254284        -0.303892        -0.042706
WF3  INWARD              0.4       2        3      18517                     0.0                          0.000000                        0.000000         0.107692           0.189178            0.175892           -0.013285       -0.255187        -0.297681        -0.028844       -0.227010        -0.286269        -0.031089
WF1  INWARD              0.8       2        3      13521                     0.0                          0.000000                        0.000000         0.142857           0.254419            0.216774           -0.037645       -0.137554        -0.236261        -0.079516       -0.126847        -0.232215        -0.086188
WF2  INWARD              0.8       2        3      12989                     0.0                          0.000000                       -0.065359         0.189565           0.245207            0.204866           -0.040342       -0.176024        -0.254322        -0.068658       -0.167073        -0.246348        -0.069532

## 3b. Value of Waiting — P12 Secondary（固定 h=2 cohort 配对，节选前 20 行）

 wf  action  structure_scale  base_h  later_h  matched_n  median_delta_entry_atr  median_delta_target_distance_atr  median_delta_risk_distance_atr  median_delta_RR  target_first_base  target_first_later  delta_target_first  E_R_lower_base  E_R_lower_later  delta_E_R_lower  E_R_upper_base  E_R_upper_later  delta_E_R_upper
WF1 OUTWARD              0.2       2        3      20182                     0.0                          0.000000                        0.000000         0.206349           0.241056            0.207611           -0.033446       -0.295255        -0.377887        -0.072875       -0.247165        -0.334595        -0.073948
WF2 OUTWARD              0.2       2        3      19571                     0.0                          0.000000                        0.000000         0.194444           0.239436            0.217158           -0.022278       -0.287564        -0.344043        -0.047807       -0.226842        -0.317973        -0.059890
WF3 OUTWARD              0.2       2        3      19778                     0.0                          0.059173                        0.000000         0.250000           0.231773            0.209273           -0.022500       -0.278699        -0.296265        -0.010947       -0.240167        -0.262183        -0.013274
WF1 OUTWARD              0.4       2        3      18201                     0.0                          0.056818                        0.000000         0.238095           0.264051            0.219384           -0.044668       -0.242886        -0.337803        -0.083161       -0.201911        -0.297843        -0.080205
WF2 OUTWARD              0.4       2        3      17408                     0.0                          0.000000                        0.000000         0.208333           0.262810            0.229952           -0.032858       -0.263316        -0.315071        -0.039916       -0.207219        -0.288345        -0.045935
WF3 OUTWARD              0.4       2        3      17400                     0.0                          0.064935                        0.000000         0.225083           0.259828            0.225460           -0.034368       -0.213937        -0.253264        -0.030572       -0.177146        -0.227581        -0.041047
WF1 OUTWARD              0.8       2        3      12101                     0.0                          0.142863                       -0.119048         0.392857           0.329477            0.264854           -0.064623       -0.162636        -0.248813        -0.064744       -0.141849        -0.230013        -0.066264
WF2 OUTWARD              0.8       2        3      11180                     0.0                          0.083333                       -0.101010         0.323684           0.343470            0.275760           -0.067710       -0.111873        -0.214965        -0.092662       -0.087822        -0.194929        -0.095285
WF3 OUTWARD              0.8       2        3      11095                     0.0                          0.107527                       -0.132898         0.430769           0.334205            0.271924           -0.062280       -0.105251        -0.157892        -0.038787       -0.091214        -0.145501        -0.039443
WF1 OUTWARD              1.2       2        3       6820                     0.0                          0.204082                       -0.200000         0.422876           0.397361            0.307918           -0.089443       -0.096554        -0.198520        -0.076333       -0.091687        -0.189101        -0.071649
WF2 OUTWARD              1.2       2        3       6712                     0.0                          0.147059                       -0.156250         0.324108           0.386770            0.314213           -0.072557       -0.113951        -0.184116        -0.057127       -0.086546        -0.172957        -0.072222
WF3 OUTWARD              1.2       2        3       6224                     0.0                          0.191815                       -0.226840         0.400000           0.385443            0.320855           -0.064589       -0.086618        -0.090459         0.017278       -0.074513        -0.081779         0.013887
WF1  INWARD              0.2       2        3      21032                     0.0                          0.000000                        0.000000         0.116429           0.183577            0.162847           -0.020730       -0.321403        -0.354559        -0.021885       -0.303750        -0.329675        -0.013985
WF2  INWARD              0.2       2        3      20839                     0.0                          0.000000                        0.000000         0.234698           0.177024            0.157397           -0.019627       -0.335936        -0.368722        -0.028484       -0.315967        -0.359244        -0.039148
WF3  INWARD              0.2       2        3      20476                     0.0                          0.000000                        0.000000         0.050000           0.171811            0.164143           -0.007668       -0.301498        -0.325121        -0.016266       -0.275529        -0.310943        -0.014949
WF1  INWARD              0.4       2        3      19346                     0.0                          0.000000                        0.000000         0.100000           0.198387            0.170061           -0.028326       -0.272434        -0.328987        -0.041819       -0.255782        -0.302377        -0.031018
WF2  INWARD              0.4       2        3      18867                     0.0                          0.000000                        0.000000         0.166667           0.194891            0.171357           -0.023533       -0.271595        -0.313746        -0.035233       -0.254284        -0.303892        -0.042706
WF3  INWARD              0.4       2        3      18517                     0.0                          0.000000                        0.000000         0.107692           0.189178            0.175892           -0.013285       -0.255187        -0.297681        -0.028844       -0.227010        -0.286269        -0.031089
WF1  INWARD              0.8       2        3      13521                     0.0                          0.000000                        0.000000         0.142857           0.254419            0.216774           -0.037645       -0.137554        -0.236261        -0.079516       -0.126847        -0.232215        -0.086188
WF2  INWARD              0.8       2        3      12989                     0.0                          0.000000                       -0.065359         0.189565           0.245207            0.204866           -0.040342       -0.176024        -0.254322        -0.068658       -0.167073        -0.246348        -0.069532

## 3c. Scale-Matched Diagnostic（同 contact/horizon/action 的 scale 配对，节选前 20 行）

 wf  action  h  scale_low  scale_high  matched_n  median_delta_target_distance  median_delta_risk_distance  median_delta_RR mean_delta_target_first  mean_delta_E_R_lower
WF1  INWARD  2        0.2         0.4      19205                           0.0                         0.0              0.0                     0.0              0.000000
WF1 OUTWARD  2        0.2         0.4      18981                           0.0                         0.0              0.0                     0.0              0.000000
WF1  INWARD  3        0.2         0.4      21325                           0.0                         0.0              0.0                 0.00211              0.007663
WF1 OUTWARD  3        0.2         0.4      20917                           0.0                         0.0              0.0                0.005594              0.024619
WF1  INWARD  5        0.2         0.4      21500                           0.0                         0.0              0.0                 0.00586              0.012715
WF1 OUTWARD  5        0.2         0.4      21595                           0.0                         0.0              0.0                0.002825              0.009365
WF1  INWARD  8        0.2         0.4      21428                           0.0                         0.0              0.0                  0.0035              0.020978
WF1 OUTWARD  8        0.2         0.4      21378                           0.0                         0.0              0.0                0.004818              0.001209
WF1  INWARD 13        0.2         0.4      21417                           0.0                         0.0              0.0                0.003222             -0.017201
WF1 OUTWARD 13        0.2         0.4      21202                           0.0                         0.0              0.0                 0.00217             -0.011954
WF2  INWARD  2        0.2         0.4      18429                           0.0                         0.0              0.0                     0.0              0.000000
WF2 OUTWARD  2        0.2         0.4      18166                           0.0                         0.0              0.0                     0.0              0.000000
WF2  INWARD  3        0.2         0.4      20572                           0.0                         0.0              0.0                0.005201              0.017838
WF2 OUTWARD  3        0.2         0.4      20021                           0.0                         0.0              0.0                0.004146              0.016064
WF2  INWARD  5        0.2         0.4      20975                           0.0                         0.0              0.0                0.004338              0.011872
WF2 OUTWARD  5        0.2         0.4      20975                           0.0                         0.0              0.0                0.001812              0.005015
WF2  INWARD  8        0.2         0.4      20992                           0.0                         0.0              0.0                0.001858             -0.004478
WF2 OUTWARD  8        0.2         0.4      20855                           0.0                         0.0              0.0                0.004363             -0.009963
WF2  INWARD 13        0.2         0.4      21057                           0.0                         0.0              0.0                0.005129              0.038007
WF2 OUTWARD 13        0.2         0.4      20576                           0.0                         0.0              0.0                0.004082             -0.019489

## 4. 当前结论

- **P11 横截面观察**：在各 horizon 当时可形成结构的样本中，较晚 horizon 的
  `target_first_rate` / `E[R]_lower` 更低；但 available cohort 随 horizon 从约
  0.66 升到约 0.90，**各 h 之间不是同一批 contact**，因此这一结果**不能归因于
  waiting 本身**。
- **WAIT 的因果/配对结论以 P12 matched comparison 为准**：P12 用「同一 contact 在
  first-available 之后 vs 之后各 horizon」的配对差（median delta），排除了 cohort
  composition 混淆。
- morphology / field 增量（第三张表 `geometry_conditioned_*.csv`）仅在固定 RR bin
  下看单变量 conditional，不自动切 cut point。
- 若 Gate A/B 未满足 → `STOP_BEFORE_MODEL`。
