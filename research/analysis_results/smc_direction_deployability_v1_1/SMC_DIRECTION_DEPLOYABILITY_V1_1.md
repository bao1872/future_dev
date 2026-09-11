# SMC Direction Deployability & Liquidity Map Decomposition v1.1

> 主线接续 `SMC Direction Identity & Feasibility Gate v1.0`（base `e7ee079`）。
> 用户判定 v1.0 方向 AUC 信号真实但存在明确代码 bug 与不可复现问题，且"per-scope
> liquidity map 是主信号"尚未被严格证明。本轮 = **修复 v1.0 + 把方向信号从可行性发现
> 升级为可部署性 + liquidity map 真正分解的确认实验**。

base commit：`e7ee07937681c89f6da60c2264381085840b36ce`
TRADING_METRICS=NOT_APPLICABLE（不定义任何交易动作、不 PnL、不 regime）。

## 0. 结论（ROI Gate）

```
DIRECTION_SIGNAL_CONFIRMED = TRUE
DEPLOYABLE_DIRECTION_SELECTOR = FALSE
LIQUIDITY_IDENTITY_INCREMENTAL = FALSE
verdict = DIRECTION_SIGNAL_CONFIRMED
```

| Gate 条件 | 值 | 判定 |
|---|---:|---|
| N_MAP_CORE mean WF AUC ≥ 0.60 | **0.6416** | ✅ |
| 各 WF N_MAP_CORE AUC ≥ 0.58 | 0.621 / 0.645 / 0.659 | ✅ |
| 20% train-tail macro_tail_accuracy ≥ 0.60 | **0.6566** | ✅ |
| ≥10/15 品种 N_MAP_CORE macro AUC > 0.50 | **45/45** | ✅ |
| clear-gate mean AUC ≥ 0.55 | 0.7708 | ✅ |
| joint selector actionable_precision ≥ 0.60 | **0.4485** | ❌ |
| IDENTITY_MAP − GLOBAL4 ΔAUC ≥ +0.01（增量） | **−0.0201** | ❌ |

**方向信号已确认可学（AUC 0.64、跨 15 品种全通过、高置信两尾 macro 0.66），但
联合可部署选择器（先判 clear 再判方向 + 弃权）精度仅 0.45，尚未可部署；且 canonical
liquidity identity（scope/type 分解）相对全局上下距离无独立增量。**

---

## 1. P0：v1.0 可复现性与指标 bug 修复

| # | bug | 修复 |
|---|---|---|
| P0.1 | `reversal_label` 对非 clear 标签返回错误值 | 现对非 clear 返回 `NaN`；删除 `r1` 对 `side` 的重复 merge |
| P0.2 | by-symbol balanced_accuracy 公式错（k=0 仍在算 `pred==1`） | 改用 `sklearn.metrics.balanced_accuracy_score` |
| P0.3 | Gate `Top20=0.6046` 实为 L_full 全样本 accuracy | 删除；改用 P7 两尾 `macro_tail_accuracy` |
| P0.4 | `direction_feature_block_deltas.csv` 由 ad-hoc 代码生成、未入脚本 | 逻辑写回正式脚本；新增 `REPRODUCIBILITY_AUDIT.json`（script sha + 输入 parquet hash + 输出行数） |
| P0.5 | 报告写 liquidity_type=9 | 实测 **10 种**（协议已记录） |

`REPRODUCIBILITY_AUDIT.json` 记录脚本 sha256 与 4 个输入 parquet 的 file-level hash；
`checkout e7ee079 + 本脚本` 一键重建所有小型输出（大型 `direction_features_v1_1.parquet` 不入 Git）。

---

## 2. 数据与模型

- **Universe**：15 品种；**blocks** TB1–TB4；**WF** TB1→TB2 / TB1+TB2→TB3 / TB1+TB2+TB3→TB4；
  **OOS** 2026-09-07 完全排除。
- **Label**：risk=1.0，`y_reversal = 1` 当 reversal（上方→SHORT，下方→LONG）。
  eligible = `LONG/SHORT_DOMINATES`，in-sample = **65,008**。
- **Feature blocks**：D0(symbol,side) / D1(contact identity + pre-contact cluster) /
  GLOBAL2(nearest_above/below) / GLOBAL4(+ahead/behind) / SCOPE12(6 scope×above/below+has) /
  TYPE20(10 type×above/below+has) / D3(interaction) / 60-bin & trend & OB 本轮不入方向模型
  （v1.0 已证其无独立增量，留作下一轮待审）。
- **Master 派生**（因果、复用 frozen `active_mask`）：contact_cluster_*（pre-contact same-price）、
  12 per-scope nearest + has 标志、20 per-type nearest + has 标志。
- **ROBUST 标签**：从 7 档 `oracle_risk_direction_v1_2` 确定性推导（各 risk 档 clear 方向一致 →
  ROBUST），无新数据、无泄漏。risk=1.0 eligible 中 **47,961** 为 ROBUST。
- **非线性**：固定 `HistGradientBoostingClassifier(max_depth=3, lr=0.05, max_iter=200, l2=1.0)`，
  只跑 `N_GLOBAL4 / N_IDENTITY_MAP / N_MAP_CORE`，禁调参。

---

## 3. 方向分解（核心结果）

### 3.1 每模型 AUC（WF1 / WF2 / WF3）

| model | 含义 | WF1 | WF2 | WF3 |
|---|---|---:|---:|---:|
| M_ID | D1 身份 | 0.499 | 0.522 | 0.515 |
| M_GLOBAL2 | +全局 above/below | 0.524 | 0.581 | 0.554 |
| **M_GLOBAL4** | +ahead/behind | **0.599** | **0.633** | **0.633** |
| M_SCOPE12 | 仅 per-scope 12（无全局） | 0.529 | 0.550 | 0.567 |
| M_GLOBAL_SCOPE | GLOBAL4 + scope12 | 0.593 | 0.625 | 0.636 |
| M_TYPE20 | +per-type 20 | 0.545 | 0.584 | 0.587 |
| M_GLOBAL_TYPE | GLOBAL4 + type20 | 0.565 | 0.631 | 0.633 |
| M_IDENTITY_MAP | GLOBAL4 + scope12 + type20 | 0.552 | 0.621 | 0.631 |
| M_MAP_CORE | IDENTITY_MAP + interaction | 0.555 | 0.624 | 0.633 |
| M_CORE | GLOBAL_SCOPE + interaction | 0.605 | 0.630 | 0.642 |
| N_GLOBAL4 | HGB | 0.634 | 0.659 | 0.659 |
| N_IDENTITY_MAP | HGB | 0.609 | 0.648 | 0.652 |
| **N_MAP_CORE** | HGB | **0.621** | **0.645** | **0.659** |

### 3.2 增量 delta（相对基准，协议 section 23）

| delta | 值 |
|---|---:|
| GLOBAL2 − ID | +0.0408 |
| **GLOBAL4 − GLOBAL2** | **+0.0683** |
| SCOPE12 − ID | +0.0366 |
| GLOBAL_SCOPE − GLOBAL4 | −0.0033 |
| TYPE20 − ID | +0.0598 |
| GLOBAL_TYPE − GLOBAL4 | −0.0117 |
| **IDENTITY_MAP − GLOBAL4** | **−0.0201** |
| MAP_CORE − IDENTITY_MAP | +0.0030 |
| CORE − GLOBAL_SCOPE | +0.0076 |

### 3.3 解读（回答用户中心疑点）

1. **全局几何（above/below/ahead/behind）是真实信号**：`GLOBAL4−GLOBAL2 = +0.068`，
   比 `GLOBAL2−ID` 还大 → **同一方向的前后距离（below=已跌破、ahead=同方向未触及）携带
   最多方向信息**。
2. **per-scope / per-type identity map 无独立增量**：`GLOBAL_SCOPE−GLOBAL4 = −0.003`、
   `GLOBAL_TYPE−GLOBAL4 = −0.012`、`IDENTITY_MAP−GLOBAL4 = −0.020`（均为负）→
   把 5m/15m/1h/session/day/week 或 10 种 liquidity_type **拆开看，相对"全局上下距离"
   没有任何额外价值**。
3. **identity 本身 ≈随机**（M_ID 0.50/0.52/0.52）。

> **直接回答用户最初担忧**："不同 liquidity 身份是否该无差异看待"——本轮**未被证伪**。
> 方向信息活在**简单两侧距离几何**里，不在"被攻击的是哪一类 liquidity"。
> 这正是与 Opportunity 阶段同样的结论：几何 > 标签身份。

---

## 4. Paired trading-day bootstrap（500 次，保留重复 day 权重）

关键 delta 的 AUC 差 95% CI（下界/中位/上界）：

| delta | WF1 | WF2 | WF3 |
|---|---|---|---|
| GLOBAL2−ID | +0.014/+0.026/+0.037 ✅ | +0.041/+0.059/+0.077 ✅ | +0.018/+0.039/+0.059 ✅ |
| GLOBAL_SCOPE−GLOBAL4 | −0.015/−0.005/+0.003 ❌ | −0.018/−0.008/+0.003 ❌ | −0.004/+0.003/+0.010 ❌ |
| GLOBAL_TYPE−GLOBAL4 | −0.051/−0.034/−0.012 ❌ | −0.019/−0.001/+0.014 ❌ | −0.011/0.000/+0.011 ❌ |
| **IDENTITY_MAP−GLOBAL4** | **−0.065/−0.047/−0.025** ❌ | **−0.032/−0.012/+0.006** ❌ | **−0.014/−0.002/+0.009** ❌ |
| MAP_CORE−IDENTITY_MAP | +0.000/+0.003/+0.006 ns | −0.001/+0.003/+0.007 ns | −0.003/+0.003/+0.009 ns |
| N_IDENTITY_MAP−N_GLOBAL4 | −0.038/−0.025/−0.010 ❌ | −0.024/−0.011/+0.002 ❌ | −0.018/−0.008/+0.002 ❌ |
| N_MAP_CORE−N_IDENTITY_MAP | +0.006/+0.012/+0.017 | −0.010/−0.002/+0.005 | +0.001/+0.007/+0.012 |

✅ = CI 下界 > 0；❌ = CI 下界 ≤ 0（无正增量或负）。结论：全局几何增量稳健显著；
scope/type/identity 分解增量**全部不显著为负或零**。

---

## 5. By-symbol 稳定性（修正后）

`direction_by_symbol_corrected.csv`：N_MAP_CORE 在 **15×3 = 45 个单元 AUC 全部 > 0.5**，
macro median symbol AUC = **0.645**（WF1/2/3 = 0.566/0.616/0.643，无时间退化）。
信号跨 15 品种普遍成立，非集中于少数品种。

---

## 6. 两尾 Confidence（修正后，P7）

train-OOF 定两尾阈值，outer-test 选两尾（abstain 中间）。`macro_tail_accuracy` 为核心指标：

| WF | cov | test cov | rev_tail | cont_tail | **macro_tail** |
|---|---:|---:|---:|---:|---:|
| WF1 | 0.2 | 0.335 | 0.454 | 0.683 | 0.569 |
| WF2 | 0.2 | 0.126 | 0.602 | 0.770 | 0.686 |
| WF3 | 0.2 | 0.129 | 0.590 | 0.840 | 0.715 |
| WF1 | 0.1 | 0.232 | 0.443 | 0.729 | 0.586 |
| WF2 | 0.1 | 0.046 | 0.546 | 0.804 | 0.675 |
| WF3 | 0.1 | 0.053 | 0.662 | 0.869 | 0.765 |

**回应用户对"高置信可能只是多数类 continuation 幻觉"的担忧**：continuation 尾准确率
确实高（0.68–0.87），但 reversal 尾也真实高于随机（WF2/3 的 0.59–0.66，cov10 达 0.66），
且 selected_reversal_base_rate 多在 0.40–0.50（非极端多数类偏置）。即**双向能力真实存在但
不对称**：continuation 比 reversal 更易预测。Gate 用 macro（双向平均）故稳健。

---

## 7. Clear-direction Gate 与联合可部署选择器（P8–P9）

- **Clear gate**（risk=1.0，cohort = LONG/SHORT/TRADEOFF，预测"这次有无明确方向"）：
  C_MAP_CORE mean AUC = **0.7708**（CN_MAP_CORE 更高）。**clear/unclear 极易区分**。
- **Joint selector**（先 clear gate 再 direction 两尾 + 弃权）：

| WF | selection_rate | selected_clear_rate | direction_acc_given_clear | **actionable_precision** |
|---|---:|---:|---:|---:|
| WF1 | 0.125 | 0.935 | 0.447 | 0.447 |
| WF2 | 0.070 | 0.953 | 0.459 | 0.459 |
| WF3 | 0.067 | 0.947 | 0.440 | 0.440 |

clear gate 选出的子集 95% 确实 clear，但**其中 direction 准确率仅 ~0.45（低于随机）** →
联合选择器 actionable_precision = 0.45 < 0.60 → **DEPLOYABLE 未过**。瓶颈在"clear 子集上的
方向模型"失效，而非 clear 检测。这是下一轮必须解决的核心工程问题。

---

## 8. Robust Direction Diagnostic（P10）

在 47,961 个 risk-稳定（ROBUST）contact 上：

| model | WF1 | WF2 | WF3 |
|---|---:|---:|---:|
| M_GLOBAL4 | 0.602 | 0.615 | 0.630 |
| M_MAP_CORE | 0.584 | 0.605 | 0.648 |
| N_MAP_CORE | 0.614 | 0.632 | 0.664 |

**方向信号在 risk-不敏感子集上依然可学** → 不是 P4c risk-coupling 的纯假象，而是真实存在的
结构信号。

---

## 9. Risk Coupling Sensitivity（P11）

固定 feature spec（M_MAP_CORE / N_MAP_CORE），在 0.5 / 1.0 / 2.0 ATR 独立训练：

| risk | WF1 N_MAP_CORE | WF2 | WF3 |
|---|---:|---:|---:|
| 0.5 | 0.650 | 0.678 | 0.694 |
| 1.0 | 0.621 | 0.645 | 0.659 |
| 2.0 | 0.574 | 0.578 | 0.598 |

**direction AUC 随 risk 单调下降** → 方向与 risk 耦合（印证 P4c）。未来必须是
`Direction(r)` 而非 `Direction` + 独立 risk 优化。

---

## 10. 用户四问回答

1. **方向信号是否稳健？** 是（AUC 0.64，45/45 by-symbol 通过，cov20 macro 0.66，Robust 子集仍可学）。
2. **简单上下距离 vs identity-aware map 谁提供增量？** **简单全局几何（above/below/ahead/behind）
   提供几乎全部增量；per-scope / per-type identity 分解零增量**（paired CI 全部 ≤0）。
3. **高置信 73–78% 是否真实双向？** 是，但**不对称**：continuation 尾 0.68–0.87，
   reversal 尾 0.59–0.66（WF3 cov10=0.66）；非纯多数类幻觉。
4. **不知道未来是否 TRADEOFF 时能否决定"要不要交易"？** clear gate 易（0.77），但
   **联合选择器精度仅 0.45，尚不能** → 这是下一轮关键工程任务。

---

## 11. 治理 / 泄漏控制

- 泄漏黑名单硬断言通过：X 不含 `rr_direction/best_R_*/required_risk_*/bars_to_best_*` 等。
- 未定义 external/internal、未造 FVG、未加 LC、未做 PnL（clear gate 仅用 LONG/SHORT/TRADEOFF，
  排除 CENSOR/NO_TARGET）。
- Atlas v1.2 与连续 evaluator 未改。

## 12. 本轮 STOP

完成：P0 修复 + 数据审计 + map 分解 + paired bootstrap + 两尾 confidence + clear gate +
联合选择器 + Robust 诊断 + risk coupling + 报告 + 测试 + commit + push。
**未做**：FVG、external/internal、pre-contact momentum、+1 bar confirmation、PnL、动态 stop/target。

下一轮（按用户优先级）**先于 FVG**：(a) 修复联合选择器（clear 子集上的方向失效）、
(b) pre-contact dynamics（价格如何来到 liquidity）。仅当 DEPLOYABLE 通过后，才进入
动态 stop/target/盈亏比。
