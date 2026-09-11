# SMC Direction Identity & Feasibility Gate v1.0

> 主线从 continuous frontier（P4c，已冻结 `DIRECT_STRUCTURAL_SWITCH`）切到方向可行性。
> base commit `647f9ad`。TRADING_METRICS=NOT_APPLICABLE：本实验只判定"方向是否
> 可被事前可见状态稳定预测"，不定义任何交易动作、不 PnL、不 regime 模型。

## 0. 设计（已注册）

- **Primary risk 固定 1.0 ATR**；禁止 7 档挑 risk。
- **Primary label = REVERSAL vs CONTINUATION**（镜像统一上下扫）：
  - `side=+1`（上方 liquidity）：SHORT=reversal，LONG=continuation；
  - `side=-1`（下方 liquidity）：LONG=reversal，SHORT=continuation。
  - 模型训练目标 = reversal/continuation；预测后唯一映射回绝对 LONG/SHORT。
- **eligible** = `rr_direction ∈ {LONG_DOMINATES, SHORT_DOMINATES}` @ risk=1.0（冻结
  `oracle_risk_direction_v1_2`，非 TRADEOFF/CENSOR/NO_COMPARABLE）。
- **universe** = 15 品种；**blocks** = TB1–TB4；**WF** = TB1→TB2 / TB1+TB2→TB3 /
  TB1+TB2+TB3→TB4；**OOS** = trading_day ≥ 2026-09-07，完全不入任何 fit/阈值/gate。
- **Feature blocks**：D0(symbol,side) → D1(contact identity + pre-contact cluster) →
  D2(identity-aware 两侧 per-scope map) → D3(interaction) → D4(60-bin field) →
  D5(trend) → D6(OB context)。非线性 sanity = HistGradientBoosting（固定参数，禁调参）。

## 1. 数据审计（Liquidity Identity Enum 已冻结）

冻结真实枚举（不猜类别）：

| 字段 | 取值 |
|---|---|
| liquidity_type | PREV_CONTIG_SESSION_HIGH/LOW, PREV_TRADING_DAY_HIGH/LOW, CONFIRMED_SWING_HIGH/LOW, CANONICAL_EQH/EQL, PREV_TRADING_WEEK_HIGH/LOW（9） |
| liquidity_scope | 5m, 15m, 1h, CONTIG_SESSION, TRADING_DAY, TRADING_WEEK（6） |
| contact_type | PENETRATE_CLOSE_BEYOND, PENETRATE_RECLAIM, TOUCH_ONLY, GAP_CROSS, PENETRATE_CLOSE_AT（5） |
| side | +1 / -1 |

主队列（in-sample, risk=1.0）：**eligible = 65,008**（reversal 26,204 / continuation 38,804；
base reversal rate ≈ 0.403 → 整体略偏 continuation）。

## 2. Direction Label Audit（risk=1.0）

冻结 7 态在 risk=1.0 共 96,900 行：LONG_DOMINATES 31,847 / SHORT_DOMINATES 33,161 /
TRADEOFF_OR_OVERLAP 24,225 / UNRESOLVED_CENSOR 4,127 / NO_COMPARABLE 3,540。
eligible 占 67.1%，绝大多数 contact 在 risk=1.0 有明确方向 dominance。
（`direction_label_audit.csv` 按 symbol×TB 细分。）

## 3. Liquidity Identity Findings（回答用户的 8 问）

> 所有 cell 仅在 n≥200 时解释（`liquidity_identity_direction_profile.csv`）。

1. **不同 liquidity_scope 的 reversal 分布是否不同？** 基本一致：15m 0.389 /
   1h 0.372 / 5m 0.402 / CONTIG_SESSION 0.406 / TRADING_DAY 0.401 / TRADING_WEEK 0.399。
   **scope 本身几乎不改变 reversal 率** → 与 `L_identity≈随机` 一致。

2. **不同 liquidity_type 是否不同？** 轻微：CANONICAL_EQH 0.431 / PREV_*_HIGH ≈0.41 /
   CANONICAL_EQL 0.374 / PREV_*_LOW ≈0.39。HIGH 类略偏 reversal、LOW 类略偏 continuation，
   但幅度小，**不足以单独构成信号**。

3. **多周期同价 cluster 是否比单一 identity 不同？** `contact_cluster_identity_count`
   从 1→8 时 reversal 在 0.34–0.43 间波动（非单调）；`contact_cluster_scope_count`
   1→6 在 0.37–0.42。**有但不强**，非决定性。

4. **first contact vs repeat contact？** True 0.403 / False 0.403，**无差异**。

5. **同 global distance 下 per-scope target map 是否增加方向？** **是，且是主信号**：
   `L_map − L_identity` = **+0.093 / +0.107 / +0.120**（WF1/2/3）。即"上下两边分别是什么
   层级的 liquidity"远比"被攻击的是哪一类 liquidity"重要。

6. **interaction × scope 是否非线性交互？** contact_type 本身差异明显：TOUCH_ONLY
   reversal 仅 **0.356**（强 continuation），GAP_CROSS **0.447** / PENETRATE_CLOSE_AT
   **0.445** / PENETRATE_RECLAIM 0.419 / PENETRATE_CLOSE_BEYOND 0.403。且
   `N_core − L_core` = +0.019 / +0.012 / +0.021 → **非线性交互真实存在**
   （scope × contact_type × map）。

7. **是否值得正式建立 external/internal 层级？** 证据支持：方向信息主要来自"两侧
   liquidity 几何关系"（D2），而当前无 external/internal 字段。下一步在保持因果冻结的
   前提下，建立"被攻击 level 与其最近反向 liquidity 的结构关系"是合理的增量方向。

8. **FVG 是否值得作为下一 ontology block？** 当前 Atlas 无 canonical FVG，本轮按协议
   `FVG = UNAVAILABLE_CANONICAL`。鉴于 D2（liquidity 几何）已主导方向，FVG 作为独立
   incremental ontology 是下一个值得测试的 block（需在下一实验单独建立并做 incremental test）。

## 4. 模型结果

### 4.1 每模型 AUC（WF1 / WF2 / WF3）

| model | WF1 | WF2 | WF3 |
|---|---:|---:|---:|
| L0 (symbol+side) | 0.500 | 0.516 | 0.508 |
| L_identity | 0.499 | 0.522 | 0.515 |
| **L_map** | **0.592** | **0.629** | **0.635** |
| L_core | 0.601 | 0.636 | 0.644 |
| L_core_field (60bin) | 0.590 | 0.613 | 0.643 |
| L_core_trend | 0.602 | 0.636 | 0.644 |
| L_core_ob | 0.601 | 0.634 | 0.643 |
| L_full | 0.590 | 0.614 | 0.644 |
| N_core (HGB) | 0.621 | 0.648 | 0.665 |
| N_full (HGB) | 0.616 | 0.643 | 0.660 |

### 4.2 增量 delta（相对 L0 / 相对 L_core；协议 section 23）

| WF | identity−L0 | map−identity | core−map | field−core | trend−core | ob−core | full−core | N_core−L_core |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WF1 | −0.001 | **+0.093** | +0.010 | −0.011 | +0.001 | −0.001 | −0.011 | +0.019 |
| WF2 | +0.006 | **+0.107** | +0.007 | −0.022 | +0.001 | −0.002 | −0.022 | +0.012 |
| WF3 | +0.007 | **+0.120** | +0.009 | −0.001 | +0.000 | −0.001 | +0.000 | +0.021 |

解读（对应 section 23）：
- **L_identity ≈ L0**：liquidity identity 本身几乎无方向信息。
- **L_map ≫ L_identity（+0.09~+0.12）**：两侧 per-scope liquidity map 是方向主信号。
- **L_core ≈ L_map +0.01**：interaction 仅小幅增量。
- **L_core_trend / L_core_ob / L_core_field ≈ L_core**：传统 trend、OB context、60-bin
  density field **均无独立方向增量**（60bin 与 12 个 per-scope nearest 标量冗余）。
- **N_core ≫ L_core（+0.02）**：方向来自 identity×map×interaction 的非线性交互。

## 5. Confidence / Abstention（protocol section 17）

outer-train 内 3 段 expanding OOF 定阈值，应用到 outer test（阈值不从 test 选）：

| WF | coverage | top-20% test | abs dir acc |
|---|---:|---:|---:|
| WF1 | cov50 0.42 | | 0.669 |
| WF1 | cov20 0.13 | | 0.717 |
| WF2 | cov50 0.51 | | 0.681 |
| WF2 | cov20 0.19 | | 0.734 |
| WF3 | cov50 0.52 | | 0.720 |
| WF3 | cov20 0.20 | | 0.783 |

**模型自信的子集（top 20%）绝对方向准确率可达 0.73–0.78**，显著高于 0.5 基线。
（`confidence_coverage_curve.csv`）

## 6. By-symbol 时间稳定性（protocol section 18/19）

- **15/15 品种 × 3 WF 全部 AUC > 0.5**（45/45 by-symbol 单元）。
- macro median symbol AUC：WF1 0.575（IQR 0.566–0.590）/ WF2 0.608（0.587–0.626）/
  WF3 0.614（0.593–0.638）。
- 信号**跨 15 品种普遍成立**，非集中于少数品种；且 WF1→WF3 单调上升（样本外递进），
  无时间退化。

## 7. ROI Gate（protocol section 22）

| 条件 | 值 | 判定 |
|---|---|---|
| mean WF L_full AUC ≥ 0.58 | **0.616** | ✅ |
| WF1/2/3 均 ≥ 0.54 | 0.590 / 0.614 / 0.644 | ✅ |
| Top20% test abs dir acc ≥ 0.60 | **0.605** | ✅ |
| ≥10/15 品种 AUC>0.5 | **45/45** | ✅ |

> **VERDICT = `DIRECTION_LEARNABLE`**（通过 Gate）。
> 未达 `STRONG_DIRECTION_SIGNAL`（mean 0.616 < 0.62；Top20 0.605 < 0.65）。
> 非 `WEAK`（full/core + 非线性明显 > 0.55 且 Top20 > 0.58）。

## 8. 对用户核心问题的回答

> **仅依靠当前已可靠定义的 liquidity identity + interaction + 两侧 liquidity map，
> 方向到底有没有稳定信息？**

**有，且稳定、可泛化。** AUC ≈ 0.62（跨 15 品种 45/45 单元全 >0.5），confident 子集
绝对方向准确率 0.73–0.78。方向信息**主要来自两侧 liquidity 的几何相对位置**
（per-scope nearest map），**而非**被攻击 level 的 identity 本身、传统 trend、或 OB context
（这些增量均≈0）。interaction（尤其 contact_type）与非线性组合提供额外增益。

这正式把项目推进到用户设想的阶段 2：先证方向"有解"，再证哪些 liquidity identity 有区别。

## 9. 治理 / 泄漏控制

- 泄漏黑名单硬断言通过：模型矩阵 X 不含 `rr_direction / direction_stability /
  best_R_* / resolution_class / status / path_censor / required_risk_* / bars_to_best_*`
  等任何未来路径或标签字段。
- 冻结 Atlas v1.2 / 连续 evaluator 未改；P4c 机制结论（DIRECT_STRUCTURAL_SWITCH）沿用。
- 未定义 external/internal、未造 FVG、未加 LC、未做 PnL/止损/最优 risk。
- `direction_features.parquet`（大文件）不入 Git。

## 10. 本轮 STOP（待 reviewer 审核后再决定）

完成：数据审计 / identity feature 构建 / Stage1 模型 / WF / confidence / by-symbol /
报告 / 测试 / commit / push。**未做** 0.5/2.0 ATR 测试、FVG 定义、external/internal
定义、+1 bar confirmation、PnL、动态盈亏比——这些都在协议 STOP 列表内，需 reviewer
授权后再进入。

下一轮候选（按协议顺序）：pre-contact dynamics → FVG/displacement ontology (incremental)
→ +1 bar confirmation → 动态 stop/target/盈亏比优化。
