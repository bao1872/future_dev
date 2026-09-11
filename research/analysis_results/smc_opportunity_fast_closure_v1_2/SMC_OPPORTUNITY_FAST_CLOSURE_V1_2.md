# SMC Opportunity Fast Closure Audit v1.2-lite

> 实验：SMC Opportunity Fast Closure（低成本两阶段 Gate）
> 仓库：bao1872/future_dev　分支：main
> 基准提交：a746e3e5402e71a44cf57dbf1b718f4610f8e2bd
> 延续：SMC Opportunity Geometry Decomposition v1.1（ff36930，已修复 P1–P7）
> 证据等级：内部因果时间泛化（TB1→TB4 walk-forward）。非独立 OOS。前瞻性 OOS 边界：2026-09-07。

> **目标**：用最低成本回答两个问题，不扩展复杂模型。
> 1. 一条"最近目标距离 / stop 距离"几何规则是否已足以替代 Opportunity 模型？
> 2. G4 在 G1 已知后，是否还有"大到值得继续维护这 60 个字段"的残余信息？
>
> **结论**：Opportunity 可压缩为"上下两条 target 距离"的 2 特征几何；metadata 与多周期 G4 字段均无增量。停止 Opportunity 复杂研究，资源转入 **RISK_DEPENDENT**。

---

## 0. 方法（刻意从简）

- 仅 3 档 risk：0.5 / 1.0(primary) / 2.0 ATR（覆盖紧/中/宽，不调参）。
- WF1/2/3 点估计；**Stage 1 无 bootstrap**（Gate 未通过即停）。
- 标签门沿用 ff36930：`model_eligible = primary_eligible & label∈{DELIVERY,NO_DELIVERY}`；三路管线 + tie-aware AUC。

### Stage 1-A 简单几何压缩
| 模型 | 内容 |
|---|---|
| R0 | `score = -min(nearest_above_R, nearest_below_R) / risk_ATR`，纯规则不训练 |
| R2 | 训练 `[nearest_above_R, nearest_below_R]`（上下两距离） |
| R3 | `B0 + G1`（metadata + 上下两距离 = 此前 M_G1） |

### Stage 1-B G4 残余（防泄漏 meta）
- Temporal inner OOF：expanding 40%→预测 20%、60%→20%、80%→20%，生成 `p_g1`/`p_g4`（禁止 in-sample stacking）。
- meta 输入 `logit(p_g1)` vs `logit(p_g1), logit(p_g4)`；比较 `META_G1_G4 − META_G1`。

---

## 1. Stage 1-A 结果（ROC-AUC）

| risk | WF | R0 | R2 | R3 |
|---|---|---:|---:|---:|
| 0.5 | WF1/2/3 | 0.775 / 0.773 / 0.769 | 0.792 / 0.787 / 0.780 | 0.790 / 0.785 / 0.780 |
| 1.0 | WF1/2/3 | 0.790 / 0.775 / 0.779 | 0.816 / 0.803 / 0.799 | 0.815 / 0.801 / 0.799 |
| 2.0 | WF1/2/3 | 0.815 / 0.806 / 0.815 | 0.858 / 0.849 / 0.844 | 0.855 / 0.842 / 0.843 |

逐单元 ΔAUC：
- **R2 − R0 = +0.011 ~ +0.043**（9/9 全部为正，均值 **+0.026**）→ 单条归一化距离规则 R0 **不够**；上下不对称的两条距离有真实增量。
- **R3 − R2 = −0.003 ~ +0.000**（均值 −0.002 ≈ 0）→ 在已有两距离后，metadata（B0）**零增量**。

**A Gate** → `R0_INSUFFICIENT_TWO_SIDED_GEOMETRY_SUFFICIENT`：R0 不够，但 R2（两特征几何）即封顶，无需复杂 ML。

---

## 2. Stage 1-B 结果（G4 残余）

| risk | WF | base_G1 | base_G4 | meta_G1 | meta_G1_G4 | **dAUC_meta** |
|---|---|---:|---:|---:|---:|---:|
| 0.5 | WF1/2/3 | 0.790 / 0.785 / 0.780 | 0.721 / 0.727 / 0.743 | 0.790 / 0.785 / 0.780 | 0.790 / 0.785 / 0.780 | 0.000 / 0.000 / −0.000 |
| 1.0 | WF1/2/3 | 0.815 / 0.801 / 0.799 | 0.747 / 0.750 / 0.764 | 0.815 / 0.801 / 0.799 | 0.815 / 0.801 / 0.799 | 0.000 / 0.000 / −0.000 |
| 2.0 | WF1/2/3 | 0.855 / 0.842 / 0.843 | 0.795 / 0.797 / 0.805 | 0.855 / 0.842 / 0.843 | 0.855 / 0.842 / 0.842 | 0.000 / +0.000 / −0.001 |

- **dAUC_meta 范围 [−0.0007, +0.0003]，均值 −0.0001；9/9 全部 < 0.005**。
- base_G4（0.72~0.80）明显弱于 base_G1（0.78~0.86）→ G4 单独有信号，但**在 G1 之上残余≈0**。

**B Gate** → `G4_RESIDUAL_NOT_MATERIAL`：冻结 Atlas 的 60 个 bin 字段无独立增量价值，停止维护/研究。

---

## 3. 综合裁决

| Gate | 结果 |
|---|---|
| Stage 1-A | `R0_INSUFFICIENT_TWO_SIDED_GEOMETRY_SUFFICIENT` |
| Stage 1-B | `G4_RESIDUAL_NOT_MATERIAL` |
| **Overall** | **`STOP_OPPORTUNITY_TWO_SIDED_GEOMETRY_SUFFICIENT`** |

**含义**：
- Opportunity 的本质是**上下两条 target 距离的几何关系**——不是一条归一化比值（R0 不够，两侧不对称有信息），但也**不需要 metadata、不需要多周期 G4、不需要复杂 ML**（R3=R2，G4 残余≈0）。
- 这把模型从"7 档 × 多模型 × 500 bootstrap"压缩到**2 个特征 + 逻辑回归**（或等价的几何规则）。

---

## 4. 与 prior v1.1（ff36930）的关系

- ff36930 已修复 P1–P7，结论为 "SIMPLE_GEOMETRY（G4 真实但被 G1 吸收）"。
- 本 lite Gate 在修复后基础上进一步收紧：**G4 残余≈0**，且明确了"压缩为 2 特征几何"这一可操作形态。
- 二者一致，无矛盾；本实验是更低成本的收口，不引入新变量、不重做全量 bootstrap。

---

## 5. 治理

- `TRADING_METRICS = NOT_APPLICABLE`（机制/去混淆实验，只回答"机制是否独立"，不定义交易动作、不产出交易指标）。
- 未进入 PnL / 方向模型 / 最佳 ATR / LightGBM / SHAP。
- Atlas v1.2 未改动；模型仅用已冻结特征定义。

---

## 6. 输出文件

`research/analysis_results/smc_opportunity_fast_closure_v1_2/`
- `fast_geometry_metrics.csv`（R0/R2/R3 × 3 risk × 3 WF 的 ROC/PR/Top20/Bottom20）
- `g4_residual_fast_metrics.csv`（G4 OOF-meta 残余增量）
- `stage1_gate_decision.json`（A/B/Overall Gate）
- `FAST_CLOSURE_AUDIT.json`
- `SMC_OPPORTUNITY_FAST_CLOSURE_V1_2.md`

## 7. 下一步

**停止 Opportunity 复杂研究。下一主线 = RISK_DEPENDENT mechanism**：
紧 stop 单方向 → 风险尺度扩大 → 方向翻转（switch ATR 中位≈1.25ATR）。
该现象不由标签定义天然决定，理论含金量高于 Opportunity 几何，研究 ROI 更高。
