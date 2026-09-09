# M2 结果状态说明（权威）

## SNAPSHOT（快照）—— VALID

`m2_main.csv` / `m2_by_fold.csv` / `m2_by_symbol.csv` 中
**特征视图 = SNAPSHOT** 的行全部有效、可用、冻结。

| 模型 | 夏普率 |
|---|---:|
| CatBoost回归 | −0.9173 |
| LightGBM回归 | 1.207 |
| XGBoost回归 | −0.0936 |
| Ridge回归 | 0.3601 |

## TEMPORAL（时间增强）—— **INVALIDATED（作废，禁止引用）**

`m2_main.csv` 等文件中 **特征视图 = TEMPORAL** 的行全部作废。

原因：`build_feature_matrices` 使用 `pd.concat([Xsnap, temporal], axis=1)`
按**行位置**对齐，而 Xsnap 为动作级 119,640 行、temporal 为事件级 19,940 行，
导致只有 **16.7%** 的动作行拿到时间特征，其余 **83.3%** 全为 NaN
（被中位数填成常数后失去区分力）。

后果（实测）：TEMPORAL 与 SNAPSHOT 的逐事件预测在 F3、F4 **100% 完全相同**，
所谓「TEMPORAL 增量」只集中在 F1/F2，是构造错误的产物。

**作废的历史结论（禁止再次引用）：**

- ❌ 「CatBoost TEMPORAL 夏普 0.3917」
- ❌ 「LightGBM TEMPORAL 夏普 1.4591」（曾被称为当前冠军）
- ❌ 「XGBoost TEMPORAL 夏普 0.0756」
- ❌ 「Ridge TEMPORAL 夏普 0.876」
- ❌ `m2_temporal.csv` 中全部「快照 vs 时间增强」差值

## 请使用修正后结果

见 `research/analysis_results/m2_temporal_fix/`：

- `m2fix_compare.csv` —— 快照 vs **修正后**时间增强
- `m2fix_main.csv` / `m2fix_by_fold.csv` / `m2fix_by_symbol.csv`
- `m2fix_m3_levels.csv` / `m2fix_m3_contrib.csv`
- `m2fix_direction.csv`

修复方式：以 `candidate_id` 做 many-to-one 显式关联（`validate="many_to_one"`）
并加入硬断言，覆盖率为 119,640/119,640 = 100%。

**修正后结论：4 个模型只有 1 个（CatBoost）改善，其余 3 个恶化，
时间增强第一版判定为失败，已停止研究。**
