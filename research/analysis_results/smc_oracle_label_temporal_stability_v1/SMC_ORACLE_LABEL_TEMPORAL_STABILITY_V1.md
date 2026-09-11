# SMC Oracle 标签时间稳定性审计 v1.0

> **证据等级**：Atlas v1.2 全历史数据的**描述性时间稳定性审计**，不是独立 OOS validation。禁止声称“独立验证 / 真正OOS / 未见数据确认”。

- 冻结提交：`7ae7c1ae57b45bbc8bade1b5742fdc0353859c83`
- TB1–TB4 定义已冻结，hash=`b51fe7952bc1311a...`
- 唯一交易日数：407，时间顺序 max(TBi)<min(TBi+1)：True

## TB1–TB4 边界（冻结）

| block | min_day | max_day | n_days | n_contacts | n_symbols |
|---|---|---|---:|---:|---:|
| TB1 | 2025-01-02 | 2025-06-09 | 102 | 22688 | 15 |
| TB2 | 2025-06-10 | 2025-11-06 | 102 | 25903 | 15 |
| TB3 | 2025-11-07 | 2026-04-10 | 102 | 24440 | 15 |
| TB4 | 2026-04-13 | 2026-09-04 | 101 | 23869 | 15 |

## 核心发现稳定性（原始 TB1–TB4 数字 + 摘要）

| finding | TB1 | TB2 | TB3 | TB4 | verdict |
|---|---:|---:|---:|---:|---|
| A. NO_REACH / NO_DIRECTION 跨块稳定 | 0.8783 | 0.8448 | 0.8542 | 0.8647 | STABLE |
| B. ROBUST (L+S) 跨块稳定 | 0.1885 | 0.20700000000000002 | 0.19090000000000001 | 0.19269999999999998 | STABLE |
| C. ROBUST 中 reversal rate 跨块稳定 | 0.4263 | 0.3766 | 0.4127 | 0.3823 | STABLE |
| D. align_1h delta_pp 跨块稳定 | 0.0073 | -0.0006 | -0.0049 | 0.0219 | STABLE |
| D. align_4h delta_pp 跨块稳定 | 0.0044 | -0.0173 | -0.0261 | -0.0089 | STABLE |
| E. HTF pullback+sweep(LONG) delta_pp 跨块稳定 | -0.0037 | 0.0 | -0.0034 | -0.0055 | STABLE |
| F. S->L 与 L->S 对称且跨块稳定 | 0.9314671814671814 | 0.9006823351023503 | 1.0423377818683848 | 1.0635903207653348 | STABLE |
| G. switch ATR median 跨块稳定 | 1.25 | 1.25 | 1.25 | 1.25 | STABLE |

## 研究裁决（回答 9 问）

1. **NO_REACH≈86% 跨 TB1–TB4 稳定吗？** NO_REACH/NO_DIRECTION = [0.8783, 0.8448, 0.8542, 0.8647]，verdict=STABLE。品种 median≈0.855。（NO_REACH 定义：没有任何一个风险档里 long 与 short 同时到达 target，即从不出现双向同时确认——与原始画像 86% 结论的定义完全一致。）

2. **ROBUST_DIRECTIONAL≈19.5% 跨时间稳定吗？** ROBUST(L+S) share = [0.1885, 0.2070, 0.1909, 0.1927]，verdict=STABLE。

3. **Robust 方向更偏 reversal 还是 continuation，是否稳定？** reversal rate = [0.4263, 0.3766, 0.4127, 0.3823]，verdict=STABLE。（reversal=oracle_sign==-side，continuation=oracle_sign==side）

4. **多周期 trend alignment 相对条件基准是否有稳定 delta？** align_1h delta_pp=[0.0073, -0.0006, -0.0049, 0.0219]，align_4h delta_pp=[0.0044, -0.0173, -0.0261, -0.0089]，align_15m delta_pp=[-0.0245, -0.0132, -0.0278, 0.0063]，align_5m delta_pp=[-0.0135, 0.0263, 0.0141, 0.0374]。**注意：50% ≠ 随机；以上 delta_pp 是扣除了同(block,symbol,side)边际独立基准后的增量。**

5. **HTF trend + LTF pullback + 逆向 liquidity sweep 是否稳定偏向恢复 HTF 趋势？** HTF LONG-state restore delta_pp = [-0.0037, 0.0000, -0.0034, -0.0055]，verdict=STABLE。matched baseline 仅做描述性条件基准（同 block/symbol/side），非 outcome matching。

6. **RISK_DEPENDENT 的 S→L / L→S 是否保持对称？** S→L = [1930, 2376, 2265, 1890]，L→S = [2072, 2638, 2173, 1777]。两序列数量级接近，无明显系统性不对称。

7. **单次 switch ATR 尺度是否稳定？** median = [1.2500, 1.2500, 1.2500, 1.2500]（全样本参考 median≈1.25 ATR，IQR≈0.625–1.75）。仅作参考，不得作为通过条件。

8. **品种间异质性有多大？** 见 `by_symbol_summary.csv` 与 `no_direction_by_temporal_block.csv` 的品种 median/IQR。若 pooled 稳定但部分品种反向，应标记为 HETEROGENEOUS 而非 STABLE。

9. **下一阶段研究优先级**：在另一侧验证结论不被推翻的前提下，预注册优先级为 **Opportunity > Risk-dependent mechanism > Robust Direction > OB survival**。理由：最大可分离对象很可能是“这里到底有没有真正的 delivery”（NO_REACH 占 NO_DIRECTION 的绝大部分），而非多空本身；过滤无 delivery 事件的价值可能大于预测 Long/Short。

## 输出文件

- temporal_blocks_v1.json（冻结定义 + hash）
- temporal_block_audit.csv
- no_direction_by_temporal_block.csv
- direction_stability_by_temporal_block.csv
- trend_alignment_by_temporal_block.csv
- reversal_continuation_by_temporal_block.csv / by_stratum.csv
- htf_pullback_sweep_by_temporal_block.csv
- risk_switch_by_temporal_block.csv / risk_switch_top_sequences.csv
- switch_atr_by_temporal_block.csv
- by_symbol_summary.csv
- core_finding_stability.csv
- STABILITY_AUDIT.json
