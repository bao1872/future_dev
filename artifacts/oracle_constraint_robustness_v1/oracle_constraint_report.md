# Oracle Constraint Robustness v1 (R2.1)

> **Status**: `INDEPENDENT_REVIEW_CORE_PASS / METADATA_CLOSURE_COMPLETE`. This is a SENSITIVITY /
> ROBUSTNESS surface, not parameter tuning. No "best lambda" is selected.

**Task**: `FUTURE-ORACLE-R2.1-CORRECTNESS-CLOSURE`
**Base**: 0bee4029a40a8985f1be6da7431d2edeb0109dd6 (R2.1 base commit)
**Metadata closure**: Task `FUTURE-ORACLE-R2-FINAL-METADATA-CLOSURE`. `code_sha_used_for_t2` = `47a53405b8ef60208bfdc612ccf47a3076777f44` recorded in PROTOCOL + AUDIT; both parquet SHA256 pinned in AUDIT; no T2 re-run.
**Horizons**: [6, 12, 24]
**Grid**: 27 core + 7 stress
= 34 thetas. Core = λR∈(0, 0.25, 0.5) × λT∈(0, 0.005, 0.01) ×
c∈(0, 0.025, 0.05).

## 0. Frozen contract (unchanged from R1.1)
5m decision clock; H=6/12/24; next-open execution; at most one round-trip;
discontinuity ban; Long/Short/Wait/Tie. Only risk/time/friction change.

## 1. Critical DP rule
Bellman is performed ENTIRELY in price points. Penalties use the DECISION-time
ATR_t. We never feed ATR-normalized V back into the recursion.

## 2. Sample counts
- n_decisions = 505161
- runtime_sec = 201.47
- peak_rss_mb = 975.3

## 3. Headline robustness (Core region only)
- **Direction robustness** (Long<->Short direct flip, Core grid):
  joint_opposite_flip_rate_mean = 0.0
- **Opportunity robustness** (trade -> Wait/Tie suppression, Core): see
  direction_vs_suppression_core + trade_suppression_rate in one_factor_sensitivity.
- **Exact exit-timing robustness**: **NOT supported**. Direction/opportunity are
  stable but optimal exit is materially sensitive — baseline_direction_holding
  mean span = 11.39 bars, P(span>=6)=76.8%, P(span>=12)=53.9%; edge_ATR spread
  (per oracle_constraint_edge_summary.json: H6/H12/H24 median 0.383/0.541/0.808,
  p10 0.068/0.095/0.141, p90 1.069/1.514/2.307) confirms holding/exit sensitivity.
  Conclusion: model environment -> stable direction / opportunity value first;
  treat exit timing as a separate downstream problem.
- NOTE: Stress region (direction_vs_suppression_stress) is reported separately
  and only characterizes extreme-condition behaviour, never the headline.

## 3b. Research conclusions (independent review)
- **Direction robustness: SUPPORTED.** Stable-cohort joint retention mean = 0.9882;
  95.58% of stable baseline samples stay fully consistent across all Core theta x H
  cells; zero Long<->Short direct flip in Core.
- **Opportunity robustness: SUPPORTED.** Core trade->Wait/Tie suppression ~3.7%
  (Long) / 3.5% (Short); risk lambda_R=0.5 suppresses ~8.5% of baseline trades
  with no direct direction flip; time/friction effects small in tested range.
- **Exact exit-timing robustness: NOT SUPPORTED; exit timing is materially
  sensitive** (see above). This is the key structural result guiding next phase.

## 4. Joint retention distribution
Headline = **stable-cohort** joint retention (baseline action itself stable across
H6/H12/H24; Ambiguous/Tie baselines excluded). All-row diagnostic retained.

**Stable cohort** (n = 347519):
{'n': 347519, 'p10': 1.0, 'p25': 1.0, 'p50': 1.0, 'p75': 1.0, 'p90': 1.0, 'mean': 0.9882, 'frac_full': 0.9558, 'frac_above_0_8': 0.9755}

**All rows (diagnostic only, NOT the headline)**:
{'n': 505161, 'p10': 0.679, 'p25': 1.0, 'p50': 1.0, 'p75': 1.0, 'p90': 1.0, 'mean': 0.9105, 'frac_full': 0.8066, 'frac_above_0_8': 0.8391}

## 5. Strict robust action counts
{'Wait': 240067, '': 171454, 'Long': 47970, 'Short': 44129, 'Tie': 1541}

## 6. One-factor sensitivity
- risk only: [{'lambda_r': 0.0, 'lambda_t': 0.0, 'friction_hurdle_atr': 0.0, 'opposite_flip_rate': np.float64(0.0), 'opposite_flip_count': 0, 'baseline_trade_count': 449168, 'trade_suppression_rate': np.float64(0.0), 'trade_suppression_count': 0, 'trade_creation_rate': np.float64(0.0), 'trade_creation_count': 0, 'baseline_wait_count': 933165}, {'lambda_r': 0.25, 'lambda_t': 0.0, 'friction_hurdle_atr': 0.0, 'opposite_flip_rate': np.float64(0.0), 'opposite_flip_count': 0, 'baseline_trade_count': 449168, 'trade_suppression_rate': np.float64(0.0285), 'trade_suppression_count': 12790, 'trade_creation_rate': np.float64(0.0016), 'trade_creation_count': 1534, 'baseline_wait_count': 933165}, {'lambda_r': 0.5, 'lambda_t': 0.0, 'friction_hurdle_atr': 0.0, 'opposite_flip_rate': np.float64(0.0), 'opposite_flip_count': 0, 'baseline_trade_count': 449168, 'trade_suppression_rate': np.float64(0.085), 'trade_suppression_count': 38193, 'trade_creation_rate': np.float64(0.004), 'trade_creation_count': 3752, 'baseline_wait_count': 933165}]
- time only: [{'lambda_r': 0.0, 'lambda_t': 0.0, 'friction_hurdle_atr': 0.0, 'opposite_flip_rate': np.float64(0.0), 'opposite_flip_count': 0, 'baseline_trade_count': 449168, 'trade_suppression_rate': np.float64(0.0), 'trade_suppression_count': 0, 'trade_creation_rate': np.float64(0.0), 'trade_creation_count': 0, 'baseline_wait_count': 933165}, {'lambda_r': 0.0, 'lambda_t': 0.005, 'friction_hurdle_atr': 0.0, 'opposite_flip_rate': np.float64(0.0), 'opposite_flip_count': 0, 'baseline_trade_count': 449168, 'trade_suppression_rate': np.float64(0.001), 'trade_suppression_count': 454, 'trade_creation_rate': np.float64(0.0011), 'trade_creation_count': 1018, 'baseline_wait_count': 933165}, {'lambda_r': 0.0, 'lambda_t': 0.01, 'friction_hurdle_atr': 0.0, 'opposite_flip_rate': np.float64(0.0), 'opposite_flip_count': 0, 'baseline_trade_count': 449168, 'trade_suppression_rate': np.float64(0.003), 'trade_suppression_count': 1353, 'trade_creation_rate': np.float64(0.0031), 'trade_creation_count': 2918, 'baseline_wait_count': 933165}]
- friction only: [{'lambda_r': 0.0, 'lambda_t': 0.0, 'friction_hurdle_atr': 0.0, 'opposite_flip_rate': np.float64(0.0), 'opposite_flip_count': 0, 'baseline_trade_count': 449168, 'trade_suppression_rate': np.float64(0.0), 'trade_suppression_count': 0, 'trade_creation_rate': np.float64(0.0), 'trade_creation_count': 0, 'baseline_wait_count': 933165}, {'lambda_r': 0.0, 'lambda_t': 0.0, 'friction_hurdle_atr': 0.025, 'opposite_flip_rate': np.float64(0.0), 'opposite_flip_count': 0, 'baseline_trade_count': 449168, 'trade_suppression_rate': np.float64(0.0004), 'trade_suppression_count': 199, 'trade_creation_rate': np.float64(0.0002), 'trade_creation_count': 184, 'baseline_wait_count': 933165}, {'lambda_r': 0.0, 'lambda_t': 0.0, 'friction_hurdle_atr': 0.05, 'opposite_flip_rate': np.float64(0.0), 'opposite_flip_count': 0, 'baseline_trade_count': 449168, 'trade_suppression_rate': np.float64(0.0013), 'trade_suppression_count': 581, 'trade_creation_rate': np.float64(0.0004), 'trade_creation_count': 332, 'baseline_wait_count': 933165}]

## 7. Cost metadata
```json
{'canonical_table_found': False, 'NET_PNL': 'UNAVAILABLE_COST_METADATA', 'rule': 'friction hurdle c*ATR_t is NOT a real fee; only gross price-point utility + risk/time/friction penalties', 'FRICTION_HURDLE_NOT_ACTUAL_TRANSACTION_COST': True}
```
