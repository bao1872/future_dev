# Oracle Constraint Robustness v1 (R2)

> **Status**: `PROVISIONAL_PENDING_USER_AUDIT`. This is a SENSITIVITY /
> ROBUSTNESS surface, not parameter tuning. No "best lambda" is selected.

**Task**: `FUTURE-ORACLE-R2-CONSTRAINT-ROBUSTNESS`
**Base**: 1211e7117955d794a59e1df0d30849db02900db1 (R1.1)
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
- runtime_sec = 155.19
- peak_rss_mb = 883.8

## 3. Headline robustness (user's three layers)
- **Direction robustness** (Long<->Short direct flip):
  joint_opposite_flip_rate_mean = 0.0
- **Opportunity robustness** (trade -> Wait/Tie suppression): see
  trade_suppression_rate in one_factor_sensitivity + parameter_summary.
- **Timing robustness** (action same but holding/exit sensitive):
  edge_ATR / value_ATR min-median-max per (theta, H) in rows + parameter_summary.

## 4. Joint retention distribution
{'n': 505161, 'p10': 0.679, 'p25': 1.0, 'p50': 1.0, 'p75': 1.0, 'p90': 1.0, 'mean': 0.9105, 'frac_full': 0.8066, 'frac_above_0_8': 0.8391}

## 5. Strict robust action counts
{'Wait': 240067, '': 171454, 'Long': 47970, 'Short': 44129, 'Tie': 1541}

## 6. One-factor sensitivity
- risk only: [{'lambda_r': 0.0, 'lambda_t': 0.0, 'friction_hurdle_atr': 0.0, 'opposite_flip_rate': np.float64(0.0), 'trade_suppression_rate': np.float64(0.0), 'trade_creation_rate': np.float64(0.0)}, {'lambda_r': 0.25, 'lambda_t': 0.0, 'friction_hurdle_atr': 0.0, 'opposite_flip_rate': np.float64(0.0), 'trade_suppression_rate': np.float64(0.0084), 'trade_creation_rate': np.float64(0.001)}, {'lambda_r': 0.5, 'lambda_t': 0.0, 'friction_hurdle_atr': 0.0, 'opposite_flip_rate': np.float64(0.0), 'trade_suppression_rate': np.float64(0.0252), 'trade_creation_rate': np.float64(0.0025)}]
- time only: [{'lambda_r': 0.0, 'lambda_t': 0.0, 'friction_hurdle_atr': 0.0, 'opposite_flip_rate': np.float64(0.0), 'trade_suppression_rate': np.float64(0.0), 'trade_creation_rate': np.float64(0.0)}, {'lambda_r': 0.0, 'lambda_t': 0.005, 'friction_hurdle_atr': 0.0, 'opposite_flip_rate': np.float64(0.0), 'trade_suppression_rate': np.float64(0.0003), 'trade_creation_rate': np.float64(0.0007)}, {'lambda_r': 0.0, 'lambda_t': 0.01, 'friction_hurdle_atr': 0.0, 'opposite_flip_rate': np.float64(0.0), 'trade_suppression_rate': np.float64(0.0009), 'trade_creation_rate': np.float64(0.0019)}]
- friction only: [{'lambda_r': 0.0, 'lambda_t': 0.0, 'friction_hurdle_atr': 0.0, 'opposite_flip_rate': np.float64(0.0), 'trade_suppression_rate': np.float64(0.0), 'trade_creation_rate': np.float64(0.0)}, {'lambda_r': 0.0, 'lambda_t': 0.0, 'friction_hurdle_atr': 0.025, 'opposite_flip_rate': np.float64(0.0), 'trade_suppression_rate': np.float64(0.0001), 'trade_creation_rate': np.float64(0.0001)}, {'lambda_r': 0.0, 'lambda_t': 0.0, 'friction_hurdle_atr': 0.05, 'opposite_flip_rate': np.float64(0.0), 'trade_suppression_rate': np.float64(0.0004), 'trade_creation_rate': np.float64(0.0002)}]

## 7. Cost metadata
```json
{'canonical_table_found': False, 'NET_PNL': 'UNAVAILABLE_COST_METADATA', 'rule': 'friction hurdle c*ATR_t is NOT a real fee; only gross price-point utility + risk/time/friction penalties', 'FRICTION_HURDLE_NOT_ACTUAL_TRANSACTION_COST': True}
```
