# Short-Swing Futures ML Research V1

- Decision cycle: 15m
- Primary target: normalized future return
- Horizons: [2, 4, 8, 16]
- SMC / BOS / CHoCH / SQZMOM as features: **NO**
- Oracle as model feature: **NO**
- Validation: expanding OOS + horizon-sized purge
- Hyperparameter search: **NO**
- Backtest: **NO**

## Best factual result by horizon

| Horizon | OOS R² | Model | Feature set | Spearman IC | AUC | D10-D1 raw |
|---:|---:|---|---|---:|---:|---:|
| 2 (30m) | -0.0015 | rf | stats_15m_1h | -0.0061 | 0.4917 | 0.000550 |
| 4 (60m) | -0.0065 | rf | stats_15m_1h | -0.0100 | 0.4902 | 0.000310 |
| 8 (120m) | -0.0049 | rf | stats_15m | -0.0191 | 0.4867 | -0.000416 |
| 16 (240m) | -0.0069 | rf | stats_15m | -0.0317 | 0.4824 | -0.000191 |

## Guardrails

- Positive OOS R² is stronger evidence than in-sample fit.
- Decile spread must be read with IC and monotonicity.
- One good model/horizon is not enough to claim a tradable strategy.
- Overlapping horizons mean this run makes no naive t-stat significance claim.
- Next step should be driven by cross-fold / cross-horizon stability.
