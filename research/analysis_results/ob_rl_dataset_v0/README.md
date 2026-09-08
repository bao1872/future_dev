# OB RL Dataset V0

Offline `State x Action x Reward` dataset for OB touch research.

- State rows: 21,481 (one per OB touch)
- Action rows: 150,367 (candidates x 7 actions)
- Reward version: GROSS_R_V0 (no cost model yet)
- Primary horizon: H12 (H6 / H24 diagnostics)

## Storage

Parquet (pyarrow / zstd) is the local authoritative format. CSV is a
one-off build artifact. Neither the CSV nor the Parquet is committed;
the dataset is reproducible from the frozen V3 source data.

## Model View V0

62 whitelisted fields:
Event (6) + SMC (36) + DSA (6) + Momentum (9) + Quantile (2) +
Action (3).

Continuous values stay continuous; binning is an audit-layer concern
only.

META and WEIGHT columns are never model features.

## Missing-value contract

- Categorical active-OB fields may be filled with `NO_OB`.
- Numeric OB distance / fit stays NaN. `NO_OB` is NOT `distance = 0`.
- Quantile `UNKNOWN` is preserved, never imputed or re-ranked.

## Not in V0

- absolute pivot prices (internal/swing high/low level)
- absolute OB zone prices (zone_low / zone_high)
- all 70 dsa_raw_* fields
- last BOS / CHoCH structure type, bias and age
- group confluence flags
- quantile q10 / q50 / q90 / crossed / top30
- raw OHLC and volume / OI
- raw forward / backward ATR distances (recoverable from fit)
- future outcome diagnostics

Excluded from V0 does not mean rejected; it means deferred for later
ablation.
