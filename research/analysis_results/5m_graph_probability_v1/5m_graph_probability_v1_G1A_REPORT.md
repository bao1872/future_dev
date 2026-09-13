# 5M-GPM1 — G1a: Liquidity Timeframe Identity Increment — Report

## Verdict

```
NO_TIMEFRAME_IDENTITY_INCREMENT_WF1
```

After controlling ordered-chain geometry, adjacent liquidity distance, cumulative
distance, edge position, direction, symbol and contact type (the frozen G0), adding
the 6 raw multi-hot timeframe flags **5m / 15m / 1h / Session / Day / Week** does
**NOT** provide a stable WF1 OOS increment. The stricter G1a gate (both NLL and
Brier must improve AND both bootstrap 95% CIs must lie above 0) fails on C and D.

This is a clean negative result. It does **NOT** block G1b (type) / G1c (structure
size) / G1d (freshness/confluence), and it does **NOT** negate the G0 graph-structure
finding (still `GRAPH_STRUCTURE_NECESSITY_WF1_HARDENED_PASS`).

## Method (frozen except the one added feature set)

- Reused hardened `TB12_HARDENED` transition cache — **no transition rebuild**.
- WF1 contract unchanged: `train = TB1`, `test = TB2`, strict whole-path purge
  (`signal_resolution_end_time < TB2_start_time`).
- G0 frozen: `G0_NUM=[delta_R, cum_distance_R, risk_R]`, `G0_CAT=[edge_index,
  direction, symbol, contact_type]`.
- G1a: `G1A_NUM = G0_NUM + [has_5m, has_15m, has_1h, has_session, has_day,
  has_week]` (raw 0/1 numeric; NUM column is SimpleImputer-only, NOT scaled, so
  coefficients are on the 0/1 scale), `G1A_CAT = G0_CAT`.
- Same `LogisticRegression(l2, C=1.0, lbfgs, 3000)`, same `[NEXT, LOSS, CENSOR]`.
- Same-sample contract: G0 and G1a are fit on the identical `tr_purged` and
  evaluated on the identical `te` (no per-model sample drift). The purge-count
  guard reproduced the hardened G0 baseline exactly → no sample drift.

## Reporting answers

1. **WF1 TB2 start** = `2025-06-11 00:00:00` (same as G0).
2. **TB1 original train signals** = `21165` (matches hardened G0 baseline).
3. **Purged** = `30` (matches).
4. **Purge rate** = `0.00142` (0.14%) (matches).
5. **After purge, last train path ends before** `2025-06-11 00:00:00` (same as G0).
6. **G0 vs G1a (test = TB2):**

   | metric | G0 | G1a | Δ = G0 − G1a |
   |--------|----|----|--------------|
   | joint NLL / signal (primary) | 1.680916 | 1.681977 | **−0.00106** (G1a worse) |
   | joint NLL / edge (secondary) | 0.749765 | 0.750238 | −0.00047 |
   | equal-signal Brier (primary) | 0.228656 | 0.228452 | +0.00020 (negligible) |
   | path-sum Brier (secondary) | 0.683454 | 0.683378 | +0.00008 |
   | edge logloss | 0.749765 | 0.750238 | −0.00047 |

   G0's `joint NLL/signal = 1.680916` is identical to the hardened G0 run
   (476a817) → confirms the same frozen sample.

7. **Paired trading-day bootstrap (500 resamples, Δ = G0 − G1a):**

   | | mean | 95% CI | P(Δ>0) |
   |---|---|---|---|
   | ΔNLL | +0.00128 | [−0.0029, +0.0069] | 0.674 |
   | ΔBrier | +0.00044 | [−0.0003, +0.0012] | 0.848 |

   Both CIs **straddle 0** → no stable increment.

8. **G1a gate:** A(ΔNLL>0)=**False**, B(ΔBrier>0)=True, C(ΔNLL CI lo>0)=**False**,
   D(ΔBrier CI lo>0)=**False** → `NO_TIMEFRAME_IDENTITY_INCREMENT_WF1`.
9. **P1_read = false** — runner unchanged from 476a817; freeze read-only contract
   intact (S26 PASS, unchanged).
10. **Freeze hashes unchanged** (S27/S28 PASS): master `9a48cc…bda4f`, contacts
    `e172c39…781ab`, manifest `c0b10c6…73d5`.

## Scope coverage (TB2 test; all non-trivial — none dropped/merged)

| scope | n_edges | n_signals | NEXT% | LOSS% | CENSOR% | avg delta_R | avg cum_distance_R |
|-------|--------:|---------:|------:|------:|--------:|------------:|-------------------:|
| has_5m | 19022 | 12836 | 0.571 | 0.314 | 0.115 | 1.51 | 2.97 |
| has_15m | 7896 | 6341 | 0.556 | 0.313 | 0.131 | 1.65 | 3.25 |
| has_1h | 3196 | 2889 | 0.512 | 0.334 | 0.154 | 1.94 | 3.75 |
| has_session | 53295 | 24393 | 0.554 | 0.323 | 0.123 | 1.68 | 3.04 |
| has_day | 22847 | 14599 | 0.540 | 0.323 | 0.137 | 1.74 | 3.20 |
| has_week | 6786 | 5586 | 0.506 | 0.318 | 0.176 | 1.87 | 3.55 |

## Descriptive coefficients (multinomial Logistic; NOT weights — multi-hot correlation)

| scope | NEXT coef | LOSS coef | CENSOR coef |
|-------|----------:|----------:|------------:|
| has_5m | −0.010 | +0.033 | −0.023 |
| has_15m | +0.138 | −0.029 | −0.109 |
| has_1h | −0.346 | +0.069 | +0.277 |
| has_session | −0.051 | −0.043 | +0.095 |
| has_day | +0.061 | −0.018 | −0.043 |
| has_week | −0.096 | −0.078 | +0.174 |

Reading: higher-timeframe flags tilt slightly toward CENSOR (e.g. has_week
CENSOR%=17.6% vs has_5m 11.5%; CENSOR coef +0.174). But these are tiny and NOT
statistically stable (bootstrap CIs on the OOS increment cross 0). The raw
descriptive hit rates must **not** be read as independent timeframe value — the
only valid conclusion is the G0→G1a OOS increment above.

## Interpretation (per user spec)

- **G1a FAIL means:** the ordered graph structure is valuable, but adding the
  raw 5m/15m/1h/session/day/week identity alone (as multi-hot flags) does not
  give a stable OOS increment over G0.
- **G1a FAIL ≠ the whole liquidity-identity hypothesis fails.** It only tells us
  this particular featureization of timeframe identity (indicator-present flags)
  carries no extra signal once geometry/order/distance are already controlled.
- No monotonic constraint was imposed; if the data had shown 1h/Day/Week adding
  value while 5m/15m did not, it would have been reported as-is. It did not.
- No claim of trading-edge improvement (G1a is still a mechanism/probability
  experiment; TRADING_METRICS: NOT_APPLICABLE).

## Why this is informative, not a dead end

The negative is specific: the *indicator-presence* flags add nothing. Plausible
reasons the broader identity question stays open:
- The signal may live in **structure size within a timeframe** (G1c) rather than
  mere presence.
- Or in **liquidity type** (Swing/EQH-EQL/Session-Day-Week H-L; G1b).
- Or in **freshness / multi-timeframe confluence** (G1d) — e.g. a 1h swing that
  also coincides with a Day High may carry value that a flat `has_1h && has_day`
  multi-hot cannot express.

## Governance

- Stopped here. **G1b NOT auto-started** (per user authorization of G1a only).
- No WF2/WF3, no selector, no GNN/RL.
- Commit (this round): `research: test liquidity timeframe identity increment`;
  push `origin/main`.
- Suggested next (user-authorized separately): G1b liquidity type, then G1c
  structure size, then G1d freshness/confluence — each tested independently from
  G0, only blocks with isolated increment enter a future G1-FINAL.

## Artifacts

- `g1a_wf1_summary.csv` (committed) — G0 vs G1a metrics + gate.
- `g1a_scope_descriptive.csv` (committed) — per-scope coverage/hit-rates.
- `g1a_scope_coefficients.csv` (committed) — per-scope multinomial coefs.
- `g1a_signal_losses_wf1.parquet` (gitignored) — per-signal paired losses for
  later G1-FINAL combination.
