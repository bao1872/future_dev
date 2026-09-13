# 5M-GPM1 — G1b-core (formation type) & G1c0 (type-controlled structure size) — Report

## Verdict

```
G1b-core : NO_FORMATION_TYPE_INCREMENT_WF1
G1c0     : NO_STRUCTURE_SIZE_INCREMENT_WF1
```

Both blocks FAIL the strict 4-gate (ΔNLL>0, ΔBrier>0, both bootstrap 95% CIs lo>0).
Per the user's governance: **G1c0 FAIL ⇒ do NOT authorize the expensive G1c1
per-timeframe rebuild. Next candidate = G1d (freshness / confluence).**

## Method (frozen except the added feature sets)

- Reused hardened `TB12_HARDENED` transition cache — **no transition rebuild**.
- WF1 contract unchanged: `train = TB1`, `test = TB2`, strict whole-path purge.
- G0 frozen. New feature sets:

  ```
  G1b-core : G0_NUM + [has_swing, has_eq]                       (NO has_prev_* — those re-encode timeframe scope = G1a)
  C0 (base): G0_NUM + [has_swing, has_eq, has_prev_session,
                        has_prev_day, has_prev_week] + [structure_size_missing, period_range_missing]
  C1 (aug) : C0 + [structure_size_R, period_range_R]
  ```
- Missing flags `structure_size_missing`, `period_range_missing` are computed from
  the cached magnitudes and added to **both** C0 and C1 identically, so C1's only
  new information is the magnitude — presence can no longer leak formation type.
- Same `LogisticRegression(l2, C=1.0, lbfgs, 3000)`, same `[NEXT, LOSS, CENSOR]`.
- Same-sample contract: each pair (G0/G1b-core, C0/C1) fit on identical `tr_purged`
  and evaluated on identical `te`. Feature-set asserts verify the declared columns.
- Sample-drift guard reproduced the hardened G0 baseline exactly (see below).

## Reporting answers

1. **WF1 TB2 start** = `2025-06-11 00:00:00` (same as G0).
2. **TB1 original train signals** = `21165` (matches hardened G0 baseline).
3. **Purged** = `30` (matches).
4. **Purge rate** = `0.00142` (matches).
5. **After purge, last train path ends before** `2025-06-11 00:00:00` (same as G0). → no sample drift.
6. **Baseline vs augmented (test = TB2, full 15 symbols):**

   | block | base model | base NLL/sig | aug NLL/sig | ΔNLL | base Brier | aug Brier | ΔBrier |
   |-------|-----------|-------------|-------------|------|-----------|-----------|--------|
   | G1b-core | G0 | 1.680916 | 1.681025 | **−0.00011** | 0.228656 | 0.228636 | +0.00002 |
   | G1c0 | C0 | 1.681078 | 1.680670 | +0.00041 | 0.228648 | 0.228756 | **−0.00011** |

   (G0's `joint NLL/signal = 1.680916` is identical to the hardened G0 run → same frozen sample.)

7. **Paired trading-day bootstrap (500 resamples, Δ = base − aug):**

   | block | metric | mean | 95% CI | P(Δ>0) |
   |-------|--------|------|--------|--------|
   | G1b-core | ΔNLL | +0.00008 | [−0.0002, +0.0004] | 0.676 |
   | G1b-core | ΔBrier | +0.00010 | [+0.00002, +0.0002] | 1.000 |
   | G1c0 | ΔNLL | +0.00082 | [−0.0006, +0.0024] | 0.854 |
   | G1c0 | ΔBrier | +0.00005 | [−0.0002, +0.0003] | 0.588 |

8. **Gates:** G1b-core A=F B=T C=F D=T → FAIL (A,C fail). G1c0 A=T B=F C=F D=F → FAIL.
9. **P1_read = false** — runner unchanged from 72c16cd; freeze read-only contract intact.
10. **Freeze hashes unchanged** (S27/S28 PASS): master `9a48cc…bda4f`, contacts
    `e172c39…781ab`, manifest `c0b10c6…73d5`.

## Smoke / Pilot (validation only — not economic conclusions)

- Smoke AG 100: G1b-core dNLL=−0.034, dBr=+0.006; G1c0 dNLL=−0.117, dBr=−0.003 (both NO).
  Mechanics, missing flags, same-sample, bootstrap all run.
- Pilot AG/CU/RB/MA × 500: G1b-core dNLL=−0.00307, dBr=+0.00001; G1c0
  dNLL=+0.00336, dBr=+0.00079 — point estimates near zero, both bootstrap CIs
  straddle 0 → both NO. Confirms full-run direction at larger scale.

## Descriptive quintile audit (TB2, train cutpoints — NOT a gate)

### swing `structure_size_R` (present_col = has_swing)

| Q | n_edges | NEXT% | LOSS% | CENSOR% | avg delta_R |
|---|--------:|------:|------:|--------:|------------:|
| 1 (smallest) | 3624 | 0.606 | 0.281 | 0.113 | 1.10 |
| 2 | 3486 | 0.590 | 0.317 | 0.093 | 1.33 |
| 3 | 2950 | 0.559 | 0.325 | 0.116 | 1.34 |
| 4 | 2432 | 0.540 | 0.355 | 0.105 | 1.89 |
| 5 (largest) | 2670 | **0.450** | 0.365 | **0.185** | **2.50** |

→ Clear monotonic gradient: **bigger swing structures → LOWER NEXT%, HIGHER CENSOR%,
larger delta_R**. This directly contradicts the "bigger structure = stronger/more
important liquidity that reaches the next target" intuition: larger swing structures
look *more exhausted* (already moved further), not more potent.

### `period_range_R` (session/day/week levels)

| Q | n_edges | NEXT% | CENSOR% | avg delta_R |
|---|--------:|------:|--------:|------------:|
| 1 (smallest) | 12653 | 0.535 | 0.139 | 1.99 |
| 2 | 10387 | 0.591 | 0.109 | 1.41 |
| 3 | 10111 | 0.576 | 0.100 | 1.37 |
| 4 | 10086 | 0.554 | 0.127 | 1.71 |
| 5 (largest) | 10607 | 0.521 | 0.132 | 1.83 |

→ Mild inverted-U, no clean monotonic structure-magnitude signal.

**Why this is still a negative result on the formal gate:** the quintile gradient is
*raw* (ignores G0 geometry). The model (G1c0) controls for geometry/order/distance
and formation type via C0, then asks whether the magnitude adds *independent* OOS
signal. It does not — bootstrap CIs straddle 0. So the descriptive gradient is an
interesting observation, but it is NOT evidence of an isolated structure-size
increment that would justify a model feature.

## Interpretation

- **G1b-core FAIL:** once ordered-chain geometry, adjacent/distance, edge position,
  direction, symbol and contact type are controlled, the bare Swing-vs-EQ formation
  *type* carries no stable OOS increment.
- **G1c0 FAIL:** after also controlling formation type (and presence via missing
  flags), the *magnitude* of structure size / period range adds no stable isolated
  increment.
- **Combined with G1a (NO_TIMEFRAME_IDENTITY_INCREMENT_WF1):** the coarse liquidity
  identity axes tested so far — timeframe (G1a), formation type (G1b-core), and
  structure magnitude (G1c0) — each independently fail to beat G0 under the strict
  gate. The G0 graph-structure finding (`GRAPH_STRUCTURE_NECESSITY_WF1_HARDENED_PASS`)
  remains the only supported increment.
- **Do NOT over-read the quintile gradient as a win.** It is descriptive, not a gate
  result, and it actually runs *against* the bigger-structure hypothesis at the
  NEXT/LOSS/CENSOR endpoint.

## Governance / next

- Stopped here. Per user (M): G1c0 FAIL ⇒ **skip the expensive G1c1 per-timeframe
  rebuild** (scope_max_struct_5m/15m/1h… need not be written to cache yet).
- Suggested next (user authorization required): **G1d — freshness / prior-touch /
  multi-timeframe confluence** (age, prior_touch count, multi-TF aligned count),
  each tested independently from G0.
- No WF2/WF3, no selector, no GNN/RL.
- Commit (this round): `research: test liquidity type and structure-size increments`;
  push `origin/main`.

## Artifacts

- `g1bc_summary.csv` (committed) — G1b-core + G1c0 metrics + gates.
- `g1bc_structure_quintile.csv` (committed) — descriptive per-quintile audit.
- `g1bc_signal_losses_wf1.parquet` (gitignored) — per-signal paired losses for
  later G1-FINAL combination.
