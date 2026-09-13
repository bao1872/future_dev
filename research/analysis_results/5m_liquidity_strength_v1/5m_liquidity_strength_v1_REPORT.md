# 5M-LS1 — 5m Liquidity Strength & Target Selection v1 — Final Report

## Verdict

```
STOP_LIQUIDITY_STRENGTH_AT_WF1
```

(no incremental liquidity-strength signal beyond distance; WF2/WF3, selector and
full-policy rebuild are NOT executed.)

## Two guiding questions — answered

1. **Can liquidity strength be identified from data?**
   After controlling for distance / risk / RR (the Baseline geometry model), the
   strength features — cycle (scope), type, structure_size_R, period_range_R,
   freshness (level_age / prior_touch_count / time_since_prev_touch), confluence
   (identity / scope / type counts, scope×structure interactions) — add **no
   incremental** predictive power for "is this liquidity reached before the
   structural stop".

2. **Can it select a better target than "nearest liquidity"?**
   Gated out by Q1 failing. Not reached. No target-selection experiment run.

## Why this is the correct cheap stop

The Baseline (distance-only) Logistic is already **AUC = 0.9436** at WF1 — distance
alone explains almost all of "will this liquidity be touched before the 1-ATR stop".
Adding the strength feature block makes the model *marginally worse* on every
calibration / likelihood metric. That is the textbook signature of strength features
being pure noise on top of the distance residual, not a learnable signal.

## Strength metrics (WF1, full 15-symbol universe, TB1 train → TB2 test)

| wf | n_test | base_auc | full_auc | base_brier | full_brier | base_logloss | full_logloss | d_auc | d_brier | d_logloss |
|----|--------|----------|----------|------------|------------|--------------|--------------|-------|---------|-----------|
| WF1 | 812027 | 0.94356 | 0.941031 | 0.020917 | 0.021198 | 0.073531 | 0.074809 | **-0.00253** | **-0.000281** | **-0.001278** |

Early-stop rule (section 二十二): `ΔAUC ≤ 0 AND ΔBrier ≤ 0 AND ΔLogLoss ≤ 0`
→ STOP. All three deltas are negative → gate fires. WF2/WF3 not run.

## Execution facts

- starting_commit: `7e6c9cc860215a22aea665142aad25fba3487073`
- Interrupted Native15 work (`research/liquidity_oracle_atlas/timeframe_15m_native/`,
  `research/analysis_results/timeframe_15m_native_v1/`): **git-ignored** (status `!!`,
  rule `research/analysis_results/**/*.parquet` + dir contents), so it is already
  excluded from every commit and remains on disk — protected and NOT mixed into the
  5M-LS1 commit. `git stash -u` cannot capture ignored files, so no stash was
  created; the working tree is clean of Native15 from git's perspective. If the user
  later wants the Native15 tree physically shelved, use `git stash -a` (includes
  ignored) — but that is NOT required for protection.
- Pipeline order (section 一): P0 contract → P1 smoke (AG,100) → P2 pilot
  (AG/CU/RB/MA,500) → P3 WF1 (15 symbols, TB1→TB2). WF1 early-stop fired.

### Timing

- Enrichment (all 15 symbols, one-time, cached): 11.3 s
- Candidate build (15 symbols, TB1+TB2 subset): ~4.5 min (per-symbol 11–30 s)
- Model fit / eval: seconds
- Total WF1 run: ~5 min

### Candidate explosion (flagged, section 十九/四十)

- ~20–35 candidates / signal (avg ~25). 812,027 test candidates at WF1.
- 88% of candidates are far session/day/week levels 10–40 ATR away that are
  *mechanically* never reached before a 1-ATR stop (label = 0 by construction).
- Overall positive (target-first) rate ≈ 2% (nearby 0–2 ATR targets: 26–33%;
  far >5 ATR: ≈0–3%).
- Mitigation: candidates cached per symbol with a `max_signals + scope_tag` keyed
  filename; analysis loads only the blocks each WF needs (never concatenates all
  ~3.7M rows into RAM). Smoke/pilot/wf1/full caches do not collide.

## Frozen-data governance (read-only)

- 5m freeze / E1.1 freeze / pre-P1 manifest: **read-only, unchanged**.
- `P1_read = false` (S26 PASS).
- master parquet sha256 = `9a48cc35987b85970dab60219ed5e276fb85096802ada4dd56578e9597cbda4f` (S27 PASS)
- contacts parquet sha256 = `e172c39dccb8f4c71f6ec1643fa416f897320703801b4144d095d133655781ab` (S27 PASS)
- pre-P1 manifest hash = `c0b10c6ef3899b5a32ce8082128649ae8e63247d9a062f7af2b791c1253673d5` (S28 PASS)

## Contract tests S1–S28

All executed in smoke/pilot (`--mode smoke` runs run_tests). Results:
S1 active causal PASS · S2 consumed excluded PASS · S3 same-price cluster PASS ·
S4 scope flags PASS · S5 type flags PASS · S6 structure causal PASS ·
S7 prior_touch past PASS · S8 profitable direction PASS · S9 risk>0 PASS ·
S10 target-first scalar/vector parity PASS · S11 stop-first parity PASS ·
S12 ambiguity exact PASS · S13 train-only fit PASS · S14 TB order exact PASS ·
S15 no reward overlap PASS · S16 baseline/full same rows PASS ·
S17 full model no future fields PASS · S18–S21 selectors exact PASS ·
S22 equity chronological PASS · S23 maxDD recompute PASS · S24 payoff ratio PASS ·
S25 PF PASS · S26 P1_read=false PASS · S27 freeze hashes unchanged PASS ·
S28 manifest unchanged PASS.

## Does NOT proceed to

- WF2 / WF3
- strength-full / selector (T0/T1/T2 equity, bootstrap)
- policy-full (full Stage4A rebuild)
- Limit / Reassess

## Conclusion for reviewer

The experiment was designed to fail cheaply at the most likely failure point.
WF1 confirms: **after distance is controlled, the richer liquidity-strength
description (cycle / type / structure / freshness / confluence) does not help
predict whether a liquidity level is reached before the stop, and therefore
cannot be shown to improve 5m target selection.** The previously-suspected
multi-cycle "G4" field strength was, consistent with this result, UNAVAILABLE
as an incremental edge. No full 5m strategy rebuild is justified by this evidence.
