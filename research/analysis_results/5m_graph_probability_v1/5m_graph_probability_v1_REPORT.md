# 5M-GPM1 — 5m Liquidity Graph Probability Model v1 — Report (causal-hardening round)

## Verdict (this round: P0 → P1 → P2 → WF1 G0, hardened)

```
GRAPH_STRUCTURE_NECESSITY_WF1_HARDENED_PASS
```

After fixing the 4 audit gaps (real `resolution_time`, real outer-WF purge,
signal-equal Brier, WF1 build scope = TB1+TB2 only), the Graph Necessity Gate
still passes on **all three** criteria. The previous `258939d` conclusion is
upgraded from `PRELIMINARY_PASS_PENDING_CAUSAL_HARDENING` to `HARDENED_PASS`.

This round is **pure hardening** — no model/feature/horizon change, no G1, no
WF2/WF3, no selector. G1 remains PAUSED (pending explicit user authorization).

## The 4 audit fixes (this round)

1. **Real `resolution_time`.** Transition `resolution_time` is now the bar-end
   time (`bar_start_time + 5min`) of the actual resolution bar, not `decision_time`:
   - NEXT: `entry_bar + first_target_index`
   - LOSS (stop-first): `entry_bar + first_stop_index`
   - LOSS (ambiguous): `entry_bar + first_target_index` (== stop bar)
   - CENSOR: last actually-read child bar (`window_start + W - 1`)
   `signal_resolution_end_time = max(edge resolution_time)` shared by all edges
   of a signal (whole-path purge key).
2. **Real outer-WF purge (S15 executable).** TB1 train signals kept only if
   `signal_resolution_end_time < TB2_start_time`; otherwise the WHOLE signal is
   dropped (a graph path is a joint sample — cannot keep half).
3. **Signal-equal primary Brier.** `brier_signal_equal = mean_i( mean_j (p_ij-y_ij)^2 )`
   (each signal weight = 1). Old `path-sum brier / n_signal` kept as secondary.
4. **WF1 build scope = TB1+TB2 only** (`contacts_build` filter; new cache tag
   `TB12_HARDENED`). No TB3/TB4 caches; WF2/WF3 not run.

## P0 label + resolution parity (frozen oracle)

- Classification parity vs frozen `first_hit_bounds`: AG 50 → 189 pairs, AG 300
  → 4865 pairs, **0 mismatch** on target/stop/ambiguous/censored.
- Resolution parity vs independent raw-bar recompute: **0 mismatch** on
  `resolution_bar_index` and `resolution_time` (both AG 50 and AG 300).
- Synthetic A/B/C/D all correct.

## P1 / P2 engineering validation

- edges/signal = **2.14** (AG/CU/RB/MA × 500) vs ~25 candidate/signal in LS1 →
  candidate explosion resolved by construction (survival/risk-set semantics).
- Transition states (pilot): **NEXT 54.6% / LOSS 32.8% / CENSOR 12.6%** — all
  three real, not mechanical zeros.
- S1–S28 PASS (incl. real S15 purge, real resolution parity, graph invariants).

## WF1 hardened Gate — answers to the 15 reporting questions

1. **WF1 TB2 start time** = `2025-06-11 00:00:00` (first TB2 trading day).
2. **TB1 original train signals** = `21165`.
3. **Purged** = `30`.
4. **Purge rate** = `0.00142` (0.14%) — these are TB1 signals decided near the
   TB1/TB2 boundary whose 34-bar path resolved on/after 2025-06-11 (pre-purge
   max resolution = `2025-06-11 11:15:00`, which is exactly the boundary signals
   removed).
5. **After purge, last train path ends before** `2025-06-11 00:00:00` (every
   kept signal satisfies `signal_resolution_end_time < test_start_time`).
6. **M0 vs G0 (test = TB2; train = purged TB1):**

   | metric | M0 | G0 | Δ |
   |--------|----|----|---|
   | joint NLL / signal (primary) | 2.02233 | 1.68092 | **+0.34141** |
   | joint NLL / edge (secondary) | 0.90205 | 0.74976 | +0.15229 |
   | equal-signal Brier (primary) | 0.28046 | 0.22866 | **+0.05180** |
   | path-sum Brier (secondary) | 0.78214 | 0.68345 | +0.09869 |
   | edge logloss | 0.90205 | 0.74976 | +0.15229 |

7. **Paired trading-day bootstrap (500 resamples, Δ = M0 − G0):**
   - ΔNLL: mean +0.4090, 95% CI [+0.2869, +0.5926], P(Δ>0)=1.0
   - ΔBrier: mean +0.0565, 95% CI [+0.0509, +0.0630], P(Δ>0)=1.0

8. **G0 hardened gate = PASS.** Gate A (Δ joint_NLL/signal > 0) ✓; Gate B
   (Δ equal-signal Brier > 0) ✓; Gate C (at least one bootstrap 95% CI lower > 0)
   ✓ (both CIs far above 0). → `GRAPH_STRUCTURE_NECESSITY_WF1_HARDENED_PASS`.

9. **P1_read = false** (S26 PASS).
10. **Freeze hashes unchanged** (S27/S28 PASS):
    - master `9a48cc...bda4f`, contacts `e172c39...781ab`, manifest `c0b10c6...73d5`.

## Interpretation for reviewer

- The honest **signal-equal Brier** improvement is **+0.052**, not the old
  chain-length-inflated **+0.099** (path-sum). After the weighting fix, G0 still
  wins robustly — so the conclusion is *more* credible, not less. The joint-NLL
  improvement (+0.341) is unchanged in direction.
- Graph structure (local spacing `delta_R` + chain position `edge_index` + order)
  carries genuine OOS increment beyond independent distance, under strict causal
  purge + equal-signal Brier + trading-day bootstrap.
- This answers the core hypothesis: representing liquidity as an ordered chain
  (instead of 25 independent samples) creates new OOS information value. It does
  **NOT** yet establish liquidity *identity* increment (G1), nor any trading
  P&L edge (selector/T0-T2 — not run).
- No `SKIP`, no GNN/NN/RL, no class_weight tuning (real probabilities intended).

## Why the result is now trustworthy where the old one was not

| gap | old `258939d` | this round |
|-----|---------------|------------|
| `resolution_time` | = decision_time | = bar-end of resolved bar |
| outcome purge | S15 placeholder `True` | real whole-signal purge (30 removed) |
| Brier weighting | path-sum (long chains ×5) | signal-equal primary + path-sum secondary |
| WF1 build scope | all 4 blocks (leaked WF2/WF3) | TB1+TB2 only |

## Frozen-data governance (read-only)

- 5m freeze / E1.1 freeze / pre-P1 manifest: unchanged.
- Old `66bff6d` (LS1) report marked INVALIDATED; committed history preserved.

## Next (requires user authorization)

- **G1a — period/timeframe identity (5m/15m/1h/session/day/week flags on top of G0):
  TESTED 2026-09-13 → `NO_TIMEFRAME_IDENTITY_INCREMENT_WF1`.** Adding the 6 raw
  multi-hot scope flags does NOT give stable OOS increment over G0 (ΔNLL=−0.00106,
  ΔBrier=+0.00020; both bootstrap CIs straddle 0; gates C/D fail). Same frozen
  sample confirmed (purge 21165→21135/30 matches G0; G0 NLL 1.680916 identical).
  Detailed in `5m_graph_probability_v1_G1A_REPORT.md`. G1a FAIL ≠ whole liquidity
  identity hypothesis fails; it only rules out this featureization.
- **G1b — liquidity type** (Swing, EQH/EQL, Session/Day/Week H/L): NOT started,
  pending user authorization. Tested independently from G0 (not chained on G1a).
- **G1c — structure size** (structure_size_R, period_range_R, confluence): deferred.
- **G1d — freshness / prior-touch / multi-TF confluence**: deferred.
- Each G1 block compared G1x − G0 independently; only blocks with isolated increment
  enter a future `G1-FINAL`.
- Only if a G1 block passes: WF1/WF2/WF3 full, then T0 Nearest / T1 Indep-EV /
  T2 Graph-EV selector with equity curves + bootstrap (no SKIP first round).
- Then (if static Graph EV increment holds): Dynamic Graph Reassess vs old RR3.

## Commit / governance

- `fix: harden 5m graph probability WF causality` + `research: close hardened WF1
  graph-structure gate` (one commit), pushed `origin/main`.
- IDE stops here; G1 not auto-started.
