# 5M-GPM1 — 5m Liquidity Graph Probability Model v1 — Report

## Verdict (this turn: P0 → P1 → P2 → WF1 G0)

```
GRAPH_STRUCTURE_NECESSITY_GATE @ WF1: PASSED
  G0 (ordered-chain graph geometry) improves BOTH primary metrics vs
  M0 (independent distance geometry) on the clean TB1->TB2 out-of-block test.
  -> graph structure carries OOS information beyond independent distance.
  -> report to user; do NOT auto-start G1 / WF2 / WF3 (await authorization).
```

This supersedes the INVALIDATED 5M-LS1 (`66bff6d`), whose `target-first`
label used `tfirst = in_t & ~in_s` (P0 bug: missed `first_target < first_stop`).

## Two guiding questions (this turn answers #1 structurally)

1. **Does representing liquidity as an ordered chain (instead of 25 independent
   samples) itself create new OOS information value?**
   **YES, at WF1.** Graph geometry (delta_R + chain position) beats independent
   distance geometry on joint NLL (−0.34) and signal-weighted Brier (−0.099).

2. **Does liquidity *identity* (scope/type/structure/freshness/confluence) add
   incremental value over the graph structure?** → **G1, not yet run** (pending
   authorization). This turn only establishes that the *ordered-chain* premise
   is worth pursuing.

3. **Does graph target selection beat Nearest / Independent EV?** → **selector
   stage, not yet run** (T0/T1/T2, requires G1 + WF1/WF2/WF3 first).

## Label fix (P0)

Corrected competing-risk label (parity-checked against frozen
`first_hit_bounds`):
```python
target_first = first_target < first_stop
stop_first   = first_stop  < first_target
ambiguous    = first_target == first_stop   (conservative -> LOSS, -1R)
censored     = neither hit within 34-bar window
```
P0 parity on AG (92 signal×target pairs + synthetic A/B/C/D): **0 mismatch** on
target_first / stop_first / ambiguous / censored vs the frozen oracle.

## Graph construction (ordered liquidity chain + competing risk)

- Same-price liquidity collapse → one price cluster (multi-identity).
- Nodes sorted by `direction*(price-decision_price)` ascending (profitable chain).
- Chain: ROOT → L1 → L2 → … → terminal(LOSS | CENSOR). A node after the terminal
  edge is NOT added to the risk set (standard survival semantics) → candidate
  explosion disappears.
- Each edge ∈ {NEXT, LOSS, CENSOR}; AMBIGUOUS → LOSS (conservative economics).
- Propagation asserts `p_reach[k+1] <= p_reach[k]` and
  `p_reach + p_loss + p_censor ≈ 1` per chain.

## P1 / P2 engineering validation

- edges/signal = **2.14** (AG/CU/RB/MA × 500) vs ~25 candidate/signal in LS1 →
  candidate explosion resolved by construction.
- Transition state mix (pilot, all 4 symbols): **NEXT 54.6% / LOSS 32.8% /
  CENSOR 12.6%** — all three states real, not mechanical zeros.
- S1–S28 contract tests PASS (incl. real parity S10–S12, graph invariants).

## WF1 Graph Necessity Gate (15 symbols, TB1 train → TB2 test)

| wf | n_test_edges | n_test_signals | m0_joint_nll | g0_joint_nll | m0_brier | g0_brier | m0_edge_logloss | g0_edge_logloss | d_joint_nll | d_brier |
|----|-------------|---------------|--------------|--------------|----------|----------|-----------------|---------------|------------|---------|
| WF1 | 55945 | 24954 | 2.02212 | 1.68122 | 0.78176 | 0.68285 | 0.90196 | 0.74990 | **+0.34090** | **+0.09891** |

Primary metrics (lower is better): G0 joint NLL 1.681 vs M0 2.022; G0
signal-weighted Brier 0.683 vs M0 0.782. Both deltas > 0 → gate passes.

Gate rule (section twelve): if G0 joint NLL not < M0 AND G0 Brier not < M0 →
STOP_GRAPH_STRUCTURE_AT_WF1. Here BOTH improve → **graph structure has increment**.

> Note: because the transition cache was built for all 4 blocks (the wf1 mode
> omitted the TB1+TB2 restriction in `main`), WF2/WF3 were also computable and
> show consistent positive deltas (d_joint_nll +0.330 / +0.340; d_brier +0.094 /
> +0.088). These are **informational only**; per the staged plan they are NOT a
> basis for proceeding — G1 / WF2 / WF3 require explicit user authorization.

## Interpretation for reviewer

- The win is **graph structure** (chain order + delta_R + edge position), not
  yet liquidity *identity*. The independent-distance baseline (M0) already
  captures pure distance; G0 adds the *ordinal relationship* between successive
  liquidity levels, which the data says is informative.
- This answers the user's core hypothesis: turning 25 independent samples into an
  ordered competing-risk path yields new OOS signal. It does NOT yet justify a
  full 5m rebuild — that needs G1 (identity) + selector (T0/T1/T2) + the WF1/WF2/WF3
  sequence.
- No `SKIP`, no GNN/NN/RL, no class_weight tuning (real probabilities intended).

## Frozen-data governance (read-only)

- 5m freeze / E1.1 freeze / pre-P1 manifest: **unchanged**.
- `P1_read = false` (S26 PASS).
- master sha256 = `9a48cc35987b85970dab60219ed5e276fb85096802ada4dd56578e9597cbda4f` (S27)
- contacts sha256 = `e172c39dccb8f4c71f6ec1643fa416f897320703801b4144d095d133655781ab` (S27)
- pre-P1 manifest hash = `c0b10c6ef3899b5a32ce8082128649ae8e63247d9a062f7af2b791c1253673d5` (S28)
- Old `66bff6d` LS1 left in history, report marked INVALIDATED.

## Next (requires user authorization)

- G1: add scope/type/structure/freshness/confluence to G0; compare G1−G0.
- If G1 passes: WF1/WF2/WF3 full, then T0 Nearest / T1 Indep-EV / T2 Graph-EV
  selector with equity curves + bootstrap, then Dynamic Graph Reassess vs old
  RR3.
