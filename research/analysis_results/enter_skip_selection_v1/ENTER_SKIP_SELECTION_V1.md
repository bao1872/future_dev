# ENTER / SKIP Selection Gate v1.0

## Verdict

**FROZEN_CANDIDATE_POLICY / NO_UNTOUCHED_PROSPECTIVE_BLOCK**

WF1-WF3 participated in region discovery. Results below are **DEVELOPMENT_REPLAY**,
not OOS or prospective confirmation. The auxiliary RR x target plane is shadow-only.

## Frozen sequential policy

- R1 at h=2 enters OUTWARD; otherwise wait.
- At h=5, R2 enters OUTWARD or R3 enters INWARD; simultaneous match is
  `SKIP_DUAL_ACTION_CONFLICT`.
- At h=8, R4 enters OUTWARD; otherwise `SKIP_NO_GEOMETRY_EDGE`.
- One liquidity contact can enter at most once.

## Answers

1. Primary unique contact coverage: **9,015** of
   73,885 contacts (12.20%).
2. Multiple-region matches: **504**;
   dual-action conflicts: **0**;
   earlier-region supersedes a later match: **504**.
3. Whole-policy development-replay minimum gross `E[R]lower` across WF:
   **+0.012635R**. Under the -1R censor
   stress the minimum is **-0.001948R**,
   so the result is **CENSORING_SENSITIVE**.
4. Generic cost capacity on the preregistered grid: **0.01R**. This is not a
   real transaction-cost estimate; the repository has no authoritative cost metadata.
5. Untouched prospective block: **NO**. Current data ends at
   `2026-09-04 14:55:00` and TB1-TB4 exhaust it.
6. No prospective PASS/FAIL can be issued.
7. Stop at **FROZEN_CANDIDATE_POLICY** until genuinely untouched data exists.

## Development replay by WF

```
             scope  wf  n_contacts  n_trades  trade_rate  resolved_n  censored_n  censored_rate  mean_R_lower  median_R_lower  profit_factor_lower  win_rate  total_R_lower  ambiguity_rate  E_R_lower_resolved  E_R_lower_censor_worst  max_drawdown_R  final_cumulative_R
DEVELOPMENT_REPLAY WF1       25383      3160    0.124493        3060         100       0.031646      0.033192        0.428571             1.084219  0.605882     101.568489        0.000316            0.033192                0.000496      -81.463406          101.568489
DEVELOPMENT_REPLAY WF2       24692      2847    0.115301        2806          41       0.014401      0.012635        0.420204             1.030910  0.591233      35.454037        0.000351            0.012635               -0.001948      -85.980856           35.454037
DEVELOPMENT_REPLAY WF3       23810      3008    0.126333        2905         103       0.034242      0.034556        0.444444             1.088211  0.608262     100.383835        0.000332            0.034556               -0.000870     -115.292747          100.383835
```

## Censoring

`E_R_lower_resolved` preserves the frozen outcome semantics. The separate
`E_R_lower_censor_worst` stress assigns every censored trade -1R; it does not alter
the stored outcomes.

## STOP

No bin changes, classifier, morphology/liquidity search, Autoencoder, DP, or RL.
