# V2-B / V2-C / V2-D: Nonlinear ML x Recency x Context x Multi-Action Value Experiment Report

**Final Verdict**: `MULTI_ACTION_Q_EDGE_SURVIVES_BUT_DOES_NOT_BEAT_FIXED_REASSESS`

## 1. Executive Summary & Gates

| Gate | Description | Status | Evidence |
|---|---|---|---|
| **G1** | `Q_POLICY_EDGE_SURVIVES` | `PASSED` | WF1=0.0162, WF2=0.0303, WF3=0.0926 |
| **G2** | `Q_POLICY_BEATS_MARKET` | `FAILED` | ΔMarket: WF1=+0.0059, WF2=-0.0028, WF3=+0.0488 |
| **G3** | `Q_POLICY_BEATS_FIXED_REASSESS` | `FAILED` | ΔReassess: WF1=-0.0180, WF2=-0.0141, WF3=-0.0115 |
| **G4** | `ML_ACTION_SELECTION_ADDS_MATERIAL_VALUE` | `FAILED` | Mean Δ=-0.01453 R/sig, 95% CI lower > 0 count: 0/3 |

### Benchmark Comparison (EV_cw per signal)

| Walk-Forward | E1.1 Market | Fixed RR=3 Reassess | Learned Q Policy | Δ vs Reassess | 95% Bootstrap CI |
|---|---:|---:|---:|---:|---:|
| **WF1** | 0.0104 | 0.0342 | 0.0162 | -0.0180 | [-0.0409, +0.0034] |
| **WF2** | 0.0331 | 0.0444 | 0.0303 | -0.0141 | [-0.0402, +0.0105] |
| **WF3** | 0.0438 | 0.1041 | 0.0926 | -0.0115 | [-0.0382, +0.0153] |

## 2. Selected Architectures & Action Mix

```
model_type  half_life feature_block  selected_score  wf
   HistGBR        480            S2       -0.034461 WF1
   HistGBR        480            S1        0.042479 WF2
    Linear         60            S1        0.036351 WF3
```

Action distribution across test splits:
```
action  LIMIT_RR3    MARKET  REASSESS_RR3      SKIP
wf                                                 
WF1      0.170570  0.243987      0.259494  0.325949
WF2      0.217422  0.219178      0.246575  0.316825
WF3      0.258644  0.394614      0.156582  0.190160
```

## 3. Policy Collapse Check
- Max share WF1: 32.6%
- Max share WF2: 31.7%
- Max share WF3: 39.5%
- Status: `HEALTHY_STATE_DEPENDENT_SWITCHING`

## 4. Verification and Governance
- P1 read: `False` (all sample times <= 2026-09-04 14:55:00)
- Tests passed: 14/14
