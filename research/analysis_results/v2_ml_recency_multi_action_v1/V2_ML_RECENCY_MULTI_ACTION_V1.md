# V2-B / V2-C / V2-D: Nonlinear ML x Recency x Context x Multi-Action Value Experiment Report

**Final Verdict**: `STOP_CURRENT_MULTI_ACTION_ML`

## 1. Executive Summary & Gates

| Gate | Description | Status | Evidence |
|---|---|---|---|
| **G1** | `Q_POLICY_EDGE_SURVIVES` | `FAILED` | WF1=-0.0253, WF2=0.0228, WF3=0.0973 |
| **G2** | `Q_POLICY_BEATS_MARKET` | `FAILED` | ΔMarket: WF1=-0.0357, WF2=-0.0103, WF3=+0.0535 |
| **G3** | `Q_POLICY_BEATS_FIXED_REASSESS` | `FAILED` | ΔReassess: WF1=-0.0595, WF2=-0.0216, WF3=-0.0068 |
| **G4** | `ML_ACTION_SELECTION_ADDS_MATERIAL_VALUE` | `FAILED` | Mean Δ=-0.02931 R/sig, 95% CI lower > 0 count: 0/3 |

### Benchmark Comparison (EV_cw per signal)

| Walk-Forward | E1.1 Market | Fixed RR=3 Reassess | Learned Q Policy | Δ vs Reassess | 95% Bootstrap CI |
|---|---:|---:|---:|---:|---:|
| **WF1** | 0.0104 | 0.0342 | -0.0253 | -0.0595 | [-0.0847, -0.0326] |
| **WF2** | 0.0331 | 0.0444 | 0.0228 | -0.0216 | [-0.0482, +0.0044] |
| **WF3** | 0.0438 | 0.1041 | 0.0973 | -0.0068 | [-0.0344, +0.0202] |

## 2. Selected Architectures & Action Mix

```
model_type  half_life feature_block  selected_score  wf
   HistGBR        120            S1       -0.010842 WF1
   HistGBR        240            S1        0.020379 WF2
   HistGBR         60            S2        0.046642 WF3
```

Action distribution across test splits:
```
action  LIMIT_RR3    MARKET  REASSESS_RR3      SKIP
wf                                                 
WF1      0.177532  0.260759      0.243987  0.317722
WF2      0.235335  0.193537      0.217071  0.354057
WF3      0.233045  0.288896      0.206117  0.271941
```

## 3. Policy Collapse Check
- Max share WF1: 31.8%
- Max share WF2: 35.4%
- Max share WF3: 28.9%
- Status: `HEALTHY_STATE_DEPENDENT_SWITCHING`

## 4. Verification and Governance
- P1 read: `False` (all sample times <= 2026-09-04 14:55:00)
- Tests passed: 14/14 via executable assertions (see purge, preprocessing-scope, and prediction-reproducibility audits)
