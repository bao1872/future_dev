# V2-B / V2-C / V2-D: Nonlinear ML x Recency x Context x Multi-Action Value Experiment Report

**Final Verdict**: `STOP_CURRENT_MULTI_ACTION_ML`

## 1. Executive Summary & Gates

| Gate | Description | Status | Evidence |
|---|---|---|---|
| **G1** | `Q_POLICY_EDGE_SURVIVES` | `FAILED` | WF1=-0.0109, WF2=0.0132, WF3=0.0909 |
| **G2** | `Q_POLICY_BEATS_MARKET` | `FAILED` | ΔMarket: WF1=-0.0213, WF2=-0.0199, WF3=+0.0471 |
| **G3** | `Q_POLICY_BEATS_FIXED_REASSESS` | `FAILED` | ΔReassess: WF1=-0.0452, WF2=-0.0312, WF3=-0.0132 |
| **G4** | `ML_ACTION_SELECTION_ADDS_MATERIAL_VALUE` | `FAILED` | Mean Δ=-0.02986 R/sig, 95% CI lower > 0 count: 0/3 |

### Benchmark Comparison (EV_cw per signal)

| Walk-Forward | E1.1 Market | Fixed RR=3 Reassess | Learned Q Policy | Δ vs Reassess | 95% Bootstrap CI |
|---|---:|---:|---:|---:|---:|
| **WF1** | 0.0104 | 0.0342 | -0.0109 | -0.0452 | [-0.0725, -0.0179] |
| **WF2** | 0.0331 | 0.0444 | 0.0132 | -0.0312 | [-0.0590, -0.0038] |
| **WF3** | 0.0438 | 0.1041 | 0.0909 | -0.0132 | [-0.0392, +0.0108] |

## 2. Selected Architectures & Action Mix

```
model_type  half_life feature_block  selected_score  wf
   HistGBR         60            S1       -0.018807 WF1
   HistGBR        240            S1        0.023182 WF2
   HistGBR         60            S2        0.050127 WF3
```

Action distribution across test splits:
```
action  LIMIT_RR3    MARKET  REASSESS_RR3      SKIP
wf                                                 
WF1      0.100633  0.330696      0.203797  0.364873
WF2      0.145065  0.247629      0.224447  0.382859
WF3      0.150931  0.302859      0.273604  0.272606
```

## 3. Policy Collapse Check
- Max share WF1: 36.5%
- Max share WF2: 38.3%
- Max share WF3: 30.3%
- Status: `HEALTHY_STATE_DEPENDENT_SWITCHING`

## 4. Verification and Governance
- P1 read: `False` (all sample times <= 2026-09-04 14:55:00)
- Tests passed: 14/14 via executable assertions (see purge, preprocessing-scope, and prediction-reproducibility audits)
