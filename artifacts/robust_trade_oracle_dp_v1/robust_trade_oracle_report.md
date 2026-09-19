# Robust 5m Trade Oracle DP v1.1 (Correctness Hardening)

> **Status**: `PROVISIONAL_PENDING_USER_AUDIT`. No model yet (R1/R1.1 = label
> stability & correctness only). Utility = gross price-point PnL, cost = 0.
> Built on R1 after user review found 4 substantive label-semantics issues.

**Task**: `FUTURE-ORACLE-R1.1-CORRECTNESS-HARDENING`
**Horizons**: [6, 12, 24] (30/60/120 min)
**Environment-independent**: only 5m OHLC + discontinuity + causal ATR5 (units).
No DTP / SR / Liquidity / HTF / indicators.

## 0. Frozen semantics

- Decision at 5m close `t`; entry fill = open of next valid bar `e=t+1`.
- Exit fill = open of bar `e+h`; holding `h` in 1..H.
- **At most one round-trip trade per Oracle horizon** (exit does not re-enter
  in the same horizon). Oracle answers "is THIS bar a good trade opportunity?".
- Single position, fixed 1 unit, no add, no hedge.
- **Whole horizon banned from crossing a discontinuity** (Fix A).
- `V_flat[t][h] = max(Q_L, Q_S, Q_W(t,h))`; `Q_W=0` if `disc[t+1]=1`;
  `Q_W=V_flat[t+1][h-1]` otherwise.
- Ties use numeric eps; only n_best==1 yields a unique action (else Tie).
  stable_action needs all 3 horizons = SAME UNIQUE.
- Shared compute: one load + one segment + one QL/QS precompute + one backward
  flat DP; H=6/12/24 read from tables.

## 1. Sample counts

| metric | value |
|---|---:|
| n_decisions | 505161 |
| runtime_sec | 65.87 |
| peak_rss_mb | 885.3 |
| n_symbols | 15 |

## 2. Stable action distribution

| class | pct |
|---|---:|
| Long | 10.679 |
| Short | 9.798 |
| Wait | 48.317 |
| Ambiguous | 31.206 |

Agreement: 3/3 = 73.594% ; 2/3+ = 99.224%
Tie rows = 69883; stable_trade_zero_edge_count = 0

## 3. By symbol (with exclusion accounting)

### symbol

| symbol | input_5m_rows | output_decision_rows | excluded_tail | excluded_disc_before_entry | excluded_no_valid_roundtrip | tie_rows | stable_Long_pct | stable_Short_pct | stable_Wait_pct | stable_Ambiguous_pct | agreement_3of3_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AG | 44451 | 44449 | 2 | 0 | 0 | 2684 | 13.01 | 10.194 | 50.897 | 25.899 | 76.179 |
| AL | 37323 | 37321 | 2 | 0 | 0 | 8672 | 9.678 | 8.606 | 44.334 | 37.381 | 70.622 |
| AU | 44451 | 44449 | 2 | 0 | 0 | 1057 | 13.373 | 10.835 | 52.35 | 23.443 | 77.347 |
| CF | 27819 | 27817 | 2 | 0 | 0 | 7427 | 8.577 | 8.243 | 43.502 | 39.677 | 69.45 |
| CU | 37323 | 37321 | 2 | 0 | 0 | 4015 | 11.286 | 9.885 | 49.503 | 29.327 | 74.323 |
| I | 27819 | 27817 | 2 | 0 | 0 | 8256 | 8.056 | 7.884 | 42.064 | 41.996 | 68.541 |
| M | 27819 | 27817 | 2 | 0 | 0 | 6028 | 9.325 | 8.865 | 45.544 | 36.266 | 71.197 |
| MA | 27819 | 27817 | 2 | 0 | 0 | 4939 | 9.595 | 9.527 | 46.863 | 34.015 | 72.204 |
| NI | 37323 | 37321 | 2 | 0 | 0 | 2281 | 10.798 | 11.235 | 51.74 | 26.227 | 75.729 |
| P | 27819 | 27817 | 2 | 0 | 0 | 2778 | 10.957 | 10.192 | 50.566 | 28.285 | 75.152 |
| RB | 27819 | 27817 | 2 | 0 | 0 | 6874 | 8.261 | 9.271 | 44.577 | 37.89 | 70.597 |
| RU | 27819 | 27817 | 2 | 0 | 0 | 4252 | 9.904 | 9.757 | 48.043 | 32.297 | 72.912 |
| SC | 44449 | 44443 | 2 | 2 | 2 | 4903 | 11.12 | 10.335 | 49.101 | 29.444 | 74.642 |
| SN | 37323 | 37321 | 2 | 0 | 0 | 1045 | 12.583 | 11.117 | 52.52 | 23.78 | 77.313 |
| TA | 27819 | 27817 | 2 | 0 | 0 | 4672 | 9.832 | 9.278 | 47.083 | 33.807 | 72.053 |


## 4. By time block (calendar month; "TB block" mapped to month)

### time_block

| time_block | n | stable_Long_pct | stable_Short_pct | stable_Wait_pct | stable_Ambiguous_pct |
|---|---|---|---|---|---|
| 2025-01 | 22044 | 10.733 | 9.431 | 48.512 | 31.324 |
| 2025-02 | 22481 | 10.907 | 9.697 | 48.792 | 30.604 |
| 2025-03 | 26397 | 10.744 | 9.694 | 48.255 | 31.307 |
| 2025-04 | 25378 | 10.162 | 10.229 | 47.99 | 31.618 |
| 2025-05 | 23301 | 9.665 | 10.3 | 47.371 | 32.664 |
| 2025-06 | 24995 | 10.842 | 8.726 | 47.061 | 33.371 |
| 2025-07 | 28911 | 10.515 | 9.159 | 47.778 | 32.548 |
| 2025-08 | 26542 | 9.509 | 9.314 | 47.525 | 33.652 |
| 2025-09 | 27072 | 10.424 | 9.667 | 47.056 | 32.853 |
| 2025-10 | 21224 | 10.568 | 10.055 | 47.451 | 31.926 |
| 2025-11 | 25285 | 10.524 | 9.713 | 47.17 | 32.592 |
| 2025-12 | 28329 | 11.299 | 9.224 | 48.35 | 31.127 |
| 2026-01 | 25140 | 12.148 | 8.886 | 49.276 | 29.69 |
| 2026-02 | 17016 | 9.955 | 9.949 | 48.449 | 31.647 |
| 2026-03 | 27503 | 11.646 | 10.21 | 49.664 | 28.481 |
| 2026-04 | 25378 | 11.301 | 9.753 | 49.803 | 29.143 |
| 2026-05 | 22626 | 10.413 | 11.279 | 48.754 | 29.554 |
| 2026-06 | 25670 | 8.551 | 11.987 | 49.497 | 29.965 |
| 2026-07 | 28911 | 10.425 | 10.276 | 48.784 | 30.514 |
| 2026-08 | 26397 | 12.645 | 8.361 | 48.464 | 30.53 |
| 2026-09 | 4561 | 11.774 | 11.883 | 49.857 | 26.485 |


## 5. Distribution quantiles (H=24, ATR-normalized primary)

- edge_ATR: {'n': 504426, 'p10': 0.11628, 'p25': 0.3125, 'p50': 0.67568, 'p75': 1.25, 'p90': 1.96807, 'mean': 0.92083}
- holding (trade actions): {'n': 103439, 'p10': 6.0, 'p25': 11.0, 'p50': 17.0, 'p75': 22.0, 'p90': 24.0, 'mean': 15.89543}
- MFE_ATR: {'n': 103402, 'p10': 2.42417, 'p25': 3.36618, 'p50': 5.0, 'p75': 7.67442, 'p90': 11.66667, 'mean': 6.36454}
- MAE_ATR: {'n': 103402, 'p10': 0.0, 'p25': 0.05153, 'p50': 0.20408, 'p75': 0.37037, 'p90': 0.58824, 'mean': 0.27105}

Raw (per-symbol scale, not cross-comparable): edge_raw {'n': 505161, 'p10': 0.29999, 'p25': 1.47998, 'p50': 7.0, 'p75': 33.0, 'p90': 160.0, 'mean': 77.95527},
MFE_raw {'n': 103439, 'p10': 5.10004, 'p25': 15.0, 'p50': 74.0, 'p75': 340.03174, 'p90': 1409.975, 'mean': 628.53004}, MAE_raw {'n': 103439, 'p10': 0.0, 'p25': 0.09998, 'p50': 2.0, 'p75': 10.0, 'p90': 59.9375, 'mean': 28.53969}.

## 6. Stable-class edge / holding (ATR-normalized)

| class | median_edge_ATR | median_holding |
|---|---:|---:|
| Long | 0.41463 | 17.0 |
| Short | 0.41176 | 16.0 |
| Wait | 0.73027 | None |

## 7. Cross-horizon transition (Long/Short/Wait/Tie)

- 6->12: [[75052, 0, 25013, 3220], [0, 69708, 24699, 3277], [0, 0, 244080, 0], [0, 0, 23674, 36438]]
- 12->24: [[53945, 0, 19411, 1696], [0, 49494, 18636, 1578], [0, 0, 317466, 0], [0, 0, 16106, 26829]]
- 6->24: [[53945, 0, 46293, 3047], [0, 49494, 45384, 2806], [0, 0, 244080, 0], [0, 0, 35862, 24250]]

## 8. Exclusion accounting (input = output + excluded, mutually exclusive)

### symbol

| input_5m_rows | output_decision_rows | excluded_tail | excluded_discontinuity_before_entry | excluded_no_valid_roundtrip | tie_rows |
|---|---|---|---|---|---|
| 505195 | 505161 | 30 | 2 | 2 | 69883 |


## 9. Human oracle samples (5)

```json
[
  [
    "stable_Long",
    {
      "symbol": "SN",
      "decision_bar_start_time": "2026-03-23 14:50:00",
      "decision_time": "2026-03-23 14:55:00",
      "decision_bar_index": 26863,
      "segment": 0,
      "agreement": 1.0,
      "stable_action": "Long",
      "atr5_t": 2436.0,
      "H6": {
        "QL": 16929.96875,
        "QS": -12280.0,
        "QW": 4649.96875,
        "QL_ATR": 6.949905069786535,
        "QS_ATR": -5.041050903119869,
        "QW_ATR": 1.9088541666666667,
        "action": "Long",
        "edge": 12280.0,
        "edge_ATR": 5.041050903119869,
        "exit_bars": 26865.0,
        "holding_bars": 1.0,
        "realized_move": 16929.96875,
        "MFE": 16929.96875,
        "MAE": 120.0,
        "label_available_time": "2026-03-23 21:30:00",
        "terminal_reason": "HORIZON_END"
      },
      "H12": {
        "QL": 16929.96875,
        "QS": -12280.0,
        "QW": 4649.96875,
        "QL_ATR": 6.949905069786535,
        "QS_ATR": -5.041050903119869,
        "QW_ATR": 1.9088541666666667,
        "action": "Long",
        "edge": 12280.0,
        "edge_ATR": 5.041050903119869,
        "exit_bars": 26865.0,
        "holding_bars": 1.0,
        "realized_move": 16929.96875,
        "MFE": 16929.96875,
        "MAE": 120.0,
        "label_available_time": "2026-03-23 22:00:00",
        "terminal_reason": "HORIZON_END"
      },
      "H24": {
        "QL": 26709.96875,
        "QS": -12280.0,
        "QW": 14429.96875,
        "QL_ATR": 10.964683394909688,
        "QS_ATR": -5.041050903119869,
        "QW_ATR": 5.923632491789819,
        "action": "Long",
        "edge": 12280.0,
        "edge_ATR": 5.041050903119869,
        "exit_bars": 26882.0,
        "holding_bars": 18.0,
        "realized_move": 26709.96875,
        "MFE": 27120.0,
        "MAE": 120.0,
        "label_available_time": "2026-03-23 23:00:00",
        "terminal_reason": "HORIZON_END"
      }
    }
  ],
  [
    "stable_Short",
    {
      "symbol": "SN",
      "decision_bar_start_time": "2026-01-15 14:55:00",
      "decision_time": "2026-01-15 15:00:00",
      "decision_bar_index": 23099,
      "segment": 0,
      "agreement": 1.0,
      "stable_action": "Short",
      "atr5_t": 2551.9875,
      "H6": {
        "QL": -15569.96875,
        "QS": 22700.0,
        "QW": 7130.03125,
        "QL_ATR": -6.101114817372734,
        "QS_ATR": 8.895027894925033,
        "QW_ATR": 2.7939130775522996,
        "action": "Short",
        "edge": 15569.96875,
        "edge_ATR": 6.101114817372734,
        "exit_bars": 23104.0,
        "holding_bars": 4.0,
        "realized_move": 22700.0,
        "MFE": 23440.0,
        "MAE": 3020.0,
        "label_available_time": "2026-01-15 21:35:00",
        "terminal_reason": "HORIZON_END"
      },
      "H12": {
        "QL": -15569.96875,
        "QS": 22700.0,
        "QW": 7130.03125,
        "QL_ATR": -6.101114817372734,
        "QS_ATR": 8.895027894925033,
        "QW_ATR": 2.7939130775522996,
        "action": "Short",
        "edge": 15569.96875,
        "edge_ATR": 6.101114817372734,
        "exit_bars": 23104.0,
        "holding_bars": 4.0,
        "realized_move": 22700.0,
        "MFE": 23440.0,
        "MAE": 3020.0,
        "label_available_time": "2026-01-15 22:05:00",
        "terminal_reason": "HORIZON_END"
      },
      "H24": {
        "QL": -13310.03125,
        "QS": 22700.0,
        "QW": 9389.96875,
        "QL_ATR": -5.2155550330869564,
        "QS_ATR": 8.895027894925033,
        "QW_ATR": 3.679472861838077,
        "action": "Short",
        "edge": 13310.03125,
        "edge_ATR": 5.2155550330869564,
        "exit_bars": 23104.0,
        "holding_bars": 4.0,
        "realized_move": 22700.0,
        "MFE": 23440.0,
        "MAE": 3020.0,
        "label_available_time": "2026-01-15 23:05:00",
        "terminal_reason": "HORIZON_END"
      }
    }
  ],
  [
    "stable_Wait",
    {
      "symbol": "SN",
      "decision_bar_start_time": "2026-01-30 00:25:00",
      "decision_time": "2026-01-30 00:30:00",
      "decision_bar_index": 24071,
      "segment": 0,
      "agreement": 1.0,
      "stable_action": "Wait",
      "atr5_t": 2778.00625,
      "H6": {
        "QL": 3210.03125,
        "QS": 1500.0,
        "QW": 4710.03125,
        "QL_ATR": 1.155516208791827,
        "QS_ATR": 0.5399555886528333,
        "QW_ATR": 1.6954717974446603,
        "action": "Wait",
        "edge": 1500.0,
        "edge_ATR": 0.5399555886528333,
        "exit_bars": NaN,
        "holding_bars": NaN,
        "realized_move": NaN,
        "MFE": NaN,
        "MAE": NaN,
        "label_available_time": "2026-01-30 09:05:00",
        "terminal_reason": "HORIZON_END"
      },
      "H12": {
        "QL": 5730.03125,
        "QS": 1760.0,
        "QW": 7490.03125,
        "QL_ATR": 2.062641597728587,
        "QS_ATR": 0.633547890685991,
        "QW_ATR": 2.696189488414578,
        "action": "Wait",
        "edge": 1760.0,
        "edge_ATR": 0.633547890685991,
        "exit_bars": NaN,
        "holding_bars": NaN,
        "realized_move": NaN,
        "MFE": NaN,
        "MAE": NaN,
        "label_available_time": "2026-01-30 09:35:00",
        "terminal_reason": "HORIZON_END"
      },
      "H24": {
        "QL": 5730.03125,
        "QS": 30660.0,
        "QW": 36390.03125,
        "QL_ATR": 2.062641597728587,
        "QS_ATR": 11.036692232063913,
        "QW_ATR": 13.0993338297925,
        "action": "Wait",
        "edge": 5730.03125,
        "edge_ATR": 2.062641597728587,
        "exit_bars": NaN,
        "holding_bars": NaN,
        "realized_move": NaN,
        "MFE": NaN,
        "MAE": NaN,
        "label_available_time": "2026-01-30 10:50:00",
        "terminal_reason": "HORIZON_END"
      }
    }
  ],
  [
    "tie",
    {
      "symbol": "SC",
      "decision_bar_start_time": "2025-02-17 21:30:00",
      "decision_time": "2025-02-17 21:35:00",
      "decision_bar_index": 2871,
      "segment": 0,
      "agreement": 1.0,
      "stable_action": "Ambiguous",
      "atr5_t": 1.14000244140625,
      "H6": {
        "QL": 1.10003662109375,
        "QS": 0.0,
        "QW": 1.10003662109375,
        "QL_ATR": 0.9649423379627151,
        "QS_ATR": 0.0,
        "QW_ATR": 0.9649423379627151,
        "action": "Tie",
        "edge": 0.0,
        "edge_ATR": 0.0,
        "exit_bars": NaN,
        "holding_bars": NaN,
        "realized_move": NaN,
        "MFE": NaN,
        "MAE": NaN,
        "label_available_time": "2025-02-17 22:10:00",
        "terminal_reason": "HORIZON_END"
      },
      "H12": {
        "QL": 1.10003662109375,
        "QS": 0.0,
        "QW": 1.10003662109375,
        "QL_ATR": 0.9649423379627151,
        "QS_ATR": 0.0,
        "QW_ATR": 0.9649423379627151,
        "action": "Tie",
        "edge": 0.0,
        "edge_ATR": 0.0,
        "exit_bars": NaN,
        "holding_bars": NaN,
        "realized_move": NaN,
        "MFE": NaN,
        "MAE": NaN,
        "label_available_time": "2025-02-17 22:40:00",
        "terminal_reason": "HORIZON_END"
      },
      "H24": {
        "QL": 1.60003662109375,
        "QS": 0.0,
        "QW": 1.60003662109375,
        "QL_ATR": 1.4035378899013804,
        "QS_ATR": 0.0,
        "QW_ATR": 1.4035378899013804,
        "action": "Tie",
        "edge": 0.0,
        "edge_ATR": 0.0,
        "exit_bars": NaN,
        "holding_bars": NaN,
        "realized_move": NaN,
        "MFE": NaN,
        "MAE": NaN,
        "label_available_time": "2025-02-17 23:40:00",
        "terminal_reason": "HORIZON_END"
      }
    }
  ],
  [
    "discontinuity_near",
    {
      "symbol": "SC",
      "decision_bar_start_time": "2026-03-04 00:25:00",
      "decision_time": "2026-03-04 00:30:00",
      "decision_bar_index": 30482,
      "segment": 0,
      "agreement": 1.0,
      "stable_action": "Ambiguous",
      "atr5_t": 0.0,
      "H6": {
        "QL": 0.0,
        "QS": 0.0,
        "QW": 0.0,
        "QL_ATR": NaN,
        "QS_ATR": NaN,
        "QW_ATR": NaN,
        "action": "Tie",
        "edge": 0.0,
        "edge_ATR": NaN,
        "exit_bars": NaN,
        "holding_bars": NaN,
        "realized_move": NaN,
        "MFE": NaN,
        "MAE": NaN,
        "label_available_time": "2026-03-04 01:05:00",
        "terminal_reason": "HORIZON_END"
      },
      "H12": {
        "QL": 0.0,
        "QS": 0.0,
        "QW": 0.0,
        "QL_ATR": NaN,
        "QS_ATR": NaN,
        "QW_ATR": NaN,
        "action": "Tie",
        "edge": 0.0,
        "edge_ATR": NaN,
        "exit_bars": NaN,
        "holding_bars": NaN,
        "realized_move": NaN,
        "MFE": NaN,
        "MAE": NaN,
        "label_available_time": "2026-03-04 01:35:00",
        "terminal_reason": "HORIZON_END"
      },
      "H24": {
        "QL": 0.0,
        "QS": 0.0,
        "QW": 0.0,
        "QL_ATR": NaN,
        "QS_ATR": NaN,
        "QW_ATR": NaN,
        "action": "Tie",
        "edge": 0.0,
        "edge_ATR": NaN,
        "exit_bars": NaN,
        "holding_bars": NaN,
        "realized_move": NaN,
        "MFE": NaN,
        "MAE": NaN,
        "label_available_time": "2026-03-04 01:35:00",
        "terminal_reason": "DISCONTINUITY"
      }
    }
  ]
]
```

## 10. Cost metadata

```json
{
  "canonical_table_found": false,
  "scanned": [
    "tick_size",
    "contract_multiplier",
    "commission",
    "exchange_fee",
    "broker_fee",
    "slippage"
  ],
  "NET_PNL": "UNAVAILABLE_COST_METADATA",
  "rule": "\u7981\u6b62\u51ed\u8bb0\u5fc6\u586b\u5199\u624b\u7eed\u8d39/\u6ed1\u70b9\uff1b\u53ea\u62a5 GROSS price-point PnL (ATR-normalized \u4f9b\u8de8\u54c1\u79cd\u6bd4\u8f83)",
  "break_even_note": "utility v1 = GrossPnL - Cost, Cost=0; net PnL requires per-symbol tick value + fee table which is not present in the project. Do NOT claim net profitable."
}
```

## 11. Next

User audits R1.1 fixes (A-H). Then the real Oracle robustness experiment:
scan `risk penalty x time penalty x friction hurdle` over the SAME precomputed
future paths and measure how many Long/Short/Wait labels stay stable. Only
after that is the Oracle a credible label for Environment -> Oracle mapping.
