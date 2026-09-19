# Robust 5m Trade Oracle DP v1.0

> **Status**: `PROVISIONAL_PENDING_USER_AUDIT`. No model yet (R1 = label
> stability only). Utility = gross price-point PnL, cost = 0.

**Task**: `FUTURE-ORACLE-R1-ROBUST-5M-DP`
**Horizons**: [6, 12, 24] (30/60/120 min)
**Environment-independent**: uses only 5m OHLC + discontinuity flags.
No DTP / SR / Liquidity / HTF / indicators.

## 0. Contract (frozen)

- Decision at 5m close `t`; entry fill = open of next valid bar `e=t+1`.
- Exit fill = open of bar `e+h` (next-open convention); holding `h` in 1..H.
- Single position, fixed 1 unit, no add, no same-symbol hedging.
- No crossing discontinuity; horizon caps holding; data end caps path.
- `V_flat[t][h] = max(Q_L, Q_S, V_flat[t+1][h-1])`; `Q_W = V_flat[t+1][h-1]`.
- Shared compute: one load + one segment build + one QL/QS precompute + one
  backward flat DP. H=6/12/24 read from tables (never re-run data).

## 1. Sample counts

| metric | value |
|---|---:|
| n_decisions | 505163 |
| runtime_sec | 41.82 |
| peak_rss_mb | 468.7 |
| n_symbols | 15 |

## 2. Stable action distribution

| class | pct |
|---|---:|
| Long | 13.749 |
| Short | 12.643 |
| Wait | 48.317 |
| Ambiguous | 25.291 |

Agreement: 3/3 = 74.709% ; 2/3+ = 99.996%

## 3. By symbol

### symbol

| symbol | n | stable_Long_pct | stable_Short_pct | stable_Wait_pct | stable_Ambiguous_pct | agreement_3of3_pct |
|---|---|---|---|---|---|---|
| AG | 44449 | 14.536 | 11.235 | 50.897 | 23.332 | 76.668 |
| AL | 37321 | 14.97 | 13.156 | 44.334 | 27.539 | 72.461 |
| AU | 44449 | 13.926 | 11.305 | 52.35 | 22.419 | 77.581 |
| CF | 27817 | 14.283 | 13.898 | 43.502 | 28.317 | 71.683 |
| CU | 37321 | 13.847 | 11.969 | 49.503 | 24.68 | 75.32 |
| I | 27817 | 14.66 | 14.096 | 42.064 | 29.18 | 70.82 |
| M | 27817 | 14.049 | 13.37 | 45.544 | 27.037 | 72.963 |
| MA | 27817 | 13.334 | 13.467 | 46.863 | 26.336 | 73.664 |
| NI | 37321 | 11.969 | 12.583 | 51.74 | 23.708 | 76.292 |
| P | 27817 | 13.247 | 12.147 | 50.566 | 24.039 | 75.961 |
| RB | 27817 | 13.308 | 14.678 | 44.577 | 27.436 | 72.564 |
| RU | 27817 | 13.132 | 13.021 | 48.043 | 25.804 | 74.196 |
| SC | 44445 | 13.828 | 12.51 | 49.103 | 24.558 | 75.442 |
| SN | 37321 | 13.231 | 11.677 | 52.52 | 22.572 | 77.428 |
| TA | 27817 | 13.628 | 12.654 | 47.083 | 26.635 | 73.365 |


## 4. By time block (calendar month; "TB block" mapped to month)

### time_block

| time_block | n | stable_Long_pct | stable_Short_pct | stable_Wait_pct | stable_Ambiguous_pct |
|---|---|---|---|---|---|
| 2025-01 | 22044 | 13.954 | 12.412 | 48.512 | 25.122 |
| 2025-02 | 22488 | 14.412 | 12.651 | 48.799 | 24.137 |
| 2025-03 | 26397 | 13.808 | 12.926 | 48.24 | 25.026 |
| 2025-04 | 25371 | 13.149 | 13.019 | 48.0 | 25.833 |
| 2025-05 | 23301 | 12.605 | 13.729 | 47.371 | 26.295 |
| 2025-06 | 25002 | 14.727 | 12.067 | 47.06 | 26.146 |
| 2025-07 | 28911 | 14.033 | 12.231 | 47.778 | 25.959 |
| 2025-08 | 26535 | 13.217 | 13.021 | 47.526 | 26.237 |
| 2025-09 | 27072 | 13.963 | 13.494 | 47.056 | 25.488 |
| 2025-10 | 21231 | 13.899 | 13.132 | 47.454 | 25.515 |
| 2025-11 | 25278 | 14.265 | 12.976 | 47.167 | 25.591 |
| 2025-12 | 28329 | 14.293 | 12.309 | 48.35 | 25.049 |
| 2026-01 | 25140 | 14.785 | 10.967 | 49.276 | 24.972 |
| 2026-02 | 17016 | 12.7 | 12.853 | 48.449 | 25.999 |
| 2026-03 | 27512 | 14.463 | 11.933 | 49.658 | 23.946 |
| 2026-04 | 25371 | 13.961 | 11.722 | 49.813 | 24.504 |
| 2026-05 | 22626 | 12.822 | 13.759 | 48.754 | 24.666 |
| 2026-06 | 25677 | 10.461 | 14.764 | 49.496 | 25.279 |
| 2026-07 | 28911 | 12.898 | 12.86 | 48.791 | 25.451 |
| 2026-08 | 26397 | 15.945 | 10.236 | 48.464 | 25.355 |
| 2026-09 | 4554 | 14.603 | 13.834 | 49.824 | 21.739 |


## 5. Distribution quantiles (H=24)

- edge: {'n': 505161, 'p10': 0.3, 'p25': 1.48, 'p50': 7.0, 'p75': 33.0, 'p90': 160.0, 'mean': 77.9558}
- holding (trade actions): {'n': 133322, 'p10': 6.0, 'p25': 11.0, 'p50': 17.0, 'p75': 22.0, 'p90': 24.0, 'mean': 15.8823}
- MFE (trade actions): {'n': 133322, 'p10': 5.0, 'p25': 14.0, 'p50': 60.0, 'p75': 240.0, 'p90': 1100.0, 'mean': 520.7012}
- MAE (trade actions): {'n': 133322, 'p10': 0.0, 'p25': 0.2, 'p50': 2.0, 'p75': 10.0, 'p90': 45.0, 'mean': 24.6336}

## 6. Stable-class edge / holding

| class | median_edge | min_edge | median_holding |
|---|---:|---:|---:|
| Long | 2.5 | 0.0 | 17.0 |
| Short | 2.35 | 0.0 | 17.0 |
| Wait | 10.0 | 0.001 | None |

## 7. Human oracle samples (5)

```json
[
  [
    "stable_Long_high_edge",
    {
      "symbol": "SN",
      "decision_time": "2026-03-23 14:50:00",
      "decision_bar_index": 26863,
      "segment": 0,
      "agreement": 1.0,
      "stable_action": "Long",
      "H6": {
        "QL": 16929.96875,
        "QS": -12280.0,
        "QW": 4649.96875,
        "action": "Long",
        "edge": 12280.0,
        "exit_bars": 26865.0,
        "realized_move": 16929.96875,
        "MFE": 16929.96875,
        "MAE": 120.0
      },
      "H12": {
        "QL": 16929.96875,
        "QS": -12280.0,
        "QW": 4649.96875,
        "action": "Long",
        "edge": 12280.0,
        "exit_bars": 26865.0,
        "realized_move": 16929.96875,
        "MFE": 16929.96875,
        "MAE": 120.0
      },
      "H24": {
        "QL": 26709.96875,
        "QS": -12280.0,
        "QW": 14429.96875,
        "action": "Long",
        "edge": 12280.0,
        "exit_bars": 26882.0,
        "realized_move": 26709.96875,
        "MFE": 27120.0,
        "MAE": 120.0
      }
    }
  ],
  [
    "stable_Short_high_edge",
    {
      "symbol": "SN",
      "decision_time": "2026-01-15 14:55:00",
      "decision_bar_index": 23099,
      "segment": 0,
      "agreement": 1.0,
      "stable_action": "Short",
      "H6": {
        "QL": -15569.96875,
        "QS": 22700.0,
        "QW": 7130.03125,
        "action": "Short",
        "edge": 15569.96875,
        "exit_bars": 23104.0,
        "realized_move": 22700.0,
        "MFE": 23440.0,
        "MAE": 3020.0
      },
      "H12": {
        "QL": -15569.96875,
        "QS": 22700.0,
        "QW": 7130.03125,
        "action": "Short",
        "edge": 15569.96875,
        "exit_bars": 23104.0,
        "realized_move": 22700.0,
        "MFE": 23440.0,
        "MAE": 3020.0
      },
      "H24": {
        "QL": -13310.03125,
        "QS": 22700.0,
        "QW": 9389.96875,
        "action": "Short",
        "edge": 13310.03125,
        "exit_bars": 23104.0,
        "realized_move": 22700.0,
        "MFE": 23440.0,
        "MAE": 3020.0
      }
    }
  ],
  [
    "stable_Wait",
    {
      "symbol": "SN",
      "decision_time": "2026-01-30 00:25:00",
      "decision_bar_index": 24071,
      "segment": 0,
      "agreement": 1.0,
      "stable_action": "Wait",
      "H6": {
        "QL": 3210.03125,
        "QS": 1500.0,
        "QW": 4710.03125,
        "action": "Wait",
        "edge": 1500.0,
        "exit_bars": NaN,
        "realized_move": NaN,
        "MFE": NaN,
        "MAE": NaN
      },
      "H12": {
        "QL": 5730.03125,
        "QS": 1760.0,
        "QW": 7490.03125,
        "action": "Wait",
        "edge": 1760.0,
        "exit_bars": NaN,
        "realized_move": NaN,
        "MFE": NaN,
        "MAE": NaN
      },
      "H24": {
        "QL": 5730.03125,
        "QS": 30660.0,
        "QW": 36390.03125,
        "action": "Wait",
        "edge": 5730.03125,
        "exit_bars": NaN,
        "realized_move": NaN,
        "MFE": NaN,
        "MAE": NaN
      }
    }
  ],
  [
    "ambiguous",
    {
      "symbol": "RB",
      "decision_time": "2026-07-07 11:15:00",
      "decision_bar_index": 24831,
      "segment": 0,
      "agreement": 0.6667,
      "stable_action": "Ambiguous",
      "H6": {
        "QL": -2.0,
        "QS": 7.0,
        "QW": 5.0,
        "action": "Short",
        "edge": 2.0,
        "exit_bars": 24837.0,
        "realized_move": 7.0,
        "MFE": 7.0,
        "MAE": 0.0
      },
      "H12": {
        "QL": -2.0,
        "QS": 7.0,
        "QW": 5.0,
        "action": "Short",
        "edge": 2.0,
        "exit_bars": 24837.0,
        "realized_move": 7.0,
        "MFE": 7.0,
        "MAE": 0.0
      },
      "H24": {
        "QL": 3.0,
        "QS": 7.0,
        "QW": 10.0,
        "action": "Wait",
        "edge": 3.0,
        "exit_bars": NaN,
        "realized_move": NaN,
        "MFE": NaN,
        "MAE": NaN
      }
    }
  ],
  [
    "max_edge_any",
    {
      "symbol": "SN",
      "decision_time": "2026-01-15 14:55:00",
      "decision_bar_index": 23099,
      "segment": 0,
      "agreement": 1.0,
      "stable_action": "Short",
      "H6": {
        "QL": -15569.96875,
        "QS": 22700.0,
        "QW": 7130.03125,
        "action": "Short",
        "edge": 15569.96875,
        "exit_bars": 23104.0,
        "realized_move": 22700.0,
        "MFE": 23440.0,
        "MAE": 3020.0
      },
      "H12": {
        "QL": -15569.96875,
        "QS": 22700.0,
        "QW": 7130.03125,
        "action": "Short",
        "edge": 15569.96875,
        "exit_bars": 23104.0,
        "realized_move": 22700.0,
        "MFE": 23440.0,
        "MAE": 3020.0
      },
      "H24": {
        "QL": -13310.03125,
        "QS": 22700.0,
        "QW": 9389.96875,
        "action": "Short",
        "edge": 13310.03125,
        "exit_bars": 23104.0,
        "realized_move": 22700.0,
        "MFE": 23440.0,
        "MAE": 3020.0
      }
    }
  ]
]
```

## 8. Cost metadata

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
  "rule": "\u7981\u6b62\u51ed\u8bb0\u5fc6\u586b\u5199\u624b\u7eed\u8d39/\u6ed1\u70b9\uff1b\u53ea\u62a5 GROSS price-point PnL",
  "break_even_note": "utility v1 = GrossPnL - Cost, Cost=0; net PnL requires per-symbol tick value + fee table which is not present in the project. Do NOT claim net profitable."
}
```

## 9. Next

User audits DP Bellman, Wait option value, next-open execution, discontinuity
boundary, horizon off-by-one, MFE/MAE interval, H=6/12/24 shared compute, and
row-level artifact completeness. Then decides Oracle R2 (risk/time penalty,
longer horizon, 2/3 consensus) or proceeds to Environment -> Oracle mapping.
