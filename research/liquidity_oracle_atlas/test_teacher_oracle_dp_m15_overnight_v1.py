"""Phase-0.5 synthetic proofs for the overnight Teacher Oracle.

Contract:
  2. Synthetic hard-boundary tests -- position cannot cross segment; terminal
     trade at a hard boundary => HARD_BOUNDARY + training_eligible=False;
     DATA_END => training_eligible=False; OPTIMAL_FLAT / REVERSAL => eligible.
  3. Candidate->teacher mapping may NEVER cross a hard segment boundary.
"""

import numpy as np
import pandas as pd
import pytest

from research.liquidity_oracle_atlas.build_teacher_oracle_dp_m15_overnight_v1 import (
    DATA_END,
    HARD_BOUNDARY,
    OPTIMAL_FLAT,
    REVERSAL,
    _append_unit_trades,
    _dp_from_proximity,
    check_oracle_invariants,
    map_candidates_to_trades,
)
from research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1 import (
    KernelCounters,
)


# --------------------------------------------------------------------------- #
# Unit-level per-trade exit_reason + eligibility (deterministic path)           #
# --------------------------------------------------------------------------- #
def test_per_trade_exit_reason_and_eligibility():
    n = 200
    opens = np.arange(100.0, 100.0 + n, dtype=float)
    highs = opens + 1.0
    lows = opens - 1.0
    times = pd.date_range("2024-01-02 09:00", periods=n, freq="15min")
    td_arr = np.array(["2024-01-02"] * n)
    prox_bits = np.zeros(n, dtype=np.int64)
    prox_ep = np.full(n, 0, dtype=np.int64)
    cost = np.zeros(n, dtype=float)

    # end of the unit; boundary close happens at t == end-1
    end = 12
    terminal_reason = HARD_BOUNDARY  # this unit ends at a hard boundary
    eligible = False

    # crafted optimal (t, p, a, q) path:
    #  t0  : flat -> long  (new entry)                 start Trade A
    #  t5  : long -> flat  (voluntary)                 close A = OPTIMAL_FLAT
    #  t6  : flat -> long  (new entry)                 start Trade B
    #  t10 : long -> short (reversal/new entry)         close B = REVERSAL, start C
    #  t11 : short -> flat (forced at boundary t==end-1) close C = HARD_BOUNDARY
    path = [
        {"t": 0, "pb": 0, "pa": 1, "qb": 1, "qa": 1, "ne": True, "si": 0, "local": 0},
        {"t": 1, "pb": 1, "pa": 1, "qb": 1, "qa": 1, "ne": False, "si": 0, "local": 1},
        {"t": 2, "pb": 1, "pa": 1, "qb": 1, "qa": 1, "ne": False, "si": 0, "local": 2},
        {"t": 3, "pb": 1, "pa": 1, "qb": 1, "qa": 1, "ne": False, "si": 0, "local": 3},
        {"t": 4, "pb": 1, "pa": 1, "qb": 1, "qa": 1, "ne": False, "si": 0, "local": 4},
        {"t": 5, "pb": 1, "pa": 0, "qb": 1, "qa": 0, "ne": False, "si": 0, "local": 5},
        {"t": 6, "pb": 0, "pa": 1, "qb": 1, "qa": 1, "ne": True, "si": 0, "local": 6},
        {"t": 7, "pb": 1, "pa": 1, "qb": 1, "qa": 1, "ne": False, "si": 0, "local": 7},
        {"t": 8, "pb": 1, "pa": 1, "qb": 1, "qa": 1, "ne": False, "si": 0, "local": 8},
        {"t": 9, "pb": 1, "pa": 1, "qb": 1, "qa": 1, "ne": False, "si": 0, "local": 9},
        {"t": 10, "pb": 1, "pa": -1, "qb": 1, "qa": -1, "ne": True, "si": 0, "local": 10},
        {"t": 11, "pb": -1, "pa": 0, "qb": -1, "qa": 0, "ne": False, "si": 0, "local": 11},
    ]

    trades = []
    _append_unit_trades(
        trades, "SYN", td_arr, times, opens, highs, lows,
        prox_bits, prox_ep, cost, path, terminal_reason, eligible, end,
    )

    assert len(trades) == 3, trades
    a, b, c = trades
    assert a["terminal_reason"] == OPTIMAL_FLAT and a["training_eligible"] is True
    assert b["terminal_reason"] == REVERSAL and b["training_eligible"] is True
    assert c["terminal_reason"] == HARD_BOUNDARY and c["training_eligible"] is False


# --------------------------------------------------------------------------- #
# Full-Teacher on a synthetic 2-segment series: structural invariants           #
# --------------------------------------------------------------------------- #
def _synthetic_prox(n, boundary_at, day_changes):
    seg = np.zeros(n, dtype=np.int64)
    seg[boundary_at:] = 1
    td = np.zeros(n, dtype=np.int64)
    cur = 0
    for d in day_changes:
        td[d:] += 1
    opens = 100.0 + np.arange(n) * 1.0  # monotonic up -> DP wants long, holds
    df = pd.DataFrame(
        {
            "open": opens,
            "high": opens + 1.0,
            "low": opens - 1.0,
            "bar_start_time": pd.date_range("2024-01-02 09:00", periods=n, freq="15min"),
            "trading_day": ["2024-01-%02d" % (2 + int(t)) for t in td],
            "segment": seg,
            "dp_proximity_bits": np.ones(n, dtype=np.int64),
            "dp_proximity_any": np.ones(n, dtype=bool),
        }
    )
    return df, seg


def test_overnight_teacher_segment_invariants():
    n = 80
    boundary = 40
    # trading-day changes INSIDE segment 0 and segment 1 (must NOT force flat)
    prox, seg = _synthetic_prox(n, boundary, day_changes=[20, 40, 60])

    counters = KernelCounters()
    result = _dp_from_proximity("SYN", prox, counters, cost_points=None)
    inv = check_oracle_invariants(result)

    # hard invariants
    assert inv["cross_segment"] == 0, inv
    assert inv["new_entry_outside_proximity"] == 0, inv
    assert inv["illegal_reversal"] == 0, inv
    assert inv["max_new_entries_per_episode"] == 1, inv

    trades = result["trades"]
    assert trades, "expected at least one trade"

    # eligibility consistency
    for t in trades:
        if t["terminal_reason"] in (HARD_BOUNDARY, DATA_END):
            assert t["training_eligible"] is False, t
        else:
            assert t["terminal_reason"] in (OPTIMAL_FLAT, REVERSAL), t
            assert t["training_eligible"] is True, t

    # the segment-0 boundary trade must exist and be HARD_BOUNDARY + ineligible,
    # and it must have carried ACROSS a trading-day change (overnight works).
    hb = [t for t in trades if t["terminal_reason"] == HARD_BOUNDARY]
    assert hb, "expected a HARD_BOUNDARY trade at the segment boundary"
    for t in hb:
        ei, xi = int(t["entry_fill_index"]), int(t["exit_fill_index"])
        assert seg[ei] == seg[xi] == 0  # within segment 0
        # carried across the day-change at bar 20
        assert int(t["entry_decision_index"]) < 20 < xi, (ei, xi)

    # data-end trade must be DATA_END + ineligible
    de = [t for t in trades if t["terminal_reason"] == DATA_END]
    assert de, "expected a DATA_END trade"
    for t in de:
        assert t["training_eligible"] is False


# --------------------------------------------------------------------------- #
# Candidate -> teacher mapping must NEVER cross a hard segment boundary         #
# --------------------------------------------------------------------------- #
def test_mapping_never_crosses_segment():
    seg = np.zeros(100, dtype=np.int64)
    seg[50:] = 1

    # seg0 trade + seg1 trade
    trades = pd.DataFrame(
        [
            {
                "trade_id": "T0", "direction": "LONG",
                "entry_fill_index": 10, "exit_fill_index": 30,
                "entry_fill_price": 100.0, "exit_fill_price": 110.0,
                "terminal_reason": OPTIMAL_FLAT, "training_eligible": True,
            },
            {
                "trade_id": "T1", "direction": "LONG",
                "entry_fill_index": 60, "exit_fill_index": 80,
                "entry_fill_price": 120.0, "exit_fill_price": 130.0,
                "terminal_reason": OPTIMAL_FLAT, "training_eligible": True,
            },
        ]
    )

    cand = pd.DataFrame(
        {
            "candidate_fill_index": [5, 35, 45, 55],  # seg0, seg0, seg0, seg1
        }
    )
    mapped = map_candidates_to_trades(cand, trades, seg)

    # cand 0 (fill 5, seg0) -> T0 (exit 30 > 5)  OK
    assert bool(mapped.loc[0, "mapped"]) is True
    assert mapped.loc[0, "trade_id"] == "T0"

    # cand 1 (fill 35, seg0) -> no forward trade in seg0 -> CENSORED
    assert bool(mapped.loc[1, "mapped"]) is False
    assert mapped.loc[1, "drop_reason"] == "CENSORED_NO_FUTURE_TEACHER"

    # cand 2 (fill 45, seg0) -> MUST NOT map to the seg1 trade (T1, exit 80)
    assert bool(mapped.loc[2, "mapped"]) is False
    assert pd.isna(mapped.loc[2, "trade_id"])
    assert mapped.loc[2, "drop_reason"] == "CENSORED_NO_FUTURE_TEACHER"

    # cand 3 (fill 55, seg1) -> T1 (exit 80 > 55)  OK
    assert bool(mapped.loc[3, "mapped"]) is True
    assert mapped.loc[3, "trade_id"] == "T1"
