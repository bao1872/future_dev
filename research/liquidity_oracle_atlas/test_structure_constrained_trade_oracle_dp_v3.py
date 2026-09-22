"""test_structure_constrained_trade_oracle_dp_v3
==============================================

Committed (clean-checkout) R3 DP gate tests
(FUTURE-R3-CANONICAL-M5-TOUCH-NEXTBAR-GATE-V1, checkpoint A).

The Bellman kernel is reused from V2 unchanged; only the exogenous entry gate
switches from distance-based ``proximity_any`` to the canonical R3
``candidate_any``. These tests MUST run in a clean checkout (``git clone &&
pytest``), so they use only SYNTHETIC gates (no committed parquet artifact):

1. DP = exhaustive DFS over all 6 states with the candidate gate (no math drift).
2. A new-entry action can NEVER occur at a non-candidate bar (gate breach = 0).
3. With the gate fully off, the oracle cannot open at all (stays flat).

The AG / RB / AU full-history smoke (illegal new-entry = 0 on real artifacts)
lives in ``verify_candidate_gate_r3_artifacts.py`` because the parquet artifacts
are intentionally NOT committed to Git.

No model / Y / Q / action artifact is produced here.
"""

import numpy as np
import pytest

from research.liquidity_oracle_atlas.build_structure_constrained_trade_oracle_dp_v2 import (
    STATES,
    _solve_unit_v2,
    _walk_unit_path,
    exhaustive_reference_v2,
    solve_day_dp_v2,
)
from research.liquidity_oracle_atlas.build_structure_constrained_trade_oracle_dp_v3 import (
    MATH_VERSION,
    solve_day_dp_v3,
)


def _pad(opens, cand, cost, n):
    """Pad to the day-DP convention: arrays need length end+2 for final fill."""
    tail_o = np.array([opens[-1], opens[-1]], dtype=float)
    tail_b = np.array([False, False])
    tail_c = np.array([0.0, 0.0], dtype=float)
    return (
        np.concatenate([opens, tail_o]),
        np.concatenate([cand, tail_b]),
        np.concatenate([cost, tail_c]),
    )


def _dp_vs_dfs(opens, candidate_any, cost, start, end):
    """Return (exhaustive best, dp value) for the frozen (Flat, armed=1) start."""
    o2, c2, k2 = _pad(opens, candidate_any, cost, end)
    core = solve_day_dp_v3(o2, c2, k2, start, end)
    s_f1 = STATES.index((0, 1))
    dp_val = float(np.nanmax(core["Q"][0, s_f1, :]))
    best, _ = exhaustive_reference_v2(
        o2, c2, k2, start, end, start_pos=0, start_armed=1
    )
    return best, dp_val


def test_dp_equals_dfs_synthetic():
    opens = np.array(
        [100.0, 101.0, 100.2, 102.0, 99.5, 101.3, 100.1, 103.0, 98.0, 99.5,
         101.0, 97.0, 100.5, 99.0], dtype=float
    )
    cand = np.zeros(14, dtype=bool)
    cand[[1, 3, 4, 7, 9, 11]] = True
    cost = np.full(14, 0.02, dtype=float)
    best, dp_val = _dp_vs_dfs(opens, cand, cost, 0, 14)
    assert abs(best - dp_val) < 1e-9, f"dp={dp_val} dfs={best}"


def test_dp_equals_dfs_all_candidate():
    opens = 100.0 + np.array(
        [0, 1, -1, 2, -2, 1, 1, -1, 0, 2, -1, -2, 3, -1, 1, -1, 0, 1, -1, 2],
        dtype=float,
    )
    cand = np.ones(20, dtype=bool)
    cost = np.full(20, 0.05, dtype=float)
    best, dp_val = _dp_vs_dfs(opens, cand, cost, 0, 20)
    assert abs(best - dp_val) < 1e-9


def test_no_illegal_new_entry_when_gate_off():
    # gate fully off -> the executed path never takes a new-entry action
    opens = 100.0 + np.arange(12, dtype=float)
    cand = np.zeros(12, dtype=bool)
    cost = np.zeros(12, dtype=float)
    o2, c2, k2 = _pad(opens, cand, cost, 12)
    core = solve_day_dp_v3(o2, c2, k2, 0, 12)
    path = _walk_unit_path(core, c2, 0, 12)
    for d in path:
        if d["ne"]:
            assert bool(c2[d["t"]]), f"illegal new entry at non-candidate bar {d['t']}"


def test_new_entry_only_at_candidate_bars():
    # only bar 3 is a candidate; any new entry must occur there
    opens = 100.0 + np.concatenate(
        [np.zeros(5), np.array([1.0, -1.0, 0.5, -0.5, 2.0, -2.0])]
    )
    n = len(opens)
    cand = np.zeros(n, dtype=bool)
    cand[3] = True  # only one eligible bar
    cost = np.full(n, 0.01, dtype=float)
    o2, c2, k2 = _pad(opens, cand, cost, n)
    core = solve_day_dp_v3(o2, c2, k2, 0, n)
    path = _walk_unit_path(core, c2, 0, n)
    for d in path:
        if d["ne"]:
            assert bool(c2[d["t"]]), f"new entry at non-candidate bar {d['t']}"


def test_math_version_frozen():
    assert MATH_VERSION == "intraday_dp_oracle_r3_m5_touch_nextbar_candidate"
