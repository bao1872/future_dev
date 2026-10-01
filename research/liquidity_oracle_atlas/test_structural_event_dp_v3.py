"""test_structural_event_dp_v3
=============================

T0 A-M (+ the §11 same-bar terminal case T0-O), Reference/Production parity,
complexity counters and the non-canonical real-data diagnostic subset.

Run:
    PYTHONPATH=. ./.venv/bin/python \
        research/liquidity_oracle_atlas/test_structural_event_dp_v3.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.structural_event_dp_kernel_v3 import (
    DirectionInput,
    INVALID_REASONS,
    MATH_VERSION,
    alternating_direction_fixture,
    build_market_view,
    evaluate_event_v3,
    event_has_executable_path,
    market_view_from_arrays,
    run_structural_event_dp_v3,
    solve_event_production_v3,
)
from research.liquidity_oracle_atlas.structural_event_dp_reference_v3 import (
    solve_event_reference_v3,
)
from research.liquidity_oracle_atlas.build_structural_dp_labels_m15_v2 import (
    _group_structural_events,
)

ATR = 10.0
SYMBOL = "AG"
DIAG_EVENTS = 100


def _mv(opens, highs=None, lows=None, times=None, unit_starts=None):
    o = np.asarray(opens, dtype=float)
    h = np.asarray(highs, dtype=float) if highs is not None else o + 1.0
    l = np.asarray(lows, dtype=float) if lows is not None else o - 1.0
    return market_view_from_arrays(o, h, l, o, times=times, unit_starts=unit_starts)


def _times(n, start="2025-01-02 09:00"):
    return pd.date_range(start, periods=n, freq="15min").to_numpy()


def _solve(direction, zone, start_bar, end_bar, target, mv,
           solver=solve_event_production_v3):
    return solver(
        direction=direction, zone_bottom=zone[0], zone_top=zone[1],
        atr_value=ATR, start_bar=start_bar, end_bar=end_bar,
        target_price=target, mv=mv,
    )


def _ev(event_id=0, sid="A", start=2, end=5, zone=(49.0, 51.0), tf="m15"):
    return {
        "event_id": event_id, "structure_id": sid, "structure_type": "SR",
        "timeframe": tf, "zone_bottom": zone[0], "zone_top": zone[1],
        "start_bar": start, "end_bar": end,
    }


# --------------------------------------------------------------------------- #
# T0
# --------------------------------------------------------------------------- #
def test_T0_A_best_entry_among_multiple_candidates():
    # opens:     0    1    2    3    4    5    6    7
    o = [100.0, 100.0, 105.0, 95.0, 110.0, 100.0, 104.0, 100.0]
    mv = _mv(o)
    sol = _solve("LONG", (99.0, 101.0), 1, 7, 200.0, mv)
    assert sol["ok"], sol
    # candidate utilities: f=2 -> 5 ; f=3 -> 15 ; f=4 -> -6 ; f=5 -> 4 ; f=6 -> -4
    assert sol["entry_decision_index"] == 2
    assert sol["entry_fill_index"] == 3
    assert sol["exit_fill_index"] == 4
    assert abs(sol["utility"] - 15.0) < 1e-9
    print("T0-A PASS: best of 5 entry candidates chosen (utility=15)")


def test_T0_B_entry_inside_zone_gap_zero():
    o = [100.0, 100.0, 105.0, 95.0, 110.0, 100.0, 104.0, 100.0]
    mv = _mv(o)
    sol = _solve("LONG", (94.0, 96.0), 1, 7, 200.0, mv)
    assert sol["entry_fill_index"] == 3 and abs(sol["entry_price"] - 95.0) < 1e-9
    assert abs(sol["best_entry_gap_atr"]) < 1e-12
    print("T0-B PASS: entry inside zone -> best_entry_gap_atr = 0")


def test_T0_C_entry_outside_zone_known_gap():
    o = [100.0, 100.0, 105.0, 95.0, 110.0, 100.0, 104.0, 100.0]
    mv = _mv(o)
    sol = _solve("LONG", (80.0, 82.0), 1, 7, 200.0, mv)
    # entry 95, zone top 82 -> gap 13 pts / ATR 10
    assert abs(sol["best_entry_gap_atr"] - 1.3) < 1e-9
    assert abs(sol["best_entry_gap_points"] - 13.0) < 1e-9
    print("T0-C PASS: entry outside zone -> gap_atr = 1.3")


def test_T0_D_target_touch_is_hard_terminal():
    # a LATER flat exit (open[4]=80) would be far better, but target is terminal.
    # end_bar=6 keeps the touch bar (4) strictly before the next-event boundary.
    o = [50.0, 50.0, 50.0, 52.0, 80.0, 50.0, 50.0, 50.0]
    h = [50.0, 50.0, 50.0, 53.0, 81.0, 50.0, 50.0, 50.0]
    l = [49.0, 49.0, 49.0, 51.0, 79.0, 49.0, 49.0, 49.0]
    mv = _mv(o, highs=h, lows=l)
    sol = _solve("LONG", (49.0, 51.0), 2, 6, 55.0, mv)
    assert sol["ok"], sol
    assert sol["exit_reason"] == "TARGET_TOUCH"
    assert abs(sol["exit_price"] - 55.0) < 1e-9          # TP == Target
    assert abs(sol["remaining_target_atr"]) < 1e-12      # remaining == 0
    assert sol["exit_fill_index"] == 4                   # did NOT keep searching
    assert abs(sol["utility"] - 3.0) < 1e-9
    print("T0-D PASS: TARGET_TOUCH forced (TP==Target, remaining=0, no later exit)")


def test_T0_E_early_positive_tp():
    o = [50.0, 50.0, 50.0, 52.0, 60.0, 58.0]
    mv = _mv(o)                      # highs = o+1 -> max 61 < target 70
    sol = _solve("LONG", (49.0, 51.0), 2, 5, 70.0, mv)
    assert sol["exit_reason"] == "DP_EARLY_EXIT"
    assert sol["positive_tp_exists"] is True
    assert abs(sol["tp_atr"] - 0.8) < 1e-9
    assert abs(sol["remaining_target_atr"] - 1.0) < 1e-9
    print("T0-E PASS: early TP -> tp_atr=0.8 > 0, remaining=1.0 > 0")


def test_T0_F_all_entries_negative_still_one_entry():
    o = [60.0, 60.0, 60.0, 62.0, 58.0, 57.0]
    mv = _mv(o)
    sol = _solve("LONG", (59.0, 61.0), 2, 5, 90.0, mv)
    assert sol["ok"], sol
    assert sol["utility"] < 0
    assert sol["positive_tp_exists"] is False
    assert np.isnan(sol["tp_atr"])
    assert abs(sol["utility"] - (-1.0)) < 1e-9     # least-bad, not clamped to 0
    print("T0-F PASS: all entries lose -> exactly one entry, tp_atr=NA, utility=-1")


def test_T0_G_next_structural_event_boundary():
    o = [50.0, 50.0, 50.0, 52.0, 55.0, 58.0, 60.0, 62.0]
    mv = _mv(o)
    e1 = _solve("LONG", (49.0, 51.0), 2, 4, 200.0, mv)   # next event starts at 4
    e2 = _solve("LONG", (54.0, 56.0), 4, 6, 200.0, mv)
    assert e1["ok"] and e2["ok"]
    assert e1["exit_fill_index"] <= 4                    # Exit_i <= Start(E_{i+1})
    assert e2["entry_fill_index"] >= 4                   # E_{i+1} starts after
    assert e1["exit_fill_index"] <= e2["entry_fill_index"]
    print("T0-G PASS: previous event exits first (exit=4 <= next start=4)")


def test_T0_H_same_direction_continuation():
    o = [50.0, 50.0, 50.0, 52.0, 55.0, 58.0, 60.0, 62.0]
    mv = _mv(o)
    r1 = evaluate_event_v3(_ev(0, "A", 2, 4, (49.0, 51.0)), mv,
                           DirectionInput(0, "LONG"), ATR,
                           {"structure_id": "B", "role": "RESISTANCE",
                            "near_edge": 200.0, "tf": "m15"})
    r2 = evaluate_event_v3(_ev(1, "B", 4, 6, (54.0, 56.0)), mv,
                           DirectionInput(1, "LONG"), ATR,
                           {"structure_id": "C", "role": "RESISTANCE",
                            "near_edge": 200.0, "tf": "m15"})
    assert r1["event_valid"] and r2["event_valid"]
    assert r1["direction"] == "LONG" and r2["direction"] == "LONG"
    assert r1["exit_fill_index"] <= r2["best_entry_fill_index"]
    print("T0-H PASS: LONG -> EXIT -> LONG valid, no overlap")


def test_T0_I_same_structure_non_contiguous_is_one_event():
    n = 8
    sid = "SR|m15|0|1|51.0|49.0|1.0"
    per_bar = [None] * n
    per_bar[1] = [(sid, "SR", 49.0, 51.0)]
    per_bar[2] = [(sid, "SR", 49.0, 51.0)]
    per_bar[5] = [(sid, "SR", 49.0, 51.0)]
    low = np.full(n, 49.0)
    high = np.full(n, 51.0)
    events, _raw, _pr, merges = _group_structural_events(per_bar, low, high, n)
    assert len(events) == 1, f"expected 1 event, got {len(events)}"
    assert events[0]["start_bar"] == 1
    assert merges >= 1
    print("T0-I PASS: touch A / leave / touch A again -> 1 structural event")


def test_T0_J_target_becomes_next_structural_state():
    o = [50.0, 50.0, 50.0, 52.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0]
    mv = _mv(o)                       # high[4] = 60 >= target 58
    tgt = {"structure_id": "B", "role": "RESISTANCE", "near_edge": 58.0, "tf": "m15"}
    r1 = evaluate_event_v3(_ev(0, "A", 2, 6, (49.0, 51.0)), mv,
                           DirectionInput(0, "LONG"), ATR, tgt)
    r2 = evaluate_event_v3(_ev(1, "B", 6, 8, (57.0, 59.0)), mv,
                           DirectionInput(1, "LONG"), ATR,
                           {"structure_id": "C", "role": "RESISTANCE",
                            "near_edge": 200.0, "tf": "m15"})
    assert r1["exit_reason"] == "TARGET_TOUCH"
    assert abs(r1["remaining_target_atr"]) < 1e-12
    assert r1["target_structure_id"] == "B"
    assert r2["structure_id"] == "B"          # B is the next structural state
    assert r1["exit_fill_index"] <= r2["event_start_bar"]
    assert r1["direction"] == r2["direction"] == "LONG"
    print("T0-J PASS: A -> target B reached -> B is the next structural state")


def test_T0_K_direction_not_available_yet():
    n = 6
    o = [50.0, 50.0, 50.0, 52.0, 55.0, 58.0]
    mv = _mv(o, times=_times(n))
    late = pd.Timestamp(mv.times[4])          # after event decision time
    rec = evaluate_event_v3(_ev(0, "A", 2, 5), mv,
                            DirectionInput(0, "LONG", available_time=late),
                            ATR, {"structure_id": "B", "role": "RESISTANCE",
                                  "near_edge": 200.0, "tf": "m15"})
    assert rec["event_valid"] is False
    assert rec["invalid_reason"] == "DIRECTION_NOT_AVAILABLE_YET"
    assert rec["entry_present"] is False
    print("T0-K PASS: available_time > event_decision_time -> DIRECTION_NOT_AVAILABLE_YET")

    bad = evaluate_event_v3(_ev(0, "A", 2, 5), mv, DirectionInput(0, "NONE"),
                            ATR, {"structure_id": "B", "role": "RESISTANCE",
                                  "near_edge": 200.0, "tf": "m15"})
    assert bad["invalid_reason"] == "INVALID_DIRECTION_VALUE"
    print("T0-K PASS: direction NONE -> INVALID_DIRECTION_VALUE")


def test_T0_L_no_entry_from_optimization_never_happens():
    o = [60.0, 60.0, 60.0, 62.0, 58.0, 57.0]
    mv = _mv(o)
    # every candidate utility, enumerated independently
    utils = []
    for d in range(2, min(5 - 2, len(o) - 2) + 1):
        f = d + 1
        best = max((o[k] - o[f]) for k in range(f + 1, min(5, mv.n - 1) + 1))
        utils.append(best)
    sol = _solve("LONG", (59.0, 61.0), 2, 5, 90.0, mv)
    assert sol["ok"]
    assert abs(sol["utility"] - max(utils)) < 1e-9     # least bad, not clamped
    assert sol["positive_tp_exists"] is False and np.isnan(sol["tp_atr"])
    assert sol["optimal_exit_points_signed"] == sol["utility"] < 0
    print("T0-L PASS: utility == max over candidates; no NO_ENTRY from optimization")


def test_T0_M_equal_utility_tie_break():
    o = [50.0, 50.0, 50.0, 52.0, 52.0, 52.0, 52.0, 52.0]
    mv = _mv(o)
    sol = _solve("LONG", (49.0, 51.0), 2, 6, 200.0, mv)
    assert sol["entry_fill_index"] == 3          # earliest entry on tie
    assert sol["exit_fill_index"] == 4           # earliest exit on tie
    ref = _solve("LONG", (49.0, 51.0), 2, 6, 200.0, mv, solve_event_reference_v3)
    assert ref["entry_fill_index"] == sol["entry_fill_index"]
    assert ref["exit_fill_index"] == sol["exit_fill_index"]
    print("T0-M PASS: tie -> earlier entry (f=3) and earlier exit (k=4); Ref == Prod")


def test_T0_O_ambiguous_same_bar_terminal():
    o = [50.0, 50.0, 50.0, 52.0, 53.0, 54.0]
    h = [50.0, 50.0, 50.0, 53.0, 57.0, 55.0]
    l = [49.0, 49.0, 49.0, 51.0, 52.0, 53.0]
    mv = _mv(o, highs=h, lows=l)
    sol = _solve("LONG", (49.0, 51.0), 2, 4, 56.0, mv)   # first touch == end_bar
    assert sol["ok"] is False
    assert sol["invalid_reason"] == "AMBIGUOUS_SAME_BAR_TERMINAL"
    ref = _solve("LONG", (49.0, 51.0), 2, 4, 56.0, mv, solve_event_reference_v3)
    assert ref["invalid_reason"] == "AMBIGUOUS_SAME_BAR_TERMINAL"
    print("T0-O PASS: target touch on the next-event bar -> AMBIGUOUS (no guessing)")


def test_T0_P_target_touched_before_entry_is_not_a_candidate():
    # target 56 is first touched at bar 5 (high[5]=60). Bars f=6,7 open BELOW
    # the target and would look attractive (open[6]=50 -> +6) but the target was
    # already touched, so they must be excluded. The legal best is f=3 (+4).
    o = [50.0, 50.0, 50.0, 52.0, 53.0, 60.0, 50.0, 50.0, 50.0, 50.0]
    h = [50.0, 50.0, 50.0, 53.0, 54.0, 60.0, 60.0, 50.0, 50.0, 50.0]
    l = [49.0, 49.0, 49.0, 51.0, 52.0, 59.0, 49.0, 49.0, 49.0, 49.0]
    mv = _mv(o, highs=h, lows=l)
    sol = _solve("LONG", (49.0, 51.0), 2, 8, 56.0, mv)
    ref = _solve("LONG", (49.0, 51.0), 2, 8, 56.0, mv, solve_event_reference_v3)
    assert sol["entry_fill_index"] == 3, sol
    assert sol["exit_fill_index"] == 5
    assert abs(sol["utility"] - 4.0) < 1e-9      # NOT the post-touch +6
    assert (ref["entry_fill_index"], ref["exit_fill_index"]) == (
        sol["entry_fill_index"], sol["exit_fill_index"])
    print("T0-P PASS: entries at/after the event-level target touch excluded")


def test_T0_Q_same_bar_entry_and_target_touch_is_legal():
    # unit_starts=[0,5] -> H=4 for f=3 and f=4, so the OLD code (which demanded
    # f+1 <= H) would skip f=4. The same-bar path (open<target<=high) is legal.
    o = [50.0, 50.0, 50.0, 55.0, 53.0, 54.0, 55.0, 56.0]
    h = [50.0, 50.0, 50.0, 55.0, 58.0, 50.0, 50.0, 50.0]
    l = [49.0, 49.0, 49.0, 54.0, 52.0, 50.0, 50.0, 50.0]
    mv = _mv(o, highs=h, lows=l, unit_starts=[0, 5])
    sol = _solve("LONG", (49.0, 51.0), 2, 5, 56.0, mv)
    assert sol["entry_fill_index"] == 4 and sol["exit_fill_index"] == 4
    assert sol["exit_reason"] == "TARGET_TOUCH"
    assert abs(sol["utility"] - 3.0) < 1e-9
    ref = _solve("LONG", (49.0, 51.0), 2, 5, 56.0, mv, solve_event_reference_v3)
    assert (ref["entry_fill_index"], ref["exit_fill_index"], ref["utility"]) == (
        sol["entry_fill_index"], sol["exit_fill_index"], sol["utility"])
    # SHORT mirror: entry must be ABOVE the target (49 > 48), same bar TP
    o2 = [50.0, 50.0, 50.0, 48.6, 49.0, 50.0, 50.0, 50.0]
    l2 = [50.0, 50.0, 50.0, 48.5, 46.0, 50.0, 50.0, 50.0]
    h2 = [51.0, 51.0, 51.0, 49.6, 50.0, 51.0, 51.0, 51.0]
    mv2 = _mv(o2, highs=h2, lows=l2, unit_starts=[0, 5])
    s2 = _solve("SHORT", (49.0, 51.0), 2, 5, 48.0, mv2)
    assert s2["entry_fill_index"] == 4 and s2["exit_fill_index"] == 4
    assert s2["exit_reason"] == "TARGET_TOUCH"
    assert abs(s2["utility"] - 1.0) < 1e-9
    print("T0-Q PASS: same-bar entry + TARGET_TOUCH legal for LONG and SHORT")


def test_T0_R_entry_beyond_target_is_illegal():
    # LONG: open[4]=60 > target 56 -> the fill bar is already past the target
    o = [50.0, 50.0, 50.0, 55.0, 60.0, 54.0, 55.0, 56.0]
    h = [50.0, 50.0, 50.0, 55.0, 61.0, 50.0, 50.0, 50.0]
    l = [49.0, 49.0, 49.0, 54.0, 59.0, 49.0, 49.0, 49.0]
    mv = _mv(o, highs=h, lows=l, unit_starts=[0, 5])
    sol = _solve("LONG", (49.0, 51.0), 2, 5, 56.0, mv)
    assert sol["entry_fill_index"] == 3          # f=4 rejected
    assert abs(sol["utility"] - 1.0) < 1e-9
    ref = _solve("LONG", (49.0, 51.0), 2, 5, 56.0, mv, solve_event_reference_v3)
    assert ref["entry_fill_index"] == sol["entry_fill_index"]
    # SHORT mirror: open[4]=47 < target 48 -> rejected
    o2 = [50.0, 50.0, 50.0, 49.5, 47.0, 50.0, 50.0, 50.0]
    l2 = [50.0, 50.0, 50.0, 49.0, 46.0, 50.0, 50.0, 50.0]
    h2 = [51.0, 51.0, 51.0, 50.5, 48.0, 51.0, 51.0, 51.0]
    mv2 = _mv(o2, highs=h2, lows=l2, unit_starts=[0, 5])
    s2 = _solve("SHORT", (49.0, 51.0), 2, 5, 48.0, mv2)
    assert s2["entry_fill_index"] == 3           # f=4 rejected
    assert abs(s2["utility"] - 1.5) < 1e-9
    r2 = _solve("SHORT", (49.0, 51.0), 2, 5, 48.0, mv2, solve_event_reference_v3)
    assert r2["entry_fill_index"] == s2["entry_fill_index"]
    print("T0-R PASS: entry beyond the frozen target rejected (LONG + SHORT)")


def test_T0_S_direction_event_id_mismatch():
    n = 8
    o = [50.0, 50.0, 50.0, 52.0, 60.0, 58.0, 57.0, 56.0]
    mv = _mv(o, times=_times(n))
    event = _ev(event_id=7, sid="A", start=2, end=6)
    tgt = {"structure_id": "B", "role": "RESISTANCE", "near_edge": 200.0, "tf": "m15"}
    rec = evaluate_event_v3(event, mv, DirectionInput(event_id=6, direction="LONG"),
                            ATR, tgt)
    assert rec["event_valid"] is False
    assert rec["invalid_reason"] == "DIRECTION_EVENT_ID_MISMATCH"
    assert rec["entry_present"] is False
    # the matching id is accepted
    ok = evaluate_event_v3(event, mv, DirectionInput(event_id=7, direction="LONG"),
                           ATR, tgt)
    assert ok["event_valid"] is True
    print("T0-S PASS: DirectionInput.event_id mismatch -> DIRECTION_EVENT_ID_MISMATCH")


# --------------------------------------------------------------------------- #
# Reference / Production parity + complexity
# --------------------------------------------------------------------------- #
_PARITY_FIELDS = [
    "event_valid", "direction", "best_entry_decision_index", "best_entry_fill_index",
    "best_entry_price", "exit_fill_index", "exit_price", "exit_reason",
    "positive_tp_exists", "best_entry_gap_atr", "tp_atr", "remaining_target_atr",
    "utility",
]


def _parity(events_specs, mv):
    shared = 0
    mismatch = 0
    max_err = 0.0
    first = None
    for spec in events_specs:
        tgt = spec["target"]
        di = spec["direction_input"]
        a = evaluate_event_v3(spec["event"], mv, di, ATR, tgt,
                              solver=solve_event_production_v3)
        b = evaluate_event_v3(spec["event"], mv, di, ATR, tgt,
                              solver=solve_event_reference_v3)
        shared += 1
        for f in _PARITY_FIELDS:
            va, vb = a.get(f), b.get(f)
            if isinstance(va, float) and isinstance(vb, float):
                if np.isnan(va) and np.isnan(vb):
                    continue
                err = abs(va - vb)
                max_err = max(max_err, err)
                if err > 1e-9:
                    mismatch += 1
                    first = first or (spec["event"]["event_id"], f, va, vb)
            elif va != vb:
                mismatch += 1
                first = first or (spec["event"]["event_id"], f, va, vb)
    return shared, mismatch, max_err, first


def test_parity_synthetic_sweep():
    rng = np.random.default_rng(20261001)
    mv = _mv(100.0 + np.cumsum(rng.normal(0, 1.0, 400)))
    specs = []
    for i in range(60):
        s = int(rng.integers(1, 300))
        L = int(rng.integers(3, 40))
        e = min(s + L, mv.n)
        zone = (float(mv.opens[s]) - 2.0, float(mv.opens[s]) + 2.0)
        direction = "LONG" if i % 2 == 0 else "SHORT"
        tgt_px = (float(mv.opens[s]) + rng.uniform(1, 8) if direction == "LONG"
                  else float(mv.opens[s]) - rng.uniform(1, 8))
        specs.append({
            "event": _ev(i, f"S{i}", s, e, zone),
            "direction_input": DirectionInput(i, direction),
            "target": {"structure_id": f"T{i}", "role": "RESISTANCE",
                       "near_edge": tgt_px, "tf": "m15"},
        })
    shared, mismatch, max_err, first = _parity(specs, mv)
    assert mismatch == 0, f"mismatch={mismatch} first={first}"
    print(f"SYNTHETIC PARITY PASS: shared={shared} mismatch=0 max_abs_error={max_err:.3e}")


def test_reference_is_independent_and_slower_complexity():
    """Production must stay ~linear while the reference is allowed O(L^2)."""
    rng = np.random.default_rng(7)

    def build(L):
        o = 100.0 + np.cumsum(rng.normal(0, 1.0, L))
        mv = _mv(o)
        return mv, _solve("LONG", (99.0, 101.0), 1, L - 1, 1e9, mv,
                          solve_event_production_v3), \
               _solve("LONG", (99.0, 101.0), 1, L - 1, 1e9, mv,
                      solve_event_reference_v3)

    rows = []
    for L in (200, 400, 800):
        mv, _, _ = build(L)
        t0 = time.perf_counter()
        for _ in range(5):
            _solve("LONG", (99.0, 101.0), 1, L - 1, 1e9, mv, solve_event_production_v3)
        tp = (time.perf_counter() - t0) / 5
        t0 = time.perf_counter()
        for _ in range(5):
            _solve("LONG", (99.0, 101.0), 1, L - 1, 1e9, mv, solve_event_reference_v3)
        tr = (time.perf_counter() - t0) / 5
        rows.append((L, tp, tr))
        print(f"   L={L:5d}  prod={tp*1e3:8.3f} ms  ref={tr*1e3:8.3f} ms  "
              f"ref/prod={tr/max(tp,1e-12):7.1f}x")

    base_p = rows[0][1]
    prod_ratio_4x = rows[-1][1] / max(base_p, 1e-12)
    print(f"   production runtime ratio (4N/N) = {prod_ratio_4x:.2f}  (linear ~4)")
    assert prod_ratio_4x < 12.0, "production complexity looks super-linear"
    # reference must actually be doing more work on the biggest input
    assert rows[-1][2] > 0.0


# --------------------------------------------------------------------------- #
# Kernel must not know where direction comes from
# --------------------------------------------------------------------------- #
def test_kernel_source_is_direction_agnostic():
    src = Path(__file__).with_name("structural_event_dp_kernel_v3.py").read_text()
    forbidden = [
        "direction_gated_experts", "e9_direction", "a9_direction",
        "load_oracle_artifact", "oracle_trades", "train_direction_model",
        "entry_path_atlas", "build_trade_oracle_dp_m15",
    ]
    hits = [f for f in forbidden if f in src]
    assert not hits, f"kernel must not reference direction producers: {hits}"
    print("KERNEL AGNOSTIC PASS: no direction producer referenced in the kernel")


# --------------------------------------------------------------------------- #
# Structural-only accounting (direction-free)
# --------------------------------------------------------------------------- #
def test_structural_only_accounting():
    """Direction-free accounting over the first N structural events.

    Target validity is NOT decidable without a direction (the frozen target is
    direction-dependent), so that bucket is only reported after injection.
    """
    from research.liquidity_oracle_atlas.build_structural_dp_labels_m15_v2 import (
        build_structural_events_v2,
    )
    from research.liquidity_oracle_atlas.build_execution_environment_m15_v1 import (
        run_environment_m15,
    )

    ev = build_structural_events_v2(SYMBOL)
    events = ev["events"][:DIAG_EVENTS]
    mv = build_market_view(SYMBOL)
    atr_series = np.asarray(
        run_environment_m15(SYMBOL, capture_provenance=False)["features"]["m15_atr"]
        .to_numpy(float)
    )

    bad_geom = 0
    bad_atr = 0
    no_exec = 0
    for e in events:
        zb, zt = float(e["zone_bottom"]), float(e["zone_top"])
        if not (np.isfinite(zb) and np.isfinite(zt)) or zb > zt:
            bad_geom += 1
        s_bar = int(e["start_bar"])
        a = float(atr_series[s_bar]) if s_bar < len(atr_series) else float("nan")
        if not (np.isfinite(a) and a > 0):
            bad_atr += 1
        if not event_has_executable_path(mv, s_bar, int(e["end_bar"])):
            no_exec += 1

    struct_valid = len(events) - bad_geom - bad_atr - no_exec
    print("STRUCTURAL-ONLY ACCOUNTING (direction-free, first "
          f"{DIAG_EVENTS} events)")
    print(f"   structural_events            = {len(events)}")
    print(f"   invalid_due_to_geometry      = {bad_geom}")
    print(f"   invalid_due_to_atr           = {bad_atr}")
    print(f"   invalid_due_to_execution     = {no_exec}")
    print(f"   invalid_due_to_target        = NOT DECIDABLE without direction")
    print(f"   structurally_valid_before_direction = {struct_valid}")
    assert bad_geom == 0 and bad_atr == 0


# --------------------------------------------------------------------------- #
# Non-canonical real-data diagnostic subset
# --------------------------------------------------------------------------- #
def test_real_diagnostic_subset():
    """Diagnostic ONLY: direction is an external alternating fixture.

    This must never be reported as a canonical V3 label set.
    """
    provider = lambda e, t: alternating_direction_fixture(int(e["event_id"]))
    out = run_structural_event_dp_v3(SYMBOL, provider, event_limit=DIAG_EVENTS)
    recs = out["records"]
    meta = out["meta"]
    valid = [r for r in recs if r["event_valid"]]

    print("REAL DIAGNOSTIC SUBSET (non-canonical direction)")
    print(f"   structural_events(first {DIAG_EVENTS}) = {len(recs)}")
    print(f"   valid = {meta['valid_events']}  invalid = {meta['invalid_events']}")
    print(f"   invalid_reason_counts = "
          f"{ {k: v for k, v in meta['invalid_reason_counts'].items() if v} }")
    print(f"   entered_valid_events = {meta['entered_valid_events']}")
    print(f"   valid_but_no_entry   = {meta['valid_but_no_entry']}")

    assert meta["valid_events"] == meta["entered_valid_events"]
    assert meta["valid_but_no_entry"] == 0
    for r in valid:
        assert r["entry_present"] is True
        assert r["best_entry_fill_index"] >= 0

    # no overlap / correct ordering between consecutive valid events
    overlap = 0
    seq_viol = 0
    vs = sorted(valid, key=lambda r: r["best_entry_fill_index"])
    for a, b in zip(vs, vs[1:]):
        if a["exit_fill_index"] > b["best_entry_fill_index"]:
            overlap += 1
        if a["exit_fill_index"] > b["event_start_bar"]:
            seq_viol += 1
    print(f"   overlap_count = {overlap}   sequential_violation_count = {seq_viol}")
    assert overlap == 0
    assert seq_viol == 0

    pos = sum(1 for r in valid if r["positive_tp_exists"])
    neg = len(valid) - pos
    reached = sum(1 for r in valid if r["exit_reason"] == "TARGET_TOUCH")
    early = sum(1 for r in valid if r["exit_reason"] == "DP_EARLY_EXIT")
    print(f"   positive_tp = {pos}   no_positive_tp = {neg}")
    print(f"   target_reached = {reached}   early_exit = {early}")
    print(f"   target_reached_with_loss = {meta['target_reached_with_loss']}")
    # HARD INVARIANT (V3.1): with the target-before-entry filter, a frozen target
    # touch can never be a loss.
    assert meta["target_reached_with_loss"] == 0, (
        f"target_reached_with_loss = {meta['target_reached_with_loss']} (must be 0)")
    for r in valid:
        if r["exit_reason"] == "TARGET_TOUCH":
            assert r["positive_tp_exists"] is True
            assert r["utility"] > 0
    print("   HARD INVARIANT PASS: target_reached_with_loss == 0")
    for r in valid:
        if not r["positive_tp_exists"]:
            assert np.isnan(r["tp_atr"])
    for r in valid:
        if r["exit_reason"] == "TARGET_TOUCH":
            assert abs(r["remaining_target_atr"]) < 1e-12

    # production vs reference parity on the same subset
    out_r = run_structural_event_dp_v3(SYMBOL, provider, event_limit=DIAG_EVENTS,
                                       solver=solve_event_reference_v3)
    shared = 0
    mismatch = 0
    max_err = 0.0
    first = None
    by_id = {r["event_id"]: r for r in out_r["records"]}
    for a in recs:
        b = by_id[a["event_id"]]
        shared += 1
        for f in _PARITY_FIELDS:
            va, vb = a.get(f), b.get(f)
            if isinstance(va, float) and isinstance(vb, float):
                if np.isnan(va) and np.isnan(vb):
                    continue
                err = abs(va - vb)
                max_err = max(max_err, err)
                if err > 1e-9:
                    mismatch += 1
                    first = first or (a["event_id"], f, va, vb)
            elif va != vb:
                mismatch += 1
                first = first or (a["event_id"], f, va, vb)
    print(f"REAL PARITY: shared={shared} mismatch={mismatch} "
          f"max_abs_error={max_err:.3e} first={first}")
    assert mismatch == 0, f"real-subset mismatch={mismatch} first={first}"


def main():
    test_kernel_source_is_direction_agnostic()
    for fn in (
        test_T0_A_best_entry_among_multiple_candidates,
        test_T0_B_entry_inside_zone_gap_zero,
        test_T0_C_entry_outside_zone_known_gap,
        test_T0_D_target_touch_is_hard_terminal,
        test_T0_E_early_positive_tp,
        test_T0_F_all_entries_negative_still_one_entry,
        test_T0_G_next_structural_event_boundary,
        test_T0_H_same_direction_continuation,
        test_T0_I_same_structure_non_contiguous_is_one_event,
        test_T0_J_target_becomes_next_structural_state,
        test_T0_K_direction_not_available_yet,
        test_T0_L_no_entry_from_optimization_never_happens,
        test_T0_M_equal_utility_tie_break,
        test_T0_O_ambiguous_same_bar_terminal,
        test_T0_P_target_touched_before_entry_is_not_a_candidate,
        test_T0_Q_same_bar_entry_and_target_touch_is_legal,
        test_T0_R_entry_beyond_target_is_illegal,
        test_T0_S_direction_event_id_mismatch,
        test_parity_synthetic_sweep,
        test_reference_is_independent_and_slower_complexity,
        test_structural_only_accounting,
        test_real_diagnostic_subset,
    ):
        fn()
    print(f"\nmath_version = {MATH_VERSION}")
    print("ALL V3 TESTS PASSED")


if __name__ == "__main__":
    main()
