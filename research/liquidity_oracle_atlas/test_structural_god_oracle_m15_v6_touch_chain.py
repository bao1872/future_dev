"""Minimal tests for the V6.1 LOCATION-CHAIN oracle core.

These tests exercise ONLY the first-principles location model:
    True Touch -> Location -> frozen A -> distinct B -> Entry -> Exit.

No episode owner, no source-cluster, no global DP.
"""
import numpy as np
import pytest

from research.liquidity_oracle_atlas.structural_god_oracle_m15_v6_touch_chain import (
    build_location_frames,
    locations_from_bar_matches,
    solve_location_touch_chain,
    LONG,
    SHORT,
    TARGET_TOUCH,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def mk_match(bottom, top, family="SR", side=None, level=None, tf="m15"):
    m = {"tf": tf, "family": family, "bottom": float(bottom), "top": float(top)}
    if side is not None:
        m["side"] = side
    if level is not None:
        m["level"] = float(level)
    return m


def frames_from_spec(spec):
    """spec: list of (bar, [(bottom, top, family?, side?, level?), ...])."""
    n = max(b for b, _ in spec) + 1
    matches = [[] for _ in range(n)]
    for bar, items in spec:
        for it in items:
            bottom, top = it[0], it[1]
            family = it[2] if len(it) > 2 else "SR"
            side = it[3] if len(it) > 3 else None
            level = it[4] if len(it) > 4 else None
            matches[bar].append(mk_match(bottom, top, family, side, level))
    return build_location_frames(matches)


def solve(spec, opens):
    frames = frames_from_spec(spec)
    n = len(frames)
    segments = np.ones(n, dtype=np.int64)
    tdays = np.ones(n, dtype=np.int64)
    opens = np.asarray(opens, dtype=float)
    return solve_location_touch_chain(frames, opens, segments, tdays)


def solve_with(spec, opens, segments, tdays):
    frames = frames_from_spec(spec)
    opens = np.asarray(opens, dtype=float)
    segments = np.asarray(segments, dtype=np.int64)
    tdays = np.asarray(tdays, dtype=np.int64)
    return solve_location_touch_chain(frames, opens, segments, tdays)


# --------------------------------------------------------------------------- #
# 1. same bar, overlapping -> ONE location
# --------------------------------------------------------------------------- #
def test_same_bar_overlapping_merge_to_one_location():
    locs = locations_from_bar_matches(0, [
        mk_match(100, 110),
        mk_match(105, 115),
    ])
    assert len(locs) == 1
    assert locs[0].bottom == pytest.approx(100)
    assert locs[0].top == pytest.approx(115)


# --------------------------------------------------------------------------- #
# 2. same bar, disjoint -> TWO locations
# --------------------------------------------------------------------------- #
def test_same_bar_disjoint_two_locations():
    locs = locations_from_bar_matches(0, [
        mk_match(100, 110),
        mk_match(200, 210),
    ])
    assert len(locs) == 2


# --------------------------------------------------------------------------- #
# 3. TRANSITIVE DRIFT: frozen A must NOT expand
# --------------------------------------------------------------------------- #
def test_frozen_source_no_transitive_drift():
    spec = [
        (0, [(100, 110)]),       # A (frozen)
        (1, [(109, 119)]),       # X overlaps frozen A -> retouch
        (2, [(118, 128)]),       # Y overlaps X, NOT frozen A -> NEW target
    ]
    opens = [0.0, 100.0, 100.0, 100.0, 0.0]
    trades, audit = solve(spec, opens)
    assert audit["same_location_retouch_bars"] == 1
    assert audit["target_transitions"] == 1
    assert audit["canonical_trades"] == 1
    t = trades[0]
    # frozen A must stay [100,110]; never expand to 119 or 128
    assert t["zone_bottom"] == pytest.approx(100)
    assert t["zone_top"] == pytest.approx(110)
    assert t["oracle_direction"] == LONG


# --------------------------------------------------------------------------- #
# 4. A A A B -> ONE trade
# --------------------------------------------------------------------------- #
def test_A_A_A_B_one_trade():
    spec = [
        (0, [(100, 110)]),
        (1, [(100, 110)]),
        (2, [(100, 110)]),
        (3, [(200, 210)]),
    ]
    opens = [0.0, 100.0, 100.0, 100.0, 0.0]
    trades, audit = solve(spec, opens)
    assert audit["canonical_trades"] == 1
    assert len(trades) == 1
    assert trades[0]["oracle_direction"] == LONG


# --------------------------------------------------------------------------- #
# 5. A A B B C -> TWO trades (A->B, B->C)
# --------------------------------------------------------------------------- #
def test_A_A_B_B_C_two_trades():
    spec = [
        (0, [(100, 110)]),
        (1, [(100, 110)]),
        (2, [(200, 210)]),
        (3, [(200, 210)]),
        (4, [(150, 160)]),
    ]
    opens = [0.0, 100.0, 100.0, 170.0, 170.0, 0.0]
    trades, audit = solve(spec, opens)
    assert audit["canonical_trades"] == 2
    assert trades[0]["oracle_direction"] == LONG   # A -> B (up)
    assert trades[1]["oracle_direction"] == SHORT  # B -> C (down)


# --------------------------------------------------------------------------- #
# 6. A -> upper B -> LONG
# --------------------------------------------------------------------------- #
def test_upper_target_is_long():
    spec = [(0, [(100, 110)]), (1, [(200, 210)])]
    opens = [0.0, 100.0, 0.0]
    trades, audit = solve(spec, opens)
    assert audit["canonical_trades"] == 1
    assert trades[0]["oracle_direction"] == LONG
    assert trades[0]["target_price"] == pytest.approx(200)


# --------------------------------------------------------------------------- #
# 7. A -> lower B -> SHORT
# --------------------------------------------------------------------------- #
def test_lower_target_is_short():
    spec = [(0, [(200, 210)]), (1, [(100, 110)])]
    opens = [0.0, 200.0, 0.0]
    trades, audit = solve(spec, opens)
    assert audit["canonical_trades"] == 1
    assert trades[0]["oracle_direction"] == SHORT
    assert trades[0]["target_price"] == pytest.approx(110)  # B.top


# --------------------------------------------------------------------------- #
# 8. same target bar: upper + lower distinct -> AMBIGUOUS
# --------------------------------------------------------------------------- #
def test_ambiguous_upper_and_lower_same_bar():
    spec = [
        (0, [(150, 160)]),                 # A
        (1, [(200, 210), (100, 110)]),     # C above, D below -> ambiguous
    ]
    opens = [0.0, 150.0, 0.0]
    trades, audit = solve(spec, opens)
    assert audit["canonical_trades"] == 0
    assert audit["ambiguous_target_bars"] == 1


# --------------------------------------------------------------------------- #
# 9. repeated A contacts -> only ONE Entry survives (min/max next-open)
# --------------------------------------------------------------------------- #
def test_one_entry_from_repeated_contacts():
    # LONG: smallest next open is at d=1 (open[2]=90)
    spec = [
        (0, [(100, 110)]),
        (1, [(100, 110)]),
        (2, [(100, 110)]),
        (3, [(200, 210)]),
    ]
    opens = [0.0, 100.0, 90.0, 110.0, 0.0]
    trades, audit = solve(spec, opens)
    assert audit["canonical_trades"] == 1
    t = trades[0]
    assert t["oracle_direction"] == LONG
    assert t["best_entry_decision_index"] == 1
    assert t["best_entry_price"] == pytest.approx(90.0)

    # SHORT: largest next open is at d=0 (open[1]=110) for B below
    spec2 = [
        (0, [(100, 110)]),
        (1, [(100, 110)]),
        (2, [(100, 110)]),
        (3, [(50, 60)]),
    ]
    opens2 = [0.0, 110.0, 90.0, 100.0, 0.0]
    trades2, _ = solve(spec2, opens2)
    assert len(trades2) == 1
    assert trades2[0]["oracle_direction"] == SHORT
    assert trades2[0]["best_entry_decision_index"] == 0
    assert trades2[0]["best_entry_price"] == pytest.approx(110.0)


# --------------------------------------------------------------------------- #
# 10. same target bar, TWO distinct ABOVE -> AMBIGUOUS (no intrabar ordering)
# --------------------------------------------------------------------------- #
def test_multi_upper_same_bar_ambiguous():
    spec = [
        (0, [(100, 110)]),                 # A
        (1, [(150, 160), (200, 210)]),     # two distinct ABOVE A
    ]
    opens = [0.0, 100.0, 0.0]
    trades, audit = solve(spec, opens)
    assert audit["canonical_trades"] == 0
    assert audit["ambiguous_target_bars"] == 1


# --------------------------------------------------------------------------- #
# 11. same target bar, TWO distinct BELOW -> AMBIGUOUS
# --------------------------------------------------------------------------- #
def test_multi_lower_same_bar_ambiguous():
    spec = [
        (0, [(200, 210)]),                 # A
        (1, [(100, 110), (50, 60)]),       # two distinct BELOW A
    ]
    opens = [0.0, 200.0, 0.0]
    trades, audit = solve(spec, opens)
    assert audit["canonical_trades"] == 0
    assert audit["ambiguous_target_bars"] == 1


# --------------------------------------------------------------------------- #
# 12. source A retouch + exactly one new upper B on same bar -> LONG
#     (A retouch is NOT part of distinct_locations)
# --------------------------------------------------------------------------- #
def test_retouch_plus_one_upper_is_long():
    spec = [
        (0, [(100, 110)]),                 # A
        (1, [(100, 110), (200, 210)]),     # A retouch + one upper B
    ]
    opens = [0.0, 100.0, 0.0]
    trades, audit = solve(spec, opens)
    assert audit["canonical_trades"] == 1
    assert trades[0]["oracle_direction"] == LONG
    assert trades[0]["target_price"] == pytest.approx(200)


# --------------------------------------------------------------------------- #
# 13. SHORT: Entry over whole leg, NOT A-touch next-open
# --------------------------------------------------------------------------- #
def test_short_entry_whole_leg_not_contact_only():
    spec = [
        (0, [(200, 210)]),     # A
        (5, [(100, 110)]),     # B below -> SHORT
    ]
    # opens[1..5] = 150,160,155,140,180 -> max fill is 180 (open[5], d=4)
    opens = [0.0, 150.0, 160.0, 155.0, 140.0, 180.0, 0.0]
    trades, audit = solve(spec, opens)
    assert audit["canonical_trades"] == 1
    t = trades[0]
    assert t["oracle_direction"] == SHORT
    # best Entry uses the highest whole-leg next-open (open[5]=180),
    # NOT the A-touch next-open (open[1]=150).
    assert t["best_entry_decision_index"] == 4
    assert t["best_entry_fill_index"] == 5
    assert t["best_entry_price"] == pytest.approx(180.0)
    assert t["best_entry_price"] != 150.0


# --------------------------------------------------------------------------- #
# 14. LONG: lowest valid next-open occurs mid-leg without A retouch
# --------------------------------------------------------------------------- #
def test_long_entry_whole_leg_mid_leg_low():
    spec = [
        (0, [(100, 110)]),     # A
        (5, [(200, 210)]),     # B above -> LONG
    ]
    # opens[1..5] = 150,120,160,155,170 -> min fill is 120 (open[2], d=1)
    opens = [0.0, 150.0, 120.0, 160.0, 155.0, 170.0, 0.0]
    trades, audit = solve(spec, opens)
    assert audit["canonical_trades"] == 1
    t = trades[0]
    assert t["oracle_direction"] == LONG
    assert t["best_entry_decision_index"] == 1
    assert t["best_entry_fill_index"] == 2
    assert t["best_entry_price"] == pytest.approx(120.0)


# --------------------------------------------------------------------------- #
# 15. Repeated A touches do NOT create multiple trades and do NOT constrain Entry
# --------------------------------------------------------------------------- #
def test_repeated_a_touches_single_trade_unconstrained_entry():
    spec = [
        (0, [(200, 210)]),
        (1, [(200, 210)]),     # A retouch
        (2, [(200, 210)]),     # A retouch
        (5, [(100, 110)]),     # B below -> SHORT
    ]
    # opens[1..5] = 150,155,160,145,175 -> max fill is 175 (open[5], d=4)
    opens = [0.0, 150.0, 155.0, 160.0, 145.0, 175.0, 0.0]
    trades, audit = solve(spec, opens)
    # exactly ONE trade, not one per A touch
    assert audit["canonical_trades"] == 1
    assert len(trades) == 1
    # Entry is the whole-leg max, NOT constrained to an A-touch bar
    assert trades[0]["best_entry_decision_index"] == 4
    assert trades[0]["best_entry_price"] == pytest.approx(175.0)


# --------------------------------------------------------------------------- #
# 16. Entry fix must not change A->B direction / target / exit
# --------------------------------------------------------------------------- #
def test_entry_fix_preserves_direction_target_exit():
    spec = [(0, [(100, 110)]), (5, [(200, 210)])]
    opens1 = [0.0, 150.0, 120.0, 160.0, 155.0, 170.0, 0.0]
    opens2 = [0.0, 140.0, 100.0, 130.0, 145.0, 160.0, 0.0]
    t1, _ = solve(spec, opens1)
    t2, _ = solve(spec, opens2)
    a, b = t1[0], t2[0]
    # A->B labels identical regardless of where Entry lands
    assert a["oracle_direction"] == b["oracle_direction"] == LONG
    assert a["target_price"] == b["target_price"] == pytest.approx(200)
    assert a["exit_fill_index"] == b["exit_fill_index"] == 5
    assert a["exit_price"] == b["exit_price"] == pytest.approx(200)
    # Entry genuinely changed
    assert a["best_entry_price"] != b["best_entry_price"]


# --------------------------------------------------------------------------- #
# 17. Same execution-boundary gate (decision d -> fill d+1) still applies
# --------------------------------------------------------------------------- #
def test_execution_boundary_gate_still_applied():
    spec = [
        (0, [(200, 210)]),     # A
        (4, [(100, 110)]),     # B below -> SHORT
    ]
    # segment changes between bar1 and bar2 -> d=1 fill is rejected
    segments = [1, 1, 2, 2, 2]
    tdays = [1, 1, 1, 1, 1]
    # opens[1..4] = 150,200,160,155 ; 200 (d=1) is highest but cross-boundary
    opens = [0.0, 150.0, 200.0, 160.0, 155.0, 0.0]
    trades, audit = solve_with(spec, opens, segments, tdays)
    assert audit["canonical_trades"] == 1
    t = trades[0]
    # best legal fill is 160 (open[3], d=2), NOT the cross-boundary 200
    assert t["best_entry_decision_index"] == 2
    assert t["best_entry_price"] == pytest.approx(160.0)
    assert t["best_entry_price"] != 200.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
