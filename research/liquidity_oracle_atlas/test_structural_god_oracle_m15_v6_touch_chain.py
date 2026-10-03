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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
