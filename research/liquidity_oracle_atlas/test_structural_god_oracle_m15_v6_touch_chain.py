"""T0 unit tests for the V6 touch-chain God oracle (synthetic inputs).

These tests need NO market data / environment: they build per-bar touch
lists directly and exercise the frozen model primitives.

T0 list (from the freeze spec):
  - A A A consecutive -> one episode.
  - A separated then A again -> two episodes.
  - A -> upper B => LONG.
  - A -> lower B => SHORT.
  - overlapping A/B zones => ambiguous.
  - same future bar touches upper B and lower C => ambiguous target group.
  - LONG best entry = min next-open among A contact bars.
  - SHORT best entry = max next-open among A contact bars.
  - entry fill crossing execution boundary => rejected.
  - A->B->C produces exactly two transitions/trades when both have legal entries.
"""

import numpy as np
import pytest

from research.liquidity_oracle_atlas.structural_god_oracle_m15_v6_touch_chain import (
    Touch,
    Episode,
    TouchGroup,
    best_entry,
    build_episodes,
    build_touch_groups,
    solve_touch_chain,
    _find_next_distinct_group,
    _classify_transition,
    _choose_anchor,
    extract_touch_records,
    run_touch_chain_oracle,
    print_screenshot_audit,
)


def mk(bar, sid, zb, zt, fam="SR", side=None):
    return Touch(
        bar=bar, structure_id=sid, family=fam, tf="m15", side=side,
        zone_bottom=zb, zone_top=zt, level=None, strength=None,
    )


def common_arrays(n):
    segments = np.zeros(n, dtype=np.int64)
    tds = np.zeros(n, dtype=np.int64)
    return segments, tds


# --------------------------------------------------------------------------- #
# Episodes
# --------------------------------------------------------------------------- #
def test_AAA_consecutive_one_episode():
    touch_by_bar = [
        [mk(0, "A", 100, 110)],
        [mk(1, "A", 100, 110)],
        [mk(2, "A", 100, 110)],
    ]
    eps = build_episodes(touch_by_bar)
    assert len(eps) == 1
    assert eps[0].touched_bars == [0, 1, 2]
    assert eps[0].start_bar == 0 and eps[0].end_bar == 2


def test_A_separated_two_episodes():
    touch_by_bar = [
        [mk(0, "A", 100, 110)],
        [mk(1, "A", 100, 110)],
        [mk(2, "A", 100, 110)],
        [],
        [],
        [mk(5, "A", 100, 110)],
    ]
    eps = build_episodes(touch_by_bar)
    assert len(eps) == 2
    assert eps[0].touched_bars == [0, 1, 2]
    assert eps[1].touched_bars == [5]
    assert eps[1].start_bar == 5


def test_extract_builds_stable_id_and_zone():
    entry_matches = [
        [{"tf": "m15", "family": "SR", "side": None, "top": 110.0,
          "bottom": 100.0, "level": None, "strength": 1.0}],
        [],
    ]
    tbb = extract_touch_records(entry_matches)
    assert len(tbb[0]) == 1
    r = tbb[0][0]
    assert r.structure_id == "SR|m15|110.000000|100.000000"
    assert r.zone_bottom == 100.0 and r.zone_top == 110.0


# --------------------------------------------------------------------------- #
# Direction (Step 6)
# --------------------------------------------------------------------------- #
def test_A_upper_B_LONG():
    Azone = {"A": (100.0, 110.0)}
    Bzone = {"B": (200.0, 210.0)}
    status, direction, tid, exit_price = _classify_transition(Azone, Bzone)
    assert status == "CANONICAL"
    assert direction == "LONG"
    assert tid == "B"
    assert exit_price == 200.0  # B.zone_bottom


def test_A_lower_B_SHORT():
    Azone = {"A": (100.0, 110.0)}
    Bzone = {"B": (50.0, 60.0)}
    status, direction, tid, exit_price = _classify_transition(Azone, Bzone)
    assert status == "CANONICAL"
    assert direction == "SHORT"
    assert tid == "B"
    assert exit_price == 60.0  # B.zone_top


def test_overlapping_zones_ambiguous():
    Azone = {"A": (100.0, 110.0)}
    Bzone = {"B": (104.0, 114.0)}
    status, direction, tid, exit_price = _classify_transition(Azone, Bzone)
    assert status == "OVERLAP"
    assert direction is None


def test_same_bar_upper_and_lower_AMBIGUOUS_SAME_BAR():
    Azone = {"A": (100.0, 110.0)}
    Bzone = {"B": (200.0, 210.0), "C": (50.0, 60.0)}
    status, direction, tid, exit_price = _classify_transition(Azone, Bzone)
    assert status == "AMBIGUOUS_SAME_BAR"
    assert direction is None


def test_classify_requires_distinct_B():
    """find_next_distinct_group owns same-location skipping; _classify_transition
    must never be called with a same-only B (it is a programming error)."""
    Azone = {"A": (100.0, 110.0)}
    Bzone = {"A": (100.0, 110.0)}  # identical revisit
    with pytest.raises(AssertionError):
        _classify_transition(Azone, Bzone)


# --------------------------------------------------------------------------- #
# Entry (Step 9)
# --------------------------------------------------------------------------- #
def test_LONG_best_entry_is_min_next_open():
    n = 6
    segments, tds = common_arrays(n)
    opens = np.array([0.0, 102.0, 101.0, 103.0, 0.0, 0.0])
    contact = [0, 1, 2]
    exit_price = 200.0
    exit_bar = 5
    res = best_entry(contact, opens, segments, tds, exit_price, "LONG", n, exit_bar)
    assert res is not None
    d_star, entry_price = res
    # min next-open is open[2]=101 at decision bar d=1
    assert d_star == 1
    assert entry_price == pytest.approx(101.0)


def test_SHORT_best_entry_is_max_next_open():
    n = 6
    segments, tds = common_arrays(n)
    opens = np.array([0.0, 198.0, 199.0, 197.0, 0.0, 0.0])
    contact = [0, 1, 2]
    exit_price = 50.0
    exit_bar = 5
    res = best_entry(contact, opens, segments, tds, exit_price, "SHORT", n, exit_bar)
    assert res is not None
    d_star, entry_price = res
    # max next-open is open[2]=199 at decision bar d=1
    assert d_star == 1
    assert entry_price == pytest.approx(199.0)


def test_entry_fill_crossing_execution_boundary_rejected():
    n = 3
    segments = np.array([0, 1, 0], dtype=np.int64)  # segment[1] != segment[0]
    tds = np.zeros(n, dtype=np.int64)
    opens = np.array([0.0, 102.0, 0.0])
    contact = [0]  # decision bar 0 -> fill open[1]; crosses unit boundary
    res = best_entry(contact, opens, segments, tds, 200.0, "LONG", n, exit_bar=2)
    assert res is None


def test_entry_after_exit_bar_rejected():
    n = 6
    segments, tds = common_arrays(n)
    opens = np.array([0.0, 102.0, 101.0, 103.0, 0.0, 0.0])
    contact = [0, 1, 2]
    res = best_entry(contact, opens, segments, tds, 200.0, "LONG", n, exit_bar=0)
    # every contact bar >= exit_bar (0) -> no legal entry
    assert res is None


# --------------------------------------------------------------------------- #
# Full chain (Step 4-10)
# --------------------------------------------------------------------------- #
def _build_chain_groups():
    """A(100,110)@0-2, B(200,210)@3-5, C(150,160)@6-8."""
    touch_by_bar = [
        [mk(0, "A", 100, 110)],
        [mk(1, "A", 100, 110)],
        [mk(2, "A", 100, 110)],
        [mk(3, "B", 200, 210)],
        [mk(4, "B", 200, 210)],
        [mk(5, "B", 200, 210)],
        [mk(6, "C", 150, 160)],
        [mk(7, "C", 150, 160)],
        [mk(8, "C", 150, 160)],
    ]
    eps = build_episodes(touch_by_bar)
    groups = build_touch_groups(eps)
    return groups


def test_A_B_C_two_transactions_two_trades():
    groups = _build_chain_groups()
    assert len(groups) == 3
    n = 10
    segments, tds = common_arrays(n)
    # opens: A->B LONG needs open[d+1] < 200; B->C SHORT needs open[d+1] > 160
    opens = np.array([0.0, 150.0, 150.0, 150.0, 200.0, 200.0, 200.0, 0.0, 0.0, 0.0])
    trades, audit = solve_touch_chain(
        groups, opens, segments, tds, n,
        total_true_touch_records=9, total_touch_episodes=3, total_touch_groups=3,
    )
    assert len(trades) == 2
    assert audit["source_groups_with_future"] == 2
    assert audit["canonical_trades"] == 2
    assert audit["no_later_distinct_target_transitions"] == 0
    assert audit["same_location_groups_skipped"] == 0
    # reconciliation
    assert audit["source_groups_with_future"] == (
        audit["canonical_trades"]
        + audit["ambiguous_same_bar_target_groups"]
        + audit["overlapping_zone_transitions"]
        + audit["no_legal_entry_transitions"]
        + audit["no_later_distinct_target_transitions"]
    )
    # first trade A->B is LONG, second B->C is SHORT
    assert trades[0]["oracle_direction"] == "LONG"
    assert trades[1]["oracle_direction"] == "SHORT"
    # no overlap on the emitted stream
    assert trades[0]["exit_fill_index"] <= trades[1]["best_entry_decision_index"]


def test_reconciliation_overlap_and_ambiguous_buckets():
    """A->B overlap, B->C ambiguous same bar -> no canonical trades but
    reconciliation still holds exactly."""
    touch_by_bar = [[] for _ in range(7)]
    touch_by_bar[0] = [mk(0, "A", 100, 110)]
    touch_by_bar[3] = [mk(3, "B", 104, 114)]   # overlap with A
    touch_by_bar[6] = [mk(6, "C", 200, 210), mk(6, "D", 50, 60)]  # same bar -> ambiguous
    # Build groups: A@0, B@3, (C,D)@6
    eps = build_episodes(touch_by_bar)
    groups = build_touch_groups(eps)
    n = 8
    segments, tds = common_arrays(n)
    opens = np.full(n, 150.0)
    trades, audit = solve_touch_chain(
        groups, opens, segments, tds, n,
        total_true_touch_records=4, total_touch_episodes=4, total_touch_groups=3,
    )
    # A->B overlap, B->(C,D) ambiguous same bar
    assert len(trades) == 0
    assert audit["overlapping_zone_transitions"] == 1
    assert audit["ambiguous_same_bar_target_groups"] == 1
    assert audit["source_groups_with_future"] == 2
    assert audit["no_later_distinct_target_transitions"] == 0
    assert audit["source_groups_with_future"] == (
        audit["canonical_trades"]
        + audit["ambiguous_same_bar_target_groups"]
        + audit["overlapping_zone_transitions"]
        + audit["no_legal_entry_transitions"]
        + audit["no_later_distinct_target_transitions"]
    )


# --------------------------------------------------------------------------- #
# T1 -- real AG slice (screenshot region, old wrong SHORT disappears)
# --------------------------------------------------------------------------- #
def test_T1_real_AG_support_region_no_old_artifact_short():
    """Run a real AG slice and assert the frozen touch-chain model holds.

    Verifies:
      * hard reconciliation holds on real data,
      * the model emits trades (far richer than the old 94-label stream),
      * BOTH directions occur (it is not degenerate),
      * the old "10-day SHORT-to-7655" artifact fingerprint is GONE:
        no trade is SHORT with exit <= 7660 AND a > 100-bar duration
        (the old V4/PREEMPT label had entry@348, exit@570 -> dur 222),
      * every support-region (anchor mid ~7690) trade is a genuine
        next-distinct-touch label (short duration), never a long wait.
    """
    res = run_touch_chain_oracle("AG", max_bars=2000)
    a = res["audit"]
    trades = res["trades"]

    # 1) reconciliation
    assert a["source_groups_with_future"] == (
        a["canonical_trades"]
        + a["ambiguous_same_bar_target_groups"]
        + a["overlapping_zone_transitions"]
        + a["no_legal_entry_transitions"]
        + a["no_later_distinct_target_transitions"]
    )

    # 2) model emits a real trade stream
    assert a["canonical_trades"] > 0
    assert a["total_touch_groups"] > 0

    # 3) both directions present (non-degenerate)
    dirs = {t["oracle_direction"] for t in trades}
    assert "LONG" in dirs and "SHORT" in dirs

    # 4) old-artifact SHORT fingerprint is gone
    old_fingerprint = [
        t for t in trades
        if t["oracle_direction"] == "SHORT"
        and t["exit_price"] <= 7660.0
        and (t["exit_fill_index"] - t["best_entry_decision_index"]) > 100
    ]
    assert not old_fingerprint, (
        f"old 10-day SHORT artifact reappeared: {old_fingerprint[:2]}"
    )

    # 5) every support-region trade is a genuine next-touch label (short dur)
    for t in trades:
        mid = (t["zone_bottom"] + t["zone_top"]) / 2.0
        if 7660.0 <= mid <= 7720.0:
            dur = t["exit_fill_index"] - t["best_entry_decision_index"]
            assert dur <= 100, (
                f"support-region trade has implausible duration {dur}; "
                f"old artifact pattern revived: {t}"
            )

    # 6) manual-review aid (does not affect pass/fail)
    print("\n[T1 audit]", a)
    print_screenshot_audit(res, around_price=7690.0, band=30.0)


# --------------------------------------------------------------------------- #
# T0b -- V6 core corrections (next-distinct search, near-edge, anchor ownership)
# --------------------------------------------------------------------------- #
def mkgrp(start, sid, zb, zt, bars):
    ep = Episode(sid, list(bars), min(bars), max(bars), zb, zt, "SR", "m15", None)
    return TouchGroup(start_bar=start, episodes=[ep], structures={sid: ep})


def test_A_A_B_resolves_not_lost():
    """Repeated same-location touches of A are searched THROUGH; the source A
    still resolves to the next DISTINCT B. No continuation loss."""
    groups = [
        mkgrp(0, "A", 100, 110, [0, 1]),
        mkgrp(2, "A", 100, 110, [2, 3]),
        mkgrp(4, "B", 200, 210, [4, 5]),
    ]
    n = 6
    segments, tds = common_arrays(n)
    opens = np.full(n, 150.0)  # open[d+1]=150 < 200 -> legal LONG entry
    trades, audit = solve_touch_chain(
        groups, opens, segments, tds, n,
        total_true_touch_records=6, total_touch_episodes=3, total_touch_groups=3,
    )
    # both A groups resolve to B (not lost)
    assert len(trades) == 2
    assert all(t["oracle_direction"] == "LONG" for t in trades)
    assert all(t["target_structure_id"] == "B" for t in trades)
    assert audit["no_later_distinct_target_transitions"] == 0
    assert audit["same_location_groups_skipped"] >= 1


def test_A_A_A_B_three_resolutions():
    """A -> A -> A -> B yields THREE trades (each A occurrence -> same B)."""
    groups = [
        mkgrp(0, "A", 100, 110, [0]),
        mkgrp(1, "A", 100, 110, [1]),
        mkgrp(2, "A", 100, 110, [2]),
        mkgrp(3, "B", 200, 210, [3]),
    ]
    n = 4
    segments, tds = common_arrays(n)
    opens = np.full(n, 150.0)
    trades, audit = solve_touch_chain(
        groups, opens, segments, tds, n,
        total_true_touch_records=4, total_touch_episodes=4, total_touch_groups=4,
    )
    assert len(trades) == 3
    assert audit["canonical_trades"] == 3
    # skipped = 2 (from k=0) + 1 (from k=1) + 0 (from k=2)
    assert audit["same_location_groups_skipped"] == 3


def test_no_later_distinct_B():
    """When no later distinct structure exists, the source resolves to the
    no_later_distinct bucket (diagnostic, not a lost label)."""
    groups = [
        mkgrp(0, "A", 100, 110, [0]),
        mkgrp(1, "A", 100, 110, [1]),
    ]
    n = 2
    segments, tds = common_arrays(n)
    opens = np.full(n, 150.0)
    trades, audit = solve_touch_chain(
        groups, opens, segments, tds, n,
        total_true_touch_records=2, total_touch_episodes=2, total_touch_groups=2,
    )
    assert len(trades) == 0
    assert audit["no_later_distinct_target_transitions"] == 1
    assert audit["source_groups_with_future"] == 1


def test_two_upper_targets_pick_lowest_bottom():
    """LONG near edge = lowest zone_bottom. Exposing case: the zone with the
    smaller TOP is NOT the one with the smaller BOTTOM."""
    Azone = {"A": (100.0, 110.0)}
    Bzone = {"B1": (200.0, 210.0), "B2": (150.0, 300.0)}
    status, direction, tid, exit_price = _classify_transition(Azone, Bzone)
    assert status == "CANONICAL"
    assert direction == "LONG"
    assert tid == "B2"
    assert exit_price == pytest.approx(150.0)


def test_two_lower_targets_pick_highest_top():
    """SHORT near edge = highest zone_top. Exposing case: the zone with the
    larger BOTTOM is NOT the one with the larger TOP."""
    Azone = {"A": (100.0, 110.0)}
    Bzone = {"B1": (90.0, 95.0), "B2": (20.0, 98.0)}  # both below A
    status, direction, tid, exit_price = _classify_transition(Azone, Bzone)
    assert status == "CANONICAL"
    assert direction == "SHORT"
    assert tid == "B2"
    assert exit_price == pytest.approx(98.0)


def test_multi_A_nearest_anchor():
    """Same A group with two structures; recorded Candidate A must be the one
    nearest the target (A2 below B)."""
    A1 = Episode("A1", [0, 1], 0, 1, 100, 110, "SR", "m15", None)
    A2 = Episode("A2", [0, 1], 0, 1, 120, 130, "SR", "m15", None)
    group = TouchGroup(
        start_bar=0, episodes=[A1, A2], structures={"A1": A1, "A2": A2},
    )
    groups = [group, mkgrp(2, "B", 200, 210, [2])]
    n = 3
    segments, tds = common_arrays(n)
    opens = np.array([0.0, 150.0, 200.0])
    trades, audit = solve_touch_chain(
        groups, opens, segments, tds, n,
        total_true_touch_records=3, total_touch_episodes=3, total_touch_groups=2,
    )
    assert len(trades) == 1
    assert trades[0]["structure_id"] == "A2"
    assert trades[0]["oracle_direction"] == "LONG"


def test_entry_only_from_anchor_contact_bars():
    """Best Entry must use ONLY the chosen anchor's contact bars, never the
    union of all A structures. An attractive open on an A1-only bar must be
    ineligible."""
    A1 = Episode("A1", [0, 5], 0, 5, 100, 110, "SR", "m15", None)
    A2 = Episode("A2", [0, 1], 0, 1, 120, 130, "SR", "m15", None)
    group = TouchGroup(
        start_bar=0, episodes=[A1, A2], structures={"A1": A1, "A2": A2},
    )
    groups = [group, mkgrp(2, "B", 200, 210, [2])]
    n = 7
    segments, tds = common_arrays(n)
    # attractive open only on A1-only bar 5 -> fill open[6]=50
    opens = np.array([0.0, 150.0, 200.0, 0.0, 0.0, 0.0, 50.0])
    trades, audit = solve_touch_chain(
        groups, opens, segments, tds, n,
        total_true_touch_records=4, total_touch_episodes=4, total_touch_groups=2,
    )
    assert len(trades) == 1
    # anchor A2 -> only A2 contact bars [0,1] eligible -> entry decision 0,
    # NOT 5 (which would require A1's union of contact bars).
    assert trades[0]["structure_id"] == "A2"
    assert trades[0]["best_entry_decision_index"] == 0


def test_hard_reconciliation_after_skips():
    """After skipping same-location groups, the hard reconciliation still holds
    exactly (no unexplained disappearance)."""
    groups = [
        mkgrp(0, "A", 100, 110, [0]),
        mkgrp(1, "A", 100, 110, [1]),
        mkgrp(2, "A", 100, 110, [2]),
        mkgrp(3, "B", 200, 210, [3]),
    ]
    n = 4
    segments, tds = common_arrays(n)
    opens = np.full(n, 150.0)
    trades, audit = solve_touch_chain(
        groups, opens, segments, tds, n,
        total_true_touch_records=4, total_touch_episodes=4, total_touch_groups=4,
    )
    assert audit["source_groups_with_future"] == 3
    assert audit["source_groups_with_future"] == (
        audit["canonical_trades"]
        + audit["ambiguous_same_bar_target_groups"]
        + audit["overlapping_zone_transitions"]
        + audit["no_legal_entry_transitions"]
        + audit["no_later_distinct_target_transitions"]
    )
    assert audit["canonical_trades"] == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
