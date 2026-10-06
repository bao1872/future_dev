"""Golden tests for the V6.2 ENTRY-TIME TARGET label kernel.

Semantics under test:
    the target must come ONLY from geom_by_decision[d] where
    d == best_entry_decision_index. Future geometry may never retroactively
    become the target of an earlier entry, and the future first distinct
    touched location may only bound the entry-opportunity interval.
"""
import numpy as np
import pytest

from research.liquidity_oracle_atlas.structural_god_oracle_m15_v6_2_entry_target import (
    LONG,
    SHORT,
    TARGET_TOUCH,
    choose_entry_time_target,
    evaluate_entry_branch,
    first_target_touch,
    geometry_zones,
    locations_from_geometry_snapshot,
    solve_entry_time_target_oracle,
)
from research.liquidity_oracle_atlas.structural_god_oracle_m15_v6_touch_chain import (
    ZoneTouch,
    LocationTouch,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def loc(bottom, top):
    b, t = float(bottom), float(top)
    z = ZoneTouch(
        structure_id=f"SR|m15|{t:.6f}|{b:.6f}",
        family="SR",
        timeframe="m15",
        side=None,
        bottom=b,
        top=t,
        level=None,
    )
    return LocationTouch(bar=-1, bottom=b, top=t, members=[z])


def frames_from(spec, n):
    frames = [[] for _ in range(n)]
    for bar, items in spec:
        for (b, t) in items:
            frames[bar].append(loc(b, t))
    return frames


def mk_geom(channels=(), liq_up=(), liq_down=(), atr=np.nan):
    """geom[tf] = (channels, liq_up, liq_down, atr)."""
    return {
        "m15": (list(channels), list(liq_up), list(liq_down), atr),
    }


def liq(top, bottom, level=None, broken=False):
    return {
        "top": float(top),
        "bottom": float(bottom),
        "level": float(level if level is not None else bottom),
        "broken": broken,
    }


def run_solve(spec, n, opens, highs, lows, geoms,
              segments=None, tdays=None):
    frames = frames_from(spec, n)
    if segments is None:
        segments = np.ones(n, dtype=np.int64)
    if tdays is None:
        tdays = np.ones(n, dtype=np.int64)
    return solve_entry_time_target_oracle(
        frames=frames,
        opens=np.asarray(opens, dtype=float),
        highs=np.asarray(highs, dtype=float),
        lows=np.asarray(lows, dtype=float),
        segments=np.asarray(segments, dtype=np.int64),
        trading_days=np.asarray(tdays, dtype=np.int64),
        geoms=geoms,
    )


NO_HIT_HIGH = -1.0
NO_HIT_LOW = 1e9


# --------------------------------------------------------------------------- #
# T1 -- FUTURE SR FORBIDDEN
# --------------------------------------------------------------------------- #
def test_T1_future_sr_cannot_be_target():
    source = loc(100, 110)
    # SR [150,160] exists ONLY from geom[1] onwards
    geoms = [mk_geom(), mk_geom(channels=[(160.0, 150.0, 1.0)])]
    highs = np.array([0.0, 0.0, 200.0])
    lows = np.array([NO_HIT_LOW, NO_HIT_LOW, NO_HIT_LOW])

    r0 = evaluate_entry_branch(
        d=0, entry_price=120.0, direction=LONG, source=source,
        geoms=geoms, highs=highs, lows=lows,
    )
    assert not r0["ok"]
    assert r0["reason"] == "no_visible_target"

    # same SR is legal for a later decision where it IS visible
    r1 = evaluate_entry_branch(
        d=1, entry_price=125.0, direction=LONG, source=source,
        geoms=geoms, highs=highs, lows=lows,
    )
    assert r1["ok"]
    assert r1["target_price"] == pytest.approx(150.0)


# --------------------------------------------------------------------------- #
# T2 -- FUTURE LIQUIDITY FORBIDDEN
# --------------------------------------------------------------------------- #
def test_T2_future_liquidity_cannot_be_target():
    source = loc(100, 110)
    geoms = [mk_geom(), mk_geom(liq_up=[liq(160.0, 150.0)])]
    highs = np.array([0.0, 0.0, 200.0])
    lows = np.array([NO_HIT_LOW, NO_HIT_LOW, NO_HIT_LOW])

    r0 = evaluate_entry_branch(
        d=0, entry_price=120.0, direction=LONG, source=source,
        geoms=geoms, highs=highs, lows=lows,
    )
    assert not r0["ok"]
    assert r0["reason"] == "no_visible_target"

    r1 = evaluate_entry_branch(
        d=1, entry_price=125.0, direction=LONG, source=source,
        geoms=geoms, highs=highs, lows=lows,
    )
    assert r1["ok"]
    assert r1["target_price"] == pytest.approx(150.0)

    # broken liquidity is not tradeable structure
    geoms_broken = [mk_geom(liq_up=[liq(160.0, 150.0, broken=True)])]
    rb = evaluate_entry_branch(
        d=0, entry_price=120.0, direction=LONG, source=source,
        geoms=geoms_broken, highs=highs, lows=lows,
    )
    assert not rb["ok"]


# --------------------------------------------------------------------------- #
# T3 -- STRUCTURE FORMED BEFORE THE ACTUAL ENTRY IS LEGAL
# --------------------------------------------------------------------------- #
def test_T3_structure_formed_before_entry_is_legal():
    source = loc(100, 110)
    # structure appears at bar 1, decision is at bar 2 -> already visible
    geoms = [
        mk_geom(),
        mk_geom(channels=[(160.0, 150.0, 1.0)]),
        mk_geom(channels=[(160.0, 150.0, 1.0)]),
    ]
    highs = np.array([0.0, 0.0, 0.0, 200.0])
    lows = np.array([NO_HIT_LOW] * 4)

    r = evaluate_entry_branch(
        d=2, entry_price=130.0, direction=LONG, source=source,
        geoms=geoms, highs=highs, lows=lows,
    )
    assert r["ok"], "structure formed before the actual entry MUST be usable"
    assert r["target_price"] == pytest.approx(150.0)


# --------------------------------------------------------------------------- #
# T4 -- LONG NEAREST VISIBLE TARGET
# --------------------------------------------------------------------------- #
def test_T4_long_nearest_visible_target():
    geom = mk_geom(
        channels=[(160.0, 150.0, 1.0), (210.0, 190.0, 1.0)],
        liq_up=[liq(185.0, 175.0)],
    )
    locations = locations_from_geometry_snapshot(geom)
    t, price = choose_entry_time_target(locations, 120.0, LONG)
    assert price == pytest.approx(150.0)   # nearest first-touch edge
    assert t.bottom == pytest.approx(150.0)


# --------------------------------------------------------------------------- #
# T5 -- SHORT NEAREST VISIBLE TARGET
# --------------------------------------------------------------------------- #
def test_T5_short_nearest_visible_target():
    geom = mk_geom(
        channels=[(160.0, 150.0, 1.0), (110.0, 100.0, 1.0)],
        liq_down=[liq(95.0, 80.0)],
    )
    locations = locations_from_geometry_snapshot(geom)
    t, price = choose_entry_time_target(locations, 300.0, SHORT)
    assert price == pytest.approx(160.0)   # largest top below entry
    assert t.top == pytest.approx(160.0)


# --------------------------------------------------------------------------- #
# T6 -- ZONE EDGE (LONG lower edge / SHORT upper edge)
# --------------------------------------------------------------------------- #
def test_T6_zone_edge_semantics():
    geom = mk_geom(channels=[(200.0, 180.0, 1.0), (120.0, 100.0, 1.0)])
    locations = locations_from_geometry_snapshot(geom)

    _, lp = choose_entry_time_target(locations, 150.0, LONG)
    assert lp == pytest.approx(180.0)   # lower edge

    _, sp = choose_entry_time_target(locations, 150.0, SHORT)
    assert sp == pytest.approx(120.0)   # upper edge


# --------------------------------------------------------------------------- #
# T7 -- OVERLAPPING TARGET STRUCTURES MERGE
# --------------------------------------------------------------------------- #
def test_T7_overlapping_structures_one_location():
    geom = {
        "m15": ([(7814.0, 7808.0, 1.0)], [], [], np.nan),
        "h1": ([(7818.0, 7810.0, 1.0)], [], [], np.nan),
        "h4": ([], [liq(7813.0, 7812.0)], [], np.nan),
    }
    locations = locations_from_geometry_snapshot(geom)
    assert len(locations) == 1, "overlapping zones must be ONE price location"
    assert locations[0].bottom == pytest.approx(7808.0)
    assert locations[0].top == pytest.approx(7818.0)

    # and therefore exactly ONE target choice, not three
    _, price = choose_entry_time_target(locations, 7700.0, LONG)
    assert price == pytest.approx(7808.0)


# --------------------------------------------------------------------------- #
# T8 -- SOURCE / SELF TARGET EXCLUDED
# --------------------------------------------------------------------------- #
def test_T8_source_self_target_excluded():
    source = loc(100, 110)
    # the ONLY visible structure is frozen source A itself
    geom = mk_geom(channels=[(110.0, 100.0, 1.0)])
    locations = locations_from_geometry_snapshot(geom)
    # raw selection would allow it ...
    assert choose_entry_time_target(locations, 90.0, LONG) is not None
    # ... but evaluate_entry_branch excludes it
    r = evaluate_entry_branch(
        d=0, entry_price=90.0, direction=LONG, source=source,
        geoms=[geom], highs=np.array([500.0]), lows=np.array([NO_HIT_LOW]),
    )
    assert not r["ok"]
    assert r["reason"] == "no_visible_target"


# --------------------------------------------------------------------------- #
# T9 -- target_snapshot_index == best_entry_decision_index
# --------------------------------------------------------------------------- #
def test_T9_snapshot_equals_entry_decision():
    n = 8
    spec = [(0, [(100, 110)]), (5, [(400, 410)])]
    opens = [100, 105, 120, 130, 140, 145, 100, 100]
    highs = np.full(n, NO_HIT_HIGH)
    lows = np.full(n, NO_HIT_LOW)
    lows[5] = 50.0
    geoms = [
        mk_geom(channels=[(160.0, 150.0, 1.0), (60.0, 50.0, 1.0)])
        for _ in range(n)
    ]
    trades, audit = run_solve(spec, n, opens, highs, lows, geoms)
    assert audit["canonical_trades"] == 1
    t = trades[0]
    assert t["target_snapshot_index"] == t["best_entry_decision_index"]
    assert audit["target_snapshot_mismatch_count"] == 0


# --------------------------------------------------------------------------- #
# T10 -- FROZEN AFTER ENTRY
# --------------------------------------------------------------------------- #
def test_T10_target_frozen_after_entry():
    source = loc(100, 110)
    geoms = [
        mk_geom(channels=[(160.0, 150.0, 1.0)]),
        mk_geom(channels=[(160.0, 150.0, 1.0)]),
    ]
    highs = np.array([0.0, 200.0])
    lows = np.array([NO_HIT_LOW, NO_HIT_LOW])

    r_before = evaluate_entry_branch(
        d=0, entry_price=120.0, direction=LONG, source=source,
        geoms=geoms, highs=highs, lows=lows,
    )
    # geometry AFTER the decision changes / new structures appear
    geoms[1] = mk_geom(channels=[(400.0, 390.0, 1.0)])
    r_after = evaluate_entry_branch(
        d=0, entry_price=120.0, direction=LONG, source=source,
        geoms=geoms, highs=highs, lows=lows,
    )
    assert r_before["ok"] and r_after["ok"]
    assert r_after["target_price"] == r_before["target_price"]


# --------------------------------------------------------------------------- #
# T11 -- SAME-BAR TARGET TOUCH IS LEGAL
# --------------------------------------------------------------------------- #
def test_T11_same_bar_target_touch():
    source = loc(100, 110)
    geoms = [mk_geom(channels=[(160.0, 150.0, 1.0)])]
    # fill at bar 1, target 150 touched in bar 1 itself
    highs = np.array([0.0, 155.0])
    lows = np.array([NO_HIT_LOW, NO_HIT_LOW])
    r = evaluate_entry_branch(
        d=0, entry_price=120.0, direction=LONG, source=source,
        geoms=geoms, highs=highs, lows=lows,
    )
    assert r["ok"]
    assert r["exit_bar"] == 1          # same bar as the fill
    assert r["utility"] == pytest.approx(30.0)


# --------------------------------------------------------------------------- #
# T12 -- TARGET NEVER TOUCHED -> not a completed canonical trade
# --------------------------------------------------------------------------- #
def test_T12_target_never_touched():
    n = 4
    spec = [(0, [(100, 110)]), (2, [(400, 410)])]
    opens = [100, 120, 130, 100]
    highs = np.full(n, NO_HIT_HIGH)     # 150 never reached
    lows = np.full(n, NO_HIT_LOW)
    geoms = [mk_geom(channels=[(160.0, 150.0, 1.0)]) for _ in range(n)]
    trades, audit = run_solve(spec, n, opens, highs, lows, geoms)
    assert audit["canonical_trades"] == 0
    assert audit["target_never_touched_count"] > 0


# --------------------------------------------------------------------------- #
# T13 -- LONG / SHORT SYMMETRY
# --------------------------------------------------------------------------- #
def test_T13_long_short_symmetry():
    geom = mk_geom(channels=[(160.0, 150.0, 1.0), (60.0, 50.0, 1.0)])
    locations = locations_from_geometry_snapshot(geom)
    entry = 105.0
    c = entry

    _, tl = choose_entry_time_target(locations, entry, LONG)
    _, ts = choose_entry_time_target(locations, entry, SHORT)

    # mirror every price about c
    mirrored = []
    for L in locations:
        mirrored.append(type(L)(
            bar=-1,
            bottom=2 * c - L.top,
            top=2 * c - L.bottom,
            members=list(L.members),
        ))
    _, tl_m = choose_entry_time_target(mirrored, 2 * c - entry, LONG)
    _, ts_m = choose_entry_time_target(mirrored, 2 * c - entry, SHORT)

    assert tl_m == pytest.approx(2 * c - ts)
    assert ts_m == pytest.approx(2 * c - tl)


# --------------------------------------------------------------------------- #
# T14 -- ENTRY EXECUTION BOUNDARY
# --------------------------------------------------------------------------- #
def test_T14_entry_execution_boundary():
    n = 5
    spec = [(0, [(100, 110)]), (2, [(400, 410)])]
    opens = [100, 120, 130, 100, 100]
    highs = np.full(n, NO_HIT_HIGH)
    lows = np.full(n, NO_HIT_LOW)
    lows[2] = 50.0
    geoms = [
        mk_geom(channels=[(160.0, 150.0, 1.0), (60.0, 50.0, 1.0)])
        for _ in range(n)
    ]
    # d=0 -> fill 1 crosses a segment boundary; only d=1 is executable
    segments = np.array([1, 2, 2, 2, 2], dtype=np.int64)
    trades, audit = run_solve(spec, n, opens, highs, lows, geoms,
                              segments=segments)
    assert audit["entry_boundary_rejection_count"] >= 1
    assert audit["canonical_trades"] == 1
    assert trades[0]["best_entry_decision_index"] == 1


# --------------------------------------------------------------------------- #
# T15 -- OVERLAP IS ALLOWED (portfolio non-overlap constraint removed)
# --------------------------------------------------------------------------- #
def test_T15_overlap_allowed():
    n = 14
    spec = [
        (0, [(100, 110)]),     # source A
        (4, [(400, 410)]),     # source B (first distinct touch after A)
    ]
    opens = [100, 100, 110, 120, 140, 430, 420, 410, 405, 400,
             400, 400, 400, 400]
    highs = np.full(n, NO_HIT_HIGH)
    lows = np.full(n, NO_HIT_LOW)
    lows[9] = 300.0    # B SHORT target 310 first touched at bar 9
    lows[12] = 50.0    # A SHORT target 60 first touched at bar 12

    geoms = []
    for _ in range(n):
        geoms.append(mk_geom(channels=[
            (160.0, 150.0, 1.0),
            (60.0, 50.0, 1.0),
            (510.0, 500.0, 1.0),
            (310.0, 300.0, 1.0),
        ]))

    trades, audit = run_solve(spec, n, opens, highs, lows, geoms)
    # A -> label [entry bar 4, exit bar 12]; B -> label [entry bar 5, exit bar 9].
    # B's label lives INSIDE A's window, so the two overlap in calendar time.
    assert audit["canonical_trades"] == 2
    # With the portfolio non-overlap constraint REMOVED, labels from different
    # sources are allowed (and here DO) overlap in calendar time.
    assert audit["labels_with_overlapping_exit"] >= 1

    # The causal invariant is unchanged: target frozen at the entry decision.
    assert audit["target_snapshot_mismatch_count"] == 0

    for t in trades:
        assert t["exit_reason"] == TARGET_TOUCH
        assert t["utility"] > 0


# --------------------------------------------------------------------------- #
# T16 -- the future terminal location must NOT become the target
# --------------------------------------------------------------------------- #
def test_T16_terminal_location_is_not_the_target():
    n = 6
    spec = [
        (0, [(100, 110)]),
        (3, [(300, 310)]),     # terminal of the entry-opportunity interval
    ]
    opens = [100, 120, 125, 130, 100, 100]
    highs = np.full(n, NO_HIT_HIGH)
    lows = np.full(n, NO_HIT_LOW)
    lows[3] = 50.0
    # ONLY a visible target at 150/60 exists; 300/310 is never in geometry
    geoms = [
        mk_geom(channels=[(160.0, 150.0, 1.0), (60.0, 50.0, 1.0)])
        for _ in range(n)
    ]
    trades, audit = run_solve(spec, n, opens, highs, lows, geoms)
    assert audit["canonical_trades"] == 1
    t = trades[0]
    assert t["oracle_direction"] == SHORT
    assert t["target_price"] == pytest.approx(60.0)
    assert t["target_price"] != pytest.approx(300.0)
    assert t["target_price"] != pytest.approx(310.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
