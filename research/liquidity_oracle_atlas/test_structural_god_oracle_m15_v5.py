"""test_structural_god_oracle_m15_v5
====================================

V5 oracle tests. Two layers exactly as the spec demands:

  T0 -- DIRECTION SEMANTICS (the atomic trade is correct first)
    1..10  competing-first-passage + per-decision target freeze + no primary gating
    11..14 global non-overlap DP scheduler

  T1 -- REAL SCREENSHOT REGION (the previously-wrong SHORT=7655 case)
    15  print + assert the diagnosed structure now resolves as LONG
    16  prove the old SHORT=7655 is rejected (upper target reached first)
    17  compare V4 (old) first-20 vs V5 (new) first-20

No artifact is regenerated; Viewer is untouched.
"""

import numpy as np
import pytest

from research.liquidity_oracle_atlas.structural_god_oracle_m15_v5 import (
    TradeOption,
    _build_target_tree,
    _first_target_touch,
    _load_inputs,
    _make_mv_helper,
    enumerate_trade_options,
    evaluate_candidate_at_decision,
    nearest_upper_and_lower_target,
    run_god_oracle_v5,
    solve_nonoverlap_god_dp,
)
from research.liquidity_oracle_atlas.structural_god_oracle_m15_v4 import (
    run_god_oracle_v4,
)

DIAG_SID = "SR|m15|0|346|7705.0|7694.0|75.0"
PREEMPT_SID = "SR|m15|0|346|7691.0|7681.0|72.0"

EPS = 1e-6


# --------------------------------------------------------------------------- #
# Synthetic helpers
# --------------------------------------------------------------------------- #
def _sr_geom(channels, liq_up=(), liq_down=()):
    return {"m15": (list(channels), list(liq_up), list(liq_down), 1.0)}


def _make_mv(opens, highs, lows, closes, times, td, seg):
    return _make_mv_helper(
        np.asarray(opens, float),
        np.asarray(highs, float),
        np.asarray(lows, float),
        np.asarray(closes, float),
        np.asarray(times),
        np.asarray(td, np.int64),
        np.asarray(seg, np.int64),
    )


def _scenario(decision=10, highs_extra=None, lows_extra=None, opens_extra=None):
    """Flat-ish 30-bar world; one upper SR (110) and one lower SR (90)."""
    n = 30
    opens = [100.0] * n
    highs = [100.0] * n
    lows = [100.0] * n
    closes = [100.0] * n
    for i, v in (highs_extra or []):
        highs[i] = v
    for i, v in (lows_extra or []):
        lows[i] = v
    for i, v in (opens_extra or []):
        opens[i] = v
    times = list(range(n))
    td = [0] * n
    seg = [0] * n
    mv = _make_mv(opens, highs, lows, closes, times, td, seg)
    g = _sr_geom([(115.0, 110.0, 1.0), (90.0, 85.0, 1.0)])
    geom = [g] * n
    atr = [1.0] * n
    tree = _build_target_tree(mv)
    cand_sid = "SR|m15|0|10|130.0|120.0|1.0"  # above close; not self vs 110/90
    tr = evaluate_candidate_at_decision(
        d=decision,
        candidate_sid=cand_sid,
        candidate_type="SR",
        zone_bottom=120.0,
        zone_top=130.0,
        geom_prev=geom[decision - 1],
        sr_first_seen={},
        opens=np.asarray(opens),
        highs=np.asarray(highs),
        lows=np.asarray(lows),
        closes=np.asarray(closes),
        atr_series=np.asarray(atr),
        trading_day=np.asarray(td),
        segment=np.asarray(seg),
        mv=mv,
        target_tree=tree,
        event_id=-1,
    )
    return tr


# --------------------------------------------------------------------------- #
# T0 -- DIRECTION SEMANTICS
# --------------------------------------------------------------------------- #
def test_t0_1_upper_first_then_long():
    tr = _scenario(highs_extra=[(11, 120.0)])  # upper touched @11, lower never
    assert tr is not None
    assert tr.direction == "LONG"
    assert tr.exit == 11
    assert abs(tr.utility_points - 10.0) < EPS


def test_t0_2_lower_first_then_short():
    tr = _scenario(lows_extra=[(11, 80.0)])  # lower touched @11, upper never
    assert tr is not None
    assert tr.direction == "SHORT"
    assert tr.exit == 11
    assert abs(tr.utility_points - 10.0) < EPS


def test_t0_3_both_touched_first_wins_not_profit():
    # upper @11, lower @15 -> first touch (upper) wins even though profits equal
    tr = _scenario(highs_extra=[(11, 120.0)], lows_extra=[(15, 80.0)])
    assert tr is not None
    assert tr.direction == "LONG"
    assert tr.exit == 11  # upper first, not the later lower


def test_t0_4_same_bar_both_touched_ambiguous():
    tr = _scenario(highs_extra=[(11, 120.0)], lows_extra=[(11, 80.0)])
    assert tr is None  # AMBIGUOUS -> no canonical label


def test_t0_5_upper_touched_on_decision_bar_not_future():
    g = _sr_geom([(115.0, 110.0, 1.0), (90.0, 85.0, 1.0)])
    up, lo = nearest_upper_and_lower_target(
        g, close_d=100.0, high_d=115.0, low_d=100.0,
        seg=0, d=10, sr_first_seen={}, candidate_sid="X",
    )
    assert up is None  # already touched -> not a future upper target
    assert lo is not None and lo.near_edge == 90.0
    # and the direction correctly falls to SHORT via the lower target
    tr = _scenario(highs_extra=[(10, 115.0)], lows_extra=[(11, 80.0)])
    assert tr is not None and tr.direction == "SHORT"


def test_t0_6_lower_touched_on_decision_bar_not_future():
    g = _sr_geom([(115.0, 110.0, 1.0), (90.0, 85.0, 1.0)])
    up, lo = nearest_upper_and_lower_target(
        g, close_d=100.0, high_d=100.0, low_d=80.0,
        seg=0, d=10, sr_first_seen={}, candidate_sid="X",
    )
    assert lo is None  # already touched -> not a future lower target
    assert up is not None and up.near_edge == 110.0
    tr = _scenario(lows_extra=[(10, 80.0)], highs_extra=[(11, 120.0)])
    assert tr is not None and tr.direction == "LONG"


def test_t0_7_fill_gaps_through_upper_invalid_branch():
    # entry=120 skips the upper 110 target -> upper branch invalid -> SHORT(lower)
    tr = _scenario(
        opens_extra=[(11, 120.0)], highs_extra=[(11, 120.0)], lows_extra=[(12, 80.0)]
    )
    assert tr is not None
    assert tr.direction == "SHORT"
    assert abs(tr.target_price - 90.0) < EPS


def test_t0_8_fill_gaps_through_lower_invalid_branch():
    tr = _scenario(opens_extra=[(11, 80.0)], highs_extra=[(12, 120.0)])
    assert tr is not None
    assert tr.direction == "LONG"
    assert abs(tr.target_price - 110.0) < EPS


def test_t0_9_target_frozen_independently_per_decision():
    g_a = _sr_geom([(115.0, 110.0, 1.0), (90.0, 85.0, 1.0)])
    g_b = _sr_geom([(205.0, 200.0, 1.0), (90.0, 85.0, 1.0)])
    u_a, _ = nearest_upper_and_lower_target(
        g_a, close_d=100.0, high_d=100.0, low_d=100.0,
        seg=0, d=10, sr_first_seen={}, candidate_sid="X",
    )
    u_b, _ = nearest_upper_and_lower_target(
        g_b, close_d=100.0, high_d=100.0, low_d=100.0,
        seg=0, d=10, sr_first_seen={}, candidate_sid="X",
    )
    assert u_a is not None and u_a.near_edge == 110.0
    assert u_b is not None and u_b.near_edge == 200.0  # different decision -> different target


def test_t0_10_two_candidates_same_decision_independent_options():
    n = 30
    per_bar = [None] * n
    sid_a = "SR|m15|0|10|120.0|110.0|1.0"
    sid_b = "SR|m15|0|10|80.0|70.0|1.0"
    per_bar[10] = [(sid_a, "SR", 110.0, 120.0), (sid_b, "SR", 70.0, 80.0)]
    opens = [100.0] * n
    highs = [100.0] * n
    lows = [100.0] * n
    closes = [100.0] * n
    highs[11] = 120.0
    lows[12] = 80.0
    times = list(range(n))
    td = [0] * n
    seg = [0] * n
    mv = _make_mv(opens, highs, lows, closes, times, td, seg)
    g = _sr_geom([(115.0, 110.0, 1.0), (90.0, 85.0, 1.0)])
    geom = [g] * n
    atr = [1.0] * n
    options = enumerate_trade_options(
        per_bar=per_bar, geom=geom,
        opens=np.asarray(opens), highs=np.asarray(highs), lows=np.asarray(lows),
        closes=np.asarray(closes), atr_series=np.asarray(atr),
        trading_day=np.asarray(td), segment=np.asarray(seg), mv=mv,
        sr_first_seen={}, event_id_by_sid={},
    )
    assert len(options[10]) == 2
    assert {o.candidate_structure_id for o in options[10]} == {sid_a, sid_b}
    assert all(o.direction == "LONG" for o in options[10])


# --------------------------------------------------------------------------- #
# T0 -- SCHEDULER
# --------------------------------------------------------------------------- #
def _opt(dec, fill, exit, util, direction="LONG"):
    return TradeOption(
        decision=dec, fill=fill, exit=exit, direction=direction,
        candidate_structure_id="X", candidate_structure_type="SR",
        zone_bottom=0.0, zone_top=0.0,
        target_structure_id="Y", target_structure_type="SR",
        target_price=0.0, entry_price=0.0, exit_price=0.0,
        utility_points=float(util), utility_atr=0.0, entry_gap_atr=0.0,
        event_id=-1,
    )


def test_t0_11_scheduler_picks_two_small_over_one_large():
    n = 200
    options = [[] for _ in range(n)]
    options[0] = [_opt(0, 1, 100, 50.0)]
    options[10] = [_opt(10, 11, 20, 40.0)]
    options[30] = [_opt(30, 31, 40, 40.0)]
    sel = solve_nonoverlap_god_dp(options, n)
    assert [o.decision for o in sel] == [10, 30]
    assert abs(sum(o.utility_points for o in sel) - 80.0) < EPS


def test_t0_12_scheduler_picks_one_large_over_two_small():
    n = 200
    options = [[] for _ in range(n)]
    options[0] = [_opt(0, 1, 100, 100.0)]
    options[10] = [_opt(10, 11, 20, 40.0)]
    options[30] = [_opt(30, 31, 40, 40.0)]
    sel = solve_nonoverlap_god_dp(options, n)
    assert [o.decision for o in sel] == [0]
    assert abs(sel[0].utility_points - 100.0) < EPS


def test_t0_13_selected_intervals_never_overlap():
    n = 200
    options = [[] for _ in range(n)]
    options[5] = [_opt(5, 6, 15, 10.0)]
    options[12] = [_opt(12, 13, 25, 10.0)]
    options[20] = [_opt(20, 21, 30, 10.0)]
    options[0] = [_opt(0, 1, 100, 5.0)]  # overlaps the rest
    sel = solve_nonoverlap_god_dp(options, n)
    for i in range(len(sel) - 1):
        assert sel[i].exit < sel[i + 1].decision


def test_t0_14_same_structure_generates_options_again_later():
    n = 60
    per_bar = [None] * n
    sid = "SR|m15|0|5|120.0|110.0|1.0"
    per_bar[5] = [(sid, "SR", 110.0, 120.0)]
    per_bar[50] = [(sid, "SR", 110.0, 120.0)]
    opens = [100.0] * n
    highs = [100.0] * n
    lows = [100.0] * n
    closes = [100.0] * n
    highs[6] = 120.0
    highs[51] = 120.0
    times = list(range(n))
    td = [0] * n
    seg = [0] * n
    mv = _make_mv(opens, highs, lows, closes, times, td, seg)
    g = _sr_geom([(115.0, 110.0, 1.0), (90.0, 85.0, 1.0)])
    geom = [g] * n
    atr = [1.0] * n
    options = enumerate_trade_options(
        per_bar=per_bar, geom=geom,
        opens=np.asarray(opens), highs=np.asarray(highs), lows=np.asarray(lows),
        closes=np.asarray(closes), atr_series=np.asarray(atr),
        trading_day=np.asarray(td), segment=np.asarray(seg), mv=mv,
        sr_first_seen={}, event_id_by_sid={},
    )
    assert len(options[5]) == 1 and options[5][0].candidate_structure_id == sid
    assert len(options[50]) == 1 and options[50][0].candidate_structure_id == sid


# --------------------------------------------------------------------------- #
# T1 -- REAL SCREENSHOT REGION
# --------------------------------------------------------------------------- #
def test_t1_15_diag_resolves_as_long():
    res = run_god_oracle_v5("AG")
    canon = [r for r in res["records"] if r["canonical_oracle_trade"]]
    diag = [r for r in canon if r["structure_id"] == DIAG_SID]
    assert diag, "diagnosed structure must appear in V5 canonical stream"
    d = diag[0]
    print("DIAG trade:", {
        "decision": d["best_entry_decision_index"],
        "fill": d["best_entry_fill_index"],
        "direction": d["oracle_direction"],
        "entry": d["best_entry_price"],
        "target": d["target_price"],
        "exit": d["exit_fill_index"],
        "utility": d["utility"],
    })
    assert d["oracle_direction"] == "LONG"
    assert d["best_entry_decision_index"] == 349
    assert d["best_entry_price"] == 7700.0
    assert d["target_price"] == 7773.0
    assert d["exit_fill_index"] == 351
    assert d["utility"] == 73.0


def test_t1_16_old_short_7655_rejected_upper_first():
    res = run_god_oracle_v5("AG")
    canon = [r for r in res["records"] if r["canonical_oracle_trade"]]

    # (a) the old PREEMPT SHORT=7655 must NOT survive as a direction
    short_prem = [r for r in canon
                  if r["structure_id"] == PREEMPT_SID
                  and r["oracle_direction"] == "SHORT"]
    assert not short_prem, "old PREEMPT SHORT=7655 must NOT survive as direction"

    # (b) MANDATORY SCREENSHOT-2 AUDIT at the DIAG decision bar (d=349)
    inp = _load_inputs("AG")
    geom = inp["geom"]
    closes = inp["closes"]
    highs = inp["highs"]
    lows = inp["lows"]
    opens = inp["opens"]
    seg_arr = inp["seg_arr"]
    mv = inp["mv"]
    tree = _build_target_tree(mv)
    d = 349
    gp = geom[d - 1]
    up, lo = nearest_upper_and_lower_target(
        gp, close_d=float(closes[d]), high_d=float(highs[d]),
        low_d=float(lows[d]), seg=int(seg_arr[d]), d=d,
        sr_first_seen={}, candidate_sid=DIAG_SID,
    )
    tau_up = _first_target_touch(tree, d + 1, "LONG", float(up.near_edge)) if up else -1
    tau_dn = _first_target_touch(tree, d + 1, "SHORT", float(lo.near_edge)) if lo else -1
    print("SCREENSHOT-2 AUDIT d=", d)
    print("  close[d]=", closes[d], "fill open[d+1]=", opens[d + 1])
    print("  upper:", up.structure_id if up else None,
          up.near_edge if up else None,
          "touched@decision?", (float(highs[d]) >= up.near_edge) if up else None,
          "tau_up=", tau_up)
    print("  lower:", lo.structure_id if lo else None,
          lo.near_edge if lo else None,
          "touched@decision?", (float(lows[d]) <= lo.near_edge) if lo else None,
          "tau_dn=", tau_dn)
    assert up is not None, "DIAG must have an upper target"
    assert tau_up >= 0, "upper target must be reachable"
    assert tau_dn < 0 or tau_up < tau_dn, (
        "upper target reached first => LONG; lower 7655 cannot define direction"
    )


def test_t1_17_compare_v4_vs_v5_first_20():
    res5 = run_god_oracle_v5("AG")
    canon5 = [r for r in res5["records"] if r["canonical_oracle_trade"]]
    canon5.sort(key=lambda r: int(r["best_entry_decision_index"]))
    res4 = run_god_oracle_v4("AG")
    canon4 = [r for r in res4["records"] if r["canonical_oracle_trade"]]
    canon4.sort(key=lambda r: int(r["candidate_start_bar"]))

    v5_first = [(r["best_entry_decision_index"], r["oracle_direction"], r["structure_id"])
                for r in canon5[:20]]
    v4_first = [(r["candidate_start_bar"], r["oracle_direction"], r["structure_id"])
                for r in canon4[:20]]
    print("V5 first-20:", v5_first)
    print("V4 first-20:", v4_first)

    # V5 recovers the diagnosed LONG that V4's greedy cursor pushed out of the stream
    assert any(r["structure_id"] == DIAG_SID and r["oracle_direction"] == "LONG"
               for r in canon5)
    assert not any(r["structure_id"] == DIAG_SID for r in canon4), \
        "V4 had the diagnosed structure excluded from the full stream"
