"""test_structural_god_oracle_m15_v4
====================================

T0 A-R + negative controls + production/reference parity for the V4 God-mode
oracle (FUT-M15-STRUCTURAL-GOD-ORACLE-V4).

The God-mode contract under test:
  * direction is decided by comparing V_LONG vs V_SHORT (no external model)
  * a canonical oracle trade MUST be profitable (PnL > 0)
  * entry may ONLY come from this event's own contact bars
  * the candidate structure can never be its own target
"""

import numpy as np
import pandas as pd
import pytest

from research.liquidity_oracle_atlas.structural_god_oracle_m15_v4 import (
    solve_direction_god_v4,
    pick_directional_target_v4,
    decide_oracle_v4,
    target_is_self,
    run_god_oracle_v4,
    build_structural_events_v2,
    build_per_bar_proximity,
    build_event_contact_bars,
    market_view_from_arrays,
    _contact_is_consumed,
    _scan_next_candidate,
    R_TARGET,
    ORACLE_LONG,
    ORACLE_SHORT,
)
from research.liquidity_oracle_atlas.structural_god_oracle_reference_m15_v4 import (
    solve_direction_god_reference_v4,
    pick_directional_target_v4_ref,
)
from research.liquidity_oracle_atlas.build_dp_proximity_m15_v1 import (
    build_dp_proximity_m15,
)
from research.liquidity_oracle_atlas.build_execution_environment_m15_v1 import (
    run_environment_m15,
)
from research.liquidity_oracle_atlas.structural_event_dp_kernel_v3 import (
    build_market_view,
)


# --------------------------------------------------------------------------- #
# Synthetic helpers
# --------------------------------------------------------------------------- #
def make_mv(opens, highs, lows, unit_starts=None):
    n = len(opens)
    closes = list(opens)
    times = [pd.Timestamp("2020-01-01") + pd.Timedelta(minutes=15 * i) for i in range(n)]
    return market_view_from_arrays(
        opens, highs, lows, closes, times=times, unit_starts=unit_starts, symbol="SYNTH"
    )


def god_solve(direction, mv, contact_bars, target_price, atr=1.0, zb=0.0, zt=0.0,
              s_bar=0, e_bar=None, trading_day=None, segment=None):
    if e_bar is None:
        e_bar = mv.n
    return solve_direction_god_v4(
        direction=direction, zone_bottom=zb, zone_top=zt, atr_value=atr,
        start_bar=s_bar, end_bar=e_bar, target_price=target_price,
        contact_bars=contact_bars, mv=mv,
        trading_day=trading_day, segment=segment,
    )


# --------------------------------------------------------------------------- #
# T0 A-F : oracle direction decision
# --------------------------------------------------------------------------- #
def test_decide_a_long_wins():
    dec, lv, sv, side = decide_oracle_v4(
        {"ok": True, "utility": 10.0}, {"ok": True, "utility": 3.0})
    assert dec == "LONG" and side == "LONG"
    assert abs(lv - 10.0) < 1e-9 and abs(sv - 3.0) < 1e-9


def test_decide_b_short_wins():
    dec, lv, sv, side = decide_oracle_v4(
        {"ok": True, "utility": -0.2}, {"ok": True, "utility": 1.5})
    assert dec == "SHORT" and side == "SHORT"


def test_decide_c_pick_larger_not_just_positive():
    dec, lv, sv, side = decide_oracle_v4(
        {"ok": True, "utility": 5.0}, {"ok": True, "utility": 8.0})
    assert dec == "SHORT"  # both positive, but only the larger is the oracle


def test_decide_d_one_loss_one_profit():
    dec, lv, sv, side = decide_oracle_v4(
        {"ok": True, "utility": -0.3}, {"ok": True, "utility": 2.1})
    assert dec == "SHORT" and side == "SHORT"


def test_t0_a_long_clearly_more():
    opens = [100] * 7
    highs = [100, 100, 100, 110, 100, 100, 100]
    lows = [100, 100, 100, 100, 97, 100, 100]
    mv = make_mv(opens, highs, lows)
    cb = [1]
    long_s = god_solve("LONG", mv, cb, 110.0)
    short_s = god_solve("SHORT", mv, cb, 97.0)
    dec, lv, sv, side = decide_oracle_v4(long_s, short_s)
    assert dec == "LONG"
    assert long_s["exit_reason"] == "TARGET_TOUCH"
    assert long_s["tp_atr"] == 10.0
    assert abs(lv - 10.0) < 1e-9 and abs(sv - 3.0) < 1e-9
    assert long_s["utility"] > 0 and long_s["tp_atr"] > 0


def test_t0_b_short_clearly_more():
    opens = [100] * 7
    highs = [100, 100, 101, 100, 100, 100, 100]
    lows = [100, 100, 100, 100, 90, 100, 100]
    mv = make_mv(opens, highs, lows)
    cb = [1]
    long_s = god_solve("LONG", mv, cb, 101.0)
    short_s = god_solve("SHORT", mv, cb, 90.0)
    dec, lv, sv, side = decide_oracle_v4(long_s, short_s)
    assert dec == "SHORT"
    assert abs(lv - 1.0) < 1e-9 and abs(sv - 10.0) < 1e-9


def test_t0_c_both_profit_pick_larger():
    opens = [100] * 7
    highs = [100, 100, 105, 100, 100, 100, 100]
    lows = [100, 100, 100, 100, 92, 100, 100]
    mv = make_mv(opens, highs, lows)
    cb = [1]
    long_s = god_solve("LONG", mv, cb, 105.0)
    short_s = god_solve("SHORT", mv, cb, 92.0)
    dec, lv, sv, side = decide_oracle_v4(long_s, short_s)
    assert dec == "SHORT"
    assert abs(lv - 5.0) < 1e-9 and abs(sv - 8.0) < 1e-9


def test_t0_d_one_loss_one_profit():
    # V4.3: a trade only forms on a profitable structural-target touch. An
    # unreachable target does not form a trade (branch not ok); the reachable,
    # profitable branch wins.
    opens = [100, 100, 100, 110, 100, 100, 100]
    highs = list(opens)
    lows = list(opens)
    mv = make_mv(opens, highs, lows)
    cb = [1]
    long_s = god_solve("LONG", mv, cb, 110.0)   # reachable & profitable
    short_s = god_solve("SHORT", mv, cb, 90.0)  # unreachable -> not ok
    dec, lv, sv, side = decide_oracle_v4(long_s, short_s)
    assert dec == "LONG"
    assert long_s["ok"] and long_s["utility"] > 0
    assert not short_s["ok"]


def test_t0_e_no_positive_opportunity():
    dec, lv, sv, side = decide_oracle_v4(
        {"ok": True, "utility": -1.0}, {"ok": True, "utility": -2.0})
    assert dec == "NO_POSITIVE_OPPORTUNITY"
    assert side is None


def test_t0_f_exact_direction_tie():
    dec, lv, sv, side = decide_oracle_v4(
        {"ok": True, "utility": 5.0}, {"ok": True, "utility": 5.0})
    assert dec == "ORACLE_DIRECTION_TIE"
    assert side is None


# --------------------------------------------------------------------------- #
# T0 G-I : target selection
# --------------------------------------------------------------------------- #
def _geom_with(channels, liq_up=None, liq_down=None, atr=1.0):
    return {"m15": (channels, liq_up or [], liq_down or [], atr)}


def test_t0_g_self_target_excluded():
    cand_sid = "SR|m15|0|0|110|100|1.0"  # zone [100,110]
    geom = _geom_with([(110, 100, 1.0), (130, 120, 1.0)])  # self + one above
    tgt = pick_directional_target_v4(geom, "LONG", 100.0, 110.0, "m15", 0, 0, {}, cand_sid)
    assert tgt is not None
    assert target_is_self(tgt["structure_id"], cand_sid) is False
    assert tgt["structure_id"] == "SR|m15|0|0|130|120|1.0"


def test_t0_h_long_target_above_zone():
    cand_sid = "SR|m15|0|0|110|100|1.0"
    geom = _geom_with([(110, 100, 1.0), (130, 120, 1.0)])
    tgt = pick_directional_target_v4(geom, "LONG", 100.0, 110.0, "m15", 0, 0, {}, cand_sid)
    assert float(tgt["near_edge"]) > 110.0 + 1e-9


def test_t0_i_short_target_below_zone():
    cand_sid = "SR|m15|0|0|100|90|1.0"  # SUPPORT zone [90,100]
    geom = _geom_with([(100, 90, 1.0), (80, 70, 1.0)])  # self + one below
    tgt = pick_directional_target_v4(geom, "SHORT", 90.0, 100.0, "m15", 0, 0, {}, cand_sid)
    assert tgt is not None
    assert target_is_self(tgt["structure_id"], cand_sid) is False
    assert float(tgt["near_edge"]) < 90.0 - 1e-9
    assert tgt["structure_id"] == "SR|m15|0|0|80|70|1.0"


def test_t0_g_reference_picker_matches():
    cand_sid = "SR|m15|0|0|110|100|1.0"
    geom = _geom_with([(110, 100, 1.0), (130, 120, 1.0)])
    tgt_p = pick_directional_target_v4(geom, "LONG", 100.0, 110.0, "m15", 0, 0, {}, cand_sid)
    tgt_r = pick_directional_target_v4_ref(geom, "LONG", 100.0, 110.0, "m15", 0, 0, {}, cand_sid)
    assert tgt_p["structure_id"] == tgt_r["structure_id"]
    assert abs(tgt_p["near_edge"] - tgt_r["near_edge"]) < 1e-12


# --------------------------------------------------------------------------- #
# T0 J-L : entry/exit lifecycle
# --------------------------------------------------------------------------- #
def test_t0_j_target_before_entry_illegal():
    opens = [100] * 5
    highs = [100, 115, 100, 100, 100]  # target 110 touched at bar 1, before fill 2
    lows = [100] * 5
    mv = make_mv(opens, highs, lows)
    cb = [1]
    long_s = god_solve("LONG", mv, cb, 110.0)
    assert long_s["ok"] is False  # entry rejected: target already touched


def test_t0_k_same_bar_entry_target_long():
    opens = [100] * 6
    highs = [100, 100, 115, 100, 100, 100]
    lows = [100] * 6
    mv = make_mv(opens, highs, lows)
    cb = [1]
    long_s = god_solve("LONG", mv, cb, 110.0)
    assert long_s["ok"]
    assert long_s["exit_reason"] == "TARGET_TOUCH"
    assert long_s["exit_fill_index"] == 2
    assert long_s["tp_atr"] == 10.0


def test_t0_k_same_bar_entry_target_short():
    opens = [100] * 6
    highs = [100] * 6
    lows = [100, 100, 90, 100, 100, 100]
    mv = make_mv(opens, highs, lows)
    cb = [1]
    short_s = god_solve("SHORT", mv, cb, 90.0)
    assert short_s["ok"]
    assert short_s["exit_reason"] == "TARGET_TOUCH"
    assert short_s["exit_fill_index"] == 2
    assert short_s["tp_atr"] == 10.0


def test_t0_l_target_touch_hard_terminal():
    opens = [100] * 7
    highs = [100, 100, 100, 110, 100, 125, 100]  # touch at 3, higher later at 5
    lows = [100] * 7
    mv = make_mv(opens, highs, lows)
    cb = [1]
    long_s = god_solve("LONG", mv, cb, 110.0)
    assert long_s["exit_fill_index"] == 3
    assert long_s["exit_price"] == 110.0
    assert long_s["tp_atr"] == 10.0  # NOT 25 from bar 5


def test_t0_m_target_not_reached_no_trade():
    # V4.3: an unreachable structural target does NOT open a trade. The branch is
    # simply not ok (TARGET_NOT_REACHED) -- there is no early-exit fallback.
    opens = [100, 100, 100, 105, 108, 103, 100]
    highs = list(opens)
    lows = list(opens)
    mv = make_mv(opens, highs, lows)
    cb = [1]
    long_s = god_solve("LONG", mv, cb, 200.0)  # unreachable
    assert not long_s["ok"]
    assert long_s["invalid_reason"] == "TARGET_NOT_REACHED"


def test_t0_n_entry_only_from_contact_bars():
    opens = [100] * 7
    highs = [100, 100, 110, 100, 100, 100, 100]
    lows = [100, 100, 100, 50, 100, 100, 100]  # bar3 super low but NOT contact
    mv = make_mv(opens, highs, lows)
    cb = [1]
    long_s = god_solve("LONG", mv, cb, 110.0)
    assert long_s["ok"]
    assert long_s["entry_decision_index"] == 1
    assert long_s["exit_fill_index"] == 2


def test_t0_o_leave_and_return_two_segments():
    opens = [100, 100, 100, 100, 100, 100, 95, 100, 100]
    highs = [100, 100, 100, 100, 100, 100, 110, 100, 100]  # target touched only at bar 6
    lows = [100] * 9
    mv = make_mv(opens, highs, lows)
    cb = [1, 2, 5, 6]  # two contact segments
    long_s = god_solve("LONG", mv, cb, 110.0)
    assert long_s["ok"]
    assert long_s["entry_decision_index"] == 5  # best pnl among contacts


def test_t0_p_next_event_boundary_caps_exit():
    # V4.3: the exit is the first structural-target touch. A target touched only
    # at the next-event boundary (e_bar) is ambiguous and rejected (capped); a
    # touch strictly before e_bar yields a valid exit with exit_fill_index < e_bar.
    opens = [100, 100, 100, 100, 100, 100, 100, 100]
    lows = [100] * 8
    # LONG: target 105 first touched at bar 3 (< e_bar=4)
    highs = [100, 100, 100, 105, 100, 100, 100, 100]
    mv = make_mv(opens, highs, lows)
    cb = [1]
    long_s = god_solve("LONG", mv, cb, 105, e_bar=4)
    assert long_s["ok"]
    assert long_s["exit_fill_index"] == 3
    assert long_s["exit_fill_index"] < 4

    # LONG: target 105 first touched exactly at e_bar=4 -> ambiguous -> capped
    highs2 = [100, 100, 100, 100, 105, 100, 100, 100]
    mv2 = make_mv(opens, highs2, lows)
    long_s2 = god_solve("LONG", mv2, cb, 105, e_bar=4)
    assert not long_s2["ok"]
    assert long_s2["invalid_reason"] == "AMBIGUOUS_SAME_BAR_TERMINAL"

    # SHORT analogue: target 95 first touched at bar 3 (< e_bar=4)
    opens_s = [100, 100, 100, 100, 100, 100, 100, 100]
    lows_s = [100, 100, 100, 95, 100, 100, 100, 100]
    mv_s = make_mv(opens_s, [100] * 8, lows_s)
    short_s = god_solve("SHORT", mv_s, cb, 95, e_bar=4)
    assert short_s["ok"]
    assert short_s["exit_fill_index"] == 3
    assert short_s["exit_fill_index"] < 4


def test_t0_p_ambiguous_terminal_rejected():
    opens = [100] * 8
    highs = [100, 100, 100, 100, 110, 100, 100, 100]  # target touched exactly at e_bar=4
    lows = [100] * 8
    mv = make_mv(opens, highs, lows)
    cb = [1]
    long_s = god_solve("LONG", mv, cb, 110.0, e_bar=4)
    assert long_s["ok"] is False
    assert long_s["invalid_reason"] == "AMBIGUOUS_SAME_BAR_TERMINAL"


def test_t0_q_winning_oracle_profitable():
    # reuse T0-A numbers: canonical trade is profitable
    opens = [100] * 7
    highs = [100, 100, 100, 110, 100, 100, 100]
    lows = [100, 100, 100, 100, 97, 100, 100]
    mv = make_mv(opens, highs, lows)
    long_s = god_solve("LONG", mv, [1], 110.0)
    assert long_s["utility"] > 0
    assert long_s["tp_atr"] > 0


def test_t0_r_legacy_direction_irrelevant():
    # God-mode decision is a pure function of the two branch values, never of any
    # external direction. Calling with identical branches yields identical output.
    a = {"ok": True, "utility": 7.0}
    b = {"ok": True, "utility": 2.0}
    d1 = decide_oracle_v4(a, b)
    d2 = decide_oracle_v4(a, b)
    assert d1 == d2
    assert d1[0] == "LONG"  # decided purely by V_LONG > V_SHORT


# --------------------------------------------------------------------------- #
# Negative controls
# --------------------------------------------------------------------------- #
def test_nc1_future_permutation_changes_oracle():
    opens = [100, 100, 100, 90, 110, 95, 105]
    cb = [1]

    def _result(op):
        m = make_mv(op, list(op), list(op))
        # reachable targets; the oracle must use the future path to locate the
        # structural-target touch (which differs across permutations)
        rl = god_solve("LONG", m, cb, 110.0)
        rs = god_solve("SHORT", m, cb, 90.0)
        return (
            rl["ok"], rl.get("exit_fill_index"),
            rs["ok"], rs.get("exit_fill_index"),
        )

    tup1 = _result(opens)
    permuted = [100, 100, 100, 95, 105, 110, 90]
    tup2 = _result(permuted)
    assert tup1 != tup2, "god-mode solver must actually use the future path"


def test_nc2_noncontact_fake_perfect_entry_ignored():
    opens = [100] * 7
    highs = [100, 100, 110, 100, 100, 100, 100]
    lows = [100, 100, 100, 40, 100, 100, 100]  # bar3 super low but not contact
    mv = make_mv(opens, highs, lows)
    long_s = god_solve("LONG", mv, [1], 110.0)
    assert long_s["ok"]
    assert long_s["entry_decision_index"] == 1


def test_nc3_self_target_injection_excluded():
    cand_sid = "SR|m15|0|0|110|100|1.0"
    geom = _geom_with([(110, 100, 1.0), (130, 120, 1.0)])
    tgt = pick_directional_target_v4(geom, "LONG", 100.0, 110.0, "m15", 0, 0, {}, cand_sid)
    assert tgt is not None
    assert target_is_self(tgt["structure_id"], cand_sid) is False


# --------------------------------------------------------------------------- #
# V4.1 execution-boundary + target-gated negative controls
# --------------------------------------------------------------------------- #
def test_t0_s1_cross_day_fill_rejected():
    # decision = last bar of day 0; fill = next day's open -> MUST be rejected
    opens = [100] * 6
    highs = [100, 100, 100, 110, 100, 100]
    lows = [100] * 6
    td = [0, 0, 0, 1, 1, 1]
    seg = [0, 0, 0, 0, 0, 0]
    unit_starts = np.array([0, 3], dtype=np.int64)
    mv = make_mv(opens, highs, lows, unit_starts=unit_starts)
    cb = [2]  # fill bar 3 is a different trading_day
    long_s = god_solve("LONG", mv, cb, 110.0, trading_day=td, segment=seg)
    assert long_s["ok"] is False


def test_t0_s2_cross_unit_fill_rejected():
    # even when time is continuous, a unit change must reject the fill
    opens = [100] * 6
    highs = [100, 100, 100, 110, 100, 100]
    lows = [100] * 6
    td = [0, 0, 0, 0, 0, 0]
    seg = [0, 0, 0, 1, 1, 1]
    unit_starts = np.array([0, 3], dtype=np.int64)
    mv = make_mv(opens, highs, lows, unit_starts=unit_starts)
    cb = [2]  # fill bar 3 is a different segment/unit
    long_s = god_solve("LONG", mv, cb, 110.0, trading_day=td, segment=seg)
    assert long_s["ok"] is False


def test_t0_s3_ordinary_next_bar_fill_preserved():
    # same day / segment / unit -> ordinary t+1 fill stays legal
    opens = [100] * 6
    highs = [100, 100, 110, 100, 100, 100]
    lows = [100] * 6
    td = [0, 0, 0, 0, 0, 0]
    seg = [0, 0, 0, 0, 0, 0]
    unit_starts = np.array([0], dtype=np.int64)
    mv = make_mv(opens, highs, lows, unit_starts=unit_starts)
    cb = [1]
    long_s = god_solve("LONG", mv, cb, 110.0, trading_day=td, segment=seg)
    assert long_s["ok"]
    assert long_s["exit_reason"] == "TARGET_TOUCH"
    assert long_s["exit_fill_index"] == 2


def test_nc4_overnight_fake_opportunity_rejected():
    # session close 100, next session open 150 -> the cross-day entry (contact 2
    # -> fill 3) would be a huge fake profit. The execution-boundary gate MUST
    # reject it; the only legal entry is the same-session contact 0 (fill 1).
    opens = [100, 100, 110, 150, 200, 100]
    highs = list(opens)
    lows = list(opens)
    td = [0, 0, 0, 1, 1, 1]
    seg = [0, 0, 0, 0, 0, 0]
    unit_starts = np.array([0, 3], dtype=np.int64)
    mv = make_mv(opens, highs, lows, unit_starts=unit_starts)
    cb = [0, 2]
    long_s = god_solve("LONG", mv, cb, 110.0, trading_day=td, segment=seg)
    assert long_s["ok"]
    # the cross-day candidate (contact 2 -> fill 3, different trading_day) is
    # rejected by the boundary gate; the legal entry is contact 0.
    assert long_s["entry_decision_index"] == 0
    assert long_s["exit_fill_index"] == 2
    assert long_s["utility"] == 10.0
    assert long_s["entry_decision_index"] == 0
    # the cross-day-only candidate alone must be entirely rejected
    long_s2 = god_solve("LONG", mv, [2], 999.0, trading_day=td, segment=seg)
    assert long_s2["ok"] is False


# --------------------------------------------------------------------------- #
# Production / Reference parity
# --------------------------------------------------------------------------- #
AG_MAX_BARS = 1500
AG_EVENT_LIMIT = 15


def _ag_context():
    ev = build_structural_events_v2("AG", AG_MAX_BARS)
    events = ev["events"][:AG_EVENT_LIMIT]
    prox = build_dp_proximity_m15("AG", AG_MAX_BARS)
    env = run_environment_m15("AG", AG_MAX_BARS, capture_provenance=False)
    geom = env["geom_by_decision"]
    per_bar = build_per_bar_proximity(prox, geom)
    build_event_contact_bars(events, per_bar)
    mv = build_market_view("AG", AG_MAX_BARS)
    return events, per_bar, mv, env, prox


def test_parity_solver_level_on_ag():
    events, per_bar, mv, env, prox = _ag_context()
    atr = np.asarray(env["features"]["m15_atr"].to_numpy(float))
    seg = prox_seg = prox_seg_of(events, mv)
    td_arr = prox["trading_day"].to_numpy(np.int64)
    seg_arr = prox["segment"].to_numpy(np.int64)
    sr_fs = {}
    mismatches = 0
    max_err = 0.0
    compared = 0
    for e in events:
        s_bar = int(e["start_bar"])
        e_bar = int(e["end_bar"])
        zb, zt = float(e["zone_bottom"]), float(e["zone_top"])
        atr_v = float(atr[s_bar]) if s_bar < len(atr) else float("nan")
        geom_prev = geom_at(env, s_bar)
        tgt_l = pick_directional_target_v4(geom_prev, "LONG", zb, zt, e["timeframe"],
                                           int(seg[s_bar]) if s_bar < mv.n else 0, s_bar, sr_fs, e["structure_id"])
        tgt_s = pick_directional_target_v4(geom_prev, "SHORT", zb, zt, e["timeframe"],
                                           int(seg[s_bar]) if s_bar < mv.n else 0, s_bar, sr_fs, e["structure_id"])
        cb = e["contact_bars"]
        for direction, tgt in (("LONG", tgt_l), ("SHORT", tgt_s)):
            tp = None if tgt is None else float(tgt["near_edge"])
            a = solve_direction_god_v4(direction=direction, zone_bottom=zb, zone_top=zt,
                                       atr_value=atr_v, start_bar=s_bar, end_bar=e_bar,
                                       target_price=tp, contact_bars=cb, mv=mv,
                                       trading_day=td_arr, segment=seg_arr)
            b = solve_direction_god_reference_v4(direction=direction, zone_bottom=zb, zone_top=zt,
                                                 atr_value=atr_v, start_bar=s_bar, end_bar=e_bar,
                                                 target_price=tp, contact_bars=cb, mv=mv,
                                                 trading_day=td_arr, segment=seg_arr)
            compared += 1
            if a.get("ok") != b.get("ok"):
                mismatches += 1
                continue
            if not a.get("ok"):
                continue
            for k in ("entry_decision_index", "entry_fill_index", "exit_fill_index", "utility"):
                err = abs(float(a[k]) - float(b[k]))
                max_err = max(max_err, err)
                if err > 1e-9:
                    mismatches += 1
                    break
    assert compared > 0
    assert mismatches == 0, f"solver parity mismatches={mismatches} max_err={max_err}"
    assert max_err <= 1e-9


def prox_seg_of(events, mv):
    # segment array reconstructed from the market view is not stored; rebuild from
    # the AG proximity frame for parity target picking.
    prox = build_dp_proximity_m15("AG", AG_MAX_BARS)
    return prox["segment"].to_numpy(np.int64)


def geom_at(env, s_bar):
    g = env["geom_by_decision"]
    return g[s_bar - 1] if s_bar >= 1 else None


def test_parity_full_run_on_ag():
    """Production oracle now emits a single sequential trade stream
    (restart immediately after each TARGET_TOUCH exit) instead of at most
    one record per static structural event. Validate the new stream
    contract. (Math-function parity against the reference solver is covered
    independently by test_parity_solver_level_on_ag.)"""
    prod = run_god_oracle_v4("AG", max_bars=AG_MAX_BARS, event_limit=AG_EVENT_LIMIT)
    rp = prod["records"]
    canon = [r for r in rp if r.get("canonical_oracle_trade")]

    # the stream emits ONLY canonical, completed TARGET_TOUCH trades
    assert len(rp) == len(canon)
    assert len(canon) > 0
    for r in canon:
        assert r["exit_reason"] == R_TARGET
        assert r["oracle_direction"] in (ORACLE_LONG, ORACLE_SHORT)
        assert r["utility"] > 0 and r["tp_atr"] > 0
        assert r["direction_margin_atr"] >= -1e-12, r["direction_margin_atr"]

    # global non-overlap hard invariant: Entry_1 <= Exit_1 < Entry_2 <= ...
    ents = sorted(int(r["best_entry_fill_index"]) for r in canon)
    ex = {int(r["best_entry_fill_index"]): int(r["exit_fill_index"]) for r in canon}
    prev_exit = -1
    for e in ents:
        assert e > prev_exit, f"overlap detected at entry bar {e}"
        prev_exit = ex[e]

    # hard contract guards from the production god-mode contract
    assert prod["meta"]["canonical_loss_count"] == 0
    assert prod["meta"]["self_target_count"] == 0
    assert prod["meta"]["valid_but_no_entry"] == 0


# --------------------------------------------------------------------------- #
# God-mode best-entry restoration regression (sequential-stream refactor)
# --------------------------------------------------------------------------- #
def test_god_mode_best_entry_full_contact_set():
    # Step 11 (user scenario): Candidate A contacts 10, 11, 12; frozen LONG
    # target = 120. fills: open[11]=100, open[12]=105, open[13]=112.
    # The God-mode solver MUST choose the MAX-PROFIT entry across ALL of A's
    # contacts (decision 10 -> fill 11 -> 100), not merely the first / a single
    # contact. This is the behavior the cb_full orchestration fix restores.
    n = 20
    opens = [100.0] * n
    opens[11] = 100.0
    opens[12] = 105.0
    opens[13] = 112.0
    highs = [100.0] * n
    highs[15] = 120.0  # target first touched at bar 15 (LONG)
    lows = [100.0] * n
    mv = make_mv(opens, highs, lows, unit_starts=np.array([0], dtype=np.int64))
    sol = god_solve("LONG", mv, [10, 11, 12], 120.0)
    assert sol["ok"]
    assert sol["exit_reason"] == R_TARGET
    assert sol["entry_decision_index"] == 10
    assert sol["entry_fill_index"] == 11
    assert abs(sol["entry_price"] - 100.0) < 1e-9


def test_god_mode_best_entry_lost_if_single_contact():
    # Step 11 discriminator: when the MAX-PROFIT contact is NOT the first one,
    # a single-contact (cb_now=[c]) orchestration cannot find it. The full
    # contact set (cb_full) fixes this. This test FAILS under cb_now behavior.
    n = 20
    opens = [100.0] * n
    opens[11] = 112.0  # decision 10 -> fill 11 -> expensive
    opens[12] = 105.0  # decision 11 -> fill 12
    opens[13] = 100.0  # decision 12 -> fill 13 -> cheapest (God-mode best)
    highs = [100.0] * n
    highs[15] = 120.0
    lows = [100.0] * n
    mv = make_mv(opens, highs, lows, unit_starts=np.array([0], dtype=np.int64))
    # full contact set -> God-mode picks the cheapest entry (decision 12)
    full = god_solve("LONG", mv, [10, 11, 12], 120.0)
    assert full["ok"]
    assert full["entry_decision_index"] == 12
    assert abs(full["entry_price"] - 100.0) < 1e-9
    # single-contact (cb_now) at the first contact -> misses the best entry
    single = god_solve("LONG", mv, [10], 120.0)
    assert single["ok"]
    assert single["entry_decision_index"] == 10
    assert abs(single["entry_price"] - 112.0) < 1e-9
    # the two diverge -> proves cb_now loses the God-mode best entry
    assert full["entry_decision_index"] != single["entry_decision_index"]


def test_cursor_boundary_contact_at_cursor_eligible():
    # Step 12: previous exit = 20 -> cursor = 21. A candidate contact exactly at
    # the cursor (bar 21) MUST remain eligible; a contact strictly before it
    # (bar 20) must be skipped. This is the off-by-one (< cursor, not <= cursor)
    # guard in the sequential orchestration.
    assert _contact_is_consumed(20, 21) is True   # before cursor -> consumed
    assert _contact_is_consumed(21, 21) is False  # at cursor -> eligible
    assert _contact_is_consumed(22, 21) is False  # after cursor -> eligible


# --------------------------------------------------------------------------- #
# Sequential candidate discovery DECOUPLED from the static-event window
# --------------------------------------------------------------------------- #
def test_sequential_candidate_decoupled_from_event_window():
    # A structure is present in per_bar at bars 348/349/350 only. The static
    # builder might place the event's start_bar at 350 while the structure is
    # ALREADY in proximity at 348/349. The sequential scan MUST anchor the
    # candidate at the EARLIEST per_bar appearance (348), not at the static
    # event start (350). This is the precise decoupling contract.
    n = 360
    S = "SR|m15|0|346|7705.0|7694.0|75.0"
    per_bar = [None] * n
    for t in (348, 349, 350):
        per_bar[t] = [(S, "RESISTANCE", 0.0, 0.0)]
    c, sid = _scan_next_candidate(per_bar, cursor=347, eligible_sids={S}, n=n)
    assert sid == S
    assert c == 348  # NOT 350 -- proof of decoupling
    # full legal contact set from cursor forward
    cb = [t for t in range(c, n)
          if per_bar[t] is not None and any(x[0] == S for x in per_bar[t])]
    assert cb == [348, 349, 350]


def test_sequential_best_entry_trade2_region():
    # Mirrors the diagnosed Trade-2 region: candidate structure contacts
    # 348/349/350, frozen LONG target 7773; fills open[349]=7705, open[350]=7700,
    # open[351]=7741; target touched later. God-mode MUST pick the max-profit
    # entry (decision 349 -> fill 350 -> 7700 -> +73), NOT the late first-contact
    # 350 (fill 351 -> 7741 -> +32).
    n = 400
    opens = [7700.0] * n
    opens[349] = 7705.0
    opens[350] = 7700.0
    opens[351] = 7741.0
    highs = [7700.0] * n
    highs[360] = 7773.0  # first LONG target touch
    lows = [7700.0] * n
    mv = make_mv(opens, highs, lows, unit_starts=np.array([0], dtype=np.int64))
    sol = god_solve("LONG", mv, [348, 349, 350], 7773.0)
    assert sol["ok"]
    assert sol["exit_reason"] == R_TARGET
    assert sol["entry_decision_index"] == 349
    assert sol["entry_fill_index"] == 350
    assert abs(sol["entry_price"] - 7700.0) < 1e-9
    assert abs(sol["utility"] - 73.0) < 1e-6


def test_ag_trade2_decoupled_best_entry():
    # End-to-end regression for the diagnosed Trade-2 region. After decoupling
    # candidate discovery from the static event window, the candidate structure
    # SR|m15|0|346|7705.0|7694.0|75.0 MUST be eligible from its first per_bar
    # appearance (348/349/350) and the God-mode solver MUST select the best
    # entry (decision 349 -> fill 350 -> 7700), not the previously-reported
    # late first-contact entry at ~7741.
    res = run_god_oracle_v4("AG", max_bars=AG_MAX_BARS)
    canon = [r for r in res["records"] if r["canonical_oracle_trade"]]
    t2 = next((r for r in canon
               if r["structure_id"] == "SR|m15|0|346|7705.0|7694.0|75.0"), None)
    assert t2 is not None, "decoupled candidate structure must form a trade"
    assert t2["oracle_direction"] == "LONG"
    assert t2["best_entry_decision_index"] == 349
    assert t2["best_entry_fill_index"] == 350
    assert abs(t2["best_entry_price"] - 7700.0) < 1e-6
    assert abs(t2["target_price"] - 7773.0) < 1e-6


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
