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
)
from research.liquidity_oracle_atlas.structural_god_oracle_reference_m15_v4 import (
    solve_direction_god_reference_v4,
    pick_directional_target_v4_ref,
    run_god_oracle_v4_reference,
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
    opens = [100, 100, 100, 98, 97, 99, 98]
    highs = list(opens)
    lows = list(opens)
    mv = make_mv(opens, highs, lows)
    cb = [1]
    long_s = god_solve("LONG", mv, cb, 200.0)   # unreachable -> early exit negative
    short_s = god_solve("SHORT", mv, cb, 95.0)  # early exit positive
    dec, lv, sv, side = decide_oracle_v4(long_s, short_s)
    assert dec == "SHORT"
    assert lv < 0 < sv


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


def test_t0_m_target_not_reached_early_exit():
    opens = [100, 100, 100, 105, 108, 103, 100]
    highs = list(opens)
    lows = list(opens)
    mv = make_mv(opens, highs, lows)
    cb = [1]
    long_s = god_solve("LONG", mv, cb, 200.0)  # unreachable
    assert long_s["exit_reason"] == "DP_EARLY_EXIT"
    assert long_s["exit_fill_index"] == 4
    assert long_s["utility"] == 8.0


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
    opens = [100, 100, 100, 105, 100, 100, 100, 100]
    highs = [100] * 8
    lows = [100] * 8
    mv = make_mv(opens, highs, lows)
    cb = [1]
    long_s = god_solve("LONG", mv, cb, None, e_bar=4)  # next event at bar 4
    assert long_s["ok"]
    assert long_s["exit_fill_index"] < 4  # cannot pass the next structural event
    assert long_s["exit_fill_index"] == 3


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
        # no target -> pure early-exit branch; direction picks opposite extremes
        rl = god_solve("LONG", m, cb, None)
        rs = god_solve("SHORT", m, cb, None)
        return (
            rl["exit_fill_index"], rl["utility"],
            rs["exit_fill_index"], rs["utility"],
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
    # session close 100, next session open 150 -> the cross-day entry would be a
    # huge (+50) fake profit. God-mode MUST NOT select it. A same-session
    # (+10) entry must win instead.
    opens = [100, 100, 110, 150, 200, 100]
    highs = list(opens)
    lows = list(opens)
    td = [0, 0, 0, 1, 1, 1]
    seg = [0, 0, 0, 0, 0, 0]
    unit_starts = np.array([0, 3], dtype=np.int64)
    mv = make_mv(opens, highs, lows, unit_starts=unit_starts)
    # legal same-session trade (+10) must win over the cross-day fake (+50)
    cb = [0, 2]
    long_s = god_solve("LONG", mv, cb, 999.0, trading_day=td, segment=seg)
    assert long_s["ok"]
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
    prod = run_god_oracle_v4("AG", max_bars=AG_MAX_BARS, event_limit=AG_EVENT_LIMIT)
    ref = run_god_oracle_v4_reference("AG", max_bars=AG_MAX_BARS, event_limit=AG_EVENT_LIMIT)
    rp = prod["records"]
    rr = ref["records"]
    assert len(rp) == len(rr)
    compared = 0
    mismatch = 0
    max_err = 0.0
    for a, b in zip(rp, rr):
        compared += 1
        for k in ("oracle_decision", "best_entry_fill_index", "exit_fill_index",
                  "target_price"):
            if (a.get(k) is None) != (b.get(k) is None):
                mismatch += 1
                break
            if a.get(k) is None:
                continue
            if isinstance(a[k], float):
                max_err = max(max_err, abs(a[k] - b[k]))
                if abs(a[k] - b[k]) > 1e-9:
                    mismatch += 1
                    break
            elif a[k] != b[k]:
                mismatch += 1
                break
    assert compared > 0
    assert mismatch == 0, f"full-run parity mismatch={mismatch} max_err={max_err}"
    # hard sanity from the production god-mode contract
    assert prod["meta"]["canonical_loss_count"] == 0
    assert prod["meta"]["self_target_count"] == 0
    assert prod["meta"]["valid_but_no_entry"] == 0
    # direction_margin_atr must be V_winner - V_loser >= 0 for canonical trades
    for r in rp:
        if r["canonical_oracle_trade"]:
            assert r["direction_margin_atr"] >= -1e-12, r["direction_margin_atr"]
            assert r["utility"] > 0 and r["tp_atr"] > 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
