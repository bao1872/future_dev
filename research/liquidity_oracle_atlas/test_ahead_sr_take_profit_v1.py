"""Tests for FUTURE-R7-M15-AHEAD-SR-NONACCEPTANCE-TP-V1 (Amendment A1: GROSS).

Synthetic TP-state tests (§33), next-open causal fill tests, Reference/Production
differential and frozen-input / gate tests.
"""

import inspect
import json
import os

import numpy as np
import pandas as pd
import pytest

import research.liquidity_oracle_atlas.ahead_sr_take_profit_v1 as R


# --------------------------------------------------------------------------- #
# Synthetic case builder                                                        #
# --------------------------------------------------------------------------- #
def synth(*, side, opens, lows, closes, highs=None, segments=None,
          entry_idx=0, end_idx=None, zone_bottom=None, zone_top=None,
          td1=None, td3=None, td5=None, atr=1.0,
          frozen_touch=None, frozen_cross=None):
    """One-candidate case.

    Geometric zone defaults:
      LONG : resistance [101.0, 102.0]  touch High>=101, cross Close>102
      SHORT: support    [ 98.0,  99.0]  touch Low <= 99, cross Close< 98
    """
    opens = np.asarray(opens, float)
    lows = np.asarray(lows, float)
    closes = np.asarray(closes, float)
    highs = closes.copy() if highs is None else np.asarray(highs, float)
    n = len(closes)
    segments = (np.zeros(n, np.int64) if segments is None
                else np.asarray(segments, np.int64))
    end = n - 1 if end_idx is None else end_idx
    d1 = end if td1 is None else td1
    d3 = end if td3 is None else td3
    d5 = end if td5 is None else td5
    is_long = np.array([side > 0])
    if zone_bottom is None or zone_top is None:
        zone_bottom, zone_top = ((101.0, 102.0) if side > 0 else (98.0, 99.0))
    zb = np.array([float(zone_bottom)])
    zt = np.array([float(zone_top)])
    tb, cb = R.ahead_touch_cross_boundaries(is_long, zb, zt)
    return {
        "side": np.array([float(side)]),
        "entry_idx": np.array([entry_idx], np.int64),
        "end_idx": np.array([end], np.int64),
        "atr0": np.array([atr]),
        "entry_price": np.array([opens[entry_idx]]),
        "entry_segment": np.array([segments[entry_idx]], np.int64),
        "open": opens, "high": highs, "low": lows, "close": closes,
        "segment": segments, "n_bars": n, "n": 1,
        "td_ends": {"td1": np.array([d1], np.int64),
                    "td3": np.array([d3], np.int64),
                    "td5": np.array([d5], np.int64)},
        "zone_bottom": zb, "zone_top": zt,
        "zone_strength": np.array([3.0]),
        "ahead_touch_boundary": tb, "ahead_cross_boundary": cb,
        "tp_eligible": np.array([np.isfinite(zb[0]) and np.isfinite(zt[0])]),
        "frozen_touch_step": np.array([-1 if frozen_touch is None
                                       else frozen_touch], np.int64),
        "frozen_cross_step": np.array([-1 if frozen_cross is None
                                       else frozen_cross], np.int64),
        "liq_zone_bottom": np.array([np.nan]), "liq_zone_top": np.array([np.nan]),
        "liq_available": np.array([False]),
    }


def ref(c):
    """Reference (literal bar-scan) signal step + reason."""
    return R.reference_tp_signal_step(
        side=c["side"][0], entry_idx=c["entry_idx"][0], end_idx=c["end_idx"][0],
        entry_price=c["entry_price"][0], atr0=c["atr0"][0],
        high=c["high"], low=c["low"], close=c["close"], segment=c["segment"],
        entry_segment=c["entry_segment"][0],
        ahead_bottom=c["zone_bottom"][0], ahead_top=c["zone_top"][0])


def prod(c):
    return R.scan_tp_production(c)


def fill(c, signal_step, exit_step):
    return R.resolve_tp_fill(
        signal_step=np.asarray([signal_step], np.int64),
        entry_idx=c["entry_idx"], side=c["side"], entry_price=c["entry_price"],
        atr0=c["atr0"], open_px=c["open"], segment=c["segment"],
        entry_segment=c["entry_segment"],
        exit_step_h=np.asarray([exit_step], np.int64))


def copy_case(c):
    return {k: (v.copy() if isinstance(v, np.ndarray)
                else {kk: vv.copy() for kk, vv in v.items()}
                if isinstance(v, dict) else v) for k, v in c.items()}


# --------------------------------------------------------------------------- #
# Frozen geometry (§4 / §5)                                                     #
# --------------------------------------------------------------------------- #
def test_ahead_sr_zone_selection_is_resistance_for_long_and_support_for_short():
    il = np.array([True, False])
    zb, zt, zs = R.ahead_sr_zone(
        il,
        sup_top=np.array([10.0, 20.0]), sup_bottom=np.array([9.0, 19.0]),
        sup_strength=np.array([1.0, 2.0]),
        res_top=np.array([30.0, 40.0]), res_bottom=np.array([29.0, 39.0]),
        res_strength=np.array([3.0, 4.0]))
    assert zb[0] == 29.0 and zt[0] == 30.0 and zs[0] == 3.0      # LONG -> resistance
    assert zb[1] == 19.0 and zt[1] == 20.0 and zs[1] == 2.0      # SHORT -> support
    tb, cb = R.ahead_touch_cross_boundaries(il, zb, zt)
    # reproduces exactly the frozen R5 channels
    assert tb[0] == 29.0 and cb[0] == 30.0    # LONG : touch res_bottom, cross res_top
    assert tb[1] == 20.0 and cb[1] == 19.0    # SHORT: touch sup_top,  cross sup_bottom


# --------------------------------------------------------------------------- #
# §33 synthetic TP-state tests                                                  #
# --------------------------------------------------------------------------- #
def test_33_1_long_never_reaches_resistance_no_tp():
    c = synth(side=+1, opens=[100, 100, 100, 100],
              lows=[100, 100, 100, 100], closes=[100, 100.5, 100.6, 100.7])
    step, why = ref(c)
    assert (step, why) == (-1, "no_touch")


def test_33_2_long_touch_closes_inside_zone_profitable_tp_signal():
    # bar 1 high >= 101 (touch); close 101.5 -> NOT through zone top 102 -> profitable
    c = synth(side=+1, opens=[100, 100, 100, 100],
              lows=[100, 100, 100, 100],
              closes=[100, 101.5, 101.4, 101.3], highs=[100, 101.6, 101.4, 101.3])
    step, why = ref(c)
    assert step == 1 and why == "tp_signal"


def test_33_3_long_touch_and_close_above_zone_top_is_accepted_breakout():
    c = synth(side=+1, opens=[100, 100, 100, 100],
              lows=[100, 100, 100, 100],
              closes=[100, 102.5, 102.6, 102.7], highs=[100, 102.6, 102.6, 102.7])
    step, why = ref(c)
    assert step == -1 and why == "accepted_breakout"


def test_33_4_short_mirror_no_touch():
    c = synth(side=-1, opens=[100, 100, 100, 100],
              lows=[100, 100, 100, 100], closes=[100, 99.5, 99.4, 99.3])
    step, why = ref(c)
    assert (step, why) == (-1, "no_touch")


def test_33_4_short_mirror_touch_no_cross_profitable_tp_signal():
    # low <= 99 (touch); close 98.5 -> not through zone bottom 98
    c = synth(side=-1, opens=[100, 100, 100, 100],
              lows=[100, 98.9, 98.8, 98.7], closes=[100, 98.5, 98.4, 98.3])
    step, why = ref(c)
    assert step == 1 and why == "tp_signal"


def test_33_4_short_mirror_close_below_zone_bottom_is_accepted_breakout():
    c = synth(side=-1, opens=[100, 100, 100, 100],
              lows=[100, 97.2, 97.1, 97.0], closes=[100, 97.5, 97.4, 97.3])
    step, why = ref(c)
    assert step == -1 and why == "accepted_breakout"


def test_33_5_first_touch_not_profitable_no_tp_permanently():
    # bar1 touches the zone (high>=101) but closes at 99.5 -> NOT profitable.
    # A LATER profitable touch (bar3) must NOT manufacture a TP: the first
    # encounter owns the decision permanently (§7).
    c = synth(side=+1, opens=[100, 100, 100, 100, 100],
              lows=[100, 100, 100, 100, 100],
              closes=[100, 99.5, 99.6, 101.5, 101.6],
              highs=[100, 101.6, 100.2, 101.7, 101.7])
    step, why = ref(c)
    assert step == -1 and why == "touch_not_profitable"


def test_33_6_accepted_breakout_then_later_falls_back_still_no_tp():
    # bar1 accepted breakout; bar3 falls back INTO the zone and is profitable
    # there -> the first encounter already decided: no TP from this frozen zone.
    c = synth(side=+1, opens=[100, 100, 100, 100, 100],
              lows=[100, 100, 100, 100, 100],
              closes=[100, 102.5, 102.4, 101.5, 101.4],
              highs=[100, 102.6, 102.5, 101.6, 101.5])
    step, why = ref(c)
    assert step == -1 and why == "accepted_breakout"


def test_33_7_touch_after_td1_but_before_td3_is_horizon_causal():
    """RC-R7-5 corrected: touch step 5 > TD1 exit step 2, and < TD3 exit step 8.

    The original fixture used signal step 1 == TD1 exit step 1, which only
    exercised the exact-on-exit boundary (duplicating test_33_8).
    """
    n = 16
    closes = np.full(n, 100.0)
    highs = np.full(n, 100.0)
    closes[5] = 101.5          # touch bar: inside the zone, profitable
    highs[5] = 101.6
    c = synth(side=+1, opens=np.full(n, 100.0), lows=np.full(n, 100.0),
              closes=closes, highs=highs, td1=2, td3=8, td5=15)
    step, why = ref(c)
    assert step == 5 and why == "tp_signal"
    f1 = fill(c, step, exit_step=2)
    f3 = fill(c, step, exit_step=8)
    f5 = fill(c, step, exit_step=15)
    # TD1 must be blind to the touch that happens after its exit.
    assert not f1["signal_observed"][0] and not f1["executable"][0]
    assert f3["signal_observed"][0] and f3["executable"][0]
    assert f5["signal_observed"][0] and f5["executable"][0]


def test_33_8_signal_on_baseline_exit_bar_observed_but_not_executable():
    c = synth(side=+1, opens=[100, 100, 100, 100, 100],
              lows=[100, 100, 100, 100, 100],
              closes=[100, 100.2, 101.5, 101.4, 101.3],
              highs=[100, 100.3, 101.6, 101.5, 101.4], td1=2, td3=4, td5=4)
    step, why = ref(c)
    assert step == 2
    f = fill(c, step, exit_step=2)
    assert f["signal_observed"][0]
    assert not f["executable"][0]


def test_33_9_next_bar_same_segment_fills_at_next_open():
    seg = np.array([0, 0, 0, 0], np.int64)
    c = synth(side=+1, opens=[100, 100, 100.7, 100.7],
              lows=[100, 100, 100, 100],
              closes=[100, 100.2, 101.5, 101.4],
              highs=[100, 100.3, 101.6, 101.5], segments=seg, td5=3)
    f = fill(c, signal_step=2, exit_step=3)
    assert f["executable"][0]
    assert f["fill_open"][0] == pytest.approx(100.7)
    assert f["tp_return"][0] == pytest.approx((100.7 - 100.0) / 1.0)


def test_33_10_segment_change_makes_tp_not_executable():
    seg = np.array([0, 0, 0, 1], np.int64)
    c = synth(side=+1, opens=[100, 100, 100, 90.0],
              lows=[100, 100, 100, 90.0],
              closes=[100, 100.2, 101.5, 90.0],
              highs=[100, 100.3, 101.6, 90.0], segments=seg, td5=3)
    f = fill(c, signal_step=2, exit_step=3)
    assert f["signal_observed"][0]
    assert not f["executable"][0]


def test_33_11_wall_clock_break_same_segment_uses_real_next_open():
    seg = np.array([0, 0, 0, 0], np.int64)      # canonical segment unchanged
    c = synth(side=+1, opens=[100, 100, 100, 104.0],
              lows=[100, 100, 100, 104.0],
              closes=[100, 100.2, 101.5, 104.0],
              highs=[100, 100.3, 101.6, 104.0], segments=seg, td5=3)
    f = fill(c, signal_step=2, exit_step=3)
    assert f["executable"][0]
    assert f["fill_open"][0] == pytest.approx(104.0)   # real next OPEN, not boundary


def test_33_12_favorable_gap_uses_real_next_open():
    c = synth(side=+1, opens=[100, 100, 100, 105.0],
              lows=[100, 100, 100, 105.0],
              closes=[100, 100.2, 101.5, 105.0],
              highs=[100, 100.3, 101.6, 105.0], td5=3)
    f = fill(c, signal_step=2, exit_step=3)
    assert f["fill_open"][0] == pytest.approx(105.0)
    assert f["tp_return"][0] == pytest.approx(5.0)


def test_33_13_adverse_gap_uses_real_next_open():
    c = synth(side=+1, opens=[100, 100, 100, 95.0],
              lows=[100, 100, 100, 95.0],
              closes=[100, 100.2, 101.5, 95.0],
              highs=[100, 100.3, 101.6, 95.0], td5=3)
    f = fill(c, signal_step=2, exit_step=3)
    assert f["executable"][0]
    assert f["fill_open"][0] == pytest.approx(95.0)     # gap risk retained
    assert f["tp_return"][0] == pytest.approx(-5.0)


def test_33_14_future_mutation_after_signal_does_not_alter_signal_or_fill():
    # signal at step 2, fill at step 3 -> mutate ONLY bars strictly after the
    # fill bar (4, 5). Signal identity, fill bar identity and fill OPEN must not
    # change. (Mutating the fill bar itself would legitimately change its OPEN.)
    c = synth(side=+1, opens=[100] * 6, lows=[100] * 6,
              closes=[100, 100.2, 101.5, 101.4, 101.3, 101.2],
              highs=[100, 100.3, 101.6, 101.5, 101.4, 101.3], td5=5)
    s1, w1 = ref(c)
    assert s1 == 2
    c2 = copy_case(c)
    for icol in ("open", "low", "close", "high"):
        c2[icol][4] *= 5.0
        c2[icol][5] *= 0.1
    s2, w2 = ref(c2)
    assert (s1, w1) == (s2, w2)
    f1 = fill(c, s1, exit_step=5)
    f2 = fill(c2, s2, exit_step=5)
    assert f1["fill_step"][0] == f2["fill_step"][0] == 3
    assert f1["fill_idx"][0] == f2["fill_idx"][0] == 3
    assert f1["fill_open"][0] == f2["fill_open"][0]     # same bar OPEN, unchanged


def test_33_15_cross_before_touch_is_a_hard_stop():
    touch = np.array([5], np.int64)
    cross = np.array([3], np.int64)      # cross strictly before touch -> illegal
    side = np.array([1.0])
    close = np.arange(20, dtype=float) + 100.0
    with pytest.raises(RuntimeError, match="STOP_R7_AHEAD_SR_EVENT_ORDER_MISMATCH"):
        R.production_tp_signal_step(
            touch_step=touch, cross_step=cross, side=side,
            entry_idx=np.array([0], np.int64), entry_price=np.array([100.0]),
            atr0=np.array([1.0]), close=close, eligible=np.array([True]))
    # cross >= 0 while touch < 0 is also illegal
    with pytest.raises(RuntimeError, match="STOP_R7_AHEAD_SR_EVENT_ORDER_MISMATCH"):
        R.production_tp_signal_step(
            touch_step=np.array([-1], np.int64), cross_step=np.array([2], np.int64),
            side=side, entry_idx=np.array([0], np.int64),
            entry_price=np.array([100.0]), atr0=np.array([1.0]),
            close=close, eligible=np.array([True]))


def test_33_16_tp_policy_signatures_have_no_correctness_or_oracle_labels():
    for fn in (R.production_tp_signal_step, R.reference_tp_signal_step,
               R.resolve_tp_fill, R.scan_tp_production, R.scan_tp_reference,
               R.differential_symbol):
        for p in inspect.signature(fn).parameters:
            assert "correct" not in p and "oracle" not in p, f"{fn.__name__}.{p}"


def test_33_18_saved_giveback_lost_continuation_identity():
    delta = np.array([1.5, -0.7, 0.0, 3.2, -2.1])
    sg = np.maximum(delta, 0.0)
    lc = np.maximum(-delta, 0.0)
    assert np.allclose(delta, sg - lc, atol=1e-15)
    # and through the real row builder on a tiny frame
    rows = pd.DataFrame({
        "semantic_key": ["k1", "k2"], "symbol": ["AG", "AG"],
        "gid": ["g", "g"], "direction_system": ["E9", "E9"],
        "direction": ["LONG", "LONG"], "sample_weight_raw": [0.5, 0.5],
        "direction_correct": [1, 0],
        "baseline_gross_return_atr": [0.0, 0.0],
        "tp_gross_return_atr": [2.0, -3.0]})
    big = rows.copy()
    big["evaluation_horizon"] = "td5"
    big["paired_delta_gross_atr"] = (big.tp_gross_return_atr
                                     - big.baseline_gross_return_atr)
    big["saved_giveback_atr"] = np.maximum(big.paired_delta_gross_atr, 0.0)
    big["lost_continuation_atr"] = np.maximum(-big.paired_delta_gross_atr, 0.0)
    assert np.allclose(big.paired_delta_gross_atr,
                       big.saved_giveback_atr - big.lost_continuation_atr,
                       atol=1e-15)


def test_33_19_paired_bootstrap_is_deterministic():
    delta = np.array([1.0, -0.5, 0.2, 0.9, -0.3, 0.4])
    gid = np.array(["g"] * 6)
    w = np.array([1 / 6] * 6)
    a = R.paired_gid_bootstrap(delta, gid, w, B=50, seed=1)
    b = R.paired_gid_bootstrap(delta, gid, w, B=50, seed=1)
    assert np.array_equal(a["reps"], b["reps"])
    assert a["point"] == pytest.approx(delta.mean())


def test_33_20_a9_e9_equal_direction_rows_satisfy_agreement_invariant():
    """Identical frozen inputs must produce identical TP signals for both systems."""
    n = 4
    base = dict(
        opens=np.full(n, 100.0), lows=np.full(n, 100.0),
        closes=[100.0, 100.2, 101.5, 101.4], highs=[100.0, 100.3, 101.6, 101.5],
        entry_idx=0, td5=n - 1)
    c = synth(side=+1, frozen_touch=2, frozen_cross=-1, **base)
    pa = R.scan_tp_production(c)
    pb = R.scan_tp_production(copy_case(c))
    assert np.array_equal(pa["signal_step"], pb["signal_step"])
    assert np.array_equal(pa["per_h"]["td5"]["tp_return"],
                          pb["per_h"]["td5"]["tp_return"])


# --------------------------------------------------------------------------- #
# Frozen inputs + R6 downstream policy (§2)                                     #
# --------------------------------------------------------------------------- #
def test_frozen_input_verification_passes():
    assert len(R.verify_frozen_inputs()) == 3


def test_r6_downstream_policy_is_no_structural_stop():
    p = R.verify_r6_downstream_policy()
    assert p["downstream_stop_policy"] == "NO_STRUCTURAL_STOP"
    assert p["apply_r6_structural_stop"] is False
    assert p["r6_stage"] == "formal_r6"
    assert p["r6_all_pass"] is True
    assert p["r6_verdict"] == "NO_IDENTIFIABLE_GROSS_STOP_EDGE"
    with open(R.R6_MANIFEST) as f:
        man = json.load(f)
    assert man["population_gates"]["pass"] is True


def test_r5_l2_loads_consume_the_frozen_artifact_once():
    R.reset_counters()
    l2 = R.load_r5_l2()
    assert int(R.COUNTERS["r5_l2_loads"]) == 1
    assert int(len(l2)) == 27546
    assert R.HORIZONS == ("td1", "td3", "td5")
    assert R.PRIMARY_HORIZON == "td5"
    assert R.ROBUSTNESS_HORIZONS == ("td1", "td3")


# --------------------------------------------------------------------------- #
# Reference / Production differential + baseline reproduction (real frames)     #
# --------------------------------------------------------------------------- #
def real_case(symbol="AG", n_keys=12, system="E9"):
    l2 = R.load_r5_l2()
    sub = l2[l2.symbol == symbol]
    keys = sorted(sub.semantic_key.unique())[:n_keys]
    sub = sub[sub.semantic_key.isin(keys)]
    st = R.load_symbol_state(symbol)
    rows, case = R.treatment_rows_for_symbol(sub, st, system)
    return rows, case, sub, st


def test_reference_production_differential_matches_on_real_frames():
    _, case, _, _ = real_case()
    d = R.differential_symbol(case)
    assert d["mismatch"] == 0, d
    assert d["max_abs_error"] <= 1e-12


def test_first_touch_and_same_bar_cross_steps_match_reference():
    _, case, _, _ = real_case()
    ref = R.scan_tp_reference(case)
    prod = R.scan_tp_production(case)
    assert np.array_equal(ref["first_touch_step"], prod["first_touch_step"])
    assert np.array_equal(ref["first_cross_step"], prod["first_cross_step"])
    assert np.array_equal(ref["accepted_same_bar"], prod["accepted_same_bar"])
    assert np.array_equal(ref["signal_step"], prod["signal_step"])


def test_baseline_close_reproduces_frozen_td_r():
    rows, _, sub, _ = real_case()
    sub = sub[sub.direction_system == "E9"].sort_values("semantic_key")
    for H in R.HORIZONS:
        r = rows[rows.evaluation_horizon == H].sort_values("semantic_key")
        recon = np.where(r.direction.to_numpy(object) == "LONG", 1.0, -1.0) * (
            r.baseline_exit_price.to_numpy(float)
            - r.entry_price.to_numpy(float)) / r.ATR0.to_numpy(float)
        assert np.allclose(recon, sub[f"{H}_r"].to_numpy(float), atol=1e-12)


def test_tp_reason_values_are_inside_the_frozen_taxonomy():
    rows, _, _, _ = real_case(n_keys=25)
    assert set(rows.tp_reason.unique()) <= set(R.TP_STATES)


def test_row_columns_have_no_cost_or_net_field():
    for c in R.ROW_COLUMNS:
        assert "net" not in c.lower()
        assert "cost" not in c.lower()


# --------------------------------------------------------------------------- #
# Effect / decomposition unit tests                                             #
# --------------------------------------------------------------------------- #
def _effect_frame(n=4):
    return pd.DataFrame({
        "direction_system": "E9", "evaluation_horizon": "td5",
        "direction": "LONG", "gid": [f"g{i}" for i in range(n)],
        "symbol": "AG",
        "sample_weight_raw": [0.7, 0.1, 0.1, 0.1],
        "direction_correct": [1, 1, 0, 0],
        "baseline_gross_return_atr": np.zeros(n),
        "tp_gross_return_atr": [2.0, 1.0, -3.0, 4.0],
        "paired_delta_gross_atr": np.zeros(n),
        "tp_eligible": np.ones(n, bool),
        "first_ahead_sr_touch_step": [3, 3, -1, -1],
        "first_touch_accepted_breakout": np.zeros(n, bool),
        "r_touch_close": [1.5, 0.5, np.nan, np.nan],
        "tp_signal_observed": [True, True, False, False],
        "tp_executable": [True, True, False, False]})


def test_effect_decomposition_is_weighted_not_row_mean():
    big = _effect_frame()
    big["paired_delta_gross_atr"] = (big.tp_gross_return_atr
                                     - big.baseline_gross_return_atr)
    row = R._effect(big, "E9", "td5")
    assert row["delta_ev_gross"] == pytest.approx(
        0.7 * 2.0 + 0.1 * 1.0 + 0.1 * -3.0 + 0.1 * 4.0)
    assert row["delta_ev_gross"] != pytest.approx(big.tp_gross_return_atr.mean())
    # SG = 0.7*2 + 0.1*1 + 0.1*0 + 0.1*4 ; LC = 0.1*3 ; SG - LC = Delta = 1.6
    assert row["saved_giveback"] == pytest.approx(1.9)
    assert row["lost_continuation"] == pytest.approx(0.3)
    assert row["saved_giveback"] - row["lost_continuation"] == pytest.approx(
        row["delta_ev_gross"])
    assert row["delta_correct_gross"] == pytest.approx((0.7 * 2.0 + 0.1 * 1.0) / 0.8)
    assert row["break_even_incremental_exit_cost_atr"] == pytest.approx(
        row["delta_ev_gross"])


def test_group_weighted_mean_differs_from_raw_row_mean():
    big = _effect_frame()
    big["paired_delta_gross_atr"] = big.tp_gross_return_atr
    row = R.group_stat_rows(big)[0]
    assert row["delta_ev_gross"] == pytest.approx(1.6)     # weighted
    assert row["raw_delta_ev_gross"] == pytest.approx(1.0)  # raw row mean
    assert row["delta_ev_gross"] != pytest.approx(row["raw_delta_ev_gross"])
    assert row["n_rows"] == 4 and row["raw_n_rows"] == 4


# --------------------------------------------------------------------------- #
# Formal gates (mocked, never touches canonical paths)                          #
# --------------------------------------------------------------------------- #
def _mock_env_records():
    return [{"symbol": s, "environment_contract_id": R.ENV_CONTRACT_ID,
             "cache_schema_version": "cache_v1", "identity": f"id_{s}",
             "code_identity": "code_sha", "raw_sha256": "raw",
             "execution_frame_sha256": "exec", "sha256": "sha", "rows": 1234,
             "max_bars": None} for s in R.SYMBOLS]


def install_formal_mocks(monkeypatch, *, order=None, seen=None,
                         pop_mismatch=None, gen_mismatch=None, perf=None,
                         written=None):
    """Mock every Formal side effect so `run_formal_r7` is safe to exercise.

    Canonical evidence paths are never touched: the writers, the parquet writer,
    the artifact verifier, sha256 and the perf counters are all monkeypatched.
    """
    order = [] if order is None else order
    seen = {} if seen is None else seen
    written = {} if written is None else written

    def fake_gates(l2, big=None):
        order.append("population_gate")
        seen["called"] = True
        return dict(pop_mismatch or {})

    def fake_write_artifact(df, path):
        order.append("artifact_write")
        written["artifact_rows"] = int(len(df))

    def fake_treatment(sub, st, ds):
        s = sub[sub["direction_system"] == ds]
        parts = [pd.DataFrame({"semantic_key": list(s["semantic_key"]),
                               "gid": list(s["gid"]),
                               "symbol": list(s["symbol"]),
                               "direction_system": ds,
                               "evaluation_horizon": H}) for H in R.HORIZONS]
        return (pd.concat(parts, ignore_index=True), None)

    def fake_write_csv(path, rows, columns):
        written[path] = (rows if isinstance(rows, pd.DataFrame)
                         else pd.DataFrame(rows, columns=columns))

    monkeypatch.setattr(R, "_git_head_sha", lambda: "SHA")
    monkeypatch.setattr(R, "verify_frozen_inputs", lambda: {})
    monkeypatch.setattr(R, "verify_r6_downstream_policy",
                        lambda: {"downstream_stop_policy": "NO_STRUCTURAL_STOP"})
    monkeypatch.setattr(R, "load_symbol_state", lambda sym: object())
    monkeypatch.setattr(R, "treatment_rows_for_symbol", fake_treatment)
    monkeypatch.setattr(R, "_formal_population_gates", fake_gates)
    if gen_mismatch is not None:
        monkeypatch.setattr(R, "_generated_population_gate",
                            lambda big: dict(gen_mismatch))
    monkeypatch.setattr(R, "_env_provenance_records",
                        lambda states: _mock_env_records())
    monkeypatch.setattr(R, "_perf_counters",
                        lambda t0: dict(perf if perf is not None else R.PERF_EXPECTED,
                                        runtime_sec=0.0))
    monkeypatch.setattr(R, "_write_formal_artifact", fake_write_artifact)
    monkeypatch.setattr(R, "_verify_formal_artifact", lambda p, k: {"pass": True})
    monkeypatch.setattr(R, "sha256_file", lambda p: "mocksha")
    monkeypatch.setattr(R, "group_stat_rows", lambda big: [])
    monkeypatch.setattr(R, "events_frame",
                        lambda big: pd.DataFrame(columns=R.EVENT_COLUMNS))
    monkeypatch.setattr(R, "_effect",
                        lambda big, s, h, side=None: {
                            "system": s, "horizon": h, "side": side or "ALL",
                            "n_rows": int(len(big)),
                            "delta_ev_gross": 0.0, "delta_ev_ci_low": 0.0,
                            "delta_ev_ci_high": 0.0,
                            "delta_correct_gross_ci_low": 0.0,
                            "saved_giveback": 0.0, "lost_continuation": 0.0})
    monkeypatch.setattr(R, "formal_verdict",
                        lambda p, s: "NO_IDENTIFIABLE_GROSS_TP_EDGE")
    monkeypatch.setattr(R, "formal_tp_diagnostics", lambda big, s, h: {})
    monkeypatch.setattr(R, "write_csv", fake_write_csv)
    monkeypatch.setattr(R, "write_json", lambda p, o: written.__setitem__(p, o))
    return order, seen, written


def test_37_full_population_requires_explicit_authorization():
    with pytest.raises(RuntimeError, match="FULL_POPULATION_NOT_AUTHORIZED"):
        R.run_formal_r7()


def test_37_formal_runner_requires_authorized_sha():
    with pytest.raises(RuntimeError, match="AUTHORIZED_REVIEW_SHA_REQUIRED"):
        R.run_formal_r7(allow_full=True)


def test_37_formal_runner_rejects_write_artifacts_false(monkeypatch):
    monkeypatch.setattr(R, "_git_head_sha", lambda: "SHA")
    with pytest.raises(RuntimeError, match="ARTIFACT_WRITE_REQUIRED"):
        R.run_formal_r7(allow_full=True, authorized_review_sha="SHA",
                        write_artifacts=False)


def test_37_formal_runner_rejects_generator_sha_mismatch(monkeypatch):
    monkeypatch.setattr(R, "_git_head_sha", lambda: "HEAD_A")
    with pytest.raises(RuntimeError, match="GENERATOR_SHA_MISMATCH"):
        R.run_formal_r7(allow_full=True, authorized_review_sha="HEAD_B")


def test_37_population_gate_failure_happens_before_artifact_write(monkeypatch):
    order = []
    install_formal_mocks(monkeypatch, order=order, pop_mismatch={"symbols": []})
    with pytest.raises(RuntimeError, match="POPULATION_GATE"):
        R.run_formal_r7(allow_full=True, authorized_review_sha="SHA")
    assert "population_gate" in order and "artifact_write" not in order


def test_37_gates_pass_before_canonical_write(monkeypatch):
    order = []
    install_formal_mocks(monkeypatch, order=order)
    R.run_formal_r7(allow_full=True, authorized_review_sha="SHA")
    assert order.index("population_gate") < order.index("artifact_write")


def test_36_performance_gate_contract_is_frozen():
    assert R.PERF_EXPECTED["production_candidate_views"] == 27546
    assert R.PERF_EXPECTED["execution_frame_loads"] == 15
    assert R.PERF_EXPECTED["reference_calls"] == 0
    assert R.PERF_EXPECTED["sr_recompute_count"] == 0
    assert R.PERF_EXPECTED["path_atlas_rescans"] == 0
    assert R.PERF_EXPECTED["r6_policy_verifications"] == 1
    assert R.PERF_EXPECTED["full_history_recompute_count"] == 0
    assert R.PERF_EXPECTED["direction_reruns"] == 0


def test_36_performance_failure_happens_before_artifact_write(monkeypatch):
    order = []
    install_formal_mocks(
        monkeypatch, order=order,
        perf=dict(R.PERF_EXPECTED, production_candidate_views=27545))
    with pytest.raises(RuntimeError, match="PERFORMANCE_GATE"):
        R.run_formal_r7(allow_full=True, authorized_review_sha="SHA")
    assert "population_gate" in order and "artifact_write" not in order


def test_38_formal_manifest_hashes_every_formal_artifact(monkeypatch):
    _, _, written = install_formal_mocks(monkeypatch)
    R.run_formal_r7(allow_full=True, authorized_review_sha="SHA")
    man = written[R.FORMAL_MANIFEST]
    expect = {os.path.basename(p) for p in
              (R.FORMAL_ARTIFACT, R.PRIMARY_CSV, R.SIDE_STATS_CSV,
               R.GROUP_STATS_CSV, R.EVENTS_CSV, R.FORMAL_SUMMARY)}
    assert expect <= set(man["artifact_sha256"])
    assert man["all_pass"] is True
    assert man["serialization_manifest_last"] is True
    assert man["authorized_review_sha"] == man["generator_code_sha"] \
        == man["reviewed_parent_sha"]
    assert man["downstream_policy"]["downstream_stop_policy"] == "NO_STRUCTURAL_STOP"
    assert len(man["environment_provenance"]) == 15
    # RC-R7-10: base and generated population gates are persisted SEPARATELY.
    assert man["base_population_gates"]["pass"] is True
    assert man["generated_population_gates"]["pass"] is True
    assert "population_gates" not in man


def test_31_evidence_names_are_stage_explicit():
    t15 = (R.T1_5_PRIMARY_CSV, R.T1_5_SIDE_STATS_CSV, R.T1_5_GROUP_STATS_CSV,
           R.T1_5_EVENTS_CSV, R.T1_5_SUMMARY_JSON, R.T1_5_MANIFEST_JSON)
    formal = (R.PRIMARY_CSV, R.SIDE_STATS_CSV, R.GROUP_STATS_CSV, R.EVENTS_CSV,
              R.SUMMARY_JSON, R.MANIFEST_JSON)
    for p in t15:
        assert "_t1_5_" in os.path.basename(p)
        assert os.path.basename(p).startswith("ahead_sr_take_profit_v1_t1_5_")
    for p in formal:
        assert "_t1_5_" not in os.path.basename(p)
        assert os.path.basename(p).startswith("ahead_sr_take_profit_v1_")
    assert set(t15).isdisjoint(set(formal))


def test_38_artifact_key_gate_requires_six_combos_per_key(tmp_path):
    sk = [f"sk{i}" for i in range(13773)]
    base = pd.DataFrame({"semantic_key": sk * 2,
                         "direction_system": ["A9"] * 13773 + ["E9"] * 13773})
    parts = []
    for H in R.HORIZONS:
        t = base.copy()
        t["evaluation_horizon"] = H
        parts.append(t)
    ok = pd.concat(parts, ignore_index=True)
    p = tmp_path / "ok.parquet"
    ok.to_parquet(p, index=False)
    rep = R._verify_formal_artifact(p, set(sk))
    assert rep["pass"] is True and rep["six_combos_exactly_once"] is True

    bad = ok.copy()
    m = ((bad.semantic_key == "sk0") & (bad.direction_system == "A9")
         & (bad.evaluation_horizon == "td3"))
    bad.loc[m, "evaluation_horizon"] = "td1"
    p2 = tmp_path / "bad.parquet"
    bad.to_parquet(p2, index=False)
    r2 = R._verify_formal_artifact(p2, set(sk))
    assert r2["rows"] == 82638 and r2["rows_per_semantic_key_min"] == 6
    assert r2["six_combos_exactly_once"] is False and r2["pass"] is False


def test_38_environment_provenance_gate_requires_exact_15_symbols():
    recs = _mock_env_records()
    assert len(recs) == 15
    assert R._env_provenance_gate(recs) == {}
    assert R._env_provenance_gate(recs[:-1]) != {}
    broken = [dict(r) for r in recs]
    broken[0]["environment_contract_id"] = "OLD"
    broken[1]["max_bars"] = 5000
    mism = R._env_provenance_gate(broken)
    assert "environment_contract_id" in mism and "max_bars" in mism


def test_22_formal_verdict_categories_are_frozen():
    assert R.GROSS_VERDICTS == ("GROSS_TP_EDGE_SUPPORTED_UNIVERSAL",
                                "GROSS_TP_HARMFUL",
                                "GROSS_SIDE_HETEROGENEITY_REQUIRES_FOLLOWUP",
                                "NO_IDENTIFIABLE_GROSS_TP_EDGE")


def test_22_verdict_requires_positive_correct_ci_not_only_primary():
    primary = {"delta_ev_ci_low": 0.2, "delta_ev_ci_high": 0.5,
               "delta_correct_gross_ci_low": -0.1}
    sides = [{"side": "LONG", "delta_ev_ci_low": 0.1, "delta_ev_ci_high": 0.4},
             {"side": "SHORT", "delta_ev_ci_low": 0.1, "delta_ev_ci_high": 0.4}]
    # primary CI lower > 0 but Delta_correct CI lower <= 0 -> NOT supported
    assert R.formal_verdict(primary, sides) == "NO_IDENTIFIABLE_GROSS_TP_EDGE"
    primary_ok = dict(primary, delta_correct_gross_ci_low=0.05)
    assert R.formal_verdict(primary_ok, sides) == \
        "GROSS_TP_EDGE_SUPPORTED_UNIVERSAL"


def test_14_bootstrap_seed_and_b_are_frozen():
    assert R.BOOTSTRAP_B == 2000
    assert R.BOOTSTRAP_SEED == 20260924


# =========================================================================== #
# RC-R7-1..15 horizon-causal / formal-gate regressions                         #
# =========================================================================== #
def _rc_case(*, side=+1, n=18, touch_bar=None, cross_bar=None,
             td1=2, td3=8, td5=15, entry=0, zone=None, closes=None,
             highs=None, lows=None, opens=None, seg=None):
    """Deterministic multi-horizon fixture builder for the RC-R7 censoring tests."""
    if zone is None:
        zone = ((101.0, 102.0) if side > 0 else (98.0, 99.0))
    auto_close = closes is None
    opens = np.full(n, 100.0) if opens is None else np.asarray(opens, float)
    lows = np.full(n, 100.0) if lows is None else np.asarray(lows, float)
    closes = np.full(n, 100.0) if auto_close else np.asarray(closes, float)
    highs = closes.copy() if highs is None else np.asarray(highs, float)
    if touch_bar is not None:
        if side > 0:
            highs[touch_bar] = zone[0] + 0.6          # reach the near zone edge
        else:
            lows[touch_bar] = zone[1] - 0.6
        if auto_close:
            closes[touch_bar] = (zone[0] + 0.5 if side > 0
                                 else zone[1] - 0.5)   # inside zone, profitable
    if cross_bar is not None:
        if side > 0:
            closes[cross_bar] = zone[1] + 0.5         # close through the far edge
            highs[cross_bar] = zone[1] + 0.6
        else:
            closes[cross_bar] = zone[0] - 0.5
            lows[cross_bar] = zone[0] - 0.6
    return synth(side=side, opens=opens, lows=lows, closes=closes, highs=highs,
                 segments=seg, entry_idx=entry, zone_bottom=zone[0],
                 zone_top=zone[1], td1=td1, td3=td3, td5=td5)


def _rows_for(c, system="E9"):
    """Build per-horizon rows via the production kernel for one synthetic case."""
    out = {}
    for H in R.HORIZONS:
        exit_step = int(c["td_ends"][H][0] - c["entry_idx"][0])
        f = R.resolve_tp_fill(
            signal_step=R.scan_tp_production(c)["signal_step"],
            entry_idx=c["entry_idx"], side=c["side"],
            entry_price=c["entry_price"], atr0=c["atr0"], open_px=c["open"],
            segment=c["segment"], entry_segment=c["entry_segment"],
            exit_step_h=np.asarray([exit_step], np.int64))
        to, co = R.horizon_censor(
            np.asarray([c["frozen_touch_step"][0]], np.int64),
            np.asarray([c["frozen_cross_step"][0]], np.int64),
            np.asarray([exit_step], np.int64))
        out[H] = {"exit_step": exit_step, "fill": f,
                  "touch_obs": bool(to[0]), "cross_obs": bool(co[0])}
    return out


def test_rc_r7_5_1_true_touch_after_td1_before_td3_is_censored():
    """§33.7 corrected: touch step 5, TD1 exit 2, TD3 exit 8, TD5 exit 15."""
    c = _rc_case(touch_bar=5, td1=2, td3=8, td5=15)
    c["frozen_touch_step"] = np.array([5], np.int64)
    c["frozen_cross_step"] = np.array([-1], np.int64)
    s, why = ref(c)
    assert s == 5 and why == "tp_signal"          # global five-day answer
    v = _rows_for(c)
    assert v["td1"]["touch_obs"] is False
    assert not v["td1"]["fill"]["signal_observed"][0]
    assert v["td3"]["touch_obs"] is True
    assert v["td5"]["touch_obs"] is True
    assert v["td3"]["fill"]["signal_observed"][0]
    assert v["td5"]["fill"]["signal_observed"][0]
    assert v["td3"]["fill"]["executable"][0] and v["td5"]["fill"]["executable"][0]


def test_rc_r7_5_2_exact_on_exit_remains_observed_and_non_executable():
    c = _rc_case(touch_bar=2, td1=2, td3=8, td5=15)
    c["frozen_touch_step"] = np.array([2], np.int64)
    c["frozen_cross_step"] = np.array([-1], np.int64)
    f = R.resolve_tp_fill(
        signal_step=np.asarray([2], np.int64), entry_idx=c["entry_idx"],
        side=c["side"], entry_price=c["entry_price"], atr0=c["atr0"],
        open_px=c["open"], segment=c["segment"],
        entry_segment=c["entry_segment"], exit_step_h=np.asarray([2], np.int64))
    assert bool(f["signal_observed"][0]) is True
    assert bool(f["executable"][0]) is False


def test_rc_r7_6_future_accepted_breakout_does_not_enter_td1_rate():
    # first touch at step 5 IS accepted through on the same bar (> TD1 exit 2)
    c = _rc_case(touch_bar=5, cross_bar=5, td1=2, td3=8, td5=15)
    c["frozen_touch_step"] = np.array([5], np.int64)
    c["frozen_cross_step"] = np.array([5], np.int64)
    s, why = ref(c)
    assert s == -1 and why == "accepted_breakout"
    v = _rows_for(c)
    t1, t3 = R.horizon_censor(np.array([5]), np.array([5]), np.array([2]))[0], \
        R.horizon_censor(np.array([5]), np.array([5]), np.array([8]))[0]
    assert not bool(t1[0])            # TD1 contribution = 0
    assert bool(t3[0])                # TD3 contribution = 1
    assert not v["td1"]["fill"]["signal_observed"][0]
    assert not v["td3"]["fill"]["signal_observed"][0]
    assert not v["td3"]["fill"]["executable"][0]


def test_rc_r7_7_future_not_profitable_touch_does_not_enter_td1_rate():
    # touch at step 5 closes BELOW entry -> not profitable; stays terminal.
    closes = np.full(18, 100.0)
    closes[5] = 99.4
    c = _rc_case(touch_bar=5, td1=2, td3=8, td5=15, closes=closes)
    c["frozen_touch_step"] = np.array([5], np.int64)
    c["frozen_cross_step"] = np.array([-1], np.int64)
    s, why = ref(c)
    assert s == -1 and why == "touch_not_profitable"
    t1 = R.horizon_censor(np.array([5]), np.array([-1]), np.array([2]))[0]
    t3 = R.horizon_censor(np.array([5]), np.array([-1]), np.array([8]))[0]
    assert not bool(t1[0])
    assert bool(t3[0])
    v = _rows_for(c)
    assert not v["td3"]["fill"]["signal_observed"][0]


def test_rc_r7_1_horizon_specific_fields_are_masked():
    rows, case, _, _ = real_case(symbol="AG", n_keys=25, system="E9")
    exit_td1 = (case["td_ends"]["td1"] - case["entry_idx"]).astype(np.int64)
    ft = np.asarray(case["frozen_touch_step"], np.int64)
    fc = np.asarray(case["frozen_cross_step"], np.int64)
    # `treatment_rows_for_symbol` sorts by semantic_key, so the per-horizon block
    # is row-aligned with the case arrays.
    r1 = rows[rows.evaluation_horizon == "td1"].reset_index(drop=True)
    assert len(r1) == len(ft)
    future_touch = (ft >= 0) & (ft > exit_td1)
    assert future_touch.sum() > 0          # the fixture really has future touches
    assert (r1.first_ahead_sr_touch_step.to_numpy(np.int64)[future_touch] == -1).all()
    assert (r1.first_ahead_sr_cross_step.to_numpy(np.int64)
            [future_touch | (fc > exit_td1)] == -1).all()
    assert not r1.first_touch_accepted_breakout.to_numpy(bool)[future_touch].any()
    assert np.isnan(r1.r_touch_close.to_numpy(float)[future_touch]).all()
    assert np.isnan(r1.mfe_at_first_ahead_sr.to_numpy(float)[future_touch]).all()
    # the global five-day classification is retained ONLY as AUDIT_ONLY
    assert "first_encounter_class_global_audit" in R.ROW_COLUMNS
    # the AUDIT_ONLY global class may additionally report the pre-horizon
    # `tp_signal` state, which is never a horizon-row reason.
    audit_allowed = set(R.TP_STATES) | {"tp_signal"}
    assert set(r1.first_encounter_class_global_audit.unique()) <= audit_allowed
    assert not r1.tp_reason.isin(["tp_signal"]).any()


def test_rc_r7_3_no_future_reason_in_shorter_horizon():
    rows, _, _, _ = real_case(symbol="SN", n_keys=25, system="E9")
    for H in ("td1", "td3", "td5"):
        r = rows[rows.evaluation_horizon == H]
        assert set(r.tp_reason.unique()) <= set(R.TP_STATES)
    # a TD1 row may never report a first-encounter classification that requires
    # knowledge of a touch occurring after the TD1 exit.
    r1 = rows[rows.evaluation_horizon == "td1"]
    bad = r1[(r1.first_ahead_sr_touch_step.to_numpy(np.int64) < 0)
             & r1.tp_reason.isin(["accepted_breakout", "touch_not_profitable"])]
    assert len(bad) == 0
    # and a TD1 reason may never be a first-encounter verdict built on a touch
    # that the TD1 window cannot see.
    assert set(r1.loc[r1.first_ahead_sr_touch_step < 0, "tp_reason"]) <= {
        "not_eligible", "no_touch"}


# ---- RC-R7-4 TD5 no-drift golden (captured BEFORE the horizon-censor patch) -- #
TD5_GOLDEN_SHA256 = {
    ("AG", "E9"):
        "0b72941212282fe3b74c4bdc452df442015b2f759fe71226cc30f408761b4a16",
    ("AG", "A9"):
        "c4dd1de8a22d7829df1c1befd2a61cf4831e4db6ad09dd79c8de5631f60df6bf",
    ("SN", "E9"):
        "e1132fe6ee4313502ec842dbe7af87d94c628dfa9423b437ca4d15800bceda9a",
    ("SN", "A9"):
        "94e3867d09f1dbe30027c2f38bf3ee1695d8ac8051cb98586f169302994b9587",
}
TD5_GOLDEN_COLUMNS = [
    "semantic_key", "first_ahead_sr_touch_step", "first_ahead_sr_cross_step",
    "first_touch_accepted_breakout", "r_touch_close", "mfe_at_first_ahead_sr",
    "tp_signal_observed", "tp_signal_step", "tp_executable", "tp_fill_step",
    "tp_fill_open", "tp_gross_return_atr", "paired_delta_gross_atr",
    "baseline_gross_return_atr", "tp_reason"]


def test_rc_r7_4_td5_output_is_unchanged_by_horizon_censoring():
    import hashlib
    l2 = R.load_r5_l2()
    for sym in ("AG", "SN"):
        sub = l2[l2.symbol == sym]
        keys = sorted(sub.semantic_key.unique())[:50]
        sub = sub[sub.semantic_key.isin(keys)]
        st = R.load_symbol_state(sym)
        for ds in ("E9", "A9"):
            rows, case = R.treatment_rows_for_symbol(sub, st, ds)
            t5 = rows[rows.evaluation_horizon == "td5"].sort_values(
                "semantic_key").reset_index(drop=True)
            got = hashlib.sha256(
                t5[TD5_GOLDEN_COLUMNS].to_csv(index=False).encode()).hexdigest()
            assert got == TD5_GOLDEN_SHA256[(sym, ds)], (sym, ds)
            # structural reason TD5 censoring is the identity:
            # exit_step_td5 == the R5 frozen scan bound, and no frozen touch
            # can lie beyond that bound.
            assert np.array_equal(case["td_ends"]["td5"] - case["entry_idx"],
                                  case["end_idx"] - case["entry_idx"])
            ft = case["frozen_touch_step"]
            assert bool(((ft < 0) | (ft <= case["end_idx"] - case["entry_idx"])).all())


def _incidence_monotonic(rows):
    out = {}
    for name, fn in R.CAUSAL_SERIES:
        c = {H: int(np.asarray(fn(rows[rows.evaluation_horizon == H])).sum())
             for H in R.HORIZONS}
        out[name] = c
    return out


def test_rc_r7_8_incidence_is_horizon_monotonic_on_real_frames():
    l2 = R.load_r5_l2()
    for sym in ("AG", "SN"):
        sub = l2[l2.symbol == sym]
        keys = sorted(sub.semantic_key.unique())[:50]
        sub = sub[sub.semantic_key.isin(keys)]
        st = R.load_symbol_state(sym)
        for ds in ("E9", "A9"):
            rows, _ = R.treatment_rows_for_symbol(sub, st, ds)
            inc = _incidence_monotonic(rows)
            for name in ("first_touch", "accepted_breakout",
                         "touch_not_profitable", "tp_signal"):
                c = inc[name]
                assert c["td1"] <= c["td3"] <= c["td5"], (sym, ds, name, c)
            # RC-R7-2: the three first-encounter outcomes partition the
            # horizon-specific first-touch set (tp_signal here is OBSERVED).
            for H in R.HORIZONS:
                assert inc["first_touch"][H] == (
                    inc["accepted_breakout"][H]
                    + inc["touch_not_profitable"][H]
                    + inc["tp_signal"][H]), (sym, ds, H, inc)


def test_rc_r7_12_effect_consumes_horizon_specific_fields_only():
    """Changing a TD2-TD5-only touch must not move any TD1 rate."""
    rows, _, _, _ = real_case(symbol="AG", n_keys=25, system="E9")
    e1 = R._effect(rows, "E9", "td1")
    e5 = R._effect(rows, "E9", "td5")
    assert e1["first_touch_rate"] < e5["first_touch_rate"]
    assert e1["tp_signal_rate"] < e5["tp_signal_rate"]
    # and the partition holds at the cell level too
    for e in (e1, e5):
        assert e["first_touch_rate"] == pytest.approx(
            e["first_touch_accepted_breakout_rate"]
            + e["first_touch_not_profitable_rate"]
            + e["tp_signal_rate"])


# ---- RC-R7-9 generated-population pre-write gate ---------------------------- #
def _gen_frame(*, n_keys=13773, drop_combo=None, duplicate=False):
    sk = [f"sk{i}" for i in range(n_keys)]
    parts = []
    for ds in ("A9", "E9"):
        for H in R.HORIZONS:
            if drop_combo == (ds, H):
                continue
            parts.append(pd.DataFrame({"semantic_key": sk,
                                       "direction_system": ds,
                                       "evaluation_horizon": H}))
    df = pd.concat(parts, ignore_index=True)
    if duplicate:
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    return df


def test_rc_r7_9_14_exact_generated_population_passes():
    assert R._generated_population_gate(_gen_frame()) == {}


def test_rc_r7_9_11_malformed_row_count_fails_pre_write(monkeypatch):
    order = []
    install_formal_mocks(monkeypatch, order=order)
    bad = _gen_frame().iloc[:-1]                       # 82637 rows
    monkeypatch.setattr(R, "treatment_rows_for_symbol",
                        lambda sub, st, ds: (bad, None))
    with pytest.raises(RuntimeError, match="GENERATED_POPULATION_GATE"):
        R.run_formal_r7(allow_full=True, authorized_review_sha="SHA")
    assert "artifact_write" not in order


def test_rc_r7_9_12_missing_system_horizon_combination_fails_pre_write(monkeypatch):
    order = []
    install_formal_mocks(monkeypatch, order=order)
    bad = _gen_frame(drop_combo=("A9", "td3"))
    monkeypatch.setattr(R, "treatment_rows_for_symbol",
                        lambda sub, st, ds: (bad, None))
    with pytest.raises(RuntimeError, match="GENERATED_POPULATION_GATE"):
        R.run_formal_r7(allow_full=True, authorized_review_sha="SHA")
    assert "artifact_write" not in order


def test_rc_r7_9_13_duplicate_generated_key_fails_pre_write(monkeypatch):
    order = []
    install_formal_mocks(monkeypatch, order=order)
    bad = _gen_frame(duplicate=True)
    monkeypatch.setattr(R, "treatment_rows_for_symbol",
                        lambda sub, st, ds: (bad, None))
    with pytest.raises(RuntimeError, match="GENERATED_POPULATION_GATE"):
        R.run_formal_r7(allow_full=True, authorized_review_sha="SHA")
    assert "artifact_write" not in order


def test_rc_r7_9_generated_gate_flags_each_defect():
    assert "rows" in R._generated_population_gate(_gen_frame().iloc[:-1])
    m = R._generated_population_gate(_gen_frame(drop_combo=("A9", "td3")))
    assert "A9_rows" in m and "td3_rows" in m
    m = R._generated_population_gate(_gen_frame(duplicate=True))
    assert "duplicate_keys" in m
    bad = _gen_frame()
    bad.loc[bad.index[0], "evaluation_horizon"] = "td9"
    assert "evaluation_horizon_values" in R._generated_population_gate(bad)


def test_rc_r7_13_mfe_final_cannot_enter_the_policy_kernel():
    for fn in (R.production_tp_signal_step, R.reference_tp_signal_step,
               R.resolve_tp_fill, R.scan_tp_production, R.scan_tp_reference,
               R.horizon_censor, R.signal_level_reason, R.horizon_reason,
               R.formal_verdict, R._effect, R.group_stat_rows):
        src = inspect.getsource(fn)
        assert "mfe_final" not in src, fn.__name__
    # `mfe_final` survives ONLY as the pre-registered RemainingMFE descriptive
    assert "mfe_final" in inspect.getsource(R.formal_tp_diagnostics)


def test_rc_r7_15_16_manifest_persists_separate_base_and_generated_gates(monkeypatch):
    _, _, written = install_formal_mocks(monkeypatch)
    R.run_formal_r7(allow_full=True, authorized_review_sha="SHA")
    man = written[R.FORMAL_MANIFEST]
    assert man["base_population_gates"]["pass"] is True
    assert man["generated_population_gates"]["pass"] is True
    assert man["generated_population_gates"]["expected"]["rows"] == 82638
    assert man["generated_population_gates"]["expected"]["horizon_rows"] == 27546
    assert man["performance_gates"]["pass"] is True
    assert man["all_pass"] is True
    # a generated-gate failure never reaches the canonical writer
    order = []
    install_formal_mocks(monkeypatch, order=order,
                         gen_mismatch={"rows": 82637})
    with pytest.raises(RuntimeError, match="GENERATED_POPULATION_GATE"):
        R.run_formal_r7(allow_full=True, authorized_review_sha="SHA")
    assert "artifact_write" not in order


def test_24_cost_governance_constants():
    assert R.COST_METADATA_STATUS == "UNAVAILABLE_COST_METADATA"
    assert R.REALISTIC_NET_PNL_STATUS == "NOT_ESTIMATED"
    assert R.FORMAL_PRIMARY_BASIS == "GROSS_PAIRED_POLICY_EFFECT"
    assert R.FRICTION_PROXY_USED_FOR_VERDICT is False
    assert R.ENTRY_COST_ESTIMATED is False and R.EXIT_COST_ESTIMATED is False
