"""Tests for FUTURE-R8-M15-STRUCTURAL-RENEWAL-DATASET-V1 (§48 / §49)."""

import inspect

import numpy as np
import pandas as pd
import pytest

import research.liquidity_oracle_atlas.structural_renewal_dataset_v1 as R


# --------------------------------------------------------------------------- #
# synthetic fixtures                                                            #
# --------------------------------------------------------------------------- #
def synth_case(*, side=+1, n=12, highs=None, lows=None, closes=None,
               opens=None, segments=None, favorable=None, adverse=None,
               entry=1, end=None, atr=1.0):
    opens = np.full(n, 100.0) if opens is None else np.asarray(opens, float)
    closes = np.full(n, 100.0) if closes is None else np.asarray(closes, float)
    highs = closes.copy() if highs is None else np.asarray(highs, float)
    lows = closes.copy() if lows is None else np.asarray(lows, float)
    segments = (np.zeros(n, np.int64) if segments is None
                else np.asarray(segments, np.int64))
    if favorable is None:
        favorable = 102.0 if side > 0 else 98.0
    if adverse is None:
        adverse = 98.0 if side > 0 else 102.0
    end = n - 1 if end is None else end
    return dict(
        entry_idx=np.array([entry], np.int64),
        end_idx=np.array([end], np.int64),
        side=np.array([float(side)]),
        favorable_boundary=np.array([favorable], float),
        adverse_boundary=np.array([adverse], float),
        high=highs, low=lows, segment=segments,
        entry_segment=np.array([segments[entry]], np.int64),
        eligible=np.array([True]),
        open_px=opens, close_px=closes,
        entry_open=np.array([opens[entry]], float),
        atr0=np.array([atr], float))


def scan(c):
    return R.scan_first_structural_event(
        entry_idx=c["entry_idx"], end_idx=c["end_idx"], side=c["side"],
        favorable_boundary=c["favorable_boundary"],
        adverse_boundary=c["adverse_boundary"], high=c["high"], low=c["low"],
        segment=c["segment"], entry_segment=c["entry_segment"],
        eligible=c["eligible"])


def label(c, end=None):
    step, code = scan(c)
    return R.episode_label(
        event_step=step, event_code=code, entry_idx=c["entry_idx"],
        end_idx_H=np.array([c["end_idx"][0] if end is None else end], np.int64),
        side=c["side"], entry_open=c["entry_open"], atr0=c["atr0"],
        open_px=c["open_px"], close_px=c["close_px"], segment=c["segment"],
        entry_segment=c["entry_segment"])


# --------------------------------------------------------------------------- #
# §48.1-5 event classes                                                        #
# --------------------------------------------------------------------------- #
def test_48_1_long_favorable_touched_first():
    highs = np.full(12, 100.0); highs[3] = 102.5
    lows = np.full(12, 100.0)
    c = synth_case(side=+1, highs=highs, lows=lows, favorable=102.0, adverse=98.0)
    step, code = scan(c)
    assert step[0] == 2 and code[0] == R.EVENT_FAV


def test_48_2_long_adverse_touched_first():
    lows = np.full(12, 100.0); lows[3] = 97.5
    c = synth_case(side=+1, lows=lows, favorable=102.0, adverse=98.0)
    step, code = scan(c)
    assert step[0] == 2 and code[0] == R.EVENT_ADV


def test_48_3_long_both_touched_same_bar_is_both():
    highs = np.full(12, 100.0); highs[3] = 102.5
    lows = np.full(12, 100.0); lows[3] = 97.5
    c = synth_case(side=+1, highs=highs, lows=lows)
    step, code = scan(c)
    assert step[0] == 2 and code[0] == R.EVENT_BOTH
    assert R.EVENT_CLASS_NAME[code[0]] == "BOTH_SAME_BAR"


def test_48_4_short_mirror_all_three():
    # SHORT: favorable = support (low <= F), adverse = resistance (high >= A)
    lows = np.full(12, 100.0); lows[3] = 97.5
    c = synth_case(side=-1, lows=lows, favorable=98.0, adverse=102.0)
    assert scan(c)[1][0] == R.EVENT_FAV

    highs = np.full(12, 100.0); highs[3] = 102.5
    c = synth_case(side=-1, highs=highs, favorable=98.0, adverse=102.0)
    assert scan(c)[1][0] == R.EVENT_ADV

    highs = np.full(12, 100.0); highs[3] = 102.5
    lows = np.full(12, 100.0); lows[3] = 97.5
    c = synth_case(side=-1, highs=highs, lows=lows,
                   favorable=98.0, adverse=102.0)
    assert scan(c)[1][0] == R.EVENT_BOTH


def test_48_5_neither_touched_is_none():
    c = synth_case(side=+1)
    step, code = scan(c)
    assert step[0] == -1 and code[0] == R.EVENT_NONE
    assert label(c)["event_class"][0] == "NONE"


# --------------------------------------------------------------------------- #
# §48.6-9 bracket eligibility                                                   #
# --------------------------------------------------------------------------- #
def test_48_6_no_support_long_is_ineligible():
    sup_top = np.array([np.nan]); res_bottom = np.array([102.0])
    f, a = R.structural_barriers(np.array([True]), sup_top, res_bottom)
    g, l, el = R.bracket_metrics(np.array([1.0]), np.array([100.0]), f, a,
                                 np.array([1.0]))
    assert not el[0] and not np.isfinite(l[0])


def test_48_7_no_resistance_long_is_ineligible():
    f, a = R.structural_barriers(np.array([True]), np.array([98.0]),
                                 np.array([np.nan]))
    g, l, el = R.bracket_metrics(np.array([1.0]), np.array([100.0]), f, a,
                                 np.array([1.0]))
    assert not el[0] and not np.isfinite(g[0])


def test_48_8_g_le_zero_long_is_ineligible():
    # resistance at/below close -> no structural reward
    f, a = R.structural_barriers(np.array([True]), np.array([98.0]),
                                 np.array([99.0]))
    g, l, el = R.bracket_metrics(np.array([1.0]), np.array([100.0]), f, a,
                                 np.array([1.0]))
    assert g[0] <= 0 and not el[0]


def test_48_9_l_le_zero_long_is_ineligible():
    # support at/above close -> no structural risk budget
    f, a = R.structural_barriers(np.array([True]), np.array([101.0]),
                                 np.array([102.0]))
    g, l, el = R.bracket_metrics(np.array([1.0]), np.array([100.0]), f, a,
                                 np.array([1.0]))
    assert l[0] <= 0 and not el[0]


def test_structural_barriers_are_the_near_edges():
    il = np.array([True, False])
    f, a = R.structural_barriers(il, np.array([98.0, 98.0]),
                                 np.array([102.0, 102.0]))
    assert f[0] == 102.0 and a[0] == 98.0      # LONG
    assert f[1] == 98.0 and a[1] == 102.0      # SHORT


def test_log_rr_is_log_difference_not_clipped_ratio():
    g = np.array([2.0, 1.0, -1.0, np.nan])
    l = np.array([1.0, 2.0, 1.0, 1.0])
    out = R._log_rr(g, l)
    assert out[0] == pytest.approx(np.log(2.0))
    assert out[1] == pytest.approx(-np.log(2.0))
    assert np.isnan(out[2]) and np.isnan(out[3])


# --------------------------------------------------------------------------- #
# §48.10-14 fill / horizon semantics                                            #
# --------------------------------------------------------------------------- #
def test_48_10_real_next_open_gap_is_retained():
    opens = np.full(12, 100.0); opens[4] = 93.0      # adverse gap at fill bar
    highs = np.full(12, 100.0); highs[3] = 102.5
    c = synth_case(side=+1, opens=opens, highs=highs)
    lab = label(c)
    assert lab["renewal_executable"][0]
    assert lab["episode_exit_price"][0] == pytest.approx(93.0)
    assert lab["episode_return_atr"][0] == pytest.approx(-7.0)


def test_48_11_event_on_horizon_end_has_no_next_open_renewal():
    highs = np.full(12, 100.0); highs[5] = 102.5
    c = synth_case(side=+1, highs=highs, end=5)
    lab = label(c)
    assert lab["event_observed"][0]
    assert not lab["renewal_executable"][0]
    assert lab["episode_exit_price"][0] == pytest.approx(c["close_px"][5])


def test_48_12_hard_segment_boundary_stops_the_scan():
    seg = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], np.int64)
    highs = np.full(12, 100.0); highs[4] = 102.5      # beyond the segment break
    c = synth_case(side=+1, highs=highs, segments=seg)
    step, code = scan(c)
    assert step[0] == -1 and code[0] == R.EVENT_NONE


def test_48_13_td1_cannot_see_a_td3_event():
    lows = np.full(12, 100.0); lows[6] = 97.5
    c = synth_case(side=+1, lows=lows, end=11)
    step, code = scan(c)
    assert step[0] == 5
    lab_td1 = label(c, end=3)
    lab_td3 = label(c, end=7)
    assert not lab_td1["event_observed"][0]
    assert lab_td1["event_class"][0] == "NONE"
    assert lab_td3["event_observed"][0]
    assert lab_td3["event_class"][0] == "ADVERSE_FIRST"


def test_48_14_td3_cannot_see_a_td5_only_event():
    lows = np.full(12, 100.0); lows[9] = 97.5
    c = synth_case(side=+1, lows=lows, end=11)
    step, _ = scan(c)
    assert step[0] == 8
    assert not label(c, end=7)["event_observed"][0]
    assert label(c, end=11)["event_observed"][0]


# --------------------------------------------------------------------------- #
# §48.15 reference / production differential (real frames)                      #
# --------------------------------------------------------------------------- #
def _real_symbol_labels(symbol="AG", n_bars=None):
    st = R.load_symbol_state(symbol)
    n = st.n_bars
    if n_bars is not None:
        n = min(n, n_bars)
    views = R.build_side_views(st)
    ends = R.horizon_end_indices(st.trading_day, st.segment, st.n_bars, (1, 3, 5))
    idx = np.arange(n)
    entry_idx = idx + 1
    has = entry_idx < st.n_bars
    entry_idx_safe = np.where(has, entry_idx, 0)
    return st, views, ends, idx, entry_idx_safe, has


def test_48_15_one_scan_matches_literal_reference_rescans():
    st, views, ends, idx, entry_idx, has = _real_symbol_labels("AG", n_bars=400)
    n = len(idx)
    for name, is_long in (("LONG", True), ("SHORT", False)):
        block = slice(0, n) if is_long else slice(n, 2 * n)
        side = views["side"][block]
        fav = views["favorable"][block]
        adv = views["adverse"][block]
        elig = views["eligible"][block] & has
        end5 = ends[5][idx]
        step, code = R.scan_first_structural_event(
            entry_idx=entry_idx, end_idx=end5, side=side,
            favorable_boundary=fav, adverse_boundary=adv, high=st.high,
            low=st.low, segment=st.segment,
            entry_segment=st.segment[entry_idx], eligible=elig)
        for k in (1, 3, 5):
            end_H = ends[k][idx]
            lab = R.episode_label(
                event_step=step, event_code=code, entry_idx=entry_idx,
                end_idx_H=end_H, side=side,
                entry_open=st.open[entry_idx], atr0=st.atr[entry_idx],
                open_px=st.open, close_px=st.close, segment=st.segment,
                entry_segment=st.segment[entry_idx])
            for i in range(0, n, 7):
                if not elig[i] or not has[i]:
                    assert not bool(lab["event_observed"][i])
                    continue
                ref = R.reference_episode(
                    entry_idx=int(entry_idx[i]), end_idx=int(end_H[i]),
                    side=float(side[i]), favorable=float(fav[i]),
                    adverse=float(adv[i]), high=st.high, low=st.low,
                    open_px=st.open, close_px=st.close, segment=st.segment,
                    entry_segment=int(st.segment[entry_idx[i]]))
                r_step = -1 if ref is None else ref["event_step"]
                r_code = R.EVENT_NONE if ref is None else ref["event_code"]
                p_step = int(step[i])
                p_obs = bool(lab["event_observed"][i])
                if r_step >= 0 and (entry_idx[i] + r_step) <= end_H[i]:
                    assert p_obs and p_step == r_step
                    assert int(code[i]) == int(r_code), (name, k, i)
                else:
                    assert not p_obs


# --------------------------------------------------------------------------- #
# §48.16 candidate clock                                                        #
# --------------------------------------------------------------------------- #
def test_48_16_candidate_clock_shifts_exactly_one_bar():
    cand = np.array([False, True, False, True, False], dtype=bool)
    bits = np.array([0, 3, 0, 5, 0], dtype=np.uint8)
    at_dec, at_bits = R.shift_candidate_clock(cand, bits)
    assert at_dec.tolist() == [True, False, True, False, False]
    assert at_bits.tolist() == [3, 0, 5, 0, 0]
    # last bar can never be a decision bar (no next bar to fill at)
    assert not at_dec[-1]


# --------------------------------------------------------------------------- #
# §48.17-20                                                                     #
# --------------------------------------------------------------------------- #
def test_48_17_both_is_never_arbitrarily_classified():
    highs = np.full(12, 100.0); highs[3] = 102.5
    lows = np.full(12, 100.0); lows[3] = 97.5
    c = synth_case(side=+1, highs=highs, lows=lows)
    _, code = scan(c)
    assert code[0] == R.EVENT_BOTH
    assert code[0] not in (R.EVENT_FAV, R.EVENT_ADV)
    assert label(c)["event_class"][0] == "BOTH_SAME_BAR"


def test_48_18_long_short_weights_sum_to_one_per_eligible_epoch():
    el_l = np.array([True, True, False, False, True])
    el_s = np.array([True, False, True, False, True])
    wl, ws = R._epoch_weights(el_l, el_s)
    assert np.allclose(wl + ws, np.array([1.0, 1.0, 1.0, 0.0, 1.0]))
    assert wl[0] == 0.5 and ws[0] == 0.5
    assert wl[1] == 1.0 and ws[1] == 0.0
    assert wl[2] == 0.0 and ws[2] == 1.0
    assert wl[3] == 0.0 and ws[3] == 0.0


def test_48_19_future_mutation_after_event_cannot_change_first_event():
    lows = np.full(12, 100.0); lows[3] = 97.5
    c = synth_case(side=+1, lows=lows)
    s1, c1 = scan(c)
    c2 = dict(c)
    c2["high"] = c["high"].copy(); c2["low"] = c["low"].copy()
    c2["high"][5:] *= 5.0
    c2["low"][5:] *= 0.1
    s2, c2_code = scan(c2)
    assert s1[0] == s2[0] and c1[0] == c2_code[0]
    assert label(c)["episode_exit_price"][0] == pytest.approx(
        label(c2)["episode_exit_price"][0])


def test_48_20_opp36_has_no_forbidden_future_field():
    assert R.N_OPP36 == 36
    assert len(R.OPP36) == 36
    for name in R.OPP36:
        low = name.lower()
        for tok in R.FORBIDDEN_FEATURE_TOKENS:
            assert tok not in low, (name, tok)
    assert "symbol" not in R.OPP36


def test_opp36_schema_sha_is_stable():
    assert len(R.opp36_schema_sha256()) == 64


# --------------------------------------------------------------------------- #
# split purity + orientation + real-frame gates                                 #
# --------------------------------------------------------------------------- #
def test_orient_wrapper_produces_36_and_matches_frozen_orientation():
    st = R.load_symbol_state("AG")
    views = R.build_side_views(st)
    assert views["X36"].shape == (2 * st.n_bars, 36)
    from research.liquidity_oracle_atlas.direction_gated_experts_v1 import (
        orient_struct33_router_side)
    is_long = views["is_long"]
    expect = orient_struct33_router_side(np.concatenate([st.X33, st.X33]),
                                         is_long)
    assert np.array_equal(views["X36"][:, :33], expect, equal_nan=True)


def test_split_purity_on_real_labels():
    from research.liquidity_oracle_atlas.direction_null_baseline_v1 import (
        build_frozen_split)
    split = build_frozen_split()
    _, _, _, lab = R.build_symbol_dataset("AG", split=split)
    t1 = np.datetime64(split["cal"]["cuts"][0], "ns")
    t2 = np.datetime64(split["cal"]["cuts"][1], "ns")
    end = np.datetime64(pd.Timestamp(split["cal"]["end"]).to_datetime64(), "ns")
    dt = lab["decision_time"].to_numpy("datetime64[ns]")
    lav = lab["label_available_time"].to_numpy("datetime64[ns]")
    m = lab["split"].to_numpy(object)
    assert ((dt[m == "train"] < t1) & (lav[m == "train"] < t1)).all()
    assert ((dt[m == "val"] >= t1) & (dt[m == "val"] < t2)
            & (lav[m == "val"] < t2)).all()
    assert ((dt[m == "test"] >= t2) & (lav[m == "test"] <= end)).all()
    # a row that crosses a boundary is dropped, never reassigned
    assert set(np.unique(m)) <= {"train", "val", "test"}


def test_real_symbol_performance_counters():
    R.reset_counters()
    R.build_symbol_dataset("SN")
    assert R.COUNTERS["environment_loads"] == 1
    assert R.COUNTERS["geometry_extract_calls"] == 1
    assert R.COUNTERS["candidate_gate_derivations"] == 1
    assert R.COUNTERS["feature_matrix_builds"] == 1
    assert R.COUNTERS["side_feature_builds"] == 1
    assert R.COUNTERS["structural_stream_scans"] == 1
    assert R.COUNTERS["direction_reruns"] == 0
    assert R.COUNTERS["candidate_python_loops"] == 0
    assert R.COUNTERS["side_python_loops"] == 0
    assert R.COUNTERS["horizon_path_rescans"] == 0


def test_no_candidate_or_side_python_loop_in_production_source():
    src = inspect.getsource(R.build_symbol_dataset)
    assert "for candidate in" not in src
    assert "for side in" not in src
    assert "for horizon in" not in src
    scan_src = inspect.getsource(R.scan_first_structural_event)
    assert scan_src.count("for h in range") == 1


def test_reference_episode_is_not_called_in_production():
    assert "reference_episode(" not in inspect.getsource(R.build_symbol_dataset)
