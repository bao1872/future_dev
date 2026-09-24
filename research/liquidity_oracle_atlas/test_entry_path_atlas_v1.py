"""Tests for entry_path_atlas_v1 (FUTURE-R5-M15-ENTRY-PATH-ATLAS-V1).

Kernel Checkpoint gates:
  T0  - 15 synthetic/structural correctness tests + zone/checkpoint extras
  RC1 - entering a zone is NEVER a pierce-through
  RC3 - SR backstop and Liquidity-behind backstop are tracked independently
  RC4 - per-Candidate checkpoint availability (NaN, never inherited)
  RC5 - Reference vs Production over ALL research-consumed outputs
  RC6 - fill AND decision alignment negative controls
  RC7 - primary contrast cluster bootstrap (cluster dependence retained)
  RC8 - Oracle perturbation BEFORE anchor construction
  TP  - static audit, counters, scaling, peak RSS

The full path experiment (T1.5/T2) is NOT run here.
"""

import inspect
import json
import os
import resource
import tempfile
import time

import numpy as np
import pandas as pd
import pytest

import research.liquidity_oracle_atlas.entry_path_atlas_v1 as M


# --------------------------------------------------------------------------- #
# Builders                                                                     #
# --------------------------------------------------------------------------- #
def case(*, closes, highs=None, lows=None, side=1.0, p0=100.0, atr=1.0,
         sr_enter=99.0, sr_pierce=98.0, sr_reclaim=98.0,
         lb_enter=97.5, lb_pierce=96.5, lb_reclaim=96.5,
         ahead_sr_touch=102.0, ahead_sr_cross=103.0,
         ahead_liq_touch=104.0, ahead_liq_cross=105.0,
         segment=None, entry_idx=0, end_idx=None, td_ends=None):
    """One-Candidate synthetic case.

    LONG defaults: support zone [98.0(bottom), 99.0(top)]
      enter = low <= 99.0 ; pierce-through = low < 98.0 ; reclaim = close >= 98.0
    """
    closes = np.asarray(closes, dtype=np.float64)
    highs = closes if highs is None else np.asarray(highs, dtype=np.float64)
    lows = closes if lows is None else np.asarray(lows, dtype=np.float64)
    n = len(closes)
    segment = np.zeros(n, dtype=np.int64) if segment is None else segment
    return dict(
        entry_idx=np.array([entry_idx], dtype=np.int64),
        end_idx=np.array([n - 1 if end_idx is None else end_idx], dtype=np.int64),
        entry_price=np.array([p0], dtype=np.float64),
        atr0=np.array([atr], dtype=np.float64),
        side=np.array([side], dtype=np.float64),
        high=highs, low=lows, close=closes,
        segment=segment,
        entry_segment=np.array([segment[entry_idx]], dtype=np.int64),
        sr_enter=np.array([sr_enter]), sr_pierce=np.array([sr_pierce]),
        sr_reclaim=np.array([sr_reclaim]),
        lb_enter=np.array([lb_enter]), lb_pierce=np.array([lb_pierce]),
        lb_reclaim=np.array([lb_reclaim]),
        ahead_sr_touch=np.array([ahead_sr_touch]),
        ahead_sr_cross=np.array([ahead_sr_cross]),
        ahead_liq_touch=np.array([ahead_liq_touch]),
        ahead_liq_cross=np.array([ahead_liq_cross]),
        td_ends=td_ends,
    )


def multi(n_cand, *, n_bars, entry_idx, end_idx, td_ends=None, side=1.0,
          p0=100.0, highs=None, lows=None, closes=None, segment=None):
    closes = (np.full(n_bars, 100.0) if closes is None
              else np.asarray(closes, dtype=np.float64))
    highs = closes if highs is None else np.asarray(highs, dtype=np.float64)
    lows = closes if lows is None else np.asarray(lows, dtype=np.float64)
    segment = np.zeros(n_bars, dtype=np.int64) if segment is None else segment
    f = lambda v: np.full(n_cand, float(v))
    return dict(
        entry_idx=np.asarray(entry_idx, dtype=np.int64),
        end_idx=np.asarray(end_idx, dtype=np.int64),
        entry_price=f(p0), atr0=np.ones(n_cand),
        side=np.full(n_cand, float(side)),
        high=highs, low=lows, close=closes, segment=segment,
        entry_segment=np.asarray(segment)[np.asarray(entry_idx, dtype=np.int64)],
        sr_enter=f(99.0), sr_pierce=f(98.0), sr_reclaim=f(98.0),
        lb_enter=f(97.5), lb_pierce=f(96.5), lb_reclaim=f(96.5),
        ahead_sr_touch=f(102.0), ahead_sr_cross=f(103.0),
        ahead_liq_touch=f(104.0), ahead_liq_cross=f(105.0),
        td_ends=td_ends,
    )


def prod(c):
    return M.scan_paths_streaming(**c)


def ref(c):
    return M.scan_paths_reference(**c)


def diff_report(c):
    rep = M.diff_reference_vs_production(c)
    return rep


def _arr_eq(a, b):
    """NaN-aware array equality: equal shapes, equal finite masks, equal finite values."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    fa, fb = np.isfinite(a), np.isfinite(b)
    if not np.array_equal(fa, fb):
        return False
    return np.array_equal(a[fa], b[fb])


# --------------------------------------------------------------------------- #
# Real-data fixtures                                                           #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def e9_state():
    return M.materialize_e9_direction_state(save=True, verbose=False)


@pytest.fixture(scope="session")
def ag_state():
    return M.load_symbol_state("AG")


@pytest.fixture(scope="session")
def ag_anchors(e9_state, ag_state):
    return M.build_anchors_for_symbol(e9_state, ag_state)


@pytest.fixture(scope="session")
def ag_dual(e9_state, ag_state):
    sel = e9_state[e9_state["symbol"] == "AG"].sort_values("semantic_key").head(80)
    base = M.build_base_anchors(sel, ag_state)
    return base, M.run_symbol_paths_dual(ag_state, base)


# =========================================================================== #
# T0                                                                          #
# =========================================================================== #
def test_t0_1_long_mfe_mae_synthetic():
    c = case(closes=[100.5, 102.0], highs=[101.0, 103.0], lows=[99.0, 98.0], side=1.0)
    r = prod(c)
    assert r["mfe_final"][0] == pytest.approx(3.0)
    assert r["mae_final"][0] == pytest.approx(2.0)
    s = r["checkpoints"]["m15"]
    assert s["mfe"][0] == pytest.approx(1.0)
    assert s["mae"][0] == pytest.approx(1.0)
    assert s["r"][0] == pytest.approx(0.5)


def test_t0_2_short_exact_mirror():
    c = case(closes=[99.5, 98.0], highs=[101.0, 102.0], lows=[99.0, 97.0], side=-1.0,
             sr_enter=101.0, sr_pierce=102.0, sr_reclaim=102.0,
             lb_enter=103.0, lb_pierce=104.0, lb_reclaim=104.0)
    r = prod(c)
    assert r["mfe_final"][0] == pytest.approx(3.0)
    assert r["mae_final"][0] == pytest.approx(2.0)


def test_t0_3_hard_segment_truncates_path():
    seg = np.array([0, 0, 1, 1], dtype=np.int64)
    c = case(closes=[100.0] * 4, highs=[100.0, 100.0, 120.0, 120.0],
             lows=[100.0] * 4, side=1.0, segment=seg, end_idx=3)
    assert prod(c)["mfe_final"][0] == pytest.approx(0.0)


def test_t0_4_trading_day_boundary_truncates_path():
    n = 6
    highs = np.full(n, 100.0)
    lows = np.full(n, 100.0)
    highs[5] = 150.0
    lows[5] = 50.0
    c = case(closes=np.full(n, 100.0), highs=highs, lows=lows, side=1.0, end_idx=4)
    c["td_ends"] = {"td1": np.array([1]), "td3": np.array([3]), "td5": np.array([4])}
    r = prod(c)
    assert r["mfe_final"][0] == pytest.approx(0.0)
    assert r["mae_final"][0] == pytest.approx(0.0)


def test_t0_5_same_bar_reclaim():
    # LONG: pierce-through support (low<98.0) but close back at/above 98.0 same bar
    c = case(closes=[98.5], highs=[100.0], lows=[97.5], side=1.0)
    r = prod(c)
    assert r["sr_first_pierce"][0] == 0
    assert bool(r["sr_same_bar_reclaim"][0]) is True
    assert r["sr_first_reclaim"][0] == 0
    assert r["sr_bars_to_reclaim"][0] == 0


def test_t0_6_late_reclaim():
    # bar0 pierce+close invalid, bar1 still invalid, bar2 closes valid
    c = case(closes=[97.5, 97.0, 99.0], highs=[100.0] * 3,
             lows=[97.5, 96.0, 99.0], side=1.0)
    r = prod(c)
    assert r["sr_first_pierce"][0] == 0
    assert bool(r["sr_same_bar_reclaim"][0]) is False
    assert r["sr_first_reclaim"][0] == 2
    assert r["sr_bars_to_reclaim"][0] == 2


def test_t0_7_pierce_invalid_then_reclaim_is_late_not_same_bar():
    c = case(closes=[97.5, 97.0, 99.0], highs=[100.0] * 3,
             lows=[97.5, 96.0, 99.0], side=1.0)
    r = prod(c)
    assert r["sr_first_pierce"][0] == 0
    assert r["sr_first_reclaim"][0] == 2
    assert bool(r["sr_same_bar_reclaim"][0]) is False


def test_t0_8_failed_reclaim():
    # pierce -> reclaim at bar1 -> close back invalid at bar2
    c = case(closes=[97.5, 99.0, 97.0], highs=[100.0] * 3,
             lows=[97.5, 99.0, 96.0], side=1.0)
    r = prod(c)
    assert r["sr_first_reclaim"][0] == 1
    assert r["sr_first_failed_reclaim"][0] == 2


def test_t0_9_ahead_sr_touch_and_cross():
    c = case(closes=[101.0, 103.5], highs=[102.0, 104.0], lows=[100.0, 101.0],
             side=1.0)
    r = prod(c)
    assert r["first_ahead_sr_touch"][0] == 0
    assert r["first_ahead_sr_cross"][0] == 1
    assert r["mfe_at_first_ahead_sr"][0] == pytest.approx(2.0)


def test_t0_10_ahead_liquidity_touch_and_cross():
    c = case(closes=[103.0, 105.5], highs=[104.0, 106.0], lows=[100.0, 102.0],
             side=1.0)
    r = prod(c)
    assert r["first_ahead_liq_touch"][0] == 0
    assert r["first_ahead_liq_cross"][0] == 1


def test_t0_11_nan_zone_is_unavailable():
    c = case(closes=[100.0, 100.0], highs=[101.0] * 2, lows=[90.0, 90.0],
             side=1.0, sr_enter=np.nan, sr_pierce=np.nan, sr_reclaim=np.nan)
    r = prod(c)
    assert r["sr_first_pierce"][0] == -1
    assert r["sr_first_touch"][0] == -1
    assert bool(r["sr_break_continue"][0]) is False


def test_t0_12_oracle_fields_never_reach_realtime_state():
    params = inspect.signature(M.scan_paths_streaming).parameters
    for f in ("oracle_direction", "oracle_exit_fill_time", "entry_quality_atr",
              "direction_correct", "e9_teacher_exit_return_atr"):
        assert f not in params
    assert set(M.AUDIT_ONLY_FIELDS) & set(params) == set()


def test_t0_13_future_mutation_after_checkpoint_unchanged():
    n = 10
    highs = np.full(n, 101.0)
    lows = np.full(n, 99.0)
    c = case(closes=np.full(n, 100.0), highs=highs, lows=lows, side=1.0, end_idx=n - 1)
    before = prod(c)
    mut = dict(c)
    mut["high"] = highs.copy()
    mut["low"] = lows.copy()
    mut["high"][4:] = 160.0
    mut["low"][4:] = 40.0
    after = prod(mut)
    for name in ("m15", "h1"):
        assert np.allclose(before["checkpoints"][name]["mfe"],
                           after["checkpoints"][name]["mfe"], atol=1e-12)
        assert np.allclose(before["checkpoints"][name]["mae"],
                           after["checkpoints"][name]["mae"], atol=1e-12)


def test_t0_14_semantic_key_uniqueness():
    df = pd.DataFrame({
        "symbol": ["AG", "AG", "AU"],
        "oracle_trade_id": [1, 1, 1],
        "candidate_decision_time": ["t1", "t2", "t1"],
        "candidate_fill_time": ["f1", "f2", "f1"],
    })
    assert len(set(M.build_semantic_key(df).tolist())) == 3
    dup = pd.concat([df.iloc[[0]], df.iloc[[0]]]).reset_index(drop=True)
    assert len(set(M.build_semantic_key(dup).tolist())) == 1


def test_t0_15_gid_weight_sum_canonical(e9_state):
    g = e9_state.groupby("gid")["sample_weight_raw"].sum()
    assert np.allclose(g.to_numpy(), 1.0, atol=1e-9)


# =========================================================================== #
# RC1 -- entering a zone is NOT piercing through it                            #
# =========================================================================== #
def test_rc1_entering_support_zone_is_not_a_pierce():
    """LONG support zone [98.0, 99.0]: low=98.5 ENTERS but must NOT pierce."""
    c = case(closes=[98.6], highs=[100.0], lows=[98.5], side=1.0,
             sr_enter=99.0, sr_pierce=98.0, sr_reclaim=98.0)
    r = prod(c)
    assert r["sr_first_touch"][0] == 0, "entering the zone must register as touch"
    assert r["sr_first_pierce"][0] == -1, "entering a zone must NOT be a pierce-through"
    assert bool(r["sr_break_continue"][0]) is False


def test_rc1_pierce_through_support_zone():
    c = case(closes=[97.5], highs=[100.0], lows=[97.5], side=1.0,
             sr_enter=99.0, sr_pierce=98.0, sr_reclaim=98.0)
    r = prod(c)
    assert r["sr_first_touch"][0] == 0
    assert r["sr_first_pierce"][0] == 0


def test_rc1_short_entering_resistance_is_not_a_pierce():
    """SHORT resistance zone [101.0, 102.0]: high=101.5 enters but must NOT pierce."""
    c = case(closes=[101.4], highs=[101.5], lows=[100.0], side=-1.0,
             sr_enter=101.0, sr_pierce=102.0, sr_reclaim=102.0,
             lb_enter=103.0, lb_pierce=104.0, lb_reclaim=104.0)
    r = prod(c)
    assert r["sr_first_touch"][0] == 0
    assert r["sr_first_pierce"][0] == -1


def test_rc1_short_pierce_through_resistance():
    c = case(closes=[102.5], highs=[102.5], lows=[100.0], side=-1.0,
             sr_enter=101.0, sr_pierce=102.0, sr_reclaim=102.0,
             lb_enter=103.0, lb_pierce=104.0, lb_reclaim=104.0)
    r = prod(c)
    assert r["sr_first_pierce"][0] == 0


def test_rc1_zone_geometry_persisted_in_anchors(ag_state, ag_anchors):
    """Exact zone boundaries (not a collapsed price point) must be preserved."""
    for k in ("raw_sup_top", "raw_sup_bottom", "raw_res_top", "raw_res_bottom",
              "raw_liq_up_top", "raw_liq_up_bottom", "raw_liq_dn_top",
              "raw_liq_dn_bottom"):
        assert k in ag_anchors, f"missing zone boundary {k}"
    # backstop ENTER boundary differs from PIERCE boundary (i.e. it is a band)
    ok = np.isfinite(ag_anchors["sr_enter"]) & np.isfinite(ag_anchors["sr_pierce"])
    assert ok.sum() > 0
    assert not np.allclose(ag_anchors["sr_enter"][ok], ag_anchors["sr_pierce"][ok])


def test_rc2_liquidity_selection_matches_canonical_level(ag_state):
    """Selected liquidity level must reproduce canonical liq_*_level_price (ATR-free)."""
    a = ag_state.zone_audit
    assert a["liq_up_level_max_dev"] < 1e-9, a
    assert a["liq_dn_level_max_dev"] < 1e-9, a


def test_rc2_sr_selection_matches_canonical_scalar(ag_state):
    a = ag_state.zone_audit
    assert a["sr_support_price_max_dev"] < 1e-9, a
    assert a["sr_resistance_price_max_dev"] < 1e-9, a


# =========================================================================== #
# RC3 -- both structural backstops tracked independently                       #
# =========================================================================== #
def test_rc3_both_backstops_tracked_independently():
    # pierces the SR band but NOT the deeper liquidity-behind band
    c = case(closes=[97.0], highs=[101.0], lows=[97.0], side=1.0,
             sr_enter=99.0, sr_pierce=98.0, sr_reclaim=98.0,
             lb_enter=97.5, lb_pierce=96.5, lb_reclaim=96.5)
    r = prod(c)
    assert r["sr_first_pierce"][0] == 0
    assert r["lb_first_pierce"][0] == -1
    assert bool(r["sr_break_continue"][0]) is True
    assert bool(r["lb_break_continue"][0]) is False


def test_rc3_liquidity_behind_break_is_recorded():
    c = case(closes=[96.0], highs=[101.0], lows=[96.0], side=1.0,
             sr_enter=99.0, sr_pierce=98.0, sr_reclaim=98.0,
             lb_enter=97.5, lb_pierce=96.5, lb_reclaim=96.5)
    r = prod(c)
    assert r["lb_first_pierce"][0] == 0
    assert bool(r["lb_break_continue"][0]) is True


def test_rc3_backstop_keys_present():
    c = case(closes=[100.0], highs=[101.0], lows=[100.0], side=1.0)
    r = prod(c)
    for pfx in ("sr", "lb"):
        for k in ("first_touch", "first_pierce", "same_bar_reclaim", "first_reclaim",
                  "bars_to_reclaim", "first_failed_reclaim", "break_continue"):
            assert f"{pfx}_{k}" in r, f"missing {pfx}_{k}"


# =========================================================================== #
# RC4 -- per-Candidate checkpoint availability                                 #
# =========================================================================== #
def test_rc4_truncated_candidate_has_no_h4_but_reaching_one_does():
    n_bars = 20
    c = multi(2, n_bars=n_bars, entry_idx=[0, 0], end_idx=[2, 19])
    r = prod(c)
    assert np.isnan(r["checkpoints"]["h4"]["mfe"][0]), (
        "a Candidate truncated before 16 bars must have NO h4 observation")
    assert not np.isnan(r["checkpoints"]["h4"]["mfe"][1]), (
        "a Candidate reaching 16 bars must have h4 populated")
    assert not np.isnan(r["checkpoints"]["m15"]["mfe"][0])


def test_rc4_truncated_candidate_never_inherits_shorter_accumulator():
    n_bars = 20
    highs = np.full(n_bars, 101.0)
    lows = np.full(n_bars, 99.0)
    c = multi(2, n_bars=n_bars, entry_idx=[0, 0], end_idx=[2, 19],
              highs=highs, lows=lows)
    r = prod(c)
    assert np.isnan(r["checkpoints"]["h4"]["mae"][0])
    assert not np.isnan(r["checkpoints"]["h4"]["mae"][1])


def test_rc4_td_checkpoints_filled_per_candidate():
    n_bars = 10
    highs = np.full(n_bars, 101.0)
    lows = np.full(n_bars, 99.0)
    # Candidate A ends day1 at bar 0 ; Candidate B ends day1 at bar 3
    c = multi(2, n_bars=n_bars, entry_idx=[0, 0], end_idx=[9, 9],
              highs=highs, lows=lows,
              td_ends={"td1": np.array([0, 3]), "td3": np.array([2, 5]),
                       "td5": np.array([4, 7])})
    r = prod(c)
    assert not np.isnan(r["checkpoints"]["td1"]["mfe"][0])
    assert not np.isnan(r["checkpoints"]["td1"]["mfe"][1]), (
        "later Candidate must still get its own td1 (no global None latch)")
    assert not np.isnan(r["checkpoints"]["td5"]["mfe"][0])
    assert not np.isnan(r["checkpoints"]["td5"]["mfe"][1])


def test_rc4_segment_truncation_leaves_later_checkpoints_unavailable():
    n_bars = 20
    seg = np.zeros(n_bars, dtype=np.int64)
    seg[5:] = 1
    c = multi(1, n_bars=n_bars, entry_idx=[0], end_idx=[19], segment=seg)
    r = prod(c)
    assert not np.isnan(r["checkpoints"]["m15"]["mfe"][0])
    assert not np.isnan(r["checkpoints"]["h1"]["mfe"][0])
    assert np.isnan(r["checkpoints"]["h4"]["mfe"][0]), (
        "hard-segment truncation must leave h4 unavailable")


# =========================================================================== #
# RC5 -- expanded T1 differential                                              #
# =========================================================================== #
def test_t1_synthetic_long():
    c = case(closes=[100.5, 102.0, 101.0], highs=[101.0, 103.0, 102.0],
             lows=[99.0, 96.0, 95.0], side=1.0)
    rep = diff_report(c)
    assert rep["mismatch"] == 0, rep["first_mismatch"]
    assert rep["max_abs_error"] <= 1e-12


def test_t1_synthetic_short():
    c = case(closes=[99.5, 98.0, 99.0], highs=[101.0, 102.0, 104.0],
             lows=[99.0, 97.0, 96.0], side=-1.0,
             sr_enter=101.0, sr_pierce=102.0, sr_reclaim=102.0,
             lb_enter=103.0, lb_pierce=104.0, lb_reclaim=104.0)
    rep = diff_report(c)
    assert rep["mismatch"] == 0
    assert rep["max_abs_error"] <= 1e-12


def test_t1_boundary_segment_path():
    seg = np.array([0, 0, 1, 1], dtype=np.int64)
    c = case(closes=[100.0] * 4, highs=[101.0, 100.0, 120.0, 121.0],
             lows=[99.0, 96.0, 119.0, 119.0], side=1.0, segment=seg, end_idx=3)
    rep = diff_report(c)
    assert rep["mismatch"] == 0
    assert rep["max_abs_error"] <= 1e-12


def test_t1_multi_candidate_mixed_horizons():
    n_bars = 20
    c = multi(3, n_bars=n_bars, entry_idx=[0, 0, 1], end_idx=[2, 19, 18],
              td_ends={"td1": np.array([0, 3, 2]), "td3": np.array([1, 5, 4]),
                       "td5": np.array([2, 7, 6])})
    rep = diff_report(c)
    assert rep["mismatch"] == 0, rep["first_mismatch"]
    assert rep["max_abs_error"] <= 1e-12


def test_t1_real_ag_subset(ag_state, ag_anchors):
    def _slice_val(v, n):
        if isinstance(v, np.ndarray):
            return v[:n]
        if isinstance(v, dict):
            return {kk: (vv[:n] if isinstance(vv, np.ndarray) else vv)
                    for kk, vv in v.items()}
        return v
    sub = {k: _slice_val(v, 150) for k, v in ag_anchors.items() if k != "df"}
    sub.update(high=ag_state.high, low=ag_state.low, close=ag_state.close,
               segment=ag_state.segment)
    rep = diff_report(sub)
    assert rep["rows"] == 150
    assert rep["mismatch"] == 0, rep["first_mismatch"]
    assert rep["max_abs_error"] <= 1e-12
    assert rep["cells"] > 150 * len(M.DIFF_FIELDS), "checkpoints must be compared too"


# =========================================================================== #
# RC6 -- alignment negative controls                                           #
# =========================================================================== #
def test_rc6_fill_alignment_gate_fails_on_time_swap(ag_state, e9_state):
    bad = e9_state.copy()
    idx = np.flatnonzero(bad["symbol"].to_numpy(object) == "AG")
    shifted = (pd.to_datetime(bad.loc[idx[:5], "candidate_fill_time"])
               + pd.Timedelta(minutes=15))
    bad.loc[idx[:5], "candidate_fill_time"] = shifted
    with pytest.raises(RuntimeError) as e:
        M.build_anchors_for_symbol(bad, ag_state)
    assert "FILL_ALIGNMENT_MISMATCH" in str(e.value)


def test_rc6_decision_alignment_gate_fails_on_decision_time_swap(ag_state, e9_state):
    bad = e9_state.copy()
    idx = np.flatnonzero(bad["symbol"].to_numpy(object) == "AG")
    shifted = (pd.to_datetime(bad.loc[idx[:5], "candidate_decision_time"])
               + pd.Timedelta(minutes=15))
    bad.loc[idx[:5], "candidate_decision_time"] = shifted
    with pytest.raises(RuntimeError) as e:
        M.build_anchors_for_symbol(bad, ag_state)
    assert "DECISION_ALIGNMENT_MISMATCH" in str(e.value)


# =========================================================================== #
# RC7 -- primary contrast cluster bootstrap                                    #
# =========================================================================== #
def test_rc7_single_gid_with_both_classes_keeps_cluster_dependence():
    ps = np.array([1.0, 3.0, 0.0, 2.0])
    correct = np.array([1, 1, 0, 0], dtype=np.uint8)
    gid = np.array(["g1", "g1", "g1", "g1"])
    w = np.full(4, 0.25)
    r = M.delta_ps_cluster_bootstrap(ps, correct, gid, w, B=500, seed=3)
    assert abs(r["point"] - 1.0) < 1e-12      # correct mean 2 - wrong mean 1
    assert r["n_gids"] == 1
    assert abs(r["correct_mass"] - 0.5) < 1e-12
    assert abs(r["wrong_mass"] - 0.5) < 1e-12
    # one cluster -> every replicate is that cluster -> CI collapses onto the point
    assert abs(r["ci_low"] - 1.0) < 1e-12
    assert abs(r["ci_high"] - 1.0) < 1e-12


def test_rc7_multi_gid_ci_brackets_point():
    ps = np.array([3.0, 3.0, 1.0, 1.0, 1.0, 1.0])
    correct = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)
    gid = np.array(["a", "a", "b", "b", "c", "c"])
    w = np.full(6, 1 / 6)
    r = M.delta_ps_cluster_bootstrap(ps, correct, gid, w, B=1000, seed=5)
    assert r["n_gids"] == 3
    assert abs(r["point"] - 2.0) < 1e-12
    assert r["ci_low"] <= r["point"] <= r["ci_high"]


# =========================================================================== #
# RC8 -- Oracle perturbation BEFORE anchor construction                        #
# =========================================================================== #
def test_rc8_oracle_perturbation_before_anchor_construction(ag_state, e9_state):
    a1 = M.build_anchors_for_symbol(e9_state, ag_state)
    bad = e9_state.copy()
    idx = np.flatnonzero(bad["symbol"].to_numpy(object) == "AG")
    od = bad.loc[idx, "oracle_direction"].to_numpy(object)
    bad.loc[idx, "oracle_direction"] = np.where(od == "LONG", "SHORT", "LONG")
    bad.loc[idx, "oracle_entry_quality_atr"] = (
        -bad.loc[idx, "oracle_entry_quality_atr"].to_numpy(np.float64))
    bad.loc[idx, "e9_teacher_exit_return_atr"] = (
        -bad.loc[idx, "e9_teacher_exit_return_atr"].to_numpy(np.float64))
    bad.loc[idx, "direction_correct"] = (
        1 - bad.loc[idx, "direction_correct"].to_numpy())
    bad.loc[idx, "oracle_exit_fill_time"] = (
        pd.to_datetime(bad.loc[idx, "oracle_exit_fill_time"])
        + pd.Timedelta(days=3))

    a2 = M.build_anchors_for_symbol(bad, ag_state)
    for k in ("entry_idx", "end_idx", "entry_price", "atr0", "side", "entry_segment",
              "sr_enter", "sr_pierce", "sr_reclaim", "lb_enter", "lb_pierce",
              "lb_reclaim", "ahead_sr_touch", "ahead_sr_cross", "ahead_liq_touch",
              "ahead_liq_cross", "dec_idx"):
        assert _arr_eq(a1[k], a2[k]), (
            f"oracle perturbation changed real-time anchor field {k}")

    p1 = M.run_symbol_paths(ag_state, a1)
    p2 = M.run_symbol_paths(ag_state, a2)
    for f in M.DIFF_FIELDS:
        assert _arr_eq(p1[f], p2[f]), (
            f"oracle perturbation changed production output {f}")
    for name in M.CHECKPOINT_NAMES:
        for m in ("mfe", "mae", "r"):
            assert _arr_eq(p1["checkpoints"][name][m],
                           p2["checkpoints"][name][m])


# =========================================================================== #
# Negative controls A / C / D                                                  #
# =========================================================================== #
def test_negative_control_A_future_perturbation_after_checkpoint():
    n = 10
    highs = np.full(n, 101.0)
    lows = np.full(n, 99.0)
    c = case(closes=np.full(n, 100.0), highs=highs, lows=lows, side=1.0, end_idx=n - 1)
    before = prod(c)
    mut = dict(c)
    mut["high"] = highs.copy()
    mut["low"] = lows.copy()
    mut["high"][4:] = 160.0
    mut["low"][4:] = 40.0
    after = prod(mut)
    for name in ("m15", "h1"):
        assert np.allclose(before["checkpoints"][name]["mfe"],
                           after["checkpoints"][name]["mfe"], atol=1e-12)
        assert np.allclose(before["checkpoints"][name]["mae"],
                           after["checkpoints"][name]["mae"], atol=1e-12)
    assert after["mfe_final"][0] > before["mfe_final"][0]


def test_negative_control_C_altered_boundary_must_fail():
    c = case(closes=[97.5, 97.0, 99.0], highs=[100.0] * 3,
             lows=[97.5, 96.0, 99.0], side=1.0)
    assert diff_report(c)["mismatch"] == 0
    corrupt = dict(c)
    corrupt["sr_pierce"] = np.array([90.0])
    assert not np.array_equal(prod(corrupt)["sr_first_pierce"],
                              ref(c)["sr_first_pierce"]), (
        "differential must FAIL when a production boundary is altered")


def test_negative_control_D_checkpoint_alteration_must_fail():
    """Altering checkpoint availability (RC4) must be detectable."""
    n_bars = 20
    c = multi(2, n_bars=n_bars, entry_idx=[0, 0], end_idx=[2, 19])
    r = prod(c)
    assert np.isnan(r["checkpoints"]["h4"]["mfe"][0])
    corrupt = dict(c)
    corrupt["end_idx"] = np.array([19, 19], dtype=np.int64)
    r2 = prod(corrupt)
    assert not np.isnan(r2["checkpoints"]["h4"]["mfe"][0]), (
        "gate must catch a Candidate wrongly granted a 4h observation")


# =========================================================================== #
# TP performance gate                                                          #
# =========================================================================== #
def test_tp_static_audit():
    src = inspect.getsource(M.scan_paths_streaming)
    for tok in ("pd.concat", "for i in range", ".iloc[", "history[",
                "for candidate", "np.vstack"):
        assert tok not in src, f"forbidden token in production kernel: {tok}"
    assert "for step in range" in src
    assert "for i in range(n)" in inspect.getsource(M.scan_paths_reference)


def test_tp_counters(ag_state, ag_anchors):
    M.reset_counters()
    M.run_symbol_paths(ag_state, ag_anchors)
    c = M.COUNTERS
    assert c["full_history_recompute_count"] == 0
    assert c["reference_call_count_production"] == 0
    assert c["candidate_python_loop_count"] == 0
    assert c["hotloop_dataframe_concat_count"] == 0
    assert c["direction_chain_run_count"] <= 1


def test_tp_scaling_candidates_and_horizon():
    def timed(N, H, seed=0):
        c = M.make_synthetic_case(N, H, seed)
        t = time.perf_counter()
        prod(c)
        return time.perf_counter() - t

    H = 32
    t1 = timed(200, H)
    t4 = timed(800, H)
    assert t4 / max(t1, 1e-9) < 10.0, f"candidate scaling not linear: {t4 / t1:.2f}"
    h1 = timed(200, 32)
    h2 = timed(200, 64)
    assert h2 / max(h1, 1e-9) < 10.0, f"horizon scaling not linear: {h2 / h1:.2f}"


def test_tp_peak_rss_recorded(ag_state, ag_anchors):
    M.run_symbol_paths(ag_state, ag_anchors)
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 0


def test_env_completeness_gate(ag_state):
    assert ag_state.env_complete is True, ag_state.env_note


def test_cluster_bootstrap_opportunity_weighted():
    gid = np.array(["a", "a", "b", "b", "b"])
    v = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    m, lo, hi = M.cluster_bootstrap_gid(v, gid, B=200, seed=1)
    assert abs(m - 0.5) < 1e-12
    assert lo <= m <= hi


# =========================================================================== #
# A9 comparator framework (frozen alias of chain["A"])                         #
# =========================================================================== #
def test_a9_frozen_identity(e9_state):
    df = e9_state
    assert "a9_direction" in df.columns
    assert "a9_p_long" in df.columns
    assert "a9_direction_correct" in df.columns
    assert "a9_teacher_exit_return_atr" in df.columns
    # A9 is the frozen direct DTP9 router == router_te (hard invariant upstream)
    assert np.array_equal(df["a9_direction"].to_numpy(object),
                          df["router_direction"].to_numpy(object))
    # A9 and E9 are distinct systems: not identically labeled in general
    assert "e9_direction" in df.columns


def test_a9_e9_agreement_invariant(ag_state, e9_state):
    sel = e9_state[e9_state["symbol"] == "AG"].sort_values("semantic_key").head(80)
    base = M.build_base_anchors(sel, ag_state)
    dual = M.run_symbol_paths_dual(ag_state, base)
    agreement = (np.asarray(base["a9_direction"]) == np.asarray(base["e9_direction"]))
    # Hard gate: for agreement rows A9 and E9 path outputs must be identical.
    M.check_agreement_invariant(dual["A9"], dual["E9"], agreement, "AG")
    # sanity: the gate actually inspects a non-empty agreement set here
    assert agreement.any()


def test_a9_e9_disagreement_decomposition(ag_state, e9_state):
    sel = e9_state[e9_state["symbol"] == "AG"].sort_values("semantic_key")
    base = M.build_base_anchors(sel, ag_state)
    d = M.decompose_disagreement(base, "AG")
    assert d["agreement"] + d["disagreement"] == d["n"]
    assert d["e9_fix"] + d["e9_break"] == d["disagreement"]
    a9_c = np.asarray(base["a9_direction_correct"], dtype=np.uint8)
    e9_c = np.asarray(base["e9_direction_correct"], dtype=np.uint8)
    disag = (a9_c != e9_c)
    # binary oracle => exactly one of {A9, E9} correct on every disagreement row
    assert np.array_equal(a9_c[disag] + e9_c[disag], np.ones(int(disag.sum()), dtype=np.uint8))


# =========================================================================== #
# RC9 environment provenance (canonical keys, fail-closed)                      #
# =========================================================================== #
def test_rc9_provenance_canonical_keys(ag_state):
    pv = M.load_env_provenance("AG")
    for k in ("environment_contract_id", "cache_schema_version", "identity",
              "code_identity", "raw_sha256", "execution_frame_sha256", "max_bars"):
        assert k in pv, f"missing canonical provenance key: {k}"
    assert pv["max_bars"] is None, "full cache must report max_bars=None"
    assert pv["rows"] is not None


def test_rc9_provenance_smoke_cache_fails_closed(monkeypatch, tmp_path):
    fake = tmp_path / "r4_env_manifest.json"
    fake.write_text(json.dumps({
        "ZZ": {
            "environment_contract_id": "env-c", "cache_schema_version": 1,
            "identity": "id", "code_identity": "ci", "raw_sha256": "r",
            "execution_frame_sha256": "e", "max_bars": "2000",
            "sha256": "s", "rows": 10,
        }
    }))
    monkeypatch.setattr(M, "R4_ENV_MANIFEST", str(fake))
    with pytest.raises(RuntimeError):
        M.load_env_provenance("ZZ")


# =========================================================================== #
# Formal T2 (PRE-T2) curve-capture + simultaneous-band + invariants            #
# =========================================================================== #
def test_curve_h1_equals_m15_checkpoint(ag_dual):
    base, dual = ag_dual
    a9 = dual["A9"]
    assert _arr_eq(a9["curve_mfe"][:, 0], a9["checkpoints"]["m15"]["mfe"])
    assert _arr_eq(a9["curve_mae"][:, 0], a9["checkpoints"]["m15"]["mae"])
    assert _arr_eq(a9["curve_r"][:, 0], a9["checkpoints"]["m15"]["r"])


def test_curve_h4_equals_h1_checkpoint(ag_dual):
    base, dual = ag_dual
    a9 = dual["A9"]
    assert _arr_eq(a9["curve_mfe"][:, 3], a9["checkpoints"]["h1"]["mfe"])
    assert _arr_eq(a9["curve_mae"][:, 3], a9["checkpoints"]["h1"]["mae"])


def test_curve_h16_equals_h4_checkpoint_where_available(ag_dual):
    base, dual = ag_dual
    a9 = dual["A9"]
    cp = a9["checkpoints"]["h4"]
    valid = np.isfinite(cp["mfe"])
    assert _arr_eq(a9["curve_mfe"][valid, 15], cp["mfe"][valid])
    assert _arr_eq(a9["curve_mae"][valid, 15], cp["mae"][valid])


def test_truncated_candidate_nan_for_later_h():
    c = M.make_synthetic_case(6, 30, seed=3)
    c["end_idx"][0] = c["entry_idx"][0] + 3   # candidate 0 truncated at step 3
    sig = inspect.signature(M.scan_paths_streaming)
    out = M.scan_paths_streaming(**{k: c[k] for k in c if k in sig.parameters},
                                 capture_curve=True)
    assert not np.any(np.isnan(out["curve_mfe"][0, :4]))
    assert np.all(np.isnan(out["curve_mfe"][0, 4:]))


def test_future_bar_mutation_cannot_change_earlier_curve_cells():
    c = M.make_synthetic_case(4, 40, seed=11)
    sig = inspect.signature(M.scan_paths_streaming)
    kw = {k: c[k] for k in c if k in sig.parameters}
    o1 = M.scan_paths_streaming(**kw, capture_curve=True)
    c2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in c.items()}
    mut = int(c2["entry_idx"][0] + 12)
    c2["high"][mut] *= 2.0
    c2["low"][mut] *= 0.5
    kw2 = {k: c2[k] for k in c2 if k in sig.parameters}
    o2 = M.scan_paths_streaming(**kw2, capture_curve=True)
    # isolate candidate 0: the mutated bar is step 12 FOR candidate 0 only; earlier
    # cells (steps 0..11) must be byte-identical. (A global bar maps to different
    # steps across candidates, so we compare a single candidate, not all.)
    assert np.array_equal(o1["curve_mfe"][0, :12], o2["curve_mfe"][0, :12])
    assert np.array_equal(o1["curve_ps"][0, :12], o2["curve_ps"][0, :12])
    # and the mutated step (12) itself must differ
    assert not np.array_equal(o1["curve_mfe"][0, 12], o2["curve_mfe"][0, 12])


def test_reference_vs_production_curve_differential():
    c = M.make_synthetic_case(20, 60, seed=5)
    rep = M.diff_reference_vs_production(c)
    assert rep["mismatch"] == 0
    assert rep["curve_mismatch"] == 0
    assert rep["max_abs_error"] <= 1e-12


def test_a9_e9_agreement_whole_curve_invariant(ag_dual):
    base, dual = ag_dual
    agreement = (np.asarray(base["a9_direction"]) == np.asarray(base["e9_direction"]))
    idx = np.flatnonzero(agreement)
    assert idx.size > 0
    a9, e9 = dual["A9"], dual["E9"]
    for c in ("curve_mfe", "curve_mae", "curve_r", "curve_ps"):
        assert _arr_eq(a9[c][idx], e9[c][idx])
    M.check_agreement_invariant(a9, e9, agreement, "AG")


def test_a9_e9_disagreement_mfe_mae_mirror_every_h():
    a9_r = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    e9_r = -a9_r
    a9_mfe = np.array([[1.0, 1.1, 1.2, 1.3, 1.4, 1.5]])
    e9_mae = a9_mfe
    a9_mae = np.array([[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    e9_mfe = a9_mae
    a9_ps = a9_mfe - a9_mae
    e9_ps = -a9_ps
    a9 = {"curve_mfe": a9_mfe, "curve_mae": a9_mae, "curve_r": a9_r, "curve_ps": a9_ps}
    e9 = {"curve_mfe": e9_mfe, "curve_mae": e9_mae, "curve_r": e9_r, "curve_ps": e9_ps}
    dis = np.array([True])
    M.check_disagreement_mirror(a9, e9, dis, "SYN")
    bad = {"curve_mfe": a9_mfe.copy(), "curve_mae": a9_mae.copy(),
           "curve_r": a9_r.copy(), "curve_ps": a9_ps.copy()}
    bad["curve_mfe"][0, 0] = 999.0
    with pytest.raises(RuntimeError):
        M.check_disagreement_mirror(bad, e9, dis, "SYN")


def test_a9_e9_disagreement_signed_return_negation_every_h(ag_dual):
    base, dual = ag_dual
    dis = (np.asarray(base["a9_direction"]) != np.asarray(base["e9_direction"]))
    idx = np.flatnonzero(dis)
    if idx.size == 0:
        pytest.skip("no disagreement rows in AG subset")
    a9, e9 = dual["A9"], dual["E9"]
    assert _arr_eq(a9["curve_r"][idx], -e9["curve_r"][idx])
    assert _arr_eq(a9["curve_ps"][idx], -e9["curve_ps"][idx])


def test_a9_e9_availability_masks_identical(ag_dual):
    base, dual = ag_dual
    a9, e9 = dual["A9"], dual["E9"]
    assert np.array_equal(np.isfinite(a9["curve_ps"]), np.isfinite(e9["curve_ps"]))


def test_per_system_gid_weight_remains_one(e9_state):
    g = e9_state.groupby("gid")["sample_weight_raw"].sum().to_numpy(np.float64)
    assert np.allclose(g, 1.0, atol=1e-9)


def test_whole_gid_curve_bootstrap_cluster_dependence():
    rng = np.random.default_rng(0)
    n = 60
    gid = np.array([f"g{i // 4}" for i in range(n)])
    correct = rng.integers(0, 2, n).astype(bool)
    val = rng.normal(size=15)
    value = (np.tile(val[:, None], (1, 4)).reshape(-1, 1)
             + rng.normal(0, 1e-9, (n, 1)))
    value = np.repeat(value, 10, axis=1)
    ug, nc, dc, nw, dw = M.build_group_curve_sufficient_stats(
        value, correct, gid, np.ones(n))
    point, reps1, _, _ = M.bootstrap_delta_curve(nc, dc, nw, dw, B=300, seed=20260924)
    _, reps2, _, _ = M.bootstrap_delta_curve(nc, dc, nw, dw, B=300, seed=20260924)
    assert np.array_equal(reps1, reps2, equal_nan=True)  # deterministic under frozen seed
    # point matches the analytic whole-gid aggregation (cluster-respecting)
    analytic = M._safe_div(nc.sum(0), dc.sum(0)) - M._safe_div(nw.sum(0), dw.sum(0))
    assert np.allclose(point, analytic, atol=1e-12, equal_nan=True)


def test_simultaneous_band_deterministic_frozen_seed():
    rng = np.random.default_rng(1)
    val = rng.normal(size=(50, 30))
    correct = rng.integers(0, 2, 50).astype(bool)
    gid = np.array([f"g{i // 3}" for i in range(50)])
    ug, nc, dc, nw, dw = M.build_group_curve_sufficient_stats(
        val, correct, gid, np.ones(50))
    _, r1, vc1, vw1 = M.bootstrap_delta_curve(nc, dc, nw, dw, B=200, seed=20260924)
    _, r2, _, _ = M.bootstrap_delta_curve(nc, dc, nw, dw, B=200, seed=20260924)
    assert np.array_equal(r1, r2, equal_nan=True)
    band = M.simultaneous_band(np.nanmean(r1, axis=0), r1, vc1, vw1)
    assert band["lower"].shape[0] == 30
    finite = np.isfinite(band["lower"])
    assert np.all(band["upper"][finite] >= band["lower"][finite] - 1e-12)


def test_formal_runner_has_no_best_h_selection():
    src = inspect.getsource(M.run_formal_t2)
    for tok in ("argmax", "argmin", "best_h", "select_primary", "best primary",
                "optimal_h", "best bar"):
        assert tok not in src, f"forbidden token present in formal runner: {tok}"


def test_production_scan_signature_excludes_oracle_fields():
    sig = inspect.signature(M.scan_paths_streaming)
    for p in sig.parameters:
        assert "oracle" not in p
    for f in M.AUDIT_ONLY_FIELDS:
        assert f not in sig.parameters


def test_formal_runner_one_dual_scan_per_symbol_and_no_reference():
    orig_ref = M.scan_paths_reference
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("REFERENCE_CALLED")
    M.scan_paths_reference = boom
    try:
        M.reset_counters()
        M.run_formal_t2(symbols=("AG",), n_subset=30, population="small", verbose=False)
    finally:
        M.scan_paths_reference = orig_ref
    assert calls["n"] == 0, "formal runner must not call the Reference kernel"
    assert M.COUNTERS["path_scan_count"] == 1
    assert M.COUNTERS["direction_chain_run_count"] == 1
    assert M.COUNTERS["candidate_python_loop_count"] == 0
    assert M.COUNTERS["full_history_recompute_count"] == 0
    with pytest.raises(RuntimeError):
        M.run_formal_t2(symbols=("AG",), population="full")


def test_curve_artifact_key_uniqueness(ag_dual):
    base, dual = ag_dual
    chunk = M.curve_chunk_from_dual(base, dual, "AG")
    assert set(M.CURVE_COLUMNS).issubset(chunk.columns)
    keys = chunk[["semantic_key", "direction_system", "h_bar"]]
    assert keys.drop_duplicates().shape[0] == keys.shape[0]


def test_l2_row_metrics_key_uniqueness(ag_dual):
    base, dual = ag_dual
    frame = M.assemble_row_metrics(base, dual, "AG")
    assert set(M.ROW_METRICS_COLUMNS).issubset(frame.columns)
    keys = frame[["semantic_key", "direction_system"]]
    assert keys.drop_duplicates().shape[0] == frame.shape[0]


def test_weighted_km_first_event_synthetic_oracle():
    ev = np.array([-1, 2, -1], dtype=np.int64)
    cs = np.array([4, 4, 4], dtype=np.int64)
    wt = np.ones(3)
    F, R, D = M.weighted_km_first_event(ev, cs, wt, 4)
    assert abs(F[2] - 1.0 / 3.0) < 1e-12
    assert abs(F[4] - 1.0 / 3.0) < 1e-12
    assert np.all(F >= -1e-12) and np.all(F <= 1 + 1e-12)
    assert R[2] == 3.0 and D[2] == 1.0
    # adverse orientation: delta = Wrong - Correct
    corr = np.array([False, True, True])
    d = M.km_event_delta(ev, "sr_first_pierce", corr, wt, cs, 4)
    assert d["orientation"] == "wrong_minus_correct"
    assert np.allclose(d["delta"], d["F_wrong"] - d["F_correct"], equal_nan=True)


def test_e9_fix_break_arithmetic_identity():
    a9 = np.array([1, 1, 0, 0, 1, 0])
    e9 = np.array([1, 0, 1, 0, 1, 1])
    res = M.verify_disagreement_arithmetic(a9, e9)
    assert res["identity_holds"]
    assert (res["e9_fix"] - res["e9_break"]) == (res["n_correct_e9"] - res["n_correct_a9"])


# =========================================================================== #
# RC-T2 revision: availability / orientation / support / events / evidence     #
# (reviewer RC-T2-18 required regression tests)                                #
# =========================================================================== #
@pytest.fixture(scope="session")
def small_t2():
    return M.run_formal_t2(symbols=("AG",), n_subset=25, population="small",
                           write_artifacts=False, verbose=False)


def test_rc_t2_1_unavailable_row_contributes_neither_num_nor_den():
    value = np.array([[1.0, 1.0, 1.0, 1.0],
                      [3.0, 3.0, np.nan, np.nan]])
    correct = np.array([True, False])
    gid = np.array(["g1", "g2"])
    w = np.array([1.0, 1.0])
    ug, nc, dc, nw, dw = M.build_group_curve_sufficient_stats(value, correct, gid, w)
    # h index 2: correct available, wrong unavailable -> wrong den must be 0
    assert dc.sum(0)[2] == 1.0
    assert dw.sum(0)[2] == 0.0
    pc = M._safe_div(nc.sum(0), dc.sum(0))
    pw = M._safe_div(nw.sum(0), dw.sum(0))
    assert pc[2] == 1.0
    assert np.isnan(pw[2])
    # an unavailable row wrongly placed in the denominator would give 0.0 here
    assert not np.isclose(pw[2], 0.0)


def test_rc_t2_2_zero_group_denominator_returns_nan_not_zero():
    out = M._safe_div(np.array([2.0, 4.0]), np.array([1.0, 0.0]))
    assert out[0] == 2.0
    assert np.isnan(out[1])


def test_rc_t2_3_full_gate_fails_closed_on_incomplete_support():
    ok = dict(n_symbols=M.FROZEN_FULL_SYMBOLS, n_candidates=M.FROZEN_FULL_CANDIDATE_ROWS,
              n_gids=M.FROZEN_FULL_ORACLE_GIDS, a9_l2=M.FROZEN_FULL_A9_L2_ROWS,
              e9_l2=M.FROZEN_FULL_E9_L2_ROWS, availability_masks_identical=True,
              inferential_support_any=True, inferential_support_complete=True)
    M.check_full_population_gates(**ok)
    bad_complete = dict(ok); bad_complete["inferential_support_complete"] = False
    with pytest.raises(RuntimeError):
        M.check_full_population_gates(**bad_complete)
    bad_any = dict(ok); bad_any["inferential_support_any"] = False
    with pytest.raises(RuntimeError):
        M.check_full_population_gates(**bad_any)


def _orient_fixture():
    matrices = {"PS": np.array([[3.0], [1.0]]), "MFE": np.array([[3.0], [1.0]]),
                "MAE": np.array([[1.0], [3.0]]), "R": np.array([[3.0], [1.0]])}
    return matrices, np.array([True, False]), np.array(["a", "b"]), np.ones(2)


def test_rc_t2_4_ps_orientation_correct_minus_wrong():
    matrices, correct, gid, w = _orient_fixture()
    c = M.build_system_curves(matrices, correct, gid, w, B=50)
    assert M.CURVE_ORIENTATIONS["PS"] == "correct_minus_wrong"
    assert c["PS"]["contrast_orientation"] == "correct_minus_wrong"
    assert c["PS"]["point"][0] > 0    # correct(3) - wrong(1)


def test_rc_t2_5_mfe_orientation_correct_minus_wrong():
    matrices, correct, gid, w = _orient_fixture()
    c = M.build_system_curves(matrices, correct, gid, w, B=50)
    assert M.CURVE_ORIENTATIONS["MFE"] == "correct_minus_wrong"
    assert c["MFE"]["contrast_orientation"] == "correct_minus_wrong"
    assert c["MFE"]["point"][0] > 0


def test_rc_t2_6_signed_return_orientation_correct_minus_wrong():
    matrices, correct, gid, w = _orient_fixture()
    c = M.build_system_curves(matrices, correct, gid, w, B=50)
    assert M.CURVE_ORIENTATIONS["R"] == "correct_minus_wrong"
    assert c["R"]["contrast_orientation"] == "correct_minus_wrong"
    assert c["R"]["point"][0] > 0


def test_rc_t2_7_mae_orientation_wrong_minus_correct():
    matrices, correct, gid, w = _orient_fixture()
    c = M.build_system_curves(matrices, correct, gid, w, B=50)
    assert M.CURVE_ORIENTATIONS["MAE"] == "wrong_minus_correct"
    assert c["MAE"]["contrast_orientation"] == "wrong_minus_correct"
    assert c["MAE"]["point"][0] > 0    # wrong MAE(3) - correct MAE(1)
    raw = M.oriented_delta_curve(matrices["MAE"], correct, gid, w,
                                 "correct_minus_wrong", B=50)
    assert raw["point"][0] < 0         # opposite sign


def test_rc_t2_8_simultaneous_band_excludes_unsupported_h():
    n = 50
    correct = np.array([True] * n + [False] * n)
    gid = np.array([f"c{i}" for i in range(n)] + [f"w{i}" for i in range(n)])
    value = np.empty((2 * n, 2), dtype=np.float64)
    value[:, 0] = 1.0                              # all available at h=1
    value[:n, 1] = 1.0                             # correct available at h=2
    value[n:, 1] = np.nan                          # wrong unavailable at h=2
    c = M.oriented_delta_curve(value, correct, gid, np.ones(2 * n),
                               "correct_minus_wrong", B=100)
    assert bool(c["inferential_support"][0]) is True
    assert bool(c["inferential_support"][1]) is False
    assert c["n_valid_wrong"][1] == 0
    assert np.isnan(c["simul_lower"][1]) and np.isnan(c["simul_upper"][1])
    assert np.isnan(c["se"][1])


def test_rc_t2_9_same_bar_boolean_never_event_step(small_t2):
    keys = set(small_t2["event_curves"].keys())
    assert "sr_same_bar_reclaim" not in keys
    assert "lb_same_bar_reclaim" not in keys
    assert "sr_same_bar_reclaim" in M.NON_EVENT_FIELDS
    assert "sr_same_bar_reclaim" not in M.FIRST_PASSAGE_EVENTS
    assert "sr_same_bar_reclaim_time" in keys    # derived event time instead


def test_rc_t2_10_bars_to_reclaim_never_event_step(small_t2):
    keys = set(small_t2["event_curves"].keys())
    assert "sr_bars_to_reclaim" not in keys
    assert "lb_bars_to_reclaim" not in keys
    assert "sr_bars_to_reclaim" in M.NON_EVENT_FIELDS


def test_rc_t2_11_break_continue_is_terminal_only(small_t2):
    keys = set(small_t2["event_curves"].keys())
    assert "sr_break_continue" not in keys
    assert "lb_break_continue" not in keys
    assert "sr_break_continue" in M.NON_EVENT_FIELDS
    term = small_t2["break_continue_terminal"]
    for bs in ("SR", "LB"):
        assert set(term[bs].keys()) == {"correct", "wrong"}


def test_rc_t2_12_weighted_km_censoring_oracle():
    # event before censor
    F, _, _ = M.weighted_km_first_event(
        np.array([2, -1, -1]), np.array([4, 4, 4]), np.ones(3), 4)
    assert abs(F[2] - 1 / 3) < 1e-12 and abs(F[4] - 1 / 3) < 1e-12
    # censor before event -> not counted
    F2, _, _ = M.weighted_km_first_event(
        np.array([-1, 2, 3]), np.array([4, 4, 2]), np.ones(3), 4)
    assert abs(F2[2] - 1 / 3) < 1e-12 and abs(F2[4] - 1 / 3) < 1e-12
    # never-event -> F stays 0
    F3, _, _ = M.weighted_km_first_event(
        np.array([-1, -1]), np.array([3, 3]), np.ones(2), 3)
    assert np.allclose(F3, 0.0)
    # different weights: event weight 3 of total 4 -> hazard 0.75
    F4, _, _ = M.weighted_km_first_event(
        np.array([1, -1]), np.array([2, 2]), np.array([3.0, 1.0]), 2)
    assert abs(F4[1] - 0.75) < 1e-12


def test_rc_t2_13_broken_unreclaimed_can_rise_then_fall():
    fp = np.array([1]); fr = np.array([3])     # pierce at 1, reclaim at 3
    state = [bool(M.broken_unreclaimed_state(fp, fr, h)[0]) for h in range(6)]
    assert state == [False, True, True, False, False, False]
    pc, _ = M.broken_unreclaimed_prevalence(fp, fr, np.array([True]),
                                            np.array([1.0]), 5)
    assert pc[1] == 1.0 and pc[2] == 1.0 and pc[3] == 0.0


def test_rc_t2_14_conditional_reclaim_rates_oracle():
    fp = np.array([1, 1, 1])
    fr = np.array([1, 4, -1])
    sb = np.array([True, False, False])
    ff = np.array([-1, -1, 2])
    btr = np.array([0.0, 3.0, np.nan])
    correct = np.array([True, True, True])
    d = M.conditional_reclaim_diagnostics(fp, fr, sb, ff, btr, correct, np.ones(3))
    c = d["correct"]
    assert abs(c["any_reclaim_rate"] - 2 / 3) < 1e-12
    assert abs(c["same_bar_reclaim_rate"] - 1 / 3) < 1e-12
    assert abs(c["late_reclaim_rate"] - 1 / 3) < 1e-12
    assert abs(c["bars_to_reclaim_median"] - 1.5) < 1e-12
    assert abs(c["failed_reclaim_rate"] - 0.0) < 1e-12


def test_rc_t2_15_availability_oracle():
    value = np.array([[1.0, 1.0], [2.0, np.nan], [3.0, 3.0]])
    w = np.ones(3)
    correct = np.array([True, False, True])
    side = np.array([1.0, -1.0, 1.0])
    gid = np.array(["g0", "g1", "g2"])
    av = M.availability_curves(value, w, correct, side, gid, 2)
    assert list(av["overall"]["available_rows"]) == [3, 2]
    assert abs(av["overall"]["availability_fraction"][1] - 2 / 3) < 1e-12
    assert list(av["wrong"]["available_rows"]) == [1, 0]
    assert list(av["LONG"]["available_rows"]) == [2, 2]
    assert abs(av["overall"]["available_weight_mass"][1] - 2.0) < 1e-12


def test_rc_t2_16_a9_e9_availability_equality_all_h(ag_dual, small_t2):
    base, dual = ag_dual
    a9, e9 = dual["A9"], dual["E9"]
    assert np.array_equal(np.isfinite(a9["curve_ps"]), np.isfinite(e9["curve_ps"]))
    assert small_t2["availability_masks_identical"] is True
    assert small_t2["per_system_gid_weight_ok"] is True


def test_rc_t2_17_side_landmark_diagnostics_schema(small_t2):
    rows = small_t2["side_landmark_rows"]
    assert rows
    required = {"direction_system", "landmark", "stratum", "metric", "weighted_mean",
                "p25", "median", "p75", "n_rows", "n_gids", "weight_mass"}
    for r in rows:
        assert required.issubset(r.keys())
        assert r["landmark"] in {"m15", "h1", "h4", "td1", "td3", "td5"}
        assert r["metric"] in {"MFE", "MAE", "PS", "R"}
        assert r["direction_system"] in {"E9", "A9"}
    assert {"overall", "LONG", "SHORT"}.issubset({r["stratum"] for r in rows})


def test_rc_t2_18_observed_bar_minutes_semantics(ag_dual):
    base, dual = ag_dual
    assert "observed_bar_minutes" in M.CURVE_COLUMNS
    assert "elapsed_minutes" not in M.CURVE_COLUMNS
    chunk = M.curve_chunk_from_dual(base, dual, "AG")
    assert np.all(chunk["observed_bar_minutes"].to_numpy()
                  == 15 * chunk["h_bar"].to_numpy())


def test_rc_t2_19_real_trading_day_preserved(ag_dual, ag_state):
    base, dual = ag_dual
    frame = M.assemble_row_metrics(base, dual, "AG")
    assert frame["trading_day"].notna().all()
    exp = np.asarray(ag_state.trading_day[base["entry_idx"]])
    got = np.unique(frame["trading_day"].to_numpy())
    assert set(got.tolist()).issubset(set(np.unique(exp).tolist()))


def _path_curve_row():
    row = {"direction_system": "E9", "metric": "PS",
           "contrast_orientation": "correct_minus_wrong", "h_bar": 1,
           "observed_bar_minutes": 15, "point": 0.1, "pointwise_lo": 0.0,
           "pointwise_hi": 0.2, "simul_lower": -0.1, "simul_upper": 0.3,
           "inferential_support": True, "n_valid_correct": 5, "n_valid_wrong": 5,
           "flag": ""}
    for s in ("overall", "correct", "wrong", "LONG", "SHORT"):
        row[f"{s}_rows"] = 10
        row[f"{s}_gids"] = 4
        row[f"{s}_mass"] = 10.0
        row[f"{s}_fraction"] = 1.0
    return row


def test_rc_t2_20_formal_evidence_writer_schema_round_trip(tmp_path):
    p = tmp_path / "pc.csv"
    M.write_path_curves_csv(p, [_path_curve_row()])
    df = pd.read_csv(p)
    assert list(df.columns) == M.PATH_CURVE_COLUMNS
    assert bool(df["inferential_support"].iloc[0]) is True

    e = tmp_path / "ec.csv"
    M.write_event_curves_csv(e, [{
        "event_type": "first_event_curve", "name": "sr_first_pierce", "backstop": "SR",
        "direction_system": "E9", "group": "", "stat_name": "F", "h_bar": 1,
        "observed_bar_minutes": 15, "value_correct": 0.1, "value_wrong": 0.2,
        "delta": 0.1, "orientation": "wrong_minus_correct", "value": np.nan}])
    assert list(pd.read_csv(e).columns) == M.EVENT_CURVE_COLUMNS

    g = tmp_path / "gs.csv"
    M.write_group_stats_csv(g, [{
        "direction_system": "E9", "landmark": "h1", "stratum": "overall", "metric": "PS",
        "weighted_mean": 0.1, "p25": 0.0, "median": 0.1, "p75": 0.2, "n_rows": 5,
        "n_gids": 3, "weight_mass": 5.0}])
    assert list(pd.read_csv(g).columns)[:4] == ["direction_system", "landmark",
                                                "stratum", "metric"]

    d = tmp_path / "dd.csv"
    M.write_a9_e9_disagreement_csv(d, [{"scope": "overall", "scope_value": "ALL",
                                        "agreement": 3, "disagreement": 2, "e9_fix": 1,
                                        "e9_break": 1, "n_rows": 5, "weight_mass": 5.0}])
    assert list(pd.read_csv(d).columns) == ["scope", "scope_value", "agreement",
                                            "disagreement", "e9_fix", "e9_break",
                                            "n_rows", "weight_mass"]

    s = tmp_path / "s.json"
    M.write_summary_json(s, {"pipeline_smoke_completed": True, "n": 1})
    with open(s) as f:
        assert json.load(f)["pipeline_smoke_completed"] is True


def test_rc_t2_21_full_population_gate_constants():
    assert M.FROZEN_FULL_SYMBOLS == 15
    assert M.FROZEN_FULL_CANDIDATE_ROWS == 13773
    assert M.FROZEN_FULL_ORACLE_GIDS == 638
    assert M.FROZEN_FULL_A9_L2_ROWS == 13773
    assert M.FROZEN_FULL_E9_L2_ROWS == 13773
    assert M.FROZEN_FULL_L2_ROWS == 27546


def test_rc_t2_22_formal_runner_has_no_reference_call():
    assert "scan_paths_reference" not in inspect.getsource(M.run_formal_t2)
    orig_ref = M.scan_paths_reference
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("REFERENCE_CALLED")
    M.scan_paths_reference = boom
    try:
        M.reset_counters()
        M.run_formal_t2(symbols=("AG",), n_subset=25, population="small", verbose=False)
    finally:
        M.scan_paths_reference = orig_ref
    assert calls["n"] == 0, "formal runner must not call the Reference kernel"


def test_rc_t2_23_one_dual_scan_per_symbol():
    M.reset_counters()
    M.run_formal_t2(symbols=("AG",), n_subset=25, population="small", verbose=False)
    assert M.COUNTERS["path_scan_count"] == 1
    assert M.COUNTERS["direction_chain_run_count"] == 1
    assert M.COUNTERS["candidate_python_loop_count"] == 0
    assert M.COUNTERS["hotloop_dataframe_concat_count"] == 0
    assert M.COUNTERS["full_history_recompute_count"] == 0


def test_rc_t2_24_no_best_h_selection():
    src = inspect.getsource(M.run_formal_t2)
    for tok in ("argmax", "argmin", "best_h", "select_primary", "optimal_h", "best bar"):
        assert tok not in src, f"forbidden token present in formal runner: {tok}"


def test_rc_t2_25_audit_only_labels_absent_from_production_signature():
    sig = inspect.signature(M.scan_paths_streaming)
    for p in sig.parameters:
        assert "oracle" not in p
    for f in M.AUDIT_ONLY_FIELDS:
        assert f not in sig.parameters


# =========================================================================== #
# FC1..FC16 revision regressions                                               #
# =========================================================================== #
def _mae_fixture():
    # correct MAE small (1), wrong MAE large (5): raw correct-wrong = -4.
    # 50 gids per group so bootstrap support is stable.
    n = 50
    value = np.concatenate([np.full((n, 1), 1.0), np.full((n, 1), 5.0)])
    correct = np.array([True] * n + [False] * n)
    gid = np.array([f"c{i}" for i in range(n)] + [f"w{i}" for i in range(n)])
    return value, correct, gid, np.ones(2 * n)


def test_fc16_1_mae_pointwise_interval_order_after_flip():
    value, correct, gid, w = _mae_fixture()
    c = M.oriented_delta_curve(value, correct, gid, w, "wrong_minus_correct", B=200)
    assert c["contrast_orientation"] == "wrong_minus_correct"
    assert c["point"][0] > 0
    assert c["pointwise_lo"][0] <= c["point"][0] <= c["pointwise_hi"][0]


def test_fc16_2_mae_simultaneous_interval_order_after_flip():
    value, correct, gid, w = _mae_fixture()
    c = M.oriented_delta_curve(value, correct, gid, w, "wrong_minus_correct", B=200)
    ok = np.isfinite(c["simul_lower"]) & np.isfinite(c["simul_upper"])
    assert ok.any()
    assert np.all(c["simul_lower"][ok] <= c["point"][ok] + 1e-12)
    assert np.all(c["point"][ok] <= c["simul_upper"][ok] + 1e-12)
    assert np.all(c["simul_lower"][ok] <= c["simul_upper"][ok] + 1e-12)


def test_fc16_3_mae_se_remains_nonnegative():
    value, correct, gid, w = _mae_fixture()
    c = M.oriented_delta_curve(value, correct, gid, w, "wrong_minus_correct", B=200)
    ok = np.isfinite(c["se"])
    assert np.all(c["se"][ok] >= 0.0)


def test_fc16_4_km_correct_group_excludes_wrong_rows():
    first_step = np.array([1] + [-1] * 9)
    censor = np.full(10, 5)
    weight = np.ones(10)
    correct = np.array([True] + [False] * 9)
    d = M.km_event_delta(first_step, "sr_first_pierce", correct, weight, censor, 5)
    assert abs(d["F_correct"][1] - 1.0) < 1e-12   # must be 1.0, NOT 0.1
    assert abs(d["F_wrong"][1] - 0.0) < 1e-12


def test_fc16_5_km_wrong_group_excludes_correct_rows():
    first_step = np.array([-1] + [1] * 9)
    censor = np.full(10, 5)
    weight = np.ones(10)
    correct = np.array([True] + [False] * 9)
    d = M.km_event_delta(first_step, "sr_first_pierce", correct, weight, censor, 5)
    assert abs(d["F_wrong"][1] - 1.0) < 1e-12
    assert abs(d["F_correct"][1] - 0.0) < 1e-12   # correct risk set must be pure


def test_fc16_6_backstop_touch_orientation_wrong_minus_correct():
    for nm in ("sr_first_touch", "sr_first_pierce", "sr_first_failed_reclaim",
               "lb_first_touch", "lb_first_pierce", "lb_first_failed_reclaim"):
        assert M._event_orientation(nm) == "wrong_minus_correct"
    for nm in ("sr_first_reclaim", "lb_first_reclaim", "sr_same_bar_reclaim_time",
               "lb_late_reclaim_time", "first_ahead_sr_cross",
               "first_ahead_liq_touch"):
        assert M._event_orientation(nm) == "correct_minus_wrong"


def test_fc16_7_censor_aware_broken_unreclaimed_denominator():
    # A: pierce 1, reclaim 3, censor 5 ; B: pierce 1, never reclaim, censor 2
    fp = np.array([1, 1]); fr = np.array([3, -1])
    correct = np.array([True, True]); w = np.ones(2)
    censor = np.array([5, 2])
    pc, _ = M.broken_unreclaimed_prevalence(fp, fr, correct, w, 5, censor)
    assert abs(pc[1] - 1.0) < 1e-12        # both available and broken
    assert abs(pc[4] - 0.0) < 1e-12        # A reclaimed, B censored -> excluded (not 0.5)


def _landmark_curve(n_steps=25):
    ps = np.zeros((1, n_steps)); ps[0, 0] = 7.0; ps[0, 3] = 5.0; ps[0, 15] = 9.0
    z = np.zeros((1, n_steps))
    return {"PS": ps, "MFE": z.copy(), "MAE": z.copy(), "R": z.copy()}


def _landmark_rows():
    lm = M._build_landmark_steps(
        [{"td1": np.array([18]), "td3": np.array([20]), "td5": np.array([24])}],
        [np.array([0])])
    return M._side_landmark_rows(_landmark_curve(), np.array([True]), np.array([1.0]),
                                 np.array(["g0"]), lm, np.array([1.0]), "E9")


def _land_val(rows, landmark, metric="PS", stratum="overall"):
    for r in rows:
        if r["landmark"] == landmark and r["metric"] == metric and r["stratum"] == stratum:
            return r["weighted_mean"]
    raise KeyError((landmark, metric, stratum))


def test_fc16_8_m15_side_landmark_equals_curve_step0():
    assert _land_val(_landmark_rows(), "m15") == 7.0


def test_fc16_9_h1_side_landmark_equals_curve_step3():
    assert _land_val(_landmark_rows(), "h1") == 5.0


def test_fc16_10_16bar_side_landmark_equals_curve_step15():
    assert _land_val(_landmark_rows(), "h4") == 9.0


def test_fc16_11_zero_se_supported_band_collapses_to_point():
    rng = np.random.default_rng(3)
    reps = np.stack([rng.normal(size=8), np.full(8, 5.0)], axis=1)
    point = np.array([0.0, 5.0])
    band = M.simultaneous_band(point, reps, np.ones((8, 2), bool), np.ones((8, 2), bool))
    assert band["inferential_support"].all()
    assert band["se"][1] == 0.0
    assert band["lower"][1] == 5.0 and band["upper"][1] == 5.0
    assert band["se"][0] > 0.0
    assert np.isfinite(band["lower"][0]) and np.isfinite(band["upper"][0])


def test_fc16_12_all_zero_se_supported_band_q_is_zero():
    reps = np.full((6, 3), 2.0)
    point = np.full(3, 2.0)
    band = M.simultaneous_band(point, reps, np.ones((6, 3), bool), np.ones((6, 3), bool))
    assert band["q"] == 0.0
    assert np.all(band["lower"] == 2.0) and np.all(band["upper"] == 2.0)


def test_fc16_13_unsupported_late_h_does_not_fail_full_gate():
    M.check_full_population_gates(
        n_symbols=M.FROZEN_FULL_SYMBOLS, n_candidates=M.FROZEN_FULL_CANDIDATE_ROWS,
        n_gids=M.FROZEN_FULL_ORACLE_GIDS, a9_l2=M.FROZEN_FULL_A9_L2_ROWS,
        e9_l2=M.FROZEN_FULL_E9_L2_ROWS, availability_masks_identical=True,
        inferential_support_any=True, inferential_support_complete=True,
        counters={"direction_chain_run_count": 1, "raw_exec_load_count": 15,
                  "path_scan_count": 15, "reference_call_count_production": 0,
                  "full_history_recompute_count": 0, "candidate_python_loop_count": 0,
                  "hotloop_dataframe_concat_count": 0})
    with pytest.raises(RuntimeError):
        M.check_full_population_gates(
            n_symbols=M.FROZEN_FULL_SYMBOLS, n_candidates=M.FROZEN_FULL_CANDIDATE_ROWS,
            n_gids=M.FROZEN_FULL_ORACLE_GIDS, a9_l2=M.FROZEN_FULL_A9_L2_ROWS,
            e9_l2=M.FROZEN_FULL_E9_L2_ROWS, availability_masks_identical=True,
            inferential_support_any=False, inferential_support_complete=True)


def test_fc16_14_conditional_reclaim_evidence_long_schema(tmp_path):
    p = tmp_path / "ec.csv"
    M.write_event_curves_csv(p, [{
        "event_type": "conditional_reclaim_stat", "name": "sr", "backstop": "SR",
        "direction_system": "E9", "group": "correct", "stat_name": "any_reclaim_rate",
        "h_bar": 0, "observed_bar_minutes": 0, "value_correct": np.nan,
        "value_wrong": np.nan, "delta": np.nan, "orientation": "rate", "value": 0.5}])
    df = pd.read_csv(p)
    assert list(df.columns) == M.EVENT_CURVE_COLUMNS
    assert df["stat_name"].iloc[0] == "any_reclaim_rate"
    assert df["group"].iloc[0] == "correct"
    assert df["value"].iloc[0] == 0.5


def test_fc16_15_path_curve_availability_all_strata():
    for s in ("overall", "correct", "wrong", "LONG", "SHORT"):
        for f in ("rows", "gids", "mass", "fraction"):
            assert f"{s}_{f}" in M.PATH_CURVE_COLUMNS


def test_fc16_16_a9_event_atlas_without_extra_path_scan():
    M.reset_counters()
    res = M.run_formal_t2(symbols=("AG",), n_subset=20, population="small",
                          verbose=False)
    assert res["a9_event_curves"]
    assert res["a9_broken_unreclaimed"] and res["a9_conditional_reclaim"]
    assert res["a9_break_continue_terminal"]
    assert M.COUNTERS["path_scan_count"] == 1


def test_fc16_17_full_disagreement_decomposition_scopes(small_t2):
    rows = small_t2["disagreement_decomposition"]
    scopes = {r["scope"] for r in rows}
    assert {"overall", "symbol", "oracle_direction", "a9_predicted_direction",
            "e9_predicted_direction"}.issubset(scopes)
    for r in rows:
        assert {"agreement", "disagreement", "e9_fix", "e9_break", "n_rows",
                "weight_mass"}.issubset(r.keys())


def test_fc16_18_full_runner_guard_requires_allow_full():
    with pytest.raises(RuntimeError):
        M.run_formal_t2(symbols=("AG",), population="full")


def test_fc16_19_mocked_full_path_invokes_hard_gates(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(M, "check_full_population_gates",
                        lambda **kw: seen.update(kw))
    monkeypatch.setattr(M, "ARTIFACT_DIR", str(tmp_path / "art"))
    monkeypatch.setattr(M, "EVIDENCE_DIR", str(tmp_path / "ev"))
    M.run_formal_t2(symbols=("AG",), n_subset=20, population="full",
                    allow_full=True, write_artifacts=False, verbose=False)
    assert seen, "full path must invoke the hard gates"
    assert "n_candidates" in seen and "inferential_support_any" in seen


def test_fc16_20_mocked_full_evidence_uses_canonical_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "check_full_population_gates", lambda **kw: None)
    art = tmp_path / "art"; ev = tmp_path / "ev"
    monkeypatch.setattr(M, "ARTIFACT_DIR", str(art))
    monkeypatch.setattr(M, "EVIDENCE_DIR", str(ev))
    res = M.run_formal_t2(symbols=("AG",), n_subset=20, population="full",
                          allow_full=True, write_artifacts=True, verbose=False)
    assert res["artifacts"]["directory"] == str(art)
    assert os.path.isfile(art / "entry_path_row_metrics_v1.parquet")
    assert os.path.isfile(art / "entry_path_curve_v1.parquet")
    assert os.path.isfile(ev / "entry_path_atlas_v1_path_curves.csv")
    assert os.path.isfile(ev / "entry_path_atlas_v1_manifest.json")
    assert not str(res["artifacts"]["directory"]).startswith(tempfile.gettempdir())


def test_fc16_21_full_evidence_manifest_contains_artifact_sha(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "check_full_population_gates", lambda **kw: None)
    art = tmp_path / "art"; ev = tmp_path / "ev"
    monkeypatch.setattr(M, "ARTIFACT_DIR", str(art))
    monkeypatch.setattr(M, "EVIDENCE_DIR", str(ev))
    M.run_formal_t2(symbols=("AG",), n_subset=20, population="full",
                    allow_full=True, write_artifacts=True, verbose=False)
    with open(ev / "entry_path_atlas_v1_manifest.json") as f:
        man = json.load(f)
    shas = man["artifact_sha256"]
    for k in ("row_metrics_parquet", "curve_parquet", "path_curves_csv",
              "event_curves_csv", "group_stats_csv", "disagreement_csv",
              "summary_json"):
        assert shas.get(k) and len(shas[k]) == 64
    for k in ("generator_code_sha", "reviewed_parent_sha", "bootstrap_seed",
              "bootstrap_B", "direction_artifact", "environment_identities",
              "peak_rss", "counters", "verdict_e9", "unsupported_h"):
        assert k in man
