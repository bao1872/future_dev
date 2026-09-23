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
import resource
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
    sub = {k: (v[:150] if isinstance(v, np.ndarray)
               else {kk: vv[:150] for kk, vv in v.items()})
           for k, v in ag_anchors.items() if k != "df"}
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
