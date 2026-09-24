"""Tests for FUTURE-R6-M15-STRUCTURAL-STOP-V1 (Amendment A1: GROSS primary).

T0/T1 synthetic stop-state tests, A12 fill tests, and Reference-vs-Production
differential on real frozen frames.
"""

import inspect

import numpy as np
import pandas as pd
import pytest

import research.liquidity_oracle_atlas.structural_stop_v1 as R


# --------------------------------------------------------------------------- #
# Synthetic case builder                                                        #
# --------------------------------------------------------------------------- #
def synth(*, side, opens, lows, closes, highs=None, segments=None,
          entry_idx=0, end_idx=None, B=100.0, td_end=None, td1=None, td3=None,
          td5=None, atr=1.0):
    opens = np.asarray(opens, float)
    lows = np.asarray(lows, float)
    closes = np.asarray(closes, float)
    highs = closes.copy() if highs is None else np.asarray(highs, float)
    n = len(closes)
    segments = np.zeros(n, np.int64) if segments is None else np.asarray(segments, np.int64)
    end = n - 1 if end_idx is None else end_idx
    d1 = end if td_end is None and td1 is None else (td_end if td1 is None else td1)
    d3 = end if td_end is None and td3 is None else (td_end if td3 is None else td3)
    d5 = end if td_end is None and td5 is None else (td_end if td5 is None else td5)
    is_long = side > 0
    # LONG: support zone [B, B+1]; boundary B = support_bottom.  SHORT mirror.
    sup_bottom = np.array([B if is_long else B - 10.0])
    sup_top = np.array([B + 1.0 if is_long else B - 9.0])
    res_top = np.array([B + 10.0 if is_long else B])
    res_bottom = np.array([B + 9.0 if is_long else B - 1.0])
    return {
        "side": np.array([float(side)]),
        "entry_idx": np.array([entry_idx], np.int64),
        "end_idx": np.array([end], np.int64),
        "atr0": np.array([atr]),
        "entry_price": np.array([opens[entry_idx]]),
        "entry_segment": np.array([segments[entry_idx]], np.int64),
        "open": opens, "high": highs, "low": lows, "close": closes,
        "segment": segments, "n_bars": n,
        "raw_sup_top": sup_top, "raw_sup_bottom": sup_bottom,
        "raw_res_top": res_top, "raw_res_bottom": res_bottom,
        "td_ends": {"td1": np.array([d1], np.int64),
                    "td3": np.array([d3], np.int64),
                    "td5": np.array([d5], np.int64)},
    }


def ref(case):
    return R.scan_structural_stop_reference(case)


# --------------------------------------------------------------------------- #
# §23 rule tests                                                                #
# --------------------------------------------------------------------------- #
def test_long_enters_support_without_pierce_no_stop():
    # low touches B (<=) but never goes below B -> no invalidation
    c = synth(side=+1, opens=[100, 100, 100, 100], lows=[100, 100.5, 100.2, 100.1],
              closes=[100, 100.5, 100.3, 100.2], B=100.0, entry_idx=1)
    out = ref(c)
    assert out["inv_step"][0] == -1
    assert not out["per_h"]["td5"]["stop_signal_observed"][0]


def test_long_pierce_same_bar_reclaim_no_stop():
    # bar 1 pierces (low<B) but close>=B -> same-bar reclaim -> no stop
    c = synth(side=+1, opens=[100, 100, 100, 100], lows=[100, 99.0, 100, 100],
              closes=[100, 100.4, 100, 100], B=100.0, entry_idx=0)
    out = ref(c)
    assert out["inv_step"][0] == -1


def test_long_pierce_and_close_below_stop_signal():
    c = synth(side=+1, opens=[100, 100, 100, 100], lows=[100, 99.0, 99.0, 99.0],
              closes=[100, 99.5, 99.2, 99.1], B=100.0, entry_idx=0)
    out = ref(c)
    assert out["inv_step"][0] == 1
    m = out["per_h"]["td5"]
    assert m["stop_signal_observed"][0]
    assert m["stop_signal_step"][0] == 1
    assert m["stop_executable"][0]
    assert m["stop_fill_step"][0] == 2
    assert m["stop_fill_open"][0] == c["open"][2]


def test_short_mirror_no_reclaim_requires_high_and_close_above():
    # SHORT: invalidation needs high>B AND close>B; high>B with close<=B is a reclaim
    c = synth(side=-1, opens=[100, 100, 100, 100], lows=[100, 100, 100, 100],
              highs=[100, 101.0, 100.0, 100.0], closes=[100, 99.9, 100.0, 100.0],
              B=100.0, entry_idx=0)
    out = ref(c)
    assert out["inv_step"][0] == -1  # high>B but close<=B -> same-bar reclaim


def test_short_pierce_and_close_above_stop_signal():
    c = synth(side=-1, opens=[100, 100, 100, 100], lows=[100, 100, 100, 100],
              highs=[100, 101.0, 101.0, 101.0], closes=[100, 100.6, 100.7, 100.8],
              B=100.0, entry_idx=0)
    out = ref(c)
    assert out["inv_step"][0] == 1
    assert out["per_h"]["td5"]["stop_executable"][0]


def test_reclaim_then_later_fail_stops_on_later_failure():
    # bar1 pierce+reclaim (close>=B); bar3 closes below B -> invalidation at 3
    c = synth(side=+1, opens=[100, 100, 100, 100, 100],
              lows=[100, 99.0, 100.0, 99.0, 99.0],
              closes=[100, 100.4, 100.2, 99.3, 99.2], B=100.0, entry_idx=0)
    out = ref(c)
    assert out["inv_step"][0] == 3


def test_no_zone_no_stop():
    c = synth(side=+1, opens=[100, 100, 100, 100],
              lows=[100, 99, 99, 99], closes=[100, 99, 99, 99], B=100.0, entry_idx=0)
    c["raw_sup_bottom"] = np.array([np.nan])   # no support zone
    out = ref(c)
    assert not out["stop_eligible"][0]
    m = out["per_h"]["td5"]
    assert not m["stop_signal_observed"][0]
    assert m["stop_reason"][0] == "not_eligible"


def test_signal_on_final_bar_no_next_bar_no_fill():
    c = synth(side=+1, opens=[100, 100, 100], lows=[100, 100, 99],
              closes=[100, 100, 99.5], B=100.0, entry_idx=0)  # end = last bar
    out = ref(c)
    assert out["inv_step"][0] == 2
    m = out["per_h"]["td5"]
    assert m["stop_signal_observed"][0]
    assert not m["stop_executable"][0]
    assert m["stop_reason"][0] == "no_next_bar"


def test_signal_before_segment_break_next_bar_same_segment_fills_at_next_open():
    seg = np.array([0, 0, 0, 0], np.int64)     # next bar same segment
    c = synth(side=+1, opens=[100, 100, 100.7, 100], lows=[100, 99, 99, 99],
              closes=[100, 99.5, 99.2, 99.1], segments=seg, B=100.0, entry_idx=0)
    out = ref(c)
    m = out["per_h"]["td5"]
    assert m["stop_executable"][0]
    assert m["stop_fill_open"][0] == 100.7


def test_segment_change_makes_stop_not_executable():
    seg = np.array([0, 0, 1, 1], np.int64)     # next bar new segment
    c = synth(side=+1, opens=[100, 100, 90, 100], lows=[100, 99, 90, 90],
              closes=[100, 99.5, 90, 90], segments=seg, B=100.0, entry_idx=0)
    out = ref(c)
    m = out["per_h"]["td5"]
    assert m["stop_signal_observed"][0]
    assert not m["stop_executable"][0]
    assert m["stop_reason"][0] == "segment_change"


def test_same_trading_day_change_but_same_segment_is_allowed():
    # kernel keyed on segment only; trading_day is NOT part of the case
    seg = np.array([0, 0, 0, 0], np.int64)
    c = synth(side=+1, opens=[100, 100, 100.5, 100], lows=[100, 99, 99, 99],
              closes=[100, 99.5, 99.2, 99.1], segments=seg, B=100.0, entry_idx=0)
    m = ref(c)["per_h"]["td5"]
    assert m["stop_executable"][0]


def test_gap_against_trade_uses_real_next_open_not_boundary():
    # next open gaps far below the support boundary; fill must be that real open
    c = synth(side=+1, opens=[100, 100, 95.0, 95.0], lows=[100, 99, 95.0, 95.0],
              closes=[100, 99.5, 95.0, 95.0], B=100.0, entry_idx=0)
    m = ref(c)["per_h"]["td5"]
    assert m["stop_fill_open"][0] == 95.0
    assert m["stop_gross"][0] == pytest.approx((95.0 - 100.0) / 1.0)


def test_gap_in_favour_uses_real_next_open():
    c = synth(side=+1, opens=[100, 100, 103.0, 103.0], lows=[100, 99, 103.0, 103.0],
              closes=[100, 99.5, 103.0, 103.0], B=100.0, entry_idx=0)
    m = ref(c)["per_h"]["td5"]
    assert m["stop_fill_open"][0] == 103.0


def test_future_mutation_after_signal_cannot_change_signal_or_fill():
    c = synth(side=+1, opens=[100, 100, 100.5, 100, 100],
              lows=[100, 99, 99, 99, 99], closes=[100, 99.5, 99.2, 99.1, 99.0],
              B=100.0, entry_idx=0)
    o1 = ref(c)
    c2 = {k: (v.copy() if isinstance(v, np.ndarray) else
              {kk: vv.copy() for kk, vv in v.items()} if isinstance(v, dict) else v)
          for k, v in c.items()}
    for icol in ("open", "low", "close"):
        c2[icol][3] *= 5.0
        c2[icol][4] *= 0.1
    o2 = ref(c2)
    assert o1["inv_step"][0] == o2["inv_step"][0]
    for H in R.HORIZONS:
        assert (o1["per_h"][H]["stop_signal_step"][0]
                == o2["per_h"][H]["stop_signal_step"][0])
        assert (o1["per_h"][H]["stop_fill_step"][0]
                == o2["per_h"][H]["stop_fill_step"][0])
        assert (o1["per_h"][H]["stop_fill_open"][0]
                == o2["per_h"][H]["stop_fill_open"][0])


def test_stop_policy_signatures_have_no_correctness_or_oracle_labels():
    for fn in (R.scan_structural_stop_reference, R.scan_structural_stop_production,
               R.production_invalidation_step, R.reference_invalidation_step,
               R.resolve_treatment):
        for p in inspect.signature(fn).parameters:
            assert "correct" not in p and "oracle" not in p, f"{fn.__name__}.{p}"


# --------------------------------------------------------------------------- #
# §24 differential (synthetic + real)                                           #
# --------------------------------------------------------------------------- #
def _prod_case_from_ref(case):
    r = R.scan_structural_stop_reference(case)
    c2 = dict(case)
    c2["inv_step"] = r["inv_step"]
    return c2, r


def test_differential_synthetic_long_and_short_match():
    cases = [
        synth(side=+1, opens=[100, 100, 100, 100], lows=[100, 99, 99, 99],
              closes=[100, 99.5, 99.2, 99.1], B=100.0, entry_idx=0),
        synth(side=-1, opens=[100, 100, 100, 100], lows=[100, 100, 100, 100],
              highs=[100, 101, 101, 101], closes=[100, 100.6, 100.7, 100.8],
              B=100.0, entry_idx=0),
    ]
    for c in cases:
        c2, r = _prod_case_from_ref(c)
        p = R.scan_structural_stop_production(c2)
        assert np.array_equal(r["inv_step"], p["inv_step"])
        for H in R.HORIZONS:
            for f in ("stop_signal_observed", "stop_executable", "stop_fill_step"):
                assert np.array_equal(r["per_h"][H][f], p["per_h"][H][f])


def test_baseline_td_returns_reproduce_frozen_r5_checkpoints():
    l2 = R.load_r5_l2()
    sub = l2[l2.symbol == "AG"]
    keys = sorted(sub.semantic_key.unique())[:15]
    sub = sub[sub.semantic_key.isin(keys)]
    st = R.load_symbol_state("AG")
    rows, _ = R.treatment_rows_for_symbol(sub, st, "E9")
    e9 = sub[sub.direction_system == "E9"].sort_values("semantic_key")
    for H in R.HORIZONS:
        r = rows[rows.evaluation_horizon == H].sort_values("semantic_key")
        ref = e9[f"{H}_r"].to_numpy(float)
        assert np.allclose(r["baseline_gross_return_atr"].to_numpy(float), ref,
                           atol=1e-12)


def test_no_net_cost_fields_and_governance_constants():
    assert R.COST_METADATA_STATUS == "UNAVAILABLE_COST_METADATA"
    assert R.REALISTIC_NET_PNL_STATUS == "NOT_ESTIMATED"
    assert R.FORMAL_PRIMARY_BASIS == "GROSS_PAIRED_POLICY_EFFECT"
    assert R.FRICTION_PROXY_USED_FOR_VERDICT is False
    assert R.ENTRY_COST_ESTIMATED is False and R.EXIT_COST_ESTIMATED is False
    for c in R.ROW_COLUMNS:
        assert "net" not in c.lower()
        assert "cost" not in c.lower()


def test_frozen_input_verification_passes():
    frozen = R.verify_frozen_inputs()
    assert len(frozen) == 3


def test_paired_bootstrap_is_deterministic_and_paired():
    delta = np.array([1.0, -0.5, 0.2, 0.9, -0.3, 0.4])
    gid = np.array(["g"] * 6)
    w = np.array([1 / 6] * 6)
    a = R.paired_gid_bootstrap(delta, gid, w, B=50, seed=1)
    b = R.paired_gid_bootstrap(delta, gid, w, B=50, seed=1)
    assert np.array_equal(a["reps"], b["reps"])
    assert a["point"] == pytest.approx(delta.mean())


# =========================================================================== #
# RC-R6-1..15 revision regressions                                             #
# =========================================================================== #
def flat_case(*, side=+1, n=32, pierce_step=10, pierce_close=None, td=(4, 12, 30),
              seg=None, B=100.0):
    opens = np.full(n, 100.0)
    lows = np.full(n, 100.5)
    closes = np.full(n, 100.6)
    highs = np.full(n, 100.7)
    if pierce_step is not None:
        close = (99.0 if pierce_close is None else pierce_close)
        if side > 0:
            lows[pierce_step] = 99.0
        else:
            highs[pierce_step] = 101.0
        closes[pierce_step] = (-close if side < 0 else close)
    return synth(side=side, opens=opens, lows=lows, closes=closes, highs=highs,
                 segments=seg, entry_idx=0, end_idx=n - 1, B=B,
                 td1=td[0], td3=td[1], td5=td[2])


def test_rc_r6_1_invalidation_after_td1_exit_not_observed_in_td1():
    c = flat_case(pierce_step=10, td=(4, 12, 30))
    out = ref(c)
    m1 = out["per_h"]["td1"]
    assert out["inv_step"][0] == 10
    assert not m1["stop_signal_observed"][0]
    assert m1["stop_signal_step"][0] == -1
    assert m1["stop_reason"][0] == "no_signal_before_baseline_exit"
    assert not m1["stop_executable"][0]


def test_rc_r6_1_same_invalidation_observed_under_td3_and_td5():
    c = flat_case(pierce_step=10, td=(4, 12, 30))
    out = ref(c)
    for H in ("td3", "td5"):
        m = out["per_h"][H]
        assert m["stop_signal_observed"][0]
        assert m["stop_signal_step"][0] == 10
        assert m["stop_executable"][0]
        assert m["stop_fill_step"][0] == 11


def test_rc_r6_1_signal_exactly_on_exit_bar_observed_but_not_executable():
    c = flat_case(pierce_step=5, td=(5, 20, 30))
    m = ref(c)["per_h"]["td1"]
    assert m["stop_signal_observed"][0]
    assert m["stop_signal_step"][0] == 5
    assert not m["stop_executable"][0]
    assert m["stop_reason"][0] == "beyond_baseline_exit"


def _copy_case(c):
    return {k: (v.copy() if isinstance(v, np.ndarray) else
                {kk: vv.copy() for kk, vv in v.items()} if isinstance(v, dict) else v)
            for k, v in c.items()}


def test_rc_r6_1_td1_diagnostics_invariant_to_post_td1_mutation():
    c = flat_case(pierce_step=10, td=(4, 12, 30))
    a = ref(c)["per_h"]["td1"]
    # (a) mutate strictly AFTER the signal bar: the whole TD1 row must be identical
    c2 = _copy_case(c)
    for icol in ("open", "low", "close", "high"):
        c2[icol][11:] *= 3.0
    b = ref(c2)["per_h"]["td1"]
    for f in ("stop_signal_observed", "stop_signal_step", "stop_executable",
              "stop_fill_step", "stop_reason"):
        assert np.array_equal(np.asarray(a[f]), np.asarray(b[f]))
    # (b) mutate anywhere after the TD1 exit (including the later signal bar):
    # the TD1 causal fields stay "no signal in TD1"
    c3 = _copy_case(c)
    for icol in ("open", "low", "close", "high"):
        c3[icol][5:] *= 3.0
    d = ref(c3)["per_h"]["td1"]
    assert not d["stop_signal_observed"][0]
    assert d["stop_signal_step"][0] == -1
    assert not d["stop_executable"][0]
    assert d["stop_reason"][0] in ("no_signal", "no_signal_before_baseline_exit")


def test_rc_r6_2_weighted_delta_wrong_differs_from_raw_row_mean():
    base = np.array([0.0, 0.0, 0.0])
    stop = np.array([1.0, 2.0, 3.0])
    correct = np.array([True, False, False])
    w = np.array([0.8, 0.15, 0.05])
    dec = R.decomposition_weighted(base, stop, correct, w)
    raw_wrong = np.mean(np.array([2.0, 3.0]))
    assert raw_wrong == pytest.approx(2.5)
    assert dec["delta_wrong_gross"] == pytest.approx(2.25)
    assert dec["delta_wrong_gross"] != pytest.approx(raw_wrong)


def test_rc_r6_2_weighted_wlr_oracle():
    base = np.array([-2.0, -4.0, 1.0])
    stop = np.array([-1.0, -1.0, 1.0])
    correct = np.array([False, False, True])
    w = np.array([0.3, 0.3, 0.4])
    dec = R.decomposition_weighted(base, stop, correct, w)
    assert dec["wlr_gross"] == pytest.approx(2.0)   # (1*0.3 + 3*0.3)/0.6


def test_rc_r6_2_weighted_delta_correct_oracle():
    base = np.array([0.0, 0.0, 5.0])
    stop = np.array([1.0, 1.0, 3.0])
    correct = np.array([False, False, True])
    w = np.array([0.5, 0.2, 0.3])
    dec = R.decomposition_weighted(base, stop, correct, w)
    assert dec["delta_correct_gross"] == pytest.approx(-2.0)


def test_rc_r6_2_weighted_cpe_oracle():
    base = np.array([0.0, 0.0, 5.0])
    stop = np.array([1.0, 1.0, 3.0])
    correct = np.array([False, False, True])
    w = np.array([0.5, 0.2, 0.3])
    dec = R.decomposition_weighted(base, stop, correct, w)
    assert dec["cpe_gross"] == pytest.approx(2.0)


def _make_big(*, weights, signals, delta, correct, system="E9", horizon="td5",
              direction="LONG", gids=None):
    n = len(weights)
    gids = gids if gids is not None else [f"g{i}" for i in range(n)]
    return pd.DataFrame({
        "direction_system": system, "evaluation_horizon": horizon,
        "direction": direction,
        "baseline_gross_return_atr": np.zeros(n),
        "stop_gross_return_atr": np.asarray(delta, float),
        "sample_weight_raw": np.asarray(weights, float),
        "gid": gids, "direction_correct": np.asarray(correct, bool),
        "stop_eligible": np.ones(n, bool),
        "stop_signal_observed": np.asarray(signals, bool),
        "stop_executable": np.asarray(signals, bool),
    })


def test_rc_r6_3_delta_wrong_bootstrap_deterministic():
    big = _make_big(weights=[0.6, 0.2, 0.2], signals=[1, 0, 0],
                    delta=[1.0, -1.0, -1.0], correct=[False, False, True])
    r1 = R._effect(big, "E9", "td5")
    r2 = R._effect(big, "E9", "td5")
    assert r1["delta_wrong_gross"] == pytest.approx(r2["delta_wrong_gross"])
    assert r1["delta_wrong_gross_ci_low"] == pytest.approx(r2["delta_wrong_gross_ci_low"])


def test_rc_r6_4_formal_stop_rate_uses_weights():
    big = _make_big(weights=[0.8, 0.1, 0.1], signals=[1, 0, 0],
                    delta=[1.0, 1.0, 1.0], correct=[True, True, True])
    r = R._effect(big, "E9", "td5")
    assert r["stop_signal_rate"] == pytest.approx(0.8)      # weighted
    assert r["stop_signal_rate"] != pytest.approx(1 / 3)    # not the raw row mean


def test_rc_r6_6_break_even_fields_equal_gross_delta_and_ci():
    big = _make_big(weights=[0.5, 0.5], signals=[1, 0],
                    delta=[2.0, -1.0], correct=[False, True])
    r = R._effect(big, "E9", "td5")
    assert r["break_even_incremental_exit_cost_atr"] == pytest.approx(r["delta_ev_gross"])
    assert r["break_even_incremental_exit_cost_ci_low"] == pytest.approx(r["delta_ev_ci_low"])
    assert r["break_even_incremental_exit_cost_ci_high"] == pytest.approx(r["delta_ev_ci_high"])


def test_rc_r6_5_baseline_exit_price_reproduces_frozen_td_r():
    l2 = R.load_r5_l2()
    sub = l2[l2.symbol == "AG"]
    keys = sorted(sub.semantic_key.unique())[:12]
    sub = sub[sub.semantic_key.isin(keys)]
    st = R.load_symbol_state("AG")
    rows, _ = R.treatment_rows_for_symbol(sub, st, "E9")
    side = np.where(rows.direction.to_numpy(object) == "LONG", 1.0, -1.0)
    recon = side * (rows.baseline_exit_price.to_numpy(float)
                    - rows.entry_price.to_numpy(float)) / rows.ATR0.to_numpy(float)
    assert np.allclose(recon, rows.baseline_gross_return_atr.to_numpy(float), atol=1e-12)


def test_rc_r6_9_mocked_formal_path_invokes_population_gates(monkeypatch):
    seen = {}

    def fake_load_state(sym):
        return object()   # run_formal_r6 charges one frame load per symbol itself

    monkeypatch.setattr(R, "_git_head_sha", lambda: "SHA")
    monkeypatch.setattr(R, "load_symbol_state", fake_load_state)
    monkeypatch.setattr(R, "treatment_rows_for_symbol",
                        lambda sub, st, ds: (pd.DataFrame({"x": [1]}), None))
    monkeypatch.setattr(R, "_formal_population_gates",
                        lambda l2, big: seen.update({"called": True}) or {})
    monkeypatch.setattr(R, "_effect",
                        lambda big, s, h, side=None: {
                            "side": side or "ALL", "delta_ev_ci_low": 0.0,
                            "delta_ev_ci_high": 0.0, "delta_wrong_gross_ci_low": 0.0,
                            "wlr_gross": 0.0})
    monkeypatch.setattr(R, "formal_verdict", lambda p, s: "NO_IDENTIFIABLE_GROSS_STOP_EDGE")
    monkeypatch.setattr(R, "formal_stop_diagnostics", lambda big, s, h: {})
    monkeypatch.setattr(R, "write_json", lambda p, o: None)
    R.run_formal_r6(allow_full=True, authorized_review_sha="SHA",
                    write_artifacts=False)
    assert seen.get("called") is True


def test_rc_r6_8_formal_runner_rejects_allow_full_false():
    with pytest.raises(RuntimeError, match="FULL_POPULATION_NOT_AUTHORIZED"):
        R.run_formal_r6()


def test_rc_r6_8_formal_runner_rejects_generator_sha_mismatch(monkeypatch):
    monkeypatch.setattr(R, "_git_head_sha", lambda: "HEAD_A")
    with pytest.raises(RuntimeError, match="GENERATOR_SHA_MISMATCH"):
        R.run_formal_r6(allow_full=True, authorized_review_sha="HEAD_B")


def test_rc_r6_8_formal_runner_requires_authorized_sha():
    with pytest.raises(RuntimeError, match="AUTHORIZED_REVIEW_SHA_REQUIRED"):
        R.run_formal_r6(allow_full=True)


def test_rc_r6_11_formal_artifact_duplicate_key_fails(tmp_path):
    n = 41319
    sk = [f"sk{i}" for i in range(13773)]
    df = pd.DataFrame({"semantic_key": sk * 2, "direction_system": ["A9"] * 13773 + ["E9"] * 13773})
    rows = []
    for H in R.HORIZONS:
        t = df.copy(); t["evaluation_horizon"] = H; rows.append(t)
    ok = pd.concat(rows, ignore_index=True)
    p = tmp_path / "ok.parquet"; ok.to_parquet(p, index=False)
    rep = R._verify_formal_artifact(p, set(sk))
    assert rep["rows"] == 82638 and rep["pass"] is True
    dup = pd.concat([ok, ok.iloc[[0]]], ignore_index=True)
    p2 = tmp_path / "dup.parquet"; dup.to_parquet(p2, index=False)
    assert R._verify_formal_artifact(p2, set(sk))["pass"] is False
    assert n > 0


def test_rc_r6_11_invalid_horizon_or_system_fails(tmp_path):
    sk = [f"sk{i}" for i in range(13773)]
    df = pd.DataFrame({"semantic_key": sk * 2,
                       "direction_system": ["A9"] * 13773 + ["XX"] * 13773})
    rows = []
    for H in ("td1", "td3", "bad"):
        t = df.copy(); t["evaluation_horizon"] = H; rows.append(t)
    bad = pd.concat(rows, ignore_index=True)
    p = tmp_path / "bad.parquet"; bad.to_parquet(p, index=False)
    rep = R._verify_formal_artifact(p, set(sk))
    assert rep["pass"] is False
    assert rep["invalid_system_rows"] > 0 and rep["invalid_horizon_rows"] > 0


def test_rc_r6_10_formal_runner_has_no_reference_call():
    assert "scan_structural_stop_reference" not in inspect.getsource(R.run_formal_r6)


def test_rc_r6_12_verdict_categories_are_frozen():
    assert R.GROSS_VERDICTS == (
        "GROSS_STOP_EDGE_SUPPORTED_UNIVERSAL", "GROSS_STOP_HARMFUL",
        "GROSS_SIDE_HETEROGENEITY_REQUIRES_FOLLOWUP",
        "NO_IDENTIFIABLE_GROSS_STOP_EDGE")
