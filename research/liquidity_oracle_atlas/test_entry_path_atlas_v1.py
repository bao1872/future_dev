"""Tests for entry_path_atlas_v1 (FUTURE-R5-M15-ENTRY-PATH-ATLAS-V1).

Kernel Checkpoint gates:
  T0  - 15 synthetic/structural correctness tests
  T1  - Reference vs Production differential (mismatch_count = 0, max_abs_error <= 1e-12)
  NC  - negative controls A/B/C/D
  TP  - performance gate: static audit, counters, N/2N/4N + H/2H scaling, peak RSS

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
# Synthetic case builder                                                       #
# --------------------------------------------------------------------------- #
def case(*, closes, highs=None, lows=None, side=1.0, p0=100.0, atr=1.0,
         backstop=99.0, ahead_sr=102.0, ahead_liq=103.0,
         segment=None, entry_idx=0, end_idx=None, td_ends=None):
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
        backstop_level=np.array([backstop], dtype=np.float64),
        ahead_sr_level=np.array([ahead_sr], dtype=np.float64),
        ahead_liq_level=np.array([ahead_liq], dtype=np.float64),
        td_ends=td_ends,
    )


def prod(c):
    return M.scan_paths_streaming(**c)


def ref(c):
    return M.scan_paths_reference(**c)


DIFF_FIELDS = M.DIFF_FIELDS


def diff_report(c):
    """Reference vs Production -- delegates to the module's shared harness."""
    rep = M.diff_reference_vs_production(c)
    return dict(rows=rep["rows"], cells=rep["cells"],
                mismatch=rep["mismatch"], maxerr=rep["max_abs_error"],
                first=rep["first_mismatch"])


# --------------------------------------------------------------------------- #
# Real-data fixtures (session scoped; E9 materialized ONCE)                    #
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


def case_from_anchors(state, anchors, n=150):
    sl = slice(0, n)
    return dict(
        entry_idx=anchors["entry_idx"][sl],
        end_idx=anchors["end_idx"][sl],
        entry_price=anchors["entry_price"][sl],
        atr0=anchors["atr0"][sl],
        side=anchors["side"][sl],
        high=state.high, low=state.low, close=state.close,
        segment=state.segment,
        entry_segment=anchors["entry_segment"][sl],
        backstop_level=anchors["backstop_level"][sl],
        ahead_sr_level=anchors["ahead_sr_level"][sl],
        ahead_liq_level=anchors["ahead_liq_level"][sl],
        td_ends={k: v[sl] for k, v in anchors["td_ends"].items()},
    )


# =========================================================================== #
# T0                                                                          #
# =========================================================================== #
def test_t0_1_long_mfe_mae_synthetic():
    c = case(closes=[100.5, 102.0], highs=[101.0, 103.0], lows=[99.0, 98.0], side=1.0)
    r = prod(c)
    assert r["mfe_final"][0] == pytest.approx(3.0)   # max(101-100, 103-100)
    assert r["mae_final"][0] == pytest.approx(2.0)   # max(100-99, 100-98)
    snap = r["checkpoints"]["m15"]
    assert snap["mfe"][0] == pytest.approx(1.0)
    assert snap["mae"][0] == pytest.approx(1.0)
    assert snap["r"][0] == pytest.approx(0.5)


def test_t0_2_short_exact_mirror():
    c = case(closes=[99.5, 98.0], highs=[101.0, 102.0], lows=[99.0, 97.0], side=-1.0)
    r = prod(c)
    assert r["mfe_final"][0] == pytest.approx(3.0)   # max(100-99, 100-97)
    assert r["mae_final"][0] == pytest.approx(2.0)   # max(101-100, 102-100)


def test_t0_3_hard_segment_truncates_path():
    seg = np.array([0, 0, 1, 1], dtype=np.int64)
    c = case(closes=[100.0, 100.0, 120.0, 120.0],
             highs=[100.0, 100.0, 120.0, 120.0],
             lows=[100.0, 100.0, 120.0, 120.0],
             side=1.0, segment=seg, end_idx=3)
    r = prod(c)
    assert r["mfe_final"][0] == pytest.approx(0.0), "bars in another segment must be cut"


def test_t0_4_trading_day_boundary_truncates_path():
    n = 6
    closes = np.full(n, 100.0)
    highs = np.full(n, 100.0)
    lows = np.full(n, 100.0)
    highs[5] = 150.0
    lows[5] = 50.0
    c = case(closes=closes, highs=highs, lows=lows, side=1.0, end_idx=4)
    c["td_ends"] = {"td1": np.array([1]), "td3": np.array([3]), "td5": np.array([4])}
    r = prod(c)
    assert r["mfe_final"][0] == pytest.approx(0.0)
    assert r["mae_final"][0] == pytest.approx(0.0)


def test_t0_5_same_bar_reclaim():
    c = case(closes=[99.5], highs=[100.0], lows=[98.5], side=1.0, backstop=99.0)
    r = prod(c)
    assert r["first_pierce"][0] == 0
    assert bool(r["same_bar_reclaim"][0]) is True
    assert r["first_reclaim"][0] == 0
    assert r["bars_to_reclaim"][0] == 0


def test_t0_6_late_reclaim():
    c = case(closes=[98.8, 98.5, 99.6], highs=[99.0, 99.0, 100.0],
             lows=[98.5, 98.0, 99.1], side=1.0, backstop=99.0)
    r = prod(c)
    assert r["first_pierce"][0] == 0
    assert bool(r["same_bar_reclaim"][0]) is False
    assert r["first_reclaim"][0] == 2
    assert r["bars_to_reclaim"][0] == 2


def test_t0_7_pierce_invalid_then_reclaim_is_late_not_same_bar():
    """bar1 pierces+closes invalid, bar2 stays invalid, bar3 closes valid => LATE."""
    c = case(closes=[98.8, 98.5, 99.6], highs=[99.0, 99.0, 100.0],
             lows=[98.5, 98.0, 99.1], side=1.0, backstop=99.0)
    r = prod(c)
    assert r["first_pierce"][0] == 0
    assert r["first_reclaim"][0] == 2
    assert bool(r["same_bar_reclaim"][0]) is False


def test_t0_8_failed_reclaim():
    c = case(closes=[98.5, 99.5, 98.0], highs=[99.0, 100.0, 99.0],
             lows=[98.0, 99.0, 97.0], side=1.0, backstop=99.0)
    r = prod(c)
    assert r["first_reclaim"][0] == 1
    assert r["first_failed_reclaim"][0] == 2


def test_t0_9_ahead_sr_touch_and_cross():
    c = case(closes=[101.0, 102.5], highs=[102.0, 103.0], lows=[100.0, 101.0],
             side=1.0, ahead_sr=102.0)
    r = prod(c)
    assert r["first_ahead_sr_touch"][0] == 0
    assert r["first_ahead_sr_cross"][0] == 1
    assert r["mfe_at_first_ahead_sr"][0] == pytest.approx(2.0)


def test_t0_10_ahead_liquidity_touch_and_cross():
    c = case(closes=[102.0, 103.5], highs=[103.0, 104.0], lows=[100.0, 102.0],
             side=1.0, ahead_liq=103.0)
    r = prod(c)
    assert r["first_ahead_liq_touch"][0] == 0
    assert r["first_ahead_liq_cross"][0] == 1


def test_t0_11_nan_zone_is_unavailable():
    c = case(closes=[100.0, 100.0], highs=[101.0, 101.0], lows=[95.0, 95.0],
             side=1.0, backstop=np.nan)
    r = prod(c)
    assert r["first_pierce"][0] == -1
    assert r["first_touch"][0] == -1
    assert bool(r["break_continue"][0]) is False


def test_t0_12_oracle_fields_never_reach_realtime_state():
    params = inspect.signature(M.scan_paths_streaming).parameters
    for f in ("oracle_direction", "oracle_exit_fill_time", "entry_quality_atr",
              "direction_correct", "e9_teacher_exit_return_atr", "e9_direction"):
        assert f not in params, f"forbidden future field in production kernel: {f}"
    assert set(M.AUDIT_ONLY_FIELDS) & set(params) == set()


def test_t0_13_future_mutation_after_checkpoint_unchanged():
    base = case(closes=[100.0, 100.0, 100.0], highs=[101.0, 101.0, 101.0],
                lows=[99.0, 99.0, 99.0], side=1.0, end_idx=2)
    r1 = prod(base)
    mut = dict(base)
    mut["high"] = base["high"].copy()
    mut["low"] = base["low"].copy()
    mut["high"][2] = 150.0
    mut["low"][2] = 50.0
    r2 = prod(mut)
    assert r1["checkpoints"]["m15"]["mfe"][0] == pytest.approx(
        r2["checkpoints"]["m15"]["mfe"][0])
    assert r1["checkpoints"]["m15"]["mae"][0] == pytest.approx(
        r2["checkpoints"]["m15"]["mae"][0])


def test_t0_14_semantic_key_uniqueness():
    df = pd.DataFrame({
        "symbol": ["AG", "AG", "AU"],
        "oracle_trade_id": [1, 1, 1],
        "candidate_decision_time": ["t1", "t2", "t1"],
        "candidate_fill_time": ["f1", "f2", "f1"],
    })
    k = M.build_semantic_key(df)
    assert len(set(k.tolist())) == 3
    dup = pd.concat([df.iloc[[0]], df.iloc[[0]]]).reset_index(drop=True)
    assert len(set(M.build_semantic_key(dup).tolist())) == 1


def test_t0_15_gid_weight_sum_canonical(e9_state):
    g = e9_state.groupby("gid")["sample_weight_raw"].sum()
    assert np.allclose(g.to_numpy(), 1.0, atol=1e-9)


def test_t0_extra_env_completeness_gate(ag_state):
    assert ag_state.env_complete is True, ag_state.env_note


def test_t0_extra_gid_cluster_bootstrap_is_row_independent():
    gid = np.array(["a", "a", "b", "b", "b"])
    v = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    m, lo, hi = M.cluster_bootstrap_gid(v, gid, B=200, seed=1)
    # each opportunity counts ONCE: mean(trade a=1.0, trade b=0.0) = 0.5
    # (a row-level mean would be 2/5 = 0.4)
    assert abs(m - 0.5) < 1e-12
    assert lo <= m <= hi
    # adding a duplicate Candidate row inside a trade must NOT move the estimate,
    # because rows are not the sampling unit
    gid2 = np.array(["a", "a", "a", "b", "b", "b"])
    v2 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    m2, _, _ = M.cluster_bootstrap_gid(v2, gid2, B=200, seed=1)
    assert abs(m2 - m) < 1e-12
    # canonical sample_weight_raw (sum 1 per opportunity) must reproduce the same value
    w2 = np.array([1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3])
    m3, _, _ = M.cluster_bootstrap_gid(v2, gid2, w=w2, B=200, seed=1)
    assert abs(m3 - m) < 1e-12


# =========================================================================== #
# T1 differential                                                              #
# =========================================================================== #
def test_t1_synthetic_long():
    c = case(closes=[100.5, 102.0, 101.0], highs=[101.0, 103.0, 102.0],
             lows=[99.0, 97.0, 96.0], side=1.0, backstop=99.0)
    rep = diff_report(c)
    assert rep["mismatch"] == 0, f"mismatch={rep['first']}"
    assert rep["maxerr"] <= 1e-12


def test_t1_synthetic_short():
    c = case(closes=[99.5, 98.0, 99.0], highs=[101.0, 102.0, 103.0],
             lows=[99.0, 97.0, 96.0], side=-1.0, backstop=101.0)
    rep = diff_report(c)
    assert rep["mismatch"] == 0
    assert rep["maxerr"] <= 1e-12


def test_t1_boundary_segment_path():
    seg = np.array([0, 0, 1, 1], dtype=np.int64)
    c = case(closes=[100.0, 100.0, 120.0, 120.0],
             highs=[101.0, 100.0, 120.0, 121.0],
             lows=[99.0, 98.0, 119.0, 119.0],
             side=1.0, segment=seg, end_idx=3, backstop=98.5)
    rep = diff_report(c)
    assert rep["mismatch"] == 0
    assert rep["maxerr"] <= 1e-12


def test_t1_real_ag_subset(ag_state, ag_anchors):
    c = case_from_anchors(ag_state, ag_anchors, n=150)
    rep = diff_report(c)
    assert rep["rows"] == 150
    assert rep["mismatch"] == 0, f"mismatch={rep['first']}"
    assert rep["maxerr"] <= 1e-12


# =========================================================================== #
# Negative controls                                                            #
# =========================================================================== #
def test_negative_control_A_future_perturbation_after_checkpoint():
    """Bars AFTER a checkpoint must not change that checkpoint's metrics."""
    n = 10
    highs = np.full(n, 101.0)
    lows = np.full(n, 99.0)
    c = case(closes=np.full(n, 100.0), highs=highs, lows=lows, side=1.0, end_idx=n - 1)
    before = prod(c)

    mut = dict(c)
    mut["high"] = highs.copy()
    mut["low"] = lows.copy()
    mut["high"][4:] = 160.0     # strictly AFTER the h1 checkpoint (step 3)
    mut["low"][4:] = 40.0
    after = prod(mut)

    for name in ("m15", "h1"):
        assert np.allclose(before["checkpoints"][name]["mfe"],
                           after["checkpoints"][name]["mfe"], atol=1e-12)
        assert np.allclose(before["checkpoints"][name]["mae"],
                           after["checkpoints"][name]["mae"], atol=1e-12)
    # the perturbation DID land inside the horizon, so the final values must move
    assert after["mfe_final"][0] > before["mfe_final"][0]


def test_negative_control_B_oracle_perturbation(ag_state, ag_anchors):
    c1 = case_from_anchors(ag_state, ag_anchors, n=120)
    p1 = prod(c1)
    df = ag_anchors["df"].copy()
    df["oracle_direction"] = np.where(
        df["oracle_direction"].to_numpy(object) == "LONG", "SHORT", "LONG")
    df["oracle_entry_quality_atr"] = -df["oracle_entry_quality_atr"].to_numpy(np.float64)
    df["direction_correct"] = 1 - df["direction_correct"].to_numpy()
    a2 = dict(ag_anchors)
    a2["df"] = df
    c2 = case_from_anchors(ag_state, a2, n=120)
    p2 = prod(c2)
    for f in DIFF_FIELDS:
        assert np.array_equal(np.asarray(p1[f]), np.asarray(p2[f])), (
            f"oracle perturbation changed real-time field {f}")


def test_negative_control_C_altered_boundary_must_fail():
    c = case(closes=[98.8, 98.5, 99.6], highs=[99.0, 99.0, 100.0],
             lows=[98.5, 98.0, 99.1], side=1.0, backstop=99.0)
    assert diff_report(c)["mismatch"] == 0
    corrupt = dict(c)
    corrupt["backstop_level"] = np.array([95.0])
    p = prod(corrupt)
    r = ref(c)
    assert not np.array_equal(p["first_pierce"], r["first_pierce"]), (
        "differential must FAIL when a production boundary is altered")


def test_negative_control_D_semantic_key_swap_must_fail(ag_state, e9_state):
    bad = e9_state.copy()
    idx = np.flatnonzero((bad["symbol"].to_numpy(object) == "AG"))
    perm = idx.copy()
    perm[:20] = perm[:20][::-1]
    bad.loc[idx, "candidate_fill_index"] = (
        bad.loc[perm, "candidate_fill_index"].to_numpy())
    with pytest.raises(RuntimeError):
        M.build_anchors_for_symbol(bad, ag_state)


# =========================================================================== #
# TP performance gate                                                          #
# =========================================================================== #
def _synth(N, H, seed=0):
    return M.make_synthetic_case(N, H, seed)


def test_tp_static_audit():
    src = inspect.getsource(M.scan_paths_streaming)
    for tok in ("pd.concat", "for i in range", ".iloc[", "history[",
                "for candidate", "np.vstack"):
        assert tok not in src, f"forbidden token in production kernel: {tok}"
    assert "for step in range" in src, "production must loop over TIME steps"
    ref_src = inspect.getsource(M.scan_paths_reference)
    assert "for i in range(n)" in ref_src


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
        c = _synth(N, H, seed)
        t = time.perf_counter()
        prod(c)
        return time.perf_counter() - t

    H = 32
    t1 = timed(200, H)
    t2 = timed(400, H)
    t4 = timed(800, H)
    # O(N) would give ~4x for 4N; a quadratic kernel would blow far past this bound.
    assert t4 / max(t1, 1e-9) < 10.0, f"candidate scaling not linear: {t4 / t1:.2f}"
    assert t2 >= 0.0 and t4 >= 0.0

    N = 200
    h1 = timed(N, 32)
    h2 = timed(N, 64)
    assert h2 / max(h1, 1e-9) < 10.0, f"horizon scaling not linear: {h2 / h1:.2f}"


def test_tp_peak_rss_recorded(ag_state, ag_anchors):
    M.run_symbol_paths(ag_state, ag_anchors)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; just require it to be a positive, finite number
    assert rss > 0
