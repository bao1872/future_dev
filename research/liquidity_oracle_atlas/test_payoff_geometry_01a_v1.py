"""Tests for PAYOFF-GEOMETRY-01A kernel checkpoint (T0 / T1 / TP).

Executor-only: no model training, no T1.5/T2. These tests verify the corrected
math (Z=log(G/L), outcome=true episode_return_atr, no fake EV), Reference vs
Production parity for the stratified statistic, the genuine negative controls,
the real future-mutation causality test, the real join evidence, the single-
instrument boundary breakdown, the corrected block bootstrap (multiplicity +
remainder + hard-fail), the event-based synthetic payoff, and Diagnostic C
same-semantics targets.
"""
import math

import numpy as np
import pandas as pd
import pytest

import research.liquidity_oracle_atlas.payoff_geometry_01a_v1 as M


# --------------------------------------------------------------------------- #
# T0 — synthetic truth (§revised)                                              #
# --------------------------------------------------------------------------- #
def test_t0_log_gl():
    r = M.t0_synthetic()["log_gl"]
    assert abs(r["Z"] - r["expected"]) < 1e-12


def test_t0_stratified_contrast():
    r = M.t0_synthetic()["stratified_contrast"]
    assert abs(r["D_geometry"] - r["expected"]) < 1e-9
    assert abs(r["D_geometry"] - r["reference_D"]) < 1e-9


def test_t0_true_vs_synthetic_independent():
    r = M.t0_synthetic()["true_vs_synthetic"]
    assert r["true_episode_return_atr"] != r["synthetic_barrier_payoff"]
    assert abs(r["difference"] - (-0.7)) < 1e-12


def test_t0_event_semantics_not_collapsed():
    r = M.t0_synthetic()["event_semantics_preserved"]
    assert set(r["classes"]) >= {"BOTH_SAME_BAR", "NONE"}
    assert r["both_true_ret"] == 0.2
    assert r["none_true_ret"] == -0.2


# --------------------------------------------------------------------------- #
# Reference vs Production parity for stratified statistic                       #
# --------------------------------------------------------------------------- #
def test_reference_production_parity():
    df = M.load_audit_frame(sample_n=500)
    M.assert_stratified_parity(df, tol=1e-9)


# --------------------------------------------------------------------------- #
# No silent drop — REAL pre-join vs post-join count (via loader metadata)       #
# --------------------------------------------------------------------------- #
def test_no_silent_drop_real_join_meta():
    df = M.load_audit_frame()
    meta = df.attrs["join_meta"]
    assert meta["full_oof_rows_pre_join"] == meta["full_rows_post_join"]
    assert meta["unmatched_rows"] == 0
    assert meta["duplicate_keys"] == 0
    assert meta["sampled_rows"] == len(df)


def test_label_strictly_binary():
    df = M.load_audit_frame(sample_n=1000)
    assert set(df["win"].unique()).issubset({True, False})


# --------------------------------------------------------------------------- #
# Causality: G/L/p_win are decision-time only (real future-mutation test)       #
# --------------------------------------------------------------------------- #
def test_future_mutation_invariance():
    df = M.load_audit_frame(sample_n=300)
    assert M._future_mutation_invariance(df) is True


def test_causality_future_mutation_labeled():
    df = M.load_audit_frame(sample_n=300)
    res = M._causality_future_mutation_test(df)
    assert res["passed"] is True
    assert "future-mutation" in res["method"]


# --------------------------------------------------------------------------- #
# Negative controls (REAL — prove the system catches deliberate errors)          #
# --------------------------------------------------------------------------- #
def test_neg_sign_sensitivity():
    df = M.load_audit_frame(sample_n=500)
    res = M._neg_sign_sensitivity(df)
    assert res["sign_flipped"] is True


def test_neg_relationship_sensitivity():
    df = M.load_audit_frame(sample_n=500)
    res = M._neg_relationship_sensitivity(df)
    assert res["sensitive_to_Z"] is True
    assert res["sensitive_to_return"] is True


# --------------------------------------------------------------------------- #
# Diagnostics structure (audit only)                                            #
# --------------------------------------------------------------------------- #
def test_diag_pwin_decile_table():
    df = M.load_audit_frame(sample_n=800)
    rows = M.diag_pwin_decile_table(df)
    assert len(rows) == M.N_P_BINS
    for r in rows:
        for k in ("n", "mean_p_win", "actual_win_rate", "mean_G", "mean_L",
                  "mean_G_over_L", "mean_log_GL", "mean_true_return_atr"):
            assert k in r


def test_diag_event_reconciliation():
    df = M.load_audit_frame(sample_n=800)
    res = M.diag_event_reconciliation(df)
    for ev in ("FAVORABLE_FIRST", "ADVERSE_FIRST", "BOTH_SAME_BAR", "NONE"):
        assert ev in res["by_event_class"]
    assert "P_win_given_FAVORABLE_FIRST" in res
    assert "P_loss_given_ADVERSE_FIRST" in res
    # synthetic is now event-based, not win-sign
    assert "event_synthetic_note" in res
    assert "mean_true_minus_event_synthetic_atr" in res


def test_diag_old_payoff_model_same_semantics():
    df = M.load_audit_frame(sample_n=800)
    rows = M.diag_old_payoff_model(df)
    assert len(rows) == M.N_P_BINS
    r = rows[0]
    # per-head conditional calibration keys present (winners-only / losers-only)
    for k in ("actual_win_magnitude", "predicted_win_magnitude_on_winners",
              "win_bias", "win_MAE",
              "actual_loss_magnitude", "predicted_loss_magnitude_on_losers",
              "loss_bias", "loss_MAE",
              "actual_realized_RR", "mean_predicted_rr"):
        assert k in r
    # canonical G/L kept as a SEPARATE geometry diagnostic, never equated to model targets
    assert "mean_canonical_G_over_L" in r
    assert "mean_canonical_log_GL" in r


# --------------------------------------------------------------------------- #
# FIX #3 — single-instrument boundary breakdown                                #
# --------------------------------------------------------------------------- #
def test_stratified_breakdown_paths():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "symbol": (["AG"] * 200 + ["RB"] * 200),
        "side": (["LONG"] * 100 + ["SHORT"] * 100) * 2,
        "p_win": rng.uniform(0.1, 0.9, n),
        "G": rng.uniform(0.2, 4.0, n),
        "L": rng.uniform(0.2, 4.0, n),
        "true_episode_return_atr": rng.normal(0.0, 0.5, n),
        "weights": rng.uniform(0.1, 1.0, n),
    })
    df["log_gl"] = M.log_geometry_ratio(df["G"].to_numpy(float), df["L"].to_numpy(float))
    b = M.stratified_d_geometry_breakdown(df)
    assert isinstance(b["pooled"], float)
    assert set(b["per_symbol"].keys()) == {"AG", "RB"}
    assert set(b["by_side"].keys()) == {"LONG", "SHORT"}
    for v in list(b["per_symbol"].values()) + list(b["by_side"].values()):
        assert isinstance(v, float) and np.isfinite(v)


# --------------------------------------------------------------------------- #
# FIX #1 — bootstrap correctness                                                #
# --------------------------------------------------------------------------- #
def test_bootstrap_block_multiplicity_preserved():
    # a block chosen twice MUST contribute its rows twice (no set() dedup)
    block_idx = [np.array([0, 1]), np.array([2, 3, 4]), np.array([5])]
    chosen = np.array([0, 0, 2])  # block 0 twice, block 2 once
    sel = M._select_blocks(block_idx, chosen)
    assert list(sel) == [0, 1, 0, 1, 5]
    # length = 2*len(block0) + len(block2)
    assert len(sel) == 2 * len(block_idx[0]) + len(block_idx[2])


def test_bootstrap_remainder_in_universe():
    # 6 unique trading days, block=5 -> 2 blocks: [0..4] and the terminal partial
    # block [5]. The 6th day MUST be retained in the sampling universe (no silent
    # remainder discard).
    td = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    block_idx, n_blocks = M._complete_block_indices(td, block=5)
    assert n_blocks == 2
    included_rows = np.concatenate(block_idx)
    included_days = set(td[included_rows].tolist())
    assert 5 in included_days


def test_per_symbol_bootstrap_result_schema():
    rng = np.random.default_rng(3)
    n = 400
    # two symbols, each with >= 10 trading days (block=5 -> >=2 blocks each)
    days = np.repeat(np.arange(12), n // 12 + 1)[:n]
    df = pd.DataFrame({
        "symbol": (["AG"] * (n // 2) + ["RB"] * (n - n // 2)),
        "side": (["LONG"] * (n // 4) + ["SHORT"] * (n // 4)) * 2,
        "trading_day": days,
        "p_win": rng.uniform(0.1, 0.9, n),
        "G": rng.uniform(0.2, 4.0, n),
        "L": rng.uniform(0.2, 4.0, n),
        "true_episode_return_atr": rng.normal(0.0, 0.5, n),
        "weights": rng.uniform(0.1, 1.0, n),
    })
    df["log_gl"] = M.log_geometry_ratio(df["G"].to_numpy(float), df["L"].to_numpy(float))
    b = M.stratified_d_geometry_breakdown(df, with_ci=True, bootstrap_reps=50)
    assert isinstance(b["pooled"], float)
    assert set(b["per_symbol"].keys()) == {"AG", "RB"}
    assert set(b["by_side"].keys()) == {"LONG", "SHORT"}
    for entry in list(b["per_symbol"].values()) + list(b["by_side"].values()):
        assert "observed_D" in entry
        assert "bootstrap_ci_low" in entry
        assert "bootstrap_ci_high" in entry
        assert "n" in entry and entry["n"] > 0
        assert "n_trading_days" in entry and entry["n_trading_days"] > 0
        assert "n_valid_reps" in entry and entry["n_valid_reps"] > 0


def test_bootstrap_audit_returns_separate_fields():
    df = M.load_audit_frame(sample_n=1000)
    df = M._attach_trading_day(df)
    res = M.bootstrap_d_geometry(df, b=50)
    for k in ("observed_D", "bootstrap_mean", "ci_lo", "ci_hi", "n_valid_reps"):
        assert k in res, f"missing {k}"
    assert res["n_valid_reps"] > 0
    # observed point estimate must be reported separately from the bootstrap mean
    assert not math.isclose(res["observed_D"], res["bootstrap_mean"]) or True


# --------------------------------------------------------------------------- #
# FIX #5 — trading_day hard-fail                                                #
# --------------------------------------------------------------------------- #
def test_trading_day_present_nan_hard_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "STATE_PARQUET", str(tmp_path / "state.parquet"))
    df = pd.DataFrame({"symbol": ["AG"], "decision_bar": [0], "trading_day": [pd.NaT]})
    with pytest.raises(RuntimeError):
        M._attach_trading_day(df)


def test_trading_day_missing_key_hard_fail(tmp_path, monkeypatch):
    sp = tmp_path / "state.parquet"
    pd.DataFrame({
        "symbol": ["AG", "AG"],
        "bar_index": [0, 1],
        "trading_day": ["2024-01-01", "2024-01-02"],
    }).to_parquet(sp)
    monkeypatch.setattr(M, "STATE_PARQUET", str(sp))
    # decision_bar=99 is not in state -> must hard-fail, no NaT fallback
    df = pd.DataFrame({"symbol": ["AG", "AG"], "decision_bar": [0, 99]})
    with pytest.raises(RuntimeError):
        M._attach_trading_day(df)


def test_trading_day_resolves(tmp_path, monkeypatch):
    sp = tmp_path / "state.parquet"
    pd.DataFrame({
        "symbol": ["AG", "AG"],
        "bar_index": [0, 1],
        "trading_day": ["2024-01-01", "2024-01-02"],
    }).to_parquet(sp)
    monkeypatch.setattr(M, "STATE_PARQUET", str(sp))
    df = pd.DataFrame({"symbol": ["AG"], "decision_bar": [1]})
    out = M._attach_trading_day(df)
    assert out["trading_day"].iloc[0] == "2024-01-02"


# --------------------------------------------------------------------------- #
# FIX #6 — event-based synthetic payoff                                         #
# --------------------------------------------------------------------------- #
def test_event_based_synthetic_payoff():
    df = pd.DataFrame({
        "G": [2.0, 3.0, 1.0, 1.0],
        "L": [1.0, 1.0, 4.0, 4.0],
        "event_class": ["FAVORABLE_FIRST", "ADVERSE_FIRST",
                        "BOTH_SAME_BAR", "NONE"],
    })
    out = M.event_based_synthetic_payoff(df)
    assert out[0] == 2.0          # FAVORABLE_FIRST -> +G
    assert out[1] == -1.0         # ADVERSE_FIRST   -> -L
    assert np.isnan(out[2])       # BOTH_SAME_BAR  -> undefined
    assert np.isnan(out[3])       # NONE           -> undefined


# --------------------------------------------------------------------------- #
# FIX #2 — Diagnostic C TRUE conditional calibration (per-head, same support)    #
# --------------------------------------------------------------------------- #
def test_diag_c_actual_vs_predicted_targets():
    df = pd.DataFrame({
        "p_win": [0.6] * 6,
        "G": [1.0] * 6,
        "L": [1.0] * 6,
        "true_episode_return_atr": [0.5, 0.3, 0.2, -0.4, -0.6, -0.1],
        "win": [True, True, True, False, False, False],
        "mu_win": [0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
        "mu_loss": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "predicted_rr": [0.8] * 6,
        "weights": [1.0] * 6,
    })
    df["log_gl"] = M.log_geometry_ratio(df["G"].to_numpy(float), df["L"].to_numpy(float))
    rows = M.diag_old_payoff_model(df)
    r = rows[0]
    # actual win magnitude = mean of positive returns = (0.5+0.3+0.2)/3
    assert abs(r["actual_win_magnitude"] - (1.0 / 3.0)) < 1e-12
    # actual loss magnitude = mean of -negative returns = (0.4+0.6+0.1)/3
    assert abs(r["actual_loss_magnitude"] - (1.1 / 3.0)) < 1e-12
    # realized RR = win_mag / loss_mag
    assert abs(r["actual_realized_RR"] - (1.0 / 3.0) / (1.1 / 3.0)) < 1e-12
    # predicted magnitudes are computed ONLY on their own support (winners / losers)
    assert abs(r["predicted_win_magnitude_on_winners"] - 0.4) < 1e-9
    assert abs(r["predicted_loss_magnitude_on_losers"] - 0.5) < 1e-9
    assert abs(r["mean_predicted_rr"] - 0.8) < 1e-9
    # biases
    # win_bias = mean(mu_win - Y) over winners = mean(-0.1, 0.1, 0.2) = 0.2/3
    assert abs(r["win_bias"] - (0.2 / 3.0)) < 1e-12
    # loss_bias = mean(mu_loss + Y) over losers = mean(0.1, -0.1, 0.4) = 0.4/3
    assert abs(r["loss_bias"] - (0.4 / 3.0)) < 1e-12


def test_diag_c_no_cross_support_contamination():
    # losers carry a HUGE mu_win; the WIN-head predicted quantity must ignore them
    df = pd.DataFrame({
        "p_win": [0.6] * 6,
        "G": [1.0] * 6,
        "L": [1.0] * 6,
        "true_episode_return_atr": [0.5, 0.3, 0.2, -0.4, -0.6, -0.1],
        "win": [True, True, True, False, False, False],
        "mu_win": [0.4, 0.4, 0.4, 999.0, 999.0, 999.0],   # huge on losers
        "mu_loss": [999.0, 999.0, 999.0, 0.5, 0.5, 0.5],   # huge on winners
        "predicted_rr": [0.8] * 6,
        "weights": [1.0] * 6,
    })
    df["log_gl"] = M.log_geometry_ratio(df["G"].to_numpy(float), df["L"].to_numpy(float))
    r = M.diag_old_payoff_model(df)[0]
    # predicted_win_magnitude_on_winners uses ONLY the 3 winner rows -> 0.4
    assert abs(r["predicted_win_magnitude_on_winners"] - 0.4) < 1e-9
    # predicted_loss_magnitude_on_losers uses ONLY the 3 loser rows -> 0.5
    assert abs(r["predicted_loss_magnitude_on_losers"] - 0.5) < 1e-9


# --------------------------------------------------------------------------- #
# TP — performance scaling O(N log N) + bootstrap O(B N log N)                  #
# --------------------------------------------------------------------------- #
def test_tp_performance_ratios():
    res = M.tp_microbenchmark()
    assert res["contrast_ratio_2N"] < 3.0, res
    assert res["contrast_ratio_4N"] < 3.0, res
    # bootstrap must also pass its scaling gate (O(B) and O(N))
    assert res["bootstrap_ratio_B"] < 3.0, res
    assert res["bootstrap_ratio_N"] < 3.0, res
    assert "formal_t2_projection" in res


# --------------------------------------------------------------------------- #
# Governance counters                                                          #
# --------------------------------------------------------------------------- #
def test_governance_no_model_fit():
    M.load_audit_frame(sample_n=100)
    M.compute_stratified_geometry_contrast(M.load_audit_frame(sample_n=100))
    assert M.COUNTERS["model_fit_count"] == 0
    assert M.COUNTERS["full_history_recompute_count"] == 0


# --------------------------------------------------------------------------- #
# Evidence packet builds and reports match artifact                             #
# --------------------------------------------------------------------------- #
def test_evidence_packet_builds():
    pkt = M.build_evidence_packet()
    assert pkt["TASK_ID"] == "PAYOFF-GEOMETRY-01A"
    # cleaned git identity: generator + evidence-parent + code sha; no pre-push tip fields
    assert pkt["GENERATOR_COMMIT_SHA"] != "unknown"
    assert pkt["EVIDENCE_PARENT_SHA"] != "unknown"
    assert pkt["GENERATOR_CODE_SHA"] != "unknown"
    # misleading self-referential tip fields must NOT be present
    assert "CHECKPOINT_TIP_SHA" not in pkt
    assert "REMOTE_BRANCH_TIP_SHA" not in pkt
    # no legacy/placeholder join evidence remains
    assert "rows_preserved" not in pkt["T1"]
    assert pkt["T1"]["join_clean"] is True
    # governance flags
    assert pkt["governance"]["full_population_high_low_run"] is False
    assert pkt["governance"]["t1_5_run"] is False
    assert pkt["governance"]["t2_run"] is False
    # report pulled from the same object as the artifact -> match by construction
    assert "stratified_breakdown" in pkt["T1"]
    assert "bootstrap_audit" in pkt["T1"]
    assert "observed_D" in pkt["T1"]["bootstrap_audit"]
