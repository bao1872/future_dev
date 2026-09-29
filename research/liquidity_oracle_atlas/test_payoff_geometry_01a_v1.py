"""Tests for PAYOFF-GEOMETRY-01A kernel checkpoint (T0 / T1 / TP).

Executor-only: no model training, no T1.5/T2. These tests verify the corrected
math (Z=log(G/L), outcome=true episode_return_atr, no fake EV), Reference vs
Production parity for the stratified statistic, the genuine negative controls,
the real future-mutation causality test, the pre/post key-alignment (no silent
drop), and performance scaling.
"""
import numpy as np
import pandas as pd

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
# No silent drop — REAL pre-join vs post-join count                            #
# --------------------------------------------------------------------------- #
def test_no_silent_drop_real_count():
    df = M.load_audit_frame()
    # every OOF row must survive; load_audit_frame raises otherwise, so reaching
    # here already implies pre == post. We also assert the explicit invariant.
    assert len(df) > 0
    for c in ("p_win", "G", "L", "true_episode_return_atr", "log_gl", "weights"):
        assert df[c].notna().all(), f"NaN in {c}"
    assert df.duplicated(subset=["symbol", "decision_bar", "side"]).sum() == 0


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
    assert res["synthetic_note"].startswith("DIAGNOSTIC_ONLY")


def test_diag_old_payoff_model_present():
    df = M.load_audit_frame(sample_n=800)
    rows = M.diag_old_payoff_model(df)
    assert len(rows) == M.N_P_BINS
    assert "mean_predicted_rr" in rows[0]


# --------------------------------------------------------------------------- #
# TP — performance scaling O(N log N)                                          #
# --------------------------------------------------------------------------- #
def test_tp_performance_ratios():
    res = M.tp_microbenchmark()
    assert res["ratio_2N"] < 3.0, res
    assert res["ratio_4N"] < 3.0, res


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
    assert pkt["LOCAL_SHA"] == pkt["REMOTE_SHA"]  # real SHA, no PENDING_PUSH
    assert pkt["GENERATOR_CODE_SHA"] != "unknown"
    assert pkt["governance"]["full_population_high_low_run"] is False
    assert pkt["governance"]["t1_5_run"] is False
    assert pkt["governance"]["t2_run"] is False
    # report pulled from the same object as the artifact -> match by construction
    assert "stratified_D_geometry_audit" in pkt["T1"]
