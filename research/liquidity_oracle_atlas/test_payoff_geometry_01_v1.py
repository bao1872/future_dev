"""Tests for PAYOFF-GEOMETRY-01 kernel checkpoint (T0 / T1 / TP).

Executor-only: no model training, no T1.5/T2. These tests verify the frozen
kernels, the geometry gate, the reference/production differential, negative
controls, performance scaling, and the no-silent-drop guarantee.
"""
import numpy as np
import pandas as pd

import research.liquidity_oracle_atlas.payoff_geometry_01_v1 as M


# --------------------------------------------------------------------------- #
# T0 — synthetic truth (§26)                                                   #
# --------------------------------------------------------------------------- #
def test_t0_case1_ev_and_realized():
    r = M.t0_synthetic()["case1"]
    assert abs(r["known_ev_score"] - 0.5) < 1e-12
    assert abs(r["realized_label_return_atr"] - 1.5) < 1e-12
    assert abs(r["prod_known_ev"] - 0.5) < 1e-12
    assert abs(r["prod_realized"] - 1.5) < 1e-12


def test_t0_case2_ev_and_realized():
    r = M.t0_synthetic()["case2"]
    # EV unchanged by label; realized flips sign (proves label is NOT used in score)
    assert abs(r["known_ev_score"] - 0.5) < 1e-12
    assert abs(r["realized_label_return_atr"] - (-1.0)) < 1e-12


# --------------------------------------------------------------------------- #
# Reference vs Production differential (§21)                                    #
# --------------------------------------------------------------------------- #
def test_reference_production_differential():
    df = M.load_audit_frame(sample_n=500)
    diff = M.differential_check(df)
    assert diff["mismatch_count"] == 0, diff
    assert diff["max_abs_error"] < 1e-12
    assert diff["rows_compared"] == 500


# --------------------------------------------------------------------------- #
# Geometry gate (§20) — real G/L must be VARIABLE (Case A)                    #
# --------------------------------------------------------------------------- #
def test_geometry_gate_variable():
    df = M.load_audit_frame(sample_n=2000)
    gate = M.payoff_geometry_gate(
        df["tp_distance_atr"].to_numpy(float), df["sl_distance_atr"].to_numpy(float))
    assert gate == "VARIABLE_GEOMETRY_CONTINUE"


def test_fixed_geometry_raises():
    # Case B: constant G/L must trigger rank-equivalence STOP
    n = 10
    p = np.linspace(0.3, 0.8, n)
    g = np.full(n, 2.0)
    l = np.full(n, 1.0)
    try:
        M._enforce_variable_geometry(g, l, p)
        raised = False
    except SystemExit:
        raised = True
    assert raised


# --------------------------------------------------------------------------- #
# Negative controls (§22)                                                      #
# --------------------------------------------------------------------------- #
def test_neg_math_wrongsign():
    assert M._neg_math_wrongsign_raises() is True


def test_neg_label_swap_changes_realized():
    df = M.load_audit_frame(sample_n=50)
    assert M._neg_label_swap_raises(df) is True


def test_neg_key_misalign_raises():
    assert M._neg_key_misalign_raises() is True


def test_neg_causality_unchanged():
    assert M._neg_causality_unchanged() is True


def test_neg_reference_call_in_prod():
    df = M.load_audit_frame(sample_n=50)
    assert M._neg_reference_call_in_prod(df) is True


# --------------------------------------------------------------------------- #
# No silent drop / exact key alignment (§13)                                   #
# --------------------------------------------------------------------------- #
def test_no_silent_drop_full():
    df = M.load_audit_frame()
    assert len(df) > 0
    for c in M.REQUIRED_COLUMNS:
        assert df[c].notna().all(), f"NaN in {c}"
    # every OOF key must be present (loader raises otherwise)
    assert df.duplicated(subset=["symbol", "decision_bar", "side"]).sum() == 0


def test_label_strictly_binary():
    df = M.load_audit_frame(sample_n=1000)
    assert set(df["y_win"].unique()).issubset({0, 1})


# --------------------------------------------------------------------------- #
# TP — performance scaling O(N) (§23/§25)                                      #
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
    assert M.COUNTERS["model_fit_count"] == 0
    assert M.COUNTERS["full_history_recompute_count"] == 0
    assert M.COUNTERS["reference_call_count"] == 0
