"""Tests for the R13.5 TD5 decomposed EV-error closure audit (plan §41).

No DEV VAL, no old TEST, no model fit, no Direction fit.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import pytest

from research.liquidity_oracle_atlas import (
    run_decomposed_v2_research as R,
)
from research.liquidity_oracle_atlas import (
    decomposed_value_closure_audit_v1 as C,
)

ARCH_PRIMARY = C.ARCH_PRIMARY
ARCH_SECONDARY = C.ARCH_SECONDARY
POP_ALL = C.POP_ALL
POP_ROOT = C.POP_ROOT


# --------------------------------------------------------------------------- #
# Loaders (cheap; no fit, no VAL/TEST)                                         #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def a0_oof():
    return C.load_td5_oof(ARCH_PRIMARY)


@pytest.fixture(scope="module")
def a1_oof():
    return C.load_td5_oof(ARCH_SECONDARY)


# 1. TD5 only
def test_td5_only(a0_oof):
    assert set(a0_oof["horizon"].unique()) == {C.HORIZON}


# 2. Exactly five OOF folds
def test_exactly_five_folds(a0_oof):
    assert set(a0_oof["fold"].unique()) == {0, 1, 2, 3, 4}


# 3. A0 OOF keys unique
def test_a0_oof_keys_unique(a0_oof):
    key = ["symbol", "decision_bar", "side", "horizon"]
    assert not a0_oof.duplicated(key).any()


# 4. A1 OOF keys unique
def test_a1_oof_keys_unique(a1_oof):
    key = ["symbol", "decision_bar", "side", "horizon"]
    assert not a1_oof.duplicated(key).any()


# 5. A0/A1 key universes identical
def test_arch_key_universes_identical(a0_oof, a1_oof):
    key = ["symbol", "decision_bar", "side", "horizon"]
    set_a0 = set(map(tuple, a0_oof[key].to_numpy()))
    set_a1 = set(map(tuple, a1_oof[key].to_numpy()))
    assert set_a0 == set_a1


# 6 & 7. OOF actual Y and sample weights exactly match TRAIN labels
def test_oof_labels_exact_match(a0_oof):
    # verify_oof_labels asserts exact equality internally
    merged = C.verify_oof_labels(a0_oof)
    diff_y = (
        merged["episode_return_atr_oof"].to_numpy(float)
        - merged["episode_return_atr_label"].to_numpy(float)
    )
    diff_w = (
        merged["sample_weight_oof"].to_numpy(float)
        - merged["sample_weight_label"].to_numpy(float)
    )
    assert np.max(np.abs(diff_y)) <= 1e-12
    assert np.max(np.abs(diff_w)) <= 1e-12


# 8/9/10/11. No DEV VAL / old TEST / model fit / Direction fit
def test_no_val_test_or_fit_in_source():
    src = open(os.path.abspath(C.__file__)).read()
    assert "read_val_labels" not in src
    assert "read_test_labels" not in src
    assert "labels_test" not in src
    assert "labels_val" not in src
    assert "decomposed_models_v2" not in src
    assert ".fit(" not in src


# 12. Exact per-row EV-error decomposition
def test_per_row_ev_error_decomposition(a0_oof):
    df = C.add_closure_columns(a0_oof)
    np.testing.assert_allclose(
        df["total_ev_error"].to_numpy(float),
        (df["probability_error_component"] + df["magnitude_error_component"]).to_numpy(float),
        rtol=1e-10, atol=1e-10,
    )


# 13. Exact MSE decomposition including cross term
def test_mse_decomposition_identity(a0_oof):
    df = C.add_closure_columns(a0_oof)
    s = C.closure_summary(df)
    np.testing.assert_allclose(
        s["total_mse"],
        s["probability_mse_component"] + s["magnitude_mse_component"] + s["cross_term"],
        rtol=1e-10, atol=1e-10,
    )


# 14. Oracle-sign uses actual sign only; magnitude_error = oracle_sign - Y
def test_oracle_sign_uses_actual_sign(a0_oof):
    df = C.add_closure_columns(a0_oof)
    y = df["episode_return_atr"].to_numpy(float)
    z = (y > 0).astype(float)
    mw = np.maximum(df["mu_win"].to_numpy(float), 0.0)
    ml = np.maximum(df["mu_loss"].to_numpy(float), 0.0)
    expected = z * mw - (1.0 - z) * ml
    np.testing.assert_allclose(
        df["oracle_sign_return"].to_numpy(float), expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        df["magnitude_error_component"].to_numpy(float),
        df["oracle_sign_return"].to_numpy(float) - y, rtol=1e-10, atol=1e-10)


# 15. No field named oracle_profit / oracle_strategy_return
def test_no_oracle_profit_field(a0_oof):
    df = C.add_closure_columns(a0_oof)
    assert "oracle_profit" not in df.columns
    assert "oracle_strategy_return" not in df.columns
    assert "oracle_profit" not in C.LEDGER_COLUMNS
    assert "oracle_strategy_return" not in C.LEDGER_COLUMNS


# 16. Gate attribution arithmetic identity
def test_gate_attribution_identity(a0_oof):
    df = C.add_closure_columns(a0_oof)
    g = C.gate_attribution(df)

    y = df["episode_return_atr"].to_numpy(float)
    w = df["sample_weight"].to_numpy(float)
    gate = (df["ev_recomputed"].to_numpy(float) > 0)
    win = y > 0
    captured = float(np.sum(w[gate & win] * y[gate & win]))
    admitted = float(-np.sum(w[gate & ~win] * y[gate & ~win]))
    missed = float(np.sum(w[~gate & win] * y[~gate & win]))
    avoided = float(-np.sum(w[~gate & ~win] * y[~gate & ~win]))

    assert np.isclose(g["captured_profit"], captured, atol=1e-9)
    assert np.isclose(g["admitted_loss"], admitted, atol=1e-9)
    assert np.isclose(g["missed_profit"], missed, atol=1e-9)
    assert np.isclose(g["avoided_loss"], avoided, atol=1e-9)
    assert np.isclose(g["selected_net"], captured - admitted, atol=1e-9)


# 17. EV>0  <->  p > p_break_even (except numerical ties / denom==0)
def test_ev_pos_equiv_break_even(a0_oof):
    df = C.add_closure_columns(a0_oof)
    denom = df["mu_win"].to_numpy(float) + df["mu_loss"].to_numpy(float)
    mask = denom > 0
    ev_pos = df["ev_recomputed"].to_numpy(float)[mask] > 0
    margin_pos = (df["p_win"].to_numpy(float)[mask] - df["p_break_even"].to_numpy(float)[mask]) > 0
    assert np.array_equal(ev_pos, margin_pos)


# 18. Candidate flag joins one-to-one from state
def test_candidate_flag_joins_state(a0_oof):
    merged = C.join_state(a0_oof)
    assert "candidate_at_decision" in merged.columns
    assert "trading_day" in merged.columns
    assert not merged["candidate_at_decision"].isna().any()


# 19. A0/A1 payoff prediction difference is reported
def test_a0_a1_payoff_diff_reported(a0_oof, a1_oof):
    key = ["symbol", "decision_bar", "side", "horizon"]
    a0k = a0_oof.set_index(key)
    a1k = a1_oof.set_index(key)
    diff = {
        "max_abs_mu_win_diff": float((a0k["mu_win"] - a1k["mu_win"]).abs().max()),
        "max_abs_mu_loss_diff": float((a0k["mu_loss"] - a1k["mu_loss"]).abs().max()),
    }
    assert np.isfinite(diff["max_abs_mu_win_diff"])
    assert np.isfinite(diff["max_abs_mu_loss_diff"])
    # A0 and A1 share PAY8 payoff => predictions must be identical
    assert diff["max_abs_mu_win_diff"] <= 1e-9
    assert diff["max_abs_mu_loss_diff"] <= 1e-9


# 20. Output manifest binds all input SHAs + governance counters zero
def test_manifest_binds_input_shas_and_governance():
    audit_like = {
        "a0_vs_a1_payoff_diff": {
            "max_abs_mu_win_diff": 0.0,
            "max_abs_mu_loss_diff": 0.0,
        },
        "closure_identity_validated": True,
        "populations": {ARCH_PRIMARY: {}, ARCH_SECONDARY: {}},
    }
    artifact_shas = {
        "ledger": "x", "deciles": "x", "gate": "x", "forensics": "x",
    }
    code_sha = C.sha256_of(os.path.abspath(C.__file__))
    man = C.build_manifest(audit_like, artifact_shas, code_sha)

    ins = man["input_artifact_shas"]
    assert len(ins["a0_td5_oof_shards"]) == 5
    assert len(ins["a1_td5_oof_shards"]) == 5
    assert "labels_train_v1" in ins and isinstance(ins["labels_train_v1"], str)
    assert "state_v1" in ins and isinstance(ins["state_v1"], str)
    assert "phase4_manifest" in ins and isinstance(ins["phase4_manifest"], str)

    gov = man["governance"]
    assert gov["model_fits"] == 0
    assert gov["direction_model_fits"] == 0
    assert gov["dev_val_reads"] == 0
    assert gov["old_test_label_reads"] == 0
    assert gov["old_test_policy_reads"] == 0
    assert gov["E9_train_oof_axis_reads"] == 0
    assert gov["hyperparameter_searches"] == 0
    assert gov["feature_changes"] == 0

    assert man["direction_layer"]["audited_in_r13_5"] is False
    assert man["direction_layer"]["reason"] == "NO_FROZEN_CAUSAL_TRAIN_OOF_E9_ROOT_AXIS"
    assert man["phase4_status"] == "NO_V2_MODEL_IMPROVEMENT"
    assert man["reviewed_base_sha"] == C.REVIEWED_BASE_SHA


# Bootstrap structural check (small n_boot for speed; plan §32)
def test_block_bootstrap_runs(a0_oof):
    df = C.add_closure_columns(C.join_state(a0_oof))
    df = df[df["candidate_at_decision"] == True].copy()  # noqa: E712
    ci = C.bootstrap_contrasts(df, n_boot=30, seed=12345)
    for k in ["p_top20_minus_bottom20", "mu_win_top20_minus_bottom20",
              "mu_loss_top20_minus_bottom20", "ev_top20_minus_bottom20",
              "ev_positive_weighted_return"]:
        assert k in ci
        assert np.isfinite(ci[k]["point"])
        assert np.isfinite(ci[k]["ci_lo"])
        assert np.isfinite(ci[k]["ci_hi"])
