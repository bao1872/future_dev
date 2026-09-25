"""Tests for the R13.5B Composer Isolation Audit (plan R13.5B).

No DEV VAL, no old TEST, no model fit, no Direction fit.
"""
from __future__ import annotations

import os
import hashlib

import numpy as np
import pandas as pd
import pytest

from research.liquidity_oracle_atlas import run_decomposed_v2_research as R
from research.liquidity_oracle_atlas import (
    decomposed_value_composer_isolation_audit_v1 as B,
)
from research.liquidity_oracle_atlas import (
    decomposed_value_closure_audit_v1 as C,
)

ARCH_PRIMARY = B.ARCH_PRIMARY
ARCH_SECONDARY = B.ARCH_SECONDARY
POP_ALL = B.POP_ALL
POP_ROOT = B.POP_ROOT


@pytest.fixture(scope="module")
def a0_oof():
    return C.load_td5_oof(ARCH_PRIMARY)


@pytest.fixture(scope="module")
def a1_oof():
    return C.load_td5_oof(ARCH_SECONDARY)


# 1. TD5 only
def test_td5_only(a0_oof):
    assert set(a0_oof["horizon"].unique()) == {B.HORIZON}


# 2. Exactly five OOF folds
def test_exactly_five_folds(a0_oof):
    assert set(a0_oof["fold"].unique()) == {0, 1, 2, 3, 4}


# 3. A0/A1 key universes identical
def test_arch_key_universes_identical(a0_oof, a1_oof):
    key = ["symbol", "decision_bar", "side", "horizon"]
    assert set(map(tuple, a0_oof[key].to_numpy())) == set(map(tuple, a1_oof[key].to_numpy()))


# 4. OOF Y matches TRAIN labels exactly
def test_oof_labels_exact_match(a0_oof):
    merged = C.verify_oof_labels(a0_oof)
    dy = (merged["episode_return_atr_oof"] - merged["episode_return_atr_label"]).to_numpy(float)
    dw = (merged["sample_weight_oof"] - merged["sample_weight_label"]).to_numpy(float)
    assert np.max(np.abs(dy)) <= 1e-12
    assert np.max(np.abs(dw)) <= 1e-12


# 5. No DEV VAL / old TEST / model fit / Direction fit
def test_no_val_test_or_fit_in_source():
    src = open(os.path.abspath(B.__file__)).read()
    assert "read_val_labels" not in src
    assert "read_test_labels" not in src
    assert "labels_test" not in src
    assert "labels_val" not in src
    assert "decomposed_models_v2" not in src
    assert ".fit(" not in src


# 6. EV_C == (mu_W+mu_L)(p-p_BE) exact identity
def test_evc_identity(a0_oof):
    full = C.add_closure_columns(a0_oof)
    W0, L0, p0 = B._train_priors(full)
    df = B.add_isolation_scores(full, W0, L0, p0)
    W = np.maximum(df["mu_win"].to_numpy(float), 0.0)
    L = np.maximum(df["mu_loss"].to_numpy(float), 0.0)
    p = np.clip(df["p_win"].to_numpy(float), 0.0, 1.0)
    denom = W + L
    pbe = np.where(denom > 0, L / denom, np.nan)
    expected = (W + L) * (p - pbe)
    np.testing.assert_allclose(
        df["score_EV_C"].to_numpy(float), expected, rtol=1e-10, atol=1e-10)


# 7. EV_C equals reconstructed EV (ev_recomputed)
def test_evc_equals_ev_recomputed(a0_oof):
    full = C.add_closure_columns(a0_oof)
    W0, L0, p0 = B._train_priors(full)
    df = B.add_isolation_scores(full, W0, L0, p0)
    np.testing.assert_allclose(
        df["score_EV_C"].to_numpy(float),
        df["ev_recomputed"].to_numpy(float), rtol=1e-10, atol=1e-10)


# 8. p_BE uses L/(W+L)
def test_pbe_definition(a0_oof):
    full = C.add_closure_columns(a0_oof)
    W0, L0, p0 = B._train_priors(full)
    df = B.add_isolation_scores(full, W0, L0, p0)
    W = np.maximum(df["mu_win"].to_numpy(float), 0.0)
    L = np.maximum(df["mu_loss"].to_numpy(float), 0.0)
    denom = W + L
    pbe = np.where(denom > 0, L / denom, np.nan)
    np.testing.assert_allclose(
        df["score_p_minus_pBE"].to_numpy(float),
        df["p_win"].to_numpy(float) - pbe, rtol=1e-10, atol=1e-10)


# 9. isolation dict has exactly the 5 scores; bootstrap finite for ROOT
def test_isolation_scores_present(a0_oof):
    full = C.add_closure_columns(a0_oof)
    full = C.join_state(full)
    W0, L0, p0 = B._train_priors(full)
    df = full[full["candidate_at_decision"] == True].copy()  # noqa: E712
    df = B.add_isolation_scores(df, W0, L0, p0)
    boot = B.composer_isolation_bootstrap(df, n_boot=30, seed=1)
    assert set(boot.keys()) == set(B.SCORE_NAMES)
    for v in boot.values():
        assert np.isfinite(v["point"])
        assert np.isfinite(v["ci_lo"])
        assert np.isfinite(v["ci_hi"])
        assert v["n_complete_blocks"] is not None


# 10. EV_W is a monotonic (positive-slope) function of p (fixed prior scale)
def test_evw_linear_in_p(a0_oof):
    full = C.add_closure_columns(a0_oof)
    W0, L0, p0 = B._train_priors(full)
    df = B.add_isolation_scores(full, W0, L0, p0)
    p = np.clip(df["p_win"].to_numpy(float), 0.0, 1.0)
    evw = df["score_EV_W"].to_numpy(float)
    # EV_W = p*(W0+L0) - L0  => slope (W0+L0) > 0, so ranking == ranking of p
    np.testing.assert_allclose(evw, p * (W0 + L0) - L0, rtol=1e-10, atol=1e-10)


# 11. No oracle_profit / oracle_strategy_return field anywhere
def test_no_oracle_profit_field(a0_oof):
    full = C.add_closure_columns(a0_oof)
    W0, L0, p0 = B._train_priors(full)
    df = B.add_isolation_scores(full, W0, L0, p0)
    for bad in ["oracle_profit", "oracle_strategy_return"]:
        assert bad not in df.columns
        assert bad not in B.SCORE_NAMES


# 12. build_audit produces 2 arch x 2 population summaries with ROOT bootstrap
def test_build_audit_structure(monkeypatch):
    monkeypatch.setattr(B, "BOOTSTRAP_B", 40)  # speed up the test only
    audit, lp, detail = B.build_audit()
    assert set(audit["populations"].keys()) == {ARCH_PRIMARY, ARCH_SECONDARY}
    for arch in [ARCH_PRIMARY, ARCH_SECONDARY]:
        for pop in [POP_ALL, POP_ROOT]:
            seg = audit["populations"][arch][pop]
            assert set(seg["point"].keys()) == set(B.SCORE_NAMES)
            if pop == POP_ROOT:
                assert seg["bootstrap"] is not None
                assert seg["bootstrap"]["p"]["ci_excludes_zero"] is True
    assert len(detail) == 2 * 2 * len(B.SCORE_NAMES)


# 13. Manifest binds input SHAs + governance counters zero + direction not audited
def test_manifest_binds_and_governance(monkeypatch):
    monkeypatch.setattr(B, "BOOTSTRAP_B", 40)  # speed up the test only
    audit, lp, detail = B.build_audit()
    artifact_shas = B.write_artifacts(audit, lp, detail)
    code_sha = B.sha256_of(os.path.abspath(B.__file__))
    man = B.build_manifest(audit, artifact_shas, code_sha)

    ins = man["input_artifact_shas"]
    assert len(ins["a0_td5_oof_shards"]) == 5
    assert len(ins["a1_td5_oof_shards"]) == 5
    assert isinstance(ins["labels_train_v1"], str)
    assert isinstance(ins["state_v1"], str)
    assert isinstance(ins["phase4_manifest"], str)
    assert man["parent_r135_sha"] == B.PARENT_R135_SHA
    assert man["phase4_status"] == "NO_V2_MODEL_IMPROVEMENT"

    gov = man["governance"]
    assert gov["model_fits"] == 0
    assert gov["direction_model_fits"] == 0
    assert gov["dev_val_reads"] == 0
    assert gov["old_test_label_reads"] == 0
    assert gov["old_test_policy_reads"] == 0
    assert gov["E9_train_oof_axis_reads"] == 0
    assert gov["hyperparameter_searches"] == 0
    assert gov["feature_changes"] == 0

    assert man["direction_layer"]["audited_in_r13_5b"] is False
    assert man["direction_layer"]["reason"] == "NO_FROZEN_CAUSAL_TRAIN_OOF_E9_ROOT_AXIS"


# 14. Complete-block bootstrap drops the terminal remainder
def test_complete_block_drops_remainder():
    # 23 unique days -> floor(23/5)=4 complete blocks (20 days), 3 dropped
    td = np.array([f"2026-01-{d:02d}" for d in list(range(1, 24)) * 3])
    idx = B._complete_block_indices(td, 5)
    total = sum(len(b) for b in idx)
    assert total == 20 * 3  # only complete blocks retained
    assert len(idx) == 4
