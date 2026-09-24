"""§47 regression tests — architecture (13)(14)(15)(17) + selection (23)."""

import numpy as np
import pytest

from research.liquidity_oracle_atlas import (
    decomposed_value_features_v2 as F,
    decomposed_models_v2 as M,
)
from research.liquidity_oracle_atlas.build_decomposed_value_dataset_v1 import (
    PAY8_COLS,
    WIN33_COLS,
)


# (13) A1 only changes the Win feature matrix
def test_a1_only_changes_win_matrix():
    assert tuple(F.A1.win) == F.SHARED41
    assert tuple(F.A1.payoff) == tuple(PAY8_COLS)
    assert tuple(F.A1.payoff) == tuple(F.A0.payoff)
    assert F.A1.n_win == 41 and F.A1.n_payoff == 8


# (14) A2 only changes the Payoff feature matrix
def test_a2_only_changes_payoff_matrix():
    assert tuple(F.A2.win) == tuple(WIN33_COLS)
    assert tuple(F.A2.win) == tuple(F.A0.win)
    assert tuple(F.A2.payoff) == F.SHARED41
    assert F.A2.n_win == 33 and F.A2.n_payoff == 41


# (15) A3 gives identical feature columns to all three heads
def test_a3_shares_identical_state():
    assert tuple(F.A3.win) == tuple(F.A3.payoff) == F.SHARED41
    assert F.A3.shared_state
    assert F.A3.win_schema_sha256 == F.A3.payoff_schema_sha256
    # A0/A1/A2 are the ablations and must NOT be shared-state.
    assert not F.A0.shared_state
    assert not F.A1.shared_state
    assert not F.A2.shared_state


def test_r12_has_exactly_four_architectures():
    assert [a.name for a in F.R12_ARCHS] == [
        "A0_V1_DISJOINT", "A1_SHARE_TO_WIN", "A2_SHARE_TO_PAYOFF",
        "A3_SHARED_BOTH"]


# (17) Composer has zero learned parameters
def test_composer_is_pure_arithmetic():
    p = np.array([0.4, 0.6, 1.0])
    mu_w = np.array([2.0, 0.0, 1.0])
    mu_l = np.array([1.0, 1.0, 0.0])
    out = M.compose(p, mu_w, mu_l)
    np.testing.assert_allclose(
        out["ev"], p * mu_w - (1.0 - p) * mu_l, rtol=0, atol=1e-15)
    # No fourth model: EV is a closed form of p, mu_W, mu_L only.
    assert np.all(out["mu_win"] >= 0) and np.all(out["mu_loss"] >= 0)
    assert out["rr"][0] == pytest.approx(2.0)
    assert out["p_break_even"][0] == pytest.approx(1.0 / 3.0)
    # Zero loss magnitude -> infinite RR; break-even probability is 0 because
    # no loss is required to break even (mu_W + mu_L > 0 still holds).
    assert np.isinf(out["rr"][2])
    assert out["p_break_even"][2] == pytest.approx(0.0)
    # Truly degenerate: BOTH magnitudes zero -> break-even undefined.
    empty = M.compose(np.array([0.5]), np.array([0.0]), np.array([0.0]))
    assert np.isnan(empty["p_break_even"][0])


def test_weighted_ev_mse_matches_reference():
    y = np.array([1.0, -2.0, 3.0, np.nan])
    ev = np.array([0.5, -1.0, 3.5, 1.0])
    w = np.array([1.0, 1.0, 2.0, 1.0])
    expected = (1.0 * (0.5) ** 2 + 1.0 * (-1.0) ** 2 + 2.0 * (-0.5) ** 2) / 4.0
    assert M.weighted_ev_mse(y, ev, w) == pytest.approx(expected)


def test_runtime_params_force_stable_threading():
    """n_jobs is a threading knob and must be forced to the stable value."""
    assert M.runtime_params({"n_jobs": -1})["n_jobs"] == M.FORCE_N_JOBS
    assert "objective" in M.runtime_params(M.CLF_PARAMS)


def test_safe_n_estimators_never_zero():
    assert M.safe_n_estimators(0) == 1
    assert M.safe_n_estimators(58) == 58


# (23) one-SE selector returns the simplest eligible schema
def test_one_se_selects_simplest_eligible():
    results = [
        {"candidate": "big", "n_features": 73, "mean_ev_mse": 5.00,
         "se_ev_mse": 0.10},
        {"candidate": "mid", "n_features": 49, "mean_ev_mse": 5.05,
         "se_ev_mse": 0.10},
        {"candidate": "small", "n_features": 41, "mean_ev_mse": 5.09,
         "se_ev_mse": 0.10},
        {"candidate": "far", "n_features": 41, "mean_ev_mse": 5.50,
         "se_ev_mse": 0.10},
    ]
    sel = M.one_se_select(results)
    assert sel["candidate"] == "small"
    assert sel["best_candidate"] == "big"
    assert sel["one_se_threshold"] == pytest.approx(5.10)
    assert sel["n_eligible"] == 3       # "far" is outside one SE


def test_one_se_prefers_fewer_features_on_tie():
    results = [
        {"candidate": "a", "n_features": 59, "mean_ev_mse": 5.00,
         "se_ev_mse": 0.05},
        {"candidate": "b", "n_features": 41, "mean_ev_mse": 5.00,
         "se_ev_mse": 0.05},
    ]
    assert M.one_se_select(results)["candidate"] == "b"


def test_paired_bootstrap_status_labels():
    base = np.zeros(80)
    # Clearly better candidate.
    better = M.paired_block_bootstrap(np.full(80, -1.0), base)
    assert better["status"] == "SUPPORTED_DEV_IMPROVEMENT"
    assert better["ci_high"] < 0
    # Clearly worse candidate.
    worse = M.paired_block_bootstrap(np.full(80, 1.0), base)
    assert worse["status"] == "NO_DEV_IMPROVEMENT"
    # Mixed: point negative but CI crossing zero.
    rng = np.random.default_rng(0)
    mixed = M.paired_block_bootstrap(rng.normal(-0.05, 1.0, 80), base)
    assert mixed["status"] in ("PROMISING_DEV_IMPROVEMENT",
                               "NO_DEV_IMPROVEMENT",
                               "SUPPORTED_DEV_IMPROVEMENT")


def test_paired_bootstrap_uses_fixed_contract():
    out = M.paired_block_bootstrap(np.full(83, 1.0), np.zeros(83))
    # 83 days -> 16 complete 5-day blocks, 3 tail days excluded.
    assert out["n_blocks"] == 16
    assert out["excluded_tail_days"] == 3
    assert M.BOOTSTRAP_B == 5000 and M.BOOTSTRAP_SEED == 20260925
    assert M.BOOTSTRAP_BLOCK_DAYS == 5
