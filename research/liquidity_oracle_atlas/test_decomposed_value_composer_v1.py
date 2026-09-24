"""§58 Composer tests — deterministic arithmetic, no materialized data."""

import numpy as np
import pytest

from research.liquidity_oracle_atlas import decomposed_value_composer_v1 as C


def test_compose_probability_and_payoff_basic():
    p = np.array([0.6, 0.3, 0.8])
    mw = np.array([1.0, 0.5, 2.0])
    ml = np.array([0.5, 1.0, 1.0])
    r = C.compose_probability_and_payoff(p, mw, ml)
    assert np.allclose(r["mu_win"], mw)
    assert np.allclose(r["mu_loss"], ml)
    assert np.allclose(r["predicted_rr"], mw / ml)
    assert np.allclose(r["p_break_even"], ml / (mw + ml))
    assert np.allclose(
        r["ev_combined"], p * mw - (1 - p) * ml)
    assert r["take_combined"].tolist() == [True, False, True]


def test_ev_c_identity_p_gt_pstar():
    p = np.array([0.4, 0.7])
    mw = np.array([1.0, 1.0])
    ml = np.array([1.0, 1.0])  # p* = 0.5
    r = C.compose_probability_and_payoff(p, mw, ml)
    # p=0.4 < 0.5 -> EV <= 0 ; p=0.7 > 0.5 -> EV > 0
    assert r["ev_combined"][0] < 0
    assert r["ev_combined"][1] > 0
    assert r["ev_combined"][0] <= 0 and r["ev_combined"][1] > 0


def test_compose_scores_ev_w_ev_r_ev_c():
    # vary Win, fix Payoff at TRAIN priors
    p = np.array([0.2, 0.9])
    mw = np.array([1.0, 1.0])
    ml = np.array([1.0, 1.0])
    ev_w, ev_r, ev_c = C.compose_scores(
        p, mw, ml, p_train_prior=0.5, mu_w_train_prior=1.0,
        mu_l_train_prior=1.0)
    # EV_W uses priors muW0=1, muL0=1 -> ev_w = p*1-(1-p)*1 = 2p-1
    assert np.allclose(ev_w, 2 * p - 1)
    # EV_R uses p0=0.5 -> ev_r = 0.5*1-(0.5)*1 = 0 regardless of p
    assert np.allclose(ev_r, 0.0)
    assert np.allclose(ev_c, p * mw - (1 - p) * ml)


def test_compose_scores_ev_r_uses_prior_only():
    # With p_train_prior fixed, EV_R must be invariant to p_hat.
    phat = np.array([0.1, 0.99])
    mw = np.array([3.0, 0.5])
    ml = np.array([1.0, 2.0])
    _, ev_r_a, _ = C.compose_scores(phat, mw, ml, p_train_prior=0.5,
                                     mu_w_train_prior=1.0, mu_l_train_prior=1.0)
    _, ev_r_b, _ = C.compose_scores(phat[::-1], mw, ml, p_train_prior=0.5,
                                    mu_w_train_prior=1.0, mu_l_train_prior=1.0)
    assert np.allclose(ev_r_a, ev_r_b)


def test_composer_has_no_model_fit():
    assert C.COUNTERS["model_fits"] == 0


def test_train_priors_are_train_only():
    import os
    path = os.path.join("artifacts", "decomposed_value_v1",
                        "labels_train_v1.parquet")
    if not os.path.exists(path):
        pytest.skip("decomposed labels not materialized yet")
    priors = C.compute_train_priors()
    for H in ["td1", "td3", "td5"]:
        pr = priors[H]
        assert 0.0 < pr["p0"] < 1.0 or np.isnan(pr["p0"])
        # RR0 = muW0 / muL0
        if pr["muL0"] and pr["muL0"] > 0:
            assert np.isclose(pr["RR0"], pr["muW0"] / pr["muL0"])
