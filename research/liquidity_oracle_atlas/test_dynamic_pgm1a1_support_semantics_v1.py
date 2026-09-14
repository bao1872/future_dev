"""Support-semantics tests for DYNAMIC-PGM-1A.1 (exact zero-truncated Poisson).

Covers the math the 1A.1 count node depends on:
  * block definition (7 continuous + 2 count increments)
  * exact zero-truncated Poisson rate differs from the positive-only Poisson mean
  * truncated PMF over y>=1 sums to 1
  * Hurdle NLL decomposes into Logistic P(>0) + ZTP NLL on positives
  * the cumulative-activation-counter reconstruction invariant holds per row
"""
import sys
from pathlib import Path

import numpy as np
from scipy.special import gammaln

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a1_support_semantics_v1 as m


def test_block_definition():
    assert len(m.CONT_Z) == 7
    assert len(m.COUNT_Z) == 2
    assert "z_delta_upper_count" in m.COUNT_Z
    assert "z_delta_lower_count" in m.COUNT_Z


def test_ztp_rate_differs_from_positive_mean():
    y = np.array([1, 1, 1, 1, 2], dtype=np.int64)
    pos_mean = float(np.mean(y))           # 1.2  (plain Poisson MLE on positives)
    ztp_lam = m._fit_constant_ztp_rate(y)   # exact ZTP constant rate
    # ZTP lam is far smaller than the positive-only mean (different probability family)
    assert abs(pos_mean - ztp_lam) > 0.4


def test_ztp_pmf_sums_to_one():
    y = np.array([1, 1, 1, 1, 2], dtype=np.int64)
    X = np.ones((len(y), 1), dtype=np.float64)
    reg = m.ZeroTruncatedPoissonRegressor(alpha=1e-6).fit(X, y)
    lam = float(reg.predict_rate(np.ones((1, 1)))[0])
    M = 30
    ys = np.arange(1, M + 1, dtype=np.float64)
    log_pmf = (-lam + ys * np.log(lam) - gammaln(ys + 1)
               - np.log(-np.expm1(-lam)))
    pmf = np.exp(log_pmf)
    assert abs(float(pmf.sum()) - 1.0) < 1e-8


def test_ztp_nll_finite_and_min_near_mle():
    # For fixed y=2 the ZTP NLL is convex in lam, minimized at the MLE
    # (ZTP E[Y|Y>0]=lam/(1-e^-lam)=2 -> lam~1.6).
    y = np.array([2.0])
    nll_at = lambda lam: float(m._ztnp_nll(y, np.array([lam]))[0])
    lo, mle, hi = nll_at(0.1), nll_at(1.6), nll_at(5.0)
    assert np.isfinite(lo) and np.isfinite(mle) and np.isfinite(hi)
    assert mle < lo and mle < hi   # convex, minimum at the MLE


def test_hurdle_nll_decomposes():
    y = np.array([0.0, 1.0, 2.0])
    p0 = np.array([0.3, 0.3, 0.3])
    rate = np.array([0.5, 0.5, 0.5])
    nll = m._hurdle_nll(y, p0, rate)
    assert abs(nll[0] - (-np.log1p(-0.3))) < 1e-12
    expected_y1 = -np.log(0.3) + m._ztnp_nll(np.array([1.0]), np.array([0.5]))
    assert abs(nll[1] - expected_y1) < 1e-12
    expected_y2 = -np.log(0.3) + m._ztnp_nll(np.array([2.0]), np.array([0.5]))
    assert abs(nll[2] - expected_y2) < 1e-12


def test_build_transition_sample_count_invariant(monkeypatch):
    # exercise the cumulative-activation-counter increment guards on a tiny sample
    monkeypatch.setattr(m, "EXPECTED_TRANSITIONS", 1)
    import pandas as pd
    df = pd.DataFrame({
        "episode_id": [0, 0],
        "bar_t": [0, 1],
        "hazard": [0, 0],
        "cur_width_R": [1.0, 1.0],
        "cur_up_distance_R": [10.0, 10.5],
        "path_max_up_excursion_R": [1.0, 1.2],
        "path_max_down_excursion_R": [0.5, 0.6],
        "path_direction_change_rate": [0.1, 0.2],
        "path_current_bar_range_R": [2.0, 2.1],
        "upper_newest_log_age_residual": [0.1, 0.2],
        "lower_newest_log_age_residual": [0.3, 0.4],
        "upper_active_identity_count_delta": [5.0, 5.0],
        "lower_active_identity_count_delta": [3.0, 3.0],
        "upper_current_newest_age_zero": [0, 1],
        "lower_current_newest_age_zero": [1, 0],
        "path_total_variation_R": [1.0, 1.5],
        "path_last_return_R": [0.0, -0.5],
    })
    cur, nxt = m.build_transition_sample(df)
    assert len(cur) == 1
    for side in ("upper", "lower"):
        inc = cur[f"z_delta_{side}_count"].to_numpy(float)
        curc = cur[f"{side}_active_identity_count_delta"].to_numpy(float)
        nxtc = nxt[f"{side}_active_identity_count_delta"].to_numpy(float)
        assert np.allclose(curc + inc, nxtc)


if __name__ == "__main__":
    test_block_definition()
    test_ztp_rate_differs_from_positive_mean()
    test_ztp_pmf_sums_to_one()
    test_ztp_nll_finite_and_min_near_mle()
    test_hurdle_nll_decomposes()
    # monkeypatch is a pytest fixture; emulate minimally for the standalone runner
    class _M:
        def setattr(self, mod, name, val):
            setattr(mod, name, val)
    test_build_transition_sample_count_invariant(_M())
    print("test_dynamic_pgm1a1_support_semantics_v1: ALL OK")
