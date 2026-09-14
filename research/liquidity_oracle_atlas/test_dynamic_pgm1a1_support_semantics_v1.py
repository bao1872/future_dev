"""Support-semantics tests for DYNAMIC-PGM-1A.1 (exact zero-truncated Poisson +
per-node support-correct transition kernel).

Covers:
  * node specification (7 continuous nodes + 2 count + 1 discrete)
  * exact zero-truncated Poisson rate differs from positive-mean Poisson
  * truncated PMF over y>=1 sums to 1
  * Hurdle NLL decomposes into Logistic P(>0) + ZTP NLL on positives
  * count reconstruction invariant (cur + increment == next)
  * FULL reconstruction closure: F(S_t, Z_{t+1}) == S_{t+1} (synthetic)
  * sufficient-stat Ridge == sklearn Ridge (parity)
  * shared-transform K1 prefix == standalone K1 (parity)
  * ZTP analytic gradient == finite-difference + n_eta_clipped == 0
  * age-zero audit harness logic (real formula deferred to when data present)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a1_support_semantics_v1 as m


# --------------------------------------------------------------------------- #
def test_block_definition():
    kinds = {name: kind for name, kind, _, _, _ in m.NODE_SPECS}
    assert len(m.NODE_SPECS) == 7
    assert kinds["z_d_up"] == "gaussian_delta"
    assert kinds["z_dmfe"] == "hurdle_ln_delta"
    assert kinds["z_dmae"] == "hurdle_ln_delta"
    assert kinds["z_range"] == "ln_value"
    assert kinds["z_uresid"] == "hurdle_ln_value"
    assert kinds["z_lresid"] == "hurdle_ln_value"
    assert kinds["z_dcr"] == "zero_interior_one"
    assert len(m.COUNT_Z) == 2
    assert m.DISC_Z == "z_agezero_code"
    # every z column is referenced exactly once across nodes
    seen = [c for _, cols in m.Z_LAYOUT for c in cols]
    assert seen == m.ALL_Z_COLS


def test_ztp_rate_differs_from_positive_mean():
    y = np.array([1, 1, 1, 1, 2], dtype=np.int64)
    pos_mean = float(np.mean(y))           # 1.2  (plain Poisson MLE on positives)
    ztp_lam = m._fit_constant_ztp_rate(y)   # exact ZTP constant rate
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
    y = np.array([2.0])
    nll_at = lambda lam: float(m._ztnp_nll(y, np.array([lam]))[0])
    lo, mle, hi = nll_at(0.1), nll_at(1.6), nll_at(5.0)
    assert np.isfinite(lo) and np.isfinite(mle) and np.isfinite(hi)
    assert mle < lo and mle < hi


def test_hurdle_nll_decomposes():
    y = np.array([0.0, 1.0, 2.0])
    p0 = np.array([0.3, 0.3, 0.3])
    rate = np.array([0.5, 0.5, 0.5])
    nll = m._hurdle_nll(y, p0, rate)
    assert abs(nll[0] - (-np.log1p(-0.3))) < 1e-12
    expected_y1 = -np.log(0.3) + m._ztnp_nll(np.array([1.0]), np.array([0.5]))[0]
    assert abs(nll[1] - expected_y1) < 1e-12
    expected_y2 = -np.log(0.3) + m._ztnp_nll(np.array([2.0]), np.array([0.5]))[0]
    assert abs(nll[2] - expected_y2) < 1e-12


def test_build_transition_sample_count_invariant(monkeypatch):
    monkeypatch.setattr(m, "EXPECTED_TRANSITIONS", 1)
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


def test_full_reconstruction_closure(monkeypatch):
    """F(S_t, Z_{t+1}) == S_{t+1} exactly (synthetic, invariants obeyed)."""
    monkeypatch.setattr(m, "EXPECTED_TRANSITIONS", 2)
    price = np.array([10.0, 10.5, 11.2])
    mfe = np.array([1.0, 1.2, 1.5])
    mae = np.array([0.5, 0.6, 0.7])
    tv = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(price)))])
    dcr = np.array([0.1, 0.2, 0.3])
    rng = np.array([2.0, 2.1, 2.2])
    ures = np.array([0.1, 0.2, 0.3])
    lres = np.array([0.3, 0.4, 0.5])
    ucount = np.array([5.0, 5.0, 5.0])
    lcount = np.array([3.0, 3.0, 3.0])
    uaz = np.array([0, 1, 0])
    laz = np.array([1, 0, 1])
    n = 3
    df = pd.DataFrame({
        "episode_id": [0] * n, "bar_t": [0, 1, 2], "hazard": [0] * n,
        "cur_width_R": [1.0] * n,
        "cur_up_distance_R": price,
        "path_max_up_excursion_R": mfe,
        "path_max_down_excursion_R": mae,
        "path_direction_change_rate": dcr,
        "path_current_bar_range_R": rng,
        "upper_newest_log_age_residual": ures,
        "lower_newest_log_age_residual": lres,
        "upper_active_identity_count_delta": ucount,
        "lower_active_identity_count_delta": lcount,
        "upper_current_newest_age_zero": uaz,
        "lower_current_newest_age_zero": laz,
        "path_total_variation_R": tv,
        "path_last_return_R": np.array([0.0, -0.5, -0.7]),
    })
    cur, nxt = m.build_transition_sample(df)
    assert len(cur) == 2
    recon = m.reconstruct_next_state(cur, cur)
    state_cols = [
        "cur_up_distance_R", "path_max_up_excursion_R", "path_max_down_excursion_R",
        "path_direction_change_rate", "path_current_bar_range_R",
        "upper_newest_log_age_residual", "lower_newest_log_age_residual",
        "upper_active_identity_count_delta", "lower_active_identity_count_delta",
    ]
    max_err = max(np.abs(recon[c] - nxt[c].to_numpy(float)).max() for c in state_cols)
    assert max_err < 1e-8, f"reconstruction max_err={max_err}"


def test_ridge_parity():
    rng = np.random.default_rng(0)
    n, p = 200, 10
    X = rng.normal(size=(n, p))
    Btrue = rng.normal(size=(p, 1))
    Z = X @ Btrue + rng.normal(scale=0.1, size=(n, 1))
    head = m.GaussianTransitionHead(alpha=1.0).fit(X, Z)
    # head.mean is (n, 1); sklearn predict is (n,) -> ravel to compare element-wise
    # (do NOT rely on broadcasting, which would form an (n, n) difference matrix).
    pred_g = np.asarray(head.mean).ravel()
    rg = Ridge(alpha=1.0, fit_intercept=True, tol=1e-12).fit(X, Z)
    pred_sk = rg.predict(X)
    assert pred_g.shape == pred_sk.shape
    assert np.max(np.abs(pred_g - pred_sk)) < 1e-9


def test_shared_transform_parity():
    rng = np.random.default_rng(0)
    num = list(m.OBS_STATE_NUM)
    cat = list(m.OBS_STATE_CAT)
    lag = list(m.LAG_BASE)
    n = 60
    data = {}
    for c in num:
        data[c] = rng.normal(size=n)
    for c in cat:
        data[c] = rng.integers(0, 3, size=n).astype(str)
    for c in lag:
        data[f"lag1_{c}"] = rng.normal(size=n)
    data["lag1_available"] = np.ones(n)
    df = pd.DataFrame(data)
    lag_cols = [f"lag1_{c}" for c in lag] + ["lag1_available"]

    ct_k2 = ColumnTransformer([
        ("cat", Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num + lag_cols),
    ])
    X2 = np.asarray(ct_k2.fit_transform(df))
    k1_dim = X2.shape[1] - len(lag_cols)
    X2_pre = X2[:, :k1_dim]

    ct_k1 = ColumnTransformer([
        ("cat", Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
    ])
    X1 = np.asarray(ct_k1.fit_transform(df))
    assert np.max(np.abs(X2_pre - X1)) < 1e-9


def test_ztp_finite_diff_gradient():
    rng = np.random.default_rng(1)
    y = np.array([1, 1, 1, 2, 3], dtype=np.int64)
    X = np.column_stack([np.ones(5), np.linspace(-1, 1, 5)])
    reg = m.ZeroTruncatedPoissonRegressor(alpha=0.0).fit(X, y)
    for _ in range(3):
        theta = rng.normal(size=X.shape[1] + 1)
        eps = 1e-6
        g_ana = reg.gradient(theta)
        g_fd = np.zeros_like(theta)
        for i in range(len(theta)):
            tp = theta.copy(); tp[i] += eps
            tm = theta.copy(); tm[i] -= eps
            g_fd[i] = (reg.objective(tp) - reg.objective(tm)) / (2 * eps)
        assert np.max(np.abs(g_ana - g_fd)) < 1e-5, \
            f"FD mismatch ana={g_ana} fd={g_fd}"
    assert reg.n_eta_clipped == 0


def test_agezero_audit_harness():
    codes = np.array([0, 1, 2, 3, 1])
    df = pd.DataFrame({"z_agezero_code": codes})
    # identity reconstruction -> exact (validates harness only)
    res = m.audit_agezero_reconstruction(df, df, reconstruct_fn=None)
    assert res["exact"] is True and res["n_mismatch"] == 0
    # wrong reconstruction -> detected mismatch
    def wrong(cur, z):
        return (z["z_agezero_code"].to_numpy(int) + 1) % 4
    res2 = m.audit_agezero_reconstruction(df, df, reconstruct_fn=wrong)
    assert res2["exact"] is False and res2["n_mismatch"] == 5


if __name__ == "__main__":
    test_block_definition()
    test_ztp_rate_differs_from_positive_mean()
    test_ztp_pmf_sums_to_one()
    test_ztp_nll_finite_and_min_near_mle()
    test_hurdle_nll_decomposes()
    print("test_dynamic_pgm1a1_support_semantics_v1 (static parts) OK; "
          "run pytest for monkeypatch-based tests.")
