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
    assert kinds["z_range"] == "hurdle_ln_value"
    assert kinds["z_uresid"] == "hurdle_ln_neg"
    assert kinds["z_lresid"] == "hurdle_ln_neg"
    assert kinds["z_dcr"] == "zero_interior_one"
    assert len(m.COUNT_Z) == 2
    assert m.DISC_Z == "z_agezero_code"
    # Z_LAYOUT = [(node, contiguous integer offsets into the concatenated Z matrix)];
    # ALL_Z_COLS = the flat z-column names. Verify the layout is a faithful,
    # non-overlapping indexing: offsets cover range(n) and each offset maps to the
    # corresponding node's z-column name.
    offs = [i for _, cols in m.Z_LAYOUT for i in cols]
    assert offs == list(range(len(m.ALL_Z_COLS)))
    names = [m.NODE_ZCOLS[n][k] for n, cols in m.Z_LAYOUT for k in range(len(cols))]
    assert names == m.ALL_Z_COLS


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
        "start_bar": [0, 0],
        "hazard": [0, 0],
        "cur_width_R": [1.0, 1.0],
        "cur_up_distance_R": [10.0, 10.5],
        "cur_down_distance_R": [20.0, 20.0],
        "cur_log_ratio": [-0.693, -0.647],
        "path_max_up_excursion_R": [1.0, 1.2],
        "path_max_down_excursion_R": [0.5, 0.6],
        "path_direction_change_rate": [0.1, 0.2],
        "path_current_bar_range_R": [2.0, 2.1],
        "upper_newest_log_age_residual": [-0.1, -0.2],
        "lower_newest_log_age_residual": [-0.3, -0.4],
        "upper_newest_log_age": [np.log1p(5.0), 0.0],
        "lower_newest_log_age": [0.0, np.log1p(3.0)],
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
    dn_price = np.array([20.0, 19.5, 18.8])
    width = 30.0
    clr = np.log((price * 1.0 + 1e-9) / (dn_price * 1.0 + 1e-9))
    mfe = np.array([1.0, 1.2, 1.5])
    mae = np.array([0.5, 0.6, 0.7])
    tv = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(price)))])
    dcr = np.array([0.1, 0.2, 0.3])
    rng = np.array([2.0, 2.1, 2.2])
    ures = np.array([0.0, -0.1, -0.2])
    lres = np.array([0.0, -0.3, -0.4])
    ucount = np.array([5.0, 5.0, 5.0])
    lcount = np.array([3.0, 3.0, 3.0])
    n = 3
    ulog = np.log1p(np.full(n, 5.0))
    llog = np.zeros(n)
    uaz = np.array([0, 0, 0])
    laz = np.array([1, 0, 0])
    df = pd.DataFrame({
        "episode_id": [0] * n, "bar_t": [0, 1, 2], "start_bar": [0] * n, "hazard": [0] * n,
        "cur_width_R": [width] * n,
        "cur_up_distance_R": price,
        "cur_down_distance_R": dn_price,
        "cur_log_ratio": clr,
        "path_max_up_excursion_R": mfe,
        "path_max_down_excursion_R": mae,
        "path_direction_change_rate": dcr,
        "path_current_bar_range_R": rng,
        "upper_newest_log_age_residual": ures,
        "lower_newest_log_age_residual": lres,
        "upper_newest_log_age": ulog,
        "lower_newest_log_age": llog,
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
        "cur_up_distance_R", "cur_down_distance_R", "cur_width_R", "cur_log_ratio",
        "path_max_up_excursion_R", "path_max_down_excursion_R",
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
    """audit_agezero_reconstruction(cur_df, z_df, nxt_df=None) -> per-side exactness.

    Canonical provenance reconstruction:
        newest_{t+1} = exp(r_{t+1} + log1p(new_0 + elapsed_{t+1})) - 1
        age_zero_{t+1} = 1[newest_{t+1} == 0]
    """
    n = 4
    ulog = np.log1p(np.array([0.0, 5.0, 0.0, 5.0]))
    llog = np.log1p(np.array([5.0, 5.0, 0.0, 0.0]))
    cur = pd.DataFrame({
        "upper_newest_log_age": ulog,
        "lower_newest_log_age": llog,
        "upper_newest_log_age_residual": [0.0] * n,
        "lower_newest_log_age_residual": [0.0] * n,
        "bar_t": np.array([0, 1, 0, 1], dtype=np.int64),
        "start_bar": np.array([0, 0, 0, 0], dtype=np.int64),
    })
    # create matching next state where newest age advances or is reset
    exp_u = np.expm1(ulog) + (cur["bar_t"] + 1 - cur["start_bar"])
    res_u = np.array([0.0, -np.log1p(7.0), -np.log1p(1.0), 0.0])
    age_next_u = np.expm1(res_u + np.log1p(exp_u))
    az_u = np.isclose(age_next_u, 0.0, atol=1e-10).astype(int)

    exp_l = np.expm1(llog) + (cur["bar_t"] + 1 - cur["start_bar"])
    res_l = np.array([-np.log1p(6.0), 0.0, 0.0, -np.log1p(2.0)])
    age_next_l = np.expm1(res_l + np.log1p(exp_l))
    az_l = np.isclose(age_next_l, 0.0, atol=1e-10).astype(int)

    # z_df with the residuals
    z_u = m.build_z_columns("z_uresid", "hurdle_ln_neg", res_u)
    z_l = m.build_z_columns("z_lresid", "hurdle_ln_neg", res_l)
    z_df = pd.DataFrame({**z_u, **z_l})
    nxt = pd.DataFrame({
        "upper_newest_log_age_residual": res_u,
        "lower_newest_log_age_residual": res_l,
        "upper_current_newest_age_zero": az_u,
        "lower_current_newest_age_zero": az_l,
    })
    res = m.audit_agezero_reconstruction(cur, z_df, nxt)
    assert res["upper"]["exact"] is True and res["upper"]["n_mismatch"] == 0
    assert res["lower"]["exact"] is True and res["lower"]["n_mismatch"] == 0
    assert "delta_newest_age" in res["upper"]

    # a mismatched next state -> mismatch is detected
    bad_nxt = nxt.copy()
    bad_nxt["upper_current_newest_age_zero"] = 1 - bad_nxt["upper_current_newest_age_zero"]
    res_bad = m.audit_agezero_reconstruction(cur, z_df, bad_nxt)
    assert res_bad["upper"]["exact"] is False and res_bad["upper"]["n_mismatch"] > 0


def test_child_agezero_configuration():
    # 1. Default initial state has discrete agezero node
    m.configure_child_semantics(agezero_deterministic=False)
    assert m.AGEZERO_DETERMINISTIC is False
    assert m.disc_spec() == ["z_agezero_code"]
    assert "z_agezero_code" in m.BLOCKS["LiquidityComposition"]["disc"]

    # 2. When agezero_deterministic=True, discrete node is eliminated from BLOCKS and disc_spec()
    m.configure_child_semantics(agezero_deterministic=True)
    assert m.AGEZERO_DETERMINISTIC is True
    assert m.disc_spec() == []
    assert m.BLOCKS["LiquidityComposition"]["disc"] == []

    # 3. Restoring agezero_deterministic=False restores the discrete node
    m.configure_child_semantics(agezero_deterministic=False)
    assert m.AGEZERO_DETERMINISTIC is False
    assert m.disc_spec() == ["z_agezero_code"]

    # 4. Test subprocess CLI behavior:
    # 4a. Child with --window-json but WITHOUT --agezero-deterministic exits with hard guard
    import subprocess
    cmd_err = [
        sys.executable,
        str(REPO / "research" / "liquidity_oracle_atlas" / "experiment_dynamic_pgm1a1_support_semantics_v1.py"),
        "--window-json", "{}",
    ]
    r_err = subprocess.run(cmd_err, capture_output=True, text=True)
    assert r_err.returncode != 0
    assert "STOP_DYNAMIC_PGM1A1_CHILD_AGEZERO_STATE_NOT_PROPAGATED" in (r_err.stderr + r_err.stdout)

    # 4b. Child with --window-json AND --agezero-deterministic propagates past the agezero guard
    # (it may fail later on empty json or missing data, but must NOT fail on the agezero guard)
    cmd_with_flag = [
        sys.executable,
        str(REPO / "research" / "liquidity_oracle_atlas" / "experiment_dynamic_pgm1a1_support_semantics_v1.py"),
        "--window-json", "{}",
        "--agezero-deterministic",
    ]
    r_flag = subprocess.run(cmd_with_flag, capture_output=True, text=True)
    assert "STOP_DYNAMIC_PGM1A1_CHILD_AGEZERO_STATE_NOT_PROPAGATED" not in (r_flag.stderr + r_flag.stdout)
    assert "STOP_DYNAMIC_PGM1A1_CHILD_DISC_SPEC_NONEMPTY" not in (r_flag.stderr + r_flag.stdout)


def test_gaussian_head_n_params():
    rng = np.random.default_rng(42)
    n, p, q = 50, 4, 3
    X = rng.normal(size=(n, p))
    Z = rng.normal(size=(n, q))
    head = m.GaussianTransitionHead(alpha=1.0)
    head.fit(X, Z)
    assert hasattr(head, "n_params")
    assert head.n_params == p * q + q


def test_fit_nodes_has_parameter_count():
    rng = np.random.default_rng(123)
    n_tr, n_ev, p = 100, 30, 5
    Xtr = rng.normal(size=(n_tr, p))
    Xev = rng.normal(size=(n_ev, p))

    def make_z(n):
        d = {}
        d["z_d_up"] = rng.normal(size=n)
        for k in ["z_dmfe", "z_dmae", "z_range", "z_uresid", "z_lresid"]:
            ispos = (rng.uniform(size=n) > 0.3).astype(float)
            logv = np.where(ispos > 0.5, rng.normal(size=n), 0.0)
            d[f"{k}_ispos"] = ispos
            d[f"{k}_log"] = logv
        u = rng.uniform(size=n)
        is0 = (u < 0.1).astype(float)
        is1 = (u > 0.9).astype(float)
        interior = (is0 < 0.5) & (is1 < 0.5)
        logit = np.where(interior, rng.normal(size=n), 0.0)
        d["z_dcr_is0"] = is0
        d["z_dcr_is1"] = is1
        d["z_dcr_logit"] = logit
        df = pd.DataFrame(d)[m.ALL_Z_COLS]
        return df.to_numpy(np.float64)

    Zc_tr = make_z(n_tr)
    Zc_ev = make_z(n_ev)

    m.configure_child_semantics(agezero_deterministic=True)
    nodes, n_params = m._fit_nodes(Xtr, Xev, Zc_tr, Zc_ev)
    assert isinstance(n_params, int)
    assert n_params > 0
    for n, _ in m.Z_LAYOUT:
        assert n in nodes
        assert hasattr(nodes[n]["head"], "n_params")
        assert isinstance(nodes[n]["head"].n_params, int)
        assert nodes[n]["head"].n_params > 0

    k0 = m.fit_constant_heads(Zc_tr, Zc_ev, None, None)
    assert isinstance(k0["n_params"], int)
    assert k0["n_params"] > 0
    m.configure_child_semantics(agezero_deterministic=False)


if __name__ == "__main__":
    class _MonkeyPatch:
        def setattr(self, target, name, value):
            setattr(target, name, value)

    test_block_definition()
    test_ztp_rate_differs_from_positive_mean()
    test_ztp_pmf_sums_to_one()
    test_ztp_nll_finite_and_min_near_mle()
    test_hurdle_nll_decomposes()
    test_build_transition_sample_count_invariant(_MonkeyPatch())
    test_full_reconstruction_closure(_MonkeyPatch())
    test_ridge_parity()
    test_agezero_audit_harness()
    test_child_agezero_configuration()
    test_gaussian_head_n_params()
    test_fit_nodes_has_parameter_count()
    print("ALL tests in test_dynamic_pgm1a1_support_semantics_v1 PASSED successfully.")
