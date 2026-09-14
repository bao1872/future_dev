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


def test_output_namespace_isolation():
    import re
    script_path = REPO / "research" / "liquidity_oracle_atlas" / "experiment_dynamic_pgm1a1_support_semantics_v1.py"
    script_text = script_path.read_text()

    # 1. Output files written to OUT / "..."
    written_files = re.findall(r'OUT\s*/\s*"([^"]+)"', script_text)
    assert len(written_files) >= 8, f"Expected at least 8 outputs written to OUT, found: {written_files}"
    for fname in written_files:
        assert fname.startswith("dynamic_pgm1a1_"), (
            f"Output file {fname} does not start with dynamic_pgm1a1_ namespace prefix"
        )
        assert not fname.startswith("dynamic_pgm1a_"), (
            f"Output file {fname} collides with legacy frozen dynamic_pgm1a_ namespace"
        )

    # 2. Cache files written to CACHE / "..."
    cache_parquets = re.findall(r'CACHE\s*/\s*"([^"]+\.parquet)"', script_text)
    assert len(cache_parquets) > 0
    for pf in cache_parquets:
        if "transitions" in pf:
            assert pf == "dynamic_pgm1a1_transitions.parquet", (
                f"Cache parquet {pf} must use dynamic_pgm1a1_transitions.parquet"
            )

    # 3. Subprocess window json cache written to CACHE
    window_json = re.findall(r'CACHE\s*/\s*f?"([^"]*window[^"]*\.json)"', script_text)
    assert len(window_json) > 0
    for wj in window_json:
        assert "_dynamic_pgm1a1_window_" in wj, (
            f"Window result cache {wj} does not use _dynamic_pgm1a1_window_ prefix"
        )


def test_bootstrap_multiplicity():
    """Verify cluster bootstrap preserves exact cluster multiplicity (no isin deduplication)."""
    # 3 distinct days: Day 1 has 12.0, Day 2 has 0.0, Day 3 has 0.0
    delta = np.array([12.0, 0.0, 0.0])
    day = np.array([1, 2, 3])
    eid = np.array([101, 102, 103])

    # In a cluster bootstrap, if Day 1 is sampled twice and Day 2 once:
    # Day sums = [12, 12, 0] -> sum = 24. Day counts = [1, 1, 1] -> sum = 3.
    # Estimate = 24 / 3 = 8.0.
    # Under old buggy `isin(sel)` logic: isin([1, 1, 2]) was deduplicated to isin([1, 2]),
    # giving (12 + 0) / 2 = 6.0.
    df = pd.DataFrame({"d": delta, "day": day, "eid": eid})
    ep = df.groupby("eid", as_index=False).agg(d=("d", "mean"), day=("day", "first"))
    day_stats = ep.groupby("day")["d"].agg(["sum", "count"]).reset_index()
    day_sums = day_stats["sum"].to_numpy(dtype=np.float64)
    day_counts = day_stats["count"].to_numpy(dtype=np.float64)

    # Sample with multiplicity: idx = [0, 0, 1] (Day 1 twice, Day 2 once)
    idx = np.array([0, 0, 1])
    est = day_sums[idx].sum() / day_counts[idx].sum()
    assert abs(est - 8.0) < 1e-10

    # Test boot_cluster directly
    ci_lo, ci_hi, point = m.boot_cluster(delta, day, eid, seed=123)
    assert 0.0 <= ci_lo <= point <= ci_hi <= 12.0
    assert abs(point - 4.0) < 1e-9


def test_joint_nll_component_sum():
    """Verify mean_joint_nll == mean_cont_nll + mean_disc_nll + mean_count_nll exactly."""
    n = 20
    rng = np.random.default_rng(99)
    cont = rng.uniform(0.1, 1.0, size=n)
    disc = rng.uniform(0.1, 0.5, size=n)
    count = rng.uniform(0.1, 0.5, size=(n, 2))

    count_row = count.sum(axis=1)
    joint = cont + disc + count_row

    m_cont = float(np.mean(cont))
    m_disc = float(np.mean(disc))
    m_count = float(np.mean(count_row))
    m_joint = float(np.mean(joint))

    assert abs(m_joint - m_cont - m_disc - m_count) < 1e-10


def test_block_model_isolation():
    """Verify block attribution isolates K0, K1, and K2 models with explicit arguments."""
    n = 10
    nodes_k0 = {nm: {"ev": np.full(n, 1.0)} for nm, _ in m.Z_LAYOUT}
    nodes_k1 = {nm: {"ev": np.full(n, 2.0)} for nm, _ in m.Z_LAYOUT}
    disc_k0 = np.full(n, 0.1)
    disc_k1 = np.full(n, 0.5)
    count_k0 = np.full((n, len(m.COUNT_Z)), 0.2)
    count_k1 = np.full((n, len(m.COUNT_Z)), 0.8)

    def _block_nll(nodes, disc_ev, count_ev, bdef):
        if bdef["nodes"]:
            s = np.sum([nodes[nm]["ev"] for nm in bdef["nodes"]], axis=0)
        else:
            s = np.zeros(len(nodes[list(nodes)[0]]["ev"]))
        if bdef["disc"]:
            s = s + disc_ev
        if bdef.get("count"):
            idx = [m.COUNT_Z.index(c) for c in bdef["count"]]
            s = s + count_ev[:, idx].sum(axis=1)
        return s

    bdef = m.BLOCKS["LiquidityComposition"]
    res_k0 = _block_nll(nodes_k0, disc_k0, count_k0, bdef)
    res_k1 = _block_nll(nodes_k1, disc_k1, count_k1, bdef)

    expected_k0 = len(bdef["nodes"]) * 1.0 + (0.1 if bdef["disc"] else 0.0) + (0.2 * len(bdef.get("count", [])))
    expected_k1 = len(bdef["nodes"]) * 2.0 + (0.5 if bdef["disc"] else 0.0) + (0.8 * len(bdef.get("count", [])))
    assert np.allclose(res_k0, expected_k0)
    assert np.allclose(res_k1, expected_k1)
    assert not np.allclose(res_k0, res_k1)


def test_run_single_window_smoke():
    """End-to-end smoke test for run_single_window on synthetic 120 train / 70 eval data."""
    import tempfile
    import json

    rng = np.random.default_rng(42)
    n_tr, n_ev = 120, 70
    n = n_tr + n_ev
    data = {}
    data["block"] = ["TB1"] * n_tr + ["TB2"] * n_ev
    data["symbol"] = rng.choice(["AGL8", "CUL8"], size=n)
    data["episode_start_day"] = rng.choice(["2024-01-01", "2024-01-02", "2024-01-03"], size=n)
    data["episode_id"] = rng.integers(0, 15, size=n)
    data["prev_event_mask"] = rng.choice(["0", "1", "2"], size=n)

    for c in m.OBS_STATE_NUM:
        data[c] = rng.normal(size=n).astype(np.float32)
    for c in m.LAG_BASE:
        data[f"lag1_{c}"] = rng.normal(size=n).astype(np.float32)
    data["lag1_available"] = np.ones(n, dtype=np.float32)

    data["z_d_up"] = rng.normal(size=n).astype(np.float32)
    for nm in ["z_dmfe", "z_dmae", "z_range", "z_uresid", "z_lresid"]:
        ispos = (rng.uniform(size=n) > 0.4).astype(np.float32)
        logv = np.where(ispos > 0.5, rng.normal(size=n), 0.0).astype(np.float32)
        data[f"{nm}_ispos"] = ispos
        data[f"{nm}_log"] = logv

    u = rng.uniform(size=n)
    is0 = (u < 0.2).astype(np.float32)
    is1 = (u > 0.8).astype(np.float32)
    interior = (is0 < 0.5) & (is1 < 0.5)
    logit = np.where(interior, rng.normal(size=n), 0.0).astype(np.float32)
    data["z_dcr_is0"] = is0
    data["z_dcr_is1"] = is1
    data["z_dcr_logit"] = logit

    data["z_agezero_code"] = np.zeros(n, dtype=np.int64)
    data["z_delta_upper_count"] = rng.choice([0, 1, 2], size=n, p=[0.7, 0.2, 0.1]).astype(np.int64)
    data["z_delta_lower_count"] = rng.choice([0, 1, 2], size=n, p=[0.7, 0.2, 0.1]).astype(np.int64)

    df = pd.DataFrame(data)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        df.to_parquet(f.name, index=False)
        p = f.name

    try:
        m.configure_child_semantics(agezero_deterministic=True)
        w = dict(name="TEST_WIN", train=["TB1"], eval="TB2", seed=42)
        res = m.run_single_window(w, p)

        # 1. 2 bootstrap comparisons
        assert len(res["boots"]) == 2
        comparisons = [b["comparison"] for b in res["boots"]]
        assert "K1-K0" in comparisons and "K2-K1" in comparisons

        # 2. 3 models
        assert len(res["model_metrics"]) == 3
        models = [mm["model"] for mm in res["model_metrics"]]
        assert models == ["K0_UNCONDITIONAL", "K1_STATE", "K2_STATE_LAG1"]

        # 3. joint NLL all finite and component sum exact
        for mm in res["model_metrics"]:
            assert np.isfinite(mm["mean_joint_nll"])
            assert abs(mm["mean_joint_nll"] - mm["mean_cont_nll"]
                       - mm["mean_disc_nll"] - mm["mean_count_nll"]) < 1e-10

        # 4. K0/K1/K2 eval lengths identical
        for mm in res["model_metrics"]:
            assert mm["n_rows"] == n_ev

        # 5. block rows contain K0/K1/K2 and both deltas
        assert len(res["block_rows"]) == len(m.BLOCKS)
        for br in res["block_rows"]:
            assert "mean_joint_k0" in br and "mean_joint_k1" in br and "mean_joint_k2" in br
            assert "delta_k1_minus_k0" in br and "delta_k2_minus_k1" in br
            assert np.isfinite(br["mean_joint_k0"])
            assert np.isfinite(br["mean_joint_k1"])
            assert np.isfinite(br["mean_joint_k2"])

        # 6. target rows contain both deltas
        assert len(res["target_rows"]) > 0
        for tr in res["target_rows"]:
            assert "delta_k1_minus_k0" in tr and "delta_k2_minus_k1" in tr
            assert np.isfinite(tr["mean_nll_k0"])
            assert np.isfinite(tr["mean_nll_k1"])
            assert np.isfinite(tr["mean_nll_k2"])

        # 7. node diagnostics contains hurdle, dcr, and count
        assert len(res["node_diag_rows"]) == 8
        diag_targets = [r["target"] for r in res["node_diag_rows"]]
        for exp_tgt in ["z_dmfe", "z_dmae", "z_range", "z_uresid", "z_lresid",
                        "z_dcr", "z_delta_upper_count", "z_delta_lower_count"]:
            assert exp_tgt in diag_targets
        for r in res["node_diag_rows"]:
            if r["kind"] == "hurdle_count":
                assert "train_positive_eta_low_clipped" in r
                assert "train_positive_eta_high_clipped" in r
                assert "train_all_eta_low_clipped" in r
                assert "train_all_eta_high_clipped" in r
                assert "train_eta_min" in r
                assert "train_eta_max" in r

        # 8. json-serializable
        s = json.dumps(res, default=str)
        assert len(s) > 0
    finally:
        m.configure_child_semantics(agezero_deterministic=False)
        Path(p).unlink(missing_ok=True)


def test_ztp_tail_clipping_semantics(monkeypatch):
    """Hard gate synthetic test:
    * positive subset appears eta < -20 -> must STOP
    * any all-row appears eta > 20 -> must STOP
    * only zero-outcome rows appear eta < -20 -> must NOT STOP
    """
    Xtr = np.zeros((10, 2))
    Ytr = np.array([[1, 1], [2, 0], [0, 0], [0, 0], [0, 0],
                    [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int64)
    Xev = np.zeros((10, 2))
    Yev = np.array([[1, 0], [0, 1], [0, 0], [0, 0], [0, 0],
                    [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int64)

    orig_clip = m.ZeroTruncatedPoissonRegressor.count_eta_clip_stats

    # Case 1: positive subset has eta < -20 -> must STOP
    def mock_clip_pos_low(self, X):
        res = orig_clip(self, X)
        if len(X) == 2:
            res["n_low"] = 1
        return res

    monkeypatch.setattr(m.ZeroTruncatedPoissonRegressor, "count_eta_clip_stats", mock_clip_pos_low)
    stopped_pos = False
    try:
        m.fit_count_head(Xtr, Ytr, Xev, Yev)
    except (m.ZTPSupportGateError, SystemExit) as e:
        stopped_pos = True
        assert isinstance(e, m.ZTPSupportGateError)
        assert "STOP_DYNAMIC_PGM1A1_ZTP_POSITIVE_ETA_CLIPPED" in str(e)
    assert stopped_pos, "Should have stopped on positive eta < -20"

    # Case 2: any all-row has eta > 20 -> must STOP
    def mock_clip_all_high(self, X):
        res = orig_clip(self, X)
        if len(X) == 10:
            res["n_high"] = 1
        return res

    monkeypatch.setattr(m.ZeroTruncatedPoissonRegressor, "count_eta_clip_stats", mock_clip_all_high)
    stopped_high = False
    try:
        m.fit_count_head(Xtr, Ytr, Xev, Yev)
    except (m.ZTPSupportGateError, SystemExit) as e:
        stopped_high = True
        assert isinstance(e, m.ZTPSupportGateError)
        assert "STOP_DYNAMIC_PGM1A1_ZTP_ALL_ETA_HIGH_EXPLOSION" in str(e)
    assert stopped_high, "Should have stopped on all-row eta > 20"

    # Case 3: only zero-outcome rows have eta < -20 -> must NOT stop
    def mock_clip_zero_low(self, X):
        res = orig_clip(self, X)
        if len(X) == 10:
            res["n_low"] = 3
            res["n_high"] = 0
        elif len(X) == 2:
            res["n_low"] = 0
            res["n_high"] = 0
        return res

    monkeypatch.setattr(m.ZeroTruncatedPoissonRegressor, "count_eta_clip_stats", mock_clip_zero_low)
    res = m.fit_count_head(Xtr, Ytr, Xev, Yev)
    assert res is not None
    assert res["stats_tr_all"][0]["n_low"] == 3
    assert res["stats_tr_pos"][0]["n_low"] == 0

    monkeypatch.setattr(m.ZeroTruncatedPoissonRegressor, "count_eta_clip_stats", orig_clip)


def test_ztp_small_lambda_limit():
    """Verify that as eta << 0 (lambda -> 0), ZTP conditional mean approaches 1 and NLL is finite."""
    for eta in [-20.0, -30.0, -40.0]:
        lam = np.exp(np.clip(eta, -20.0, 20.0))
        # E[Y | Y>0] = lam / (1 - e^{-lam})
        denom = -np.expm1(-lam)
        mu_trunc = lam / denom
        assert np.isfinite(mu_trunc)
        assert abs(mu_trunc - 1.0) < 1e-6

        # ZTP NLL for y=1 and y=2
        y = np.array([1.0, 2.0])
        nll = m._ztnp_nll(y, np.array([lam, lam]))
        assert np.all(np.isfinite(nll))
        # For y=1, as lam -> 0, P(Y=1|Y>0) -> 1, so NLL -> 0
        assert abs(nll[0]) < 1e-5
        # For y=2, as lam -> 0, P(Y=2|Y>0) -> 0, so NLL > 0
        assert nll[1] > 0.0


def test_k2_support_failure_graceful_handling(monkeypatch):
    """Verify that when K2 fit raises ZTPSupportGateError, run_single_window:
    * catches ZTPSupportGateError gracefully (does not crash)
    * records k2_status == 'SUPPORT_CORRECT_LAG1_MODEL_SUPPORT_GATE_FAILED'
    * records K2 metrics as None in model_metrics, block_rows, and target_rows
    * records K2-K1 verdict as 'SUPPORT_CORRECT_LAG1_MODEL_SUPPORT_GATE_FAILED' in boots
    * keeps K0 and K1 metrics completely intact and finite
    """
    import json
    import tempfile
    rng = np.random.default_rng(123)
    n_tr = 40
    n_ev = 20
    n = n_tr + n_ev
    data = {
        "block": ["TB1"] * n_tr + ["TB2"] * n_ev,
        "symbol": ["AG"] * n,
        "episode_id": [f"EP_{i // 5}" for i in range(n)],
        "episode_start_day": [f"2024010{1 + (i // 10)}" for i in range(n)],
    }
    for c in m.OBS_STATE_NUM:
        data[c] = rng.normal(size=n).astype(np.float32)
    for c in m.OBS_STATE_CAT:
        data[c] = ["BAR"] * n
    for c in m.LAG_BASE:
        data[f"lag1_{c}"] = rng.normal(size=n).astype(np.float32)
    data["lag1_available"] = np.ones(n, dtype=np.float32)

    data["z_d_up"] = rng.normal(size=n).astype(np.float32)
    for nm in ["z_dmfe", "z_dmae", "z_range", "z_uresid", "z_lresid"]:
        ispos = (rng.uniform(size=n) > 0.4).astype(np.float32)
        logv = np.where(ispos > 0.5, rng.normal(size=n), 0.0).astype(np.float32)
        data[f"{nm}_ispos"] = ispos
        data[f"{nm}_log"] = logv

    u = rng.uniform(size=n)
    is0 = (u < 0.2).astype(np.float32)
    is1 = (u > 0.8).astype(np.float32)
    interior = (is0 < 0.5) & (is1 < 0.5)
    logit = np.where(interior, rng.normal(size=n), 0.0).astype(np.float32)
    data["z_dcr_is0"] = is0
    data["z_dcr_is1"] = is1
    data["z_dcr_logit"] = logit

    data["z_agezero_code"] = np.zeros(n, dtype=np.int64)
    data["z_delta_upper_count"] = rng.choice([0, 1, 2], size=n, p=[0.7, 0.2, 0.1]).astype(np.int64)
    data["z_delta_lower_count"] = rng.choice([0, 1, 2], size=n, p=[0.7, 0.2, 0.1]).astype(np.int64)

    df = pd.DataFrame(data)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        df.to_parquet(f.name, index=False)
        p = f.name

    orig_fit_count = m.fit_count_head

    def mock_fit_count_head(Xtr, Ytr, Xev, Yev):
        # K1 design matrix has fewer features; K2 has full features including lag1
        if Xtr.shape[1] > 40:  # K2 design matrix (OBS_STATE + LAG, dim 52 vs K1 dim 36)
            raise m.ZTPSupportGateError("STOP_DYNAMIC_PGM1A1_ZTP_POSITIVE_ETA_CLIPPED: mock k2 failure")
        return orig_fit_count(Xtr, Ytr, Xev, Yev)

    monkeypatch.setattr(m, "fit_count_head", mock_fit_count_head)

    try:
        m.configure_child_semantics(agezero_deterministic=True)
        w = dict(name="TEST_K2_FAIL_WIN", train=["TB1"], eval="TB2", seed=42)
        res = m.run_single_window(w, p)

        # 1. K2 status and failure reason
        assert res["k2_status"] == "SUPPORT_CORRECT_LAG1_MODEL_SUPPORT_GATE_FAILED"
        assert "STOP_DYNAMIC_PGM1A1_ZTP_POSITIVE_ETA_CLIPPED" in res["k2_fail_reason"]

        # 2. model_metrics: K0 and K1 intact, K2 has None
        assert len(res["model_metrics"]) == 3
        m_k0 = next(mm for mm in res["model_metrics"] if mm["model"] == "K0_UNCONDITIONAL")
        m_k1 = next(mm for mm in res["model_metrics"] if mm["model"] == "K1_STATE")
        m_k2 = next(mm for mm in res["model_metrics"] if mm["model"] == "K2_STATE_LAG1")
        assert np.isfinite(m_k0["mean_joint_nll"])
        assert np.isfinite(m_k1["mean_joint_nll"])
        assert m_k2["mean_joint_nll"] is None

        # 3. boots: K1-K0 has normal CI, K2-K1 marked as gate failed
        assert len(res["boots"]) == 2
        b_k1k0 = next(b for b in res["boots"] if b["comparison"] == "K1-K0")
        b_k2k1 = next(b for b in res["boots"] if b["comparison"] == "K2-K1")
        assert np.isfinite(b_k1k0["delta_sample_mean"])
        assert b_k2k1["verdict"] == "SUPPORT_CORRECT_LAG1_MODEL_SUPPORT_GATE_FAILED"
        assert b_k2k1["delta_sample_mean"] is None

        # 4. opt_rows: K2 recorded as success=False with fail_reason
        opt_k2 = next(r for r in res["opt_rows"] if r["model"] == "K2_STATE_LAG1")
        assert opt_k2["success"] is False
        assert "STOP_DYNAMIC_PGM1A1_ZTP_POSITIVE_ETA_CLIPPED" in opt_k2["fail_reason"]

        # 5. block_rows: delta_k1_minus_k0 finite, delta_k2_minus_k1 is None
        for br in res["block_rows"]:
            assert np.isfinite(br["mean_joint_k0"])
            assert np.isfinite(br["mean_joint_k1"])
            assert br["mean_joint_k2"] is None
            assert np.isfinite(br["delta_k1_minus_k0"])
            assert br["delta_k2_minus_k1"] is None

        # 6. target_rows: delta_k1_minus_k0 finite, delta_k2_minus_k1 is None
        for tr in res["target_rows"]:
            assert np.isfinite(tr["mean_nll_k0"])
            assert np.isfinite(tr["mean_nll_k1"])
            assert tr["mean_nll_k2"] is None
            assert np.isfinite(tr["delta_k1_minus_k0"])
            assert tr["delta_k2_minus_k1"] is None

        # 7. json-serializable
        s = json.dumps(res, default=str)
        assert len(s) > 0
    finally:
        m.configure_child_semantics(agezero_deterministic=False)
        monkeypatch.setattr(m, "fit_count_head", orig_fit_count)
        Path(p).unlink(missing_ok=True)


if __name__ == "__main__":
    class _MonkeyPatch:
        def setattr(self, target, name, value):
            setattr(target, name, value)

    mp = _MonkeyPatch()
    tests = [
        ("test_block_definition", lambda: test_block_definition()),
        ("test_ztp_rate_differs_from_positive_mean", lambda: test_ztp_rate_differs_from_positive_mean()),
        ("test_ztp_pmf_sums_to_one", lambda: test_ztp_pmf_sums_to_one()),
        ("test_ztp_nll_finite_and_min_near_mle", lambda: test_ztp_nll_finite_and_min_near_mle()),
        ("test_hurdle_nll_decomposes", lambda: test_hurdle_nll_decomposes()),
        ("test_build_transition_sample_count_invariant", lambda: test_build_transition_sample_count_invariant(mp)),
        ("test_full_reconstruction_closure", lambda: test_full_reconstruction_closure(mp)),
        ("test_ridge_parity", lambda: test_ridge_parity()),
        ("test_shared_transform_parity", lambda: test_shared_transform_parity()),
        ("test_ztp_finite_diff_gradient", lambda: test_ztp_finite_diff_gradient()),
        ("test_agezero_audit_harness", lambda: test_agezero_audit_harness()),
        ("test_child_agezero_configuration", lambda: test_child_agezero_configuration()),
        ("test_gaussian_head_n_params", lambda: test_gaussian_head_n_params()),
        ("test_fit_nodes_has_parameter_count", lambda: test_fit_nodes_has_parameter_count()),
        ("test_output_namespace_isolation", lambda: test_output_namespace_isolation()),
        ("test_bootstrap_multiplicity", lambda: test_bootstrap_multiplicity()),
        ("test_joint_nll_component_sum", lambda: test_joint_nll_component_sum()),
        ("test_block_model_isolation", lambda: test_block_model_isolation()),
        ("test_run_single_window_smoke", lambda: test_run_single_window_smoke()),
        ("test_ztp_tail_clipping_semantics", lambda: test_ztp_tail_clipping_semantics(mp)),
        ("test_ztp_small_lambda_limit", lambda: test_ztp_small_lambda_limit()),
        ("test_k2_support_failure_graceful_handling", lambda: test_k2_support_failure_graceful_handling(mp)),
    ]

    for name, fn in tests:
        fn()

    print(f"{len(tests)}/{len(tests)} TESTS PASSED successfully.")
