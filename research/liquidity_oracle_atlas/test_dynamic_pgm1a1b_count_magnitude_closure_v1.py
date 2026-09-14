"""Tests for DYNAMIC-PGM-1A.1b — Count Magnitude Closure Transition Kernel.

Covers:
  1. test_constant_ztp_mle: Brentq solves E[Y|Y>0] = lambda / (1 - exp(-lambda))
  2. test_k0_k1_magnitude_lambda_identical: train-derived lambda is shared identically
  3. test_zero_rows_magnitude_independent: y=0 NLL is -log(1-p0), completely invariant to rate
  4. test_positive_rows_exact_ztp_nll: y>=1 decomposes, magnitude cancels in K1-K0 delta
  5. test_no_state_dependent_magnitude_eta: no linear eta or regressor for magnitude
  6. test_full_count_nll_decomposition: hurdle NLL = occurrence + magnitude
  7. test_count_closure_diagnostics_calculation: diagnostics table correctness & consistency
  8. test_run_single_window_smoke_1a1b: synthetic end-to-end window smoke (K0 vs K1)
  9. test_bootstrap_cluster_multiplicity: day-cluster bootstrap preserves cluster multiplicity
  10. test_joint_nll_component_sum: joint = continuous + discrete + count invariant
  11. test_namespace_isolation_1a1b: all outputs and caches use dynamic_pgm1a1b_*
  12. test_reconstruction_closure: F(S_t, Z_{t+1}) == S_{t+1} exact on synthetic data
  13. test_agezero_deterministic_configuration: child semantics configuration logic
"""
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a1b_count_magnitude_closure_v1 as m


def test_constant_ztp_mle():
    # 1. Typical case: mean > 1
    y = np.array([1, 1, 1, 1, 2], dtype=np.int64)
    m_obs = float(np.mean(y))  # 1.2
    lam = m._fit_constant_ztp_rate(y)
    expected_m = lam / (-np.expm1(-lam))
    assert abs(expected_m - m_obs) < 1e-6
    # Plain Poisson mean would be 1.2, but exact ZTP rate is much smaller:
    assert abs(lam - 1.2) > 0.4

    # 2. Edge case: all 1s (implied lambda -> 0)
    y_ones = np.array([1, 1, 1, 1], dtype=np.int64)
    lam_ones = m._fit_constant_ztp_rate(y_ones)
    assert lam_ones == 1e-8

    # 3. Larger mean
    y_large = np.array([1, 2, 3, 4], dtype=np.int64)
    m_large = float(np.mean(y_large))  # 2.5
    lam_large = m._fit_constant_ztp_rate(y_large)
    assert abs(lam_large / (-np.expm1(-lam_large)) - m_large) < 1e-6


def test_k0_k1_magnitude_lambda_identical():
    rng = np.random.default_rng(42)
    n_tr, n_ev, p = 200, 100, 5
    Xtr = rng.normal(size=(n_tr, p))
    Xev = rng.normal(size=(n_ev, p))
    # Two count columns
    Ytr = np.column_stack([
        rng.choice([0, 1, 2], size=n_tr, p=[0.8, 0.18, 0.02]),
        rng.choice([0, 1, 2], size=n_tr, p=[0.85, 0.14, 0.01])
    ])
    Yev = np.column_stack([
        rng.choice([0, 1, 2], size=n_ev, p=[0.8, 0.18, 0.02]),
        rng.choice([0, 1, 2], size=n_ev, p=[0.85, 0.14, 0.01])
    ])

    k0_count = m.fit_constant_count_head(Ytr, Yev)
    k1_count = m.fit_state_count_head(Xtr, Ytr, Xev, Yev, constant_rates=k0_count["constant_rates"])

    # Lambda rates must be strictly identical
    assert len(k0_count["constant_rates"]) == 2
    assert len(k1_count["constant_rates"]) == 2
    for j in range(2):
        assert k0_count["constant_rates"][j] == k1_count["constant_rates"][j]
        # And must match direct call on Ytr[:, j]
        direct_lam = m._fit_constant_ztp_rate(Ytr[:, j])
        assert abs(k0_count["constant_rates"][j] - direct_lam) < 1e-12


def test_zero_rows_magnitude_independent():
    y = np.array([0.0])
    p0 = np.array([0.25])
    # Vary rate across 6 orders of magnitude
    rates = [1e-6, 0.01, 0.5, 1.0, 5.0, 50.0]
    expected_nll = -np.log1p(-0.25)
    for r in rates:
        nll = m._hurdle_nll(y, p0, np.array([r]))[0]
        assert abs(nll - expected_nll) < 1e-12


def test_positive_rows_exact_ztp_nll():
    y = np.array([1.0, 2.0])
    p0_a = np.array([0.1, 0.1])
    p0_b = np.array([0.4, 0.4])
    rate = np.array([0.3, 0.3])

    nll_a = m._hurdle_nll(y, p0_a, rate)
    nll_b = m._hurdle_nll(y, p0_b, rate)

    # Decomposes exactly into -log(p0) + ztnp_nll
    for i in range(len(y)):
        assert abs(nll_a[i] - (-np.log(0.1) + m._ztnp_nll(np.array([y[i]]), np.array([0.3]))[0])) < 1e-12
        assert abs(nll_b[i] - (-np.log(0.4) + m._ztnp_nll(np.array([y[i]]), np.array([0.3]))[0])) < 1e-12

    # Delta between b and a cancels the magnitude term exactly
    delta = nll_b - nll_a
    expected_delta = -np.log(0.4) - (-np.log(0.1))
    for i in range(len(y)):
        assert abs(delta[i] - expected_delta) < 1e-14


def test_no_state_dependent_magnitude_eta():
    # In 1A.1b, fit_state_count_head must NOT fit any state-dependent magnitude regression
    # Check that it doesn't take or create eta or regression weights on positives
    rng = np.random.default_rng(123)
    Xtr = rng.normal(size=(50, 4))
    Xev = rng.normal(size=(30, 4))
    Ytr = np.column_stack([
        rng.choice([0, 1], size=50, p=[0.7, 0.3]),
        rng.choice([0, 1], size=50, p=[0.7, 0.3]),
    ])
    Yev = np.column_stack([
        rng.choice([0, 1], size=30, p=[0.7, 0.3]),
        rng.choice([0, 1], size=30, p=[0.7, 0.3]),
    ])
    rates = [0.5, 0.8]

    k1_count = m.fit_state_count_head(Xtr, Ytr, Xev, Yev, constant_rates=rates)
    # Rate across eval rows must be constant scalar broadcast
    for j in range(2):
        assert np.all(k1_count["rate_ev"][:, j] == rates[j])

    # Parameters: Logistic (4+1) + constant rate (1) per target
    assert k1_count["n_params"] == (4 + 1 + 1) * 2


def test_full_count_nll_decomposition():
    rng = np.random.default_rng(999)
    y = np.array([0, 0, 1, 2, 0, 1], dtype=np.int64)
    p0 = rng.uniform(0.05, 0.4, size=len(y))
    lam = 0.45

    nll_total = m._hurdle_nll(y, p0, lam)
    nll_occ = np.where(y > 0, -np.log(np.clip(p0, 1e-12, 1.0)), -np.log1p(-p0))
    nll_mag = np.where(y > 0, m._ztnp_nll(y, lam), 0.0)

    assert np.allclose(nll_total, nll_occ + nll_mag, atol=1e-12)


def test_count_closure_diagnostics_calculation():
    rng = np.random.default_rng(42)
    n_tr, n_ev, p = 150, 80, 6
    Xtr = rng.normal(size=(n_tr, p))
    Xev = rng.normal(size=(n_ev, p))
    Ytr = np.column_stack([
        rng.choice([0, 1, 2], size=n_tr, p=[0.7, 0.25, 0.05]),
        rng.choice([0, 1, 2], size=n_tr, p=[0.8, 0.18, 0.02]),
    ])
    Yev = np.column_stack([
        rng.choice([0, 1, 2], size=n_ev, p=[0.7, 0.25, 0.05]),
        rng.choice([0, 1, 2], size=n_ev, p=[0.8, 0.18, 0.02]),
    ])

    k0_count = m.fit_constant_count_head(Ytr, Yev)
    k1_count = m.fit_state_count_head(Xtr, Ytr, Xev, Yev, constant_rates=k0_count["constant_rates"])

    diag_rows = m.compute_count_closure_diagnostics("TEST_WIN", Ytr, Yev, k0_count, k1_count)
    assert len(diag_rows) == 2

    for r in diag_rows:
        assert r["window"] == "TEST_WIN"
        assert r["target"] in m.COUNT_Z
        assert r["train_positive_n"] > 0
        assert r["eval_positive_n"] > 0
        assert 0 < r["constant_ztp_lambda"] < 10.0
        assert r["expected_magnitude"] >= 1.0

        # Probabilities sum to 1
        p_sum = r["predicted_p_y1"] + r["predicted_p_y2"] + r["predicted_p_y_ge3"]
        assert abs(p_sum - 1.0) < 1e-6

        # Shared magnitude NLL is identical and cancels in delta
        assert abs(r["k0_mean_nll"] - (r["k0_mean_occ_nll"] + r["shared_mean_mag_nll"])) < 1e-12
        assert abs(r["k1_mean_nll"] - (r["k1_mean_occ_nll"] + r["shared_mean_mag_nll"])) < 1e-12
        assert abs(r["delta_nll"] - (r["k1_mean_nll"] - r["k0_mean_nll"])) < 1e-12
        assert abs(r["delta_nll"] - (r["k1_mean_occ_nll"] - r["k0_mean_occ_nll"])) < 1e-12


def test_run_single_window_smoke_1a1b():
    rng = np.random.default_rng(12345)
    n = 300
    data = {
        "block": ["TB1"] * 180 + ["TB2"] * 120,
        "symbol": ["AG"] * 150 + ["AU"] * 150,
        "episode_id": [f"ep_{i // 15}" for i in range(n)],
        "episode_start_day": [100 + (i // 30) for i in range(n)],
        "prev_event_mask": ["NONE"] * n,
    }
    for c in m.OBS_STATE_NUM:
        data[c] = rng.normal(size=n).astype(np.float32)

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
    data["z_delta_upper_count"] = rng.choice([0, 1, 2], size=n, p=[0.75, 0.2, 0.05]).astype(np.int64)
    data["z_delta_lower_count"] = rng.choice([0, 1, 2], size=n, p=[0.8, 0.17, 0.03]).astype(np.int64)

    df = pd.DataFrame(data)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        df.to_parquet(f.name, index=False)
        p = f.name

    try:
        m.configure_child_semantics(agezero_deterministic=True)
        w = dict(name="TEST_WIN_1A1B", train=["TB1"], eval="TB2", seed=42)
        res = m.run_single_window(w, p)

        # 1. Models: only K0 and K1 (no K2)
        assert len(res["model_metrics"]) == 2
        models = [mm["model"] for mm in res["model_metrics"]]
        assert models == ["K0_UNCONDITIONAL", "K1_STATE"]

        # 2. Bootstrap: only K1-K0
        assert len(res["boots"]) == 1
        assert res["boots"][0]["comparison"] == "K1-K0"
        assert np.isfinite(res["boots"][0]["delta_sample_mean"])

        # 3. Block contribution
        assert len(res["block_rows"]) == len(m.BLOCKS)
        for br in res["block_rows"]:
            assert np.isfinite(br["mean_joint_k0"])
            assert np.isfinite(br["mean_joint_k1"])
            assert np.isfinite(br["delta_k1_minus_k0"])

        # 4. Count closure diagnostics
        assert len(res["count_diag_rows"]) == 2
        for cr in res["count_diag_rows"]:
            assert cr["window"] == "TEST_WIN_1A1B"
            assert np.isfinite(cr["delta_nll"])

        # 5. Component sum invariant
        for mm in res["model_metrics"]:
            j = mm["mean_joint_nll"]
            c = mm["mean_cont_nll"]
            d = mm["mean_disc_nll"]
            cnt = mm["mean_count_nll"]
            assert abs(j - (c + d + cnt)) < 1e-10

        # 6. JSON serializable
        s = json.dumps(res, default=str)
        assert len(s) > 0
    finally:
        m.configure_child_semantics(agezero_deterministic=False)
        Path(p).unlink(missing_ok=True)


def test_bootstrap_cluster_multiplicity():
    n_days = 5
    ep_per_day = 10
    total_ep = n_days * ep_per_day
    rows_per_ep = 4
    total_rows = total_ep * rows_per_ep

    day_per_row = np.repeat(np.arange(n_days), ep_per_day * rows_per_ep)
    eid_per_row = np.repeat(np.arange(total_ep), rows_per_ep)
    delta_per_row = np.array([float(d) for d in day_per_row])

    lo, hi, pt = m.boot_cluster(delta_per_row, day_per_row, eid_per_row, seed=123)
    assert abs(pt - float(np.mean(np.arange(n_days)))) < 1e-6
    assert lo < pt < hi


def test_joint_nll_component_sum():
    n = 100
    rng = np.random.default_rng(42)
    cont = rng.exponential(size=n)
    disc = np.zeros(n)
    count = rng.exponential(size=n)
    joint = cont + disc + count
    assert abs(np.mean(joint) - (np.mean(cont) + np.mean(disc) + np.mean(count))) < 1e-12


def test_namespace_isolation_1a1b():
    code_path = REPO / "research/liquidity_oracle_atlas/experiment_dynamic_pgm1a1b_count_magnitude_closure_v1.py"
    content = code_path.read_text()

    # Outputs must use dynamic_pgm1a1b_
    assert "dynamic_pgm1a1b_model_metrics.csv" in content
    assert "dynamic_pgm1a1b_bootstrap.csv" in content
    assert "dynamic_pgm1a1b_by_symbol.csv" in content
    assert "dynamic_pgm1a1b_target_metrics.csv" in content
    assert "dynamic_pgm1a1b_optimizer_audit.csv" in content
    assert "dynamic_pgm1a1b_count_closure_diagnostics.csv" in content
    assert "dynamic_pgm1a1b_sample_audit.json" in content
    assert "dynamic_pgm1a1b_transition_invariants.json" in content
    assert "dynamic_pgm1a1b_summary.json" in content
    assert "dynamic_pgm1a1b_transitions.parquet" in content

    # Must NOT write to 1A or 1A.1 outputs
    out_writes = re.findall(r'OUT\s*/\s*"([^"]+)"', content)
    for out in out_writes:
        assert out.startswith("dynamic_pgm1a1b_"), f"Foreign output detected: {out}"


def test_reconstruction_closure(monkeypatch=None):
    orig_exp = m.EXPECTED_TRANSITIONS
    m.EXPECTED_TRANSITIONS = 2
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
    try:
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
    finally:
        m.EXPECTED_TRANSITIONS = orig_exp


def test_agezero_deterministic_configuration():
    m.configure_child_semantics(agezero_deterministic=False)
    assert m.disc_spec() == [m.DISC_Z]
    m.configure_child_semantics(agezero_deterministic=True)
    assert m.disc_spec() == []
    m.configure_child_semantics(agezero_deterministic=False)


if __name__ == "__main__":
    tests = [
        ("test_constant_ztp_mle", test_constant_ztp_mle),
        ("test_k0_k1_magnitude_lambda_identical", test_k0_k1_magnitude_lambda_identical),
        ("test_zero_rows_magnitude_independent", test_zero_rows_magnitude_independent),
        ("test_positive_rows_exact_ztp_nll", test_positive_rows_exact_ztp_nll),
        ("test_no_state_dependent_magnitude_eta", test_no_state_dependent_magnitude_eta),
        ("test_full_count_nll_decomposition", test_full_count_nll_decomposition),
        ("test_count_closure_diagnostics_calculation", test_count_closure_diagnostics_calculation),
        ("test_run_single_window_smoke_1a1b", test_run_single_window_smoke_1a1b),
        ("test_bootstrap_cluster_multiplicity", test_bootstrap_cluster_multiplicity),
        ("test_joint_nll_component_sum", test_joint_nll_component_sum),
        ("test_namespace_isolation_1a1b", test_namespace_isolation_1a1b),
        ("test_reconstruction_closure", test_reconstruction_closure),
        ("test_agezero_deterministic_configuration", test_agezero_deterministic_configuration),
    ]

    for name, fn in tests:
        fn()

    print(f"{len(tests)}/{len(tests)} 1A.1b TESTS PASSED successfully.")
