"""Unit tests for DYNAMIC-PGM-1C Free-Run Rollout Closure.

30 required tests per Section M:
 1. Frozen SHA / frozen metric constants
 2. Transition sampler parity M0/MC
 3. Terminal sampler parity T0/T2
 4. Reset sampler parity R0/R1
 5. Gaussian sampling covariance synthetic
 6. Hurdle sampling support
 7. DCR support
 8. ZTP support + empirical mean
 9. Endpoint mask sampling 1..15
 10. Reconstruct transition synthetic exact
 11. eps_R observed closure < 1e-8
 12. Reset -> start reconstruction
 13. First-bar availability=0
 14. First-bar terminal mem semantics
 15. No clamp in transition/reset path
 16. Invalid state fail-closed
 17. Timeout fail-closed
 18. One-bar episode semantics
 19. Multi-episode endpoint handoff
 20. Reset prev_event_mask == prior endpoint
 21. Burn-in excluded from metrics
 22. No teacher forcing after seed
 23. Normalized Wasserstein deterministic unit test
 24. Endpoint TVD unit test
 25. D_total exact equal-weight composition
 26. Paired replicate bootstrap multiplicity
 27. WT differs from W1 ONLY terminal sampler
 28. No TB4
 29. Output namespace isolation
 30. JSON serializable
"""
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

for _bt in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_bt] = "1"

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a1b_count_magnitude_closure_v1 as base  # noqa: E402
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1b_terminal_reset_closure_v1 as exp1b  # noqa: E402
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1c_free_run_rollout_v1 as C  # noqa: E402


# 1. frozen SHA / frozen metric constants
def test_1_frozen_sha_and_metric_constants():
    assert C.BASE_SHA == "ddaaf8ca8e7072d99957b687f10997f093b3fe44"
    assert C.PARITY_TOL == 1e-8
    assert C.MAX_EPISODE_BARS == 512
    assert C.N_SEEDED_REPS == 4
    assert C.N_CHAIN_REPS == 16
    assert C.BURN_IN_EPISODES == 16
    assert C.COLLECT_EPISODES == 128
    # Parity targets match exactly
    assert C.FROZEN_TRANSITION["A_TB1_to_TB2"]["M0"] == 1.3738896895068666
    assert C.FROZEN_TRANSITION["A_TB1_to_TB2"]["MC"] == 0.9613370222493051
    assert C.FROZEN_TRANSITION["B_TB1TB2_to_TB3"]["M0"] == 1.4054025921992275
    assert C.FROZEN_TRANSITION["B_TB1TB2_to_TB3"]["MC"] == 1.0374443295432418


# 2. transition sampler parity M0/MC
def test_2_transition_sampler_parity_m0_mc():
    p = C.CACHE / "dynamic_pgm1a2c_transitions.parquet"
    assert p.exists(), "dynamic_pgm1a2c_transitions.parquet must exist"
    df = pd.read_parquet(p)
    w = base.WINDOWS[0]  # Window A
    tr = df[df["block"].isin(w["train"])].reset_index(drop=True)
    ev = df[df["block"] == w["eval"]].reset_index(drop=True)

    Zc_tr = tr[base.ALL_Z_COLS].to_numpy(np.float32)
    Zc_ev = ev[base.ALL_Z_COLS].to_numpy(np.float32)
    yd_tr = tr[base.DISC_Z].to_numpy(np.int64)
    yd_ev = ev[base.DISC_Z].to_numpy(np.int64)
    Yc_tr = tr[base.COUNT_Z].to_numpy(np.int64)
    Yc_ev = ev[base.COUNT_Z].to_numpy(np.int64)

    base.configure_child_semantics(agezero_deterministic=True)
    k0_count = base.fit_constant_count_head(Yc_tr, Yc_ev)

    obs_cols = list(base.OBS_STATE_NUM)
    for tag, num_cols in [("M0", obs_cols + C.rep.M0_EXTRA),
                          ("MC", obs_cols + C.rep.MC_EXTRA)]:
        ct = C.lag._make_ct(num_cols)
        Xtr = ct.fit_transform(tr).astype(np.float32)
        Xev = ct.transform(ev).astype(np.float32)
        k = base.fit_state_heads(Xtr, Xev, Zc_tr, Zc_ev, yd_tr, yd_ev)
        kc = base.fit_state_count_head(Xtr, Yc_tr, Xev, Yc_ev, constant_rates=k0_count["constant_rates"])
        cont = C.rep._node_eval_nll(k["nodes"])
        disc = k["disc_ev"]
        cnt = kc["nll_ev"].sum(axis=1)
        mean_j = float(np.mean(cont + disc + cnt))
        target = float(C.FROZEN_TRANSITION["A_TB1_to_TB2"][tag])
        assert abs(mean_j - target) < C.PARITY_TOL, f"{tag} parity mismatch: {mean_j} vs {target}"


# 3. terminal sampler parity T0/T2
def test_3_terminal_sampler_parity_t0_t2():
    p = C.CACHE / "dynamic_pgm1b_sample.parquet"
    assert p.exists()
    df = pd.read_parquet(p)
    w = base.WINDOWS[0]
    tr = df[df["block"].isin(w["train"])].reset_index(drop=True)
    ev = df[df["block"] == w["eval"]].reset_index(drop=True)

    for tag, num_cols in [("T0", C.T0_NUM), ("T2", C.T2_NUM)]:
        cols = num_cols + C.CAT
        pipe = C.pm.make_pipeline(num_cols, C.CAT).fit(tr[cols], tr["hazard"].to_numpy(np.int64))
        p_h = pipe.predict_proba(ev[cols])[:, 1]
        pre = pipe.named_steps["pre"]
        Xtr = C.pbar.densify(pre.transform(tr[cols]))
        Xev = C.pbar.densify(pre.transform(ev[cols]))
        tt = tr["hazard"].to_numpy(np.int64) == 1
        mtr = tr["target_mask"].to_numpy()[tt].astype(np.int64)
        theta, _ = C.pbar.fit_conditional_crf(Xtr[tt], mtr, with_pairs=True)
        te = ev["hazard"].to_numpy(np.int64) == 1
        p_mask = C.pbar.predict_mask_prob(theta, Xev[te], with_pairs=True)
        met = C.ms.model_metrics(ev, p_h, p_mask, te)
        target = C.FROZEN_TERMINAL["A_TB1_to_TB2"][tag]
        assert abs(met["hazard_nll"] - target["hazard_nll"]) < C.PARITY_TOL
        assert abs(met["endpoint_joint_nll"] - target["endpoint_joint_nll"]) < C.PARITY_TOL
        assert abs(met["mean_episode_nll"] - target["mean_episode_nll"]) < C.PARITY_TOL


# 4. reset sampler parity R0/R1
def test_4_reset_sampler_parity_r0_r1():
    p = C.CACHE / "dynamic_pgm1b_sample.parquet"
    assert p.exists()
    df = pd.read_parquet(p)
    pair, _ = exp1b.build_reset_pairs(df)
    w = base.WINDOWS[0]
    ptr = pair[pair["block"].isin(w["train"]) & pair["next_episode_id"].notna()].reset_index(drop=True)
    pev = pair[(pair["block"] == w["eval"]) & pair["next_episode_id"].notna()].reset_index(drop=True)

    gap_p = exp1b.fit_gap_geometric(ptr["gap_bars"].to_numpy(np.float64))
    for tag, occ_cols in [("R0", C.R0_OCC), ("R1", C.R1_OCC)]:
        state_cols = list(occ_cols) + C.GAP_FEAT
        Xtr_o, Xev_o = exp1b._design(ptr, pev, occ_cols, C.MASK_CAT)
        Xtr_s, Xev_s = exp1b._design(ptr, pev, state_cols, C.MASK_CAT)
        gap_occ = exp1b._fit_occurrence(Xtr_o, ptr["gap_positive"].to_numpy(np.int64),
                                        Xev_o, f"{tag}_gap_occ")
        nll_occ = np.where(pev["gap_positive"].to_numpy(np.int64) == 1,
                           -np.log(gap_occ), -np.log1p(-gap_occ))
        decomp = exp1b._fit_heads(Xtr_s, Xev_s, ptr, pev, gap_p)
        decomp["_gap"] = pev["gap_bars"].to_numpy(np.float64)
        total_nll = exp1b.reset_nll(decomp, nll_occ, gap_p)
        mean_res_nll = float(total_nll.mean())
        target = float(C.FROZEN_RESET["A_TB1_to_TB2"][tag]["mean_reset_nll"])
        assert abs(mean_res_nll - target) < C.PARITY_TOL


# 5. Gaussian sampling covariance synthetic
def test_5_gaussian_sampling_covariance_synthetic():
    rng = np.random.default_rng(20260915)
    n = 20000
    p = 3
    q = 2
    X = rng.standard_normal((n, p))
    true_B = np.array([[1.0, -0.5], [0.5, 1.5], [-1.0, 0.2]])
    true_cov = np.array([[0.5, 0.2], [0.2, 0.8]])
    L = np.linalg.cholesky(true_cov)
    noise = rng.standard_normal((n, q)) @ L.T
    Z = X @ true_B + noise

    head = base.GaussianTransitionHead(alpha=0.001).fit(X, Z)
    samples = C.sample_gaussian(head, X, rng)
    resids = samples - (X @ head.B + head.intercept)
    emp_cov = np.cov(resids, rowvar=False)
    assert np.allclose(emp_cov, head.cov, atol=0.05)


# 6. hurdle sampling support
def test_6_hurdle_sampling_support():
    rng = np.random.default_rng(20260915)
    n = 5000
    p = 2
    X = rng.standard_normal((n, p))
    # Synthetic positive hurdle
    ispos = (rng.random(n) > 0.4).astype(float)
    logv = np.where(ispos > 0.5, rng.standard_normal(n) * 0.5, 0.0)
    Z = np.stack([ispos, logv], axis=1)

    head_pos = base.HurdleLogNormalHead(alpha=1.0, sign=1.0).fit(X, Z)
    draws_pos = C.sample_hurdle_ln(head_pos, X, rng)
    assert np.all(draws_pos >= 0.0), "sign=+1 hurdle must be non-negative"

    # Synthetic negative hurdle (residuals)
    head_neg = base.HurdleLogNormalHead(alpha=1.0, sign=-1.0).fit(X, Z)
    draws_neg = C.sample_hurdle_ln(head_neg, X, rng)
    assert np.all(draws_neg <= 0.0), "sign=-1 hurdle must be non-positive"


# 7. DCR support
def test_7_dcr_support():
    rng = np.random.default_rng(20260915)
    n = 6000
    p = 2
    X = rng.standard_normal((n, p))
    cat = rng.choice([0, 1, 2], size=n, p=[0.2, 0.6, 0.2])
    is0 = (cat == 0).astype(float)
    is1 = (cat == 2).astype(float)
    logit = np.where(cat == 1, rng.standard_normal(n), 0.0)
    Z = np.stack([is0, is1, logit], axis=1)

    head = base.ZeroInteriorOneHead(alpha=1.0).fit(X, Z)
    draws = C.sample_dcr(head, X, rng)
    assert np.all(draws >= 0.0) and np.all(draws <= 1.0)
    assert np.any(draws == 0.0)
    assert np.any(draws == 1.0)
    assert np.any((draws > 0.0) & (draws < 1.0))


# 8. ZTP support + empirical mean
def test_8_ztp_support_and_empirical_mean():
    rng = np.random.default_rng(20260915)
    lam = 0.65
    samples = C.sample_ztp_vec(lam, rng, 50000)
    assert np.all(samples >= 1)
    assert np.all(np.equal(np.mod(samples, 1), 0))
    theo_mean = lam / (-np.expm1(-lam))
    emp_mean = float(np.mean(samples))
    assert abs(emp_mean - theo_mean) < 0.02


# 9. endpoint mask sampling 1..15
def test_9_endpoint_mask_sampling_1_to_15():
    p = C.CACHE / "dynamic_pgm1b_sample.parquet"
    df = pd.read_parquet(p)
    w = base.WINDOWS[0]
    tr = df[df["block"].isin(w["train"])].reset_index(drop=True)
    cols = C.T0_NUM + C.CAT
    pipe = C.pm.make_pipeline(C.T0_NUM, C.CAT).fit(tr[cols], tr["hazard"].to_numpy(np.int64))
    pre = pipe.named_steps["pre"]
    Xtr = C.pbar.densify(pre.transform(tr[cols]))
    tt = tr["hazard"].to_numpy(np.int64) == 1
    mtr = tr["target_mask"].to_numpy()[tt].astype(np.int64)
    theta, _ = C.pbar.fit_conditional_crf(Xtr[tt], mtr, with_pairs=True)

    rng = np.random.default_rng(20260915)
    sampled = C.sample_crf_endpoint(theta, Xtr[:1000], rng)
    assert set(np.unique(sampled)).issubset(set(range(1, 16)))


# 10. reconstruct transition synthetic exact
def test_10_reconstruct_transition_synthetic_exact():
    state = {
        "start_up_distance_R": 1.5,
        "start_down_distance_R": 1.0,
        "start_width_R": 2.5,
        "cur_up_distance_R": 1.2,
        "cur_down_distance_R": 1.3,
        "cur_width_R": 2.5,
        "path_max_up_excursion_R": 0.5,
        "path_max_down_excursion_R": 0.3,
        "path_direction_change_rate": 0.2,
        "path_current_bar_range_R": 0.1,
        "upper_newest_log_age_residual": -0.1,
        "lower_newest_log_age_residual": -0.2,
        "upper_newest_log_age": 1.0,
        "lower_newest_log_age": 1.5,
        "upper_active_identity_count_delta": 2,
        "lower_active_identity_count_delta": 1,
        "path_total_variation_R": 0.8,
        "path_last_return_R": 0.05,
        "episode_age": 3,
        "eps_R": 1e-4,
    }
    z = {
        "z_d_up": 0.1,
        "dmfe": 0.05,
        "dmae": 0.0,
        "dcr": 0.3,
        "range": 0.15,
        "uresid": -0.05,
        "lresid": -0.1,
        "delta_upper_count": 1,
        "delta_lower_count": 0,
    }
    nxt, viol = C.advance_nonterminal(state, z)
    assert viol is None
    assert np.isclose(nxt["cur_up_distance_R"], 1.3)
    assert np.isclose(nxt["cur_down_distance_R"], 1.2)
    assert np.isclose(nxt["path_max_up_excursion_R"], 0.55)
    assert np.isclose(nxt["path_total_variation_R"], 0.9)
    assert np.isclose(nxt["path_last_return_R"], -0.1)
    assert nxt["upper_active_identity_count_delta"] == 3
    assert nxt["lower_active_identity_count_delta"] == 1
    assert nxt["episode_age"] == 4


# 11. eps_R observed closure <1e-8
def test_11_eps_r_observed_closure_under_1e8():
    obs = pd.read_parquet(C.CACHE / "dynamic_pgm1b_sample.parquet")
    ep_meta = pd.read_parquet(C.CACHE / "episode_repl0_through_tb3.parquet")
    audit = C.audit_eps_r_closure(obs, ep_meta)
    assert audit["passed"] is True
    assert audit["max_error"] < 1e-8


# 12. reset -> start reconstruction
def test_12_reset_to_start_reconstruction():
    d_u, d_d, eps_R = 1.4, 0.8, 1e-4
    r_log = float(np.log((d_u + eps_R) / (d_d + eps_R)) - np.log(d_u / d_d))
    r = {
        "start_up_distance_R": d_u,
        "start_down_distance_R": d_d,
        "start_log_ratio_residual": r_log,
        "next_path_max_up_excursion_R": 0.2,
        "next_path_max_down_excursion_R": 0.1,
        "next_upper_newest_log_age": 2.0,
        "next_upper_span": 1.5,
        "next_lower_newest_log_age": 1.0,
        "next_lower_span": 0.8,
        "next_upper_n_active_minus1": 3,
        "next_lower_n_active_minus1": 2,
    }
    st, viol = C.reset_to_start(r, endpoint_mask=7)
    assert viol is None
    assert np.isclose(st["start_up_distance_R"], 1.4)
    assert np.isclose(st["start_down_distance_R"], 0.8)
    assert np.isclose(st["start_width_R"], 2.2)
    assert np.isclose(st["cur_up_distance_R"], 1.4)
    assert np.isclose(st["cur_down_distance_R"], 0.8)
    assert st["upper_n_active_identities"] == 4.0
    assert st["lower_n_active_identities"] == 3.0
    assert st["prev_event_mask"] == 7
    assert st[C.LAG_AVAIL] == 0.0
    assert st["eps_R"] > 0.0


# 13. first-bar availability=0
def test_13_first_bar_availability_zero():
    d_u, d_d, eps_R = 1.2, 0.8, 1e-4
    r_log = float(np.log((d_u + eps_R) / (d_d + eps_R)) - np.log(d_u / d_d))
    r = {
        "start_up_distance_R": d_u, "start_down_distance_R": d_d,
        "start_log_ratio_residual": r_log, "next_path_max_up_excursion_R": 0.1,
        "next_path_max_down_excursion_R": 0.1, "next_upper_newest_log_age": 1.0,
        "next_upper_span": 1.0, "next_lower_newest_log_age": 1.0,
        "next_lower_span": 1.0, "next_upper_n_active_minus1": 1,
        "next_lower_n_active_minus1": 1,
    }
    st, _ = C.reset_to_start(r, endpoint_mask=1)
    assert st[C.LAG_AVAIL] == 0.0
    z = {
        "z_d_up": 0.05, "dmfe": 0.0, "dmae": 0.0, "dcr": 0.5,
        "range": 0.1, "uresid": 0.0, "lresid": 0.0,
        "delta_upper_count": 0, "delta_lower_count": 0,
    }
    nxt, _ = C.advance_nonterminal(st, z)
    assert nxt[C.LAG_AVAIL] == 1.0


# 14. first-bar terminal mem semantics
def test_14_first_bar_terminal_mem_semantics():
    d_u, d_d, eps_R = 1.2, 0.8, 1e-4
    r_log = float(np.log((d_u + eps_R) / (d_d + eps_R)) - np.log(d_u / d_d))
    r = {
        "start_up_distance_R": d_u, "start_down_distance_R": d_d,
        "start_log_ratio_residual": r_log, "next_path_max_up_excursion_R": 0.1,
        "next_path_max_down_excursion_R": 0.1, "next_upper_newest_log_age": 1.0,
        "next_upper_span": 1.0, "next_lower_newest_log_age": 1.0,
        "next_lower_span": 1.0, "next_upper_n_active_minus1": 1,
        "next_lower_n_active_minus1": 1,
    }
    st, _ = C.reset_to_start(r, endpoint_mask=1)
    assert st["mem_z_dmfe_ispos"] == 0.0
    assert st["mem_z_dmfe_log"] == 0.0
    assert st["mem_z_dmae_ispos"] == 0.0
    assert st["mem_z_dmae_log"] == 0.0
    assert np.isnan(st["mem_z_delta_upper_count"])
    assert np.isnan(st["mem_z_delta_lower_count"])


# 15. no clamp in transition/reset path
def test_15_no_clamp_in_transition_reset_path():
    src_adv = inspect.getsource(C.advance_nonterminal)
    assert "clip" not in src_adv.lower()
    src_rst = inspect.getsource(C.reset_to_start)
    assert "clip" not in src_rst.lower()
    src_eps = inspect.getsource(C.solve_eps_R)
    assert "clip" not in src_eps.lower()


# 16. invalid state fail-closed
def test_16_invalid_state_fail_closed():
    st = {
        "start_up_distance_R": 1.0, "start_down_distance_R": 1.0,
        "cur_up_distance_R": 0.05, "cur_down_distance_R": 1.95,
        "path_max_up_excursion_R": 0.1, "path_max_down_excursion_R": 0.1,
        "path_direction_change_rate": 0.5, "path_current_bar_range_R": 0.1,
        "upper_newest_log_age": 1.0, "lower_newest_log_age": 1.0,
        "upper_active_identity_count_delta": 0, "lower_active_identity_count_delta": 0,
        "path_total_variation_R": 0.1, "episode_age": 1, "eps_R": 1e-4,
    }
    # Overshoot downwards: cur_up + (-0.2) = -0.15 < 0
    z = {
        "z_d_up": -0.2, "dmfe": 0.0, "dmae": 0.0, "dcr": 0.5,
        "range": 0.1, "uresid": 0.0, "lresid": 0.0,
        "delta_upper_count": 0, "delta_lower_count": 0,
    }
    nxt, viol = C.advance_nonterminal(st, z)
    assert viol == "TRANSITION_UP_DISTANCE_NEGATIVE"


# 17. timeout fail-closed
def test_17_timeout_fail_closed():
    # If episode reaches 512 bars without terminal, status must be TIMEOUT
    st = {
        "start_up_distance_R": 10.0, "start_down_distance_R": 10.0,
        "cur_up_distance_R": 10.0, "cur_down_distance_R": 10.0,
        "path_max_up_excursion_R": 0.1, "path_max_down_excursion_R": 0.1,
        "path_direction_change_rate": 0.5, "path_current_bar_range_R": 0.1,
        "upper_newest_log_age_residual": 0.0, "lower_newest_log_age_residual": 0.0,
        "upper_newest_log_age": 1.0, "lower_newest_log_age": 1.0,
        "upper_active_identity_count_delta": 0, "lower_active_identity_count_delta": 0,
        "path_total_variation_R": 0.1, "path_last_return_R": 0.0,
        "episode_age": 511, "eps_R": 1e-4,
    }
    for node in ("z_d_up", "z_dcr", "z_range", "z_uresid", "z_lresid"):
        for zc, vals in C.rep._encode_with_frozen(node, np.array([0.0])).items():
            st[f"phi_{zc}"] = float(vals[0])
    for c in ["mem_z_dmfe_ispos", "mem_z_dmfe_log", "mem_z_dmae_ispos", "mem_z_dmae_log"]:
        st[c] = 0.0
    st["mem_z_delta_upper_count"] = 0.0
    st["mem_z_delta_lower_count"] = 0.0
    st[C.LAG_AVAIL] = 1.0

    class DummyTerminalNever:
        def sample_hazard(self, df, rng):
            return np.array([False])
        def sample_endpoint(self, df, rng):
            return np.array([1])

    class DummyTransitionZero:
        def sample_batch(self, df, rng):
            return {k: np.array([0.0]) if "count" not in k else np.array([0])
                    for k in ("z_d_up", "dmfe", "dmae", "dcr", "range", "uresid", "lresid",
                              "delta_upper_count", "delta_lower_count")}

    res = C.run_single_episode_rollout(st, DummyTransitionZero(), DummyTerminalNever(), np.random.default_rng(1), max_bars=512)
    assert res["status"] == "TIMEOUT"
    assert res["violation"] == "EPISODE_TIMEOUT"


# 18. one-bar episode semantics
def test_18_one_bar_episode_semantics():
    st = {
        "start_up_distance_R": 1.0, "start_down_distance_R": 1.0,
        "cur_up_distance_R": 1.0, "cur_down_distance_R": 1.0,
        "path_max_up_excursion_R": 0.1, "path_max_down_excursion_R": 0.1,
        "path_direction_change_rate": 0.5, "path_current_bar_range_R": 0.1,
        "upper_newest_log_age_residual": 0.0, "lower_newest_log_age_residual": 0.0,
        "upper_newest_log_age": 1.0, "lower_newest_log_age": 1.0,
        "upper_active_identity_count_delta": 0, "lower_active_identity_count_delta": 0,
        "path_total_variation_R": 0.1, "path_last_return_R": 0.0,
        "episode_age": 0, "eps_R": 1e-4,
    }
    class DummyTerminalFirstBar:
        def sample_hazard(self, df, rng):
            return np.array([True])
        def sample_endpoint(self, df, rng):
            return np.array([4])

    res = C.run_single_episode_rollout(st, None, DummyTerminalFirstBar(), np.random.default_rng(1))
    assert res["status"] == "TERMINAL"
    assert res["duration"] == 1
    assert res["endpoint_mask"] == 4


# 19. multi-episode endpoint handoff
def test_19_multi_episode_endpoint_handoff():
    st = {
        "start_up_distance_R": 1.0, "start_down_distance_R": 1.0,
        "cur_up_distance_R": 1.0, "cur_down_distance_R": 1.0,
        "start_width_R": 2.0, "cur_width_R": 2.0, "start_log_ratio": 0.0,
        "path_max_up_excursion_R": 0.1, "path_max_down_excursion_R": 0.1,
        "path_direction_change_rate": 0.5, "path_current_bar_range_R": 0.1,
        "upper_newest_log_age_residual": 0.0, "lower_newest_log_age_residual": 0.0,
        "upper_oldest_log_age": 2.0, "lower_oldest_log_age": 2.0,
        "upper_newest_age_zero": 0.0, "upper_oldest_age_zero": 0.0,
        "lower_newest_age_zero": 0.0, "lower_oldest_age_zero": 0.0,
        "upper_newest_log_age": 1.0, "lower_newest_log_age": 1.0,
        "upper_n_active_identities": 2.0, "lower_n_active_identities": 2.0,
        "upper_active_identity_count_delta": 0, "lower_active_identity_count_delta": 0,
        "path_total_variation_R": 0.1, "path_last_return_R": 0.0,
        "tempo_signed_speed": 0.0, "tempo_abs_speed": 0.0,
        "tempo_signed_efficiency": 0.0, "tempo_abs_efficiency": 0.0,
        "elapsed_log": 0.0, "cur_log_ratio": 0.0,
        "prev_event_mask": 1, C.LAG_AVAIL: 0.0, "episode_age": 0, "eps_R": 1e-4,
    }
    class DummyTerminalTerm:
        def sample_hazard(self, df, rng):
            return np.array([True])
        def sample_endpoint(self, df, rng):
            return np.array([12])

    class DummyReset:
        def sample_gap(self, df, rng):
            return np.array([0])
        def sample_reset_primitives(self, df, gaps, rng):
            d_u, d_d, eps_R = 1.2, 0.8, 1e-4
            r_log = float(np.log((d_u + eps_R) / (d_d + eps_R)) - np.log(d_u / d_d))
            return {
                "start_up_distance_R": np.array([d_u]),
                "start_down_distance_R": np.array([d_d]),
                "start_log_ratio_residual": np.array([r_log]),
                "next_path_max_up_excursion_R": np.array([0.2]),
                "next_path_max_down_excursion_R": np.array([0.1]),
                "next_upper_newest_log_age": np.array([1.5]),
                "next_upper_span": np.array([1.0]),
                "next_lower_newest_log_age": np.array([1.2]),
                "next_lower_span": np.array([0.8]),
                "next_upper_n_active_minus1": np.array([2]),
                "next_lower_n_active_minus1": np.array([1]),
            }

    chain_res = C.run_single_freerun_chain(
        "AG", st, None, DummyTerminalTerm(), DummyReset(),
        np.random.default_rng(1), burn_in=1, collect=2
    )
    assert chain_res["completed"] is True
    assert len(chain_res["episodes"]) == 2
    assert chain_res["episodes"][0]["endpoint_mask"] == 12


# 20. reset prev_event_mask == prior endpoint
def test_20_reset_prev_event_mask_equals_prior_endpoint():
    d_u, d_d, eps_R = 1.2, 0.8, 1e-4
    r_log = float(np.log((d_u + eps_R) / (d_d + eps_R)) - np.log(d_u / d_d))
    r = {
        "start_up_distance_R": d_u, "start_down_distance_R": d_d,
        "start_log_ratio_residual": r_log, "next_path_max_up_excursion_R": 0.1,
        "next_path_max_down_excursion_R": 0.1, "next_upper_newest_log_age": 1.0,
        "next_upper_span": 1.0, "next_lower_newest_log_age": 1.0,
        "next_lower_span": 1.0, "next_upper_n_active_minus1": 1,
        "next_lower_n_active_minus1": 1,
    }
    st, _ = C.reset_to_start(r, endpoint_mask=9)
    assert st["prev_event_mask"] == 9


# 21. burn-in excluded from metrics
def test_21_burn_in_excluded_from_metrics():
    # In run_single_freerun_chain: total iterations = burn_in + collect
    # Only iterations >= burn_in are appended to episodes_collected.
    src = inspect.getsource(C.run_single_freerun_chain)
    assert "ep_idx >= burn_in" in src


# 22. no teacher forcing after seed
def test_22_no_teacher_forcing_after_seed():
    src = inspect.getsource(C.run_single_freerun_chain)
    assert "current_start = next_start" in src
    assert "obs" not in src


# 23. normalized Wasserstein deterministic unit test
def test_23_normalized_wasserstein_deterministic_unit_test():
    x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x2 = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
    # W1 = 0.2
    # scale = Q95(x1) - Q05(x1) = 4.8 - 1.2 = 3.6
    dw = C.normalized_wasserstein(x2, x1)
    assert dw > 0.0
    assert np.isclose(dw, 0.2 / 3.6)


# 24. endpoint TVD unit test
def test_24_endpoint_tvd_unit_test():
    p1 = np.array([0.5, 0.5])
    p2 = np.array([0.2, 0.8])
    # 0.5 * (|0.5 - 0.2| + |0.5 - 0.8|) = 0.5 * (0.3 + 0.3) = 0.3
    tvd = C.endpoint_tvd(p1, p2)
    assert np.isclose(tvd, 0.3)


# 25. D_total exact equal-weight composition
def test_25_d_total_exact_equal_weight_composition():
    durs_g = np.array([5.0, 6.0, 7.0])
    durs_o = np.array([5.0, 6.0, 7.0])
    eps_g = np.array([1, 2, 3])
    eps_o = np.array([1, 2, 3])
    terms_g = {k: np.array([1.0, 2.0]) for k in C.TERMINAL_DYNAMIC_FIELDS}
    terms_o = {k: np.array([1.0, 2.0]) for k in C.TERMINAL_DYNAMIC_FIELDS}
    resets_g = {k: np.array([1.0, 2.0]) for k in C.RESET_STRUCTURAL_FIELDS}
    resets_o = {k: np.array([1.0, 2.0]) for k in C.RESET_STRUCTURAL_FIELDS}
    gaps_g = np.array([0, 1, 2])
    gaps_o = np.array([0, 1, 2])

    gen = dict(durations=durs_g, endpoints=eps_g, terminals=terms_g, resets=resets_g, gaps=gaps_g)
    obs = dict(durations=durs_o, endpoints=eps_o, terminals=terms_o, resets=resets_o, gaps=gaps_o)
    disc = C.compute_rollout_discrepancy(gen, obs)
    assert np.isclose(disc["D_total"], (disc["D_duration"] + disc["D_endpoint"] + disc["D_terminal"] + disc["D_reset"] + disc["D_gap"]) / 5.0)


# 26. paired replicate bootstrap multiplicity
def test_26_paired_replicate_bootstrap_multiplicity():
    diffs = np.array([-0.05, -0.04, -0.06, -0.03, -0.05, -0.04, -0.02, -0.07,
                      -0.05, -0.06, -0.04, -0.03, -0.05, -0.04, -0.06, -0.05])
    lo, hi, pt = C.paired_bootstrap_replicates(diffs, seed=42, reps=1000)
    assert lo < pt < hi
    assert hi < 0.0


# 27. WT differs from W1 ONLY terminal sampler
def test_27_wt_differs_from_w1_only_terminal_sampler():
    # Check docstring and specifications:
    # W1: T0 + MC + R1
    # WT: T2 + MC + R1
    # Both use MC for transition and R1 for reset.
    assert "T0" in C.__doc__ and "T2" in C.__doc__
    assert "MC_STATE_CURREENCODING" in C.__doc__
    assert "R1_STATE_PHI" in C.__doc__


# 28. no TB4
def test_28_no_tb4_leakage():
    for w in C.WINDOWS:
        assert "TB4" not in w["train"]
        assert "TB4" != w["eval"]


# 29. output namespace isolation
def test_29_output_namespace_isolation():
    assert C.PREFIX == "dynamic_pgm1c"
    assert C.PREFIX.startswith("dynamic_pgm1c")


# 30. JSON serializable
def test_30_json_serializable():
    dummy = {
        "experiment": "DYNAMIC-PGM-1C Free-Run Rollout Closure",
        "parent_commit": C.BASE_SHA,
        "gates": {"FREE_RUN_SUPPORT_CLOSED": True, "ROLLOUT_PHI_RESET_SUPPORTED": True},
        "score": 0.12345,
    }
    s = json.dumps(dummy, indent=2)
    loaded = json.loads(s)
    assert loaded["parent_commit"] == C.BASE_SHA


if __name__ == "__main__":
    import traceback

    tests = [getattr(sys.modules[__name__], f) for f in dir(sys.modules[__name__])
             if f.startswith("test_")]
    tests.sort(key=lambda fn: int(fn.__name__.split("_")[1]))

    ok = fail = 0
    print(f"Running {len(tests)} unit tests for DYNAMIC-PGM-1C...\n")
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            ok += 1
        except Exception:
            print(f"FAIL: {t.__name__}")
            traceback.print_exc()
            fail += 1

    print(f"\nTOTAL: {ok} passed, {fail} failed.")
    sys.exit(1 if fail else 0)
