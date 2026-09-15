"""
test_pgm_native0c_state_augmentation_v1.py
==========================================

Tests for PGM-NATIVE-0C: Incremental State Representation Audit.

Covers 35 required tests (see docstring list in the experiment module):
  1. start SHA correct / ancestor            19. TB3 uses Window B
  2. observation universe ALL rows           20. only four registered variants
  3. U primitives exist                      21. UE is sole PRIMARY
  4. E primitives exist                      22. Delta_LogLoss sign convention
  5. age == bar_t - start_bar                23. Delta_JNLL sign convention
  6. first row age == 0                      24. day bootstrap with replacement
  7. all U finite                            25. paired bootstrap same days
  8. all E finite                            26. bootstrap deterministic
  9. prefix invariance                       27. TB3 grid uses TB2 p_h edges
 10. no future column in U/E                 28. TB3 grid uses TB2 |m| edges
 11. no incremental name collision           29. true hazard never a predictor
 12. transition merge rowcount preserved     30. p_star diagnostic only
 13. transition merge unmatched zero         31. H1 harm label diagnostic only
 14. no hazard in augmented predictors       32. no augmented trading policy
 15. terminal wrapper baseline parity        33. no V2 / RL / Q dependency
 16. transition wrapper JNLL parity          34. smoke no verdict
 17. transition analytic z_d_up_mu parity    35. full blocked first round
 18. TB2 uses Window A
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

for _bt in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_bt] = "1"

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

import research.liquidity_oracle_atlas.experiment_pgm_native0c_state_augmentation_v1 as exp
import research.liquidity_oracle_atlas.experiment_pgm_native0a_one_step_alpha_v1 as n0a
import research.liquidity_oracle_atlas.experiment_pgm_native0b_hazard_reliability_v1 as n0b
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1c_free_run_rollout_v1 as pgm
import research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 as ex0


# ---------------------------------------------------------------------------
# Cached heavy fixtures
# ---------------------------------------------------------------------------
_FIT_A = None
_FIT_B = None
_OBS_AUG = None
_MERGED = None
_MERGED_DAY = None


def merged_with_day():
    global _MERGED_DAY
    if _MERGED_DAY is None:
        _, _, bars = ex0.load_env()
        obs_day = exp.attach_decision_day(obs_aug(), bars)
        _MERGED_DAY = exp.merge_incremental_features_into_transition(
            pd.read_parquet(pgm.TRANSITION_SAMPLE_PATH), obs_day)
    return _MERGED_DAY


def fit_A():
    global _FIT_A
    if _FIT_A is None:
        _FIT_A = pgm.fit_samplers_for_window(pgm.WINDOWS[0], pgm.SAMPLE_PATH,
                                             pgm.TRANSITION_SAMPLE_PATH)
    return _FIT_A


def obs_aug():
    global _OBS_AUG
    if _OBS_AUG is None:
        _OBS_AUG = exp.add_incremental_state_features(n0a.load_observed_decision_universe())
    return _OBS_AUG


def merged():
    global _MERGED
    if _MERGED is None:
        _MERGED = exp.merge_incremental_features_into_transition(
            pd.read_parquet(pgm.TRANSITION_SAMPLE_PATH), obs_aug())
    return _MERGED


def synth_episode(n=12, perturb_from=None, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(dict(
        symbol=["X"] * n, episode_id=["E1"] * n, block=["TB2"] * n,
        bar_t=np.arange(100, 100 + n), start_bar=[100] * n,
        cur_up_distance_R=rng.normal(0, 1, n),
        path_total_variation_R=np.abs(rng.normal(0.5, 0.2, n)),
        path_max_up_excursion_R=np.abs(rng.normal(0.3, 0.1, n)),
        path_max_down_excursion_R=np.abs(rng.normal(0.3, 0.1, n)),
        path_last_return_R=rng.normal(0, 0.5, n),
        path_current_bar_range_R=np.abs(rng.normal(0.4, 0.1, n)),
        path_direction_change_rate=rng.uniform(0, 1, n),
        hazard=[0] * n,
    ))
    if perturb_from is not None:
        cols = ["path_last_return_R", "path_current_bar_range_R",
                "path_direction_change_rate", "cur_up_distance_R",
                "path_total_variation_R", "path_max_up_excursion_R",
                "path_max_down_excursion_R"]
        df.loc[perturb_from:, cols] = df.loc[perturb_from:, cols] * 3.0 + 5.0
        df.loc[perturb_from:, "bar_t"] = df.loc[perturb_from:, "bar_t"] + 50
    return df


# ---------------------------------------------------------------------------
# 1. Start SHA / ancestor
# ---------------------------------------------------------------------------
def test_1_start_sha_correct_and_ancestor():
    assert exp.BASE_SHA == "fc3c34f27ff4efa061a268162af709a34d74d0f1"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   cwd=str(_REPO_ROOT), text=True).strip()
    res = subprocess.run(["git", "merge-base", "--is-ancestor", exp.BASE_SHA, "HEAD"],
                         cwd=str(_REPO_ROOT), capture_output=True)
    assert res.returncode == 0, f"BASE_SHA not ancestor; HEAD={head}"


# ---------------------------------------------------------------------------
# 2. Observation universe ALL rows
# ---------------------------------------------------------------------------
def test_2_observation_universe_all_rows():
    obs = n0a.load_observed_decision_universe()
    assert len(obs) == n0a.EXPECTED_ALL_OBS
    assert int((obs["hazard"] == 0).sum()) == n0a.EXPECTED_TRANSITIONS
    assert int((obs["hazard"] == 1).sum()) == n0a.EXPECTED_HAZARD1


# ---------------------------------------------------------------------------
# 3/4. Primitives exist
# ---------------------------------------------------------------------------
def test_3_u_primitives_exist():
    obs = n0a.load_observed_decision_universe()
    for c in exp.U_PRIMITIVES:
        assert c in obs.columns, f"missing U primitive {c}"


def test_4_e_primitives_exist():
    obs = n0a.load_observed_decision_universe()
    for c in exp.E_PRIMITIVES:
        assert c in obs.columns, f"missing E primitive {c}"


# ---------------------------------------------------------------------------
# 5/6. age semantics
# ---------------------------------------------------------------------------
def test_5_age_equals_bar_t_minus_start_bar():
    x = obs_aug()
    age = x["bar_t"].to_numpy(np.int64) - x["start_bar"].to_numpy(np.int64)
    assert np.all(age >= 0)


def test_6_first_row_age_zero():
    x = obs_aug()
    g = x.groupby(["symbol", "episode_id"], sort=False)
    first_age = g["bar_t"].first() - g["start_bar"].first()
    assert bool((first_age == 0).all())


# ---------------------------------------------------------------------------
# 7/8. finiteness
# ---------------------------------------------------------------------------
def test_7_all_u_finite():
    x = obs_aug()
    for c in exp.U_COLS:
        assert np.all(np.isfinite(x[c].to_numpy(dtype=np.float64))), f"non-finite {c}"


def test_8_all_e_finite():
    x = obs_aug()
    for c in exp.E_COLS:
        assert np.all(np.isfinite(x[c].to_numpy(dtype=np.float64))), f"non-finite {c}"


# ---------------------------------------------------------------------------
# 9. prefix invariance
# ---------------------------------------------------------------------------
def test_9_prefix_invariance():
    a = exp.add_incremental_state_features(synth_episode())
    b = exp.add_incremental_state_features(synth_episode(perturb_from=6))
    for c in exp.INCREMENTAL_COLS:
        da = a[c].to_numpy(dtype=np.float64)[:6]
        db = b[c].to_numpy(dtype=np.float64)[:6]
        assert np.array_equal(da, db), f"prefix dependence in {c}"


# ---------------------------------------------------------------------------
# 10. no future tokens in feature builder
# ---------------------------------------------------------------------------
def test_10_no_future_columns_in_builder():
    src = inspect.getsource(exp.add_incremental_state_features)
    for tok in exp.FUTURE_TOKENS:
        assert tok not in src, f"future token {tok!r} in feature builder"


# ---------------------------------------------------------------------------
# 11. no incremental / production column collision
# ---------------------------------------------------------------------------
def test_11_no_incremental_name_collision():
    fa = fit_A()
    t2 = set(fa["term_samplers"][exp.PRIMARY_TERMINAL_HEAD].design_cols)
    mc = set(fa["trans_samplers"][exp.PRIMARY_TRANSITION_HEAD].design_cols)
    assert len(set(exp.INCREMENTAL_COLS) & (t2 | mc)) == 0


# ---------------------------------------------------------------------------
# 12/13. transition merge contract
# ---------------------------------------------------------------------------
def test_12_transition_merge_rowcount_preserved():
    m = merged()
    assert len(m) == exp.EXPECTED_TRANSITION_ROWS


def test_13_transition_merge_unmatched_zero():
    m = merged()
    for c in exp.INCREMENTAL_COLS:
        assert int(m[c].isna().sum()) == 0, f"unmatched rows for {c}"


# ---------------------------------------------------------------------------
# 14. hazard never in augmented predictors
# ---------------------------------------------------------------------------
def test_14_no_hazard_in_augmented_predictors():
    assert "hazard" not in exp.INCREMENTAL_COLS
    assert "target_mask" not in exp.INCREMENTAL_COLS
    src = inspect.getsource(exp.fit_transition_variant)
    assert '"hazard"' not in src and "'hazard'" not in src


# ---------------------------------------------------------------------------
# 15. terminal wrapper baseline parity
# ---------------------------------------------------------------------------
def test_15_terminal_wrapper_baseline_parity():
    x = obs_aug()
    wA = pgm.WINDOWS[0]
    tr = x[x["block"].isin(wA["train"])].reset_index(drop=True)
    ev = x[x["block"] == wA["eval"]].reset_index(drop=True)
    wrap = exp.fit_terminal_hazard_variant(tr, ev, [])
    prod = n0b.predict_hazard_probability(
        fit_A()["term_samplers"][exp.PRIMARY_TERMINAL_HEAD], ev)
    d = float(np.max(np.abs(wrap["p_eval"] - prod)))
    assert d <= 1e-12, f"terminal baseline parity fail diff={d}"


# ---------------------------------------------------------------------------
# 16. transition wrapper mean-JNLL baseline parity
# ---------------------------------------------------------------------------
def test_16_transition_wrapper_mean_jnll_parity():
    m = merged()
    wA = pgm.WINDOWS[0]
    fa = fit_A()
    mc_cols = fa["trans_samplers"][exp.PRIMARY_TRANSITION_HEAD].design_cols
    tr = m[m["block"].isin(wA["train"])].reset_index(drop=True)
    ev = m[m["block"] == wA["eval"]].reset_index(drop=True)
    var = exp.fit_transition_variant(tr, ev, [], mc_cols, tag="T16")
    prod = fa["trans_parity"][exp.PRIMARY_TRANSITION_HEAD]["mean_joint_nll"]
    d = abs(var["mean_joint_nll"] - prod)
    assert d <= 1e-12, f"transition mean-JNLL parity fail diff={d}"


# ---------------------------------------------------------------------------
# 17. transition analytic z_d_up_mu parity
# ---------------------------------------------------------------------------
def test_17_transition_analytic_zdup_parity():
    m = merged()
    wA = pgm.WINDOWS[0]
    mc_cols = fit_A()["trans_samplers"][exp.PRIMARY_TRANSITION_HEAD].design_cols
    ev = m[m["block"] == wA["eval"]].reset_index(drop=True)
    tr = m[m["block"].isin(wA["train"])].reset_index(drop=True)
    var = exp.fit_transition_variant(tr, ev, [], mc_cols, tag="T17")
    mu_p = np.asarray(fit_A()["trans_samplers"][exp.PRIMARY_TRANSITION_HEAD]
                      .analytic_conditional_support(ev)["z_d_up_mu"], np.float64)
    mu_w = np.asarray(var["sampler"].analytic_conditional_support(ev)["z_d_up_mu"], np.float64)
    assert float(np.max(np.abs(mu_p - mu_w))) <= 1e-12


# ---------------------------------------------------------------------------
# 18/19. window routing
# ---------------------------------------------------------------------------
def test_18_tb2_uses_window_a():
    src = inspect.getsource(exp.build_baseline_scored_frame)
    assert "(TB2_BLOCK, fit_A)" in src


def test_19_tb3_uses_window_b():
    src = inspect.getsource(exp.build_baseline_scored_frame)
    assert "(TB3_BLOCK, fit_B)" in src


# ---------------------------------------------------------------------------
# 20/21. variants
# ---------------------------------------------------------------------------
def test_20_only_four_registered_variants():
    assert set(exp.VARIANTS.keys()) == {"PGM0", "PGM_U", "PGM_E", "PGM_UE"}


def test_21_ue_is_sole_primary():
    assert exp.PRIMARY_VARIANT == "PGM_UE"
    assert exp.VARIANTS["PGM_UE"] == exp.U_COLS + exp.E_COLS
    assert exp.VARIANTS["PGM0"] == []


# ---------------------------------------------------------------------------
# 22/23. delta sign conventions
# ---------------------------------------------------------------------------
def test_22_delta_logloss_sign_convention():
    day = np.array(["d1"] * 50 + ["d2"] * 50)
    base = np.full(100, 0.5)
    aug = np.full(100, 0.4)  # augmented better -> positive delta
    r = exp.fast_cluster_bootstrap_delta(day, base, aug, n_boot=100)
    assert r["point"] > 0 and r["p_pos"] == 1.0


def test_23_delta_jnll_sign_convention():
    day = np.array(["d1"] * 50 + ["d2"] * 50)
    base = np.full(100, 2.0)
    aug = np.full(100, 1.5)
    r = exp.fast_cluster_bootstrap_delta(day, base, aug, n_boot=100)
    assert r["point"] > 0
    r2 = exp.fast_cluster_bootstrap_delta(day, aug, base, n_boot=100)
    assert r2["point"] < 0


# ---------------------------------------------------------------------------
# 24/25/26. bootstrap
# ---------------------------------------------------------------------------
def test_24_day_bootstrap_with_replacement():
    src = inspect.getsource(exp.fast_cluster_bootstrap_delta)
    assert "rng.multinomial" in src
    assert "np.full(D, 1.0 / D)" in src


def test_25_paired_bootstrap_same_sampled_days():
    src = inspect.getsource(exp.fast_cluster_bootstrap_delta)
    assert "base_loss = (counts @ B) / denom" in src
    assert "aug_loss = (counts @ A) / denom" in src


def test_26_bootstrap_deterministic():
    day = np.repeat(np.arange(20), 10)
    rng = np.random.default_rng(1)
    base = rng.random(200) + 0.5
    aug = rng.random(200) + 0.4
    r1 = exp.fast_cluster_bootstrap_delta(day, base, aug, n_boot=100)
    r2 = exp.fast_cluster_bootstrap_delta(day, base, aug, n_boot=100)
    assert r1 == r2


# ---------------------------------------------------------------------------
# 27/28. mechanism grid frozen edges
# ---------------------------------------------------------------------------
def test_27_tb3_grid_uses_tb2_ph_edges():
    src = inspect.getsource(exp.build_baseline_mechanism_grid)
    assert 'p2 = df_tb2["p_h"]' in src          # edges frozen on TB2
    assert "np.quantile(p2" in src
    assert "p_edges[1:-1]" in src               # same edges reused for every block
    assert 'df_tb3["p_h"]' not in src.replace('p2 = df_tb2["p_h"]', "")


def test_28_tb3_grid_uses_tb2_abs_m_edges():
    src = inspect.getsource(exp.build_baseline_mechanism_grid)
    assert 'm2 = np.abs(df_tb2["score_mu"]' in src   # edges frozen on TB2
    assert "np.quantile(m2" in src
    assert "m_edges[1:-1]" in src                    # same edges reused


# ---------------------------------------------------------------------------
# 29. true hazard never enters model predictors
# ---------------------------------------------------------------------------
def test_29_true_hazard_never_enters_predictors():
    fa = fit_A()
    t2 = fa["term_samplers"][exp.PRIMARY_TERMINAL_HEAD].design_cols
    mc = fa["trans_samplers"][exp.PRIMARY_TRANSITION_HEAD].design_cols
    assert "hazard" not in t2 and "hazard" not in mc
    assert "target_mask" not in mc


# ---------------------------------------------------------------------------
# 30. p_star diagnostic only
# ---------------------------------------------------------------------------
def test_30_p_star_diagnostic_only():
    src = inspect.getsource(exp.build_baseline_mechanism_grid)
    assert "EX_POST_MECHANISM_DIAGNOSTIC_ONLY" in src
    assert "never enters policy" in src


# ---------------------------------------------------------------------------
# 31. H1 harm labels diagnostic only
# ---------------------------------------------------------------------------
def test_31_h1_harm_label_diagnostic_only():
    src = inspect.getsource(exp.compute_h1_harm_diagnostics)
    assert "no model is trained" in src
    for tok in ["position", "ret_0c", "policy", "threshold"]:
        assert tok not in src, f"forbidden token {tok!r} in H1 harm diagnostic"


# ---------------------------------------------------------------------------
# 32. no augmented trading policy
# ---------------------------------------------------------------------------
def test_32_no_augmented_trading_policy():
    text = Path(exp.__file__).read_text()
    for tok in ["position_0c", "ret_0c", "action_0c", "EV_0c"]:
        assert tok not in text, f"augmented trading policy token {tok!r} present"


# ---------------------------------------------------------------------------
# 33. no V2 / RL / Q dependency
# ---------------------------------------------------------------------------
def test_33_no_v2_rl_q_dependency():
    text = Path(exp.__file__).read_text().lower()
    for tok in ["market_regime_v2", "q_learning", "reinforcement_learning",
                "stable_baselines", "torch", "tensorflow"]:
        assert tok not in text, f"forbidden dependency {tok!r}"


# ---------------------------------------------------------------------------
# 34. smoke emits no verdict
# ---------------------------------------------------------------------------
def test_34_smoke_no_verdict():
    src = inspect.getsource(exp.run_smoke_test)
    assert "NO SCIENTIFIC VERDICT" in src
    assert "determine_state_augmentation_verdict" not in src


# ---------------------------------------------------------------------------
# 35. full blocked first round (even with env var set)
# ---------------------------------------------------------------------------
def test_35_full_blocked_without_authorization():
    old = os.environ.pop("AUTHORIZE_PGM_NATIVE0C_FULL_EXPLORATORY", None)
    try:
        raised = None
        try:
            exp.run_full_exploratory()
        except SystemExit as e:
            raised = str(e)
        assert raised is not None
        assert "STOP_PGM_NATIVE0C_FULL_EXPLORATORY_NOT_AUTHORIZED" in raised
    finally:
        if old is not None:
            os.environ["AUTHORIZE_PGM_NATIVE0C_FULL_EXPLORATORY"] = old


# ---------------------------------------------------------------------------
# 36. Conditional bootstrap CI must be non-degenerate
# ---------------------------------------------------------------------------
_PE = np.array([-np.inf, 0.08, 0.12, 0.18, 0.25, np.inf])
_ME = np.array([-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf])


def _synth_contrast_frame(n_days=30, per_day=200, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        p_h = rng.uniform(0.02, 0.35, per_day)
        m = rng.uniform(0.0, 1.0, per_day)
        hz = (rng.random(per_day) < (0.05 + 0.5 * p_h)).astype(np.int64)
        pi = np.where(hz == 1,
                      rng.normal(-0.30, 0.20, per_day),
                      rng.normal(0.05, 0.20, per_day))
        rows.append(pd.DataFrame(dict(entry_day=np.int64(d), p_h=p_h,
                                      score_mu=m, hazard=hz, pi=pi)))
    return pd.concat(rows, ignore_index=True)


def test_36_conditional_bootstrap_ci_non_degenerate():
    df = _synth_contrast_frame()
    r = exp.conditional_hazard_contrast(df, _PE, _ME, n_boot=400, seed=20260915)
    for k in ["Delta_H1_cond", "Delta_EV_cond"]:
        s = r[k]
        assert s["ci95_upper"] > s["ci95_lower"], f"{k} CI degenerate: {s}"
        assert s["ci95_upper"] != s["ci95_lower"]


# ---------------------------------------------------------------------------
# 37. Conditional bootstrap matches an INDEPENDENT brute-force reference
# ---------------------------------------------------------------------------
def _ref_conditional_bruteforce(df, p_edges, m_edges, n_boot, seed):
    """Independent slow reference: explicit day-resampled row reconstruction.

    Deliberately does NOT use the production matrix algebra -- it rebuilds the
    resampled rows and recomputes every rate with pandas grouping.
    """
    p_b = np.digitize(df["p_h"].to_numpy(np.float64), np.asarray(p_edges)[1:-1])
    m_b = np.digitize(np.abs(df["score_mu"].to_numpy(np.float64)), np.asarray(m_edges)[1:-1])
    keep = (p_b == 0) | (p_b == 4)
    d = df.loc[keep].copy()
    d["hb"] = p_b[keep]
    d["cb"] = m_b[keep]

    day_arr = d["entry_day"].to_numpy()
    days = np.unique(day_arr)
    D = len(days)
    didx = {dd: i for i, dd in enumerate(days)}
    dpos = np.array([didx[x] for x in day_arr])
    order = np.argsort(dpos, kind="stable")
    dpos_s = dpos[order]
    starts = np.searchsorted(dpos_s, np.arange(D), side="left")
    ends = np.searchsorted(dpos_s, np.arange(D), side="right")

    rng = np.random.default_rng(seed)
    counts = rng.multinomial(D, np.full(D, 1.0 / D), size=n_boot)

    bh, be = [], []
    for b in range(n_boot):
        parts = []
        for j in range(D):
            cj = int(counts[b, j])
            if cj:
                seg = order[starts[j]:ends[j]]
                parts.append(np.tile(seg, cj))
        idx = np.concatenate(parts)
        sub = d.iloc[idx]
        dh, de = [], []
        for q in range(5):
            top = sub[(sub["cb"] == q) & (sub["hb"] == 4)]
            bot = sub[(sub["cb"] == q) & (sub["hb"] == 0)]
            dh.append(float(top["hazard"].mean()) - float(bot["hazard"].mean()))
            de.append(float(top["pi"].mean()) - float(bot["pi"].mean()))
        bh.append(float(np.mean(dh)))
        be.append(float(np.mean(de)))
    bh = np.asarray(bh)
    be = np.asarray(be)
    return dict(
        h1=dict(ci_lo=float(np.percentile(bh, 2.5)), ci_hi=float(np.percentile(bh, 97.5)),
                p_pos=float(np.mean(bh > 0))),
        ev=dict(ci_lo=float(np.percentile(be, 2.5)), ci_hi=float(np.percentile(be, 97.5)),
                p_pos=float(np.mean(be > 0)), p_neg=float(np.mean(be < 0))),
    )


def test_37_conditional_bootstrap_matches_independent_reference():
    df = _synth_contrast_frame(n_days=8, per_day=60, seed=11)
    nb, sd = 60, 20260915
    got = exp.conditional_hazard_contrast(df, _PE, _ME, n_boot=nb, seed=sd)
    ref = _ref_conditional_bruteforce(df, _PE, _ME, n_boot=nb, seed=sd)
    assert np.isclose(got["Delta_H1_cond"]["ci95_lower"], ref["h1"]["ci_lo"], atol=1e-12, rtol=0)
    assert np.isclose(got["Delta_H1_cond"]["ci95_upper"], ref["h1"]["ci_hi"], atol=1e-12, rtol=0)
    assert np.isclose(got["Delta_H1_cond"]["p_pos"], ref["h1"]["p_pos"], atol=1e-12, rtol=0)
    assert np.isclose(got["Delta_EV_cond"]["ci95_lower"], ref["ev"]["ci_lo"], atol=1e-12, rtol=0)
    assert np.isclose(got["Delta_EV_cond"]["ci95_upper"], ref["ev"]["ci_hi"], atol=1e-12, rtol=0)
    assert np.isclose(got["Delta_EV_cond"]["p_neg"], ref["ev"]["p_neg"], atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# 38. Cluster owner is entry_day
# ---------------------------------------------------------------------------
def test_38_conditional_bootstrap_uses_entry_day():
    src = inspect.getsource(exp.conditional_hazard_contrast)
    assert 'df_tb3["entry_day"]' in src
    assert "episode_start_day" not in src


# ---------------------------------------------------------------------------
# 39. Same sampled days drive H1 / EV and top / bottom
# ---------------------------------------------------------------------------
def test_39_same_sampled_days_for_h1_ev_top_bottom():
    src = inspect.getsource(exp.conditional_hazard_contrast)
    assert "Nt_b = counts @ top_n" in src
    assert "Nb_b = counts @ bot_n" in src
    assert "Ht_b = counts @ top_h1" in src
    assert "Pt_b = counts @ top_pi" in src


# ---------------------------------------------------------------------------
# 40. Real transition seq join passes state parity
# ---------------------------------------------------------------------------
def test_40_transition_seq_join_real_state_parity():
    m = merged()
    jp = m.attrs.get("join_parity")
    assert jp is not None and jp.get("passed") is True
    assert len(jp["parity_columns"]) >= exp.MIN_JOIN_PARITY_COLS
    assert jp["overall_max_abs_diff"] < 1e-3


# ---------------------------------------------------------------------------
# 41. Adversarial within-episode permutation must trip the parity STOP
# ---------------------------------------------------------------------------
def test_41_adversarial_within_episode_permutation_trips_stop():
    trans = pd.read_parquet(pgm.TRANSITION_SAMPLE_PATH)
    feat = obs_aug()
    cols = [c for c in exp.JOIN_PARITY_CORE_COLS if c in trans.columns and c in feat.columns]
    cnt = trans.groupby(["symbol", "episode_id"], sort=False).size()
    sym, ep = cnt[cnt >= 3].index[0]
    idx = trans.index[(trans["symbol"] == sym) & (trans["episode_id"] == ep)].to_numpy()

    bad = trans.copy()
    v0 = bad.loc[idx, cols[0]].to_numpy(dtype=np.float64)
    i0 = int(idx[0])
    i1 = int(idx[int(np.argmax(np.abs(v0 - v0[0])))])  # pair with the most different payload
    for c in cols:
        a, b = bad.at[i0, c], bad.at[i1, c]
        bad.at[i0, c] = b
        bad.at[i1, c] = a

    raised = None
    try:
        exp.merge_incremental_features_into_transition(bad, feat)
    except SystemExit as e:
        raised = str(e)
    assert raised is not None, "adversarial permutation was NOT caught"
    assert "STOP_PGM_NATIVE0C_TRANSITION_JOIN_STATE_PARITY_FAIL" in raised


# ---------------------------------------------------------------------------
# 42. H1 harm diagnostic no longer re-quantiles internally
# ---------------------------------------------------------------------------
def test_42_h1_harm_has_no_internal_quantile():
    src = inspect.getsource(exp.compute_h1_harm_diagnostics)
    assert "np.quantile" not in src
    sig = inspect.signature(exp.compute_h1_harm_diagnostics)
    assert "p_edges" in sig.parameters and "m_edges" in sig.parameters


# ---------------------------------------------------------------------------
# 43. TB2 / TB3 H1 harm share the same frozen edges
# ---------------------------------------------------------------------------
def test_43_h1_harm_uses_same_frozen_edges_for_both_blocks():
    df = _synth_contrast_frame(n_days=6, per_day=80, seed=5)
    df["bar_t"] = np.arange(len(df))
    df["start_bar"] = 0
    a = exp.compute_h1_harm_diagnostics(df, "TB2", _PE, _ME)
    b = exp.compute_h1_harm_diagnostics(df, "TB3", _PE, _ME)
    # identical edges -> identical bin membership, only the block label differs
    assert [r["bin"] for r in a["by_hazard_quintile"]] == \
           [r["bin"] for r in b["by_hazard_quintile"]]
    src = inspect.getsource(exp.run_smoke_test)
    assert "compute_h1_harm_diagnostics(g3, TB3_BLOCK, p_edges, m_edges)" in src


# ---------------------------------------------------------------------------
# 44. Age Spearman uses real numeric bucket values
# ---------------------------------------------------------------------------
def test_44_age_spearman_uses_real_bucket_values():
    src = inspect.getsource(exp.compute_age_hazard_curve)
    assert "age_numeric" in src
    assert "np.arange(len(rows))" not in src


# ---------------------------------------------------------------------------
# 45-47. Window B baseline parity
# ---------------------------------------------------------------------------
def fit_B():
    global _FIT_B
    if _FIT_B is None:
        _FIT_B = pgm.fit_samplers_for_window(pgm.WINDOWS[1], pgm.SAMPLE_PATH,
                                             pgm.TRANSITION_SAMPLE_PATH)
    return _FIT_B


def test_45_window_b_terminal_baseline_parity():
    x = obs_aug()
    wB = pgm.WINDOWS[1]
    tr = x[x["block"].isin(wB["train"])].reset_index(drop=True)
    ev = x[x["block"] == wB["eval"]].reset_index(drop=True)
    wrap = exp.fit_terminal_hazard_variant(tr, ev, [])
    prod = n0b.predict_hazard_probability(
        fit_B()["term_samplers"][exp.PRIMARY_TERMINAL_HEAD], ev)
    d = float(np.max(np.abs(wrap["p_eval"] - prod)))
    assert d <= 1e-12, f"Window B terminal parity fail diff={d}"


def test_46_window_b_transition_mean_jnll_parity():
    m = merged()
    wB = pgm.WINDOWS[1]
    fb = fit_B()
    mc_cols = fb["trans_samplers"][exp.PRIMARY_TRANSITION_HEAD].design_cols
    tr = m[m["block"].isin(wB["train"])].reset_index(drop=True)
    ev = m[m["block"] == wB["eval"]].reset_index(drop=True)
    var = exp.fit_transition_variant(tr, ev, [], mc_cols, tag="T46")
    prod = fb["trans_parity"][exp.PRIMARY_TRANSITION_HEAD]["mean_joint_nll"]
    d = abs(var["mean_joint_nll"] - prod)
    assert d <= 1e-12, f"Window B mean-JNLL parity fail diff={d}"


def test_47_window_b_zdup_parity():
    m = merged()
    wB = pgm.WINDOWS[1]
    fb = fit_B()
    mc_cols = fb["trans_samplers"][exp.PRIMARY_TRANSITION_HEAD].design_cols
    tr = m[m["block"].isin(wB["train"])].reset_index(drop=True)
    ev = m[m["block"] == wB["eval"]].reset_index(drop=True)
    var = exp.fit_transition_variant(tr, ev, [], mc_cols, tag="T47")
    mu_p = np.asarray(fb["trans_samplers"][exp.PRIMARY_TRANSITION_HEAD]
                      .analytic_conditional_support(ev)["z_d_up_mu"], np.float64)
    mu_w = np.asarray(var["sampler"].analytic_conditional_support(ev)["z_d_up_mu"], np.float64)
    d = float(np.max(np.abs(mu_p - mu_w)))
    assert d <= 1e-12, f"Window B z_d_up_mu parity fail diff={d}"


# ---------------------------------------------------------------------------
# 48/49. Authorization gate
# ---------------------------------------------------------------------------
def test_48_authorization_gate_blocks_without_env():
    os.environ.pop("AUTHORIZE_PGM_NATIVE0C_FULL_EXPLORATORY", None)
    raised = None
    try:
        exp.require_full_authorization()
    except SystemExit as e:
        raised = str(e)
    assert raised is not None
    assert "STOP_PGM_NATIVE0C_FULL_EXPLORATORY_NOT_AUTHORIZED" in raised


def test_49_authorization_helper_env_one_passes():
    old = os.environ.get("AUTHORIZE_PGM_NATIVE0C_FULL_EXPLORATORY")
    os.environ["AUTHORIZE_PGM_NATIVE0C_FULL_EXPLORATORY"] = "1"
    try:
        exp.require_full_authorization()  # must not raise
    finally:
        if old is None:
            os.environ.pop("AUTHORIZE_PGM_NATIVE0C_FULL_EXPLORATORY", None)
        else:
            os.environ["AUTHORIZE_PGM_NATIVE0C_FULL_EXPLORATORY"] = old


# ---------------------------------------------------------------------------
# 50. decision_day exact owner
# ---------------------------------------------------------------------------
def test_50_decision_day_exact_owner():
    days = np.array(["2026-01-01", "2026-01-02", "2026-01-03"], dtype="datetime64[us]")
    bars = {"X": {"day": days, "n": 3}}
    obs = pd.DataFrame({"symbol": ["X"] * 3, "bar_t": [0, 1, 2]})
    out = exp.attach_decision_day(obs, bars)
    assert out["decision_day"].to_numpy()[1] == bars["X"]["day"][1]
    assert out["decision_day"].to_numpy()[2] == bars["X"]["day"][2]


# ---------------------------------------------------------------------------
# 51/52. decision_day never a predictor
# ---------------------------------------------------------------------------
def test_51_decision_day_not_in_terminal_predictors():
    assert "decision_day" not in pgm.T2_NUM and "decision_day" not in pgm.CAT
    src = inspect.getsource(exp.fit_terminal_hazard_variant)
    assert "decision_day" not in src


def test_52_decision_day_not_in_transition_predictors():
    mc = fit_A()["trans_samplers"][exp.PRIMARY_TRANSITION_HEAD].design_cols
    assert "decision_day" not in mc
    assert "decision_day" not in exp.INCREMENTAL_COLS
    src = inspect.getsource(exp.fit_transition_variant)
    assert "decision_day" not in src


# ---------------------------------------------------------------------------
# 53. Transition merge preserves decision_day
# ---------------------------------------------------------------------------
def test_53_transition_merge_preserves_decision_day():
    m = merged_with_day()
    assert "decision_day" in m.columns
    assert len(m) == exp.EXPECTED_TRANSITION_ROWS
    assert m["decision_day"].notna().all()


# ---------------------------------------------------------------------------
# 54/55. Registered variants / primary
# ---------------------------------------------------------------------------
def test_54_full_registered_variants_exactly_four():
    assert set(exp.VARIANTS.keys()) == {"PGM0", "PGM_U", "PGM_E", "PGM_UE"}
    src = inspect.getsource(exp.execute_full_pipeline)
    assert "for v, extra in VARIANTS.items()" in src


def test_55_pgm_ue_sole_primary_in_full():
    assert exp.PRIMARY_VARIANT == "PGM_UE"
    src = inspect.getsource(exp.execute_full_pipeline)
    assert 'r["variant"] == "PGM_UE"' in src          # primary verdict reads PGM_UE only
    assert 'for v in ["PGM_U", "PGM_E", "PGM_UE"]' in src  # only augmentations are bootstrapped


# ---------------------------------------------------------------------------
# 56/57. Full routing
# ---------------------------------------------------------------------------
def test_56_full_terminal_routing():
    src = inspect.getsource(exp.execute_full_pipeline)
    assert "windows = [(TB2_BLOCK, pgm.WINDOWS[0], fit_A), (TB3_BLOCK, pgm.WINDOWS[1], fit_B)]" in src


def test_57_full_transition_routing():
    src = inspect.getsource(exp.execute_full_pipeline)
    assert 'tr_t = merged_trans[merged_trans["block"].isin(w["train"])]' in src
    assert 'ev_t = merged_trans[merged_trans["block"] == w["eval"]]' in src


# ---------------------------------------------------------------------------
# 58/59. Bootstrap cluster owners
# ---------------------------------------------------------------------------
def test_58_terminal_bootstrap_cluster_owner_decision_day():
    src = inspect.getsource(exp.execute_full_pipeline)
    assert 'day_o = ev_o["decision_day"].to_numpy()' in src
    assert "episode_start_day" not in src


def test_59_transition_bootstrap_cluster_owner_decision_day():
    src = inspect.getsource(exp.execute_full_pipeline)
    assert 'day_t = ev_t["decision_day"].to_numpy()' in src


# ---------------------------------------------------------------------------
# 60. Primary verdict consumes only TB3 UE Delta_LogLoss + Delta_JNLL
# ---------------------------------------------------------------------------
def test_60_primary_verdict_wiring():
    src = inspect.getsource(exp.execute_full_pipeline)
    assert "determine_state_augmentation_verdict(ll_boot, jnll_boot)" in src
    assert 'r["metric"] == "Delta_LogLoss"' in src
    assert "TB3_BLOCK and r[\"variant\"] == \"PGM_UE\"" in src


# ---------------------------------------------------------------------------
# 61/62. Mechanism baseline-only + frozen edges
# ---------------------------------------------------------------------------
def test_61_mechanism_grid_uses_baseline_models_only():
    src = inspect.getsource(exp.execute_full_pipeline)
    assert "build_baseline_scored_frame(aligned_econ, fit_A, fit_B)" in src


def test_62_full_conditional_contrast_uses_tb2_frozen_edges():
    src = inspect.getsource(exp.execute_full_pipeline)
    assert "build_baseline_mechanism_grid(s2, s3)" in src
    assert "conditional_hazard_contrast(s3, p_edges, m_edges" in src


# ---------------------------------------------------------------------------
# 63. Age hazard uses FULL observation eval rows
# ---------------------------------------------------------------------------
def test_63_age_hazard_uses_full_observation_rows():
    src = inspect.getsource(exp.execute_full_pipeline)
    assert 'eval_obs_by_block[block][["block", "bar_t", "start_bar", "hazard"]]' in src


# ---------------------------------------------------------------------------
# 64. H1 harm uses frozen TB2 edges
# ---------------------------------------------------------------------------
def test_64_h1_harm_uses_frozen_tb2_edges():
    src = inspect.getsource(exp.execute_full_pipeline)
    assert "compute_h1_harm_diagnostics(s2, TB2_BLOCK, p_edges, m_edges)" in src
    assert "compute_h1_harm_diagnostics(s3, TB3_BLOCK, p_edges, m_edges)" in src


# ---------------------------------------------------------------------------
# 65. Artifact file set is exactly 9 planned files
# ---------------------------------------------------------------------------
def test_65_artifact_file_set_exactly_nine():
    assert len(exp.ARTIFACT_FILES) == 9
    assert len(set(exp.ARTIFACT_FILES)) == 9
    for fn in exp.ARTIFACT_FILES:
        assert fn.startswith(exp.PREFIX)
    assert f"{exp.PREFIX}_formal_summary.json" in exp.ARTIFACT_FILES


# ---------------------------------------------------------------------------
# 66. Formal summary required keys
# ---------------------------------------------------------------------------
def test_66_formal_summary_required_keys():
    src = (inspect.getsource(exp.execute_full_pipeline)
           + inspect.getsource(exp.run_full_exploratory))
    for k in ["EXPERIMENT_NAME", "EXPERIMENT_SCOPE", "base_sha", "run_head",
              "sample_artifact_sha256", "transition_artifact_sha256",
              "max_abs_atr0_owner_error", "n_all_obs", "n_H0", "n_H1", "symbols",
              "U_COLS", "E_COLS", "VARIANTS", "PRIMARY_VARIANT", "join_parity",
              "WindowA_baseline_parity", "WindowB_baseline_parity",
              "primary", "mechanism", "age", "formal_verdict", "known_scope_limitations"]:
        assert k in src, f"summary key {k} missing"
    for lim in ["previously inspected TB3", "cross-block episodes excluded",
                "event_mask==0 censored episodes excluded", "frozen PGM-bar sample",
                "no trading-policy optimization", "economic mechanism diagnostics are ex-post",
                "augmentation uses transforms of existing state primitives"]:
        assert lim in src, f"scope limitation {lim!r} missing"


# ---------------------------------------------------------------------------
# 67. Output parity checker catches a mutated artifact
# ---------------------------------------------------------------------------
def _synthetic_bundle():
    """A fully self-consistent synthetic 9-artifact bundle for parity-gate tests."""
    variants = ["PGM0", "PGM_U", "PGM_E", "PGM_UE"]
    terminal_rows, tboot, trans_rows, jboot = [], [], [], []
    for blk in ["TB2", "TB3"]:
        b_ll, b_br, b_j = 0.50, 0.25, 1.00
        for v in variants:
            ll = b_ll if v == "PGM0" else b_ll - 0.01
            br = b_br if v == "PGM0" else b_br - 0.005
            terminal_rows.append(dict(block=blk, variant=v, n=10, log_loss=ll, brier=br,
                                      roc_auc=0.60, pr_auc=0.20,
                                      Delta_LogLoss_vs_PGM0=b_ll - ll,
                                      Delta_Brier_vs_PGM0=b_br - br))
        for v in ["PGM_U", "PGM_E", "PGM_UE"]:
            for metric in ["Delta_LogLoss", "Delta_Brier"]:
                tboot.append(dict(block=blk, variant=v, metric=metric, point=0.01,
                                  ci95_lower=0.002, ci95_upper=0.02, p_pos=0.90))
        for v in variants:
            mj = b_j if v == "PGM0" else b_j - 0.02
            trans_rows.append(dict(block=blk, variant=v, n=10, mean_joint_nll=mj,
                                   rho_zdup=0.10, Delta_JNLL_vs_PGM0=b_j - mj))
        for v in ["PGM_U", "PGM_E", "PGM_UE"]:
            jboot.append(dict(block=blk, variant=v, metric="Delta_JNLL", point=0.02,
                              ci95_lower=0.005, ci95_upper=0.03, p_pos=0.95))

    grid = [dict(block=blk, hazard_bin=hb, conviction_bin=cb, n=5, mean_p_h=0.1,
                 mean_abs_m=0.2, observed_H1_rate=0.05, mu0=0.1, mu1=-0.2, EV=0.0,
                 H1_loss_rate=0.5, H1_mean_pi=-0.2, p_star=float("nan"))
            for blk in ["TB2", "TB3"] for hb in range(5) for cb in range(5)]
    contrast = dict(
        Delta_H1_cond=dict(point=0.01, ci95_lower=0.0, ci95_upper=0.02, p_pos=0.90),
        Delta_EV_cond=dict(point=-0.01, ci95_lower=-0.02, ci95_upper=0.0, p_pos=0.30,
                           p_neg=0.70),
        per_conviction=[dict(conviction_bin=q, n_top=5.0, n_bottom=5.0, H1_top=0.10,
                             H1_bottom=0.05, Delta_H1=0.05, EV_top=0.01, EV_bottom=0.02,
                             Delta_EV=-0.01) for q in range(5)],
        n_days=5, n_boot=10, n_invalid_replicates=0, cluster_owner="entry_day")
    age_rows = []
    for blk in ["TB2", "TB3"]:
        age_rows.append(dict(block=blk, age_bucket="0", age_numeric=0.0, n=5, H1_rate=0.1,
                             mean_p_h=0.1, spearman_age_vs_H1=0.5))
        age_rows.append(dict(block=blk, age_bucket="1", age_numeric=1.0, n=5, H1_rate=0.1,
                             mean_p_h=0.1, spearman_age_vs_H1=0.5))
    harm_rows = []
    for blk in ["TB2", "TB3"]:
        for b in range(5):
            harm_rows.append(dict(block=blk, group="hazard_quintile", bin=b, n=3,
                                  harm_rate=0.5, mean_harm=0.1, mean_pi=-0.1))
            harm_rows.append(dict(block=blk, group="conviction_quintile", bin=b, n=3,
                                  harm_rate=0.5, mean_harm=0.1, mean_pi=-0.1))
        harm_rows.append(dict(block=blk, group="age_bucket", bin=0, n=3, harm_rate=0.5,
                              mean_harm=0.1, mean_pi=-0.1))

    ll_boot = next(r for r in tboot if r["block"] == "TB3" and r["variant"] == "PGM_UE"
                   and r["metric"] == "Delta_LogLoss")
    jnll_boot = next(r for r in jboot if r["block"] == "TB3" and r["variant"] == "PGM_UE")
    summary = dict(
        EXPERIMENT_NAME=exp.EXPERIMENT_NAME, EXPERIMENT_SCOPE=exp.EXPERIMENT_SCOPE,
        base_sha=exp.BASE_SHA, run_head="deadbeef",
        sample_artifact_sha256="a" * 64, transition_artifact_sha256="b" * 64,
        n_all_obs=100, n_H0=90, n_H1=10, symbols=["AG"],
        max_abs_atr0_owner_error=0.0,
        U_COLS=list(exp.U_COLS), E_COLS=list(exp.E_COLS),
        VARIANTS={k: list(v) for k, v in exp.VARIANTS.items()},
        PRIMARY_VARIANT=exp.PRIMARY_VARIANT,
        join_parity=dict(parity_columns=["x"], per_column_max_abs_diff={"x": 0.0},
                         overall_max_abs_diff=0.0, passed=True),
        WindowA_baseline_parity=dict(label="WindowA"),
        WindowB_baseline_parity=dict(label="WindowB"),
        TB2=dict(terminal_metrics=[r for r in terminal_rows if r["block"] == "TB2"],
                 terminal_bootstrap=[r for r in tboot if r["block"] == "TB2"],
                 transition_metrics=[r for r in trans_rows if r["block"] == "TB2"],
                 transition_bootstrap=[r for r in jboot if r["block"] == "TB2"]),
        TB3=dict(terminal_metrics=[r for r in terminal_rows if r["block"] == "TB3"],
                 terminal_bootstrap=[r for r in tboot if r["block"] == "TB3"],
                 transition_metrics=[r for r in trans_rows if r["block"] == "TB3"],
                 transition_bootstrap=[r for r in jboot if r["block"] == "TB3"]),
        primary=dict(TB3_PGM_UE_Delta_LogLoss_bootstrap=ll_boot,
                     TB3_PGM_UE_Delta_JNLL_bootstrap=jnll_boot),
        mechanism=dict(p_edges=[-1.0, 0.0, 1.0], m_edges=[-1.0, 0.0, 1.0],
                       conditional_contrast=contrast),
        age=dict(TB2_spearman=0.5, TB3_spearman=0.5),
        formal_verdict=exp.determine_state_augmentation_verdict(ll_boot, jnll_boot),
        known_scope_limitations=["x"],
    )
    return dict(summary=summary, terminal_rows=terminal_rows, terminal_boot=tboot,
                transition_rows=trans_rows, transition_boot=jboot, grid=grid,
                contrast=contrast, age_rows=age_rows, harm_rows=harm_rows)


def _write_bundle(tmpdir, b):
    td = Path(tmpdir)
    exp._write_artifacts(b["summary"], b["terminal_rows"], b["terminal_boot"],
                         b["transition_rows"], b["transition_boot"], b["grid"],
                         b["contrast"], b["age_rows"], b["harm_rows"], td)
    (td / f"{exp.PREFIX}_formal_summary.json").write_text(
        json.dumps(b["summary"], indent=2, default=str))
    return td


def _expect_parity_fail(b, td):
    raised = None
    try:
        exp.validate_output_artifacts(b["summary"], td)
    except SystemExit as e:
        raised = str(e)
    assert raised is not None, "parity checker did NOT fail"
    assert "STOP_PGM_NATIVE0C_OUTPUT_PARITY_FAIL" in raised


def test_67_output_parity_catches_mutated_artifact():
    import tempfile
    b = _synthetic_bundle()
    with tempfile.TemporaryDirectory() as tmp:
        td = _write_bundle(tmp, b)
        assert exp.validate_output_artifacts(b["summary"], td) is True
        p = td / f"{exp.PREFIX}_terminal_metrics.csv"
        df = pd.read_csv(p)
        df.loc[0, "log_loss"] = df.loc[0, "log_loss"] + 1.0
        df.to_csv(p, index=False)
        _expect_parity_fail(b, td)


# ---------------------------------------------------------------------------
# 69. Final formal_summary JSON parity
# ---------------------------------------------------------------------------
def test_69_formal_summary_json_parity():
    import tempfile
    b = _synthetic_bundle()
    with tempfile.TemporaryDirectory() as tmp:
        td = _write_bundle(tmp, b)
        assert exp.validate_output_artifacts(b["summary"], td) is True


# ---------------------------------------------------------------------------
# 70. Mutated formal_verdict is caught
# ---------------------------------------------------------------------------
def test_70_mutated_formal_verdict_caught():
    import tempfile
    b = _synthetic_bundle()
    with tempfile.TemporaryDirectory() as tmp:
        td = _write_bundle(tmp, b)
        p = td / f"{exp.PREFIX}_formal_summary.json"
        js = json.loads(p.read_text())
        js["formal_verdict"] = "TAMPERED"
        p.write_text(json.dumps(js, indent=2, default=str))
        _expect_parity_fail(b, td)


# ---------------------------------------------------------------------------
# 71. Mutated primary bootstrap point is caught
# ---------------------------------------------------------------------------
def test_71_mutated_primary_bootstrap_point_caught():
    import tempfile
    b = _synthetic_bundle()
    with tempfile.TemporaryDirectory() as tmp:
        td = _write_bundle(tmp, b)
        p = td / f"{exp.PREFIX}_terminal_bootstrap.csv"
        df = pd.read_csv(p)
        m = (df["block"] == "TB3") & (df["variant"] == "PGM_UE") & (df["metric"] == "Delta_LogLoss")
        df.loc[m, "point"] = df.loc[m, "point"] + 0.5
        df.to_csv(p, index=False)
        _expect_parity_fail(b, td)


# ---------------------------------------------------------------------------
# 72. Mutated conditional contrast point is caught
# ---------------------------------------------------------------------------
def test_72_mutated_conditional_contrast_point_caught():
    import tempfile
    b = _synthetic_bundle()
    with tempfile.TemporaryDirectory() as tmp:
        td = _write_bundle(tmp, b)
        p = td / f"{exp.PREFIX}_conditional_contrast.csv"
        df = pd.read_csv(p)
        m = (df["row_type"] == "summary") & (df["metric"] == "Delta_H1_cond")
        df.loc[m, "point"] = df.loc[m, "point"] + 1.0
        df.to_csv(p, index=False)
        _expect_parity_fail(b, td)


# ---------------------------------------------------------------------------
# 73. Dropping a conviction-bin contrast row is caught
# ---------------------------------------------------------------------------
def test_73_dropped_conviction_row_caught():
    import tempfile
    b = _synthetic_bundle()
    with tempfile.TemporaryDirectory() as tmp:
        td = _write_bundle(tmp, b)
        p = td / f"{exp.PREFIX}_conditional_contrast.csv"
        df = pd.read_csv(p)
        df = df[~((df["row_type"] == "conviction_bin") & (df["conviction_bin"] == 3))]
        df.to_csv(p, index=False)
        _expect_parity_fail(b, td)


# ---------------------------------------------------------------------------
# 74. Dropping a mechanism 5x5 cell is caught
# ---------------------------------------------------------------------------
def test_74_dropped_mechanism_cell_caught():
    import tempfile
    b = _synthetic_bundle()
    with tempfile.TemporaryDirectory() as tmp:
        td = _write_bundle(tmp, b)
        p = td / f"{exp.PREFIX}_mechanism_grid.csv"
        df = pd.read_csv(p)
        df = df.drop(index=df.index[0])
        df.to_csv(p, index=False)
        _expect_parity_fail(b, td)


# ---------------------------------------------------------------------------
# 75. age hazard sum(n) != terminal PGM0 n is caught
# ---------------------------------------------------------------------------
def test_75_age_rowcount_closure_caught():
    import tempfile
    b = _synthetic_bundle()
    with tempfile.TemporaryDirectory() as tmp:
        td = _write_bundle(tmp, b)
        p = td / f"{exp.PREFIX}_age_hazard.csv"
        df = pd.read_csv(p)
        df.loc[df.index[0], "n"] = df.loc[df.index[0], "n"] + 1
        df.to_csv(p, index=False)
        _expect_parity_fail(b, td)


# ---------------------------------------------------------------------------
# 76. Mutated terminal non-logloss field (roc_auc) is caught
# ---------------------------------------------------------------------------
def test_76_mutated_terminal_roc_auc_caught():
    import tempfile
    b = _synthetic_bundle()
    with tempfile.TemporaryDirectory() as tmp:
        td = _write_bundle(tmp, b)
        p = td / f"{exp.PREFIX}_terminal_metrics.csv"
        df = pd.read_csv(p)
        df.loc[df.index[0], "roc_auc"] = df.loc[df.index[0], "roc_auc"] + 0.1
        df.to_csv(p, index=False)
        _expect_parity_fail(b, td)


# ---------------------------------------------------------------------------
# 77. Mutated transition rho_zdup is caught
# ---------------------------------------------------------------------------
def test_77_mutated_transition_rho_zdup_caught():
    import tempfile
    b = _synthetic_bundle()
    with tempfile.TemporaryDirectory() as tmp:
        td = _write_bundle(tmp, b)
        p = td / f"{exp.PREFIX}_transition_metrics.csv"
        df = pd.read_csv(p)
        df.loc[df.index[0], "rho_zdup"] = df.loc[df.index[0], "rho_zdup"] + 0.1
        df.to_csv(p, index=False)
        _expect_parity_fail(b, td)


# ---------------------------------------------------------------------------
# 78. Final parity runs AFTER all governance summary fields are written
# ---------------------------------------------------------------------------
def test_78_final_parity_after_governance_fields():
    src_pipe = inspect.getsource(exp.execute_full_pipeline)
    assert "validate_output_artifacts" not in src_pipe  # not validated mid-pipeline
    src_run = inspect.getsource(exp.run_full_exploratory)
    i_fields = src_run.index("WindowB_baseline_parity")
    i_write = src_run.index('{PREFIX}_formal_summary.json')  # f-string source form
    i_validate = src_run.index("validate_output_artifacts")
    assert i_fields < i_write < i_validate, "final write/validate must follow governance fields"


# ---------------------------------------------------------------------------
# 68. Full runner contains no strategy-optimization tokens
# ---------------------------------------------------------------------------
def test_68_full_runner_has_no_strategy_optimization_tokens():
    src = (inspect.getsource(exp.execute_full_pipeline)
           + inspect.getsource(exp.run_full_exploratory))
    for tok in ["best_variant", "select_model", "argmax", "threshold", "position",
                "ret_0c", "action_0c", "q_learning", "reinforcement_learning",
                "market_regime_v2"]:
        assert tok not in src, f"forbidden token {tok!r} in full runner"


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import traceback
    tests = [getattr(sys.modules[__name__], f) for f in dir(sys.modules[__name__])
             if f.startswith("test_")]
    tests.sort(key=lambda fn: int(fn.__name__.split("_")[1]))
    ok = fail = 0
    print(f"Running {len(tests)} unit tests for PGM-NATIVE-0C...\n")
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
