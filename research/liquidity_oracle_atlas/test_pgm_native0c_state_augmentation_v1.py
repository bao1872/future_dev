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


# ---------------------------------------------------------------------------
# Cached heavy fixtures
# ---------------------------------------------------------------------------
_FIT_A = None
_OBS_AUG = None
_MERGED = None


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
def test_35_full_blocked_first_round():
    old = os.environ.pop("AUTHORIZE_PGM_NATIVE0C_FULL_EXPLORATORY", None)
    try:
        failed = False
        try:
            exp.run_full_exploratory()
        except SystemExit as e:
            failed = True
            assert "STOP_PGM_NATIVE0C_FULL_NOT_AUTHORIZED_FIRST_ROUND" in str(e)
        assert failed
        # even WITH the env var, the first-round gate must hold
        os.environ["AUTHORIZE_PGM_NATIVE0C_FULL_EXPLORATORY"] = "1"
        failed2 = False
        try:
            exp.run_full_exploratory()
        except SystemExit as e:
            failed2 = True
            assert "STOP_PGM_NATIVE0C_FULL_NOT_AUTHORIZED_FIRST_ROUND" in str(e)
        assert failed2, "full must remain blocked in first round even with env var"
    finally:
        os.environ.pop("AUTHORIZE_PGM_NATIVE0C_FULL_EXPLORATORY", None)
        if old is not None:
            os.environ["AUTHORIZE_PGM_NATIVE0C_FULL_EXPLORATORY"] = old


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
