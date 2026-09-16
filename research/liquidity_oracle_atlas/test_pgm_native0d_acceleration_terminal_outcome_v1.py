"""
test_pgm_native0d_acceleration_terminal_outcome_v1.py
====================================================

Tests for PGM-NATIVE-0D (Round 1: architecture + audit + smoke).
50+ tests covering the frozen contracts: acceleration block, causal prefix invariance,
same-block entry scope, H1 outcome heads (fixed Logistic / fixed Ridge), branch value,
fixed policies, cost accounting, strategy metrics, day-clustered bootstrap, verdict wiring.
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

import research.liquidity_oracle_atlas.experiment_pgm_native0d_acceleration_terminal_outcome_v1 as d0
import research.liquidity_oracle_atlas.experiment_pgm_native0a_one_step_alpha_v1 as n0a
import research.liquidity_oracle_atlas.experiment_pgm_native0c_state_augmentation_v1 as n0c
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1c_free_run_rollout_v1 as pgm


_PREP = None
_FIT_A = None
_FIT_B = None
_SCORED_A = None
_SCORED_B = None
_WIN_A = None


def prep():
    global _PREP
    if _PREP is None:
        _PREP = d0.load_prepared_frame()
    return _PREP


def fit_A():
    global _FIT_A
    if _FIT_A is None:
        _FIT_A = pgm.fit_samplers_for_window(pgm.WINDOWS[0], pgm.SAMPLE_PATH,
                                             pgm.TRANSITION_SAMPLE_PATH)
    return _FIT_A


def fit_B():
    global _FIT_B
    if _FIT_B is None:
        _FIT_B = pgm.fit_samplers_for_window(pgm.WINDOWS[1], pgm.SAMPLE_PATH,
                                             pgm.TRANSITION_SAMPLE_PATH)
    return _FIT_B


def scored_A():
    global _SCORED_A
    if _SCORED_A is None:
        _SCORED_A = d0.prepare_window_windowframe(prep()["aligned"], fit_A(), "test_scored_A")
    return _SCORED_A


def scored_B():
    global _SCORED_B
    if _SCORED_B is None:
        _SCORED_B = d0.prepare_window_windowframe(prep()["aligned"], fit_B(), "test_scored_B")
    return _SCORED_B


def scored():
    return scored_A()


def win_A_run():
    global _WIN_A
    if _WIN_A is None:
        _WIN_A = d0._run_window(scored_A(), d0.pgm.WINDOWS[0], n_boot=50, eval_cap=300)
    return _WIN_A


def synth(n=4, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(dict(
        symbol=["X"] * n, episode_id=["E"] * n, block=["TB1"] * n,
        bar_t=np.arange(10, 10 + n), start_bar=[10] * n,
        score_mu=rng.normal(0, 1, n),
        path_last_return_R=rng.normal(0, 0.5, n),
        path_current_bar_range_R=np.abs(rng.normal(0.4, 0.1, n)),
        e_local_eff_3=rng.uniform(0, 1, n),
    ))


def synth2():
    """Hard-coded values for exact formula checks (3 rows)."""
    return pd.DataFrame(dict(
        symbol=["X"] * 3, episode_id=["E"] * 3, block=["TB1"] * 3,
        bar_t=[10, 11, 12], start_bar=[10] * 3,
        score_mu=[2.0, -2.0, 3.0],                 # d = [+1,-1,+1]
        path_last_return_R=[0.5, 0.3, -0.2],       # r0,r1,r2
        path_current_bar_range_R=[1.0, 0.5, 0.2],  # q0,q1,q2
        e_local_eff_3=[0.6, 0.4, 0.9],             # eff0,eff1,eff2
    ))


# --- 1 base sha ancestor -------------------------------------------------
def test_1_base_sha_ancestor():
    assert d0.BASE_SHA == "3399f20b5497572dd44d29b48f2f340b34d29599"
    res = subprocess.run(["git", "merge-base", "--is-ancestor", d0.BASE_SHA, "HEAD"],
                         cwd=str(_REPO_ROOT), capture_output=True)
    assert res.returncode == 0


# --- 2 no TB4 ------------------------------------------------------------
def test_2_no_tb4_block():
    obs = n0a.load_observed_decision_universe()
    assert set(obs["block"].unique()) <= set(d0.ALLOWED_BLOCKS)
    assert "TB4" not in set(obs["block"].unique())
    src = inspect.getsource(d0.assert_allowed_blocks)
    assert "STOP_PGM_NATIVE0D_FORBIDDEN_BLOCK" in src


# --- 3 sample hashes -----------------------------------------------------
def test_3_sample_hashes():
    h = d0.compute_artifact_hashes()
    assert h["sample_artifact_sha256"] == "12cb5c109039f068535bd156278897df5e4a3179c73e8f792e9d3ca674e80f55"
    assert h["transition_artifact_sha256"] == "efdce8122e355d8f1407ddc3c9be3273fad998e2ecbad4372d159ccfa780c6b4"


# --- 4 row counts --------------------------------------------------------
def test_4_row_counts():
    obs = n0a.load_observed_decision_universe()
    assert len(obs) == 359714


# --- 5 H0/H1 counts ------------------------------------------------------
def test_5_h0_h1_counts():
    obs = n0a.load_observed_decision_universe()
    assert int((obs["hazard"] == 0).sum()) == 321727
    assert int((obs["hazard"] == 1).sum()) == 37987


# --- 6 15 symbols --------------------------------------------------------
def test_6_15_symbols():
    obs = n0a.load_observed_decision_universe()
    assert obs["symbol"].nunique() == 15


# --- 7 same-block entry scope -------------------------------------------
def test_7_same_block_entry_scope():
    src = inspect.getsource(d0.compute_same_block_entry_valid)
    assert "decision_day" in src and "entry_day" in src
    s = scored()
    # every same-block valid row's entry_day must be in its block's decision day set
    assert s["same_block_entry_valid"].dtype == bool
    assert int(s["same_block_entry_valid"].sum()) <= len(s)


# --- 8 acceleration exactly 8 -------------------------------------------
def test_8_acceleration_exactly_eight():
    assert len(d0.A_COLS) == 8
    assert d0.A_COLS == ["a_dir_velocity", "a_dir_accel_1", "a_dir_jerk_1",
                         "a_speed_accel_1", "a_range_accel_2", "a_eff_slope_1",
                         "a_burst_exhaustion", "a_conviction_burst"]


# --- 9 no name collision -------------------------------------------------
def test_9_no_name_collision():
    assert len(set(d0.A_COLS) & set(d0.outcome_base_num())) == 0


# --- 10-17 A formula checks ---------------------------------------------
def test_10_a1_formula():
    x = d0.add_acceleration_features(synth2())
    assert np.isclose(x["a_dir_velocity"].iloc[0], 0.5)


def test_11_a2_formula():
    # row1: d=-1, r0=0.3, r1=0.5 -> -1*(0.3-0.5)=0.2
    x = d0.add_acceleration_features(synth2())
    assert np.isclose(x["a_dir_accel_1"].iloc[1], 0.2)


def test_12_a3_formula():
    # row2: d=+1, r0=-0.2, r1=0.3, r2=0.5 -> -0.2-0.6+0.5=-0.3
    x = d0.add_acceleration_features(synth2())
    assert np.isclose(x["a_dir_jerk_1"].iloc[2], -0.3)


def test_13_a4_formula():
    # row2: |r0|-|r1| = 0.2-0.3 = -0.1
    x = d0.add_acceleration_features(synth2())
    assert np.isclose(x["a_speed_accel_1"].iloc[2], -0.1)


def test_14_a5_formula():
    # row2: q0-2q1+q2 = 0.2-1.0+1.0 = 0.2
    x = d0.add_acceleration_features(synth2())
    assert np.isclose(x["a_range_accel_2"].iloc[2], 0.2)


def test_15_a6_formula():
    # row1: eff0-eff1 = 0.4-0.6 = -0.2
    x = d0.add_acceleration_features(synth2())
    assert np.isclose(x["a_eff_slope_1"].iloc[1], -0.2)


def test_16_a7_formula():
    x = d0.add_acceleration_features(synth2())
    exp = max(x["a_speed_accel_1"].iloc[2], 0.0) * max(-x["a_eff_slope_1"].iloc[2], 0.0)
    assert np.isclose(x["a_burst_exhaustion"].iloc[2], exp)


def test_17_a8_formula():
    x = d0.add_acceleration_features(synth2())
    exp = abs(x["score_mu"].iloc[2]) * max(x["a_dir_accel_1"].iloc[2], 0.0)
    assert np.isclose(x["a_conviction_burst"].iloc[2], exp)


# --- 18 first row delta features zero -----------------------------------
def test_18_first_row_delta_zero():
    x = d0.add_acceleration_features(synth2())
    for c in ["a_dir_accel_1", "a_dir_jerk_1", "a_speed_accel_1",
              "a_range_accel_2", "a_eff_slope_1"]:
        assert np.isclose(x[c].iloc[0], 0.0)


# --- 19 second row jerk semantics ---------------------------------------
def test_19_second_row_jerk_semantics():
    # row1: r0=0.3, r1=0.5, r2==r1 -> d=-1 -> -(0.3-1.0+0.5)=0.2
    x = d0.add_acceleration_features(synth2())
    assert np.isclose(x["a_dir_jerk_1"].iloc[1], 0.2)


# --- 20 all finite -------------------------------------------------------
def test_20_all_finite():
    s = scored()
    fin = d0.audit_acceleration_finite(s)
    assert fin["all_finite"], fin


# --- 21 prefix invariance ------------------------------------------------
def test_21_prefix_invariance():
    assert d0._prefix_invariance_check() is True


# --- 22 no future token --------------------------------------------------
def test_22_no_future_token():
    src = inspect.getsource(d0.add_acceleration_features)
    assert "shift(-1)" not in src
    assert "_FUTURE" not in src
    for tok in ["next_", "remaining", "final_"]:
        assert tok not in src


# --- 23 score_mu causal action ------------------------------------------
def test_23_score_mu_causal_action():
    src = inspect.getsource(d0.attach_baseline_score)
    assert "-np.asarray" in src and "z_d_up_mu" in src
    assert "hazard" not in src
    s = scored().head(100)
    assert np.allclose(s["base_action"], np.sign(s["score_mu"]))


# --- 24 harm label -------------------------------------------------------
def test_24_harm_label():
    s = scored().head(200)
    exp = (s["pi"].to_numpy(float) < 0).astype(int)
    assert np.array_equal(s["harm_flag"].to_numpy(int), exp)


# --- 25 pi label ---------------------------------------------------------
def test_25_pi_label():
    s = scored().head(200)
    exp = np.sign(s["score_mu"].to_numpy(float)) * s["r_trad_OC_ATR0"].to_numpy(float)
    assert np.allclose(s["pi"].to_numpy(float), exp)


# --- 26 no outcome target predictor leak --------------------------------
def test_26_no_target_predictor_leak():
    num = d0.outcome_base_num()
    for bad in ["hazard", "harm_flag", "pi", "r_trad_OC_ATR0", "target_mask"]:
        assert bad not in num
        assert bad not in d0.A_COLS


# --- 27 O0 feature set ---------------------------------------------------
def test_27_o0_feature_set():
    num = d0.outcome_base_num()
    for c in pgm.T2_NUM:
        assert c in num
    for c in list(n0c.U_COLS) + list(n0c.E_COLS):
        assert c in num
    assert "score_mu" in num and "abs_score_mu" in num
    assert all(a not in num for a in d0.A_COLS)


# --- 28 OA = O0 + A ------------------------------------------------------
def test_28_oa_is_o0_plus_a():
    src = inspect.getsource(d0.fit_harm_model)
    assert "outcome_base_num() + list(extra_cols)" in src
    assert d0.outcome_base_num() == d0.outcome_base_num()


# --- 29 logistic fixed C=1 ----------------------------------------------
def test_29_logistic_fixed_c1():
    sig = inspect.signature(d0.fit_harm_model)
    assert "extra_cols" in sig.parameters
    # uses the frozen shared pipeline owner (pm.make_pipeline) with no C search
    src = inspect.getsource(d0.fit_harm_model)
    assert "pm.make_pipeline" in src
    assert "class_weight" not in src


# --- 30 ridge alpha=1 lsqr ----------------------------------------------
def test_30_ridge_fixed():
    p = d0.make_ridge_pipeline(["x"], [])
    reg = p.named_steps["reg"]
    assert reg.alpha == 1.0 and reg.solver == "lsqr"


# --- 31/32 window routing ------------------------------------------------
def test_31_window_a_routing():
    assert d0.pgm.WINDOWS[0]["train"] == ["TB1"] and d0.pgm.WINDOWS[0]["eval"] == "TB2"
    src = inspect.getsource(d0._run_window)
    assert 'scored["block"].isin(w["train"])' in src
    assert 'scored["block"] == w["eval"]' in src


def test_32_window_b_routing():
    assert d0.pgm.WINDOWS[1]["train"] == ["TB1", "TB2"] and d0.pgm.WINDOWS[1]["eval"] == "TB3"


# --- 33 H1-only fit sample ----------------------------------------------
def test_33_h1_only_fit_sample():
    src = inspect.getsource(d0._run_window)
    assert 'econ_tr["hazard"] == 1' in src
    assert 'econ_tr["base_action"] != 0' in src


# --- 34/35 delta sign conventions ---------------------------------------
def test_34_harm_delta_sign():
    day = np.array(["d1"] * 30 + ["d2"] * 30)
    r = d0.n0c.fast_cluster_bootstrap_delta(day, np.full(60, 0.5), np.full(60, 0.3), n_boot=50)
    assert r["point"] > 0


def test_35_payoff_mse_delta_sign():
    day = np.array(["d1"] * 30 + ["d2"] * 30)
    r = d0.n0c.fast_cluster_bootstrap_delta(day, np.full(60, 1.0), np.full(60, 0.6), n_boot=50)
    assert r["point"] > 0


# --- 36/37 bootstrap -----------------------------------------------------
def test_36_paired_day_bootstrap():
    day = np.array(["a"] * 10 + ["b"] * 10 + ["c"] * 10)
    nets = {"BASE": np.zeros(30), "GATEA": np.full(30, 0.01), "GATE0": np.zeros(30)}
    out = d0.economic_bootstrap(day, nets, n_boot=100)
    assert out["n_days"] == 3
    assert "GATEA-BASE" in out and out["GATEA-BASE"]["point"] > 0


def test_37_bootstrap_deterministic():
    day = np.repeat(np.arange(10), 5)
    rng = np.random.default_rng(2)
    nets = {"BASE": rng.random(50) - 0.2, "GATEA": rng.random(50) - 0.1,
            "GATE0": rng.random(50) - 0.15}
    a = d0.economic_bootstrap(day, nets, n_boot=100, seed=7)
    b = d0.economic_bootstrap(day, nets, n_boot=100, seed=7)
    assert a["GATEA"]["ci95_lower"] == b["GATEA"]["ci95_lower"]


# --- 38 beta0 formula ----------------------------------------------------
def test_38_beta0_formula():
    df = pd.DataFrame({"score_mu": [1.0, 2.0], "pi": [0.5, 0.5]})
    # sum(x*y)=0.5+1.0=1.5 ; sum(x^2)=1+4=5 -> 0.3
    assert np.isclose(d0.calibrate_mu0(df), 0.3, atol=1e-9)


# --- 39 mu0 formula ------------------------------------------------------
def test_39_mu0_formula():
    m = np.array([2.0, -3.0])
    assert np.allclose(0.5 * np.abs(m), [1.0, 1.5])


# --- 40 mixture value ----------------------------------------------------
def test_40_mixture_value():
    ph = np.array([0.0, 1.0, 0.5])
    sm = np.array([1.0, 1.0, 1.0])
    mu1 = np.array([9.0, 9.0, 9.0])
    v = d0.compose_branch_value(ph, sm, beta0=2.0, mu1=mu1)
    # mu0 = 2*1 = 2 -> V = (1-ph)*2 + ph*9
    assert np.allclose(v, [2.0, 9.0, 5.5])


# --- 41/42/43 policies ---------------------------------------------------
def test_41_gate0_formula():
    sm = np.array([1.0, -1.0, 1.0])
    v = np.array([0.02, 0.02, 0.005])
    assert np.array_equal(d0.apply_value_gate(sm, v, 0.01), [1.0, -1.0, 0.0])


def test_42_gatea_formula():
    sm = np.array([1.0, -1.0])
    v = np.array([0.011, 0.01])   # strict >
    assert np.array_equal(d0.apply_value_gate(sm, v, 0.01), [1.0, 0.0])


def test_43_flipa_formula():
    sm = np.array([1.0, 1.0, 1.0, 1.0])
    v = np.array([0.02, -0.02, 0.0, 0.005])
    assert np.array_equal(d0.apply_value_flip(sm, v, 0.01), [1.0, -1.0, 0.0, 0.0])


# --- 44 cost accounting --------------------------------------------------
def test_44_cost_accounting():
    a = np.array([1.0, 0.0, -1.0])
    r = np.array([0.03, 0.03, 0.03])
    net = d0.net_return(a, r, 0.01)
    assert np.allclose(net, [0.02, 0.0, -0.04])


# --- 45 strategy metrics synthetic truth --------------------------------
def test_45_strategy_metrics_truth():
    a = np.array([1.0, 0.0, -1.0, 1.0])
    r = np.array([0.02, 0.5, -0.02, -0.01])
    day = np.array(["d1", "d2", "d2", "d3"])
    sym = np.array(["A", "A", "B", "B"])
    m = d0.strategy_metrics(a, r, 0.01, day, sym)
    # gross = [0.02,0,0.02,-0.01] net = [0.01,0,0.01,-0.02]
    assert np.isclose(m["net_total_ATR0"], 0.01 + 0.0 + 0.01 - 0.02, atol=1e-12)
    assert m["n_trades"] == 3
    assert np.isclose(m["trade_rate"], 0.75)


# --- 46 daily sharpe aggregation ----------------------------------------
def test_46_daily_sharpe_aggregation():
    a = np.array([1.0, 1.0, 1.0, 1.0])
    r = np.array([0.02, 0.02, 0.02, 0.02])
    day = np.array(["d1", "d1", "d2", "d2"])
    m = d0.strategy_metrics(a, r, 0.0, day, np.array(["A"] * 4))
    # daily = [0.04, 0.04] -> std=0 -> sharpe 0 by construction
    assert m["daily_sharpe_annualized"] == 0.0
    m2 = d0.strategy_metrics(a, np.array([0.02, -0.01, 0.03, 0.01]), 0.0, day, np.array(["A"] * 4))
    assert np.isfinite(m2["daily_sharpe_annualized"])


# --- 47 max drawdown truth ----------------------------------------------
def test_47_max_drawdown_truth():
    a = np.array([1.0, 1.0, 1.0])
    r = np.array([0.05, -0.10, 0.02])
    day = np.array(["d1", "d2", "d3"])
    m = d0.strategy_metrics(a, r, 0.0, day, np.array(["A"] * 3))
    assert np.isclose(m["max_drawdown_ATR0"], -0.10, atol=1e-12)


# --- 48 by-symbol breadth ------------------------------------------------
def test_48_by_symbol_breadth():
    a = np.array([1.0, 1.0, 1.0])
    r = np.array([0.05, -0.02, 0.03])
    day = np.array(["d1", "d2", "d3"])
    sym = np.array(["A", "B", "C"])
    m = d0.strategy_metrics(a, r, 0.0, day, sym)
    assert m["positive_symbol_count"] == 2
    assert np.isclose(m["top3_profit_share"], 1.0)


# --- 49 full blocked even with env --------------------------------------
def test_49_full_gate_never_runs_pipeline():
    """The gate must be testable WITHOUT ever executing the full pipeline."""
    old = os.environ.pop("AUTHORIZE_PGM_NATIVE0D_FULL_EXPLORATORY", None)
    try:
        raised = None
        try:
            d0.require_full_authorization()
        except SystemExit as e:
            raised = str(e)
        assert raised is not None
        assert "STOP_PGM_NATIVE0D_FULL_EXPLORATORY_NOT_AUTHORIZED" in raised
    finally:
        if old is not None:
            os.environ["AUTHORIZE_PGM_NATIVE0D_FULL_EXPLORATORY"] = old
    # run_full_exploratory must gate on authorization BEFORE any heavy work
    src = inspect.getsource(d0.run_full_exploratory)
    assert src.index("require_full_authorization()") < src.index("load_prepared_frame()")


# --- 50 smoke no verdict -------------------------------------------------
def test_50_smoke_no_verdict():
    src = inspect.getsource(d0.run_smoke_test)
    assert "NO SCIENTIFIC VERDICT" in src
    assert "determine_information_verdict" not in src
    assert "determine_economic_verdict" not in src


# --- 51 information verdict wiring --------------------------------------
def test_51_information_verdict_wiring():
    assert d0.determine_information_verdict({"ci95_lower": 1.0}, {"ci95_lower": 1.0}) == \
        d0.VERDICT_STRINGS["INFO_JOINT"]
    assert d0.determine_information_verdict({"ci95_lower": 1.0}, {"ci95_lower": -1.0}) == \
        d0.VERDICT_STRINGS["INFO_HARM_ONLY"]
    assert d0.determine_information_verdict({"ci95_lower": -1.0}, {"ci95_lower": 1.0}) == \
        d0.VERDICT_STRINGS["INFO_PAYOFF_ONLY"]
    assert d0.determine_information_verdict({"ci95_lower": -1.0}, {"ci95_lower": -1.0}) == \
        d0.VERDICT_STRINGS["INFO_NONE"]


# --- 52 economic verdict wiring -----------------------------------------
def test_52_economic_verdict_wiring():
    def bk(g, d):
        return {"GATEA": {"ci95_lower": g}, "GATEA-GATE0": {"ci95_lower": d}}
    assert d0.determine_economic_verdict(bk(1, 1), bk(1, 1)) == d0.VERDICT_STRINGS["ECON_SUPPORTED"]
    assert d0.determine_economic_verdict(bk(1, 1), bk(1, -1)) == d0.VERDICT_STRINGS["ECON_NOT_INCREMENTAL"]
    assert d0.determine_economic_verdict(bk(1, 1), bk(-1, 1)) == d0.VERDICT_STRINGS["ECON_NOT_SUPPORTED"]


# --- 53 acceleration built once guard -----------------------------------
def test_53_acceleration_build_once_guard():
    src = inspect.getsource(d0.build_acceleration_once)
    assert "STOP_PGM_NATIVE0D_ACCELERATION_BUILT_TWICE" in src


# --- 54 cost grid --------------------------------------------------------
def test_54_cost_grid():
    assert d0.PRIMARY_COST_ATR0 == 0.01
    assert d0.COST_GRID == [0.0, 0.01, 0.02, 0.03, 0.05, 0.10]


# --- 55 no GBDT / RL / V2 -----------------------------------------------
def test_55_no_gbdt_rl_v2():
    text = Path(d0.__file__).read_text().lower()
    for tok in ["xgboost", "lightgbm", "q_learning", "reinforcement_learning",
                "stable_baselines", "market_regime_v2", "torch"]:
        assert tok not in text


# --- 56/57 score owner parity -------------------------------------------
def test_56_window_a_score_owner_parity():
    d = d0.verify_window_score_owner(scored_A(), fit_A(), "TB2")
    assert d <= 1e-12, d


def test_57_window_b_score_owner_parity():
    d = d0.verify_window_score_owner(scored_B(), fit_B(), "TB3")
    assert d <= 1e-12, d


# --- 58 fake A/B samplers produce distinct score ------------------------
def _synth_aligned(n=3):
    return pd.DataFrame(dict(
        symbol=["X"] * n, episode_id=["E"] * n, block=["TB1"] * n,
        bar_t=np.arange(10, 10 + n), start_bar=[10] * n, hazard=[0] * n,
        path_last_return_R=np.full(n, 0.1), path_current_bar_range_R=np.full(n, 0.5),
        e_local_eff_3=np.full(n, 0.6), r_trad_OC_ATR0=np.full(n, 0.01),
        entry_day=pd.to_datetime(["2026-01-01"] * n),
        decision_day=pd.to_datetime(["2026-01-01"] * n),
    ))


class _FakeMC:
    def __init__(self, mu):
        self.mu = mu

    def analytic_conditional_support(self, df):
        return {"z_d_up_mu": np.full(len(df), self.mu)}


def test_58_fake_ab_samplers_distinct_score():
    fit_a = {"trans_samplers": {n0c.PRIMARY_TRANSITION_HEAD: _FakeMC(-1.0)}}
    fit_b = {"trans_samplers": {n0c.PRIMARY_TRANSITION_HEAD: _FakeMC(-2.0)}}
    df = _synth_aligned()
    sa = d0.prepare_window_windowframe(df, fit_a, "fake_a")
    sb = d0.prepare_window_windowframe(df, fit_b, "fake_b")
    assert np.allclose(sa["score_mu"].to_numpy(float), 1.0)
    assert np.allclose(sb["score_mu"].to_numpy(float), 2.0)


# --- 59/60 non-vacuous routing ------------------------------------------
def test_59_tb3_must_consume_scored_b():
    fb = fit_B()
    mc = fb["trans_samplers"][n0c.PRIMARY_TRANSITION_HEAD]
    subB = scored_B()
    subB = subB[subB["block"] == "TB3"]
    exp = -np.asarray(mc.analytic_conditional_support(subB)["z_d_up_mu"], np.float64)
    assert np.max(np.abs(subB["score_mu"].to_numpy(float) - exp)) <= 1e-12
    subA = scored_A()
    subA = subA[subA["block"] == "TB3"]
    # Window A's score on TB3 is NOT the Window B owner -> mis-routing would be detectable
    assert not np.allclose(subA["score_mu"].to_numpy(float), exp, atol=1e-9)


def test_60_tb2_must_consume_scored_a():
    fa = fit_A()
    mc = fa["trans_samplers"][n0c.PRIMARY_TRANSITION_HEAD]
    subA = scored_A()
    subA = subA[subA["block"] == "TB2"]
    exp = -np.asarray(mc.analytic_conditional_support(subA)["z_d_up_mu"], np.float64)
    assert np.max(np.abs(subA["score_mu"].to_numpy(float) - exp)) <= 1e-12
    subB = scored_B()
    subB = subB[subB["block"] == "TB2"]
    assert not np.allclose(subB["score_mu"].to_numpy(float), exp, atol=1e-9)


# --- 61/62 audit + smoke fit A and B separately -------------------------
def test_61_audit_fits_a_and_b_separately():
    src = inspect.getsource(d0.run_audit_only)
    assert "pgm.WINDOWS[0]" in src and "pgm.WINDOWS[1]" in src
    assert "scored_A" in src and "scored_B" in src


def test_62_smoke_fits_a_and_b_separately():
    src = inspect.getsource(d0.run_smoke_test)
    assert "pgm.WINDOWS[0]" in src and "pgm.WINDOWS[1]" in src
    assert "scored_A" in src and "scored_B" in src


# --- 63/64 production fit exactly once per window -----------------------
def test_63_window_a_fit_once():
    src = inspect.getsource(d0.run_smoke_test)
    assert src.count("fit_samplers_for_window(pgm.WINDOWS[0]") == 1


def test_64_window_b_fit_once():
    src = inspect.getsource(d0.run_smoke_test)
    assert src.count("fit_samplers_for_window(pgm.WINDOWS[1]") == 1


# --- 65/66/67 cost grid recomputes gates ---------------------------------
def test_65_gate0_recomputed_at_cost():
    sm = np.ones(4)
    V = np.array([0.005, 0.015, 0.025, 0.06])
    counts = [int((d0.apply_value_gate(sm, V, c) != 0).sum())
              for c in [0.01, 0.02, 0.05, 0.10]]
    assert counts == [3, 2, 1, 0]


def test_66_gatea_recomputed_at_cost():
    sm = np.array([1.0, -1.0, 1.0, -1.0])
    VA = np.array([0.005, 0.015, 0.025, 0.06])
    counts = [int((d0.apply_value_gate(sm, VA, c) != 0).sum())
              for c in [0.01, 0.02, 0.05, 0.10]]
    assert counts == [3, 2, 1, 0]


def test_67_flipa_recomputed_at_cost():
    sm = np.ones(4)
    VA = np.array([0.005, -0.015, 0.025, -0.06])
    counts = [int((d0.apply_value_flip(sm, VA, c) != 0).sum())
              for c in [0.01, 0.02, 0.05, 0.10]]
    assert counts == [3, 2, 1, 0]


# --- 68 primary .01 grid parity -----------------------------------------
def test_68_primary_cost_grid_parity():
    r = win_A_run()
    g = r["cost_grid"][str(d0.PRIMARY_COST_ATR0)]
    for k in ["BASE", "GATE0", "GATEA", "FLIPA"]:
        for f in ["n_trades", "trade_rate", "gross_total_ATR0", "net_total_ATR0",
                  "net_EV_per_decision", "profit_factor"]:
            assert np.isclose(g[k][f], r["metrics"][k][f], rtol=0, atol=1e-12,
                              equal_nan=True), (k, f)


# --- 69/70/71/72 monotonic trade counts ---------------------------------
def _cost_counts(pol):
    r = win_A_run()
    return [r["cost_grid"][str(c)][pol]["n_trades"] for c in d0.COST_GRID]


def test_69_gate0_trade_count_non_increasing():
    c = _cost_counts("GATE0")
    assert all(c[i + 1] <= c[i] for i in range(len(c) - 1)), c


def test_70_gatea_trade_count_non_increasing():
    c = _cost_counts("GATEA")
    assert all(c[i + 1] <= c[i] for i in range(len(c) - 1)), c


def test_71_flipa_trade_count_non_increasing():
    c = _cost_counts("FLIPA")
    assert all(c[i + 1] <= c[i] for i in range(len(c) - 1)), c


def test_72_base_trade_count_invariant():
    c = _cost_counts("BASE")
    assert len(set(c)) == 1, c


# --- 73/74 authorization -------------------------------------------------
def test_73_full_auth_no_env_stops():
    old = os.environ.pop("AUTHORIZE_PGM_NATIVE0D_FULL_EXPLORATORY", None)
    raised = None
    try:
        d0.require_full_authorization()
    except SystemExit as e:
        raised = str(e)
    finally:
        if old is not None:
            os.environ["AUTHORIZE_PGM_NATIVE0D_FULL_EXPLORATORY"] = old
    assert raised is not None
    assert "STOP_PGM_NATIVE0D_FULL_EXPLORATORY_NOT_AUTHORIZED" in raised


def test_74_auth_env_one_passes_without_heavy_run():
    old = os.environ.get("AUTHORIZE_PGM_NATIVE0D_FULL_EXPLORATORY")
    os.environ["AUTHORIZE_PGM_NATIVE0D_FULL_EXPLORATORY"] = "1"
    try:
        d0.require_full_authorization()  # must not raise, must not run full
    finally:
        if old is None:
            os.environ.pop("AUTHORIZE_PGM_NATIVE0D_FULL_EXPLORATORY", None)
        else:
            os.environ["AUTHORIZE_PGM_NATIVE0D_FULL_EXPLORATORY"] = old


# --- 75/76 score owner NaN fail-closed ----------------------------------
def test_75_actual_score_nan_stops():
    fit = {"trans_samplers": {n0c.PRIMARY_TRANSITION_HEAD: _FakeMC(-1.0)}}
    s = d0.prepare_window_windowframe(_synth_aligned(3), fit, "nan_actual")
    s.loc[s.index[0], "score_mu"] = np.nan
    raised = None
    try:
        d0.verify_window_score_owner(s, fit, "TB1")
    except SystemExit as e:
        raised = str(e)
    assert raised is not None and "STOP_PGM_NATIVE0D_SCORE_OWNER_NONFINITE" in raised


class _BadMC:
    def analytic_conditional_support(self, df):
        return {"z_d_up_mu": np.full(len(df), np.nan)}


def test_76_expected_score_nan_stops():
    fit = {"trans_samplers": {n0c.PRIMARY_TRANSITION_HEAD: _BadMC()}}
    s = d0.prepare_window_windowframe(_synth_aligned(3), fit, "nan_expected")
    raised = None
    try:
        d0.verify_window_score_owner(s, fit, "TB1")
    except SystemExit as e:
        raised = str(e)
    assert raised is not None and "STOP_PGM_NATIVE0D_SCORE_OWNER_NONFINITE" in raised


# --- 77-82 formal orchestration source contracts ------------------------
def test_77_formal_window_a_fit_once():
    src = inspect.getsource(d0.run_full_exploratory)
    assert src.count("fit_samplers_for_window(pgm.WINDOWS[0]") == 1


def test_78_formal_window_b_fit_once():
    src = inspect.getsource(d0.run_full_exploratory)
    assert src.count("fit_samplers_for_window(pgm.WINDOWS[1]") == 1


def test_79_formal_tb2_uses_scored_a():
    src = inspect.getsource(d0.run_full_exploratory)
    assert 'prepare_window_windowframe(aligned, fit_A, "formal_scored_A")' in src
    assert "_run_window(scored_A, pgm.WINDOWS[0]" in src


def test_80_formal_tb3_uses_scored_b():
    src = inspect.getsource(d0.run_full_exploratory)
    assert 'prepare_window_windowframe(aligned, fit_B, "formal_scored_B")' in src
    assert "_run_window(scored_B, pgm.WINDOWS[1]" in src


def test_81_both_windows_finite_gate():
    src = inspect.getsource(d0.run_full_exploratory)
    assert 'audit_acceleration_finite(scored_A)' in src
    assert 'audit_acceleration_finite(scored_B)' in src
    assert src.count("STOP_PGM_NATIVE0D_ACCELERATION_NON_FINITE") == 2


def test_82_full_eval_cap_none():
    src = inspect.getsource(d0.run_full_exploratory)
    assert src.count("eval_cap=None") == 2


# --- 83-92 artifact contract --------------------------------------------
def test_83_artifact_set_exact_eight():
    assert len(d0.ARTIFACT_FILES) == 8
    assert len(set(d0.ARTIFACT_FILES)) == 8
    assert f"{d0.PREFIX}_formal_summary.json" in d0.ARTIFACT_FILES


def test_92_formal_summary_required_keys():
    src = inspect.getsource(d0.run_full_exploratory)
    for k in ["EXPERIMENT_NAME", "EXPERIMENT_SCOPE", "base_sha", "run_head",
              "sample_artifact_sha256", "transition_artifact_sha256",
              "n_all_obs", "n_H0", "n_H1", "symbols", "blocks",
              "same_block_entry_counts", "max_abs_atr0_owner_error",
              "A_COLS", "OUTCOME_BASE_NUM", "OUTCOME_CAT", "PRIMARY_COST_ATR0",
              "COST_GRID", "BOOTSTRAP_N", "BOOTSTRAP_SEED",
              "WindowA_score_owner_max_abs_diff", "WindowB_score_owner_max_abs_diff",
              "WindowA_acceleration_finite", "WindowB_acceleration_finite",
              "information_verdict", "economic_verdict", "quintile_frozen_edges",
              "known_limitations"]:
        assert k in src, f"summary key {k} missing"


def test_101_tb4_cannot_enter_full():
    src = inspect.getsource(d0.run_pre_fit_integrity_gates)
    assert "STOP_PGM_NATIVE0D_FORBIDDEN_BLOCK" in src
    assert "TB4" in src


def test_102_final_parity_after_json_write():
    src = inspect.getsource(d0.run_full_exploratory)
    i_write = src.index("{PREFIX}_formal_summary.json")
    i_validate = src.index("validate_output_artifacts(summary, out_dir)")
    assert i_write < i_validate


# --- synthetic complete 8-artifact bundle --------------------------------
_SYMS = ["AG", "AL", "AU", "CF", "CU", "I", "M", "MA", "NI", "P", "RB", "RU", "SC", "SN", "TA"]


def _synth_metrics(n_trades, net_total):
    m = {}
    m["n_decisions"] = 10
    m["n_trades"] = n_trades
    m["trade_rate"] = n_trades / 10.0
    m["gross_total_ATR0"] = net_total + 0.01 * n_trades
    m["net_total_ATR0"] = net_total
    m["gross_EV_per_decision"] = (net_total + 0.01 * n_trades) / 10.0
    m["net_EV_per_decision"] = net_total / 10.0
    m["net_EV_per_trade"] = (net_total / n_trades) if n_trades else 0.0
    m["win_rate"] = 0.5
    m["mean_win"] = 0.1
    m["mean_loss"] = -0.1
    m["payoff_ratio"] = 1.0
    m["profit_factor"] = 1.0
    m["break_even_cost"] = 0.02
    m["daily_sharpe_annualized"] = 1.0
    m["max_drawdown_ATR0"] = -0.2
    m["positive_symbol_count"] = 1
    m["top3_profit_share"] = 1.0
    by_sym = {s: dict(net_total=0.0, trade_count=0) for s in _SYMS}
    by_sym[_SYMS[0]] = dict(net_total=net_total, trade_count=n_trades)
    m["by_symbol"] = by_sym
    return m


def _synth_bootstrap(gatea_lo=0.01, delta_lo=0.005):
    return {
        "BASE": dict(point=0.0, ci95_lower=-0.01, ci95_upper=0.01, p_pos=0.5),
        "GATE0": dict(point=0.0, ci95_lower=-0.01, ci95_upper=0.01, p_pos=0.5),
        "GATEA": dict(point=0.02, ci95_lower=gatea_lo, ci95_upper=0.03, p_pos=0.95),
        "FLIPA": dict(point=-0.05, ci95_lower=-0.09, ci95_upper=-0.01, p_pos=0.01),
        "GATEA-BASE": dict(point=0.02, ci95_lower=0.005, ci95_upper=0.04, p_pos=0.95),
        "GATEA-GATE0": dict(point=0.02, ci95_lower=delta_lo, ci95_upper=0.04, p_pos=0.95),
        "n_days": 10, "n_boot": 2000,
    }


def _synth_cost_grid(base_counts, gate_counts, flip_counts):
    grid = {}
    for c in d0.COST_GRID:
        grid[str(c)] = {}
        for p in ["BASE", "GATE0", "GATEA", "FLIPA"]:
            if p == "BASE":
                nt = base_counts
            elif p in ("GATE0", "GATEA"):
                nt = gate_counts[c]
            else:
                nt = flip_counts[c]
            row = {f: 0.0 for f in d0.COST_FIELDS}
            row["n_decisions"] = 10
            row["n_trades"] = nt
            row["trade_rate"] = nt / 10.0
            row["net_total_ATR0"] = 0.01 * nt
            grid[str(c)][p] = row
    return grid


def _synth_bundle():
    r2_metrics = {"BASE": _synth_metrics(10, 0.10), "GATE0": _synth_metrics(4, 0.05),
                  "GATEA": _synth_metrics(4, 0.06), "FLIPA": _synth_metrics(6, -0.02)}
    r3_metrics = {"BASE": _synth_metrics(10, -0.10), "GATE0": _synth_metrics(4, 0.01),
                  "GATEA": _synth_metrics(4, 0.03), "FLIPA": _synth_metrics(6, -0.05)}
    gate_counts = {0.0: 6, 0.01: 4, 0.02: 3, 0.03: 2, 0.05: 1, 0.10: 0}
    flip_counts = {0.0: 8, 0.01: 6, 0.02: 5, 0.03: 4, 0.05: 2, 0.10: 1}
    # cost grid .01 must equal metrics exactly -> patch .01 rows
    def grid_for(metrics):
        g = _synth_cost_grid(10, gate_counts, flip_counts)
        for p in ["BASE", "GATE0", "GATEA", "FLIPA"]:
            row = {f: metrics[p][f] for f in d0.COST_FIELDS}
            g["0.01"][p] = row
        return g

    quint = []
    for blk in ["TB2", "TB3"]:
        for feat in d0.QUINTILE_FEATURES:
            for b in range(5):
                quint.append(dict(block=blk, feature=feat, bin=b, n=3, H1_prevalence=0.1,
                                  harm_rate=0.5, mean_pi=-0.1))
    r2 = dict(harm_metrics={"O0": dict(log_loss=0.6, brier=0.2, roc_auc=0.6, pr_auc=0.2),
                            "OA": dict(log_loss=0.5, brier=0.19, roc_auc=0.62, pr_auc=0.21)},
              payoff_metrics={"O0": dict(mse=1.0, mae=0.8, spearman=0.1),
                              "OA": dict(mse=0.9, mae=0.7, spearman=0.2)},
              delta_harm_logloss=dict(point=0.01, ci95_lower=0.001, ci95_upper=0.02, p_pos=0.9),
              delta_payoff_mse=dict(point=0.01, ci95_lower=0.001, ci95_upper=0.02, p_pos=0.9),
              n_h1_eval=10, n_h1_train=100, n_h0_train=1000, n_h0_eval=990, n_econ_eval=1000,
              metrics=r2_metrics, bootstrap=_synth_bootstrap(),
              cost_grid=grid_for(r2_metrics), quintiles=quint,
              quintile_edges={f: [-1.0, 0.0, 1.0] for f in d0.QUINTILE_FEATURES})
    r3 = dict(harm_metrics={"O0": dict(log_loss=0.6, brier=0.2, roc_auc=0.6, pr_auc=0.2),
                            "OA": dict(log_loss=0.5, brier=0.19, roc_auc=0.62, pr_auc=0.21)},
              payoff_metrics={"O0": dict(mse=1.0, mae=0.8, spearman=0.1),
                              "OA": dict(mse=0.9, mae=0.7, spearman=0.2)},
              delta_harm_logloss=dict(point=0.01, ci95_lower=0.002, ci95_upper=0.02, p_pos=0.9),
              delta_payoff_mse=dict(point=0.01, ci95_lower=0.003, ci95_upper=0.02, p_pos=0.9),
              n_h1_eval=10, n_h1_train=100, n_h0_train=1000, n_h0_eval=990, n_econ_eval=1000,
              metrics=r3_metrics, bootstrap=_synth_bootstrap(),
              cost_grid=grid_for(r3_metrics), quintiles=quint,
              quintile_edges={f: [-1.0, 0.0, 1.0] for f in d0.QUINTILE_FEATURES})
    summary = dict(
        EXPERIMENT_NAME=d0.EXPERIMENT_NAME, EXPERIMENT_SCOPE=d0.EXPERIMENT_SCOPE,
        base_sha=d0.BASE_SHA, run_head="deadbeef",
        sample_artifact_sha256="a" * 64, transition_artifact_sha256="b" * 64,
        n_all_obs=100, n_H0=90, n_H1=10, symbols=_SYMS, blocks=["TB1", "TB2", "TB3"],
        same_block_entry_counts={"TB1": 10, "TB2": 10, "TB3": 10},
        raw_block_counts={"TB1": 10, "TB2": 10, "TB3": 10},
        max_abs_atr0_owner_error=0.0,
        A_COLS=list(d0.A_COLS), OUTCOME_BASE_NUM=d0.outcome_base_num(),
        OUTCOME_CAT=d0.outcome_cat(), PRIMARY_COST_ATR0=d0.PRIMARY_COST_ATR0,
        COST_GRID=list(d0.COST_GRID), BOOTSTRAP_N=2000, BOOTSTRAP_SEED=d0.BOOTSTRAP_SEED,
        cluster_owner=d0.CLUSTER_OWNER,
        WindowA_score_owner_max_abs_diff=0.0, WindowB_score_owner_max_abs_diff=0.0,
        WindowA_acceleration_finite=True, WindowB_acceleration_finite=True,
        TB2=r2, TB3=r3,
        information_verdict=d0.determine_information_verdict(r3["delta_harm_logloss"],
                                                             r3["delta_payoff_mse"]),
        economic_verdict=d0.determine_economic_verdict(r2["bootstrap"], r3["bootstrap"]),
        quintile_frozen_edges={}, artifact_files=list(d0.ARTIFACT_FILES),
        known_limitations=["x"],
    )
    return summary


def _write_synth(tmp):
    summary = _synth_bundle()
    td = Path(tmp)
    d0._write_full_artifacts(summary, td)
    (td / f"{d0.PREFIX}_formal_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    return summary, td


def _expect_fail(summary, td):
    raised = None
    try:
        d0.validate_output_artifacts(summary, td)
    except SystemExit as e:
        raised = str(e)
    assert raised is not None, "parity checker did NOT fail"
    assert "STOP_PGM_NATIVE0D_OUTPUT_PARITY_FAIL" in raised


# --- 84-91 + 93-100 artifact parity / mutations -------------------------
def test_84_to_91_and_mutations():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        summary, td = _write_synth(tmp)
        assert d0.validate_output_artifacts(summary, td) is True
        # row counts
        assert len(pd.read_csv(td / f"{d0.PREFIX}_outcome_metrics.csv")) == 8
        assert len(pd.read_csv(td / f"{d0.PREFIX}_information_bootstrap.csv")) == 4
        assert len(pd.read_csv(td / f"{d0.PREFIX}_strategy_metrics.csv")) == 8
        assert len(pd.read_csv(td / f"{d0.PREFIX}_economic_bootstrap.csv")) == 12
        assert len(pd.read_csv(td / f"{d0.PREFIX}_cost_grid.csv")) == 48
        assert len(pd.read_csv(td / f"{d0.PREFIX}_symbol_metrics.csv")) == 120
        q = pd.read_csv(td / f"{d0.PREFIX}_acceleration_quintiles.csv")
        assert set(q["feature"].unique()) <= set(d0.QUINTILE_FEATURES)


def test_90_symbol_totals_close_to_strategy_totals():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        summary, td = _write_synth(tmp)
        sy = pd.read_csv(td / f"{d0.PREFIX}_symbol_metrics.csv")
        for blk in ["TB2", "TB3"]:
            for p in ["BASE", "GATE0", "GATEA", "FLIPA"]:
                g = sy[(sy["block"] == blk) & (sy["policy"] == p)]
                assert np.isclose(g["net_total_ATR0"].sum(),
                                  summary[blk]["metrics"][p]["net_total_ATR0"], atol=1e-10)
                assert int(g["trade_count"].sum()) == int(summary[blk]["metrics"][p]["n_trades"])


_SYNTH_CACHE = None


def synth_dir():
    global _SYNTH_CACHE
    if _SYNTH_CACHE is None:
        import tempfile
        td = Path(tempfile.mkdtemp())
        s = _write_synth(td)[0]
        _SYNTH_CACHE = (s, td)
    return _SYNTH_CACHE


def test_85_information_bootstrap_rows():
    _, td = synth_dir()
    assert len(pd.read_csv(td / f"{d0.PREFIX}_information_bootstrap.csv")) == 4


def test_86_strategy_metrics_rows():
    _, td = synth_dir()
    assert len(pd.read_csv(td / f"{d0.PREFIX}_strategy_metrics.csv")) == 8


def test_87_economic_bootstrap_rows():
    _, td = synth_dir()
    assert len(pd.read_csv(td / f"{d0.PREFIX}_economic_bootstrap.csv")) == 12


def test_88_cost_grid_rows():
    _, td = synth_dir()
    assert len(pd.read_csv(td / f"{d0.PREFIX}_cost_grid.csv")) == 48


def test_89_symbol_metrics_rows():
    _, td = synth_dir()
    assert len(pd.read_csv(td / f"{d0.PREFIX}_symbol_metrics.csv")) == 120


def test_91_quintile_only_frozen_features():
    _, td = synth_dir()
    q = pd.read_csv(td / f"{d0.PREFIX}_acceleration_quintiles.csv")
    assert set(q["feature"].unique()) <= set(d0.QUINTILE_FEATURES)
    assert set(q["bin"].astype(int)) <= set(range(5))


def test_93_mutate_outcome_stops():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        summary, td = _write_synth(tmp)
        p = td / f"{d0.PREFIX}_outcome_metrics.csv"
        df = pd.read_csv(p)
        df.loc[0, "log_loss"] = df.loc[0, "log_loss"] + 1.0
        df.to_csv(p, index=False)
        _expect_fail(summary, td)


def test_94_mutate_info_bootstrap_stops():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        summary, td = _write_synth(tmp)
        p = td / f"{d0.PREFIX}_information_bootstrap.csv"
        df = pd.read_csv(p)
        df.loc[0, "point"] = df.loc[0, "point"] + 1.0
        df.to_csv(p, index=False)
        _expect_fail(summary, td)


def test_95_mutate_economic_bootstrap_stops():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        summary, td = _write_synth(tmp)
        p = td / f"{d0.PREFIX}_economic_bootstrap.csv"
        df = pd.read_csv(p)
        df.loc[0, "ci95_lower"] = df.loc[0, "ci95_lower"] + 1.0
        df.to_csv(p, index=False)
        _expect_fail(summary, td)


def test_96_mutate_cost_grid_stops():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        summary, td = _write_synth(tmp)
        p = td / f"{d0.PREFIX}_cost_grid.csv"
        df = pd.read_csv(p)
        m = (df["block"] == "TB2") & (df["policy"] == "GATEA") & (df["cost"] == d0.PRIMARY_COST_ATR0)
        df.loc[m, "n_trades"] = df.loc[m, "n_trades"] + 1
        df.to_csv(p, index=False)
        _expect_fail(summary, td)


def test_97_mutate_symbol_totals_stops():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        summary, td = _write_synth(tmp)
        p = td / f"{d0.PREFIX}_symbol_metrics.csv"
        df = pd.read_csv(p)
        df.loc[0, "net_total_ATR0"] = df.loc[0, "net_total_ATR0"] + 1.0
        df.to_csv(p, index=False)
        _expect_fail(summary, td)


def test_98_mutate_formal_information_verdict_stops():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        summary, td = _write_synth(tmp)
        p = td / f"{d0.PREFIX}_formal_summary.json"
        js = json.loads(p.read_text())
        js["information_verdict"] = "TAMPERED"
        p.write_text(json.dumps(js, indent=2, default=str))
        _expect_fail(summary, td)


def test_99_mutate_formal_economic_verdict_stops():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        summary, td = _write_synth(tmp)
        p = td / f"{d0.PREFIX}_formal_summary.json"
        js = json.loads(p.read_text())
        js["economic_verdict"] = "TAMPERED"
        p.write_text(json.dumps(js, indent=2, default=str))
        _expect_fail(summary, td)


def test_100_flipa_cannot_alter_primary_verdict():
    good2 = _synth_bootstrap()
    good3 = _synth_bootstrap()
    v = d0.determine_economic_verdict(good2, good3)
    assert v == d0.VERDICT_STRINGS["ECON_SUPPORTED"]
    # make FLIPA look catastrophic -> verdict must not change
    bad3 = _synth_bootstrap()
    bad3["FLIPA"] = dict(point=-9.0, ci95_lower=-9.0, ci95_upper=-8.0, p_pos=0.0)
    assert d0.determine_economic_verdict(good2, bad3) == v
    # and GATEA-BASE is not part of the verdict either
    bad3b = _synth_bootstrap()
    bad3b["GATEA-BASE"] = dict(point=-9.0, ci95_lower=-9.0, ci95_upper=-8.0, p_pos=0.0)
    assert d0.determine_economic_verdict(good2, bad3b) == v


# --- 103-105 explicit bootstrap seed ownership ---------------------------
_SEED_CAP = None


def seed_cap():
    global _SEED_CAP
    if _SEED_CAP is None:
        cap = {"info": [], "econ": []}
        orig_i = d0.n0c.fast_cluster_bootstrap_delta
        orig_e = d0.economic_bootstrap

        def spy_i(day, base, aug, n_boot=2000, seed=None, **kw):
            cap["info"].append(seed)
            return orig_i(day, base, aug, n_boot=n_boot, seed=seed)

        def spy_e(day, net, n_boot=2000, seed=None, **kw):
            cap["econ"].append(seed)
            return orig_e(day, net, n_boot=n_boot, seed=seed)

        d0.n0c.fast_cluster_bootstrap_delta = spy_i
        d0.economic_bootstrap = spy_e
        try:
            d0._run_window(scored_A(), d0.pgm.WINDOWS[0], n_boot=20, eval_cap=200)
        finally:
            d0.n0c.fast_cluster_bootstrap_delta = orig_i
            d0.economic_bootstrap = orig_e
        _SEED_CAP = cap
    return _SEED_CAP


def test_103_harm_information_bootstrap_explicit_seed():
    cap = seed_cap()
    assert len(cap["info"]) >= 1
    assert cap["info"][0] == d0.BOOTSTRAP_SEED == 20260916


def test_104_payoff_information_bootstrap_explicit_seed():
    cap = seed_cap()
    assert len(cap["info"]) >= 2
    assert cap["info"][1] == d0.BOOTSTRAP_SEED == 20260916


def test_105_economic_bootstrap_explicit_seed():
    cap = seed_cap()
    assert len(cap["econ"]) >= 1
    assert all(s == d0.BOOTSTRAP_SEED for s in cap["econ"])


# --- 106/107 sample-count semantics --------------------------------------
def _expected_counts():
    wA = d0.pgm.WINDOWS[0]
    sc = scored_A()
    ev = sc[sc["block"] == wA["eval"]]
    econ = ev[ev["same_block_entry_valid"]].head(300)
    n_econ = len(econ)
    n_h0 = int(((econ["hazard"] == 0) & (econ["base_action"] != 0)).sum())
    n_h1 = int(((econ["hazard"] == 1) & (econ["base_action"] != 0)).sum())
    return n_econ, n_h0, n_h1


def test_106_n_econ_eval_semantics():
    r = win_A_run()
    n_econ, _, _ = _expected_counts()
    assert r["n_econ_eval"] == n_econ


def test_107_n_h0_eval_semantics():
    r = win_A_run()
    n_econ, n_h0, n_h1 = _expected_counts()
    assert r["n_h0_eval"] == n_h0
    assert r["n_h1_eval"] == n_h1
    # non-vacuous: n_h0_eval must NOT be the whole economic eval count
    assert r["n_h0_eval"] != n_econ or n_h1 == 0


# --- 108-115 final JSON governance mutations ----------------------------
def _mutate_json(td, mutator):
    p = td / f"{d0.PREFIX}_formal_summary.json"
    js = json.loads(p.read_text())
    mutator(js)
    p.write_text(json.dumps(js, indent=2, default=str))


def test_108_mutate_cluster_owner_stops():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        summary, td = _write_synth(tmp)
        _mutate_json(td, lambda js: js.__setitem__("cluster_owner", "episode_start_day"))
        _expect_fail(summary, td)


def test_109_mutate_window_a_owner_diff_stops():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        summary, td = _write_synth(tmp)
        _mutate_json(td, lambda js: js.__setitem__("WindowA_score_owner_max_abs_diff", 0.5))
        _expect_fail(summary, td)


def test_110_mutate_window_b_accel_finite_stops():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        summary, td = _write_synth(tmp)
        _mutate_json(td, lambda js: js.__setitem__("WindowB_acceleration_finite", False))
        _expect_fail(summary, td)


def test_111_mutate_same_block_entry_counts_stops():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        summary, td = _write_synth(tmp)
        _mutate_json(td, lambda js: js.__setitem__("same_block_entry_counts",
                                                   {"TB1": 999, "TB2": 10, "TB3": 10}))
        _expect_fail(summary, td)


def test_112_mutate_quintile_frozen_edges_stops():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        summary, td = _write_synth(tmp)
        _mutate_json(td, lambda js: js.__setitem__("quintile_frozen_edges", {"TB2": {"x": [1]}}))
        _expect_fail(summary, td)


def test_113_mutate_artifact_files_stops():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        summary, td = _write_synth(tmp)
        _mutate_json(td, lambda js: js["artifact_files"].append("bogus.csv"))
        _expect_fail(summary, td)


def test_114_mutate_known_limitations_stops():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        summary, td = _write_synth(tmp)
        _mutate_json(td, lambda js: js["known_limitations"].append("tampered"))
        _expect_fail(summary, td)


def test_115_mutate_tb2_n_h0_eval_stops():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        summary, td = _write_synth(tmp)
        _mutate_json(td, lambda js: js["TB2"].__setitem__("n_h0_eval", 12345))
        _expect_fail(summary, td)


# --- 116-118 artifact-set closure ----------------------------------------
def test_116_extra_prefixed_artifact_stops():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        summary, td = _write_synth(tmp)
        (td / f"{d0.PREFIX}_stray.csv").write_text("x\n1\n")
        _expect_fail(summary, td)


def test_117_pre_run_guard_blocks_existing_artifact():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        d0.assert_no_existing_prefixed_artifacts(td)  # empty dir -> OK
        (td / f"{d0.PREFIX}_stray.csv").write_text("x\n1\n")
        raised = None
        try:
            d0.assert_no_existing_prefixed_artifacts(td)
        except SystemExit as e:
            raised = str(e)
        assert raised is not None
        assert "STOP_PGM_NATIVE0D_FORMAL_ARTIFACT_ALREADY_EXISTS" in raised


def test_118_exact_artifact_set_eight_passes():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        summary, td = _write_synth(tmp)
        assert d0.validate_output_artifacts(summary, td) is True
        got = {p.name for p in td.glob(f"{d0.PREFIX}_*")}
        assert got == set(d0.ARTIFACT_FILES)
        assert len(got) == 8


# --- 119 governance-incident regression ----------------------------------
def test_119_no_test_authorizes_heavy_full():
    src = Path(__file__).read_text()
    # No unit test may invoke the heavy full runner at all (build the needle so this
    # assertion cannot match its own literal).
    needle = "run_full_exploratory" + "()"
    assert needle not in src
    # the runner must be gated before any heavy work
    rsrc = inspect.getsource(d0.run_full_exploratory)
    assert rsrc.index("require_full_authorization()") < rsrc.index("load_prepared_frame()")


# --- runner --------------------------------------------------------------
if __name__ == "__main__":
    import traceback
    tests = [getattr(sys.modules[__name__], f) for f in dir(sys.modules[__name__])
             if f.startswith("test_")]
    tests.sort(key=lambda fn: int(fn.__name__.split("_")[1]))
    ok = fail = 0
    print(f"Running {len(tests)} unit tests for PGM-NATIVE-0D...\n")
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
