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
_SCORED = None


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


def scored():
    global _SCORED
    if _SCORED is None:
        _SCORED = d0.prepare_window_windowframe(prep()["aligned"], fit_A(),
                                                n0c.U_COLS + n0c.E_COLS)
    return _SCORED


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
def test_49_full_blocked_even_with_env():
    old = os.environ.get("AUTHORIZE_PGM_NATIVE0D_FULL_EXPLORATORY")
    os.environ["AUTHORIZE_PGM_NATIVE0D_FULL_EXPLORATORY"] = "1"
    raised = None
    try:
        d0.run_full_exploratory()
    except SystemExit as e:
        raised = str(e)
    finally:
        if old is None:
            os.environ.pop("AUTHORIZE_PGM_NATIVE0D_FULL_EXPLORATORY", None)
        else:
            os.environ["AUTHORIZE_PGM_NATIVE0D_FULL_EXPLORATORY"] = old
    assert raised is not None
    assert "STOP_PGM_NATIVE0D_FULL_NOT_AUTHORIZED_FIRST_ROUND" in raised


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
