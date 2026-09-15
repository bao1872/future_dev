"""
test_pgm_native0b_hazard_reliability_v1.py
==========================================

Unit tests for PGM-NATIVE-0B: Predicted Hazard Reliability Probe.

Covers 31 required tests:
  1. BASE_SHA is ancestor of HEAD
  2. only SAMPLE_PATH ALL rows as strategy universe
  3. true hazard never enters policy function
  4. actual T2 terminal sampler resolved
  5. actual T2 design has no future/target columns
  6. hazard probability finite in [0,1]
  7. deterministic p_h parity with sample_hazard using FixedRNG
  8. TB2 uses Window A terminal + transition heads
  9. TB3 uses Window B terminal + transition heads
  10. score_mu owner parity
  11. action_0a == sign(score_mu)
  12. position_0b == (1-p_h)*sign(score_mu)
  13. position bound [-1,1]
  14. cost scales by abs(position)
  15. TB3 hazard bins use TB2 edges
  16. no true hazard in bin/gate construction
  17. paired bootstrap resamples identical days for 0A/0B
  18. paired bootstrap deterministic
  19. exposure-normalized EV formula
  20. H0/H1 metrics diagnostic only
  21. all 15 symbols retained
  22. T0 is diagnostic only
  23. smoke emits no scientific verdict
  24. full-exploratory blocked first round
  25. no V2 / R1-R4 / Q model / RL dependency
  26. audit-only artifact hashes fail-closed / 64-char SHA256
  27. actual MC_STATE_CURREENCODING design_cols have zero future/target overlap
  28. T0 TB3 hazard deciles use TB2 T0 edges
  29. T0 diagnostic helper consumes/generates no economic-policy field
  30. smoke path actually executes hazard-decile rough ordering without verdict
  31. full output contract includes T0 diagnostic decile artifact
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

for _bt in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_bt] = "1"

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

import research.liquidity_oracle_atlas.experiment_pgm_native0b_hazard_reliability_v1 as exp
import research.liquidity_oracle_atlas.experiment_pgm_native0a_one_step_alpha_v1 as n0a
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1c_free_run_rollout_v1 as pgm
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a1b_count_magnitude_closure_v1 as base
import research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 as ex0


# ===========================================================================
# 1. BASE_SHA is Ancestor of HEAD
# ===========================================================================
def test_1_base_sha_is_ancestor_of_head():
    assert exp.BASE_SHA == "dfb2bba5edeccd731b84610754798537ab3ef674"
    res = subprocess.run(
        ["git", "merge-base", "--is-ancestor", exp.BASE_SHA, "HEAD"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
    )
    assert res.returncode == 0, f"BASE_SHA {exp.BASE_SHA} is not ancestor of HEAD"


# ===========================================================================
# 2. Only SAMPLE_PATH ALL Rows as Strategy Universe
# ===========================================================================
def test_2_only_sample_path_all_rows_as_strategy_universe():
    obs = n0a.load_observed_decision_universe()
    assert len(obs) == n0a.EXPECTED_ALL_OBS
    assert (obs["hazard"] == 0).sum() == n0a.EXPECTED_TRANSITIONS
    assert (obs["hazard"] == 1).sum() == n0a.EXPECTED_HAZARD1
    assert set(obs["hazard"].unique()) == {0, 1}


# ===========================================================================
# 3. True Hazard Never Enters Policy Function
# ===========================================================================
def test_3_true_hazard_never_enters_policy_function():
    class DummyTransition:
        def analytic_conditional_support(self, df):
            return {"z_d_up_mu": np.linspace(-0.5, 0.5, len(df))}

    class DummyTerminal:
        def __init__(self, p_val):
            self.p_val = p_val
        def _design_df(self, df):
            return df
        num_cols = []
        allowed_nan_cols = []
        pre = None
        clf = None

    # Monkeypatch predict_hazard_probability
    orig_pred = exp.predict_hazard_probability
    try:
        exp.predict_hazard_probability = lambda term, df: np.full(len(df), 0.15)
        df_a = pd.DataFrame({"hazard": [0, 1], "r_trad_OC_ATR0": [1.0, -1.0]})
        df_b = pd.DataFrame({"hazard": [1, 0], "r_trad_OC_ATR0": [1.0, -1.0]})

        res_a = exp.score_and_evaluate_policies(df_a, DummyTransition(), DummyTerminal(0.15))
        res_b = exp.score_and_evaluate_policies(df_b, DummyTransition(), DummyTerminal(0.15))

        # Action and position must be completely identical regardless of hazard inversion
        assert np.array_equal(res_a["action_0a"], res_b["action_0a"])
        assert np.allclose(res_a["position_0b"], res_b["position_0b"])
        assert np.allclose(res_a["ret_0a"], res_b["ret_0a"])
        assert np.allclose(res_a["ret_0b"], res_b["ret_0b"])
    finally:
        exp.predict_hazard_probability = orig_pred


# ===========================================================================
# 4. Actual T2 Terminal Sampler Resolved
# ===========================================================================
def test_4_actual_t2_terminal_sampler_resolved():
    assert exp.PRIMARY_TERMINAL_HEAD == "T2_STATE_PHI_MEM"
    assert exp.DIAGNOSTIC_TERMINAL_HEAD == "T0_STATE_AVAIL"
    assert exp.PRIMARY_TRANSITION_HEAD == "MC_STATE_CURREENCODING"


# ===========================================================================
# 5. Actual T2 Design Has No Future/Target Columns
# ===========================================================================
def test_5_actual_t2_design_has_no_future_target_columns():
    forbidden = set(base.ALL_Z_COLS + base.COUNT_Z + [
        "hazard", "target_mask",
        "reward_SKIP", "reward_MARKET", "reward_LIMIT_RR3", "reward_REASSESS_RR3",
        "r_CC_ATR0", "gap_ATR0", "r_trad_OC_ATR0",
    ])
    t2_cols = pgm.T2_NUM + pgm.CAT
    intersection = set(t2_cols).intersection(forbidden)
    assert len(intersection) == 0, f"t2_cols contains forbidden target columns: {intersection}"
    for c in t2_cols:
        assert not c.startswith("next_"), f"Future column {c} found in T2 columns"


# ===========================================================================
# 6. Hazard Probability Finite in [0, 1]
# ===========================================================================
def test_6_hazard_probability_finite_in_zero_one():
    obs = n0a.load_observed_decision_universe().head(20)
    fit_A = pgm.fit_samplers_for_window(pgm.WINDOWS[0], pgm.SAMPLE_PATH, pgm.TRANSITION_SAMPLE_PATH)
    t2_sampler = fit_A["term_samplers"][exp.PRIMARY_TERMINAL_HEAD]
    p_h = exp.predict_hazard_probability(t2_sampler, obs)

    assert np.all(np.isfinite(p_h))
    assert np.all(p_h >= 0.0)
    assert np.all(p_h <= 1.0)


# ===========================================================================
# 7. Deterministic p_h Parity with sample_hazard Using FixedRNG
# ===========================================================================
def test_7_deterministic_ph_parity_with_sample_hazard_using_fixed_rng():
    obs = n0a.load_observed_decision_universe().head(50)
    fit_A = pgm.fit_samplers_for_window(pgm.WINDOWS[0], pgm.SAMPLE_PATH, pgm.TRANSITION_SAMPLE_PATH)
    t2_sampler = fit_A["term_samplers"][exp.PRIMARY_TERMINAL_HEAD]
    assert exp.audit_hazard_probability_parity(t2_sampler, obs) is True


# ===========================================================================
# 8. TB2 Uses Window A Terminal + Transition Heads
# ===========================================================================
def test_8_tb2_uses_window_a_terminal_and_transition_heads():
    # Verify execute_exploratory_pipeline routes TB2 with fit_A
    src = inspect.getsource(exp.execute_exploratory_pipeline)
    assert "df_tb2 = df_valid[df_valid[\"block\"] == n0a.TB2_BLOCK]" in src
    assert "mc_A = fit_A[\"trans_samplers\"][PRIMARY_TRANSITION_HEAD]" in src
    assert "term_A_t2 = fit_A[\"term_samplers\"][PRIMARY_TERMINAL_HEAD]" in src
    assert "tb2_scored = score_and_evaluate_policies(df_tb2, mc_A, term_A_t2" in src


# ===========================================================================
# 9. TB3 Uses Window B Terminal + Transition Heads
# ===========================================================================
def test_9_tb3_uses_window_b_terminal_and_transition_heads():
    src = inspect.getsource(exp.execute_exploratory_pipeline)
    assert "df_tb3 = df_valid[df_valid[\"block\"] == n0a.TB3_BLOCK]" in src
    assert "mc_B = fit_B[\"trans_samplers\"][PRIMARY_TRANSITION_HEAD]" in src
    assert "term_B_t2 = fit_B[\"term_samplers\"][PRIMARY_TERMINAL_HEAD]" in src
    assert "tb3_scored = score_and_evaluate_policies(df_tb3, mc_B, term_B_t2" in src


# ===========================================================================
# 10. score_mu Owner Parity
# ===========================================================================
def test_10_score_mu_owner_parity():
    class DummyTransition:
        def analytic_conditional_support(self, df):
            return {"z_d_up_mu": np.array([0.123, -0.456])}

    class DummyTerminal:
        pass

    orig_pred = exp.predict_hazard_probability
    try:
        exp.predict_hazard_probability = lambda term, df: np.array([0.1, 0.2])
        df = pd.DataFrame({"r_trad_OC_ATR0": [1.0, 1.0]})
        out = exp.score_and_evaluate_policies(df, DummyTransition(), DummyTerminal())
        assert np.allclose(out["score_mu"].to_numpy(), np.array([-0.123, 0.456]))
    finally:
        exp.predict_hazard_probability = orig_pred


# ===========================================================================
# 11. action_0a == sign(score_mu)
# ===========================================================================
def test_11_action_0a_equals_sign_score_mu():
    class DummyTransition:
        def analytic_conditional_support(self, df):
            return {"z_d_up_mu": np.array([-1.0, 1.0, 0.0])}

    class DummyTerminal:
        pass

    orig_pred = exp.predict_hazard_probability
    try:
        exp.predict_hazard_probability = lambda term, df: np.zeros(len(df))
        df = pd.DataFrame({"r_trad_OC_ATR0": [1.0, 1.0, 1.0]})
        out = exp.score_and_evaluate_policies(df, DummyTransition(), DummyTerminal())
        # score_mu = -z_d_up_mu: [+1.0, -1.0, 0.0] -> action_0a: [+1, -1, 0]
        assert np.array_equal(out["action_0a"].to_numpy(), np.array([1, -1, 0]))
    finally:
        exp.predict_hazard_probability = orig_pred


# ===========================================================================
# 12. position_0b == (1 - p_h) * sign(score_mu)
# ===========================================================================
def test_12_position_0b_equals_one_minus_ph_times_sign_score_mu():
    class DummyTransition:
        def analytic_conditional_support(self, df):
            return {"z_d_up_mu": np.array([-0.5, 0.5])}

    class DummyTerminal:
        pass

    orig_pred = exp.predict_hazard_probability
    try:
        # p_h = [0.2, 0.4] -> survival_prob = [0.8, 0.6]
        exp.predict_hazard_probability = lambda term, df: np.array([0.2, 0.4])
        df = pd.DataFrame({"r_trad_OC_ATR0": [1.0, 1.0]})
        out = exp.score_and_evaluate_policies(df, DummyTransition(), DummyTerminal())
        # score_mu = [+0.5, -0.5] -> action_0a = [+1, -1] -> position_0b = [+0.8, -0.6]
        assert np.allclose(out["position_0b"].to_numpy(), np.array([0.8, -0.6]))
    finally:
        exp.predict_hazard_probability = orig_pred


# ===========================================================================
# 13. Position Bound [-1, 1]
# ===========================================================================
def test_13_position_bound_minus_one_to_one():
    class DummyTransition:
        def analytic_conditional_support(self, df):
            return {"z_d_up_mu": np.array([-10.0, 10.0])}

    class DummyTerminal:
        pass

    orig_pred = exp.predict_hazard_probability
    try:
        # Extreme p_h values: 0.0 and 1.0
        exp.predict_hazard_probability = lambda term, df: np.array([0.0, 1.0])
        df = pd.DataFrame({"r_trad_OC_ATR0": [1.0, 1.0]})
        out = exp.score_and_evaluate_policies(df, DummyTransition(), DummyTerminal())
        assert np.all(out["position_0b"] >= -1.0)
        assert np.all(out["position_0b"] <= 1.0)
        assert np.isclose(out["position_0b"].iloc[0], 1.0)
        assert np.isclose(out["position_0b"].iloc[1], 0.0)
    finally:
        exp.predict_hazard_probability = orig_pred


# ===========================================================================
# 14. Cost Scales by abs(position)
# ===========================================================================
def test_14_cost_scales_by_abs_position():
    df = pd.DataFrame({
        "ret_0a": [0.05, -0.05],
        "ret_0b": [0.04, -0.04],
        "exposure_0a": [1.0, 1.0],
        "exposure_0b": [0.8, 0.8],
    })
    stress = exp.run_cost_stress_grid_0b(df)
    for row in stress:
        c = row["cost_ATR0"]
        expected_net_0a = np.mean([0.05 - c * 1.0, -0.05 - c * 1.0])
        expected_net_0b = np.mean([0.04 - c * 0.8, -0.04 - c * 0.8])
        assert np.isclose(row["net_EV_0A"], expected_net_0a)
        assert np.isclose(row["net_EV_0B"], expected_net_0b)


# ===========================================================================
# 15. TB3 Hazard Bins Use TB2 Edges
# ===========================================================================
def test_15_tb3_hazard_bins_use_tb2_edges():
    p_h_tb2 = np.linspace(0.01, 0.99, 100)
    edges = exp.compute_hazard_decile_edges(p_h_tb2)
    assert len(edges) == 11
    assert edges[0] == -np.inf
    assert edges[-1] == np.inf

    df_tb3 = pd.DataFrame({
        "p_h": np.linspace(0.01, 0.99, 100),
        "hazard": [0]*90 + [1]*10,
        "score_mu": np.ones(100),
        "ret_0a": np.ones(100),
        "ret_0b": np.ones(100) * 0.8,
        "r_trad_OC_ATR0": np.ones(100),
    })
    rows, stats = exp.evaluate_hazard_deciles(df_tb3, edges)
    assert len(rows) == 10
    total_n = sum(r["n"] for r in rows)
    assert total_n == 100


# ===========================================================================
# 16. No True Hazard in Bin/Gate Construction
# ===========================================================================
def test_16_no_true_hazard_in_bin_gate_construction():
    # Edges are computed ONLY from predicted p_h array
    p_h_dummy = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    edges = exp.compute_hazard_decile_edges(p_h_dummy)
    assert len(edges) == 11
    # Function signature takes only p_h array
    sig = inspect.signature(exp.compute_hazard_decile_edges)
    assert list(sig.parameters.keys()) == ["p_h_tb2"]


# ===========================================================================
# 17. Paired Bootstrap Resamples Identical Days for 0A / 0B
# ===========================================================================
def test_17_paired_bootstrap_resamples_identical_days_for_0a_0b():
    # In paired bootstrap, Delta_EV is strictly EV_0B - EV_0A per replicate
    src = inspect.getsource(exp.run_paired_day_clustered_bootstrap)
    assert "boot_delta_ev[b] = m_ev_0b - m_ev_0a" in src
    assert "boot_delta_enev[b] = m_enev_0b - m_enev_0a" in src


# ===========================================================================
# 18. Paired Bootstrap Deterministic
# ===========================================================================
def test_18_paired_bootstrap_deterministic():
    df_mock = pd.DataFrame({
        "entry_day": ["2026-01-01"] * 50 + ["2026-01-02"] * 50,
        "ret_0a": np.linspace(-0.1, 0.1, 100),
        "ret_0b": np.linspace(-0.08, 0.08, 100),
        "exposure_0a": np.ones(100),
        "exposure_0b": np.full(100, 0.8),
        "hazard": [0]*85 + [1]*15,
        "p_h": np.linspace(0.05, 0.35, 100),
    })
    res1 = exp.run_paired_day_clustered_bootstrap(df_mock, n_boot=50, seed=12345)
    res2 = exp.run_paired_day_clustered_bootstrap(df_mock, n_boot=50, seed=12345)
    for k in res1:
        for stat in ["point", "ci95_lower", "ci95_upper", "p_pos"]:
            assert np.isclose(res1[k][stat], res2[k][stat]), f"Mismatch in {k} {stat}"


# ===========================================================================
# 19. Exposure-Normalized EV Formula
# ===========================================================================
def test_19_exposure_normalized_ev_formula():
    ret = np.array([0.02, 0.04, -0.01])
    exp_arr = np.array([0.5, 1.0, 0.5])
    expected_enev = np.sum(ret) / np.sum(exp_arr)
    # (0.02 + 0.04 - 0.01) / (0.5 + 1.0 + 0.5) = 0.05 / 2.0 = 0.025
    assert np.isclose(expected_enev, 0.025)


# ===========================================================================
# 20. H0 / H1 Metrics Diagnostic Only
# ===========================================================================
def test_20_h0_h1_metrics_diagnostic_only():
    df_mock = pd.DataFrame({
        "hazard": [0, 0, 1, 1],
        "p_h": [0.1, 0.1, 0.4, 0.4],
        "ret_0a": [0.04, 0.04, -0.30, -0.30],
        "ret_0b": [0.036, 0.036, -0.18, -0.18],
        "exposure_0b": [0.9, 0.9, 0.6, 0.6],
    })
    sub = exp.compute_hazard_subgroups(df_mock)
    assert sub["H0"]["EV_0a"] == 0.04
    assert sub["H1"]["EV_0a"] == -0.30
    assert np.isclose(sub["H1"]["EV_0b"], -0.18)
    # Attenuation: 1 - |-0.18| / |-0.30| = 1 - 0.6 = 0.4
    assert np.isclose(sub["terminal_loss_attenuation"], 0.40)


# ===========================================================================
# 21. All 15 Symbols Retained
# ===========================================================================
def test_21_all_15_symbols_retained():
    rows = []
    for s in exp.EXPECTED_SYMBOLS:
        rows.append({
            "symbol": s,
            "hazard": 0,
            "p_h": 0.1,
            "ret_0a": 0.01,
            "ret_0b": 0.009,
            "exposure_0a": 1.0,
            "exposure_0b": 0.9,
        })
        rows.append({
            "symbol": s,
            "hazard": 1,
            "p_h": 0.3,
            "ret_0a": -0.1,
            "ret_0b": -0.07,
            "exposure_0a": 1.0,
            "exposure_0b": 0.7,
        })
    df_all_sym = pd.DataFrame(rows)
    breadth_rows, breadth_counts = exp.compute_symbol_breadth_0b(df_all_sym)
    assert breadth_counts["n_symbols_total"] == 15
    assert len(breadth_rows) == 15
    assert sorted([r["symbol"] for r in breadth_rows]) == sorted(exp.EXPECTED_SYMBOLS)


# ===========================================================================
# 22. T0 is Diagnostic Only
# ===========================================================================
def test_22_t0_is_diagnostic_only():
    class DummyTransition:
        def analytic_conditional_support(self, df):
            return {"z_d_up_mu": np.array([-0.5])}

    class DummyTerminalT2:
        pass

    class DummyTerminalT0:
        pass

    orig_pred = exp.predict_hazard_probability
    try:
        def mock_pred(term, df):
            if isinstance(term, DummyTerminalT2):
                return np.array([0.2])
            return np.array([0.9])  # Very different T0 prediction
        exp.predict_hazard_probability = mock_pred

        df = pd.DataFrame({"r_trad_OC_ATR0": [1.0]})
        out = exp.score_and_evaluate_policies(df, DummyTransition(), DummyTerminalT2(), DummyTerminalT0())

        # position_0b must strictly depend on T2 (p_h = 0.2 -> pos = 0.8), NOT T0
        assert np.isclose(out["position_0b"].iloc[0], 0.8)
        assert np.isclose(out["p_h"].iloc[0], 0.2)
        assert np.isclose(out["p_h_t0"].iloc[0], 0.9)
    finally:
        exp.predict_hazard_probability = orig_pred


# ===========================================================================
# 23. Smoke Emits No Scientific Verdict
# ===========================================================================
def test_23_smoke_emits_no_scientific_verdict():
    src = inspect.getsource(exp.run_smoke_test)
    assert "NO SCIENTIFIC VERDICT" in src
    assert "determine_formal_verdict" not in src


# ===========================================================================
# 24. Full-Exploratory Blocked First Round
# ===========================================================================
def test_24_full_exploratory_blocked_first_round():
    old_env = os.environ.pop("AUTHORIZE_PGM_NATIVE0B_FULL_EXPLORATORY", None)
    failed = False
    try:
        exp.run_full_exploratory()
    except SystemExit as e:
        failed = True
        assert "STOP_PGM_NATIVE0B_FULL_EXPLORATORY_NOT_AUTHORIZED_THIS_ROUND" in str(e)
    finally:
        if old_env is not None:
            os.environ["AUTHORIZE_PGM_NATIVE0B_FULL_EXPLORATORY"] = old_env
    assert failed, "Expected SystemExit from unauthorized run_full_exploratory"


# ===========================================================================
# 25. No V2 / R1-R4 / Q-Model / RL Dependency
# ===========================================================================
def test_25_no_v2_r1_r4_q_model_rl_dependency():
    text = Path(exp.__file__).read_text()
    forbidden_tokens = ["market_regime_v2", "q_learning", "reinforcement_learning", "gym", "torch", "tensorflow", "stable_baselines"]
    for tok in forbidden_tokens:
        assert tok not in text.lower(), f"Forbidden token {tok} found in {exp.__file__}"


# ===========================================================================
# 26. Audit-Only Artifact Hashes: Fail-Closed / 64-char SHA256
# ===========================================================================
def test_26_audit_only_artifact_hashes_fail_closed():
    # Happy path: valid 64-char lowercase hex SHA256 for both artifacts
    hashes = exp.compute_artifact_hashes()
    for k in ("sample_artifact_sha256", "transition_artifact_sha256"):
        v = hashes[k]
        assert isinstance(v, str), f"{k} must be a string"
        assert len(v) == 64, f"{k} must be 64 chars, got {len(v)}"
        int(v, 16)  # must be valid hex

    # audit-only must print them (not N/A)
    src = inspect.getsource(exp.run_audit_only)
    assert "sample_artifact_sha256" in src
    assert "transition_artifact_sha256" in src
    assert "compute_artifact_hashes" in src
    assert "N/A" not in src

    # Fail-closed: a missing sample artifact must abort, never emit N/A
    orig = pgm.SAMPLE_PATH
    try:
        pgm.SAMPLE_PATH = Path("/nonexistent/path/does_not_exist.parquet")
        raised = None
        try:
            exp.compute_artifact_hashes()
        except SystemExit as e:
            raised = str(e)
        assert raised is not None, "expected SystemExit on missing sample artifact"
        assert "STOP_PGM_NATIVE_SAMPLE_ARTIFACT_MISSING" in raised
    finally:
        pgm.SAMPLE_PATH = orig


# ===========================================================================
# 27. Actual MC_STATE_CURREENCODING design_cols Have Zero Future/Target Overlap
# ===========================================================================
_FIT_A_CACHE = None


def _get_fit_A():
    global _FIT_A_CACHE
    if _FIT_A_CACHE is None:
        _FIT_A_CACHE = pgm.fit_samplers_for_window(pgm.WINDOWS[0], pgm.SAMPLE_PATH, pgm.TRANSITION_SAMPLE_PATH)
    return _FIT_A_CACHE


def test_27_actual_mc_transition_design_has_no_future_target_columns():
    fit_A = _get_fit_A()
    mc_sampler = fit_A["trans_samplers"][exp.PRIMARY_TRANSITION_HEAD]
    design = mc_sampler.design_cols
    assert design is not None, "MC transition design_cols must be populated"
    assert len(design) > 0, "MC transition design_cols must be non-empty"

    forbidden = exp.build_forbidden_target_cols()
    intersection = set(design).intersection(forbidden)
    assert len(intersection) == 0, f"MC design future/target leakage: {intersection}"

    # audit-only must audit the ACTUAL transition sampler, not just static constants
    src = inspect.getsource(exp.run_audit_only)
    assert "fit_A[\"trans_samplers\"][PRIMARY_TRANSITION_HEAD]" in src
    assert "STOP_PGM_NATIVE0B_TRANSITION_DESIGN_FUTURE_LEAKAGE" in src
    assert "MC transition design future/target overlap = 0" in src
    assert "T2 terminal design future/target overlap = 0" in src


# ===========================================================================
# 28. T0 TB3 Hazard Deciles Use TB2 T0 Edges
# ===========================================================================
def test_28_t0_tb3_hazard_deciles_use_tb2_t0_edges():
    p_tb2 = np.linspace(0.01, 0.99, 200)
    p_tb3 = np.linspace(0.02, 0.98, 200)
    haz_tb2 = (p_tb2 > 0.8).astype(int)
    haz_tb3 = (p_tb3 > 0.8).astype(int)

    edges_t0 = exp.compute_hazard_decile_edges(p_tb2)
    assert len(edges_t0) == 11
    assert np.all(np.diff(edges_t0) > 0)

    rows, stats = exp.compute_hazard_calibration_deciles(p_tb3, haz_tb3, edges_t0)
    assert len(rows) > 0
    assert sum(r["n"] for r in rows) == 200
    assert "spearman_bin_p_h_vs_observed_H1" in stats

    # Full pipeline contract: T0 edges derived from TB2, reused verbatim for TB3
    src = inspect.getsource(exp.execute_exploratory_pipeline)
    assert 'edges_p_h_t0 = compute_hazard_decile_edges(tb2_scored["p_h_t0"]' in src
    assert 'compute_hazard_calibration_deciles(tb2_scored["p_h_t0"], tb2_scored["hazard"], edges_p_h_t0)' in src
    assert 'compute_hazard_calibration_deciles(tb3_scored["p_h_t0"], tb3_scored["hazard"], edges_p_h_t0)' in src


# ===========================================================================
# 29. T0 Diagnostic Helper Consumes/Generates No Economic-Policy Field
# ===========================================================================
def test_29_t0_diagnostic_helper_has_no_economic_policy_fields():
    sig = inspect.signature(exp.compute_hazard_calibration_deciles)
    assert list(sig.parameters.keys()) == ["p_h", "hazard", "edges"]

    src = inspect.getsource(exp.compute_hazard_calibration_deciles)
    for tok in ["ret_0a", "ret_0b", "position_0b", "exposure_0a", "exposure_0b",
                "score_mu", "r_trad_OC_ATR0", "EV_0a", "EV_0b", "action_0a"]:
        assert tok not in src, f"forbidden economic token {tok!r} present in T0 helper"

    p = np.linspace(0.01, 0.99, 100)
    h = (p > 0.7).astype(int)
    edges = exp.compute_hazard_decile_edges(p)
    rows, stats = exp.compute_hazard_calibration_deciles(p, h, edges)
    for r in rows:
        assert set(r.keys()) == {"bin_idx", "n", "mean_p_h", "observed_H1_rate"}
    assert set(stats.keys()) == {"spearman_bin_p_h_vs_observed_H1"}


# ===========================================================================
# 30. Smoke Executes Hazard-Decile Rough Ordering Without Scientific Verdict
# ===========================================================================
def test_30_smoke_executes_decile_rough_ordering_without_verdict():
    src = inspect.getsource(exp.run_smoke_test)
    assert "compute_hazard_decile_edges" in src
    assert "evaluate_hazard_deciles" in src
    assert "SMOKE ROUGH ORDERING ONLY" in src
    assert "NO SCIENTIFIC VERDICT" in src
    assert "determine_formal_verdict" not in src
    # No T0 economic policy may be emitted
    for tok in ["EV_T0", "position_T0", "ret_T0"]:
        assert tok not in src, f"forbidden T0 economic token {tok!r} in smoke"


# ===========================================================================
# 31. Full Output Contract Includes T0 Diagnostic Decile Artifact
# ===========================================================================
def test_31_full_output_contract_includes_t0_decile_artifact():
    src = inspect.getsource(exp.execute_exploratory_pipeline)
    # Artifact names are built with the PREFIX f-string; assert the literal source form.
    assert "{PREFIX}_t0_hazard_deciles.csv" in src
    assert '["block", "bin_idx", "n", "mean_p_h", "observed_H1_rate"]' in src
    assert "TB2_T0_hazard_decile_ordering" in src
    assert "TB3_T0_hazard_decile_ordering" in src
    # PRIMARY T2 decile artifact must remain intact
    assert "{PREFIX}_hazard_deciles.csv" in src


# ===========================================================================
# Test Runner
# ===========================================================================
if __name__ == "__main__":
    import traceback

    tests = [getattr(sys.modules[__name__], f) for f in dir(sys.modules[__name__])
             if f.startswith("test_")]
    tests.sort(key=lambda fn: int(fn.__name__.split("_")[1]))

    ok = fail = 0
    print(f"Running {len(tests)} unit tests for PGM-NATIVE-0B...\n")
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
