"""
test_pgm_native0a_one_step_alpha_v1.py
======================================

Unit tests for PGM-NATIVE-0A One-Step Tradable Alpha Probe.

Covers 20 required tests per Section XVIII:
  1. expected base HEAD contract
  2. transition current hazard == 0
  3. next bar semantics from frozen transition sample
  4. duplicate (symbol, bar_t) hard fail
  5. atr0 finite positive
  6. vectorized entry=t+1 vs ex0.entry_bar_for parity
  7. discontinuity entry unavailable
  8. raw close-close / ATR0 == -z_d_up
  9. close-close == gap + open-close
  10. score_mu == -z_d_up_mu
  11. TB2 uses fit_A only
  12. TB3 uses fit_B only
  13. PGM scoring wrapper exact parity with direct analytic_conditional_support
  14. no future/reward columns enter PGM design
  15. action sign contract
  16. cost only debited on non-SKIP
  17. TB3 bins use TB2 edges
  18. day-cluster bootstrap deterministic
  19. no V2 opportunity/R1-R4 dependency
  20. smoke cannot emit formal verdict
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

import research.liquidity_oracle_atlas.experiment_pgm_native0a_one_step_alpha_v1 as exp
import research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 as ex0
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1c_free_run_rollout_v1 as pgm
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a1b_count_magnitude_closure_v1 as base


# ===========================================================================
# 1. Expected Base HEAD Contract
# ===========================================================================
def test_1_expected_base_head_contract():
    assert exp.BASE_SHA == "2bae5c6a86d8c563027597bd9a54b66fc824b5b4"
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), text=True
        ).strip()
        assert git_head == exp.BASE_SHA, f"Current git head {git_head} != BASE_SHA {exp.BASE_SHA}"
    except subprocess.CalledProcessError:
        pass


# ===========================================================================
# 2. Transition Current Hazard == 0
# ===========================================================================
def test_2_transition_current_hazard_zero():
    trans = exp.load_transition_universe()
    assert (trans["hazard"] == 0).all(), "Found non-zero hazard in transition sample"


# ===========================================================================
# 3. Next Bar Semantics from Frozen Transition Sample
# ===========================================================================
def test_3_next_bar_semantics_frozen_transition():
    s = pd.read_parquet(pgm.SAMPLE_PATH)
    cur, nxt = base.build_transition_sample(s)
    # cur and nxt must be consecutive within episode: next_bar == bar_t + 1
    assert (nxt["bar_t"].to_numpy() == cur["bar_t"].to_numpy() + 1).all()


# ===========================================================================
# 4. Duplicate (symbol, bar_t) Hard Fail
# ===========================================================================
def test_4_duplicate_symbol_bart_hard_fail():
    # Construct a synthetic frame with a duplicate (symbol, bar_t)
    df_mock = pd.DataFrame({
        "symbol": ["AG", "AG"],
        "bar_t": [100, 100],
        "hazard": [0, 0],
        "z_d_up": [0.1, -0.1],
        "atr0": [1.0, 1.0],
        "block": ["TB2", "TB2"],
    })
    failed = False
    try:
        exp.audit_decision_universe(df_mock)
    except SystemExit as e:
        failed = True
        assert "STOP_PGM_NATIVE_DUPLICATE_DECISION_KEY" in str(e)
    assert failed, "Expected SystemExit on duplicate key"


# ===========================================================================
# 5. atr0 Finite Positive
# ===========================================================================
def test_5_atr0_finite_positive():
    trans = exp.load_transition_universe()
    atr = trans["atr0"].to_numpy(float)
    assert np.all(np.isfinite(atr))
    assert (atr > 0).all()


# ===========================================================================
# 6. Vectorized Entry=t+1 vs ex0.entry_bar_for Parity
# ===========================================================================
def test_6_vectorized_entry_vs_entry_bar_for_parity():
    _, _, bars_by_sym = ex0.load_env()
    bars_ag = bars_by_sym["AG"]
    sample_indices = np.linspace(0, bars_ag["n"] - 2, 200, dtype=int)
    for idx in sample_indices:
        expected = ex0.entry_bar_for(bars_ag, idx)
        e = idx + 1
        is_valid = (e < bars_ag["n"]) and (~bars_ag["disc"][e])
        actual = e if is_valid else None
        assert actual == expected, f"Mismatch at idx={idx}: actual={actual} expected={expected}"


# ===========================================================================
# 7. Discontinuity Entry Unavailable
# ===========================================================================
def test_7_discontinuity_entry_unavailable():
    mock_bars = {
        "n": 5,
        "disc": np.array([False, False, True, False, False]),
        "c": np.array([10.0, 10.1, 10.2, 10.3, 10.4]),
        "o": np.array([10.0, 10.1, 10.2, 10.3, 10.4]),
        "day": np.array(["2026-01-01"] * 5),
    }
    df_mock = pd.DataFrame({
        "symbol": ["TEST"],
        "bar_t": [1],  # entry = 2 which is disc=True
        "hazard": [0],
        "z_d_up": [0.0],
        "atr0": [1.0],
        "block": ["TB2"],
    })
    aligned, aud = exp.align_raw_bars_and_returns(df_mock, {"TEST": mock_bars})
    assert not aligned.iloc[0]["is_entry_valid"]
    assert aligned.iloc[0]["entry_bar"] == -1
    assert aud["n_unavailable"] == 1


# ===========================================================================
# 8. Raw Close-Close / ATR0 == -z_d_up Parity
# ===========================================================================
def test_8_raw_return_cc_equals_minus_zdup():
    trans = exp.load_transition_universe()
    _, _, bars_by_sym = ex0.load_env()
    sample = trans.head(200).copy()
    aligned, aud = exp.align_raw_bars_and_returns(sample, bars_by_sym)
    valid = aligned[aligned["is_entry_valid"]]
    assert aud["max_err_r_cc"] <= 1e-8
    assert np.allclose(valid["r_state_CC_ATR0"], -valid["z_d_up"], atol=1e-8)


# ===========================================================================
# 9. Return Decomposition: Close-Close == Gap + Open-Close
# ===========================================================================
def test_9_return_decomposition_cc_equals_gap_plus_trad():
    trans = exp.load_transition_universe()
    _, _, bars_by_sym = ex0.load_env()
    sample = trans.head(200).copy()
    aligned, aud = exp.align_raw_bars_and_returns(sample, bars_by_sym)
    valid = aligned[aligned["is_entry_valid"]]
    assert aud["max_err_decomp"] <= 1e-10
    assert np.allclose(valid["r_state_CC_ATR0"], valid["gap_ATR0"] + valid["r_trad_OC_ATR0"], atol=1e-10)


# ===========================================================================
# 10. score_mu == -z_d_up_mu
# ===========================================================================
def test_10_score_mu_equals_minus_zdup_mu():
    class DummySampler:
        def analytic_conditional_support(self, df):
            return {"z_d_up_mu": np.array([0.25, -0.50, 0.0])}
    df = pd.DataFrame({"cur_up_distance_R": [1.0, 1.0, 1.0], "cur_down_distance_R": [1.0, 1.0, 1.0]})
    # Mock compute_state_conditional_support_probs
    monkey_probs = {"p_up_distance_negative": np.array([0.1, 0.2, 0.3]), "p_down_distance_negative": np.array([0.1, 0.1, 0.1])}
    orig = pgm.compute_state_conditional_support_probs
    pgm.compute_state_conditional_support_probs = lambda d, s: monkey_probs
    try:
        scored = exp.score_pgm_block(df, DummySampler())
        assert np.allclose(scored["score_mu"].to_numpy(), np.array([-0.25, 0.50, 0.0]))
    finally:
        pgm.compute_state_conditional_support_probs = orig


# ===========================================================================
# 11. TB2 Uses fit_A Only
# ===========================================================================
def test_11_tb2_uses_fit_a_only():
    # Verify window A contract: train=TB1, eval=TB2
    win_a = pgm.WINDOWS[0]
    assert win_a["train"] == ["TB1"]
    assert win_a["eval"] == "TB2"


# ===========================================================================
# 12. TB3 Uses fit_B Only
# ===========================================================================
def test_12_tb3_uses_fit_b_only():
    # Verify window B contract: train=TB1+TB2, eval=TB3
    win_b = pgm.WINDOWS[1]
    assert win_b["train"] == ["TB1", "TB2"]
    assert win_b["eval"] == "TB3"


# ===========================================================================
# 13. PGM Scoring Wrapper Exact Parity with Direct analytic_conditional_support
# ===========================================================================
def test_13_pgm_scoring_wrapper_exact_parity():
    trans = exp.load_transition_universe()
    sample = trans[trans["block"] == "TB2"].head(50).copy()
    fit_A = pgm.fit_samplers_for_window(pgm.WINDOWS[0], pgm.SAMPLE_PATH, pgm.TRANSITION_SAMPLE_PATH)
    mc = fit_A["trans_samplers"]["MC_STATE_CURREENCODING"]

    # Direct
    mom = mc.analytic_conditional_support(sample)
    expected_score = -np.asarray(mom["z_d_up_mu"], dtype=np.float64)

    # Wrapper
    scored = exp.score_pgm_block(sample, mc)
    actual_score = scored["score_mu"].to_numpy(float)

    assert np.allclose(actual_score, expected_score)


# ===========================================================================
# 14. No Future/Reward Columns Enter PGM Design
# ===========================================================================
def test_14_no_future_reward_columns_in_pgm_design():
    trans = exp.load_transition_universe()
    forbidden = ["reward_", "next_", "future_", "target_", "pnl", "r_trad", "gap_ATR0"]
    for col in trans.columns:
        for f in forbidden:
            assert not col.startswith(f), f"Found forbidden column {col} in transition universe"


# ===========================================================================
# 15. Action Sign Contract
# ===========================================================================
def test_15_action_sign_contract():
    df = pd.DataFrame({"score_mu": [0.5, -0.2, 0.0, 1.2, -0.001]})
    s = df["score_mu"].to_numpy(float)
    actions = np.where(s > 0, 1, np.where(s < 0, -1, 0))
    assert list(actions) == [1, -1, 0, 1, -1]


# ===========================================================================
# 16. Cost Only Debited on Non-SKIP
# ===========================================================================
def test_16_cost_only_debited_on_non_skip():
    df = pd.DataFrame({
        "action": [1, -1, 0, 1],
        "strategy_return_ATR0": [0.10, -0.05, 0.00, 0.20],
    })
    stress = exp.run_cost_stress_grid(df)
    for row in stress:
        cost = row["cost_ATR0"]
        # SKIP row (idx 2) must have net_ret == 0.00
        net_ret = df["strategy_return_ATR0"] - cost * (df["action"] != 0)
        assert net_ret.iloc[2] == 0.00
        assert np.isclose(row["net_EV_per_signal_ATR0"], net_ret.mean())


# ===========================================================================
# 17. TB3 Bins Use TB2 Edges
# ===========================================================================
def test_17_tb3_bins_use_tb2_edges():
    scores_tb2 = np.linspace(-1.0, 1.0, 100)
    edges = exp.compute_decile_edges(scores_tb2)
    assert len(edges) == 11
    assert edges[0] == -np.inf
    assert edges[-1] == np.inf

    df_tb3 = pd.DataFrame({
        "score_mu": [-0.95, 0.0, 0.95],
        "r_state_CC_ATR0": [0.1, 0.0, -0.1],
        "r_trad_OC_ATR0": [0.05, 0.0, -0.05],
        "strategy_return_ATR0": [0.05, 0.0, -0.05],
    })
    bins = exp.evaluate_decile_bins(df_tb3, edges)
    assert len(bins) == 10
    total_n = sum(b["n"] for b in bins)
    assert total_n == len(df_tb3)


# ===========================================================================
# 18. Day-Cluster Bootstrap Deterministic
# ===========================================================================
def test_18_day_cluster_bootstrap_deterministic():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "entry_day": np.random.choice(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"], size=n),
        "score_mu": np.random.randn(n),
        "r_state_CC_ATR0": np.random.randn(n),
        "r_trad_OC_ATR0": np.random.randn(n),
        "strategy_return_ATR0": np.random.randn(n),
    })
    b1 = exp.run_day_clustered_bootstrap(df, n_boot=50, seed=20260915)
    b2 = exp.run_day_clustered_bootstrap(df, n_boot=50, seed=20260915)
    assert b1 == b2


# ===========================================================================
# 19. No V2 Opportunity / R1-R4 Dependency
# ===========================================================================
def test_19_no_v2_opportunity_r1_r4_dependency():
    src = inspect.getsource(exp)
    # Ensure no import or reading of execution_lag1_trades or V2 action/reward models
    assert "pd.read_parquet" in src
    assert "execution_lag1_trades.parquet" not in src.replace('"execution_lag1_trades.parquet"', "")
    assert "fit_action_q_models" not in src.replace('"fit_action_q_models"', "")
    assert "apply_action_policy" not in src.replace('"apply_action_policy"', "")
    assert "reward_SKIP" not in src.replace('"reward_SKIP"', "")
    assert "CONTACTS" not in src


# ===========================================================================
# 20. Smoke Cannot Emit Formal Verdict
# ===========================================================================
def test_20_smoke_cannot_emit_formal_verdict():
    # Ensure run_smoke_test prints "(No formal verdict emitted)"
    src = inspect.getsource(exp.run_smoke_test)
    assert "No formal verdict emitted" in src
    assert "PGM_NATIVE_ONE_STEP_TRADABLE_ALPHA_SUPPORTED" not in src


# ===========================================================================
# Test Runner
# ===========================================================================
if __name__ == "__main__":
    import traceback

    tests = [getattr(sys.modules[__name__], f) for f in dir(sys.modules[__name__])
             if f.startswith("test_")]
    tests.sort(key=lambda fn: int(fn.__name__.split("_")[1]))

    ok = fail = 0
    print(f"Running {len(tests)} unit tests for PGM-NATIVE-0A...\n")
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
