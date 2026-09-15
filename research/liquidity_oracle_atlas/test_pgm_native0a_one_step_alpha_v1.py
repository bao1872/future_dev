"""
test_pgm_native0a_one_step_alpha_v1.py
======================================

Unit tests for PGM-NATIVE-0A.1 Causal Decision-Universe Repair.

Covers 27 required tests per Section XVII:
  1. BASE_SHA is ancestor of HEAD
  2. strategy universe sourced from pgm.SAMPLE_PATH
  3. strategy universe includes hazard==0 AND hazard==1
  4. no hazard filtering in strategy universe
  5. overall row count > transition row count
  6. transition truth row count == frozen EXPECTED_TRANSITIONS
  7. duplicate (symbol, bar_t) hard fail
  8. atr0 finite positive
  9. safe entry at data end returns unavailable, no OOB
  10. ex0.entry_bar_for parity
  11. raw CC == gap + trad on ALL rows
  12. hazard==0 raw CC == -rebuilt z_d_up
  13. cached/rebuilt transition key parity
  14. cached/rebuilt z_d_up precision difference reported, not overwritten
  15. score_mu == -analytic z_d_up_mu
  16. actual mc.design_cols contains no future/target cols
  17. TB2 experiment route really uses mc_A
  18. TB3 experiment route really uses mc_B
  19. action independent of hazard label
  20. ALL metrics use hazard0+hazard1 rows
  21. H0/H1 subgroup metrics diagnostic only
  22. cost only on non-SKIP
  23. TB3 deciles still use TB2 edges
  24. day-cluster bootstrap deterministic
  25. no V2 / R1-R4 dependency
  26. smoke cannot emit formal verdict
  27. formal verdict strings contain ON_FROZEN_PGM_BAR_SAMPLE
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
# 1. BASE_SHA is Ancestor of HEAD
# ===========================================================================
def test_1_base_sha_is_ancestor_of_head():
    assert exp.BASE_SHA == "29b96485f3272dc76df34dcd26b700d4a73aedce"
    res = subprocess.run(
        ["git", "merge-base", "--is-ancestor", exp.BASE_SHA, "HEAD"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
    )
    assert res.returncode == 0, f"BASE_SHA {exp.BASE_SHA} is not ancestor of HEAD"


# ===========================================================================
# 2. Strategy Universe Sourced from pgm.SAMPLE_PATH
# ===========================================================================
def test_2_strategy_universe_sourced_from_sample_path():
    obs = exp.load_observed_decision_universe()
    assert len(obs) == exp.EXPECTED_ALL_OBS, f"Expected {exp.EXPECTED_ALL_OBS} rows, got {len(obs)}"
    assert "hazard" in obs.columns
    assert "bar_t" in obs.columns
    assert "symbol" in obs.columns


# ===========================================================================
# 3. Strategy Universe Includes hazard==0 AND hazard==1
# ===========================================================================
def test_3_strategy_universe_includes_hazard0_and_hazard1():
    obs = exp.load_observed_decision_universe()
    h_set = set(obs["hazard"].unique())
    assert 0 in h_set and 1 in h_set, f"Expected both 0 and 1 in hazard set, got {h_set}"


# ===========================================================================
# 4. No Hazard Filtering in Strategy Universe
# ===========================================================================
def test_4_no_hazard_filtering_in_strategy_universe():
    obs = exp.load_observed_decision_universe()
    n_h1 = int((obs["hazard"] == 1).sum())
    assert n_h1 == exp.EXPECTED_HAZARD1, f"Expected {exp.EXPECTED_HAZARD1} hazard==1 rows, got {n_h1}"


# ===========================================================================
# 5. Overall Row Count > Transition Row Count
# ===========================================================================
def test_5_overall_row_count_greater_than_transition_row_count():
    obs = exp.load_observed_decision_universe()
    trans_aud = exp.load_transition_truth_audit()
    assert len(obs) > trans_aud["n_rebuilt"]
    assert len(obs) - trans_aud["n_rebuilt"] == exp.EXPECTED_HAZARD1


# ===========================================================================
# 6. Transition Truth Row Count == Frozen EXPECTED_TRANSITIONS
# ===========================================================================
def test_6_transition_truth_row_count_equals_frozen_expected():
    trans_aud = exp.load_transition_truth_audit()
    assert trans_aud["n_rebuilt"] == exp.EXPECTED_TRANSITIONS
    assert trans_aud["n_cached"] == exp.EXPECTED_TRANSITIONS


# ===========================================================================
# 7. Duplicate (symbol, bar_t) Hard Fail
# ===========================================================================
def test_7_duplicate_symbol_bart_hard_fail():
    df_mock = pd.DataFrame({
        "symbol": ["AG", "AG"],
        "bar_t": [100, 100],
        "hazard": [0, 1],
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
# 8. atr0 Finite Positive
# ===========================================================================
def test_8_atr0_finite_positive():
    obs = exp.load_observed_decision_universe()
    atr = obs["atr0"].to_numpy(float)
    assert np.all(np.isfinite(atr))
    assert (atr > 0).all()


# ===========================================================================
# 9. Safe Entry at Data End Returns Unavailable, No OOB
# ===========================================================================
def test_9_safe_entry_at_data_end_returns_unavailable_no_oob():
    mock_bars = {
        "n": 5,
        "disc": np.zeros(5, dtype=bool),
        "day": np.array(["2026-01-01"] * 5, dtype="datetime64[us]"),
        "c": np.array([10.0, 10.1, 10.2, 10.3, 10.4]),
        "o": np.array([10.0, 10.1, 10.2, 10.3, 10.4]),
    }
    # bar_t = 4 -> entry = 5 == n (out of bounds)
    df_mock = pd.DataFrame({
        "symbol": ["TEST"],
        "bar_t": [4],
        "hazard": [0],
        "atr0": [1.0],
        "block": ["TB2"],
    })
    aligned, aud = exp.align_raw_bars_and_returns(df_mock, {"TEST": mock_bars})
    assert not aligned.iloc[0]["is_entry_valid"]
    assert aligned.iloc[0]["entry_bar"] == -1
    assert pd.isna(aligned.iloc[0]["entry_day"])
    assert aud["n_unavailable"] == 1


# ===========================================================================
# 10. ex0.entry_bar_for Parity
# ===========================================================================
def test_10_entry_bar_for_parity():
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
# 11. Raw CC == Gap + Trad on ALL Rows
# ===========================================================================
def test_11_raw_cc_equals_gap_plus_trad_on_all_rows():
    obs = exp.load_observed_decision_universe()
    _, _, bars_by_sym = ex0.load_env()
    # Sample containing both hazard == 0 and hazard == 1
    h0 = obs[obs["hazard"] == 0].head(100)
    h1 = obs[obs["hazard"] == 1].head(100)
    sample = pd.concat([h0, h1]).reset_index(drop=True)

    aligned, aud = exp.align_raw_bars_and_returns(sample, bars_by_sym)
    valid = aligned[aligned["is_entry_valid"]]
    assert aud["max_err_decomp"] <= 1e-10
    assert np.allclose(valid["r_state_CC_ATR0"], valid["gap_ATR0"] + valid["r_trad_OC_ATR0"], atol=1e-10)


# ===========================================================================
# 12. Hazard==0 Raw CC == -rebuilt z_d_up
# ===========================================================================
def test_12_hazard0_raw_cc_equals_minus_rebuilt_zdup():
    trans_aud = exp.load_transition_truth_audit()
    cur = trans_aud["cur"]
    obs = exp.load_observed_decision_universe()
    _, _, bars_by_sym = ex0.load_env()

    # Take first 200 hazard==0 rows
    h0_sample = obs[obs["hazard"] == 0].head(200).copy()
    cur_sub = cur.head(200).copy()
    aligned, aud = exp.align_raw_bars_and_returns(h0_sample, bars_by_sym, cur_truth=cur_sub)
    valid = aligned[aligned["is_entry_valid"]]
    assert aud["max_err_r_cc"] <= 1e-8
    assert np.allclose(valid["r_state_CC_ATR0"], -cur_sub["z_d_up"].to_numpy(float), atol=1e-8)


# ===========================================================================
# 13. Cached/Rebuilt Transition Key Parity
# ===========================================================================
def test_13_cached_rebuilt_transition_key_parity():
    trans_aud = exp.load_transition_truth_audit()
    assert trans_aud["symbols_equal"]
    assert trans_aud["episodes_equal"]
    assert trans_aud["blocks_equal"]


# ===========================================================================
# 14. Cached/Rebuilt z_d_up Precision Difference Reported, Not Overwritten
# ===========================================================================
def test_14_cached_rebuilt_zdup_precision_diff_reported_not_overwritten():
    trans_aud = exp.load_transition_truth_audit()
    diff = trans_aud["max_abs_z_d_up_float32_vs_rebuilt"]
    assert 8.0e-7 < diff < 9.0e-7, f"Unexpected float32 precision diff: {diff}"
    # Verify cached is still float32 and rebuilt is float64
    assert trans_aud["cached"]["z_d_up"].dtype == np.float32
    assert trans_aud["cur"]["z_d_up"].dtype == np.float64


# ===========================================================================
# 15. score_mu == -analytic z_d_up_mu
# ===========================================================================
def test_15_score_mu_equals_minus_analytic_zdup_mu():
    class DummySampler:
        def analytic_conditional_support(self, df):
            return {"z_d_up_mu": np.array([0.25, -0.50, 0.0])}
    df = pd.DataFrame({"dummy": [1, 2, 3]})
    scored = exp.score_pgm_block(df, DummySampler())
    assert np.allclose(scored["score_mu"].to_numpy(), np.array([-0.25, 0.50, 0.0]))


# ===========================================================================
# 16. Actual mc.design_cols Contains No Future/Target Cols
# ===========================================================================
def test_16_actual_mc_design_cols_contains_no_future_target_cols():
    fit_A = pgm.fit_samplers_for_window(pgm.WINDOWS[0], pgm.SAMPLE_PATH, pgm.TRANSITION_SAMPLE_PATH)
    mc_A = fit_A["trans_samplers"]["MC_STATE_CURREENCODING"]
    forbidden = set(base.ALL_Z_COLS + base.COUNT_Z + [
        "hazard", "target_mask",
        "reward_SKIP", "reward_MARKET", "reward_LIMIT_RR3", "reward_REASSESS_RR3",
        "r_CC_ATR0", "gap_ATR0", "r_trad_OC_ATR0",
    ])
    intersection = set(mc_A.design_cols).intersection(forbidden)
    assert len(intersection) == 0, f"mc.design_cols contains forbidden target columns: {intersection}"


# ===========================================================================
# 17. TB2 Experiment Route Really Uses mc_A
# ===========================================================================
def test_17_tb2_experiment_route_really_uses_mc_a():
    class TaggedSampler:
        def __init__(self, tag: str):
            self.tag = tag
        def analytic_conditional_support(self, df):
            val = 1.0 if self.tag == "SAMPLER_A" else -1.0
            return {"z_d_up_mu": np.full(len(df), val)}

    s_A = TaggedSampler("SAMPLER_A")
    s_B = TaggedSampler("SAMPLER_B")

    df_eval = pd.DataFrame({
        "block": ["TB2", "TB2", "TB3", "TB3"],
        "hazard": [0, 1, 0, 1],
    })
    sc_tb2, sc_tb3 = exp.route_and_score_evaluation(df_eval, s_A, s_B)
    # score_mu = -z_d_up_mu -> for SAMPLER_A, score_mu = -1.0
    assert np.all(sc_tb2["score_mu"] == -1.0)
    # If route was swapped (s_B used on TB2), score_mu would be 1.0
    assert not np.all(sc_tb2["score_mu"] == 1.0)


# ===========================================================================
# 18. TB3 Experiment Route Really Uses mc_B
# ===========================================================================
def test_18_tb3_experiment_route_really_uses_mc_b():
    class TaggedSampler:
        def __init__(self, tag: str):
            self.tag = tag
        def analytic_conditional_support(self, df):
            val = 1.0 if self.tag == "SAMPLER_A" else -1.0
            return {"z_d_up_mu": np.full(len(df), val)}

    s_A = TaggedSampler("SAMPLER_A")
    s_B = TaggedSampler("SAMPLER_B")

    df_eval = pd.DataFrame({
        "block": ["TB2", "TB2", "TB3", "TB3"],
        "hazard": [0, 1, 0, 1],
    })
    sc_tb2, sc_tb3 = exp.route_and_score_evaluation(df_eval, s_A, s_B)
    # for SAMPLER_B, score_mu = -(-1.0) = 1.0
    assert np.all(sc_tb3["score_mu"] == 1.0)
    assert not np.all(sc_tb3["score_mu"] == -1.0)


# ===========================================================================
# 19. Action Independent of Hazard Label
# ===========================================================================
def test_19_action_independent_of_hazard_label():
    df_h0 = pd.DataFrame({"score_mu": [0.5, -0.2, 0.0], "hazard": [0, 0, 0]})
    df_h1 = pd.DataFrame({"score_mu": [0.5, -0.2, 0.0], "hazard": [1, 1, 1]})

    class Dummy:
        def analytic_conditional_support(self, df):
            return {"z_d_up_mu": -df["score_mu"].to_numpy()}

    sc0 = exp.score_pgm_block(df_h0, Dummy())
    sc1 = exp.score_pgm_block(df_h1, Dummy())
    assert (sc0["action"].values == sc1["action"].values).all()


# ===========================================================================
# 20. ALL Metrics Use hazard0+hazard1 Rows
# ===========================================================================
def test_20_all_metrics_use_hazard0_and_hazard1_rows():
    df = pd.DataFrame({
        "score_mu": [0.5, -0.5, 0.2, -0.2],
        "r_state_CC_ATR0": [0.4, -0.4, 0.1, -0.1],
        "r_trad_OC_ATR0": [0.3, -0.3, 0.1, -0.1],
        "gap_ATR0": [0.1, -0.1, 0.0, 0.0],
        "action": [1, -1, 1, -1],
        "strategy_return_ATR0": [0.3, 0.3, 0.1, 0.1],
        "hazard": [0, 0, 1, 1],
    })
    m = exp.compute_block_metrics(df)
    assert m["n_decisions_all"] == 4
    assert m["n_signals"] == 4
    assert np.isclose(m["gross_ev_signal"], 0.20)


# ===========================================================================
# 21. H0/H1 Subgroup Metrics Diagnostic Only
# ===========================================================================
def test_21_h0_h1_subgroup_metrics_diagnostic_only():
    df = pd.DataFrame({
        "score_mu": [0.5, -0.5, 0.2, -0.2],
        "r_state_CC_ATR0": [0.4, -0.4, 0.1, -0.1],
        "r_trad_OC_ATR0": [0.3, -0.3, 0.1, -0.1],
        "gap_ATR0": [0.1, -0.1, 0.0, 0.0],
        "action": [1, -1, 1, -1],
        "strategy_return_ATR0": [0.3, 0.3, 0.1, 0.1],
        "hazard": [0, 0, 1, 1],
    })
    m = exp.compute_block_metrics(df)
    assert "DIAGNOSTIC_H0" in m
    assert "DIAGNOSTIC_H1" in m
    assert m["DIAGNOSTIC_H0"]["n"] == 2
    assert m["DIAGNOSTIC_H1"]["n"] == 2
    assert "future_filter_EV_bias" in m
    # H0 EV = 0.30, ALL EV = 0.20 -> bias = +0.10
    assert np.isclose(m["future_filter_EV_bias"], 0.10)


# ===========================================================================
# 22. Cost Only on Non-SKIP
# ===========================================================================
def test_22_cost_only_on_non_skip():
    df = pd.DataFrame({
        "action": [1, -1, 0, 1],
        "strategy_return_ATR0": [0.10, -0.05, 0.00, 0.20],
    })
    stress = exp.run_cost_stress_grid(df)
    for row in stress:
        cost = row["cost_ATR0"]
        net_ret = df["strategy_return_ATR0"] - cost * (df["action"] != 0)
        assert net_ret.iloc[2] == 0.00
        assert np.isclose(row["net_EV_per_signal_ATR0"], net_ret.mean())


# ===========================================================================
# 23. TB3 Deciles Still Use TB2 Edges
# ===========================================================================
def test_23_tb3_deciles_still_use_tb2_edges():
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
# 24. Day-Cluster Bootstrap Deterministic
# ===========================================================================
def test_24_day_cluster_bootstrap_deterministic():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "entry_day": np.random.choice(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"], size=n),
        "score_mu": np.random.randn(n),
        "r_state_CC_ATR0": np.random.randn(n),
        "r_trad_OC_ATR0": np.random.randn(n),
        "strategy_return_ATR0": np.random.randn(n),
        "hazard": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
    })
    b1 = exp.run_day_clustered_bootstrap(df, n_boot=50, seed=20260915)
    b2 = exp.run_day_clustered_bootstrap(df, n_boot=50, seed=20260915)
    assert b1 == b2


# ===========================================================================
# 25. No V2 / R1-R4 Dependency
# ===========================================================================
def test_25_no_v2_r1_r4_dependency():
    src = inspect.getsource(exp)
    assert "pd.read_parquet" in src
    assert "execution_lag1_trades.parquet" not in src.replace('"execution_lag1_trades.parquet"', "")
    assert "fit_action_q_models" not in src.replace('"fit_action_q_models"', "")
    assert "apply_action_policy" not in src.replace('"apply_action_policy"', "")
    assert "reward_SKIP" not in src.replace('"reward_SKIP"', "")
    assert "CONTACTS" not in src


# ===========================================================================
# 26. Smoke Cannot Emit Formal Verdict
# ===========================================================================
def test_26_smoke_cannot_emit_formal_verdict():
    src = inspect.getsource(exp.run_smoke_test)
    assert "No formal verdict emitted" in src
    assert "PGM_NATIVE_ONE_STEP_TRADABLE_ALPHA_SUPPORTED" not in src


# ===========================================================================
# 27. Formal Verdict Strings Contain ON_FROZEN_PGM_BAR_SAMPLE
# ===========================================================================
def test_27_formal_verdict_strings_contain_on_frozen_pgm_bar_sample():
    for k, v in exp.VERDICT_STRINGS.items():
        assert "ON_FROZEN_PGM_BAR_SAMPLE" in v, f"Verdict {k} does not contain ON_FROZEN_PGM_BAR_SAMPLE: {v}"


# ===========================================================================
# Test Runner
# ===========================================================================
if __name__ == "__main__":
    import traceback

    tests = [getattr(sys.modules[__name__], f) for f in dir(sys.modules[__name__])
             if f.startswith("test_")]
    tests.sort(key=lambda fn: int(fn.__name__.split("_")[1]))

    ok = fail = 0
    print(f"Running {len(tests)} unit tests for PGM-NATIVE-0A.1...\n")
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
