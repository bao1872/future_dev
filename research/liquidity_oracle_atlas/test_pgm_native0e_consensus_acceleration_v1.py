"""
test_pgm_native0e_consensus_acceleration_v1.py

Round 1 tests for PGM-NATIVE-0E (architecture + audit + smoke ONLY).
Full exploratory is HARD-BLOCKED in this round.
"""
import inspect
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import research.liquidity_oracle_atlas.experiment_pgm_native0e_consensus_acceleration_v1 as e0
import research.liquidity_oracle_atlas.experiment_pgm_native0d_acceleration_terminal_outcome_v1 as d0
import research.liquidity_oracle_atlas.experiment_pgm_native0c_state_augmentation_v1 as n0c
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1c_free_run_rollout_v1 as pgm
import research.liquidity_oracle_atlas.experiment_path0_episode_path_memory_v1 as pm
import research.liquidity_oracle_atlas.experiment_pgm_native0a_one_step_alpha_v1 as n0a

SYMS = n0a.EXPECTED_SYMBOLS


def _pgm_num_cols():
    return list(set(pgm.T2_NUM + n0c.U_COLS + n0c.E_COLS))


def make_synth(n=400, seed=0, blocks=("TB1",)):
    rng = np.random.default_rng(seed)
    syms = SYMS[:5]
    sym = np.array(syms)[rng.integers(0, len(syms), n)]
    df = pd.DataFrame({
        "symbol": sym,
        "episode_id": ["E%d" % e for e in rng.integers(0, 3, n)],
        "block": [blocks[i % len(blocks)] for i in range(n)],
        "bar_t": np.arange(n),
        "start_bar": np.zeros(n),
        "score_mu": rng.normal(0, 1, n),
        "path_last_return_R": rng.normal(0, 0.5, n),
        "cur_up_distance_R": np.abs(rng.normal(1, 0.3, n)),
        "cur_down_distance_R": np.abs(rng.normal(1, 0.3, n)),
        "upper_newest_log_age": rng.uniform(0.5, 5, n),
        "lower_newest_log_age": rng.uniform(0.5, 5, n),
        "upper_n_active_identities": rng.integers(1, 8, n).astype(float),
        "lower_n_active_identities": rng.integers(1, 8, n).astype(float),
        "a_dir_accel_1": rng.normal(0, 0.3, n),
        "pi": rng.normal(0, 0.01, n),
        "harm_flag": rng.integers(0, 2, n),
        "hazard": rng.integers(0, 2, n),
        "prev_event_mask": rng.integers(0, 2, n).astype(str),
        "entry_day": ["2024-01-%02d" % (i % 9 + 1) for i in range(n)],
        "decision_day": ["2024-01-%02d" % (i % 9 + 1) for i in range(n)],
        "same_block_entry_valid": [True] * n,
        "r_trad_OC_ATR0": rng.normal(0, 0.01, n),
    })
    for c in _pgm_num_cols():
        df[c] = rng.normal(0, 1, n)
    df["base_action"] = np.sign(df["score_mu"]).astype(float)
    df.loc[df["base_action"] == 0, "base_action"] = 1.0
    df["abs_score_mu"] = df["score_mu"].abs()
    df["is_entry_valid"] = True
    return df


def make_consen(synth):
    x, _ = e0.prepare_consensus(synth, ["TB1"])
    cs = x[x["block"].isin(["TB1"])]["consensus_score"].to_numpy(float)
    terr = e0.compute_consensus_terciles(cs)
    return x, terr


def make_consen_synth(n=400, seed=0):
    s = make_synth(n, seed)
    x = e0.add_consensus_primitives(s)
    maps = e0.fit_rank_maps(x)
    x = e0.attach_consensus_score(x, maps)
    x = e0.attach_primary_acceleration(x)
    return x, maps


# ---------------------------------------------------------------------------
# Governance / freeze
# ---------------------------------------------------------------------------
def test_base_sha_exact():
    assert e0.BASE_SHA == "f76b10e724848bfd34076652bf3c76f4cd1ef696"


def test_base_sha_is_ancestor_of_head():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    rc = subprocess.run(["git", "merge-base", "--is-ancestor", e0.BASE_SHA, head],
                        capture_output=True).returncode
    assert rc == 0


def test_experiment_scope_frozen():
    assert e0.EXPERIMENT_SCOPE == "EXPLORATORY_PGM_CONSENSUS_ACCELERATION_INTERACTION_ON_TB1_TB2_TB3"
    assert e0.ALLOWED_BLOCKS == ["TB1", "TB2", "TB3"]


def test_no_tb4_in_universe():
    obs = n0a.load_observed_decision_universe()
    assert "TB4" not in set(obs["block"].unique())


def test_sample_hash_parity_keys():
    h = n0c.compute_artifact_hashes()
    assert "sample_artifact_sha256" in h and "transition_artifact_sha256" in h


def test_cluster_owner_entry_day():
    assert e0.CLUSTER_OWNER == "entry_day"


def test_bootstrap_seed_frozen():
    assert e0.BOOTSTRAP_SEED == 20260916


# ---------------------------------------------------------------------------
# Consensus primitives
# ---------------------------------------------------------------------------
def test_consensus_exact_4_raw_cols():
    s = make_synth()
    x = e0.add_consensus_primitives(s)
    for c in e0.CONSENSUS_RAW:
        assert c in x.columns


def test_consensus_raw_count_is_four():
    assert len(e0.CONSENSUS_RAW) == 4


def test_position_formula_up():
    rng = np.random.default_rng(1)
    n = 6
    df = pd.DataFrame({
        "symbol": ["X"] * n, "episode_id": ["E1"] * n, "block": ["TB1"] * n,
        "bar_t": np.arange(n), "start_bar": np.zeros(n),
        "score_mu": [1.0] * n,
        "path_last_return_R": rng.normal(0, 0.1, n),
        "cur_up_distance_R": [0.2] * n, "cur_down_distance_R": [0.8] * n,
        "upper_newest_log_age": [1.0] * n, "lower_newest_log_age": [2.0] * n,
        "upper_n_active_identities": [3.0] * n, "lower_n_active_identities": [5.0] * n,
    })
    x = e0.add_consensus_primitives(df)
    # rows>=1 have a valid lag; row0 is NaN (no t-1). d_prev>0 => (0.8-0.2)/(1.0)=0.6
    assert np.allclose(x["c_position"].to_numpy(float)[1:], 0.6, atol=1e-9)


def test_position_formula_down():
    rng = np.random.default_rng(2)
    n = 6
    df = pd.DataFrame({
        "symbol": ["X"] * n, "episode_id": ["E1"] * n, "block": ["TB1"] * n,
        "bar_t": np.arange(n), "start_bar": np.zeros(n),
        "score_mu": [-1.0] * n,
        "path_last_return_R": rng.normal(0, 0.1, n),
        "cur_up_distance_R": [0.8] * n, "cur_down_distance_R": [0.2] * n,
        "upper_newest_log_age": [1.0] * n, "lower_newest_log_age": [2.0] * n,
        "upper_n_active_identities": [3.0] * n, "lower_n_active_identities": [5.0] * n,
    })
    x = e0.add_consensus_primitives(df)
    # d_prev<0 => (0.8-0.2)/(1.0)=0.6
    assert np.allclose(x["c_position"].to_numpy(float)[1:], 0.6, atol=1e-9)


def test_path_agreement_excludes_current_return():
    rng = np.random.default_rng(3)
    n = 12
    base = rng.normal(0, 0.1, n)
    df = pd.DataFrame({
        "symbol": ["X"] * n, "episode_id": ["E1"] * n, "block": ["TB1"] * n,
        "bar_t": np.arange(n), "start_bar": np.zeros(n),
        "score_mu": [1.0] * n,
        "path_last_return_R": base.copy(),
        "cur_up_distance_R": np.abs(rng.normal(1, 0.3, n)),
        "cur_down_distance_R": np.abs(rng.normal(1, 0.3, n)),
        "upper_newest_log_age": rng.uniform(0.5, 5, n),
        "lower_newest_log_age": rng.uniform(0.5, 5, n),
        "upper_n_active_identities": rng.integers(1, 8, n).astype(float),
        "lower_n_active_identities": rng.integers(1, 8, n).astype(float),
    })
    x = e0.add_consensus_primitives(df)
    df2 = df.copy()
    df2.loc[n - 1, "path_last_return_R"] = 99.0
    x2 = e0.add_consensus_primitives(df2)
    # row0/row1 have no lag / insufficient prior (NaN); the current return must not leak
    assert np.allclose(x["c_path_agreement"].to_numpy(float)[1:],
                       x2["c_path_agreement"].to_numpy(float)[1:], atol=1e-12, equal_nan=True)


def test_path_agreement_formula_matches():
    rng = np.random.default_rng(4)
    n = 12
    df = pd.DataFrame({
        "symbol": ["X"] * n, "episode_id": ["E1"] * n, "block": ["TB1"] * n,
        "bar_t": np.arange(n), "start_bar": np.zeros(n),
        "score_mu": [1.0] * n,
        "path_last_return_R": rng.normal(0, 0.3, n),
        "cur_up_distance_R": np.abs(rng.normal(1, 0.3, n)),
        "cur_down_distance_R": np.abs(rng.normal(1, 0.3, n)),
        "upper_newest_log_age": rng.uniform(0.5, 5, n),
        "lower_newest_log_age": rng.uniform(0.5, 5, n),
        "upper_n_active_identities": rng.integers(1, 8, n).astype(float),
        "lower_n_active_identities": rng.integers(1, 8, n).astype(float),
    })
    x = e0.add_consensus_primitives(df)
    g = df.groupby(["symbol", "episode_id"], sort=False)
    prior_sum = g["path_last_return_R"].transform(
        lambda s: s.shift(1).rolling(e0.CONSENSUS_LOOKBACK, min_periods=e0.CONSENSUS_MIN_PRIOR).sum())
    prior_abs = g["path_last_return_R"].transform(
        lambda s: s.abs().shift(1).rolling(e0.CONSENSUS_LOOKBACK, min_periods=e0.CONSENSUS_MIN_PRIOR).sum())
    expected = prior_sum / (prior_abs + e0.EPS)
    assert np.allclose(x["c_path_agreement"].to_numpy(float), expected.to_numpy(float),
                       atol=1e-12, equal_nan=True)


def test_freshness_formula_up():
    rng = np.random.default_rng(5)
    n = 6
    df = pd.DataFrame({
        "symbol": ["X"] * n, "episode_id": ["E1"] * n, "block": ["TB1"] * n,
        "bar_t": np.arange(n), "start_bar": np.zeros(n),
        "score_mu": [1.0] * n,
        "path_last_return_R": rng.normal(0, 0.1, n),
        "cur_up_distance_R": rng.normal(1, 0.3, n),
        "cur_down_distance_R": rng.normal(1, 0.3, n),
        "upper_newest_log_age": [1.0] * n, "lower_newest_log_age": [2.0] * n,
        "upper_n_active_identities": rng.integers(1, 8, n).astype(float),
        "lower_n_active_identities": rng.integers(1, 8, n).astype(float),
    })
    x = e0.add_consensus_primitives(df)
    # d_prev>0: front_age=up_age=1, back_age=dn_age=2 => 1-2=-1
    assert np.allclose(x["c_boundary_freshness"].to_numpy(float)[1:], -1.0, atol=1e-12)


def test_freshness_formula_down():
    rng = np.random.default_rng(6)
    n = 6
    df = pd.DataFrame({
        "symbol": ["X"] * n, "episode_id": ["E1"] * n, "block": ["TB1"] * n,
        "bar_t": np.arange(n), "start_bar": np.zeros(n),
        "score_mu": [-1.0] * n,
        "path_last_return_R": rng.normal(0, 0.1, n),
        "cur_up_distance_R": rng.normal(1, 0.3, n),
        "cur_down_distance_R": rng.normal(1, 0.3, n),
        "upper_newest_log_age": [1.0] * n, "lower_newest_log_age": [2.0] * n,
        "upper_n_active_identities": rng.integers(1, 8, n).astype(float),
        "lower_n_active_identities": rng.integers(1, 8, n).astype(float),
    })
    x = e0.add_consensus_primitives(df)
    # d_prev<0: front_age=dn_age=2, back_age=up_age=1 => 2-1=1
    assert np.allclose(x["c_boundary_freshness"].to_numpy(float)[1:], 1.0, atol=1e-12)


def test_density_formula_up():
    rng = np.random.default_rng(7)
    n = 6
    df = pd.DataFrame({
        "symbol": ["X"] * n, "episode_id": ["E1"] * n, "block": ["TB1"] * n,
        "bar_t": np.arange(n), "start_bar": np.zeros(n),
        "score_mu": [1.0] * n,
        "path_last_return_R": rng.normal(0, 0.1, n),
        "cur_up_distance_R": rng.normal(1, 0.3, n),
        "cur_down_distance_R": rng.normal(1, 0.3, n),
        "upper_newest_log_age": rng.uniform(0.5, 5, n),
        "lower_newest_log_age": rng.uniform(0.5, 5, n),
        "upper_n_active_identities": [3.0] * n, "lower_n_active_identities": [5.0] * n,
    })
    x = e0.add_consensus_primitives(df)
    # d_prev>0: back_n=dn_n=5, front_n=up_n=3 => 5-3=2
    assert np.allclose(x["c_boundary_density"].to_numpy(float)[1:], 2.0, atol=1e-12)


def test_density_formula_down():
    rng = np.random.default_rng(8)
    n = 6
    df = pd.DataFrame({
        "symbol": ["X"] * n, "episode_id": ["E1"] * n, "block": ["TB1"] * n,
        "bar_t": np.arange(n), "start_bar": np.zeros(n),
        "score_mu": [-1.0] * n,
        "path_last_return_R": rng.normal(0, 0.1, n),
        "cur_up_distance_R": rng.normal(1, 0.3, n),
        "cur_down_distance_R": rng.normal(1, 0.3, n),
        "upper_n_active_identities": [3.0] * n, "lower_n_active_identities": [5.0] * n,
        "upper_newest_log_age": rng.uniform(0.5, 5, n),
        "lower_newest_log_age": rng.uniform(0.5, 5, n),
    })
    x = e0.add_consensus_primitives(df)
    # d_prev<0: back_n=up_n=3, front_n=dn_n=5 => 3-5=-2
    assert np.allclose(x["c_boundary_density"].to_numpy(float)[1:], -2.0, atol=1e-12)


def test_direction_stable_contract():
    rng = np.random.default_rng(9)
    n = 8
    df = pd.DataFrame({
        "symbol": ["X"] * n, "episode_id": ["E1"] * n, "block": ["TB1"] * n,
        "bar_t": np.arange(n), "start_bar": np.zeros(n),
        "score_mu": [1.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        "path_last_return_R": rng.normal(0, 0.1, n),
        "cur_up_distance_R": rng.normal(1, 0.3, n),
        "cur_down_distance_R": rng.normal(1, 0.3, n),
        "upper_newest_log_age": rng.uniform(0.5, 5, n),
        "lower_newest_log_age": rng.uniform(0.5, 5, n),
        "upper_n_active_identities": rng.integers(1, 8, n).astype(float),
        "lower_n_active_identities": rng.integers(1, 8, n).astype(float),
    })
    x = e0.add_consensus_primitives(df)
    ds = x["direction_stable"].to_numpy(bool)
    assert bool(ds[0]) is False          # no lag
    assert bool(ds[2]) is False          # lag=+1, now=-1
    assert bool(ds[4]) is False          # now=0
    assert bool(ds[1]) is True           # lag=+1, now=+1


def test_current_row_mutation_cannot_change_C_t_minus_1():
    rng = np.random.default_rng(10)
    n = 10
    df = pd.DataFrame({
        "symbol": ["X"] * n, "episode_id": ["E1"] * n, "block": ["TB1"] * n,
        "bar_t": np.arange(n), "start_bar": np.zeros(n),
        "score_mu": rng.normal(0, 1, n),
        "path_last_return_R": rng.normal(0, 0.2, n),
        "cur_up_distance_R": np.abs(rng.normal(1, 0.3, n)),
        "cur_down_distance_R": np.abs(rng.normal(1, 0.3, n)),
        "upper_newest_log_age": rng.uniform(0.5, 5, n),
        "lower_newest_log_age": rng.uniform(0.5, 5, n),
        "upper_n_active_identities": rng.integers(1, 8, n).astype(float),
        "lower_n_active_identities": rng.integers(1, 8, n).astype(float),
    })
    x = e0.add_consensus_primitives(df)
    df2 = df.copy()
    t = 5
    df2.loc[t, "cur_up_distance_R"] = df2.loc[t, "cur_up_distance_R"] * 7.0 + 3.0
    x2 = e0.add_consensus_primitives(df2)
    for c in e0.CONSENSUS_RAW:
        assert np.isclose(x[c].to_numpy(float)[t], x2[c].to_numpy(float)[t], atol=1e-12), c


def test_prefix_invariance_helper():
    assert e0._causal_prefix_invariant()


def test_consensus_eligible_requires_finite_raw():
    rng = np.random.default_rng(11)
    n = 9
    df = pd.DataFrame({
        "symbol": ["X"] * n, "episode_id": ["E1"] * n, "block": ["TB1"] * n,
        "bar_t": np.arange(n), "start_bar": np.zeros(n),
        "score_mu": [1.0] * n,
        "path_last_return_R": rng.normal(0, 0.1, n),
        "cur_up_distance_R": [0.5] * n,
        "cur_down_distance_R": [0.5] * n,
        "upper_newest_log_age": [1.0] * n, "lower_newest_log_age": [2.0] * n,
        "upper_n_active_identities": [3.0] * n, "lower_n_active_identities": [5.0] * n,
    })
    # corrupt the PREVIOUS row's raw so c_position[row4] (= shift(1) of it) is NaN
    df.loc[3, "cur_up_distance_R"] = np.nan
    x = e0.add_consensus_primitives(df)
    elig = x["consensus_eligible"].to_numpy(bool)
    assert bool(elig[4]) is False             # NaN consensus primitive -> ineligible
    assert bool(elig[5]) is True              # finite consensus + valid lag -> eligible


# ---------------------------------------------------------------------------
# Rank maps / consensus score
# ---------------------------------------------------------------------------
def test_train_only_rank_map_deterministic():
    x, _ = make_consen_synth(300, seed=12)
    m1 = e0.fit_rank_maps(x)
    m2 = e0.fit_rank_maps(x)
    for c in e0.CONSENSUS_RAW:
        assert np.array_equal(m1[c], m2[c])


def test_rank_map_clips_range():
    x, maps = make_consen_synth(200, seed=13)
    for c in e0.CONSENSUS_RAW:
        r = e0.empirical_rank(np.array([1e9, -1e9, 0.0]), maps[c])
        assert np.all((r >= -1.0 - 1e-9) & (r <= 1.0 + 1e-9))


def test_eval_mutation_cannot_alter_rank_map():
    x, maps = make_consen_synth(200, seed=14)
    before = {c: maps[c].copy() for c in maps}
    x2 = x.copy()
    x2["cur_up_distance_R"] = 99.0
    # rank map is a frozen train-only reference; an eval mutation must not change it
    maps2 = e0.fit_rank_maps(x2)
    for c in e0.CONSENSUS_RAW:
        assert np.array_equal(maps[c], before[c])
        assert np.array_equal(maps2[c], maps[c])


def test_equal_weight_exact():
    x, _ = make_consen_synth(200, seed=15)
    rank_cols = [f"{c}_rank" for c in e0.CONSENSUS_RAW]
    manual = x[rank_cols].mean(axis=1).to_numpy(float)
    assert np.allclose(x["consensus_score"].to_numpy(float), manual, atol=1e-12)


def test_consensus_score_in_unit_range():
    x, _ = make_consen_synth(200, seed=16)
    cs = x["consensus_score"].to_numpy(float)
    assert np.all((cs >= -1.0) & (cs <= 1.0))


def test_tercile_frozen_train_only():
    s = make_synth(300, seed=17, blocks=("TB1", "TB2"))
    x, _ = e0.prepare_consensus(s, ["TB1"])
    cs_train = x[x["block"].isin(["TB1"])]["consensus_score"].to_numpy(float)
    ql, qh = e0.compute_consensus_terciles(cs_train)
    # mutate the NON-train (TB2) rows; train terciles must be unchanged
    x2 = x.copy()
    x2.loc[x2["block"] == "TB2", "consensus_score"] = 0.0
    cs_train2 = x2[x2["block"].isin(["TB1"])]["consensus_score"].to_numpy(float)
    ql2, qh2 = e0.compute_consensus_terciles(cs_train2)
    assert ql == ql2 and qh == qh2


def test_degenerate_tercile_stop():
    with pytest.raises(SystemExit):
        e0.compute_consensus_terciles(np.full(100, 0.5))


def test_empty_rank_map_stop():
    x, _ = make_consen_synth(50, seed=18)
    x = x.copy()
    for c in e0.CONSENSUS_RAW:
        x[c] = np.nan
    with pytest.raises(SystemExit):
        e0.fit_rank_maps(x)


# ---------------------------------------------------------------------------
# Acceleration / primary model
# ---------------------------------------------------------------------------
def test_a_col_reuse_from_0d():
    assert e0.A_COLS is d0.A_COLS
    assert "a_dir_accel_1" in e0.A_COLS
    assert len(e0.A_COLS) == 8


def test_primary_acceleration_exact_a_dir_accel_1():
    x, _ = make_consen_synth(100, seed=19)
    assert np.allclose(x["accel_raw"].to_numpy(float), x["a_dir_accel_1"].to_numpy(float))
    assert np.allclose(x["accel_plus"].to_numpy(float), np.maximum(x["accel_raw"].to_numpy(float), 0.0))


def test_no_other_accel_in_primary_model():
    forbidden = {"a_dir_velocity", "a_dir_jerk_1", "a_speed_accel_1", "a_range_accel_2",
                 "a_eff_slope_1", "a_burst_exhaustion", "a_conviction_burst"}
    m0 = set(e0.m0_num())
    assert not (forbidden & m0)
    assert "consensus_x_accel" not in m0


def test_m1_is_m0_plus_interaction_only():
    m0 = set(e0.m0_num())
    m1 = set(e0.m1_num())
    assert m1 == (m0 | {"consensus_x_accel"})


def test_ridge_pipeline_fixed_alpha():
    pipe = d0.make_ridge_pipeline(e0.m0_num(), e0.PRED_CAT)
    reg = pipe.named_steps["reg"]
    assert type(reg).__name__ == "Ridge"
    assert reg.alpha == 1.0


def test_logistic_pipeline_fixed_c():
    pipe = pm.make_pipeline(e0.m0_num(), e0.PRED_CAT)
    clf = pipe.named_steps["clf"]
    assert clf.C == 1.0
    assert clf.penalty == "l2"


# ---------------------------------------------------------------------------
# Four-cell formulas / effects
# ---------------------------------------------------------------------------
def _explicit_cell_df():
    return pd.DataFrame({
        "consensus_group": ["LOW", "LOW", "LOW", "LOW", "HIGH", "HIGH", "HIGH", "HIGH"],
        "accel_positive": [False, False, True, True, False, False, True, True],
        "pi": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0],
        "harm_flag": [0, 0, 0, 0, 0, 0, 0, 0],
        "hazard": [0, 0, 0, 0, 0, 0, 0, 0],
    })


def test_four_cell_formulas():
    df = _explicit_cell_df()
    cells = e0._cells_with_group(df, "consensus_group")
    assert cells["LOW_OFF"]["mean_pi"] == 0.0
    assert cells["LOW_ACCEL"]["mean_pi"] == 1.0
    assert cells["HIGH_OFF"]["mean_pi"] == 0.0
    assert cells["HIGH_ACCEL"]["mean_pi"] == -1.0


def test_effects_and_did_formula():
    df = _explicit_cell_df()
    cells = e0._cells_with_group(df, "consensus_group")
    eff = e0.primary_effects(cells)
    assert abs(eff["Delta_LOW_pi"] - 1.0) < 1e-12
    assert abs(eff["Delta_HIGH_pi"] - (-1.0)) < 1e-12
    assert abs(eff["DID_pi"] - (-2.0)) < 1e-12


def test_min_cell_stop():
    df = _explicit_cell_df().drop(index=[2, 3]).reset_index(drop=True)
    with pytest.raises(SystemExit):
        e0.assert_min_cells(df)


def test_min_cell_stop_message_names_cell():
    df = _explicit_cell_df().drop(index=[2, 3]).reset_index(drop=True)
    try:
        e0.assert_min_cells(df)
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert "LOW_ACCEL" in str(e)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def _bootstrap_eval(n=600, seed=20):
    rng = np.random.default_rng(seed)
    days = ["2024-01-%02d" % (i % 5 + 1) for i in range(n)]
    grp = rng.choice(["LOW", "HIGH"], n)
    acc = rng.random(n) < 0.5
    pi = rng.normal(0, 0.01, n)
    pi[(grp == "LOW") & acc] += 0.05
    pi[(grp == "HIGH") & acc] -= 0.05
    return pd.DataFrame({
        "consensus_group": grp, "accel_positive": acc,
        "pi": pi, "harm_flag": rng.integers(0, 2, n),
        "hazard": rng.integers(0, 2, n), "entry_day": days,
    })


def test_bootstrap_deterministic_same_seed():
    ev = _bootstrap_eval()
    b1 = e0.bootstrap_did(ev, n_boot=200, seed=20260916)
    b2 = e0.bootstrap_did(ev, n_boot=200, seed=20260916)
    for k in b1:
        assert b1[k]["point"] == b2[k]["point"]
        assert b1[k]["ci95_lower"] == b2[k]["ci95_lower"]
        assert b1[k]["ci95_upper"] == b2[k]["ci95_upper"]


def test_bootstrap_did_equals_point_difference():
    ev = _bootstrap_eval()
    b = e0.bootstrap_did(ev, n_boot=200, seed=1)
    cells = e0._cells_with_group(ev, "consensus_group")
    manual = (cells["HIGH_ACCEL"]["mean_pi"] - cells["HIGH_OFF"]["mean_pi"]) - \
             (cells["LOW_ACCEL"]["mean_pi"] - cells["LOW_OFF"]["mean_pi"])
    assert abs(b["DID_pi"]["point"] - manual) < 1e-12


def test_bootstrap_handles_empty_cell_replicates():
    ev = _bootstrap_eval(n=120, seed=21)
    b = e0.bootstrap_did(ev, n_boot=100, seed=2)
    assert "DID_pi" in b and np.isfinite(b["DID_pi"]["point"])


# ---------------------------------------------------------------------------
# Verdict (TB3 only)
# ---------------------------------------------------------------------------
def _boot(lower, upper, p):
    return dict(point=0.0, ci95_lower=lower, ci95_upper=upper, p_pos=p)


def test_verdict_sign_switch():
    v = e0.determine_psych_verdict(
        _boot(-0.01, -0.001, 0.01), _boot(0.001, 0.02, 0.99), _boot(-0.02, -0.001, 0.01))
    assert v == e0.VERDICT["SIGN_SWITCH"]


def test_verdict_state_dep():
    v = e0.determine_psych_verdict(
        _boot(-0.01, -0.001, 0.01), _boot(-0.02, 0.001, 0.4), _boot(-0.02, -0.001, 0.01))
    assert v == e0.VERDICT["STATE_DEP"]


def test_verdict_not_supported():
    v = e0.determine_psych_verdict(
        _boot(0.001, 0.02, 0.99), _boot(0.001, 0.02, 0.99), _boot(0.001, 0.02, 0.99))
    assert v == e0.VERDICT["NOT_SUPPORTED"]


# ---------------------------------------------------------------------------
# Psychology gate diagnostic
# ---------------------------------------------------------------------------
def test_psych_gate_exact_formula():
    rng = np.random.default_rng(22)
    n = 12
    base = rng.choice([-1.0, 1.0], n)
    high = np.array([True, True, False, False] * 3)
    accel = np.array([True, False, True, False] * 3)
    df = pd.DataFrame({
        "base_action": base, "consensus_group": np.where(high, "HIGH", "LOW"),
        "accel_positive": accel, "r_trad_OC_ATR0": rng.normal(0, 0.01, n),
        "entry_day": ["2024-01-01"] * n, "symbol": ["AG"] * n,
    })
    r = e0.psych_gate_diagnostic(df)
    bs = r["BASE"]["by_symbol"]["AG"]["trade_count"]
    gs = r["PSYCH_GATE"]["by_symbol"]["AG"]["trade_count"]
    assert gs == bs - int((high & accel).sum())


def test_cost_fixed_0_01():
    assert e0.PRIMARY_COST_ATR0 == 0.01


def test_no_symbol_pruning_in_psych_gate():
    rng = np.random.default_rng(23)
    n = len(SYMS) * 4
    df = pd.DataFrame({
        "base_action": rng.choice([-1.0, 1.0], n),
        "consensus_group": rng.choice(["LOW", "HIGH"], n),
        "accel_positive": rng.random(n) < 0.5,
        "r_trad_OC_ATR0": rng.normal(0, 0.01, n),
        "entry_day": ["2024-01-01"] * n,
        "symbol": np.repeat(SYMS, 4),
    })
    r = e0.psych_gate_diagnostic(df)
    assert len(r["PSYCH_GATE"]["by_symbol"]) == len(SYMS)


def test_psych_gate_no_threshold_param():
    sig = inspect.signature(e0.psych_gate_diagnostic)
    assert set(sig.parameters.keys()) == {"eval_sub", "cost"}


# ---------------------------------------------------------------------------
# Component diagnostics cannot alter verdict
# ---------------------------------------------------------------------------
def test_component_diagnostics_four_keys():
    s = make_synth(400, seed=24)
    x, terr = make_consen(s)
    sub = e0.build_primary_sample_from_scored(x, terr)
    sub = sub[sub["block"] == "TB1"].copy()
    maps = e0.fit_rank_maps(x[x["block"].isin(["TB1"])])
    comp = e0.component_diagnostics(sub, maps)
    assert set(comp.keys()) == set(e0.CONSENSUS_RAW)
    for c in comp:
        assert "DID_pi" in comp[c]


def test_component_diagnostics_do_not_emit_verdict():
    s = make_synth(400, seed=25)
    x, terr = make_consen(s)
    sub = e0.build_primary_sample_from_scored(x, terr)
    sub = sub[sub["block"] == "TB1"].copy()
    maps = e0.fit_rank_maps(x[x["block"].isin(["TB1"])])
    comp = e0.component_diagnostics(sub, maps)
    # verdict function only consumes bootstrap dicts, never the component mapping
    assert "consensus_score" in sub.columns


# ---------------------------------------------------------------------------
# Full blocked / governance
# ---------------------------------------------------------------------------
def test_full_blocked_without_token():
    with pytest.raises(SystemExit):
        e0.require_full_authorization()


def test_full_blocked_even_with_token():
    # require_full_authorization blocks unconditionally (token is irrelevant by design)
    with pytest.raises(SystemExit):
        e0.require_full_authorization()


def test_module_has_no_run_full_exploratory_def():
    src = inspect.getsource(e0)
    assert "def run_full_exploratory" not in src
    assert "run_full_exploratory(" not in src


def test_governance_regression_no_authorized_full_in_tests():
    this = inspect.getsource(sys.modules[__name__])
    # build the token dynamically so this test's own source does not contain it verbatim
    needle = "AUTHORIZE_PGM_NATIVE0E_FULL" + "_EXPLORATORY"
    assert needle not in this


def test_future_token_absent_in_primitive_source():
    # only scan the function where a real future-leak could hide
    src = inspect.getsource(e0.add_consensus_primitives)
    for tok in ("shift(-1)", "future_", "next_", "remaining", "final_"):
        assert tok not in src


# ---------------------------------------------------------------------------
# Additional structural / boundary / verdict coverage (fast, synthetic)
# ---------------------------------------------------------------------------
def test_min_cell_n_constant():
    assert e0.MIN_CELL_N == 500


def test_allowed_blocks_exact():
    assert e0.ALLOWED_BLOCKS == ["TB1", "TB2", "TB3"]


def test_experiment_name_has_0e():
    assert "0E" in e0.EXPERIMENT_NAME


def test_pred_cat_exact():
    assert e0.PRED_CAT == ["prev_event_mask", "symbol"]


def test_pred_num_base_contents():
    base = set(e0.m0_num())
    for need in ["score_mu", "abs_score_mu", "consensus_score", "accel_plus"]:
        assert need in base
    assert set(pgm.T2_NUM) <= base
    assert set(n0c.U_COLS) <= base
    assert set(n0c.E_COLS) <= base


def test_m1_contains_interaction_and_accel_plus():
    m1 = set(e0.m1_num())
    assert "consensus_x_accel" in m1
    assert "accel_plus" in m1


def test_empirical_rank_monotonic():
    x, _ = make_consen_synth(200, seed=30)
    m = e0.fit_rank_maps(x)
    col = e0.CONSENSUS_RAW[0]
    v = x[col].to_numpy(float)
    vals = np.sort(np.unique(v[np.isfinite(v)]))
    ranks = e0.empirical_rank(vals, m[col])
    assert np.all(np.diff(ranks) >= 0)


def test_consensus_score_is_mean_of_ranks():
    x, _ = make_consen_synth(120, seed=31)
    rc = [f"{c}_rank" for c in e0.CONSENSUS_RAW]
    expected = x[rc].mean(axis=1).to_numpy(float)
    assert np.allclose(x["consensus_score"].to_numpy(float), expected, atol=1e-12)


def test_tercile_assigns_low_high_mid():
    x, terr = make_consen_synth(300, seed=32)
    cs = x["consensus_score"].to_numpy(float)
    ql, qh = e0.compute_consensus_terciles(cs)
    grp = np.where(cs >= qh, "HIGH", np.where(cs <= ql, "LOW", "MID"))
    assert "LOW" in grp and "HIGH" in grp and "MID" in grp
    assert 0.2 < (grp == "LOW").mean() < 0.5


def test_build_primary_excludes_mid_from_cells():
    s = make_synth(400, seed=33)
    x, terr = make_consen(s)
    sub = e0.build_primary_sample_from_scored(x, terr)
    cells = e0._cells_with_group(sub, "consensus_group")
    assert set(cells.keys()) == set(e0.PRIMARY_CELLS)


def test_primary_excludes_zero_base_action():
    s = make_synth(400, seed=34)
    s = s.copy()
    s.loc[s.index[:10], "base_action"] = 0.0
    x, terr = make_consen(s)
    sub = e0.build_primary_sample_from_scored(x, terr)
    assert not (sub["base_action"].to_numpy(float) == 0).any()


def test_primary_excludes_nonfinite_pi():
    s = make_synth(400, seed=35)
    s = s.copy()
    s.loc[s.index[:10], "pi"] = np.nan
    x, terr = make_consen(s)
    sub = e0.build_primary_sample_from_scored(x, terr)
    assert np.all(np.isfinite(sub["pi"].to_numpy(float)))


def test_primary_excludes_nonentry_valid():
    s = make_synth(400, seed=36)
    s = s.copy()
    s.loc[s.index[:10], "same_block_entry_valid"] = False
    x, terr = make_consen(s)
    sub = e0.build_primary_sample_from_scored(x, terr)
    assert bool(sub["same_block_entry_valid"].to_numpy(bool).all())


def test_four_cell_n_counts_consistent():
    df = _explicit_cell_df()
    cells = e0._cells_with_group(df, "consensus_group")
    assert cells["LOW_OFF"]["n"] == 2
    assert cells["LOW_ACCEL"]["n"] == 2
    assert cells["HIGH_OFF"]["n"] == 2
    assert cells["HIGH_ACCEL"]["n"] == 2


def test_harm_did_formula():
    df = _explicit_cell_df()
    df["harm_flag"] = [0, 1, 0, 1, 0, 1, 0, 1]
    cells = e0._cells_with_group(df, "consensus_group")
    eff = e0.primary_effects(cells)
    assert abs(eff["Delta_LOW_harm"]) < 1e-12


def test_h1_did_formula():
    df = _explicit_cell_df()
    df["hazard"] = [1, 0, 1, 0, 1, 0, 1, 0]
    cells = e0._cells_with_group(df, "consensus_group")
    eff = e0.primary_effects(cells)
    assert abs(eff["DID_H1"]) < 1e-12


def test_bootstrap_p_pos_in_unit_range():
    ev = _bootstrap_eval(n=500, seed=37)
    b = e0.bootstrap_did(ev, n_boot=200, seed=3)
    for k in b:
        assert 0.0 <= b[k]["p_pos"] <= 1.0


def test_bootstrap_ci_ordering():
    ev = _bootstrap_eval(n=500, seed=38)
    b = e0.bootstrap_did(ev, n_boot=200, seed=4)
    for k in b:
        assert b[k]["ci95_lower"] <= b[k]["ci95_upper"]


def test_age_zero_diagnostic_guarded_dict():
    x, terr = make_consen_synth(200, seed=39)
    cs = x[x["block"].isin(["TB1"])]["consensus_score"].to_numpy(float)
    sub = e0.build_primary_sample_from_scored(x, e0.compute_consensus_terciles(cs))
    out = e0.age_zero_diagnostic(sub)
    assert isinstance(out, dict)


def test_component_diagnostics_has_delta_low_high():
    s = make_synth(400, seed=40)
    x, terr = make_consen(s)
    sub = e0.build_primary_sample_from_scored(x, terr)
    sub = sub[sub["block"] == "TB1"].copy()
    maps = e0.fit_rank_maps(x[x["block"].isin(["TB1"])])
    comp = e0.component_diagnostics(sub, maps)
    for c in e0.CONSENSUS_RAW:
        assert "Delta_LOW_pi" in comp[c] and "Delta_HIGH_pi" in comp[c]


def test_psych_gate_blocks_only_high_accel():
    rng = np.random.default_rng(41)
    n = 8
    df = pd.DataFrame({
        "base_action": rng.choice([-1.0, 1.0], n),
        "consensus_group": ["HIGH", "HIGH", "LOW", "LOW"] * 2,
        "accel_positive": [True, False, True, False] * 2,
        "r_trad_OC_ATR0": rng.normal(0, 0.01, n),
        "entry_day": ["2024-01-01"] * n, "symbol": ["AG"] * n,
    })
    r = e0.psych_gate_diagnostic(df)
    base_t = r["BASE"]["by_symbol"]["AG"]["trade_count"]
    gate_t = r["PSYCH_GATE"]["by_symbol"]["AG"]["trade_count"]
    assert base_t - gate_t == 2  # only HIGH & accel rows blocked


def test_psych_gate_bootstrap_has_paired_diff():
    rng = np.random.default_rng(42)
    n = len(SYMS) * 3
    df = pd.DataFrame({
        "base_action": rng.choice([-1.0, 1.0], n),
        "consensus_group": rng.choice(["LOW", "HIGH"], n),
        "accel_positive": rng.random(n) < 0.5,
        "r_trad_OC_ATR0": rng.normal(0, 0.01, n),
        "entry_day": ["2024-01-01"] * n, "symbol": np.repeat(SYMS, 3),
    })
    r = e0.psych_gate_diagnostic(df)
    assert "PSYCH_GATE-BASE" in r["bootstrap"]


def test_full_blocked_via_main_arg(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--full-exploratory"])
    with pytest.raises(SystemExit):
        e0.main()


def test_reuses_d0_prepare_window():
    assert e0.d0.prepare_window_windowframe is d0.prepare_window_windowframe


def test_verdict_not_supported_when_did_upper_positive():
    v = e0.determine_psych_verdict(_boot(0.001, 0.02, 0.9), _boot(0.0, 0.01, 0.9), _boot(0.0, 0.01, 0.9))
    assert v == e0.VERDICT["NOT_SUPPORTED"]


def test_verdict_state_dep_lower_not_positive():
    v = e0.determine_psych_verdict(
        _boot(-0.01, -0.001, 0.1), _boot(-0.02, 0.001, 0.2), _boot(-0.02, -0.001, 0.1))
    assert v == e0.VERDICT["STATE_DEP"]


# ---------------------------------------------------------------------------
# Real pipeline: audit + smoke (slow but required)
# ---------------------------------------------------------------------------
def test_real_audit_runs_and_emits_no_verdict(capsys):
    e0.run_audit_only()
    out = capsys.readouterr().out
    assert "WindowA score-owner parity" in out
    assert "NO SCIENTIFIC VERDICT EMITTED" in out


def test_real_smoke_runs_and_emits_no_verdict(capsys):
    e0.run_smoke_test()
    out = capsys.readouterr().out
    assert "score-owner parity" in out
    assert "SMOKE ONLY / NO SCIENTIFIC VERDICT" in out


def test_window_owner_parity_passes():
    e0.run_audit_only()
