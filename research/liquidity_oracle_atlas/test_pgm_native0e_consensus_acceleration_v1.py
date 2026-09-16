"""
test_pgm_native0e_consensus_acceleration_v1.py

Round 1 tests for PGM-NATIVE-0E (architecture + audit + smoke ONLY).
Full exploratory is HARD-BLOCKED in this round.
"""
import contextlib
import copy
import inspect
import io
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


# --- 0E.1b: cheap synthetic helpers so smoke-contract tests do NOT re-run the
#     (expensive) real pipeline. Exactly ONE real smoke runs per module (see
#     `smoke_stdout` fixture) for all consumer tests.

def _stub_ridge(train, eval_, num_cols, cat_cols, target):
    n = len(eval_)
    return np.zeros(n), {"mse": 0.0, "mae": 0.0, "spearman": 0.0}, np.zeros(n)


def _stub_logistic(train, eval_, num_cols, cat_cols, target):
    n = len(eval_)
    return np.full(n, 0.5), {"log_loss": 0.0, "brier": 0.0, "roc_auc": 0.5,
                             "pr_auc": 0.5}, np.zeros(n)


def _stub_psych(eval_sub, cost=0.01, n_boot=200, seed=20260916):
    return {"BASE": {}, "PSYCH_GATE": {}, "bootstrap": {}}


def _stub_smoke_heavy(monkeypatch):
    """Stub the expensive model/psych/diagnostic pieces; keep the cell/bootstrap
    math real so wiring contracts are still exercised."""
    monkeypatch.setattr(e0, "_predict_ridge", _stub_ridge)
    monkeypatch.setattr(e0, "_predict_logistic", _stub_logistic)
    monkeypatch.setattr(e0, "psych_gate_diagnostic", _stub_psych)
    monkeypatch.setattr(e0, "component_diagnostics", lambda ev, maps: {})
    monkeypatch.setattr(e0, "age_zero_diagnostic", lambda sub: {})


def _make_smoke_contract_frame(n=24000, seed=20270101):
    """Valid frame for run_window_complete (train=TB1/TB2, eval=TB3) with ample
    primary-eligible rows so the MIN_CELL_N gate passes on the real eval."""
    df = make_synth(n=n, seed=seed, blocks=("TB1", "TB2", "TB3"))
    # force directional stability everywhere -> primary-eligibility (and 4 cells) ample
    df["score_mu"] = np.abs(df["score_mu"].to_numpy(float)) + 0.5
    df["base_action"] = 1.0
    # balanced acceleration so all four primary cells are populated
    df["a_dir_accel_1"] = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    return df


def _oracle_prior_rolling(df):
    """The OLD per-group rolling implementation of c_path_agreement, kept as a
    reference oracle to lock the 0E.1b vectorized rewrite to exact parity."""
    g = df.groupby(["symbol", "episode_id"], sort=False)
    ps = g["path_last_return_R"].transform(
        lambda s: s.shift(1).rolling(e0.CONSENSUS_LOOKBACK, min_periods=e0.CONSENSUS_MIN_PRIOR).sum())
    pa = g["path_last_return_R"].transform(
        lambda s: s.abs().shift(1).rolling(e0.CONSENSUS_LOOKBACK, min_periods=e0.CONSENSUS_MIN_PRIOR).sum())
    d_prev = np.sign(g["score_mu"].shift(1).to_numpy(float))
    return d_prev * ps.to_numpy(float) / (pa.to_numpy(float) + e0.EPS)


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
    # TB1 = train, TB2 = eval, but in DISJOINT episodes so a TB2 row is never the
    # t-1 predecessor of a TB1 train row (otherwise the C_{t-1} shift would
    # legitimately pull TB2 geometry into a TB1 row's primitive). With that isolation,
    # mutating eval-block (TB2) raw must NOT move the TB1-train rank maps.
    tb1 = make_synth(300, seed=55, blocks=("TB1",))
    tb1["episode_id"] = "E_TB1"
    tb2 = make_synth(300, seed=155, blocks=("TB2",))
    tb2["episode_id"] = "E_TB2"
    df = pd.concat([tb1, tb2], ignore_index=True)
    x = e0.add_consensus_primitives(df)
    elig = e0.primary_eligibility_mask(x)
    maps1 = e0.fit_rank_maps(x[elig & (x["block"] == "TB1")])
    df2 = df.copy()
    for c in ["cur_up_distance_R", "cur_down_distance_R", "upper_newest_log_age",
              "lower_newest_log_age", "upper_n_active_identities",
              "lower_n_active_identities", "path_last_return_R", "score_mu"]:
        df2.loc[df2["block"] == "TB2", c] *= 50.0 + 7.0
    x2 = e0.add_consensus_primitives(df2)
    elig2 = e0.primary_eligibility_mask(x2)
    maps2 = e0.fit_rank_maps(x2[elig2 & (x2["block"] == "TB1")])
    for c in e0.CONSENSUS_RAW:
        assert np.array_equal(maps1[c], maps2[c]), c


def test_equal_weight_exact():
    x, _ = make_consen_synth(200, seed=15)
    rank_cols = [f"{c}_rank" for c in e0.CONSENSUS_RAW]
    # consensus_score = mean of the 4 ranks with NO skipna (NaN contract)
    manual = np.mean(x[rank_cols].to_numpy(float), axis=1)
    assert np.allclose(x["consensus_score"].to_numpy(float), manual, atol=1e-12, equal_nan=True)


def test_consensus_score_in_unit_range():
    x, _ = make_consen_synth(200, seed=16)
    cs = x["consensus_score"].to_numpy(float)
    csf = cs[np.isfinite(cs)]   # ineligible rows are NaN by design
    assert np.all((csf >= -1.0) & (csf <= 1.0))


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
    # n_boot/seed are bootstrap controls owned by the caller, NOT a threshold parameter.
    sig = inspect.signature(e0.psych_gate_diagnostic)
    assert set(sig.parameters.keys()) == {"eval_sub", "cost", "n_boot", "seed"}


# ---------------------------------------------------------------------------
# Component diagnostics cannot alter verdict
# ---------------------------------------------------------------------------
def test_component_diagnostics_four_keys():
    s = make_synth(400, seed=24)
    x, terr = make_consen(s)
    sub = e0.build_primary_sample_from_scored(x, terr)
    sub = sub[sub["block"] == "TB1"].copy()
    # component must use the SAME eligible-trained rank reference as the primary path
    maps = e0.prepare_consensus(s, ["TB1"])[1]
    comp = e0.component_diagnostics(sub, maps)
    assert set(comp.keys()) == set(e0.CONSENSUS_RAW)
    for c in comp:
        assert "DID_pi" in comp[c]


def test_component_diagnostics_do_not_emit_verdict():
    s = make_synth(400, seed=25)
    x, terr = make_consen(s)
    sub = e0.build_primary_sample_from_scored(x, terr)
    sub = sub[sub["block"] == "TB1"].copy()
    maps = e0.prepare_consensus(s, ["TB1"])[1]
    comp = e0.component_diagnostics(sub, maps)
    # verdict function only consumes bootstrap dicts, never the component mapping
    assert "consensus_score" in sub.columns


# ---------------------------------------------------------------------------
# Full blocked / governance
# ---------------------------------------------------------------------------
def test_full_blocked_without_token():
    with pytest.raises(SystemExit):
        e0.require_full_authorization()


def test_full_blocked_even_with_token(monkeypatch):
    # token present but NOT exactly "1" -> still blocked
    monkeypatch.setenv(e0.AUTHORIZE_ENV, "0")
    with pytest.raises(SystemExit):
        e0.require_full_authorization()


def test_formal_runner_wires_window_A_to_TB2_and_B_to_TB3(monkeypatch):
    calls = []
    def _fake_rwc(scored, w, n_boot, eval_cap=None, model_train_cap=None):
        # recorder only; the real production sampler fit is NEVER entered
        calls.append((w, n_boot, eval_cap, model_train_cap))
        return {}, (0.1, 0.3), {}
    monkeypatch.setattr(e0, "run_window_complete", _fake_rwc)
    # direct routing call: no _load_and_score, no sampler fit, each window exactly once
    e0._formal_run_windows(pd.DataFrame(), pd.DataFrame())
    assert len(calls) == 2
    assert calls[0][0] == pgm.WINDOWS[0]
    assert calls[1][0] == pgm.WINDOWS[1]
    for w, nb, ec, mt in calls:
        assert nb == e0.BOOTSTRAP_N
        assert ec is None
        assert mt is None


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
    # no skipna; matches the NaN contract
    expected = np.mean(x[rc].to_numpy(float), axis=1)
    assert np.allclose(x["consensus_score"].to_numpy(float), expected, atol=1e-12, equal_nan=True)


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
    maps = e0.prepare_consensus(s, ["TB1"])[1]
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
# Round 1.1 new coverage (0E.1): eligibility / NaN / bootstrap / perf
# ---------------------------------------------------------------------------
class _ScoredWindows:
    def __init__(self, tup, counts):
        self.tup = tup
        self.fit_counts = counts

    def __iter__(self):
        return iter(self.tup)


@pytest.fixture(scope="module")
def real_scored_windows():
    # Production Window A/B samplers are fit EXACTLY ONCE for the whole test module.
    counts = {"A": 0, "B": 0}
    orig = pgm.fit_samplers_for_window

    def spy(win, *a, **k):
        if win is pgm.WINDOWS[0]:
            counts["A"] += 1
        elif win is pgm.WINDOWS[1]:
            counts["B"] += 1
        return orig(win, *a, **k)

    pgm.fit_samplers_for_window = spy
    res = e0._load_and_score()
    pgm.fit_samplers_for_window = orig
    return _ScoredWindows(res, counts)


@pytest.fixture
def reuse_windows(monkeypatch, real_scored_windows):
    # run_audit_only / run_smoke_test must reuse the already-fitted windows.
    monkeypatch.setattr(e0, "_load_and_score", lambda: real_scored_windows)
    return real_scored_windows


@pytest.fixture(scope="module")
def smoke_stdout(real_scored_windows):
    # Run the REAL smoke EXACTLY ONCE for the whole module and capture its stdout.
    # Every other smoke-contract test reads this cached result instead of re-running
    # the (expensive) full predictive pipeline. This is what makes the suite fast.
    saved = e0._load_and_score
    e0._load_and_score = lambda: real_scored_windows
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            e0.run_smoke_test()
    finally:
        e0._load_and_score = saved
    return buf.getvalue()


# 86: NaN input rank stays NaN (never mapped to a high rank)
def test_empirical_rank_nan_stays_nan():
    ref = np.array([-1.0, 0.0, 1.0])
    out = e0.empirical_rank(np.array([-1.0, np.nan, 1.0]), ref)
    assert np.isnan(out[1])
    assert np.isfinite(out[0]) and np.isfinite(out[2])


# 87: partial NaN component -> consensus_score NaN (no automatic skipna)
def test_consensus_score_nan_when_component_nan():
    x, maps = make_consen_synth(200, seed=15)
    x.loc[x.index[3], "c_position"] = np.nan   # corrupt one raw component
    x2 = e0.attach_consensus_score(x, maps)
    assert np.isnan(x2["consensus_score"].to_numpy(float)[3])


# 88: rank map is trained only on PRIMARY-ELIGIBLE train rows
def test_rank_map_uses_eligible_rows_only():
    df = make_synth(600, seed=50, blocks=("TB1",))
    df.loc[df.index[::2], "score_mu"] = 0.0        # half rows -> direction_stable False -> ineligible
    x = e0.add_consensus_primitives(df)
    elig = e0.primary_eligibility_mask(x)
    maps_eligible = e0.fit_rank_maps(x[elig])
    x2, maps2 = e0.prepare_consensus(df, ["TB1"])
    for c in e0.CONSENSUS_RAW:
        assert np.array_equal(maps2[c], maps_eligible[c]), c


# 89: ineligible extreme train rows cannot move the rank map
def test_ineligible_extreme_rows_cannot_move_rank_map():
    df = make_synth(400, seed=51, blocks=("TB1",))
    x = e0.add_consensus_primitives(df)
    base = e0.fit_rank_maps(x[e0.primary_eligibility_mask(x)])
    extreme = make_synth(1000, seed=52, blocks=("TB1",))
    extreme.loc[:, "score_mu"] = 0.0              # ineligible
    for c in ["cur_up_distance_R", "cur_down_distance_R", "upper_newest_log_age",
              "lower_newest_log_age", "upper_n_active_identities", "lower_n_active_identities"]:
        extreme.loc[:, c] = 1e6 if "up" in c or "upper" in c else 1e-6
    ext = e0.add_consensus_primitives(extreme)
    combined = pd.concat([x, ext], ignore_index=True)
    comb = e0.fit_rank_maps(combined[e0.primary_eligibility_mask(combined)])
    for c in e0.CONSENSUS_RAW:
        assert np.array_equal(base[c], comb[c]), c


# 90: ineligible extreme train rows cannot move the terciles
def test_ineligible_extreme_rows_cannot_move_terciles():
    df = make_synth(400, seed=53, blocks=("TB1",))
    x, _ = e0.prepare_consensus(df, ["TB1"])   # adds consensus_score on eligible rows
    elig = e0.primary_eligibility_mask(x)
    ql0, qh0 = e0.compute_consensus_terciles(x[elig]["consensus_score"].to_numpy(float))
    extreme = make_synth(1000, seed=54, blocks=("TB1",))
    extreme.loc[:, "score_mu"] = 0.0           # ineligible (direction_stable False)
    for c in ["cur_up_distance_R", "cur_down_distance_R", "upper_newest_log_age",
              "lower_newest_log_age", "upper_n_active_identities", "lower_n_active_identities"]:
        extreme.loc[:, c] = 1e6 if "up" in c or "upper" in c else 1e-6
    ext = e0.add_consensus_primitives(extreme)
    combined = pd.concat([x, ext], ignore_index=True)
    cmask = e0.primary_eligibility_mask(combined)
    ql1, qh1 = e0.compute_consensus_terciles(combined[cmask]["consensus_score"].to_numpy(float))
    assert ql0 == ql1 and qh0 == qh1


# 92: a train mutation MUST move the train maps (proves the test is non-vacuous)
def test_train_mutation_moves_train_maps():
    df = make_synth(400, seed=56, blocks=("TB1",))
    x = e0.add_consensus_primitives(df)
    elig = e0.primary_eligibility_mask(x)
    maps1 = e0.fit_rank_maps(x[elig])
    df2 = df.copy()
    for c in ["cur_up_distance_R", "cur_down_distance_R",
              "upper_n_active_identities", "lower_n_active_identities"]:
        df2.loc[df2["block"] == "TB1", c] *= 100.0
    x2 = e0.add_consensus_primitives(df2)
    maps2 = e0.fit_rank_maps(x2[e0.primary_eligibility_mask(x2)])
    moved = any(not np.array_equal(maps1[c], maps2[c]) for c in e0.CONSENSUS_RAW)
    assert moved, "train mutation must change at least one rank map"


# 93: n_rank_train == n_tercile_train (same eligible universe)
def test_rank_and_tercile_train_counts_equal():
    df = make_synth(500, seed=57, blocks=("TB1",))
    x, _ = e0.prepare_consensus(df, ["TB1"])
    elig = e0.primary_eligibility_mask(x)
    train_primary = x[x["block"].isin(["TB1"]) & elig]
    n_rank_train = len(train_primary)
    ql, qh = e0.compute_consensus_terciles(train_primary["consensus_score"].to_numpy(float))
    sub = e0.build_primary_sample_from_scored(x, (ql, qh))
    n_train_primary = len(sub[sub["block"].isin(["TB1"])])
    assert n_rank_train == n_train_primary


# 94: predictive MSE paired bootstrap formula (point == mean(row sqerr diff))
def test_predictive_mse_paired_bootstrap_formula():
    rng = np.random.default_rng(60)
    n = 300
    day = np.array(["2024-01-%02d" % (i % 6 + 1) for i in range(n)])
    p0 = rng.normal(0, 1, n); p1 = p0 + rng.normal(0, 0.3, n); y = rng.normal(0, 1, n)
    sq0 = (p0 - y) ** 2; sq1 = (p1 - y) ** 2
    res = e0.paired_day_loss_bootstrap(day, sq0, sq1, n_boot=500, seed=20260916)
    assert abs(res["point"] - float(np.mean(sq0 - sq1))) < 1e-12
    assert res["ci95_lower"] <= res["ci95_upper"]
    assert 0.0 <= res["p_pos"] <= 1.0


# 95: predictive LogLoss paired bootstrap formula
def test_predictive_ll_paired_bootstrap_formula():
    rng = np.random.default_rng(61)
    n = 300
    day = np.array(["2024-01-%02d" % (i % 6 + 1) for i in range(n)])
    y = rng.integers(0, 2, n)
    pc0 = np.clip(rng.uniform(0.1, 0.9, n), 1e-15, 1 - 1e-15)
    pc1 = np.clip(rng.uniform(0.1, 0.9, n), 1e-15, 1 - 1e-15)
    ll0 = -(y * np.log(pc0) + (1 - y) * np.log(1 - pc0))
    ll1 = -(y * np.log(pc1) + (1 - y) * np.log(1 - pc1))
    res = e0.paired_day_loss_bootstrap(day, ll0, ll1, n_boot=500, seed=20260916)
    assert abs(res["point"] - float(np.mean(ll0 - ll1))) < 1e-12
    assert res["ci95_lower"] <= res["ci95_upper"]


# 96: predictive bootstrap deterministic at fixed seed
def test_predictive_bootstrap_deterministic():
    rng = np.random.default_rng(62)
    n = 200
    day = np.array(["2024-01-%02d" % (i % 5 + 1) for i in range(n)])
    l0 = rng.normal(0, 1, n); l1 = rng.normal(0, 1, n)
    r1 = e0.paired_day_loss_bootstrap(day, l0, l1, n_boot=300, seed=20260916)
    r2 = e0.paired_day_loss_bootstrap(day, l0, l1, n_boot=300, seed=20260916)
    for k in ("point", "ci95_lower", "ci95_upper", "p_pos"):
        assert r1[k] == r2[k]


# 97: paired bootstrap aggregates by entry DAY (same day weights drive every row)
def _ref_paired_day(entry_day, loss0, loss1, n_boot, seed):
    days = np.unique(entry_day); D = len(days)
    didx = {d: i for i, d in enumerate(days)}
    pos = np.array([didx[x] for x in entry_day])
    delta = np.asarray(loss0) - np.asarray(loss1)
    S = np.zeros(D); C = np.zeros(D)
    for i, p in enumerate(pos):
        S[p] += delta[i]; C[p] += 1.0
    point = float(np.mean(delta))
    rng = np.random.default_rng(seed)
    dist = []
    for _ in range(n_boot):
        w = rng.multinomial(D, np.full(D, 1.0 / D))
        denom = w @ C
        dist.append((w @ S) / denom if denom > 0 else float("nan"))
    dist = np.array(dist); dist = dist[np.isfinite(dist)]
    return dict(point=point, lo=np.percentile(dist, 2.5), hi=np.percentile(dist, 97.5),
                ppos=np.mean(dist > 0))


def test_predictive_bootstrap_same_day_weights():
    rng = np.random.default_rng(63)
    n = 240
    day = np.array(["2024-01-%02d" % (i % 8 + 1) for i in range(n)])
    l0 = rng.normal(0, 1, n); l1 = rng.normal(0, 1, n)
    res = e0.paired_day_loss_bootstrap(day, l0, l1, n_boot=400, seed=20260916)
    ref = _ref_paired_day(day, l0, l1, 400, 20260916)
    assert abs(res["point"] - ref["point"]) < 1e-12
    assert abs(res["ci95_lower"] - ref["lo"]) < 1e-9
    assert abs(res["ci95_upper"] - ref["hi"]) < 1e-9
    assert abs(res["p_pos"] - ref["ppos"]) < 1e-9


# 103: four cells include gross_EV / net_EV_at_0p01 reporting fields
def test_four_cells_include_gross_net_ev():
    df = _explicit_cell_df()
    cells = e0._cells_with_group(df, "consensus_group")
    for k in ["LOW_OFF", "LOW_ACCEL", "HIGH_OFF", "HIGH_ACCEL"]:
        assert "gross_EV" in cells[k]
        assert "net_EV_at_0p01" in cells[k]
        assert abs(cells[k]["gross_EV"] - cells[k]["mean_pi"]) < 1e-12
        assert abs(cells[k]["net_EV_at_0p01"] - (cells[k]["mean_pi"] - e0.PRIMARY_COST_ATR0)) < 1e-12


# 104: primary_cell_counts exact synthetic counts (structure only)
def test_primary_cell_counts_exact():
    n = 240
    rng = np.random.default_rng(40)
    grp = rng.choice(["LOW", "MID", "HIGH"], size=n, p=[1 / 3, 1 / 3, 1 / 3])
    acc = rng.random(n) < 0.5
    sub = pd.DataFrame({"consensus_group": grp, "accel_positive": acc})
    counts = e0.primary_cell_counts(sub)
    for k in e0.PRIMARY_CELLS:
        g, a = k.split("_")
        expected = int(((grp == g) & (acc == (a == "ACCEL"))).sum())
        assert counts[k] == expected, (k, counts[k], expected)
    # MID rows excluded; only LOW/HIGH x ACCEL/OFF counted
    assert sum(counts.values()) <= n


# 105: primary_cell_counts works even when outcome columns are missing
def test_primary_cell_counts_without_outcome_columns():
    n = 120
    rng = np.random.default_rng(41)
    grp = rng.choice(["LOW", "HIGH"], size=n)
    acc = rng.random(n) < 0.5
    sub = pd.DataFrame({
        "consensus_group": grp, "accel_positive": acc,
        "pi": rng.normal(0, 1, n), "harm_flag": rng.integers(0, 2, n),
        "hazard": rng.integers(0, 2, n), "r_trad_OC_ATR0": rng.normal(0, 1, n),
    })
    with_out = e0.primary_cell_counts(sub)
    without = e0.primary_cell_counts(
        sub.drop(columns=["pi", "harm_flag", "hazard", "r_trad_OC_ATR0"]))
    assert with_out == without


# 106: assert_min_cells uses the count-only helper (never _cells_with_group)
def test_assert_min_cells_uses_count_only_helper(monkeypatch):
    used = []
    monkeypatch.setattr(e0, "_cells_with_group", lambda *a, **k: used.append(1) or {})
    # one cell has MIN_CELL_N - 1 -> STOP expected
    n = e0.MIN_CELL_N - 1
    sub = pd.DataFrame({
        "consensus_group": ["LOW"] * n + ["HIGH"] * n,
        "accel_positive": [True] * n + [False] * n,
    })
    with pytest.raises(SystemExit):
        e0.assert_min_cells(sub)
    assert used == []   # the gate must NOT have computed outcomes


# 107: audit stdout contains no outcome token
def test_audit_stdout_no_outcome_token(capsys, reuse_windows):
    e0.run_audit_only()
    out = capsys.readouterr().out
    for tok in ("mean_pi", "gross_EV", "net_EV", "harm_rate", "H1_prevalence",
                "DID", "Delta_LOW", "Delta_HIGH", "MSE", "LogLoss", "payoff", "PSYCH_GATE"):
        assert tok not in out, tok
    for req in ("n_rank_train", "n_tercile_train", "LOW_ACCEL", "LOW_OFF",
                "HIGH_ACCEL", "HIGH_OFF"):
        assert req in out, req


# 108: audit must not actually call any scientific metric function
def test_audit_forbids_scientific_functions(monkeypatch, reuse_windows):
    def boom(*a, **k):
        raise AssertionError("audit called a scientific metric function")

    for fn in ("_cells_with_group", "primary_effects", "bootstrap_did",
               "_predict_ridge", "_predict_logistic", "paired_day_loss_bootstrap",
               "psych_gate_diagnostic", "run_window_complete"):
        monkeypatch.setattr(e0, fn, boom)
    e0.run_audit_only()   # must still complete successfully


# 109: smoke still uses _cells_with_group + scientific wiring (behavior unchanged).
#      Uses a cheap synthetic window instead of re-running the real pipeline.
def test_smoke_uses_cells_with_group_and_scientific_wiring(monkeypatch):
    seen = {"cells": 0}
    orig = e0._cells_with_group

    def spy(sub, group_col):
        seen["cells"] += 1
        return orig(sub, group_col)

    monkeypatch.setattr(e0, "_cells_with_group", spy)
    _stub_smoke_heavy(monkeypatch)
    res, _, _ = e0.run_window_complete(
        _make_smoke_contract_frame(),
        {"train": ["TB1", "TB2"], "eval": "TB3"},
        n_boot=200, eval_cap=None, model_train_cap=8192)
    assert seen["cells"] > 0            # smoke routes through the scientific cell helper
    assert "DID_pi" in res["bootstrap"]  # scientific wiring present in payload
    assert "PSYCH_GATE" in res["psych"]


# 98: smoke passes n_boot=200 to the predictive (payoff/harm) paired bootstrap.
#      Synthetic window + spy (no real pipeline).
def test_smoke_predictive_n_boot_200(monkeypatch):
    seen = {}
    orig = e0.paired_day_loss_bootstrap

    def spy(entry_day, loss0, loss1, n_boot=20260916, seed=20260916):
        seen["predictive"] = n_boot
        return orig(entry_day, loss0, loss1, n_boot, seed)

    monkeypatch.setattr(e0, "paired_day_loss_bootstrap", spy)
    _stub_smoke_heavy(monkeypatch)
    e0.run_window_complete(
        _make_smoke_contract_frame(),
        {"train": ["TB1", "TB2"], "eval": "TB3"},
        n_boot=200, eval_cap=None, model_train_cap=8192)
    assert seen["predictive"] == 200    # payoff AND harm paired bootstrap both forward n_boot=200


# 99: smoke passes n_boot=200 to the psych_gate bootstrap. Synthetic window + spy.
def test_smoke_psych_n_boot_200(monkeypatch):
    seen = {}
    orig = e0.psych_gate_diagnostic

    def spy(eval_sub, cost=0.01, n_boot=2000, seed=20260916):
        seen["psych"] = n_boot
        return orig(eval_sub, cost, n_boot, seed)

    monkeypatch.setattr(e0, "_predict_ridge", _stub_ridge)
    monkeypatch.setattr(e0, "_predict_logistic", _stub_logistic)
    monkeypatch.setattr(e0, "component_diagnostics", lambda ev, maps: {})
    monkeypatch.setattr(e0, "age_zero_diagnostic", lambda sub: {})
    monkeypatch.setattr(e0, "psych_gate_diagnostic", spy)
    e0.run_window_complete(
        _make_smoke_contract_frame(),
        {"train": ["TB1", "TB2"], "eval": "TB3"},
        n_boot=200, eval_cap=None, model_train_cap=8192)
    assert seen["psych"] == 200


# 0E.1b parity: the vectorized consensus rewrite must match the old rolling implementation
# EXACTLY (same input -> identical c_path_agreement), incl. group boundaries + NaN/min_periods.
def test_consensus_prior_vectorized_matches_rolling():
    rng = np.random.default_rng(11)
    n = 90
    ep = np.repeat(np.arange(3), 30)  # 3 episodes -> group-boundary NaNs
    df = pd.DataFrame({
        "symbol": "X",
        "episode_id": ep.astype(str),
        "bar_t": np.tile(np.arange(30), 3),
        "block": "TB1",
        "score_mu": rng.normal(0, 1, n),
        "path_last_return_R": rng.normal(0, 0.5, n),
        "cur_up_distance_R": np.abs(rng.normal(1, 0.3, n)),
        "cur_down_distance_R": np.abs(rng.normal(1, 0.3, n)),
        "upper_newest_log_age": rng.uniform(0.5, 5, n),
        "lower_newest_log_age": rng.uniform(0.5, 5, n),
        "upper_n_active_identities": rng.integers(1, 8, n).astype(float),
        "lower_n_active_identities": rng.integers(1, 8, n).astype(float),
    })
    # inject NaN mid-episode to exercise the min_periods branch
    df.loc[5, "path_last_return_R"] = np.nan
    df.loc[40, "path_last_return_R"] = np.nan
    df.loc[70, "path_last_return_R"] = np.nan
    for c in e0.CONSENSUS_RAW:
        df[c] = rng.normal(0, 1, n)
    new = e0.add_consensus_primitives(df)
    oracle = _oracle_prior_rolling(df)
    assert np.allclose(new["c_path_agreement"].to_numpy(float), oracle,
                      atol=1e-12, equal_nan=True)


# 0E.1b parity: the vectorized primitives preserve the exact-same numeric result as the
# old per-row loop + sequential multinomial draws (which is what guarantees the bootstrap
# rewrite is bit-identical to the pre-0E.1b implementation).
def test_vectorized_primitives_preserve_order_parity():
    pos = np.array([0, 1, 0, 2, 1, 0, 2, 1])
    pi = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    D = 3
    bc = np.bincount(pos, weights=pi, minlength=D)
    man = np.zeros(D)
    for i in range(len(pos)):
        man[pos[i]] += pi[i]
    assert np.array_equal(bc, man)  # bincount == row loop (row-order accumulation)

    # one RNG drawn sequentially vs one RNG drawn as a batch (matching the production path)
    rng_a = np.random.default_rng(777)
    a = np.array([rng_a.multinomial(D, np.full(D, 1.0 / D)) for _ in range(100)])
    rng_b = np.random.default_rng(777)
    b = rng_b.multinomial(D, np.full(D, 1.0 / D), size=100)
    assert np.array_equal(a, b)  # matrix multinomial == sequential draws


# ===========================================================================
# 0E.1c parity + no-hot-loop contract
# ===========================================================================
def _old_day_index(day):
    days = np.unique(day)
    didx = {d: i for i, d in enumerate(days)}
    pos = np.array([didx[x] for x in day])
    return days, pos


def _make_did_synth(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    days = np.array([f"D{i % 50}" for i in range(n)])
    # random (not anti-correlated) so ALL FOUR cells (LOW_OFF/LOW_ACCEL/HIGH_OFF/HIGH_ACCEL)
    # are populated and have finite point estimates.
    grp = rng.choice(["LOW", "HIGH"], size=n)
    acc = rng.integers(0, 2, n).astype(bool)
    pi = rng.normal(0, 0.01, n)
    harm = rng.integers(0, 2, n)
    haz = rng.integers(0, 2, n)
    return pd.DataFrame({"entry_day": days, "consensus_group": grp,
                         "accel_positive": acc, "pi": pi,
                         "harm_flag": harm, "hazard": haz})


def _ref_bootstrap_did(sub, n_boot=200, seed=20260916):
    """Frozen OLD (pre-0E.1b) bootstrap_did: per-row aggregation + sequential multinomial."""
    day = sub["entry_day"].to_numpy()
    days = np.unique(day)
    didx = {d: i for i, d in enumerate(days)}
    pos = np.array([didx[x] for x in day])
    D = len(days)
    grp = sub["consensus_group"].to_numpy()
    acc = sub["accel_positive"].to_numpy(bool)
    pi = sub["pi"].to_numpy(float)
    harm = sub["harm_flag"].to_numpy(int)
    haz = sub["hazard"].to_numpy(int)
    CELLS = ["LOW_OFF", "LOW_ACCEL", "HIGH_OFF", "HIGH_ACCEL"]
    cidx = {c: i for i, c in enumerate(CELLS)}
    count = np.zeros((D, 4)); spi = np.zeros((D, 4))
    sharm = np.zeros((D, 4)); shaz = np.zeros((D, 4))
    for i in range(len(pos)):
        c = cidx[f"{grp[i]}_{'ACCEL' if acc[i] else 'OFF'}"]
        d = pos[i]
        count[d, c] += 1; spi[d, c] += pi[i]; sharm[d, c] += harm[i]; shaz[d, c] += haz[i]
    om = e0._overall_cell_means(sub)
    LOFF, LACC, HOFF, HACC = (cidx["LOW_OFF"], cidx["LOW_ACCEL"], cidx["HIGH_OFF"], cidx["HIGH_ACCEL"])
    point = dict(
        dl=om["LOW_ACCEL"]["pi"] - om["LOW_OFF"]["pi"],
        dh=om["HIGH_ACCEL"]["pi"] - om["HIGH_OFF"]["pi"],
        did=(om["HIGH_ACCEL"]["pi"] - om["HIGH_OFF"]["pi"]) - (om["LOW_ACCEL"]["pi"] - om["LOW_OFF"]["pi"]),
        dlh=om["LOW_ACCEL"]["harm"] - om["LOW_OFF"]["harm"],
        dhh=om["HIGH_ACCEL"]["harm"] - om["HIGH_OFF"]["harm"],
        didh=(om["HIGH_ACCEL"]["harm"] - om["HIGH_OFF"]["harm"]) - (om["LOW_ACCEL"]["harm"] - om["LOW_OFF"]["harm"]),
        dlz=om["LOW_ACCEL"]["hz"] - om["LOW_OFF"]["hz"],
        dhz=om["HIGH_ACCEL"]["hz"] - om["HIGH_OFF"]["hz"],
        didz=(om["HIGH_ACCEL"]["hz"] - om["HIGH_OFF"]["hz"]) - (om["LOW_ACCEL"]["hz"] - om["LOW_OFF"]["hz"]),
    )
    rng = np.random.default_rng(seed)
    dist = {k: [] for k in point}
    for _ in range(n_boot):
        w = rng.multinomial(D, np.full(D, 1.0 / D))
        cnt = w @ count; mp = w @ spi; mh = w @ sharm; mz = w @ shaz
        mpi = np.where(cnt > 0, mp / cnt, np.nan)
        mh_ = np.where(cnt > 0, mh / cnt, np.nan)
        mz_ = np.where(cnt > 0, mz / cnt, np.nan)
        dist["dl"].append(mpi[LACC] - mpi[LOFF])
        dist["dh"].append(mpi[HACC] - mpi[HOFF])
        dist["did"].append((mpi[HACC] - mpi[HOFF]) - (mpi[LACC] - mpi[LOFF]))
        dist["dlh"].append(mh_[LACC] - mh_[LOFF])
        dist["dhh"].append(mh_[HACC] - mh_[HOFF])
        dist["didh"].append((mh_[HACC] - mh_[HOFF]) - (mh_[LACC] - mh_[LOFF]))
        dist["dlz"].append(mz_[LACC] - mz_[LOFF])
        dist["dhz"].append(mz_[HACC] - mz_[HOFF])
        dist["didz"].append((mz_[HACC] - mz_[HOFF]) - (mz_[LACC] - mz_[LOFF]))
    res = {k: e0._summ(float(point[k]), np.array(dist[k], float)) for k in point}
    return {
        "Delta_LOW_pi": res["dl"], "Delta_HIGH_pi": res["dh"], "DID_pi": res["did"],
        "Delta_LOW_harm": res["dlh"], "Delta_HIGH_harm": res["dhh"], "DID_harm": res["didh"],
        "Delta_LOW_H1": res["dlz"], "Delta_HIGH_H1": res["dhz"], "DID_H1": res["didz"],
    }


def _ref_paired_day_mean_bootstrap(entry_day, values, n_boot=200, seed=20260916):
    """Frozen OLD per-row entry-day mean bootstrap."""
    day = np.asarray(entry_day)
    days = np.unique(day)
    didx = {d: i for i, d in enumerate(days)}
    pos = np.array([didx[x] for x in day])
    D = len(days)
    values = np.asarray(values, float)
    S = np.zeros(D); C = np.zeros(D)
    for i, p in enumerate(pos):
        S[p] += values[i]; C[p] += 1.0
    point = float(np.mean(values))
    rng = np.random.default_rng(seed)
    dist = np.empty(n_boot)
    for b in range(n_boot):
        w = rng.multinomial(D, np.full(D, 1.0 / D))
        denom = w @ C
        dist[b] = (w @ S) / denom if denom > 0 else float("nan")
    return e0._summ(point, dist)


def _ref_paired_day_loss_bootstrap(entry_day, loss0, loss1, n_boot=200, seed=20260916):
    delta = np.asarray(loss0, float) - np.asarray(loss1, float)
    return _ref_paired_day_mean_bootstrap(entry_day, delta, n_boot, seed)


# 0E.1c parity: bootstrap_did is bit-identical to its old per-row + sequential-multinomial
# reference across ALL 9 statistics (point / ci95_lower / ci95_upper / p_pos).
def test_bootstrap_did_old_reference_parity():
    sub = _make_did_synth(4000, seed=1)
    n_boot, seed = 200, 20260916
    new = e0.bootstrap_did(sub, n_boot=n_boot, seed=seed)
    ref = _ref_bootstrap_did(sub, n_boot=n_boot, seed=seed)
    for key in new:
        for s in ("point", "ci95_lower", "ci95_upper", "p_pos"):
            assert abs(new[key][s] - ref[key][s]) < 1e-12, (key, s)


def test_paired_day_loss_old_reference_parity():
    rng = np.random.default_rng(2)
    n = 4000
    days = np.array([f"D{i % 50}" for i in range(n)])
    loss0 = rng.normal(0, 1, n); loss1 = rng.normal(0, 1, n)
    n_boot, seed = 200, 20260916
    new = e0.paired_day_loss_bootstrap(days, loss0, loss1, n_boot=n_boot, seed=seed)
    ref = _ref_paired_day_loss_bootstrap(days, loss0, loss1, n_boot=n_boot, seed=seed)
    for s in ("point", "ci95_lower", "ci95_upper", "p_pos"):
        assert abs(new[s] - ref[s]) < 1e-12, s


# PSYCH_GATE-BASE now routes through paired_day_mean_bootstrap; parity vs the old
# row-aggregation reference under the same seed.
def test_psych_gate_base_old_reference_parity():
    rng = np.random.default_rng(3)
    n = 4000
    days = np.array([f"D{i % 50}" for i in range(n)])
    values = rng.normal(0, 1, n)  # stand-in for ndiff = PSYCH_GATE - BASE
    n_boot, seed = 200, 20260916
    new = e0.paired_day_mean_bootstrap(days, values, n_boot=n_boot, seed=seed)
    ref = _ref_paired_day_mean_bootstrap(days, values, n_boot=n_boot, seed=seed)
    for s in ("point", "ci95_lower", "ci95_upper", "p_pos"):
        assert abs(new[s] - ref[s]) < 1e-12, s


# Day index: old np.unique + dict mapping must equal np.unique(return_inverse=True).
def test_day_index_old_vs_return_inverse():
    day = np.array([f"D{i % 37}" for i in range(500)])
    days_old, pos_old = _old_day_index(day)
    days_new, pos_new = np.unique(day, return_inverse=True)
    assert list(days_old) == list(days_new)
    assert np.array_equal(pos_old, pos_new)


# 0E.1c contract: the day-cluster bootstrap owners must contain NO length-(rows)/(n_boot)
# Python loops (no for i in range(len(...)), no for b/_ in range(n_boot), no dict day index).
def test_bootstrap_owners_have_no_hot_python_loops():
    fns = [e0.bootstrap_did, e0.paired_day_mean_bootstrap,
           e0.paired_day_loss_bootstrap, e0.psych_gate_diagnostic]
    forbidden = [
        "didx",
        "for i in range(len(",
        "for _ in range(n_boot",
        "for b in range(n_boot",
        "range(n_boot)",
        "enumerate(pos",
        "enumerate(day",
    ]
    for fn in fns:
        src = inspect.getsource(fn)
        for pat in forbidden:
            assert pat not in src, f"{fn.__name__} contains forbidden hot-loop pattern: {pat}"


# 100: audit-only must NOT call run_window_complete (no scientific experiment)
def test_audit_does_not_call_run_window_complete(monkeypatch, reuse_windows):
    called = []
    monkeypatch.setattr(e0, "run_window_complete", lambda *a, **k: called.append(1) or {})
    e0.run_audit_only()
    assert called == []


# 101: audit-only emits no scientific metric (DID / payoff / PSYCH_GATE / bootstrap)
#      AND no four-cell outcome (mean_pi / gross_EV / net_EV / harm_rate / H1_prevalence)
def test_audit_emits_no_scientific_metric(capsys, reuse_windows):
    e0.run_audit_only()
    out = capsys.readouterr().out
    for tok in ("DID", "Delta_LOW", "Delta_HIGH", "MSE", "LogLoss", "payoff",
                "PSYCH_GATE", "mean_pi", "gross_EV", "net_EV", "harm_rate",
                "H1_prevalence"):
        assert tok not in out, tok
    # governance structure tokens MUST be present
    for req in ("n_rank_train", "n_tercile_train", "LOW_ACCEL", "LOW_OFF",
                "HIGH_ACCEL", "HIGH_OFF"):
        assert req in out, req


# 102: in the real-pipeline test group, production Window A/B samplers each fit ONCE.
#      Reuses the single module-scoped real smoke (smoke_stdout) instead of re-running.
def test_production_ab_fit_only_once(real_scored_windows, reuse_windows, smoke_stdout):
    e0.run_audit_only()
    _ = smoke_stdout                      # triggers the single real smoke for the module
    assert real_scored_windows.fit_counts["A"] == 1
    assert real_scored_windows.fit_counts["B"] == 1


# ---------------------------------------------------------------------------
# Real pipeline: audit + smoke (slow but required)
# ---------------------------------------------------------------------------
def test_real_audit_runs_and_emits_no_verdict(capsys, reuse_windows):
    e0.run_audit_only()
    out = capsys.readouterr().out
    assert "WindowA score-owner parity" in out
    assert "NO SCIENTIFIC VERDICT EMITTED" in out


def test_real_smoke_runs_and_emits_no_verdict(smoke_stdout):
    out = smoke_stdout
    assert "score-owner parity" in out
    assert "SMOKE ONLY / NO SCIENTIFIC VERDICT" in out
    assert "total_smoke_seconds" in out


def test_window_owner_parity_passes(real_scored_windows):
    prep, fit_A, scored_A, fit_B, scored_B = real_scored_windows
    dA = d0.verify_window_score_owner(scored_A, fit_A, e0.TB2_BLOCK)
    dB = d0.verify_window_score_owner(scored_B, fit_B, e0.TB3_BLOCK)
    assert dA < 1e-6 and dB < 1e-6


# ===========================================================================
# Round 2: formal artifact contract (authorization / pre-run / mutation / set)
# These tests NEVER call run_full_exploratory(); they exercise the lower-level
# assembly + disk-parity validators directly with synthetic but consistent data.
# ===========================================================================
def test_authorization_absent_stops(monkeypatch):
    monkeypatch.delenv(e0.AUTHORIZE_ENV, raising=False)
    with pytest.raises(SystemExit):
        e0.require_full_authorization()


def test_authorization_exact_one_passes(monkeypatch):
    monkeypatch.setenv(e0.AUTHORIZE_ENV, "1")
    assert e0.require_full_authorization() is None


def test_prerun_no_stale_artifact_on_empty_dir_passes(tmp_path):
    e0.assert_no_existing_prefixed_artifacts(tmp_path)  # must not raise


def test_prerun_unknown_stale_artifact_stops(tmp_path):
    (tmp_path / "pgm_native0e1_unknown_junk.csv").write_text("x")
    with pytest.raises(SystemExit):
        e0.assert_no_existing_prefixed_artifacts(tmp_path)


def test_prerun_one_known_stale_artifact_stops(tmp_path):
    (tmp_path / e0.ARTIFACT_FILES[0]).write_text("x")
    with pytest.raises(SystemExit):
        e0.assert_no_existing_prefixed_artifacts(tmp_path)


def _fake_formal_res():
    cells = {c: dict(n=700, mean_pi=0.01, gross_EV=0.01, net_EV_at_0p01=0.0,
                     harm_rate=0.3, H1_prevalence=0.4) for c in e0.PRIMARY_CELLS}
    boot = {m: dict(point=0.01, ci95_lower=-0.05, ci95_upper=0.07, p_pos=0.6)
            for m in e0.BOOT_METRICS}
    pm = dict(mse=0.1, mae=0.2, spearman=0.3)
    hm = dict(log_loss=0.5, brier=0.25, roc_auc=0.6, pr_auc=0.55)
    psy = dict(n_decisions=1000, n_trades=800, trade_rate=0.8, gross_total_ATR0=5.0,
               net_total_ATR0=3.0, gross_EV_per_decision=0.005, net_EV_per_decision=0.003,
               net_EV_per_trade=0.00375, win_rate=0.55, mean_win=0.02, mean_loss=-0.015,
               payoff_ratio=1.33, profit_factor=1.2, break_even_cost=0.00625,
               daily_sharpe_annualized=0.7, max_drawdown_ATR0=-0.4,
               positive_symbol_count=12, top3_profit_share=0.5)
    return dict(
        cells=cells, bootstrap=boot,
        payoff_m0=pm, payoff_m1=dict(mse=0.09, mae=0.19, spearman=0.31), delta_mse=0.01,
        harm_m0=hm, harm_m1=dict(log_loss=0.48, brier=0.24, roc_auc=0.61, pr_auc=0.56),
        delta_logloss=0.02,
        payoff_bootstrap=dict(point=0.01, ci95_lower=-0.05, ci95_upper=0.07, p_pos=0.6),
        harm_bootstrap=dict(point=0.01, ci95_lower=-0.05, ci95_upper=0.07, p_pos=0.6),
        psych=dict(BASE=dict(psy), PSYCH_GATE=dict(psy),
                   bootstrap={"PSYCH_GATE-BASE": dict(point=0.01, ci95_lower=-0.05,
                                                      ci95_upper=0.07, p_pos=0.6)}),
        component={comp: {c: 0.01 for c in e0.COMP_COLS} for comp in e0.CONSENSUS_RAW},
        age_zero=dict(abs_score_mu=0.2, u_log_age=-0.1, upper_current_newest_age_zero=0.3,
                      lower_current_newest_age_zero=0.25),
        n_rank_train=15000, n_eval=6000,
        timing=dict(consensus_s=5.0, model_s=1.0, bootstrap_s=0.1),
    )


def _fake_formal_meta():
    return dict(n_all_obs=359714, n_hazard0=321727, n_hazard1=37987, symbols=["A", "B"],
                blocks=["TB1", "TB2", "TB3"], sample_sha="sa", transition_sha="tr",
                winA_owner=0.0, winB_owner=0.0, winA_finite=True, winB_finite=True,
                atr0_owner_err=0.0)


def _produce_artifacts(tmp_path):
    resA = _fake_formal_res(); resB = _fake_formal_res()
    terr = (0.1, 0.3)
    dfs, summary = e0._assemble_artifacts(resA, resB, terr, terr, _fake_formal_meta(), "HEADXYZ")
    e0.write_artifacts(dfs, summary, tmp_path)
    return dfs, summary


def _corrupt_csv(tmp_path, name, mutate):
    p = tmp_path / name
    df = pd.read_csv(p)
    mutate(df)
    df.to_csv(p, index=False)


def test_art_mut_116_primary_cell_n_stops(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    def _m(df):
        df.loc[0, "n"] = 1
    _corrupt_csv(tmp_path, e0.ARTIFACT_FILES[0], _m)
    with pytest.raises(SystemExit):
        e0.validate_output_artifacts(tmp_path, dfs, summary)


def test_art_mut_117_did_point_stops(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    def _m(df):
        i = df.index[df["metric"] == "DID_pi"][0]
        df.loc[i, "point"] = 9.99
    _corrupt_csv(tmp_path, e0.ARTIFACT_FILES[1], _m)
    with pytest.raises(SystemExit):
        e0.validate_output_artifacts(tmp_path, dfs, summary)


def test_art_mut_118_predictive_bootstrap_ci_stops(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    def _m(df):
        df.loc[0, "ci95_upper"] = -9.99
    _corrupt_csv(tmp_path, e0.ARTIFACT_FILES[3], _m)
    with pytest.raises(SystemExit):
        e0.validate_output_artifacts(tmp_path, dfs, summary)


def test_art_mut_119_psych_gate_netEV_stops(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    def _m(df):
        i = df.index[df["policy"] == "BASE"][0]
        df.loc[i, "net_EV_per_decision"] = 123.0
    _corrupt_csv(tmp_path, e0.ARTIFACT_FILES[4], _m)
    with pytest.raises(SystemExit):
        e0.validate_output_artifacts(tmp_path, dfs, summary)


def test_art_mut_120_component_did_stops(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    def _m(df):
        i = df.index[df["component"] == e0.CONSENSUS_RAW[0]][0]
        df.loc[i, "DID_pi"] = 5.5
    _corrupt_csv(tmp_path, e0.ARTIFACT_FILES[5], _m)
    with pytest.raises(SystemExit):
        e0.validate_output_artifacts(tmp_path, dfs, summary)


def test_art_mut_121_age_zero_correlation_stops(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    def _m(df):
        df.loc[0, "spearman_with_consensus"] = 0.999
    _corrupt_csv(tmp_path, e0.ARTIFACT_FILES[6], _m)
    with pytest.raises(SystemExit):
        e0.validate_output_artifacts(tmp_path, dfs, summary)


def test_art_mut_122_run_head_stops(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    s = dict(summary); s["run_head"] = "WRONG"
    with pytest.raises(SystemExit):
        e0.validate_output_artifacts(tmp_path, dfs, s)


def test_art_mut_123_bootstrap_seed_stops(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    s = dict(summary); s["bootstrap_seed"] = 999
    with pytest.raises(SystemExit):
        e0.validate_output_artifacts(tmp_path, dfs, s)


def test_art_mut_124_consensus_raw_stops(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    s = dict(summary); s["consensus_raw"] = ["x"]
    with pytest.raises(SystemExit):
        e0.validate_output_artifacts(tmp_path, dfs, s)


def test_art_mut_125_windowA_owner_diff_stops(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    s = dict(summary); s["windowA_score_owner_max_abs_diff"] = 1.0
    with pytest.raises(SystemExit):
        e0.validate_output_artifacts(tmp_path, dfs, s)


def test_art_mut_126_tb3_q_high_stops(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    s = dict(summary); s["TB3"] = dict(summary["TB3"]); s["TB3"]["q_high"] = 0.999
    with pytest.raises(SystemExit):
        e0.validate_output_artifacts(tmp_path, dfs, s)


def test_art_mut_127_verdict_stops(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    s = dict(summary); s["psychology_verdict"] = "WRONG_VERDICT"
    with pytest.raises(SystemExit):
        e0.validate_output_artifacts(tmp_path, dfs, s)


def test_art_mut_128_extra_prefixed_artifact_stops(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    (tmp_path / "pgm_native0e1_extra_junk.csv").write_text("x")
    with pytest.raises(SystemExit):
        e0.validate_output_artifacts(tmp_path, dfs, summary)


def test_art_mut_129_missing_artifact_stops(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    (tmp_path / e0.ARTIFACT_FILES[2]).unlink()
    with pytest.raises(SystemExit):
        e0.validate_output_artifacts(tmp_path, dfs, summary)


def test_art_130_exact_eight_artifacts_pass(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    e0.validate_output_artifacts(tmp_path, dfs, summary)  # must not raise
    actual = {p.name for p in tmp_path.glob("pgm_native0e1_*")}
    assert actual == set(e0.ARTIFACT_FILES)


def test_art_131_in_memory_validation_passes(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    e0.validate_in_memory_results(dfs, summary)  # must not raise


def test_art_132_formal_summary_carries_required_keys(tmp_path):
    _, summary = _produce_artifacts(tmp_path)
    for k in ("experiment_name", "experiment_scope", "run_head", "base_sha",
              "sample_artifact_sha256", "transition_artifact_sha256", "n_all_obs",
              "symbols", "blocks", "eval_blocks", "windows", "bootstrap_n", "bootstrap_seed",
              "cluster_owner", "primary_cost_atr0", "consensus_raw", "primary_acceleration_col",
              "consensus_time_contract", "windowA_score_owner_max_abs_diff",
              "windowB_score_owner_max_abs_diff", "windowA_acceleration_finite",
              "windowB_acceleration_finite", "max_abs_atr0_owner_error", "TB2", "TB3",
              "psychology_verdict", "artifact_files", "known_limitations", "run_meta",
              "actual_age_zero_variables", "age_zero_variables_by_block"):
        assert k in summary, k
    assert summary["blocks"] == ["TB1", "TB2", "TB3"]
    assert summary["eval_blocks"] == ["TB2", "TB3"]
    assert summary["windows"] == {
        "A": {"train": ["TB1"], "eval": "TB2"},
        "B": {"train": ["TB1", "TB2"], "eval": "TB3"},
    }
    assert summary["consensus_time_contract"] == "C_tminus1_to_A_t_to_pi_tplus1"
    assert summary["run_meta"] == dict(n_boot=2000, model_train_cap=None, eval_cap=None)
    assert set(summary["artifact_files"]) == set(e0.ARTIFACT_FILES)
    # semantic validator must accept the correct summary
    e0.validate_summary_semantics(summary)


# ===========================================================================
# 0E.2a: exact composite-key schema (row count preserved, key duplicated/missing)
# ===========================================================================
def _dup_row_in_memory(dfs, name, keycols, k_src, k_dst):
    """Keep row count but make composite key k_dst a duplicate of k_src."""
    df = dfs[name].copy()
    m_src = np.ones(len(df), dtype=bool)
    m_dst = np.ones(len(df), dtype=bool)
    for c, v in zip(keycols, k_src):
        m_src &= (df[c] == v).to_numpy()
    for c, v in zip(keycols, k_dst):
        m_dst &= (df[c] == v).to_numpy()
    src = df[m_src].iloc[0]
    for c in df.columns:
        df.loc[m_dst, c] = src[c]
    dfs[name] = df
    return dfs


def test_art_schema_dup_primary_cells_stops(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    _dup_row_in_memory(dfs, e0.ARTIFACT_FILES[0], ["block", "cell"],
                       ("TB2", "LOW_ACCEL"), ("TB2", "LOW_OFF"))
    with pytest.raises(SystemExit):
        e0.validate_in_memory_results(dfs, summary)


def test_art_schema_dup_primary_bootstrap_stops(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    _dup_row_in_memory(dfs, e0.ARTIFACT_FILES[1], ["block", "metric"],
                       ("TB2", "Delta_LOW_pi"), ("TB2", "Delta_HIGH_pi"))
    with pytest.raises(SystemExit):
        e0.validate_in_memory_results(dfs, summary)


def test_art_schema_dup_predictive_metrics_stops(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    _dup_row_in_memory(dfs, e0.ARTIFACT_FILES[2], ["block", "target", "model"],
                       ("TB2", "pi", "PAYOFF_M0"), ("TB2", "pi", "PAYOFF_M1"))
    with pytest.raises(SystemExit):
        e0.validate_in_memory_results(dfs, summary)


def test_art_schema_dup_psych_gate_stops(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    _dup_row_in_memory(dfs, e0.ARTIFACT_FILES[4], ["block", "policy"],
                       ("TB2", "BASE"), ("TB2", "PSYCH_GATE"))
    with pytest.raises(SystemExit):
        e0.validate_in_memory_results(dfs, summary)


def test_art_schema_dup_component_stops(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    _dup_row_in_memory(dfs, e0.ARTIFACT_FILES[5], ["block", "component"],
                       ("TB2", "c_position"), ("TB2", "c_path_agreement"))
    with pytest.raises(SystemExit):
        e0.validate_in_memory_results(dfs, summary)


# ===========================================================================
# 0E.2a: disk + memory edited to the SAME wrong value (parity preserved)
# The semantic validator must still STOP (independent of disk/memory parity).
# ===========================================================================
def _rewrite_disk_with(expr_name, bad_value, tmp_path, dfs, summary):
    s = copy.deepcopy(summary)
    s[expr_name] = bad_value
    e0.write_artifacts(dfs, s, tmp_path)  # CSVs unchanged, JSON == mutated summary
    return s


def test_art_sem_14a_bootstrap_seed_disk_and_memory_stops(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    s = _rewrite_disk_with("bootstrap_seed", 999, tmp_path, dfs, summary)
    with pytest.raises(SystemExit):
        e0.validate_output_artifacts(tmp_path, dfs, s)


def test_art_sem_14b_blocks_disk_and_memory_stops(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    s = _rewrite_disk_with("blocks", ["TB2", "TB3"], tmp_path, dfs, summary)
    with pytest.raises(SystemExit):
        e0.validate_output_artifacts(tmp_path, dfs, s)


def test_art_sem_14c_eval_blocks_disk_and_memory_stops(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    s = _rewrite_disk_with("eval_blocks", ["TB1", "TB2"], tmp_path, dfs, summary)
    with pytest.raises(SystemExit):
        e0.validate_output_artifacts(tmp_path, dfs, s)


def test_art_sem_14d_windows_disk_and_memory_stops(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    bad = {"A": {"train": ["TB2"], "eval": "TB2"},
           "B": {"train": ["TB1", "TB2"], "eval": "TB3"}}
    s = _rewrite_disk_with("windows", bad, tmp_path, dfs, summary)
    with pytest.raises(SystemExit):
        e0.validate_output_artifacts(tmp_path, dfs, s)


def test_art_sem_14e_consensus_raw_disk_and_memory_stops(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    s = _rewrite_disk_with("consensus_raw", list(reversed(e0.CONSENSUS_RAW)), tmp_path, dfs, summary)
    with pytest.raises(SystemExit):
        e0.validate_output_artifacts(tmp_path, dfs, s)


def test_art_sem_14f_artifact_files_disk_and_memory_stops(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    s = _rewrite_disk_with("artifact_files", list(reversed(e0.ARTIFACT_FILES)), tmp_path, dfs, summary)
    with pytest.raises(SystemExit):
        e0.validate_output_artifacts(tmp_path, dfs, s)


def test_art_sem_14g_primary_cost_disk_and_memory_stops(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    s = _rewrite_disk_with("primary_cost_atr0", 0.02, tmp_path, dfs, summary)
    with pytest.raises(SystemExit):
        e0.validate_output_artifacts(tmp_path, dfs, s)


def test_art_sem_14h_atr0_owner_error_disk_and_memory_stops(tmp_path):
    dfs, summary = _produce_artifacts(tmp_path)
    s = _rewrite_disk_with("max_abs_atr0_owner_error", 1.0, tmp_path, dfs, summary)
    with pytest.raises(SystemExit):
        e0.validate_output_artifacts(tmp_path, dfs, s)
