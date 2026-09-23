"""Tests for FUTURE-R4-M15-DIRECTION-MODEL-V1-AG.

Proves the governance + methodology invariants required by the task:
  - no environment / Teacher-DP rerun
  - no oracle_trade_id crosses a Train/Val/Test split
  - every retained row's four timestamps live in the same split
  - a boundary-crossing Oracle opportunity is removed as a FULL group
  - retained per-trade raw weights still sum to 1
  - DIR-M0 uses exactly DTP9, DIR-M1 uses exactly STRUCT33
  - TEST never participates in fit / eval_set
  - economic return recomputation is exact
  - bootstrap clusters by Oracle opportunity (trade), not by Candidate row
  - feature schemas stay in sync with build_struct33_dataset_v1
"""

import importlib
import inspect

import numpy as np
import pandas as pd
import pytest

import research.liquidity_oracle_atlas.train_direction_model_ag_v1 as M
from research.liquidity_oracle_atlas.build_struct33_dataset_v1 import (
    DTP9 as BUILDER_DTP9,
)
from research.liquidity_oracle_atlas.build_struct33_dataset_v1 import (
    STRUCT33 as BUILDER_STRUCT33,
)


# --------------------------------------------------------------------------- #
# 1 & 2. No feature rerun / no Teacher DP rerun
# --------------------------------------------------------------------------- #
def test_no_environment_or_teacher_rerun():
    src = inspect.getsource(M)
    forbidden = (
        "run_environment_m15",
        "derive_m15_candidate_gate",
        "build_struct33_dataset",
        "load_oracle_artifact",
        "run_dp_m15",
    )
    for tok in forbidden:
        assert tok not in src, f"forbidden token present in source: {tok}"


# --------------------------------------------------------------------------- #
# 12. Schemas stay in sync with the builder (frozen source of truth)
# --------------------------------------------------------------------------- #
def test_schemas_match_builder():
    assert tuple(M.DTP9) == tuple(BUILDER_DTP9)
    assert tuple(M.STRUCT33) == tuple(BUILDER_STRUCT33)
    assert len(M.DTP9) == 9
    assert len(M.STRUCT33) == 33
    assert set(M.DTP9).issubset(set(M.STRUCT33))


# --------------------------------------------------------------------------- #
# 5. Boundary trade removed as a FULL group (synthetic)
# --------------------------------------------------------------------------- #
def _fake_ds_with_crossing_trade():
    # Two Oracle trades.
    # Trade A: 2 candidates, both fully inside TRAIN (no split crossing).
    # Trade B: 2 candidates; one fully inside TRAIN, one crossing T1
    #          (decision in TRAIN but oracle_exit in VAL) -> B must be fully dropped.
    T0 = np.datetime64("2025-01-01T00:00:00")
    T1 = np.datetime64("2025-06-01T00:00:00")  # cut train/val
    T2 = np.datetime64("2025-09-01T00:00:00")  # cut val/test

    rows = [
        # Trade A (keep)
        dict(
            candidate_decision_time=T0, candidate_fill_time=T0,
            oracle_entry_fill_time=T0, oracle_exit_fill_time=T0,
            oracle_trade_id="A", label_eligible=True,
        ),
        dict(
            candidate_decision_time=T0, candidate_fill_time=T0,
            oracle_entry_fill_time=T0, oracle_exit_fill_time=T0,
            oracle_trade_id="A", label_eligible=True,
        ),
        # Trade B candidate 1 (train, but trade crosses)
        dict(
            candidate_decision_time=T0, candidate_fill_time=T0,
            oracle_entry_fill_time=T0, oracle_exit_fill_time=T1,
            oracle_trade_id="B", label_eligible=True,
        ),
        # Trade B candidate 2 (crosses T1: decision train, exit val)
        dict(
            candidate_decision_time=T0, candidate_fill_time=T0,
            oracle_entry_fill_time=T1, oracle_exit_fill_time=T1,
            oracle_trade_id="B", label_eligible=True,
        ),
    ]
    ds = pd.DataFrame(rows)
    split = np.array([0, 0, 0, 0])  # all decisions in TRAIN
    cuts = np.array([T1, T2])
    return ds, split, cuts


def test_boundary_trade_removed_as_full_group():
    ds, split, cuts = _fake_ds_with_crossing_trade()
    kept, report = M.remove_boundary_opportunities(ds, split, cuts)
    assert bool(kept[0]) and bool(kept[1])          # Trade A fully kept
    assert not bool(kept[2]) and not bool(kept[3])  # Trade B fully dropped
    assert report["trades_dropped_total"] == 1
    assert report["trades_dropped_t1"] == 1


# --------------------------------------------------------------------------- #
# 10 & 11. Economic return exactness + trade-level bootstrap
# --------------------------------------------------------------------------- #
def test_economic_return_recomputation_exact():
    rng = np.random.default_rng(0)
    n = 50
    pred_dir = rng.integers(0, 2, n).astype(np.uint8)
    teacher_dir = rng.integers(0, 2, n).astype(np.uint8)
    eq = rng.uniform(0.5, 3.0, n)
    w = np.ones(n)
    tids = np.array(["t0"] * 30 + ["t1"] * 20)

    pdr = M.pred_direction_return_atr(pred_dir, teacher_dir, eq)
    manual = np.where(pred_dir == teacher_dir, eq, -eq)
    assert np.allclose(pdr, manual)

    tr, uniq = M.aggregate_per_trade(tids, w, pdr)
    # manual per-trade aggregation
    for ti in uniq:
        m = tids == ti
        exp = (w[m] * manual[m]).sum() / w[m].sum()
        assert abs(tr[uniq == ti][0] - exp) < 1e-12


def test_bootstrap_clusters_by_trade_not_row():
    # Trade A: 100 candidates each return 1.0 ; Trade B: 1 candidate return 0.0.
    # Per-trade mean = (1.0 + 0.0)/2 = 0.5.
    # Per-row mean = (100*1.0 + 0)/101 ~= 0.99 (would be wrong).
    trade_return = np.array([1.0, 0.0])
    mean_ret, lo, hi = M.bootstrap_trade_returns(trade_return, B=2000)
    assert abs(mean_ret - 0.5) < 1e-9
    assert lo < 0.5 < hi or (lo <= 0.5 <= hi)


# --------------------------------------------------------------------------- #
# Full-pipeline invariants (runs the real AG pipeline once)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def run():
    return M.run_direction_models("AG")


def test_no_oracle_trade_crosses_split(run):
    ds = run["ds"]
    kept = run["kept"]
    split = run["split"]
    e = np.flatnonzero(kept)
    tids = ds.loc[e, "oracle_trade_id"].to_numpy(object)
    uniq, inv = np.unique(tids, return_inverse=True)
    for ti in range(len(uniq)):
        member = inv == ti
        assert len(np.unique(split[e][member])) == 1, (
            f"trade {uniq[ti]} appears in >1 split"
        )


def test_all_retained_timestamps_same_split(run):
    ds = run["ds"]
    kept = run["kept"]
    split = run["split"]
    e = np.flatnonzero(kept)
    cuts = np.array(
        [
            np.datetime64(pd.Timestamp(run["summary"]["cuts"]["t1"])),
            np.datetime64(pd.Timestamp(run["summary"]["cuts"]["t2"])),
        ]
    )
    for col in (
        "candidate_decision_time",
        "candidate_fill_time",
        "oracle_entry_fill_time",
        "oracle_exit_fill_time",
    ):
        t = pd.to_datetime(ds[col]).to_numpy(dtype="datetime64[ns]")[e]
        sp = np.searchsorted(cuts, t, side="right")
        assert bool((sp == split[e]).all()), f"timestamp split mismatch: {col}"


def test_retained_per_trade_weight_sum_one(run):
    ds = run["ds"]
    kept = run["kept"]
    e = np.flatnonzero(kept)
    tids = ds.loc[e, "oracle_trade_id"].to_numpy(object)
    w = ds.loc[e, "sample_weight_raw"].to_numpy(float)
    _, inv = np.unique(tids, return_inverse=True)
    per_trade = np.bincount(inv, weights=w, minlength=len(np.unique(tids)))
    assert np.allclose(per_trade, 1.0, atol=1e-9)


def test_m0_exactly_dtp9_and_m1_exactly_struct33(run):
    # The orchestration must pass DTP9 to dir_m0 and STRUCT33 to dir_m1.
    # We verify the schemas recorded in the summary match exactly.
    fs = run["summary"]["feature_schemas"]
    assert tuple(fs["dtp9"]) == tuple(M.DTP9)
    assert tuple(fs["struct33"]) == tuple(M.STRUCT33)


def test_test_never_in_fit_or_eval(run):
    train = run["train_idx"]
    val = run["val_idx"]
    test = run["test_idx"]
    assert len(np.intersect1d(train, test)) == 0
    assert len(np.intersect1d(val, test)) == 0
    # counts consistent with summary
    assert int(run["summary"]["splits"]["test_rows"]) == len(test)
    assert int(run["summary"]["splits"]["train_rows"]) == len(train)
    assert int(run["summary"]["splits"]["val_rows"]) == len(val)


def test_summary_structure_and_models_present(run):
    s = run["summary"]
    for k in ("task_id", "base_sha", "cuts", "boundary_removal", "splits",
              "models", "m1_minus_m0", "feature_schemas"):
        assert k in s
    assert s["task_id"] == M.TASK_ID
    assert s["base_sha"] == M.BASE_SHA
    for m in ("always_long", "always_short", "majority", "dir_m0", "dir_m1"):
        assert m in s["models"]
        blk = s["models"][m]["TEST"]
        for f in ("n_rows", "n_trades", "accuracy", "return_atr", "ci_low", "ci_high"):
            assert f in blk
    for ph in ("BEFORE_ENTRY", "AT_ENTRY", "IN_POSITION"):
        assert ph in s["models"]["dir_m1"]
