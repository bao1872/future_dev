"""Tests for FUTURE-R4-M15-DIRECTION-MODEL-V1-15SYM-CONFIRMATION (FIX1).

Proves the governance + methodology invariants required by the task:
  - no environment / Candidate / Teacher-DP rerun (source-level guard)
  - fail-closed 15-symbol dataset identity verification
  - common STUDY WINDOW enforced as whole-opportunity eligibility (not a purge)
  - T1/T2 snapped to the canonical 15-minute decision grid (ceil)
  - no oracle_trade_id crosses a Train/Val/Test split (pooled)
  - every retained row AND all four timestamps live in the common window + one split
  - retained per-trade raw weights still sum to 1 (pooled)
  - DIR-M0 = DTP9, DIR-M1 = STRUCT33; no symbol feature
  - TEST never participates in fit / eval_set
  - chunked bootstrap == unchunked reference (identical seed/B)
  - majority baseline is opportunity-weighted
  - required outputs + old->new deltas present
"""

import inspect
import json

import numpy as np
import pandas as pd
import pytest

import research.liquidity_oracle_atlas.train_direction_model_15sym_v1 as M
from research.liquidity_oracle_atlas.build_struct33_dataset_v1 import (
    DTP9 as BUILDER_DTP9,
)
from research.liquidity_oracle_atlas.build_struct33_dataset_v1 import (
    STRUCT33 as BUILDER_STRUCT33,
)


# --------------------------------------------------------------------------- #
# 1. No environment / Candidate / Teacher rerun (source-level guard)
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
# 2. Schemas stay in sync with the builder (frozen source of truth)
# --------------------------------------------------------------------------- #
def test_schemas_match_builder():
    assert tuple(M.DTP9) == tuple(BUILDER_DTP9)
    assert tuple(M.STRUCT33) == tuple(BUILDER_STRUCT33)
    assert len(M.DTP9) == 9
    assert len(M.STRUCT33) == 33
    assert set(M.DTP9).issubset(set(M.STRUCT33))


def test_schema_hash_matches_manifest_formula():
    man = {m["symbol"]: m for m in json.loads(open(M.MANIFEST_PATH).read())}
    assert man["AG"]["struct33_schema_hash"] == M.struct33_schema_hash()


# --------------------------------------------------------------------------- #
# 3. Fail-closed manifest verification
# --------------------------------------------------------------------------- #
def test_manifest_verification_passes_15():
    rep = M.verify_manifest()
    assert len(rep) == 15
    for s, r in rep.items():
        assert r["dataset_sha256_match"], s
        assert r["teacher_sha256_match"], s
        assert r["exec_frame_sha256_match"], s
        assert r["struct33_schema_hash_match"], s
        assert r["builder_provenance_ok"], s


def test_manifest_verification_is_fail_closed(tmp_path):
    man = json.loads(open(M.MANIFEST_PATH).read())
    man[0]["dataset_sha256"] = "0" * 64
    p = tmp_path / "man.json"
    p.write_text(json.dumps(man))
    with pytest.raises(RuntimeError):
        M.verify_manifest(manifest_path=str(p))


# --------------------------------------------------------------------------- #
# 4. Bootstrap + majority helpers (pure)
# --------------------------------------------------------------------------- #
def test_chunked_bootstrap_clusters_by_trade_not_row():
    trade_return = np.array([1.0, 0.0])
    mean, lo, hi = M.bootstrap_trade_returns_chunked(trade_return, B=2000)
    assert abs(mean - 0.5) < 1e-9
    assert lo <= 0.5 <= hi


def test_chunked_bootstrap_is_deterministic():
    tr = np.linspace(-1.0, 1.0, 37)
    a = M.bootstrap_trade_returns_chunked(tr, B=1000, chunk=100)
    b = M.bootstrap_trade_returns_chunked(tr, B=1000, chunk=100)
    assert a == b


def test_chunked_bootstrap_mean_independent_of_chunk():
    tr = np.random.default_rng(7).normal(size=50)
    m1, _, _ = M.bootstrap_trade_returns_chunked(tr, B=1000, chunk=1000)
    m2, _, _ = M.bootstrap_trade_returns_chunked(tr, B=1000, chunk=123)
    assert abs(m1 - m2) < 1e-9


def test_chunked_bootstrap_matches_reference_exactly():
    # Same seed, same B: the chunked bootstrap must equal the unchunked reference
    # on mean AND both CI bounds (numpy Generator.integers is chunk-invariant).
    tr = np.random.default_rng(20260923).normal(size=40)
    ref = M.bootstrap_reference(tr, B=3000)
    for chunk in (1, 7, 250, 3000):
        got = M.bootstrap_trade_returns_chunked(tr, B=3000, chunk=chunk)
        assert np.allclose(ref, got, atol=1e-12), chunk


def test_opportunity_weighted_majority_differs_from_row_weighted():
    y = np.array([1] * 99 + [0] * 3, dtype=np.uint8)
    w = np.array([1.0 / 99] * 99 + [1.0, 1.0, 1.0])
    assert M.opportunity_weighted_majority(y, w) == 0
    assert int(round(float(y.mean()))) == 1


# --------------------------------------------------------------------------- #
# Full-pipeline invariants (runs the real 15-symbol pipeline once)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def run():
    return M.run_direction_models_15sym()


def test_calendar_snapped_to_15m_grid(run):
    cal = run["summary"]["calendar"]
    start = pd.Timestamp(cal["start"])
    end = pd.Timestamp(cal["end"])
    raw1 = pd.Timestamp(cal["raw_t1"])
    raw2 = pd.Timestamp(cal["raw_t2"])
    t1 = pd.Timestamp(cal["t1"])
    t2 = pd.Timestamp(cal["t2"])

    assert cal["snap_rule"] == "CEIL_15MIN"
    assert cal["frac_train"] == 0.6 and cal["frac_val"] == 0.2
    assert start < raw1 < raw2 < end
    assert raw1 <= t1 < raw2 <= t2

    span = end.value - start.value
    assert abs((raw1.value - start.value) - 0.60 * span) <= 1
    assert abs((raw2.value - start.value) - 0.80 * span) <= 1

    for t in (t1, t2):
        assert t.minute % 15 == 0
        assert t.second == 0 and t.microsecond == 0 and t.nanosecond == 0
    assert 0 <= t1.value - raw1.value < 15 * 60 * 10 ** 9
    assert 0 <= t2.value - raw2.value < 15 * 60 * 10 ** 9


def test_common_window_enforced(run):
    ds, kept = run["ds"], run["kept"]
    cal = run["summary"]["calendar"]
    start = pd.Timestamp(cal["start"]).value
    end = pd.Timestamp(cal["end"]).value

    e = np.flatnonzero(kept)
    for col in M._WINDOW_TIME_COLS:
        t = pd.to_datetime(ds[col]).to_numpy(dtype="datetime64[ns]").astype("int64")[e]
        assert bool(((t >= start) & (t <= end)).all()), f"retained row outside window: {col}"

    cw = run["summary"]["common_window"]
    for k in ("rows_outside_common_window", "opportunities_total",
              "opportunities_dropped_common_start", "opportunities_dropped_common_end",
              "opportunities_dropped_common_window", "opportunities_after_window",
              "eligible_rows_after_window"):
        assert k in cw
    assert cw["opportunities_dropped_common_window"] > 0
    assert (cw["opportunities_dropped_common_window"]
            <= cw["opportunities_dropped_common_start"]
            + cw["opportunities_dropped_common_end"])
    assert cw["opportunities_after_window"] == cw["opportunities_total"] - cw["opportunities_dropped_common_window"]


def test_no_oracle_trade_crosses_split(run):
    ds, kept, split = run["ds"], run["kept"], run["split"]
    e = np.flatnonzero(kept)
    tids = ds.loc[e, "oracle_trade_id"].to_numpy(object)
    uniq, inv = np.unique(tids, return_inverse=True)
    for ti in range(len(uniq)):
        member = inv == ti
        assert len(np.unique(split[e][member])) == 1, f"trade {uniq[ti]} spans >1 split"


def test_all_retained_timestamps_same_split(run):
    ds, kept, split, cuts = run["ds"], run["kept"], run["split"], run["cuts"]
    e = np.flatnonzero(kept)
    for col in M._WINDOW_TIME_COLS:
        t = pd.to_datetime(ds[col]).to_numpy(dtype="datetime64[ns]")[e]
        sp = np.searchsorted(cuts, t, side="right")
        assert bool((sp == split[e]).all()), f"timestamp split mismatch: {col}"


def test_retained_per_trade_weight_sum_one(run):
    ds, kept = run["ds"], run["kept"]
    e = np.flatnonzero(kept)
    tids = ds.loc[e, "oracle_trade_id"].to_numpy(object)
    w = ds.loc[e, "sample_weight_raw"].to_numpy(float)
    _, inv = np.unique(tids, return_inverse=True)
    per_trade = np.bincount(inv, weights=w, minlength=len(np.unique(tids)))
    assert np.allclose(per_trade, 1.0, atol=1e-9)


def test_m0_exactly_dtp9_and_m1_exactly_struct33(run):
    fs = run["summary"]["feature_schemas"]
    assert tuple(fs["dtp9"]) == tuple(M.DTP9)
    assert tuple(fs["struct33"]) == tuple(M.STRUCT33)
    assert "symbol" not in M.DTP9 and "symbol" not in M.STRUCT33


def test_test_never_in_fit_or_eval(run):
    train, val, test = run["train_idx"], run["val_idx"], run["test_idx"]
    assert len(np.intersect1d(train, test)) == 0
    assert len(np.intersect1d(val, test)) == 0
    assert len(np.intersect1d(train, val)) == 0
    s = run["summary"]["splits"]["pooled"]
    assert s["test"]["rows"] == len(test)
    assert s["train"]["rows"] == len(train)
    assert s["val"]["rows"] == len(val)


def test_required_outputs_present(run):
    s = run["summary"]
    for k in ("task_id", "base_sha", "calendar", "common_window", "boundary_removal",
              "splits", "params", "feature_schemas", "pre_fix1_deltas",
              "scopes", "manifest_verification"):
        assert k in s
    assert s["task_id"] == M.TASK_ID
    assert s["base_sha"] == M.BASE_SHA

    scopes = s["scopes"]
    assert "POOLED" in scopes and "POOLED_EX_AG" in scopes
    for sym in M.SYMBOLS:
        assert f"SYM_{sym}" in scopes
    assert len(scopes) == 17

    for name in ("POOLED", "POOLED_EX_AG", "SYM_AG"):
        blk = scopes[name]
        for m in ("always_long", "always_short", "majority", "dir_m0", "dir_m1"):
            assert m in blk["models"]
            mb = blk["models"][m]
            for ph in ("ALL", "BEFORE_ENTRY", "AT_ENTRY", "IN_POSITION",
                       "TEACHER_LONG", "TEACHER_SHORT"):
                assert ph in mb, f"{name}/{m}/{ph}"
                for f in ("n_rows", "n_trades", "accuracy", "return_atr",
                          "ci_low", "ci_high"):
                    assert f in mb[ph]
        assert "m1_minus_m0" in blk
        for f in ("mean_delta", "ci_low", "ci_high", "n_trades"):
            assert f in blk["m1_minus_m0"]


def test_majority_baseline_reflects_opportunity_weighted_class(run):
    s = run["summary"]
    maj = s["params"]["majority_class"]
    mb = s["scopes"]["POOLED"]["models"]["majority"]
    if maj == 1:
        assert mb["TEACHER_LONG"]["accuracy"] == 1.0
        assert mb["TEACHER_SHORT"]["accuracy"] == 0.0
    else:
        assert mb["TEACHER_SHORT"]["accuracy"] == 1.0
        assert mb["TEACHER_LONG"]["accuracy"] == 0.0


def test_manifest_verification_recorded_in_summary(run):
    rep = run["summary"]["manifest_verification"]
    assert len(rep) == 15
    for r in rep.values():
        assert r["builder_provenance_ok"] and r["struct33_schema_hash_match"]
        assert r["dataset_sha256_match"] and r["teacher_sha256_match"]


def test_pre_fix1_deltas_present(run):
    d = run["summary"]["pre_fix1_deltas"]
    assert d["pre_fix1_sha"] == M.PRE_FIX1_SHA
    for k in ("train_rows", "train_trades", "val_rows", "val_trades",
              "test_rows", "test_trades"):
        assert "old" in d["splits_pooled"][k] and "new" in d["splits_pooled"][k]
    assert "delta" in d["boundary_trades_dropped_total"]
    for k in ("pooled_m0_return_atr", "pooled_m1_return_atr",
              "pooled_ex_ag_m0_return_atr", "pooled_ex_ag_m1_return_atr"):
        assert "delta" in d[k]
    assert "new" in d["pooled_m1_minus_m0"] and "old" in d["pooled_m1_minus_m0"]
    assert "new" in d["pooled_ex_ag_m1_minus_m0"]
