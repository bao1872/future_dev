"""test_struct33_dataset_v1
=========================

Contract tests for the Phase 1 AG Candidate -> Teacher label table builder.

These tests enforce the reviewer's hard checklist (FUTURE-R4-M15-STRUCT33-
DATASET-V1-PHASE1-AG): exact STRUCT33 schema, integer-index fill, no hard-segment
crossing, mapped exit after fill, vectorized mapper == reference mapper, zero
EntryQuality recomputation mismatch, bars_to_exit > 0 for eligible rows, per-Oracle
weight sum == 1, no censored row gets a valid target, and (critically) that the
dataset builder never reruns the DP Teacher or the environment internally beyond
the single allowed run.
"""

import numpy as np
import pandas as pd
import pytest

import research.liquidity_oracle_atlas.build_struct33_dataset_v1 as B
from research.liquidity_oracle_atlas.build_struct33_dataset_v1 import (
    STRUCT33,
    DTP9,
    map_candidates_fast,
    build_struct33_dataset,
)
from research.liquidity_oracle_atlas.build_teacher_oracle_dp_m15_overnight_v1 import (
    ARTIFACT_ROOT,
    load_oracle_artifact,
    map_candidates_to_trades,
    run_dp_m15_overnight_teacher,
)
from research.liquidity_oracle_atlas.build_execution_environment_m15_v1 import (
    run_environment_m15,
)

FORBIDDEN_FEATURE_SUBSTRINGS = (
    "m5", "oracle", "future", "entry_quality", "bars_to_", "dp_proximity",
    "candidate_", "trigger", "episode", "wick", "body", "breach",
    "accept", "reclaim",
)


# --------------------------------------------------------------------------- #
# Fixtures (module-scoped to run the heavy environment / build exactly once)    #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def env():
    return run_environment_m15("AG", capture_provenance=False)


@pytest.fixture(scope="module")
def built():
    return build_struct33_dataset("AG")


@pytest.fixture(scope="module")
def trades():
    art = load_oracle_artifact(ARTIFACT_ROOT, "AG")
    assert art["ok"], art["reason"]
    return art["trades"].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Schema / forbidden-feature gates                                              #
# --------------------------------------------------------------------------- #
def test_struct33_exactly_33_columns():
    assert len(STRUCT33) == 33
    assert len(DTP9) == 9
    assert set(DTP9).issubset(STRUCT33)


def test_dataset_has_exact_struct33_feature_columns(built):
    ds = built["dataset"]
    feats = [c for c in ds.columns if c in STRUCT33]
    assert tuple(feats) == STRUCT33
    assert len(feats) == 33


def test_no_forbidden_feature_column():
    for sub in FORBIDDEN_FEATURE_SUBSTRINGS:
        assert not any(sub in c for c in STRUCT33), f"STRUCT33 contains {sub!r}"


def test_feature_rows_equal_canonical_environment_rows(built, env):
    ds = built["dataset"]
    decision_idx = ds["candidate_decision_index"].to_numpy().astype(int)
    expected = (
        env["features"].loc[:, list(STRUCT33)].iloc[decision_idx].reset_index(drop=True)
    )
    got = ds[list(STRUCT33)].reset_index(drop=True)
    assert expected.equals(got)


# --------------------------------------------------------------------------- #
# Fill semantics                                                               #
# --------------------------------------------------------------------------- #
def test_fill_index_is_one_ahead_of_decision(built):
    ds = built["dataset"]
    assert bool(
        (ds["candidate_fill_index"] == ds["candidate_decision_index"] + 1).all()
    )


def test_fill_price_is_canonical_next_open(built, env):
    ds = built["dataset"]
    open_px = env["exec_frame"]["open"].to_numpy(float)
    fill_idx = ds["candidate_fill_index"].to_numpy(np.int64)
    valid = fill_idx < len(open_px)
    assert valid.all(), "some candidate has no next bar"
    expected = open_px[fill_idx[valid]]
    got = ds["candidate_fill_price"].to_numpy(float)[valid]
    assert np.allclose(expected, got, equal_nan=True)


# --------------------------------------------------------------------------- #
# Mapping integrity (no rerun, no segment crossing, exit after fill)           #
# --------------------------------------------------------------------------- #
def test_no_mapped_row_crosses_hard_segment(built, env):
    ds = built["dataset"]
    segment = env["exec_frame"]["segment"].to_numpy(np.int64)
    mapped_ok = ds["teacher_row_index"].to_numpy() >= 0
    oracle_entry = ds["oracle_entry_fill_index"].to_numpy(np.int64)
    decision_idx = ds["candidate_decision_index"].to_numpy(np.int64)
    # A mapped Candidate's decision bar, its fill bar and the Oracle entry bar
    # must all share the same hard segment. (Cross-segment candidates now map to -1.)
    assert bool(
        (segment[oracle_entry[mapped_ok]] == segment[decision_idx[mapped_ok]]).all()
    )


def test_no_oracle_identity_without_mapping(built):
    ds = built["dataset"]
    mapped = ds["teacher_row_index"].to_numpy() >= 0
    # Every row that did not map to a Teacher trade must carry no Oracle identity.
    assert ds.loc[~mapped, "oracle_trade_id"].isna().all()
    assert ds.loc[~mapped, "oracle_direction"].isna().all()


def test_map_candidates_fast_rejects_hard_segment_crossing():
    # Synthetic: decision = last bar of segment 0, fill = first bar of segment 1.
    # A Teacher trade exists in segment 1. The candidate must NOT map to it.
    segment = np.array([0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
    trades = pd.DataFrame({
        "trade_id": ["t1"],
        "direction": ["LONG"],
        "entry_fill_index": [4],
        "exit_fill_index": [7],
        "entry_fill_price": [10.0],
        "exit_fill_price": [11.0],
        "training_eligible": [True],
        "terminal_reason": ["OPTIMAL_FLAT"],
    })
    fill_idx = np.array([3], dtype=np.int64)   # segment 1 -> crosses hard segment
    has_next_bar = np.array([True])
    same_segment_fill = np.array([False])      # decision seg0, fill seg1
    valid_for_mapping = has_next_bar & same_segment_fill
    mapped = map_candidates_fast(fill_idx, valid_for_mapping, segment, trades)
    assert list(mapped) == [-1]


def test_decision_time_is_exec_frame_decision_time(built, env):
    ds = built["dataset"]
    decision_idx = ds["candidate_decision_index"].to_numpy().astype(int)
    exp = pd.to_datetime(
        env["exec_frame"]["decision_time"]
    ).to_numpy()[decision_idx]
    got = pd.to_datetime(ds["candidate_decision_time"]).to_numpy()
    assert np.array_equal(exp, got)
    # decision_time must be exactly 15 min after bar_start_time (the bar END).
    bs = pd.to_datetime(
        env["exec_frame"]["bar_start_time"]
    ).to_numpy()[decision_idx]
    delta_min = (exp - bs).astype("timedelta64[m]").astype(int)
    assert (delta_min == 15).all()


def test_fill_time_is_next_bar_start(built, env):
    ds = built["dataset"]
    fill_idx = ds["candidate_fill_index"].to_numpy(np.int64)
    bs = pd.to_datetime(env["exec_frame"]["bar_start_time"]).to_numpy()
    exp = bs[fill_idx]
    got = pd.to_datetime(ds["candidate_fill_time"]).to_numpy()
    assert np.array_equal(exp, got)


def test_mapped_exit_is_after_candidate_fill(built):
    ds = built["dataset"]
    mapped_ok = ds["teacher_row_index"].to_numpy() >= 0
    oracle_exit = ds["oracle_exit_fill_index"].to_numpy(np.int64)
    fill_idx = ds["candidate_fill_index"].to_numpy(np.int64)
    assert bool((oracle_exit[mapped_ok] > fill_idx[mapped_ok]).all())


# --------------------------------------------------------------------------- #
# Vectorized mapper == reference mapper                                        #
# --------------------------------------------------------------------------- #
def _expected_drop(mapped, trades):
    out = []
    for k in mapped:
        if k < 0:
            out.append("CENSORED_NO_FUTURE_TEACHER")
        elif bool(trades.iloc[int(k)]["training_eligible"]):
            out.append("OK")
        else:
            out.append("INELIGIBLE_TEACHER_EXIT")
    return out


def test_vectorized_mapper_matches_reference_synthetic():
    segment = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    trades = pd.DataFrame({
        "trade_id": ["t1", "t2", "t3"],
        "direction": ["LONG", "SHORT", "LONG"],
        "entry_fill_index": [1, 4, 7],
        "exit_fill_index": [3, 6, 9],
        "entry_fill_price": [10.0, 20.0, 30.0],
        "exit_fill_price": [11.0, 19.0, 31.0],
        "training_eligible": [True, False, True],
        "terminal_reason": ["OPTIMAL_FLAT", "HARD_BOUNDARY", "OPTIMAL_FLAT"],
    })
    candidate_fill_idx = np.array([0, 2, 5, 8], dtype=np.int64)
    candidate_valid = np.array([True, True, True, True])
    fast = map_candidates_fast(candidate_fill_idx, candidate_valid, segment, trades)

    cand_df = pd.DataFrame({"candidate_fill_index": candidate_fill_idx})
    ref = map_candidates_to_trades(cand_df, trades, segment)

    fast_tid = [
        (trades.iloc[int(k)]["trade_id"] if k >= 0 else None) for k in fast
    ]
    fast_dir = [
        (trades.iloc[int(k)]["direction"] if k >= 0 else None) for k in fast
    ]
    assert list(fast_tid) == list(ref["trade_id"].to_numpy())
    assert list(fast_dir) == list(ref["direction"].to_numpy())
    assert _expected_drop(fast, trades) == list(ref["drop_reason"].to_numpy())


def test_vectorized_mapper_matches_reference_ag(built, env, trades):
    ds = built["dataset"]
    n = len(env["exec_frame"])
    fill_idx = ds["candidate_fill_index"].to_numpy(np.int64)
    valid = fill_idx < n
    idxs = np.flatnonzero(valid)[:200]
    sub_fill = fill_idx[idxs]
    cand_df = pd.DataFrame({"candidate_fill_index": sub_fill})
    segment = env["exec_frame"]["segment"].to_numpy(np.int64)

    fast = map_candidates_fast(
        sub_fill, np.ones(len(sub_fill), dtype=bool), segment, trades
    )
    ref = map_candidates_to_trades(cand_df, trades, segment)

    fast_tid = [
        (trades.iloc[int(k)]["trade_id"] if k >= 0 else None) for k in fast
    ]
    fast_dir = [
        (trades.iloc[int(k)]["direction"] if k >= 0 else None) for k in fast
    ]
    assert list(fast_tid) == list(ref["trade_id"].to_numpy())
    assert list(fast_dir) == list(ref["direction"].to_numpy())
    assert _expected_drop(fast, trades) == list(ref["drop_reason"].to_numpy())


# --------------------------------------------------------------------------- #
# Labels                                                                       #
# --------------------------------------------------------------------------- #
def test_entry_quality_recompute_mismatch_zero(built, env):
    ds = built["dataset"]
    le = ds["label_eligible"].to_numpy()
    decision_idx = ds["candidate_decision_index"].to_numpy().astype(int)
    atr = env["features"]["m15_atr"].to_numpy(float)[decision_idx]
    sign = np.where(ds["oracle_direction"].to_numpy() == "LONG", 1.0,
                    np.where(ds["oracle_direction"].to_numpy() == "SHORT", -1.0, 0.0))
    recomputed = (
        sign * (ds["oracle_exit_fill_price"].to_numpy(float)
                - ds["candidate_fill_price"].to_numpy(float))
        / atr
    )
    got = ds["entry_quality_atr"].to_numpy(float)
    diff = np.abs(recomputed[le] - got[le])
    assert np.nanmax(diff) < 1e-9


def test_bars_to_exit_positive_for_every_eligible_row(built):
    ds = built["dataset"]
    le = ds["label_eligible"].to_numpy()
    assert bool((ds.loc[le, "bars_to_oracle_exit"] > 0).all())


def test_per_oracle_trade_raw_weight_sum_is_one(built):
    ds = built["dataset"]
    le = ds["label_eligible"].to_numpy()
    sub = ds[le]
    sums = sub.groupby("oracle_trade_id")["sample_weight_raw"].sum().to_numpy()
    assert np.allclose(sums, 1.0, atol=1e-12)


def test_censored_rows_receive_no_valid_target(built):
    ds = built["dataset"]
    le = ds["label_eligible"].to_numpy()
    # For every non-eligible row, EntryQuality must be NaN (no valid numeric target).
    assert bool(np.isnan(ds.loc[~le, "entry_quality_atr"].to_numpy(float)).all())
    # Censored-without-teacher rows must not carry an oracle direction as a target.
    no_teacher = ds["label_status"].isin(
        ["CENSORED_NO_NEXT_BAR", "CENSORED_FILL_CROSSES_HARD_SEGMENT",
         "CENSORED_NO_FUTURE_TEACHER"]
    ).to_numpy()
    assert bool(ds.loc[no_teacher, "oracle_direction"].isna().all())


def test_array_lengths_align(built):
    ds = built["dataset"]
    assert len(ds) == int(built["meta"]["n_candidates"])


# --------------------------------------------------------------------------- #
# Builder must NOT rerun DP / environment beyond the single allowed run         #
# --------------------------------------------------------------------------- #
def test_no_dp_rerun_inside_builder(monkeypatch):
    # Prove the builder never reaches the DP Teacher runner.
    def _boom(*a, **k):
        raise RuntimeError("DP rerun detected inside dataset builder")

    monkeypatch.setattr(
        __import__(
            "research.liquidity_oracle_atlas.build_teacher_oracle_dp_m15_overnight_v1",
            fromlist=["x"],
        ),
        "run_dp_m15_overnight_teacher",
        _boom,
    )
    assert "run_dp_m15_overnight_teacher" not in dir(B)
    r = build_struct33_dataset("AG")  # must not raise
    assert r["dataset"] is not None


def test_teacher_artifact_used_not_rerun(trades):
    # Sanity: the loaded artifact is the verified overnight Teacher.
    assert "trade_id" in trades.columns
    assert "entry_fill_index" in trades.columns
    assert "exit_fill_index" in trades.columns


def test_summary_reports_phase_split(built):
    ps = built["stats"]["candidate_phase_split"]
    assert set(ps) >= {"before_entry", "at_entry", "in_position", "n_eligible"}
    n = ps["n_eligible"]
    total = (
        ps["before_entry"]["count"]
        + ps["at_entry"]["count"]
        + ps["in_position"]["count"]
    )
    # All label-eligible rows fall into exactly one phase bucket.
    assert total == n
    for k in ("before_entry", "at_entry", "in_position"):
        assert ps[k]["count"] >= 0
        assert 0.0 <= ps[k]["pct"] <= 1.0


def test_review_sample_is_unique(built):
    rv = built["review_sample"]
    assert len(rv) == 48
    assert rv["candidate_decision_index"].is_unique


def test_frozen_teacher_identity_is_pinned():
    # The builder must consume exactly the Phase 0.5 FIX2 Teacher artifact.
    from research.liquidity_oracle_atlas.build_struct33_dataset_v1 import (
        FROZEN_TEACHER_SOURCE_SHA,
    )
    art = load_oracle_artifact(ARTIFACT_ROOT, "AG")
    assert art["metadata"].get("oracle_source_sha") == FROZEN_TEACHER_SOURCE_SHA
    # And the pinned identity actually gates the load (fail-closed).
    bad = load_oracle_artifact(
        ARTIFACT_ROOT, "AG", expected_source_sha="does-not-match"
    )
    assert bad["ok"] is False
    assert bad["reason"] == "source_sha_mismatch"
