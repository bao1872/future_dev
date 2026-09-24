"""§47 regression tests — walk-forward engine + leakage guard.

Covers plan requirements:
  (1)  old TEST labels cannot be opened by the V2 runner
  (2)  current Formal evidence cannot enter model selection
  (3)  walk-forward training rows all precede outer validation
  (4)  training label_available_time < outer validation start
  (5)  two-side pair purity preserved
  (18) OUTER fold never used for early stopping
  (19) ES block never used as outer score
  (21) each OOF row predicted exactly once
  (22) no OOF prediction for the warm-up region

All fixtures are synthetic; no heavy artifacts or LightGBM are involved.
"""

import os

import numpy as np
import pandas as pd
import pytest

from research.liquidity_oracle_atlas import (
    run_decomposed_v2_research as R,
    walkforward_development_v1 as W,
)


# --------------------------------------------------------------------------- #
# Synthetic labels frame                                                       #
# --------------------------------------------------------------------------- #
def _mk_frame(
    n_days: int = 40,
    rows_per_day: int = 2,
    symbols=("SYN",),
    horizons=("td5",),
    start: str = "2025-01-01",
    label_lag_days: int = 1,
    duplicate_availability: bool = True,
) -> pd.DataFrame:
    """Build a well-formed two-side labels frame.

    Every (symbol, decision_bar, horizon) epoch has exactly one LONG row and
    one SHORT row whose sample_weight sums to 1, matching the frozen V1 layout.
    """
    days = pd.date_range(start, periods=n_days, freq="D")
    recs = []
    bar = 0
    for d_i, day in enumerate(days):
        for _ in range(rows_per_day):
            for sym in symbols:
                for h in horizons:
                    decision_time = day
                    available = pd.Timestamp(
                        days[min(d_i + label_lag_days, n_days - 1)])
                    for side in ("LONG", "SHORT"):
                        recs.append({
                            "symbol": sym,
                            "decision_bar": bar,
                            "side": side,
                            "horizon": h,
                            "decision_time": decision_time,
                            "label_available_time": available,
                            "sample_weight": 0.5,
                            "episode_return_atr": 0.1 if side == "LONG" else -0.1,
                        })
            bar += 1
    df = pd.DataFrame(recs)
    if not duplicate_availability:
        df = df.copy()
    return df


def _plan(frame, **kw):
    return W.build_fold_plan(frame, **kw)


# --------------------------------------------------------------------------- #
# (1) / (2) leakage guard                                                      #
# --------------------------------------------------------------------------- #
def test_old_test_labels_cannot_be_opened():
    R.reset_counters()
    with pytest.raises(R.StopV2Leakage) as exc:
        R.read_parquet(
            os.path.join(R.V1_ARTIFACT_DIR, "labels_test_v1.parquet"))
    assert "STOP_V2_OLD_TEST_LEAKAGE" in str(exc.value)
    assert R.COUNTERS["old_test_label_reads"] == 1
    assert R.COUNTERS["old_test_policy_reads"] == 0


def test_formal_test_policy_evidence_cannot_enter_selection():
    R.reset_counters()
    for tok in ("decomposed_value_test_diagnostics",
                "decomposed_value_daily_returns",
                "decomposed_value_trade_ledger",
                "decomposed_value_policy_summary",
                "decomposed_value_per_symbol"):
        path = os.path.join(R.EVIDENCE_DIR, tok + ".csv")
        with pytest.raises(R.StopV2Leakage):
            R.read_frame(path)
    assert R.COUNTERS["old_test_policy_reads"] == 5
    assert R.COUNTERS["old_test_label_reads"] == 0


def test_allowed_train_inputs_pass_the_guard():
    R.reset_counters()
    # These are the only frozen V1 inputs V2 may consume.
    R.read_train_labels(columns=["symbol", "decision_bar"])
    R.read_state(columns=["symbol", "bar_index"])
    assert R.COUNTERS["old_test_label_reads"] == 0
    assert R.COUNTERS["old_test_policy_reads"] == 0


def test_val_locked_until_selection_is_frozen(tmp_path, monkeypatch):
    missing = str(tmp_path / "does_not_exist.json")
    monkeypatch.setattr(R, "SELECTION_JSON", missing)
    with pytest.raises(R.StopV2ValUnlocked) as exc:
        R.read_val_labels()
    assert "STOP_V2_VAL_UNLOCKED" in str(exc.value)


# --------------------------------------------------------------------------- #
# (3) / (4) causal ordering                                                    #
# --------------------------------------------------------------------------- #
def test_training_rows_precede_outer_validation():
    frame = _mk_frame()
    plan = _plan(frame)
    for k in range(plan.n_outer):
        s = W.fold_split(frame, plan, k)
        start = pd.Timestamp(plan.outer[k][0])
        train = s.fit_core | s.es
        assert W.decision_day(frame)[train].max() < start
        # And every row decided on/after the start is out of the training set.
        assert not bool(
            ((W.decision_day(frame) >= start) & train).any())


def test_training_label_available_time_precedes_outer_start():
    frame = _mk_frame(label_lag_days=3)
    plan = _plan(frame)
    for k in range(plan.n_outer):
        s = W.fold_split(frame, plan, k)
        start = pd.Timestamp(plan.outer[k][0])
        av = W.label_available_day(frame)
        assert (av[s.fit_core] < start).all()
        assert (av[s.es] < start).all()


def test_unresolved_labels_are_excluded_from_training():
    frame = _mk_frame(label_lag_days=2)
    plan = _plan(frame)
    s = W.fold_split(frame, plan, 0)
    start = pd.Timestamp(plan.outer[0][0])
    av = W.label_available_day(frame)
    # Any row whose label only becomes known afterwards must be absent.
    late = (av >= start).to_numpy()
    assert not bool((late & (s.fit_core | s.es)).any())


# --------------------------------------------------------------------------- #
# (18) / (19) ES vs OUTER separation                                           #
# --------------------------------------------------------------------------- #
def test_outer_fold_never_used_for_early_stopping():
    frame = _mk_frame()
    plan = _plan(frame)
    for k in range(plan.n_outer):
        s = W.fold_split(frame, plan, k)
        assert not bool((s.outer & s.es).any())
        assert not bool((s.outer & s.fit_core).any())
        # Mutually exclusive and exhaustive over the training side.
        assert not bool((s.es & s.fit_core).any())


def test_es_block_is_tail_of_training_days_not_random():
    frame = _mk_frame(n_days=60, rows_per_day=1)
    plan = _plan(frame)
    s = W.fold_split(frame, plan, 2)
    d = W.decision_day(frame)
    train_days = pd.DatetimeIndex(np.sort(d[s.fit_core | s.es].unique()))
    es_days = pd.DatetimeIndex(np.sort(d[s.es].unique()))
    n_es = int(np.floor(len(train_days) * plan.es_frac))
    # ES must be exactly the LAST n_es training days (contiguous tail).
    assert list(es_days) == list(train_days[-n_es:])
    # And ES is chronologically after everything in FIT_CORE.
    assert d[s.fit_core].max() <= d[s.es].min()


def test_val_block_never_used_as_outer_score():
    frame = _mk_frame()
    plan = _plan(frame)
    for k in range(plan.n_outer):
        s = W.fold_split(frame, plan, k)
        d = W.decision_day(frame)
        outer_days = set(d[s.outer].unique())
        assert not (set(s.es_days) & {x.date().isoformat() for x in outer_days})


# --------------------------------------------------------------------------- #
# (21) / (22) OOF coverage                                                     #
# --------------------------------------------------------------------------- #
def test_each_oof_row_predicted_exactly_once():
    frame = _mk_frame(n_days=60, rows_per_day=2)
    plan = _plan(frame)
    splits = list(W.iter_fold_splits(frame, plan))
    cov = W.oof_row_count_check(splits, frame, plan)
    assert cov["max_times_predicted"] == 1
    assert cov["n_nonwarmup_rows_never_predicted"] == 0
    assert cov["n_rows_predicted_once"] == (
        cov["n_rows"] - cov["n_warmup_rows"])


def test_no_oof_prediction_for_warmup_region():
    frame = _mk_frame(n_days=60, rows_per_day=2)
    plan = _plan(frame)
    splits = list(W.iter_fold_splits(frame, plan))
    cov = W.oof_row_count_check(splits, frame, plan)
    assert cov["n_warmup_rows_predicted"] == 0
    # The first outer block must start strictly after the warm-up day.
    assert pd.Timestamp(plan.outer[0][0]) > pd.Timestamp(plan.warmup_end)


def test_outer_blocks_are_contiguous_and_non_overlapping():
    frame = _mk_frame(n_days=70)
    plan = _plan(frame)
    assert W._assert_outer_partitions_once(plan.outer)
    prev_end = None
    for s, e in plan.outer:
        if prev_end is not None:
            assert (pd.Timestamp(s) - pd.Timestamp(prev_end)).days == 1
        prev_end = e
    assert plan.n_outer == W.N_OUTER


# --------------------------------------------------------------------------- #
# (5) pair purity + weight gate                                                #
# --------------------------------------------------------------------------- #
def test_pair_purity_drops_half_available_epoch():
    frame = _mk_frame(n_days=40, rows_per_day=1, label_lag_days=1)
    # Make one epoch's SHORT row resolve late (after the fold-0 cutoff).
    plan = _plan(frame)
    cutoff = pd.Timestamp(plan.outer[0][0])
    # Pick an epoch comfortably inside the warm-up so BOTH sides would normally
    # be available before the cutoff. Choosing the last pre-cutoff bar would be
    # wrong: its label resolves exactly at the cutoff, so it is merely censored
    # rather than half-available.
    elig = frame.loc[
        (W.decision_day(frame) < cutoff)
        & (W.label_available_day(frame) < cutoff)
        & (frame["side"] == "LONG"), "decision_bar"]
    victim_bar = int(elig.iloc[len(elig) // 2])
    mask = (frame["decision_bar"] == victim_bar) & (frame["side"] == "SHORT")
    frame.loc[mask, "label_available_time"] = cutoff + pd.Timedelta(days=30)

    avail = W.available_training_mask(frame, cutoff)
    purged, stats = W.enforce_pair_purity(frame, avail)

    rows = frame[purged]
    counts = rows.groupby(["symbol", "decision_bar", "horizon"]).size()
    # No retained epoch may have a single side when two existed originally.
    assert set(counts.unique()) <= {2}
    assert victim_bar not in set(rows["decision_bar"])
    assert stats["dropped_pair_break"] == 1


def test_no_half_epoch_survives_on_real_shape():
    frame = _mk_frame(n_days=50, rows_per_day=2)
    plan = _plan(frame)
    for k in range(plan.n_outer):
        s = W.fold_split(frame, plan, k)
        rows = frame[s.fit_core | s.es]
        counts = rows.groupby(["symbol", "decision_bar", "horizon"]).size()
        assert set(counts.unique()) <= {2}
        assert s.purity_stats["dropped_pair_break"] == 0


def test_epoch_weight_unit_gate_fires_on_corrupt_weights():
    frame = _mk_frame(n_days=40, rows_per_day=1)
    frame.loc[frame["side"] == "SHORT", "sample_weight"] = 0.3  # sums to 0.8
    plan = _plan(frame)
    with pytest.raises(W.StopV2EpochWeightNotUnit) as exc:
        W.fold_split(frame, plan, 0)
    assert "STOP_V2_EPOCH_WEIGHT_NOT_UNIT" in str(exc.value)


def test_epoch_weight_unit_gate_accepts_exact_one():
    frame = _mk_frame(n_days=40, rows_per_day=1)
    plan = _plan(frame)
    s = W.fold_split(frame, plan, 0)
    rows = frame[s.fit_core | s.es]
    sums = rows.groupby(["symbol", "decision_bar", "horizon"])[
        "sample_weight"].sum()
    assert np.allclose(sums.to_numpy(), 1.0, atol=W.WEIGHT_TOL)


# --------------------------------------------------------------------------- #
# Plan shape guarantees                                                        #
# --------------------------------------------------------------------------- #
def test_plan_defaults_match_frozen_fractions():
    frame = _mk_frame(n_days=100)
    plan = _plan(frame)
    assert plan.n_outer == W.N_OUTER == 5
    assert plan.es_frac == W.ES_FRAC == 0.15
    n_warm = int(np.floor(len(plan.days) * W.WARMUP_FRAC))
    assert plan.warmup_end == plan.days[n_warm - 1]
    # Outer blocks cover every post-warm-up day.
    covered = sum(
        (pd.Timestamp(e) - pd.Timestamp(s)).days + 1 for s, e in plan.outer)
    assert covered == len(plan.days) - n_warm


def test_plan_is_immutable():
    frame = _mk_frame(n_days=40)
    plan = _plan(frame)
    with pytest.raises(Exception):
        plan.outer = ()
