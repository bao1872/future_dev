"""Tests for FUTURE-R4-M15-DIRECTION-NULL-BASELINE-AUDIT-V1.

Proves the frozen DTP9 direction result cannot be explained by label/opportunity/
payoff imbalance or a trivial LONG-biased predictor:

  - frozen split + M0 deterministic-reproduction guard (frozen params, no tuning)
  - TEST is 50/50 at the Oracle-opportunity level (319 LONG / 319 SHORT)
  - always_long / fair_coin / train_prior_coin economic returns are ~0
  - M0 clearly beats every null (always_long, always_short, fair_coin, prior_coin)
  - M0 has a measurable LONG prediction bias (recall LONG >> recall SHORT)
  - class-balanced counterfactual is a no-op (already 319/319)
  - bootstrap chunked == unchunked reference
  - evidence files written with the explicit metric name
"""

import inspect
import json
import os

import numpy as np
import pandas as pd
import pytest

from research.liquidity_oracle_atlas.train_direction_model_15sym_v1 import (
    bootstrap_reference,
    bootstrap_trade_returns_chunked,
)

import research.liquidity_oracle_atlas.direction_null_baseline_v1 as M


def test_no_environment_or_teacher_rerun():
    src = inspect.getsource(M)
    for tok in ("run_environment_m15", "derive_m15_candidate_gate",
                "build_struct33_dataset", "load_oracle_artifact", "run_dp_m15"):
        assert tok not in src, f"forbidden token present in source: {tok}"


def test_metric_naming_has_no_pnl_drift():
    assert M.METRIC_NAME == "TeacherFixedExitDirectionReturnATR"
    assert "strategy PnL" in M.METRIC_NOT_LABELS
    d = " ".join((M.__doc__ or "").lower().split())
    assert "not strategy pnl" in d


@pytest.fixture(scope="module")
def run():
    return M.run_null_baseline_audit(save=False, verbose=False)


def test_frozen_split_reproduced(run):
    s = run["summary"]
    rep = s["frozen_split_reproduction"]
    assert rep["test_rows"] == 13773
    assert rep["test_trades"] == 638
    assert abs(rep["recomputed_m0_return_atr"] - 0.6014965284150177) < 1e-6
    assert rep["match"] is True


def test_hard_fact_check_319_319_638(run):
    g = run["summary"]["G_hard_fact_check"]
    assert g["unique_teacher_opportunities"] == 638
    assert g["teacher_long_opps"] == 319
    assert g["teacher_short_opps"] == 319


def test_label_balance_is_50_50(run):
    s = run["summary"]
    test = s["A_label_balance"]["TEST"]
    assert test["long_opps"] == 319 and test["short_opps"] == 319
    assert abs(test["long_rows_pct"] - 0.5) < 0.05
    # per-phase also roughly balanced at opportunity level
    for ph in ("BEFORE_ENTRY", "AT_ENTRY", "IN_POSITION"):
        b = s["A_label_balance"]["TEST_by_phase"][ph]
        assert abs(b["long_opps_pct"] - 0.5) < 0.15


def test_economic_payoff_nulls_are_zero(run):
    nulls = run["summary"]["D_null_predictors"]["POOLED"]
    for k in ("always_long", "always_short", "train_majority",
              "fair_coin", "train_prior_coin"):
        r = nulls[k]["mean_return"]
        assert abs(r) < 0.05, f"{k} return {r} not ~0"
        # CI crosses (or sits at) 0
        assert nulls[k]["ci_low"] < 0 < nulls[k]["ci_high"]


def test_m0_long_prediction_bias(run):
    bias = run["summary"]["C_m0_prediction_bias"]["POOLED"]
    assert bias["predicted_long_pct"] > 0.55          # model is LONG-biased
    assert bias["long_recall"] > bias["short_recall"] + 0.1
    assert bias["long_recall"] > 0.70
    assert bias["short_recall"] < 0.55


def test_m0_beats_every_null(run):
    e = run["summary"]["E_model_vs_null"]["POOLED"]
    for k in ("always_long", "always_short", "fair_coin", "train_prior_coin"):
        d = e[k]
        assert d["mean_delta"] > 0.3, f"{k} delta {d['mean_delta']} not clearly positive"
        assert d["ci_low"] > 0, f"{k} CI low {d['ci_low']} not > 0"


def test_class_balanced_is_noop(run):
    f = run["summary"]["F_class_balanced_counterfactual"]
    assert f["n_long_opps"] == 319 and f["n_short_opps"] == 319
    assert abs(f["M0"]["unweighted_return"] - f["M0"]["class_balanced_return"]) < 1e-12
    assert abs(f["M0"]["unweighted_accuracy"] - f["M0"]["class_balanced_accuracy"]) < 1e-12
    assert abs(f["ALWAYS_LONG"]["class_balanced_return"] - 0.017734) < 2e-3
    assert abs(f["ALWAYS_SHORT"]["class_balanced_return"] + 0.017734) < 2e-3
    assert all(f["no_op_assertion"].values())


def test_bootstrap_chunked_equals_reference():
    rng = np.random.default_rng(0)
    tr = rng.normal(0.5, 2.0, size=638)
    ref = bootstrap_reference(tr, B=3000)
    for chunk in (1, 7, 250, 3000):
        chk = bootstrap_trade_returns_chunked(tr, B=3000, chunk=chunk)
        assert np.allclose(ref, chk), f"chunk={chunk} mismatch vs reference"


def test_confusion_matrix_precision_recall_synthetic():
    # Synthetic Oracle-opportunity confusion matrix (LONG = positive class):
    #   actual  y = [L, L, L, S, S, S]
    #   pred    d = [L, L, S, L, L, S]
    #   => TP=2, FN=1, FP=2, TN=1
    y = np.array([1, 1, 1, 0, 0, 0])
    d = np.array([1, 1, 0, 1, 1, 0])
    c = M._confusion(d, y)
    assert c["n_long_opps"] == 3 and c["n_short_opps"] == 3
    assert abs(c["long_recall"] - 2 / 3) < 1e-12
    assert abs(c["short_recall"] - 1 / 3) < 1e-12
    assert abs(c["balanced_accuracy"] - 0.5) < 1e-12
    assert abs(c["precision_long"] - 0.5) < 1e-12
    # SHORT as positive class: precision_short = TN / (TN + FN) = 1 / (1 + 1) = 0.5
    assert abs(c["precision_short"] - 0.5) < 1e-12
    # guard against the old buggy formula TN / (TN + FP) = 1/3
    assert abs(c["precision_short"] - 1 / 3) > 1e-9


def test_evidence_files_and_metric_column(run):
    paths = run["paths"]
    for p in (paths["summary"], paths["per_symbol_csv"]):
        assert os.path.exists(p)
    df = pd.read_csv(paths["per_symbol_csv"])
    assert (df["metric"] == M.METRIC_NAME).all()
    assert len(df) == 15
    for col in ("symbol", "n_test_trades", "m0_predicted_long_pct", "m0_long_recall",
                "m0_short_recall", "m0_return_atr", "always_long_return_atr",
                "always_short_return_atr", "fair_coin_mean_return",
                "train_prior_coin_mean_return"):
        assert col in df.columns
    # precision_short column must now be the corrected field (not short_recall)
    assert "m0_precision_short" in df.columns
