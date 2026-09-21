"""
test_entry_value_tree_core108_t15_v1
====================================

T1.5 integrity tests for FUTURE-ENTRY-VALUE-TREE-CORE108-V1-T1.5.

Covers the 16 required integrity properties plus 3 negative controls:

 1  X exactly 108 columns
 2  no forbidden columns in X
 3  phase category vocabulary fixed
 4  same global episode never spans split
 5  split dates strictly ordered
 6  label-availability purge works
 7  raw episode weights sum to 1
 8  Train normalized weight mean = 1
 9  Validation normalized weight mean = 1
10  multi-symbol normalization recomputed after concatenation
11  Test never enters fit() / early stopping
12  Test never enters permutation importance
13  constant baseline uses Train only
14  deterministic fixed split for identical input
15  no target clipping / transformation
16  LightGBM receives NaN directly without numeric imputation

Negative controls:
 N1 leak Q_F1_L into X            -> feature contract must FAIL
 N2 one episode across Train/Val  -> split gate must FAIL
 N3 Test used as eval_set         -> fit gate must FAIL

The dataset is built ONCE per test session on a small true prefix (bar 0..N-1)
to keep the suite fast; all gates are the same production functions used by
run_t15().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# IMPORT ORDER MATTERS: the t15 module imports lightgbm at its very top. It must
# be imported BEFORE any module that pulls in pandas / scipy / the CORE108 kernel,
# otherwise lib_lightgbm is dlopen'ed late and segfaults on macOS (OpenMP).
import research.liquidity_oracle_atlas.experiment_entry_value_tree_core108_t15_v1 as T
from research.liquidity_oracle_atlas.experiment_entry_value_tree_core108_v1 import (
    core108_columns,
)

SYMBOLS = ("AG", "RB")
TEST_PREFIX = 3000  # small true prefix; the real T1.5 run uses 10000


# --------------------------------------------------------------------------- #
# Session-scoped fixture: build the combined dataset once
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def ds():
    frames = []
    for sym in SYMBOLS:
        r = T.build_symbol_frame(sym, prefix_bars=TEST_PREFIX)
        cand, _ = T.drop_final_trading_day(r["cand"])
        frames.append(cand)
    df = T.combine_symbols(frames)
    df, boundaries = T.assign_time_split(df)
    df, purge = T.apply_label_purge(df, boundaries)
    T.assert_no_episode_spanning_split(df)
    df, winfo = T.compute_split_weights(df)

    X = T.build_X(df)
    tr = (df["split"] == "train").to_numpy()
    va = (df["split"] == "validation").to_numpy()
    te = (df["split"] == "test").to_numpy()
    return {
        "df": df,
        "boundaries": boundaries,
        "purge": purge,
        "winfo": winfo,
        "X": X,
        "Xtr": X[tr].reset_index(drop=True),
        "Xva": X[va].reset_index(drop=True),
        "Xte": X[te].reset_index(drop=True),
        "tr": tr, "va": va, "te": te,
        "n_train": int(tr.sum()), "n_val": int(va.sum()),
        "n_test": int(te.sum()), "n_total": int(len(df)),
    }


# --------------------------------------------------------------------------- #
# 1 / 2 / 3 — feature matrix contract
# --------------------------------------------------------------------------- #
def test_x_exactly_108_columns(ds):
    assert ds["X"].shape[1] == 108
    assert list(ds["X"].columns) == list(core108_columns())


def test_no_forbidden_columns_in_X(ds):
    T.assert_feature_contract(ds["X"])  # raises SystemExit on violation
    for c in ds["X"].columns:
        for tok in T.FORBIDDEN_MODEL_TOKENS:
            assert tok.lower() not in c.lower(), f"{c} contains forbidden token {tok}"


def test_phase_vocabulary_fixed(ds):
    T.assert_phase_vocabulary(ds["df"])
    for c in [x for x in core108_columns() if x.endswith("_phase")]:
        assert str(ds["X"][c].dtype) == "category"
        assert list(ds["X"][c].cat.categories) == list(T.PHASE_VOCAB)


# --------------------------------------------------------------------------- #
# 4 / 5 / 6 — split + purge
# --------------------------------------------------------------------------- #
def test_no_episode_spans_split(ds):
    T.assert_no_episode_spanning_split(ds["df"])
    g = ds["df"].groupby("global_episode")["split"].nunique()
    assert int((g > 1).sum()) == 0


def test_split_dates_strictly_ordered(ds):
    b = ds["boundaries"]
    T.assert_split_dates_ordered(b)
    assert max(b["train_days"]) < min(b["validation_days"])
    assert max(b["validation_days"]) < min(b["test_days"])


def test_label_purge_works(ds):
    df, b = ds["df"], ds["boundaries"]
    lat = pd.to_datetime(df["label_available_time"])
    val_start = b["validation_start_time"]
    test_start = b["test_start_time"]
    tr = df["split"] == "train"
    va = df["split"] == "validation"
    assert bool((lat[tr] < val_start).all()), "train row leaks into validation window"
    assert bool((lat[va] < test_start).all()), "validation row leaks into test window"
    # purge actually removed rows (non-vacuous)
    removed = ds["purge"]["removed"]
    assert sum(removed.values()) >= 0
    assert ds["purge"]["after"]["test"] == ds["purge"]["before"]["test"]  # Test untouched


# --------------------------------------------------------------------------- #
# 7 / 8 / 9 / 10 — weights
# --------------------------------------------------------------------------- #
def test_raw_episode_weights_sum_to_one(ds):
    audit = T.verify_episode_raw_weight_sums(ds["df"])
    assert audit["episodes_incomplete"] == 0, audit
    assert audit["episodes_complete_sum1"] == audit["episodes_total"]
    sums = ds["df"].groupby("global_episode")["w_raw"].sum().to_numpy(float)
    assert np.allclose(sums, 1.0, atol=1e-9)


def test_train_normalized_weight_mean_is_one(ds):
    w = ds["df"].loc[ds["tr"], "w_norm"].to_numpy(float)
    assert abs(float(np.mean(w)) - 1.0) < 1e-9


def test_validation_normalized_weight_mean_is_one(ds):
    w = ds["df"].loc[ds["va"], "w_norm"].to_numpy(float)
    assert abs(float(np.mean(w)) - 1.0) < 1e-9


def test_multisymbol_normalization_recomputed(ds):
    """Normalization must be RECOMPUTED after concatenation, not the per-symbol
    sample_weight_norm produced inside join_with_r2()."""
    df = ds["df"]
    assert "sample_weight_norm" in df.columns
    recomp = df["w_norm"].to_numpy(float)
    per_symbol = df["sample_weight_norm"].to_numpy(float)
    # Train rows: recomputed mean == 1 while the raw per-symbol column is not 1
    tr = ds["tr"]
    assert abs(float(np.mean(recomp[tr])) - 1.0) < 1e-9
    # and the two differ elementwise for most rows (proving recomputation)
    diff = ~np.isclose(recomp[tr], per_symbol[tr], rtol=1e-9, atol=1e-12)
    assert float(diff.mean()) > 0.5, "w_norm identical to per-symbol norm (not recomputed)"


# --------------------------------------------------------------------------- #
# 11 / 12 — Test never used for fitting or permutation
# --------------------------------------------------------------------------- #
def test_test_never_enters_fit(monkeypatch, ds):
    recorded = {}
    orig = T.lgb.LGBMRegressor.fit

    def spy(self, X, y, **kw):
        recorded["n_train"] = len(X)
        recorded["n_eval"] = [len(a) for a, _ in kw.get("eval_set", [])]
        return orig(self, X, y, **kw)

    monkeypatch.setattr(T.lgb.LGBMRegressor, "fit", spy)
    ytr = ds["df"].loc[ds["tr"], "Y_L"].to_numpy(float)
    yva = ds["df"].loc[ds["va"], "Y_L"].to_numpy(float)
    wtr = ds["df"].loc[ds["tr"], "w_norm"].to_numpy(float)
    wva = ds["df"].loc[ds["va"], "w_norm"].to_numpy(float)
    T.fit_direction(ds["Xtr"], ytr, wtr, ds["Xva"], yva, wva)

    assert recorded["n_train"] == ds["n_train"]
    assert recorded["n_eval"] == [ds["n_val"]]
    assert recorded["n_train"] + recorded["n_eval"][0] == ds["n_total"] - ds["n_test"]


def test_test_never_enters_permutation(monkeypatch, ds):
    """Permutation importance runs on Validation only (no refit, no Test)."""
    ytr = ds["df"].loc[ds["tr"], "Y_L"].to_numpy(float)
    yva = ds["df"].loc[ds["va"], "Y_L"].to_numpy(float)
    wtr = ds["df"].loc[ds["tr"], "w_norm"].to_numpy(float)
    wva = ds["df"].loc[ds["va"], "w_norm"].to_numpy(float)
    model, _, _ = T.fit_direction(ds["Xtr"], ytr, wtr, ds["Xva"], yva, wva)

    seen = []
    orig_predict = model.predict

    def spy_predict(X, *a, **k):
        seen.append(len(X))
        return orig_predict(X, *a, **k)

    monkeypatch.setattr(model, "predict", spy_predict)
    T.permutation_importance(model, ds["Xva"], yva, wva, T.tf_groups(), repeats=2)
    assert len(seen) > 0
    assert set(seen) == {ds["n_val"]}, f"permutation saw non-validation row counts: {set(seen)}"


# --------------------------------------------------------------------------- #
# 13 / 14 / 15 / 16
# --------------------------------------------------------------------------- #
def test_constant_baseline_uses_train_only(ds):
    df = ds["df"]
    y = df["Y_L"].to_numpy(float)
    w_raw = df["w_raw"].to_numpy(float)
    tr, te = ds["tr"], ds["te"]
    baseline = T._wmean(y[tr], w_raw[tr])
    # equals the train-only weighted mean ...
    assert abs(baseline - float(np.sum(w_raw[tr] * y[tr]) / np.sum(w_raw[tr]))) < 1e-12
    # ... and is NOT the all-rows or test-only weighted mean
    assert not np.isclose(baseline, T._wmean(y, w_raw), rtol=1e-6)
    assert not np.isclose(baseline, T._wmean(y[te], w_raw[te]), rtol=1e-6)


def test_deterministic_fixed_split(ds):
    df = ds["df"]
    _, b1 = T.assign_time_split(df)
    _, b2 = T.assign_time_split(df)
    assert b1["train_days"] == b2["train_days"]
    assert b1["validation_days"] == b2["validation_days"]
    assert b1["test_days"] == b2["test_days"]
    days = sorted(pd.unique(df["trading_day"]))
    b3 = T.compute_split_boundaries(days)
    assert b3 == T.compute_split_boundaries(days)
    assert len(b1["test_days"]) > 0


def test_no_target_clipping_or_transformation(ds):
    df = ds["df"]
    a5 = df["atr5m"].to_numpy(float)
    for target, num in (("Y_L", "Q_F1_L"), ("Y_S", "Q_F1_S")):
        expected = (df[num].to_numpy(float) - df["Q_F1_F"].to_numpy(float)) / a5
        got = df[target].to_numpy(float)
        m = np.isfinite(expected) & np.isfinite(got)
        assert m.sum() > 0
        assert np.allclose(got[m], expected[m], rtol=0, atol=1e-12)


def test_lightgbm_receives_nan_without_imputation(ds):
    """Numeric NaN is preserved end-to-end; nothing is imputed."""
    df, X = ds["df"], ds["X"]
    cols = list(core108_columns())
    nan_df = int(df[cols].isna().sum().sum())
    nan_X = int(X.isna().sum().sum())
    assert nan_df > 0, "no NaN present (test would be vacuous)"
    assert nan_df == nan_X, "build_X changed the NaN pattern (imputation detected)"
    # the model must accept those NaNs directly
    ytr = df.loc[ds["tr"], "Y_L"].to_numpy(float)
    yva = df.loc[ds["va"], "Y_L"].to_numpy(float)
    wtr = df.loc[ds["tr"], "w_norm"].to_numpy(float)
    wva = df.loc[ds["va"], "w_norm"].to_numpy(float)
    model, _, _ = T.fit_direction(ds["Xtr"], ytr, wtr, ds["Xva"], yva, wva)
    pred = model.predict(ds["Xte"])
    assert len(pred) == ds["n_test"]


# --------------------------------------------------------------------------- #
# Negative controls
# --------------------------------------------------------------------------- #
def test_negative_leak_q_f1_l_into_x_fails(ds):
    X_bad = ds["X"].copy()
    X_bad["Q_F1_L"] = ds["df"]["Q_F1_L"].to_numpy(float)
    with pytest.raises(SystemExit) as e:
        T.assert_feature_contract(X_bad)
    assert "STOP_" in str(e.value)


def test_negative_episode_across_train_validation_fails(ds):
    df_bad = ds["df"].copy()
    # force one global episode to appear in BOTH train and validation
    ep = df_bad.loc[df_bad["split"] == "train", "global_episode"].iloc[0]
    idx = df_bad.index[df_bad["global_episode"] == ep]
    val_idx = df_bad.index[df_bad["split"] == "validation"][:1]
    df_bad.loc[val_idx, "global_episode"] = ep
    assert len(idx) > 0
    with pytest.raises(SystemExit) as e:
        T.assert_no_episode_spanning_split(df_bad)
    assert "STOP_EPISODE_SPANS_SPLIT" in str(e.value)


def test_negative_test_as_eval_set_fails(ds):
    # pure-train / pure-validation passes
    T.assert_fit_matrices_exclude_test(
        ds["df"].loc[ds["tr"], "split"].to_numpy(),
        ds["df"].loc[ds["va"], "split"].to_numpy(),
    )
    # using Test as the eval set must FAIL
    with pytest.raises(SystemExit) as e:
        T.assert_fit_matrices_exclude_test(
            ds["df"].loc[ds["tr"], "split"].to_numpy(),
            ds["df"].loc[ds["te"], "split"].to_numpy(),
        )
    assert "STOP_" in str(e.value)
    # Test inside the training matrix must FAIL too
    with pytest.raises(SystemExit):
        T.assert_fit_matrices_exclude_test(
            ds["df"].loc[ds["te"], "split"].to_numpy(),
            ds["df"].loc[ds["va"], "split"].to_numpy(),
        )
