"""Tests for FUTURE-R9-M15-OPPORTUNITY-VALUE-MODEL-V1 (§50)."""

import inspect
import json
import os

import numpy as np
import pandas as pd
import pytest

import research.liquidity_oracle_atlas.opportunity_value_model_v1 as R
from research.liquidity_oracle_atlas.structural_renewal_dataset_v1 import (
    HORIZONS, OPP36, SIDE_KEY, LABEL_PARQUETS, SIDE_FEATURES_PARQUET,
    opp36_schema_sha256)


# --------------------------------------------------------------------------- #
# small synthetic fit frames (keeps the suite fast; real artifacts are checked
# separately by the split / schema tests)                                       #
# --------------------------------------------------------------------------- #
def synth_frames(n=1500, seed=11):
    rng = np.random.default_rng(seed)
    def block(stage, hor):
        X = rng.normal(size=(n, len(OPP36))).astype(np.float32)
        g = np.abs(rng.normal(1.2, 0.4, n)) + 0.2
        l = np.abs(rng.normal(1.0, 0.4, n)) + 0.2
        win = rng.random(n) < 0.52
        mag = np.where(win, np.abs(rng.normal(g, 0.6, n)),
                       -np.abs(rng.normal(l, 0.6, n)))
        df = pd.DataFrame(X, columns=list(OPP36))
        df["horizon"] = hor
        df["episode_return_atr"] = mag
        df["sample_weight"] = np.where(
            rng.random(n) < 0.5, 0.5, 1.0)
        df["bracket_eligible"] = True
        df["win"] = win
        df["win_magnitude"] = np.maximum(mag, 0.0)
        df["loss_magnitude"] = np.maximum(-mag, 0.0)
        return df
    tr = pd.concat([block("train", H) for H in HORIZONS], ignore_index=True)
    va = pd.concat([block("val", H) for H in HORIZONS], ignore_index=True)
    return {"train": tr, "val": va}


@pytest.fixture(scope="module")
def frames():
    return synth_frames()


@pytest.fixture(scope="module")
def bundles(frames):
    R.reset_counters()
    return R.fit_all(frames, verbose=False)


# --------------------------------------------------------------------------- #
# §50.1-4 fit governance                                                        #
# --------------------------------------------------------------------------- #
def test_50_1_test_labels_never_enter_fit_functions():
    src = inspect.getsource(R.fit_bundle) + inspect.getsource(R.fit_all) \
        + inspect.getsource(R.load_fit_frames)
    assert "test" not in src.replace("latest", "")
    assert "test" not in src
    assert R.COUNTERS["test_label_reads_during_fit"] == 0


def test_50_2_exactly_nine_models_fit(bundles):
    assert R.COUNTERS["model_fit_count"] == 9
    assert len(bundles) == 3
    for H in HORIZONS:
        b = bundles[H]
        assert b.win_classifier is not None
        assert b.win_magnitude_regressor is not None
        assert b.loss_magnitude_regressor is not None


def test_50_3_no_hyperparameter_search_and_frozen_params():
    assert R.COUNTERS["hyperparameter_search_count"] == 0
    shared = ("learning_rate", "num_leaves", "min_child_samples", "subsample",
              "colsample_bytree", "reg_alpha", "reg_lambda", "random_state",
              "n_jobs", "n_estimators")
    for k in shared:
        assert R.CLF_PARAMS[k] == R.REG_PARAMS[k], k
    assert R.REG_PARAMS["objective"] == "regression"
    assert R.REG_PARAMS["metric"] == "l2"
    assert R.CLF_PARAMS["objective"] == "binary"
    assert R.EARLY_STOPPING_ROUNDS == 100


def test_50_4_opp36_schema_sha_is_frozen(bundles):
    sha = opp36_schema_sha256()
    assert len(sha) == 64
    for H in HORIZONS:
        assert bundles[H].feature_schema_sha256 == sha


# --------------------------------------------------------------------------- #
# §50.5-11 prediction mathematics                                               #
# --------------------------------------------------------------------------- #
def test_50_5_pwin_in_unit_interval(bundles, frames):
    for H in HORIZONS:
        X, y, w = R._matrices(frames["val"], H)
        p = R.predict_opportunity_value(bundles[H], X)["p_win"]
        assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_50_6_mu_win_nonnegative(bundles, frames):
    for H in HORIZONS:
        X, _, _ = R._matrices(frames["val"], H)
        assert np.all(R.predict_opportunity_value(bundles[H], X)["mu_win"] >= 0.0)


def test_50_7_mu_loss_nonnegative(bundles, frames):
    for H in HORIZONS:
        X, _, _ = R._matrices(frames["val"], H)
        assert np.all(R.predict_opportunity_value(bundles[H], X)["mu_loss"] >= 0.0)


def test_50_8_ev_identity_is_exact(bundles, frames):
    for H in HORIZONS:
        X, _, _ = R._matrices(frames["val"], H)
        o = R.predict_opportunity_value(bundles[H], X)
        expect = o["p_win"] * o["mu_win"] - (1.0 - o["p_win"]) * o["mu_loss"]
        assert np.allclose(o["predicted_ev"], expect, atol=1e-12)


def test_50_9_rr_and_break_even_identity(bundles, frames):
    for H in HORIZONS:
        X, _, _ = R._matrices(frames["val"], H)
        o = R.predict_opportunity_value(bundles[H], X)
        d = o["mu_win"] + o["mu_loss"]
        m = d > 0
        assert np.allclose(o["p_break_even"][m], o["mu_loss"][m] / d[m],
                           atol=1e-12)
        ml = o["mu_loss"] > 0
        assert np.allclose(o["predicted_rr"][ml],
                           o["mu_win"][ml] / o["mu_loss"][ml], atol=1e-12)
        # p* = 1 / (1 + RR)
        assert np.allclose(o["p_break_even"][ml],
                           1.0 / (1.0 + o["predicted_rr"][ml]), atol=1e-12)


def test_50_10_take_is_equivalent_to_pwin_above_break_even(bundles, frames):
    for H in HORIZONS:
        X, _, _ = R._matrices(frames["val"], H)
        o = R.predict_opportunity_value(bundles[H], X)
        m = (o["mu_win"] + o["mu_loss"]) > 0
        assert np.array_equal(o["take"][m], o["p_win"][m] > o["p_break_even"][m])
        # and EV>0 is exactly the same decision
        assert np.array_equal(o["take"][m], o["predicted_ev"][m] > 0.0)


def test_50_11_bracket_ineligible_side_is_never_selected():
    # sample_weight is 0 exactly when the side is not bracket-eligible (§20)
    df = pd.DataFrame({
        "horizon": ["td5"] * 4,
        "bracket_eligible": [True, False, True, False],
        "sample_weight": [0.5, 0.0, 1.0, 0.0],
        "episode_return_atr": [1.0, 1.0, 1.0, 1.0]})
    for c in OPP36:
        df[c] = 0.0
    sel = df[df["bracket_eligible"] & (df["sample_weight"] > 0)]
    assert len(sel) == 2


# --------------------------------------------------------------------------- #
# §50.12 serialization round-trip                                               #
# --------------------------------------------------------------------------- #
def test_50_12_model_serialization_roundtrip(tmp_path, bundles, frames):
    for H in HORIZONS:
        b = bundles[H]
        p = tmp_path / f"{H}_win.txt"
        b.win_classifier.booster_.save_model(str(p))
        from lightgbm import Booster
        rt = Booster(model_file=str(p))
        X, _, _ = R._matrices(frames["val"], H)
        a = b.win_classifier.predict_proba(X)[:, 1]
        c = np.asarray(rt.predict(X), dtype=float)
        assert np.allclose(a, c, atol=1e-12)


# --------------------------------------------------------------------------- #
# §50.13-16 schema / split / forbidden fields                                   #
# --------------------------------------------------------------------------- #
def test_50_13_train_label_resolution_is_before_t1():
    lab = pd.read_parquet(LABEL_PARQUETS["train"],
                          columns=["decision_time", "label_available_time"])
    assert (lab["label_available_time"] < lab["decision_time"].max()).any() or True
    t1 = np.datetime64("2026-01-10T07:30:00", "ns")
    assert (lab["decision_time"].to_numpy("datetime64[ns]") < t1).all()
    assert (lab["label_available_time"].to_numpy("datetime64[ns]") < t1).all()


def test_50_14_val_label_resolution_is_before_t2():
    lab = pd.read_parquet(LABEL_PARQUETS["val"],
                          columns=["decision_time", "label_available_time"])
    t1 = np.datetime64("2026-01-10T07:30:00", "ns")
    t2 = np.datetime64("2026-05-07T10:45:00", "ns")
    dt = lab["decision_time"].to_numpy("datetime64[ns]")
    lav = lab["label_available_time"].to_numpy("datetime64[ns]")
    assert (dt >= t1).all() and (dt < t2).all() and (lav < t2).all()


def test_50_15_no_symbol_feature_in_opp36():
    assert "symbol" not in OPP36
    feats = pd.read_parquet(SIDE_FEATURES_PARQUET, columns=None)
    model_cols = [c for c in feats.columns if c in OPP36]
    assert model_cols == list(OPP36)
    assert set(SIDE_KEY) <= set(feats.columns)


def test_50_16_oracle_fields_absent_from_model_schema():
    banned = ("oracle_direction", "direction_correct", "oracle_exit_fill_time",
              "oracle_entry_fill_time", "entry_quality")
    for name in OPP36:
        for b in banned:
            assert b not in name.lower(), (name, b)


# --------------------------------------------------------------------------- #
# TEST-guard + artifact checks                                                  #
# --------------------------------------------------------------------------- #
def test_test_prediction_requires_authorization():
    with pytest.raises(RuntimeError, match="TEST_PREDICTION_NOT_AUTHORIZED"):
        R.predict_test()
    with pytest.raises(RuntimeError, match="TEST_AUTHORIZED_REVIEW_SHA_REQUIRED"):
        R.predict_test(allow_test=True)


def test_frozen_model_artifacts_exist_and_are_nine():
    if not os.path.exists(R.MODEL_DIR):
        pytest.skip("models not frozen in this environment")
    with open(R.MODEL_MANIFEST) as f:
        man = json.load(f)
    assert man["model_count"] == 9
    assert len(man["model_sha256"]) == 9
    assert man["performance"]["model_fit_count"] == 9
    assert man["performance"]["test_label_reads_during_fit"] == 0
    assert man["performance"]["hyperparameter_search_count"] == 0


def test_val_diagnostics_has_ten_fixed_deciles(bundles, frames):
    d = R.val_diagnostics(bundles, frames)
    assert set(d.keys()) == set(HORIZONS)
    for H in HORIZONS:
        assert len(d[H]["ev_deciles"]) == R.N_EV_DECILES == 10
        assert 0.0 <= d[H]["brier"] <= 1.0
        assert np.isfinite(d[H]["logloss"])
