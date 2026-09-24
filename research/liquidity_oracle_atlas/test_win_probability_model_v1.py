"""§57-style Win-Probability (R9A) tests — synthetic, no materialized data."""

import numpy as np
import pandas as pd
import pytest

from research.liquidity_oracle_atlas import win_probability_model_v1 as R9A
from research.liquidity_oracle_atlas.build_decomposed_value_dataset_v1 import (
    WIN33_COLS, SIDE_KEY, HORIZONS)


def _synth_fit_frames(seed=1, n=400):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, len(WIN33_COLS))).astype("float32")
    logit = X[:, 0] - X[:, 1]
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    w = rng.uniform(0.5, 1.5, n)
    df = pd.DataFrame(X, columns=list(WIN33_COLS))
    df["horizon"] = np.resize(np.array(HORIZONS, dtype=object), n)
    df["episode_return_atr"] = (y * 2.0 - 1.0) * rng.uniform(0.5, 1.5, n)
    df["sample_weight"] = w
    df["bracket_eligible"] = True
    df["symbol"] = "SYN"
    df["decision_bar"] = np.arange(n)
    df["side"] = np.where(np.arange(n) % 2 == 0, "LONG", "SHORT")
    return {"train": df, "val": df.copy()}


def test_win33_frozen_width():
    assert len(WIN33_COLS) == 33


def test_fit_predict_returns_probabilities():
    frames = _synth_fit_frames()
    Xtr, ytr, wtr = R9A._matrices(frames["train"], "td5")
    Xv, yv, wv = R9A._matrices(frames["val"], "td5")
    b = R9A.fit_bundle(Xtr, ytr, wtr, Xv, yv, wv, "td5")
    p = R9A.predict_win_probability(b, Xv)
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert p.shape == (len(Xv),)


def test_val_diagnostics_keys():
    frames = _synth_fit_frames()
    bundles = {}
    for H in HORIZONS:
        f = {k: v.assign(horizon=H) for k, v in frames.items()}
        Xtr, ytr, wtr = R9A._matrices(f["train"], H)
        Xv, yv, wv = R9A._matrices(f["val"], H)
        bundles[H] = R9A.fit_bundle(Xtr, ytr, wtr, Xv, yv, wv, H)
    diag = R9A.val_diagnostics(bundles, frames)
    for H in HORIZONS:
        d = diag[H]
        assert {"brier", "logloss", "auc", "prob_deciles"} <= set(d)
        assert len(d["prob_deciles"]) == 10
        for row in d["prob_deciles"]:
            assert 0.0 <= row["actual_win_rate"] <= 1.0
            assert np.isfinite(row["actual_mean_return_atr"])


def test_three_horizon_fit():
    R9A.reset_counters()
    frames = _synth_fit_frames()
    bundles = {}
    for H in HORIZONS:
        f = {k: v.assign(horizon=H) for k, v in frames.items()}
        Xtr, ytr, wtr = R9A._matrices(f["train"], H)
        Xv, yv, wv = R9A._matrices(f["val"], H)
        bundles[H] = R9A.fit_bundle(Xtr, ytr, wtr, Xv, yv, wv, H)
    assert set(bundles) == set(HORIZONS)
    assert R9A.COUNTERS["classifier_fits"] == 3
    assert R9A.COUNTERS["hyperparameter_search_count"] == 0


def test_test_prediction_gated():
    with pytest.raises(RuntimeError):
        R9A.predict_test(allow_test=False)
    with pytest.raises(RuntimeError):
        R9A.predict_test(allow_test=True)  # missing authorized_review_sha
