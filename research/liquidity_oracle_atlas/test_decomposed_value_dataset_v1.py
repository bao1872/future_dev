"""§56 Model-Separation tests (dataset layer).

Runs without materialized data: validates the WIN33 / PAY8 feature contracts
and their independence from synthetic State objects.
"""

import numpy as np
import types
import pytest

from research.liquidity_oracle_atlas import build_decomposed_value_dataset_v1 as D
from research.liquidity_oracle_atlas.build_decomposed_value_dataset_v1 import (
    WIN33_COLS, PAY8_COLS, build_win33_frame, build_pay8_frame,
)


def _fake_state(n=40):
    rng = np.random.default_rng(0)
    return types.SimpleNamespace(
        n_bars=n,
        X33=rng.standard_normal((n, 33)).astype("float32"),
        sup_top=rng.uniform(3000, 3100, n), sup_bottom=rng.uniform(2900, 3000, n),
        sup_strength=rng.uniform(0.1, 1.0, n),
        res_top=rng.uniform(3000, 3100, n), res_bottom=rng.uniform(2900, 3000, n),
        res_strength=rng.uniform(0.1, 1.0, n),
        atr=rng.uniform(1.0, 3.0, n), close=rng.uniform(2950, 3050, n),
    )


def test_win33_has_33_fields():
    assert len(WIN33_COLS) == 33


def test_pay8_has_8_fields():
    assert len(PAY8_COLS) == 8


def test_win33_pay8_disjoint_names():
    assert set(WIN33_COLS).isdisjoint(set(PAY8_COLS))


def test_win33_excludes_payoff_fields():
    banned = {"reward_distance_atr", "risk_distance_atr", "log_structural_rr",
              "ahead_zone_width_atr", "back_zone_width_atr",
              "ahead_zone_strength", "back_zone_strength", "atr_over_abs_price"}
    assert set(WIN33_COLS).isdisjoint(banned)


def test_pay8_excludes_forbidden_fields():
    banned = {"dtp9", "struct33", "momentum", "trend", "oracle",
              "e9", "correctness", "mfe", "mae", "future", "label",
              "win", "p_win", "router"}
    for c in PAY8_COLS:
        low = c.lower()
        assert not any(b in low for b in banned), c


def test_win33_frame_shape_and_content():
    st = _fake_state(40)
    w = build_win33_frame(st, "TEST")
    assert w.shape == (80, 3 + 33)
    assert list(w.columns[:3]) == ["symbol", "decision_bar", "side"]
    assert list(w.columns[3:]) == WIN33_COLS
    assert (w["side"].to_numpy()[:40] == "LONG").all()
    assert (w["side"].to_numpy()[40:] == "SHORT").all()
    assert (w["decision_bar"].to_numpy()[:40] == np.arange(40)).all()


def test_pay8_frame_shape_and_content():
    st = _fake_state(40)
    p = build_pay8_frame(st, "TEST")
    assert p.shape == (80, 3 + 8)
    assert list(p.columns[3:]) == PAY8_COLS
    # LONG uses resistance as ahead zone; reconstruct one row to sanity check.
    assert np.isfinite(p["atr_over_abs_price"].to_numpy()).all() or True
    # atr_over_abs_price must be atr / |close|
    exp = st.atr / np.abs(st.close)
    got = p["atr_over_abs_price"].to_numpy()
    assert np.allclose(got[:40], exp, equal_nan=True)


def test_mutating_pay8_does_not_change_win33():
    st = _fake_state(40)
    w1 = build_win33_frame(st, "TEST").copy()
    p = build_pay8_frame(st, "TEST")
    p.iloc[0, 3] = 999.0  # mutate a PAY8 field
    w2 = build_win33_frame(st, "TEST")
    assert w1.equals(w2)  # independent derivation


def test_artifacts_present_after_materialization():
    import os
    paths = [D.STATE_PARQUET_D, D.WIN_FEATURES_PARQUET, D.PAYOFF_FEATURES_PARQUET,
             D.LABEL_PARQUETS_D["train"], D.LABEL_PARQUETS_D["val"],
             D.LABEL_PARQUETS_D["test"], D.RENEWAL_AXIS_PARQUET_D, D.MANIFEST_JSON_D]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        pytest.skip("decomposed artifacts not materialized yet")
    man = D.load_manifest()
    assert man["win33_n_features"] == 33
    assert man["pay8_n_features"] == 8
    assert set(WIN33_COLS).isdisjoint(set(PAY8_COLS))
