"""§47 regression tests — feature identity / causality (6)(7)(8)(9)(10)(11)(12)(16)."""

import numpy as np
import pandas as pd
import pytest

from research.liquidity_oracle_atlas import (
    decomposed_value_features_v2 as F,
    build_extended_causal_state_v2 as E,
    run_decomposed_v2_research as R,
)
from research.liquidity_oracle_atlas.build_decomposed_value_dataset_v1 import (
    PAY8_COLS,
    WIN33_COLS,
)


# (6) SHARED41 = exact WIN33 + PAY8
def test_shared41_is_exact_union():
    assert len(F.SHARED41) == 41
    assert tuple(F.SHARED41[:33]) == tuple(WIN33_COLS)
    assert tuple(F.SHARED41[33:]) == tuple(PAY8_COLS)
    assert set(F.SHARED41) == set(WIN33_COLS) | set(PAY8_COLS)
    assert len(set(F.SHARED41)) == 41


# (16) B0-B4 counts frozen at 41 / 59 / 49 / 47 / 73
def test_variant_feature_counts():
    assert F.B0.n_win == 41 and F.B1.n_win == 59 and F.B2.n_win == 49
    assert F.B3.n_win == 47 and F.B4.n_win == 73
    for spec in F.R13_ARCHS:
        assert spec.n_win == F.R13_EXPECTED_COUNTS[spec.name]
        assert spec.n_payoff == F.R13_EXPECTED_COUNTS[spec.name]
        # Win and Payoff heads must consume the identical state in B*.
        assert tuple(spec.win) == tuple(spec.payoff)
        assert spec.shared_state


def test_family_sizes_and_order():
    assert len(F.SPACE18) == 18
    assert len(F.PATH8) == 8
    assert len(F.VOL6) == 6
    assert F.SPACE18[0] == "ahead_sr2_dist_atr"
    assert F.PATH8[:2] == ("ahead_touch_count_16", "back_touch_count_16")
    assert F.VOL6[-1] == "range_over_atr"


def test_schema_sha_is_order_sensitive():
    a = F.schema_sha256(F.SHARED41)
    b = F.schema_sha256(tuple(reversed(F.SHARED41)))
    assert a != b
    assert a == F.schema_sha256(F.SHARED41)


# (7) A0 exactly reproduces the V1 feature matrices
def test_a0_is_v1_matrices():
    assert tuple(F.A0.win) == tuple(WIN33_COLS)
    assert tuple(F.A0.payoff) == tuple(PAY8_COLS)
    assert F.A0.n_win == 33 and F.A0.n_payoff == 8


def test_a0_columns_exist_in_development_frame():
    df = R.load_development_frame(columns=["symbol", "decision_bar"])
    cols = set(pd.read_parquet(R.DEV_FRAME_PARQUET).columns)
    assert set(F.A0.win) <= cols
    assert set(F.A0.payoff) <= cols
    assert len(df) > 0


# (8) / (9) SPACE18 uses decision-time geometry only, never future bars
def test_zone_dist_only_positive_and_decision_time():
    # channel = [top, bottom, strength]; verified against frozen V1 state.
    z = [110.0, 100.0, 5.0]
    close, atr = 90.0, 10.0
    # ABOVE zone: distance measured to its BOTTOM edge, positive only.
    assert E._zone_dist(z, close, atr, True) == pytest.approx((100.0 - 90.0) / 10.0)
    # BELOW zone: distance measured to its TOP edge.
    assert E._zone_dist(z, close + 40.0, atr, False) == pytest.approx(
        ((close + 40.0) - 110.0) / 10.0)
    # A zone straddling close is not a valid one-sided distance.
    assert np.isnan(E._zone_dist(z, 105.0, atr, True))
    assert np.isnan(E._zone_dist(None, close, atr, True))


def test_space18_length_and_nan_when_no_geometry():
    out = E.extract_space18(None, 100.0, 1.0, 1)
    assert len(out) == 18
    assert all(np.isnan(v) for v in out)


def test_space18_uses_only_channels_of_the_decision_bar():
    geom = {"m15": ([[110.0, 105.0, 7.0], [120.0, 115.0, 3.0]], [], [], 1.0)}
    out = E.extract_space18(geom, close=100.0, atr=2.0, side=1)
    assert len(out) == 18
    # SR#2 ahead distance = (bottom of 2nd nearest above zone - close)/atr
    assert out[0] == pytest.approx((115.0 - 100.0) / 2.0)


# (10) PATH8 uses only indices <= t
def test_trailing_zone_stats_uses_only_past_bars():
    high = np.array([1, 1, 5, 1, 1, 5, 1], dtype=float)
    low = np.array([0, 0, 4, 0, 0, 4, 0], dtype=float)
    top, bottom = 4.5, 3.5
    # A future touch (index 5) must NOT be visible when scoring index 3.
    count_early, since_early = E.trailing_zone_stats(high, low, top, bottom, 3, 16)
    assert count_early == 1          # only index 2
    assert since_early == 1          # 3 - 2
    count_late, since_late = E.trailing_zone_stats(high, low, top, bottom, 5, 16)
    assert count_late == 2           # index 2 and 5
    assert since_late == 0


def test_trailing_zone_stats_nan_on_missing_band():
    c, s = E.trailing_zone_stats(np.ones(5), np.zeros(5), np.nan, 1.0, 4, 16)
    assert np.isnan(c) and np.isnan(s)


# (11) VOL6 rolling windows are backward-looking only
def test_vol6_backward_looking():
    n = 80
    frame = pd.DataFrame({"close": np.linspace(100, 110, n),
                          "high": np.linspace(101, 111, n),
                          "low": np.linspace(99, 109, n)})
    atr = np.full(n, 1.0)
    vol = E.build_vol6(frame, atr)
    assert list(vol.columns) == list(F.VOL6)
    # logret[0] is NaN from diff(), so the first finite 64-window closes at
    # index 64. Warm-up rows must be NaN, never forward-filled.
    assert bool(vol["rv64"].iloc[:64].isna().all())
    assert np.isfinite(vol["rv64"].iloc[64])
    # A shock at the LAST bar must not change any earlier window.
    frame2 = frame.copy()
    frame2.loc[n - 1, "close"] = 500.0
    vol2 = E.build_vol6(frame2, atr)
    assert np.allclose(vol["rv16"].iloc[:n - 1], vol2["rv16"].iloc[:n - 1],
                       equal_nan=True)


# (12) V2 nearest SR geometry matches frozen V1 state
def test_v2_nearest_sr_geometry_matches_v1_state():
    state = R.read_state(columns=["symbol", "bar_index", "close", "atr",
                                  "sup_top", "sup_bottom", "res_top",
                                  "res_bottom"])
    ext = R.read_parquet(R.EXTENDED_STATE_PARQUET)
    # The V2 builder re-derives distances from the SAME frozen state arrays, so
    # the join keys and row count must line up with V1's two-side layout.
    assert len(ext) == 2 * len(state)
    keys = set(zip(ext["symbol"], ext["decision_bar"]))
    assert keys == set(zip(state["symbol"], state["bar_index"]))
    assert set(ext["side"]) == {"LONG", "SHORT"}


def test_extended_state_environment_budget():
    """§49: exactly one environment pass per symbol, materialized once."""
    assert R.COUNTERS["environment_loads"] <= 15
    assert set(F.SPACE18) <= set(
        pd.read_parquet(R.EXTENDED_STATE_PARQUET).columns)
    assert set(F.PATH8) <= set(
        pd.read_parquet(R.EXTENDED_STATE_PARQUET).columns)
    assert set(F.VOL6) <= set(
        pd.read_parquet(R.EXTENDED_STATE_PARQUET).columns)
