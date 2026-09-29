"""T0 tests for PAYOFF-TEMPORAL-01A.

Focus: TEMP57 mathematics, causal purity, boundary resets and common-support
parity. No scientific interpretation and no formal run happens here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import research.liquidity_oracle_atlas.payoff_temporal_01a_v1 as M


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def make_frame(z, *, sym="SYN", side="LONG", days=None, segs=None,
               folds=None, start_bar=0, n=None):
    n = len(z) if n is None else n
    times = pd.date_range("2024-01-01 09:00", periods=n, freq="15min")
    df = pd.DataFrame({
        "symbol": sym,
        "side": side,
        "decision_bar": np.arange(start_bar, start_bar + n),
        "decision_time": times,
        "trading_day": days if days is not None else ["2024-01-01"] * n,
        "segment": segs if segs is not None else [0] * n,
    })
    if folds is not None:
        df["fold"] = folds
    for c in M.PAY8_COLS:
        df[c] = 1.0
    df[M.Z_COL] = np.asarray(z, dtype=float)
    df[M.G_COL] = np.asarray(z, dtype=float) * 2.0
    df[M.L_COL] = 1.0
    return df


def ref_slope(y, window):
    """Slow mathematical reference (T0 only, tiny arrays)."""
    y = np.asarray(y, dtype=float)
    out = np.full(len(y), np.nan)
    for t in range(window - 1, len(y)):
        w = y[t - window + 1:t + 1]
        if np.isfinite(w).all():
            x = np.arange(window, dtype=float)
            xm, ym = x.mean(), w.mean()
            out[t] = ((x - xm) * (w - ym)).sum() / ((x - xm) ** 2).sum()
    return out


# --------------------------------------------------------------------------- #
# F. schema                                                                    #
# --------------------------------------------------------------------------- #
def test_schema_counts():
    assert len(M.PAY8_COLS) == 8
    assert len(M.TEMP57_COLS) == 57
    assert len(M.M0_COLS) == 8
    assert len(M.M1_COLS) == 65
    assert set(M.PAY8_COLS).issubset(set(M.M1_COLS))
    assert set(M.TEMP57_COLS).issubset(set(M.M1_COLS))
    assert len(set(M.M1_COLS)) == 65


def test_pay8_names_frozen():
    assert list(M.PAY8_COLS) == [
        "reward_distance_atr", "risk_distance_atr", "log_structural_rr",
        "ahead_zone_width_atr", "back_zone_width_atr", "ahead_zone_strength",
        "back_zone_strength", "atr_over_abs_price",
    ]


def test_no_excluded_features_in_m1():
    low = [c.lower() for c in M.M1_COLS]
    for tok in M.EXCLUDED_TOKENS:
        assert not any(tok in c for c in low), f"excluded token {tok} in M1"


# --------------------------------------------------------------------------- #
# A. hand-computed path                                                        #
# --------------------------------------------------------------------------- #
def test_hand_computed_path():
    Z = [1.0, 1.1, 1.3, 1.6, 2.0]
    out = M.attach_temp57(make_frame(Z), enforce_15m=True)
    last = out.iloc[4]

    assert np.isclose(last["z__delta1"], 0.40), last["z__delta1"]
    assert np.isclose(last["z__delta2"], 0.70), last["z__delta2"]
    assert np.isclose(last["z__delta4"], 1.00), last["z__delta4"]
    assert last["z__positive_run"] == 4, last["z__positive_run"]
    assert last["z__negative_run"] == 0

    # Per-bar run lengths: 0,1,2,3,4 (bar 0 has no history).
    assert out["z__positive_run"].tolist() == [0, 1, 2, 3, 4]
    # accel improves by 0.1 each step after it exists.
    assert np.isclose(out["z__accel"].iloc[2], 0.1)
    assert np.isclose(out["z__accel"].iloc[4], 0.1)

    first = out.iloc[0]
    assert pd.isna(first["z__delta1"])


def test_lag_values_are_true_lags():
    Z = [1.0, 1.1, 1.3, 1.6, 2.0]
    out = M.attach_temp57(make_frame(Z), enforce_15m=True)
    assert np.isclose(out[f"{M.Z_COL}__lag1"].iloc[4], 1.6)
    assert np.isclose(out[f"{M.Z_COL}__lag2"].iloc[4], 1.3)
    assert np.isclose(out[f"{M.Z_COL}__lag4"].iloc[4], 1.0)


# --------------------------------------------------------------------------- #
# B. reference vs production slope                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("window", [4, 16])
def test_slope_matches_reference(window):
    rng = np.random.default_rng(7)
    y = np.cumsum(rng.normal(size=40))
    prod = M.rolling_ols_slope(y, window)
    ref = ref_slope(y, window)
    assert np.allclose(prod, ref, atol=1e-10, equal_nan=True)


def test_slope_nan_on_incomplete_window():
    y = np.arange(5, dtype=float)
    s = M.rolling_ols_slope(y, 4)
    assert np.isnan(s[:3]).all()
    assert np.isfinite(s[3:]).all()


def test_slope_constant_series_is_zero():
    y = np.full(20, 3.5)
    s = M.rolling_ols_slope(y, 4)
    assert np.allclose(s[3:], 0.0, atol=1e-12)


# --------------------------------------------------------------------------- #
# C. future-mutation causality                                                 #
# --------------------------------------------------------------------------- #
def test_future_mutation_does_not_change_past():
    Z8 = [1.0, 1.1, 1.3, 1.6, 2.0, 2.5, 2.2, 3.0]
    before = M.attach_temp57(make_frame(Z8), enforce_15m=True)

    mut = make_frame(Z8)
    mut.loc[5:, M.Z_COL] = [99.0, -99.0, 42.0]
    mut.loc[5:, M.G_COL] = [7.0, 8.0, 9.0]
    mut["future_oracle_probe"] = 1.0          # must be ignored entirely
    after = M.attach_temp57(mut, enforce_15m=True)

    b = before.loc[:4, M.TEMP57_COLS].to_numpy(dtype=float)
    a = after.loc[:4, M.TEMP57_COLS].to_numpy(dtype=float)
    assert np.allclose(b, a, equal_nan=True)


# --------------------------------------------------------------------------- #
# D. boundary resets                                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kwargs,label", [
    (dict(days=["2024-01-01"] * 3 + ["2024-01-02"] * 3), "trading_day"),
    (dict(segs=[0] * 3 + [1] * 3), "segment"),
    (dict(folds=[0] * 3 + [1] * 3), "fold"),
])
def test_boundary_reset_block_scoped(kwargs, label):
    df = make_frame([1.0, 1.1, 1.2, 5.0, 5.5, 6.0], **kwargs)
    out = M.attach_temp57(df, enforce_15m=False)
    assert out["_block"].iloc[3] != out["_block"].iloc[2], label
    assert pd.isna(out["z__delta1"].iloc[3]), label
    assert out["_pos"].iloc[3] == 0, label


def test_symbol_and_side_reset():
    a = make_frame([1.0, 1.1, 1.2], sym="AAA", start_bar=0)
    b = make_frame([9.0, 9.5, 9.9], sym="BBB", start_bar=0)
    out = M.attach_temp57(pd.concat([a, b], ignore_index=True),
                          enforce_15m=False)
    assert out["_block"].iloc[3] != out["_block"].iloc[2]
    assert pd.isna(out["z__delta1"].iloc[3])

    l = make_frame([1.0, 1.1, 1.2], side="LONG")
    s = make_frame([2.0, 2.1, 2.2], side="SHORT")
    out2 = M.attach_temp57(pd.concat([l, s], ignore_index=True),
                           enforce_15m=False)
    assert out2["_block"].iloc[3] != out2["_block"].iloc[2]


def test_non_contiguous_bar_resets():
    a = make_frame([1.0, 1.1], start_bar=0)
    b = make_frame([7.0, 7.2], start_bar=10)      # gap in decision_bar
    out = M.attach_temp57(pd.concat([a, b], ignore_index=True),
                          enforce_15m=False)
    assert out["_block"].iloc[2] != out["_block"].iloc[1]
    assert pd.isna(out["z__delta1"].iloc[2])


# --------------------------------------------------------------------------- #
# E. common-support parity                                                     #
# --------------------------------------------------------------------------- #
def test_common_support_identical_for_m0_and_m1():
    # TEMP57 needs a complete 16-bar retrospective window, so inside one block
    # only rows with _pos >= 16 can enter common support.
    z = np.linspace(1.0, 3.0, 20)
    df = M.attach_temp57(make_frame(z), enforce_15m=True)
    sup = M.common_support_mask(df)
    assert int(sup.sum()) == 4, int(sup.sum())       # rows 16..19
    assert not sup[:16].any()
    assert sup[16:].all()

    kept = df[sup]
    # The SAME rows feed both arms: M0 columns are a strict subset of M1.
    assert set(M.M0_COLS).issubset(set(M.M1_COLS))
    assert kept[M.M0_COLS].notna().all().all()
    assert kept[M.TEMP57_COLS].notna().all().all()


def test_support_requires_all_temp57_finite():
    z = np.linspace(1.0, 3.0, 20)
    df = M.attach_temp57(make_frame(z), enforce_15m=True)
    sup = M.common_support_mask(df)
    assert sup[16:].all()

    # One non-finite Z removes that row AND every row whose retrospective
    # window still contains it (row 19 needs Z at 18 for delta1).
    z2 = z.copy()
    z2[18] = np.nan
    df2 = M.attach_temp57(make_frame(z2), enforce_15m=True)
    sup2 = M.common_support_mask(df2)
    assert sup2[16] and sup2[17]
    assert not sup2[18] and not sup2[19]


# --------------------------------------------------------------------------- #
# G. no future / non-whitelisted column ownership                              #
# --------------------------------------------------------------------------- #
def test_temp57_ignores_unknown_columns():
    base = make_frame([1.0, 1.1, 1.3, 1.6, 2.0])
    a = M.attach_temp57(base, enforce_15m=True)[M.TEMP57_COLS]

    polluted = base.copy()
    polluted["future_mfe"] = 123.0
    polluted["oracle_exit"] = 456.0
    b = M.attach_temp57(polluted, enforce_15m=True)[M.TEMP57_COLS]
    assert np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)


def test_temp57_schema_is_stable():
    out = M.build_temp57(M.add_continuity_blocks(
        make_frame([1.0, 1.1, 1.3, 1.6, 2.0]), enforce_15m=True))
    assert list(out.columns) == M.TEMP57_COLS


# --------------------------------------------------------------------------- #
# H. TEST never read                                                           #
# --------------------------------------------------------------------------- #
def test_only_train_labels_path_is_used():
    assert "train" in M.LABEL_TRAIN_PARQUET.name
    assert "test" not in M.LABEL_TRAIN_PARQUET.name.lower()
    assert "test" not in str(M.LABEL_TRAIN_PARQUET).lower()


def test_no_model_fit_without_running_pipeline():
    M.reset_counters()
    assert M.COUNTERS["model_fit_count"] == 0
    assert M.COUNTERS["hyperparameter_search_count"] == 0
    assert M.COUNTERS["raw_market_rescan_count"] == 0
    assert M.COUNTERS["environment_recompute_count"] == 0
    assert M.COUNTERS["geometry_recompute_count"] == 0
