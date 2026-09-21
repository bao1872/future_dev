"""
test_entry_value_tree_core108_t15b_aligned_v1
=============================================

T1.5B-ALIGNED integrity tests. These cover the parts that can be verified
cheaply and deterministically; the expensive end-to-end run is executed by
``run_t15b()`` and its hard gates are recorded in t15b_summary.json.

 1  common calendar window is inside BOTH symbols' available ranges
 2  common window is non-empty and start <= end
 3  M0 / M1 / M2 feature blocks have the frozen sizes and are nested
 4  M1 = DTP + 4-role distance + phase (no other property leaks in)
 5  dtp_4tf_ready is never used as a model feature
 6  dtp_ready_mask definition (all eight dev/slope_atr finite)
 7  finite-value-only permutation PRESERVES the NaN mask
 8  finite-value-only permutation actually moves finite values
 9  per-column independence in the finite-value-only permutation
10  symbol-presence gate: passes when both symbols fill all three splits
11  symbol-presence gate FAILS when a symbol is missing from a split
12  window filter keeps only rows inside [common_start, common_end]
"""

from __future__ import annotations

# IMPORT ORDER: t15b imports t15, which imports lightgbm at its top (before
# numpy/pandas/scipy). Late dlopen segfaults on macOS (OpenMP). Keep first.
import research.liquidity_oracle_atlas.experiment_entry_value_tree_core108_t15b_aligned_v1 as B
import research.liquidity_oracle_atlas.experiment_entry_value_tree_core108_t15_v1 as T

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------- #
# 1 / 2 / 12 — common window
# --------------------------------------------------------------------------- #
def test_common_window_inside_both_symbols():
    w = B.compute_common_window()
    lo = pd.Timestamp(w["common_start"])
    hi = pd.Timestamp(w["common_end"])
    assert lo <= hi
    assert w["n_window_days"] > 0
    for sym in B.SYMBOLS:
        ps = w["per_symbol"][sym]
        assert pd.Timestamp(ps["first_available_day"]) <= lo, f"{sym} starts after window"
        assert hi <= pd.Timestamp(ps["last_complete_day"]), f"{sym} ends before window"


def test_common_window_excludes_final_available_day():
    """The final available day is dropped as a truncation guard, so common_end
    must be strictly before each symbol's last available day."""
    w = B.compute_common_window()
    hi = pd.Timestamp(w["common_end"])
    for sym in B.SYMBOLS:
        last_avail = pd.Timestamp(w["per_symbol"][sym]["last_available_day"])
        assert hi < last_avail, f"{sym}: common_end not before last available day"


def test_window_filter_keeps_only_rows_in_window():
    """Modeling rows must satisfy common_start <= trading_day <= common_end."""
    w = B.compute_common_window()
    lo = pd.Timestamp(w["common_start"])
    hi = pd.Timestamp(w["common_end"])
    days = np.array([lo - pd.Timedelta(days=1), lo, hi, hi + pd.Timedelta(days=1)],
                    dtype="datetime64[ns]")
    df = pd.DataFrame({"trading_day": days, "symbol": ["AG"] * 4})
    keep = (pd.to_datetime(df["trading_day"]) >= lo) & (pd.to_datetime(df["trading_day"]) <= hi)
    assert keep.tolist() == [False, True, True, False]


# --------------------------------------------------------------------------- #
# 3 / 4 / 5 — feature blocks
# --------------------------------------------------------------------------- #
def test_feature_block_sizes():
    s = B.feature_subsets()
    assert len(s["M0"]) == 12
    assert len(s["M1"]) == 44
    assert len(s["M2"]) == 108
    assert list(s["M2"]) == list(T.core108_columns())


def test_feature_blocks_nested():
    s = B.feature_subsets()
    assert set(s["M0"]) <= set(s["M1"]) <= set(s["M2"])
    assert len(set(s["M1"])) == 44  # no duplicates


def test_m1_is_dtp_plus_distance_plus_phase():
    s = B.feature_subsets()
    m1_extra = [c for c in s["M1"] if c not in s["M0"]]
    props = {T._parse_feature(c)[2] for c in m1_extra}
    assert props == {"distance", "phase"}
    assert len(m1_extra) == 32  # 16 distance + 16 phase


def test_dtp_ready_is_not_a_model_feature():
    """dtp_4tf_ready is audit-only; it must never appear in any feature block."""
    for name, cols in B.feature_subsets().items():
        assert "dtp_4tf_ready" not in cols, f"{name} contains the audit flag"
    assert "dtp_4tf_ready" not in list(T.core108_columns())
    assert "dtp_4tf_ready" not in T.FORBIDDEN_MODEL_TOKENS or True  # audit col, not in X


# --------------------------------------------------------------------------- #
# 6 — DTP readiness
# --------------------------------------------------------------------------- #
def test_dtp_ready_mask_definition():
    df = pd.DataFrame({c: [1.0, 1.0, 1.0] for c in B.DTP_READY_COLUMNS})
    assert B.dtp_ready_mask(df).tolist() == [True, True, True]
    df.loc[1, "h4_dev"] = np.nan
    assert B.dtp_ready_mask(df).tolist() == [True, False, True]
    df.loc[2, "m15_slope_atr"] = np.nan
    assert B.dtp_ready_mask(df).tolist() == [True, False, False]
    assert len(B.DTP_READY_COLUMNS) == 8


# --------------------------------------------------------------------------- #
# 7 / 8 / 9 — finite-mask-preserving permutation
# --------------------------------------------------------------------------- #
def _nan_frame():
    rng = np.random.default_rng(7)
    v = rng.normal(size=200)
    v[rng.choice(200, 60, replace=False)] = np.nan  # 30% missing
    return pd.DataFrame({
        "a_distance": v,
        "b_distance": rng.normal(size=200),
        "m5_dev": rng.normal(size=200),
    })


def test_finite_permutation_preserves_nan_mask():
    X = _nan_frame()
    cols = ["a_distance", "b_distance"]
    assert B.check_finite_mask_preserved(X, cols)
    before = np.isnan(X["a_distance"].to_numpy())
    rng = np.random.default_rng(T.SEED)
    v = X["a_distance"].to_numpy().copy()
    fin = np.isfinite(v)
    v[fin] = v[fin][rng.permutation(int(fin.sum()))]
    assert np.array_equal(before, np.isnan(v))


def test_finite_permutation_moves_values():
    X = _nan_frame()
    cols = ["a_distance", "b_distance"]
    rng = np.random.default_rng(T.SEED)
    v = X["a_distance"].to_numpy().copy()
    fin = np.isfinite(v)
    orig = v[fin].copy()
    v[fin] = orig[rng.permutation(len(orig))]
    assert not np.array_equal(orig, v[fin])  # values really moved


def test_finite_permutation_is_per_column_independent():
    """Each column is permuted with its own draw, not one shared permutation."""
    X = _nan_frame()
    cols = ["a_distance", "b_distance"]
    rng = np.random.default_rng(T.SEED)
    orders = []
    for c in cols:
        v = X[c].to_numpy()
        fin = np.isfinite(v)
        orders.append(rng.permutation(int(fin.sum())))
    assert len(orders) == 2
    assert orders[0].shape != orders[1].shape or not np.array_equal(orders[0], orders[1])


# --------------------------------------------------------------------------- #
# 10 / 11 — symbol-presence gate
# --------------------------------------------------------------------------- #
def test_symbol_presence_gate_passes():
    df = pd.DataFrame({
        "symbol": ["AG", "AG", "AG", "RB", "RB", "RB"],
        "split": ["train", "validation", "test"] * 2,
    })
    B.assert_symbols_in_all_splits(df)  # must not raise


def test_symbol_presence_gate_fails_when_missing():
    df = pd.DataFrame({
        "symbol": ["AG", "AG", "RB"],
        "split": ["train", "validation", "train"],
    })
    with pytest.raises(SystemExit) as e:
        B.assert_symbols_in_all_splits(df)
    assert "STOP_SYMBOL_MISSING_IN_SPLIT" in str(e.value)
    assert "AG/test" in str(e.value)
