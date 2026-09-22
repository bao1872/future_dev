"""
test_entry_value_tree_core108_t2_formal_v1
===========================================

T2 preflight gate tests. The local environment only has 2 R2 Oracle symbols
(AG, RB), so the formal run is contractually BLOCKED. These tests verify the
gate logic itself: it must STOP when the universe != 15 symbols, and must PASS
when a valid 15-symbol universe is presented (stubbed loads).
"""

from __future__ import annotations

import research.liquidity_oracle_atlas.experiment_entry_value_tree_core108_t2_formal_v1 as T2
import research.liquidity_oracle_atlas.experiment_entry_value_tree_core108_t15b_aligned_v1 as B

import numpy as np
import pandas as pd
import pytest


def test_expected_symbols_constant():
    assert T2.EXPECTED_SYMBOLS == 15
    assert T2.EXPECTED_MATH_VERSION == "intraday_dp_oracle_r2_one_entry_proximity"
    assert T2.EXPECTED_COST_MODE == "zero_cost"


def test_build_coverage_no_gate_current_env(monkeypatch):
    """build_coverage reports the universe without raising (it only reports)."""
    symbols = [f"SYM{i:02d}" for i in range(2)]
    _stub_loads(monkeypatch, symbols)
    cov = T2.build_coverage(symbols)
    assert cov["n_discovered"] == 2
    assert cov["n_oracle_ok"] == 2
    assert len(cov["coverage_rows"]) == 2
    # common window among the 2 oracle-ok symbols is computable
    assert cov["common_window"] is not None
    assert "common_start" in cov["common_window"]


def test_preflight_blocks_when_not_15(monkeypatch):
    """A 14-symbol universe (not 15) must block with T2_PREFLIGHT_BLOCKED."""
    symbols = [f"SYM{i:02d}" for i in range(14)]
    _stub_loads(monkeypatch, symbols)
    with pytest.raises(SystemExit) as exc:
        T2.preflight(symbols=symbols)
    assert "T2_PREFLIGHT_BLOCKED" in str(exc.value)


def _stub_loads(monkeypatch, symbols):
    """Make load_raw_5m / load_oracle_artifact_v2 / window helpers succeed for
    the given 15 symbols so the gate logic can be exercised positively."""
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "Q_F1_S": [0.0, 0.0, 0.0],
        "Q_F1_F": [0.0, 0.0, 0.0],
        "Q_F1_L": [0.0, 0.0, 0.0],
    })

    def fake_raw(sym):
        if sym in symbols:
            return df
        raise FileNotFoundError(sym)

    def fake_art(root, sym):
        if sym in symbols:
            return {
                "ok": True,
                "reason": None,
                "actions": df,
                "trades": df,
                "metadata": {
                    "math_version": T2.EXPECTED_MATH_VERSION,
                    "cost_mode": T2.EXPECTED_COST_MODE,
                    "oracle_source_sha": "cc7891723beb7298aa5225275b0697969d8e19bb",
                    "row_count_actions": int(len(df)),
                    "row_count_trades": int(len(df)),
                },
            }
        return {"ok": False, "reason": "missing", "actions": df, "metadata": {}}

    monkeypatch.setattr(T2, "load_raw_5m", fake_raw)
    monkeypatch.setattr(T2, "load_oracle_artifact_v2", fake_art)

    days = [pd.Timestamp("2025-01-02").date()]
    monkeypatch.setattr(
        B, "symbol_available_days",
        lambda sym, artifact_root=None: {
            "symbol": sym, "n_rows": 1, "n_days": 1,
            "first_day": days[0], "last_day": days[0], "days": days,
        },
    )
    monkeypatch.setattr(
        B, "compute_common_window",
        lambda symbols_list, artifact_root=None, drop_final_day=True: {
            "common_start": "2025-01-02", "common_end": "2025-01-02",
            "n_window_days": 1,
            "per_symbol": {s: {"first_available_day": "2025-01-02",
                               "last_available_day": "2025-01-02",
                               "last_complete_day": "2025-01-02",
                               "n_days": 1, "n_oracle_rows": 1} for s in symbols_list},
        },
    )


def test_preflight_passes_with_15_valid_symbols(monkeypatch):
    """When a valid 15-symbol universe is presented, the gate must NOT raise."""
    symbols = [f"SYM{i:02d}" for i in range(15)]
    _stub_loads(monkeypatch, symbols)
    cov = T2.preflight(symbols=symbols)
    assert cov["n_discovered"] == 15
    assert cov["n_oracle_ok"] == 15


def test_preflight_blocks_with_wrong_math_version(monkeypatch):
    """A symbol with the wrong Oracle math_version must block, even at n=15."""
    symbols = [f"SYM{i:02d}" for i in range(15)]
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})

    def fake_raw(sym):
        return df

    def fake_art(root, sym):
        return {
            "ok": True, "reason": None, "actions": df,
            "metadata": {
                # deliberately wrong math version
                "math_version": "intraday_dp_oracle_r1",
                "cost_mode": T2.EXPECTED_COST_MODE, "oracle_source_sha": "x",
            },
        }

    monkeypatch.setattr(T2, "load_raw_5m", fake_raw)
    monkeypatch.setattr(T2, "load_oracle_artifact_v2", fake_art)
    monkeypatch.setattr(
        B, "symbol_available_days",
        lambda sym, artifact_root=None: {
            "symbol": sym, "n_rows": 1, "n_days": 1,
            "first_day": pd.Timestamp("2025-01-02").date(),
            "last_day": pd.Timestamp("2025-01-02").date(), "days": [pd.Timestamp("2025-01-02").date()],
        },
    )
    monkeypatch.setattr(
        B, "compute_common_window",
        lambda symbols_list, artifact_root=None, drop_final_day=True: {
            "common_start": "2025-01-02", "common_end": "2025-01-02",
            "n_window_days": 1, "per_symbol": {},
        },
    )
    with pytest.raises(SystemExit) as exc:
        T2.preflight(symbols=symbols)
    assert "T2_PREFLIGHT_BLOCKED" in str(exc.value)


# --------------------------------------------------------------------------- #
# T2-FIX1: cross-symbol holdout + DTP-ready sensitivity unit tests
# --------------------------------------------------------------------------- #
def _fake_df(n_per_split=20, seed=0):
    rng = np.random.default_rng(seed)
    syms = [f"SYM{i:02d}" for i in range(15)]
    rows = []
    for sym in syms:
        for split, n in (("train", n_per_split), ("validation", n_per_split // 2),
                         ("test", n_per_split // 2)):
            for _ in range(n):
                rows.append({
                    "symbol": sym, "split": split,
                    "Y_L": float(rng.normal(0.0, 1.0)),
                    "Y_S": float(rng.normal(5.0, 2.0)),  # Long != Short on purpose
                    "w_norm": 1.0, "w_raw": 1.0,
                    "dtp_4tf_ready": True,
                    "f0": float(rng.normal()), "f1": float(rng.normal()),
                    "f2": float(rng.normal()),
                })
    return pd.DataFrame(rows)


def _install(monkeypatch):
    """Stub the heavy Oracle/CORE108 pieces so the fold scaffold runs fast + deterministically."""
    monkeypatch.setattr(
        T2.B, "feature_subsets",
        lambda: {"M0": ["f0"], "M1": ["f0", "f1"], "M2": ["f0", "f1", "f2"]})
    monkeypatch.setattr(
        T2.B, "build_X_subset",
        lambda df_part, cols: np.zeros((len(df_part), len(cols))))

    class _FM:
        def __init__(self, v):
            self.v = v

        def predict(self, X):
            return np.full(len(X), self.v)

    def _fit(Xtr, ytr, wtr, Xva, yva, wva):
        ytr = np.asarray(ytr, float); wtr = np.asarray(wtr, float)
        v = float(np.average(ytr, weights=wtr)) if wtr.sum() > 0 else float(np.mean(ytr))
        return _FM(v), 0, 0.0

    monkeypatch.setattr(T2.T, "fit_direction", _fit)


def test_cross_symbol_five_folds(monkeypatch):
    _install(monkeypatch)
    df = _fake_df()
    fold_rows, _, _ = T2._cross_symbol_holdout(
        df, T2.B.feature_subsets(), list(df["symbol"].unique()))
    assert len(fold_rows) == 5


def test_cross_symbol_three_held_out_per_fold(monkeypatch):
    _install(monkeypatch)
    df = _fake_df()
    fold_rows, _, _ = T2._cross_symbol_holdout(
        df, T2.B.feature_subsets(), list(df["symbol"].unique()))
    assert all(len(r["held_out_symbols"].split(",")) == 3 for r in fold_rows)


def test_cross_symbol_twelve_train_per_fold(monkeypatch):
    _install(monkeypatch)
    df = _fake_df()
    fold_rows, _, _ = T2._cross_symbol_holdout(
        df, T2.B.feature_subsets(), list(df["symbol"].unique()))
    assert all(len(r["train_symbols"].split(",")) == 12 for r in fold_rows)


def test_cross_symbol_every_symbol_held_exactly_once(monkeypatch):
    _install(monkeypatch)
    df = _fake_df()
    fold_rows, _, _ = T2._cross_symbol_holdout(
        df, T2.B.feature_subsets(), list(df["symbol"].unique()))
    held = [s for r in fold_rows for s in r["held_out_symbols"].split(",")]
    assert sorted(held) == sorted(df["symbol"].unique())
    assert len(held) == 15 and len(set(held)) == 15


def test_cross_symbol_held_absent_from_train_and_validation(monkeypatch):
    _install(monkeypatch)
    df = _fake_df()
    fold_rows, _, _ = T2._cross_symbol_holdout(
        df, T2.B.feature_subsets(), list(df["symbol"].unique()))
    sorted_syms = sorted(df["symbol"].unique())
    fold_of = {s: i % 5 for i, s in enumerate(sorted_syms)}
    folds = {f: [s for s in sorted_syms if fold_of[s] == f] for f in range(5)}
    for r in fold_rows:
        f = r["fold"]
        seen = [s for s in sorted_syms if s not in folds[f]]
        exp_tr = int(((df["symbol"].isin(seen)) & (df["split"] == "train")).sum())
        exp_va = int(((df["symbol"].isin(seen)) & (df["split"] == "validation")).sum())
        # held-out symbols must NOT contribute to this fold's Train/Validation
        assert r["n_train_rows"] == exp_tr
        assert r["n_validation_rows"] == exp_va


def test_cross_symbol_all_symbols_in_unseen_predictions(monkeypatch):
    _install(monkeypatch)
    df = _fake_df()
    _, cs_rows, _ = T2._cross_symbol_holdout(
        df, T2.B.feature_subsets(), list(df["symbol"].unique()))
    seen = {r["symbol"] for r in cs_rows if r["symbol"] != "POOLED"}
    assert seen == set(df["symbol"].unique())


def test_cross_symbol_unseen_baseline_uses_train_not_test(monkeypatch):
    """Unseen baseline must be the per-fold TRAIN weighted mean of the 12 seen
    symbols, and must NOT equal the held-out Test weighted mean (no leakage)."""
    _install(monkeypatch)
    df = _fake_df()
    _, cs_rows, _ = T2._cross_symbol_holdout(
        df, T2.B.feature_subsets(), list(df["symbol"].unique()))
    bw = [r["baseline_wRMSE"] for r in cs_rows
          if r["symbol"] == "POOLED" and r["model"] == "M0" and r["direction"] == "long"][0]

    sym_arr = df["symbol"].to_numpy()
    tr = (df["split"] == "train").to_numpy()
    te = (df["split"] == "test").to_numpy()
    w_norm = df["w_norm"].to_numpy(float)
    yL = df["Y_L"].to_numpy(float)
    sorted_syms = sorted(df["symbol"].unique())
    fold_of = {s: i % 5 for i, s in enumerate(sorted_syms)}
    folds = {f: [s for s in sorted_syms if fold_of[s] == f] for f in range(5)}
    y_all, w_all, base_parts = [], [], []
    for f in range(5):
        seen = [s for s in sorted_syms if s not in folds[f]]
        trm = np.isin(sym_arr, seen) & tr
        train_mean = float(np.average(yL[trm], weights=w_norm[trm]))
        tem = np.isin(sym_arr, folds[f]) & te
        y_all.append(yL[tem]); w_all.append(w_norm[tem])
        base_parts.append(np.full(int(tem.sum()), train_mean))
    y_all = np.concatenate(y_all); w_all = np.concatenate(w_all)
    base_pred = np.concatenate(base_parts)
    expected = T2.T.weighted_rmse(y_all, base_pred, w_all)
    assert abs(bw - expected) < 1e-9
    leaked = T2.T.weighted_rmse(
        y_all, np.full(len(y_all), float(np.average(y_all, weights=w_all))), w_all)
    assert abs(bw - leaked) > 1e-6


def test_cross_symbol_baseline_prediction_invariant_to_heldout_y(monkeypatch):
    """Mutating the held-out Test Y must NOT change the unseen baseline PREDICTION
    (it is derived from TRAIN labels only)."""
    _install(monkeypatch)
    df = _fake_df()
    _, cs_rows, _ = T2._cross_symbol_holdout(
        df, T2.B.feature_subsets(), list(df["symbol"].unique()))
    # shift held-out Test Y by a constant
    df2 = df.copy()
    df2.loc[df2["split"] == "test", "Y_L"] = df2.loc[df2["split"] == "test", "Y_L"] + 1000.0
    _, cs_rows2, _ = T2._cross_symbol_holdout(
        df2, T2.B.feature_subsets(), list(df2["symbol"].unique()))
    bw_after = [r["baseline_wRMSE"] for r in cs_rows2
                if r["symbol"] == "POOLED" and r["model"] == "M0" and r["direction"] == "long"][0]

    sym_arr = df2["symbol"].to_numpy(); tr = (df2["split"] == "train").to_numpy()
    te = (df2["split"] == "test").to_numpy(); w_norm = df2["w_norm"].to_numpy(float)
    yL = df2["Y_L"].to_numpy(float)
    sorted_syms = sorted(df2["symbol"].unique())
    fold_of = {s: i % 5 for i, s in enumerate(sorted_syms)}
    folds = {f: [s for s in sorted_syms if fold_of[s] == f] for f in range(5)}
    y_all, w_all, base_parts = [], [], []
    for f in range(5):
        seen = [s for s in sorted_syms if s not in folds[f]]
        trm = np.isin(sym_arr, seen) & tr
        train_mean = float(np.average(yL[trm], weights=w_norm[trm]))
        tem = np.isin(sym_arr, folds[f]) & te
        y_all.append(yL[tem]); w_all.append(w_norm[tem])
        base_parts.append(np.full(int(tem.sum()), train_mean))
    y_all = np.concatenate(y_all); w_all = np.concatenate(w_all)
    base_pred = np.concatenate(base_parts)
    expected_after = T2.T.weighted_rmse(y_all, base_pred, w_all)
    assert abs(bw_after - expected_after) < 1e-9


def test_dtp_ready_long_uses_long_validation_y(monkeypatch):
    """FIX: Long DTP-ready retrain must use the Long validation target, not a
    leaked Short target."""
    captured = {"calls": []}

    class _FM:
        def __init__(self, v):
            self.v = v

        def predict(self, X):
            return np.full(len(X), self.v)

    def _fit(Xtr, ytr, wtr, Xva, yva, wva):
        captured["calls"].append(np.asarray(yva, float))
        v = float(np.average(np.asarray(ytr, float), weights=np.asarray(wtr, float)))
        return _FM(v), 0, 0.0

    monkeypatch.setattr(T2.T, "fit_direction", _fit)
    monkeypatch.setattr(
        T2.B, "feature_subsets",
        lambda: {"M0": ["f0"], "M1": ["f0", "f1"], "M2": ["f0", "f1", "f2"]})
    monkeypatch.setattr(
        T2.B, "build_X_subset",
        lambda df_part, cols: np.zeros((len(df_part), len(cols))))

    df = _fake_df()
    tr = (df["split"] == "train").to_numpy()
    va = (df["split"] == "validation").to_numpy()
    te = (df["split"] == "test").to_numpy()
    w_norm = df["w_norm"].to_numpy(float)
    wva = w_norm[va]
    wva_raw = df["w_raw"].to_numpy(float)[va]
    wte_raw = df["w_raw"].to_numpy(float)[te]
    y = {"long": df["Y_L"].to_numpy(float), "short": df["Y_S"].to_numpy(float)}
    ybar = {d: float(np.average(y[d][tr], weights=df["w_raw"].to_numpy(float)[tr]))
            for d in ("long", "short")}
    ready = df["dtp_4tf_ready"].to_numpy()
    base_w = {}
    for d in ("long", "short"):
        yt = y[d][te]; yv = y[d][va]
        for m in ("M0", "M1", "M2"):
            base_w[(m, "validation")] = T2.T.weighted_rmse(
                yv, np.full(len(yv), ybar[d]), wva_raw)
            base_w[(m, "test")] = T2.T.weighted_rmse(
                yt, np.full(len(yt), ybar[d]), wte_raw)

    T2._dtp_ready_sensitivity(
        df, T2.B.feature_subsets(), y, ybar, w_norm, wva, wva_raw, wte_raw,
        va, tr, te, ready, base_w)

    long_val_y = y["long"][va]
    short_val_y = y["short"][va]
    assert len(captured["calls"]) == 4
    n_long = sum(1 for c in captured["calls"] if np.array_equal(c, long_val_y))
    n_short = sum(1 for c in captured["calls"] if np.array_equal(c, short_val_y))
    assert n_long == 2   # M1, M2 long
    assert n_short == 2  # M1, M2 short
