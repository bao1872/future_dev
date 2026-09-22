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


def test_build_coverage_no_gate_current_env():
    """build_coverage reports AG/RB without raising (it only reports)."""
    cov = T2.build_coverage(T2.discover_symbols())
    assert cov["n_discovered"] == 2
    assert cov["n_oracle_ok"] == 2
    assert len(cov["coverage_rows"]) == 2
    # common window among the 2 oracle-ok symbols is computable
    assert cov["common_window"] is not None
    assert "common_start" in cov["common_window"]


def test_preflight_blocks_when_not_15():
    """In the real env only 2 symbols exist -> T2_PREFLIGHT_BLOCKED."""
    with pytest.raises(SystemExit) as exc:
        T2.preflight()
    assert "T2_PREFLIGHT_BLOCKED" in str(exc.value)


def _stub_loads(monkeypatch, symbols):
    """Make load_raw_5m / load_oracle_artifact_v2 / window helpers succeed for
    the given 15 symbols so the gate logic can be exercised positively."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})

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
                "metadata": {
                    "math_version": T2.EXPECTED_MATH_VERSION,
                    "cost_mode": T2.EXPECTED_COST_MODE,
                    "oracle_source_sha": "deadbeef",
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
