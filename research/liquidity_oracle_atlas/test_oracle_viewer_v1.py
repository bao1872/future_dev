"""
test_oracle_viewer_v1
=====================

Checkpoint B tests (task FUTURE-INTRADAY-DP-ORACLE-VIEWER-R1): the read-only
DP Oracle audit overlay.

Run with the project interpreter (Python 3.11+):
    .venv/bin/python -m pytest research/liquidity_oracle_atlas/test_oracle_viewer_v1.py -v
"""

from __future__ import annotations

import importlib as _il
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from research.liquidity_oracle_atlas.build_structure_constrained_trade_oracle_dp_v1 import (
    ORACLE_ACTIONS_FILE,
    ORACLE_METADATA_FILE,
    ORACLE_TRADES_FILE,
    build_artifact_frames,
    load_oracle_artifact,
    run_base_dp,
    write_oracle_artifact,
)
from research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1 import (
    KernelCounters,
    build_base_from_arrays,
)
from research.liquidity_oracle_atlas.indicator_viewer_v1 import (
    ORACLE_TF_ONLY,
    add_dp_oracle_overlay,
    build_viewer_track,
    compute_viewport,
    oracle_fill_index,
    oracle_viewport_summary,
    select_visible_oracle_trades,
)

_page = _il.import_module("pages.6_Indicator_Viewer")
build_figure = _page.build_figure


# --------------------------------------------------------------------------- #
# synthetic helpers                                                            #
# --------------------------------------------------------------------------- #
def _base(n=900, seed=0, day_len=300, disc_index=None):
    rng = np.random.default_rng(seed)
    x = 100.0
    o = np.empty(n)
    h = np.empty(n)
    low = np.empty(n)
    c = np.empty(n)
    for i in range(n):
        x += rng.normal(0.0, 0.6)
        x += 0.08 * (100.0 - x)
        c[i] = x
        o[i] = x + rng.normal(0.0, 0.2)
        h[i] = max(o[i], c[i]) + abs(rng.normal(0.0, 0.4))
        low[i] = min(o[i], c[i]) - abs(rng.normal(0.0, 0.4))
    times = pd.date_range("2024-01-01 09:00", periods=n, freq="5min")
    day = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        np.arange(n) // day_len, unit="D"
    )
    disc = np.zeros(n, dtype=bool)
    if disc_index is not None:
        disc[disc_index] = True
    return build_base_from_arrays(
        times.to_numpy(), day.to_numpy(), o, h, low, c, disc, KernelCounters()
    )["base"]


def _track(tf="5m", n=200, seed=0):
    return build_viewer_track(
        _base(n=n, seed=seed, day_len=n), tf, symbol="SYNTH", source_sha="TESTSHA"
    )


def _trades(track, specs):
    """specs: list of (entry_x, exit_x, direction)."""
    rows = []
    for ei, xi, direction in specs:
        s = 1 if direction == "LONG" else -1
        gross = s * (float(track.open[xi]) - float(track.open[ei]))
        rows.append({
            "trade_id": f"T_{direction}_{ei}",
            "symbol": track.symbol,
            "trading_day": pd.Timestamp(track.time[ei]),
            "direction": direction,
            "entry_decision_index": ei - 1,
            "entry_fill_index": ei,
            "entry_fill_time": pd.Timestamp(track.time[ei]),
            "entry_fill_price": float(track.open[ei]),
            "entry_source_bits": 1,
            "exit_decision_index": xi - 1,
            "exit_fill_index": xi,
            "exit_fill_time": pd.Timestamp(track.time[xi]),
            "exit_fill_price": float(track.open[xi]),
            "holding_bars": xi - ei,
            "gross_points": gross,
            "cost_points": 0.0,
            "net_points": gross,
            "MFE": abs(gross) + 1.0,
            "MAE": -abs(gross) - 1.0,
            "terminal_reason": "TRADING_DAY_END",
            "training_eligible": True,
        })
    return pd.DataFrame(rows)


# =========================================================================== #
# Artifact writer / reader                                                     #
# =========================================================================== #
def test_artifact_write_read_roundtrip(tmp_path):
    res = run_base_dp(_base(n=900, day_len=300), KernelCounters(), symbol="SYNTH")
    assert res["trades"]
    outdir = write_oracle_artifact(
        res, tmp_path, oracle_source_sha="TESTSHA",
        generated_at="2024-01-01T00:00:00Z",
    )
    for fn in (ORACLE_ACTIONS_FILE, ORACLE_TRADES_FILE, ORACLE_METADATA_FILE):
        assert (outdir / fn).exists()

    loaded = load_oracle_artifact(tmp_path, "SYNTH")
    assert loaded["ok"] is True and loaded["reason"] is None
    frames = build_artifact_frames(res)
    assert len(loaded["actions"]) == len(frames["oracle_actions"])
    assert len(loaded["trades"]) == len(frames["oracle_trades"])
    assert list(loaded["actions"].columns) == list(frames["oracle_actions"].columns)


def test_artifact_metadata_preserved(tmp_path):
    res = run_base_dp(_base(n=900, day_len=300), KernelCounters(), symbol="SYNTH")
    write_oracle_artifact(res, tmp_path, oracle_source_sha="ABC123",
                          generated_at="2024-05-05T12:00:00Z")
    meta = load_oracle_artifact(tmp_path, "SYNTH")["metadata"]
    assert meta["task_id"] == "FUTURE-INTRADAY-DP-ORACLE-R1"
    assert meta["oracle_source_sha"] == "ABC123"
    assert meta["symbol"] == "SYNTH"
    assert meta["objective"] == "gross_open_to_open_pnl"
    assert meta["cost_mode"] == "zero_cost"
    assert meta["generated_at"] == "2024-05-05T12:00:00Z"
    assert meta["row_count_actions"] == len(load_oracle_artifact(tmp_path, "SYNTH")["actions"])
    assert meta["row_count_trades"] == len(load_oracle_artifact(tmp_path, "SYNTH")["trades"])


def test_artifact_missing_fail_closed(tmp_path):
    r = load_oracle_artifact(tmp_path, "NOPE")
    assert r["ok"] is False and r["reason"] == "missing_artifact"
    assert r["actions"] is None and r["trades"] is None


def test_artifact_missing_metadata_fail_closed(tmp_path):
    res = run_base_dp(_base(n=900, day_len=300), KernelCounters(), symbol="SYNTH")
    outdir = write_oracle_artifact(res, tmp_path, oracle_source_sha="S")
    (outdir / ORACLE_METADATA_FILE).unlink()
    r = load_oracle_artifact(tmp_path, "SYNTH")
    assert r["ok"] is False and r["reason"] == "missing_artifact"


def test_artifact_wrong_math_version_fail_closed(tmp_path):
    res = run_base_dp(_base(n=900, day_len=300), KernelCounters(), symbol="SYNTH")
    write_oracle_artifact(res, tmp_path, oracle_source_sha="S")
    r = load_oracle_artifact(tmp_path, "SYNTH", expected_math_version="bogus")
    assert r["ok"] is False and r["reason"] == "math_version_mismatch"


def test_artifact_wrong_symbol_fail_closed(tmp_path):
    res = run_base_dp(_base(n=900, day_len=300), KernelCounters(), symbol="SYNTH")
    outdir = write_oracle_artifact(res, tmp_path, oracle_source_sha="S")
    meta = json.loads((outdir / ORACLE_METADATA_FILE).read_text())
    meta["symbol"] = "EVIL"
    (outdir / ORACLE_METADATA_FILE).write_text(json.dumps(meta))
    r = load_oracle_artifact(tmp_path, "SYNTH")
    assert r["ok"] is False and r["reason"] == "symbol_mismatch"


def test_artifact_source_sha_fail_closed(tmp_path):
    res = run_base_dp(_base(n=900, day_len=300), KernelCounters(), symbol="SYNTH")
    write_oracle_artifact(res, tmp_path, oracle_source_sha="AAA")
    r = load_oracle_artifact(tmp_path, "SYNTH", expected_source_sha="BBB")
    assert r["ok"] is False and r["reason"] == "source_sha_mismatch"


# =========================================================================== #
# 5m time-keyed alignment                                                      #
# =========================================================================== #
def test_fill_time_to_index_exact_mapping():
    track = _track("5m", 200)
    x = 37
    assert oracle_fill_index(track, track.time[x], track.open[x]) == x
    mapped = oracle_fill_index(track, track.time[x], track.open[x])
    assert float(track.open[mapped]) == float(track.open[x])


def test_fill_price_mismatch_detected():
    track = _track("5m", 200)
    assert oracle_fill_index(track, track.time[42], float(track.open[42]) + 5.0) == -1


def test_stale_fill_time_detected():
    track = _track("5m", 200)
    bogus = pd.Timestamp(track.time[10]) + pd.Timedelta(minutes=1)
    assert oracle_fill_index(track, bogus, float(track.open[10])) == -1


def test_artifact_trades_align_to_viewer_track():
    base = _base(n=900, day_len=300)
    res = run_base_dp(base, KernelCounters(), symbol="SYNTH")
    track = build_viewer_track(base, "5m", symbol="SYNTH", source_sha="S")
    trades = build_artifact_frames(res)["oracle_trades"]
    assert len(trades) > 0
    for _, row in trades.head(80).iterrows():
        assert oracle_fill_index(track, row["entry_fill_time"], row["entry_fill_price"]) >= 0
        assert oracle_fill_index(track, row["exit_fill_time"], row["exit_fill_price"]) >= 0


# =========================================================================== #
# Viewport filtering + markers                                                 #
# =========================================================================== #
def test_viewport_filtering():
    track = _track("5m", 400)
    selected = track.n - 1
    lo, _hi = compute_viewport(track, selected)
    trades = _trades(track, [
        (lo + 1, lo + 3, "LONG"),
        (lo + 5, lo + 6, "SHORT"),
        (2, 3, "LONG"),  # outside the viewport (lo ~ 100)
    ])
    rec, mism = select_visible_oracle_trades(track, selected, trades)
    assert mism == 0
    ids = {r["trade_id"] for (r, _e, _x) in rec}
    assert "T_LONG_2" not in ids
    assert len(rec) == 2


def test_marker_traces_long_short():
    track = _track("5m", 200)
    trades = _trades(track, [(20, 30, "LONG"), (40, 55, "SHORT")])
    fig = go.Figure()
    add_dp_oracle_overlay(fig, track, track.n - 1, trades)
    names = {t.name for t in fig.data}
    assert {"Long Entry", "Long Exit", "Short Entry", "Short Exit", "Oracle trade"} <= names
    assert len(fig.data) == 5  # 4 markers + 1 connector


def test_no_one_trace_per_trade():
    track = _track("5m", 200)
    specs = [(20 + k * 15, 25 + k * 15, "LONG" if k % 2 == 0 else "SHORT")
             for k in range(11)]
    trades = _trades(track, specs)
    fig = go.Figure()
    add_dp_oracle_overlay(fig, track, track.n - 1, trades)
    assert len(fig.data) <= 5
    assert len([t for t in fig.data if t.mode == "lines"]) <= 1


def test_reversal_dual_marker():
    track = _track("5m", 200)
    trades = _trades(track, [(20, 35, "LONG"), (35, 50, "SHORT")])
    fig = go.Figure()
    add_dp_oracle_overlay(fig, track, track.n - 1, trades)
    by_name = {t.name: t for t in fig.data if t.mode != "lines"}
    assert 35 in list(by_name["Long Exit"].x)   # old trade closes
    assert 35 in list(by_name["Short Entry"].x)  # new trade opens
    # Short -> Long mirror
    track2 = _track("5m", 200)
    trades2 = _trades(track2, [(20, 35, "SHORT"), (35, 50, "LONG")])
    fig2 = go.Figure()
    add_dp_oracle_overlay(fig2, track2, track2.n - 1, trades2)
    by2 = {t.name: t for t in fig2.data if t.mode != "lines"}
    assert 35 in list(by2["Short Exit"].x)
    assert 35 in list(by2["Long Entry"].x)


def test_viewport_summary():
    track = _track("5m", 200)
    trades = _trades(track, [(20, 30, "LONG"), (40, 50, "SHORT")])
    summ = oracle_viewport_summary(track, track.n - 1, trades)
    assert summ["visible_trades"] == 2
    assert summ["long_trades"] == 1 and summ["short_trades"] == 1
    assert summ["alignment_mismatch"] == 0
    assert summ["median_holding_bars"] == 10.0


def test_misaligned_timestamp_fail_closed():
    track = _track("5m", 200)
    trades = _trades(track, [(20, 30, "LONG")])
    # corrupt the exit timestamp -> unmappable
    trades.loc[0, "exit_fill_time"] = pd.Timestamp(track.time[30]) + pd.Timedelta(minutes=7)
    rec, mism = select_visible_oracle_trades(track, track.n - 1, trades)
    assert mism == 1 and rec == []
    fig = go.Figure()
    add_dp_oracle_overlay(fig, track, track.n - 1, trades)
    assert len(fig.data) == 0  # fail-closed: nothing drawn


# =========================================================================== #
# Non-5m / OFF paths                                                           #
# =========================================================================== #
@pytest.mark.parametrize("tf", ["15m", "1H", "4H"])
def test_non_5m_no_execution_markers(tf):
    track = _track(tf, 400)
    trades = _trades(_track("5m", 200), [(5, 10, "LONG")])
    assert track.tf_label != ORACLE_TF_ONLY
    assert select_visible_oracle_trades(track, track.n - 1, trades) == ([], 0)
    fig = go.Figure()
    add_dp_oracle_overlay(fig, track, track.n - 1, trades)
    assert len(fig.data) == 0


def test_overlay_off_figure_unchanged():
    track_15m = _track("15m", 400)
    fig = build_figure(track_15m, track_15m.n - 1, True, True, True)
    n0 = len(fig.data)
    names0 = [t.name for t in fig.data]
    trades = _trades(_track("5m", 200), [(5, 10, "LONG")])
    add_dp_oracle_overlay(fig, track_15m, track_15m.n - 1, trades)
    assert len(fig.data) == n0
    assert [t.name for t in fig.data] == names0


def test_overlay_only_appends_traces():
    track = _track("5m", 200)
    fig = build_figure(track, track.n - 1, True, True, True)
    n0 = len(fig.data)
    names0 = [t.name for t in fig.data]
    add_dp_oracle_overlay(fig, track, track.n - 1, _trades(track, [(20, 30, "LONG")]))
    assert len(fig.data) > n0
    assert [t.name for t in fig.data[:n0]] == names0  # existing traces untouched


def test_overlay_does_not_alter_indicator_arrays():
    track = _track("5m", 200)
    snap = {
        "sma": track.sma.copy(), "atr": track.atr.copy(),
        "sr_top": track.sr_top.copy(), "sr_valid": track.sr_valid.copy(),
        "liq_up_valid": track.liq_up_valid.copy(),
        "trend_state": track.trend_state.copy(),
    }
    add_dp_oracle_overlay(go.Figure(), track, track.n - 1,
                          _trades(track, [(20, 30, "LONG"), (40, 55, "SHORT")]))
    assert np.array_equal(track.sma, snap["sma"], equal_nan=True)
    assert np.array_equal(track.atr, snap["atr"], equal_nan=True)
    assert np.array_equal(track.sr_top, snap["sr_top"], equal_nan=True)
    assert np.array_equal(track.sr_valid, snap["sr_valid"], equal_nan=True)
    assert np.array_equal(track.liq_up_valid, snap["liq_up_valid"], equal_nan=True)
    assert np.array_equal(track.trend_state, snap["trend_state"], equal_nan=True)


def test_no_trades_adds_no_traces():
    track = _track("5m", 200)
    fig = go.Figure()
    add_dp_oracle_overlay(fig, track, track.n - 1, _trades(track, []))
    assert len(fig.data) == 0
