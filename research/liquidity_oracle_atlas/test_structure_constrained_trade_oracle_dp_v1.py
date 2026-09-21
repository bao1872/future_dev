"""
test_structure_constrained_trade_oracle_dp_v1
=============================================

T0 gates + T1 differential + TP performance/memory gate for the Intraday
Structure-Constrained Trade Oracle DP V1 (task FUTURE-INTRADAY-DP-ORACLE-R1).

Contract covered (§19):
  * entry mask = current 5m range touches a PRE-EXISTING SR/Liquidity zone
    (delta = 0), 8-bit TF x {SR,LIQ}; broken liquidity excluded; a structure
    formed by the current bar can never retroactively qualify the current bar;
  * no 0.5*ATR proximity on the DP entry;
  * Flat->entry / reversal only when eligible; exit free anywhere;
  * per (trading_day, segment) unit, Flat at both ends, no cross day / segment;
  * next-open execution; gross PnL objective with a cost interface (V1 c=0);
  * full 3x3 Q counterfactuals;
  * production DP == independent exhaustive reference;
  * PnL reconstruction == Bellman value; sign symmetry; true tie;
  * N/2N/4N time + memory scaling.

Run with the project interpreter (Python 3.11+):
    .venv/bin/python -m pytest research/liquidity_oracle_atlas/test_structure_constrained_trade_oracle_dp_v1.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from research.export_ob_trigger_execution_v21 import load_raw_5m
from research.liquidity_oracle_atlas.build_structure_constrained_trade_oracle_dp_v1 import (
    DATA_END,
    DISCONTINUITY,
    P2I,
    POS,
    TRADING_DAY_END,
    artifact_metadata,
    build_artifact_frames,
    build_intraday_units,
    exhaustive_reference,
    run_arrays_dp,
    run_base_dp,
    run_symbol_dp,
    solve_day_dp,
)
from research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1 import (
    MASK_BIT,
    KernelCounters,
    bar_hits_zone,
    build_base_from_arrays,
    entry_bits_from_prev_geometry,
    stream_from_base,
)
from research.phase1_tradability.phase1_contract_v1 import discontinuity_flags


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def prod_path(core, start, end):
    p = 0
    out = []
    for t in range(start, end):
        pi = P2I[p]
        a = int(core["actions"][t, pi])
        out.append(a)
        p = a
    return out


def _solve(opens, entry_ok, cost=None, start=0, end=None):
    opens = np.asarray(opens, float)
    entry_ok = np.asarray(entry_ok, bool)
    if cost is None:
        cost = np.zeros(len(opens))
    cost = np.asarray(cost, float)
    if end is None:
        end = len(opens) - 2
    core = solve_day_dp(opens, entry_ok, cost, start, end)
    best, paths = exhaustive_reference(opens, entry_ok, cost, start, end)
    return core, best, paths, start, end, opens, entry_ok, cost


def _synth_ohlc(n, seed=0, disc_index=None, day_len=None):
    rng = np.random.default_rng(seed)
    x = 100.0
    o = np.empty(n)
    h = np.empty(n)
    l = np.empty(n)
    c = np.empty(n)
    for i in range(n):
        x += rng.normal(0.0, 0.6)
        x += 0.08 * (100.0 - x)
        c[i] = x
        o[i] = x + rng.normal(0.0, 0.2)
        h[i] = max(o[i], c[i]) + abs(rng.normal(0.0, 0.4))
        l[i] = min(o[i], c[i]) - abs(rng.normal(0.0, 0.4))
    t = pd.date_range("2024-01-01 09:00", periods=n, freq="5min")
    # trading_day: bump every `day_len` bars (default: single day)
    day_len = day_len or n
    day_codes = np.arange(n) // day_len
    day = pd.to_datetime("2024-01-01") + pd.to_timedelta(day_codes, unit="D")
    disc = np.zeros(n, dtype=bool)
    if disc_index is not None:
        disc[disc_index] = True
    return t, day.to_numpy(), o, h, l, c, disc


def _synth_base(n=900, seed=0, disc_index=None, day_len=None):
    t, day, o, h, l, c, disc = _synth_ohlc(
        n, seed=seed, disc_index=disc_index, day_len=day_len
    )
    return build_base_from_arrays(t, day, o, h, l, c, disc, KernelCounters())["base"]


def _geom(channels=None, liq_up=None, liq_down=None):
    return {
        "m5": (
            list(channels or []),
            list(liq_up or []),
            list(liq_down or []),
        )
    }


def _liq(level, top, bottom, broken=False):
    return {
        "left": 10,
        "level": level,
        "top": top,
        "bottom": bottom,
        "broken": broken,
        "breach_i": None,
    }


# =========================================================================== #
# T0 — Entry mask (kernel)                                                     #
# =========================================================================== #
def test_mask_sr_touch_eligible():
    prev = _geom(channels=[(105.0, 100.0, 2.0)])  # zone [100,105]
    assert entry_bits_from_prev_geometry(99.0, 101.0, prev) != 0
    bit = MASK_BIT[("m5", "SR")]
    assert entry_bits_from_prev_geometry(99.0, 101.0, prev) == (1 << bit)


def test_mask_sr_touch_then_close_far_still_eligible():
    # range touches the zone but close is far away -> still eligible (range based)
    prev = _geom(channels=[(105.0, 100.0, 2.0)])
    assert bar_hits_zone(98.0, 100.5, 100.0, 105.0) is True
    assert entry_bits_from_prev_geometry(98.0, 100.5, prev) != 0


def test_mask_liquidity_touch_eligible():
    prev = _geom(liq_up=[_liq(103.0, 104.0, 102.0)])
    bit = MASK_BIT[("m5", "LIQ")]
    assert entry_bits_from_prev_geometry(101.5, 103.5, prev) == (1 << bit)


def test_mask_no_touch_ineligible():
    prev = _geom(channels=[(105.0, 100.0, 2.0)], liq_down=[_liq(90.0, 91.0, 89.0)])
    assert entry_bits_from_prev_geometry(120.0, 121.0, prev) == 0
    assert entry_bits_from_prev_geometry(80.0, 81.0, prev) == 0


def test_mask_broken_liquidity_excluded():
    prev = _geom(liq_up=[_liq(103.0, 104.0, 102.0, broken=True)])
    assert entry_bits_from_prev_geometry(101.5, 103.5, prev) == 0


def test_mask_new_structure_cannot_retroactively_qualify():
    # The mask is a pure function of the PREVIOUS geometry: a zone present only
    # "now" (empty prev) can never make the current bar eligible.
    assert entry_bits_from_prev_geometry(100.0, 106.0, {}) == 0
    # first bar of a stream / segment has no previous geometry -> mask 0
    base = _synth_base(n=900, seed=2)
    r = stream_from_base(
        base, KernelCounters(), capture_entry_mask=True, emit_events=False
    )
    assert int(r["entry_mask"][0]) == 0


def test_mask_future_mutation_unchanged():
    n = 900
    base = _synth_base(n=n, seed=3)
    a = stream_from_base(
        base, KernelCounters(), capture_entry_mask=True, emit_events=False
    )
    mask_a = a["entry_mask"]
    assert int((mask_a != 0).sum()) > 0

    K = n - 1
    base2 = base.copy().reset_index(drop=True)
    base2.at[base2.index[K], "close"] = float(base["close"].iloc[K]) + 5.0
    b = stream_from_base(
        base2, KernelCounters(), capture_entry_mask=True, emit_events=False
    )
    assert np.array_equal(mask_a[:K], b["entry_mask"][:K]), (
        "future bar changed past mask"
    )


def test_mask_8bit_range():
    base = _synth_base(n=900, seed=4)
    r = stream_from_base(
        base, KernelCounters(), capture_entry_mask=True, emit_events=False
    )
    assert int(r["entry_mask"].max()) < 256
    assert r["entry_eligible"].dtype == bool


# =========================================================================== #
# T0 — DP action space                                                         #
# =========================================================================== #
def test_dp_flat_entry_only_when_eligible():
    opens = [100, 100, 105, 110, 115, 120, 125]
    core, _best, _paths, s, e, *_ = _solve(opens, [False] * 5)
    for t in range(s, e):
        assert int(core["actions"][t, P2I[0]]) == 0  # never enters when ineligible


def test_dp_exit_allowed_anywhere():
    # enter long at t=1 (eligible), price falls; optimal exit occurs while the
    # entry mask is 0 (exit is unrestricted).
    opens = [100, 100, 100, 110, 105, 100, 95]
    entry = [False, True, False, False, False]
    core = solve_day_dp(
        np.asarray(opens, float),
        np.asarray(entry, bool),
        np.zeros(len(opens)),
        0,
        len(opens) - 2,
    )
    path = prod_path(core, 0, 5)
    assert path[1] == 1 and path[2] == 0  # long entry then exit at a non-eligible bar
    assert entry[2] is False


def test_dp_reversal_requires_eligible():
    # reversal candidate present but ineligible -> no reversal happens
    opens = [100, 100, 100, 100, 100, 100, 100]
    entry = [True, False, False, False, False]
    core = solve_day_dp(
        np.asarray(opens, float),
        np.asarray(entry, bool),
        np.zeros(len(opens)),
        0,
        len(opens) - 2,
    )
    path = prod_path(core, 0, 5)
    assert path[0] == 0  # flat stays flat (no eligible, no move)
    for k in range(1, len(path)):
        assert not (path[k - 1] != 0 and path[k] == -path[k - 1])


def test_dp_day_start_end_flat():
    opens = [100, 101, 103, 106, 110, 115, 121]
    core, _b, _p, s, e, *_ = _solve(opens, [True] * 5)
    assert prod_path(core, s, e)[0] in (-1, 0, 1)  # start decision
    assert prod_path(core, s, e)[-1] == 0  # terminal forced flat
    for p in POS:
        assert int(core["actions"][e - 1, P2I[int(p)]]) == 0


def test_dp_next_open_fill():
    base = _synth_base(n=900, seed=5, day_len=300)
    res = run_base_dp(base, KernelCounters(), symbol="SYNTH")
    assert res["trades"], "expected trades"
    for tr in res["trades"]:
        assert tr["entry_fill_index"] == tr["entry_decision_index"] + 1
        assert tr["exit_fill_index"] == tr["exit_decision_index"] + 1


def test_dp_no_cross_unit():
    base = _synth_base(n=900, seed=6, disc_index=450, day_len=200)
    res = run_base_dp(base, KernelCounters(), symbol="SYNTH")
    td = pd.to_datetime(res["trading_day"]).to_numpy()
    seg = res["segment"]
    assert res["trades"], "expected trades"
    for tr in res["trades"]:
        ei, xi = int(tr["entry_fill_index"]), int(tr["exit_fill_index"])
        assert td[ei] == td[xi], "trade crosses trading_day"
        assert seg[ei] == seg[xi], "trade crosses discontinuity"
    # no unit ends non-flat
    for u in res["units"]:
        e = u["seg_end"]
        if u["seg_end"] - u["seg_start"] < 2:
            continue
        assert int(res["decision"]["pos_after"][e - 1]) == 0


def test_build_intraday_units():
    td = np.array(["2024-01-01"] * 3 + ["2024-01-02"] * 3, dtype="datetime64[ns]")
    seg = np.array([0, 0, 0, 0, 0, 0])
    starts, ends = build_intraday_units(td, seg)
    assert list(starts) == [0, 3]
    assert list(ends) == [2, 5]

    seg2 = np.array([0, 0, 1, 1, 1, 1])  # discontinuity mid-day
    starts2, ends2 = build_intraday_units(td, seg2)
    assert list(starts2) == [0, 2, 3]
    assert list(ends2) == [1, 2, 5]


# =========================================================================== #
# T0 — correctness vs exhaustive reference / reconstruction / symmetry / tie    #
# =========================================================================== #
def test_dp_production_equals_reference():
    mism = {"val": 0, "path": 0, "state": 0}
    for seed in range(50):
        rng = np.random.default_rng(seed)
        D = 5 + seed % 6
        opens = np.array([100.0] + list(100.0 + np.cumsum(rng.normal(0, 1.0, D + 1))))
        entry = rng.random(D + 2) < 0.5
        cost = np.zeros(len(opens))
        core, best, paths, s, e, *_ = _solve(opens, entry, cost, end=D)
        v = float(np.nanmax(core["Q"][0, P2I[0], :]))
        if abs(v - best) > 1e-6:
            mism["val"] += 1
        pp = tuple(prod_path(core, s, e))
        if pp not in paths:
            mism["path"] += 1
        for p0 in POS:
            b2, _ = exhaustive_reference(
                np.asarray(opens, float),
                np.asarray(entry, bool),
                cost,
                0,
                D,
                start_pos=int(p0),
            )
            vv = float(np.nanmax(core["Q"][0, P2I[int(p0)], :]))
            if abs(vv - b2) > 1e-6:
                mism["state"] += 1
    assert mism == {"val": 0, "path": 0, "state": 0}, mism


def test_dp_pnl_reconstruction_equals_value():
    base = _synth_base(n=900, seed=7, day_len=300)
    res = run_base_dp(base, KernelCounters(), symbol="SYNTH")
    total_val = sum(res["unit_values"])
    total_gross = sum(float(t["gross_points"]) for t in res["trades"])
    assert math.isclose(total_val, total_gross, rel_tol=1e-9, abs_tol=1e-6)
    for tr in res["trades"]:
        assert tr["holding_bars"] >= 1
        assert tr["entry_fill_index"] < tr["exit_fill_index"]
        assert tr["MFE"] >= tr["gross_points"] - 1e-9
        assert tr["MAE"] <= tr["gross_points"] + 1e-9
        assert math.isclose(
            tr["net_points"],
            tr["gross_points"] - tr["cost_points"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )


def test_dp_sign_symmetry():
    opens = np.array([100, 101, 103, 106, 110, 115, 121], float)
    mirror = 200.0 - opens
    entry = [True] * 5
    ca, _ba, _pa, s, e, *_ = _solve(opens, entry)
    cb, _bb, _pb, sb, eb, *_ = _solve(mirror, entry)
    pa = prod_path(ca, s, e)
    pb = prod_path(cb, sb, eb)
    assert pa == [-x for x in pb]
    assert pa[0] == 1 and pb[0] == -1


def test_dp_true_tie():
    opens = np.array([100.0] * 7)
    entry = [True] * 5
    core, best, paths, s, _e, *_ = _solve(opens, entry, end=5)
    # flat prices -> every action ties at value 0; tie prefers no turnover
    assert int(core["actions"][s, P2I[0]]) == 0
    assert bool(core["ambiguous"][s, 1]) is True
    assert best == 0.0
    assert (0, 0, 0, 0, 0) in paths


def test_cost_interface_active():
    # a large cost must suppress trading entirely (cost interface is wired)
    opens = np.array([100, 101, 103, 106, 110, 115, 121], float)
    entry = [True] * 5
    cost_big = np.full(len(opens), 1000.0)
    core = solve_day_dp(opens, np.asarray(entry, bool), cost_big, 0, 5)
    assert prod_path(core, 0, 5) == [0, 0, 0, 0, 0]
    # with zero cost a strictly rising series must trade
    core0 = solve_day_dp(opens, np.asarray(entry, bool), np.zeros(len(opens)), 0, 5)
    assert prod_path(core0, 0, 5)[0] == 1


# =========================================================================== #
# Artifact schema                                                              #
# =========================================================================== #
def test_artifact_schema():
    base = _synth_base(n=900, seed=8, disc_index=450, day_len=300)
    res = run_base_dp(base, KernelCounters(), symbol="SYNTH")
    frames = build_artifact_frames(res)
    action_cols = {
        "symbol",
        "trading_day",
        "decision_bar_index",
        "decision_time",
        "entry_eligible",
        "entry_source_bits",
        "Q_F_S",
        "Q_F_F",
        "Q_F_L",
        "best_flat_action",
        "flat_edge",
        "Q_L_S",
        "Q_L_F",
        "Q_L_L",
        "best_long_action",
        "long_edge",
        "Q_S_S",
        "Q_S_F",
        "Q_S_L",
        "best_short_action",
        "short_edge",
        "position_before",
        "position_after",
        "transition",
        "ambiguous",
        "label_available_time",
        "terminal_reason",
        "training_eligible",
    }
    trade_cols = {
        "trade_id",
        "symbol",
        "trading_day",
        "direction",
        "entry_decision_index",
        "entry_fill_index",
        "entry_fill_time",
        "entry_fill_price",
        "entry_source_bits",
        "exit_decision_index",
        "exit_fill_index",
        "exit_fill_time",
        "exit_fill_price",
        "holding_bars",
        "gross_points",
        "cost_points",
        "net_points",
        "MFE",
        "MAE",
        "terminal_reason",
        "training_eligible",
    }
    assert action_cols <= set(frames["oracle_actions"].columns)
    assert trade_cols <= set(frames["oracle_trades"].columns)

    meta = artifact_metadata("deadbeef", "SYNTH")
    for k in (
        "source_sha",
        "task_id",
        "math_version",
        "entry_semantics",
        "execution_semantics",
        "objective",
        "cost_mode",
    ):
        assert k in meta


def test_data_end_and_trading_day_end_reasons():
    base = _synth_base(n=900, seed=9, day_len=300)
    res = run_base_dp(base, KernelCounters(), symbol="SYNTH")
    reasons = [u["terminal_reason"] for u in res["units"]]
    assert reasons[-1] == DATA_END
    assert all(r in (TRADING_DAY_END, DISCONTINUITY) for r in reasons[:-1])
    assert res["units"][-1]["training_eligible"] is False
    assert all(u["training_eligible"] for u in res["units"][:-1])


# =========================================================================== #
# T1 — real-data: already-causal unit inputs sliced into short windows          #
# =========================================================================== #
@pytest.mark.parametrize("symbol", ["AG", "CU"])
def test_T1_real_data(symbol):
    N = 1500
    counters = KernelCounters()
    res = run_symbol_dp(symbol, counters, max_bars=N)

    assert counters.raw_load_count == 1
    assert counters.resample_count == 4
    assert counters.reference_call_count == 0
    assert counters.full_history_recompute_count == 0
    assert counters.concat_count == 0
    assert counters.dp_state_count > 0

    opens = res["open"]
    entry_ok = res["entry_eligible"]
    cost = res["cost_points"]
    n = res["n"]

    collected = {"val": 0, "path": 0, "state": 0}
    n_win = 0
    eligible_win = 0

    def _run_window(ws, we):
        nonlocal n_win, eligible_win
        D = we - ws
        os_ = opens[ws : we + 2]
        es_ = entry_ok[ws:we]
        cs_ = cost[ws:we]
        core = solve_day_dp(os_, es_, cs_, 0, D)
        best, paths = exhaustive_reference(os_, es_, cs_, 0, D)
        v = float(np.nanmax(core["Q"][0, P2I[0], :]))
        if abs(v - best) > 1e-6:
            collected["val"] += 1
        pp = tuple(int(core["actions"][t, P2I[p]]) for t, p in _walk_path(core, D))
        if pp not in paths:
            collected["path"] += 1
        for p0 in POS:
            b2, _ = exhaustive_reference(os_, es_, cs_, 0, D, start_pos=int(p0))
            vv = float(np.nanmax(core["Q"][0, P2I[int(p0)], :]))
            if abs(vv - b2) > 1e-6:
                collected["state"] += 1
        if bool(es_.any()):
            eligible_win += 1
        n_win += 1

    for u in res["units"]:
        s, e = u["seg_start"], u["seg_end"]
        if e - s < 12 or n_win >= 12:
            continue
        cap = min(e, n - 2)
        elig = [t for t in range(s, cap) if bool(entry_ok[t])]
        if not elig:
            continue
        step = max(1, len(elig) // 8)
        for idx in elig[::step]:
            if n_win >= 12:
                break
            D = 8
            ws = max(s, idx - 4)
            we = ws + D
            if we > cap:
                we = cap
                ws = we - D
            if ws < s or (we - ws) < 7 or we + 2 > n:
                continue
            _run_window(ws, we)

    assert n_win >= 5, f"{symbol}: too few windows ({n_win})"
    assert eligible_win >= 1, f"{symbol}: no eligible window"
    assert collected == {"val": 0, "path": 0, "state": 0}, collected

    # long-unit invariants
    td = pd.to_datetime(res["trading_day"]).to_numpy()
    seg = res["segment"]
    assert res["trades"]
    for tr in res["trades"]:
        ei, xi = int(tr["entry_fill_index"]), int(tr["exit_fill_index"])
        assert td[ei] == td[xi]
        assert seg[ei] == seg[xi]
        assert tr["entry_fill_index"] == tr["entry_decision_index"] + 1
        assert tr["exit_fill_index"] == tr["exit_decision_index"] + 1
    for u in res["units"]:
        e = u["seg_end"]
        if u["seg_end"] - u["seg_start"] < 2:
            continue
        assert int(res["decision"]["pos_after"][e - 1]) == 0

    total_val = sum(res["unit_values"])
    total_gross = sum(float(t["gross_points"]) for t in res["trades"])
    assert math.isclose(total_val, total_gross, rel_tol=1e-9, abs_tol=1e-6)


def _walk_path(core, D):
    p = 0
    for t in range(D):
        pi = P2I[p]
        a = int(core["actions"][t, pi])
        yield t, p
        p = a


# =========================================================================== #
# FIX1 — true mask-only fast path                                              #
# =========================================================================== #
def test_mask_only_differential_and_counters():
    """mask_only must be byte-identical to the normal mask path AND must not
    execute any of the (tf, role) event machinery."""
    base = _synth_base(n=900, seed=5, disc_index=450, day_len=300)

    c_norm = KernelCounters()
    r_norm = stream_from_base(
        base, c_norm, capture_entry_mask=True, emit_events=False, mask_only=False
    )
    c_fast = KernelCounters()
    r_fast = stream_from_base(
        base, c_fast, capture_entry_mask=True, emit_events=False, mask_only=True
    )

    # exact entry-mask equality
    assert np.array_equal(r_norm["entry_mask"], r_fast["entry_mask"])
    assert np.array_equal(r_norm["entry_eligible"], r_fast["entry_eligible"])

    # mask-only: no event machinery at all, no DTP context allocated
    assert c_fast.event_role_iteration_count == 0
    assert c_fast.event_classifier_call_count == 0
    assert c_fast.outcome_call_count == 0
    assert r_fast["dtp_ctx"] is None
    assert r_fast["atr5m"] is None
    assert r_fast["events"] == []

    # normal path (emit_events=False) still runs the role loop but no outcome
    assert c_norm.event_role_iteration_count > 0
    assert c_norm.event_classifier_call_count > 0
    assert c_norm.outcome_call_count == 0


def _raw_ag():
    raw = load_raw_5m("AG").sort_values("bar_start_time").reset_index(drop=True)
    disc = np.asarray(discontinuity_flags("AG"), dtype=bool)
    return (
        raw["bar_start_time"].to_numpy(),
        raw["trading_day"].to_numpy(),
        raw["open"].to_numpy(float),
        raw["high"].to_numpy(float),
        raw["low"].to_numpy(float),
        raw["close"].to_numpy(float),
        disc,
    )


# =========================================================================== #
# TP — TRUE N / 2N / 4N prefix benchmark (time and memory pass separated)       #
# =========================================================================== #
def test_TP_performance_gate():
    # raw load happens ONCE, outside every timed section
    T, D, O, H, L, C, disc = _raw_ag()
    sizes = [5000, 10000, 20000]

    # ---- TIME pass: tracemalloc OFF (min of reps: robust to load noise) --- #
    REPS = 2
    times = {}
    for N in sizes:
        best = float("inf")
        for _ in range(REPS):
            c = KernelCounters()
            res = run_arrays_dp(
                T[:N], D[:N], O[:N], H[:N], L[:N], C[:N], disc[:N], c, symbol="AG"
            )
            best = min(best, res["runtime_total_sec"])
            assert c.reference_call_count == 0
            assert c.full_history_recompute_count == 0
            assert c.concat_count == 0
            assert c.dp_state_count > 0
            # true mask-only path -> event machinery is never executed
            assert c.event_role_iteration_count == 0
            assert c.event_classifier_call_count == 0
            assert c.outcome_call_count == 0
        times[N] = best

    r1 = times[10000] / times[5000]
    r2 = times[20000] / times[10000]
    print(
        f"  TP-TIME  5k={times[5000]:.2f}s 10k={times[10000]:.2f}s "
        f"20k={times[20000]:.2f}s ratios={r1:.2f}/{r2:.2f}"
    )
    assert r1 < 2.8, f"5k->10k time scaling {r1} exceeds 2.8"
    assert r2 < 2.8, f"10k->20k time scaling {r2} exceeds 2.8"

    # ---- MEMORY pass: tracemalloc ON, separate run ------------------------ #
    peaks = {}
    for N in sizes:
        c = KernelCounters()
        res = run_arrays_dp(
            T[:N], D[:N], O[:N], H[:N], L[:N], C[:N], disc[:N], c,
            symbol="AG", profile_memory=True,
        )
        peaks[N] = res["peak_tracemalloc_mb"]

    m1 = peaks[10000] / peaks[5000]
    m2 = peaks[20000] / peaks[10000]
    print(
        f"  TP-MEM   5k={peaks[5000]:.2f}MB 10k={peaks[10000]:.2f}MB "
        f"20k={peaks[20000]:.2f}MB ratios={m1:.2f}/{m2:.2f}"
    )
    # approximately linear memory (loose sanity gate)
    assert peaks[20000] < 6.0 * peaks[5000] + 100.0
    assert m1 < 3.2 and m2 < 3.2, f"memory scaling {m1}/{m2} grew super-linearly"
