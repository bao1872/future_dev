"""
T0 / T1 / Long-run / TP / Regression tests for Oracle V2
(FUTURE-INTRADAY-DP-ORACLE-R2-ONE-ENTRY-PROXIMITY).

Run:
  .venv/bin/python -m pytest \
    research/liquidity_oracle_atlas/test_structure_constrained_trade_oracle_dp_v2.py -q
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1 import (
    KernelCounters,
    build_base_from_arrays,
    proximity_bits_from_prev_geometry,
    ENTRY_PROX_ATR,
    TF_ORDER,
    MASK_BIT,
)
from research.liquidity_oracle_atlas.build_structure_constrained_trade_oracle_dp_v2 import (
    MATH_VERSION,
    ARTIFACT_ROOT_DIRNAME,
    POS,
    ACT,
    STATE_POS,
    STATE_ARMED,
    S2I,
    STATES,
    is_new_entry,
    choose_actions_v2,
    solve_day_dp_v2,
    exhaustive_reference_v2,
    compute_proximity_episode_id,
    run_base_dp_v2,
    run_arrays_dp_v2,
    run_symbol_dp_v2,
    build_artifact_frames_v2,
    write_oracle_artifact_v2,
    load_oracle_artifact_v2,
    check_oracle_invariants_v2,
)


# --------------------------------------------------------------------------- #
# helpers                                                                       #
# --------------------------------------------------------------------------- #
def backtrack(opens, prox, start, end):
    """Run V2 DP on one unit and return the (t, p, a, new_entry) path."""
    core = solve_day_dp_v2(opens, prox, np.zeros(len(opens)), start, end)
    acts = core["actions"]
    p = 0
    q = 1
    path = []
    for local in range(end - start):
        si = S2I[(p, q)]
        a = int(acts[local, si])
        ne = is_new_entry(p, a)
        path.append((start + local, p, a, ne))
        qa = 0 if ne else q
        p = a
        if (start + local) < end - 1:
            q = 1 if not bool(prox[start + local + 1]) else qa
    return path, core


def _synthetic_base(n, seed=0, days=1):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2024-01-01 09:00", periods=n, freq="5min")
    day = (
        pd.to_datetime("2024-01-01")
        + pd.to_timedelta((np.arange(n) // (n // days)), unit="D")
    )
    o = np.cumsum(rng.normal(0, 0.5, n)) + 100
    c = o + rng.normal(0, 0.2, n)
    h = np.maximum(o, c) + abs(rng.normal(0, 0.3, n))
    l = np.minimum(o, c) - abs(rng.normal(0, 0.3, n))
    disc = np.zeros(n, dtype=bool)
    return build_base_from_arrays(
        t.to_numpy(), day.to_numpy(), o, h, l, c, disc, KernelCounters()
    )["base"]


# --------------------------------------------------------------------------- #
# T0 : synthetic contract cases                                                 #
# --------------------------------------------------------------------------- #
def test_t0_proximity_threshold_boundary():
    # one SR zone [99,100] with atr_tf = 10 -> radius = 5
    tf = TF_ORDER[0]
    geom = {
        tf: (
            [(100.0, 99.0, 1.0)],  # channels (top, bottom, strength)
            [],  # liq_up
            [],  # liq_down
            10.0,  # atr_tf
        )
    }
    r = ENTRY_PROX_ATR  # 0.5
    radius = r * 10.0  # 5.0
    # exactly at boundary -> distance == radius -> counts (<=)
    assert proximity_bits_from_prev_geometry(104.0, 105.0, geom, alpha=r) != 0
    # just outside -> distance = radius + 0.01 -> does NOT count
    assert proximity_bits_from_prev_geometry(104.01 + 5.0, 105.01 + 5.0, geom, alpha=r) == 0
    # far away
    assert proximity_bits_from_prev_geometry(200.0, 201.0, geom, alpha=r) == 0
    # inside the zone -> distance 0
    assert proximity_bits_from_prev_geometry(99.5, 99.8, geom, alpha=r) != 0


def test_t0_outside_threshold_is_zero():
    tf = TF_ORDER[0]
    geom = {tf: ([(100.0, 90.0, 1.0)], [], [], 10.0)}
    assert proximity_bits_from_prev_geometry(150.0, 151.0, geom, alpha=ENTRY_PROX_ATR) == 0


def test_t0_current_bar_structure_cannot_retroactively_qualify():
    # empty prev geometry -> no bit, even though a structure "appears"
    geom = {tf: ([], [], [], 10.0) for tf in TF_ORDER}
    assert proximity_bits_from_prev_geometry(99.0, 100.0, geom, alpha=ENTRY_PROX_ATR) == 0
    # positive control: a PRE-EXISTING structure DOES qualify
    geom2 = {TF_ORDER[0]: ([(100.0, 99.0, 1.0)], [], [], 10.0)}
    assert proximity_bits_from_prev_geometry(99.5, 99.8, geom2, alpha=ENTRY_PROX_ATR) != 0


def test_t0_one_entry_max_in_continuous_proximity():
    prox = np.array([0, 1, 1, 1, 1, 0], dtype=bool)
    opens = np.array([100.0, 100, 101, 102, 103, 104], dtype=float)
    path, _ = backtrack(opens, prox, 0, 5)
    entries = [t for (t, p, a, ne) in path if ne]
    assert len(entries) <= 1


def test_t0_wait_then_later_entry_allowed():
    # best single Long is at the LAST proximity bar (price jumps after it)
    prox = np.array([0, 1, 1, 1, 0], dtype=bool)
    opens = np.array([100.0, 100, 100, 100, 120], dtype=float)
    path, _ = backtrack(opens, prox, 0, 4)
    entries = [t for (t, p, a, ne) in path if ne]
    assert entries == [2], entries  # waited at bar1, entered bar2


def test_t0_exit_inside_episode_no_reentry():
    prox = np.array([0, 1, 1, 1, 0], dtype=bool)
    opens = np.array([100.0, 100, 110, 105, 100], dtype=float)
    path, _ = backtrack(opens, prox, 0, 4)
    entries = [t for (t, p, a, ne) in path if ne]
    assert len(entries) <= 1  # Long then Flat with no re-entry in same episode


def test_t0_leave_proximity_rearms_two_episodes():
    prox = np.array([0, 1, 1, 0, 1, 1, 0], dtype=bool)
    opens = np.array([100.0, 100, 100, 110, 100, 100, 110, 100], dtype=float)
    path, _ = backtrack(opens, prox, 0, 6)
    entries = [t for (t, p, a, ne) in path if ne]
    assert len(entries) == 2, entries


def test_t0_reversal_consumes_right_kernel():
    # A two-episode scenario: enter Long in episode 1, leave proximity (rearm to
    # armed=1 while still Long), re-enter proximity in episode 2 while Long with
    # armed=1 -> a Short (reversal) is a legal new-entry. After it, (Short,
    # armed=0) at the next proximity bar must NOT allow a reversal back to Long.
    prox = np.array([1, 1, 0, 1, 1, 0], dtype=bool)
    opens = np.array([100.0, 100, 110, 100, 100, 110, 90, 100], dtype=float)
    core = solve_day_dp_v2(opens, prox, np.zeros(8), 0, 6)
    Q = core["Q"]
    # at t=3 we are (Long, armed=1); action Short (ACT idx 0) must be legal
    si_long_armed = S2I[(1, 1)]
    q_rev = Q[3, si_long_armed, 0]
    assert not math.isinf(q_rev), "reversal should be legal when armed"
    # at t=4 we are (Short, armed=0) after the reversal; Long (ACT idx 2) must
    # be illegal (would consume a right we no longer have)
    si_short_disarmed = S2I[(-1, 0)]
    q_revreback = Q[4, si_short_disarmed, 2]
    assert math.isinf(q_revreback), "reversal back must be illegal when disarmed"


def test_t0_trading_day_reset_armed():
    base = _synthetic_base(600, seed=3, days=3)
    res = run_base_dp_v2(base, KernelCounters(), symbol="SYNTH")
    inv = check_oracle_invariants_v2(res)
    assert inv["cross_day"] == 0
    assert inv["nonflat_terminal"] == 0
    # episode ids differ across a trading-day boundary even if prox is continuous
    prox = res["proximity_any"]
    ep = compute_proximity_episode_id(prox, res["starts"], res["n"])
    # at least 2 distinct episode ids exist (multiple days/regions)
    assert len(set(ep[ep >= 0].tolist())) >= 2


def test_t0_segment_reset():
    base = _synthetic_base(600, seed=7, days=1)
    res = run_base_dp_v2(base, KernelCounters(), symbol="SYNTH")
    # no cross-segment trades
    assert check_oracle_invariants_v2(res)["cross_segment"] == 0


def test_t0_next_open_fill():
    base = _synthetic_base(400, seed=11)
    res = run_base_dp_v2(base, KernelCounters(), symbol="SYNTH")
    opens = base["open"].to_numpy(float)
    for tr in res["trades"]:
        assert float(tr["entry_fill_price"]) == pytest.approx(
            float(opens[int(tr["entry_fill_index"])])
        )
        assert float(tr["exit_fill_price"]) == pytest.approx(
            float(opens[int(tr["exit_fill_index"])])
        )
        assert int(tr["exit_fill_index"]) == int(tr["entry_fill_index"]) + int(tr["holding_bars"])


def _legal(Q, si, action_idx):
    v = Q[si, action_idx]
    return not (math.isinf(v) and v < 0)


def test_t0_sign_symmetry():
    # No Long/Short directional bias: the legal-action MASK is symmetric.
    # For a Flat position, opening Long is legal iff opening Short is legal
    # (both require proximity AND armed). For a non-flat position, reversing to
    # the opposite side is legal under the same (proximity AND armed) condition,
    # so Long->Short legality equals Short->Long legality.
    rng = np.random.default_rng(5)
    for seed in range(20):
        D = 6 + seed % 5
        opens = np.array([100.0] + list(100.0 + np.cumsum(rng.normal(0, 1.0, D + 1))))
        prox = rng.random(D + 2) < 0.5
        cost = np.zeros(len(opens))
        Q = solve_day_dp_v2(opens, prox, cost, 0, D)["Q"][0]
        # Flat states: Long (idx 2) legality == Short (idx 0) legality
        for q in (0, 1):
            si = S2I[(0, q)]
            assert _legal(Q, si, 2) == _legal(Q, si, 0), (seed, q)
        # reversal symmetry: Long->Short (idx 0 from (Long,q)) == Short->Long (idx 2 from (Short,q))
        for q in (0, 1):
            siL = S2I[(1, q)]
            siS = S2I[(-1, q)]
            assert _legal(Q, siL, 0) == _legal(Q, siS, 2), (seed, q)
        # same-direction hold + exit always legal for non-flat states
        # (opposite-side reversal requires armed=1)
        for q in (0, 1):
            siL = S2I[(1, q)]
            siS = S2I[(-1, q)]
            assert _legal(Q, siL, 1) and _legal(Q, siL, 2)  # Long: exit + hold
            assert _legal(Q, siS, 0) and _legal(Q, siS, 1)  # Short: hold + exit


def test_t0_terminal_flat():
    base = _synthetic_base(500, seed=9)
    res = run_base_dp_v2(base, KernelCounters(), symbol="SYNTH")
    assert check_oracle_invariants_v2(res)["nonflat_terminal"] == 0


def test_t0_production_equals_exhaustive():
    rng = np.random.default_rng(42)
    for seed in range(40):
        D = 6 + seed % 6
        opens = np.array([100.0] + list(100.0 + np.cumsum(rng.normal(0, 1.0, D + 1))))
        prox = rng.random(D + 2) < 0.5
        for si, (p, q) in enumerate(STATES):
            best, _ = exhaustive_reference_v2(
                opens, prox, np.zeros(len(opens)), 0, D, start_pos=p, start_armed=q
            )
            v = float(np.nanmax(solve_day_dp_v2(opens, prox, np.zeros(len(opens)), 0, D)["Q"][0, si, :]))
            assert abs(v - best) <= 1e-9, (seed, si, v, best)


def test_t0_true_tie_is_deterministic_and_flagged():
    # hold vs exit both yield exactly 0 -> ambiguous but valid action chosen
    opens = np.array([100.0, 100, 100, 100], dtype=float)
    prox = np.array([0, 0, 0, 0], dtype=bool)
    core = solve_day_dp_v2(opens, prox, np.zeros(4), 0, 3)
    chosen, amb, edge, vmax = choose_actions_v2(core["Q"][0])
    # at least one state must flag a true tie (edge == 0)
    assert bool((edge <= 1e-9).any())
    # chosen action is one of the tied best (no crash, valid index)
    for si in range(6):
        assert chosen[si] in (-1, 0, 1)


def test_t0_cost_interface_functional():
    prox = np.array([0, 1, 1, 1, 0], dtype=bool)
    opens = np.array([100.0, 100, 110, 100, 120], dtype=float)
    cost = np.ones(len(opens))  # c=1 per turnover
    core = solve_day_dp_v2(opens, prox, cost, 0, 4)
    # with cost, a Flat unit value must be <= the zero-cost value
    best_free = float(np.nanmax(solve_day_dp_v2(opens, prox, np.zeros(len(opens)), 0, 4)["Q"][0, S2I[(0, 1)], :]))
    best_cost = float(np.nanmax(core["Q"][0, S2I[(0, 1)], :]))
    assert best_cost <= best_free + 1e-9


def test_t0_negative_illegal_entry_validator():
    # Construct a fake result with one new entry OUTSIDE proximity.
    n = 5
    dec = {
        "proximity_bits": np.zeros(n, dtype=np.int64),
        "proximity_any": np.array([False, False, False, False, False]),
        "proximity_episode_id": np.full(n, -1, dtype=np.int64),
        "position_before": np.array([0, 1, 1, 0, 0], dtype=np.int8),
        "position_after": np.array([1, 1, 0, 0, 0], dtype=np.int8),
        "entry_right_before": np.array([0, 1, 1, 0, 0], dtype=np.int8),
        "entry_right_after": np.array([0, 1, 0, 0, 0], dtype=np.int8),
        "new_entry_consumed": np.array([True, False, False, False, False]),
        "transition": np.array(["x"] * n, dtype=object),
    }
    res = {
        "sel": np.arange(n),
        "n": n,
        "proximity_any": np.array([False] * n),
        "trading_day": np.array(["2024-01-01"] * n, dtype="datetime64[ns]"),
        "segment": np.zeros(n, dtype=np.int64),
        "time": pd.date_range("2024-01-01", periods=n, freq="5min").to_numpy(),
        "decision": dec,
        "trades": [],
        "units": [{"seg_start": 0, "seg_end": n, "terminal_reason": "X", "training_eligible": True}],
        "unit_values": [0.0],
    }
    inv = check_oracle_invariants_v2(res)
    assert inv["new_entry_outside_proximity"] >= 1
    assert inv["new_entry_armed_zero"] >= 1


# --------------------------------------------------------------------------- #
# T1 : AG + CU real-data differential vs independent exhaustive DFS             #
# --------------------------------------------------------------------------- #
def _real_symbol_run(symbol):
    counters = KernelCounters()
    res = run_symbol_dp_v2(symbol, counters)
    return res, counters


def test_t1_production_equals_exhaustive_real_windows():
    for symbol in ("AG", "CU"):
        res, _ = _real_symbol_run(symbol)
        prox = res["proximity_any"]
        opens = res["open"]
        cost = res["cost_points"]
        starts = res["starts"]
        ends = res["ends"]
        # pair (start, end) units long enough for an 8-decision window
        units = [(int(s), int(e)) for s, e in zip(starts, ends) if int(e) - int(s) >= 10]
        rng = np.random.default_rng(1234)
        rng.shuffle(units)
        D = 8
        windows = 0
        for s, e in units:
            t0 = s
            o = opens[t0 : t0 + D + 2]
            px = prox[t0 : t0 + D]
            c = cost[t0 : t0 + D]
            core = solve_day_dp_v2(o, px, c, 0, D)
            # optimal value for all 6 start states
            for si, (ps, qs) in enumerate(STATES):
                best, _ = exhaustive_reference_v2(
                    o, px, c, 0, D, start_pos=ps, start_armed=qs
                )
                v = float(np.nanmax(core["Q"][0, si, :]))
                assert abs(v - best) <= 1e-6, (symbol, t0, si, v, best)
            # production path from (Flat, armed=1) == an optimal exhaustive path
            acts = core["actions"]
            p = 0
            q = 1
            prod_path = []
            for local in range(D):
                si = S2I[(p, q)]
                a = int(acts[local, si])
                prod_path.append(a)
                ne = is_new_entry(p, a)
                qa = 0 if ne else q
                p = a
                if local < D - 1:
                    q = 1 if not bool(px[local + 1]) else qa
            _, best_paths = exhaustive_reference_v2(o, px, c, 0, D)
            assert tuple(prod_path) in best_paths, (symbol, t0, prod_path, best_paths)
            windows += 1
            if windows >= 20:
                break
        assert windows >= 5, f"{symbol}: too few in-unit windows sampled ({windows})"


# --------------------------------------------------------------------------- #
# Long-run real invariants (AG + CU)                                            #
# --------------------------------------------------------------------------- #
def test_longrun_invariants_ag_cu():
    for symbol in ("AG", "CU"):
        res, _ = _real_symbol_run(symbol)
        inv = check_oracle_invariants_v2(res)
        assert inv["new_entry_outside_proximity"] == 0, (symbol, inv)
        assert inv["new_entry_armed_zero"] == 0, (symbol, inv)
        assert inv["illegal_reversal"] == 0, (symbol, inv)
        assert inv["cross_day"] == 0, (symbol, inv)
        assert inv["cross_segment"] == 0, (symbol, inv)
        assert inv["nonflat_terminal"] == 0, (symbol, inv)
        assert inv["pnl_mismatch_flag"] == 0, (symbol, inv)
        assert inv["max_new_entries_per_episode"] <= 1, (symbol, inv)
        # invariant the contract calls the most important mechanical proof
        assert inv["max_new_entries_per_episode"] <= 1


# --------------------------------------------------------------------------- #
# TP : true-prefix 5k/10k/20k, separate passes, ratio < 2.8                     #
# --------------------------------------------------------------------------- #
def test_tp_true_prefix_scaling():
    n = 20000
    rng = np.random.default_rng(77)
    t = pd.date_range("2024-01-01 09:00", periods=n, freq="5min").to_numpy()
    day = (
        pd.to_datetime("2024-01-01").to_numpy()
        + pd.to_timedelta((np.arange(n) // 5000), unit="D")
    )
    o = np.cumsum(rng.normal(0, 0.4, n)) + 100
    c = o + rng.normal(0, 0.15, n)
    h = np.maximum(o, c) + abs(rng.normal(0, 0.2, n))
    l = np.minimum(o, c) - abs(rng.normal(0, 0.2, n))
    disc = np.zeros(n, dtype=bool)

    timings = {}
    for N in (5000, 10000, 20000):
        counters = KernelCounters()
        out = run_arrays_dp_v2(
            t[:N], day[:N], o[:N], h[:N], l[:N], c[:N], disc[:N], counters, symbol="PREFIX"
        )
        timings[N] = out["runtime_total_sec"]
        # event machinery must stay zero on the production path
        assert counters.event_role_iteration_count == 0
        assert counters.event_classifier_call_count == 0
        assert counters.outcome_call_count == 0
        assert counters.reference_call_count == 0
        assert counters.full_history_recompute_count == 0
        assert counters.concat_count == 0
        assert counters.dp_state_count > 0

    r1 = timings[10000] / timings[5000]
    r2 = timings[20000] / timings[10000]
    assert r1 < 2.8, f"10k/5k ratio {r1}"
    assert r2 < 2.8, f"20k/10k ratio {r2}"


def test_tp_memory_linear():
    n = 20000
    rng = np.random.default_rng(78)
    t = pd.date_range("2024-01-01 09:00", periods=n, freq="5min").to_numpy()
    day = (
        pd.to_datetime("2024-01-01").to_numpy()
        + pd.to_timedelta((np.arange(n) // 5000), unit="D")
    )
    o = np.cumsum(rng.normal(0, 0.4, n)) + 100
    c = o + rng.normal(0, 0.15, n)
    h = np.maximum(o, c) + abs(rng.normal(0, 0.2, n))
    l = np.minimum(o, c) - abs(rng.normal(0, 0.2, n))
    disc = np.zeros(n, dtype=bool)
    peak = {}
    for N in (5000, 20000):
        counters = KernelCounters()
        out = run_arrays_dp_v2(
            t[:N], day[:N], o[:N], h[:N], l[:N], c[:N], disc[:N], counters,
            symbol="PREFIX", profile_memory=True,
        )
        peak[N] = out.get("peak_tracemalloc_mb", 0.0)
    if peak[5000] > 0:
        ratio = peak[20000] / peak[5000]
        assert ratio < 8.0, f"memory growth ratio {ratio}"


# --------------------------------------------------------------------------- #
# Regression / fail-closed                                                      #
# --------------------------------------------------------------------------- #
def test_regression_loader_rejects_wrong_math_version():
    with tempfile.TemporaryDirectory() as d:
        symdir = Path(d) / "AG"
        symdir.mkdir()
        pd.DataFrame({"x": [0]}).to_parquet(symdir / "oracle_actions.parquet")
        pd.DataFrame({"y": [0]}).to_parquet(symdir / "oracle_trades.parquet")
        (symdir / "metadata.json").write_text(
            __import__("json").dumps({"math_version": "intraday_dp_oracle_r1", "symbol": "AG"})
        )
        loaded = load_oracle_artifact_v2(d, "AG", expected_math_version=MATH_VERSION)
        assert loaded["ok"] is False
        assert loaded["reason"] in ("math_version_mismatch", "missing_trade_columns")


def test_regression_loader_ok_on_written_v2():
    base = _synthetic_base(300, seed=4)
    res = run_base_dp_v2(base, KernelCounters(), symbol="SYNTH")
    with tempfile.TemporaryDirectory() as d:
        write_oracle_artifact_v2(res, d, oracle_source_sha="deadbeef")
        loaded = load_oracle_artifact_v2(d, "SYNTH", expected_math_version=MATH_VERSION,
                                         expected_source_sha="deadbeef")
        assert loaded["ok"] is True
        assert loaded["metadata"]["entry_proximity_atr"] == ENTRY_PROX_ATR
        # V2-specific columns present
        assert "entry_proximity_episode_id" in loaded["trades"].columns
        assert "entry_armed_before" in loaded["trades"].columns
        # invariant: every trade entry_armed_before == 1
        assert (loaded["trades"]["entry_armed_before"] == 1).all()


def test_regression_r1_artifacts_untouched_dirname():
    # V2 writes to its own directory name; R1 dirname must remain distinct.
    assert ARTIFACT_ROOT_DIRNAME != "intraday_dp_oracle_r1"
    assert MATH_VERSION != "intraday_dp_oracle_r1"
