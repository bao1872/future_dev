"""Tests for the M1 / STRUCT44 cost-robustness L2 label kernel (Phase 1).

Task ID: FUTURE-ENTRY-VALUE-M1-STRUCT44-COST-ROBUSTNESS-V1

These tests lock the kernel before any formal 15x5 experiment:

T0  - kernel contract unit tests (cost accounting, ATR capture, candidate
      invariance, forward-ATR gate, performance-counters contract).
T1  - fast solver vs full DP path differential (synthetic + AU + RB).
PAR - 15-symbol kappa=0 exact parity vs frozen R2 Oracle.

Heavy (real-data) tests are marked ``slow`` so the default ``pytest`` run stays
fast; they are also exercised by ``run_cost_kernel_checkpoint``.
"""

import numpy as np
import pandas as pd
import pytest

from research.liquidity_oracle_atlas.experiment_entry_value_cost_robustness_v1 import (
    ACTION_F,
    ACTION_L,
    ACTION_S,
    F1_INDEX,
    KAPPAS,
    SYMBOLS_15,
    CostKernelStats,
    compare_q_arrays,
    make_cost_points,
    parity_against_r2,
    prepare_cost_oracle,
    run_cost_kernel_checkpoint,
    solve_cost_labels_fast,
    suffix_valid_atr_mask,
)
from research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1 import (
    KernelCounters,
    _stream_from_base,
    build_base_from_arrays,
)
from research.liquidity_oracle_atlas.build_structure_constrained_trade_oracle_dp_v2 import (
    build_artifact_frames_v2,
    run_symbol_dp_v2,
    solve_day_dp_v2,
)
from research.liquidity_oracle_atlas.build_structure_constrained_trade_oracle_dp_v1 import (
    build_intraday_units,
)

TOL = 1e-12


# --------------------------------------------------------------------------- #
# Synthetic helpers                                                            #
# --------------------------------------------------------------------------- #
def _synthetic_base(n: int = 30):
    """Minimal valid base frame (single trading day, no discontinuities)."""
    t0 = np.datetime64("2024-01-02T09:00:00")
    times = np.array([t0 + np.timedelta64(5 * i, "m") for i in range(n)])
    days = np.array(["2024-01-02"] * n, dtype="datetime64[D]")
    o = 1000.0 + np.arange(n, dtype=float)
    h = o + 1.0
    l = o - 1.0
    c = o + 0.5
    disc = np.zeros(n, dtype=bool)
    counters = KernelCounters()
    info = build_base_from_arrays(times, days, o, h, l, c, disc, counters)
    return info


def _random_base(n: int = 200, seed: int = 7):
    """Volatile synthetic base so the proximity stream actually fires."""
    rng = np.random.default_rng(seed)
    t0 = np.datetime64("2024-01-02T09:00:00")
    times = np.array([t0 + np.timedelta64(5 * i, "m") for i in range(n)])
    days = np.array(["2024-01-02"] * n, dtype="datetime64[D]")
    o = 1000.0 + np.cumsum(rng.normal(0, 2.0, size=n))
    h = o + np.abs(rng.normal(0, 1.0, size=n)) + 0.5
    l = o - np.abs(rng.normal(0, 1.0, size=n)) - 0.5
    c = o + rng.normal(0, 1.0, size=n)
    disc = np.zeros(n, dtype=bool)
    counters = KernelCounters()
    info = build_base_from_arrays(times, days, o, h, l, c, disc, counters)
    return info


# --------------------------------------------------------------------------- #
# T0 - kernel contract unit tests                                              #
# --------------------------------------------------------------------------- #
def test_t0_cost_vector_zero_at_kappa0():
    atr = np.array([1.0, 2.0, np.nan, 0.0, 5.0])
    cost = make_cost_points(atr, 0.0)
    assert np.allclose(cost, 0.0)


def test_t0_cost_uses_causal_atr():
    atr = np.array([1.0, np.nan, 0.0, 2.5, 4.0])
    kappa = 0.01
    cost = make_cost_points(atr, kappa)
    good = np.isfinite(atr) & (atr > 0)
    assert np.allclose(cost[good], kappa * atr[good])
    # NaN / non-positive ATR -> 0 (kept legal; forward-ATR gate rejects candidates)
    assert cost[~good].sum() == 0.0


def test_t0_bellman_turnover_costs():
    """Flat->dir=1c, dir->Flat=1c, reversal=2c, hold=0, terminal forced-flat."""
    # unit length 2: decisions 0 (non-terminal) and 1 (terminal forced-flat).
    opens = np.array([10.0, 10.0, 10.0, 10.0])  # delta -> 0 for simplicity
    proximity_any = np.array([True, False])
    c = 0.07
    cost = np.array([c, 0.0])
    Q = solve_day_dp_v2(opens, proximity_any, cost, 0, 2)["Q"][0]  # decision 0

    # state index: F1=3, L0=4, L1=5 (armed Long)
    # Flat state F1: F allowed (0), L/S allowed (proximity True) -> cost 1c each
    assert abs(Q[3, ACTION_F] - 0.0) < 1e-12
    assert abs(Q[3, ACTION_L] - (-c)) < 1e-12   # Flat->Long = 1c
    assert abs(Q[3, ACTION_S] - (-c)) < 1e-12   # Flat->Short = 1c
    # Armed-Long state L1: hold(1)=0c, Flat(0)=1c, reversal Short(-1)=2c.
    # (Reversal is only legal from an ARMED Long: L0 with armed=0 forbids new
    #  entries, so L0->S is -inf in the frozen Bellman.)
    assert abs(Q[5, ACTION_L] - 0.0) < 1e-12    # hold = 0c
    assert abs(Q[5, ACTION_F] - (-c)) < 1e-12   # Long->Flat = 1c
    assert abs(Q[5, ACTION_S] - (-2 * c)) < 1e-12  # reversal = 2c

    # terminal decision (t=1): only Flat allowed -> S/L must be -inf
    Q1 = solve_day_dp_v2(opens, proximity_any, cost, 0, 2)["Q"][1]
    assert np.isneginf(Q1[3, ACTION_S])
    assert np.isneginf(Q1[3, ACTION_L])
    assert np.isfinite(Q1[3, ACTION_F])


def test_t0_higher_kappa_does_not_increase_state_value():
    """V (and every finite Q) is non-increasing in kappa."""
    rng = np.random.default_rng(0)
    opens = 100.0 + np.cumsum(rng.normal(0, 0.5, size=40))
    proximity_any = rng.random(38) > 0.4
    atr = np.abs(rng.normal(1.0, 0.3, size=38)) + 0.1

    q_lo = solve_day_dp_v2(opens, proximity_any, make_cost_points(atr, 0.0), 0, 38)["Q"]
    q_hi = solve_day_dp_v2(opens, proximity_any, make_cost_points(atr, 0.05), 0, 38)["Q"]
    fin = np.isfinite(q_lo) & np.isfinite(q_hi)
    # higher friction can only decrease or leave Q unchanged
    assert np.all(q_hi[fin] <= q_lo[fin] + 1e-9)


@pytest.mark.slow
def test_t0_candidate_keys_invariant_across_kappa():
    """Candidate mask must be identical for every kappa (kappa-invariant)."""
    stats = CostKernelStats()
    counters = KernelCounters()
    ctx = prepare_cost_oracle("AU", counters, stats)
    masks = []
    for kappa in KAPPAS:
        df = solve_cost_labels_fast(ctx, kappa, stats)
        masks.append(df["is_candidate"].to_numpy(bool))
    for m in masks[1:]:
        assert np.array_equal(masks[0], m)


@pytest.mark.slow
def test_t0_forward_atr_gate_nonvacuous():
    """Suffix-valid mask must be non-trivial and cover all candidates."""
    stats = CostKernelStats()
    counters = KernelCounters()
    ctx = prepare_cost_oracle("AU", counters, stats)
    suffix_ok = suffix_valid_atr_mask(ctx.atr5m, ctx.starts, ctx.ends)
    assert suffix_ok.any() and (~suffix_ok).any()

    df = solve_cost_labels_fast(ctx, 0.0, stats)
    cand = df["is_candidate"].to_numpy(bool)
    cand_bar = df["decision_bar_index"].to_numpy(int)[cand]
    assert np.all(suffix_ok[cand_bar])


def test_t0_mask_only_atr_capture_leaves_event_counters_zero():
    info = _random_base(n=200)
    counters = KernelCounters()
    res = _stream_from_base(
        info["base"], info["form"], info["seg_completed"], counters,
        symbol="SYNTH", capture_geom=False, capture_entry_mask=False,
        emit_events=False, mask_only=True, capture_proximity=True,
        capture_atr5m=True,
    )
    assert counters.event_role_iteration_count == 0
    assert counters.event_classifier_call_count == 0
    assert counters.outcome_call_count == 0
    assert res["atr5m"] is not None
    assert np.isfinite(res["atr5m"]).any()


def test_t0_default_capture_atr5m_preserves_old_output():
    info = _random_base(n=200)
    c1, c2 = KernelCounters(), KernelCounters()
    res_off = _stream_from_base(
        info["base"], info["form"], info["seg_completed"], c1,
        symbol="SYNTH", capture_proximity=True, mask_only=True,
    )
    res_on = _stream_from_base(
        info["base"], info["form"], info["seg_completed"], c2,
        symbol="SYNTH", capture_proximity=True, mask_only=True,
        capture_atr5m=True,
    )
    # default path: atr5m is None (old behaviour)
    assert res_off["atr5m"] is None
    assert res_on["atr5m"] is not None
    # proximity outputs identical
    assert np.array_equal(res_off["proximity_bits"], res_on["proximity_bits"])
    assert np.array_equal(res_off["proximity_any"], res_on["proximity_any"])


# --------------------------------------------------------------------------- #
# T1 - fast vs full DP differential                                            #
# --------------------------------------------------------------------------- #
def _fast_vs_reference_report(symbol, kappa, counters, stats):
    ctx = prepare_cost_oracle(symbol, counters, stats)
    fast = solve_cost_labels_fast(ctx, kappa, stats)

    cost = make_cost_points(ctx.atr5m, kappa)
    ref = run_symbol_dp_v2(symbol, counters, cost_points=cost)
    ref_actions = build_artifact_frames_v2(ref)["oracle_actions"]

    merged = fast.merge(
        ref_actions[[
            "symbol", "decision_time", "decision_bar_index",
            "proximity_bits", "proximity_any", "training_eligible",
            "label_available_time", "Q_F1_S", "Q_F1_F", "Q_F1_L",
        ]],
        on=["symbol", "decision_time"], how="outer", indicator=True,
    )
    report = {"key_mismatch": int((merged["_merge"] != "both").sum())}
    for col in ("Q_F1_S", "Q_F1_F", "Q_F1_L"):
        ok, err, detail = compare_q_arrays(
            merged[f"{col}_x"].to_numpy(float),
            merged[f"{col}_y"].to_numpy(float), TOL,
        )
        report[col] = (ok, detail)
    for col in ("proximity_bits", "decision_bar_index"):
        report[col] = int((merged[f"{col}_x"].to_numpy() != merged[f"{col}_y"].to_numpy()).sum())
    for col in ("proximity_any", "training_eligible"):
        report[col] = int((merged[f"{col}_x"].to_numpy(bool) != merged[f"{col}_y"].to_numpy(bool)).sum())
    a = merged["label_available_time_x"].to_numpy("datetime64[ns]")
    b = merged["label_available_time_y"].to_numpy("datetime64[ns]")
    report["label_available_time"] = int(((a != b) & ~(np.isnat(a) & np.isnat(b))).sum())
    return report


def test_t1_synthetic_differential():
    """Fast solver must equal the full DP on a synthetic base at kappa=0.01."""
    info = _random_base(n=200)
    rng = np.random.default_rng(1)
    # re-randomize OHLC a bit so there is genuine structure
    base = info["base"]
    counters = KernelCounters()
    stats = CostKernelStats()
    ctx = prepare_cost_oracle_from_base(info, counters, stats)
    fast = solve_cost_labels_fast(ctx, 0.01, stats)

    cost = make_cost_points(ctx.atr5m, 0.01)
    ref = run_arrays_dp_v2_from_info(info, counters, cost)
    ref_actions = build_artifact_frames_v2(ref)["oracle_actions"]

    merged = fast.merge(
        ref_actions[[
            "symbol", "decision_time", "decision_bar_index",
            "proximity_bits", "proximity_any", "training_eligible",
            "label_available_time", "Q_F1_S", "Q_F1_F", "Q_F1_L",
        ]],
        on=["symbol", "decision_time"], how="outer", indicator=True,
    )
    assert (merged["_merge"] == "both").all()
    for col in ("Q_F1_S", "Q_F1_F", "Q_F1_L"):
        ok, err, _ = compare_q_arrays(
            merged[f"{col}_x"].to_numpy(float),
            merged[f"{col}_y"].to_numpy(float), TOL,
        )
        assert ok, f"{col} max abs err {err}"
    for col in ("proximity_bits", "decision_bar_index", "proximity_any",
                "training_eligible", "label_available_time"):
        if col == "label_available_time":
            a = merged["label_available_time_x"].to_numpy("datetime64[ns]")
            b = merged["label_available_time_y"].to_numpy("datetime64[ns]")
            assert ((a != b) & ~(np.isnat(a) & np.isnat(b))).sum() == 0
        else:
            assert (merged[f"{col}_x"].to_numpy() == merged[f"{col}_y"].to_numpy()).all()


def test_t1_au_rb_differential():
    """Fast vs full DP on real AU/RB at kappa in {0, 0.01, 0.05}."""
    for symbol in ("AU", "RB"):
        for kappa in (0.0, 0.01, 0.05):
            stats = CostKernelStats()
            counters = KernelCounters()
            rep = _fast_vs_reference_report(symbol, kappa, counters, stats)
            assert rep["key_mismatch"] == 0, (symbol, kappa, rep)
            for col in ("Q_F1_S", "Q_F1_F", "Q_F1_L"):
                assert rep[col][0], (symbol, kappa, col, rep[col])
            assert rep["proximity_bits"] == 0
            assert rep["proximity_any"] == 0
            assert rep["training_eligible"] == 0
            assert rep["label_available_time"] == 0


# --------------------------------------------------------------------------- #
# Helper used by T1 synthetic (prepare / reference from a synthetic base)       #
# --------------------------------------------------------------------------- #
def prepare_cost_oracle_from_base(info, counters, stats):
    from research.liquidity_oracle_atlas.experiment_entry_value_cost_robustness_v1 import (
        PreparedCostOracleV1,
    )
    stats.prepare_count += 1
    stats.stream_count += 1
    base = info["base"]
    res = _stream_from_base(
        base, info["form"], info["seg_completed"], counters,
        symbol="SYNTH", capture_geom=False, capture_entry_mask=False,
        emit_events=False, mask_only=True, capture_proximity=True,
        capture_atr5m=True,
    )
    td = pd.to_datetime(base["trading_day"]).to_numpy()
    seg = base["segment"].to_numpy(np.int64)
    starts, ends = build_intraday_units(td, seg)
    from research.liquidity_oracle_atlas.build_structure_constrained_trade_oracle_dp_v2 import (
        compute_proximity_episode_id,
    )
    prox_ep = compute_proximity_episode_id(
        np.asarray(res["proximity_any"], bool), starts, len(base)
    )
    return PreparedCostOracleV1(
        symbol="SYNTH", n=int(len(base)),
        base=base,
        proximity_bits=np.asarray(res["proximity_bits"], np.int64),
        proximity_any=np.asarray(res["proximity_any"], bool),
        atr5m=np.asarray(res["atr5m"], float),
        starts=np.asarray(starts, np.int64),
        ends=np.asarray(ends, np.int64),
        proximity_episode_id=prox_ep,
    )


def run_arrays_dp_v2_from_info(info, counters, cost):
    from research.liquidity_oracle_atlas.build_structure_constrained_trade_oracle_dp_v2 import (
        run_arrays_dp_v2,
    )
    base = info["base"]
    return run_arrays_dp_v2(
        base["time"].to_numpy(), base["trading_day"].to_numpy(),
        base["open"].to_numpy(float), base["high"].to_numpy(float),
        base["low"].to_numpy(float), base["close"].to_numpy(float),
        base["discontinuity"].to_numpy(bool) if "discontinuity" in base
        else base["disc"].to_numpy(bool),
        counters, symbol="SYNTH", cost_points=cost,
    )


# --------------------------------------------------------------------------- #
# PAR - 15-symbol kappa=0 exact parity vs frozen R2 Oracle (slow)               #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_parity_15_symbols_kappa0():
    for sym in SYMBOLS_15:
        stats = CostKernelStats()
        counters = KernelCounters()
        rep = parity_against_r2(sym, stats, counters, atol=TOL)
        assert rep["ok"], (sym, rep)
        assert rep["key_mismatch"] == 0


@pytest.mark.slow
def test_performance_gate_au_rb():
    """prepare once + 5 DP passes; canonical event counters stay 0."""
    for sym in ("AU", "RB"):
        stats = CostKernelStats()
        counters = KernelCounters()
        ctx = prepare_cost_oracle(sym, counters, stats)
        for kappa in KAPPAS:
            solve_cost_labels_fast(ctx, kappa, stats)
        assert stats.prepare_count == 1
        assert stats.stream_count == 1
        assert stats.dp_pass_count == len(KAPPAS)
        assert counters.raw_load_count == 1
        assert counters.resample_count == 4
        assert counters.full_history_recompute_count == 0
        assert counters.event_role_iteration_count == 0
        assert counters.event_classifier_call_count == 0
        assert counters.outcome_call_count == 0


@pytest.mark.slow
def test_checkpoint_artifacts_written():
    summary = run_cost_kernel_checkpoint()
    assert summary["parity_all_ok"]
    assert summary["forward_atr_gate"] == "PASS"
    assert summary["performance_gate"] == "PASS"
