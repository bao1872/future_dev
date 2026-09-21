"""
test_structure_constrained_trade_oracle_dp_v1
=============================================

T0 gates + T1 differential + TP performance gate for the Structure-Constrained
Trade Oracle DP V1 (task FUTURE-STRATEGY-DP-ORACLE-V1).

Contract covered:
  * allowed-action set + entry/reversal gating,
  * next-open execution (decision t -> fill t+1),
  * per-segment natural terminal (flat start/end, no cross-discontinuity),
  * primary objective = gross open-to-open PnL (c_roundtrip_atr = 0),
  * independent brute-force reference equivalence,
  * T1 real-data differential MUST slice already-causal DP inputs from the
    full-history streaming run (it MUST NOT rebuild SR/Liquidity/DTP from the
    short window); the short-window terminal is TEST-ONLY.

Run with the project interpreter (Python 3.11+):
    .venv/bin/python -m pytest research/liquidity_oracle_atlas/test_structure_constrained_trade_oracle_dp_v1.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1 import (
    KernelCounters,
    build_base_from_arrays,
    stream_from_base,
)
from research.liquidity_oracle_atlas.build_structure_constrained_trade_oracle_dp_v1 import (
    DATA_END,
    DISCONTINUITY,
    INVALID_ACTION,
    P2I,
    POS,
    _allowed_actions,
    _ref_allowed,
    _transition_label,
    artifact_metadata,
    backtrack_segment,
    build_artifact_frames,
    reference_solve_segment,
    run_base_dp,
    run_symbol_dp,
    solve_segment_dp,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def run_seg(opens, entry_ok, atr5=None, seg_start=0, seg_end=None):
    opens = np.asarray(opens, float)
    if seg_end is None:
        seg_end = len(opens) - 2
    if atr5 is None:
        atr5 = np.ones(len(opens))
    else:
        atr5 = np.asarray(atr5, float)
    entry_ok = np.asarray(entry_ok, bool)
    core = solve_segment_dp(opens, atr5, entry_ok, seg_start, seg_end)
    ref = reference_solve_segment(opens, atr5, entry_ok, seg_start, seg_end)
    return core, ref, seg_start, seg_end


def prod_path(core, s, e):
    p = 0
    out = []
    for t in range(s, e):
        q = int(core["action"][t, P2I[p]])
        out.append(q)
        p = q
    return out


def ref_path(ref, s, e):
    p = 0
    out = []
    for t in range(s, e):
        q = int(ref["chosen"][(t, p)])
        out.append(q)
        p = q
    return out


def _synth_ohlc(n, seed=0, disc_index=None):
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
    day = pd.to_datetime(["2024-01-01"] * n)
    disc = np.zeros(n, dtype=bool)
    if disc_index is not None:
        disc[disc_index] = True
    return t, day, o, h, l, c, disc


def _synth_base(n=400, seed=0, disc_index=None):
    t, day, o, h, l, c, disc = _synth_ohlc(n, seed=seed, disc_index=disc_index)
    return build_base_from_arrays(t, day, o, h, l, c, disc, KernelCounters())["base"]


def assert_dp_matches_reference(core, ref, s, e, collected):
    """Accumulate value/chosen/ambiguous mismatches into ``collected``."""
    for t in range(s, e):
        for p in POS:
            pi = P2I[int(p)]
            pv = float(core["value"][t, pi])
            rv = float(ref["value"][(t, int(p))])
            if not math.isclose(pv, rv, rel_tol=1e-9, abs_tol=1e-6):
                collected["value"] += 1
            if int(core["action"][t, pi]) != int(ref["chosen"][(t, int(p))]):
                collected["chosen"] += 1
            if bool(core["ambiguous"][t, pi]) != bool(ref["ambiguous"][(t, int(p))]):
                collected["ambiguous"] += 1
            # best-action-set consistency
            acts = ref["best_actions"][(t, int(p))]
            if bool(core["ambiguous"][t, pi]):
                if len(acts) <= 1 or int(core["action"][t, pi]) not in acts:
                    collected["best_set"] += 1
            else:
                if len(acts) != 1 or int(core["action"][t, pi]) not in acts:
                    collected["best_set"] += 1


def invariant_violations(action_rows):
    """Return a list of illegal-entry / illegal-reversal violations."""
    bad = []
    for r in action_rows:
        pb = int(r["oracle_position_before"])
        pa = int(r["oracle_position_after"])
        eligible = bool(r["entry_eligible"])
        if pb == 0 and pa != 0 and not eligible:
            bad.append(("illegal_entry", r["decision_bar_index"]))
        if pb != 0 and pa == -pb and not eligible:
            bad.append(("illegal_reversal", r["decision_bar_index"]))
    return bad


# =========================================================================== #
# T0 gates (§24)                                                               #
# =========================================================================== #
def test_T0_no_structure_always_flat():
    opens = [100, 100, 105, 110, 115, 120, 125]  # rising, but no entry allowed
    entry_ok = [False] * 5
    core, ref, s, e = run_seg(opens, entry_ok)
    assert prod_path(core, s, e) == [0, 0, 0, 0, 0]
    assert float(core["value"][s, P2I[0]]) == 0.0
    assert ref_path(ref, s, e) == [0, 0, 0, 0, 0]


def test_T0_long_synthetic():
    opens = [100, 100, 100, 105, 110, 115, 120]
    entry_ok = [False, True, False, False, False]
    core, ref, s, e = run_seg(opens, entry_ok)
    assert int(core["action"][1, P2I[0]]) == 1  # LONG_ENTRY
    assert _transition_label(0, 1) == "LONG_ENTRY"
    path = prod_path(core, s, e)
    assert path[0] == 0 and path[1] == 1  # flat then long
    assert path[-1] == 0  # terminal flat
    assert path == ref_path(ref, s, e)


def test_T0_short_synthetic():
    opens = [100, 100, 100, 95, 90, 85, 80]
    entry_ok = [False, True, False, False, False]
    core, ref, s, e = run_seg(opens, entry_ok)
    assert int(core["action"][1, P2I[0]]) == -1  # SHORT_ENTRY
    assert _transition_label(0, -1) == "SHORT_ENTRY"
    assert prod_path(core, s, e) == ref_path(ref, s, e)
    assert prod_path(core, s, e)[-1] == 0


def test_T0_illegal_entry_blocked():
    # unit level
    assert _allowed_actions(0, False, False) == (0,)
    assert _allowed_actions(0, True, False) == (-1, 0, 1)
    assert _ref_allowed(0, False, False) == (0,)
    assert _ref_allowed(0, True, False) == (-1, 0, 1)
    # DP level: no entry when all-ineligible
    opens = [100, 101, 102, 103, 104, 105, 106]
    core, _ref, s, e = run_seg(opens, [False] * 5)
    for t in range(s, e):
        assert int(core["action"][t, P2I[0]]) == 0


def test_T0_free_exit_where_mask_zero():
    # enter long at t=1, then price falls; the optimal exit happens at t=2,
    # where entry mask == 0 (exits are unrestricted).
    opens = [100, 100, 100, 110, 105, 100, 95]
    entry_ok = [False, True, False, False, False]
    core, ref, s, e = run_seg(opens, entry_ok)
    transitions = backtrack_segment(core, np.asarray(opens, float),
                                    np.ones(len(opens)), np.arange(len(opens)),
                                    s, e)
    by_t = {int(tr["decision_bar_index"]): tr["transition"] for tr in transitions}
    assert by_t.get(1) == "LONG_ENTRY"
    assert by_t.get(2) == "LONG_EXIT"      # exit at a non-eligible decision
    assert entry_ok[2] is False
    assert prod_path(core, s, e) == ref_path(ref, s, e)


def test_T0_reversal_requires_entry_mask():
    # unit level
    assert -1 in _allowed_actions(1, True, False)
    assert _allowed_actions(1, False, False) == (1, 0)
    assert -1 in _ref_allowed(1, True, False)
    assert _ref_allowed(1, False, False) == (1, 0)
    # DP level: after a long entry, no reversal while ineligible.
    opens = [100, 100, 100, 105, 100, 95, 90]
    entry_ok = [True, False, False, False, False]
    core, _ref, s, e = run_seg(opens, entry_ok)
    path = prod_path(core, s, e)
    for k in range(1, len(path)):
        assert not (path[k - 1] != 0 and path[k] == -path[k - 1]), "illegal reversal"


def test_T0_next_open_fill():
    opens = [100, 100, 100, 105, 110, 115, 120]
    entry_ok = [False, True, False, False, False]
    arr = np.asarray(opens, float)
    core, _ref, s, e = run_seg(opens, entry_ok)
    times = np.arange(len(opens))
    transitions = backtrack_segment(core, arr, np.ones(len(opens)), times, s, e)
    assert transitions, "expected at least one transition"
    for tr in transitions:
        assert tr["fill_bar_index"] == tr["decision_bar_index"] + 1
        assert tr["fill_price"] == arr[tr["decision_bar_index"] + 1]


def test_T0_no_cross_segment():
    # segments must be long enough for the ATR200/SR warmup to produce structure
    base = _synth_base(n=900, seed=5, disc_index=450)
    res = run_base_dp(base, KernelCounters(), symbol="SYNTH")
    seg = res["segment"]
    trades = res["trades"]
    assert trades, "expected some trades on synthetic data"
    for tr in trades:
        ei, xi = int(tr["entry_fill_index"]), int(tr["exit_fill_index"])
        assert seg[ei] == seg[xi], "trade crosses a discontinuity boundary"
    # every segment must be flat at its own end
    for segm in res["segments"]:
        s, e = segm["seg_start"], segm["seg_end"]
        if e - s < 2:
            continue
        core = res["cores"][(s, e)]
        p = 0
        for t in range(s, e):
            p = int(core["action"][t, P2I[p]])
        assert p == 0, "segment did not terminate flat"


def test_T0_terminal_flat():
    opens = [100, 101, 103, 106, 110, 115, 121]
    core, _ref, s, e = run_seg(opens, [True] * 5)
    assert prod_path(core, s, e)[-1] == 0
    # terminal decision forces a=0 for every state
    for p in POS:
        assert int(core["action"][e - 1, P2I[int(p)]]) == 0


def test_T0_sign_symmetry():
    opens = np.array([100, 101, 103, 106, 110, 115, 121], float)
    c = 100.0
    mirror = 2.0 * c - opens
    entry_ok = [True] * 5
    core_a, _ra, s, e = run_seg(opens, entry_ok)
    core_b, _rb, sb, eb = run_seg(mirror, entry_ok)
    pa = prod_path(core_a, s, e)
    pb = prod_path(core_b, sb, eb)
    assert pa == [-x for x in pb]
    assert pa[0] == 1 and pb[0] == -1
    va = float(core_a["value"][s, P2I[0]])
    vb = float(core_b["value"][sb, P2I[0]])
    assert math.isclose(va, vb, rel_tol=1e-9, abs_tol=1e-9)


def test_T0_brute_force_equivalence():
    collected = {"value": 0, "chosen": 0, "ambiguous": 0, "best_set": 0}
    for seed in range(60):
        rng = np.random.default_rng(seed)
        D = 6 + seed % 5
        base = 100.0
        opens = np.array([base] + list(base + np.cumsum(rng.normal(0, 1.0, D + 1))))
        entry_ok = rng.random(D + 2) < 0.5
        s, e = 0, D
        core, ref, s, e = run_seg(opens, entry_ok, seg_start=s, seg_end=e)
        assert_dp_matches_reference(core, ref, s, e, collected)
        # production flat-start path must be an optimal reference path
        pp = tuple(prod_path(core, s, e))
        opt = {tuple(pth) for pth in ref["optimal_paths"]}
        assert pp in opt, f"seed {seed}: production path not optimal"
    assert collected == {"value": 0, "chosen": 0, "ambiguous": 0, "best_set": 0}, collected


def test_T0_future_mutation_entry_mask():
    n = 400
    base = _synth_base(n=n, seed=2)
    res_a = stream_from_base(base, KernelCounters(), symbol="SYNTH",
                             capture_entry_bits=True, emit_events=False)
    bits_a = res_a["entry_candidate_bits"]
    assert int((bits_a != 0).sum()) > 0

    K = n - 1
    base2 = base.copy().reset_index(drop=True)
    base2.at[base2.index[K], "close"] = float(base["close"].iloc[K]) + 5.0
    res_b = stream_from_base(base2, KernelCounters(), symbol="SYNTH",
                             capture_entry_bits=True, emit_events=False)
    bits_b = res_b["entry_candidate_bits"]
    assert np.array_equal(bits_a[:K], bits_b[:K]), "future bar changed past entry mask"


def test_T0_capture_does_not_change_events():
    base = _synth_base(n=400, seed=3)
    r0 = stream_from_base(base, KernelCounters(), symbol="SYNTH",
                          capture_entry_bits=False, emit_events=True)
    r1 = stream_from_base(base, KernelCounters(), symbol="SYNTH",
                          capture_entry_bits=True, emit_events=True)
    ev0 = [(e["event_type"], e["decision_bar_index"], e["structure_id"],
            e["near_edge"], e["far_edge"]) for e in r0["events"]]
    ev1 = [(e["event_type"], e["decision_bar_index"], e["structure_id"],
            e["near_edge"], e["far_edge"]) for e in r1["events"]]
    assert ev0 == ev1, "capture_entry_bits changed the emitted event rows"
    assert len(ev0) > 0

    # emit_events=False must not change the entry mask
    r2 = stream_from_base(base, KernelCounters(), symbol="SYNTH",
                          capture_entry_bits=True, emit_events=False)
    assert r2["events"] == []
    assert np.array_equal(r1["entry_candidate_bits"], r2["entry_candidate_bits"])


def test_T0_data_end_censored():
    # (a) single segment -> DATA_END (censored, training_eligible=False)
    base = _synth_base(n=300, seed=4)
    res = run_base_dp(base, KernelCounters(), symbol="SYNTH")
    assert len(res["segments"]) == 1
    assert res["segments"][0]["terminal_reason"] == DATA_END
    assert all(r["terminal_reason"] == DATA_END for r in res["action_rows"])
    assert all(r["training_eligible"] is False for r in res["action_rows"])

    # (b) multi-segment: first ends by DISCONTINUITY (eligible), last is DATA_END
    base2 = _synth_base(n=300, seed=4, disc_index=150)
    res2 = run_base_dp(base2, KernelCounters(), symbol="SYNTH")
    assert len(res2["segments"]) == 2
    assert res2["segments"][0]["terminal_reason"] == DISCONTINUITY
    assert res2["segments"][0]["training_eligible"] is True
    assert res2["segments"][-1]["terminal_reason"] == DATA_END
    assert res2["segments"][-1]["training_eligible"] is False
    first_seg_rows = [r for r in res2["action_rows"]
                      if r["segment"] == res2["segments"][0]["seg_start"]
                      or r["decision_bar_index"] <= res2["segments"][0]["seg_end"]]
    assert any(r["terminal_reason"] == DISCONTINUITY for r in first_seg_rows)


def test_T0_negative_control_illegal_entry_must_fail():
    good_rows = [
        {"decision_bar_index": 0, "oracle_position_before": 0,
         "oracle_position_after": 1, "entry_eligible": True},
        {"decision_bar_index": 1, "oracle_position_before": 1,
         "oracle_position_after": 1, "entry_eligible": False},
        {"decision_bar_index": 2, "oracle_position_before": 1,
         "oracle_position_after": 0, "entry_eligible": False},
    ]
    assert invariant_violations(good_rows) == []

    # deliberately inject an illegal Flat->Long when not eligible; the checker
    # MUST flag it (this is the negative control: it must FAIL, not pass).
    bad_entry = list(good_rows)
    bad_entry[0] = dict(bad_entry[0], entry_eligible=False)
    assert invariant_violations(bad_entry) != []

    bad_rev = good_rows + [{"decision_bar_index": 3, "oracle_position_before": 1,
                            "oracle_position_after": -1, "entry_eligible": False}]
    assert invariant_violations(bad_rev) != []


def test_T0_artifact_frames_schema():
    """The oracle tables carry the frozen §14 columns (no disk write here)."""
    base = _synth_base(n=900, seed=5, disc_index=450)
    res = run_base_dp(base, KernelCounters(), symbol="SYNTH")
    frames = build_artifact_frames(res)

    action_cols = {
        "symbol", "decision_bar_index", "decision_time", "segment",
        "entry_candidate_bits", "entry_eligible", "oracle_position_before",
        "oracle_position_after", "oracle_transition", "oracle_edge_points",
        "oracle_edge_ATR", "oracle_ambiguous", "label_available_time",
        "terminal_reason", "training_eligible",
    }
    trade_cols = {
        "trade_id", "symbol", "direction", "entry_decision_index",
        "entry_fill_index", "entry_fill_time", "entry_fill_price",
        "entry_candidate_bits", "exit_decision_index", "exit_fill_index",
        "exit_fill_time", "exit_fill_price", "holding_bars", "gross_points",
        "gross_ATR", "MFE_ATR", "MAE_ATR", "entry_edge_ATR", "exit_edge_ATR",
        "terminal_reason",
    }
    assert action_cols <= set(frames["oracle_actions"].columns)
    assert trade_cols <= set(frames["oracle_trades"].columns)

    meta = artifact_metadata("deadbeef", "SYNTH", "2024-01-01", "2024-01-02")
    for k in ("source_sha", "task_id", "math_version", "NEAR_ATR",
              "execution_semantics", "objective", "c_roundtrip_atr"):
        assert k in meta
    assert meta["c_roundtrip_atr"] == 0.0


# =========================================================================== #
# T1 tie handling — a TRUE tie must never be counted as a mismatch             #
# =========================================================================== #
def test_T1_tie_not_counted_as_mismatch():
    # flat prices -> every action ties at value 0
    opens = [100.0] * 7
    entry_ok = [True] * 5
    core, ref, s, e = run_seg(opens, entry_ok)
    assert bool(core["ambiguous"][s, P2I[0]]) is True
    assert int(core["action"][s, P2I[0]]) == 0  # tie prefers no turnover
    assert ref["best_actions"][(s, 0)] == frozenset({-1, 0, 1})
    collected = {"value": 0, "chosen": 0, "ambiguous": 0, "best_set": 0}
    assert_dp_matches_reference(core, ref, s, e, collected)
    assert collected == {"value": 0, "chosen": 0, "ambiguous": 0, "best_set": 0}


# =========================================================================== #
# T1 — real-data differential (already-causal inputs sliced from the full      #
#      streaming run) + long-segment invariants                                #
# =========================================================================== #
@pytest.mark.parametrize("symbol", ["AG", "CU"])
def test_T1_real_data(symbol):
    N = 1500
    counters = KernelCounters()
    res = run_symbol_dp(symbol, counters, max_bars=N)

    # --- counter contract (production path) ---
    assert counters.raw_load_count == 1
    assert counters.resample_count == 4
    assert counters.reference_call_count == 0
    assert counters.full_history_recompute_count == 0
    assert counters.concat_count == 0
    assert counters.dp_state_count > 0

    opens = res["open"]
    atr5 = res["atr5"]
    entry_ok = res["entry_eligible"]

    # --- T1-B: slice SHORT windows from the ALREADY-CAUSAL DP inputs ---
    # (never rebuild SR/Liquidity/DTP from the window)
    seg0 = next(sm for sm in res["segments"] if sm["seg_end"] - sm["seg_start"] >= 40)
    s, e = seg0["seg_start"], seg0["seg_end"]
    n_cap = min(e, res["n"] - 1)

    windows: list = []
    eligible_idx = [int(i) for i in np.where(entry_ok[: e + 1])[0] if int(i) >= s]
    if eligible_idx:
        step = max(1, len(eligible_idx) // 10)
        for idx in eligible_idx[::step]:
            D = 8 + (idx % 3)  # 8..10 decisions
            ws = max(s, idx - 3)
            we = ws + D
            if we > n_cap:
                we = n_cap
                ws = we - D
            if ws >= s and 7 <= (we - ws) <= 10:
                windows.append((ws, we))
            if len(windows) >= 10:
                break
    # a few all-ineligible windows (entry mask == 0 over the whole window)
    for idx in range(s, max(s, n_cap - 11)):
        if not entry_ok[idx: idx + 8].any():
            windows.append((idx, idx + 8))
        if len(windows) >= 12:
            break

    seen = set()
    uniq = []
    for w in windows:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    windows = uniq
    assert len(windows) >= 5, f"{symbol}: too few T1 windows ({len(windows)})"

    collected = {"value": 0, "chosen": 0, "ambiguous": 0, "best_set": 0}
    eligible_windows = 0
    for (ws, we) in windows:
        core = solve_segment_dp(opens, atr5, entry_ok, ws, we)
        ref = reference_solve_segment(opens, atr5, entry_ok, ws, we)
        assert_dp_matches_reference(core, ref, ws, we, collected)
        # value_start parity (V(s;-1), V(s;0), V(s;+1))
        for p in POS:
            pv = float(core["value"][ws, P2I[int(p)]])
            rv = float(ref["value"][(ws, int(p))])
            assert math.isclose(pv, rv, rel_tol=1e-9, abs_tol=1e-6)
        # production flat-start path must be an optimal reference path
        pp = tuple(prod_path(core, ws, we))
        opt = {tuple(pth) for pth in ref["optimal_paths"]}
        assert pp in opt, f"{symbol}: window {ws}-{we} path not optimal"
        assert pp[-1] == 0, "window terminal must be flat"
        # illegal entry / reversal must be zero on the realised path
        p = 0
        if bool(entry_ok[ws:we].any()):
            eligible_windows += 1
        for t in range(ws, we):
            q = int(core["action"][t, P2I[p]])
            if p == 0 and q != 0:
                assert bool(entry_ok[t]), "illegal entry on window path"
            if p != 0 and q == -p:
                assert bool(entry_ok[t]), "illegal reversal on window path"
            p = q

    assert collected == {"value": 0, "chosen": 0, "ambiguous": 0, "best_set": 0}, collected
    assert eligible_windows >= 1, f"{symbol}: no eligible window exercised"

    # --- long-segment invariants (no brute force on the full segment) ---
    ar = res["action_rows"]
    assert ar, "expected action rows"
    assert ar[0]["oracle_position_before"] == 0
    assert invariant_violations(ar) == []

    seg_arr = res["segment"]
    for tr in res["trades"]:
        ei, xi = int(tr["entry_fill_index"]), int(tr["exit_fill_index"])
        assert seg_arr[ei] == seg_arr[xi], "trade crosses a discontinuity"
        assert tr["entry_fill_index"] == tr["entry_decision_index"] + 1
        assert tr["exit_fill_index"] == tr["exit_decision_index"] + 1
        assert tr["holding_bars"] >= 1
        assert tr["entry_fill_index"] < tr["exit_fill_index"]
        assert tr["MFE_ATR"] >= tr["gross_ATR"] - 1e-9
        assert tr["MAE_ATR"] <= tr["gross_ATR"] + 1e-9

    for sm in res["segments"]:
        s2, e2 = sm["seg_start"], sm["seg_end"]
        if (s2, e2) not in res["cores"]:
            continue
        core = res["cores"][(s2, e2)]
        p = 0
        for t in range(s2, e2):
            q = int(core["action"][t, P2I[p]])
            assert q != int(INVALID_ACTION)
            p = q
        assert p == 0, "segment did not terminate flat"

    # reconstruction PnL == DP value (flat-start) across all segments
    total_val = sum(
        float(res["cores"][(sm["seg_start"], sm["seg_end"])]["value"][sm["seg_start"], P2I[0]])
        for sm in res["segments"]
        if (sm["seg_start"], sm["seg_end"]) in res["cores"]
    )
    total_gross = sum(float(t["gross_points"]) for t in res["trades"])
    assert math.isclose(total_val, total_gross, rel_tol=1e-9, abs_tol=1e-6), (
        f"{symbol}: DP value {total_val} != reconstruction PnL {total_gross}"
    )


# =========================================================================== #
# TP — performance gate (5k / 10k / 20k)                                        #
# =========================================================================== #
def test_TP_performance_gate():
    sizes = [5000, 10000, 20000]
    times = {}
    for N in sizes:
        c = KernelCounters()
        res = run_symbol_dp("AG", c, max_bars=N)
        times[N] = res["runtime_total_sec"]
        assert c.reference_call_count == 0
        assert c.full_history_recompute_count == 0
        assert c.concat_count == 0
        assert c.raw_load_count == 1
        assert c.resample_count == 4
        assert c.dp_state_count > 0

    r1 = times[10000] / times[5000]
    r2 = times[20000] / times[10000]
    print(
        f"  TP timing: 5k={times[5000]:.2f}s 10k={times[10000]:.2f}s "
        f"20k={times[20000]:.2f}s  ratios={r1:.2f}/{r2:.2f}"
    )
    assert r1 < 2.8, f"5k->10k scaling {r1} exceeds 2.8"
    assert r2 < 2.8, f"10k->20k scaling {r2} exceeds 2.8"
