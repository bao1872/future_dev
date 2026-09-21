"""
test_structure_interaction_entry_v1
====================================

T0 unit tests + NC1-NC3 negative controls + T1 differential + TP performance
gate for the Research-1 Structure Interaction Entry Kernel.

The production kernel (experiment_structure_interaction_entry_v1) is the
EXECUTOR: it returns an Evidence Packet only. These tests verify the contract
(geometry orientation, event taxonomy, causal isolation, parity with the slow
reference, and the performance/counter gates) without interpreting research
significance.

Run with the project interpreter (Python 3.11+, requires enum.StrEnum):
    .venv/bin/python -m pytest research/liquidity_oracle_atlas/test_structure_interaction_entry_v1.py -v
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import pytest

from research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1 import (
    Episode,
    HORIZONS,
    KernelCounters,
    NEAR_ATR,
    REV_SIGN,
    ROLES,
    TF_MINUTES,
    TF_ORDER,
    _first_passage,
    _stream_from_base,
    build_base_frame,
    build_base_from_arrays,
    classify_bar,
    classify_reference,
    compute_outcome,
    recedes_without_touch,
    run_symbol_streaming,
    select_target,
    slow_preview_state_reference,
    stream_from_base,
)

# --------------------------------------------------------------------------- #
# Event taxonomy (contract §9 / §4) — role-prefixed, 4 roles independent      #
# --------------------------------------------------------------------------- #
_EVENT_CORE = [
    "TOUCH_REJECT",
    "ZONE_REJECT",
    "ENTER_ZONE",
    "FIRST_BREAK",
    "SAME_BAR_FULL_RECLAIM",
    "SAME_BAR_PARTIAL_RECLAIM",
    "DELAYED_FULL_RECLAIM",
    "DELAYED_PARTIAL_RECLAIM",
    "BREAK_EXTENSION",
    "BREAK_RETURNING",
    "BREAK_HOLD",
    "REBREAK_AFTER_PARTIAL",
    "APPROACH_NO_TOUCH_REJECT",
]
KNOWN_EVENT_TYPES = {
    f"{role}_{e}" for role in ROLES for e in _EVENT_CORE
}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def mk_ep(
    role: str,
    near: float,
    far: float,
    has_broken: bool = False,
    touched: bool = False,
    entered: bool = False,
    start_atr: float = 1.0,
    rev_sign: int = None,
) -> Episode:
    return Episode(
        symbol="SYNTH",
        tf="m5",
        role=role,
        segment=0,
        structure_id="SYNTH|0",
        start_i=0,
        start_time=np.datetime64("2024-01-01T09:00"),
        near_edge=near,
        far_edge=far,
        strength=1.0,
        rev_sign=REV_SIGN[role] if rev_sign is None else rev_sign,
        start_atr=start_atr,
        has_broken=has_broken,
        touched=touched,
        entered=entered,
    )


def _synth_ohlc(n: int, seed: int = 0, disc_index: int = None):
    """Mean-reverting synthetic 5m bars that reliably form SR/liquidity
    swings, with valid OHLC (high>=max(o,c), low<=min(o,c))."""
    rng = np.random.default_rng(seed)
    x = 100.0
    o = np.empty(n)
    h = np.empty(n)
    l = np.empty(n)
    c = np.empty(n)
    for i in range(n):
        x += rng.normal(0.0, 0.6)
        x += 0.08 * (100.0 - x)  # mean reversion -> repeated approaches
        c[i] = x
        o[i] = x + rng.normal(0.0, 0.2)
        hi = max(o[i], c[i]) + abs(rng.normal(0.0, 0.4))
        lo = min(o[i], c[i]) - abs(rng.normal(0.0, 0.4))
        h[i] = hi
        l[i] = lo
    t = pd.date_range("2024-01-01 09:00", periods=n, freq="5min")
    day = pd.to_datetime(["2024-01-01"] * n)
    disc = np.zeros(n, dtype=bool)
    if disc_index is not None:
        disc[disc_index] = True
    return t, day, o, h, l, c, disc


def _synth_base(n=400, seed=0, disc_index=None, counters=None):
    counters = counters if counters is not None else KernelCounters()
    t, day, o, h, l, c, disc = _synth_ohlc(n, seed=seed, disc_index=disc_index)
    info = build_base_from_arrays(t, day, o, h, l, c, disc, counters)
    return info["base"]


def geom_equal(a, b, tol=1e-9):
    """Production geometry ``a = (channels, liq_up, liq_down, atr)`` vs
    reference dict ``b``. Order-stable (both come from the same IndicatorState
    stepping identical data) with float tolerance."""
    ch_a, up_a, down_a, atr_a = a
    ch_b = b["sr_channels"]
    up_b = b["liq_up"]
    down_b = b["liq_down"]
    atr_b = b["atr"]

    def feq(x, y, tol_=1e-6):
        if x is None or y is None:
            return x is y
        try:
            return math.isclose(float(x), float(y), abs_tol=tol_)
        except (TypeError, ValueError):
            return x == y

    if len(ch_a) != len(ch_b):
        return False
    for x, y in zip(ch_a, ch_b):
        if not all(feq(p, q) for p, q in zip(x, y)):
            return False
    for la, lb in ((up_a, up_b), (down_a, down_b)):
        if len(la) != len(lb):
            return False
        for x, y in zip(la, lb):
            for k in x:
                if not feq(x[k], y.get(k)):
                    return False
    if (math.isnan(atr_a) and math.isnan(atr_b)) or (
        math.isnan(atr_a) is False and math.isnan(atr_b) is False
        and math.isclose(atr_a, atr_b, rel_tol=1e-6, abs_tol=1e-9)
    ):
        return True
    return False


# =========================================================================== #
# T0.1 - SUPPORT touch-reject (price from above touches near, closes above)    #
# =========================================================================== #
def test_T0_1_support_touch_reject():
    ep = mk_ep("SUPPORT", near=100.0, far=98.0)  # near=top, far=bottom
    ev = classify_bar(ep, O=102.0, H=103.0, L=99.0, C=101.0)
    assert ev == "TOUCH_REJECT"
    assert f"SUPPORT_{ev}" in KNOWN_EVENT_TYPES
    assert not ep.has_broken


# =========================================================================== #
# T0.2 - SUPPORT same-bar full reclaim (break then close back above near)      #
# =========================================================================== #
def test_T0_2_support_same_bar_full_reclaim():
    ep = mk_ep("SUPPORT", near=100.0, far=98.0)
    ev = classify_bar(ep, O=102.0, H=103.0, L=97.0, C=101.0)
    assert ev == "SAME_BAR_FULL_RECLAIM"
    assert ep.has_broken is True
    assert ep.phase == "FULL_RECLAIM"


# =========================================================================== #
# T0.3 - RESISTANCE touch-reject (mirror of T0.1, rev_sign = -1)               #
# =========================================================================== #
def test_T0_3_resistance_touch_reject():
    ep = mk_ep("RESISTANCE", near=100.0, far=102.0)  # near=bottom, far=top
    ev = classify_bar(ep, O=98.0, H=101.0, L=97.0, C=99.0)
    assert ev == "TOUCH_REJECT"
    assert f"RESISTANCE_{ev}" in KNOWN_EVENT_TYPES
    # orientation: u_near > u_far
    s = ep.rev_sign
    assert s * ep.near_edge > s * ep.far_edge


# =========================================================================== #
# T0.4 - Liquidity orientation: BUYSIDE near=bottom, SELLSIDE near=top          #
# =========================================================================== #
def test_T0_4_liquidity_orientation():
    # BUYSIDE_LIQUIDITY: candidate below price, near=bottom, far=top
    liq_up = [{"left": 10, "level": 100.0, "top": 102.0, "bottom": 98.0,
               "broken": False, "breach_i": None}]
    cand = select_target("BUYSIDE_LIQUIDITY", [], liq_up, [], 95.0, 1.0, "m5", 0, 0, {})
    assert cand is not None
    assert cand["near_edge"] == 98.0  # bottom
    assert cand["far_edge"] == 102.0  # top
    s = REV_SIGN["BUYSIDE_LIQUIDITY"]
    assert s * cand["near_edge"] > s * cand["far_edge"]

    # SELLSIDE_LIQUIDITY: candidate above price, near=top, far=bottom
    liq_down = [{"left": 11, "level": 96.0, "top": 98.0, "bottom": 94.0,
                 "broken": False, "breach_i": None}]
    cand2 = select_target("SELLSIDE_LIQUIDITY", [], [], liq_down, 100.0, 1.0, "m5", 0, 0, {})
    assert cand2 is not None
    assert cand2["near_edge"] == 98.0  # top
    assert cand2["far_edge"] == 94.0   # bottom
    s2 = REV_SIGN["SELLSIDE_LIQUIDITY"]
    assert s2 * cand2["near_edge"] > s2 * cand2["far_edge"]


# =========================================================================== #
# T0.5 - SUPPORT same-bar partial reclaim (break, close between far and near)   #
# =========================================================================== #
def test_T0_5_support_same_bar_partial_reclaim():
    ep = mk_ep("SUPPORT", near=100.0, far=98.0)
    ev = classify_bar(ep, O=102.0, H=103.0, L=97.0, C=98.5)
    assert ev == "SAME_BAR_PARTIAL_RECLAIM"
    assert ep.has_broken is True
    assert ep.had_partial_reclaim is True


# =========================================================================== #
# T0.6 - Break -> reclaim sequence (FIRST_BREAK then DELAYED_FULL_RECLAIM)      #
# =========================================================================== #
def test_T0_6_break_then_delayed_reclaim_sequence():
    ep = mk_ep("SUPPORT", near=100.0, far=98.0, has_broken=False)
    e1 = classify_bar(ep, O=102.0, H=103.0, L=97.0, C=97.0)  # break, close<=far
    assert e1 == "FIRST_BREAK"
    assert ep.has_broken is True
    e2 = classify_bar(ep, O=98.0, H=101.0, L=97.0, C=101.0)  # later full reclaim
    assert e2 == "DELAYED_FULL_RECLAIM"
    assert f"SUPPORT_{e2}" in KNOWN_EVENT_TYPES


# =========================================================================== #
# T0.7 - No-touch reject logic (causal terminal condition, never on start bar)  #
# =========================================================================== #
def test_T0_7_recedes_without_touch():
    # receded beyond proximity, never interacted -> True
    ep = mk_ep("SUPPORT", near=100.0, far=98.0, start_atr=1.0)
    assert recedes_without_touch(ep, C=100.6, atr_tf=1.0) is True
    # not receded (within radius) -> False
    assert recedes_without_touch(ep, C=100.4, atr_tf=1.0) is False
    # touched -> False
    ep2 = mk_ep("SUPPORT", near=100.0, far=98.0, touched=True)
    assert recedes_without_touch(ep2, C=100.6, atr_tf=1.0) is False
    # entered -> False
    ep3 = mk_ep("SUPPORT", near=100.0, far=98.0, entered=True)
    assert recedes_without_touch(ep3, C=100.6, atr_tf=1.0) is False
    # already emitted an event -> False
    ep4 = mk_ep("SUPPORT", near=100.0, far=98.0, start_atr=1.0)
    ep4.events_emitted = 1
    assert recedes_without_touch(ep4, C=100.6, atr_tf=1.0) is False
    # invalid ATR -> False
    ep5 = mk_ep("SUPPORT", near=100.0, far=98.0, start_atr=0.0)
    assert recedes_without_touch(ep5, C=100.6, atr_tf=0.0) is False


# =========================================================================== #
# T0.8 - Future-mutation guard: corrupting a FUTURE bar does not alter past     #
#        events (production is single-pass causal).                             #
# =========================================================================== #
def test_T0_8_future_mutation_guard():
    n = 400
    base = _synth_base(n=n, seed=2, disc_index=None)
    res_a = stream_from_base(base, KernelCounters(), symbol="SYNTH")
    events_a = res_a["events"]
    assert events_a, "expected at least some events on synthetic data"

    K = n - 1
    base2 = base.copy()
    base2 = base2.reset_index(drop=True)
    new_close = float(base["close"].iloc[K]) + 5.0
    base2.at[base2.index[K], "close"] = new_close
    res_b = stream_from_base(base2, KernelCounters(), symbol="SYNTH")
    events_b = res_b["events"]

    key_b = {
        (e["structure_id"], e["decision_bar_index"], e["event_type"]): e
        for e in events_b
    }
    struct_fields = [
        "event_type", "decision_bar_index", "structure_id", "structure_tf",
        "structure_type", "near_edge", "far_edge", "structure_top",
        "structure_bottom", "visit_count",
    ]
    compared = 0
    for e in events_a:
        if e["decision_bar_index"] >= K:
            continue  # this decision may include the corrupted bar in its forming
        compared += 1
        match = key_b.get((e["structure_id"], e["decision_bar_index"], e["event_type"]))
        assert match is not None, (
            f"past event missing after future corruption: "
            f"{e['structure_id']}@{e['decision_bar_index']}:{e['event_type']}"
        )
        for f in struct_fields:
            if isinstance(e[f], str):
                assert e[f] == match[f], f"field {f} changed for past event under future corruption"
            else:
                assert math.isclose(e[f], match[f], abs_tol=1e-9, rel_tol=1e-9), (
                    f"field {f} changed for past event under future corruption"
                )
    assert compared > 0, "no comparable past events generated"


# =========================================================================== #
# T0.9 - Segment reset: no episode spans a discontinuity boundary               #
# =========================================================================== #
def test_T0_9_segment_reset():
    n = 400
    disc_index = 150
    c = KernelCounters()
    base = _synth_base(n=n, seed=2, disc_index=disc_index, counters=c)
    seg = base["segment"].to_numpy()
    assert seg[disc_index - 1] == 0 and seg[disc_index] == 1, "segment must change at disc"

    res = stream_from_base(base, KernelCounters(), symbol="SYNTH")
    events = res["events"]
    assert events, "expected events across both segments"

    by_ep: dict = {}
    for e in events:
        by_ep.setdefault(e["episode_id"], []).append(e["decision_bar_index"])

    spans = 0
    for ep_id, idxs in by_ep.items():
        segs = {int(seg[i]) for i in idxs}
        if len(segs) > 1:
            spans += 1
    assert spans == 0, f"{spans} episodes span the discontinuity boundary"

    seg1_events = [e for e in events if int(seg[e["decision_bar_index"]]) == 1]
    assert seg1_events, "streaming must continue past the segment reset"


# =========================================================================== #
# T0.10 - Geometry freeze: per-episode structure geometry is constant          #
# =========================================================================== #
def test_T0_10_geometry_freeze():
    base = _synth_base(n=400, seed=3)
    res = stream_from_base(base, KernelCounters(), symbol="SYNTH")
    events = res["events"]
    assert events

    by_ep: dict = {}
    for e in events:
        by_ep.setdefault(e["episode_id"], []).append(e)

    frozen = ["structure_tf", "structure_type", "structure_id",
              "near_edge", "far_edge", "structure_top", "structure_bottom"]
    for ep_id, rows in by_ep.items():
        ref = {f: rows[0][f] for f in frozen}
        for r in rows[1:]:
            for f in frozen:
                if isinstance(ref[f], float):
                    assert math.isclose(ref[f], r[f], abs_tol=1e-9), (
                        f"episode {ep_id} geometry field {f} not frozen"
                    )
                else:
                    assert ref[f] == r[f], (
                        f"episode {ep_id} geometry field {f} not frozen"
                    )


# =========================================================================== #
# T0.11 - Ambiguous first passage returns 0.0 (both sides touched same bar)     #
# =========================================================================== #
def test_T0_11_ambiguous_first_passage():
    a0 = 1.0
    # both +/-0.5 ATR touched in the same bar -> AMBIGUOUS
    assert _first_passage(+1, 100.0, np.array([100.6]), np.array([99.4]), a0, 0.5) == 0.0
    # only target side -> favorable
    assert _first_passage(+1, 100.0, np.array([100.6]), np.array([100.2]), a0, 0.5) == 1.0
    # only stop side -> adverse
    assert _first_passage(+1, 100.0, np.array([99.8]), np.array([99.4]), a0, 0.5) == -1.0
    # neither -> NaN (censored)
    v = _first_passage(+1, 100.0, np.array([100.2]), np.array([99.8]), a0, 0.5)
    assert math.isnan(v)


# =========================================================================== #
# T0.12 - Semantic key uniqueness: event_type is role-prefixed, known, stable   #
# =========================================================================== #
def test_T0_12_semantic_key_unique():
    base = _synth_base(n=400, seed=4)
    res = stream_from_base(base, KernelCounters(), symbol="SYNTH")
    events = res["events"]
    assert events, "expected events to assert semantic keys"

    seen_types = set()
    for e in events:
        et = e["event_type"]
        seen_types.add(et)
        assert et in KNOWN_EVENT_TYPES, f"unknown event type {et}"
        assert et.startswith(e["structure_type"]), (
            f"event type {et} not prefixed by role {e['structure_type']}"
        )
        assert e["structure_type"] in ROLES
        assert e["structure_id"].startswith("SR|") or e["structure_id"].startswith("LIQ|")
    # at least two roles exercised (four structure types kept independent)
    roles_seen = {e["structure_type"] for e in events}
    assert len(roles_seen) >= 2


# =========================================================================== #
# NC1 - Differential sensitivity: perturbing geometry MUST be detected          #
# =========================================================================== #
def test_NC1_differential_sensitivity():
    a = (
        [(100.0, 98.0, 1.0)],
        [{"left": 1, "level": 100.0, "top": 102.0, "bottom": 98.0,
          "broken": False, "breach_i": None}],
        [],
        1.5,
    )
    b_clean = {
        "sr_channels": [(100.0, 98.0, 1.0)],
        "liq_up": [{"left": 1, "level": 100.0, "top": 102.0, "bottom": 98.0,
                    "broken": False, "breach_i": None}],
        "liq_down": [],
        "atr": 1.5,
    }
    assert geom_equal(a, b_clean) is True

    # perturb one channel top by +0.01 -> must fail
    a_perturbed = (
        [(100.01, 98.0, 1.0)],
        [{"left": 1, "level": 100.0, "top": 102.0, "bottom": 98.0,
          "broken": False, "breach_i": None}],
        [],
        1.5,
    )
    assert geom_equal(a_perturbed, b_clean) is False

    # perturb ATR by +0.01 -> must fail
    b_atr = dict(b_clean)
    b_atr["atr"] = 1.51
    assert geom_equal(a, b_atr) is False


# =========================================================================== #
# NC2 - Causality gate: slow reference ignores data beyond decision i           #
# =========================================================================== #
def test_NC2_reference_causality():
    n = 200
    c = KernelCounters()
    base = _synth_base(n=n, seed=5, counters=c)

    i = 50
    ref1 = slow_preview_state_reference(base, i, 15)

    base2 = base.copy().reset_index(drop=True)
    # corrupt a FUTURE bar (well beyond decision i)
    future = 150
    base2.at[base2.index[future], "close"] = float(base["close"].iloc[future]) + 7.0
    ref2 = slow_preview_state_reference(base2, i, 15)

    assert geom_equal(
        (ref1["sr_channels"], ref1["liq_up"], ref1["liq_down"], ref1["atr"]), ref2
    ), "slow reference leaked a future bar"


# =========================================================================== #
# NC3 - Break->reclaim sequence integrity: DELAYED_* requires prior FIRST_BREAK  #
# =========================================================================== #
def test_NC3_break_reclaim_sequence_integrity():
    # A fresh episode can NEVER emit a DELAYED_* (no prior break)
    ep = mk_ep("SUPPORT", near=100.0, far=98.0, has_broken=False)
    e1 = classify_bar(ep, O=102.0, H=103.0, L=97.0, C=97.0)
    assert e1 == "FIRST_BREAK"  # first break, never DELAYED
    assert "DELAYED" not in e1

    # After the break, a full reclaim is DELAYED_*, not SAME_BAR_*
    e2 = classify_bar(ep, O=98.0, H=101.0, L=97.0, C=101.0)
    assert e2 == "DELAYED_FULL_RECLAIM"

    # A brand-new episode whose single bar breaks AND reclaims is SAME_BAR_*
    ep2 = mk_ep("SUPPORT", near=100.0, far=98.0, has_broken=False)
    e3 = classify_bar(ep2, O=102.0, H=103.0, L=97.0, C=101.0)
    assert e3 == "SAME_BAR_FULL_RECLAIM"  # not DELAYED (no prior break)


# =========================================================================== #
# T1 - Differential: production geometry == slow reference (AG + CU, sampled)   #
#      The slow reference is O(N^2); we sample decisions at a stride across the #
#      decision space (parity is systematic, not random, so sampling suffices). #
# =========================================================================== #
@pytest.mark.parametrize("symbol", ["AG", "CU"])
def test_T1_differential(symbol):
    N = 1500
    STRIDE = 12
    counters = KernelCounters()
    info = build_base_frame(symbol, counters)
    base = info["base"]
    res = _stream_from_base(
        info["base"], info["form"], info["seg_completed"],
        counters, max_bars=N, symbol=symbol, capture_geom=True,
    )
    dg = res["decision_geom"]

    mismatch = 0
    samples = 0
    for i in range(0, N, STRIDE):
        for tf in TF_ORDER:
            samples += 1
            ref = slow_preview_state_reference(base, i, TF_MINUTES[tf])
            if not geom_equal(dg[i][tf], ref):
                mismatch += 1
                if mismatch <= 5:
                    prod = dg[i][tf]
                    print(
                        f"  [{symbol}] mismatch i={i} tf={tf} "
                        f"prod_ch={prod[0][:1]} ref_ch={ref['sr_channels'][:1]}"
                    )
    assert mismatch == 0, (
        f"{symbol}: T1 differential mismatch_count={mismatch} / {samples}"
    )

    # production-path counter contract
    assert counters.raw_load_count == 1
    assert counters.resample_count == len(TF_ORDER)
    assert counters.reference_call_count == 0
    assert counters.full_history_recompute_count == 0
    assert counters.concat_count == 0


# =========================================================================== #
# TP - Performance gate: linear scaling across N=5k/10k/20k, counters clean      #
# =========================================================================== #
def test_TP_performance_gate():
    sizes = [5000, 10000, 20000]
    times = {}
    for N in sizes:
        c = KernelCounters()
        t0 = time.perf_counter()
        run_symbol_streaming("AG", c, max_bars=N)
        times[N] = time.perf_counter() - t0
        assert c.reference_call_count == 0
        assert c.full_history_recompute_count == 0
        assert c.concat_count == 0
        assert c.raw_load_count == 1

    r1 = times[10000] / times[5000]
    r2 = times[20000] / times[10000]
    print(f"  TP timing: 5k={times[5000]:.2f}s 10k={times[10000]:.2f}s "
          f"20k={times[20000]:.2f}s  ratios={r1:.2f}/{r2:.2f}")
    assert r1 < 2.8, f"5k->10k scaling {r1} exceeds 2.8"
    assert r2 < 2.8, f"10k->20k scaling {r2} exceeds 2.8"
