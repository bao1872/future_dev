"""test_candidate_gate_r3_v1
===========================

Committed (clean-checkout) tests for the canonical R3 Candidate Gate
(FUTURE-R3-CANONICAL-M5-TOUCH-NEXTBAR-GATE-V1).

These tests MUST run in a clean checkout (``git clone && pytest``). They are
therefore restricted to:

  * Gate LOGIC (unit): ``derive_nextbar_candidate_gate`` +
    ``compute_candidate_episode_id`` over hand-built 4TF touch-bit sequences.
    These encode the user's frozen decision table directly:

        Candidate[t] = same_unit[t] AND (touch_bits[t-1] has 5m SR or 5m LIQ)

    and the full 4TF trigger context is preserved as candidate_trigger_bits.
  * Scalar / array bit-decoder contract (``touch_bit``).

Full-history artifact verification (differential vs canonical entry_mask,
row-count / SHA / version checks) lives in ``verify_candidate_gate_r3_artifacts.py``
because the parquet artifacts are intentionally NOT committed to Git.

No model / Y / Q / Oracle-action content is produced here.
"""

import numpy as np
import pytest

from research.liquidity_oracle_atlas.build_candidate_gate_r3_v1 import (
    CANDIDATE_MATH_VERSION,
    MASK_BIT,
    compute_candidate_episode_id,
    derive_nextbar_candidate_gate,
    touch_bit,
)


# --------------------------------------------------------------------------- #
# Synthetic bit helpers                                                         #
# --------------------------------------------------------------------------- #
def _bit(tf, fam):
    return 1 << MASK_BIT[(tf, fam)]


def _mask_of(*pairs):
    return np.uint16(sum(_bit(tf, fam) for tf, fam in pairs))


M5_SR = _bit("m5", "SR")
M5_LIQ = _bit("m5", "LIQ")
H1_SR = _bit("h1", "SR")   # canonical higher-TF keys: m15 / h1 / h4
H1_LIQ = _bit("h1", "LIQ")
H4_SR = _bit("h4", "SR")
H4_LIQ = _bit("h4", "LIQ")


def _run(prev_bits, cur_bits, same_unit):
    """Build a touch_bits array and run the gate.

    ``prev_bits`` / ``cur_bits`` are the touch mask for bar t-1 / bar t. The
    candidate decision lives at bar index 1 (the 'next bar'). ``same_unit`` is
    realized by breaking the segment at the indices where it is False, so the
    gate derives the requested no-cross-session / cross-unit inheritance itself.
    """
    touch = np.array(list(prev_bits) + list(cur_bits), dtype=np.uint16)
    n = len(touch)
    seg = np.ones(n, dtype=np.int64)
    td = np.array([np.datetime64("2026-01-02")] * n, dtype="datetime64[D]")
    for i in range(1, n):
        if not same_unit[i]:
            seg[i] = seg[i - 1] + 1  # break unit -> same_unit False
    return derive_nextbar_candidate_gate(touch, seg, td)


# --------------------------------------------------------------------------- #
# Gate logic: frozen decision table                                            #
# --------------------------------------------------------------------------- #
def test_5m_sr_prev_makes_candidate():
    g = _run([_mask_of(("m5", "SR"))], [_mask_of(("m5", "SR"))], [False, True])
    assert bool(g["candidate_any"][1])
    assert int(g["candidate_trigger_bits"][1]) == M5_SR


def test_5m_liq_prev_makes_candidate():
    g = _run([_mask_of(("m5", "LIQ"))], [_mask_of(("m5", "LIQ"))], [False, True])
    assert bool(g["candidate_any"][1])
    assert int(g["candidate_trigger_bits"][1]) == M5_LIQ


def test_5m_sr_plus_1h_sr_preserves_both_bits():
    prev = _mask_of(("m5", "SR"), ("h1", "SR"))
    g = _run([prev], [prev], [False, True])
    assert bool(g["candidate_any"][1])
    # trigger context keeps BOTH 5m and 1h SR bits
    assert int(g["candidate_trigger_bits"][1]) == (M5_SR | H1_SR)


def test_5m_sr_plus_4h_liq_preserves_both_bits():
    prev = _mask_of(("m5", "SR"), ("h4", "LIQ"))
    g = _run([prev], [prev], [False, True])
    assert bool(g["candidate_any"][1])
    assert int(g["candidate_trigger_bits"][1]) == (M5_SR | H4_LIQ)


def test_higher_tf_only_never_candidate():
    prev = _mask_of(("h1", "SR"))
    g = _run([prev], [prev], [False, True])
    assert not bool(g["candidate_any"][1])


def test_4h_liq_only_never_candidate():
    prev = _mask_of(("h4", "LIQ"))
    g = _run([prev], [prev], [False, True])
    assert not bool(g["candidate_any"][1])


def test_no_touch_prev_no_candidate():
    g = _run([0], [0], [False, True])
    assert not bool(g["candidate_any"][1])


def test_current_bar_new_sr_not_retroactive():
    # previous bar touched nothing; current bar freshly touches 5m SR.
    # Candidate may NOT be granted retroactively from the current bar's own touch.
    g = _run([0], [_mask_of(("m5", "SR"))], [False, True])
    assert not bool(g["candidate_any"][1])
    assert int(g["candidate_trigger_bits"][1]) == 0


def test_cross_segment_blocks_candidate():
    prev = _mask_of(("m5", "SR"))
    g = _run([prev], [prev], [False, False])  # same_unit False across segment
    assert not bool(g["candidate_any"][1])


def test_cross_trading_day_blocks_candidate():
    prev = _mask_of(("m5", "SR"))
    # same_unit False because trading_day differs
    g = _run([prev], [prev], [False, True])
    # force td to differ at index 1 to simulate a new trading day
    td = np.array(
        [np.datetime64("2026-01-02"), np.datetime64("2026-01-05")], dtype="datetime64[D]"
    )
    g2 = derive_nextbar_candidate_gate(
        np.array([prev, prev], dtype=np.uint16), np.ones(2, dtype=np.int64), td
    )
    assert not bool(g2["candidate_any"][1])


def test_broken_liquidity_not_counted():
    # Broken liquidity never sets a bit in touch_bits; only that flag is 0.
    g = _run([0], [0], [False, True])
    assert not bool(g["candidate_any"][1])


def test_trigger_bits_equal_prev_touch_for_candidates():
    # 5 consecutive bars: bar0 touch, bars1-4 candidate (same unit)
    prev = _mask_of(("m5", "SR"), ("h1", "LIQ"))
    touch = np.array([prev] * 5, dtype=np.uint16)
    seg = np.ones(5, dtype=np.int64)
    td = np.array([np.datetime64("2026-01-02")] * 5)
    g = derive_nextbar_candidate_gate(touch, seg, td)
    cand = g["candidate_any"]
    # bar0 cannot be candidate (no previous bar); bars1-4 are
    assert list(cand) == [False, True, True, True, True]
    # every candidate's trigger bits == the previous bar's touch bits
    for i in range(1, 5):
        assert int(g["candidate_trigger_bits"][i]) == int(touch[i - 1])


def test_episode_id_consecutive_then_gap():
    # three candidate bars same unit -> one episode; then a gap -> new episode
    prev = _mask_of(("m5", "SR"))
    touch = np.array([prev, prev, prev, prev, prev], dtype=np.uint16)
    seg = np.ones(5, dtype=np.int64)
    td = np.array([np.datetime64("2026-01-02")] * 5)
    cand = derive_nextbar_candidate_gate(touch, seg, td)["candidate_any"]
    ep = compute_candidate_episode_id(cand, np.ones(5, dtype=bool))
    assert list(ep) == [-1, 1, 1, 1, 1]

    # introduce a non-candidate at index 3 -> episode split
    cand2 = cand.copy()
    cand2[3] = False
    ep2 = compute_candidate_episode_id(cand2, np.ones(5, dtype=bool))
    assert list(ep2) == [-1, 1, 1, -1, 2]


# --------------------------------------------------------------------------- #
# Bit decoder contract (scalar + array)                                         #
# --------------------------------------------------------------------------- #
def test_touch_bit_scalar():
    bits = np.uint16(1 << MASK_BIT[("m5", "SR")])
    x = touch_bit(bits, "m5", "SR")
    # scalar input must return a 1-d array, so [0] indexing is valid
    assert x.shape == (1,)
    assert bool(x[0])
    # and direct bool() of the array works too
    assert bool(x)


def test_touch_bit_array():
    bits = np.array(
        [1 << MASK_BIT[("m5", "SR")], 1 << MASK_BIT[("h1", "LIQ")], 0],
        dtype=np.uint16,
    )
    xsr = touch_bit(bits, "m5", "SR")
    xliq = touch_bit(bits, "h1", "LIQ")
    assert xsr.shape == (3,)
    assert list(xsr) == [True, False, False]
    assert list(xliq) == [False, True, False]


def test_candidate_math_version_constant():
    assert CANDIDATE_MATH_VERSION == "r3_m5_touch_nextbar_gate_v1"
