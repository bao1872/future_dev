"""test_remote_audit_fix_02_v1
================================

REMOTE-AUDIT-FIX-02 T0/T1 tests (remote audit of 06ec42c + FIX-02 diff).

Blockers A/B/C/E/F are implemented in ``structural_god_oracle_m15_v4.py``.
This module locks in the FIX-02 checkpoint decisions:

D-UPDATE (user choice A): the diagnosed 349/350/7700/7773 case is NO LONGER
  a full-stream canonical invariant. After Block B removes the hidden holding
  cap, the earlier legal PREEMPT structure (bar 346) completes first and
  advances the global cursor past bar 348, so the DIAG opportunity is not
  reachable in the sequential one-position stream. The case is retained ONLY
  as a LOCAL / conditional best-entry regression
  (``test_sequential_best_entry_trade2_region`` in
  ``test_structural_god_oracle_m15_v4.py``), which still asserts exactly
  decision=349 / fill=350 / entry=7700 / target=7773 / utility=73.

FULL-STREAM INVARIANTS (requirement 3): earlier legal completed trade has
  temporal priority; one open position only; Exit_i < Entry_{i+1}; cursor
  restarts at exit+1; same structure may re-candidate later; no unit/day
  holding cap after entry; first TARGET_TOUCH is terminal.
"""

import pytest

from research.liquidity_oracle_atlas.structural_god_oracle_m15_v4 import (
    R_TARGET,
    run_god_oracle_v4,
)

DIAG = "SR|m15|0|346|7705.0|7694.0|75.0"
PREEMPT = "SR|m15|0|346|7691.0|7681.0|72.0"


@pytest.fixture(scope="module")
def ag_full_stream():
    """Run the full AG sequential oracle ONCE and return (canon, mv)."""
    res = run_god_oracle_v4("AG", max_bars=None)
    canon = [r for r in res["records"] if r["canonical_oracle_trade"]]
    return canon, res["market_view"]


def test_ag_trade2_diag_not_full_stream_invariant(ag_full_stream):
    """FIX-02 D-update: the diagnosed 349/350/7700/7773 case must NOT appear in
    the full AG sequential canonical stream.

    After Block B removes the hidden holding cap, the earlier legal PREEMPT
    structure (bar 346) completes first and consumes the global cursor before
    the DIAG structure (bar 348) is ever reached. This is the CORRECT, expected
    consequence of the fix -- not a regression. The DIAG case survives only as
    the local best-entry regression in test_structural_god_oracle_m15_v4.py.
    """
    canon, _mv = ag_full_stream
    diag = [r for r in canon if r["structure_id"] == DIAG]
    prec = [r for r in canon if r["structure_id"] == PREEMPT]
    assert not diag, (
        "diagnosed trade MUST NOT be a full-stream invariant after Block B "
        "(an earlier PREEMPT trade should have consumed the cursor)"
    )
    assert prec, (
        "the earlier PREEMPT trade at bar 346 must complete and advance the "
        "cursor past the DIAG opportunity"
    )


def test_full_stream_sequential_invariants(ag_full_stream):
    """Requirement 3: full-stream invariants of the sequential one-position
    trade stream."""
    canon, mv = ag_full_stream
    seq = sorted(canon, key=lambda r: r["best_entry_decision_index"])
    assert seq, "expected some canonical trades in the full stream"

    # one open position only / no overlap / earlier trade has temporal priority
    for i in range(len(seq) - 1):
        assert seq[i]["exit_fill_index"] < seq[i + 1]["best_entry_decision_index"], (
            f"trades overlap at i={i}: exit={seq[i]['exit_fill_index']} "
            f"next_entry={seq[i + 1]['best_entry_decision_index']}"
        )

    # cursor restarts at exit+1 (no gap, no overlap)
    for i in range(len(seq) - 1):
        assert seq[i]["exit_fill_index"] + 1 <= seq[i + 1]["best_entry_decision_index"], (
            f"cursor did not restart at exit+1 at i={i}"
        )

    # first TARGET_TOUCH is terminal: every canonical trade exits on TARGET_TOUCH
    assert all(r["exit_reason"] == R_TARGET for r in seq), (
        "a canonical trade did not terminate on first TARGET_TOUCH"
    )

    # same structure may re-candidate later (no success-blacklist)
    from collections import Counter
    cnt = Counter(r["structure_id"] for r in seq)
    assert any(c >= 2 for c in cnt.values()), (
        "no structure ever re-candidates after a completed trade "
        "(a permanent success skip set may have been reintroduced)"
    )

    # no unit/day holding cap after entry: at least one trade spans an
    # intraday-unit boundary (holding window reaches beyond the entry unit).
    spans_unit = any(
        mv.unit_end(int(r["best_entry_decision_index"])) < int(r["exit_fill_index"])
        for r in seq
    )
    assert spans_unit, (
        "no trade spans an intraday unit boundary -- the hidden holding cap "
        "may still be present after entry"
    )
