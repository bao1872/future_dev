"""Tests for the 15m port of the frozen 5m R2 one-entry-proximity DP.

Mechanical port contract (SHA cc7891723...): the ONLY substitutions are
5m->15m execution clock, TF universe 5m/15m/1h/4h -> 15m/1h/4h, ATR5m ->
ATR15m, +15min availability, 5m opens -> 15m opens. Everything else is
carried over EXACTLY. These tests lock the contract:

  * Bellman value == independent exhaustive DFS for ALL 6 (p, q) start states.
  * One new entry per continuous proximity episode; rearm only after leaving
    ALL proximity (no gap<=3 merging of separate runs).
  * Free exit / hold; reversal counts as a new entry.
  * (trading_day, segment) units with forced-flat terminals (no overnight).
  * Tie-breaking prefers keep current position -> flat -> fallback.
  * Real-data AG audit: all invariants 0, <=1 new entry per episode.
"""

import numpy as np

import research.liquidity_oracle_atlas.build_trade_oracle_dp_m15_one_entry_proximity_v1 as M


# --------------------------------------------------------------------------- #
# Bellman == DFS (independent exhaustive reference)                             #
# --------------------------------------------------------------------------- #
def test_bellman_equals_dfs_all_six_states():
    rng = np.random.default_rng(7)
    for _ in range(40):
        n = int(rng.integers(4, 12))
        opens = 100.0 + np.cumsum(rng.normal(0, 1, n + 2))
        prox = (rng.random(n) > 0.4)
        cost = np.zeros(n)
        core = M.solve_day_dp_v2(opens, prox, cost, 0, n)
        for sp, sq in [(0, 0), (0, 1), (-1, 0), (-1, 1), (1, 0), (1, 1)]:
            si = M.S2I[(sp, sq)]
            best, _ = M.exhaustive_reference_v2(
                opens, prox, cost, 0, n, start_pos=sp, start_armed=sq
            )
            av = int(core["actions"][0, si])
            a_idx = av + 1  # actions store the action VALUE (-1/0/1); Q index is value+1
            assert abs(best - core["Q"][0, si, a_idx]) < 1e-6, (
                f"V mismatch state ({sp},{sq}): dp={core['Q'][0, si, a_idx]:.6f} "
                f"dfs={best:.6f} (n={n})"
            )


# --------------------------------------------------------------------------- #
# Proximity episode semantics                                                  #
# --------------------------------------------------------------------------- #
def _walk(opens, prox, cost, s, e):
    core = M.solve_day_dp_v2(opens[s : e + 2], prox[s:e], cost[s:e], 0, e - s)
    return M._walk_unit_path(core, prox, s, e)


def test_one_entry_per_contiguous_episode():
    n = 12
    opens = np.arange(n + 2, dtype=float) + 100.0
    prox = np.ones(n, dtype=bool)  # single continuous episode
    path = _walk(opens, prox, np.zeros(n), 0, n)
    entries = [d for d in path if d["ne"]]
    assert len(entries) <= 1  # one-entry-per-episode quota


def test_gap_zero_splits_episodes_no_merge():
    n = 10
    opens = np.arange(n + 2, dtype=float) + 100.0
    prox = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1, 0], dtype=bool)
    ep = M.compute_proximity_episode_id(prox, np.array([0]), n)
    # episode 1 = bars 0..2 ; episode 2 = bars 5..8 ; P=0 bars get -1
    assert ep[0] == 1 and ep[2] == 1
    assert ep[5] == 2 and ep[8] == 2
    assert ep[3] == -1 and ep[4] == -1 and ep[9] == -1
    # each episode may own its own entry (no gap<=3 re-merge)
    path = _walk(opens, prox, np.zeros(n), 0, n)
    assert len([d for d in path if d["ne"]]) <= 2


def test_rearm_after_leaving_all_proximity():
    n = 12
    opens = np.arange(n + 2, dtype=float) + 100.0
    # two disjoint episodes -> up to two entries, second only after leaving all P
    prox = np.array([1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
    path = _walk(opens, prox, np.zeros(n), 0, n)
    entries = [d for d in path if d["ne"]]
    assert len(entries) <= 2
    # the second entry can only occur at/after bar 5 (after P left 0)
    if len(entries) == 2:
        assert entries[1]["t"] >= 5


def test_entry_requires_proximity_and_armed():
    n = 8
    opens = np.arange(n + 2, dtype=float) + 100.0
    prox = np.zeros(n, dtype=bool)  # no proximity anywhere
    path = _walk(opens, prox, np.zeros(n), 0, n)
    # with no proximity, no new entry (Flat->Long/Short) may ever occur
    assert all(not d["ne"] for d in path)


# --------------------------------------------------------------------------- #
# Exit / hold freedom + reversal-as-new-entry                                  #
# --------------------------------------------------------------------------- #
def test_free_exit_and_hold_are_base_allowed():
    # Action INDEX = value + 1: idx0=Short(-1), idx1=Flat(0), idx2=Long(+1).
    # Long(p=1) -> Flat(idx1) exit and Long -> Long(idx2) hold need no permission.
    assert bool(M.BASE_ALLOWED[M.S2I[(1, 1)], 1])  # Long -> Flat (exit)
    assert bool(M.BASE_ALLOWED[M.S2I[(1, 1)], 2])  # Long -> Long (hold)
    # Short(p=-1) -> Flat(idx1) exit and Short -> Short(idx0) hold need no permission.
    assert bool(M.BASE_ALLOWED[M.S2I[(-1, 1)], 1])  # Short -> Flat (exit)
    assert bool(M.BASE_ALLOWED[M.S2I[(-1, 1)], 0])  # Short -> Short (hold)


def test_reversal_is_a_new_entry():
    # Long(p=1) -> Short(idx0, a=-1) is a NEW_ENTRY (reversal: consumes the right,
    # needs proximity + armed). Holding (idx2) / exiting (idx1) are NOT new entries.
    assert bool(M.NEW_ENTRY[M.S2I[(1, 1)], 0])  # Long -> Short reversal
    # Flat -> Short / Flat -> Long are new entries.
    assert bool(M.NEW_ENTRY[M.S2I[(0, 1)], 0])  # Flat -> Short
    assert bool(M.NEW_ENTRY[M.S2I[(0, 1)], 2])  # Flat -> Long


# --------------------------------------------------------------------------- #
# Forced-flat unit terminals (no overnight)                                    #
# --------------------------------------------------------------------------- #
def test_forced_flat_terminal():
    n = 9
    opens = np.arange(n + 2, dtype=float) + 100.0
    for prox in (np.zeros(n, dtype=bool), np.ones(n, dtype=bool)):
        path = _walk(opens, prox, np.zeros(n), 0, n)
        assert path[-1]["pa"] == 0  # terminal decision is forced Flat


# --------------------------------------------------------------------------- #
# Tie-breaking                                                                 #
# --------------------------------------------------------------------------- #
def test_tie_break_keep_then_flat():
    Q = np.full((6, 3), 5.0)  # all actions tie
    chosen, amb, edge, vmax = M.choose_actions_v2(Q)
    for si in range(6):
        p = int(M.STATE_POS[si])
        assert chosen[si] == p  # keep current position on a full tie
    assert bool(amb.all())


def test_tie_break_flat_preferred_over_opposite():
    # For a Flat-state row, if Flat and Long both tie at the top, Flat wins
    # (keep current position = Flat).
    Q = np.zeros((6, 3))
    Q[M.S2I[(0, 1)], :] = [0.0, 5.0, 5.0]  # Short=0, Flat=5, Long=5 -> tie Flat/Long
    chosen, amb, edge, vmax = M.choose_actions_v2(Q)
    assert chosen[M.S2I[(0, 1)]] == 0  # Flat preferred over Long on tie


# --------------------------------------------------------------------------- #
# Real-data AG audit (no R4 Candidate fields referenced)                       #
# --------------------------------------------------------------------------- #
def test_ag_audit_invariants_clean():
    r = M.run_dp_m15_one_entry_proximity("AG", max_bars=2000)
    inv = M.check_oracle_invariants(r)
    assert inv["new_entry_outside_proximity"] == 0
    assert inv["new_entry_armed_zero"] == 0
    assert inv["illegal_reversal"] == 0
    assert inv["cross_day"] == 0
    assert inv["cross_segment"] == 0
    assert inv["nonflat_terminal"] == 0
    assert inv["pnl_mismatch_flag"] == 0
    assert inv["max_new_entries_per_episode"] <= 1
    assert inv["pnl_abs_diff"] < 1e-6
    # DP proximity is structurally independent of the R4 candidate gate
    assert "candidate_any" not in r
    assert "merged_episode_id" not in r
