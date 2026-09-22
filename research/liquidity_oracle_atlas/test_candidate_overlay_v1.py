"""test_candidate_overlay_v1
====================

Focused tests for the candidate-zone audit overlay (Section 20).

Covers:
  * exact time alignment (decision_time <-> 5m availability_time)
  * non-candidate bar returns Candidate=False
  * candidate lookup returns exact episode
  * segment merge (consecutive candidate bars in one episode -> one segment)
  * session gap splits visual segments
  * episode isolation (two episodes never merge)
  * duplicate candidate key hard fail
  * selected-time clipping (no future indicator state leaks)
  * integration: real AG candidate rows align to the real 5m track with
    0 unmatched / 0 duplicate / matched == candidate rows.

Reuses synthetic-base helpers and ``build_viewer_track`` from the existing
Viewer test module (no indicator math duplicated here).
"""

import numpy as np
import pandas as pd
import pytest

from research.liquidity_oracle_atlas.indicator_viewer_candidate_overlay_v1 import (
    build_candidate_segments,
    candidate_for_symbol,
    candidate_state_at,
    load_candidate_rows,
    match_available_index,
    snapshot_as_of,
)
from research.liquidity_oracle_atlas.test_indicator_viewer_v1 import (
    build_viewer_track,
    gen_trend_base,
)


def _make_rows(symbol, indices, track, episode_of, prox_any=True, prox_ep=100,
               dup=False):
    """Build candidate rows whose decision_time == track.available_time[idx]."""
    recs = []
    for k, idx in enumerate(indices):
        recs.append({
            "symbol": symbol,
            "decision_time": pd.Timestamp(track.available_time[idx]),
            "global_episode": episode_of[k],
            "proximity_any": prox_any,
            "proximity_episode_id": prox_ep,
            "is_candidate": True,
        })
    df = pd.DataFrame(recs)
    if dup:
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    return df


def test_exact_time_alignment():
    base = gen_trend_base(n=1000, seed=3)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    idxs = [10, 11, 12, 50, 51]
    rows = _make_rows("AG", idxs, track, [0, 0, 0, 1, 1])
    segs, audit, _ = build_candidate_segments(rows, track)
    assert audit["t2_candidate_rows"] == 5
    assert audit["matched_5m_bars"] == 5
    assert audit["unmatched_candidate_rows"] == 0
    assert audit["duplicate_decision_keys"] == 0
    # decision_time maps exactly to its 5m availability_time
    for i in idxs:
        dt = pd.Timestamp(track.available_time[i])
        assert match_available_index(track, dt) == i
        # availability_time at that index equals the decision_time
        assert pd.Timestamp(track.available_time[i]) == dt


def test_non_candidate_bar():
    base = gen_trend_base(n=1000, seed=5)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    idxs = [10, 11, 12]
    rows = _make_rows("AG", idxs, track, [0, 0, 0])
    _, _, cand_by_time = build_candidate_segments(rows, track)
    # a bar that is not in cand_by_time
    dt = pd.Timestamp(track.available_time[100])
    st = candidate_state_at(cand_by_time, dt)
    assert st["is_candidate"] is False
    assert st["episode"] is None
    assert st["proximity_any"] is None


def test_candidate_lookup_exact_episode():
    base = gen_trend_base(n=1000, seed=7)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    idxs = [10, 11, 12]
    rows = _make_rows("AG", idxs, track, [0, 0, 0])
    _, _, cand_by_time = build_candidate_segments(rows, track)
    dt = pd.Timestamp(track.available_time[10])
    st = candidate_state_at(cand_by_time, dt)
    assert st["is_candidate"] is True
    assert st["episode"] == 0
    assert st["proximity_any"] is True


def test_segment_merge_same_episode():
    base = gen_trend_base(n=1000, seed=11)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    idxs = [10, 11, 12, 50, 51]
    rows = _make_rows("AG", idxs, track, [0, 0, 0, 1, 1])
    segs, audit, _ = build_candidate_segments(rows, track)
    # 2 episodes -> 2 segments (each episode's bars are consecutive)
    assert audit["n_segments"] == 2
    assert audit["n_episodes"] == 2
    ep0 = [s for s in segs if s.episode_id == 0][0]
    assert ep0.start_idx == 10 and ep0.end_idx == 12 and ep0.n_bars == 3
    ep1 = [s for s in segs if s.episode_id == 1][0]
    assert ep1.start_idx == 50 and ep1.end_idx == 51 and ep1.n_bars == 2


def test_session_gap_splits_segments():
    base = gen_trend_base(n=1000, seed=13)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    idxs = [10, 11, 12, 13, 14]
    rows = _make_rows("AG", idxs, track, [0, 0, 0, 0, 0])
    # simulate an exchange-session gap between bar 12 and 13 (segment array is
    # read-only in place, so replace with a writable copy)
    seg = np.array(track.segment, copy=True)
    seg[13] = int(seg[12]) + 1
    seg[14] = int(seg[12]) + 1
    track.segment = seg
    segs, audit, _ = build_candidate_segments(rows, track)
    # one episode but split across the gap -> 2 visual segments
    assert audit["n_episodes"] == 1
    assert audit["n_segments"] == 2
    s0 = sorted(segs, key=lambda s: s.start_idx)
    assert s0[0].end_idx == 12 and s0[1].start_idx == 13
    assert s0[0].n_segments_for_episode == 2


def test_episode_isolation_never_merges():
    base = gen_trend_base(n=1000, seed=17)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    # two episodes, each contiguous; they must never merge into one segment
    idxs = [10, 11, 12, 50, 51, 52]
    rows = _make_rows("AG", idxs, track, [0, 0, 0, 1, 1, 1])
    segs, audit, _ = build_candidate_segments(rows, track)
    assert audit["n_episodes"] == 2
    assert audit["n_segments"] == 2
    ep0 = [s for s in segs if s.episode_id == 0][0]
    ep1 = [s for s in segs if s.episode_id == 1][0]
    # episodes never mixed: ep0 fully precedes ep1
    assert ep0.end_idx < ep1.start_idx


def test_duplicate_candidate_key_hard_fail():
    base = gen_trend_base(n=1000, seed=23)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    idxs = [10, 11, 12]
    rows = _make_rows("AG", idxs, track, [0, 0, 0], dup=True)
    with pytest.raises(RuntimeError):
        build_candidate_segments(rows, track)


def test_selected_time_clipping_no_future_leak():
    base = gen_trend_base(n=1000, seed=29)
    track5 = build_viewer_track(base, "5m", raw_load_count=1)
    track15 = build_viewer_track(base, "15m", raw_load_count=1)
    # decision time on the 5m axis
    dt = pd.Timestamp(track5.available_time[100])
    snap = snapshot_as_of(track15, dt)
    assert snap is not None
    # leakage-free: the chosen tf bar must be fully closed as-of dt
    assert pd.Timestamp(snap["available_time"]) <= dt


def test_non_5m_track_returns_empty():
    base = gen_trend_base(n=1000, seed=31)
    track15 = build_viewer_track(base, "15m", raw_load_count=1)
    idxs = [5, 6]
    rows = _make_rows("AG", idxs, track15, [0, 0])
    segs, audit, cbt = build_candidate_segments(rows, track15)
    assert segs == []
    assert cbt == {}
    assert audit["matched_5m_bars"] == 0


def test_integration_ag_alignment():
    """Real T2 candidate truth aligns to the real AG 5m track with no loss."""
    t2 = load_candidate_rows()
    ca = candidate_for_symbol(t2, "AG")
    assert len(ca) > 0
    from research.liquidity_oracle_atlas.build_forming_environment_v1 import (
        FormingEnvironmentBuilder,
    )
    b = FormingEnvironmentBuilder(symbol="AG")
    b.load_raw()
    track = build_viewer_track(b.base, "5m", symbol="AG", source_sha="x")
    segs, audit, cbt = build_candidate_segments(ca, track)
    assert audit["unmatched_candidate_rows"] == 0
    assert audit["duplicate_decision_keys"] == 0
    assert audit["matched_5m_bars"] == audit["t2_candidate_rows"]
    # no (start,end) segment spans a session gap incorrectly
    seg_arr = track.segment
    for s in segs:
        for j in range(s.start_idx, s.end_idx + 1):
            assert int(seg_arr[j]) == int(seg_arr[s.start_idx])
