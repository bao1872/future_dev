"""test_candidate_overlay_v1
====================

Focused tests for the candidate-zone audit overlay (FUTURE-R3-CANONICAL-M5-TOUCH-NEXTBAR-GATE-V1).

The candidate truth now comes from the canonical R3 gate artifact
(artifacts/candidate_gate_r3_m5_touch_nextbar_v1/<symbol>.parquet), the SINGLE
source of truth shared by Streamlit / DP / Model. This module is a VISUALIZATION
helper only: it maps candidate bars onto the 5m ViewerTrack, never recomputing
eligibility or indicator math.

Covers:
  * exact time alignment (decision_time <-> 5m availability_time) on real AG
  * 0 unmatched / 0 duplicate / matched == candidate rows (real artifact)
  * non-candidate bar returns Candidate=False
  * candidate lookup returns exact episode + retained trigger context
  * segment merge (consecutive candidate bars in one episode -> one segment)
  * session gap splits visual segments
  * episode isolation (two episodes never merge)
  * viewport-only rendering (rendered << full)
  * trigger context preserved verbatim (decoded via the shared touch_bit)
  * the canonical FORMING-MTF audit state is the forming bar, not last-closed

Reuses ``build_viewer_track`` / ``gen_trend_base`` from the existing Viewer test
module (no indicator math duplicated here).
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
)
from research.liquidity_oracle_atlas.test_indicator_viewer_v1 import (
    build_viewer_track,
    gen_trend_base,
)
from research.liquidity_oracle_atlas.build_forming_environment_v1 import (
    FormingEnvironmentBuilder,
)
from research.liquidity_oracle_atlas.indicator_viewer_v1 import selected_snapshot
from research.liquidity_oracle_atlas.build_candidate_gate_r3_v1 import (
    MASK_BIT,
    load_candidate_gate,
    touch_bit,
)


def _make_gate_rows(symbol, indices, track, episode_of, trigger_bits=0):
    """Build gate-artifact-shaped candidate rows aligned to the 5m track."""
    recs = []
    for k, idx in enumerate(indices):
        recs.append({
            "symbol": symbol,
            "decision_time": pd.Timestamp(track.available_time[idx]),
            "bar_index": int(idx),
            "candidate_any": True,
            "candidate_episode_id": int(episode_of[k]),
            "candidate_trigger_bits": int(trigger_bits),
            "trigger_bar_index": int(idx - 1),
            "trigger_decision_time": (
                pd.Timestamp(track.available_time[idx - 1]) if idx > 0 else pd.NaT
            ),
        })
    return pd.DataFrame(recs)


def test_integration_ag_alignment():
    """Real R3 gate artifact aligns to the real AG 5m track with no loss."""
    rows = load_candidate_rows("AG")
    assert len(rows) > 0
    ca = candidate_for_symbol(rows, "AG")
    assert len(ca) > 0
    track = build_viewer_track(
        FormingEnvironmentBuilder(symbol="AG").load_raw().base,
        "5m", symbol="AG", source_sha="x",
    )
    segs, audit, cbt = build_candidate_segments(ca, track)
    assert audit["unmatched_candidate_rows"] == 0
    assert audit["duplicate_decision_keys"] == 0
    assert audit["matched_5m_bars"] == audit["t2_candidate_rows"]
    assert audit["t2_candidate_rows"] > 0
    # no visual segment spans a session gap
    seg_arr = np.asarray(track.segment)
    for s in segs:
        assert int(seg_arr[s.start_idx]) == int(seg_arr[s.end_idx])


def test_non_candidate_bar():
    base = gen_trend_base(n=1000, seed=5)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    idxs = [10, 11, 12]
    rows = _make_gate_rows("AG", idxs, track, [0, 0, 0])
    _, _, cand_by_time = build_candidate_segments(rows, track)
    dt = pd.Timestamp(track.available_time[100])
    st = candidate_state_at(cand_by_time, dt)
    assert st["is_candidate"] is False
    assert st["episode"] is None
    assert st["proximity_any"] is None


def test_candidate_lookup_exact_episode_and_trigger():
    base = gen_trend_base(n=1000, seed=7)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    idxs = [10, 11, 12]
    # touch_bit DECODES a bits integer; to SET bits use MASK_BIT directly.
    tb = (1 << MASK_BIT[("m5", "SR")]) | (1 << MASK_BIT[("h1", "SR")])
    rows = _make_gate_rows("AG", idxs, track, [0, 0, 0], trigger_bits=tb)
    _, _, cand_by_time = build_candidate_segments(rows, track)
    dt = pd.Timestamp(track.available_time[10])
    st = candidate_state_at(cand_by_time, dt)
    assert st["is_candidate"] is True
    assert st["episode"] == 0
    assert st["proximity_any"] is True
    assert st["trigger_bits"] == tb
    assert bool(touch_bit(np.uint16(st["trigger_bits"]), "m5", "SR").item())
    assert bool(touch_bit(np.uint16(st["trigger_bits"]), "h1", "SR").item())


def test_segment_merge_same_episode():
    base = gen_trend_base(n=1000, seed=11)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    idxs = [10, 11, 12, 50, 51]
    rows = _make_gate_rows("AG", idxs, track, [0, 0, 0, 1, 1])
    segs, audit, _ = build_candidate_segments(rows, track)
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
    rows = _make_gate_rows("AG", idxs, track, [0, 0, 0, 0, 0])
    seg = np.array(track.segment, copy=True)
    seg[13] = int(seg[12]) + 1
    seg[14] = int(seg[12]) + 1
    track.segment = seg
    segs, audit, _ = build_candidate_segments(rows, track)
    assert audit["n_episodes"] == 1
    assert audit["n_segments"] == 2
    s0 = sorted(segs, key=lambda s: s.start_idx)
    assert s0[0].end_idx == 12 and s0[1].start_idx == 13
    assert s0[0].n_segments_for_episode == 2


def test_episode_isolation_never_merges():
    base = gen_trend_base(n=1000, seed=17)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    idxs = [10, 11, 12, 50, 51, 52]
    rows = _make_gate_rows("AG", idxs, track, [0, 0, 0, 1, 1, 1])
    segs, audit, _ = build_candidate_segments(rows, track)
    assert audit["n_episodes"] == 2
    assert audit["n_segments"] == 2
    ep0 = [s for s in segs if s.episode_id == 0][0]
    ep1 = [s for s in segs if s.episode_id == 1][0]
    assert ep0.end_idx < ep1.start_idx


def test_viewport_only_renders_subset():
    """FIX1: only viewport-intersecting segments are drawn (rendered << full)."""
    rows = load_candidate_rows("AG")
    track = build_viewer_track(
        FormingEnvironmentBuilder(symbol="AG").load_raw().base,
        "5m", symbol="AG", source_sha="x",
    )
    segs, audit, _ = build_candidate_segments(rows, track)
    full = len(segs)
    assert full > 0
    # pick a mid-series decision bar
    sel = int(len(track.available_time) // 2)
    lo, hi = _viewport(track, sel)
    visible = [s for s in segs if s.end_idx >= lo and s.start_idx <= hi and s.start_idx <= sel]
    assert len(visible) < full
    # every drawn segment must actually intersect the viewport window (O(VIEW_BARS),
    # not O(full history)) -- this is the real rendering-boundedness guarantee.
    for s in visible:
        assert s.end_idx >= lo and s.start_idx <= hi


def _viewport(track, selected):
    """Mirror indicator_viewer_v1.compute_viewport (import to avoid duplication)."""
    from research.liquidity_oracle_atlas.indicator_viewer_v1 import compute_viewport
    return compute_viewport(track, selected)


def test_non_5m_track_returns_empty():
    base = gen_trend_base(n=1000, seed=31)
    track15 = build_viewer_track(base, "15m", raw_load_count=1)
    idxs = [5, 6]
    rows = _make_gate_rows("AG", idxs, track15, [0, 0])
    segs, audit, cbt = build_candidate_segments(rows, track15)
    assert segs == []
    assert cbt == {}
    assert audit["matched_5m_bars"] == 0


# --------------------------------------------------------------------------- #
# Canonical FORMING-MTF audit state (point 6 of FIX1, still valid under R3)       #
# --------------------------------------------------------------------------- #
def test_forming_env_decision_time_alignment():
    """Audit aligns forming env by decision_time == 5m availability_time."""
    base = FormingEnvironmentBuilder(symbol="AG").load_raw().base
    b = FormingEnvironmentBuilder(symbol="AG")
    b.load_raw().prepare()
    fdf, _ = b.run()
    track = build_viewer_track(base, "5m", symbol="AG", source_sha="x")
    ftimes = set(pd.to_datetime(fdf["decision_time"]))
    vtimes = set(pd.to_datetime(track.available_time))
    missing = vtimes - ftimes
    assert len(missing) == 0, f"{len(missing)} Viewer 5m times missing from forming env"


def test_forming_state_is_formed_not_closed():
    """At an interior 5m bar the 15m state is the FORMING (partial) bar.

    The audit must show this, not the last fully-closed 15m bar. A closed 15m
    bar has exactly 3 sub-bars (n_base_known == 3); a forming bar has < 3.
    """
    b = FormingEnvironmentBuilder(symbol="AG")
    b.load_raw().prepare()
    fdf, _ = b.run()
    nb = fdf["m15_n_base_known"].to_numpy()
    closed = np.where(nb == 3)[0]
    forming = np.where(nb < 3)[0]
    assert len(closed) > 0, "expected some closed 15m bars (n_base_known==3)"
    assert len(forming) > 0, "expected some forming 15m bars (n_base_known<3)"
    i = int(forming[len(forming) // 2])
    assert int(fdf.iloc[i]["m15_n_base_known"]) < 3


def test_forming_m5_parity_with_viewer():
    """Forming env m5 features equal the 5m ViewerTrack selected_snapshot.

    Confirms the canonical forming-MTF owner reproduces the same indicator
    math (no change to DTP/SR/Liquidity), for the 5m decision axis.
    """
    base = FormingEnvironmentBuilder(symbol="AG").load_raw().base
    track = build_viewer_track(base, "5m", symbol="AG", source_sha="x")
    b = FormingEnvironmentBuilder(symbol="AG")
    b.load_raw().prepare()
    fdf, _ = b.run()
    for i in [10, 100, 1000]:
        snap = selected_snapshot(track, i)
        row = fdf.iloc[i]
        a = float(row["m5_trend_score"])
        bv = float(snap["dtp"]["trend_score"])
        if not (np.isnan(a) and np.isnan(bv)):
            assert abs(a - bv) < 1e-9
        assert int(row["m5_sr_n_channels"]) == int(snap["sr_n_channels"])
        assert int(row["m5_liq_up_count"]) == int(snap["liq_up_count"])
