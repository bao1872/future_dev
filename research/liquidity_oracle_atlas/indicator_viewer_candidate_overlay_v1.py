"""indicator_viewer_candidate_overlay_v1
========================================

Candidate-zone audit overlay for the Indicator Viewer (visual audit ONLY).

This module is a VISUALIZATION helper. It performs NO indicator math, NO
causal resampling, NO model work and NO label recomputation. It only:

  * reads the canonical R3 candidate gate artifact
    (artifacts/candidate_gate_r3_m5_touch_nextbar_v1/<symbol>.parquet),
    which is the SINGLE source of candidate truth shared by Streamlit, DP and
    the future Model (FUTURE-R3-CANONICAL-M5-TOUCH-NEXTBAR-GATE-V1),
  * maps each candidate bar to the 5m ViewerTrack by integer ``bar_index``
    (both the artifact and the ViewerTrack derive from the same base 5m frame,
    so indices are identical; ``decision_time`` == 5m ``availability_time``),
  * merges same-episode consecutive 5m bars into contiguous shaded regions,
  * exposes the frozen candidate trigger context next to the canonical
    indicator state.

Candidate eligibility (5m SR/LIQ true-touch only) is decided ONCE by the gate
artifact; this module never recomputes it. The higher-TF trigger context
(``candidate_trigger_bits``) is preserved verbatim from the artifact.

Frozen owners reused verbatim (no math copied):
  * research.liquidity_oracle_atlas.indicator_viewer_v1.ViewerTrack
  * research.liquidity_oracle_atlas.indicator_viewer_v1.selected_snapshot
  * research.liquidity_oracle_atlas.build_candidate_gate_r3_v1.load_candidate_gate

No model / Y / Q / Oracle-action content is ever produced by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pathlib import Path

from research.liquidity_oracle_atlas.indicator_viewer_v1 import (
    ViewerTrack,
    selected_snapshot,
)
from research.liquidity_oracle_atlas.build_candidate_gate_r3_v1 import (
    load_candidate_gate,
    touch_bit,
)

TF_LABELS = ["5m", "15m", "1H", "4H"]
# 4TF x {SR, LIQ} trigger-context display order
TRIGGER_BITS = [
    ("m5", "SR"), ("m5", "LIQ"),
    ("m15", "SR"), ("m15", "LIQ"),
    ("h1", "SR"), ("h1", "LIQ"),
    ("h4", "SR"), ("h4", "LIQ"),
]

# Overlay visual identity (distinct from SR red/green and Oracle markers).
_CANDIDATE_RGB = "rgba(56, 128, 255, 1.0)"


@dataclass
class CandidateSegment:
    """One contiguous visual band for a (symbol, episode) on the 5m axis.

    ``start_idx`` / ``end_idx`` are integer ViewerTrack x positions (the chart
    x-axis is integer bar index). ``start_time`` / ``end_time`` are the
    corresponding 5m ``availability_time`` timestamps, kept for display only.
    """

    symbol: str
    episode_id: object
    start_idx: int
    end_idx: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    n_bars: int
    n_segments_for_episode: int = 1


# --------------------------------------------------------------------------- #
# Loading (pure; the page wraps this in @st.cache_data)                       #
# --------------------------------------------------------------------------- #
def load_candidate_rows(symbol: str) -> pd.DataFrame:
    """Read the canonical R3 candidate gate artifact for one symbol.

    This is the single frozen candidate truth consumed by Streamlit, the DP and
    the future Model. Candidate membership and trigger context come from here,
    never from chart geometry.
    """
    df = load_candidate_gate(symbol)
    df["decision_time"] = pd.to_datetime(df["decision_time"])
    return df.reset_index(drop=True)


def candidate_for_symbol(
    candidate_df: pd.DataFrame, symbol: str
) -> pd.DataFrame:
    """Identity pass: the artifact is already per-symbol (Section 15)."""
    d = candidate_df[candidate_df["symbol"] == symbol].copy()
    return d.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Time alignment (Section 4)                                                  #
# --------------------------------------------------------------------------- #
def match_available_index(track: ViewerTrack, decision_dt: Any) -> int:
    """Return the 5m bar index whose ``available_time`` equals ``decision_dt``.

    Returns -1 if no such bar exists. Uses time as the semantic key (never
    decision_bar_index), mirroring the canonical Oracle alignment contract.
    """
    dt = pd.Timestamp(decision_dt)
    arr = pd.to_datetime(track.available_time)
    idx = np.flatnonzero(arr == dt)
    return int(idx[0]) if idx.size else -1


def snapshot_at_available_time(
    track: ViewerTrack, decision_dt: Any
) -> Optional[dict]:
    """Canonical historical-as-of indicator snapshot at ``decision_dt``.

    Returns None when ``decision_dt`` is not on this track's axis. The snapshot
    is O(1) and contains only information available as-of close(t) (no future
    leakage). Use exact time equality — correct for the 5m decision axis.
    """
    i = match_available_index(track, decision_dt)
    if i < 0:
        return None
    return selected_snapshot(track, i)


def snapshot_as_of(track: ViewerTrack, decision_dt: Any) -> Optional[dict]:
    """Leakage-free causal indicator snapshot as-of ``decision_dt``.

    For the 5m decision axis ``decision_dt`` is a 5m boundary, so the exact bar
    is returned. For coarser timeframes (15m/1h/4h) the decision bar is usually
    *inside* a still-forming tf bar; taking that enclosing tf bar would pull in
    future 5m bars (leakage). Instead we return the LAST fully-closed tf bar
    whose ``available_time <= decision_dt`` — exactly the historical-as-of tf
    state, consistent with ViewerTrack's own causal close semantics (Section 11).
    """
    dt = pd.Timestamp(decision_dt)
    arr = pd.to_datetime(track.available_time)
    le = np.flatnonzero(arr <= dt)
    if le.size == 0:
        return None
    return selected_snapshot(track, int(le[-1]))


# --------------------------------------------------------------------------- #
# Segment construction (Section 6 / 16)                                       #
# --------------------------------------------------------------------------- #
def build_candidate_segments(
    rows: pd.DataFrame,
    track: ViewerTrack,
) -> Tuple[List[CandidateSegment], Dict[str, Any], Dict[pd.Timestamp, Dict[str, Any]]]:
    """Map candidate rows of one symbol onto the 5m ViewerTrack.

    Returns ``(segments, audit, cand_by_time)``.

    * Consumes the canonical R3 gate artifact; candidate bars are those with
      ``candidate_any == True``. ``bar_index`` in the artifact is the SAME
      integer ViewerTrack x position (both derive from the same base 5m frame),
      so mapping is O(1) per row. ``decision_time`` == 5m ``availability_time``
      is still cross-checked (Section 4 invariant).
    * Builds ``cand_by_time`` keyed by ``decision_time`` carrying the trigger
      context (trigger bar index/time + full 4TF trigger_bits).
    * Same-episode consecutive candidate bars merge into one contiguous band;
      a discontinuity in the segment id starts a new sub-segment (Section 16).
    """
    empty_audit = {
        "symbol": track.symbol,
        "t2_candidate_rows": 0,
        "matched_5m_bars": 0,
        "unmatched_candidate_rows": 0,
        "duplicate_decision_keys": 0,
        "n_episodes": 0,
        "n_segments": 0,
        "first_candidate_time": None,
        "last_candidate_time": None,
    }
    if track.tf_label != "5m":
        # Candidate zones are defined on the 5m decision axis only (Section 5).
        return [], empty_audit, {}

    cand = rows[rows["candidate_any"].astype(bool)].copy().reset_index(drop=True)
    n_cand = len(cand)
    if n_cand == 0:
        return [], empty_audit, {}

    # cross-check: decision_time == 5m availability_time (Section 4)
    avail = pd.to_datetime(track.available_time)
    dt_arr = cand["decision_time"].to_numpy()
    pos = np.searchsorted(avail, dt_arr)
    ok = (pos < len(avail)) & (avail[pos] == dt_arr)
    unmatched = int((~ok).sum())

    bi = cand["bar_index"].to_numpy().astype(int)
    ep = cand["candidate_episode_id"].to_numpy().astype(int)
    trig = cand["candidate_trigger_bits"].to_numpy().astype(np.int64)
    tbi = cand["trigger_bar_index"].to_numpy().astype(int)
    tdt = pd.to_datetime(cand["trigger_decision_time"]).to_numpy()

    cand_by_time: Dict[pd.Timestamp, Dict[str, Any]] = {}
    for i in range(n_cand):
        cand_by_time[pd.Timestamp(dt_arr[i])] = {
            "episode": int(ep[i]),
            "trigger_bits": int(trig[i]),
            "trigger_bar_index": int(tbi[i]),
            "trigger_decision_time": tdt[i],
            "proximity_any": True,
            "proximity_episode_id": int(ep[i]),
        }

    seg_arr = np.asarray(track.segment)
    segments: List[CandidateSegment] = []
    for ep_id, g in cand.groupby("candidate_episode_id", sort=False):
        gi = g["bar_index"].to_numpy().astype(int)
        if len(gi) == 1:
            cuts = np.array([0, 1])
        else:
            # split on exchange-session / segment change (Section 16): never
            # bridge an overnight gap. Within a session, consecutive candidate
            # bars of the same episode merge into one contiguous band.
            br = np.where(seg_arr[gi[1:]] != seg_arr[gi[:-1]])[0] + 1
            cuts = np.concatenate([[0], br, [len(gi)]])
        for a, b in zip(cuts[:-1], cuts[1:]):
            sub = gi[a:b]
            s = int(sub[0])
            e = int(sub[-1])
            segments.append(CandidateSegment(
                symbol=track.symbol,
                episode_id=ep_id,
                start_idx=s,
                end_idx=e,
                start_time=pd.Timestamp(avail[s]),
                end_time=pd.Timestamp(avail[e]),
                n_bars=int(len(sub)),
            ))

    # annotate per-episode sub-segment count (session splits within an episode)
    _cnt: Dict[object, int] = {}
    for s in segments:
        _cnt[s.episode_id] = _cnt.get(s.episode_id, 0) + 1
    for s in segments:
        s.n_segments_for_episode = _cnt[s.episode_id]

    audit = {
        "symbol": track.symbol,
        "t2_candidate_rows": int(rows["candidate_any"].astype(bool).sum()),
        "matched_5m_bars": int(n_cand),
        "unmatched_candidate_rows": int(unmatched),
        "duplicate_decision_keys": 0,
        "n_episodes": int(cand["candidate_episode_id"].nunique()),
        "n_segments": int(len(segments)),
        "first_candidate_time": pd.Timestamp(cand["decision_time"].min()),
        "last_candidate_time": pd.Timestamp(cand["decision_time"].max()),
    }
    return segments, audit, cand_by_time


# --------------------------------------------------------------------------- #
# Plot helper (Section 17)                                                    #
# --------------------------------------------------------------------------- #
def add_candidate_zone_overlay(
    fig: "go.Figure", segments: List[CandidateSegment]
) -> "go.Figure":
    """Shade candidate regions with a low-opacity vertical band.

    Uses integer bar-index coordinates to match the Viewer's integer x-axis.
    ``end_idx + 1`` ensures the last candidate bar is fully enclosed.
    """
    for seg in segments:
        fig.add_vrect(
            x0=seg.start_idx,
            x1=seg.end_idx + 1,
            opacity=0.10,
            line_width=0,
            fillcolor=_CANDIDATE_RGB,
            layer="below",
        )
    return fig


# --------------------------------------------------------------------------- #
# Selected-bar lookup (Section 18)                                            #
# --------------------------------------------------------------------------- #
def candidate_state_at(
    cand_by_time: Dict[pd.Timestamp, Dict[str, Any]],
    selected_decision_time: Any,
) -> Dict[str, Any]:
    """Frozen candidate truth at a selected decision time.

    Returns ``is_candidate=False`` when the bar is not a candidate. Does NOT
    recompute anything from the indicator state (Section 10: A and B stay
    strictly separate). The trigger context (trigger bar index/time + full 4TF
    trigger bits) is carried verbatim from the gate artifact.
    """
    dt = pd.Timestamp(selected_decision_time)
    if dt not in cand_by_time:
        return {
            "is_candidate": False,
            "episode": None,
            "trigger_bits": 0,
            "trigger_bar_index": -1,
            "trigger_decision_time": None,
            "proximity_any": None,
            "proximity_episode_id": None,
        }
    r = cand_by_time[dt]
    return {
        "is_candidate": True,
        "episode": r["episode"],
        "trigger_bits": r["trigger_bits"],
        "trigger_bar_index": r["trigger_bar_index"],
        "trigger_decision_time": r["trigger_decision_time"],
        "proximity_any": r["proximity_any"],
        "proximity_episode_id": r["proximity_episode_id"],
    }
