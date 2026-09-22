"""indicator_viewer_candidate_overlay_v1
========================================

Candidate-zone audit overlay for the Indicator Viewer (visual audit ONLY).

This module is a VISUALIZATION helper. It performs NO indicator math, NO
causal resampling, NO model work and NO label recomputation. It only:

  * reads the frozen formal T2 candidate truth
    (artifacts/intraday_entry_value_tree_core108_v1/t2/t2_dataset.parquet),
  * maps each candidate ``decision_time`` to the 5m bar whose ``available_time``
    equals it (Section 4 hard rule: decision_time <-> 5m availability_time,
    one-to-one),
  * merges same-episode consecutive 5m bars into contiguous shaded regions,
  * exposes the frozen candidate truth next to the canonical indicator state.

Source of truth for candidate membership is the formal T2 parquet, which
persists exactly: ``is_candidate`` / ``proximity_any`` /
``proximity_episode_id`` / ``global_episode`` / ``decision_time``.  The
per-timeframe SR/LIQ ``proximity_bits`` bitmask is NOT persisted in that
parquet, so finer provenance is intentionally NOT reconstructed here (Section 9:
"do not manufacture finer provenance").

Frozen owners reused verbatim (no math copied):
  * research.liquidity_oracle_atlas.indicator_viewer_v1.ViewerTrack
  * research.liquidity_oracle_atlas.indicator_viewer_v1.selected_snapshot

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

# --------------------------------------------------------------------------- #
# Frozen source of truth                                                      #
# --------------------------------------------------------------------------- #
T2_PATH = Path(
    "artifacts/intraday_entry_value_tree_core108_v1/"
    "t2/t2_dataset.parquet"
)

CANDIDATE_COLS = [
    "symbol",
    "decision_time",
    "global_episode",
    "proximity_any",
    "proximity_episode_id",
    "is_candidate",
]

TF_LABELS = ["5m", "15m", "1H", "4H"]

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
def load_candidate_rows() -> pd.DataFrame:
    """Read only candidate-truth columns from the formal T2 parquet.

    The formal T2 parquet is candidate-only (``is_candidate`` is True for every
    row), so the filter is a no-op guard that also documents the hard rule from
    Section 3: candidate membership comes from frozen data, never from chart
    geometry.
    """
    df = pd.read_parquet(T2_PATH, columns=list(CANDIDATE_COLS))
    df["decision_time"] = pd.to_datetime(df["decision_time"])
    if "is_candidate" in df.columns:
        df = df[df["is_candidate"]].copy()
    return df.reset_index(drop=True)


def candidate_for_symbol(
    candidate_df: pd.DataFrame, symbol: str
) -> pd.DataFrame:
    """Filter + sort candidate rows for one symbol (Section 15)."""
    d = candidate_df[candidate_df["symbol"] == symbol].copy()
    return d.sort_values("decision_time").reset_index(drop=True)


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

    * Aligns every ``decision_time`` to the 5m bar via ``available_time``
      (Section 4). Unmatched candidate rows are counted, never silently
      dropped.
    * Builds ``cand_by_time`` keyed by ``decision_time``; a duplicate key is a
      hard error (Section 18). Values are compact dicts (episode + proximity
      flags only) so the result stays cheap to cache/serialize.
    * Per episode, consecutive candidate bars are merged into one contiguous
      band; a discontinuity in the session/segment id (which always
      accompanies an exchange-session / overnight gap) starts a new
      sub-segment (Section 16).
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

    dt_arr = rows["decision_time"].to_numpy()
    # duplicate decision_time is a hard error (Section 18)
    if len(np.unique(dt_arr)) != len(dt_arr):
        raise RuntimeError("CANDIDATE_DUPLICATE_KEY")

    avail = pd.to_datetime(track.available_time)
    # Vectorized alignment: avail is monotonic, so searchsorted maps each
    # decision_time to its 5m bar index on the decision (availability) axis.
    pos = np.searchsorted(avail, dt_arr)
    ok = (pos < len(avail)) & (avail[pos] == dt_arr)
    unmatched = int((~ok).sum())
    matched = rows.iloc[np.where(ok)[0]].copy()
    matched["bar_idx"] = pos[ok].astype(int)
    matched = matched.sort_values(["global_episode", "bar_idx"]).reset_index(drop=True)

    cand_by_time: Dict[pd.Timestamp, Dict[str, Any]] = {}
    dts = matched["decision_time"].to_numpy()
    eps = matched["global_episode"].to_numpy()
    pa = matched["proximity_any"].to_numpy() if "proximity_any" in matched.columns else None
    pe = (
        matched["proximity_episode_id"].to_numpy()
        if "proximity_episode_id" in matched.columns
        else None
    )
    for i in range(len(matched)):
        cand_by_time[pd.Timestamp(dts[i])] = {
            "episode": eps[i],
            "proximity_any": bool(pa[i]) if pa is not None else None,
            "proximity_episode_id": (pe[i] if pe is not None else None),
        }

    segments: List[CandidateSegment] = []
    seg_arr = np.asarray(track.segment)
    for ep_id, g in matched.groupby("global_episode", sort=False):
        gi = g["bar_idx"].to_numpy().astype(int)
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
        "t2_candidate_rows": int(len(rows)),
        "matched_5m_bars": int(len(matched)),
        "unmatched_candidate_rows": int(unmatched),
        "duplicate_decision_keys": 0,
        "n_episodes": int(matched["global_episode"].nunique()),
        "n_segments": int(len(segments)),
        "first_candidate_time": (
            pd.Timestamp(matched["decision_time"].min()) if len(matched) else None
        ),
        "last_candidate_time": (
            pd.Timestamp(matched["decision_time"].max()) if len(matched) else None
        ),
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
    strictly separate).
    """
    dt = pd.Timestamp(selected_decision_time)
    if dt not in cand_by_time:
        return {
            "is_candidate": False,
            "episode": None,
            "proximity_any": None,
            "proximity_episode_id": None,
        }
    r = cand_by_time[dt]
    return {
        "is_candidate": True,
        "episode": r["episode"],
        "proximity_any": r["proximity_any"],
        "proximity_episode_id": r["proximity_episode_id"],
    }
