"""
indicator_viewer_v1
====================

Pure-Python production helper for the Indicator Viewer
(Task FUTURE-INDICATOR-VIEWER-V1-KERNEL-UI-P1).

This module is a VISUALIZATION ENGINE. It does NOT own, modify, or
re-interpret any indicator math, parameters, or causal semantics.

Frozen owners reused verbatim (no math copied):
  * research.liquidity_oracle_atlas.experiment_structural_reversion_pgm_v1
        - PINE_DEFAULT            (frozen parameters)
        - resample_causal         (canonical 5m -> TF aggregation)
  * research.liquidity_oracle_atlas.forming_indicator_state_v1
        - IndicatorState          (canonical streaming DTP/SR/Liquidity)

Key causal contract (frozen):
  The streaming IndicatorState is stepped once over the TF bars, resetting
  on every discontinuity/segment change. The compact snapshot stored at
  TF bar index `t` is therefore exactly the indicator state as-of
  close(t):  IndicatorState_t  ⊆  Information_<= close(t).

No future bar (t+1 or later) ever enters a snapshot at t. Selecting a
different bar only changes `selected_index` + the rendered snapshot; it
NEVER recomputes history. The DTP profile is computed only at render time
of the selected bar (O(L x bins), L = current trend length).

Visual-only state (NOT fed back into any model / FEATURE_COLS):
  * Per-level Liquidity post-break zone lifecycle (zone_exists / active /
    left / right / top / bottom), keyed by stable (segment, left, level)
    so two levels breaching at different times do not overwrite each other.
    Buyside and Sellside use DISTINCT frozen formulas (side parameter).

No oracle / label / PGM / model code is imported here. The literal Pine
profile port lives ONLY in the test file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from research.liquidity_oracle_atlas.experiment_structural_reversion_pgm_v1 import (
    PINE_DEFAULT,
    resample_causal,
)
from research.liquidity_oracle_atlas.forming_indicator_state_v1 import (
    IndicatorState,
)

# --------------------------------------------------------------------------- #
# Frozen constants (read-only; never re-derived)                              #
# --------------------------------------------------------------------------- #
TF_LABEL_TO_MINUTES = {"5m": 5, "15m": 15, "1H": 60, "4H": 240}
MINUTES_TO_LABEL = {v: k for k, v in TF_LABEL_TO_MINUTES.items()}

# DTP profile (frozen Pine inputs)
BINS = 50
PROFILE_OFFSET = 30  # Pine `offset` input

# Compact array upper bounds
SR_MAX = 6          # PINE_DEFAULT.sr_max_channels
LIQ_VISIBLE = 3     # PINE_DEFAULT.liq_visible

# Historical-as-of viewport: render a fixed trailing window ending at the
# selected bar so no future OHLC is ever drawn.
VIEW_BARS = 300

# TradingView-like palette
C_BG = "#131722"
C_GRID = "#2A2E39"
C_TEXT = "#D1D4DC"
C_DTP_UP = "rgb(18, 209, 235)"
C_DTP_DOWN = "rgb(250, 40, 86)"
C_BULL = "#26A69A"
C_BEAR = "#F23645"
C_SR_RES = "rgba(242, 54, 69, 0.16)"
C_SR_SUP = "rgba(38, 166, 154, 0.16)"
C_SR_IN = "rgba(130, 130, 130, 0.16)"
C_BUY = "#4caf50"
C_SELL = "#f23645"

_NAN = float("nan")


# --------------------------------------------------------------------------- #
# Compact track                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class ViewerTrack:
    """Bounded compact arrays for one (symbol, timeframe).

    No full IndicatorState deepcopy is ever stored. Every field is a fixed
    upper-bound numpy array indexed by TF bar index.
    """

    symbol: str
    tf_label: str
    minutes: int
    n: int

    # TF bars (canonical resample output)
    time: np.ndarray            # bar_start_time (datetime64[ns])
    available_time: np.ndarray  # datetime64[ns]
    segment: np.ndarray         # int64
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray

    # DTP compact
    sma: np.ndarray
    atr: np.ndarray
    atr_liq: np.ndarray
    trend_score: np.ndarray
    trend_state: np.ndarray     # int8: -1 down / 1 up
    trend_start_global: np.ndarray  # int64, -1 = no switch yet

    # SR compact
    sr_top: np.ndarray          # (N, SR_MAX)
    sr_bottom: np.ndarray
    sr_strength: np.ndarray
    sr_valid: np.ndarray         # bool (N, SR_MAX)
    sr_n_channels: np.ndarray
    sr_in_zone: np.ndarray

    # Liquidity scalars
    liq_up_count: np.ndarray
    liq_down_count: np.ndarray
    liq_breach_up: np.ndarray
    liq_breach_down: np.ndarray

    # Liquidity compact (up)
    liq_up_valid: np.ndarray        # (N, LIQ_VISIBLE) bool
    liq_up_left: np.ndarray
    liq_up_level: np.ndarray
    liq_up_top: np.ndarray
    liq_up_bottom: np.ndarray
    liq_up_broken: np.ndarray
    liq_up_breach: np.ndarray       # global bar index or nan
    liq_up_zone_exists: np.ndarray  # 1.0 once a post-break zone was created
    liq_up_zone_active: np.ndarray
    liq_up_zone_left: np.ndarray
    liq_up_zone_right: np.ndarray
    liq_up_zone_top: np.ndarray
    liq_up_zone_bottom: np.ndarray

    # Liquidity compact (down)
    liq_down_valid: np.ndarray
    liq_down_left: np.ndarray
    liq_down_level: np.ndarray
    liq_down_top: np.ndarray
    liq_down_bottom: np.ndarray
    liq_down_broken: np.ndarray
    liq_down_breach: np.ndarray
    liq_down_zone_exists: np.ndarray
    liq_down_zone_active: np.ndarray
    liq_down_zone_left: np.ndarray
    liq_down_zone_right: np.ndarray
    liq_down_zone_top: np.ndarray
    liq_down_zone_bottom: np.ndarray

    # Performance counters
    raw_load_count: int = 1
    resample_count: int = 1
    indicator_step_count: int = 0
    full_history_recompute_count: int = 0
    reference_call_count: int = 0
    visual_snapshot_write_count: int = 0
    source_sha: str = ""


# --------------------------------------------------------------------------- #
# Liquidity visual-only post-break zone tracker                                #
# --------------------------------------------------------------------------- #
def _store_liq(
    out_valid, out_left, out_level, out_top, out_bottom, out_broken, out_breach,
    out_zone_exists, out_zone_active, out_zone_left, out_zone_right,
    out_zone_top, out_zone_bottom,
    levels: list[dict[str, Any]],
    tracker: dict[tuple[int, int, float], dict[str, Any]],
    seg: int, i: int, side: int, atr_liq: float, H: float, L: float,
) -> None:
    """Update the visual-only per-level post-break zone and write the compact
    row for bar ``i``.

    ``side`` selects the DISTINCT frozen formula:
      * side > 0 (Buyside):  zone_bottom = level, zone_top = min(level+2.3ATR, H)
      * side < 0 (Sellside): zone_top    = level, zone_bottom = max(level-2.3ATR, L)

    Level identity = (segment, left, level) so that two levels breaching at
    different times keep independent zone lifecycles.

    A post-break zone that has closed (price left the inside band) keeps its
    final frozen geometry: ``zone_exists`` stays True while ``zone_active`` is
    False. It only disappears once the level object itself is dropped from the
    visible collection (not drawn here -> slot becomes invalid).
    """
    pb = 2.3 * atr_liq if np.isfinite(atr_liq) else _NAN
    n = len(levels)
    for slot in range(LIQ_VISIBLE):
        if slot >= n:
            break
        lev = levels[slot]
        key = (int(seg), int(lev["left"]), float(lev["level"]))
        broken = bool(lev["broken"])
        tr = tracker.get(key)
        if tr is None:
            tr = {
                "state": "pre", "broken_seen": False,
                "zone_left": _NAN, "zone_right": _NAN,
                "zone_top": _NAN, "zone_bottom": _NAN, "breach_i": _NAN,
            }
            tracker[key] = tr

        if broken and not tr["broken_seen"]:
            # New breach at bar i (breach bar itself never runs the zone test)
            tr["broken_seen"] = True
            tr["breach_i"] = i
            tr["state"] = "active"
            tr["zone_left"] = i - 1
            tr["zone_right"] = i + 1
            if side > 0:  # Buyside
                tr["zone_bottom"] = float(lev["level"])
                tr["zone_top"] = (
                    min(float(lev["level"]) + pb, H) if np.isfinite(pb)
                    else float(lev["level"])
                )
            else:  # Sellside
                tr["zone_top"] = float(lev["level"])
                tr["zone_bottom"] = (
                    max(float(lev["level"]) - pb, L) if np.isfinite(pb)
                    else float(lev["level"])
                )
        elif tr["state"] == "active":
            if np.isfinite(pb) and (L > float(lev["level"]) - pb) and (H < float(lev["level"]) + pb):
                tr["zone_right"] = i + 1
                if side > 0:
                    tr["zone_top"] = max(tr["zone_top"], H)
                else:
                    tr["zone_bottom"] = min(tr["zone_bottom"], L)
            else:
                tr["state"] = "closed"

        out_valid[i, slot] = True
        out_left[i, slot] = float(lev["left"])
        out_level[i, slot] = float(lev["level"])
        out_top[i, slot] = float(lev["top"])
        out_bottom[i, slot] = float(lev["bottom"])
        out_broken[i, slot] = 1.0 if broken else 0.0
        out_breach[i, slot] = float(tr["breach_i"]) if np.isfinite(tr["breach_i"]) else _NAN
        out_zone_exists[i, slot] = 1.0 if tr["state"] != "pre" else 0.0
        out_zone_active[i, slot] = 1.0 if tr["state"] == "active" else 0.0
        out_zone_left[i, slot] = float(tr["zone_left"]) if np.isfinite(tr["zone_left"]) else _NAN
        out_zone_right[i, slot] = float(tr["zone_right"]) if np.isfinite(tr["zone_right"]) else _NAN
        out_zone_top[i, slot] = float(tr["zone_top"]) if np.isfinite(tr["zone_top"]) else _NAN
        out_zone_bottom[i, slot] = float(tr["zone_bottom"]) if np.isfinite(tr["zone_bottom"]) else _NAN


# --------------------------------------------------------------------------- #
# Build                                                                        #
# --------------------------------------------------------------------------- #
def build_viewer_track(
    base: pd.DataFrame,
    timeframe: Any,
    *,
    symbol: str = "",
    source_sha: str = "",
    raw_load_count: int = 1,
) -> ViewerTrack:
    """Build a compact ViewerTrack for one symbol / timeframe.

    Pipeline (frozen):
      canonical 5m base  ->  canonical resample (ONCE)  ->  single-pass
      IndicatorState (reset on segment change)  ->  compact per-bar snapshot.

    ``base`` must carry the canonical columns produced by
    ``raw_frame_from_owner``: time, trading_day, segment, open, high, low,
    close (and disc). The caller is responsible for raw loading / caching.
    """
    if isinstance(timeframe, str):
        minutes = TF_LABEL_TO_MINUTES.get(timeframe)
        if minutes is None:
            raise ValueError(f"unknown timeframe label: {timeframe!r}")
    else:
        minutes = int(timeframe)
    tf_label = MINUTES_TO_LABEL.get(minutes, f"{minutes}m")

    tf_bars = resample_causal(base, minutes)
    n = len(tf_bars)

    time = tf_bars["time"].to_numpy()
    available_time = tf_bars["available_time"].to_numpy()
    seg_arr = tf_bars["segment"].to_numpy(np.int64)
    o = tf_bars["open"].to_numpy(float)
    h = tf_bars["high"].to_numpy(float)
    l = tf_bars["low"].to_numpy(float)
    c = tf_bars["close"].to_numpy(float)

    sma = np.full(n, _NAN)
    atr = np.full(n, _NAN)
    atr_liq = np.full(n, _NAN)
    trend_score = np.full(n, _NAN)
    trend_state = np.full(n, -1, dtype=np.int8)
    trend_start = np.full(n, -1, dtype=np.int64)

    sr_top = np.full((n, SR_MAX), _NAN)
    sr_bottom = np.full((n, SR_MAX), _NAN)
    sr_strength = np.full((n, SR_MAX), _NAN)
    sr_valid = np.zeros((n, SR_MAX), dtype=bool)
    sr_n_channels = np.zeros(n, dtype=np.int64)
    sr_in_zone = np.zeros(n, dtype=np.int8)

    liq_up_count = np.zeros(n, dtype=np.int64)
    liq_down_count = np.zeros(n, dtype=np.int64)
    liq_breach_up = np.zeros(n, dtype=np.int64)
    liq_breach_down = np.zeros(n, dtype=np.int64)

    def _mk() -> np.ndarray:
        return np.full((n, LIQ_VISIBLE), _NAN)

    def _mkz() -> np.ndarray:
        return np.zeros((n, LIQ_VISIBLE))

    liq_up_valid = np.zeros((n, LIQ_VISIBLE), dtype=bool)
    liq_up_left = _mk(); liq_up_level = _mk(); liq_up_top = _mk(); liq_up_bottom = _mk()
    liq_up_broken = np.zeros((n, LIQ_VISIBLE))
    liq_up_breach = _mk()
    liq_up_zone_exists = _mkz()
    liq_up_zone_active = _mkz()
    liq_up_zone_left = _mk(); liq_up_zone_right = _mk()
    liq_up_zone_top = _mk(); liq_up_zone_bottom = _mk()

    liq_down_valid = np.zeros((n, LIQ_VISIBLE), dtype=bool)
    liq_down_left = _mk(); liq_down_level = _mk(); liq_down_top = _mk(); liq_down_bottom = _mk()
    liq_down_broken = np.zeros((n, LIQ_VISIBLE))
    liq_down_breach = _mk()
    liq_down_zone_exists = _mkz()
    liq_down_zone_active = _mkz()
    liq_down_zone_left = _mk(); liq_down_zone_right = _mk()
    liq_down_zone_top = _mk(); liq_down_zone_bottom = _mk()

    state = IndicatorState(PINE_DEFAULT, include_sr=True)
    cur_seg: int | None = None
    ci = 0
    prev_trend = _NAN
    current_trend_start = -1  # explicit local; never copied from previous row
    up_tracker: dict[tuple[int, int, float], dict[str, Any]] = {}
    down_tracker: dict[tuple[int, int, float], dict[str, Any]] = {}
    step_count = 0

    for i in range(n):
        seg = int(seg_arr[i])
        if seg != cur_seg:
            # Hard reset on discontinuity / segment change. The new segment
            # must NOT inherit any previous-segment DTP trend metadata.
            state.reset()
            cur_seg = seg
            ci = 0
            prev_trend = _NAN
            current_trend_start = -1
            up_tracker = {}
            down_tracker = {}

        feats = state.step(ci, o[i], h[i], l[i], c[i])
        step_count += 1

        sma[i] = feats["sma"]
        atr[i] = feats["atr"]
        atr_liq[i] = feats["atr_liq"]
        trend_score[i] = feats["trend_score"]
        ts = round(feats["trend_state"])
        trend_state[i] = ts
        # Real trend switch (prev finite and state changed) -> new start.
        if np.isfinite(prev_trend) and ts != int(prev_trend):
            current_trend_start = i
        trend_start[i] = current_trend_start
        prev_trend = float(ts)

        # SR channels (as-of close(t))
        chans = state.sr.channels
        sr_n_channels[i] = len(chans)
        sr_in_zone[i] = int(feats["sr_in_zone"])
        for j in range(min(SR_MAX, len(chans))):
            top, bot, str_ = chans[j]
            sr_top[i, j] = top
            sr_bottom[i, j] = bot
            sr_strength[i, j] = str_
            sr_valid[i, j] = True

        _store_liq(
            liq_up_valid, liq_up_left, liq_up_level, liq_up_top, liq_up_bottom,
            liq_up_broken, liq_up_breach, liq_up_zone_exists, liq_up_zone_active,
            liq_up_zone_left, liq_up_zone_right, liq_up_zone_top, liq_up_zone_bottom,
            state.liq.levels_up, up_tracker, seg, i, +1, atr_liq[i], h[i], l[i],
        )
        _store_liq(
            liq_down_valid, liq_down_left, liq_down_level, liq_down_top, liq_down_bottom,
            liq_down_broken, liq_down_breach, liq_down_zone_exists, liq_down_zone_active,
            liq_down_zone_left, liq_down_zone_right, liq_down_zone_top, liq_down_zone_bottom,
            state.liq.levels_down, down_tracker, seg, i, -1, atr_liq[i], h[i], l[i],
        )
        liq_up_count[i] = int(feats["liq_up_count"])
        liq_down_count[i] = int(feats["liq_down_count"])
        liq_breach_up[i] = int(feats["liq_breach_up"])
        liq_breach_down[i] = int(feats["liq_breach_down"])
        ci += 1

    track = ViewerTrack(
        symbol=symbol, tf_label=tf_label, minutes=minutes, n=n,
        time=time, available_time=available_time, segment=seg_arr,
        open=o, high=h, low=l, close=c,
        sma=sma, atr=atr, atr_liq=atr_liq, trend_score=trend_score,
        trend_state=trend_state, trend_start_global=trend_start,
        sr_top=sr_top, sr_bottom=sr_bottom, sr_strength=sr_strength, sr_valid=sr_valid,
        sr_n_channels=sr_n_channels, sr_in_zone=sr_in_zone,
        liq_up_count=liq_up_count, liq_down_count=liq_down_count,
        liq_breach_up=liq_breach_up, liq_breach_down=liq_breach_down,
        liq_up_valid=liq_up_valid, liq_up_left=liq_up_left, liq_up_level=liq_up_level,
        liq_up_top=liq_up_top, liq_up_bottom=liq_up_bottom, liq_up_broken=liq_up_broken,
        liq_up_breach=liq_up_breach, liq_up_zone_exists=liq_up_zone_exists,
        liq_up_zone_active=liq_up_zone_active,
        liq_up_zone_left=liq_up_zone_left, liq_up_zone_right=liq_up_zone_right,
        liq_up_zone_top=liq_up_zone_top, liq_up_zone_bottom=liq_up_zone_bottom,
        liq_down_valid=liq_down_valid, liq_down_left=liq_down_left, liq_down_level=liq_down_level,
        liq_down_top=liq_down_top, liq_down_bottom=liq_down_bottom, liq_down_broken=liq_down_broken,
        liq_down_breach=liq_down_breach, liq_down_zone_exists=liq_down_zone_exists,
        liq_down_zone_active=liq_down_zone_active,
        liq_down_zone_left=liq_down_zone_left, liq_down_zone_right=liq_down_zone_right,
        liq_down_zone_top=liq_down_zone_top, liq_down_zone_bottom=liq_down_zone_bottom,
        raw_load_count=int(raw_load_count), resample_count=1,
        indicator_step_count=step_count,
        full_history_recompute_count=0, reference_call_count=0,
        visual_snapshot_write_count=n, source_sha=source_sha,
    )
    return track


# --------------------------------------------------------------------------- #
# Viewport (Historical-as-of: nothing beyond `selected` is ever drawn)          #
# --------------------------------------------------------------------------- #
def compute_viewport(track: ViewerTrack, selected: int) -> tuple[int, int]:
    """Fixed trailing window ending at the selected bar.

    Returns (lo, hi) with hi == selected and (hi - lo + 1) <= VIEW_BARS.
    The window is clamped to the start of the selected bar's own segment so a
    segment boundary never leaks future-of-other-segment context.
    """
    n = track.n
    i = int(selected)
    if i < 0 or i >= n:
        raise IndexError(f"selected_index {i} out of range [0, {n})")
    seg = int(track.segment[i])
    seg_start = int(np.argmax(track.segment == seg))
    lo = max(seg_start, i - VIEW_BARS + 1)
    return lo, i


# --------------------------------------------------------------------------- #
# DTP profile (literal port of ref/DeviationTrendProfile.pine::profile)         #
# --------------------------------------------------------------------------- #
def dtp_profile(track: ViewerTrack, selected_index: int) -> tuple[np.ndarray | None, int | None]:
    """Compute the Trend Distribution Profile counts at the selected bar.

    Literal port of the Pine ``profile()`` counting loop:

        for l = 0 to loockback:                 # l walks trend_start..selected
            c  = close[l]
            mi = min[l]   = avg[l] - 3*atr[l]
            s  = step[l]  = 6*atr[l] / bins
            for i = 0 to bins-1:
                lower = mi + s*i
                upper = lower + s
                if c >= lower - s and c <= upper + s:
                    bin[i] += 1

    NOTE: the *counts* use the historical bar's own min/step (per Pine),
    while the RENDERED price levels use the selected bar's min/step (also
    per Pine). The page computes the rendered geometry; this helper returns
    only the counts + lookback (trend age in bars).
    """
    n = track.n
    i = int(selected_index)
    if i < 0 or i >= n:
        return None, None
    ts = int(track.trend_start_global[i])
    if ts < 0:
        return None, None  # profile unavailable before first valid trend switch

    lookback = i - ts
    bins = BINS
    counts = np.zeros(bins, dtype=np.int64)
    close = track.close
    sma = track.sma
    atr = track.atr
    for l in range(ts, i + 1):
        cl = close[l]
        mi = sma[l] - 3.0 * atr[l]
        s = 6.0 * atr[l] / bins
        if not (np.isfinite(cl) and np.isfinite(mi) and np.isfinite(s) and s != 0.0):
            continue
        for b in range(bins):
            lower = mi + s * b
            upper = lower + s
            if cl >= lower - s and cl <= upper + s:
                counts[b] += 1
    return counts, lookback


# --------------------------------------------------------------------------- #
# Selected-bar snapshot (O(1) lookup)                                          #
# --------------------------------------------------------------------------- #
def _liq_levels(
    i: int,
    valid, left, level, top, bottom, broken, breach,
    zone_exists, zone_active, zone_left, zone_right, zone_top, zone_bottom,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for j in range(LIQ_VISIBLE):
        if not valid[i, j]:
            continue
        out.append({
            "left": float(left[i, j]),
            "level": float(level[i, j]),
            "top": float(top[i, j]),
            "bottom": float(bottom[i, j]),
            "broken": bool(broken[i, j]),
            "breach": (float(breach[i, j]) if np.isfinite(breach[i, j]) else None),
            "zone_exists": bool(zone_exists[i, j]),
            "zone_active": bool(zone_active[i, j]),
            "zone_left": (float(zone_left[i, j]) if np.isfinite(zone_left[i, j]) else None),
            "zone_right": (float(zone_right[i, j]) if np.isfinite(zone_right[i, j]) else None),
            "zone_top": (float(zone_top[i, j]) if np.isfinite(zone_top[i, j]) else None),
            "zone_bottom": (float(zone_bottom[i, j]) if np.isfinite(zone_bottom[i, j]) else None),
        })
    return out


def selected_snapshot(track: ViewerTrack, selected_index: int) -> dict[str, Any]:
    """O(1) as-of snapshot of the selected bar (no recomputation)."""
    i = int(selected_index)
    n = track.n
    if i < 0 or i >= n:
        raise IndexError(f"selected_index {i} out of range [0, {n})")

    sr: list[dict[str, Any]] = []
    for j in range(SR_MAX):
        if track.sr_valid[i, j]:
            sr.append({
                "top": float(track.sr_top[i, j]),
                "bottom": float(track.sr_bottom[i, j]),
                "strength": float(track.sr_strength[i, j]),
            })

    up = _liq_levels(
        i, track.liq_up_valid, track.liq_up_left, track.liq_up_level, track.liq_up_top,
        track.liq_up_bottom, track.liq_up_broken, track.liq_up_breach,
        track.liq_up_zone_exists, track.liq_up_zone_active, track.liq_up_zone_left,
        track.liq_up_zone_right, track.liq_up_zone_top, track.liq_up_zone_bottom,
    )
    down = _liq_levels(
        i, track.liq_down_valid, track.liq_down_left, track.liq_down_level, track.liq_down_top,
        track.liq_down_bottom, track.liq_down_broken, track.liq_down_breach,
        track.liq_down_zone_exists, track.liq_down_zone_active, track.liq_down_zone_left,
        track.liq_down_zone_right, track.liq_down_zone_top, track.liq_down_zone_bottom,
    )

    ts = int(track.trend_start_global[i])
    profile_available = ts >= 0
    trend_age = (i - ts) if profile_available else None

    return {
        "index": i,
        "symbol": track.symbol,
        "tf": track.tf_label,
        "bar_start_time": pd.Timestamp(track.time[i]),
        "available_time": pd.Timestamp(track.available_time[i]),
        "segment": int(track.segment[i]),
        "o": float(track.open[i]), "h": float(track.high[i]),
        "l": float(track.low[i]), "c": float(track.close[i]),
        "dtp": {
            "trend": int(track.trend_state[i]),
            "sma": float(track.sma[i]),
            "atr": float(track.atr[i]),
            "atr_liq": float(track.atr_liq[i]),
            "trend_score": float(track.trend_score[i]),
            "trend_age": trend_age,
            "profile_available": profile_available,
        },
        "sr_channels": sr,
        "sr_n_channels": int(track.sr_n_channels[i]),
        "sr_in_zone": bool(track.sr_in_zone[i]),
        "liq_up": up,
        "liq_up_count": int(track.liq_up_count[i]),
        "liq_down": down,
        "liq_down_count": int(track.liq_down_count[i]),
    }


# --------------------------------------------------------------------------- #
# DP Oracle overlay (Checkpoint B — FUTURE / HINDSIGHT AUDIT, read-only)        #
# --------------------------------------------------------------------------- #
# The oracle is defined on the 5m decision clock. Executions are matched to the
# ViewerTrack by TIME (never by decision_bar_index) and validated against the
# bar open. This overlay is a HINDSIGHT audit label; it is NOT a causal signal
# and never modifies any DTP / SR / Liquidity array.
ORACLE_TF_ONLY = "5m"
ORACLE_ALIGN_RTOL = 1e-9
ORACLE_ALIGN_ATOL = 1e-6

_ENTRY_HOVER = (
    "<b>%{customdata[0]} ENTRY</b><br>"
    "fill %{customdata[1]}<br>"
    "price %{customdata[2]}<br>"
    "source_bits %{customdata[3]}<br>"
    "trading_day %{customdata[4]}<br>"
    "%{customdata[5]}<extra></extra>"
)
_EXIT_HOVER = (
    "<b>%{customdata[0]} EXIT (Gross Oracle)</b><br>"
    "fill %{customdata[1]}<br>"
    "price %{customdata[2]}<br>"
    "gross %{customdata[3]:.2f} pts<br>"
    "net %{customdata[4]:.2f} pts<br>"
    "holding %{customdata[5]} bars<br>"
    "MFE %{customdata[6]:.2f} / MAE %{customdata[7]:.2f}<br>"
    "%{customdata[8]}<extra></extra>"
)


def oracle_fill_index(
    track: "ViewerTrack",
    fill_time: Any,
    fill_price: float,
    *,
    rtol: float = ORACLE_ALIGN_RTOL,
    atol: float = ORACLE_ALIGN_ATOL,
) -> int:
    """Map an oracle execution ``(fill_time, fill_price)`` to a ViewerTrack x index.

    Alignment uses TIME as the semantic key (never ``decision_bar_index``), then
    VALIDATES ``open(fill) == fill_price``. Returns -1 on any mismatch (non-5m
    track, missing bar, or price mismatch) so a discontinuity / missing bar /
    reindexed series can never silently draw the wrong position.
    """
    if track.tf_label != ORACLE_TF_ONLY:
        return -1
    ft = np.datetime64(pd.Timestamp(fill_time).to_datetime64(), "ns")
    idx = np.flatnonzero(track.time == ft)
    if idx.size == 0:
        return -1
    x = int(idx[0])
    if not np.isclose(float(track.open[x]), float(fill_price), rtol=rtol, atol=atol):
        return -1
    return x


def select_visible_oracle_trades(track: "ViewerTrack", selected: int, trades: Any):
    """Vectorized viewport filter, then per-trade time->index alignment.

    Returns ``(records, mismatch_count)`` where ``records`` is a list of
    ``(row_dict, entry_x, exit_x)``. Non-5m tracks return ``([], 0)`` so the
    overlay never draws execution markers on 15m / 1H / 4H.
    """
    if track.tf_label != ORACLE_TF_ONLY:
        return [], 0
    if trades is None:
        return [], 0
    if not isinstance(trades, pd.DataFrame):
        trades = pd.DataFrame(trades)
    if len(trades) == 0:
        return [], 0

    lo, hi = compute_viewport(track, selected)
    t_lo = track.time[lo]
    t_hi = track.time[hi]
    ent_t = pd.to_datetime(trades["entry_fill_time"]).to_numpy()
    ext_t = pd.to_datetime(trades["exit_fill_time"]).to_numpy()
    mask = (ext_t >= t_lo) & (ent_t <= t_hi)
    visible = trades.loc[mask]

    records: list = []
    mismatch = 0
    for row in visible.to_dict("records"):
        ex = oracle_fill_index(track, row["entry_fill_time"], row["entry_fill_price"])
        xx = oracle_fill_index(track, row["exit_fill_time"], row["exit_fill_price"])
        if ex < 0 or xx < 0:
            mismatch += 1
            continue
        records.append((row, ex, xx))
    return records, mismatch


def _push_marker(d, x, y, cd, text):
    d["x"].append(x)
    d["y"].append(float(y))
    d["cd"].append(cd)
    d["text"].append(text)


def _add_marker_trace(fig, d, *, symbol, color, name, textpos, hovertemplate):
    if not d["x"]:
        return
    fig.add_trace(go.Scatter(
        x=d["x"], y=d["y"], mode="markers+text",
        marker={"symbol": symbol, "size": 12, "color": color},
        text=d["text"], textposition=textpos,
        textfont={"size": 9, "color": color},
        customdata=d["cd"], hovertemplate=hovertemplate,
        name=name, showlegend=True,
    ))


def add_dp_oracle_overlay(fig, track: "ViewerTrack", selected: int, trades: Any):
    """Add the hindsight oracle Entry/Exit/Reversal markers to ``fig`` (5m only).

    At most FOUR marker traces (Long Entry/Exit, Short Entry/Exit) plus ONE
    connector trace are added, regardless of how many trades are visible.
    A reversal (e.g. LONG -> SHORT) naturally yields BOTH an exit marker of the
    closing trade and an entry marker of the opening trade at the same fill.
    """
    if track.tf_label != ORACLE_TF_ONLY:
        return fig
    records, _mismatch = select_visible_oracle_trades(track, selected, trades)
    if not records:
        return fig

    # Historical-as-of clipping: `selected` is the LAST visible bar, so NO oracle
    # marker / connector may extend beyond it. A trade whose fill is still in the
    # future (entry_x <= S < exit_x) shows its Entry marker ONLY; its Exit marker
    # and any future execution price stay hidden until selected reaches exit_x.
    S = int(selected)

    le = {"x": [], "y": [], "cd": [], "text": []}
    lx = {"x": [], "y": [], "cd": [], "text": []}
    se = {"x": [], "y": [], "cd": [], "text": []}
    sx = {"x": [], "y": [], "cd": [], "text": []}
    conn_x: list = []
    conn_y: list = []

    for row, ex, xx in records:
        direction = str(row["direction"])
        is_long = direction == "LONG"
        entry_cd = (
            direction,
            str(pd.Timestamp(row["entry_fill_time"])),
            float(row["entry_fill_price"]),
            int(row.get("entry_source_bits", 0)),
            str(row.get("trading_day", "")),
            str(row.get("trade_id", "")),
        )
        exit_cd = (
            direction,
            str(pd.Timestamp(row["exit_fill_time"])),
            float(row["exit_fill_price"]),
            float(row.get("gross_points", np.nan)),
            float(row.get("net_points", np.nan)),
            int(row.get("holding_bars", 0)),
            float(row.get("MFE", np.nan)),
            float(row.get("MAE", np.nan)),
            str(row.get("trade_id", "")),
        )
        if ex <= S:
            if is_long:
                _push_marker(le, ex, row["entry_fill_price"], entry_cd, "L IN")
            else:
                _push_marker(se, ex, row["entry_fill_price"], entry_cd, "S IN")
        if xx <= S:
            if is_long:
                _push_marker(lx, xx, row["exit_fill_price"], exit_cd, "L OUT")
            else:
                _push_marker(sx, xx, row["exit_fill_price"], exit_cd, "S OUT")
            # connector only when the WHOLE trade lies inside the as-of view
            conn_x += [ex, xx, None]
            conn_y += [float(row["entry_fill_price"]), float(row["exit_fill_price"]), None]

    _add_marker_trace(fig, le, symbol="triangle-up", color=C_BUY,
                      name="Long Entry", textpos="top center", hovertemplate=_ENTRY_HOVER)
    _add_marker_trace(fig, lx, symbol="x", color=C_BUY,
                      name="Long Exit", textpos="bottom center", hovertemplate=_EXIT_HOVER)
    _add_marker_trace(fig, se, symbol="triangle-down", color=C_SELL,
                      name="Short Entry", textpos="bottom center", hovertemplate=_ENTRY_HOVER)
    _add_marker_trace(fig, sx, symbol="x", color=C_SELL,
                      name="Short Exit", textpos="top center", hovertemplate=_EXIT_HOVER)

    if conn_x:
        fig.add_trace(go.Scatter(
            x=conn_x, y=conn_y, mode="lines",
            line={"width": 1, "dash": "dot", "color": C_TEXT}, opacity=0.35,
            name="Oracle trade", showlegend=False, hoverinfo="skip",
        ))
    return fig


def oracle_viewport_summary(track: "ViewerTrack", selected: int, trades: Any) -> dict:
    """Audit-only summary of the visible oracle trades (no strategy verdict).

    Historical-as-of: PnL / holding statistics are computed over CLOSED trades
    only (exit_x <= selected); trades still open at ``selected`` are reported
    separately as ``open_at_selected`` so no FUTURE exit PnL leaks into the
    audit line.
    """
    records, mismatch = select_visible_oracle_trades(track, selected, trades)
    S = int(selected)
    closed = [(r, e, x) for (r, e, x) in records if x <= S]
    longs = sum(1 for (r, _e, _x) in records if str(r["direction"]) == "LONG")
    gross = sum(float(r.get("gross_points", 0.0)) for (r, _e, _x) in closed)
    holds = [int(r.get("holding_bars", 0)) for (r, _e, _x) in closed]
    return {
        "visible_trades": len(records),
        "closed_trades": len(closed),
        "open_at_selected": len(records) - len(closed),
        "long_trades": longs,
        "short_trades": len(records) - longs,
        "total_gross_points": float(gross),
        "median_holding_bars": float(np.median(holds)) if holds else 0.0,
        "alignment_mismatch": int(mismatch),
    }
