"""
pages/6_Indicator_Viewer.py
===========================

Independent read-only Indicator Viewer (Task FUTURE-INDICATOR-VIEWER-V1-KERNEL-UI-P1).

Shows the three raw indicators (DTP / SR / Liquidity) as-of a selected TF bar's
close:  IndicatorState_t  ⊆  Information_<= close(t).

The BASE indicator view is causal / historical-as-of: the selected bar is the
LAST visible candle and nothing beyond `selected` is drawn. An OPTIONAL DP
Oracle overlay exists purely as a FUTURE / HINDSIGHT AUDIT layer (task
FUTURE-INTRADAY-DP-ORACLE-VIEWER-R1): it is OFF by default, read-only, clipped
to `selected`, and is NEVER a causal signal or a model feature. No Label / PGM /
model output is rendered.

Visual contract (frozen):
  * Historical-as-of: the selected bar is the LAST visible candle; nothing
    beyond `selected` is drawn (no future OHLC / DTP / markers).
  * Fixed trailing viewport (VIEW_BARS) ending at `selected` -> O(VIEW_BARS)
    render, independent of full history N.
  * DTP ±1/±2/±3 ATR drawn as short horizontal levels at the selected bar
    (Pine current-bar semantics), not full-history band curves.
  * DTP profile computed from Pine literal counting semantics.
  * SR / Liquidity extend through the visible history [lo, selected].
  * Liquidity post-break zone retains frozen geometry after it closes.
"""

from __future__ import annotations

from pathlib import Path

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from research.liquidity_oracle_atlas.audit_view_v1 import SYMBOLS
from research.liquidity_oracle_atlas.build_forming_environment_v1 import (
    FormingEnvironmentBuilder,
)
from research.liquidity_oracle_atlas.build_structure_constrained_trade_oracle_dp_v2 import (
    ARTIFACT_ROOT_DIRNAME,
    MATH_VERSION,
    load_oracle_artifact_v2 as load_oracle_artifact,
)
from research.liquidity_oracle_atlas.build_structure_constrained_trade_oracle_dp_v1 import (
    oracle_cache_token,
)
from research.liquidity_oracle_atlas.git_head import git_head
from research.liquidity_oracle_atlas.indicator_viewer_v1 import (
    BINS,
    LIQ_VISIBLE,
    ORACLE_TF_ONLY,
    PROFILE_OFFSET,
    SR_MAX,
    add_dp_oracle_overlay,
    build_viewer_track,
    compute_viewport,
    dtp_profile,
    oracle_viewport_summary,
    select_visible_oracle_trades,
    selected_snapshot,
)
from research.liquidity_oracle_atlas.indicator_viewer_candidate_overlay_v1 import (
    TRIGGER_BITS,
    build_candidate_segments,
    candidate_for_symbol,
    candidate_state_at,
    load_candidate_rows,
    match_available_index,
)
from research.liquidity_oracle_atlas.build_candidate_gate_r3_v1 import (
    load_candidate_gate_summary,
    load_candidate_proof_verified,
    touch_bit,
)
from research.liquidity_oracle_atlas.build_candidate_gate_r4_m15_v1 import (
    load_candidate_gate_summary as load_candidate_gate_summary_r4,
    touch_bit as touch_bit_r4,
)
from research.liquidity_oracle_atlas.build_trade_oracle_dp_m15_one_entry_proximity_v1 import (
    MATH_VERSION as DP_M15_MATH_VERSION,
    event_sort_key,
    load_oracle_artifact as load_oracle_artifact_m15,
    validate_execution_events,
)
from research.liquidity_oracle_atlas.dp_label_lifecycle_v1 import (
    build_lifecycle_records,
    run_lifecycle_assertions,
    find_representative_cases,
    run_negative_controls,
    evidence_summary,
)
from research.liquidity_oracle_atlas.indicator_viewer_candidate_overlay_r4_v1 import (
    TRIGGER_BITS_R4,
    build_candidate_marks_r4,
    build_proof_by_trigger,
    build_candidate_segments_r4,
    candidate_for_symbol_r4,
    candidate_state_at_r4,
    load_candidate_rows_r4,
    run_viewport_candidate_audit,
)

TF_LABELS = ["5m", "15m", "1H", "4H"]
TF_PREFIX = {"5m": "m5", "15m": "m15", "1H": "h1", "4H": "h4"}

# forming-MTF owner prefixes used by build_forming_environment_v1.run()
EMPTY_CAND_AUDIT = {
    "symbol": "",
    "t2_candidate_rows": 0,
    "matched_5m_bars": 0,
    "unmatched_candidate_rows": 0,
    "duplicate_decision_keys": 0,
    "n_episodes": 0,
    "n_segments": 0,
    "first_candidate_time": None,
    "last_candidate_time": None,
}

# DP Oracle audit artifacts live outside the indicator pipeline (read-only).
ORACLE_ARTIFACT_ROOT = (
    Path(__file__).resolve().parents[1] / "artifacts" / ARTIFACT_ROOT_DIRNAME
)
# R4 15m Oracle: mechanical 5m R2 port (FUTURE-INTRADAY-DP-ORACLE-R2-ONE-ENTRY-PROXIMITY-15M).
# DP-internal proximity ONLY; fully decoupled from the R4 Candidate Trading Zones.
ORACLE_ARTIFACT_ROOT_V4 = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "trade_oracle_dp_m15_one_entry_proximity_v1"
)

# --- palette --------------------------------------------------------------- #
C_BG = "#131722"
C_GRID = "#2A2E39"
C_TEXT = "#D1D4DC"
C_DTP_UP = "rgb(18, 209, 235)"
C_DTP_DOWN = "rgb(250, 40, 86)"
C_BULL = "#26A69A"
C_BEAR = "#F23645"
C_BUY = "#4caf50"
C_SELL = "#f23645"


def _rgba(rgb: str, a: float) -> str:
    """Accept either 'rgb(r, g, b)' or '#rrggbb' / '#rgb' and return rgba()."""
    rgb = rgb.strip()
    if rgb.startswith("#"):
        hv = rgb[1:]
        if len(hv) == 6:
            r, g, b = int(hv[0:2], 16), int(hv[2:4], 16), int(hv[4:6], 16)
        elif len(hv) == 3:
            r, g, b = int(hv[0] * 2, 16), int(hv[1] * 2, 16), int(hv[2] * 2, 16)
        else:
            return f"rgba(0, 0, 0, {a})"
        return f"rgba({r}, {g}, {b}, {a})"
    rgb = rgb.replace("rgb(", "").replace(")", "").strip()
    return f"rgba({rgb}, {a})"


def _parse_selection(event) -> int | None:
    """Parse a Plotly selection event into a TF bar index.

    Streamlit's on_select="rerun" returns the PlotlyState; read its
    `.selection` attribute (or `.get("selection")` if it ever arrives as a
    dict). Never read a self-invented session key.
    """
    if event is None:
        return None
    sel = event.get("selection") if isinstance(event, dict) else getattr(event, "selection", None)
    if not sel:
        return None
    pts = sel.get("points") if isinstance(sel, dict) else getattr(sel, "points", None)
    if not pts:
        return None
    cd = pts[0].get("customdata") if isinstance(pts[0], dict) else getattr(pts[0], "customdata", None)
    if cd is None:
        return None
    return int(cd[0])


def dtp_box_x(count: int, start: int) -> tuple[int, int]:
    """Pine ``box.new(start-val, upper, start, lower)`` horizontal span.

    The pinned source builds the profile box with its RIGHT edge at ``start``
    and its LEFT edge at ``start - val``, i.e. the profile grows to the LEFT of
    ``start`` (``start = bar_index + offset``). Returns ``(x0, x1)``.
    """
    return (int(start) - int(count), int(start))


def build_figure(track, selected, show_dtp, show_sr, show_liq) -> go.Figure:
    """TradingView-like figure ending exactly at `selected` (no future)."""
    n = track.n
    lo, hi = compute_viewport(track, selected)
    view = slice(lo, hi + 1)
    idx = np.arange(n)
    idx_v = idx[view]
    time_v = pd.to_datetime(track.time[view])

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=idx_v,
        open=track.open[view], high=track.high[view],
        low=track.low[view], close=track.close[view],
        name="OHLC",
        increasing_line_color=C_BULL, decreasing_line_color=C_BEAR,
        increasing_fillcolor=C_BULL, decreasing_fillcolor=C_BEAR,
        line={"width": 1},
    ))
    # selected-bar highlight
    fig.add_vline(x=selected, line={"color": "#FFD166", "width": 1, "dash": "dot"})

    # ---- DTP ------------------------------------------------------------ #
    if show_dtp:
        trend = track.trend_state[view]
        sma_v = track.sma[view]
        # Full-x NaN-masked series: Plotly breaks the line at every trend change
        # instead of connecting two disjoint UP (or DOWN) stretches.
        up_y = np.where(trend == 1, sma_v, np.nan)
        dn_y = np.where(trend == -1, sma_v, np.nan)
        fig.add_trace(go.Scatter(
            x=idx_v, y=up_y, mode="lines", connectgaps=False,
            line={"color": C_DTP_UP, "width": 1.5}, name="SMA UP", showlegend=False))
        fig.add_trace(go.Scatter(
            x=idx_v, y=dn_y, mode="lines", connectgaps=False,
            line={"color": C_DTP_DOWN, "width": 1.5}, name="SMA DOWN", showlegend=False))

        # ±1/±2/±3 ATR short horizontal levels at the selected bar (right side)
        s = float(track.sma[selected]); a = float(track.atr[selected])
        if np.isfinite(s) and np.isfinite(a) and a > 0:
            xb = [selected, selected + 5]
            for k in (1, 2, 3):
                # short level lines (kept; not full-history ATR curves)
                fig.add_trace(go.Scatter(
                    x=xb, y=[s + k * a, s + k * a], mode="lines",
                    line={"color": _rgba(C_DTP_UP, 0.5), "width": 1, "dash": "dot"},
                    showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(
                    x=xb, y=[s - k * a, s - k * a], mode="lines",
                    line={"color": _rgba(C_DTP_DOWN, 0.5), "width": 1, "dash": "dot"},
                    showlegend=False, hoverinfo="skip"))
            # small ±1/±2/±3 labels at the short levels
            for k in (1, 2, 3):
                fig.add_annotation(
                    x=selected + 5, y=s + k * a, text=f"+{k}", showarrow=False,
                    xanchor="left", yshift=0,
                    font={"color": _rgba(C_DTP_UP, 0.75), "size": 9})
                fig.add_annotation(
                    x=selected + 5, y=s - k * a, text=f"-{k}", showarrow=False,
                    xanchor="left", yshift=0,
                    font={"color": _rgba(C_DTP_DOWN, 0.75), "size": 9})

        # trend switch markers
        diff = np.where(trend[1:] != trend[:-1])[0] + 1
        for di in diff:
            x = int(idx_v[di]); d = int(trend[di])
            fig.add_trace(go.Scatter(
                x=[x], y=[float(track.sma[x])], mode="markers",
                marker={"color": C_DTP_UP if d == 1 else C_DTP_DOWN, "size": 9, "symbol": "circle"},
                showlegend=False, hoverinfo="skip"))

        # DTP Trend Distribution Profile
        # Pine: start = bar_index + offset; box.new(start-val, upper, start, lower)
        # -> every profile box extends to the LEFT of `start`, and its gradient
        #    driver is the bin COUNT (val), not the vertical bin index.
        counts, lookback = dtp_profile(track, selected)
        if counts is not None and lookback is not None:
            pmin = s - 3.0 * a
            pstep = 6.0 * a / BINS
            start = selected + PROFILE_OFFSET
            trend_up = int(track.trend_state[selected]) == 1
            base = C_DTP_UP if trend_up else C_DTP_DOWN
            mx = int(counts.max()) if counts.size else 0
            for b in range(BINS):
                cnt = int(counts[b])
                if cnt <= 0:
                    continue
                lower = pmin + pstep * b
                bx0, bx1 = dtp_box_x(cnt, start)
                op = 0.22 + 0.6 * (cnt / mx) if mx > 0 else 0.6
                fig.add_shape(type="rect", xref="x", yref="y",
                              x0=bx0, x1=bx1, y0=lower, y1=lower + pstep,
                              fillcolor=_rgba(base, op), line={"width": 0}, layer="above")

    # ---- SR ------------------------------------------------------------- #
    if show_sr:
        csel = float(track.close[selected])
        for j in range(SR_MAX):
            if not track.sr_valid[selected, j]:
                continue
            top = float(track.sr_top[selected, j])
            bot = float(track.sr_bottom[selected, j])
            if top > csel and bot > csel:
                fill = "rgba(242, 54, 69, 0.16)"
            elif top < csel and bot < csel:
                fill = "rgba(38, 166, 154, 0.16)"
            else:
                fill = "rgba(130, 130, 130, 0.16)"
            fig.add_shape(type="rect", xref="x", yref="y", x0=lo, x1=selected,
                          y0=bot, y1=top, fillcolor=fill,
                          line={"color": "rgba(180,180,180,0.4)", "width": 1},
                          opacity=0.55, layer="below")

    # ---- Liquidity ------------------------------------------------------ #
    if show_liq:
        _draw_liq(fig, track, selected, lo, side=+1)
        _draw_liq(fig, track, selected, lo, side=-1)

    # ---- selectable hit layer (direct K-line click) --------------------- #
    # One real selectable marker per visible bar, spanning exactly the
    # viewport. Each point carries its TF bar index so a click maps back to
    # `selected`. Very low opacity but still clickable (not an invisible
    # size-1 trick that cannot be hit).
    cdata = [[int(x), str(t)] for x, t in zip(idx_v, time_v)]
    fig.add_trace(go.Scatter(
        x=idx_v, y=track.close[view], mode="markers",
        marker={"size": 10, "color": "rgba(255,255,255,0.07)", "line": {"width": 0}},
        customdata=cdata,
        hoverinfo="skip", showlegend=False, name="iv_hit",
    ))

    # ---- layout --------------------------------------------------------- #
    right = selected + PROFILE_OFFSET + 40
    tick_step = max(1, len(idx_v) // 10)
    tick_vals = idx_v[::tick_step].tolist()
    tick_text = [t.strftime("%Y-%m-%d %H:%M") for t in time_v[::tick_step]]
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=C_BG, plot_bgcolor=C_BG,
        font={"color": C_TEXT, "size": 11},
        xaxis={
            "range": [lo, right], "gridcolor": C_GRID,
            "tickmode": "array", "tickvals": tick_vals, "ticktext": tick_text,
            "title": None,
        },
        yaxis={"gridcolor": C_GRID, "title": None, "side": "right"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 40, "r": 60, "t": 30, "b": 30},
        showlegend=False,
        hovermode="x unified",
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def segment_start_global(track, i: int) -> int:
    """First GLOBAL TF index of the segment containing bar ``i``."""
    seg = int(track.segment[i])
    return int(np.argmax(track.segment == seg))


def segment_local_to_global(track, selected, local_index) -> int:
    """Map a canonical SEGMENT-LOCAL indicator index to the global Plotly x.

    The canonical single-pass ``IndicatorState`` is stepped with ``ci``, which
    RESTARTS at 0 on every segment change (reset). Therefore anything produced
    inside it -- notably Liquidity ``level["left"]`` -- is a SEGMENT-LOCAL TF
    index, whereas the Viewer x-axis (``selected`` / ``lo`` / ``idx_v``) is the
    GLOBAL TF index. Such values MUST be converted before being drawn.

    Within one segment ``ci == global_index - segment_start_global``.
    """
    return segment_start_global(track, selected) + int(local_index)


def first_seen_i(valid, left, level, selected, slot, lo) -> int:
    """First bar at which this liquidity level object exists (creation bar).

    Pine only starts drawing a level once the cluster has formed, so the
    creation bar must be known to avoid drawing an extension that never
    existed before discovery. Level identity = (left, level).

    Walk back from ``selected`` while the same identity is visible. The scan is
    bounded by the viewport start ``lo``; anything created at or before ``lo``
    is reported as ``lo`` (callers clamp with ``max(lo, ...)`` anyway).
    """
    kl = int(left[selected, slot])
    kv = float(level[selected, slot])
    i = int(selected)
    while i > lo:
        seen = False
        for j in range(LIQ_VISIBLE):
            if (
                valid[i, j]
                and int(left[i, j]) == kl
                and abs(float(level[i, j]) - kv) <= 1e-12
            ):
                seen = True
                break
        if not seen:
            return i + 1
        i -= 1
    return lo


def _draw_liq(fig, track, selected, lo, side):
    if side > 0:
        valid = track.liq_up_valid; left = track.liq_up_left
        level = track.liq_up_level; broken = track.liq_up_broken
        ze = track.liq_up_zone_exists; za = track.liq_up_zone_active
        zl = track.liq_up_zone_left; zr = track.liq_up_zone_right
        ztop = track.liq_up_zone_top; zbot = track.liq_up_zone_bottom
        color = C_BUY; label = "Buyside"
    else:
        valid = track.liq_down_valid; left = track.liq_down_left
        level = track.liq_down_level; broken = track.liq_down_broken
        ze = track.liq_down_zone_exists; za = track.liq_down_zone_active
        zl = track.liq_down_zone_left; zr = track.liq_down_zone_right
        ztop = track.liq_down_zone_top; zbot = track.liq_down_zone_bottom
        color = C_SELL; label = "Sellside"

    for j in range(LIQ_VISIBLE):
        if not valid[selected, j]:
            continue
        lvl = float(level[selected, j])
        # `left` comes from the canonical LiquidityState, whose `ci` restarts at
        # 0 per segment -> it is SEGMENT-LOCAL. Convert to the global x used by
        # the Plotly axis. (zone_left/right and breach_i are already global --
        # they are written by the Viewer with the global row index `i`.)
        left_local = int(left[selected, j])
        left_global = segment_local_to_global(track, selected, left_local)
        brk = bool(broken[selected, j])

        # The level object became visible at `first_seen` (its creation bar),
        # already a GLOBAL TF index (it scans global track rows).
        # Pine never draws anything for this level before that bar.
        fs = first_seen_i(valid, left, level, selected, j, lo)

        # solid: max(lo, left_global) -> first_seen - 1
        solid_from = max(lo, left_global)
        solid_to = min(fs - 1, selected)
        if solid_to > solid_from:
            fig.add_shape(type="line", xref="x", yref="y",
                          x0=solid_from, x1=solid_to, y0=lvl, y1=lvl,
                          line={"color": color, "width": 2}, layer="above")

        # dotted: max(lo, first_seen - 1) -> lifecycle end (never before discovery)
        dotted_from = max(lo, fs - 1)
        if dotted_from < selected:
            # unbroken -> selected ; breached/closed -> zone_right (frozen)
            end_i = float(selected)
            if brk and bool(ze[selected, j]) and np.isfinite(zr[selected, j]):
                end_i = float(zr[selected, j])
            end_i = min(max(end_i, lo), selected)
            if end_i > dotted_from:
                fig.add_shape(type="line", xref="x", yref="y",
                              x0=dotted_from, x1=end_i, y0=lvl, y1=lvl,
                              line={"color": _rgba(color, 0.5), "width": 1, "dash": "dot"},
                              layer="above")
        # label
        fig.add_annotation(x=selected, y=lvl, text=f"{label} {lvl:.2f}",
                           showarrow=False, font={"color": color, "size": 10},
                           yshift=10, xanchor="right")

        # post-break zone (keep frozen geometry after close)
        if bool(ze[selected, j]) and np.isfinite(zl[selected, j]) and np.isfinite(zr[selected, j]):
            zactive = bool(za[selected, j])
            fig.add_shape(type="rect", xref="x", yref="y",
                          x0=float(zl[selected, j]), x1=float(zr[selected, j]),
                          y0=float(zbot[selected, j]), y1=float(ztop[selected, j]),
                          fillcolor=_rgba(color, 0.45 if zactive else 0.18),
                          line={"color": _rgba(color, 0.9 if zactive else 0.4), "width": 1},
                          layer="above")


# --------------------------------------------------------------------------- #
# Streamlit page                                                               #
# --------------------------------------------------------------------------- #
# Top-level cache owners (stable, readable, unit-testable; no nested per-call
# @st.cache_data closure that is rebuilt on every invocation).
@st.cache_data(show_spinner="加载 5m 基础数据…")
def load_base_cached(symbol: str):
    b = FormingEnvironmentBuilder(symbol=symbol)
    b.load_raw()
    return b.base


@st.cache_resource(show_spinner="构建指标时间轴…")
def build_track_cached(symbol: str, tf: str, source_sha: str):
    # cache_resource (not cache_data): the ViewerTrack is a large immutable
    # bundle of numpy arrays. Returning the same in-memory object on every hit
    # avoids the per-rerun pickle/copy cost of cache_data (FIX1 point 5).
    base = load_base_cached(symbol)
    return build_viewer_track(base, tf, symbol=symbol, source_sha=source_sha)


@st.cache_data(show_spinner="构建候选区域段…")
def candidate_segments_cached(symbol: str, source_sha: str):
    """Per-symbol candidate segments + frozen-truth lookup.

    Reads the canonical R3 candidate gate artifact (ONE canonical owner,
    shared with DP and the future Model). Args are strings only, so cache hits
    are cheap (no array hashing); switching bars never re-runs the candidate
    computation (FIX1 point 2 / R3 checkpoint A).
    """
    cand_sym = load_candidate_rows(symbol)
    track5 = build_track_cached(symbol, "5m", source_sha)
    return build_candidate_segments(cand_sym, track5)


@st.cache_data(show_spinner="加载触发证据 (touch proof)…")
def load_candidate_proof_cached(symbol: str):
    """Per-symbol R3 touch-proof sidecar (FIX2: exact hit zones per trigger bar).

    Reads the canonical proof artifact via the verified loader — the Viewer must
    never re-run ``bar_hits_zone`` itself. Keyed by symbol only (string arg), so
    cache hits are cheap; the proof is computed once at artifact-generation time.
    """
    return load_candidate_proof_verified(symbol)


@st.cache_data(show_spinner="加载 R4 候选区域…")
def candidate_segments_r4_cached(symbol: str, source_sha: str):
    """Per-symbol R4 (15m) candidate segments + frozen-truth lookup.

    Reads the canonical R4 candidate gate artifact (ONE canonical owner, shared with
    DP and the future Model). The 15m candidate maps 1:1 onto the 15m ViewerTrack.
    """
    cand_sym = candidate_for_symbol_r4(load_candidate_rows_r4(symbol), symbol)
    track = build_track_cached(symbol, "15m", source_sha)
    return build_candidate_segments_r4(cand_sym, track)


@st.cache_data(show_spinner="映射 R4 候选 bar（bar-level）…")
def candidate_marks_r4_cached(symbol: str, source_sha: str):
    """ALL candidate bars mapped onto the 15m track — one mark per candidate bar.

    AUDIT-FIX1: the audit unit is a single bar pair (Trigger(t-1) -> Candidate(t)),
    never a merged episode. Cached per (symbol, source_sha) so Prev/Next clicks
    never re-map (point 9: no per-click history scan).
    """
    rows = candidate_for_symbol_r4(load_candidate_rows_r4(symbol), symbol)
    track = build_track_cached(symbol, "15m", source_sha)
    return build_candidate_marks_r4(rows, track)


@st.cache_data(show_spinner="加载 R4 触发证据…")
def load_candidate_proof_r4_cached(symbol: str):
    """Per-symbol R4 touch-proof sidecar (exact hit zones per 15m trigger bar)."""
    from research.liquidity_oracle_atlas.build_candidate_gate_r4_m15_v1 import (
        load_candidate_proof_verified as _load,
    )
    return _load(symbol)


@st.cache_data(show_spinner="加载 R4 执行帧…")
def load_exec_frame_r4_cached(symbol: str):
    from research.liquidity_oracle_atlas.build_execution_frame_m15_v1 import (
        load_execution_frame_m15_verified as _load,
    )
    return _load(symbol)


@st.cache_data(show_spinner="构建 R4 环境特征…")
def load_env_features_r4_cached(symbol: str):
    from research.liquidity_oracle_atlas.build_execution_environment_m15_v1 import (
        run_environment_m15,
    )
    return run_environment_m15(symbol)["features"]


@st.cache_resource(show_spinner="构建 forming-MTF 环境…")
def load_forming_env_cached(symbol: str):
    # Canonical forming-MTF owner (build_forming_environment_v1). Returns the
    # per-5m-decision-bar FORMING DTP/SR/Liquidity state for 15m/1h/4h.
    # NOTE: the 5m forming state is identically the ViewerTrack's selected
    # snapshot (parity-tested), so we omit m5 here — running m5 through the
    # forming owner is ~35s of trivial 1-bar previews for no extra information.
    # cache_resource: large immutable DataFrame, returned by reference.
    b = FormingEnvironmentBuilder(symbol=symbol)
    b.load_raw().prepare()
    b.tf_minutes = [15, 60, 240]
    df, _ = b.run()
    return df


@st.cache_data(show_spinner="加载 DP Oracle artifact…")
def load_oracle_artifact_cached(
    root: str, symbol: str, math_version: str, cache_token: str
):
    """Read-only, fail-closed oracle artifact load.

    Cache key = (root, symbol, math_version, cache_token). ``cache_token`` is the
    artifact metadata mtime/size, so REGENERATING the same root/symbol/math_version
    artifact invalidates the cache. The Viewer NEVER recomputes the DP on rerun.
    """
    return load_oracle_artifact(root, symbol, expected_math_version=math_version)


@st.cache_data(show_spinner="加载 R4 15m DP Oracle artifact…")
def load_oracle_artifact_v4_cached(
    root: str, symbol: str, math_version: str, cache_token: str
):
    """R4 15m Oracle (decoupled R2-15m port) read-only, fail-closed load.

    Same cache contract as the legacy loaders; the 15m overlay NEVER falls back
    to a V2 (5m) artifact.
    """
    return load_oracle_artifact_m15(root, symbol, expected_math_version=math_version)


def resolve_selection(symbol, tf, n_bars, prev_selected, prev_ctx):
    """Pure data-selection contract for Symbol / Timeframe switching.

    Returns ``(selected, ctx)``:
      * symbol/TF changed -> selection resets to the LATEST bar;
      * otherwise -> previous selection clamped into ``[0, n_bars-1]``.

    Pure (no Streamlit) so the contract can be regression-tested directly.
    """
    ctx = (symbol, tf)
    last = max(0, int(n_bars) - 1)
    if prev_ctx != ctx:
        return (last, ctx)
    if prev_selected is None:
        return (last, ctx)
    return (int(min(max(int(prev_selected), 0), last)), ctx)


def main() -> None:
    st.set_page_config(page_title="指标观察器", layout="wide")
    st.markdown("### 指标观察器  ·  Historical-as-of")
    st.caption("将选中 K 线当作该时刻最后一根已形成的 K 线；基础视图只显示该 TF 三个原始指标"
               "当时的状态，且不绘制 selected 之后的任何内容。另有一个**可选**的 "
               "DP Oracle **FUTURE / HINDSIGHT AUDIT** 覆盖层（默认关闭、只读、裁剪到 selected）；"
               "它不是因果信号，也不参与任何模型。不含 Label / PGM / 模型结论。")

    st.session_state.setdefault("iv_symbol", SYMBOLS[0])
    st.session_state.setdefault("iv_tf", "1H")
    st.session_state.setdefault("iv_show_dtp", True)
    st.session_state.setdefault("iv_show_sr", True)
    st.session_state.setdefault("iv_show_liq", True)
    st.session_state.setdefault("iv_show_oracle", False)
    st.session_state.setdefault("iv_show_candidate", False)
    st.session_state.setdefault("iv_show_dp_proximity", False)
    st.session_state.setdefault("iv_show_lifecycle", False)

    # ---- toolbar columns ------------------------------------------------ #
    col_sym, col_tf, col_date, col_bt, col_prev, col_next, c1, c2, c3 = st.columns(
        [1.1, 1.2, 1.6, 1.1, 0.6, 0.6, 0.6, 0.6, 0.6])

    # (1) Symbol / Timeframe widgets FIRST. They own their own state via key=,
    #     so the returned value is already THIS rerun's choice.
    with col_sym:
        symbol = st.selectbox("Symbol", SYMBOLS,
                              index=SYMBOLS.index(st.session_state.iv_symbol),
                              key="iv_symbol")
    with col_tf:
        tf = st.selectbox("Timeframe", TF_LABELS,
                          index=TF_LABELS.index(st.session_state.iv_tf),
                          key="iv_tf")
    with c1:
        st.session_state.iv_show_dtp = st.checkbox("DTP", value=st.session_state.iv_show_dtp)
    with c2:
        st.session_state.iv_show_sr = st.checkbox("SR", value=st.session_state.iv_show_sr)
    with c3:
        st.session_state.iv_show_liq = st.checkbox("Liquidity", value=st.session_state.iv_show_liq)

    # Candidate-zone audit overlay (visual audit only; frozen T2 truth).
    with st.sidebar:
        st.session_state.iv_show_candidate = st.checkbox(
            "Show Candidate Trading Zones",
            value=st.session_state.iv_show_candidate,
            help="Frozen Oracle candidate regions from the formal T2 parquet. "
                 "Decision axis = 5m; no t+1 shift; shows candidate truth only "
                 "(no model / Y / Q / Oracle action).",
        )
        # AUDIT-FIX1 point 8: NO "All bars / Candidate episodes only" radio.
        # The manual-audit visual unit is ONE bar pair
        # (Trigger(t-1) -> Candidate(t)); episode shading is a DP concept and
        # is disabled for manual audit.
        if st.session_state.iv_show_candidate:
            st.caption(
                "R4 bar-level audit: one candidate = ONE blue 15m bar; its trigger "
                "(candidate idx - 1) = ONE orange bar. No episode merging."
            )

        # DP-internal proximity audit (decoupled from R4 Candidate).
        # Shows the 0.5-ATR proximity band the DP itself uses — NOT the Candidate
        # Trading Zone. Debug / visual separation only.
        st.session_state.iv_show_dp_proximity = st.checkbox(
            "Show DP Proximity Audit",
            value=st.session_state.iv_show_dp_proximity,
            help="0.5-ATR proximity — DP INTERNAL ONLY (15m/1h/4h SR/LIQ). "
                 "This is NOT the R4 Candidate Trading Zone. Visual separation/debug.",
        )

    # DP Oracle audit overlay — OFF by default, read-only, 5m-clock only.
    st.session_state.iv_show_oracle = st.checkbox(
        "DP Oracle — FUTURE / HINDSIGHT AUDIT",
        value=st.session_state.iv_show_oracle,
        help="Future-derived hindsight labels. For manual audit only; NOT a causal "
             "trading signal and never a model feature.",
    )

    # DP Label Lifecycle Gate (FUT-M15-DP-LABEL-VIZ-GATE-01) — read-only overlay.
    st.session_state.iv_show_lifecycle = st.checkbox(
        "DP Label Lifecycle Gate (15m)",
        value=st.session_state.iv_show_lifecycle,
        help="Draws the canonical DP trade lifecycle: Candidate Region (DP proximity "
             "episode) -> one ENTRY -> holding -> EXIT -> next Candidate, plus the "
             "Lifecycle table + auto-assertions (A-E) + negative controls. Read-only; "
             "does NOT recompute the DP.",
    )

    # (2) NOW build the track from the CURRENT widget values, so the chart
    #     always corresponds to the dropdown in the SAME rerun.
    timing: Dict[str, float] = {}
    _t0 = time.perf_counter()
    track = build_track_cached(symbol, tf, git_head())
    timing["track_build_ms"] = (time.perf_counter() - _t0) * 1000.0

    # (2b) Candidate-zone overlay data (frozen R4 truth, 15m decision axis).
    # R4: the only valid candidate decision axis is 15m. The 5m axis is raw
    # plumbing only and never a candidate / feature / decision variable.
    # The per-symbol segments are served from a cached owner, so switching bars
    # never re-runs the candidate groupby.
    segments, cand_audit, cand_by_time = [], EMPTY_CAND_AUDIT, {}
    if st.session_state.iv_show_candidate and tf == "15m":
        _t0 = time.perf_counter()
        try:
            segments, cand_audit, cand_by_time = candidate_segments_r4_cached(
                track.symbol, git_head()
            )
            timing["candidate_segment_ms"] = (time.perf_counter() - _t0) * 1000.0
            timing["candidate_load_ms"] = 0.0  # artifact read inside cached owner
            try:
                _summ = load_candidate_gate_summary_r4()
                timing["candidate_math_version"] = _summ.get("candidate_math_version", "")
                _sa = _summ.get("symbol_artifacts", {}).get(symbol, {})
                timing["candidate_artifact_sha"] = _sa.get("sha256", "")[:12]
            except FileNotFoundError:
                timing["candidate_math_version"] = "MISSING"
                timing["candidate_artifact_sha"] = "MISSING"
        except Exception as _e:
            st.warning(f"R4 candidate gate not available: {_e}")
            segments, cand_audit, cand_by_time = [], EMPTY_CAND_AUDIT, {}

    # (3) Selection: reset to latest bar when symbol/TF changed.
    prev_ctx = st.session_state.get("_iv_ctx")
    prev_sel = st.session_state.get("iv_selected")
    selected, ctx = resolve_selection(symbol, tf, track.n, prev_sel, prev_ctx)
    st.session_state["_iv_ctx"] = ctx
    st.session_state.iv_selected = selected
    selected = int(st.session_state.iv_selected)
    # deterministic selectors (fallback if click fails)
    dates = pd.to_datetime(track.time).normalize()
    uniq = sorted(set(dates))
    cur_date = dates[selected]
    with col_date:
        di = uniq.index(cur_date)
        new_di = st.selectbox("Date", uniq, index=di, format_func=lambda d: d.strftime("%Y-%m-%d"))
        if new_di != cur_date:
            # last bar of that date
            idxs = np.where(dates == new_di)[0]
            st.session_state.iv_selected = int(idxs[-1])
            st.rerun()
    with col_bt:
        same_day = np.where(dates == cur_date)[0]
        times = [pd.Timestamp(track.time[i]).strftime("%H:%M") for i in same_day]
        cur_pos = int(np.searchsorted(same_day, selected))
        cur_pos = min(max(cur_pos, 0), len(same_day) - 1)
        new_pos = st.selectbox("Bar time", times, index=cur_pos)
        target = int(same_day[times.index(new_pos)])
        if target != selected:
            st.session_state.iv_selected = target
            st.rerun()
    with col_prev:
        if st.button("Prev"):
            st.session_state.iv_selected = max(0, selected - 1)
            st.rerun()
    with col_next:
        if st.button("Next"):
            st.session_state.iv_selected = min(track.n - 1, selected + 1)
            st.rerun()

    selected = int(st.session_state.iv_selected)
    snap = selected_snapshot(track, selected)

    # ---- DP Oracle audit overlay (read-only, fail-closed) ---------------- #
    # R4 V4 (FUTURE-R4-M15-DP-ORACLE-V4): on the 15m decision axis the overlay
    # loads ONLY the V4 15m artifact and NEVER the legacy V2 (5m) artifact.
    # The 5m overlay keeps the legacy V2 path unchanged.
    show_oracle = bool(st.session_state.iv_show_oracle)
    oracle_trades = None
    oracle_meta = None
    oracle_tf = None  # decision clock of the loaded artifact ("5m" / "15m")
    lifecycle_records = None  # DP Label Lifecycle Gate (15m)
    if show_oracle:
        st.warning(
            "**DP Oracle uses future prices to find hindsight-optimal intraday "
            "trades. It is for audit / research labels only — NOT a causal "
            "trading signal and never a model feature.**"
        )
        if tf == "15m":
            _loaded = load_oracle_artifact_v4_cached(
                str(ORACLE_ARTIFACT_ROOT_V4),
                symbol,
                DP_M15_MATH_VERSION,
                oracle_cache_token(str(ORACLE_ARTIFACT_ROOT_V4), symbol),
            )
            if not _loaded["ok"]:
                st.error(
                    f"15m DP Oracle overlay disabled (fail-closed): "
                    f"`{_loaded['reason']}`. Generate the artifact: "
                    f"`python -m research.liquidity_oracle_atlas."
                    f"build_trade_oracle_dp_m15_one_entry_proximity_v1 {symbol}`"
                )
            else:
                oracle_trades = _loaded["trades"]
                oracle_meta = _loaded["metadata"]
                oracle_tf = "15m"
                timing["oracle_integrity"] = oracle_meta.get("integrity")
                st.session_state["_iv_oracle_trades"] = oracle_trades
        elif tf == ORACLE_TF_ONLY:
            loaded = load_oracle_artifact_cached(
                str(ORACLE_ARTIFACT_ROOT),
                symbol,
                MATH_VERSION,
                oracle_cache_token(str(ORACLE_ARTIFACT_ROOT), symbol),
            )
            if not loaded["ok"]:
                st.error(f"Oracle overlay disabled (fail-closed): `{loaded['reason']}`.")
            else:
                oracle_trades = loaded["trades"]
                oracle_meta = loaded["metadata"]
                oracle_tf = ORACLE_TF_ONLY
        else:
            st.info(
                "Oracle executions are defined on the 5m (legacy V2) and 15m "
                "(R4 V4) decision clocks. Switch to 5m or 15m to inspect exact "
                "Entry / Exit points."
            )
        if oracle_trades is not None:
            _rec, _mism = select_visible_oracle_trades(
                track, selected, oracle_trades, tf_label=oracle_tf
            )
            if _mism > 0:
                st.error(
                    f"Oracle overlay disabled (fail-closed): time alignment "
                    f"mismatch on {_mism} visible trade(s)."
                )
                oracle_trades = None
            else:
                _summ = oracle_viewport_summary(
                    track, selected, oracle_trades, tf_label=oracle_tf
                )
                st.caption(
                    f"Oracle audit ({oracle_tf}) · visible trades={_summ['visible_trades']} "
                    f"(L={_summ['long_trades']} / S={_summ['short_trades']}) · "
                    f"closed(<= selected)={_summ['closed_trades']} · "
                    f"open at selected={_summ['open_at_selected']} · "
                    f"total gross oracle PnL (closed)={_summ['total_gross_points']:.2f} pts · "
                    f"median holding={_summ['median_holding_bars']:.0f} bars · "
                    f"objective={oracle_meta.get('objective')} · "
                    f"cost_mode={oracle_meta.get('cost_mode')}"
                )
                if oracle_meta.get("oracle_source_sha") != git_head():
                    st.warning(
                        "Oracle artifact oracle_source_sha="
                        f"`{oracle_meta.get('oracle_source_sha')}` != current "
                        f"HEAD=`{git_head()}` — possibly stale artifact."
                    )

    # DP-internal proximity audit (decoupled from R4 Candidate Trading Zones).
    if st.session_state.iv_show_dp_proximity and tf == "15m":
        _pa = load_oracle_artifact_v4_cached(
            str(ORACLE_ARTIFACT_ROOT_V4),
            symbol,
            DP_M15_MATH_VERSION,
            oracle_cache_token(str(ORACLE_ARTIFACT_ROOT_V4), symbol),
        )
        if not _pa["ok"]:
            st.error(
                f"DP Proximity audit disabled (fail-closed): `{_pa['reason']}`. "
                f"Generate the artifact first."
            )
        else:
            _acts = _pa["actions"]
            _dt = pd.to_datetime(_acts["decision_time"])
            _pa_flag = _acts["dp_proximity_any"].to_numpy(bool)
            import matplotlib.pyplot as plt

            _f, _ax = plt.subplots(figsize=(10, 1.4))
            _ax.fill_between(_dt, 0, 1, where=_pa_flag, color="#3a6ea5",
                             step="mid", alpha=0.5)
            _ax.set_yticks([])
            _ax.set_ylim(0, 1)
            _ax.set_title(
                "DP-internal 0.5-ATR proximity (15m/1h/4h SR/LIQ) — NOT Candidate",
                fontsize=9,
            )
            st.pyplot(_f)
            st.caption(
                "Blue bands = bars where the DP's OWN proximity P_t=1 (distance to any "
                "pre-existing 15m/1h/4h SR/LIQ <= 0.5 ATR). This is the DP's internal "
                "entry-permission gate only; it is independent of the R4 Candidate "
                "Trading Zones."
            )

    # ---- main + snapshot ------------------------------------------------ #
    chart_col, snap_col = st.columns([4, 1])
    with chart_col:
        _t0 = time.perf_counter()
        fig = build_figure(track, selected, st.session_state.iv_show_dtp,
                           st.session_state.iv_show_sr, st.session_state.iv_show_liq)

        # ---- DP Label Lifecycle Gate (FUT-M15-DP-LABEL-VIZ-GATE-01) ---------- #
        # Read-only overlay: shades each DP "candidate region" (its proximity
        # episode band) and labels "Candidate #k" at the entry. Reuses the SAME
        # canonical 15m oracle artifact as the DP overlay; never recomputes DP.
        if st.session_state.iv_show_lifecycle and tf == "15m":
            _lacy = load_oracle_artifact_v4_cached(
                str(ORACLE_ARTIFACT_ROOT_V4), symbol, DP_M15_MATH_VERSION,
                oracle_cache_token(str(ORACLE_ARTIFACT_ROOT_V4), symbol))
            if not _lacy["ok"]:
                st.error(
                    f"DP Label Lifecycle disabled (fail-closed): `{_lacy['reason']}`. "
                    f"Generate the artifact first: `python -m "
                    f"research.liquidity_oracle_atlas."
                    f"build_trade_oracle_dp_m15_one_entry_proximity_v1 {symbol}`"
                )
            else:
                lifecycle_records = build_lifecycle_records(
                    _lacy["trades"], _lacy["actions"])
                lo_v, hi_v = compute_viewport(track, int(selected))
                for _r in lifecycle_records:
                    if _r.band_lo < 0 or _r.band_hi < 0:
                        continue
                    if _r.band_hi < lo_v or _r.band_lo > hi_v:
                        continue
                    _is_cur = (
                        (_r.entry_fill_index <= int(selected) <= _r.exit_fill_index)
                        or (_r.band_lo <= int(selected) <= _r.band_hi)
                    )
                    _col = ("rgba(255,209,102,0.16)" if _is_cur
                            else "rgba(56,128,255,0.10)")
                    fig.add_vrect(x0=_r.band_lo - 0.5, x1=_r.band_hi + 0.5,
                                  fillcolor=_col, line_width=0, layer="below")
                    fig.add_annotation(
                        x=_r.entry_fill_index, y=_r.entry_price,
                        text=f"C#{_r.candidate_id}", showarrow=True, arrowhead=2,
                        ax=0, ay=-28, font={"size": 9, "color": "#FFD166"},
                        opacity=0.9)
        if st.session_state.iv_show_candidate:
            if tf == "15m":
                # AUDIT-FIX1: BAR-LEVEL audit view. One candidate = ONE blue 15m
                # bar; its trigger bar (candidate idx - 1) = ONE orange bar.
                # Merged episode shading is DISABLED for manual audit.
                lo, hi = compute_viewport(track, selected)
                all_marks = candidate_marks_r4_cached(track.symbol, git_head())
                marks = [m for m in all_marks if lo <= m["cand_track_idx"] <= hi]
                try:
                    proof_df = load_candidate_proof_r4_cached(symbol)
                except Exception as _e:
                    st.error(f"R4 touch proof unavailable (fail-closed): {_e}")
                    proof_df = None
                vaudit = (
                    run_viewport_candidate_audit(marks, proof_df)
                    if proof_df is not None else None
                )
                timing["viewport_audit"] = vaudit
                # Fail-closed invariants (points 3/4/5): hard STOP on violation.
                if vaudit is not None and vaudit["missing_m15_proof"] > 0:
                    _bad = ", ".join(
                        f"{i}@{t}" for i, t in zip(
                            vaudit["missing_proof_indices"][:10],
                            vaudit["missing_proof_times"][:10],
                        )
                    )
                    st.error(
                        f"STOP_R4_VIEWER_CANDIDATE_WITHOUT_M15_PROOF :: "
                        f"missing={vaudit['missing_m15_proof']} ; "
                        f"bad[15m idx@decision_time] = {_bad}"
                    )
                if vaudit is not None and vaudit["bits_proof_mismatch"] > 0:
                    st.error(
                        "STOP_R4_TRIGGER_BITS_PROOF_MISMATCH :: "
                        + " | ".join(vaudit["mismatch_details"][:10])
                    )
                if (
                    vaudit is not None
                    and vaudit["missing_m15_proof"] == 0
                    and vaudit["bits_proof_mismatch"] == 0
                ):
                    # orange trigger bar + blue candidate bar (one bar each)
                    for m in marks:
                        if m["trig_track_idx"] >= 0:
                            fig.add_vrect(
                                x0=m["trig_track_idx"] - 0.5,
                                x1=m["trig_track_idx"] + 0.5,
                                fillcolor="rgba(255,165,0,0.30)",
                                line_width=0, layer="below",
                            )
                        fig.add_vrect(
                            x0=m["cand_track_idx"] - 0.5,
                            x1=m["cand_track_idx"] + 0.5,
                            fillcolor="rgba(56,128,255,0.30)",
                            line_width=0, layer="below",
                        )
                    # exact HISTORICAL m15 hit-zone for EVERY visible trigger
                    # (audit-panel-only for h1/h4 — never drawn on the main chart)
                    _pbt = build_proof_by_trigger(proof_df)
                    for m in marks:
                        _t = m["trig_track_idx"]
                        if _t < 0:
                            continue
                        _rows15 = [
                            r for r in _pbt.get(m["trigger_bar_index"], [])
                            if r["tf"] == "m15"
                        ]
                        _ytop = None
                        for r in _rows15:
                            _is_sr = r["family"] == "SR"
                            _color = "rgba(255,193,7,1)" if _is_sr else "rgba(255,140,0,1)"
                            fig.add_shape(
                                type="rect", xref="x", yref="y",
                                x0=_t - 0.5, x1=_t + 0.5,
                                y0=float(r["bottom"]), y1=float(r["top"]),
                                fillcolor="rgba(0,0,0,0)",
                                line={"color": _color, "width": 2},
                                layer="above",
                            )
                            _ytop = (
                                float(r["top"])
                                if _ytop is None else max(_ytop, float(r["top"]))
                            )
                        if _rows15:
                            _has_sr = any(r["family"] == "SR" for r in _rows15)
                            _has_liq = any(r["family"] == "LIQ" for r in _rows15)
                            _lbl = (
                                "S+L" if (_has_sr and _has_liq)
                                else ("S" if _has_sr else "L")
                            )
                            fig.add_annotation(
                                x=_t, y=_ytop, text=_lbl, showarrow=False,
                                yanchor="bottom", yshift=2,
                                font={"color": "rgba(255,200,80,1)", "size": 10},
                            )
                    # selected-candidate orientation line at ITS trigger bar
                    _cstate = candidate_state_at_r4(cand_by_time, selected)
                    if _cstate["is_candidate"] and int(_cstate["trigger_bar_index"]) >= 0:
                        _tbi = int(_cstate["trigger_bar_index"])
                        if 0 <= _tbi < track.n:
                            fig.add_vline(
                                x=_tbi - 0.5, line_color="gold", line_width=2.5,
                                opacity=0.9,
                            )
                if vaudit is not None:
                    timing["candidate_bars_drawn"] = len(marks)
                    st.caption(
                        f"R4 bar-level audit: BLUE bar = candidate (one 15m bar); ORANGE bar = "
                        f"its trigger (candidate idx - 1). Outlined zone = HISTORICAL 15m proof "
                        f"at the trigger (S=SR, L=LIQ), spanning only that trigger bar — visually "
                        f"distinct from the CURRENT SR/Liquidity drawn across the viewport; never "
                        f"judge a past candidate against current SR. Counters: "
                        f"visible_candidates={vaudit['visible_candidates']} · "
                        f"visible_triggers={vaudit['visible_triggers']} · "
                        f"candidates_with_m15_proof={vaudit['candidates_with_m15_proof']} · "
                        f"missing_m15_proof={vaudit['missing_m15_proof']} · "
                        f"bits_proof_mismatch={vaudit['bits_proof_mismatch']} "
                        f"(last three must be 0). Episode shading disabled for manual audit. "
                        f"No model / Y / Q / Oracle action shown.")
                    # per-Candidate navigation over the WHOLE symbol (bar-level)
                    cand_idx = sorted(cand_by_time.keys())
                    if cand_idx:
                        cp2, cn2 = st.columns(2)
                        with cp2:
                            if st.button("◀ Prev Candidate"):
                                prev = [x for x in cand_idx if x < selected]
                                if prev:
                                    st.session_state.iv_selected = int(max(prev))
                                    st.rerun()
                        with cn2:
                            if st.button("Next Candidate ▶"):
                                nxt = [x for x in cand_idx if x > selected]
                                if nxt:
                                    st.session_state.iv_selected = int(min(nxt))
                                    st.rerun()
            else:
                st.info(
                    "R4 Candidate zones are defined on the 15m decision axis. "
                    "Switch the primary chart to 15m for candidate-region audit.")
        if show_oracle and oracle_trades is not None and oracle_tf is not None:
            fig = add_dp_oracle_overlay(
                fig, track, selected, oracle_trades, tf_label=oracle_tf
            )
        timing["figure_build_ms"] = (time.perf_counter() - _t0) * 1000.0
        event = st.plotly_chart(
            fig, key="iv_chart", on_select="rerun", selection_mode="points",
            use_container_width=True)
        sel_idx = _parse_selection(event)
        if sel_idx is not None and sel_idx != st.session_state.iv_selected:
            st.session_state.iv_selected = sel_idx
            st.rerun()

    with snap_col:
        st.markdown("**Snapshot**")
        st.text(f"Symbol : {snap['symbol']}")
        st.text(f"TF     : {snap['tf']}")
        st.text(f"Bar    : {snap['bar_start_time']}")
        st.text(f"Avail  : {snap['available_time']}")
        st.text(f"Seg id : {snap['segment']}")
        st.text(f"OHLC   : {snap['o']:.2f}/{snap['h']:.2f}/{snap['l']:.2f}/{snap['c']:.2f}")
        d = snap["dtp"]
        st.markdown("**DTP**")
        st.text(f"trend  : {'UP' if d['trend'] == 1 else 'DOWN'}")
        st.text(f"SMA50  : {d['sma']:.3f}")
        st.text(f"ATR200 : {d['atr']:.3f}")
        st.text(f"score  : {d['trend_score']:.4f}")
        st.text(f"age    : {d['trend_age']}")
        st.markdown("**SR**")
        st.text(f"channels: {snap['sr_n_channels']}  in_zone: {snap['sr_in_zone']}")
        for k, ch in enumerate(snap["sr_channels"]):
            st.text(f"  #{k+1} {ch['top']:.2f}/{ch['bottom']:.2f} s={ch['strength']:.1f}")
        st.markdown("**Liquidity**")
        st.text(f"Buyside active : {snap['liq_up_count']}")
        st.text(f"Sellside active: {snap['liq_down_count']}")
        for k, lv in enumerate(snap["liq_up"]):
            st.text(f"  B#{k+1} {lv['level']:.2f} br={lv['broken']} z={lv['zone_active']}")
        for k, lv in enumerate(snap["liq_down"]):
            st.text(f"  S#{k+1} {lv['level']:.2f} br={lv['broken']} z={lv['zone_active']}")

        # ---- candidate audit (A: frozen truth, B: canonical state) ------ #
        # AUDIT-FIX1: R4 candidates live on the 15m decision axis; the panel
        # must render there (the old "5m" gate was an R3 leftover that made the
        # audit panel unreachable on the 15m chart).
        if st.session_state.iv_show_candidate and track.tf_label == "15m":
            _render_candidate_audit(track, symbol, selected, cand_by_time, cand_audit, segments, timing)

    # ---- DP Label Lifecycle Gate panel (FUT-M15-DP-LABEL-VIZ-GATE-01) ---- #
    if st.session_state.iv_show_lifecycle and track.tf_label == "15m":
        if lifecycle_records is None:
            _lacy = load_oracle_artifact_v4_cached(
                str(ORACLE_ARTIFACT_ROOT_V4), symbol, DP_M15_MATH_VERSION,
                oracle_cache_token(str(ORACLE_ARTIFACT_ROOT_V4), symbol))
            if _lacy["ok"]:
                lifecycle_records = build_lifecycle_records(
                    _lacy["trades"], _lacy["actions"])
        if lifecycle_records is not None:
            _render_dp_label_lifecycle(track, symbol, selected, lifecycle_records)

    # ---- bottom debug --------------------------------------------------- #
    with st.expander("Technical snapshot", expanded=False):
        st.text(f"global TF index : {snap['index']}")
        st.text(f"segment id      : {snap['segment']}")
        st.text(f"source SHA      : {track.source_sha}")
        st.text(f"counters        : raw_load={track.raw_load_count} resample={track.resample_count} "
                f"steps={track.indicator_step_count} full_recompute={track.full_history_recompute_count} "
                f"reference={track.reference_call_count} writes={track.visual_snapshot_write_count}")
        st.text("")  # FIX1 point 7: timing instrumentation
        st.text("RENDER TIMING (ms)")
        st.text(f"  track_build_ms        : {timing.get('track_build_ms', 0.0):.1f}")
        st.text(f"  candidate_load_ms     : {timing.get('candidate_load_ms', 0.0):.1f}")
        st.text(f"  candidate_segment_ms  : {timing.get('candidate_segment_ms', 0.0):.1f}")
        st.text(f"  figure_build_ms       : {timing.get('figure_build_ms', 0.0):.1f}")
        st.text(f"  forming_env_ms        : {timing.get('forming_env_ms', 0.0):.1f}")
        st.text(f"  candidate_bars_drawn  : {timing.get('candidate_bars_drawn', 0)}")
        st.text("")
        st.text("CANDIDATE ARTIFACT (shared by Viewer / DP / Model)")
        st.text(f"  math_version          : {timing.get('candidate_math_version', '—')}")
        st.text(f"  artifact SHA          : {timing.get('candidate_artifact_sha', '—')}")
        _ig = timing.get("oracle_integrity")
        if _ig is not None:
            st.text("R4 DP V4 PATH / TRADE INTEGRITY (must all be 0)")
            st.text(f"  path_errors           : {_ig.get('path_errors')}")
            st.text(f"  trade_sequence_errors : {_ig.get('trade_sequence_errors')}")
            st.text(
                "  viewer_event_errors   : "
                f"{_viewer_event_errors(st.session_state.get('_iv_oracle_trades'))}"
            )
        st.json({
            "dtp": d,
            "sr": snap["sr_channels"],
            "liq_up": snap["liq_up"],
            "liq_down": snap["liq_down"],
        })


def _viewer_event_errors(trades) -> int:
    """Re-validate the execution events the Viewer derived from the trades.

    The Viewer only CONSUMES validated trades; this is a cheap independent
    check that what it is about to draw is still a legal Entry/Exit sequence.
    """
    if trades is None or len(trades) == 0:
        return 0
    events = []
    for r in trades.sort_values(["entry_fill_index"]).to_dict("records"):
        d = str(r["direction"])
        events.append((int(r["entry_fill_index"]),
                       "LONG_ENTRY" if d == "LONG" else "SHORT_ENTRY"))
        events.append((int(r["exit_fill_index"]),
                       "LONG_EXIT" if d == "LONG" else "SHORT_EXIT"))
    events.sort(key=event_sort_key)
    return len(validate_execution_events(events))


def _ff(x, p: int = 2) -> str:
    """Format a forming-feature scalar; show '—' for NaN / missing."""
    try:
        v = float(x)
        if not np.isfinite(v):
            return "—"
        return f"{v:.{p}f}"
    except (TypeError, ValueError):
        return "—"


def _render_dp_label_lifecycle(track, symbol, selected, lifecycle_records):
    """Render the DP Label Lifecycle Gate panel (FUT-M15-DP-LABEL-VIZ-GATE-01).

    Read-only: reads ONLY the canonical 15m DP oracle artifact (cached), derives
    the lifecycle view, runs the A-E assertions + NC1-NC5 negative controls, and
    displays the Lifecycle table + prev/current/next timeline. No DP recompute,
    no model, no Layer-2.
    """
    st.markdown("---")
    st.markdown("### DP Label Lifecycle Gate · 15m")
    st.caption(
        "Read-only view of the canonical DP label artifact. One Candidate Region "
        "(DP proximity episode) -> exactly one ENTRY -> holding -> EXIT -> next "
        "Candidate. No DP recompute; no model; no Layer-2. Exit reasons are "
        "STRICTLY canonical: FLAT_EXIT / REVERSAL (this DP has no TP/STOP/TARGET)."
    )

    _lacy = load_oracle_artifact_v4_cached(
        str(ORACLE_ARTIFACT_ROOT_V4), symbol, DP_M15_MATH_VERSION,
        oracle_cache_token(str(ORACLE_ARTIFACT_ROOT_V4), symbol))
    if not _lacy["ok"]:
        st.error(f"DP Label Lifecycle: artifact unavailable (`{_lacy['reason']}`).")
        return
    trades = _lacy["trades"]
    recs = lifecycle_records
    ass = run_lifecycle_assertions(trades, recs)
    cases = find_representative_cases(trades, recs)
    neg = run_negative_controls(trades, recs)

    sel = int(selected)
    cur = next((r for r in recs if r.entry_fill_index <= sel <= r.exit_fill_index), None)
    if cur is None and recs:
        cur = min(recs, key=lambda r: abs(r.entry_fill_index - sel))

    # ---- Lifecycle status ----
    status = "VALID" if ass["all_pass"] else "INVALID"
    _gate_keys = [
        "A_one_entry_per_candidate", "B_no_overlapping_trades",
        "C_previous_closed_before_new_entry", "D_allowed", "E_one_label_per_candidate",
    ]
    _n_pass = sum(1 for k in _gate_keys if ass[k])
    st.markdown(f"**Lifecycle status: {status}**  (A-E pass = {_n_pass}/5)")

    # ---- manual-case navigation (task #13) ----
    st.markdown("**Jump to required manual cases**")
    cbtns = st.columns(4)
    _case_map = [
        ("Case1: prev exits → next", "case1_previous_exits_then_next"),
        ("Case2: next quick after exit", "case2_next_quick_after_exit"),
        ("Case3: same-direction cont.", "case3_same_direction_continuation"),
        ("Case4: region re-entered → 1 entry", "case4_longest_proximity_episode_one_entry"),
    ]
    for col, (label, key) in zip(cbtns, _case_map):
        with col:
            if st.button(label, key=f"lc_{key}"):
                cid = cases.get(key)
                _tgt = next((r for r in recs if r.candidate_id == cid), None)
                if _tgt is not None:
                    st.session_state.iv_selected = int(_tgt.entry_fill_index)
                    st.rerun()

    # ---- Lifecycle table for the current candidate ----
    if cur is not None:
        st.markdown(f"**Lifecycle table — Candidate #{cur.candidate_id}**")
        lrows = [
            ("candidate_id", cur.candidate_id),
            ("candidate_region_id", cur.candidate_region_id),
            ("candidate_time", cur.candidate_time),
            ("candidate_anchor", round(float(cur.candidate_anchor_price), 2)),
            ("direction", cur.direction),
            ("entry_time", cur.entry_time),
            ("entry_price", round(float(cur.entry_price), 2)),
            ("number_of_entries", cur.number_of_entries),
            ("exit_time", cur.exit_time),
            ("exit_price", round(float(cur.exit_price), 2)),
            ("exit_reason", cur.exit_reason),
            ("next_candidate_id", cur.next_candidate_id),
            ("next_candidate_time", cur.next_candidate_time),
            ("overlap_with_next", cur.overlap_with_next),
        ]
        st.table(pd.DataFrame(lrows, columns=["Field", "Value"]))

        st.markdown("**Previous / Current / Next timeline**")
        prev_exit = cur.prev_exit_time if cur.prev_candidate_id is not None else "—"
        next_entry = cur.next_entry_time if cur.next_candidate_id is not None else "—"
        tl = [
            ("Previous candidate", cur.prev_candidate_id, "exit", prev_exit),
            ("Current candidate", cur.candidate_id, "entry", cur.entry_time),
            ("Current candidate", cur.candidate_id, "exit", cur.exit_time),
            ("Next candidate", cur.next_candidate_id, "entry", next_entry),
        ]
        st.table(pd.DataFrame(tl, columns=["Candidate", "id", "event", "time"]))
        if cur.prev_candidate_id is not None:
            ok_prev = pd.Timestamp(prev_exit) <= pd.Timestamp(cur.entry_time)
            st.caption(
                f"Exit(prev) ≤ Entry(curr): {ok_prev}  |  "
                f"Exit(curr) ≤ Entry(next): {cur.overlap_with_next == False}"
            )

    # ---- assertions A-E ----
    with st.expander("Lifecycle assertions (A–E)", expanded=True):
        arows = [
            ("A. one candidate → one entry", ass["A_one_entry_per_candidate"],
             f"max entries/candidate={ass['A_max_entries_per_candidate']}"),
            ("B. no overlapping trades (exit ≤ next entry)", ass["B_no_overlapping_trades"],
             f"violations={len(ass['B_overlap_indices'])}"),
            ("C. prev trade closed before new entry", ass["C_previous_closed_before_new_entry"],
             f"violations={len(ass['C_violations'])}"),
            ("D. same-direction continuation allowed", ass["D_allowed"],
             f"count={ass['D_same_direction_continuation_count']}"),
            ("E. one label per candidate", ass["E_one_label_per_candidate"],
             f"violations={len(ass['E_violations'])}"),
        ]
        st.table(pd.DataFrame(arows, columns=["Gate", "PASS", "detail"]))

    # ---- negative controls NC1-NC5 ----
    with st.expander("Negative controls (NC1–NC5)", expanded=True):
        nrows = [(r["name"], r["expect"], "PASS" if r["passed"] else "FAIL") for r in neg]
        st.table(pd.DataFrame(nrows, columns=["Control", "Expected", "Result"]))
        st.caption(
            "NC1/NC3 must trip A (two entries in one candidate region); NC2/NC4 must "
            "trip B/C (overlapping positions); NC5 (LONG→LONG) must PASS — proving the "
            "gate checks position OVERLAP, not direction alternation."
        )


def _render_candidate_audit(track, symbol, selected, cand_by_time, cand_audit, segments, timing):
    """Render the candidate audit panel (Section 7/8/10/11/19).

    A. Frozen candidate truth from T2 (proximity flag + episode id);
    B. Canonical FORMING multi-timeframe state at the SAME 5m decision bar.
       FIX1 point 6: B is sourced from ``build_forming_environment_v1`` — the
       canonical forming-MTF owner — so the 15m/1h/4h states shown are the
       bars AS-FORMING at the 5m close, NOT the last fully-closed HTF bar.
       A and B are computed from independent sources and never cross-derived.
    """
    dt = pd.Timestamp(track.available_time[selected])
    st.markdown("**Candidate Audit**")
    state = candidate_state_at_r4(cand_by_time, selected)
    if not state["is_candidate"]:
        st.caption(
            "Candidate: NO at this bar (frozen T2 truth). "
            "Shaded bands mark Oracle candidates; this bar is not one."
        )
        return

    # A. frozen candidate truth  (R3 gate artifact = single source of truth)
    st.markdown("**A. Frozen candidate truth (R4 gate artifact)**")
    st.text(f"Decision time : {dt}")
    st.text(f"Candidate     : YES")
    st.text(f"Episode       : {state['episode']}")
    trig_dt = state.get("trigger_decision_time")
    st.text(f"Trigger bar   : {trig_dt}")
    # Decode the retained full 4TF true-touch mask of the trigger bar.
    tb = int(state["trigger_bits"])
    st.markdown("**Trigger context (true-touch at trigger bar, 15m close-known)**")
    for tf, fam in TRIGGER_BITS_R4:
        hit = bool(touch_bit_r4(np.uint8(tb), tf, fam)[0]) if tb else False
        kind = "ELIG" if tf == "m15" else "conf"
        st.text(f"  {tf} {fam:3s} : {'YES' if hit else 'no '}  ({kind})")
    st.text("  15m SR/LIQ = execution eligibility; higher-TF = confluence/context.")

    # C. FIX2 exact trigger proof: WHICH zone each bit came from (single source).
    #    AUDIT-FIX1: overlap interval = AUDIT DISPLAY ONLY (never redefines the
    #    Candidate); trigger OHLC + zone + overlap shown as exact numbers.
    st.markdown("**C. Trigger proof (exact hit zones at trigger bar)**")
    _tbi = int(state.get("trigger_bar_index", -1))
    if _tbi < 0:
        st.caption("Trigger bar index unknown for this candidate.")
    else:
        _tlow = _thigh = None
        try:
            _ex = load_exec_frame_r4_cached(symbol)
            _trow = _ex.iloc[_tbi]
            _tlow = float(_trow["low"])
            _thigh = float(_trow["high"])
            st.text(
                f"Trigger bar {_tbi} @ {pd.Timestamp(_ex['decision_time'].iloc[_tbi])}  "
                f"OHLC {float(_trow['open']):.2f}/{_thigh:.2f}/"
                f"{_tlow:.2f}/{float(_trow['close']):.2f}"
            )
        except Exception as _e:
            st.warning(f"R4 exec frame unavailable: {_e}")
        try:
            _proof = load_candidate_proof_r4_cached(symbol)
            _rows = _proof[_proof["trigger_bar_index"] == _tbi].to_dict("records")
        except Exception as _e:
            _rows = []
            st.warning(f"touch proof unavailable: {_e}")
        if not _rows:
            st.caption("No hit-zone proof recorded for this trigger bar.")
        else:
            for _m in _rows:
                _is_m15 = _m["tf"] == "m15"
                _tag = "ELIG" if _is_m15 else "conf"
                _bot = float(_m["bottom"])
                _top = float(_m["top"])
                _zone = f"[{_bot:.2f}, {_top:.2f}]"
                _ov = ""
                if _tlow is not None and bool(_m.get("intersects", True)):
                    _ov = (
                        f" overlap=[{max(_tlow, _bot):.2f},"
                        f" {min(_thigh, _top):.2f}]"
                    )
                if _m["family"] == "LIQ":
                    _lvl = f" level={float(_m['level']):.2f}" if pd.notna(_m.get("level")) else ""
                    _side = f" {_m.get('side')}" if _m.get("side") else ""
                    st.text(
                        f"  {_m['tf']} {_m['family']}{_side} #{int(_m['slot'])} "
                        f"{_zone}{_lvl} intersect={_m.get('intersects')}{_ov} ({_tag})"
                    )
                else:
                    _str = (
                        f" strength={float(_m['strength']):.1f}"
                        if pd.notna(_m.get("strength")) else ""
                    )
                    st.text(
                        f"  {_m['tf']} {_m['family']} #{int(_m['slot'])} "
                        f"{_zone}{_str} intersect={_m.get('intersects')}{_ov} ({_tag})"
                    )
            st.text("  Trigger range = [Low, High] of trigger bar; intersect is the")
            st.text("  stored hit result (Viewer does NOT re-judge bar_hits_zone).")
            st.text("  overlap = [max(L,bottom), min(H,top)] — AUDIT DISPLAY ONLY;")
            st.text("  it NEVER redefines the Candidate.")

    # B. canonical R4 execution-environment state (canonical owner, single pass).
    st.markdown("**B. Current canonical R4 execution-environment state**")
    st.text("→ manually judge: does the candidate make sense vs SR/Liq/DTP?")
    st.text("   m15 = committed (15m bar complete at this close); 1h/4h = forming.")
    _t0 = time.perf_counter()
    fdf = load_env_features_r4_cached(symbol)
    timing["forming_env_ms"] = (time.perf_counter() - _t0) * 1000.0
    if 0 <= selected < len(fdf):
        for tf in ("m15", "h1", "h4"):
            _r = fdf.iloc[selected]
            ts = int(_r[f"{tf}_trend_state"])
            tlabel = "UP" if ts == 1 else ("DOWN" if ts == -1 else "FLAT/NA")
            st.text(
                f"{tf}: DTP {tlabel} score={_ff(_r[f'{tf}_trend_score'], 3)} "
                f"dev={_ff(_r[f'{tf}_dev'], 3)} slope={_ff(_r[f'{tf}_slope_atr'], 3)} "
                f"sma={_ff(_r[f'{tf}_sma'])} atr={_ff(_r[f'{tf}_atr'])}"
            )
            st.text(
                f"     SR in_zone={int(_r[f'{tf}_sr_in_zone'])} ch={int(_r[f'{tf}_sr_n_channels'])} "
                f"sup={_ff(_r[f'{tf}_sr_support_price'])} "
                f"({_ff(_r[f'{tf}_sr_support_dist_atr'])}σ,{_ff(_r[f'{tf}_sr_support_strength'], 0)}) "
                f"res={_ff(_r[f'{tf}_sr_resistance_price'])} "
                f"({_ff(_r[f'{tf}_sr_resistance_dist_atr'])}σ,{_ff(_r[f'{tf}_sr_resistance_strength'], 0)})"
            )
            st.text(
                f"     LIQ up={int(_r[f'{tf}_liq_up_count'])} dn={int(_r[f'{tf}_liq_down_count'])} "
                f"upLvl={_ff(_r[f'{tf}_liq_up_level_price'])} dnLvl={_ff(_r[f'{tf}_liq_down_level_price'])} "
                f"upDist={_ff(_r[f'{tf}_liq_up_dist_atr'])} dnDist={_ff(_r[f'{tf}_liq_down_dist_atr'])}"
            )
    else:
        st.text("   (no R4 environment row for this bar)")

    # Hard validation table (Section 19)
    with st.expander("Candidate Overlay Audit", expanded=False):
        a = cand_audit
        lo, hi = compute_viewport(track, selected)
        vis_seg = [s for s in segments if s.start_idx <= hi and s.end_idx >= lo]
        vis_rows = sum(s.n_bars for s in vis_seg)
        st.text(f"Symbol                    : {a['symbol']}")
        st.text(f"T2 candidate rows         : {a['t2_candidate_rows']}")
        st.text(f"Matched 5m bars           : {a['matched_5m_bars']}")
        st.text(f"Unmatched candidate rows  : {a['unmatched_candidate_rows']}  (must be 0)")
        st.text(f"Duplicate decision keys   : {a['duplicate_decision_keys']}  (must be 0)")
        st.text(f"Episodes                  : {a['n_episodes']}")
        st.text(f"Visual segments           : {a['n_segments']}")
        if a["first_candidate_time"] is not None:
            st.text(f"First candidate time      : {a['first_candidate_time']}")
            st.text(f"Last candidate time       : {a['last_candidate_time']}")
        st.text(f"Full-symbol candidate rows: {a['t2_candidate_rows']}")
        st.text(f"Visible candidate rows     : {vis_rows}")
        # AUDIT-FIX1 point 10: viewport bar-level audit counters
        _va = timing.get("viewport_audit")
        if _va is not None:
            st.text("--- bar-level viewport audit (last three must be 0) ---")
            st.text(f"Visible candidates        : {_va['visible_candidates']}")
            st.text(f"Visible triggers          : {_va['visible_triggers']}")
            st.text(f"Candidates w/ m15 proof   : {_va['candidates_with_m15_proof']}")
            st.text(f"Missing m15 proof         : {_va['missing_m15_proof']}  (must be 0)")
            st.text(f"Bits/proof mismatch       : {_va['bits_proof_mismatch']}  (must be 0)")
        invariants_ok = (
            a["unmatched_candidate_rows"] == 0
            and a["duplicate_decision_keys"] == 0
            and a["matched_5m_bars"] == a["t2_candidate_rows"]
            and _va is not None
            and _va["missing_m15_proof"] == 0
            and _va["bits_proof_mismatch"] == 0
        )
        st.text(f"INVARIANTS MET            : {invariants_ok}")


if __name__ == "__main__":
    main()
