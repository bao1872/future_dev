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
    add_candidate_zone_overlay,
    build_candidate_segments,
    candidate_for_symbol,
    candidate_state_at,
    load_candidate_rows,
    match_available_index,
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


@st.cache_data(show_spinner="加载候选区域真值…")
def load_candidate_rows_cached():
    """Frozen T2 candidate truth (cached; never recomputed per interaction)."""
    return load_candidate_rows()


@st.cache_data(show_spinner="构建候选区域段…")
def candidate_segments_cached(symbol: str, source_sha: str):
    """Per-symbol candidate segments + frozen-truth lookup.

    Args are strings only, so cache hits are cheap (no array hashing). The 5m
    track is fetched from the cache_resource owner, so switching bars never
    re-runs the candidate groupby (FIX1 point 2).
    """
    cand_df = load_candidate_rows_cached()
    cand_sym = candidate_for_symbol(cand_df, symbol)
    track5 = build_track_cached(symbol, "5m", source_sha)
    return build_candidate_segments(cand_sym, track5)


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
    st.session_state.setdefault("iv_cand_view", "All bars")

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
        if st.session_state.iv_show_candidate:
            st.session_state.iv_cand_view = st.radio(
                "Candidate display",
                ["All bars", "Candidate episodes only"],
                index=0 if st.session_state.iv_cand_view == "All bars" else 1,
            )

    # DP Oracle audit overlay — OFF by default, read-only, 5m-clock only.
    st.session_state.iv_show_oracle = st.checkbox(
        "DP Oracle — FUTURE / HINDSIGHT AUDIT",
        value=st.session_state.iv_show_oracle,
        help="Future-derived hindsight labels. For manual audit only; NOT a causal "
             "trading signal and never a model feature.",
    )

    # (2) NOW build the track from the CURRENT widget values, so the chart
    #     always corresponds to the dropdown in the SAME rerun.
    timing: Dict[str, float] = {}
    _t0 = time.perf_counter()
    track = build_track_cached(symbol, tf, git_head())
    timing["track_build_ms"] = (time.perf_counter() - _t0) * 1000.0

    # (2b) Candidate-zone overlay data (frozen T2 truth, 5m decision axis).
    # FIX1 point 1: only touch the T2 candidate data when the overlay is ON and
    # the primary chart is 5m. Otherwise leave it entirely untouched.
    # FIX1 point 2: the per-symbol segments are served from a cached owner, so
    # switching bars never re-runs the candidate groupby.
    segments, cand_audit, cand_by_time = [], EMPTY_CAND_AUDIT, {}
    if st.session_state.iv_show_candidate and track.tf_label == "5m":
        _t0 = time.perf_counter()
        _ = load_candidate_rows_cached()  # parquet read / cache hit
        timing["candidate_load_ms"] = (time.perf_counter() - _t0) * 1000.0
        _t0 = time.perf_counter()
        segments, cand_audit, cand_by_time = candidate_segments_cached(symbol, git_head())
        timing["candidate_segment_ms"] = (time.perf_counter() - _t0) * 1000.0

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

    # ---- DP Oracle audit overlay (read-only, fail-closed, 5m only) ------- #
    show_oracle = bool(st.session_state.iv_show_oracle)
    oracle_trades = None
    oracle_meta = None
    if show_oracle:
        st.warning(
            "**DP Oracle uses future prices to find hindsight-optimal intraday "
            "trades. It is for audit / research labels only — NOT a causal "
            "trading signal and never a model feature.**"
        )
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
            if tf != ORACLE_TF_ONLY:
                st.info(
                    "Oracle executions are defined on the 5m clock. Switch to 5m "
                    "to inspect exact Entry / Exit points."
                )
                oracle_trades = None
            else:
                _rec, _mism = select_visible_oracle_trades(track, selected, oracle_trades)
                if _mism > 0:
                    st.error(
                        f"Oracle overlay disabled (fail-closed): time alignment "
                        f"mismatch on {_mism} visible trade(s)."
                    )
                    oracle_trades = None
                else:
                    _summ = oracle_viewport_summary(track, selected, oracle_trades)
                    st.caption(
                        f"Oracle audit · visible trades={_summ['visible_trades']} "
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

    # ---- main + snapshot ------------------------------------------------ #
    chart_col, snap_col = st.columns([4, 1])
    with chart_col:
        _t0 = time.perf_counter()
        fig = build_figure(track, selected, st.session_state.iv_show_dtp,
                           st.session_state.iv_show_sr, st.session_state.iv_show_liq)
        if st.session_state.iv_show_candidate:
            if track.tf_label == "5m":
                # FIX1 point 3: only draw segments that intersect the CURRENT
                # viewport AND start at/before the selected bar. This keeps the
                # figure to a handful of vrects instead of every historical
                # episode (AG has ~1392 episodes; never all go into the figure).
                lo, hi = compute_viewport(track, selected)
                visible = [
                    s for s in segments
                    if s.end_idx >= lo and s.start_idx <= hi and s.start_idx <= selected
                ]
                fig = add_candidate_zone_overlay(fig, visible)
                timing["plot_segment_count"] = len(visible)
                timing["full_symbol_segments"] = len(segments)
                st.caption(
                    f"Blue shaded bands = frozen Oracle candidate trading zones "
                    f"(formal T2). Rendered {len(visible)} / {len(segments)} segments "
                    f"in current viewport. Decision axis = 5m; no t+1 shift. "
                    f"Candidate truth only — no model / Y / Q / Oracle action shown.")
                if st.session_state.iv_cand_view == "Candidate episodes only" and segments:
                    starts = sorted({int(s.start_idx) for s in segments})
                    cp, cn = st.columns(2)
                    with cp:
                        if st.button("◀ Prev candidate episode"):
                            prev = [x for x in starts if x < selected]
                            if prev:
                                st.session_state.iv_selected = int(max(prev))
                                st.rerun()
                    with cn:
                        if st.button("Next candidate episode ▶"):
                            nxt = [x for x in starts if x > selected]
                            if nxt:
                                st.session_state.iv_selected = int(min(nxt))
                                st.rerun()
            else:
                st.info(
                    "Candidate zones are defined on the 5m decision axis. "
                    "Switch the primary chart to 5m for candidate-region audit.")
        if show_oracle and oracle_trades is not None and tf == ORACLE_TF_ONLY:
            fig = add_dp_oracle_overlay(fig, track, selected, oracle_trades)
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
        if st.session_state.iv_show_candidate and track.tf_label == "5m":
            _render_candidate_audit(track, symbol, selected, cand_by_time, cand_audit, segments, timing)

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
        st.text(f"  plot_segment_count    : {timing.get('plot_segment_count', 0)} "
                f"/ {timing.get('full_symbol_segments', 0)}  (rendered / full)")
        st.json({
            "dtp": d,
            "sr": snap["sr_channels"],
            "liq_up": snap["liq_up"],
            "liq_down": snap["liq_down"],
        })


def _ff(x, p: int = 2) -> str:
    """Format a forming-feature scalar; show '—' for NaN / missing."""
    try:
        v = float(x)
        if not np.isfinite(v):
            return "—"
        return f"{v:.{p}f}"
    except (TypeError, ValueError):
        return "—"


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
    state = candidate_state_at(cand_by_time, dt)
    if not state["is_candidate"]:
        st.caption(
            "Candidate: NO at this bar (frozen T2 truth). "
            "Shaded bands mark Oracle candidates; this bar is not one."
        )
        return

    # A. frozen candidate truth
    st.markdown("**A. Frozen candidate truth (T2)**")
    st.text(f"Decision time : {dt}")
    st.text(f"Candidate     : YES")
    st.text(f"Episode       : {state['episode']}")
    st.text(f"Proximity any : {state['proximity_any']}")
    st.text(f"Prox. episode : {state['proximity_episode_id']}")

    # B. canonical FORMING multi-timeframe state (canonical owner)
    st.markdown("**B. Current canonical FORMING indicator state**")
    st.text("→ manually judge: does Oracle's candidate make sense vs SR/Liq/DTP?")
    st.text("   5m = ViewerTrack snapshot (forming 5m bar); 15m/1h/4h = forming")
    st.text("   bar at this 5m close via canonical forming-MTF owner (NOT the")
    st.text("   last fully-closed HTF bar).")
    _t0 = time.perf_counter()
    fdf = load_forming_env_cached(symbol)
    timing["forming_env_ms"] = (time.perf_counter() - _t0) * 1000.0

    # 5m: the forming 5m state IS the ViewerTrack selected snapshot (parity
    # tested). Sourcing it here avoids the ~35s of trivial m5 previews that the
    # forming owner would otherwise do.
    snap5 = selected_snapshot(track, selected)
    d5 = snap5["dtp"]
    t5 = d5["trend"]
    st.text(
        f"5m: DTP {'UP' if t5 == 1 else ('DOWN' if t5 == -1 else 'FLAT/NA')} "
        f"score={_ff(d5['trend_score'], 3)} dev={_ff(snap5['dev'], 3)} "
        f"slope={_ff(snap5['slope'], 3)} sma={_ff(snap5['ma'])} atr={_ff(snap5['atr'])}"
    )
    ch5 = snap5["sr_channels"]
    sup5 = min((c["bottom"] for c in ch5), default=float("nan"))
    res5 = max((c["top"] for c in ch5), default=float("nan"))
    st.text(
        f"     SR in_zone={int(snap5['sr_in_zone'])} ch={int(snap5['sr_n_channels'])} "
        f"sup={_ff(sup5)} res={_ff(res5)}"
    )
    st.text(
        f"     LIQ up={int(snap5['liq_up_count'])} dn={int(snap5['liq_down_count'])}"
    )

    # 15m/1h/4h: forming bar at this 5m close (canonical owner).
    sub = fdf[fdf["decision_time"] == dt]
    row = sub.iloc[0] if len(sub) else (fdf.iloc[selected] if 0 <= selected < len(fdf) else None)
    if row is None:
        st.text("   (no forming-state row for this decision time)")
    else:
        for tf in ("15m", "1H", "4H"):
            p = TF_PREFIX[tf]
            ts = int(row[f"{p}_trend_state"])
            tlabel = "UP" if ts == 1 else ("DOWN" if ts == -1 else "FLAT/NA")
            st.text(
                f"{tf}: DTP {tlabel} score={_ff(row[f'{p}_trend_score'], 3)} "
                f"dev={_ff(row[f'{p}_dev'], 3)} slope={_ff(row[f'{p}_slope_atr'], 3)} "
                f"sma={_ff(row[f'{p}_sma'])} atr={_ff(row[f'{p}_atr'])}"
            )
            st.text(
                f"     SR in_zone={int(row[f'{p}_sr_in_zone'])} ch={int(row[f'{p}_sr_n_channels'])} "
                f"sup={_ff(row[f'{p}_sr_support_price'])} "
                f"({_ff(row[f'{p}_sr_support_dist_atr'])}σ,{_ff(row[f'{p}_sr_support_strength'], 0)}) "
                f"res={_ff(row[f'{p}_sr_resistance_price'])} "
                f"({_ff(row[f'{p}_sr_resistance_dist_atr'])}σ,{_ff(row[f'{p}_sr_resistance_strength'], 0)})"
            )
            st.text(
                f"     LIQ up={int(row[f'{p}_liq_up_count'])} dn={int(row[f'{p}_liq_down_count'])} "
                f"upLvl={_ff(row[f'{p}_liq_up_level_price'])} dnLvl={_ff(row[f'{p}_liq_down_level_price'])} "
                f"upDist={_ff(row[f'{p}_liq_up_dist_atr'])} dnDist={_ff(row[f'{p}_liq_down_dist_atr'])}"
            )

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
        invariants_ok = (
            a["unmatched_candidate_rows"] == 0
            and a["duplicate_decision_keys"] == 0
            and a["matched_5m_bars"] == a["t2_candidate_rows"]
        )
        st.text(f"INVARIANTS MET            : {invariants_ok}")


if __name__ == "__main__":
    main()
