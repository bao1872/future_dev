"""
pages/6_Indicator_Viewer.py
============================

Indicator Viewer (Task FUTURE-INDICATOR-VIEWER-V1-KERNEL-UI-P1).

A read-only, TradingView-like viewer for the three frozen raw indicators
(DTP / SR / Liquidity). Selecting a TF bar shows the indicator state
*as-of that bar's close* — never the forming 5m environment, never any
Oracle / label / PGM / model output.

Frozen owners (no math reimplemented here):
  * research.liquidity_oracle_atlas.build_forming_environment_v1
        .FormingEnvironmentBuilder.load_raw   -> canonical 5m base
  * research.liquidity_oracle_atlas.indicator_viewer_v1
        .build_viewer_track / dtp_profile / selected_snapshot
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from research.liquidity_oracle_atlas.experiment_structural_reversion_pgm_v1 import (
    git_head,
)
from research.liquidity_oracle_atlas.build_forming_environment_v1 import (
    FormingEnvironmentBuilder,
)
from research.liquidity_oracle_atlas.audit_view_v1 import SYMBOLS
from research.liquidity_oracle_atlas.indicator_viewer_v1 import (
    BINS,
    C_BEAR,
    C_BG,
    C_BULL,
    C_BUY,
    C_DTP_DOWN,
    C_DTP_UP,
    C_GRID,
    C_SELL,
    C_SR_IN,
    C_SR_RES,
    C_SR_SUP,
    C_TEXT,
    PROFILE_OFFSET,
    build_viewer_track,
    dtp_profile,
    selected_snapshot,
)

TF_OPTIONS = ["5m", "15m", "1H", "4H"]

SOURCE_SHA = git_head()


# --------------------------------------------------------------------------- #
# Caching (frozen pipeline)                                                     #
# --------------------------------------------------------------------------- #
@st.cache_data
def _load_base(symbol: str) -> pd.DataFrame:
    b = FormingEnvironmentBuilder(symbol)
    b.load_raw()
    return b.base


@st.cache_data
def _build_track(symbol: str, tf_label: str, source_sha: str) -> Any:
    base = _load_base(symbol)
    return build_viewer_track(
        base, tf_label, symbol=symbol, source_sha=source_sha, raw_load_count=1
    )


# --------------------------------------------------------------------------- #
# Chart                                                                         #
# --------------------------------------------------------------------------- #
def _fmt_time(track, i: int) -> str:
    return pd.Timestamp(track.time[i]).strftime("%Y-%m-%d %H:%M")


def build_figure(track, selected: int, show_dtp: bool, show_sr: bool, show_liq: bool) -> go.Figure:
    n = track.n
    idx = np.arange(n)
    fig = go.Figure()

    # --- candlestick ---
    fig.add_trace(go.Candlestick(
        x=idx, open=track.open, high=track.high, low=track.low, close=track.close,
        name="OHLC", increasing_line_color=C_BULL, decreasing_line_color=C_BEAR,
        increasing_fillcolor=C_BULL, decreasing_fillcolor=C_BEAR,
    ))

    if show_dtp:
        sma = track.sma
        atr = track.atr
        trend = track.trend_state
        # SMA colored by trend
        sma_up = np.where(trend == 1, sma, np.nan)
        sma_dn = np.where(trend == -1, sma, np.nan)
        fig.add_trace(go.Scatter(x=idx, y=sma_up, mode="lines", name="SMA50 UP",
                                 line=dict(color=C_DTP_UP, width=2)))
        fig.add_trace(go.Scatter(x=idx, y=sma_dn, mode="lines", name="SMA50 DN",
                                 line=dict(color=C_DTP_DOWN, width=2)))
        # ATR bands
        for k in (1, 2, 3):
            for sign, col in ((1, "rgba(18,209,235,0.25)"), (-1, "rgba(250,40,86,0.25)")):
                fig.add_trace(go.Scatter(
                    x=idx, y=sma + sign * k * atr, mode="lines",
                    name=f"SMA{k*sign:+d}ATR", line=dict(color=col, width=1, dash="dot"),
                    showlegend=False, hoverinfo="skip",
                ))
        # trend switch markers
        switches = []
        for i in range(1, n):
            if track.trend_start_global[i] >= 0 and trend[i] != trend[i - 1]:
                switches.append(i)
        if switches:
            sx = np.array(switches)
            sy = sma[sx]
            scol = [C_DTP_UP if trend[i] == 1 else C_DTP_DOWN for i in switches]
            fig.add_trace(go.Scatter(
                x=sx, y=sy, mode="markers", name="trend switch",
                marker=dict(color=scol, size=9, symbol="circle"),
                showlegend=False, hoverinfo="skip",
            ))
        # DTP profile (rendered to the right of selected bar)
        counts, lookback = dtp_profile(track, selected)
        if counts is not None:
            min_t = sma[selected] - 3.0 * atr[selected]
            step_t = (6.0 * atr[selected]) / BINS
            start = selected + PROFILE_OFFSET
            max_count = int(counts.max()) if counts.max() > 0 else 1
            for b in range(BINS):
                cnt = int(counts[b])
                if cnt == 0:
                    continue
                lower = min_t + step_t * b
                upper = lower + step_t
                frac = cnt / max_count
                col = f"rgba(18,209,235,{0.15 + 0.7*frac:.2f})" if trend[selected] == 1 else \
                      f"rgba(250,40,86,{0.15 + 0.7*frac:.2f})"
                fig.add_shape(type="rect", x0=start - cnt, x1=start, y0=lower, y1=upper,
                              fillcolor=col, line=dict(width=0), layer="above")
            fig.add_annotation(x=start, y=min_t + step_t * BINS,
                              text=f"trend age {lookback}", showarrow=False,
                              font=dict(color=C_TEXT, size=10))

    if show_sr:
        snap = selected_snapshot(track, selected)
        c = snap["c"]
        for ch in snap["sr_channels"]:
            top, bot = ch["top"], ch["bottom"]
            if top > c and bot > c:
                fill = C_SR_RES
            elif top < c and bot < c:
                fill = C_SR_SUP
            else:
                fill = C_SR_IN
            fig.add_shape(type="rect", x0=0, x1=n - 1, y0=bot, y1=top,
                          fillcolor=fill, line=dict(color="rgba(209,212,220,0.5)", width=1),
                          layer="below")

    if show_liq:
        snap = selected_snapshot(track, selected)
        for lev in snap["liq_up"]:
            fig.add_shape(type="rect", x0=0, x1=n - 1, y0=lev["bottom"], y1=lev["top"],
                          fillcolor="rgba(76,175,80,0.12)",
                          line=dict(color=C_BUY, width=1), layer="below")
            if lev["zone_active"] and lev["zone_left"] is not None and lev["zone_right"] is not None:
                fig.add_shape(type="rect", x0=lev["zone_left"], x1=lev["zone_right"],
                              y0=lev["zone_bottom"], y1=lev["zone_top"],
                              fillcolor="rgba(76,175,80,0.30)", line=dict(width=0), layer="above")
        for lev in snap["liq_down"]:
            fig.add_shape(type="rect", x0=0, x1=n - 1, y0=lev["bottom"], y1=lev["top"],
                          fillcolor="rgba(242,54,69,0.12)",
                          line=dict(color=C_SELL, width=1), layer="below")
            if lev["zone_active"] and lev["zone_left"] is not None and lev["zone_right"] is not None:
                fig.add_shape(type="rect", x0=lev["zone_left"], x1=lev["zone_right"],
                              y0=lev["zone_bottom"], y1=lev["zone_top"],
                              fillcolor="rgba(242,54,69,0.30)", line=dict(width=0), layer="above")

    # selected bar highlight
    fig.add_vline(x=selected, line=dict(color="rgba(255,255,255,0.6)", width=1, dash="dash"))

    # hit-layer for direct click selection (transparent)
    fig.add_trace(go.Scatter(
        x=idx, y=track.close, mode="markers",
        marker=dict(size=1, opacity=0.0, color="rgba(0,0,0,0)"),
        customdata=np.stack([idx, [str(pd.Timestamp(track.time[i])) for i in idx]], axis=-1),
        name="hit", hoverinfo="skip", showlegend=False,
    ))

    # x tick labels = real timestamps (stride)
    stride = max(1, n // 12)
    tickvals = list(range(0, n, stride))
    ticktext = [_fmt_time(track, t) for t in tickvals]
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=C_BG, plot_bgcolor=C_BG,
        font=dict(color=C_TEXT), xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=20, b=30),
    )
    fig.update_xaxes(gridcolor=C_GRID, tickmode="array", tickvals=tickvals, ticktext=ticktext,
                     showspikes=True, spikemode="across", spikesnap="cursor")
    fig.update_yaxes(gridcolor=C_GRID, side="right")
    return fig


# --------------------------------------------------------------------------- #
# Page                                                                          #
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(layout="wide")
    st.title("Indicator Viewer · DTP / SR / Liquidity")
    st.caption("Historical-as-of mode — snapshot as of the selected bar's close. "
               "No Oracle / label / PGM / model output.")

    # session state
    for key, default in (("iv_symbol", SYMBOLS[0]), ("iv_tf", "1H"), ("iv_selected", None)):
        if key not in st.session_state:
            st.session_state[key] = default

    symbol = st.sidebar.selectbox("Symbol", SYMBOLS, index=SYMBOLS.index(st.session_state.iv_symbol))
    st.session_state.iv_symbol = symbol
    tf_label = st.sidebar.selectbox("Timeframe", TF_OPTIONS, index=TF_OPTIONS.index(st.session_state.iv_tf))
    st.session_state.iv_tf = tf_label

    with st.spinner("Building viewer track (canonical indicators)..."):
        track = _build_track(symbol, tf_label, SOURCE_SHA)

    n = track.n
    if st.session_state.iv_selected is None or st.session_state.iv_selected >= n:
        st.session_state.iv_selected = n - 1
    selected = int(st.session_state.iv_selected)

    # deterministic selector (Method B)
    dates = sorted({pd.Timestamp(track.time[i]).date() for i in range(n)})
    cur_date = pd.Timestamp(track.time[selected]).date()
    date_idx = dates.index(cur_date) if cur_date in dates else len(dates) - 1
    sel_date = st.sidebar.selectbox("Date", dates, index=date_idx)
    day_idx = [i for i in range(n) if pd.Timestamp(track.time[i]).date() == sel_date]
    day_times = [_fmt_time(track, i) for i in day_idx]
    cur_pos = day_idx.index(selected) if selected in day_idx else len(day_idx) - 1
    bar_time = st.sidebar.selectbox("Bar time", day_times, index=cur_pos)
    sel_from_date = day_idx[day_times.index(bar_time)]

    c_prev, c_next, c_sync = st.sidebar.columns(3)
    if c_prev.button("Prev"):
        selected = max(0, selected - 1)
    if c_next.button("Next"):
        selected = min(n - 1, selected + 1)
    if c_sync.button("Go to date"):
        selected = sel_from_date
    st.session_state.iv_selected = selected

    show_dtp = st.sidebar.checkbox("DTP", value=True)
    show_sr = st.sidebar.checkbox("SR", value=True)
    show_liq = st.sidebar.checkbox("Liquidity", value=True)

    # layout: chart (4) + snapshot (1)
    col_chart, col_snap = st.columns([4, 1])
    with col_chart:
        fig = build_figure(track, selected, show_dtp, show_sr, show_liq)
        event = st.plotly_chart(
            fig, key="iv_chart", on_select="rerun", selection_mode="points",
            use_container_width=True,
        )
        sel = None
        if isinstance(event, dict):
            sel = event.get("selection")
        if sel is None:
            sel = st.session_state.get("iv_chart_selection", {})
        if sel and sel.get("points"):
            cd = sel["points"][0].get("customdata")
            if cd is not None:
                st.session_state.iv_selected = int(cd[0])

    with col_snap:
        snap = selected_snapshot(track, selected)
        st.markdown(f"**{snap['symbol']} · {snap['tf']}**")
        st.markdown(f"Bar start: `{snap['bar_start_time']}`")
        st.markdown(f"Available: `{snap['available_time']}`")
        st.markdown(f"Segment: `{snap['segment']}`")
        st.markdown(f"OHLC: `{snap['o']:.4f} / {snap['h']:.4f} / {snap['l']:.4f} / {snap['c']:.4f}`")

        st.markdown("**DTP**")
        d = snap["dtp"]
        trend_lbl = {1: "UP", -1: "DOWN"}.get(d["trend"], "NONE")
        st.markdown(f"Trend: `{trend_lbl}`")
        st.markdown(f"SMA50: `{d['sma']:.4f}`")
        st.markdown(f"ATR200: `{d['atr']:.4f}`")
        st.markdown(f"Trend score: `{d['trend_score']:.4f}`")
        age = d["trend_age"]
        st.markdown(f"Trend age: `{'N/A' if age is None else age}`")
        st.markdown(f"Profile: `{'available' if d['profile_available'] else 'unavailable'}`")

        st.markdown("**SR**")
        st.markdown(f"Channels: `{snap['sr_n_channels']}`  In zone: `{snap['sr_in_zone']}`")
        for j, ch in enumerate(snap["sr_channels"]):
            st.markdown(f"#{j+1} top `{ch['top']:.4f}` bot `{ch['bottom']:.4f}` str `{ch['strength']:.0f}`")

        st.markdown("**Liquidity**")
        st.markdown(f"Buyside active: `{snap['liq_up_count']}`  Sellside active: `{snap['liq_down_count']}`")
        for side, label in (("liq_up", "BUY"), ("liq_down", "SELL")):
            for lev in snap[side]:
                za = "zone" if lev["zone_active"] else ("broken" if lev["broken"] else "active")
                st.markdown(f"{label} `{lev['level']:.4f}` [{za}] "
                            f"z=({lev['zone_left']},{lev['zone_right']})")

    with st.expander("Technical snapshot", expanded=False):
        st.markdown(f"global TF bar index: `{selected}`")
        st.markdown(f"segment-local index: `{int(track.segment[selected])}`")
        st.markdown(f"source SHA: `{track.source_sha}`")
        st.markdown("counters: "
                    f"raw_load={track.raw_load_count} resample={track.resample_count} "
                    f"steps={track.indicator_step_count} full_recompute={track.full_history_recompute_count} "
                    f"reference_calls={track.reference_call_count} writes={track.visual_snapshot_write_count}")


if __name__ == "__main__":
    main()
