"""
pages/6_Indicator_Viewer.py
===========================

Independent read-only Indicator Viewer (Task FUTURE-INDICATOR-VIEWER-V1-KERNEL-UI-P1).

Shows the three raw indicators (DTP / SR / Liquidity) as-of a selected TF bar's
close:  IndicatorState_t  ⊆  Information_<= close(t).

No Oracle / Label / PGM / model / future-path output is ever rendered.

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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from research.liquidity_oracle_atlas.audit_view_v1 import SYMBOLS
from research.liquidity_oracle_atlas.build_forming_environment_v1 import (
    FormingEnvironmentBuilder,
)
from research.liquidity_oracle_atlas.git_head import git_head
from research.liquidity_oracle_atlas.indicator_viewer_v1 import (
    BINS,
    PROFILE_OFFSET,
    SR_MAX,
    build_viewer_track,
    compute_viewport,
    dtp_profile,
    selected_snapshot,
)

TF_LABELS = ["5m", "15m", "1H", "4H"]

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
        up_mask = trend == 1
        dn_mask = trend == -1
        if up_mask.any():
            fig.add_trace(go.Scatter(
                x=idx_v[up_mask], y=sma_v[up_mask], mode="lines",
                line={"color": C_DTP_UP, "width": 1.5}, name="SMA UP", showlegend=False))
        if dn_mask.any():
            fig.add_trace(go.Scatter(
                x=idx_v[dn_mask], y=sma_v[dn_mask], mode="lines",
                line={"color": C_DTP_DOWN, "width": 1.5}, name="SMA DOWN", showlegend=False))

        # ±1/±2/±3 ATR short horizontal levels at the selected bar (right side)
        s = float(track.sma[selected]); a = float(track.atr[selected])
        if np.isfinite(s) and np.isfinite(a) and a > 0:
            xb = [selected, selected + 5]
            for k in (1, 2, 3):
                fig.add_trace(go.Scatter(
                    x=xb, y=[s + k * a, s + k * a], mode="lines",
                    line={"color": _rgba(C_DTP_UP, 0.5), "width": 1, "dash": "dot"},
                    showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(
                    x=xb, y=[s - k * a, s - k * a], mode="lines",
                    line={"color": _rgba(C_DTP_DOWN, 0.5), "width": 1, "dash": "dot"},
                    showlegend=False, hoverinfo="skip"))

        # trend switch markers
        diff = np.where(trend[1:] != trend[:-1])[0] + 1
        for di in diff:
            x = int(idx_v[di]); d = int(trend[di])
            fig.add_trace(go.Scatter(
                x=[x], y=[float(track.sma[x])], mode="markers",
                marker={"color": C_DTP_UP if d == 1 else C_DTP_DOWN, "size": 9, "symbol": "circle"},
                showlegend=False, hoverinfo="skip"))

        # DTP Trend Distribution Profile (literal counting semantics)
        counts, lookback = dtp_profile(track, selected)
        if counts is not None and lookback is not None:
            pmin = s - 3.0 * a
            pstep = 6.0 * a / BINS
            x0 = selected + PROFILE_OFFSET
            trend_up = int(track.trend_state[selected]) == 1
            base = C_DTP_UP if trend_up else C_DTP_DOWN
            for b in range(BINS):
                cnt = int(counts[b])
                if cnt <= 0:
                    continue
                lower = pmin + pstep * b
                op = 0.22 + 0.6 * (b / (BINS - 1)) if BINS > 1 else 0.6
                fig.add_shape(type="rect", xref="x", yref="y",
                              x0=x0, x1=x0 + min(cnt, 40), y0=lower, y1=lower + pstep,
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


def _draw_liq(fig, track, selected, lo, side):
    if side > 0:
        valid = track.liq_up_valid; left = track.liq_up_left
        level = track.liq_up_level; top = track.liq_up_top
        bottom = track.liq_up_bottom; broken = track.liq_up_broken
        ze = track.liq_up_zone_exists; za = track.liq_up_zone_active
        zl = track.liq_up_zone_left; zr = track.liq_up_zone_right
        ztop = track.liq_up_zone_top; zbot = track.liq_up_zone_bottom
        color = C_BUY; label = "Buyside"
    else:
        valid = track.liq_down_valid; left = track.liq_down_left
        level = track.liq_down_level; top = track.liq_down_top
        bottom = track.liq_down_bottom; broken = track.liq_down_broken
        ze = track.liq_down_zone_exists; za = track.liq_down_zone_active
        zl = track.liq_down_zone_left; zr = track.liq_down_zone_right
        ztop = track.liq_down_zone_top; zbot = track.liq_down_zone_bottom
        color = C_SELL; label = "Sellside"

    for j in range(3):
        if not valid[selected, j]:
            continue
        lvl = float(level[selected, j])
        tp = float(top[selected, j]); bt = float(bottom[selected, j])
        lft = int(left[selected, j])
        bool(broken[selected, j])

        # margin region (faint) across the visible window
        fig.add_shape(type="rect", xref="x", yref="y", x0=lo, x1=selected,
                      y0=bt, y1=tp, fillcolor=_rgba(color, 0.10),
                      line={"width": 0}, layer="below")
        # dotted extension before discovery (lo -> left)
        if lft > lo:
            fig.add_shape(type="line", xref="x", yref="y", x0=lo, x1=lft,
                          y0=lvl, y1=lvl, line={"color": _rgba(color, 0.5), "width": 1, "dash": "dot"},
                          layer="above")
        # solid level line from discovery to selected
        fig.add_shape(type="line", xref="x", yref="y", x0=lft, x1=selected,
                      y0=lvl, y1=lvl, line={"color": color, "width": 2}, layer="above")
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
def _load_base(symbol: str):
    @st.cache_data(show_spinner="加载 5m 基础数据…")
    def _load(sym: str):
        b = FormingEnvironmentBuilder(symbol=sym)
        b.load_raw()
        return b.base
    return _load(symbol)


def _build_track(symbol: str, tf: str):
    @st.cache_data(show_spinner="构建指标时间轴…")
    def _build(sym: str, timeframe: str):
        base = _load_base(sym)
        return build_viewer_track(base, timeframe, symbol=sym, source_sha=git_head())
    return _build(symbol, tf)


def main() -> None:
    st.set_page_config(page_title="指标观察器", layout="wide")
    st.markdown("### 指标观察器  ·  Historical-as-of")
    st.caption("将选中 K 线当作该时刻最后一根已形成的 K 线；只显示该 TF 三个原始指标当时的状态。"
               "不含 Oracle / Label / PGM / 模型结论。")

    if "iv_symbol" not in st.session_state:
        st.session_state.iv_symbol = SYMBOLS[0]
    if "iv_tf" not in st.session_state:
        st.session_state.iv_tf = "1H"
    if "iv_show_dtp" not in st.session_state:
        st.session_state.iv_show_dtp = True
    if "iv_show_sr" not in st.session_state:
        st.session_state.iv_show_sr = True
    if "iv_show_liq" not in st.session_state:
        st.session_state.iv_show_liq = True

    symbol = st.session_state.iv_symbol
    tf = st.session_state.iv_tf
    track = _build_track(symbol, tf)

    if "iv_selected" not in st.session_state:
        st.session_state.iv_selected = track.n - 1
    st.session_state.iv_selected = int(min(max(st.session_state.iv_selected, 0), track.n - 1))

    # ---- top toolbar ---------------------------------------------------- #
    col_sym, col_tf, col_date, col_bt, col_prev, col_next, c1, c2, c3 = st.columns(
        [1.1, 1.2, 1.6, 1.1, 0.6, 0.6, 0.6, 0.6, 0.6])
    with col_sym:
        st.session_state.iv_symbol = st.selectbox("Symbol", SYMBOLS, index=SYMBOLS.index(symbol))
    with col_tf:
        st.session_state.iv_tf = st.selectbox("Timeframe", TF_LABELS, index=TF_LABELS.index(tf))
    with c1:
        st.session_state.iv_show_dtp = st.checkbox("DTP", value=st.session_state.iv_show_dtp)
    with c2:
        st.session_state.iv_show_sr = st.checkbox("SR", value=st.session_state.iv_show_sr)
    with c3:
        st.session_state.iv_show_liq = st.checkbox("Liquidity", value=st.session_state.iv_show_liq)

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

    # ---- main + snapshot ------------------------------------------------ #
    chart_col, snap_col = st.columns([4, 1])
    with chart_col:
        fig = build_figure(track, selected, st.session_state.iv_show_dtp,
                           st.session_state.iv_show_sr, st.session_state.iv_show_liq)
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

    # ---- bottom debug --------------------------------------------------- #
    with st.expander("Technical snapshot", expanded=False):
        st.text(f"global TF index : {snap['index']}")
        st.text(f"segment id      : {snap['segment']}")
        st.text(f"source SHA      : {track.source_sha}")
        st.text(f"counters        : raw_load={track.raw_load_count} resample={track.resample_count} "
                f"steps={track.indicator_step_count} full_recompute={track.full_history_recompute_count} "
                f"reference={track.reference_call_count} writes={track.visual_snapshot_write_count}")
        st.json({
            "dtp": d,
            "sr": snap["sr_channels"],
            "liq_up": snap["liq_up"],
            "liq_down": snap["liq_down"],
        })


if __name__ == "__main__":
    main()
