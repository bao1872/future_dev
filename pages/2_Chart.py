from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from market_data.offline_store import available_timeframes, load_bars, to_indicator_frame
from research.indicator_adapter import compute_smc_momentum_bundle

# Pine visual semantics
BULL = "#089981"
BEAR = "#F23645"

OB_BULL_FILL = "rgba(49,121,245,0.20)"
OB_BEAR_FILL = "rgba(247,124,128,0.20)"

MOM_COLORS = {
    "lime": "#00FF00",
    "green": "#008000",
    "red": "#FF0000",
    "maroon": "#800000",
}

SQZ_COLORS = {
    "blue": "#2157F3",
    "black": "#000000",
    "gray": "#878B94",
}

st.set_page_config(page_title="Chart · future_dev", layout="wide")
st.title("Chart")
st.caption("Canonical SMC + SQZMOM on full offline history; display cropping happens afterward.")

tfs = available_timeframes()
if not tfs:
    st.warning("No offline data found.")
    st.stop()

c1, c2 = st.columns([1, 2])
with c1:
    tf = st.selectbox("Timeframe", tfs, index=tfs.index("1h") if "1h" in tfs else 0)
with c2:
    plot_bars = st.slider("Display bars", 100, 1000, 300, 50)

raw = load_bars(tf)
full = to_indicator_frame(raw)

with st.spinner("Computing canonical SMC + SQZMOM on full history..."):
    bundle = compute_smc_momentum_bundle(full)

smc = bundle.smc
momentum = bundle.momentum

# ---------------------------------------------------------------------------
# Canonical output consumed as-is, never recomputed
# ---------------------------------------------------------------------------

active_internal_obs = [
    ob
    for ob in smc.get("order_blocks", [])
    if ob.get("internal") is True and ob.get("mitigated") is False
][:5]

val = np.array(
    [np.nan if v is None else float(v) for v in momentum.get("val", [])],
    dtype=float,
)
bcolor = np.array(momentum.get("bcolor", []), dtype=object)
scolor = np.array(momentum.get("scolor", []), dtype=object)

if len(val) != len(full):
    st.error(f"Momentum length mismatch: {len(val)} != {len(full)}")
    st.stop()

# ---------------------------------------------------------------------------
# Cropping happens only here, after the canonical calculation
# ---------------------------------------------------------------------------

view = full.tail(plot_bars).copy()
start_pos = len(full) - len(view)
view_start = view.index[0]
view_end = view.index[-1]

latest_val = float(val[-1]) if len(val) and np.isfinite(val[-1]) else None
latest_bcolor = str(bcolor[-1]) if len(bcolor) else None

m1, m2, m3, m4 = st.columns(4)
m1.metric("Swing bias", str(smc.get("swing_bias")))
m2.metric("Internal bias", str(smc.get("internal_bias")))
m3.metric("Active internal OB", str(len(active_internal_obs)))
m4.metric(
    "Momentum",
    f"{latest_val:.2f} / {latest_bcolor}" if latest_val is not None else "N/A",
)

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.76, 0.24],
    vertical_spacing=0.025,
)

fig.add_trace(
    go.Candlestick(
        x=view.index,
        open=view["open"],
        high=view["high"],
        low=view["low"],
        close=view["close"],
        name="Price",
        increasing_line_color=BULL,
        decreasing_line_color=BEAR,
    ),
    row=1,
    col=1,
)

# --- BOS / CHoCH: line pivot -> breakout, label at the midpoint (Pine semantics)
for ev in smc.get("events", []):
    ai = ev.get("anchor_index")
    ci = ev.get("confirmed_index")

    if ai is None or ci is None:
        continue
    if ci < start_pos or ci >= len(full):
        continue

    ai = int(ai)
    ci = int(ci)

    x0 = full.index[max(ai, start_pos)]
    x1 = full.index[ci]
    level = float(ev["level"])

    bullish = bool(ev.get("bullish"))
    internal = bool(ev.get("internal"))

    color = BULL if bullish else BEAR
    dash = "dash" if internal else "solid"
    width = 1 if internal else 2

    fig.add_trace(
        go.Scatter(
            x=[x0, x1],
            y=[level, level],
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_annotation(
        x=x0 + (x1 - x0) / 2,
        y=level,
        text=str(ev["type"]),  # BOS / CHoCH
        showarrow=False,
        font=dict(color=color, size=9 if internal else 11),
        yshift=10 if bullish else -10,
        row=1,
        col=1,
    )

# --- EQH / EQL
for eq in smc.get("equal_highs_lows", []):
    ai = eq.get("anchor_index")
    si = eq.get("second_pivot_index")

    if ai is None or si is None:
        continue
    if si < start_pos or si >= len(full):
        continue

    ai = int(ai)
    si = int(si)

    x0 = full.index[max(ai, start_pos)]
    x1 = full.index[si]

    y0 = float(eq["prev_level"] if eq.get("prev_level") is not None else eq["level"])
    y1 = float(eq["level"])

    is_high = eq["type"] == "EQH"
    color = BEAR if is_high else BULL

    fig.add_trace(
        go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(color=color, width=1, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_annotation(
        x=x0 + (x1 - x0) / 2,
        y=(y0 + y1) / 2,
        text=eq["type"],
        showarrow=False,
        font=dict(color=color, size=9),
        yshift=8 if is_high else -8,
        row=1,
        col=1,
    )

# --- Active Internal Order Blocks (Pine live array: newest first, max 5)
for ob in active_internal_obs:
    ai = ob.get("anchor_index")
    if ai is None:
        continue

    ai = int(ai)
    if ai >= len(full):
        continue

    anchor_time = full.index[ai]
    if anchor_time > view_end:
        continue

    x0 = max(anchor_time, view_start)
    x1 = view_end

    bullish = int(ob["bias"]) == 1

    fig.add_shape(
        type="rect",
        x0=x0,
        x1=x1,
        y0=float(ob["bar_low"]),
        y1=float(ob["bar_high"]),
        fillcolor=OB_BULL_FILL if bullish else OB_BEAR_FILL,
        line=dict(width=0),
        layer="below",
        row=1,
        col=1,
    )

# --- Strong / Weak High / Low (canonical trailing + swing_bias only)
trailing = smc.get("trailing") or {}
swing_bias = int(smc.get("swing_bias") or 0)

top = trailing.get("top")
bottom = trailing.get("bottom")
top_time = trailing.get("last_top_time")
bottom_time = trailing.get("last_bottom_time")

if top is not None and top_time:
    x0 = max(pd.Timestamp(top_time), view_start)
    if x0 <= view_end:
        fig.add_trace(
            go.Scatter(
                x=[x0, view_end],
                y=[float(top), float(top)],
                mode="lines",
                line=dict(color=BEAR, width=1),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=view_end,
            y=float(top),
            text="Strong High" if swing_bias == -1 else "Weak High",
            showarrow=False,
            xanchor="left",
            xshift=6,
            font=dict(color=BEAR, size=9),
            row=1,
            col=1,
        )

if bottom is not None and bottom_time:
    x0 = max(pd.Timestamp(bottom_time), view_start)
    if x0 <= view_end:
        fig.add_trace(
            go.Scatter(
                x=[x0, view_end],
                y=[float(bottom), float(bottom)],
                mode="lines",
                line=dict(color=BULL, width=1),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=view_end,
            y=float(bottom),
            text="Strong Low" if swing_bias == 1 else "Weak Low",
            showarrow=False,
            xanchor="left",
            xshift=6,
            font=dict(color=BULL, size=9),
            row=1,
            col=1,
        )

# --- SQZMOM histogram: per-bar canonical bcolor
mom_view = val[start_pos:]
mom_colors = [MOM_COLORS.get(str(c), "#878B94") for c in bcolor[start_pos:]]

fig.add_trace(
    go.Bar(
        x=view.index,
        y=mom_view,
        marker_color=mom_colors,
        marker_line_width=0,
        name="SQZMOM",
        showlegend=False,
        hovertemplate="%{x}<br>SQZMOM=%{y:.2f}<extra></extra>",
    ),
    row=2,
    col=1,
)

# --- Squeeze state on the zero line (canonical scolor)
sqz_colors = [SQZ_COLORS.get(str(c), "#878B94") for c in scolor[start_pos:]]

fig.add_trace(
    go.Scatter(
        x=view.index,
        y=np.zeros(len(view)),
        mode="markers",
        marker=dict(symbol="x", size=5, color=sqz_colors),
        showlegend=False,
        hoverinfo="skip",
    ),
    row=2,
    col=1,
)

fig.update_layout(
    height=900,
    template="plotly_dark",
    showlegend=False,
    margin=dict(l=10, r=110, t=40, b=10),
    bargap=0.05,
)

fig.update_xaxes(rangeslider_visible=False)

fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(
    title_text="SQZMOM",
    zeroline=True,
    zerolinewidth=1,
    row=2,
    col=1,
)

st.plotly_chart(fig, use_container_width=True)
