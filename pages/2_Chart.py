from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from market_data.offline_store import available_timeframes, load_bars, to_indicator_frame
from research.indicator_adapter import compute_canonical_bundle

st.set_page_config(page_title="Chart · future_dev", layout="wide")
st.title("Chart")
st.caption("Canonical calculations run on full offline history; display cropping happens afterward.")

tfs = available_timeframes()
if not tfs:
    st.warning("No offline data found.")
    st.stop()

c1, c2 = st.columns([1, 2])
with c1:
    tf = st.selectbox("Timeframe", tfs, index=tfs.index("15m") if "15m" in tfs else 0)
with c2:
    plot_bars = st.slider("Display bars", 100, 1000, 300, 50)

show_dsa = st.checkbox("DSA VWAP", value=True)
show_structure = st.checkbox("BOS / CHoCH", value=True)
show_ob = st.checkbox("Order Blocks", value=True)
show_momentum = st.checkbox("Momentum", value=True)

raw = load_bars(tf)
full = to_indicator_frame(raw)

with st.spinner("Computing canonical indicators on full history..."):
    bundle = compute_canonical_bundle(full)

view = full.tail(plot_bars).copy()
start_pos = len(full) - len(view)

fig = go.Figure()
fig.add_trace(
    go.Candlestick(
        x=view.index,
        open=view["open"],
        high=view["high"],
        low=view["low"],
        close=view["close"],
        name="Price",
    )
)

if show_dsa:
    dsa = bundle.dsa_vwap.loc[view.index]
    fig.add_trace(go.Scatter(x=view.index, y=dsa, mode="lines", name="DSA VWAP"))

if show_structure:
    for ev in bundle.smc.get("events", []):
        i = ev.get("confirmed_index")
        if i is None or i < start_pos or i >= len(full):
            continue
        ts = full.index[i]
        price = float(ev.get("level"))
        label = ("i" if ev.get("internal") else "s") + " " + str(ev.get("type"))
        fig.add_annotation(
            x=ts,
            y=price,
            text=label,
            showarrow=True,
            arrowhead=1,
            yshift=12 if ev.get("bullish") else -12,
        )

if show_ob:
    view_start = view.index[0]
    view_end = view.index[-1]
    for ob in bundle.smc.get("order_blocks", []):
        ci = ob.get("confirmed_index")
        if ci is None or ci >= len(full):
            continue
        confirmed_time = full.index[ci]
        mi = ob.get("mitigated_index")
        end_time = full.index[mi] if mi is not None and mi < len(full) else view_end
        if end_time < view_start or confirmed_time > view_end:
            continue
        x0 = max(confirmed_time, view_start)
        x1 = min(end_time, view_end)
        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=float(ob["bar_low"]),
            y1=float(ob["bar_high"]),
            opacity=0.10,
            line_width=1,
        )

fig.update_layout(
    height=700,
    xaxis_rangeslider_visible=False,
    margin=dict(l=10, r=10, t=40, b=10),
)
st.plotly_chart(fig, use_container_width=True)

if show_momentum:
    val = np.asarray(bundle.momentum.get("val", []), dtype=float)
    mom = pd.Series(val, index=full.index).loc[view.index]
    mfig = go.Figure(go.Bar(x=mom.index, y=mom.values, name="SQZMOM"))
    mfig.add_hline(y=0)
    mfig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(mfig, use_container_width=True)

with st.expander("Latest canonical state"):
    timeline = bundle.smc.get("state_timeline") or []
    latest_state = timeline[-1] if timeline else {}
    st.json(
        {
            "swing_bias": bundle.smc.get("swing_bias"),
            "internal_bias": bundle.smc.get("internal_bias"),
            "active_internal_ob_count": latest_state.get("active_internal_ob_count"),
            "active_swing_ob_count": latest_state.get("active_swing_ob_count"),
            "momentum_last": float(val[-1]) if len(val) and np.isfinite(val[-1]) else None,
        }
    )
