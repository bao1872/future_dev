"""
V6.2 AUDIT VIEWER  (EXPERIMENTAL, standalone page)

Reads ONLY the frozen audit artifact under
    artifacts/god_oracle_m15_v62_audit/

It NEVER re-runs Oracle label math. The production V6.1 Viewer / artifact
are untouched. Launch it directly:

    streamlit run pages/7_V62_Audit_Viewer.py

Visual contract
---------------
For each V6.2 canonical label the page draws, on an OHLC chart:

  * Source A zone
  * observed_leg_terminal B
  * next source B
  * best Entry decision / fill / price
  * Planned Target (frozen at entry decision)
  * actual first Target Touch / exit
  * entry-time visible SR / Liquidity geometry

Hard visual rules:
  * The Planned Target must overlap an ENTRY-TIME VISIBLE structure.
    If not -> red HARD FAIL banner (do not silently render).
  * Future structures (not present at the entry decision) are greyed and
    labelled "FUTURE -- NOT ELIGIBLE" and only shown when the toggle is on.
  * A terminal bar that is (A retouch + exactly one distinct B) is annotated
    "A RETOUCH" / "B = NEXT SOURCE".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

ART = _REPO / "artifacts" / "god_oracle_m15_v62_audit"

st.set_page_config(page_title="V6.2 Audit Viewer (experimental)", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.0rem; padding-bottom: 1.5rem; max-width: 1900px; }
    .ev-warn {
        padding: 0.5rem 0.9rem; margin: 0 0 0.7rem 0;
        background: #2A1410; border-left: 3px solid #C9602E;
        border-radius: 4px; color: #E8A87A; font-size: 0.8rem; line-height: 1.5;
    }
    .hard-fail {
        padding: 0.55rem 0.9rem; margin: 0.4rem 0;
        background: #3A0D0D; border: 1px solid #E8493C; border-radius: 5px;
        color: #FF9A8F; font-weight: 650; font-size: 0.9rem;
    }
    .hard-pass {
        padding: 0.55rem 0.9rem; margin: 0.4rem 0;
        background: #0E2A17; border: 1px solid #2EBD85; border-radius: 5px;
        color: #8FF0B6; font-weight: 650; font-size: 0.9rem;
    }
    .kv { font-size: 0.82rem; line-height: 1.7; color: #C7D1DB; }
    .kv b { color: #E6EDF3; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div style='font-size:1.1rem;font-weight:650;letter-spacing:-0.01em;margin:0 0 0.2rem 0'>"
    "God Oracle V6.2 · Entry-Time Audit Viewer</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='ev-warn'>EXPERIMENTAL. 本页只读取冻结的 V6.2 审计 artifact，"
    "不重跑 Oracle 标签数学。生产 V6.1 Viewer 与 artifact 完全未改动。"
    "用于人工逐笔核对：Target 在 Entry 时是否已存在、A→B handoff 是否符合直觉。</div>",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# load artifact (frozen)                                                       #
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading frozen V6.2 audit artifact...")
def load_artifact():
    if not (ART / "trades.parquet").exists():
        return None, None, None
    trades = pd.read_parquet(ART / "trades.parquet")
    bars = pd.read_parquet(ART / "bars.parquet")
    meta = json.loads((ART / "meta.json").read_text())
    return trades, bars, meta


trades, bars, meta = load_artifact()

if trades is None:
    st.error(f"Artifact not found at {ART}. Run "
             "`python research/liquidity_oracle_atlas/build_v62_audit_artifact_v1.py` "
             "first.")
    st.stop()

N = len(trades)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _overlap(b0, t0, b1, t1, eps=1e-6):
    return not (b1 > t0 + eps or t1 < b0 - eps)


def _zones(json_str):
    try:
        return json.loads(json_str) if json_str else []
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# sidebar navigation                                                           #
# --------------------------------------------------------------------------- #
if "v62_idx" not in st.session_state:
    st.session_state["v62_idx"] = 0

special = (meta or {}).get("special_cases", {})
special_opts = ["—"] + [f"{k} ({len(v)})" for k, v in special.items() if v]

c_nav = st.sidebar.columns([1, 1])
if c_nav[0].button("◀ Prev", width="stretch"):
    st.session_state["v62_idx"] = max(0, st.session_state["v62_idx"] - 1)
if c_nav[1].button("Next ▶", width="stretch"):
    st.session_state["v62_idx"] = min(N - 1, st.session_state["v62_idx"] + 1)

idx = st.sidebar.number_input("Trade #", 0, N - 1,
                              value=st.session_state["v62_idx"], step=1,
                              key="v62_num")
st.session_state["v62_idx"] = int(idx)

sel = st.sidebar.selectbox("Quick jump (special review cases)", special_opts)
if sel != "—":
    key = sel.split(" (")[0]
    ids = special.get(key, [])
    if ids and int(idx) not in ids:
        st.session_state["v62_idx"] = ids[0]
        idx = ids[0]
        st.rerun()

st.sidebar.radio("Geometry view", ["Entry-time geometry", "Full evolving geometry"],
                 index=0, key="geom_view", horizontal=True)
st.sidebar.checkbox("Show future geometry (greyed, NOT ELIGIBLE)",
                    value=False, key="show_future")

row = trades.iloc[int(idx)]
src_bar = int(row["source_bar"])
d = int(row["best_entry_decision_index"])
f = int(row["best_entry_fill_index"])
exit_idx = int(row["exit_fill_index"])
src_bot, src_top = float(row["zone_bottom"]), float(row["zone_top"])
tgt_bot = float(row["target_location_bottom"])
tgt_top = float(row["target_location_top"])
tgt_price = float(row["target_price"])

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<div class='kv'>"
    f"<b>source_bar</b> : {src_bar}<br/>"
    f"<b>entry_decision</b> : {d}<br/>"
    f"<b>entry_fill</b> : {f}<br/>"
    f"<b>exit_bar</b> : {exit_idx}<br/>"
    f"<b>target_snapshot_index</b> : {int(row['target_snapshot_index'])}<br/>"
    f"<b>direction</b> : {row['oracle_direction']}<br/>"
    f"<b>utility</b> : {float(row['utility']):.2f}</div>",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# chart window                                                                 #
# --------------------------------------------------------------------------- #
lo = max(0, min(src_bar, d) - 25)
hi = min(len(bars) - 1, exit_idx + 25)
win = bars[(bars["bar_index"] >= lo) & (bars["bar_index"] <= hi)].copy()
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=win["bar_index"], open=win["open"], high=win["high"],
    low=win["low"], close=win["close"], name="OHLC",
    increasing_line_color="#2EBD85", decreasing_line_color="#E8493C",
))

xrange = (lo, hi)


def add_band(y0, y1, color, opacity, label, x0=None, x1=None):
    x0 = lo if x0 is None else x0
    x1 = hi if x1 is None else x1
    fig.add_hrect(y0=y0, y1=y1, x0=x0, x1=x1, fillcolor=color,
                  opacity=opacity, line_width=0, layer="below")
    fig.add_annotation(x=x0, y=y1, text=label, showarrow=False,
                       font=dict(size=10, color=color), xanchor="left",
                       yanchor="bottom", bgcolor="rgba(0,0,0,0.35)")


def add_vline(x, color, label):
    fig.add_vline(x=x, line_color=color, line_width=1.5, line_dash="dot")
    fig.add_annotation(x=x, y=1.0, yref="paper", text=label, showarrow=False,
                       font=dict(size=10, color=color), yanchor="top")


# Source A
add_band(src_bot, src_top, "#2EBD85", 0.18, "SOURCE A")

# Next source (if present)
if pd.notna(row.get("next_source_bar")):
    nb = int(row["next_source_bar"])
    nbot, ntop = float(row["next_source_bottom"]), float(row["next_source_top"])
    add_band(nbot, ntop, "#8E7CE8", 0.15, f"NEXT SOURCE (bar {nb})")

# Entry-time geometry zones (the eye-test core)
entry_zones = _zones(row["entry_geometry"])
future_zones = _zones(row["future_geometry"])
show_full = st.session_state["geom_view"] == "Full evolving geometry"
show_future = st.session_state["show_future"]

for z in entry_zones:
    zb, zt = float(z["bottom"]), float(z["top"])
    color = "#4EA8FF" if z["family"] == "SR" else "#36C5C0"
    tag = f"{z['tf']} {z['family']}" + (f" {z['side']}" if z.get("side") else "")
    add_band(zb, zt, color, 0.10, tag)

if (show_full or show_future):
    for z in future_zones:
        zb, zt = float(z["bottom"]), float(z["top"])
        add_band(zb, zt, "#7C8798", 0.08, "FUTURE -- NOT ELIGIBLE")

# Planned Target (must overlap an entry-time visible structure)
add_band(tgt_bot, tgt_top, "#E8A33C", 0.22, "PLANNED TARGET (ENTRY-TIME VISIBLE)")
fig.add_hline(y=tgt_price, line_color="#E8A33C", line_width=1.5, line_dash="dash")

# Entry marker
add_vline(f, "#FFD166", f"ENTRY @ {float(row['best_entry_price']):.1f}")
fig.add_trace(go.Scatter(
    x=[f], y=[float(row["best_entry_price"])],
    mode="markers", marker=dict(size=11, color="#FFD166", symbol="circle"),
    name="Entry fill", showlegend=False,
))

# Exit marker (actual first target touch)
if exit_idx <= hi:
    fig.add_trace(go.Scatter(
        x=[exit_idx], y=[float(row["exit_price"])],
        mode="markers", marker=dict(size=11, color="#E8493C", symbol="x"),
        name="Target touch", showlegend=False,
    ))
    add_vline(exit_idx, "#E8493C", "EXIT")

# Terminal bar handoff annotation
if bool(row["terminal_had_retouch_plus_distinct"]):
    tb = int(row["observed_leg_terminal"])
    add_vline(tb, "#C9602E", "TERMINAL")
    for loc in json.loads(row["terminal_locations"]):
        if loc["role"] == "retouch":
            add_band(float(loc["bottom"]), float(loc["top"]), "#2EBD85", 0.16,
                     "A RETOUCH", x0=tb - 0.5, x1=tb + 0.5)
        else:
            add_band(float(loc["bottom"]), float(loc["top"]), "#E8A33C", 0.22,
                     "B = NEXT SOURCE", x0=tb - 0.5, x1=tb + 0.5)

fig.update_layout(
    height=560, margin=dict(l=10, r=10, t=10, b=10),
    xaxis_rangeslider_visible=False, showlegend=False,
    title=dict(text=f"AG · V6.2 trade #{int(idx)} · {row['oracle_direction']} · "
                    f"source_bar {src_bar}", font=dict(size=13)),
)
st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- #
# Target visibility HARD check                                                  #
# --------------------------------------------------------------------------- #
target_overlaps = any(
    _overlap(tgt_bot, tgt_top, float(z["bottom"]), float(z["top"]))
    for z in entry_zones
)
if target_overlaps:
    st.markdown(
        "<div class='hard-pass'>PLANNED TARGET AT ENTRY — PASS: target overlaps an "
        "ENTRY-TIME VISIBLE SR / Liquidity structure.</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<div class='hard-fail'>PLANNED TARGET AT ENTRY — HARD FAIL: target does NOT "
        "overlap any ENTRY-TIME VISIBLE structure. It would be a future-leak label.</div>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
# info cards                                                                   #
# --------------------------------------------------------------------------- #
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("<div class='kv'><b>Source A</b><br/>"
                f"bar {src_bar}<br/>zone [{src_bot:.1f}, {src_top:.1f}]<br/>"
                f"id {row.get('source_structure_id')}</div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='kv'><b>Entry</b><br/>"
                f"decision {d}<br/>fill {f} @ {float(row['best_entry_price']):.1f}<br/>"
                f"direction {row['oracle_direction']}</div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='kv'><b>Target / Exit</b><br/>"
                f"snapshot {int(row['target_snapshot_index'])} == decision {d} "
                f"({'OK' if int(row['target_snapshot_index']) == d else 'MISMATCH'})<br/>"
                f"price {tgt_price:.1f}<br/>exit {exit_idx} @ "
                f"{float(row['exit_price']):.1f} ({row['exit_reason']})</div>",
                unsafe_allow_html=True)

st.markdown("---")
st.caption(
    f"artifact: artifacts/god_oracle_m15_v62_audit/ · "
    f"{N} V6.2 canonical labels · geometry view = "
    f"{st.session_state['geom_view']} · "
    f"target_snapshot==entry_decision invariant (causal freeze). "
    f"本页不重跑 Oracle 数学。"
)
