"""
pages/6_Indicator_Viewer.py
==========================

Indicator Viewer — current God-Mode Oracle.

This page shows ONLY the current production God-Mode Oracle. There is no
selector and no old Oracle overlay: the page's single source of truth is the
materialized LATEST artifact (`artifacts/god_oracle_m15_latest/`), produced
offline by `build_god_oracle_latest_artifact_v1.py`. No Oracle or environment
math executes inside the viewer.

For each candidate event the oracle decides a single completed trade:

    Candidate A (structural zone)
        -> best Entry
        -> frozen Target B
        -> first touch of B = Trade terminal (TARGET_TOUCH, no early exit)

Rendering rules (manual label review):

* Kline uses the GLOBAL 15m BAR INDEX as the geometric x-axis, so trading
  bars stay contiguous (i, i+1, i+2 …) with no artificial overnight /
  weekend gaps. The real `bar_start_time` is kept only for hover text and
  x-axis tick labels.
* One contiguous index slice `[lo, hi]` is rendered (never a concatenated
  subset of event bars). It contains candidate start, Entry, Exit plus
  padding.
* SR / Liquidity structures are read from the materialized latest artifact
  (`structures.parquet`) — the EXACT decision-time geometry the Oracle used.
  No viewer-side SR/liquidity calculation is invented or recomputed.
* Every canonical trade whose Entry or Exit lies in the visible index range
  is drawn (Entry + Exit + Entry->Exit line). LONG entry = up triangle,
  SHORT entry = down triangle, Exit = X. Non-selected are smaller / lighter;
  the selected trade is larger / stronger.
* Selected Candidate A and selected Target B are highlighted more strongly
  than the background structures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import hashlib
import json
from pathlib import Path

# Single "latest" artifact location. All Oracle / environment math is computed
# OFFLINE by build_god_oracle_latest_artifact_v1.py; the viewer only reads it.
_ARTIFACT_DIR = (
    Path(__file__).resolve().parents[1] / "artifacts" / "god_oracle_m15_latest"
)

SYMBOLS = ["AG"]
VIEW_CONTEXT = 35  # bars of context before A and after Exit

C_BG = "#131722"
C_GRID = "#2A2E39"
C_TEXT = "#D1D4DC"
C_BULL = "#26A69A"
C_BEAR = "#F23645"


def load_oracle(symbol: str) -> dict:
    """Load the materialized LATEST oracle artifact (trades/bars/structures +
    manifest). No Oracle or environment math runs in the viewer — the artifact
    is the single source of truth, produced offline by
    build_god_oracle_latest_artifact_v1.py."""
    trades = pd.read_parquet(_ARTIFACT_DIR / "trades.parquet")
    bars = pd.read_parquet(_ARTIFACT_DIR / "bars.parquet")
    structs = pd.read_parquet(_ARTIFACT_DIR / "structures.parquet")
    with open(_ARTIFACT_DIR / "manifest.json") as fh:
        manifest = json.load(fh)

    canon = trades.to_dict("records")
    canon.sort(key=lambda r: int(r["candidate_decision_index"]))

    n = int(len(bars))
    geom = [None] * n
    for _, row in structs.iterrows():
        b = int(row["bar_index"])
        if 0 <= b < n:
            g = json.loads(row["geom"])
            geom[b] = g if g else None

    artifact_sha = hashlib.sha256(
        (_ARTIFACT_DIR / "manifest.json").read_bytes()
    ).hexdigest()

    return {
        "canon": canon,
        "meta": manifest.get("oracle_meta", {}),
        "opens": np.asarray(bars["open"].to_numpy(), dtype=float),
        "highs": np.asarray(bars["high"].to_numpy(), dtype=float),
        "lows": np.asarray(bars["low"].to_numpy(), dtype=float),
        "closes": np.asarray(bars["close"].to_numpy(), dtype=float),
        "times": np.asarray(bars["bar_start_time"].to_numpy()),
        "n": n,
        "geom": geom,
        "artifact_sha": artifact_sha,
        "canonical_trade_count": int(
            manifest.get("canonical_trade_count", len(canon))
        ),
    }


def _to_dt(arr) -> pd.DatetimeIndex:
    return pd.to_datetime(np.asarray(arr))


def _build_fig(data: dict, sel: dict) -> go.Figure:
    n = data["n"]
    s = int(sel["candidate_decision_index"])
    e = int(sel["exit_fill_index"])
    entry_idx = int(sel["best_entry_fill_index"])

    # one contiguous global-index window around the selected trade
    lo = max(0, s - VIEW_CONTEXT)
    hi = min(n - 1, max(e, s) + VIEW_CONTEXT)
    idx = np.arange(lo, hi + 1, dtype=int)

    times = _to_dt(data["times"][lo : hi + 1])
    time_str = [t.strftime("%Y-%m-%d %H:%M") for t in times]

    # candlestick on the bar-index axis; real time carried via customdata
    fig = go.Figure(
        go.Candlestick(
            x=idx,
            open=data["opens"][lo : hi + 1],
            high=data["highs"][lo : hi + 1],
            low=data["lows"][lo : hi + 1],
            close=data["closes"][lo : hi + 1],
            customdata=time_str,
            hovertemplate=(
                "%{customdata}<br>O %{open:.1f}  H %{high:.1f}"
                "<br>L %{low:.1f}  C %{close:.1f}<extra></extra>"
            ),
            name="Kline",
            increasing={"line": {"color": C_BULL}},
            decreasing={"line": {"color": C_BEAR}},
        )
    )

    direction = str(sel["oracle_direction"])
    bull = direction == "LONG"
    col = C_BULL if bull else C_BEAR
    zb, zt = float(sel["zone_bottom"]), float(sel["zone_top"])
    target_price = float(sel["target_price"])
    entry_price = float(sel["best_entry_price"])
    exit_price = float(sel["exit_price"])

    # visible price window — include key levels so Candidate A / Target B
    # structures are not clipped away
    ymin = float(min(data["lows"][lo : hi + 1]))
    ymax = float(max(data["highs"][lo : hi + 1]))
    for v in (zb, zt, target_price, entry_price, exit_price):
        ymin, ymax = min(ymin, v), max(ymax, v)
    pad = 0.02 * (ymax - ymin) if ymax > ymin else 1.0
    ymin, ymax = ymin - pad, ymax + pad

    # ---- production geometry (decision-time), reused EXACTLY from the Oracle ----
    di = (s - 1) if s >= 1 else s
    g = None
    if data["geom"] is not None and 0 <= di < len(data["geom"]):
        g = data["geom"][di]
    if g:
        ref_price = (
            float(data["closes"][di]) if 0 <= di < len(data["closes"]) else 0.5 * (zb + zt)
        )
        for tf, val in g.items():
            channels, liq_up, liq_down, _atr = val
            # SR zones
            for (top, bottom, strength) in channels:
                top, bottom = float(top), float(bottom)
                if top < ymin or bottom > ymax:
                    continue
                role = "SUP" if 0.5 * (top + bottom) < ref_price else "RES"
                fig.add_shape(
                    type="rect", xref="x", yref="y", x0=lo, x1=hi,
                    y0=bottom, y1=top,
                    fillcolor="rgba(150,160,190,0.07)", line={"width": 0},
                    layer="below",
                )
                fig.add_annotation(
                    x=lo, y=top, text=f"{role} {tf}", showarrow=False,
                    font={"color": C_TEXT, "size": 9},
                    xanchor="left", yanchor="bottom",
                )
            # liquidity — buy-side / sell-side
            for z in liq_up:
                if z.get("broken"):
                    continue
                lvl = float(z["level"])
                if lvl < ymin or lvl > ymax:
                    continue
                fig.add_shape(
                    type="line", xref="x", yref="y", x0=lo, x1=hi, y0=lvl, y1=lvl,
                    line={"color": "rgba(38,166,154,0.40)", "width": 1, "dash": "dash"},
                )
                fig.add_annotation(
                    x=hi, y=lvl, text=f"BUY {tf} {lvl:.0f}", showarrow=False,
                    font={"color": C_BULL, "size": 9},
                    xanchor="right", yanchor="bottom",
                )
            for z in liq_down:
                if z.get("broken"):
                    continue
                lvl = float(z["level"])
                if lvl < ymin or lvl > ymax:
                    continue
                fig.add_shape(
                    type="line", xref="x", yref="y", x0=lo, x1=hi, y0=lvl, y1=lvl,
                    line={"color": "rgba(242,54,69,0.40)", "width": 1, "dash": "dash"},
                )
                fig.add_annotation(
                    x=hi, y=lvl, text=f"SELL {tf} {lvl:.0f}", showarrow=False,
                    font={"color": C_BEAR, "size": 9},
                    xanchor="right", yanchor="bottom",
                )

    # ---- selected Candidate A (highlighted) ----
    fig.add_shape(
        type="rect", xref="x", yref="y", x0=lo, x1=hi, y0=zb, y1=zt,
        fillcolor="rgba(120,160,220,0.16)",
        line={"color": "rgba(180,200,240,0.85)", "width": 1}, layer="below",
    )
    fig.add_annotation(
        x=lo, y=zt, text="Candidate A", showarrow=False,
        font={"color": C_TEXT, "size": 11}, xanchor="left", yanchor="bottom",
    )

    # ---- selected Target B (highlighted) ----
    fig.add_shape(
        type="line", xref="x", yref="y", x0=lo, x1=hi, y0=target_price, y1=target_price,
        line={"color": col, "width": 2, "dash": "dot"},
    )
    fig.add_annotation(
        x=hi, y=target_price, text=f"Target B {target_price:.1f}", showarrow=False,
        font={"color": col, "size": 11}, xanchor="right", yanchor="bottom",
    )

    # ---- ALL canonical trades visible in this index window ----
    for cr in data["canon"]:
        ei = int(cr["best_entry_fill_index"])
        xi = int(cr["exit_fill_index"])
        if ei > hi or xi < lo:
            continue
        is_sel = ei == entry_idx
        cdir = str(cr["oracle_direction"])
        c = C_BULL if cdir == "LONG" else C_BEAR
        op = 1.0 if is_sel else 0.45
        esize = 12 if is_sel else 7
        eprice = float(cr["best_entry_price"])
        xprice = float(cr["exit_price"])
        fig.add_trace(go.Scatter(
            x=[ei], y=[eprice], mode="markers",
            marker={"size": esize, "color": c,
                    "symbol": "triangle-up" if cdir == "LONG" else "triangle-down",
                    "opacity": op, "line": {"color": "white", "width": 1}},
            name=(f"Entry {cr['event_id']}" if is_sel else None),
            showlegend=is_sel,
            hovertemplate=(
                f"Entry {cr['event_id']}<br>"
                f"{pd.Timestamp(cr['best_entry_fill_time']).strftime('%Y-%m-%d %H:%M')}<br>"
                "%{y:.1f}<extra></extra>"
            ),
        ))
        fig.add_trace(go.Scatter(
            x=[xi], y=[xprice], mode="markers",
            marker={"size": esize, "color": c, "symbol": "x",
                    "opacity": op, "line": {"color": "white", "width": 1}},
            name=(f"Exit {cr['event_id']}" if is_sel else None),
            showlegend=is_sel,
            hovertemplate=(
                f"Exit {cr['event_id']}<br>"
                f"{pd.Timestamp(cr['exit_fill_time']).strftime('%Y-%m-%d %H:%M')}<br>"
                "%{y:.1f}<extra></extra>"
            ),
        ))
        fig.add_trace(go.Scatter(
            x=[ei, xi], y=[eprice, xprice], mode="lines",
            line={"color": c, "width": 2 if is_sel else 1,
                  "dash": "solid" if is_sel else "dot"},
            opacity=op, showlegend=False, hoverinfo="skip",
        ))

    step = max(1, len(idx) // 8)
    fig.update_layout(
        title=f"Event {sel['event_id']} · {direction} · TARGET_TOUCH",
        plot_bgcolor=C_BG, paper_bgcolor=C_BG, font={"color": C_TEXT},
        xaxis={"type": "linear", "gridcolor": C_GRID, "title": "15m bar index",
               "tickmode": "array", "tickvals": list(idx[::step]),
               "ticktext": [time_str[i] for i in range(0, len(idx), step)]},
        yaxis={"gridcolor": C_GRID, "title": "price", "range": [ymin, ymax]},
        legend={"orientation": "h", "y": 1.02},
        margin={"l": 55, "r": 20, "t": 40, "b": 20},
        height=600,
    )
    fig.update_xaxes(range=[lo, hi])
    return fig


def _meta_table(sel: dict) -> pd.DataFrame:
    rows = [
        ("event_id", sel["event_id"]),
        ("direction", sel["oracle_direction"]),
        ("candidate_start_time", pd.Timestamp(sel["candidate_start_time"])),
        ("candidate zone", f"{float(sel['zone_bottom']):.1f} – {float(sel['zone_top']):.1f}"),
        ("entry time", pd.Timestamp(sel["best_entry_fill_time"])),
        ("entry price", f"{float(sel['best_entry_price']):.1f}"),
        ("best_entry_gap_atr", f"{float(sel['best_entry_gap_atr']):.3f}"),
        ("target type", sel.get("target_structure_type")),
        ("target price", f"{float(sel['target_price']):.1f}"),
        ("exit time", pd.Timestamp(sel["exit_fill_time"])),
        ("exit price", f"{float(sel['exit_price']):.1f}"),
        ("tp_atr", f"{float(sel['tp_atr']):.3f}"),
    ]
    return pd.DataFrame(rows, columns=["field", "value"])


def main() -> None:
    st.markdown("### God-Mode Oracle · Indicator Viewer")
    st.caption(
        "Latest production oracle only: each valid opportunity becomes one "
        "completed trade in a single sequential stream — Candidate A → best "
        "Entry → frozen Target B → first touch of B (TARGET_TOUCH), then the "
        "search restarts immediately after the exit. Previous / Next step "
        "through the trade stream; event_id is diagnostic metadata only. "
        "Kline uses the global 15m bar index (no overnight / weekend gaps); "
        "SR & Liquidity are read from the materialized latest artifact "
        "(structures.parquet: decision-time production geometry, no runtime "
        "computation)."
    )

    symbol = st.sidebar.selectbox("Symbol", SYMBOLS, index=0)
    data = load_oracle(symbol)
    canon = data["canon"]
    meta = data["meta"]

    if not canon:
        st.warning("No canonical labels for this symbol.")
        return

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**canonical labels:** {len(canon)}  ·  "
        f"TARGET_TOUCH: {meta.get('target_touch', 0)}  ·  "
        f"early_exit: {meta.get('early_exit', 0)}"
    )

    idx = st.session_state.get("v_idx", 0)
    idx = max(0, min(idx, len(canon) - 1))

    c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
    if c1.button("◀ Previous", use_container_width=True) and idx > 0:
        idx -= 1
    if c2.button("Next ▶", use_container_width=True) and idx < len(canon) - 1:
        idx += 1
    c3.number_input(
        "Trade #", min_value=1, max_value=len(canon),
        value=idx + 1, key="v_jump",
        on_change=lambda: st.session_state.update(v_idx=int(st.session_state.v_jump) - 1),
    )
    c4.markdown(
        f"selected **Trade {idx + 1} / {len(canon)}**  "
        f"(event_id = `{canon[idx]['event_id']}`)"
    )
    st.session_state["v_idx"] = idx

    # ---- temporary page-load diagnostics (artifact sourcing; no runtime oracle) ----
    sel = canon[idx]
    _sha = data["artifact_sha"]
    _canon_n = data["canonical_trade_count"]
    _seq = idx + 1
    _entry = float(sel["best_entry_price"])
    print(
        f"[artifact] SHA={_sha}\n"
        f"[artifact] canonical count={_canon_n}\n"
        f"[artifact] selected trade seq={_seq}\n"
        f"[artifact] selected entry={_entry}"
    )
    st.markdown(
        f"**artifact SHA** = `{_sha[:16]}`  ·  "
        f"**artifact canonical count** = {_canon_n}  ·  "
        f"**selected trade seq** = {_seq}  ·  "
        f"**selected entry** = {_entry:.1f}"
    )

    fig = _build_fig(data, sel)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Selected event")
    st.table(_meta_table(sel))


if __name__ == "__main__":
    main()
