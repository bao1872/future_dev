from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Palette: Panji StrategyChart SSOT. Do not invent a future_dev palette.
# ---------------------------------------------------------------------------

CHART_BG = "#0d1118"
PANEL_BG = "#0a0e15"

GRID = "#252c39"
GRID_SOFT = "#1b2230"

TEXT = "#778297"
TEXT_BRIGHT = "#aab4c8"

PRICE_UP = "#ef5350"
PRICE_DOWN = "#26a69a"

SMC_BULL = "#FF4D4F"
SMC_BEAR = "#22C55E"

# Canonical LazyBear SQZMOM colours - never remapped to brand green.
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

OB_BULL_FILL = "rgba(49,121,245,0.20)"
OB_BEAR_FILL = "rgba(247,124,128,0.20)"

# Strategy decision UI colours - deliberately NOT the SMC indicator colours.
STRATEGY_BUY = "#00F6C2"
STRATEGY_SELL = "#F59E0B"

# Oracle hindsight labels - open symbols keep them visually distinct.
ORACLE_BUY = "#82A0FF"
ORACLE_SELL = "#8B5CF6"

# Panji keeps ~20% of the plot width as the structure extension area.
RIGHT_PAD_RATIO = 0.20


def split_active_order_blocks(smc: dict, limit: int = 5) -> tuple[list[dict], list[dict]]:
    """Active (unmitigated) order blocks, swing and internal, newest first.

    `smc["order_blocks"]` is Panji's historical archive: mitigation only sets
    `mitigated = True`, it never removes the row. This filters it back down to
    what a live chart would currently show.
    """
    active = [ob for ob in smc.get("order_blocks", []) if not ob.get("mitigated", False)]
    swing = [ob for ob in active if ob.get("internal") is False][:limit]
    internal = [ob for ob in active if ob.get("internal") is True][:limit]
    return swing, internal


def build_smc_momentum_figure(
    full: pd.DataFrame,
    smc: dict,
    momentum: dict,
    *,
    display_bars: int | None,
    show_structure: bool = True,
    show_order_blocks: bool = True,
    show_equal_levels: bool = True,
    show_trailing: bool = True,
    show_momentum: bool = True,
    strategy_signals: pd.DataFrame | None = None,
    oracle_labels_df: pd.DataFrame | None = None,
) -> go.Figure:
    """Price + SMC / SQZMOM figure on a continuous bar-index x axis.

    The x coordinate is the bar sequence position, NOT real time. Night
    breaks, lunch breaks, weekends and holidays therefore produce no
    horizontal gaps. Real time is only used for tick labels, hover and
    annotation metadata.

    Canonical output is consumed as-is: nothing is recomputed here.
    """
    view = full.tail(display_bars) if display_bars else full.copy()
    n = len(view)
    if n == 0:
        raise ValueError("empty view")

    start_pos = len(full) - n
    x = np.arange(n)

    rows = 2 if show_momentum else 1
    row_heights = [0.76, 0.24] if show_momentum else [1.0]
    vertical_spacing = 0.025 if show_momentum else 0.0

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=vertical_spacing,
    )

    hover_time = view.index.strftime("%Y-%m-%d %H:%M")

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=view["open"],
            high=view["high"],
            low=view["low"],
            close=view["close"],
            name="Price",
            increasing_line_color=PRICE_UP,
            decreasing_line_color=PRICE_DOWN,
            text=hover_time,
            hoverinfo="all",
        ),
        row=1,
        col=1,
    )

    def to_local(global_index, *, clamp_left: bool = False):
        """Global bar index -> local bar index (None = outside the view)."""
        if global_index is None:
            return None
        local = int(global_index) - start_pos
        if local < 0:
            return 0 if clamp_left else None
        if local >= n:
            return None
        return local

    def time_to_local(t, *, clamp_left: bool = False):
        if not t:
            return None
        try:
            return to_local(full.index.get_loc(pd.Timestamp(t)), clamp_left=clamp_left)
        except (KeyError, TypeError):
            return None

    # --- BOS / CHoCH: pivot -> breakout, label at the segment midpoint
    if show_structure:
        for ev in smc.get("events", []):
            confirmed_x = to_local(ev.get("confirmed_index"))
            if confirmed_x is None:
                continue
            anchor_x = to_local(ev.get("anchor_index"), clamp_left=True)
            if anchor_x is None:
                continue

            level = float(ev["level"])
            bullish = bool(ev.get("bullish"))
            internal = bool(ev.get("internal"))

            color = SMC_BULL if bullish else SMC_BEAR

            fig.add_trace(
                go.Scatter(
                    x=[anchor_x, confirmed_x],
                    y=[level, level],
                    mode="lines",
                    line=dict(
                        color=color,
                        width=1 if internal else 2,
                        dash="dash" if internal else "solid",
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            fig.add_annotation(
                x=(anchor_x + confirmed_x) / 2,
                y=level,
                text=str(ev["type"]),  # BOS / CHoCH
                showarrow=False,
                font=dict(color=color, size=9 if internal else 11),
                yshift=10 if bullish else -10,
                row=1,
                col=1,
            )

    # --- Active order blocks: swing and internal, max 5 each
    if show_order_blocks:
        swing_obs, internal_obs = split_active_order_blocks(smc)
        for ob in swing_obs + internal_obs:
            anchor_x = to_local(ob.get("anchor_index"), clamp_left=True)
            if anchor_x is None:
                continue

            bullish = int(ob["bias"]) == 1
            fig.add_shape(
                type="rect",
                x0=anchor_x,
                x1=n - 1,
                y0=float(ob["bar_low"]),
                y1=float(ob["bar_high"]),
                fillcolor=OB_BULL_FILL if bullish else OB_BEAR_FILL,
                line=dict(width=0),
                layer="below",
                row=1,
                col=1,
            )

    # --- EQH / EQL
    if show_equal_levels:
        for eq in smc.get("equal_highs_lows", []):
            second_x = to_local(eq.get("second_pivot_index"))
            if second_x is None:
                continue
            anchor_x = to_local(eq.get("anchor_index"), clamp_left=True)
            if anchor_x is None:
                continue

            y_prev = eq.get("prev_level")
            y0 = float(y_prev if y_prev is not None else eq["level"])
            y1 = float(eq["level"])

            is_high = eq["type"] == "EQH"
            color = SMC_BEAR if is_high else SMC_BULL

            fig.add_trace(
                go.Scatter(
                    x=[anchor_x, second_x],
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
                x=(anchor_x + second_x) / 2,
                y=(y0 + y1) / 2,
                text=eq["type"],
                showarrow=False,
                font=dict(color=color, size=9),
                yshift=8 if is_high else -8,
                row=1,
                col=1,
            )

    # --- Strong / Weak High / Low (canonical trailing + swing_bias only)
    if show_trailing:
        trailing = smc.get("trailing") or {}
        swing_bias = int(smc.get("swing_bias") or 0)

        for level_key, time_key, is_high in (
            ("top", "last_top_time", True),
            ("bottom", "last_bottom_time", False),
        ):
            level = trailing.get(level_key)
            if level is None:
                continue
            x0 = time_to_local(trailing.get(time_key), clamp_left=True)
            if x0 is None:
                continue

            color = SMC_BEAR if is_high else SMC_BULL
            label = (
                ("Strong High" if swing_bias == -1 else "Weak High")
                if is_high
                else ("Strong Low" if swing_bias == 1 else "Weak Low")
            )

            fig.add_trace(
                go.Scatter(
                    x=[x0, n - 1],
                    y=[float(level), float(level)],
                    mode="lines",
                    line=dict(color=color, width=1),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            fig.add_annotation(
                x=n - 1,
                y=float(level),
                text=label,
                showarrow=False,
                xanchor="left",
                xshift=6,
                font=dict(color=color, size=9),
                row=1,
                col=1,
            )

    # --- Strategy A candidate signals + Oracle hindsight labels
    price_span = float(view["high"].max() - view["low"].min())
    marker_pad = price_span * 0.018 if price_span > 0 else 1.0

    def _strategy_trace(side: str):
        if strategy_signals is None or strategy_signals.empty:
            return

        subset = strategy_signals[strategy_signals["side"] == side].copy()

        if subset.empty:
            return

        xs = []
        ys = []
        hover = []

        for _, sig in subset.iterrows():
            gx = int(sig["bar_index"])
            lx = to_local(gx)

            if lx is None:
                continue

            if side == "BUY":
                y = float(full["low"].iloc[gx]) - marker_pad
            else:
                y = float(full["high"].iloc[gx]) + marker_pad

            reasons = sig["reason"]

            if isinstance(reasons, list):
                reason_html = "<br>".join(str(x) for x in reasons)
            else:
                reason_html = str(reasons)

            hover_text = (
                f"<b>{side} · {sig['event_type']}</b>"
                f"<br>Structure: {sig['structure_level']}"
                f"<br>Signal Bar: {sig['signal_bar_time']}"
                f"<br>Decision Available: {sig['decision_time']}"
                f"<br>4H Swing Bias: {sig['higher_swing_bias']}"
                f"<br>1H Momentum: {sig['momentum_val']} / {sig['momentum_bcolor']}"
                f"<br><br>{reason_html}"
            )

            xs.append(lx)
            ys.append(y)
            hover.append(hover_text)

        if not xs:
            return

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(
                    symbol="triangle-up" if side == "BUY" else "triangle-down",
                    size=13,
                    color=STRATEGY_BUY if side == "BUY" else STRATEGY_SELL,
                    line=dict(width=1, color=CHART_BG),
                ),
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                name=f"Strategy {side}",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    _strategy_trace("BUY")
    _strategy_trace("SELL")

    if oracle_labels_df is not None and not oracle_labels_df.empty:
        for side in ("BUY", "SELL"):
            subset = oracle_labels_df[
                oracle_labels_df["oracle_action"] == side
            ]

            xs = []
            ys = []
            hover = []

            for _, row in subset.iterrows():
                gx = int(row["bar_index"])
                lx = to_local(gx)

                if lx is None:
                    continue

                if side == "BUY":
                    y = float(full["low"].iloc[gx]) - marker_pad * 2.0
                else:
                    y = float(full["high"].iloc[gx]) + marker_pad * 2.0

                xs.append(lx)
                ys.append(y)

                hover.append(
                    f"<b>ORACLE {side}</b>"
                    f"<br>HINDSIGHT LABEL"
                    f"<br>Time: {row['time']}"
                    f"<br>Close: {float(row['close']):.2f}"
                    f"<br>Target position: {int(row['oracle_position'])}"
                )

            if xs:
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers",
                        marker=dict(
                            symbol=(
                                "triangle-up-open"
                                if side == "BUY"
                                else "triangle-down-open"
                            ),
                            size=15,
                            color=ORACLE_BUY if side == "BUY" else ORACLE_SELL,
                            line=dict(width=2),
                        ),
                        text=hover,
                        hovertemplate="%{text}<extra></extra>",
                        name=f"Oracle {side}",
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )

    # --- SQZMOM: per-bar canonical bcolor + canonical scolor on the zero line
    if show_momentum:
        val = np.array(
            [np.nan if v is None else float(v) for v in momentum.get("val", [])],
            dtype=float,
        )
        if len(val) != len(full):
            raise ValueError(f"momentum length {len(val)} != bars {len(full)}")

        bcolor = np.array(momentum.get("bcolor", []), dtype=object)
        scolor = np.array(momentum.get("scolor", []), dtype=object)

        fig.add_trace(
            go.Bar(
                x=x,
                y=val[start_pos:],
                marker_color=[MOM_COLORS.get(str(c), "#878B94") for c in bcolor[start_pos:]],
                marker_line_width=0,
                name="SQZMOM",
                showlegend=False,
                customdata=hover_time,
                hovertemplate="%{customdata}<br>SQZMOM=%{y:.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.zeros(n),
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=5,
                    color=[SQZ_COLORS.get(str(c), "#878B94") for c in scolor[start_pos:]],
                ),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )

    # --- x axis: bar sequence, real time only as labels, 20% right padding
    tick_count = min(8, n)
    tickvals = np.unique(np.linspace(0, n - 1, tick_count).astype(int))
    ticktext = [view.index[i].strftime("%m-%d<br>%H:%M") for i in tickvals]

    fig.update_xaxes(
        range=[-0.5, n - 0.5 + n * RIGHT_PAD_RATIO],
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        gridcolor=GRID,
        zerolinecolor=GRID_SOFT,
    )

    # Candlestick automatically enables Plotly's range slider thumbnail.
    # Disable it: the Panji-style research chart uses the SQZMOM subplot here.
    fig.update_xaxes(rangeslider_visible=False)

    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID_SOFT)

    fig.update_layout(
        height=900 if show_momentum else 700,
        paper_bgcolor="#0A0F14",
        plot_bgcolor=CHART_BG,
        font=dict(color="#98A1B3"),
        showlegend=False,
        hovermode="x unified",
        margin=dict(l=20, r=80, t=20, b=20),
        bargap=0.05,
    )

    return fig
