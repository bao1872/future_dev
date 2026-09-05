from __future__ import annotations

import numpy as np
import streamlit as st

from market_data.offline_store import available_timeframes, load_bars, to_indicator_frame
from research.charting import (
    PRICE_DOWN,
    PRICE_UP,
    build_smc_momentum_figure,
    split_active_order_blocks,
)
from research.indicator_adapter import compute_smc_momentum_bundle

st.markdown(
    """
    <style>
    .research-status {
        display: flex;
        flex-wrap: wrap;
        gap: 1.75rem;
        align-items: baseline;
        padding: 0.55rem 0.9rem;
        margin: 0.35rem 0 0.75rem 0;
        background: #111A23;
        border: 1px solid #263440;
        border-radius: 6px;
        font-size: 0.82rem;
        color: #98A1B3;
    }
    .research-status b { font-weight: 600; }
    .research-status .muted { color: #98A1B3; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("研究工作台")
st.caption("沪银主连 · Offline CSV · Panji Canonical SMC + SQZMOM")

tfs = available_timeframes()
if not tfs:
    st.warning("No offline data found. Run `python scripts/refresh_data.py`.")
    st.stop()

st.radio("周期", tfs, key="tf", horizontal=True,
         index=tfs.index("1h") if "1h" in tfs else 0)

disp = st.radio("显示", ["150", "300", "600", "1000", "全部"], key="disp",
                horizontal=True, index=1)
display_bars = None if disp == "全部" else int(disp)

l1, l2, l3, l4, l5 = st.columns(5)
show_structure = l1.toggle("结构", value=True)
show_order_blocks = l2.toggle("OB", value=True)
show_equal_levels = l3.toggle("EQH/EQL", value=True)
show_trailing = l4.toggle("Strong/Weak", value=True)
show_momentum = l5.toggle("Momentum", value=True)

st.markdown("**研究信号**")

s1, s2 = st.columns(2)
show_strategy = s1.toggle("策略A · 4H→1H", value=True)
show_oracle = s2.toggle("Oracle 最优标签", value=False)

oracle_penalty_bps = st.select_slider(
    "Oracle 换仓惩罚",
    options=[5, 10, 20, 30, 50, 80],
    value=20,
    format_func=lambda x: f"{x} bp",
    disabled=not show_oracle,
)

raw = load_bars(st.session_state.tf)
full = to_indicator_frame(raw)

with st.spinner("Computing canonical SMC + SQZMOM on full history..."):
    bundle = compute_smc_momentum_bundle(full)

smc = bundle.smc
momentum = bundle.momentum

# ---------------------------------------------------------------------------
# Compact status strip
# ---------------------------------------------------------------------------

swing_obs, internal_obs = split_active_order_blocks(smc)

val = np.array(
    [np.nan if v is None else float(v) for v in momentum.get("val", [])],
    dtype=float,
)
latest_val = float(val[-1]) if len(val) and np.isfinite(val[-1]) else None
latest_bcolor = str(momentum.get("bcolor", [""])[-1])


def bias_text(bias) -> tuple[str, str]:
    b = int(bias or 0)
    if b == 1:
        return "多头", PRICE_UP
    if b == -1:
        return "空头", PRICE_DOWN
    return "中性", "#98A1B3"


swing_text, swing_color = bias_text(smc.get("swing_bias"))
internal_text, internal_color = bias_text(smc.get("internal_bias"))

mom_text = f"{latest_val:+.0f} · {latest_bcolor}" if latest_val is not None else "N/A"

st.markdown(
    f"""
    <div class="research-status">
      <span class="muted">Swing <b style="color:{swing_color}">{swing_text}</b></span>
      <span class="muted">Internal <b style="color:{internal_color}">{internal_text}</b></span>
      <span class="muted">OB <b style="color:#aab4c8">Swing {len(swing_obs)} / Internal {len(internal_obs)}</b></span>
      <span class="muted">Momentum <b style="color:#aab4c8">{mom_text}</b></span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Strategy A candidates (4H trend -> 1H event + momentum)
# ---------------------------------------------------------------------------

strategy_signals = None

if show_strategy:
    if st.session_state.tf != "1h":
        st.caption("策略A的执行周期是1H；当前周期仅显示指标，不显示策略信号。")
    else:
        from strategies.smc_momentum_signals import generate_signals

        with st.spinner("Generating Strategy A candidates (4H -> 1H)..."):
            strategy_signals = generate_signals(
                higher_4h=to_indicator_frame(load_bars("4h")),
                lower_1h=full,
            )

# ---------------------------------------------------------------------------
# Oracle hindsight labels (close price only, independent of any indicator)
# ---------------------------------------------------------------------------

oracle_df = None
oracle_meta = None

if show_oracle:
    from research.oracle_labels import oracle_labels

    with st.spinner("Solving Oracle hindsight optimum..."):
        oracle_df, oracle_meta = oracle_labels(
            full,
            trade_penalty=oracle_penalty_bps / 10_000.0,
        )

    st.warning(
        "Oracle 是使用完整未来价格路径计算的事后最优训练标签，"
        "包含未来信息，只能用于标签/对照，严禁作为实时策略信号。"
    )
    st.caption(
        f"Oracle penalty: {oracle_penalty_bps} bp · "
        f"actions: {oracle_meta['action_count']} · "
        f"turnover units: {oracle_meta['turnover_units']}"
    )

fig = build_smc_momentum_figure(
    full,
    smc,
    momentum,
    display_bars=display_bars,
    show_structure=show_structure,
    show_order_blocks=show_order_blocks,
    show_equal_levels=show_equal_levels,
    show_trailing=show_trailing,
    show_momentum=show_momentum,
    strategy_signals=strategy_signals,
    oracle_labels_df=oracle_df,
)

st.plotly_chart(fig, use_container_width=True)

view = full.tail(display_bars) if display_bars else full
st.caption(
    f"数据范围 {full.index[0]:%Y-%m-%d %H:%M} → {full.index[-1]:%Y-%m-%d %H:%M}"
    f"（共 {len(full)} 根）· 当前显示 {len(view)} 根："
    f"{view.index[0]:%Y-%m-%d %H:%M} → {view.index[-1]:%Y-%m-%d %H:%M}"
    f" · 指标在完整历史上计算，裁剪只发生在绘图阶段 · "
    f"横轴为连续 bar 序列，非日历时间"
)

# ---------------------------------------------------------------------------
# Decision evidence
# ---------------------------------------------------------------------------

if strategy_signals is not None:
    view_start_pos = max(
        0,
        len(full) - (display_bars if display_bars else len(full)),
    )

    visible_signals = strategy_signals[
        strategy_signals["bar_index"] >= view_start_pos
    ]

    with st.expander(f"策略A信号明细 ({len(visible_signals)})"):
        if visible_signals.empty:
            st.caption("当前视窗没有满足条件的信号。")
        else:
            table = visible_signals[
                [
                    "signal_bar_time",
                    "decision_time",
                    "side",
                    "event_type",
                    "structure_level",
                    "higher_swing_bias",
                    "momentum_val",
                    "momentum_bcolor",
                    "reason",
                ]
            ].copy()

            table["reason"] = table["reason"].apply(
                lambda x: " | ".join(x) if isinstance(x, list) else str(x)
            )

            st.dataframe(table, use_container_width=True, hide_index=True)

if oracle_df is not None:
    with st.expander("Oracle 标签明细"):
        actions = oracle_df[oracle_df["oracle_action"] != "HOLD"]
        st.dataframe(
            actions[
                [
                    "time",
                    "close",
                    "oracle_action",
                    "oracle_position",
                    "oracle_delta",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
