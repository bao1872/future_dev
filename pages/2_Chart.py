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
