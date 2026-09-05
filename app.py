from pathlib import Path
import streamlit as st

from market_data.offline_store import get_market_status

st.set_page_config(
    page_title="future_dev Research Lab",
    page_icon="📈",
    layout="wide",
)

st.title("future_dev · Futures Research Lab")
st.caption("TqSdk acquisition · offline research · Streamlit workbench")

status = get_market_status()

c1, c2, c3 = st.columns(3)
c1.metric("Instrument", status["display_name"])
c2.metric("Source", status["source"])
c3.metric("Available TF", ", ".join(status["available_timeframes"]) or "None")

st.markdown("### Research boundaries")
st.markdown(
    """
- **TqSdk** only acquires market data.
- **Offline CSV** is the strategy/research input.
- **Panji canonical indicators** are calculation SSOT.
- **Streamlit** is the research UI.
- Current phase is **research**, not live trading.
"""
)

st.markdown("### Current offline data")
if not status["timeframes"]:
    st.warning("No offline market data found. Run `python scripts/refresh_data.py`.")
else:
    rows = []
    for tf, item in status["timeframes"].items():
        rows.append(
            {
                "timeframe": tf,
                "rows": item.get("rows"),
                "start": item.get("start"),
                "end": item.get("end"),
                "file": item.get("path"),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

st.info("Use the pages in the sidebar for Data, Chart, Strategy Lab and Results.")
