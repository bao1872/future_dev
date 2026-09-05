import streamlit as st

from market_data.offline_store import get_market_status, load_bars
from market_data.validation import validate_current_offline_data

st.set_page_config(page_title="Data · future_dev", layout="wide")
st.title("Data")
st.caption("Current offline market data. No dataset versioning.")

status = get_market_status()

rows = []
for tf, info in status["timeframes"].items():
    rows.append({"timeframe": tf, **info})

if rows:
    st.dataframe(rows, use_container_width=True, hide_index=True)
else:
    st.warning("No offline data found. Run `python scripts/refresh_data.py`.")

st.markdown("### Targeted validation")
if st.button("Validate current offline data", type="primary"):
    with st.spinner("Running relevant offline validation..."):
        report = validate_current_offline_data(include_cross_tf=True)
    if report["ok"]:
        st.success("Validation PASS")
    else:
        st.error("Validation FAIL")
    st.json(report)

st.markdown("### Inspect bars")
if status["available_timeframes"]:
    tf = st.selectbox("Timeframe", status["available_timeframes"])
    n = st.slider("Rows", 20, 500, 100, 20)
    df = load_bars(tf)
    st.dataframe(df.tail(n), use_container_width=True, hide_index=True)
