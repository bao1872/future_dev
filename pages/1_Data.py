import streamlit as st

from market_data.offline_store import get_market_status, load_bars
from market_data.validation import validate_current_offline_data

st.title("数据状态")
st.caption("当前离线行情。TqSdk 只负责下载，页面不会自动刷新或重建连续合约。")

status = get_market_status()

rows = []
for tf, info in status["timeframes"].items():
    rows.append(
        {
            "Timeframe": tf,
            "Rows": info.get("rows"),
            "Start": info.get("start"),
            "End": info.get("end"),
            "Path": info.get("path"),
        }
    )

if rows:
    st.dataframe(rows, use_container_width=True, hide_index=True)
else:
    st.warning("No offline data found. Run `python scripts/refresh_data.py`.")

if st.button("Validate current offline data", type="primary"):
    with st.spinner("Running relevant offline validation..."):
        report = validate_current_offline_data(include_cross_tf=True)
    if report["ok"]:
        st.success("Validation PASS")
    else:
        st.error("Validation FAIL")
    st.json(report)

with st.expander("查看原始行情"):
    if status["available_timeframes"]:
        tf = st.selectbox("Timeframe", status["available_timeframes"], key="inspect_tf")
        n = st.slider("Rows", 20, 500, 100, 20)
        st.dataframe(load_bars(tf).tail(n), use_container_width=True, hide_index=True)
    else:
        st.info("No offline data to inspect.")
