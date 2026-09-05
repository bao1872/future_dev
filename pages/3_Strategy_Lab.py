from __future__ import annotations

import json

import streamlit as st

from market_data.offline_store import load_bundle
from research.experiment_store import save_experiment
from strategies.registry import all_strategies

st.title("策略实验")
st.caption("只在离线行情上验证假设。策略不接触 PyTDX。")

registry = all_strategies()
if not registry:
    st.info("当前尚未定义交易假设。")
    st.markdown(
        """
预定研究框架：

- **A · 4H → 1H**
- **B · 1H → 15m**

策略规则将在研究工作台指标验证完成后定义。
"""
    )
    st.stop()

name = st.selectbox("Strategy", list(registry))
spec = registry[name]
st.write(spec.description)

with st.expander("高级参数"):
    params_text = st.text_area(
        "Parameters (JSON)",
        value=json.dumps(spec.default_params, ensure_ascii=False, indent=2),
        height=180,
    )

note = st.text_input("Research note", "")

bundle = load_bundle()
common_start = max(df["datetime"].min() for df in bundle.values())
common_end = min(df["datetime"].max() for df in bundle.values())

c1, c2 = st.columns(2)
start = c1.date_input(
    "Start",
    value=common_start.date(),
    min_value=common_start.date(),
    max_value=common_end.date(),
)
end = c2.date_input(
    "End",
    value=common_end.date(),
    min_value=common_start.date(),
    max_value=common_end.date(),
)

if st.button("Run strategy", type="primary"):
    try:
        params = json.loads(params_text)
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON: {exc}")
        st.stop()

    data = load_bundle(start=str(start), end=str(end) + " 23:59:59")
    result = spec.run(data, params)
    st.json(result)

    path = save_experiment(
        strategy=spec.name,
        params=params,
        data_start=str(start),
        data_end=str(end),
        result=result,
        note=note,
    )
    st.success(f"Saved: {path}")
