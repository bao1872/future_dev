from __future__ import annotations

import json

import streamlit as st

from market_data.offline_store import load_bundle
from research.experiment_store import save_experiment
from strategies.registry import all_strategies

st.set_page_config(page_title="Strategy Lab · future_dev", layout="wide")
st.title("Strategy Lab")
st.caption("Offline strategy research only. Strategies do not access TqSdk.")

registry = all_strategies()
if not registry:
    st.info(
        "No strategy is registered yet. This is intentional: define a concrete hypothesis, "
        "copy `strategies/template_strategy.py`, then register it in `strategies/registry.py`."
    )
    st.stop()

name = st.selectbox("Strategy", list(registry))
spec = registry[name]
st.write(spec.description)

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
start = c1.date_input("Start", value=common_start.date(), min_value=common_start.date(), max_value=common_end.date())
end = c2.date_input("End", value=common_end.date(), min_value=common_start.date(), max_value=common_end.date())

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
