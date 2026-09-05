import pandas as pd
import streamlit as st

from research.experiment_store import list_experiments

st.set_page_config(page_title="Results · future_dev", layout="wide")
st.title("Results")
st.caption("Lightweight experiment records: strategy + params + data range + result.")

items = list_experiments()
if not items:
    st.info("No saved experiment results yet.")
    st.stop()

summary = []
for item in items:
    summary.append(
        {
            "experiment_id": item.get("experiment_id"),
            "created_at": item.get("created_at"),
            "strategy": item.get("strategy"),
            "data_start": item.get("data_start"),
            "data_end": item.get("data_end"),
            "git_sha": item.get("git_sha"),
            "note": item.get("note"),
        }
    )

st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

selected = st.selectbox("Inspect", [x["experiment_id"] for x in items])
item = next(x for x in items if x["experiment_id"] == selected)
st.json(item)
