from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from research.experiment_store import list_experiments

st.title("实验对比")
st.caption("轻量实验记录：strategy + params + data range + result。")

items = list_experiments()
if not items:
    st.info("No saved experiment results yet.")
    st.stop()

summary = [
    {
        "experiment_id": item.get("experiment_id"),
        "created_at": item.get("created_at"),
        "strategy": item.get("strategy"),
        "data_start": item.get("data_start"),
        "data_end": item.get("data_end"),
        "git_sha": item.get("git_sha"),
        "note": item.get("note"),
    }
    for item in items
]

st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

st.markdown("### 指标对比")
ids = [x["experiment_id"] for x in items]
selected = st.multiselect("选择 1–3 个实验进行比较", ids, default=ids[:1], max_selections=3)

if not selected:
    st.info("选择至少一个实验。")
    st.stop()

if len(selected) > 3:
    st.warning("最多比较 3 个实验。")
    st.stop()

chosen = [next(x for x in items if x["experiment_id"] == sid) for sid in selected]


def flatten_metrics(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested metric dicts into flat columns. Never invents fields."""
    out: dict[str, Any] = {}
    if not isinstance(value, dict):
        return {prefix.rstrip(".") or "value": value}
    for key, val in value.items():
        path = f"{prefix}{key}"
        if isinstance(val, dict):
            out.update(flatten_metrics(val, f"{path}."))
        else:
            out[path] = val
    return out


metric_rows = {}
for item in chosen:
    metrics = (item.get("result") or {}).get("metrics")
    if metrics is None:
        metric_rows[item["experiment_id"]] = {"_metrics": "result 中无 metrics 字段"}
    else:
        metric_rows[item["experiment_id"]] = flatten_metrics(metrics)

st.dataframe(
    pd.DataFrame(metric_rows).rename_axis("metric").reset_index(),
    use_container_width=True,
    hide_index=True,
)

st.markdown("### 单个实验详情")
detail_id = st.selectbox("实验", selected)
detail = next(x for x in chosen if x["experiment_id"] == detail_id)

m1, m2, m3 = st.columns(3)
m1.metric("Strategy", str(detail.get("strategy")))
m2.metric("Git SHA", str(detail.get("git_sha"))[:12])
m3.metric("Data range", f"{detail.get('data_start')} → {detail.get('data_end')}")

st.json(detail.get("params"), expanded=False)
st.json(detail.get("result"))
if detail.get("note"):
    st.caption(f"Note: {detail['note']}")
