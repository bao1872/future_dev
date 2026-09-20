"""
Page: 标签与指标审计 · R1/R2
Task: PANJI-R3-AUDIT-UI-V1-LABEL-INDICATOR

Read-only audit UI. Reads the frozen R1/R2 robust oracle artifacts and the
canonical Forming-MTF environment. Performs no label modification, no
oracle re-solve, no indicator/parameter change and no model work.

This page is NOT the same oracle as pages/2_Chart.py, which uses
research.oracle_labels.oracle_labels() (hindsight oracle).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import research.liquidity_oracle_atlas.audit_view_v1 as AV  # noqa: E402
import research.liquidity_oracle_atlas.experiment_vol_normalization_falsification_v1 as E12  # noqa: E402
from research.liquidity_oracle_atlas.build_forming_environment_v1 import (  # noqa: E402
    FormingEnvironmentBuilder,
)
from research.liquidity_oracle_atlas.forming_indicator_state_v1 import (  # noqa: E402
    DISCRETE_COLS,
    FEATURE_COLS,
)

st.set_page_config(page_title="标签与指标审计 · R1/R2", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1800px; }
    .audit-warn {
        padding: 0.7rem 1rem; margin: 0.2rem 0 0.9rem 0;
        background: #3A2A00; border: 1px solid #8A6D00;
        border-radius: 6px; color: #FFD98A; font-size: 0.86rem; line-height: 1.55;
    }
    .audit-chain {
        padding: 0.55rem 0.9rem; margin: 0.2rem 0 0.8rem 0;
        background: #111A23; border: 1px solid #263440; border-radius: 6px;
        font-size: 0.82rem; color: #98A1B3;
    }
    .chip {
        display: inline-block; padding: 0.1rem 0.5rem; border-radius: 4px;
        font-size: 0.75rem; font-weight: 600; margin-right: 0.4rem;
    }
    .chip-known { background: #12351F; color: #7FE3A0; border: 1px solid #1E6B3A; }
    .chip-future { background: #3A1A22; color: #FF9CB0; border: 1px solid #7A2634; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("标签与指标审计 · R1/R2")
st.markdown(
    """
    <div class="audit-warn">
    <b>R1/R2 ROBUST ORACLE · USED BY E1–E4</b><br/>
    本页使用 E1–E4 实验中的 <b>R1/R2 Robust Oracle</b> 和 <b>Forming-MTF Environment</b>。<br/>
    它不是「研究工作台」中的 <code>oracle_labels</code> hindsight Oracle。<br/>
    左侧指标区只允许使用 decision time 已知信息；
    右侧标签未来路径专门用于事后标签审计。
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# cached frozen computations                                                   #
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Preparing canonical forming environment...")
def get_builder(symbol: str):
    b = FormingEnvironmentBuilder(symbol)
    b.load_raw()
    b.prepare()
    return b


@st.cache_data(show_spinner="Building canonical OLD96 environment...")
def get_environment(symbol: str):
    b = get_builder(symbol)
    env, audit = b.run(profile_memory=False)
    env = env.copy()
    if "symbol" not in env.columns:
        env["symbol"] = env["data_object"]
    return env, audit


@st.cache_data(show_spinner="Running independent slow reference...")
def slow_reference(symbol: str, i: int, tf: str):
    b = get_builder(symbol)
    return b.slow_forming_snapshot_reference(int(i), tf)


@st.cache_data(show_spinner="Verifying R1/R2 artifact identity...")
def oracle_gate():
    r1, r2 = E12.load_oracle_e12()
    return int(len(r1)), int(len(r2))


@st.cache_data(show_spinner="Loading R1/R2 rows for symbol...")
def oracle_rows(symbol: str):
    return AV.load_symbol_oracle(symbol)


@st.cache_data(show_spinner="Reading disc continuity flags...")
def disc_arr(symbol: str):
    return AV.disc_flags(symbol)


@st.cache_data
def schema_report():
    return AV.oracle_schema_report()


# --------------------------------------------------------------------------- #
# artifact identity gate                                                       #
# --------------------------------------------------------------------------- #
with st.expander("Artifact identity gate (R1 / R2 hashes, keys, duplicates)", expanded=False):
    r1_rows, r2_rows = oracle_gate()
    rep = schema_report()
    c1, c2, c3 = st.columns(3)
    c1.metric("R1 rows", f"{r1_rows:,}")
    c2.metric("R2 rows", f"{r2_rows:,}")
    c3.metric("key_sets_identical", "True")
    st.caption(
        f"R1 `{rep['r1_path']}` · R2 `{rep['r2_path']}` "
        f"（`E12.load_oracle_e12()` 已校验 SHA、duplicate key、R1/R2 key 相等）"
    )

    sch = pd.DataFrame({
        "group": ["H6", "H12", "H24"],
        "QL/QS/QW raw": [rep[f"H{h}_QLQSQW_present"] for h in (6, 12, 24)],
        "QL/QS/QW ATR": [rep[f"H{h}_QLQSQW_ATR_present"] for h in (6, 12, 24)],
        "action": [rep[f"H{h}_action_present"] for h in (6, 12, 24)],
        "label_available_time": [rep[f"H{h}_label_available_time_present"] for h in (6, 12, 24)],
    })
    st.markdown("**实际发现的 per-horizon 列**")
    st.dataframe(sch, hide_index=True, use_container_width=True)
    st.caption(
        f"R1 `stable_action` = {rep['r1_stable_action_present']} · "
        f"R2 `baseline_stable_action` = {rep['r2_baseline_stable_action_present']} · "
        f"R2 `joint_retention_stable` = {rep['r2_joint_retention_stable_present']} · "
        f"R2 `strict_robust_action` = {rep['r2_strict_robust_action_present']}"
    )
    with st.expander("R1 / R2 full column list"):
        st.write("**R1**")
        st.code("\n".join(rep["r1_columns"]))
        st.write("**R2**")
        st.code("\n".join(rep["r2_columns"]))


# --------------------------------------------------------------------------- #
# sidebar                                                                      #
# --------------------------------------------------------------------------- #
st.sidebar.header("Audit selection")
symbol = st.sidebar.selectbox("Symbol", AV.SYMBOLS, index=0, key="audit_symbol")

builder = get_builder(symbol)
env, audit = get_environment(symbol)
merged, linfo = oracle_rows(symbol)
n = builder.n

vocab = AV.action_vocabulary(merged)
action_filter = st.sidebar.selectbox("Action filter", ["All"] + vocab, index=0)
retention_min = st.sidebar.slider("R2 joint retention ≥", 0.0, 1.0, 0.0, 0.05)
ctx_bars = st.sidebar.slider("Context bars (each side)", 5, 120, 30, 5)

events = AV.filter_events(merged, action_filter, retention_min)
if len(events) == 0:
    st.warning("筛选后没有事件。请放宽 Action filter 或降低 retention 阈值。")
    st.stop()

ev_idx = events["decision_bar_index"].to_numpy(np.int64)

# Deterministic default: middle of the filtered event list (mature history),
# so the first view is never the warmup region. Reproducible, never random.
_default_t = int(ev_idx[len(ev_idx) // 2])
if st.session_state.get("audit_prev_symbol") != symbol:
    st.session_state["audit_prev_symbol"] = symbol
    st.session_state["audit_t"] = _default_t
elif st.session_state.get("audit_t") is None or not (0 <= int(st.session_state["audit_t"]) < n):
    st.session_state["audit_t"] = _default_t

pos = int(np.searchsorted(ev_idx, int(st.session_state["audit_t"]), side="right")) - 1
pos = max(0, min(pos, len(ev_idx) - 1))

nav1, nav2 = st.sidebar.columns(2)
if nav1.button("◀ Previous", use_container_width=True):
    p = max(0, pos - 1)
    st.session_state["audit_t"] = int(ev_idx[p])
if nav2.button("Next ▶", use_container_width=True):
    p = min(len(ev_idx) - 1, pos + 1)
    st.session_state["audit_t"] = int(ev_idx[p])

t = int(st.sidebar.number_input(
    "Decision bar index", min_value=0, max_value=n - 1, step=1, key="audit_t"))

st.sidebar.caption(
    f"filtered events = {len(events):,} / {len(merged):,} · "
    f"position {pos + 1:,} · env rows {len(env):,}"
)

# current artifact row for decision t
mpos = int(np.searchsorted(merged["decision_bar_index"].to_numpy(np.int64), t))
have_oracle_row = bool(mpos < len(merged)
                       and int(merged["decision_bar_index"].iloc[mpos]) == t)
mrow = merged.iloc[mpos] if have_oracle_row else None

ts = AV.time_semantics(builder, t)


# --------------------------------------------------------------------------- #
# 1. time semantics                                                            #
# --------------------------------------------------------------------------- #
st.markdown("---")
st.subheader("1 · 时间语义")
st.markdown(
    '<div class="audit-chain">Decision = C<sub>t</sub>（decision bar 的 close） &nbsp;→&nbsp; '
    'Entry = O<sub>t+1</sub>（下一根 5m bar 的 open）。两者全部来自 canonical base frame。</div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Decision t", f"{ts['t']}")
m1.caption(f"{ts['bar_start_time']} → decision_time {ts['decision_time']}")
m2.metric("Entry t+1", f"{ts.get('entry_bar_index', 'n/a')}")
m2.caption(f"{ts.get('entry_bar_start_time', 'n/a')} · open {ts.get('entry_open', float('nan')):.4f}"
           if ts["has_entry"] else "no next bar")
action_now = str(mrow["stable_action"]) if mrow is not None else "n/a (no oracle row)"
m3.metric("Stable Action (R1)", action_now)
m3.caption(f"baseline (R2) = {str(mrow['baseline_stable_action']) if mrow is not None else 'n/a'}")
m4.metric("Decision close C_t", f"{ts['decision_close']:.4f}")
m4.caption(f"segment {ts['segment']}")

tbl_ts = pd.DataFrame([{
    "symbol": symbol,
    "decision_bar_index": t,
    "5m bar start": ts["bar_start_time"],
    "decision_time (= start + 5min)": ts["decision_time"],
    "decision_close C_t": ts["decision_close"],
    "entry bar index": ts.get("entry_bar_index", np.nan),
    "entry bar start": ts.get("entry_bar_start_time", pd.NaT),
    "entry open O_(t+1)": ts.get("entry_open", np.nan),
}])
st.dataframe(tbl_ts, hide_index=True, use_container_width=True)

st.markdown("**时间对齐 checks（R1/R2 artifact ↔ canonical base frame）**")
if mrow is not None:
    atc = AV.artifact_timing_checks(builder, t, mrow)
    st.dataframe(pd.DataFrame([{"check": k, "ok": bool(v)} for k, v in atc.items()]),
                 hide_index=True, use_container_width=True)
else:
    st.caption("无 oracle row，跳过 artifact 时间对齐检查。")


# --------------------------------------------------------------------------- #
# 2. raw 5m path                                                               #
# --------------------------------------------------------------------------- #
st.markdown("---")
st.subheader("2 · 5m 原始价格路径")
st.markdown(
    '<span class="chip chip-known">KNOWN AT DECISION ( -∞ , t ]</span>'
    '<span class="chip chip-future">LABEL FUTURE [ t+1 , t+25 ]</span>'
    '<span style="color:#98A1B3;font-size:0.8rem;">两块区域严格分开：左侧指标只允许用 KNOWN，'
    '右侧未来路径仅供事后标签审计。</span>',
    unsafe_allow_html=True,
)

win = AV.window_slice(builder, t, ctx_bars)
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=win["bar_start_time"], open=win["open"], high=win["high"],
    low=win["low"], close=win["close"], name="5m",
    increasing_line_color="#2EBD85", decreasing_line_color="#E8493C",
))
for j, colr, dash in ((t, "#FFD166", "solid"), (t + 1, "#4EA8FF", "solid")):
    if 0 <= j < n:
        fig.add_vline(x=builder.base["time"].iloc[j], line_color=colr,
                      line_dash=dash, line_width=2)
for h, colr in ((6, "#A78BFA"), (12, "#F472B6"), (24, "#FB923C")):
    if t + h < n:
        fig.add_vline(x=builder.base["time"].iloc[t + h], line_color=colr,
                      line_dash="dot", line_width=1.2,
                      annotation_text=f"H{h}", annotation_position="top")
if t + 1 < n:
    fig.add_vrect(x0=builder.base["time"].iloc[t + 1],
                  x1=builder.base["time"].iloc[min(n - 1, t + 25)],
                  fillcolor="#E8493C", opacity=0.07, line_width=0)
fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10),
                  xaxis_rangeslider_visible=False, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.markdown("**原始 path table（t-5 … t+25）**")
pt = AV.path_table(builder, t, disc_arr(symbol), lo=-5, hi=25)
st.dataframe(
    pt.style.apply(
        lambda r: ["background-color: #17301F" if r["phase"] == "KNOWN"
                   else ("background-color: #1B2A3A" if r["phase"] == "ENTRY"
                         else "background-color: #331A20")] * len(r), axis=1),
    hide_index=True, use_container_width=True,
)


# --------------------------------------------------------------------------- #
# 3. R1 label inspector                                                        #
# --------------------------------------------------------------------------- #
st.markdown("---")
st.subheader("3 · R1 Label Inspector（直接展开 artifact，不重算 Oracle）")

if mrow is None:
    st.warning("该 decision bar 没有对应 oracle row。")
else:
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("R1 stable_action", str(mrow["stable_action"]))
    a2.metric("R2 baseline_stable_action", str(mrow["baseline_stable_action"]))
    a3.metric("R2 joint_retention_stable", f"{float(mrow['joint_retention_stable']):.3f}")
    match = str(mrow["stable_action"]) == str(mrow["baseline_stable_action"])
    a4.metric("R1/R2 action match", "True" if match else "False")
    if not match:
        st.warning("stable_action != baseline_stable_action（仅展示事实，不修改结果）")

    extra = {
        "sym": symbol, "t": t,
        "entry_bar_index (R1)": mrow.get("entry_bar_index", np.nan),
        "entry_valid": mrow.get("entry_valid", np.nan),
        "agreement": mrow.get("agreement", np.nan),
        "_any_tie": mrow.get("_any_tie", np.nan),
        "_agree23": mrow.get("_agree23", np.nan),
        "atr5_t": mrow.get("atr5_t", np.nan),
        "joint_retention": mrow.get("joint_retention", np.nan),
        "joint_opposite_flip_rate": mrow.get("joint_opposite_flip_rate", np.nan),
        "strict_robust_action": mrow.get("strict_robust_action", np.nan),
    }
    st.dataframe(pd.DataFrame([extra]), hide_index=True, use_container_width=True)

    st.markdown("**Horizon 表（H = 6 / 12 / 24，仅显示 artifact 实际存在的列）**")
    st.dataframe(AV.horizon_rows(merged, mpos), hide_index=True, use_container_width=True)

    # ---- oracle future path (visual aid only) ----
    ent = int(mrow["entry_bar_index"]) if np.isfinite(mrow.get("entry_bar_index", np.nan)) else t + 1
    st.markdown(f"**Oracle 未来路径（entry = {ent}，O_e … O_(e+24)）**")
    st.caption(
        "这些列只用于人工检查未来价格路径，不代表完整 QW/DP 重新计算；"
        "本页不重新实现 Wait DP。"
    )
    fp = AV.oracle_future_path(builder, t, ent, horizon=24)
    st.dataframe(fp, hide_index=True, use_container_width=True)


# --------------------------------------------------------------------------- #
# 4. forming-MTF source inspector                                              #
# --------------------------------------------------------------------------- #
st.markdown("---")
st.subheader("4 · Forming-MTF Source Inspector（防未来泄漏核心）")
st.caption(
    "对每个 TF，展示 decision t 时刻构建中的 bucket：起点、已有 base bar 数、forming OHLC "
    "与 maturity 诊断（maturity 不是模型 feature）。"
)

tabs = st.tabs(AV.TF_LIST)
forming_ok = True
for tab, tf in zip(tabs, AV.TF_LIST):
    with tab:
        fi = AV.forming_info(builder, t, tf)
        chk = AV.forming_checks(builder, t, tf)
        forming_ok = forming_ok and all(chk.values())

        c = st.columns(4)
        c[0].metric("bucket_start", str(fi["bucket_start"]))
        c[1].metric("start_idx → end_idx", f"{fi['start_idx']} → {fi['end_idx']}")
        c[2].metric("n_base_known / expected", f"{fi['n_base_known']} / {fi['expected_bars']}")
        c[3].metric("maturity (diagnostic)", f"{fi['maturity']:.2f}")

        st.dataframe(pd.DataFrame([{
            "forming open": fi["forming_open"], "forming high": fi["forming_high"],
            "forming low": fi["forming_low"], "forming close": fi["forming_close"],
        }]), hide_index=True, use_container_width=True)

        st.markdown("**Checks**")
        st.dataframe(pd.DataFrame(
            [{"check": k, "ok": bool(v)} for k, v in chk.items()]),
            hide_index=True, use_container_width=True)


# --------------------------------------------------------------------------- #
# 5. indicator inspector                                                       #
# --------------------------------------------------------------------------- #
st.markdown("---")
st.subheader("5 · Indicator Inspector（streaming, canonical OLD96）")
st.caption(
    "DTP: dev = (C - SMA) / ATR ； slope_atr = (SMA_t - SMA_(t-lag)) / ATR 。"
    "price 类字段为 diagnostic only，不一定是模型 feature。"
)

LQ_CANON = [c for c in FEATURE_COLS if c.startswith("liq_")]
DTP_CANON = [c for c in FEATURE_COLS if c in ("sma", "atr", "dev", "slope_atr",
                                              "trend_score", "trend_state")]
SR_CANON = [c for c in FEATURE_COLS if c.startswith("sr_")]

itabs = st.tabs(AV.TF_LIST)
for tab, tf in zip(itabs, AV.TF_LIST):
    with tab:
        row = env.iloc[t]
        b1, b2, b3 = st.columns([1, 2, 2])
        with b1:
            st.markdown("**DTP**")
            st.dataframe(pd.DataFrame(
                [{"feature": c, "value": float(row[f"{tf}_{c}"])} for c in DTP_CANON]),
                hide_index=True, use_container_width=True)
        with b2:
            st.markdown("**SR**（diagnostic only / not necessarily model feature）")
            st.dataframe(pd.DataFrame(
                [{"feature": c, "value": float(row[f"{tf}_{c}"])} for c in SR_CANON]),
                hide_index=True, use_container_width=True)
        with b3:
            st.markdown("**Liquidity**（canonical FEATURE_COLS ∩ `liq_*`）")
            st.dataframe(pd.DataFrame(
                [{"feature": c, "value": float(row[f"{tf}_{c}"])} for c in LQ_CANON]),
                hide_index=True, use_container_width=True)


# --------------------------------------------------------------------------- #
# 6. slow reference check                                                      #
# --------------------------------------------------------------------------- #
st.markdown("---")
st.subheader("6 · Slow Reference Check（streaming vs 独立慢速实现）")
st.caption(
    "slow reference = `builder.slow_forming_snapshot_reference(t, tf)`："
    "从 raw<=t 重建当前 forming bar，再走 canonical batch `compute_tf_features`。"
    "PASS 规则：continuous max abs error <= 1e-9，NaN pattern mismatch = 0，discrete mismatch = 0。"
)

run_slow = st.button("运行选中 bar 的独立慢速指标对照")

if run_slow:
    summaries = []
    details = {}
    for tf in AV.TF_LIST:
        if not have_oracle_row:
            continue
        stream = AV.streaming_row(env, t, tf, FEATURE_COLS)
        slow = slow_reference(symbol, t, tf)
        s, d = AV.diff_stream_vs_slow(stream, slow, FEATURE_COLS, DISCRETE_COLS)
        summaries.append({
            "TF": tf,
            "continuous max abs error": s["continuous_max_abs_error"],
            "NaN pattern mismatch": s["nan_pattern_mismatch"],
            "discrete mismatch": s["discrete_mismatch"],
            "mismatch rows": s["mismatch_rows"],
            "PASS/FAIL": "PASS" if s["passed"] else "FAIL",
        })
        details[tf] = d
    summary_df = pd.DataFrame(summaries)
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    for tf in AV.TF_LIST:
        d = details.get(tf)
        if d is None:
            continue
        bad = d[d["status"] == "MISMATCH"]
        if len(bad):
            st.error(f"{tf}: {len(bad)} 条 mismatch")
            st.dataframe(bad, hide_index=True, use_container_width=True)
        with st.expander(f"{tf}: {'All FEATURE_COLS matched' if not len(bad) else str(len(bad)) + ' mismatch'}"
                         f"（展开查看完整值）"):
            st.dataframe(d, hide_index=True, use_container_width=True)
else:
    st.info("点击上面的按钮运行。该计算只针对当前选中的 bar 与 4 个 TF，不会预计算全历史。")


# --------------------------------------------------------------------------- #
# 7. manual review records                                                     #
# --------------------------------------------------------------------------- #
st.markdown("---")
st.subheader("7 · 人工审计记录（仅 session_state，不写数据库/artifact）")

r1_, r2_, r3_ = st.columns(3)
label_ok = r1_.checkbox("标签检查通过")
indicator_ok = r2_.checkbox("指标检查通过")
timing_ok = r3_.checkbox("时间对齐检查通过")
notes = st.text_area("Notes", height=80)

if st.button("Add review"):
    st.session_state.setdefault("audit_reviews", []).append({
        "symbol": symbol,
        "decision_bar_index": int(t),
        "decision_time": str(ts["decision_time"]),
        "stable_action": action_now,
        "retention": float(mrow["joint_retention_stable"]) if mrow is not None else np.nan,
        "label_ok": bool(label_ok),
        "indicator_ok": bool(indicator_ok),
        "timing_ok": bool(timing_ok),
        "notes": notes,
    })
    st.success("recorded")

reviews = st.session_state.get("audit_reviews", [])
if reviews:
    rdf = pd.DataFrame(reviews)
    st.dataframe(rdf, hide_index=True, use_container_width=True)
    st.download_button(
        "下载本次人工审计 CSV",
        rdf.to_csv(index=False).encode("utf-8"),
        file_name=f"label_indicator_audit_{symbol}.csv",
        mime="text/csv",
    )
else:
    st.caption("尚无记录。")

st.markdown("---")
st.caption(
    "证据输出约定：本页只展示事实（streaming 值、slow reference 值、abs error、artifact 标签），"
    "不对「标签正确 / 指标正确 / 无未来泄漏」下判断。"
)
