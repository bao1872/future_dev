"""
Page: 标签与指标审计 · R1/R2
Task: PANJI-R3-AUDIT-UI-V1-FIX1-SIMPLIFY

Read-only audit UI. Reads the frozen R1/R2 robust oracle artifacts and the
canonical Forming-MTF environment. Performs no label modification, no
oracle re-solve, no indicator/parameter change and no model work.

This page is NOT the same oracle as pages/2_Chart.py, which uses
research.oracle_labels.oracle_labels() (hindsight oracle).

Information architecture (FIX1):
  screen 1  event summary -> single main chart (bar-relative axis) + label card
  screen 2  three tabs: 标签审计 / 指标审计 / 慢速对照
  bottom    collapsed: artifact identity gate, manual review records
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
    .block-container { padding-top: 1.0rem; padding-bottom: 1.5rem; max-width: 1800px; }
    .audit-warn {
        padding: 0.5rem 0.9rem; margin: 0 0 0.7rem 0;
        background: #33260A; border-left: 3px solid #C79A17;
        border-radius: 4px; color: #E8C77A; font-size: 0.78rem; line-height: 1.5;
    }
    .ev-sum {
        font-size: 1.22rem; font-weight: 650; letter-spacing: -0.01em;
        margin: 0 0 0.15rem 0; color: #E6EDF3;
    }
    .ev-sub { font-size: 0.86rem; color: #98A1B3; margin: 0 0 0.6rem 0; }
    .lc {
        background: #0F1922; border: 1px solid #263440; border-radius: 6px;
        padding: 0.7rem 0.85rem; font-size: 0.84rem; line-height: 1.65; color: #C7D1DB;
    }
    .lc h4 { margin: 0 0 0.35rem 0; font-size: 0.72rem; letter-spacing: 0.09em;
             color: #7C8798; text-transform: uppercase; font-weight: 600; }
    .lc .big { font-size: 1.5rem; font-weight: 700; letter-spacing: 0.02em; }
    .lc .row { display: flex; justify-content: space-between; gap: 0.6rem; }
    .lc .k { color: #8A94A6; }
    .lc .h { color: #7FE3A0; }
    .lc hr { border: 0; border-top: 1px solid #263440; margin: 0.6rem 0; }
    .ic {
        background: #0F1922; border: 1px solid #263440; border-radius: 6px;
        padding: 0.55rem 0.7rem; font-size: 0.8rem; line-height: 1.6; color: #C7D1DB;
    }
    .ic .tf { font-weight: 700; color: #E6EDF3; font-size: 0.9rem; }
    .ic .k { color: #8A94A6; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div style='font-size:1.05rem;font-weight:650;letter-spacing:-0.01em;"
    "margin:0 0 0.45rem 0'>标签与指标审计 · R1/R2</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="audit-warn">
    <b>R1/R2 ROBUST ORACLE · USED BY E1–E4</b> — 本页使用 E1–E4 的
    <b>R1/R2 Robust Oracle</b> 与 <b>Forming-MTF Environment</b>，
    不是「研究工作台」中的 <code>oracle_labels</code> hindsight Oracle。
    指标区只用 decision time 已知信息；未来路径仅用于事后标签审计。
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
# sidebar: only Symbol / Action / R2 retention  (+ advanced)                   #
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

with st.sidebar.expander("高级设置"):
    ctx_bars = st.slider("Context bars (each side)", 5, 120, 40, 5)

events = AV.filter_events(merged, action_filter, retention_min)
if len(events) == 0:
    st.warning("筛选后没有事件。请放宽 Action filter 或降低 retention 阈值。")
    st.stop()

ev_idx = events["decision_bar_index"].to_numpy(np.int64)

# Deterministic default: middle of the filtered event list (mature history).
_default_t = int(ev_idx[len(ev_idx) // 2])
if st.session_state.get("audit_prev_symbol") != symbol:
    st.session_state["audit_prev_symbol"] = symbol
    st.session_state["audit_t"] = _default_t
elif st.session_state.get("audit_t") is None or not (0 <= int(st.session_state["audit_t"]) < n):
    st.session_state["audit_t"] = _default_t


# --------------------------------------------------------------------------- #
# navigation row (moved out of the sidebar)                                    #
# --------------------------------------------------------------------------- #
def _cur_pos() -> int:
    p = int(np.searchsorted(ev_idx, int(st.session_state["audit_t"]), side="right")) - 1
    return max(0, min(p, len(ev_idx) - 1))


c_nav = st.columns([1.1, 2.0, 1.1, 1.6, 6.0])
if c_nav[0].button("◀ Previous", use_container_width=True):
    st.session_state["audit_t"] = int(ev_idx[max(0, _cur_pos() - 1)])
_pos = _cur_pos()
c_nav[1].markdown(
    f"<div style='padding-top:0.42rem;color:#98A1B3;font-size:0.86rem;text-align:center'>"
    f"Event <b style='color:#E6EDF3'>{_pos + 1:,}</b> / {len(ev_idx):,}</div>",
    unsafe_allow_html=True,
)
if c_nav[2].button("Next ▶", use_container_width=True):
    st.session_state["audit_t"] = int(ev_idx[min(len(ev_idx) - 1, _cur_pos() + 1)])
t = int(c_nav[3].number_input("bar index", min_value=0, max_value=n - 1, step=1,
                              key="audit_t", label_visibility="collapsed"))
horizon = c_nav[4].radio("Horizon", [6, 12, 24], index=2, horizontal=True,
                         label_visibility="collapsed")

# current artifact row for decision t
mpos = int(np.searchsorted(merged["decision_bar_index"].to_numpy(np.int64), t))
have_oracle_row = bool(mpos < len(merged)
                       and int(merged["decision_bar_index"].iloc[mpos]) == t)
mrow = merged.iloc[mpos] if have_oracle_row else None

ts = AV.time_semantics(builder, t)


def _f(v, nd=4):
    try:
        v = float(v)
    except Exception:
        return "n/a"
    return "n/a" if not np.isfinite(v) else f"{v:.{nd}f}"


def _g(key, default="n/a"):
    if mrow is None or key not in mrow.index:
        return default
    v = mrow[key]
    try:
        if isinstance(v, (int, float, np.integer, np.floating)) and not np.isfinite(v):
            return default
    except Exception:
        pass
    return v


# --------------------------------------------------------------------------- #
# SCREEN 1 — event summary                                                     #
# --------------------------------------------------------------------------- #
st.markdown("---")
if mrow is None:
    st.markdown(f'<div class="ev-sum">{symbol} · t={t} · 无 oracle row</div>',
                unsafe_allow_html=True)
else:
    st.markdown(
        f'<div class="ev-sum">{symbol} · t={t} · {ts["decision_time"]:%Y-%m-%d %H:%M} · '
        f'Stable={str(_g("stable_action")).upper()} · '
        f'R2 retention={_f(_g("joint_retention_stable"), 2)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="ev-sub">Decision Close {_f(ts["decision_close"])} · '
        f'Entry O(t+1) {_f(ts.get("entry_open", np.nan))} · '
        f'H6 {str(_g("action_6")).upper()} · '
        f'H12 {str(_g("action_12")).upper()} · '
        f'H24 {str(_g("action_24")).upper()}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# SCREEN 1 — main chart (bar-relative axis) + label card                       #
# --------------------------------------------------------------------------- #
col_chart, col_card = st.columns([3, 1])

with col_chart:
    win = AV.window_slice(builder, t, ctx_bars)
    rel = (win["bar_index"].to_numpy(np.int64) - t)
    times = pd.DatetimeIndex(win["bar_start_time"])
    hover = [
        f"bar {int(b_i)} ({int(r):+d})<br>{tm:%Y-%m-%d %H:%M}"
        f"<br>O {o:.4f}  H {h:.4f}<br>L {l:.4f}  C {c:.4f}"
        for b_i, r, tm, o, h, l, c in zip(
            win["bar_index"], rel, times, win["open"], win["high"],
            win["low"], win["close"])
    ]

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=rel, open=win["open"], high=win["high"], low=win["low"],
        close=win["close"], name="5m", text=hover, hoverinfo="text",
        increasing_line_color="#2EBD85", decreasing_line_color="#E8493C",
        increasing_fillcolor="#2EBD85", decreasing_fillcolor="#E8493C",
    ))
    fig.add_vrect(x0=1, x1=horizon, fillcolor="#E8493C", opacity=0.09,
                  line_width=0, layer="below")
    fig.add_vline(x=0, line_color="#FFD166", line_width=2)
    fig.add_vline(x=1, line_color="#4EA8FF", line_width=2)
    if ts["has_entry"] and np.isfinite(ts["entry_open"]):
        fig.add_shape(type="line", x0=1, x1=horizon,
                      y0=ts["entry_open"], y1=ts["entry_open"],
                      line=dict(color="#4EA8FF", width=1, dash="dot"))
    fig.update_xaxes(title_text="bars relative to decision t",
                     tickmode="linear", dtick=5, zeroline=False)
    fig.update_layout(
        height=520, margin=dict(l=10, r=10, t=28, b=10),
        xaxis_rangeslider_visible=False, showlegend=False,
        title=dict(text=f"5m price · decision t (yellow) · entry t+1 (blue) · "
                        f"future band t+1 → t+{horizon}", font=dict(size=12)),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"<span style='color:#FFD166'>▮</span> decision t={t} · "
        f"<span style='color:#4EA8FF'>▮</span> entry t+1={t + 1} "
        f"(open {_f(ts.get('entry_open', np.nan))}) · "
        f"<span style='color:#E8493C'>▮</span> future band t+1 → t+{horizon} · "
        f"hover 显示真实 bar_start_time",
        unsafe_allow_html=True,
    )

with col_card:
    if mrow is None:
        st.markdown('<div class="lc">无 oracle row</div>', unsafe_allow_html=True)
    else:
        rows_h = []
        for h in (6, 12, 24):
            rows_h.append(
                f'<div class="row"><span class="k">H{h}</span>'
                f'<span><b class="h">{str(_g(f"action_{h}")).upper()}</b></span></div>'
                f'<div class="row"><span class="k">QL / QS / QW</span>'
                f'<span>{_f(_g(f"QL_{h}"), 1)} / {_f(_g(f"QS_{h}"), 1)} / '
                f'{_f(_g(f"QW_{h}"), 1)}</span></div>'
            )
        st.markdown(
            '<div class="lc">'
            '<h4>R1 Label</h4>'
            f'<div class="k">Stable</div>'
            f'<div class="big">{str(_g("stable_action")).upper()}</div>'
            '<hr/>'
            + '<hr/>'.join(rows_h) +
            '<hr/>'
            '<h4>R2 Robustness</h4>'
            f'<div class="row"><span class="k">baseline</span>'
            f'<span>{str(_g("baseline_stable_action")).upper()}</span></div>'
            f'<div class="row"><span class="k">retention</span>'
            f'<span>{_f(_g("joint_retention_stable"), 2)}</span></div>'
            f'<div class="row"><span class="k">strict</span>'
            f'<span>{str(_g("strict_robust_action")).upper()}</span></div>'
            f'<div class="row"><span class="k">R1/R2 match</span>'
            f'<span>{"True" if str(_g("stable_action")) == str(_g("baseline_stable_action")) else "False"}'
            f'</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# SCREEN 2 — three tabs                                                        #
# --------------------------------------------------------------------------- #
tab_label, tab_ind, tab_slow = st.tabs(["标签审计", "指标审计", "慢速对照"])

# ---- Tab A: label audit ---------------------------------------------------- #
with tab_label:
    if mrow is None:
        st.info("该 decision bar 没有对应 oracle row。")
    else:
        ent = int(_g("entry_bar_index", t + 1)) if np.isfinite(_g("entry_bar_index", np.nan)) else t + 1
        st.markdown(f"**Future path（entry = {ent} → +24）**")
        st.caption("ΔLong / ΔShort 仅为视觉辅助（相对 entry open），不代表完整 QW/DP 重算。")
        fp = AV.oracle_future_path(builder, t, ent, horizon=24)
        fp = fp.assign(rel=fp["bar_index"] - ent)
        st.dataframe(
            fp[["rel", "bar_index", "bar_start_time", "open",
                "long_move_from_entry", "short_move_from_entry"]]
            .rename(columns={"rel": "offset", "long_move_from_entry": "ΔLong",
                             "short_move_from_entry": "ΔShort"}),
            hide_index=True, use_container_width=True, height=260,
        )

        st.markdown("**H6 / H12 / H24 artifact Q table**")
        st.dataframe(AV.horizon_rows(merged, mpos), hide_index=True,
                     use_container_width=True)

        with st.expander("查看原始 OHLC 明细（t-5 … t+25）"):
            pt = AV.path_table(builder, t, disc_arr(symbol), lo=-5, hi=25)
            st.dataframe(pt, hide_index=True, use_container_width=True)

# ---- Tab B: indicator audit ------------------------------------------------ #
with tab_ind:
    DTP_KEYS = ["trend_state", "dev"]
    LQ_KEYS = ["liq_up_dist_atr", "liq_down_dist_atr", "liq_last_breach_side"]
    SR_KEYS = ["sr_in_zone", "sr_support_dist_atr", "sr_resistance_dist_atr"]

    def _trend_glyph(v):
        try:
            v = float(v)
        except Exception:
            return "n/a"
        return {1.0: "↑", -1.0: "↓", 0.0: "→"}.get(v, "?")

    st.markdown("**四个 timeframe 状态摘要**（先看状态，再查细节）")
    cols = st.columns(4)
    for col, tf in zip(cols, AV.TF_LIST):
        row = env.iloc[t]
        sup = float(row[f"{tf}_sr_support_dist_atr"])
        res = float(row[f"{tf}_sr_resistance_dist_atr"])
        nearest = "sup" if abs(sup) <= abs(res) else "res"
        col.markdown(
            '<div class="ic">'
            f'<div class="tf">{tf}</div>'
            f'<div><span class="k">Trend</span> {_trend_glyph(row[f"{tf}_trend_state"])} '
            f'({_f(row[f"{tf}_trend_state"], 0)}) · '
            f'dev {_f(row[f"{tf}_dev"], 2)}</div>'
            f'<div><span class="k">SR</span> in_zone {_f(row[f"{tf}_sr_in_zone"], 0)} · '
            f'sup {_f(sup, 2)} · res {_f(res, 2)} '
            f'<span class="k">(nearest {nearest})</span></div>'
            f'<div><span class="k">Liq</span> up {_f(row[f"{tf}_liq_up_dist_atr"], 2)} · '
            f'down {_f(row[f"{tf}_liq_down_dist_atr"], 2)} · '
            f'breach_side {_f(row[f"{tf}_liq_last_breach_side"], 0)}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    tf_sel = st.radio("Timeframe", AV.TF_LIST, index=0, horizontal=True)
    row = env.iloc[t]
    DTP_ALL = [c for c in ("sma", "atr", "dev", "slope_atr", "trend_score", "trend_state")
               if c in FEATURE_COLS]
    SR_ALL = [c for c in FEATURE_COLS if c.startswith("sr_")]
    LQ_ALL = [c for c in FEATURE_COLS if c.startswith("liq_")]

    a, b, c = st.columns(3)
    for col, title, keys in ((a, "DTP", DTP_ALL), (b, "SR", SR_ALL), (c, "Liquidity", LQ_ALL)):
        with col:
            note = "（SR：diagnostic only）" if title == "SR" else ""
            st.markdown(f"**{title}** {note}")
            st.dataframe(
                pd.DataFrame([{"feature": k, "value": float(row[f"{tf_sel}_{k}"]),
                               "model_feature": k in FEATURE_COLS} for k in keys]),
                hide_index=True, use_container_width=True, height=420,
            )

# ---- Tab C: slow reference ------------------------------------------------- #
with tab_slow:
    st.caption(
        "slow reference = `builder.slow_forming_snapshot_reference(t, tf)`：从 raw<=t 重建当前 "
        "forming bar，再走 canonical batch `compute_tf_features`。"
        "PASS 规则：continuous max abs error ≤ 1e-9，NaN pattern mismatch = 0，discrete mismatch = 0。"
    )
    if st.button("运行选中 bar 的独立慢速指标对照"):
        summaries, details = [], {}
        for tf in AV.TF_LIST:
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
            details[tf] = (s, d)
        st.dataframe(pd.DataFrame(summaries), hide_index=True, use_container_width=True)

        for tf in AV.TF_LIST:
            s, d = details[tf]
            if s["passed"]:
                st.markdown(
                    f"<span style='color:#7FE3A0'>✔ {tf} PASS</span> — "
                    f"cont max abs error {s['continuous_max_abs_error']:.3e}，"
                    f"NaN mismatch {s['nan_pattern_mismatch']}，"
                    f"discrete mismatch {s['discrete_mismatch']}",
                    unsafe_allow_html=True,
                )
        fails = [tf for tf in AV.TF_LIST if not details[tf][0]["passed"]]
        if fails:
            st.error(f"FAIL: {', '.join(fails)}")
            for tf in fails:
                st.markdown(f"**{tf} 差异明细**")
                st.dataframe(details[tf][1][details[tf][1]["status"] == "MISMATCH"],
                             hide_index=True, use_container_width=True)
        with st.expander("查看完整差异（feature-by-feature）"):
            for tf in AV.TF_LIST:
                st.markdown(f"**{tf}**")
                st.dataframe(details[tf][1], hide_index=True,
                             use_container_width=True)
    else:
        st.info("点击上面的按钮运行。只针对当前选中的 bar 与 4 个 TF，不预计算全历史。")


# --------------------------------------------------------------------------- #
# bottom — collapsed                                                           #
# --------------------------------------------------------------------------- #
st.markdown("---")

with st.expander("Artifact identity gate（R1/R2 hashes, keys, duplicates）"):
    r1_rows, r2_rows = oracle_gate()
    rep = schema_report()
    g1, g2, g3 = st.columns(3)
    g1.metric("R1 rows", f"{r1_rows:,}")
    g2.metric("R2 rows", f"{r2_rows:,}")
    g3.metric("key_sets_identical", "True")
    st.caption(
        f"R1 `{rep['r1_path']}` · R2 `{rep['r2_path']}` · "
        f"`E12.load_oracle_e12()` 已校验 SHA、duplicate key、R1/R2 key 相等 · "
        f"symbol rows merged {linfo['merged_rows']:,}"
    )
    st.dataframe(pd.DataFrame({
        "group": ["H6", "H12", "H24"],
        "QL/QS/QW raw": [rep[f"H{h}_QLQSQW_present"] for h in (6, 12, 24)],
        "QL/QS/QW ATR": [rep[f"H{h}_QLQSQW_ATR_present"] for h in (6, 12, 24)],
        "action": [rep[f"H{h}_action_present"] for h in (6, 12, 24)],
        "label_available_time": [rep[f"H{h}_label_available_time_present"] for h in (6, 12, 24)],
    }), hide_index=True, use_container_width=True)
    with st.expander("R1 / R2 full column list"):
        st.write("**R1**")
        st.code("\n".join(rep["r1_columns"]))
        st.write("**R2**")
        st.code("\n".join(rep["r2_columns"]))

with st.expander("人工审计记录"):
    r1_, r2_, r3_ = st.columns(3)
    label_ok = r1_.checkbox("标签检查通过")
    indicator_ok = r2_.checkbox("指标检查通过")
    timing_ok = r3_.checkbox("时间对齐检查通过")
    notes = st.text_area("Notes", height=70)
    if st.button("Add review"):
        st.session_state.setdefault("audit_reviews", []).append({
            "symbol": symbol,
            "decision_bar_index": int(t),
            "decision_time": str(ts["decision_time"]),
            "stable_action": str(_g("stable_action")),
            "retention": float(_g("joint_retention_stable", np.nan))
            if mrow is not None else np.nan,
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

st.caption(
    "本页只展示事实（streaming 值、slow reference 值、abs error、artifact 标签），"
    "不对「标签正确 / 指标正确 / 无未来泄漏」下判断。"
)
