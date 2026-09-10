"""订单块边界实验 v3.2 —— 硬断言测试（用户 §22 的 14 条）。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from research.liquidity_state_machine.build_boundary_preoutcome_v3_2 import (
    RESULTS, build_trend_series, boundary_state,
)
from research.liquidity_state_machine.run_ob_boundary_v3_2 import (
    band, fit_local, subgroup_gate,
)

PRE = RESULTS / "boundary_preoutcome_v3_2.parquet"


def _pre():
    if not PRE.exists():
        pytest.skip("preoutcome v3.2 尚未生成")
    return pd.read_parquet(PRE)


# ---------- 1/2/3. 三周期趋势字段真实存在且有效 ----------
def test_trend_field_present_and_valid():
    d = _pre()
    for col in ("trend_struct_5m", "trend_struct_15m", "trend_struct_1h"):
        assert col in d.columns, f"{col} 缺失"
        v = pd.to_numeric(d[col], errors="coerce")
        share = float(v.isin([-1, 0, 1]).mean())
        assert share >= 0.95, f"{col} 有效率仅 {share:.3f}"
        assert (v == 1).sum() > 0 and (v == -1).sum() > 0, \
            f"{col} 全为单向值"


def test_trend_series_builder_names():
    """build_trend_series 必须产出带周期后缀的字段，而非恒为 1h。"""
    from research.export_ob_trigger_execution_v21 import load_raw_5m
    five = load_raw_5m("AG").sort_values("bar_start_time").reset_index(
        drop=True)
    five["volume"] = five["trade"].astype(float)
    t5 = build_trend_series(five, "5m")
    assert "trend_struct_5m" in t5.columns
    assert "trend_struct_1h" not in t5.columns
    assert "available_time" in t5.columns
    assert set(pd.to_numeric(t5["trend_struct_5m"]).unique()) <= {-1, 0, 1}


# ---------- 4. 三周期 available_time 因果正确 ----------
def test_trend_available_time_causal():
    d = _pre()
    pb = pd.to_datetime(d["penetration_bar_start"])
    for tag in ("5m", "15m", "1h"):
        col = f"trend_avail_{tag}"
        if col not in d.columns:
            continue
        s = pd.to_datetime(d[col]).dropna()
        assert bool((s <= pb[s.index]).all()), f"{tag} 趋势晚于 penetration"


# ---------- 5/6/7. 三态语义 ----------
def test_boundary_state_trichotomy():
    # margin < 0 -> NEAR_MISS
    st, m = boundary_state(+1, 100.0, 100.5, 1.0)
    assert st == "NEAR_MISS" and m < 0
    st, m = boundary_state(-1, 100.5, 100.0, 1.0)
    assert st == "NEAR_MISS" and m < 0
    # 严格价格相等 -> EDGE_TOUCH（不引入 ATR 阈值）
    st, m = boundary_state(+1, 100.5, 100.5, 1.0)
    assert st == "EDGE_TOUCH" and m == 0
    st, m = boundary_state(-1, 100.0, 100.0, 1.0)
    assert st == "EDGE_TOUCH" and m == 0
    # margin > 0 -> ENTERED_OB
    st, m = boundary_state(+1, 101.0, 100.5, 1.0)
    assert st == "ENTERED_OB" and m > 0
    st, m = boundary_state(-1, 99.5, 100.0, 1.0)
    assert st == "ENTERED_OB" and m > 0


def test_state_margin_consistency_on_data():
    d = _pre()
    for state, cond in (("NEAR_MISS", lambda v: v < 0),
                        ("EDGE_TOUCH", lambda v: v == 0),
                        ("ENTERED_OB", lambda v: v > 0)):
        sub = d.loc[d["ob_boundary_state"] == state, "ob_margin_R"].dropna()
        assert bool(cond(sub).all()), f"{state} margin 符号不一致"


# ---------- 8/9. EDGE_TOUCH 不进入模型与支持数 ----------
def test_edge_touch_excluded():
    d = _pre().dropna(subset=["ob_margin_R"])
    g = band(d, 0.3)
    assert "EDGE_TOUCH" not in set(g["ob_boundary_state"]), \
        "EDGE_TOUCH 进入了局部模型样本"
    sup = pd.read_csv(RESULTS / "bandwidth_support.csv")
    row = sup[sup.bandwidth == 0.3].iloc[0]
    primary = band(d, float(row.bandwidth))
    assert int(row.n_left) == int(
        (primary["ob_boundary_state"] == "NEAR_MISS").sum())
    assert int(row.n_right) == int(
        (primary["ob_boundary_state"] == "ENTERED_OB").sum())
    assert int(row.n_left) + int(row.n_right) == int(row.n), \
        "左右支持数包含 EDGE_TOUCH"


# ---------- 10. 1h×15m 有有效样本 ----------
def test_1h_vs_15m_has_valid_samples():
    rc = pd.read_csv(RESULTS / "trend_relation_counts.csv")
    r = rc[(rc.relation == "trend_1h_vs_15m")
           & (rc.value.isin(["WITH_TREND", "AGAINST_TREND"]))]
    assert int(r["n"].sum()) > 0, "TREND_CONTEXT_BUILD_FAIL"


# ---------- 11. subgroup infer 必须同时要求 continuity gate ----------
def test_subgroup_infer_requires_gate():
    p = RESULTS / "boundary_subgroups.csv"
    if not p.exists():
        pytest.skip("subgroup 结果尚未生成")
    s = pd.read_csv(p)
    if "infer" not in s.columns:
        pytest.skip("无 infer 列")
    ok = s["infer"].astype(bool)
    if not ok.any():
        pytest.skip("无合格 subgroup")
    q = s[ok]
    assert bool(q["subgroup_gate"].astype(bool).all()), \
        "存在未过 gate 却被判可推断的 subgroup"
    assert bool(q["support_pass"].astype(bool).all()), \
        "存在支持度不足却被判可推断的 subgroup"


# ---------- 12. preoutcome 不含未来 outcome ----------
def test_preoutcome_no_outcome():
    d = _pre()
    forbidden = {"later_reclaim", "stage2", "stage2_state",
                 "structural_acceptance"}
    assert not (forbidden & set(d.columns))


# ---------- 13. canonical trading_day bootstrap ----------
def test_canonical_trading_day():
    d = _pre()
    assert "trading_day" in d.columns
    assert d["trading_day"].notna().all()
    assert d["trading_day"].nunique() > 50, "trading_day 分块过少"


# ---------- 14. 4h 仍只有环境方向 ----------
def test_4h_environment_only():
    d = _pre()
    cols = [c for c in d.columns if "4h" in c]
    assert cols
    for c in cols:
        assert ("env" in c or c == "env4h_vs_1h" or "tuple" in c), \
            f"{c} 不是 4h 环境字段"
