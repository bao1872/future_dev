"""订单块边界局部反应实验 v3.1 —— 硬断言测试。

覆盖用户 §23 要求的 12 条。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.liquidity_state_machine.build_ob_confluence_v3 import classify
from research.liquidity_state_machine.run_ob_boundary_v3_1 import (
    RESULTS, _fit_core, boot_tau,
)

PRE = RESULTS / "boundary_preoutcome.parquet"


def _pre():
    if not PRE.exists():
        pytest.skip("boundary_preoutcome 尚未生成")
    return pd.read_parquet(PRE)


def _mk_ob(ob_id, bias, zl, zh, avail=None, inact=None, tf="5m"):
    return dict(ob_id=ob_id, symbol="AG", ob_source_tf=tf, ob_internal=False,
                ob_bias=bias, ob_available_time=avail,
                ob_inactive_time=inact, ob_inactive_reason=None,
                zone_low=zl, zone_high=zh, endpoints_swapped=False,
                ob_enter_times=[])


# ---------- 1/2. ob_margin_R 方向 ----------
def test_margin_direction_consistent_with_hit():
    d = _pre()
    left = d[d["ob_margin_R"] < 0]
    right = d[d["ob_margin_R"] >= 0]
    assert (~left["has_opposing_ob_hit"].astype(bool)).all(), \
        "margin<0 却判定为已 hit"
    assert right["has_opposing_ob_hit"].astype(bool).all(), \
        "margin>=0 却判定为未 hit"


def test_margin_equals_depth_minus_distance():
    d = _pre().dropna(subset=["ob_margin_R", "penetration_depth_R",
                              "nearest_ob_distance_R"])
    if not len(d):
        pytest.skip("无可用行")
    calc = d["penetration_depth_R"] - d["nearest_ob_distance_R"]
    assert np.allclose(d["ob_margin_R"], calc, atol=1e-6), \
        "ob_margin_R != penetration_depth_R - nearest_ob_distance_R"


# ---------- 3. 只含 CLOSE_BEYOND ----------
def test_boundary_only_close_beyond():
    d = _pre()
    assert (d["stage1"] == "CLOSE_BEYOND").all()


# ---------- 4. outcome lock ----------
def test_outcome_lock_before_gate():
    d = _pre()
    forbidden = {"later_reclaim", "stage2", "stage2_state",
                 "structural_acceptance"}
    assert not (forbidden & set(d.columns)), "outcome 字段泄漏到 preoutcome"


# ---------- 5. primary bandwidth 依据 pre-outcome ----------
def test_primary_bandwidth_from_preoutcome_support():
    p = RESULTS / "primary_bandwidth_decision.json"
    if not p.exists():
        pytest.skip("未生成决策文件")
    j = json.load(open(p))
    assert j["status"] in ("OK", "LOCAL_SUPPORT_FAIL")
    if j["status"] == "OK":
        s = pd.read_csv(RESULTS / "bandwidth_support.csv")
        row = s[s["bandwidth"] == j["primary_bandwidth"]].iloc[0]
        assert bool(row["support_ok"])
        # 必须是满足条件的**最小**窗口
        smaller = s[s["bandwidth"] < j["primary_bandwidth"]]
        assert (~smaller["support_ok"]).all(), \
            "primary 不是从小到大第一个满足支持度的窗口"
        # 决策文件不得包含任何 outcome 字段
        assert not ({"tau", "later_reclaim", "p_left"} & set(j))


# ---------- 6/7. canonical trading_day ----------
def test_trading_day_canonical_and_block_bootstrap():
    d = _pre()
    assert "trading_day" in d.columns
    assert d["trading_day"].notna().all()
    g = d[d["ob_margin_R"].abs() <= 0.5].dropna(subset=["trading_day"])
    if len(g) < 50 or "later_reclaim" not in g.columns:
        # 无 outcome 时只验证分块结构
        days = g["trading_day"].unique()
        assert len(days) > 10
        return
    rng = np.random.default_rng(0)
    lo, hi = boot_tau(g, 0.5, g["trading_day"].unique(), rng)
    assert np.isfinite(lo) and np.isfinite(hi) and hi > lo, \
        "block bootstrap 未按 trading_day 整块重拟合"


# ---------- 8. 4h 仅环境 ----------
def test_4h_is_environment_only():
    d = _pre()
    cols = [c for c in d.columns if "4h" in c]
    assert cols, "缺少 4h 字段"
    for c in cols:
        assert "env" in c or c == "env4h_vs_1h" or "tuple" in c, \
            f"{c} 不是 4h 环境字段"


# ---------- 9. future OB 不进入 nearest OB ----------
def test_future_ob_excluded():
    t0 = pd.Timestamp("2025-01-10 10:00:00")
    past = _mk_ob("PAST", -1, 101.0, 102.0, avail=pd.Timestamp("2025-01-01"))
    future = _mk_ob("FUTURE", -1, 100.2, 100.8,
                    avail=pd.Timestamp("2025-01-20"))
    obm = pd.DataFrame([future, past])
    # 向上扫到 100.5：未来 OB 更近，但必须被排除
    r = classify(100.0, +1, 100.5, 100.4, t0, obm, 1.0)
    assert r["nearest_ob_id"] == "PAST", "future OB 进入了 nearest OB"


# ---------- 10. inactive OB 不进入 nearest OB ----------
def test_inactive_ob_excluded():
    t0 = pd.Timestamp("2025-01-10 10:00:00")
    live = _mk_ob("LIVE", -1, 101.0, 102.0, avail=pd.Timestamp("2025-01-01"))
    dead = _mk_ob("DEAD", -1, 100.2, 100.8,
                  avail=pd.Timestamp("2025-01-01"),
                  inact=pd.Timestamp("2025-01-05"))
    obm = pd.DataFrame([dead, live])
    r = classify(100.0, +1, 100.5, 100.4, t0, obm, 1.0)
    assert r["nearest_ob_id"] == "LIVE", "inactive OB 进入了 nearest OB"


# ---------- 11. 同一 interaction 只出现一次 ----------
def test_interaction_unique():
    d = _pre()
    assert d["level_key"].is_unique
    assert d["level_key"].notna().all()


# ---------- 12. bootstrap 每次重新拟合 ----------
def test_bootstrap_refits_each_draw():
    rng = np.random.default_rng(1)
    n = 400
    x = np.concatenate([rng.uniform(-0.5, 0, n // 2),
                        rng.uniform(0, 0.5, n // 2)])
    y = (rng.random(n) < 0.8).astype(float)
    days = np.array([f"d{i//20}" for i in range(n)])
    g = pd.DataFrame({"ob_margin_R": x, "later_reclaim": y,
                      "trading_day": days})
    lo, hi = boot_tau(g, 0.5, np.unique(days), rng)
    assert np.isfinite(lo) and np.isfinite(hi) and (hi - lo) > 1e-6, \
        "bootstrap 未产生重拟合分布"
    # 局部线性模型本身可用
    r = _fit_core(x, y, 0.5)
    assert r is not None and np.isfinite(r["tau"])
