"""SMC Oracle Atlas v1.1 —— 数据语义修复硬断言测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.liquidity_oracle_atlas.build_liquidity_lifecycle_v1_1 import (
    RESULTS, find_contacts,
)
from research.liquidity_oracle_atlas.build_oracle_atlas_v1_1 import (
    active_mask,
)


def _m():
    p = RESULTS / "liquidity_master_v1_1.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _c():
    p = RESULTS / "liquidity_contacts_v1_1.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _o():
    p = RESULTS / "oracle_risk_frontier_v1_1.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _s():
    p = RESULTS / "liquidity_state_snapshot_v1_1.parquet"
    return pd.read_parquet(p) if p.exists() else None


# ---------- 1. decision_time = bar 结束 ----------
def test_decision_time_is_bar_end():
    c = _c()
    if c is None:
        pytest.skip("无 contacts")
    dt = pd.to_datetime(c["decision_time"])
    ct = pd.to_datetime(c["contact_time"])
    assert bool(((dt - ct) == pd.Timedelta(minutes=5)).all()), \
        "decision_time 不是 contact bar 结束时间"


# ---------- 2. 生命周期 roll censor 存在且有效 ----------
def test_lifecycle_roll_censor():
    m = _m()
    if m is None:
        pytest.skip("无 master")
    assert "roll_censored" in m.columns
    assert int(m["roll_censored"].sum()) > 0, "未发现任何 roll censor"
    rc = m[m["roll_censored"]]
    # 被 roll censor 的 liquidity 不应有 penetration
    assert bool((~rc["consumed"]).all()), \
        "roll censor 后仍标记 consumed"


def test_find_contacts_respects_limit():
    hi = np.array([1.0, 1.0, 5.0, 5.0])
    lo = hi - 0.5
    op = hi - 0.1
    cl = hi - 0.2
    # limit=2 -> 不应看到 index 2,3 的触及
    out = find_contacts(hi, lo, op, cl, 0, 2.0, +1, 2)
    assert out == [], "find_contacts 越过了 discontinuity 上界"


# ---------- 3. Oracle 路径 roll censor ----------
def test_oracle_path_censor():
    o = _o()
    if o is None:
        pytest.skip("无 oracle")
    assert "path_censor" in o.columns
    assert set(o["path_censor"].unique()) <= {"ROLL_CENSORED",
                                              "DATA_END_CENSORED"}
    assert int((o["path_censor"] == "ROLL_CENSORED").sum()) > 0


# ---------- 4. active liquidity 排除已消费 ----------
def test_active_mask_excludes_consumed():
    df = pd.DataFrame({
        "available_time": pd.to_datetime(["2025-01-01", "2025-01-01",
                                          "2025-01-01"]),
        "first_penetration_time": pd.to_datetime([pd.NaT, "2025-01-05",
                                                 "2025-01-15"])})
    m = active_mask(df, np.datetime64("2025-01-10"))
    # 第1：未消费 -> active；第2：01-05 消费 -> 排除；第3：01-15 消费 -> 未到，active
    assert bool(np.array_equal(m, np.array([True, False, True]))), \
        "active_mask 未正确排除已消费 liquidity"


def test_active_less_than_historical():
    s = _s()
    if s is None:
        pytest.skip("无 state")
    a = pd.to_numeric(s["active_visible_count"])
    h = pd.to_numeric(s["historical_visible_count"])
    assert bool((a <= h).all()), "active 数不应超过历史可见数"
    assert float(a.mean()) < float(h.mean()), \
        "active 平均未低于历史可见（说明未排除 consumed）"


# ---------- 5. first_mR 相对 stop 判定 ----------
def test_first_mR_states_valid():
    o = _o()
    if o is None:
        pytest.skip("无 oracle")
    allowed = {"REACHED_BEFORE_STOP", "NOT_REACHED_BEFORE_STOP",
               "AMBIGUOUS_SAME_BAR", "REACHED_BEFORE_CENSOR",
               "CENSORED_NOT_REACHED"}
    for m in (1, 2, 3):
        col = f"first_{m}R_state"
        assert col in o.columns, f"缺少 {col}"
        assert set(o[col].dropna().unique()) <= allowed, f"{col} 状态非法"


def test_first_mR_consistent_with_stop():
    o = _o()
    if o is None:
        pytest.skip("无 oracle")
    d = o[o["first_1R_state"] == "REACHED_BEFORE_STOP"]
    if len(d):
        assert bool((d["first_1R_bar"] < d["bars_to_stop"]).all()), \
            "REACHED_BEFORE_STOP 但未早于 stop"
    d2 = o[o["first_1R_state"] == "NOT_REACHED_BEFORE_STOP"]
    assert bool(d2["first_1R_bar"].isna().all()), \
        "NOT_REACHED 却记录了 bar"


# ---------- 6. Oracle status 显式 censor ----------
def test_oracle_status_explicit():
    o = _o()
    if o is None:
        pytest.skip("无 oracle")
    allowed = {"STOPPED", "TARGET_REACHED_THEN_STOPPED",
               "TARGET_REACHED_CENSORED", "CENSORED_NO_TARGET",
               "AMBIGUOUS_INTRABAR_ORDER", "NO_ACTIVE_TARGET"}
    assert set(o["status"].unique()) <= allowed, "存在未定义的 status"


# ---------- 7. 无 500 bar 主截断 / 无 400 截断 ----------
def test_no_primary_truncation():
    o = _o()
    if o is None:
        pytest.skip("无 oracle")
    # 路径长度应常常远超 500（说明未以 500 为主 horizon）
    assert float(o["path_len"].median()) > 500, \
        "path_len 中位数不足，疑似仍以 500 为主 horizon"
    # 400 target 截断在 v1.1 不应生效
    assert float((o["n_target_clipped_count"] > 0).mean()) < 0.01, \
        "400 target 截断仍在生效"


# ---------- 8. duplicate identity ----------
def test_no_duplicate_identity():
    m = _m()
    if m is None:
        pytest.skip("无 master")
    keys = ["symbol", "price", "side", "liquidity_type", "liquidity_scope",
            "available_time"]
    n = int((m.groupby(keys).size() > 1).sum())
    assert n == 0, f"存在 {n} 组完全重复 identity"


# ---------- 9. Oracle 标签不进入事前特征 ----------
def test_oracle_not_in_state_features():
    s = _s()
    if s is None:
        pytest.skip("无 state")
    banned = {"conservative_best_R", "optimistic_best_R", "stop_hit",
              "bars_to_stop", "best_R", "status", "path_censor"}
    assert not (banned & set(s.columns)), "Oracle 字段进入了事前特征表"


# ---------- 10. contact hazard 单调性（画像，非断言） ----------
def test_contact_hazard_computed():
    p = RESULTS / "contact_hazard.csv"
    if not p.exists():
        pytest.skip("无 hazard")
    h = pd.read_csv(p)
    assert len(h) >= 4, "hazard 未覆盖 k=1..4"
    assert bool((h["penetration_hazard"] >= 0).all()
                and (h["penetration_hazard"] <= 1).all())
