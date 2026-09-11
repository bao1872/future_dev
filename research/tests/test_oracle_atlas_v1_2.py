"""SMC Oracle Atlas v1.2 —— 方向标签与不确定性语义测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.liquidity_oracle_atlas.build_liquidity_lifecycle_v1_1 import (
    RESULTS,
)
from research.liquidity_oracle_atlas.build_oracle_atlas_v1_2 import (
    active_mask,
)
from research.liquidity_oracle_atlas.profile_oracle_atlas_v1_2 import (
    _cmp_dir, _stability, _time_dir,
)


def _o():
    p = RESULTS / "oracle_risk_frontier_v1_2.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _s():
    p = RESULTS / "liquidity_state_snapshot_v1_2.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _row(low, up, cls="EXACT_RESOLVED", bars=3):
    return dict(best_R_lower=low, best_R_upper=up, resolution_class=cls,
                bars_to_best_lower=bars)


# ---------- 1. first_penetration_time == decision_time 不再 active ----------
def test_same_bar_consume_excluded():
    df = pd.DataFrame({
        "available_time": pd.to_datetime(["2025-01-01", "2025-01-01"]),
        "first_penetration_time": pd.to_datetime(["2025-01-10", "2025-01-11"])})
    m = active_mask(df, np.datetime64("2025-01-10"))
    # 第一个 == decision_time -> 已消费；第二个更晚 -> 仍 active
    assert bool(np.array_equal(m, np.array([False, True]))), \
        "first_penetration_time == decision_time 仍被判 active"


def test_active_not_greater_than_v11():
    s = _s()
    if s is None:
        pytest.skip("无 state")
    a = pd.to_numeric(s["same_price_identity_count"])
    b = pd.to_numeric(s["same_price_identity_count_v11"])
    assert bool((a <= b).all()), "v1.2 同价计数不应超过 v1.1"


# ---------- 3. censored best_R_upper = unknown ----------
def test_censored_upper_is_nan():
    o = _o()
    if o is None:
        pytest.skip("无 oracle")
    c = o[o["resolution_class"] == "CENSORED_LOWER_BOUND"]
    assert len(c) > 0
    assert bool(c["best_R_upper"].isna().all()), \
        "censored 的 upper 不是未知"
    assert bool((~c["best_R_is_exact"]).all())


# ---------- 4. no-active-target 不等于 R=0 ----------
def test_no_active_target_is_nan():
    o = _o()
    if o is None:
        pytest.skip("无 oracle")
    c = o[o["resolution_class"] == "NO_ACTIVE_TARGET"]
    assert len(c) > 0
    assert bool(c["best_R_lower"].isna().all()
                and c["best_R_upper"].isna().all()), \
        "NO_ACTIVE_TARGET 被当成 R=0"


def test_no_active_target_not_compared():
    assert _cmp_dir(_row(np.nan, np.nan, "NO_ACTIVE_TARGET"),
                    _row(2.0, 2.0)) == "NO_COMPARABLE_TARGET"


# ---------- 5. ambiguous lower/upper ----------
def test_ambiguous_interval():
    o = _o()
    if o is None:
        pytest.skip("无 oracle")
    c = o[o["resolution_class"] == "AMBIGUOUS_INTERVAL"]
    assert len(c) > 0
    assert bool((c["best_R_upper"] >= c["best_R_lower"]).all()), \
        "ambiguous 上界小于下界"
    assert bool((~c["best_R_is_exact"]).all())


# ---------- 6/7/8/9. risk-specific 比较 ----------
def test_exact_long_greater():
    assert _cmp_dir(_row(2.3, 2.3), _row(0.8, 0.8)) == "LONG_DOMINATES"


def test_exact_short_greater():
    assert _cmp_dir(_row(0.5, 0.5), _row(1.9, 1.9)) == "SHORT_DOMINATES"


def test_interval_overlap_not_forced():
    assert _cmp_dir(_row(0.3, 2.0, "AMBIGUOUS_INTERVAL"),
                    _row(0.5, 0.5)) == "TRADEOFF_OR_OVERLAP"


def test_censor_cannot_prove_direction():
    assert _cmp_dir(_row(2.3, np.nan, "CENSORED_LOWER_BOUND"),
                    _row(2.0, 2.0)) == "UNRESOLVED_CENSOR"


# ---------- 10. robust direction 计数 ----------
def test_stability_rules():
    assert _stability(["LONG_DOMINATES"] * 7) == "ROBUST_LONG"
    assert _stability(["SHORT_DOMINATES"] * 7) == "ROBUST_SHORT"
    assert _stability(["LONG_DOMINATES", "SHORT_DOMINATES"]
                      + ["LONG_DOMINATES"] * 5) == "RISK_DEPENDENT"
    assert _stability(["TRADEOFF_OR_OVERLAP"] * 7) == "NO_DIRECTION"
    assert _stability(["UNRESOLVED_CENSOR"] * 7) == "UNRESOLVED"


def test_stability_counts_saved():
    p = RESULTS / "oracle_direction_stability_v1_2.parquet"
    if not p.exists():
        pytest.skip("无 stability")
    d = pd.read_parquet(p)
    for c in ("n_long_dom", "n_short_dom", "n_tradeoff", "n_unresolved",
              "n_no_target"):
        assert c in d.columns, f"缺少计数 {c}"
    assert bool((d["n_long_dom"] + d["n_short_dom"] + d["n_tradeoff"]
                 + d["n_unresolved"] + d["n_no_target"]
                 == d["n_risk"]).all())


# ---------- 11. risk_ATR 不作为 Pareto 越小越好 ----------
def test_time_dir_compares_same_risk_only():
    # 同 risk 下 RR 更高且 bars 更短 -> LONG time 支配
    assert _time_dir(_row(2.5, 2.5, bars=8), _row(1.7, 1.7, bars=20)) == \
        "LONG_TIME_RR_DOMINATES"
    # RR 更高但更慢、对方更快更弱 -> TRADEOFF，不得强行判优劣
    assert _time_dir(_row(3.0, 3.0, bars=40), _row(2.0, 2.0, bars=6)) == \
        "TRADEOFF"


# ---------- 12. Oracle 标签不进入事前状态表 ----------
def test_oracle_not_in_state():
    s = _s()
    if s is None:
        pytest.skip("无 state")
    banned = {"best_R_lower", "best_R_upper", "rr_direction",
              "direction_stability", "time_direction", "resolution_class"}
    assert not (banned & set(s.columns)), "Oracle 标签进入事前状态表"


# ---------- 13. roll / data-end censor 保持 ----------
def test_path_censor_preserved():
    o = _o()
    if o is None:
        pytest.skip("无 oracle")
    assert set(o["path_censor"].unique()) <= {"ROLL_CENSORED",
                                              "DATA_END_CENSORED"}
    assert int((o["path_censor"] == "ROLL_CENSORED").sum()) > 0


# ---------- 14. target cluster identity 保留 ----------
def test_target_cluster_fields():
    o = _o()
    if o is None:
        pytest.skip("无 oracle")
    c = o[o["best_target_price"].notna()]
    assert len(c) > 0
    for f in ("best_target_scopes", "best_target_scope_count",
              "best_target_min_scope", "best_target_max_scope",
              "best_target_cluster_size"):
        assert f in c.columns, f"缺少 {f}"
    assert bool((c["best_target_scope_count"] >= 1).all())
