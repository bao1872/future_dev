"""LOCAL-0 — 小型确定性合成测试（不依赖任何离线数据 / 缓存）。

运行：
  .venv/bin/python research/liquidity_oracle_atlas/test_local_liquidity_transition_v0.py

覆盖用户 §21 要求的 A–G，外加严格不等号与因果可用性两个结构性用例。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import research.liquidity_oracle_atlas.experiment_local_liquidity_transition_v0 as m  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} :: {detail}")
        FAILS.append(name)


T0 = pd.Timestamp("2025-01-02 09:05:00")     # decision_time (bar END)


def mk_master(rows) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["liquidity_id", "symbol", "liquidity_type", "liquidity_scope",
                 "side", "price", "available_time", "first_penetration_time"],
    )


def pair_and_label(master_df, close=100.0, decision_time=T0):
    """返回 (upper_price, lower_price, has_upper, has_lower, up_fp, dn_fp,
    up_conflict, dn_conflict)。"""
    info = m.build_price_group_info(master_df)
    dt = m.to_ns_int(np.array([decision_time]))
    pair = m.nearest_active_pair_chunk(m.to_dt64_ns(dt),
                                       np.array([close], dtype=float), info)
    out = dict(
        upper_price=float(pair["upper_price"][0]),
        lower_price=float(pair["lower_price"][0]),
        has_upper=bool(pair["has_upper"][0]),
        has_lower=bool(pair["has_lower"][0]),
        up_fp=m.INAT, dn_fp=m.INAT, up_conflict=False, dn_conflict=False,
    )
    if out["has_upper"]:
        r = m.resolve_group_fp(dt, pair["upper_group"], info)
        out["up_fp"] = int(r["fp"][0])
        out["up_conflict"] = bool(r["conflict"][0])
    if out["has_lower"]:
        r = m.resolve_group_fp(dt, pair["lower_group"], info)
        out["dn_fp"] = int(r["fp"][0])
        out["dn_conflict"] = bool(r["conflict"][0])
    out["label"] = int(m.classify_pair_time(
        m.to_dt64_ns(np.array([out["up_fp"]])),
        m.to_dt64_ns(np.array([out["dn_fp"]])),
    )[0])
    return out


# --------------------------------------------------------------- A
def test_A_upper_first():
    ms = mk_master([
        ("U1", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 101.0,
         pd.Timestamp("2025-01-02 09:00:00"), pd.Timestamp("2025-01-02 10:05:00")),
        ("D1", "AG", "CONFIRMED_SWING_LOW", "5m", -1, 99.0,
         pd.Timestamp("2025-01-02 09:00:00"), pd.Timestamp("2025-01-02 11:05:00")),
    ])
    r = pair_and_label(ms)
    check("A upper_first -> UP", r["label"] == m.UP,
          f"label={r['label']} up={r['upper_price']} dn={r['lower_price']}")
    check("A upper price selected", r["upper_price"] == 101.0, str(r["upper_price"]))
    check("A lower price selected", r["lower_price"] == 99.0, str(r["lower_price"]))


# --------------------------------------------------------------- B
def test_B_lower_first():
    ms = mk_master([
        ("U1", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 101.0,
         pd.Timestamp("2025-01-02 09:00:00"), pd.Timestamp("2025-01-02 11:05:00")),
        ("D1", "AG", "CONFIRMED_SWING_LOW", "5m", -1, 99.0,
         pd.Timestamp("2025-01-02 09:00:00"), pd.Timestamp("2025-01-02 10:05:00")),
    ])
    r = pair_and_label(ms)
    check("B lower_first -> DOWN", r["label"] == m.DOWN, f"label={r['label']}")


# --------------------------------------------------------------- C
def test_C_same_bar_penetration():
    ms = mk_master([
        ("U1", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 101.0,
         pd.Timestamp("2025-01-02 09:00:00"), pd.Timestamp("2025-01-02 10:05:00")),
        ("D1", "AG", "CONFIRMED_SWING_LOW", "5m", -1, 99.0,
         pd.Timestamp("2025-01-02 09:00:00"), pd.Timestamp("2025-01-02 10:05:00")),
    ])
    r = pair_and_label(ms)
    check("C same_bar -> AMBIGUOUS", r["label"] == m.AMBIGUOUS, f"label={r['label']}")


# --------------------------------------------------------------- D
def test_D_no_penetration():
    ms = mk_master([
        ("U1", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 101.0,
         pd.Timestamp("2025-01-02 09:00:00"), pd.NaT),
        ("D1", "AG", "CONFIRMED_SWING_LOW", "5m", -1, 99.0,
         pd.Timestamp("2025-01-02 09:00:00"), pd.NaT),
    ])
    r = pair_and_label(ms)
    check("D both_NaT -> CENSOR", r["label"] == m.CENSOR, f"label={r['label']}")
    # 只有 upper 有 finite fp -> UP；只有 lower 有 -> DOWN
    ms2 = ms.copy()
    ms2.loc[0, "first_penetration_time"] = pd.Timestamp("2025-01-02 10:05:00")
    check("D upper_only_finite -> UP", pair_and_label(ms2)["label"] == m.UP)
    ms3 = ms.copy()
    ms3.loc[1, "first_penetration_time"] = pd.Timestamp("2025-01-02 10:05:00")
    check("D lower_only_finite -> DOWN", pair_and_label(ms3)["label"] == m.DOWN)


# --------------------------------------------------------------- E
def test_E_consumed_not_active():
    # fp == decision_time -> consumed (active requires fp > decision_time)
    ms = mk_master([
        ("U1", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 101.0,
         pd.Timestamp("2025-01-02 09:00:00"), pd.Timestamp("2025-01-02 09:05:00")),
        ("U2", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 105.0,
         pd.Timestamp("2025-01-02 09:00:00"), pd.NaT),
        ("D1", "AG", "CONFIRMED_SWING_LOW", "5m", -1, 99.0,
         pd.Timestamp("2025-01-02 09:00:00"), pd.NaT),
    ])
    r = pair_and_label(ms)
    check("E consumed_upper_skipped", r["upper_price"] == 105.0,
          f"upper={r['upper_price']}")
    # fp < decision_time 也算 consumed
    ms.loc[0, "first_penetration_time"] = pd.Timestamp("2025-01-02 08:00:00")
    check("E earlier_fp_also_consumed", pair_and_label(ms)["upper_price"] == 105.0)


# --------------------------------------------------------------- F
def test_F_future_availability_not_active():
    ms = mk_master([
        ("U1", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 101.0,
         pd.Timestamp("2025-01-02 09:10:00"), pd.NaT),   # > decision_time
        ("U2", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 105.0,
         pd.Timestamp("2025-01-02 09:00:00"), pd.NaT),
        ("D1", "AG", "CONFIRMED_SWING_LOW", "5m", -1, 99.0,
         pd.Timestamp("2025-01-02 09:00:00"), pd.NaT),
    ])
    r = pair_and_label(ms)
    check("F future_available_skipped", r["upper_price"] == 105.0,
          f"upper={r['upper_price']}")


# --------------------------------------------------------------- G
def test_G_same_price_fp_conflict():
    ms = mk_master([
        ("UA", "AG", "PREV_CONTIG_SESSION_HIGH", "CONTIG_SESSION", +1, 101.0,
         pd.Timestamp("2025-01-02 09:00:00"), pd.Timestamp("2025-01-02 10:05:00")),
        ("UB", "AG", "PREV_CONTIG_SESSION_LOW", "CONTIG_SESSION", -1, 101.0,
         pd.Timestamp("2025-01-02 09:00:00"), pd.Timestamp("2025-01-02 11:05:00")),
        ("D1", "AG", "CONFIRMED_SWING_LOW", "5m", -1, 99.0,
         pd.Timestamp("2025-01-02 09:00:00"), pd.NaT),
    ])
    r = pair_and_label(ms)
    check("G same_price_conflict_detected", r["up_conflict"] is True,
          f"conflict={r['up_conflict']}")
    # 一致时不应报冲突：两个 identity 完全同 fp
    ms2 = ms.copy()
    ms2.loc[1, "first_penetration_time"] = pd.Timestamp("2025-01-02 10:05:00")
    r2 = pair_and_label(ms2)
    check("G identical_fp_no_conflict", r2["up_conflict"] is False)
    # 一个 NaT 一个 finite：允许，使用 finite，不报冲突
    ms3 = ms.copy()
    ms3.loc[1, "first_penetration_time"] = pd.NaT
    r3 = pair_and_label(ms3)
    check("G nat_plus_finite_no_conflict", r3["up_conflict"] is False)
    check("G nat_plus_finite_uses_finite",
          m.to_dt64_ns(np.array([r3["up_fp"]]))[0]
          == np.datetime64("2025-01-02T10:05:00"), str(r3["up_fp"]))


# --------------------------------------------------------------- 额外结构用例
def test_strict_inequality():
    ms = mk_master([
        ("EQ", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 100.0,
         pd.Timestamp("2025-01-02 09:00:00"), pd.NaT),   # == close
        ("U1", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 101.0,
         pd.Timestamp("2025-01-02 09:00:00"), pd.NaT),
        ("D1", "AG", "CONFIRMED_SWING_LOW", "5m", -1, 99.0,
         pd.Timestamp("2025-01-02 09:00:00"), pd.NaT),
    ])
    r = pair_and_label(ms, close=100.0)
    check("level_equal_close_is_neither_side",
          r["upper_price"] == 101.0 and r["lower_price"] == 99.0,
          f"up={r['upper_price']} dn={r['lower_price']}")


def test_missing_side_flagged():
    ms = mk_master([
        ("U1", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 101.0,
         pd.Timestamp("2025-01-02 09:00:00"), pd.NaT),
    ])
    r = pair_and_label(ms, close=100.0)
    check("no_lower_pair_flagged", r["has_lower"] is False and r["has_upper"])


# ===========================================================================
# LOCAL-0 FIX — corrected lifecycle tests
# ===========================================================================
def mk_bars(h, l, c, disc=None, start="2025-01-02 09:00"):
    n = len(h)
    t = pd.date_range(start, periods=n, freq="5min")
    return dict(
        n=n,
        t=pd.to_datetime(t).to_numpy().astype("datetime64[ns]"),
        h=np.asarray(h, float), l=np.asarray(l, float),
        c=np.asarray(c, float), o=np.asarray(c, float),
        disc=np.asarray(disc if disc is not None else [False] * n, bool),
        atr=np.ones(n),
        td=np.asarray(t).astype("datetime64[D]"),
    )


def mk_master_fix(rows) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["liquidity_id", "symbol", "liquidity_type", "liquidity_scope",
                 "side", "price", "available_time", "available_bar_index",
                 "first_penetration_time"],
    )


T_BASE = pd.Timestamp("2025-01-02 09:00:00")


def fp_bar_of(master_df, bars, i=0):
    lc = m.build_corrected_lifecycle(master_df, bars)
    return int(lc["new_fp_bar"][i]), lc


# --- 1 -------------------------------------------------------------------
def test_F1_touch_then_strict_break():
    bars = mk_bars([99, 100, 101, 100], [98, 98, 98, 98], [99, 100, 101, 100])
    ms = mk_master_fix([("U", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 100.0,
                         T_BASE + pd.Timedelta(minutes=5), 0, pd.NaT)])
    fp, _ = fp_bar_of(ms, bars)
    # 旧 re-arm 逻辑会在 bar1 touch 后等待离开 level，从而漏掉 bar2 的突破
    check("F1 touch==level then strict break -> fp=2", fp == 2, f"fp={fp}")


# --- 2 -------------------------------------------------------------------
def test_F2_multiple_touch_then_strict_break():
    bars = mk_bars([99, 100, 100, 100, 101],
                   [98, 98, 98, 98, 98],
                   [99, 100, 100, 100, 101])
    ms = mk_master_fix([("U", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 100.0,
                         T_BASE + pd.Timedelta(minutes=5), 0, pd.NaT)])
    fp, _ = fp_bar_of(ms, bars)
    check("F2 several touch==level then strict break -> fp=4", fp == 4, f"fp={fp}")


# --- 3 -------------------------------------------------------------------
def test_F3_side_plus_strict_high_gt_level():
    bars = mk_bars([99, 101, 100], [98, 98, 98], [99, 101, 100])
    ms = mk_master_fix([("U", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 100.0,
                         T_BASE + pd.Timedelta(minutes=5), 0, pd.NaT)])
    fp, _ = fp_bar_of(ms, bars)
    check("F3 side=+1 high>level consumed at first bar", fp == 1, f"fp={fp}")
    # 精确相等不消费
    bars2 = mk_bars([99, 100, 100], [98, 98, 98], [99, 100, 100])
    fp2, _ = fp_bar_of(ms, bars2)
    check("F3 exact high==level is NOT consumed", fp2 == -1, f"fp={fp2}")


# --- 4 -------------------------------------------------------------------
def test_F4_side_minus_strict_low_lt_level():
    bars = mk_bars([101, 99, 100], [101, 99, 100], [101, 99, 100])
    ms = mk_master_fix([("D", "AG", "CONFIRMED_SWING_LOW", "5m", -1, 100.0,
                         T_BASE + pd.Timedelta(minutes=5), 0, pd.NaT)])
    fp, _ = fp_bar_of(ms, bars)
    check("F4 side=-1 low<level consumed at first bar", fp == 1, f"fp={fp}")
    bars2 = mk_bars([101, 100, 100], [101, 100, 100], [101, 100, 100])
    fp2, _ = fp_bar_of(ms, bars2)
    check("F4 exact low==level is NOT consumed", fp2 == -1, f"fp={fp2}")


# --- 5 -------------------------------------------------------------------
def test_F5_gap_cross_identified():
    # open 直接跳到 level 之上（gap），high > level 必须被识别
    bars = mk_bars([99, 103], [98, 102], [99, 103])
    ms = mk_master_fix([("U", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 100.0,
                         T_BASE + pd.Timedelta(minutes=5), 0, pd.NaT)])
    fp, _ = fp_bar_of(ms, bars)
    check("F5 gap_cross identified", fp == 1, f"fp={fp}")


# --- 6 -------------------------------------------------------------------
def test_F6_discontinuity_truncation():
    bars = mk_bars([99, 101, 101, 101], [98, 98, 98, 98],
                   [99, 101, 101, 101], disc=[False, True, False, False])
    ms = mk_master_fix([("U", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 100.0,
                         T_BASE, -1 + 0, pd.NaT)])
    ms["available_bar_index"] = [0.0]
    ms["available_time"] = [T_BASE + pd.Timedelta(minutes=5)]
    fp, _ = fp_bar_of(ms, bars)
    check("F6 search truncated at discontinuity -> no fp", fp == -1, f"fp={fp}")


# --- 7 -------------------------------------------------------------------
def test_F7_side_position_invariant():
    # level=105 side=+1，available_bar_index=0 -> 首根可观察 bar=1
    # bar0 的 close 已经在 105 之上（旧 time-only active 会把它算作 active）
    bars = mk_bars([106] * 4, [104] * 4, [106] * 4)
    ms = mk_master_fix([("U", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 105.0,
                         T_BASE + pd.Timedelta(minutes=5), 0, pd.NaT)])
    lc = m.build_corrected_lifecycle(ms, bars)
    v_timeonly, _, _ = m.side_position_audit("AG", ms, bars, lc, cap_examples=0)
    v_obs, v_after, _ = m.side_position_audit(
        "AG", ms, bars, lc, av_ns_override=lc["av_obs_ns"])
    check("F7 time-only active shows the stale-level violation",
          v_timeonly >= 1, f"v_timeonly={v_timeonly}")
    check("F7 observable-active removes it (total=0)", v_obs == 0, f"v_obs={v_obs}")
    check("F7 observable-active after-first-observed-bar=0", v_after == 0,
          f"v_after={v_after}")


# --- 8 -------------------------------------------------------------------
def test_F8_same_price_same_side_fp_consistent():
    # price=101 side=+1 两个 identity，首次突破都在 bar4
    bars = mk_bars([99, 99, 99, 99, 105, 99], [98] * 6, [99] * 6)
    ms = mk_master_fix([
        ("UA", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 101.0,
         T_BASE + pd.Timedelta(minutes=5), 0, pd.NaT),
        ("UB", "AG", "PREV_CONTIG_SESSION_HIGH", "CONTIG_SESSION", +1, 101.0,
         T_BASE + pd.Timedelta(minutes=15), 2, pd.NaT),
    ])
    lc = m.build_corrected_lifecycle(ms, bars)
    check("F8 both identities see the same strict crossing",
          int(lc["new_fp_bar"][0]) == 4 and int(lc["new_fp_bar"][1]) == 4,
          f"{lc['new_fp_bar'].tolist()}")
    info = m.build_level_groups(ms, lc["new_fp_ns"],
                                av_obs_ns=lc["av_obs_ns"])
    check("F8 same (price,side) collapsed into ONE group",
          int((info["unique_price"] == 101.0).sum()) == 1,
          str(info["unique_price"].tolist()))
    # decision at bar3 -> 两个 identity 都已 observable 且都未穿透
    dt = m.to_ns_int(bars["t"][3:4]) + m.BAR_NS
    g = np.array([int(np.flatnonzero(info["unique_price"] == 101.0)[0])])
    r = m.resolve_group_fp(dt, g, info)
    check("F8 no conflict for same-(price,side)", bool(r["conflict"][0]) is False)
    check("F8 two active identities", int(r["n_active"][0]) == 2,
          str(r["n_active"].tolist()))


# --- 9 -------------------------------------------------------------------
def test_F9_same_price_opposite_side_not_merged():
    # price=101 同时有 side=+1 (上破 bar4) 与 side=-1 (下破 bar2)
    # bar1 是双侧精确 touch（h==101 且 l==101），两侧都不消费
    bars = mk_bars([99, 101, 99, 99, 105, 99],
                   [98, 101, 97, 98, 98, 98],
                   [99, 101, 97, 99, 105, 99])
    ms = mk_master_fix([
        ("UA", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 101.0,
         T_BASE + pd.Timedelta(minutes=5), 0, pd.NaT),
        ("DB", "AG", "PREV_CONTIG_SESSION_LOW", "CONTIG_SESSION", -1, 101.0,
         T_BASE + pd.Timedelta(minutes=5), 0, pd.NaT),
    ])
    lc = m.build_corrected_lifecycle(ms, bars)
    check("F9 opposite sides get different fp (allowed)",
          int(lc["new_fp_bar"][0]) == 4 and int(lc["new_fp_bar"][1]) == 2,
          f"{lc['new_fp_bar'].tolist()}")
    info = m.build_level_groups(ms, lc["new_fp_ns"],
                                av_obs_ns=lc["av_obs_ns"])
    sel = info["unique_price"] == 101.0
    check("F9 two distinct (price,side) groups at same price",
          int(sel.sum()) == 2 and sorted(info["unique_side"][sel].tolist()) == [-1, 1],
          str(info["unique_side"][sel].tolist()))
    # decision bar2, close=97 -> upper 必须只取 side=+1 的那个 group
    dt = m.to_dt64_ns(m.to_ns_int(bars["t"][2:3]) + m.BAR_NS)
    pair = m.nearest_active_pair_chunk_v2(dt, np.array([97.0]), info)
    up_g = int(pair["upper_group"][0])
    check("F9 upper selects the side=+1 group only",
          up_g >= 0 and int(info["unique_side"][up_g]) == 1,
          f"up_g={up_g} side={info['unique_side'][up_g] if up_g >= 0 else None}")
    check("F9 lower not present (no active side=-1 below close)",
          bool(pair["has_lower"][0]) is False)


def test_classify_matrix():
    t1 = np.datetime64("2025-01-02T10:00:00")
    t2 = np.datetime64("2025-01-02T11:00:00")
    na = np.datetime64("NaT")
    got = m.classify_pair_time(
        np.array([t1, t2, na, na, t1]),
        np.array([t2, t1, na, t1, t1]),
    )
    want = np.array([m.UP, m.DOWN, m.CENSOR, m.DOWN, m.AMBIGUOUS])
    check("classify_pair_time matrix", np.array_equal(got, want),
          f"got={got.tolist()} want={want.tolist()}")


def main():
    test_A_upper_first()
    test_B_lower_first()
    test_C_same_bar_penetration()
    test_D_no_penetration()
    test_E_consumed_not_active()
    test_F_future_availability_not_active()
    test_G_same_price_fp_conflict()
    test_strict_inequality()
    test_missing_side_flagged()
    test_classify_matrix()
    # LOCAL-0 FIX — corrected lifecycle
    test_F1_touch_then_strict_break()
    test_F2_multiple_touch_then_strict_break()
    test_F3_side_plus_strict_high_gt_level()
    test_F4_side_minus_strict_low_lt_level()
    test_F5_gap_cross_identified()
    test_F6_discontinuity_truncation()
    test_F7_side_position_invariant()
    test_F8_same_price_same_side_fp_consistent()
    test_F9_same_price_opposite_side_not_merged()
    print(f"\n==== {len(FAILS)} FAIL ====")
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
