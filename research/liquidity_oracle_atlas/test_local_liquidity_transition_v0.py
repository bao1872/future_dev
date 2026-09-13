"""LOCAL-0 — 小型确定性合成测试（不依赖任何离线数据 / 缓存）。

运行：
  .venv/bin/python research/liquidity_oracle_atlas/test_local_liquidity_transition_v0.py

覆盖：
  * A–G   方向 outcome / consumed / not-yet-activated / 同组冲突（bar-index 版）
  * F1–F6 corrected strict-crossing lifecycle
  * L1–L6 activation / expiry / discontinuity 守卫
  * P    与 frozen active_prices_chunk 的 parity（无 discontinuity 时）
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


# ---------------------------------------------------------------- helpers
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


def mk_master(rows) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["liquidity_id", "symbol", "liquidity_type", "liquidity_scope",
                 "side", "price", "available_time", "available_bar_index",
                 "first_penetration_time"],
    )


def run_pair(master_df, bars, close=100.0, decision_bar=1, bar_idx=0):
    lc = m.build_corrected_lifecycle(master_df, bars)
    info = m.build_level_groups(master_df, lc, bars)
    bi = np.array([decision_bar], dtype=np.int64)
    pair = m.nearest_active_pair_chunk_v2(bi, np.array([close], dtype=float), info)
    out = dict(
        upper=float(pair["upper_price"][0]), lower=float(pair["lower_price"][0]),
        has_upper=bool(pair["has_upper"][0]), has_lower=bool(pair["has_lower"][0]),
        up_pen=-1, dn_pen=-1, up_exp=-1, dn_exp=-1,
        conflict=False, conflict_pen=False, conflict_exp=False,
        label=int(m.ROLL_RIGHT_CENSOR), lc=lc, info=info,
    )
    if out["has_upper"]:
        r = m.resolve_group_event(bi, pair["upper_group"], info)
        out["up_pen"] = int(r["pen"][0])
        out["up_exp"] = int(r["expiry"][0])
        out["conflict"] |= bool(r["conflict"][0])
        out["conflict_pen"] |= bool(r["conflict_pen"][0])
        out["conflict_exp"] |= bool(r["conflict_exp"][0])
    if out["has_lower"]:
        r = m.resolve_group_event(bi, pair["lower_group"], info)
        out["dn_pen"] = int(r["pen"][0])
        out["dn_exp"] = int(r["expiry"][0])
        out["conflict"] |= bool(r["conflict"][0])
        out["conflict_pen"] |= bool(r["conflict_pen"][0])
        out["conflict_exp"] |= bool(r["conflict_exp"][0])
    out["label"] = int(m.classify_pair_event(
        np.array([out["up_pen"]]), np.array([out["dn_pen"]]))[0])
    # 与 build_symbol_samples 一致：两侧均无 penetration 时按 expiry 拆分
    if out["label"] == m.ROLL_RIGHT_CENSOR and out["up_exp"] >= bars["n"]:
        out["label"] = int(m.END_OF_DATA_RIGHT_CENSOR)
    return out


# bars: 8 bars, high/low 在指定 bar 突破 105 / 跌破 95
BARS_UP_AT_2 = mk_bars([99, 99, 106, 99, 99, 99, 99, 99],
                       [98] * 8, [99] * 8)
BARS_DN_AT_2 = mk_bars([101] * 8, [100, 100, 94, 100, 100, 100, 100, 100],
                       [100] * 8)


# ============================================================ A–G
def test_A_upper_first():
    ms = mk_master([
        ("U", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 105.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
        ("D", "AG", "CONFIRMED_SWING_LOW", "5m", -1, 95.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
    ])
    bars = mk_bars([99, 99, 106, 99, 99, 99, 99, 99],
                   [100, 100, 100, 100, 100, 94, 100, 100], [99] * 8)
    r = run_pair(ms, bars, close=100.0, decision_bar=1)
    check("A upper_first -> UP", r["label"] == m.UP, f"label={r['label']}")
    check("A upper selected", r["upper"] == 105.0, str(r["upper"]))
    check("A lower selected", r["lower"] == 95.0, str(r["lower"]))


def test_B_lower_first():
    ms = mk_master([
        ("U", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 105.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
        ("D", "AG", "CONFIRMED_SWING_LOW", "5m", -1, 95.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
    ])
    bars = mk_bars([100, 100, 100, 100, 106, 100, 100, 100],
                   [100, 100, 94, 100, 100, 100, 100, 100], [100] * 8)
    r = run_pair(ms, bars, close=100.0, decision_bar=1)
    check("B lower_first -> DOWN", r["label"] == m.DOWN, f"label={r['label']}")


def test_C_same_bar():
    ms = mk_master([
        ("U", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 105.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
        ("D", "AG", "CONFIRMED_SWING_LOW", "5m", -1, 95.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
    ])
    bars = mk_bars([100, 100, 106, 100, 100, 100, 100, 100],
                   [100, 100, 94, 100, 100, 100, 100, 100], [100] * 8)
    r = run_pair(ms, bars, close=100.0, decision_bar=1)
    check("C same_bar -> AMBIGUOUS", r["label"] == m.AMBIGUOUS,
          f"label={r['label']} up_pen={r['up_pen']} dn_pen={r['dn_pen']}")


def test_D_right_censor():
    ms = mk_master([
        ("U", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 105.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
        ("D", "AG", "CONFIRMED_SWING_LOW", "5m", -1, 95.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
    ])
    bars = mk_bars([100] * 8, [100] * 8, [100] * 8)
    r = run_pair(ms, bars, close=100.0, decision_bar=1)
    check("D no penetration + expiry==n -> END_OF_DATA_RIGHT_CENSOR",
          r["label"] == m.END_OF_DATA_RIGHT_CENSOR, f"label={r['label']}")
    # 只有 upper 有 penetration -> UP；只有 lower -> DOWN
    bars2 = mk_bars([100, 100, 106, 100, 100, 100, 100, 100],
                    [100] * 8, [100] * 8)
    check("D upper_only -> UP",
          run_pair(ms, bars2, 100.0, 1)["label"] == m.UP)
    bars3 = mk_bars([100] * 8, [100, 100, 94, 100, 100, 100, 100, 100],
                    [100] * 8)
    check("D lower_only -> DOWN",
          run_pair(ms, bars3, 100.0, 1)["label"] == m.DOWN)


def test_E_consumed_not_active():
    # 已 penetration 的 level：penetration_bar <= decision_bar -> 不 active
    bars = mk_bars([100, 106, 100, 100, 100, 100, 100, 100],
                   [100, 100, 100, 100, 100, 94, 100, 100], [100] * 8)
    ms = mk_master([
        ("U", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 105.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
        ("U2", "AG", "PREV_TRADING_WEEK_HIGH", "TRADING_WEEK", +1, 120.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
        ("D", "AG", "CONFIRMED_SWING_LOW", "5m", -1, 95.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
    ])
    # decision_bar=2: U 已在 bar1 被突破（pen=1 <= 2） -> 不得 active
    r = run_pair(ms, bars, close=100.0, decision_bar=2)
    check("E consumed_upper_skipped", r["upper"] == 120.0, str(r["upper"]))


def test_F_not_yet_activated():
    bars = mk_bars([100] * 8, [100] * 8, [100] * 8)
    ms = mk_master([
        ("U", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 105.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
        ("U2", "AG", "PREV_TRADING_WEEK_HIGH", "TRADING_WEEK", +1, 102.0,
         pd.Timestamp("2025-01-02 09:30:00"), 5, pd.NaT),
        ("D", "AG", "CONFIRMED_SWING_LOW", "5m", -1, 95.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
    ])
    # decision_bar=1: U2 的 activation_bar=5 > 1 -> 不得 active
    r = run_pair(ms, bars, close=100.0, decision_bar=1)
    check("F future_activation_skipped", r["upper"] == 105.0, str(r["upper"]))
    # decision_bar=5: U2 已 activation -> 它是更近的 upper
    r5 = run_pair(ms, bars, close=100.0, decision_bar=5)
    check("F become_active_at_activation_bar", r5["upper"] == 102.0,
          str(r5["upper"]))


def test_G_same_group_conflict():
    # 同一 (price, side) 两个 identity 给出不同 finite penetration -> conflict
    bars = mk_bars([99] * 8, [98] * 8, [99] * 8)
    ms = mk_master([
        ("UA", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 105.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
        ("UB", "AG", "PREV_CONTIG_SESSION_HIGH", "CONTIG_SESSION", +1, 105.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
    ])
    # 手工构造不一致：直接改 penetration_bar
    lc = m.build_corrected_lifecycle(ms, bars)
    lc["penetration_bar"] = np.array([3, 5], dtype=np.int64)
    info = m.build_level_groups(ms, lc, bars)
    g = np.array([int(np.flatnonzero(info["unique_price"] == 105.0)[0])])
    r = m.resolve_group_event(np.array([1], dtype=np.int64), g, info)
    check("G same_group_penetration_conflict", bool(r["conflict"][0]) is True)
    # 一致 -> 无冲突
    lc2 = m.build_corrected_lifecycle(ms, bars)
    lc2["penetration_bar"] = np.array([3, 3], dtype=np.int64)
    info2 = m.build_level_groups(ms, lc2, bars)
    r2 = m.resolve_group_event(np.array([1], dtype=np.int64), g, info2)
    check("G consistent_penetration_no_conflict",
          bool(r2["conflict"][0]) is False)


# ============================================================ F1–F6
def test_F1_touch_then_strict_break():
    bars = mk_bars([99, 100, 101, 100], [98, 98, 98, 98], [99, 100, 101, 100])
    ms = mk_master([("U", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 100.0,
                     pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT)])
    lc = m.build_corrected_lifecycle(ms, bars)
    check("F1 touch==level then strict break -> bar2",
          int(lc["penetration_bar"][0]) == 2, str(lc["penetration_bar"].tolist()))


def test_F2_multiple_touch_then_strict_break():
    bars = mk_bars([99, 100, 100, 100, 101], [98] * 5, [99, 100, 100, 100, 101])
    ms = mk_master([("U", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 100.0,
                     pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT)])
    lc = m.build_corrected_lifecycle(ms, bars)
    check("F2 several touch then strict break -> bar4",
          int(lc["penetration_bar"][0]) == 4, str(lc["penetration_bar"].tolist()))


def test_F3_F4_strict_sides():
    bars_up = mk_bars([99, 101, 100], [98, 98, 98], [99, 101, 100])
    ms_up = mk_master([("U", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 100.0,
                        pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT)])
    check("F3 side=+1 high>level consumed",
          int(m.build_corrected_lifecycle(ms_up, bars_up)["penetration_bar"][0]) == 1)
    bars_t = mk_bars([99, 100, 100], [98, 98, 98], [99, 100, 100])
    check("F3 exact high==level NOT consumed",
          int(m.build_corrected_lifecycle(ms_up, bars_t)["penetration_bar"][0]) == -1)

    bars_dn = mk_bars([101, 99, 100], [101, 99, 100], [101, 99, 100])
    ms_dn = mk_master([("D", "AG", "CONFIRMED_SWING_LOW", "5m", -1, 100.0,
                        pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT)])
    check("F4 side=-1 low<level consumed",
          int(m.build_corrected_lifecycle(ms_dn, bars_dn)["penetration_bar"][0]) == 1)
    bars_dt = mk_bars([101, 100, 100], [101, 100, 100], [101, 100, 100])
    check("F4 exact low==level NOT consumed",
          int(m.build_corrected_lifecycle(ms_dn, bars_dt)["penetration_bar"][0]) == -1)


def test_F5_gap_cross():
    bars = mk_bars([99, 103], [98, 102], [99, 103])
    ms = mk_master([("U", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 100.0,
                     pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT)])
    check("F5 gap_cross identified",
          int(m.build_corrected_lifecycle(ms, bars)["penetration_bar"][0]) == 1)


def test_F6_discontinuity_truncation():
    bars = mk_bars([99, 101, 101, 101], [98] * 4, [99, 101, 101, 101],
                   disc=[False, True, False, False])
    ms = mk_master([("U", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 100.0,
                     pd.Timestamp("2025-01-02 09:00:00"), 0, pd.NaT)])
    lc = m.build_corrected_lifecycle(ms, bars)
    check("F6 search truncated at discontinuity -> no penetration",
          int(lc["penetration_bar"][0]) == -1, str(lc["penetration_bar"].tolist()))
    check("F6 expiry_bar == first discontinuity bar",
          int(lc["expiry_bar"][0]) == 1, str(lc["expiry_bar"].tolist()))


# ============================================================ L1–L6
def test_L1_selected_identity_not_expired():
    bars = mk_bars([100] * 8, [100] * 8, [100] * 8)
    ms = mk_master([
        ("U", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 105.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
        ("D", "AG", "CONFIRMED_SWING_LOW", "5m", -1, 95.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
    ])
    r = run_pair(ms, bars, 100.0, 1)
    check("L1 selected upper decision_bar < expiry_bar",
          r["up_exp"] > 1 and r["dn_exp"] > 1,
          f"up_exp={r['up_exp']} dn_exp={r['dn_exp']}")


def test_L2_expiry_kills_level():
    # disc at bar 3；level 在 [1,3) 未被突破 -> expiry=3
    bars = mk_bars([99] * 8, [98] * 8, [99] * 8,
                   disc=[False, False, False, True, False, False, False, False])
    ms = mk_master([("U", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 105.0,
                     pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT)])
    lc = m.build_corrected_lifecycle(ms, bars)
    info = m.build_level_groups(ms, lc, bars)
    check("L2 expiry_bar == discontinuity bar",
          int(lc["expiry_bar"][0]) == 3, str(lc["expiry_bar"].tolist()))
    a = m.active_level_groups_chunk(np.array([2], dtype=np.int64), info)
    b = m.active_level_groups_chunk(np.array([3], dtype=np.int64), info)
    check("L2 active before expiry", bool(a.any()))
    check("L2 NOT active at expiry bar", not bool(b.any()))
    c = m.active_level_groups_chunk(np.array([5], dtype=np.int64), info)
    check("L2 NOT active in later segment", not bool(c.any()))


def test_L3_old_level_never_reappears():
    bars = mk_bars([99] * 8, [98] * 8, [99] * 8,
                   disc=[False, False, True, False, False, False, False, False])
    ms = mk_master([("U", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 105.0,
                     pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT)])
    lc = m.build_corrected_lifecycle(ms, bars)
    info = m.build_level_groups(ms, lc, bars)
    act = m.active_level_groups_chunk(np.arange(8, dtype=np.int64), info)
    check("L3 active only in [activation, expiry)",
          act.flatten().tolist() == [True, True, False, False, False, False,
                                     False, False],
          str(act.flatten().tolist()))


def test_L4_activation_bar_is_immediate_boundary():
    bars = mk_bars([100] * 8, [100] * 8, [100] * 8)
    ms = mk_master([
        ("U", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 105.0,
         pd.Timestamp("2025-01-02 09:15:00"), 2, pd.NaT),
        ("D", "AG", "CONFIRMED_SWING_LOW", "5m", -1, 95.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
    ])
    lc = m.build_corrected_lifecycle(ms, bars)
    check("L4 activation_bar == available_bar_index (not +1)",
          int(lc["activation_bar"][0]) == 2, str(lc["activation_bar"].tolist()))
    r = run_pair(ms, bars, 100.0, decision_bar=2)
    check("L4 level usable at its activation bar", r["upper"] == 105.0,
          str(r["upper"]))
    r1 = run_pair(ms, bars, 100.0, decision_bar=1)
    check("L4 not usable one bar before activation", r1["upper"] != 105.0,
          str(r1["upper"]))


def test_L5_next_bar_penetration_is_outcome():
    bars = mk_bars([100, 106, 100, 100, 100, 100, 100, 100],
                   [100, 100, 100, 100, 100, 94, 100, 100], [100] * 8)
    ms = mk_master([
        ("U", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 105.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
        ("D", "AG", "CONFIRMED_SWING_LOW", "5m", -1, 95.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
    ])
    # decision at activation bar 0；penetration 在下一根 bar 1
    r = run_pair(ms, bars, 100.0, decision_bar=0)
    check("L5 next-bar strict penetration -> UP", r["label"] == m.UP,
          f"label={r['label']} up_pen={r['up_pen']}")
    check("L5 penetration bar == decision_bar + 1", r["up_pen"] == 1,
          str(r["up_pen"]))


def test_L6_activation_not_delayed():
    bars = mk_bars([100] * 8, [100] * 8, [100] * 8)
    ms = mk_master([("U", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 105.0,
                     pd.Timestamp("2025-01-02 09:05:00"), 3, pd.NaT)])
    lc = m.build_corrected_lifecycle(ms, bars)
    check("L6 activation_bar == 3", int(lc["activation_bar"][0]) == 3)
    check("L6 search starts at activation+1",
          int(lc["search_start"][0]) == 4, str(lc["search_start"].tolist()))


# ============================================================ parity
def test_P_parity_with_frozen_active_prices_chunk():
    """无 discontinuity 时，bar-index kernel 必须与 frozen datetime kernel 一致。"""
    bars = mk_bars([100] * 8, [100] * 8, [100] * 8)
    ms = mk_master([
        ("U", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 105.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
        ("D", "AG", "CONFIRMED_SWING_LOW", "5m", -1, 95.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
        ("U2", "AG", "PREV_TRADING_WEEK_HIGH", "TRADING_WEEK", +1, 110.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
    ])
    lc = m.build_corrected_lifecycle(ms, bars)
    # 让 U 在 bar 4 被突破
    bars["h"] = np.array([100, 100, 100, 100, 106, 100, 100, 100], float)
    lc = m.build_corrected_lifecycle(ms, bars)
    info = m.build_level_groups(ms, lc, bars)
    bi = np.arange(8, dtype=np.int64)
    mine = m.active_level_groups_chunk(bi, info)
    dt = m.to_dt64_ns(m.to_ns_int(bars["t"]) + m.BAR_NS)
    theirs = m.active_prices_chunk(dt[bi], info)
    check("P bar-index kernel == frozen active_prices_chunk (no discontinuity)",
          np.array_equal(mine, theirs),
          f"mine={mine.tolist()} theirs={theirs.tolist()}")


# ============================================================ censor source
def _censor_split(bars, up_pen, dn_pen, expiry):
    """复现 build_symbol_samples 里的 censor 拆分规则（用于合成测试）。"""
    lab = int(m.classify_pair_event(np.array([up_pen]), np.array([dn_pen]))[0])
    if lab == m.ROLL_RIGHT_CENSOR and expiry >= bars["n"]:
        lab = m.END_OF_DATA_RIGHT_CENSOR
    return lab


def test_C1_roll_right_censor():
    # disc at bar 4 -> 两侧在 expiry 前都没 penetration
    bars = mk_bars([100] * 8, [100] * 8, [100] * 8,
                   disc=[False, False, False, False, True, False, False, False])
    ms = mk_master([
        ("U", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 105.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
        ("D", "AG", "CONFIRMED_SWING_LOW", "5m", -1, 95.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
    ])
    r = run_pair(ms, bars, 100.0, decision_bar=1)
    check("C1 both no penetration + expiry<n -> ROLL_RIGHT_CENSOR",
          r["label"] == m.ROLL_RIGHT_CENSOR,
          f"label={r['label']} expiry={r['up_exp']}")
    check("C1 synthetic split rule agrees",
          _censor_split(bars, -1, -1, r["up_exp"]) == m.ROLL_RIGHT_CENSOR)


def test_C2_end_of_data_right_censor():
    bars = mk_bars([100] * 8, [100] * 8, [100] * 8)
    r = _censor_split(bars, -1, -1, 8)
    check("C2 both no penetration + expiry==n -> END_OF_DATA_RIGHT_CENSOR",
          r == m.END_OF_DATA_RIGHT_CENSOR, str(r))
    check("C2 ROLL + EOD == RIGHT_CENSOR_CODES",
          tuple(sorted(m.RIGHT_CENSOR_CODES)) == (m.ROLL_RIGHT_CENSOR,
                                                  m.END_OF_DATA_RIGHT_CENSOR))


def test_C3_one_side_penetrates_before_expiry():
    # U 在 bar 2 突破（< expiry 4），D 永不 -> resolved UP，不是 censor
    bars = mk_bars([100, 100, 106, 100, 100, 100, 100, 100],
                   [100] * 8, [100] * 8,
                   disc=[False, False, False, False, True, False, False, False])
    ms = mk_master([
        ("U", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 105.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
        ("D", "AG", "CONFIRMED_SWING_LOW", "5m", -1, 95.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
    ])
    r = run_pair(ms, bars, 100.0, decision_bar=1)
    check("C3 one side penetrates before expiry -> resolved UP",
          r["label"] == m.UP, f"label={r['label']}")
    check("C3 not classified as censor",
          r["label"] not in m.RIGHT_CENSOR_CODES, str(r["label"]))


def test_C4_pair_expiry_mismatch_stops():
    bars = mk_bars([100] * 8, [100] * 8, [100] * 8)
    ms = mk_master([
        ("U", "AG", "CONFIRMED_SWING_HIGH", "5m", +1, 105.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
        ("D", "AG", "CONFIRMED_SWING_LOW", "5m", -1, 95.0,
         pd.Timestamp("2025-01-02 09:05:00"), 0, pd.NaT),
    ])
    lc = m.build_corrected_lifecycle(ms, bars)
    # 人为让同一 decision bar 上两侧 group 的 expiry 不一致
    lc["expiry_bar"] = np.array([8, 4], dtype=np.int64)
    info = m.build_level_groups(ms, lc, bars)
    bi = np.array([1], dtype=np.int64)
    up_g = int(np.flatnonzero(
        (info["unique_price"] == 105.0) & (info["unique_side"] > 0))[0])
    dn_g = int(np.flatnonzero(
        (info["unique_price"] == 95.0) & (info["unique_side"] < 0))[0])
    eu = int(m.resolve_group_event(bi, np.array([up_g]), info)["expiry"][0])
    ed = int(m.resolve_group_event(bi, np.array([dn_g]), info)["expiry"][0])
    check("C4 synthetic expiry mismatch is detectable",
          eu != ed, f"up_exp={eu} dn_exp={ed}")
    # build_symbol_samples 里对 up_exp != dn_exp 直接 SystemExit；
    # 这里用同一条件断言守卫表达式成立
    check("C4 guard condition matches build_symbol_samples",
          (eu != ed) == True)  # noqa: E712


def test_classify_matrix():
    got = m.classify_pair_event(
        np.array([3, 5, -1, -1, 3]), np.array([5, 3, -1, 7, 3]))
    want = np.array([m.UP, m.DOWN, m.ROLL_RIGHT_CENSOR, m.DOWN, m.AMBIGUOUS])
    check("classify_pair_event matrix", np.array_equal(got, want),
          f"got={got.tolist()} want={want.tolist()}")


def main():
    test_A_upper_first()
    test_B_lower_first()
    test_C_same_bar()
    test_D_right_censor()
    test_E_consumed_not_active()
    test_F_not_yet_activated()
    test_G_same_group_conflict()
    test_F1_touch_then_strict_break()
    test_F2_multiple_touch_then_strict_break()
    test_F3_F4_strict_sides()
    test_F5_gap_cross()
    test_F6_discontinuity_truncation()
    test_L1_selected_identity_not_expired()
    test_L2_expiry_kills_level()
    test_L3_old_level_never_reappears()
    test_L4_activation_bar_is_immediate_boundary()
    test_L5_next_bar_penetration_is_outcome()
    test_L6_activation_not_delayed()
    test_P_parity_with_frozen_active_prices_chunk()
    test_C1_roll_right_censor()
    test_C2_end_of_data_right_censor()
    test_C3_one_side_penetrates_before_expiry()
    test_C4_pair_expiry_mismatch_stops()
    test_classify_matrix()
    print(f"\n==== {len(FAILS)} FAIL ====")
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
