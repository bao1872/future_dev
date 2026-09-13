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
    print(f"\n==== {len(FAILS)} FAIL ====")
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
