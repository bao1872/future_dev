"""STEP-0 — synthetic / contract tests。

运行：
  .venv/bin/python research/liquidity_oracle_atlas/test_step0_step_reconfiguration_v1.py

覆盖用户 §18 的 A–I，以及 change-point 原语。
audit_states 是标量/向量同一实现，因此合成用例直接跑真实函数。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import research.liquidity_oracle_atlas.experiment_step0_step_reconfiguration_v1 as m  # noqa: E402
import research.liquidity_oracle_atlas.experiment_local_liquidity_transition_v0 as L0  # noqa: E402

FAILS = []
SKIPPED = []


def check(name, cond, detail=""):
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} :: {detail}")
        FAILS.append(name)


def skip(name, why):
    print(f"[SKIP] {name} :: {why}")
    SKIPPED.append(name)


def mk_seq(up_g, dn_g, up_px, dn_px, close, up_act=None, dn_act=None):
    n = len(close)
    z = [-1] * n
    return dict(
        upper_group=np.asarray(up_g, np.int64),
        lower_group=np.asarray(dn_g, np.int64),
        upper_price=np.asarray(up_px, float),
        lower_price=np.asarray(dn_px, float),
        upper_max_act=np.asarray(up_act if up_act is not None else z, np.int64),
        lower_max_act=np.asarray(dn_act if dn_act is not None else z, np.int64),
        n=n,
    )


CLOSE = [100.0] * 8


def run(seq, t0, T):
    return m.audit_states(
        seq, np.asarray(seq.get("close", CLOSE), float),
        np.array([t0], np.int64), np.array([T], np.int64))


def g(r, k):
    return r[k][0]


# --------------------------------------------------------------- primitives
def test_primitives():
    a = np.array([1, 1, 5, 5, 1, 1], np.int64)
    cp = m.change_positions(a)
    check("change_positions A->B->A", cp.tolist() == [2, 4], str(cp.tolist()))
    check("next_after", int(m.next_after(cp, np.array([0]))[0]) == 2)
    check("next_after none", int(m.next_after(cp, np.array([4]))[0]) == -1)
    check("count_in_range (0,4]",
          int(m.count_in_range(cp, np.array([0]), np.array([4]))[0]) == 2)


# --------------------------------------------------------------- A
def test_A_no_new_level():
    seq = mk_seq([1] * 8, [2] * 8, [105.0] * 8, [95.0] * 8, CLOSE)
    r = run(seq, 1, 6)
    check("A no new level -> NONE", not bool(g(r, "reconfigured_before_resolution")))
    check("A side == NONE", int(g(r, "first_reconfig_side")) == m.S_NONE)
    check("A zero transitions",
          int(g(r, "n_upper_pair_changes")) == 0
          and int(g(r, "n_lower_pair_changes")) == 0)


# --------------------------------------------------------------- B
def test_B_upper_inward():
    # bar3 新 upper group 5，price 103（close 100 < 103 < 105），activation=3 > t0=1
    seq = mk_seq([1, 1, 1, 5, 5, 5, 1, 1], [2] * 8,
                 [105, 105, 105, 103, 103, 103, 105, 105],
                 [95.0] * 8, CLOSE,
                 up_act=[-1, 1, 1, 3, 3, 3, 3, -1])
    r = run(seq, 1, 6)
    check("B upper change -> UPPER_ONLY",
          int(g(r, "first_reconfig_side")) == m.S_UPPER,
          str(g(r, "first_reconfig_side")))
    check("B UPPER_INWARD", bool(g(r, "upper_inward")))
    check("B not lower inward", not bool(g(r, "lower_inward")))
    check("B first_reconfig_bar == 3", int(g(r, "first_reconfig_bar_index")) == 3)
    check("B bars_to_first == 2", int(g(r, "bars_to_first_reconfig")) == 2)
    check("B provenance ok", bool(g(r, "provenance_upper_ok")))


# --------------------------------------------------------------- C
def test_C_lower_inward():
    # bar2 新 lower group 7，price 97（95 < 97 < close 100），activation=2 > 1
    seq = mk_seq([1] * 8, [2, 2, 7, 7, 7, 2, 2, 2],
                 [105.0] * 8,
                 [95, 95, 97, 97, 97, 95, 95, 95], CLOSE,
                 dn_act=[-1, 1, 2, 2, 2, 2, -1, -1])
    r = run(seq, 1, 6)
    check("C lower change -> LOWER_ONLY",
          int(g(r, "first_reconfig_side")) == m.S_LOWER,
          str(g(r, "first_reconfig_side")))
    check("C LOWER_INWARD", bool(g(r, "lower_inward")))
    check("C provenance ok", bool(g(r, "provenance_lower_ok")))


# --------------------------------------------------------------- D
def test_D_both_same_bar():
    seq = mk_seq([1, 1, 1, 5, 5, 5, 1, 1], [2, 2, 2, 7, 7, 7, 2, 2],
                 [105, 105, 105, 103, 103, 103, 105, 105],
                 [95, 95, 95, 97, 97, 97, 95, 95], CLOSE,
                 up_act=[-1, 1, 1, 3, 3, 3, 3, -1],
                 dn_act=[-1, 1, 1, 3, 3, 3, 3, -1])
    r = run(seq, 1, 6)
    check("D both sides at same bar -> BOTH",
          int(g(r, "first_reconfig_side")) == m.S_BOTH,
          str(g(r, "first_reconfig_side")))
    check("D both inward",
          bool(g(r, "upper_inward")) and bool(g(r, "lower_inward")))


# --------------------------------------------------------------- E
def test_E_provenance_fail():
    # 新 upper 是 inward，但 group 的 activation_bar == t0（不是 > t0）
    seq = mk_seq([1, 1, 1, 5, 5, 5, 1, 1], [2] * 8,
                 [105, 105, 105, 103, 103, 103, 105, 105],
                 [95.0] * 8, CLOSE,
                 up_act=[-1, 1, 1, 1, 1, 1, 1, -1])
    r = run(seq, 1, 6)
    check("E inward flagged", bool(g(r, "upper_inward")))
    check("E provenance FAIL", not bool(g(r, "provenance_upper_ok")),
          f"act={g(r, 'new_upper_activation_bar')} t0=1")


# --------------------------------------------------------------- F
def test_F_change_at_resolution_bar_not_counted():
    # 唯一变化发生在 resolution bar 本身 (T=4)，区间是 (1,3]
    seq = mk_seq([1, 1, 1, 1, 5, 5, 1, 1], [2] * 8,
                 [105, 105, 105, 105, 103, 103, 105, 105],
                 [95.0] * 8, CLOSE,
                 up_act=[-1, 1, 1, 1, 4, 4, 4, -1])
    r = run(seq, 1, 4)
    check("F change at resolution bar -> not pre-resolution reconfig",
          not bool(g(r, "reconfigured_before_resolution")),
          f"first={g(r, 'first_reconfig_bar_index')}")
    check("F zero transitions",
          int(g(r, "n_upper_pair_changes")) == 0,
          str(g(r, "n_upper_pair_changes")))


# --------------------------------------------------------------- G
def test_G_sustained_group_counts_once():
    # U: 1 1 5 5 5 -> 只有一次 transition
    seq = mk_seq([1, 1, 1, 5, 5, 5, 1, 1], [2] * 8,
                 [105, 105, 105, 103, 103, 103, 105, 105],
                 [95.0] * 8, CLOSE,
                 up_act=[-1, 1, 1, 3, 3, 3, 3, -1])
    r = run(seq, 1, 6)
    check("G sustained new group -> 1 transition",
          int(g(r, "n_upper_pair_changes")) == 1,
          str(g(r, "n_upper_pair_changes")))


# --------------------------------------------------------------- H
def test_H_A_B_A_counts_two():
    # U: 1 1 5 5 1 -> transitions at 3 and 5 => 2
    seq = mk_seq([1, 1, 1, 5, 5, 1, 1, 1], [2] * 8,
                 [105, 105, 105, 103, 103, 105, 105, 105],
                 [95.0] * 8, CLOSE,
                 up_act=[-1, 1, 1, 3, 3, 5, 5, 5])
    r = run(seq, 1, 6)
    check("H A->B->A -> 2 transitions",
          int(g(r, "n_upper_pair_changes")) == 2,
          str(g(r, "n_upper_pair_changes")))


# --------------------------------------------------------------- I
def test_I_tb3_tb4_excluded():
    missing = [s for s in L0.FULL_UNIV
               if not (m.CACHE / f"local0_samples_{s}.parquet").exists()]
    if missing:
        skip("I TB3/TB4 excluded from statistics", f"cache missing {missing}")
        return
    frames = [pd.read_parquet(m.CACHE / f"local0_samples_{s}.parquet")
              for s in L0.FULL_UNIV]
    cached = pd.concat(frames, ignore_index=True)
    states = cached[(cached["block"].isin([L0.TRAIN_BLOCK, L0.TEST_BLOCK]))
                    & cached["label"].isin(L0.RESOLVED_CODES)]
    check("I no TB3/TB4 in STEP-0 state universe",
          not states["block"].isin(["TB3", "TB4"]).any())
    check("I TB3/TB4 still present in cache (guard meaningful)",
          int(cached["block"].isin(["TB3", "TB4"]).sum()) > 0)
    check("I full-universe counts reproduce frozen baseline",
          dict(tb1_resolved=int((states["block"] == L0.TRAIN_BLOCK).sum()),
               tb2_resolved=int((states["block"] == L0.TEST_BLOCK).sum()))
          == m.EXPECT,
          str(dict(tb1=int((states["block"] == L0.TRAIN_BLOCK).sum()),
                   tb2=int((states["block"] == L0.TEST_BLOCK).sum()))))


def main():
    test_primitives()
    test_A_no_new_level()
    test_B_upper_inward()
    test_C_lower_inward()
    test_D_both_same_bar()
    test_E_provenance_fail()
    test_F_change_at_resolution_bar_not_counted()
    test_G_sustained_group_counts_once()
    test_H_A_B_A_counts_two()
    test_I_tb3_tb4_excluded()
    print(f"\n==== {len(FAILS)} FAIL / {len(SKIPPED)} SKIP ====")
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
