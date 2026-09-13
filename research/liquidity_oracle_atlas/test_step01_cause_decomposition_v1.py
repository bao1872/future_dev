"""STEP-0.1 — cause decomposition 的确定性合成 / 契约测试。

运行：
  .venv/bin/python research/liquidity_oracle_atlas/test_step01_cause_decomposition_v1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import research.liquidity_oracle_atlas.experiment_step01_cause_decomposition_v1 as m  # noqa: E402
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


def mkgrp(prices, sides, acts, exps=None, pens=None):
    n = len(prices)
    if exps is None:
        exps = [8] * n
    if pens is None:
        pens = [-1] * n
    return dict(group_starts=np.arange(n, dtype=np.int64),
                group_lengths=np.ones(n, dtype=np.int64),
                price=np.asarray(prices, float),
                side=np.asarray(sides, np.int64),
                act=np.asarray(acts, np.int64),
                exp=np.asarray(exps, np.int64),
                pen=np.asarray(pens, np.int64))


# ---------------------------------------------------------------- 1
def test_1_new_activation():
    # G1 在 t0 不 active；activation 在 (t0, j]
    grp = mkgrp([105.0, 103.0, 95.0], [1, 1, -1], [1, 3, 1])
    close = np.full(8, 100.0)
    cat = m.classify_inward(grp, 1, 3, 1, 105.0, close, True)
    check("1 NEW_ACTIVATION", cat == "NEW_ACTIVATION", cat)


# ---------------------------------------------------------------- 2
def test_2_equality_eligibility():
    # G1 在 t0 已 active，但 price == close[t0]（严格 > 把它排除）
    grp = mkgrp([105.0, 100.0, 95.0], [1, 1, -1], [1, 1, 1])
    close = np.array([100.0, 100.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0])
    cat = m.classify_inward(grp, 1, 2, 1, 105.0, close, True)
    check("2 EQUALITY_ELIGIBILITY", cat == "EQUALITY_ELIGIBILITY", cat)


# ---------------------------------------------------------------- 3
def test_3_active_eligible_already_is_p0():
    # G1 在 t0 已 active、方向正确、且更 inward —— 理论上必须为 0
    grp = mkgrp([105.0, 103.0, 95.0], [1, 1, -1], [1, 1, 1])
    close = np.full(8, 100.0)
    cat = m.classify_inward(grp, 1, 2, 1, 105.0, close, True)
    check("3 ACTIVE_ELIGIBLE_ALREADY detected",
          cat == "ACTIVE_ELIGIBLE_ALREADY", cat)
    # lower 侧对称
    grp2 = mkgrp([105.0, 97.0, 95.0], [1, -1, -1], [1, 1, 1])
    cat2 = m.classify_inward(grp2, 1, 2, 1, 95.0, close, False)
    check("3b lower ACTIVE_ELIGIBLE_ALREADY detected",
          cat2 == "ACTIVE_ELIGIBLE_ALREADY", cat2)


# ---------------------------------------------------------------- 4
def test_4_touch_eligibility():
    # frozen 边界被 exact touch：close[j] == frozen price -> 临时不是候选
    grp = mkgrp([105.0, 103.0, 95.0], [1, 1, -1], [1, 1, 1])
    close = np.full(8, 100.0)
    close[4] = 105.0
    cat = m.classify_non_inward(grp, 1, 4, -1, 105.0, close)
    check("4 TOUCH_ELIGIBILITY (candidate disappears)",
          cat == "TOUCH_ELIGIBILITY", cat)
    cat2 = m.classify_non_inward(grp, 1, 4, 1, 105.0, close)
    check("4b TOUCH_ELIGIBILITY (outward group)",
          cat2 == "TOUCH_ELIGIBILITY", cat2)


# ---------------------------------------------------------------- 5
def test_5_touch_A_B_A_not_two_structural():
    price = np.array([105.0, 103.0])
    arr = np.array([0, 0, 1, 1, 0, 0, 0, 0], np.int64)
    # bar1 close==price[1] (103) -> A->B 是 touch；bar3 close==price[0] (105) -> B->A 是 touch
    close = np.array([100.0, 103.0, 100.0, 105.0, 100.0, 100.0, 100.0, 100.0])
    cum_all, cum_touch, sp = m.transition_flags(arr, close, price, 8)
    check("5 raw transitions == 2", int(cum_all[7]) == 2, str(cum_all[7]))
    check("5 touch transitions == 2", int(cum_touch[7]) == 2, str(cum_touch[7]))
    check("5 structural positions empty", len(sp) == 0, str(sp.tolist()))

    # 非 touch 的 A->B->A 必须算 2 次 structural
    close2 = np.full(8, 100.0)
    cum_all2, cum_touch2, sp2 = m.transition_flags(arr, close2, price, 8)
    check("5b non-touch raw == 2", int(cum_all2[7]) == 2)
    check("5b non-touch touch == 0", int(cum_touch2[7]) == 0)
    check("5b non-touch structural == 2", len(sp2) == 2, str(sp2.tolist()))


# ---------------------------------------------------------------- 6
def test_6_resolution_bar_change_excluded():
    cp = np.array([5], np.int64)
    t0 = np.array([1], np.int64)
    T = np.array([5], np.int64)
    f = m.next_after(cp, t0)
    hi = T - 1
    check("6 change at resolution bar not counted",
          not bool((f >= 0) & (f <= hi[0])), f"f={f} hi={hi}")


# ---------------------------------------------------------------- 7
def test_7_tb3_tb4_absent():
    p = m.CACHE / "step0_state_audit.parquet"
    if not p.exists():
        skip("7 TB3/TB4 absent from STEP-0.1 input", "step0 cache missing")
        return
    a = pd.read_parquet(p)
    check("7a no TB3/TB4 in STEP-0 audit input",
          set(a["block"].unique().tolist()) <= {"TB1", "TB2"},
          str(sorted(a["block"].unique())))
    check("7b tb1+tb2 counts == frozen baseline",
          int((a["block"] == L0.TRAIN_BLOCK).sum())
          + int((a["block"] == L0.TEST_BLOCK).sum()) == 239510,
          str(len(a)))


def main():
    test_1_new_activation()
    test_2_equality_eligibility()
    test_3_active_eligible_already_is_p0()
    test_4_touch_eligibility()
    test_5_touch_A_B_A_not_two_structural()
    test_6_resolution_bar_change_excluded()
    test_7_tb3_tb4_absent()
    print(f"\n==== {len(FAILS)} FAIL / {len(SKIPPED)} SKIP ====")
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
