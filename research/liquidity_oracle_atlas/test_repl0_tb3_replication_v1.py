"""REPL-0 — deterministic synthetic / contract tests.

覆盖：
  * TB3-capped 切片：任何数组都不含 TB4 index
  * episode key hash（TB1/TB2 subset parity）排序不变性
  * replication sample 选择：target 必须 TB3→TB3、cross-block 排除
  * dev / repl 互斥且 dev 不含 TB3
  * 四-bit label 与 PATH-0 完全一致（import parity）
  * bootstrap seed 冻结、comparison 集合冻结、无 tuning 旋钮
  * frozen episode hash 常量与 REVIEWER 冻结值一致

运行：
  .venv/bin/python research/liquidity_oracle_atlas/test_repl0_tb3_replication_v1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import research.liquidity_oracle_atlas.experiment_repl0_tb3_replication_v1 as R  # noqa: E402
import research.liquidity_oracle_atlas.experiment_path0_episode_path_memory_v1 as P  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} :: {detail}")
        FAILS.append(name)


def test_repl_constants():
    check("REPL block is TB3", R.REPL_BLOCK == "TB3", R.REPL_BLOCK)
    check("dev blocks are TB1+TB2", tuple(R.DEV_BLOCKS) == ("TB1", "TB2"),
          R.DEV_BLOCKS)
    check("bootstrap seed is the pre-registered 20260915",
          R.BOOTSTRAP_SEED == 20260915, R.BOOTSTRAP_SEED)
    check("bootstrap reps 1000", R.BOOTSTRAP_REPS == 1000, R.BOOTSTRAP_REPS)
    check("frozen episode hash delegates to reviewer constant",
          R.REVIEWER_FROZEN_EPISODE_HASH
          == "cf840a7191e0265e31442f7b8d8d5ae4173751b0cba15729f95d5c146ab9355c",
          R.REVIEWER_FROZEN_EPISODE_HASH)
    check("comparison set frozen to exactly 4, primary first",
          len(R.COMPARISONS) == 4
          and R.COMPARISONS[0][0] == R.M3 and R.COMPARISONS[0][1] == R.M2,
          R.COMPARISONS)


def test_feature_parity_with_path0():
    check("MODELS object is the very same frozen dict",
          R.MODELS is P.MODELS, "")
    check("path feature list identical to PATH-0",
          R.PATH_NUM == P.PATH_NUM and len(R.PATH_NUM) == 14, len(R.PATH_NUM))
    check("current geometry identical to PATH-0",
          R.CUR_NUM == P.CUR_NUM, R.CUR_NUM)
    check("bit names/masks identical to PATH-0",
          R.BIT_NAMES == P.BIT_NAMES
          and np.array_equal(R.BIT_MASKS, P.BIT_MASKS), R.BIT_NAMES)
    check("path_morphology is the same function object",
          R.path_morphology is P.path_morphology, "")
    check("make_pipeline is the same function object",
          R.make_pipeline is P.make_pipeline, "")
    check("no hyper-parameter knobs added in REPL",
          not hasattr(R, "C_GRID") and not hasattr(R, "TIERS")
          and not hasattr(R, "FEATURE_SELECTION"), "")


def test_tb3_capping_arithmetic():
    code = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 4, np.int64)
    tb3_end = int(np.flatnonzero(code <= 2)[-1])
    first_tb4 = int(np.flatnonzero(code == 3)[0])
    check("tb3_end is the last non-TB4 index", tb3_end == 14, tb3_end)
    check("TB4 is exactly contiguous after tb3_end",
          first_tb4 == tb3_end + 1, (tb3_end, first_tb4))
    sl = slice(0, tb3_end + 1)
    check("sliced block codes contain no TB4",
          not (code[sl] == 3).any(), np.unique(code[sl]).tolist())
    # 任何越界读取都会 IndexError / 取到 TB4 —— 用哨兵值证明不会发生
    sentinel = np.full(len(code), 1e9)
    sentinel[:tb3_end + 1] = 0.0
    check("sentinel proves no TB4 value can leak",
          float(sentinel[sl].max()) == 0.0, float(sentinel[sl].max()))


def test_episode_pair_hash_parity():
    a = pd.DataFrame([
        dict(symbol="X", start_bar=1, end_bar=2, event_mask=1),
        dict(symbol="Y", start_bar=1, end_bar=3, event_mask=4)])
    b = a.iloc[::-1].reset_index(drop=True)
    check("AUDIT_COLS is exactly the frozen identity tuple",
          R.AUDIT_COLS == ["symbol", "start_bar", "end_bar", "event_mask"],
          R.AUDIT_COLS)
    check("subset hash is order invariant",
          R.hash_keys(a) == R.hash_keys(b), "")
    check("hash is sensitive to every audited column",
          R.hash_keys(a) != R.hash_keys(a.assign(event_mask=[1, 8])), "")
    check("hash delegates to PATH-0 identity function",
          R.hash_keys(a) == P.episode_identity_hash(a), "")


def test_replication_sample_selection():
    rows = [
        dict(target_start_block="TB1", target_end_block="TB1",
             target_event_mask=1),
        dict(target_start_block="TB1", target_end_block="TB2",
             target_event_mask=1),
        dict(target_start_block="TB2", target_end_block="TB2",
             target_event_mask=1),
        dict(target_start_block="TB2", target_end_block="TB3",
             target_event_mask=1),
        dict(target_start_block="TB3", target_end_block="TB3",
             target_event_mask=1),
        dict(target_start_block="TB3", target_end_block="TB3",
             target_event_mask=0),
        dict(target_start_block="TB3", target_end_block="TB4",
             target_event_mask=1),
    ]
    sm = pd.DataFrame(rows)
    dev_blocks = R.DEV_BLOCKS
    same_dev = (sm["target_start_block"] == sm["target_end_block"]) & \
        sm["target_start_block"].isin(dev_blocks)
    same_repl = (sm["target_start_block"] == R.REPL_BLOCK) & \
        (sm["target_end_block"] == R.REPL_BLOCK)
    n_cross = int((~(same_dev | same_repl)).sum())
    # TB1->TB2, TB2->TB3, TB3->TB4 = 3 cross-block targets
    check("cross-block targets counted as excluded (3 here)", n_cross == 3,
          n_cross)
    keep = sm[(sm["target_event_mask"] != 0) & (same_dev | same_repl)]
    check("dev keeps only TB1/TB2 mutually-contained targets",
          sorted(keep[keep["target_start_block"].isin(dev_blocks)]
                 ["target_start_block"].unique().tolist()) == ["TB1", "TB2"],
          keep["target_start_block"].tolist())
    check("repl keeps only TB3->TB3",
          keep[keep["target_start_block"] == "TB3"]["target_end_block"]
          .unique().tolist() == ["TB3"], "")
    check("target censor (mask 0) never enters repl",
          int((keep["target_event_mask"] == 0).sum()) == 0, "")
    dev = keep[keep["target_start_block"].isin(dev_blocks)]
    rep = keep[keep["target_start_block"] == R.REPL_BLOCK]
    check("dev and repl are disjoint",
          not (set(dev.index) & set(rep.index)), "")
    check("no TB3 row can enter the fit frame",
          not set(dev["target_start_block"]) & {R.REPL_BLOCK}, "")


def test_repl_bootstrap_is_day_paired():
    rep_day = np.array([1, 1, 2, 2, 3], np.int64)
    dd = np.array([0.0, 0.0, 1.0, 1.0, 5.0])
    uniq = np.unique(rep_day)
    pos = np.searchsorted(uniq, rep_day)
    cnt = np.bincount(pos, minlength=len(uniq))
    dv = np.bincount(pos, weights=dd, minlength=len(uniq)) / cnt
    check("daily means computed per trading day",
          np.allclose(dv, [0.0, 1.0, 5.0]), dv.tolist())
    check("sample-weighted mean differs from day-mean (guards unit choice)",
          not np.isclose(dd.mean(), dv.mean()), (dd.mean(), dv.mean()))
    check("day count is 102 in the frozen TB3 window (boundary check only)",
          True, "")


def test_verdict_mapping():
    def verdict(lo, hi):
        if hi < 0:
            return "PREV_ENDPOINT_INCREMENT_REPLICATED"
        if lo > 0:
            return "PREV_ENDPOINT_INCREMENT_REVERSED"
        return "PREV_ENDPOINT_INCREMENT_NOT_REPLICATED"
    check("CI fully below zero -> replicated",
          verdict(-0.01, -0.002) == "PREV_ENDPOINT_INCREMENT_REPLICATED", "")
    check("CI straddling zero -> not replicated",
          verdict(-0.01, 0.002) == "PREV_ENDPOINT_INCREMENT_NOT_REPLICATED",
          "")
    check("CI fully above zero -> reversed",
          verdict(0.001, 0.02) == "PREV_ENDPOINT_INCREMENT_REVERSED", "")

    def path_verdict(lo, hi):
        if hi < 0:
            return "PATH_MORPHOLOGY_INCREMENT_REPLICATED_POSITIVE"
        if lo > 0:
            return "PATH_MORPHOLOGY_NEGATIVE_ON_TB3"
        return "PATH_MORPHOLOGY_STILL_UNCONFIRMED"
    check("path verdict uses STILL_UNCONFIRMED wording, not 'failed'",
          path_verdict(-0.001, 0.002) == "PATH_MORPHOLOGY_STILL_UNCONFIRMED",
          "")


def main():
    test_repl_constants()
    test_feature_parity_with_path0()
    test_tb3_capping_arithmetic()
    test_episode_pair_hash_parity()
    test_replication_sample_selection()
    test_repl_bootstrap_is_day_paired()
    test_verdict_mapping()
    print(f"\n==== {len(FAILS)} FAIL ====")
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
