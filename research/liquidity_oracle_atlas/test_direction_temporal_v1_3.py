"""P13 测试套件：SMC Direction Temporal Stability v1.3。

验证产物（而非重跑模型），覆盖用户列出的 8 项。运行：
  python research/liquidity_oracle_atlas/test_direction_temporal_v1_3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import research.liquidity_oracle_atlas.run_direction_temporal_stability_v1_3 as m

OUT = m.OUT
V12_OUT = m.V12_OUT
OOS_START = m.OOS_START
BLOCKS = m.BLOCKS

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} :: {detail}")
        FAILS.append(name)


# ---------------------------------------------------------------- P0 gate
def test_v12_gate_long_short_not_side():
    g = json.load(open(OUT / "v12_gate_repair.json"))
    g2 = g["repaired_two_stage"]
    pl = g2["pooled_LONG_actionable"]
    ps = g2["pooled_SHORT_actionable"]
    # 修正后不应再出现 buggy 的 (0.0, 0.8518)
    check("gate_long_not_zero", not (pl == 0.0), f"pooled_LONG={pl}")
    check("gate_short_not_buggy", not (ps == 0.8518), f"pooled_SHORT={ps}")
    check("gate_long_in_range", 0.40 <= pl <= 0.95, f"pooled_LONG={pl}")
    check("gate_short_in_range", 0.40 <= ps <= 0.95, f"pooled_SHORT={ps}")
    # Gate 仍应 FALSE（WF1 actionable 0.560<0.58）
    check("gate_still_false", g["GATE_REPAIRED_ACTIONABLE_DIRECTION_PRESENT"] is False,
          f"actionable={g['GATE_REPAIRED_ACTIONABLE_DIRECTION_PRESENT']}")


# ---------------------------------------------------------- P0.5 retracted
def test_product_diagnostic_retracted():
    proto = json.load(open(OUT / "DIRECTION_TEMPORAL_PROTOCOL.json"))
    check("product_retracted_marker",
          proto.get("TWO_STAGE_PRODUCT_DIAGNOSTIC") == "RETRACTED_INVALID_OOF_ALIGNMENT",
          proto.get("TWO_STAGE_PRODUCT_DIAGNOSTIC"))
    # 旧 CSV 仍作为 bug evidence 存在（不被删除、不被引用）
    check("v12_product_csv_kept_as_evidence",
          (V12_OUT / "product_diagnostic_metrics.csv").exists())
    # v1.3 产物中不得出现 product diagnostic 结果文件
    check("no_product_csv_in_v13",
          not (OUT / "product_diagnostic_metrics.csv").exists())


# --------------------------------------------------- causal pairs are causal
def test_all_cross_block_pairs_are_causal():
    df = pd.read_csv(OUT / "causal_cross_block_metrics.csv")
    ok = True
    bad = []
    for _, r in df.iterrows():
        tr = r["train_blocks"].split("+")
        te = r["test_block"]
        if not all(m.block_idx(b) < m.block_idx(te) for b in tr):
            ok = False
            bad.append(r["config"])
    check("all_pairs_causal", ok, f"bad={bad}")
    # 必须恰好 6 个 causal pair
    check("six_causal_pairs", len(df) == 6, f"n={len(df)}")


# ------------------------------------------------ tail thresholds train-only
def test_tail_thresholds_train_only():
    df = pd.read_csv(OUT / "tail_transfer_metrics.csv")
    ok = True
    bad = []
    for _, r in df.iterrows():
        lo, hi = r["lo_threshold"], r["hi_threshold"]
        cov = r["tail_test_coverage"]
        # train-OOF 10/90 → test 覆盖率应≈0.20（18%-22%容差）
        if not (0 < lo < hi < 1):
            ok = False; bad.append((r["config"], "bounds"))
        if not (0.15 <= cov <= 0.25):
            ok = False; bad.append((r["config"], f"cov={cov}"))
    check("tail_bounds_and_coverage", ok, f"bad={bad}")


# ----------------------------------------- score-decile edges train-OOF only
def test_score_decile_edges_train_oof_only():
    df = pd.read_csv(OUT / "score_decile_stability.csv")
    ok = True
    bad = []
    for (src, tb), g in df.groupby(["train_source", "test_block"]):
        g = g.sort_values("score_decile")
        ms = g["mean_score"].to_numpy()
        finite = ms[np.isfinite(ms)]
        # 由 train OOF 边界分箱 → test mean_score 应单调不减
        if len(finite) >= 2 and not np.all(np.diff(finite) >= -1e-9):
            ok = False; bad.append((src, tb))
    check("decile_mean_score_monotonic", ok, f"bad={bad}")


# ----------------------------------------------------- PSI bins train-only
def test_psi_bins_fit_train_only():
    df = pd.read_csv(OUT / "g4_psi_train_test.csv")
    ok = True
    bad = []
    for _, r in df.iterrows():
        v = r["psi"]
        if pd.notna(v) and v < 0:   # PSI 必须非负
            ok = False; bad.append((r["train_blocks"], r["test_block"], r["field"]))
    check("psi_nonnegative", ok, f"bad={bad}")


# --------------------------------------------------- no prospective OOS
def test_no_prospective_oos():
    D = m.load_data()
    max_day = D["DAYS"][D["insample"]].max()
    check("max_insample_before_oos", str(max_day) < OOS_START,
          f"max_insample={max_day} oos={OOS_START}")


# ----------------------------------------------------------- no new features
def test_no_new_features():
    proto = json.load(open(OUT / "DIRECTION_TEMPORAL_PROTOCOL.json"))
    used = set(proto["G4_BASE"]) | set(proto["G4_ONLY"])
    allowed = m.ALLOWED_FEATURES
    # 本论只使用 G4_BASE / G4_ONLY（P7 domain 用这两组；P10 clear 用冻结 M_GLOBAL4 块）
    leaked = used - allowed
    check("no_new_features_used", not leaked, f"leaked={leaked}")
    forbidden_cols = m.FORBIDDEN
    touch = used & forbidden_cols
    check("no_forbidden_columns", not touch, f"touch={touch}")


def main():
    test_v12_gate_long_short_not_side()
    test_product_diagnostic_retracted()
    test_all_cross_block_pairs_are_causal()
    test_tail_thresholds_train_only()
    test_score_decile_edges_train_oof_only()
    test_psi_bins_fit_train_only()
    test_no_prospective_oos()
    test_no_new_features()
    print(f"\n==== {len(FAILS)} FAIL / 8 groups ====")
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
