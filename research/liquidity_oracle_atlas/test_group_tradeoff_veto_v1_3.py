"""P 测试套件：SMC Group-Level Conditional Tradeoff Veto v1.3。

运行（实验已跑完、产物已生成后）：
  python research/liquidity_oracle_atlas/test_group_tradeoff_veto_v1_3.py

约定：import 仅加载模块级常量（不重新加载 15 品种原始行情）；
需要产物的断言依赖已生成的 analysis_results。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import research.liquidity_oracle_atlas.run_group_tradeoff_veto_v1_3 as m

OUT = m.OUT
FAILS = []

# 冻结规格禁止新增的行情特征 token
FORBIDDEN_TOKENS = ("fvg", "ob_", "trend", "volume", "precontact")
EXPECTED_AGG = {
    "n_selected_contacts", "outer_p_clear", "outer_p_rev",
    "min_p_clear", "mean_p_clear", "max_p_clear",
    "min_p_rev", "mean_p_rev", "max_p_rev",
}


def check(name, cond, detail=""):
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} :: {detail}")
        FAILS.append(name)


def csv(name):
    return pd.read_csv(OUT / name)


# ---------------------------------------------------------- feature routing
def test_feature_blocks_routing():
    check("G0_OUTERMOST_equals_G4_BASE",
          list(m.G0_OUTERMOST) == list(m.G4_BASE),
          f"{m.G0_OUTERMOST} vs {list(m.G4_BASE)}")
    check("G1_has_9_aggregates",
          len(m.G1_ARCH_AGG) == len(m.G0_OUTERMOST) + 9,
          f"{len(m.G1_ARCH_AGG)} vs {len(m.G0_OUTERMOST) + 9}")
    for c in m.G1_AGG:
        check(f"aggregate_routed_ordinary:{c}", c in m.rp.ORDINARY_NUMERIC)


def test_no_new_market_features():
    all_cols = set(m.G0_OUTERMOST) | set(m.G1_ARCH_AGG)
    bad = [c for c in all_cols if any(t in c.lower() for t in FORBIDDEN_TOKENS)]
    check("no_fvg_ob_trend_volume_precontact_features", not bad, str(bad))
    check("aggregates_are_selector_metadata", set(m.G1_AGG) == EXPECTED_AGG,
          str(set(m.G1_AGG)))


def test_toxic_label_not_in_features():
    all_cols = set(m.G0_OUTERMOST) | set(m.G1_ARCH_AGG)
    check("rr_direction_not_in_model_cols", "rr_direction" not in all_cols)
    check("attack_rr_not_in_model_cols", "attack_rr" not in all_cols)


def test_frozen_thresholds():
    check("clear_precision_frozen_0_85", abs(m.CLEAR_PRECISION - 0.85) < 1e-9)
    check("continuation_q_frozen_0_10", abs(m.CONT_Q - 0.10) < 1e-9)
    check("primary_risk_frozen_1_0", abs(m.PRIMARY_RISK - 1.0) < 1e-9)
    check("oos_start_frozen", m.OOS_START == m.m101.OOS_START)


# ---------------------------------------------------------- outputs (post-run)
def test_baseline_reproduction():
    audit = json.loads((OUT / "GROUP_TRADEOFF_AUDIT.json").read_text(
        encoding="utf-8"))
    br = audit["baseline_reproduction"]
    check("matches_v1_0_1", br["matches_v1_0_1"] is True, str(br))
    exp_e = [-0.0420, 0.0253, -0.0376]
    exp_t = [614, 856, 848]
    ok_e = all(abs(a - b) < 1e-3 for a, b in zip(br["per_wf_E_R"], exp_e))
    ok_t = all(a == b for a, b in zip(br["per_wf_trades"], exp_t))
    check("baseline_E_R_exact", ok_e, str(br["per_wf_E_R"]))
    check("baseline_trades_exact", ok_t, str(br["per_wf_trades"]))


def test_expected_outputs_present():
    for f in ("baseline_reproduction.csv", "baseline_group_oof_audit.csv",
              "group_toxic_metrics.csv", "group_toxic_thresholds.csv",
              "group_veto_mechanism.csv", "group_veto_execution.csv",
              "group_unknown_diagnostic.csv", "oos_guard_audit.csv",
              "group_veto_bootstrap.csv", "GROUP_TRADEOFF_AUDIT.json",
              "GROUP_TRADEOFF_PROTOCOL.json",
              "SMC_GROUP_TRADEOFF_VETO_V1_3.md"):
        check(f"output_present:{f}", (OUT / f).exists(), f)


def test_oos_guard_zero():
    o = csv("oos_guard_audit.csv")
    check("oos_guard_all_zero",
          int(o["n_exit_on_or_after_oos"].sum()) == 0,
          str(o.to_dict("records")))


def test_signal_level_audit_present():
    audit = json.loads((OUT / "GROUP_TRADEOFF_AUDIT.json").read_text(
        encoding="utf-8"))
    check("audit_final_present",
          "GROUP_LEVEL_TRADEOFF_VETO_EDGE" in audit["final"])


def main():
    test_feature_blocks_routing()
    test_no_new_market_features()
    test_toxic_label_not_in_features()
    test_frozen_thresholds()
    if (OUT / "GROUP_TRADEOFF_AUDIT.json").exists():
        test_baseline_reproduction()
        test_expected_outputs_present()
        test_oos_guard_zero()
        test_signal_level_audit_present()
    else:
        print("[SKIP] post-run assertions: run main() first")
    print(f"\n==== {len(FAILS)} FAIL ====")
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
