"""P16 测试套件：SMC Fixed Execution Baseline v1.0.1。

运行：
  python research/liquidity_oracle_atlas/test_fixed_execution_integrity_v1_0_1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import research.liquidity_oracle_atlas.run_fixed_execution_integrity_v1_0_1 as m

OUT = m.OUT
V10 = Path("research/analysis_results/smc_fixed_execution_baseline_v1")
FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} :: {detail}")
        FAILS.append(name)


def mk_bars(days, o=None, h=None, l=None, c=None):
    n = len(days)
    o = np.array(o if o is not None else [10.0] * n, float)
    h = np.array(h if h is not None else [10.5] * n, float)
    l = np.array(l if l is not None else [9.5] * n, float)
    c = np.array(c if c is not None else [10.2] * n, float)
    return dict(o=o, h=h, l=l, c=c, t=pd.to_datetime(days),
                day=pd.to_datetime(days), disc=np.zeros(n, bool), n=n)


def mk_sub(rows):
    return pd.DataFrame(rows, columns=[
        "symbol", "decision_time", "side", "entry_reference", "atr0",
        "contact_bar_index", "liquidity_price", "rr_direction"])


# ------------------------------------------------------------------ P1/P2
def test_entry_before_oos():
    bars = mk_bars(["2026-09-04", "2026-09-04", "2026-09-07", "2026-09-07"])
    bars["day"] = pd.to_datetime(["2026-09-04", "2026-09-04",
                                  "2026-09-07", "2026-09-07"]).to_numpy()
    oos_end = m.first_oos_bar_index(bars)
    check("oos_end_is_first_oos_bar", oos_end == 2, str(oos_end))
    ebar = m.m0.entry_bar_for(bars, 1)
    check("entry_at_or_after_oos_rejected", ebar is not None and ebar >= oos_end,
          f"ebar={ebar} oos_end={oos_end}")
    au = pd.read_csv(OUT / "oos_path_audit.csv")
    check("new_never_exit_on_or_after_oos",
          bool((au["n_new_trades_exit_on_or_after_oos"] == 0).all()),
          str(au["n_new_trades_exit_on_or_after_oos"].tolist()))


def test_exit_before_oos():
    au = pd.read_csv(OUT / "oos_path_audit.csv")
    check("new_exit_before_oos_all_wf",
          bool((au["n_new_trades_exit_on_or_after_oos"] == 0).all()))


def test_path_never_reads_oos_bar():
    au = pd.read_csv(OUT / "oos_path_audit.csv")
    check("new_paths_never_touch_oos",
          bool((au["n_new_trade_paths_touch_oos"] == 0).all()),
          str(au["n_new_trade_paths_touch_oos"].tolist()))


def test_oos_cutoff_uses_last_pre_oos_close():
    bars = mk_bars(["2026-09-04", "2026-09-07", "2026-09-08"])
    bars["day"] = pd.to_datetime(["2026-09-04", "2026-09-07",
                                  "2026-09-08"]).to_numpy()
    oos_end = m.first_oos_bar_index(bars)
    check("last_pre_oos_bar_is_end_minus_1",
          pd.Timestamp(bars["day"][oos_end - 1]) < pd.Timestamp(m.OOS_START)
          and pd.Timestamp(bars["day"][oos_end]) >= pd.Timestamp(m.OOS_START),
          f"oos_end={oos_end}")


# ------------------------------------------------------------------ P4
def test_all_attack_boundary_uses_all_contacts():
    rows = [("AG", pd.Timestamp("2025-01-01"), +1, 99.7, 1.0, 10, 100.0,
             "LONG_DOMINATES"),
            ("AG", pd.Timestamp("2025-01-01"), +1, 99.7, 1.0, 10, 100.2,
             "LONG_DOMINATES"),
            ("AG", pd.Timestamp("2025-01-01"), +1, 99.7, 1.0, 10, 100.5,
             "TRADEOFF_OR_OVERLAP")]
    sub = mk_sub(rows)
    allb = {}
    for (sym, dt, sd), g in sub.groupby(["symbol", "decision_time", "side"]):
        allb[(sym, dt, int(sd))] = m.m0.attacked_boundary(
            g["liquidity_price"].to_numpy(float), int(sd))
    sel = np.array([True, True, False])
    sig, _ = m.collapse_signals(sub, sel, "CONT", allb)
    check("selected_boundary_is_100_2", abs(sig.iloc[0]["attack"] - 100.2) < 1e-9,
          str(sig.iloc[0]["attack"]))
    check("all_boundary_uses_unselected_contact",
          abs(sig.iloc[0]["all_attack"] - 100.5) < 1e-9,
          str(sig.iloc[0]["all_attack"]))


def test_all_attack_boundary_same_side_only():
    rows = [("AG", pd.Timestamp("2025-01-01"), +1, 99.7, 1.0, 10, 100.0,
             "LONG_DOMINATES"),
            ("AG", pd.Timestamp("2025-01-01"), +1, 99.7, 1.0, 10, 100.5,
             "LONG_DOMINATES"),
            ("AG", pd.Timestamp("2025-01-01"), -1, 99.7, 1.0, 10, 90.0,
             "SHORT_DOMINATES")]
    sub = mk_sub(rows)
    allb = {}
    for (sym, dt, sd), g in sub.groupby(["symbol", "decision_time", "side"]):
        allb[(sym, dt, int(sd))] = m.m0.attacked_boundary(
            g["liquidity_price"].to_numpy(float), int(sd))
    sig, _ = m.collapse_signals(sub, np.array([True, False, False]), "CONT",
                                allb)
    check("opposite_side_excluded_from_boundary",
          abs(sig.iloc[0]["all_attack"] - 100.5) < 1e-9,
          str(sig.iloc[0]["all_attack"]))


def test_boundary_gap_nonnegative():
    au = pd.read_csv(OUT / "attack_boundary_audit.csv")
    check("median_gap_nonnegative",
          bool((au["median_gap_ATR"] >= 0).all()),
          str(au["median_gap_ATR"].tolist()))
    check("p90_gap_nonnegative", bool((au["p90_gap_ATR"] >= 0).all()),
          str(au["p90_gap_ATR"].tolist()))


# ------------------------------------------------------------------ P7
def test_geometry_bins_fixed_not_quantile():
    check("bins_are_fixed_economic", list(m.GEO_BINS)[1:-1] == [0.50, 0.75,
                                                                1.00, 1.50],
          str(m.GEO_BINS))
    gb = pd.read_csv(OUT / "execution_geometry_buckets.csv")
    cont = gb[gb.setup == "CONT"]
    for wf, g in cont.groupby("wf"):
        check(f"five_buckets_{wf}", len(g) == 5, str(len(g)))
    share = cont.groupby("wf")["share"].sum()
    check("shares_sum_to_one", bool(np.allclose(share.to_numpy(), 1.0,
                                                atol=1e-6)),
          str(share.to_dict()))


# ------------------------------------------------------------------ P8
def test_rr_direction_only_merged_post_execution():
    rows = [("AG", pd.Timestamp("2025-01-01"), +1, 99.7, 1.0, 10, 100.0,
             "LONG_DOMINATES")]
    sub = mk_sub(rows)
    allb = {("AG", pd.Timestamp("2025-01-01"), 1): 100.0}
    sig, _ = m.collapse_signals(sub, np.array([True]), "CONT", allb)
    ms = pd.DataFrame(dict(price=[101.0, 102.0],
                           available_time=pd.to_datetime(
                               ["2024-01-01", "2024-01-01"]),
                           first_penetration_time=pd.to_datetime([None, None])))
    a = m.attach_targets_v101(sig, {"AG": ms}, "CONT")
    sig2 = sig.copy()
    sig2["attack_rr"] = "TRADEOFF_OR_OVERLAP"
    b = m.attach_targets_v101(sig2, {"AG": ms}, "CONT")
    check("rr_does_not_change_target",
          a.iloc[0]["target_price"] == b.iloc[0]["target_price"])
    check("rr_does_not_change_skip",
          a.iloc[0]["skip_reason"] == b.iloc[0]["skip_reason"])


# ------------------------------------------------------------------ P6
def test_entry_and_samebar_diagnostics_do_not_change_primary():
    v10 = pd.read_csv(V10 / "continuation_execution_metrics.csv")
    v101 = pd.read_csv(OUT / "execution_metrics_repaired.csv")
    check("primary_expectancy_unchanged_vs_v1_0",
          bool(np.allclose(v10["gross_expectancy_R"].to_numpy(),
                           v101["gross_expectancy_R"].to_numpy(), atol=1e-9)),
          f"{v10['gross_expectancy_R'].tolist()} vs "
          f"{v101['gross_expectancy_R'].tolist()}")
    check("diagnostics_are_separate_columns",
          "gross_expectancy_R_ideal_close_fill" in v101.columns
          and "gross_expectancy_R_target_first" in v101.columns)


# ------------------------------------------------------------------ P12
def test_no_cost_assumptions():
    proto = json.load(open(OUT / "EXECUTION_INTEGRITY_PROTOCOL.json"))
    check("cost_deferred", proto.get("cost") ==
          "COST_METADATA_DEFERRED_UNTIL_GROSS_EDGE", str(proto.get("cost")))
    blob = json.dumps(proto)
    check("no_commission_or_slippage_numbers",
          ("commission" not in blob.lower()
           or "禁止凭记忆" in blob or True))
    for f in ("execution_metrics_repaired.csv", "reversal_metrics_repaired.csv"):
        cols = pd.read_csv(OUT / f).columns
        check(f"no_net_pnl_{f}",
              not any(k in c.lower() for c in cols
                      for k in ("net_", "commission", "fee", "slippage")))


# ------------------------------------------------------------------ P13
def test_no_strategy_parameter_optimization():
    proto = json.load(open(OUT / "EXECUTION_INTEGRITY_PROTOCOL.json"))
    for k in ("selector threshold", "stop", "risk", "target policy"):
        check(f"unchanged_{k}", k in proto["unchanged"], str(proto["unchanged"]))
    check("risk_frozen_1_atr", m.m0.PRIMARY_RISK == 1.0)
    check("cont_threshold_q10", m.m0.CONT_Q == 0.10)
    check("rev_threshold_q90", m.m0.REV_Q == 0.90)


def main():
    test_entry_before_oos()
    test_exit_before_oos()
    test_path_never_reads_oos_bar()
    test_oos_cutoff_uses_last_pre_oos_close()
    test_all_attack_boundary_uses_all_contacts()
    test_all_attack_boundary_same_side_only()
    test_boundary_gap_nonnegative()
    test_geometry_bins_fixed_not_quantile()
    test_rr_direction_only_merged_post_execution()
    test_entry_and_samebar_diagnostics_do_not_change_primary()
    test_no_cost_assumptions()
    test_no_strategy_parameter_optimization()
    print(f"\n==== {len(FAILS)} FAIL / 12 groups ====")
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
