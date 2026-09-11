"""P15 测试套件：SMC Execution Selection Purity Gate v1.1。

运行：
  python research/liquidity_oracle_atlas/test_execution_selection_purity_v1_1.py
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import research.liquidity_oracle_atlas.run_execution_selection_purity_v1_1 as m

OUT = m.OUT
V101 = Path("research/analysis_results/smc_fixed_execution_integrity_v1_0_1")
FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} :: {detail}")
        FAILS.append(name)


# ------------------------------------------------------------------ P1
def test_signal_rr_unique_within_execution_group():
    df = pd.read_csv(OUT / "signal_rr_consistency_audit.csv")
    check("no_group_with_multiple_rr",
          bool((df["n_groups_unique_rr_gt1"] == 0).all()),
          str(df["n_groups_unique_rr_gt1"].tolist()))
    check("max_unique_rr_is_1", bool((df["max_unique_rr"] == 1).all()),
          str(df["max_unique_rr"].tolist()))
    # 函数本身
    sub = pd.DataFrame(dict(
        symbol=["AG", "AG"], decision_time=pd.to_datetime(
            ["2025-01-01", "2025-01-01"]),
        rr_direction=["LONG_DOMINATES", "LONG_DOMINATES"]))
    out = m.signal_rr_consistency(sub)
    check("consistency_fn_unique_is_1", int(out.iloc[0]["n_unique_rr"]) == 1)


# ------------------------------------------------------------------ P1/P2
def test_rr_only_used_post_execution():
    params = set(inspect.signature(m.m101.run_execution_repaired).parameters)
    check("run_execution_no_future_param",
          not (params & {"rr_direction", "y_clear", "y_reversal"}), str(params))
    rows = [("AG", pd.Timestamp("2025-01-01"), +1, 99.7, 1.0, 10, 100.0,
             "LONG_DOMINATES")]
    sub = pd.DataFrame(rows, columns=[
        "symbol", "decision_time", "side", "entry_reference", "atr0",
        "contact_bar_index", "liquidity_price", "rr_direction"])
    allb = {("AG", pd.Timestamp("2025-01-01"), 1): 100.0}
    sig, _ = m.m101.collapse_signals(sub, np.array([True]), "CONT", allb)
    ms = pd.DataFrame(dict(price=[101.0, 102.0],
                           available_time=pd.to_datetime(["2024-01-01"] * 2),
                           first_penetration_time=pd.to_datetime([None, None])))
    a = m.m101.attach_targets_v101(sig, {"AG": ms}, "CONT")
    sig2 = sig.copy()
    sig2["attack_rr"] = "TRADEOFF_OR_OVERLAP"
    b = m.m101.attach_targets_v101(sig2, {"AG": ms}, "CONT")
    check("rr_does_not_change_target",
          a.iloc[0]["target_price"] == b.iloc[0]["target_price"])
    check("rr_does_not_change_skip",
          a.iloc[0]["skip_reason"] == b.iloc[0]["skip_reason"])


# ------------------------------------------------------------------ P4
def test_clear85_reproduces_v101():
    v101 = pd.read_csv(V101 / "execution_metrics_repaired.csv").sort_values("wf")
    cur = pd.read_csv(OUT / "clear85_vs_clear90_execution.csv")
    c85 = (cur[(cur.variant == "CLEAR85") & (cur.setup == "CONT")]
           .sort_values("wf"))
    check("clear85_expectancy_bitwise_v101",
          bool(np.allclose(c85["gross_expectancy_R"].to_numpy(),
                           v101["gross_expectancy_R"].to_numpy(), atol=1e-9)),
          f"{c85['gross_expectancy_R'].tolist()} vs "
          f"{v101['gross_expectancy_R'].tolist()}")
    check("clear85_trades_match_v101",
          bool((c85["executed_trades"].to_numpy()
                == v101["n_executed_trades"].to_numpy()).all()),
          f"{c85['executed_trades'].tolist()} vs "
          f"{v101['n_executed_trades'].tolist()}")


# ------------------------------------------------------------------ P5
def test_clear90_threshold_train_oof_only():
    y = np.array([1] * 20 + [0] * 10)
    p = np.linspace(0.95, 0.05, 30)
    thr = m.clear_threshold_at(y, p, 0.90)
    check("clear90_threshold_found", thr is not None, str(thr))
    if thr is not None:
        prec = float(y[p >= thr].mean())
        check("oof_precision_at_least_090", prec >= 0.90, str(prec))


def test_clear90_target_precision_exactly_090():
    check("PREC85_is_085", m.PREC85 == 0.85)
    check("PREC90_is_090", m.PREC90 == 0.90)
    proto = json.load(open(OUT / "EXECUTION_PURITY_PROTOCOL.json"))
    check("protocol_clear90_precision",
          proto["variants"]["CLEAR90"]["target_precision"] == 0.90)
    check("protocol_clear90_min_sel",
          proto["variants"]["CLEAR90"]["min_oof_selection"] == 0.05)


def test_clear90_no_test_threshold():
    sel = pd.read_csv(OUT / "clear85_vs_clear90_selector.csv")
    check("clear90_available_or_flagged",
          bool(sel["clear90_available"].notna().all()))
    sub = sel[sel["clear_thr_90"].notna()]
    check("clear90_stricter_than_clear85",
          bool((sub["clear_thr_90"] > sub["clear_thr_85"]).all()),
          str(sub[["clear_thr_85", "clear_thr_90"]].to_dict("records")))


# ------------------------------------------------------------------ P6
def test_direction_q10_unchanged():
    check("CONT_Q_fixed", m.CONT_Q == 0.10)
    check("REV_Q_fixed", m.REV_Q == 0.90)
    sel = pd.read_csv(OUT / "clear85_vs_clear90_selector.csv")
    check("cont_thr_single_column_not_per_variant",
          "cont_thr" in sel.columns and "cont_thr_90" not in sel.columns,
          str(list(sel.columns)))


def test_risk_target_entry_unchanged():
    check("risk_frozen_1_atr", m.m0.PRIMARY_RISK == 1.0)
    check("stop_from_decision_close",
          m.m0.stop_price_for(100.0, +1, 2.0) == 98.0
          and m.m0.stop_price_for(100.0, -1, 2.0) == 102.0)
    bars = m.m0  # entry = next bar
    b = dict(o=np.array([1.0, 2.0, 3.0]), h=np.array([1.0] * 3),
             l=np.array([1.0] * 3), c=np.array([1.0] * 3),
             disc=np.zeros(3, bool), n=3)
    check("entry_is_next_bar", m.m0.entry_bar_for(b, 0) == 1)
    proto = json.load(open(OUT / "EXECUTION_PURITY_PROTOCOL.json"))
    for k in ("risk", "target", "entry", "stop", "same_bar"):
        check(f"frozen_{k}", k in proto["frozen"], str(list(proto["frozen"])))


# ------------------------------------------------------------------ P6/P2
def test_full_live_universe():
    """UNRESOLVED_CENSOR / NO_COMPARABLE 的 y_clear 为 NaN；
    若 test 仍要求 y_clear.notna，则不可能出现在 executed trades 中。"""
    cls = pd.read_csv(OUT / "execution_by_frozen_class.csv")
    cont = cls[cls.setup.str.contains("CONT")]
    for wf, g in cont.groupby("wf"):
        unres = g[(g.frozen_class == "UNRESOLVED_CENSOR")]["n"].sum()
        nocmp = g[(g.frozen_class == "NO_COMPARABLE_TARGET")]["n"].sum()
        check(f"nan_yclear_classes_present_{wf}", int(unres + nocmp) > 0,
              f"unres={unres} nocmp={nocmp}")


def test_nonclear_counts_as_nonclear_not_dropped():
    mix = pd.read_csv(OUT / "clear_nonclear_mixture.csv")
    check("clear_plus_nonclear_equals_total",
          bool((mix["n_clear"] + mix["n_nonclear"]
                == mix["n_executed"]).all()),
          str(mix[["n_clear", "n_nonclear", "n_executed"]].to_dict("records")))
    cls = pd.read_csv(OUT / "execution_by_frozen_class.csv")
    tr = cls[(cls.frozen_class == "TRADEOFF_OR_OVERLAP")]
    check("tradeoff_not_dropped", bool((tr["n"] > 0).any()),
          str(tr["n"].tolist()))


# ------------------------------------------------------------------ P11
def test_geometry_bins_unchanged():
    check("bins_fixed", list(m.m101.GEO_BINS)[1:-1] == [0.50, 0.75, 1.00, 1.50],
          str(m.m101.GEO_BINS))
    geo = pd.read_csv(OUT / "geometry_by_clear_status.csv")
    check("has_clear_and_nonclear",
          set(geo["clear_status"].unique()) == {"CLEAR", "NONCLEAR"},
          str(geo["clear_status"].unique()))
    check("five_buckets_each", bool((geo.groupby(["wf", "clear_status"]).size()
                                     == 5).all()))


# ------------------------------------------------------------------ P12
def test_no_risk_scan():
    proto = json.load(open(OUT / "EXECUTION_PURITY_PROTOCOL.json"))
    check("risk_scan_forbidden", "risk 0.5/2.0" in proto["forbidden"],
          str(proto["forbidden"]))
    check("risk_single_1_atr", proto["frozen"]["risk"] == 1.0)


def test_no_clear_precision_scan():
    proto = json.load(open(OUT / "EXECUTION_PURITY_PROTOCOL.json"))
    check("only_two_variants",
          set(proto["variants"].keys()) == {"CLEAR85", "CLEAR90"},
          str(list(proto["variants"])))


# ------------------------------------------------------------------ P-OOS
def test_no_prospective_oos():
    D = m.m0.t3.load_data()
    check("max_insample_before_oos",
          str(D["days"][D["insample"]].max()) < m.OOS_START,
          f"{str(D['days'][D['insample']].max())} vs {m.OOS_START}")


def main():
    test_signal_rr_unique_within_execution_group()
    test_rr_only_used_post_execution()
    test_clear85_reproduces_v101()
    test_clear90_threshold_train_oof_only()
    test_clear90_target_precision_exactly_090()
    test_clear90_no_test_threshold()
    test_direction_q10_unchanged()
    test_risk_target_entry_unchanged()
    test_full_live_universe()
    test_nonclear_counts_as_nonclear_not_dropped()
    test_geometry_bins_unchanged()
    test_no_risk_scan()
    test_no_clear_precision_scan()
    test_no_prospective_oos()
    print(f"\n==== {len(FAILS)} FAIL / 14 groups ====")
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
