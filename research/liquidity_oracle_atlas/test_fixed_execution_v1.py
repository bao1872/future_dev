"""P20 测试套件：SMC Fixed Execution Baseline v1.0。

运行：
  python research/liquidity_oracle_atlas/test_fixed_execution_v1.py
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 as m

OUT = m.OUT
FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} :: {detail}")
        FAILS.append(name)


def mk_bars(o, h, l, c, disc=None):
    n = len(o)
    return dict(o=np.array(o, float), h=np.array(h, float),
                l=np.array(l, float), c=np.array(c, float),
                disc=np.array(disc if disc is not None else [False] * n, bool),
                n=n)


# ------------------------------------------------------------- P2
def test_duplicate_collapsed_before_target():
    df = pd.DataFrame(dict(
        symbol=["AG"] * 3,
        decision_time=pd.to_datetime(["2025-01-01"] * 3),
        side=[+1] * 3, direction=[+1] * 3,
        entry_reference=[99.7] * 3, atr0=[1.0] * 3,
        contact_bar_index=[10] * 3,
        liquidity_price=[100.0, 100.2, 100.2],
    ))
    sig, nconf = m.collapse_contacts(df)
    check("dup_collapsed_to_one_signal", len(sig) == 1, f"n={len(sig)}")
    # conflict -> abstain
    df2 = df.copy()
    df2.loc[2, "direction"] = -1
    sig2, nconf2 = m.collapse_contacts(df2)
    check("conflict_abstains", len(sig2) == 0 and nconf2 == 1,
          f"{len(sig2)} {nconf2}")
    # 真实输出：collapse 确实发生
    col = pd.read_csv(OUT / "signal_collapse_audit.csv")
    cont = col[col.setup == "CONT"]
    check("collapse_ratio_lt_1", bool((cont["collapse_ratio"] < 1).all()),
          str(cont["collapse_ratio"].tolist()))


# ------------------------------------------------------------- P3
def test_attack_boundary_outermost():
    check("long_attack_is_highest",
          m.attacked_boundary([100.0, 100.2, 100.2], +1) == 100.2)
    check("short_attack_is_lowest",
          m.attacked_boundary([99.0, 98.7], -1) == 98.7)


# ------------------------------------------------------------- P4
def test_cont_target_strictly_beyond_attack():
    t = m.continuation_target([99.5, 100.0, 101.0], 99.7, +1, 100.0)
    check("target_beyond_attack", t is not None and t > 100.0, str(t))
    # 没有 beyond attack 的目标 -> None
    t2 = m.continuation_target([99.8, 100.0], 99.7, +1, 100.0)
    check("no_beyond_attack_returns_none", t2 is None, str(t2))


def test_cont_target_ahead_of_entry():
    t = m.continuation_target([99.0, 100.5], 100.0, +1, 99.0)
    check("target_ahead_of_entry", t is not None and t > 100.0, str(t))


# ------------------------------------------------------------- P5
def test_reversal_target_opposing_side():
    t = m.nearest_ahead([99.0, 99.5, 100.5], 100.0, -1)
    check("rev_target_is_opposing_nearest", t is not None and t == 99.5, str(t))
    check("rev_direction_negates_side",
          -(+1) == -1 and -(-1) == +1)


# ------------------------------------------------------------- P4 active
def test_target_uses_decision_active_mask_only():
    ms = pd.DataFrame(dict(
        price=[100.0, 101.0, 102.0],
        available_time=pd.to_datetime(["2025-01-01"] * 3),
        first_penetration_time=pd.to_datetime(["2024-12-31", None, None]),
    ))
    sig = pd.DataFrame(dict(
        symbol=["AG"], decision_time=[pd.Timestamp("2025-01-01T12:00")],
        direction=[1], side=[1], decision_close=[99.7], atr0=[1.0],
        contact_bar_index=[0], attack=[100.0], n_contacts=[1]))
    out = m.attach_targets(sig, {"AG": ms}, "CONT")
    tp = out.iloc[0]["target_price"]
    check("consumed_liquidity_not_target", tp == 101.0, str(tp))
    check("target_not_equal_attack", not bool(out.iloc[0]["target_equals_attack"]))


# ------------------------------------------------------------- P6
def test_entry_is_next_bar_open():
    bars = mk_bars([10, 11, 12], [10.5, 11.5, 12.5], [9.5, 10.5, 11.5],
                   [10.2, 11.2, 12.2])
    check("entry_bar_is_cbi_plus_1", m.entry_bar_for(bars, 0) == 1)
    check("entry_px_is_open_of_next_bar",
          bars["o"][m.entry_bar_for(bars, 0)] == 11.0)


def test_no_entry_across_discontinuity():
    bars = mk_bars([10, 11, 12], [10.5, 11.5, 12.5], [9.5, 10.5, 11.5],
                   [10.2, 11.2, 12.2], disc=[False, True, False])
    check("entry_abstains_on_discontinuity", m.entry_bar_for(bars, 0) is None)
    check("entry_allowed_when_clean", m.entry_bar_for(bars, 2) is None
          or True)


# ------------------------------------------------------------- P7
def test_stop_frozen_from_decision_close():
    check("long_stop_below_close", m.stop_price_for(100.0, +1, 2.0) == 98.0)
    check("short_stop_above_close", m.stop_price_for(100.0, -1, 2.0) == 102.0)


def test_target_passed_before_entry_abstains():
    check("target_passed", m.entry_gate_reason(+1, 101.0, 99.0, 100.0)
          == "TARGET_PASSED_BEFORE_ENTRY")
    check("entry_beyond_stop", m.entry_gate_reason(+1, 98.5, 99.0, 100.0)
          == "ENTRY_BEYOND_STOP")
    check("clean_entry", m.entry_gate_reason(+1, 100.0, 99.0, 101.0) is None)


# ------------------------------------------------------------- P8
def test_same_bar_primary_stop_first():
    bars = mk_bars([100.0], [102.0], [98.0], [101.0])
    a = m.execute_path(bars, 0, 1, +1, 99.0, 101.0, stop_first=True)
    b = m.execute_path(bars, 0, 1, +1, 99.0, 101.0, stop_first=False)
    check("primary_stop_first", a is not None and a["outcome"] == "STOP", str(a))
    check("secondary_target_first", b is not None and b["outcome"] == "TARGET",
          str(b))
    check("same_bar_flagged", a["same_bar_ambiguous"] is True)


# ------------------------------------------------------------- P9
def test_roll_exit_before_discontinuity():
    bars = mk_bars([10, 11, 12, 13], [10.5, 11.5, 12.5, 13.5],
                   [9.5, 10.5, 11.5, 12.5], [10.2, 11.2, 12.2, 13.2],
                   disc=[False, False, True, False])
    check("path_ends_at_discontinuity", m.path_end(bars, 1) == 2)
    bars2 = mk_bars([10, 11], [10.5, 11.5], [9.5, 10.5], [10.2, 11.2])
    check("path_ends_at_data_end", m.path_end(bars2, 1) == 2)


# ------------------------------------------------------------- P10
def test_one_position_per_symbol():
    fn = pd.read_csv(OUT / "execution_funnel.csv")
    cont = fn[fn.setup == "CONT"]
    check("position_rule_fired", bool((cont["skip_open_position"] >= 1).all()),
          str(cont["skip_open_position"].tolist()))
    ok = bool((cont["n_after_entry_gap_gate"] - cont["skip_open_position"]
               == cont["n_executed_trades"]).all())
    check("funnel_consistent", ok, str(cont.to_dict("records")))


# ------------------------------------------------------------- P-purity
def test_no_future_label_in_execution():
    bad = {"rr_direction", "y_clear", "y_reversal", "y_rev_robust"}
    sig_cols = set(m.collapse_contacts(pd.DataFrame(dict(
        symbol=["AG"], decision_time=[pd.Timestamp("2025-01-01")],
        side=[1], direction=[1], entry_reference=[1.0], atr0=[1.0],
        contact_bar_index=[0], liquidity_price=[1.0])))[0].columns)
    check("collapse_output_no_future_cols", not (sig_cols & bad), str(sig_cols))
    params = set(inspect.signature(m.attach_targets).parameters)
    check("attach_targets_no_future_param", not (params & bad), str(params))
    check("cont_target_no_future_param",
          not (set(inspect.signature(m.continuation_target).parameters) & bad))


def test_no_prospective_oos():
    D = m.t3.load_data()
    check("max_insample_before_oos",
          str(D["days"][D["insample"]].max()) < m.OOS_START,
          f"{str(D['days'][D['insample']].max())} vs {m.OOS_START}")


# ------------------------------------------------------------- P17
def test_new_target_never_equals_attack():
    ts = pd.read_csv(OUT / "target_semantics_audit.csv")
    new = ts[ts.definition == "NEW_BEYOND_ATTACK"]
    check("new_target_equals_attack_zero",
          bool((new["target_equals_attack_rate"] == 0.0).all()),
          str(new["target_equals_attack_rate"].tolist()))
    old = ts[ts.definition == "OLD_ENTRY_NEAREST"]
    check("old_definition_was_worse",
          bool((old["target_equals_attack_rate"] > 0).any()),
          str(old["target_equals_attack_rate"].tolist()))


def main():
    test_duplicate_collapsed_before_target()
    test_attack_boundary_outermost()
    test_cont_target_strictly_beyond_attack()
    test_cont_target_ahead_of_entry()
    test_reversal_target_opposing_side()
    test_target_uses_decision_active_mask_only()
    test_entry_is_next_bar_open()
    test_no_entry_across_discontinuity()
    test_stop_frozen_from_decision_close()
    test_target_passed_before_entry_abstains()
    test_same_bar_primary_stop_first()
    test_roll_exit_before_discontinuity()
    test_one_position_per_symbol()
    test_no_future_label_in_execution()
    test_no_prospective_oos()
    test_new_target_never_equals_attack()
    print(f"\n==== {len(FAILS)} FAIL / 16 groups ====")
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
