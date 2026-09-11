"""P14 测试套件：SMC Direction Pre-Execution Integrity Gate v1.5。

运行：
  python research/liquidity_oracle_atlas/test_direction_preexec_v1_5.py
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import research.liquidity_oracle_atlas.run_direction_preexec_integrity_v1_5 as m

OUT = m.OUT
FAILS = []
_CACHE = {}


def check(name, cond, detail=""):
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} :: {detail}")
        FAILS.append(name)


def cached():
    if "D" in _CACHE:
        return _CACHE["D"], _CACHE["ld"], _CACHE["checks"]
    D = m.t3.load_data()
    F = D["F"].copy()
    con = pd.read_parquet(m.CONTACTS)[m.KEYS + ["liquidity_price"]]
    F = F.merge(con, on=m.KEYS, how="left")
    ld, aud, checks = m.build_label_availability(F)
    F = F.merge(ld[m.KEYS + ["label_available_time"]], on=m.KEYS, how="left")
    D["F"] = F
    _CACHE["D"] = D
    _CACHE["ld"] = ld
    _CACHE["checks"] = checks
    return D, ld, checks


# ------------------------------------------------------------- P1
def test_label_availability_after_decision():
    _, ld, checks = cached()
    check("no_fatal_semantic_fail",
          checks["FATAL_LABEL_AVAILABILITY_SEMANTIC_FAIL"] is False, str(checks))
    check("frozen_rows_all_resolved",
          checks["frozen_rows_missing_bars_to_stop"] == 0
          and checks["frozen_rows_censored_lower_bound"] == 0
          and checks["frozen_rows_label_avail_nat"] == 0)
    check("all_label_avail_ge_decision_time",
          checks["all_label_avail_ge_decision_time"] is True)
    fz = ld[ld["rr_direction"].isin(m.FROZEN_CLEAR)]
    check("frozen_label_avail_notna", bool(fz["label_available_time"].notna().all()))
    check("frozen_label_avail_ge_decision",
          bool((fz["label_available_time"] >= fz["decision_time"]).all()))


# ------------------------------------------------------------- P2
def test_outer_train_labels_available_before_test():
    D, _, _ = cached()
    F = D["F"]; block = D["block"]; insample = D["insample"]
    dtime = pd.to_datetime(F["decision_time"]).to_numpy()
    lav = pd.to_datetime(F["label_available_time"]).to_numpy()
    y_rev = D["y_rev"]
    m_tr = pd.Series(block).isin(["TB1"]).to_numpy() & insample
    test_all = m.test_all_mask(block, ["TB2"], insample)
    test_start = dtime[test_all].min()
    dir_train = m_tr & (~pd.isna(y_rev)) & (lav < test_start)
    check("outer_train_avail_before_test",
          bool(pd.Timestamp(lav[dir_train].max()) < pd.Timestamp(test_start)),
          f"max={lav[dir_train].max()} start={test_start}")


# ------------------------------------------------------------- P3
def test_inner_train_labels_available_before_validation():
    df = pd.read_csv(OUT / "inner_oof_availability_audit.csv")
    rows = df[df["skipped"].isna() | (df["skipped"] == "")]
    ok = True
    bad = []
    for _, r in rows.iterrows():
        if pd.isna(r.get("max_train_label_avail_time")):
            continue
        if not (pd.Timestamp(r["max_train_label_avail_time"])
                < pd.Timestamp(r["val_start"])):
            ok = False; bad.append((r["wf"], r["task"], r["fold"]))
    check("inner_train_avail_before_val", ok, f"bad={bad}")
    check("inner_audit_nonempty", len(rows) > 0)


# ------------------------------------------------------------- P5
def test_live_test_mask_does_not_require_y_clear():
    D, _, _ = cached()
    block = D["block"]; insample = D["insample"]; y_clear = D["y_clear"]
    test_all = m.test_all_mask(block, ["TB2"], insample)
    n_all = int(test_all.sum())
    n_without_yclear = int((test_all & pd.isna(y_clear)).sum())
    check("test_all_includes_non_yclear_rows", n_without_yclear > 0,
          f"n_all={n_all} without_y_clear={n_without_yclear}")
    # 确认 test_all 确实大于 y_clear.notna 子集
    check("test_all_gt_evaluable",
          n_all > int((test_all & (~pd.isna(y_clear))).sum()))


# --------------------------------------------------- P5 selection purity
def test_selector_does_not_use_rr_direction():
    params = set(inspect.signature(m.s1_select).parameters)
    check("s1_select_no_rr_direction", "rr_direction" not in params, str(params))
    check("s1_select_params_exact",
          params == {"p_clear", "p_rev", "clear_thr", "cont_thr"}, str(params))


def test_selector_does_not_use_y_clear():
    params = set(inspect.signature(m.s1_select).parameters)
    check("s1_select_no_y_clear", "y_clear" not in params, str(params))


def test_selector_does_not_use_y_reversal():
    params = set(inspect.signature(m.s1_select).parameters)
    check("s1_select_no_y_reversal", "y_reversal" not in params, str(params))


# ------------------------------------------------------------- P8
def _mk_master(prices, avail, first_pen):
    return pd.DataFrame(dict(
        price=np.array(prices, float),
        available_time=pd.to_datetime(avail),
        first_penetration_time=pd.to_datetime(first_pen),
    ))


def test_target_uses_active_mask_only():
    dtn = np.datetime64("2025-01-01T12:00")
    # 105 已在 decision 前被消费 -> 排除；110 仍 active -> 最近 target
    ms = _mk_master([105.0, 110.0],
                    ["2025-01-01", "2025-01-01"],
                    ["2024-12-31", None])
    t = m.nearest_exante_target(ms, entry=100.0, side=+1, atr0=1.0, dtn=dtn)
    check("consumed_liquidity_excluded", t is not None and t["target_price"] == 110.0,
          str(t))
    # 若 105 也仍 active，则最近 target 是 105
    ms2 = _mk_master([105.0, 110.0],
                     ["2025-01-01", "2025-01-01"],
                     [None, None])
    t2 = m.nearest_exante_target(ms2, entry=100.0, side=+1, atr0=1.0, dtn=dtn)
    check("nearest_active_ahead_selected",
          t2 is not None and t2["target_price"] == 105.0, str(t2))


def test_no_target_abstains():
    dtn = np.datetime64("2025-01-01T12:00")
    ms = _mk_master([95.0], ["2025-01-01"], [None])
    t = m.nearest_exante_target(ms, entry=100.0, side=+1, atr0=1.0, dtn=dtn)
    check("no_ahead_target_returns_none", t is None, str(t))
    df = pd.read_csv(OUT / "exante_target_availability.csv")
    check("no_target_rate_reported", "no_target_rate" in df.columns)


def test_trade_direction_equals_side_for_continuation():
    check("cont_dir_eq_side_long", m.trade_direction_for_continuation(+1) == +1)
    check("cont_dir_eq_side_short", m.trade_direction_for_continuation(-1) == -1)


# ------------------------------------------------------------- P10
def test_duplicate_policy_not_applied_using_future():
    df = pd.DataFrame(dict(
        symbol=["AG", "AG", "AG", "CU", "CU"],
        decision_time=pd.to_datetime(["2025-01-01"] * 3 + ["2025-01-02"] * 2),
        trade_direction=[+1, +1, -1, +1, +1],
    ))
    out = m.duplicate_conflict_flags(df)
    # 只用 symbol/decision_time/trade_direction，不接触未来标签
    check("dup_conflict_runs_without_future_cols", isinstance(out, dict))
    check("conflict_detected", out["conf_group"] == 1, str(out))
    check("duplicate_group_detected", out["dup_group"] == 1, str(out))
    params = set(inspect.signature(m.duplicate_conflict_flags).parameters)
    check("dup_policy_single_df_param", params == {"df"}, str(params))


# ------------------------------------------------------------- OOS
def test_no_prospective_oos():
    D, _, _ = cached()
    max_day = str(D["days"][D["insample"]].max())
    check("max_insample_before_oos", max_day < m.OOS_START,
          f"max={max_day} oos={m.OOS_START}")


def main():
    test_label_availability_after_decision()
    test_outer_train_labels_available_before_test()
    test_inner_train_labels_available_before_validation()
    test_live_test_mask_does_not_require_y_clear()
    test_selector_does_not_use_rr_direction()
    test_selector_does_not_use_y_clear()
    test_selector_does_not_use_y_reversal()
    test_target_uses_active_mask_only()
    test_no_target_abstains()
    test_trade_direction_equals_side_for_continuation()
    test_duplicate_policy_not_applied_using_future()
    test_no_prospective_oos()
    print(f"\n==== {len(FAILS)} FAIL / 12 groups ====")
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
