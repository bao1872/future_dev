"""P16 测试套件：SMC Direction Asymmetric Selective Abstention v1.4。

运行：
  python research/liquidity_oracle_atlas/test_direction_asymmetric_v1_4.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import research.liquidity_oracle_atlas.run_direction_asymmetric_abstention_v1_4 as m

OUT = m.OUT
FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} :: {detail}")
        FAILS.append(name)


# ----------------------------------------------------------- thresholds OOF
def test_thresholds_fit_oof_only():
    D = m.load_data()
    wf1 = m.build_wf(["TB1"], ["TB2"], D)
    lo = float(np.quantile(wf1["p_rev_oof"], 0.10))
    hi = float(np.quantile(wf1["p_rev_oof"], 0.90))
    thr = pd.read_csv(OUT / "class_thresholds_by_wf.csv")
    s0 = thr[(thr.selector == "S0_SYMMETRIC_10_10") & (thr.wf == "WF1")].iloc[0]
    check("s0_cont_thr_is_oof_q10", abs(s0.cont_threshold - lo) < 1e-9,
          f"{s0.cont_threshold} vs {lo}")
    check("s0_rev_thr_is_oof_q90", abs(s0.rev_threshold - hi) < 1e-9,
          f"{s0.rev_threshold} vs {hi}")


# -------------------------------------------------- cont/rev thr independent
def test_cont_and_rev_thresholds_independent():
    thr = pd.read_csv(OUT / "class_thresholds_by_wf.csv")
    s3 = thr[thr.selector == "S3_CLASS_SPECIFIC_PRECISION"]
    both = s3[s3.cont_enabled & s3.rev_enabled]
    check("s3_has_both_thresholds", len(both) >= 1, f"n={len(both)}")
    if len(both):
        # cont 阈值来自 1-p_rev，rev 阈值来自 p_rev：不应数值相同
        differ = (both.cont_threshold - both.rev_threshold).abs() > 1e-6
        check("s3_cont_rev_thr_differ", bool(differ.all()), str(both.to_dict("records")))
    # 重算 S3 WF1，确认与存储一致（独立性 + OOF-only）
    D = m.load_data()
    wf1 = m.build_wf(["TB1"], ["TB2"], D)
    cont = m.choose_class_precision_threshold(
        wf1["y_rev_oof"], 1 - wf1["p_rev_oof"], target_class=0,
        target_precision=m.TARGET_DIRECTION_PRECISION,
        min_coverage=m.MIN_CLASS_OOF_COVERAGE)
    rev = m.choose_class_precision_threshold(
        wf1["y_rev_oof"], wf1["p_rev_oof"], target_class=1,
        target_precision=m.TARGET_DIRECTION_PRECISION,
        min_coverage=m.MIN_CLASS_OOF_COVERAGE)
    s3w1 = s3[s3.wf == "WF1"].iloc[0]
    check("s3_wf1_cont_thr_matches",
          abs(s3w1.cont_threshold - cont["threshold"]) < 1e-9,
          f"{s3w1.cont_threshold} vs {cont['threshold']}")
    check("s3_wf1_rev_thr_matches",
          abs(s3w1.rev_threshold - rev["threshold"]) < 1e-9,
          f"{s3w1.rev_threshold} vs {rev['threshold']}")


# ------------------------------------------------------ disabled not forced
def test_disabled_class_not_forced():
    thr = pd.read_csv(OUT / "class_thresholds_by_wf.csv")
    sel = pd.read_csv(OUT / "selector_metrics.csv")
    ok = True
    bad = []
    for _, t in thr.iterrows():
        if t.selector not in ("S3_CLASS_SPECIFIC_PRECISION",
                              "S1_CONTINUATION_ONLY_10", "S2_REVERSAL_ONLY_10"):
            continue
        row = sel[(sel.selector == t.selector) & (sel.wf == t.wf)]
        if row.empty:
            continue
        row = row.iloc[0]
        if not bool(t.cont_enabled) and row.n_pred_continuation != 0:
            ok = False; bad.append((t.selector, t.wf, "cont"))
        if not bool(t.rev_enabled) and row.n_pred_reversal != 0:
            ok = False; bad.append((t.selector, t.wf, "rev"))
    check("disabled_class_produces_zero", ok, f"bad={bad}")
    # 明确 S3 WF2 rev disabled 的证据
    s3w2 = thr[(thr.selector == "S3_CLASS_SPECIFIC_PRECISION") & (thr.wf == "WF2")]
    if len(s3w2):
        check("s3_wf2_rev_disabled_evidence",
              (not bool(s3w2.iloc[0].rev_enabled))
              and (sel[(sel.selector == "S3_CLASS_SPECIFIC_PRECISION")
                       & (sel.wf == "WF2")].iloc[0].n_pred_reversal == 0),
              "rev not disabled or forced")


# ------------------------------------------- continuation -> LONG and SHORT
def test_continuation_maps_to_both_long_short():
    sel = pd.read_csv(OUT / "selector_metrics.csv")
    s1 = sel[sel.selector == "S1_CONTINUATION_ONLY_10"]
    check("s1_all_wf_have_long", bool((s1.n_predicted_LONG > 0).all()),
          str(s1[["wf", "n_predicted_LONG", "n_predicted_SHORT"]].to_dict("records")))
    check("s1_all_wf_have_short", bool((s1.n_predicted_SHORT > 0).all()))
    s3 = sel[sel.selector == "S3_CLASS_SPECIFIC_PRECISION"]
    check("s3_all_wf_have_long_short",
          bool((s3.n_predicted_LONG > 0).all() and (s3.n_predicted_SHORT > 0).all()))


# ------------------------------------------------- tradeoff counts as failure
def test_tradeoff_counts_as_failure():
    sel = pd.read_csv(OUT / "selector_metrics.csv")
    ok = True
    bad = []
    for _, r in sel.iterrows():
        if pd.notna(r.direction_accuracy_given_clear) and \
                r.actionable_precision > r.direction_accuracy_given_clear + 1e-9:
            ok = False; bad.append((r.selector, r.wf))
    check("actionable_le_dagc", ok, f"bad={bad}")
    check("tradeoff_rate_reported", sel.tradeoff_selected_rate.notna().all())


# ------------------------------------------------ bootstrap duplicates days
def test_tail_bootstrap_duplicates_days():
    D = m.load_data()
    wf1 = m.build_wf(["TB1"], ["TB2"], D)
    days = np.sort(pd.unique(wf1["day"]))
    rng = np.random.default_rng(0)
    dup_seen = False
    for _ in range(200):
        samp = m.draw_day_sample(days, rng)
        if len(samp) != len(days):
            dup_seen = False
            break
        if len(np.unique(samp)) < len(samp):
            dup_seen = True
    check("bootstrap_samples_days_with_replacement", dup_seen)
    ta = m.tail_asymmetry(wf1)
    b = m.block_bootstrap_delta(wf1["day"], ta["cont_pick"], ta["cont_correct"],
                                ta["rev_pick"], ta["rev_correct"], n_boot=100)
    check("bootstrap_ci_ordered", b is not None and b["ci_lo"] <= b["ci_hi"],
          str(b))


# ------------------------------------------------------ no test quantile
def test_no_test_quantile_threshold():
    D = m.load_data()
    wf1 = m.build_wf(["TB1"], ["TB2"], D)
    lo_oof = float(np.quantile(wf1["p_rev_oof"], 0.10))
    lo_test = float(np.quantile(wf1["p_rev_test"], 0.10))
    thr = pd.read_csv(OUT / "class_thresholds_by_wf.csv")
    s0 = thr[(thr.selector == "S0_SYMMETRIC_10_10") & (thr.wf == "WF1")].iloc[0]
    check("thr_differs_from_test_quantile",
          abs(s0.cont_threshold - lo_test) > 1e-6,
          f"thr={s0.cont_threshold} test_q10={lo_test}")
    check("thr_equals_oof_quantile", abs(s0.cont_threshold - lo_oof) < 1e-9)


# ------------------------------------------------------------- no new feat
def test_no_new_features():
    proto = json.load(open(OUT / "DIRECTION_ASYM_PROTOCOL.json"))
    used = set(proto["direction"]["features"])
    check("no_new_features", not (used - m.ALLOWED_FEATURES),
          f"leaked={used - m.ALLOWED_FEATURES}")
    check("no_forbidden_columns", not (used & m.FORBIDDEN),
          f"touch={used & m.FORBIDDEN}")


# ------------------------------------------------------------ no OOS leak
def test_no_prospective_oos():
    D = m.load_data()
    check("max_insample_before_oos",
          str(D["DAYS"][D["insample"]].max()) < m.OOS_START,
          str(D["DAYS"][D["insample"]].max()))


def main():
    test_thresholds_fit_oof_only()
    test_cont_and_rev_thresholds_independent()
    test_disabled_class_not_forced()
    test_continuation_maps_to_both_long_short()
    test_tradeoff_counts_as_failure()
    test_tail_bootstrap_duplicates_days()
    test_no_test_quantile_threshold()
    test_no_new_features()
    test_no_prospective_oos()
    print(f"\n==== {len(FAILS)} FAIL / 9 groups ====")
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
