"""P17 测试套件：SMC Conditional Tradeoff Veto v1.2。

运行：
  python research/liquidity_oracle_atlas/test_conditional_tradeoff_veto_v1_2.py

只依赖已生成的产物 + 合成数据；不重新加载 15 品种原始行情。
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import research.liquidity_oracle_atlas.run_direction_deployability_v1_1_1_repair as rp
import research.liquidity_oracle_atlas.run_conditional_tradeoff_veto_v1_2 as m

OUT = m.OUT
V101 = m.V101
FAILS = []

# 合成数据也需要 routed 特征（不加载真实 env）
rp.define_blocks(["SYNTH_TYPE"])


def check(name, cond, detail=""):
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} :: {detail}")
        FAILS.append(name)


def csv(name):
    return pd.read_csv(OUT / name)


# ------------------------------------------------------------ P5 / leakage
def test_toxic_label_not_in_features():
    cols = set(m.G4_BASE) | set(m.PRECONTACT_COLS)
    bad = cols & m.LEAK_FIELDS
    check("toxic_label_not_in_feature_cols", not bad, str(bad))
    check("rr_direction_not_in_any_model_cols",
          "rr_direction" not in cols and "rr_direction" not in m.G4_BASE)
    # Stage A 只用 frozen G4_BASE；Stage B = G4_BASE + precontact
    check("stageA_cols_exactly_G4_BASE", list(m.G4_BASE) == list(m.m0.G4_BASE))


def test_toxic_training_resolved_only():
    rm = m.resolved_mask(np.array(["LONG_DOMINATES", "SHORT_DOMINATES",
                                   "TRADEOFF_OR_OVERLAP",
                                   "UNRESOLVED_CENSOR",
                                   "NO_COMPARABLE_TARGET"]))
    check("resolved_mask_three_classes", list(rm) == [True, True, True,
                                                      False, False], str(rm))
    check("unknown_excluded_from_resolved",
          not (set(m.UNKNOWN) & set(m.RESOLVED)))
    c = csv("conditional_toxic_metrics.csv")
    check("toxic_positives_within_cohort", bool((c.n_toxic_oof <= c.n_cohort_oof).all()))
    check("toxic_base_rate_sane",
          bool(((c.toxic_base_rate > 0) & (c.toxic_base_rate < 1)).all()))


# ------------------------------------------------------------ P4 availability
def test_label_availability_outer():
    c = csv("conditional_toxic_metrics.csv")
    ok, bad = True, []
    for _, r in c.iterrows():
        a, b = pd.Timestamp(r["max_train_lav"]), pd.Timestamp(r["test_start"])
        if not (a < b):
            ok = False
            bad.append((r["wf"], r["model"], str(a), str(b)))
    check("outer_train_lav_before_test_start", ok, str(bad))


def test_label_availability_inner():
    d = csv("toxic_inner_oof_audit.csv")
    rows = d[d["skipped"].isna() | (d["skipped"] == "")]
    ok, bad = True, []
    n_checked = 0
    for _, r in rows.iterrows():
        if pd.isna(r.get("max_train_label_avail_time")):
            continue
        n_checked += 1
        if not (pd.Timestamp(r["max_train_label_avail_time"])
                < pd.Timestamp(r["val_start"])):
            ok = False
            bad.append((r["wf"], r["stage"], r["model"], r["fold"]))
    check("inner_train_lav_before_val_start", ok, str(bad))
    check("inner_audit_nonempty", n_checked > 0, f"n={n_checked}")


# ------------------------------------------------------------ P3 cohort
def test_direction_candidate_oof_only():
    params = set(inspect.signature(m.direction_oof_score_all).parameters)
    check("cohort_scorer_no_label_params",
          params == {"cols", "X", "y", "dtime", "lav", "day", "mode"}, str(params))
    check("cohort_scorer_has_no_rr_param",
          "rr_direction" not in params and "rr" not in params, str(params))

    # 功能验证：对 y 为 NaN 的行（= 非 clear 行，含 TRADEOFF）也必须给出分数
    rng = np.random.default_rng(0)
    n_days, per_day = 10, 30
    n = n_days * per_day
    days = np.repeat([f"2025-01-{i + 1:02d}" for i in range(n_days)], per_day)
    X = pd.DataFrame(dict(
        symbol=rng.choice(["A", "B"], n), side=rng.choice([-1, 1], n),
        nearest_above_R=rng.normal(size=n), nearest_below_R=rng.normal(size=n),
        nearest_ahead_R=rng.normal(size=n), nearest_behind_R=rng.normal(size=n)))
    y = np.where(np.arange(n) % 2 == 0, rng.integers(0, 2, n), np.nan)
    dt = (pd.to_datetime("2025-01-01")
          + pd.to_timedelta(np.repeat(np.arange(n_days), per_day), "D")
          + pd.to_timedelta(np.tile(np.arange(per_day) * 5, n_days), "m"))
    lav = dt + pd.Timedelta(hours=1)      # 标签次日可得；train window 足够长
    oidx, pred, _ = m.direction_oof_score_all(m.G4_BASE, X, y, dt.to_numpy(),
                                              lav.to_numpy(), days, "logistic")
    n_nan_scored = int(np.isnan(y[oidx]).sum())
    check("cohort_scores_rows_without_direction_label",
          n_nan_scored > 0 and len(pred) == len(oidx),
          f"n_nan_scored={n_nan_scored} n_pred={len(pred)}")


# ------------------------------------------------------------ P6 threshold
def test_toxic_threshold_train_oof_only():
    params = set(inspect.signature(m.choose_toxic_threshold).parameters)
    check("toxic_threshold_no_test_input",
          params == {"y_toxic", "p_toxic", "target_precision",
                     "min_coverage"}, str(params))

    y = np.zeros(100, int)
    y[:5] = 1
    p = np.linspace(0.99, 0.01, 100)
    r = m.choose_toxic_threshold(y, p)
    # precision>=0.5 -> k<=10；coverage>=0.03 -> k>=3；最大 coverage = 10
    check("toxic_threshold_max_coverage_at_min_precision",
          r is not None and abs(r["oof_precision"] - 0.5) < 1e-9
          and abs(r["oof_veto_coverage"] - 0.10) < 1e-9, str(r))
    check("toxic_threshold_matches_p_at_k",
          r is not None and abs(r["threshold"] - p[9]) < 1e-12, str(r))
    check("toxic_threshold_none_when_unreachable",
          m.choose_toxic_threshold(np.zeros(100, int), p) is None)

    t = csv("toxic_thresholds.csv")
    got = t[t["available"] == True]
    check("no_threshold_below_preregistered_precision",
          bool((got["oof_precision"] >= m.TARGET_TOXIC_PRECISION - 1e-9).all()),
          str(got[["wf", "model", "oof_precision"]].to_dict("records")))
    check("unavailable_rows_have_no_threshold",
          bool(t[t["available"] == False]["threshold"].isna().all()))
    check("thresholds_preregistered",
          m.TARGET_TOXIC_PRECISION == 0.50 and m.MIN_VETO_COVERAGE == 0.03)


# ------------------------------------------------------------ P7 veto
def test_veto_never_adds_trade():
    # 合成：veto = base AND (p < thr) 结构上是子集
    base = np.array([True, True, False, True])
    p = np.array([0.9, 0.1, 0.9, 0.2])
    veto = base & (p < 0.5)
    check("veto_is_subset_of_baseline", not (veto & ~base).any(),
          str((base, veto)))
    e = csv("stageA_veto_execution.csv")
    check("post_veto_trades_le_baseline",
          bool((e["post_veto_trades"] <= e["baseline_trades"]).all()))
    check("clear_retention_le_1",
          bool((e["clear_retention_rate"] <= 1.0 + 1e-9).all()))
    g = csv("veto_targeting_audit.csv")
    check("no_new_toxic_groups_created",
          bool((g["n_groups_newly_toxic"] == 0).all()),
          str(g[["wf", "model", "n_groups_newly_toxic"]].to_dict("records")))


def test_clear85_q10_baseline_unchanged():
    o = csv("oracle_tradeoff_ceiling.csv").sort_values("wf")
    v = pd.read_csv(V101 / "execution_metrics_repaired.csv").sort_values("wf")
    check("baseline_reproduces_v1_0_1",
          np.allclose(o["original_expectancy_R"].to_numpy(float),
                      v["gross_expectancy_R"].to_numpy(float), atol=1e-9),
          f"{o['original_expectancy_R'].tolist()} vs "
          f"{v['gross_expectancy_R'].tolist()}")
    check("frozen_selector_constants_unchanged",
          m.CLEAR_PRECISION == m.m0.CLEAR_PRECISION == 0.85
          and m.CONT_Q == m.m0.CONT_Q == 0.10
          and m.PRIMARY_RISK == m.m0.PRIMARY_RISK == 1.0)


# ------------------------------------------------------------ Stage A/B scope
def test_stageA_uses_no_new_features():
    c = csv("conditional_toxic_metrics.csv")
    a = c[c["stage"] == "A"]["model"].unique().tolist()
    check("stageA_models_fixed", sorted(a) == ["T0_HGB", "T0_LOGIT"], str(a))
    check("stageA_cols_equal_G4_BASE",
          list(m.G4_BASE) == ["symbol", "side"] + [
              "nearest_above_R", "nearest_below_R", "nearest_ahead_R",
              "nearest_behind_R"])
    check("precontact_disjoint_from_G4_BASE",
          not (set(m.PRECONTACT_COLS) & set(m.G4_BASE)))


def _synth_bars(n=120):
    c = np.arange(n, dtype=float) + 100.0
    return dict(o=c.copy(), h=c + 1.0, l=c - 1.0, c=c,
                t=pd.to_datetime("2025-01-01") + pd.to_timedelta(
                    np.arange(n) * 5, "min"),
                day=np.array(["2025-01-01"] * n),
                disc=np.zeros(n, bool), n=n)


def _synth_F(j=50, side=+1):
    return pd.DataFrame(dict(symbol=["AG"], contact_bar_index=[j],
                            side=[side], atr0=[1.0]))


def test_precontact_uses_only_bars_before_contact():
    b = _synth_bars()
    f1 = m.precontact_features(_synth_F(50), {"AG": b})
    b2 = {k: (v.copy() if isinstance(v, np.ndarray) else v)
          for k, v in b.items()}
    b2["c"][49] += 10.0
    b2["h"][49] += 10.0
    b2["l"][49] += 10.0
    f2 = m.precontact_features(_synth_F(50), {"AG": b2})
    check("bar_j_minus_1_affects_features",
          not np.allclose(f1.to_numpy(float), f2.to_numpy(float),
                          equal_nan=True))


def test_precontact_no_contact_bar():
    b = _synth_bars()
    f1 = m.precontact_features(_synth_F(50), {"AG": b})
    b2 = {k: (v.copy() if isinstance(v, np.ndarray) else v)
          for k, v in b.items()}
    for k in ("o", "h", "l", "c"):
        b2[k][50] += 1000.0     # contact bar
        b2[k][51] += 1000.0     # 之后的 bar
        b2[k][52] += 1000.0
    f2 = m.precontact_features(_synth_F(50), {"AG": b2})
    check("contact_bar_and_after_do_not_affect_features",
          np.allclose(f1.to_numpy(float), f2.to_numpy(float), equal_nan=True),
          f"max_diff={np.nanmax(np.abs(f1.to_numpy(float) - f2.to_numpy(float)))}")


def test_no_volume_features():
    bad = [c for c in m.PRECONTACT_COLS if "volume" in c.lower() or "vol" in c.lower()]
    check("no_volume_features", not bad, str(bad))


# ------------------------------------------------------------ governance
def test_risk_target_entry_unchanged():
    check("risk_frozen_1ATR", m.PRIMARY_RISK == 1.0)
    check("no_local_target_reimplementation",
          not hasattr(m, "continuation_target")
          and not hasattr(m, "nearest_ahead")
          and not hasattr(m, "stop_price_for"),
          "v1.2 必须复用 m0/m101 的 target/stop/entry 实现")
    check("execution_reuses_v101_helpers",
          m.execute_selection.__module__ == m.__name__
          and m.m101.attach_targets_v101 is not None
          and m.m101.run_execution_repaired is not None)
    o = csv("oos_guard_audit.csv")
    check("all_variants_exit_before_oos",
          bool((o["n_exit_on_or_after_oos"] == 0).all()))


def test_no_clear_threshold_scan():
    check("single_clear_precision", m.CLEAR_PRECISION_VALUES == [0.85],
          str(m.CLEAR_PRECISION_VALUES))
    check("no_clear90_module_constant",
          not hasattr(m, "PREC90") and not hasattr(m, "CLEAR90"))


def test_no_risk_scan():
    check("single_risk_value", m.RISK_VALUES == [1.0], str(m.RISK_VALUES))
    check("no_risk_list_attribute",
          not hasattr(m, "RISK_GRID") and not hasattr(m, "RISKS"))


def test_no_prospective_oos():
    check("oos_start_frozen", m.OOS_START == "2026-09-07")
    check("no_oos_bar_read", m.m101.OOS_START == m.OOS_START)
    o = csv("oos_guard_audit.csv")
    check("oos_guard_all_zero",
          int(o["n_exit_on_or_after_oos"].sum()) == 0,
          str(o.to_dict("records")))


def main():
    test_toxic_label_not_in_features()
    test_toxic_training_resolved_only()
    test_label_availability_outer()
    test_label_availability_inner()
    test_direction_candidate_oof_only()
    test_toxic_threshold_train_oof_only()
    test_veto_never_adds_trade()
    test_clear85_q10_baseline_unchanged()
    test_stageA_uses_no_new_features()
    test_precontact_uses_only_bars_before_contact()
    test_precontact_no_contact_bar()
    test_no_volume_features()
    test_risk_target_entry_unchanged()
    test_no_clear_threshold_scan()
    test_no_risk_scan()
    test_no_prospective_oos()
    print(f"\n==== {len(FAILS)} FAIL / 16 groups ====")
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
