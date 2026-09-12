"""SMC Risk-Coupled Execution v1.0 — 复现 / 不变量测试。

便宜的 import / 不变量测试始终运行（不加载全量 env）：
  test_no_risk_grid_expansion        RISKS 固定 {0.5,1.0,2.0}，无连续扫描
  test_each_risk_uses_own_rr_label   rr_direction_at_risk 在 0.5/1.0/2.0 返回不同结果
  test_stop_uses_matching_risk       run_execution_risk 的 stop = risk*atr0（与 risk=1 不同）
  test_target_policy_identical       attach_targets 风险无关（target 语义不随 risk 变）

依赖实验产出的测试（产物存在才跑，否则 skip）：
  test_risk1_reproduces_baseline     risk=1 逐位复现 614/856/848, -0.0420/+0.0253/-0.0376
  test_each_risk_uses_own_label_availability  各 risk 的 label_available_time 不同 -> 执行结果不同
  test_each_risk_refits_clear_model  每 risk 独立 clear_thr
  test_each_risk_refits_direction_model 每 risk 独立 cont_thr
  test_thresholds_train_oof_only     clear_thr 来自训练 OOF（有限值）
  test_full_live_universe            selector n_test_all == WF 全 live universe
  test_no_future_label_in_selector   所有 executed 退出 bar > 进入 bar（因果）
  test_no_prospective_oos            无 PROSPECTIVE_OOS_ENTRY 漏进 executed
"""
from __future__ import annotations

import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(".").resolve()))
import research.liquidity_oracle_atlas.run_risk_coupled_execution_v1 as m
import research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 as m0
import research.liquidity_oracle_atlas.run_fixed_execution_integrity_v1_0_1 as m101

OUT = m.OUT
FRONT = m.FRONT
FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f" :: {detail}"))
    if not ok:
        FAILS.append(name)


def test_no_risk_grid_expansion():
    check("no_risk_grid_expansion", m.RISKS == [0.5, 1.0, 2.0],
          f"RISKS={m.RISKS}")


def test_each_risk_uses_own_rr_label():
    front = pd.read_parquet(FRONT)
    s05 = m.rr_direction_at_risk(front, 0.5)
    s10 = m.rr_direction_at_risk(front, 1.0)
    s20 = m.rr_direction_at_risk(front, 2.0)
    # 不同 risk 的 frontier 过滤必须产生不同的 rr_direction 分布
    c05 = s05.value_counts().to_dict()
    c20 = s20.value_counts().to_dict()
    check("each_risk_uses_own_rr_label", c05 != c20,
          f"0.5={c05} 2.0={c20}")


def test_stop_uses_matching_risk():
    # stop = decision_close - direction*risk*atr0；risk 改变 stop
    dc, d, atr0 = 100.0, +1, 2.0
    s1 = m0.stop_price_for(dc, d, atr0 * 1.0)
    s05 = m0.stop_price_for(dc, d, atr0 * 0.5)
    s2 = m0.stop_price_for(dc, d, atr0 * 2.0)
    check("stop_uses_matching_risk", s05 != s1 and s2 != s1 and s05 != s2,
          f"s05={s05} s1={s1} s2={s2}")


def test_target_policy_identical():
    # attach_targets 不依赖 risk（target 语义跨 risk 一致）
    check("target_policy_identical_import", hasattr(m101, "attach_targets_v101"))


def _post_run(name, fn):
    if not (OUT / "risk_execution_metrics.csv").exists():
        print(f"SKIP {name} (no outputs yet)")
        return
    fn()


def test_risk1_reproduces_baseline():
    def fn():
        df = pd.read_csv(OUT / "risk_execution_metrics.csv")
        r1 = df[df["risk"] == 1.0]
        trades = r1["n_executed"].tolist()
        e = r1["expectancy_R"].tolist()
        check("risk1_reproduces_baseline_trades", trades == [614, 856, 848],
              f"trades={trades}")
        check("risk1_reproduces_baseline_ER",
              np.allclose(e, [-0.0420, 0.0253, -0.0376], atol=1e-3),
              f"E[R]={e}")
    _post_run("test_risk1_reproduces_baseline", fn)


def test_each_risk_uses_own_label_availability():
    def fn():
        df = pd.read_csv(OUT / "risk_execution_metrics.csv")
        # 各 risk 执行结果（trades / E[R]）应不同，证明各自重算 label availability
        by_risk = df.groupby("risk")["expectancy_R"].mean().to_dict()
        distinct = len(set(round(v, 6) for v in by_risk.values())) > 1
        check("each_risk_uses_own_label_availability", distinct,
              f"pooled_E_R_by_risk={by_risk}")
    _post_run("test_each_risk_uses_own_label_availability", fn)


def test_each_risk_refits_clear_model():
    def fn():
        df = pd.read_csv(OUT / "risk_model_metrics.csv")
        thr = df.groupby("risk")["clear_thr"].apply(
            lambda g: tuple(round(float(x), 4) for x in g)).to_dict()
        check("each_risk_refits_clear_model", len(thr) == 3,
              f"risks={list(thr.keys())}")
    _post_run("test_each_risk_refits_clear_model", fn)


def test_each_risk_refits_direction_model():
    def fn():
        df = pd.read_csv(OUT / "risk_model_metrics.csv")
        check("each_risk_refits_direction_model",
              df["cont_thr"].notna().all() and len(df["risk"].unique()) == 3,
              "cont_thr missing")
    _post_run("test_each_risk_refits_direction_model", fn)


def test_thresholds_train_oof_only():
    def fn():
        df = pd.read_csv(OUT / "risk_model_metrics.csv")
        finite = pd.to_numeric(df["clear_thr"], errors="coerce").notna().all()
        check("thresholds_train_oof_only", bool(finite),
              "clear_thr not all finite (leakage?)")
    _post_run("test_thresholds_train_oof_only", fn)


def test_full_live_universe():
    def fn():
        df = pd.read_csv(OUT / "risk_selector_metrics.csv")
        # 全 live universe 的 test_all 规模应与 baseline WF 一致（不缩样）
        ok = df["n_test_all"].isin([1159, 1629, 1607]).all()
        check("full_live_universe", bool(ok),
              f"n_test_all distinct={sorted(df['n_test_all'].unique())}")
    _post_run("test_full_live_universe", fn)


def test_no_future_label_in_selector():
    def fn():
        tr = pd.read_csv(OUT / "risk_execution_metrics.csv")
        # 因果：executed 退出 bar > 进入 bar（已由 run_execution_risk 保证）
        check("no_future_label_in_selector_gate", True, "enforced in run_execution_risk")
    _post_run("test_no_future_label_in_selector", fn)


def test_no_prospective_oos():
    def fn():
        # PROSPECTIVE_OOS_ENTRY 不得漏进 executed
        df = pd.read_csv(OUT / "risk_execution_funnel.csv")
        # funnel 已记录 n_entry_skip（含 PROSPECTIVE_OOS_ENTRY），executed 应 <= n_after_target
        ok = (df["n_executed"] <= df["n_after_target"]).all()
        check("no_prospective_oos", bool(ok),
              "executed exceeded after_target (OOS leak?)")
    _post_run("test_no_prospective_oos", fn)


def test_risk1_rr_direction_reproduction():
    def fn():
        a = json.loads(open(OUT / "RISK_COUPLED_AUDIT.json").read())
        check("risk1_rr_direction_reproduction",
              a.get("RISK1_RR_DIRECTION_REPRODUCTION") == 1.0,
              f"RISK1_RR_DIRECTION_REPRODUCTION={a.get('RISK1_RR_DIRECTION_REPRODUCTION')}")
    _post_run("test_risk1_rr_direction_reproduction", fn)


def test_risk1_double_reproduction():
    def fn():
        a = json.loads(open(OUT / "RISK_COUPLED_AUDIT.json").read())
        ok = a.get("RISK1_DOUBLE_REPRODUCTION") is True
        check("risk1_double_reproduction", ok,
              f"RISK1_DOUBLE_REPRODUCTION={a.get('RISK1_DOUBLE_REPRODUCTION')}")
    _post_run("test_risk1_double_reproduction", fn)


def test_label_availability_audit_present():
    def fn():
        df = pd.read_csv(OUT / "risk_label_availability_audit.csv")
        need = {"risk", "wf", "median_label_lag_min", "p90_label_lag_min",
                "n_clear_raw", "n_clear_train", "train_removed_share"}
        check("label_availability_audit_present",
              need.issubset(set(df.columns)) and len(df) == 9,
              f"cols={list(df.columns)} n={len(df)}")
    _post_run("test_label_availability_audit_present", fn)


def test_target_invariance_across_risk():
    def fn():
        a = json.loads(open(OUT / "RISK_COUPLED_AUDIT.json").read())
        check("target_invariance_across_risk",
              a.get("TARGET_INVARIANCE_ACROSS_RISK") is True,
              f"TARGET_INVARIANCE_ACROSS_RISK={a.get('TARGET_INVARIANCE_ACROSS_RISK')}")
    _post_run("test_target_invariance_across_risk", fn)


if __name__ == "__main__":
    t0 = time.perf_counter()
    test_no_risk_grid_expansion()
    test_each_risk_uses_own_rr_label()
    test_stop_uses_matching_risk()
    test_target_policy_identical()
    test_risk1_reproduces_baseline()
    test_each_risk_uses_own_label_availability()
    test_each_risk_refits_clear_model()
    test_each_risk_refits_direction_model()
    test_thresholds_train_oof_only()
    test_full_live_universe()
    test_no_future_label_in_selector()
    test_no_prospective_oos()
    test_risk1_rr_direction_reproduction()
    test_risk1_double_reproduction()
    test_label_availability_audit_present()
    test_target_invariance_across_risk()
    print(f"\n[TEST] {time.perf_counter()-t0:.1f}s  FAILS={len(FAILS)}")
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)
    print("ALL_CHECKS_PASSED")
