"""Validation tests for the Phase-2 M1 cost-robustness formal experiment.

These tests DO NOT re-run the experiment. They assert that the committed
evidence artifacts are complete and internally consistent with the frozen
Phase-1 cost kernel contract:

  * the join hard gate (identical joined row universe across all five kappas);
  * the primary statistic DeltaSTRUCT(kappa) = WRMSE(M0) - WRMSE(M1) > 0 on Test
    and on the unseen-symbol holdout, for every kappa;
  * the episode-bootstrap 95% CI lower bound > 0 for the M1-M0 comparison;
  * the frozen LightGBM version (4.7.0).

Run:
  .venv/bin/python -m pytest research/liquidity_oracle_atlas/test_entry_value_cost_robustness_phase2_formal_v1.py -v
"""

from pathlib import Path

import json

import pandas as pd
import pytest

OUT = Path("artifacts/entry_value_cost_robustness_m1_v1")
KAPPAS = [0.0, 0.005, 0.01, 0.02, 0.05]
ARTIFACTS = [
    "cost_label_stats.csv",
    "cost_metrics.csv",
    "cost_deciles.csv",
    "cost_symbol_metrics.csv",
    "cost_cross_symbol_metrics.csv",
    "cost_bootstrap.csv",
    "cost_target_stability.csv",
    "cost_curve.csv",
    "cost_summary.json",
]


def test_artifacts_present():
    for a in ARTIFACTS:
        assert (OUT / a).exists(), f"missing required artifact {a}"


def test_join_hard_gate():
    s = json.load(open(OUT / "cost_summary.json"))
    audit = s["join_audit"]
    assert audit["hard_gate_universe_identical"] is True
    n_t2 = audit["n_t2_rows"]
    for k in KAPPAS:
        assert audit["matched_rows_per_kappa"][str(k)] == n_t2, (
            f"kappa={k} matched universe != t2 rows (kappa-dependent candidate?)"
        )


def test_curve_completeness_and_sign():
    c = pd.read_csv(OUT / "cost_curve.csv")
    assert list(c["kappa"]) == KAPPAS, "cost_curve.csv must have exactly the 5 frozen kappas"
    for _, r in c.iterrows():
        assert r["DeltaSTRUCT_long"] > 0, f"long DeltaSTRUCT<=0 at kappa={r['kappa']}"
        assert r["DeltaSTRUCT_short"] > 0, f"short DeltaSTRUCT<=0 at kappa={r['kappa']}"
        assert r["DeltaSTRUCT_unseen_long"] > 0, (
            f"unseen long DeltaSTRUCT<=0 at kappa={r['kappa']}"
        )
        assert r["DeltaSTRUCT_unseen_short"] > 0, (
            f"unseen short DeltaSTRUCT<=0 at kappa={r['kappa']}"
        )


def test_bootstrap_ci_excludes_zero():
    b = pd.read_csv(OUT / "cost_bootstrap.csv")
    sub = b[b["comparison"] == "M1-M0"]
    assert len(sub) == 2 * len(KAPPAS), "need M1-M0 bootstrap rows for both directions"
    for _, r in sub.iterrows():
        assert r["ci_lower"] > 0, (
            f"bootstrap CI lower<=0 for {r['kappa']}/{r['direction']}"
        )
        assert r["ci_excludes_zero"] is True


def test_symbol_metrics_rows():
    sm = pd.read_csv(OUT / "cost_symbol_metrics.csv")
    # kappa x 15 symbols x 2 directions x 2 models
    assert len(sm) == len(KAPPAS) * 15 * 2 * 2, f"unexpected symbol_metric rows: {len(sm)}"
    assert sm["model_wRMSE"].notna().all()
    assert sm["baseline_wRMSE"].notna().all()


def test_label_stats_rows():
    ls = pd.read_csv(OUT / "cost_label_stats.csv")
    assert len(ls) == len(KAPPAS)
    assert (ls["n_candidate"] > 0).all()


def test_lightgbm_frozen():
    import research.liquidity_oracle_atlas.experiment_entry_value_tree_core108_t15_v1 as T  # noqa: F401
    assert T.check_lightgbm() == "4.7.0"
