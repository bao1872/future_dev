"""Tests for FUTURE-R4-M15-DIRECTION-ROBUSTNESS-V1.

Proves:
  - no environment / Candidate / Teacher / dataset rerun (source-level guard)
  - the frozen FIX1 split is reproduced exactly (rows/trades + pooled M0 return)
  - time-block audit partitions TEST rows and never retrains per block
  - symbol robustness partitions TEST trades
  - LOSO covers 15 folds and never trains on the held-out symbol
  - metric is named TeacherFixedExitDirectionReturnATR and is not labelled PnL/alpha
  - cluster bootstraps are deterministic
  - evidence files are written with the explicit metric name
"""

import inspect
import os

import numpy as np
import pandas as pd
import pytest

import research.liquidity_oracle_atlas.direction_robustness_v1 as M


def test_no_environment_or_teacher_rerun():
    src = inspect.getsource(M)
    for tok in ("run_environment_m15", "derive_m15_candidate_gate",
                "build_struct33_dataset", "load_oracle_artifact", "run_dp_m15"):
        assert tok not in src, f"forbidden token present in source: {tok}"


def test_metric_naming_has_no_pnl_drift():
    assert M.METRIC_NAME == "TeacherFixedExitDirectionReturnATR"
    assert "strategy PnL" in M.METRIC_NOT_LABELS
    doc = " ".join((M.__doc__ or "").lower().split())
    assert "not strategy pnl" in doc


def test_cluster_bootstraps_deterministic():
    vals = [0.1, -0.2, 0.5, 0.3, 0.0]
    assert M.symbol_cluster_bootstrap(vals) == M.symbol_cluster_bootstrap(vals)
    blocks = {"2026-05": np.array([1.0, 0.5]), "2026-06": np.array([-1.0]),
              "2026-07": np.array([0.2, 0.4]), "2026-08": np.array([0.9])}
    assert M.month_block_bootstrap(blocks) == M.month_block_bootstrap(blocks)


@pytest.fixture(scope="module")
def run():
    return M.run_direction_robustness(save=True, verbose=False)


def test_frozen_split_reproduced(run):
    s = run["summary"]
    rep = s["frozen_split_reproduction"]
    assert rep["match"] is True
    assert abs(rep["recomputed_pooled_m0_return_atr"]
               - M.FIX1_REFERENCE["pooled_m0_return_atr"]) < 1e-6
    assert s["provenance"]["fix1_base_sha"] == M.FIX1_BASE_SHA
    assert s["provenance"]["confirmation_source_sha"] == M.CONFIRMATION_SOURCE_SHA


def test_time_blocks_partition_and_bootstrap(run):
    s = run["summary"]
    tb = s["time_blocks"]
    pc = tb["partition_check"]
    # opportunity-owned partition is disjoint + complete
    assert pc["pooled_unique_test_trades"] == 638
    assert pc["sum_block_unique_trade_counts"] == pc["pooled_unique_test_trades"]
    assert pc["union_equals_pooled"] is True
    assert pc["pairwise_disjoint"] is True
    # the fixed partition removes the previous double counting (678 -> 638)
    assert pc["pre_fix_row_based_sum_block_trade_counts"] == 678
    assert pc["post_fix_opportunity_owned_sum_block_trade_counts"] == 638

    pooled_rows = s["pooled_m0_test"]["n_rows"]
    assert sum(b["n_rows"] for b in tb["blocks"].values()) == pooled_rows
    assert sum(b["n_trades"] for b in tb["blocks"].values()) == 638

    diag = tb["temporal_block_robustness_diagnostic"]
    assert diag["label"] == "temporal_block_robustness_diagnostic"
    assert diag["n_blocks"] == len(tb["blocks"])
    assert diag["ci_low"] <= diag["ci_high"]
    for b in tb["blocks"].values():
        for k in ("n_rows", "n_trades", "return_atr", "ci_low", "ci_high",
                  "accuracy", "roc_auc", "teacher_long_return_atr",
                  "teacher_short_return_atr"):
            assert k in b


def test_time_block_ownership_is_one_entry_month_per_trade(run):
    # Independent recomputation: each TEST oracle_trade_id has exactly one entry
    # month, blocks are disjoint, and the row-based (pre-fix) grouping double counts.
    sd = run["split_data"]
    ds, test_idx = sd["ds"], sd["test_idx"]
    tids = ds.loc[test_idx, "oracle_trade_id"].to_numpy(object)
    entry_month = pd.to_datetime(ds.loc[test_idx, "oracle_entry_fill_time"]).dt.strftime("%Y-%m")
    df = pd.DataFrame({"tid": tids, "month": entry_month.to_numpy()})
    assert (df.groupby("tid")["month"].nunique() == 1).all()

    per_month_ids = {m: set(g["tid"]) for m, g in df.groupby("month")}
    union = set()
    for v in per_month_ids.values():
        assert not (union & v)          # pairwise disjoint
        union |= v
    assert union == set(tids.tolist())
    assert sum(len(v) for v in per_month_ids.values()) == 638

    row_month = pd.to_datetime(ds.loc[test_idx, "candidate_decision_time"]).dt.strftime("%Y-%m")
    pre = sum(pd.Series(tids).groupby(row_month.to_numpy()).nunique())
    assert pre == 678                   # pre-FIX1 row-based double count


def test_symbol_robustness_partitions_trades(run):
    s = run["summary"]
    sr = s["symbol_robustness"]
    pooled_trades = s["pooled_m0_test"]["n_trades"]
    assert sum(v["n_trades"] for v in sr["per_symbol"].values()) == pooled_trades
    assert sr["positive_symbol_count"] + sr["negative_symbol_count"] <= sr["n_symbols"] == 15
    assert sr["symbol_cluster_bootstrap"]["ci_low"] <= sr["symbol_cluster_bootstrap"]["ci_high"]


def test_loso_covers_all_symbols_and_partitions_trades(run):
    s = run["summary"]
    loso = s["loso"]
    assert len(loso["folds"]) == 15
    pooled_trades = s["pooled_m0_test"]["n_trades"]
    assert sum(f["n_trades"] for f in loso["folds"].values()) == pooled_trades
    assert 0 <= loso["held_out_positive_count"] <= 15
    for f in loso["folds"].values():
        for k in ("n_trades", "return_atr", "accuracy", "roc_auc",
                  "teacher_long_return_atr", "teacher_short_return_atr"):
            assert k in f


def test_loso_excludes_held_out_symbol_from_train_and_val(monkeypatch):
    sd = M.build_frozen_split()
    seen = {}
    orig = M._fit_m0

    def spy(ds, tr, va):
        seen[len(seen)] = (set(ds.loc[tr, "symbol"].unique().tolist()),
                           set(ds.loc[va, "symbol"].unique().tolist()))
        return orig(ds, tr, va)

    monkeypatch.setattr(M, "_fit_m0", spy)
    M.leave_one_symbol_out(sd, symbols=["AG", "CU"], verbose=False)

    tr0, va0 = seen[0]   # held out AG
    tr1, va1 = seen[1]   # held out CU
    assert "AG" not in tr0 and "AG" not in va0
    assert "CU" not in tr1 and "CU" not in va1
    # the OTHER symbol is present in both TRAIN and VALIDATION (real 14-symbol fit)
    assert "CU" in tr0 and "CU" in va0
    assert "AG" in tr1 and "AG" in va1


def test_evidence_files_and_metric_column(run):
    paths = run["paths"]
    for p in (paths["summary"], paths["loso_csv"], paths["time_blocks_csv"]):
        assert os.path.exists(p)
    for key in ("loso_csv", "time_blocks_csv"):
        df = pd.read_csv(paths[key])
        assert (df["metric"] == M.METRIC_NAME).all()
    blk = pd.read_csv(paths["time_blocks_csv"])
    assert set(("block", "n_rows", "n_trades", "m0_return_atr", "ci_low", "ci_high",
                "teacher_long_return_atr", "teacher_short_return_atr", "accuracy",
                "auc")).issubset(blk.columns)
    loso = pd.read_csv(paths["loso_csv"])
    assert len(loso) == 15
    assert set(("held_out_symbol", "n_test_rows", "n_test_trades", "return_atr",
                "accuracy", "auc", "teacher_long_return_atr",
                "teacher_short_return_atr")).issubset(loso.columns)
