"""direction_robustness_v1
========================

FUTURE-R4-M15-DIRECTION-ROBUSTNESS-V1

Robustness audit of the frozen 15-symbol Direction result (FUTURE-R4-M15-DIRECTION-
MODEL-V1-15SYM-CONFIRMATION FIX1). PRIMARY MODEL = DIR-M0 = DTP9 (frozen). No model,
feature, label, Teacher, Candidate or hyperparameter is changed; no threshold tuning;
TEST is never used for selection.

The frozen split (common study window + 15m-snapped T1/T2) is reproduced by REUSING
the trainer's exact helpers, so the audit is guaranteed to run on the same TEST rows
as the frozen confirmation evidence. A hard reproduction assert guards against drift.

Three audits:
  B. TEST TIME-BLOCK audit: split the frozen TEST interval into calendar-month blocks.
     Per block: n rows, n Oracle trades, M0 return, trade-level 95% CI, LONG/SHORT
     return, accuracy, AUC. No retraining per block. Plus a MONTH-BLOCK bootstrap
     (resample months with replacement) to test whether the pooled positive result
     survives temporal clustering.
  C. SYMBOL robustness: per-symbol TEST metrics; positive/negative symbol count,
     mean/median per-symbol return; SYMBOL-CLUSTER bootstrap (resample symbols).
  D. LEAVE-ONE-SYMBOL-OUT (15 folds): train M0 on the other 14 symbols (fit on their
     TRAIN, early-stop on their VALIDATION), evaluate ONLY the held-out symbol's TEST.
     Frozen params, no per-fold tuning. Aggregate mean/median/positive-count +
     symbol-cluster bootstrap.

METRIC (E): all reported returns are
    TeacherFixedExitDirectionReturnATR
= (+entry_quality_atr if predicted direction == Teacher direction else
   -entry_quality_atr), aggregated per Oracle opportunity (trade), bootstrapped at
   the trade level. It is a teacher-fixed-exit DIRECTION DIAGNOSTIC. It is NOT
   strategy PnL, NOT live alpha and NOT a tradable return.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.train_direction_model_15sym_v1 import (
    BASE_PARAMS,
    BASE_SHA,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIRMATION_SOURCE_SHA,
    DECISION_THRESHOLD,
    DTP9,
    EVIDENCE_DIR,
    FIX1_BASE_SHA,
    FRAC_TRAIN,
    FRAC_VAL,
    SYMBOLS,
    aggregate_per_trade,
    bootstrap_trade_returns_chunked,
    common_calendar,
    common_window_eligibility,
    evaluate_subset,
    fit_direction_model,
    load_pooled,
    pred_direction_return_atr,
    predict_direction,
    prepare_xy,
    remove_boundary_opportunities,
)

TASK_ID = "FUTURE-R4-M15-DIRECTION-ROBUSTNESS-V1"
BASE_SHA_ROBUSTNESS = "686cb06fa5199f2712b1d8f8cfddd77f6c1c0a79"
PRIMARY_MODEL = "DIR-M0 = DTP9"

METRIC_NAME = "TeacherFixedExitDirectionReturnATR"
METRIC_DEFINITION = (
    "(+entry_quality_atr if predicted direction == Teacher direction else "
    "-entry_quality_atr), aggregated per Oracle opportunity (trade), bootstrapped "
    "at the trade level"
)
METRIC_NOT_LABELS = ["strategy PnL", "live alpha", "tradable return"]

EVIDENCE_SUMMARY = os.path.join(EVIDENCE_DIR, "direction_robustness_v1_summary.json")
EVIDENCE_LOSO = os.path.join(EVIDENCE_DIR, "direction_robustness_loso.csv")
EVIDENCE_BLOCKS = os.path.join(EVIDENCE_DIR, "direction_robustness_time_blocks.csv")

BOOTSTRAP_CHUNK = 500

# Frozen FIX1 (4e1629f) TEST shape used to guard the audit against split drift.
FIX1_REFERENCE = {
    "source_sha": CONFIRMATION_SOURCE_SHA,
    "test_rows": 13773,
    "test_trades": 638,
    "pooled_m0_return_atr": 0.6014965284150177,
}


# --------------------------------------------------------------------------- #
# Frozen split reproduction (reuses the trainer helpers verbatim)               #
# --------------------------------------------------------------------------- #
def build_frozen_split(symbols=SYMBOLS, frac_train: float = FRAC_TRAIN,
                       frac_val: float = FRAC_VAL) -> dict:
    ds = load_pooled(symbols)
    cal = common_calendar(ds, symbols, frac_train, frac_val)
    cuts = cal["cuts"]
    dt = pd.to_datetime(ds["candidate_decision_time"]).to_numpy(dtype="datetime64[ns]")
    split = np.searchsorted(cuts, dt, side="right")
    win_ok, window_report = common_window_eligibility(ds, cal["start_ns"], cal["end_ns"])
    kept, boundary_report = remove_boundary_opportunities(
        ds.assign(label_eligible=win_ok), split, cuts
    )
    k = np.flatnonzero(kept)
    return {
        "ds": ds,
        "cal": cal,
        "cuts": cuts,
        "split": split,
        "kept": kept,
        "train_idx": k[split[k] == 0],
        "val_idx": k[split[k] == 1],
        "test_idx": k[split[k] == 2],
        "window_report": window_report,
        "boundary_report": boundary_report,
    }


def _fit_m0(ds, train_idx, val_idx):
    Xtr, ytr, wtr = prepare_xy(ds, train_idx, DTP9)
    wtr_norm = wtr / wtr.mean() if wtr.mean() > 0 else wtr
    Xv, yv, wv = prepare_xy(ds, val_idx, DTP9)
    return fit_direction_model(Xtr, ytr, wtr_norm, Xv, yv, wv)


def _predict_m0(model, ds, idx):
    return predict_direction(model, ds.loc[idx, list(DTP9)])


# --------------------------------------------------------------------------- #
# Metrics helpers                                                               #
# --------------------------------------------------------------------------- #
def _trade_returns(ds, idx, pred_dir):
    y = (ds.loc[idx, "oracle_direction"].to_numpy(object) == "LONG").astype(np.uint8)
    w = ds.loc[idx, "sample_weight_raw"].to_numpy(float)
    eq = ds.loc[idx, "entry_quality_atr"].to_numpy(float)
    tids = ds.loc[idx, "oracle_trade_id"].to_numpy(object)
    return aggregate_per_trade(tids, w, pred_direction_return_atr(pred_dir, y, eq))


def _subset_eval(ds, idx, pred_dir, p_long):
    """ALL + TEACHER_LONG + TEACHER_SHORT metrics (all returns = METRIC_NAME)."""
    empty = {
        "n_rows": 0, "n_trades": 0, "return_atr": None, "ci_low": None, "ci_high": None,
        "accuracy": None, "roc_auc": None,
        "teacher_long_return_atr": None, "teacher_long_n_trades": 0,
        "teacher_short_return_atr": None, "teacher_short_n_trades": 0,
    }
    if idx.size == 0:
        return empty
    allb = evaluate_subset(ds, idx, pred_dir, p_long)
    y = ds.loc[idx, "oracle_direction"].to_numpy(object) == "LONG"
    lb = evaluate_subset(ds, idx[y], pred_dir[y], None if p_long is None else p_long[y])
    sb = evaluate_subset(ds, idx[~y], pred_dir[~y], None if p_long is None else p_long[~y])
    return {
        "n_rows": allb["n_rows"], "n_trades": allb["n_trades"],
        "return_atr": allb["return_atr"], "ci_low": allb["ci_low"], "ci_high": allb["ci_high"],
        "accuracy": allb["accuracy"], "roc_auc": allb["roc_auc"],
        "teacher_long_return_atr": lb["return_atr"], "teacher_long_n_trades": lb["n_trades"],
        "teacher_short_return_atr": sb["return_atr"], "teacher_short_n_trades": sb["n_trades"],
    }


# --------------------------------------------------------------------------- #
# Cluster bootstraps                                                            #
# --------------------------------------------------------------------------- #
def symbol_cluster_bootstrap(values, B: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED):
    """Resample the symbol-level values (one per symbol) with replacement."""
    arr = np.asarray([float(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    draws = arr[rng.integers(0, arr.size, size=(B, arr.size))].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(arr.mean()), float(lo), float(hi)


def month_block_bootstrap(block_trade_returns: dict, B: int = BOOTSTRAP_REPLICATES,
                          seed: int = BOOTSTRAP_SEED):
    """Resample calendar-month blocks with replacement; pool their trade returns."""
    keys = sorted(block_trade_returns)
    groups = [np.asarray(block_trade_returns[k], dtype=float) for k in keys]
    nb = len(groups)
    if nb == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    stats = np.empty(B, dtype=float)
    for i in range(B):
        pick = rng.integers(0, nb, size=nb)
        stats[i] = np.concatenate([groups[j] for j in pick]).mean()
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(stats.mean()), float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Audits                                                                        #
# --------------------------------------------------------------------------- #
def time_block_audit(ds, test_idx, pred_dir, p_long, seed: int = BOOTSTRAP_SEED) -> dict:
    months = pd.to_datetime(ds.loc[test_idx, "candidate_decision_time"]).dt.strftime("%Y-%m").to_numpy(object)
    blocks, block_trade_returns = {}, {}
    for m in sorted(set(months.tolist())):
        m_mask = months == m
        idx = test_idx[m_mask]
        pm = pred_dir[m_mask]
        pl = None if p_long is None else p_long[m_mask]
        blocks[m] = _subset_eval(ds, idx, pm, pl)
        tr, _ = _trade_returns(ds, idx, pm)
        block_trade_returns[m] = tr

    all_tr, _ = _trade_returns(ds, test_idx, pred_dir)
    bmean, blo, bhi = month_block_bootstrap(block_trade_returns, seed=seed)
    return {
        "blocks": blocks,
        "overall_return_atr": float(all_tr.mean()),
        "block_bootstrap": {
            "n_blocks": len(block_trade_returns),
            "cluster": "calendar month (candidate_decision_time)",
            "resampled_mean": bmean, "ci_low": blo, "ci_high": bhi,
        },
    }


def symbol_robustness(ds, test_idx, pred_dir, p_long, symbols, seed: int = BOOTSTRAP_SEED) -> dict:
    sym_test = ds.loc[test_idx, "symbol"].to_numpy(object)
    per = {}
    for s in symbols:
        m = sym_test == s
        idx = test_idx[m]
        per[s] = _subset_eval(ds, idx, pred_dir[m], None if p_long is None else p_long[m])

    rets = [per[s]["return_atr"] for s in symbols if per[s]["return_atr"] is not None]
    pos = int(sum(1 for r in rets if r > 0))
    neg = int(sum(1 for r in rets if r < 0))
    mean, lo, hi = symbol_cluster_bootstrap(rets, seed=seed)
    return {
        "per_symbol": per,
        "positive_symbol_count": pos,
        "negative_symbol_count": neg,
        "n_symbols": len(rets),
        "mean_symbol_return_atr": float(np.mean(rets)) if rets else None,
        "median_symbol_return_atr": float(np.median(rets)) if rets else None,
        "symbol_cluster_bootstrap": {
            "cluster": "symbol", "mean": mean, "ci_low": lo, "ci_high": hi,
        },
    }


def leave_one_symbol_out(split_data: dict, symbols=SYMBOLS, seed: int = BOOTSTRAP_SEED,
                         verbose: bool = True) -> dict:
    ds = split_data["ds"]
    train_idx, val_idx, test_idx = split_data["train_idx"], split_data["val_idx"], split_data["test_idx"]
    sym_all = ds["symbol"].to_numpy(object)

    folds = {}
    for s in symbols:
        tr = train_idx[sym_all[train_idx] != s]
        va = val_idx[sym_all[val_idx] != s]
        te = test_idx[sym_all[test_idx] == s]
        model = _fit_m0(ds, tr, va)
        pdir, plong = _predict_m0(model, ds, te)
        folds[s] = _subset_eval(ds, te, pdir, plong)
        if verbose:
            print(f"[LOSO] held_out={s} test_trades={folds[s]['n_trades']} "
                  f"return_atr={folds[s]['return_atr']}")

    rets = [folds[s]["return_atr"] for s in symbols if folds[s]["return_atr"] is not None]
    pos = int(sum(1 for r in rets if r > 0))
    mean, lo, hi = symbol_cluster_bootstrap(rets, seed=seed)
    return {
        "folds": folds,
        "held_out_mean_return_atr": float(np.mean(rets)) if rets else None,
        "held_out_median_return_atr": float(np.median(rets)) if rets else None,
        "held_out_positive_count": pos,
        "held_out_n_symbols": len(rets),
        "symbol_cluster_bootstrap": {
            "cluster": "held-out symbol", "mean": mean, "ci_low": lo, "ci_high": hi,
        },
    }


# --------------------------------------------------------------------------- #
# Orchestration                                                                 #
# --------------------------------------------------------------------------- #
def run_direction_robustness(symbols=SYMBOLS, save: bool = True, verbose: bool = True) -> dict:
    split_data = build_frozen_split(symbols)
    ds = split_data["ds"]
    test_idx = split_data["test_idx"]

    # ---- frozen-split reproduction guard ----
    n_test = int(test_idx.size)
    n_test_trades = int(pd.unique(ds.loc[test_idx, "oracle_trade_id"]).size)
    if n_test != FIX1_REFERENCE["test_rows"] or n_test_trades != FIX1_REFERENCE["test_trades"]:
        raise RuntimeError(
            "STOP_15SYM_ROBUSTNESS_SPLIT_DRIFT:"
            f"{n_test}/{n_test_trades} != {FIX1_REFERENCE['test_rows']}/{FIX1_REFERENCE['test_trades']}"
        )

    # ---- pooled M0 (frozen params) -> TEST predictions ----
    pooled_model = _fit_m0(ds, split_data["train_idx"], split_data["val_idx"])
    pooled_dir, pooled_long = _predict_m0(pooled_model, ds, test_idx)
    pooled_eval = _subset_eval(ds, test_idx, pooled_dir, pooled_long)

    repro_ok = abs(pooled_eval["return_atr"] - FIX1_REFERENCE["pooled_m0_return_atr"]) < 1e-6
    if not repro_ok:
        raise RuntimeError(
            "STOP_15SYM_ROBUSTNESS_POOLED_M0_REPRODUCTION_FAIL:"
            f"{pooled_eval['return_atr']} != {FIX1_REFERENCE['pooled_m0_return_atr']}"
        )

    if verbose:
        print(f"[ROBUST] pooled M0 TEST return={pooled_eval['return_atr']} "
              f"trades={pooled_eval['n_trades']} (FIX1 reproduced)")

    # ---- B. time blocks ----
    time_blocks = time_block_audit(ds, test_idx, pooled_dir, pooled_long)
    # ---- C. symbol robustness ----
    sym_rob = symbol_robustness(ds, test_idx, pooled_dir, pooled_long, symbols)
    # ---- D. leave-one-symbol-out ----
    loso = leave_one_symbol_out(split_data, symbols, verbose=verbose)

    summary = {
        "task_id": TASK_ID,
        "base_sha": BASE_SHA_ROBUSTNESS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_model": PRIMARY_MODEL,
        "provenance": {
            "confirmation_base_sha": BASE_SHA,
            "fix1_base_sha": FIX1_BASE_SHA,
            "confirmation_source_sha": CONFIRMATION_SOURCE_SHA,
            "robustness_base_sha": BASE_SHA_ROBUSTNESS,
        },
        "metric": {
            "name": METRIC_NAME,
            "definition": METRIC_DEFINITION,
            "is_not": METRIC_NOT_LABELS,
            "note": ("Teacher-fixed-exit direction diagnostic. The exit is the Teacher "
                     "(oracle) exit, so this does NOT represent a strategy that must "
                     "choose its own exit."),
        },
        "frozen_split_reproduction": {
            "fix1_source_sha": FIX1_REFERENCE["source_sha"],
            "fix1_test_rows": FIX1_REFERENCE["test_rows"],
            "fix1_test_trades": FIX1_REFERENCE["test_trades"],
            "fix1_pooled_m0_return_atr": FIX1_REFERENCE["pooled_m0_return_atr"],
            "recomputed_pooled_m0_return_atr": pooled_eval["return_atr"],
            "match": bool(repro_ok),
        },
        "calendar": {
            "t1": split_data["cal"]["t1"].isoformat(),
            "t2": split_data["cal"]["t2"].isoformat(),
            "snap_rule": split_data["cal"]["snap_rule"],
        },
        "pooled_m0_test": pooled_eval,
        "time_blocks": time_blocks,
        "symbol_robustness": sym_rob,
        "loso": loso,
        "params": {"base_params": BASE_PARAMS, "decision_threshold": DECISION_THRESHOLD,
                   "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED},
    }

    # ---- evidence ----
    if save:
        os.makedirs(EVIDENCE_DIR, exist_ok=True)
        with open(EVIDENCE_SUMMARY, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)

        blk_rows = []
        for m, b in time_blocks["blocks"].items():
            blk_rows.append({
                "metric": METRIC_NAME, "block": m,
                "n_rows": b["n_rows"], "n_trades": b["n_trades"],
                "m0_return_atr": b["return_atr"], "ci_low": b["ci_low"], "ci_high": b["ci_high"],
                "teacher_long_return_atr": b["teacher_long_return_atr"],
                "teacher_short_return_atr": b["teacher_short_return_atr"],
                "accuracy": b["accuracy"], "auc": b["roc_auc"],
            })
        pd.DataFrame(blk_rows).to_csv(EVIDENCE_BLOCKS, index=False)

        loso_rows = []
        for s in symbols:
            f = loso["folds"][s]
            loso_rows.append({
                "metric": METRIC_NAME, "held_out_symbol": s,
                "n_test_rows": f["n_rows"], "n_test_trades": f["n_trades"],
                "return_atr": f["return_atr"], "ci_low": f["ci_low"], "ci_high": f["ci_high"],
                "accuracy": f["accuracy"], "auc": f["roc_auc"],
                "teacher_long_return_atr": f["teacher_long_return_atr"],
                "teacher_short_return_atr": f["teacher_short_return_atr"],
            })
        pd.DataFrame(loso_rows).to_csv(EVIDENCE_LOSO, index=False)

    return {
        "summary": summary,
        "split_data": split_data,
        "pooled_pred": (pooled_dir, pooled_long),
        "paths": {"summary": EVIDENCE_SUMMARY, "loso_csv": EVIDENCE_LOSO,
                  "time_blocks_csv": EVIDENCE_BLOCKS},
    }


if __name__ == "__main__":
    r = run_direction_robustness()
    s = r["summary"]
    print(json.dumps({
        "task_id": s["task_id"],
        "metric": s["metric"]["name"],
        "pooled_m0_test": {k: s["pooled_m0_test"][k] for k in
                           ("n_rows", "n_trades", "return_atr", "ci_low", "ci_high",
                            "accuracy", "roc_auc", "teacher_long_return_atr",
                            "teacher_short_return_atr")},
        "time_blocks": {m: {"n_trades": b["n_trades"], "return_atr": b["return_atr"],
                            "ci_low": b["ci_low"], "ci_high": b["ci_high"],
                            "teacher_long_return_atr": b["teacher_long_return_atr"],
                            "teacher_short_return_atr": b["teacher_short_return_atr"]}
                        for m, b in s["time_blocks"]["blocks"].items()},
        "time_block_bootstrap": s["time_blocks"]["block_bootstrap"],
        "symbol_robustness": {
            "positive_symbol_count": s["symbol_robustness"]["positive_symbol_count"],
            "negative_symbol_count": s["symbol_robustness"]["negative_symbol_count"],
            "mean_symbol_return_atr": s["symbol_robustness"]["mean_symbol_return_atr"],
            "median_symbol_return_atr": s["symbol_robustness"]["median_symbol_return_atr"],
            "bootstrap": s["symbol_robustness"]["symbol_cluster_bootstrap"]},
        "loso_aggregate": {
            "mean": s["loso"]["held_out_mean_return_atr"],
            "median": s["loso"]["held_out_median_return_atr"],
            "positive": s["loso"]["held_out_positive_count"],
            "bootstrap": s["loso"]["symbol_cluster_bootstrap"]},
    }, indent=2, default=str))
    print("paths:", r["paths"])
