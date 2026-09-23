"""train_direction_model_ag_v1
===========================

FUTURE-R4-M15-DIRECTION-MODEL-V1-AG

Direction-only models trained on the EXISTING Phase-1 AG Candidate->Teacher
dataset (produced by the Phase-1 Candidate->Teacher builder, BASE_SHA df868eb).

Hard governance contracts (enforced structurally + by tests):
  1. NO environment rerun. We only read the frozen Phase-1 parquet. We do NOT
     import or call the execution-environment runner, the candidate-gate
     builder, the Phase-1 dataset builder, or the overnight Teacher loader /
     DP runner. Features were already computed continuously over the full
     history before any split.
  2. Train/Val/Test are PURE row slicing by candidate_decision_time. The split
     module never touches raw market data or indicators.
  3. Boundary rule: a retained Candidate must have decision / fill / oracle_entry
     / oracle_exit times inside the SAME provisional split; then if ANY Candidate
     of an oracle_trade_id violates that, the ENTIRE oracle_trade_id is dropped
     from ALL splits (no 50/100/200-bar purge, no opportunity leakage).
  4. Only direction is trained. entry_quality_atr is used ONLY to translate a
     predicted direction into an economic return; no EntryQuality model.

Two models, identical fixed params:
  DIR-M0 : DTP9  (15m/1h/4h trend only)
  DIR-M1 : STRUCT33 (trend + SR + Liquidity)

Primary metric: MeanPredictedDirectionReturnATR per Oracle opportunity, with a
trade-level (NOT row-level) bootstrap 95% CI. Plus always-long / always-short /
train-majority baselines and an M1-vs-M0 paired trade-level bootstrap.

Phase breakdown (ALL / BEFORE_ENTRY / AT_ENTRY / IN_POSITION) is reported
separately and never merged across phases.
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# --------------------------------------------------------------------------- #
# Frozen feature schemas (must match the Phase-1 dataset builder exactly).
# Hardcoded here on purpose so this module imports NOTHING from the builder /
# environment / Teacher modules (no rerun risk). test_schemas_match_builder
# cross-checks against the source of truth.
# --------------------------------------------------------------------------- #
TF_ORDER = ("m15", "h1", "h4")
_FEAT_PER_TF = (
    "trend_state",
    "slope_atr",
    "dev",
    "sr_support_dist_atr",
    "sr_resistance_dist_atr",
    "sr_support_strength",
    "sr_resistance_strength",
    "liq_up_dist_atr",
    "liq_down_dist_atr",
    "liq_up_count",
    "liq_down_count",
)
STRUCT33 = tuple(f"{tf}_{n}" for tf in TF_ORDER for n in _FEAT_PER_TF)
DTP9 = tuple(f"{tf}_{n}" for tf in TF_ORDER for n in ("trend_state", "slope_atr", "dev"))

assert len(DTP9) == 9
assert len(STRUCT33) == 33
assert set(DTP9).issubset(STRUCT33)

TASK_ID = "FUTURE-R4-M15-DIRECTION-MODEL-V1-AG"
BASE_SHA = "df868eb8790438ee55db3fc817bcf2650bc994e3"

DATASET_PARQUET = os.path.join(
    "artifacts", "struct33_dataset_v1", "{symbol}", "candidate_teacher_dataset.parquet"
)
EVIDENCE_DIR = os.path.join("research", "liquidity_oracle_atlas", "evidence")
ARTIFACT_DIR = os.path.join("artifacts", "direction_model_ag_v1")

# Fixed params for BOTH models. No hyperparameter search, no threshold tuning.
BASE_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=100,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_estimators=2000,
    random_state=20260923,
    n_jobs=-1,
    verbosity=-1,
)

FRAC_TRAIN = 0.60
FRAC_VAL = 0.20
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260923
DECISION_THRESHOLD = 0.5


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_dataset(symbol: str = "AG") -> pd.DataFrame:
    path = DATASET_PARQUET.format(symbol=symbol)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"STOP: Phase-1 dataset parquet missing: {path}. "
            "Run the Phase-1 dataset builder first."
        )
    return pd.read_parquet(path)


# --------------------------------------------------------------------------- #
# Chronological split
# --------------------------------------------------------------------------- #
def make_chronological_splits(
    ds: pd.DataFrame,
    frac_train: float = FRAC_TRAIN,
    frac_val: float = FRAC_VAL,
):
    """Assign each row to TRAIN/VAL/TEST by candidate_decision_time quantile.

    TRAIN: decision_time < T1
    VAL  : T1 <= decision_time < T2
    TEST : decision_time >= T2

    Cuts are frozen quantiles of the full candidate_decision_time series and are
    persisted in the summary (must not be moved to improve results).
    """
    dt = pd.to_datetime(ds["candidate_decision_time"]).to_numpy(dtype="datetime64[ns]")
    ns = dt.astype("int64")
    q1 = int(np.quantile(ns, frac_train))
    q2 = int(np.quantile(ns, frac_train + frac_val))
    cuts = np.array([q1, q2], dtype="datetime64[ns]")
    split = np.searchsorted(cuts, dt, side="right")  # 0/1/2
    return split, cuts


# --------------------------------------------------------------------------- #
# Boundary Oracle-opportunity removal (vectorized)
# --------------------------------------------------------------------------- #
def remove_boundary_opportunities(
    ds: pd.DataFrame, split: np.ndarray, cuts: np.ndarray
):
    """Drop whole Oracle trades that cross a split boundary.

    A retained row must have decision/fill/oracle_entry/oracle_exit in the SAME
    split. Then ANY trade with at least one violating Candidate is dropped
    entirely. Returns (kept_mask_over_all_rows, report_dict).
    """
    eligible = ds["label_eligible"].to_numpy(bool)
    e = np.flatnonzero(eligible)

    decision_t = pd.to_datetime(ds["candidate_decision_time"]).to_numpy(
        dtype="datetime64[ns]"
    )[e]
    fill_t = pd.to_datetime(ds["candidate_fill_time"]).to_numpy(dtype="datetime64[ns]")[
        e
    ]
    entry_t = pd.to_datetime(ds["oracle_entry_fill_time"]).to_numpy(
        dtype="datetime64[ns]"
    )[e]
    exit_t = pd.to_datetime(ds["oracle_exit_fill_time"]).to_numpy(
        dtype="datetime64[ns]"
    )[e]

    sp_dec = np.searchsorted(cuts, decision_t, side="right")
    sp_fill = np.searchsorted(cuts, fill_t, side="right")
    sp_entry = np.searchsorted(cuts, entry_t, side="right")
    sp_exit = np.searchsorted(cuts, exit_t, side="right")

    row_same_split = (
        (sp_dec == sp_fill) & (sp_dec == sp_entry) & (sp_dec == sp_exit)
    )

    trade_ids = ds["oracle_trade_id"].to_numpy(object)[e]
    uniq, inv = np.unique(trade_ids, return_inverse=True)

    bad_trade = np.zeros(len(uniq), dtype=bool)
    np.logical_or.at(bad_trade, inv, ~row_same_split)

    keep_eligible = row_same_split & ~bad_trade[inv]

    kept = np.zeros(len(ds), dtype=bool)
    kept[e] = keep_eligible

    # classify dropped trades by which boundary they straddle (report only)
    dropped = bad_trade  # per unique trade
    t1 = 0
    t2 = 0
    for ti in np.flatnonzero(dropped):
        tid = uniq[ti]
        member = trade_ids == tid
        spans = np.unique(
            np.concatenate(
                [
                    sp_dec[member],
                    sp_fill[member],
                    sp_entry[member],
                    sp_exit[member],
                ]
            )
        )
        if 0 in spans and (1 in spans or 2 in spans):
            t1 += 1
        elif 1 in spans and 2 in spans:
            t2 += 1

    report = {
        "eligible_before": int(eligible.sum()),
        "eligible_after": int(kept.sum()),
        "candidate_rows_removed": int((eligible & ~kept).sum()),
        "trades_total": int(len(uniq)),
        "trades_dropped_total": int(dropped.sum()),
        "trades_dropped_t1": int(t1),
        "trades_dropped_t2": int(t2),
    }
    return kept, report


# --------------------------------------------------------------------------- #
# XY assembly
# --------------------------------------------------------------------------- #
def prepare_xy(ds: pd.DataFrame, idx: np.ndarray, feature_cols):
    assert tuple(feature_cols) in (DTP9, STRUCT33), (
        "feature_cols must be exactly DTP9 or STRUCT33"
    )
    X = ds.loc[idx, list(feature_cols)].copy()
    y = (ds.loc[idx, "oracle_direction"].to_numpy(object) == "LONG").astype(np.uint8)
    w = ds.loc[idx, "sample_weight_raw"].to_numpy(float)
    return X, y, w


# --------------------------------------------------------------------------- #
# Model fit / predict
# --------------------------------------------------------------------------- #
def fit_direction_model(X_train, y_train, w_train, X_val, y_val, w_val):
    model = lgb.LGBMClassifier(**BASE_PARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[w_val],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
    return model


def predict_direction(model, X):
    p_long = model.predict_proba(X)[:, 1]
    pred_dir = (p_long >= DECISION_THRESHOLD).astype(np.uint8)
    return pred_dir, p_long


# --------------------------------------------------------------------------- #
# Economic return + aggregation + bootstrap
# --------------------------------------------------------------------------- #
def pred_direction_return_atr(
    pred_dir: np.ndarray, teacher_dir: np.ndarray, entry_quality: np.ndarray
) -> np.ndarray:
    correct = pred_dir == teacher_dir
    return np.where(correct, entry_quality, -entry_quality)


def aggregate_per_trade(trade_ids, weights, values):
    uniq, inv = np.unique(trade_ids, return_inverse=True)
    num = np.bincount(inv, weights=weights * values, minlength=len(uniq))
    den = np.bincount(inv, weights=weights, minlength=len(uniq))
    trade_return = num / den
    return trade_return, uniq


def bootstrap_trade_returns(trade_return: np.ndarray, B: int = BOOTSTRAP_REPLICATES):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(trade_return)
    if n == 0:
        return 0.0, 0.0, 0.0
    idx = rng.integers(0, n, size=(B, n))
    boot = trade_return[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(trade_return.mean()), float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_class_metrics(y_true, pred_dir, weights, pred_long_proba=None):
    acc = float(np.average(pred_dir == y_true, weights=weights))
    sens = []
    for c in (0, 1):
        m = y_true == c
        if m.sum() > 0:
            sens.append(float(np.average(pred_dir[m] == c, weights=weights[m])))
        else:
            sens.append(float("nan"))
    bal_acc = float(np.nanmean(sens))
    auc = None
    if pred_long_proba is not None and len(np.unique(y_true)) == 2:
        try:
            auc = float(
                roc_auc_score(y_true, pred_long_proba, sample_weight=weights)
            )
        except ValueError:
            auc = None
    return {"accuracy": acc, "balanced_accuracy": bal_acc, "roc_auc": auc}


# --------------------------------------------------------------------------- #
# Evaluate one method on a row subset
# --------------------------------------------------------------------------- #
def evaluate_subset(
    ds: pd.DataFrame,
    idx: np.ndarray,
    pred_dir: np.ndarray,
    pred_long_proba: np.ndarray | None,
):
    if idx.size == 0:
        return {
            "n_rows": 0,
            "n_trades": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "roc_auc": None,
            "return_atr": None,
            "ci_low": None,
            "ci_high": None,
        }
    y = (ds.loc[idx, "oracle_direction"].to_numpy(object) == "LONG").astype(np.uint8)
    w = ds.loc[idx, "sample_weight_raw"].to_numpy(float)
    eq = ds.loc[idx, "entry_quality_atr"].to_numpy(float)
    tids = ds.loc[idx, "oracle_trade_id"].to_numpy(object)

    pdr = pred_direction_return_atr(pred_dir, y, eq)
    trade_ret, uniq = aggregate_per_trade(tids, w, pdr)
    mean_ret, lo, hi = bootstrap_trade_returns(trade_ret)

    cls = compute_class_metrics(y, pred_dir, w, pred_long_proba)
    return {
        "n_rows": int(idx.size),
        "n_trades": int(len(uniq)),
        "accuracy": cls["accuracy"],
        "balanced_accuracy": cls["balanced_accuracy"],
        "roc_auc": cls["roc_auc"],
        "return_atr": mean_ret,
        "ci_low": lo,
        "ci_high": hi,
    }


def phase_mask(ds: pd.DataFrame, idx: np.ndarray, phase: str):
    be = ds.loc[idx, "bars_to_oracle_entry"].to_numpy(float)
    if phase == "ALL":
        return np.ones(idx.size, dtype=bool)
    if phase == "BEFORE_ENTRY":
        return be > 0
    if phase == "AT_ENTRY":
        return be == 0
    if phase == "IN_POSITION":
        return be < 0
    raise ValueError(phase)


# --------------------------------------------------------------------------- #
# Main orchestration
# --------------------------------------------------------------------------- #
def run_direction_models(symbol: str = "AG"):
    ds = load_dataset(symbol)

    split, cuts = make_chronological_splits(ds)
    kept, boundary_report = remove_boundary_opportunities(ds, split, cuts)

    # index sets (over kept eligible rows)
    k = np.flatnonzero(kept)
    train_idx = k[split[k] == 0]
    val_idx = k[split[k] == 1]
    test_idx = k[split[k] == 2]

    def _n_trades(idxs):
        if idxs.size == 0:
            return 0
        return int(
            len(
                np.unique(
                    ds.loc[idxs, "oracle_trade_id"].to_numpy(object)
                )
            )
        )

    split_report = {
        "train_rows": int(train_idx.size),
        "val_rows": int(val_idx.size),
        "test_rows": int(test_idx.size),
        "train_trades": _n_trades(train_idx),
        "val_trades": _n_trades(val_idx),
        "test_trades": _n_trades(test_idx),
    }

    y_test = (
        ds.loc[test_idx, "oracle_direction"].to_numpy(object) == "LONG"
    ).astype(np.uint8)

    # ---- baselines (predict on TEST) ----
    pred_always_long = np.ones(test_idx.size, dtype=np.uint8)
    pred_always_short = np.zeros(test_idx.size, dtype=np.uint8)
    # TRAIN majority
    y_train = (
        ds.loc[train_idx, "oracle_direction"].to_numpy(object) == "LONG"
    ).astype(np.uint8)
    majority = int(np.round(y_train.mean())) if train_idx.size else 0
    pred_majority = np.full(test_idx.size, majority, dtype=np.uint8)

    methods = {}

    # helper closure
    def eval_method(name, pred_dir, pred_long_proba):
        block = {"TEST": evaluate_subset(ds, test_idx, pred_dir, pred_long_proba)}
        for ph in ("BEFORE_ENTRY", "AT_ENTRY", "IN_POSITION"):
            sub = phase_mask(ds, test_idx, ph)
            block[ph] = evaluate_subset(
                ds, test_idx[sub], pred_dir[sub],
                None if pred_long_proba is None else pred_long_proba[sub],
            )
        return block

    methods["always_long"] = eval_method("always_long", pred_always_long, None)
    methods["always_short"] = eval_method("always_short", pred_always_short, None)
    methods["majority"] = eval_method("majority", pred_majority, None)

    # ---- models ----
    results = {}
    per_trade_ret = {}
    for mname, cols in (("dir_m0", DTP9), ("dir_m1", STRUCT33)):
        Xtr, ytr, wtr = prepare_xy(ds, train_idx, cols)
        wtr_norm = wtr / wtr.mean() if wtr.mean() > 0 else wtr
        Xv, yv, wv = prepare_xy(ds, val_idx, cols)
        model = fit_direction_model(Xtr, ytr, wtr_norm, Xv, yv, wv)

        pdir_test, plong_test = predict_direction(model, ds.loc[test_idx, list(cols)])
        methods[mname] = eval_method(mname, pdir_test, plong_test)

        # per-trade returns on TEST for M1-vs-M0 paired comparison
        yt = y_test
        eqt = ds.loc[test_idx, "entry_quality_atr"].to_numpy(float)
        wt = ds.loc[test_idx, "sample_weight_raw"].to_numpy(float)
        tids_t = ds.loc[test_idx, "oracle_trade_id"].to_numpy(object)
        pdr = pred_direction_return_atr(pdir_test, yt, eqt)
        tr, uniq = aggregate_per_trade(tids_t, wt, pdr)
        per_trade_ret[mname] = tr
        if mname == "dir_m0":
            test_uniq = uniq
            _, inv_t0 = np.unique(tids_t, return_inverse=True)
            test_counts = np.bincount(inv_t0)

        # persist model + predictions locally (not committed)
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        model.booster_.save_model(
            os.path.join(ARTIFACT_DIR, f"{mname}_{symbol}.txt")
        )
        results[mname] = (pdir_test, plong_test)

    # ---- M1 vs M0 paired (TEST trades) ----
    m0 = per_trade_ret["dir_m0"]
    m1 = per_trade_ret["dir_m1"]
    assert len(m0) == len(m1), "M0/M1 must share the same TEST trades"
    delta = m1 - m0
    dmean = float(delta.mean())
    if len(delta):
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        n = len(delta)
        bidx = rng.integers(0, n, size=(BOOTSTRAP_REPLICATES, n))
        bmean = delta[bidx].mean(axis=1)
        _lohi = np.quantile(bmean, [0.025, 0.975])
        dlo, dhi = float(_lohi[0]), float(_lohi[1])
    else:
        dlo = dhi = 0.0
    m1_minus_m0 = {
        "mean_delta": dmean,
        "ci_low": dlo,
        "ci_high": dhi,
        "n_trades": int(len(delta)),
    }

    # ---- persist trade_returns CSV (per-TEST-trade) ----
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    trade_df = pd.DataFrame(
        {
            "oracle_trade_id": test_uniq,
            "n_candidates": test_counts,
            "return_atr_m0": per_trade_ret["dir_m0"],
            "return_atr_m1": per_trade_ret["dir_m1"],
        }
    )
    trade_csv = os.path.join(EVIDENCE_DIR, "direction_model_AG_v1_trade_returns.csv")
    trade_df.to_csv(trade_csv, index=False)

    summary = {
        "task_id": TASK_ID,
        "base_sha": BASE_SHA,
        "symbol": symbol,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cuts": {
            "t1": str(pd.Timestamp(cuts[0]).isoformat()),
            "t2": str(pd.Timestamp(cuts[1]).isoformat()),
            "frac_train": FRAC_TRAIN,
            "frac_val": FRAC_VAL,
        },
        "boundary_removal": boundary_report,
        "splits": split_report,
        "models": methods,
        "m1_minus_m0": m1_minus_m0,
        "feature_schemas": {"dtp9": list(DTP9), "struct33": list(STRUCT33)},
    }

    summary_path = os.path.join(EVIDENCE_DIR, "direction_model_AG_v1_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    return {
        "summary": summary,
        "ds": ds,
        "split": split,
        "kept": kept,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "per_trade_ret": per_trade_ret,
        "paths": {"summary": summary_path, "trade_returns": trade_csv},
    }


if __name__ == "__main__":
    r = run_direction_models("AG")
    print(json.dumps(r["summary"], indent=2, default=str))
    print("paths:", r["paths"])
