"""train_direction_model_15sym_v1
================================

FUTURE-R4-M15-DIRECTION-MODEL-V1-15SYM-CONFIRMATION

Pooled 15-symbol Direction confirmation trainer. It is the pooled generalization
of FUTURE-R4-M15-DIRECTION-MODEL-V1-AG (``train_direction_model_ag_v1``), with the
SAME frozen methodology and the SAME frozen LightGBM params. No tuning.

Hard governance contracts (enforced structurally + by tests):
  1. NO environment rerun, NO candidate-gate rerun, NO Phase-1 dataset rebuild,
     NO Teacher DP rerun. This module only READS the frozen per-symbol Phase-1
     parquets. Forbidden tokens (checked by the test) never appear in this source.
  2. Fail-closed dataset identity check BEFORE any read of the data: dataset SHA,
     STRUCT33 schema hash, dataset builder provenance, Teacher artifact SHA and
     execution-frame SHA are all verified against the committed 15-symbol manifest.
     Any mismatch STOPS.
  3. One unified calendar shared by all 15 symbols:
        START = max over symbols of first eligible candidate_decision_time
        END   = min over symbols of last  eligible candidate_decision_time
        T1    = START + 60% * (END - START)
        T2    = START + 80% * (END - START)
     All symbols use the SAME absolute T1/T2 (no per-symbol quantiles).
  4. Boundary rule (frozen, reused verbatim): a retained Candidate must have
     decision / fill / oracle_entry / oracle_exit times in the SAME split; if ANY
     Candidate of an oracle_trade_id violates that, the ENTIRE trade is dropped
     from ALL splits. No additional bar purge.
  5. Only direction is trained. entry_quality_atr is used ONLY to translate a
     predicted direction into an economic return.

Two models, identical fixed params (imported from the AG module so they cannot
drift):
  DIR-M0 : DTP9     (15m/1h/4h trend only)
  DIR-M1 : STRUCT33 (trend + SR + Liquidity)
No symbol feature is used (the pooled model sees features only).

Three engineering fixes over the AG trainer:
  (a) dataset SHA (and all manifest identities) verified fail-closed;
  (b) majority baseline is OPPORTUNITY-weighted (per Oracle trade), not row-weighted;
  (c) the trade-level bootstrap is CHUNKED (bounded memory on the pooled trade set).

Primary metric: mean PredictedDirectionReturnATR per Oracle opportunity, with a
trade-level (NOT row-level) bootstrap 95% CI. Reported for: POOLED, POOLED_EX_AG,
and all 15 per-symbol scopes; broken down by ALL / BEFORE_ENTRY / AT_ENTRY /
IN_POSITION and TEACHER_LONG / TEACHER_SHORT; plus an M1-vs-M0 paired bootstrap.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.train_direction_model_ag_v1 import (
    BASE_PARAMS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DECISION_THRESHOLD,
    DTP9,
    FRAC_TRAIN,
    FRAC_VAL,
    STRUCT33,
    aggregate_per_trade,
    compute_class_metrics,
    fit_direction_model,
    phase_mask,
    pred_direction_return_atr,
    predict_direction,
    prepare_xy,
    remove_boundary_opportunities,
)

TASK_ID = "FUTURE-R4-M15-DIRECTION-MODEL-V1-15SYM-CONFIRMATION"
BASE_SHA = "48605666cd9e3cb0c2d49848df47bb7050a0dabc"

# Frozen 15-symbol universe (same order as the upstream materialization).
SYMBOLS = (
    "AG", "AU", "CU", "AL", "SN", "NI", "RB", "I", "SC", "RU",
    "MA", "TA", "M", "P", "CF",
)

MANIFEST_PATH = os.path.join("artifacts", "15sym_dataset_manifest.json")
DATASET_PARQUET = os.path.join(
    "artifacts", "struct33_dataset_v1", "{symbol}", "candidate_teacher_dataset.parquet"
)
TEACHER_PARQUET = os.path.join(
    "artifacts", "teacher_oracle_dp_m15_overnight_v1", "{symbol}", "oracle_trades.parquet"
)
EXEC_PARQUET = os.path.join(
    "artifacts", "candidate_gate_r4_m15_touch_nextbar_v1", "{symbol}_exec_frame.parquet"
)
EVIDENCE_DIR = os.path.join("research", "liquidity_oracle_atlas", "evidence")
ARTIFACT_DIR = os.path.join("artifacts", "direction_model_15sym_v1")

SUMMARY_JSON = os.path.join(EVIDENCE_DIR, "direction_model_15sym_v1_summary.json")
TRADE_CSV = os.path.join(EVIDENCE_DIR, "direction_model_15sym_v1_trade_returns.csv")

# Expected artifact-generation provenance per symbol (frozen at FIX2).
EXPECTED_DATASET_BUILDER_SHA = {"AG": "df868eb8790438ee55db3fc817bcf2650bc994e3"}
DEFAULT_DATASET_BUILDER_SHA = "c45a1efa8042d44cb36282d15f0ff7b7fea0d23e"

BOOTSTRAP_CHUNK = 500

_TEST_PHASES = ("ALL", "BEFORE_ENTRY", "AT_ENTRY", "IN_POSITION")
_TEST_TEACHER = ("TEACHER_LONG", "TEACHER_SHORT")


# --------------------------------------------------------------------------- #
# Fail-closed manifest verification                                             #
# --------------------------------------------------------------------------- #
def _sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        h = hashlib.sha256()
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def struct33_schema_hash() -> str:
    """Identical formula to the upstream builder: sha256('|'.join(STRUCT33))."""
    return hashlib.sha256("|".join(STRUCT33).encode()).hexdigest()


def verify_manifest(symbols=SYMBOLS, manifest_path: str | None = None) -> dict:
    """Fail-closed identity check of the 15 committed datasets.

    Verifies, per symbol, against ``artifacts/15sym_dataset_manifest.json``:
      * dataset parquet SHA256
      * STRUCT33 schema hash (recomputed from the frozen schema)
      * dataset builder provenance (contract id + expected generation SHA)
      * Teacher artifact SHA256
      * execution-frame SHA256
    Raises on ANY mismatch (no partial pass).
    """
    manifest_path = manifest_path or MANIFEST_PATH
    if not os.path.exists(manifest_path):
        raise RuntimeError(f"STOP_15SYM_MANIFEST_MISSING:{manifest_path}")
    man = {m["symbol"]: m for m in json.loads(open(manifest_path).read())}
    expected_schema = struct33_schema_hash()

    report: dict = {}
    problems: list = []
    for s in symbols:
        m = man.get(s)
        row: dict = {"symbol": s}
        report[s] = row
        if m is None:
            problems.append(f"{s}:manifest_absent")
            continue

        checks = (
            ("dataset", DATASET_PARQUET.format(symbol=s), "dataset_sha256"),
            ("teacher", TEACHER_PARQUET.format(symbol=s), "teacher_artifact_sha256"),
            ("exec_frame", EXEC_PARQUET.format(symbol=s), "execution_frame_sha256"),
        )
        for name, path, key in checks:
            if not os.path.exists(path):
                row[f"{name}_present"] = False
                problems.append(f"{s}:{name}_missing")
                continue
            actual = _sha256_file(path)
            exp = m.get(key)
            ok = exp is not None and actual == exp
            row[f"{name}_sha256_match"] = bool(ok)
            if not ok:
                problems.append(f"{s}:{name}_sha_mismatch")

        sh_ok = m.get("struct33_schema_hash") == expected_schema
        row["struct33_schema_hash_match"] = bool(sh_ok)
        if not sh_ok:
            problems.append(f"{s}:schema_hash_mismatch")

        exp_bsha = EXPECTED_DATASET_BUILDER_SHA.get(s, DEFAULT_DATASET_BUILDER_SHA)
        prov_ok = (
            m.get("dataset_builder_source_git_sha") == exp_bsha
            and bool(m.get("dataset_builder_contract_id"))
        )
        row["dataset_builder_source_git_sha"] = m.get("dataset_builder_source_git_sha")
        row["dataset_builder_contract_id"] = m.get("dataset_builder_contract_id")
        row["builder_provenance_ok"] = bool(prov_ok)
        if not prov_ok:
            problems.append(f"{s}:builder_provenance_mismatch")

        row["teacher_contract_id"] = m.get("teacher_contract_id")
        row["teacher_source_git_sha"] = m.get("teacher_source_git_sha")
        if not m.get("teacher_source_git_sha"):
            problems.append(f"{s}:teacher_source_missing")

    if problems:
        raise RuntimeError(
            "STOP_15SYM_MANIFEST_VERIFICATION_FAILED: " + " | ".join(problems)
        )
    return report


# --------------------------------------------------------------------------- #
# Pooled load + unified calendar                                                #
# --------------------------------------------------------------------------- #
def load_pooled(symbols=SYMBOLS) -> pd.DataFrame:
    frames = []
    for s in symbols:
        path = DATASET_PARQUET.format(symbol=s)
        if not os.path.exists(path):
            raise FileNotFoundError(f"STOP_15SYM_DATASET_MISSING:{path}")
        ds = pd.read_parquet(path)
        ds["symbol"] = s
        frames.append(ds)
    return pd.concat(frames, ignore_index=True)


def common_calendar(ds: pd.DataFrame, symbols=SYMBOLS,
                    frac_train: float = FRAC_TRAIN, frac_val: float = FRAC_VAL) -> dict:
    """One unified calendar shared by all symbols (see module docstring)."""
    el = ds["label_eligible"].to_numpy(bool)
    sym = ds["symbol"].to_numpy(object)
    dt = pd.to_datetime(ds["candidate_decision_time"]).to_numpy(dtype="datetime64[ns]")

    firsts, lasts = [], []
    for s in symbols:
        m = el & (sym == s)
        if not m.any():
            raise RuntimeError(f"STOP_15SYM_NO_ELIGIBLE:{s}")
        firsts.append(dt[m].min())
        lasts.append(dt[m].max())
    start = max(firsts)
    end = min(lasts)

    start_ns = int(np.datetime64(start, "ns").astype("int64"))
    end_ns = int(np.datetime64(end, "ns").astype("int64"))
    span = end_ns - start_ns
    t1_ns = start_ns + int(round(span * frac_train))
    t2_ns = start_ns + int(round(span * (frac_train + frac_val)))
    cuts = np.array([t1_ns, t2_ns], dtype="datetime64[ns]")
    return {
        "start": pd.Timestamp(start),
        "end": pd.Timestamp(end),
        "t1": pd.Timestamp(t1_ns),
        "t2": pd.Timestamp(t2_ns),
        "start_ns": start_ns,
        "end_ns": end_ns,
        "cuts": cuts,
        "span_days": round(span / 1e9 / 86400.0, 3),
        "frac_train": frac_train,
        "frac_val": frac_val,
    }


# --------------------------------------------------------------------------- #
# Chunked trade-level bootstrap + opportunity-weighted majority                 #
# --------------------------------------------------------------------------- #
def bootstrap_trade_returns_chunked(
    trade_return: np.ndarray,
    B: int = BOOTSTRAP_REPLICATES,
    chunk: int = BOOTSTRAP_CHUNK,
    seed: int = BOOTSTRAP_SEED,
):
    """Trade-clustered bootstrap with bounded memory (chunked replicates)."""
    n = len(trade_return)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty(B, dtype=float)
    done = 0
    while done < B:
        m = min(chunk, B - done)
        idx = rng.integers(0, n, size=(m, n))
        means[done:done + m] = trade_return[idx].mean(axis=1)
        done += m
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(trade_return.mean()), float(lo), float(hi)


def opportunity_weighted_majority(y: np.ndarray, w: np.ndarray) -> int:
    """Majority class weighting each Oracle opportunity equally.

    Because each retained Oracle trade's ``sample_weight_raw`` sums to 1, the
    weight-averaged label equals (#LONG trades) / (#trades); rounding gives the
    opportunity-weighted majority (NOT the row-weighted one).
    """
    if len(y) == 0:
        return 0
    return int(round(float(np.average(y, weights=w))))


# --------------------------------------------------------------------------- #
# Evaluation                                                                    #
# --------------------------------------------------------------------------- #
def evaluate_subset(ds: pd.DataFrame, idx: np.ndarray, pred_dir: np.ndarray,
                    pred_long_proba: np.ndarray | None):
    if idx.size == 0:
        return {
            "n_rows": 0, "n_trades": 0, "accuracy": None, "balanced_accuracy": None,
            "roc_auc": None, "return_atr": None, "ci_low": None, "ci_high": None,
        }
    y = (ds.loc[idx, "oracle_direction"].to_numpy(object) == "LONG").astype(np.uint8)
    w = ds.loc[idx, "sample_weight_raw"].to_numpy(float)
    eq = ds.loc[idx, "entry_quality_atr"].to_numpy(float)
    tids = ds.loc[idx, "oracle_trade_id"].to_numpy(object)

    pdr = pred_direction_return_atr(pred_dir, y, eq)
    trade_ret, uniq = aggregate_per_trade(tids, w, pdr)
    mean_ret, lo, hi = bootstrap_trade_returns_chunked(trade_ret)

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


def evaluate_block(ds, idx, pred_dir, pred_long_proba):
    block = {}
    for ph in _TEST_PHASES:
        sub = phase_mask(ds, idx, ph)
        block[ph] = evaluate_subset(
            ds, idx[sub], pred_dir[sub],
            None if pred_long_proba is None else pred_long_proba[sub],
        )
    y = ds.loc[idx, "oracle_direction"].to_numpy(object) == "LONG"
    block["TEACHER_LONG"] = evaluate_subset(
        ds, idx[y], pred_dir[y], None if pred_long_proba is None else pred_long_proba[y]
    )
    block["TEACHER_SHORT"] = evaluate_subset(
        ds, idx[~y], pred_dir[~y], None if pred_long_proba is None else pred_long_proba[~y]
    )
    return block


def paired_m1_minus_m0(ds, idx, pred_dir_m0, pred_dir_m1):
    if idx.size == 0:
        return {"mean_delta": None, "ci_low": None, "ci_high": None, "n_trades": 0}
    y = (ds.loc[idx, "oracle_direction"].to_numpy(object) == "LONG").astype(np.uint8)
    w = ds.loc[idx, "sample_weight_raw"].to_numpy(float)
    eq = ds.loc[idx, "entry_quality_atr"].to_numpy(float)
    tids = ds.loc[idx, "oracle_trade_id"].to_numpy(object)
    tr0, _ = aggregate_per_trade(tids, w, pred_direction_return_atr(pred_dir_m0, y, eq))
    tr1, _ = aggregate_per_trade(tids, w, pred_direction_return_atr(pred_dir_m1, y, eq))
    delta = tr1 - tr0
    mean, lo, hi = bootstrap_trade_returns_chunked(delta)
    return {"mean_delta": mean, "ci_low": lo, "ci_high": hi, "n_trades": int(len(delta))}


# --------------------------------------------------------------------------- #
# Main orchestration                                                            #
# --------------------------------------------------------------------------- #
def run_direction_models_15sym(symbols=SYMBOLS, frac_train: float = FRAC_TRAIN,
                               frac_val: float = FRAC_VAL, save: bool = True):
    manifest_report = verify_manifest(symbols)
    ds = load_pooled(symbols)
    cal = common_calendar(ds, symbols, frac_train, frac_val)
    cuts = cal["cuts"]

    dt = pd.to_datetime(ds["candidate_decision_time"]).to_numpy(dtype="datetime64[ns]")
    split = np.searchsorted(cuts, dt, side="right")
    kept, boundary_report = remove_boundary_opportunities(ds, split, cuts)

    k = np.flatnonzero(kept)
    train_idx = k[split[k] == 0]
    val_idx = k[split[k] == 1]
    test_idx = k[split[k] == 2]

    # window membership report (all rows are kept; no extra purge)
    dt_ns = dt.astype("int64")
    in_win = (dt_ns >= cal["start_ns"]) & (dt_ns <= cal["end_ns"])

    # ---- fit pooled models (fixed params; no tuning) ----
    preds = {}
    for mname, cols in (("dir_m0", DTP9), ("dir_m1", STRUCT33)):
        Xtr, ytr, wtr = prepare_xy(ds, train_idx, cols)
        wtr_norm = wtr / wtr.mean() if wtr.mean() > 0 else wtr
        Xv, yv, wv = prepare_xy(ds, val_idx, cols)
        model = fit_direction_model(Xtr, ytr, wtr_norm, Xv, yv, wv)
        pdir, plong = predict_direction(model, ds.loc[test_idx, list(cols)])
        preds[mname] = (pdir, plong)
        if save:
            os.makedirs(ARTIFACT_DIR, exist_ok=True)
            model.booster_.save_model(os.path.join(ARTIFACT_DIR, f"{mname}_15sym.txt"))

    m0_dir, m0_p = preds["dir_m0"]
    m1_dir, m1_p = preds["dir_m1"]

    # ---- baselines (predict on TEST rows) ----
    y_train = (ds.loc[train_idx, "oracle_direction"].to_numpy(object) == "LONG").astype(np.uint8)
    w_train = ds.loc[train_idx, "sample_weight_raw"].to_numpy(float)
    majority = opportunity_weighted_majority(y_train, w_train)
    n_test = test_idx.size
    baseline_preds = {
        "always_long": np.ones(n_test, dtype=np.uint8),
        "always_short": np.zeros(n_test, dtype=np.uint8),
        "majority": np.full(n_test, majority, dtype=np.uint8),
    }
    model_preds = {"dir_m0": (m0_dir, m0_p), "dir_m1": (m1_dir, m1_p)}

    # ---- scopes over the pooled TEST rows ----
    sym_test = ds.loc[test_idx, "symbol"].to_numpy(object)
    scope_masks = {"POOLED": np.ones(n_test, dtype=bool),
                   "POOLED_EX_AG": sym_test != "AG"}
    for s in symbols:
        scope_masks[f"SYM_{s}"] = sym_test == s

    def scope_block(mask):
        sub = test_idx[mask]
        models_out = {}
        for bname, bpred in baseline_preds.items():
            models_out[bname] = evaluate_block(ds, sub, bpred[mask], None)
        for mname, (pdir, plong) in model_preds.items():
            models_out[mname] = evaluate_block(ds, sub, pdir[mask], plong[mask])
        return {
            "n_rows": int(mask.sum()),
            "n_trades": int(pd.unique(ds.loc[sub, "oracle_trade_id"]).size) if sub.size else 0,
            "symbols_present": sorted(set(sym_test[mask].tolist())),
            "models": models_out,
            "m1_minus_m0": paired_m1_minus_m0(ds, sub, m0_dir[mask], m1_dir[mask]),
        }

    scopes = {name: scope_block(mask) for name, mask in scope_masks.items()}

    # ---- split report (pooled + per-symbol) ----
    def _counts(idxs):
        if idxs.size == 0:
            return {"rows": 0, "trades": 0}
        return {"rows": int(idxs.size),
                "trades": int(pd.unique(ds.loc[idxs, "oracle_trade_id"]).size)}

    sym_all = ds["symbol"].to_numpy(object)
    split_report = {
        "pooled": {"train": _counts(train_idx), "val": _counts(val_idx), "test": _counts(test_idx)},
        "per_symbol": {},
    }
    for s in symbols:
        ms = sym_all == s
        split_report["per_symbol"][s] = {
            "train": _counts(train_idx[ms[train_idx]]),
            "val": _counts(val_idx[ms[val_idx]]),
            "test": _counts(test_idx[ms[test_idx]]),
        }

    # ---- per-trade TEST returns CSV (pooled) ----
    y_test = (ds.loc[test_idx, "oracle_direction"].to_numpy(object) == "LONG").astype(np.uint8)
    eq_test = ds.loc[test_idx, "entry_quality_atr"].to_numpy(float)
    w_test = ds.loc[test_idx, "sample_weight_raw"].to_numpy(float)
    tids_test = ds.loc[test_idx, "oracle_trade_id"].to_numpy(object)
    tr0, uniq = aggregate_per_trade(tids_test, w_test, pred_direction_return_atr(m0_dir, y_test, eq_test))
    tr1, _ = aggregate_per_trade(tids_test, w_test, pred_direction_return_atr(m1_dir, y_test, eq_test))
    _, inv = np.unique(tids_test, return_inverse=True)
    counts = np.bincount(inv, minlength=len(uniq))
    tid_to_sym = (ds.loc[test_idx, ["oracle_trade_id", "symbol"]]
                  .drop_duplicates("oracle_trade_id")
                  .set_index("oracle_trade_id")["symbol"])
    trade_df = pd.DataFrame({
        "symbol": [tid_to_sym.get(t, t.split("_")[0]) for t in uniq],
        "oracle_trade_id": uniq,
        "n_candidates": counts,
        "return_atr_m0": tr0,
        "return_atr_m1": tr1,
    })
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    trade_df.to_csv(TRADE_CSV, index=False)

    summary = {
        "task_id": TASK_ID,
        "base_sha": BASE_SHA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols": list(symbols),
        "manifest_verification": manifest_report,
        "calendar": {
            "start": cal["start"].isoformat(),
            "end": cal["end"].isoformat(),
            "t1": cal["t1"].isoformat(),
            "t2": cal["t2"].isoformat(),
            "frac_train": frac_train,
            "frac_val": frac_val,
            "span_days": cal["span_days"],
            "rows_inside_window": int(in_win.sum()),
            "rows_outside_window": int((~in_win).sum()),
            "note": "shared absolute T1/T2 for all symbols; all rows kept (no extra purge)",
        },
        "boundary_removal": boundary_report,
        "splits": split_report,
        "params": {
            "model": "LightGBM binary classifier, fixed params (no hyperparameter search)",
            "decision_threshold": DECISION_THRESHOLD,
            "base_params": BASE_PARAMS,
            "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                          "chunk": BOOTSTRAP_CHUNK, "cluster": "oracle opportunity (trade)"},
            "majority_baseline": "opportunity-weighted (train)",
            "majority_class": int(majority),
            "symbol_feature_used": False,
        },
        "feature_schemas": {"dtp9": list(DTP9), "struct33": list(STRUCT33)},
        "scopes": scopes,
    }

    with open(SUMMARY_JSON, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    return {
        "summary": summary,
        "ds": ds,
        "split": split,
        "kept": kept,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "cuts": cuts,
        "paths": {"summary": SUMMARY_JSON, "trade_returns": TRADE_CSV},
    }


if __name__ == "__main__":
    r = run_direction_models_15sym()
    s = r["summary"]
    print(json.dumps({
        "task_id": s["task_id"],
        "calendar": s["calendar"],
        "splits_pooled": s["splits"]["pooled"],
        "POOLED_dir_m1_TEST": s["scopes"]["POOLED"]["models"]["dir_m1"]["ALL"],
        "POOLED_dir_m0_TEST": s["scopes"]["POOLED"]["models"]["dir_m0"]["ALL"],
        "POOLED_m1_minus_m0": s["scopes"]["POOLED"]["m1_minus_m0"],
        "POOLED_EX_AG_m1_minus_m0": s["scopes"]["POOLED_EX_AG"]["m1_minus_m0"],
    }, indent=2, default=str))
    print("paths:", r["paths"])
