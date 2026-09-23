"""train_direction_model_15sym_v1
================================

FUTURE-R4-M15-DIRECTION-MODEL-V1-15SYM-CONFIRMATION  (FIX1)

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
  3. Common STUDY WINDOW (FIX1). The 15-symbol common study universe is
        START = max over symbols of first eligible candidate_decision_time
        END   = min over symbols of last  eligible candidate_decision_time
     START/END are STUDY BOUNDARIES, not a bar-count purge. An Oracle opportunity
     participates only if ALL of its retained Candidate / Teacher timestamps
     (candidate_decision_time, candidate_fill_time, oracle_entry_fill_time,
     oracle_exit_fill_time) lie inside [START, END]; otherwise the ENTIRE
     opportunity is dropped. This is study-window eligibility, which is distinct
     from the (still absent) T1/T2 bar-count purge.
  4. Canonical 15m clock (FIX1). The split cuts are derived on the common span and
     then SNAPPED to the canonical 15-minute decision grid:
        raw_T1 = START + 60% * (END - START)
        raw_T2 = START + 80% * (END - START)
        T1 = ceil(raw_T1, 15min);  T2 = ceil(raw_T2, 15min)
     All symbols share the SAME absolute (snapped) T1/T2.
  5. Boundary rule (frozen, reused verbatim from the AG module): after the common
     study window is enforced, a retained Candidate must have decision / fill /
     oracle_entry / oracle_exit times in the SAME split; if ANY Candidate of an
     oracle_trade_id violates that, the ENTIRE trade is dropped. No additional
     50/100/200-bar purge.
  6. Only direction is trained. entry_quality_atr is used ONLY to translate a
     predicted direction into an economic return.

Two models, identical fixed params (imported from the AG module so they cannot
drift):
  DIR-M0 : DTP9     (15m/1h/4h trend only)
  DIR-M1 : STRUCT33 (trend + SR + Liquidity)
No symbol feature is used (the pooled model sees features only).

Three engineering fixes over the AG trainer:
  (a) dataset SHA (and all manifest identities) verified fail-closed;
  (b) majority baseline is OPPORTUNITY-weighted (per Oracle trade), not row-weighted;
  (c) the trade-level bootstrap is CHUNKED (bounded memory) and provably identical
      to the unchunked reference for the same seed/B.

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
SNAP_RULE = "CEIL_15MIN"

_WINDOW_TIME_COLS = (
    "candidate_decision_time",
    "candidate_fill_time",
    "oracle_entry_fill_time",
    "oracle_exit_fill_time",
)

# Frozen pre-FIX1 (ea49c84) result, used only to report old->new deltas.
PRE_FIX1_SHA = "ea49c84161c5a859d893ae3b6dc9dcc75de1c0f9"
PRE_FIX1_BASELINE = {
    "splits_pooled": {
        "train_rows": 43235, "train_trades": 2396,
        "val_rows": 12046, "val_trades": 643,
        "test_rows": 14722, "test_trades": 677,
    },
    "boundary_trades_dropped_total": 26,
    "pooled_m0": {"return_atr": 0.6427545328750323,
                  "ci_low": 0.3513963604033949, "ci_high": 0.9144324641189835},
    "pooled_m1": {"return_atr": 0.6522414528446739,
                  "ci_low": 0.3807024587029232, "ci_high": 0.908452300455112},
    "pooled_m1_minus_m0": {"mean_delta": 0.009486919969641716,
                           "ci_low": -0.12792035947455227,
                           "ci_high": 0.15073942739541704, "n_trades": 677},
    "pooled_ex_ag_m0": {"return_atr": 0.5918661651099241,
                        "ci_low": 0.2920386862364857, "ci_high": 0.8945227991358379},
    "pooled_ex_ag_m1": {"return_atr": 0.6145575549815256,
                        "ci_low": 0.33023390067685476, "ci_high": 0.898653026571771},
    "pooled_ex_ag_m1_minus_m0": {"mean_delta": 0.022691389871601644,
                                 "ci_low": -0.11598376042970478,
                                 "ci_high": 0.1677519468719708, "n_trades": 600},
}


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
# Pooled load + common calendar (study window + 15m-snapped cuts)               #
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
    """One unified calendar shared by all symbols (see module docstring).

    raw_T1/raw_T2 are the exact 60%/80% fractions of the common span; T1/T2 are
    snapped to the canonical 15-minute decision grid via ceiling.
    """
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

    raw_t1 = pd.Timestamp(start_ns + int(round(span * frac_train)))
    raw_t2 = pd.Timestamp(start_ns + int(round(span * (frac_train + frac_val))))
    t1 = raw_t1.ceil("15min")
    t2 = raw_t2.ceil("15min")

    for t in (t1, t2):
        if not (t.minute % 15 == 0 and t.second == 0
                and t.microsecond == 0 and t.nanosecond == 0):
            raise RuntimeError(f"STOP_15SYM_CUT_NOT_ON_15M_GRID:{t}")

    cuts = np.array([t1.value, t2.value], dtype="datetime64[ns]")
    return {
        "start": pd.Timestamp(start),
        "end": pd.Timestamp(end),
        "raw_t1": raw_t1,
        "raw_t2": raw_t2,
        "t1": t1,
        "t2": t2,
        "cuts": cuts,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "span_days": round(span / 1e9 / 86400.0, 3),
        "frac_train": frac_train,
        "frac_val": frac_val,
        "snap_rule": SNAP_RULE,
    }


def common_window_eligibility(ds: pd.DataFrame, start_ns: int, end_ns: int):
    """Whole-opportunity eligibility against the common study window [START, END].

    A row is window-eligible iff ALL of its four timestamps lie inside the window.
    If ANY eligible row of an oracle_trade_id is not window-eligible, the ENTIRE
    opportunity is dropped (no partial trades). Returns (win_ok_mask_over_all_rows,
    report).
    """
    el = ds["label_eligible"].to_numpy(bool)
    e = np.flatnonzero(el)

    times = {
        c: pd.to_datetime(ds[c]).to_numpy(dtype="datetime64[ns]").astype("int64")[e]
        for c in _WINDOW_TIME_COLS
    }
    row_in = np.ones(e.size, dtype=bool)
    below = np.zeros(e.size, dtype=bool)
    above = np.zeros(e.size, dtype=bool)
    for c in _WINDOW_TIME_COLS:
        t = times[c]
        row_in &= (t >= start_ns) & (t <= end_ns)
        below |= t < start_ns
        above |= t > end_ns

    tids = ds["oracle_trade_id"].to_numpy(object)[e]
    uniq, inv = np.unique(tids, return_inverse=True)

    trade_bad = np.zeros(len(uniq), dtype=bool)
    np.logical_or.at(trade_bad, inv, ~row_in)
    trade_below = np.zeros(len(uniq), dtype=bool)
    np.logical_or.at(trade_below, inv, below)
    trade_above = np.zeros(len(uniq), dtype=bool)
    np.logical_or.at(trade_above, inv, above)

    keep = row_in & ~trade_bad[inv]
    win_ok = np.zeros(len(ds), dtype=bool)
    win_ok[e] = keep

    report = {
        "start": pd.Timestamp(start_ns).isoformat(),
        "end": pd.Timestamp(end_ns).isoformat(),
        "eligible_rows_total": int(e.size),
        "rows_outside_common_window": int((~row_in).sum()),
        "opportunities_total": int(len(uniq)),
        "opportunities_dropped_common_start": int((trade_bad & trade_below).sum()),
        "opportunities_dropped_common_end": int((trade_bad & trade_above).sum()),
        "opportunities_dropped_common_window": int(trade_bad.sum()),
        "opportunities_after_window": int((~trade_bad).sum()),
        "eligible_rows_after_window": int(keep.sum()),
    }
    return win_ok, report


# --------------------------------------------------------------------------- #
# Chunked trade-level bootstrap (+ unchunked reference) + weighted majority     #
# --------------------------------------------------------------------------- #
def bootstrap_trade_returns_chunked(
    trade_return: np.ndarray,
    B: int = BOOTSTRAP_REPLICATES,
    chunk: int = BOOTSTRAP_CHUNK,
    seed: int = BOOTSTRAP_SEED,
):
    """Trade-clustered bootstrap with bounded memory (chunked replicates).

    numpy's Generator.integers is chunk-invariant (the stream advances
    element-wise), so for the same seed/B this is EXACTLY the unchunked reference
    computed by ``bootstrap_reference``.
    """
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


def bootstrap_reference(
    trade_return: np.ndarray,
    B: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
):
    """Unchunked reference (single draw). Same result as the chunked version."""
    n = len(trade_return)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boot = trade_return[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
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
    for ph in ("ALL", "BEFORE_ENTRY", "AT_ENTRY", "IN_POSITION"):
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


def _delta(old, new):
    if old is None or new is None:
        return {"old": old, "new": new, "delta": None}
    return {"old": old, "new": new, "delta": new - old}


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

    # (1) common study window (whole-opportunity eligibility)
    win_ok, window_report = common_window_eligibility(ds, cal["start_ns"], cal["end_ns"])
    # (2) T1/T2 whole-opportunity boundary removal among window-eligible rows
    #     (frozen rule reused verbatim; fed the window-eligible label mask).
    kept, boundary_report = remove_boundary_opportunities(
        ds.assign(label_eligible=win_ok), split, cuts
    )

    k = np.flatnonzero(kept)
    train_idx = k[split[k] == 0]
    val_idx = k[split[k] == 1]
    test_idx = k[split[k] == 2]

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

    # ---- pre-FIX1 -> post-FIX1 deltas ----
    new_pool = split_report["pooled"]
    b = PRE_FIX1_BASELINE
    deltas = {
        "pre_fix1_sha": PRE_FIX1_SHA,
        "splits_pooled": {
            "train_rows": _delta(b["splits_pooled"]["train_rows"], new_pool["train"]["rows"]),
            "train_trades": _delta(b["splits_pooled"]["train_trades"], new_pool["train"]["trades"]),
            "val_rows": _delta(b["splits_pooled"]["val_rows"], new_pool["val"]["rows"]),
            "val_trades": _delta(b["splits_pooled"]["val_trades"], new_pool["val"]["trades"]),
            "test_rows": _delta(b["splits_pooled"]["test_rows"], new_pool["test"]["rows"]),
            "test_trades": _delta(b["splits_pooled"]["test_trades"], new_pool["test"]["trades"]),
        },
        "boundary_trades_dropped_total": _delta(
            b["boundary_trades_dropped_total"], boundary_report["trades_dropped_total"]),
        "pooled_m0_return_atr": _delta(
            b["pooled_m0"]["return_atr"], scopes["POOLED"]["models"]["dir_m0"]["ALL"]["return_atr"]),
        "pooled_m1_return_atr": _delta(
            b["pooled_m1"]["return_atr"], scopes["POOLED"]["models"]["dir_m1"]["ALL"]["return_atr"]),
        "pooled_m1_minus_m0": {
            "old": b["pooled_m1_minus_m0"], "new": scopes["POOLED"]["m1_minus_m0"]},
        "pooled_ex_ag_m0_return_atr": _delta(
            b["pooled_ex_ag_m0"]["return_atr"],
            scopes["POOLED_EX_AG"]["models"]["dir_m0"]["ALL"]["return_atr"]),
        "pooled_ex_ag_m1_return_atr": _delta(
            b["pooled_ex_ag_m1"]["return_atr"],
            scopes["POOLED_EX_AG"]["models"]["dir_m1"]["ALL"]["return_atr"]),
        "pooled_ex_ag_m1_minus_m0": {
            "old": b["pooled_ex_ag_m1_minus_m0"],
            "new": scopes["POOLED_EX_AG"]["m1_minus_m0"]},
    }

    summary = {
        "task_id": TASK_ID,
        "base_sha": BASE_SHA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols": list(symbols),
        "manifest_verification": manifest_report,
        "calendar": {
            "start": cal["start"].isoformat(),
            "end": cal["end"].isoformat(),
            "raw_t1": cal["raw_t1"].isoformat(),
            "raw_t2": cal["raw_t2"].isoformat(),
            "t1": cal["t1"].isoformat(),
            "t2": cal["t2"].isoformat(),
            "frac_train": frac_train,
            "frac_val": frac_val,
            "span_days": cal["span_days"],
            "snap_rule": cal["snap_rule"],
            "note": ("T1/T2 = ceil(raw, 15min) on the canonical 15m decision grid; "
                     "shared by all symbols. START/END are study boundaries."),
        },
        "common_window": window_report,
        "boundary_removal": boundary_report,
        "splits": split_report,
        "params": {
            "model": "LightGBM binary classifier, fixed params (no hyperparameter search)",
            "decision_threshold": DECISION_THRESHOLD,
            "base_params": BASE_PARAMS,
            "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                          "chunk": BOOTSTRAP_CHUNK, "cluster": "oracle opportunity (trade)",
                          "reference": "unchunked single draw (provably identical)"},
            "majority_baseline": "opportunity-weighted (train)",
            "majority_class": int(majority),
            "symbol_feature_used": False,
            "snap_rule": cal["snap_rule"],
        },
        "feature_schemas": {"dtp9": list(DTP9), "struct33": list(STRUCT33)},
        "pre_fix1_deltas": deltas,
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
        "common_window": s["common_window"],
        "boundary_removal": s["boundary_removal"],
        "splits_pooled": s["splits"]["pooled"],
        "POOLED_dir_m0_TEST": s["scopes"]["POOLED"]["models"]["dir_m0"]["ALL"],
        "POOLED_dir_m1_TEST": s["scopes"]["POOLED"]["models"]["dir_m1"]["ALL"],
        "POOLED_m1_minus_m0": s["scopes"]["POOLED"]["m1_minus_m0"],
        "POOLED_EX_AG_m1_minus_m0": s["scopes"]["POOLED_EX_AG"]["m1_minus_m0"],
        "pre_fix1_deltas": s["pre_fix1_deltas"],
    }, indent=2, default=str))
    print("paths:", r["paths"])
