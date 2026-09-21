"""
experiment_entry_value_tree_core108_t15_v1
==========================================
FUTURE-ENTRY-VALUE-TREE-CORE108-V1-T1.5

Small end-to-end integration / model smoke (NOT the formal T2 experiment).

Chain under test
----------------
    CORE108 production kernel
    -> R2 Q labels
    -> multi-symbol dataset (AG + RB)
    -> time split + purge
    -> LightGBM Long / Short regressions
    -> constant-baseline comparison
    -> OOS metrics
    -> feature importance (gain + grouped permutation)
    -> small auditable artifacts

T1.5 deliberately does NOT:
  * change features or labels;
  * tune hyperparameters;
  * optimize trading thresholds;
  * backtest a strategy;
  * add exit / reversal logic;
  * run the 15-symbol T2;
  * produce any research verdict.

T1.5 only proves the chain runs correctly and surfaces the first factual
look at whether this modeling direction has signal at all.

Key frozen decisions
--------------------
* Candidate definition is the kernel's frozen ``is_candidate``
  (proximity_any AND training_eligible AND finite Q_F1_L/F/S AND finite
  positive ATR_5m AND CORE108 row joined by (symbol, decision_time)).
* Targets:  Y_L = (Q(F1,L) - Q(F1,F)) / ATR_5m,t ,  Y_S = (Q(F1,S) - Q(F1,F)) / ATR_5m,t
  No clipping / winsorization / transformation / rescaling.
* Raw episode weight w = 1/N_e over the composite key (symbol, proximity_episode_id).
  Per-symbol ``sample_weight_norm`` from join_with_r2() is NEVER concatenated and
  used directly; normalization is RECOMPUTED after concatenation, separately per
  Train and Validation.
* Split: 60 / 20 / 20 over globally sorted unique trading_day; no row
  randomization; no episode may span two splits; label-availability purge.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# LightGBM MUST be imported EARLY (before numpy/pandas/scipy get heavy use).
# On macOS a late dlopen of lib_lightgbm.so -- after other OpenMP-backed
# libraries are already active -- segfaults (SIGSEGV) inside fit(). Importing it
# first loads its bundled OpenMP runtime cleanly. This is an import-order fix
# only; the frozen BASE_PARAMS are untouched.
try:
    import lightgbm as lgb  # noqa: E402
except Exception as _exc:  # pragma: no cover - environment gate
    raise SystemExit(f"STOP_LIGHTGBM_UNAVAILABLE: {_exc}") from _exc

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from research.liquidity_oracle_atlas.experiment_entry_value_tree_core108_v1 import (
    DEFAULT_ARTIFACT_ROOT,
    Core108Counters,
    build_base_prefix,
    core108_columns,
    join_with_r2,
    run_streaming,
)

# --------------------------------------------------------------------------- #
# Frozen T1.5 constants
# --------------------------------------------------------------------------- #
TASK_ID = "FUTURE-ENTRY-VALUE-TREE-CORE108-V1-T1.5"
BASE_SHA = "641b5c435763dfebef40d36b44fa01f865af1a72"

T15_SYMBOLS: Tuple[str, ...] = ("AG", "RB")
T15_PREFIX_BARS = 10000
SEED = 20260921
REQUIRED_LGB_VERSION = "4.7.0"
EARLY_STOPPING_ROUNDS = 100
PERMUTATION_REPEATS = 5

SPLIT_FRACS = (0.60, 0.20, 0.20)  # train / validation / test over unique trading days
SPLIT_NAMES = ("train", "validation", "test")

TF_ORDER: Tuple[str, ...] = ("m5", "m15", "h1", "h4")
ROLES: Tuple[str, ...] = ("SUPPORT", "RESISTANCE", "BUYSIDE_LIQUIDITY", "SELLSIDE_LIQUIDITY")

# Fixed phase vocabulary (never learned from Test).
PHASE_VOCAB: Tuple[str, ...] = (
    "NO_STRUCTURE",
    "OUTSIDE",
    "APPROACH",
    "REJECTED",
    "ZONE",
    "BREAK",
    "BREAK_EXTENSION",
    "BREAK_RETURNING",
    "BREAK_HOLD",
    "PARTIAL_RECLAIM",
    "FULL_RECLAIM",
    "REBREAK",
)

BASE_PARAMS: Dict[str, Any] = dict(
    objective="regression",
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=100,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_estimators=2000,
    random_state=20260921,
    n_jobs=-1,
)

# Anything carrying label / oracle / audit / time semantics must never enter X.
FORBIDDEN_MODEL_TOKENS: Tuple[str, ...] = (
    "Q_F1", "Y_L", "Y_S", "label_available", "outcome_ret", "mfe", "mae",
    "next_bar_direction", "edge_F1", "amb_F1", "proximity_episode_id",
    "proximity_bits", "proximity_any", "training_eligible", "is_candidate",
    "decision_time", "decision_bar_index", "atr5m", "sample_weight",
    "trading_day", "split", "global_episode", "symbol", "w_norm", "w_raw",
)

DEFAULT_OUT_DIR = Path("artifacts/intraday_entry_value_tree_core108_v1")


# --------------------------------------------------------------------------- #
# Environment gate
# --------------------------------------------------------------------------- #
def check_lightgbm() -> str:
    """Require lightgbm==4.7.0 exactly; STOP otherwise."""
    ver = str(getattr(lgb, "__version__", ""))
    if ver != REQUIRED_LGB_VERSION:
        raise SystemExit(
            f"STOP_LIGHTGBM_VERSION: required {REQUIRED_LGB_VERSION}, found {ver}"
        )
    return ver


def _git_code_sha() -> Dict[str, Any]:
    """Best-effort code identity for the evidence packet."""
    out = {"sha": None, "dirty": None, "ok": False}
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=10,
        )
        if sha.returncode == 0:
            out["sha"] = sha.stdout.strip()
            out["dirty"] = bool(dirty.stdout.strip())
            out["ok"] = True
    except Exception:  # pragma: no cover - non-git environment
        pass
    return out


def _peak_memory_mb() -> Optional[float]:
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports kilobytes.
        return float(rss / (1024 * 1024)) if sys.platform == "darwin" else float(rss / 1024)
    except Exception:  # pragma: no cover
        return None


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy / pandas scalars into JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    return str(obj)


# --------------------------------------------------------------------------- #
# Step 1 — per-symbol CORE108 build (one pass, true prefix)
# --------------------------------------------------------------------------- #
def build_symbol_frame(
    symbol: str,
    prefix_bars: int = T15_PREFIX_BARS,
    artifact_root: Any = DEFAULT_ARTIFACT_ROOT,
) -> Dict[str, Any]:
    """Build CORE108 for ONE symbol on a TRUE prefix (bar 0..prefix_bars-1).

    Returns candidate rows only, with ``trading_day`` attached from the canonical
    raw column (mapped through ``decision_bar_index``, no extra raw reload).
    """
    counters = Core108Counters()
    t0 = time.perf_counter()
    info = build_base_prefix(symbol, counters, max_bars=prefix_bars)
    feature_df = run_streaming(info, counters, symbol, max_bars=prefix_bars)
    jr = join_with_r2(feature_df, symbol, counters, artifact_root=artifact_root)
    seconds = time.perf_counter() - t0

    df = jr["df"]
    integrity = dict(jr["integrity"])
    cand = df[df["is_candidate"]].copy().reset_index(drop=True)

    base_td = pd.to_datetime(pd.Series(info["base"]["trading_day"])).reset_index(drop=True)
    cand = attach_trading_day(cand, base_td)
    cand["global_episode"] = (
        cand["symbol"].astype(str) + "|" + cand["proximity_episode_id"].astype(str)
    )
    return {
        "symbol": symbol,
        "cand": cand,
        "integrity": integrity,
        "seconds": float(seconds),
        "n_prefix_bars": int(len(info["base"])),
        "counters": counters,
    }


def attach_trading_day(cand: pd.DataFrame, base_td: pd.Series) -> pd.DataFrame:
    """Map decision_bar_index -> canonical trading_day (night session already
    belongs to the correct trading day in the raw column)."""
    out = cand.copy()
    idx = out["decision_bar_index"].to_numpy(dtype=int)
    td = base_td.to_numpy()
    if len(td) and (idx.max(initial=0) >= len(td)):
        raise SystemExit("STOP_TRADING_DAY_INDEX_OUT_OF_RANGE")
    out["trading_day"] = td[idx]
    return out


def drop_final_trading_day(df: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
    """Remove the final trading day of the prefix so a partially truncated
    intraday unit / proximity episode cannot receive artificial episode weight."""
    final_day = df["trading_day"].max()
    keep = df["trading_day"] != final_day
    return df[keep].copy().reset_index(drop=True), final_day


# --------------------------------------------------------------------------- #
# Step 2 — combine + time split + purge
# --------------------------------------------------------------------------- #
def combine_symbols(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(list(frames), ignore_index=True)
    df["global_episode"] = (
        df["symbol"].astype(str) + "|" + df["proximity_episode_id"].astype(str)
    )
    return df


def compute_split_boundaries(unique_days_sorted: Sequence[Any]) -> Dict[str, Any]:
    """60 / 20 / 20 over globally sorted unique trading days (deterministic)."""
    days = list(unique_days_sorted)
    n = len(days)
    i_train = int(math.floor(SPLIT_FRACS[0] * n))
    i_val = int(math.floor((SPLIT_FRACS[0] + SPLIT_FRACS[1]) * n))
    return {
        "n_days": n,
        "train_days": days[:i_train],
        "validation_days": days[i_train:i_val],
        "test_days": days[i_val:],
        "i_train": i_train,
        "i_val": i_val,
    }


def assign_time_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Assign split by globally sorted unique trading_day. No randomization."""
    days = sorted(pd.unique(df["trading_day"]))
    b = compute_split_boundaries(days)
    out = df.copy()
    out["split"] = "unassigned"
    out.loc[out["trading_day"].isin(b["train_days"]), "split"] = "train"
    out.loc[out["trading_day"].isin(b["validation_days"]), "split"] = "validation"
    out.loc[out["trading_day"].isin(b["test_days"]), "split"] = "test"
    if (out["split"] == "unassigned").any():
        raise SystemExit("STOP_SPLIT_UNASSIGNED_ROWS")
    # boundary timestamps used by the purge
    b["validation_start_time"] = (
        out.loc[out["split"] == "validation", "decision_time"].min()
        if (out["split"] == "validation").any() else None
    )
    b["test_start_time"] = (
        out.loc[out["split"] == "test", "decision_time"].min()
        if (out["split"] == "test").any() else None
    )
    return out, b


def apply_label_purge(df: pd.DataFrame, boundaries: Dict[str, Any]
                     ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Train rows require label_available_time < validation_start;
    Validation rows require label_available_time < test_start. Test untouched."""
    out = df.copy()
    before = {s: int((out["split"] == s).sum()) for s in SPLIT_NAMES}
    lat = pd.to_datetime(out["label_available_time"])
    val_start = boundaries.get("validation_start_time")
    test_start = boundaries.get("test_start_time")

    keep = pd.Series(True, index=out.index)
    if val_start is not None:
        keep &= ~((out["split"] == "train") & ~(lat < val_start))
    if test_start is not None:
        keep &= ~((out["split"] == "validation") & ~(lat < test_start))
    # NaT availability cannot be verified -> drop from train/validation
    keep &= ~(((out["split"] == "train") | (out["split"] == "validation")) & lat.isna())

    out = out[keep].copy().reset_index(drop=True)
    after = {s: int((out["split"] == s).sum()) for s in SPLIT_NAMES}
    info = {
        "before": before,
        "after": after,
        "removed": {s: int(before[s] - after[s]) for s in SPLIT_NAMES},
        "validation_start_time": str(val_start),
        "test_start_time": str(test_start),
    }
    return out, info


def assert_no_episode_spanning_split(df: pd.DataFrame) -> None:
    """Hard gate: one global episode must live in exactly one split."""
    if "split" not in df.columns or "global_episode" not in df.columns:
        raise SystemExit("STOP_SPLIT_GATE_MISSING_COLUMNS")
    g = df.groupby("global_episode")["split"].nunique()
    bad = g[g > 1]
    if len(bad) > 0:
        raise SystemExit(
            "STOP_EPISODE_SPANS_SPLIT: "
            f"{len(bad)} episode(s) span multiple splits, e.g. {list(bad.index[:3])}"
        )


def assert_fit_matrices_exclude_test(train_split: Sequence[str],
                                     eval_split: Sequence[str]) -> None:
    """Hard gate for 'Test never enters fit() / early stopping'.

    The TRAIN matrix must contain only train rows and the EVAL matrix only
    validation rows; any test row (or mixed content) is a STOP.
    """
    tr = set(np.unique(np.asarray(train_split, dtype=object)).tolist())
    ev = set(np.unique(np.asarray(eval_split, dtype=object)).tolist())
    if tr != {"train"}:
        raise SystemExit(f"STOP_TRAIN_MATRIX_NOT_PURE_TRAIN: {sorted(tr)[:5]}")
    if ev != {"validation"}:
        raise SystemExit(f"STOP_EVAL_MATRIX_NOT_PURE_VALIDATION: {sorted(ev)[:5]}")


def assert_split_dates_ordered(boundaries: Dict[str, Any]) -> None:
    tr, va, te = (boundaries["train_days"], boundaries["validation_days"],
                  boundaries["test_days"])
    if not (len(tr) and len(va) and len(te)):
        raise SystemExit("STOP_SPLIT_EMPTY_BUCKET")
    if not (max(tr) < min(va) and max(va) < min(te)):
        raise SystemExit("STOP_SPLIT_DATES_NOT_ORDERED")


# --------------------------------------------------------------------------- #
# Step 3 — weights (recomputed AFTER concatenation)
# --------------------------------------------------------------------------- #
def compute_split_weights(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Recompute normalized weights from RAW w=1/N_e after concatenation.

    w_norm_train = w_raw / mean(w_raw_train)
    w_norm_val   = w_raw / mean(w_raw_val)
    Test keeps raw weights (normalization is irrelevant for weighted ratios).
    """
    out = df.copy()
    w_raw = out["sample_weight"].to_numpy(dtype=float)
    out["w_raw"] = w_raw
    out["w_norm"] = np.nan

    info: Dict[str, Any] = {}
    for split in ("train", "validation"):
        m = (out["split"] == split).to_numpy()
        if m.any():
            mean_w = float(np.mean(w_raw[m]))
            info[f"{split}_raw_mean"] = mean_w
            if mean_w <= 0:
                raise SystemExit(f"STOP_NON_POSITIVE_RAW_WEIGHT_MEAN_{split}")
            out.loc[m, "w_norm"] = w_raw[m] / mean_w
    # Test: raw weights only (documented; ratios are scale invariant)
    m_test = (out["split"] == "test").to_numpy()
    if m_test.any():
        out.loc[m_test, "w_norm"] = w_raw[m_test]
        info["test_raw_mean"] = float(np.mean(w_raw[m_test]))
    return out, info


def verify_episode_raw_weight_sums(df: pd.DataFrame) -> Dict[str, Any]:
    """Complete global episodes must have raw-weight sum == 1.

    An episode is 'complete' when every candidate row it owns survived the
    final-trading-day removal and the purge (i.e. none of its rows were dropped).
    """
    sums = df.groupby("global_episode")["w_raw"].sum()
    counts = df.groupby("global_episode").size()
    # an episode is complete iff sum(w_raw) == n_rows * (1/N_e) == 1 within tol
    ok = np.isclose(sums.to_numpy(dtype=float), 1.0, rtol=0, atol=1e-9)
    bad = int((~ok).sum())
    complete = int(ok.sum())
    res = {
        "episodes_total": int(len(sums)),
        "episodes_complete_sum1": complete,
        "episodes_incomplete": bad,
        "max_abs_deviation": float(np.max(np.abs(sums.to_numpy(dtype=float) - 1.0)))
        if len(sums) else 0.0,
        "rows_per_episode_min": int(counts.min()) if len(counts) else 0,
        "rows_per_episode_max": int(counts.max()) if len(counts) else 0,
    }
    if bad > 0:
        res["example_incomplete"] = list(sums[~ok].index[:5])
    return res


# --------------------------------------------------------------------------- #
# Step 4 — feature matrix contract
# --------------------------------------------------------------------------- #
def assert_feature_contract(X: pd.DataFrame) -> None:
    """X must be EXACTLY core108_columns(), no forbidden column, no extra."""
    cols = list(core108_columns())
    if list(X.columns) != cols:
        missing = [c for c in cols if c not in X.columns]
        extra = [c for c in X.columns if c not in cols]
        raise SystemExit(
            f"STOP_FEATURE_CONTRACT: missing={missing[:5]} extra={extra[:5]}"
        )
    if X.shape[1] != 108:
        raise SystemExit(f"STOP_FEATURE_COUNT: {X.shape[1]} != 108")
    for c in X.columns:
        low = c.lower()
        for tok in FORBIDDEN_MODEL_TOKENS:
            if tok.lower() in low:
                raise SystemExit(f"STOP_FORBIDDEN_FEATURE: {c} (token {tok})")


def assert_phase_vocabulary(df: pd.DataFrame) -> None:
    """Every *_phase value must belong to the fixed vocabulary (hard STOP)."""
    for c in [x for x in core108_columns() if x.endswith("_phase")]:
        vals = pd.unique(df[c].dropna())
        bad = [v for v in vals if v not in PHASE_VOCAB]
        if bad:
            raise SystemExit(f"STOP_UNKNOWN_PHASE_CATEGORY: {c} -> {bad[:5]}")


def build_X(df: pd.DataFrame) -> pd.DataFrame:
    """108 columns; *_phase as ordered Categorical with the fixed vocabulary.
    Numeric NaN is preserved (no imputation)."""
    assert_phase_vocabulary(df)
    cols = list(core108_columns())
    X = df[cols].copy()
    for c in cols:
        if c.endswith("_phase"):
            X[c] = pd.Categorical(X[c], categories=list(PHASE_VOCAB))
    assert_feature_contract(X)
    return X


# --------------------------------------------------------------------------- #
# Step 5 — metrics
# --------------------------------------------------------------------------- #
def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    sw = float(np.sum(w))
    return float(np.sum(w * x) / sw) if sw > 0 else float("nan")


def weighted_rmse(y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> float:
    y = np.asarray(y, float); yhat = np.asarray(yhat, float); w = np.asarray(w, float)
    sw = float(np.sum(w))
    return float(math.sqrt(float(np.sum(w * (y - yhat) ** 2)) / sw)) if sw > 0 else float("nan")


def metrics_block(y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> Dict[str, Any]:
    """Row-level + episode-weighted metrics.

    Weighted Spearman is intentionally OMITTED (no invented weighted-rank
    definition); the unweighted Spearman is never called 'episode-weighted'.
    """
    y = np.asarray(y, float); yhat = np.asarray(yhat, float); w = np.asarray(w, float)
    n = int(len(y))
    res: Dict[str, Any] = {"rows": n}
    if n == 0:
        return res
    res["mae"] = float(np.mean(np.abs(y - yhat)))
    res["rmse"] = float(math.sqrt(float(np.mean((y - yhat) ** 2))))
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    res["r2"] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    if n > 2 and np.std(y) > 0 and np.std(yhat) > 0:
        res["spearman"] = float(spearmanr(y, yhat).statistic)
    else:
        res["spearman"] = None
    sw = float(np.sum(w))
    res["w_mae"] = float(np.sum(w * np.abs(y - yhat)) / sw) if sw > 0 else float("nan")
    res["w_rmse"] = weighted_rmse(y, yhat, w)
    ybar_w = _wmean(y, w)
    ss_res_w = float(np.sum(w * (y - yhat) ** 2))
    ss_tot_w = float(np.sum(w * (y - ybar_w) ** 2))
    res["w_r2"] = float(1.0 - ss_res_w / ss_tot_w) if ss_tot_w > 0 else float("nan")
    res["w_spearman"] = None  # deliberately omitted
    return res


def decile_table(y: np.ndarray, yhat: np.ndarray, w: np.ndarray,
                 ep: np.ndarray) -> pd.DataFrame:
    """Prediction-value deciles (decile 0 = lowest predicted)."""
    d = pd.DataFrame({"y": np.asarray(y, float), "yhat": np.asarray(yhat, float),
                      "w": np.asarray(w, float), "ep": np.asarray(ep)})
    try:
        d["decile"] = pd.qcut(d["yhat"], 10, labels=False, duplicates="drop")
    except ValueError:
        d["decile"] = 0
    rows = []
    for dec, g in d.groupby("decile"):
        sw = float(np.sum(g["w"].to_numpy()))
        rows.append({
            "decile": int(dec),
            "rows": int(len(g)),
            "distinct_episodes": int(g["ep"].nunique()),
            "raw_episode_weight": sw,
            "mean_prediction": float(g["yhat"].mean()),
            "actual_mean_y": float(g["y"].mean()),
            "weighted_actual_mean_y": (float(np.sum(g["w"].to_numpy() * g["y"].to_numpy()) / sw)
                                       if sw > 0 else float("nan")),
            "positive_y_rate": float((g["y"] > 0).mean()),
        })
    return pd.DataFrame(rows).sort_values("decile").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Step 6 — model
# --------------------------------------------------------------------------- #
def fit_direction(Xtr: pd.DataFrame, ytr: np.ndarray, wtr: np.ndarray,
                  Xval: pd.DataFrame, yval: np.ndarray, wval: np.ndarray
                  ) -> Tuple[Any, int, float]:
    """Fit one LGBMRegressor with frozen BASE_PARAMS; early stop on Validation.

    Test NEVER enters fit() or early stopping (enforced by construction: the
    caller only ever passes train/validation matrices).
    """
    check_lightgbm()  # version gate; module-level import already happened
    model = lgb.LGBMRegressor(**BASE_PARAMS)
    t0 = time.perf_counter()
    model.fit(
        Xtr, ytr,
        sample_weight=np.asarray(wtr, float),
        eval_set=[(Xval, yval)],
        eval_sample_weight=[np.asarray(wval, float)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                   lgb.log_evaluation(0)],
    )
    seconds = time.perf_counter() - t0
    best = int(getattr(model, "best_iteration_", 0) or 0)
    return model, best, float(seconds)


def gain_importance(model: Any) -> pd.DataFrame:
    imp = np.asarray(model.booster_.feature_importance(importance_type="gain"), float)
    names = list(model.booster_.feature_name())
    tot = float(imp.sum())
    df = pd.DataFrame({"feature": names, "gain": imp})
    df["gain_pct"] = (100.0 * df["gain"] / tot) if tot > 0 else 0.0
    return df.sort_values("gain", ascending=False).reset_index(drop=True)


def _permute_group(X: pd.DataFrame, cols: Sequence[str], perm: np.ndarray) -> pd.DataFrame:
    Xp = X.copy()
    for c in cols:
        vals = Xp[c].to_numpy()[perm]
        if isinstance(Xp[c].dtype, pd.CategoricalDtype):
            Xp[c] = pd.Categorical(vals, categories=Xp[c].cat.categories)
        else:
            Xp[c] = vals
    return Xp


def permutation_importance(model: Any, X: pd.DataFrame, y: np.ndarray, w: np.ndarray,
                           groups: Dict[str, List[str]], repeats: int = PERMUTATION_REPEATS,
                           seed: int = SEED) -> pd.DataFrame:
    """Grouped permutation on VALIDATION only.

    Metric: delta_RMSE_episode = RMSE_permuted - RMSE_baseline (episode-weighted,
    using the validation episode weights). All features in a group are permuted
    jointly with ONE shared row permutation. No refitting.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float); w = np.asarray(w, float)
    base = weighted_rmse(y, np.asarray(model.predict(X), float), w)
    rows = []
    for gname, cols in groups.items():
        cols = [c for c in cols if c in X.columns]
        deltas = []
        for _ in range(int(repeats)):
            perm = rng.permutation(len(X))
            Xp = _permute_group(X, cols, perm)
            deltas.append(weighted_rmse(y, np.asarray(model.predict(Xp), float), w) - base)
        d = np.asarray(deltas, float)
        rows.append({
            "group": gname,
            "n_features": len(cols),
            "mean_delta_rmse_episode": float(np.mean(d)),
            "std_delta_rmse_episode": float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
            "baseline_rmse_episode": float(base),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Step 7 — grouping helpers for importance / missingness
# --------------------------------------------------------------------------- #
def _parse_feature(col: str) -> Tuple[str, str, str]:
    tf = col.split("_")[0]
    prop_map = {
        "distance_atr": "distance",
        "phase": "phase",
        "episode_age_5m": "age",
        "approach_velocity": "velocity",
        "path_efficiency": "efficiency",
        "max_penetration_atr": "penetration",
    }
    for f, p in prop_map.items():
        suf = "_" + f
        if col.endswith(suf):
            role = col[len(tf) + 1: len(col) - len(suf)]
            return tf, role, p
    for f in ("trend_state", "slope_atr", "dev"):
        suf = "_" + f
        if col.endswith(suf):
            return tf, "DTP", f
    return tf, "UNKNOWN", col


def tf_groups() -> Dict[str, List[str]]:
    return {tf: [c for c in core108_columns() if c.split("_")[0] == tf] for tf in TF_ORDER}


def semantic_groups() -> Dict[str, List[str]]:
    fams = ["DTP"] + list(ROLES)
    out: Dict[str, List[str]] = {f: [] for f in fams}
    for c in core108_columns():
        _, role, _ = _parse_feature(c)
        if role in out:
            out[role].append(c)
    return {k: v for k, v in out.items() if v}


def property_groups() -> Dict[str, List[str]]:
    props = ["distance", "phase", "age", "velocity", "efficiency", "penetration"]
    out: Dict[str, List[str]] = {p: [] for p in props}
    for c in core108_columns():
        _, _, p = _parse_feature(c)
        if p in out:
            out[p].append(c)
    return {k: v for k, v in out.items() if v}


# --------------------------------------------------------------------------- #
# Step 8 — diagnostics
# --------------------------------------------------------------------------- #
def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missingness + aggregates by TF / semantic family / property.

    Aggregates are computed ONLY over the 108 column-level rows (never over
    previously emitted aggregate rows).
    """
    cols = list(core108_columns())
    n = len(df)
    col_rows = []
    for c in cols:
        tf, role, prop = _parse_feature(c)
        miss = int(df[c].isna().sum())
        col_rows.append({"level": "column", "name": c, "tf": tf, "family": role,
                         "property": prop, "n_rows": n, "missing": miss,
                         "missing_fraction": float(miss / n) if n else float("nan")})
    base = pd.DataFrame(col_rows)

    agg_rows = []
    for key in ("tf", "family", "property"):
        g = base.groupby(key)
        missing = g["missing"].sum()
        n_cols = g.size()
        for name in missing.index:
            cells = int(n * int(n_cols[name]))
            agg_rows.append({"level": f"by_{key}", "name": str(name), "tf": "",
                             "family": "", "property": "", "n_rows": n,
                             "missing": int(missing[name]),
                             "missing_fraction": float(missing[name] / cells) if cells else float("nan")})
    return pd.DataFrame(col_rows + agg_rows)


def phase_counts_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in [x for x in core108_columns() if x.endswith("_phase")]:
        tf = c.split("_")[0]
        role = c[len(tf) + 1: -len("_phase")]
        vc = df[c].value_counts(dropna=False)
        for ph in PHASE_VOCAB:
            rows.append({"tf": tf, "role": role, "phase": ph,
                         "rows": int(vc.get(ph, 0))})
        extra = [v for v in vc.index if v not in PHASE_VOCAB]
        for ph in extra:
            rows.append({"tf": tf, "role": role, "phase": f"UNKNOWN:{ph}",
                         "rows": int(vc[ph])})
    return pd.DataFrame(rows)


def counts_by_split(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for s in SPLIT_NAMES:
        sub = df[df["split"] == s]
        out[s] = {
            "rows": int(len(sub)),
            "episodes": int(sub["global_episode"].nunique()),
            "trading_days": int(sub["trading_day"].nunique()),
            "by_symbol": {sym: int((sub["symbol"] == sym).sum()) for sym in T15_SYMBOLS},
        }
    return out


def target_quantiles(y: np.ndarray) -> Dict[str, Optional[float]]:
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return {k: None for k in ("min", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max")}
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    vals = np.percentile(y, qs)
    keys = ["min", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max"]
    return {k: float(v) for k, v in zip(keys, vals)}


# --------------------------------------------------------------------------- #
# Step 9 — full T1.5 run
# --------------------------------------------------------------------------- #
def run_t15(
    symbols: Sequence[str] = T15_SYMBOLS,
    prefix_bars: int = T15_PREFIX_BARS,
    out_dir: Any = DEFAULT_OUT_DIR,
    artifact_root: Any = DEFAULT_ARTIFACT_ROOT,
    write_local_dataset: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full T1.5 smoke and write small auditable artifacts."""
    lgb_version = check_lightgbm()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    local_dir = out_dir / "t15"
    local_dir.mkdir(parents=True, exist_ok=True)

    S: Dict[str, Any] = {
        "task_id": TASK_ID,
        "base_sha": BASE_SHA,
        "code_sha": _git_code_sha(),
        "symbols": list(symbols),
        "prefix_bars": int(prefix_bars),
        "lightgbm_version": lgb_version,
        "core108_feature_count": len(core108_columns()),
        "seed": SEED,
        "base_params": dict(BASE_PARAMS),
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "split_fracs": list(SPLIT_FRACS),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    # ---- 1. per-symbol build -------------------------------------------------
    frames: List[pd.DataFrame] = []
    build_seconds: Dict[str, float] = {}
    integrity_by_symbol: Dict[str, Any] = {}
    final_days: Dict[str, str] = {}
    for sym in symbols:
        r = build_symbol_frame(sym, prefix_bars=prefix_bars, artifact_root=artifact_root)
        build_seconds[sym] = r["seconds"]
        integrity_by_symbol[sym] = {
            k: r["integrity"].get(k) for k in
            ("feature_rows", "candidate_rows", "unmatched_feature_rows",
             "duplicate_feature_keys", "duplicate_oracle_keys", "math_version",
             "oracle_source_sha", "cost_mode", "nan_by_family")
        }
        cand, fd = drop_final_trading_day(r["cand"])
        final_days[sym] = str(pd.Timestamp(fd).date()) if fd is not None else None
        frames.append(cand)
        if verbose:
            print(f"[build] {sym}: prefix_bars={r['n_prefix_bars']} "
                  f"candidates={len(r['cand'])} -> after final-day drop={len(cand)} "
                  f"({r['seconds']:.2f}s)")

    # ---- 2. combine + split + purge -----------------------------------------
    t0 = time.perf_counter()
    df = combine_symbols(frames)
    df, boundaries = assign_time_split(df)
    assert_split_dates_ordered(boundaries)
    df, purge = apply_label_purge(df, boundaries)
    assert_no_episode_spanning_split(df)
    df, winfo = compute_split_weights(df)
    assembly_seconds = time.perf_counter() - t0

    weight_audit = verify_episode_raw_weight_sums(df)
    S["removed_final_trading_day"] = final_days
    S["split_boundaries"] = {
        "n_trading_days": boundaries["n_days"],
        "train_days": [str(pd.Timestamp(d).date()) for d in boundaries["train_days"]],
        "validation_days": [str(pd.Timestamp(d).date()) for d in boundaries["validation_days"]],
        "test_days": [str(pd.Timestamp(d).date()) for d in boundaries["test_days"]],
        "i_train": boundaries["i_train"],
        "i_val": boundaries["i_val"],
    }
    S["split_day_ranges"] = {
        s: [str(pd.Timestamp(d).date()) for d in ([min(boundaries[f"{s}_days"]),
                                                   max(boundaries[f"{s}_days"])])]
        for s in ("train", "validation", "test")
    }
    S["purge"] = purge
    S["weight_raw_means"] = winfo
    S["weight_audit"] = weight_audit
    S["counts_by_split"] = counts_by_split(df)
    S["rows_total"] = int(len(df))
    S["episodes_total"] = int(df["global_episode"].nunique())

    if verbose:
        print(f"[split] days={boundaries['n_days']} "
              f"train/val/test rows={purge['after']['train']}/"
              f"{purge['after']['validation']}/{purge['after']['test']}")
        print(f"[weights] episodes complete(sum=1)={weight_audit['episodes_complete_sum1']} "
              f"incomplete={weight_audit['episodes_incomplete']}")

    # ---- 3. diagnostics ------------------------------------------------------
    miss = missingness_table(df)
    miss.to_csv(out_dir / "t15_missingness.csv", index=False)
    phases = phase_counts_table(df)
    S["phase_counts_total_rows"] = int(phases["rows"].sum())

    # ---- 4. matrices ---------------------------------------------------------
    X_all = build_X(df)
    cols = list(core108_columns())
    tr = (df["split"] == "train").to_numpy()
    va = (df["split"] == "validation").to_numpy()
    te = (df["split"] == "test").to_numpy()

    Xtr, Xva, Xte = X_all[tr].reset_index(drop=True), X_all[va].reset_index(drop=True), X_all[te].reset_index(drop=True)
    wtr = df.loc[tr, "w_norm"].to_numpy(float)
    wva = df.loc[va, "w_norm"].to_numpy(float)
    wte = df.loc[te, "w_raw"].to_numpy(float)

    # hard gate: Test must never reach fit() or early stopping
    assert_fit_matrices_exclude_test(df.loc[tr, "split"].to_numpy(),
                                     df.loc[va, "split"].to_numpy())

    if write_local_dataset:
        df.to_parquet(local_dir / "t15_dataset.parquet", index=False)

    # ---- 5. fit + metrics + importance ---------------------------------------
    fit_seconds: Dict[str, float] = {}
    metrics_rows: List[Dict[str, Any]] = []
    decile_frames: List[pd.DataFrame] = []
    gain_frames: List[pd.DataFrame] = []
    perm_frames: List[pd.DataFrame] = []
    direction_summary: Dict[str, Any] = {}

    t_perm_total = 0.0
    for direction, ycol in (("long", "Y_L"), ("short", "Y_S")):
        y = df[ycol].to_numpy(float)
        ytr, yva, yte = y[tr], y[va], y[te]

        # constant baseline = weighted Train mean
        ybar = _wmean(ytr, df.loc[tr, "w_raw"].to_numpy(float))
        model, best_iter, fit_sec = fit_direction(Xtr, ytr, wtr, Xva, yva, wva)
        fit_seconds[direction] = fit_sec

        pred_va = np.asarray(model.predict(Xva), float)
        pred_te = np.asarray(model.predict(Xte), float)

        blocks = {
            "baseline_validation": metrics_block(yva, np.full(len(yva), ybar), df.loc[va, "w_raw"].to_numpy(float)),
            "baseline_test": metrics_block(yte, np.full(len(yte), ybar), wte),
            "model_validation": metrics_block(yva, pred_va, df.loc[va, "w_raw"].to_numpy(float)),
            "model_test": metrics_block(yte, pred_te, wte),
        }
        for name, b in blocks.items():
            b = dict(b); b.update({"direction": direction, "block": name})
            metrics_rows.append(b)

        # deciles on untouched Test
        d = decile_table(yte, pred_te, wte, df.loc[te, "global_episode"].to_numpy())
        d.insert(0, "direction", direction)
        decile_frames.append(d)

        # per-symbol Test
        per_symbol = {}
        for sym in symbols:
            m_sym = te & (df["symbol"].to_numpy() == sym)
            if not m_sym.any():
                continue
            yy, pp, ww = y[m_sym], pred_te[(df.loc[te, "symbol"].to_numpy() == sym)], wte[(df.loc[te, "symbol"].to_numpy() == sym)]
            dd = decile_table(yy, pp, ww, df.loc[m_sym, "global_episode"].to_numpy())
            top = dd.sort_values("decile").iloc[-1] if len(dd) else None
            per_symbol[sym] = {
                "rows": int(m_sym.sum()),
                "episodes": int(df.loc[m_sym, "global_episode"].nunique()),
                "mae": float(np.mean(np.abs(yy - pp))),
                "rmse": float(math.sqrt(float(np.mean((yy - pp) ** 2)))),
                "spearman": float(spearmanr(yy, pp).statistic) if len(yy) > 2 else None,
                "w_mae": float(np.sum(ww * np.abs(yy - pp)) / np.sum(ww)),
                "top_decile_weighted_actual_mean_y": (float(top["weighted_actual_mean_y"]) if top is not None else None),
            }

        # importance
        g = gain_importance(model)
        g.insert(0, "direction", direction)
        gain_frames.append(g)

        t0 = time.perf_counter()
        for kind, groups in (("tf", tf_groups()), ("semantic", semantic_groups()),
                             ("property", property_groups())):
            p = permutation_importance(model, Xva, yva, wva, groups)
            p.insert(0, "direction", direction)
            p.insert(1, "group_kind", kind)
            perm_frames.append(p)
        t_perm_total += time.perf_counter() - t0

        direction_summary[direction] = {
            "best_iteration": best_iter,
            "fit_seconds": fit_sec,
            "baseline_train_weighted_mean": ybar,
            "target_quantiles": target_quantiles(y),
            "metrics": {k: {kk: vv for kk, vv in v.items()} for k, v in blocks.items()},
            "per_symbol_test": per_symbol,
        }
        if verbose:
            print(f"[{direction}] best_iter={best_iter} fit={fit_sec:.2f}s "
                  f"val_rmse={blocks['model_validation']['rmse']:.6f} "
                  f"test_rmse={blocks['model_test']['rmse']:.6f} "
                  f"test_spearman={blocks['model_test']['spearman']}")

    # ---- 6. artifacts --------------------------------------------------------
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "t15_metrics.csv", index=False)
    deciles_df = pd.concat(decile_frames, ignore_index=True)
    deciles_df.to_csv(out_dir / "t15_deciles.csv", index=False)
    gain_df = pd.concat(gain_frames, ignore_index=True)
    gain_df.to_csv(out_dir / "t15_feature_gain.csv", index=False)
    perm_df = pd.concat(perm_frames, ignore_index=True)
    perm_df.to_csv(out_dir / "t15_group_permutation.csv", index=False)

    S["runtime"] = {
        "feature_build_seconds": build_seconds,
        "dataset_assembly_seconds": float(assembly_seconds),
        "fit_seconds": fit_seconds,
        "permutation_seconds": float(t_perm_total),
        "peak_memory_mb": _peak_memory_mb(),
    }
    S["model"] = direction_summary
    S["oracle"] = {
        sym: {k: integrity_by_symbol[sym].get(k) for k in
              ("math_version", "oracle_source_sha", "cost_mode", "candidate_rows",
               "unmatched_feature_rows", "duplicate_feature_keys", "duplicate_oracle_keys")}
        for sym in symbols
    }
    S["integrity_by_symbol"] = integrity_by_symbol
    S["phase_counts"] = phases.to_dict(orient="records")
    S["missingness_summary"] = {
        "by_tf": miss[miss["level"] == "by_tf"][["name", "missing", "missing_fraction"]].to_dict(orient="records"),
        "by_family": miss[miss["level"] == "by_family"][["name", "missing", "missing_fraction"]].to_dict(orient="records"),
        "by_property": miss[miss["level"] == "by_property"][["name", "missing", "missing_fraction"]].to_dict(orient="records"),
        "columns_with_any_missing": int((miss[miss["level"] == "column"]["missing"] > 0).sum()),
    }

    # Artifact hashes: the 5 CSVs are hashed and embedded in summary.json.
    # summary.json cannot contain its own hash (self-referential), so its hash is
    # computed after the single write and returned for the evidence packet only.
    art_specs = [
        (out_dir / "t15_metrics.csv", len(metrics_df)),
        (out_dir / "t15_deciles.csv", len(deciles_df)),
        (out_dir / "t15_feature_gain.csv", len(gain_df)),
        (out_dir / "t15_group_permutation.csv", len(perm_df)),
        (out_dir / "t15_missingness.csv", len(miss)),
    ]
    S["artifacts"] = {
        p.name: {"path": str(p), "rows": int(n), "bytes": int(p.stat().st_size),
                 "sha256": _sha256_file(p)}
        for p, n in art_specs
    }

    summary_path = out_dir / "t15_summary.json"
    summary_path.write_text(json.dumps(_jsonable(S), indent=2, sort_keys=False))
    S["artifacts"]["t15_summary.json"] = {
        "path": str(summary_path), "rows": None,
        "bytes": int(summary_path.stat().st_size), "sha256": _sha256_file(summary_path),
    }
    return S


if __name__ == "__main__":
    res = run_t15()
    print("\n=== T1.5 DONE ===")
    print(json.dumps(_jsonable({k: res[k] for k in
                                ("rows_total", "episodes_total", "counts_by_split",
                                 "purge", "weight_audit", "runtime")}), indent=2))
