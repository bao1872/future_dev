"""
experiment_entry_value_tree_core108_t15b_aligned_v1
===================================================
FUTURE-ENTRY-VALUE-TREE-CORE108-V1-T1.5B-ALIGNED

Fixes the two confounds found in the T1.5 pre-T2 audit. This is a
DATA-WINDOW / DIAGNOSTIC correction only. Nothing below changes CORE108,
the Oracle, labels, LightGBM parameters, path semantics or cost mode.

Confound 1 -- equal bar prefixes implied unequal calendar coverage.
  In T1.5 a fixed 10,000-bar prefix put AG through 2025-05-26 but RB through
  2025-08-11, so Test ended up 100% RB (AG = 0 rows): Train~AG+RB but
  Test=RB, mixing temporal change with symbol-composition change.

  T1.5B instead models the COMMON CALENDAR WINDOW:
      common_end   = min(lastCompleteDay per symbol)
      common_start = max(firstAvailableDay per symbol)
  while state is still built from bar 0 forward (full warm-up history stays
  available to the indicator kernel).

Confound 2 -- ordinary distance permutation conflates three things:
  (a) real distance magnitude, (b) structure existence / missingness
  (LightGBM splits on NaN vs non-NaN), (c) for path properties, state
  combinations that cannot exist in the real state machine.
  T1.5B adds a FINITE-VALUE-ONLY permutation (NaN mask preserved) and an
  M0/M1/M2 feature-block ablation to replace unconstrained property
  permutation as primary evidence.

Do NOT run T2 from this module. No research verdict is produced.
"""

from __future__ import annotations

# IMPORT ORDER: T imports lightgbm at its top (before numpy/pandas/scipy).
# A late dlopen of lib_lightgbm segfaults on macOS (OpenMP). Keep first.
import research.liquidity_oracle_atlas.experiment_entry_value_tree_core108_t15_v1 as T

import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.build_structure_constrained_trade_oracle_dp_v2 import (
    load_oracle_artifact_v2,
)
from research.liquidity_oracle_atlas.experiment_entry_value_tree_core108_v1 import (
    DEFAULT_ARTIFACT_ROOT,
    load_raw_5m,
)

TASK_ID = "FUTURE-ENTRY-VALUE-TREE-CORE108-V1-T1.5B-ALIGNED"
BASE_SHA = "27dafdf36c97eadb4563e4d397a8d2c95a6a3415"

SYMBOLS: Tuple[str, ...] = ("AG", "RB")
DTP_READY_COLUMNS: Tuple[str, ...] = (
    "m5_dev", "m5_slope_atr",
    "m15_dev", "m15_slope_atr",
    "h1_dev", "h1_slope_atr",
    "h4_dev", "h4_slope_atr",
)

DEFAULT_OUT_DIR = Path("artifacts/intraday_entry_value_tree_core108_v1")


# --------------------------------------------------------------------------- #
# 1. Common calendar window
# --------------------------------------------------------------------------- #
def symbol_available_days(symbol: str, artifact_root: Any = DEFAULT_ARTIFACT_ROOT
                          ) -> Dict[str, Any]:
    """Trading days that actually carry R2 Oracle decision rows for a symbol.

    ``decision_time`` equals the bar's ``availability_time``, so each Oracle row
    maps onto the canonical trading_day of that bar.
    """
    raw = load_raw_5m(symbol).sort_values("bar_start_time").reset_index(drop=True)
    av2td = dict(zip(pd.to_datetime(raw["availability_time"]),
                     pd.to_datetime(raw["trading_day"])))
    art = load_oracle_artifact_v2(artifact_root, symbol)
    if not art["ok"]:
        raise SystemExit(f"STOP_ARTIFACT_LOAD_FAILED: {symbol} {art.get('reason')}")
    dec = pd.to_datetime(art["actions"]["decision_time"])
    td = dec.map(av2td)
    if td.isna().any():
        raise SystemExit(f"STOP_DECISION_TIME_UNMAPPED: {symbol}")
    days = sorted(pd.unique(td))
    return {
        "symbol": symbol,
        "n_rows": int(len(dec)),
        "n_days": len(days),
        "first_day": days[0],
        "last_day": days[-1],
        "days": days,
    }


def compute_common_window(symbols: Sequence[str] = SYMBOLS,
                          artifact_root: Any = DEFAULT_ARTIFACT_ROOT,
                          drop_final_day: bool = True) -> Dict[str, Any]:
    """common_start = max(first days); common_end = min(last COMPLETE days).

    ``drop_final_day`` removes the last available trading day of each symbol as a
    truncation guard: its Q labels may be incomplete because they need future
    bars. This mirrors the T1.5 rule that dropped the final day of each prefix.
    State itself is still built from bar 0 (see ``build_symbol_frame_window``).
    """
    info: Dict[str, Any] = {"per_symbol": {}, "drop_final_day": bool(drop_final_day)}
    day_map: Dict[str, List[Any]] = {}
    starts, ends = [], []
    for sym in symbols:
        d = symbol_available_days(sym, artifact_root)
        days = list(d["days"])
        day_map[sym] = days  # cached; no second raw load below
        last_available = days[-1]
        last_complete = days[-2] if (drop_final_day and len(days) >= 2) else days[-1]
        info["per_symbol"][sym] = {
            "first_available_day": str(pd.Timestamp(days[0]).date()),
            "last_available_day": str(pd.Timestamp(last_available).date()),
            "last_complete_day": str(pd.Timestamp(last_complete).date()),
            "n_days": d["n_days"],
            "n_oracle_rows": d["n_rows"],
        }
        starts.append(days[0])
        ends.append(last_complete)

    common_start = max(starts)
    common_end = min(ends)
    if common_start > common_end:
        raise SystemExit("STOP_COMMON_WINDOW_EMPTY")
    info["common_start"] = str(pd.Timestamp(common_start).date())
    info["common_end"] = str(pd.Timestamp(common_end).date())
    all_days: set = set()
    for sym in symbols:
        all_days |= set(day_map[sym])
    window_days = sorted(d for d in all_days if common_start <= d <= common_end)
    info["n_window_days"] = len(window_days)
    return info


def build_symbol_frame_window(symbol: str, window: Dict[str, Any],
                              artifact_root: Any = DEFAULT_ARTIFACT_ROOT,
                              ) -> Dict[str, Any]:
    """Build CORE108 from bar 0 over the FULL history, then keep only rows whose
    trading_day lies inside the common window.

    Building the full history (no bar cap) is what preserves causal warm-up:
    the indicator/path state at every kept row was produced exactly as in
    production, streaming forward from bar 0.
    """
    t0 = time.perf_counter()
    r = T.build_symbol_frame(symbol, prefix_bars=None, artifact_root=artifact_root)
    cand = r["cand"]
    td = pd.to_datetime(cand["trading_day"])
    lo = pd.Timestamp(window["common_start"])
    hi = pd.Timestamp(window["common_end"])
    keep = (td >= lo) & (td <= hi)
    kept = cand[keep].copy().reset_index(drop=True)
    return {
        "symbol": symbol,
        "cand": kept,
        "seconds": float(time.perf_counter() - t0),
        "rows_full_history": int(len(cand)),
        "rows_in_window": int(len(kept)),
        "n_prefix_bars": r["n_prefix_bars"],
        "integrity": r["integrity"],
    }


# --------------------------------------------------------------------------- #
# 2. Split / presence gate
# --------------------------------------------------------------------------- #
def assert_symbols_in_all_splits(df: pd.DataFrame,
                                 symbols: Sequence[str] = SYMBOLS) -> None:
    """Hard gate: every symbol must have rows in Train AND Validation AND Test."""
    missing: List[str] = []
    for sym in symbols:
        for split in T.SPLIT_NAMES:
            n = int(((df["symbol"] == sym) & (df["split"] == split)).sum())
            if n <= 0:
                missing.append(f"{sym}/{split}={n}")
    if missing:
        raise SystemExit("STOP_SYMBOL_MISSING_IN_SPLIT: " + ", ".join(missing))


def symbol_split_counts(df: pd.DataFrame,
                        symbols: Sequence[str] = SYMBOLS) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        for split in T.SPLIT_NAMES:
            sub = df[(df["symbol"] == sym) & (df["split"] == split)]
            rows.append({
                "symbol": sym, "split": split,
                "rows": int(len(sub)),
                "episodes": int(sub["global_episode"].nunique()) if len(sub) else 0,
                "trading_days": int(sub["trading_day"].nunique()) if len(sub) else 0,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. DTP readiness diagnostic (audit only, never a model feature)
# --------------------------------------------------------------------------- #
def dtp_ready_mask(df: pd.DataFrame) -> np.ndarray:
    """True iff all eight m5/m15/h1/h4 dev+slope_atr values are finite."""
    ok = np.ones(len(df), dtype=bool)
    for c in DTP_READY_COLUMNS:
        ok &= np.isfinite(df[c].to_numpy(dtype=float))
    return ok


def dtp_readiness_table(df: pd.DataFrame, ready: np.ndarray,
                        symbols: Sequence[str] = SYMBOLS) -> pd.DataFrame:
    tmp = pd.DataFrame({
        "symbol": df["symbol"].to_numpy(),
        "split": df["split"].to_numpy(),
        "ready": np.asarray(ready, dtype=bool),
    })
    rows = []
    for (sym, split), g in tmp.groupby(["symbol", "split"], sort=False):
        n = len(g)
        k = int(g["ready"].sum())
        rows.append({"symbol": sym, "split": split, "rows": n, "ready_rows": k,
                     "ready_fraction": float(k / n) if n else float("nan")})
    out = pd.DataFrame(rows)
    # global row per split
    for split, g in tmp.groupby("split", sort=False):
        n = len(g); k = int(g["ready"].sum())
        rows.append({"symbol": "ALL", "split": split, "rows": n, "ready_rows": k,
                     "ready_fraction": float(k / n) if n else float("nan")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. Finite-mask-preserving permutation (distance diagnostics)
# --------------------------------------------------------------------------- #
def finite_mask_permutation_delta(model: Any, X: pd.DataFrame, y: np.ndarray,
                                  w: np.ndarray, cols: Sequence[str],
                                  repeats: int = T.PERMUTATION_REPEATS,
                                  seed: int = T.SEED) -> Dict[str, Any]:
    """Permute only FINITE values, independently per column, keeping every NaN
    position exactly where it was.

    Row order is irrelevant here (no state machine across rows), so permuting
    within the finite subset isolates feature MAGNITUDE from structure
    EXISTENCE: the NaN/non-NaN pattern the splitter may be using is untouched.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    base = T.weighted_rmse(y, np.asarray(model.predict(X), float), w)
    deltas: List[float] = []
    for _ in range(int(repeats)):
        Xp = X.copy()
        for c in cols:
            v = np.asarray(Xp[c], dtype=float).copy()
            fin = np.isfinite(v)
            if fin.sum() > 1:
                vals = v[fin]
                v[fin] = vals[rng.permutation(len(vals))]
            Xp[c] = v
        deltas.append(T.weighted_rmse(y, np.asarray(model.predict(Xp), float), w) - base)
    d = np.asarray(deltas, float)
    return {
        "mean_delta_rmse_episode": float(np.mean(d)),
        "std_delta_rmse_episode": float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
        "baseline_rmse_episode": float(base),
        "n_features": len(list(cols)),
    }


def check_finite_mask_preserved(X: pd.DataFrame, cols: Sequence[str],
                                seed: int = T.SEED) -> bool:
    """Return True iff a finite-only permutation leaves the NaN mask unchanged."""
    rng = np.random.default_rng(seed)
    Xp = X.copy()
    for c in cols:
        v = np.asarray(Xp[c], dtype=float).copy()
        fin = np.isfinite(v)
        if fin.sum() > 1:
            vals = v[fin]
            v[fin] = vals[rng.permutation(len(vals))]
        Xp[c] = v
    for c in cols:
        if not np.array_equal(np.isnan(np.asarray(X[c], dtype=float)),
                              np.isnan(np.asarray(Xp[c], dtype=float))):
            return False
    return True


# --------------------------------------------------------------------------- #
# 5. Feature blocks for the M0 / M1 / M2 ablation
# --------------------------------------------------------------------------- #
def feature_subsets() -> Dict[str, List[str]]:
    """M0 = DTP only (12); M1 = DTP + 4-role distance + phase (44); M2 = all 108."""
    cols = list(T.core108_columns())
    dtp = [c for c in cols if T._parse_feature(c)[1] == "DTP"]
    dist = [c for c in cols if T._parse_feature(c)[2] == "distance"]
    phase = [c for c in cols if T._parse_feature(c)[2] == "phase"]
    return {"M0": dtp, "M1": dtp + dist + phase, "M2": cols}


def build_X_subset(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Same construction as T.build_X but restricted to a feature block."""
    T.assert_phase_vocabulary(df)
    X = df[list(cols)].copy()
    for c in cols:
        if c.endswith("_phase"):
            X[c] = pd.Categorical(X[c], categories=list(T.PHASE_VOCAB))
    return X


# --------------------------------------------------------------------------- #
# 6. T1.5B run
# --------------------------------------------------------------------------- #
def run_t15b(symbols: Sequence[str] = SYMBOLS,
             out_dir: Any = DEFAULT_OUT_DIR,
             artifact_root: Any = DEFAULT_ARTIFACT_ROOT,
             write_local_dataset: bool = True,
             verbose: bool = True) -> Dict[str, Any]:
    lgb_version = T.check_lightgbm()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    local_dir = out_dir / "t15b"
    local_dir.mkdir(parents=True, exist_ok=True)

    S: Dict[str, Any] = {
        "task_id": TASK_ID,
        "base_sha": BASE_SHA,
        "code_sha": T._git_code_sha(),
        "symbols": list(symbols),
        "lightgbm_version": lgb_version,
        "core108_feature_count": len(T.core108_columns()),
        "seed": T.SEED,
        "base_params": dict(T.BASE_PARAMS),
        "early_stopping_rounds": T.EARLY_STOPPING_ROUNDS,
        "split_fracs": list(T.SPLIT_FRACS),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    # ---- window -------------------------------------------------------------
    window = compute_common_window(symbols, artifact_root)
    S["common_window"] = {k: v for k, v in window.items() if k != "per_symbol"}
    S["window_per_symbol"] = window["per_symbol"]
    if verbose:
        print(f"[window] {window['common_start']} -> {window['common_end']} "
              f"({window['n_window_days']} days)")

    # ---- per-symbol build (full history, then window filter) -----------------
    frames, build_seconds, integrity_by_symbol = [], {}, {}
    for sym in symbols:
        r = build_symbol_frame_window(sym, window, artifact_root)
        build_seconds[sym] = r["seconds"]
        integrity_by_symbol[sym] = {
            k: r["integrity"].get(k) for k in
            ("feature_rows", "candidate_rows", "unmatched_feature_rows",
             "duplicate_feature_keys", "duplicate_oracle_keys", "math_version",
             "oracle_source_sha", "cost_mode")
        }
        frames.append(r["cand"])
        if verbose:
            print(f"[build] {sym}: full-bars={r['n_prefix_bars']} "
                  f"cand_full={r['rows_full_history']} -> in_window={r['rows_in_window']} "
                  f"({r['seconds']:.2f}s)")

    # ---- combine / split / purge / weights ----------------------------------
    t0 = time.perf_counter()
    df = T.combine_symbols(frames)
    df, boundaries = T.assign_time_split(df)
    T.assert_split_dates_ordered(boundaries)
    df, purge = T.apply_label_purge(df, boundaries)
    T.assert_no_episode_spanning_split(df)
    df, winfo = T.compute_split_weights(df)
    # hard gate: both symbols present in all three splits
    assert_symbols_in_all_splits(df, symbols)
    assembly_seconds = time.perf_counter() - t0

    weight_audit = T.verify_episode_raw_weight_sums(df)
    counts_tbl = symbol_split_counts(df, symbols)
    counts_tbl.to_csv(out_dir / "t15b_counts_by_symbol_split.csv", index=False)

    S["purge"] = purge
    S["weight_raw_means"] = winfo
    S["weight_audit"] = weight_audit
    S["counts_by_split"] = T.counts_by_split(df)
    S["counts_by_symbol_split"] = counts_tbl.to_dict(orient="records")
    S["rows_total"] = int(len(df))
    S["episodes_total"] = int(df["global_episode"].nunique())
    S["split_day_ranges"] = {
        s: [str(pd.Timestamp(min(boundaries[f"{s}_days"])).date()),
            str(pd.Timestamp(max(boundaries[f"{s}_days"])).date())]
        for s in ("train", "validation", "test")
    }
    if verbose:
        print(f"[split] rows tr/va/te = {purge['after']['train']}/"
              f"{purge['after']['validation']}/{purge['after']['test']}")
        print(counts_tbl.to_string(index=False))

    # ---- matrices -----------------------------------------------------------
    ready = dtp_ready_mask(df)
    df["dtp_4tf_ready"] = ready  # audit column; NOT in any feature list
    read_tbl = dtp_readiness_table(df, ready, symbols)
    read_tbl.to_csv(out_dir / "t15b_dtp_readiness.csv", index=False)
    S["dtp_readiness"] = read_tbl.to_dict(orient="records")

    X_all = T.build_X(df)
    tr = (df["split"] == "train").to_numpy()
    va = (df["split"] == "validation").to_numpy()
    te = (df["split"] == "test").to_numpy()
    Xtr, Xva, Xte = (X_all[tr].reset_index(drop=True), X_all[va].reset_index(drop=True),
                     X_all[te].reset_index(drop=True))
    wtr = df.loc[tr, "w_norm"].to_numpy(float)
    wva = df.loc[va, "w_norm"].to_numpy(float)
    wte = df.loc[te, "w_raw"].to_numpy(float)

    if write_local_dataset:
        df.to_parquet(local_dir / "t15b_dataset.parquet", index=False)

    # ---- main models (== M2) ------------------------------------------------
    metrics_rows: List[Dict[str, Any]] = []
    decile_frames: List[pd.DataFrame] = []
    perm_frames: List[pd.DataFrame] = []
    dist_frames: List[pd.DataFrame] = []
    direction_summary: Dict[str, Any] = {}
    models: Dict[str, Any] = {}

    t_fit_total = 0.0
    t_perm_total = 0.0
    for direction, ycol in (("long", "Y_L"), ("short", "Y_S")):
        y = df[ycol].to_numpy(float)
        ytr, yva, yte = y[tr], y[va], y[te]
        w_raw_va = df.loc[va, "w_raw"].to_numpy(float)
        w_raw_tr = df.loc[tr, "w_raw"].to_numpy(float)

        ybar = T._wmean(ytr, w_raw_tr)
        model, best_iter, fit_sec = T.fit_direction(Xtr, ytr, wtr, Xva, yva, wva)
        t_fit_total += fit_sec
        models[direction] = model

        pred_va = np.asarray(model.predict(Xva), float)
        pred_te = np.asarray(model.predict(Xte), float)

        blocks = {
            "baseline_validation": T.metrics_block(yva, np.full(len(yva), ybar), w_raw_va),
            "baseline_test": T.metrics_block(yte, np.full(len(yte), ybar), wte),
            "model_validation": T.metrics_block(yva, pred_va, w_raw_va),
            "model_test": T.metrics_block(yte, pred_te, wte),
        }
        # Test subset where all four TF DTP blocks are ready (same model, no refit)
        ready_te = ready[te]
        if ready_te.any():
            blocks[f"model_test_dtp_ready"] = T.metrics_block(
                yte[ready_te], pred_te[ready_te], wte[ready_te])
            blocks[f"baseline_test_dtp_ready"] = T.metrics_block(
                yte[ready_te], np.full(int(ready_te.sum()), ybar), wte[ready_te])
        for name, b in blocks.items():
            bb = dict(b); bb.update({"direction": direction, "block": name})
            metrics_rows.append(bb)

        d = T.decile_table(yte, pred_te, wte, df.loc[te, "global_episode"].to_numpy())
        d.insert(0, "direction", direction)
        decile_frames.append(d)

        # per-symbol Test
        per_symbol: Dict[str, Any] = {}
        for sym in symbols:
            m_sym = te & (df["symbol"].to_numpy() == sym)
            if not m_sym.any():
                continue
            within = (df.loc[te, "symbol"].to_numpy() == sym)
            yy, pp, ww = y[m_sym], pred_te[within], wte[within]
            dd = T.decile_table(yy, pp, ww, df.loc[m_sym, "global_episode"].to_numpy())
            top = dd.sort_values("decile").iloc[-1] if len(dd) else None
            per_symbol[sym] = {
                "rows": int(m_sym.sum()),
                "episodes": int(df.loc[m_sym, "global_episode"].nunique()),
                "mae": float(np.mean(np.abs(yy - pp))),
                "rmse": float(math.sqrt(float(np.mean((yy - pp) ** 2)))),
                "spearman": (float(T.spearmanr(yy, pp).statistic)
                             if (len(yy) > 2 and np.std(yy) > 0 and np.std(pp) > 0) else None),
                "w_mae": float(np.sum(ww * np.abs(yy - pp)) / np.sum(ww)),
                "w_rmse": T.weighted_rmse(yy, pp, ww),
                "top_decile_weighted_actual_mean_y": (
                    float(top["weighted_actual_mean_y"]) if top is not None else None),
                "dtp_ready_rows": int(ready[m_sym].sum()),
            }
            # Test metrics on the DTP-ready subset of this symbol.
            # `within` / `ready_te` are TEST-ordered masks, so they must be
            # sliced down with the symbol selector BEFORE indexing the
            # per-symbol arrays (which are already test-restricted).
            ready_sym = ready_te[within]
            if ready_sym.any():
                ry, rp, rw = yy[ready_sym], pp[ready_sym], ww[ready_sym]
                per_symbol[sym]["w_rmse_dtp_ready"] = T.weighted_rmse(ry, rp, rw)
                per_symbol[sym]["spearman_dtp_ready"] = (
                    float(T.spearmanr(ry, rp).statistic)
                    if (len(ry) > 2 and np.std(ry) > 0 and np.std(rp) > 0) else None)

        # ---- permutation: standard groups + finite-mask-preserving distance --
        t0 = time.perf_counter()
        for kind, groups in (("tf", T.tf_groups()), ("semantic", T.semantic_groups()),
                             ("property", T.property_groups())):
            p = T.permutation_importance(model, Xva, yva, wva, groups)
            p.insert(0, "direction", direction)
            p.insert(1, "group_kind", kind)
            p.insert(2, "permutation", "full")
            perm_frames.append(p)

        dist_cols = T.property_groups()["distance"]
        full_res = T.permutation_importance(model, Xva, yva, wva, {"distance": dist_cols})
        finite_res = finite_mask_permutation_delta(model, Xva, yva, wva, dist_cols)
        dist_frames.append({
            "direction": direction, "group": "distance", "n_features": len(dist_cols),
            "permutation": "full",
            "mean_delta_rmse_episode": float(full_res.loc[0, "mean_delta_rmse_episode"]),
            "std_delta_rmse_episode": float(full_res.loc[0, "std_delta_rmse_episode"]),
            "baseline_rmse_episode": float(full_res.loc[0, "baseline_rmse_episode"]),
        })
        dist_frames.append({
            "direction": direction, "group": "distance", "n_features": len(dist_cols),
            "permutation": "finite_value_only",
            "mean_delta_rmse_episode": finite_res["mean_delta_rmse_episode"],
            "std_delta_rmse_episode": finite_res["std_delta_rmse_episode"],
            "baseline_rmse_episode": finite_res["baseline_rmse_episode"],
        })
        t_perm_total += time.perf_counter() - t0

        direction_summary[direction] = {
            "best_iteration": best_iter,
            "fit_seconds": fit_sec,
            "baseline_train_weighted_mean": ybar,
            "target_quantiles": T.target_quantiles(y),
            "metrics": {k: dict(v) for k, v in blocks.items()},
            "per_symbol_test": per_symbol,
        }
        if verbose:
            print(f"[{direction}] best_iter={best_iter} fit={fit_sec:.2f}s "
                  f"val_wrmse={blocks['model_validation']['w_rmse']:.6f} "
                  f"test_wrmse={blocks['model_test']['w_rmse']:.6f} "
                  f"test_spearman={blocks['model_test']['spearman']}")

    # ---- M0 / M1 / M2 ablation (no tuning; same split/weights/params) -------
    subsets = feature_subsets()
    ablation_rows: List[Dict[str, Any]] = []
    t_abl_start = time.perf_counter()
    for direction, ycol in (("long", "Y_L"), ("short", "Y_S")):
        y = df[ycol].to_numpy(float)
        ytr, yva, yte = y[tr], y[va], y[te]
        for mname in ("M0", "M1", "M2"):
            cols = subsets[mname]
            Xtr_s = build_X_subset(df[tr], cols)
            Xva_s = build_X_subset(df[va], cols)
            Xte_s = build_X_subset(df[te], cols)
            if mname == "M2":
                model = models[direction]
                best_iter = direction_summary[direction]["best_iteration"]
                fit_sec = direction_summary[direction]["fit_seconds"]
            else:
                model, best_iter, fit_sec = T.fit_direction(Xtr_s, ytr, wtr, Xva_s, yva, wva)
            p_va = np.asarray(model.predict(Xva_s), float)
            p_te = np.asarray(model.predict(Xte_s), float)
            mva = T.metrics_block(yva, p_va, df.loc[va, "w_raw"].to_numpy(float))
            mte = T.metrics_block(yte, p_te, wte)
            dtop = T.decile_table(yte, p_te, wte, df.loc[te, "global_episode"].to_numpy())
            spread = (float(dtop.sort_values("decile").iloc[-1]["weighted_actual_mean_y"]
                            - dtop.sort_values("decile").iloc[0]["weighted_actual_mean_y"])
                      if len(dtop) > 1 else float("nan"))
            ablation_rows.append({
                "direction": direction, "model": mname, "n_features": len(cols),
                "best_iteration": best_iter, "fit_seconds": fit_sec,
                "val_w_rmse": mva["w_rmse"], "val_w_r2": mva["w_r2"],
                "val_spearman": mva["spearman"],
                "test_w_rmse": mte["w_rmse"], "test_rmse": mte["rmse"],
                "test_w_r2": mte["w_r2"], "test_spearman": mte["spearman"],
                "test_decile_spread_weighted": spread,
            })
            if verbose and direction == "long":
                print(f"  [ablation {mname}] feat={len(cols)} best={best_iter} "
                      f"test_wrmse={mte['w_rmse']:.6f} spearman={mte['spearman']}")
    ablation_seconds = time.perf_counter() - t_abl_start

    # ---- artifacts ----------------------------------------------------------
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "t15b_metrics.csv", index=False)
    deciles_df = pd.concat(decile_frames, ignore_index=True)
    deciles_df.to_csv(out_dir / "t15b_deciles.csv", index=False)
    perm_df = pd.concat(perm_frames, ignore_index=True)
    perm_df.to_csv(out_dir / "t15b_group_permutation.csv", index=False)
    dist_df = pd.DataFrame(dist_frames)
    dist_df.to_csv(out_dir / "t15b_distance_permutation.csv", index=False)
    abl_df = pd.DataFrame(ablation_rows)
    abl_df.to_csv(out_dir / "t15b_ablation.csv", index=False)

    S["runtime"] = {
        "feature_build_seconds": build_seconds,
        "dataset_assembly_seconds": float(assembly_seconds),
        "main_fit_seconds_total": float(t_fit_total),
        "permutation_seconds": float(t_perm_total),
        "ablation_seconds": float(ablation_seconds),
        "peak_memory_mb": T._peak_memory_mb(),
    }
    S["model"] = direction_summary
    S["ablation"] = abl_df.to_dict(orient="records")
    S["distance_permutation"] = dist_df.to_dict(orient="records")
    S["oracle"] = {
        sym: {k: integrity_by_symbol[sym].get(k) for k in
              ("math_version", "oracle_source_sha", "cost_mode", "candidate_rows",
               "unmatched_feature_rows", "duplicate_feature_keys", "duplicate_oracle_keys")}
        for sym in symbols
    }
    art_specs = [
        (out_dir / "t15b_metrics.csv", len(metrics_df)),
        (out_dir / "t15b_deciles.csv", len(deciles_df)),
        (out_dir / "t15b_group_permutation.csv", len(perm_df)),
        (out_dir / "t15b_distance_permutation.csv", len(dist_df)),
        (out_dir / "t15b_ablation.csv", len(abl_df)),
        (out_dir / "t15b_dtp_readiness.csv", len(read_tbl)),
        (out_dir / "t15b_counts_by_symbol_split.csv", len(counts_tbl)),
    ]
    S["artifacts"] = {
        p.name: {"path": str(p), "rows": int(n), "bytes": int(p.stat().st_size),
                 "sha256": T._sha256_file(p)}
        for p, n in art_specs
    }
    summary_path = out_dir / "t15b_summary.json"
    summary_path.write_text(json.dumps(T._jsonable(S), indent=2, sort_keys=False))
    S["artifacts"]["t15b_summary.json"] = {
        "path": str(summary_path), "rows": None,
        "bytes": int(summary_path.stat().st_size), "sha256": T._sha256_file(summary_path)}
    return S


if __name__ == "__main__":
    run_t15b()
