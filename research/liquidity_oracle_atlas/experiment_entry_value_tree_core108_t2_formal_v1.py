"""
experiment_entry_value_tree_core108_t2_formal_v1
================================================

FUTURE-ENTRY-VALUE-TREE-CORE108-V1-T2-FORMAL

This module implements the **preflight gate** required by the T2 contract
(Sections 2 and 19) and the full formal pipeline skeleton. The pipeline is
guarded behind ``preflight``: it MUST NOT run unless exactly 15 valid R2 Oracle
symbols are present.

The contract is explicit:
  * Section 2  -- "Require exactly the intended 15-symbol research universe.
                   If artifact count != 15: STOP."
  * Section 19 -- "If exactly 15 valid symbols and the common window leaves
                   every symbol represented in all 3 splits: continue T2.
                   Otherwise: STOP and return T2_PREFLIGHT_BLOCKED.
                   Do not improvise exclusions."

Running this module with the current local artifacts blocks with
``T2_PREFLIGHT_BLOCKED`` and prints coverage evidence, because only AG and RB
have ``intraday_dp_oracle_r2_one_entry_proximity`` artifacts.

Nothing here changes CORE108, the Oracle, labels, cost mode, LightGBM params or
episode weights. The full M0/M1/M2 + cross-symbol-holdout + bootstrap pipeline
is intended to execute only after the 15-symbol universe exists.
"""

from __future__ import annotations

# IMPORT ORDER: importing t15b first pulls in t15, which imports lightgbm at its
# top (before numpy/pandas/scipy). A late dlopen segfaults on macOS (OpenMP).
import research.liquidity_oracle_atlas.experiment_entry_value_tree_core108_t15b_aligned_v1 as B

import glob
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.build_structure_constrained_trade_oracle_dp_v2 import (
    load_oracle_artifact_v2,
)
from research.liquidity_oracle_atlas.experiment_entry_value_tree_core108_v1 import (
    DEFAULT_ARTIFACT_ROOT,
    load_raw_5m,
)
import research.liquidity_oracle_atlas.experiment_entry_value_tree_core108_t15_v1 as T
from scipy.stats import spearmanr

import json
import platform
import time

TASK_ID = "FUTURE-ENTRY-VALUE-TREE-CORE108-V1-T2-FORMAL"
BASE_SHA = "1b6f171c65bc5a538b7744e624c56196b2f2b236"
EXPECTED_SYMBOLS = 15
EXPECTED_MATH_VERSION = "intraday_dp_oracle_r2_one_entry_proximity"
EXPECTED_COST_MODE = "zero_cost"
EXPECTED_ORACLE_SOURCE_SHA = "cc7891723beb7298aa5225275b0697969d8e19bb"


# --------------------------------------------------------------------------- #
# Preflight: discover + verify the symbol universe
# --------------------------------------------------------------------------- #
def discover_symbols(artifact_root: Any = DEFAULT_ARTIFACT_ROOT) -> List[str]:
    """Symbols are discovered ONLY from the canonical R2 Oracle artifact dir."""
    return sorted(
        os.path.basename(p)
        for p in glob.glob(os.path.join(artifact_root, "*"))
        if os.path.isdir(p)
    )


def build_coverage(symbols: List[str],
                   artifact_root: Any = DEFAULT_ARTIFACT_ROOT) -> Dict[str, Any]:
    """Per-symbol availability/metadata + common calendar window. No gate."""
    rows: List[Dict[str, Any]] = []
    for s in symbols:
        raw_ok, nbar = False, None
        try:
            raw_ok = True
            nbar = int(len(load_raw_5m(s)))
        except Exception:
            raw_ok = False
        a = load_oracle_artifact_v2(artifact_root, s)
        ok = bool(a["ok"])
        mv = a["metadata"].get("math_version") if ok else None
        cm = a["metadata"].get("cost_mode") if ok else None
        sha = a["metadata"].get("oracle_source_sha") if ok else None
        nact = int(len(a["actions"])) if ok else None
        rca = a["metadata"].get("row_count_actions") if ok else None
        rct = a["metadata"].get("row_count_trades") if ok else None
        actions = a.get("actions")
        has_q_f1 = bool(ok) and actions is not None and all(
            c in actions.columns for c in ("Q_F1_S", "Q_F1_F", "Q_F1_L"))
        trades = a.get("trades")
        rows_match = bool(ok) and (rca is not None) and (int(rca) == int(nact))
        if ok and rct is not None and trades is not None:
            rows_match = rows_match and (int(rct) == len(trades))
        first_c, last_c = None, None
        if ok:
            try:
                d = B.symbol_available_days(s, artifact_root)
                days = list(d["days"])
                last_complete = days[-2] if len(days) >= 2 else days[-1]
                first_c = str(pd.Timestamp(days[0]).date())
                last_c = str(pd.Timestamp(last_complete).date())
            except Exception:
                pass
        rows.append({
            "symbol": s,
            "raw_5m_available": raw_ok,
            "oracle_ok": ok,
            "oracle_reason": (None if ok else a.get("reason")),
            "math_version": mv,
            "cost_mode": cm,
            "oracle_source_sha": sha,
            "row_count_actions": rca,
            "row_count_trades": rct,
            "rows_match": bool(rows_match),
            "has_q_f1": has_q_f1,
            "n_oracle_rows": nact,
            "n_raw_bars": nbar,
            "first_complete_day": first_c,
            "last_complete_day": last_c,
        })
    coverage = pd.DataFrame(rows)

    n_ok = int(coverage["oracle_ok"].sum()) if len(coverage) else 0
    window: Optional[Dict[str, Any]] = None
    if n_ok >= 1:
        try:
            ok_syms = [r["symbol"] for r in rows if r["oracle_ok"]]
            window = B.compute_common_window(ok_syms, artifact_root)
            window = {
                "common_start": window["common_start"],
                "common_end": window["common_end"],
                "n_window_days": window["n_window_days"],
                "per_symbol": window["per_symbol"],
            }
        except Exception as exc:  # pragma: no cover - defensive
            window = {"error": str(exc)}

    return {
        "expected_symbols": EXPECTED_SYMBOLS,
        "discovered_symbols": list(symbols),
        "n_discovered": len(symbols),
        "n_oracle_ok": n_ok,
        "expected_math_version": EXPECTED_MATH_VERSION,
        "expected_cost_mode": EXPECTED_COST_MODE,
        "coverage_rows": rows,
        "common_window": window,
    }


def preflight(symbols: Optional[List[str]] = None,
              artifact_root: Any = DEFAULT_ARTIFACT_ROOT) -> Dict[str, Any]:
    """Run the preflight gate.

    Returns the coverage dict only when exactly EXPECTED_SYMBOLS valid R2 Oracle
    symbols are present. Otherwise raises ``SystemExit('T2_PREFLIGHT_BLOCKED …')``.
    """
    if symbols is None:
        symbols = discover_symbols(artifact_root)
    cov = build_coverage(symbols, artifact_root)

    n_ok = cov["n_oracle_ok"]
    problems: List[str] = []
    if len(symbols) != EXPECTED_SYMBOLS:
        problems.append(f"discovered {len(symbols)} != expected {EXPECTED_SYMBOLS}")
    if n_ok != EXPECTED_SYMBOLS:
        problems.append(f"oracle-ok {n_ok} != expected {EXPECTED_SYMBOLS}")
    # metadata consistency among oracle-ok symbols
    ok_rows = [r for r in cov["coverage_rows"] if r["oracle_ok"]]
    for r in ok_rows:
        if r["math_version"] != EXPECTED_MATH_VERSION:
            problems.append(f"{r['symbol']} math_version={r['math_version']}")
        if r["cost_mode"] != EXPECTED_COST_MODE:
            problems.append(f"{r['symbol']} cost_mode={r['cost_mode']}")
        if r.get("oracle_source_sha") != EXPECTED_ORACLE_SOURCE_SHA:
            problems.append(f"{r['symbol']} oracle_source_sha={r.get('oracle_source_sha')}")
        if not r.get("rows_match", False):
            problems.append(f"{r['symbol']} oracle row counts mismatch")
        if not r.get("has_q_f1", False):
            problems.append(f"{r['symbol']} missing Q_F1 columns")
        if not r["raw_5m_available"]:
            problems.append(f"{r['symbol']} raw_5m missing")

    if problems:
        msg = (
            f"T2_PREFLIGHT_BLOCKED: {'; '.join(problems)}. "
            f"Discovered symbols={symbols}. "
            f"Do NOT improvise exclusions; 15-symbol R2 Oracle universe required."
        )
        raise SystemExit(msg)
    return cov


# --------------------------------------------------------------------------- #
# Formal pipeline (executes only after preflight passes — currently blocked)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Formal T2 pipeline helpers
# --------------------------------------------------------------------------- #
def _wrmse(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    return T.weighted_rmse(np.asarray(y, float), np.asarray(p, float), np.asarray(w, float))


def _rel_improve(b: float, m: float) -> float:
    return float((b - m) / b) if b and b > 0 else float("nan")


def _spear(y: np.ndarray, p: np.ndarray):
    y = np.asarray(y, float); p = np.asarray(p, float)
    if len(y) > 2 and np.std(y) > 0 and np.std(p) > 0:
        return float(spearmanr(y, p).statistic)
    return None


def _r2(y: np.ndarray, p: np.ndarray, w: np.ndarray):
    return T.metrics_block(np.asarray(y, float), np.asarray(p, float),
                           np.asarray(w, float))["r2"]


def _decile_diag(y, pred, w, ep):
    d = T.decile_table(np.asarray(y, float), np.asarray(pred, float),
                       np.asarray(w, float), np.asarray(ep))
    ds = d.sort_values("decile").reset_index(drop=True)
    wam = ds["weighted_actual_mean_y"].to_numpy(float)
    spread = float(wam[-1] - wam[0]) if len(wam) >= 2 else float("nan")
    if len(ds) >= 3 and ds["weighted_actual_mean_y"].std() > 0:
        rho = float(spearmanr(ds["decile"].to_numpy(), wam).statistic)
    else:
        rho = float("nan")
    viol = int(np.sum(np.diff(wam) < 0))
    return ds, spread, rho, viol


COMPARISONS = [
    ("M0-Baseline", None, "M0"),
    ("M1-M0", "M0", "M1"),
    ("M2-M1", "M1", "M2"),
]


def run_t2(symbols: Optional[List[str]] = None,
           artifact_root: Any = DEFAULT_ARTIFACT_ROOT,
           out_dir: Any = Path("artifacts/intraday_entry_value_tree_core108_v1"),
           write_local_dataset: bool = True) -> Dict[str, Any]:
    """Formal T2 run. Guarded: preflight() raises before any expensive work."""
    cov = preflight(symbols, artifact_root)  # raises T2_PREFLIGHT_BLOCKED if not 15
    symbols = list(cov["discovered_symbols"])
    return run_t2_pipeline(symbols, cov, artifact_root=artifact_root, out_dir=out_dir,
                           write_local_dataset=write_local_dataset)


def run_t2_pipeline(symbols, cov,
                    artifact_root: Any = DEFAULT_ARTIFACT_ROOT,
                    out_dir: Any = Path("artifacts/intraday_entry_value_tree_core108_v1"),
                    write_local_dataset: bool = True) -> Dict[str, Any]:
    """FUTURE-ENTRY-VALUE-TREE-CORE108-V1-T2-FORMAL — the ONE formal run.

    Frozen: CORE108 features, Oracle labels, cost_mode, LightGBM params, weights.
    Never recomputes the Oracle. Produces the 14 required small-evidence artifacts.
    """
    symbols = list(symbols)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    local_dir = out_dir / "t2"
    local_dir.mkdir(parents=True, exist_ok=True)
    lgb = T.check_lightgbm()
    subsets = B.feature_subsets()

    S: Dict[str, Any] = {
        "task_id": TASK_ID, "base_sha": BASE_SHA,
        "code_sha": T._git_code_sha(), "symbols": symbols,
        "lightgbm_version": lgb, "core108_feature_count": len(T.core108_columns()),
        "seed": T.SEED, "base_params": dict(T.BASE_PARAMS),
        "early_stopping_rounds": T.EARLY_STOPPING_ROUNDS,
        "split_fracs": list(T.SPLIT_FRACS), "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    # ---- coverage csv (gate evidence) -----------------------------------------
    pd.DataFrame(cov["coverage_rows"]).to_csv(out_dir / "t2_coverage.csv", index=False)

    # ---- common calendar window ----------------------------------------------
    window = B.compute_common_window(symbols, artifact_root)
    S["common_window"] = {k: v for k, v in window.items() if k != "per_symbol"}
    S["window_per_symbol"] = window["per_symbol"]
    print(f"[window] {window['common_start']} -> {window['common_end']} "
          f"({window['n_window_days']} days)", flush=True)

    # ---- per-symbol CORE108 build (full history, then window filter) ----------
    frames: List[pd.DataFrame] = []
    build_seconds: Dict[str, float] = {}
    integrity: Dict[str, Any] = {}
    for sym in symbols:
        r = B.build_symbol_frame_window(sym, window, artifact_root)
        build_seconds[sym] = r["seconds"]
        integrity[sym] = {k: r["integrity"].get(k) for k in
                          ("feature_rows", "candidate_rows", "unmatched_feature_rows",
                           "duplicate_feature_keys", "duplicate_oracle_keys",
                           "math_version", "oracle_source_sha", "cost_mode")}
        frames.append(r["cand"])
        print(f"[build] {sym}: full-bars={r['n_prefix_bars']} "
              f"cand_full={r['rows_full_history']} -> in_window={r['rows_in_window']} "
              f"({r['seconds']:.2f}s)", flush=True)

    # ---- combine / split / purge / weights -----------------------------------
    t0 = time.perf_counter()
    df = T.combine_symbols(frames)
    df, boundaries = T.assign_time_split(df)
    T.assert_split_dates_ordered(boundaries)
    df, purge = T.apply_label_purge(df, boundaries)
    T.assert_no_episode_spanning_split(df)
    df, winfo = T.compute_split_weights(df)
    B.assert_symbols_in_all_splits(df, symbols)
    assembly_seconds = time.perf_counter() - t0

    counts_tbl = B.symbol_split_counts(df, symbols)
    counts_tbl.to_csv(out_dir / "t2_counts_symbol_split.csv", index=False)
    S["purge"] = purge
    S["weight_raw_means"] = winfo
    S["weight_audit"] = T.verify_episode_raw_weight_sums(df)
    S["counts_by_symbol_split"] = counts_tbl.to_dict(orient="records")
    S["rows_total"] = int(len(df))
    S["episodes_total"] = int(df["global_episode"].nunique())
    S["split_day_ranges"] = {
        s: [str(pd.Timestamp(min(boundaries[f"{s}_days"])).date()),
            str(pd.Timestamp(max(boundaries[f"{s}_days"])).date())]
        for s in ("train", "validation", "test")
    }
    print(f"[split] rows tr/va/te = {purge['after']['train']}/"
          f"{purge['after']['validation']}/{purge['after']['test']}", flush=True)
    print(counts_tbl.to_string(index=False), flush=True)

    # ---- masks / weights / dtp ------------------------------------------------
    tr = (df["split"] == "train").to_numpy()
    va = (df["split"] == "validation").to_numpy()
    te = (df["split"] == "test").to_numpy()
    w_norm = df["w_norm"].to_numpy(float)
    w_raw = df["w_raw"].to_numpy(float)
    wtr, wva = w_norm[tr], w_norm[va]
    wva_raw, wte_raw = w_raw[va], w_raw[te]
    ready = B.dtp_ready_mask(df)
    df["dtp_4tf_ready"] = ready
    read_tbl = B.dtp_readiness_table(df, ready, symbols)
    read_tbl.to_csv(out_dir / "t2_dtp_readiness.csv", index=False)
    S["dtp_readiness"] = read_tbl.to_dict(orient="records")

    if write_local_dataset:
        df.to_parquet(local_dir / "t2_dataset.parquet", index=False)

    # ---- main M0 / M1 / M2 fits (Long / Short) -------------------------------
    predictions: Dict[tuple, tuple] = {}   # (dir, model) -> (p_va, p_te)
    fit_info: Dict[tuple, tuple] = {}      # (dir, model) -> (model, best, sec)
    y: Dict[str, np.ndarray] = {}
    ybar: Dict[str, float] = {}
    for direction, ycol in (("long", "Y_L"), ("short", "Y_S")):
        yv = df[ycol].to_numpy(float)
        y[direction] = yv
        ytr, yva_, yte_ = yv[tr], yv[va], yv[te]
        ybar[direction] = T._wmean(ytr, w_raw[tr])
        for mname in ("M0", "M1", "M2"):
            cols = subsets[mname]
            Xtr_s = B.build_X_subset(df[tr], cols)
            Xva_s = B.build_X_subset(df[va], cols)
            Xte_s = B.build_X_subset(df[te], cols)
            model, best, sec = T.fit_direction(Xtr_s, ytr, wtr, Xva_s, yva_, wva)
            p_va = np.asarray(model.predict(Xva_s), float)
            p_te = np.asarray(model.predict(Xte_s), float)
            predictions[(direction, mname)] = (p_va, p_te)
            fit_info[(direction, mname)] = (model, best, sec)
            print(f"[{direction} {mname}] best={best} fit={sec:.2f}s "
                  f"val_wrmse={_wrmse(yva_, p_va, wva_raw):.6f} "
                  f"test_wrmse={_wrmse(yte_, p_te, wte_raw):.6f}", flush=True)

    # ---- metrics / deciles / per-symbol / comparisons ------------------------
    metrics_rows: List[Dict[str, Any]] = []
    symbol_metric_rows: List[Dict[str, Any]] = []
    decile_frames: List[pd.DataFrame] = []
    ablation_rows: List[Dict[str, Any]] = []

    for direction, ycol in (("long", "Y_L"), ("short", "Y_S")):
        yv = y[direction]
        yte = yv[te]; yva_ = yv[va]
        p_te = {m: predictions[(direction, m)][1] for m in ("M0", "M1", "M2")}
        p_va = {m: predictions[(direction, m)][0] for m in ("M0", "M1", "M2")}
        ybar_v = ybar[direction]
        base_const = np.full(len(yte), ybar_v)
        base_w = {}  # baseline w_rmse by split for rel improvement
        for mname in ("M0", "M1", "M2"):
            base_const_va = np.full(len(yva_), ybar_v)
            for block, yt, pt, wt, split in (
                ("baseline_validation", yva_, base_const_va, wva_raw, "validation"),
                ("baseline_test", yte, base_const, wte_raw, "test"),
            ):
                b = T.metrics_block(yt, pt, wt)
                base_w[(mname, split)] = b["w_rmse"]
                bb = dict(b); bb.update({"model": mname, "direction": direction, "block": block})
                metrics_rows.append(bb)
            for block, yt, pt, wt, split in (
                ("model_validation", yva_, p_va[mname], wva_raw, "validation"),
                ("model_test", yte, p_te[mname], wte_raw, "test"),
            ):
                b = T.metrics_block(yt, pt, wt)
                rel = _rel_improve(base_w[(mname, split)], b["w_rmse"])
                b["rel_wrmse_improvement"] = rel
                bb = dict(b); bb.update({"model": mname, "direction": direction, "block": block})
                metrics_rows.append(bb)
            ready_te = ready[te]
            if ready_te.any():
                yt = yte[ready_te]; pt = p_te[mname][ready_te]; wt = wte_raw[ready_te]
                bc = T.metrics_block(yt, np.full(len(yt), ybar_v), wt)
                mc = T.metrics_block(yt, pt, wt)
                mc["rel_wrmse_improvement"] = _rel_improve(bc["w_rmse"], mc["w_rmse"])
                for blk, bb in (("baseline_test_dtp_ready", bc), ("model_test_dtp_ready", mc)):
                    bb2 = dict(bb); bb2.update({"model": mname, "direction": direction, "block": blk})
                    metrics_rows.append(bb2)
        # ablation
        for mname in ("M0", "M1", "M2"):
            mva = T.metrics_block(yva_, p_va[mname], wva_raw)
            mte = T.metrics_block(yte, p_te[mname], wte_raw)
            dtop = T.decile_table(yte, p_te[mname], wte_raw, df.loc[te, "global_episode"].to_numpy())
            spread = (float(dtop.sort_values("decile").iloc[-1]["weighted_actual_mean_y"]
                            - dtop.sort_values("decile").iloc[0]["weighted_actual_mean_y"])
                      if len(dtop) > 1 else float("nan"))
            _, best, sec = fit_info[(direction, mname)]
            ablation_rows.append({
                "direction": direction, "model": mname, "n_features": len(subsets[mname]),
                "best_iteration": best, "fit_seconds": sec,
                "val_w_rmse": mva["w_rmse"], "val_w_r2": mva["w_r2"],
                "val_spearman": mva["spearman"], "test_w_rmse": mte["w_rmse"],
                "test_rmse": mte["rmse"], "test_w_r2": mte["w_r2"],
                "test_spearman": mte["spearman"], "test_decile_spread_weighted": spread,
            })
        # decile diagnostics (pooled test) per model
        for mname in ("M0", "M1", "M2"):
            ds, spread, rho, viol = _decile_diag(
                yte, p_te[mname], wte_raw, df.loc[te, "global_episode"].to_numpy())
            dd = ds.copy(); dd.insert(0, "model", mname); dd.insert(1, "direction", direction)
            dd["spread"] = spread; dd["rho_D"] = rho; dd["monotonicity_violations"] = viol
            decile_frames.append(dd)
        # per-symbol metrics + comparisons
        sym_arr = df["symbol"].to_numpy()
        ep_te = df.loc[te, "global_episode"].to_numpy()
        for sym in symbols:
            m_sym = te & (sym_arr == sym)
            if not m_sym.any():
                continue
            within = (sym_arr[te] == sym)
            for mname in ("M0", "M1", "M2"):
                ps = p_te[mname][within]
                base_pred = (base_const[within] if mname == "M0"
                             else p_te["M0" if mname == "M1" else "M1"][within])
                bw = _wrmse(yte[within], base_pred, wte_raw[within])
                mw = _wrmse(yte[within], ps, wte_raw[within])
                dd = T.decile_table(yte[within], ps, wte_raw[within], ep_te[within])
                dds = dd.sort_values("decile")
                top = float(dds.iloc[-1]["weighted_actual_mean_y"]) if len(dds) else float("nan")
                bot = float(dds.iloc[0]["weighted_actual_mean_y"]) if len(dds) else float("nan")
                tbr = float(top - bot) if len(dds) >= 2 else float("nan")
                tpr = float(dds.iloc[-1]["positive_y_rate"]) if len(dds) else float("nan")
                symbol_metric_rows.append({
                    "symbol": sym, "model": mname, "direction": direction,
                    "rows": int(m_sym.sum()),
                    "episodes": int(df.loc[m_sym, "global_episode"].nunique()),
                    "baseline_wRMSE": bw, "model_wRMSE": mw,
                    "rel_wrmse_improvement": _rel_improve(bw, mw),
                    "spearman": _spear(yte[within], ps),
                    "top_decile_weighted_Y": top, "bottom_decile_weighted_Y": bot,
                    "top_bottom_spread": tbr, "top_decile_positive_Y_rate": tpr,
                })

    pd.DataFrame(metrics_rows).to_csv(out_dir / "t2_metrics.csv", index=False)
    pd.DataFrame(symbol_metric_rows).to_csv(out_dir / "t2_symbol_metrics.csv", index=False)
    pd.concat(decile_frames, ignore_index=True).to_csv(out_dir / "t2_deciles.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(out_dir / "t2_ablation.csv", index=False)

    # ---- comparison stats (pooled + per-symbol) ------------------------------
    comparison_stats: Dict[tuple, Any] = {}
    for direction, _ in (("long", None), ("short", None)):
        yte = y[direction][te]
        for comp, base_name, model_name in COMPARISONS:
            base_pred = (np.full(len(yte), ybar[direction]) if base_name is None
                         else predictions[(direction, base_name)][1])
            model_pred = predictions[(direction, model_name)][1]
            delta_pooled = _wrmse(yte, base_pred, wte_raw) - _wrmse(yte, model_pred, wte_raw)
            sym_d: Dict[str, float] = {}
            for sym in symbols:
                m_sym = te & (df["symbol"].to_numpy() == sym)
                if not m_sym.any():
                    continue
                within = (df["symbol"].to_numpy()[te] == sym)
                bws = _wrmse(yte[within], base_pred[within], wte_raw[within])
                mws = _wrmse(yte[within], model_pred[within], wte_raw[within])
                sym_d[sym] = bws - mws
            improved = int(sum(1 for v in sym_d.values() if v > 0))
            med = float(np.median(list(sym_d.values()))) if sym_d else float("nan")
            comparison_stats[(direction, comp)] = {
                "pooled_test_delta_wrmse": delta_pooled,
                "symbols_improved": improved, "n_symbols": len(sym_d),
                "median_symbol_delta": med, "symbol_delta_distribution": sym_d,
            }

    # ---- episode bootstrap (resample global_episode, no refit) ---------------
    bootstrap_rows: List[Dict[str, Any]] = []
    for direction, _ in (("long", None), ("short", None)):
        yte = y[direction][te]
        ep_te_arr = df.loc[te, "global_episode"].to_numpy()
        uniq, inv = np.unique(ep_te_arr, return_inverse=True)
        n_ep = len(uniq)
        groups = [np.where(inv == k)[0] for k in range(n_ep)]
        rng = np.random.default_rng(T.SEED)
        for comp, base_name, model_name in COMPARISONS:
            base_pred = (np.full(len(yte), ybar[direction]) if base_name is None
                         else predictions[(direction, base_name)][1])
            model_pred = predictions[(direction, model_name)][1]
            deltas = []
            for _ in range(1000):
                samp = rng.integers(0, n_ep, size=n_ep)
                idx = np.concatenate([groups[j] for j in samp])
                if len(idx) == 0:
                    continue
                d = (_wrmse(yte[idx], base_pred[idx], wte_raw[idx])
                      - _wrmse(yte[idx], model_pred[idx], wte_raw[idx]))
                deltas.append(d)
            deltas = np.asarray(deltas, float)
            lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
            bootstrap_rows.append({
                "comparison": comp, "direction": direction, "n_reps": len(deltas),
                "delta_mean": float(np.mean(deltas)),
                "delta_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                "ci_lower": lo, "ci_upper": hi,
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
            })
    pd.DataFrame(bootstrap_rows).to_csv(out_dir / "t2_bootstrap.csv", index=False)

    # ---- mandatory cross-symbol holdout (5 folds, 3 unseen each) -------------
    sorted_syms = sorted(symbols)
    fold_of = {s: i for i, s in enumerate(sorted_syms)}
    folds = {i: [s for s in sorted_syms if fold_of[s] == i] for i in range(5)}
    sym_arr = df["symbol"].to_numpy()
    cs_pred: Dict[tuple, tuple] = {}
    fold_rows: List[Dict[str, Any]] = []
    for f in range(5):
        held = folds[f]
        train_syms = [s for s in sorted_syms if s not in held]
        trm = np.isin(sym_arr, train_syms) & (df["split"].to_numpy() == "train")
        vam = np.isin(sym_arr, train_syms) & (df["split"].to_numpy() == "validation")
        tem = np.isin(sym_arr, held) & (df["split"].to_numpy() == "test")
        if tem.sum() == 0:  # safety; never triggers for the 15-symbol universe
            continue
        fold_rows.append({
            "fold": f, "held_out_symbols": ",".join(held),
            "train_symbols": ",".join(train_syms),
            "n_train_rows": int(trm.sum()), "n_validation_rows": int(vam.sum()),
            "n_eval_rows": int(tem.sum()),
        })
        for direction, ycol in (("long", "Y_L"), ("short", "Y_S")):
            yv = df[ycol].to_numpy(float)
            for mname in ("M0", "M1", "M2"):
                cols = subsets[mname]
                model, _, _ = T.fit_direction(
                    B.build_X_subset(df[trm], cols), yv[trm], w_norm[trm],
                    B.build_X_subset(df[vam], cols), yv[vam], w_norm[vam])
                p = np.asarray(model.predict(B.build_X_subset(df[tem], cols)), float)
                cs_pred[(f, direction, mname)] = (yv[tem], p, w_raw[tem], sym_arr[tem])
    pd.DataFrame(fold_rows).to_csv(out_dir / "t2_cross_symbol_folds.csv", index=False)

    cs_rows: List[Dict[str, Any]] = []
    cs_comparison_delta: Dict[tuple, float] = {}
    for direction, _ in (("long", None), ("short", None)):
        for mname in ("M0", "M1", "M2"):
            y_all = np.concatenate([cs_pred[(f, direction, mname)][0] for f in range(5)])
            p_all = np.concatenate([cs_pred[(f, direction, mname)][1] for f in range(5)])
            w_all = np.concatenate([cs_pred[(f, direction, mname)][2] for f in range(5)])
            s_all = np.concatenate([cs_pred[(f, direction, mname)][3] for f in range(5)])
            ybar_cs = T._wmean(y_all, w_all)
            bw = _wrmse(y_all, np.full(len(y_all), ybar_cs), w_all)
            mw = _wrmse(y_all, p_all, w_all)
            for sym in symbols:
                m = s_all == sym
                if not m.any():
                    continue
                ys_, ps_, ws_ = y_all[m], p_all[m], w_all[m]
                bws = _wrmse(ys_, np.full(len(ys_), ybar_cs), ws_)
                cs_rows.append({
                    "symbol": sym, "model": mname, "direction": direction,
                    "fold": fold_of[sym], "rows": int(m.sum()),
                    "w_rmse": _wrmse(ys_, ps_, ws_), "r2": _r2(ys_, ps_, ws_),
                    "spearman": _spear(ys_, ps_), "baseline_wRMSE": bws,
                    "rel_wrmse_improvement": _rel_improve(bws, _wrmse(ys_, ps_, ws_)),
                })
            cs_rows.append({
                "symbol": "POOLED", "model": mname, "direction": direction,
                "fold": -1, "rows": int(len(y_all)), "w_rmse": mw, "r2": _r2(y_all, p_all, w_all),
                "spearman": _spear(y_all, p_all), "baseline_wRMSE": bw,
                "rel_wrmse_improvement": _rel_improve(bw, mw),
            })
        for comp, base_name, model_name in COMPARISONS:
            y_all = np.concatenate([cs_pred[(f, direction, model_name)][0] for f in range(5)])
            w_all = np.concatenate([cs_pred[(f, direction, model_name)][2] for f in range(5)])
            if base_name is None:
                base_pred = np.full(len(y_all), T._wmean(y_all, w_all))
            else:
                base_pred = np.concatenate([cs_pred[(f, direction, base_name)][1] for f in range(5)])
            model_pred = np.concatenate([cs_pred[(f, direction, model_name)][1] for f in range(5)])
            cs_comparison_delta[(direction, comp)] = (
                _wrmse(y_all, base_pred, w_all) - _wrmse(y_all, model_pred, w_all))
    pd.DataFrame(cs_rows).to_csv(out_dir / "t2_cross_symbol_metrics.csv", index=False)

    # ---- importance: gain + grouped permutation + distance decomposition ------
    gain_rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []
    dist_rows: List[Dict[str, Any]] = []
    for direction, _ in (("long", None), ("short", None)):
        model = fit_info[(direction, "M2")][0]
        g = T.gain_importance(model)
        for _, r in g.iterrows():
            gain_rows.append({"direction": direction, "feature": r["feature"],
                              "gain": float(r["gain"]), "gain_pct": float(r["gain_pct"])})
        Xva = B.build_X_subset(df[va], subsets["M2"])
        yva_ = y[direction][va]
        for kind, groups in (("tf", T.tf_groups()), ("semantic", T.semantic_groups()),
                             ("property", T.property_groups())):
            p = T.permutation_importance(model, Xva, yva_, wva_raw, groups)
            for _, r in p.iterrows():
                group_rows.append({
                    "direction": direction, "group_kind": kind, "group": r["group"],
                    "n_features": int(r["n_features"]),
                    "permutation": "full",
                    "mean_delta_rmse_episode": float(r["mean_delta_rmse_episode"]),
                    "std_delta_rmse_episode": float(r["std_delta_rmse_episode"]),
                    "baseline_rmse_episode": float(r["baseline_rmse_episode"]),
                })
        dist_cols = T.property_groups()["distance"]
        full_res = T.permutation_importance(model, Xva, yva_, wva_raw, {"distance": dist_cols})
        finite_res = B.finite_mask_permutation_delta(model, Xva, yva_, wva_raw, dist_cols)
        assert B.check_finite_mask_preserved(Xva, dist_cols)
        for label, res in (("full", full_res), ("finite_value_only", finite_res)):
            if isinstance(res, pd.DataFrame):
                rr = res.iloc[0]
                row = {"direction": direction, "group": "distance", "permutation": label,
                       "n_features": len(dist_cols),
                       "mean_delta_rmse_episode": float(rr["mean_delta_rmse_episode"]),
                       "std_delta_rmse_episode": float(rr["std_delta_rmse_episode"]),
                       "baseline_rmse_episode": float(rr["baseline_rmse_episode"])}
            else:
                row = {"direction": direction, "group": "distance", "permutation": label,
                       "n_features": len(dist_cols),
                       "mean_delta_rmse_episode": float(res["mean_delta_rmse_episode"]),
                       "std_delta_rmse_episode": float(res["std_delta_rmse_episode"]),
                       "baseline_rmse_episode": float(res["baseline_rmse_episode"])}
            dist_rows.append(row)
    pd.DataFrame(gain_rows).to_csv(out_dir / "t2_importance_gain.csv", index=False)
    pd.DataFrame(group_rows).to_csv(out_dir / "t2_importance_group.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(out_dir / "t2_distance_decomposition.csv", index=False)

    # ---- DTP-ready sensitivity (retrain M1/M2 on Train & dtp_4tf_ready) ------
    for direction, _ in (("long", None), ("short", None)):
        yv = y[direction]
        for mname in ("M1", "M2"):
            cols = subsets[mname]
            trd = tr & ready
            Xtr_s = B.build_X_subset(df[trd], cols)
            Xva_s = B.build_X_subset(df[va], cols)
            Xte_s = B.build_X_subset(df[te], cols)
            model, _, _ = T.fit_direction(Xtr_s, yv[trd], w_norm[trd], Xva_s, yva_, wva)
            p_va = np.asarray(model.predict(Xva_s), float)
            p_te = np.asarray(model.predict(Xte_s), float)
            for block, yt, pt, wt, split in (
                ("model_validation_dtp_ready_retrain", yva_, p_va, wva_raw, "validation"),
                ("model_test_dtp_ready_retrain", yv[te], p_te, wte_raw, "test"),
            ):
                b = T.metrics_block(yt, pt, wt)
                b["rel_wrmse_improvement"] = _rel_improve(base_w.get((mname, split), float("nan")), b["w_rmse"])
                bb = dict(b); bb.update({"model": mname, "direction": direction, "block": block})
                metrics_rows.append(bb)
    # re-write metrics with the sensitivity rows appended
    pd.DataFrame(metrics_rows).to_csv(out_dir / "t2_metrics.csv", index=False)

    # ---- summary + formal evidence table -------------------------------------
    formal: Dict[str, Any] = {}
    for direction in ("long", "short"):
        formal[direction] = {}
        for comp, _, _ in COMPARISONS:
            cs = comparison_stats[(direction, comp)]
            boot = next(b for b in bootstrap_rows
                        if b["comparison"] == comp and b["direction"] == direction)
            formal[direction][comp] = {
                "pooled_test_delta_wrmse": cs["pooled_test_delta_wrmse"],
                "bootstrap_n_reps": boot["n_reps"],
                "bootstrap_ci_lower": boot["ci_lower"],
                "bootstrap_ci_upper": boot["ci_upper"],
                "ci_excludes_zero": boot["ci_excludes_zero"],
                "symbols_improved": cs["symbols_improved"],
                "n_symbols": cs["n_symbols"],
                "median_symbol_delta": cs["median_symbol_delta"],
                "symbol_delta_distribution": cs["symbol_delta_distribution"],
                "unseen_symbol_holdout_delta": cs_comparison_delta[(direction, comp)],
                # Reviewer assigns SUPPORTED / MIXED / NOT SUPPORTED
            }

    S["runtime"] = {
        "feature_build_seconds": build_seconds,
        "dataset_assembly_seconds": float(assembly_seconds),
        "peak_memory_mb": T._peak_memory_mb(),
    }
    S["model_fit"] = {
        f"{d}/{m}": {"best_iteration": fit_info[(d, m)][1],
                     "fit_seconds": fit_info[(d, m)][2]}
        for d in ("long", "short") for m in ("M0", "M1", "M2")
    }
    S["oracle"] = {
        sym: {k: integrity[sym].get(k) for k in
              ("math_version", "oracle_source_sha", "cost_mode", "candidate_rows",
               "unmatched_feature_rows", "duplicate_feature_keys", "duplicate_oracle_keys")}
        for sym in symbols
    }
    S["comparison_stats"] = {
        f"{d}/{c}": comparison_stats[(d, c)] for d, c in comparison_stats
    }
    S["formal_evidence"] = formal

    art_specs = [
        (out_dir / "t2_coverage.csv", len(cov["coverage_rows"])),
        (out_dir / "t2_counts_symbol_split.csv", len(counts_tbl)),
        (out_dir / "t2_metrics.csv", len(metrics_rows)),
        (out_dir / "t2_symbol_metrics.csv", len(symbol_metric_rows)),
        (out_dir / "t2_deciles.csv", sum(len(d) for d in decile_frames)),
        (out_dir / "t2_ablation.csv", len(ablation_rows)),
        (out_dir / "t2_cross_symbol_folds.csv", len(fold_rows)),
        (out_dir / "t2_cross_symbol_metrics.csv", len(cs_rows)),
        (out_dir / "t2_bootstrap.csv", len(bootstrap_rows)),
        (out_dir / "t2_importance_gain.csv", len(gain_rows)),
        (out_dir / "t2_importance_group.csv", len(group_rows)),
        (out_dir / "t2_distance_decomposition.csv", len(dist_rows)),
        (out_dir / "t2_dtp_readiness.csv", len(read_tbl)),
    ]
    S["artifacts"] = {
        p.name: {"path": str(p), "rows": int(n), "bytes": int(p.stat().st_size),
                 "sha256": T._sha256_file(p)}
        for p, n in art_specs
    }
    summary_path = out_dir / "t2_summary.json"
    summary_path.write_text(json.dumps(T._jsonable(S), indent=2, sort_keys=False))
    S["artifacts"]["t2_summary.json"] = {
        "path": str(summary_path), "rows": None,
        "bytes": int(summary_path.stat().st_size), "sha256": T._sha256_file(summary_path)}
    return S


if __name__ == "__main__":
    try:
        run_t2()
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
