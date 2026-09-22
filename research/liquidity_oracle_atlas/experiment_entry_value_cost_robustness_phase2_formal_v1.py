"""Phase 2 — M1 / STRUCT44 cost-robustness formal experiment.

Task ID: FUTURE-ENTRY-VALUE-M1-STRUCT44-COST-ROBUSTNESS-V1-PHASE2-FORMAL

Reuses (DOES NOT MODIFY):
  * the frozen Phase-1 cost kernel (experiment_entry_value_cost_robustness_v1):
    prepare_cost_oracle + solve_cost_labels_fast, run once per symbol.
  * the frozen Formal-T2 local dataset
    (artifacts/intraday_entry_value_tree_core108_v1/t2/t2_dataset.parquet):
    M0/M1 features, common calendar, Train/Validation/Test, global episode,
    raw/normalized weights, symbol, decision keys. NOT rebuilt per kappa.
  * the frozen Formal-T2 LightGBM config (T.BASE_PARAMS / SEED /
    EARLY_STOPPING_ROUNDS), M0=DTP12, M1=STRUCT44 only (no M2).

Per kappa:
  1. prepare each of the 15 symbols EXACTLY ONCE, run all five kappas through
     solve_cost_labels_fast (full Bellman, F1-only persistence).
  2. join cost labels onto the frozen modeling dataset on (symbol, decision_time)
     and audit decision_bar_index. HARD GATE: the joined modeling row universe
     must be identical for all five kappas (candidate mask is kappa-independent).
  3. model M0/M1 for Long/Short with the exact Formal-T2 LightGBM config.
     Validation only for early stopping; Test never enters training/selection.
  4. compute primary statistic, opportunity survival, target stability vs kappa=0,
     per-symbol Test, unseen-symbol (5-fold leave-3-out) evaluation, and a 1000x
     episode-bootstrap CI.

Artifacts (small, committed):
  cost_label_stats.csv, cost_metrics.csv, cost_deciles.csv, cost_symbol_metrics.csv,
  cost_cross_symbol_metrics.csv, cost_bootstrap.csv, cost_target_stability.csv,
  cost_curve.csv, cost_summary.json
Phase-1 artifacts are retained.

Large label / model / data parquets stay local and untracked (phase2_cache/).

Run:
  .venv/bin/python -m research.liquidity_oracle_atlas.experiment_entry_value_cost_robustness_phase2_formal_v1 [--smoke] [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# IMPORT ORDER (mirror Formal-T2): importing B (t15b) first pulls t15 which
# imports lightgbm before numpy/pandas/scipy to avoid an OpenMP dlopen segfault.
import research.liquidity_oracle_atlas.experiment_entry_value_tree_core108_t15b_aligned_v1 as B  # noqa: E402
import research.liquidity_oracle_atlas.experiment_entry_value_tree_core108_t15_v1 as T  # noqa: E402
import research.liquidity_oracle_atlas.experiment_entry_value_cost_robustness_v1 as C  # noqa: E402
from research.liquidity_oracle_atlas.experiment_entry_value_cost_robustness_v1 import (  # noqa: E402
    KAPPAS,
    SYMBOLS_15,
    CostKernelStats,
    prepare_cost_oracle,
    solve_cost_labels_fast,
)
from research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1 import (  # noqa: E402
    KernelCounters,
)

TASK_ID = "FUTURE-ENTRY-VALUE-M1-STRUCT44-COST-ROBUSTNESS-V1-PHASE2-FORMAL"
DATASET_PATH = Path(
    "artifacts/intraday_entry_value_tree_core108_v1/t2/t2_dataset.parquet"
)
OUT_DIR = Path("artifacts/entry_value_cost_robustness_m1_v1")
CACHE_DIR = OUT_DIR / "phase2_cache"
MODELS: Tuple[str, ...] = ("M0", "M1")
COMPARISONS = [
    ("M0-Baseline", None, "M0"),
    ("M1-M0", "M0", "M1"),
]
CORE_FRICTION_MAX = 0.02
STRESS_KAPPA = 0.05


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _wrmse(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    return T.weighted_rmse(np.asarray(y, float), np.asarray(p, float), np.asarray(w, float))


def _rel_improve(b: float, m: float) -> float:
    return float((b - m) / b) if b and b > 0 else float("nan")


def _spear(y: np.ndarray, p: np.ndarray):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if len(y) > 2 and np.std(y) > 0 and np.std(p) > 0:
        return float(spearmanr(y, p).statistic)
    return None


def _pear(y: np.ndarray, p: np.ndarray):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if len(y) > 2 and np.std(y) > 0 and np.std(p) > 0:
        return float(pearsonr(y, p).statistic)
    return None


# --------------------------------------------------------------------------- #
# Stage 1 — cost labels (prepare once per symbol, 5 kappa passes)
# --------------------------------------------------------------------------- #
def compute_cost_labels(
    symbols: List[str] = list(SYMBOLS_15),
    force: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Prepare each symbol once, solve 5 kappas, cache per symbol.

    Cached per symbol in CACHE_DIR/cost_{sym}.parquet so a timeout loses at most
    the in-flight symbol. Returns symbol -> long DataFrame (all kappas stacked).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        p = CACHE_DIR / f"cost_{sym}.parquet"
        if p.exists() and not force:
            out[sym] = pd.read_parquet(p)
            print(f"[cost] cached {sym} rows={len(out[sym])}", flush=True)
            continue
        t0 = time.perf_counter()
        counters = KernelCounters()
        stats = CostKernelStats()
        ctx = prepare_cost_oracle(sym, counters, stats)
        frames = [solve_cost_labels_fast(ctx, k, stats) for k in KAPPAS]
        comb = pd.concat(frames, ignore_index=True)
        comb.to_parquet(p, index=False)
        out[sym] = comb
        print(
            f"[cost] {sym}: prepare={stats.prepare_count} dp_pass={stats.dp_pass_count} "
            f"({time.perf_counter() - t0:.1f}s) rows={len(comb)}",
            flush=True,
        )
    return out


# --------------------------------------------------------------------------- #
# Stage 2 — join + audit + hard gate
# --------------------------------------------------------------------------- #
def join_and_audit(
    t2: pd.DataFrame,
    cost_by_sym: Dict[str, pd.DataFrame],
) -> Tuple[Dict[float, pd.DataFrame], Dict[str, Any]]:
    """Join cost labels onto t2 per kappa; audit; enforce hard gate.

    Returns (joined_by_kappa, audit_dict). joined_by_kappa[kappa] has the t2
    feature/meta columns plus the cost-kernel Y_L/Y_S (renamed over t2's R2 Y),
    best_F1, is_candidate, atr5m.
    """
    # Stack cost labels into one long frame with kappa.
    long_frames = []
    for sym, df in cost_by_sym.items():
        d = df.copy()
        d["symbol"] = sym
        long_frames.append(
            d[
                [
                    "symbol", "decision_time", "decision_bar_index", "kappa",
                    "Y_L", "Y_S", "best_F1", "is_candidate", "atr5m",
                ]
            ]
        )
    long = pd.concat(long_frames, ignore_index=True)

    joined: Dict[float, pd.DataFrame] = {}
    matched_universe: Dict[float, set] = {}
    for k in KAPPAS:
        sub = long[long["kappa"] == k][
            ["symbol", "decision_time", "decision_bar_index", "Y_L", "Y_S",
             "best_F1", "is_candidate", "atr5m"]
        ].rename(columns={
            "decision_bar_index": "decision_bar_index_cost",
            "best_F1": "best_F1_cost",
            "is_candidate": "is_candidate_cost",
            "atr5m": "atr5m_cost",
        })
        m = t2.drop(columns=["Y_L", "Y_S"]).merge(
            sub, on=["symbol", "decision_time"], how="left"
        )
        # Audit: cost decision_bar_index must equal t2 decision_bar_index on
        # matched rows (unmatched rows have NaN cost bar index by construction).
        tbar = m["decision_bar_index"].to_numpy(float)
        cbar = m["decision_bar_index_cost"].to_numpy(float)
        finite = np.isfinite(cbar)
        bar_mismatch = int((tbar[finite] != cbar[finite]).sum())
        notna = m["Y_L"].notna().to_numpy()
        universe = set(
            zip(m["symbol"].to_numpy()[notna], m["decision_time"].to_numpy()[notna])
        )
        joined[k] = m
        matched_universe[k] = universe
        print(
            f"[join] kappa={k}: rows={len(m)} y_finite={int(notna.sum())} "
            f"bar_index_mismatch={bar_mismatch}",
            flush=True,
        )

    # Hard gate: identical matched universe across all kappas.
    ref = matched_universe[KAPPAS[0]]
    universe_ok = all(matched_universe[k] == ref for k in KAPPAS)
    audit = {
        "hard_gate_universe_identical": bool(universe_ok),
        "matched_rows_per_kappa": {str(k): len(matched_universe[k]) for k in KAPPAS},
        "n_t2_rows": int(len(t2)),
    }
    if not universe_ok:
        diffs = {}
        for k in KAPPAS[1:]:
            diffs[str(k)] = {
                "extra": len(matched_universe[k] - ref),
                "missing": len(ref - matched_universe[k]),
            }
        audit["universe_diffs"] = diffs
        raise SystemExit(f"STOP_COST_JOIN_UNIVERSE_MISMATCH: {json.dumps(audit)}")
    return joined, audit


# --------------------------------------------------------------------------- #
# Stage 3 — modeling (per kappa; resumable per kappa)
# --------------------------------------------------------------------------- #
def _cross_symbol_holdout(
    df: pd.DataFrame, subsets: Dict[str, List[str]], symbols: List[str], kappa: float
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[tuple, float]]:
    """Frozen 5-fold leave-3-symbols-out. fold = sorted_symbol_index % 5.

    Mirrors Formal-T2 _cross_symbol_holdout but limited to M0/M1 and one kappa.
    Returns (fold_rows, cs_rows, cs_comparison_delta).
    """
    sorted_syms = sorted(symbols)
    fold_of = {s: i % 5 for i, s in enumerate(sorted_syms)}
    folds = {f: [s for s in sorted_syms if fold_of[s] == f] for f in range(5)}
    # The 5-fold x 3-held-out contract holds only for the full 15-symbol universe.
    if len(symbols) == 15:
        assert all(len(folds[f]) == 3 for f in range(5)), "each fold must hold 3 symbols"
        held_all = [s for f in range(5) for s in folds[f]]
        assert sorted(held_all) == sorted(symbols)
        assert len(held_all) == 15 and len(set(held_all)) == 15

    sym_arr = df["symbol"].to_numpy()
    tr = (df["split"].to_numpy() == "train")
    va = (df["split"].to_numpy() == "validation")
    te = (df["split"].to_numpy() == "test")
    w_norm = df["w_norm"].to_numpy(float)
    w_raw = df["w_raw"].to_numpy(float)

    fold_rows: List[Dict[str, Any]] = []
    cs_pred: Dict[tuple, tuple] = {}
    for f in range(5):
        held = folds[f]
        train_syms = [s for s in sorted_syms if s not in held]
        trm = np.isin(sym_arr, train_syms) & tr
        vam = np.isin(sym_arr, train_syms) & va
        tem = np.isin(sym_arr, held) & te
        fold_rows.append({
            "fold": f, "held_out_symbols": ",".join(held),
            "train_symbols": ",".join(train_syms),
            "n_train_rows": int(trm.sum()), "n_validation_rows": int(vam.sum()),
            "n_eval_rows": int(tem.sum()),
        })
        for direction, ycol in (("long", "Y_L"), ("short", "Y_S")):
            yv = df[ycol].to_numpy(float)
            for mname in MODELS:
                cols = subsets[mname]
                # Robust to an empty held-out set (cannot occur with the full
                # 15-symbol universe, but keeps the function well-defined).
                if not tem.any():
                    cs_pred[(f, direction, mname)] = (
                        np.array([]), np.array([]), np.array([]), np.array([]))
                    continue
                model, _, _ = T.fit_direction(
                    B.build_X_subset(df[trm], cols), yv[trm], w_norm[trm],
                    B.build_X_subset(df[vam], cols), yv[vam], w_norm[vam])
                p = np.asarray(model.predict(B.build_X_subset(df[tem], cols)), float)
                cs_pred[(f, direction, mname)] = (yv[tem], p, w_raw[tem], sym_arr[tem])

    fold_train_mean: Dict[tuple, float] = {}
    for f in range(5):
        for direction, ycol in (("long", None), ("short", None)):
            held = folds[f]
            seen = [s for s in sorted_syms if s not in held]
            trm = np.isin(sym_arr, seen) & tr
            yv = df[ycol].to_numpy(float) if False else df[
                "Y_L" if direction == "long" else "Y_S"].to_numpy(float)
            fold_train_mean[(f, direction)] = T._wmean(yv[trm], w_norm[trm])

    cs_rows: List[Dict[str, Any]] = []
    cs_comparison_delta: Dict[tuple, float] = {}
    for direction, ycol in (("long", "Y_L"), ("short", "Y_S")):
        base_pred_all = np.concatenate([
            np.full(int((np.isin(sym_arr, folds[f]) & te).sum()),
                    fold_train_mean[(f, direction)])
            for f in range(5)])
        yv = df[ycol].to_numpy(float)
        for mname in MODELS:
            y_all = np.concatenate([cs_pred[(f, direction, mname)][0] for f in range(5)])
            p_all = np.concatenate([cs_pred[(f, direction, mname)][1] for f in range(5)])
            w_all = np.concatenate([cs_pred[(f, direction, mname)][2] for f in range(5)])
            s_all = np.concatenate([cs_pred[(f, direction, mname)][3] for f in range(5)])
            bw = _wrmse(y_all, base_pred_all, w_all)
            mw = _wrmse(y_all, p_all, w_all)
            for sym in symbols:
                m = s_all == sym
                if not m.any():
                    continue
                ys_, ps_, ws_ = y_all[m], p_all[m], w_all[m]
                bws = _wrmse(
                    ys_, np.full(len(ys_), fold_train_mean[(fold_of[sym], direction)]), ws_)
                cs_rows.append({
                    "symbol": sym, "model": mname, "direction": direction,
                    "fold": fold_of[sym], "rows": int(m.sum()),
                    "w_rmse": _wrmse(ys_, ps_, ws_), "r2": T.metrics_block(ys_, ps_, ws_)["r2"],
                    "spearman": _spear(ys_, ps_), "baseline_wRMSE": bws,
                    "rel_wrmse_improvement": _rel_improve(bws, _wrmse(ys_, ps_, ws_)),
                })
            cs_rows.append({
                "symbol": "POOLED", "model": mname, "direction": direction,
                "fold": -1, "rows": int(len(y_all)), "w_rmse": mw,
                "r2": T.metrics_block(y_all, p_all, w_all)["r2"],
                "spearman": _spear(y_all, p_all), "baseline_wRMSE": bw,
                "rel_wrmse_improvement": _rel_improve(bw, mw),
            })
        for comp, base_name, model_name in COMPARISONS:
            y_all = np.concatenate([cs_pred[(f, direction, model_name)][0] for f in range(5)])
            w_all = np.concatenate([cs_pred[(f, direction, model_name)][2] for f in range(5)])
            if base_name is None:
                base_pred = base_pred_all
            else:
                base_pred = np.concatenate(
                    [cs_pred[(f, direction, base_name)][1] for f in range(5)])
            model_pred = np.concatenate(
                [cs_pred[(f, direction, model_name)][1] for f in range(5)])
            cs_comparison_delta[(direction, comp)] = (
                _wrmse(y_all, base_pred, w_all) - _wrmse(y_all, model_pred, w_all))
    return pd.DataFrame(fold_rows), pd.DataFrame(cs_rows), cs_comparison_delta


def run_modeling(
    joined_by_kappa: Dict[float, pd.DataFrame],
    symbols: List[str] = list(SYMBOLS_15),
    kappas: List[float] = list(KAPPAS),
    force: bool = False,
) -> Dict[float, Dict[str, Any]]:
    """For each kappa: main M0/M1 fits + cross-symbol holdout. Cache per kappa.

    Returns kappa -> dict with 'pred_path', 'cross_path', 'cs_delta', 'fold_rows'.
    """
    subsets = B.feature_subsets()
    t0_all = time.perf_counter()
    result: Dict[float, Dict[str, Any]] = {}
    for k in kappas:
        mpath = CACHE_DIR / f"pred_{k:.4f}.parquet"
        cpath = CACHE_DIR / f"cross_{k:.4f}.parquet"
        dpath = CACHE_DIR / f"cross_meta_{k:.4f}.json"
        if mpath.exists() and cpath.exists() and dpath.exists() and not force:
            print(f"[model] kappa={k}: cached", flush=True)
            result[k] = {"pred_path": mpath, "cross_path": cpath,
                         "cs_delta": json.loads(dpath.read_text())["cs_comparison_delta"],
                         "fold_rows": json.loads(dpath.read_text())["fold_rows"]}
            continue

        t0 = time.perf_counter()
        df = joined_by_kappa[k]
        tr = (df["split"].to_numpy() == "train")
        va = (df["split"].to_numpy() == "validation")
        te = (df["split"].to_numpy() == "test")
        w_norm = df["w_norm"].to_numpy(float)
        w_raw = df["w_raw"].to_numpy(float)
        y = {"long": df["Y_L"].to_numpy(float), "short": df["Y_S"].to_numpy(float)}
        ybar = {d: T._wmean(y[d][tr], w_raw[tr]) for d in ("long", "short")}

        # ---- main M0/M1 fits ----
        predictions: Dict[tuple, tuple] = {}
        for direction, _ in (("long", None), ("short", None)):
            yv = y[direction]
            ytr, yva_, yte_ = yv[tr], yv[va], yv[te]
            for mname in MODELS:
                cols = subsets[mname]
                Xtr = B.build_X_subset(df[tr], cols)
                Xva = B.build_X_subset(df[va], cols)
                Xte = B.build_X_subset(df[te], cols)
                model, best, sec = T.fit_direction(
                    Xtr, ytr, w_norm[tr], Xva, yva_, w_norm[va])
                p_va = np.asarray(model.predict(Xva), float)
                p_te = np.asarray(model.predict(Xte), float)
                # store full-length aligned arrays (NaN outside validation/test)
                p_va_full = np.full(len(df), np.nan, dtype=float)
                p_te_full = np.full(len(df), np.nan, dtype=float)
                p_va_full[va] = p_va
                p_te_full[te] = p_te
                predictions[(direction, mname)] = (p_va_full, p_te_full)
                print(f"[model] kappa={k} {direction} {mname} best={best} "
                      f"fit={sec:.1f}s test_wrmse={_wrmse(yte_, p_te, w_raw[te]):.5f}",
                      flush=True)

        # ---- assemble per-row prediction frame ----
        pred = pd.DataFrame({
            "symbol": df["symbol"].to_numpy(),
            "decision_time": df["decision_time"].to_numpy(),
            "split": df["split"].to_numpy(),
            "global_episode": df["global_episode"].to_numpy(),
            "w_raw": w_raw, "w_norm": w_norm,
            "Y_L": y["long"], "Y_S": y["short"],
        })
        for direction, ycol in (("long", "Y_L"), ("short", "Y_S")):
            for mname in MODELS:
                pred[f"p_va_{direction}_{mname}"] = predictions[(direction, mname)][0]
                pred[f"p_te_{direction}_{mname}"] = predictions[(direction, mname)][1]
        pred.to_parquet(mpath, index=False)

        # ---- cross-symbol holdout ----
        fold_rows, cs_rows, cs_delta = _cross_symbol_holdout(df, subsets, symbols, k)
        cs_rows.to_parquet(cpath, index=False)
        dpath.write_text(json.dumps({
            "fold_rows": fold_rows.to_dict(orient="records"),
            "cs_comparison_delta": {
                f"{d}/{c}": v for (d, c), v in cs_delta.items()
            },
        }, indent=2))

        result[k] = {"pred_path": mpath, "cross_path": cpath,
                     "cs_delta": {f"{d}/{c}": v for (d, c), v in cs_delta.items()},
                     "fold_rows": fold_rows.to_dict(orient="records")}
        print(f"[model] kappa={k} done ({time.perf_counter() - t0:.1f}s)", flush=True)
    print(f"[model] ALL kappas done ({time.perf_counter() - t0_all:.1f}s)", flush=True)
    return result


# --------------------------------------------------------------------------- #
# Stage 4 — artifacts
# --------------------------------------------------------------------------- #
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


def _comparison_stats(y_test, predictions, w_test, symbols, sym_arr, te_mask):
    """Pooled + per-symbol delta for M0-Baseline and M1-M0 (Test)."""
    stats: Dict[tuple, Any] = {}
    for direction, _ in (("long", None), ("short", None)):
        yte = y_test[direction][te_mask]
        for comp, base_name, model_name in COMPARISONS:
            base_pred = (np.full(len(yte), T._wmean(y_test[direction][te_mask], w_test[te_mask]))
                         if base_name is None
                         else predictions[(direction, base_name)][1][te_mask])
            model_pred = predictions[(direction, model_name)][1][te_mask]
            delta_pooled = _wrmse(yte, base_pred, w_test[te_mask]) - _wrmse(
                yte, model_pred, w_test[te_mask])
            sym_d: Dict[str, float] = {}
            for sym in symbols:
                m_sym = te_mask & (sym_arr == sym)
                if not m_sym.any():
                    continue
                within = (sym_arr[te_mask] == sym)
                bws = _wrmse(yte[within], base_pred[within], w_test[te_mask][within])
                mws = _wrmse(yte[within], model_pred[within], w_test[te_mask][within])
                sym_d[sym] = bws - mws
            improved = int(sum(1 for v in sym_d.values() if v > 0))
            med = float(np.median(list(sym_d.values()))) if sym_d else float("nan")
            stats[(direction, comp)] = {
                "pooled_test_delta_wrmse": delta_pooled,
                "symbols_improved": improved, "n_symbols": len(sym_d),
                "median_symbol_delta": med, "symbol_delta_distribution": sym_d,
            }
    return stats


def write_artifacts(
    joined_by_kappa: Dict[float, pd.DataFrame],
    cost_by_sym: Dict[str, pd.DataFrame],
    model_result: Dict[float, Dict[str, Any]],
    symbols: List[str] = list(SYMBOLS_15),
    kappas: List[float] = list(KAPPAS),
) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subsets = B.feature_subsets()

    # ---- reload t2 meta constants ----
    t2 = next(iter(joined_by_kappa.values()))
    sym_arr_all = t2["symbol"].to_numpy()
    te_all = (t2["split"].to_numpy() == "test")

    # ===================================================================== #
    # cost_label_stats.csv  (per kappa: candidate / opportunity survival / F1)
    # ===================================================================== #
    label_rows: List[Dict[str, Any]] = []
    for k in kappas:
        # candidate / Y from cost labels (use joined, which carries cost Y)
        m = joined_by_kappa[k]
        cand = m["is_candidate_cost"].to_numpy(bool)
        yl = m["Y_L"].to_numpy(float)
        ys = m["Y_S"].to_numpy(float)
        best = m["best_F1_cost"].to_numpy(int)
        # opportunity survival on the candidate set
        yl_c = yl[cand]
        ys_c = ys[cand]
        p_l = float(np.mean(yl_c > 0)) if cand.sum() else float("nan")
        p_s = float(np.mean(ys_c > 0)) if cand.sum() else float("nan")
        p_max = float(np.mean(np.maximum(yl_c, ys_c) > 0)) if cand.sum() else float("nan")
        # best_F1 is populated (0/1/2) only on valid decision rows; non-decision
        # rows carry 127 (fill) or -1 (no action). Count only valid action codes.
        valid_f1 = (best >= 0) & (best <= 2)
        f1_counts = np.bincount(best[cand & valid_f1], minlength=3)  # 0=S,1=F,2=L
        total = int(cand.sum())
        label_rows.append({
            "kappa": k,
            "n_rows": int(len(m)),
            "n_candidate": total,
            "candidate_rate": float(total / len(m)) if len(m) else float("nan"),
            "P_YL_gt_0": p_l,
            "P_YS_gt_0": p_s,
            "P_max_YL_YS_gt_0": p_max,
            "F1_rate_short": float(f1_counts[0] / total) if total else float("nan"),
            "F1_rate_flat": float(f1_counts[1] / total) if total else float("nan"),
            "F1_rate_long": float(f1_counts[2] / total) if total else float("nan"),
            "mean_YL": float(np.mean(yl_c)) if total else float("nan"),
            "std_YL": float(np.std(yl_c)) if total else float("nan"),
            "mean_YS": float(np.mean(ys_c)) if total else float("nan"),
            "std_YS": float(np.std(ys_c)) if total else float("nan"),
        })
    pd.DataFrame(label_rows).to_csv(OUT_DIR / "cost_label_stats.csv", index=False)

    # ===================================================================== #
    # per-kappa metrics / deciles / symbol / comparison / bootstrap
    # ===================================================================== #
    metrics_rows: List[Dict[str, Any]] = []
    decile_frames: List[pd.DataFrame] = []
    symbol_metric_rows: List[Dict[str, Any]] = []
    bootstrap_rows: List[Dict[str, Any]] = []
    comparison_stats_all: Dict[float, Dict[tuple, Any]] = {}
    target_stability_rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []

    # reference kappa=0 Y for target stability
    ref = {d: joined_by_kappa[0.0][d].to_numpy(float)
           for d in ("Y_L", "Y_S")} if 0.0 in joined_by_kappa else None

    for k in kappas:
        pred = pd.read_parquet(model_result[k]["pred_path"])
        df = joined_by_kappa[k]
        # masks
        tr = (pred["split"].to_numpy() == "train")
        va = (pred["split"].to_numpy() == "validation")
        te = (pred["split"].to_numpy() == "test")
        w_norm = pred["w_norm"].to_numpy(float)
        w_raw = pred["w_raw"].to_numpy(float)
        wva_raw = w_raw[va]
        wte_raw = w_raw[te]
        sym_arr = pred["symbol"].to_numpy()
        ep_te = pred["global_episode"].to_numpy()[te]
        y = {"long": pred["Y_L"].to_numpy(float), "short": pred["Y_S"].to_numpy(float)}
        ybar = {d: T._wmean(y[d][tr], w_raw[tr]) for d in ("long", "short")}

        predictions: Dict[tuple, tuple] = {}
        for direction, _ in (("long", None), ("short", None)):
            for mname in MODELS:
                predictions[(direction, mname)] = (
                    pred[f"p_va_{direction}_{mname}"].to_numpy(float),
                    pred[f"p_te_{direction}_{mname}"].to_numpy(float),
                )
        # validation/test-length slices (NaN-masked outside the split)
        pred_p_va = {d: {m: predictions[(d, m)][0][va] for m in MODELS}
                     for d in ("long", "short")}
        pred_p_te = {d: {m: predictions[(d, m)][1][te] for m in MODELS}
                     for d in ("long", "short")}

        base_w: Dict[tuple, float] = {}
        for direction, _ in (("long", None), ("short", None)):
            yv = y[direction]
            yte = yv[te]
            yva_ = yv[va]
            ybar_v = ybar[direction]
            base_const_va = np.full(len(yva_), ybar_v)
            base_const_te = np.full(len(yte), ybar_v)
            for mname in MODELS:
                base_w[(mname, "validation")] = _wrmse(yva_, base_const_va, wva_raw)
                base_w[(mname, "test")] = _wrmse(yte, base_const_te, wte_raw)

        for direction, _ in (("long", None), ("short", None)):
            yv = y[direction]
            yte = yv[te]
            yva_ = yv[va]
            p_te = {m: pred_p_te[direction][m] for m in MODELS}
            p_va = {m: pred_p_va[direction][m] for m in MODELS}
            ybar_v = ybar[direction]
            base_const_va = np.full(len(yva_), ybar_v)
            base_const_te = np.full(len(yte), ybar_v)
            for mname in MODELS:
                for block, yt, pt, wt, split in (
                    ("baseline_validation", yva_, base_const_va, wva_raw, "validation"),
                    ("baseline_test", yte, base_const_te, wte_raw, "test"),
                ):
                    b = T.metrics_block(yt, pt, wt)
                    base_w[(mname, split)] = b["w_rmse"]
                    bb = dict(b)
                    bb.update({"kappa": k, "model": mname, "direction": direction, "block": block})
                    metrics_rows.append(bb)
                for block, yt, pt, wt, split in (
                    ("model_validation", yva_, p_va[mname], wva_raw, "validation"),
                    ("model_test", yte, p_te[mname], wte_raw, "test"),
                ):
                    b = T.metrics_block(yt, pt, wt)
                    rel = _rel_improve(base_w[(mname, split)], b["w_rmse"])
                    b["rel_wrmse_improvement"] = rel
                    bb = dict(b)
                    bb.update({"kappa": k, "model": mname, "direction": direction, "block": block})
                    metrics_rows.append(bb)
            # decile diagnostics (pooled test) per model
            for mname in MODELS:
                ds, spread, rho, viol = _decile_diag(
                    yte, p_te[mname], wte_raw, ep_te)
                dd = ds.copy()
                dd.insert(0, "kappa", k)
                dd.insert(1, "model", mname)
                dd.insert(2, "direction", direction)
                dd["spread"] = spread
                dd["rho_D"] = rho
                dd["monotonicity_violations"] = viol
                decile_frames.append(dd)
            # per-symbol metrics + comparisons
            for sym in symbols:
                m_sym = te & (sym_arr == sym)
                if not m_sym.any():
                    continue
                within = (sym_arr[te] == sym)
                for mname in MODELS:
                    ps = p_te[mname][within]
                    base_pred = (base_const_te[within] if mname == "M0"
                                 else p_te["M0"][within])
                    bw = _wrmse(yte[within], base_pred, wte_raw[within])
                    mw = _wrmse(yte[within], ps, wte_raw[within])
                    dd = T.decile_table(yte[within], ps, wte_raw[within], ep_te[within])
                    dds = dd.sort_values("decile")
                    top = float(dds.iloc[-1]["weighted_actual_mean_y"]) if len(dds) else float("nan")
                    bot = float(dds.iloc[0]["weighted_actual_mean_y"]) if len(dds) else float("nan")
                    tbr = float(top - bot) if len(dds) >= 2 else float("nan")
                    tpr = float(dds.iloc[-1]["positive_y_rate"]) if len(dds) else float("nan")
                    symbol_metric_rows.append({
                        "kappa": k, "symbol": sym, "model": mname, "direction": direction,
                        "rows": int(m_sym.sum()),
                        "episodes": int(pred.loc[m_sym, "global_episode"].nunique()),
                        "baseline_wRMSE": bw, "model_wRMSE": mw,
                        "rel_wrmse_improvement": _rel_improve(bw, mw),
                        "spearman": _spear(yte[within], ps),
                        "top_decile_weighted_Y": top, "bottom_decile_weighted_Y": bot,
                        "top_bottom_spread": tbr, "top_decile_positive_Y_rate": tpr,
                        "positive_target_fraction": float((yte[within] > 0).mean()),
                    })

        # comparison stats (pooled + per-symbol, Test)
        cstats = _comparison_stats(y, predictions, w_raw, symbols, sym_arr, te)
        comparison_stats_all[k] = cstats

        # ---- bootstrap (1000x episode resample, no refit) ----
        for direction, _ in (("long", None), ("short", None)):
            yte = y[direction][te]
            ep_te_arr = ep_te
            uniq, inv = np.unique(ep_te_arr, return_inverse=True)
            n_ep = len(uniq)
            groups = [np.where(inv == j)[0] for j in range(n_ep)]
            rng = np.random.default_rng(T.SEED)
            for comp, base_name, model_name in COMPARISONS:
                base_pred = (np.full(len(yte), ybar[direction]) if base_name is None
                             else pred_p_te[direction][base_name])
                model_pred = pred_p_te[direction][model_name]
                deltas = []
                for _ in range(1000):
                    samp = rng.integers(0, n_ep, size=n_ep)
                    idx = np.concatenate([groups[j] for j in samp])
                    if len(idx) == 0:
                        continue
                    deltas.append(
                        _wrmse(yte[idx], base_pred[idx], wte_raw[idx])
                        - _wrmse(yte[idx], model_pred[idx], wte_raw[idx]))
                deltas = np.asarray(deltas, float)
                lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
                bootstrap_rows.append({
                    "kappa": k, "comparison": comp, "direction": direction,
                    "n_reps": len(deltas),
                    "delta_mean": float(np.mean(deltas)),
                    "delta_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                    "ci_lower": lo, "ci_upper": hi,
                    "ci_excludes_zero": bool(lo > 0 or hi < 0),
                })

        # ---- target stability vs kappa=0 ----
        if ref is not None and k != 0.0:
            for direction, ycol in (("long", "Y_L"), ("short", "Y_S")):
                yk = joined_by_kappa[k][ycol].to_numpy(float)
                y0 = ref[ycol]
                both = np.isfinite(yk) & np.isfinite(y0)
                yk_b = yk[both]
                y0_b = y0[both]
                if both.sum() == 0:
                    continue
                sign_agree = float(np.mean(np.sign(yk_b) == np.sign(y0_b)))
                pos0 = y0_b > 0
                flip = (float(np.mean((yk_b <= 0)[pos0])) if pos0.sum() else float("nan"))
                # top-decile overlap by value
                def _top_decile_overlap(a, b):
                    da = pd.Series(a).rank(pct=True).to_numpy()
                    db = pd.Series(b).rank(pct=True).to_numpy()
                    top_a = da >= 0.9
                    top_b = db >= 0.9
                    return float(np.mean(top_a & top_b)) if top_a.sum() else float("nan")
                target_stability_rows.append({
                    "kappa": k, "direction": direction,
                    "pearson": _pear(y0_b, yk_b),
                    "spearman": _spear(y0_b, yk_b),
                    "sign_agreement": sign_agree,
                    "pos_to_nonpos_flip_rate": flip,
                    "top_decile_overlap": _top_decile_overlap(y0_b, yk_b),
                    "n_compared": int(both.sum()),
                })

        # ---- curve row (per kappa) ----
        cs_delta = model_result[k]["cs_delta"]
        curve_rows.append({
            "kappa": k,
            "DeltaSTRUCT_long": cstats[("long", "M1-M0")]["pooled_test_delta_wrmse"],
            "DeltaSTRUCT_short": cstats[("short", "M1-M0")]["pooled_test_delta_wrmse"],
            "DeltaSTRUCT_unseen_long": cs_delta.get("long/M1-M0"),
            "DeltaSTRUCT_unseen_short": cs_delta.get("short/M1-M0"),
            "bootstrap_ci_lower_long": next(
                b["ci_lower"] for b in bootstrap_rows
                if b["kappa"] == k and b["direction"] == "long" and b["comparison"] == "M1-M0"),
            "bootstrap_ci_upper_long": next(
                b["ci_upper"] for b in bootstrap_rows
                if b["kappa"] == k and b["direction"] == "long" and b["comparison"] == "M1-M0"),
            "symbols_improved_long": cstats[("long", "M1-M0")]["symbols_improved"],
            "symbols_improved_short": cstats[("short", "M1-M0")]["symbols_improved"],
            "P_max_YL_YS_gt_0": next(r["P_max_YL_YS_gt_0"] for r in label_rows if r["kappa"] == k),
            "P_YL_gt_0": next(r["P_YL_gt_0"] for r in label_rows if r["kappa"] == k),
            "P_YS_gt_0": next(r["P_YS_gt_0"] for r in label_rows if r["kappa"] == k),
        })

    pd.DataFrame(metrics_rows).to_csv(OUT_DIR / "cost_metrics.csv", index=False)
    pd.concat(decile_frames, ignore_index=True).to_csv(OUT_DIR / "cost_deciles.csv", index=False)
    pd.DataFrame(symbol_metric_rows).to_csv(OUT_DIR / "cost_symbol_metrics.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(OUT_DIR / "cost_bootstrap.csv", index=False)
    pd.DataFrame(target_stability_rows).to_csv(OUT_DIR / "cost_target_stability.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(OUT_DIR / "cost_curve.csv", index=False)

    # ===================================================================== #
    # cross-symbol metrics csv (per kappa) — combine the per-kappa frames
    # ===================================================================== #
    cross_frames: List[pd.DataFrame] = []
    for k in kappas:
        cdf = pd.read_parquet(model_result[k]["cross_path"])
        cdf.insert(0, "kappa", k)
        cross_frames.append(cdf)
    pd.concat(cross_frames, ignore_index=True).to_csv(
        OUT_DIR / "cost_cross_symbol_metrics.csv", index=False)

    # ===================================================================== #
    # summary json
    # ===================================================================== #
    lgb = T.check_lightgbm()
    summary: Dict[str, Any] = {
        "task_id": TASK_ID,
        "parent_task_id": C.TASK_ID,
        "dataset": str(DATASET_PATH),
        "kappas": list(KAPPAS),
        "core_friction_max": CORE_FRICTION_MAX,
        "stress_kappa": STRESS_KAPPA,
        "models": list(MODELS),
        "comparisons": [c[0] for c in COMPARISONS],
        "lightgbm_version": lgb,
        "seed": T.SEED,
        "base_params": dict(T.BASE_PARAMS),
        "early_stopping_rounds": T.EARLY_STOPPING_ROUNDS,
        "symbols": list(SYMBOLS_15),
        "n_symbols": len(SYMBOLS_15),
        "join_audit": None,  # filled by caller
        "primary_statistic": "DeltaSTRUCT(kappa) = WRMSE(M0) - WRMSE(M1) on Test",
        "delta_struct": {},
        "rubric_inputs": {},
    }
    # delta struct per kappa/direction
    for k in kappas:
        cs = comparison_stats_all[k]
        summary["delta_struct"][str(k)] = {
            "long": {
                "pooled_test_delta_wrmse": cs[("long", "M1-M0")]["pooled_test_delta_wrmse"],
                "symbols_improved": cs[("long", "M1-M0")]["symbols_improved"],
                "median_symbol_delta": cs[("long", "M1-M0")]["median_symbol_delta"],
            },
            "short": {
                "pooled_test_delta_wrmse": cs[("short", "M1-M0")]["pooled_test_delta_wrmse"],
                "symbols_improved": cs[("short", "M1-M0")]["symbols_improved"],
                "median_symbol_delta": cs[("short", "M1-M0")]["median_symbol_delta"],
            },
        }
    # rubric inputs (reviewer assigns final verdict)
    for k in kappas:
        boot = {b["direction"]: b for b in bootstrap_rows if b["kappa"] == k}
        csd = model_result[k]["cs_delta"]
        rubric: Dict[str, Any] = {}
        for direction in ("long", "short"):
            b = boot.get(direction, {})
            cs = comparison_stats_all[k][(direction, "M1-M0")]
            rubric[direction] = {
                "bootstrap_ci_lower": b.get("ci_lower"),
                "bootstrap_ci_upper": b.get("ci_upper"),
                "ci_excludes_zero": b.get("ci_excludes_zero"),
                "unseen_symbol_deltaSTRUCT": csd.get(f"{direction}/M1-M0"),
                "symbols_improved": cs["symbols_improved"],
                "n_symbols": cs["n_symbols"],
                "meets_rubric": (
                    bool(b.get("ci_lower", -1) > 0)
                    and (csd.get(f"{direction}/M1-M0") is not None
                         and csd.get(f"{direction}/M1-M0") > 0)
                    and cs["symbols_improved"] >= 10
                ),
            }
        summary["rubric_inputs"][str(k)] = rubric

    (OUT_DIR / "cost_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run_phase2(
    symbols: Optional[List[str]] = None,
    kappas: Optional[List[float]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    syms = list(symbols) if symbols else list(SYMBOLS_15)
    kapps = list(kappas) if kappas else list(KAPPAS)
    print(f"[phase2] symbols={syms} kappas={kapps} force={force}", flush=True)

    t2 = pd.read_parquet(DATASET_PATH)
    # Restrict the frozen dataset to the requested universe. For the full run this
    # is all 15 symbols (a no-op); for a smoke run it keeps targets self-consistent.
    if set(syms) != set(SYMBOLS_15):
        t2 = t2[t2["symbol"].isin(syms)].reset_index(drop=True)
    cost_by_sym = compute_cost_labels(syms, force=force)
    joined, audit = join_and_audit(t2, cost_by_sym)
    # restrict joined to requested kappas
    joined = {k: joined[k] for k in kapps}
    model_result = run_modeling(joined, syms, kapps, force=force)

    summary = write_artifacts(joined, cost_by_sym, model_result, syms, kapps)
    summary["join_audit"] = audit
    (OUT_DIR / "cost_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("[phase2] DONE", flush=True)
    return summary


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="run on AU,RB with kappa 0.0,0.02 only (validation)")
    ap.add_argument("--force", action="store_true", help="ignore caches")
    args = ap.parse_args()
    if args.smoke:
        run_phase2(symbols=["AU", "RB"], kappas=[0.0, 0.02], force=args.force)
    else:
        run_phase2(force=args.force)


if __name__ == "__main__":
    _main()
