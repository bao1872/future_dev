"""FUTURE-ENTRY-ACTION3-STRUCT44-V1 — three-action (S/F/L) scoring formulation.

Scientific question
-------------------
The previous dual-regression formulation learned X_t -> (Y_L, Y_S) and then
constructed a trade decision afterward. This experiment asks whether the SAME
frozen STRUCT44 features perform materially better when the model is trained
directly on the joint three-action decision SHORT / WAIT / LONG derived from
the DP Oracle.

It is designed to separate two failure modes:

    ACTION3-M1 >> REG-M1   -> the previous modeling objective was wrong
    ACTION3-M1 >> ACTION3-M0 -> STRUCT44 structure adds value under the
                                correct decision objective

Discipline (frozen)
-------------------
* Features are NOT changed. M0 = DTP12, M1 = STRUCT44 only.
* The Oracle is NOT called. Cost labels come from the existing Phase-2 cache
  (cost_{sym}.parquet), joined exactly like Phase-2 (join_and_audit).
* The DP `best_F1` (cache encoding -1/0/+1 = SHORT/FLAT/LONG) is the
  authoritative action label; it is mapped to {0,1,2} via +1.
* Validation is used only for early stopping; Test never enters fit/selection.
* No hyperparameter tuning, no class balancing, no SMOTE.

Primary metrics (NOT classification accuracy)
--------------------------------------------
* Regret            = U* - U[chosen],  U* = max(Y_S, 0, Y_L)
* First non-WAIT per episode entry value (causal execution proxy)
* Coverage-adjusted entry value (CAEV) = sum(I_e * Y_e) / N_all_episodes

Outputs (small, committed): artifacts/entry_action3_scoring_v1/*.csv + .json
Models and large prediction parquets stay local (cache/) and untracked.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import order mirrors Phase-2 (B first pulls t15 which imports lightgbm before
# numpy/pandas to avoid an OpenMP dlopen segfault).
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import lightgbm as lgb  # noqa: E402
import research.liquidity_oracle_atlas.experiment_entry_value_tree_core108_t15b_aligned_v1 as B  # noqa: E402
import research.liquidity_oracle_atlas.experiment_entry_value_cost_robustness_phase2_formal_v1 as P2  # noqa: E402

TASK_ID = "FUTURE-ENTRY-ACTION3-STRUCT44-V1"
BASE_SHA = "0d003d0660171772f715d6610602ecc0d361ffc2"
BRANCH = "entry-value-cost-robustness-v1"

DATASET_PATH = P2.DATASET_PATH
COST_DIR = P2.CACHE_DIR
KAPPAS: Tuple[float, ...] = P2.KAPPAS
SYMBOLS_15: List[str] = list(P2.SYMBOLS_15)

ACTION_SHORT = 0
ACTION_FLAT = 1
ACTION_LONG = 2

ACTION3_PARAMS = dict(
    objective="multiclass",
    num_class=3,
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
SEED = 20260921
BOOTSTRAP_N = 1000

OUT_DIR = Path("artifacts/entry_action3_scoring_v1")
CACHE_DIR = OUT_DIR / "cache"

# frozen 5-fold leave-3-symbols-out (matches Phase-2)
_FOLD_OF = {s: i % 5 for i, s in enumerate(sorted(SYMBOLS_15))}


# --------------------------------------------------------------------------- #
# data loading / join (reuse Phase-2 exactly)
# --------------------------------------------------------------------------- #
def load_joined() -> Tuple[Dict[float, pd.DataFrame], Dict[str, Any]]:
    """Reuse Phase-2 cost caches (no Oracle call) and join semantics."""
    cost_by_sym: Dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS_15:
        p = COST_DIR / f"cost_{sym}.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"STOP_ACTION3_COST_CACHE_MISSING: {p}")
        cost_by_sym[sym] = pd.read_parquet(p)
    t2 = pd.read_parquet(DATASET_PATH)
    joined, audit = P2.join_and_audit(t2, cost_by_sym)
    return joined, audit


def load_reg_predictions(kappa: float) -> pd.DataFrame:
    p = COST_DIR / f"pred_{kappa:.4f}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"STOP_ACTION3_PRED_CACHE_MISSING: {p}")
    cols = ["symbol", "decision_time"] + [
        f"{pre}_{d}_{m}" for pre in ("p_va", "p_te")
        for d in ("long", "short") for m in ("M0", "M1")
    ]
    return pd.read_parquet(p, columns=cols)


def analysis_frame(kappa: float,
                   joined_by_kappa: Dict[float, pd.DataFrame]
                   ) -> pd.DataFrame:
    """Merge Phase-2 REG predictions onto the joined frame; keep candidates."""
    df = joined_by_kappa[kappa].copy()
    reg = load_reg_predictions(kappa)
    df = df.merge(reg, on=["symbol", "decision_time"], how="left")
    cand = df["is_candidate_cost"].to_numpy(bool) & df["Y_L"].notna().to_numpy()
    return df[cand].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# action targets / utilities
# --------------------------------------------------------------------------- #
def build_action_targets(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Authoritative DP action label + utility + margin.

    Returns (action, utility, margin). action is {0,1,2} SHORT/FLAT/LONG
    mapped from the cache's {-1,0,+1} best_F1. The utility audit only checks
    unique-margin rows (ties are decided by the DP, not raw argmax).
    """
    ys = df["Y_S"].to_numpy(float)
    yl = df["Y_L"].to_numpy(float)
    utility = np.column_stack([ys, np.zeros(len(df), dtype=float), yl])

    best_raw = df["best_F1_cost"].to_numpy(float)
    if not np.isin(best_raw, [-1.0, 0.0, 1.0]).all():
        raise RuntimeError("STOP_ACTION3_INVALID_BEST_F1")
    # cache -1/0/+1 (SHORT/FLAT/LONG) -> 0/1/2
    action = best_raw.astype(int) + 1

    argmax_u = np.argmax(utility, axis=1)
    sorted_u = np.sort(utility, axis=1)
    margin = sorted_u[:, -1] - sorted_u[:, -2]
    unique = margin > 1e-12
    if not np.array_equal(action[unique], argmax_u[unique]):
        raise RuntimeError("STOP_ACTION3_DP_UTILITY_MISMATCH")
    return action, utility, margin


def action_utilities(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        df["Y_S"].to_numpy(float),
        np.zeros(len(df), dtype=float),
        df["Y_L"].to_numpy(float),
    ])


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def _masks(df: pd.DataFrame):
    tr = (df["split"].to_numpy() == "train")
    va = (df["split"].to_numpy() == "validation")
    te = (df["split"].to_numpy() == "test")
    return tr, va, te


def margin_weights(base_w: np.ndarray, margin: np.ndarray) -> np.ndarray:
    m = np.asarray(margin, float)
    factor = m / (1.0 + m)
    w = np.asarray(base_w, float) * factor
    positive = w > 0
    if not positive.any():
        raise RuntimeError("STOP_ACTION3_MARGIN_EMPTY")
    w = w / np.mean(w[positive])
    return w


def fit_action3(df: pd.DataFrame, cols: List[str], use_margin: bool = False,
                params: Optional[dict] = None,
                tr_mask: Optional[np.ndarray] = None,
                va_mask: Optional[np.ndarray] = None,
                te_mask: Optional[np.ndarray] = None):
    params = dict(params or ACTION3_PARAMS)
    if tr_mask is not None and va_mask is not None and te_mask is not None:
        tr, va, te = tr_mask, va_mask, te_mask
    else:
        tr, va, te = _masks(df)
    Xtr = B.build_X_subset(df[tr], cols)
    Xva = B.build_X_subset(df[va], cols)
    Xte = B.build_X_subset(df[te], cols)
    ytr = (df["best_F1_cost"].to_numpy(float)[tr]).astype(int) + 1
    yva = (df["best_F1_cost"].to_numpy(float)[va]).astype(int) + 1
    w = df["w_norm"].to_numpy(float)
    wtr = w[tr].copy()
    wva = w[va].copy()
    if use_margin:
        _, _, margin = build_action_targets(df[tr])
        wtr = margin_weights(w[tr], margin)
        _, _, margin_va = build_action_targets(df[va])
        wva = margin_weights(w[va], margin_va)
    model = lgb.LGBMClassifier(**params)
    model.fit(
        Xtr, ytr,
        sample_weight=np.asarray(wtr, float),
        eval_set=[(Xva, yva)],
        eval_sample_weight=[np.asarray(wva, float)],
        eval_metric="multi_logloss",
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return model, Xte


def predict_action3(model, X) -> Dict[str, np.ndarray]:
    p = np.asarray(model.predict_proba(X), float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise RuntimeError("STOP_ACTION3_BAD_PROBA_SHAPE")
    if not np.isfinite(p).all():
        raise RuntimeError("STOP_ACTION3_NONFINITE_PROBA")
    pred = np.argmax(p, axis=1)
    ps = np.sort(p, axis=1)
    confidence = ps[:, -1] - ps[:, -2]
    return {"proba": p, "pred_action": pred, "confidence": confidence}


def reg_action(df: pd.DataFrame, mname: str, split: str) -> np.ndarray:
    if split == "validation":
        pre = "p_va"
    elif split == "test":
        pre = "p_te"
    else:
        raise ValueError(split)
    ps = df[f"{pre}_short_{mname}"].to_numpy(float)
    pl = df[f"{pre}_long_{mname}"].to_numpy(float)
    score = np.column_stack([ps, np.zeros(len(df), dtype=float), pl])
    return np.argmax(score, axis=1)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def row_metrics(df: pd.DataFrame, pred_action: np.ndarray) -> Dict[str, float]:
    u = action_utilities(df)
    idx = np.arange(len(df))
    chosen = u[idx, pred_action]
    oracle = np.max(u, axis=1)
    regret = oracle - chosen
    w = df["w_raw"].to_numpy(float)
    wsum = w.sum()
    def wmean(x):
        return float(np.sum(w * x) / wsum) if wsum > 0 else float("nan")
    trade = pred_action != ACTION_FLAT
    return {
        "weighted_action_accuracy": wmean(
            pred_action == (df["best_F1_cost"].to_numpy(float).astype(int) + 1)),
        "weighted_mean_chosen_utility": wmean(chosen),
        "weighted_mean_oracle_utility": wmean(oracle),
        "weighted_mean_regret": wmean(regret),
        "predicted_short_rate": float(np.mean(pred_action == ACTION_SHORT)),
        "predicted_wait_rate": float(np.mean(pred_action == ACTION_FLAT)),
        "predicted_long_rate": float(np.mean(pred_action == ACTION_LONG)),
        "trade_rows": int(trade.sum()),
        "mean_chosen_entry_value": (float(np.mean(chosen[trade]))
                                    if trade.any() else float("nan")),
        "positive_entry_value_rate": (float(np.mean(chosen[trade] > 0))
                                      if trade.any() else float("nan")),
    }


def first_nonwait_by_episode(df: pd.DataFrame,
                             pred_action: np.ndarray) -> pd.DataFrame:
    d = df[["symbol", "global_episode", "decision_time",
            "Y_L", "Y_S"]].copy()
    d["pred_action"] = pred_action
    trade = d[d["pred_action"] != ACTION_FLAT].copy()
    trade = (
        trade.sort_values(["symbol", "global_episode", "decision_time"])
        .drop_duplicates(["symbol", "global_episode"], keep="first")
    )
    assert not trade.duplicated(
        ["symbol", "global_episode"]).any(), "episode dup after first-nonwait"
    trade["entry_value"] = np.where(
        trade["pred_action"] == ACTION_LONG, trade["Y_L"], trade["Y_S"])
    return trade


def episode_metrics(trade: pd.DataFrame,
                    total_test_episodes: int) -> Dict[str, float]:
    sel = len(trade)
    ev = trade["entry_value"].to_numpy(float)
    return {
        "total_test_episodes": total_test_episodes,
        "selected_episodes": sel,
        "episode_trade_coverage": (sel / total_test_episodes
                                   if total_test_episodes else float("nan")),
        "selected_mean_entry_value": (float(np.mean(ev)) if sel else float("nan")),
        "selected_median_entry_value": (float(np.median(ev)) if sel else float("nan")),
        "selected_positive_rate": (float(np.mean(ev > 0)) if sel else float("nan")),
        "selected_long_fraction": (float(np.mean(
            trade["pred_action"].to_numpy() == ACTION_LONG)) if sel else float("nan")),
        "selected_short_fraction": (float(np.mean(
            trade["pred_action"].to_numpy() == ACTION_SHORT)) if sel else float("nan")),
        "selected_symbols": int(trade["symbol"].nunique()),
        "caev": (float(np.sum(ev) / total_test_episodes)
                 if total_test_episodes else float("nan")),
    }


def episode_value_table(df: pd.DataFrame,
                        pred_action: np.ndarray) -> pd.DataFrame:
    """One row per test episode with entry value (0 if no trade)."""
    total = df[["symbol", "global_episode"]].drop_duplicates().reset_index(drop=True)
    trade = first_nonwait_by_episode(df, pred_action)
    val = trade[["symbol", "global_episode", "entry_value"]]
    merged = total.merge(val, on=["symbol", "global_episode"], how="left")
    merged["value"] = merged["entry_value"].fillna(0.0)
    return merged


def bootstrap_mean(values: np.ndarray, n: int = BOOTSTRAP_N,
                   seed: int = SEED) -> Tuple[float, float]:
    values = np.asarray(values, float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n, dtype=float)
    for i in range(n):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def bootstrap_episode(ep_value: np.ndarray, n: int = BOOTSTRAP_N,
                     seed: int = SEED):
    """CI for CAEV (resample episodes)."""
    return bootstrap_mean(ep_value, n, seed)


def paired_bootstrap(epA: np.ndarray, epB: np.ndarray, n: int = BOOTSTRAP_N,
                     seed: int = SEED) -> Dict[str, float]:
    epA = np.asarray(epA, float)
    epB = np.asarray(epB, float)
    if len(epA) != len(epB) or len(epA) == 0:
        return {"delta_mean": float("nan"), "ci_lower": float("nan"),
                "ci_upper": float("nan")}
    rng = np.random.default_rng(seed)
    deltas = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.integers(0, len(epA), size=len(epA))
        deltas[i] = epA[idx].mean() - epB[idx].mean()
    return {
        "delta_mean": float(np.mean(deltas)),
        "ci_lower": float(np.quantile(deltas, 0.025)),
        "ci_upper": float(np.quantile(deltas, 0.975)),
    }


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run_temporal(force: bool = False, smoke: bool = False
                 ) -> Dict[str, Any]:
    joined, audit = load_joined()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    params = dict(ACTION3_PARAMS)
    if smoke:
        params = dict(params)
        params["n_estimators"] = 20
        params["random_state"] = SEED

    subsets = B.feature_subsets()
    kappas = [0.0] if smoke else list(KAPPAS)

    row_rows: List[dict] = []
    epi_rows: List[dict] = []
    sym_rows: List[dict] = []
    boot_rows: List[dict] = []
    cost_rows: List[dict] = []
    label_rows: List[dict] = []

    # per (kappa, model) storage for paired bootstrap
    ep_value_store: Dict[Tuple[float, str], np.ndarray] = {}
    sel_value_store: Dict[Tuple[float, str], np.ndarray] = {}

    for k in kappas:
        df = analysis_frame(k, joined)
        action, _, _ = build_action_targets(df)
        total_test_eps = int(df[df["split"] == "test"][
            ["symbol", "global_episode"]].drop_duplicates().shape[0])
        # label stats
        label_rows.append({
            "kappa": k,
            "candidate_rows": len(df),
            "train_rows": int((df["split"] == "train").sum()),
            "validation_rows": int((df["split"] == "validation").sum()),
            "test_rows": int((df["split"] == "test").sum()),
            "n_short": int((action == ACTION_SHORT).sum()),
            "n_flat": int((action == ACTION_FLAT).sum()),
            "n_long": int((action == ACTION_LONG).sum()),
            "total_test_episodes": total_test_eps,
        })

        models = ["REG-M0", "REG-M1", "ACTION3-M0", "ACTION3-M1"]
        if abs(k) < 1e-12:
            models = models + ["ACTION3-M1-MARGIN"]

        preds: Dict[str, np.ndarray] = {}
        for mname in models:
            df_te = df[df["split"] == "test"].reset_index(drop=True)
            if mname.startswith("REG"):
                m = mname.split("-")[1]  # M0 / M1
                pa = reg_action(df_te, m, "test")
            elif mname == "ACTION3-M1-MARGIN":
                model, Xte = fit_action3(
                    df, subsets["M1"], use_margin=True, params=params)
                pa = predict_action3(model, Xte)["pred_action"]
            else:  # ACTION3-M0 / ACTION3-M1
                mkey = "M0" if mname == "ACTION3-M0" else "M1"
                model, Xte = fit_action3(
                    df, subsets[mkey], use_margin=False, params=params)
                pa = predict_action3(model, Xte)["pred_action"]
            preds[mname] = pa

            rm = row_metrics(df_te, pa)
            row_rows.append({"kappa": k, "model": mname, **rm})

            trade = first_nonwait_by_episode(df_te, pa)
            em = episode_metrics(trade, total_test_eps)
            epi_rows.append({"kappa": k, "model": mname, **em})

            # per-symbol (pa is aligned to df_te positions; align to group rows)
            for sym, g in df_te.groupby("symbol"):
                gtrade = first_nonwait_by_episode(g, pa[g.index.to_numpy()])
                gem = episode_metrics(gtrade, int(g[
                    ["symbol", "global_episode"]].drop_duplicates().shape[0]))
                w = g["w_raw"].to_numpy(float)
                u = action_utilities(g)
                pa_g = pa[g.index.to_numpy()]
                chosen = u[np.arange(len(g)), pa_g]
                sym_rows.append({
                    "kappa": k, "model": mname, "symbol": sym,
                    "test_episodes": gem["total_test_episodes"],
                    "selected_episodes": gem["selected_episodes"],
                    "trade_coverage": gem["episode_trade_coverage"],
                    "selected_mean_entry_value": gem["selected_mean_entry_value"],
                    "positive_rate": gem["selected_positive_rate"],
                    "caev": gem["caev"],
                    "weighted_row_regret": float(
                        np.sum(w * (np.max(u, axis=1) - chosen)) / w.sum()),
                })

            # bootstrap
            ep_tab = episode_value_table(df_te, pa)
            ep_value_store[(k, mname)] = ep_tab["value"].to_numpy(float)
            sel_val = trade["entry_value"].to_numpy(float)
            sel_value_store[(k, mname)] = sel_val
            caev_ci = bootstrap_episode(ep_tab["value"].to_numpy(float))
            sel_ci = bootstrap_mean(sel_val)
            boot_rows.append({
                "kappa": k, "model": mname,
                "selected_mean_entry_value": em["selected_mean_entry_value"],
                "sel_ci_lower": sel_ci[0], "sel_ci_upper": sel_ci[1],
                "caev": em["caev"],
                "caev_ci_lower": caev_ci[0], "caev_ci_upper": caev_ci[1],
            })

            cost_rows.append({
                "kappa": k, "model": mname,
                "weighted_mean_regret": rm["weighted_mean_regret"],
                "weighted_action_accuracy": rm["weighted_action_accuracy"],
                "episode_trade_coverage": em["episode_trade_coverage"],
                "selected_mean_entry_value": em["selected_mean_entry_value"],
                "selected_positive_rate": em["selected_positive_rate"],
                "caev": em["caev"],
            })

        # paired bootstrap (primary comparisons) at this kappa
        for base in ("REG-M1", "ACTION3-M0"):
            pb = paired_bootstrap(
                ep_value_store[(k, "ACTION3-M1")], ep_value_store[(k, base)])
            boot_rows.append({
                "kappa": k, "model": f"ACTION3-M1_minus_{base}",
                "delta_mean": pb["delta_mean"],
                "ci_lower": pb["ci_lower"], "ci_upper": pb["ci_upper"],
            })

    results = {
        "audit": audit,
        "row_rows": row_rows,
        "epi_rows": epi_rows,
        "sym_rows": sym_rows,
        "boot_rows": boot_rows,
        "cost_rows": cost_rows,
        "label_rows": label_rows,
    }
    return results


def run_cross_symbol(force: bool = False, smoke: bool = False) -> Dict[str, Any]:
    joined, _ = load_joined()
    params = dict(ACTION3_PARAMS)
    if smoke:
        params = dict(params)
        params["n_estimators"] = 20
    subsets = B.feature_subsets()
    df = analysis_frame(0.0, joined)
    sym_arr = df["symbol"].to_numpy()
    tr = (df["split"].to_numpy() == "train")
    va = (df["split"].to_numpy() == "validation")
    te = (df["split"].to_numpy() == "test")

    folds = {f: [s for s in sorted(SYMBOLS_15) if _FOLD_OF[s] == f]
             for f in range(5)}
    rows: List[dict] = []
    per_sym: List[dict] = []

    fold_list = [0] if smoke else range(5)
    for f in fold_list:
        held = folds[f]
        train_syms = [s for s in sorted(SYMBOLS_15) if s not in held]
        trm = np.isin(sym_arr, train_syms) & tr
        vam = np.isin(sym_arr, train_syms) & va
        tem = np.isin(sym_arr, held) & te
        n_eval = int(tem.sum())
        total_eps = int(df[tem][
            ["symbol", "global_episode"]].drop_duplicates().shape[0])
        for mname in ("ACTION3-M0", "ACTION3-M1"):
            mkey = "M0" if mname == "ACTION3-M0" else "M1"
            model, Xte = fit_action3(
                df, subsets[mkey], use_margin=False, params=params,
                tr_mask=trm, va_mask=vam, te_mask=tem)
            pa = predict_action3(model, Xte)["pred_action"]  # aligned to df[tem]
            df_te = df[tem].reset_index(drop=True)
            rm = row_metrics(df_te, pa)
            trade = first_nonwait_by_episode(df_te, pa)
            em = episode_metrics(trade, total_eps)
            rows.append({
                "fold": f, "held_out_symbols": ",".join(held),
                "train_symbols": ",".join(train_syms),
                "n_train_rows": int(trm.sum()),
                "n_validation_rows": int(vam.sum()),
                "n_eval_rows": n_eval, "model": mname,
                "weighted_action_accuracy": rm["weighted_action_accuracy"],
                "weighted_mean_chosen_utility": rm["weighted_mean_chosen_utility"],
                "weighted_mean_regret": rm["weighted_mean_regret"],
                "episode_trade_coverage": em["episode_trade_coverage"],
                "selected_mean_entry_value": em["selected_mean_entry_value"],
                "caev": em["caev"],
            })
            for sym, g in df_te.groupby("symbol"):
                gtrade = first_nonwait_by_episode(g, pa[g.index.to_numpy()])
                gem = episode_metrics(gtrade, int(g[
                    ["symbol", "global_episode"]].drop_duplicates().shape[0]))
                w = g["w_raw"].to_numpy(float)
                u = action_utilities(g)
                pa_g = pa[g.index.to_numpy()]
                chosen = u[np.arange(len(g)), pa_g]
                per_sym.append({
                    "fold": f, "model": mname, "symbol": sym,
                    "test_episodes": gem["total_test_episodes"],
                    "selected_episodes": gem["selected_episodes"],
                    "trade_coverage": gem["episode_trade_coverage"],
                    "selected_mean_entry_value": gem["selected_mean_entry_value"],
                    "positive_rate": gem["selected_positive_rate"],
                    "caev": gem["caev"],
                    "weighted_row_regret": float(
                        np.sum(w * (np.max(u, axis=1) - chosen)) / w.sum()),
                })
    return {"rows": rows, "per_symbol": per_sym}


def confidence_deciles(df_test_full: pd.DataFrame,
                       pred_action: np.ndarray,
                       proba: np.ndarray) -> List[dict]:
    ps = np.sort(proba, axis=1)
    conf = ps[:, -1] - ps[:, -2]
    d = df_test_full.copy()
    d["pred_action"] = pred_action
    d["confidence"] = conf
    d["decile"] = pd.qcut(conf, 10, labels=False, duplicates="drop")
    out = []
    for dec, g in d.groupby("decile"):
        u = action_utilities(g)
        chosen = u[np.arange(len(g)), g["pred_action"].to_numpy()]
        w = g["w_raw"].to_numpy(float)
        wsum = w.sum()
        oracle = np.max(u, axis=1)
        out.append({
            "decile": int(dec),
            "rows": len(g),
            "weighted_action_accuracy": float(np.sum(
                w * (g["pred_action"].to_numpy()
                     == (g["best_F1_cost"].to_numpy(float).astype(int) + 1)))
                / wsum),
            "mean_chosen_utility": float(np.sum(w * chosen) / wsum),
            "mean_regret": float(np.sum(w * (oracle - chosen)) / wsum),
            "trade_rate": float(np.mean(g["pred_action"].to_numpy() != ACTION_FLAT)),
        })
    return out


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #
def write_artifacts(results: Dict[str, Any],
                    cross: Dict[str, Any],
                    conf: List[dict],
                    hard_checks: Dict[str, Any],
                    lgb_version: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results["label_rows"]).to_csv(
        OUT_DIR / "action3_label_stats.csv", index=False)
    pd.DataFrame(results["row_rows"]).to_csv(
        OUT_DIR / "action3_row_metrics.csv", index=False)
    pd.DataFrame(results["epi_rows"]).to_csv(
        OUT_DIR / "action3_episode_metrics.csv", index=False)
    pd.DataFrame(results["sym_rows"]).to_csv(
        OUT_DIR / "action3_symbol_metrics.csv", index=False)
    pd.DataFrame(cross["rows"]).to_csv(
        OUT_DIR / "action3_cross_symbol.csv", index=False)
    if cross["per_symbol"]:
        pd.DataFrame(cross["per_symbol"]).to_csv(
            OUT_DIR / "action3_cross_symbol_symbol.csv", index=False)
    pd.DataFrame(conf).to_csv(
        OUT_DIR / "action3_confidence_deciles.csv", index=False)
    pd.DataFrame(results["cost_rows"]).to_csv(
        OUT_DIR / "action3_cost_curve.csv", index=False)
    pd.DataFrame(results["boot_rows"]).to_csv(
        OUT_DIR / "action3_bootstrap.csv", index=False)

    # primary comparisons at kappa=0
    sym_df = pd.DataFrame(results["sym_rows"])
    verdict_inputs = {}
    for k in (results["label_rows"][0]["kappa"],):
        pass
    kappas_present = sorted({r["kappa"] for r in results["row_rows"]})
    comparisons = {}
    for k in kappas_present:
        a3 = {r["symbol"]: r["caev"]
              for r in sym_df[(sym_df["kappa"] == k) &
                             (sym_df["model"] == "ACTION3-M1")].to_dict("records")}
        rm1 = {r["symbol"]: r["caev"]
               for r in sym_df[(sym_df["kappa"] == k) &
                              (sym_df["model"] == "REG-M1")].to_dict("records")}
        am0 = {r["symbol"]: r["caev"]
               for r in sym_df[(sym_df["kappa"] == k) &
                              (sym_df["model"] == "ACTION3-M0")].to_dict("records")}
        syms = sorted(set(a3) | set(rm1) | set(am0))
        better_reg = sum(1 for s in syms if a3.get(s, 0) > rm1.get(s, 0))
        better_m0 = sum(1 for s in syms if a3.get(s, 0) > am0.get(s, 0))
        deltas_reg = [a3.get(s, 0) - rm1.get(s, 0) for s in syms]
        deltas_m0 = [a3.get(s, 0) - am0.get(s, 0) for s in syms]
        comparisons[str(k)] = {
            "symbols_action3_better_than_reg": better_reg,
            "symbols_action3_better_than_m0": better_m0,
            "n_symbols": len(syms),
            "median_symbol_delta_vs_reg": float(np.median(deltas_reg)) if deltas_reg else float("nan"),
            "median_symbol_delta_vs_m0": float(np.median(deltas_m0)) if deltas_m0 else float("nan"),
        }

    # standalone entry-selector viability (gate C) at each kappa for ACTION3-M1
    gate_c = {}
    for k in kappas_present:
        epi = {r["model"]: r for r in results["epi_rows"]
               if r["kappa"] == k and r["model"] == "ACTION3-M1"}
        e = epi.get("ACTION3-M1")
        boot = {r["model"]: r for r in results["boot_rows"]
                if r["kappa"] == k and r["model"] == "ACTION3-M1"}
        b = boot.get("ACTION3-M1", {})
        if e is None:
            gate_c[str(k)] = {"selected_episodes": 0, "selected_mean_entry_value": None,
                              "caev_ci_lower": None}
        else:
            gate_c[str(k)] = {
                "selected_episodes": e["selected_episodes"],
                "selected_mean_entry_value": e["selected_mean_entry_value"],
                "caev": e["caev"],
                "caev_ci_lower": b.get("caev_ci_lower"),
                "caev_ci_upper": b.get("caev_ci_upper"),
            }

    summary = {
        "task_id": TASK_ID,
        "base_sha": BASE_SHA,
        "branch": BRANCH,
        "lightgbm_version": lgb_version,
        "kappas": list(kappas_present),
        "models": ["REG-M0", "REG-M1", "ACTION3-M0", "ACTION3-M1",
                   "ACTION3-M1-MARGIN"],
        "action_prior": "SHORT/FLAT/LONG from DP best_F1 (encoded -1/0/+1)",
        "features": {"M0": "DTP12", "M1": "STRUCT44"},
        "hard_checks": hard_checks,
        "join_audit": results["audit"],
        "gate_C_standalone_viability": gate_c,
        "primary_comparison_symbol_deltas": comparisons,
        "verdict_inputs": {
            "A_formulation_improvement": "ACTION3-M1 vs REG-M1: lower regret, "
                "higher CAEV, paired bootstrap Delta CAEV > 0 (CI lower > 0 preferred)",
            "B_struct44_incremental": "ACTION3-M1 vs ACTION3-M0: lower regret, "
                "higher CAEV, >=10/15 symbols improve",
            "C_standalone_viability": "ACTION3-M1 Test first-entry: episodes>=100, "
                "mean entry value>0, bootstrap CI lower>0",
        },
    }
    (OUT_DIR / "action3_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))


def run(force: bool = False, smoke: bool = False) -> None:
    t0 = time.perf_counter()
    lgb_version = lgb.__version__
    results = run_temporal(force=force, smoke=smoke)
    cross = run_cross_symbol(force=force, smoke=smoke)

    # confidence deciles for ACTION3-M1 at kappa=0
    joined, _ = load_joined()
    df0 = analysis_frame(0.0, joined)
    subsets = B.feature_subsets()
    model, Xte = fit_action3(df0, subsets["M1"], use_margin=False,
                             params=(dict(ACTION3_PARAMS) if not smoke
                                     else {**ACTION3_PARAMS, "n_estimators": 20}))
    pred = predict_action3(model, Xte)
    df_te = df0[df0["split"] == "test"].reset_index(drop=True)
    conf = confidence_deciles(df_te, pred["pred_action"], pred["proba"])

    hard_checks = {
        "all_cost_caches_present": True,
        "all_pred_caches_present": True,
        "no_oracle_call": True,
        "no_model_training_on_test": True,
        "best_F1_utility_audit_passed": True,
        "first_nonwait_one_row_per_episode": True,
        "all_action_labels_in_0_1_2": True,
        "feature_lists_m0_m1_only": True,
        "candidate_universe_kappa_independent": bool(
            results["audit"].get("hard_gate_universe_identical", False)),
    }
    write_artifacts(results, cross, conf, hard_checks, lgb_version)
    print(f"[action3] done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run(force=args.force, smoke=args.smoke)
