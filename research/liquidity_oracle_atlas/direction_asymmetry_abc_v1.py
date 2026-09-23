"""direction_asymmetry_abc_v1
===========================

FUTURE-R4-M15-DIRECTION-ASYMMETRY-ABC-V1

One-shot architecture-diagnostic A/B/C experiment answering whether LONG and SHORT
need *separate* decision functions.

**THIS IS AN ARCHITECTURE DIAGNOSTIC, not a pristine confirmation.** The asymmetry
hypothesis was motivated by already-observed TEST LONG/SHORT asymmetry (LONG recall
>> SHORT recall), therefore the current TEST result is NOT a pristine new confirmation.
If the experiment favors C, the next step is to freeze C and wait / build a fresh
*untouched* forward period before adopting C as the confirmed production research
baseline -- NOT to keep tuning on this TEST.

Models (DTP9 only, no new features, params frozen):
  A - Direct M0:        original DTP9 -> LONG/SHORT (mechanical reproduction of baseline)
  B - Shared Side Model: every Candidate yields two side proposals
                        (+X -> target=y, -X -> target=1-y); ONE shared classifier decides
                        whether the proposed side is correct. Inference: pL=f(X), pS=f(-X),
                        pred=LONG iff pL>=pS.
  C - Two Specialists:  LongExpert(+X, target=y) and ShortExpert(-X, target=1-y);
                        BOTH experts are trained on the FULL TRAIN population (not only the
                        true-LONG / true-SHORT rows). This is the only place the two
                        functions are allowed to differ. Inference: pL=fL(X), pS=fS(-X),
                        pred=LONG iff pL>=pS.

Primary contrast:  C - B  (specialist / asymmetry increment)
Secondary:         B - A  (side-normalization / representation gain)
                   C - A  (total architecture gain)
All comparisons are PAIRED at the Oracle-opportunity (trade) level.

Metric: TeacherFixedExitDirectionReturnATR -- a fixed-exit direction diagnostic,
explicitly NOT strategy PnL. BASE_PARAMS, the common calendar, snapped T1/T2, the
whole-opportunity boundary removal, and the frozen split are all reused verbatim.

No environment/Candidate/Teacher/dataset rebuild, no STRUCT33, no symbol feature, no
EntryQualityATR in X or as target, no LightGBM tuning, no threshold tuning.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from research.liquidity_oracle_atlas.train_direction_model_15sym_v1 import (
    BASE_PARAMS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DECISION_THRESHOLD,
    DTP9,
    FRAC_TRAIN,
    FRAC_VAL,
    SYMBOLS,
    aggregate_per_trade,
    bootstrap_trade_returns_chunked,
    fit_direction_model,
    phase_mask,
    pred_direction_return_atr,
    verify_manifest,
)
# Read-only reuse of the exact frozen split + the exact M0 fit/predict path so that
# model A reproduces the frozen evidence byte-for-byte (reproduction gate below).
from research.liquidity_oracle_atlas.direction_null_baseline_v1 import (
    build_frozen_split,
    _fit_m0,
    _predict_m0,
)

TASK_ID = "FUTURE-R4-M15-DIRECTION-ASYMMETRY-ABC-V1"
# The exact frozen baseline commit this experiment is built on (reviewer-given).
BASE_SHA = "a5c17964cbadc55a3368b09de630175c2683f3e1"

METRIC_NAME = "TeacherFixedExitDirectionReturnATR"
METRIC_DEFINITION = (
    "(+entry_quality_atr if predicted direction == Teacher direction else "
    "-entry_quality_atr), aggregated per Oracle opportunity (trade), bootstrapped "
    "at the trade level"
)
METRIC_NOT_LABELS = ["strategy PnL", "live alpha", "tradable return"]

EVIDENCE_DIR = os.path.join("research", "liquidity_oracle_atlas", "evidence")
SUMMARY_JSON = os.path.join(EVIDENCE_DIR, "direction_asymmetry_abc_v1_summary.json")
PER_SYMBOL_CSV = os.path.join(EVIDENCE_DIR, "direction_asymmetry_abc_v1_per_symbol.csv")
TRADE_RETURNS_CSV = os.path.join(EVIDENCE_DIR, "direction_asymmetry_abc_v1_trade_returns.csv")
LOSO_CSV = os.path.join(EVIDENCE_DIR, "direction_asymmetry_abc_v1_loso.csv")

# Frozen TEST shape (from the confirmation evidence) used as a drift guard.
A_REFERENCE = {
    "test_rows": 13773,
    "test_trades": 638,
    "m0_return_atr": 0.6014965284150177,
}

DTP_COLS = list(DTP9)


# --------------------------------------------------------------------------- #
# Data container (built ONCE; reused for pooled / per-symbol / LOSO)            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ABCData:
    X: np.ndarray          # [N, 9] float32
    X_neg: np.ndarray      # [N, 9] float32  (= -X, computed once)
    y: np.ndarray          # [N] uint8  (1=LONG, 0=SHORT)
    w: np.ndarray          # [N] float64 (sample_weight_raw)
    eq: np.ndarray         # [N] float64 (entry_quality_atr)
    symbol: np.ndarray     # [N] object
    trade_id: np.ndarray   # [N] object
    gid: np.ndarray        # [N] str (symbol::trade_id, collision-safe)


def build_abc_data(ds: pd.DataFrame) -> ABCData:
    X = ds.loc[:, DTP_COLS].to_numpy(dtype=np.float32, copy=True)
    X_neg = np.negative(X)  # computed exactly once
    y = (ds["oracle_direction"].to_numpy(object) == "LONG").astype(np.uint8)
    w = ds["sample_weight_raw"].to_numpy(np.float64)
    eq = ds["entry_quality_atr"].to_numpy(np.float64)
    symbol = ds["symbol"].to_numpy(object)
    trade_id = ds["oracle_trade_id"].to_numpy(object)
    gid = np.char.add(
        np.char.add(symbol.astype(str), "::"),
        trade_id.astype(str),
    )
    assert X.shape == (len(ds), len(DTP9)), X.shape
    return ABCData(X, X_neg, y, w, eq, symbol, trade_id, gid)


# --------------------------------------------------------------------------- #
# Vectorized side-transforms / shared-label construction                       #
# --------------------------------------------------------------------------- #
def make_shared_side_xy(X, X_neg, y, w, idx):
    """Shared side-correctness dataset (no row loops).

    First N:  proposed side = LONG,  features = +X, target = (Teacher is LONG)
    Second N: proposed side = SHORT, features = -X, target = (Teacher is SHORT)

    => target_long = y, target_short = 1 - y.
    """
    n = idx.size
    d = X.shape[1]
    X2 = np.empty((2 * n, d), dtype=np.float32)
    X2[:n] = X[idx]
    X2[n:] = X_neg[idx]
    y2 = np.empty(2 * n, dtype=np.uint8)
    y2[:n] = y[idx]
    y2[n:] = 1 - y[idx]
    w2 = np.empty(2 * n, dtype=np.float64)
    w2[:n] = w[idx]
    w2[n:] = w[idx]
    return X2, y2, w2


def combine_side_scores(p_long_side, p_short_side):
    """Map side-correctness scores to a direction prediction (NO threshold tuning).

    pred = LONG iff pL >= pS.  q_long = pL/(pL+pS) is only a ranking/AUC score.
    """
    pL = np.asarray(p_long_side, dtype=np.float64)
    pS = np.asarray(p_short_side, dtype=np.float64)
    denom = pL + pS
    q_long = np.divide(
        pL, denom, out=np.full_like(pL, 0.5), where=denom > 1e-12
    )
    pred = (pL >= pS).astype(np.uint8)
    return pred, q_long


# --------------------------------------------------------------------------- #
# Frozen fit helper (no copied LightGBM params)                                #
# --------------------------------------------------------------------------- #
def fit_frozen_binary(Xtr, ytr, wtr, Xv, yv, wv):
    wtr = np.asarray(wtr, dtype=np.float64)
    wtr_norm = wtr / wtr.mean() if wtr.mean() > 0 else wtr
    return fit_direction_model(Xtr, ytr, wtr_norm, Xv, yv, wv)


# --------------------------------------------------------------------------- #
# Model A / B / C fit & predict                                                #
# --------------------------------------------------------------------------- #
def fit_a(ds, train_idx, val_idx):
    # Exact frozen M0 reproduction path (byte-consistent with the confirmation evidence).
    return _fit_m0(ds, train_idx, val_idx)


def predict_a(model, ds, idx):
    pred_dir, p_long = _predict_m0(model, ds, idx)
    return pred_dir, p_long


def fit_b(arr: ABCData, train_idx, val_idx):
    Xtr, ytr, wtr = make_shared_side_xy(arr.X, arr.X_neg, arr.y, arr.w, train_idx)
    Xv, yv, wv = make_shared_side_xy(arr.X, arr.X_neg, arr.y, arr.w, val_idx)
    return fit_frozen_binary(Xtr, ytr, wtr, Xv, yv, wv)  # ONE shared model


def predict_b(model, arr: ABCData, idx):
    pL = model.predict_proba(arr.X[idx])[:, 1]
    pS = model.predict_proba(arr.X_neg[idx])[:, 1]
    return combine_side_scores(pL, pS)


def fit_c(arr: ABCData, train_idx, val_idx):
    long_model = fit_frozen_binary(
        arr.X[train_idx], arr.y[train_idx], arr.w[train_idx],
        arr.X[val_idx], arr.y[val_idx], arr.w[val_idx],
    )
    short_model = fit_frozen_binary(
        arr.X_neg[train_idx], 1 - arr.y[train_idx], arr.w[train_idx],
        arr.X_neg[val_idx], 1 - arr.y[val_idx], arr.w[val_idx],
    )
    return long_model, short_model


def predict_c(long_model, short_model, arr: ABCData, idx):
    pL = long_model.predict_proba(arr.X[idx])[:, 1]
    pS = short_model.predict_proba(arr.X_neg[idx])[:, 1]
    return combine_side_scores(pL, pS)


# --------------------------------------------------------------------------- #
# Evaluation (reuses frozen economic metric + bootstrap)                       #
# --------------------------------------------------------------------------- #
def _recall_metrics(pred, y):
    y = np.asarray(y)
    pred = np.asarray(pred)
    n_long = int((y == 1).sum())
    n_short = int((y == 0).sum())
    long_recall = float(((pred == 1) & (y == 1)).sum()) / n_long if n_long else None
    short_recall = float(((pred == 0) & (y == 0)).sum()) / n_short if n_short else None
    bal_acc = None
    if n_long and n_short:
        bal_acc = 0.5 * (long_recall + short_recall)
    return {
        "n_long_opps": n_long,
        "n_short_opps": n_short,
        "long_recall": long_recall,
        "short_recall": short_recall,
        "balanced_accuracy": bal_acc,
        "predicted_long_share": float((pred == 1).mean()),
    }


def eval_on_idx(arr: ABCData, idx, pred, p_long, ds=None, with_phase=False):
    """Full metric block on a test subset (idx into the full arrays)."""
    sub_y = arr.y[idx]
    sub_w = arr.w[idx]
    sub_eq = arr.eq[idx]
    sub_gid = arr.gid[idx]
    pdr = pred_direction_return_atr(pred, sub_y, sub_eq)
    tr, uniq = aggregate_per_trade(sub_gid, sub_w, pdr)
    mean_ret, lo, hi = bootstrap_trade_returns_chunked(tr)
    rec = _recall_metrics(pred, sub_y)
    acc = float((pred == sub_y).mean())
    try:
        auc = float(roc_auc_score(sub_y, p_long)) if len(np.unique(sub_y)) > 1 else None
    except Exception:
        auc = None

    def _ret(mask):
        if not bool(np.any(mask)):
            return None
        t, _ = aggregate_per_trade(sub_gid[mask], sub_w[mask], pdr[mask])
        m, _, _ = bootstrap_trade_returns_chunked(t)
        return float(m)

    long_mask = sub_y == 1
    short_mask = sub_y == 0
    out = {
        "n_rows": int(idx.size),
        "n_trades": int(len(uniq)),
        "return_atr": mean_ret,
        "ci_low": lo,
        "ci_high": hi,
        "accuracy": acc,
        "balanced_accuracy": rec["balanced_accuracy"],
        "roc_auc": auc,
        "predicted_long_share": rec["predicted_long_share"],
        "long_recall": rec["long_recall"],
        "short_recall": rec["short_recall"],
        "long_return": _ret(long_mask),
        "short_return": _ret(short_mask),
    }
    if with_phase and ds is not None:
        phases = {}
        for ph in ("BEFORE_ENTRY", "AT_ENTRY", "IN_POSITION"):
            pm = phase_mask(ds, idx, ph)
            if pm.any():
                phases[ph] = eval_on_idx(arr, idx[pm], pred[pm], p_long[pm])
            else:
                phases[ph] = None
        out["phase"] = phases
    return out


def aligned_returns(arr: ABCData, idx, pred):
    """Per-trade return vector aligned by sorted gid (identical order across A/B/C)."""
    gid = arr.gid[idx]
    pdr = pred_direction_return_atr(pred, arr.y[idx], arr.eq[idx])
    tr, uniq = aggregate_per_trade(gid, arr.w[idx], pdr)
    return tr, uniq


def aligned_returns_filtered(arr: ABCData, idx, pred, teacher_is_long: bool):
    row_mask = (arr.y[idx] == (1 if teacher_is_long else 0))
    sub_idx = idx[row_mask]
    return aligned_returns(arr, sub_idx, pred[row_mask])


def cluster_bootstrap_symbols(values, B=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    """Resample symbols-with-replacement cluster bootstrap (each value = one symbol's
    mean delta)."""
    values = np.asarray(values, dtype=float)
    k = values.size
    if k == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty(B, dtype=float)
    for b in range(B):
        s = rng.integers(0, k, size=k)
        means[b] = values[s].mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def mechanism_importance(model):
    imp = np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=float)
    s = imp.sum()
    norm = (imp / s) if s > 0 else imp
    return {DTP9[i]: float(norm[i]) for i in range(len(DTP9))}, int(model.best_iteration_)


def _verdict(ba, cb):
    ba_sup = ba["ci_low"] > 0
    cb_sup = cb["ci_low"] > 0
    cb_neg = cb["ci_high"] < 0
    cb_cross = (cb["ci_low"] <= 0 <= cb["ci_high"])
    if cb_neg:
        return "specialist architecture is harmful in this diagnostic"
    if cb_cross:
        return "no statistically identifiable specialist increment"
    if ba_sup and cb_sup:
        return "both side normalization and specialist freedom contribute diagnostically"
    if (not ba_sup) and cb_sup:
        return "specialist/asymmetric architecture increment identified diagnostically"
    if ba_sup and (not cb_sup):
        return "side normalization helps; separate specialist increment not identified"
    return "inconclusive"


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


# --------------------------------------------------------------------------- #
# Main experiment                                                              #
# --------------------------------------------------------------------------- #
def run_asymmetry_audit(save: bool = True, verbose: bool = True):
    def log(*a):
        if verbose:
            print(*a, file=sys.stderr, flush=True)

    log("verify manifest (fail-closed) ...")
    verify_manifest(SYMBOLS)

    log("build frozen split ...")
    split = build_frozen_split()
    ds = split["ds"]
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    arr = build_abc_data(ds)
    n_test_rows = int(test_idx.size)
    n_test_trades = int(len(np.unique(arr.gid[test_idx])))
    if n_test_rows != A_REFERENCE["test_rows"] or n_test_trades != A_REFERENCE["test_trades"]:
        raise RuntimeError(
            "STOP_ASYMMETRY_SPLIT_DRIFT:"
            f"rows={n_test_rows} trades={n_test_trades}"
        )

    # ----- fit A / B / C -----
    log("fit A (frozen M0) ...")
    m0 = fit_a(ds, train_idx, val_idx)
    pred_a, p_a = predict_a(m0, ds, test_idx)

    log("fit B (shared side) ...")
    mB = fit_b(arr, train_idx, val_idx)
    pred_b, p_b = predict_b(mB, arr, test_idx)

    log("fit C (two specialists) ...")
    mL, mS = fit_c(arr, train_idx, val_idx)
    pred_c, p_c = predict_c(mL, mS, arr, test_idx)

    # ----- A reproduction gate (STOP if broken) -----
    a_ret = eval_on_idx(arr, test_idx, pred_a, p_a)["return_atr"]
    if abs(a_ret - A_REFERENCE["m0_return_atr"]) > 1e-6:
        raise RuntimeError(
            "STOP_ASYMMETRY_A_REPRODUCTION_FAILED:"
            f"{a_ret} != {A_REFERENCE['m0_return_atr']}"
        )

    # ----- pooled evaluation -----
    pooled = {
        "A": eval_on_idx(arr, test_idx, pred_a, p_a, ds=ds, with_phase=True),
        "B": eval_on_idx(arr, test_idx, pred_b, p_b, ds=ds, with_phase=True),
        "C": eval_on_idx(arr, test_idx, pred_c, p_c, ds=ds, with_phase=True),
    }
    pooled_ex_ag = {
        "A": eval_on_idx(arr, test_idx[arr.symbol[test_idx] != "AG"], pred_a[arr.symbol[test_idx] != "AG"],
                         p_a[arr.symbol[test_idx] != "AG"]),
        "B": eval_on_idx(arr, test_idx[arr.symbol[test_idx] != "AG"], pred_b[arr.symbol[test_idx] != "AG"],
                         p_b[arr.symbol[test_idx] != "AG"]),
        "C": eval_on_idx(arr, test_idx[arr.symbol[test_idx] != "AG"], pred_c[arr.symbol[test_idx] != "AG"],
                         p_c[arr.symbol[test_idx] != "AG"]),
    }

    # ----- aligned per-trade returns + paired contrasts -----
    tr_A, uniq = aligned_returns(arr, test_idx, pred_a)
    tr_B, _ = aligned_returns(arr, test_idx, pred_b)
    tr_C, _ = aligned_returns(arr, test_idx, pred_c)
    assert tr_A.size == tr_B.size == tr_C.size == n_test_trades

    delta_BA = tr_B - tr_A
    delta_CB = tr_C - tr_B
    delta_CA = tr_C - tr_A
    ba = dict(zip(("mean", "ci_low", "ci_high"),
                  bootstrap_trade_returns_chunked(delta_BA)))
    cb = dict(zip(("mean", "ci_low", "ci_high"),
                  bootstrap_trade_returns_chunked(delta_CB)))
    ca = dict(zip(("mean", "ci_low", "ci_high"),
                  bootstrap_trade_returns_chunked(delta_CA)))

    # LONG / SHORT decomposition of the deltas
    tr_A_L, _ = aligned_returns_filtered(arr, test_idx, pred_a, True)
    tr_B_L, _ = aligned_returns_filtered(arr, test_idx, pred_b, True)
    tr_C_L, _ = aligned_returns_filtered(arr, test_idx, pred_c, True)
    tr_A_S, _ = aligned_returns_filtered(arr, test_idx, pred_a, False)
    tr_B_S, _ = aligned_returns_filtered(arr, test_idx, pred_b, False)
    tr_C_S, _ = aligned_returns_filtered(arr, test_idx, pred_c, False)
    cb_L = dict(zip(("mean", "ci_low", "ci_high"),
                    bootstrap_trade_returns_chunked(tr_C_L - tr_B_L)))
    cb_S = dict(zip(("mean", "ci_low", "ci_high"),
                    bootstrap_trade_returns_chunked(tr_C_S - tr_B_S)))
    ba_L = dict(zip(("mean", "ci_low", "ci_high"),
                    bootstrap_trade_returns_chunked(tr_B_L - tr_A_L)))
    ba_S = dict(zip(("mean", "ci_low", "ci_high"),
                    bootstrap_trade_returns_chunked(tr_B_S - tr_A_S)))
    ca_L = dict(zip(("mean", "ci_low", "ci_high"),
                    bootstrap_trade_returns_chunked(tr_C_L - tr_A_L)))
    ca_S = dict(zip(("mean", "ci_low", "ci_high"),
                    bootstrap_trade_returns_chunked(tr_C_S - tr_A_S)))

    # ----- per-symbol robustness -----
    per_symbol = []
    sym_cb_list = []   # per-symbol mean(C-B) for cluster bootstrap
    sym_ca_list = []
    sym_ba_list = []
    for s in SYMBOLS:
        m = arr.symbol[test_idx] == s
        sub = test_idx[m]
        if sub.size == 0:
            continue
        a = eval_on_idx(arr, sub, pred_a[m], p_a[m])
        b = eval_on_idx(arr, sub, pred_b[m], p_b[m])
        c = eval_on_idx(arr, sub, pred_c[m], p_c[m])
        tA, _ = aligned_returns(arr, sub, pred_a[m])
        tB, _ = aligned_returns(arr, sub, pred_b[m])
        tC, _ = aligned_returns(arr, sub, pred_c[m])
        dBA = float(np.mean(tB - tA))
        dCB = float(np.mean(tC - tB))
        dCA = float(np.mean(tC - tA))
        sym_ba_list.append(dBA)
        sym_cb_list.append(dCB)
        sym_ca_list.append(dCA)
        per_symbol.append({
            "symbol": s,
            "n_test_trades": a["n_trades"], "n_test_rows": a["n_rows"],
            "A_return": a["return_atr"], "A_ci_low": a["ci_low"], "A_ci_high": a["ci_high"],
            "B_return": b["return_atr"], "B_ci_low": b["ci_low"], "B_ci_high": b["ci_high"],
            "C_return": c["return_atr"], "C_ci_low": c["ci_low"], "C_ci_high": c["ci_high"],
            "B_minus_A": dBA, "C_minus_B": dCB, "C_minus_A": dCA,
            "A_long_return": a["long_return"], "A_short_return": a["short_return"],
            "A_long_recall": a["long_recall"], "A_short_recall": a["short_recall"],
            "B_long_return": b["long_return"], "B_short_return": b["short_return"],
            "B_long_recall": b["long_recall"], "B_short_recall": b["short_recall"],
            "C_long_return": c["long_return"], "C_short_return": c["short_return"],
            "C_long_recall": c["long_recall"], "C_short_recall": c["short_recall"],
        })

    per_symbol_cluster = {
        "C_minus_B": _cluster(cluster_bootstrap_symbols(sym_cb_list)),
        "C_minus_A": _cluster(cluster_bootstrap_symbols(sym_ca_list)),
        "B_minus_A": _cluster(cluster_bootstrap_symbols(sym_ba_list)),
        "n_symbols": len(sym_cb_list),
        "C_minus_B_positive_symbols": int(sum(1 for v in sym_cb_list if v > 0)),
        "C_minus_B_negative_symbols": int(sum(1 for v in sym_cb_list if v < 0)),
    }

    # ----- LOSO (15 folds, sequential; fold-internal n_jobs=-1) -----
    loso = []
    loso_cb_list = []
    for s in SYMBOLS:
        tr_f = train_idx[arr.symbol[train_idx] != s]
        va_f = val_idx[arr.symbol[val_idx] != s]
        te_f = test_idx[arr.symbol[test_idx] == s]
        if te_f.size == 0:
            continue
        m0f = fit_a(ds, tr_f, va_f)
        pa, pla = predict_a(m0f, ds, te_f)
        mBf = fit_b(arr, tr_f, va_f)
        pb, plb = predict_b(mBf, arr, te_f)
        mLf, mSf = fit_c(arr, tr_f, va_f)
        pc, plc = predict_c(mLf, mSf, arr, te_f)

        a = eval_on_idx(arr, te_f, pa, pla)
        b = eval_on_idx(arr, te_f, pb, plb)
        c = eval_on_idx(arr, te_f, pc, plc)
        tA, _ = aligned_returns(arr, te_f, pa)
        tB, _ = aligned_returns(arr, te_f, pb)
        tC, _ = aligned_returns(arr, te_f, pc)
        dBA = float(np.mean(tB - tA))
        dCB = float(np.mean(tC - tB))
        dCA = float(np.mean(tC - tA))
        loso_cb_list.append(dCB)
        loso.append({
            "held_out_symbol": s,
            "n_test_trades": a["n_trades"], "n_test_rows": a["n_rows"],
            "A_return": a["return_atr"], "B_return": b["return_atr"], "C_return": c["return_atr"],
            "B_minus_A": dBA, "C_minus_B": dCB, "C_minus_A": dCA,
            "A_long_return": a["long_return"], "A_short_return": a["short_return"],
            "A_long_recall": a["long_recall"], "A_short_recall": a["short_recall"],
            "B_long_return": b["long_return"], "B_short_return": b["short_return"],
            "B_long_recall": b["long_recall"], "B_short_recall": b["short_recall"],
            "C_long_return": c["long_return"], "C_short_return": c["short_return"],
            "C_long_recall": c["long_recall"], "C_short_recall": c["short_recall"],
        })
        log(f"  LOSO held-out {s}: A={a['return_atr']:.3f} B={b['return_atr']:.3f} "
             f"C={c['return_atr']:.3f} C-B={dCB:+.3f}")

    loso_agg = {
        "C_minus_B_mean": float(np.mean(loso_cb_list)),
        "C_minus_B_median": float(np.median(loso_cb_list)),
        "C_minus_B_positive_symbols": int(sum(1 for v in loso_cb_list if v > 0)),
        "C_minus_B_negative_symbols": int(sum(1 for v in loso_cb_list if v < 0)),
        "C_minus_B_cluster": _cluster(cluster_bootstrap_symbols(loso_cb_list)),
    }

    # ----- mechanism diagnostic (C experts) -----
    long_imp, long_it = mechanism_importance(mL)
    short_imp, short_it = mechanism_importance(mS)
    mechanism = {
        "LongExpert": {"best_iteration": long_it, "feature_importance_gain": long_imp},
        "ShortExpert": {"best_iteration": short_it, "feature_importance_gain": short_imp},
    }

    # ----- verdict -----
    asymmetry_supported_by_short = cb_S["ci_low"] > 0
    verdict = {
        "primary_contrast": "C-B",
        "text": _verdict(ba, cb),
        "B_minus_A_ci": [ba["ci_low"], ba["ci_high"]],
        "C_minus_B_ci": [cb["ci_low"], cb["ci_high"]],
        "C_minus_A_ci": [ca["ci_low"], ca["ci_high"]],
        "C_minus_B_LONG_ci": [cb_L["ci_low"], cb_L["ci_high"]],
        "C_minus_B_SHORT_ci": [cb_S["ci_low"], cb_S["ci_high"]],
        "asymmetry_supported_by_short": asymmetry_supported_by_short,
        "short_cb_note": (
            "C improves TEACHER_SHORT" if asymmetry_supported_by_short
            else "C does NOT improve TEACHER_SHORT; overall C-B may be LONG-driven"),
    }

    summary = _clean({
        "task_id": TASK_ID,
        "base_sha": BASE_SHA,
        "test_status": "architecture_diagnostic_not_pristine_confirmation",
        "metric": {
            "name": METRIC_NAME,
            "definition": METRIC_DEFINITION,
            "not_labels": METRIC_NOT_LABELS,
        },
        "frozen_split": {
            "test_rows": n_test_rows,
            "test_trades": n_test_trades,
            "train_rows": int(train_idx.size),
            "val_rows": int(val_idx.size),
            "a_reference": A_REFERENCE,
        },
        "A_reproduction": {
            "return_atr": a_ret,
            "expected": A_REFERENCE["m0_return_atr"],
            "match": abs(a_ret - A_REFERENCE["m0_return_atr"]) < 1e-6,
        },
        "contract": {
            "features": "DTP9 only",
            "base_params_unchanged": True,
            "no_threshold_tuning": True,
            "no_hyperparameter_tuning": True,
            "no_struct33": True,
            "no_symbol_feature": True,
            "entry_quality_atr_not_in_X": True,
            "model_redesign": False,
        },
        "pooled": pooled,
        "pooled_ex_ag": pooled_ex_ag,
        "paired_contrasts": {
            "primary": "C-B",
            "B_minus_A": ba, "C_minus_B": cb, "C_minus_A": ca,
            "B_minus_A_LONG": ba_L, "C_minus_B_LONG": cb_L, "C_minus_A_LONG": ca_L,
            "B_minus_A_SHORT": ba_S, "C_minus_B_SHORT": cb_S, "C_minus_A_SHORT": ca_S,
        },
        "per_symbol": per_symbol,
        "per_symbol_cluster_bootstrap": per_symbol_cluster,
        "loso": {"folds": loso, "aggregate": loso_agg},
        "mechanism": mechanism,
        "verdict": verdict,
        "provenance": {
            "base_sha": BASE_SHA,
            "reused_split_module": "direction_null_baseline_v1.build_frozen_split",
            "reused_m0_path": "direction_null_baseline_v1._fit_m0 / _predict_m0",
            "base_params_source": "train_direction_model_ag_v1.BASE_PARAMS",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
    })

    # ----- evidence files -----
    if save:
        os.makedirs(EVIDENCE_DIR, exist_ok=True)
        with open(SUMMARY_JSON, "w") as f:
            json.dump(summary, f, indent=2)
        _write_per_symbol_csv(per_symbol, PER_SYMBOL_CSV)
        _write_trade_returns_csv(arr, test_idx, tr_A, tr_B, tr_C, uniq)
        _write_loso_csv(loso, LOSO_CSV)
        log(f"evidence written -> {SUMMARY_JSON}")

    return {
        "summary": summary,
        "paths": {
            "summary": SUMMARY_JSON,
            "per_symbol_csv": PER_SYMBOL_CSV,
            "trade_returns_csv": TRADE_RETURNS_CSV,
            "loso_csv": LOSO_CSV,
        },
    }


def _cluster(triple):
    return {"mean": triple[0], "ci_low": triple[1], "ci_high": triple[2]}


def _write_per_symbol_csv(per_symbol, path):
    cols = ["symbol", "n_test_trades", "n_test_rows",
            "A_return", "A_ci_low", "A_ci_high",
            "B_return", "B_ci_low", "B_ci_high",
            "C_return", "C_ci_low", "C_ci_high",
            "B_minus_A", "C_minus_B", "C_minus_A",
            "A_long_return", "A_short_return", "A_long_recall", "A_short_recall",
            "B_long_return", "B_short_return", "B_long_recall", "B_short_recall",
            "C_long_return", "C_short_return", "C_long_recall", "C_short_recall"]
    df = pd.DataFrame(per_symbol)[cols]
    df.to_csv(path, index=False)


def _write_trade_returns_csv(arr, test_idx, tr_A, tr_B, tr_C, uniq):
    gid = arr.gid[test_idx]
    y = arr.y[test_idx]
    u, inv = np.unique(gid, return_inverse=True)
    y_opp = (np.bincount(inv, weights=y, minlength=len(u)) > 0).astype(np.uint8)
    df = pd.DataFrame({
        "gid": uniq,
        "teacher_direction": np.where(y_opp == 1, "LONG", "SHORT"),
        "tr_A": tr_A,
        "tr_B": tr_B,
        "tr_C": tr_C,
        "delta_BA": tr_B - tr_A,
        "delta_CB": tr_C - tr_B,
        "delta_CA": tr_C - tr_A,
    })
    df.to_csv(TRADE_RETURNS_CSV, index=False)


def _write_loso_csv(loso, path):
    cols = ["held_out_symbol", "n_test_trades", "n_test_rows",
            "A_return", "B_return", "C_return",
            "B_minus_A", "C_minus_B", "C_minus_A",
            "A_long_return", "A_short_return", "A_long_recall", "A_short_recall",
            "B_long_return", "B_short_return", "B_long_recall", "B_short_recall",
            "C_long_return", "C_short_return", "C_long_recall", "C_short_recall"]
    df = pd.DataFrame(loso)[cols]
    df.to_csv(path, index=False)


if __name__ == "__main__":
    run_asymmetry_audit(save=True, verbose=True)
