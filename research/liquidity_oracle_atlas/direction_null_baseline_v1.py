"""direction_null_baseline_v1
===========================

FUTURE-R4-M15-DIRECTION-NULL-BASELINE-AUDIT-V1

Goal: decide whether the frozen DTP9 direction result (+0.601 ATR/trade on TEST)
can be explained by LONG/SHORT label imbalance, opportunity imbalance, economic
payoff imbalance, or a trivial LONG-biased predictor. The frozen DTP9 (DIR-M0) is
re-fit DETERMINISTICALLY from the frozen datasets and the frozen LightGBM params
(no model redesign, no hyperparameter tuning, no threshold tuning); its TEST
predictions are reproduced byte-for-byte against the frozen evidence.

The frozen split is reproduced by REUSING the confirmation trainer's exact helpers
(build_frozen_split-equivalent), so this audit runs on the SAME TEST rows/trades
as the frozen evidence. A hard reproduction assert guards against drift.

Null predictors (evaluated on the frozen TEST):
  N0 ALWAYS_LONG
  N1 ALWAYS_SHORT
  N2 TRAIN_MAJORITY   (opportunity-weighted majority class of TRAIN)
  N3 FAIR_COIN        P(Long)=0.50          (stochastic, >=10000 draws)
  N4 TRAIN_PRIOR_COIN P(Long)=opp-weighted LONG share in TRAIN (stochastic)

Key structural fact used for fast Monte Carlo: for any predictor that assigns ONE
direction per Oracle opportunity, the per-opportunity economic return is

    return_i = eqw_i * (2*d_i - 1) * s_teacher_i

where eqw_i is the opportunity's (sample_weight-weighted) entry_quality_atr, d_i is
the predicted class (0/1) and s_teacher_i = +1 for Teacher LONG else -1. This makes
10,000+digit Monte Carlo a fully vectorized O(B * n) op. (For M0, which predicts per
Candidate row, the headline return is taken from the row-level evaluate_subset so it
stays byte-consistent with the frozen evidence; the per-opportunity form is used for
the deterministic nulls and for M0-vs-null deltas.)

This metric is a fixed-exit direction diagnostic and is NOT strategy PnL: it measures
the teacher's fixed-exit entry-quality under the predicted vs. teacher direction, not a
tradable, live, or realized return.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.train_direction_model_15sym_v1 import (
    BASE_PARAMS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIRMATION_SOURCE_SHA,
    DECISION_THRESHOLD,
    DTP9,
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
    opportunity_weighted_majority,
    phase_mask,
    pred_direction_return_atr,
    predict_direction,
    prepare_xy,
    remove_boundary_opportunities,
    verify_manifest,
)

TASK_ID = "FUTURE-R4-M15-DIRECTION-NULL-BASELINE-AUDIT-V1"
# The BASE SHA given by the reviewer (parent commit for this audit task).
TASK_BASE_SHA = "ab55121ed9171b80d2f1668c3e48aaa6f5d230bc"
# The commit this FIX1 is based on (the initial audit commit that was remote-reviewed).
FIX1_BASE_SHA = "f8ace222a63dfc5d5633101e9e4b994491e3e041"
FIX_VERSION = "FIX1"
# The frozen DTP9 confirmation model this audit reproduces.
FROZEN_MODEL_SHA = "686cb06fa5199f2712b1d8f8cfddd77f6c1c0a79"

METRIC_NAME = "TeacherFixedExitDirectionReturnATR"
METRIC_DEFINITION = (
    "(+entry_quality_atr if predicted direction == Teacher direction else "
    "-entry_quality_atr), aggregated per Oracle opportunity (trade), bootstrapped "
    "at the trade level"
)
METRIC_NOT_LABELS = ["strategy PnL", "live alpha", "tradable return"]

EVIDENCE_DIR = os.path.join("research", "liquidity_oracle_atlas", "evidence")
SUMMARY_JSON = os.path.join(EVIDENCE_DIR, "direction_null_baseline_v1_summary.json")
PER_SYMBOL_CSV = os.path.join(EVIDENCE_DIR, "direction_null_baseline_per_symbol.csv")

# Frozen TEST shape (from the confirmation evidence) used as a drift guard.
FIX1_REFERENCE = {
    "source_sha": FROZEN_MODEL_SHA,
    "test_rows": 13773,
    "test_trades": 638,
    "m0_return_atr": 0.6014965284150177,
    "m0_long_recall": 0.718,   # reviewer-stated; recomputed and checked below
    "m0_short_recall": 0.477,  # reviewer-stated
}

B_NULL = 10000          # >= 10000 Monte Carlo draws for the stochastic nulls
NULL_SEED = 20260923


# --------------------------------------------------------------------------- #
# Frozen split reproduction (verbatim from the confirmation trainer)            #
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
# Per-opportunity economic structures (vectorized Monte Carlo core)             #
# --------------------------------------------------------------------------- #
def _opp_structures(ds, idx):
    """Precompute per-opportunity structures for the TEST rows in `idx`."""
    tids = ds.loc[idx, "oracle_trade_id"].to_numpy(object)
    w = ds.loc[idx, "sample_weight_raw"].to_numpy(float)
    eq = ds.loc[idx, "entry_quality_atr"].to_numpy(float)
    y = (ds.loc[idx, "oracle_direction"].to_numpy(object) == "LONG").astype(np.uint8)
    uniq, inv = np.unique(tids, return_inverse=True)

    eqw = np.zeros(len(uniq), dtype=float)
    wsum = np.zeros(len(uniq), dtype=float)
    np.add.at(eqw, inv, w * eq)           # weighted by sample_weight_raw
    np.add.at(wsum, inv, w)               # per-opp weight sum
    eqw /= np.maximum(wsum, 1e-12)        # -> weighted AVERAGE (matches aggregate_per_trade)
    y_opp = (np.bincount(inv, weights=y, minlength=len(uniq)) > 0).astype(np.uint8)
    s_teacher = (2.0 * y_opp.astype(float) - 1.0)  # cast to float BEFORE subtract (uint8 0-1 underflows)
    return {
        "tids": tids, "w": w, "eq": eq, "y": y, "uniq": uniq, "inv": inv,
        "eqw": eqw, "y_opp": y_opp, "s_teacher": s_teacher, "n": len(uniq),
    }


def _return_for_pred(st, d):
    """Per-opportunity return when predictor assigns class d_i (0/1) to every row
    of opportunity i. eqw already encodes the opportunity's weighted entry_quality,
    so this is exact for any predictor that is constant within an opportunity."""
    return st["eqw"] * (2.0 * np.asarray(d, dtype=float) - 1.0) * st["s_teacher"]


def _m0_tr_canonical(ds, idx, m0_dir):
    y = (ds.loc[idx, "oracle_direction"].to_numpy(object) == "LONG").astype(np.uint8)
    w = ds.loc[idx, "sample_weight_raw"].to_numpy(float)
    eq = ds.loc[idx, "entry_quality_atr"].to_numpy(float)
    tids = ds.loc[idx, "oracle_trade_id"].to_numpy(object)
    pdr = pred_direction_return_atr(m0_dir, y, eq)
    tr, _ = aggregate_per_trade(tids, w, pdr)
    return tr


def _m0_opp_pred(ds, idx, m0_dir):
    """Opportunity-level M0 decision = opportunity-weighted majority of its row
    predictions. predict_direction returns uint8 (1=LONG, 0=SHORT)."""
    yp = (np.asarray(m0_dir) == 1).astype(np.uint8)
    w = ds.loc[idx, "sample_weight_raw"].to_numpy(float)
    tids = ds.loc[idx, "oracle_trade_id"].to_numpy(object)
    uniq, inv = np.unique(tids, return_inverse=True)
    d = np.empty(len(uniq), dtype=np.uint8)
    for i in range(len(uniq)):
        rows = inv == i
        d[i] = opportunity_weighted_majority(yp[rows], w[rows])
    return d


def _confusion(d, y_opp):
    d = np.asarray(d, dtype=int)
    y_opp = np.asarray(y_opp, dtype=int)
    tp = int(((d == 1) & (y_opp == 1)).sum())
    fn = int(((d == 0) & (y_opp == 1)).sum())
    fp = int(((d == 1) & (y_opp == 0)).sum())
    tn = int(((d == 0) & (y_opp == 0)).sum())
    long_rec = tp / (tp + fn) if (tp + fn) else None
    short_rec = tn / (tn + fp) if (tn + fp) else None
    bal_acc = (None if None in (long_rec, short_rec)
               else 0.5 * (long_rec + short_rec))
    prec_l = tp / (tp + fp) if (tp + fp) else None
    # SHORT as the positive class: TP_short = TN, FP_short = FN
    #   Precision_SHORT = TP_short / (TP_short + FP_short) = TN / (TN + FN)
    prec_s = tn / (tn + fn) if (tn + fn) else None
    return {
        "n_long_opps": int((y_opp == 1).sum()),
        "n_short_opps": int((y_opp == 0).sum()),
        "pred_long_opps": int((d == 1).sum()),
        "pred_short_opps": int((d == 0).sum()),
        "predicted_long_pct": float(d.mean()),
        "long_recall": long_rec, "short_recall": short_rec,
        "balanced_accuracy": bal_acc,
        "precision_long": prec_l, "precision_short": prec_s,
    }


# --------------------------------------------------------------------------- #
# Null predictors + M0-vs-null                                                 #
# --------------------------------------------------------------------------- #
def _null_evaluate(st, kind, p, B: int = B_NULL, seed: int = NULL_SEED):
    """Return/CI for a null predictor on the opportunity set described by `st`."""
    if kind in ("always_long", "always_short", "train_majority"):
        d = np.full(st["n"], int(round(p)), dtype=np.uint8)
        null_tr = _return_for_pred(st, d)
        mean, lo, hi = bootstrap_trade_returns_chunked(null_tr, B=B, seed=seed)
        acc = float((d == st["y_opp"]).mean())
        return {"mean_return": mean, "ci_low": lo, "ci_high": hi,
                "mean_accuracy": acc, "mean_predicted_long_pct": float(d.mean())}

    # stochastic coin: fully vectorized Monte Carlo
    rng = np.random.default_rng(seed)
    draws = (rng.random((B, st["n"])) < p).astype(np.uint8)
    sign = (2.0 * draws - 1.0) * st["s_teacher"]
    ret = (st["eqw"] * sign).mean(axis=1)
    acc = (draws == st["y_opp"]).mean(axis=1)
    pl = draws.mean(axis=1)
    lo, hi = np.quantile(ret, [0.025, 0.975])
    return {"mean_return": float(ret.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "mean_accuracy": float(acc.mean()),
            "mean_predicted_long_pct": float(pl.mean())}


def _null_sim_distribution(st, p, B: int = B_NULL, seed: int = NULL_SEED):
    """Raw Monte Carlo distribution of null mean returns (for M0-vs-null deltas)."""
    rng = np.random.default_rng(seed)
    draws = (rng.random((B, st["n"])) < p).astype(np.uint8)
    sign = (2.0 * draws - 1.0) * st["s_teacher"]
    return (st["eqw"] * sign).mean(axis=1)


def _m0_minus_null(m0_tr, st, kind, p, B: int = B_NULL, seed: int = NULL_SEED):
    m0_mean = float(m0_tr.mean())
    if kind in ("always_long", "always_short", "train_majority"):
        d = np.full(st["n"], int(round(p)), dtype=np.uint8)
        delta = m0_tr - _return_for_pred(st, d)
        mean, lo, hi = bootstrap_trade_returns_chunked(delta, B=B, seed=seed)
        return {"kind": kind, "mean_delta": mean, "ci_low": lo, "ci_high": hi,
                "n_trades": int(len(delta)), "monte_carlo": False}
    null_ret = _null_sim_distribution(st, p, B=B, seed=seed)
    deltas = m0_mean - null_ret
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {"kind": kind, "mean_delta": float(deltas.mean()), "ci_low": float(lo),
            "ci_high": float(hi), "n_trades": int(st["n"]), "monte_carlo": True}


# --------------------------------------------------------------------------- #
# Audit helpers                                                                #
# --------------------------------------------------------------------------- #
def _label_balance(ds, idx):
    if idx.size == 0:
        return {"n_rows": 0, "n_trades": 0}
    df = pd.DataFrame({
        "tid": ds.loc[idx, "oracle_trade_id"].to_numpy(object),
        "dir": ds.loc[idx, "oracle_direction"].to_numpy(object),
        "w": ds.loc[idx, "sample_weight_raw"].to_numpy(float),
    })
    g = df.groupby("tid")["dir"].agg(lambda s: s.iloc[0])
    long_rows = int((df["dir"] == "LONG").sum())
    short_rows = int((df["dir"] == "SHORT").sum())
    long_opps = int((g == "LONG").sum())
    short_opps = int((g == "SHORT").sum())
    w_long = float(df.loc[df["dir"] == "LONG", "w"].sum())
    w_short = float(df.loc[df["dir"] == "SHORT", "w"].sum())
    return {
        "n_rows": int(idx.size),
        "n_trades": int(g.size),
        "long_rows": long_rows, "short_rows": short_rows,
        "long_rows_pct": long_rows / max(1, long_rows + short_rows),
        "long_opps": long_opps, "short_opps": short_opps,
        "long_opps_pct": long_opps / max(1, long_opps + short_opps),
        "sum_w_long": w_long, "sum_w_short": w_short,
        "sum_w_long_pct": w_long / max(1e-12, w_long + w_short),
    }


def _phase_balances(ds, test_idx):
    out = {}
    for ph in ("BEFORE_ENTRY", "AT_ENTRY", "IN_POSITION"):
        mask = phase_mask(ds, test_idx, ph)
        out[ph] = _label_balance(ds, test_idx[mask])
    return out


def _economic_payoff(ds, idx):
    """EntryQualityATR statistics split by true Teacher direction (per opportunity)."""
    st = _opp_structures(ds, idx)
    eqw = st["eqw"]
    long_mask = st["y_opp"] == 1
    short_mask = st["y_opp"] == 0

    def stats(a):
        if a.size == 0:
            return None
        a = np.asarray(a, float)
        return {
            "n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p25": float(np.percentile(a, 25)),
            "p50": float(np.percentile(a, 50)), "p75": float(np.percentile(a, 75)),
            "p90": float(np.percentile(a, 90)),
            "mean_abs": float(np.abs(a).mean()),
        }

    return {
        "long": stats(eqw[long_mask]),
        "short": stats(eqw[short_mask]),
        "all": stats(eqw),
    }


def _audit_subset(ds, idx, m0_dir, m0_plong, p_train_prior, B: int = B_NULL,
                  seed: int = NULL_SEED):
    if idx.size == 0:
        return {}
    st = _opp_structures(ds, idx)
    d_m0 = _m0_opp_pred(ds, idx, m0_dir)
    m0_tr = _m0_tr_canonical(ds, idx, m0_dir)
    m0_block = evaluate_subset(ds, idx, m0_dir, m0_plong)
    conf = _confusion(d_m0, st["y_opp"])

    nulls = {}
    for kind, p in (("always_long", 1.0), ("always_short", 0.0),
                    ("train_majority", round(p_train_prior)),
                    ("fair_coin", 0.5), ("train_prior_coin", p_train_prior)):
        nulls[kind] = _null_evaluate(st, kind, p, B=B, seed=seed)

    m0_minus = {}
    for kind, p in (("always_long", 1.0), ("always_short", 0.0),
                    ("train_majority", round(p_train_prior)),
                    ("fair_coin", 0.5), ("train_prior_coin", p_train_prior)):
        m0_minus[kind] = _m0_minus_null(m0_tr, st, kind, p, B=B, seed=seed)

    return {
        "n_rows": int(idx.size),
        "n_trades": int(st["n"]),
        "m0_block": m0_block,
        "m0_prediction_bias": conf,
        "null_predictors": nulls,
        "m0_minus_null": m0_minus,
    }


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def run_null_baseline_audit(symbols=SYMBOLS, save: bool = True, verbose: bool = True):
    verify_manifest(symbols)
    split_data = build_frozen_split(symbols)
    ds = split_data["ds"]
    test_idx = split_data["test_idx"]
    train_idx = split_data["train_idx"]
    val_idx = split_data["val_idx"]

    # ---- frozen-split reproduction guard ----
    n_test = int(test_idx.size)
    n_test_trades = int(pd.unique(ds.loc[test_idx, "oracle_trade_id"]).size)
    if n_test != FIX1_REFERENCE["test_rows"] or n_test_trades != FIX1_REFERENCE["test_trades"]:
        raise RuntimeError(
            f"STOP_NULL_AUDIT_SPLIT_DRIFT:{n_test}/{n_test_trades}")
    if verbose:
        print(f"[NULL] frozen TEST rows={n_test} trades={n_test_trades}")

    # ---- reproduce frozen DTP9 (M0) pooled predictions ----
    model = _fit_m0(ds, train_idx, val_idx)
    m0_dir, m0_plong = _predict_m0(model, ds, test_idx)
    # also predict train/val for payoff-diagnostic baselines
    m0_dir_tr, _ = _predict_m0(model, ds, train_idx)
    m0_dir_va, _ = _predict_m0(model, ds, val_idx)

    # canonical per-opp M0 TEST return (row-level, matches frozen evidence)
    m0_tr = _m0_tr_canonical(ds, test_idx, m0_dir)
    repro_ok = abs(float(m0_tr.mean()) - FIX1_REFERENCE["m0_return_atr"]) < 1e-6
    if not repro_ok:
        raise RuntimeError(
            "STOP_NULL_AUDIT_M0_REPRODUCTION_FAIL:"
            f"{float(m0_tr.mean())} != {FIX1_REFERENCE['m0_return_atr']}")
    if verbose:
        print(f"[NULL] M0 TEST return={float(m0_tr.mean()):.4f} (FIX1 reproduced)")

    # TRAIN opportunity-weighted LONG share -> prior coin p
    y_tr = (ds.loc[train_idx, "oracle_direction"].to_numpy(object) == "LONG").astype(np.uint8)
    w_tr = ds.loc[train_idx, "sample_weight_raw"].to_numpy(float)
    p_train_prior = float(np.average(y_tr, weights=w_tr))
    majority = opportunity_weighted_majority(y_tr, w_tr)

    sym_test = ds.loc[test_idx, "symbol"].to_numpy(object)

    # ---- A. label balance ----
    label_balance = {
        "TRAIN": _label_balance(ds, train_idx),
        "VALIDATION": _label_balance(ds, val_idx),
        "TEST": _label_balance(ds, test_idx),
        "TEST_by_symbol": {s: _label_balance(ds, test_idx[sym_test == s])
                           for s in symbols},
        "TEST_by_phase": _phase_balances(ds, test_idx),
    }

    # ---- B. economic payoff balance ----
    economic_payoff = {
        "TRAIN_diagnostic": _economic_payoff(ds, train_idx),
        "VALIDATION_diagnostic": _economic_payoff(ds, val_idx),
        "TEST": _economic_payoff(ds, test_idx),
        "TEST_by_symbol": {s: _economic_payoff(ds, test_idx[sym_test == s])
                           for s in symbols},
        "always_long": {
            "TRAIN": evaluate_subset(ds, train_idx, np.full(train_idx.size, 1, dtype=np.uint8), None),
            "VALIDATION": evaluate_subset(ds, val_idx, np.full(val_idx.size, 1, dtype=np.uint8), None),
            "TEST": evaluate_subset(ds, test_idx, np.full(test_idx.size, 1, dtype=np.uint8), None),
            "TEST_by_symbol": {s: evaluate_subset(ds, test_idx[sym_test == s],
                                                  np.full(int((sym_test == s).sum()), 1, dtype=np.uint8), None)
                               for s in symbols},
        },
        "always_short": {
            "TRAIN": evaluate_subset(ds, train_idx, np.full(train_idx.size, 0, dtype=np.uint8), None),
            "VALIDATION": evaluate_subset(ds, val_idx, np.full(val_idx.size, 0, dtype=np.uint8), None),
            "TEST": evaluate_subset(ds, test_idx, np.full(test_idx.size, 0, dtype=np.uint8), None),
            "TEST_by_symbol": {s: evaluate_subset(ds, test_idx[sym_test == s],
                                                  np.full(int((sym_test == s).sum()), 0, dtype=np.uint8), None)
                               for s in symbols},
        },
    }

    # ---- C/D/E. M0 prediction bias, null predictors, M0-vs-null ----
    pooled = _audit_subset(ds, test_idx, m0_dir, m0_plong, p_train_prior)
    pooled_ex_ag = _audit_subset(ds, test_idx[sym_test != "AG"], m0_dir[sym_test != "AG"],
                                 m0_plong[sym_test != "AG"], p_train_prior)
    per_symbol = {}
    for s in symbols:
        m = sym_test == s
        per_symbol[s] = _audit_subset(ds, test_idx[m], m0_dir[m], m0_plong[m], p_train_prior)

    # ---- core-result preservation guards (FIX1) ----
    _al_ret = pooled["null_predictors"]["always_long"]["mean_return"]
    _fc_ret = pooled["null_predictors"]["fair_coin"]["mean_return"]
    _m0_al = pooled["m0_minus_null"]["always_long"]
    _m0_fc = pooled["m0_minus_null"]["fair_coin"]
    assert abs(float(m0_tr.mean()) - 0.6014965284150177) < 1e-6
    assert abs(_al_ret - 0.017734) < 2e-3, f"Always Long drift: {_al_ret}"
    assert abs(_fc_ret) < 0.05, f"Fair Coin MC not ~0: {_fc_ret}"
    assert _m0_al["ci_low"] > 0, f"M0-AlwaysLong CI not >0: {_m0_al}"
    assert _m0_fc["ci_low"] > 0, f"M0-FairCoin CI not >0: {_m0_fc}"

    # ---- F. class-balanced counterfactual (TEST) ----
    st_test = _opp_structures(ds, test_idx)
    n_long = int((st_test["y_opp"] == 1).sum())
    n_short = int((st_test["y_opp"] == 0).sum())
    m0_tr_test = _m0_tr_canonical(ds, test_idx, m0_dir)
    d_m0_test = _m0_opp_pred(ds, test_idx, m0_dir)
    # canonical per-trade returns for the constant predictors (== opportunity-level form,
    # because the predictor is constant within every opportunity)
    al_tr = _m0_tr_canonical(ds, test_idx, np.full(test_idx.size, 1, dtype=np.uint8))
    as_tr = _m0_tr_canonical(ds, test_idx, np.full(test_idx.size, 0, dtype=np.uint8))
    # opportunity-level reweight so each side totals 0.5 (matches contract)
    rew = np.where(st_test["y_opp"] == 1, 0.5 / max(1, n_long), 0.5 / max(1, n_short))
    assert abs(float(rew.sum()) - 1.0) < 1e-12

    def _cb(per_trade_ret):
        return float(np.sum(rew * per_trade_ret))   # sum(rew) == 1.0

    m0_cb_ret = _cb(m0_tr_test)
    al_cb_ret = _cb(al_tr)
    as_cb_ret = _cb(as_tr)
    m0_cb_acc = float(np.sum(rew * (d_m0_test == st_test["y_opp"]).astype(float)))
    al_cb_acc = float(np.sum(rew * (np.ones(st_test["n"], dtype=np.uint8) == st_test["y_opp"]).astype(float)))
    as_cb_acc = float(np.sum(rew * (np.zeros(st_test["n"], dtype=np.uint8) == st_test["y_opp"]).astype(float)))

    # Because TEST is 319/319, opportunity-level class reweighting must be a no-op:
    # assert the class-balanced values equal the ordinary opportunity-level values.
    if (abs(m0_cb_ret - float(m0_tr_test.mean())) > 1e-9
            or abs(al_cb_ret - float(al_tr.mean())) > 1e-9
            or abs(as_cb_ret - float(as_tr.mean())) > 1e-9):
        raise RuntimeError(
            "STOP_NULL_AUDIT_CLASS_BALANCE_NOT_NOOP:"
            f"{m0_cb_ret}/{float(al_tr.mean())}/{float(as_tr.mean())}")

    class_balanced = {
        "n_long_opps": n_long, "n_short_opps": n_short,
        "M0": {
            "class_balanced_accuracy": m0_cb_acc,
            "class_balanced_return": m0_cb_ret,
            "unweighted_return": float(m0_tr_test.mean()),
            "unweighted_accuracy": float((d_m0_test == st_test["y_opp"]).mean()),
        },
        "ALWAYS_LONG": {
            "class_balanced_accuracy": al_cb_acc,
            "class_balanced_return": al_cb_ret,
        },
        "ALWAYS_SHORT": {
            "class_balanced_accuracy": as_cb_acc,
            "class_balanced_return": as_cb_ret,
        },
        "no_op_assertion": {
            "m0_return_match": abs(m0_cb_ret - float(m0_tr_test.mean())) < 1e-9,
            "always_long_match": abs(al_cb_ret - float(al_tr.mean())) < 1e-9,
            "always_short_match": abs(as_cb_ret - float(as_tr.mean())) < 1e-9,
        },
        "note": ("TEST already 50/50 at the opportunity level (319 LONG / 319 SHORT), "
                 "so opportunity-level class reweighting is a no-op; reported to confirm, not to change."),
    }

    # ---- G. hard fact check (recompute from frozen dataset) ----
    hard_fact = {
        "unique_teacher_opportunities": int(st_test["n"]),
        "teacher_long_opps": int((st_test["y_opp"] == 1).sum()),
        "teacher_short_opps": int((st_test["y_opp"] == 0).sum()),
        "expected": {"total": 638, "long": 319, "short": 319},
    }
    if (hard_fact["unique_teacher_opportunities"] != 638
            or hard_fact["teacher_long_opps"] != 319
            or hard_fact["teacher_short_opps"] != 319):
        raise RuntimeError(f"STOP_NULL_AUDIT_FACT_CHECK_FAILED:{hard_fact}")

    summary = {
        "task_id": TASK_ID,
        "base_sha": FROZEN_MODEL_SHA,
        "task_base_sha": TASK_BASE_SHA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric": {"name": METRIC_NAME, "definition": METRIC_DEFINITION,
                   "is_not": METRIC_NOT_LABELS,
                   "note": ("teacher-fixed-exit direction diagnostic; not strategy PnL")},
        "provenance": {
            "confirmation_base_sha": "48605666cd9e3cb0c2d49848df47bb7050a0dabc",
            "fix1_base_sha": FIX1_BASE_SHA,
            "confirmation_source_sha": CONFIRMATION_SOURCE_SHA,
            "frozen_model_sha": FROZEN_MODEL_SHA,
            "task_base_sha": TASK_BASE_SHA,
        },
        "model_redesign": False,
        "hyperparameter_tuning": False,
        "threshold_tuning": False,
        "deterministic_model_refit_for_reproduction": True,
        "fix_version": FIX_VERSION,
        "fix1_base_sha": FIX1_BASE_SHA,
        "frozen_split_reproduction": {
            "fix1_source_sha": FIX1_REFERENCE["source_sha"],
            "test_rows": FIX1_REFERENCE["test_rows"], "test_trades": FIX1_REFERENCE["test_trades"],
            "recomputed_m0_return_atr": float(m0_tr.mean()), "match": bool(repro_ok),
        },
        "calendar": {
            "t1": split_data["cal"]["t1"].isoformat(),
            "t2": split_data["cal"]["t2"].isoformat(),
            "snap_rule": split_data["cal"]["snap_rule"],
        },
        "train_prior_long_share": p_train_prior,
        "train_majority_class": int(majority),
        "m0_test_headline": pooled["m0_block"],
        "A_label_balance": label_balance,
        "B_economic_payoff": economic_payoff,
        "C_m0_prediction_bias": {
            "POOLED": pooled["m0_prediction_bias"],
            "POOLED_EX_AG": pooled_ex_ag["m0_prediction_bias"],
            "per_symbol": {s: per_symbol[s]["m0_prediction_bias"] for s in symbols},
        },
        "D_null_predictors": {
            "POOLED": pooled["null_predictors"],
            "POOLED_EX_AG": pooled_ex_ag["null_predictors"],
            "per_symbol": {s: per_symbol[s]["null_predictors"] for s in symbols},
        },
        "E_model_vs_null": {
            "POOLED": pooled["m0_minus_null"],
            "POOLED_EX_AG": pooled_ex_ag["m0_minus_null"],
        },
        "fair_coin_analytical_null": {
            "expected_return": 0.0,
            "explanation": (
                "For each opportunity a fair coin is correct with P=0.5 and wrong with "
                "P=0.5, so E[+entry_quality_atr | correct] + E[-entry_quality_atr | wrong] = 0 "
                "regardless of the LONG/SHORT class balance."),
        },
        "F_class_balanced_counterfactual": class_balanced,
        "G_hard_fact_check": hard_fact,
        "params": {"decision_threshold": DECISION_THRESHOLD, "base_params": BASE_PARAMS,
                   "bootstrap_replicates": BOOTSTRAP_REPLICATES, "null_seed": NULL_SEED,
                   "null_draws": B_NULL},
    }
    # fix placeholder
    summary["provenance"]["confirmation_base_sha"] = "48605666cd9e3cb0c2d49848df47bb7050a0dabc"

    # ---- evidence ----
    if save:
        os.makedirs(EVIDENCE_DIR, exist_ok=True)
        with open(SUMMARY_JSON, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)

        rows = []
        for s in symbols:
            a = per_symbol[s]
            pb = a["m0_prediction_bias"]
            nl = a["null_predictors"]
            rows.append({
                "metric": METRIC_NAME, "symbol": s,
                "n_test_rows": a["n_rows"], "n_test_trades": a["n_trades"],
                "n_long_opps": pb["n_long_opps"], "n_short_opps": pb["n_short_opps"],
                "m0_predicted_long_pct": pb["predicted_long_pct"],
                "m0_long_recall": pb["long_recall"], "m0_short_recall": pb["short_recall"],
                "m0_balanced_accuracy": pb["balanced_accuracy"],
                "m0_precision_long": pb["precision_long"], "m0_precision_short": pb["precision_short"],
                "m0_return_atr": a["m0_block"]["return_atr"],
                "m0_ci_low": a["m0_block"]["ci_low"], "m0_ci_high": a["m0_block"]["ci_high"],
                "m0_accuracy": a["m0_block"]["accuracy"], "m0_auc": a["m0_block"]["roc_auc"],
                "always_long_return_atr": nl["always_long"]["mean_return"],
                "always_short_return_atr": nl["always_short"]["mean_return"],
                "fair_coin_mean_return": nl["fair_coin"]["mean_return"],
                "train_prior_coin_mean_return": nl["train_prior_coin"]["mean_return"],
            })
        pd.DataFrame(rows).to_csv(PER_SYMBOL_CSV, index=False)

    return {
        "summary": summary,
        "split_data": split_data,
        "paths": {"summary": SUMMARY_JSON, "per_symbol_csv": PER_SYMBOL_CSV},
    }


if __name__ == "__main__":
    r = run_null_baseline_audit()
    s = r["summary"]
    print(json.dumps({
        "task_id": s["task_id"],
        "metric": s["metric"]["name"],
        "m0_test_headline": {k: s["m0_test_headline"][k] for k in
                             ("n_rows", "n_trades", "return_atr", "ci_low", "ci_high",
                              "accuracy", "roc_auc")},
        "train_prior_long_share": s["train_prior_long_share"],
        "C_POOLED_bias": s["C_m0_prediction_bias"]["POOLED"],
        "D_POOLED_nulls": {k: {kk: v[kk] for kk in ("mean_return", "ci_low", "ci_high")}
                            for k, v in s["D_null_predictors"]["POOLED"].items()},
        "E_POOLED_m0_minus_null": {k: {kk: v[kk] for kk in ("mean_delta", "ci_low", "ci_high")}
                                    for k, v in s["E_model_vs_null"]["POOLED"].items()},
        "F_class_balanced": s["F_class_balanced_counterfactual"],
        "G_fact": s["G_hard_fact_check"],
    }, indent=2, default=str))
    print("paths:", r["paths"])
