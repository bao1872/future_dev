"""direction_gated_experts_v1
================================

FUTURE-R4-M15-DIRECTION-GATED-EXPERTS-V1

True gated mixture-of-experts Direction architecture.

The frozen router (A = Direct DTP9 M0) owns the preliminary LONG/SHORT thesis.
A second-stage model ONLY judges whether that router thesis is correct:

    router_correct = 1{router_pred == Teacher direction}

Teacher direction is LABEL ONLY. It must NOT orient features, route experts,
enter X, or set router side. Routing and orientation use ONLY the router's own
prediction -- this is what makes it a genuinely deployable two-stage Direction
architecture (and what distinguishes it from the earlier complementary/isomorphic
mirror control and from the Teacher-oriented Entry-Quality diagnostic).

Systems compared:
  A   - frozen Direct DTP9 router (M0)
  M9  - ONE shared router-correctness classifier on META10 (router-oriented DTP9
        + router_p_side)
  E9  - TWO experts: LongExpert (router-LONG rows) + ShortExpert (router-SHORT rows),
        same META10 schema, different training populations and different fitted models
  M33 - ONE shared classifier on META34 (router-oriented STRUCT33 + router_p_side)
  E33 - TWO experts on META34

Preregistered contrasts:
  E9  - A     performance primary   (does the gated system beat the frozen router?)
  E9  - M9    architecture primary  (is the gain from having TWO experts?)
  E33 - E9    structure secondary   (STRUCT33 increment inside true experts?)
  E33 - M33   structure/expert interaction

STATUS: architecture_diagnostic_not_pristine_confirmation. The expert hypothesis was
motivated by already-observed TEST LONG/SHORT asymmetry, so the current TEST is usable
for architecture diagnosis but NOT as pristine confirmation for adoption.

No Entry / Wait / Hold / Exit / stop-loss logic is part of this task.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from research.liquidity_oracle_atlas.train_direction_model_15sym_v1 import (
    BASE_PARAMS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DTP9,
    SYMBOLS,
    STRUCT33,
    aggregate_per_trade,
    bootstrap_trade_returns_chunked,
    fit_direction_model,
    phase_mask,
    pred_direction_return_atr,
    prepare_xy,
    verify_manifest,
)
# Read-only reuse of the exact frozen split and M0 fit/predict path.
from research.liquidity_oracle_atlas.direction_null_baseline_v1 import (
    build_frozen_split,
    _fit_m0,
    _predict_m0,
)

TASK_ID = "FUTURE-R4-M15-DIRECTION-GATED-EXPERTS-V1"
BASE_SHA = "89f655be47794f06146daef0573f8ddcbfdaa7b2"
TEST_STATUS = "architecture_diagnostic_not_pristine_confirmation"
METRIC_NAME = "TeacherFixedExitDirectionReturnATR"

EVIDENCE_DIR = os.path.join("research", "liquidity_oracle_atlas", "evidence")
SUMMARY_JSON = os.path.join(EVIDENCE_DIR, "direction_gated_experts_v1_summary.json")
PER_SYMBOL_CSV = os.path.join(EVIDENCE_DIR, "direction_gated_experts_v1_per_symbol.csv")
TRADE_RETURNS_CSV = os.path.join(EVIDENCE_DIR, "direction_gated_experts_v1_trade_returns.csv")
LOSO_CSV = os.path.join(EVIDENCE_DIR, "direction_gated_experts_v1_loso.csv")
TIME_BLOCKS_CSV = os.path.join(EVIDENCE_DIR, "direction_gated_experts_v1_time_blocks.csv")
OOF_AUDIT_CSV = os.path.join(EVIDENCE_DIR, "direction_gated_experts_v1_oof_router_audit.csv")

SYSTEMS = ("A", "M9", "E9", "M33", "E33")

# Frozen TEST universe + frozen A numerics that every run must reproduce.
A_REFERENCE = {
    "test_rows": 13773,
    "test_trades": 638,
    "long_trades": 319,
    "short_trades": 319,
    "return_atr": 0.6014965284150177,
    "accuracy": 0.5973520786915624,
    "roc_auc": 0.6368109177293068,
    "long_recall": 0.7177186815261937,
    "short_recall": 0.47698547585693135,
    "pred_long_share": 0.6203666028346312,
    "long_return": 1.525813321755975,
    "short_return": -0.32282026492593974,
}

TF_ORDER = ("m15", "h1", "h4")
_PER_TF = (
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
# Router-oriented canonical names (documentation / tests only; X is a numpy array).
_ORIENT_PER_TF = (
    "trend_with_side",
    "slope_with_side",
    "dev_with_side",
    "sr_backstop_dist_atr",
    "sr_ahead_dist_atr",
    "sr_backstop_strength",
    "sr_ahead_strength",
    "liq_ahead_dist_atr",
    "liq_behind_dist_atr",
    "liq_ahead_count",
    "liq_behind_count",
)
ORIENT33_COLS = [f"{tf}_{c}" for tf in TF_ORDER for c in _ORIENT_PER_TF]
ORIENT9_COLS = [f"{tf}_{c}" for tf in TF_ORDER for c in _ORIENT_PER_TF[:3]]
ROUTER_CONF_COL = "router_p_side"
META10_COLS = ORIENT9_COLS + [ROUTER_CONF_COL]
META34_COLS = ORIENT33_COLS + [ROUTER_CONF_COL]

STRUCT33_COLS = list(STRUCT33)
DTP_COLS = list(DTP9)


# --------------------------------------------------------------------------- #
# Data container (loaded ONCE, re-read for no model and no fold)               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DirectionExpertData:
    X9: np.ndarray                # [N, 9]  float32 (DTP9)
    X33: np.ndarray               # [N, 33] float32 (STRUCT33)
    y: np.ndarray                 # [N] uint8  1 = Teacher LONG
    w: np.ndarray                 # [N] float64 sample_weight_raw
    gid: np.ndarray               # [N] str    symbol::oracle_trade_id
    symbol: np.ndarray            # [N] object
    decision_time_ns: np.ndarray  # [N] int64  candidate_decision_time
    oef_time_ns: np.ndarray       # [N] int64  oracle_entry_fill_time (time blocks only)
    entry_quality_atr: np.ndarray  # [N] float64


def build_direction_expert_data(ds: pd.DataFrame) -> DirectionExpertData:
    X9 = ds.loc[:, DTP_COLS].to_numpy(dtype=np.float32, copy=True)
    X33 = ds.loc[:, STRUCT33_COLS].to_numpy(dtype=np.float32, copy=True)
    y = (ds["oracle_direction"].to_numpy(object) == "LONG").astype(np.uint8)
    w = ds["sample_weight_raw"].to_numpy(np.float64)
    symbol = ds["symbol"].to_numpy(object)
    gid = np.char.add(
        np.char.add(symbol.astype(str), "::"),
        ds["oracle_trade_id"].to_numpy(object).astype(str),
    )
    dt = pd.to_datetime(ds["candidate_decision_time"]).to_numpy(dtype="datetime64[ns]")
    ot = pd.to_datetime(ds["oracle_entry_fill_time"]).to_numpy(dtype="datetime64[ns]")
    eq = ds["entry_quality_atr"].to_numpy(np.float64)
    return DirectionExpertData(
        X9, X33, y, w, gid, symbol,
        dt.astype("int64"), ot.astype("int64"), eq,
    )


# --------------------------------------------------------------------------- #
# Router-side feature orientation (Router side, NEVER Teacher side)            #
# --------------------------------------------------------------------------- #
def orient_dtp9_router_side(X9: np.ndarray, router_long: np.ndarray) -> np.ndarray:
    """DTP9 is exactly the 3 signed trend fields per TF, so all 9 columns flip sign."""
    side = np.where(np.asarray(router_long, dtype=bool), 1.0, -1.0).astype(np.float32)
    return np.asarray(X9, dtype=np.float32) * side[:, None]


def orient_struct33_router_side(X33: np.ndarray, router_long: np.ndarray) -> np.ndarray:
    """Canonical orientation by ROUTER side.

    Router LONG:  backstop=support,           ahead=resistance, liq ahead=up.
    Router SHORT: backstop=resistance,        ahead=support,    liq ahead=down.
    Signed trend fields (indices 0,1,2 per TF) are multiplied by side.
    """
    x = np.asarray(X33, dtype=np.float32)
    assert x.ndim == 2 and x.shape[1] == 33, x.shape
    n = x.shape[0]
    raw = x.reshape(n, 3, 11)
    out = np.empty_like(raw)
    is_long = np.asarray(router_long, dtype=bool)[:, None]
    side = np.where(is_long, 1.0, -1.0).astype(np.float32)

    out[:, :, 0] = raw[:, :, 0] * side
    out[:, :, 1] = raw[:, :, 1] * side
    out[:, :, 2] = raw[:, :, 2] * side

    out[:, :, 3] = np.where(is_long, raw[:, :, 3], raw[:, :, 4])
    out[:, :, 4] = np.where(is_long, raw[:, :, 4], raw[:, :, 3])
    out[:, :, 5] = np.where(is_long, raw[:, :, 5], raw[:, :, 6])
    out[:, :, 6] = np.where(is_long, raw[:, :, 6], raw[:, :, 5])
    out[:, :, 7] = np.where(is_long, raw[:, :, 7], raw[:, :, 8])
    out[:, :, 8] = np.where(is_long, raw[:, :, 8], raw[:, :, 7])
    out[:, :, 9] = np.where(is_long, raw[:, :, 9], raw[:, :, 10])
    out[:, :, 10] = np.where(is_long, raw[:, :, 10], raw[:, :, 9])
    return out.reshape(n, 33)


def router_confidence(p_long: np.ndarray, router_long: np.ndarray) -> np.ndarray:
    """router_p_side in [0.5, 1.0] -- derived router signal, not future information."""
    p_long = np.asarray(p_long, dtype=np.float64)
    router_long = np.asarray(router_long, dtype=bool)
    return np.where(router_long, p_long, 1.0 - p_long)


def build_meta_features(X9: np.ndarray, X33: np.ndarray,
                        router_pred: np.ndarray, p_long: np.ndarray):
    """Return (META10 [N,10], META34 [N,34], router_long bool[N]).

    The correctness target is NOT returned here: it requires Teacher direction and is
    built explicitly at the call site so the label never leaks into X construction.
    """
    router_long = np.asarray(router_pred, dtype=np.int8) >= 1
    X9o = orient_dtp9_router_side(X9, router_long)
    X33o = orient_struct33_router_side(X33, router_long)
    ps = router_confidence(p_long, router_long)
    m10 = np.hstack([X9o, ps.reshape(-1, 1)])
    m34 = np.hstack([X33o, ps.reshape(-1, 1)])
    return m10, m34, router_long


# --------------------------------------------------------------------------- #
# Fixed router + prequential OOF gates                                         #
# --------------------------------------------------------------------------- #
def fit_fixed_router(X, y, w, n_estimators):
    """Frozen router recipe with a FIXED tree count: no early stopping, no tuning."""
    params = dict(BASE_PARAMS)
    params["n_estimators"] = int(n_estimators)
    w = np.asarray(w, dtype=np.float64)
    w = w / w.mean() if w.mean() > 0 else w
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y, sample_weight=w)
    return model


def build_prequential_router_oof(X, y, w, decision_time_ns, train_idx, n_estimators,
                                 return_audit=False):
    """Calendar-quarter expanding/prequential OOF router predictions inside TRAIN.

    The first TRAIN quarter is warm-up only and receives NO fake prediction, so expert
    training gates are never in-sample. Every block satisfies
    max(router-fit decision time) < min(prediction-block decision time).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    w = np.asarray(w, dtype=np.float64)
    t = pd.to_datetime(decision_time_ns).astype("datetime64[ns]")
    pidx = pd.PeriodIndex(t, freq="Q")
    # monotonic integer quarter ordinal avoids any string-compare ordering risk
    qord = (pidx.year * 4 + (pidx.quarter - 1)).to_numpy()
    qname = pidx.astype(str).to_numpy()

    train_idx = np.asarray(train_idx)
    tq = np.unique(qord[train_idx])

    p_long = np.full(len(X), np.nan, dtype=np.float64)
    pred = np.full(len(X), -1, dtype=np.int8)
    audit = []

    # first quarter = warm-up
    for q in tq[1:]:
        pred_idx = train_idx[qord[train_idx] == q]
        fit_idx = train_idx[qord[train_idx] < q]
        assert fit_idx.size > 0, "empty router fit block"
        assert len(np.unique(y[fit_idx])) == 2, "router fit block must be two-class"
        # hard temporal invariant
        assert np.max(decision_time_ns[fit_idx]) < np.min(decision_time_ns[pred_idx]), (
            "OOF_TEMPORAL_VIOLATION")
        m = fit_fixed_router(X[fit_idx], y[fit_idx], w[fit_idx], n_estimators)
        p = m.predict_proba(X[pred_idx])[:, 1]
        p_long[pred_idx] = p
        pred[pred_idx] = (p >= 0.5).astype(np.int8)
        audit.append({
            "quarter": str(qname[pred_idx][0]),
            "n_fit_rows": int(fit_idx.size),
            "n_pred_rows": int(pred_idx.size),
            "fit_max_time": str(pd.to_datetime(np.max(decision_time_ns[fit_idx]))),
            "pred_min_time": str(pd.to_datetime(np.min(decision_time_ns[pred_idx]))),
            "fit_has_both_classes": True,
            "router_long_share": float(np.mean((p >= 0.5))),
        })

    available = pred >= 0
    out = {"p_long": p_long, "pred": pred, "available": available}
    if return_audit:
        out["audit"] = pd.DataFrame(audit)
    return out


def equal_trade_weights(gid: np.ndarray) -> np.ndarray:
    """Equal-opportunity weights: within this population every trade sums to 1."""
    _, inv, counts = np.unique(gid, return_inverse=True, return_counts=True)
    return (1.0 / counts[inv]).astype(np.float64)


# --------------------------------------------------------------------------- #
# Second-stage meta models                                                     #
# --------------------------------------------------------------------------- #
def fit_meta_classifier(Xtr, ytr, wtr, Xv, yv, wv):
    """Frozen binary params + early stopping 100. No tuning, threshold fixed at 0.5."""
    wtr = np.asarray(wtr, dtype=np.float64)
    wtr = wtr / wtr.mean() if wtr.mean() > 0 else wtr
    return fit_direction_model(Xtr, ytr, wtr, Xv, yv, np.asarray(wv, dtype=np.float64))


def fit_shared_corrector(Xtr, ytr, gid_tr, Xv, yv, gid_v):
    wtr = equal_trade_weights(gid_tr)
    wv = equal_trade_weights(gid_v)
    return fit_meta_classifier(Xtr, ytr, wtr, Xv, yv, wv)


def fit_expert_pair(Xtr, correct_tr, gid_tr, router_tr,
                    Xv, correct_v, gid_v, router_v):
    """LongExpert sees ONLY router-LONG rows; ShortExpert ONLY router-SHORT rows."""
    m_long_tr = router_tr == 1
    m_short_tr = ~m_long_tr
    m_long_v = router_v == 1
    m_short_v = ~m_long_v
    # fail closed: each expert must see both correctness classes in TRAIN and VAL
    for nm, m in (("LONG_TRAIN", m_long_tr), ("SHORT_TRAIN", m_short_tr)):
        assert m.sum() > 0, f"empty expert population: {nm}"
        assert len(np.unique(correct_tr[m])) == 2, f"single-class expert target: {nm}"
    for nm, m in (("LONG_VAL", m_long_v), ("SHORT_VAL", m_short_v)):
        assert m.sum() > 0, f"empty expert population: {nm}"
        assert len(np.unique(correct_v[m])) == 2, f"single-class expert target: {nm}"

    long_model = fit_shared_corrector(Xtr[m_long_tr], correct_tr[m_long_tr], gid_tr[m_long_tr],
                                      Xv[m_long_v], correct_v[m_long_v], gid_v[m_long_v])
    short_model = fit_shared_corrector(Xtr[m_short_tr], correct_tr[m_short_tr], gid_tr[m_short_tr],
                                       Xv[m_short_v], correct_v[m_short_v], gid_v[m_short_v])
    return long_model, short_model


def apply_shared_corrector(router_pred: np.ndarray, p_correct: np.ndarray) -> np.ndarray:
    final = np.asarray(router_pred, dtype=np.int8).copy()
    final[p_correct < 0.5] = 1 - final[p_correct < 0.5]
    return final.astype(np.uint8)


def predict_experts(long_model, short_model, X, router_pred):
    p_correct = np.empty(len(router_pred), dtype=np.float64)
    m_long = np.asarray(router_pred, dtype=np.int8) == 1
    m_short = ~m_long
    if m_long.any():
        p_correct[m_long] = long_model.predict_proba(X[m_long])[:, 1]
    if m_short.any():
        p_correct[m_short] = short_model.predict_proba(X[m_short])[:, 1]
    final = apply_shared_corrector(router_pred, p_correct)
    return final, p_correct


# --------------------------------------------------------------------------- #
# Evaluation helpers                                                           #
# --------------------------------------------------------------------------- #
def _weighted_class_metrics(pred, y, w):
    y = np.asarray(y, dtype=np.uint8)
    pred = np.asarray(pred, dtype=np.uint8)
    w = np.asarray(w, dtype=np.float64)
    long_mask = y == 1
    short_mask = y == 0
    long_recall = (float(np.average(pred[long_mask] == 1, weights=w[long_mask]))
                   if long_mask.any() else None)
    short_recall = (float(np.average(pred[short_mask] == 0, weights=w[short_mask]))
                    if short_mask.any() else None)
    bal = (0.5 * (long_recall + short_recall)
           if (long_mask.any() and short_mask.any()) else None)
    return {
        "accuracy": float(np.average(pred == y, weights=w)),
        "balanced_accuracy": bal,
        "long_recall": long_recall,
        "short_recall": short_recall,
        "pred_long_share": float(np.average(pred == 1, weights=w)),
    }


def _trade_returns(data: DirectionExpertData, idx, pred):
    pdr = pred_direction_return_atr(
        np.asarray(pred, dtype=np.uint8), data.y[idx], data.entry_quality_atr[idx])
    return aggregate_per_trade(data.gid[idx], data.w[idx], pdr)


def _repair_damage(router_pred, final_pred, y, w, mask=None):
    router_pred = np.asarray(router_pred, dtype=np.uint8)
    final_pred = np.asarray(final_pred, dtype=np.uint8)
    y = np.asarray(y, dtype=np.uint8)
    w = np.asarray(w, dtype=np.float64)
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    if not mask.any():
        return {"repair_rate": None, "damage_rate": None, "flip_rate": None, "n_rows": 0}
    rp, fp, yy, ww = router_pred[mask], final_pred[mask], y[mask], w[mask]
    a_correct = rp == yy
    n_correct = fp == yy
    repair = (~a_correct) & n_correct
    damage = a_correct & (~n_correct)
    flip = fp != rp
    denom_wrong = float(ww[~a_correct].sum())
    denom_right = float(ww[a_correct].sum())
    tot = float(ww.sum())
    return {
        "n_rows": int(mask.sum()),
        "repair_rate": float(ww[repair].sum() / denom_wrong) if denom_wrong > 0 else None,
        "damage_rate": float(ww[damage].sum() / denom_right) if denom_right > 0 else None,
        "flip_rate": float(ww[flip].sum() / tot) if tot > 0 else None,
    }


def evaluate_system(data: DirectionExpertData, idx, final_pred, router_pred,
                    p_score=None, auc_scope="router_correctness",
                    with_phase=False, ds=None):
    y = data.y[idx]
    w = data.w[idx]
    tr, uniq = _trade_returns(data, idx, final_pred)
    mean_ret, lo, hi = bootstrap_trade_returns_chunked(tr)
    clf = _weighted_class_metrics(final_pred, y, w)
    out = {
        "n_rows": int(idx.size),
        "n_trades": int(tr.size),
        "return_atr": mean_ret, "ci_low": lo, "ci_high": hi,
        "accuracy": clf["accuracy"],
        "balanced_accuracy": clf["balanced_accuracy"],
        "long_recall": clf["long_recall"],
        "short_recall": clf["short_recall"],
        "pred_long_share": clf["pred_long_share"],
        "repair_damage": {"ALL": _repair_damage(router_pred, final_pred, y, w)},
    }
    if p_score is not None:
        # A: standard directional AUC (label = Teacher direction).
        # M/E: no coherent final pLong exists after keep/flip, so the AUC of the
        # correctness model is reported separately instead of inventing one.
        auc_label = (y if auc_scope == "direction"
                     else (np.asarray(router_pred, dtype=np.uint8) == y).astype(np.uint8))
        try:
            out["roc_auc"] = float(roc_auc_score(
                auc_label, np.asarray(p_score, dtype=np.float64), sample_weight=w))
        except Exception:
            out["roc_auc"] = None
        out["roc_auc_scope"] = auc_scope
    # Teacher-side economic decomposition
    long_mask = y == 1
    short_mask = y == 0
    for nm, m in (("LONG", long_mask), ("SHORT", short_mask)):
        if m.any():
            sub_tr, _ = aggregate_per_trade(
                data.gid[idx][m], w[m],
                pred_direction_return_atr(
                    np.asarray(final_pred, dtype=np.uint8)[m], y[m],
                    data.entry_quality_atr[idx][m]))
            out[f"{nm.lower()}_return"] = float(sub_tr.mean()) if sub_tr.size else None
            out["repair_damage"][f"TEACHER_{nm}"] = _repair_damage(
                router_pred, final_pred, y, w, m)
        else:
            out[f"{nm.lower()}_return"] = None
    # Router-gate decomposition (descriptive)
    for nm, gv in (("ROUTER_LONG", 1), ("ROUTER_SHORT", 0)):
        m = np.asarray(router_pred, dtype=np.uint8) == gv
        if m.any():
            gate_tr, _ = aggregate_per_trade(
                data.gid[idx][m], w[m],
                pred_direction_return_atr(
                    np.asarray(final_pred, dtype=np.uint8)[m], y[m],
                    data.entry_quality_atr[idx][m]))
            out["repair_damage"][nm] = _repair_damage(router_pred, final_pred, y, w, m)
            out["repair_damage"][nm]["n_opportunities"] = int(gate_tr.size)
        else:
            out["repair_damage"][nm] = None
    if with_phase and ds is not None:
        phases = {}
        for ph in ("BEFORE_ENTRY", "AT_ENTRY", "IN_POSITION"):
            pm = phase_mask(ds, idx, ph)
            phases[ph] = (evaluate_system(data, idx[pm], final_pred[pm],
                                          router_pred[pm]) if pm.any() else None)
        out["phase"] = phases
    return out


def _paired(pairs, n_trades):
    return dict(zip(("mean", "ci_low", "ci_high"), bootstrap_trade_returns_chunked(pairs)))


def _paired_side(tr_map, key_a, key_b, teacher_mask=None):
    """Paired contrast restricted to a Teacher-side trade subset."""
    ta = np.asarray(tr_map[key_a], dtype=np.float64)
    tb = np.asarray(tr_map[key_b], dtype=np.float64)
    if teacher_mask is not None:
        ta, tb = ta[teacher_mask], tb[teacher_mask]
    return dict(zip(("mean", "ci_low", "ci_high"), bootstrap_trade_returns_chunked(tb - ta)))


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
# One full chain: router A -> fixed router -> OOF gates -> M/E -> TEST          #
# --------------------------------------------------------------------------- #
def run_chain(data: DirectionExpertData, ds: pd.DataFrame,
              train_idx, val_idx, test_idx, n_estimators=None):
    """Fit the frozen router + all meta systems and return TEST predictions.

    Used identically for the pooled fit and every LOSO fold so that held-out symbols
    never enter Router TRAIN/VAL, OOF router TRAIN, or Expert TRAIN/VAL.
    """
    # ---- frozen router A (reproduction of the already-frozen baseline) ----
    m0 = _fit_m0(ds, train_idx, val_idx)
    if n_estimators is None:
        n_estimators = int(m0.best_iteration_)
    pred_a, p_a = _predict_m0(m0, ds, test_idx)

    # ---- fixed router: full train/<fold-train>, frozen tree count ----
    # Use the FROZEN prepare_xy extraction (same DataFrame representation A was trained
    # on) rather than a re-cast numpy array, so the fitted model is not perturbed by
    # dtype differences. Meta features below use the numpy float32 arrays instead.
    Xtr9, ytr, wtr = prepare_xy(ds, train_idx, DTP9)
    router = fit_fixed_router(Xtr9, ytr, wtr, n_estimators)

    Xte9, _, _ = prepare_xy(ds, test_idx, DTP9)
    Xva9, _, _ = prepare_xy(ds, val_idx, DTP9)
    p_te = router.predict_proba(Xte9)[:, 1]
    router_te = (p_te >= 0.5).astype(np.int8)
    p_va = router.predict_proba(Xva9)[:, 1]
    router_va = (p_va >= 0.5).astype(np.int8)

    # fail closed: fixed router must reproduce the frozen A router on TEST.
    # LightGBM multi-threaded training is reproducible only to ~1e-5 in probability, so
    # the gate requires (a) IDENTICAL hard predictions and (b) a tight probability
    # tolerance. Semantic drift would change hard labels far beyond this budget; the
    # frozen economic result is asserted downstream as the real reproduction criterion.
    p_a = np.asarray(p_a, dtype=np.float64)
    max_dev = float(np.max(np.abs(p_te - p_a)))
    if not np.array_equal(router_te, np.asarray(pred_a, dtype=np.int8)):
        raise RuntimeError("STOP_FIXED_ROUTER_A_PREDICTION_MISMATCH")
    if max_dev > 1e-4:
        raise RuntimeError(f"STOP_FIXED_ROUTER_A_MISMATCH: max_dev={max_dev}")

    # ---- prequential OOF gates inside TRAIN (expert gates are never in-sample) ----
    oof = build_prequential_router_oof(
        data.X9, data.y, data.w, data.decision_time_ns, train_idx, n_estimators)
    # oof arrays are FULL-length; restrict to this fold's TRAIN rows first
    avail_tr = oof["available"][train_idx]
    tr_gate = train_idx[avail_tr]
    oof_pred = oof["pred"][tr_gate]
    oof_p = oof["p_long"][tr_gate]

    # ---- meta feature matrices (built once each, vectorized) ----
    m10_te, m34_te, _ = build_meta_features(
        data.X9[test_idx], data.X33[test_idx], router_te, p_te)
    m10_tr, m34_tr, _ = build_meta_features(
        data.X9[tr_gate], data.X33[tr_gate], oof_pred, oof_p)
    m10_va, m34_va, _ = build_meta_features(
        data.X9[val_idx], data.X33[val_idx], router_va, p_va)

    gid_tr = data.gid[tr_gate]
    gid_va = data.gid[val_idx]

    # ---- META training target: teacher may appear ONLY as this label ----
    y_meta_tr = (np.asarray(oof_pred, dtype=np.int8) == data.y[tr_gate]).astype(np.uint8)
    y_meta_va = (np.asarray(router_va, dtype=np.int8) == data.y[val_idx]).astype(np.uint8)

    # shared correctors
    m9 = fit_shared_corrector(m10_tr, y_meta_tr, gid_tr, m10_va, y_meta_va, gid_va)
    m33 = fit_shared_corrector(m34_tr, y_meta_tr, gid_tr, m34_va, y_meta_va, gid_va)

    # gated experts
    e9 = fit_expert_pair(m10_tr, y_meta_tr, gid_tr, oof_pred,
                         m10_va, y_meta_va, gid_va, router_va)
    e33 = fit_expert_pair(m34_tr, y_meta_tr, gid_tr, oof_pred,
                          m34_va, y_meta_va, gid_va, router_va)

    # ---- TEST inference ----
    p9 = m9.predict_proba(m10_te)[:, 1]
    p33 = m33.predict_proba(m34_te)[:, 1]
    fin_m9 = apply_shared_corrector(router_te, p9)
    fin_m33 = apply_shared_corrector(router_te, p33)
    fin_e9, pe9 = predict_experts(e9[0], e9[1], m10_te, router_te)
    fin_e33, pe33 = predict_experts(e33[0], e33[1], m34_te, router_te)

    return {
        "n_estimators": int(n_estimators),
        "A": np.asarray(pred_a, dtype=np.uint8),
        "M9": fin_m9, "M33": fin_m33, "E9": fin_e9, "E33": fin_e33,
        "p_score": {"M9": p9, "M33": p33, "E9": pe9, "E33": pe33},
        "router_te": router_te, "p_te": p_te,
        "A_p": np.asarray(p_a, dtype=np.float64),
        "models": {"router": router, "m9": m9, "m33": m33,
                   "e9": e9, "e33": e33},
        "oof": {"available": oof["available"], "pred": oof["pred"], "p_long": oof["p_long"],
                "tr_gate": tr_gate},
    }


def _oof_audit(data: DirectionExpertData, train_idx, oof):
    avail = oof["available"]
    tr_mask = avail[train_idx]
    idx = train_idx[tr_mask]
    y = data.y[idx]
    w = data.w[idx]
    pred = oof["pred"][idx]
    long_mask = y == 1
    short_mask = y == 0
    correct = (pred == data.y[idx]).astype(np.uint8)
    return {
        "train_rows": int(train_idx.size),
        "oof_eligible_rows": int(idx.size),
        "oof_coverage_fraction": float(idx.size / train_idx.size),
        "router_long_share": float(np.average(pred == 1, weights=w)),
        "router_short_share": float(np.average(pred == 0, weights=w)),
        "oof_weighted_accuracy": float(np.average(correct == 1, weights=w)),
        "oof_long_recall": (float(np.average(pred[long_mask] == 1, weights=w[long_mask]))
                            if long_mask.any() else None),
        "oof_short_recall": (float(np.average(pred[short_mask] == 0, weights=w[short_mask]))
                             if short_mask.any() else None),
    }


# --------------------------------------------------------------------------- #
# Main experiment                                                              #
# --------------------------------------------------------------------------- #
def run_gated_experts(save: bool = True, verbose: bool = True):
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

    data = build_direction_expert_data(ds)
    n_rows = int(test_idx.size)
    n_trades = int(len(np.unique(data.gid[test_idx])))
    n_long = int(np.unique(data.gid[test_idx][data.y[test_idx] == 1]).size)
    n_short = int(np.unique(data.gid[test_idx][data.y[test_idx] == 0]).size)
    if (n_rows != A_REFERENCE["test_rows"] or n_trades != A_REFERENCE["test_trades"]
            or n_long != A_REFERENCE["long_trades"] or n_short != A_REFERENCE["short_trades"]):
        raise RuntimeError(
            f"STOP_GE_SPLIT_DRIFT: rows={n_rows} trades={n_trades} "
            f"long={n_long} short={n_short}")

    log("fit pooled chain (A + M9/E9/M33/E33) ...")
    chain = run_chain(data, ds, train_idx, val_idx, test_idx)
    n_est = chain["n_estimators"]
    log(f"  router best_iteration = {n_est}")

    # ---- frozen A reproduction gate ----
    oof_audit = _oof_audit(data, train_idx, chain["oof"])
    oof_df = build_prequential_router_oof(
        data.X9, data.y, data.w, data.decision_time_ns, train_idx, n_est,
        return_audit=True)["audit"]

    preds = {"A": chain["A"], "M9": chain["M9"], "E9": chain["E9"],
             "M33": chain["M33"], "E33": chain["E33"]}
    pscores = {"A": chain["A_p"],
               "M9": chain["p_score"]["M9"], "E9": chain["p_score"]["E9"],
               "M33": chain["p_score"]["M33"], "E33": chain["p_score"]["E33"]}
    auc_scopes = {"A": "direction", "M9": "router_correctness",
                  "E9": "router_correctness", "M33": "router_correctness",
                  "E33": "router_correctness"}

    pooled = {}
    for s in SYSTEMS:
        pooled[s] = evaluate_system(
            data, test_idx, preds[s], chain["router_te"],
            p_score=pscores.get(s), auc_scope=auc_scopes[s], with_phase=True, ds=ds)

    # fail closed: A must reproduce every frozen number
    a = pooled["A"]
    for k in ("return_atr", "accuracy", "roc_auc", "long_recall", "short_recall",
              "pred_long_share", "long_return", "short_return"):
        got = a.get(k)
        assert got is not None, f"STOP_GE_A_METRIC_MISSING:{k}"
        if abs(got - A_REFERENCE[k]) > 1e-6:
            raise RuntimeError(f"STOP_GE_A_REPRODUCTION_FAILED:{k} {got} != {A_REFERENCE[k]}")

    pooled_ex_ag = {}
    sym_te = data.symbol[test_idx]
    ex_mask = sym_te != "AG"
    for s in SYSTEMS:
        pooled_ex_ag[s] = evaluate_system(
            data, test_idx[ex_mask], preds[s][ex_mask], chain["router_te"][ex_mask],
            p_score=pscores[s][ex_mask], auc_scope=auc_scopes[s])

    # ---- aligned per-trade returns + paired contrasts ----
    tr_map = {s: _trade_returns(data, test_idx, preds[s])[0] for s in SYSTEMS}
    _, uniq = _trade_returns(data, test_idx, preds["A"])
    teacher_long_trade = (np.bincount(
        np.searchsorted(uniq, data.gid[test_idx]),
        weights=(data.y[test_idx] == 1).astype(np.float64), minlength=len(uniq)) > 0)

    contrasts = {
        "E9_minus_A": _paired(tr_map["E9"] - tr_map["A"], n_trades),
        "E9_minus_M9": _paired(tr_map["E9"] - tr_map["M9"], n_trades),
        "E33_minus_E9": _paired(tr_map["E33"] - tr_map["E9"], n_trades),
        "E33_minus_M33": _paired(tr_map["E33"] - tr_map["M33"], n_trades),
        "M9_minus_A": _paired(tr_map["M9"] - tr_map["A"], n_trades),
        "M33_minus_M9": _paired(tr_map["M33"] - tr_map["M9"], n_trades),
        "E33_minus_A": _paired(tr_map["E33"] - tr_map["A"], n_trades),
    }
    primary = ("E9_minus_A", "E9_minus_M9", "E33_minus_E9", "E33_minus_M33")
    side_contrasts = {}
    for c in contrasts:
        ka, kb = c.split("_minus_")
        side_contrasts[c] = {
            "LONG": _paired_side(tr_map, ka, kb, teacher_long_trade),
            "SHORT": _paired_side(tr_map, ka, kb, ~teacher_long_trade),
        }

    # ---- per-symbol robustness ----
    log("per-symbol ...")
    per_symbol = []
    sym_lists = {c: [] for c in contrasts}
    for s in SYMBOLS:
        m = data.symbol[test_idx] == s
        sub = test_idx[m]
        if sub.size == 0:
            continue
        sub_tr = {}
        for sysname in SYSTEMS:
            sub_tr[sysname] = evaluate_system(
                data, sub, preds[sysname][m], chain["router_te"][m],
                p_score=pscores[sysname][m], auc_scope=auc_scopes[sysname])
        row = {"symbol": s, "n_trades": int(np.unique(data.gid[sub]).size)}
        for sysname in SYSTEMS:
            row[f"{sysname}_return"] = sub_tr[sysname]["return_atr"]
            row[f"{sysname}_long_return"] = sub_tr[sysname]["long_return"]
            row[f"{sysname}_short_return"] = sub_tr[sysname]["short_return"]
            row[f"{sysname}_long_recall"] = sub_tr[sysname]["long_recall"]
            row[f"{sysname}_short_recall"] = sub_tr[sysname]["short_recall"]
        # per-symbol paired deltas use that symbol's own trades
        sub_tr_map = {k: (aggregate_per_trade(
            data.gid[sub], data.w[sub],
            pred_direction_return_atr(
                np.asarray(preds[k], dtype=np.uint8)[m], data.y[sub],
                data.entry_quality_atr[sub]))[0]) for k in SYSTEMS}
        for c in contrasts:
            ka, kb = c.split("_minus_")
            d = float(np.mean(sub_tr_map[kb] - sub_tr_map[ka]))
            row[f"{c}_mean"] = d
            sym_lists[c].append(d)
        per_symbol.append(row)

    def _cluster(vals):
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            return {"mean": None, "ci_low": None, "ci_high": None,
                    "positive_symbols": 0, "negative_symbols": 0}
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        means = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
        for b in range(BOOTSTRAP_REPLICATES):
            means[b] = vals[rng.integers(0, vals.size, size=vals.size)].mean()
        return {
            "mean": float(vals.mean()),
            "ci_low": float(np.quantile(means, 0.025)),
            "ci_high": float(np.quantile(means, 0.975)),
            "positive_symbols": int((vals > 0).sum()),
            "negative_symbols": int((vals < 0).sum()),
        }

    per_symbol_cluster = {c: _cluster(sym_lists[c]) for c in primary}

    # ---- LOSO (end to end, no held-out leakage) ----
    log("LOSO ...")
    loso = []
    loso_lists = {c: [] for c in primary}
    for s in SYMBOLS:
        tr_s = train_idx[data.symbol[train_idx] != s]
        va_s = val_idx[data.symbol[val_idx] != s]
        te_s = test_idx[data.symbol[test_idx] == s]
        if te_s.size == 0:
            continue
        ch = run_chain(data, ds, tr_s, va_s, te_s)
        preds_s = {"A": ch["A"], "M9": ch["M9"], "E9": ch["E9"],
                   "M33": ch["M33"], "E33": ch["E33"]}
        ev = {}
        trs = {}
        pscores_s = {"A": ch["A_p"], **ch["p_score"]}
        for sysname in SYSTEMS:
            ev[sysname] = evaluate_system(
                data, te_s, preds_s[sysname], ch["router_te"],
                p_score=pscores_s.get(sysname), auc_scope=auc_scopes[sysname])
            trs[sysname] = _trade_returns(data, te_s, preds_s[sysname])[0]
        row = {"held_out_symbol": s,
               "router_best_iteration": ch["n_estimators"],
               "n_trades": ev["A"]["n_trades"]}
        for sysname in SYSTEMS:
            row[f"{sysname}_return"] = ev[sysname]["return_atr"]
            row[f"{sysname}_long_return"] = ev[sysname]["long_return"]
            row[f"{sysname}_short_return"] = ev[sysname]["short_return"]
        for c in primary:
            ka, kb = c.split("_minus_")
            d = float(np.mean(trs[kb] - trs[ka]))
            row[f"{c}_mean"] = d
            loso_lists[c].append(d)
        loso.append(row)
        log(f"  LOSO held-out {s}: A={row['A_return']:.3f} E9={row['E9_return']:.3f} "
            f"E33={row['E33_return']:.3f} E9-A={row['E9_minus_A_mean']:+.3f}")

    loso_agg = {c: _cluster(loso_lists[c]) for c in primary}

    # ---- time-block diagnostic (monthly, no retraining) ----
    months = pd.PeriodIndex(
        pd.to_datetime(data.oef_time_ns[test_idx]).astype("datetime64[ns]"),
        freq="M").astype(str).to_numpy()
    gid_te = data.gid[test_idx]
    mb = pd.DataFrame({"gid": gid_te, "month": months}).drop_duplicates()
    if not mb["gid"].is_unique:
        raise RuntimeError("STOP_GE_TIME_BLOCK_GID_IN_TWO_MONTHS")
    uniq_sorted, inverse = np.unique(gid_te, return_inverse=True)
    month_of_trade = np.empty(len(uniq_sorted), dtype=object)
    g2m = dict(zip(mb["gid"], mb["month"]))
    for g, i in zip(uniq_sorted, range(len(uniq_sorted))):
        month_of_trade[i] = g2m[g]
    rows_tb = []
    for mon in sorted(set(month_of_trade.tolist())):
        sel = month_of_trade == mon
        rows_tb.append({
            "month": mon,
            "n_opportunities": int(sel.sum()),
            "A": float(tr_map["A"][sel].mean()),
            "E9": float(tr_map["E9"][sel].mean()),
            "E33": float(tr_map["E33"][sel].mean()),
            "E9_minus_A": float((tr_map["E9"] - tr_map["A"])[sel].mean()),
            "E33_minus_E9": float((tr_map["E33"] - tr_map["E9"])[sel].mean()),
        })
    time_blocks = pd.DataFrame(rows_tb)

    # ---- verdict ----
    verdict = {}
    for key, name in (("expert_total_increment", "E9_minus_A"),
                      ("expert_split_increment", "E9_minus_M9"),
                      ("struct33_within_experts_increment", "E33_minus_E9"),
                      ("struct33_specialist_increment", "E33_minus_M33")):
        c = contrasts[name]
        if c["ci_low"] > 0:
            state = "supported_diagnostically"
        elif c["ci_high"] < 0:
            state = "harmful_diagnostically"
        else:
            state = "no_identifiable_increment"
        verdict[key] = {
            "contrast": name,
            "state": state,
            "mean": c["mean"], "ci": [c["ci_low"], c["ci_high"]],
            "LONG": side_contrasts[name]["LONG"],
            "SHORT": side_contrasts[name]["SHORT"],
            "LONG_state": ("supported_diagnostically"
                           if side_contrasts[name]["LONG"]["ci_low"] > 0
                           else "harmful_diagnostically"
                           if side_contrasts[name]["LONG"]["ci_high"] < 0
                           else "no_identifiable_increment"),
            "SHORT_state": ("supported_diagnostically"
                            if side_contrasts[name]["SHORT"]["ci_low"] > 0
                            else "harmful_diagnostically"
                            if side_contrasts[name]["SHORT"]["ci_high"] < 0
                            else "no_identifiable_increment"),
        }

    summary = _clean({
        "task_id": TASK_ID,
        "base_sha": BASE_SHA,
        "test_status": TEST_STATUS,
        "metric": METRIC_NAME,
        "model_roles": {
            "A": "frozen_direct_dtp9_router",
            "M9": "shared_router_correctness_META10",
            "E9": "gated_long_short_experts_META10",
            "M33": "shared_router_correctness_META34",
            "E33": "gated_long_short_experts_META34",
        },
        "frozen_split": {
            "test_rows": n_rows, "test_trades": n_trades,
            "long_trades": n_long, "short_trades": n_short,
            "train_rows": int(train_idx.size), "val_rows": int(val_idx.size),
        },
        "A_reproduction": {
            "return_atr": a["return_atr"], "expected": A_REFERENCE["return_atr"],
            "max_abs_dev_prob_fixed_router": float(
                np.max(np.abs(chain["p_te"] - chain["A_p"]))),
            "router_hard_prediction_match": bool(np.array_equal(
                chain["router_te"], chain["A"].astype(np.int8))),
            "match": True,
        },
        "router": {
            "best_iteration": int(n_est),
            "n_estimators_fixed": int(n_est),
            "oof_audit": oof_audit,
        },
        "features": {
            "META10_cols": META10_COLS,
            "META34_cols": META34_COLS,
            "orientation_source": "router_prediction (NOT Teacher)",
            "n_meta10": len(META10_COLS),
            "n_meta34": len(META34_COLS),
        },
        "contract": {
            "threshold": 0.5,
            "no_threshold_tuning": True,
            "no_hyperparameter_tuning": True,
            "teacher_direction_is_label_only": True,
            "prequential_quarter_oof_gates": True,
            "equal_opportunity_meta_weights": True,
            "final_evaluation_uses_sample_weight_raw": True,
        },
        "pooled": pooled,
        "pooled_ex_ag": pooled_ex_ag,
        "paired_contrasts": contrasts,
        "side_contrasts": side_contrasts,
        "per_symbol": per_symbol,
        "per_symbol_cluster_bootstrap": per_symbol_cluster,
        "loso": {"folds": loso, "aggregate": loso_agg},
        "time_blocks": time_blocks.to_dict(orient="records"),
        "verdict": verdict,
        "provenance": {
            "base_sha": BASE_SHA,
            "model_architecture_experiment": True,
            "upstream_changed": False,
            "feature_schema_changed": False,
            "hyperparameter_tuning": False,
            "threshold_tuning": False,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "note": ("Two-stage gated Direction architecture. Router = frozen DTP9 M0. "
                     "Second stage judges only whether the router thesis is correct; "
                     "Teacher direction is the label and never orients or routes."),
        },
    })

    if save:
        os.makedirs(EVIDENCE_DIR, exist_ok=True)
        with open(SUMMARY_JSON, "w") as f:
            json.dump(summary, f, indent=2)
        _write_per_symbol_csv(per_symbol, PER_SYMBOL_CSV)
        _write_trade_returns_csv(uniq, teacher_long_trade, tr_map, TRADE_RETURNS_CSV)
        _write_loso_csv(loso, LOSO_CSV)
        time_blocks.to_csv(TIME_BLOCKS_CSV, index=False)
        oof_df.to_csv(OOF_AUDIT_CSV, index=False)
        log(f"evidence written -> {SUMMARY_JSON}")

    return {
        "summary": summary,
        "paths": {
            "summary": SUMMARY_JSON, "per_symbol_csv": PER_SYMBOL_CSV,
            "trade_returns_csv": TRADE_RETURNS_CSV, "loso_csv": LOSO_CSV,
            "time_blocks_csv": TIME_BLOCKS_CSV, "oof_audit_csv": OOF_AUDIT_CSV,
        },
    }


def _write_per_symbol_csv(per_symbol, path):
    base = ["symbol", "n_trades"]
    for s in SYSTEMS:
        base += [f"{s}_return", f"{s}_long_return", f"{s}_short_return",
                 f"{s}_long_recall", f"{s}_short_recall"]
    contrasts_cols = ["E9_minus_A_mean", "E9_minus_M9_mean",
                      "E33_minus_E9_mean", "E33_minus_M33_mean",
                      "M9_minus_A_mean", "M33_minus_M9_mean", "E33_minus_A_mean"]
    df = pd.DataFrame(per_symbol)
    cols = [c for c in base + contrasts_cols if c in df.columns]
    df[cols].to_csv(path, index=False)


def _write_trade_returns_csv(uniq, teacher_long, tr_map, path):
    df = pd.DataFrame({"gid": uniq,
                       "teacher_direction": np.where(teacher_long, "LONG", "SHORT")})
    for s in SYSTEMS:
        df[f"tr_{s}"] = tr_map[s]
    for ka, kb in (("A", "E9"), ("M9", "E9"), ("E9", "E33"), ("M33", "E33"),
                   ("A", "M9"), ("M9", "M33"), ("A", "E33")):
        df[f"delta_{kb}_minus_{ka}"] = tr_map[kb] - tr_map[ka]
    df.to_csv(path, index=False)


def _write_loso_csv(loso, path):
    base = ["held_out_symbol", "router_best_iteration", "n_trades"]
    for s in SYSTEMS:
        base += [f"{s}_return", f"{s}_long_return", f"{s}_short_return"]
    contrasts_cols = ["E9_minus_A_mean", "E9_minus_M9_mean",
                      "E33_minus_E9_mean", "E33_minus_M33_mean"]
    df = pd.DataFrame(loso)
    cols = [c for c in base + contrasts_cols if c in df.columns]
    df[cols].to_csv(path, index=False)


if __name__ == "__main__":
    run_gated_experts(save=True, verbose=True)
