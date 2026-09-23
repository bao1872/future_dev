"""entry_quality_v1
===================

FUTURE-R4-M15-ENTRY-QUALITY-V1

Conditional Entry-Quality ranking diagnostic.

Business questions (given direction is assumed correct):
  1. Can the current observable market state predict which Candidate has better
     remaining entry value (RemainingEdgeATR)?
  2. Does STRUCT33 (SR / Liquidity) add Entry-Quality information beyond DTP9?
  3. Do LONG and SHORT Entry-Quality mappings benefit from separate side-specific
     regression functions?

This is a CONDITIONAL ENTRY-QUALITY RESEARCH DIAGNOSTIC, NOT a deployable
end-to-end strategy. Teacher direction is used ONLY to:
  - orient the feature coordinate system (Long/Short put in the same frame), and
  - route Q2 specialists during this diagnostic.

A later integration experiment must replace Teacher direction with a frozen
Direction-model prediction before any deployable interpretation.

No environment / Candidate / Teacher / Dataset rebuild. The frozen 15-symbol
Phase1 dataset (loaded via build_frozen_split) already carries STRUCT33, the
price fields, bars fields and weights. BASE_PARAMS are reused; only the objective
changes to regression_l1. No tuning, no target clipping, no ENTER/SKIP threshold.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import kendalltau

from research.liquidity_oracle_atlas.train_direction_model_15sym_v1 import (
    BASE_PARAMS,
    SYMBOLS,
    verify_manifest,
)
from research.liquidity_oracle_atlas.direction_null_baseline_v1 import (
    build_frozen_split,
)

TASK_ID = "FUTURE-R4-M15-ENTRY-QUALITY-V1"
# ABC FIX1 commit this diagnostic is built on (its BASE_SHA).
BASE_SHA = "0ee10dc2cf7fc76510181008ed6c306c1de38088"

TEST_STATUS = "conditional_entry_quality_diagnostic"
# Frozen TEST universe (drift guard, must equal the ABC experiment).
FROZEN_TEST = {"test_rows": 13773, "test_trades": 638}

EQ_BOOTSTRAP_REPLICATES = 2000
EQ_BOOTSTRAP_SEED = 20260923

EVIDENCE_DIR = os.path.join("research", "liquidity_oracle_atlas", "evidence")
SUMMARY_JSON = os.path.join(EVIDENCE_DIR, "entry_quality_v1_summary.json")
PER_SYMBOL_CSV = os.path.join(EVIDENCE_DIR, "entry_quality_v1_per_symbol.csv")
TRADE_METRICS_CSV = os.path.join(EVIDENCE_DIR, "entry_quality_v1_trade_metrics.csv")
LOSO_CSV = os.path.join(EVIDENCE_DIR, "entry_quality_v1_loso.csv")
LABEL_AUDIT_CSV = os.path.join(EVIDENCE_DIR, "entry_quality_v1_label_audit.csv")

# --------------------------------------------------------------------------- #
# Canonical feature schema                                                    #
# --------------------------------------------------------------------------- #
TF_ORDER = ("m15", "h1", "h4")

# Raw STRUCT33 field order per TF (matches the frozen parquet column order and the
# orientation logic in orient_struct33: 0 trend, 1 slope, 2 dev, 3 support_dist,
# 4 resistance_dist, 5 support_strength, 6 resistance_strength, 7 liq_up, 8 liq_down,
# 9 liq_up_count, 10 liq_down_count).
RAW_PER_TF = (
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
RAW_STRUCT33_COLS = [f"{tf}_{c}" for tf in TF_ORDER for c in RAW_PER_TF]

# Canonical oriented EQ names (documentation / tests only; X is a numpy array).
EQ_PER_TF = (
    "trend_with_side",
    "slope_with_side_atr",
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
EQ33_COLS = [f"{tf}_{c}" for tf in TF_ORDER for c in EQ_PER_TF]
EQ9_COLS = [f"{tf}_{c}" for tf in TF_ORDER for c in EQ_PER_TF[:3]]


# --------------------------------------------------------------------------- #
# Direction-canonical feature orientation (NumPy-vectorized, no row loops)     #
# --------------------------------------------------------------------------- #
def orient_struct33(raw_struct33: np.ndarray, teacher_is_long: np.ndarray):
    """raw_struct33: [N,33] in frozen STRUCT33 order. teacher_is_long: [N] bool.

    Diagnostic only: orientation uses Teacher direction. Returns (eq9 [N,9],
    eq33 [N,33]); NaN values are preserved; no Candidate-row Python loop.
    """
    x = np.asarray(raw_struct33, dtype=np.float32)
    assert x.ndim == 2 and x.shape[1] == 33

    n = x.shape[0]
    raw = x.reshape(n, 3, 11)
    out = np.empty_like(raw)

    is_long = np.asarray(teacher_is_long, dtype=bool)[:, None]
    side = np.where(is_long, 1.0, -1.0).astype(np.float32)

    # signed trend fields
    out[:, :, 0] = raw[:, :, 0] * side
    out[:, :, 1] = raw[:, :, 1] * side
    out[:, :, 2] = raw[:, :, 2] * side

    # backstop / ahead SR (support<->resistance swap on SHORT)
    out[:, :, 3] = np.where(is_long, raw[:, :, 3], raw[:, :, 4])
    out[:, :, 4] = np.where(is_long, raw[:, :, 4], raw[:, :, 3])

    out[:, :, 5] = np.where(is_long, raw[:, :, 5], raw[:, :, 6])
    out[:, :, 6] = np.where(is_long, raw[:, :, 6], raw[:, :, 5])

    # liquidity ahead / behind (liq_up <-> liq_down swap on SHORT)
    out[:, :, 7] = np.where(is_long, raw[:, :, 7], raw[:, :, 8])
    out[:, :, 8] = np.where(is_long, raw[:, :, 8], raw[:, :, 7])

    out[:, :, 9] = np.where(is_long, raw[:, :, 9], raw[:, :, 10])
    out[:, :, 10] = np.where(is_long, raw[:, :, 10], raw[:, :, 9])

    eq33 = out.reshape(n, 33)
    eq9 = out[:, :, :3].reshape(n, 9)
    return eq9, eq33


# --------------------------------------------------------------------------- #
# Labels (vectorized)                                                         #
# --------------------------------------------------------------------------- #
def build_eq_labels(oracle_direction, candidate_fill_price,
                    oracle_entry_fill_price, oracle_exit_fill_price):
    """RemainingEdgeATR is provided separately (entry_quality_atr). This builds the
    audit-only labels RemainingFraction / MissedFraction. No clipping."""
    is_long = np.asarray(oracle_direction) == "LONG"
    side = np.where(is_long, 1.0, -1.0)
    cand = np.asarray(candidate_fill_price, dtype=np.float64)
    ent = np.asarray(oracle_entry_fill_price, dtype=np.float64)
    ext = np.asarray(oracle_exit_fill_price, dtype=np.float64)

    total_move = side * (ext - ent)
    remaining_move = side * (ext - cand)

    valid = (
        np.isfinite(total_move)
        & np.isfinite(remaining_move)
        & (total_move > 0.0)
    )
    remaining_fraction = np.full(len(is_long), np.nan, dtype=np.float64)
    remaining_fraction[valid] = remaining_move[valid] / total_move[valid]
    missed_fraction = 1.0 - remaining_fraction

    return {
        "side": side,
        "total_move": total_move,
        "remaining_fraction": remaining_fraction,
        "missed_fraction": missed_fraction,
        "fraction_valid": valid,
    }


# --------------------------------------------------------------------------- #
# Regression (frozen BASE_PARAMS, objective=regression_l1)                    #
# --------------------------------------------------------------------------- #
def fit_eq_regressor(Xtr, ytr, wtr, Xv, yv, wv):
    """Robust L1 regression. Training weights normalized to TRAIN mean 1 (frozen
    methodology). No tuning, no target clipping."""
    params = dict(BASE_PARAMS)
    params["objective"] = "regression_l1"
    wtr = np.asarray(wtr, np.float64)
    wtr = wtr / wtr.mean() if wtr.mean() > 0 else wtr
    model = lgb.LGBMRegressor(**params)
    model.fit(
        Xtr, ytr,
        sample_weight=wtr,
        eval_set=[(Xv, yv)],
        eval_sample_weight=[wv],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


# --------------------------------------------------------------------------- #
# Per-trade ranking metric (allowed ~638-opportunity loop)                    #
# --------------------------------------------------------------------------- #
def per_trade_quality_metrics(gid, candidate_fill_index, y_true, y_pred):
    """Within-trade Kendall tau + selection regret. One row per Oracle opportunity.

    tau is defined only for trades with >=2 Candidates and non-constant true
    RemainingEdgeATR; otherwise tau=0 and qualified=False. Prediction ties are
    broken by expected random tie-break (mean true of tied rows).
    """
    gid = np.asarray(gid, dtype=object)
    order = np.argsort(gid, kind="stable")
    g = gid[order]
    yt = np.asarray(y_true, dtype=np.float64)[order]
    yp = np.asarray(y_pred, dtype=np.float64)[order]
    fi = np.asarray(candidate_fill_index)[order]

    starts = np.r_[0, 1 + np.flatnonzero(g[1:] != g[:-1])]
    ends = np.r_[starts[1:], len(g)]

    rows = []
    for a, b in zip(starts, ends):
        t = yt[a:b]
        p = yp[a:b]
        f = fi[a:b]
        n = b - a
        ptp_valid = (n >= 2) and (np.ptp(t) > 1e-12)
        if ptp_valid:
            stat = kendalltau(t, p).statistic
            tau = 0.0 if not np.isfinite(stat) else float(stat)
        else:
            tau = 0.0
        best_true = float(np.max(t))
        random_expected = float(np.mean(t))
        first_true = float(t[np.argmin(f)])
        pmax = np.max(p)
        chosen_true = float(np.mean(t[p == pmax]))
        rows.append((
            g[a], n, tau,
            best_true - chosen_true,   # selection regret (lower better)
            best_true - first_true,    # vs FIRST candidate baseline
            best_true - random_expected,  # vs RANDOM expected baseline
            bool(ptp_valid),
        ))
    return rows


def weighted_mae(y, pred, w):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    return float(np.average(np.abs(y - pred), weights=w))


# --------------------------------------------------------------------------- #
# Bootstrap helpers                                                           #
# --------------------------------------------------------------------------- #
def _bootstrap_mean(values, B=EQ_BOOTSTRAP_REPLICATES, seed=EQ_BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=float)
    k = values.size
    if k == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty(B, dtype=float)
    for b in range(B):
        s = rng.integers(0, k, size=k)
        means[b] = values[s].mean()
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _bootstrap_symbols(values, B=EQ_BOOTSTRAP_REPLICATES, seed=EQ_BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=float)
    k = values.size
    if k == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty(B, dtype=float)
    for b in range(B):
        s = rng.integers(0, k, size=k)
        means[b] = values[s].mean()
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _cluster(triple):
    return {"mean": triple[0], "ci_low": triple[1], "ci_high": triple[2]}


# --------------------------------------------------------------------------- #
# Data container (built ONCE; reused for pooled / per-symbol / LOSO)          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EQData:
    eq9: np.ndarray          # [N, 9]  float32
    eq33: np.ndarray         # [N, 33] float32
    y: np.ndarray            # [N] float64  (RemainingEdgeATR)
    w: np.ndarray            # [N] float64  (sample_weight_raw)
    long: np.ndarray         # [N] bool
    gid: np.ndarray          # [N] str
    symbol: np.ndarray       # [N] object
    cand_fill_index: np.ndarray  # [N] int64
    bars_entry: np.ndarray  # [N] float64
    bars_exit: np.ndarray   # [N] float64
    remaining_fraction: np.ndarray  # [N] float64
    elig: np.ndarray         # [N] bool


def build_eq_data(ds: pd.DataFrame) -> EQData:
    raw_struct33 = ds[RAW_STRUCT33_COLS].to_numpy(dtype=np.float32)
    oracle_direction = ds["oracle_direction"].to_numpy(object)
    teacher_is_long = (oracle_direction == "LONG")
    eq9, eq33 = orient_struct33(raw_struct33, teacher_is_long)

    y = ds["entry_quality_atr"].to_numpy(np.float64)        # RemainingEdgeATR
    w = ds["sample_weight_raw"].to_numpy(np.float64)
    symbol = ds["symbol"].to_numpy(object)
    gid = np.char.add(
        np.char.add(symbol.astype(str), "::"),
        ds["oracle_trade_id"].to_numpy(object).astype(str),
    )
    cand_fill_index = ds["candidate_fill_index"].to_numpy(np.int64)
    bars_entry = ds["bars_to_oracle_entry"].to_numpy(np.float64)
    bars_exit = ds["bars_to_oracle_exit"].to_numpy(np.float64)

    lab = build_eq_labels(
        oracle_direction,
        ds["candidate_fill_price"].to_numpy(np.float64),
        ds["oracle_entry_fill_price"].to_numpy(np.float64),
        ds["oracle_exit_fill_price"].to_numpy(np.float64),
    )
    remaining_fraction = lab["remaining_fraction"]

    elig = (
        np.isfinite(y)
        & np.isfinite(w)
        & (w > 0)
        & np.isfinite(bars_exit)
        & (bars_exit > 0)
    )
    return EQData(
        eq9, eq33, y, w, teacher_is_long, gid, symbol,
        cand_fill_index, bars_entry, bars_exit, remaining_fraction, elig,
    )


def _split_arrays(data: EQData, full_idx):
    """Select eligible rows within a full-index set (train/val/test)."""
    m = data.elig[full_idx]
    idx = full_idx[m]
    return {
        "X9": data.eq9[idx],
        "X33": data.eq33[idx],
        "y": data.y[idx],
        "w": data.w[idx],
        "long": data.long[idx],
        "gid": data.gid[idx],
        "sym": data.symbol[idx],
        "cand": data.cand_fill_index[idx],
        "bars_entry": data.bars_entry[idx],
        "rf": data.remaining_fraction[idx],
    }


def _fit_predict(tr, va, te):
    """Fit Q0 (EQ9), Q1 (EQ33), Q2 (EQ33 side-specific) and predict on te."""
    q0 = fit_eq_regressor(tr["X9"], tr["y"], tr["w"], va["X9"], va["y"], va["w"])
    q1 = fit_eq_regressor(tr["X33"], tr["y"], tr["w"], va["X33"], va["y"], va["w"])

    long_tr = tr["long"]
    long_va = va["long"]
    long_te = te["long"]
    qL = fit_eq_regressor(
        tr["X33"][long_tr], tr["y"][long_tr], tr["w"][long_tr],
        va["X33"][long_va], va["y"][long_va], va["w"][long_va],
    )
    qS = fit_eq_regressor(
        tr["X33"][~long_tr], tr["y"][~long_tr], tr["w"][~long_tr],
        va["X33"][~long_va], va["y"][~long_va], va["w"][~long_va],
    )
    p0 = q0.predict(te["X9"])
    p1 = q1.predict(te["X33"])
    p2 = np.empty(len(te["y"]), dtype=np.float64)
    p2[long_te] = qL.predict(te["X33"][long_te])
    p2[~long_te] = qS.predict(te["X33"][~long_te])
    return p0, p1, p2


# --------------------------------------------------------------------------- #
# Per-trade table + aggregation                                               #
# --------------------------------------------------------------------------- #
def build_per_trade_table(rows0, rows1, rows2, te):
    gids0 = [r[0] for r in rows0]
    df = pd.DataFrame({
        "gid": gids0,
        "n": [r[1] for r in rows0],
        "tau0": [r[2] for r in rows0], "sel0": [r[3] for r in rows0],
        "first0": [r[4] for r in rows0], "rand0": [r[5] for r in rows0],
        "qual": [r[6] for r in rows0],
        "tau1": [r[2] for r in rows1], "sel1": [r[3] for r in rows1],
        "tau2": [r[2] for r in rows2], "sel2": [r[3] for r in rows2],
    })
    gid2long = {}
    gid2sym = {}
    for g, l, s in zip(te["gid"], te["long"], te["sym"]):
        gid2long.setdefault(g, bool(l))
        gid2sym.setdefault(g, s)
    df["long"] = df["gid"].map(gid2long)
    df["symbol"] = df["gid"].map(gid2sym)
    return df


def _agg(df: pd.DataFrame) -> dict:
    q = df[df["qual"]]
    if len(q) == 0:
        return {
            "n_trades": int(len(df)), "n_qualified": 0,
            "tau0": None, "tau1": None, "tau2": None,
            "sel0": None, "sel1": None, "sel2": None,
            "rand": None, "first": None,
            "delta10": (0.0, 0.0, 0.0), "delta21": (0.0, 0.0, 0.0),
            "regret_improve10": (0.0, 0.0, 0.0),
            "regret_improve21": (0.0, 0.0, 0.0),
        }
    tau0 = float(q["tau0"].mean())
    tau1 = float(q["tau1"].mean())
    tau2 = float(q["tau2"].mean())
    sel0 = float(q["sel0"].mean())
    sel1 = float(q["sel1"].mean())
    sel2 = float(q["sel2"].mean())
    rand = float(q["rand0"].mean())
    first = float(q["first0"].mean())
    d10 = _bootstrap_mean((q["tau1"] - q["tau0"]).to_numpy())
    d21 = _bootstrap_mean((q["tau2"] - q["tau1"]).to_numpy())
    ri10 = _bootstrap_mean((q["sel0"] - q["sel1"]).to_numpy())
    ri21 = _bootstrap_mean((q["sel1"] - q["sel2"]).to_numpy())
    return {
        "n_trades": int(len(df)), "n_qualified": int(len(q)),
        "tau0": tau0, "tau1": tau1, "tau2": tau2,
        "sel0": sel0, "sel1": sel1, "sel2": sel2,
        "rand": rand, "first": first,
        "delta10": d10, "delta21": d21,
        "regret_improve10": ri10, "regret_improve21": ri21,
    }


# --------------------------------------------------------------------------- #
# Label audit                                                                #
# --------------------------------------------------------------------------- #
def _trade_group_sums(gid, values):
    """Return per-trade sum of values, one entry per unique trade (in first-seen order)."""
    gid = np.asarray(gid, dtype=object)
    order = np.argsort(gid, kind="stable")
    g = gid[order]
    v = np.asarray(values, dtype=np.float64)[order]
    starts = np.r_[0, 1 + np.flatnonzero(g[1:] != g[:-1])]
    ends = np.r_[starts[1:], len(g)]
    out = np.empty(len(starts), dtype=np.float64)
    for i, (a, b) in enumerate(zip(starts, ends)):
        out[i] = v[a:b].sum()
    return out


def run_label_audit(data: EQData, ds: pd.DataFrame, test_idx):
    e = data.elig
    # audit scope = TEST eligible rows
    m = e[test_idx]
    idx = test_idx[m]

    y = data.y[idx]
    w = data.w[idx]
    gid = data.gid[idx]
    sym = data.symbol[idx]
    long = data.long[idx]
    rf = data.remaining_fraction[idx]
    mf = 1.0 - rf
    be = data.bars_entry[idx]
    cfp = ds["candidate_fill_price"].to_numpy(np.float64)[idx]
    oefp = ds["oracle_entry_fill_price"].to_numpy(np.float64)[idx]
    oe = ds["oracle_exit_fill_price"].to_numpy(np.float64)[idx]
    cd = ds["oracle_direction"].to_numpy(object)[idx]
    side = np.where(long, 1.0, -1.0)
    total_move = side * (oe - oefp)

    rows = []

    def _quantiles(arr):
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {f"p{p}": None for p in (10, 25, 50, 75, 90)}
        return {f"p{p}": float(np.percentile(arr, p)) for p in (10, 25, 50, 75, 90)}

    def _record(scope, group, mask):
        ym = y[mask]
        rfm = rf[mask]
        mfm = mf[mask]
        qe = _quantiles(ym)
        qr = _quantiles(rfm)
        qm = _quantiles(mfm)
        rows.append({
            "scope": scope, "group": group,
            "n_rows": int(mask.sum()),
            "eq_p10": qe["p10"], "eq_p50": qe["p50"], "eq_p90": qe["p90"],
            "rf_p10": qr["p10"], "rf_p50": qr["p50"], "rf_p90": qr["p90"],
            "mf_p10": qm["p10"], "mf_p50": qm["p50"], "mf_p90": qm["p90"],
        })

    _record("OVERALL", "ALL", np.ones(len(y), dtype=bool))
    _record("TEACHER_SIDE", "LONG", long)
    _record("TEACHER_SIDE", "SHORT", ~long)
    for s in SYMBOLS:
        _record("SYMBOL", s, sym == s)
    phase = np.where(be > 0, "BEFORE_ENTRY", np.where(be == 0, "AT_ENTRY", "IN_POSITION"))
    for ph in ("BEFORE_ENTRY", "AT_ENTRY", "IN_POSITION"):
        _record("PHASE", ph, phase == ph)

    # ---- gates ----
    # gate 4: per global trade, sum sample_weight_raw == 1
    trade_w = _trade_group_sums(gid, w)
    gate4_max_dev = float(np.max(np.abs(trade_w - 1.0))) if trade_w.size else None
    gate4_ok = bool(np.allclose(trade_w, 1.0, atol=1e-9))

    # gate 5: bars_to_oracle_entry == 0  => candidate_fill_price == oracle_entry_fill_price
    at_entry = be == 0
    gate5_max_dev = float(np.max(np.abs(cfp[at_entry] - oefp[at_entry]))) if at_entry.any() else None
    gate5_ok = bool(np.allclose(cfp[at_entry], oefp[at_entry], atol=1e-6)) if at_entry.any() else True

    # gate 6: bars_to_oracle_entry == 0 & total_move > 0 => RemainingFraction == 1
    gate6_mask = at_entry & np.isfinite(total_move) & (total_move > 0)
    gate6_max_dev = float(np.max(np.abs(rf[gate6_mask] - 1.0))) if gate6_mask.any() else None
    gate6_ok = bool(np.allclose(rf[gate6_mask], 1.0, atol=1e-6)) if gate6_mask.any() else True

    summary = {
        "n_eligible_rows": int(y.size),
        "n_eligible_trades": int(len(np.unique(gid))),
        "gate_weight_sum_eq_1": {"ok": gate4_ok, "max_abs_dev": gate4_max_dev},
        "gate_at_entry_price_identity": {"ok": gate5_ok, "max_abs_dev": gate5_max_dev},
        "gate_at_entry_remaining_fraction_eq_1": {"ok": gate6_ok, "max_abs_dev": gate6_max_dev},
        "bars_to_oracle_exit_gt_0": bool(np.all(data.bars_exit[idx] > 0)) if idx.size else None,
        "remaining_edge_atr_finite": bool(np.all(np.isfinite(y))) if y.size else None,
        "sample_weight_finite_positive": bool(np.all(np.isfinite(w) & (w > 0))) if w.size else None,
    }
    audit_df = pd.DataFrame(rows)
    return summary, audit_df


# --------------------------------------------------------------------------- #
# Verdict (frozen before TEST)                                                #
# --------------------------------------------------------------------------- #
def _state(delta):
    lo, hi = delta[1], delta[2]
    if hi < 0:
        return "harmful"
    if lo > 0:
        return "supported"
    return "no_identifiable"


def _eq_verdict(pooled, long_m, short_m):
    return {
        "STRUCT33_increment": {
            "pooled": _state(pooled["delta10"]),
            "LONG": _state(long_m["delta10"]),
            "SHORT": _state(short_m["delta10"]),
            "pooled_ci": list(pooled["delta10"]),
        },
        "SIDE_SPECIALIST_increment": {
            "pooled": _state(pooled["delta21"]),
            "LONG": _state(long_m["delta21"]),
            "SHORT": _state(short_m["delta21"]),
            "pooled_ci": list(pooled["delta21"]),
        },
        "interpretation": (
            "Q1-Q0 answers whether SR/Liquidity adds Entry-Quality ranking information "
            "beyond DTP9. Q2-Q1 answers whether LONG and SHORT Entry-Quality mappings "
            "need side-specific functions. LONG/SHORT must be reported separately; a "
            "pooled null does not adjudicate a genuine side asymmetry."),
    }


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
# Main experiment                                                            #
# --------------------------------------------------------------------------- #
def run_entry_quality(save: bool = True, verbose: bool = True):
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

    data = build_eq_data(ds)
    n_test_rows = int(test_idx.size)
    n_test_trades = int(len(np.unique(data.gid[test_idx])))
    if n_test_rows != FROZEN_TEST["test_rows"] or n_test_trades != FROZEN_TEST["test_trades"]:
        raise RuntimeError(
            f"STOP_EQ_SPLIT_DRIFT: rows={n_test_rows} trades={n_test_trades}")

    tr = _split_arrays(data, train_idx)
    va = _split_arrays(data, val_idx)
    te = _split_arrays(data, test_idx)

    # ----- label audit -----
    log("label audit ...")
    audit_summary, audit_df = run_label_audit(data, ds, test_idx)

    # ----- fit pooled Q0/Q1/Q2 -----
    log("fit pooled Q0/Q1/Q2 ...")
    pred_q0, pred_q1, pred_q2 = _fit_predict(tr, va, te)

    rows0 = per_trade_quality_metrics(te["gid"], te["cand"], te["y"], pred_q0)
    rows1 = per_trade_quality_metrics(te["gid"], te["cand"], te["y"], pred_q1)
    rows2 = per_trade_quality_metrics(te["gid"], te["cand"], te["y"], pred_q2)
    df = build_per_trade_table(rows0, rows1, rows2, te)

    pooled = _agg(df)
    pooled_ex_ag = _agg(df[df["symbol"] != "AG"])
    long_m = _agg(df[df["long"]])
    short_m = _agg(df[~df["long"]])

    # weighted MAE (all eligible test rows) + constant baseline
    mae_q0 = weighted_mae(te["y"], pred_q0, te["w"])
    mae_q1 = weighted_mae(te["y"], pred_q1, te["w"])
    mae_q2 = weighted_mae(te["y"], pred_q2, te["w"])
    train_median = float(np.median(tr["y"]))
    mae_const = weighted_mae(te["y"], np.full(len(te["y"]), train_median), te["w"])

    # ----- per-symbol -----
    log("per-symbol ...")
    per_symbol = []
    sym_d10 = []
    sym_d21 = []
    sym_ri10 = []
    sym_ri21 = []
    for s in SYMBOLS:
        sub = df[df["symbol"] == s]
        if len(sub) == 0:
            continue
        a = _agg(sub)
        per_symbol.append({
            "symbol": s,
            "n_trades": a["n_trades"], "n_qualified": a["n_qualified"],
            "tau0": a["tau0"], "tau1": a["tau1"], "tau2": a["tau2"],
            "sel0": a["sel0"], "sel1": a["sel1"], "sel2": a["sel2"],
            "rand": a["rand"], "first": a["first"],
            "delta10_mean": a["delta10"][0], "delta10_ci_low": a["delta10"][1],
            "delta10_ci_high": a["delta10"][2],
            "delta21_mean": a["delta21"][0], "delta21_ci_low": a["delta21"][1],
            "delta21_ci_high": a["delta21"][2],
            "regret_improve10": a["regret_improve10"][0],
            "regret_improve21": a["regret_improve21"][0],
        })
        sym_d10.append(a["delta10"][0])
        sym_d21.append(a["delta21"][0])
        sym_ri10.append(a["regret_improve10"][0])
        sym_ri21.append(a["regret_improve21"][0])

    per_symbol_cluster = {
        "Q1_minus_Q0": _cluster(_bootstrap_symbols(sym_d10)),
        "Q2_minus_Q1": _cluster(_bootstrap_symbols(sym_d21)),
        "regret_improve10": _cluster(_bootstrap_symbols(sym_ri10)),
        "regret_improve21": _cluster(_bootstrap_symbols(sym_ri21)),
        "n_symbols": len(sym_d10),
        "Q1_minus_Q0_positive_symbols": int(sum(1 for v in sym_d10 if v > 0)),
        "Q1_minus_Q0_negative_symbols": int(sum(1 for v in sym_d10 if v < 0)),
    }

    # ----- LOSO (15 folds, sequential) -----
    log("LOSO ...")
    loso = []
    loso_d10 = []
    loso_d21 = []
    loso_ri10 = []
    loso_ri21 = []
    for s in SYMBOLS:
        tr_s = _split_arrays(data, train_idx[data.symbol[train_idx] != s])
        va_s = _split_arrays(data, val_idx[data.symbol[val_idx] != s])
        te_s = _split_arrays(data, test_idx[data.symbol[test_idx] == s])
        if te_s["y"].size == 0:
            continue
        p0, p1, p2 = _fit_predict(tr_s, va_s, te_s)
        r0 = per_trade_quality_metrics(te_s["gid"], te_s["cand"], te_s["y"], p0)
        r1 = per_trade_quality_metrics(te_s["gid"], te_s["cand"], te_s["y"], p1)
        r2 = per_trade_quality_metrics(te_s["gid"], te_s["cand"], te_s["y"], p2)
        df_s = build_per_trade_table(r0, r1, r2, te_s)
        a = _agg(df_s)
        loso.append({
            "held_out_symbol": s,
            "n_trades": a["n_trades"], "n_qualified": a["n_qualified"],
            "tau0": a["tau0"], "tau1": a["tau1"], "tau2": a["tau2"],
            "sel0": a["sel0"], "sel1": a["sel1"], "sel2": a["sel2"],
            "delta10_mean": a["delta10"][0], "delta10_ci_low": a["delta10"][1],
            "delta10_ci_high": a["delta10"][2],
            "delta21_mean": a["delta21"][0], "delta21_ci_low": a["delta21"][1],
            "delta21_ci_high": a["delta21"][2],
            "regret_improve10": a["regret_improve10"][0],
            "regret_improve21": a["regret_improve21"][0],
        })
        loso_d10.append(a["delta10"][0])
        loso_d21.append(a["delta21"][0])
        loso_ri10.append(a["regret_improve10"][0])
        loso_ri21.append(a["regret_improve21"][0])
        log(f"  LOSO held-out {s}: tau Q0={a['tau0']:.3f} Q1={a['tau1']:.3f} "
             f"Q2={a['tau2']:.3f} Q1-Q0={a['delta10'][0]:+.3f}")

    loso_agg = {
        "Q1_minus_Q0_mean": float(np.mean(loso_d10)) if loso_d10 else None,
        "Q1_minus_Q0_median": float(np.median(loso_d10)) if loso_d10 else None,
        "Q1_minus_Q0_positive_symbols": int(sum(1 for v in loso_d10 if v > 0)),
        "Q1_minus_Q0_negative_symbols": int(sum(1 for v in loso_d10 if v < 0)),
        "Q1_minus_Q0_cluster": _cluster(_bootstrap_symbols(loso_d10)),
        "Q2_minus_Q1_cluster": _cluster(_bootstrap_symbols(loso_d21)),
        "regret_improve10_cluster": _cluster(_bootstrap_symbols(loso_ri10)),
        "regret_improve21_cluster": _cluster(_bootstrap_symbols(loso_ri21)),
    }

    # ----- verdict -----
    verdict = _eq_verdict(pooled, long_m, short_m)

    summary = _clean({
        "task_id": TASK_ID,
        "base_sha": BASE_SHA,
        "test_status": TEST_STATUS,
        "frozen_split": {
            "test_rows": n_test_rows,
            "test_trades": n_test_trades,
            "train_rows": int(train_idx.size),
            "val_rows": int(val_idx.size),
            "frozen_test": FROZEN_TEST,
        },
        "label_audit": audit_summary,
        "features": {
            "Q0": "EQ9 (direction-normalized DTP9, 9 features)",
            "Q1": "EQ33 (direction-normalized STRUCT33, 33 features)",
            "Q2": "EQ33 side-specific (LONG expert + SHORT expert)",
            "eq9_cols": list(EQ9_COLS),
            "eq33_cols": list(EQ33_COLS),
            "orientation": "Teacher direction (diagnostic only)",
        },
        "contract": {
            "no_upstream_rerun": True,
            "no_struct33_rebuild": True,
            "objective": "regression_l1",
            "base_params_unchanged": True,
            "no_threshold_tuning": True,
            "no_hyperparameter_tuning": True,
            "no_target_clipping": True,
            "no_entry_skip_threshold": True,
            "teacher_direction_only_for_orientation": True,
        },
        "pooled": pooled,
        "pooled_ex_ag": pooled_ex_ag,
        "teacher_LONG": long_m,
        "teacher_SHORT": short_m,
        "weighted_mae": {"Q0": mae_q0, "Q1": mae_q1, "Q2": mae_q2,
                         "constant_baseline": mae_const},
        "constant_baseline": {
            "train_median_remaining_edge_atr": train_median,
            "weighted_mae": mae_const,
        },
        "per_symbol": per_symbol,
        "per_symbol_cluster_bootstrap": per_symbol_cluster,
        "loso": {"folds": loso, "aggregate": loso_agg},
        "verdict": verdict,
        "provenance": {
            "base_sha": BASE_SHA,
            "reused_split_module": "direction_null_baseline_v1.build_frozen_split",
            "base_params_source": "train_direction_model_15sym_v1.BASE_PARAMS",
            "data_source": "frozen 15-symbol Phase1 parquet (STRUCT33 columns present)",
            "bootstraps_replicates": EQ_BOOTSTRAP_REPLICATES,
            "bootstrap_seed": EQ_BOOTSTRAP_SEED,
            "note": ("Conditional Entry-Quality diagnostic. Teacher direction orients "
                     "features and routes Q2; not a deployable entry strategy."),
        },
    })

    if save:
        os.makedirs(EVIDENCE_DIR, exist_ok=True)
        with open(SUMMARY_JSON, "w") as f:
            json.dump(summary, f, indent=2)
        _write_per_symbol_csv(per_symbol, PER_SYMBOL_CSV)
        _write_trade_metrics_csv(df, TRADE_METRICS_CSV)
        _write_loso_csv(loso, LOSO_CSV)
        audit_df.to_csv(LABEL_AUDIT_CSV, index=False)
        log(f"evidence written -> {SUMMARY_JSON}")

    return {
        "summary": summary,
        "per_trade_df": df,
        "audit_df": audit_df,
        "paths": {
            "summary": SUMMARY_JSON,
            "per_symbol_csv": PER_SYMBOL_CSV,
            "trade_metrics_csv": TRADE_METRICS_CSV,
            "loso_csv": LOSO_CSV,
            "label_audit_csv": LABEL_AUDIT_CSV,
        },
    }


def _write_per_symbol_csv(per_symbol, path):
    cols = ["symbol", "n_trades", "n_qualified",
            "tau0", "tau1", "tau2",
            "sel0", "sel1", "sel2", "rand", "first",
            "delta10_mean", "delta10_ci_low", "delta10_ci_high",
            "delta21_mean", "delta21_ci_low", "delta21_ci_high",
            "regret_improve10", "regret_improve21"]
    df = pd.DataFrame(per_symbol)[cols]
    df.to_csv(path, index=False)


def _write_trade_metrics_csv(df, path):
    cols = ["gid", "symbol", "long", "n", "qual",
            "tau0", "tau1", "tau2",
            "sel0", "sel1", "sel2", "first0", "rand0"]
    out = df[cols].copy()
    out["long"] = out["long"].map({True: "LONG", False: "SHORT"})
    out.to_csv(path, index=False)


def _write_loso_csv(loso, path):
    cols = ["held_out_symbol", "n_trades", "n_qualified",
            "tau0", "tau1", "tau2",
            "sel0", "sel1", "sel2",
            "delta10_mean", "delta10_ci_low", "delta10_ci_high",
            "delta21_mean", "delta21_ci_low", "delta21_ci_high",
            "regret_improve10", "regret_improve21"]
    df = pd.DataFrame(loso)[cols]
    df.to_csv(path, index=False)


if __name__ == "__main__":
    run_entry_quality(save=True, verbose=True)
