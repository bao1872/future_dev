"""PAYOFF-CANDIDATE-SEQUENCE-01A -- Candidate-Event Memory Incremental Information Audit.

This supersedes PAYOFF-TEMPORAL-01A as the CURRENT PRIMARY temporal research
direction. PAYOFF-TEMPORAL-01A is left archived (bar-level); this module studies
CANDIDATE EVENTS, not market bars.

Core question
=============
Does the causal history of the previous 3-5 candidate EVENTS add incremental
information about CURRENT TD5 payoff magnitude, beyond the current static state
and the frozen current model predictions?

The event unit is a candidate opportunity (symbol, side, decision_bar). History
means the kth PREVIOUS candidate event for the SAME (symbol, side), ordered by
decision_time. No fixed-bar lag is used. gap_bars (current bar - previous bar)
is itself a feature.

Frozen model state
==================
The base p_win / mu_win / mu_loss predictions are LAYER-0 OOF predictions from
the existing A0 OOF shards. They are NEVER refit. base_model_fit_count must be 0.

New model
=========
A second-level (meta) correction model predicts the RESIDUAL of the frozen base
model:
    r_win_i = Y_i - frozen_mu_win_i        (winners only)
    r_loss_i = (-Y_i) - frozen_mu_loss_i   (losers only)
    mu_win_seq  = max(0, frozen_mu_win  + predicted_r_win)
    mu_loss_seq = max(0, frozen_mu_loss + predicted_r_loss)
The same frozen LightGBM family / params are used for EVERY arm and head.

Four frozen arms
=================
    S0  : STATIC (PAY8 x8 + p_win + mu_win + mu_loss) -- static recalibration
    S3P : S0 + prediction-state history for lag 1..3 (no realized outcomes)
    S3F : S3P + outcome-feedback history for lag 1..3  (MAIN arm)
    S5F : S0 + prediction-state + outcome-feedback history for lag 1..5

Second-level causal OOF
=======================
The frozen A0 predictions are layer-0 OOF. This experiment is a stacked/meta
experiment, so it must NOT randomly cross-validate layer 1. Base fold 0 is META
WARMUP (no formal meta evaluation). Formal meta evaluation folds:
    meta1 : train folds 0      ; evaluate fold 1
    meta2 : train folds 0..1   ; evaluate fold 2
    meta3 : train folds 0..2   ; evaluate fold 3
    meta4 : train folds 0..3   ; evaluate fold 4
For every meta fold with cutoff T (start of evaluation fold): a training row
requires decision_time < T AND label_available_time < T.

Expected fit count: 4 meta folds x 4 arms x 2 heads = 32 correction fits.
No hyperparameter search. No TEST read.

Performance contract
====================
Build the candidate-event ledger ONCE; build lag1..5 history ONCE; reuse across
all meta folds / heads / arms. Track counters; the expected invariants are
enforced in tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb

from research.liquidity_oracle_atlas.build_decomposed_value_dataset_v1 import (
    build_pay8_from_state,
    PAY8_COLS,
)
from research.liquidity_oracle_atlas.walkforward_development_v1 import build_fold_plan

# --------------------------------------------------------------------------- #
# Paths / constants                                                            #
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
STATE_PARQUET = ROOT / "artifacts/decomposed_value_v1/state_v1.parquet"
LABEL_PATH = ROOT / "artifacts/opportunity_value_v1/labels_train_v1.parquet"
OOF_DIR = ROOT / "artifacts/decomposed_value_v2/oof"

PRIMARY_HORIZON = "td5"
MAX_LAG = 5
LOW_P_THRESHOLD = 0.40
BOOTSTRAP_BLOCK_DAYS = 5
BOOTSTRAP_SEED = 20260929
N_JOBS = 1

GROUP = ["symbol", "side"]
KEY = ["symbol", "decision_bar", "side"]
PRED_COLS = ["p_win", "mu_win", "mu_loss"]
STATIC = [*PAY8_COLS, "p_win", "mu_win", "mu_loss"]

# Frozen LightGBM regression family / params (identical for EVERY arm & head).
LGB_PARAMS = dict(
    objective="regression_l1",
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    n_jobs=N_JOBS,
    random_state=BOOTSTRAP_SEED,
    verbose=-1,
)
ES_ROUNDS = 50
ES_FRAC = 0.85  # keep earliest 85% of available training days for fit, last 15% ES


# --------------------------------------------------------------------------- #
# Counters                                                                     #
# --------------------------------------------------------------------------- #
_COUNTERS = {
    "label_load_count": 0,
    "oof_load_count": 0,
    "pay8_load_count": 0,
    "candidate_history_build_count": 0,
    "base_model_fit_count": 0,
    "correction_model_fit_count": 0,
    "hyperparameter_search_count": 0,
    "test_label_read_count": 0,
    "raw_market_rescan_count": 0,
    "environment_recompute_count": 0,
    "geometry_recompute_count": 0,
    "concat_hotloop_count": 0,
}


def _bump(name: str, n: int = 1) -> None:
    if name not in _COUNTERS:
        raise KeyError(f"unknown counter {name!r}")
    _COUNTERS[name] += n


def reset_counters() -> None:
    for k in _COUNTERS:
        _COUNTERS[k] = 0


def counters() -> dict:
    return dict(_COUNTERS)


# --------------------------------------------------------------------------- #
# Feature contract                                                             #
# --------------------------------------------------------------------------- #
def pred_history_cols(n: int) -> list[str]:
    cols: list[str] = []
    for k in range(1, n + 1):
        cols += [
            f"h{k}__exists",
            f"h{k}__gap_bars",
            f"h{k}__pred_available",
            f"h{k}__p_win",
            f"h{k}__mu_win",
            f"h{k}__mu_loss",
            f"h{k}__delta_p",
            f"h{k}__delta_mu_win",
            f"h{k}__delta_mu_loss",
        ]
    return cols


def outcome_history_cols(n: int) -> list[str]:
    cols: list[str] = []
    for k in range(1, n + 1):
        cols += [
            f"h{k}__resolved",
            f"h{k}__return",
            f"h{k}__win",
            f"h{k}__prob_surprise",
            f"h{k}__econ_surprise",
        ]
    return cols


# Frozen arm feature contracts.
ARMS = {
    "S0": STATIC,
    "S3P": STATIC + pred_history_cols(3),
    "S3F": STATIC + pred_history_cols(3) + outcome_history_cols(3),
    "S5F": STATIC + pred_history_cols(5) + outcome_history_cols(5),
}


# --------------------------------------------------------------------------- #
# Loaders                                                                      #
# --------------------------------------------------------------------------- #
def _load_oof(oof_dir: Path | str = OOF_DIR) -> pd.DataFrame:
    """Load the 5 canonical A0 OOF shards once (frozen layer-0 predictions)."""
    import glob

    shards = sorted(Path(oof_dir).glob("A0_V1_DISJOINT_f*_td5.parquet"))
    if not shards:
        raise AssertionError(f"STOP_OOF_SHARDS_MISSING {oof_dir}")
    frames = [pd.read_parquet(s) for s in shards]
    _bump("oof_load_count")
    return pd.concat(frames, ignore_index=True)


def _load_pay8(state_path: Path | str = STATE_PARQUET,
               symbols: list | None = None) -> pd.DataFrame:
    """Frozen PAY8 owner: derive PAY8 once per symbol from state. No refit."""
    sdf = pd.read_parquet(state_path)
    _bump("pay8_load_count")
    if symbols is not None:
        sdf = sdf[sdf["symbol"].isin(set(symbols))]
    frames = []
    for sym, g in sdf.groupby("symbol", sort=True):
        g = g.sort_values("bar_index", kind="stable")
        if (g["bar_index"].to_numpy() != np.arange(len(g))).any():
            raise AssertionError(f"STOP_STATE_BAR_INDEX_NOT_DENSE {sym}")
        frames.append(build_pay8_from_state(g, sym))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Candidate-event ledger                                                       #
# --------------------------------------------------------------------------- #
def build_candidate_ledger(state_path: Path | str = STATE_PARQUET,
                           label_path: Path | str = LABEL_PATH,
                           oof_dir: Path | str = OOF_DIR,
                           *,
                           symbols: list | None = None) -> pd.DataFrame:
    """One row per canonical TD5 candidate-side event, with frozen base state.

    Built FROM ALL canonical TD5 TRAIN labels, then left-joined with frozen PAY8
    and frozen A0 OOF predictions. Rows without OOF predictions (e.g. warmup or
    non-selected TD5) are KEPT -- they remain as history candidates. The
    candidate-event history is built on this full ledger, BEFORE restricting the
    scientific rows to OOF rows.
    """
    lab = pd.read_parquet(label_path)
    _bump("label_load_count")
    td5 = lab[lab["horizon"] == PRIMARY_HORIZON].reset_index(drop=True).copy()
    if symbols is not None:
        td5 = td5[td5["symbol"].isin(set(symbols))].reset_index(drop=True)

    # PAY8 (frozen owner)
    pay = _load_pay8(state_path, symbols=symbols)
    # OOF (frozen layer-0 predictions)
    oof = _load_oof(oof_dir)
    oof_cols = ["symbol", "decision_bar", "side", "fold",
                "p_win", "mu_win", "mu_loss", "predicted_rr"]

    led = td5.copy()
    # PAY8 may collide with label geometry (log_structural_rr); drop label side.
    led = led.drop(columns=[c for c in PAY8_COLS if c in led.columns])
    led = led.merge(pay, on=KEY, how="left", validate="m:1", indicator="_mp")
    unmatched_pay = int((led["_mp"] != "both").sum())
    led = led.drop(columns=["_mp"])
    if unmatched_pay:
        raise AssertionError(f"STOP_PAY8_JOIN_UNMATCHED n={unmatched_pay}")

    # OOF join: unmatched is EXPECTED (many TD5 labels have no OOF prediction).
    led = led.merge(oof[oof_cols], on=KEY, how="left", validate="m:1")

    # Hard gate: no duplicate candidate key in the TD5 label set.
    dup = int(led.duplicated(subset=KEY).sum())
    if dup:
        raise AssertionError(f"STOP_LEDGER_DUPLICATE_KEY n={dup}")
    return led


# --------------------------------------------------------------------------- #
# Candidate-event history                                                      #
# --------------------------------------------------------------------------- #
def build_candidate_event_history(ledger: pd.DataFrame,
                                 max_lag: int = MAX_LAG) -> pd.DataFrame:
    """Attach lag1..max_lag candidate-EVENT history features.

    History means previous candidate EVENTS for the same (symbol, side), ordered
    by (decision_time, decision_bar). Actual historical outcomes are exposed
    ONLY when:
        previous.label_available_time < current.decision_time
    Unresolved candidates still occupy their own lag slot (resolved=0, outcome
    fields NaN); we do NOT skip them to substitute an older resolved event.
    """
    x = ledger.sort_values(
        ["symbol", "side", "decision_time", "decision_bar"],
        kind="stable",
    ).reset_index(drop=True).copy()

    grp = x.groupby(GROUP, sort=False)
    cur_time = pd.to_datetime(x["decision_time"])
    cur_bar = x["decision_bar"].to_numpy(np.int64)

    for k in range(1, max_lag + 1):
        prev_dt = grp["decision_time"].shift(k)
        prev_av = grp["label_available_time"].shift(k)
        prev_bar = grp["decision_bar"].shift(k)

        exists = prev_dt.notna()
        x[f"h{k}__exists"] = exists.astype(np.int8)
        x[f"h{k}__gap_bars"] = np.where(
            exists, cur_bar - prev_bar.to_numpy(np.float64), np.nan)

        prev_p = grp["p_win"].shift(k)
        prev_mw = grp["mu_win"].shift(k)
        prev_ml = grp["mu_loss"].shift(k)
        pred_avail = prev_p.notna() & prev_mw.notna() & prev_ml.notna()
        x[f"h{k}__pred_available"] = pred_avail.astype(np.int8)
        x[f"h{k}__p_win"] = prev_p
        x[f"h{k}__mu_win"] = prev_mw
        x[f"h{k}__mu_loss"] = prev_ml

        # change from historical frozen prediction to current frozen prediction
        x[f"h{k}__delta_p"] = x["p_win"] - prev_p
        x[f"h{k}__delta_mu_win"] = x["mu_win"] - prev_mw
        x[f"h{k}__delta_mu_loss"] = x["mu_loss"] - prev_ml

        resolved = (
            exists
            & prev_av.notna()
            & (pd.to_datetime(prev_av) < cur_time)
        )
        x[f"h{k}__resolved"] = resolved.astype(np.int8)

        prev_y = grp["episode_return_atr"].shift(k)
        actual_y = prev_y.where(resolved)
        actual_win = (prev_y > 0).astype(np.float64).where(resolved)
        x[f"h{k}__return"] = actual_y
        x[f"h{k}__win"] = actual_win

        # Calibration surprise: actual binary result minus historical p_win.
        prob_surprise = (actual_win - prev_p).where(resolved & pred_avail)
        x[f"h{k}__prob_surprise"] = prob_surprise

        # Economic surprise (history only): realized - model-expected (signed).
        prev_y_a = prev_y.to_numpy()
        pred_signed = pd.Series(
            np.where(prev_y_a > 0, prev_mw.to_numpy(np.float64),
                     -prev_ml.to_numpy(np.float64)),
            index=x.index,
        )
        econ_surprise = (prev_y - pred_signed).where(resolved & pred_avail)
        x[f"h{k}__econ_surprise"] = econ_surprise

    _bump("candidate_history_build_count")
    return x


# --------------------------------------------------------------------------- #
# Residual-correction targets                                                  #
# --------------------------------------------------------------------------- #
def residual_targets(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-row residual of the frozen base model (NaN where undefined).

    Winners:  r_win = Y - frozen_mu_win
    Losers:   r_loss = (-Y) - frozen_mu_loss
    """
    y = df["episode_return_atr"].to_numpy(np.float64)
    mw = df["mu_win"].to_numpy(np.float64)
    ml = df["mu_loss"].to_numpy(np.float64)
    win = y > 0
    r_win = np.full(len(df), np.nan)
    r_loss = np.full(len(df), np.nan)
    r_win[win] = y[win] - mw[win]
    r_loss[~win] = (-y[~win]) - ml[~win]
    return r_win, r_loss


# --------------------------------------------------------------------------- #
# Second-level causal OOF splits                                               #
# --------------------------------------------------------------------------- #
def meta_fold_splits(ledger: pd.DataFrame, plan=None):
    """Return list of (eval_fold_k, train_mask, eval_mask) for k in 1..4.

    fold0 is META WARMUP (excluded). Training rows for eval fold k are OOF rows
    with fold < k and decision_time < T and label_available_time < T, where T is
    the start of eval fold k (from the canonical plan if available, else from the
    data).
    """
    folds = sorted(int(f) for f in ledger["fold"].dropna().unique())
    if not folds:
        raise AssertionError("STOP_OOF_FOLD_MISSING")

    if plan is None:
        bounds = {}
        for f in folds:
            sub = ledger[ledger["fold"] == f]
            d = pd.to_datetime(sub["decision_time"]).dt.normalize()
            bounds[f] = (d.min(), d.max())
    else:
        bounds = {k: (pd.Timestamp(s), pd.Timestamp(e))
                  for k, (s, e) in enumerate(plan.outer)}

    dt = pd.to_datetime(ledger["decision_time"])
    av = pd.to_datetime(ledger["label_available_time"])
    has_fold = ledger["fold"].notna().to_numpy()

    splits = []
    for k in folds:
        if k == min(folds):  # meta warmup
            continue
        T = bounds[k][0]
        train = (ledger["fold"].to_numpy() < k) & has_fold \
            & (dt < T).to_numpy() & (av < T).to_numpy()
        evalm = ledger["fold"].to_numpy() == k
        splits.append((k, train, evalm))
    return splits


def _es_split(X: pd.DataFrame, y: pd.Series, mask: np.ndarray,
             times: pd.Series):
    """Select rows where mask is True, then split by time (earliest 85% fit,
    last 15% early-stopping). Returns (X_fit, y_fit, X_es, y_es). If there are
    too few valid rows, fall back to using all of them for both fit and ES so a
    model is still fit (deterministic, no early stop).
    """
    idx = np.where(mask)[0]
    if len(idx) < 2:
        return X.iloc[0:0], y.iloc[0:0], X.iloc[0:0], y.iloc[0:0]
    Xs = X.iloc[idx]
    ys = y.iloc[idx]
    if len(idx) < 50:
        return Xs, ys, Xs, ys
    ts = times.iloc[idx].to_numpy()
    order = np.argsort(ts, kind="stable")
    cut = max(1, int(len(order) * ES_FRAC))
    tr_o = order[:cut]
    es_o = order[cut:]
    return Xs.iloc[tr_o], ys.iloc[tr_o], Xs.iloc[es_o], ys.iloc[es_o]


def _fit_correction(X_fit, y_fit, X_es, y_es):
    _bump("correction_model_fit_count")
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        X_fit, y_fit,
        eval_X=X_es, eval_y=y_es,
        callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False),
                   lgb.log_evaluation(0)],
    )
    return model


# --------------------------------------------------------------------------- #
# Pipeline                                                                     #
# --------------------------------------------------------------------------- #
def run_sequence_oof(ledger: pd.DataFrame, plan=None,
                     arms: dict | None = None):
    """Run the 4x4x2 correction-model OOF. Returns (pred_store, meta_rows).

    pred_store[(arm, head)] is an array aligned to ledger index, with the
    corrected mu for the evaluation rows of each meta fold (NaN elsewhere).
    """
    if arms is None:
        arms = ARMS
    splits = meta_fold_splits(ledger, plan)
    if not splits:
        raise AssertionError("STOP_NO_META_FOLDS")

    r_win, r_loss = residual_targets(ledger)
    n = len(ledger)
    pred_store = {(arm, head): np.full(n, np.nan)
                  for arm in arms for head in ("win", "loss")}
    meta_rows = {}

    for (k, tr_mask, ev_mask) in splits:
        tr_idx = np.where(tr_mask)[0]
        ev_idx = np.where(ev_mask)[0]
        tr_times = pd.to_datetime(ledger.iloc[tr_idx]["decision_time"])

        for arm, cols in arms.items():
            Xtr = ledger.iloc[tr_idx][cols]
            Xev = ledger.iloc[ev_idx][cols]

            # ---- WIN residual head (fit on winners only) ----
            y_win = pd.Series(r_win[tr_idx], index=Xtr.index)
            m_win = ~np.isnan(r_win[tr_idx])
            Xf, yf, Xes, yes = _es_split(Xtr, y_win, m_win, tr_times)
            if len(Xf):
                mw = _fit_correction(Xf, yf, Xes, yes)
                r_pred = mw.predict(Xev)
                base = ledger.iloc[ev_idx]["mu_win"].to_numpy(np.float64)
                pred_store[(arm, "win")][ev_idx] = np.maximum(0.0, base + r_pred)

            # ---- LOSS residual head (fit on losers only) ----
            y_los = pd.Series(r_loss[tr_idx], index=Xtr.index)
            m_los = ~np.isnan(r_loss[tr_idx])
            Xfl, yfl, Xesl, yesl = _es_split(Xtr, y_los, m_los, tr_times)
            if len(Xfl):
                ml = _fit_correction(Xfl, yfl, Xesl, yesl)
                r_pred_l = ml.predict(Xev)
                base_l = ledger.iloc[ev_idx]["mu_loss"].to_numpy(np.float64)
                pred_store[(arm, "loss")][ev_idx] = np.maximum(
                    0.0, base_l + r_pred_l)

        meta_rows[k] = (int(len(tr_idx)), int(len(ev_idx)))
    return pred_store, meta_rows


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    fin = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not fin.any():
        return float("nan")
    return float(np.sum(w[fin] * x[fin]) / np.sum(w[fin]))


def paired_mae_diff(ledger: pd.DataFrame, mu_a: np.ndarray, mu_b: np.ndarray,
                    mask: np.ndarray) -> float:
    """Weighted mean of (|Y-mu_a| - |Y-mu_b|) over mask. >0 => mu_b better."""
    y = ledger["episode_return_atr"].to_numpy(dtype=float)
    w = ledger["sample_weight"].to_numpy(dtype=float)
    sel = (mask & np.isfinite(mu_a) & np.isfinite(mu_b)
           & np.isfinite(y) & np.isfinite(w) & (w > 0))
    if not sel.any():
        return float("nan")
    d = np.abs(y[sel] - mu_a[sel]) - np.abs(y[sel] - mu_b[sel])
    return _weighted_mean(d, w[sel])


def _mask_win_low(ledger: pd.DataFrame) -> np.ndarray:
    y = ledger["episode_return_atr"].to_numpy(dtype=float)
    p = ledger["p_win"].to_numpy(dtype=float)
    return (y > 0) & np.isfinite(p) & (p <= LOW_P_THRESHOLD)


def _mask_loss_low(ledger: pd.DataFrame) -> np.ndarray:
    y = ledger["episode_return_atr"].to_numpy(dtype=float)
    p = ledger["p_win"].to_numpy(dtype=float)
    return (y <= 0) & np.isfinite(p) & (p <= LOW_P_THRESHOLD)


def analyze(ledger: pd.DataFrame, pred_store: dict) -> dict:
    """Frozen metric set. Raw weighted means only (no bootstrap in this task).

    Primary: S3F vs S0 on current winners within LOW_P (p_win <= 0.40).
    Mechanism contrasts: S3P-S0, S3F-S3P, S5F-S3F (paired on identical rows).
    Also loss-head safety and ALL-population, per-symbol/LONG/SHORT within LOW_P.
    """
    res: dict = {}

    m_win_low = _mask_win_low(ledger)
    m_los_low = _mask_loss_low(ledger)
    all_win = (ledger["episode_return_atr"].to_numpy(float) > 0)
    all_los = ~all_win

    def d(arm_a, arm_b, head, mask):
        return paired_mae_diff(ledger, pred_store[(arm_a, head)],
                               pred_store[(arm_b, head)], mask)

    res["primary_D_win_low_S3F_vs_S0"] = d("S0", "S3F", "win", m_win_low)
    res["D_loss_low_S3F_vs_S0"] = d("S0", "S3F", "loss", m_los_low)
    # mechanism contrasts
    res["contrast_S3P_minus_S0_win_low"] = d("S0", "S3P", "win", m_win_low)
    res["contrast_S3F_minus_S3P_win_low"] = d("S3P", "S3F", "win", m_win_low)
    res["contrast_S5F_minus_S3F_win_low"] = d("S3F", "S5F", "win", m_win_low)
    # all-population
    res["D_win_all_S3F_vs_S0"] = d("S0", "S3F", "win", all_win)
    res["D_loss_all_S3F_vs_S0"] = d("S0", "S3F", "loss", all_los)

    # per-symbol / side within LOW_P (winners)
    res["per_symbol_win_low"] = {}
    for sym in sorted(ledger["symbol"].unique()):
        sub = ledger["symbol"].to_numpy() == sym
        m = m_win_low & sub
        if m.any():
            res["per_symbol_win_low"][sym] = d("S0", "S3F", "win", m)
    long_mask = ledger["side"].to_numpy() == "LONG"
    short_mask = ledger["side"].to_numpy() == "SHORT"
    res["D_win_low_LONG"] = d("S0", "S3F", "win", m_win_low & long_mask)
    res["D_win_low_SHORT"] = d("S0", "S3F", "win", m_win_low & short_mask)
    res["D_loss_low_LONG"] = d("S0", "S3F", "loss", m_los_low & long_mask)
    res["D_loss_low_SHORT"] = d("S0", "S3F", "loss", m_los_low & short_mask)
    return res


# --------------------------------------------------------------------------- #
# Small T1 smoke (AG/AU only) -- verification helper, NOT an artifact producer  #
# --------------------------------------------------------------------------- #
def run_t1(*, symbols=("AG", "AU"), plan=None) -> dict:
    """Run the full sequence OOF pipeline on a small symbol subset for
    verification. Produces NO committed artifact."""
    reset_counters()
    ledger = build_candidate_ledger(symbols=list(symbols))
    ledger = build_candidate_event_history(ledger)
    preds, meta_rows = run_sequence_oof(ledger, plan)
    metrics = analyze(ledger, preds)
    return {
        "n_rows_ledger": int(len(ledger)),
        "n_rows_with_oof": int(ledger["fold"].notna().sum()),
        "meta_rows": meta_rows,
        "counters": counters(),
        "metrics": metrics,
        "pred_store_eval_nonnull_win_S3F": int(
            np.count_nonzero(np.isfinite(preds[("S3F", "win")]))),
    }
