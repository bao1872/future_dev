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

The EXACT FROZEN LightGBM family / params from `payoff_ratio_model_v1` are used
for EVERY arm and head (objective=regression / metric=l2 / frozen REG_PARAMS /
EARLY_STOPPING_ROUNDS=100). Only `n_jobs` is overridden to 1 at call time for the
already-audited macOS/libomp stability reason. No tuning. No new parameters.

Canonical weighting
====================
Correction fitting uses the canonical `sample_weight` (both fit and early-
stopping rows), exactly matching the frozen payoff economics contract.

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

For every meta fold with cutoff T (start of evaluation fold from the CANONICAL
frozen plan):
  * training rows require: layer-0 OOF row exists AND fold < k AND
    decision_time < T AND label_available_time < T;
  * then the canonical two-side EPOCH PAIR PURITY (from
    walkforward_development_v1) is enforced, so a two-side epoch that loses
    exactly one side to availability is dropped (not half-trained);
  * ES (early stopping) = LAST 15% of the AVAILABLE TRAINING DAYS (not rows);
    FIT_CORE = earlier available days (canonical walk-forward semantics).

Expected fit count: 4 meta folds x 4 arms x 2 heads = 32 correction fits.
No hyperparameter search. No TEST read.

Performance contract
====================
Build the candidate-event ledger ONCE; build lag1..5 history ONCE; reuse across
all meta folds / heads / arms. Track counters; the expected invariants are
enforced in tests.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from research.liquidity_oracle_atlas.build_decomposed_value_dataset_v1 import (
    build_pay8_from_state,
    PAY8_COLS,
)
from research.liquidity_oracle_atlas.payoff_ratio_model_v1 import (
    REG_PARAMS as _FROZEN_REG_PARAMS,
    EARLY_STOPPING_ROUNDS as _FROZEN_ES_ROUNDS,
)
from research.liquidity_oracle_atlas.walkforward_development_v1 import (
    build_fold_plan,
    decision_day,
    enforce_pair_purity,
    StopV2EmptyEsBlock,
)

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
ES_FRAC = 0.15  # last 15% of available TRAINING DAYS are early-stopping

# Frozen LightGBM regression contract (identical for EVERY arm & head).
# A runtime COPY is taken per fit; only n_jobs is overridden to 1.
FROZEN_ES_ROUNDS = _FROZEN_ES_ROUNDS  # 100
assert FROZEN_ES_ROUNDS == 100, "frozen early-stopping rounds changed"

GROUP = ["symbol", "side"]
KEY = ["symbol", "decision_bar", "side"]
PRED_COLS = ["p_win", "mu_win", "mu_loss"]
STATIC = [*PAY8_COLS, "p_win", "mu_win", "mu_loss"]


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

    A ``horizon`` column (= PRIMARY_HORIZON) is added so the canonical
    two-side epoch pair-purity logic (EPOCH_KEYS = symbol, decision_bar,
    horizon) can group LONG/SHORT sides of the same epoch. ``sample_weight`` is
    inherited from the labels frame (canonical weights, sums to 1 per epoch).
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
    if "sample_weight" not in led.columns:
        raise AssertionError("STOP_LEDGER_MISSING_SAMPLE_WEIGHT")
    # PAY8 may collide with label geometry (log_structural_rr); drop label side.
    led = led.drop(columns=[c for c in PAY8_COLS if c in led.columns])
    led = led.merge(pay, on=KEY, how="left", validate="m:1", indicator="_mp")
    unmatched_pay = int((led["_mp"] != "both").sum())
    led = led.drop(columns=["_mp"])
    if unmatched_pay:
        raise AssertionError(f"STOP_PAY8_JOIN_UNMATCHED n={unmatched_pay}")

    # OOF join: unmatched is EXPECTED (many TD5 labels have no OOF prediction).
    led = led.merge(oof[oof_cols], on=KEY, how="left", validate="m:1")

    # Canonical epoch grouping key for pair purity.
    led["horizon"] = PRIMARY_HORIZON

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
# Canonical outer plan (frozen, verified against OOF shards)                    #
# --------------------------------------------------------------------------- #
def build_canonical_oof_plan(label_path: Path | str = LABEL_PATH,
                             oof_dir: Path | str = OOF_DIR):
    """Build the canonical outer-fold plan ONCE from the FULL TRAIN label
    calendar (all horizons), then hard-verify each outer fold's decision-day
    min/max against the A0_V1_DISJOINT OOF shards.

    Mismatch -> STOP_CANONICAL_OOF_PLAN_MISMATCH.
    """
    lab = pd.read_parquet(label_path)
    if "horizon" not in lab.columns:
        raise AssertionError("STOP_PLAN_LABELS_MISSING_HORIZON")
    plan = build_fold_plan(lab)
    for k in range(plan.n_outer):
        sh = pd.read_parquet(
            Path(oof_dir) / f"A0_V1_DISJOINT_f{k}_{PRIMARY_HORIZON}.parquet")
        d = decision_day(sh)
        smin, smax = str(d.min().date()), str(d.max().date())
        if (smin, smax) != plan.outer[k]:
            raise AssertionError(
                f"STOP_CANONICAL_OOF_PLAN_MISMATCH fold={k} "
                f"shard=({smin},{smax}) plan={plan.outer[k]}")
    return plan


# --------------------------------------------------------------------------- #
# Second-level causal OOF splits (with canonical pair purity)                   #
# --------------------------------------------------------------------------- #
def meta_fold_splits(ledger: pd.DataFrame, plan) -> list:
    """Return list of dicts for k in 1..4 (fold0 is META WARMUP).

    Each entry:
        k            : meta eval fold
        train_mask   : pair-pure AVAILABLE training rows (causal + epoch purity)
        eval_mask    : fold == k rows (predicted once each)
        purity_stats : dict from enforce_pair_purity (dropped epochs, etc.)
        n_avail      : rows passing the causal availability gate (pre-purity)

    A training row requires: layer-0 OOF row exists AND fold < k AND
    decision_time < T AND label_available_time < T, where T is the start of eval
    fold k (from the canonical plan). The canonical two-side epoch pair purity is
    then enforced so no half-epoch is trained.
    """
    folds = sorted(int(f) for f in ledger["fold"].dropna().unique())
    if not folds:
        raise AssertionError("STOP_OOF_FOLD_MISSING")
    if plan is None:
        raise AssertionError("STOP_META_SPLIT_REQUIRES_CANONICAL_PLAN")

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
        # causal availability mask
        avail = (has_fold
                 & (ledger["fold"].to_numpy() < k)
                 & (dt < T).to_numpy()
                 & (av < T).to_numpy())
        purged, purity_stats = enforce_pair_purity(ledger, avail)
        evalm = ledger["fold"].to_numpy() == k
        splits.append({
            "k": k,
            "train_mask": purged,
            "eval_mask": evalm,
            "purity_stats": purity_stats,
            "n_avail": int(avail.sum()),
        })
    return splits


def _day_es_split(ledger: pd.DataFrame, mask: np.ndarray):
    """Split pair-pure available training rows (already head-filtered) into
    FIT_CORE | ES using the canonical DAY-based rule:
        ES  = last ES_FRAC of the AVAILABLE TRAINING DAYS
        FIT = earlier available days.

    HARD STOP if a valid FIT_CORE/ES split cannot be formed.
    Returns (fit_idx, es_idx) as numpy int arrays (possibly empty only on a
    deliberately-empty input mask).
    """
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return idx, idx
    d = pd.to_datetime(ledger.iloc[idx]["decision_time"]).dt.normalize()
    days = np.sort(pd.unique(d))
    n_days = len(days)
    n_es = int(np.floor(n_days * ES_FRAC))
    if n_es < 1:
        raise StopV2EmptyEsBlock(
            f"STOP_V2_EMPTY_ES_BLOCK days={n_days} n_es={n_es}")
    es_day_values = set(days[-n_es:])
    is_es = d.isin(es_day_values).to_numpy(dtype=bool)
    fit_idx = idx[~is_es]
    es_idx = idx[is_es]
    if len(fit_idx) == 0 or len(es_idx) == 0:
        raise StopV2EmptyEsBlock(
            f"STOP_V2_EMPTY_ES_BLOCK fit={len(fit_idx)} es={len(es_idx)}")
    return fit_idx, es_idx


def _fit_correction(X_fit, y_fit, w_fit, X_es, y_es, w_es):
    """Fit one correction head with the FROZEN LightGBM contract and the
    canonical sample weights (fit + early-stopping)."""
    _bump("correction_model_fit_count")
    # Runtime COPY: override ONLY n_jobs for macOS/libomp stability. Never mutate
    # the global frozen dict.
    params = dict(_FROZEN_REG_PARAMS)
    params["n_jobs"] = 1
    model = LGBMRegressor(**params)
    model.fit(
        X_fit, y_fit, sample_weight=w_fit,
        eval_set=[(X_es, y_es)], eval_sample_weight=[w_es],
        callbacks=[early_stopping(FROZEN_ES_ROUNDS, verbose=False),
                   log_evaluation(0)],
    )
    return model


# --------------------------------------------------------------------------- #
# Pipeline                                                                     #
# --------------------------------------------------------------------------- #
def run_sequence_oof(ledger: pd.DataFrame, plan,
                     arms: dict | None = None):
    """Run the 4x4x2 correction-model OOF. Returns (pred_store, meta_rows).

    pred_store[(arm, head)] is an array aligned to ledger index, with the
    corrected mu for the evaluation rows of each meta fold (NaN elsewhere).

    `plan` is REQUIRED (the canonical frozen outer plan). No production fallback
    that derives outer bounds from the current ledger is permitted.
    """
    if plan is None:
        raise AssertionError("STOP_RUN_SEQUENCE_REQUIRES_CANONICAL_PLAN")
    if arms is None:
        arms = ARMS
    splits = meta_fold_splits(ledger, plan)
    if not splits:
        raise AssertionError("STOP_NO_META_FOLDS")

    r_win, r_loss = residual_targets(ledger)
    w_all = ledger["sample_weight"].to_numpy(dtype=float)
    n = len(ledger)
    pred_store = {(arm, head): np.full(n, np.nan)
                  for arm in arms for head in ("win", "loss")}
    meta_rows = {}

    for sp in splits:
        k = sp["k"]
        tr_mask = sp["train_mask"]
        ev_mask = sp["eval_mask"]
        tr_idx = np.where(tr_mask)[0]
        ev_idx = np.where(ev_mask)[0]
        ev_weights = w_all[ev_idx]

        for arm, cols in arms.items():
            Xtr = ledger.iloc[tr_idx][cols].to_numpy(dtype=float)
            Xev = ledger.iloc[ev_idx][cols].to_numpy(dtype=float)

            # ---- WIN residual head (fit on winners only) ----
            if (~np.isnan(r_win))[tr_mask].any():
                fit_idx, es_idx = _day_es_split(ledger, tr_mask & ~np.isnan(r_win))
                wf = w_all[fit_idx]; we = w_all[es_idx]
                mw = _fit_correction(
                    ledger.iloc[fit_idx][cols].to_numpy(dtype=float),
                    r_win[fit_idx], wf,
                    ledger.iloc[es_idx][cols].to_numpy(dtype=float),
                    r_win[es_idx], we)
                r_pred = mw.predict(Xev)
                base = ledger.iloc[ev_idx]["mu_win"].to_numpy(np.float64)
                pred_store[(arm, "win")][ev_idx] = np.maximum(0.0, base + r_pred)

            # ---- LOSS residual head (fit on losers only) ----
            if (~np.isnan(r_loss))[tr_mask].any():
                fit_idx, es_idx = _day_es_split(ledger, tr_mask & ~np.isnan(r_loss))
                wf = w_all[fit_idx]; we = w_all[es_idx]
                ml = _fit_correction(
                    ledger.iloc[fit_idx][cols].to_numpy(dtype=float),
                    r_loss[fit_idx], wf,
                    ledger.iloc[es_idx][cols].to_numpy(dtype=float),
                    r_loss[es_idx], we)
                r_pred_l = ml.predict(Xev)
                base_l = ledger.iloc[ev_idx]["mu_loss"].to_numpy(np.float64)
                pred_store[(arm, "loss")][ev_idx] = np.maximum(
                    0.0, base_l + r_pred_l)

        meta_rows[k] = {
            "train_n": int(len(tr_idx)),
            "eval_n": int(len(ev_idx)),
            "purity": sp["purity_stats"],
            "n_avail": sp["n_avail"],
            "n_purged": int(tr_mask.sum()),
        }
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


def _paired_abs_diff(ledger: pd.DataFrame, target: str,
                     mu_a: np.ndarray, mu_b: np.ndarray,
                     mask: np.ndarray) -> float:
    """Weighted mean of (|t-mu_a| - |t-mu_b|) over mask, where the head-aware
    target is:
        WIN  target t = Y
        LOSS target t = -Y
    >0 => mu_b is a better magnitude fit. Canonical sample weights used.
    """
    y = ledger["episode_return_atr"].to_numpy(dtype=float)
    w = ledger["sample_weight"].to_numpy(dtype=float)
    t = y if target == "win" else -y
    sel = (mask & np.isfinite(mu_a) & np.isfinite(mu_b)
           & np.isfinite(t) & np.isfinite(w) & (w > 0))
    if not sel.any():
        return float("nan")
    d = np.abs(t[sel] - mu_a[sel]) - np.abs(t[sel] - mu_b[sel])
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

    WIN head uses target Y; LOSS head uses target -Y (head-aware).
    """
    res: dict = {}

    m_win_low = _mask_win_low(ledger)
    m_los_low = _mask_loss_low(ledger)
    all_win = (ledger["episode_return_atr"].to_numpy(float) > 0)
    all_los = ~all_win

    def d(arm_a, arm_b, head, mask):
        target = "win" if head == "win" else "loss"
        return _paired_abs_diff(ledger, target,
                                pred_store[(arm_a, head)],
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
# Paired 5-day block bootstrap (frozen kernel)                                  #
# --------------------------------------------------------------------------- #
def paired_block_bootstrap(ledger: pd.DataFrame, target: str,
                           mu_a: np.ndarray, mu_b: np.ndarray,
                           mask: np.ndarray, B: int,
                           block_days: int = BOOTSTRAP_BLOCK_DAYS,
                           seed: int = BOOTSTRAP_SEED) -> dict:
    """Paired block bootstrap of the SAME row-level error difference used by the
    observed statistic.

    block = `block_days` trading days; only complete blocks used; tail days are
    excluded and reported. Uses canonical sample weights. Frozen seed.

    Returns observed_D, bootstrap_mean, ci_low, ci_high (2.5/97.5 percentile),
    n_valid_reps, n_blocks, excluded_tail_days.
    """
    y = ledger["episode_return_atr"].to_numpy(dtype=float)
    w = ledger["sample_weight"].to_numpy(dtype=float)
    t = y if target == "win" else -y
    sel = (mask & np.isfinite(mu_a) & np.isfinite(mu_b)
           & np.isfinite(t) & np.isfinite(w) & (w > 0))
    if not sel.any():
        raise AssertionError("STOP_BOOTSTRAP_NO_VALID_ROWS")
    idx = np.where(sel)[0]
    e = np.abs(t[idx] - mu_a[idx]) - np.abs(t[idx] - mu_b[idx])
    ww = w[idx]
    observed_D = float(np.sum(ww * e) / np.sum(ww))
    days = decision_day(ledger.iloc[idx]).to_numpy()
    means = _block_bootstrap_means(e, ww, days, block_days, B, seed)
    return {
        "observed_D": observed_D,
        "bootstrap_mean": float(np.mean(means)),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "n_valid_reps": int(B),
        "n_blocks": int(len(np.unique(days)) // block_days),
        "excluded_tail_days": int(len(np.unique(days))
                                  - (len(np.unique(days)) // block_days) * block_days),
    }


def _block_bootstrap_means(e, w, days, block_days, B, seed):
    uniq = np.sort(np.unique(days))
    n_days = len(uniq)
    n_blocks = n_days // block_days
    if n_blocks < 1:
        raise AssertionError("STOP_BOOTSTRAP_NO_COMPLETE_BLOCKS")
    day_to_rows = {d: np.where(days == d)[0] for d in uniq}
    blocks = [np.concatenate([day_to_rows[d] for d in uniq[b * block_days:(b + 1) * block_days]])
              for b in range(n_blocks)]
    rng = np.random.default_rng(seed)
    means = np.empty(B)
    for b in range(B):
        chosen = rng.integers(0, n_blocks, size=n_blocks)
        rows = np.concatenate([blocks[c] for c in chosen])
        means[b] = np.sum(w[rows] * e[rows]) / np.sum(w[rows])
    return means


def paired_block_bootstrap_reference(ledger: pd.DataFrame, target: str,
                                     mu_a: np.ndarray, mu_b: np.ndarray,
                                     mask: np.ndarray, B: int,
                                     block_days: int = BOOTSTRAP_BLOCK_DAYS,
                                     seed: int = BOOTSTRAP_SEED) -> dict:
    """Slow/reference implementation of `paired_block_bootstrap`.

    Independently coded with explicit Python loops (no numpy aggregation) so it
    serves as a correctness cross-check. Must agree with the production kernel
    on tiny synthetic data within float tolerance.
    """
    y = ledger["episode_return_atr"].to_numpy(dtype=float)
    w = ledger["sample_weight"].to_numpy(dtype=float)
    t = y if target == "win" else -y
    sel = (mask & np.isfinite(mu_a) & np.isfinite(mu_b)
           & np.isfinite(t) & np.isfinite(w) & (w > 0))
    if not sel.any():
        raise AssertionError("STOP_BOOTSTRAP_NO_VALID_ROWS")
    idx = [i for i in range(len(sel)) if sel[i]]
    e = [abs(t[i] - mu_a[i]) - abs(t[i] - mu_b[i]) for i in idx]
    ww = [float(w[i]) for i in idx]
    observed_D = float(sum(ww[i] * e[i] for i in range(len(e)))
                       / sum(ww))

    days = [decision_day(ledger.iloc[[idx[i]]]).to_numpy()[0] for i in range(len(idx))]
    uniq = sorted(set(days))
    n_blocks = len(uniq) // block_days
    if n_blocks < 1:
        raise AssertionError("STOP_BOOTSTRAP_NO_COMPLETE_BLOCKS")
    day_to_rows = {d: [idx[i] for i in range(len(idx)) if days[i] == d] for d in uniq}
    blocks = [sum([day_to_rows[d] for d in uniq[b * block_days:(b + 1) * block_days]], [])
              for b in range(n_blocks)]

    rng = np.random.default_rng(seed)
    means = []
    for _ in range(B):
        chosen = [int(rng.integers(0, n_blocks)) for _ in range(n_blocks)]
        rows = []
        for c in chosen:
            rows.extend(blocks[c])
        num = sum(ww[idx.index(r)] * e[idx.index(r)] for r in rows)
        den = sum(ww[idx.index(r)] for r in rows)
        means.append(num / den)
    means_arr = np.asarray(means, dtype=float)
    return {
        "observed_D": observed_D,
        "bootstrap_mean": float(np.mean(means_arr)),
        "ci_low": float(np.percentile(means_arr, 2.5)),
        "ci_high": float(np.percentile(means_arr, 97.5)),
        "n_valid_reps": int(B),
        "n_blocks": int(n_blocks),
        "excluded_tail_days": int(len(uniq) - n_blocks * block_days),
    }


# --------------------------------------------------------------------------- #
# Frozen diagnostics (raw, NO interpretation)                                    #
# --------------------------------------------------------------------------- #
def ranking_diagnostic(ledger: pd.DataFrame, pred_store: dict) -> dict:
    """Low-p big-winner ranking (DIAGNOSTIC ONLY).

    Universe: p_win <= 0.40 AND actual winner (Y > 0). For every arm:
      * rank corrected mu_win; take top 20% / bottom 20%;
      * weighted actual-Y spread (top mean - bottom mean).
    Reports winner_spread_arm[arm] and differences relative to S0.
    """
    p = ledger["p_win"].to_numpy(dtype=float)
    y = ledger["episode_return_atr"].to_numpy(dtype=float)
    w = ledger["sample_weight"].to_numpy(dtype=float)
    uni = np.isfinite(p) & (p <= LOW_P_THRESHOLD) & (y > 0)
    out: dict = {"winner_spread_arm": {}, "winner_spread_vs_S0": {}}
    for arm in ARMS:
        mu = pred_store[(arm, "win")]
        sub = uni & np.isfinite(mu)
        if not sub.any():
            continue
        sub_idx = np.where(sub)[0]
        rank = mu[sub_idx].argsort(kind="stable")
        n = len(rank)
        top_n = max(1, int(np.floor(n * 0.20)))
        bottom_n = top_n
        top = rank[-top_n:]
        bottom = rank[:bottom_n]
        ys = y[sub_idx]
        ws = w[sub_idx]
        top_mean = float(np.sum(ws[top] * ys[top]) / np.sum(ws[top]))
        bot_mean = float(np.sum(ws[bottom] * ys[bottom]) / np.sum(ws[bottom]))
        out["winner_spread_arm"][arm] = top_mean - bot_mean
    s0 = out["winner_spread_arm"].get("S0", float("nan"))
    for arm in ARMS:
        if arm in out["winner_spread_arm"]:
            out["winner_spread_vs_S0"][arm] = (
                out["winner_spread_arm"][arm] - s0)
    return out


def value_diagnostic(ledger: pd.DataFrame, pred_store: dict) -> dict:
    """Decomposed value diagnostic (DIAGNOSTIC ONLY).

    value_score = p_win * corrected_mu_win - (1-p_win) * corrected_mu_loss.
    Reports weighted MAE / MSE vs true episode_return_atr. No policy / PnL.
    """
    p = ledger["p_win"].to_numpy(dtype=float)
    y = ledger["episode_return_atr"].to_numpy(dtype=float)
    w = ledger["sample_weight"].to_numpy(dtype=float)
    out: dict = {"value_mae": {}, "value_mse": {}}
    for arm in ARMS:
        mw = pred_store[(arm, "win")]
        ml = pred_store[(arm, "loss")]
        sub = np.isfinite(mw) & np.isfinite(ml) & np.isfinite(p)
        if not sub.any():
            continue
        vs = p[sub] * mw[sub] - (1.0 - p[sub]) * ml[sub]
        err = vs - y[sub]
        ws = w[sub]
        mae = float(np.sum(ws * np.abs(err)) / np.sum(ws))
        mse = float(np.sum(ws * err * err) / np.sum(ws))
        out["value_mae"][arm] = mae
        out["value_mse"][arm] = mse
    return out


# --------------------------------------------------------------------------- #
# Feature-list hash (frozen expected hashes live in the test file)              #
# --------------------------------------------------------------------------- #
def feature_list_hash(cols) -> str:
    h = hashlib.sha256()
    for c in cols:
        h.update((str(c) + "\n").encode("utf-8"))
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Small T1 smoke (AG/AU only) -- verification helper, NOT an artifact producer  #
# --------------------------------------------------------------------------- #
def run_t1(*, symbols=("AG", "AU"), plan=None) -> dict:
    """Run the full sequence OOF pipeline on a small symbol subset for
    verification. Produces NO committed artifact."""
    reset_counters()
    if plan is None:
        plan = build_canonical_oof_plan()
    ledger = build_candidate_ledger(symbols=list(symbols))
    ledger = build_candidate_event_history(ledger)
    preds, meta_rows = run_sequence_oof(ledger, plan)
    metrics = analyze(ledger, preds)
    diags = {
        "ranking": ranking_diagnostic(ledger, preds),
        "value": value_diagnostic(ledger, preds),
    }
    return {
        "n_rows_ledger": int(len(ledger)),
        "n_rows_with_oof": int(ledger["fold"].notna().sum()),
        "meta_rows": meta_rows,
        "counters": counters(),
        "metrics": metrics,
        "diagnostics": diags,
        "pred_store_eval_nonnull_win_S3F": int(
            np.count_nonzero(np.isfinite(preds[("S3F", "win")]))),
    }
