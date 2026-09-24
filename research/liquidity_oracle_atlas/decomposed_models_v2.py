"""FUTURE-R11-R14 V2 — generic three-head decomposed model core (§34-§36).

TRADING_METRICS: NOT_APPLICABLE
reason: No trading action has been defined.

What is deliberately NOT here
-----------------------------
* No hyperparameter search — the frozen V1 LightGBM family is imported.
* No alternative algorithm — plan §32 holds the model family fixed so that any
  improvement can be attributed to the STATE REPRESENTATION, not capacity.
* No fourth model — the Composer is pure arithmetic (§14) and has zero learned
  parameters (§47 #17).

Per horizon a complete fit is exactly three estimators (§12):
    1 classifier      : p      = P(Y > 0 | X)
    1 regressor       : mu_W   = E[ Y | Y > 0, X]
    1 regressor       : mu_L   = E[-Y | Y <= 0, X]
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping

# Frozen V1 params, imported read-only (plan §12 / §34).
from research.liquidity_oracle_atlas.train_direction_model_ag_v1 import (
    BASE_PARAMS,
)
from research.liquidity_oracle_atlas.win_probability_model_v1 import (
    CLF_PARAMS,
    EARLY_STOPPING_ROUNDS,
)
from research.liquidity_oracle_atlas.payoff_ratio_model_v1 import REG_PARAMS

assert set(BASE_PARAMS) <= set(CLF_PARAMS), "frozen V1 CLF params drifted"
assert REG_PARAMS["objective"] == "regression"

# Plan §51 process isolation. The frozen V1 params carry n_jobs=-1, which
# deterministically SEGFAULTS the interpreter on this macOS/libomp setup
# (reproduced: identical call crashes with n_jobs=-1 and succeeds with
# n_jobs=1, giving the same best_iteration 58 / 228 / 70). n_jobs is a
# THREADING knob only -- it does not change the learned model -- so it is
# forced to 1 here. This is NOT a model change and must not be reported as one.
FORCE_N_JOBS = 1


def runtime_params(p):
    """Frozen params with the threading knob forced to the stable value."""
    out = dict(p)
    out["n_jobs"] = FORCE_N_JOBS
    return out


def safe_n_estimators(v) -> int:
    """LightGBM rejects n_estimators=0, which early stopping can legitimately
    select when no tree improves the ES score. Clamp to the minimum valid
    value; the raw ES choice stays visible in the fit ledger / audit."""
    return max(1, int(v))

# Plan §26 fixed development bootstrap contract.
BOOTSTRAP_B = 5000
BOOTSTRAP_SEED = 20260925
BOOTSTRAP_BLOCK_DAYS = 5

# Plan §35: refit uses best_iteration determined on FIT_CORE -> ES.
MIN_REQUIRED_ROWS = 50


class StopV2Fit(RuntimeError):
    pass


class StopV2InsufficientRows(StopV2Fit):
    pass


@dataclass
class FitRecord:
    """One estimator fit, recorded for §50 fit-count accounting."""

    arch: str
    fold: int
    horizon: str
    head: str          # "win" | "win_mag" | "loss_mag"
    stage: str         # "es" | "refit"
    n_rows: int
    best_iteration: Optional[int] = None


@dataclass
class ThreeHeads:
    clf: Any
    mw: Any
    ml: Any
    horizon: str
    best_iteration: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# §34 generic three-head fitter                                                #
# --------------------------------------------------------------------------- #
def fit_three_heads(train_df, es_df, cols, *, clf_params=None, reg_params=None,
                    callbacks=None, payoff_cols=None):
    """Fit p / mu_W / mu_L with early stopping ONLY on the inner ES block.

    `cols` feeds the classifier; `payoff_cols` feeds BOTH magnitude regressors
    and defaults to `cols`. They only differ for the R12 ablations A1/A2, which
    deliberately give one head extra information to test the mechanism.
    """
    clf_params = runtime_params(CLF_PARAMS if clf_params is None else clf_params)
    reg_params = runtime_params(REG_PARAMS if reg_params is None else reg_params)

    # A LightGBM callback holds PER-BOOSTER state. One shared instance passed to
    # all three estimators corrupts that state and segfaults the interpreter, so
    # each fit gets its own instance.
    def fresh_callbacks():
        return [early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]

    clf_cb = list(callbacks) if callbacks is not None else fresh_callbacks()
    mw_cb = list(callbacks) if callbacks is not None else fresh_callbacks()
    ml_cb = list(callbacks) if callbacks is not None else fresh_callbacks()

    cols = list(cols)
    payoff_cols = list(cols if payoff_cols is None else payoff_cols)
    Xtr = train_df[cols].to_numpy(np.float32)
    Xes = es_df[cols].to_numpy(np.float32)
    Xtr_p = train_df[payoff_cols].to_numpy(np.float32)
    Xes_p = es_df[payoff_cols].to_numpy(np.float32)

    ytr = train_df["episode_return_atr"].to_numpy(float)
    yes = es_df["episode_return_atr"].to_numpy(float)

    wtr = train_df["sample_weight"].to_numpy(float)
    wes = es_df["sample_weight"].to_numpy(float)

    if len(Xtr) < MIN_REQUIRED_ROWS or len(Xes) < MIN_REQUIRED_ROWS:
        raise StopV2InsufficientRows(
            f"STOP_V2_INSUFFICIENT_ROWS train={len(Xtr)} es={len(Xes)}")

    win_tr = ytr > 0
    win_es = yes > 0

    clf = LGBMClassifier(**clf_params)
    clf.fit(
        Xtr, win_tr.astype(np.uint8), sample_weight=wtr,
        eval_set=[(Xes, win_es.astype(np.uint8))],
        eval_sample_weight=[wes],
        callbacks=clf_cb,
    )

    mw = LGBMRegressor(**reg_params)
    mw.fit(
        Xtr_p[win_tr], ytr[win_tr], sample_weight=wtr[win_tr],
        eval_set=[(Xes_p[win_es], yes[win_es])],
        eval_sample_weight=[wes[win_es]],
        callbacks=mw_cb,
    )

    loss_tr = ~win_tr
    loss_es = ~win_es

    ml = LGBMRegressor(**reg_params)
    ml.fit(
        Xtr_p[loss_tr], -ytr[loss_tr], sample_weight=wtr[loss_tr],
        eval_set=[(Xes_p[loss_es], -yes[loss_es])],
        eval_sample_weight=[wes[loss_es]],
        callbacks=ml_cb,
    )
    return clf, mw, ml


def _best_iteration(model) -> Optional[int]:
    for attr in ("best_iteration_", "best_iteration"):
        v = getattr(model, attr, None)
        if v:
            return int(v)
    return None


# --------------------------------------------------------------------------- #
# §35 refit on all pre-outer data with the frozen inner iteration              #
# --------------------------------------------------------------------------- #
def refit_three_heads(all_train_df, cols, *, best_iters, horizon, arch="",
                      fold=-1, clf_params=None, reg_params=None,
                      payoff_cols=None):
    """Refit with n_estimators pinned to the ES-derived best_iteration.

    No early stopping here: the tree count is already frozen, so the OUTER
    block cannot influence model selection (§47 #18).
    """
    cols = list(cols)
    payoff_cols = list(cols if payoff_cols is None else payoff_cols)
    X = all_train_df[cols].to_numpy(np.float32)
    Xp = all_train_df[payoff_cols].to_numpy(np.float32)
    y = all_train_df["episode_return_atr"].to_numpy(float)
    w = all_train_df["sample_weight"].to_numpy(float)

    clf_p = runtime_params(clf_params or CLF_PARAMS)
    reg_p = runtime_params(reg_params or REG_PARAMS)
    clf_p["n_estimators"] = safe_n_estimators(best_iters["win"])
    reg_p_win = dict(reg_p)
    reg_p_win["n_estimators"] = safe_n_estimators(best_iters["win_mag"])
    reg_p_loss = dict(reg_p)
    reg_p_loss["n_estimators"] = safe_n_estimators(best_iters["loss_mag"])

    win = y > 0
    clf = LGBMClassifier(**clf_p)
    clf.fit(X, win.astype(np.uint8), sample_weight=w)

    mw = LGBMRegressor(**reg_p_win)
    mw.fit(Xp[win], y[win], sample_weight=w[win])

    loss = ~win
    ml = LGBMRegressor(**reg_p_loss)
    ml.fit(Xp[loss], -y[loss], sample_weight=w[loss])

    return ThreeHeads(
        clf=clf, mw=mw, ml=ml, horizon=horizon,
        best_iteration=dict(best_iters),
    )


# --------------------------------------------------------------------------- #
# §14 / §36 composer + prediction                                              #
# --------------------------------------------------------------------------- #
def compose(p, mu_w, mu_l):
    """Deterministic Composer: EV_C = p*mu_W - (1-p)*mu_L. Zero learned params."""
    p = np.asarray(p, float)
    mu_w = np.maximum(np.asarray(mu_w, float), 0.0)
    mu_l = np.maximum(np.asarray(mu_l, float), 0.0)

    rr = np.where(mu_l > 0, mu_w / mu_l, np.inf)
    p_be = np.where((mu_w + mu_l) > 0, mu_l / (mu_w + mu_l), np.nan)
    ev = p * mu_w - (1.0 - p) * mu_l

    return {
        "p_win": p,
        "mu_win": mu_w,
        "mu_loss": mu_l,
        "rr": rr,
        "p_break_even": p_be,
        "ev": ev,
    }


def predict_decomposed(clf, mw, ml, X, clf_X=None):
    """X feeds both magnitude heads; `clf_X` overrides the classifier matrix
    (only the A1/A2 ablations give the two heads different states)."""
    Xw = X if clf_X is None else clf_X
    p = clf.predict_proba(Xw)[:, 1]
    mu_w = np.maximum(mw.predict(X), 0.0)
    mu_l = np.maximum(ml.predict(X), 0.0)
    ev = p * mu_w - (1.0 - p) * mu_l
    rr = np.where(mu_l > 0, mu_w / mu_l, np.inf)
    return p, mu_w, mu_l, rr, ev


# --------------------------------------------------------------------------- #
# §13 primary development loss + secondary diagnostics                         #
# --------------------------------------------------------------------------- #
def weighted_ev_mse(y, ev, w):
    """L_EV = E_w[(Y - EV_C)^2] — the PRIMARY development loss (§13)."""
    y = np.asarray(y, float)
    ev = np.asarray(ev, float)
    w = np.asarray(w, float)
    m = np.isfinite(y) & np.isfinite(ev) & np.isfinite(w)
    return float(np.average((y[m] - ev[m]) ** 2, weights=w[m]))


def weighted_ev_mae(y, ev, w):
    y = np.asarray(y, float)
    ev = np.asarray(ev, float)
    w = np.asarray(w, float)
    m = np.isfinite(y) & np.isfinite(ev) & np.isfinite(w)
    return float(np.average(np.abs(y[m] - ev[m]), weights=w[m]))


def weighted_brier(win, p, w):
    y = np.asarray(win, float)
    p = np.asarray(p, float)
    w = np.asarray(w, float)
    m = np.isfinite(y) & np.isfinite(p) & np.isfinite(w)
    return float(np.average((p[m] - y[m]) ** 2, weights=w[m]))


def weighted_logloss(win, p, w, eps=1e-12):
    y = np.asarray(win, float)
    p = np.clip(np.asarray(p, float), eps, 1.0 - eps)
    w = np.asarray(w, float)
    m = np.isfinite(y) & np.isfinite(p) & np.isfinite(w)
    ll = -(y[m] * np.log(p[m]) + (1.0 - y[m]) * np.log(1.0 - p[m]))
    return float(np.average(ll, weights=w[m]))


def weighted_mae(y, pred, w):
    y = np.asarray(y, float)
    pr = np.asarray(pred, float)
    w = np.asarray(w, float)
    m = np.isfinite(y) & np.isfinite(pr) & np.isfinite(w)
    if not m.any():
        return float("nan")
    return float(np.average(np.abs(y[m] - pr[m]), weights=w[m]))


def weighted_mean(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w)
    if not m.any() or w[m].sum() <= 0:
        return float("nan")
    return float(np.average(x[m], weights=w[m]))


# --------------------------------------------------------------------------- #
# §24 one-standard-error selection                                             #
# --------------------------------------------------------------------------- #
def one_se_select(results):
    """Pick the SIMPLEST candidate within one SE of the best mean EV MSE."""
    if not results:
        raise ValueError("one_se_select: empty results")
    best = min(results, key=lambda r: r["mean_ev_mse"])
    threshold = best["mean_ev_mse"] + best["se_ev_mse"]
    eligible = [r for r in results if r["mean_ev_mse"] <= threshold]
    return dict(
        min(eligible, key=lambda r: (r["n_features"], r["mean_ev_mse"])),
        one_se_threshold=threshold,
        best_candidate=best["candidate"],
        n_eligible=len(eligible),
    )


def fold_summary(candidate: str, n_features: int, fold_losses: Sequence[float]):
    arr = np.asarray(list(fold_losses), float)
    return {
        "candidate": candidate,
        "n_features": int(n_features),
        "fold_ev_mse": [float(v) for v in arr],
        "mean_ev_mse": float(np.mean(arr)),
        "se_ev_mse": (float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
                      if len(arr) > 1 else 0.0),
        "n_folds": int(len(arr)),
    }


# --------------------------------------------------------------------------- #
# §26 paired development bootstrap (explanatory only)                          #
# --------------------------------------------------------------------------- #
def _complete_block_starts(n_days: int, block: int = BOOTSTRAP_BLOCK_DAYS):
    n_blocks = n_days // block
    return n_blocks, n_blocks * block


def paired_block_bootstrap(d_candidate, d_baseline, *, b=BOOTSTRAP_B,
                           seed=BOOTSTRAP_SEED, block=BOOTSTRAP_BLOCK_DAYS):
    """Paired 5-day block bootstrap over per-TRADING-DAY loss differences.

    ``d_*`` are per-day mean differences (candidate minus baseline) aligned on
    the same trading days. Only complete blocks are resampled. Returns point,
    CI and the frozen interpretation label (§26).
    """
    d = np.asarray(d_candidate, float) - np.asarray(d_baseline, float)
    d = d[np.isfinite(d)]
    n_days = len(d)
    if n_days == 0:
        return {"point": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "status": "NO_DEV_IMPROVEMENT",
                "n_days": 0, "n_blocks": 0}
    n_blocks, usable = _complete_block_starts(n_days, block)
    if n_blocks < 1:
        return {"point": float(np.mean(d)), "ci_low": float("nan"),
                "ci_high": float("nan"),
                "status": "NO_DEV_IMPROVEMENT",
                "n_days": n_days, "n_blocks": 0}
    usable_d = d[:usable]
    starts = np.arange(0, usable, block)
    rng = np.random.default_rng(seed)
    draws = np.empty(b, float)
    for i in range(b):
        pick = rng.integers(0, len(starts), size=len(starts))
        sample = np.concatenate(
            [usable_d[s:s + block] for s in starts[pick]])
        draws[i] = float(np.mean(sample))
    point = float(np.mean(usable_d))
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    if hi < 0:
        status = "SUPPORTED_DEV_IMPROVEMENT"
    elif point < 0:
        status = "PROMISING_DEV_IMPROVEMENT"
    else:
        status = "NO_DEV_IMPROVEMENT"
    return {"point": point, "ci_low": lo, "ci_high": hi, "status": status,
            "n_days": n_days, "n_blocks": n_blocks,
            "excluded_tail_days": int(n_days - usable)}


def daily_loss_series(frame, ev_col="ev_c", y_col="episode_return_atr",
                      w_col="sample_weight", day_col=None):
    """Per-trading-day weighted squared EV error — the §26 day-loss series."""
    dts = pd.to_datetime(frame["decision_time"]).dt.normalize()
    y = frame[y_col].to_numpy(float)
    ev = frame[ev_col].to_numpy(float)
    w = frame[w_col].to_numpy(float)
    sq = (y - ev) ** 2
    m = np.isfinite(sq) & np.isfinite(w)
    df = pd.DataFrame({"day": dts.to_numpy(), "num": np.where(m, sq * w, 0.0),
                       "den": np.where(m, w, 0.0)})
    g = df.groupby("day").sum()
    out = (g["num"] / g["den"].replace(0.0, np.nan)).sort_index()
    return out


# --------------------------------------------------------------------------- #
# Fit ledger (§50)                                                             #
# --------------------------------------------------------------------------- #
FIT_LEDGER: list[FitRecord] = []


def reset_ledger():
    FIT_LEDGER.clear()


def record_fit(**kw):
    FIT_LEDGER.append(FitRecord(**kw))
    return FIT_LEDGER[-1]


def ledger_frame() -> pd.DataFrame:
    if not FIT_LEDGER:
        return pd.DataFrame(columns=[f.name for f in FitRecord.__dataclass_fields__.values()])
    return pd.DataFrame([r.__dict__ for r in FIT_LEDGER])


def dump_ledger(path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ledger_frame().to_json(path, orient="records", indent=2)
    return path


# --------------------------------------------------------------------------- #
# Walk-forward OOF unit (plan §4-§6, §35, §47 #18-#21)                         #
# --------------------------------------------------------------------------- #
META_COLS: tuple[str, ...] = (
    "symbol", "decision_bar", "side", "horizon", "decision_time",
    "label_available_time", "sample_weight", "episode_return_atr",
)

OOF_COLS: tuple[str, ...] = (
    "symbol", "decision_bar", "decision_time", "side", "horizon", "fold",
    "p_win", "mu_win", "mu_loss", "predicted_rr", "ev_c",
    "episode_return_atr", "sample_weight",
)


def fit_unit(arch, fold, outer_bounds, es_frac, horizon, dev_path, out_path):
    """One (architecture, fold, horizon) unit.

    ES fit -> best_iteration -> refit on ALL pre-outer rows -> predict the
    outer block exactly once. Six estimator fits in total (3 ES + 3 refit).
    """
    from research.liquidity_oracle_atlas import (
        walkforward_development_v1 as W,
    )

    needed = list(dict.fromkeys(
        list(META_COLS) + list(arch.win) + list(arch.payoff)))
    frame = pd.read_parquet(dev_path, columns=needed)
    frame = frame[frame["horizon"] == horizon].reset_index(drop=True)

    # The outer calendar is decided once by the parent on the FULL frame and
    # handed down, so every horizon/fold sees identical boundaries.
    plan = W.FoldPlan(days=(), warmup_end="",
                      outer=(tuple(outer_bounds),), es_frac=es_frac)
    s = W.fold_split(frame, plan, 0)

    core = frame[s.fit_core]
    es = frame[s.es]
    pre_outer = frame[s.fit_core | s.es]

    clf_e, mw_e, ml_e = fit_three_heads(
        core, es, arch.win, payoff_cols=arch.payoff)
    for head, m in (("win", clf_e), ("win_mag", mw_e), ("loss_mag", ml_e)):
        record_fit(arch=arch.name, fold=fold, horizon=horizon, head=head,
                   stage="es", n_rows=int(len(core)),
                   best_iteration=_best_iteration(m))

    best = {
        "win": _best_iteration(clf_e),
        "win_mag": _best_iteration(mw_e),
        "loss_mag": _best_iteration(ml_e),
    }
    if any(v is None for v in best.values()):
        raise StopV2Fit(f"STOP_V2_MISSING_BEST_ITERATION {arch.name} {best}")

    heads = refit_three_heads(
        pre_outer, arch.win, payoff_cols=arch.payoff, best_iters=best,
        horizon=horizon)
    for head, m in (("win", heads.clf), ("win_mag", heads.mw),
                    ("loss_mag", heads.ml)):
        record_fit(arch=arch.name, fold=fold, horizon=horizon, head=head,
                   stage="refit", n_rows=int(len(pre_outer)),
                   best_iteration=best[head])

    out = frame[s.outer]
    Xp = out[list(arch.payoff)].to_numpy(np.float32)
    Xw = out[list(arch.win)].to_numpy(np.float32)
    p, mu_w, mu_l, rr, ev = predict_decomposed(
        heads.clf, heads.mw, heads.ml, Xp, clf_X=Xw)

    shard = pd.DataFrame({
        "symbol": out["symbol"].to_numpy(),
        "decision_bar": out["decision_bar"].to_numpy(),
        "decision_time": out["decision_time"].to_numpy(),
        "side": out["side"].to_numpy(),
        "horizon": out["horizon"].to_numpy(),
        "fold": np.full(len(out), int(fold), dtype=np.int64),
        "p_win": p, "mu_win": mu_w, "mu_loss": mu_l,
        "predicted_rr": rr, "ev_c": ev,
        "episode_return_atr": out["episode_return_atr"].to_numpy(float),
        "sample_weight": out["sample_weight"].to_numpy(float),
    })
    shard = shard[list(OOF_COLS)]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    shard.to_parquet(out_path, index=False)

    return {
        "arch": arch.name,
        "fold": int(fold),
        "horizon": horizon,
        "best_iteration": best,
        "n_fit_core": int(s.n_fit_core),
        "n_es": int(s.n_es),
        "n_outer": int(len(out)),
        "purity": s.purity_stats,
        "out_path": out_path,
        "ev_mse": weighted_ev_mse(
            shard["episode_return_atr"], shard["ev_c"], shard["sample_weight"]),
    }


def worker_main(args_path: str) -> dict:
    """Subprocess entry (plan §51): one unit per process, no shared libomp."""
    from research.liquidity_oracle_atlas import (
        decomposed_value_features_v2 as F,
    )
    with open(args_path) as f:
        args = json.load(f)
    arch = F.get_arch(args["arch"])
    if arch is None:
        raise StopV2Fit(f"STOP_V2_UNKNOWN_ARCH {args['arch']}")
    res = fit_unit(
        arch=arch,
        fold=int(args["fold"]),
        outer_bounds=tuple(args["outer_bounds"]),
        es_frac=float(args["es_frac"]),
        horizon=args["horizon"],
        dev_path=args["dev_path"],
        out_path=args["out_path"],
    )
    with open(args["out_meta"], "w") as f:
        json.dump(res, f, indent=2, default=str)
    return res


if __name__ == "__main__":
    import sys
    worker_main(sys.argv[1])
