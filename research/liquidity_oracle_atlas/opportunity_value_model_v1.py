"""FUTURE-R9-M15-OPPORTUNITY-VALUE-MODEL-V1.

Fits the frozen Opportunity Value model family (§24-§26) and predicts

    EV_hat = p_win * mu_win - (1 - p_win) * mu_loss
    RR_hat = mu_win / mu_loss
    p*     = mu_loss / (mu_win + mu_loss) = 1 / (1 + RR_hat)
    take   <=> EV_hat > 0   <=>   p_win > p*

There is NO fixed RR threshold and NO fixed win-probability threshold (§9/§28).

TEST is NEVER used for fitting, feature selection, model selection,
hyperparameter selection, threshold selection or calibration fitting (§23).
The TEST prediction path is implemented but NOT executed until authorized (§31).
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping

from research.liquidity_oracle_atlas.train_direction_model_ag_v1 import (
    BASE_PARAMS,
)
from research.liquidity_oracle_atlas.structural_renewal_dataset_v1 import (
    ARTIFACT_DIR,
    HORIZONS,
    OPP36,
    SIDE_KEY,
    STATE_PARQUET,
    SIDE_FEATURES_PARQUET,
    LABEL_PARQUETS,
    opp36_schema_sha256,
    sha256_file,
)

TASK_ID = "FUTURE-R9-M15-OPPORTUNITY-VALUE-MODEL-V1"
BASE_SHA = "f3ca3a04317126e8f35afe7430d7aea202a9175a"

MODEL_DIR = os.path.join(ARTIFACT_DIR, "models")
MODEL_MANIFEST = os.path.join(MODEL_DIR, "model_manifest.json")
MODEL_FILES = {H: (os.path.join(MODEL_DIR, f"{H}_win.txt"),
                   os.path.join(MODEL_DIR, f"{H}_win_mag.txt"),
                   os.path.join(MODEL_DIR, f"{H}_loss_mag.txt"))
               for H in HORIZONS}
MODEL_MANIFEST_EVIDENCE = os.path.join(
    "research", "liquidity_oracle_atlas", "evidence",
    "opportunity_value_model_v1_summary.json")

# §25: regressor parameters are MECHANICALLY derived from the frozen trainer.
CLF_PARAMS = dict(BASE_PARAMS)
REG_PARAMS = dict(BASE_PARAMS)
REG_PARAMS.update(objective="regression", metric="l2")

EARLY_STOPPING_ROUNDS = 100
N_EV_DECILES = 10

COUNTERS = {
    "state_artifact_loads": 0,
    "feature_artifact_loads": 0,
    "train_label_loads": 0,
    "val_label_loads": 0,
    "test_label_reads_during_fit": 0,
    "environment_loads": 0,
    "geometry_extract_calls": 0,
    "model_fit_count": 0,
    "hyperparameter_search_count": 0,
}


def _bump(name, n=1):
    COUNTERS[name] = COUNTERS.get(name, 0) + int(n)


def reset_counters():
    for k in COUNTERS:
        COUNTERS[k] = 0


def _git_head_sha():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# 1. Model bundle (§26)                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class OpportunityValueBundle:
    win_classifier: object
    win_magnitude_regressor: object
    loss_magnitude_regressor: object
    horizon: str
    feature_schema_sha256: str


# --------------------------------------------------------------------------- #
# 2. Data loading (§47)                                                         #
# --------------------------------------------------------------------------- #
def load_fit_frames():
    """Load state + side features + TRAIN/VAL labels. TEST is NEVER read."""
    feats = pd.read_parquet(SIDE_FEATURES_PARQUET)
    _bump("feature_artifact_loads")
    out = {}
    for stage in ("train", "val"):
        lab = pd.read_parquet(LABEL_PARQUETS[stage])
        _bump(f"{stage}_label_loads")
        # G / L / log_structural_rr are carried by BOTH frames (label audit copy
        # and canonical OPP36 copy). They are identical by construction, so the
        # canonical model columns win and the duplicate is dropped.
        dup = [c for c in ("G", "L", "log_structural_rr") if c in lab.columns]
        df = lab.drop(columns=dup).merge(feats, on=list(SIDE_KEY), how="left",
                                         validate="m:1", indicator=True)
        if (df["_merge"].to_numpy(object) != "both").any():
            raise RuntimeError("STOP_R9_FEATURE_JOIN_UNMATCHED")
        df = df.drop(columns=["_merge"])
        # NOTE: canonical STRUCT33 carries genuine NaN (e.g. liquidity distance
        # when no active level exists). LightGBM handles missing values natively
        # and the frozen Direction trainer never imputes them, so rows are NOT
        # dropped here -- only the join completeness is enforced.
        out[stage] = df[df["bracket_eligible"] & (df["sample_weight"] > 0)]
    return out


def _matrices(df, horizon):
    q = df[df["horizon"] == horizon]
    X = q[list(OPP36)].to_numpy(np.float32)
    y = q["episode_return_atr"].to_numpy(float)
    w = q["sample_weight"].to_numpy(float)
    return X, y, w


# --------------------------------------------------------------------------- #
# 3. Fitting (§24-§26)                                                          #
# --------------------------------------------------------------------------- #
def fit_bundle(Xtr, ytr, wtr, Xv, yv, wv, horizon):
    """Exactly three LightGBM fits. No search, no class weighting."""
    ytr_win = (ytr > 0).astype(np.int32)
    yv_win = (yv > 0).astype(np.int32)

    clf = LGBMClassifier(**CLF_PARAMS)
    clf.fit(Xtr, ytr_win, sample_weight=wtr,
            eval_set=[(Xv, yv_win)], eval_sample_weight=[wv],
            callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    _bump("model_fit_count")

    mtr = ytr > 0
    mva = yv > 0
    reg_w = LGBMRegressor(**REG_PARAMS)
    reg_w.fit(Xtr[mtr], ytr[mtr], sample_weight=wtr[mtr],
              eval_set=[(Xv[mva], yv[mva])], eval_sample_weight=[wv[mva]],
              callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    _bump("model_fit_count")

    ltr = ytr <= 0
    lva = yv <= 0
    reg_l = LGBMRegressor(**REG_PARAMS)
    reg_l.fit(Xtr[ltr], (-ytr)[ltr], sample_weight=wtr[ltr],
              eval_set=[(Xv[lva], (-yv)[lva])], eval_sample_weight=[wv[lva]],
              callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    _bump("model_fit_count")

    return OpportunityValueBundle(
        win_classifier=clf, win_magnitude_regressor=reg_w,
        loss_magnitude_regressor=reg_l, horizon=horizon,
        feature_schema_sha256=opp36_schema_sha256())


def fit_all(frames=None, verbose=True):
    if frames is None:
        frames = load_fit_frames()
    bundles = {}
    for H in HORIZONS:
        Xtr, ytr, wtr = _matrices(frames["train"], H)
        Xv, yv, wv = _matrices(frames["val"], H)
        bundles[H] = fit_bundle(Xtr, ytr, wtr, Xv, yv, wv, H)
        if verbose:
            print(f"{H}: train={len(Xtr)} val={len(Xv)}", flush=True)
    return bundles


# --------------------------------------------------------------------------- #
# 4. Prediction mathematics (§27 / §28)                                         #
# --------------------------------------------------------------------------- #
def _predict_p_win(model, X):
    """Accept both the sklearn classifier and a frozen raw Booster."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return np.asarray(model.predict(X), dtype=float)


def predict_opportunity_value(bundle: OpportunityValueBundle, X):
    p = _predict_p_win(bundle.win_classifier, X)
    mu_win = np.maximum(bundle.win_magnitude_regressor.predict(X), 0.0)
    mu_loss = np.maximum(bundle.loss_magnitude_regressor.predict(X), 0.0)
    ev = p * mu_win - (1.0 - p) * mu_loss
    denom = mu_win + mu_loss
    p_break_even = np.where(denom > 0, mu_loss / denom, 1.0)
    rr = np.where(mu_loss > 0, mu_win / mu_loss, np.inf)
    take = (ev > 0.0) & np.isfinite(ev)
    return {"p_win": p, "mu_win": mu_win, "mu_loss": mu_loss,
            "predicted_rr": rr, "p_break_even": p_break_even,
            "predicted_ev": ev, "take": take}


# --------------------------------------------------------------------------- #
# 5. VAL diagnostics (§29)                                                      #
# --------------------------------------------------------------------------- #
def _brier(y01, p, w):
    return float(np.average((p - y01) ** 2, weights=w))


def _logloss(y01, p, w):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.average(y01 * np.log(p) + (1 - y01) * np.log(1 - p),
                             weights=w))


def _auc(y01, p, w):
    from sklearn.metrics import roc_auc_score
    y01 = np.asarray(y01)
    if np.unique(y01).size < 2:
        return float("nan")
    return float(roc_auc_score(y01, np.asarray(p), sample_weight=np.asarray(w)))


def _wmae(pred, true, w):
    return float(np.average(np.abs(pred - true), weights=w))


def _wrmse(pred, true, w):
    return float(np.sqrt(np.average((pred - true) ** 2, weights=w)))


def val_diagnostics(bundles, frames):
    """VAL only. Ten EV deciles are fixed ex ante; they never become thresholds."""
    out = {}
    for H in HORIZONS:
        b = bundles[H]
        X, y, w = _matrices(frames["val"], H)
        pr = predict_opportunity_value(b, X)
        y01 = (y > 0).astype(np.int32)

        dec = pd.qcut(pd.Series(pr["predicted_ev"]).rank(method="first"),
                      N_EV_DECILES, labels=False)
        dec = np.asarray(dec, dtype=int)
        rows = []
        for d in range(N_EV_DECILES):
            m = dec == d
            if not m.any():
                continue
            ww = w[m]
            yy = y[m]
            win = yy > 0
            # FIX7: the previous code averaged the WEIGHTS themselves.
            avg_win = (float(np.average(yy[win], weights=ww[win]))
                       if win.any() else np.nan)
            avg_loss = (float(np.average((-yy)[~win], weights=ww[~win]))
                        if (~win).any() else np.nan)
            p_act = float(np.average(win.astype(float), weights=ww))
            mean_ret = float(np.average(yy, weights=ww))
            ev_ident = (p_act * avg_win - (1.0 - p_act) * avg_loss)
            dev = abs(ev_ident - mean_ret)
            if np.isfinite(dev) and dev > 1e-12:
                raise RuntimeError(
                    f"STOP_R9_EV_DECILE_IDENTITY horizon={H} decile={d + 1} "
                    f"dev={dev}")
            denom = avg_win + avg_loss
            rows.append({
                "decile": d + 1,
                "n_rows": int(m.sum()),
                "mean_predicted_ev": float(
                    np.average(pr["predicted_ev"][m], weights=ww)),
                "actual_mean_return_atr": mean_ret,
                "actual_win_rate": p_act,
                "actual_avg_win": avg_win,
                "actual_avg_loss": avg_loss,
                "actual_payoff_ratio": (float(avg_win / avg_loss)
                                        if (avg_loss and avg_loss > 0) else np.nan),
                "actual_break_even_win_rate": (float(avg_loss / denom)
                                               if denom > 0 else np.nan),
                "actual_ev_identity": float(ev_ident),
                "actual_ev_identity_abs_dev": float(dev),
            })
        out[H] = {
            "n_val_rows": int(len(X)),
            "brier": _brier(y01, pr["p_win"], w),
            "logloss": _logloss(y01, pr["p_win"], w),
            "auc": _auc(y01, pr["p_win"], w),
            "win_magnitude_wmae": _wmae(pr["mu_win"][y01 == 1], y[y01 == 1],
                                        w[y01 == 1]) if (y01 == 1).any() else np.nan,
            "win_magnitude_wrmse": _wrmse(pr["mu_win"][y01 == 1], y[y01 == 1],
                                          w[y01 == 1]) if (y01 == 1).any() else np.nan,
            "loss_magnitude_wmae": _wmae(pr["mu_loss"][y01 == 0], (-y)[y01 == 0],
                                         w[y01 == 0]) if (y01 == 0).any() else np.nan,
            "loss_magnitude_wrmse": _wrmse(pr["mu_loss"][y01 == 0], (-y)[y01 == 0],
                                           w[y01 == 0]) if (y01 == 0).any() else np.nan,
            "ev_deciles": rows,
            "best_iteration": {
                "win": int(getattr(b.win_classifier, "best_iteration_", -1) or -1),
                "win_mag": int(getattr(b.win_magnitude_regressor, "best_iteration_", -1) or -1),
                "loss_mag": int(getattr(b.loss_magnitude_regressor, "best_iteration_", -1) or -1)},
        }
    return out


# --------------------------------------------------------------------------- #
# 6. Freeze models (§30)                                                        #
# --------------------------------------------------------------------------- #
def freeze_models(bundles, diag=None, verbose=True, frames=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    hashes = {}
    for H in HORIZONS:
        b = bundles[H]
        b.win_classifier.booster_.save_model(MODEL_FILES[H][0])
        b.win_magnitude_regressor.booster_.save_model(MODEL_FILES[H][1])
        b.loss_magnitude_regressor.booster_.save_model(MODEL_FILES[H][2])
        for p in MODEL_FILES[H]:
            hashes[os.path.basename(p)] = sha256_file(p)
    def _best(model):
        v = getattr(model, "best_iteration_", None)
        return int(v) if v else -1

    manifest = {
        "task_id": TASK_ID,
        "base_sha": BASE_SHA,
        "generator_code_sha": _git_head_sha(),
        "opp36_schema_sha256": opp36_schema_sha256(),
        "horizons": list(HORIZONS),
        "model_count": len(hashes),
        "clf_params": CLF_PARAMS,
        "reg_params": REG_PARAMS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "best_iteration": {
            H: {"win": _best(bundles[H].win_classifier),
                "win_mag": _best(bundles[H].win_magnitude_regressor),
                "loss_mag": _best(bundles[H].loss_magnitude_regressor)}
            for H in HORIZONS},
        "train_rows": int(len(frames["train"])) if frames else None,
        "val_rows": int(len(frames["val"])) if frames else None,
        "model_sha256": hashes,
        "performance": dict(COUNTERS),
    }
    with open(MODEL_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    if verbose:
        print(json.dumps({k: v for k, v in manifest.items()
                          if k != "model_sha256"}, indent=2, default=str))
    return manifest


# --------------------------------------------------------------------------- #
# 7. TEST prediction path — implemented, NOT executed (§31)                     #
# --------------------------------------------------------------------------- #
TEST_PRED_PARQUET = os.path.join(
    ARTIFACT_DIR, "opportunity_value_predictions_test_v1.parquet")


def load_bundle(horizon: str) -> OpportunityValueBundle:
    """Load a frozen horizon bundle from the persisted boosters."""
    from lightgbm import Booster
    w, wm, lm = MODEL_FILES[horizon]
    return OpportunityValueBundle(
        win_classifier=Booster(model_file=w),
        win_magnitude_regressor=Booster(model_file=wm),
        loss_magnitude_regressor=Booster(model_file=lm),
        horizon=horizon, feature_schema_sha256=opp36_schema_sha256())


def predict_test(allow_test: bool = False, authorized_review_sha: Optional[str] = None,
                 write_artifacts: bool = True, verbose: bool = False):
    """§31: batch-predict BOTH sides for every TEST state, ONE pass per horizon.

    Implemented but NOT executed here. Requires explicit TEST authorization.
    """
    if allow_test is not True:
        raise RuntimeError("STOP_R9_TEST_PREDICTION_NOT_AUTHORIZED")
    if not authorized_review_sha:
        raise RuntimeError("STOP_R9_TEST_AUTHORIZED_REVIEW_SHA_REQUIRED")
    head = _git_head_sha()
    if head != authorized_review_sha:
        raise RuntimeError(
            f"STOP_R9_GENERATOR_SHA_MISMATCH head={head} "
            f"authorized_review_sha={authorized_review_sha}")

    from research.liquidity_oracle_atlas.direction_null_baseline_v1 import (
        build_frozen_split)
    split = build_frozen_split()
    t2 = np.datetime64(split["cal"]["cuts"][1], "ns")

    feats = pd.read_parquet(SIDE_FEATURES_PARQUET)
    state = pd.read_parquet(STATE_PARQUET, columns=[
        "symbol", "bar_index", "decision_time"])
    dt = state["decision_time"].to_numpy("datetime64[ns]")
    test_keys = set(zip(state["symbol"].to_numpy(object)[dt >= t2],
                        state["bar_index"].to_numpy(np.int64)[dt >= t2]))
    mask = np.fromiter(
        ((s, int(b)) in test_keys
         for s, b in zip(feats["symbol"].to_numpy(object),
                         feats["decision_bar"].to_numpy(np.int64))),
        dtype=bool, count=len(feats))
    sub = feats[mask].reset_index(drop=True)
    X = sub[list(OPP36)].to_numpy(np.float32)

    out = sub[list(SIDE_KEY)].copy()
    for H in HORIZONS:
        pr = predict_opportunity_value(load_bundle(H), X)
        out[f"{H}_p_win"] = pr["p_win"]
        out[f"{H}_mu_win"] = pr["mu_win"]
        out[f"{H}_mu_loss"] = pr["mu_loss"]
        out[f"{H}_predicted_rr"] = pr["predicted_rr"]
        out[f"{H}_p_break_even"] = pr["p_break_even"]
        out[f"{H}_predicted_ev"] = pr["predicted_ev"]
        out[f"{H}_take"] = pr["take"]
    if write_artifacts:
        out.to_parquet(TEST_PRED_PARQUET, index=False)
    return out


if __name__ == "__main__":
    t0 = time.time()
    reset_counters()
    frames = load_fit_frames()
    bundles = fit_all(frames)
    diag = val_diagnostics(bundles, frames)
    manifest = freeze_models(bundles, diag)
    print("runtime_sec", time.time() - t0)
