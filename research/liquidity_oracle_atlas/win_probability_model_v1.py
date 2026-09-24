"""FUTURE-R9A-M15-WIN-PROBABILITY-MODEL-V1.

R9A: Win-Probability Model. One fixed LightGBM classifier per horizon
(TD1 / TD3 / TD5) trained on the WIN33 feature contract (side-oriented
STRUCT33, exactly 33 features). The target is the economic win indicator

    Win_H = 1[episode_return_atr > 0].

This model answers "how likely is this side to be profitable?" ONLY. It must
not consume any Payoff (PAY8) feature, and the Payoff model must not consume
WIN33. The two research models are kept independently auditable; the final
decision layer (R9C) combines their predictions by arithmetic only.

TEST labels are NEVER read during fitting (§23). The TEST prediction path is
implemented but gated and NOT executed until authorized.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping

from research.liquidity_oracle_atlas.train_direction_model_ag_v1 import (
    BASE_PARAMS,
)
from research.liquidity_oracle_atlas.build_decomposed_value_dataset_v1 import (
    ARTIFACT_DIR,  # noqa: F401 (kept for import-surface symmetry)
    SIDE_KEY,
    HORIZONS,
    WIN33_COLS,
    load_win_features,
    load_labels,
    load_manifest,
    sha256_file,  # noqa: F401
)

TASK_ID = "FUTURE-R9A-M15-WIN-PROBABILITY-MODEL-V1"
BASE_SHA = "f3ca3a04317126e8f35afe7430d7aea202a9175a"

MODEL_DIR = os.path.join("artifacts", "decomposed_value_v1", "models", "win")
MODEL_FILES = {H: os.path.join(MODEL_DIR, f"{H}_win.txt") for H in HORIZONS}
MODEL_MANIFEST = os.path.join(MODEL_DIR, "model_manifest.json")

CLF_PARAMS = dict(BASE_PARAMS)
EARLY_STOPPING_ROUNDS = 100
N_PROB_DECILES = 10

COUNTERS = {
    "state_artifact_loads": 0,
    "feature_artifact_loads": 0,
    "train_label_loads": 0,
    "val_label_loads": 0,
    "test_label_reads_during_fit": 0,
    "classifier_fits": 0,
    "hyperparameter_search_count": 0,
}


def _bump(name, n=1):
    COUNTERS[name] = COUNTERS.get(name, 0) + int(n)


def reset_counters():
    for k in list(COUNTERS):
        COUNTERS[k] = 0


def _git_head_sha():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def win33_schema_sha256():
    h = hashlib.sha256()
    for c in WIN33_COLS:
        h.update(str(c).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# 1. Model bundle                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class WinProbabilityBundle:
    classifier: object
    horizon: str
    feature_schema_sha256: str


# --------------------------------------------------------------------------- #
# 2. Data loading (TRAIN / VAL only; TEST never read during fit)                #
# --------------------------------------------------------------------------- #
def load_fit_frames():
    feats = load_win_features()
    _bump("feature_artifact_loads")
    out = {}
    for stage in ("train", "val"):
        lab = load_labels(stage)
        _bump(f"{stage}_label_loads")
        df = lab.merge(feats, on=list(SIDE_KEY), how="left",
                       validate="m:1", indicator=True)
        if (df["_merge"].to_numpy(object) != "both").any():
            raise RuntimeError("STOP_R9A_FEATURE_JOIN_UNMATCHED")
        df = df.drop(columns=["_merge"])
        out[stage] = df[df["bracket_eligible"] & (df["sample_weight"] > 0)]
    return out


def _matrices(df, horizon):
    q = df[df["horizon"] == horizon]
    X = q[list(WIN33_COLS)].to_numpy(np.float32)
    y = (q["episode_return_atr"].to_numpy(float) > 0).astype(np.int32)
    w = q["sample_weight"].to_numpy(float)
    return X, y, w


# --------------------------------------------------------------------------- #
# 3. Fitting (no search, no class weighting)                                    #
# --------------------------------------------------------------------------- #
def fit_bundle(Xtr, ytr, wtr, Xv, yv, wv, horizon):
    clf = LGBMClassifier(**CLF_PARAMS)
    clf.fit(Xtr, ytr, sample_weight=wtr,
            eval_set=[(Xv, yv)], eval_sample_weight=[wv],
            callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    _bump("classifier_fits")
    return WinProbabilityBundle(
        classifier=clf, horizon=horizon,
        feature_schema_sha256=win33_schema_sha256())


def fit_all(frames=None, verbose: bool = True):
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
# 4. Prediction                                                                #
# --------------------------------------------------------------------------- #
def predict_win_probability(bundle: WinProbabilityBundle, X):
    if hasattr(bundle.classifier, "predict_proba"):
        return np.asarray(bundle.classifier.predict_proba(X)[:, 1], dtype=float)
    return np.asarray(bundle.classifier.predict(X), dtype=float)


# --------------------------------------------------------------------------- #
# 5. VAL diagnostics                                                           #
# --------------------------------------------------------------------------- #
def _brier(y01, p, w):
    return float(np.average((p - y01) ** 2, weights=w))


def _logloss(y01, p, w):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.average(y01 * np.log(p) + (1 - y01) * np.log(1 - p),
                             weights=w))


def _auc(y01, p, w):
    from sklearn.metrics import roc_auc_score
    if np.unique(y01).size < 2:
        return float("nan")
    return float(roc_auc_score(y01, np.asarray(p), sample_weight=np.asarray(w)))


def val_diagnostics(bundles, frames):
    """VAL only. Ten fixed probability deciles; they never become thresholds."""
    out = {}
    for H in HORIZONS:
        b = bundles[H]
        X, y, w = _matrices(frames["val"], H)
        p = predict_win_probability(b, X)
        y01 = y.astype(np.int32)
        ret = frames["val"].loc[
            frames["val"]["horizon"] == H, "episode_return_atr"].to_numpy(float)

        dec = pd.qcut(pd.Series(p).rank(method="first"),
                      N_PROB_DECILES, labels=False).to_numpy()
        rows = []
        for d in range(N_PROB_DECILES):
            m = dec == d
            if not m.any():
                continue
            ww = w[m]
            yy = y01[m]
            p_act = float(np.average(yy, weights=ww))
            mean_p = float(np.average(p[m], weights=ww))
            mean_ret = float(np.average(ret[m], weights=ww))
            rows.append({
                "decile": d + 1,
                "n_rows": int(m.sum()),
                "mean_predicted_pwin": mean_p,
                "actual_win_rate": p_act,
                "calibration_error": abs(mean_p - p_act),
                "actual_mean_return_atr": mean_ret,
            })
        out[H] = {
            "n_val_rows": int(len(X)),
            "brier": _brier(y01, p, w),
            "logloss": _logloss(y01, p, w),
            "auc": _auc(y01, p, w),
            "prob_deciles": rows,
            "best_iteration": int(
                getattr(b.classifier, "best_iteration_", -1) or -1),
        }
    return out


# --------------------------------------------------------------------------- #
# 6. Freeze models                                                             #
# --------------------------------------------------------------------------- #
def freeze_models(bundles, diag=None, verbose: bool = True, frames=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    hashes = {}
    for H in HORIZONS:
        b = bundles[H]
        b.classifier.booster_.save_model(MODEL_FILES[H])
        hashes[os.path.basename(MODEL_FILES[H])] = sha256_file(MODEL_FILES[H])

    dec_manifest = load_manifest()
    artifact_sha = dec_manifest.get("artifact_sha256", {})

    manifest = {
        "task_id": TASK_ID,
        "base_sha": BASE_SHA,
        "generator_code_sha": _git_head_sha(),
        "win33_schema_sha256": win33_schema_sha256(),
        "horizons": list(HORIZONS),
        "model_count": len(hashes),
        "clf_params": CLF_PARAMS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "upstream": {
            "state_sha256": artifact_sha.get("state_v1.parquet"),
            "win_features_sha256": artifact_sha.get("win_features_v1.parquet"),
            "labels_train_sha256": artifact_sha.get("labels_train_v1.parquet"),
            "labels_val_sha256": artifact_sha.get("labels_val_v1.parquet"),
        },
        "best_iteration": {H: int(getattr(bundles[H].classifier,
                                          "best_iteration_", -1) or -1)
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
# 7. TEST prediction path — implemented, NOT executed (§31)                    #
# --------------------------------------------------------------------------- #
def load_bundle(horizon: str) -> WinProbabilityBundle:
    from lightgbm import Booster
    return WinProbabilityBundle(
        classifier=Booster(model_file=MODEL_FILES[horizon]),
        horizon=horizon, feature_schema_sha256=win33_schema_sha256())


def predict_test(allow_test: bool = False, authorized_review_sha: Optional[str] = None,
                 write_artifacts: bool = True):
    """Batch-predict Pwin for every TEST state, ONE pass per horizon.

    Implemented but NOT executed here. Requires explicit TEST authorization.
    Returns a DataFrame with SIDE_KEY + one ``<H>_p_win`` column per horizon.
    The unified TEST prediction artifact (Pwin + payoff) is assembled by R10.
    """
    if allow_test is not True:
        raise RuntimeError("STOP_R9A_TEST_PREDICTION_NOT_AUTHORIZED")
    if not authorized_review_sha:
        raise RuntimeError("STOP_R9A_TEST_AUTHORIZED_REVIEW_SHA_REQUIRED")
    head = _git_head_sha()
    if head != authorized_review_sha:
        raise RuntimeError(
            f"STOP_R9A_GENERATOR_SHA_MISMATCH head={head} "
            f"authorized_review_sha={authorized_review_sha}")

    from research.liquidity_oracle_atlas.direction_null_baseline_v1 import (
        build_frozen_split)
    split = build_frozen_split()
    t2 = np.datetime64(split["cal"]["cuts"][1], "ns")

    feats = load_win_features()
    state = pd.read_parquet(
        os.path.join("artifacts", "decomposed_value_v1", "state_v1.parquet"),
        columns=["bar_index", "decision_time"])
    dt = state["decision_time"].to_numpy("datetime64[ns]")
    test_bars = set(state["bar_index"].to_numpy(np.int64)[dt >= t2])
    mask = np.fromiter(
        (b in test_bars for b in feats["decision_bar"].to_numpy(np.int64)),
        dtype=bool, count=len(feats))
    sub = feats[mask].reset_index(drop=True)
    X = sub[list(WIN33_COLS)].to_numpy(np.float32)

    out = sub[list(SIDE_KEY)].copy()
    for H in HORIZONS:
        out[f"{H}_p_win"] = predict_win_probability(load_bundle(H), X)
    if write_artifacts:
        out.to_parquet(
            os.path.join("artifacts", "decomposed_value_v1",
                         "win_probability_test_v1.parquet"), index=False)
    return out


if __name__ == "__main__":
    t0 = time.time()
    reset_counters()
    frames = load_fit_frames()
    bundles = fit_all(frames)
    diag = val_diagnostics(bundles, frames)
    manifest = freeze_models(bundles, diag, frames=frames)
    print("runtime_sec", time.time() - t0)
