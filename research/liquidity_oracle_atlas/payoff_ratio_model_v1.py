"""FUTURE-R9B-M15-PAYOFF-RATIO-MODEL-V1.

R9B: Payoff-Ratio Model. One conceptual Payoff module per horizon (TD1 / TD3 /
TD5) implemented as TWO conditional LightGBM regressors on the PAY8 feature
contract (8 structural / payoff geometry features):

    Win  head : trained on Y > 0 rows, target  Y
    Loss head : trained on Y <= 0 rows, target -Y

so that

    mu_W = E[Y | Y>0],  mu_L = E[-Y | Y<=0],  RR = mu_W / mu_L.

There is no honest single-trade RR label (a trade observes W or L, never both
counterfactuals), hence two conditional heads. The Payoff module must NOT
consume WIN33 / trend / oracle / probability features; it learns opportunity
geometry, not direction.

TEST labels are NEVER read during fitting. The TEST prediction path is
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
from lightgbm import LGBMRegressor, early_stopping

from research.liquidity_oracle_atlas.train_direction_model_ag_v1 import (
    BASE_PARAMS,
)
from research.liquidity_oracle_atlas.build_decomposed_value_dataset_v1 import (
    SIDE_KEY,
    HORIZONS,
    PAY8_COLS,
    load_payoff_features,
    load_labels,
    load_manifest,
    sha256_file,
)

TASK_ID = "FUTURE-R9B-M15-PAYOFF-RATIO-MODEL-V1"
BASE_SHA = "f3ca3a04317126e8f35afe7430d7aea202a9175a"

MODEL_DIR = os.path.join("artifacts", "decomposed_value_v1", "models", "payoff")
MODEL_FILES = {
    H: (os.path.join(MODEL_DIR, f"{H}_win_mag.txt"),
        os.path.join(MODEL_DIR, f"{H}_loss_mag.txt"))
    for H in HORIZONS
}
MODEL_MANIFEST = os.path.join(MODEL_DIR, "model_manifest.json")

REG_PARAMS = dict(BASE_PARAMS)
REG_PARAMS.update(objective="regression", metric="l2")
EARLY_STOPPING_ROUNDS = 100
N_RR_DECILES = 10

COUNTERS = {
    "state_artifact_loads": 0,
    "feature_artifact_loads": 0,
    "train_label_loads": 0,
    "val_label_loads": 0,
    "test_label_reads_during_fit": 0,
    "payoff_regressor_fits": 0,
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


def pay8_schema_sha256():
    h = hashlib.sha256()
    for c in PAY8_COLS:
        h.update(str(c).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# 1. Bundle                                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class PayoffRatioBundle:
    win_magnitude_model: object
    loss_magnitude_model: object
    horizon: str
    feature_schema_sha256: str


# --------------------------------------------------------------------------- #
# 2. Data loading (TRAIN / VAL only)                                           #
# --------------------------------------------------------------------------- #
def load_fit_frames():
    feats = load_payoff_features()
    _bump("feature_artifact_loads")
    # The R8 label frame also carries geometry columns (e.g. log_structural_rr)
    # that coincide with PAY8 feature names. The feature parquet is the
    # authoritative PAY8 source, so drop any such collision from the label side
    # before the SIDE_KEY merge (otherwise pandas appends _x/_y and _matrices
    # can no longer locate the bare PAY8 column names).
    out = {}
    for stage in ("train", "val"):
        lab = load_labels(stage)
        _bump(f"{stage}_label_loads")
        lab = lab.drop(columns=[c for c in PAY8_COLS if c in lab.columns])
        df = lab.merge(feats, on=list(SIDE_KEY), how="left",
                       validate="m:1", indicator=True)
        if (df["_merge"].to_numpy(object) != "both").any():
            raise RuntimeError("STOP_R9B_FEATURE_JOIN_UNMATCHED")
        df = df.drop(columns=["_merge"])
        out[stage] = df[df["bracket_eligible"] & (df["sample_weight"] > 0)]
    return out


def _matrices(df, horizon):
    q = df[df["horizon"] == horizon]
    X = q[list(PAY8_COLS)].to_numpy(np.float32)
    y = q["episode_return_atr"].to_numpy(float)
    w = q["sample_weight"].to_numpy(float)
    return X, y, w


# --------------------------------------------------------------------------- #
# 3. Fitting (two conditional heads)                                            #
# --------------------------------------------------------------------------- #
def fit_bundle(Xtr, ytr, wtr, Xv, yv, wv, horizon):
    # Win head: Y > 0 only.
    tr_w = ytr > 0
    va_w = yv > 0
    win_m = LGBMRegressor(**REG_PARAMS)
    win_m.fit(Xtr[tr_w], ytr[tr_w], sample_weight=wtr[tr_w],
              eval_set=[(Xv[va_w], yv[va_w])], eval_sample_weight=[wv[va_w]],
              callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    _bump("payoff_regressor_fits")

    # Loss head: Y <= 0 only, target -Y.
    tr_l = ytr <= 0
    va_l = yv <= 0
    loss_m = LGBMRegressor(**REG_PARAMS)
    loss_m.fit(Xtr[tr_l], (-ytr)[tr_l], sample_weight=wtr[tr_l],
               eval_set=[(Xv[va_l], (-yv)[va_l])], eval_sample_weight=[wv[va_l]],
               callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    _bump("payoff_regressor_fits")

    return PayoffRatioBundle(
        win_magnitude_model=win_m, loss_magnitude_model=loss_m,
        horizon=horizon, feature_schema_sha256=pay8_schema_sha256())


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
def predict_payoff(bundle: PayoffRatioBundle, X):
    mu_w = np.maximum(np.asarray(bundle.win_magnitude_model.predict(X), float), 0.0)
    mu_l = np.maximum(np.asarray(bundle.loss_magnitude_model.predict(X), float), 0.0)
    rr = np.where(mu_l > 0, mu_w / mu_l, np.inf)
    return mu_w, mu_l, rr


# --------------------------------------------------------------------------- #
# 5. VAL diagnostics                                                           #
# --------------------------------------------------------------------------- #
def _wmae(pred, true, w):
    return float(np.average(np.abs(pred - true), weights=w))


def _wrmse(pred, true, w):
    return float(np.sqrt(np.average((pred - true) ** 2, weights=w)))


def val_diagnostics(bundles, frames):
    """VAL only. Ten fixed predicted-RR deciles; they never become thresholds.

    Every bin hard-checks the payoff identities (tolerance 1e-12).
    """
    out = {}
    for H in HORIZONS:
        b = bundles[H]
        X, y, w = _matrices(frames["val"], H)
        mu_w, mu_l, rr = predict_payoff(b, X)

        win_mag_wmae = _wmae(mu_w[y > 0], y[y > 0], w[y > 0]) if (y > 0).any() else np.nan
        win_mag_wrmse = _wrmse(mu_w[y > 0], y[y > 0], w[y > 0]) if (y > 0).any() else np.nan
        loss_mag_wmae = _wmae(mu_l[y <= 0], (-y)[y <= 0], w[y <= 0]) if (y <= 0).any() else np.nan
        loss_mag_wrmse = _wrmse(mu_l[y <= 0], (-y)[y <= 0], w[y <= 0]) if (y <= 0).any() else np.nan

        dec = pd.qcut(pd.Series(rr).rank(method="first"),
                      N_RR_DECILES, labels=False).to_numpy()
        rows = []
        for d in range(N_RR_DECILES):
            m = dec == d
            if not m.any():
                continue
            ww = w[m]
            yy = y[m]
            win = yy > 0
            avg_win = (float(np.average(yy[win], weights=ww[win]))
                       if win.any() else np.nan)
            avg_loss = (float(np.average((-yy)[~win], weights=ww[~win]))
                        if (~win).any() else np.nan)
            p_act = float(np.average(win.astype(float), weights=ww))
            mean_ret = float(np.average(yy, weights=ww))
            actual_rr = (float(avg_win / avg_loss)
                         if (avg_loss and avg_loss > 0) else np.nan)
            ev_ident = (p_act * avg_win - (1.0 - p_act) * avg_loss)
            dev = abs(ev_ident - mean_ret)
            if np.isfinite(dev) and dev > 1e-12:
                raise RuntimeError(
                    f"STOP_R9B_PAYOFF_IDENTITY horizon={H} decile={d + 1} dev={dev}")
            rows.append({
                "decile": d + 1,
                "n_rows": int(m.sum()),
                "mean_predicted_rr": float(
                    np.average(rr[m], weights=ww) if np.isfinite(rr[m]).any() else np.nan),
                "actual_win_rate": p_act,
                "actual_avg_win": avg_win,
                "actual_avg_loss": avg_loss,
                "actual_payoff_ratio": actual_rr,
                "actual_mean_return_atr": mean_ret,
                "actual_ev_identity_abs_dev": float(dev),
            })
        out[H] = {
            "n_val_rows": int(len(X)),
            "win_magnitude_wmae": win_mag_wmae,
            "win_magnitude_wrmse": win_mag_wrmse,
            "loss_magnitude_wmae": loss_mag_wmae,
            "loss_magnitude_wrmse": loss_mag_wrmse,
            "rr_deciles": rows,
            "best_iteration": {
                "win_mag": int(getattr(b.win_magnitude_model, "best_iteration_", -1) or -1),
                "loss_mag": int(getattr(b.loss_magnitude_model, "best_iteration_", -1) or -1)},
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
        b.win_magnitude_model.booster_.save_model(MODEL_FILES[H][0])
        b.loss_magnitude_model.booster_.save_model(MODEL_FILES[H][1])
        for p in MODEL_FILES[H]:
            hashes[os.path.basename(p)] = sha256_file(p)

    dec_manifest = load_manifest()
    artifact_sha = dec_manifest.get("artifact_sha256", {})

    manifest = {
        "task_id": TASK_ID,
        "base_sha": BASE_SHA,
        "generator_code_sha": _git_head_sha(),
        "pay8_schema_sha256": pay8_schema_sha256(),
        "horizons": list(HORIZONS),
        "model_count": len(hashes),
        "reg_params": REG_PARAMS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "upstream": {
            "state_sha256": artifact_sha.get("state_v1.parquet"),
            "payoff_features_sha256": artifact_sha.get("payoff_features_v1.parquet"),
            "labels_train_sha256": artifact_sha.get("labels_train_v1.parquet"),
            "labels_val_sha256": artifact_sha.get("labels_val_v1.parquet"),
        },
        "best_iteration": {
            H: {"win_mag": int(getattr(bundles[H].win_magnitude_model,
                                       "best_iteration_", -1) or -1),
                "loss_mag": int(getattr(bundles[H].loss_magnitude_model,
                                        "best_iteration_", -1) or -1)}
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
def load_bundle(horizon: str) -> PayoffRatioBundle:
    from lightgbm import Booster
    return PayoffRatioBundle(
        win_magnitude_model=Booster(model_file=MODEL_FILES[horizon][0]),
        loss_magnitude_model=Booster(model_file=MODEL_FILES[horizon][1]),
        horizon=horizon, feature_schema_sha256=pay8_schema_sha256())


def predict_test(allow_test: bool = False, authorized_review_sha: Optional[str] = None,
                 write_artifacts: bool = True):
    """Batch-predict payoff (mu_W, mu_L, RR) for every TEST state, one pass/horizon.

    Implemented but NOT executed here. Requires explicit TEST authorization.
    """
    if allow_test is not True:
        raise RuntimeError("STOP_R9B_TEST_PREDICTION_NOT_AUTHORIZED")
    if not authorized_review_sha:
        raise RuntimeError("STOP_R9B_TEST_AUTHORIZED_REVIEW_SHA_REQUIRED")
    head = _git_head_sha()
    if head != authorized_review_sha:
        raise RuntimeError(
            f"STOP_R9B_GENERATOR_SHA_MISMATCH head={head} "
            f"authorized_review_sha={authorized_review_sha}")

    from research.liquidity_oracle_atlas.direction_null_baseline_v1 import (
        build_frozen_split)
    split = build_frozen_split()
    t2 = np.datetime64(split["cal"]["cuts"][1], "ns")

    feats = load_payoff_features()
    state = pd.read_parquet(
        os.path.join("artifacts", "decomposed_value_v1", "state_v1.parquet"),
        columns=["bar_index", "decision_time"])
    dt = state["decision_time"].to_numpy("datetime64[ns]")
    test_bars = set(state["bar_index"].to_numpy(np.int64)[dt >= t2])
    mask = np.fromiter(
        (b in test_bars for b in feats["decision_bar"].to_numpy(np.int64)),
        dtype=bool, count=len(feats))
    sub = feats[mask].reset_index(drop=True)
    X = sub[list(PAY8_COLS)].to_numpy(np.float32)

    out = sub[list(SIDE_KEY)].copy()
    for H in HORIZONS:
        mu_w, mu_l, rr = predict_payoff(load_bundle(H), X)
        out[f"{H}_mu_win"] = mu_w
        out[f"{H}_mu_loss"] = mu_l
        out[f"{H}_predicted_rr"] = rr
    if write_artifacts:
        out.to_parquet(
            os.path.join("artifacts", "decomposed_value_v1",
                         "payoff_ratio_test_v1.parquet"), index=False)
    return out


if __name__ == "__main__":
    t0 = time.time()
    reset_counters()
    frames = load_fit_frames()
    bundles = fit_all(frames)
    diag = val_diagnostics(bundles, frames)
    manifest = freeze_models(bundles, diag, frames=frames)
    print("runtime_sec", time.time() - t0)
