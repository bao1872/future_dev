"""FUTURE-R13.6 / R13.7 -- Output Integration experiment (plan frozen).

Scientific question (plan §0):
    The three frozen base-model outputs (p_win, mu_win, mu_loss) each carry
    information, but the current hard-coded EV formula (pW - (1-p)L) loses
    most of the p_win ranking power once the payoff-supplied break-even
    threshold is introduced.  Can a *meta* integrator that only consumes the
    three outputs identify a smaller, genuinely profitable candidate subset
    WITHOUT re-fitting any base model or touching E9 / DEV VAL / TEST?

Hard constraints (plan §2, §30):
    * Only p_win, mu_win, mu_loss enter learned models (as z1,z2,z3).
    * No WIN33 / PAY8 / SPACE / VOL / symbol / side / time / regime features.
    * No base-model refit, no E9 fit/read, no DEV VAL, no old TEST.
    * All priors / scalers / GMM / thresholds use ONLY meta-train history
      (folds < k, decision_time < cutoff, label_available_time < cutoff).

This module implements:
    * meta_state(z) and WeightedScaler (plan §12, §13)
    * M1_POLY2_RIDGE, M2_SPLINE_RIDGE, M3_PGM3, M4_GMM_MOE3 (plan §8-§17)
    * causal baselines B0_P / B1_EV_C / B2_MARGIN / B3_EV_W (plan §7)
    * second-level cross-fit engine (plan §5, §18, §19)
    * block bootstrap + paired contrast vs B0_P (plan §25)
    * evaluation / evidence writers (plan §31, §32, §34)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

import research.liquidity_oracle_atlas.run_decomposed_v2_research as R
from research.liquidity_oracle_atlas.decomposed_value_closure_audit_v1 import (
    load_td5_oof,
    sha256_of,
    wmean,
)

# --------------------------------------------------------------------------- #
# Constants (plan §1, §34)                                                    #
# --------------------------------------------------------------------------- #
HORIZON = "td5"
N_FOLDS = 5
BOOTSTRAP_B = 5000
BOOTSTRAP_SEED = 20260925
BOOTSTRAP_BLOCK = 5  # complete 5-trading-day blocks only (plan §25)

REVIEWED_BASE_SHA = "b1ee4d3e94c9709f176afe915c6f68ad7c047153"
ARCH_PRIMARY = "A0_V1_DISJOINT"
ARCH_SECONDARY = "A1_SHARE_TO_WIN"
POP_ROOT = "ROOT_CANDIDATE_TD5"

EPS = 1e-6
KEY = ["symbol", "decision_bar", "side", "horizon"]

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))  # .../future_dev
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "decomposed_value_v2")
EVIDENCE_DIR = R.EVIDENCE_DIR
if not os.path.isabs(EVIDENCE_DIR):
    EVIDENCE_DIR = os.path.join(PROJECT_ROOT, EVIDENCE_DIR)

OOF_PARQUET = os.path.join(ARTIFACT_DIR, "meta_integration_oof_v1.parquet")

ATLAS_CSV = os.path.join(EVIDENCE_DIR, "output_integration_atlas_v1.csv")
SUMMARY_JSON = os.path.join(EVIDENCE_DIR, "meta_integration_summary_v1.json")
MODELS_CSV = os.path.join(EVIDENCE_DIR, "meta_integration_models_v1.csv")
FOLD_CSV = os.path.join(EVIDENCE_DIR, "meta_integration_fold_stability_v1.csv")
SYMSIDE_CSV = os.path.join(EVIDENCE_DIR, "meta_integration_symbol_side_v1.csv")
REGIMES_CSV = os.path.join(EVIDENCE_DIR, "meta_integration_regimes_v1.csv")
MANIFEST_JSON = os.path.join(EVIDENCE_DIR, "meta_integration_manifest_v1.json")

META_MODELS = [
    "B0_P", "B1_EV_C", "B2_MARGIN", "B3_EV_W",
    "M1_POLY2_RIDGE", "M2_SPLINE_RIDGE", "M3_PGM3", "M4_GMM_MOE3",
]
MODEL_LABEL = {
    "B0_P": "P", "B1_EV_C": "EV_C", "B2_MARGIN": "MARGIN", "B3_EV_W": "EV_W",
    "M1_POLY2_RIDGE": "POLY2_RIDGE", "M2_SPLINE_RIDGE": "SPLINE_RIDGE",
    "M3_PGM3": "PGM3", "M4_GMM_MOE3": "GMM_MOE3",
}
PGM_MODELS = ("M3_PGM3", "M4_GMM_MOE3")


# --------------------------------------------------------------------------- #
# Primitive meta state (plan §2, §12)                                          #
# --------------------------------------------------------------------------- #
def meta_state(df: pd.DataFrame) -> np.ndarray:
    """Map the three frozen outputs to the only meta-model inputs z1,z2,z3."""
    p = np.clip(df["p_win"].to_numpy(float), EPS, 1.0 - EPS)
    mw = np.maximum(df["mu_win"].to_numpy(float), 0.0)
    ml = np.maximum(df["mu_loss"].to_numpy(float), 0.0)
    return np.column_stack([
        np.log(p / (1.0 - p)),
        np.log1p(mw),
        np.log1p(ml),
    ])


# --------------------------------------------------------------------------- #
# Weighted scaler (plan §13)                                                   #
# --------------------------------------------------------------------------- #
class WeightedScaler:
    def fit(self, X, w):
        X = np.asarray(X, float)
        w = np.asarray(w, float)
        self.mean_ = np.average(X, axis=0, weights=w)
        self.var_ = np.average((X - self.mean_) ** 2, axis=0, weights=w)
        self.scale_ = np.sqrt(np.maximum(self.var_, 1e-12))
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.mean_) / self.scale_


# --------------------------------------------------------------------------- #
# Baseline score helpers (plan §7)                                             #
# --------------------------------------------------------------------------- #
def _clip_out(df: pd.DataFrame):
    p = np.clip(df["p_win"].to_numpy(float), 0.0, 1.0)
    mw = np.maximum(df["mu_win"].to_numpy(float), 0.0)
    ml = np.maximum(df["mu_loss"].to_numpy(float), 0.0)
    return p, mw, ml


def _ev_c(df: pd.DataFrame) -> np.ndarray:
    p, mw, ml = _clip_out(df)
    return p * mw - (1.0 - p) * ml


def _margin(df: pd.DataFrame) -> np.ndarray:
    p, mw, ml = _clip_out(df)
    denom = mw + ml
    pbe = np.where(denom > 0, ml / denom, 0.5)
    return p - pbe


def _ev_w(df: pd.DataFrame, w0: float, l0: float) -> np.ndarray:
    p, _, _ = _clip_out(df)
    return p * w0 - (1.0 - p) * l0


# --------------------------------------------------------------------------- #
# Learned models (plan §8-§17)                                                 #
# --------------------------------------------------------------------------- #
class Poly2Ridge:
    """M1_POLY2_RIDGE -- quadratic statistical interaction (plan §8, §14)."""

    def __init__(self):
        self.poly = PolynomialFeatures(degree=2, include_bias=False)

    def fit(self, X, y, w):
        self.scaler = WeightedScaler().fit(X, w)
        xs = self.scaler.transform(X)
        xp = self.poly.fit_transform(xs)
        self.poly_scaler = WeightedScaler().fit(xp, w)
        xp = self.poly_scaler.transform(xp)
        self.model = Ridge(alpha=1.0)
        self.model.fit(xp, y, sample_weight=w)
        return self

    def predict(self, X):
        xs = self.scaler.transform(X)
        xp = self.poly.transform(xs)
        xp = self.poly_scaler.transform(xp)
        return self.model.predict(xp)


class SplineRidge:
    """M2_SPLINE_RIDGE -- additive cubic spline per output (plan §9, §15)."""

    def __init__(self):
        self.spline = SplineTransformer(n_knots=4, degree=3, include_bias=False)

    def fit(self, X, y, w):
        self.scaler = WeightedScaler().fit(X, w)
        xs = self.scaler.transform(X)
        xb = self.spline.fit_transform(xs)
        self.basis_scaler = WeightedScaler().fit(xb, w)
        xb = self.basis_scaler.transform(xb)
        self.model = Ridge(alpha=1.0)
        self.model.fit(xb, y, sample_weight=w)
        return self

    def predict(self, X):
        xs = self.scaler.transform(X)
        xb = self.spline.transform(xs)
        xb = self.basis_scaler.transform(xb)
        return self.model.predict(xb)


class PGM3:
    """M3_PGM3 -- 3-state Gaussian latent regime, outcome weighted (plan §10, §16).

    NOTE: the GMM *state clustering* is unweighted (sklearn limitation);
    only the per-regime expected-return estimate is sample-weighted.  This
    limitation is recorded in the manifest.
    """

    def fit(self, X, y, w):
        self.scaler = WeightedScaler().fit(X, w)
        xs = self.scaler.transform(X)
        self.gmm = GaussianMixture(
            n_components=3, covariance_type="full",
            reg_covar=1e-4, n_init=10, random_state=20260925,
        )
        self.gmm.fit(xs)
        g = self.gmm.predict_proba(xs)
        self.theta_ = np.empty(3)
        for k in range(3):
            wk = w * g[:, k]
            self.theta_[k] = np.sum(wk * y) / np.sum(wk)
        self._order = np.argsort(self.theta_)  # REGIME_LOW/MID/HIGH
        return self

    def predict(self, X):
        xs = self.scaler.transform(X)
        g = self.gmm.predict_proba(xs)
        return g @ self.theta_

    def regime_prob(self, X):
        xs = self.scaler.transform(X)
        g = self.gmm.predict_proba(xs)
        return g[:, self._order]  # columns now REGIME_LOW/MID/HIGH


class GmmLinearMoE3:
    """M4_GMM_MOE3 -- probabilistic 3-state gate + 3 linear experts (plan §11, §17)."""

    def fit(self, X, y, w):
        self.scaler = WeightedScaler().fit(X, w)
        xs = self.scaler.transform(X)
        self.gmm = GaussianMixture(
            n_components=3, covariance_type="full",
            reg_covar=1e-4, n_init=10, random_state=20260925,
        )
        self.gmm.fit(xs)
        g = self.gmm.predict_proba(xs)
        self.experts = []
        self._expert_theta = np.empty(3)
        for k in range(3):
            wk = w * g[:, k]
            model = Ridge(alpha=1.0)
            model.fit(xs, y, sample_weight=wk)
            self.experts.append(model)
            self._expert_theta[k] = np.sum(wk * model.predict(xs)) / np.sum(wk)
        self._order = np.argsort(self._expert_theta)
        return self

    def predict(self, X):
        xs = self.scaler.transform(X)
        g = self.gmm.predict_proba(xs)
        expert_pred = np.column_stack([m.predict(xs) for m in self.experts])
        return np.sum(g * expert_pred, axis=1)

    def regime_prob(self, X):
        xs = self.scaler.transform(X)
        g = self.gmm.predict_proba(xs)
        return g[:, self._order]


# --------------------------------------------------------------------------- #
# Pair purity (plan §5)                                                        #
# --------------------------------------------------------------------------- #
def _epoch_sides(frame: pd.DataFrame) -> Dict[Tuple, set]:
    out: Dict[Tuple, set] = {}
    for key, grp in frame.groupby(["symbol", "decision_bar", "horizon"]):
        out[key] = set(grp["side"].tolist())
    return out


def enforce_pair_purity(frame: pd.DataFrame, full_sides: Dict[Tuple, set],
                        cutoff) -> pd.DataFrame:
    """Drop a whole epoch if a 2-sided epoch lacks all sides available < cutoff."""
    keep = []
    for key, grp in frame.groupby(["symbol", "decision_bar", "horizon"]):
        sides_present = set(grp["side"].tolist())
        full = full_sides.get(key, set())
        avail = (pd.to_datetime(grp["label_available_time"]) < cutoff).all()
        if len(full) >= 2:
            if sides_present == full and avail:
                keep.append(grp)
        else:
            if avail:
                keep.append(grp)
    if not keep:
        return frame.iloc[0:0]
    return pd.concat(keep, ignore_index=True)


# --------------------------------------------------------------------------- #
# Cross-fit engine (plan §5, §18, §19)                                         #
# --------------------------------------------------------------------------- #
def meta_splits(frame: pd.DataFrame):
    """Yield (target_fold, hist, target, cutoff) for the 4 evaluation folds.

    History respects all causal guards (plan §5): folds < k, decision_time <
    cutoff, label_available_time < cutoff, and epoch pair purity.  Exposed
    for direct unit testing of the second-level cross-fit.
    """
    full_sides = _epoch_sides(frame)
    out = []
    for target_fold in range(1, N_FOLDS):  # folds 1..4; fold0 = warmup only
        target = frame[frame["fold"] == target_fold].copy()
        cutoff = pd.to_datetime(target["decision_time"]).min()
        hist = frame[frame["fold"] < target_fold].copy()
        hist = hist[pd.to_datetime(hist["decision_time"]) < cutoff]
        hist = hist[pd.to_datetime(hist["label_available_time"]) < cutoff]
        hist = enforce_pair_purity(hist, full_sides, cutoff)
        out.append((target_fold, hist, target, cutoff))
    return out


def run_one_model(frame: pd.DataFrame, meta_model: str):
    """Run one meta model across the 4 evaluation folds.

    Returns (pred_part, regime_probs_or_None).  pred_part contains NO outcome.
    """
    parts: List[pd.DataFrame] = []
    regime_probs: Optional[np.ndarray] = None

    for target_fold, hist, target, _cutoff in meta_splits(frame):
        y_hist = hist["episode_return_atr"].to_numpy(float)
        w_hist = hist["sample_weight"].to_numpy(float)

        if meta_model == "B0_P":
            score_hist = hist["p_win"].to_numpy(float)
            score_test = target["p_win"].to_numpy(float)
        elif meta_model == "B1_EV_C":
            score_hist = _ev_c(hist)
            score_test = _ev_c(target)
        elif meta_model == "B2_MARGIN":
            score_hist = _margin(hist)
            score_test = _margin(target)
        elif meta_model == "B3_EV_W":
            w0 = wmean(hist["mu_win"].to_numpy(float), w_hist)
            l0 = wmean(hist["mu_loss"].to_numpy(float), w_hist)
            score_hist = _ev_w(hist, w0, l0)
            score_test = _ev_w(target, w0, l0)
        elif meta_model == "M1_POLY2_RIDGE":
            m = Poly2Ridge().fit(meta_state(hist), y_hist, w_hist)
            score_hist = m.predict(meta_state(hist))
            score_test = m.predict(meta_state(target))
        elif meta_model == "M2_SPLINE_RIDGE":
            m = SplineRidge().fit(meta_state(hist), y_hist, w_hist)
            score_hist = m.predict(meta_state(hist))
            score_test = m.predict(meta_state(target))
        elif meta_model == "M3_PGM3":
            m = PGM3().fit(meta_state(hist), y_hist, w_hist)
            score_hist = m.predict(meta_state(hist))
            score_test = m.predict(meta_state(target))
            rp = m.regime_prob(meta_state(target))
        elif meta_model == "M4_GMM_MOE3":
            m = GmmLinearMoE3().fit(meta_state(hist), y_hist, w_hist)
            score_hist = m.predict(meta_state(hist))
            score_test = m.predict(meta_state(target))
            rp = m.regime_prob(meta_state(target))
        else:
            raise RuntimeError(f"STOP_R13_7_UNKNOWN_MODEL {meta_model}")

        thresholds = {
            "q90": float(np.quantile(score_hist, 0.90)),
            "q80": float(np.quantile(score_hist, 0.80)),
            "q70": float(np.quantile(score_hist, 0.70)),
        }

        out = target[["symbol", "decision_bar", "decision_time",
                      "trading_day", "side", "fold", "sample_weight"]].copy()
        out["score"] = score_test
        out["threshold10"] = thresholds["q90"]
        out["threshold20"] = thresholds["q80"]
        out["threshold30"] = thresholds["q70"]
        out["select10"] = score_test >= thresholds["q90"]
        out["select20"] = score_test >= thresholds["q80"]
        out["select30"] = score_test >= thresholds["q70"]
        parts.append(out)

        if meta_model in PGM_MODELS:
            regime_probs = rp if regime_probs is None else np.concatenate(
                [regime_probs, rp], axis=0)

    pred = pd.concat(parts, ignore_index=True)
    return pred, regime_probs


# --------------------------------------------------------------------------- #
# Base frame builder (ROOT candidate population, plan §6)                      #
# --------------------------------------------------------------------------- #
def build_base_frame(arch: str) -> pd.DataFrame:
    oof = load_td5_oof(arch)  # verified TRAIN-OOF, 5 folds (plan §33-#24)
    lab = R.read_train_labels(columns=KEY + ["label_available_time", "win"])
    lab = lab[lab["horizon"] == HORIZON]
    base = oof.merge(
        lab[KEY + ["label_available_time", "win"]],
        on=KEY, how="left", validate="one_to_one",
    )
    if base["label_available_time"].isna().any():
        raise RuntimeError("STOP_R13_7_LABEL_AVAIL_JOIN_MISSING")
    st = R.read_state(columns=["symbol", "bar_index", "trading_day",
                               "candidate_at_decision"]).rename(
        columns={"bar_index": "decision_bar"})
    base = base.merge(st, on=["symbol", "decision_bar"], how="left",
                      validate="many_to_one")
    if base["candidate_at_decision"].isna().any():
        raise RuntimeError("STOP_R13_7_STATE_JOIN_MISSING")
    return base


def root_frame(arch: str) -> pd.DataFrame:
    base = build_base_frame(arch)
    return base[base["candidate_at_decision"] == True].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Block bootstrap (plan §25; mirrors R13.5 complete-block logic)               #
# --------------------------------------------------------------------------- #
def _block_index_map(trading_days: np.ndarray, block: int):
    days = np.sort(pd.unique(trading_days))
    n_complete = len(days) // block
    days = days[: n_complete * block]
    pos = {d: np.where(trading_days == d)[0] for d in days}
    blocks = [days[i * block:(i + 1) * block] for i in range(n_complete)]
    return [np.concatenate([pos[d] for d in b]) for b in blocks]


def _stat_r20(y, w, score, sel, idx):
    m = sel[idx]
    yy = y[idx][m]
    ww = w[idx][m]
    if ww.sum() <= 0:
        return np.nan
    return float(np.average(yy, weights=ww))


def _stat_spread(y, w, score, sel, idx):
    yy = y[idx].astype(float)
    ww = w[idx].astype(float)
    ss = score[idx].astype(float)
    mm = np.isfinite(ss)
    yy, ww, ss = yy[mm], ww[mm], ss[mm]
    if len(ss) < 20:
        return np.nan
    hi, lo = np.quantile(ss, 0.8), np.quantile(ss, 0.2)
    top, bot = ss >= hi, ss <= lo
    if not (top.any() and bot.any()):
        return np.nan
    return float(np.average(yy[top], weights=ww[top]) -
                 np.average(yy[bot], weights=ww[bot]))


def _bootstrap(arr, stat_fn, block_idx, b: int, seed: int) -> np.ndarray:
    y, w, score, sel = arr[0], arr[1], arr[2], arr[3]
    n_blocks = len(block_idx)
    rng = np.random.default_rng(seed)
    boots = np.empty(b)
    for i in range(b):
        chosen = rng.integers(0, n_blocks, size=n_blocks)
        idx = np.concatenate([block_idx[c] for c in chosen])
        boots[i] = stat_fn(y, w, score, sel, idx)
    return boots[np.isfinite(boots)]


def _boot_paired(arr_m, arr_p, block_idx, b: int, seed: int) -> np.ndarray:
    ym, wm, sm, sem = arr_m[0], arr_m[1], arr_m[2], arr_m[3]
    yp, wp, sp, sep = arr_p[0], arr_p[1], arr_p[2], arr_p[3]
    n_blocks = len(block_idx)
    rng = np.random.default_rng(seed)
    diffs = np.empty(b)
    for i in range(b):
        chosen = rng.integers(0, n_blocks, size=n_blocks)
        idx = np.concatenate([block_idx[c] for c in chosen])
        rm = _stat_r20(ym, wm, sm, sem, idx)
        rp = _stat_r20(yp, wp, sp, sep, idx)
        diffs[i] = rm - rp if np.isfinite(rm) and np.isfinite(rp) else np.nan
    return diffs[np.isfinite(diffs)]


def _ci(boots: np.ndarray):
    if boots.size == 0:
        return (np.nan, np.nan, np.nan)
    return (float(np.nanmean(boots)),
            float(np.nanpercentile(boots, 2.5)),
            float(np.nanpercentile(boots, 97.5)))


# --------------------------------------------------------------------------- #
# Evaluation (plan §21-§28, §31-§32)                                           #
# --------------------------------------------------------------------------- #
def evaluate(pred: pd.DataFrame, build_frames: Dict[str, pd.DataFrame],
             regime_probs: Dict[Tuple[str, str], np.ndarray],
             artifact_sha: str) -> None:
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    model_rows = []
    fold_rows = []
    symside_rows = []
    regime_rows = []

    for arch in [ARCH_PRIMARY, ARCH_SECONDARY]:
        ap = pred[pred["architecture"] == arch].copy()
        bf = build_frames[arch]
        evf = pd.concat([bf[bf["fold"] == k] for k in range(1, N_FOLDS)],
                        ignore_index=True)
        ap = ap.merge(
            evf[["symbol", "decision_bar", "side", "fold", "sample_weight",
                 "episode_return_atr", "win", "p_win", "mu_win", "mu_loss"]],
            on=["symbol", "decision_bar", "side", "fold"],
            how="left", validate="many_to_one",
        )
        if ap["episode_return_atr"].isna().any():
            raise RuntimeError("STOP_R13_7_OUTCOME_JOIN_MISSING")

        # All meta models share the identical target row universe/order, so the
        # block index built from any one model applies to all (plan #18).
        ref = ap[ap["meta_model"] == "B0_P"]
        block_idx = _block_index_map(ref["trading_day"].to_numpy(), BOOTSTRAP_BLOCK)

        arrays: Dict[str, Tuple] = {}
        for model in META_MODELS:
            mp = ap[ap["meta_model"] == model]
            arrays[model] = (
                mp["episode_return_atr"].to_numpy(float),
                mp["sample_weight"].to_numpy(float),
                mp["score"].to_numpy(float),
                mp["select20"].to_numpy(bool),
                mp["fold"].to_numpy(),
            )

        p_arr = arrays["B0_P"]

        for model in META_MODELS:
            arr = arrays[model]
            y, w, score, sel, fold = arr
            boots_r20 = _bootstrap(arr, _stat_r20, block_idx, BOOTSTRAP_B,
                                   BOOTSTRAP_SEED)
            boots_spread = _bootstrap(arr, _stat_spread, block_idx, BOOTSTRAP_B,
                                      BOOTSTRAP_SEED)
            pt_r20, lo_r20, hi_r20 = _ci(boots_r20)
            pt_sp, lo_sp, hi_sp = _ci(boots_spread)

            cov = float(np.sum(w[sel]) / np.sum(w)) if np.sum(w) > 0 else np.nan

            if model == "B0_P":
                delta_pt, delta_lo, delta_hi = (0.0, 0.0, 0.0)
                status = "REFERENCE"
            else:
                diffs = _boot_paired(arr, p_arr, block_idx, BOOTSTRAP_B,
                                     BOOTSTRAP_SEED)
                delta_pt, delta_lo, delta_hi = _ci(diffs)
                if delta_lo > 0:
                    status = "META_SUPPORTED_OVER_P"
                elif delta_pt > 0:
                    status = "META_PROMISING_OVER_P"
                else:
                    status = "META_NO_IMPROVEMENT_OVER_P"

            # positive TRADE20 folds / 4
            pos_folds = 0
            for k in range(1, N_FOLDS):
                fmask = fold == k
                fsel = sel[fmask]
                fy = y[fmask][fsel]
                fw = w[fmask][fsel]
                if fw.sum() > 0 and np.average(fy, weights=fw) > 0:
                    pos_folds += 1

            model_rows.append({
                "architecture": arch,
                "MODEL": MODEL_LABEL[model],
                "TRADE20_mean_return": round(pt_r20, 6),
                "TRADE20_ci_lo": round(lo_r20, 6),
                "TRADE20_ci_hi": round(hi_r20, 6),
                "coverage": round(cov, 6),
                "delta_TRADE20_vs_p": round(delta_pt, 6),
                "paired_ci_lo": round(delta_lo, 6),
                "paired_ci_hi": round(delta_hi, 6),
                "status": status,
                "Top20_Bottom20": round(pt_sp, 6),
                "Top20_Bottom20_ci_lo": round(lo_sp, 6),
                "Top20_Bottom20_ci_hi": round(hi_sp, 6),
                "positive_folds_4": pos_folds,
            })

            # fold stability (plan §26)
            for k in range(1, N_FOLDS):
                fmask = fold == k
                fsel = sel[fmask]
                fy = y[fmask]
                fw = w[fmask]
                sel_y = fy[fsel]
                sel_w = fw[fsel]
                fm_ret = (float(np.average(sel_y, weights=sel_w))
                          if sel_w.sum() > 0 else np.nan)
                fm_cov = (float(np.sum(fw[fsel]) / np.sum(fw))
                          if np.sum(fw) > 0 else np.nan)
                ss = score[fmask]
                fmm = np.isfinite(ss)
                ss, yy, ww = ss[fmm], fy[fmm], fw[fmm]
                if len(ss) >= 20:
                    hi, lo = np.quantile(ss, 0.8), np.quantile(ss, 0.2)
                    top, bot = ss >= hi, ss <= lo
                    fm_spread = (float(np.average(yy[top], weights=ww[top]) -
                                      np.average(yy[bot], weights=ww[bot]))
                                 if (top.any() and bot.any()) else np.nan)
                else:
                    fm_spread = np.nan
                fold_rows.append({
                    "architecture": arch, "MODEL": MODEL_LABEL[model],
                    "eval_fold": k, "TRADE20_coverage": round(fm_cov, 6),
                    "TRADE20_mean_return": round(fm_ret, 6),
                    "Top20_Bottom20": round(fm_spread, 6) if fm_spread == fm_spread else np.nan,
                })

            # symbol / side stability (plan §27)
            for (sym, side), gp in ap[ap["meta_model"] == model].groupby(
                    ["symbol", "side"]):
                gsel = gp["select20"].to_numpy(bool)
                gn = len(gp)
                gseln = int(gsel.sum())
                gcov = (float(np.sum(gp["sample_weight"].to_numpy(float)[gsel]) /
                              np.sum(gp["sample_weight"].to_numpy(float)))
                        if np.sum(gp["sample_weight"]) > 0 else np.nan)
                gy = gp["episode_return_atr"].to_numpy(float)[gsel]
                gw = gp["sample_weight"].to_numpy(float)[gsel]
                gret = (float(np.average(gy, weights=gw))
                        if gw.sum() > 0 else np.nan)
                symside_rows.append({
                    "architecture": arch, "MODEL": MODEL_LABEL[model],
                    "symbol": sym, "side": side, "n": gn,
                    "selected_n": gseln, "coverage": round(gcov, 6),
                    "selected_mean_return": round(gret, 6),
                })

        # regime interpretation (plan §28)
        for model in PGM_MODELS:
            rp = regime_probs.get((arch, model))
            if rp is None:
                continue
            mp = ap[ap["meta_model"] == model].reset_index(drop=True)
            if len(rp) != len(mp):
                raise RuntimeError("STOP_R13_7_REGIME_ALIGNMENT")
            p_win = mp["p_win"].to_numpy(float)
            mu_win = mp["mu_win"].to_numpy(float)
            mu_loss = mp["mu_loss"].to_numpy(float)
            denom = np.maximum(mu_win, 0.0) + np.maximum(mu_loss, 0.0)
            p_be = np.where(denom > 0, np.maximum(mu_loss, 0.0) / denom, 0.5)
            y = mp["episode_return_atr"].to_numpy(float)
            win = mp["win"].to_numpy(float)
            w = mp["sample_weight"].to_numpy(float)
            fold = mp["fold"].to_numpy()
            labels = ["REGIME_LOW", "REGIME_MID", "REGIME_HIGH"]
            for k in range(1, N_FOLDS):
                fmask = fold == k
                for r in range(3):
                    g = rp[fmask, r]
                    gw = g * w[fmask]
                    s = gw.sum()
                    if s <= 0:
                        continue
                    regime_rows.append({
                        "architecture": arch, "MODEL": MODEL_LABEL[model],
                        "eval_fold": k, "regime": labels[r],
                        "weight_share": round(float(s / w[fmask].sum()), 6),
                        "mean_p_win": round(float(np.sum(gw * p_win[fmask]) / s), 6),
                        "mean_mu_win": round(float(np.sum(gw * mu_win[fmask]) / s), 6),
                        "mean_mu_loss": round(float(np.sum(gw * mu_loss[fmask]) / s), 6),
                        "mean_p_break_even": round(float(np.sum(gw * p_be[fmask]) / s), 6),
                        "mean_actual_return": round(float(np.sum(gw * y[fmask]) / s), 6),
                        "actual_win_rate": round(float(np.sum(gw * win[fmask]) / s), 6),
                    })

    R.write_csv_evidence(pd.DataFrame(model_rows), MODELS_CSV)
    R.write_csv_evidence(pd.DataFrame(fold_rows), FOLD_CSV)
    R.write_csv_evidence(pd.DataFrame(symside_rows), SYMSIDE_CSV)
    R.write_csv_evidence(pd.DataFrame(regime_rows), REGIMES_CSV)

    summary = {
        "experiment": "FUTURE-R13.6-R13.7",
        "reviewed_parent_sha": REVIEWED_BASE_SHA,
        "population": POP_ROOT,
        "horizon": HORIZON,
        "meta_models": META_MODELS,
        "prediction_artifact_sha256": artifact_sha,
        "governance": {
            "dev_val_reads": 0,
            "old_test_label_reads": 0,
            "old_test_policy_reads": 0,
            "base_model_fits": 0,
            "direction_model_fits": 0,
            "hyperparameter_searches": 0,
        },
        "no_threshold_mining": True,
        "no_hyperparameter_search": True,
        "no_dev_val": True,
        "no_test": True,
        "stop": "AWAITING_R13_7_META_INTEGRATION_REVIEW",
    }
    R.write_json_evidence(summary, SUMMARY_JSON)
    _write_manifest(artifact_sha, summary)
    return summary


def _write_manifest(artifact_sha: str, summary: dict) -> None:
    manifest = dict(summary)
    manifest["generator_code_sha"] = os.environ.get(
        "R13_7_CODE_SHA", "PENDING_COMMIT")
    manifest["prediction_artifact"] = {
        "path": OOF_PARQUET, "sha256": artifact_sha,
    }
    manifest["frozen_inputs"] = [
        "WIN33", "PAY8", "SHARED41", "SPACE18", "PATH8", "VOL6",
        "A0_V1_DISJOINT", "A1_SHARE_TO_WIN", "TD5 labels", "Phase-4 result",
    ]
    manifest["prohibited"] = [
        "E9 fit/read", "DEV VAL read", "old TEST read", "base-model refit",
        "symbol/side/time/regime features", "hyperparameter search",
        "threshold mining",
    ]
    manifest["bootstrap"] = {
        "B": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED, "block": BOOTSTRAP_BLOCK,
        "complete_blocks_only": True,
    }
    manifest["selection_thresholds"] = {
        "q10": 0.90, "q20": 0.80, "q30": 0.70,
        "source": "meta_train_history_only",
    }
    R.write_json_evidence(manifest, MANIFEST_JSON)


# --------------------------------------------------------------------------- #
# Orchestration (plan §34)                                                     #
# --------------------------------------------------------------------------- #
def build_predictions() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame],
                                 Dict[Tuple[str, str], np.ndarray], str]:
    """Build (and freeze) the outcome-free prediction artifact for both archs."""
    all_parts: List[pd.DataFrame] = []
    build_frames: Dict[str, pd.DataFrame] = {}
    regime_probs: Dict[Tuple[str, str], np.ndarray] = {}

    for arch in [ARCH_PRIMARY, ARCH_SECONDARY]:
        frame = root_frame(arch)
        build_frames[arch] = frame
        for model in META_MODELS:
            pred_part, rp = run_one_model(frame, model)
            pred_part["architecture"] = arch
            pred_part["meta_model"] = model
            all_parts.append(pred_part)
            if rp is not None:
                regime_probs[(arch, model)] = rp

    pred = pd.concat(all_parts, ignore_index=True)
    cols = ["architecture", "meta_model", "symbol", "decision_bar",
            "decision_time", "trading_day", "side", "fold", "score",
            "threshold10", "threshold20", "threshold30",
            "select10", "select20", "select30"]
    pred = pred[cols]
    R.write_parquet(pred, OOF_PARQUET)
    artifact_sha = sha256_of(OOF_PARQUET)
    return pred, build_frames, regime_probs, artifact_sha


def main() -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    pred, build_frames, regime_probs, artifact_sha = build_predictions()
    print(f"[R13.7] prediction artifact frozen: {OOF_PARQUET}")
    print(f"[R13.7] artifact sha256: {artifact_sha}")

    # A0 benchmark must be frozen before A1 replication (plan §29, test #25).
    a0 = pred[pred["architecture"] == ARCH_PRIMARY]
    if a0["score"].isna().any():
        raise RuntimeError("STOP_R13_7_A0_SCORE_MISSING")

    summary = evaluate(pred, build_frames, regime_probs, artifact_sha)
    print("[R13.7] evidence written; stop =", summary["stop"])


if __name__ == "__main__":
    main()
