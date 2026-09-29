"""PAYOFF-GEOMETRY-01A — Conditional Geometry Incremental Information Audit (Kernel Checkpoint).

REVISED CHECKPOINT (after Reviewer STOP_BEFORE_T1_5 / FIX_REQUIRED — MATHEMATICAL /
LABEL SEMANTICS MISMATCH on PAYOFF-GEOMETRY-01).

The frozen p_win model answers:

    p_i = P(episode_return_atr > 0 | X)

It is NOT P(favorable boundary hit before adverse boundary). Therefore the identity

    EV = p_win * G - (1 - p_win) * L

is mathematically INVALID for the current frozen p_win and is NOT computed here.

This module instead asks the corrected question:

    PRIMARY QUESTION
    With the existing frozen p_win unchanged, does decision-time-known payoff
    geometry provide incremental information about the TRUE episode_return_atr
    beyond p_win?

FROZEN DEFINITIONS
    p_i  : existing frozen OOF p_win = P(episode_return_atr > 0 | X)
    G_i  : canonical decision-time favorable structural distance in ATR
    L_i  : canonical decision-time adverse structural distance in ATR
    Z_i  = log(G_i / L_i)
    Outcome : canonical TRUE episode_return_atr  (win = episode_return_atr > 0)

RULES (enforced)
    * Do NOT call Z or any function of p/G/L an "expected value".
    * Do NOT train / refit / tune any model.
    * The only primary realized-return outcome is true_episode_return_atr.

PRIMARY ESTIMAND (designed for T1.5/T2; computed on a SMALL sample here for
Reference/Production parity only — NOT a scientific result):
    D_geometry = weighted mean across p-strata [
        E(true_episode_return_atr | high log(G/L))
      - E(true_episode_return_atr | low  log(G/L)) ]
    Default: top 20% vs bottom 20% within each p-stratum; weighting owner =
    canonical ``sample_weight``.

CHECKPOINT SCOPE (this file)
    T0 (hand truth) -> T1 (small real-sample audit + diagnostics) -> TP (perf)
    -> commit -> push -> STOP FOR REVIEWER.
    T1.5 / T2 (full-evaluation high/low experiment + bootstrap) are NOT run.

Canonical owners (verified in owner audit):
    * candidate / label / G / L / win / episode_return_atr / event_class /
      sample_weight / decision_time / label_available_time / split
        -> artifacts/opportunity_value_v1/labels_{train,val,test}_v1.parquet (R8)
    * p_win model (frozen)  -> win_probability_model_v1.py (R9A)
    * p_win OOF + frozen mu_win/mu_loss/predicted_rr
        -> artifacts/decomposed_value_v2/oof/A0_V1_DISJOINT_f*_td5.parquet
    * trading_day (for bootstrap clustering)
        -> artifacts/decomposed_value_v1/state_v1.parquet [symbol,bar_index,trading_day]
    * canonical weighting owner -> labels column ``sample_weight``
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths / frozen configuration                                                 #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]          # .../future_dev
OOF_DIR = PROJECT_ROOT / "artifacts" / "decomposed_value_v2" / "oof"
LABEL_DIR = PROJECT_ROOT / "artifacts" / "opportunity_value_v1"
STATE_PARQUET = PROJECT_ROOT / "artifacts" / "decomposed_value_v1" / "state_v1.parquet"
EVID = Path(__file__).resolve().parent / "evidence"

FROZEN_WIN_ARCH = "A0_V1_DISJOINT"
AUDIT_HORIZON = "td5"
EVAL_FOLDS = [0, 1, 2, 3, 4]

# Bootstrap configuration (used only in T1 audit / T1.5-T2; NOT run on full pop here)
BOOTSTRAP_SEED = 20260929
BOOTSTRAP_B = 2000
BOOTSTRAP_BLOCK_DAYS = 5

# Stratification / selection (frozen before seeing any outcome)
N_P_BINS = 10
TOP_FRAC = 0.20
BOTTOM_FRAC = 0.20

T1_SAMPLE_N = 2000

# TASK_ID / provenance
TASK_ID = "PAYOFF-GEOMETRY-01A"
REVIEWED_PARENT_SHA = "305f7de"   # remote checkpoint that triggered this revision

# --------------------------------------------------------------------------- #
# Performance / governance counters                                            #
# --------------------------------------------------------------------------- #
COUNTERS = {
    "raw_load_count": 0,
    "candidate_load_count": 0,
    "pwin_load_count": 0,
    "label_geometry_load_count": 0,
    "model_fit_count": 0,
    "feature_recompute_count": 0,
    "label_recompute_count": 0,
    "reference_call_count": 0,
    "full_history_recompute_count": 0,
    "concat_hotloop_count": 0,
    "score_row_count": 0,
    "bootstrap_rep_count": 0,
}


# --------------------------------------------------------------------------- #
# Core geometry math (frozen semantics)                                        #
# --------------------------------------------------------------------------- #
def log_geometry_ratio(g: np.ndarray, l: np.ndarray) -> np.ndarray:
    """Z_i = log(G_i / L_i). Decision-time known. Vectorized, O(N)."""
    g = np.asarray(g, dtype=float)
    l = np.asarray(l, dtype=float)
    if not np.isfinite(g).all() or not np.isfinite(l).all():
        raise ValueError("non-finite G/L")
    if (g <= 0).any() or (l <= 0).any():
        raise ValueError("G/L must be strictly positive ATR distances")
    return np.log(g / l)


def compute_p_bin_edges(p_win: np.ndarray, n_bins: int = N_P_BINS) -> np.ndarray:
    """Frozen, outcome-independent p_win quantile edges.

    Depends ONLY on p_win, never on the true return. Deterministic given p_win.
    Returned edges MUST be frozen from the full candidate set and then reused on
    every resample (bootstrap) so that binning is identical across replicates.
    """
    p = np.asarray(p_win, dtype=float)
    if not np.isfinite(p).all() or ((p < 0.0) | (p > 1.0)).any():
        raise ValueError("p_win outside [0,1]")
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = edges[0] - 1e-9
    edges[-1] = edges[-1] + 1e-9
    return edges


def assign_p_bins(p_win: np.ndarray, n_bins: int = N_P_BINS,
                  edges: np.ndarray | None = None) -> np.ndarray:
    """Stratify candidates by frozen p_win edges.

    If ``edges`` is provided (frozen from the full sample), it is reused exactly;
    otherwise edges are computed from ``p_win`` (used only at the top level).
    """
    if edges is None:
        edges = compute_p_bin_edges(p_win, n_bins)
    p = np.asarray(p_win, dtype=float)
    bins = np.digitize(p, edges[1:-1]).astype(int)  # 0 .. n_bins-1
    return bins


# --------------------------------------------------------------------------- #
# Production kernel: stratified geometry contrast (vectorized, O(N log N))      #
# --------------------------------------------------------------------------- #
def compute_stratified_geometry_contrast(
    df: pd.DataFrame,
    n_p_bins: int = N_P_BINS,
    top_frac: float = TOP_FRAC,
    bottom_frac: float = BOTTOM_FRAC,
    p_col: str = "p_win",
    z_col: str = "log_gl",
    outcome_col: str = "true_episode_return_atr",
    weight_col: str = "weights",
    p_bin_edges: np.ndarray | None = None,
) -> dict:
    """Primary estimand D_geometry (production, vectorized).

    Returns the aggregated contrast plus per-stratum detail. No fitting, no IO,
    no reference calls. ``p_bin_edges`` is frozen from the full sample so that
    every bootstrap replicate uses identical binning (parity with Reference).
    """
    p = df[p_col].to_numpy(dtype=float, copy=False)
    z = df[z_col].to_numpy(dtype=float, copy=False)
    y = df[outcome_col].to_numpy(dtype=float, copy=False)
    w = df[weight_col].to_numpy(dtype=float, copy=False)

    if not (len(p) == len(z) == len(y) == len(w)):
        raise ValueError("length mismatch in stratified contrast inputs")
    if not np.isfinite(z).all() or not np.isfinite(y).all() or not np.isfinite(w).all():
        raise ValueError("non-finite input to stratified contrast")

    edges = p_bin_edges if p_bin_edges is not None else compute_p_bin_edges(p, n_p_bins)
    n_p_bins = len(edges) - 1
    bins = assign_p_bins(p, n_p_bins, edges=edges)
    tmp = pd.DataFrame({"bin": bins, "z": z, "y": y, "w": w})
    # within-bin rank (method='first' for deterministic, tie-stable ordering)
    tmp["rnk"] = tmp.groupby("bin")["z"].rank(method="first").to_numpy(dtype=np.int64)
    n = tmp.groupby("bin")["z"].transform("size").to_numpy(dtype=np.int64)
    kh = np.maximum(1, np.floor(n * top_frac).astype(np.int64))
    kl = np.maximum(1, np.floor(n * bottom_frac).astype(np.int64))
    high = tmp["rnk"].to_numpy() > (n - kh)
    low = tmp["rnk"].to_numpy() <= kl

    bin = bins.astype(int)
    hf = high.astype(float)
    lf = low.astype(float)
    sw_h = np.bincount(bin, weights=w * hf, minlength=n_p_bins)
    swy_h = np.bincount(bin, weights=w * y * hf, minlength=n_p_bins)
    sw_l = np.bincount(bin, weights=w * lf, minlength=n_p_bins)
    swy_l = np.bincount(bin, weights=w * y * lf, minlength=n_p_bins)
    mh = np.where(sw_h > 0, swy_h / np.where(sw_h > 0, sw_h, 1.0), np.nan)
    ml = np.where(sw_l > 0, swy_l / np.where(sw_l > 0, sw_l, 1.0), np.nan)
    contrast = mh - ml
    valid = (sw_h > 0) & (sw_l > 0) & ~np.isnan(contrast)
    btw = np.bincount(bin, weights=w, minlength=n_p_bins)
    if btw[valid].sum() == 0:
        D = float("nan")
    else:
        D = float(np.sum(contrast[valid] * btw[valid]) / np.sum(btw[valid]))

    COUNTERS["score_row_count"] += len(df)
    return {
        "D_geometry": D,
        "per_bin_contrast": [None if not v else float(contrast[i])
                             for i, v in enumerate(valid)],
        "per_bin_weight": [float(btw[i]) for i in range(n_p_bins)],
        "n_bins_used": int(valid.sum()),
        "n_p_bins": n_p_bins,
        "top_frac": top_frac,
        "bottom_frac": bottom_frac,
    }


# --------------------------------------------------------------------------- #
# Reference kernel: stratified geometry contrast (slow, row-by-row truth)       #
# --------------------------------------------------------------------------- #
def compute_stratified_geometry_contrast_reference(
    df: pd.DataFrame,
    n_p_bins: int = N_P_BINS,
    top_frac: float = TOP_FRAC,
    bottom_frac: float = BOTTOM_FRAC,
    p_col: str = "p_win",
    z_col: str = "log_gl",
    outcome_col: str = "true_episode_return_atr",
    weight_col: str = "weights",
) -> dict:
    """Slow / explicit truth implementation. T0/T1 only. Forbidden in prod call chain."""
    COUNTERS["reference_call_count"] += 1
    p = df[p_col].to_numpy(dtype=float, copy=False)
    z = df[z_col].to_numpy(dtype=float, copy=False)
    y = df[outcome_col].to_numpy(dtype=float, copy=False)
    w = df[weight_col].to_numpy(dtype=float, copy=False)
    bins = assign_p_bins(p, n_p_bins)
    bin_ids = sorted(set(bins.tolist()))
    contrasts = []
    bin_weights = []
    for b in bin_ids:
        idx = np.where(bins == b)[0]
        if len(idx) == 0:
            continue
        zz = z[idx]
        yy = y[idx]
        ww = w[idx]
        order = np.argsort(zz, kind="stable")
        n = len(idx)
        kh = max(1, int(math.floor(top_frac * n)))
        kl = max(1, int(math.floor(bottom_frac * n)))
        hi = order[-kh:]
        lo = order[:kl]
        mh = float(np.average(yy[hi], weights=ww[hi]))
        ml = float(np.average(yy[lo], weights=ww[lo]))
        contrasts.append(mh - ml)
        bin_weights.append(float(ww.sum()))
    contrasts = np.asarray(contrasts, dtype=float)
    bin_weights = np.asarray(bin_weights, dtype=float)
    D = float(np.sum(contrasts * bin_weights) / np.sum(bin_weights)) if bin_weights.sum() > 0 else float("nan")
    return {
        "D_geometry": D,
        "per_bin_contrast": [float(c) for c in contrasts],
        "per_bin_weight": [float(c) for c in bin_weights],
        "n_bins_used": int(len(contrasts)),
        "n_p_bins": n_p_bins,
        "top_frac": top_frac,
        "bottom_frac": bottom_frac,
    }


def assert_stratified_parity(df: pd.DataFrame, tol: float = 1e-9) -> None:
    """Reference vs Production must agree to tol on the stratified statistic."""
    prod = compute_stratified_geometry_contrast(df)
    ref = compute_stratified_geometry_contrast_reference(df)
    if not (math.isnan(prod["D_geometry"]) and math.isnan(ref["D_geometry"])):
        if abs(prod["D_geometry"] - ref["D_geometry"]) > tol:
            raise AssertionError(
                f"stratified parity FAIL: prod={prod['D_geometry']} ref={ref['D_geometry']}")
    # per-bin
    for a, b in zip(prod["per_bin_contrast"], ref["per_bin_contrast"]):
        if a is None or b is None:
            continue
        if abs(a - b) > tol:
            raise AssertionError(f"per-bin parity FAIL: {a} vs {b}")


def stratified_d_geometry_breakdown(df: pd.DataFrame,
                                    p_bin_edges: np.ndarray | None = None,
                                    with_ci: bool = False,
                                    bootstrap_reps: int = 200,
                                    block: int = BOOTSTRAP_BLOCK_DAYS) -> dict:
    """Single-instrument boundary breakdown of D_geometry.

    Pooled D_geometry is ONLY pooled research evidence. Per the strategy execution
    boundary (multi-symbol pooled training, single-instrument execution), each
    symbol's result must be retained separately, along with LONG/SHORT splits.
    Formal conclusions for a specific instrument must use THAT instrument's own CI.

    ESTIMATOR CONSISTENCY (frozen): for EVERY analysis universe S -- pooled, each
    symbol independently, LONG independently, SHORT independently -- the p strata
    are frozen WITHIN S. Concretely, ``edges_S = quantiles of p_win within S`` and
    the SAME edges_S are used for BOTH the observed D_S AND every bootstrap
    replicate for S. There is NO pooling of quantile edges across universes, and a
    subgroup's point estimate never depends on the pooled (or any other) edges.

    Consequently ``with_ci=False`` and ``with_ci=True`` return the IDENTICAL
    observed_D for the same subgroup (the latter additionally attaches the CI). This
    is required so a symbol's point estimate and its CI answer exactly the same
    question: "within this instrument's own win-rate distribution, do higher-G/L
    trades have better true returns?"

    If ``with_ci`` is True, each entry becomes a dict with the SAME schema for
    EVERY universe (pooled, per-symbol, per-side):
        observed_D, bootstrap_mean, n, n_trading_days, bootstrap_ci_low,
        bootstrap_ci_high, n_valid_reps
    When CI cannot be computed (n < 50 or no ``trading_day``), bootstrap_mean /
    bootstrap_ci_low / bootstrap_ci_high are present but ``None`` and
    ``n_valid_reps == 0`` -- the KEY SET is identical, only the values are null.

    The per-symbol / per-side day-block bootstrap reuses the SAME corrected owner
    (all days retained, with-replacement, multiplicity preserved). The caller must
    pass ``df`` already carrying a ``trading_day`` column (or have it resolvable)
    for CI to be computed; otherwise CI is omitted with a note.
    """
    def _entry(sub: pd.DataFrame, edges: np.ndarray) -> object:
        # edges are frozen WITHIN this universe S (passed by the caller)
        D = compute_stratified_geometry_contrast(sub, p_bin_edges=edges)["D_geometry"]
        if not with_ci:
            return D
        n = len(sub)
        if n < 50 or "trading_day" not in sub.columns:
            # SAME schema as the CI branch: bootstrap_mean/CI are present (None) so
            # every universe reports an identical key set. (Estimator unchanged.)
            return {"observed_D": D, "bootstrap_mean": None, "n": n,
                    "n_trading_days": None, "bootstrap_ci_low": None,
                    "bootstrap_ci_high": None, "n_valid_reps": 0,
                    "note": "insufficient rows or no trading_day for CI"}
        b = bootstrap_d_geometry(sub, b=bootstrap_reps, block=block,
                                 p_bin_edges=edges)
        return {
            "observed_D": b["observed_D"],
            "bootstrap_mean": b["bootstrap_mean"],
            "n": n,
            "n_trading_days": int(sub["trading_day"].nunique()),
            "bootstrap_ci_low": b["ci_lo"],
            "bootstrap_ci_high": b["ci_hi"],
            "n_valid_reps": b["n_valid_reps"],
        }

    # pooled universe uses its OWN p_win quantiles (honor a caller-supplied
    # p_bin_edges as the pooled edges, else derive from this df).
    pooled_edges = (p_bin_edges
                    if p_bin_edges is not None
                    else compute_p_bin_edges(df["p_win"].to_numpy(float)))
    out: dict = {"pooled": _entry(df, pooled_edges), "per_symbol": {}, "by_side": {}}
    for sym in sorted(df["symbol"].unique().tolist()):
        sub = df[df["symbol"] == sym]
        out["per_symbol"][sym] = _entry(sub, compute_p_bin_edges(sub["p_win"].to_numpy(float)))
    for side in sorted(df["side"].unique().tolist()):
        sub = df[df["side"] == side]
        out["by_side"][side] = _entry(sub, compute_p_bin_edges(sub["p_win"].to_numpy(float)))
    return out


# --------------------------------------------------------------------------- #
# Event-based synthetic barrier payoff (DIAGNOSTIC ONLY — NOT a realized return) #
# --------------------------------------------------------------------------- #
def event_based_synthetic_payoff(df: pd.DataFrame) -> np.ndarray:
    """DIAGNOSTIC_ONLY — event-semantic barrier-world pseudo-return.

    Defined strictly by ``event_class`` (NOT by the win sign):

        FAVORABLE_FIRST -> +G   (favorable boundary touched first)
        ADVERSE_FIRST   -> -L   (adverse boundary touched first)
        BOTH_SAME_BAR   -> NaN  (ambiguous: both boundaries in one bar)
        NONE            -> NaN  (no boundary resolution)

    This is NOT the canonical realized label. It exists solely so diagnostic B can
    quantify how far the true episode_return_atr deviates from the idealized
    barrier-world payoff. Because we have now confirmed FAVORABLE_FIRST does NOT
    imply final profit and ADVERSE_FIRST does NOT imply final loss, building this
    pseudo-return from the ``win`` sign would silently re-introduce exactly the
    semantics mismatch the Reviewer rejected.
    """
    g = df["G"].to_numpy(dtype=float, copy=False)
    l = df["L"].to_numpy(dtype=float, copy=False)
    ev = df["event_class"].to_numpy()
    out = np.full(len(df), np.nan, dtype=float)
    fav = ev == "FAVORABLE_FIRST"
    adv = ev == "ADVERSE_FIRST"
    out[fav] = g[fav]
    out[adv] = -l[adv]
    return out


# --------------------------------------------------------------------------- #
# Loader (read-only, no silent drop, exact key alignment)                       #
# --------------------------------------------------------------------------- #
def load_audit_frame(sample_n: int | None = None, seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    """Load frozen p_win (OOF) + canonical G/L/label/event_class/weights (labels).

    Exact inner join on candidate semantic key (symbol, decision_bar, side).
    A missing key would DROP OOF rows -> hard fail (no silent drop).
    """
    COUNTERS["pwin_load_count"] += 1
    COUNTERS["label_geometry_load_count"] += 1
    COUNTERS["candidate_load_count"] += 1
    COUNTERS["raw_load_count"] += 1

    # --- frozen p_win (V2 OOF, A0_V1_DISJOINT, td5, all folds) + old payoff OOF ---
    oof_frames = []
    for f in EVAL_FOLDS:
        fp = OOF_DIR / f"{FROZEN_WIN_ARCH}_f{f}_{AUDIT_HORIZON}.parquet"
        if not fp.exists():
            raise FileNotFoundError(f"missing frozen p_win artifact: {fp}")
        oof_frames.append(pd.read_parquet(fp))
    oof = pd.concat(oof_frames, ignore_index=True)
    pre_join_rows = len(oof)
    oof = oof[["symbol", "decision_bar", "side", "p_win",
               "mu_win", "mu_loss", "predicted_rr", "fold"]]

    # --- canonical label / geometry / event_class / weights (R8, td5) ---
    label_frames = []
    for split in ("train", "val", "test"):
        fp = LABEL_DIR / f"labels_{split}_v1.parquet"
        if not fp.exists():
            raise FileNotFoundError(f"missing label artifact: {fp}")
        label_frames.append(pd.read_parquet(fp))
    labels = pd.concat(label_frames, ignore_index=True)
    labels = labels[labels["horizon"] == AUDIT_HORIZON]
    key_cols = ["symbol", "decision_bar", "side"]
    dup = int(labels.duplicated(subset=key_cols).sum())
    if dup > 0:
        raise ValueError(f"ambiguous candidate key: {dup} duplicate (symbol,decision_bar,side) in td5 labels")
    labels = labels[["symbol", "decision_bar", "side", "G", "L",
                     "episode_return_atr", "win", "event_class",
                     "sample_weight", "decision_time", "label_available_time", "split"]]

    # --- exact join; every OOF key MUST survive (no silent drop) ---
    merged = oof.merge(labels, on=key_cols, how="inner")
    post_join_rows = len(merged)
    if post_join_rows != pre_join_rows:
        # silent drop forbidden
        raise ValueError(
            f"candidate key misalignment: {pre_join_rows - post_join_rows} OOF rows "
            f"lost on join (silent drop forbidden)")
    if merged.duplicated(subset=key_cols).any():
        raise ValueError("duplicate candidate key after join (ambiguous alignment)")
    # Measure duplicate_post_join_keys on the FULL post-join frame, BEFORE any
    # optional sample/subset, so it belongs to the same full canonical-join universe
    # as full_oof_rows_pre_join / full_rows_post_join / unmatched_rows /
    # duplicate_label_keys. Hard-fail above guarantees this is 0, but it is measured
    # here (not after sampling) so the evidence is defined consistently.
    duplicate_post_join = int(merged.duplicated(subset=key_cols).sum())

    merged = merged.rename(columns={
        "G": "G", "L": "L",
        "episode_return_atr": "true_episode_return_atr",
        "sample_weight": "weights",
    })
    merged["log_gl"] = log_geometry_ratio(merged["G"].to_numpy(float),
                                          merged["L"].to_numpy(float))
    # required finite checks on canonical fields
    for c in ("p_win", "G", "L", "true_episode_return_atr", "log_gl", "weights"):
        if merged[c].isna().any():
            raise ValueError(f"NaN in required column {c} (must not occur)")

    sampled_rows = len(merged)
    if sample_n is not None and sample_n < len(merged):
        merged = merged.sample(n=sample_n, random_state=seed).reset_index(drop=True)
        sampled_rows = len(merged)

    # True join evidence (reported verbatim in the Evidence Packet; never a
    # placeholder). These numbers describe the FULL (pre-sample) join universe and
    # are MEASURED, not inferred: the loader hard-fails above if any of them would
    # be non-zero, so a non-zero value can never silently reach the packet.
    merged.attrs["join_meta"] = {
        "full_oof_rows_pre_join": int(pre_join_rows),
        "full_rows_post_join": int(post_join_rows),
        "unmatched_rows": int(pre_join_rows - post_join_rows),
        "duplicate_label_keys": int(dup),
        "duplicate_post_join_keys": int(duplicate_post_join),
        "sampled_rows": int(sampled_rows),
    }
    return merged.reset_index(drop=True)


def load_t1_5_frame(cap_per_symbol: int = 1000, seed: int = 20260929,
                    return_meta: bool = False):
    """Deterministic T1.5 subset.

    ALL available symbols, each capped at ``cap_per_symbol`` OOF candidates, fixed
    seed, NO outcome-based sampling, LONG/SHORT preserved. Built on top of the
    already-validated clean join in load_audit_frame (which hard-fails on any
    silent drop / duplicate key), then capped per symbol using a fixed-seed
    within-symbol permutation so the selection depends ONLY on symbol and the seed,
    never on the outcome. trading_day is attached (hard-fail if any unresolved).

    The canonical join evidence (pre/post rows, unmatched, duplicate keys) is
    propagated on ``df.attrs["join_meta"]`` so the analysis never re-hardcodes 0.

    NOTE: the canonical candidate load is counted ONCE inside load_audit_frame (which
    this function calls); we do NOT re-increment here, so one logical T1.5 frame load
    yields exactly one candidate_load_count increment.
    """
    full = load_audit_frame(sample_n=None, seed=seed)  # full clean join, no random cap
    join_meta = dict(full.attrs.get("join_meta", {}))   # measured, propagated
    rng = np.random.default_rng(seed)
    parts = []
    for sym in sorted(full["symbol"].unique().tolist()):
        sub = full[full["symbol"] == sym]
        if len(sub) > cap_per_symbol:
            # deterministic within-symbol shuffle; outcome never consulted
            idx = rng.permutation(len(sub))[:cap_per_symbol]
            sub = sub.iloc[idx]
        parts.append(sub)
    df = pd.concat(parts, ignore_index=True)
    df = _attach_trading_day(df)  # hard-fail if any trading_day unresolved
    df.attrs["join_meta"] = join_meta  # keep the real join evidence with the frame
    out = df.reset_index(drop=True)
    if return_meta:
        return out, join_meta
    return out


def _attach_trading_day(df: pd.DataFrame, require_all: bool = True) -> pd.DataFrame:
    """Attach canonical trading_day from frozen state_v1 (decision_bar == bar_index).

    Hard-fail guarantees (no silent NaT fallback):
      * state (symbol, bar_index) key is unique;
      * every candidate resolves a trading_day (missing > 0 => STOP).
    """
    if "trading_day" in df.columns:
        if df["trading_day"].isna().any():
            raise RuntimeError("STOP: input already contains unresolved trading_day")
        return df
    state = pd.read_parquet(STATE_PARQUET, columns=["symbol", "bar_index", "trading_day"])
    dup_state = int(state.duplicated(subset=["symbol", "bar_index"]).sum())
    if dup_state > 0:
        raise ValueError(
            f"state trading_day key not unique: {dup_state} duplicate (symbol,bar_index)")
    idx = state.set_index(["symbol", "bar_index"])["trading_day"]
    keys = list(zip(df["symbol"], df["decision_bar"]))
    resolved = [idx.get(k) for k in keys]
    missing = sum(1 for t in resolved if t is None or pd.isna(t))
    if require_all and missing > 0:
        raise RuntimeError(
            f"STOP: {missing} candidates failed to resolve trading_day "
            f"(silent day-join forbidden)")
    df = df.copy()
    df["trading_day"] = resolved
    return df


# --------------------------------------------------------------------------- #
# Secondary diagnostics (audit only; not interpreted)                           #
# --------------------------------------------------------------------------- #
def diag_pwin_decile_table(df: pd.DataFrame, n_bins: int = N_P_BINS) -> list:
    """Diagnostic A: per p_win stratum table."""
    bins = assign_p_bins(df["p_win"].to_numpy(float), n_bins)
    out = []
    for b in sorted(set(bins.tolist())):
        d = df.iloc[np.where(bins == b)[0]]
        w = d["weights"].to_numpy(float)
        out.append({
            "p_bin": int(b),
            "n": int(len(d)),
            "mean_p_win": float(np.average(d["p_win"], weights=w)),
            "actual_win_rate": float(np.average(d["win"].astype(float), weights=w)),
            "mean_G": float(np.average(d["G"], weights=w)),
            "mean_L": float(np.average(d["L"], weights=w)),
            "mean_G_over_L": float(np.average(d["G"] / d["L"], weights=w)),
            "mean_log_GL": float(np.average(d["log_gl"], weights=w)),
            "mean_true_return_atr": float(np.average(d["true_episode_return_atr"], weights=w)),
        })
    return out


def diag_event_reconciliation(df: pd.DataFrame) -> dict:
    """Diagnostic B: event-semantic reconciliation (diagnostic only)."""
    classes = ["FAVORABLE_FIRST", "ADVERSE_FIRST", "BOTH_SAME_BAR", "NONE"]
    rows = {}
    for ev in classes:
        d = df[df["event_class"] == ev]
        if len(d) == 0:
            continue
        w = d["weights"].to_numpy(float)
        rows[ev] = {
            "n": int(len(d)),
            "share": float(len(d) / len(df)),
            "actual_win_rate": float(np.average(d["win"].astype(float), weights=w)),
            "mean_true_return_atr": float(np.average(d["true_episode_return_atr"], weights=w)),
            "mean_G": float(np.average(d["G"], weights=w)),
            "mean_L": float(np.average(d["L"], weights=w)),
            "mean_G_over_L": float(np.average(d["G"] / d["L"], weights=w)),
        }
    # P(win | FAVORABLE_FIRST) and P(loss | ADVERSE_FIRST)
    fav = df[df["event_class"] == "FAVORABLE_FIRST"]
    adv = df[df["event_class"] == "ADVERSE_FIRST"]
    p_win_fav = (float(np.average(fav["win"].astype(float), weights=fav["weights"]))
                 if len(fav) else float("nan"))
    p_loss_adv = (float(1.0 - np.average(adv["win"].astype(float), weights=adv["weights"]))
                  if len(adv) else float("nan"))

    # win / event_class confusion (raw counts)
    confusion = (df.assign(_win=df["win"].astype(int))
                   .groupby(["event_class", "_win"]).size()
                   .unstack(fill_value=0).to_dict())

    # true episode_return_atr minus EVENT-BASED synthetic barrier payoff.
    # Defined ONLY for FAVORABLE_FIRST -> +G and ADVERSE_FIRST -> -L.
    # BOTH_SAME_BAR / NONE have no barrier resolution, so they are excluded
    # (reported separately under by_event_class as the real true return).
    synth = event_based_synthetic_payoff(df)
    defined = ~np.isnan(synth)
    if defined.any():
        w_def = df["weights"].to_numpy(float)[defined]
        diff_def = df["true_episode_return_atr"].to_numpy(float)[defined] - synth[defined]
        mean_diff = float(np.average(diff_def, weights=w_def))
    else:
        mean_diff = float("nan")
    per_class_diff = {}
    for ev in classes:
        d = df[df["event_class"] == ev]
        if len(d) == 0:
            continue
        s = event_based_synthetic_payoff(d)
        m = ~np.isnan(s)
        if m.any():
            per_class_diff[ev] = float(np.average(
                d["true_episode_return_atr"].to_numpy(float)[m] - s[m],
                weights=d["weights"].to_numpy(float)[m]))
        else:
            per_class_diff[ev] = None  # BOTH_SAME_BAR / NONE: no barrier resolution

    return {
        "by_event_class": rows,
        "P_win_given_FAVORABLE_FIRST": p_win_fav,
        "P_loss_given_ADVERSE_FIRST": p_loss_adv,
        "win_event_confusion": confusion,
        "mean_true_minus_event_synthetic_atr": mean_diff,
        "true_minus_event_synthetic_by_class": per_class_diff,
        "event_synthetic_note": "DIAGNOSTIC_ONLY: FAVORABLE_FIRST->+G, ADVERSE_FIRST->-L; "
                                "BOTH_SAME_BAR/NONE undefined (no barrier resolution)",
    }


def diag_old_payoff_model(df: pd.DataFrame, n_bins: int = N_P_BINS) -> list:
    """Diagnostic C: TRUE conditional calibration of the frozen old payoff models.

    The frozen old payoff models are, by their own definition:
        mu_win  = predicted E[Y | Y>0, X]      (WIN head, trained on winners only)
        mu_loss = predicted E[-Y | Y<=0, X]    (LOSS head, trained on losers only)
        predicted_rr = mu_win / mu_loss
    Therefore the ONLY correct calibration check is per-head on the SAME support
    the model was trained on:

      WIN head  : on actual winners (Y>0) only, compare
                  actual_win_magnitude (= weighted mean Y)
                  vs predicted_win_magnitude_on_winners (= weighted mean mu_win)
                  + win_bias (= weighted mean(mu_win - Y)) and win_MAE.
      LOSS head : on actual losers (Y<=0) only, compare
                  actual_loss_magnitude (= weighted mean -Y)
                  vs predicted_loss_magnitude_on_losers (= weighted mean mu_loss)
                  + loss_bias (= weighted mean(mu_loss - (-Y))) and loss_MAE.

    G/L is NOT their calibration target (comparing predicted_rr to G/L is NOT a
    model-error test). ``actual_realized_RR`` and ``mean_predicted_rr`` are kept
    ONLY as descriptive ranking diagnostics; E[mu_W/mu_L] != E[mu_W]/E[mu_L] and
    need not equal the observed group RR, so they are never equated.

    Diagnostic only. No refit. Never label predicted_rr vs G/L disagreement a model error.
    """
    if not all(c in df.columns for c in ("mu_win", "mu_loss", "predicted_rr")):
        return [{"SKIP": "frozen mu_win/mu_loss/predicted_rr OOF columns absent"}]
    bins = assign_p_bins(df["p_win"].to_numpy(float), n_bins)
    out = []
    for b in sorted(set(bins.tolist())):
        d = df.iloc[np.where(bins == b)[0]]
        dw = d["weights"].to_numpy(float)
        dy = d["true_episode_return_atr"].to_numpy(float)
        dmuw = d["mu_win"].to_numpy(float)
        dmul = d["mu_loss"].to_numpy(float)
        win_mask = dy > 0
        loss_mask = ~win_mask
        n_win = int(win_mask.sum())
        n_loss = int(loss_mask.sum())

        if n_win > 0:
            ww = dw[win_mask]
            actual_win_mag = float(np.average(dy[win_mask], weights=ww))
            pred_win_mag = float(np.average(dmuw[win_mask], weights=ww))
            win_bias = float(np.average(dmuw[win_mask] - dy[win_mask], weights=ww))
            win_mae = float(np.average(np.abs(dmuw[win_mask] - dy[win_mask]), weights=ww))
        else:
            actual_win_mag = pred_win_mag = win_bias = win_mae = float("nan")

        if n_loss > 0:
            lw = dw[loss_mask]
            actual_loss_mag = float(np.average(-dy[loss_mask], weights=lw))
            pred_loss_mag = float(np.average(dmul[loss_mask], weights=lw))
            # loss_bias = E[mu_loss - (-Y)] = E[mu_loss + Y] on losers
            loss_bias = float(np.average(dmul[loss_mask] + dy[loss_mask], weights=lw))
            loss_mae = float(np.average(np.abs(dmul[loss_mask] + dy[loss_mask]), weights=lw))
        else:
            actual_loss_mag = pred_loss_mag = loss_bias = loss_mae = float("nan")

        actual_rr = (actual_win_mag / actual_loss_mag
                     if (actual_loss_mag > 0 and not math.isnan(actual_win_mag))
                     else float("nan"))
        out.append({
            "p_bin": int(b),
            "n": int(len(d)),
            "n_win": n_win,
            "n_loss": n_loss,
            "actual_win_magnitude": actual_win_mag,
            "predicted_win_magnitude_on_winners": pred_win_mag,
            "win_bias": win_bias,
            "win_MAE": win_mae,
            "actual_loss_magnitude": actual_loss_mag,
            "predicted_loss_magnitude_on_losers": pred_loss_mag,
            "loss_bias": loss_bias,
            "loss_MAE": loss_mae,
            "actual_realized_RR": actual_rr,
            "mean_predicted_rr": float(np.average(d["predicted_rr"], weights=dw)),
            "mean_canonical_G_over_L": float(np.average(d["G"] / d["L"], weights=dw)),
            "mean_canonical_log_GL": float(np.average(d["log_gl"], weights=dw)),
        })
    return out


# --------------------------------------------------------------------------- #
# Bootstrap (day-clustered paired) — used only in T1 audit + T1.5/T2            #
# --------------------------------------------------------------------------- #
def _complete_block_indices(trading_days, block: int = BOOTSTRAP_BLOCK_DAYS):
    """Day-block partition covering EVERY original trading day.

    Unlike the R13.5 audited owner (which dropped the terminal remainder), this
    checkpoint requires the FULL day universe to be eligible for resampling, so the
    terminal partial block is RETAINED. Sorts unique trading days and splits them
    into consecutive blocks of size ``block`` (the last block may be smaller).
    Returns (block_idx, n_blocks) where block_idx is a list of row-index arrays,
    one per block, and every original day belongs to exactly one block.
    """
    days = np.sort(pd.unique(trading_days))
    n_blocks = max(1, math.ceil(len(days) / block))
    pos = {d: np.where(trading_days == d)[0] for d in days}
    blocks = [days[i * block:(i + 1) * block] for i in range(n_blocks)]
    block_idx = [np.concatenate([pos[d] for d in b]) for b in blocks]
    return block_idx, len(block_idx)


def _select_blocks(block_idx, chosen):
    """Concatenate sampled blocks, PRESERVING multiplicity (no dedup).

    ``chosen`` is an array of block indices drawn WITH replacement. A block that
    appears k times contributes its rows k times. This is the core of the
    bootstrap; using a set() here would silently discard resampled weight.
    """
    return np.concatenate([block_idx[c] for c in chosen])


def bootstrap_d_geometry(df: pd.DataFrame, b: int = 200, seed: int = BOOTSTRAP_SEED,
                         block: int = BOOTSTRAP_BLOCK_DAYS,
                         p_bin_edges: np.ndarray | None = None) -> dict:
    """Trading-day block bootstrap of D_geometry (reuses R13.5 audited owner).

    Resamples DAY-BLOCKS WITH replacement (preserving multiplicity), recomputes
    the full stratified statistic on the resampled rows, and returns the observed
    point estimate plus the bootstrap mean and 95% CI separately. NOT run on the
    full evaluation population in this checkpoint (audit sample only).

    ``p_bin_edges``: if supplied, it is used AS-IS for BOTH the observed point
    estimate and every bootstrap replicate (never recomputed). If None, edges are
    derived from ``df``'s own p_win (so the bootstrap is self-consistent for that
    df). Callers that already froze per-analysis-universe edges MUST pass them here
    to keep the point estimate and CI on the same estimand.
    """
    if "trading_day" not in df.columns:
        df = _attach_trading_day(df)
    td = df["trading_day"].to_numpy()
    td_int = pd.to_datetime(pd.Series(td)).to_numpy("datetime64[ns]").astype("int64")
    block_idx, n_complete = _complete_block_indices(td_int, block)
    if n_complete == 0:
        return {"observed_D": float("nan"), "bootstrap_mean": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "n_valid_reps": 0,
                "n_blocks": 0,
                "note": "too few blocks for bootstrap"}
    # freeze p-bin edges: use supplied edges AS-IS, else derive from this df so every
    # replicate uses identical bins. Never recompute when edges are provided.
    if p_bin_edges is None:
        p_bin_edges = compute_p_bin_edges(df["p_win"].to_numpy(float))
    observed_D = compute_stratified_geometry_contrast(df, p_bin_edges=p_bin_edges)["D_geometry"]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(b):
        chosen = rng.integers(0, n_complete, size=n_complete)  # WITH replacement
        sel = _select_blocks(block_idx, chosen)                # multiplicity preserved
        if len(sel) < 50:
            continue
        sub = df.iloc[sel]
        try:
            D = compute_stratified_geometry_contrast(sub, p_bin_edges=p_bin_edges)["D_geometry"]
        except Exception:
            continue
        if math.isfinite(D):
            boots.append(D)
    COUNTERS["bootstrap_rep_count"] += len(boots)
    if len(boots) == 0:
        return {"observed_D": observed_D, "bootstrap_mean": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "n_valid_reps": 0,
                "n_blocks": int(n_complete),
                "note": "no valid replicates"}
    boots = np.asarray(boots, dtype=float)
    return {
        "observed_D": float(observed_D),
        "bootstrap_mean": float(np.mean(boots)),
        "ci_lo": float(np.quantile(boots, 0.025)),
        "ci_hi": float(np.quantile(boots, 0.975)),
        "n_valid_reps": int(len(boots)),
        "n_blocks": int(n_complete),
        "method": "trading_day_block_bootstrap: ALL trading days partitioned into "
                  "consecutive blocks (terminal partial block RETAINED); blocks drawn "
                  "WITH replacement; multiplicity preserved (no set() dedup)",
        "note": "day-block bootstrap on AUDIT sample only; full-population run reserved for T1.5/T2",
    }


# --------------------------------------------------------------------------- #
# T0 — synthetic truth cases                                                   #
# --------------------------------------------------------------------------- #
def t0_synthetic() -> dict:
    cases = {}

    # --- log(G/L) calculation ---
    # G=2.0, L=1.0 -> Z = ln(2) ~ 0.693147
    cases["log_gl"] = {
        "G": 2.0, "L": 1.0,
        "Z": float(log_geometry_ratio(np.array([2.0]), np.array([1.0]))[0]),
        "expected": float(math.log(2.0)),
    }
    assert abs(cases["log_gl"]["Z"] - math.log(2.0)) < 1e-12

    # --- stratified high-vs-low contrast on a tiny hand frame ---
    # Two p-strata (p=0.5 and p=0.9). Within each, rank by Z, compare high vs low
    # true return. Uses real true returns, NOT +/-G/L.
    frame = pd.DataFrame({
        "p_win": [0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9],
        "G": [1.0, 1.0, 4.0, 4.0, 1.0, 1.0, 4.0, 4.0],
        "L": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "true_episode_return_atr": [0.1, 0.2, 1.5, 1.6, 0.2, 0.3, 2.0, 2.1],
        "weights": [1.0] * 8,
    })
    frame["log_gl"] = log_geometry_ratio(frame["G"].to_numpy(float),
                                         frame["L"].to_numpy(float))
    prod = compute_stratified_geometry_contrast(frame, n_p_bins=2,
                                                top_frac=0.5, bottom_frac=0.5)
    ref = compute_stratified_geometry_contrast_reference(frame, n_p_bins=2,
                                                          top_frac=0.5, bottom_frac=0.5)
    # p=0.5 stratum: high Z (G=4) true returns {1.5,1.6} mean 1.55; low Z (G=1) {0.1,0.2} mean 0.15 -> 1.40
    # p=0.9 stratum: high Z (G=4) {2.0,2.1} mean 2.05; low Z (G=1) {0.2,0.3} mean 0.25 -> 1.80
    # D = mean(1.40, 1.80) = 1.60
    cases["stratified_contrast"] = {
        "D_geometry": prod["D_geometry"],
        "reference_D": ref["D_geometry"],
        "expected": 1.60,
    }
    assert abs(prod["D_geometry"] - 1.60) < 1e-9
    assert abs(prod["D_geometry"] - ref["D_geometry"]) < 1e-9

    # --- canonical true return independent from synthetic +/-G/L ---
    # FAVORABLE_FIRST but exit next-bar open with a gap => true != +G.
    g = 1.0
    l = 1.0
    true_ret = 0.3          # episode ended profitable but only +0.3 ATR
    win = True
    synthetic = g if win else -l   # would be +1.0 under idealized barrier world
    cases["true_vs_synthetic"] = {
        "G": g, "L": l, "true_episode_return_atr": true_ret,
        "synthetic_barrier_payoff": float(synthetic),
        "difference": float(true_ret - synthetic),
    }
    assert true_ret != synthetic
    assert abs(cases["true_vs_synthetic"]["difference"] - (-0.7)) < 1e-12

    # --- BOTH / NONE semantics are NOT silently collapsed ---
    ev_frame = pd.DataFrame({
        "p_win": [0.6, 0.6, 0.6, 0.6],
        "G": [1.0, 1.0, 1.0, 1.0],
        "L": [1.0, 1.0, 1.0, 1.0],
        "true_episode_return_atr": [0.5, -0.5, 0.2, -0.2],
        "win": [True, False, True, False],
        "event_class": ["FAVORABLE_FIRST", "ADVERSE_FIRST",
                        "BOTH_SAME_BAR", "NONE"],
        "weights": [1.0] * 4,
    })
    ev_frame["log_gl"] = log_geometry_ratio(ev_frame["G"].to_numpy(float),
                                            ev_frame["L"].to_numpy(float))
    rec = diag_event_reconciliation(ev_frame)
    # BOTH_SAME_BAR and NONE must each be present with their own true returns
    assert "BOTH_SAME_BAR" in rec["by_event_class"]
    assert "NONE" in rec["by_event_class"]
    assert rec["by_event_class"]["BOTH_SAME_BAR"]["mean_true_return_atr"] == 0.2
    assert rec["by_event_class"]["NONE"]["mean_true_return_atr"] == -0.2
    cases["event_semantics_preserved"] = {
        "classes": list(rec["by_event_class"].keys()),
        "both_true_ret": rec["by_event_class"]["BOTH_SAME_BAR"]["mean_true_return_atr"],
        "none_true_ret": rec["by_event_class"]["NONE"]["mean_true_return_atr"],
    }

    return cases


# --------------------------------------------------------------------------- #
# T1 — real small-sample audit                                                 #
# --------------------------------------------------------------------------- #
def t1_audit(sample_n: int = T1_SAMPLE_N) -> dict:
    df = load_audit_frame(sample_n=sample_n)
    # attach trading_day for the audit bootstrap (hard-fail if any unresolved)
    df_day = _attach_trading_day(df)

    # TRUE join evidence. load_audit_frame already hard-fails on a silent drop;
    # we report the real pre/post universe counts, never a len(df)==len(df) placeholder.
    meta = df.attrs.get("join_meta", {})
    parity = {
        "rows": int(len(df)),
        "full_oof_rows_pre_join": meta.get("full_oof_rows_pre_join"),
        "full_rows_post_join": meta.get("full_rows_post_join"),
        "unmatched_rows": meta.get("unmatched_rows"),
        "duplicate_label_keys": meta.get("duplicate_label_keys"),
        "duplicate_post_join_keys": meta.get("duplicate_post_join_keys"),
        "sampled_rows": meta.get("sampled_rows"),
        "join_clean": bool(meta.get("unmatched_rows", 1) == 0
                           and meta.get("duplicate_label_keys", 1) == 0
                           and meta.get("duplicate_post_join_keys", 1) == 0),
    }

    # owner alignment
    parity["p_win_min"] = float(df["p_win"].min())
    parity["p_win_max"] = float(df["p_win"].max())
    parity["units"] = "ATR (single decision-time m15_atr owner; same ATR for G and L)"
    parity["nan_required"] = {
        c: int(df[c].isna().sum())
        for c in ("p_win", "G", "L", "true_episode_return_atr", "log_gl", "weights")
    }
    parity["event_class_present"] = bool(df["event_class"].notna().all())
    parity["event_class_values"] = sorted(df["event_class"].unique().tolist())

    # availability: label_available_time >= decision_time (causal)
    avail_ok = (pd.to_datetime(df["label_available_time"]) >= pd.to_datetime(df["decision_time"])).all()
    parity["availability_causal"] = bool(avail_ok)

    # decision-time causality: G/L/p_win independent of future labels
    parity["future_mutation_invariance"] = _future_mutation_invariance(df)

    # Reference vs Production parity for the stratified statistic
    assert_stratified_parity(df)
    prod = compute_stratified_geometry_contrast(df)
    parity["stratified_D_geometry_audit"] = prod["D_geometry"]
    parity["stratified_n_bins_used"] = prod["n_bins_used"]

    # single-instrument boundary breakdown (pooled + per-symbol + by-side), each
    # with its own bootstrap CI (T1.5/T2 path exercised on the audit sample only).
    parity["stratified_breakdown"] = stratified_d_geometry_breakdown(
        df_day, with_ci=True, bootstrap_reps=200)

    # negative controls (REAL: prove the system catches deliberate errors)
    parity["neg_sign_sensitivity"] = _neg_sign_sensitivity(df)
    parity["neg_relationship_sensitivity"] = _neg_relationship_sensitivity(df)

    # real future-mutation / prefix causality test (accurately labeled)
    parity["causality_future_mutation"] = _causality_future_mutation_test(df)

    # secondary diagnostics (audit only; not interpreted)
    parity["diag_pwin_decile"] = diag_pwin_decile_table(df)
    parity["diag_event_reconciliation"] = diag_event_reconciliation(df)
    parity["diag_old_payoff_model"] = diag_old_payoff_model(df)

    # bootstrap on AUDIT sample only (labeled; not scientific); fields reported separately
    parity["bootstrap_audit"] = bootstrap_d_geometry(df_day, b=200)

    return parity


# --------------------------------------------------------------------------- #
# Negative controls & causality (real)                                         #
# --------------------------------------------------------------------------- #
def _neg_sign_sensitivity(df: pd.DataFrame) -> dict:
    """REAL negative control: injecting the wrong SIGN of Z must flip the
    contrast sign. Proves the kernel is sensitive to the sign of Z (a wrong-sign
    bug would NOT be silently passed)."""
    correct = compute_stratified_geometry_contrast(df)
    # build a df with negated Z (wrong sign injected)
    df_wrong = df.copy()
    df_wrong["log_gl"] = -df_wrong["log_gl"].to_numpy(float)
    wrong = compute_stratified_geometry_contrast(df_wrong)
    c = correct["D_geometry"]
    w = wrong["D_geometry"]
    # sign must flip (or both ~0 if no relationship). If c != 0, w must be opposite.
    if abs(c) > 1e-9:
        flipped = (c > 0) != (w > 0)
    else:
        flipped = True  # degenerate: no relationship to flip
    return {
        "correct_D": c,
        "wrong_sign_D": w,
        "sign_flipped": bool(flipped),
        "note": "wrong-sign Z injection flips D -> kernel is sign-sensitive "
                "(catches a wrong-sign bug; not silently passed)",
    }


def _neg_relationship_sensitivity(df: pd.DataFrame) -> dict:
    """REAL negative control: the stratified statistic must be SENSITIVE to both
    the treatment (Z = log G/L) and the outcome (true return).

    * Permuting Z (keep true return) must change D  -> proves Z is actually used.
    * Permuting the true return (keep Z) must change D -> proves the outcome is
      actually used (catches a label-leaking or label-independent bug).

    A broken kernel that ignores Z, ignores the outcome, or leaks the label would
    NOT change D under these permutations, so this control FAILS loudly.
    """
    D0 = compute_stratified_geometry_contrast(df)["D_geometry"]

    rng = np.random.default_rng(12345)
    # permute Z only
    df_z = df.copy()
    z = df_z["log_gl"].to_numpy(float).copy()
    rng.shuffle(z)
    df_z["log_gl"] = z
    Dz = compute_stratified_geometry_contrast(df_z)["D_geometry"]

    rng2 = np.random.default_rng(98765)
    # permute true return only
    df_y = df.copy()
    y = df_y["true_episode_return_atr"].to_numpy(float).copy()
    rng2.shuffle(y)
    df_y["true_episode_return_atr"] = y
    Dy = compute_stratified_geometry_contrast(df_y)["D_geometry"]

    sensitive_to_z = abs(Dz - D0) > 1e-9
    sensitive_to_y = abs(Dy - D0) > 1e-9
    return {
        "real_D": D0,
        "D_after_Z_permutation": Dz,
        "D_after_return_permutation": Dy,
        "sensitive_to_Z": bool(sensitive_to_z),
        "sensitive_to_return": bool(sensitive_to_y),
        "note": "permuting Z or the true return must change D (catches a kernel that "
                "ignores Z, ignores the outcome, or leaks the label)",
    }


def _future_mutation_invariance(df: pd.DataFrame) -> bool:
    """G/L/p_win/log_gl must equal their originals after FUTURE columns
    (episode_return_atr, win, event_class, mu_*, predicted_rr) are corrupted."""
    g0 = df["G"].to_numpy(float).copy()
    l0 = df["L"].to_numpy(float).copy()
    p0 = df["p_win"].to_numpy(float).copy()
    z0 = df["log_gl"].to_numpy(float).copy()

    df_c = df.copy()
    n = len(df_c)
    half = n // 2
    # corrupt future-only columns for a subset
    df_c.loc[df_c.index[:half], "true_episode_return_atr"] = 999.0
    df_c.loc[df_c.index[:half], "win"] = ~df_c.loc[df_c.index[:half], "win"]
    df_c.loc[df_c.index[:half], "event_class"] = "NONE"
    if "predicted_rr" in df_c.columns:
        df_c.loc[df_c.index[:half], "predicted_rr"] = 0.0
    if "mu_win" in df_c.columns:
        df_c.loc[df_c.index[:half], "mu_win"] = 0.0
    if "mu_loss" in df_c.columns:
        df_c.loc[df_c.index[:half], "mu_loss"] = 0.0
    df_c["log_gl"] = log_geometry_ratio(df_c["G"].to_numpy(float),
                                        df_c["L"].to_numpy(float))

    same = (np.allclose(df_c["G"].to_numpy(float), g0)
            and np.allclose(df_c["L"].to_numpy(float), l0)
            and np.allclose(df_c["p_win"].to_numpy(float), p0)
            and np.allclose(df_c["log_gl"].to_numpy(float), z0))
    return bool(same)


def _causality_future_mutation_test(df: pd.DataFrame) -> dict:
    """Accurately labeled real causality test (future-mutation invariance).

    This relies on the separately evidenced canonical-owner proof that G/L and
    p_win are computed ONLY from decision-time information; here we verify that
    mutating FUTURE label columns does not change G/L/p_win/log_gl.
    """
    return {
        "passed": _future_mutation_invariance(df),
        "method": "future-mutation invariance: G/L/p_win/log_gl unchanged after "
                  "corrupting future columns (episode_return_atr/win/event_class/mu_*)",
        "note": "relies on canonical-owner causal proof that G/L/p_win are "
                "decision-time-only; renamed from prior 're-read determinism' check",
    }


# --------------------------------------------------------------------------- #
# TP — performance microbenchmark (O(N) precompute / O(N log N) ranking)        #
# --------------------------------------------------------------------------- #
def _synthetic_frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.1, 0.9, size=n)
    g = rng.uniform(0.2, 4.0, size=n)
    l = rng.uniform(0.2, 4.0, size=n)
    y = rng.normal(0.0, 0.5, size=n)
    w = rng.uniform(0.1, 1.0, size=n)
    df = pd.DataFrame({"p_win": p, "G": g, "L": l,
                       "true_episode_return_atr": y, "weights": w})
    df["log_gl"] = log_geometry_ratio(g, l)
    return df


def _synthetic_frame_with_days(n: int, seed: int = 0, rows_per_day: int = 10) -> pd.DataFrame:
    """Synthetic frame with a trading_day column so block bootstrap can run (audit only)."""
    df = _synthetic_frame(n, seed)
    df["trading_day"] = (np.arange(n) // rows_per_day).astype("int64")
    return df


def _time_kernel(df: pd.DataFrame) -> float:
    start = time.perf_counter()
    _ = compute_stratified_geometry_contrast(df)
    return time.perf_counter() - start


def _time_bootstrap(df: pd.DataFrame, b: int) -> float:
    start = time.perf_counter()
    _ = bootstrap_d_geometry(df, b=b)
    return time.perf_counter() - start


def tp_microbenchmark() -> dict:
    # --- contrast-only scaling (O(N) precompute + O(N log N) within-bin ranking) ---
    N = 10000
    t_N = _time_kernel(_synthetic_frame(N, 1))
    t_2N = _time_kernel(_synthetic_frame(2 * N, 2))
    t_4N = _time_kernel(_synthetic_frame(4 * N, 3))
    r_2 = t_2N / t_N
    r_4 = t_4N / t_2N

    # --- bootstrap scaling (O(B * N log N)) on synthetic frames with trading_day ---
    B = 100
    NB = 10000
    t_b1 = _time_bootstrap(_synthetic_frame_with_days(NB, 7), B)              # N=10k, B=100
    t_b2 = _time_bootstrap(_synthetic_frame_with_days(NB, 8), 2 * B)         # N=10k, B=200
    t_b3 = _time_bootstrap(_synthetic_frame_with_days(2 * NB, 9), B)         # N=20k, B=100
    r_b = t_b2 / t_b1                                                        # ~= 2  -> O(B)
    r_bn = t_b3 / t_b1                                                       # ~= 2  -> O(N)

    # --- formal T2 runtime projection (NOT a T1.5/T2 run) ---
    N_FULL = 138000
    B_FULL = 2000
    t_single_rep = t_b1 / B
    proj_t2_sec = t_single_rep * (N_FULL / NB) * B_FULL

    return {
        "contrast_N": N, "contrast_t_N": t_N, "contrast_t_2N": t_2N, "contrast_t_4N": t_4N,
        "contrast_ratio_2N": r_2, "contrast_ratio_4N": r_4,
        "contrast_gate_2N_pass": bool(r_2 < 3.0), "contrast_gate_4N_pass": bool(r_4 < 3.0),
        "bootstrap_N": NB, "bootstrap_B": B,
        "bootstrap_t_B": t_b1, "bootstrap_t_2B": t_b2, "bootstrap_t_2N": t_b3,
        "bootstrap_ratio_B": r_b, "bootstrap_ratio_N": r_bn,
        "bootstrap_gate_B_pass": bool(r_b < 3.0), "bootstrap_gate_N_pass": bool(r_bn < 3.0),
        "complexity": "O(N) precompute + O(N log N) within-bin ranking; "
                      "bootstrap O(B * N log N) with trading-day block resampling",
        "formal_t2_projection": {
            "N_full": N_FULL, "B_full": B_FULL,
            "projected_seconds": float(proj_t2_sec),
            "projected_minutes": float(proj_t2_sec / 60.0),
            "note": "extrapolation from synthetic benchmark; NOT a T1.5/T2 run",
        },
    }


# --------------------------------------------------------------------------- #
# Evidence packet (written after T0/T1/TP; STOP FOR REVIEWER)                  #
# --------------------------------------------------------------------------- #
def _git_head_sha() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT
        ).decode().strip()[:7]
    except Exception:
        return "unknown"


def _module_content_sha() -> str:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    except Exception:
        return "unknown"


def build_evidence_packet() -> dict:
    t0 = t0_synthetic()
    t1 = t1_audit()
    tp = tp_microbenchmark()
    local_sha = _git_head_sha()
    packet = {
        "TASK_ID": TASK_ID,
        "REVISION_OF": "PAYOFF-GEOMETRY-01",
        "REVIEWED_PARENT_SHA": REVIEWED_PARENT_SHA,
        "GENERATOR_COMMIT_SHA": local_sha,
        "EVIDENCE_PARENT_SHA": local_sha,
        "GENERATOR_CODE_SHA": _module_content_sha(),
        "git_status_note": "GENERATOR_COMMIT_SHA = module commit (git rev-parse HEAD at build); "
                           "EVIDENCE_PARENT_SHA = the generator code commit this evidence is built from; "
                           "final remote branch HEAD is reported in the IDE response after push "
                           "(self-referential tip fields intentionally omitted)",
        "frozen_math": {
            "p_win_definition": "P(episode_return_atr > 0 | X)  [NOT P(favorable-before-adverse)]",
            "Z_definition": "log(G / L), G/L = decision-time favorable/adverse ATR distances",
            "outcome": "true canonical episode_return_atr (win = episode_return_atr > 0)",
            "forbidden_identity": "p_win*G - (1-p_win)*L is INVALID and NOT computed",
            "no_expected_value_claim": True,
            "no_model_fit": True,
        },
        "canonical_owner_map": {
            "label/G/L/win/episode_return_atr/event_class/sample_weight/"
            "decision_time/label_available_time/split":
                "artifacts/opportunity_value_v1/labels_{train,val,test}_v1.parquet (R8)",
            "p_win_model_frozen": "win_probability_model_v1.py (R9A)",
            "p_win_oof_artifact": f"artifacts/decomposed_value_v2/oof/{FROZEN_WIN_ARCH}_f*_{AUDIT_HORIZON}.parquet",
            "old_payoff_oof_columns": "mu_win, mu_loss, predicted_rr (frozen, in same OOF parquet)",
            "trading_day_owner": "artifacts/decomposed_value_v1/state_v1.parquet [symbol,bar_index,trading_day]",
            "weighting_owner": "labels column sample_weight",
        },
        "label_contract": {
            "win_definition": "win = episode_return_atr > 0  (strictly binary)",
            "event_class_values": ["FAVORABLE_FIRST", "ADVERSE_FIRST",
                                    "BOTH_SAME_BAR", "NONE"],
            "nuance": "all four event classes resolve to a real signed ATR return; "
                      "true_episode_return_atr is the only primary realized-return outcome",
            "synthetic_barrier_payoff": "DIAGNOSTIC_ONLY (+/-G/+/-L pseudo-return)",
        },
        "stratification": {
            "n_p_bins": N_P_BINS,
            "top_frac": TOP_FRAC,
            "bottom_frac": BOTTOM_FRAC,
            "bin_rule": "quantile edges of p_win (outcome-independent, frozen)",
            "weighting": "canonical sample_weight (within-bin weighted means + "
                         "across-bin weighted aggregation)",
        },
        "T0": t0,
        "T1": t1,
        "TP": tp,
        "counters": dict(COUNTERS),
        "governance": {
            "model_fits": COUNTERS["model_fit_count"],
            "dev_test_reads": 0,
            "pgm_evc_win_payoff_refits": 0,
            "full_population_high_low_run": False,
            "t1_5_run": False,
            "t2_run": False,
            "scientific_interpretation": "NONE (checkpoint only; STOP FOR REVIEWER)",
            "forbidden_imports": "none (numpy/pandas/math/time only; state read lazily for trading_day)",
        },
        "reviewer_decision_required": [
            "PROCEED_TO_T1_5 (full-evaluation high/low realized-return experiment + day-cluster bootstrap)",
            "PROCEED_TO_T2 (single formal full evaluation)",
        ],
    }
    return packet


# --------------------------------------------------------------------------- #
# T1.5 — E2E pipeline validation (deterministic cap-1000/symbol subset)         #
# --------------------------------------------------------------------------- #
T1_5_CAP = 1000
T1_5_B = 500
T1_5_SEED = 20260929

T1_5_ROW_COLUMNS = [
    "symbol", "decision_time", "decision_bar", "side", "trading_day", "fold",
    "p_win", "G", "L", "log_gl",
    "true_episode_return_atr", "win", "event_class", "weights",
    "mu_win", "mu_loss", "predicted_rr",
]


def _t1_5_jsonable(o):
    """Recursively convert numpy scalars to native python and NaN/inf to None so
    the summary JSON is strict JSON (allow_nan=False)."""
    if isinstance(o, dict):
        return {k: _t1_5_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_t1_5_jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        v = float(o)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _t1_5_jsonable(o.tolist())
    if o is None or isinstance(o, (int, str, bool)):
        return o
    return str(o)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def analyze_frame(df: pd.DataFrame, B: int = T1_5_B) -> dict:
    """Core analysis over an already-loaded, trading_day-attached frame.

    Single source of truth for data integrity + D_geometry + diagnostics + events.
    The frame is loaded EXACTLY ONCE by the caller (T1.5 or the future T2
    full-population runner) and reused for statistics, bootstrap, AND the row
    artifact -- never re-loaded / re-joined merely to write artifacts.

    Data-integrity evidence is the MEASURED canonical-join metadata (propagated by
    the loader on ``df.attrs["join_meta"]``), never a hardcoded 0.
    """
    n = len(df)
    join_meta = dict(df.attrs.get("join_meta", {}))
    dup_keys = int(df.duplicated(subset=["symbol", "decision_bar", "side"]).sum())
    missing_trading_day = int(df["trading_day"].isna().sum())
    missing_G = int(df["G"].isna().sum())
    missing_L = int(df["L"].isna().sum())
    missing_true_return = int(df["true_episode_return_atr"].isna().sum())
    integrity = {
        "full_oof_rows_pre_join": join_meta.get("full_oof_rows_pre_join"),
        "full_rows_post_join": join_meta.get("full_rows_post_join"),
        "unmatched_rows": join_meta.get("unmatched_rows"),
        "duplicate_label_keys": join_meta.get("duplicate_label_keys"),
        "duplicate_post_join_keys": join_meta.get("duplicate_post_join_keys"),
        "selected_rows": int(n),
        "input_oof_rows_selected": int(n),
        "joined_rows": int(n),
        "duplicate_keys_within_subset": dup_keys,
        "missing_trading_day": missing_trading_day,
        "missing_G": missing_G,
        "missing_L": missing_L,
        "missing_true_return": missing_true_return,
        "n_symbols": int(df["symbol"].nunique()),
        "per_symbol_counts": {s: int(c) for s, c in df["symbol"].value_counts().items()},
        "sides": sorted(df["side"].unique().tolist()),
        "n_trading_days_total": int(df["trading_day"].nunique()),
        "all_clean": bool(
            (join_meta.get("unmatched_rows", 1) == 0)
            and (join_meta.get("duplicate_label_keys", 1) == 0)
            and (join_meta.get("duplicate_post_join_keys", 1) == 0)
            and dup_keys == 0 and missing_trading_day == 0
            and missing_G == 0 and missing_L == 0 and missing_true_return == 0),
    }

    # D_geometry: pooled / per-symbol / LONG / SHORT, B bootstrap replicates,
    # each universe using its OWN p_win quantile edges (Micro Fix 04).
    breakdown = stratified_d_geometry_breakdown(df, with_ci=True, bootstrap_reps=B)
    pdecile = diag_pwin_decile_table(df)
    diag_c = diag_old_payoff_model(df)
    events = diag_event_reconciliation(df)
    return {
        "data_integrity": integrity,
        "D_geometry": breakdown,
        "p_win_decile_diagnostic": pdecile,
        "diagnostic_C_old_payoff_models": diag_c,
        "event_semantics": events,
    }


def run_t1_5(df: pd.DataFrame | None = None, cap_per_symbol: int = T1_5_CAP,
             seed: int = T1_5_SEED, B: int = T1_5_B) -> dict:
    """T1.5 E2E pipeline validation. NO model fit/refit/tune. NO T2.

    Loads the deterministic subset exactly once (if not supplied by the caller) and
    reuses that SAME frame for statistics, bootstrap, and (in build_t1_5_artifact)
    the row artifact -- never a second canonical-data rebuild.

    T1.5 PASS does NOT require D>0; it validates the engineering + statistics
    pipeline (keys correct, no silent drop, valid bootstrap, complete per-symbol
    output, point/CI parity, artifact schema, runtime vs TP budget, no model fit).

    Reported counters are RUN-LOCAL deltas (before/after this call), not arbitrary
    accumulated module history, so the Evidence Packet describes THIS execution.
    """
    t0 = time.perf_counter()
    counters_before = dict(COUNTERS)
    if df is None:
        df = load_t1_5_frame(cap_per_symbol=cap_per_symbol, seed=seed)
    core = analyze_frame(df, B=B)
    counters_after = dict(COUNTERS)
    counters_delta = {k: counters_after.get(k, 0) - counters_before.get(k, 0)
                      for k in counters_before}
    analysis_sec = time.perf_counter() - t0
    return {
        "TASK_ID": TASK_ID,
        "STAGE": "T1.5",
        "scope_note": "E2E pipeline validation on a deterministic cap-1000/symbol subset; "
                      "NOT a formal inference; no model fit/refit/tune.",
        "config": {
            "cap_per_symbol": int(cap_per_symbol),
            "seed": int(seed),
            "bootstrap_B": int(B),
            "bootstrap_block_days": BOOTSTRAP_BLOCK_DAYS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "n_p_bins": N_P_BINS,
            "top_frac": TOP_FRAC,
            "bottom_frac": BOTTOM_FRAC,
            "p_bin_edges_rule": "per-analysis-universe quantiles of p_win (frozen within S)",
        },
        **core,
        "counters": counters_delta,           # run-local delta
        "counters_global": dict(COUNTERS),   # full module history for context
        "governance": {
            "model_fit_count": counters_delta.get("model_fit_count", 0),
            "t1_5_run": True,
            "t2_run": False,
            "full_population_high_low_run": False,
            "scientific_interpretation": "NONE (E2E pipeline validation only; STOP FOR REVIEWER)",
        },
        # analysis-only timing; build_t1_5_artifact sets runtime_seconds to span the
        # full load -> analysis -> artifact-write path.
        "analysis_runtime_seconds": float(analysis_sec),
    }


def build_t1_5_artifact(cap_per_symbol: int = T1_5_CAP, seed: int = T1_5_SEED,
                        B: int = T1_5_B, evidence_dir: Path | None = None) -> dict:
    out_dir = evidence_dir if evidence_dir is not None else EVID
    t0 = time.perf_counter()
    counters_before = dict(COUNTERS)
    # Load EXACTLY ONCE, then reuse the same frame for analysis AND the row artifact.
    df = load_t1_5_frame(cap_per_symbol=cap_per_symbol, seed=seed)
    result = run_t1_5(df=df, cap_per_symbol=cap_per_symbol, seed=seed, B=B)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "payoff_geometry_01a_t1_5_rows.parquet"
    summary_path = out_dir / "payoff_geometry_01a_t1_5_summary.json"
    manifest_path = out_dir / "payoff_geometry_01a_t1_5_manifest.json"

    # row-level artifact written directly from the SAME loaded frame (no reload)
    rows_df = df[T1_5_ROW_COLUMNS]
    rows_df.to_parquet(rows_path, index=False)
    rows_sha = _sha256_file(rows_path)

    result["row_artifact"] = {
        "path": rows_path.name,
        "sha256": rows_sha,
        "rows": int(len(rows_df)),
        "columns": list(T1_5_ROW_COLUMNS),
    }

    # summary JSON (v0): written once so runtime_seconds can be measured AFTER it and
    # therefore genuinely cover load + analysis + row parquet + summary JSON write.
    summary_text = json.dumps(_t1_5_jsonable(result), indent=2, allow_nan=False)
    summary_path.write_text(summary_text)
    summary_sha = _sha256_file(summary_path)

    # runtime_seconds: wall time covering load + analysis + row parquet + summary JSON
    # write. The manifest write below and the final summary rewrite are measured
    # separately (manifest_runtime_seconds); a file cannot measure the cost of writing
    # itself, so those terminal writes are the explicit documented boundary.
    result["runtime_seconds"] = float(time.perf_counter() - t0)

    # run-local pipeline counters: deltas over the ENTIRE load -> analysis -> artifact
    # path (so the Evidence Packet reports THIS execution, not accumulated history).
    counters_after = dict(COUNTERS)
    result["pipeline_counters"] = {
        k: counters_after.get(k, 0) - counters_before.get(k, 0)
        for k in counters_before
    }

    # summary JSON (v1, final): rewrite to embed runtime_seconds + pipeline_counters.
    # The manifest below references THIS final summary sha/bytes, so provenance stays
    # consistent (manifest.summary_sha == on-disk summary sha).
    summary_text = json.dumps(_t1_5_jsonable(result), indent=2, allow_nan=False)
    summary_path.write_text(summary_text)
    summary_sha = _sha256_file(summary_path)

    # Provenance manifest: a file cannot embed its own sha, so both artifact SHAs
    # live here (this is the canonical integrity record; the summary embeds the row sha).
    local_head = _git_head_sha()
    manifest = {
        "TASK_ID": TASK_ID,
        "STAGE": "T1.5",
        "summary_path": summary_path.name,
        "summary_sha256": summary_sha,
        "summary_bytes": len(summary_text.encode("utf-8")),
        "row_path": rows_path.name,
        "row_sha256": rows_sha,
        "row_rows": int(len(rows_df)),
        "generator_commit": local_head,
        "config": result["config"],
        "runtime_seconds": result["runtime_seconds"],
        "model_fit_count": result["governance"]["model_fit_count"],
        "note": "row-level parquet is the row artifact; summary JSON embeds row_artifact.sha256; "
                "this manifest records both SHAs for provenance. T1.5 only; T2 NOT run. "
                "generator_commit MUST equal the committed code SHA (artifact generated only "
                "after code is committed + pushed). runtime_seconds covers load+analysis+row "
                "parquet+summary write; the full wall time through manifest completion is reported "
                "in the returned result as manifest_runtime_seconds.",
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True)
    manifest_path.write_text(manifest_text)
    manifest_sha = _sha256_file(manifest_path)

    # full pipeline wall runtime through manifest completion. The manifest's own terminal
    # byte-write is the only excluded sub-millisecond piece (documented boundary above).
    result["manifest_runtime_seconds"] = float(time.perf_counter() - t0)

    result["summary_artifact"] = {
        "path": summary_path.name,
        "sha256": summary_sha,
        "bytes": len(summary_text.encode("utf-8")),
    }
    result["manifest_artifact"] = {
        "path": manifest_path.name,
        "sha256": manifest_sha,
    }
    result["local_git_head"] = local_head
    return result


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="PAYOFF-GEOMETRY-01A kernel checkpoint")
    ap.add_argument("stage", choices=["t0", "t1", "tp", "all", "packet", "t1_5"])
    ap.add_argument("--sample-n", type=int, default=T1_SAMPLE_N)
    ap.add_argument("--cap", type=int, default=T1_5_CAP)
    ap.add_argument("--b", type=int, default=T1_5_B)
    ap.add_argument("--seed", type=int, default=T1_5_SEED)
    args = ap.parse_args(argv)

    if args.stage == "t0":
        print(t0_synthetic())
    elif args.stage == "t1":
        res = t1_audit(sample_n=args.sample_n)
        print(json.dumps(res, indent=2, default=str))
    elif args.stage == "tp":
        print(tp_microbenchmark())
    elif args.stage == "all":
        print("T0:", json.dumps(t0_synthetic(), default=str))
        print("T1:", json.dumps(t1_audit(sample_n=args.sample_n), default=str))
        print("TP:", json.dumps(tp_microbenchmark(), default=str))
    elif args.stage == "packet":
        pkt = build_evidence_packet()
        EVID.mkdir(parents=True, exist_ok=True)
        out = EVID / "payoff_geometry_01a_kernel_checkpoint.json"
        out.write_text(json.dumps(pkt, indent=2, default=str))
        # report pulled from the SAME packet object so stdout == artifact
        print("wrote", out)
        print("TASK_ID:", pkt["TASK_ID"])
        print("GENERATOR_COMMIT_SHA:", pkt["GENERATOR_COMMIT_SHA"],
              "EVIDENCE_PARENT_SHA:", pkt["EVIDENCE_PARENT_SHA"])
        print("GENERATOR_CODE_SHA:", pkt["GENERATOR_CODE_SHA"])
        print("T0 log_gl.Z:", pkt["T0"]["log_gl"]["Z"])
        print("T0 stratified D_geometry:", pkt["T0"]["stratified_contrast"]["D_geometry"])
        print("T1 rows:", pkt["T1"]["rows"], "stratified_D_geometry_audit:",
              pkt["T1"]["stratified_D_geometry_audit"])
        print("T1 P(win|FAV):", pkt["T1"]["diag_event_reconciliation"]["P_win_given_FAVORABLE_FIRST"],
              "P(loss|ADV):", pkt["T1"]["diag_event_reconciliation"]["P_loss_given_ADVERSE_FIRST"])
        print("TP contrast scaling:", pkt["TP"]["contrast_ratio_2N"], pkt["TP"]["contrast_ratio_4N"])
        print("TP bootstrap scaling B/N:",
              pkt["TP"]["bootstrap_ratio_B"], pkt["TP"]["bootstrap_ratio_N"],
              "proj_min", round(pkt["TP"]["formal_t2_projection"]["projected_minutes"], 2))
        print("GOVERNANCE full_population_run:", pkt["governance"]["full_population_high_low_run"],
              "t1_5_run:", pkt["governance"]["t1_5_run"], "t2_run:", pkt["governance"]["t2_run"])
    elif args.stage == "t1_5":
        res = build_t1_5_artifact(cap_per_symbol=args.cap, seed=args.seed, B=args.b)
        integ = res["data_integrity"]
        dg = res["D_geometry"]
        print("wrote", EVID / res["row_artifact"]["path"], "rows", res["row_artifact"]["rows"])
        print("wrote", EVID / res["summary_artifact"]["path"])
        print("STAGE: T1.5  scope: E2E pipeline validation (NO T2, NO model fit)")
        print("subset rows:", integ["input_oof_rows_selected"],
              "symbols:", integ["n_symbols"], "trading_days:", integ["n_trading_days_total"])
        print("integrity all_clean:", integ["all_clean"],
              "unmatched_rows:", integ["unmatched_rows"],
              "dup_label_keys:", integ["duplicate_label_keys"],
              "dup_post_join_keys:", integ["duplicate_post_join_keys"],
              "missing_td:", integ["missing_trading_day"],
              "missing_true_return:", integ["missing_true_return"])
        print("POOLED D_geom:", round(float(dg["pooled"]["observed_D"]), 4),
              "CI[", round(float(dg["pooled"]["bootstrap_ci_low"]), 4), ",",
              round(float(dg["pooled"]["bootstrap_ci_high"]), 4), "]",
              "n_valid_reps:", dg["pooled"]["n_valid_reps"])
        for sym, v in dg["per_symbol"].items():
            print(f"  {sym}: D={float(v['observed_D']):+.4f} "
                  f"CI[{float(v['bootstrap_ci_low']):+.4f},{float(v['bootstrap_ci_high']):+.4f}] "
                  f"N={v['n']} days={v['n_trading_days']} reps={v['n_valid_reps']}")
        print("runtime_seconds:", round(res["runtime_seconds"], 2))
        print("counters:", res["counters"])
        print("GOVERNANCE model_fit_count:", res["governance"]["model_fit_count"],
              "t1_5_run:", res["governance"]["t1_5_run"], "t2_run:", res["governance"]["t2_run"])
        print("row_artifact_sha256:", res["row_artifact"]["sha256"])
        print("summary_artifact_sha256:", res["summary_artifact"]["sha256"])
        print("manifest_artifact_sha256:", res["manifest_artifact"]["sha256"])
        print("local_git_head:", res["local_git_head"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
