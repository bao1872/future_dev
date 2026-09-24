"""FUTURE-R11-R14 V2 — R11 Stability Atlas (plan §8 / §9).

TRADING_METRICS: NOT_APPLICABLE
reason: No trading action has been defined.

R11 is DIAGNOSTIC ONLY. It answers "where is the model stable?" and makes no
pass/fail judgement per regime, and it must never be used to select symbols,
sides or regimes (§31 / §45 / §46).

Dimensions
----------
time      : outer fold 1..5
symbol    : all 15 symbols, none removed
side      : LONG / SHORT
trend     : trend_alignment_count 0..3 from side-oriented STRUCT33 trend_state
            over TF = m15, h1, h4  -> aligned_tf = 1[side * trend_state_tf > 0]
slope     : slope_alignment_count 0..3, same rule on slope_atr
volatility: tertiles of atr_over_abs_price
structure : tertiles of log_structural_rr

Tertile thresholds are estimated on each fold's TRAINING rows only and then
applied to that fold's outer rows (§8: no global threshold leakage).
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas import run_decomposed_v2_research as R
from research.liquidity_oracle_atlas import walkforward_development_v1 as W
from research.liquidity_oracle_atlas import decomposed_models_v2 as M

TF_ORDER = ("m15", "h1", "h4")
TREND_COLS = tuple(f"{tf}_trend_state" for tf in TF_ORDER)
SLOPE_COLS = tuple(f"{tf}_slope_atr" for tf in TF_ORDER)

GROUP_DIMS = (
    "dimension", "group", "fold", "symbol", "side",
    "trend_alignment_count", "slope_alignment_count",
    "vol_tertile", "structural_rr_tertile",
)


# --------------------------------------------------------------------------- #
# Supporting columns                                                           #
# --------------------------------------------------------------------------- #
def load_support_frame() -> pd.DataFrame:
    """Dev-frame columns plus raw per-TF trend/slope from the frozen state.

    The OOF shards intentionally carry only predictions and labels; the atlas
    re-attaches the grouping inputs here.
    """
    dev = R.load_development_frame(columns=[
        "symbol", "decision_bar", "side", "horizon", "decision_time",
        "label_available_time", "sample_weight", "episode_return_atr",
        "atr_over_abs_price", "log_structural_rr",
    ])
    st = R.read_state(columns=["symbol", "bar_index"] + list(TREND_COLS)
                      + list(SLOPE_COLS))
    st = st.rename(columns={"bar_index": "decision_bar"})
    out = dev.merge(st, on=["symbol", "decision_bar"], how="left",
                    validate="many_to_one")
    missing = out[list(TREND_COLS)].isna().all(axis=1)
    if bool(missing.any()):
        raise R.StopV2Leakage(
            "STOP_V2_ATLAS_STATE_JOIN_INCOMPLETE "
            f"n_unmatched={int(missing.sum())}")
    return out


def side_sign(side) -> np.ndarray:
    return np.where(np.asarray(side) == "LONG", 1.0, -1.0)


def alignment_count(frame: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """count over TF of 1[side * value_tf > 0]."""
    s = side_sign(frame["side"].to_numpy())
    total = np.zeros(len(frame), dtype=np.int64)
    for c in cols:
        v = frame[c].to_numpy(float)
        aligned = (s * v) > 0
        total += np.where(np.isfinite(v), aligned.astype(np.int64), 0)
    return total


# --------------------------------------------------------------------------- #
# Tertiles estimated on TRAINING rows only                                     #
# --------------------------------------------------------------------------- #
def fold_tertile_thresholds(support: pd.DataFrame, plan, column: str):
    """Per-fold tertile cut points, computed on that fold's training rows."""
    out = {}
    for k in range(plan.n_outer):
        s = W.fold_split(support, plan, k)
        train = support[s.fit_core | s.es]
        vals = train[column].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 3:
            out[k] = (np.nan, np.nan)
        else:
            out[k] = tuple(np.percentile(vals, [33.3333, 66.6667]))
    return out


def apply_tertiles(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    v = np.asarray(values, float)
    out = np.full(v.shape, -1, dtype=np.int64)
    ok = np.isfinite(v)
    out[ok & (v <= lo)] = 1
    out[ok & (v > lo) & (v <= hi)] = 2
    out[ok & (v > hi)] = 3
    return out


# --------------------------------------------------------------------------- #
# §9 metrics                                                                   #
# --------------------------------------------------------------------------- #
def group_metrics(frame: pd.DataFrame) -> dict:
    """§9 metric block for ONE group of OOF rows."""
    y = frame["episode_return_atr"].to_numpy(float)
    w = frame["sample_weight"].to_numpy(float)
    p = frame["p_win"].to_numpy(float)
    mu_w = frame["mu_win"].to_numpy(float)
    mu_l = frame["mu_loss"].to_numpy(float)
    rr = frame["predicted_rr"].to_numpy(float)
    ev = frame["ev_c"].to_numpy(float)

    win = y > 0
    loss = ~win
    avg_win = M.weighted_mean(y[win], w[win])
    avg_loss = M.weighted_mean(np.abs(y[loss]), w[loss])

    return {
        "n": int(len(frame)),
        "weight_sum": float(np.nansum(w[np.isfinite(w)])),

        "weighted_brier": M.weighted_brier(win.astype(float), p, w),
        "weighted_logloss": M.weighted_logloss(win.astype(float), p, w),
        "actual_win_rate": M.weighted_mean(win.astype(float), w),
        "mean_predicted_pwin": M.weighted_mean(p, w),

        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "actual_rr": (avg_win / avg_loss
                      if np.isfinite(avg_win) and np.isfinite(avg_loss)
                      and avg_loss > 0 else float("nan")),
        "mean_predicted_rr": M.weighted_mean(
            rr[np.isfinite(rr)], w[np.isfinite(rr)]),
        "win_magnitude_mae": M.weighted_mae(y[win], mu_w[win], w[win]),
        "loss_magnitude_mae": M.weighted_mae(
            np.abs(y[loss]), mu_l[loss], w[loss]),

        "actual_mean_return": M.weighted_mean(y, w),
        "mean_predicted_ev": M.weighted_mean(ev, w),
        "weighted_ev_mse": M.weighted_ev_mse(y, ev, w),
        "weighted_ev_mae": M.weighted_ev_mae(y, ev, w),
    }


def _rows_for(dim: str, frame: pd.DataFrame, key) -> list:
    base = {"dimension": dim, "group": str(key)}
    metrics = group_metrics(frame)
    return [{**base, **metrics}]


# --------------------------------------------------------------------------- #
# Atlas                                                                        #
# --------------------------------------------------------------------------- #
def build_atlas(oof: pd.DataFrame, support: Optional[pd.DataFrame] = None,
                plan=None, label: str = "A0") -> pd.DataFrame:
    """Attach grouping keys, then emit one §9 metric row per group."""
    if support is None:
        support = load_support_frame()
    if plan is None:
        plan = R.build_plan_from_devframe()

    keys = ["symbol", "decision_bar", "side", "horizon"]
    df = oof.merge(
        support[keys + ["decision_time_x"] + list(TREND_COLS) + list(SLOPE_COLS)
                + ["atr_over_abs_price", "log_structural_rr"]]
        if "decision_time_x" in support.columns else
        support[keys + list(TREND_COLS) + list(SLOPE_COLS)
                + ["atr_over_abs_price", "log_structural_rr"]],
        on=keys, how="left", validate="many_to_one")

    df["trend_alignment_count"] = alignment_count(df, TREND_COLS)
    df["slope_alignment_count"] = alignment_count(df, SLOPE_COLS)

    vol_t = fold_tertile_thresholds(support, plan, "atr_over_abs_price")
    rr_t = fold_tertile_thresholds(support, plan, "log_structural_rr")
    vol_col = np.full(len(df), -1, dtype=np.int64)
    rr_col = np.full(len(df), -1, dtype=np.int64)
    for k in range(plan.n_outer):
        m = (df["fold"].to_numpy() == k)
        vol_col[m] = apply_tertiles(
            df.loc[m, "atr_over_abs_price"].to_numpy(float), *vol_t[k])
        rr_col[m] = apply_tertiles(
            df.loc[m, "log_structural_rr"].to_numpy(float), *rr_t[k])
    df["vol_tertile"] = vol_col
    df["structural_rr_tertile"] = rr_col

    rows = []
    rows += _rows_for("ALL", df, "all")
    for f in sorted(df["fold"].unique()):
        rows += _rows_for("time_fold", df[df["fold"] == f], int(f))
    for sym in sorted(df["symbol"].unique()):
        rows += _rows_for("symbol", df[df["symbol"] == sym], sym)
    for sd in ("LONG", "SHORT"):
        rows += _rows_for("side", df[df["side"] == sd], sd)
    for c in range(4):
        rows += _rows_for("trend_alignment",
                          df[df["trend_alignment_count"] == c], c)
    for c in range(4):
        rows += _rows_for("slope_alignment",
                          df[df["slope_alignment_count"] == c], c)
    for t in (1, 2, 3):
        rows += _rows_for("volatility_tertile", df[df["vol_tertile"] == t], t)
    for t in (1, 2, 3):
        rows += _rows_for("structural_rr_tertile",
                          df[df["structural_rr_tertile"] == t], t)

    out = pd.DataFrame(rows)
    out.insert(0, "object", label)
    return out


def run(label: str = "A0", oof: Optional[pd.DataFrame] = None,
        write: bool = True) -> pd.DataFrame:
    if oof is None:
        from research.liquidity_oracle_atlas.run_decomposed_v2_research import (
            _unit_paths,
        )
        shards = []
        plan = R.build_plan_from_devframe()
        arch_label = label
        for k in range(plan.n_outer):
            for h in ("td1", "td3", "td5"):
                p, _ = _unit_paths(arch_label, k, h)
                if os.path.exists(p):
                    shards.append(p)
        oof = pd.concat([R.read_parquet(p) for p in shards], ignore_index=True)
    atlas = build_atlas(oof, label=label)
    if write:
        R.write_csv_evidence(atlas, R.STABILITY_ATLAS_CSV)
    return atlas
