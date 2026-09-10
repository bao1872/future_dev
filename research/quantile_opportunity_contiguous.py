"""Contiguous-only out-of-sample Quantile opportunity state.

Reuses the FIXED model spec / feature set / fold authority from the
Quantile rebaseline experiment, but regenerates the state under the
strict H4 (15-minute) calendar-continuity constraint required for the
OB candidate universe.

The state is used ONLY as a future-volatility (opportunity) variable.
It provides NO directional prior: ``q50`` is stored as-is, never as a
direction feature.

Source-owner model / feature / fold authority (do NOT redefine):
    research/fit_quantile_v2_models.py   -> make_model
    research/run_quantile_rebaseline.py  -> FEATURE_SETS, make_folds
    research/build_pytdx_panel.py        -> aggregate_15m, build_features, build_targets
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.fit_quantile_v2_models import make_model
from research.run_quantile_rebaseline import (
    FEATURE_SETS,
    make_folds,
)
from research.build_pytdx_panel import (
    aggregate_15m,
    build_features,
    build_targets,
)

HORIZON = 4
FEATURE_SET = "F1_VOL"
MODEL = "gbr_quantile"
QUANTILES = (0.10, 0.50, 0.90)

FIFTEEN_NS = 15 * 60 * 1_000_000_000


def strict_target_contiguous(
    bars: pd.DataFrame,
    horizon: int = 4,
) -> np.ndarray:
    """Strict H4 (15-minute) calendar continuity for the target window.

    Every bar in base..base+horizon must be exactly 15 minutes apart.
    This is stricter than the old Q-Audit (which only checked the
    future segment internally).
    """
    t = (
        pd.to_datetime(bars["bar_start_time"])
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )
    n = len(t)
    out = np.zeros(n, dtype=bool)
    for i in range(n - horizon):
        seg = t[i : i + horizon + 1]
        out[i] = bool(np.all(np.diff(seg) == FIFTEEN_NS))
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 10:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    da = ra - ra.mean()
    db = rb - rb.mean()
    denom = np.sqrt((da ** 2).sum() * (db ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((da * db).sum() / denom)


def compute_quantile_diagnostics(
    *,
    out: pd.DataFrame,
    panel: pd.DataFrame,
    y: np.ndarray,
    contig: np.ndarray,
    bars: pd.DataFrame,
) -> dict:
    """Descriptive diagnostics only — never a STOP gate.

    Reports contiguous-target rate, OOS state count, crossing rate,
    80% interval coverage, width vs |return| Spearman, width vs future
    path-range Spearman, and TOP30/BOTTOM30 mean-|return| ratio.
    """
    n = len(panel)
    contig_rate = float(np.mean(contig)) if n else float("nan")
    oos_states = int(len(out))

    if oos_states:
        crossing_rate = float(out["quantile_crossed"].mean())
    else:
        crossing_rate = float("nan")

    actual = out["actual_return"].to_numpy(float)
    lower = out["q10"].to_numpy(float)
    upper = out["q90"].to_numpy(float)
    valid = np.isfinite(actual) & np.isfinite(lower) & np.isfinite(upper)
    if valid.any():
        covered = (actual[valid] >= lower[valid]) & (
            actual[valid] <= upper[valid]
        )
        coverage_80 = float(np.mean(covered))
    else:
        coverage_80 = float("nan")

    width = out["width"].to_numpy(float)
    absret = np.abs(actual)
    if np.isfinite(width).any() and np.isfinite(absret).any():
        m = np.isfinite(width) & np.isfinite(absret)
        width_absret = _spearman(width[m], absret[m])
    else:
        width_absret = float("nan")

    # Future path range: forward horizon 15m high-low range.
    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    nb = len(bars)
    path_range = np.full(nb, np.nan)
    for i in range(nb - HORIZON):
        path_range[i] = (
            np.max(high[i + 1 : i + 1 + HORIZON])
            - np.min(low[i + 1 : i + 1 + HORIZON])
        )
    pr = path_range[out["panel_index"].to_numpy(int)]
    if np.isfinite(width).any() and np.isfinite(pr).any():
        m2 = np.isfinite(width) & np.isfinite(pr)
        width_pathrange = _spearman(width[m2], pr[m2])
    else:
        width_pathrange = float("nan")

    pct = out["width_percentile_train"].to_numpy(float)
    top30 = np.isfinite(pct) & (pct >= 0.70)
    bottom30 = np.isfinite(pct) & (pct <= 0.30)
    if top30.any() and bottom30.any():
        top_mean = np.mean(np.abs(actual[top30]))
        bot_mean = np.mean(np.abs(actual[bottom30]))
        ratio = (
            float(top_mean / bot_mean)
            if bot_mean != 0
            else float("nan")
        )
    else:
        ratio = float("nan")

    return {
        "contiguous_target_rate": contig_rate,
        "oos_states": oos_states,
        "crossing_rate": crossing_rate,
        "interval_coverage_80": coverage_80,
        "width_absret_spearman": width_absret,
        "width_pathrange_spearman": width_pathrange,
        "top30_bottom30_absret_ratio": ratio,
    }


def quantile_state_contiguous_oos(
    five: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    bars = aggregate_15m(five)

    features = build_features(bars, five)
    targets = build_targets(bars, HORIZON)

    panel = pd.concat(
        [
            bars[["bar_start_time", "bar_end_time"]].rename(
                columns={
                    "bar_start_time": "base_time",
                    "bar_end_time": "decision_time",
                }
            ),
            features,
        ],
        axis=1,
    )
    for name, values in targets.items():
        panel[name] = values

    feature_cols = FEATURE_SETS[FEATURE_SET]
    missing = [c for c in feature_cols if c not in panel.columns]
    if missing:
        raise RuntimeError(
            f"Quantile feature columns missing: {missing}"
        )

    X = panel[feature_cols].apply(pd.to_numeric, errors="coerce")
    target_col = f"target_raw_return_h{HORIZON}"
    if target_col not in panel.columns:
        raise RuntimeError(f"Quantile target missing: {target_col}")

    y = panel[target_col].to_numpy(float)

    contig = strict_target_contiguous(bars, HORIZON)

    feature_ok = np.isfinite(X.to_numpy(float)).all(axis=1)
    target_ok = np.isfinite(y)
    trainable = feature_ok & target_ok & contig

    n = len(panel)
    folds = make_folds(n, horizon=HORIZON)

    qout = {q: np.full(n, np.nan) for q in QUANTILES}
    percentile = np.full(n, np.nan)
    fold_out = np.full(n, -1, dtype=int)

    for fold in folds:
        tr_all = np.arange(
            fold["train_start"], fold["train_end_exclusive"]
        )
        te_all = np.arange(
            fold["test_start"], fold["test_end_exclusive"]
        )

        tr = tr_all[trainable[tr_all]]
        te = te_all[trainable[te_all]]

        if len(tr) < 600:
            raise RuntimeError(
                "contiguous Quantile train set too small"
            )
        if len(te) == 0:
            continue

        train_pred: dict[float, np.ndarray] = {}
        test_pred: dict[float, np.ndarray] = {}

        for q in QUANTILES:
            model = make_model(MODEL, q)
            model.fit(X.iloc[tr], y[tr])
            train_pred[q] = model.predict(X.iloc[tr])
            test_pred[q] = model.predict(X.iloc[te])
            qout[q][te] = test_pred[q]

        # Percentile is computed from the CURRENT fold's TRAIN width
        # distribution only — never a global rank over the OOS set.
        train_width = train_pred[0.90] - train_pred[0.10]
        test_width = test_pred[0.90] - test_pred[0.10]
        good_train_width = (
            np.isfinite(train_width) & (train_width >= 0.0)
        )
        ref = np.sort(train_width[good_train_width])
        if len(ref) < 100:
            raise RuntimeError(
                "quantile width reference too small"
            )
        percentile[te] = (
            np.searchsorted(ref, test_width, side="right") / len(ref)
        )
        fold_out[te] = int(fold["fold"])

    out = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(panel["decision_time"]),
            "fold": fold_out,
            "q10": qout[0.10],
            "q50": qout[0.50],
            "q90": qout[0.90],
            "width": qout[0.90] - qout[0.10],
            "width_percentile_train": percentile,
            "actual_return": y,
            "panel_index": np.arange(n),
        }
    )

    out["quantile_crossed"] = (out["q10"] > out["q50"]) | (
        out["q50"] > out["q90"]
    )
    out["top30_train"] = out["width_percentile_train"] >= 0.70
    out["valid_until"] = out["decision_time"] + pd.Timedelta(minutes=15)

    out = out[out["fold"] >= 0].reset_index(drop=True)

    diagnostics = compute_quantile_diagnostics(
        out=out,
        panel=panel,
        y=y,
        contig=contig,
        bars=bars,
    )

    return out, diagnostics


def attach_quantile_state(
    candidates: pd.DataFrame,
    state: pd.DataFrame,
) -> pd.DataFrame:
    """Attach a 15m Quantile state to each candidate under strict PIT.

    A state at ``decision_time`` may only serve a trigger in
    ``[decision_time, decision_time + 15min)``. No backward/forward
    fill beyond that window.
    """
    s = state.sort_values("decision_time").reset_index(drop=True)
    state_t = (
        s["decision_time"]
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )

    cand = candidates.copy()
    cand_t = (
        pd.to_datetime(cand["touch_time"])
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )
    pos = np.searchsorted(state_t, cand_t, side="right") - 1

    cols = [
        "quant_state_decision_time",
        "quant_state_age_minutes",
        "quant_q10",
        "quant_q50",
        "quant_q90",
        "quant_width",
        "quant_width_percentile_train",
        "quant_top30_train",
        "quant_crossed",
        "quant_fold",
    ]
    # pandas 3.0 起禁止把字符串/bool 写入 float64 列（LossySetitemError），
    # 旧版本则会把整列 upcast 成 object。DEV4 已生成的结果里这三个列正是
    # object（str / bool），因此这里直接以 object 初始化，保证语义与 dtype
    # 与既有 DEV4 产物完全一致。
    _OBJECT_COLS = {
        "quant_state_decision_time",   # 时间戳字符串
        "quant_top30_train",           # bool
        "quant_crossed",               # bool
    }
    for c in cols:
        if c in _OBJECT_COLS:
            cand[c] = pd.Series([None] * len(cand), dtype="object",
                                index=cand.index)
        else:
            cand[c] = np.nan

    locs = {c: cand.columns.get_loc(c) for c in cols}

    for i in range(len(cand)):
        p = int(pos[i])
        if p < 0:
            continue
        decision = pd.Timestamp(s.iloc[p]["decision_time"])
        valid_until = pd.Timestamp(s.iloc[p]["valid_until"])
        trigger = pd.Timestamp(cand.iloc[i]["touch_time"])

        if not (decision <= trigger < valid_until):
            continue

        cand.iloc[i, locs["quant_state_decision_time"]] = str(decision)
        cand.iloc[i, locs["quant_state_age_minutes"]] = (
            trigger - decision
        ).total_seconds() / 60.0
        cand.iloc[i, locs["quant_q10"]] = float(s.iloc[p]["q10"])
        cand.iloc[i, locs["quant_q50"]] = float(s.iloc[p]["q50"])
        cand.iloc[i, locs["quant_q90"]] = float(s.iloc[p]["q90"])
        cand.iloc[i, locs["quant_width"]] = float(s.iloc[p]["width"])
        cand.iloc[i, locs["quant_width_percentile_train"]] = float(
            s.iloc[p]["width_percentile_train"]
        )
        cand.iloc[i, locs["quant_top30_train"]] = bool(
            s.iloc[p]["top30_train"]
        )
        cand.iloc[i, locs["quant_crossed"]] = bool(
            s.iloc[p]["quantile_crossed"]
        )
        cand.iloc[i, locs["quant_fold"]] = int(s.iloc[p]["fold"])

    return cand
