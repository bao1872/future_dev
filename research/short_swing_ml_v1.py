#!/usr/bin/env python3
"""
Short-Swing Futures ML Research V1

Goal
----
Classical/statistical ML study for short-swing futures prediction.

- Decision cycle: 15m
- Horizons: 2/4/8/16 bars ~= 30m/1h/2h/4h
- Entry reference: next observed 15m bar OPEN after the feature bar closes
- Features: stationary price/volatility/volume/OI features only
- Multi-TF: 15m + last causally available 1h + last causally available 4h
- EXCLUDES SMC / BOS / CHoCH / SQZMOM as model inputs
- DP Oracle is post-hoc reference only, never a feature
- Expanding OOS validation with horizon-sized purge
- Fixed model parameters; no hyperparameter search in V1
- No backtest / position sizing / trading-rule optimization
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.base import clone
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required.\n"
        "Install only in the current research environment:\n"
        "  python -m pip install scikit-learn"
    ) from exc


HORIZONS_DEFAULT = (2, 4, 8, 16)
FEATURE_SETS_DEFAULT = ("stats_15m", "stats_15m_1h", "stats_15m_1h_4h")
MODELS_DEFAULT = ("ridge", "elastic", "rf", "hgb")
RANDOM_STATE = 42

STAT_SUFFIXES = (
    "log_ret_1", "log_ret_3", "log_ret_5", "log_ret_10", "log_ret_20",
    "gap_log_ret_1",
    "range_pct", "body_pct", "upper_wick_pct", "lower_wick_pct",
    "bar_close_location",
    "atr14_pct", "realized_vol_20",
    "close_location_20", "close_location_50",
    "close_vs_sma_20", "close_vs_sma_50",
    "volume_ratio_20",
    "oi_log_change_1", "oi_log_change_5",
)
FORBIDDEN = ("smc_", "momentum", "sqz", "oracle", "target_")


def repo_root() -> Path:
    starts = [Path.cwd().resolve()]
    try:
        starts.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    for start in starts:
        for p in [start, *start.parents]:
            if (
                p / "research/exports/state_research/research_panel_15m.csv"
            ).is_file() and (p / "silver_main_data/silver_main_15m.csv").is_file():
                return p
    raise FileNotFoundError(
        "Run inside future_dev and ensure state_research export + 15m raw CSV exist."
    )


def load_data(root: Path):
    rp = root / "research/exports/state_research/research_panel_15m.csv"
    mp = root / "research/exports/state_research/manifest.json"
    bp = root / "silver_main_data/silver_main_15m.csv"

    df = pd.read_csv(
        rp, parse_dates=["decision_time", "base_bar_time"], low_memory=False
    )
    bars = pd.read_csv(bp, parse_dates=["datetime"], low_memory=False)
    bars = bars.sort_values("datetime_ns").reset_index(drop=True)
    manifest = json.loads(mp.read_text(encoding="utf-8"))

    exp_df = int(manifest["research_panel_rows"]["15m"])
    exp_bars = int(manifest["market_data"]["15m"]["rows"])
    if len(df) != exp_df:
        raise RuntimeError(f"research rows {len(df)} != manifest {exp_df}")
    if len(bars) != exp_bars:
        raise RuntimeError(f"raw 15m rows {len(bars)} != manifest {exp_bars}")

    required = {
        "decision_time", "base_bar_time", "base_bar_index",
        "feat_15m__realized_vol_20",
    }
    miss = required - set(df.columns)
    if miss:
        raise RuntimeError(f"research panel missing: {sorted(miss)}")
    return df, bars, manifest


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    t = pd.to_datetime(out["decision_time"])
    minute = t.dt.hour * 60 + t.dt.minute
    dow = t.dt.dayofweek

    out["feat_time__tod_sin"] = np.sin(2*np.pi*minute/(24*60))
    out["feat_time__tod_cos"] = np.cos(2*np.pi*minute/(24*60))
    out["feat_time__dow_sin"] = np.sin(2*np.pi*dow/7)
    out["feat_time__dow_cos"] = np.cos(2*np.pi*dow/7)

    gap_min = (
        pd.to_datetime(out["decision_time"]) - pd.to_datetime(out["base_bar_time"])
    ).dt.total_seconds() / 60.0
    out["feat_time__decision_gap_log1p_minutes"] = np.log1p(
        gap_min.clip(lower=0)
    )
    return out


def feature_columns(df: pd.DataFrame, tf: str) -> list[str]:
    cols = [f"feat_{tf}__{s}" for s in STAT_SUFFIXES]
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise RuntimeError(f"{tf} missing expected statistical features: {miss}")
    return cols


def make_feature_sets(df: pd.DataFrame, include_time: bool):
    work = add_time_features(df) if include_time else df.copy()
    c15 = feature_columns(work, "15m")
    c1h = feature_columns(work, "1h")
    c4h = feature_columns(work, "4h")
    ct = [c for c in work.columns if c.startswith("feat_time__")] if include_time else []

    sets = {
        "stats_15m": c15 + ct,
        "stats_15m_1h": c15 + c1h + ct,
        "stats_15m_1h_4h": c15 + c1h + c4h + ct,
    }
    for name, cols in sets.items():
        bad = [c for c in cols if any(x in c.lower() for x in FORBIDDEN)]
        if bad:
            raise RuntimeError(f"{name}: forbidden/leaky features selected: {bad}")
    return work, sets


def make_targets(df: pd.DataFrame, bars: pd.DataFrame, horizons: tuple[int, ...]):
    out = df.copy()
    base = out["base_bar_index"].astype(int).to_numpy()
    entry_idx = base + 1

    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    c = bars["close"].to_numpy(float)

    entry = np.full(len(out), np.nan)
    ok_entry = entry_idx < len(bars)
    entry[ok_entry] = o[entry_idx[ok_entry]]

    out["target_entry_bar_index"] = entry_idx
    out["target_entry_price"] = entry

    rv = pd.to_numeric(
        out["feat_15m__realized_vol_20"], errors="coerce"
    ).to_numpy(float)

    for H in horizons:
        exit_idx = base + H
        valid = ok_entry & (exit_idx < len(bars)) & np.isfinite(entry) & (entry > 0)

        raw_ret = np.full(len(out), np.nan)
        norm_ret = np.full(len(out), np.nan)
        lmfe = np.full(len(out), np.nan)
        lmae = np.full(len(out), np.nan)
        smfe = np.full(len(out), np.nan)
        smae = np.full(len(out), np.nan)

        for row in np.flatnonzero(valid):
            i0 = int(entry_idx[row])
            i1 = int(exit_idx[row])
            e = float(entry[row])
            raw_ret[row] = math.log(c[i1] / e)

            wh = float(np.max(h[i0:i1+1]))
            wl = float(np.min(l[i0:i1+1]))
            lmfe[row] = math.log(wh / e)
            lmae[row] = math.log(wl / e)
            smfe[row] = math.log(e / wl)
            smae[row] = math.log(e / wh)

        scale = rv * math.sqrt(H)
        norm_ok = valid & np.isfinite(scale) & (scale > 1e-8)
        norm_ret[norm_ok] = raw_ret[norm_ok] / scale[norm_ok]

        out[f"target_raw_log_return_h{H}"] = raw_ret
        out[f"target_norm_return_h{H}"] = norm_ret
        out[f"target_long_mfe_h{H}"] = lmfe
        out[f"target_long_mae_h{H}"] = lmae
        out[f"target_short_mfe_h{H}"] = smfe
        out[f"target_short_mae_h{H}"] = smae
        out[f"target_exit_bar_index_h{H}"] = exit_idx
    return out


def models(names: tuple[str, ...], with_lgbm: bool):
    out = {}
    if "ridge" in names:
        out["ridge"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ])
    if "elastic" in names:
        out["elastic"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(
                alpha=0.001, l1_ratio=0.10, max_iter=20000,
                tol=1e-5, random_state=RANDOM_STATE
            )),
        ])
    if "rf" in names:
        out["rf"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=250, max_depth=7, min_samples_leaf=25,
                max_features=0.70, n_jobs=-1, random_state=RANDOM_STATE
            )),
        ])
    if "hgb" in names:
        out["hgb"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                min_samples_leaf=30, l2_regularization=1.0,
                random_state=RANDOM_STATE
            )),
        ])

    if with_lgbm or "lgbm" in names:
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise SystemExit(
                "LightGBM requested but missing. Install only if intentionally testing it:\n"
                "  python -m pip install lightgbm"
            ) from exc
        out["lgbm"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMRegressor(
                n_estimators=300, learning_rate=0.03, num_leaves=15,
                min_child_samples=30, subsample=0.80, colsample_bytree=0.80,
                reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1
            )),
        ])

    unknown = set(names) - {"ridge", "elastic", "rf", "hgb", "lgbm"}
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")
    return out


def folds(n: int, H: int, min_train: int, test_rows: int, step_rows: int):
    result = []
    train_end = min_train
    k = 0
    while True:
        test_start = train_end + H
        test_end = min(test_start + test_rows, n)
        if test_start >= n:
            break
        if test_end - test_start < max(200, test_rows // 3):
            break
        result.append((k, train_end, test_start, test_end))
        k += 1
        train_end += step_rows
    if len(result) < 3:
        raise RuntimeError(f"only {len(result)} folds generated")
    return result


def corr(a: pd.Series, b: pd.Series, method: str) -> float:
    m = a.notna() & b.notna()
    if m.sum() < 3 or a[m].nunique() < 2 or b[m].nunique() < 2:
        return float("nan")
    return float(a[m].corr(b[m], method=method))


def auc(raw_return: pd.Series, score: pd.Series) -> float:
    m = raw_return.notna() & score.notna()
    y = (raw_return[m] > 0).astype(int)
    if len(y) < 20 or y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, score[m]))


def deciles(pred_df: pd.DataFrame, H: int, fs: str, model: str):
    w = pred_df.dropna(subset=["prediction", "realized_raw_return"]).copy()
    if len(w) < 200:
        return pd.DataFrame()
    w["decile"] = pd.qcut(
        w["prediction"].rank(method="first"),
        10, labels=False, duplicates="drop"
    ) + 1
    g = (
        w.groupby("decile", observed=True)
        .agg(
            rows=("realized_raw_return", "size"),
            mean_raw_return=("realized_raw_return", "mean"),
            median_raw_return=("realized_raw_return", "median"),
            positive_rate=("realized_raw_return", lambda s: float((s > 0).mean())),
            mean_prediction=("prediction", "mean"),
        )
        .reset_index()
    )
    g.insert(0, "model", model)
    g.insert(0, "feature_set", fs)
    g.insert(0, "horizon", H)
    return g


def importance_rows(pipe, feature_names, H, fs, model_name, fold_id):
    if not isinstance(pipe, Pipeline) or "model" not in pipe.named_steps:
        return pd.DataFrame()
    est = pipe.named_steps["model"]
    try:
        names = list(pipe.named_steps["imputer"].get_feature_names_out(feature_names))
    except Exception:
        names = feature_names

    vals = None
    kind = None
    if hasattr(est, "coef_"):
        vals = np.asarray(est.coef_, float).reshape(-1)
        kind = "coefficient"
    elif hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_, float).reshape(-1)
        kind = "importance"
    if vals is None or len(vals) != len(names):
        return pd.DataFrame()

    return pd.DataFrame({
        "horizon": H, "feature_set": fs, "model": model_name, "fold": fold_id,
        "feature": names, "importance_kind": kind, "value": vals,
        "abs_value": np.abs(vals),
    })


def run(df, fsets, args):
    model_map = models(args.models, args.with_lightgbm)
    metrics_rows, fold_rows = [], []
    pred_parts, dec_parts, imp_parts = [], [], []

    for H in args.horizons:
        ycol = (
            f"target_norm_return_h{H}"
            if args.target == "normalized"
            else f"target_raw_log_return_h{H}"
        )
        rawcol = f"target_raw_log_return_h{H}"
        usable = df[ycol].notna() & df[rawcol].notna()
        frame = df.loc[usable].reset_index(drop=True).copy()
        cv = folds(len(frame), H, args.min_train_rows, args.test_rows, args.step_rows)

        for fs in args.feature_sets:
            cols = fsets[fs]
            # Feature membership is fixed a priori. Do not inspect future OOS
            # distribution to decide which columns survive.
            good_cols = list(cols)
            X = frame[good_cols].apply(pd.to_numeric, errors="coerce")
            y = pd.to_numeric(frame[ycol], errors="coerce")
            raw_y = pd.to_numeric(frame[rawcol], errors="coerce")

            for model_name, template in model_map.items():
                pred = np.full(len(frame), np.nan)
                bench = np.full(len(frame), np.nan)
                fold_arr = np.full(len(frame), -1, dtype=int)

                for fold_id, train_end, test_start, test_end in cv:
                    if test_start - train_end < H:
                        raise RuntimeError("purge invariant failed")

                    train_last_base = int(
                        frame["base_bar_index"].iloc[train_end - 1]
                    )
                    test_first_base = int(
                        frame["base_bar_index"].iloc[test_start]
                    )
                    if test_first_base - train_last_base <= H:
                        raise RuntimeError(
                            "purge invariant failed in original 15m bar space"
                        )

                    tr = np.arange(0, train_end)
                    te = np.arange(test_start, test_end)

                    train_ok = y.iloc[tr].notna().to_numpy()
                    test_ok = y.iloc[te].notna().to_numpy()
                    tr2 = tr[train_ok]
                    te2 = te[test_ok]

                    m = clone(template)
                    m.fit(X.iloc[tr2], y.iloc[tr2])
                    p = np.asarray(m.predict(X.iloc[te2]), float)

                    pred[te2] = p
                    hist_mean = float(y.iloc[tr2].mean())
                    bench[te2] = hist_mean
                    fold_arr[te2] = fold_id

                    fy = y.iloc[te2]
                    fp = pd.Series(p, index=fy.index)
                    fb = pd.Series(hist_mean, index=fy.index)
                    fr = raw_y.iloc[te2]

                    sse = float(np.sum((fy-fp)**2))
                    bsse = float(np.sum((fy-fb)**2))
                    fold_rows.append({
                        "horizon": H, "feature_set": fs, "model": model_name,
                        "fold": fold_id, "train_rows": len(tr2), "test_rows": len(te2),
                        "purge_rows": H,
                        "train_end_time": str(frame["decision_time"].iloc[train_end-1]),
                        "test_start_time": str(frame["decision_time"].iloc[test_start]),
                        "test_end_time": str(frame["decision_time"].iloc[test_end-1]),
                        "oos_r2_vs_hist_mean": 1-sse/bsse if bsse > 0 else np.nan,
                        "mae": mean_absolute_error(fy, fp),
                        "rmse": math.sqrt(mean_squared_error(fy, fp)),
                        "pearson_ic": corr(fp.reset_index(drop=True), fy.reset_index(drop=True), "pearson"),
                        "spearman_ic": corr(fp.reset_index(drop=True), fy.reset_index(drop=True), "spearman"),
                        "sign_accuracy_raw": float(((p > 0) == (fr.to_numpy() > 0)).mean()),
                    })
                    imp = importance_rows(m, good_cols, H, fs, model_name, fold_id)
                    if not imp.empty:
                        imp_parts.append(imp)

                mask = np.isfinite(pred) & np.isfinite(bench)
                yo = y[mask]
                ro = raw_y[mask]
                po = pd.Series(pred[mask], index=yo.index)
                bo = pd.Series(bench[mask], index=yo.index)

                sse = float(np.sum((yo-po)**2))
                bsse = float(np.sum((yo-bo)**2))
                oos_r2 = 1-sse/bsse if bsse > 0 else np.nan

                pdf = pd.DataFrame({
                    "decision_time": frame.loc[mask, "decision_time"].to_numpy(),
                    "base_bar_index": frame.loc[mask, "base_bar_index"].to_numpy(),
                    "horizon": H, "feature_set": fs, "model": model_name,
                    "fold": fold_arr[mask], "prediction": pred[mask],
                    "benchmark_prediction": bench[mask],
                    "realized_target": yo.to_numpy(),
                    "realized_raw_return": ro.to_numpy(),
                    "long_mfe": frame.loc[mask, f"target_long_mfe_h{H}"].to_numpy(),
                    "long_mae": frame.loc[mask, f"target_long_mae_h{H}"].to_numpy(),
                    "short_mfe": frame.loc[mask, f"target_short_mfe_h{H}"].to_numpy(),
                    "short_mae": frame.loc[mask, f"target_short_mae_h{H}"].to_numpy(),
                })

                oc = "target_15m__oracle_consensus"
                ou = "target_15m__oracle_unanimous_direction"
                pdf["oracle_consensus"] = (
                    frame.loc[mask, oc].to_numpy() if oc in frame.columns else np.nan
                )
                pdf["oracle_unanimous_direction"] = (
                    frame.loc[mask, ou].to_numpy() if ou in frame.columns else np.nan
                )

                d = deciles(pdf, H, fs, model_name)
                if not d.empty:
                    dec_parts.append(d)

                spread = mono = top = bottom = top_hit = bottom_hit = np.nan
                if len(d) >= 10:
                    ds = d.sort_values("decile")
                    bottom = float(ds.iloc[0]["mean_raw_return"])
                    top = float(ds.iloc[-1]["mean_raw_return"])
                    spread = top-bottom
                    mono = corr(
                        ds["decile"].astype(float),
                        ds["mean_raw_return"].astype(float),
                        "spearman",
                    )
                    top_hit = float(ds.iloc[-1]["positive_rate"])
                    bottom_hit = 1-float(ds.iloc[0]["positive_rate"])

                strong = (
                    pdf["oracle_consensus"].notna()
                    & (pdf["oracle_consensus"].abs() >= 0.60)
                    & (pdf["prediction"] != 0)
                )
                oracle_agree = (
                    float((
                        np.sign(pdf.loc[strong, "prediction"])
                        == np.sign(pdf.loc[strong, "oracle_consensus"])
                    ).mean())
                    if strong.sum() >= 20 else np.nan
                )

                row = {
                    "horizon": H, "approx_minutes": H*15, "target_kind": args.target,
                    "feature_set": fs, "feature_count": len(good_cols),
                    "model": model_name, "fold_count": len(cv),
                    "oos_rows": int(mask.sum()),
                    "oos_start": str(pdf["decision_time"].min()),
                    "oos_end": str(pdf["decision_time"].max()),
                    "oos_r2_vs_hist_mean": oos_r2,
                    "mae": mean_absolute_error(yo, po),
                    "rmse": math.sqrt(mean_squared_error(yo, po)),
                    "pearson_ic": corr(po.reset_index(drop=True), yo.reset_index(drop=True), "pearson"),
                    "spearman_ic": corr(po.reset_index(drop=True), yo.reset_index(drop=True), "spearman"),
                    "sign_accuracy_raw": float(((po.to_numpy()>0)==(ro.to_numpy()>0)).mean()),
                    "direction_auc_raw": auc(ro.reset_index(drop=True), po.reset_index(drop=True)),
                    "top_decile_mean_raw_return": top,
                    "bottom_decile_mean_raw_return": bottom,
                    "top_minus_bottom_decile_raw": spread,
                    "decile_monotonicity_spearman": mono,
                    "top_decile_positive_rate": top_hit,
                    "bottom_decile_negative_rate": bottom_hit,
                    "strong_oracle_direction_agreement": oracle_agree,
                    "prediction_std": float(np.std(po)),
                }
                metrics_rows.append(row)
                pred_parts.append(pdf)

                print(
                    f"[DONE] h={H:<2} {fs:<18} {model_name:<8} "
                    f"R2={oos_r2:+.4f} IC={row['spearman_ic']:+.4f} "
                    f"D10-D1={spread:+.6f}"
                )

    metrics = pd.DataFrame(metrics_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    preds = pd.concat(pred_parts, ignore_index=True)
    decs = pd.concat(dec_parts, ignore_index=True) if dec_parts else pd.DataFrame()
    imps = pd.concat(imp_parts, ignore_index=True) if imp_parts else pd.DataFrame()

    if not imps.empty:
        imps = (
            imps.groupby(
                ["horizon", "feature_set", "model", "feature", "importance_kind"],
                observed=True,
            )
            .agg(
                fold_count=("value", "size"),
                mean_value=("value", "mean"),
                median_value=("value", "median"),
                mean_abs_value=("abs_value", "mean"),
                positive_share=("value", lambda s: float((s>0).mean())),
            )
            .reset_index()
        )
        imps["sign_consistency"] = np.maximum(
            imps["positive_share"], 1-imps["positive_share"]
        )
        imps = imps.sort_values(
            ["horizon", "feature_set", "model", "mean_abs_value"],
            ascending=[True, True, True, False],
        )

    return metrics, fold_metrics, preds, decs, imps


def validate(metrics, fold_metrics, preds, args):
    expected = len(args.horizons) * len(args.feature_sets) * len(args.models)
    if len(metrics) != expected:
        raise RuntimeError(f"metrics rows {len(metrics)} != expected {expected}")
    if metrics["oos_rows"].min() < 500:
        raise RuntimeError("too few OOS predictions")
    if (fold_metrics["purge_rows"] < fold_metrics["horizon"]).any():
        raise RuntimeError("purge validation failed")
    key = ["decision_time", "horizon", "feature_set", "model"]
    if preds.duplicated(key).any():
        raise RuntimeError("duplicate OOS predictions")
    if np.isinf(metrics.select_dtypes(include=[np.number]).to_numpy(float)).any():
        raise RuntimeError("metrics contain inf")


def summary_md(metrics: pd.DataFrame, args) -> str:
    lines = [
        "# Short-Swing Futures ML Research V1",
        "",
        "- Decision cycle: 15m",
        f"- Primary target: {args.target} future return",
        f"- Horizons: {list(args.horizons)}",
        "- SMC / BOS / CHoCH / SQZMOM as features: **NO**",
        "- Oracle as model feature: **NO**",
        "- Validation: expanding OOS + horizon-sized purge",
        "- Hyperparameter search: **NO**",
        "- Backtest: **NO**",
        "",
        "## Best factual result by horizon",
        "",
        "| Horizon | OOS R² | Model | Feature set | Spearman IC | AUC | D10-D1 raw |",
        "|---:|---:|---|---|---:|---:|---:|",
    ]
    for H in sorted(metrics["horizon"].unique()):
        best = (
            metrics[metrics["horizon"]==H]
            .sort_values(["oos_r2_vs_hist_mean", "spearman_ic"], ascending=False)
            .iloc[0]
        )
        lines.append(
            f"| {H} ({H*15}m) | {best['oos_r2_vs_hist_mean']:.4f} | "
            f"{best['model']} | {best['feature_set']} | "
            f"{best['spearman_ic']:.4f} | {best['direction_auc_raw']:.4f} | "
            f"{best['top_minus_bottom_decile_raw']:.6f} |"
        )
    lines += [
        "",
        "## Guardrails",
        "",
        "- Positive OOS R² is stronger evidence than in-sample fit.",
        "- Decile spread must be read with IC and monotonicity.",
        "- One good model/horizon is not enough to claim a tradable strategy.",
        "- Overlapping horizons mean this run makes no naive t-stat significance claim.",
        "- Next step should be driven by cross-fold / cross-horizon stability.",
        "",
    ]
    return "\n".join(lines)


def parse_tuple_int(s: str):
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def parse_tuple_str(s: str):
    return tuple(x.strip() for x in s.split(",") if x.strip())


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["normalized", "raw"], default="normalized")
    p.add_argument("--horizons", default="2,4,8,16")
    p.add_argument(
        "--feature-sets",
        default="stats_15m,stats_15m_1h,stats_15m_1h_4h",
    )
    p.add_argument("--models", default="ridge,elastic,rf,hgb")
    p.add_argument("--min-train-rows", type=int, default=3500)
    p.add_argument("--test-rows", type=int, default=750)
    p.add_argument("--step-rows", type=int, default=750)
    p.add_argument("--no-time-features", action="store_true")
    p.add_argument("--with-lightgbm", action="store_true")
    p.add_argument("--output", default="research/exports/ml_research_v1")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()

    a.horizons = parse_tuple_int(a.horizons)
    a.feature_sets = parse_tuple_str(a.feature_sets)
    a.models = parse_tuple_str(a.models)

    if a.quick:
        a.horizons = (8,)
        a.feature_sets = ("stats_15m_1h_4h",)
        a.models = ("ridge", "hgb")
        a.test_rows = 600
        a.step_rows = 900

    if a.with_lightgbm and "lgbm" not in a.models:
        a.models = (*a.models, "lgbm")

    unknown_fs = set(a.feature_sets) - set(FEATURE_SETS_DEFAULT)
    if unknown_fs:
        p.error(f"unknown feature sets: {sorted(unknown_fs)}")
    return a


def main():
    args = cli()
    root = repo_root()
    out = root / args.output

    if out.exists():
        if not args.overwrite:
            raise SystemExit(
                f"{out} already exists. Use --overwrite only for intentional rerun."
            )
        shutil.rmtree(out)
    out.mkdir(parents=True)

    df, bars, manifest = load_data(root)
    df, fsets = make_feature_sets(df, include_time=not args.no_time_features)
    df = make_targets(df, bars, args.horizons)

    print("="*80)
    print("SHORT-SWING FUTURES ML RESEARCH V1")
    print(f"repo={root}")
    print(f"rows={len(df)} target={args.target}")
    print(f"horizons={args.horizons}")
    print(f"feature_sets={args.feature_sets}")
    print(f"models={args.models}")
    print("SMC/SQZMOM features = NO")
    print("Oracle feature = NO")
    print("="*80)

    metrics, fold_metrics, preds, decs, imps = run(df, fsets, args)
    validate(metrics, fold_metrics, preds, args)

    metrics.to_csv(out/"metrics.csv", index=False)
    fold_metrics.to_csv(out/"fold_metrics.csv", index=False)
    preds.to_csv(out/"oos_predictions.csv", index=False)
    decs.to_csv(out/"deciles.csv", index=False)
    imps.to_csv(out/"feature_importance.csv", index=False)

    config = {
        "source_instrument": manifest.get("instrument"),
        "source_research_panel_rows": manifest["research_panel_rows"]["15m"],
        "target": args.target,
        "horizons": args.horizons,
        "feature_sets": args.feature_sets,
        "models": args.models,
        "time_features": not args.no_time_features,
        "stat_feature_suffixes": list(STAT_SUFFIXES),
        "selected_features_by_set": {
            name: fsets[name] for name in args.feature_sets
        },
        "target_definition": {
            "decision": "next observed 15m bar start",
            "entry": "next observed 15m bar open",
            "exit": "close at base_bar_index + horizon",
            "normalized_return": (
                "raw log return / (causal realized_vol_20 * sqrt(horizon))"
            ),
            "MFE_MAE": "future high/low excursion from next-bar open",
        },
        "leakage_guards": {
            "random_split": False,
            "purge": "horizon-sized",
            "SMC_as_feature": False,
            "SQZMOM_as_feature": False,
            "Oracle_as_feature": False,
        },
        "fixed_model_params": {
            "ridge": {"alpha": 10.0},
            "elastic": {"alpha": 0.001, "l1_ratio": 0.10},
            "rf": {
                "n_estimators": 250, "max_depth": 7,
                "min_samples_leaf": 25, "max_features": 0.70,
            },
            "hgb": {
                "max_iter": 200, "learning_rate": 0.05,
                "max_leaf_nodes": 15, "min_samples_leaf": 30,
                "l2_regularization": 1.0,
            },
        },
    }
    (out/"config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out/"run_summary.md").write_text(summary_md(metrics, args), encoding="utf-8")

    print()
    print("FACTUAL BEST BY HORIZON")
    for H in sorted(metrics["horizon"].unique()):
        best = (
            metrics[metrics["horizon"]==H]
            .sort_values(["oos_r2_vs_hist_mean", "spearman_ic"], ascending=False)
            .iloc[0]
        )
        print(
            f"h={H:>2} {H*15:>3}m | {best['model']:<8} | "
            f"{best['feature_set']:<18} | "
            f"R2={best['oos_r2_vs_hist_mean']:+.4f} | "
            f"IC={best['spearman_ic']:+.4f} | "
            f"AUC={best['direction_auc_raw']:.4f} | "
            f"D10-D1={best['top_minus_bottom_decile_raw']:+.6f}"
        )
    print()
    print("OUTPUT_VALIDATION_PASS")
    print(f"output={out}")
    print("STOP: no backtest / no strategy rules / no hyperparameter search.")


if __name__ == "__main__":
    main()
