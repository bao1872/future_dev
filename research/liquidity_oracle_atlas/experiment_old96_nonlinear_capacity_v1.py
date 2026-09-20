#!/usr/bin/env python3
"""
experiment_old96_nonlinear_capacity_v1
=====================================

R3D-E4 OLD96 NONLINEAR CAPACITY DIAGNOSTIC.

One question only:

    Is OLD96's weak signal limited by the information itself, or by the
    linear capacity of Ridge?

Design: Ridge and HistGradientBoostingRegressor are fitted on the EXACT
SAME transformed matrix, produced once per stage by the frozen
E1.make_preprocessor(). The only variable is model capacity.

    M0  Ridge(alpha=1.0)                     frozen linear baseline
    M1  HistGradientBoostingRegressor(...)   frozen, parameters untouched

Features: E1.VARIANTS["V0_OLD96"] (96). No Field extractor is used this
round -- the canonical FormingEnvironmentBuilder is called directly.

Targets: Yopp_PCT (primary), Yopp_ATR200 (robustness).
Yopp_ATR5 / Ydir / TaskA / TaskB are not trained.

No tuning, no feature importance, no SHAP, no second nonlinear model.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import research.liquidity_oracle_atlas.experiment_field_representation_v1 as E1  # noqa: E402
import research.liquidity_oracle_atlas.experiment_vol_normalization_falsification_v1 as E12  # noqa: E402
from research.liquidity_oracle_atlas.build_forming_environment_v1 import (  # noqa: E402
    FormingEnvironmentBuilder,
)

SYMBOLS = [
    "AG", "AU", "CU", "AL", "SN",
    "NI", "RB", "I", "SC", "RU",
    "MA", "TA", "M", "P", "CF",
]

TARGETS = ["Yopp_PCT", "Yopp_ATR200"]
MODEL_NAMES = ("RIDGE", "HGB")

FEATURES = E1.VARIANTS["V0_OLD96"]
assert len(FEATURES) == 96
assert not (set(FEATURES) & (set(E1.VOL_COLS) | set(E1.MATURITY_COLS)))

KEEP_COLS = (
    list(E1.KEY)
    + [
        "trading_day", "decision_close", "m5_atr",
        "QL_24", "QS_24", "QW_24",
        "label_available_time_6", "label_available_time_12",
        "label_available_time_24",
        "joint_retention_stable",
        "stable_action",  # decile bookkeeping only
    ]
    + FEATURES
)

assert "stable_action" not in FEATURES
assert "joint_retention_stable" not in FEATURES
assert "decision_close" not in FEATURES
assert "m5_atr" not in FEATURES

E2_RESULTS_REL = os.path.join(
    "artifacts", "environment_field_full_validation_v1",
    "field_full_validation_results_v1.json")
ABS_TOL = 1e-10

RUNTIME_ENV_GUARD_SEC = 120.0
RUNTIME_FIT_GUARD_SEC = 300.0


def _stop(msg: str):
    raise SystemExit(f"STOP: {msg}")


def git_head_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "UNKNOWN"


def build_old96_environment(symbol):
    """Canonical builder only -- no Field extractor this round."""
    b = FormingEnvironmentBuilder(symbol, max_bars=None)
    b.load_raw()
    b.prepare()
    env, audit = b.run(profile_memory=False)

    if len(env) != len(b.base):
        _stop(f"STOP_ENV_BASE_LENGTH {symbol} {len(env)} != {len(b.base)}")

    env["decision_close"] = b.base["close"].to_numpy(np.float64)
    if "m5_atr" not in env.columns:
        _stop(f"STOP_MISSING_M5_ATR {symbol}")
    # canonical run() emits `data_object`; E1.KEY uses `symbol`.
    if "symbol" not in env.columns:
        env["symbol"] = env["data_object"].astype(str)
    return env, audit


def build_targets(df):
    ql = df["QL_24"].to_numpy(float)
    qs = df["QS_24"].to_numpy(float)
    qw = df["QW_24"].to_numpy(float)
    raw = np.maximum(ql, qs) - qw

    close = df["decision_close"].to_numpy(float)
    df["Yopp_PCT"] = np.divide(
        raw, close,
        out=np.full(len(df), np.nan, dtype=np.float64),
        where=np.isfinite(close) & (close != 0),
    )
    atr200 = df["m5_atr"].to_numpy(float)
    df["Yopp_ATR200"] = np.divide(
        raw, atr200,
        out=np.full(len(df), np.nan, dtype=np.float64),
        where=np.isfinite(atr200) & (atr200 > 0),
    )


def make_models():
    return {
        "RIDGE": Ridge(alpha=1.0),
        "HGB": HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            max_depth=4,
            min_samples_leaf=500,
            l2_regularization=1.0,
            max_bins=127,
            early_stopping=False,
            random_state=20260920,
        ),
    }


def fit_stage(joined, train_mask, eval_mask, boundary, counters, timings):
    lat = joined["LabelAvailableTime"]
    use_train = train_mask & (lat < boundary)
    tr = joined[use_train]
    ev = joined[eval_mask]

    t0 = time.perf_counter()
    pre = E1.make_preprocessor(FEATURES)
    Xtr = pre.fit_transform(tr[FEATURES])
    Xev = pre.transform(ev[FEATURES])
    prep_sec = time.perf_counter() - t0
    counters["preprocessor_fit_count"] += 1
    counters["train_transform_count"] += 1
    counters["eval_transform_count"] += 1
    timings.setdefault("preprocess_sec", []).append(round(prep_sec, 3))

    if not isinstance(Xtr, np.ndarray):
        _stop("STOP_EXPECTED_DENSE_XTR")
    if not isinstance(Xev, np.ndarray):
        _stop("STOP_EXPECTED_DENSE_XEV")
    if Xtr.shape[1] != Xev.shape[1]:
        _stop("STOP_TRANSFORM_WIDTH_MISMATCH")

    result = {
        "n_train_used": int(len(tr)),
        "n_eval": int(len(ev)),
        "purged_rows": int((train_mask & ~(lat < boundary)).sum()),
        "matrix": {
            "train_shape": list(Xtr.shape),
            "eval_shape": list(Xev.shape),
            "dtype": str(Xtr.dtype),
        },
        "models": {},
    }
    prediction_pack = {}

    for target in TARGETS:
        ytr = tr[target].to_numpy(float)
        yev = ev[target].to_numpy(float)
        ok_tr = np.isfinite(ytr)
        ok_ev = np.isfinite(yev)

        for model_name, model in make_models().items():
            t0 = time.perf_counter()
            model.fit(Xtr[ok_tr], ytr[ok_tr])
            fit_sec = time.perf_counter() - t0
            if fit_sec > RUNTIME_FIT_GUARD_SEC:
                _stop(f"STOP_MODEL_RUNTIME {model_name}/{target} {fit_sec:.1f}s")

            counters["ridge_fit_count"] += int(model_name == "RIDGE")
            counters["hgb_fit_count"] += int(model_name == "HGB")
            timings.setdefault(f"{model_name}_fit_sec", []).append(round(fit_sec, 3))

            pred_eval = model.predict(Xev[ok_ev])
            pred_fit = model.predict(Xtr[ok_tr])  # fit-sample diagnostic only

            full_eval = np.full(len(ev), np.nan, dtype=np.float64)
            full_eval[np.flatnonzero(ok_ev)] = pred_eval
            prediction_pack[(model_name, target)] = full_eval

            eval_metrics = E1.reg_metrics(yev[ok_ev], pred_eval)
            fit_metrics = E1.reg_metrics(ytr[ok_tr], pred_fit)
            result["models"].setdefault(model_name, {})[target] = {
                "fit": fit_metrics,
                "eval": eval_metrics,
                "fit_sec": fit_sec,
                "gap_spearman": float(fit_metrics["spearman"] - eval_metrics["spearman"]),
            }

    del Xtr, Xev, pre
    return result, prediction_pack, ev


def per_symbol_metrics(ev, predictions):
    symbol_arr = ev["symbol"].astype(str).to_numpy()
    rows = []
    for sym in SYMBOLS:
        sm = symbol_arr == sym
        for target in TARGETS:
            y = ev[target].to_numpy(float)
            for model_name in MODEL_NAMES:
                p = predictions[(model_name, target)]
                ok = sm & np.isfinite(y) & np.isfinite(p)
                if int(ok.sum()) < 3:
                    continue
                mt = E1.reg_metrics(y[ok], p[ok])
                rows.append({
                    "symbol": sym, "target": target, "model": model_name,
                    "n": mt["n"], "spearman": mt["spearman"],
                    "r2": mt["r2"], "mae": mt["mae"],
                })
    return pd.DataFrame(rows)


def cross_symbol_summary(ridge_s, hgb_s):
    a = ridge_s.astype(float)
    b = hgb_s.astype(float)
    d = (b - a).dropna()
    av = a.reindex(d.index).to_numpy(float)
    bv = b.reindex(d.index).to_numpy(float)
    dv = d.to_numpy(float)
    fin = np.isfinite(dv)
    av, bv, dv = av[fin], bv[fin], dv[fin]
    return dict(
        n_symbols=int(len(dv)),
        positive_delta_count=int(np.sum(dv > 0)),
        non_positive_delta_count=int(np.sum(dv <= 0)),
        ridge_positive_count=int(np.sum(av > 0)),
        hgb_positive_count=int(np.sum(bv > 0)),
        median_ridge_spearman=float(np.median(av)),
        median_hgb_spearman=float(np.median(bv)),
        median_delta=float(np.median(dv)),
        mean_delta=float(np.mean(dv)),
        min_delta=float(np.min(dv)),
        max_delta=float(np.max(dv)),
    )


def dual_target_summary(pct_delta, atr_delta):
    p = np.asarray(pct_delta, dtype=float)
    a = np.asarray(atr_delta, dtype=float)
    ok = np.isfinite(p) & np.isfinite(a)
    p, a = p[ok], a[ok]
    return {
        "n_symbols": int(len(p)),
        "joint_positive_count": int(np.sum((p > 0) & (a > 0))),
        "both_negative_count": int(np.sum((p <= 0) & (a <= 0))),
        "mixed_sign_count": int(np.sum(((p > 0) & (a <= 0)) | ((p <= 0) & (a > 0)))),
        "median_pct_delta": float(np.median(p)),
        "median_atr200_delta": float(np.median(a)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(
        _REPO_ROOT, "artifacts", "environment_old96_nonlinear_capacity_v1"))
    ap.add_argument("--results-name", default="old96_nonlinear_capacity_results_v1.json")
    ap.add_argument("--per-symbol-name", default="old96_nonlinear_capacity_per_symbol_v1.csv")
    ap.add_argument("--deciles-name", default="old96_nonlinear_capacity_deciles_v1.csv")
    ap.add_argument("--parity-bars", type=int, default=2000)
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    t_all = time.perf_counter()
    counters = dict(environment_build_count=0, preprocessor_fit_count=0,
                    train_transform_count=0, eval_transform_count=0,
                    ridge_fit_count=0, hgb_fit_count=0)
    timings = {}

    # ---- Gate: OLD96 compatibility (canonical run() vs frozen extractor) ----
    parity = E12.old96_parity_e12("AG", args.parity_bars)
    print(f"[gate] OLD96 compat {parity}", flush=True)

    # ---- Gate: oracle identity ----
    r1, r2 = E12.load_oracle_e12()

    # ---- one canonical environment build per symbol ----
    parts = []
    env_rows_total = 0
    env_sec_total = 0.0
    runtime_rows = []
    for i, sym in enumerate(SYMBOLS, start=1):
        t0 = time.perf_counter()
        env, _audit = build_old96_environment(sym)
        counters["environment_build_count"] += 1
        env_rows_total += len(env)

        o1 = r1[r1["symbol"] == sym]
        o2 = r2[r2["symbol"] == sym]

        env_keys = set(zip(env["decision_bar_index"].astype("int64"),
                           env["decision_time"].astype("int64")))
        oracle_keys = set(zip(o1["decision_bar_index"].astype("int64"),
                              o1["decision_time"].astype("int64")))
        missing = oracle_keys - env_keys
        if missing:
            _stop(f"STOP_ORACLE_NOT_COVERED {sym} {len(missing)}")

        full = env.merge(o1, on=E1.KEY, how="inner").merge(
            o2[E1.KEY + ["joint_retention_stable"]], on=E1.KEY, how="inner")
        parts.append(full[KEEP_COLS])

        sec = time.perf_counter() - t0
        env_sec_total += sec
        runtime_rows.append([sym, int(len(env)), round(sec, 2)])
        if sec > RUNTIME_ENV_GUARD_SEC:
            _stop(f"STOP_ENVIRONMENT_RUNTIME {sym} {sec:.1f}s")
        print(f"[{i:02d}/15] {sym} rows={len(env)} sec={sec:.1f}", flush=True)

    t_join0 = time.perf_counter()
    joined = pd.concat(parts, ignore_index=True)
    del parts
    join_sec = time.perf_counter() - t_join0

    oracle_rows = int(len(r1[r1["symbol"].isin(SYMBOLS)]))
    matched = int(len(joined))
    env_only = int(env_rows_total - matched)
    oracle_only = int(oracle_rows - matched)
    if oracle_only != 0:
        _stop(f"STOP_ORACLE_ONLY_ROWS {oracle_only}")

    build_targets(joined)
    act = joined["stable_action"].astype(str)
    joined["Y_trade"] = np.where(
        act.isin(["Long", "Short"]), 1.0, np.where(act == "Wait", 0.0, np.nan))
    joined["LabelAvailableTime"] = pd.concat([
        joined["label_available_time_6"],
        joined["label_available_time_12"],
        joined["label_available_time_24"],
    ], axis=1).max(axis=1)

    # ---- split (frozen) + drift gate ----
    days = np.sort(pd.to_datetime(joined["trading_day"]).dt.normalize().unique())
    n_d = len(days)
    n_tr, n_va = int(n_d * 0.6), int(n_d * 0.2)
    tr_days, va_days, te_days = days[:n_tr], days[n_tr:n_tr + n_va], days[n_tr + n_va:]
    split = dict(
        train_start=str(tr_days[0]), train_end=str(tr_days[-1]),
        val_start=str(va_days[0]), val_end=str(va_days[-1]),
        test_start=str(te_days[0]), test_end=str(te_days[-1]),
    )
    dcol = pd.to_datetime(joined["trading_day"]).dt.normalize()
    m_tr, m_va, m_te = dcol.isin(tr_days), dcol.isin(va_days), dcol.isin(te_days)
    split["n_train"], split["n_val"], split["n_test"] = (
        int(m_tr.sum()), int(m_va.sum()), int(m_te.sum()))

    e2_path = os.path.join(_REPO_ROOT, E2_RESULTS_REL)
    with open(e2_path) as fh:
        e2 = json.load(fh)
    for k, v in e2["split"].items():
        if k in split and k.startswith("n_"):
            if int(split[k]) != int(v):
                _stop(f"STOP_SPLIT_DRIFT {k} {split[k]} != {v}")
        elif k in split:
            if pd.Timestamp(split[k]) != pd.Timestamp(v):
                _stop(f"STOP_SPLIT_DRIFT {k} {split[k]} != {v}")
    print(f"[split] {split}", flush=True)

    va_start, te_start = va_days[0], te_days[0]
    stages = [
        ("VAL", m_tr, m_va, va_start),
        ("TEST", (m_tr | m_va), m_te, te_start),
    ]

    # ---- shared-matrix fits ----
    stage_out = {}
    test_ev = None
    test_pack = None
    for stage_name, train_mask, eval_mask, boundary in stages:
        res, pack, ev = fit_stage(
            joined, train_mask, eval_mask, boundary, counters, timings)
        stage_out[stage_name] = res
        if stage_name == "TEST":
            test_ev, test_pack = ev, pack
        print(f"[stage] {stage_name} train={res['n_train_used']} "
              f"eval={res['n_eval']} matrix={res['matrix']['train_shape']}", flush=True)

    # ---- Gate: E2 Ridge baseline parity ----
    parity_rows = []
    max_diff = 0.0
    for target in TARGETS:
        for stage_name in ("VAL", "TEST"):
            cur = stage_out[stage_name]["models"]["RIDGE"][target]["eval"]
            ref = e2["pooled"][target]["V0_OLD96"][stage_name.lower()]
            for mk in ("spearman", "r2", "mae"):
                diff = abs(float(cur[mk]) - float(ref[mk]))
                max_diff = max(max_diff, diff)
                parity_rows.append(dict(target=target, stage=stage_name, metric=mk,
                                        e2=float(ref[mk]), e4=float(cur[mk]), abs_diff=diff))
            if int(cur["n"]) != int(ref["n"]):
                _stop(f"STOP_RIDGE_BASELINE_DRIFT n {target} {stage_name}")
    if max_diff > ABS_TOL:
        _stop(f"STOP_RIDGE_BASELINE_DRIFT max_diff={max_diff:.3e}")
    print(f"[gate] Ridge baseline parity max_diff={max_diff:.3e}", flush=True)

    # ---- results ----
    results = dict(
        task_id="FUTURE-ENV-R3D-E4-OLD96-NONLINEAR-CAPACITY",
        code_sha=git_head_sha(),
        gates=dict(
            old96_compatibility=parity,
            oracle_identity=dict(rows=int(len(r1)), duplicate_keys=0,
                                 key_sets_identical=True),
            join=dict(environment_rows=env_rows_total, oracle_rows=oracle_rows,
                      matched_rows=matched, environment_only_rows=env_only,
                      oracle_only_rows=oracle_only),
            ridge_baseline_parity=dict(max_abs_diff=max_diff, rows=parity_rows),
        ),
        split=split,
        features=dict(
            raw_feature_count=len(FEATURES),
            transformed_width_val=stage_out["VAL"]["matrix"]["train_shape"][1],
            transformed_width_test=stage_out["TEST"]["matrix"]["train_shape"][1],
        ),
        models={m: {} for m in MODEL_NAMES},
        pooled_delta={t: {} for t in TARGETS},
        cross_symbol_summary={},
        dual_target_summary={},
        robustness_ge_09={},
        overfit_diagnostic=[],
    )

    for model_name in MODEL_NAMES:
        for stage_name in ("VAL", "TEST"):
            for target in TARGETS:
                m = stage_out[stage_name]["models"][model_name][target]
                results["models"][model_name].setdefault(stage_name, {})[target] = m

    for target in TARGETS:
        for stage_name in ("VAL", "TEST"):
            a = stage_out[stage_name]["models"]["RIDGE"][target]["eval"]
            b = stage_out[stage_name]["models"]["HGB"][target]["eval"]
            results["pooled_delta"][target][stage_name] = dict(
                ridge_spearman=float(a["spearman"]),
                hgb_spearman=float(b["spearman"]),
                delta_nonlinear=float(b["spearman"] - a["spearman"]),
                ridge_r2=float(a["r2"]), hgb_r2=float(b["r2"]),
                ridge_mae=float(a["mae"]), hgb_mae=float(b["mae"]),
                n=int(a["n"]),
            )
        for stage_name in ("VAL", "TEST"):
            for model_name in MODEL_NAMES:
                m = stage_out[stage_name]["models"][model_name][target]
                results["overfit_diagnostic"].append(dict(
                    stage=stage_name, target=target, model=model_name,
                    fit_spearman=float(m["fit"]["spearman"]),
                    eval_spearman=float(m["eval"]["spearman"]),
                    gap_spearman=float(m["gap_spearman"]),
                    fit_sec=float(m["fit_sec"]),
                ))

    # ---- per-symbol TEST ----
    ps_long = per_symbol_metrics(test_ev, test_pack)
    piv = ps_long.pivot_table(index="symbol", columns=["target", "model"],
                              values="spearman")
    rows = []
    for sym in SYMBOLS:
        row = {"symbol": sym}
        for target, tkey in (("Yopp_PCT", "PCT"), ("Yopp_ATR200", "ATR200")):
            r_ = float(piv.loc[sym, (target, "RIDGE")])
            h_ = float(piv.loc[sym, (target, "HGB")])
            row[f"{tkey}_Ridge"] = r_
            row[f"{tkey}_HGB"] = h_
            row[f"{tkey}_delta"] = h_ - r_
        rows.append(row)
    per_symbol_df = pd.DataFrame(rows)
    ps_path = os.path.join(args.out_dir, args.per_symbol_name)
    per_symbol_df.to_csv(ps_path, index=False)
    results["per_symbol_rows"] = per_symbol_df.to_dict("records")
    results["per_symbol_long"] = ps_long.to_dict("records")

    for target in TARGETS:
        results["cross_symbol_summary"][target] = cross_symbol_summary(
            piv[(target, "RIDGE")], piv[(target, "HGB")])
    results["dual_target_summary"] = dual_target_summary(
        per_symbol_df["PCT_delta"].to_numpy(float),
        per_symbol_df["ATR200_delta"].to_numpy(float))

    # ---- robustness >= 0.9 (slice global TEST predictions) ----
    for target in TARGETS:
        y = test_ev[target].to_numpy(float)
        mask = test_ev["joint_retention_stable"].to_numpy(float) >= 0.9
        base = None
        for model_name in MODEL_NAMES:
            p = test_pack[(model_name, target)]
            ok = mask & np.isfinite(y) & np.isfinite(p)
            sp = float(E1.reg_metrics(y[ok], p[ok])["spearman"])
            if model_name == "RIDGE":
                base = sp
            results["robustness_ge_09"].setdefault(target, {})[model_name] = dict(
                n=int(ok.sum()), spearman=sp,
                delta_vs_ridge=(sp - base) if base is not None else None)

    # ---- deciles (TEST only, 4 tables) ----
    deciles = []
    for target in TARGETS:
        for model_name in MODEL_NAMES:
            dd = test_ev.copy()
            dd["_pred"] = test_pack[(model_name, target)]
            deciles += E12.decile_rows(dd, "_pred", target, f"{model_name}__{target}")
    dpath = os.path.join(args.out_dir, args.deciles_name)
    pd.DataFrame(deciles).to_csv(dpath, index=False)

    # ---- counter hard gate ----
    expected = dict(environment_build_count=15, preprocessor_fit_count=2,
                    train_transform_count=2, eval_transform_count=2,
                    ridge_fit_count=4, hgb_fit_count=4)
    if counters != expected:
        _stop(f"STOP_DUPLICATED_WORK {counters} != {expected}")

    results["runtime"] = dict(
        per_object=runtime_rows,
        environment_total_sec=round(env_sec_total, 3),
        join_sec=round(join_sec, 3),
        preprocess_sec=timings.get("preprocess_sec"),
        ridge_fit_sec=timings.get("RIDGE_fit_sec"),
        hgb_fit_sec=timings.get("HGB_fit_sec"),
        total_sec=round(time.perf_counter() - t_all, 3),
        counters=counters,
    )

    rpath = os.path.join(args.out_dir, args.results_name)
    with open(rpath, "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    print(f"[counters] {counters}", flush=True)
    print(f"[save] {rpath}\n[save] {ps_path}\n[save] {dpath}", flush=True)
    print(f"[done] total_sec={results['runtime']['total_sec']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
