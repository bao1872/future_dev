#!/usr/bin/env python3
"""
experiment_field_component_full_validation_v1
=============================================

R3C-E3 COMPONENT-LEVEL FULL (15-symbol) VALIDATION.

One question only:

    Of the three new field blocks (Trend / SR / Liquidity), which one, if
    any, carries a cross-symbol stable increment over OLD96?

V4 (all three replaced at once) did not beat OLD96 pooled in E2, so this
round isolates each replacement:

    V1: Trend_new  + SR_old      + Liq_old      (108)
    V2: Trend_old  + SR_new      + Liq_old      (112)
    V3: Trend_old  + SR_old      + Liq_new      (120)
    V0: OLD96                                   ( 96)   reference

No new features, no new math, no model change. Frozen implementation is
imported and reused verbatim:

    E1  = experiment_field_representation_v1     (field math, preprocessor)
    E12 = experiment_vol_normalization_falsification_v1
          (extended extractor, oracle loader, parity, deciles)

VolRegime and Maturity are excluded from every variant (hard gate).

Targets: Yopp_PCT (primary), Yopp_ATR200 (robustness). Yopp_ATR5 /
Ydir / TaskA / TaskB are NOT trained this round.
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
from sklearn.linear_model import Ridge

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import research.liquidity_oracle_atlas.experiment_field_representation_v1 as E1  # noqa: E402
import research.liquidity_oracle_atlas.experiment_vol_normalization_falsification_v1 as E12  # noqa: E402

SYMBOLS = [
    "AG", "AU", "CU", "AL", "SN",
    "NI", "RB", "I", "SC", "RU",
    "MA", "TA", "M", "P", "CF",
]

TARGETS = ["Yopp_PCT", "Yopp_ATR200"]
COMPONENTS = ["trend", "sr", "liq"]
COMPONENT_VARIANT = {"trend": "V1_TREND_REPR", "sr": "V2_SR_REPR", "liq": "V3_LIQ_REPR"}

VARIANTS_E3 = {
    "V0_OLD96": E1.VARIANTS["V0_OLD96"],
    "V1_TREND_REPR": E1.VARIANTS["V1_TREND_REPR"],
    "V2_SR_REPR": E1.VARIANTS["V2_SR_REPR"],
    "V3_LIQ_REPR": E1.VARIANTS["V3_LIQ_REPR"],
}
assert len(VARIANTS_E3["V0_OLD96"]) == 96
assert len(VARIANTS_E3["V1_TREND_REPR"]) == 108
assert len(VARIANTS_E3["V2_SR_REPR"]) == 112
assert len(VARIANTS_E3["V3_LIQ_REPR"]) == 120

_ORDER = ["V0_OLD96", "V1_TREND_REPR", "V2_SR_REPR", "V3_LIQ_REPR"]
assert list(VARIANTS_E3) == _ORDER

BANNED = set(E1.VOL_COLS) | set(E1.MATURITY_COLS)
for _name, _cols in VARIANTS_E3.items():
    _bad = set(_cols) & BANNED
    assert not _bad, (_name, _bad)

FEATURE_UNION = sorted(set().union(*VARIANTS_E3.values()))

KEEP_COLS = (
    list(E1.KEY)
    + [
        "trading_day", "decision_close", "m5_atr",
        "QL_24", "QS_24", "QW_24",
        "label_available_time_6", "label_available_time_12",
        "label_available_time_24",
        "joint_retention_stable",
        # label bookkeeping only (frozen Y_trade for deciles); NOT a feature
        "stable_action",
    ]
    + FEATURE_UNION
)

RUNTIME_GUARD_SEC = 150.0


def _stop(msg: str):
    raise SystemExit(f"STOP: {msg}")


def git_head_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "UNKNOWN"


def build_targets(joined):
    """E2-validated target math. PCT + ATR200 only."""
    ql = joined["QL_24"].to_numpy(float)
    qs = joined["QS_24"].to_numpy(float)
    qw = joined["QW_24"].to_numpy(float)
    raw = np.maximum(ql, qs) - qw

    close = joined["decision_close"].to_numpy(float)
    joined["Yopp_PCT"] = np.divide(
        raw, close,
        out=np.full(len(joined), np.nan, dtype=float),
        where=np.isfinite(close) & (close != 0),
    )
    atr200 = joined["m5_atr"].to_numpy(float)
    joined["Yopp_ATR200"] = np.divide(
        raw, atr200,
        out=np.full(len(joined), np.nan, dtype=float),
        where=np.isfinite(atr200) & (atr200 > 0),
    )


DECILE_VARIANTS = ("V0_OLD96", "V1_TREND_REPR", "V2_SR_REPR", "V3_LIQ_REPR")


def fit_stage(joined, cols, train_mask, eval_mask, boundary, counters):
    """ONE preprocessor fit + ONE train transform + ONE eval transform,
    shared by both targets (so 4 variants x 2 stages = 8 of each)."""
    lat = joined["LabelAvailableTime"]
    use_train = train_mask & (lat < boundary)
    tr = joined[use_train]
    ev = joined[eval_mask]

    pre = E1.make_preprocessor(cols)
    Xtr = pre.fit_transform(tr[cols])          # 1 fit + 1 train transform
    Xev = pre.transform(ev[cols])              # 1 eval transform
    counters["preprocessor_fit_count"] += 1
    counters["train_transform_count"] += 1
    counters["eval_transform_count"] += 1

    metrics = {}
    predictions = {}
    for target in TARGETS:
        ytr = tr[target].to_numpy(float)
        yev = ev[target].to_numpy(float)
        ok_tr = np.isfinite(ytr)
        ok_ev = np.isfinite(yev)

        model = Ridge(alpha=1.0)
        model.fit(Xtr[ok_tr], ytr[ok_tr])
        counters["ridge_fit_count"] += 1
        p = model.predict(Xev[ok_ev])

        full_pred = np.full(len(ev), np.nan, dtype=np.float64)
        full_pred[np.flatnonzero(ok_ev)] = p
        predictions[target] = full_pred
        metrics[target] = E1.reg_metrics(yev[ok_ev], p)

    result = {
        "purged_rows": int((train_mask & ~(lat < boundary)).sum()),
        "n_train_used": int(use_train.sum()),
        "n_eval": int(eval_mask.sum()),
        "targets": metrics,
    }
    del Xtr, Xev, pre
    return result, predictions, ev


def per_symbol_metrics(ev, predictions, variant):
    symbol_arr = ev["symbol"].astype(str).to_numpy()
    rows = []
    for sym in SYMBOLS:
        sm = symbol_arr == sym
        for target in TARGETS:
            y = ev[target].to_numpy(float)
            p = predictions[target]
            ok = sm & np.isfinite(y) & np.isfinite(p)
            if int(ok.sum()) < 3:
                continue
            mt = E1.reg_metrics(y[ok], p[ok])
            rows.append({
                "variant": variant, "symbol": sym, "target": target,
                "n": mt["n"], "spearman": mt["spearman"],
                "r2": mt["r2"], "mae": mt["mae"],
            })
    return pd.DataFrame(rows)


def component_summary(v0_series, vt_series):
    a = v0_series.astype(float)
    b = vt_series.astype(float)
    d = (b - a).dropna()
    av = a.reindex(d.index).to_numpy(float)
    bv = b.reindex(d.index).to_numpy(float)
    dv = d.to_numpy(float)
    fin = np.isfinite(dv)
    av, bv, dv = av[fin], bv[fin], dv[fin]
    return dict(
        n_symbols=int(len(dv)),
        variant_positive_count=int(np.sum(bv > 0)),
        v0_positive_count=int(np.sum(av > 0)),
        positive_delta_count=int(np.sum(dv > 0)),
        non_positive_delta_count=int(np.sum(dv <= 0)),
        median_v0=float(np.median(av)),
        median_variant=float(np.median(bv)),
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
        _REPO_ROOT, "artifacts", "environment_field_component_full_validation_v1"))
    ap.add_argument("--results-name", default="field_component_full_results_v1.json")
    ap.add_argument("--per-symbol-name", default="field_component_per_symbol_v1.csv")
    ap.add_argument("--deciles-name", default="field_component_deciles_v1.csv")
    ap.add_argument("--parity-bars", type=int, default=2000)
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    t_all = time.perf_counter()
    counters = dict(preprocessor_fit_count=0, train_transform_count=0,
                    eval_transform_count=0, ridge_fit_count=0)

    # ---- Gate A: OLD96 parity ----
    parity = E12.old96_parity_e12("AG", args.parity_bars)
    if parity["max_abs_error"] > 1e-9 or parity["discrete_mismatch"] != 0:
        _stop(f"STOP_OLD96_PARITY {parity}")
    print(f"[gateA] {parity}", flush=True)

    # ---- Gate B: oracle identity ----
    r1, r2 = E12.load_oracle_e12()

    # ---- one environment build + one join per symbol ----
    parts = []
    env_rows_total = 0
    env_sec_total = 0.0
    runtime_rows = []
    for i, sym in enumerate(SYMBOLS, start=1):
        t0 = time.perf_counter()
        env = E12.build_field_environment_e12(sym, max_bars=None)
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
        if sec > RUNTIME_GUARD_SEC:
            _stop(f"STOP_UNEXPECTED_RUNTIME {sym} {sec:.1f}s")
        print(f"[{i:02d}/15] {sym} rows={len(env)} sec={sec:.1f}", flush=True)

    # ---- Gate C: environment coverage ----
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

    # ---- targets / label / purge column ----
    build_targets(joined)
    act = joined["stable_action"].astype(str)
    joined["Y_trade"] = np.where(
        act.isin(["Long", "Short"]), 1.0, np.where(act == "Wait", 0.0, np.nan))
    joined["LabelAvailableTime"] = pd.concat([
        joined["label_available_time_6"],
        joined["label_available_time_12"],
        joined["label_available_time_24"],
    ], axis=1).max(axis=1)

    # ---- split (frozen: 60/20/20 on unique trading days) ----
    days = np.sort(pd.to_datetime(joined["trading_day"]).dt.normalize().unique())
    n_d = len(days)
    n_tr, n_va = int(n_d * 0.6), int(n_d * 0.2)
    tr_days, va_days, te_days = days[:n_tr], days[n_tr:n_tr + n_va], days[n_tr + n_va:]
    dcol = pd.to_datetime(joined["trading_day"]).dt.normalize()
    m_tr, m_va, m_te = dcol.isin(tr_days), dcol.isin(va_days), dcol.isin(te_days)
    va_start, te_start = va_days[0], te_days[0]
    stages = [("VAL", m_tr, m_va, va_start), ("TEST", (m_tr | m_va), m_te, te_start)]
    print(f"[split] train={int(m_tr.sum())} val={int(m_va.sum())} "
          f"test={int(m_te.sum())}", flush=True)

    # ---- models ----
    results = dict(
        task_id="FUTURE-ENV-R3C-E3-COMPONENT-FULL-VALIDATION",
        code_sha=git_head_sha(),
        gates=dict(
            old96_parity=parity,
            oracle_identity=dict(rows=int(len(r1)), duplicate_keys=0,
                                 key_sets_identical=True),
            join=dict(environment_rows=env_rows_total, oracle_rows=oracle_rows,
                      matched_rows=matched, environment_only_rows=env_only,
                      oracle_only_rows=oracle_only),
        ),
        split=dict(
            train_start=str(tr_days[0]), train_end=str(tr_days[-1]),
            val_start=str(va_start), val_end=str(va_days[-1]),
            test_start=str(te_start), test_end=str(te_days[-1]),
            n_train=int(m_tr.sum()), n_val=int(m_va.sum()), n_test=int(m_te.sum())),
        variant_dims={k: len(v) for k, v in VARIANTS_E3.items()},
        pooled={t: {} for t in TARGETS},
        component_pooled_delta={t: {} for t in TARGETS},
        cross_symbol_summary={t: {} for t in TARGETS},
        dual_target_summary={},
        robustness_ge_09={t: {} for t in TARGETS},
    )

    deciles = []
    per_symbol_tables = {}
    test_pack = {}
    model_secs = {}

    for vname in _ORDER:
        cols = VARIANTS_E3[vname]
        t_v = time.perf_counter()
        vres = {}
        for stage_name, train_mask, eval_mask, boundary in stages:
            out, predictions, ev = fit_stage(
                joined, cols, train_mask, eval_mask, boundary, counters)
            vres[stage_name] = out
            if stage_name == "TEST":
                test_pack[vname] = dict(ev=ev, predictions=predictions)
                per_symbol_tables[vname] = per_symbol_metrics(ev, predictions, vname)
                for target in TARGETS:
                    dd = ev.copy()
                    dd["_pred"] = predictions[target]
                    deciles += E12.decile_rows(
                        dd, "_pred", target, f"{vname}__{target}")
        model_secs[vname] = round(time.perf_counter() - t_v, 3)
        for target in TARGETS:
            results["pooled"][target][vname] = dict(
                val=vres["VAL"]["targets"][target],
                test=vres["TEST"]["targets"][target])
        print(f"[variant] {vname} dim={len(cols)} "
              f"TEST_spearman={ {t: round(vres['TEST']['targets'][t]['spearman'], 4) for t in TARGETS} }",
              flush=True)

    # ---- component pooled delta (vs V0) ----
    for target in TARGETS:
        for comp in COMPONENTS:
            vn = COMPONENT_VARIANT[comp]
            results["component_pooled_delta"][target][comp] = dict(
                val=results["pooled"][target][vn]["val"]["spearman"]
                - results["pooled"][target]["V0_OLD96"]["val"]["spearman"],
                test=results["pooled"][target][vn]["test"]["spearman"]
                - results["pooled"][target]["V0_OLD96"]["test"]["spearman"],
                val_r2=results["pooled"][target][vn]["val"]["r2"]
                - results["pooled"][target]["V0_OLD96"]["val"]["r2"],
                test_r2=results["pooled"][target][vn]["test"]["r2"]
                - results["pooled"][target]["V0_OLD96"]["test"]["r2"],
            )

    # ---- per-symbol wide table ----
    long = pd.concat(list(per_symbol_tables.values()), ignore_index=True)
    piv = long.pivot_table(index="symbol", columns=["target", "variant"],
                           values="spearman")
    rows = []
    for sym in SYMBOLS:
        row = {"symbol": sym}
        for target, tkey in (("Yopp_PCT", "PCT"), ("Yopp_ATR200", "ATR200")):
            v0 = float(piv.loc[sym, (target, "V0_OLD96")])
            row[f"{tkey}_V0"] = v0
            for comp in COMPONENTS:
                vn = COMPONENT_VARIANT[comp]
                vv = float(piv.loc[sym, (target, vn)])
                row[f"{tkey}_{vn}"] = vv
                row[f"{tkey}_delta_{comp}"] = vv - v0
        rows.append(row)
    per_symbol_df = pd.DataFrame(rows)
    ps_path = os.path.join(args.out_dir, args.per_symbol_name)
    per_symbol_df.to_csv(ps_path, index=False)
    results["per_symbol_rows"] = per_symbol_df.to_dict("records")

    # ---- cross-symbol summary + dual target ----
    for target in TARGETS:
        v0s = piv[(target, "V0_OLD96")]
        for comp in COMPONENTS:
            results["cross_symbol_summary"][target][comp] = component_summary(
                v0s, piv[(target, COMPONENT_VARIANT[comp])])
    for comp in COMPONENTS:
        results["dual_target_summary"][comp] = dual_target_summary(
            per_symbol_df[f"PCT_delta_{comp}"].to_numpy(float),
            per_symbol_df[f"ATR200_delta_{comp}"].to_numpy(float))

    # ---- robustness >= 0.9 (slice global TEST predictions, no refit) ----
    for target in TARGETS:
        base = None
        for vname in _ORDER:
            ev = test_pack[vname]["ev"]
            y = ev[target].to_numpy(float)
            p = test_pack[vname]["predictions"][target]
            mask = ev["joint_retention_stable"].to_numpy(float) >= 0.9
            ok = mask & np.isfinite(y) & np.isfinite(p)
            sp = float(E1.reg_metrics(y[ok], p[ok])["spearman"])
            if vname == "V0_OLD96":
                base = sp
            results["robustness_ge_09"][target][vname] = dict(
                n=int(ok.sum()), spearman=sp,
                delta_vs_v0=(sp - base) if base is not None else None)

    dpath = os.path.join(args.out_dir, args.deciles_name)
    pd.DataFrame(deciles).to_csv(dpath, index=False)

    results["runtime"] = dict(
        per_object=runtime_rows,
        environment_total_sec=round(env_sec_total, 3),
        join_sec=round(join_sec, 3),
        V0_model_sec=model_secs["V0_OLD96"],
        V1_model_sec=model_secs["V1_TREND_REPR"],
        V2_model_sec=model_secs["V2_SR_REPR"],
        V3_model_sec=model_secs["V3_LIQ_REPR"],
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
