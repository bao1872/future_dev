#!/usr/bin/env python3
"""
experiment_field_full_validation_v1
===================================

R3C-E2 15-SYMBOL FIELD VALIDATION.

One question only:

    Does Field Core (V4) improve over OLD96 (V0) consistently ACROSS
    15 symbols, or is any pooled gain driven by a couple of objects?

No new features, no new math, no model tuning. The frozen field
implementation is imported and reused verbatim:

    E1  = experiment_field_representation_v1   (field math, preprocessor)
    E12 = experiment_vol_normalization_falsification_v1
          (extended extractor, oracle loader, parity, deciles)

Representations (only two):
    V0_OLD96        96
    V4_FIELD_CORE  148

Targets (this order):
    Yopp_PCT      raw / close        PRIMARY
    Yopp_ATR200   raw / m5 ATR200    ROBUSTNESS
    Yopp_ATR5     frozen target      HISTORICAL DIAGNOSTIC ONLY

VolRegime and Maturity are deliberately NOT included in any variant.
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

TARGETS = ["Yopp_PCT", "Yopp_ATR200", "Yopp_ATR5"]

VARIANTS_E2 = {
    "V0_OLD96": E1.VARIANTS["V0_OLD96"],
    "V4_FIELD_CORE": E1.VARIANTS["V4_FIELD_CORE"],
}
assert len(VARIANTS_E2["V0_OLD96"]) == 96
assert len(VARIANTS_E2["V4_FIELD_CORE"]) == 148

# E2 runs only these two representations
for _banned in ("V1_TREND_REPR", "V2_SR_REPR", "V3_LIQ_REPR", "V5_FIELD_FULL",
                "V6_OLD96_AUX", "V7_FIELD_VOL", "V8_FIELD_MAT", "V9_OLD96_VOL"):
    assert _banned not in VARIANTS_E2

FEAT_COLS = sorted(set(VARIANTS_E2["V0_OLD96"]) | set(VARIANTS_E2["V4_FIELD_CORE"]))
KEEP_COLS = (
    list(E1.KEY)
    + [
        "trading_day", "decision_close", "m5_atr",
        "QL_24", "QS_24", "QW_24",
        "QL_24_ATR", "QS_24_ATR", "QW_24_ATR", "atr5_t",
        "label_available_time_6", "label_available_time_12",
        "label_available_time_24",
        "stable_action", "baseline_stable_action", "joint_retention_stable",
    ]
    + FEAT_COLS
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


def build_targets(joined: pd.DataFrame) -> float:
    ql = joined["QL_24"].to_numpy(float)
    qs = joined["QS_24"].to_numpy(float)
    qw = joined["QW_24"].to_numpy(float)
    raw = np.maximum(ql, qs) - qw
    joined["Yopp_raw"] = raw

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
    joined["Yopp_ATR5"] = (
        np.maximum(joined["QL_24_ATR"], joined["QS_24_ATR"]) - joined["QW_24_ATR"]
    )

    # Gate D: ATR5 reconstruction
    a = joined["Yopp_ATR5"].to_numpy(float)
    bb = joined["Yopp_raw"].to_numpy(float) / joined["atr5_t"].to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(bb)
    return float(np.max(np.abs(a[ok] - bb[ok]))) if ok.any() else float("nan")


def fit_stage(joined, cols, train_mask, eval_mask, boundary):
    """ONE fit + ONE train transform + ONE eval transform; 3 targets reused."""
    lat = joined["LabelAvailableTime"]
    tr = joined[train_mask & ~(lat >= boundary)]
    ev = joined[eval_mask]

    pre = E1.make_preprocessor(cols)
    Xtr = pre.fit_transform(tr[cols])
    Xev = pre.transform(ev[cols])

    out = {
        "purged_rows": int((lat[train_mask] >= boundary).sum()),
        "n_train_used": int(len(tr)),
        "n_eval": int(len(ev)),
        "targets": {},
    }
    predictions = {}

    for target in TARGETS:
        ytr = tr[target].to_numpy(float)
        yev = ev[target].to_numpy(float)
        mtr, mev = np.isfinite(ytr), np.isfinite(yev)

        model = Ridge(alpha=1.0)
        model.fit(Xtr[mtr], ytr[mtr])
        p = model.predict(Xev[mev])

        full = np.full(len(ev), np.nan, dtype=float)
        full[np.flatnonzero(mev)] = p
        predictions[target] = full
        out["targets"][target] = E1.reg_metrics(yev[mev], p)

    del Xtr, Xev, pre
    return out, predictions, ev


def per_symbol_metrics(ev, predictions):
    rows = []
    symbols = ev["symbol"].astype(str).to_numpy()
    for sym in SYMBOLS:
        sm = symbols == sym
        for target in TARGETS:
            y = ev[target].to_numpy(float)
            p = predictions[target]
            ok = sm & np.isfinite(y) & np.isfinite(p)
            if int(ok.sum()) < 3:
                continue
            mt = E1.reg_metrics(y[ok], p[ok])
            rows.append(dict(symbol=sym, target=target, n=mt["n"],
                             spearman=mt["spearman"], r2=mt["r2"], mae=mt["mae"]))
    return pd.DataFrame(rows)


def cross_symbol_summary(ps_v0, ps_v4, target):
    a = (ps_v0[ps_v0["target"] == target]
         .set_index("symbol")["spearman"].astype(float))
    b = (ps_v4[ps_v4["target"] == target]
         .set_index("symbol")["spearman"].astype(float))
    d = (b - a).dropna()
    av, bv = a.reindex(d.index).to_numpy(float), b.reindex(d.index).to_numpy(float)
    dv = d.to_numpy(float)
    fin = np.isfinite(dv)
    dv = dv[fin]
    av, bv = av[fin], bv[fin]
    return dict(
        n_symbols=int(len(dv)),
        v0_positive_count=int(np.sum(av > 0)),
        v4_positive_count=int(np.sum(bv > 0)),
        positive_delta_count=int(np.sum(dv > 0)),
        non_positive_delta_count=int(np.sum(dv <= 0)),
        median_v0=float(np.median(av)),
        median_v4=float(np.median(bv)),
        median_delta=float(np.median(dv)),
        mean_delta=float(np.mean(dv)),
        min_delta=float(np.min(dv)),
        max_delta=float(np.max(dv)),
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(
        _REPO_ROOT, "artifacts", "environment_field_full_validation_v1"))
    ap.add_argument("--results-name", default="field_full_validation_results_v1.json")
    ap.add_argument("--per-symbol-name", default="field_full_validation_per_symbol_v1.csv")
    ap.add_argument("--deciles-name", default="field_full_validation_deciles_v1.csv")
    ap.add_argument("--parity-bars", type=int, default=2000)
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    t_all = time.perf_counter()
    code_sha = git_head_sha()

    # ---- Gate A: OLD96 parity ----
    parity = E12.old96_parity_e12("AG", args.parity_bars)
    print(f"[gateA] {parity}", flush=True)
    if parity["max_abs_error"] > 1e-9 or parity["discrete_mismatch"] != 0:
        _stop("OLD96_PARITY_FAIL")

    # ---- Gate B: oracle identity ----
    r1, r2 = E12.load_oracle_e12()

    # ---- one environment build + one join per symbol ----
    parts = []
    env_rows_total = 0
    env_sec_total = 0.0
    per_object_runtime = []
    for i, sym in enumerate(SYMBOLS, start=1):
        t0 = time.perf_counter()
        env = E12.build_field_environment_e12(sym, max_bars=None)
        env_sec = time.perf_counter() - t0
        env_rows_total += len(env)
        env_sec_total += env_sec
        if env_sec > RUNTIME_GUARD_SEC:
            _stop(f"STOP_UNEXPECTED_RUNTIME {sym} {env_sec:.1f}s > {RUNTIME_GUARD_SEC}")

        o1 = r1[r1["symbol"] == sym]
        o2 = r2[r2["symbol"] == sym]

        env_keys = set(zip(
            env["decision_bar_index"].astype("int64").tolist(),
            env["decision_time"].astype("int64").tolist(),
        ))
        oracle_keys = set(zip(
            o1["decision_bar_index"].to_numpy().tolist(),
            o1["decision_time"].astype("int64").to_numpy().tolist(),
        ))
        oracle_only = oracle_keys - env_keys
        if oracle_only:
            _stop(f"STOP_ORACLE_KEYS_NOT_SUBSET {sym}: {len(oracle_only)}")

        full = env.merge(o1, on=E1.KEY, how="inner").merge(
            o2[E1.KEY + ["baseline_stable_action", "joint_retention_stable"]],
            on=E1.KEY, how="inner",
        )
        parts.append(full[KEEP_COLS])
        per_object_runtime.append((sym, len(env), round(env_sec, 1)))
        print(f"[{i:02d}/15] {sym} env={len(env)} joined={len(full)} "
              f"sec={env_sec:.1f}", flush=True)

    t_join0 = time.perf_counter()
    joined = pd.concat(parts, ignore_index=True)
    del parts
    t_join = time.perf_counter() - t_join0
    oracle_rows = int(len(r1[r1["symbol"].isin(SYMBOLS)]))
    print(f"[join] env_rows={env_rows_total} oracle_rows={oracle_rows} "
          f"matched={len(joined)} env_only={env_rows_total - len(joined)} oracle_only=0",
          flush=True)

    # ---- Gate D: ATR5 reconstruction ----
    tgt_err = build_targets(joined)
    if not (tgt_err <= 1e-10):
        _stop(f"ATR5_RECONSTRUCTION_MAX_ERR {tgt_err:.3e}")
    print(f"[gateD] ATR5 reconstruction max_err={tgt_err:.3e}", flush=True)

    act = joined["stable_action"].astype(str)
    joined["Y_trade"] = np.where(
        act.isin(["Long", "Short"]), 1.0, np.where(act == "Wait", 0.0, np.nan))

    joined["LabelAvailableTime"] = pd.concat([
        joined["label_available_time_6"],
        joined["label_available_time_12"],
        joined["label_available_time_24"],
    ], axis=1).max(axis=1)

    # ---- split by unique trading days (60/20/20, frozen) ----
    days = np.sort(pd.to_datetime(joined["trading_day"]).dt.normalize().unique())
    n_d = len(days)
    n_tr, n_va = int(n_d * 0.6), int(n_d * 0.2)
    tr_days = days[:n_tr]
    va_days = days[n_tr:n_tr + n_va]
    te_days = days[n_tr + n_va:]
    dcol = pd.to_datetime(joined["trading_day"]).dt.normalize()
    m_tr, m_va, m_te = dcol.isin(tr_days), dcol.isin(va_days), dcol.isin(te_days)
    va_start, te_start = va_days[0], te_days[0]

    print(f"[split] train n={int(m_tr.sum())} val n={int(m_va.sum())} "
          f"test n={int(m_te.sum())}", flush=True)

    stages = [("VAL", m_tr, m_va, va_start), ("TEST", (m_tr | m_va), m_te, te_start)]

    results = dict(
        task_id="FUTURE-ENV-R3C-E2-15SYMBOL-FIELD-VALIDATION",
        code_sha=code_sha,
        gates=dict(old96_parity=parity, atr5_reconstruction=tgt_err, oracle_identity={
            "rows": int(len(r1)), "duplicate_keys": 0, "key_sets_identical": True},
            join=dict(environment_rows=int(env_rows_total),
                      oracle_rows=oracle_rows,
                      matched_rows=int(len(joined)),
                      environment_only_rows=int(env_rows_total - len(joined)),
                      oracle_only_rows=0)),
        split=dict(
            train_start=str(tr_days[0]), train_end=str(tr_days[-1]),
            val_start=str(va_start), val_end=str(va_days[-1]),
            test_start=str(te_start), test_end=str(te_days[-1]),
            n_train=int(m_tr.sum()), n_val=int(m_va.sum()), n_test=int(m_te.sum())),
        pooled={t: {} for t in TARGETS},
        cross_symbol_summary={},
        robustness_ge_09={},
    )

    deciles = []
    per_symbol_tables = {}
    test_pack = {}
    model_secs = {}

    for vname, cols in VARIANTS_E2.items():
        t_v = time.perf_counter()
        vres = {}
        ps_rows = []
        for stage_name, train_mask, eval_mask, boundary in stages:
            out, predictions, ev = fit_stage(
                joined, cols, train_mask, eval_mask, boundary)
            vres[stage_name] = out
            if stage_name == "TEST":
                test_pack[vname] = dict(ev=ev, predictions=predictions)
                ps_rows = per_symbol_metrics(ev, predictions).to_dict("records")
                for target in ("Yopp_PCT", "Yopp_ATR200"):
                    dd = ev.copy()
                    dd["_pred"] = predictions[target]
                    deciles += E12.decile_rows(
                        dd, "_pred", target, f"{vname}__{target}")
        per_symbol_tables[vname] = pd.DataFrame(ps_rows)
        model_secs[vname] = round(time.perf_counter() - t_v, 3)
        for target in TARGETS:
            results["pooled"][target][vname] = dict(
                val=vres["VAL"]["targets"][target],
                test=vres["TEST"]["targets"][target])
        print(f"[variant] {vname} dim={len(cols)} TEST_spear="
              f"{ {t: round(vres['TEST']['targets'][t]['spearman'],4) for t in TARGETS} }",
              flush=True)

    # ---- pooled deltas ----
    results["pooled_delta_test_spearman"] = {
        t: results["pooled"][t]["V4_FIELD_CORE"]["test"]["spearman"]
        - results["pooled"][t]["V0_OLD96"]["test"]["spearman"]
        for t in TARGETS
    }
    results["pooled_delta_val_spearman"] = {
        t: results["pooled"][t]["V4_FIELD_CORE"]["val"]["spearman"]
        - results["pooled"][t]["V0_OLD96"]["val"]["spearman"]
        for t in TARGETS
    }

    # ---- per-symbol tables + deltas ----
    ps0, ps4 = per_symbol_tables["V0_OLD96"], per_symbol_tables["V4_FIELD_CORE"]
    merged = ps0.merge(ps4, on=["symbol", "target"], suffixes=("_V0", "_V4"))
    merged["delta_spearman"] = merged["spearman_V4"] - merged["spearman_V0"]
    ps_path = os.path.join(args.out_dir, args.per_symbol_name)
    merged.to_csv(ps_path, index=False)
    results["per_symbol_rows"] = merged.to_dict("records")

    # ---- cross-symbol stability ----
    for target in TARGETS:
        results["cross_symbol_summary"][target] = cross_symbol_summary(
            ps0, ps4, target)

    # ---- robustness >= 0.9 (slice global TEST predictions, no refit) ----
    rob = {}
    for target in ("Yopp_PCT", "Yopp_ATR200"):
        entry = {}
        for vname in VARIANTS_E2:
            ev = test_pack[vname]["ev"]
            p = test_pack[vname]["predictions"][target]
            mask = (ev["joint_retention_stable"].to_numpy() >= 0.9)
            y = ev[target].to_numpy(float)
            ok = mask & np.isfinite(y) & np.isfinite(p)
            entry[vname] = dict(n=int(ok.sum()),
                                spearman=float(E1.reg_metrics(y[ok], p[ok])["spearman"]))
        entry["delta_test_spearman"] = (
            entry["V4_FIELD_CORE"]["spearman"] - entry["V0_OLD96"]["spearman"])
        rob[target] = entry
    results["robustness_ge_09"] = rob

    dpath = os.path.join(args.out_dir, args.deciles_name)
    pd.DataFrame(deciles).to_csv(dpath, index=False)

    results["runtime"] = dict(
        per_object=per_object_runtime,
        environment_total_sec=round(env_sec_total, 3),
        join_sec=round(t_join, 3),
        V0_model_sec=model_secs.get("V0_OLD96"),
        V4_model_sec=model_secs.get("V4_FIELD_CORE"),
        total_sec=round(time.perf_counter() - t_all, 3),
    )

    rpath = os.path.join(args.out_dir, args.results_name)
    with open(rpath, "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    print(f"[save] {rpath}", flush=True)
    print(f"[save] {ps_path}", flush=True)
    print(f"[save] {dpath}", flush=True)
    print(f"[done] total_sec={results['runtime']['total_sec']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
