#!/usr/bin/env python3
"""
experiment_environment_oracle_signal_v1
=======================================

R3B-E0 SIGNAL SANITY.

One question only:

    Does E_t = [D, SR, L]_{5m,15m,1H,4H} contain learnable information
    about the Oracle, out-of-sample, versus simple priors?

This is an EXPLORATORY sanity test, NOT model selection, NOT a backtest,
NOT tuning. Simplest possible models (LogisticRegression / Ridge with
fixed defaults) against global-prior and per-symbol-prior baselines.

Owners are frozen and are only *used*, never modified:
    build_forming_environment_v1.py          (environment)
    robust_trade_oracle_dp_v1 parquet        (R1.1 oracle)
    oracle_constraint_robustness_v1 parquet  (R2 robustness)

One environment pass per object, one join, one preprocessing fit per
walk-forward window, matrices reused for all four models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.liquidity_oracle_atlas.build_forming_environment_v1 import (  # noqa: E402
    FormingEnvironmentBuilder,
)

# --------------------------------------------------------------------------- #
# frozen contract                                                              #
# --------------------------------------------------------------------------- #
SYMBOLS = [
    "AG", "AU", "CU", "AL", "SN",
    "NI", "RB", "I", "SC", "RU",
    "MA", "TA", "M", "P", "CF",
]

R1_REL = os.path.join(
    "artifacts", "robust_trade_oracle_dp_v1", "robust_trade_oracle_rows.parquet"
)
R1_SHA = "ea30c67db0464da9fe4f80ede8dd597336ef6d69e08220ed6a4bb94210fcacfa"
R2_REL = os.path.join(
    "artifacts", "oracle_constraint_robustness_v1", "oracle_constraint_rows.parquet"
)
R2_SHA = "3980177b78d46cbc4ab7ffcd84dedebee3c5c62e396b11ec0da478b463e3f265"

KEY = ["symbol", "decision_bar_index", "decision_time"]

TF = ["m5", "m15", "h1", "h4"]
N_TF = len(TF)

# 24 base environment features per timeframe (96 total)
CONT_BASE = [
    "dev", "slope_atr", "trend_score",
    "sr_support_dist_atr", "sr_resistance_dist_atr",
    "sr_support_strength", "sr_resistance_strength", "sr_zone_strength",
    "liq_up_dist_atr", "liq_down_dist_atr", "liq_last_breach_age",
]
DISC_BASE = [
    "trend_state",
    "sr_in_zone", "sr_broken_up", "sr_broken_down", "sr_n_channels",
    "liq_breach_up", "liq_breach_down", "liq_last_breach_side",
    "liq_last_accept", "liq_last_reclaim", "liq_last_zone_active",
    "liq_up_count", "liq_down_count",
]
CONT_COLS = [f"{t}_{c}" for t in TF for c in CONT_BASE]
DISC_COLS = [f"{t}_{c}" for t in TF for c in DISC_BASE]
FEATURE_COLS = CONT_COLS + DISC_COLS

TRADE_ACTIONS = ["Long", "Short", "Wait"]      # Task A vocabulary
DIRECTION_ACTIONS = ["Long", "Short"]          # Task B vocabulary


def _stop(msg: str):
    raise SystemExit(f"STOP: {msg}")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# oracle load + identity gate                                                  #
# --------------------------------------------------------------------------- #
def load_oracle():
    r1_path = os.path.join(_REPO_ROOT, R1_REL)
    r2_path = os.path.join(_REPO_ROOT, R2_REL)
    for p, expect, tag in ((r1_path, R1_SHA, "R1.1"), (r2_path, R2_SHA, "R2")):
        if not os.path.exists(p):
            _stop(f"{tag} oracle parquet missing: {p}")
        got = sha256_file(p)
        if got != expect:
            _stop(
                f"STOP_ORACLE_ARTIFACT_IDENTITY_MISMATCH {tag}: "
                f"expected {expect} got {got}"
            )
        print(f"[oracle] {tag} sha256 OK ({got[:16]}...)", flush=True)

    r1 = pd.read_parquet(
        r1_path,
        columns=[
            "symbol", "decision_bar_index", "decision_time",
            "stable_action",
            "QL_24_ATR", "QS_24_ATR", "QW_24_ATR",
            "label_available_time_6", "label_available_time_12",
            "label_available_time_24",
        ],
    )
    r2 = pd.read_parquet(
        r2_path,
        columns=[
            "symbol", "decision_bar_index", "decision_time",
            "baseline_stable_action", "joint_retention_stable",
        ],
    )

    d1 = int(r1.duplicated(subset=KEY).sum())
    d2 = int(r2.duplicated(subset=KEY).sum())
    if d1 != 0 or d2 != 0:
        _stop(f"STOP_ORACLE_DUPLICATE_KEYS R1={d1} R2={d2}")

    k1 = set(map(tuple, r1[KEY].to_numpy(dtype=object).tolist()))
    k2 = set(map(tuple, r2[KEY].to_numpy(dtype=object).tolist()))
    if k1 != k2 or len(r1) != len(r2):
        _stop(
            f"STOP_R1_R2_KEY_MISMATCH len(r1)={len(r1)} len(r2)={len(r2)} "
            f"keys_equal={k1 == k2}"
        )
    print(
        f"[oracle] R1/R2 keys identical: rows={len(r1)} dup={d1}/{d2}",
        flush=True,
    )
    return r1, r2


# --------------------------------------------------------------------------- #
# environment: one pass per object, join once, append                          #
# --------------------------------------------------------------------------- #
def build_joined(symbols, r1, r2, limit_bars=None):
    parts = []
    env_rows_total = 0
    t_env = 0.0
    t_join = 0.0
    for i, sym in enumerate(symbols, start=1):
        t0 = time.perf_counter()
        b = FormingEnvironmentBuilder(sym, max_bars=limit_bars)
        b.load_raw()
        b.prepare()
        env, _audit = b.run(profile_memory=False)
        t_env += time.perf_counter() - t0

        if int(b.stats.raw_load_count) != 1:
            _stop(f"{sym} raw_load_count={b.stats.raw_load_count} != 1")
        if int(b.stats.preview_step_count) != len(env) * N_TF:
            _stop(
                f"{sym} preview {b.stats.preview_step_count} != "
                f"rows*n_tf {len(env) * N_TF}"
            )
        env_rows_total += len(env)

        t1 = time.perf_counter()
        e = env.rename(columns={"data_object": "symbol"})[
            KEY + ["trading_day"] + FEATURE_COLS
        ].copy()
        e["decision_bar_index"] = e["decision_bar_index"].astype("int64")

        o1 = r1[r1["symbol"] == sym]
        o2 = r2[r2["symbol"] == sym]
        env_keys = set(zip(
            e["decision_bar_index"].tolist(),
            e["decision_time"].astype("int64").tolist(),
        ))
        for tag, o in (("R1", o1), ("R2", o2)):
            orc_keys = set(zip(
                o["decision_bar_index"].tolist(),
                o["decision_time"].astype("int64").tolist(),
            ))
            only = orc_keys - env_keys
            if only:
                _stop(
                    f"STOP_ORACLE_KEYS_NOT_SUBSET {sym}/{tag}: "
                    f"oracle_only_rows={len(only)}"
                )

        m = e.merge(o1, on=KEY, how="inner").merge(
            o2[KEY + ["baseline_stable_action", "joint_retention_stable"]],
            on=KEY, how="inner",
        )
        parts.append(m)
        t_join += time.perf_counter() - t1
        print(
            f"[{i}/{len(symbols)}] {sym} env_rows={len(env)} "
            f"joined={len(m)} env_sec={time.perf_counter() - t0:.2f}",
            flush=True,
        )

    joined = pd.concat(parts, ignore_index=True)
    return joined, dict(
        environment_rows=env_rows_total,
        env_sec=t_env,
        join_sec=t_join,
    )


# --------------------------------------------------------------------------- #
# targets                                                                      #
# --------------------------------------------------------------------------- #
def make_targets(joined):
    j = joined
    j["QL"] = j["QL_24_ATR"].astype(float)
    j["QS"] = j["QS_24_ATR"].astype(float)
    j["QW"] = j["QW_24_ATR"].astype(float)
    j["Y_opp"] = np.maximum(j["QL"], j["QS"]) - j["QW"]
    j["Y_dir"] = j["QL"] - j["QS"]
    j["LabelAvailableTime"] = pd.concat(
        [
            j["label_available_time_6"],
            j["label_available_time_12"],
            j["label_available_time_24"],
        ],
        axis=1,
    ).max(axis=1)
    a = j["stable_action"].astype(str)
    j["Y_trade"] = np.where(a.isin(["Long", "Short"]), 1.0, np.where(a == "Wait", 0.0, np.nan))
    j["Y_long"] = np.where(a == "Long", 1.0, np.where(a == "Short", 0.0, np.nan))
    return j


# --------------------------------------------------------------------------- #
# metrics                                                                      #
# --------------------------------------------------------------------------- #
def clf_metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    out = dict(n=int(len(y)), class_share=float(np.mean(y)))
    out["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
    try:
        out["roc_auc"] = float(roc_auc_score(y, p))
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y, p))
    except Exception:
        out["pr_auc"] = float("nan")
    out["balanced_accuracy"] = float(balanced_accuracy_score(y, (p >= 0.5).astype(int)))
    out["brier"] = float(brier_score_loss(y, p))
    return out


def reg_metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    out = dict(n=int(len(y)))
    try:
        out["r2"] = float(r2_score(y, p))
    except Exception:
        out["r2"] = float("nan")
    out["mae"] = float(mean_absolute_error(y, p))
    try:
        out["spearman"] = float(spearmanr(y, p).statistic)
    except Exception:
        out["spearman"] = float("nan")
    return out


def prior_vector(train_sym, y_train, eval_sym):
    rate = float(np.mean(y_train))
    per = pd.Series(y_train).groupby(np.asarray(train_sym)).mean().to_dict()
    return np.array([per.get(s, rate) for s in eval_sym], dtype=float), rate


# --------------------------------------------------------------------------- #
# preprocessing (fit on train only)                                            #
# --------------------------------------------------------------------------- #
def make_preprocessor():
    cont = Pipeline([
        ("imp", SimpleImputer(strategy="median", add_indicator=True)),
        ("sc", StandardScaler()),
    ])
    disc = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("c", cont, CONT_COLS), ("d", disc, DISC_COLS)])


# --------------------------------------------------------------------------- #
# deciles                                                                      #
# --------------------------------------------------------------------------- #
def decile_table(df, pred_col, actual_col, extra_rate_cols, label):
    d = df[[pred_col, actual_col] + extra_rate_cols].dropna(subset=[pred_col]).copy()
    if len(d) < 10:
        return []
    try:
        d["_bin"] = pd.qcut(d[pred_col], 10, labels=False, duplicates="drop")
    except Exception:
        return []
    rows = []
    for bval, g in d.groupby("_bin"):
        rec = dict(
            task=label, decile=int(bval), n=int(len(g)),
            predicted_mean=float(g[pred_col].mean()),
            actual_mean=float(g[actual_col].mean()),
        )
        for c in extra_rate_cols:
            rec[c + "_rate"] = float(np.mean(g[c])) if len(g) else float("nan")
        rows.append(rec)
    return rows


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(SYMBOLS))
    ap.add_argument(
        "--limit-bars", type=int, default=None,
        help="SMOKE ONLY. Note: truncating the environment trips the "
             "oracle_only_rows==0 gate, so smoke uses full history.",
    )
    ap.add_argument("--out-dir", default=os.path.join(
        _REPO_ROOT, "artifacts", "environment_oracle_signal_v1"))
    ap.add_argument("--joined-name", default="environment_oracle_joined_v1.parquet")
    ap.add_argument("--results-name", default="environment_oracle_signal_results_v1.json")
    ap.add_argument("--deciles-name", default="environment_oracle_signal_deciles_v1.csv")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    t_all = time.perf_counter()

    r1, r2 = load_oracle()
    joined, counters = build_joined(symbols, r1, r2, args.limit_bars)
    joined = make_targets(joined)

    oracle_rows = int(len(r1[r1["symbol"].isin(symbols)]))
    env_only = counters["environment_rows"] - len(joined)
    print(
        f"[join] environment_rows={counters['environment_rows']} "
        f"oracle_rows={oracle_rows} matched={len(joined)} "
        f"env_only={env_only} oracle_only=0",
        flush=True,
    )

    # ---- baseline label consistency ----
    a1 = joined["stable_action"].astype(str)
    a2 = joined["baseline_stable_action"].astype(str)
    mismatch_all = int((a1 != a2).sum())
    used_mask = a1.isin(TRADE_ACTIONS) | a2.isin(TRADE_ACTIONS)
    mismatch_used = int((a1[used_mask] != a2[used_mask]).sum())
    print(
        f"[label] vocab R1={sorted(a1.unique().tolist())} "
        f"R2={sorted(a2.unique().tolist())}",
        flush=True,
    )
    print(
        f"[label] stable_action vs baseline mismatch: all={mismatch_all} "
        f"used_vocab={mismatch_used}",
        flush=True,
    )
    if mismatch_used != 0:
        _stop(f"STOP_ORACLE_BASELINE_MISMATCH used_vocab mismatch={mismatch_used}")

    # ---- chronological split by unique trading days ----
    days = np.sort(
        pd.to_datetime(joined["trading_day"]).dt.normalize().unique()
    )
    n_days = len(days)
    n_tr = int(n_days * 0.6)
    n_va = int(n_days * 0.2)
    tr_days, va_days, te_days = (
        days[:n_tr], days[n_tr:n_tr + n_va], days[n_tr + n_va:]
    )
    tr_start, tr_end = tr_days[0], tr_days[-1]
    va_start, va_end = va_days[0], va_days[-1]
    te_start, te_end = te_days[0], te_days[-1]

    dcol = pd.to_datetime(joined["trading_day"]).dt.normalize()
    m_tr = dcol.isin(tr_days)
    m_va = dcol.isin(va_days)
    m_te = dcol.isin(te_days)
    print(
        f"[split] train {pd.Timestamp(tr_start).date()}.."
        f"{pd.Timestamp(tr_end).date()} n={int(m_tr.sum())} | "
        f"val {pd.Timestamp(va_start).date()}..{pd.Timestamp(va_end).date()} "
        f"n={int(m_va.sum())} | "
        f"test {pd.Timestamp(te_start).date()}..{pd.Timestamp(te_end).date()} "
        f"n={int(m_te.sum())}",
        flush=True,
    )

    lat = joined["LabelAvailableTime"]
    results = dict(
        task_id="FUTURE-ENV-R3B-E0-SIGNAL-SANITY",
        symbols=symbols,
        join=dict(
            environment_rows=int(counters["environment_rows"]),
            oracle_rows=oracle_rows,
            matched_rows=int(len(joined)),
            environment_only_rows=int(env_only),
            oracle_only_rows=0,
        ),
        label=dict(
            r1_vocabulary=sorted(a1.unique().tolist()),
            r2_vocabulary=sorted(a2.unique().tolist()),
            mismatch_all_rows=mismatch_all,
            mismatch_used_vocabulary=mismatch_used,
        ),
        split=dict(
            train_start=str(tr_start), train_end=str(tr_end),
            val_start=str(va_start), val_end=str(va_end),
            test_start=str(te_start), test_end=str(te_end),
            n_train=int(m_tr.sum()), n_val=int(m_va.sum()), n_test=int(m_te.sum()),
        ),
        stages={},
    )

    deciles = []

    # ---- two walk-forward stages: evaluate VAL, then TEST ----
    stages = [
        ("VAL", m_tr, m_va, va_start),
        ("TEST", (m_tr | m_va), m_te, te_start),
    ]
    for stage_name, train_mask, eval_mask, boundary in stages:
        t_stage = time.perf_counter()
        tr = joined[train_mask]
        purge = lat[train_mask] >= boundary
        tr_p = joined[train_mask & ~purge]
        ev = joined[eval_mask]
        print(
            f"[{stage_name}] train={len(tr)} purged={int(purge.sum())} "
            f"train_used={len(tr_p)} eval={len(ev)}",
            flush=True,
        )

        t_pre = time.perf_counter()
        pre = make_preprocessor()
        Xtr = pre.fit_transform(tr_p[FEATURE_COLS])
        Xev = pre.transform(ev[FEATURE_COLS])
        t_pre = time.perf_counter() - t_pre

        t_mod = time.perf_counter()
        stage_res = dict(
            purged_rows=int(purge.sum()),
            train_used=int(len(tr_p)),
            eval_rows=int(len(ev)),
            preprocess_sec=round(t_pre, 3),
            model_sec=None,
        )

        for task, ycol, vocab in (
            ("TaskA_trade_vs_wait", "Y_trade", TRADE_ACTIONS),
            ("TaskB_direction", "Y_long", DIRECTION_ACTIONS),
        ):
            sub_tr = tr_p[tr_p["stable_action"].astype(str).isin(vocab)]
            sub_ev = ev[ev["stable_action"].astype(str).isin(vocab)]
            if sub_tr["Y_" + ycol.split("_")[1]].notna().sum() == 0 or len(sub_ev) == 0:
                continue
            ytr = sub_tr[ycol].to_numpy(float)
            yev = sub_ev[ycol].to_numpy(float)
            ok_tr = ~np.isnan(ytr)
            ok_ev = ~np.isnan(yev)
            if ok_tr.sum() == 0 or ok_ev.sum() == 0:
                continue
            Xtr_t = pre.transform(sub_tr[FEATURE_COLS])
            Xev_t = pre.transform(sub_ev[FEATURE_COLS])
            clf = LogisticRegression(max_iter=2000)
            clf.fit(Xtr_t[ok_tr], ytr[ok_tr].astype(int))
            p = clf.predict_proba(Xev_t[ok_ev])[:, 1]

            gp, grate = prior_vector(
                sub_tr["symbol"].to_numpy()[ok_tr], ytr[ok_tr],
                sub_ev["symbol"].to_numpy()[ok_ev],
            )
            sp, _ = prior_vector(
                sub_tr["symbol"].to_numpy()[ok_tr], ytr[ok_tr],
                sub_ev["symbol"].to_numpy()[ok_ev],
            )
            stage_res[task] = dict(
                n=int(ok_ev.sum()),
                global_prior=clf_metrics(yev[ok_ev], np.full(ok_ev.sum(), grate)),
                symbol_prior=clf_metrics(yev[ok_ev], sp),
                logistic=clf_metrics(yev[ok_ev], p),
            )

        for task, ycol in (("Yopp", "Y_opp"), ("Ydir", "Y_dir")):
            ytr = tr_p[ycol].to_numpy(float)
            yev = ev[ycol].to_numpy(float)
            ok_tr = np.isfinite(ytr)
            ok_ev = np.isfinite(yev)
            if ok_tr.sum() == 0 or ok_ev.sum() == 0:
                continue
            Xtr_r = pre.transform(tr_p[FEATURE_COLS])
            Xev_r = pre.transform(ev[FEATURE_COLS])
            rg = Ridge(alpha=1.0)
            rg.fit(Xtr_r[ok_tr], ytr[ok_tr])
            p = rg.predict(Xev_r[ok_ev])
            gmean = float(np.mean(ytr[ok_tr]))
            per = pd.Series(ytr[ok_tr]).groupby(
                tr_p["symbol"].to_numpy()[ok_tr]).mean().to_dict()
            sym_mean = np.array([
                per.get(s, gmean) for s in ev["symbol"].to_numpy()[ok_ev]
            ], dtype=float)
            stage_res[task] = dict(
                n=int(ok_ev.sum()),
                global_mean=reg_metrics(yev[ok_ev], np.full(ok_ev.sum(), gmean)),
                symbol_mean=reg_metrics(yev[ok_ev], sym_mean),
                ridge=reg_metrics(yev[ok_ev], p),
            )

            # deciles (prediction vs realised)
            dd = ev[ok_ev].copy()
            dd["_pred"] = p
            extra = (
                ["Y_trade"] if task == "Yopp" else ["Y_long"]
            )
            deciles += decile_table(
                dd.rename(columns={ycol: "_act"}), "_pred", "_act", extra,
                f"{stage_name}_{task}",
            )

        stage_res["model_sec"] = round(time.perf_counter() - t_mod, 3)
        results["stages"][stage_name] = stage_res

    # ---- per-symbol TEST ----
    m_te = dcol.isin(te_days)
    m_tr_va = m_tr | m_va
    tr2 = joined[m_tr_va & ~(lat >= te_start)]
    ev2 = joined[m_te]
    pre2 = make_preprocessor()
    Xtr2 = pre2.fit_transform(tr2[FEATURE_COLS])
    Xev2 = pre2.transform(ev2[FEATURE_COLS])

    per_symbol = {}
    for sym in symbols:
        rec = dict(object=sym)
        for task, ycol, vocab, mname in (
            ("TaskA", "Y_trade", TRADE_ACTIONS, "taskA"),
            ("TaskB", "Y_long", DIRECTION_ACTIONS, "taskB"),
        ):
            st = tr2[tr2["symbol"] == sym]
            se = ev2[ev2["symbol"] == sym]
            st = st[st["stable_action"].astype(str).isin(vocab)]
            se = se[se["stable_action"].astype(str).isin(vocab)]
            ytr = st[ycol].to_numpy(float)
            yev = se[ycol].to_numpy(float)
            ok_tr, ok_ev = ~np.isnan(ytr), ~np.isnan(yev)
            if ok_tr.sum() < 2 or ok_ev.sum() < 2 or len(np.unique(yev[ok_ev])) < 2:
                rec[mname] = dict(n=int(ok_ev.sum()), roc_auc=None, balanced_accuracy=None)
                continue
            clf = LogisticRegression(max_iter=2000)
            clf.fit(pre2.transform(st[FEATURE_COLS])[ok_tr], ytr[ok_tr].astype(int))
            p = clf.predict_proba(pre2.transform(se[FEATURE_COLS])[ok_ev])[:, 1]
            mt = clf_metrics(yev[ok_ev].astype(int), p)
            rec[mname] = dict(
                n=mt["n"], roc_auc=mt["roc_auc"], balanced_accuracy=mt["balanced_accuracy"]
            )
        st = tr2[tr2["symbol"] == sym]
        se = ev2[ev2["symbol"] == sym]
        ytr = st["Y_opp"].to_numpy(float)
        yev = se["Y_opp"].to_numpy(float)
        ok_tr, ok_ev = np.isfinite(ytr), np.isfinite(yev)
        if ok_tr.sum() >= 2 and ok_ev.sum() >= 2:
            rg = Ridge(alpha=1.0)
            rg.fit(pre2.transform(st[FEATURE_COLS])[ok_tr], ytr[ok_tr])
            mt = reg_metrics(yev[ok_ev], rg.predict(pre2.transform(se[FEATURE_COLS])[ok_ev]))
            rec["Yopp_ridge"] = dict(
                n=mt["n"], spearman=mt["spearman"], r2=mt["r2"]
            )
        else:
            rec["Yopp_ridge"] = dict(n=int(ok_ev.sum()), spearman=None, r2=None)
        per_symbol[sym] = rec
    results["test_per_symbol"] = per_symbol

    # ---- robustness subset diagnostic (no retrain) ----
    rob = {}
    mask_rob = (ev2["joint_retention_stable"] >= 0.9).to_numpy()
    if mask_rob.sum() > 0:
        sub = ev2[mask_rob]
        for task, ycol, vocab in (
            ("TaskA", "Y_trade", TRADE_ACTIONS),
            ("TaskB", "Y_long", DIRECTION_ACTIONS),
        ):
            s = sub[sub["stable_action"].astype(str).isin(vocab)]
            y = s[ycol].to_numpy(float)
            ok = ~np.isnan(y)
            if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
                rob[task] = dict(n=int(ok.sum()))
                continue
            clf = LogisticRegression(max_iter=2000)
            st = tr2[tr2["stable_action"].astype(str).isin(vocab)]
            ytr = st[ycol].to_numpy(float)
            ok_tr = ~np.isnan(ytr)
            clf.fit(pre2.transform(st[FEATURE_COLS])[ok_tr], ytr[ok_tr].astype(int))
            p = clf.predict_proba(pre2.transform(s[FEATURE_COLS])[ok])[:, 1]
            rob[task] = clf_metrics(y[ok].astype(int), p)
        s = sub.copy()
        y = s["Y_opp"].to_numpy(float)
        ok = np.isfinite(y)
        if ok.sum() >= 2:
            rg = Ridge(alpha=1.0)
            ytr = tr2["Y_opp"].to_numpy(float)
            ok_tr = np.isfinite(ytr)
            rg.fit(pre2.transform(tr2[FEATURE_COLS])[ok_tr], ytr[ok_tr])
            rob["Yopp"] = reg_metrics(y[ok], rg.predict(pre2.transform(s[FEATURE_COLS])[ok]))
    results["test_robustness_ge_0.9"] = rob

    # ---- save ----
    jpath = os.path.join(args.out_dir, args.joined_name)
    joined.to_parquet(jpath, index=False)
    rpath = os.path.join(args.out_dir, args.results_name)
    with open(rpath, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    dpath = os.path.join(args.out_dir, args.deciles_name)
    pd.DataFrame(deciles).to_csv(dpath, index=False)

    results["runtime"] = dict(
        environment_sec=round(counters["env_sec"], 3),
        join_sec=round(counters["join_sec"], 3),
        total_sec=round(time.perf_counter() - t_all, 3),
    )
    with open(rpath, "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    print(f"[save] joined={jpath}", flush=True)
    print(f"[save] results={rpath}", flush=True)
    print(f"[save] deciles={dpath}", flush=True)
    print(
        f"[done] total_sec={results['runtime']['total_sec']} "
        f"env_sec={results['runtime']['environment_sec']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
