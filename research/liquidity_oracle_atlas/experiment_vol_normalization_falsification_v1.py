#!/usr/bin/env python3
"""
experiment_vol_normalization_falsification_v1
=============================================

R3C-E1.2 VOL-NORMALIZATION FALSIFICATION.

One question only:

    Is the VolRegime -> Y_opp relation real opportunity information, or an
    artifact of normalising the target by the SHORT-horizon ATR5?

Y_opp is currently  (max(QL,QS) - QW) / ATR5, and every observed Y_opp < 0.
With raw opportunity held fixed, Y_opp = -1/ATR5, so ATR5 up -> Y_opp up.
Since m5_vol_regime = log(ATR10_RMA / ATR200_RMA) is itself a short-horizon
volatility state, a strong VolRegime -> Y_opp link may be pure denominator
coupling.

This file does NOT modify Trend / SR / Liquidity field math, FIELD_EDGES,
weights, the Oracle DP, split, purge, Ridge alpha or Logistic. The frozen
E1.1 module is imported and reused verbatim for the field state.

Targets compared:
    Yopp_ATR5    current frozen target   (raw / ATR5)
    Yopp_PCT     raw / decision close    (no volatility denominator)
    Yopp_ATR200  raw / m5 ATR200 (slow)  (causal, different horizon)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.liquidity_oracle_atlas.build_forming_environment_v1 import (  # noqa: E402
    FormingEnvironmentBuilder,
)
import research.liquidity_oracle_atlas.experiment_field_representation_v1 as E1  # noqa: E402

_stop = E1._stop
KEY = E1.KEY
TF = E1.TF
VOL_COLS = E1.VOL_COLS

DIAG_COLS = ["atr", "atr_liq"]
DIAG_OUT = ["decision_close"] + [f"{tf}_{c}" for tf in TF for c in DIAG_COLS]

VARIANTS_E12 = {
    "V0_OLD96": E1.VARIANTS["V0_OLD96"],
    "V4_FIELD_CORE": E1.VARIANTS["V4_FIELD_CORE"],
    "V9_OLD96_VOL": E1.VARIANTS["V0_OLD96"] + VOL_COLS,
    "V7_FIELD_VOL": E1.VARIANTS["V7_FIELD_VOL"],
}
assert len(VARIANTS_E12["V0_OLD96"]) == 96
assert len(VARIANTS_E12["V4_FIELD_CORE"]) == 148
assert len(VARIANTS_E12["V9_OLD96_VOL"]) == 100
assert len(VARIANTS_E12["V7_FIELD_VOL"]) == 152

# hard gate: diagnostic columns must not enter any model feature set
for _c in DIAG_OUT:
    for _vn, _cols in VARIANTS_E12.items():
        if _c in _cols:
            _stop(f"DIAG_COLUMN_IN_FEATURES {_c} in {_vn}")

TARGETS = ["Yopp_ATR5", "Yopp_PCT", "Yopp_ATR200"]
DECILE_VARIANTS = {"V0_OLD96", "V4_FIELD_CORE", "V9_OLD96_VOL", "V7_FIELD_VOL"}

EXTRACT_E12 = sorted(set(E1.EXTRACT_COLS) | set(DIAG_COLS))


def build_field_environment_e12(symbol, max_bars=None):
    """Frozen field loop + diagnostic-only extra columns (atr / atr_liq)."""
    b = FormingEnvironmentBuilder(symbol, max_bars=max_bars)
    b.load_raw()
    b.prepare()

    base = b.base
    n = b.n
    seg_arr = base["segment"].to_numpy(np.int64)

    out = {
        "symbol": np.repeat(symbol, n),
        "decision_bar_index": np.arange(n, dtype=np.int64),
        "decision_time": (
            pd.DatetimeIndex(base["time"]).to_numpy() + np.timedelta64(5, "m")
        ),
        "trading_day": base["trading_day"].to_numpy(),
        "decision_close": base["close"].to_numpy(np.float64),
    }

    for tf, minutes in b.tf_minutes.items():
        state = E1.FieldIndicatorState(b.params)
        form = b._form[tf]
        seg_completed = b._seg_completed[tf]

        cur_seg = None
        ci = 0
        seg_list = []
        tf_arr = {
            c: np.full(n, np.nan, dtype=np.float64) for c in EXTRACT_E12
        }

        for i in range(n):
            seg = int(seg_arr[i])
            if seg != cur_seg:
                state.reset()
                cur_seg = seg
                ci = 0
                seg_list = seg_completed.get(seg, [])

            while ci < len(seg_list) and seg_list[ci][0] < i:
                bar = seg_list[ci][1]
                state.step(
                    ci, bar["open"], bar["high"], bar["low"], bar["close"],
                    emit_field=False,
                )
                ci += 1

            feats = state.preview(
                ci, form["open"][i], form["high"][i],
                form["low"][i], form["close"][i],
            )
            for c in EXTRACT_E12:
                if c in feats:
                    tf_arr[c][i] = feats[c]

        for c, arr in tf_arr.items():
            out[f"{tf}_{c}"] = arr

        if tf != "m5":
            expected = minutes // 5
            out[f"{tf}_maturity"] = np.minimum(
                form["n_base"].astype(np.float64) / float(expected), 1.0
            )

    return pd.DataFrame(out)


def old96_parity_e12(symbol="AG", max_bars=2000):
    """Prove the extended extractor still reproduces the frozen OLD96."""
    old_df, _ = (
        FormingEnvironmentBuilder(symbol, max_bars=max_bars)
        .load_raw().prepare().run(profile_memory=False)
    )
    new_df = build_field_environment_e12(symbol, max_bars=max_bars)
    if len(old_df) != len(new_df):
        _stop(f"PARITY_ROW_COUNT {len(old_df)} != {len(new_df)}")

    cont_cells = max_err = disc_cells = disc_mis = 0
    cont_cells = 0
    for c in VARIANTS_E12["V0_OLD96"]:
        bare = c.split("_", 1)[1]
        if bare in E1.DISCRETE_SUFFIXES:
            a = old_df[c].to_numpy()
            bb = new_df[c].to_numpy()
            disc_cells += int(len(a))
            disc_mis += int(np.sum(a != bb))
            continue
        a = old_df[c].to_numpy(float)
        bb = new_df[c].to_numpy(float)
        cont_cells += int(len(a))
        if not np.array_equal(np.isnan(a), np.isnan(bb)):
            _stop(f"PARITY_NAN_PATTERN {c}")
        m = ~(np.isnan(a) & np.isnan(bb))
        if m.any():
            max_err = max(max_err, float(np.max(np.abs(a[m] - bb[m]))))

    if disc_mis or max_err > 1e-9:
        _stop(f"PARITY_FAIL max_err={max_err} disc_mis={disc_mis}")
    return dict(rows=int(len(old_df)), continuous_cells=cont_cells,
                max_abs_error=max_err, discrete_cells=disc_cells,
                discrete_mismatch=disc_mis)


def load_oracle_e12():
    r1_path = os.path.join(_REPO_ROOT, E1.R1_REL)
    r2_path = os.path.join(_REPO_ROOT, E1.R2_REL)
    for p, expect, tag in ((r1_path, E1.R1_SHA, "R1.1"),
                           (r2_path, E1.R2_SHA, "R2")):
        if not os.path.exists(p):
            _stop(f"{tag} oracle parquet missing: {p}")
        got = E1.sha256_file(p)
        if got != expect:
            _stop(f"STOP_ORACLE_ARTIFACT_IDENTITY_MISMATCH {tag}")

    r1 = pd.read_parquet(r1_path, columns=[
        "symbol", "decision_bar_index", "decision_time", "stable_action",
        "atr5_t", "QL_24", "QS_24", "QW_24",
        "QL_24_ATR", "QS_24_ATR", "QW_24_ATR",
        "label_available_time_6", "label_available_time_12",
        "label_available_time_24",
    ])
    r2 = pd.read_parquet(r2_path, columns=[
        "symbol", "decision_bar_index", "decision_time",
        "baseline_stable_action", "joint_retention_stable",
    ])

    if int(r1.duplicated(subset=KEY).sum()) or int(r2.duplicated(subset=KEY).sum()):
        _stop("STOP_ORACLE_DUPLICATE_KEYS")
    if len(r1) != len(r2):
        _stop(f"STOP_R1_R2_KEY_MISMATCH {len(r1)} != {len(r2)}")
    k1 = set(map(tuple, r1[KEY].to_numpy(dtype=object).tolist()))
    k2 = set(map(tuple, r2[KEY].to_numpy(dtype=object).tolist()))
    if k1 != k2:
        _stop(f"STOP_R1_R2_KEY_MISMATCH r1_only={len(k1-k2)} r2_only={len(k2-k1)}")
    print(f"[oracle] rows={len(r1)} dup=0/0 key_sets_identical=True", flush=True)
    return r1, r2


def decile_rows(df, pred_col, actual_col, label):
    d = df[["symbol", pred_col, actual_col, "Y_trade"]].dropna(
        subset=[pred_col, actual_col]).copy()
    if len(d) < 10:
        return []
    try:
        d["_bin"] = pd.qcut(d[pred_col], 10, labels=False, duplicates="drop")
    except Exception:
        return []
    rows = []
    for bval, g in d.groupby("_bin"):
        rows.append(dict(
            variant=label, decile=int(bval), n=int(len(g)),
            predicted_mean=float(g[pred_col].mean()),
            actual_mean=float(g[actual_col].mean()),
            trade_rate=float(np.mean(g["Y_trade"])),
        ))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="AG,AU")
    ap.add_argument("--out-dir", default=os.path.join(
        _REPO_ROOT, "artifacts", "environment_vol_normalization_v1"))
    ap.add_argument("--results-name",
                    default="environment_vol_normalization_small_results_v1.json")
    ap.add_argument("--deciles-name",
                    default="environment_vol_normalization_small_deciles_v1.csv")
    ap.add_argument("--parity-bars", type=int, default=2000)
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    t_all = time.perf_counter()

    parity = old96_parity_e12("AG", args.parity_bars)
    print(f"[parity] {parity}", flush=True)

    r1, r2 = load_oracle_e12()

    parts = []
    env_rows_total = 0
    t_field = 0.0
    for i, sym in enumerate(symbols, start=1):
        t0 = time.perf_counter()
        env = build_field_environment_e12(sym, max_bars=None)
        t_field += time.perf_counter() - t0
        env_rows_total += len(env)

        o1 = r1[r1["symbol"] == sym]
        o2 = r2[r2["symbol"] == sym]
        env_keys = set(zip(
            env["decision_bar_index"].astype("int64").tolist(),
            env["decision_time"].astype("int64").tolist(),
        ))
        for tag, o in (("R1", o1), ("R2", o2)):
            orc = set(zip(
                o["decision_bar_index"].to_numpy().tolist(),
                o["decision_time"].astype("int64").to_numpy().tolist(),
            ))
            if orc - env_keys:
                _stop(f"STOP_ORACLE_KEYS_NOT_SUBSET {sym}/{tag}")

        full = env.merge(o1, on=KEY, how="inner").merge(
            o2[KEY + ["baseline_stable_action", "joint_retention_stable"]],
            on=KEY, how="inner",
        )
        parts.append(full)
        print(f"[{i}/{len(symbols)}] {sym} env_rows={len(env)} joined={len(full)}",
              flush=True)

    joined = pd.concat(parts, ignore_index=True)
    oracle_rows = int(len(r1[r1["symbol"].isin(symbols)]))

    # ---- §3 targets ----
    joined["Yopp_raw"] = (
        np.maximum(joined["QL_24"].to_numpy(float), joined["QS_24"].to_numpy(float))
        - joined["QW_24"].to_numpy(float)
    )
    joined["Yopp_ATR5"] = (
        np.maximum(joined["QL_24_ATR"], joined["QS_24_ATR"]) - joined["QW_24_ATR"]
    )
    joined["Yopp_ATR5_rebuilt"] = joined["Yopp_raw"] / joined["atr5_t"]

    a = joined["Yopp_ATR5"].to_numpy(float)
    bb = joined["Yopp_ATR5_rebuilt"].to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(bb)
    recon_err = float(np.max(np.abs(a[ok] - bb[ok]))) if ok.any() else float("nan")
    if not (recon_err <= 1e-10):
        _stop(f"ATR5_RECONSTRUCTION_MAX_ERR {recon_err:.3e} > 1e-10")
    print(f"[gate] ATR5 reconstruction max_err={recon_err:.3e}", flush=True)

    c = joined["decision_close"].to_numpy(float)
    joined["Yopp_PCT"] = np.where(
        np.isfinite(c) & (c != 0), joined["Yopp_raw"] / c, np.nan)
    a200 = joined["m5_atr"].to_numpy(float)
    joined["Yopp_ATR200"] = np.where(
        np.isfinite(a200) & (a200 > 0), joined["Yopp_raw"] / a200, np.nan)
    joined["atr5_vs_atr200_log"] = np.log(
        joined["atr5_t"].to_numpy(float) / a200)

    # frozen labels (only Y_trade / Y_long needed for deciles)
    act = joined["stable_action"].astype(str)
    joined["Y_trade"] = np.where(
        act.isin(["Long", "Short"]), 1.0, np.where(act == "Wait", 0.0, np.nan))
    joined["Y_long"] = np.where(
        act == "Long", 1.0, np.where(act == "Short", 0.0, np.nan))

    # ---- split / purge (frozen) ----
    days = np.sort(pd.to_datetime(joined["trading_day"]).dt.normalize().unique())
    n_d = len(days)
    n_tr, n_va = int(n_d * 0.6), int(n_d * 0.2)
    tr_days = days[:n_tr]
    va_days = days[n_tr:n_tr + n_va]
    te_days = days[n_tr + n_va:]
    dcol = pd.to_datetime(joined["trading_day"]).dt.normalize()
    m_tr, m_va, m_te = dcol.isin(tr_days), dcol.isin(va_days), dcol.isin(te_days)
    joined["LabelAvailableTime"] = pd.concat([
        joined["label_available_time_6"],
        joined["label_available_time_12"],
        joined["label_available_time_24"],
    ], axis=1).max(axis=1)
    lat = joined["LabelAvailableTime"]
    va_start, te_start = va_days[0], te_days[0]

    print(
        f"[split] train n={int(m_tr.sum())} val n={int(m_va.sum())} "
        f"test n={int(m_te.sum())}",
        flush=True,
    )

    # ---- §5 denominator coupling ----
    te = joined[m_te]
    coupling = float(spearmanr(
        te["m5_vol_regime_log_ratio"], te["atr5_vs_atr200_log"],
        nan_policy="omit").statistic)

    # ---- §6 per-symbol univariate ----
    vol_cols = [f"{tf}_vol_regime_log_ratio" for tf in TF]
    uni = {}
    scopes = [("ALL", m_te)]
    for s in symbols:
        scopes.append((s, m_te & (joined["symbol"] == s).to_numpy()))
    for scope, mask in scopes:
        uni[scope] = {}
        for vc in vol_cols:
            x = joined.loc[mask, vc].to_numpy(float)
            uni[scope][vc] = {}
            for tc in TARGETS + ["Yopp_raw"]:
                y = joined.loc[mask, tc].to_numpy(float)
                okk = np.isfinite(x) & np.isfinite(y)
                uni[scope][vc][tc] = (
                    round(float(spearmanr(x[okk], y[okk]).statistic), 4)
                    if int(okk.sum()) >= 3 else None
                )

    # ---- §7-§10 models: Ridge only, 3 targets x 4 variants ----
    stages = [("VAL", m_tr, m_va, va_start), ("TEST", (m_tr | m_va), m_te, te_start)]
    results = dict(
        task_id="FUTURE-ENV-R3C-E1.2-VOL-NORMALIZATION-FALSIFICATION",
        symbols=symbols,
        old96_parity=parity,
        atr5_reconstruction_max_err=recon_err,
        join=dict(environment_rows=int(env_rows_total), oracle_rows=oracle_rows,
                  matched_rows=int(len(joined)),
                  environment_only_rows=int(env_rows_total - len(joined)),
                  oracle_only_rows=0),
        vol_vs_atr_ratio_spearman_TEST=coupling,
        univariate=uni,
        models={},
    )

    deciles = []
    for target in TARGETS:
        results["models"][target] = {}
        for vname, cols in VARIANTS_E12.items():
            vres = {}
            for stage_name, train_mask, eval_mask, boundary in stages:
                tr_p = joined[train_mask & ~(lat >= boundary)]
                ev = joined[eval_mask]
                pre = E1.make_preprocessor(cols)
                Xtr = pre.fit_transform(tr_p[cols])
                Xev = pre.transform(ev[cols])

                ytr = tr_p[target].to_numpy(float)
                yev = ev[target].to_numpy(float)
                mtr, mev = np.isfinite(ytr), np.isfinite(yev)
                if mtr.sum() == 0 or mev.sum() == 0:
                    continue
                rg = Ridge(alpha=1.0)
                rg.fit(Xtr[mtr], ytr[mtr])
                p = rg.predict(Xev[mev])
                mt = E1.reg_metrics(yev[mev], p)
                vres[stage_name] = dict(n=mt["n"], spearman=mt["spearman"],
                                        r2=mt["r2"], mae=mt["mae"])

                if (stage_name == "TEST" and vname in DECILE_VARIANTS
                        and target in ("Yopp_PCT", "Yopp_ATR200")):
                    full = np.full(len(ev), np.nan)
                    full[np.flatnonzero(mev)] = p
                    dd = ev.copy()
                    dd["_pred"] = full
                    deciles += decile_rows(dd, "_pred", target, f"{vname}__{target}")

                del Xtr, Xev, pre
            results["models"][target][vname] = vres
            print(
                f"[model] {target:12s} {vname:15s} "
                f"TEST_sp={vres.get('TEST', {}).get('spearman')}",
                flush=True,
            )

    # ---- §10 delta table ----
    def _sp(target, vname, stage):
        return results["models"][target][vname].get(stage, {}).get(
            "spearman", float("nan"))

    deltas = {}
    for target in TARGETS:
        deltas[target] = {}
        for stage in ("VAL", "TEST"):
            v0 = _sp(target, "V0_OLD96", stage)
            v4 = _sp(target, "V4_FIELD_CORE", stage)
            v9 = _sp(target, "V9_OLD96_VOL", stage)
            v7 = _sp(target, "V7_FIELD_VOL", stage)
            deltas[target][stage] = dict(
                field_effect_V4_minus_V0=v4 - v0,
                vol_on_old96_V9_minus_V0=v9 - v0,
                vol_on_field_V7_minus_V4=v7 - v4,
            )
    results["deltas"] = deltas
    results["runtime"] = dict(
        field_environment_sec=round(t_field, 3),
        total_sec=round(time.perf_counter() - t_all, 3),
    )

    rpath = os.path.join(args.out_dir, args.results_name)
    with open(rpath, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    dpath = os.path.join(args.out_dir, args.deciles_name)
    pd.DataFrame(deciles).to_csv(dpath, index=False)

    print(f"[coupling] rho(m5_vol, log(ATR5/ATR200)) = {coupling:.4f}", flush=True)
    print(f"[save] {rpath}", flush=True)
    print(f"[save] {dpath}", flush=True)
    print(f"[done] total_sec={results['runtime']['total_sec']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
