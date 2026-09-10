"""P2 + P3 + P4：Stage-2 matched placebo specificity test。

OUTCOME LOCKED：matching / coverage / balance 全部通过之前，
不计算任何 later_reclaim / BOS acceptance。
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from research.liquidity_state_machine.liquidity_specificity_placebo_v2 import (
    MATCH_COLS_NUM, PLACEBO_SEEDS, RESULTS, STAGE1_NUM, SymbolData,
)

K = 3
DAY_WINDOW = 120
CALIPER_CB = 0.50
BOOT = 500
SEED = 7
CB_IDX = MATCH_COLS_NUM.index("close_beyond_R")


def attach_trend(df, sd):
    if len(df) == 0:
        return df
    # 未发生 interaction 的行 interaction_time 为 NaT，不能进 merge_asof
    has = df["interaction_time"].notna().to_numpy()
    d = df[has].sort_values("interaction_time")
    if len(d) == 0:
        return df
    o = pd.merge_asof(d, sd.trend1h.sort_values("available_time"),
                      left_on="interaction_time",
                      right_on="available_time", direction="backward",
                      allow_exact_matches=True)
    o = pd.merge_asof(o.sort_values("interaction_time"),
                      sd.env.sort_values("available_time"),
                      left_on="interaction_time",
                      right_on="available_time", direction="backward",
                      allow_exact_matches=True)
    o = o.drop(columns=["available_time_x", "available_time_y"],
               errors="ignore")
    return pd.concat([o, df[~has]], ignore_index=True)


def _i(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return int(x)
    except (TypeError, ValueError):
        return None


def rel(a, b):
    a, b = _i(a), _i(b)
    return ("WITH_TREND" if a == b else "AGAINST_TREND") \
        if (a in (-1, 1) and b in (-1, 1)) else "UNKNOWN"


def robust_params(frame):
    med = frame[MATCH_COLS_NUM].median()
    s = (frame[MATCH_COLS_NUM].quantile(.75)
         - frame[MATCH_COLS_NUM].quantile(.25)).clip(lower=1e-8)
    return med.to_numpy(float), s.to_numpy(float)


def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.nanmean(a) - np.nanmean(b)
    p = np.sqrt(.5 * (np.nanvar(a, ddof=1) + np.nanvar(b, ddof=1)))
    return 0.0 if (not np.isfinite(p) or p < 1e-12) else float(m / p)


def stage2_pool(df):
    m = ((df["activation_state"] == "VALID_AHEAD")
         & (df["interaction_path"] == "CONTINUOUS_CROSS")
         & (df["penetrated"] == True)  # noqa
         & (df["stage1"] == "CLOSE_BEYOND"))
    return df[m & df[MATCH_COLS_NUM].notna().all(axis=1)].copy()


def match_one(tr, ctl, med, sc, buckets):
    sym = tr["symbol"]
    key = (sym, int(tr["session_type"]), int(tr["time_bucket_30m"]))
    if key not in buckets:
        return None
    Xc, dord, cid, cbz = buckets[key]
    m = ((dord != tr["day_ord"]) & (np.abs(dord - tr["day_ord"])
                                    <= DAY_WINDOW))
    xt = (pd.Series(tr)[MATCH_COLS_NUM].to_numpy(float) - med) / sc
    m = m & (np.abs(cbz - xt[CB_IDX]) <= CALIPER_CB)
    if m.sum() < K:
        return None
    Xs = Xc[m]
    dist = np.sqrt(np.sum((Xs - xt[None, :]) ** 2, axis=1))
    take = np.argpartition(dist, K - 1)[:K]
    take = take[np.argsort(dist[take])]
    return [(str(cid[m][t]), float(dist[t])) for t in take]


def resolve_stage2(sd, ti, side, lvl):
    pen_av = sd.t[ti] + pd.Timedelta(minutes=5)
    k0 = int(np.searchsorted(sd.ev_at, np.datetime64(pen_av), side="right"))
    for j in range(ti + 1, sd.n):
        if sd.disc[j]:
            return "ROLL_CENSORED", sd.t[j]
        later = (sd.cl[j] < lvl) if side == +1 else (sd.cl[j] > lvl)
        at = sd.t[j] + pd.Timedelta(minutes=5)
        k1 = int(np.searchsorted(sd.ev_at, np.datetime64(at), side="right"))
        acc = False
        if k1 > k0:
            acc = bool(((sd.ev_kind[k0:k1] == "BOS")
                        & (sd.ev_bias[k0:k1] == side)).any())
        if later and acc:
            return "AMBIGUOUS_SAME_TIMESTAMP", sd.t[j]
        if later:
            return "LATER_RECLAIM", sd.t[j]
        if acc:
            return "STRUCTURAL_ACCEPTANCE", sd.t[j]
    return "END_OF_DATA_CENSORED", None


def main():
    tr = pd.read_parquet(RESULTS / "true_interactions.parquet")
    pl = pd.read_parquet(RESULTS / "placebo_interactions.parquet")
    print(f"true={len(tr)} placebo={len(pl)}", flush=True)

    # ---- attach trend/env per symbol, then build Stage-2 pools ----
    trp, plp = [], []
    for sym in sorted(tr["symbol"].unique()):
        sd = SymbolData(sym)
        a = attach_trend(tr[tr["symbol"] == sym], sd)
        b = attach_trend(pl[pl["symbol"] == sym], sd)
        trp.append(a)
        plp.append(b)
        print(f"  [trend] {sym} ({time.perf_counter():.0f}s)", flush=True)
    tr = pd.concat(trp, ignore_index=True)
    pl = pd.concat(plp, ignore_index=True)

    for d in (tr, pl):
        d["sweep_vs_1h"] = [
            rel(s, t) for s, t in
            zip(d["side"], d.get("trend_struct_1h",
                                 pd.Series([np.nan] * len(d))))]
        d["env_align_1h"] = [
            (_i(e) in (-1, 1) and _i(t) in (-1, 1) and _i(e) == _i(t))
            for e, t in zip(d.get("env_direction_4h",
                                  pd.Series([np.nan] * len(d))),
                            d.get("trend_struct_1h",
                                  pd.Series([np.nan] * len(d))))]

    T = stage2_pool(tr)
    print(f"\nStage-2 TRUE pool = {len(T)}", flush=True)

    # day ordinal
    uday = {d: i for i, d in enumerate(sorted(set(tr["trading_day"]
                                                  .dropna().tolist())))}
    for d in (T, pl):
        d["day_ord"] = [uday.get(x, -1) for x in d["trading_day"]]

    med, sc = robust_params(pd.concat(
        [T[MATCH_COLS_NUM], stage2_pool(pl)[MATCH_COLS_NUM]],
        ignore_index=True))

    results = {}
    for rep in PLACEBO_SEEDS:
        C = stage2_pool(pl[pl["replica_id"] == rep])
        print(f"\n=== replica {rep}: control pool = {len(C)} ===",
              flush=True)
        # exact-match buckets
        buckets = {}
        for (sym, lt, side, swe), g in C.groupby(
                ["symbol", "liquidity_type", "side", "sweep_vs_1h"]):
            g = g.copy()
            X = (g[MATCH_COLS_NUM].to_numpy(float) - med) / sc
            key = (sym, lt, side, swe)
            buckets.setdefault(key, {})
            for (stt, tb), gg in g.groupby(["session_type",
                                            "time_bucket_30m"]):
                ix = g.index.get_indexer(gg.index)
                buckets[key][(int(stt), int(tb))] = (
                    X[ix], gg["day_ord"].to_numpy(),
                    gg["level_key"].to_numpy(), X[ix][:, CB_IDX])
        pairs, unmatched = [], 0
        for tr_row in T.itertuples(index=False):
            d_ = tr_row._asdict()
            key = (d_["symbol"], d_["liquidity_type"], d_["side"],
                   d_["sweep_vs_1h"])
            b = buckets.get(key, {})
            if (int(d_["session_type"]), int(d_["time_bucket_30m"])) not in b:
                unmatched += 1
                continue
            if key not in buckets:
                unmatched += 1
                continue
            # merge bucket lookup with full key
            got = match_one(d_, C, med, sc,
                            {(d_["symbol"], int(d_["session_type"]),
                              int(d_["time_bucket_30m"])):
                             b.get((int(d_["session_type"]),
                                    int(d_["time_bucket_30m"])))})
            if got is None:
                unmatched += 1
                continue
            for cid, dd in got:
                pairs.append(dict(treat_key=d_["level_key"],
                                  control_key=cid, replica=rep,
                                  distance=dd))
        P = pd.DataFrame(pairs)
        if len(P) == 0:
            print("  no pairs"); continue
        P.to_parquet(RESULTS / f"stage2_pairs_{rep}.parquet", index=False)
        rate = P["treat_key"].nunique() / len(T)
        print(f"  coverage = {rate:.4f} "
              f"({P['treat_key'].nunique()}/{len(T)})  "
              f"unique controls={P['control_key'].nunique()}")
        results[rep] = dict(pairs=P, coverage=rate,
                            unmatched=unmatched)
    (RESULTS / "match_summary.json").write_text(
        json.dumps({str(k): {kk: (float(vv) if isinstance(vv, float) else vv)
                             for kk, vv in v.items() if kk != "pairs"}
                    for k, v in results.items()}, indent=2),
        encoding="utf-8")
    print("\nMATCH_DONE")


if __name__ == "__main__":
    main()
