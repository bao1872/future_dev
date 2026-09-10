"""Step 11：CLEAN_CONTROL_3 sensitivity（预注册）。

control 定义收紧为：t0 / t0-1 / t0-2 三根内均无任何 source_tf OB_ENTERED
（只使用 control 时点及过去信息，合法）。重新匹配，只重复 H=12 的三个指标：
    max_directional_excursion_R
    forward_range_R
    hit_abs_2p5R
匹配设计其余部分与 v1.1 完全一致（含 pre_range caliper 0.50、K=3）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.ob_event_value.ev_contract_v1 import (
    ANCHOR_H, DAY_WINDOW, K_CONTROLS, MATCH_COLS, RESULTS, SYMBOLS,
    get_bars,
)
from research.ob_event_value.match_controls import (
    PRE_RANGE_CALIPER, _RANGE_IDX, attach_state, robust_scale_params, smd,
)
from research.ob_event_value.run_ob_event_value_v1 import outcome_arrays


def main():
    ev = pd.read_parquet(RESULTS / "event_bars.parquet")
    pool = pd.read_parquet(RESULTS / "control_pool.parquet")
    pool = pool[pool["clean3"]].reset_index(drop=True)
    print(f"[clean3] controls = {len(pool)}", flush=True)

    ev = attach_state(ev)
    day_ord_map = {}
    for sym in SYMBOLS:
        day = get_bars(sym)["trading_day"]
        for i, d in enumerate(sorted(set(day.tolist()))):
            day_ord_map[(sym, d)] = i
    ev["trading_day_ord"] = [
        day_ord_map[(s, d)] for s, d in
        zip(ev["symbol"], ev["trading_day"])]
    ok = np.ones(len(ev), dtype=bool)
    for c in MATCH_COLS + ["R0"]:
        ok &= np.isfinite(ev[c].to_numpy(float))
    ev = ev[ok].reset_index(drop=True)

    groups, buckets = {}, {}
    for sym in SYMBOLS:
        te = ev[ev["symbol"] == sym]
        cp = pool[pool["symbol"] == sym].copy()
        both = pd.concat([te[MATCH_COLS], cp[MATCH_COLS]],
                         ignore_index=True)
        med, scale = robust_scale_params(both)
        groups[sym] = (med.to_numpy(float), scale.to_numpy(float))
        X = (cp[MATCH_COLS].to_numpy(float) - med.to_numpy(float)) \
            / scale.to_numpy(float)
        cp["_scaled"] = list(X)
        for key, sub in cp.groupby(["session_type", "time_bucket_30m"]):
            buckets[(sym, key[0], key[1])] = (
                np.stack(sub["_scaled"].to_numpy()),
                sub["trading_day_ord"].to_numpy(),
                sub["control_id"].to_numpy(),
            )

    pairs = []
    for i, row in enumerate(ev.itertuples(index=False)):
        if i % 20000 == 0:
            print(f"  matching {i}/{len(ev)}", flush=True)
        med, scale = groups[row.symbol]
        xt = (np.asarray([getattr(row, c) for c in MATCH_COLS], float)
              - med) / scale
        key = (row.symbol, int(row.session_type),
               int(row.time_bucket_30m))
        if key not in buckets:
            continue
        Xc, dord, cid = buckets[key]
        m = (dord != row.trading_day_ord) & \
            (np.abs(dord - row.trading_day_ord) <= DAY_WINDOW) & \
            (np.abs(Xc[:, _RANGE_IDX] - xt[_RANGE_IDX])
             <= PRE_RANGE_CALIPER)
        if m.sum() < K_CONTROLS:
            continue
        Xs, cs = Xc[m], cid[m]
        dist = np.sqrt(np.sum((Xs - xt[None, :]) ** 2, axis=1))
        take = np.argpartition(dist, K_CONTROLS - 1)[:K_CONTROLS]
        take = take[np.argsort(dist[take])]
        for r in take:
            pairs.append(dict(event_bar_id=row.event_bar_id,
                              control_id=str(cs[r]),
                              match_distance=float(dist[r])))
    P = pd.DataFrame(pairs)
    P.to_parquet(RESULTS / "matching_pairs_clean3.parquet", index=False)

    matched = P["event_bar_id"].nunique()
    rate = matched / len(ev)
    print(f"\n  match_rate = {rate:.4f} ({matched}/{len(ev)})")

    oa = {s: outcome_arrays(get_bars(s)) for s in SYMBOLS}
    et = ev.set_index("event_bar_id")
    pi = pool.set_index("control_id")
    keys = ("max_directional_excursion_R", "forward_range_R",
            "hit_abs_2p5R")
    res = {}
    for k in keys:
        tv = np.full(len(P), np.nan)
        cv = np.full(len(P), np.nan)
        eid = P["event_bar_id"].to_numpy()
        cid = P["control_id"].to_numpy()
        esym = et.loc[eid, "symbol"].to_numpy()
        et0 = et.loc[eid, "t0"].to_numpy()
        csym = pi.loc[cid, "symbol"].to_numpy()
        ct0 = pi.loc[cid, "t0"].to_numpy()
        assert (esym == csym).all()
        for sym in SYMBOLS:
            m = esym == sym
            if not m.any():
                continue
            arr = oa[sym][ANCHOR_H][k]
            tv[m] = arr[et0[m]]
            cv[m] = arr[ct0[m]]
        d = pd.DataFrame(dict(eid=eid, t=tv, c=cv))
        g = d.groupby("eid", sort=False).mean()
        res[k] = (float(g["t"].mean()), float(g["c"].mean()),
                  float((g["t"] - g["c"]).mean()))

    rows = [dict(dataset="CLEAN_CONTROL_3", horizon=ANCHOR_H,
                 match_rate=round(rate, 4), n_pairs=int(matched))]
    for k in keys:
        t, c, dd = res[k]
        rows[0][f"{k}_ob"] = round(t, 4)
        rows[0][f"{k}_control"] = round(c, 4)
        rows[0][f"{k}_delta"] = round(dd, 4)
        rows[0][f"{k}_rel"] = round(t / c - 1, 4) if c > 0 else None
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "clean_control_3_sensitivity.csv", index=False,
               encoding="utf-8-sig")
    print("\n=== CLEAN_CONTROL_3 (H=12) ===")
    print(out.T.to_string(header=False))

    # balance re-check
    pr = P.merge(ev, on="event_bar_id", how="left", validate="many_to_one")
    pr = pr.merge(pool[["control_id"] + MATCH_COLS], on="control_id",
                  how="left", validate="many_to_one",
                  suffixes=("_t", "_c"))
    bl = [dict(variable=c,
               smd_after=round(smd(pr[f"{c}_t"], pr[f"{c}_c"]), 4))
          for c in MATCH_COLS]
    bd = pd.DataFrame(bl)
    bd.to_csv(RESULTS / "matching_balance_clean3.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== CLEAN_CONTROL_3 balance ===")
    print(bd.to_string(index=False))
    print(f"\n  max |SMD| = {bd['smd_after'].abs().max():.4f}")
    print("CLEAN_CONTROL_3_DONE")


if __name__ == "__main__":
    main()
