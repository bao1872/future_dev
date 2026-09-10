"""Step 4-6：Exact + nearest-neighbour matching，随后两道 Gate。

匹配只允许使用 <= t0-1 的事前状态；K=3，with replacement；
先做 COVERAGE GATE 再做 BALANCE GATE，两道都过了才允许看 outcome。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.ob_event_value.ev_contract_v1 import (
    DAY_WINDOW, K_CONTROLS, MATCH_COLS, RESULTS, SYMBOLS, get_bars,
    pre_event_state,
)


def robust_scale_params(frame: pd.DataFrame):
    med = frame[MATCH_COLS].median()
    q25 = frame[MATCH_COLS].quantile(0.25)
    q75 = frame[MATCH_COLS].quantile(0.75)
    scale = (q75 - q25).clip(lower=1e-8)
    return med, scale


def smd(treated, control) -> float:
    treated = np.asarray(treated, float)
    control = np.asarray(control, float)
    mt, mc = np.nanmean(treated), np.nanmean(control)
    vt = np.nanvar(treated, ddof=1)
    vc = np.nanvar(control, ddof=1)
    pooled = np.sqrt(0.5 * (vt + vc))
    if pooled < 1e-12 or not np.isfinite(pooled):
        return 0.0
    return float((mt - mc) / pooled)


def attach_state(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.copy()
    for c in MATCH_COLS + ["R0"]:
        ev[c] = np.nan
    for sym, sub in ev.groupby("symbol"):
        bars = get_bars(sym)
        st = pre_event_state(bars)
        idx = sub.index.to_numpy()
        t0s = sub["t0"].to_numpy()
        for c in MATCH_COLS + ["R0"]:
            ev.loc[idx, c] = st[c][t0s]
        ev.loc[idx, "session_type"] = bars["session_type"][t0s]
        ev.loc[idx, "time_bucket_30m"] = bars["time_bucket_30m"][t0s]
        ev.loc[idx, "trading_day"] = bars["trading_day"][t0s]
    return ev


def main():
    ev = pd.read_parquet(RESULTS / "event_bars.parquet")
    pool = pd.read_parquet(RESULTS / "control_pool.parquet")

    # trading_day_ord for treatments
    day_ord_map = {}
    for sym in SYMBOLS:
        day = get_bars(sym)["trading_day"]
        ud = {d: i for i, d in enumerate(sorted(set(day.tolist())))}
        for d, i in ud.items():
            day_ord_map[(sym, d)] = i

    ev = attach_state(ev)
    ev["trading_day_ord"] = [
        day_ord_map[(s, d)] for s, d in
        zip(ev["symbol"], ev["trading_day"])]

    ok = np.ones(len(ev), dtype=bool)
    for c in MATCH_COLS + ["R0"]:
        ok &= np.isfinite(ev[c].to_numpy(float))
    ev = ev[ok].reset_index(drop=True)
    print(f"[match] treatments usable = {len(ev)}", flush=True)

    # ---- robust scale：每个 symbol，用 treatment+control 合并的合法事前状态 ----
    groups = {}
    for sym in SYMBOLS:
        te = ev[ev["symbol"] == sym]
        cp = pool[pool["symbol"] == sym]
        both = pd.concat([te[MATCH_COLS], cp[MATCH_COLS]],
                         ignore_index=True)
        med, scale = robust_scale_params(both)
        groups[sym] = (med.to_numpy(float), scale.to_numpy(float))

        cp = cp.copy()
        X = (cp[MATCH_COLS].to_numpy(float) - med.to_numpy(float)) \
            / scale.to_numpy(float)
        cp["_scaled"] = list(X)
        groups[sym + "__pool"] = cp

    # 预分组：symbol|session_type|time_bucket
    buckets = {}
    for sym in SYMBOLS:
        cp = groups[sym + "__pool"]
        for key, sub in cp.groupby(["session_type", "time_bucket_30m"]):
            buckets[(sym, key[0], key[1])] = (
                np.stack(sub["_scaled"].to_numpy()),
                sub["trading_day_ord"].to_numpy(),
                sub["control_id"].to_numpy(),
            )

    pairs, unmatched = [], 0
    for i, row in enumerate(ev.itertuples(index=False)):
        if i % 10000 == 0:
            print(f"  matching {i}/{len(ev)}", flush=True)
        med, scale = groups[row.symbol]
        xt = (np.asarray([getattr(row, c) for c in MATCH_COLS], float)
              - med) / scale
        key = (row.symbol, int(row.session_type),
               int(row.time_bucket_30m))
        if key not in buckets:
            unmatched += 1
            continue
        Xc, dord, cid = buckets[key]
        m = (dord != row.trading_day_ord) & \
            (np.abs(dord - row.trading_day_ord) <= DAY_WINDOW)
        if m.sum() < K_CONTROLS:
            unmatched += 1
            continue
        Xs = Xc[m]
        cs = cid[m]
        dist = np.sqrt(np.sum((Xs - xt[None, :]) ** 2, axis=1))
        take = np.argpartition(dist, K_CONTROLS - 1)[:K_CONTROLS]
        take = take[np.argsort(dist[take])]
        for r, c, dd in zip(take, cs[take], dist[take]):
            pairs.append(dict(event_bar_id=row.event_bar_id,
                              control_id=str(c),
                              match_distance=float(dd)))
    P = pd.DataFrame(pairs)
    P.to_parquet(RESULTS / "matching_pairs.parquet", index=False)

    # ---- COVERAGE GATE ----
    matched = P["event_bar_id"].nunique()
    cov_rows = [dict(维度="ALL", treatments_total=len(ev),
                     treatments_matched=matched,
                     match_rate=round(matched / len(ev), 4),
                     unique_controls=int(P["control_id"].nunique()))]
    for tf, sub in ev.groupby("source_tf"):
        mm = P[P["event_bar_id"].isin(set(sub["event_bar_id"]))]
        cov_rows.append(dict(
            维度=tf, treatments_total=len(sub),
            treatments_matched=int(mm["event_bar_id"].nunique()),
            match_rate=round(mm["event_bar_id"].nunique() / len(sub), 4),
            unique_controls=int(mm["control_id"].nunique())))
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(RESULTS / "matching_coverage.csv", index=False,
               encoding="utf-8-sig")
    print("\n=== MATCHING COVERAGE ===")
    print(cov.to_string(index=False), flush=True)

    gate_ok = bool(
        (cov.loc[cov["维度"] == "ALL", "match_rate"].iloc[0] >= 0.85)
        and (cov[cov["维度"] != "ALL"]["match_rate"] >= 0.80).all())
    print(f"\nMATCHING_GATE: {'PASS' if gate_ok else 'FAIL'}")

    # ---- BALANCE GATE ----
    pr = P.merge(ev, on="event_bar_id", how="left", validate="many_to_one")
    pr = pr.merge(pool[["control_id"] + MATCH_COLS], on="control_id",
                  how="left", validate="many_to_one",
                  suffixes=("_t", "_c"))
    bal = []
    for c in MATCH_COLS:
        bal.append(dict(variable=c, dimension="ALL",
                        smd_before=round(smd(ev[c], pool[c]), 4),
                        smd_after=round(smd(pr[f"{c}_t"], pr[f"{c}_c"]), 4)))
    # source_tf 不属于 MATCH_COLS，merge 后不带后缀
    for tf, sub in pr.groupby("source_tf"):
        et = ev[ev["source_tf"] == tf]
        for c in MATCH_COLS:
            bal.append(dict(
                variable=c, dimension=tf, smd_before=np.nan,
                smd_after=round(smd(sub[f"{c}_t"], sub[f"{c}_c"]), 4)))
    bdf = pd.DataFrame(bal)
    bdf.to_csv(RESULTS / "matching_balance.csv", index=False,
               encoding="utf-8-sig")
    print("\n=== BALANCE (SMD) ===")
    print(bdf[bdf["dimension"] == "ALL"].to_string(index=False))
    print("\nby source_tf (after): max |SMD|")
    for tf, sub in bdf[bdf["dimension"] != "ALL"].groupby("dimension"):
        print(f"  {tf}: {sub['smd_after'].abs().max():.4f}")

    bal_ok = bool(
        (bdf[bdf["dimension"] == "ALL"]["smd_after"].abs() <= 0.10).all()
        and (bdf[bdf["dimension"] != "ALL"]
             .groupby("dimension")["smd_after"]
             .apply(lambda s: s.abs().max() <= 0.15).all()))
    print(f"\nBALANCE_GATE: {'PASS' if bal_ok else 'FAIL'}")

    reuse = (P.groupby("control_id").size().rename("times_used")
              .reset_index())
    rc = reuse.groupby("times_used").size().rename("n_controls").reset_index()
    rc.to_csv(RESULTS / "control_reuse.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== control reuse ===")
    print(rc.to_string(index=False))

    (RESULTS / "matching_gate.json").write_text(
        '{"matching_gate": %s, "balance_gate": %s}'
        % (str(gate_ok).lower(), str(bal_ok).lower()), encoding="utf-8")
    print("\nMATCH_CONTROLS_DONE")


if __name__ == "__main__":
    main()
