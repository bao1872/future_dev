"""v1.2 画像：Stage1 / Stage2、HTF 检验、canonical trading_day bootstrap。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import research.m2_nondeep_temporal_v1 as m2

RESULTS = "research/analysis_results/liquidity_state_machine_v1"
BOOT = 500
SEED = 7


def rel(a, b):
    return ("WITH_TREND" if a == b else "AGAINST_TREND") \
        if (a in (-1, 1) and b in (-1, 1)) else "UNKNOWN"


def boot_diff(df, mask_a, mask_b, col, n=BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    days = df["trading_day"].to_numpy()
    ud = np.asarray(sorted(set(days)))
    idx = {d: np.flatnonzero(days == d) for d in ud}
    v = df[col].to_numpy(float)
    ma, mb = np.asarray(mask_a), np.asarray(mask_b)
    ds, rrs = [], []
    for _ in range(n):
        pick = rng.choice(ud, size=len(ud), replace=True)
        r = np.concatenate([idx[d] for d in pick])
        a, b = v[r][ma[r]], v[r][mb[r]]
        if not (len(a) and len(b)):
            continue
        pa, pb = a.mean(), b.mean()
        ds.append(pa - pb)
        rrs.append(pa / pb if pb > 0 else np.nan)
    return dict(ci=[round(float(np.nanquantile(ds, .025)), 4),
                    round(float(np.nanquantile(ds, .975)), 4)],
                rr_ci=[round(float(np.nanquantile(rrs, .025)), 4),
                       round(float(np.nanquantile(rrs, .975)), 4)])


def cmp2(df, mask_a, mask_b, col, label):
    a, b = df[mask_a], df[mask_b]
    pa = float(a[col].mean()) if len(a) else np.nan
    pb = float(b[col].mean()) if len(b) else np.nan
    o = dict(metric=label, n_a=len(a), n_b=len(b),
             p_a=round(pa, 4), p_b=round(pb, 4),
             pp_diff=round(pa - pb, 4),
             relative_risk=round(pa / pb, 4) if pb else None)
    o.update(boot_diff(df, np.asarray(mask_a), np.asarray(mask_b), col))
    return o


def main():
    it = pd.read_parquet(f"{RESULTS}/interactions_v1_2.parquet")
    old = pd.read_parquet(f"{RESULTS}/interactions_primary.parquet")
    pen = it[it["penetrated"]].copy()
    cb = pen[pen["stage1"] == "CLOSE_BEYOND"].copy()

    pen["_sbr"] = (pen["stage1"] == "SAME_BAR_RECLAIM").astype(float)
    pen["_cb"] = (pen["stage1"] == "CLOSE_BEYOND").astype(float)
    cb["_lr"] = (cb["stage2"] == "LATER_RECLAIM").astype(float)
    cb["_sa"] = (cb["stage2"] == "STRUCTURAL_ACCEPTANCE").astype(float)

    n_pen = len(pen)
    n_sb = int(pen["_sbr"].sum())
    n_cb = int(pen["_cb"].sum())
    n_lr = int(cb["_lr"].sum())
    n_sa = int(cb["_sa"].sum())
    old_reclaim = int(old["reclaimed"].sum())

    print("=== Stage 1 ===")
    print(f"PENETRATION                 {n_pen}")
    print(f"SAME_BAR_RECLAIM            {n_sb}  ({n_sb/n_pen*100:.2f}%)")
    print(f"CLOSE_BEYOND                {n_cb}  ({n_cb/n_pen*100:.2f}%)")
    print(f"CLOSE_AT_LEVEL              "
          f"{int((pen['stage1']=='CLOSE_AT_LEVEL').sum())}")
    print(f"\nold RECLAIM = {old_reclaim};  same-bar 占 old reclaim = "
          f"{n_sb/old_reclaim*100:.2f}%")

    print("\n=== Stage 2 (CLOSE_BEYOND only) ===")
    print(f"LATER_RECLAIM             {n_lr}  ({n_lr/n_cb*100:.2f}%)")
    print(f"STRUCTURAL_ACCEPTANCE     {n_sa}  ({n_sa/n_cb*100:.2f}%)")
    print("bars_to_stage2: p25/median/p75/p90 =",
          [round(float(cb['bars_to_stage2'].dropna().quantile(q)), 1)
           for q in (.25, .5, .75, .90)])

    # trend relation
    for d in (pen, cb):
        d["sweep_vs_1h"] = [
            rel(int(p), int(t)) for p, t in
            zip(d["penetration_direction"], d["trend_struct_1h"])]
        d["env_align_1h"] = [
            (e in (-1, 1) and t in (-1, 1) and int(e) == int(t))
            for e, t in zip(d["env_direction_4h"], d["trend_struct_1h"])]

    ag = (pen["sweep_vs_1h"] == "AGAINST_TREND").to_numpy()
    wi = (pen["sweep_vs_1h"] == "WITH_TREND").to_numpy()
    cag = (cb["sweep_vs_1h"] == "AGAINST_TREND").to_numpy()
    cwi = (cb["sweep_vs_1h"] == "WITH_TREND").to_numpy()

    rows = [
        cmp2(pen, ag, wi, "_sbr", "H1: P(SAME_BAR_RECLAIM|pen) AGAINST vs WITH"),
        cmp2(cb, cag, cwi, "_lr", "H2: P(LATER_RECLAIM|close_beyond) AGAINST vs WITH"),
    ]
    bd = pd.DataFrame(rows)
    bd.to_csv(f"{RESULTS}/bootstrap_sequencing_deltas.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== H1 / H2 (canonical trading_day bootstrap 500) ===")
    print(bd.to_string(index=False))

    # 4h x 1h
    rr = []
    for al in (True, False):
        g = cb[cb["env_align_1h"] == al]
        p = pen[pen["env_align_1h"] == al]
        a1 = (g["sweep_vs_1h"] == "AGAINST_TREND").to_numpy()
        w1 = (g["sweep_vs_1h"] == "WITH_TREND").to_numpy()
        a0 = (p["sweep_vs_1h"] == "AGAINST_TREND").to_numpy()
        w0 = (p["sweep_vs_1h"] == "WITH_TREND").to_numpy()
        rr.append(dict(
            env4h_1h="ALIGNED" if al else "CONFLICT",
            n_pen=len(p),
            p_same_bar_against=round(float(p.loc[a0, "_sbr"].mean()), 4)
            if a0.any() else None,
            p_same_bar_with=round(float(p.loc[w0, "_sbr"].mean()), 4)
            if w0.any() else None,
            n_close_beyond=len(g),
            p_later_reclaim_against=round(float(g.loc[a1, "_lr"].mean()), 4)
            if a1.any() else None,
            p_later_reclaim_with=round(float(g.loc[w1, "_lr"].mean()), 4)
            if w1.any() else None))
    hd = pd.DataFrame(rr)
    hd.to_csv(f"{RESULTS}/sequencing_by_htf_relation.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== 4h env x 1h ===")
    print(hd.to_string(index=False))

    # post-reclaim strict
    rc = it[it.get("reclaim_available_time").notna()].copy() \
        if "reclaim_available_time" in it.columns else pd.DataFrame()
    if len(rc):
        print("\n=== post-reclaim (strict ordering) ===")
        print(rc["post_reclaim_state"].value_counts(
            dropna=False).to_string())
        rc["sweep_vs_1h"] = [
            rel(int(p), int(t)) for p, t in
            zip(rc["penetration_direction"], rc["trend_struct_1h"])]
        rc["_rev"] = (rc["post_reclaim_state"]
                      == "REVERSAL_MSS_CONFIRMED").astype(float)
        a2 = (rc["sweep_vs_1h"] == "AGAINST_TREND").to_numpy()
        w2 = (rc["sweep_vs_1h"] == "WITH_TREND").to_numpy()
        pr = pd.DataFrame([cmp2(rc, a2, w2, "_rev",
                                "post-reclaim REVERSAL_MSS AGAINST vs WITH")])
        pr.to_csv(f"{RESULTS}/post_reclaim_strict.csv", index=False,
                  encoding="utf-8-sig")
        print(pr.to_string(index=False))

    # stratifications
    def strat(df, col, name, colout):
        rs = []
        for k, g in df.groupby(col):
            rs.append(dict(**{name: str(k)}, n=len(g),
                           p=round(float(g[colout].mean()), 4)))
        t = pd.DataFrame(rs).sort_values("n", ascending=False)
        t.to_csv(f"{RESULTS}/seq_{name}.csv", index=False,
                 encoding="utf-8-sig")
        return t

    t1 = strat(pen, "symbol", "sbr_by_symbol", "_sbr")
    t2 = strat(cb, "symbol", "lr_by_symbol", "_lr")
    print("\n=== by symbol: later_reclaim | close_beyond ===")
    print(t2.to_string(index=False))
    print(f"  macro median={t2['p'].median():.4f}  "
          f"positive={int((t2['p'] > 0.5).sum())}/{len(t2)}")

    folds, _ = m2.build_folds(cb["trading_day"].to_numpy(), len(cb))
    fr = []
    for fi in range(4):
        te = folds[fi][2]
        g = cb[cb["trading_day"].isin(te)]
        p = pen[pen["trading_day"].isin(te)]
        fr.append(dict(fold=f"F{fi+1}", n_pen=len(p),
                       p_same_bar=round(float(p["_sbr"].mean()), 4),
                       n_cb=len(g),
                       p_later_reclaim=round(float(g["_lr"].mean()), 4)))
    fd = pd.DataFrame(fr)
    fd.to_csv(f"{RESULTS}/seq_by_fold.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== by fold (canonical trading_day) ===")
    print(fd.to_string(index=False))

    # cluster dedup
    it["cluster"] = (it["symbol"] + "|"
                     + it["interaction_time"].astype(str) + "|"
                     + it["level_price"].round(6).astype(str))
    d = it.drop_duplicates(subset=["cluster"])
    dp = d[d["penetrated"]]
    dc = dp[dp["stage1"] == "CLOSE_BEYOND"]
    cd = pd.DataFrame([dict(
        dataset="CLUSTER_DEDUP", n_pen=len(dp),
        p_same_bar=round(float((dp["stage1"]
                                == "SAME_BAR_RECLAIM").mean()), 4),
        n_close_beyond=len(dc),
        p_later_reclaim=round(float((dc["stage2"]
                                     == "LATER_RECLAIM").mean()), 4))])
    cd.to_csv(f"{RESULTS}/seq_cluster_dedup.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== cluster dedup ===")
    print(cd.to_string(index=False))

    # trading_day audit (night session)
    aud = it[["interaction_time", "trading_day"]].copy()
    aud["calendar_date"] = pd.to_datetime(
        aud["interaction_time"]).dt.strftime("%Y-%m-%d")
    mm = aud[aud["calendar_date"] != aud["trading_day"]]
    samp = pd.concat([mm.head(20), aud.head(5)]) if len(mm) else aud.head(20)
    samp.to_csv(f"{RESULTS}/trading_day_audit.csv", index=False,
                encoding="utf-8-sig")
    print(f"\n=== trading_day audit ===\n"
          f"  rows where calendar_date != trading_day: "
          f"{len(mm)} / {len(aud)} ({len(mm)/max(1,len(aud))*100:.2f}%)")
    print(samp.head(6).to_string(index=False))

    with open(f"{RESULTS}/AUDIT_V1_2.json", "w", encoding="utf-8") as f:
        json.dump(dict(
            penetrations=n_pen, same_bar_reclaim=n_sb,
            close_beyond=n_cb, close_at_level=int(
                (pen["stage1"] == "CLOSE_AT_LEVEL").sum()),
            old_reclaim=old_reclaim,
            same_bar_share_of_old_reclaim=round(n_sb / old_reclaim, 4),
            later_reclaim=n_lr, structural_acceptance=n_sa,
            p_later_reclaim_given_close_beyond=round(n_lr / n_cb, 4),
            p_structural_acceptance_given_close_beyond=round(n_sa / n_cb, 4),
        ), f, indent=2, ensure_ascii=False)
    print("\nV1_2_PROFILE_DONE")


if __name__ == "__main__":
    main()
