"""L2 — 状态转移画像。NO MODEL / NO ML。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import research.m2_nondeep_temporal_v1 as m2

RESULTS = "research/analysis_results/liquidity_state_machine_v1"
BOOT = 500
SEED = 7


def rel(a, b):
    if b in (-1, 1) and a in (-1, 1):
        return "WITH_TREND" if a == b else "AGAINST_TREND"
    return "UNKNOWN"


def block_boot(sub, day_of, n=BOOT, seed=SEED):
    """对两个子集的比例差做 trading_day block bootstrap。"""
    rng = np.random.default_rng(seed)
    days = np.asarray(sorted(set(day_of)))
    idx = {d: np.flatnonzero(day_of == d) for d in days}
    ga, gb = sub["_a"].to_numpy(float), sub["_b"].to_numpy(float)
    la, lb = sub["_in_a"].to_numpy(bool), sub["_in_b"].to_numpy(bool)
    diffs, rrs = [], []
    for _ in range(n):
        pick = rng.choice(days, size=len(days), replace=True)
        r = np.concatenate([idx[d] for d in pick])
        a, ia = ga[r], la[r]
        b, ib = gb[r], lb[r]
        pa = a[ia].mean() if ia.any() else np.nan
        pb = b[ib].mean() if ib.any() else np.nan
        if not (np.isfinite(pa) and np.isfinite(pb)):
            continue
        diffs.append(pa - pb)
        rrs.append(pa / pb if pb > 0 else np.nan)
    return dict(
        ci=[round(float(np.nanquantile(diffs, .025)), 4),
            round(float(np.nanquantile(diffs, .975)), 4)],
        rr_ci=[round(float(np.nanquantile(rrs, .025)), 4),
               round(float(np.nanquantile(rrs, .975)), 4)])


def compare(df, mask_a, mask_b, col, label, day_of=None):
    # day_of 必须与传入的 df 对齐（pen / rec 长度不同）
    day_of = df["tday"].to_numpy() if day_of is None else day_of
    a, b = df[mask_a], df[mask_b]
    pa = float(a[col].mean()) if len(a) else np.nan
    pb = float(b[col].mean()) if len(b) else np.nan
    out = dict(metric=label, n_a=len(a), n_b=len(b),
               p_a=round(pa, 4), p_b=round(pb, 4),
               pp_diff=round(pa - pb, 4),
               relative_risk=round(pa / pb, 4) if pb else None)
    sub = df.copy()
    sub["_a"] = df[col].astype(float)
    sub["_b"] = df[col].astype(float)
    sub["_in_a"] = mask_a
    sub["_in_b"] = mask_b
    sub["_d"] = day_of
    out.update(block_boot(sub, day_of))
    return out


def main():
    it = pd.read_parquet(f"{RESULTS}/interactions_primary.parquet")
    it["interaction_time"] = pd.to_datetime(it["interaction_time"])

    # trading_day for bootstrap
    lv = pd.read_parquet(f"{RESULTS}/liquidity_levels.parquet")
    it["tday"] = it["interaction_time"].dt.strftime("%Y-%m-%d")
    day_of = it["tday"].to_numpy()

    pen = it[it["penetrated"]].copy()
    rec = pen[pen["reclaimed"]].copy()

    # ---------- funnel ----------
    fn = [
        ("LIQUIDITY LEVELS", len(lv)),
        ("FIRST INTERACTIONS (touch)", len(it)),
        ("PENETRATION", len(pen)),
        ("RECLAIM", int(pen["reclaimed"].sum())),
        ("ACCEPTANCE", int(pen["accepted"].sum())),
        ("POST-RECLAIM: REVERSAL_MSS", len(rec) and int(
            (rec["post_reclaim_state"] == "REVERSAL_MSS_CONFIRMED").sum())),
        ("POST-RECLAIM: RE-ACCEPTANCE", len(rec) and int(
            (rec["post_reclaim_state"]
             == "REJECTION_FAILED_REACCEPTED").sum())),
    ]
    fd = pd.DataFrame(fn, columns=["stage", "raw_n"])
    fd["pct_of_prev"] = (fd["raw_n"] / fd["raw_n"].shift(1) * 100).round(2)
    fd.to_csv(f"{RESULTS}/interaction_funnel.csv", index=False,
              encoding="utf-8-sig")
    print("=== FUNNEL ===")
    print(fd.to_string(index=False))

    # ---------- A ----------
    pa = float(pen["reclaimed"].mean())
    pb = float(pen["accepted"].mean())
    print(f"\nA. P(RECLAIM|pen)={pa:.4f}  P(ACCEPT|pen)={pb:.4f} "
          f"(n={len(pen)})")

    # ---------- B/C/D ----------
    for tfc in ("trend_struct_1h",):
        pen["sweep_vs_1h"] = [
            rel(int(p), int(t)) for p, t in
            zip(pen["penetration_direction"], pen[tfc])]
        rec["sweep_vs_1h"] = [
            rel(int(p), int(t)) for p, t in
            zip(rec["penetration_direction"], rec[tfc])]
        rec["rev_mss"] = (rec["post_reclaim_state"]
                          == "REVERSAL_MSS_CONFIRMED").astype(float)
        rec["re_acc"] = (rec["post_reclaim_state"]
                         == "REJECTION_FAILED_REACCEPTED").astype(float)

    ag = (pen["sweep_vs_1h"] == "AGAINST_TREND").to_numpy()
    wi = (pen["sweep_vs_1h"] == "WITH_TREND").to_numpy()
    r_ag = (rec["sweep_vs_1h"] == "AGAINST_TREND").to_numpy()
    r_wi = (rec["sweep_vs_1h"] == "WITH_TREND").to_numpy()

    rows = []
    rows.append(compare(pen, ag, wi, "reclaimed",
                        "B: P(RECLAIM|pen) AGAINST vs WITH 1h"))
    rows.append(compare(rec, r_ag, r_wi, "rev_mss",
                        "C: P(REVERSAL_MSS_5M|reclaim) AGAINST vs WITH"))
    rows.append(compare(rec, r_ag, r_wi, "re_acc",
                        "C2: P(RE-ACCEPT|reclaim) AGAINST vs WITH"))
    bd = pd.DataFrame(rows)
    bd.to_csv(f"{RESULTS}/bootstrap_transition_deltas.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== B / C (bootstrap 500) ===")
    print(bd.to_string(index=False))

    # ---------- D: trend resumption ----------
    res = []
    for tag, m in (("AGAINST_1h", r_ag), ("WITH_1h", r_wi)):
        g = rec[m]
        res.append(dict(group=tag, n=len(g),
                        reversal_mss=round(float(g["rev_mss"].mean()), 4),
                        re_acceptance=round(float(g["re_acc"].mean()), 4)))
    dd = pd.DataFrame(res)
    dd.to_csv(f"{RESULTS}/trend_resumption.csv", index=False,
              encoding="utf-8-sig")
    print("\nD. trend resumption (post-reclaim):")
    print(dd.to_string(index=False))

    # ---------- E: env4h x 1h ----------
    it["env_align_1h"] = [
        (e in (-1, 1) and t in (-1, 1) and int(e) == int(t))
        for e, t in zip(it["env_direction_4h"], it["trend_struct_1h"])]
    pe2 = it[it["penetrated"]].copy()
    pe2["sweep_vs_1h"] = [
        rel(int(p), int(t)) for p, t in
        zip(pe2["penetration_direction"], pe2["trend_struct_1h"])]
    rows = []
    for al in (True, False):
        g = pe2[pe2["env_align_1h"] == al]
        a2 = (g["sweep_vs_1h"] == "AGAINST_TREND").to_numpy()
        w2 = (g["sweep_vs_1h"] == "WITH_TREND").to_numpy()
        r2 = g[g["reclaimed"]]
        ra = (r2["sweep_vs_1h"] == "AGAINST_TREND").to_numpy()
        rw = (r2["sweep_vs_1h"] == "WITH_TREND").to_numpy()
        r2 = r2.copy()
        r2["rev_mss"] = (r2["post_reclaim_state"]
                         == "REVERSAL_MSS_CONFIRMED").astype(float)
        rows.append(dict(
            env4h_1h=("ALIGNED" if al else "CONFLICT"),
            n_pen=len(g),
            p_reclaim_against=round(float(g.loc[a2, "reclaimed"].mean()), 4)
            if a2.any() else None,
            p_reclaim_with=round(float(g.loc[w2, "reclaimed"].mean()), 4)
            if w2.any() else None,
            p_rev_mss_against=round(float(r2.loc[ra, "rev_mss"].mean()), 4)
            if ra.any() else None,
            p_rev_mss_with=round(float(r2.loc[rw, "rev_mss"].mean()), 4)
            if rw.any() else None))
    ed = pd.DataFrame(rows)
    ed.to_csv(f"{RESULTS}/transition_by_htf_relation.csv", index=False,
              encoding="utf-8-sig")
    print("\nE. env4h x 1h:")
    print(ed.to_string(index=False))

    # ---------- stratifications ----------
    def strat(col, name):
        rs = []
        for k, g in pen.groupby(col):
            r2 = g[g["reclaimed"]].copy()
            r2["rev_mss"] = (r2["post_reclaim_state"]
                             == "REVERSAL_MSS_CONFIRMED").astype(float)
            rs.append(dict(**{name: str(k)}, n_pen=len(g),
                           p_reclaim=round(float(g["reclaimed"].mean()), 4),
                           p_accept=round(float(g["accepted"].mean()), 4),
                           n_reclaim=len(r2),
                           p_rev_mss=(round(float(r2["rev_mss"].mean()), 4)
                                      if len(r2) else None)))
        t = pd.DataFrame(rs).sort_values("n_pen", ascending=False)
        t.to_csv(f"{RESULTS}/transition_by_{col}.csv", index=False,
                 encoding="utf-8-sig")
        return t

    for c, nm in (("liquidity_type", "liquidity_type"),
                  ("liquidity_scope", "liquidity_scope"),
                  ("symbol", "symbol")):
        t = strat(c, nm)
        print(f"\n=== by {nm} ===")
        print(t.to_string(index=False))
        if c == "symbol":
            print(f"  macro median p_reclaim={t['p_reclaim'].median():.4f}")

    # folds
    folds, _ = m2.build_folds(pen["tday"].to_numpy(), len(pen))
    fr = []
    for fi in range(4):
        te_d = folds[fi][2]
        g = pen[pen["tday"].isin(te_d)]
        r2 = g[g["reclaimed"]].copy()
        r2["rev_mss"] = (r2["post_reclaim_state"]
                         == "REVERSAL_MSS_CONFIRMED").astype(float)
        fr.append(dict(fold=f"F{fi+1}", n_pen=len(g),
                       p_reclaim=round(float(g["reclaimed"].mean()), 4),
                       p_accept=round(float(g["accepted"].mean()), 4),
                       p_rev_mss=round(float(r2["rev_mss"].mean()), 4)))
    pd.DataFrame(fr).to_csv(f"{RESULTS}/transition_by_fold.csv", index=False,
                            encoding="utf-8-sig")
    print("\n=== by fold ===")
    print(pd.DataFrame(fr).to_string(index=False))

    # cluster dedup
    it["cluster"] = (it["symbol"] + "|"
                     + it["interaction_time"].astype(str) + "|"
                     + it["level_price"].round(6).astype(str))
    ded = it.drop_duplicates(subset=["cluster"])
    dpen = ded[ded["penetrated"]]
    dr = dpen[dpen["reclaimed"]].copy()
    dr["rev_mss"] = (dr["post_reclaim_state"]
                     == "REVERSAL_MSS_CONFIRMED").astype(float)
    cd = pd.DataFrame([dict(
        dataset="CLUSTER_DEDUP", n_pen=len(dpen),
        p_reclaim=round(float(dpen["reclaimed"].mean()), 4),
        p_accept=round(float(dpen["accepted"].mean()), 4),
        p_rev_mss=round(float(dr["rev_mss"].mean()), 4))])
    cd.to_csv(f"{RESULTS}/cluster_dedup_sensitivity.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== cluster dedup ===")
    print(cd.to_string(index=False))

    (RESULTS_DIR_AUDIT := f"{RESULTS}/AUDIT.json")
    with open(RESULTS_DIR_AUDIT, "w", encoding="utf-8") as f:
        json.dump(dict(
            levels=int(len(lv)), interactions=int(len(it)),
            penetrations=int(len(pen)),
            p_reclaim_given_pen=round(pa, 4),
            p_accept_given_pen=round(pb, 4),
            displacement="NOT TESTED — NO CANONICAL SEMANTICS",
        ), f, indent=2, ensure_ascii=False)
    print("\nL2_DONE")


if __name__ == "__main__":
    main()
