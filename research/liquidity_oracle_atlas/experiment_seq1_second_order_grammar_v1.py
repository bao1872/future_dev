"""SEQ-1 — Second-order structural event grammar

===========================================================================
唯一研究问题
===========================================================================
在已经知道 current state 与最近一次 structural endpoint E_n 后，再知道更早
一次 endpoint E_{n-1}，尤其是**有序事件 pair** (E_{n-1}, E_n)，
是否还能提高对下一 endpoint E_{n+1} 的概率预测？

    Q0_CURRENT_STATE        = geometry + 10 functional provenance   (= F1)
    Q1_FIRST_ORDER          = Q0 + prev_event_mask      one-hot     (= F3)
    Q2_SECOND_ORDER_ADDITIVE= Q1 + prevprev_event_mask  one-hot
    Q3_SECOND_ORDER_SEQUENCE= Q0 + event_pair_code one-hot   ("{E_{n-1}}>{E_n}")

    PRIMARY = Q3 - Q1   （有序 pair 相对"只知最近事件"是否还有增量）
    secondary = Q1-Q0 / Q2-Q1 / Q3-Q2

Q3 的 pair 已经包含 E_n，因此不再额外加入 prev_event_mask（避免冗余）。
事件保持原始 bitmask category，禁止 bullish/bearish / continuation/reversal
映射，禁止人工合并 mask。

Windows（与 STATE-1/1.1 一致）：
    A: fit TB1      -> eval TB2   (seed 20260918)
    B: fit TB1+TB2  -> eval TB3   (seed 20260919)

Triple chain 合同：
    prevprev.end_bar == prev.start_bar  且 gap == 0
    prev.end_bar     == target.start_bar 且 gap == 0
    三个 event_mask 全部 != 0（历史 censor 直接 STOP）

TB4：tb4_analytically_used = false，不输出任何 TB4 performance statistic。

===========================================================================
输出
===========================================================================
    seq1_summary.json
    seq1_sample_audit.json
    seq1_model_metrics.csv
    seq1_bootstrap.csv
    seq1_per_bit.csv
    seq1_by_symbol.csv
    seq1_transition_first_order.csv
    seq1_transition_pair.csv
    seq1_pair_contribution.csv
（大型 seq1_samples.parquet 存 gitignored cache）
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.liquidity_oracle_atlas.experiment_local_liquidity_transition_v0 import (  # noqa: E402
    FULL_UNIV, TEST_BLOCK, OUT, CACHE, build_blocks,
)
from research.liquidity_oracle_atlas.run_5m_graph_probability_v1 import (  # noqa: E402
    load_raw_bars,
)
from research.liquidity_oracle_atlas.experiment_step01_cause_decomposition_v1 import (  # noqa: E402
    load_seq,
)
from research.liquidity_oracle_atlas.experiment_path0_episode_path_memory_v1 import (  # noqa: E402
    BIT_NAMES, BIT_MASKS, binary_logloss, binary_brier, ece_binary,
    make_pipeline,
)
from research.liquidity_oracle_atlas.experiment_state1_provenance_decomposition_v1 import (  # noqa: E402
    GEOM, group_provenance, sample_key_hash,
)
from research.liquidity_oracle_atlas.experiment_state11_functional_provenance_v1 import (  # noqa: E402
    FUNC_PROV, FROZEN_SAMPLE_HASH,
)

REPL_BLOCK = "TB3"
DEV_BLOCKS = ("TB1", "TB2")

MODELS = {
    "Q0_CURRENT_STATE": (GEOM + FUNC_PROV, []),
    "Q1_FIRST_ORDER": (GEOM + FUNC_PROV, ["prev_event_mask"]),
    "Q2_SECOND_ORDER_ADDITIVE": (GEOM + FUNC_PROV,
                                 ["prev_event_mask", "prevprev_event_mask"]),
    "Q3_SECOND_ORDER_SEQUENCE": (GEOM + FUNC_PROV, ["event_pair_code"]),
}
Q0, Q1, Q2, Q3 = MODELS.keys()
COMPARISONS = [(Q3, Q1), (Q2, Q1), (Q3, Q2), (Q1, Q0)]

WINDOWS = [
    dict(name="A_TB1_to_TB2", train=["TB1"], eval=TEST_BLOCK, seed=20260918),
    dict(name="B_TB1TB2_to_TB3", train=["TB1", "TB2"], eval=REPL_BLOCK,
         seed=20260919),
]
BOOTSTRAP_REPS = 1000


def triple_hash(df: pd.DataFrame) -> str:
    k = (df["symbol"].astype(str)
         + "|" + df["prevprev_start_bar"].astype(str)
         + "|" + df["prevprev_end_bar"].astype(str)
         + "|" + df["prev_start_bar"].astype(str)
         + "|" + df["prev_end_bar"].astype(str)
         + "|" + df["target_start_bar"].astype(str)
         + "|" + df["target_end_bar"].astype(str)
         + "|" + df["prevprev_event_mask"].astype(str)
         + "|" + df["prev_event_mask"].astype(str)
         + "|" + df["target_event_mask"].astype(str))
    h = hashlib.sha256()
    h.update("\n".join(sorted(k.tolist())).encode())
    return h.hexdigest()


def build_triples(ep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sym, g in ep.groupby("symbol", sort=False):
        g = g.sort_values("start_bar").reset_index(drop=True)
        sb = g["start_bar"].to_numpy(np.int64)
        ebi = g["end_bar"].to_numpy(np.int64)
        gaf = g["gap_bars_after_episode"].to_numpy(np.int64)
        gbf = g["gap_bars_before_episode"].to_numpy(np.int64)
        cont = (ebi[:-1] == sb[1:]) & (gaf[:-1] == 0) & (gbf[1:] == 0)
        ok = cont[:-1] & cont[1:]
        for i in np.flatnonzero(ok):
            a, b, c = int(i), int(i + 1), int(i + 2)
            ra, rb, rc = g.loc[a], g.loc[b], g.loc[c]
            rows.append(dict(
                symbol=sym,
                prevprev_start_bar=int(ra["start_bar"]),
                prevprev_end_bar=int(ra["end_bar"]),
                prevprev_event_mask=int(ra["event_mask"]),
                prevprev_start_block=str(ra["start_block"]),
                prevprev_end_block=str(ra["end_block"]),
                prev_start_bar=int(rb["start_bar"]),
                prev_end_bar=int(rb["end_bar"]),
                prev_event_mask=int(rb["event_mask"]),
                prev_start_block=str(rb["start_block"]),
                prev_end_block=str(rb["end_block"]),
                target_start_bar=int(rc["start_bar"]),
                target_end_bar=int(rc["end_bar"]),
                target_event_mask=int(rc["event_mask"]),
                target_start_block=str(rc["start_block"]),
                target_end_block=str(rc["end_block"]),
            ))
    return pd.DataFrame(rows)


def run_window(win, sm):
    tr = sm[sm["target_start_block"].isin(win["train"])].reset_index(drop=True)
    ev = sm[sm["target_start_block"] == win["eval"]].reset_index(drop=True)
    if not len(tr) or not len(ev):
        raise SystemExit(f"STOP_SEQ1_EMPTY_SPLIT: {win['name']}")
    if win["eval"] in set(tr["target_start_block"]):
        raise SystemExit(f"STOP_SEQ1_EVAL_IN_FIT: {win['name']}")

    y_tr = np.stack([((tr["target_event_mask"].to_numpy() & m) != 0)
                     .astype(np.int64) for m in BIT_MASKS], axis=1)
    y_ev = np.stack([((ev["target_event_mask"].to_numpy() & m) != 0)
                     .astype(np.int64) for m in BIT_MASKS], axis=1)
    for k, nm in enumerate(BIT_NAMES):
        if len(np.unique(y_tr[:, k])) < 2:
            raise SystemExit(f"STOP_SEQ1_TRAIN_BIT_ABSENT: "
                             f"{win['name']} {nm}")

    P = {n: np.zeros((len(ev), 4)) for n in MODELS}
    t_fit = time.perf_counter()
    for name, (num_cols, cat_cols) in MODELS.items():
        cols = num_cols + cat_cols
        for k in range(4):
            pipe = make_pipeline(num_cols, cat_cols)
            pipe.fit(tr[cols], y_tr[:, k])
            P[name][:, k] = pipe.predict_proba(ev[cols])[:, 1]
    fit_seconds = time.perf_counter() - t_fit
    t_boot = time.perf_counter()

    def sample_loss(M):
        out = np.zeros(len(y_ev))
        for k in range(4):
            q = np.clip(M[:, k], 1e-15, 1.0 - 1e-15)
            out += -(y_ev[:, k] * np.log(q)
                     + (1.0 - y_ev[:, k]) * np.log1p(-q))
        return out / 4.0

    losses = {n: sample_loss(P[n]) for n in MODELS}
    day = ev["_eval_day"].to_numpy()
    uniq = np.unique(day)
    pos = np.searchsorted(uniq, day)
    nd = len(uniq)
    cnt = np.bincount(pos, minlength=nd)
    keep = cnt > 0
    nk = int(keep.sum())
    rng = np.random.default_rng(win["seed"])

    def boot(dv):
        b = np.empty(BOOTSTRAP_REPS)
        for i in range(BOOTSTRAP_REPS):
            b[i] = dv[rng.integers(0, nk, nk)].mean()
        return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    metrics, boots, per_bit, by_sym = [], [], [], []
    mtab = {}
    for name in MODELS:
        M = P[name]
        ll = [binary_logloss(y_ev[:, k], M[:, k]) for k in range(4)]
        br = [binary_brier(y_ev[:, k], M[:, k]) for k in range(4)]
        ec = [ece_binary(y_ev[:, k], M[:, k]) for k in range(4)]
        mtab[name] = dict(logloss=float(np.mean(ll)),
                          brier=float(np.mean(br)),
                          ece=float(np.mean(ec)))
        for k, nm in enumerate(BIT_NAMES):
            metrics.append(dict(window=win["name"], model=name, bit=nm,
                                n=len(ev),
                                prevalence=float(y_ev[:, k].mean()),
                                logloss=ll[k], brier=br[k], ece=ec[k]))
        metrics.append(dict(window=win["name"], model=name, bit="MEAN_BIT",
                            n=len(ev), prevalence=float(y_ev.mean()),
                            logloss=float(np.mean(ll)),
                            brier=float(np.mean(br)),
                            ece=float(np.mean(ec))))

    for hi_m, lo_m in COMPARISONS:
        dd = losses[hi_m] - losses[lo_m]
        dv = np.bincount(pos, weights=dd, minlength=nd)[keep] / cnt[keep]
        lo, hi = boot(dv)
        boots.append(dict(
            window=win["name"], comparison=f"{hi_m} - {lo_m}",
            role="PRIMARY" if (hi_m, lo_m) == (Q3, Q1) else "SECONDARY",
            delta_sample_weighted=float(dd.mean()),
            delta_daily_mean=float(dv.mean()), n_days=nk,
            ci_lo=lo, ci_hi=hi,
            verdict=("CI_below_zero" if hi < 0 else
                     "CI_above_zero" if lo > 0 else "CI_contains_zero")))

    for k, nm in enumerate(BIT_NAMES):
        def bce(M):
            q = np.clip(M[:, k], 1e-15, 1.0 - 1e-15)
            return -(y_ev[:, k] * np.log(q)
                     + (1.0 - y_ev[:, k]) * np.log1p(-q))
        d3 = bce(P[Q3]) - bce(P[Q1])
        d2 = bce(P[Q2]) - bce(P[Q1])
        dv3 = np.bincount(pos, weights=d3, minlength=nd)[keep] / cnt[keep]
        dv2 = np.bincount(pos, weights=d2, minlength=nd)[keep] / cnt[keep]
        lo3, hi3 = boot(dv3)
        lo2, hi2 = boot(dv2)
        per_bit.append(dict(
            window=win["name"], bit=nm, n=len(ev),
            prevalence=float(y_ev[:, k].mean()),
            Q3_minus_Q1=float(d3.mean()), Q3_minus_Q1_ci_lo=lo3,
            Q3_minus_Q1_ci_hi=hi3,
            Q2_minus_Q1=float(d2.mean()), Q2_minus_Q1_ci_lo=lo2,
            Q2_minus_Q1_ci_hi=hi2))

    for s, g in ev.groupby("symbol"):
        idx = g.index.to_numpy()
        by_sym.append(dict(
            window=win["name"], symbol=s, n=int(len(g)),
            mean_bit_logloss_Q0=float(losses[Q0][idx].mean()),
            mean_bit_logloss_Q1=float(losses[Q1][idx].mean()),
            mean_bit_logloss_Q2=float(losses[Q2][idx].mean()),
            mean_bit_logloss_Q3=float(losses[Q3][idx].mean()),
            delta_Q3_minus_Q1=float((losses[Q3][idx] - losses[Q1][idx]).mean()),
            delta_Q2_minus_Q1=float((losses[Q2][idx] - losses[Q1][idx]).mean())))

    contrib = []
    for code, g in ev.groupby("event_pair_code"):
        idx = g.index.to_numpy()
        contrib.append(dict(
            window=win["name"], event_pair_code=code, n=int(len(g)),
            mean_sample_loss_Q3_minus_Q1=float(
                (losses[Q3][idx] - losses[Q1][idx]).mean())))
    contrib_df = pd.DataFrame(contrib).sort_values(
        "n", ascending=False).reset_index(drop=True)

    # ---- sparsity audit（train/eval 两侧） ----
    tr_codes = tr["event_pair_code"].to_numpy()
    ev_codes = ev["event_pair_code"].to_numpy()
    vc = pd.Series(tr_codes).value_counts()
    unseen = int(np.sum(~np.isin(ev_codes, vc.index.to_numpy())))
    thin = vc[vc < 10]
    sparse = dict(
        window=win["name"],
        n_unique_prev_event_masks=int(tr["prev_event_mask"].nunique()),
        n_unique_prevprev_event_masks=int(
            tr["prevprev_event_mask"].nunique()),
        n_unique_event_pairs_train=int(vc.size),
        n_unique_event_pairs_eval=int(pd.Series(ev_codes).nunique()),
        pair_train_count_min=int(vc.min()),
        pair_train_count_p10=float(np.percentile(vc.to_numpy(), 10)),
        pair_train_count_p50=float(np.percentile(vc.to_numpy(), 50)),
        pair_train_count_p90=float(np.percentile(vc.to_numpy(), 90)),
        pair_train_count_max=int(vc.max()),
        eval_pairs_unseen_in_train_samples=unseen,
        eval_pairs_unseen_in_train_rate=float(unseen) / max(len(ev), 1),
        pairs_with_train_n_lt_10_categories=int(thin.size),
        pairs_with_train_n_lt_10_train_samples=int(thin.sum()),
    )

    return dict(win=win, n_train=int(len(tr)), n_eval=int(len(ev)),
                n_days=nk, mtab=mtab, metrics=metrics, boots=boots,
                per_bit=per_bit, by_sym=by_sym, contrib=contrib_df,
                sparse=sparse, fit_seconds=fit_seconds,
                bootstrap_seconds=time.perf_counter() - t_boot,
                y_ev=y_ev, ev=ev)


def main():
    t_total = time.perf_counter()
    timing = {}

    # ------------------------------------------------------ frozen caches
    t0 = time.perf_counter()
    sm2 = pd.read_parquet(CACHE / "repl0_samples.parquet")
    h2 = sample_key_hash(sm2)
    if h2 != FROZEN_SAMPLE_HASH:
        raise SystemExit(f"STOP_SEQ1_FROZEN_SAMPLE_HASH: {h2}")
    st11 = json.loads((OUT / "state11_summary.json").read_text())
    if st11["STATE11_VERDICT"] != (
            "PREV_EVENT_MEMORY_BEYOND_FUNCTIONAL_PROVENANCE_SUPPORTED"):
        raise SystemExit("STOP_SEQ1_FROZEN_CROSSCHECK_FAIL: STATE-1.1")
    ep = pd.read_parquet(CACHE / "episode_repl0_through_tb3.parquet")
    bars_full = {s: load_raw_bars(s) for s in FULL_UNIV}
    all_days, day_block_code, boundaries = build_blocks(bars_full)
    td_by_sym = {}
    for s in FULL_UNIV:
        td_full = np.asarray(bars_full[s]["td"]).astype("datetime64[D]")
        code = day_block_code[np.searchsorted(all_days, td_full)]
        tb3_end = int(np.flatnonzero(code <= 2)[-1])
        td_by_sym[s] = td_full[:tb3_end + 1]
    del bars_full
    timing["cache_load_seconds"] = round(time.perf_counter() - t0, 2)

    # -------------------------------------------------------- triple chain
    t0 = time.perf_counter()
    tri = build_triples(ep)
    n_triples_raw = int(len(tri))
    if (tri["prevprev_event_mask"] == 0).any() \
            or (tri["prev_event_mask"] == 0).any():
        raise SystemExit("STOP_SEQ1_HISTORY_CENSOR")
    n_target_censor = int((tri["target_event_mask"] == 0).sum())
    tri = tri[tri["target_event_mask"] != 0].copy()
    same_blk = ((tri["target_start_block"] == tri["target_end_block"])
                & tri["target_start_block"].isin(["TB1", "TB2", "TB3"]))
    n_cross_block = int((~same_blk).sum())
    tri = tri[same_blk].copy()
    n_triples = int(len(tri))
    timing["triple_sample_seconds"] = round(time.perf_counter() - t0, 2)

    # ------------------------ merge frozen geometry from repl0_samples
    t0 = time.perf_counter()
    keycols = ["symbol", "prev_start_bar", "prev_end_bar",
               "target_start_bar", "target_end_bar"]
    merge_cols = keycols + ["prev_event_mask", "target_event_mask",
                            "target_start_block", "target_end_block"] \
        + list(GEOM) + ["tmp_atr_cur", "tmp_atr_prev"]
    sm = tri.merge(sm2[merge_cols], on=keycols, how="inner",
                   suffixes=("", "_r0"))
    if len(sm) != n_triples:
        raise SystemExit(
            "STOP_SEQ1_FROZEN_CROSSCHECK_FAIL: triple chains not a subset of "
            f"repl0_samples ({len(sm)} vs {n_triples})")
    if not bool((sm["prev_event_mask"] == sm["prev_event_mask_r0"]).all()
                and (sm["target_event_mask"]
                     == sm["target_event_mask_r0"]).all()):
        raise SystemExit("STOP_SEQ1_FROZEN_CROSSCHECK_FAIL: mask mismatch")
    if not bool((sm["target_start_block"]
                 == sm["target_start_block_r0"]).all()
                and (sm["target_end_block"]
                     == sm["target_end_block_r0"]).all()):
        raise SystemExit("STOP_SEQ1_FROZEN_CROSSCHECK_FAIL: block mismatch")
    sm["event_pair_code"] = (sm["prevprev_event_mask"].astype(str) + ">"
                             + sm["prev_event_mask"].astype(str))
    sm["_eval_day"] = np.array(
        [td_by_sym[s][b] for s, b in
         zip(sm["symbol"], sm["target_start_bar"])], dtype="datetime64[D]")

    # ------------------------------------------------- provenance (F1/F3 冻结)
    key = ep["symbol"].astype(str) + "|" + ep["start_bar"].astype(str)
    g_up = dict(zip(key, ep["start_upper_group"].astype(np.int64)))
    g_dn = dict(zip(key, ep["start_lower_group"].astype(np.int64)))
    grps = {s: load_seq(s)[1] for s in FULL_UNIV}
    prov = np.full((len(sm), 6), np.nan)
    for i, r in enumerate(sm.itertuples()):
        k = f"{r.symbol}|{r.target_start_bar}"
        pu = group_provenance(grps[r.symbol], int(g_up.get(k, -1)),
                              int(r.target_start_bar))
        pl = group_provenance(grps[r.symbol], int(g_dn.get(k, -1)),
                              int(r.target_start_bar))
        if pu is None or pl is None:
            raise SystemExit(
                "STOP_SEQ1_PROVENANCE_CONTRACT_FAIL: "
                f"{r.symbol} t={r.target_start_bar}")
        prov[i, :] = [pu[0], pu[1], pu[2], pl[0], pl[1], pl[2]]
    for j, c in enumerate(["upper_oldest_age_bars", "upper_newest_age_bars",
                           "upper_n_active_identities", "lower_oldest_age_bars",
                           "lower_newest_age_bars",
                           "lower_n_active_identities"]):
        sm[c] = prov[:, j]
    for side in ("upper", "lower"):
        sm[f"{side}_oldest_log_age"] = np.log1p(
            sm[f"{side}_oldest_age_bars"].to_numpy(float))
        sm[f"{side}_newest_log_age"] = np.log1p(
            sm[f"{side}_newest_age_bars"].to_numpy(float))
        sm[f"{side}_newest_age_zero"] = (
            sm[f"{side}_newest_age_bars"].to_numpy() == 0).astype(float)
        sm[f"{side}_oldest_age_zero"] = (
            sm[f"{side}_oldest_age_bars"].to_numpy() == 0).astype(float)
    timing["feature_seconds"] = round(time.perf_counter() - t0, 2)

    # ------------------------------------------------------------ windows
    res = []
    for win in WINDOWS:
        res.append(run_window(win, sm))
    timing["fit_seconds"] = round(sum(r["fit_seconds"] for r in res), 2)
    timing["bootstrap_seconds"] = round(
        sum(r["bootstrap_seconds"] for r in res), 2)

    # ------------------------------------------- transition tables (§9)
    tt1, tt2 = [], []
    for r in res:
        w = r["win"]["name"]
        ev, y = r["ev"], r["y_ev"]
        for code, g in ev.groupby("prev_event_mask"):
            idx = g.index.to_numpy()
            tt1.append(dict(
                window=w, subset="eval", prev_event_mask=int(code),
                n=int(len(g)),
                **{f"p_{nm}": float(y[idx, k].mean())
                   for k, nm in enumerate(BIT_NAMES)}))
        for code, g in ev.groupby("event_pair_code"):
            idx = g.index.to_numpy()
            tt2.append(dict(
                window=w, subset="eval", event_pair_code=code,
                n=int(len(g)),
                **{f"p_{nm}": float(y[idx, k].mean())
                   for k, nm in enumerate(BIT_NAMES)}))
    pd.DataFrame(tt1).to_csv(OUT / "seq1_transition_first_order.csv",
                             index=False)
    pd.DataFrame(tt2).to_csv(OUT / "seq1_transition_pair.csv", index=False)
    contrib = pd.concat([r["contrib"] for r in res], ignore_index=True)
    contrib.to_csv(OUT / "seq1_pair_contribution.csv", index=False)

    metrics = pd.DataFrame([m for r in res for m in r["metrics"]])
    boots = pd.DataFrame([b for r in res for b in r["boots"]])
    per_bit = pd.DataFrame([p for r in res for p in r["per_bit"]])
    by_sym = pd.DataFrame([b for r in res for b in r["by_sym"]])
    metrics.to_csv(OUT / "seq1_model_metrics.csv", index=False)
    boots.to_csv(OUT / "seq1_bootstrap.csv", index=False)
    per_bit.to_csv(OUT / "seq1_per_bit.csv", index=False)
    by_sym.to_csv(OUT / "seq1_by_symbol.csv", index=False)

    # ----------------------------------------------------------- verdict
    prim = boots[boots["role"] == "PRIMARY"].set_index("window")
    both_ok = bool((prim["ci_hi"] < 0).all())
    none_inc = bool((prim["ci_lo"] >= 0).all())
    if both_ok:
        v = "SECOND_ORDER_EVENT_SEQUENCE_SUPPORTED"
    elif none_inc:
        v = "FIRST_ORDER_EVENT_STATE_SUFFICIENT_WITHIN_TESTED_GRAMMAR"
    else:
        v = "SECOND_ORDER_EVENT_SEQUENCE_NOT_STABLE"

    q2q1 = boots[(boots["comparison"] == f"{Q2} - {Q1}")]
    q3q2 = boots[(boots["comparison"] == f"{Q3} - {Q2}")]
    audit = dict(
        n_episodes_used=int(len(ep)),
        n_triple_chains_raw=n_triples_raw,
        n_triple_chains_after_history_check=int(n_triples_raw),
        excluded_target_censor=n_target_censor,
        excluded_cross_block_target=n_cross_block,
        n_triple_chains=n_triples,
        n_triple_chains_after_censor=int(len(sm)),
        triple_sample_key_sha256=triple_hash(sm),
        n_unique_event_pairs_overall=int(sm["event_pair_code"].nunique()),
        n_unique_prev_event_masks_overall=int(sm["prev_event_mask"].nunique()),
        n_unique_prevprev_event_masks_overall=int(
            sm["prevprev_event_mask"].nunique()),
        sparsity={r["sparse"]["window"]: r["sparse"] for r in res},
        window_counts={r["win"]["name"]: dict(
            n_train=r["n_train"], n_eval=r["n_eval"], n_days=r["n_days"])
            for r in res},
        q2_vs_q1_all_below_zero=bool((q2q1["ci_hi"] < 0).all()),
        q3_vs_q2_all_contains_zero=bool((q3q2["ci_lo"] < 0).any()),
        tb4_analytically_used=False,
    )
    (OUT / "seq1_sample_audit.json").write_text(
        json.dumps(audit, indent=2, default=str))
    sm.to_parquet(CACHE / "seq1_samples.parquet", index=False)

    summary = dict(
        experiment="SEQ-1 second-order structural event grammar",
        base="ca8032c289e2fef97ad416f9b2f264c15744b07e",
        question=("given current state and the most recent structural endpoint "
                  "E_n, does the earlier endpoint E_{n-1} - and especially the "
                  "ordered pair (E_{n-1}, E_n) - improve prediction of the "
                  "next endpoint E_{n+1}?"),
        frozen_context=dict(
            state11_verdict=st11["STATE11_VERDICT"],
            repl0_sample_hash=h2,
            frozen_geometry_source="repl0_samples.parquet (cur_* columns)",
            frozen_provenance_source=("group_provenance imported from the "
                                     "STATE-1 module; functional forms from "
                                     "STATE-1.1 FUNC_PROV")),
        models={k: dict(numeric=v[0], categorical=v[1])
                for k, v in MODELS.items()},
        event_representation=dict(
            code="raw bitmask category, no bullish/bearish mapping",
            pair_code="{prevprev_event_mask}>{prev_event_mask}",
            q3_uses_pair_only=("event_pair already contains E_n, so "
                               "prev_event_mask is NOT added again to Q3")),
        windows=[dict(name=w["name"], train=w["train"], eval=w["eval"],
                      seed=w["seed"]) for w in WINDOWS],
        bootstrap_reps=BOOTSTRAP_REPS,
        sample_audit=audit,
        results=[dict(window=r["win"]["name"], n_train=r["n_train"],
                      n_eval=r["n_eval"], n_days=r["n_days"],
                      mean_bit_metrics=r["mtab"]) for r in res],
        bootstrap=boots.to_dict(orient="records"),
        per_bit_Q3_minus_Q1=per_bit.to_dict(orient="records"),
        by_symbol=dict(
            n_rows=len(by_sym), n_symbol_per_window=15,
            window_A_n_negative=int(((by_sym["window"] == "A_TB1_to_TB2")
                                     & (by_sym["delta_Q3_minus_Q1"] < 0)).sum()),
            window_A_n_positive=int(((by_sym["window"] == "A_TB1_to_TB2")
                                     & (by_sym["delta_Q3_minus_Q1"] > 0)).sum()),
            window_B_n_negative=int(((by_sym["window"] == "B_TB1TB2_to_TB3")
                                     & (by_sym["delta_Q3_minus_Q1"] < 0)).sum()),
            window_B_n_positive=int(((by_sym["window"] == "B_TB1TB2_to_TB3")
                                     & (by_sym["delta_Q3_minus_Q1"] > 0)).sum())),
        pair_contribution_top=int(len(contrib)),
        tb4_analytically_used=False,
        timing=timing,
        interpretation_limits=(
            "SEQ-1 is a Markov-order audit on already-consumed development "
            "evidence (TB1/TB2/TB3). It does NOT prove a second-order Markov "
            "process, latent state, SMC validity, tradability or RL, and it "
            "does NOT authorise third-order sequences."),
    )
    summary["SEQ1_VERDICT"] = v
    if both_ok:
        summary["SEQ1_CONCLUSION"] = (
            "ordered two-event structural history contains predictive "
            "information beyond the current local state and the immediately "
            "previous structural endpoint (second-order event-history "
            "candidate).")
    elif none_inc:
        summary["SEQ1_CONCLUSION"] = (
            "first-order event-state representation is sufficient within the "
            "tested grammar; do NOT extend history length.")
    else:
        summary["SEQ1_CONCLUSION"] = (
            "the second-order sequence increment is not stable across windows; "
            "stay at the first-order event representation.")
    timing["total_seconds"] = round(time.perf_counter() - t_total, 2)
    summary["timing"] = timing
    (OUT / "seq1_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    print(f"[TRIPLES] {n_triples} hash={audit['triple_sample_key_sha256'][:16]}")
    print(f"[SPARSITY]\n{pd.DataFrame(list(audit['sparsity'].values())).to_string(index=False)}")
    for r in res:
        print(f"[WINDOW {r['win']['name']}] train={r['n_train']} "
              f"eval={r['n_eval']} days={r['n_days']}")
        print(pd.DataFrame([dict(model=k, **v)
                            for k, v in r["mtab"].items()]).to_string(
                                index=False))
    print(f"[BOOTSTRAP]\n{boots.to_string(index=False)}")
    print(f"[PER-BIT Q3-Q1]\n{per_bit.to_string(index=False)}")
    print(f"[BY-SYMBOL] {summary['by_symbol']}")
    print(f"[TIMING] {timing}")
    print(f"[VERDICT] {v}")
    print(f"[DONE] -> {OUT}")


if __name__ == "__main__":
    main()
