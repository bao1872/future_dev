"""STATE-1 — Separate current-boundary provenance from event memory

===========================================================================
唯一研究问题
===========================================================================
此前 prev_event_mask 的稳定增量，究竟是真正的跨 episode event memory，
还是因为 prev_event_mask 间接编码了 target start 时当前 liquidity boundary 的
activation provenance / age，而 current-geometry baseline 没有这些状态变量？

（target.start_bar == prev.end_bar，所以 prev 的 STRUCT 事件很可能正是
 当前边界的"出生方式"，两者天然高度耦合。本轮做机制分解。）

===========================================================================
四层模型（冻结，无 tuning / 无 feature search）
===========================================================================
    S0_GEOMETRY              = 4 current geometry
    S1_GEOMETRY_PROVENANCE   = S0 + 6 current-boundary provenance
    S2_GEOMETRY_PREV_EVENT   = S0 + prev_event_mask one-hot   (= 原 M1)
    S3_GEOMETRY_PROVENANCE_PREV_EVENT = S1 + prev_event_mask one-hot

    PRIMARY = S3 - S1     （控制 current provenance 后 prev-event 是否仍有增量）
    secondary = S1-S0 / S2-S0 / S3-S2

6 个 provenance 变量（只针对 target start bar t，两侧各 3 个）：
    upper_oldest_age_bars = t - min(active activation_bar)
    upper_newest_age_bars = t - max(active activation_bar)
    upper_n_active_identities
    lower_* 对称

active contract 复用 frozen LOCAL lifecycle：
    activation_bar <= t  AND  t < expiry_bar
    AND (penetration_bar < 0 OR t < penetration_bar)

===========================================================================
两个 stability window（都是已消费过的 development evidence）
===========================================================================
    Window A: fit TB1        -> evaluate TB2   (seed 20260916)
    Window B: fit TB1+TB2    -> evaluate TB3   (seed 20260917)

这不是新的 untouched replication，只是 mechanism audit。

TB4 合同：tb4_analytically_used = false
    TB4 不得贡献 episode endpoint / sample inclusion / feature / label /
    fit / preprocessing fit / metric / bootstrap / diagnostic。
    （loader 可能读取完整 raw 文件以便复用 cache，但所有数值数组在使用前
      已截断为 <= TB3 view；本脚本不使用任何 TB4 数值。）

===========================================================================
输出
===========================================================================
    state1_summary.json
    state1_sample_audit.json
    state1_model_metrics.csv
    state1_bootstrap.csv
    state1_per_bit.csv
    state1_by_symbol.csv
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
    FULL_UNIV, TRAIN_BLOCK, TEST_BLOCK, OUT, CACHE, build_blocks,
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

REPL_BLOCK = "TB3"
GEOM = ["cur_up_distance_R", "cur_down_distance_R", "cur_width_R",
        "cur_log_distance_ratio"]
PROV = ["upper_oldest_age_bars", "upper_newest_age_bars",
        "upper_n_active_identities",
        "lower_oldest_age_bars", "lower_newest_age_bars",
        "lower_n_active_identities"]
CAT = ["prev_event_mask"]

MODELS = {
    "S0_GEOMETRY": (GEOM, []),
    "S1_GEOMETRY_PROVENANCE": (GEOM + PROV, []),
    "S2_GEOMETRY_PREV_EVENT": (GEOM, CAT),
    "S3_GEOMETRY_PROVENANCE_PREV_EVENT": (GEOM + PROV, CAT),
}
S0, S1, S2, S3 = MODELS.keys()
COMPARISONS = [(S3, S1), (S1, S0), (S2, S0), (S3, S2)]

WINDOWS = [
    dict(name="A_TB1_to_TB2", train=["TB1"], eval=TEST_BLOCK, seed=20260916),
    dict(name="B_TB1TB2_to_TB3", train=["TB1", "TB2"], eval=REPL_BLOCK,
         seed=20260917),
]
BOOTSTRAP_REPS = 1000


# ---------------------------------------------------------------------------
# provenance（只用 target start bar t 及之前的信息）
# ---------------------------------------------------------------------------
def group_provenance(grp, g: int, t: int):
    """return (oldest_age, newest_age, n_active) or None if group not active."""
    if g < 0:
        return None
    s = int(grp["group_starts"][g])
    e = s + int(grp["group_lengths"][g])
    a = grp["act"][s:e]
    if not len(a):
        return None
    ok = ((a <= t) & (t < grp["exp"][s:e])
          & ((grp["pen"][s:e] < 0) | (t < grp["pen"][s:e])))
    if not ok.any():
        return None
    aa = a[ok]
    return (int(t) - int(aa.min()), int(t) - int(aa.max()), int(ok.sum()))


def sample_key_hash(df: pd.DataFrame) -> str:
    k = (df["symbol"].astype(str) + "|" + df["prev_start_bar"].astype(str)
         + "|" + df["prev_end_bar"].astype(str) + "|"
         + df["target_start_bar"].astype(str) + "|"
         + df["target_end_bar"].astype(str) + "|"
         + df["target_event_mask"].astype(str))
    h = hashlib.sha256()
    h.update("\n".join(sorted(k.tolist())).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# one window
# ---------------------------------------------------------------------------
def run_window(win, sm, y_all, prov_ok_mask):
    tr = sm[sm["target_start_block"].isin(win["train"])].reset_index(drop=True)
    ev = sm[sm["target_start_block"] == win["eval"]].reset_index(drop=True)
    if not len(tr) or not len(ev):
        raise SystemExit(f"STOP_STATE1_EMPTY_SPLIT: {win['name']}")
    if win["eval"] in set(tr["target_start_block"]):
        raise SystemExit(f"STOP_STATE1_EVAL_IN_FIT: {win['name']}")

    y_tr = np.stack([((tr["target_event_mask"].to_numpy() & m) != 0)
                     .astype(np.int64) for m in BIT_MASKS], axis=1)
    y_ev = np.stack([((ev["target_event_mask"].to_numpy() & m) != 0)
                     .astype(np.int64) for m in BIT_MASKS], axis=1)
    for k, nm in enumerate(BIT_NAMES):
        if len(np.unique(y_tr[:, k])) < 2:
            raise SystemExit(f"STOP_STATE1_TRAIN_BIT_ABSENT: "
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
            role="PRIMARY" if (hi_m, lo_m) == (S3, S1) else "SECONDARY",
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
        d3 = bce(P[S3]) - bce(P[S1])
        d2 = bce(P[S2]) - bce(P[S0])
        dv3 = np.bincount(pos, weights=d3, minlength=nd)[keep] / cnt[keep]
        dv2 = np.bincount(pos, weights=d2, minlength=nd)[keep] / cnt[keep]
        lo3, hi3 = boot(dv3)
        lo2, hi2 = boot(dv2)
        per_bit.append(dict(
            window=win["name"], bit=nm, n=len(ev),
            prevalence=float(y_ev[:, k].mean()),
            S3_minus_S1=float(d3.mean()), S3_minus_S1_ci_lo=lo3,
            S3_minus_S1_ci_hi=hi3,
            S2_minus_S0=float(d2.mean()), S2_minus_S0_ci_lo=lo2,
            S2_minus_S0_ci_hi=hi2))

    for s, g in ev.groupby("symbol"):
        idx = g.index.to_numpy()
        by_sym.append(dict(
            window=win["name"], symbol=s, n=int(len(g)),
            mean_bit_logloss_S1=float(losses[S1][idx].mean()),
            mean_bit_logloss_S3=float(losses[S3][idx].mean()),
            delta_S3_minus_S1=float((losses[S3][idx] - losses[S1][idx]).mean()),
            mean_bit_logloss_S0=float(losses[S0][idx].mean()),
            mean_bit_logloss_S2=float(losses[S2][idx].mean()),
            delta_S2_minus_S0=float((losses[S2][idx] - losses[S0][idx]).mean())))

    return dict(win=win, n_train=int(len(tr)), n_eval=int(len(ev)),
                n_days=nk, mtab=mtab, metrics=metrics, boots=boots,
                per_bit=per_bit, by_sym=by_sym,
                fit_seconds=fit_seconds,
                bootstrap_seconds=time.perf_counter() - t_boot)


def main():
    t_total = time.perf_counter()
    timing = {}

    # ---------------------------------------------------------- sample parity
    t0 = time.perf_counter()
    sm = pd.read_parquet(CACHE / "repl0_samples.parquet")
    repl0 = json.loads((OUT / "repl0_summary.json").read_text())
    h_now = sample_key_hash(sm)
    h_repl0 = repl0["repl0_sample_key_sha256"]
    if h_now != h_repl0:
        raise SystemExit(
            f"STOP_STATE1_SAMPLE_PARITY_FAIL: {h_now} != {h_repl0}")

    ep = pd.read_parquet(CACHE / "episode_repl0_through_tb3.parquet")
    ep["_k"] = ep["symbol"].astype(str) + "|" + ep["start_bar"].astype(str)
    grp_map = dict(zip(ep["_k"], ep["start_upper_group"].astype(np.int64)))
    grp_map_lo = dict(zip(ep["_k"], ep["start_lower_group"].astype(np.int64)))

    # 交易日 tag（与 PATH-0 / REPL-0 的 bootstrap 单位一致）。
    # block grid 是 LOCAL-0 一次性冻结的定义（不是统计量）；每个 symbol 的
    # 数值数组在使用前一律截断到 TB3 最后一根 bar，TB4 数值不参与任何计算。
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

    # ------------------------------------------------------- provenance feats
    t0 = time.perf_counter()
    grps = {s: load_seq(s)[1] for s in FULL_UNIV}
    n_ok = n_missing = 0
    feats = np.full((len(sm), 6), np.nan)
    for i, r in enumerate(sm.itertuples()):
        k = f"{r.symbol}|{r.target_start_bar}"
        g_up = grp_map.get(k, -1)
        g_dn = grp_map_lo.get(k, -1)
        pu = group_provenance(grps[r.symbol], int(g_up), int(r.target_start_bar))
        pl = group_provenance(grps[r.symbol], int(g_dn), int(r.target_start_bar))
        if pu is None or pl is None:
            n_missing += 1
            if pu is None and g_up >= 0:
                raise SystemExit(
                    f"STOP_STATE1_PROVENANCE_CONTRACT_FAIL: inactive group "
                    f"{r.symbol} t={r.target_start_bar} upper")
            if pl is None and g_dn >= 0:
                raise SystemExit(
                    f"STOP_STATE1_PROVENANCE_CONTRACT_FAIL: inactive group "
                    f"{r.symbol} t={r.target_start_bar} lower")
            continue
        feats[i, :] = [pu[0], pu[1], pu[2], pl[0], pl[1], pl[2]]
        n_ok += 1
    sm = sm.copy()
    for j, c in enumerate(PROV):
        sm[c] = feats[:, j]
    bad = ~np.isfinite(feats).all(axis=1)
    if bad.any():
        raise SystemExit(
            f"STOP_STATE1_PROVENANCE_CONTRACT_FAIL: {int(bad.sum())} samples "
            "without active provenance")
    if (feats[:, [0, 1, 3, 4]] < 0).any():
        raise SystemExit("STOP_STATE1_PROVENANCE_CONTRACT_FAIL: negative age")
    if (feats[:, [2, 5]] < 1).any():
        raise SystemExit(
            "STOP_STATE1_PROVENANCE_CONTRACT_FAIL: n_active < 1")
    sm["_eval_day"] = np.array(
        [td_by_sym[s][b] for s, b in
         zip(sm["symbol"], sm["target_start_bar"])], dtype="datetime64[D]")
    timing["provenance_feature_seconds"] = round(time.perf_counter() - t0, 2)

    prov_audit = dict(
        n_samples=int(len(sm)),
        n_provenance_ok=int(n_ok), n_provenance_missing=int(n_missing),
        upper_oldest_age_bars=dict(
            mean=float(sm["upper_oldest_age_bars"].mean()),
            p50=float(sm["upper_oldest_age_bars"].median()),
            p90=float(sm["upper_oldest_age_bars"].quantile(0.90)),
            max=int(sm["upper_oldest_age_bars"].max())),
        upper_newest_age_bars=dict(
            mean=float(sm["upper_newest_age_bars"].mean()),
            p50=float(sm["upper_newest_age_bars"].median()),
            min=int(sm["upper_newest_age_bars"].min())),
        upper_n_active_identities=dict(
            mean=float(sm["upper_n_active_identities"].mean()),
            p50=float(sm["upper_n_active_identities"].median()),
            max=int(sm["upper_n_active_identities"].max())),
        lower_oldest_age_bars=dict(
            mean=float(sm["lower_oldest_age_bars"].mean()),
            p50=float(sm["lower_oldest_age_bars"].median()),
            p90=float(sm["lower_oldest_age_bars"].quantile(0.90)),
            max=int(sm["lower_oldest_age_bars"].max())),
        frac_upper_newest_age_zero=float(
            (sm["upper_newest_age_bars"] == 0).mean()),
        frac_lower_newest_age_zero=float(
            (sm["lower_newest_age_bars"] == 0).mean()),
    )

    # -------------------------------------------- corroborate the coupling claim
    # prev_event 含 NEW_UPPER 时，当前 upper 是否"刚出生"？
    nu = (sm["prev_event_mask"] & 4) != 0
    nl = (sm["prev_event_mask"] & 8) != 0
    coupling = dict(
        frac_upper_age0_given_prev_NEW_UPPER=float(
            (sm.loc[nu, "upper_newest_age_bars"] == 0).mean()),
        frac_upper_age0_given_prev_no_NEW_UPPER=float(
            (sm.loc[~nu, "upper_newest_age_bars"] == 0).mean()),
        frac_lower_age0_given_prev_NEW_LOWER=float(
            (sm.loc[nl, "lower_newest_age_bars"] == 0).mean()),
        frac_lower_age0_given_prev_no_NEW_LOWER=float(
            (sm.loc[~nl, "lower_newest_age_bars"] == 0).mean()),
    )
    (OUT / "state1_sample_audit.json").write_text(json.dumps(
        dict(sample_parity=dict(
            n_samples=int(len(sm)), sample_key_sha256=h_now,
            matches_repl0=bool(h_now == h_repl0),
            n_dev=int((sm["target_start_block"].isin(["TB1", "TB2"])).sum()),
            n_repl=int((sm["target_start_block"] == REPL_BLOCK).sum())),
            provenance=prov_audit, prev_event_to_provenance_coupling=coupling,
            tb4_analytically_used=False), indent=2, default=str))

    # ---------------------------------------------------------------- windows
    res = []
    for win in WINDOWS:
        res.append(run_window(win, sm, None, None))
    timing["fit_seconds"] = round(sum(r["fit_seconds"] for r in res), 2)
    timing["bootstrap_seconds"] = round(
        sum(r["bootstrap_seconds"] for r in res), 2)

    metrics = pd.DataFrame([m for r in res for m in r["metrics"]])
    boots = pd.DataFrame([b for r in res for b in r["boots"]])
    per_bit = pd.DataFrame([p for r in res for p in r["per_bit"]])
    by_sym = pd.DataFrame([b for r in res for b in r["by_sym"]])
    metrics.to_csv(OUT / "state1_model_metrics.csv", index=False)
    boots.to_csv(OUT / "state1_bootstrap.csv", index=False)
    per_bit.to_csv(OUT / "state1_per_bit.csv", index=False)
    by_sym.to_csv(OUT / "state1_by_symbol.csv", index=False)

    # --------------------------------------------------------------- verdicts
    prim = boots[boots["role"] == "PRIMARY"].set_index("window")
    ok_all = bool((prim["ci_hi"] < 0).all())
    none_inc = bool((prim["ci_lo"] >= 0).all())
    if ok_all:
        v = "PREV_EVENT_MEMORY_BEYOND_CURRENT_PROVENANCE_SUPPORTED"
    elif none_inc:
        v = "PREV_EVENT_EFFECT_EXPLAINED_BY_CURRENT_PROVENANCE"
    else:
        v = "PREV_EVENT_MEMORY_BEYOND_CURRENT_PROVENANCE_NOT_STABLE"

    summary = dict(
        experiment="STATE-1 current-boundary provenance vs event memory",
        base="4fa151ae6409a6af5d5b0ad13b31cc37d3c061ab",
        question=("is the stable prev_event_mask increment genuine cross-episode "
                  "event memory, or does it merely encode the activation "
                  "provenance/age of the current boundary at target start?"),
        models={k: dict(numeric=v[0], categorical=v[1])
                for k, v in MODELS.items()},
        windows=[dict(name=w["name"], train=w["train"], eval=w["eval"],
                      seed=w["seed"]) for w in WINDOWS],
        bootstrap_reps=BOOTSTRAP_REPS,
        sample_parity=dict(n_samples=int(len(sm)), sample_key_sha256=h_now,
                           repl0_hash=h_repl0, matches=bool(h_now == h_repl0)),
        provenance_audit=prov_audit,
        prev_event_to_provenance_coupling=coupling,
        results=[dict(window=r["win"]["name"], n_train=r["n_train"],
                      n_eval=r["n_eval"], n_days=r["n_days"],
                      mean_bit_metrics=r["mtab"]) for r in res],
        bootstrap=boots.to_dict(orient="records"),
        per_bit_S3_minus_S1=per_bit.to_dict(orient="records"),
        by_symbol=dict(
            n_rows=len(by_sym), n_symbol_per_window=15,
            window_A_n_negative=int(((by_sym["window"] == "A_TB1_to_TB2")
                                     & (by_sym["delta_S3_minus_S1"] < 0)).sum()),
            window_A_n_positive=int(((by_sym["window"] == "A_TB1_to_TB2")
                                     & (by_sym["delta_S3_minus_S1"] > 0)).sum()),
            window_B_n_negative=int(((by_sym["window"] == "B_TB1TB2_to_TB3")
                                     & (by_sym["delta_S3_minus_S1"] < 0)).sum()),
            window_B_n_positive=int(((by_sym["window"] == "B_TB1TB2_to_TB3")
                                     & (by_sym["delta_S3_minus_S1"] > 0)).sum())),
        crosscheck_vs_frozen_rounds=dict(
            note=("S2 - S0 must reproduce the already-frozen M1 - M0 / REPL-A "
                  "numbers exactly, because S2 uses the identical feature set "
                  "and window split"),
            window_A_S2_minus_S0=float(
                boots[(boots["window"] == "A_TB1_to_TB2")
                      & (boots["comparison"] == f"{S2} - {S0}")
                      ]["delta_daily_mean"].iloc[0]),
            path0_M1_minus_M0=float(
                [r for r in
                 json.loads((OUT / "path0_summary.json").read_text())
                 ["bootstrap"] if r["comparison"].startswith("M1_")][0]
                ["delta_daily_mean"]),
            window_B_S2_minus_S0=float(
                boots[(boots["window"] == "B_TB1TB2_to_TB3")
                      & (boots["comparison"] == f"{S2} - {S0}")
                      ]["delta_daily_mean"].iloc[0]),
            repl0_M1_minus_M0=float(
                [r for r in repl0["bootstrap"]
                 if r["comparison"].startswith("M1_")][0]["delta_daily_mean"])),
        tb4_analytically_used=False,
        tb4_note=("TB4 may be present in the raw loader output for cache reuse, "
                  "but no TB4 value contributes to episode endpoints, sample "
                  "inclusion, features, labels, fit, preprocessing fit, "
                  "metrics, bootstrap or diagnostics; it remains blind for "
                  "final evaluation."),
        timing=timing,
        interpretation_limits=(
            "STATE-1 is a mechanism decomposition on already-consumed "
            "development evidence (TB1/TB2/TB3). It does NOT establish SMC, "
            "event sequencing, latent psychology, tradability or RL."),
    )
    summary["STATE1_VERDICT"] = v
    if ok_all:
        summary["STATE1_CONCLUSION"] = (
            "previous structural endpoint contains information about the next "
            "structural transition beyond current local geometry and current "
            "boundary activation provenance (first-order event memory "
            "candidate supported). Latent psychology is still NOT observed.")
    elif none_inc:
        summary["STATE1_CONCLUSION"] = (
            "the prev_event effect disappears once current boundary "
            "provenance is controlled: stop event-sequence escalation and "
            "improve the current state representation first.")
    else:
        summary["STATE1_CONCLUSION"] = (
            "the prev_event increment beyond current provenance is not stable "
            "across the two windows; no escalation decision is authorised by "
            "this round.")
    timing["total_seconds"] = round(time.perf_counter() - t_total, 2)
    summary["timing"] = timing
    (OUT / "state1_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    print(f"[SAMPLE PARITY] {len(sm)} hash_match={h_now == h_repl0}")
    print(f"[PROVENANCE] {prov_audit}")
    print(f"[COUPLING] {coupling}")
    for r in res:
        print(f"[WINDOW {r['win']['name']}] train={r['n_train']} "
              f"eval={r['n_eval']} days={r['n_days']}")
        print(pd.DataFrame([dict(model=k, **v)
                            for k, v in r["mtab"].items()]).to_string(
                                index=False))
    print(f"[BOOTSTRAP]\n{boots.to_string(index=False)}")
    print(f"[PER-BIT S3-S1]\n{per_bit.to_string(index=False)}")
    print(f"[BY-SYMBOL] {summary['by_symbol']}")
    print(f"[TIMING] {timing}")
    print(f"[VERDICT] {v}")
    print(f"[DONE] -> {OUT}")


if __name__ == "__main__":
    main()
