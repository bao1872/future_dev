"""MARKET-STATE-1 — 5m 尺度下的最小充分状态（严禁重跑 SMC / 触及 SMC·OB·latent·RL）

===========================================================================
唯一研究问题（2×2 因子，先冻结基准）
===========================================================================
已知 "当前在哪里 (5m geometry) + 目前怎么走过来 (running path) + 上次如何结束
(prev endpoint)" 之后，问：

    * 动态 provenance（边界流动性太老 / 组成变化）是否还提供稳定独立信息？
    * arrival tempo（同样位置、同样路径统计，速度 / 路径效率是否还改变转移）？

基准直接冻结为 PGM-BAR-0 已通过的 M2：
    B0_VALIDATED_STATE == PGM-BAR-0 M2 (start geom + start prov + prev + elapsed
                                     + cur geom + running path)

四模型（B0 / P / T / PT）回答：
    P-B0   provenance 单独有没有价值
    T-B0   tempo 单独有没有价值
    PT-T   已有 tempo 后，provenance 是否还有独立价值  ← 决定 state 是否保留 P
    PT-P   已有 provenance 后，tempo 是否还有独立价值 ← 决定 state 是否保留 T

主指标 = mean episode NLL（与 PGM-BAR-0 完全一致）。

===========================================================================
冻结边界（GOVERNANCE）
===========================================================================
* 直接复用 pgm_bar0_samples.parquet（359714 bar / 37987 episode）与两个 frozen
  episode parquet，本轮完全不调用 compute_smc_pine / load_raw_bars 重算 SMC。
* 只新增：动态 provenance 残差（10）+ arrival tempo（4）。
* 不碰 SMC / OB / latent state / HMM / RL / PnL / TB4。
* episode_id 沿用 pgm_bar0 全局唯一 id，不重建 episode。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ---- reused canonical components (READ-ONLY) ---------------------------------
from research.liquidity_oracle_atlas.experiment_pgm_bar0_path_smc_v1 import (  # noqa: E402
    M2_NUM, CAT, fit_eval, model_metrics, episode_nll, boot_delta,
    make_pipeline, densify, mask_index, BIT_MASKS, ece_binary,
)
from research.liquidity_oracle_atlas.experiment_local_liquidity_transition_v0 import (  # noqa: E402
    FULL_UNIV, OUT, CACHE,
)
from research.liquidity_oracle_atlas.experiment_step01_cause_decomposition_v1 import (  # noqa: E402
    load_seq,
)
from research.liquidity_oracle_atlas.experiment_state1_provenance_decomposition_v1 import (  # noqa: E402
    group_provenance,
)
from research.liquidity_oracle_atlas.experiment_path0_episode_path_memory_v1 import (  # noqa: E402
    episode_identity_hash,  # noqa: F401  (kept for parity with PGM-BAR-0)
)

BASE_SHA = "099fbb87b0fa5db5cce6e03342decb4838cb7c07"
BOOTSTRAP_REPS = 1000
EPS = 1e-9

# --------------------------------------------------------------- feature names
PROV_DELTA = [
    "upper_oldest_log_age_residual", "upper_newest_log_age_residual",
    "upper_current_newest_age_zero", "upper_current_oldest_age_zero",
    "upper_active_identity_count_delta",
    "lower_oldest_log_age_residual", "lower_newest_log_age_residual",
    "lower_current_newest_age_zero", "lower_current_oldest_age_zero",
    "lower_active_identity_count_delta",
]
TEMPO = [
    "tempo_signed_speed", "tempo_abs_speed",
    "tempo_signed_efficiency", "tempo_abs_efficiency",
]

MODELS = {
    "B0_VALIDATED_STATE": (list(M2_NUM), CAT),
    "P_CURRENT_PROVENANCE": (list(M2_NUM) + PROV_DELTA, CAT),
    "T_ARRIVAL_TEMPO": (list(M2_NUM) + TEMPO, CAT),
    "PT_PROVENANCE_TEMPO": (list(M2_NUM) + PROV_DELTA + TEMPO, CAT),
}
B0, P, T, PT = MODELS.keys()

COMPARISONS = [
    ("P-B0", "PROVENANCE_STANDALONE", P, B0),
    ("T-B0", "TEMPO_STANDALONE", T, B0),
    ("PT-T", "PROVENANCE_UNIQUE", PT, T),
    ("PT-P", "TEMPO_UNIQUE", PT, P),
]
COMP_LABELS = [c[0] for c in COMPARISONS]

WINDOWS = [
    dict(name="A_TB1_to_TB2", train=["TB1"], eval="TB2", seed=20260920),
    dict(name="B_TB1TB2_to_TB3", train=["TB1", "TB2"], eval="TB3", seed=20260921),
]


# ===========================================================================
# vectorized provenance (canonical group_provenance 的向量化版本，须逐一 parity)
# ===========================================================================
def group_provenance_many(grp, g, t_values):
    """返回 (oldest_age, newest_age, n_active) 数组，t_values 为多个 bar。"""
    t_values = np.asarray(t_values, dtype=np.int64)
    if g < 0:
        raise RuntimeError("STOP_MARKET_STATE1_INVALID_GROUP")
    s = int(grp["group_starts"][g])
    e = s + int(grp["group_lengths"][g])
    act = np.asarray(grp["act"][s:e], dtype=np.int64)
    exp = np.asarray(grp["exp"][s:e], dtype=np.int64)
    pen = np.asarray(grp["pen"][s:e], dtype=np.int64)
    if not len(act):
        raise RuntimeError("STOP_MARKET_STATE1_EMPTY_GROUP")
    T = t_values[:, None]
    active = ((act[None, :] <= T) & (T < exp[None, :])
              & ((pen[None, :] < 0) | (T < pen[None, :])))
    n_active = active.sum(axis=1)
    if np.any(n_active == 0):
        raise RuntimeError("STOP_MARKET_STATE1_CURRENT_PROVENANCE_INACTIVE")
    huge = np.iinfo(np.int64).max
    min_act = np.where(active, act[None, :], huge).min(axis=1)
    max_act = np.where(active, act[None, :], -1).max(axis=1)
    oldest = t_values - min_act
    newest = t_values - max_act
    return oldest, newest, n_active


# ===========================================================================
# provenance residual features（相对 "正常老化" 的变化）
# ===========================================================================
def prov_residual_arrays(grp, g, s, t_values, side_prefix):
    start = group_provenance(grp, int(g), int(s))
    if start is None:
        raise RuntimeError("STOP_MARKET_STATE1_START_PROVENANCE_INACTIVE")
    old0, new0, n0 = start
    oldest, newest, ncount = group_provenance_many(grp, int(g), t_values)
    elapsed = (t_values - int(s)).astype(float)
    expected_old = old0 + elapsed
    expected_new = new0 + elapsed
    return {
        f"{side_prefix}_oldest_log_age_residual":
            np.log1p(oldest) - np.log1p(expected_old),
        f"{side_prefix}_newest_log_age_residual":
            np.log1p(newest) - np.log1p(expected_new),
        f"{side_prefix}_current_newest_age_zero":
            (newest == 0).astype(float),
        f"{side_prefix}_current_oldest_age_zero":
            (oldest == 0).astype(float),
        f"{side_prefix}_active_identity_count_delta":
            ncount.astype(float) - float(n0),
    }


# ===========================================================================
# arrival tempo（仅用已存列，不重新读取 OHLC）
# ===========================================================================
def add_tempo_features(df):
    k = (df["bar_t"].to_numpy(np.int64)
         - df["start_bar"].to_numpy(np.int64))
    net = (df["start_up_distance_R"].to_numpy(float)
           - df["cur_up_distance_R"].to_numpy(float))     # = (C_t - C_s)/ATR0
    tv = df["path_total_variation_R"].to_numpy(float)
    den_k = np.maximum(k, 1)
    den_tv = np.maximum(tv, 1e-12)
    signed_speed = net / den_k
    signed_eff = net / den_tv
    first = k == 0
    signed_speed[first] = 0.0
    signed_eff[first] = 0.0
    df["tempo_signed_speed"] = signed_speed
    df["tempo_abs_speed"] = np.abs(signed_speed)
    df["tempo_signed_efficiency"] = signed_eff
    df["tempo_abs_efficiency"] = np.abs(signed_eff)
    return df


# ===========================================================================
# build enriched sample (add tempo + dynamic provenance residuals once)
# ===========================================================================
def build_enriched(obs, map0, map3):
    obs = add_tempo_features(obs)
    new_cols = {c: np.full(len(obs), np.nan, dtype=float) for c in PROV_DELTA}
    grp_cache = {}
    n_fail = 0
    for ep_id, gdf in obs.groupby("episode_id", sort=False):
        sym = gdf["symbol"].iloc[0]
        sb = int(gdf["start_bar"].iloc[0])
        blk = gdf["block"].iloc[0]
        mp = map0 if blk in ("TB1", "TB2") else map3
        key = (sym, sb)
        if key not in mp:
            raise SystemExit(
                f"STOP_MARKET_STATE1_EPISODE_IDENTITY_FAIL: {key} block={blk}")
        gu, gl = mp[key]
        if sym not in grp_cache:
            _, grp = load_seq(sym)
            grp_cache[sym] = grp
        grp = grp_cache[sym]
        t_vals = gdf["bar_t"].to_numpy(np.int64)
        order = np.argsort(t_vals, kind="stable")
        t_sorted = t_vals[order]
        try:
            up = prov_residual_arrays(grp, gu, sb, t_sorted, "upper")
            lo = prov_residual_arrays(grp, gl, sb, t_sorted, "lower")
        except RuntimeError as e:
            raise SystemExit(f"STOP_MARKET_STATE1_PROVENANCE_BUILD_FAIL: {e}")
        idx = gdf.index.to_numpy()[order]
        for c in PROV_DELTA:
            side = "upper" if c.startswith("upper_") else "lower"
            new_cols[c][idx] = (up if side == "upper" else lo)[c]
    for c in PROV_DELTA:
        if np.any(np.isnan(new_cols[c])):
            n_fail += int(np.isnan(new_cols[c]).sum())
        obs[c] = new_cols[c]
    if n_fail:
        raise SystemExit(f"STOP_MARKET_STATE1_PROVENANCE_NAN: {n_fail}")
    return obs


# ===========================================================================
# main
# ===========================================================================
def main():
    t_total = time.perf_counter()
    timing = {}

    # ---------------------------------------------------- load base sample
    t0 = time.perf_counter()
    obs = pd.read_parquet(CACHE / "pgm_bar0_samples.parquet")
    ep0 = pd.read_parquet(CACHE / "episode0_episodes.parquet")
    ep3 = pd.read_parquet(CACHE / "episode_repl0_through_tb3.parquet")
    map0 = {(r.symbol, int(r.start_bar)):
             (int(r.start_upper_group), int(r.start_lower_group))
             for r in ep0.itertuples()}
    map3 = {(r.symbol, int(r.start_bar)):
             (int(r.start_upper_group), int(r.start_lower_group))
             for r in ep3.itertuples()}
    timing["cache_load_seconds"] = round(time.perf_counter() - t0, 2)

    # ---------------------------------------------------- sample / TB parity
    n_ep = int(obs["episode_id"].nunique())
    n_rows = int(len(obs))
    n_term = int(obs["hazard"].sum())
    if (n_ep, n_rows, n_term) != (37987, 359714, 37987):
        raise SystemExit(
            f"STOP_MARKET_STATE1_SAMPLE_PARITY_FAIL: {n_ep},{n_rows},{n_term}")
    if set(obs["block"].unique()) - {"TB1", "TB2", "TB3"}:
        raise SystemExit("STOP_MARKET_STATE1_TB4_PRESENT")

    ep0_key = {(r.symbol, int(r.start_bar)): int(r.event_mask)
               for r in ep0.itertuples()}
    ep3_key = {(r.symbol, int(r.start_bar)): int(r.event_mask)
               for r in ep3.itertuples()}
    term = obs[obs["hazard"] == 1]
    mism = 0
    for r in term.itertuples():
        mp = ep0_key if r.block in ("TB1", "TB2") else ep3_key
        exp = mp.get((r.symbol, int(r.start_bar)))
        if exp is None or int(r.target_mask) != exp:
            mism += 1
    if mism:
        raise SystemExit(
            f"STOP_MARKET_STATE1_EPISODE_IDENTITY_FAIL: {mism} mismatches")
    print(f"[SAMPLE PARITY] n_ep={n_ep} n_rows={n_rows} n_term={n_term} "
          f"tb_identity_mismatch={mism}")

    # ---------------------------------------------------- build enriched
    t_b = time.perf_counter()
    obs = build_enriched(obs, map0, map3)
    obs.to_parquet(CACHE / "market_state1_samples.parquet", index=False)
    timing["enrich_seconds"] = round(time.perf_counter() - t_b, 2)

    # ---------------------------------------------------- provenance parity
    # sample (symbol, group, t) where the group is ACTIVE at t (canonical != None)
    t_p = time.perf_counter()
    rng = np.random.default_rng(20260922)
    grp_cache = {}
    pool = []
    for sym in FULL_UNIV:
        _, grp = load_seq(sym)
        grp_cache[sym] = grp
        gstarts = np.asarray(grp["group_starts"]).astype(np.int64)
        glens = np.asarray(grp["group_lengths"]).astype(np.int64)
        act = np.asarray(grp["act"]).astype(np.int64)
        exp = np.asarray(grp["exp"]).astype(np.int64)
        pen = np.asarray(grp["pen"]).astype(np.int64)
        ng = int(len(gstarts))
        cand_g = rng.integers(0, ng, size=300)  # ~300 groups/symbol
        for g in cand_g:
            g = int(g)
            s0 = int(gstarts[g]); e0 = s0 + int(glens[g])
            if e0 <= s0:
                continue
            a = act[s0:e0]; e = exp[s0:e0]; p = pen[s0:e0]
            valid = np.flatnonzero(
                a < np.where(p < 0, e, np.minimum(e, p)))
            if not len(valid):
                continue
            k = int(valid[rng.integers(0, len(valid))])
            hi = int(e[k] if p[k] < 0 else min(e[k], p[k]))
            t = int(rng.integers(a[k], hi))
            pool.append((sym, g, t))
    if len(pool) < 1000:
        raise SystemExit(
            f"STOP_MARKET_STATE1_PROVENANCE_PARITY_INSUFFICIENT: {len(pool)}")
    pick = rng.choice(len(pool), size=min(1500, len(pool)), replace=False)
    parity_rows = []
    n_ok = 0
    n_bad = 0
    for i in pick:
        sym, g, t = pool[int(i)]
        grp = grp_cache[sym]
        canon = group_provenance(grp, g, t)
        if canon is None:
            continue
        vo, vn, vc = group_provenance_many(grp, g, np.array([t], dtype=np.int64))
        ok = (int(vo[0]), int(vn[0]), int(vc[0])) == canon
        n_ok += int(ok)
        n_bad += int(not ok)
        parity_rows.append(dict(symbol=sym, group=g, t=t, ok=bool(ok)))
    if n_bad or n_ok < 1000:
        raise SystemExit(
            f"STOP_MARKET_STATE1_PROVENANCE_PARITY_FAIL: bad={n_bad} ok={n_ok}")
    timing["provenance_parity_seconds"] = round(time.perf_counter() - t_p, 2)
    print(f"[PROVENANCE PARITY] ok={n_ok} bad={n_bad}")

    # causality / timing sanity: the 6 true residual columns must be ~0 at the
    # first bar (k==0). The age_zero binary flags are legitimate indicators and
    # are allowed to be 1.0; count_delta is a residual and must be 0 at k=0.
    RESIDUAL_COLS = [c for c in PROV_DELTA
                     if "age_residual" in c or "count_delta" in c]
    first = (obs["bar_t"].to_numpy(np.int64)
             - obs["start_bar"].to_numpy(np.int64)) == 0
    max_resid = float(max(
        np.abs(obs.loc[first, c].to_numpy()).max() for c in RESIDUAL_COLS))
    if max_resid > 1e-9:
        raise SystemExit(
            f"STOP_MARKET_STATE1_PROVENANCE_LEAK: "
            f"max_firstbar_residual={max_resid}")
    # tempo identity at first bar = 0
    if np.any(np.abs(obs.loc[first, TEMPO].to_numpy()) > 1e-12):
        raise SystemExit("STOP_MARKET_STATE1_TEMPO_FIRSTBAR_NONZERO")

    # ---------------------------------------------------- windows + models
    res = []
    all_per_ep = {}
    for w in WINDOWS:
        tr = obs[obs["block"].isin(w["train"])].reset_index(drop=True)
        ev = obs[obs["block"] == w["eval"]].reset_index(drop=True)
        if not len(tr) or not len(ev):
            raise SystemExit(f"STOP_MARKET_STATE1_EMPTY_SPLIT: {w['name']}")
        if w["eval"] in set(tr["block"]):
            raise SystemExit(f"STOP_MARKET_STATE1_EVAL_IN_FIT: {w['name']}")

        per_ep = {}
        model_out = {}
        for name, (num_cols, cat_cols) in MODELS.items():
            out = fit_eval(name, num_cols, cat_cols, tr, ev)
            met = model_metrics(ev, out["p_h"], out["p_mask"], out["te"])
            model_out[name] = dict(metrics=met, aud=out["aud"])
            h = ev["hazard"].to_numpy().astype(np.int64)
            uniq, per = episode_nll(
                out["p_h"], out["p_mask"], h,
                ev["target_mask"].to_numpy().astype(np.int64),
                ev["episode_id"].to_numpy())
            per_ep[name] = (uniq, per)

        boots = []
        by_sym = []
        for label, gate, hi, lo in COMPARISONS:
            uniq_h, per_h = per_ep[hi]
            uniq_l, per_l = per_ep[lo]
            if not np.array_equal(uniq_h, uniq_l):
                raise SystemExit(
                    f"STOP_MARKET_STATE1_EPISODE_MISALIGN: {label}")
            delta = per_h - per_l
            day = ev.groupby("episode_id")["episode_start_day"].first()
            day = day.reindex(pd.Index(uniq_h)).to_numpy()
            a, b = boot_delta(delta, day, w["seed"] + COMP_LABELS.index(label))
            verdict = ("CI_below_zero" if b < 0
                       else "CI_above_zero" if a > 0 else "CI_contains_zero")
            sym = ev.groupby("episode_id")["symbol"].first()
            sym = sym.reindex(pd.Index(uniq_h)).to_numpy()
            for s_ in FULL_UNIV:
                idx = sym == s_
                if idx.any():
                    d = delta[idx]
                    by_sym.append(dict(window=w["name"], comparison=label,
                                       symbol=s_, n=int(idx.sum()),
                                       mean_delta=float(d.mean()),
                                       n_negative=int((d < 0).sum()),
                                       n_positive=int((d > 0).sum())))
            boots.append(dict(window=w["name"], comparison=label, gate=gate,
                              model_hi=hi, model_lo=lo,
                              delta_sample_mean=float(delta.mean()),
                              ci_lo=a, ci_hi=b, verdict=verdict))
        res.append(dict(win=w, model_out=model_out, boots=boots,
                        by_sym=by_sym, n_train=int(len(tr)),
                        n_eval=int(len(ev))))

    # ---------------------------------------------------- outputs
    mtab_rows = []
    for r in res:
        w = r["win"]["name"]
        for name, mo in r["model_out"].items():
            mtab_rows.append(dict(window=w, model=name, **mo["metrics"]))
    mtab = pd.DataFrame(mtab_rows)
    mtab.to_csv(OUT / "market_state1_model_metrics.csv", index=False)

    boots_df = pd.DataFrame([b for r in res for b in r["boots"]])
    boots_df.to_csv(OUT / "market_state1_bootstrap.csv", index=False)

    bysym_df = pd.DataFrame([b for r in res for b in r["by_sym"]])
    bysym_df.to_csv(OUT / "market_state1_by_symbol.csv", index=False)

    pd.DataFrame(parity_rows).to_csv(
        OUT / "market_state1_provenance_parity.csv", index=False)

    # feature audit
    fa_rows = []
    for c in PROV_DELTA:
        v = obs[c].to_numpy(float)
        nz = float(np.mean(np.abs(v) > 1e-9))
        fa_rows.append(dict(feature=c, kind="provenance_residual",
                            non_zero_rate=nz, mean=float(np.mean(v)),
                            std=float(np.std(v))))
    for c in TEMPO:
        v = obs[c].to_numpy(float)
        fa_rows.append(dict(feature=c, kind="tempo",
                            non_zero_rate=float(np.mean(np.abs(v) > 1e-9)),
                            mean=float(np.mean(v)), std=float(np.std(v))))
    pd.DataFrame(fa_rows).to_csv(
        OUT / "market_state1_feature_audit.csv", index=False)

    opt_rows = []
    for r in res:
        w = r["win"]["name"]
        for name, mo in r["model_out"].items():
            a = mo["aud"]
            opt_rows.append(dict(window=w, model=name, n_params=int(a["n_params"]),
                                n_iter=int(a["n_iter"]),
                                final_fun=float(a["final_fun"]),
                                success=bool(a["success"]),
                                elapsed_seconds=round(a["elapsed_seconds"], 2)))
    pd.DataFrame(opt_rows).to_csv(
        OUT / "market_state1_optimizer_audit.csv", index=False)

    # ---------------------------------------------------- verdict
    verdict = {}
    for label, gate, hi, lo in COMPARISONS:
        ws = boots_df[boots_df["comparison"] == label]
        ci_both = bool((ws["ci_hi"] < 0).all())
        # cross-symbol stability: >=10/15 symbols with mean_delta<0 in both windows
        sym_neg = {}
        for wname in [r["win"]["name"] for r in res]:
            sub = bysym_df[(bysym_df["window"] == wname)
                           & (bysym_df["comparison"] == label)]
            sym_neg[wname] = int((sub["mean_delta"] < 0).sum())
        stable = ci_both and all(v >= 10 for v in sym_neg.values())
        verdict[gate] = dict(
            supported=stable,
            windows={r["win"]["name"]: dict(
                ci_hi=float(ws[ws["window"] == r["win"]["name"]]["ci_hi"].iloc[0]))
                for r in res},
            n_symbols_delta_neg=sym_neg)

    # ---------------------------------------------------- B0 parity guard
    pgm0_summary = json.loads(
        (OUT / "pgm_bar0_summary.json").read_text())
    pgm0_m2 = pgm0_summary["model_metrics"]
    b0_out = {r["win"]["name"]: r["model_out"][B0]["metrics"] for r in res}
    parity_fail = False
    for wname in [r["win"]["name"] for r in res]:
        ref = pgm0_m2[wname]["M2_INTRA_EPISODE_PATH"]
        got = b0_out[wname]
        for k in ("mean_episode_nll", "hazard_nll", "endpoint_joint_nll"):
            if abs(float(got[k]) - float(ref[k])) > 1e-6:
                parity_fail = True
    if parity_fail:
        raise SystemExit("STOP_MARKET_STATE1_BASELINE_PARITY_FAIL")

    sample_audit = dict(
        n_episodes=n_ep, n_bar_rows=n_rows, n_terminals=n_term,
        provenance_parity_ok=bool(n_bad == 0 and n_ok >= 1000),
        n_provenance_parity_points=int(n_ok),
        tempo_firstbar_zero=bool(True),
        provenance_firstbar_residual_max=max_resid,
        base_sample="pgm_bar0_samples.parquet (reused, unmodified)",
        base_commit=BASE_SHA,
    )
    (OUT / "market_state1_sample_audit.json").write_text(
        json.dumps(sample_audit, indent=2, default=str))

    summary = dict(
        experiment="MARKET-STATE-1 minimal sufficient 5m observable state",
        base=BASE_SHA,
        question=("given current 5m geometry + running path + previous endpoint, "
                  "do dynamic provenance and arrival tempo each add stable, "
                  "independent episode-NLL increments?"),
        baseline="PGM-BAR-0 M2 (frozen)",
        models={k: dict(n_numeric=len(v[0]), n_categorical=len(v[1]),
                        numeric=v[0], categorical=v[1])
                for k, v in MODELS.items()},
        design="2x2 factorial: B0 / P(provenance) / T(tempo) / PT(both)",
        new_features=dict(provenance_residual=PROV_DELTA, tempo=TEMPO),
        windows=[dict(name=w["name"], train=w["train"], eval=w["eval"],
                      seed=w["seed"]) for w in WINDOWS],
        primary_metric="mean episode NLL",
        tb4_analytically_used=False,
        model_metrics={r["win"]["name"]: {name: mo["metrics"]
                       for name, mo in r["model_out"].items()}
                       for r in res},
        bootstrap=boots_df.to_dict(orient="records"),
        by_symbol=dict(
            n_rows=int(len(bysym_df)),
            windows={w["name"]: {label: dict(
                n_negative=int(((bysym_df["window"] == w["name"])
                                & (bysym_df["comparison"] == label)
                                & (bysym_df["n_negative"] > 0)).sum()),
                n_positive=int(((bysym_df["window"] == w["name"])
                                & (bysym_df["comparison"] == label)
                                & (bysym_df["n_positive"] > 0)).sum()),
                n_symbols_mean_delta_neg=int(((bysym_df["window"] == w["name"])
                                & (bysym_df["comparison"] == label)
                                & (bysym_df["mean_delta"] < 0)).sum()))
                for label, gate, hi, lo in COMPARISONS}
                for w in WINDOWS}),
        verdict=verdict,
        gates={
            "PROVENANCE_STANDALONE": "P-B0 CI_hi<0 both windows",
            "TEMPO_STANDALONE": "T-B0 CI_hi<0 both windows",
            "PROVENANCE_UNIQUE": "PT-T CI_hi<0 both windows AND >=10/15 sym",
            "TEMPO_UNIQUE": "PT-P CI_hi<0 both windows AND >=10/15 sym",
        },
        b0_parity_vs_pgm_bar0_m2="PASS" if not parity_fail else "FAIL",
        sample_audit=sample_audit,
        bootstrap_reps=BOOTSTRAP_REPS,
    )
    (OUT / "market_state1_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    timing["total_seconds"] = round(time.perf_counter() - t_total, 2)
    summary["timing"] = timing
    (OUT / "market_state1_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    print(f"[MODEL METRICS]\n{mtab.to_string(index=False)}")
    print(f"[BOOTSTRAP]\n{boots_df.to_string(index=False)}")
    print(f"[VERDICT] {verdict}")
    print(f"[TIMING] {timing}")
    print(f"[DONE] -> {OUT}")


if __name__ == "__main__":
    main()
