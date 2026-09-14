"""MARKET-STATE-1.1 — State Representation Closure（基于 31e651c）

===========================================================
用户审计结论驱动的收口（不是新实验方向，是补两个缺口）
===========================================================
核心结果可信，但 IDE 上一轮有两处说过头 + 一个 identity audit 缺口：

  (1) tempo 不是"新 Market State 信息"，而是 B0 已有 Location+Path 的
      非线性重参数化（speed=net/elapsed, efficiency=net/TV，三者都在 B0）。
      因此 T-B0 改善证明"当前模型类表达不了比值关系"，不是发现了新信息维度。
  (2) "10 provenance + 4 tempo = 最小充分状态" 不能成立：
      - 两个 oldest_log_age_residual 永为 0（无信息）；
      - 还没证明最小化 / 无冗余 / sufficient。
  (3) identity parity 上一轮只查了 event_mask，没查 end_bar / 上 lower group id
      —— 必须补 exact closure。

本实验只做两件事：

  A. EXACT EPISODE IDENTITY CLOSURE（全量 37987 terminal）
     对每个 terminal：(symbol,start_bar) 查 frozen episode，逐条断言
        end_bar == bar_t + 1
        start_block == block
        event_mask == target_mask
     并额外做 provenance group-id closure：用 frozen start_upper/lower_group
     重算 provenance residual，必须与已存 enriched 列逐位一致
     —— 这直接证明 provenance 是用"正确的 frozen 边界 identity"算的。

  B. COMPACT PROVENANCE（每侧 6 个 → 只留真正动态的）
        newest_age_residual, current_newest_age_zero, active_count_delta
     上下共 6，删掉：
        upper/lower_oldest_log_age_residual（永为 0）
        upper/lower_current_oldest_age_zero（高度由 起点+elapsed 决定）
     确认 Pc-B0 仍保留绝大多数增益；tempo 正式归类为 derived representation。

不进入 Dynamic PGM-1 / latent / HMM / SMC / RL / PnL / TB4。
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

from research.liquidity_oracle_atlas.experiment_market_state1_minimal_state_v1 import (  # noqa: E402
    M2_NUM, CAT, fit_eval, model_metrics, episode_nll, boot_delta,
    group_provenance_many, prov_residual_arrays, build_enriched,
    PROV_DELTA, TEMPO, BASE_SHA, WINDOWS,
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

BOOTSTRAP_REPS = 1000

# ----------------------------------------------------- compact provenance (6)
# 只留真正动态的：newest age 重置 / newest-age-zero / active count 增减。
COMPACT_PROV = [
    "upper_newest_log_age_residual", "upper_current_newest_age_zero",
    "upper_active_identity_count_delta",
    "lower_newest_log_age_residual", "lower_current_newest_age_zero",
    "lower_active_identity_count_delta",
]
# 被删除的 4 个（永为 0 / 可由起点+elapsed 决定）
DROPPED_PROV = [c for c in PROV_DELTA if c not in COMPACT_PROV]

MODELS = {
    "B0_VALIDATED_STATE": (list(M2_NUM), CAT),
    "Pc_COMPACT_PROVENANCE": (list(M2_NUM) + COMPACT_PROV, CAT),
    "PF_FULL_PROVENANCE": (list(M2_NUM) + PROV_DELTA, CAT),
    "T_DERIVED_TEMPO": (list(M2_NUM) + TEMPO, CAT),   # derived repr, not new info
    "PTc_COMPACT_PROV_TEMPO": (list(M2_NUM) + COMPACT_PROV + TEMPO, CAT),
}
B0, Pc, PF, T, PTc = MODELS.keys()

COMPARISONS = [
    ("Pc-B0", "PROVENANCE_COMPACT_STANDALONE", Pc, B0),
    ("PF-B0", "PROVENANCE_FULL_REFERENCE", PF, B0),
    ("T-B0", "TEMPO_DERIVED_REPRESENTATION", T, B0),
    ("PTc-Pc", "PROVENANCE_UNIQUE_COMPACT", PTc, Pc),
    ("PTc-T", "TEMPO_UNIQUE_COMPACT", PTc, T),
]
COMP_LABELS = [c[0] for c in COMPARISONS]


# ===========================================================================
# A. exact episode identity closure
# ===========================================================================
def exact_identity_closure(obs_base, ep_full0, ep_full3):
    term = obs_base[obs_base["hazard"] == 1]
    mism_end = mism_blk = mism_ev = missing = 0
    for r in term.itertuples():
        mp = ep_full0 if r.block in ("TB1", "TB2") else ep_full3
        fz = mp.get((r.symbol, int(r.start_bar)))
        if fz is None:
            missing += 1
            continue
        if int(fz.end_bar) != int(r.bar_t) + 1:
            mism_end += 1
        if fz.start_block != r.block:
            mism_blk += 1
        if int(fz.event_mask) != int(r.target_mask):
            mism_ev += 1
    if missing or mism_end or mism_blk or mism_ev:
        raise SystemExit(
            f"STOP_MARKET_STATE1_1_IDENTITY_CLOSURE_FAIL: "
            f"missing={missing} end={mism_end} blk={mism_blk} ev={mism_ev}")
    return dict(checked=int(len(term)), mism_end=mism_end, mism_blk=mism_blk,
                mism_ev=mism_ev, missing=missing)


def provenance_group_id_closure(enriched, map0, map3):
    """用 frozen start_upper/lower_group 重算 provenance，必须逐位匹配已存列。"""
    rng = np.random.default_rng(20260923)
    term = enriched[enriched["hazard"] == 1]
    idx = rng.choice(len(term), size=min(500, len(term)), replace=False)
    grp_cache = {}
    max_err = 0.0
    checked = 0
    for i in idx:
        r = term.iloc[int(i)]
        sym = r.symbol
        sb = int(r.start_bar)
        blk = r.block
        mp = map0 if blk in ("TB1", "TB2") else map3
        gu, gl = mp[(sym, sb)]
        if sym not in grp_cache:
            _, grp = load_seq(sym)
            grp_cache[sym] = grp
        grp = grp_cache[sym]
        sub = enriched[enriched["episode_id"] == r.episode_id]
        t_sorted = np.sort(sub["bar_t"].to_numpy(np.int64))
        up = prov_residual_arrays(grp, gu, sb, t_sorted, "upper")
        lo = prov_residual_arrays(grp, gl, sb, t_sorted, "lower")
        for c in COMPACT_PROV:
            side = "upper" if c.startswith("upper_") else "lower"
            stored = sub.set_index("bar_t").loc[t_sorted, c].to_numpy(float)
            calc = (up if side == "upper" else lo)[c]
            max_err = max(max_err, float(np.max(np.abs(stored - calc))))
        checked += 1
    if max_err > 1e-9:
        raise SystemExit(
            f"STOP_MARKET_STATE1_1_PROV_GROUP_ID_CLOSURE_FAIL: max_err={max_err}")
    return dict(checked=int(checked), max_err=max_err)


# ===========================================================================
# provenance vectorized vs canonical parity (re-verify, >=1000 points)
# ===========================================================================
def provenance_canonical_parity():
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
        for g in rng.integers(0, ng, size=300):
            g = int(g)
            s0 = int(gstarts[g]); e0 = s0 + int(glens[g])
            if e0 <= s0:
                continue
            a = act[s0:e0]; e = exp[s0:e0]; p = pen[s0:e0]
            valid = np.flatnonzero(a < np.where(p < 0, e, np.minimum(e, p)))
            if not len(valid):
                continue
            k = int(valid[rng.integers(0, len(valid))])
            hi = int(e[k] if p[k] < 0 else min(e[k], p[k]))
            t = int(rng.integers(a[k], hi))
            pool.append((sym, g, t))
    if len(pool) < 1000:
        raise SystemExit(
            f"STOP_MARKET_STATE1_1_PARITY_INSUFFICIENT: {len(pool)}")
    pick = rng.choice(len(pool), size=min(1500, len(pool)), replace=False)
    n_ok = n_bad = 0
    for i in pick:
        sym, g, t = pool[int(i)]
        canon = group_provenance(grp_cache[sym], g, t)
        if canon is None:
            continue
        vo, vn, vc = group_provenance_many(
            grp_cache[sym], g, np.array([t], dtype=np.int64))
        ok = (int(vo[0]), int(vn[0]), int(vc[0])) == canon
        n_ok += int(ok)
        n_bad += int(not ok)
    if n_bad or n_ok < 1000:
        raise SystemExit(
            f"STOP_MARKET_STATE1_1_PROVENANCE_PARITY_FAIL: bad={n_bad} ok={n_ok}")
    return n_ok


def tempo_is_derived_representation(obs):
    """tempo 只使用 B0 已有列（Location+Path），证明是 derived repr 而非新信息。"""
    k = (obs["bar_t"].to_numpy(np.int64)
         - obs["start_bar"].to_numpy(np.int64))
    net = (obs["start_up_distance_R"].to_numpy(float)
           - obs["cur_up_distance_R"].to_numpy(float))
    tv = obs["path_total_variation_R"].to_numpy(float)
    ss = obs["tempo_signed_speed"].to_numpy(float)
    as_ = obs["tempo_abs_speed"].to_numpy(float)
    se = obs["tempo_signed_efficiency"].to_numpy(float)
    ae = obs["tempo_abs_efficiency"].to_numpy(float)
    ok = (np.allclose(ss, net / np.maximum(k, 1))
          and np.allclose(as_, np.abs(net / np.maximum(k, 1)))
          and np.allclose(se, net / np.maximum(tv, 1e-12))
          and np.allclose(ae, np.abs(net / np.maximum(tv, 1e-12))))
    if not ok:
        raise SystemExit("STOP_MARKET_STATE1_1_TEMPO_NOT_DERIVED")
    return bool(ok)


def main():
    t_total = time.perf_counter()
    timing = {}

    # ---------------------------------------------------- load
    t0 = time.perf_counter()
    obs_base = pd.read_parquet(CACHE / "pgm_bar0_samples.parquet")
    enriched_path = CACHE / "market_state1_samples.parquet"
    if enriched_path.exists():
        obs = pd.read_parquet(enriched_path)
    else:
        ep0 = pd.read_parquet(CACHE / "episode0_episodes.parquet")
        ep3 = pd.read_parquet(CACHE / "episode_repl0_through_tb3.parquet")
        map0 = {(r.symbol, int(r.start_bar)): (int(r.start_upper_group),
                 int(r.start_lower_group)) for r in ep0.itertuples()}
        map3 = {(r.symbol, int(r.start_bar)): (int(r.start_upper_group),
                 int(r.start_lower_group)) for r in ep3.itertuples()}
        obs = build_enriched(obs_base, map0, map3)
        obs.to_parquet(enriched_path, index=False)
    ep0 = pd.read_parquet(CACHE / "episode0_episodes.parquet")
    ep3 = pd.read_parquet(CACHE / "episode_repl0_through_tb3.parquet")
    map0 = {(r.symbol, int(r.start_bar)): (int(r.start_upper_group),
             int(r.start_lower_group)) for r in ep0.itertuples()}
    map3 = {(r.symbol, int(r.start_bar)): (int(r.start_upper_group),
             int(r.start_lower_group)) for r in ep3.itertuples()}
    ep_full0 = {(r.symbol, int(r.start_bar)): r for r in ep0.itertuples()}
    ep_full3 = {(r.symbol, int(r.start_bar)): r for r in ep3.itertuples()}
    timing["cache_load_seconds"] = round(time.perf_counter() - t0, 2)

    # ---------------------------------------------------- sample parity
    n_ep = int(obs_base["episode_id"].nunique())
    n_rows = int(len(obs_base))
    n_term = int(obs_base["hazard"].sum())
    if (n_ep, n_rows, n_term) != (37987, 359714, 37987):
        raise SystemExit(
            f"STOP_MARKET_STATE1_1_SAMPLE_PARITY_FAIL: {n_ep},{n_rows},{n_term}")

    # ---------------------------------------------------- A. exact closure
    t_c = time.perf_counter()
    identity = exact_identity_closure(obs_base, ep_full0, ep_full3)
    prov_closure = provenance_group_id_closure(obs, map0, map3)
    timing["identity_closure_seconds"] = round(time.perf_counter() - t_c, 2)
    print(f"[IDENTITY CLOSURE] {identity}  prov_group_id={prov_closure}")

    # ---------------------------------------------------- provenance parity
    t_p = time.perf_counter()
    n_parity = provenance_canonical_parity()
    timing["provenance_parity_seconds"] = round(time.perf_counter() - t_p, 2)
    print(f"[PROVENANCE PARITY] ok={n_parity}")

    # ---------------------------------------------------- tempo derived
    tempo_derived = tempo_is_derived_representation(obs)

    # ---------------------------------------------------- first-bar residual
    RESID = [c for c in PROV_DELTA if "age_residual" in c or "count_delta" in c]
    first = (obs["bar_t"].to_numpy(np.int64)
             - obs["start_bar"].to_numpy(np.int64)) == 0
    max_resid = float(max(
        np.abs(obs.loc[first, c].to_numpy()).max() for c in RESID))
    if max_resid > 1e-9:
        raise SystemExit(
            f"STOP_MARKET_STATE1_1_PROVENANCE_LEAK: {max_resid}")

    # ---------------------------------------------------- windows + models
    res = []
    for w in WINDOWS:
        tr = obs[obs["block"].isin(w["train"])].reset_index(drop=True)
        ev = obs[obs["block"] == w["eval"]].reset_index(drop=True)
        if not len(tr) or not len(ev):
            raise SystemExit(f"STOP_MARKET_STATE1_1_EMPTY_SPLIT: {w['name']}")
        if w["eval"] in set(tr["block"]):
            raise SystemExit(f"STOP_MARKET_STATE1_1_EVAL_IN_FIT: {w['name']}")

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
                    f"STOP_MARKET_STATE1_1_EPISODE_MISALIGN: {label}")
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
    mtab = pd.DataFrame([dict(window=r["win"]["name"], model=name,
                             **mo["metrics"])
                         for r in res for name, mo in r["model_out"].items()])
    mtab.to_csv(OUT / "market_state1_1_model_metrics.csv", index=False)

    boots_df = pd.DataFrame([b for r in res for b in r["boots"]])
    boots_df.to_csv(OUT / "market_state1_1_bootstrap.csv", index=False)

    bysym_df = pd.DataFrame([b for r in res for b in r["by_sym"]])
    bysym_df.to_csv(OUT / "market_state1_1_by_symbol.csv", index=False)

    pd.DataFrame([dict(n_points=int(n_parity), ok=True)]).to_csv(
        OUT / "market_state1_1_provenance_parity.csv", index=False)

    # feature audit on compact vs dropped
    fa_rows = []
    for c in COMPACT_PROV + DROPPED_PROV:
        v = obs[c].to_numpy(float)
        fa_rows.append(dict(feature=c, kind=("compact" if c in COMPACT_PROV
                                             else "dropped"),
                            non_zero_rate=float(np.mean(np.abs(v) > 1e-9)),
                            mean=float(np.mean(v)), std=float(np.std(v))))
    for c in TEMPO:
        v = obs[c].to_numpy(float)
        fa_rows.append(dict(feature=c, kind="tempo_derived_repr",
                            non_zero_rate=float(np.mean(np.abs(v) > 1e-9)),
                            mean=float(np.mean(v)), std=float(np.std(v))))
    pd.DataFrame(fa_rows).to_csv(
        OUT / "market_state1_1_feature_audit.csv", index=False)

    opt_rows = []
    for r in res:
        for name, mo in r["model_out"].items():
            a = mo["aud"]
            opt_rows.append(dict(window=r["win"]["name"], model=name,
                                n_params=int(a["n_params"]),
                                n_iter=int(a["n_iter"]),
                                final_fun=float(a["final_fun"]),
                                success=bool(a["success"]),
                                elapsed_seconds=round(a["elapsed_seconds"], 2)))
    pd.DataFrame(opt_rows).to_csv(
        OUT / "market_state1_1_optimizer_audit.csv", index=False)

    # ---------------------------------------------------- verdict
    verdict = {}
    for label, gate, hi, lo in COMPARISONS:
        ws = boots_df[boots_df["comparison"] == label]
        ci_both = bool((ws["ci_hi"] < 0).all())
        sym_neg = {}
        for wname in [r["win"]["name"] for r in res]:
            sub = bysym_df[(bysym_df["window"] == wname)
                           & (bysym_df["comparison"] == label)]
            sym_neg[wname] = int((sub["mean_delta"] < 0).sum())
        supported = ci_both and all(v >= 10 for v in sym_neg.values())
        verdict[gate] = dict(
            supported=supported,
            windows={r["win"]["name"]: dict(
                ci_hi=float(ws[ws["window"] == r["win"]["name"]]["ci_hi"].iloc[0]))
                for r in res},
            n_symbols_delta_neg=sym_neg)

    # ---------------------------------------------------- B0 parity guard
    pgm0 = json.loads((OUT / "pgm_bar0_summary.json").read_text())
    b0_out = {r["win"]["name"]: r["model_out"][B0]["metrics"] for r in res}
    parity_fail = False
    for wname in [r["win"]["name"] for r in res]:
        ref = pgm0["model_metrics"][wname]["M2_INTRA_EPISODE_PATH"]
        for k in ("mean_episode_nll", "hazard_nll", "endpoint_joint_nll"):
            if abs(float(b0_out[wname][k]) - float(ref[k])) > 1e-6:
                parity_fail = True
    if parity_fail:
        raise SystemExit("STOP_MARKET_STATE1_1_BASELINE_PARITY_FAIL")

    # gain retention: compact vs full provenance standalone delta
    def delta_of(comp):
        row = boots_df[boots_df["comparison"] == comp]
        return {r["window"]: float(r["delta_sample_mean"])
                for _, r in row.iterrows()}
    pfull = delta_of("PF-B0")
    pcomp = delta_of("Pc-B0")

    identity_audit = dict(
        exact_episode_identity=identity,
        provenance_group_id_closure=prov_closure,
        note=("end_bar==bar_t+1, start_block==block, event_mask==target_mask "
              "for all 37987 terminals; provenance residuals recomputed from "
              "frozen start_upper/lower_group reproduce stored columns exactly"),
        base_commit=BASE_SHA,
    )

    summary = dict(
        experiment="MARKET-STATE-1.1 State Representation Closure",
        parent_commit="31e651c36d65d0b395f4c29a7addfbd239d4f7fb",
        corrections_from_audit=[
            "tempo is a derived nonlinear representation of B0 Location+Path, "
            "NOT new market-state information",
            "oldest_log_age_residual (x2) are always 0 -> dropped",
            "current_oldest_age_zero (x2) are highly determined by start+elapsed "
            "-> dropped; compact provenance = 6 truly-dynamic features",
            "exact episode identity closure now covers end_bar/start_block/"
            "event_mask + provenance group-id, not only event_mask",
            "'minimal sufficient state' is NOT claimed; this is closure only",
        ],
        baseline="PGM-BAR-0 M2 (reproduced by B0)",
        models={k: dict(n_numeric=len(v[0]), n_categorical=len(v[1]),
                        numeric=v[0], categorical=v[1],
                        role=("derived_representation" if "TEMPO" in k
                              else "new_state_information"
                              if "PROV" in k else "baseline"))
                for k, v in MODELS.items()},
        compact_provenance=COMPACT_PROV,
        dropped_provenance=DROPPED_PROV,
        tempo_features=TEMPO,
        tempo_classification=("DERIVED_REPRESENTATION: speed=net_move/elapsed, "
                              "efficiency=net_move/TV; net_move/Location, "
                              "elapsed/Path, TV/Path are all in B0 M2_NUM"),
        windows=[dict(name=w["name"], train=w["train"], eval=w["eval"],
                      seed=w["seed"]) for w in WINDOWS],
        primary_metric="mean episode NLL",
        model_metrics={r["win"]["name"]: {name: mo["metrics"]
                       for name, mo in r["model_out"].items()}
                       for r in res},
        bootstrap=boots_df.to_dict(orient="records"),
        by_symbol={w["name"]: {label: dict(
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
            for w in WINDOWS},
        gain_retention=dict(
            full_provenance_delta=pfull,
            compact_provenance_delta=pcomp,
            retained_fraction={w: (pcomp[w] / pfull[w] if pfull[w] else None)
                              for w in pfull},
        ),
        verdict=verdict,
        identity_audit=identity_audit,
        b0_parity_vs_pgm_bar0_m2="PASS" if not parity_fail else "FAIL",
        tempo_derived_representation=tempo_derived,
        bootstrap_reps=BOOTSTRAP_REPS,
    )
    (OUT / "market_state1_1_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    # timing
    timing["total_seconds"] = round(time.perf_counter() - t_total, 2)
    s2 = json.loads((OUT / "market_state1_1_summary.json").read_text())
    s2["timing"] = timing
    (OUT / "market_state1_1_summary.json").write_text(
        json.dumps(s2, indent=2, default=str))
    (OUT / "market_state1_1_identity_audit.json").write_text(
        json.dumps(identity_audit, indent=2, default=str))

    print(f"[MODEL METRICS]\n{mtab.to_string(index=False)}")
    print(f"[BOOTSTRAP]\n{boots_df.to_string(index=False)}")
    print(f"[VERDICT] {verdict}")
    print(f"[GAIN RETENTION] full={pfull} compact={pcomp}")
    print(f"[TIMING] {timing}")
    print(f"[DONE] -> {OUT}")


if __name__ == "__main__":
    main()
