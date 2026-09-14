"""REPL-0 — Frozen replication of EPISODE-0 / PATH-0 definitions on TB3

===========================================================================
唯一目标
===========================================================================
把已经冻结的 EPISODE-0 contract 与 PATH-0 model feature sets
**原封不动**应用到此前未使用的 TB3，做一次 untouched replication。

    TB1 = early development
    TB2 = development / model-selection validation
    TB3 = untouched replication（本轮消耗）
    TB4 = final holdout（绝对不碰）

禁止根据 TB3 结果调整：
    episode contract / features / hyper-parameters / event representation /
    duration representation / path features / bootstrap method

===========================================================================
复现问题（预注册，不因结果改 primary）
===========================================================================
    REPL-A  M1 - M0   previous structural endpoint increment 是否仍在
    REPL-B  M3 - M2   simple path morphology 是否仍然没有明确增量
    REPL-C  M2 - M1   duration effect 复查
    （附）   M3 - M0   total previous-history increment

模型与特征直接 import 自冻结 PATH-0 模块，保证公式级 parity（不是重写）。
episode 直接调用冻结 EPISODE-0 的 build_episodes_symbol（不是重写）。

===========================================================================
TB4 隔离
===========================================================================
每个 symbol 的所有数组（raw bars / pair sequence / path 源）
一律**切片到 TB3 最后一根 bar**，因此 TB4 的数值从未被读取。
构造范围 TB1+TB2+TB3 只为保证 TB2→TB3 边界上的 episode chaining 连续。

===========================================================================
输出
===========================================================================
    repl0_summary.json
    repl0_episode_audit.json
    repl0_model_metrics.csv
    repl0_bootstrap.csv
    repl0_per_bit.csv
    repl0_by_symbol.csv
（大型 episode_repl0_through_tb3.parquet / repl0_samples.parquet 存 gitignored）
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

from research.liquidity_oracle_atlas.run_5m_graph_probability_v1 import (  # noqa: E402
    load_raw_bars,
)
from research.liquidity_oracle_atlas.experiment_local_liquidity_transition_v0 import (  # noqa: E402
    FULL_UNIV, TRAIN_BLOCK, TEST_BLOCK, OUT, CACHE, build_blocks,
)
from research.liquidity_oracle_atlas.experiment_step01_cause_decomposition_v1 import (  # noqa: E402
    load_seq,
)
from research.liquidity_oracle_atlas.experiment_episode0_dynamic_episodes_v1 import (  # noqa: E402
    build_episodes_symbol,
)
from research.liquidity_oracle_atlas.experiment_path0_episode_path_memory_v1 import (  # noqa: E402
    MODELS, CUR_NUM, PATH_NUM, PREV_CAT, M2_EXTRA_NUM, BIT_NAMES, BIT_MASKS,
    M0, M1, M2, M3, COMPARISONS, REVIEWER_FROZEN_EPISODE_HASH,
    episode_identity_hash, path_morphology, make_pipeline,
    binary_logloss, binary_brier, ece_binary,
)

REPL_BLOCK = "TB3"
DEV_BLOCKS = (TRAIN_BLOCK, TEST_BLOCK)
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 20260915

AUDIT_COLS = ["symbol", "start_bar", "end_bar", "event_mask"]


def hash_keys(df: pd.DataFrame) -> str:
    k = (df["symbol"].astype(str) + "|" + df["start_bar"].astype(str) + "|"
         + df["end_bar"].astype(str) + "|" + df["event_mask"].astype(str))
    h = hashlib.sha256()
    h.update("\n".join(sorted(k.tolist())).encode())
    return h.hexdigest()


def main():
    t_total = time.perf_counter()
    timing = {}
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    # ------------------------------------------------------------------ load
    t0 = time.perf_counter()
    bars_full = {s: load_raw_bars(s) for s in FULL_UNIV}
    all_days, day_block_code, boundaries = build_blocks(bars_full)
    block_names = np.array([f"TB{i + 1}" for i in range(4)], dtype=object)
    frozen = pd.read_parquet(CACHE / "episode0_episodes.parquet")
    frozen_hash = episode_identity_hash(frozen)
    print(f"[FROZEN EPISODE HASH] {frozen_hash}")
    if frozen_hash != REVIEWER_FROZEN_EPISODE_HASH:
        raise SystemExit(
            "STOP_REPL0_EPISODE_PARITY_FAIL: frozen file does not match the "
            f"reviewer hash ({frozen_hash})")
    timing["load_seconds"] = round(time.perf_counter() - t0, 2)

    # ---------------------------------------- per-symbol TB3 cutoff slicing
    sl_by_sym = {}
    for s in FULL_UNIV:
        td = np.asarray(bars_full[s]["td"]).astype("datetime64[D]")
        code = day_block_code[np.searchsorted(all_days, td)]
        tb3_end = int(np.flatnonzero(code <= 2)[-1])
        tb4 = np.flatnonzero(code == 3)
        if tb4.size and int(tb4[0]) != tb3_end + 1:
            raise SystemExit("STOP_REPL0_TB4_GAP_LAYOUT")
        sl_by_sym[s] = dict(code=code, tb3_end=tb3_end)

    # ---------------------------------------------- frozen pair-key parity
    mism = 0
    for s in FULL_UNIV:
        c = pd.read_parquet(CACHE / f"local0_samples_{s}.parquet")
        bi = c["decision_bar_index"].to_numpy()
        seq0, _ = load_seq(s)
        mism += int((seq0["upper_group"][bi]
                     != c["upper_group"].to_numpy()).sum())
        mism += int((seq0["lower_group"][bi]
                     != c["lower_group"].to_numpy()).sum())
        mism += int((~np.isclose(seq0["upper_price"][bi],
                                 c["upper_price"].to_numpy(float))).sum())
        mism += int((~np.isclose(seq0["lower_price"][bi],
                                 c["lower_price"].to_numpy(float))).sum())
    print(f"[PARITY] pair-key mismatches = {mism}")
    if mism:
        raise SystemExit("STOP_REPL0_PAIR_SEQUENCE_INCOMPATIBLE")

    # ------------------------------------------- build episodes through TB3
    t0 = time.perf_counter()
    tr_full = {s: dict(h=np.asarray(bars_full[s]["h"], float),
                       l=np.asarray(bars_full[s]["l"], float),
                       c=np.asarray(bars_full[s]["c"], float),
                       atr=np.asarray(bars_full[s]["atr"], float))
               for s in FULL_UNIV}
    # sliced (TB3-capped) views actually used everywhere below
    bars_sl = {}
    for s in FULL_UNIV:
        te = sl_by_sym[s]["tb3_end"]
        sl = slice(0, te + 1)
        bars_sl[s] = dict(
            h=tr_full[s]["h"][sl], l=tr_full[s]["l"][sl],
            c=tr_full[s]["c"][sl], atr=tr_full[s]["atr"][sl],
            disc=np.asarray(bars_full[s]["disc"], bool)[sl],
            block=block_names[sl_by_sym[s]["code"]][sl],
            end_time=(np.asarray(bars_full[s]["t"])[sl].astype("datetime64[ns]")
                      + np.timedelta64(5, "m")),
            td=np.asarray(bars_full[s]["td"]).astype("datetime64[D]")[sl],
        )

    all_eps = []
    for s in FULL_UNIV:
        seq, grp = load_seq(s)
        te = sl_by_sym[s]["tb3_end"]
        seq_s = {k: (v[:te + 1] if isinstance(v, np.ndarray) and v.ndim == 1
                     else v) for k, v in seq.items()}
        seq_s["n"] = te + 1
        b = bars_sl[s]
        eps, owner, st = build_episodes_symbol(
            s, seq_s, grp, b["h"], b["l"], b["disc"], b["c"], b["block"],
            b["end_time"], te)
        for e in eps:
            e["crosses_tb1_tb2"] = bool(e["start_block"] == TRAIN_BLOCK
                                        and e["end_block"] == TEST_BLOCK)
            e["crosses_tb2_tb3"] = bool(e["start_block"] == TEST_BLOCK
                                        and e["end_block"] == REPL_BLOCK)
        all_eps += eps
    new_ep = pd.DataFrame(all_eps)
    timing["episode_extend_seconds"] = round(time.perf_counter() - t0, 2)

    if (new_ep["end_bar"] > np.array([sl_by_sym[s]["tb3_end"]
                                      for s in new_ep["symbol"]])).any():
        raise SystemExit("STOP_REPL0_TB4_ENDPOINT")

    # ------------------------------------------------------ episode parity
    frozen_nc = frozen[~frozen["censor_analysis_end"]].reset_index(drop=True)
    new_tb12 = new_ep[new_ep["start_block"].isin(DEV_BLOCKS)
                      & new_ep["end_block"].isin(DEV_BLOCKS)].reset_index(
                          drop=True)
    h_frozen_nc = hash_keys(frozen_nc)
    h_new_tb12 = hash_keys(new_tb12)
    same_set = (set(map(tuple, frozen_nc[AUDIT_COLS].to_numpy(object).tolist()))
                == set(map(tuple, new_tb12[AUDIT_COLS].to_numpy(object).tolist())))
    if not (h_frozen_nc == h_new_tb12 and same_set
            and len(frozen_nc) == len(new_tb12)):
        raise SystemExit("STOP_REPL0_EPISODE_PARITY_FAIL")

    # 唯一预期差异：13 条 frozen CENSOR_ANALYSIS_END episode 在更长的窗口下继续
    term = frozen[frozen["censor_analysis_end"]].reset_index(drop=True)
    ext_rows = []
    for r in term.itertuples():
        m = new_ep[(new_ep["symbol"] == r.symbol)
                   & (new_ep["start_bar"] == r.start_bar)]
        if len(m) != 1:
            raise SystemExit("STOP_REPL0_EPISODE_PARITY_FAIL: terminal chain")
        nrow = m.iloc[0]
        if not (int(nrow["end_bar"]) > int(r.end_bar)
                and int(nrow["start_bar"]) == int(r.start_bar)):
            raise SystemExit(
                "STOP_REPL0_EPISODE_PARITY_FAIL: terminal not extended")
        ext_rows.append(dict(
            symbol=r.symbol, start_bar=int(r.start_bar),
            frozen_end_bar=int(r.end_bar), extended_end_bar=int(nrow["end_bar"]),
            extended_end_block=str(nrow["end_block"]),
            extended_event_mask=int(nrow["event_mask"])))
    episode_audit = dict(
        frozen_episode_hash=frozen_hash,
        frozen_hash_matches_reviewer=True,
        n_frozen_episodes=int(len(frozen)),
        n_frozen_non_terminal=int(len(frozen_nc)),
        hash_frozen_non_terminal=h_frozen_nc,
        n_new_tb12_episodes=int(len(new_tb12)),
        hash_new_tb12_subset=h_new_tb12,
        tb12_subset_parity=bool(h_frozen_nc == h_new_tb12 and same_set),
        parity_definition=(
            "episodes fully inside TB1/TB2 and NOT the terminal "
            "CENSOR_ANALYSIS_END episode are byte-identical; extending the "
            "scan window can only rewrite each symbol's terminal censored "
            "episode, verified below as strict prefix extension"),
        n_terminal_censored_extended=int(len(term)),
        terminal_extensions=ext_rows,
        n_new_episodes_through_tb3=int(len(new_ep)),
        block_boundaries=boundaries,
    )
    print(f"[EPISODE PARITY] tb12_subset parity={episode_audit['tb12_subset_parity']} "
          f"n={len(new_tb12)} extended_terminal={len(term)}")
    new_ep.to_parquet(CACHE / "episode_repl0_through_tb3.parquet", index=False)

    # -------------------------------------------------------------- samples
    t0 = time.perf_counter()
    rows = []
    for sym, g in new_ep.groupby("symbol", sort=False):
        g = g.sort_values("start_bar").reset_index(drop=True)
        sb = g["start_bar"].to_numpy(np.int64)
        ebi = g["end_bar"].to_numpy(np.int64)
        gaf = g["gap_bars_after_episode"].to_numpy(np.int64)
        gbf = g["gap_bars_before_episode"].to_numpy(np.int64)
        ok = ((ebi[:-1] == sb[1:]) & (gaf[:-1] == 0) & (gbf[1:] == 0))
        pi = np.flatnonzero(ok)
        if not len(pi):
            continue
        b = bars_sl[sym]
        for a, bb in zip(pi, pi + 1):
            p = g.loc[int(a)]
            tt = g.loc[int(bb)]
            rows.append(dict(
                symbol=sym,
                prev_start_bar=int(p["start_bar"]),
                prev_end_bar=int(p["end_bar"]),
                prev_start_block=str(p["start_block"]),
                prev_end_block=str(p["end_block"]),
                prev_event_mask=int(p["event_mask"]),
                prev_duration_bars=int(p["duration_bars"]),
                prev_end_time=p["end_time"],
                target_start_bar=int(tt["start_bar"]),
                target_end_bar=int(tt["end_bar"]),
                target_start_block=str(tt["start_block"]),
                target_end_block=str(tt["end_block"]),
                target_event_mask=int(tt["event_mask"]),
                target_start_time=tt["start_time"],
                target_end_time=tt["end_time"],
                tmp_atr_prev=float(b["atr"][int(p["start_bar"])]),
                tmp_atr_cur=float(b["atr"][int(tt["start_bar"])]),
                tmp_cur_up=float(tt["start_upper_price"]),
                tmp_cur_dn=float(tt["start_lower_price"]),
                tmp_cur_close=float(tt["start_close"]),
                tmp_prev_up=float(p["start_upper_price"]),
                tmp_prev_dn=float(p["start_lower_price"]),
            ))
    sm = pd.DataFrame(rows)
    timing["sample_seconds"] = round(time.perf_counter() - t0, 2)
    n_pairs_total = int(len(sm))

    same_dev = (sm["target_start_block"] == sm["target_end_block"]) & \
        sm["target_start_block"].isin(DEV_BLOCKS)
    same_repl = (sm["target_start_block"] == REPL_BLOCK) & \
        (sm["target_end_block"] == REPL_BLOCK)
    n_target_censor = int((sm["target_event_mask"] == 0).sum())
    n_cross = int((~(same_dev | same_repl)).sum())
    n_dev_all = int(same_dev.sum())
    n_repl_all = int(same_repl.sum())
    sm = sm[sm["target_event_mask"] != 0].copy()
    same_dev = (sm["target_start_block"] == sm["target_end_block"]) & \
        sm["target_start_block"].isin(DEV_BLOCKS)
    same_repl = (sm["target_start_block"] == REPL_BLOCK) & \
        (sm["target_end_block"] == REPL_BLOCK)
    sm = sm[same_dev | same_repl].copy()
    n_after = int(len(sm))

    ok_atr = ((sm["tmp_atr_prev"] > 0) & np.isfinite(sm["tmp_atr_prev"])
              & (sm["tmp_atr_cur"] > 0) & np.isfinite(sm["tmp_atr_cur"]))
    n_bad_atr = int((~ok_atr).sum())
    sm = sm[ok_atr].reset_index(drop=True)

    # ------------------------------------------------------------- features
    t0 = time.perf_counter()
    cu = (sm["tmp_cur_up"] - sm["tmp_cur_close"]) / sm["tmp_atr_cur"]
    cd = (sm["tmp_cur_close"] - sm["tmp_cur_dn"]) / sm["tmp_atr_cur"]
    sm["cur_up_distance_R"] = cu
    sm["cur_down_distance_R"] = cd
    sm["cur_width_R"] = cu + cd
    sm["cur_log_distance_ratio"] = np.log((cu + 1e-8) / (cd + 1e-8))

    path_rows, max_src = [], []
    for sym, g in sm.groupby("symbol", sort=False):
        b = bars_sl[sym]
        for r in g.itertuples():
            pf = path_morphology(b["h"], b["l"], b["c"],
                                 b["atr"][r.prev_start_bar],
                                 r.tmp_prev_up, r.tmp_prev_dn,
                                 r.prev_start_bar, r.prev_end_bar)
            pf["_row"] = r.Index
            path_rows.append(pf)
            max_src.append(int(r.prev_end_bar) - int(r.target_start_bar))
    sm = sm.join(pd.DataFrame(path_rows).set_index("_row"))
    max_src = np.asarray(max_src, np.int64)
    timing["feature_seconds"] = round(time.perf_counter() - t0, 2)

    # ------------------------------------------------------- causal guards
    if int((sm["prev_end_bar"] != sm["target_start_bar"]).sum()):
        raise SystemExit("STOP_REPL0_PAIR_NOT_CONTIGUOUS")
    if int((sm["prev_event_mask"] == 0).sum()):
        raise SystemExit("STOP_REPL0_PREV_CENSOR_CONTIGUOUS")
    if int(max_src.max()) != 0:
        raise SystemExit("STOP_REPL0_PATH_FEATURE_FUTURE")
    if int((sm["target_event_mask"] == 0).sum()):
        raise SystemExit("STOP_REPL0_TARGET_CENSOR_LEAK")

    dev = sm[sm["target_start_block"].isin(DEV_BLOCKS)].reset_index(drop=True)
    rep = sm[sm["target_start_block"] == REPL_BLOCK].reset_index(drop=True)
    if not len(dev) or not len(rep):
        raise SystemExit("STOP_REPL0_EMPTY_SPLIT")
    if set(dev["target_start_block"]) & {REPL_BLOCK}:
        raise SystemExit("STOP_REPL0_TB3_IN_FIT")

    y_dev = np.stack([((dev["target_event_mask"].to_numpy() & m) != 0)
                      .astype(np.int64) for m in BIT_MASKS], axis=1)
    y_rep = np.stack([((rep["target_event_mask"].to_numpy() & m) != 0)
                      .astype(np.int64) for m in BIT_MASKS], axis=1)
    for k, nm in enumerate(BIT_NAMES):
        if len(np.unique(y_dev[:, k])) < 2:
            raise SystemExit(f"STOP_REPL0_DEV_BIT_ABSENT: {nm}")

    td_of = {s: bars_sl[s]["td"] for s in FULL_UNIV}
    rep_day = np.array([td_of[s][b] for s, b in
                        zip(rep["symbol"], rep["target_start_bar"])])

    # ------------------------------------------------------------------ fit
    t0 = time.perf_counter()
    P_rep = {n: np.zeros((len(rep), 4)) for n in MODELS}
    for name, (num_cols, cat_cols) in MODELS.items():
        cols = num_cols + cat_cols
        for k in range(4):
            pipe = make_pipeline(num_cols, cat_cols)
            pipe.fit(dev[cols], y_dev[:, k])     # fit on TB1+TB2 only
            P_rep[name][:, k] = pipe.predict_proba(rep[cols])[:, 1]
    timing["fit_seconds"] = round(time.perf_counter() - t0, 2)

    prior = y_dev.mean(axis=0)
    P_prior = np.tile(prior, (len(rep), 1))

    # -------------------------------------------------------------- metrics
    metric_rows = []
    for name in list(MODELS) + ["B_PRIOR"]:
        P = P_prior if name == "B_PRIOR" else P_rep[name]
        ll = [binary_logloss(y_rep[:, k], P[:, k]) for k in range(4)]
        br = [binary_brier(y_rep[:, k], P[:, k]) for k in range(4)]
        ec = [ece_binary(y_rep[:, k], P[:, k]) for k in range(4)]
        for k, nm in enumerate(BIT_NAMES):
            metric_rows.append(dict(
                model=name, bit=nm, n=len(rep),
                prevalence=float(y_rep[:, k].mean()), logloss=ll[k],
                brier=br[k], ece=ec[k], dev_prevalence=float(prior[k])))
        metric_rows.append(dict(
            model=name, bit="MEAN_BIT", n=len(rep),
            prevalence=float(y_rep.mean()), logloss=float(np.mean(ll)),
            brier=float(np.mean(br)), ece=float(np.mean(ec)),
            dev_prevalence=float(prior.mean())))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "repl0_model_metrics.csv", index=False)
    mtab = metrics[metrics["bit"] == "MEAN_BIT"].set_index("model")

    def sample_loss(P):
        out = np.zeros(len(y_rep))
        for k in range(4):
            q = np.clip(P[:, k], 1e-15, 1.0 - 1e-15)
            out += -(y_rep[:, k] * np.log(q)
                     + (1.0 - y_rep[:, k]) * np.log1p(-q))
        return out / 4.0

    losses = {n: sample_loss(P_rep[n]) for n in MODELS}
    uniq_days = np.unique(rep_day)
    day_pos = np.searchsorted(uniq_days, rep_day)
    nd = len(uniq_days)
    day_cnt = np.bincount(day_pos, minlength=nd)
    keep_days = day_cnt > 0
    nk = int(keep_days.sum())

    def boot_ci(dv):
        boot = np.empty(BOOTSTRAP_REPS)
        for b in range(BOOTSTRAP_REPS):
            boot[b] = dv[rng.integers(0, nk, nk)].mean()
        return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    t0 = time.perf_counter()
    boot_rows = []
    for hi_m, lo_m, label in COMPARISONS:
        dd = losses[hi_m] - losses[lo_m]
        dv = (np.bincount(day_pos, weights=dd, minlength=nd)[keep_days]
              / day_cnt[keep_days])
        lo, hi = boot_ci(dv)
        if hi < 0:
            v = "CI_below_zero"
        elif lo > 0:
            v = "CI_above_zero"
        else:
            v = "CI_contains_zero"
        boot_rows.append(dict(
            comparison=f"{hi_m} - {lo_m}", role=label,
            mean_bit_logloss_A=float(mtab.loc[hi_m, "logloss"]),
            mean_bit_logloss_B=float(mtab.loc[lo_m, "logloss"]),
            delta_sample_weighted=float(dd.mean()),
            delta_daily_mean=float(dv.mean()), n_days=nk,
            ci_lo=lo, ci_hi=hi, verdict=v))
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(OUT / "repl0_bootstrap.csv", index=False)

    # ------------------------------------------------------------- per bit
    per_bit = []
    for k, nm in enumerate(BIT_NAMES):
        def bce(P):
            q = np.clip(P[:, k], 1e-15, 1.0 - 1e-15)
            return -(y_rep[:, k] * np.log(q)
                     + (1.0 - y_rep[:, k]) * np.log1p(-q))
        dd = bce(P_rep[M3]) - bce(P_rep[M2])
        dv = (np.bincount(day_pos, weights=dd, minlength=nd)[keep_days]
              / day_cnt[keep_days])
        lo, hi = boot_ci(dv)
        d1 = bce(P_rep[M1]) - bce(P_rep[M0])
        dv1 = (np.bincount(day_pos, weights=d1, minlength=nd)[keep_days]
               / day_cnt[keep_days])
        lo1, hi1 = boot_ci(dv1)
        per_bit.append(dict(
            bit=nm, n=len(rep), prevalence=float(y_rep[:, k].mean()),
            M3_minus_M2=float(dd.mean()), M3_minus_M2_ci_lo=lo,
            M3_minus_M2_ci_hi=hi,
            M1_minus_M0=float(d1.mean()), M1_minus_M0_ci_lo=lo1,
            M1_minus_M0_ci_hi=hi1))
    per_bit_df = pd.DataFrame(per_bit)
    per_bit_df.to_csv(OUT / "repl0_per_bit.csv", index=False)
    timing["bootstrap_seconds"] = round(time.perf_counter() - t0, 2)

    # ----------------------------------------------------------- by symbol
    sym_rows = []
    for s, g in rep.groupby("symbol"):
        idx = g.index.to_numpy()
        sym_rows.append(dict(
            symbol=s, n=int(len(g)),
            mean_bit_logloss_M0=float(losses_samples(P_rep[M0], y_rep,
                                                     idx).mean()),
            mean_bit_logloss_M1=float(losses_samples(P_rep[M1], y_rep,
                                                     idx).mean()),
            mean_bit_logloss_M2=float(losses_samples(P_rep[M2], y_rep,
                                                     idx).mean()),
            mean_bit_logloss_M3=float(losses_samples(P_rep[M3], y_rep,
                                                     idx).mean()),
            delta_M1_M0=float((losses[M1][idx] - losses[M0][idx]).mean()),
            delta_M3_M2=float((losses[M3][idx] - losses[M2][idx]).mean())))
    sym_df = pd.DataFrame(sym_rows).sort_values("delta_M1_M0").reset_index(
        drop=True)
    sym_df.to_csv(OUT / "repl0_by_symbol.csv", index=False)

    # ------------------------------------------------- TB2 vs TB3 stability
    tb2_sum = json.loads((OUT / "path0_summary.json").read_text())
    tb2_delta = {r["comparison"]: r for r in tb2_sum["bootstrap"]}
    stability = []
    for r in boot_rows:
        t2 = tb2_delta.get(r["comparison"])
        stability.append(dict(
            comparison=r["comparison"],
            tb2_delta=None if t2 is None else t2["delta_daily_mean"],
            tb3_delta=r["delta_daily_mean"],
            tb2_ci=[None, None] if t2 is None else [t2["ci_lo"], t2["ci_hi"]],
            tb3_ci=[r["ci_lo"], r["ci_hi"]],
            same_sign=(None if t2 is None else
                       (np.sign(t2["delta_daily_mean"])
                        == np.sign(r["delta_daily_mean"])))))
    timing["total_seconds"] = round(time.perf_counter() - t_total, 2)

    # -------------------------------------------------------------- hashes
    k = (sm["symbol"].astype(str) + "|" + sm["prev_start_bar"].astype(str)
         + "|" + sm["prev_end_bar"].astype(str) + "|"
         + sm["target_start_bar"].astype(str) + "|"
         + sm["target_end_bar"].astype(str) + "|"
         + sm["target_event_mask"].astype(str))
    hh = hashlib.sha256()
    hh.update("\n".join(sorted(k.tolist())).encode())
    sample_key_sha256 = hh.hexdigest()
    sm.to_parquet(CACHE / "repl0_samples.parquet", index=False)

    # ------------------------------------------------------------- verdicts
    by = {r["comparison"]: r for r in boot_rows}
    a = by[f"{M1} - {M0}"]
    if a["ci_hi"] < 0:
        v_a = "PREV_ENDPOINT_INCREMENT_REPLICATED"
    elif a["ci_lo"] > 0:
        v_a = "PREV_ENDPOINT_INCREMENT_REVERSED"
    else:
        v_a = "PREV_ENDPOINT_INCREMENT_NOT_REPLICATED"
    b = by[f"{M3} - {M2}"]
    if b["ci_hi"] < 0:
        v_b = "PATH_MORPHOLOGY_INCREMENT_REPLICATED_POSITIVE"
    elif b["ci_lo"] > 0:
        v_b = "PATH_MORPHOLOGY_NEGATIVE_ON_TB3"
    else:
        v_b = "PATH_MORPHOLOGY_STILL_UNCONFIRMED"

    audit = dict(
        data_roles=dict(TB1="early development",
                        TB2="development / model-selection validation",
                        TB3="untouched replication (consumed by this round)",
                        TB4="final holdout (never read)"),
        frozen_episode_contract_commit=(
            "930b595c1aa2580d96c4c12444b70c2f0030fb73"),
        frozen_path0_commit="de7af87d55d01d2b8725578cfddbaa341f39429a",
        blocks=boundaries,
        n_pairs_total=n_pairs_total,
        dev_candidates=n_dev_all, repl_candidates=n_repl_all,
        excluded_target_censor=n_target_censor,
        excluded_cross_block_target=n_cross,
        excluded_invalid_atr=n_bad_atr,
        n_after_censor_and_block_exclusion=n_after,
        n_dev_final=int(len(dev)), n_repl_final=int(len(rep)),
        n_repl_prev_crosses_tb2_tb3=int(
            ((rep["prev_start_block"] == TEST_BLOCK)
             & (rep["prev_end_block"] == REPL_BLOCK)).sum()),
        n_unique_sample_keys=int(k.nunique()),
        n_path_increments_scanned=int(
            (sm["prev_end_bar"] - sm["prev_start_bar"]).sum()),
        tb4_read_for_values=False,
        tb4_cap_applied="all per-symbol arrays sliced to the last TB3 bar",
    )
    (OUT / "repl0_episode_audit.json").write_text(
        json.dumps(episode_audit, indent=2, default=str))

    summary = dict(
        experiment="REPL-0 frozen replication on TB3",
        base="de7af87d55d01d2b8725578cfddbaa341f39429a",
        frozen_episode_hash=frozen_hash,
        episode_parity=episode_audit,
        sample_audit=audit,
        feature_parity_note=(
            "MODELS / CUR_NUM / PATH_NUM / path_morphology / make_pipeline / "
            "metrics are imported directly from the frozen PATH-0 module"),
        metrics={k2: {c: float(v[c]) for c in
                      ["n", "prevalence", "logloss", "brier", "ece"]}
                 for k2, v in mtab.to_dict(orient="index").items()},
        bootstrap=boot_rows,
        per_bit=per_bit,
        stability_tb2_vs_tb3=stability,
        by_symbol=dict(
            n_symbol=len(sym_df),
            n_delta_M1_M0_negative=int((sym_df["delta_M1_M0"] < 0).sum()),
            n_delta_M1_M0_positive=int((sym_df["delta_M1_M0"] > 0).sum()),
            n_delta_M3_M2_negative=int((sym_df["delta_M3_M2"] < 0).sum()),
            n_delta_M3_M2_positive=int((sym_df["delta_M3_M2"] > 0).sum())),
        bootstrap_seed=BOOTSTRAP_SEED, bootstrap_reps=BOOTSTRAP_REPS,
        repl0_sample_key_sha256=sample_key_sha256,
        timing=timing,
        interpretation_limits=(
            "REPL-0 only replicates frozen EPISODE-0 / PATH-0 definitions on "
            "TB3. It does NOT establish SMC validity, path-signature value, "
            "event sequencing, latent state, tradability or RL. TB3 is spent "
            "after this run; TB4 remains final holdout."),
    )
    c = by[f"{M2} - {M1}"]
    if c["ci_hi"] < 0:
        v_c = "DURATION_EFFECT_REPLICATED_POSITIVE"
    elif c["ci_lo"] > 0:
        v_c = "DURATION_EFFECT_NEGATIVE_ON_TB3"
    else:
        v_c = "DURATION_EFFECT_NOT_ESTABLISHED"
    summary["REPL_A_prev_endpoint"] = v_a
    summary["REPL_B_path_morphology"] = v_b
    summary["REPL_C_duration"] = v_c
    (OUT / "repl0_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    print(f"[PARITY] {episode_audit['tb12_subset_parity']} "
          f"extended_terminal={len(term)}")
    print(f"[AUDIT] {audit}")
    print(f"[METRICS mean-bit]\n"
          f"{mtab.reset_index()[['model', 'logloss', 'brier', 'ece']].to_string()}")
    print(f"[BOOTSTRAP]\n{boot_df.to_string(index=False)}")
    print(f"[PER-BIT]\n{per_bit_df.to_string(index=False)}")
    print(f"[STABILITY] {stability}")
    print(f"[BY-SYMBOL] {summary['by_symbol']}")
    print(f"[TIMING] {timing}")
    print(f"[REPL-A] {v_a}")
    print(f"[REPL-B] {v_b}")
    print(f"[REPL-C] {v_c}")
    print(f"[DONE] -> {OUT}")


def losses_samples(P, y, idx):
    out = np.zeros(len(idx))
    for k in range(4):
        q = np.clip(P[idx, k], 1e-15, 1.0 - 1e-15)
        out += -(y[idx, k] * np.log(q) + (1.0 - y[idx, k]) * np.log1p(-q))
    return out / 4.0


if __name__ == "__main__":
    main()
