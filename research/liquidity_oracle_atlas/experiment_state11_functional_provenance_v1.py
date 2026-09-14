"""STATE-1.1 — Functional (non-linear) current-boundary provenance encoding

===========================================================================
reviewer 提出的 functional-form closure
===========================================================================
STATE-1 显示 S3-S1 < 0，但 S1 把 age 当作 **raw linear continuous** 变量喂给
LogisticRegression，而 age 极度右偏（upper oldest: p50=7, max=29288），并且

    P(upper newest age == 0 | prev NEW_UPPER) = 99.89%

也就是说 exact age==0 是一个**离散阈值状态**，线性模型未必能表达。
因此 STATE-1 的 S3-S1 < 0 有两种解释：

    (a) 真正的跨 episode event memory
    (b) prev_event_mask 只是 current provenance 的**更好的非线性/离散编码**

本轮唯一问题：在 current provenance 能明确表达 exact age-zero 与 age skew 后，
prev_event_mask 是否仍然提供稳定增量？

===========================================================================
functional provenance（由 STATE-1 已冻结的 6 个变量确定性生成，共 10 个）
===========================================================================
    upper_oldest_log_age      = log1p(upper_oldest_age_bars)
    upper_newest_log_age      = log1p(upper_newest_age_bars)
    upper_newest_age_zero     = 1 if upper_newest_age_bars == 0 else 0
    upper_oldest_age_zero     = 1 if upper_oldest_age_bars == 0 else 0
    upper_n_active_identities （原 count）
    lower_* 完全对称（4 log-age + 4 zero flags + 2 counts）

预注册理由：log1p 处理极端右偏；exact-zero flag 专门表达"当前边界 / 最新
identity 就在本 bar 出生"这一离散状态。禁止其他 age bins，禁止 feature search。

===========================================================================
四层模型（冻结）
===========================================================================
    F0 = geometry
    F1 = geometry + 10 functional provenance
    F2 = geometry + prev_event_mask            （= STATE-1 S2 = PATH-0 M1）
    F3 = geometry + 10 functional provenance + prev_event_mask

    PRIMARY = F3 - F1
    secondary = F1-F0 / F2-F0 / F3-F2

Windows（与 STATE-1 完全一致）：
    A: fit TB1      -> eval TB2  (seed 20260916)
    B: fit TB1+TB2  -> eval TB3  (seed 20260917)

zero flags 预先固定放入 frozen numeric pipeline（median impute + StandardScaler），
不做"标准化/不标准化"的事后选择，也不引入新分支。

样本完全冻结：复用 gitignored repl0_samples.parquet，
sample hash 必须 == ddf034ed30a90597aa86dbad71d6397b1934d532afb348b8a93b406984787c12。

TB4 合同：tb4_analytically_used = false。
（loader 可为 block grid 读取完整 raw 文件；TB4 不进入 endpoint / sample inclusion /
 feature / label / fit / preprocessing fit / metric / bootstrap / decision diagnostic，
 且不输出任何 TB4 performance statistic。）

===========================================================================
输出
===========================================================================
    state11_summary.json
    state11_model_metrics.csv
    state11_bootstrap.csv
    state11_per_bit.csv
    state11_by_symbol.csv
    state11_coupling_tables.csv
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
    GEOM, CAT, group_provenance, sample_key_hash,
)

REPL_BLOCK = "TB3"
FROZEN_SAMPLE_HASH = (
    "ddf034ed30a90597aa86dbad71d6397b1934d532afb348b8a93b406984787c12")

RAW_PROV = ["upper_oldest_age_bars", "upper_newest_age_bars",
            "upper_n_active_identities", "lower_oldest_age_bars",
            "lower_newest_age_bars", "lower_n_active_identities"]
FUNC_PROV = [
    "upper_oldest_log_age", "upper_newest_log_age",
    "upper_newest_age_zero", "upper_oldest_age_zero",
    "upper_n_active_identities",
    "lower_oldest_log_age", "lower_newest_log_age",
    "lower_newest_age_zero", "lower_oldest_age_zero",
    "lower_n_active_identities",
]
assert len(FUNC_PROV) == 10

MODELS = {
    "F0_GEOMETRY": (GEOM, []),
    "F1_GEOMETRY_FUNCTIONAL_PROVENANCE": (GEOM + FUNC_PROV, []),
    "F2_GEOMETRY_PREV_EVENT": (GEOM, CAT),
    "F3_GEOMETRY_FUNCTIONAL_PROVENANCE_PREV_EVENT": (GEOM + FUNC_PROV, CAT),
}
F0, F1, F2, F3 = MODELS.keys()
COMPARISONS = [(F3, F1), (F1, F0), (F2, F0), (F3, F2)]

WINDOWS = [
    dict(name="A_TB1_to_TB2", train=["TB1"], eval=TEST_BLOCK, seed=20260916),
    dict(name="B_TB1TB2_to_TB3", train=["TB1", "TB2"], eval=REPL_BLOCK,
         seed=20260917),
]
BOOTSTRAP_REPS = 1000

EXPECT_A_F2_F0 = -0.007165155454171992
EXPECT_B_F2_F0 = -0.00813512230487083


def run_window(win, sm):
    tr = sm[sm["target_start_block"].isin(win["train"])].reset_index(drop=True)
    ev = sm[sm["target_start_block"] == win["eval"]].reset_index(drop=True)
    if not len(tr) or not len(ev):
        raise SystemExit(f"STOP_STATE11_EMPTY_SPLIT: {win['name']}")
    if win["eval"] in set(tr["target_start_block"]):
        raise SystemExit(f"STOP_STATE11_EVAL_IN_FIT: {win['name']}")

    y_tr = np.stack([((tr["target_event_mask"].to_numpy() & m) != 0)
                     .astype(np.int64) for m in BIT_MASKS], axis=1)
    y_ev = np.stack([((ev["target_event_mask"].to_numpy() & m) != 0)
                     .astype(np.int64) for m in BIT_MASKS], axis=1)
    for k, nm in enumerate(BIT_NAMES):
        if len(np.unique(y_tr[:, k])) < 2:
            raise SystemExit(f"STOP_STATE11_TRAIN_BIT_ABSENT: "
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
            role="PRIMARY" if (hi_m, lo_m) == (F3, F1) else "SECONDARY",
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
        d3 = bce(P[F3]) - bce(P[F1])
        d2 = bce(P[F2]) - bce(P[F0])
        dv3 = np.bincount(pos, weights=d3, minlength=nd)[keep] / cnt[keep]
        dv2 = np.bincount(pos, weights=d2, minlength=nd)[keep] / cnt[keep]
        lo3, hi3 = boot(dv3)
        lo2, hi2 = boot(dv2)
        per_bit.append(dict(
            window=win["name"], bit=nm, n=len(ev),
            prevalence=float(y_ev[:, k].mean()),
            F3_minus_F1=float(d3.mean()), F3_minus_F1_ci_lo=lo3,
            F3_minus_F1_ci_hi=hi3,
            F2_minus_F0=float(d2.mean()), F2_minus_F0_ci_lo=lo2,
            F2_minus_F0_ci_hi=hi2))

    for s, g in ev.groupby("symbol"):
        idx = g.index.to_numpy()
        by_sym.append(dict(
            window=win["name"], symbol=s, n=int(len(g)),
            mean_bit_logloss_F0=float(losses[F0][idx].mean()),
            mean_bit_logloss_F1=float(losses[F1][idx].mean()),
            mean_bit_logloss_F2=float(losses[F2][idx].mean()),
            mean_bit_logloss_F3=float(losses[F3][idx].mean()),
            delta_F3_minus_F1=float((losses[F3][idx] - losses[F1][idx]).mean()),
            delta_F2_minus_F0=float((losses[F2][idx] - losses[F0][idx]).mean())))

    return dict(win=win, n_train=int(len(tr)), n_eval=int(len(ev)),
                n_days=nk, mtab=mtab, metrics=metrics, boots=boots,
                per_bit=per_bit, by_sym=by_sym,
                fit_seconds=fit_seconds,
                bootstrap_seconds=time.perf_counter() - t_boot)


def coupling_row(prev_flag, zero_flag, sm, prev_col, zero_col, table):
    return [dict(table=table, prev_condition=f"{prev_col} == 1",
                 n_prev=int(prev_flag.sum()),
                 n_zero_flag_1=int((prev_flag & zero_flag).sum()),
                 n_zero_flag_0=int((prev_flag & ~zero_flag).sum()),
                 rate_zero_flag_1=float(zero_flag[prev_flag].mean())
                 if prev_flag.any() else float("nan")),
            dict(table=table, prev_condition=f"{prev_col} == 0",
                 n_prev=int((~prev_flag).sum()),
                 n_zero_flag_1=int((~prev_flag & zero_flag).sum()),
                 n_zero_flag_0=int((~prev_flag & ~zero_flag).sum()),
                 rate_zero_flag_1=float(zero_flag[~prev_flag].mean()))]


def main():
    t_total = time.perf_counter()
    timing = {}

    # ------------------------------------------------- sample / cache parity
    t0 = time.perf_counter()
    sm = pd.read_parquet(CACHE / "repl0_samples.parquet")
    h_now = sample_key_hash(sm)
    if h_now != FROZEN_SAMPLE_HASH:
        raise SystemExit(
            f"STOP_STATE11_SAMPLE_PARITY_FAIL: {h_now}")
    st1 = json.loads((OUT / "state1_summary.json").read_text())
    if {r["comparison"] for r in st1["bootstrap"]} != {
            "S3_GEOMETRY_PROVENANCE_PREV_EVENT - S1_GEOMETRY_PROVENANCE",
            "S1_GEOMETRY_PROVENANCE - S0_GEOMETRY",
            "S2_GEOMETRY_PREV_EVENT - S0_GEOMETRY",
            "S3_GEOMETRY_PROVENANCE_PREV_EVENT - S2_GEOMETRY_PREV_EVENT"}:
        raise SystemExit("STOP_STATE11_FROZEN_CROSSCHECK_FAIL: STATE-1 table")
    if st1["sample_parity"]["sample_key_sha256"] != FROZEN_SAMPLE_HASH:
        raise SystemExit("STOP_STATE11_FROZEN_CROSSCHECK_FAIL: STATE-1 hash")

    ep = pd.read_parquet(CACHE / "episode_repl0_through_tb3.parquet")
    key = ep["symbol"].astype(str) + "|" + ep["start_bar"].astype(str)
    g_up = dict(zip(key, ep["start_upper_group"].astype(np.int64)))
    g_dn = dict(zip(key, ep["start_lower_group"].astype(np.int64)))

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

    # ------------------------------ recompute STATE-1 provenance exactly
    t0 = time.perf_counter()
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
                "STOP_STATE11_PROVENANCE_CONTRACT_FAIL: "
                f"{r.symbol} t={r.target_start_bar}")
        prov[i, :] = [pu[0], pu[1], pu[2], pl[0], pl[1], pl[2]]
    sm = sm.copy()
    for j, c in enumerate(RAW_PROV):
        sm[c] = prov[:, j]

    # ---- function-object parity proof vs STATE-1 stored audit ----
    pa = st1["provenance_audit"]
    chk = [
        (float(sm["upper_oldest_age_bars"].mean()),
         pa["upper_oldest_age_bars"]["mean"]),
        (float(sm["upper_oldest_age_bars"].median()),
         pa["upper_oldest_age_bars"]["p50"]),
        (float(sm["upper_oldest_age_bars"].max()),
         pa["upper_oldest_age_bars"]["max"]),
        (float(sm["upper_newest_age_bars"].mean()),
         pa["upper_newest_age_bars"]["mean"]),
        (float(sm["upper_newest_age_bars"].min()),
         pa["upper_newest_age_bars"]["min"]),
        (float(sm["upper_n_active_identities"].mean()),
         pa["upper_n_active_identities"]["mean"]),
        (float(sm["lower_oldest_age_bars"].mean()),
         pa["lower_oldest_age_bars"]["mean"]),
        (float(sm["lower_oldest_age_bars"].max()),
         pa["lower_oldest_age_bars"]["max"]),
        (float((sm["upper_newest_age_bars"] == 0).mean()),
         pa["frac_upper_newest_age_zero"]),
        (float((sm["lower_newest_age_bars"] == 0).mean()),
         pa["frac_lower_newest_age_zero"]),
    ]
    for got, exp in chk:
        if abs(got - exp) > 1e-9 * max(1.0, abs(exp)):
            raise SystemExit(
                "STOP_STATE11_FROZEN_CROSSCHECK_FAIL: provenance mismatch "
                f"({got} vs {exp})")

    # ------------------------------------- functional provenance features
    for side in ("upper", "lower"):
        sm[f"{side}_oldest_log_age"] = np.log1p(
            sm[f"{side}_oldest_age_bars"].to_numpy(float))
        sm[f"{side}_newest_log_age"] = np.log1p(
            sm[f"{side}_newest_age_bars"].to_numpy(float))
        sm[f"{side}_newest_age_zero"] = (
            sm[f"{side}_newest_age_bars"].to_numpy() == 0).astype(float)
        sm[f"{side}_oldest_age_zero"] = (
            sm[f"{side}_oldest_age_bars"].to_numpy() == 0).astype(float)
    sm["_eval_day"] = np.array(
        [td_by_sym[s][b] for s, b in
         zip(sm["symbol"], sm["target_start_bar"])], dtype="datetime64[D]")
    timing["provenance_feature_seconds"] = round(
        time.perf_counter() - t0, 2)

    # --------------------------------------------------- coupling audit
    nu = ((sm["prev_event_mask"].to_numpy() & 4) != 0)
    nl = ((sm["prev_event_mask"].to_numpy() & 8) != 0)
    uz_new = sm["upper_newest_age_zero"].to_numpy() == 1.0
    uz_old = sm["upper_oldest_age_zero"].to_numpy() == 1.0
    lz_new = sm["lower_newest_age_zero"].to_numpy() == 1.0
    lz_old = sm["lower_oldest_age_zero"].to_numpy() == 1.0
    cpl = []
    cpl += coupling_row(nu, uz_new, sm, "prev_NEW_UPPER",
                        "upper_newest_age_zero",
                        "prev_NEW_UPPER x upper_newest_age_zero")
    cpl += coupling_row(nl, lz_new, sm, "prev_NEW_LOWER",
                        "lower_newest_age_zero",
                        "prev_NEW_LOWER x lower_newest_age_zero")
    cpl += coupling_row(nu, uz_old, sm, "prev_NEW_UPPER",
                        "upper_oldest_age_zero",
                        "prev_NEW_UPPER x upper_oldest_age_zero")
    cpl += coupling_row(nl, lz_old, sm, "prev_NEW_LOWER",
                        "lower_oldest_age_zero",
                        "prev_NEW_LOWER x lower_oldest_age_zero")
    cpl_df = pd.DataFrame(cpl)
    cpl_df.to_csv(OUT / "state11_coupling_tables.csv", index=False)

    # ------------------------------------------------------------ windows
    res = []
    for win in WINDOWS:
        res.append(run_window(win, sm))
    timing["fit_seconds"] = round(sum(r["fit_seconds"] for r in res), 2)
    timing["bootstrap_seconds"] = round(
        sum(r["bootstrap_seconds"] for r in res), 2)

    metrics = pd.DataFrame([m for r in res for m in r["metrics"]])
    boots = pd.DataFrame([b for r in res for b in r["boots"]])
    per_bit = pd.DataFrame([p for r in res for p in r["per_bit"]])
    by_sym = pd.DataFrame([b for r in res for b in r["by_sym"]])
    metrics.to_csv(OUT / "state11_model_metrics.csv", index=False)
    boots.to_csv(OUT / "state11_bootstrap.csv", index=False)
    per_bit.to_csv(OUT / "state11_per_bit.csv", index=False)
    by_sym.to_csv(OUT / "state11_by_symbol.csv", index=False)

    # ------------------------------------------------------- cross-check
    f2a = float(boots[(boots["window"] == "A_TB1_to_TB2")
                      & (boots["comparison"] == f"{F2} - {F0}")
                      ]["delta_daily_mean"].iloc[0])
    f2b = float(boots[(boots["window"] == "B_TB1TB2_to_TB3")
                      & (boots["comparison"] == f"{F2} - {F0}")
                      ]["delta_daily_mean"].iloc[0])
    if abs(f2a - EXPECT_A_F2_F0) > 1e-15 or abs(f2b - EXPECT_B_F2_F0) > 1e-15:
        raise SystemExit(
            "STOP_STATE11_FROZEN_CROSSCHECK_FAIL: "
            f"F2-F0 A={f2a} (exp {EXPECT_A_F2_F0}), "
            f"B={f2b} (exp {EXPECT_B_F2_F0})")

    # ---------------------------------------------------------- verdict
    prim = boots[boots["role"] == "PRIMARY"].set_index("window")
    both_ok = bool((prim["ci_hi"] < 0).all())
    none_inc = bool((prim["ci_lo"] >= 0).all())
    if both_ok:
        v = "PREV_EVENT_MEMORY_BEYOND_FUNCTIONAL_PROVENANCE_SUPPORTED"
    elif none_inc:
        v = "PREV_EVENT_EFFECT_EXPLAINED_BY_FUNCTIONAL_PROVENANCE"
    else:
        v = "PREV_EVENT_MEMORY_NOT_SEPARATED_FROM_CURRENT_STATE_ENCODING"

    summary = dict(
        experiment="STATE-1.1 functional provenance encoding",
        base="047c4ed5e8771100332acd444070ebb8a27ebe9c",
        question=("does prev_event_mask still add stable information once the "
                  "current-boundary provenance can explicitly express exact "
                  "age-zero and age skew (log1p) in a linear model?"),
        models={k: dict(numeric=v[0], categorical=v[1])
                for k, v in MODELS.items()},
        functional_form=dict(
            log_age="log1p(age_bars)",
            zero_flag="1 if age_bars == 0 else 0",
            counts="raw active identity count",
            n_features=len(FUNC_PROV),
            zero_flag_policy=("zero flags enter the frozen numeric pipeline "
                              "(median impute + StandardScaler) together with "
                              "the other numerics; no post-hoc choice"),
            forbidden=["age bins", "feature search", "duration", "path",
                       "symbol", "type/scope", "volume"]),
        windows=[dict(name=w["name"], train=w["train"], eval=w["eval"],
                      seed=w["seed"]) for w in WINDOWS],
        bootstrap_reps=BOOTSTRAP_REPS,
        sample_parity=dict(n_samples=int(len(sm)), sample_key_sha256=h_now,
                           matches_frozen=bool(h_now == FROZEN_SAMPLE_HASH),
                           n_dev=int(sm["target_start_block"].isin(
                               ["TB1", "TB2"]).sum()),
                           n_repl=int((sm["target_start_block"]
                                       == REPL_BLOCK).sum())),
        provenance_recompute_identical=dict(
            note=("the 6 STATE-1 provenance variables were recomputed with the "
                  "very same function object (group_provenance imported from "
                  "the STATE-1 module) and validated against STATE-1's stored "
                  "audit values"),
            checks=len(chk), all_identical=True),
        coupling_tables=cpl,
        results=[dict(window=r["win"]["name"], n_train=r["n_train"],
                      n_eval=r["n_eval"], n_days=r["n_days"],
                      mean_bit_metrics=r["mtab"]) for r in res],
        bootstrap=boots.to_dict(orient="records"),
        per_bit_F3_minus_F1=per_bit.to_dict(orient="records"),
        by_symbol=dict(
            n_rows=len(by_sym), n_symbol_per_window=15,
            window_A_n_negative=int(((by_sym["window"] == "A_TB1_to_TB2")
                                     & (by_sym["delta_F3_minus_F1"] < 0)).sum()),
            window_A_n_positive=int(((by_sym["window"] == "A_TB1_to_TB2")
                                     & (by_sym["delta_F3_minus_F1"] > 0)).sum()),
            window_B_n_negative=int(((by_sym["window"] == "B_TB1TB2_to_TB3")
                                     & (by_sym["delta_F3_minus_F1"] < 0)).sum()),
            window_B_n_positive=int(((by_sym["window"] == "B_TB1TB2_to_TB3")
                                     & (by_sym["delta_F3_minus_F1"] > 0)).sum())),
        frozen_crosscheck=dict(
            window_A_F2_minus_F0=f2a, expected_A=EXPECT_A_F2_F0,
            window_B_F2_minus_F0=f2b, expected_B=EXPECT_B_F2_F0,
            state1_sample_hash_matches=bool(
                st1["sample_parity"]["sample_key_sha256"]
                == FROZEN_SAMPLE_HASH),
            exact=bool(abs(f2a - EXPECT_A_F2_F0) < 1e-15
                       and abs(f2b - EXPECT_B_F2_F0) < 1e-15)),
        tb4_analytically_used=False,
        tb4_note=("block grid uses the frozen LOCAL-0 definition and the "
                  "loader may read full raw files, but no TB4 value "
                  "contributes to episode endpoints, sample inclusion, "
                  "features, labels, fit, preprocessing fit, metrics, "
                  "bootstrap or decision diagnostics; TB4 remains blind for "
                  "final evaluation."),
        timing=timing,
        interpretation_limits=(
            "STATE-1.1 is a functional-form closure on already-consumed "
            "development evidence (TB1/TB2/TB3). It does NOT establish SEQ-1, "
            "Markov order, SMC, latent psychology, tradability or RL."),
    )
    summary["STATE11_VERDICT"] = v
    if both_ok:
        summary["STATE11_CONCLUSION"] = (
            "FIRST_ORDER_EVENT_MEMORY_CANDIDATE_SUPPORTED: previous structural "
            "endpoint contains predictive information beyond current geometry "
            "and an explicitly nonlinear representation of current boundary "
            "activation provenance. SEQ-1 is authorised by this round.")
    elif none_inc:
        summary["STATE11_CONCLUSION"] = (
            "previous event is largely a compact encoding of current boundary "
            "state. Do NOT enter SEQ-1.")
    else:
        summary["STATE11_CONCLUSION"] = (
            "prev_event increment is not separated from a nonlinear encoding "
            "of current state; do NOT enter SEQ-1.")
    timing["total_seconds"] = round(time.perf_counter() - t_total, 2)
    summary["timing"] = timing
    (OUT / "state11_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    print(f"[SAMPLE PARITY] {len(sm)} hash_match="
          f"{h_now == FROZEN_SAMPLE_HASH}")
    print(f"[COUPLING]\n{cpl_df.to_string(index=False)}")
    for r in res:
        print(f"[WINDOW {r['win']['name']}] train={r['n_train']} "
              f"eval={r['n_eval']} days={r['n_days']}")
        print(pd.DataFrame([dict(model=k, **v)
                            for k, v in r["mtab"].items()]).to_string(
                                index=False))
    print(f"[CROSSCHECK] F2-F0 A={f2a} B={f2b} exact=True")
    print(f"[BOOTSTRAP]\n{boots.to_string(index=False)}")
    print(f"[PER-BIT F3-F1]\n{per_bit.to_string(index=False)}")
    print(f"[BY-SYMBOL] {summary['by_symbol']}")
    print(f"[TIMING] {timing}")
    print(f"[VERDICT] {v}")
    print(f"[DONE] -> {OUT}")


if __name__ == "__main__":
    main()
