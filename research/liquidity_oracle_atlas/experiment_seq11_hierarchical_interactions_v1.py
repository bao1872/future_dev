"""SEQ-1.1 — Hierarchical ordered bit-interaction test (真正 nested 的二阶检验)

===========================================================================
reviewer 对 SEQ-1 的 correction
===========================================================================
SEQ-1 的 PRIMARY Q3-Q1 不是 nested interaction test：

    Q1 = X + E_n
    Q3 = X + Pair(E_{n-1}, E_n)      <- pair 替换了 E_n main effect

所以 Q3-Q1 同时做了两件事：(1) 加入二阶 joint 信息；(2) 把只需 15 个类别、
可大量池化的 E_n 主效应换成 97~108 个稀疏 pair 类别重新学习。
因此 SEQ-1 只能冻结为 "pair-only replacement representation not stable"，
不能宣布 "二阶 event 信息不存在"。

本轮唯一问题
------------
在保留 E_n 与 E_{n-1} **两个 main effects** 之后，固定的 4x4 = 16 个
**有序 structural-bit interaction** 是否仍对下一 endpoint 提供 OOS 增量？

    R0_CURRENT_STATE             = geometry + 10 functional provenance
    R1_FIRST_ORDER               = R0 + prev_event_mask(=E_n)   [= SEQ-1 Q1]
    R2_SECOND_ORDER_MAIN_EFFECTS = R1 + prevprev_event_mask     [= SEQ-1 Q2]
    R3_HIERARCHICAL_INTERACTIONS = R2 + 16 ordered bit interactions

    PRIMARY = R3 - R2

16 个 interaction（顺序不可交换）：
    interaction(i, j) = 1 iff (prevprev_event_mask & bit_i) != 0
                          AND (prev_event_mask & bit_j) != 0
    4 bits: UP_PEN=1, DOWN_PEN=2, NEW_UPPER=4, NEW_LOWER=8

禁止：三阶 sequence / 手工 same-side-opposite-side role / SMC / PATH Signature /
HMM-HSMM / latent state / RL / PnL / TB4 performance evaluation。
禁止根据结果删除或新增 interaction。

Windows（与 SEQ-1 一致）：A: fit TB1 -> eval TB2 (seed 20260920)
                          B: fit TB1+TB2 -> eval TB3 (seed 20260921)

样本完全冻结：seq1_samples.parquet，
hash 必须 == 7b28e3aa9a41a0b27e400cf9bb060f1048a8e3ebf95e0e33780855f190989067。

TB4：tb4_analytically_used = false。

===========================================================================
输出
===========================================================================
    seq11_summary.json
    seq11_model_metrics.csv
    seq11_bootstrap.csv
    seq11_per_bit.csv
    seq11_by_symbol.csv
    seq11_interaction_sparsity.csv
    seq11_interaction_coefficients.csv
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
    TEST_BLOCK, OUT, CACHE,
)
from research.liquidity_oracle_atlas.experiment_path0_episode_path_memory_v1 import (  # noqa: E402
    BIT_NAMES, BIT_MASKS, binary_logloss, binary_brier, ece_binary,
    make_pipeline,
)
from research.liquidity_oracle_atlas.experiment_state1_provenance_decomposition_v1 import (  # noqa: E402
    GEOM,
)
from research.liquidity_oracle_atlas.experiment_state11_functional_provenance_v1 import (  # noqa: E402
    FUNC_PROV,
)
from research.liquidity_oracle_atlas.experiment_seq1_second_order_grammar_v1 import (  # noqa: E402
    triple_hash,
)

REPL_BLOCK = "TB3"
FROZEN_SEQ1_HASH = (
    "7b28e3aa9a41a0b27e400cf9bb060f1048a8e3ebf95e0e33780855f190989067")

BIT_DEFS = [("UP_PEN", 1), ("DOWN_PEN", 2), ("NEW_UPPER", 4), ("NEW_LOWER", 8)]
INTERACTIONS = [f"pp_{a}_x_p_{b}" for a, _ in BIT_DEFS for b, _ in BIT_DEFS]
assert len(INTERACTIONS) == 16
N_INTER = 16

CURRENT = list(GEOM) + list(FUNC_PROV)
assert len(CURRENT) == 14

MODELS = {
    "R0_CURRENT_STATE": (CURRENT, []),
    "R1_FIRST_ORDER": (CURRENT, ["prev_event_mask"]),
    "R2_SECOND_ORDER_MAIN_EFFECTS": (
        CURRENT, ["prev_event_mask", "prevprev_event_mask"]),
    "R3_HIERARCHICAL_INTERACTIONS": (
        CURRENT + INTERACTIONS, ["prev_event_mask", "prevprev_event_mask"]),
}
R0, R1, R2, R3 = MODELS.keys()
COMPARISONS = [(R3, R2), (R2, R1), (R1, R0), (R3, R1)]

WINDOWS = [
    dict(name="A_TB1_to_TB2", train=["TB1"], eval=TEST_BLOCK, seed=20260920),
    dict(name="B_TB1TB2_to_TB3", train=["TB1", "TB2"], eval=REPL_BLOCK,
         seed=20260921),
]
BOOTSTRAP_REPS = 1000


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    pp = df["prevprev_event_mask"].to_numpy(np.int64)
    p = df["prev_event_mask"].to_numpy(np.int64)
    for a, ai in BIT_DEFS:
        base = (pp & ai) != 0
        for b, bi in BIT_DEFS:
            df[f"pp_{a}_x_p_{b}"] = (base & ((p & bi) != 0)).astype(float)
    return df


def interaction_coefficients(pipe, num_cols) -> np.ndarray:
    """从 fitted R3 pipeline 取 16 个 interaction 的（标准化后）系数。"""
    ct = pipe.named_steps["pre"]
    names = [c for tname, _, cols in ct.transformers_ if tname == "num"
             for c in cols]
    if names != list(num_cols):
        raise SystemExit("STOP_SEQ11_PIPELINE_LAYOUT_FAIL")
    coef = pipe.named_steps["clf"].coef_.ravel()
    i0 = len(num_cols) - N_INTER
    return coef[i0:i0 + N_INTER]


def run_window(win, sm):
    tr = sm[sm["target_start_block"].isin(win["train"])].reset_index(drop=True)
    ev = sm[sm["target_start_block"] == win["eval"]].reset_index(drop=True)
    if not len(tr) or not len(ev):
        raise SystemExit(f"STOP_SEQ11_EMPTY_SPLIT: {win['name']}")
    if win["eval"] in set(tr["target_start_block"]):
        raise SystemExit(f"STOP_SEQ11_EVAL_IN_FIT: {win['name']}")

    y_tr = np.stack([((tr["target_event_mask"].to_numpy() & m) != 0)
                     .astype(np.int64) for m in BIT_MASKS], axis=1)
    y_ev = np.stack([((ev["target_event_mask"].to_numpy() & m) != 0)
                     .astype(np.int64) for m in BIT_MASKS], axis=1)
    for k, nm in enumerate(BIT_NAMES):
        if len(np.unique(y_tr[:, k])) < 2:
            raise SystemExit(f"STOP_SEQ11_TRAIN_BIT_ABSENT: "
                             f"{win['name']} {nm}")

    # interaction sparsity（train 侧，固定 16 个，不做任何删除）
    spar = []
    for c in INTERACTIONS:
        v = tr[c].to_numpy(float)
        spar.append(dict(window=win["name"], interaction=c,
                         n_train=int(len(v)), n_positive=int((v > 0).sum()),
                         positive_rate=float((v > 0).mean())))

    P = {n: np.zeros((len(ev), 4)) for n in MODELS}
    coef_rows = []
    t_fit = time.perf_counter()
    for name, (num_cols, cat_cols) in MODELS.items():
        cols = num_cols + cat_cols
        for k, bitname in enumerate(BIT_NAMES):
            pipe = make_pipeline(num_cols, cat_cols)
            pipe.fit(tr[cols], y_tr[:, k])
            P[name][:, k] = pipe.predict_proba(ev[cols])[:, 1]
            if name == R3:
                coef = interaction_coefficients(pipe, num_cols)
                for c, b in zip(INTERACTIONS, coef):
                    coef_rows.append(dict(window=win["name"], bit=bitname,
                                          interaction=c, coefficient=float(b)))
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

    def boot(dv, seed):
        r = np.random.default_rng(seed)
        b = np.empty(BOOTSTRAP_REPS)
        for i in range(BOOTSTRAP_REPS):
            b[i] = dv[r.integers(0, nk, nk)].mean()
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

    for idx, (hi_m, lo_m) in enumerate(COMPARISONS):
        dd = losses[hi_m] - losses[lo_m]
        dv = np.bincount(pos, weights=dd, minlength=nd)[keep] / cnt[keep]
        lo, hi = boot(dv, win["seed"] + idx)
        boots.append(dict(
            window=win["name"], comparison=f"{hi_m} - {lo_m}",
            role="PRIMARY" if (hi_m, lo_m) == (R3, R2) else "SECONDARY",
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
        d3 = bce(P[R3]) - bce(P[R2])
        d1 = bce(P[R2]) - bce(P[R1])
        dv3 = np.bincount(pos, weights=d3, minlength=nd)[keep] / cnt[keep]
        dv1 = np.bincount(pos, weights=d1, minlength=nd)[keep] / cnt[keep]
        lo3, hi3 = boot(dv3, win["seed"] + 100)
        lo1, hi1 = boot(dv1, win["seed"] + 101)
        per_bit.append(dict(
            window=win["name"], bit=nm, n=len(ev),
            prevalence=float(y_ev[:, k].mean()),
            R3_minus_R2=float(d3.mean()), R3_minus_R2_ci_lo=lo3,
            R3_minus_R2_ci_hi=hi3,
            R2_minus_R1=float(d1.mean()), R2_minus_R1_ci_lo=lo1,
            R2_minus_R1_ci_hi=hi1))

    for s, g in ev.groupby("symbol"):
        idx = g.index.to_numpy()
        by_sym.append(dict(
            window=win["name"], symbol=s, n=int(len(g)),
            mean_bit_logloss_R2=float(losses[R2][idx].mean()),
            mean_bit_logloss_R3=float(losses[R3][idx].mean()),
            delta_R3_minus_R2=float((losses[R3][idx] - losses[R2][idx]).mean())))

    return dict(win=win, n_train=int(len(tr)), n_eval=int(len(ev)),
                n_days=nk, mtab=mtab, metrics=metrics, boots=boots,
                per_bit=per_bit, by_sym=by_sym, spar=spar,
                coef_rows=coef_rows, fit_seconds=fit_seconds,
                bootstrap_seconds=time.perf_counter() - t_boot)


def main():
    t_total = time.perf_counter()
    timing = {}

    # ------------------------------------------------- sample / hash parity
    t0 = time.perf_counter()
    sm = pd.read_parquet(CACHE / "seq1_samples.parquet")
    h = triple_hash(sm)
    if h != FROZEN_SEQ1_HASH:
        raise SystemExit(f"STOP_SEQ11_SAMPLE_PARITY_FAIL: {h}")
    seq1 = json.loads((OUT / "seq1_summary.json").read_text())
    q1q0 = {r["window"]: r["delta_daily_mean"]
            for r in seq1["bootstrap"]
            if r["comparison"].startswith("Q1_FIRST_ORDER - Q0")}
    q2q1 = {r["window"]: r["delta_daily_mean"]
            for r in seq1["bootstrap"]
            if r["comparison"].startswith("Q2_SECOND_ORDER_ADDITIVE - Q1")}
    if len(q1q0) != 2 or len(q2q1) != 2:
        raise SystemExit("STOP_SEQ11_FROZEN_CROSSCHECK_FAIL: SEQ-1 rows")
    t_feat = time.perf_counter()
    sm = add_interactions(sm.copy())
    missing = [c for c in CURRENT if c not in sm.columns]
    if missing:
        raise SystemExit(f"STOP_SEQ11_SAMPLE_PARITY_FAIL: missing {missing}")
    timing["cache_load_seconds"] = round(t_feat - t0, 2)
    timing["feature_seconds"] = round(time.perf_counter() - t_feat, 2)
    timing["raw_bars_loaded"] = False
    timing["note"] = ("no raw bar file is opened by this script at all; every "
                      "input comes from seq1_samples.parquet")

    # ------------------------------------------------------------ windows
    t0 = time.perf_counter()
    res = [run_window(w, sm) for w in WINDOWS]
    timing["fit_seconds"] = round(sum(r["fit_seconds"] for r in res), 2)
    timing["bootstrap_seconds"] = round(
        sum(r["bootstrap_seconds"] for r in res), 2)

    # ------------------------------------------- frozen cross-check (§7)
    boots = pd.DataFrame([b for r in res for b in r["boots"]])
    cc = {}
    for r in res:
        w = r["win"]["name"]
        got1 = float(boots[(boots["window"] == w)
                           & (boots["comparison"] == f"{R1} - {R0}")
                           ]["delta_daily_mean"].iloc[0])
        got2 = float(boots[(boots["window"] == w)
                           & (boots["comparison"] == f"{R2} - {R1}")
                           ]["delta_daily_mean"].iloc[0])
        cc[w] = dict(R1_minus_R0=got1, seq1_Q1_minus_Q0=q1q0[w],
                     R2_minus_R1=got2, seq1_Q2_minus_R1=q2q1[w],
                     exact=bool(abs(got1 - q1q0[w]) < 1e-15
                                and abs(got2 - q2q1[w]) < 1e-15))
        if not cc[w]["exact"]:
            raise SystemExit(
                "STOP_SEQ11_FROZEN_CROSSCHECK_FAIL: "
                f"{w} R1-R0={got1} (exp {q1q0[w]}), "
                f"R2-R1={got2} (exp {q2q1[w]})")

    # ------------------------------------------------------------ outputs
    metrics = pd.DataFrame([m for r in res for m in r["metrics"]])
    per_bit = pd.DataFrame([p for r in res for p in r["per_bit"]])
    by_sym = pd.DataFrame([b for r in res for b in r["by_sym"]])
    spar = pd.DataFrame([s for r in res for s in r["spar"]])
    coefs = pd.DataFrame([c for r in res for c in r["coef_rows"]])
    metrics.to_csv(OUT / "seq11_model_metrics.csv", index=False)
    boots.to_csv(OUT / "seq11_bootstrap.csv", index=False)
    per_bit.to_csv(OUT / "seq11_per_bit.csv", index=False)
    by_sym.to_csv(OUT / "seq11_by_symbol.csv", index=False)
    spar.to_csv(OUT / "seq11_interaction_sparsity.csv", index=False)
    coefs.to_csv(OUT / "seq11_interaction_coefficients.csv", index=False)

    spar_summary = {}
    for w, g in spar.groupby("window"):
        pr = g["positive_rate"].to_numpy()
        spar_summary[w] = dict(
            n_interactions=len(g),
            positive_rate_min=float(pr.min()),
            positive_rate_p50=float(np.median(pr)),
            positive_rate_max=float(pr.max()),
            n_always_zero_in_train=int((g["n_positive"] == 0).sum()),
            n_positive_min=int(g["n_positive"].min()),
            n_positive_max=int(g["n_positive"].max()))

    # ----------------------------------------------------------- verdict
    prim = boots[boots["role"] == "PRIMARY"].set_index("window")
    both_ok = bool((prim["ci_hi"] < 0).all())
    none_inc = bool((prim["ci_lo"] >= 0).all())
    if both_ok:
        v = "SECOND_ORDER_BIT_INTERACTION_SUPPORTED"
    elif none_inc:
        v = "FIRST_ORDER_EVENT_STATE_SUFFICIENT_WITHIN_TESTED_INTERACTION_GRAMMAR"
    else:
        v = "SECOND_ORDER_INTERACTION_NOT_STABLE"

    summary = dict(
        experiment="SEQ-1.1 hierarchical ordered bit interactions",
        base="e9405f8b004b87330f53d66763dedcb50495959e",
        question=("after keeping BOTH event main effects, do 16 fixed ordered "
                  "structural-bit interactions still add OOS information?"),
        reviewer_correction=(
            "SEQ-1 Q3-Q1 was not nested: Q3 replaced the E_n main effect with "
            "97-108 sparse pair categories. SEQ-1 can only be frozen as "
            "'pair-only replacement representation not stable'."),
        frozen_context=dict(
            seq1_sample_hash=h, seq1_sample_hash_ok=True,
            seq1_verdict=seq1["SEQ1_VERDICT"],
            seq1_frozen_conclusions=[
                "first-order E_n effect remains supported",
                "additive older-event E_{n-1} effect is not established",
                "full pair-category replacement representation is not stable"]),
        models={k: dict(numeric=v[0], categorical=v[1])
                for k, v in MODELS.items()},
        interactions=dict(
            n=len(INTERACTIONS), names=INTERACTIONS,
            definition=("interaction(i,j)=1 iff (prevprev_event_mask & bit_i)!=0 "
                        "AND (prev_event_mask & bit_j)!=0; ordered, "
                        "non-commutative; 4 bits UP_PEN=1 DOWN_PEN=2 "
                        "NEW_UPPER=4 NEW_LOWER=8"),
            policy=("fixed 16, never deleted or added based on results; always "
                    "kept alongside both main effects"),
            pipeline="numeric 0/1 with median imputer + StandardScaler"),
        windows=[dict(name=w["name"], train=w["train"], eval=w["eval"],
                      seed=w["seed"]) for w in WINDOWS],
        bootstrap_reps=BOOTSTRAP_REPS,
        bootstrap_note=("each comparison uses its own deterministic seed "
                        "(window seed + index); delta values are independent of "
                        "RNG and are what the frozen cross-check verifies"),
        frozen_crosscheck=cc,
        interaction_sparsity=spar_summary,
        results=[dict(window=r["win"]["name"], n_train=r["n_train"],
                      n_eval=r["n_eval"], n_days=r["n_days"],
                      mean_bit_metrics=r["mtab"]) for r in res],
        bootstrap=boots.to_dict(orient="records"),
        per_bit_R3_minus_R2=per_bit.to_dict(orient="records"),
        by_symbol=dict(
            n_rows=len(by_sym), n_symbol_per_window=15,
            window_A_n_negative=int(((by_sym["window"] == "A_TB1_to_TB2")
                                     & (by_sym["delta_R3_minus_R2"] < 0)).sum()),
            window_A_n_positive=int(((by_sym["window"] == "A_TB1_to_TB2")
                                     & (by_sym["delta_R3_minus_R2"] > 0)).sum()),
            window_B_n_negative=int(((by_sym["window"] == "B_TB1TB2_to_TB3")
                                     & (by_sym["delta_R3_minus_R2"] < 0)).sum()),
            window_B_n_positive=int(((by_sym["window"] == "B_TB1TB2_to_TB3")
                                     & (by_sym["delta_R3_minus_R2"] > 0)).sum())),
        coefficient_note=("post-StandardScaler coefficients are reported for "
                          "direction/relative interpretation only; they are "
                          "NOT economic effect sizes and were NOT used for "
                          "feature selection or re-fitting"),
        tb4_analytically_used=False,
        timing=timing,
        interpretation_limits=(
            "SEQ-1.1 is the final second-order closure on already-consumed "
            "development evidence. It does NOT prove a second-order Markov "
            "process, latent state, SMC validity, tradability or RL, and it "
            "does not authorise third-order history or HMM/HSMM."),
    )
    summary["SEQ11_VERDICT"] = v
    if both_ok:
        summary["SEQ11_CONCLUSION"] = (
            "a compact hierarchical representation of ordered two-event "
            "structural interactions adds predictive information beyond both "
            "event main effects (second-order bit-interaction candidate "
            "supported).")
        summary["NEXT_AUTHORISED"] = (
            "decide next round whether event-state compression / latent state "
            "is worth testing; still NO third-order history.")
    else:
        summary["SEQ11_CONCLUSION"] = (
            "no stable second-order interaction increment: formally stop "
            "second-order escalation and freeze the world-state history as "
            "current state + immediately previous structural endpoint. No "
            "third-order sequence, no FULL pair re-encoding, no pattern "
            "picking from the SEQ-1 transition tables.")
        summary["NEXT_AUTHORISED"] = (
            "no longer event history. Redirect effort either to improving the "
            "current-state representation or to testing whether the already "
            "stable first-order world model carries trading value.")
    timing["total_seconds"] = round(time.perf_counter() - t_total, 2)
    summary["timing"] = timing
    (OUT / "seq11_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    print(f"[SAMPLE] {len(sm)} hash_match={h == FROZEN_SEQ1_HASH}")
    print(f"[INTERACTION SPARSITY] {spar_summary}")
    print(f"[CROSSCHECK] {cc}")
    for r in res:
        print(f"[WINDOW {r['win']['name']}] train={r['n_train']} "
              f"eval={r['n_eval']} days={r['n_days']}")
        print(pd.DataFrame([dict(model=k, **v)
                            for k, v in r["mtab"].items()]).to_string(
                                index=False))
    print(f"[BOOTSTRAP]\n{boots.to_string(index=False)}")
    print(f"[PER-BIT R3-R2]\n{per_bit.to_string(index=False)}")
    print(f"[BY-SYMBOL] {summary['by_symbol']}")
    print(f"[TIMING] {timing}")
    print(f"[VERDICT] {v}")
    print(f"[DONE] -> {OUT}")


if __name__ == "__main__":
    main()
