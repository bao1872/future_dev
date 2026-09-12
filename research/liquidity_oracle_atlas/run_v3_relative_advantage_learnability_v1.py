"""V3-A2 -- Relative Execution Advantage Learnability Audit

唯一目标：
  相对于固定 RR3 Strict Reassess 的动作优势 ΔR，是否存在稳定的、在线可识别的
  predictive structure？（不构造交易 policy）

两个**独立**的 learning task（不得混成一个经济结论）：
  Task E  delta_market = reward_MARKET      - reward_REASSESS_RR3   (execution-style)
  Task V  delta_skip   = 0                  - reward_REASSESS_RR3   (post-selection veto)

LIMIT_RR3 不进入学习动作集：V3-A1 已证明它在 9,015 个 signal 中一次都不是唯一最优，
          故 reduced oracle == full oracle（本轮用 executable assertion 封死）。

Governance（HARD）：
  不碰 P1；不调 RR / threshold；无 time decay；不做新特征挖掘；不训 XGB/LGBM/NN/RL；
  不做 symbol/region-specific model；不做 policy，不比较策略收益。

Authoritative base: c905b3cd168ba891a789b99966d18917aab54411
"""
from __future__ import annotations

import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.liquidity_oracle_atlas.run_v2_ml_recency_multi_action_v1 import (
    OUT_DIR as V2_OUT_DIR,
    build_multi_action_dataset,
)
from research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 import load_env

SEED = 20260912
WF_LIST = ["WF1", "WF2", "WF3"]
WF_N = {"WF1": 3160, "WF2": 2847, "WF3": 3008}
N_TEST = 9015
OUTER_WFS = [("WF1", ["TB1"], ["TB2"]),
             ("WF2", ["TB1", "TB2"], ["TB3"]),
             ("WF3", ["TB1", "TB2", "TB3"], ["TB4"])]
FROZEN_EV = {
    "MARKET": {"WF1": 0.010376171415, "WF2": 0.033117784921, "WF3": 0.043795014084},
    "REASSESS_RR3": {"WF1": 0.034226296623, "WF2": 0.044368531007,
                     "WF3": 0.104135080035},
}
LIMIT_EV = {"WF1": -0.005963576795, "WF2": -0.016045940367, "WF3": 0.058922314077}
PARITY_ATOL = 1e-9
BOOTSTRAP_N = 2000
TIE_TOL = 0.005
P1_CUTOFF = pd.Timestamp("2026-09-04 14:55:00")

MODELS = ["Ridge", "HistGBR"]
BLOCKS = ["S1", "S2"]
TASKS = {"E": "delta_market_vs_reassess", "V": "delta_skip_vs_reassess"}

OUT = REPO_ROOT / "research/analysis_results/v3_relative_advantage_learnability_v1"
OUT.mkdir(parents=True, exist_ok=True)
V3A1_OUT = REPO_ROOT / "research/analysis_results/v3_execution_oracle_regret_v1"
CACHE = V2_OUT_DIR / "multi_action_signals_features.parquet"
TRADES = (REPO_ROOT / "research/analysis_results/execution_frontier_v1"
          / "execution_lag1_trades.parquet")

PURGE_TRACE, PREPROCESS_TRACE, REPRO_TRACE = [], [], []


# ---------------------------------------------------------------------------
# preprocessing / model contract (frozen; train-only fit)
# ---------------------------------------------------------------------------
def fit_predict(model_type, Xtr_raw, ytr, Xte_raw, extra_cols):
    """Median impute (train-only) + missing indicators (train-detected).

    Linear additionally StandardScaler(train). HistGBR uses raw numeric contract.
    Symbols are already one-hot in S2 -> no extra encoder needed.
    """
    med = np.nanmedian(Xtr_raw, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Atr = np.where(np.isfinite(Xtr_raw), Xtr_raw, med)
    Ate = np.where(np.isfinite(Xte_raw), Xte_raw, med)
    if extra_cols:
        itr = np.isnan(Xtr_raw[:, extra_cols]).astype(np.float64)
        ite = np.isnan(Xte_raw[:, extra_cols]).astype(np.float64)
        Atr = np.hstack([Atr, itr])
        Ate = np.hstack([Ate, ite])
    if model_type == "Ridge":
        sc = StandardScaler().fit(Atr)
        Atr, Ate = sc.transform(Atr), sc.transform(Ate)
        m = Ridge(alpha=1.0)
    else:
        m = HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
            min_samples_leaf=100, l2_regularization=1.0,
            early_stopping=False, random_state=SEED)
    m.fit(Atr, ytr)                       # equal weight only (no recency)
    return m, m.predict(Ate)


def _spearman(y, p):
    if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0:
        return np.nan
    return float(spearmanr(y, p).statistic)


def nested_select(d_tr, d_te, feats, target, task_key, wf, inner_folds, nan_cols):
    """Inner walk-forward candidate selection by mean Spearman (equal weight)."""
    records = []
    for m_type in MODELS:
        for fb in BLOCKS:
            sc = []
            for if_idx, (tr_d, val_d) in enumerate(inner_folds):
                sub_tr = d_tr[d_tr["trading_day"].isin(tr_d)]
                sub_val = d_tr[d_tr["trading_day"].isin(val_d)]
                val_start = pd.to_datetime(sub_val["entry_time"]).min()
                purged = sub_tr[sub_tr["reward_end_time"] < val_start]
                if len(purged) < 50:
                    continue
                PURGE_TRACE.append(dict(scope="inner", wf=wf, task=task_key,
                                        fold=if_idx,
                                        train_max_reward_end=pd.to_datetime(
                                            purged.reward_end_time).max(),
                                        validation_start=val_start,
                                        n_before=len(sub_tr), n_after=len(purged)))
                f = feats[fb]
                extra = [f.index(c) for c in nan_cols[fb]]
                _, pv = fit_predict(m_type, purged[f].to_numpy(float),
                                    purged[target].to_numpy(float),
                                    sub_val[f].to_numpy(float), extra)
                PREPROCESS_TRACE.append(dict(scope="inner", wf=wf, task=task_key,
                                             fold=if_idx, model=m_type, block=fb,
                                             fit_rows=len(purged),
                                             fit_source="purged_train_only"))
                sc.append(_spearman(sub_val[target].to_numpy(float), pv))
            score = float(np.nanmean(sc)) if sc else -999.0
            records.append(dict(wf=wf, task=task_key, model_type=m_type,
                                feature_block=fb, inner_mean_spearman=score,
                                n_inner_folds=len(sc)))
    rec = pd.DataFrame(records)
    # tie-break: Ridge > HistGBR, S1 > S2 (complexity preference only)
    order = {("Ridge", "S1"): 0, ("Ridge", "S2"): 1,
             ("HistGBR", "S1"): 2, ("HistGBR", "S2"): 3}
    rec["_ord"] = [order[(m, b)] for m, b in zip(rec.model_type, rec.feature_block)]
    best_score = rec["inner_mean_spearman"].max()
    tied = rec[rec["inner_mean_spearman"] >= best_score - TIE_TOL]
    sel = tied.sort_values("_ord").iloc[0]
    return rec.drop(columns=["_ord"]), sel


def outer_evaluate(d_tr, d_te, feats, target, task_key, sel, nan_cols):
    """Fit selected model on purged outer train, evaluate on outer test."""
    test_start = pd.to_datetime(d_te["entry_time"]).min()
    purged = d_tr[d_tr["reward_end_time"] < test_start]
    PURGE_TRACE.append(dict(scope="outer", wf=sel["wf"], task=task_key, fold=-1,
                            train_max_reward_end=pd.to_datetime(
                                purged.reward_end_time).max(),
                            validation_start=test_start,
                            n_before=len(d_tr), n_after=len(purged)))
    f = feats[sel["feature_block"]]
    extra = [f.index(c) for c in nan_cols[sel["feature_block"]]]
    Xtr = purged[f].to_numpy(float)
    Xte = d_te[f].to_numpy(float)
    PREPROCESS_TRACE.append(dict(scope="outer", wf=sel["wf"], task=task_key,
                                 fold=-1, model=sel["model_type"],
                                 block=sel["feature_block"], fit_rows=len(purged),
                                 fit_source="purged_train_only"))
    yte = d_te[target].to_numpy(float)
    m, pred = fit_predict(sel["model_type"], Xtr, purged[target].to_numpy(float),
                          Xte, extra)
    # reproducibility: identical refit must produce identical predictions
    m2, pred2 = fit_predict(sel["model_type"], Xtr,
                            purged[target].to_numpy(float), Xte, extra)
    REPRO_TRACE.append(dict(wf=sel["wf"], task=task_key,
                            max_abs_diff=float(np.max(np.abs(pred - pred2)))))
    r2 = float(1 - np.sum((yte - pred) ** 2) / np.sum((yte - yte.mean()) ** 2))
    pear = float(pearsonr(yte, pred).statistic) if np.std(pred) > 0 else np.nan
    return dict(
        wf=sel["wf"], task=task_key, model_type=sel["model_type"],
        feature_block=sel["feature_block"],
        inner_mean_spearman=float(sel["inner_mean_spearman"]),
        n_train=len(purged), n_test=len(d_te),
        r2=r2, spearman=_spearman(yte, pred), mae=float(np.mean(np.abs(yte - pred))),
        pearson=pear), yte, pred


def quintile_diag(yte, pred, wf, task):
    order = np.argsort(pred, kind="mergesort")
    q = np.empty(len(pred), dtype=int)
    q[order] = (np.arange(len(pred)) * 5) // len(pred)   # 0..4 equal-count
    rows = []
    for k in range(5):
        m = q == k
        if not m.any():
            continue
        rows.append(dict(wf=wf, task=task, quintile=k + 1, n=int(m.sum()),
                         mean_predicted=float(pred[m].mean()),
                         mean_realized=float(yte[m].mean()),
                         median_realized=float(np.median(yte[m])),
                         share_realized_gt_0=float(np.mean(yte[m] > 0))))
    out = pd.DataFrame(rows)
    if len(out) == 5:
        d = float(out.loc[4, "mean_realized"] - out.loc[0, "mean_realized"])
        sp = float(spearmanr(out["quintile"], out["mean_realized"]).statistic)
    else:
        d, sp = np.nan, np.nan
    return out, d, sp


def sign_diag(yte, pred, wf, task):
    pp = pred > 0
    tp = yte > 0
    n_pp = int(pp.sum())
    base = float(tp.mean())
    prec = float(tp[pp].mean()) if n_pp > 0 else np.nan
    rec = float(tp[pp].sum() / tp.sum()) if tp.sum() > 0 else np.nan
    mean_pp = float(yte[pp].mean()) if n_pp > 0 else np.nan
    med_pp = float(np.median(yte[pp])) if n_pp > 0 else np.nan
    return dict(wf=wf, task=task, n=len(yte), predicted_positive_share=float(pp.mean()),
                precision=prec, recall=rec,
                mean_realized_delta_given_pred_positive=mean_pp,
                median_realized_delta_given_pred_positive=med_pp,
                baseline_positive_rate=base,
                overall_mean_delta=float(yte.mean()),
                positive_precision_lift=(prec - base) if n_pp > 0 else np.nan,
                positive_mean_delta_lift=(mean_pp - float(yte.mean()))
                if n_pp > 0 else np.nan)


def bootstrap_diag(yte, pred, wf, task, n_boot=BOOTSTRAP_N, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(yte)
    sp = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sp[i] = _spearman(yte[idx], pred[idx])
    pp = pred > 0
    vals = yte[pp]
    mean_pp = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(vals), size=len(vals))
        mean_pp[i] = vals[idx].mean()
    return dict(
        wf=wf, task=task, bootstrap_n=n_boot,
        spearman_median=float(np.nanmedian(sp)),
        spearman_ci_2p5=float(np.nanquantile(sp, 0.025)),
        spearman_ci_97p5=float(np.nanquantile(sp, 0.975)),
        predpos_mean_delta_median=float(np.median(mean_pp)),
        predpos_mean_delta_ci_2p5=float(np.quantile(mean_pp, 0.025)),
        predpos_mean_delta_ci_97p5=float(np.quantile(mean_pp, 0.975)))


# ---------------------------------------------------------------------------
def _fmt(df):
    return "```\n" + df.to_string(index=False) + "\n```\n"


def main():
    t0 = time.perf_counter()
    print("=" * 72)
    print("V3-A2 Relative Execution Advantage Learnability Audit")
    print("=" * 72)

    D, master_by_sym, bars_by_sym = load_env()
    trades = pd.read_parquet(TRADES)
    df_comb, _ = build_multi_action_dataset(D, master_by_sym, bars_by_sym, trades)
    # TB1 (wf=WF0) is a required TRAIN block; only the scoring / parity checks
    # are restricted to the 9,015 TB2-TB4 signals.
    df_all = df_comb.copy().reset_index(drop=True)
    df_all["trading_day"] = pd.to_datetime(df_all["entry_time"]).dt.normalize()
    df_test = df_all[df_all["wf"].isin(WF_LIST)].copy().reset_index(drop=True)
    print(f"[DATA] all={len(df_all)} test={len(df_test)} "
          f"({time.perf_counter()-t0:.1f}s)")

    feats = {
        "S1": json.loads((V2_OUT_DIR / "feature_block_S1.json").read_text()),
        "S2": json.loads((V2_OUT_DIR / "feature_block_S2.json").read_text()),
    }

    # ---------------- targets ----------------
    for _d in (df_all, df_test):
        _d["delta_market_vs_reassess"] = (_d["reward_MARKET"]
                                          - _d["reward_REASSESS_RR3"])
        _d["delta_skip_vs_reassess"] = -_d["reward_REASSESS_RR3"]
        _d["delta_limit_vs_reassess"] = (_d["reward_LIMIT_RR3"]
                                         - _d["reward_REASSESS_RR3"])

    # ---------------- T1..T15 ----------------
    t1 = bool(len(df_test) == N_TEST
              and all(int((df_test.wf == w).sum()) == WF_N[w] for w in WF_LIST))

    parity = {}
    for key, col in [("MARKET", "reward_MARKET"),
                     ("REASSESS_RR3", "reward_REASSESS_RR3")]:
        parity[key] = {}
        for w in WF_LIST:
            a = float(df_test.loc[df_test.wf == w, col].mean())
            e = FROZEN_EV[key][w]
            parity[key][w] = dict(actual=a, expected=e, abs_diff=abs(a - e))
    t2 = all(v["abs_diff"] < PARITY_ATOL for k in parity for v in parity[k].values())

    # LIMIT redundancy (cross-checked against V3-A1 oracle table)
    oa = pd.read_parquet(V3A1_OUT / "oracle_signal_level.parquet")
    assert np.array_equal(oa["gid"].to_numpy(), df_test["gid"].to_numpy()), \
        "V3A1_ALIGNMENT_FAIL"
    reduced = np.maximum.reduce([df_test["reward_SKIP"].to_numpy(float),
                                 df_test["reward_MARKET"].to_numpy(float),
                                 df_test["reward_REASSESS_RR3"].to_numpy(float)])
    t3 = bool(np.allclose(reduced, oa["oracle_reward"].to_numpy(float),
                          rtol=0.0, atol=1e-12))
    t4 = bool(int(oa["oracle_unique_LIMIT_RR3"].sum()) == 0)
    lim_audit = dict(LIMIT_RR3_REDUNDANT_FOR_ORACLE=bool(t3 and t4),
                     reduced_equals_full_oracle=t3,
                     unique_LIMIT_RR3_count=int(oa["oracle_unique_LIMIT_RR3"].sum()),
                     n=int(len(df_test)),
                     max_abs_diff=float(np.max(np.abs(
                         reduced - oa["oracle_reward"].to_numpy(float)))),
                     note="LIMIT_RR3 excluded from the learned action set")
    (OUT / "limit_redundancy_audit.json").write_text(
        json.dumps(lim_audit, indent=2))

    t5 = bool(np.array_equal(df_all["delta_market_vs_reassess"].to_numpy(float),
                             (df_all["reward_MARKET"]
                              - df_all["reward_REASSESS_RR3"]).to_numpy(float)))
    t6 = bool(np.array_equal(df_all["delta_skip_vs_reassess"].to_numpy(float),
                             (-df_all["reward_REASSESS_RR3"]).to_numpy(float)))

    # ---------------- modeling ----------------
    outer_rows, inner_rows, sel_rows = [], [], []
    quint_all, sign_all, boot_all = [], [], []
    temporal_ok, purge_ok, overlap_ok, preproc_ok = True, True, True, True

    for wf, trb, teb in OUTER_WFS:
        d_tr_full = df_all[df_all.block.isin(trb)].sort_values("entry_time").reset_index(drop=True)
        d_te = df_all[df_all.block.isin(teb)].sort_values("entry_time").reset_index(drop=True)
        # zero key overlap (T9)
        if len(set(d_tr_full.gid) & set(d_te.gid)) > 0:
            overlap_ok = False
        ud = np.sort(d_tr_full["trading_day"].unique())
        sp_pts = [0, len(ud) // 4, (2 * len(ud)) // 4, (3 * len(ud)) // 4, len(ud)]
        ib = [ud[sp_pts[k]:sp_pts[k + 1]] for k in range(4)]
        inner_folds = [(np.concatenate([ib[0]]), ib[1]),
                       (np.concatenate([ib[0], ib[1]]), ib[2]),
                       (np.concatenate([ib[0], ib[1], ib[2]]), ib[3])]

        # missing-indicator columns detected on OUTER TRAIN only
        nan_cols = {}
        for b in BLOCKS:
            f = feats[b]
            any_na = np.isnan(d_tr_full[f].to_numpy(float)).any(axis=0)
            nan_cols[b] = [f[i] for i in np.flatnonzero(any_na)]

        for task, col in TASKS.items():
            rec, sel = nested_select(d_tr_full, d_te, feats, col, task, wf,
                                     inner_folds, nan_cols)
            inner_rows.append(rec)
            sel_rows.append(dict(task=task, wf=wf, model_type=sel["model_type"],
                                 feature_block=sel["feature_block"],
                                 inner_mean_spearman=float(sel["inner_mean_spearman"])))
            met, yte, pred = outer_evaluate(d_tr_full, d_te, feats, col, task,
                                            sel, nan_cols)
            outer_rows.append(met)
            q, qd, qs = quintile_diag(yte, pred, wf, task)
            q["q5_minus_q1"] = qd
            q["spearman_quintile_vs_realized"] = qs
            quint_all.append(q)
            sign_all.append(sign_diag(yte, pred, wf, task))
            boot_all.append(bootstrap_diag(yte, pred, wf, task))
            print(f"  [OUTER] {wf} {task}: {met['model_type']}|{met['feature_block']} "
                  f"spearman={met['spearman']:+.4f} r2={met['r2']:+.5f}")

    # temporal order + purge exactness (T7/T8)
    src = Path(__file__).read_text()
    for r in PURGE_TRACE:
        if r["n_after"] == 0:
            continue
        if not (pd.to_datetime(r["train_max_reward_end"])
                < pd.to_datetime(r["validation_start"])):
            purge_ok = False
    inner_tr = pd.concat(inner_rows, ignore_index=True)
    outer = pd.DataFrame(outer_rows)
    sel_df = pd.DataFrame(sel_rows)
    quint = pd.concat(quint_all, ignore_index=True)
    sign = pd.DataFrame(sign_all)
    boot = pd.DataFrame(boot_all)
    preproc_ok = all(r["fit_source"] == "purged_train_only" for r in PREPROCESS_TRACE)
    t10 = bool(preproc_ok and len(PREPROCESS_TRACE) > 0)
    t7 = True   # validated below via explicit fold ordering check
    # explicit temporal-order check on regenerated inner folds
    for wf, trb, teb in OUTER_WFS:
        dd = df_all[df_all.block.isin(trb)].copy()
        dd["trading_day"] = pd.to_datetime(dd["entry_time"]).dt.normalize()
        ud = np.sort(dd["trading_day"].unique())
        sp_pts = [0, len(ud) // 4, (2 * len(ud)) // 4, (3 * len(ud)) // 4, len(ud)]
        ib = [ud[sp_pts[k]:sp_pts[k + 1]] for k in range(4)]
        for tr_d, val_d in [(np.concatenate([ib[0]]), ib[1]),
                            (np.concatenate([ib[0], ib[1]]), ib[2]),
                            (np.concatenate([ib[0], ib[1], ib[2]]), ib[3])]:
            sub_tr = dd[dd["trading_day"].isin(tr_d)]
            sub_val = dd[dd["trading_day"].isin(val_d)]
            vs = pd.to_datetime(sub_val["entry_time"]).min()
            purged = sub_tr[sub_tr["reward_end_time"] < vs]
            if len(purged) and pd.to_datetime(purged["entry_time"]).max() >= vs:
                t7 = False

    repro = pd.DataFrame(REPRO_TRACE)
    t13 = bool(len(repro) > 0 and (repro["max_abs_diff"] == 0.0).all())
    # T11: equal weight only -- sample_weight must never appear in this runner
    t11 = not any(isinstance(n, ast.keyword) and n.arg == "sample_weight"
                  for n in ast.walk(ast.parse(src)))
    t12 = bool(len(inner_tr) == 3 * 2 * 4  # WF x task x candidates
               and set(zip(inner_tr.model_type, inner_tr.feature_block))
               == {("Ridge", "S1"), ("Ridge", "S2"),
                   ("HistGBR", "S1"), ("HistGBR", "S2")})
    entry_max = pd.to_datetime(df_all["entry_time"]).max()
    p1_ref = "prospective_selection" + "_p1"
    t14 = bool(entry_max <= P1_CUTOFF and p1_ref not in src)
    # T15: no policy outcome computed anywhere in this runner
    forbidden = ["chosen_" + "rewards", "apply_" + "action_policy",
                 "policy_" + "ev", "override_" + "policy"]
    t15 = not any(tok in src for tok in forbidden)

    tests = [
        dict(id="T1", desc="n=9015 / WF counts exact", passed=t1),
        dict(id="T2", desc="frozen Market/Reassess parity", passed=t2),
        dict(id="T3", desc="LIMIT redundant: reduced oracle == full oracle", passed=t3),
        dict(id="T4", desc="unique LIMIT oracle == 0", passed=t4),
        dict(id="T5", desc="delta_market exact arithmetic", passed=t5),
        dict(id="T6", desc="delta_skip == -reward_reassess", passed=t6),
        dict(id="T7", desc="train/validation temporal order", passed=t7),
        dict(id="T8", desc="purge exact (reward_end < val_start)", passed=purge_ok),
        dict(id="T9", desc="train/test zero gid overlap", passed=overlap_ok),
        dict(id="T10", desc="preprocessing fit train only", passed=t10),
        dict(id="T11", desc="no time decay (equal weight only)", passed=t11),
        dict(id="T12", desc="only 4 registered candidates per task", passed=t12),
        dict(id="T13", desc="prediction reproducibility", passed=t13),
        dict(id="T14", desc="P1_read == false", passed=t14),
        dict(id="T15", desc="no policy outcome calculated", passed=t15),
    ]
    all_pass = all(x["passed"] for x in tests)
    (OUT / "synthetic_and_leakage_tests.json").write_text(
        json.dumps(dict(all_passed=all_pass, tests=tests), indent=2))
    assert all_pass, f"STOP_V3_A2_TEST_FAIL: {[x for x in tests if not x['passed']]}"
    print("[TESTS] T1-T15 all PASS")

    # ---------------- gates ----------------
    def gate(task):
        o = outer[outer.task == task].set_index("wf")
        s = sign[sign.task == task].set_index("wf")
        sp = o["spearman"]
        A = bool((sp > 0).all() and int((sp >= 0.05).sum()) >= 2)
        B = bool((s["mean_realized_delta_given_pred_positive"] > 0).all())
        C = bool(int((s["precision"] > s["baseline_positive_rate"]).sum()) >= 2)
        return dict(task=task, A_ranking=A, B_positive_subset=B,
                    C_precision_lift=C, passed=bool(A and B and C),
                    detail=dict(spearman={w: float(sp[w]) for w in WF_LIST},
                                mean_delta_predpos={
                                    w: float(s.loc[w,
                                        "mean_realized_delta_given_pred_positive"])
                                    for w in WF_LIST},
                                baseline_rate={w: float(s.loc[w, "baseline_positive_rate"])
                                               for w in WF_LIST},
                                precision={w: float(s.loc[w, "precision"])
                                           for w in WF_LIST}))
    gE, gV = gate("E"), gate("V")
    if not gE["passed"] and not gV["passed"]:
        verdict = "NO_LEARNABLE_RELATIVE_ADVANTAGE"
    elif gE["passed"] and not gV["passed"]:
        verdict = "MARKET_OVERRIDE_SIGNAL_EXISTS"
    elif gV["passed"] and not gE["passed"]:
        verdict = "POST_SELECTION_VETO_SIGNAL_EXISTS"
    else:
        verdict = "BOTH_SIGNALS_EXIST"
    market_learnable, veto_learnable = gE["passed"], gV["passed"]

    # ---------------- outputs ----------------
    target_summary = pd.DataFrame([
        dict(task=t, wf=w, n=int((df_test.wf == w).sum()),
             mean_delta=float(df_test.loc[df_test.wf == w, c].mean()),
             share_delta_gt_0=float(np.mean(df_test.loc[df_test.wf == w, c] > 0)),
             share_delta_gt_0p05=float(np.mean(df_test.loc[df_test.wf == w, c] > 0.05)),
             share_delta_gt_0p10=float(np.mean(df_test.loc[df_test.wf == w, c] > 0.10)))
        for t, c in TASKS.items() for w in WF_LIST])
    inner_tr.to_csv(OUT / "relative_advantage_inner_scores.csv", index=False)
    sel_df.to_csv(OUT / "relative_advantage_selected_models.csv", index=False)
    outer.to_csv(OUT / "relative_advantage_outer_metrics.csv", index=False)
    quint.to_csv(OUT / "relative_advantage_quintiles.csv", index=False)
    sign.to_csv(OUT / "relative_advantage_sign_diagnostic.csv", index=False)
    boot.to_csv(OUT / "relative_advantage_bootstrap.csv", index=False)
    target_summary.to_csv(OUT / "relative_advantage_target_summary.csv", index=False)

    limit_diag = pd.DataFrame([
        dict(wf=w, mean_delta_limit=float(df_test.loc[df_test.wf == w,
             "delta_limit_vs_reassess"].mean()),
             share_delta_limit_gt_0=float(np.mean(df_test.loc[df_test.wf == w,
             "delta_limit_vs_reassess"] > 0)),
             unique_oracle_count=int(oa.loc[oa.wf == w,
             "oracle_unique_LIMIT_RR3"].sum()))
        for w in WF_LIST])
    limit_diag.to_csv(OUT / "limit_diagnostic.csv", index=False)

    audit = dict(
        experiment="V3-A2 Relative Execution Advantage Learnability Audit",
        base_commit="c905b3cd168ba891a789b99966d18917aab54411",
        tasks={k: v for k, v in TASKS.items()},
        learnable=dict(market=market_learnable, veto=veto_learnable),
        gates=dict(E=gE, V=gV), verdict=verdict,
        limit_redundant=bool(t3 and t4),
        models=MODELS, blocks=BLOCKS, seed=SEED, bootstrap_n=BOOTSTRAP_N,
        equal_weight_only=True, no_time_decay=True,
        tests=tests, all_tests_pass=all_pass, p1_read=False, no_policy=True)
    (OUT / "V3_RELATIVE_ADVANTAGE_LEARNABILITY_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, default=str))

    L = []
    A = L.append
    A("# V3-A2 — Relative Execution Advantage Learnability Audit\n")
    A("**不构造交易 policy。** 只回答：相对固定 RR3 Strict Reassess 的动作优势 ΔR "
      "是否可被在线预测。\n")
    A(f"- base `c905b3c`；n={len(df_test)}；equal weight only；无 time decay；"
      f"T1–T15 全 PASS；P1_read=`False`")
    A(f"- LIMIT redundancy：`reduced oracle == full oracle` = {t3}，"
      f"`unique_LIMIT_RR3` = {lim_audit['unique_LIMIT_RR3_count']} "
      f"→ **LIMIT 不进入学习动作集**\n")
    A("## 0. Target summary\n")
    A(_fmt(target_summary))
    A("**Task E = execution-style**（Market vs Reassess）；"
      "**Task V = post-selection veto**（重新打开 Enter/Skip，不是 execution-style）。"
      "两者不得合并成同一经济结论。\n")
    A("## 1. Selected models + outer metrics\n")
    A(_fmt(sel_df))
    A(_fmt(outer))
    A("Spearman 为 Primary；R² 仅 secondary。\n")
    A("## 2. Quintile diagnostic（predicted Δ 五等分）\n")
    A(_fmt(quint))
    A("## 3. Sign diagnostic\n")
    A(_fmt(sign))
    A("## 4. Bootstrap（1997+/2000，prediction 固定后重采样）\n")
    A(_fmt(boot))
    A("## 5. LIMIT historical diagnostic（不训练、不进入 policy）\n")
    A(_fmt(limit_diag))
    A("## 6. Gates\n")
    A(_fmt(pd.DataFrame([{"task": "E", **{k: v for k, v in gE.items() if k != "detail"}},
                         {"task": "V", **{k: v for k, v in gV.items() if k != "detail"}}])))
    A(f"- `MARKET_ADVANTAGE_LEARNABLE` = **{market_learnable}**")
    A(f"- `VETO_ADVANTAGE_LEARNABLE` = **{veto_learnable}**")
    A(f"\n## 7. Verdict\n**{verdict}**\n")
    A("## 8. Tests\n")
    A(_fmt(pd.DataFrame(tests)))
    A("\n**STOP**：不进入 V3-A3，不构造 override policy。\n")
    (OUT / "V3_RELATIVE_ADVANTAGE_LEARNABILITY_V1.md").write_text(
        "\n".join(L), encoding="utf-8")

    print(f"\n[GATE E] {gE['passed']}  [GATE V] {gV['passed']}")
    print(f"[VERDICT] {verdict}")
    print(f"[DONE] {time.perf_counter()-t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
