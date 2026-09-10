"""Step 10-18：label-availability purge、Baseline、Logistic、LightGBM、
fold/symbol/source_tf/direction 诊断、DEDUP、bootstrap、zero-shot、ALL refit。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

import research.m2_nondeep_temporal_v1 as m2
from research.phase1_native_direction.nd_contract_v1 import (
    DEV4, RESULTS, SYMBOLS,
)

BOOT = 500
SEED = 7


def fit_logistic(Xtr, ytr):
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=SEED)).fit(Xtr, ytr)


def fit_lgbm(Xtr, ytr):
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        random_state=SEED, verbose=-1, n_jobs=1).fit(Xtr, ytr)


def metrics(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    out = dict(n=len(y), base_rate=round(float(y.mean()), 4))
    if len(np.unique(y)) > 1:
        out["AUC"] = round(float(roc_auc_score(y, p)), 4)
        out["PR_AUC"] = round(float(average_precision_score(y, p)), 4)
        out["Brier"] = round(float(brier_score_loss(y, p)), 6)
    else:
        out["AUC"] = out["PR_AUC"] = out["Brier"] = None
    if len(y) >= 50 and y.mean() > 0:
        d = pd.qcut(pd.Series(p).rank(method="first"), 10,
                    labels=False) + 1
        out["D1"] = round(float(y[d == 1].mean()), 4)
        out["D10"] = round(float(y[d == 10].mean()), 4)
        out["D10_minus_D1"] = round(out["D10"] - out["D1"], 4)
        out["decile_spearman"] = round(float(spearmanr(p, y).statistic), 4)
        for q, k in zip(np.quantile(p, [0.9, 0.8, 0.7]), (10, 20, 30)):
            mk = p >= q
            r = float(y[mk].mean()) if mk.any() else np.nan
            out[f"Top{k}_rate"] = round(r, 4)
            out[f"Lift@{k}"] = round(r / y.mean(), 4)
            out[f"Top{k}_uplift"] = round(r - y.mean(), 4)
    return out


def load():
    lab = pd.read_parquet(RESULTS / "native_labels_v1.parquet")
    X = pd.read_parquet(RESULTS / "native_features_v1.parquet")
    lab["candidate_id"] = lab["candidate_id"].astype(str)
    X["candidate_id"] = X["candidate_id"].astype(str)
    d = lab.merge(X, on="candidate_id", how="inner", validate="one_to_one")
    d = d[d["native_status"] == "RESOLVED"].copy()
    d["y"] = d["native_label"].astype(int)
    d["symbol_code"] = d["symbol"].astype("category").cat.codes
    d["source_tf_code"] = d["source_tf"].astype("category").cat.codes
    feats = [c for c in X.columns if c != "candidate_id"]
    assert not any(c in feats for c in
                   ["native_label", "flipped_label", "resolution_time",
                    "bars_to_resolution", "resolution_type"])
    return d.reset_index(drop=True), feats


def run_folds(d, feats):
    folds, _ = m2.build_folds(d["trading_day"].to_numpy(), len(d))
    avail = []
    recs = []
    for fi in range(4):
        tr_d, se_d, te_d = folds[fi]
        tr = d[d["trading_day"].isin(tr_d)]
        se = d[d["trading_day"].isin(se_d)]
        te = d[d["trading_day"].isin(te_d)]
        if len(te) < 100:
            continue
        se_start = se["decision_time"].min()
        te_start = te["decision_time"].min()

        tr_ok = tr[tr["resolution_time"] < se_start]
        se_ok = se[se["resolution_time"] < te_start]

        avail.append(dict(
            折=f"F{fi+1}",
            train_events=len(tr), train_usable=len(tr_ok),
            train_purged=len(tr) - len(tr_ok),
            selection_events=len(se), selection_usable=len(se_ok),
            selection_purged=len(se) - len(se_ok),
            test_events=len(te),
        ))
        if len(tr_ok) < 200:
            continue

        ytr = tr_ok["y"].to_numpy(int)
        Xtr = tr_ok[feats].to_numpy(float)

        m_log = fit_logistic(Xtr, ytr)
        m_lgb = fit_lgbm(Xtr, ytr)
        m_b1 = fit_logistic(
            tr_ok[["native_direction", "source_tf_code",
                   "symbol_code"]].to_numpy(float), ytr)
        m_bzs = fit_logistic(
            tr_ok[["native_direction", "source_tf_code"]].to_numpy(float),
            ytr)

        # ---- 真正的 cross-symbol zero-shot：只用 DEV4 TRAIN 拟合 ----
        dtr = tr_ok[tr_ok["symbol"].isin(DEV4)]
        if len(dtr) >= 200:
            m_log_dev = fit_logistic(dtr[feats].to_numpy(float),
                                     dtr["y"].to_numpy(int))
            m_lgb_dev = fit_lgbm(dtr[feats].to_numpy(float),
                                 dtr["y"].to_numpy(int))
            m_b1_dev = fit_logistic(
                dtr[["native_direction", "source_tf_code"]].to_numpy(float),
                dtr["y"].to_numpy(int))
        else:
            m_log_dev = m_lgb_dev = m_b1_dev = None

        Xte = te[feats].to_numpy(float)
        t = te.copy()
        t["pred_log"] = m_log.predict_proba(Xte)[:, 1]
        t["pred_lgb"] = m_lgb.predict_proba(Xte)[:, 1]
        if m_log_dev is not None:
            t["pred_log_dev"] = m_log_dev.predict_proba(Xte)[:, 1]
            t["pred_lgb_dev"] = m_lgb_dev.predict_proba(Xte)[:, 1]
            t["pred_baseZS_dev"] = m_b1_dev.predict_proba(
                te[["native_direction", "source_tf_code"]].to_numpy(float)
            )[:, 1]
        else:
            t["pred_log_dev"] = np.nan
            t["pred_lgb_dev"] = np.nan
            t["pred_baseZS_dev"] = np.nan
        t["pred_base1"] = m_b1.predict_proba(
            te[["native_direction", "source_tf_code",
                "symbol_code"]].to_numpy(float))[:, 1]
        t["pred_baseZS"] = m_bzs.predict_proba(
            te[["native_direction", "source_tf_code"]].to_numpy(float))[:, 1]
        t["折"] = f"F{fi+1}"
        recs.append(t)
    pd.DataFrame(avail).to_csv(RESULTS / "fold_availability.csv",
                               index=False, encoding="utf-8-sig")
    return pd.concat(recs, ignore_index=True)


def block_boot(P, col, n=BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    days = P["trading_day"].dropna().unique()
    idx = {d: np.flatnonzero((P["trading_day"] == d).to_numpy())
           for d in days}
    y = P["y"].to_numpy(int)
    p = P[col].to_numpy(float)
    up, dd = [], []
    for _ in range(n):
        pick = rng.choice(days, size=len(days), replace=True)
        r = np.concatenate([idx[d] for d in pick])
        yy, pp = y[r], p[r]
        if len(np.unique(yy)) < 2 or len(yy) < 200:
            continue
        m = pp >= np.quantile(pp, 0.8)
        up.append(yy[m].mean() - yy.mean())
        dcl = pd.qcut(pd.Series(pp).rank(method="first"), 10,
                      labels=False) + 1
        dd.append(yy[dcl == 10].mean() - yy[dcl == 1].mean())
    return dict(metric_col=col, n_rep=len(up),
                Top20_uplift_ci=[round(float(np.quantile(up, .025)), 4),
                                 round(float(np.quantile(up, .975)), 4)],
                D10_minus_D1_ci=[round(float(np.quantile(dd, .025)), 4),
                                 round(float(np.quantile(dd, .975)), 4)])


def main():
    d, feats = load()
    print(f"[load] resolved={len(d)} features={len(feats)}", flush=True)
    P = run_folds(d, feats)
    P.to_parquet(RESULTS / "native_predictions.parquet", index=False)
    print("[folds] done", flush=True)

    new_sym = P[~P["symbol"].isin(DEV4)]
    dev_sym = P[P["symbol"].isin(DEV4)]

    rows = []
    # universe -> (数据子集, 模型列映射)
    # DEV4_zero_shot / NEW11_zero_shot 一律使用 **DEV4-only 训练** 的模型
    for uni, g, cols in (
        ("ALL15_refit", P,
         (("Baseline0_constant", None),
          ("Baseline1_meta", "pred_base1"),
          ("BaselineZS_nd_tf", "pred_baseZS"),
          ("Logistic", "pred_log"),
          ("LightGBM", "pred_lgb"))),
        ("DEV4_zero_shot", dev_sym,
         (("Baseline0_constant", None),
          ("BaselineZS_nd_tf", "pred_baseZS_dev"),
          ("Logistic", "pred_log_dev"),
          ("LightGBM", "pred_lgb_dev"))),
        ("NEW11_zero_shot", new_sym,
         (("Baseline0_constant", None),
          ("BaselineZS_nd_tf", "pred_baseZS_dev"),
          ("Logistic", "pred_log_dev"),
          ("LightGBM", "pred_lgb_dev"))),
    ):
        for m, col in cols:
            p = (np.full(len(g), g["y"].mean()) if col is None
                 else g[col].to_numpy(float))
            r = dict(universe=uni, 模型=m)
            r.update(metrics(g["y"], p))
            rows.append(r)
    main_df = pd.DataFrame(rows)
    main_df.to_csv(RESULTS / "model_main.csv", index=False,
                   encoding="utf-8-sig")
    print("\n=== model_main ===")
    print(main_df[["universe", "模型", "n", "base_rate", "AUC", "PR_AUC",
                   "Lift@20", "Top20_uplift", "D10_minus_D1"]]
          .to_string(index=False), flush=True)

    def by(dim, col, out):
        rs = []
        for k, g in P.groupby(col):
            for m, cc in (("Logistic", "pred_log"),
                          ("LightGBM", "pred_lgb")):
                r = {dim: k, "模型": m}
                r.update(metrics(g["y"], g[cc]))
                rs.append(r)
        t = pd.DataFrame(rs)
        t.to_csv(RESULTS / out, index=False, encoding="utf-8-sig")
        return t

    by("折", "折", "model_by_fold.csv")
    by("symbol", "symbol", "model_by_symbol.csv")
    by("source_tf", "source_tf", "model_by_source_tf.csv")
    by("native_direction", "native_direction", "model_by_direction.csv")

    # deciles
    dec = []
    for m, col in (("Logistic", "pred_log"), ("LightGBM", "pred_lgb")):
        p = P[col].to_numpy(float)
        dcl = pd.qcut(pd.Series(p).rank(method="first"), 10,
                      labels=False) + 1
        for k in range(1, 11):
            mk = dcl == k
            dec.append(dict(模型=m, decile=k, n=int(mk.sum()),
                            pred_mean=round(float(p[mk].mean()), 4),
                            actual_success_rate=round(
                                float(P["y"].to_numpy()[mk].mean()), 4)))
    pd.DataFrame(dec).to_csv(RESULTS / "probability_deciles.csv",
                             index=False, encoding="utf-8-sig")

    # coverage lift
    cov = []
    for m, col in (("Logistic", "pred_log"), ("LightGBM", "pred_lgb")):
        p = P[col].to_numpy(float)
        y = P["y"].to_numpy(int)
        for k in (5, 10, 15, 20, 25, 30, 40, 50):
            q = np.quantile(p, 1 - k / 100)
            mk = p >= q
            r = float(y[mk].mean())
            cov.append(dict(模型=m, coverage_pct=k, n=int(mk.sum()),
                            success_rate=round(r, 4),
                            Lift=round(r / y.mean(), 4),
                            uplift=round(r - y.mean(), 4)))
    pd.DataFrame(cov).to_csv(RESULTS / "coverage_lift.csv", index=False,
                             encoding="utf-8-sig")

    # DEDUP
    ded = (P.sort_values(["candidate_group_id", "decision_time"])
            .groupby("candidate_group_id", as_index=False).head(1))
    drow = []
    for tag, g in (("ALL_EVENTS", P), ("DEDUP_EVENTS", ded)):
        for m, col in (("Logistic", "pred_log"), ("LightGBM", "pred_lgb")):
            r = dict(dataset=tag, 模型=m)
            r.update(metrics(g["y"], g[col]))
            drow.append(r)
    pd.DataFrame(drow).to_csv(RESULTS / "dedup_sensitivity.csv",
                              index=False, encoding="utf-8-sig")

    # bootstrap
    b = [block_boot(P, "pred_log"), block_boot(P, "pred_lgb")]
    pd.DataFrame(b).to_csv(RESULTS / "bootstrap.csv", index=False,
                           encoding="utf-8-sig")

    # macro
    ms = pd.read_csv(RESULTS / "model_by_symbol.csv")
    lg = ms[ms["模型"] == "Logistic"]
    audit = dict(
        resolved_events=int(len(d)),
        features=int(len(feats)),
        test_events=int(len(P)),
        base_rate=round(float(P["y"].mean()), 4),
        macro_mean_AUC=round(float(lg["AUC"].mean()), 4),
        macro_median_AUC=round(float(lg["AUC"].median()), 4),
        macro_mean_Lift20=round(float(lg["Lift@20"].mean()), 4),
        macro_median_Lift20=round(float(lg["Lift@20"].median()), 4),
        positive_symbols=int((lg["Lift@20"] > 1).sum()),
        n_symbols=int(len(lg)),
        folds_Lift20_gt_1_05=int((pd.read_csv(
            RESULTS / "model_by_fold.csv").query("模型=='Logistic'")
            ["Lift@20"] > 1.05).sum()),
    )
    (RESULTS / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== by direction ===")
    print(pd.read_csv(RESULTS / "model_by_direction.csv")[
        ["native_direction", "模型", "n", "base_rate", "AUC",
         "Lift@20", "Top20_uplift"]].to_string(index=False))
    print("\n=== by fold (Logistic) ===")
    print(pd.read_csv(RESULTS / "model_by_fold.csv").query(
        "模型=='Logistic'")[["折", "n", "AUC", "Lift@20", "Top20_uplift"]]
        .to_string(index=False))
    print("\n=== deciles (Logistic) ===")
    print(pd.read_csv(RESULTS / "probability_deciles.csv").query(
        "模型=='Logistic'").to_string(index=False))
    print("\n=== DEDUP ===")
    print(pd.read_csv(RESULTS / "dedup_sensitivity.csv")[
        ["dataset", "模型", "n", "base_rate", "AUC", "Lift@20",
         "Top20_uplift"]].to_string(index=False))
    print("\n=== bootstrap ===")
    print(pd.read_csv(RESULTS / "bootstrap.csv").to_string(index=False))
    print("\naudit:", json.dumps(audit, ensure_ascii=False))
    print("NATIVE_PHASE1_DONE")


if __name__ == "__main__":
    main()
