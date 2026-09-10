"""Phase 1 补充：合并口径的 品种 / source_tf 拆分 + DEDUP 敏感性。

主脚本已产出分折结果；本脚本补齐：
  1. 跨折【合并】计算的 4 品种 / 所有 source_tf 指标（不再逐折再平均）
  2. DEDUP 敏感性：每个 candidate_group_id 只保留最早 trigger 后
     重新计算 base rate / AUC / Lift@20（不重新调参）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)

import research.m2_nondeep_temporal_v1 as m2
from research.phase1_tradability.build_features_v1 import build
from research.phase1_tradability.phase1_contract_v1 import RESULTS
from research.phase1_tradability.run_phase1_v1 import (fit_lgbm,
                                                       fit_logistic, prep)


def met(y, p):
    base = float(y.mean())
    o = dict(n=len(y), base_rate=round(base, 4),
             AUC=round(float(roc_auc_score(y, p)), 4) if len(np.unique(y)) > 1
             else None,
             PR_AUC=round(float(average_precision_score(y, p)), 4)
             if len(np.unique(y)) > 1 else None,
             Brier=round(float(brier_score_loss(y, p)), 6))
    order = np.argsort(-p)
    ys = y[order]
    for k in (0.10, 0.20, 0.30):
        m = max(1, int(len(ys) * k))
        o[f"Lift@{int(k*100)}"] = round(float(ys[:m].mean() / base), 4) \
            if base > 0 else None
    return o


def run_set(res, feats, cat_cols, tag):
    """返回逐折合并后的逐事件预测表。"""
    ev_day = res["trading_day"].to_numpy()
    folds, _ = m2.build_folds(ev_day, len(res))
    specs = {"Baseline1_symbol_source_tf": (cat_cols, "logistic"),
             "Model1_Logistic": (feats, "logistic"),
             "Model2_LightGBM": (feats, "lgbm")}
    out = []
    for fi in range(4):
        tr_d, se_d, te_d = folds[fi]
        tr = res[res["trading_day"].isin(tr_d)]
        se = res[res["trading_day"].isin(se_d)]
        te = res[res["trading_day"].isin(te_d)]
        tr_ok = tr[tr["resolution_time"] < se["decision_time"].min()]
        if len(tr_ok) < 200 or len(te) < 100:
            continue
        ytr = tr_ok["tradable"].to_numpy(int)
        for name, (cols, kind) in specs.items():
            mdl = (fit_logistic(prep(tr_ok, cols), ytr) if kind == "logistic"
                   else fit_lgbm(prep(tr_ok, cols), ytr))
            p = mdl.predict_proba(prep(te, cols))[:, 1]
            out.append(pd.DataFrame(dict(
                折=f"F{fi+1}", 模型=name, pred=p,
                actual=te["tradable"].to_numpy(int),
                symbol=te["symbol"].to_numpy(),
                source_tf=te["source_tf"].to_numpy())))
    return pd.concat(out, ignore_index=True)


def main():
    lab = pd.read_parquet(RESULTS / "labels_v1.parquet")
    cand = pd.read_parquet(RESULTS / "candidates_v1.parquet")
    X, contract, feats = build()
    df = (lab.merge(cand[["candidate_id", "trading_day"]],
                    on="candidate_id", how="left", validate="one_to_one")
             .merge(X, on="candidate_id", how="inner", validate="one_to_one")
             .rename(columns={"label": "tradable"}))
    res = df[df["status"] == "RESOLVED"].copy()
    res["symbol_code"] = res["symbol"].astype("category").cat.codes
    res["source_tf_code"] = res["source_tf"].astype("category").cat.codes
    cat_cols = ["symbol_code", "source_tf_code"]

    dedup = (res.sort_values(["candidate_group_id", "decision_time"])
                .groupby("candidate_group_id", as_index=False).head(1))

    allp = run_set(res, feats, cat_cols, "ALL")
    dedp = run_set(dedup, feats, cat_cols, "DEDUP")

    rows = []
    for tag, P in (("ALL_EVENTS", allp), ("DEDUP_EVENTS", dedp)):
        for name, g in P.groupby("模型"):
            m = met(g["actual"].to_numpy(), g["pred"].to_numpy())
            rows.append(dict(集合=tag, 模型=name, **m))
    pd.DataFrame(rows).to_csv(RESULTS / "dedup_sensitivity.csv", index=False,
                              encoding="utf-8-sig")

    sym_rows, tf_rows = [], []
    for name, g in allp.groupby("模型"):
        for s, gg in g.groupby("symbol"):
            sym_rows.append(dict(品种=s, 模型=name,
                                 **met(gg["actual"].to_numpy(),
                                       gg["pred"].to_numpy())))
        for s, gg in g.groupby("source_tf"):
            tf_rows.append(dict(source_tf=s, 模型=name,
                                **met(gg["actual"].to_numpy(),
                                      gg["pred"].to_numpy())))
    pd.DataFrame(sym_rows).to_csv(RESULTS / "model_by_symbol.csv", index=False,
                                  encoding="utf-8-sig")
    pd.DataFrame(tf_rows).to_csv(RESULTS / "model_by_source_tf.csv",
                                 index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    print("=== DEDUP 敏感性（合并口径）===")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n=== 4 品种（合并口径）===")
    print(pd.DataFrame(sym_rows).to_string(index=False))
    print("\n=== source_tf（合并口径）===")
    print(pd.DataFrame(tf_rows).to_string(index=False))
    print("\nSPLITS_DONE")


if __name__ == "__main__":
    main()
