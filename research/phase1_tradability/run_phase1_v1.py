"""Phase 1 主实验：OB 事件 Tradability 可学习性。

模型（不做 model zoo）：
    Baseline0  全体 base rate
    Baseline1  symbol + source_tf → Logistic
    Model1     完整事件级 → Logistic
    Model2     完整事件级 → LightGBM

主指标：ROC-AUC / PR-AUC / Brier / Lift@10,20,30,50。
不做交易回测，不进入 Phase 2。

时间切分沿用现有 4 折；新增 label availability purge。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import research.m2_nondeep_temporal_v1 as m2
from research.phase1_tradability.build_features_v1 import build
from research.phase1_tradability.phase1_contract_v1 import RESULTS


def prep(X: pd.DataFrame, cols: list) -> np.ndarray:
    return X[cols].to_numpy(float)


def fit_logistic(Xtr, ytr):
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=m2.SEED)).fit(Xtr, ytr)


def fit_lgbm(Xtr, ytr):
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        random_state=m2.SEED, verbose=-1, n_jobs=1).fit(Xtr, ytr)


def metrics(y, p, base):
    out = dict(n=len(y), base_rate=round(float(y.mean()), 4))
    if len(np.unique(y)) >= 2:
        out["AUC"] = round(float(roc_auc_score(y, p)), 4)
        out["PR_AUC"] = round(float(average_precision_score(y, p)), 4)
    else:
        out["AUC"] = out["PR_AUC"] = None
    out["Brier"] = round(float(brier_score_loss(y, p)), 6)
    order = np.argsort(-p)
    ys = y[order]
    for k in (0.10, 0.20, 0.30, 0.50):
        m = max(1, int(len(ys) * k))
        sel = ys[:m].mean()
        out[f"Lift@{int(k*100)}"] = round(float(sel / base), 4) if base > 0 \
            else None
    return out


def main():
    lab = pd.read_parquet(RESULTS / "labels_v1.parquet")
    cand = pd.read_parquet(RESULTS / "candidates_v1.parquet")
    X, contract, feats = build()

    # ---- 显式 candidate_id 关联（禁止位置拼接）----
    df = lab.merge(cand[["candidate_id", "trading_day"]],
                   on="candidate_id", how="left", validate="one_to_one")
    df = df.merge(X, on="candidate_id", how="inner", validate="one_to_one")
    df = df.rename(columns={"label": "tradable"})

    # ---- 硬断言 ----
    assert df["candidate_id"].is_unique, "candidate_id 不唯一"
    res = df[df["status"] == "RESOLVED"].copy()
    assert res["tradable"].isin([0, 1]).all(), "tradable 只能 0/1"
    assert (res["resolution_time"] > res["decision_time"]).all(), \
        "resolution_time 必须晚于 decision_time"
    assert not (set(feats) & {"trade_direction", "trade_mode", "target_R"}), \
        "混入了动作字段"
    assert sum("_rel_" in c for c in feats) == 0, "混入了 _rel_ 动作相对字段"
    assert not any(c.startswith("target_fit_") for c in feats), \
        "混入了 target_fit 字段"
    print(f"[phase1] resolved={len(res)} features={len(feats)}", flush=True)

    ev_day = res["trading_day"].to_numpy()
    folds, uniq = m2.build_folds(ev_day, len(res))
    cat_cols = ["symbol_code", "source_tf_code"]
    res["symbol_code"] = res["symbol"].astype("category").cat.codes
    res["source_tf_code"] = res["source_tf"].astype("category").cat.codes

    main_rows, fold_rows, sym_rows, tf_rows, dec_frames = [], [], [], [], []
    avail_rows = []

    for fi in range(4):
        tr_d, se_d, te_d = folds[fi]
        tr = res[res["trading_day"].isin(tr_d)]
        se = res[res["trading_day"].isin(se_d)]
        te = res[res["trading_day"].isin(te_d)]
        se_start = se["decision_time"].min()
        te_start = te["decision_time"].min()

        tr_ok = tr[tr["resolution_time"] < se_start]
        se_ok = se[se["resolution_time"] < te_start]
        avail_rows.append(dict(
            折=f"F{fi+1}",
            TRAIN原始=len(tr), TRAIN可用=len(tr_ok),
            TRAIN因label未resolution删除=int(len(tr) - len(tr_ok)),
            SELECTION原始=len(se), SELECTION可用=len(se_ok),
            SELECTION因label未resolution删除=int(len(se) - len(se_ok)),
            TEST=len(te)))
        if len(tr_ok) < 200 or len(te) < 100:
            continue

        ytr = tr_ok["tradable"].to_numpy(int)
        yte = te["tradable"].to_numpy(int)
        base = float(ytr.mean())

        specs = {
            "Baseline1_symbol_source_tf": (cat_cols, "logistic"),
            "Model1_Logistic": (feats, "logistic"),
            "Model2_LightGBM": (feats, "lgbm"),
        }
        for name, (cols, kind) in specs.items():
            Xtr = prep(tr_ok, cols)
            Xte = prep(te, cols)
            mdl = (fit_logistic(Xtr, ytr) if kind == "logistic"
                   else fit_lgbm(Xtr, ytr))
            p = mdl.predict_proba(Xte)[:, 1]
            m = metrics(yte, p, float(yte.mean()))
            fold_rows.append(dict(折=f"F{fi+1}", 模型=name, **m))
            d = pd.DataFrame(dict(pred=p, actual=yte))
            d["decile"] = pd.qcut(d["pred"].rank(method="first"), 10,
                                  labels=False) + 1
            dec_frames.append(d.assign(折=f"F{fi+1}", 模型=name))
            for s, gg in pd.DataFrame(dict(p=p, y=yte,
                                           sym=te["symbol"].to_numpy(),
                                           tf=te["source_tf"].to_numpy())
                                      ).groupby("sym"):
                mm = metrics(gg["y"].to_numpy(), gg["p"].to_numpy(),
                             float(gg["y"].mean()))
                sym_rows.append(dict(品种=s, 模型=name, **mm))
            for s, gg in pd.DataFrame(dict(p=p, y=yte,
                                           tf=te["source_tf"].to_numpy())
                                      ).groupby("tf"):
                mm = metrics(gg["y"].to_numpy(), gg["p"].to_numpy(),
                             float(gg["y"].mean()))
                tf_rows.append(dict(source_tf=s, 模型=name, **mm))
        print(f"[phase1] F{fi+1} done", flush=True)

    # 汇总：跨折合并预测
    all_dec = pd.concat(dec_frames, ignore_index=True)
    main_rows = []
    for name, g in all_dec.groupby("模型"):
        y = g["actual"].to_numpy()
        p = g["pred"].to_numpy()
        mm = metrics(y, p, float(y.mean()))
        mm["模型"] = name
        main_rows.append(mm)
        dg = g.groupby("decile").agg(
            n=("actual", "size"), pred=("pred", "mean"),
            actual=("actual", "mean")).reset_index()
        dg["pred"] = dg["pred"].round(6)
        dg["actual"] = dg["actual"].round(6)
        dg["模型"] = name
        dg.to_csv(RESULTS / f"probability_deciles_{name}.csv", index=False,
                  encoding="utf-8-sig")
        sp = spearmanr(dg["decile"], dg["actual"])[0]
        mm["decile_spearman"] = round(float(sp), 4)
    # Baseline0
    y_all = all_dec[all_dec["模型"] == "Model2_LightGBM"]["actual"].to_numpy()
    main_rows.append(dict(模型="Baseline0_base_rate", n=len(y_all),
                          base_rate=round(float(y_all.mean()), 4), AUC=None,
                          PR_AUC=None, Brier=None, decile_spearman=None))
    main_df = pd.DataFrame(main_rows)

    # 覆盖率 Lift 汇总
    lift_rows = []
    for name, g in all_dec.groupby("模型"):
        y = g["actual"].to_numpy()
        p = g["pred"].to_numpy()
        base = float(y.mean())
        order = np.argsort(-p)
        ys = y[order]
        for k in (0.10, 0.20, 0.30, 0.50):
            m = max(1, int(len(ys) * k))
            sel = ys[:m]
            lift_rows.append(dict(
                模型=name, coverage=k, 事件数=m,
                实际tradable率=round(float(sel.mean()), 4),
                precision=round(float(sel.mean()), 4),
                recall=round(float(sel.sum() / max(1, ys.sum())), 4),
                lift=round(float(sel.mean() / base), 4),
                base_rate=round(base, 4)))
    pd.DataFrame(lift_rows).to_csv(RESULTS / "coverage_lift.csv", index=False,
                                   encoding="utf-8-sig")

    # DEDUP 敏感性：每个 candidate_group_id 只留最早 trigger
    dedup = res.sort_values(["candidate_group_id", "decision_time"]) \
               .groupby("candidate_group_id", as_index=False).head(1)
    ded_rows = []
    ded_rows.append(dict(集合="ALL_EVENTS", 事件数=len(res),
                         base_rate=round(float(res["tradable"].mean()), 4)))
    ded_rows.append(dict(集合="DEDUP_EVENTS", 事件数=len(dedup),
                         base_rate=round(float(dedup["tradable"].mean()), 4)))

    pd.DataFrame(avail_rows).to_csv(RESULTS / "fold_availability.csv",
                                    index=False, encoding="utf-8-sig")
    main_df.to_csv(RESULTS / "model_main.csv", index=False,
                   encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(RESULTS / "model_by_fold.csv", index=False,
                                   encoding="utf-8-sig")
    pd.DataFrame(sym_rows).to_csv(RESULTS / "model_by_symbol.csv", index=False,
                                  encoding="utf-8-sig")
    pd.DataFrame(tf_rows).to_csv(RESULTS / "model_by_source_tf.csv",
                                 index=False, encoding="utf-8-sig")
    pd.DataFrame(ded_rows).to_csv(RESULTS / "dedup_sensitivity.csv",
                                  index=False, encoding="utf-8-sig")
    all_dec.to_parquet(RESULTS / "predictions_v1.parquet", index=False)

    (RESULTS / "audit.json").write_text(json.dumps(dict(
        script="research/phase1_tradability/run_phase1_v1.py",
        resolved=int(len(res)), features=len(feats),
        base_rate=round(float(res["tradable"].mean()), 4),
        dedup_base_rate=round(float(dedup["tradable"].mean()), 4),
        seed=m2.SEED,
    ), ensure_ascii=False, indent=2), encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("\n=== label availability purge ===")
    print(pd.DataFrame(avail_rows).to_string(index=False))
    print("\n=== 总体 ===")
    print(main_df.to_string(index=False))
    print("\n=== 4 折 ===")
    print(pd.DataFrame(fold_rows).round(4).to_string(index=False))
    print("\n=== 4 品种 ===")
    print(pd.DataFrame(sym_rows).round(4).to_string(index=False))
    print("\n=== source_tf ===")
    print(pd.DataFrame(tf_rows).round(4).to_string(index=False))
    print("\n=== Lift ===")
    print(pd.DataFrame(lift_rows).round(4).to_string(index=False))
    print("\n=== DEDUP ===")
    print(pd.DataFrame(ded_rows).to_string(index=False))
    print("\nPHASE1_DONE")


if __name__ == "__main__":
    main()
