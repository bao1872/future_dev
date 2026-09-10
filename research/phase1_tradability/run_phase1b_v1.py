"""Phase 1B — 16 品种跨品种泛化验证。

第一层（核心）：DEV4 训练 → NEW12 zero-shot 测试。
第二层：ALL16 重新训练 → ALL16 测试。

冻结：PHASE1_FEATURES_V1、Logistic 参数、预处理、rolling folds、
24bar 主标签、candidate_id 因果合同。symbol 不进入全模型特征。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from scipy.stats import spearmanr

import research.m2_nondeep_temporal_v1 as m2
from research.phase1_tradability.phase1_contract_v1 import RESULTS
from research.phase1_tradability.run_phase1_v1 import fit_logistic, prep

DEV4 = ("AG", "CU", "M", "RB")

# Phase 1 Candidate v1：primary label 已由 24bar 改为 **12 valid 5m bars**。
# 依据：DEV4 / Phase 1A / NEW11 zero-shot 三层验证均显示信号随 horizon
# 单调衰减，且 NEW11 从未参与 horizon discovery 仍重现 12>24>48>unbounded。
# 24 / 48 / unbounded 仅保留为历史 sensitivity，不再用于任何结论。
H_PRIMARY = "12"
H_SENSITIVITY = ("24", "48", "unb")
HORIZONS = (H_PRIMARY,) + H_SENSITIVITY
BOOT = 500


def metrics_block(y, p) -> dict:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    out = dict(n=len(y), base_rate=round(float(y.mean()), 4))
    if len(np.unique(y)) > 1:
        out["AUC"] = round(float(roc_auc_score(y, p)), 4)
        out["PR_AUC"] = round(float(average_precision_score(y, p)), 4)
        out["Brier"] = round(float(brier_score_loss(y, p)), 6)
    else:
        out["AUC"] = out["PR_AUC"] = out["Brier"] = None
    if len(y) >= 50:
        d = pd.qcut(pd.Series(p).rank(method="first"), 10,
                    labels=False) + 1
        d1 = float(y[d == 1].mean()) if (d == 1).any() else np.nan
        d10 = float(y[d == 10].mean()) if (d == 10).any() else np.nan
        out["D1"] = round(d1, 4)
        out["D10"] = round(d10, 4)
        out["D10_minus_D1"] = round(d10 - d1, 4)
        cuts = np.quantile(p, [0.9, 0.8, 0.7])
        for q, k in zip(cuts, (10, 20, 30)):
            m = p >= q
            r = float(y[m].mean()) if m.any() else np.nan
            out[f"Top{k}_rate"] = round(r, 4)
            out[f"Lift@{k}"] = round(r / y.mean(), 4) if y.mean() > 0 else None
            out[f"Top{k}_uplift"] = round(r - y.mean(), 4)
        out["decile_spearman"] = round(float(spearmanr(p, y).statistic), 4)
    return out


def load(H):
    lab = pd.read_parquet(RESULTS / "labels_v16.parquet")
    X = pd.read_parquet(RESULTS / "features_v16.parquet")
    cand = pd.read_parquet(RESULTS / "candidates_v16.parquet")
    lab["candidate_id"] = lab["candidate_id"].astype(str)
    X["candidate_id"] = X["candidate_id"].astype(str)
    r = lab[lab[f"status_{H}"] == "RESOLVED"].copy()
    r["y"] = r[f"tradable_{H}"].astype(int)
    r = r.merge(X, on="candidate_id", how="inner", validate="one_to_one")
    r = r.merge(cand[["candidate_id", "trading_day", "candidate_group_id"]],
                on="candidate_id", how="left", validate="one_to_one")
    r["symbol_code"] = r["symbol"].astype("category").cat.codes
    r["source_tf_code"] = r["source_tf"].astype("category").cat.codes
    feats = [c for c in X.columns if c != "candidate_id"]
    return r.reset_index(drop=True), feats


def run_folds(r, feats):
    """返回逐折预测：DEV4 fit → DEV4/NEW12 test；ALL16 fit → ALL16 test。"""
    folds, _ = m2.build_folds(r["trading_day"].to_numpy(), len(r))
    dev4m = r["symbol"].isin(DEV4).to_numpy()
    recs = []
    for fi in range(4):
        tr_d, se_d, te_d = folds[fi]
        tr = r[r["trading_day"].isin(tr_d)]
        se = r[r["trading_day"].isin(se_d)]
        te = r[r["trading_day"].isin(te_d)]
        if len(te) < 100:
            continue
        se_start = se["decision_time"].min()

        # ---- DEV4 fit（zero-shot 模型：只用 DEV4 TRAIN）----
        dtr = tr[tr["symbol"].isin(DEV4)]
        dtr = dtr[dtr["resolution_time"] < se_start]
        if len(dtr) < 200:
            continue
        ytr = dtr["y"].to_numpy(int)
        md = fit_logistic(prep(dtr, feats), ytr)
        # source_tf-only baseline（同样只在 DEV4 TRAIN 拟合）
        mb = fit_logistic(prep(dtr, ["source_tf_code"]), ytr)

        tem = te.copy()
        tem["pred_full"] = md.predict_proba(prep(tem, feats))[:, 1]
        tem["pred_base"] = mb.predict_proba(prep(tem, ["source_tf_code"]))[:, 1]

        # ---- ALL16 fit ----
        atr = tr[tr["resolution_time"] < se_start]
        if len(atr) >= 200:
            m16 = fit_logistic(prep(atr, feats), atr["y"].to_numpy(int))
            m16b = fit_logistic(prep(atr, ["symbol_code", "source_tf_code"]),
                                atr["y"].to_numpy(int))
            tem["pred_all16"] = m16.predict_proba(prep(tem, feats))[:, 1]
            tem["pred_all16_base"] = m16b.predict_proba(
                prep(tem, ["symbol_code", "source_tf_code"]))[:, 1]
        else:
            tem["pred_all16"] = np.nan
            tem["pred_all16_base"] = np.nan

        tem["折"] = f"F{fi+1}"
        tem["is_dev4"] = tem["symbol"].isin(DEV4)
        recs.append(tem)
    return pd.concat(recs, ignore_index=True)


def macro(P, col):
    rows = []
    for s, g in P.groupby("symbol"):
        d = metrics_block(g["y"], g[col])
        d["symbol"] = s
        rows.append(d)
    t = pd.DataFrame(rows)
    agg = dict(
        n_symbols=len(t),
        macro_mean_AUC=round(float(t["AUC"].mean()), 4),
        macro_median_AUC=round(float(t["AUC"].median()), 4),
        macro_mean_Lift20=round(float(t["Lift@20"].mean()), 4),
        macro_median_Lift20=round(float(t["Lift@20"].median()), 4),
        macro_mean_D10_minus_D1=round(float(t["D10_minus_D1"].mean()), 4),
        macro_median_D10_minus_D1=round(
            float(t["D10_minus_D1"].median()), 4),
        positive_symbols=int((t["Lift@20"] > 1).sum()),
        meaningful_symbols=int((t["Lift@20"] >= 1.03).sum()),
        median_lift20=round(float(t["Lift@20"].median()), 4),
        p25_lift20=round(float(t["Lift@20"].quantile(.25)), 4),
        p75_lift20=round(float(t["Lift@20"].quantile(.75)), 4),
    )
    return t, agg


def bootstrap(P, col, n=BOOT, seed=7):
    rng = np.random.default_rng(seed)
    days = P["trading_day"].dropna().unique()
    idx = {d: np.flatnonzero((P["trading_day"] == d).to_numpy())
           for d in days}
    y = P["y"].to_numpy(int)
    p = P[col].to_numpy(float)
    d10, up = [], []
    for _ in range(n):
        pick = rng.choice(days, size=len(days), replace=True)
        rows = np.concatenate([idx[d] for d in pick])
        yy, pp = y[rows], p[rows]
        if len(np.unique(yy)) < 2 or len(yy) < 200:
            continue
        d = pd.qcut(pd.Series(pp).rank(method="first"), 10,
                    labels=False) + 1
        d10.append(yy[d == 10].mean() - yy[d == 1].mean())
        m = pp >= np.quantile(pp, 0.8)
        up.append(yy[m].mean() - yy.mean())
    return dict(
        D10_minus_D1_ci=[round(float(np.quantile(d10, .025)), 4),
                         round(float(np.quantile(d10, .975)), 4)],
        Top20_uplift_ci=[round(float(np.quantile(up, .025)), 4),
                         round(float(np.quantile(up, .975)), 4)],
        n_rep=len(d10),
    )


def feature_shift(P, feats):
    """DEV4 TRAIN vs NEW TEST 的特征漂移审计（以 DEV4 TRAIN 标准化）。"""
    tr = P[(P["is_dev4"]) & (P["折"] == "F1")]
    te = P[~P["is_dev4"]]
    rows = []
    for c in feats:
        a = pd.to_numeric(tr[c], errors="coerce").to_numpy(float)
        b = pd.to_numeric(te[c], errors="coerce").to_numpy(float)
        if np.isnan(a).all() or np.isnan(b).all():
            continue
        mu, sd = np.nanmean(a), np.nanstd(a)
        if not np.isfinite(sd) or sd == 0:
            continue
        zb = (b - mu) / sd
        rows.append(dict(
            feature=c,
            median_shift=round(float(np.nanmedian(zb)), 4),
            missing_diff=round(float(np.isnan(b).mean()
                                     - np.isnan(a).mean()), 4),
            frac_abs_z_gt5=round(float(np.nanmean(np.abs(zb) > 5)), 5),
        ))
    t = pd.DataFrame(rows).sort_values(
        "median_shift", key=lambda s: s.abs(), ascending=False)
    t.to_csv(RESULTS / "phase1b_feature_shift.csv", index=False,
             encoding="utf-8-sig")
    return t


CANON16 = ("AG", "AU", "CU", "AL", "SN", "NI", "RB", "I",
           "SC", "RU", "MA", "TA", "M", "P", "CF", "LC")


def symbol_universe():
    """发现真实 universe 并逐品种记录纳入/排除原因。"""
    cand = pd.read_parquet(RESULTS / "candidates_v16.parquet")
    built = {p.name for p in Path(
        "research/analysis_data/ob_candidate_universe_v3/candidates"
    ).iterdir() if p.is_dir()}
    rows = []
    for s in CANON16:
        raw = Path(f"research/exports/v3r_5m/{s}_5m.csv")
        n = int((cand["symbol"] == s).sum()) if "symbol" in cand else 0
        if not raw.exists():
            rows.append(dict(symbol=s, raw_5m_available=False,
                             canonical_ob_available=False,
                             feature_available=False,
                             first_timestamp=None, last_timestamp=None,
                             candidate_count=0, source_tf_coverage="",
                             included=False,
                             exclusion_reason="无 raw 5m 数据"))
            continue
        d = pd.read_csv(raw, usecols=["bar_start_time"])
        inc = s in built
        tfs = (sorted(cand.loc[cand["symbol"] == s, "source_tf"]
                      .unique().tolist()) if n else [])
        rows.append(dict(
            symbol=s, raw_5m_available=True,
            canonical_ob_available=inc, feature_available=inc,
            first_timestamp=str(d["bar_start_time"].min()),
            last_timestamp=str(d["bar_start_time"].max()),
            candidate_count=n,
            source_tf_coverage="|".join(tfs),
            included=inc,
            exclusion_reason=("" if inc else
                              "quantile 训练样本不足(len(tr)<600)，"
                              "canonical OB universe 无法构建"),
        ))
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "symbol_universe.csv", index=False,
             encoding="utf-8-sig")
    print("\n=== symbol universe ===")
    print(t[["symbol", "raw_5m_available", "canonical_ob_available",
             "candidate_count", "included"]].to_string(index=False))
    return t


def main():
    out = {}
    symbol_universe()
    for H in HORIZONS:
        r, feats = load(H)
        P = run_folds(r, feats)
        new12 = P[~P["is_dev4"]]
        dev4 = P[P["is_dev4"]]

        rec = dict(
            horizon=H,
            DEV4=metrics_block(dev4["y"], dev4["pred_full"]),
            NEW12_full=metrics_block(new12["y"], new12["pred_full"]),
            NEW12_baselineZS=metrics_block(new12["y"], new12["pred_base"]),
        )
        if H == H_PRIMARY:
            rec["ALL16_full"] = metrics_block(P["y"], P["pred_all16"])
            rec["ALL16_baseline"] = metrics_block(P["y"],
                                                  P["pred_all16_base"])
            t16, agg16 = macro(P, "pred_all16")
            t12, agg12 = macro(new12, "pred_full")
            t12["模型"] = "NEW12_zero_shot"
            t16["模型"] = "ALL16_refit"
            rec["ALL16_macro"] = agg16
            rec["NEW12_macro"] = agg12
            rec["boot_NEW12"] = bootstrap(new12, "pred_full")
            rec["boot_ALL16"] = bootstrap(P, "pred_all16")
            pd.concat([t12, t16], ignore_index=True).to_csv(
                RESULTS / "phase1b_per_symbol.csv", index=False,
                encoding="utf-8-sig")
            # symbol × fold 矩阵
            rows = []
            for (s, f), g in new12.groupby(["symbol", "折"]):
                if len(g) < 60:
                    rows.append(dict(symbol=s, 折=f, n=len(g),
                                     Lift20="LOW_SAMPLE",
                                     D10_minus_D1="LOW_SAMPLE"))
                    continue
                d = metrics_block(g["y"], g["pred_full"])
                rows.append(dict(symbol=s, 折=f, n=d["n"],
                                 Lift20=d["Lift@20"],
                                 D10_minus_D1=d["D10_minus_D1"]))
            pd.DataFrame(rows).to_csv(
                RESULTS / "phase1b_symbol_fold_matrix.csv", index=False,
                encoding="utf-8-sig")
            # 按 fold / source_tf
            for dim in ("折", "source_tf"):
                rr = []
                for k, g in new12.groupby(dim):
                    d = metrics_block(g["y"], g["pred_full"])
                    d[dim] = k
                    rr.append(d)
                fn = ("phase1b_new12_by_fold.csv" if dim == "折"
                      else "phase1b_new12_by_sourcetf.csv")
                pd.DataFrame(rr).to_csv(RESULTS / fn, index=False,
                                        encoding="utf-8-sig")
            P.to_parquet(RESULTS / "phase1b_predictions.parquet",
                         index=False)
            fs = feature_shift(P, feats)
            print("\n=== feature shift 最大前 8 ===")
            print(fs.head(8).to_string(index=False))
            # DEDUP
            if "candidate_group_id" in P.columns:
                ded = (P.sort_values(["candidate_group_id", "decision_time"])
                        .groupby("candidate_group_id", as_index=False).head(1))
                rec["DEDUP_NEW12"] = metrics_block(
                    ded[~ded["is_dev4"]]["y"], ded[~ded["is_dev4"]]["pred_full"])
                rec["DEDUP_ALL16"] = metrics_block(ded["y"], ded["pred_all16"])
        out[H] = rec
        print(f"[H={H}] DEV4 AUC={rec['DEV4']['AUC']} "
              f"D10-D1={rec['DEV4'].get('D10_minus_D1')} | "
              f"NEW12 AUC={rec['NEW12_full']['AUC']} "
              f"D10-D1={rec['NEW12_full'].get('D10_minus_D1')} "
              f"Lift20={rec['NEW12_full'].get('Lift@20')}", flush=True)
        if H != H_PRIMARY:
            for k in ("DEV4", "NEW12_full"):
                d = dict(rec[k]); d["universe"] = k; d["horizon"] = H
    
    (RESULTS / "phase1b_results.json").write_text(
        json.dumps(out, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8")
    a = out[H_PRIMARY]
    print(f"\n=== 核心对照（{H_PRIMARY}bar primary）===")
    for k in ("DEV4", "NEW12_full", "ALL16_full"):
        d = a[k]
        print(f"  {k:12s} n={d['n']:6d} AUC={d['AUC']} "
              f"D10-D1={d.get('D10_minus_D1')} Lift20={d.get('Lift@20')} "
              f"uplift={d.get('Top20_uplift')}")
    print("  NEW12 macro:", a["NEW12_macro"])
    print("  ALL16 macro:", a["ALL16_macro"])
    print("  boot NEW12:", a["boot_NEW12"])
    print("  boot ALL16:", a["boot_ALL16"])
    print("PHASE1B_DONE")


if __name__ == "__main__":
    main()
