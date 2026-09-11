"""SMC Opportunity Geometry Decomposition v1.1 —— walk-forward 几何分解。

固定 Logistic；7 risk × 3 WF；G1–G5 子块分解；机械基准 geometry_ratio；
OB freshness bug 修复后复跑 M_ob/M_full/M_full_minus_ob；
canonical trading_day block bootstrap 500；跨15品种 delta。

G4（多周期距离分箱）在冻结 Atlas v1.2 不存在 → 本轮 UNAVAILABLE，已诚实标注。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

import opportunity_common as oc

OUT = Path("research/analysis_results/smc_opportunity_geometry_v1_1")
RNG = np.random.default_rng(20260911)
N_BOOT = 500

# ---- 子块（仅冻结 Atlas 真实存在的列）----
G1 = ["nearest_above_R", "nearest_below_R"]
G2 = ["n_targets_L", "n_targets_S"]
G3 = ["nearest_ahead_R", "nearest_behind_R"]
G4 = []   # UNAVAILABLE
G5 = ["same_price_identity_count"]
B0, B1, B2, B3, B4 = (oc.B0_COLS, oc.B1_COLS, oc.B2_COLS,
                       oc.B3_COLS, oc.B4_COLS)
ALL_FEAT = B0 + B1 + B2 + B3 + B4
SPLINE = set(oc.SPLINE_NUMERIC)

# M_field == M_simple（G4 空）；M_B2mG4 == M_B2_full（G4 空）。仍记录以便读表。
MODEL_BLOCKS = {
    "M0": B0,
    "M_G1": B0 + G1, "M_G2": B0 + G2, "M_G3": B0 + G3, "M_G4": B0 + G4,
    "M_G5": B0 + G5,
    "M_simple": B0 + G1 + G2,
    "M_field": B0 + G1 + G2,                 # == M_simple
    "M_B2_full": B0 + G1 + G2 + G3 + G4 + G5,
    "M_ob": B0 + B4,
    "M_full": B0 + B1 + B2 + B3 + B4,
    "M_full_minus_ob": B0 + B1 + B2 + B3,
    "M_B2mG1": B0 + G2 + G3 + G5,
    "M_B2mG2": B0 + G1 + G3 + G5,
    "M_B2mG3": B0 + G1 + G2 + G5,
    "M_B2mG4": B0 + G1 + G2 + G3 + G5,       # == M_B2_full
    "M_B2mG5": B0 + G1 + G2 + G3,
}
MODEL_NAMES = list(MODEL_BLOCKS.keys())

WF = [("WF1", ["TB1"], "TB2"), ("WF2", ["TB1", "TB2"], "TB3"),
      ("WF3", ["TB1", "TB2", "TB3"], "TB4")]


def make_pipeline(cols):
    cat = [c for c in cols if c not in SPLINE]
    spl = [c for c in cols if c in SPLINE]
    pre = ColumnTransformer([
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ("spl", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sp", SplineTransformer(n_knots=4, degree=2, knots="quantile",
                                    include_bias=False)),
            ("sc", StandardScaler())]), spl),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                   max_iter=3000)),
    ])


def _fast_auc(y, p):
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def metrics_vec(y, P):
    n, M = P.shape
    roc = np.full(M, np.nan)
    pr = np.full(M, np.nan)
    br = np.full(M, np.nan)
    ll = np.full(M, np.nan)
    yb = y.astype(np.int64)
    for j in range(M):
        p = np.clip(P[:, j], 1e-6, 1 - 1e-6)
        if len(np.unique(yb)) < 2:
            continue
        roc[j] = _fast_auc(yb, p)
        pr[j] = average_precision_score(yb, p)
        br[j] = brier_score_loss(yb, p)
        ll[j] = log_loss(yb, p)
    return roc, pr, br, ll


def rank_metrics(y, P, base_rate):
    n, M = P.shape
    top20 = np.full(M, np.nan)
    bot20 = np.full(M, np.nan)
    order = np.argsort(-P, axis=0)
    k = max(1, int(round(0.2 * n)))
    for j in range(M):
        o = order[:, j]
        top20[j] = y[o[:k]].mean()
        bot20[j] = y[o[-k:]].mean()
    return top20, bot20, base_rate - bot20


print("[M1] 加载标签 + v1.1 特征 + block ...", flush=True)
lab = oc.load_labels()
F = pd.read_parquet(OUT / "opportunity_features_v1_1.parquet")
keys = ["symbol", "liquidity_id", "contact_number", "risk_ATR"]
blk = oc.attach_trading_day_block(lab[keys].copy())
lab = lab.copy()
lab["block"] = blk["block"].values
lab["trading_day"] = blk["trading_day"].values
F = F.copy()
F["block"] = blk["block"].values
F["trading_day"] = blk["trading_day"].values
F["y"] = (lab["opportunity_label"].values == "DELIVERY").astype(int)
F["primary_eligible"] = lab["primary_eligible"].values

print("[M2] 逐 risk × WF × model 训练与评估 ...", flush=True)
rows, oof_rows, boot_requests = [], [], []
for r in oc.RISK_GRID:
    sub = F[(F["risk_ATR"] == r) & F["primary_eligible"]].reset_index(drop=True)
    X = sub[ALL_FEAT].copy()
    y = sub["y"].values
    base_rate = y.mean()
    for wf_name, tr_blocks, te_block in WF:
        tr = sub["block"].isin(tr_blocks).values
        te = (sub["block"] == te_block).values
        if tr.sum() == 0 or te.sum() == 0:
            continue
        ytr, yte = y[tr], y[te]
        day_te = sub["trading_day"].values[te]
        Pmat = np.zeros((te.sum(), len(MODEL_NAMES)))
        per_sym = {m: {} for m in MODEL_NAMES}
        for j, (mname, cols) in enumerate(MODEL_BLOCKS.items()):
            mcols = [c for c in cols if c in ALL_FEAT]
            pipe = make_pipeline(mcols)
            pipe.fit(X[tr][mcols], ytr)
            proba = pipe.predict_proba(X[te][mcols])[:, 1]
            Pmat[:, j] = proba
            St = sub[te].reset_index(drop=True)
            for sym, g in St.groupby("symbol"):
                if len(np.unique(g["y"].values)) < 2:
                    continue
                per_sym[mname][sym] = roc_auc_score(
                    g["y"].values, proba[g.index.values])
        roc, pr, br, ll = metrics_vec(yte, Pmat)
        top20, bot20, gain = rank_metrics(yte, Pmat, base_rate)
        for j, mname in enumerate(MODEL_NAMES):
            rows.append(dict(risk_ATR=r, wf=wf_name, model=mname,
                             n_train=int(tr.sum()), n_test=int(te.sum()),
                             base_delivery_rate=base_rate,
                             roc_auc=roc[j], pr_auc=pr[j], brier=br[j],
                             logloss=ll[j], top20_delivery_rate=top20[j],
                             bottom20_delivery_rate=bot20[j],
                             bottom20_avoidance_gain=gain[j]))
        oof_df = sub[te][["symbol", "liquidity_id", "contact_number",
                          "block", "y"]].reset_index(drop=True)
        oof_df = oof_df.rename(columns={"y": "y_true"})
        for j, mname in enumerate(MODEL_NAMES):
            oof_df[mname] = Pmat[:, j]
        oof_df["risk_ATR"] = r
        oof_df["wf"] = wf_name
        oof_rows.append(oof_df)
        boot_requests.append((r, wf_name, yte, day_te, Pmat, base_rate, per_sym))
    print(f"  [M2] risk={r} 完成", flush=True)

wf_audit = pd.DataFrame(rows)
wf_audit.to_csv(OUT / "walkforward_audit.csv", index=False, encoding="utf-8-sig")
oof_all = pd.concat(oof_rows, ignore_index=True)
oof_all.to_parquet(OUT / "oof_predictions.parquet", index=False)
print(f"    walkforward_audit rows={len(wf_audit)}", flush=True)

print("[M3] bootstrap 95% CI (canonical trading_day block) ...", flush=True)
boot_rows = []
for (r, wf_name, yte, day_te, Pmat, base_rate, _) in boot_requests:
    days = np.unique(day_te)
    n, M = Pmat.shape
    roc_b = np.zeros((N_BOOT, M))
    br_b = np.zeros((N_BOOT, M))
    gain_b = np.zeros((N_BOOT, M))
    for b in range(N_BOOT):
        if b % 100 == 0:
            print(f"    boot {b}/{N_BOOT} risk={r} wf={wf_name}", flush=True)
        samp = RNG.choice(days, size=len(days), replace=True)
        mask = np.isin(day_te, samp)
        if mask.sum() < 30:
            continue
        yb = yte[mask]
        Pb = Pmat[mask]
        roc, _, br, _ = metrics_vec(yb, Pb)
        _, bot20, gain = rank_metrics(yb, Pb, base_rate)
        roc_b[b] = roc
        br_b[b] = br
        gain_b[b] = gain
    for j, mname in enumerate(MODEL_NAMES):
        boot_rows.append(dict(
            risk_ATR=r, wf=wf_name, model=mname,
            roc_auc_lo=np.nanpercentile(roc_b[:, j], 2.5),
            roc_auc_hi=np.nanpercentile(roc_b[:, j], 97.5),
            brier_lo=np.nanpercentile(br_b[:, j], 2.5),
            brier_hi=np.nanpercentile(br_b[:, j], 97.5),
            bottom20_gain_lo=np.nanpercentile(gain_b[:, j], 2.5),
            bottom20_gain_hi=np.nanpercentile(gain_b[:, j], 97.5)))
pd.DataFrame(boot_rows).to_csv(OUT / "subblock_bootstrap_ci.csv", index=False,
                               encoding="utf-8-sig")

print("[M4] 跨品种子块 delta（per-symbol AUC vs M0）...", flush=True)
sym_rows = []
for (r, wf_name, yte, day_te, Pmat, base_rate, per_sym) in boot_requests:
    m0 = per_sym["M0"]
    for mname in MODEL_NAMES:
        if mname == "M0":
            continue
        deltas = []
        for sym in set(m0) | set(per_sym[mname]):
            a0 = m0.get(sym, np.nan)
            a1 = per_sym[mname].get(sym, np.nan)
            if np.isnan(a0) or np.isnan(a1):
                continue
            deltas.append((sym, a1 - a0))
        if not deltas:
            continue
        ds = pd.DataFrame(deltas, columns=["symbol", "delta_auc"])
        sym_rows.append(dict(
            risk_ATR=r, wf=wf_name, model=mname,
            n_symbols=len(ds),
            macro_median_delta=ds["delta_auc"].median(),
            iqr_lo=ds["delta_auc"].quantile(0.25),
            iqr_hi=ds["delta_auc"].quantile(0.75),
            positive_symbols=int((ds["delta_auc"] > 0).sum()),
            eligible_symbols=int((ds["delta_auc"] != 0).sum() or len(ds))))
pd.DataFrame(sym_rows).to_csv(OUT / "subblock_by_symbol.csv", index=False,
                              encoding="utf-8-sig")

print("[M5] 子块 marginal / conditional deltas ...", flush=True)
# marginal: each model - M0
m0a = wf_audit[wf_audit.model == "M0"].set_index(["risk_ATR", "wf"])["roc_auc"]
m0g = wf_audit[wf_audit.model == "M0"].set_index(["risk_ATR", "wf"])[
    "bottom20_avoidance_gain"]
marg = wf_audit[wf_audit.model != "M0"].copy()
idx = marg.set_index(["risk_ATR", "wf"]).index
marg["dROC_vs_M0"] = marg["roc_auc"] - idx.map(m0a)
marg["dBottom20gain_vs_M0"] = (marg["bottom20_avoidance_gain"]
                               - idx.map(m0g))
marg[["risk_ATR", "wf", "model", "dROC_vs_M0",
      "dBottom20gain_vs_M0"]].to_csv(OUT / "subblock_marginal_deltas.csv",
                                     index=False, encoding="utf-8-sig")
# conditional: M_B2_full - M_B2mGx
full = wf_audit[wf_audit.model == "M_B2_full"].set_index(["risk_ATR", "wf"])
cond = []
for blk_name, full_minus in [("G1", "M_B2mG1"), ("G2", "M_B2mG2"),
                             ("G3", "M_B2mG3"), ("G4", "M_B2mG4"),
                             ("G5", "M_B2mG5")]:
    fm = wf_audit[wf_audit.model == full_minus].set_index(["risk_ATR", "wf"])
    cidx = full.index
    cond.append(pd.DataFrame({
        "risk_ATR": [i[0] for i in cidx],
        "wf": [i[1] for i in cidx],
        "removed_block": blk_name,
        "dROC_Full_minus_block": (full.loc[cidx, "roc_auc"].values
                                  - fm.loc[cidx, "roc_auc"].values),
        "dBottom20gain_Full_minus_block": (
            full.loc[cidx, "bottom20_avoidance_gain"].values
            - fm.loc[cidx, "bottom20_avoidance_gain"].values),
    }))
pd.concat(cond, ignore_index=True).to_csv(
    OUT / "subblock_conditional_deltas.csv", index=False, encoding="utf-8-sig")

print("[M6] 汇总 CSV + 机械基准 geometry_ratio ...", flush=True)
wf_audit.groupby(["risk_ATR", "model"])[
    ["roc_auc", "pr_auc", "brier", "logloss", "top20_delivery_rate",
     "bottom20_delivery_rate", "bottom20_avoidance_gain"]].mean().reset_index()\
    .to_csv(OUT / "subblock_metrics_by_risk.csv", index=False, encoding="utf-8-sig")
wf_audit.groupby(["wf", "model"])[
    ["roc_auc", "pr_auc", "brier", "logloss", "top20_delivery_rate",
     "bottom20_delivery_rate", "bottom20_avoidance_gain"]].mean().reset_index()\
    .to_csv(OUT / "subblock_metrics_by_wf.csv", index=False, encoding="utf-8-sig")

# 机械基准：nearest_any_target_R / risk_ATR
F2 = F.copy()
F2["nearest_any_target_R"] = F2[["nearest_above_R", "nearest_below_R"]].min(axis=1)
F2["geometry_ratio"] = F2["nearest_any_target_R"] / F2["risk_ATR"]
bins = [-np.inf, 0.25, 0.5, 1.0, 2.0, np.inf]
labels = ["<=0.25", "(0.25,0.5]", "(0.5,1]", "(1,2]", ">2"]
F2["geo_bin"] = pd.cut(F2["geometry_ratio"], bins=bins, labels=labels)
prof = F2[(F2["primary_eligible"]) &
          (F2["y"].isin([0, 1]))].copy()  # DELIVERY=1, NO_DELIVERY=0
prof["is_delivery"] = prof["y"]
gp = (prof.groupby(["risk_ATR", "block", "symbol", "geo_bin"], observed=True)
      .agg(n=("is_delivery", "size"),
           delivery_rate=("is_delivery", "mean"),
           no_delivery_rate=("is_delivery", lambda s: 1 - s.mean()))
      .reset_index())
gp.to_csv(OUT / "geometry_ratio_profile.csv", index=False, encoding="utf-8-sig")
# 简洁汇总（按 risk × bin，跨 block/symbol 聚合）
gp_sum = (prof.groupby(["risk_ATR", "geo_bin"], observed=True)
          .agg(n=("is_delivery", "size"),
               delivery_rate=("is_delivery", "mean"),
               no_delivery_rate=("is_delivery", lambda s: 1 - s.mean()))
          .reset_index())
gp_sum.to_csv(OUT / "geometry_ratio_summary.csv", index=False,
              encoding="utf-8-sig")

print("[M7] OB freshness 修复前后对照审计 ...", flush=True)
v10 = pd.read_csv(Path("research/analysis_results/smc_opportunity_v1")
                  / "model_metrics_by_risk.csv")
cur = wf_audit[wf_audit.model.isin(["M_ob", "M_full", "M_full_minus_ob"])]
rows_ob = []
for (r, wf_name, model), g in cur.groupby(["risk_ATR", "wf", "model"]):
    roc_fix = g["roc_auc"].mean()
    gain_fix = g["bottom20_avoidance_gain"].mean()
    old = v10[(v10["risk_ATR"] == r) & (v10["model"] == model)]
    if len(old) == 0:
        continue
    roc_v10 = old["roc_auc"].mean()
    gain_v10 = old["bottom20_avoidance_gain"].mean()
    rows_ob.append(dict(risk_ATR=r, wf=wf_name, model=model,
                        roc_auc_fixed=roc_fix, roc_auc_v10=roc_v10,
                        delta_roc=roc_fix - roc_v10,
                        bottom20_gain_fixed=gain_fix,
                        bottom20_gain_v10=gain_v10,
                        delta_gain=gain_fix - gain_v10))
pd.DataFrame(rows_ob).to_csv(OUT / "ob_freshness_fix_audit.csv", index=False,
                             encoding="utf-8-sig")

print("[DONE] geometry decomposition 完成，输出目录:", OUT)
