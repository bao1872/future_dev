"""SMC Structural Delivery Opportunity Study v1.0 —— Walk-forward model.

§12-§二十三：透明 LogisticRegression；七档 risk 分开；冻结 TB1-TB4 做
内部因果时间泛化；同 risk 所有模型共用相同 test rows；canonical
trading_day block bootstrap 500 次给 95% CI；跨 15 品种报告 block delta。

不训练复杂模型、不做 PnL、不改 Atlas。证据等级=内部因果时间泛化，非独立 OOS。
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

OUT = oc.OUT
RNG = np.random.default_rng(20260911)
N_BOOT = 500

B0, B1, B2, B3, B4 = (oc.B0_COLS, oc.B1_COLS, oc.B2_COLS,
                       oc.B3_COLS, oc.B4_COLS)
ALL_FEAT = B0 + B1 + B2 + B3 + B4
SPLINE = set(oc.SPLINE_NUMERIC)

MODEL_BLOCKS = {
    "M0": B0,
    "M_contact": B0 + B1,
    "M_liquidity": B0 + B2,
    "M_trend": B0 + B3,
    "M_ob": B0 + B4,
    "M_full": B0 + B1 + B2 + B3 + B4,
    "M_full_minus_contact": B0 + B2 + B3 + B4,
    "M_full_minus_liquidity": B0 + B1 + B3 + B4,
    "M_full_minus_trend": B0 + B1 + B2 + B4,
    "M_full_minus_ob": B0 + B1 + B2 + B3,
}

WF = [
    ("WF1", ["TB1"], "TB2"),
    ("WF2", ["TB1", "TB2"], "TB3"),
    ("WF3", ["TB1", "TB2", "TB3"], "TB4"),
]

MODEL_NAMES = list(MODEL_BLOCKS.keys())


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
    """纯 numpy 秩公式 AUC（避免 sklearn 循环开销）。"""
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def metrics_vec(y, P):
    """P: (n, M) 预测概率矩阵；返回每模型 metric（list of arrays）。"""
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
    gain = base_rate - bot20
    return top20, bot20, gain


print("[M1] 加载标签 + 特征 + block ...")
lab = oc.load_labels()
F = pd.read_parquet(OUT / "opportunity_features.parquet")
keys = ["symbol", "liquidity_id", "contact_number", "risk_ATR"]
assert len(F) == len(lab)
# 附加 block / trading_day（按 contact 键）
blk = oc.attach_trading_day_block(lab[keys].copy())
lab = lab.copy()
lab["block"] = blk["block"].values
lab["trading_day"] = blk["trading_day"].values
F = F.copy()
F["block"] = blk["block"].values
F["trading_day"] = blk["trading_day"].values
F["y"] = (lab["opportunity_label"].values == "DELIVERY").astype(int)
F["primary_eligible"] = lab["primary_eligible"].values

print("[M2] 逐 risk × WF × model 训练与评估 ...")
rows = []          # walkforward_audit
oof_rows = []      # 紧凑 OOF 预测（仅 test）
boot_requests = []  # (risk, wf, y_test, day_test, Pmat) 供 bootstrap

for r in oc.RISK_GRID:
    sub_mask = (F["risk_ATR"] == r) & F["primary_eligible"]
    S = F[sub_mask].reset_index(drop=True)
    X = S[ALL_FEAT].copy()
    y = S["y"].values
    base_rate = y.mean()
    for wf_name, tr_blocks, te_block in WF:
        tr = S["block"].isin(tr_blocks).values
        te = (S["block"] == te_block).values
        if tr.sum() == 0 or te.sum() == 0:
            continue
        ytr, yte = y[tr], y[te]
        day_te = S["trading_day"].values[te]
        Pmat = np.zeros((te.sum(), len(MODEL_NAMES)))
        per_sym_auc = {m: {} for m in MODEL_NAMES}
        for j, (mname, cols) in enumerate(MODEL_BLOCKS.items()):
            mcols = [c for c in cols if c in ALL_FEAT]
            pipe = make_pipeline(mcols)
            pipe.fit(X[tr][mcols], ytr)
            proba = pipe.predict_proba(X[te][mcols])[:, 1]
            Pmat[:, j] = proba
            # per-symbol AUC
            St = S[te].reset_index(drop=True)
            for sym, g in St.groupby("symbol"):
                if len(np.unique(g["y"].values)) < 2:
                    continue
                per_sym_auc[mname][sym] = roc_auc_score(
                    g["y"].values, proba[g.index.values])
        roc, pr, br, ll = metrics_vec(yte, Pmat)
        top20, bot20, gain = rank_metrics(yte, Pmat, base_rate)
        for j, mname in enumerate(MODEL_NAMES):
            rows.append(dict(
                risk_ATR=r, wf=wf_name, model=mname,
                n_train=int(tr.sum()), n_test=int(te.sum()),
                base_delivery_rate=base_rate,
                roc_auc=roc[j], pr_auc=pr[j], brier=br[j], logloss=ll[j],
                top20_delivery_rate=top20[j],
                bottom20_delivery_rate=bot20[j],
                bottom20_avoidance_gain=gain[j],
            ))
        # 存 OOF（紧凑，仅 test 行）
        oof_df = S[te][["symbol", "liquidity_id", "contact_number",
                        "block", "y"]].reset_index(drop=True)
        oof_df = oof_df.rename(columns={"y": "y_true"})
        for j, mname in enumerate(MODEL_NAMES):
            oof_df[mname] = Pmat[:, j]
        oof_df["risk_ATR"] = r
        oof_df["wf"] = wf_name
        oof_rows.append(oof_df)
        boot_requests.append((r, wf_name, yte, day_te, Pmat, base_rate,
                             per_sym_auc))
    print(f"  [M2] risk={r} 完成 (WF×model 拟合 {len(MODEL_NAMES)*3} 个)",
          flush=True)

wf_audit = pd.DataFrame(rows)
wf_audit.to_csv(OUT / "walkforward_audit.csv", index=False, encoding="utf-8-sig")
oof_all = pd.concat(oof_rows, ignore_index=True)
oof_all.to_parquet(OUT / "oof_predictions.parquet", index=False)
print(f"    walkforward_audit rows={len(wf_audit)}  oof rows={len(oof_all)}")

print("[M3] bootstrap 95% CI (canonical trading_day block) ...")
boot_rows = []
for (r, wf_name, yte, day_te, Pmat, base_rate, _) in boot_requests:
    days = np.unique(day_te)
    n = len(yte)
    M = Pmat.shape[1]
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
            bottom20_gain_hi=np.nanpercentile(gain_b[:, j], 97.5),
        ))
boot_df = pd.DataFrame(boot_rows)
boot_df.to_csv(OUT / "block_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
print(f"    bootstrap rows={len(boot_df)}")

print("[M4] 跨品种 block delta（per-symbol AUC） ...")
sym_rows = []
for (r, wf_name, yte, day_te, Pmat, base_rate, per_sym_auc) in boot_requests:
    # 以 M0 为基准
    m0 = per_sym_auc["M0"]
    for mname in MODEL_NAMES:
        if mname == "M0":
            continue
        deltas = []
        for sym in set(m0) | set(per_sym_auc[mname]):
            a0 = m0.get(sym, np.nan)
            a1 = per_sym_auc[mname].get(sym, np.nan)
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
            eligible_symbols=int((ds["delta_auc"] != 0).sum() or len(ds)),
        ))
sym_df = pd.DataFrame(sym_rows)
sym_df.to_csv(OUT / "metrics_by_symbol.csv", index=False, encoding="utf-8-sig")
print(f"    metrics_by_symbol rows={len(sym_df)}")

print("[M5] block marginal / conditional deltas (聚合 audit) ...")
m0_gain = wf_audit[wf_audit.model == "M0"].set_index(["risk_ATR", "wf"])[
    "bottom20_avoidance_gain"]
m0_auc = wf_audit[wf_audit.model == "M0"].set_index(["risk_ATR", "wf"])["roc_auc"]
marg = wf_audit[wf_audit.model != "M0"].copy()
marg["dROC_vs_M0"] = marg["roc_auc"] - marg.set_index(
    ["risk_ATR", "wf"]).index.map(m0_auc)
marg["dBottom20gain_vs_M0"] = (marg["bottom20_avoidance_gain"]
                               - marg.set_index(["risk_ATR", "wf"]).index.map(m0_gain))
marg_out = marg[["risk_ATR", "wf", "model", "dROC_vs_M0",
                 "dBottom20gain_vs_M0"]].copy()
marg_out.to_csv(OUT / "block_marginal_deltas.csv",
                index=False, encoding="utf-8-sig")

# conditional: Full - Full_minus_block
cond = []
for blk_name, full_minus in [("contact", "M_full_minus_contact"),
                             ("liquidity", "M_full_minus_liquidity"),
                             ("trend", "M_full_minus_trend"),
                             ("ob", "M_full_minus_ob")]:
    f = wf_audit[wf_audit.model == "M_full"].set_index(["risk_ATR", "wf"])
    fm = wf_audit[wf_audit.model == full_minus].set_index(["risk_ATR", "wf"])
    idx = f.index
    cond.append(pd.DataFrame({
        "risk_ATR": [i[0] for i in idx],
        "wf": [i[1] for i in idx],
        "removed_block": blk_name,
        "dROC_Full_minus_block": (f.loc[idx, "roc_auc"].values
                                  - fm.loc[idx, "roc_auc"].values),
        "dBottom20gain_Full_minus_block": (f.loc[idx, "bottom20_avoidance_gain"].values
                                           - fm.loc[idx, "bottom20_avoidance_gain"].values),
    }))
cond_df = pd.concat(cond, ignore_index=True)
cond_df.to_csv(OUT / "block_conditional_deltas.csv",
               index=False, encoding="utf-8-sig")
print(f"    marginal rows={len(marg_out)} conditional rows={len(cond_df)}")

print("[M6] 汇总 CSV（by risk / by wf） + calibration + profiles ...")
model_metrics_by_risk = (wf_audit.groupby(["risk_ATR", "model"])
                         [["roc_auc", "pr_auc", "brier", "logloss",
                           "top20_delivery_rate", "bottom20_delivery_rate",
                           "bottom20_avoidance_gain"]].mean().reset_index())
model_metrics_by_risk.to_csv(OUT / "model_metrics_by_risk.csv",
                             index=False, encoding="utf-8-sig")
model_metrics_by_wf = (wf_audit.groupby(["wf", "model"])
                       [["roc_auc", "pr_auc", "brier", "logloss",
                         "top20_delivery_rate", "bottom20_delivery_rate",
                         "bottom20_avoidance_gain"]].mean().reset_index())
model_metrics_by_wf.to_csv(OUT / "model_metrics_by_walkforward.csv",
                           index=False, encoding="utf-8-sig")

# calibration deciles（full model, 合并所有 risk×wf）
cal_rows = []
for (r, wf_name, yte, day_te, Pmat, base_rate, _) in boot_requests:
    p = Pmat[:, MODEL_NAMES.index("M_full")]
    dec = pd.qcut(p, 10, labels=False, duplicates="drop")
    for d in np.unique(dec):
        m = dec == d
        cal_rows.append(dict(risk_ATR=r, wf=wf_name, decile=int(d),
                             pred_mean=p[m].mean(),
                             actual_rate=yte[m].mean(), n=int(m.sum())))
pd.DataFrame(cal_rows).to_csv(OUT / "calibration_deciles.csv",
                              index=False, encoding="utf-8-sig")

# opportunity profiles: DELIVERY vs NO_DELIVERY 关键连续特征均值（per risk）
prof_rows = []
prof_feats = ["nearest_above_R", "nearest_below_R", "penetration_depth_R",
              "n_targets_L", "n_targets_S", "nearest_opposing_ob_distance_R",
              "nearest_same_direction_ob_distance_R", "atr0"]
for r in oc.RISK_GRID:
    S = F[(F["risk_ATR"] == r) & F["primary_eligible"]]
    d = S[S["y"] == 1]
    nd = S[S["y"] == 0]
    for fcol in prof_feats:
        prof_rows.append(dict(
            risk_ATR=r, feature=fcol,
            mean_DELIVERY=d[fcol].mean(), mean_NO_DELIVERY=nd[fcol].mean(),
            median_DELIVERY=d[fcol].median(),
            median_NO_DELIVERY=nd[fcol].median()))
Path(OUT / "opportunity_profiles").mkdir(exist_ok=True)
pd.DataFrame(prof_rows).to_csv(OUT / "opportunity_profiles/feature_means_by_risk.csv",
                               index=False, encoding="utf-8-sig")
print("[DONE] walk-forward 完成，输出目录:", OUT)
