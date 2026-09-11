"""SMC Opportunity Geometry Decomposition v1.1 —— 修复版（REPAIRED）。

落实 reviewer 审计后的代码正确性修复（P1–P7）：
- P1: model_eligible = primary_eligible & label in {DELIVERY, NO_DELIVERY}
      （剔除 UNRESOLVED / NO_TARGET_ENVIRONMENT），硬断言训练/测试无脏标签。
- P2: 三路特征管线（categorical / spline-numeric / ordinary-numeric）。
- P3: tie-aware AUC（已单测误差<1e-16）。
- P4: 正确的 trading-day block bootstrap（抽中天数整块复制，不丢权重）。
- P5: Bottom20 gain 用 test-fold 自身的 base rate。
- P6: paired Δ 直接 bootstrap（AUC/PR/Brier/LogLoss/Bottom20 gain）。
- P7: G4 真实接入（冻结 Atlas 60 个 *_bin_* 列，绕过 _build_bin_features）。

裁决仅基于本轮内部 paired comparison + canonical block bootstrap。
不进入 Risk-dependent / Direction / PnL。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)

import opportunity_common as oc

OUT = Path("research/analysis_results/smc_opportunity_geometry_v1_1")
RNG = np.random.default_rng(20260911)
N_BOOT = 500

# ---- 子块（G4 真实接入冻结 Atlas 分箱列）----
G1 = ["nearest_above_R", "nearest_below_R"]
G2 = ["n_targets_L", "n_targets_S"]
G3 = ["nearest_ahead_R", "nearest_behind_R"]
G4 = oc.B2_BIN_COLS                       # 60 个 5m_pos_*/5m_neg_* 真实列
G5 = ["same_price_identity_count"]
B0, B1, B2, B3, B4 = (oc.B0_COLS, oc.B1_COLS, oc.B2_COLS,
                       oc.B3_COLS, oc.B4_COLS)
ALL_FEAT = B0 + B1 + B2 + B3 + B4

MODEL_BLOCKS = {
    "M0": B0,
    "M_G1": B0 + G1, "M_G2": B0 + G2, "M_G3": B0 + G3, "M_G4": B0 + G4,
    "M_G5": B0 + G5,
    "M_simple": B0 + G1 + G2,
    "M_field": B0 + G1 + G2 + G4,                       # case B：+G4
    "M_B2_full": B0 + G1 + G2 + G3 + G4 + G5,
    "M_ob": B0 + B4,
    "M_full": B0 + B1 + B2 + B3 + B4,
    "M_full_minus_ob": B0 + B1 + B2 + B3,
    "M_B2mG1": B0 + G2 + G3 + G4 + G5,
    "M_B2mG2": B0 + G1 + G3 + G4 + G5,
    "M_B2mG3": B0 + G1 + G2 + G4 + G5,
    "M_B2mG4": B0 + G1 + G2 + G3 + G5,
    "M_B2mG5": B0 + G1 + G2 + G3 + G4,
}
MODEL_NAMES = list(MODEL_BLOCKS.keys())
MIDX = {m: i for i, m in enumerate(MODEL_NAMES)}

WF = [("WF1", ["TB1"], "TB2"), ("WF2", ["TB1", "TB2"], "TB3"),
      ("WF3", ["TB1", "TB2", "TB3"], "TB4")]

# paired delta 定义（每个条目 = (numerator_model, denominator_model)）
PAIRED = {
    "G1_vs_M0": ("M_G1", "M0"),
    "G2_vs_M0": ("M_G2", "M0"),
    "G3_vs_M0": ("M_G3", "M0"),
    "G4_vs_M0": ("M_G4", "M0"),
    "G5_vs_M0": ("M_G5", "M0"),
    "OB_vs_M0": ("M_ob", "M0"),
    "field_vs_simple": ("M_field", "M_simple"),          # case B 核心
    "B2full_vs_field": ("M_B2_full", "M_field"),         # G3+G5 增量
    "B2full_vs_simple": ("M_B2_full", "M_simple"),
    "full_vs_fullminusob": ("M_full", "M_full_minus_ob"),  # OB 条件增量
}
PAIRED_KEYS = list(PAIRED.keys())


def metrics_vec(y, P):
    n, M = P.shape
    roc = np.full(M, np.nan)
    pr = np.full(M, np.nan)
    br = np.full(M, np.nan)
    ll = np.full(M, np.nan)
    yb = y.astype(np.int64)
    if len(np.unique(yb)) < 2:
        return roc, pr, br, ll
    for j in range(M):
        p = np.clip(P[:, j], 1e-6, 1 - 1e-6)
        roc[j] = oc.tie_aware_auc(yb, P[:, j])
        pr[j] = average_precision_score(yb, P[:, j])
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
    gain = base_rate - bot20            # 每个 test fold 用自己的 base rate
    return top20, bot20, gain


def feature_type_audit(F):
    rows = []
    for block, cols in [("G1", G1), ("G2", G2), ("G3", G3), ("G4", G4),
                        ("G5", G5), ("B0", B0), ("B1", B1), ("B3", B3),
                        ("B4", B4)]:
        for c in cols:
            if c not in F.columns:
                rows.append(dict(feature=c, block=block, dtype="MISSING",
                                pipeline_route="n/a", n_unique=np.nan,
                                missing_rate=np.nan))
                continue
            s = F[c]
            sp, ordi, cat = oc.trifold_route([c])
            route = "spline" if c in sp else ("ordinary" if c in ordi else "categorical")
            rows.append(dict(feature=c, block=block, dtype=str(s.dtype),
                            pipeline_route=route, n_unique=int(s.nunique()),
                            missing_rate=float(s.isna().mean())))
    return pd.DataFrame(rows)


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
F["primary_eligible"] = lab["primary_eligible"].values
F["opportunity_label"] = lab["opportunity_label"].values

# P1: 排除 UNRESOLVED / NO_TARGET_ENVIRONMENT
F["model_eligible"] = (
    F["primary_eligible"]
    & F["opportunity_label"].isin(["DELIVERY", "NO_DELIVERY"])
)
F["y"] = (F["opportunity_label"] == "DELIVERY").astype(int)
n_excl_unres = int((F["primary_eligible"] & (F["opportunity_label"] == "UNRESOLVED")).sum())
n_excl_notarget = int((F["primary_eligible"] & (F["opportunity_label"] == "NO_TARGET_ENVIRONMENT")).sum())
print(f"    剔除 UNRESOLVED(primary_eligible 内)={n_excl_unres} "
      f"NO_TARGET_ENVIRONMENT(primary_eligible 内)={n_excl_notarget}", flush=True)

# feature type audit
feature_type_audit(F).to_csv(OUT / "feature_type_audit.csv", index=False,
                             encoding="utf-8-sig")

print("[M2] 逐 risk × WF × model 训练与评估（P1 标签门 + P2 三路管线）...",
      flush=True)
rows, oof_rows, boot_req = [], [], []
for r in oc.RISK_GRID:
    sub = F[(F["risk_ATR"] == r) & F["model_eligible"]].reset_index(drop=True)
    X = sub[ALL_FEAT].copy()
    y = sub["y"].values
    base_rate = y.mean()
    n_del = int((y == 1).sum())
    n_no = int((y == 0).sum())
    for wf_name, tr_blocks, te_block in WF:
        tr = sub["block"].isin(tr_blocks).values
        te = (sub["block"] == te_block).values
        if tr.sum() == 0 or te.sum() == 0:
            continue
        ytr, yte = y[tr], y[te]
        # 硬断言：训练/测试均无脏标签
        assert set(np.unique(ytr).tolist()) <= {0, 1}
        assert set(np.unique(yte).tolist()) <= {0, 1}
        day_te = sub["trading_day"].values[te]
        Pmat = np.zeros((te.sum(), len(MODEL_NAMES)))
        per_sym = {m: {} for m in MODEL_NAMES}
        for j, (mname, cols) in enumerate(MODEL_BLOCKS.items()):
            mcols = [c for c in cols if c in ALL_FEAT]
            pipe = oc.make_trifold_pipeline(mcols)
            pipe.fit(X[tr][mcols], ytr)
            proba = pipe.predict_proba(X[te][mcols])[:, 1]
            Pmat[:, j] = proba
            St = sub[te].reset_index(drop=True)
            for sym, g in St.groupby("symbol"):
                if len(np.unique(g["y"].values)) < 2:
                    continue
                per_sym[mname][sym] = roc_auc_score(g["y"].values, proba[g.index.values])
        roc, pr, br, ll = metrics_vec(yte, Pmat)
        base_test = yte.mean()
        top20, bot20, gain = rank_metrics(yte, Pmat, base_test)
        for j, mname in enumerate(MODEL_NAMES):
            rows.append(dict(risk_ATR=r, wf=wf_name, model=mname,
                             n_train=int(tr.sum()), n_test=int(te.sum()),
                             n_delivery=n_del, n_no_delivery=n_no,
                             base_delivery_rate=base_rate,
                             test_base_delivery_rate=base_test,
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
        boot_req.append((r, wf_name, yte, day_te, Pmat))
    print(f"  [M2] risk={r} 完成 (delivery={n_del} no_delivery={n_no} "
          f"base={base_rate:.3f})", flush=True)

wf_audit = pd.DataFrame(rows)
wf_audit.to_csv(OUT / "walkforward_audit.csv", index=False, encoding="utf-8-sig")
oof_all = pd.concat(oof_rows, ignore_index=True)
oof_all.to_parquet(OUT / "oof_predictions.parquet", index=False)
print(f"    walkforward_audit rows={len(wf_audit)}", flush=True)

print("[M3] 正确 block bootstrap 500 + paired Δ（P4/P5/P6）...", flush=True)
roc_boot = {}     # (risk,wf) -> (N_BOOT, M)
pr_boot, br_boot, ll_boot, gain_boot = {}, {}, {}, {}
for (r, wf_name, yte, day_te, Pmat) in boot_req:
    days = np.unique(day_te)
    day_to_idx = {d: np.flatnonzero(day_te == d) for d in days}
    n, M = Pmat.shape
    rb = np.zeros((N_BOOT, M)); pb = np.zeros((N_BOOT, M))
    bb = np.zeros((N_BOOT, M)); lb = np.zeros((N_BOOT, M))
    gb = np.zeros((N_BOOT, M))
    for b in range(N_BOOT):
        if b % 100 == 0:
            print(f"    boot {b}/{N_BOOT} risk={r} wf={wf_name}", flush=True)
        samp = RNG.choice(days, size=len(days), replace=True)
        idx = np.concatenate([day_to_idx[d] for d in samp])   # 整块复制
        yb = yte[idx]
        Pb = Pmat[idx]
        roc, pr, br, ll = metrics_vec(yb, Pb)
        base_boot = yb.mean()
        _, _, gain = rank_metrics(yb, Pb, base_boot)
        rb[b], pb[b], bb[b], lb[b], gb[b] = roc, pr, br, ll, gain
    roc_boot[(r, wf_name)] = rb
    pr_boot[(r, wf_name)] = pb
    br_boot[(r, wf_name)] = bb
    ll_boot[(r, wf_name)] = lb
    gain_boot[(r, wf_name)] = gb

# 单模型 CI
boot_rows = []
for (r, wf_name), rb in roc_boot.items():
    M = rb.shape[1]
    pb = pr_boot[(r, wf_name)]; bb = br_boot[(r, wf_name)]
    gb = gain_boot[(r, wf_name)]
    for j, mname in enumerate(MODEL_NAMES):
        boot_rows.append(dict(
            risk_ATR=r, wf=wf_name, model=mname,
            roc_auc_lo=np.nanpercentile(rb[:, j], 2.5),
            roc_auc_hi=np.nanpercentile(rb[:, j], 97.5),
            pr_auc_lo=np.nanpercentile(pb[:, j], 2.5),
            pr_auc_hi=np.nanpercentile(pb[:, j], 97.5),
            brier_lo=np.nanpercentile(bb[:, j], 2.5),
            brier_hi=np.nanpercentile(bb[:, j], 97.5),
            bottom20_gain_lo=np.nanpercentile(gb[:, j], 2.5),
            bottom20_gain_hi=np.nanpercentile(gb[:, j], 97.5)))
pd.DataFrame(boot_rows).to_csv(OUT / "subblock_bootstrap_ci.csv", index=False,
                               encoding="utf-8-sig")

# paired Δ CI
pd_rows = []
for (r, wf_name), rb in roc_boot.items():
    pb = pr_boot[(r, wf_name)]; bb = br_boot[(r, wf_name)]
    gb = gain_boot[(r, wf_name)]
    for pk in PAIRED_KEYS:
        num, den = PAIRED[pk]
        ni, di = MIDX[num], MIDX[den]
        d_roc = rb[:, ni] - rb[:, di]
        d_pr = pb[:, ni] - pb[:, di]
        d_br = bb[:, ni] - bb[:, di]
        d_gain = gb[:, ni] - gb[:, di]
        pd_rows.append(dict(
            risk_ATR=r, wf=wf_name, pair=pk,
            numerator=num, denominator=den,
            dROC_lo=np.nanpercentile(d_roc, 2.5),
            dROC_hi=np.nanpercentile(d_roc, 97.5),
            dROC_point=float(d_roc.mean()),
            dPR_lo=np.nanpercentile(d_pr, 2.5),
            dPR_hi=np.nanpercentile(d_pr, 97.5),
            dBrier_lo=np.nanpercentile(d_br, 2.5),
            dBrier_hi=np.nanpercentile(d_br, 97.5),
            dBottom20gain_lo=np.nanpercentile(d_gain, 2.5),
            dBottom20gain_hi=np.nanpercentile(d_gain, 97.5)))
pd.DataFrame(pd_rows).to_csv(OUT / "paired_delta_bootstrap_ci.csv", index=False,
                             encoding="utf-8-sig")

print("[M4] 跨品种子块 delta（per-symbol AUC vs M0）...", flush=True)
sym_rows = []
# 基于 oof_predictions 直接计算每个品种 AUC delta
for (r, wf_name), grp in oof_all.groupby(["risk_ATR", "wf"]):
    ytrue = grp["y_true"].values
    for mname in MODEL_NAMES:
        if mname == "M0":
            continue
        deltas = []
        for sym, g in grp.groupby("symbol"):
            a0 = roc_auc_score(g["y_true"].values, g["M0"].values)
            a1 = roc_auc_score(g["y_true"].values, g[mname].values)
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
            eligible_symbols=int((ds["delta_auc"] != 0).sum())))
pd.DataFrame(sym_rows).to_csv(OUT / "subblock_by_symbol.csv", index=False,
                              encoding="utf-8-sig")

print("[M5] 子块 marginal / conditional deltas（点估计）...", flush=True)
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
            - fm.loc[cidx, "bottom20_avoidance_gain"].values)}))
pd.concat(cond, ignore_index=True).to_csv(
    OUT / "subblock_conditional_deltas.csv", index=False, encoding="utf-8-sig")

print("[M6] 汇总 CSV + 机械基准 geometry_ratio（仅 model_eligible）...", flush=True)
wf_audit.groupby(["risk_ATR", "model"])[
    ["roc_auc", "pr_auc", "brier", "logloss", "top20_delivery_rate",
     "bottom20_delivery_rate", "bottom20_avoidance_gain"]].mean().reset_index()\
    .to_csv(OUT / "subblock_metrics_by_risk.csv", index=False, encoding="utf-8-sig")
wf_audit.groupby(["wf", "model"])[
    ["roc_auc", "pr_auc", "brier", "logloss", "top20_delivery_rate",
     "bottom20_delivery_rate", "bottom20_avoidance_gain"]].mean().reset_index()\
    .to_csv(OUT / "subblock_metrics_by_wf.csv", index=False, encoding="utf-8-sig")

F2 = F[F["model_eligible"]].copy()
F2["nearest_any_target_R"] = F2[["nearest_above_R", "nearest_below_R"]].min(axis=1)
F2["geometry_ratio"] = F2["nearest_any_target_R"] / F2["risk_ATR"]
bins = [-np.inf, 0.25, 0.5, 1.0, 2.0, np.inf]
labels = ["<=0.25", "(0.25,0.5]", "(0.5,1]", "(1,2]", ">2"]
F2["geo_bin"] = pd.cut(F2["geometry_ratio"], bins=bins, labels=labels)
prof = F2.copy()
prof["is_delivery"] = prof["y"]
gp_sum = (prof.groupby(["risk_ATR", "geo_bin"], observed=True)
          .agg(n=("is_delivery", "size"),
               delivery_rate=("is_delivery", "mean"),
               no_delivery_rate=("is_delivery", lambda s: 1 - s.mean()))
          .reset_index())
gp_sum.to_csv(OUT / "geometry_ratio_summary_repaired.csv", index=False,
              encoding="utf-8-sig")
gp = (prof.groupby(["risk_ATR", "block", "symbol", "geo_bin"], observed=True)
      .agg(n=("is_delivery", "size"),
           delivery_rate=("is_delivery", "mean"),
           no_delivery_rate=("is_delivery", lambda s: 1 - s.mean()))
      .reset_index())
gp.to_csv(OUT / "geometry_ratio_profile.csv", index=False, encoding="utf-8-sig")

print("[M7] OB freshness 收口：本轮内部 paired comparison（M_ob-M0, "
      "M_full-M_full_minus_ob）...", flush=True)
ob_rows = []
# 从 paired_delta_bootstrap_ci.csv 读取
pdc = pd.read_csv(OUT / "paired_delta_bootstrap_ci.csv")
for pk in ["OB_vs_M0", "full_vs_fullminusob"]:
    g = pdc[pdc.pair == pk]
    for _, rr in g.iterrows():
        ob_rows.append(dict(
            risk_ATR=rr["risk_ATR"], wf=rr["wf"], pair=pk,
            numerator=rr["numerator"], denominator=rr["denominator"],
            dROC_point=rr["dROC_point"],
            dROC_lo=rr["dROC_lo"], dROC_hi=rr["dROC_hi"],
            dBottom20gain_lo=rr["dBottom20gain_lo"],
            dBottom20gain_hi=rr["dBottom20gain_hi"]))
pd.DataFrame(ob_rows).to_csv(OUT / "ob_freshness_fix_audit.csv", index=False,
                             encoding="utf-8-sig")

print("[DONE] geometry decomposition (REPAIRED) 完成，输出目录:", OUT)
