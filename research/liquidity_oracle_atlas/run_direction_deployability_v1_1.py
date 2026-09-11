"""SMC Direction Deployability & Liquidity Map Decomposition v1.1

base: e7ee079 (v1.0).  修复 v1.0 的可复现性与指标 bug，并把方向信号
从"可行性发现"升级为"可部署性 + liquidity map 真正分解"确认实验。

修复（P0）：
- reversal_label 对 non-clear 返回 NaN（删 r1 重复 merge side）。
- by-symbol balanced_accuracy 改用 sklearn balanced_accuracy_score。
- Gate Top20 删除（不再是 L_full 全样本 accuracy）；改用 P7 两尾 macro_tail。
- direction_feature_block_deltas 逻辑写回正式脚本；新增 REPRODUCIBILITY_AUDIT.json。
- enum 记录 liquidity_type = 10。

新增分解（P1-P3）：M_ID / M_GLOBAL2 / M_GLOBAL4 / M_SCOPE12 / M_GLOBAL_SCOPE /
M_CORE / M_TYPE20 / M_GLOBAL_TYPE / M_IDENTITY_MAP / M_MAP_CORE；非线性只跑
N_GLOBAL4 / N_IDENTITY_MAP / N_MAP_CORE（固定 HGB）。

确认（P4-P13）：interaction 增量、paired trading-day bootstrap、两尾 confidence、
clear-direction gate、联合可部署选择器、ROBUST diagnostic、risk coupling 0.5/1.0/2.0。

Governance: TRADING_METRICS=NOT_APPLICABLE；无 PnL/止损/最优risk/FVG/external-internal/
LC。Atlas v1.2 冻结不动。大型 parquet 不入 Git。
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score, balanced_accuracy_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, SplineTransformer,
                                   StandardScaler)
from sklearn.ensemble import HistGradientBoostingClassifier

import opportunity_common as oc
from research.liquidity_oracle_atlas.build_oracle_atlas_v1_2 import (
    active_mask, SCOPES)

ATLAS = Path("research/analysis_results/smc_oracle_atlas_v1")
OUT = Path("research/analysis_results/smc_direction_deployability_v1_1")
OUT.mkdir(parents=True, exist_ok=True)
SYMBOLS = ["AG", "AU", "CU", "AL", "SN", "NI", "RB", "I", "SC", "RU", "MA",
           "TA", "M", "P", "CF"]
PRIMARY_RISK = 1.0
OOS_START = "2026-09-07"
BLOCKS = ["TB1", "TB2", "TB3", "TB4"]
WF = [("WF1", ["TB1"], ["TB2"]),
      ("WF2", ["TB1", "TB2"], ["TB3"]),
      ("WF3", ["TB1", "TB2", "TB3"], ["TB4"])]

FORBIDDEN = {
    "rr_direction", "direction_stability", "best_R_lower", "best_R_upper",
    "resolution_class", "status", "path_censor", "critical_risk",
    "required_risk_before_target_ATR", "required_risk_through_target_ATR",
    "required_before_ATR", "required_through_ATR", "bars_to_best_lower",
    "bars_to_best_target", "best_target_price", "liquidity_price",
    "entry_reference", "decision_time", "contact_bar_index", "trading_day",
    "block", "y_reversal", "y_long", "y_clear", "y_rev_robust",
    "y_rev_r05", "y_rev_r20",
}


def reversal_label(side, rr):
    """P0.1：上方 liquidity reversal=SHORT；下方 reversal=LONG；non-clear→NaN。"""
    if rr not in ("LONG_DOMINATES", "SHORT_DOMINATES"):
        return np.nan
    is_long = (rr == "LONG_DOMINATES")
    if side == +1:
        return int(not is_long)
    if side == -1:
        return int(is_long)
    raise ValueError(side)


# ===========================================================================
# Stage 1：master 派生（contact_cluster_* + 12 per-scope + 20 per-type）
# ===========================================================================
def build_symbol_master_features(sym, master, contacts, scopes, liq_types):
    ms = master[master["symbol"] == sym]
    mp = ms["price"].to_numpy(float)
    mscope = ms["liquidity_scope"].astype(str).to_numpy()
    mtype = ms["liquidity_type"].astype(str).to_numpy()
    mav = pd.to_datetime(ms["available_time"]).to_numpy(np.datetime64)
    mfp = pd.to_datetime(ms["first_penetration_time"]).to_numpy(np.datetime64)

    def sort_struct(mask):
        sp = mp[mask]
        order = np.argsort(sp, kind="mergesort")
        return dict(price=sp[order], av=mav[mask][order], fp=mfp[mask][order])

    scope_struct = {s: sort_struct(mscope == s) for s in scopes}
    type_struct = {t: sort_struct(mtype == t) for t in liq_types}
    same = defaultdict(list)
    for i, p in enumerate(mp):
        same[p].append(i)

    cc = contacts[contacts["symbol"] == sym]
    rows = []
    for r in cc.itertuples(index=False):
        dt = np.datetime64(pd.Timestamp(r.decision_time))
        a0 = float(r.atr0)
        if not (np.isfinite(a0) and a0 > 0):
            a0 = np.nan
        entry = float(r.entry_reference)
        lvl = float(r.liquidity_price)
        rec = dict(symbol=sym, liquidity_id=r.liquidity_id,
                   contact_number=r.contact_number,
                   contact_cluster_identity_count=0,
                   contact_cluster_scope_count=0,
                   contact_cluster_type_count=0, contact_cluster_types="")
        for s in scopes:
            rec[f"contact_cluster_has_{s}"] = False

        # contact_cluster_*（pre-contact same-price，user mask: fp>=dt|isnan）
        same_list = same.get(lvl, [])
        sset, tset = set(), set()
        idc = 0
        hasd = {s: False for s in scopes}
        for i in same_list:
            if mav[i] <= dt and (np.isnat(mfp[i]) or mfp[i] >= dt):
                idc += 1
                sset.add(mscope[i])
                tset.add(mtype[i])
                hasd[mscope[i]] = True
        rec["contact_cluster_identity_count"] = idc
        rec["contact_cluster_scope_count"] = len(sset)
        rec["contact_cluster_type_count"] = len(tset)
        rec["contact_cluster_types"] = "|".join(sorted(tset))
        for s in scopes:
            rec[f"contact_cluster_has_{s}"] = bool(hasd[s])

        # per-scope map（active_mask: av<=dt & (fp>dt|isnan)）
        for s in scopes:
            ss = scope_struct[s]
            pr, avv, fpp, n = ss["price"], ss["av"], ss["fp"], len(ss["price"])
            da = np.nan
            k = int(np.searchsorted(pr, entry, side="right"))
            for t in range(k, n):
                if avv[t] <= dt and (np.isnat(fpp[t]) or fpp[t] > dt):
                    da = (pr[t] - entry) / a0
                    break
            db = np.nan
            k2 = int(np.searchsorted(pr, entry, side="left")) - 1
            for t in range(k2, -1, -1):
                if avv[t] <= dt and (np.isnat(fpp[t]) or fpp[t] > dt):
                    db = (entry - pr[t]) / a0
                    break
            rec[f"nearest_above_{s}_R"] = da
            rec[f"nearest_below_{s}_R"] = db
            rec[f"has_above_{s}"] = int(np.isfinite(da))
            rec[f"has_below_{s}"] = int(np.isfinite(db))

        # per-type map（同样 active_mask）
        for t in liq_types:
            ss = type_struct[t]
            pr, avv, fpp, n = ss["price"], ss["av"], ss["fp"], len(ss["price"])
            da = np.nan
            k = int(np.searchsorted(pr, entry, side="right"))
            for u in range(k, n):
                if avv[u] <= dt and (np.isnat(fpp[u]) or fpp[u] > dt):
                    da = (pr[u] - entry) / a0
                    break
            db = np.nan
            k2 = int(np.searchsorted(pr, entry, side="left")) - 1
            for u in range(k2, -1, -1):
                if avv[u] <= dt and (np.isnat(fpp[u]) or fpp[u] > dt):
                    db = (entry - pr[u]) / a0
                    break
            rec[f"nearest_above_type_{t}_R"] = da
            rec[f"nearest_below_type_{t}_R"] = db
            rec[f"has_above_type_{t}"] = int(np.isfinite(da))
            rec[f"has_below_type_{t}"] = int(np.isfinite(db))
        rows.append(rec)
    return pd.DataFrame(rows)


def build_features(liq_types):
    MASTER = pd.read_parquet(ATLAS / "liquidity_master_v1_1.parquet")
    CON = pd.read_parquet(ATLAS / "liquidity_contacts_v1_1.parquet")
    ST = pd.read_parquet(ATLAS / "liquidity_state_snapshot_v1_2.parquet")
    rd = pd.read_parquet(ATLAS / "oracle_risk_direction_v1_2.parquet")

    keys = ["symbol", "liquidity_id", "contact_number"]
    # ---- labels ----
    r1 = rd[np.isclose(rd["risk_ATR"], PRIMARY_RISK)].copy()
    r1 = r1.merge(CON[keys + ["side"]].drop_duplicates(),
                  on=keys, how="left")
    r1["y_reversal"] = [reversal_label(s, rr) for s, rr in
                        zip(r1["side"], r1["rr_direction"])]
    r1["y_clear"] = np.where(
        r1["rr_direction"].isin(["LONG_DOMINATES", "SHORT_DOMINATES"]), 1,
        np.where(r1["rr_direction"] == "TRADEOFF_OR_OVERLAP", 0, np.nan))

    # robust derivation（确定性，无新数据）：各 risk 档 clear 方向一致即 ROBUST
    piv = rd.pivot_table(index=keys, columns="risk_ATR",
                         values="rr_direction", aggfunc="first")

    def robust_dir(row):
        ds = []
        for v in row.values:
            if v == "LONG_DOMINATES":
                ds.append(1)
            elif v == "SHORT_DOMINATES":
                ds.append(-1)
        return ds[0] if ds and len(set(ds)) == 1 else 0
    rdir = piv.apply(robust_dir, axis=1)
    rdir_df = rdir.reset_index()
    rdir_df.columns = keys + ["rdir"]
    rdir_df = rdir_df.merge(CON[keys + ["side"]].drop_duplicates(),
                            on=keys, how="left")
    rdir_df["y_rev_robust"] = [
        reversal_label(s, "LONG_DOMINATES" if d == 1 else "SHORT_DOMINATES")
        if d != 0 else np.nan
        for s, d in zip(rdir_df["side"], rdir_df["rdir"])]

    def risk_label_df(risk):
        sub = rd[np.isclose(rd["risk_ATR"], risk)].copy()
        sub = sub.merge(CON[keys + ["side"]].drop_duplicates(),
                        on=keys, how="left")
        sub["yl"] = [reversal_label(s, rr) for s, rr in
                     zip(sub["side"], sub["rr_direction"])]
        return sub[keys + ["yl"]].rename(columns={"yl": f"y_rev_r{int(risk*100)}"})
    r05 = risk_label_df(0.5).rename(columns={"y_rev_r50": "y_rev_r05"})
    r20 = risk_label_df(2.0).rename(columns={"y_rev_r200": "y_rev_r20"})

    # master 派生
    mf = []
    for sym in SYMBOLS:
        mf.append(build_symbol_master_features(sym, MASTER, CON, list(SCOPES),
                                               liq_types))
    MFEAT = pd.concat(mf, ignore_index=True)

    con_meta = CON[keys + ["liquidity_type", "is_first_contact",
                           "bars_since_available", "bars_since_previous_contact",
                           "bar_range_R", "abs_return_R"]].copy()

    F = ST.merge(con_meta, on=keys, how="left")
    F = F.merge(MFEAT, on=keys, how="left")
    F = F.merge(r1[keys + ["rr_direction", "y_reversal", "y_clear"]],
                on=keys, how="left")
    F = F.merge(rdir_df[keys + ["y_rev_robust"]], on=keys, how="left")
    F = F.merge(r05, on=keys, how="left")
    F = F.merge(r20, on=keys, how="left")
    blk = oc.attach_trading_day_block(F[keys].copy())
    F = F.merge(blk, on=keys, how="left")
    F.to_parquet(OUT / "direction_features_v1_1.parquet", index=False)
    return F, r1, rdir_df, rd


# ===========================================================================
# Stage 3：特征块 + 路由
# ===========================================================================
SPLINE_NUMERIC = set()
ORDINARY_NUMERIC = set()
CATEGORICAL = set()
FEATURE_COLS = []
MODELS = []


def define_blocks(liq_types):
    global SPLINE_NUMERIC, ORDINARY_NUMERIC, CATEGORICAL, FEATURE_COLS
    scopes = list(SCOPES)
    dist = (["nearest_above_R", "nearest_below_R", "nearest_ahead_R",
             "nearest_behind_R"]
            + [f"nearest_above_{s}_R" for s in scopes]
            + [f"nearest_below_{s}_R" for s in scopes]
            + [f"nearest_above_type_{t}_R" for t in liq_types]
            + [f"nearest_below_type_{t}_R" for t in liq_types]
            + ["penetration_depth_R", "close_relative_to_level_R",
               "bar_range_R", "abs_return_R", "atr0"])
    SPLINE_NUMERIC = set(dist)
    ORDINARY_NUMERIC = set(
        ["contact_number", "bars_since_available", "bars_since_previous_contact",
         "contact_cluster_identity_count", "contact_cluster_scope_count",
         "contact_cluster_type_count", "same_price_identity_count",
         "same_price_identity_count_v11", "active_visible_count"]
        + [f"contact_cluster_has_{s}" for s in scopes]
        + [f"has_above_{s}" for s in scopes]
        + [f"has_below_{s}" for s in scopes]
        + [f"has_above_type_{t}" for t in liq_types]
        + [f"has_below_type_{t}" for t in liq_types])
    CATEGORICAL = set(
        ["symbol", "side", "liquidity_type", "liquidity_scope", "contact_type",
         "is_first_contact"])
    FEATURE_COLS = sorted(set(
        list(CATEGORICAL) + list(SPLINE_NUMERIC) + list(ORDINARY_NUMERIC)))


def block_cols(name, liq_types):
    scopes = list(SCOPES)
    D0 = ["symbol", "side"]
    D1 = D0 + ["liquidity_type", "liquidity_scope", "contact_number",
               "is_first_contact", "bars_since_available",
               "bars_since_previous_contact",
               "contact_cluster_identity_count", "contact_cluster_scope_count",
               "contact_cluster_type_count",
               "contact_cluster_has_5m", "contact_cluster_has_15m",
               "contact_cluster_has_1h", "contact_cluster_has_CONTIG_SESSION",
               "contact_cluster_has_TRADING_DAY",
               "contact_cluster_has_TRADING_WEEK"]
    GLOBAL2 = ["nearest_above_R", "nearest_below_R"]
    GLOBAL4 = GLOBAL2 + ["nearest_ahead_R", "nearest_behind_R"]
    SCOPE12 = ([f"nearest_above_{s}_R" for s in scopes]
               + [f"nearest_below_{s}_R" for s in scopes]
               + [f"has_above_{s}" for s in scopes]
               + [f"has_below_{s}" for s in scopes])
    TYPE20 = ([f"nearest_above_type_{t}_R" for t in liq_types]
              + [f"nearest_below_type_{t}_R" for t in liq_types]
              + [f"has_above_type_{t}" for t in liq_types]
              + [f"has_below_type_{t}" for t in liq_types])
    D3 = ["contact_type", "penetration_depth_R", "close_relative_to_level_R",
          "bar_range_R", "abs_return_R"]
    M = {
        "M_ID": D1,
        "M_GLOBAL2": D1 + GLOBAL2,
        "M_GLOBAL4": D1 + GLOBAL4,
        "M_SCOPE12": D1 + SCOPE12,
        "M_GLOBAL_SCOPE": D1 + GLOBAL4 + SCOPE12,
        "M_CORE": D1 + GLOBAL4 + SCOPE12 + D3,
        "M_TYPE20": D1 + TYPE20,
        "M_GLOBAL_TYPE": D1 + GLOBAL4 + TYPE20,
        "M_IDENTITY_MAP": D1 + GLOBAL4 + SCOPE12 + TYPE20,
        "M_MAP_CORE": D1 + GLOBAL4 + SCOPE12 + TYPE20 + D3,
    }
    return M[name]


def make_preprocessor(cols):
    sp = [c for c in cols if c in SPLINE_NUMERIC]
    ordi = [c for c in cols if c in ORDINARY_NUMERIC]
    cat = [c for c in cols if c in CATEGORICAL]
    miss = [c for c in cols if c not in SPLINE_NUMERIC
            and c not in ORDINARY_NUMERIC and c not in CATEGORICAL]
    assert not miss, f"unrouted cols: {miss}"
    return ColumnTransformer([
        ("spline", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sp", SplineTransformer(n_knots=4, degree=2, knots="quantile",
                                    include_bias=False)),
            ("sc", StandardScaler())]), sp),
        ("ord", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())]), ordi),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ], remainder="drop")


def fit_predict(cols, Xtr, ytr, Xte, mode="logistic"):
    pre = make_preprocessor(cols)
    if mode == "logistic":
        clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                 max_iter=3000)
    else:
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                            max_iter=200, l2_regularization=1.0,
                                            random_state=42)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xte)[:, 1]


def abs_dir_acc(p, side, rr):
    pred_rev = (p >= 0.5).astype(int)
    pred_long = np.where(side == +1, 1 - pred_rev, pred_rev)
    actual_long = (rr == "LONG_DOMINATES").astype(int)
    return float((pred_long == actual_long).mean())


def metrics(y, p, side=None, rr=None):
    out = dict(
        roc_auc=float(roc_auc_score(y, p)) if len(set(y)) > 1 else np.nan,
        pr_auc=float(average_precision_score(y, p)) if len(set(y)) > 1 else np.nan,
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        brier=float(brier_score_loss(y, p)),
    )
    pred = (p >= 0.5).astype(int)
    out["accuracy"] = float((pred == y).mean())
    out["balanced_accuracy"] = float(balanced_accuracy_score(y, pred))
    if side is not None and rr is not None:
        out["abs_direction_accuracy"] = abs_dir_acc(p, side, rr)
    return out


def expanding_oof_pred(cols, X_tr, y_tr, days_tr, mode, sub=None):
    """outer train 内 3 段 expanding OOF；sub=cohort mask over train。"""
    if sub is None:
        sub = np.ones(len(y_tr), bool)
    uniq = np.sort(pd.unique(days_tr))
    n = len(uniq)
    oidx, op = [], []
    for frac in (0.4, 0.6, 0.8):
        k = int(round(n * frac))
        if k >= n:
            continue
        tr_days = uniq[:k]
        oo_days = uniq[k:k + max(1, int(round(n * 0.2)))]
        if len(oo_days) == 0:
            continue
        m_tr = (pd.Series(days_tr).isin(tr_days).to_numpy() & sub)
        m_oo = (pd.Series(days_tr).isin(oo_days).to_numpy() & sub)
        if m_tr.sum() < 50 or m_oo.sum() < 20:
            continue
        p = fit_predict(cols, X_tr[m_tr], y_tr[m_tr], X_tr[m_oo], mode)
        oidx.append(np.flatnonzero(m_oo))
        op.append(p)
    if not op:
        return np.array([], int), np.array([])
    return np.concatenate(oidx), np.concatenate(op)


def paired_day_bootstrap(y, pred_a, pred_b, day, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    days = np.array(sorted(pd.unique(day)))
    idx_by_day = {d: np.flatnonzero(day == d) for d in days}
    vals = []
    for _ in range(n_boot):
        draw = rng.choice(days, size=len(days), replace=True)
        idx = np.concatenate([idx_by_day[d] for d in draw])
        if np.unique(y[idx]).size < 2:
            continue
        try:
            a = roc_auc_score(y[idx], pred_a[idx])
            b = roc_auc_score(y[idx], pred_b[idx])
        except Exception:
            continue
        vals.append(b - a)
    if not vals:
        return (np.nan, np.nan, np.nan)
    return tuple(np.quantile(vals, [0.025, 0.5, 0.975]))


def main():
    t0 = time.perf_counter()
    MASTER = pd.read_parquet(ATLAS / "liquidity_master_v1_1.parquet")
    LIQ_TYPES = sorted(MASTER["liquidity_type"].dropna().unique().tolist())
    assert len(LIQ_TYPES) == 10, f"liquidity_type count={len(LIQ_TYPES)}"
    F, r1, rdir_df, rd = build_features(LIQ_TYPES)
    CON = pd.read_parquet(ATLAS / "liquidity_contacts_v1_1.parquet")
    print(f"[S1] features built {F.shape} ({time.perf_counter()-t0:.1f}s); "
          f"liq_types={len(LIQ_TYPES)}")

    F["trading_day"] = F["trading_day"].astype(str)
    insample = F["trading_day"] < OOS_START

    # --- 泄漏断言 ---
    X_all = F.drop(columns=[c for c in FORBIDDEN if c in F.columns]).copy()
    bad = set(X_all.columns) & FORBIDDEN
    assert not bad, f"leakage in X: {bad}"

    define_blocks(LIQ_TYPES)
    X = X_all[FEATURE_COLS].copy()
    for c in X.columns:
        if c in CATEGORICAL:
            continue
        X[c] = pd.to_numeric(X[c], errors="coerce")
    print(f"[S1] feature matrix {X.shape}")

    keys = ["symbol", "liquidity_id", "contact_number"]
    # ---- 审计 ----
    enum = {c: CON[c].value_counts(dropna=False).to_dict()
            for c in ["liquidity_type", "liquidity_scope", "contact_type",
                      "side", "is_first_contact"]}
    json.dump(enum, open(OUT / "liquidity_identity_enum.json", "w"),
              indent=2, ensure_ascii=False)
    # label audit (risk=1.0)
    a = r1.merge(F[keys + ["block"]], on=keys, how="left")
    la_rows = []
    for (sym, tb), g in a.groupby(["symbol", "block"]):
        el = g[~g["y_reversal"].isna()]
        la_rows.append(dict(symbol=sym, TB=tb, risk_ATR=PRIMARY_RISK, n=len(g),
                           LONG_n=int((g.rr_direction == "LONG_DOMINATES").sum()),
                           SHORT_n=int((g.rr_direction == "SHORT_DOMINATES").sum()),
                           TRADEOFF_n=int((g.rr_direction == "TRADEOFF_OR_OVERLAP").sum()),
                           CENSOR_n=int((g.rr_direction == "UNRESOLVED_CENSOR").sum()),
                           NOCOMP_n=int((g.rr_direction == "NO_COMPARABLE_TARGET").sum()),
                           REVERSAL_n=int((el.y_reversal == 1).sum()),
                           CONTINUATION_n=int((el.y_reversal == 0).sum())))
    pd.DataFrame(la_rows).to_csv(OUT / "direction_label_audit.csv", index=False,
                                 encoding="utf-8-sig")
    # robust manifest
    json.dump(dict(liquidity_scope=list(SCOPES),
                   liquidity_type=LIQ_TYPES,
                   n_liquidity_type=len(LIQ_TYPES),
                   type_column_map={t: {"above": f"nearest_above_type_{t}_R",
                                        "below": f"nearest_below_type_{t}_R",
                                        "has_above": f"has_above_type_{t}",
                                        "has_below": f"has_below_type_{t}"}
                                   for t in LIQ_TYPES}),
              open(OUT / "liquidity_scope_type_manifest.json", "w"),
              indent=2, ensure_ascii=False)
    print("[S2] audit done")

    # ---- 标签与 mask ----
    y_rev = F["y_reversal"].to_numpy()
    side = F["side"].to_numpy()
    rr = F["rr_direction"].to_numpy()
    block = F["block"].to_numpy()
    days = F["trading_day"].to_numpy()
    elig = (~pd.isna(y_rev)) & insample
    y_clear = F["y_clear"].to_numpy()
    elig_clear = (~pd.isna(y_clear)) & insample
    y_robust = F["y_rev_robust"].to_numpy()
    elig_robust = (~pd.isna(y_robust)) & insample
    y_r05 = F["y_rev_r05"].to_numpy()
    y_r20 = F["y_rev_r20"].to_numpy()

    # ---- 模型列定义 ----
    DIR_MODELS = ["M_ID", "M_GLOBAL2", "M_GLOBAL4", "M_SCOPE12",
                  "M_GLOBAL_SCOPE", "M_CORE", "M_TYPE20", "M_GLOBAL_TYPE",
                  "M_IDENTITY_MAP", "M_MAP_CORE"]
    N_MODEL_COLS = {"N_GLOBAL4": "M_GLOBAL4",
                    "N_IDENTITY_MAP": "M_IDENTITY_MAP",
                    "N_MAP_CORE": "M_MAP_CORE"}
    CLEAR_MODELS = {"C_GLOBAL4": "M_GLOBAL4",
                    "C_IDENTITY_MAP": "M_IDENTITY_MAP",
                    "C_MAP_CORE": "M_MAP_CORE"}
    ROBUST_MODELS = ["M_GLOBAL4", "M_IDENTITY_MAP", "M_MAP_CORE"]
    RISK_MODELS = ["M_MAP_CORE"]

    map_metrics_rows = []
    bysym_rows = []
    preds_store = {}      # (wf, model) -> test pred array (aligned to dir test rows)
    paired_rows = []
    conf_rows = []
    clear_rows = []
    sel_rows = []
    robust_rows = []
    risk_rows = []

    for wf_name, tr_blocks, te_blocks in WF:
        m_tr = pd.Series(block).isin(tr_blocks).to_numpy()
        m_te = pd.Series(block).isin(te_blocks).to_numpy()
        # ---- 方向分解（eligible clear & in-sample）----
        tr = m_tr & elig
        te = m_te & elig
        if len(set(y_rev[tr])) < 2 or len(set(y_rev[te])) < 2:
            continue
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y_rev[tr], y_rev[te]
        day_te = days[te]
        preds = {}
        for m in DIR_MODELS:
            cols = [c for c in block_cols(m, LIQ_TYPES) if c in X.columns]
            p = fit_predict(cols, Xtr, ytr, Xte, "logistic")
            preds[m] = p
            preds_store[(wf_name, m)] = p
            mm = metrics(yte, p, side[te], rr[te])
            mm.update(wf=wf_name, model=m, n=int(te.sum()),
                      n_pos=int(yte.sum()))
            map_metrics_rows.append(mm)
            # by-symbol
            df_te = pd.DataFrame(dict(symbol=F["symbol"].to_numpy()[te],
                                     y=yte, p=p, side=side[te]))
            for sym, g in df_te.groupby("symbol"):
                if len(g) < 100 or g["y"].nunique() < 2:
                    continue
                auc = roc_auc_score(g["y"], g["p"])
                bal = balanced_accuracy_score(g["y"], (g["p"] >= 0.5).astype(int))
                bysym_rows.append(dict(wf=wf_name, model=m, symbol=sym, n=len(g),
                                       auc=round(float(auc), 4),
                                       accuracy=round(float((g["p"] >= 0.5).astype(int).eq(g["y"]).mean()), 4),
                                       balanced_accuracy=round(float(bal), 4)))
        # nonlinear
        for nm, base in N_MODEL_COLS.items():
            cols = [c for c in block_cols(base, LIQ_TYPES) if c in X.columns]
            p = fit_predict(cols, Xtr, ytr, Xte, "hgb")
            preds[nm] = p
            preds_store[(wf_name, nm)] = p
            mm = metrics(yte, p, side[te], rr[te])
            mm.update(wf=wf_name, model=nm, n=int(te.sum()),
                      n_pos=int(yte.sum()), nonlinear="HGB")
            map_metrics_rows.append(mm)
            df_te = pd.DataFrame(dict(symbol=F["symbol"].to_numpy()[te],
                                     y=yte, p=p, side=side[te]))
            for sym, g in df_te.groupby("symbol"):
                if len(g) < 100 or g["y"].nunique() < 2:
                    continue
                auc = roc_auc_score(g["y"], g["p"])
                bal = balanced_accuracy_score(g["y"], (g["p"] >= 0.5).astype(int))
                bysym_rows.append(dict(wf=wf_name, model=nm, symbol=sym, n=len(g),
                                       auc=round(float(auc), 4),
                                       accuracy=round(float((g["p"] >= 0.5).astype(int).eq(g["y"]).mean()), 4),
                                       balanced_accuracy=round(float(bal), 4)))

        # ---- paired bootstrap ----
        pairs = [("GLOBAL2-ID", "M_ID", "M_GLOBAL2"),
                 ("GLOBAL_SCOPE-GLOBAL4", "M_GLOBAL4", "M_GLOBAL_SCOPE"),
                 ("GLOBAL_TYPE-GLOBAL4", "M_GLOBAL4", "M_GLOBAL_TYPE"),
                 ("IDENTITY_MAP-GLOBAL4", "M_GLOBAL4", "M_IDENTITY_MAP"),
                 ("MAP_CORE-IDENTITY_MAP", "M_IDENTITY_MAP", "M_MAP_CORE"),
                 ("N_IDENTITY_MAP-N_GLOBAL4", "N_GLOBAL4", "N_IDENTITY_MAP"),
                 ("N_MAP_CORE-N_IDENTITY_MAP", "N_IDENTITY_MAP", "N_MAP_CORE")]
        for nm, a, b in pairs:
            if a not in preds or b not in preds:
                continue
            lo, med, hi = paired_day_bootstrap(
                yte, preds[a], preds[b], day_te)
            auc_a = np.nanmean([mm["roc_auc"] for mm in map_metrics_rows
                                if mm["wf"] == wf_name and mm["model"] == a])
            auc_b = np.nanmean([mm["roc_auc"] for mm in map_metrics_rows
                                if mm["wf"] == wf_name and mm["model"] == b])
            paired_rows.append(dict(wf=wf_name, delta=nm,
                                    point_estimate=round(float(auc_b - auc_a), 4),
                                    ci_lower=round(float(lo), 4),
                                    ci_median=round(float(med), 4),
                                    ci_upper=round(float(hi), 4)))

        # ---- P7 两尾 confidence（M_MAP_CORE）----
        cols = [c for c in block_cols("M_MAP_CORE", LIQ_TYPES) if c in X.columns]
        oidx, op = expanding_oof_pred(cols, X[tr], ytr, days[tr], "logistic")
        if len(op):
            p_test = preds["M_MAP_CORE"]
            for q in (0.5, 0.3, 0.2, 0.1):
                half = q / 2.0
                lo = np.quantile(op, half)
                hi = np.quantile(op, 1.0 - half)
                pc = np.full(len(p_test), -1)
                pc[p_test <= lo] = 0
                pc[p_test >= hi] = 1
                sel = pc != -1
                if sel.sum() == 0:
                    continue
                ysel = yte[sel]
                rev_mask = (pc[sel] == 1)
                cont_mask = (pc[sel] == 0)
                rev_tail = float((ysel[rev_mask] == 1).mean()) if rev_mask.sum() else np.nan
                cont_tail = float((ysel[cont_mask] == 0).mean()) if cont_mask.sum() else np.nan
                macro = np.nanmean([rev_tail, cont_tail])
                overall = float((pc[sel] == ysel).mean())
                base = float(ysel.mean())
                maj = max(base, 1 - base)
                # absolute direction on selected
                psel = pc[sel]
                pred_long = np.where(side[te][sel] == +1, 1 - psel, psel)
                actual_long = (rr[te][sel] == "LONG_DOMINATES").astype(int)
                ad_acc = float((pred_long == actual_long).mean())
                conf_rows.append(dict(wf=wf_name, coverage_target=q,
                                      actual_test_coverage=round(float(sel.mean()), 4),
                                      n=int(sel.sum()),
                                      n_pred_reversal=int((pc == 1).sum()),
                                      n_pred_continuation=int((pc == 0).sum()),
                                      reversal_tail_accuracy=round(rev_tail, 4),
                                      continuation_tail_accuracy=round(cont_tail, 4),
                                      macro_tail_accuracy=round(float(macro), 4),
                                      overall_selected_accuracy=round(overall, 4),
                                      selected_reversal_base_rate=round(base, 4),
                                      selected_majority_baseline=round(maj, 4),
                                      accuracy_minus_majority=round(overall - maj, 4),
                                      LONG_selected_accuracy=round(ad_acc, 4)))
        print(f"  {wf_name} direction done ({time.perf_counter()-t0:.1f}s)")

        # ---- P8 clear-direction gate ----
        trc = m_tr & elig_clear
        tec = m_te & elig_clear
        if len(set(y_clear[trc])) >= 2 and len(set(y_clear[tec])) >= 2:
            for cm, base in CLEAR_MODELS.items():
                cols = [c for c in block_cols(base, LIQ_TYPES) if c in X.columns]
                p = fit_predict(cols, X[trc], y_clear[trc], X[tec], "logistic")
                mm = metrics(y_clear[tec], p)
                mm.update(wf=wf_name, model=cm, n=int(tec.sum()),
                          n_pos=int(y_clear[tec].sum()), task="clear_gate")
                clear_rows.append(mm)
            # clear HGB
            cols = [c for c in block_cols("M_MAP_CORE", LIQ_TYPES) if c in X.columns]
            p = fit_predict(cols, X[trc], y_clear[trc], X[tec], "hgb")
            mm = metrics(y_clear[tec], p)
            mm.update(wf=wf_name, model="CN_MAP_CORE", n=int(tec.sum()),
                      n_pos=int(y_clear[tec].sum()), task="clear_gate",
                      nonlinear="HGB")
            clear_rows.append(mm)

            # ---- P9 联合选择器（expanding OOF 定阈值）----
            p_clear_oof_idx, p_clear_oof = expanding_oof_pred(
                cols, X[trc], y_clear[trc], days[trc], "logistic")
            p_dir_oof_idx, p_dir_oof = expanding_oof_pred(
                cols, X[trc], y_clear[trc], days[trc], "logistic")
            if len(p_clear_oof) and len(p_dir_oof):
                clear_thr = 0.5
                half = 0.1
                dlo = np.quantile(p_dir_oof, half)
                dhi = np.quantile(p_dir_oof, 1 - half)
                p_clear_test = fit_predict(cols, X[trc], y_clear[trc],
                                           X[tec], "logistic")
                p_dir_test = p_clear_test  # 同模型同 cohort
                selected = (p_clear_test >= clear_thr) & (
                    (p_dir_test <= dlo) | (p_dir_test >= dhi))
                yc = y_clear[tec]
                sel = selected & (~pd.isna(yc))
                if sel.sum() > 0:
                    sel_clear = yc[sel] == 1
                    # direction among selected-and-clear
                    pc = np.where(p_dir_test[sel] <= dlo, 0, 1)
                    ysel = y_rev[tec][sel]
                    dir_acc_clear = float((pc == ysel).mean()) if sel_clear.sum() else np.nan
                    actionable = float(((pc == ysel) & (yc[sel] == 1)).mean())
                    # LONG/SHORT actionable
                    sside = side[tec][sel]
                    srr = rr[tec][sel]
                    sabs = (np.where(sside == +1, 1 - pc, pc) ==
                            (srr == "LONG_DOMINATES").astype(int))
                    long_act = float(sabs[sside == +1].mean()) if (sside == +1).sum() else np.nan
                    short_act = float(sabs[sside == -1].mean()) if (sside == -1).sum() else np.nan
                    sel_rows.append(dict(
                        wf=wf_name, selection_rate=round(float(sel.mean()), 4),
                        n_selected=int(sel.sum()),
                        selected_clear_rate=round(float(sel_clear.mean()), 4),
                        direction_accuracy_given_clear=round(dir_acc_clear, 4),
                        actionable_precision=round(actionable, 4),
                        LONG_actionable_precision=round(long_act, 4),
                        SHORT_actionable_precision=round(short_act, 4)))

        # ---- P10 ROBUST diagnostic ----
        trb = m_tr & elig_robust
        teb = m_te & elig_robust
        if len(set(y_robust[trb])) >= 2 and len(set(y_robust[teb])) >= 2:
            for m in ROBUST_MODELS:
                cols = [c for c in block_cols(m, LIQ_TYPES) if c in X.columns]
                p = fit_predict(cols, X[trb], y_robust[trb], X[teb], "logistic")
                mm = metrics(y_robust[teb], p, side[teb], rr[teb])
                mm.update(wf=wf_name, model=m, n=int(teb.sum()),
                          n_pos=int(y_robust[teb].sum()), task="robust")
                robust_rows.append(mm)
            cols = [c for c in block_cols("M_MAP_CORE", LIQ_TYPES) if c in X.columns]
            p = fit_predict(cols, X[trb], y_robust[trb], X[teb], "hgb")
            mm = metrics(y_robust[teb], p, side[teb], rr[teb])
            mm.update(wf=wf_name, model="N_MAP_CORE", n=int(teb.sum()),
                      n_pos=int(y_robust[teb].sum()), task="robust",
                      nonlinear="HGB")
            robust_rows.append(mm)

        # ---- P11 risk coupling ----
        for risk, ylab in (("r05", y_r05), ("r10", y_rev), ("r20", y_r20)):
            mask = (~pd.isna(ylab)) & insample
            trk = m_tr & mask
            tek = m_te & mask
            if len(set(ylab[trk])) < 2 or len(set(ylab[tek])) < 2:
                continue
            for m in RISK_MODELS:
                cols = [c for c in block_cols(m, LIQ_TYPES) if c in X.columns]
                p = fit_predict(cols, X[trk], ylab[trk], X[tek], "logistic")
                mm = metrics(ylab[tek], p, side[tek], rr[tek])
                mm.update(wf=wf_name, model=f"{m}_{risk}", risk=risk,
                          n=int(tek.sum()), n_pos=int(ylab[tek].sum()),
                          task="risk")
                risk_rows.append(mm)
            cols = [c for c in block_cols("M_MAP_CORE", LIQ_TYPES) if c in X.columns]
            p = fit_predict(cols, X[trk], ylab[trk], X[tek], "hgb")
            mm = metrics(ylab[tek], p, side[tek], rr[tek])
            mm.update(wf=wf_name, model=f"N_MAP_CORE_{risk}", risk=risk,
                      n=int(tek.sum()), n_pos=int(ylab[tek].sum()),
                      task="risk", nonlinear="HGB")
            risk_rows.append(mm)

    # ---- 写出 ----
    pd.DataFrame(map_metrics_rows).to_csv(
        OUT / "map_decomposition_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(paired_rows).to_csv(
        OUT / "map_decomposition_paired_ci.csv", index=False,
        encoding="utf-8-sig")
    pd.DataFrame(bysym_rows).to_csv(
        OUT / "direction_by_symbol_corrected.csv", index=False,
        encoding="utf-8-sig")
    pd.DataFrame(conf_rows).to_csv(
        OUT / "confidence_two_tail_metrics.csv", index=False,
        encoding="utf-8-sig")
    pd.DataFrame(clear_rows).to_csv(
        OUT / "clear_direction_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sel_rows).to_csv(
        OUT / "joint_selector_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(robust_rows).to_csv(
        OUT / "robust_direction_diagnostic.csv", index=False,
        encoding="utf-8-sig")
    pd.DataFrame(risk_rows).to_csv(
        OUT / "risk_sensitivity_05_10_20.csv", index=False,
        encoding="utf-8-sig")

    # ---- block deltas（增量，相对基准，写回正式脚本）----
    piv = pd.DataFrame(map_metrics_rows).pivot_table(
        index="wf", columns="model", values="roc_auc")
    delta_defs = [("GLOBAL2-ID", "M_GLOBAL2", "M_ID"),
                  ("GLOBAL4-GLOBAL2", "M_GLOBAL4", "M_GLOBAL2"),
                  ("SCOPE12-ID", "M_SCOPE12", "M_ID"),
                  ("GLOBAL_SCOPE-GLOBAL4", "M_GLOBAL_SCOPE", "M_GLOBAL4"),
                  ("CORE-GLOBAL_SCOPE", "M_CORE", "M_GLOBAL_SCOPE"),
                  ("TYPE20-ID", "M_TYPE20", "M_ID"),
                  ("GLOBAL_TYPE-GLOBAL4", "M_GLOBAL_TYPE", "M_GLOBAL4"),
                  ("IDENTITY_MAP-GLOBAL4", "M_IDENTITY_MAP", "M_GLOBAL4"),
                  ("MAP_CORE-IDENTITY_MAP", "M_MAP_CORE", "M_IDENTITY_MAP")]
    dr = {}
    for nm, b, a in delta_defs:
        dr[nm] = float((piv[b] - piv[a]).mean())
    pd.DataFrame([dr]).to_csv(OUT / "direction_feature_block_deltas.csv",
                             index=False, encoding="utf-8-sig")

    # ---- Gate（P12）----
    def wf_auc(model, wf):
        sub = pd.DataFrame(map_metrics_rows)
        r = sub[(sub.model == model) & (sub.wf == wf)]["roc_auc"]
        return float(r.mean()) if len(r) else np.nan
    nmap_mean = np.mean([wf_auc("N_MAP_CORE", wf) for wf, _, _ in WF])
    nmap_wf = {wf: wf_auc("N_MAP_CORE", wf) for wf, _, _ in WF}
    cov20 = pd.DataFrame(conf_rows)
    if len(cov20):
        macro20 = cov20[cov20.coverage_target == 0.2]["macro_tail_accuracy"].mean()
    else:
        macro20 = np.nan
    bysym = pd.DataFrame(bysym_rows)
    n_sym_auc_gt05 = bysym[bysym.model == "N_MAP_CORE"]["auc"].gt(0.5).sum()
    macro_median_sym = bysym[bysym.model == "N_MAP_CORE"].groupby("symbol")["auc"].mean().median()
    # clear gate
    clr = pd.DataFrame(clear_rows)
    def wf_auc_c(model, wf):
        r = clr[(clr.model == model) & (clr.wf == wf)]["roc_auc"]
        return float(r.mean()) if len(r) else np.nan
    clear_mean = np.mean([wf_auc_c("C_MAP_CORE", wf) for wf, _, _ in WF])
    # joint selector
    sel = pd.DataFrame(sel_rows)
    act_prec = sel["actionable_precision"].mean() if len(sel) else np.nan
    sel_wf_min = sel.groupby("wf")["actionable_precision"].min().min() if len(sel) else np.nan
    # identity incremental
    pid = pd.DataFrame(paired_rows)
    idi = pid[pid.delta == "IDENTITY_MAP-GLOBAL4"]
    idi_lower_ok = (idi["ci_lower"] > 0).sum()
    idi_point = idi["point_estimate"].mean()
    gate = dict(
        DIRECTION_SIGNAL_CONFIRMED=bool(
            nmap_mean >= 0.60 and all(v >= 0.58 for v in nmap_wf.values())
            and pd.notna(macro20) and macro20 >= 0.60
            and n_sym_auc_gt05 >= 10),
        DEPLOYABLE_DIRECTION_SELECTOR=bool(
            pd.notna(clear_mean) and clear_mean >= 0.55
            and pd.notna(act_prec) and act_prec >= 0.60
            and pd.notna(sel_wf_min) and sel_wf_min >= 0.55),
        LIQUIDITY_IDENTITY_INCREMENTAL=bool(
            idi_lower_ok >= 2 and pd.notna(idi_point) and idi_point >= 0.01),
        n_map_core_mean_auc=round(float(nmap_mean), 4),
        n_map_core_per_wf={k: round(v, 4) for k, v in nmap_wf.items()},
        cov20_macro_tail_accuracy=round(float(macro20), 4) if pd.notna(macro20) else None,
        n_symbols_auc_gt_05=int(n_sym_auc_gt05),
        macro_median_symbol_auc=round(float(macro_median_sym), 4),
        clear_gate_mean_auc=round(float(clear_mean), 4) if pd.notna(clear_mean) else None,
        joint_actionable_precision=round(float(act_prec), 4) if pd.notna(act_prec) else None,
        identity_incremental_point_delta=round(float(idi_point), 4) if pd.notna(idi_point) else None,
        identity_incremental_wf_ci_lower_gt0=int(idi_lower_ok),
    )
    gate["verdict"] = (
        "DEPLOYABLE_DIRECTION_SELECTOR" if gate["DEPLOYABLE_DIRECTION_SELECTOR"] else
        "DIRECTION_SIGNAL_CONFIRMED" if gate["DIRECTION_SIGNAL_CONFIRMED"] else
        "LIQUIDITY_IDENTITY_INCREMENTAL_ONLY" if gate["LIQUIDITY_IDENTITY_INCREMENTAL"] else
        "DIRECTION_SIGNAL_INSUFFICIENT")

    # ---- repair audit（P0）----
    repair = dict(
        v10_bugs_fixed=[
            "reversal_label 非 clear 标签原返回错误值；现返回 NaN（P0.1）",
            "by-symbol balanced_accuracy 原公式错误（k=0 仍在算 pred==1）；现用 sklearn（P0.2）",
            "Gate Top20 原误用 L_full 全样本 accuracy；现删除，改用 P7 两尾 macro_tail（P0.3）",
            "direction_feature_block_deltas 原 ad-hoc 修改未入脚本；现写回正式脚本（P0.4）",
            "liquidity_type 真实为 10 种（原报告误写 9）（P0.5）",
        ],
        reproducibility="checkout e7ee079 + 本脚本一键重建所有小型输出",
    )

    # ---- reproducibility audit ----
    def fhash(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    reprod = dict(
        script_sha256=fhash(__file__),
        input_artifacts={
            "liquidity_master_v1_1.parquet": fhash(str(ATLAS/"liquidity_master_v1_1.parquet")),
            "liquidity_contacts_v1_1.parquet": fhash(str(ATLAS/"liquidity_contacts_v1_1.parquet")),
            "liquidity_state_snapshot_v1_2.parquet": fhash(str(ATLAS/"liquidity_state_snapshot_v1_2.parquet")),
            "oracle_risk_direction_v1_2.parquet": fhash(str(ATLAS/"oracle_risk_direction_v1_2.parquet")),
        },
        output_rowcounts={
            "map_decomposition_metrics": len(map_metrics_rows),
            "map_decomposition_paired_ci": len(paired_rows),
            "direction_by_symbol_corrected": len(bysym_rows),
            "confidence_two_tail_metrics": len(conf_rows),
            "clear_direction_metrics": len(clear_rows),
            "joint_selector_metrics": len(sel_rows),
            "robust_direction_diagnostic": len(robust_rows),
            "risk_sensitivity_05_10_20": len(risk_rows),
        },
        primary_risk=PRIMARY_RISK, oos_start=OOS_START,
        eligible_in_sample_risk1_clear=int(elig.sum()),
        eligible_in_sample_risk1_robust=int(elig_robust.sum()),
    )

    protocol = dict(
        experiment="SMC Direction Deployability & Liquidity Map Decomposition v1.1",
        base_commit="e7ee07937681c89f6da60c2264381085840b36ce",
        primary_risk=PRIMARY_RISK, oos_start=OOS_START,
        liquidity_type_count=len(LIQ_TYPES), liquidity_scope_count=len(list(SCOPES)),
        models=DIR_MODELS + list(N_MODEL_COLS) + list(CLEAR_MODELS) +
               ["CN_MAP_CORE"] + ROBUST_MODELS,
        nonlinear="HistGradientBoostingClassifier(max_depth=3,lr=0.05,max_iter=200,l2=1.0)",
        governance=dict(trading_metrics="NOT_APPLICABLE", no_pnl=True,
                        no_fvg=True, no_external_internal=True, no_lc=True,
                        atlas_v1_2_unchanged=True),
        gate=gate,
    )
    audit = dict(experiment=protocol["experiment"],
                 base_commit=protocol["base_commit"],
                 roi_gate=gate, repair_audit=repair,
                 identity_enum_summary={k: len(v) for k, v in enum.items()},
                 reproducibility=reprod)
    json.dump(protocol, open(OUT / "DIRECTION_V1_1_PROTOCOL.json", "w"),
              indent=2, ensure_ascii=False)
    json.dump(repair, open(OUT / "v10_repair_audit.json", "w"), indent=2,
              ensure_ascii=False)
    json.dump(reprod, open(OUT / "REPRODUCIBILITY_AUDIT.json", "w"), indent=2,
              ensure_ascii=False)
    json.dump(audit, open(OUT / "DIRECTION_V1_1_AUDIT.json", "w"), indent=2,
              ensure_ascii=False, default=str)
    print("\n=== ROI GATE (v1.1) ===")
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    print(f"\n[DONE] {time.perf_counter()-t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
