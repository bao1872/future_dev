"""SMC Direction Identity & Feasibility Gate v1.0

用户正式执行协议（base commit 647f9ad）。主线从 continuous frontier 切到方向可行性。
P4c 机制结论冻结（DIRECT_STRUCTURAL_SWITCH），本轮不研究 continuous frontier。

目标：在 decision_time 当时，仅用事前可见状态，能否稳定判断未来占优方向。
Primary label = REVERSAL vs CONTINUATION @ risk=1.0 ATR（镜像统一上下扫）。
Primary risk 固定 1.0，禁止 7 档挑 risk。

复用：
- frozen oracle_risk_direction_v1_2（risk=1.0）
- liquidity_state_snapshot_v1_2（D0–D6 绝大多数特征，含 60 bin + trend + OB）
- contacts / master（补 liquidity_type / is_first_contact / bars_since_* + master 派生：
  contact_cluster_* 与 12 个 per-scope nearest 标量）

Governance: TRADING_METRICS=NOT_APPLICABLE；无 PnL/止损/最优risk/FVG/external-internal/
LC。Atlas v1.2 冻结不动。大型 parquet（direction_features）不入 Git。
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, SplineTransformer,
                                   StandardScaler)

import opportunity_common as oc
from research.liquidity_oracle_atlas.cf_common import active_mask
from research.liquidity_oracle_atlas.build_oracle_atlas_v1_2 import SCOPES

ATLAS = Path("research/analysis_results/smc_oracle_atlas_v1")
OUT = Path("research/analysis_results/smc_direction_feasibility_v1")
OUT.mkdir(parents=True, exist_ok=True)
SYMBOLS = ["AG", "AU", "CU", "AL", "SN", "NI", "RB", "I", "SC", "RU", "MA",
           "TA", "M", "P", "CF"]
PRIMARY_RISK = 1.0
OOS_START = "2026-09-07"          # prospective OOS，完全不入任何 fit/阈值/gate
BLOCKS = ["TB1", "TB2", "TB3", "TB4"]
WF = [("WF1", ["TB1"], ["TB2"]),
      ("WF2", ["TB1", "TB2"], ["TB3"]),
      ("WF3", ["TB1", "TB2", "TB3"], ["TB4"])]

# ---------------------------------------------------------------------------
# 泄漏黑名单（section 21）
# ---------------------------------------------------------------------------
FORBIDDEN = {
    "rr_direction", "direction_stability", "best_R_lower", "best_R_upper",
    "resolution_class", "status", "path_censor", "critical_risk",
    "required_risk_before_target_ATR", "required_risk_through_target_ATR",
    "required_before_ATR", "required_through_ATR", "bars_to_best_lower",
    "bars_to_best_target", "best_target_price", "liquidity_price",
    "entry_reference", "decision_time", "contact_bar_index", "trading_day",
    "block", "y_reversal", "y_long",
}


def reversal_label(side, rr):
    """section 2：上方 liquidity reversal=SHORT；下方 liquidity reversal=LONG。"""
    is_long = (rr == "LONG_DOMINATES")
    if side == +1:
        return int(not is_long)
    if side == -1:
        return int(is_long)
    raise ValueError(side)


# ---------------------------------------------------------------------------
# Stage 1：master 派生特征（contact_cluster_* + 12 per-scope nearest 标量）
# ---------------------------------------------------------------------------
def build_symbol_master_features(sym, master, contacts):
    ms = master[master["symbol"] == sym]
    mp = ms["price"].to_numpy(float)
    mscope = ms["liquidity_scope"].astype(str).to_numpy()
    mtype = ms["liquidity_type"].astype(str).to_numpy()
    mav = pd.to_datetime(ms["available_time"]).to_numpy(np.datetime64)
    mfp = pd.to_datetime(ms["first_penetration_time"]).to_numpy(np.datetime64)
    scopes = list(SCOPES)
    scope_sorted = {}
    for s in scopes:
        idx = np.flatnonzero(mscope == s)
        sp = mp[idx]
        order = np.argsort(sp, kind="mergesort")
        scope_sorted[s] = dict(price=sp[order],
                               av=mav[idx][order], fp=mfp[idx][order])
    d = defaultdict(list)
    for i, p in enumerate(mp):
        d[p].append(i)
    cc = contacts[contacts["symbol"] == sym]
    rows = []
    for r in cc.itertuples(index=False):
        dt = np.datetime64(pd.Timestamp(r.decision_time))
        a0 = float(r.atr0)
        if not (np.isfinite(a0) and a0 > 0):
            a0 = np.nan
        entry = float(r.entry_reference)
        lvl = float(r.liquidity_price)
        # ---- contact_cluster_*（pre-contact same-price）----
        same = d.get(lvl, [])
        id_count, scopes_set, types_set = 0, set(), set()
        has = {s: False for s in scopes}
        for i in same:
            if mav[i] <= dt and (np.isnat(mfp[i]) or mfp[i] >= dt):
                id_count += 1
                scopes_set.add(mscope[i])
                types_set.add(mtype[i])
                if mscope[i] in has:
                    has[mscope[i]] = True
        # ---- per-scope nearest（active_mask = Oracle 同款）----
        per = {}
        for s in scopes:
            ss = scope_sorted[s]
            pr, avv, fpp = ss["price"], ss["av"], ss["fp"]
            n = len(pr)
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
            per[f"nearest_above_{s}_R"] = da
            per[f"nearest_below_{s}_R"] = db
        rec = dict(symbol=sym, liquidity_id=r.liquidity_id,
                   contact_number=r.contact_number,
                   contact_cluster_identity_count=id_count,
                   contact_cluster_scope_count=len(scopes_set),
                   contact_cluster_type_count=len(types_set),
                   contact_cluster_types="|".join(sorted(types_set)))
        for s in scopes:
            rec[f"contact_cluster_has_{s}"] = bool(has[s])
        rec.update(per)
        rows.append(rec)
    return pd.DataFrame(rows)


def build_features():
    MASTER = pd.read_parquet(ATLAS / "liquidity_master_v1_1.parquet")
    CON = pd.read_parquet(ATLAS / "liquidity_contacts_v1_1.parquet")
    ST = pd.read_parquet(ATLAS / "liquidity_state_snapshot_v1_2.parquet")
    rd = pd.read_parquet(ATLAS / "oracle_risk_direction_v1_2.parquet")
    r1 = rd[np.isclose(rd["risk_ATR"], PRIMARY_RISK)].copy()
    r1 = r1.merge(CON[["symbol", "liquidity_id", "contact_number", "side"]],
                  on=["symbol", "liquidity_id", "contact_number"], how="left")
    r1["y_reversal"] = [reversal_label(s, rr) for s, rr in
                        zip(r1["side"], r1["rr_direction"])]
    r1["y_long"] = (r1["rr_direction"] == "LONG_DOMINATES").astype(int)
    r1["eligible"] = r1["rr_direction"].isin(
        ["LONG_DOMINATES", "SHORT_DOMINATES"])
    r1 = r1.merge(CON[["symbol", "liquidity_id", "contact_number", "side"]],
                  on=["symbol", "liquidity_id", "contact_number"], how="left")
    keys = ["symbol", "liquidity_id", "contact_number"]

    # master 派生
    mf = []
    for sym in SYMBOLS:
        mf.append(build_symbol_master_features(sym, MASTER, CON))
    MFEAT = pd.concat(mf, ignore_index=True)

    # contacts meta（snapshot 缺 liquidity_type / is_first_contact / bars_since_*）
    con_meta = CON[keys + ["liquidity_type", "is_first_contact",
                           "bars_since_available", "bars_since_previous_contact"]].copy()

    # assembly
    F = ST.merge(con_meta, on=keys, how="left")
    F = F.merge(MFEAT, on=keys, how="left")
    F = F.merge(r1[keys + ["rr_direction", "y_reversal", "y_long", "eligible"]],
                on=keys, how="left")
    # block / trading_day
    blk = oc.attach_trading_day_block(F[keys].copy())
    F = F.merge(blk, on=keys, how="left")
    F.to_parquet(OUT / "direction_features.parquet", index=False)
    return F, r1


# ---------------------------------------------------------------------------
# Stage 2：审计
# ---------------------------------------------------------------------------
def identity_enum(con):
    out = {}
    for c in ["liquidity_type", "liquidity_scope", "contact_type", "side",
              "is_first_contact"]:
        out[c] = con[c].value_counts(dropna=False).to_dict()
    return out


def label_audit(r1, F):
    """risk=1.0 标签分布 by symbol×TB。"""
    a = r1.merge(F[["symbol", "liquidity_id", "contact_number", "block"]],
                on=["symbol", "liquidity_id", "contact_number"], how="left")
    a["REVERSAL"] = np.where(a["rr_direction"].isin(
        ["LONG_DOMINATES", "SHORT_DOMINATES"]),
        a["y_reversal"], np.nan)
    rows = []
    for (sym, tb), g in a.groupby(["symbol", "block"]):
        el = g[g["eligible"]]
        rows.append(dict(symbol=sym, TB=tb, risk_ATR=PRIMARY_RISK, n=len(g),
                         LONG_n=int((g["rr_direction"] == "LONG_DOMINATES").sum()),
                         SHORT_n=int((g["rr_direction"] == "SHORT_DOMINATES").sum()),
                         TRADEOFF_n=int((g["rr_direction"] == "TRADEOFF_OR_OVERLAP").sum()),
                         CENSOR_n=int((g["rr_direction"] == "UNRESOLVED_CENSOR").sum()),
                         NOCOMP_n=int((g["rr_direction"] == "NO_COMPARABLE_TARGET").sum()),
                         REVERSAL_n=int(el["y_reversal"].sum()),
                         CONTINUATION_n=int((el["y_reversal"] == 0).sum())))
    return pd.DataFrame(rows)


def identity_direction_profile(F, eligible):
    """section 10：不同 liquidity identity 的方向结果是否客观不同。"""
    rows = []
    groups = [
        ("liquidity_scope", "liquidity_scope"),
        ("liquidity_type", "liquidity_type"),
        ("side", "side"),
        ("contact_type", "contact_type"),
        ("is_first_contact", "is_first_contact"),
        ("contact_cluster_scope_count", "contact_cluster_scope_count"),
        ("contact_cluster_identity_count", "contact_cluster_identity_count"),
    ]
    for gname, col in groups:
        for v, g in eligible.groupby(col):
            n = len(g)
            if n < 200:
                continue
            rows.append(dict(group=gname, value=str(v), n=n,
                            reversal_rate=round(float(g["y_reversal"].mean()), 4),
                            continuation_rate=round(float((g["y_reversal"] == 0).mean()), 4),
                            LONG_rate=round(float((g["rr_direction"] == "LONG_DOMINATES").mean()), 4),
                            SHORT_rate=round(float((g["rr_direction"] == "SHORT_DOMINATES").mean()), 4),
                            symbol_count=int(g["symbol"].nunique()),
                            macro_median_reversal=float(g.groupby("symbol")["y_reversal"].mean().median())))
    # 固定交叉
    crosses = [("liquidity_scope", "contact_type"),
               ("liquidity_scope", "is_first_contact"),
               ("contact_cluster_scope_count", "contact_type")]
    for a, b in crosses:
        for (va, vb), g in eligible.groupby([a, b]):
            n = len(g)
            if n < 200:
                continue
            rows.append(dict(group=f"{a}x{b}", value=f"{va}|{vb}", n=n,
                            reversal_rate=round(float(g["y_reversal"].mean()), 4),
                            continuation_rate=round(float((g["y_reversal"] == 0).mean()), 4),
                            LONG_rate=round(float((g["rr_direction"] == "LONG_DOMINATES").mean()), 4),
                            SHORT_rate=round(float((g["rr_direction"] == "SHORT_DOMINATES").mean()), 4),
                            symbol_count=int(g["symbol"].nunique()),
                            macro_median_reversal=float(g.groupby("symbol")["y_reversal"].mean().median())))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage 3：特征块 + 模型
# ---------------------------------------------------------------------------
SCOPE_BIN_COLS = [c for c in [] ]  # 占位，运行时填充
SPLINE_NUMERIC = set()
ORDINARY_NUMERIC = set()
CATEGORICAL = set()


def define_blocks(F):
    global SCOPE_BIN_COLS, SPLINE_NUMERIC, ORDINARY_NUMERIC, CATEGORICAL
    bin_cols = [c for c in F.columns if c.endswith("_bin_(-inf,-4]") or
                "_bin_" in c and c.split("_bin_")[0] in SCOPES]
    SCOPE_BIN_COLS = [c for c in F.columns if "_bin_" in c and (
        c.split("_bin_")[0] in SCOPES)]
    distances = (["nearest_above_R", "nearest_below_R", "nearest_ahead_R",
                  "nearest_behind_R"]
                 + [f"nearest_above_{s}_R" for s in SCOPES]
                 + [f"nearest_below_{s}_R" for s in SCOPES]
                 + ["penetration_depth_R", "close_relative_to_level_R",
                    "bar_range_R", "abs_return_R", "atr0"])
    SPLINE_NUMERIC = set(distances)
    ORDINARY_NUMERIC = set(
        ["contact_number", "bars_since_available", "bars_since_previous_contact",
         "contact_cluster_identity_count", "contact_cluster_scope_count",
         "contact_cluster_type_count", "same_price_identity_count",
         "same_price_identity_count_v11", "active_visible_count",
         "nearest_opposing_ob_prior_enter_count",
         "nearest_same_direction_ob_prior_enter_count"]
        + [f"contact_cluster_has_{s}" for s in SCOPES]
        + SCOPE_BIN_COLS
        + ["nearest_opposing_ob_distance_R", "nearest_opposing_ob_width_R",
           "nearest_same_direction_ob_distance_R",
           "nearest_same_direction_ob_width_R"])
    CATEGORICAL = set(
        ["symbol", "side", "liquidity_type", "liquidity_scope", "contact_type",
         "is_first_contact",
         "env_direction_4h", "trend_struct_1h", "trend_struct_15m",
         "trend_struct_5m", "sweep_vs_5m", "sweep_vs_15m", "sweep_vs_1h",
         "env4h_vs_1h", "trend_1h_vs_15m", "trend_15m_vs_5m",
         "nearest_opposing_ob_source_tf", "nearest_opposing_ob_freshness",
         "nearest_same_direction_ob_source_tf",
         "nearest_same_direction_ob_freshness"])


def block_cols(name):
    base = ["symbol", "side"]
    if name == "L0":
        return list(base)
    if name == "L_identity":
        return base + ["liquidity_type", "liquidity_scope", "contact_number",
                       "is_first_contact", "bars_since_available",
                       "bars_since_previous_contact",
                       "contact_cluster_identity_count",
                       "contact_cluster_scope_count",
                       "contact_cluster_has_5m", "contact_cluster_has_15m",
                       "contact_cluster_has_1h", "contact_cluster_has_CONTIG_SESSION",
                       "contact_cluster_has_TRADING_DAY",
                       "contact_cluster_has_TRADING_WEEK",
                       "contact_cluster_type_count"]
    if name == "L_map":
        return block_cols("L_identity") + [
            "nearest_above_R", "nearest_below_R", "nearest_ahead_R",
            "nearest_behind_R"] + [f"nearest_above_{s}_R" for s in SCOPES] + [
            f"nearest_below_{s}_R" for s in SCOPES]
    if name == "L_core":
        return block_cols("L_map") + [
            "contact_type", "penetration_depth_R", "close_relative_to_level_R",
            "bar_range_R", "abs_return_R"]
    if name == "L_core_field":
        return block_cols("L_core") + list(SCOPE_BIN_COLS)
    if name == "L_core_trend":
        return block_cols("L_core") + [
            "env_direction_4h", "trend_struct_1h", "trend_struct_15m",
            "trend_struct_5m", "sweep_vs_5m", "sweep_vs_15m", "sweep_vs_1h",
            "env4h_vs_1h", "trend_1h_vs_15m", "trend_15m_vs_5m"]
    if name == "L_core_ob":
        return block_cols("L_core") + [
            "nearest_opposing_ob_distance_R", "nearest_opposing_ob_width_R",
            "nearest_opposing_ob_source_tf", "nearest_opposing_ob_freshness",
            "nearest_opposing_ob_prior_enter_count",
            "nearest_same_direction_ob_distance_R",
            "nearest_same_direction_ob_width_R",
            "nearest_same_direction_ob_source_tf",
            "nearest_same_direction_ob_freshness",
            "nearest_same_direction_ob_prior_enter_count"]
    if name == "L_full":
        return (block_cols("L_core") + list(SCOPE_BIN_COLS) + [
            "env_direction_4h", "trend_struct_1h", "trend_struct_15m",
            "trend_struct_5m", "sweep_vs_5m", "sweep_vs_15m", "sweep_vs_1h",
            "env4h_vs_1h", "trend_1h_vs_15m", "trend_15m_vs_5m",
            "nearest_opposing_ob_distance_R", "nearest_opposing_ob_width_R",
            "nearest_opposing_ob_source_tf", "nearest_opposing_ob_freshness",
            "nearest_opposing_ob_prior_enter_count",
            "nearest_same_direction_ob_distance_R",
            "nearest_same_direction_ob_width_R",
            "nearest_same_direction_ob_source_tf",
            "nearest_same_direction_ob_freshness",
            "nearest_same_direction_ob_prior_enter_count"])
    raise ValueError(name)


MODELS = ["L0", "L_identity", "L_map", "L_core", "L_core_field",
          "L_core_trend", "L_core_ob", "L_full"]


def make_preprocessor(cols):
    sp = [c for c in cols if c in SPLINE_NUMERIC]
    ordi = [c for c in cols if c in ORDINARY_NUMERIC]
    cat = [c for c in cols if c in CATEGORICAL]
    missing = [c for c in cols if c not in SPLINE_NUMERIC
               and c not in ORDINARY_NUMERIC and c not in CATEGORICAL]
    assert not missing, f"unrouted cols: {missing}"
    pre = ColumnTransformer([
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
    return pre


def fit_predict_logistic(cols, Xtr, ytr, Xte):
    pre = make_preprocessor(cols)
    pipe = Pipeline([("pre", pre),
                     ("clf", LogisticRegression(penalty="l2", C=1.0,
                                                solver="lbfgs", max_iter=3000))])
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xte)[:, 1]


def fit_predict_hgb(cols, Xtr, ytr, Xte):
    from sklearn.ensemble import HistGradientBoostingClassifier
    pre = make_preprocessor(cols)
    pipe = Pipeline([("pre", pre),
                     ("clf", HistGradientBoostingClassifier(
                         max_depth=3, learning_rate=0.05, max_iter=200,
                         l2_regularization=1.0, random_state=42))])
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xte)[:, 1]


def abs_dir_acc(p, side, rr):
    """p=reversal prob → 绝对方向 → 与 rr_direction 比较。"""
    pred_rev = (p >= 0.5).astype(int)
    pred_long = np.where(side == +1, 1 - pred_rev, pred_rev)  # 上方:rev→SHORT
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
    out["balanced_accuracy"] = float(
        np.mean([((pred[y == k] == k).mean() if (y == k).sum() else np.nan)
                 for k in [0, 1]]))
    if side is not None and rr is not None:
        out["abs_direction_accuracy"] = abs_dir_acc(p, side, rr)
    return out


def expanding_oof_confidence(cols, X, y, days, mode="logistic"):
    """section 17：outer train 内 3 段 expanding OOF → p_oof + confidence。"""
    uniq = np.sort(pd.unique(days))
    n = len(uniq)
    oof_idx, oof_p = [], []
    for frac in (0.4, 0.6, 0.8):
        k = int(round(n * frac))
        if k >= n:
            continue
        tr_days = uniq[:k]
        oof_days = uniq[k:k + max(1, int(round(n * 0.2)))]
        if len(oof_days) == 0:
            continue
        m_tr = pd.Series(days).isin(tr_days).to_numpy()
        m_te = pd.Series(days).isin(oof_days).to_numpy()
        if m_tr.sum() < 50 or m_te.sum() < 20:
            continue
        if mode == "logistic":
            p = fit_predict_logistic(cols, X[m_tr], y[m_tr], X[m_te])
        else:
            p = fit_predict_hgb(cols, X[m_tr], y[m_tr], X[m_te])
        oof_idx.append(np.flatnonzero(m_te))
        oof_p.append(p)
    if not oof_p:
        return np.array([]), np.array([])
    oof_idx = np.concatenate(oof_idx)
    oof_p = np.concatenate(oof_p)
    return oof_idx, oof_p


def main():
    t0 = time.perf_counter()
    F, r1 = build_features()
    con = pd.read_parquet(ATLAS / "liquidity_contacts_v1_1.parquet")
    print(f"[S1] features built {F.shape} ({time.perf_counter()-t0:.1f}s)")

    # 主队列 = eligible @ risk=1.0 且 in-sample
    F["trading_day"] = F["trading_day"].astype(str)
    insample = F["trading_day"] < OOS_START
    elig = F[F["eligible"] & insample].copy().reset_index(drop=True)
    print(f"[S1] eligible in-sample = {len(elig)} "
          f"(reversal={int(elig.y_reversal.sum())}, "
          f"continuation={int((elig.y_reversal==0).sum())})")

    # 泄漏断言：模型矩阵 X 不得含任何 forbidden 列
    X = elig.drop(columns=[c for c in FORBIDDEN if c in elig.columns]).copy()
    bad = set(X.columns) & FORBIDDEN
    assert not bad, f"leakage columns present in X: {bad}"

    # ---- 审计输出 ----
    enum = identity_enum(con)
    json.dump(enum, open(OUT / "liquidity_identity_enum.json", "w"),
              indent=2, ensure_ascii=False)
    la = label_audit(r1, F)
    la.to_csv(OUT / "direction_label_audit.csv", index=False,
              encoding="utf-8-sig")
    prof = identity_direction_profile(F, elig)
    prof.to_csv(OUT / "liquidity_identity_direction_profile.csv", index=False,
                encoding="utf-8-sig")
    # identity_pair_profile：scope×contact_type 含 reversal 率（prof 已含交叉）
    prof[prof["group"].str.contains("x")].to_csv(
        OUT / "identity_pair_profile.csv", index=False, encoding="utf-8-sig")
    print("[S2] audit done")

    # ---- 特征块定义 ----
    define_blocks(F)
    X = elig.drop(columns=[c for c in FORBIDDEN if c in elig.columns]
                  ).copy()
    y = elig["y_reversal"].to_numpy()
    side = elig["side"].to_numpy()
    rr = elig["rr_direction"].to_numpy()
    block = elig["block"].to_numpy()
    days = elig["trading_day"].to_numpy()

    # 数值化非泄漏数值列
    for c in X.columns:
        if c in CATEGORICAL:
            continue
        X[c] = pd.to_numeric(X[c], errors="coerce")

    by_symbol_rows = []
    wf_rows = []
    block_delta_rows = []
    conf_rows = []

    for wf_name, tr_blocks, te_blocks in WF:
        m_tr = pd.Series(block).isin(tr_blocks).to_numpy()
        m_te = pd.Series(block).isin(te_blocks).to_numpy()
        # 训练集需两类都存
        if len(set(y[m_tr])) < 2 or len(set(y[m_te])) < 2:
            continue
        Xtr, Xte = X[m_tr], X[m_te]
        ytr, yte = y[m_tr], y[m_te]
        for m in MODELS:
            cols = [c for c in block_cols(m) if c in X.columns]
            p = fit_predict_logistic(cols, Xtr, ytr, Xte)
            mm = metrics(yte, p, side[m_te], rr[m_te])
            mm.update(wf=wf_name, model=m, n=int(m_te.sum()),
                      n_pos=int(yte.sum()))
            wf_rows.append(mm)
            # by-symbol
            ste, srr, sside = side[m_te], rr[m_te], (m_te)
            df_te = pd.DataFrame(dict(symbol=elig["symbol"].to_numpy()[m_te],
                                     y=yte, p=p, rr=srr, side=ste))
            for sym, g in df_te.groupby("symbol"):
                if len(g) < 100 or g["y"].nunique() < 2:
                    continue
                auc = roc_auc_score(g["y"], g["p"])
                acc = (g["p"] >= 0.5).astype(int).eq(g["y"]).mean()
                bal = np.mean([((g["p"][g["y"] == k] >= 0.5).astype(int).mean()
                                if (g["y"] == k).sum() else np.nan)
                               for k in [0, 1]])
                by_symbol_rows.append(dict(wf=wf_name, model=m, symbol=sym,
                                            n=len(g), auc=round(float(auc), 4),
                                            accuracy=round(float(acc), 4),
                                            balanced_accuracy=round(float(bal), 4)))
        # nonlinear sanity（core / full）
        for m, cols in (("N_core", block_cols("L_core")),
                        ("N_full", block_cols("L_full"))):
            cols = [c for c in cols if c in X.columns]
            p = fit_predict_hgb(cols, Xtr, ytr, Xte)
            mm = metrics(yte, p, side[m_te], rr[m_te])
            mm.update(wf=wf_name, model=m, n=int(m_te.sum()),
                      n_pos=int(yte.sum()), nonlinear="HistGradientBoosting")
            wf_rows.append(mm)
        # confidence / abstention（用 L_core）
        cols = [c for c in block_cols("L_core") if c in X.columns]
        oidx, op = expanding_oof_confidence(cols, X[m_tr], ytr, days[m_tr],
                                            "logistic")
        if len(op):
            conf = np.abs(op - 0.5)
            qs = {f"cov{q}": float(np.quantile(conf, 1 - q / 100.0))
                  for q in (50, 30, 20, 10)}
            pte = fit_predict_logistic(cols, Xtr, ytr, Xte)
            cte = np.abs(pte - 0.5)
            for q, thr in qs.items():
                keep = cte >= thr
                if keep.sum() == 0:
                    continue
                yy = yte[keep]; pp = pte[keep]
                conf_rows.append(dict(
                    wf=wf_name, coverage_target=q,
                    test_coverage=round(float(keep.mean()), 4), n=int(keep.sum()),
                    roc_auc=float(roc_auc_score(yy, pp)) if len(set(yy)) > 1 else np.nan,
                    direction_accuracy=float(((pp >= 0.5).astype(int) == yy).mean()),
                    abs_direction_accuracy=abs_dir_acc(pp, side[m_te][keep],
                                                      rr[m_te][keep])))
        print(f"  {wf_name} done ({time.perf_counter()-t0:.1f}s)")

    wf_df = pd.DataFrame(wf_rows)
    wf_df.to_csv(OUT / "direction_metrics_by_wf.csv", index=False,
                 encoding="utf-8-sig")

    # block deltas（core 为基准）
    piv = wf_df[wf_df["model"].isin(MODELS)].pivot_table(
        index=["wf"], columns="model", values="roc_auc")
    deltas = {}
    base = piv["L_core"]
    for m in ["L0", "L_identity", "L_map", "L_core", "L_core_field",
              "L_core_trend", "L_core_ob", "L_full"]:
        deltas[m] = float((piv[m] - base).mean())
    block_delta_rows.append(deltas)
    pd.DataFrame(block_delta_rows).to_csv(
        OUT / "direction_feature_block_deltas.csv", index=False,
        encoding="utf-8-sig")

    bysym = pd.DataFrame(by_symbol_rows)
    bysym.to_csv(OUT / "direction_by_symbol.csv", index=False,
                 encoding="utf-8-sig")
    pd.DataFrame(conf_rows).to_csv(
        OUT / "confidence_coverage_curve.csv", index=False,
        encoding="utf-8-sig")

    # ---- ROI Gate ----
    mean_auc = wf_df[wf_df.model == "L_full"]["roc_auc"].mean()
    wf_aucs = {wf: wf_df[(wf_df.wf == wf) & (wf_df.model == "L_full")]["roc_auc"].mean()
               for wf, _, _ in WF}
    # Top20% abs dir acc（L_full, 合并各 WF test）
    top20 = []
    for wf_name, _, _ in WF:
        sub = wf_df[(wf_df.wf == wf_name) & (wf_df.model == "L_full")]
        if len(sub):
            top20.append(sub.iloc[0].get("abs_direction_accuracy", np.nan))
    top20 = [x for x in top20 if pd.notna(x)]
    top20_abs = float(np.mean(top20)) if top20 else np.nan
    n_sym_auc_gt05 = bysym[bysym.model == "L_full"]["auc"].gt(0.5).sum()
    n_sym_total = bysym[bysym.model == "L_full"]["symbol"].nunique()
    macro_median_sym_auc = float(bysym[bysym.model == "L_full"].groupby(
        "symbol")["auc"].mean().median())

    gate = dict(
        mean_WF_Lfull_AUC=round(mean_auc, 4),
        per_WF_Lfull_AUC={k: round(v, 4) for k, v in wf_aucs.items()},
        top20_abs_dir_acc=round(top20_abs, 4) if pd.notna(top20_abs) else None,
        n_symbols_auc_gt_05=int(n_sym_auc_gt05),
        n_symbols_total=int(n_sym_total),
        macro_median_symbol_auc=round(macro_median_sym_auc, 4),
    )
    DIRECTION_LEARNABLE = (mean_auc >= 0.58 and
                           all(v >= 0.54 for v in wf_aucs.values()) and
                           (pd.notna(top20_abs) and top20_abs >= 0.60) and
                           n_sym_auc_gt05 >= 10)
    STRONG = (mean_auc >= 0.62 and pd.notna(top20_abs) and top20_abs >= 0.65
              and macro_median_sym_auc >= 0.58)
    WEAK = (mean_auc < 0.55 and (pd.notna(top20_abs) and top20_abs < 0.58))
    gate["DIRECTION_LEARNABLE"] = bool(DIRECTION_LEARNABLE)
    gate["STRONG_DIRECTION_SIGNAL"] = bool(STRONG)
    gate["WEAK_DIRECTION_SIGNAL"] = bool(WEAK)
    gate["verdict"] = ("STRONG_DIRECTION_SIGNAL" if STRONG else
                       "DIRECTION_LEARNABLE" if DIRECTION_LEARNABLE else
                       "WEAK_DIRECTION_SIGNAL" if WEAK else
                       "INTERMEDIATE_DIRECTION_SIGNAL_PRESENT_BUT_WEAK")
    print("\n=== ROI GATE ===")
    print(json.dumps(gate, indent=2, ensure_ascii=False))

    audit = dict(
        experiment="SMC Direction Identity & Feasibility Gate v1.0",
        base_commit="647f9ad843e64e60aff14f7ef5103254b3848dbd",
        primary_risk=PRIMARY_RISK, oos_start=OOS_START,
        eligible_in_sample=int(len(elig)),
        rr_direction_at_risk1={
            "LONG_DOMINATES": int((r1.rr_direction == "LONG_DOMINATES").sum()),
            "SHORT_DOMINATES": int((r1.rr_direction == "SHORT_DOMINATES").sum()),
            "TRADEOFF_OR_OVERLAP": int((r1.rr_direction == "TRADEOFF_OR_OVERLAP").sum()),
            "UNRESOLVED_CENSOR": int((r1.rr_direction == "UNRESOLVED_CENSOR").sum()),
            "NO_COMPARABLE_TARGET": int((r1.rr_direction == "NO_COMPARABLE_TARGET").sum())},
        roi_gate=gate,
        identity_enum_summary={k: len(v) for k, v in enum.items()},
        governance=dict(trading_metrics="NOT_APPLICABLE",
                        no_pnl=True, no_fvg=True, no_external_internal=True,
                        no_lc=True, atlas_v1_2_unchanged=True),
    )
    json.dump(audit, open(OUT / "DIRECTION_AUDIT.json", "w"),
              indent=2, ensure_ascii=False, default=str)
    print(f"\n[DONE] {time.perf_counter()-t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
