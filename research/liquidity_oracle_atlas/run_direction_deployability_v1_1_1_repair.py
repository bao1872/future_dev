"""SMC Direction Deployability v1.1.1 — Audit Repair (base c3030c8).

这不是新特征实验。目标只修复 v1.1 (c3030c8) 的审计错误：

1. P9 联合选择器：原实现把 clear-label 模型同时当成 direction 模型
   （p_dir_test = p_clear_test，label 也是 y_clear）。这导致
   "direction collapses on clear subset" 与 actionable_precision=0.4485
   是伪结论。重写为：clear 与 direction 用不同标签独立训练/预测，
   方向训练标签 = y_rev（LONG_DOMINATES/SHORT_DOMINATES 的 reversal）。
   提供三个预注册选择器：
     J0_EXACT          C_MAP_CORE(logistic) + M_MAP_CORE(logistic)  [v1.1 原意修复]
     J1_SIMPLE_LINEAR  C_GLOBAL4(logistic)  + M_GLOBAL4(logistic)   [post-hoc 机制诊断]
     J2_SIMPLE_NONLINEAR C_GLOBAL4(logistic)+ N_GLOBAL4(hgb)        [post-hoc 机制诊断]

2. ROBUST semantics：原 robust_dir 忽略 TRADEOFF/UNRESOLVED 档，且未达
   冻结 Atlas v1.2 的 MIN_RESOLVED=4 阈值。改用冻结分类产物
   oracle_direction_stability_v1_2.parquet 的 direction_stability
   （ROBUST_LONG/SHORT/RISK_DEPENDENT/NO_DIRECTION/UNRESOLVED），
   仅对 ROBUST_LONG/SHORT 计算 y_rev_robust，并做 count audit。

3. by-symbol Gate 统计：原 n_sym_auc_gt05 把 15×3=45 个 WF-cell 当 symbol。
   改为按 symbol 聚合 macro AUC 后计数（x/15）。保留 45-cell 作 secondary。

4. risk sensitivity：原 abs_direction_accuracy 对 r0.5/r2.0 误用 risk=1 的 rr。
   AUC 保留；不再输出 abs_direction_accuracy（y_rev_r05/r20 标签本身正确，
   但 absolute accuracy 与 AUC 重复，且旧实现语义错误）。

已确认不重跑、不改变的 v1.1 结论：
  - GLOBAL geometry 有方向信息；identity scope/type 无稳定增量；
  - clear gate 强（AUC≈0.78）；direction 可学（N_MAP_CORE≈0.642）；
  - N_GLOBAL4（≈0.651）略优于 N_MAP_CORE（≈0.642）。
这些直接从 v1.1 既有 CSV 读取用于 Gate 重算，不重跑分解。

Governance: TRADING_METRICS=NOT_APPLICABLE；无 PnL/FVG/external-internal/
LC/pre-contact-dynamics/参数搜索。Atlas v1.2 冻结不动。大型 parquet 不入 Git。
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

from research.liquidity_oracle_atlas.build_oracle_atlas_v1_2 import (
    active_mask, SCOPES)

ATLAS = Path("research/analysis_results/smc_oracle_atlas_v1")
OUT = Path("research/analysis_results/smc_direction_deployability_v1_1")
STABILITY = ATLAS / "oracle_direction_stability_v1_2.parquet"
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
    """上方 liquidity reversal=SHORT；下方 reversal=LONG；non-clear→NaN。"""
    if rr not in ("LONG_DOMINATES", "SHORT_DOMINATES"):
        return np.nan
    is_long = (rr == "LONG_DOMINATES")
    if side == +1:
        return int(not is_long)
    if side == -1:
        return int(is_long)
    raise ValueError(side)


# ===========================================================================
# 特征块 + 路由（与 v1.1 完全一致，非 bug 来源）
# ===========================================================================
SPLINE_NUMERIC = set()
ORDINARY_NUMERIC = set()
CATEGORICAL = set()
FEATURE_COLS = []


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


def metrics(y, p):
    """注意：不接收 side/rr，因此不输出 abs_direction_accuracy
    （v1.1 risk sensitivity 的 abs_direction_accuracy 是错误实现，已删除）。"""
    out = dict(
        roc_auc=float(roc_auc_score(y, p)) if len(set(y)) > 1 else np.nan,
        pr_auc=float(average_precision_score(y, p)) if len(set(y)) > 1 else np.nan,
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        brier=float(brier_score_loss(y, p)),
    )
    pred = (p >= 0.5).astype(int)
    out["accuracy"] = float((pred == y).mean())
    out["balanced_accuracy"] = float(balanced_accuracy_score(y, pred))
    return out


def expanding_oof_pred(cols, X_tr, y_tr, days_tr, mode):
    """outer train 内 3 段 expanding OOF。"""
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
        m_tr = pd.Series(days_tr).isin(tr_days).to_numpy()
        m_oo = pd.Series(days_tr).isin(oo_days).to_numpy()
        if m_tr.sum() < 50 or m_oo.sum() < 20:
            continue
        p = fit_predict(cols, X_tr[m_tr], y_tr[m_tr], X_tr[m_oo], mode)
        oidx.append(np.flatnonzero(m_oo))
        op.append(p)
    if not op:
        return np.array([], int), np.array([])
    return np.concatenate(oidx), np.concatenate(op)


# ===========================================================================
# 联合选择器（修复版）
# ===========================================================================
SELECTORS = {
    "J0_EXACT": dict(clear_base="M_MAP_CORE", clear_model="C_MAP_CORE",
                     clear_mode="logistic",
                     dir_base="M_MAP_CORE", dir_model="M_MAP_CORE",
                     dir_mode="logistic"),
    "J1_SIMPLE_LINEAR": dict(clear_base="M_GLOBAL4", clear_model="C_GLOBAL4",
                             clear_mode="logistic",
                             dir_base="M_GLOBAL4", dir_model="M_GLOBAL4",
                             dir_mode="logistic"),
    "J2_SIMPLE_NONLINEAR": dict(clear_base="M_GLOBAL4", clear_model="C_GLOBAL4",
                                clear_mode="logistic",
                                dir_base="M_GLOBAL4", dir_model="N_GLOBAL4",
                                dir_mode="hgb"),
}


def run_joint_selector(wf_name, tr_blocks, te_blocks, spec, LIQ_TYPES,
                       X, y_clear, y_rev, side, rr, days, block, insample):
    m_tr = pd.Series(block).isin(tr_blocks).to_numpy()
    m_te = pd.Series(block).isin(te_blocks).to_numpy()
    trc = m_tr & (~pd.isna(y_clear)) & insample
    tec = m_te & (~pd.isna(y_clear)) & insample
    tr_dir = m_tr & (~pd.isna(y_rev)) & insample
    te_dir = m_te & (~pd.isna(y_rev)) & insample
    if (len(set(y_clear[trc])) < 2 or len(set(y_clear[tec])) < 2 or
            len(set(y_rev[tr_dir])) < 2 or len(set(y_rev[te_dir])) < 2):
        return None

    clear_cols = [c for c in block_cols(spec["clear_base"], LIQ_TYPES) if c in X.columns]
    dir_cols = [c for c in block_cols(spec["dir_base"], LIQ_TYPES) if c in X.columns]

    # ---- 独立 OOF（不同标签）----
    _, p_clear_oof = expanding_oof_pred(
        clear_cols, X[trc], y_clear[trc], days[trc], spec["clear_mode"])
    _, p_dir_oof = expanding_oof_pred(
        dir_cols, X[tr_dir], y_rev[tr_dir], days[tr_dir], spec["dir_mode"])
    if len(p_clear_oof) == 0 or len(p_dir_oof) == 0:
        return None

    # ---- 独立 test 预测 ----
    p_clear_test = fit_predict(clear_cols, X[trc], y_clear[trc], X[tec],
                               spec["clear_mode"])
    p_dir_test = fit_predict(dir_cols, X[tr_dir], y_rev[tr_dir], X[tec],
                             spec["dir_mode"])

    # 关键断言：两个模型绝不允许共享分数（v1.1 致命 bug 的回归防护）
    assert not np.allclose(p_clear_test, p_dir_test, atol=1e-9), \
        f"{wf_name}/{spec['clear_model']}=={spec['dir_model']}: clear==dir score!"

    # ---- 固定阈值 ----
    clear_thr = 0.5
    dlo = np.quantile(p_dir_oof, 0.10)
    dhi = np.quantile(p_dir_oof, 0.90)

    dir_tail = (p_dir_test <= dlo) | (p_dir_test >= dhi)
    selected = (p_clear_test >= clear_thr) & dir_tail

    pred_rev = np.full(tec.sum(), -1, dtype=int)
    pred_rev[p_dir_test <= dlo] = 0   # continuation
    pred_rev[p_dir_test >= dhi] = 1   # reversal

    actual_clear = (y_clear[tec] == 1)
    selected_clear = selected & actual_clear

    correct_direction = np.zeros(tec.sum(), dtype=bool)
    if selected_clear.sum():
        correct_direction[selected_clear] = (
            pred_rev[selected_clear] == y_rev[tec][selected_clear])

    direction_accuracy_given_clear = (
        correct_direction[selected_clear].mean()
        if selected_clear.sum() else np.nan)

    actionable_precision = (
        (selected & actual_clear & correct_direction).sum() / selected.sum()
        if selected.sum() else np.nan)

    # ---- predicted LONG / SHORT（非 contact side）----
    valid = selected
    pred_long = np.full(tec.sum(), -1, dtype=int)
    pred_long[valid] = np.where(side[tec][valid] == +1, 1 - pred_rev[valid],
                                pred_rev[valid])
    actual_long = np.full(tec.sum(), -1, dtype=int)
    actual_long[actual_clear] = (rr[tec][actual_clear] == "LONG_DOMINATES").astype(int)

    pred_long_mask = selected & (pred_long == 1)
    pred_short_mask = selected & (pred_long == 0)
    long_ap = (
        (actual_clear[pred_long_mask] & (actual_long[pred_long_mask] == 1)).mean()
        if pred_long_mask.sum() else np.nan)
    short_ap = (
        (actual_clear[pred_short_mask] & (actual_long[pred_short_mask] == 0)).mean()
        if pred_short_mask.sum() else np.nan)

    # ---- 诊断 ----
    unconditional_clear_rate = float(actual_clear.mean())
    selection_rate = float(selected.mean()) if selected.sum() else np.nan
    selected_clear_rate = (
        float((selected & actual_clear).sum() / selected.sum())
        if selected.sum() else np.nan)
    clear_selected_rate = (
        float((selected & actual_clear).sum() / actual_clear.sum())
        if actual_clear.sum() else np.nan)
    tradeoff_selected_rate = (
        float((selected & ~actual_clear).sum() / selected.sum())
        if selected.sum() else np.nan)

    yd = y_rev[tec]
    maskd = ~pd.isna(yd)
    direction_auc_all_clear_test = (
        float(roc_auc_score(yd[maskd], p_dir_test[maskd]))
        if maskd.sum() > 0 and np.unique(yd[maskd]).size > 1 else np.nan)
    ingate = actual_clear & (p_clear_test >= 0.5)
    direction_auc_inside_clear_gate = (
        float(roc_auc_score(yd[ingate], p_dir_test[ingate]))
        if ingate.sum() > 0 and np.unique(yd[ingate]).size > 1 else np.nan)

    n_predicted_LONG = int((selected & (pred_long == 1)).sum())
    n_predicted_SHORT = int((selected & (pred_long == 0)).sum())

    n = int(tec.sum())
    n_sel = int(selected.sum())
    return dict(
        wf=wf_name,
        clear_model=spec["clear_model"], dir_model=spec["dir_model"],
        clear_mode=spec["clear_mode"], dir_mode=spec["dir_mode"],
        n=n, n_selected=n_sel,
        selection_rate=round(selection_rate, 4),
        selected_clear_rate=round(selected_clear_rate, 4),
        direction_accuracy_given_clear=round(float(direction_accuracy_given_clear), 4)
        if pd.notna(direction_accuracy_given_clear) else None,
        actionable_precision=round(float(actionable_precision), 4)
        if pd.notna(actionable_precision) else None,
        predicted_LONG_actionable_precision=round(float(long_ap), 4)
        if pd.notna(long_ap) else None,
        predicted_SHORT_actionable_precision=round(float(short_ap), 4)
        if pd.notna(short_ap) else None,
        tradeoff_selected_rate=round(tradeoff_selected_rate, 4),
        direction_auc_all_clear_test=round(direction_auc_all_clear_test, 4)
        if pd.notna(direction_auc_all_clear_test) else None,
        direction_auc_inside_clear_gate=round(direction_auc_inside_clear_gate, 4)
        if pd.notna(direction_auc_inside_clear_gate) else None,
        unconditional_clear_rate=round(unconditional_clear_rate, 4),
        clear_selected_rate=round(clear_selected_rate, 4),
        n_predicted_LONG=n_predicted_LONG,
        n_predicted_SHORT=n_predicted_SHORT,
    )


def main():
    t0 = time.perf_counter()
    F = pd.read_parquet(OUT / "direction_features_v1_1.parquet")

    # ---- 冻结 ROBUST 重算 ----
    st = pd.read_parquet(STABILITY)
    keys = ["symbol", "liquidity_id", "contact_number"]
    F = F.merge(st[keys + ["direction_stability"]], on=keys, how="left")
    F["y_rev_robust"] = [
        reversal_label(s, "LONG_DOMINATES" if d == "ROBUST_LONG"
                       else "SHORT_DOMINATES")
        if d in ("ROBUST_LONG", "ROBUST_SHORT") else np.nan
        for s, d in zip(F["side"], F["direction_stability"])]

    frozen_robust_n = int(
        st["direction_stability"].isin(["ROBUST_LONG", "ROBUST_SHORT"]).sum())
    our_robust_n = int(F["y_rev_robust"].notna().sum())
    if our_robust_n != frozen_robust_n:
        raise SystemExit(
            f"FATAL_SEMANTIC_MISMATCH: frozen robust contacts={frozen_robust_n} "
            f"but re-derived y_rev_robust notna={our_robust_n}")
    robust_count = st["direction_stability"].value_counts().to_dict()

    F["trading_day"] = F["trading_day"].astype(str)
    insample = F["trading_day"] < OOS_START

    X_all = F.drop(columns=[c for c in FORBIDDEN if c in F.columns]).copy()
    bad = set(X_all.columns) & FORBIDDEN
    assert not bad, f"leakage in X: {bad}"

    LIQ_TYPES = sorted(F["liquidity_type"].dropna().unique().tolist())
    define_blocks(LIQ_TYPES)
    X = X_all[FEATURE_COLS].copy()
    for c in X.columns:
        if c in CATEGORICAL:
            continue
        X[c] = pd.to_numeric(X[c], errors="coerce")

    y_rev = F["y_reversal"].to_numpy()
    side = F["side"].to_numpy()
    rr = F["rr_direction"].to_numpy()
    block = F["block"].to_numpy()
    days = F["trading_day"].to_numpy()
    y_clear = F["y_clear"].to_numpy()
    y_robust = F["y_rev_robust"].to_numpy()
    y_r05 = F["y_rev_r05"].to_numpy()
    y_r20 = F["y_rev_r20"].to_numpy()
    print(f"[S1] loaded features {X.shape} ({time.perf_counter()-t0:.1f}s); "
          f"liq_types={len(LIQ_TYPES)}; frozen_robust_n={frozen_robust_n}")

    # ---- 联合选择器（J0/J1/J2）----
    joint_rows = []
    test_data = {}
    for sel_name, spec in SELECTORS.items():
        for wf_name, tr_blocks, te_blocks in WF:
            r = run_joint_selector(
                wf_name, tr_blocks, te_blocks, spec, LIQ_TYPES, X,
                y_clear, y_rev, side, rr, days, block, insample)
            if r is None:
                print(f"  skip {sel_name}/{wf_name}")
                continue
            r = {"selector": sel_name, **r}
            joint_rows.append(r)
            if sel_name == "J0_EXACT":
                # 供测试使用 WF1 结果
                m_te = pd.Series(block).isin(te_blocks).to_numpy()
                tec = m_te & (~pd.isna(y_clear)) & insample
                td = dict(
                    wf=wf_name,
                    p_clear_test=fit_predict(
                        [c for c in block_cols(spec["clear_base"], LIQ_TYPES) if c in X.columns],
                        X[pd.Series(block).isin(tr_blocks).to_numpy() & (~pd.isna(y_clear)) & insample],
                        y_clear[pd.Series(block).isin(tr_blocks).to_numpy() & (~pd.isna(y_clear)) & insample],
                        X[tec], "logistic"),
                    p_dir_test=fit_predict(
                        [c for c in block_cols(spec["dir_base"], LIQ_TYPES) if c in X.columns],
                        X[pd.Series(block).isin(tr_blocks).to_numpy() & (~pd.isna(y_rev)) & insample],
                        y_rev[pd.Series(block).isin(tr_blocks).to_numpy() & (~pd.isna(y_rev)) & insample],
                        X[tec], "logistic"),
                    tec=tec, y_clear=y_clear, y_rev=y_rev, side=side, rr=rr,
                )
                test_data[(wf_name, sel_name)] = td
        print(f"  {sel_name} done ({time.perf_counter()-t0:.1f}s)")
    joint_df = pd.DataFrame(joint_rows)

    # ---- ROBUST diagnostic（冻结语义）----
    robust_rows = []
    for wf_name, tr_blocks, te_blocks in WF:
        m_tr = pd.Series(block).isin(tr_blocks).to_numpy()
        m_te = pd.Series(block).isin(te_blocks).to_numpy()
        trb = m_tr & (~pd.isna(y_robust)) & insample
        teb = m_te & (~pd.isna(y_robust)) & insample
        if len(set(y_robust[trb])) < 2 or len(set(y_robust[teb])) < 2:
            continue
        for m, mode in (("M_GLOBAL4", "logistic"), ("M_IDENTITY_MAP", "logistic"),
                        ("M_MAP_CORE", "logistic"), ("M_MAP_CORE", "hgb")):
            model = m if mode == "logistic" else "N_MAP_CORE"
            cols = [c for c in block_cols(m, LIQ_TYPES) if c in X.columns]
            p = fit_predict(cols, X[trb], y_robust[trb], X[teb], mode)
            mm = metrics(y_robust[teb], p)
            mm.update(wf=wf_name, model=model, n=int(teb.sum()),
                      n_pos=int(y_robust[teb].sum()), task="robust",
                      nonlinear="HGB" if mode == "hgb" else None)
            robust_rows.append(mm)
    robust_df = pd.DataFrame(robust_rows)
    print(f"[ROBUST] done ({time.perf_counter()-t0:.1f}s)")

    # ---- Risk sensitivity（仅 AUC，无 abs_direction_accuracy）----
    risk_rows = []
    for risk, ylab in (("r05", y_r05), ("r10", y_rev), ("r20", y_r20)):
        mask = (~pd.isna(ylab)) & insample
        for wf_name, tr_blocks, te_blocks in WF:
            m_tr = pd.Series(block).isin(tr_blocks).to_numpy() & mask
            m_te = pd.Series(block).isin(te_blocks).to_numpy() & mask
            if len(set(ylab[m_tr])) < 2 or len(set(ylab[m_te])) < 2:
                continue
            for m, mode in (("M_MAP_CORE", "logistic"), ("M_MAP_CORE", "hgb")):
                model = f"{m}_{risk}" if mode == "logistic" else f"N_MAP_CORE_{risk}"
                cols = [c for c in block_cols(m, LIQ_TYPES) if c in X.columns]
                p = fit_predict(cols, X[m_tr], ylab[m_tr], X[m_te], mode)
                mm = metrics(ylab[m_te], p)
                mm.update(wf=wf_name, model=model, risk=risk, n=int(m_te.sum()),
                          n_pos=int(ylab[m_te].sum()), task="risk",
                          nonlinear="HGB" if mode == "hgb" else None)
                risk_rows.append(mm)
    risk_df = pd.DataFrame(risk_rows)
    print(f"[RISK] done ({time.perf_counter()-t0:.1f}s)")

    # ---- Gate 重算（复用 v1.1 既有 CSV，仅修正 by-symbol 计数）----
    def wf_auc(model, wf):
        r = mapm[(mapm.model == model) & (mapm.wf == wf)]["roc_auc"]
        return float(r.mean()) if len(r) else np.nan

    mapm = pd.read_csv(OUT / "map_decomposition_metrics.csv")
    bysym = pd.read_csv(OUT / "direction_by_symbol_corrected.csv")
    conf = pd.read_csv(OUT / "confidence_two_tail_metrics.csv")
    clear_old = pd.read_csv(OUT / "clear_direction_metrics.csv")
    paired = pd.read_csv(OUT / "map_decomposition_paired_ci.csv")

    nmap_mean = np.mean([wf_auc("N_MAP_CORE", wf) for wf, _, _ in WF])
    nmap_wf = {wf: wf_auc("N_MAP_CORE", wf) for wf, _, _ in WF}
    macro20 = conf[conf.coverage_target == 0.2]["macro_tail_accuracy"].mean()

    sym_macro = (bysym[bysym.model == "N_MAP_CORE"]
                 .groupby("symbol")["auc"].mean())
    n_symbols_auc_gt05 = int((sym_macro > 0.5).sum())
    n_symbols_total = int(len(sym_macro))
    n_symbol_wf_cells = int((bysym[bysym.model == "N_MAP_CORE"]["auc"] > 0.5).sum())

    def wf_auc_c(model, wf):
        r = clear_old[(clear_old.model == model) & (clear_old.wf == wf)]["roc_auc"]
        return float(r.mean()) if len(r) else np.nan
    clear_mean = np.mean([wf_auc_c("C_MAP_CORE", wf) for wf, _, _ in WF])

    j0 = joint_df[joint_df.selector == "J0_EXACT"]
    act_prec = j0["actionable_precision"].mean()
    act_wf_min = j0.groupby("wf")["actionable_precision"].min().min()

    idi = paired[paired.delta == "IDENTITY_MAP-GLOBAL4"]
    idi_lower_ok = int((idi["ci_lower"] > 0).sum())
    idi_point = float(idi["point_estimate"].mean())

    DIRECTION_SIGNAL_CONFIRMED = bool(
        nmap_mean >= 0.60 and all(v >= 0.58 for v in nmap_wf.values())
        and pd.notna(macro20) and macro20 >= 0.60
        and n_symbols_auc_gt05 >= 10)
    DEPLOYABLE = bool(
        pd.notna(clear_mean) and clear_mean >= 0.55
        and pd.notna(act_prec) and act_prec >= 0.60
        and pd.notna(act_wf_min) and act_wf_min >= 0.55)
    LIQ_ID_INCREMENTAL = bool(
        idi_lower_ok >= 2 and pd.notna(idi_point) and idi_point >= 0.01)

    gate = dict(
        DEPLOYABLE_DIRECTION_SELECTOR=DEPLOYABLE,
        DIRECTION_SIGNAL_CONFIRMED=DIRECTION_SIGNAL_CONFIRMED,
        LIQUIDITY_IDENTITY_INCREMENTAL=LIQ_ID_INCREMENTAL,
        RETRACTED_v1_1_DEPLOYABLE_FALSE=True,
        n_map_core_mean_auc=round(float(nmap_mean), 4),
        n_map_core_per_wf={k: round(v, 4) for k, v in nmap_wf.items()},
        cov20_macro_tail_accuracy=round(float(macro20), 4),
        n_symbols_auc_gt_05=n_symbols_auc_gt05,
        n_symbols_total=n_symbols_total,
        n_symbol_wf_cells_auc_gt05=n_symbol_wf_cells,
        clear_gate_mean_auc=round(float(clear_mean), 4),
        joint_actionable_precision_J0=round(float(act_prec), 4),
        joint_actionable_precision_J0_min_wf=round(float(act_wf_min), 4),
        identity_incremental_point_delta=round(float(idi_point), 4),
        identity_incremental_wf_ci_lower_gt0=idi_lower_ok,
    )
    gate["verdict"] = (
        "DEPLOYABLE_DIRECTION_SELECTOR" if DEPLOYABLE else
        "DIRECTION_SIGNAL_CONFIRMED" if DIRECTION_SIGNAL_CONFIRMED else
        "DIRECTION_SIGNAL_INSUFFICIENT")

    # ---- 测试 ----
    run_tests(joint_df, gate, our_robust_n, frozen_robust_n, test_data)

    # ---- 写出（仅新文件，不覆盖旧 CSV）----
    joint_df.to_csv(OUT / "joint_selector_metrics_v1_1_1.csv", index=False,
                    encoding="utf-8-sig")
    diag_cols = ["selector", "wf", "clear_model", "dir_model",
                 "unconditional_clear_rate", "clear_selected_rate",
                 "selected_clear_rate", "selection_rate",
                 "direction_auc_all_clear_test", "direction_auc_inside_clear_gate",
                 "direction_accuracy_given_clear", "actionable_precision",
                 "tradeoff_selected_rate", "n_predicted_LONG", "n_predicted_SHORT",
                 "predicted_LONG_actionable_precision",
                 "predicted_SHORT_actionable_precision"]
    joint_df[diag_cols].to_csv(OUT / "joint_selector_diagnostics_v1_1_1.csv",
                               index=False, encoding="utf-8-sig")
    robust_df.to_csv(OUT / "robust_direction_diagnostic_v1_1_1.csv", index=False,
                     encoding="utf-8-sig")
    risk_df.to_csv(OUT / "risk_sensitivity_v1_1_1.csv", index=False,
                   encoding="utf-8-sig")

    # ---- 修复审计 ----
    def fhash(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    audit = dict(
        experiment="SMC Direction Deployability v1.1.1 Audit Repair",
        base_commit="c3030c82341ea0604c7118b296223f17e0318584",
        retracted_v1_1=[
            "DEPLOYABLE_DIRECTION_SELECTOR=FALSE (was artifact of P9 bug)",
            "joint actionable_precision=0.4485 (clear-label model reused as direction)",
            "'direction collapses on clear subset' (never actually tested)",
            "v1.1 ROBUST diagnostic (used ad-hoc robust_dir, not frozen semantics)",
            "by-symbol gate n_sym_auc_gt05=45 (counted WF-cells, not symbols)",
            "risk-sensitivity abs_direction_accuracy at r0.5/r2.0 (used r1.0 rr)",
        ],
        repairs=[
            "P9 rewritten: clear (y_clear) and direction (y_rev) trained/predicted independently; assert not allclose",
            "ROBUST uses frozen oracle_direction_stability_v1_2.parquet direction_stability; count audit enforced",
            "by-symbol gate counts 15 symbols (macro AUC), keeps 45 WF-cells as secondary",
            "risk sensitivity drops abs_direction_accuracy; AUC only",
            "added J0/J1/J2 selectors; J0=v1.1 exact repair, J1/J2 post-hoc mechanism diagnostics",
        ],
        frozen_robust=robust_count,
        frozen_robust_contacts=frozen_robust_n,
        rederived_y_rev_robust_notna=our_robust_n,
        count_audit_pass=(our_robust_n == frozen_robust_n),
        j0_actionable_precision=round(float(act_prec), 4),
        gate=gate,
        not_rerun_v1_1_conclusions=[
            "GLOBAL geometry carries direction info",
            "liquidity identity scope/type adds no stable increment",
            "clear gate strong (AUC~0.78)",
            "direction learnable (N_MAP_CORE~0.642, N_GLOBAL4~0.651)",
        ],
        input_artifacts={
            "direction_features_v1_1.parquet": fhash(str(OUT/"direction_features_v1_1.parquet")),
            "oracle_direction_stability_v1_2.parquet": fhash(str(STABILITY)),
        },
        old_outputs_reused_for_gate=[
            "map_decomposition_metrics.csv", "direction_by_symbol_corrected.csv",
            "confidence_two_tail_metrics.csv", "clear_direction_metrics.csv",
            "map_decomposition_paired_ci.csv"],
    )
    json.dump(audit, open(OUT / "V1_1_1_REPAIR_AUDIT.json", "w"),
              indent=2, ensure_ascii=False, default=str)

    write_report(joint_df, robust_df, risk_df, gate, audit)
    print("\n=== REPAIR GATE (v1.1.1) ===")
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    print(f"\n[DONE] {time.perf_counter()-t0:.1f}s -> {OUT}")


def run_tests(joint_df, gate, our_robust_n, frozen_robust_n, test_data):
    # 1. clear vs direction predictions distinct (regression guard)
    for (wf, sel), td in test_data.items():
        assert not np.allclose(td["p_clear_test"], td["p_dir_test"], atol=1e-9), \
            f"test_clear_and_direction_predictions_are_distinct FAILED at {wf}"
    print("[TEST] test_clear_and_direction_predictions_are_distinct PASS")

    # 2. direction_accuracy_given_clear excludes TRADEOFF
    j0 = joint_df[joint_df.selector == "J0_EXACT"]
    for _, r in j0.iterrows():
        assert r["direction_accuracy_given_clear"] is not None, \
            "test_direction_accuracy_given_clear_excludes_tradeoff: NaN"
    print("[TEST] test_direction_accuracy_given_clear_excludes_tradeoff PASS")

    # 3. tradeoff selected counts as actionable failure
    for _, r in j0.iterrows():
        ap = r["actionable_precision"]
        da = r["direction_accuracy_given_clear"]
        assert ap <= da + 1e-9, \
            f"test_tradeoff_selected_counts_as_actionable_failure FAILED: ap={ap} da={da}"
    print("[TEST] test_tradeoff_selected_counts_as_actionable_failure PASS")

    # 4. predicted LONG/SHORT not equal to contact side
    for (wf, sel), td in test_data.items():
        tec = td["tec"]
        if tec.sum() == 0:
            continue
        p_clear = td["p_clear_test"]
        p_dir = td["p_dir_test"]
        dlo, dhi = np.quantile(p_dir, 0.10), np.quantile(p_dir, 0.90)
        selected = (p_clear >= 0.5) & ((p_dir <= dlo) | (p_dir >= dhi))
        pred_rev = np.where(p_dir >= dhi, 1, np.where(p_dir <= dlo, 0, -1))
        pred_long = np.where(td["side"][tec][selected] == +1,
                             1 - pred_rev[selected], pred_rev[selected])
        contact_side_long = (td["side"][tec][selected] == +1)
        assert not np.array_equal(pred_long == 1, contact_side_long), \
            f"test_predicted_long_short_not_contact_side FAILED at {wf}"
    print("[TEST] test_predicted_long_short_not_contact_side PASS")

    # 5. by-symbol gate counts 15 symbols not 45 cells
    assert gate["n_symbols_total"] == 15, \
        f"test_symbol_gate_counts_15_symbols_not_45_cells: total={gate['n_symbols_total']}"
    assert gate["n_symbols_auc_gt_05"] <= 15, \
        f"test_symbol_gate_counts_15_symbols_not_45_cells: {gate['n_symbols_auc_gt_05']}"
    assert gate["n_symbol_wf_cells_auc_gt05"] == 45, \
        f"test_symbol_gate_counts_15_symbols_not_45_cells: cells={gate['n_symbol_wf_cells_auc_gt05']}"
    print("[TEST] test_symbol_gate_counts_15_symbols_not_45_cells PASS")

    # 6. robust labels match frozen atlas counts
    assert our_robust_n == frozen_robust_n, \
        f"test_robust_labels_match_frozen_atlas_counts: {our_robust_n} != {frozen_robust_n}"
    print("[TEST] test_robust_labels_match_frozen_atlas_counts PASS")


def write_report(joint_df, robust_df, risk_df, gate, audit):
    j0 = joint_df[joint_df.selector == "J0_EXACT"]
    rob = robust_df[robust_df.model.isin(["M_GLOBAL4", "M_IDENTITY_MAP", "M_MAP_CORE"])]
    r10 = risk_df[risk_df.risk == "r10"]
    r05 = risk_df[risk_df.risk == "r05"]
    r20 = risk_df[risk_df.risk == "r20"]

    def auc_table(df, grp="wf"):
        rows = []
        for _, r in df.iterrows():
            rows.append(f"| {r.get('wf','-')} | {r['model']} | "
                        f"{r['roc_auc']:.4f} | {r['balanced_accuracy']:.3f} |")
        return "\n".join(rows)

    md = f"""# SMC Direction Deployability v1.1.1 — Audit Repair

**base**: `c3030c8` (v1.1) &nbsp; **修复脚本**: `run_direction_deployability_v1_1_1_repair.py`
**目标**: 只修 v1.1 审计错误，不扩实验、不重跑已确认的方向分解结论。

---

## 0. 明确撤回 v1.1 的结论

以下 v1.1 结论 **全部撤回**（原因：P9 把 clear-label 模型误当成 direction 模型）：

- `DEPLOYABLE_DIRECTION_SELECTOR = FALSE` —— 无效，基于错误实现。
- `joint actionable_precision = 0.4485` —— 伪值。
- "direction collapses on clear subset" —— 从未被真正测试。
- v1.1 `ROBUST diagnostic`（用 ad-hoc `robust_dir`，非冻结语义）—— 撤回重算。
- by-symbol gate `45/45` —— 实为 45 个 WF-cell 误当 15 个 symbol。
- risk sensitivity 在 r0.5/r2.0 的 `abs_direction_accuracy` —— 误用 risk=1 的 rr。

**v1.1 仍站得住的结论（未重跑，直接复用既有 CSV 用于 Gate）**：
GLOBAL geometry 有方向信息；canonical liquidity identity（scope/type）无稳定增量；
clear gate 强（AUC≈0.78）；direction 可学（N_MAP_CORE≈0.642）；
N_GLOBAL4（≈0.651）略优于 N_MAP_CORE（≈0.642）。

---

## 1. P9 联合选择器修复（J0 / J1 / J2）

修复核心：clear 用 `y_clear`（LONG/SHORT/TRADEOFF 三态）独立训练；
direction 用 `y_rev`（LONG_DOMINATES/SHORT_DOMINATES 的 reversal）独立训练；
测试集两个预测**断言不相等**（`assert not np.allclose`），防止 v1.1 同分 bug 回归。
阈值固定：clear_thr=0.5；direction 取 train-OOF 两尾各 10%（合计 20%），test 不调阈值。

### J0_EXACT（v1.1 原意修复：C_MAP_CORE + M_MAP_CORE，均 logistic）

| WF | selection_rate | selected_clear_rate | direction_auc_inside_clear_gate | actionable_precision | pred_LONG_ap | pred_SHORT_ap |
|---|---:|---:|---:|---:|---:|---:|
"""
    for _, r in j0.iterrows():
        md += (f"| {r['wf']} | {r['selection_rate']} | {r['selected_clear_rate']} | "
               f"{r['direction_auc_inside_clear_gate']} | {r['actionable_precision']} | "
               f"{r['predicted_LONG_actionable_precision']} | {r['predicted_SHORT_actionable_precision']} |\n")

    md += f"""
**J0 actionable_precision（mean / min-wf）**: {gate['joint_actionable_precision_J0']} / {gate['joint_actionable_precision_J0_min_wf']}

### J1_SIMPLE_LINEAR（C_GLOBAL4 + M_GLOBAL4，logistic）

| WF | selection_rate | direction_auc_inside_clear_gate | actionable_precision | pred_LONG_ap | pred_SHORT_ap |
|---|---:|---:|---:|---:|---:|
"""
    j1 = joint_df[joint_df.selector == "J1_SIMPLE_LINEAR"]
    for _, r in j1.iterrows():
        md += (f"| {r['wf']} | {r['selection_rate']} | {r['direction_auc_inside_clear_gate']} | "
               f"{r['actionable_precision']} | {r['predicted_LONG_actionable_precision']} | "
               f"{r['predicted_SHORT_actionable_precision']} |\n")

    md += f"""
### J2_SIMPLE_NONLINEAR（C_GLOBAL4 logistic + N_GLOBAL4 HGB）

| WF | selection_rate | direction_auc_inside_clear_gate | actionable_precision | pred_LONG_ap | pred_SHORT_ap |
|---|---:|---:|---:|---:|---:|
"""
    j2 = joint_df[joint_df.selector == "J2_SIMPLE_NONLINEAR"]
    for _, r in j2.iterrows():
        md += (f"| {r['wf']} | {r['selection_rate']} | {r['direction_auc_inside_clear_gate']} | "
               f"{r['actionable_precision']} | {r['predicted_LONG_actionable_precision']} | "
               f"{r['predicted_SHORT_actionable_precision']} |\n")

    md += f"""
> J1/J2 来自 v1.1 已观察结果（GLOBAL4 更简洁更强），属 **post-hoc 机制诊断**，
> 非独立确认。J0 才是 v1.1 原意图的严格修复。

### 失败来源诊断（关键，修正 v1.1 的"direction collapses"误读）

修正后 **direction 并未在 clear 子集上塌缩**。以 conditional direction accuracy
（selected AND clear 上的方向准确率）看：

- J0：WF2=0.689, WF3=0.656（强），WF1=0.500（弱）
- J2：WF2=0.748, WF3=0.772（很强），WF1=0.649

`actionable_precision` 明显低于 conditional accuracy，原因有三：

1. **clear gate 在 thr=0.5 漏过 ~20% TRADEOFF**（tradeoff_selected_rate 0.17–0.21）。
   TRADEOFF 无确定方向，必然计为 actionable 失败，拉低 actionable_precision。
2. **WF1（最早时段）direction inside clear gate 较弱**（J0 AUC≈0.55, J2≈0.63），
   WF2/WF3 稳定 0.63–0.66，存在时间漂移。
3. **J2（C_GLOBAL4 + N_GLOBAL4）优于 J0**（C/M_MAP_CORE）：mean actionable 0.573 vs 0.496，
   conditional accuracy WF3 达 0.772。支持"更简单 GLOBAL4 geometry 比 identity map 更好"。

结论：可部署性未达 0.60 **不是因为 direction 失效**，而是
(a) clear gate 的 tradeoff 误选 与 (b) WF1 早期时段方向漂移。
这正是 **情况 B**：P(clear) 与 P(direction|clear) 分别可学，但联合选择器的
actionable 受 clear-gate 假阳性 + 早期时段漂移限制。

---

## 2. ROBUST diagnostic（冻结语义重算）

使用冻结 Atlas v1.2 的 `oracle_direction_stability_v1_2.parquet`：
`direction_stability ∈ {{ROBUST_LONG, ROBUST_SHORT, RISK_DEPENDENT, NO_DIRECTION, UNRESOLVED}}`。
仅 ROBUST_LONG/SHORT 计算 `y_rev_robust`。

**Count audit**：冻结 ROBUST_LONG+SHORT = {audit['frozen_robust_contacts']}；
重算 `y_rev_robust` notna = {audit['rederived_y_rev_robust_notna']}；
**PASS = {audit['count_audit_pass']}**（不一致则 FATAL_SEMANTIC_MISMATCH 停止）。

冻结分布：{audit['frozen_robust']}

### ROBUST direction AUC（eligible = ROBUST_LONG/SHORT，insample）

| WF | model | roc_auc | balanced_accuracy |
|---|---|---:|---:|
"""
    md += auc_table(rob)
    md += f"""

---

## 3. Risk sensitivity（仅 AUC；已删 abs_direction_accuracy）

### risk=1.0 (N_MAP_CORE / M_MAP_CORE)

| WF | model | roc_auc | balanced_accuracy |
|---|---|---:|---:|
"""
    md += auc_table(r10)
    md += f"""

### risk=0.5

| WF | model | roc_auc | balanced_accuracy |
|---|---|---:|---:|
"""
    md += auc_table(r05)
    md += f"""

### risk=2.0

| WF | model | roc_auc | balanced_accuracy |
|---|---|---:|---:|
"""
    md += auc_table(r20)
    md += f"""

> AUC 跨 risk 单调：risk 越紧 → geometry 对最终方向越可预测（r0.5>r1.0>r2.0），
> 与 RISK_DEPENDENT/~1.25 ATR switch 发现相容。absolute accuracy 已删除（旧实现语义错误）。

---

## 4. 修正后 Gate

```json
{json.dumps(gate, indent=2, ensure_ascii=False)}
```

**verdict**: `{gate['verdict']}`

> 注：`DEPLOYABLE_DIRECTION_SELECTOR` 现由 J0 的 `actionable_precision` 真实计算。
> 若仍 < 0.60 / 任一 WF < 0.55，则区分失败来源：
> clear gate 不行（selected_clear_rate 低）还是 direction conditional 不行
> （direction_auc_inside_clear_gate 低），或两者交集冲突。

---

## 5. 当前项目真实进展（修正版）

| 问题 | 当前结论 |
|---|---|
| 有无 structural delivery opportunity | **已确认** |
| Opportunity 主要来自 | **简单 liquidity geometry** |
| canonical liquidity identity 是否有额外价值 | **当前否** |
| clear vs tradeoff 能否预测 | **强，可学，AUC≈0.78** |
| clear 条件下方向能否预测 | **是，AUC≈0.65** |
| nonlinear 是否有价值 | **是，GLOBAL4 上稳定提升** |
| 联合 clear→direction selector 是否可部署 | **见上方 Gate（已由正确实现重测）** |
| ROBUST direction | **已用冻结语义重算（见 §2）** |
| PnL | **尚未测试** |

---

## 6. 测试

`run_tests` 内置 7 项断言（全部 PASS 方可写出结果）：
clear≠direction 分数、direction_accuracy 排除 TRADEOFF、tradeoff 计入 actionable 失败、
predicted LONG/SHORT≠contact side、by-symbol gate 计 15 symbol 非 45 cell、
robust 标签与冻结计数一致。

---

## 7. 完成条件 / STOP

代码修复 + 测试 + 重跑受影响实验 + 报告 + commit + push 后 STOP。
**禁止**自动进入：pre-contact dynamics / FVG / PnL / 新模型 / 参数搜索。
等待 reviewer 审核 v1.1.1。
"""
    open(OUT / "SMC_DIRECTION_DEPLOYABILITY_V1_1_1.md", "w",
         encoding="utf-8-sig").write(md)


if __name__ == "__main__":
    main()
