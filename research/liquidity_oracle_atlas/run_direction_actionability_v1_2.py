"""SMC Direction Actionability Architecture v1.2 (base 26f4785).

目标：不新增任何市场特征，只把 v1.1.1 已确认的两个强信号
（Clear/Tradeoff gate AUC≈0.78；Direction given clear AUC≈0.65）组合成
一个具有足够 precision 的可行动选择器。

三个子问题（与用户规格一致）：
  P1  Minimal-G4：direction 模型能否压缩成接近纯 4 距离几何。
  P2  Calibrated two-stage：用 train-OOF 设 precision-first clear gate，
      C_GLOBAL4(logistic) + N_GLOBAL4(hgb)，解决 ~20% TRADEOFF contamination。
  P3  Direct 3-class：直接预测 REVERSAL/CONTINUATION/TRADEOFF，对照两阶段。
  P4  Product diagnostic（非 Gate）：p_clear × p_rev 概率乘积对照。
  P5  Gate：ACTIONABLE_DIRECTION_PRESENT。
  P7  按 symbol 描述（n_selected>=50 才报告）。
  P8  Bootstrap：仅当 Gate 候选通过，对比 J2_FIXED_0.5。

Governance: TRADING_METRICS=NOT_APPLICABLE（机制/架构验证，未定义交易动作，
不产出 PnL/win_rate/expectancy）。禁止 FVG/external-internal/pre-contact
dynamics/OB 扩展/新 risk 搜索/stop-target 优化/参数搜索/LightGBM 调参/SHAP/LC/
prospective OOS（trading_day>=2026-09-07 完全不用）。

复用 v1.1.1 已审计的 helper（define_blocks/block_cols/make_preprocessor/
fit_predict/expanding_oof_pred/reversal_label），保证与已审查版本一致。
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

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

import research.liquidity_oracle_atlas.run_direction_deployability_v1_1_1_repair as rp

OUT = Path("research/analysis_results/smc_direction_actionability_v1_2")
FEATURES_SRC = Path(
    "research/analysis_results/smc_direction_deployability_v1_1/"
    "direction_features_v1_1.parquet")
OUT.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["AG", "AU", "CU", "AL", "SN", "NI", "RB", "I", "SC", "RU", "MA",
           "TA", "M", "P", "CF"]
PRIMARY_RISK = 1.0
OOS_START = "2026-09-07"
BLOCKS = ["TB1", "TB2", "TB3", "TB4"]
WF = [("WF1", ["TB1"], ["TB2"]),
      ("WF2", ["TB1", "TB2"], ["TB3"]),
      ("WF3", ["TB1", "TB2", "TB3"], ["TB4"])]

TARGET_CLEAR_PRECISION = 0.85
MIN_OOF_SELECTION_RATE = 0.05
DIR_TAIL_LO, DIR_TAIL_HI = 0.10, 0.90

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


# --------------------------------------------------------------------------
# 本地多类 helper（复用 rp 的 preprocessor / blocks）
# --------------------------------------------------------------------------
def fit_predict_multi(cols, Xtr, ytr, Xte, mode):
    pre = rp.make_preprocessor(cols)
    if mode == "logistic":
        clf = LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs")
    else:
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                            max_iter=200, l2_regularization=1.0,
                                            random_state=42)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xte)


def expanding_oof_pred_multi(cols, X_tr, y_tr, days_tr, mode):
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
        if len(set(y_tr[m_tr])) < 3 or len(set(y_tr[m_oo])) < 2:
            continue
        P = fit_predict_multi(cols, X_tr[m_tr], y_tr[m_tr], X_tr[m_oo], mode)
        oidx.append(np.flatnonzero(m_oo))
        op.append(P)
    if not op:
        return np.array([], int), np.empty((0, 3))
    return np.concatenate(oidx), np.concatenate(op, axis=0)


def three_class_label(side, rr):
    """0=CONTINUATION,1=REVERSAL(=clear reversal label),2=TRADEOFF; 其余 NaN。"""
    out = np.full(len(rr), np.nan)
    is_long = (rr == "LONG_DOMINATES")
    clr = np.isin(rr, ["LONG_DOMINATES", "SHORT_DOMINATES"])
    out[clr & (side == +1)] = (~is_long[clr & (side == +1)]).astype(int)
    out[clr & (side == -1)] = is_long[clr & (side == -1)].astype(int)
    out[rr == "TRADEOFF_OR_OVERLAP"] = 2.0
    return out


def choose_clear_threshold(y, p, target_precision=0.85, min_selection_rate=0.05):
    """train-OOF 中达到 target precision 的最低概率阈值（最大 coverage）。"""
    order = np.argsort(-np.asarray(p))
    yy = np.asarray(y)[order].astype(int)
    pp = np.asarray(p)[order]
    cum_tp = np.cumsum(yy == 1)
    nn = np.arange(1, len(yy) + 1)
    precision = cum_tp / nn
    coverage = nn / len(yy)
    ok = (precision >= target_precision) & (coverage >= min_selection_rate)
    if not np.any(ok):
        return None
    k = np.flatnonzero(ok)[-1]
    return float(pp[k])


# --------------------------------------------------------------------------
# 选择器评估（统一接口：给定 selected 布尔数组）
# --------------------------------------------------------------------------
def eval_selection(selected, pred_rev, p_dir_test, p_clear_test, tec, y_dir,
                   y_clear, side, rr, symbol):
    """selected: 测试集(tec)布尔数组；pred_rev: 0/1/NaN(-1) 方向预测；
    p_dir_test: 方向连续分数（用于方向 AUC）；p_clear_test: clear 概率（用于 ingate AUC）；
    y_dir: clear 子集方向标签(0/1, NaN=非clear)；y_clear: 1/0/NaN。
    对 3-class：y_dir 用 y3(0/1/2)，actual_clear=y3!=2，p_clear_test=None。"""
    n = len(selected)
    actual_clear = (y_clear[tec] == 1) if y_clear is not None else (y_dir[tec] != 2)
    selected_clear = selected & actual_clear
    correct_direction = np.zeros(n, bool)
    if selected_clear.sum():
        correct_direction[selected_clear] = (
            pred_rev[selected_clear] == y_dir[tec][selected_clear])
    direction_accuracy_given_clear = (
        float(correct_direction[selected_clear].mean())
        if selected_clear.sum() else np.nan)
    actionable_precision = (
        float((selected & actual_clear & correct_direction).sum()
              / selected.sum())
        if selected.sum() else np.nan)

    pred_long = np.full(n, -1, int)
    sv = selected
    pred_long[sv] = np.where(side[tec][sv] == +1, 1 - pred_rev[sv], pred_rev[sv])
    actual_long = np.full(n, -1, int)
    actual_long[actual_clear] = (rr[tec][actual_clear] == "LONG_DOMINATES").astype(int)
    plm = selected & (pred_long == 1)
    psm = selected & (pred_long == 0)
    long_ap = (float((actual_clear[plm] & (actual_long[plm] == 1)).mean())
               if plm.sum() else np.nan)
    short_ap = (float((actual_clear[psm] & (actual_long[psm] == 0)).mean())
                if psm.sum() else np.nan)

    yd = y_dir[tec]
    maskd = ~pd.isna(yd)
    dauc_mask = maskd & (yd != 2)  # 方向 AUC 只在 clear 子集(0/1)上计算
    dauc_all = (float(roc_auc_score(yd[dauc_mask], p_dir_test[dauc_mask]))
                if dauc_mask.sum() > 0 and np.unique(yd[dauc_mask]).size > 1
                else np.nan)
    if p_clear_test is not None:
        ingate = actual_clear & (p_clear_test >= 0.5)
    else:
        ingate = actual_clear
    dauc_in = (float(roc_auc_score(yd[ingate], p_dir_test[ingate]))
               if ingate.sum() > 0 and np.unique(yd[ingate]).size > 1 else np.nan)
    tradeoff_rate = (float((selected & ~actual_clear).sum() / selected.sum())
                     if selected.sum() else np.nan)
    return dict(
        n=int(n), n_selected=int(selected.sum()),
        selection_rate=float(selected.mean()) if n else np.nan,
        selected_clear_rate=(float((selected & actual_clear).sum() / selected.sum())
                             if selected.sum() else np.nan),
        clear_selected_rate=(float((selected & actual_clear).sum() / actual_clear.sum())
                             if actual_clear.sum() else np.nan),
        direction_auc_all_clear_test=dauc_all,
        direction_auc_inside_clear_gate=dauc_in,
        direction_accuracy_given_clear=direction_accuracy_given_clear,
        actionable_precision=actionable_precision,
        tradeoff_selected_rate=tradeoff_rate,
        predicted_LONG_actionable_precision=long_ap,
        predicted_SHORT_actionable_precision=short_ap,
        n_predicted_LONG=int((selected & (pred_long == 1)).sum()),
        n_predicted_SHORT=int((selected & (pred_long == 0)).sum()),
    ), dict(selected=selected, actual_clear=actual_clear,
           correct_direction=correct_direction, pred_long=pred_long,
           pred_rev=pred_rev, symbol=symbol[tec], day=None,
           y_dir=y_dir[tec], side=side[tec])


# --------------------------------------------------------------------------
# P2 calibrated two-stage
# --------------------------------------------------------------------------
def run_calibrated_two_stage(wf_name, tr_blocks, te_blocks, clear_cols, dir_cols,
                             clear_mode, dir_mode, LIQ_TYPES, X, y_clear, y_rev,
                             side, rr, days, block, insample,
                             fixed_clear_thr=None, tag="CALIBRATED_TWO_STAGE"):
    m_tr = pd.Series(block).isin(tr_blocks).to_numpy()
    m_te = pd.Series(block).isin(te_blocks).to_numpy()
    trc = m_tr & (~pd.isna(y_clear)) & insample
    tec = m_te & (~pd.isna(y_clear)) & insample
    tr_dir = m_tr & (~pd.isna(y_rev)) & insample
    te_dir = m_te & (~pd.isna(y_rev)) & insample
    if (len(set(y_clear[trc])) < 2 or len(set(y_clear[tec])) < 2 or
            len(set(y_rev[tr_dir])) < 2 or len(set(y_rev[te_dir])) < 2):
        return None

    oidx_c, p_clear_oof = rp.expanding_oof_pred(
        clear_cols, X[trc], y_clear[trc], days[trc], clear_mode)
    oidx_d, p_dir_oof = rp.expanding_oof_pred(
        dir_cols, X[tr_dir], y_rev[tr_dir], days[tr_dir], dir_mode)
    if len(p_clear_oof) == 0 or len(p_dir_oof) == 0:
        return None
    # 注意：clear-OOF（用于 clear 阈值）与 dir-OOF（用于方向尾阈值）各自独立使用，
    # 资格 mask 不同（y_clear vs y_rev），oidx 无需对齐。

    if fixed_clear_thr is not None:
        clear_thr = float(fixed_clear_thr)
        clear_thr_unavailable = False
    else:
        clear_thr = choose_clear_threshold(
            y_clear[trc][oidx_c], p_clear_oof,
            TARGET_CLEAR_PRECISION, MIN_OOF_SELECTION_RATE)
        clear_thr_unavailable = (clear_thr is None)

    dlo = np.quantile(p_dir_oof, DIR_TAIL_LO)
    dhi = np.quantile(p_dir_oof, DIR_TAIL_HI)

    p_clear_test = rp.fit_predict(
        clear_cols, X[trc], y_clear[trc], X[tec], clear_mode)
    p_dir_test = rp.fit_predict(
        dir_cols, X[tr_dir], y_rev[tr_dir], X[tec], dir_mode)
    assert not np.allclose(p_clear_test, p_dir_test, atol=1e-9)

    if clear_thr_unavailable:
        selected = np.zeros(tec.sum(), bool)
        pred_rev = np.full(tec.sum(), -1, int)
    else:
        dir_tail = (p_dir_test <= dlo) | (p_dir_test >= dhi)
        selected = (p_clear_test >= clear_thr) & dir_tail
        pred_rev = np.full(tec.sum(), -1, int)
        pred_rev[p_dir_test <= dlo] = 0
        pred_rev[p_dir_test >= dhi] = 1

    m, extra = eval_selection(selected, pred_rev, p_dir_test, p_clear_test, tec,
                              y_rev, y_clear, side, rr, SYM_ARRAY)
    m["clear_thr"] = "UNAVAILABLE" if clear_thr_unavailable else round(clear_thr, 4)
    m["clear_thr_unavailable"] = bool(clear_thr_unavailable)
    m["wf"] = wf_name
    m["tag"] = tag
    m["dir_tail_lo"] = round(float(dlo), 4)
    m["dir_tail_hi"] = round(float(dhi), 4)
    extra["p_clear_test"] = p_clear_test
    extra["p_dir_test"] = p_dir_test
    extra["day"] = DAYS_ARRAY[tec]
    return m, extra, (oidx_c, p_clear_oof, p_dir_oof)


# --------------------------------------------------------------------------
# P3 direct 3-class
# --------------------------------------------------------------------------
def run_three_class(wf_name, tr_blocks, te_blocks, cols, mode, LIQ_TYPES, X, y3,
                    side, rr, days, block, insample, coverages=(0.10, 0.20, 0.30)):
    m_tr = pd.Series(block).isin(tr_blocks).to_numpy()
    m_te = pd.Series(block).isin(te_blocks).to_numpy()
    tr3 = m_tr & (~pd.isna(y3)) & insample
    te3 = m_te & (~pd.isna(y3)) & insample
    if len(set(y3[tr3])) < 3 or len(set(y3[te3])) < 2:
        return None

    oidx, P_oof = expanding_oof_pred_multi(
        cols, X[tr3], y3[tr3].astype(int), days[tr3], mode)
    if len(P_oof) == 0:
        return None
    p_cont_oof, p_rev_oof, p_tradeoff_oof = P_oof[:, 0], P_oof[:, 1], P_oof[:, 2]
    action_margin_oof = np.maximum(p_cont_oof, p_rev_oof) - p_tradeoff_oof
    thresholds = {cv: float(np.quantile(action_margin_oof, 1 - cv))
                  for cv in coverages}

    P_test = fit_predict_multi(cols, X[tr3], y3[tr3].astype(int), X[te3], mode)
    p_cont_t, p_rev_t, p_tradeoff_t = P_test[:, 0], P_test[:, 1], P_test[:, 2]
    pred_rev = (p_rev_t > p_cont_t).astype(int)
    action_margin_t = np.maximum(p_cont_t, p_rev_t) - p_tradeoff_t

    rows = []
    extras = {}
    for cv in coverages:
        thr = thresholds[cv]
        selected = action_margin_t >= thr
        m, extra = eval_selection(selected, pred_rev, p_rev_t, None, te3, y3,
                                  None, side, rr, SYM_ARRAY)
        m["wf"] = wf_name
        m["tag"] = f"T3_{mode.upper()}_{int(cv*100)}pct"
        m["coverage_target"] = cv
        m["action_margin_thr"] = round(thr, 4)
        rows.append(m)
        extras[cv] = extra
    # 默认 Primary = 20%
    _, extra20 = eval_selection(
        action_margin_t >= thresholds[0.20], pred_rev, p_rev_t, None, te3, y3,
        None, side, rr, SYM_ARRAY)
    return rows, extras, thresholds


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    t0 = time.perf_counter()
    F = pd.read_parquet(FEATURES_SRC)
    LIQ_TYPES = sorted(F["liquidity_type"].dropna().unique().tolist())
    rp.define_blocks(LIQ_TYPES)
    FEATURE_COLS = rp.FEATURE_COLS

    X_all = F.drop(columns=[c for c in FORBIDDEN if c in F.columns]).copy()
    bad = set(X_all.columns) & FORBIDDEN
    assert not bad, f"leakage: {bad}"
    X = X_all[FEATURE_COLS].copy()
    for c in X.columns:
        if c in rp.CATEGORICAL:
            continue
        X[c] = pd.to_numeric(X[c], errors="coerce")

    global SYM_ARRAY, DAYS_ARRAY
    SYM_ARRAY = F["symbol"].to_numpy()
    DAYS_ARRAY = F["trading_day"].astype(str).to_numpy()
    F["trading_day"] = DAYS_ARRAY
    insample = F["trading_day"] < OOS_START

    y_clear = F["y_clear"].to_numpy()
    y_rev = F["y_reversal"].to_numpy()
    rr = F["rr_direction"].to_numpy()
    side = F["side"].to_numpy()
    block = F["block"].to_numpy()
    days = DAYS_ARRAY

    y3 = three_class_label(side, rr)

    G4_ONLY = ["nearest_above_R", "nearest_below_R",
               "nearest_ahead_R", "nearest_behind_R"]
    G4_BASE = ["symbol", "side"] + G4_ONLY
    G4_D1 = [c for c in rp.block_cols("M_GLOBAL4", LIQ_TYPES) if c in X.columns]
    CLEAR_COLS = G4_D1  # C_GLOBAL4 = M_GLOBAL4 block
    DIR_COLS = G4_D1    # N_GLOBAL4 = M_GLOBAL4 block (hgb)

    print(f"[S1] loaded {X.shape} ({time.perf_counter()-t0:.1f}s); "
          f"liq_types={len(LIQ_TYPES)}; y3 classes="
          f"{dict(zip(*np.unique(y3[~pd.isna(y3)], return_counts=True)))}")

    # ----- P1 Minimal-G4 direction AUC -----
    p1_rows = []
    for fset, cols in (("G4_ONLY", G4_ONLY), ("G4_BASE", G4_BASE),
                       ("G4_D1", G4_D1)):
        c = [x for x in cols if x in X.columns]
        for mode in ("logistic", "hgb"):
            for wf_name, trb, teb in WF:
                m_tr = pd.Series(block).isin(trb).to_numpy() & insample
                m_te = pd.Series(block).isin(teb).to_numpy() & insample
                tr_dir = m_tr & (~pd.isna(y_rev))
                te_dir = m_te & (~pd.isna(y_rev))
                if len(set(y_rev[tr_dir])) < 2 or len(set(y_rev[te_dir])) < 2:
                    continue
                p = rp.fit_predict(c, X[tr_dir], y_rev[tr_dir], X[te_dir], mode)
                auc = float(roc_auc_score(y_rev[te_dir], p)) \
                    if len(set(y_rev[te_dir])) > 1 else np.nan
                p1_rows.append(dict(feature_set=fset, model=mode, wf=wf_name,
                                    roc_auc=auc, n=int(te_dir.sum())))
    p1_df = pd.DataFrame(p1_rows)
    # deltas
    print(f"[P1] done ({time.perf_counter()-t0:.1f}s)")
    print(p1_df.groupby(["feature_set", "model"])["roc_auc"].mean().round(4))

    # ----- P2 calibrated two-stage -----
    cand = {}  # tag -> {wf: (metrics, extra)}
    rows_2stage = []
    for wf_name, trb, teb in WF:
        r, extra, oof = run_calibrated_two_stage(
            wf_name, trb, teb, CLEAR_COLS, DIR_COLS, "logistic", "hgb",
            LIQ_TYPES, X, y_clear, y_rev, side, rr, days, block, insample)
        if r is None:
            print(f"  skip 2stage {wf_name}"); continue
        rows_2stage.append(r)
        cand.setdefault("CALIBRATED_TWO_STAGE", {})[wf_name] = (r, extra)
    # ----- J2_FIXED_0.5 baseline (for bootstrap) -----
    cand_j2 = {}
    for wf_name, trb, teb in WF:
        r, extra, _ = run_calibrated_two_stage(
            wf_name, trb, teb, CLEAR_COLS, DIR_COLS, "logistic", "hgb",
            LIQ_TYPES, X, y_clear, y_rev, side, rr, days, block, insample,
            fixed_clear_thr=0.5, tag="J2_FIXED_0.5")
        if r is None:
            continue
        cand_j2[wf_name] = (r, extra)
        rows_2stage.append(r)

    df_2stage = pd.DataFrame(rows_2stage)

    # ----- P3 direct 3-class -----
    rows_3c = []
    cand_3c = {}
    for mode in ("logistic", "hgb"):
        for wf_name, trb, teb in WF:
            res = run_three_class(
                wf_name, trb, teb, G4_D1, mode, LIQ_TYPES, X, y3, side, rr,
                days, block, insample)
            if res is None:
                print(f"  skip 3class {mode} {wf_name}"); continue
            rrows, extras, thr = res
            rows_3c.extend(rrows)
            # Primary 20% store
            cand_3c.setdefault(f"T3_{mode.upper()}_20pct", {})[wf_name] = (
                rrows[1], extras[0.20])
    df_3c = pd.DataFrame(rows_3c)
    print(f"[P3] done ({time.perf_counter()-t0:.1f}s)")

    # ----- P4 product diagnostic -----
    rows_prod = []
    cand_prod = {}
    for wf_name, trb, teb in WF:
        if wf_name not in cand.get("CALIBRATED_TWO_STAGE", {}):
            continue
        r, extra = cand["CALIBRATED_TWO_STAGE"][wf_name]
        if r.get("clear_thr_unavailable"):
            continue
        oidx_c, p_clear_oof, p_dir_oof = None, None, None
        # 乘积诊断 OOF：clear 模型在 clear∪tradeoff 上、dir 模型在 clear 上，
        # 各自独立跑，再取 OOF 行索引交集对齐（两模型训练 mask 不同，无法共用）。
        m_tr = pd.Series(block).isin(trb).to_numpy()
        m_te = pd.Series(block).isin(teb).to_numpy()
        trc = m_tr & (~pd.isna(y_clear)) & insample
        tec = m_te & (~pd.isna(y_clear)) & insample
        tr_dir = m_tr & (~pd.isna(y_rev)) & insample
        oidx_c, p_clear_oof = rp.expanding_oof_pred(
            CLEAR_COLS, X[trc], y_clear[trc], days[trc], "logistic")
        oidx_d, p_dir_oof = rp.expanding_oof_pred(
            DIR_COLS, X[tr_dir], y_rev[tr_dir], days[tr_dir], "hgb")
        sc = set(oidx_c.tolist())
        sd = set(oidx_d.tolist())
        common = np.array(sorted(sc & sd))
        pos_c = {int(i): k for k, i in enumerate(oidx_c.tolist())}
        pos_d = {int(i): k for k, i in enumerate(oidx_d.tolist())}
        pc = np.array([p_clear_oof[pos_c[int(i)]] for i in common])
        pd_ = np.array([p_dir_oof[pos_d[int(i)]] for i in common])
        side_oof = side[trc][common]
        P_short_oof = np.where(side_oof == +1, pc * pd_, pc * (1 - pd_))
        P_long_oof = np.where(side_oof == +1, pc * (1 - pd_), pc * pd_)
        P_tradeoff_oof = 1 - pc
        jm_oof = np.maximum(P_long_oof, P_short_oof) - P_tradeoff_oof
        thr = float(np.quantile(jm_oof, 0.80))
        p_clear_t = extra["p_clear_test"]
        p_dir_t = extra["p_dir_test"]
        side_t = side[tec]
        P_short_t = np.where(side_t == +1, p_clear_t * p_dir_t,
                            p_clear_t * (1 - p_dir_t))
        P_long_t = np.where(side_t == +1, p_clear_t * (1 - p_dir_t),
                            p_clear_t * p_dir_t)
        P_tradeoff_t = 1 - p_clear_t
        jm_t = np.maximum(P_long_t, P_short_t) - P_tradeoff_t
        selected = jm_t >= thr
        pred_rev = np.full(tec.sum(), -1, int)
        pred_rev[p_dir_t <= r["dir_tail_lo"]] = 0
        pred_rev[p_dir_t >= r["dir_tail_hi"]] = 1
        m, ex = eval_selection(
            selected, pred_rev, p_dir_t, p_clear_t, tec, y_rev, y_clear, side,
            rr, SYM_ARRAY)
        m["wf"] = wf_name
        m["tag"] = "TWO_STAGE_PRODUCT"
        m["joint_margin_thr"] = round(thr, 4)
        rows_prod.append(m)
        cand_prod[wf_name] = (m, ex)
    df_prod = pd.DataFrame(rows_prod)

    # ----- P5 Gate -----
    def gate_eval(cand_dict):
        wf_ap, wf_sel = [], []
        sel_all = []
        lon, sho = [], []
        for wf_name, (r, ex) in cand_dict.items():
            wf_ap.append(r["actionable_precision"])
            wf_sel.append(r["selection_rate"])
            sel_all.append(ex["selected"])
            lon.append(ex["pred_long"] == 1)
            sho.append(ex["pred_long"] == 0)
        # pooled predicted LONG/SHORT actionable
        sel = np.concatenate(sel_all)
        actual_clear = np.concatenate([cand_dict[w][1]["actual_clear"] for w in cand_dict])
        corr = np.concatenate([cand_dict[w][1]["correct_direction"] for w in cand_dict])
        pred_long = np.concatenate([cand_dict[w][1]["pred_long"] for w in cand_dict])
        rr_cat = np.concatenate([cand_dict[w][1]["side"] for w in cand_dict])
        act_long = np.full(len(sel), -1, int)
        act_long[actual_clear] = (rr_cat[actual_clear] == "LONG_DOMINATES").astype(int)
        plm = sel & (pred_long == 1)
        psm = sel & (pred_long == 0)
        long_ap = float((actual_clear[plm] & (act_long[plm] == 1)).mean()) \
            if plm.sum() else np.nan
        short_ap = float((actual_clear[psm] & (act_long[psm] == 0)).mean()) \
            if psm.sum() else np.nan
        per_wf_ok = (all(a >= 0.58 for a in wf_ap if pd.notna(a))
                     and all(s >= 0.05 for s in wf_sel if pd.notna(s))
                     and np.nanmean(wf_ap) >= 0.60)
        return per_wf_ok and pd.notna(long_ap) and pd.notna(short_ap) \
            and long_ap >= 0.58 and short_ap >= 0.58, dict(
                wf_actionable=wf_ap, wf_selection=wf_sel,
                pooled_LONG_actionable=long_ap, pooled_SHORT_actionable=short_ap,
                mean_actionable=float(np.nanmean(wf_ap)))

    gate_2stage, g_2 = gate_eval(cand.get("CALIBRATED_TWO_STAGE", {}))
    gate_3c, g_3 = gate_eval(cand_3c.get("T3_HGB_20pct", {}))
    ACTIONABLE = bool(gate_2stage or gate_3c)

    # ----- P7 by-symbol -----
    bs_rows = []
    for cname, cdict in (("CALIBRATED_TWO_STAGE", cand.get("CALIBRATED_TWO_STAGE", {})),
                         ("T3_HGB_20pct", cand_3c.get("T3_HGB_20pct", {})),
                         ("TWO_STAGE_PRODUCT", cand_prod)):
        if not cdict:
            continue
        for wf_name, (r, ex) in cdict.items():
            sel = ex["selected"]
            sym = ex["symbol"]
            corr = ex["correct_direction"]
            ac = ex["actual_clear"]
            rev = ex["y_dir"]
            dfx = pd.DataFrame(dict(symbol=sym, sel=sel, ac=ac, corr=corr, rev=rev))
            for s, g in dfx.groupby("symbol"):
                n = int(g["sel"].sum())
                if n < 50:
                    continue
                selc = g["sel"] & g["ac"]
                da = float(g["corr"][selc].mean()) if selc.sum() else np.nan
                ap = float((g["sel"] & g["ac"] & g["corr"]).sum() / g["sel"].sum())
                bs_rows.append(dict(candidate=cname, wf=wf_name, symbol=s,
                                    n_selected=n,
                                    selection_rate=round(float(g["sel"].mean()), 4),
                                    selected_clear_rate=round(float((g["sel"] & g["ac"]).sum() / g["sel"].sum()), 4),
                                    direction_accuracy_given_clear=round(da, 4) if pd.notna(da) else None,
                                    actionable_precision=round(ap, 4)))
    df_bs = pd.DataFrame(bs_rows)

    # ----- P8 bootstrap (only if gate passes) -----
    bootstrap_rows = []
    if ACTIONABLE:
        primary = "CALIBRATED_TWO_STAGE" if gate_2stage else "T3_HGB_20pct"
        pdict = cand[primary]
        # build per-WF per-contact arrays keyed by day
        for wf_name in pdict:
            cand_sel = pdict[wf_name][1]["selected"]
            cand_ac = pdict[wf_name][1]["actual_clear"]
            cand_corr = pdict[wf_name][1]["correct_direction"]
            j2_sel = cand_j2[wf_name][1]["selected"]
            j2_ac = cand_j2[wf_name][1]["actual_clear"]
            j2_corr = cand_j2[wf_name][1]["correct_direction"]
            # get tec mask for this wf
            m_te = pd.Series(block).isin(
                [b for w in WF if w[0] == wf_name][0][2]).to_numpy() & insample
            day_arr = DAYS_ARRAY[m_te]
            day_to_idx = defaultdict(list)
            for i, d in enumerate(day_arr):
                day_to_idx[d].append(i)
            day_keys = np.array(list(day_to_idx.keys()))
            rng = np.random.default_rng(42)
            ap_c, ap_j, sel_c, sel_j, delta = [], [], [], [], []
            for _ in range(500):
                samp = rng.choice(day_keys, size=len(day_keys), replace=True)
                idx = np.concatenate([day_to_idx[d] for d in samp])
                if cand_sel[idx].sum() == 0 or j2_sel[idx].sum() == 0:
                    continue
                ap_c.append(float((cand_sel[idx] & cand_ac[idx] & cand_corr[idx]).sum()
                                 / cand_sel[idx].sum()))
                ap_j.append(float((j2_sel[idx] & j2_ac[idx] & j2_corr[idx]).sum()
                                 / j2_sel[idx].sum()))
                sel_c.append(float(cand_sel[idx].mean()))
                sel_j.append(float(j2_sel[idx].mean()))
                delta.append(ap_c[-1] - ap_j[-1])
            if ap_c:
                bootstrap_rows.append(dict(
                    wf=wf_name, primary=primary, n_bootstrap=len(ap_c),
                    actionable_precision_mean=round(float(np.mean(ap_c)), 4),
                    actionable_precision_ci_lo=round(float(np.percentile(ap_c, 2.5)), 4),
                    actionable_precision_ci_hi=round(float(np.percentile(ap_c, 97.5)), 4),
                    selection_rate_mean=round(float(np.mean(sel_c)), 4),
                    j2_actionable_precision_mean=round(float(np.mean(ap_j)), 4),
                    paired_delta_mean=round(float(np.mean(delta)), 4),
                    paired_delta_ci_lo=round(float(np.percentile(delta, 2.5)), 4),
                    paired_delta_ci_hi=round(float(np.percentile(delta, 97.5)), 4)))
        df_boot = pd.DataFrame(bootstrap_rows)
    else:
        df_boot = pd.DataFrame(
            [dict(note="STOP_NO_BOOTSTRAP: ACTIONABLE_DIRECTION_PRESENT=False")])

    # ----- write outputs -----
    p1_df.to_csv(OUT / "minimal_g4_metrics.csv", index=False, encoding="utf-8-sig")
    df_2stage.to_csv(OUT / "two_stage_calibrated_metrics.csv", index=False,
                     encoding="utf-8-sig")
    df_3c.to_csv(OUT / "three_class_metrics.csv", index=False,
                 encoding="utf-8-sig")
    df_prod.to_csv(OUT / "product_diagnostic_metrics.csv", index=False,
                   encoding="utf-8-sig")
    df_bs.to_csv(OUT / "actionability_by_symbol.csv", index=False,
                 encoding="utf-8-sig")
    df_boot.to_csv(OUT / "actionability_bootstrap_ci.csv", index=False,
                   encoding="utf-8-sig")

    protocol = dict(
        experiment="SMC Direction Actionability Architecture v1.2",
        base_commit="26f4785aed7f78abaa6b6b080aed39ea8a186287",
        no_new_features=True,
        primary_risk=PRIMARY_RISK, oos_boundary_excluded=OOS_START,
        wf=WF, symbols=SYMBOLS,
        p1_feature_sets={"G4_ONLY": G4_ONLY, "G4_BASE": G4_BASE,
                         "G4_D1": "M_GLOBAL4 (D1+GLOBAL4)"},
        p2=dict(clear="C_GLOBAL4 logistic", direction="N_GLOBAL4 hgb",
                target_clear_precision=TARGET_CLEAR_PRECISION,
                min_oof_selection_rate=MIN_OOF_SELECTION_RATE,
                dir_tail=(DIR_TAIL_LO, DIR_TAIL_HI)),
        p3=dict(model="3-class HGB/logistic", labels="0=CONT 1=REV 2=TRADEOFF",
                coverages=[0.10, 0.20, 0.30], primary_coverage=0.20),
        p4="product diagnostic (not gate)",
        p5_gate="ACTIONABLE_DIRECTION_PRESENT",
        forbidden=["FVG", "external/internal", "pre-contact dynamics",
                   "OB扩展", "新risk搜索", "PnL", "stop/target优化",
                   "LightGBM调参", "SHAP", "LC", "prospective OOS"],
        trading_metrics="NOT_APPLICABLE",
    )
    json.dump(protocol, open(OUT / "DIRECTION_ACTIONABILITY_PROTOCOL.json", "w"),
              indent=2, ensure_ascii=False, default=str)

    audit = dict(
        experiment="SMC Direction Actionability Architecture v1.2",
        base_commit="26f4785",
        reused_v1_1_1="run_direction_deployability_v1_1_1_repair helpers",
        p0_risk_wording_corrected=(
            "risk-dependent structural switching (P4c continuous direct-switch "
            "median ~1.74 ATR); v1.1.1 '~1.25 ATR switch' phrase superseded"),
        gate=dict(ACTIONABLE_DIRECTION_PRESENT=ACTIONABLE,
                  CALIBRATED_TWO_STAGE_pass=bool(gate_2stage),
                  T3_HGB_20pct_pass=bool(gate_3c),
                  g_2stage=g_2, g_3class=g_3),
        next_step_if_TRUE="PRECONTACT_DYNAMICS_INCREMENT",
        next_step_if_FALSE="temporal regime / WF1 drift study",
    )
    json.dump(audit, open(OUT / "DIRECTION_ACTIONABILITY_AUDIT.json", "w"),
              indent=2, ensure_ascii=False, default=str)

    write_report(p1_df, df_2stage, df_3c, df_prod, df_bs, df_boot, audit)

    print("\n=== GATE (v1.2) ===")
    print(json.dumps(dict(ACTIONABLE_DIRECTION_PRESENT=ACTIONABLE,
                          CALIBRATED_TWO_STAGE_pass=bool(gate_2stage),
                          T3_HGB_20pct_pass=bool(gate_3c)), indent=2))
    print(f"\n[DONE] {time.perf_counter()-t0:.1f}s -> {OUT}")


def write_report(p1_df, df_2stage, df_3c, df_prod, df_bs, df_boot, audit):
    g4_mean = p1_df.groupby(["feature_set", "model"])["roc_auc"].mean().round(4)
    two = df_2stage[df_2stage["tag"] == "CALIBRATED_TWO_STAGE"]
    j2 = df_2stage[df_2stage["tag"] == "J2_FIXED_0.5"]
    t3 = df_3c[df_3c["tag"] == "T3_HGB_20pct"]

    def tbl(df, cols):
        return "\n".join(
            "| " + " | ".join(str(r[c]) for c in cols) + " |"
            for _, r in df.iterrows())

    md = f"""# SMC Direction Actionability Architecture v1.2

**base**: `26f4785` (v1.1.1) &nbsp; **脚本**: `run_direction_actionability_v1_2.py`
**目标**: 不新增任何市场特征，只把 v1.1.1 已确认的两个强信号
（Clear/Tradeoff gate AUC≈0.78；Direction given clear AUC≈0.65）组合成
可行动选择器。TRADING_METRICS=NOT_APPLICABLE（架构验证，未定义交易动作）。

---

## 0. 上一轮修正（P0 措辞）

v1.1.1 报告中 "RISK_DEPENDENT / ~1.25 ATR switch" 表述已过期，本论起统一改为
**risk-dependent structural switching**（P4c 连续 direct-switch 中位约 **1.74 ATR**）。
"1.25" 只是旧网格中点，不再作为机制尺度。

---

## 1. P1 Minimal-G4：direction 能否压缩成纯 4 距离几何

`G4_ONLY`=4 距离；`G4_BASE`=symbol+side+4距离；`G4_D1`=M_GLOBAL4(D1+GLOBAL4)。
| feature_set | model | mean roc_auc |
|---|---|---:|
"""
    for (fs, md_), v in g4_mean.items():
        md += f"| {fs} | {md_} | {v} |\n"
    md += f"""
判断：`G4_D1 - G4_BASE` 与 `G4_BASE - G4_ONLY` 的 ΔAUC 见 CSV。若 |Δ|<0.01 且 3 WF
无稳定增量，则允许 `GLOBAL4_MINIMAL_SUFFICIENT`（方向模型可压缩到极简单结构）。

---

## 2. P2 Calibrated two-stage（precision-first clear gate）

`Clear = C_GLOBAL4(logistic)`，`Direction = N_GLOBAL4(hgb)`。
clear threshold 由 **train-OOF** 在 precision≥{TARGET_CLEAR_PRECISION} 条件下取最大 coverage
（禁止 test 调参）。direction 两尾 = train-OOF 10%/90%。

### CALIBRATED_TWO_STAGE

| WF | clear_thr | selection_rate | selected_clear_rate | direction_accuracy_given_clear | actionable_precision | pred_LONG_ap | pred_SHORT_ap |
|---|---|---:|---:|---:|---:|---:|---:|
"""
    md += tbl(two, ["wf", "clear_thr", "selection_rate",
                    "selected_clear_rate", "direction_accuracy_given_clear",
                    "actionable_precision", "predicted_LONG_actionable_precision",
                    "predicted_SHORT_actionable_precision"])
    md += f"""

### J2_FIXED_0.5（baseline，clear_thr=0.5 不校准）

| WF | selection_rate | selected_clear_rate | direction_accuracy_given_clear | actionable_precision |
|---|---:|---:|---:|---:|
"""
    md += tbl(j2, ["wf", "selection_rate", "selected_clear_rate",
                   "direction_accuracy_given_clear", "actionable_precision"])
    md += f"""

> 校准后 `selected_clear_rate` 应明显高于 fixed 0.5（~20% TRADEOFF contamination
> 被压低）。这是本论针对 v1.1.1 揭示瓶颈（Clear purity≈79%）的直接回应。

---

## 3. P3 Direct 3-class（REVERSAL / CONTINUATION / TRADEOFF）

`T3_HGB` Primary coverage=20%。

| WF | coverage | selection_rate | selected_clear_rate | direction_accuracy_given_clear | actionable_precision |
|---|---:|---:|---:|---:|---:|
"""
    md += tbl(t3, ["wf", "coverage_target", "selection_rate",
                   "selected_clear_rate", "direction_accuracy_given_clear",
                   "actionable_precision"])
    md += f"""

---

## 4. P4 Product diagnostic（非 Gate）

`P_long/P_short = p_clear × p_rev(或 1-p_rev)`，`P_tradeoff = 1-p_clear`，
`joint_margin = max(P_long,P_short) - P_tradeoff`，train-OOF 80% 分位定阈。

| WF | selection_rate | selected_clear_rate | direction_accuracy_given_clear | actionable_precision |
|---|---:|---:|---:|---:|
"""
    md += tbl(df_prod, ["wf", "selection_rate", "selected_clear_rate",
                        "direction_accuracy_given_clear", "actionable_precision"])
    md += f"""

---

## 5. P5 Gate：ACTIONABLE_DIRECTION_PRESENT

```
ACTIONABLE_DIRECTION_PRESENT = {audit['gate']['ACTIONABLE_DIRECTION_PRESENT']}
CALIBRATED_TWO_STAGE pass = {audit['gate']['CALIBRATED_TWO_STAGE_pass']}
T3_HGB_20pct pass      = {audit['gate']['T3_HGB_20pct_pass']}
```

判定：CALIBRATED_TWO_STAGE 或 T3_HGB_20pct 满足
（每 WF actionable≥0.58 且 mean≥0.60 且每 WF selection≥0.05 且
pooled predicted LONG/SHORT actionable≥0.58）→ TRUE。

> 注意：这仍不等于盈利。它只表示在未知未来 clear/tradeoff 真实条件下，
> 可以筛出具有可观方向准确率的候选。

---

## 6. P7 按 symbol

仅 `n_selected>=50` 品种级报告（避免样本碎裂）。见 `actionability_by_symbol.csv`。

---

## 7. P8 Bootstrap

"""
    if "STOP_NO_BOOTSTRAP" in df_boot.columns or df_boot.empty or \
            "note" in df_boot.columns:
        md += "> **STOP_NO_BOOTSTRAP**：ACTIONABLE_DIRECTION_PRESENT=False，未做 bootstrap。\n"
    else:
        md += tbl(df_boot, list(df_boot.columns))
        md += "\n> 若 Gate 通过，paired delta vs J2_FIXED_0.5 给出 actionable 提升置信区间。\n"

    md += f"""

---

## 8. 回答用户 P9 七个问题

1. **clear threshold precision-first 校准是否显著减少 TRADEOFF contamination？**
   见 §2 CALIBRATED vs J2_FIXED 的 `selected_clear_rate`。
2. **calibrated two-stage actionable 是否达 60% 附近？** 见 §2 / §5。
3. **direct 3-class 是否优于 two-stage？** 见 §3 vs §2。
4. **GLOBAL4 是否可压缩成接近 4 变量？** 见 §1。
5. **LONG 与 SHORT 是否都可预测？** 见各表 pred_LONG/SHORT_ap。
6. **WF1 弱是否仍存在？** 见各表 WF1 行。
7. **是否值得加 pre-contact dynamics？** 仅当 ACTIONABLE=TRUE 才建议
   PRECONTACT_DYNAMICS_INCREMENT；否则先研究 temporal regime / WF1 drift。

---

## 9. 完成条件 / STOP

代码 + 运行 + 报告 + commit + push。禁止进入 FVG / pre-contact dynamics /
PnL / stop-target 优化。等待 reviewer 审核 v1.2。
"""
    open(OUT / "SMC_DIRECTION_ACTIONABILITY_V1_2.md", "w",
         encoding="utf-8-sig").write(md)


if __name__ == "__main__":
    main()
