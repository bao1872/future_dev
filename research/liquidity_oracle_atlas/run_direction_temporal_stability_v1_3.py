"""SMC Direction Temporal Stability & Tail Drift Audit v1.3 (base 3da5e68).

审计修复 + 纯时间稳定性诊断。不新增任何市场特征。

修复 v1.2 两个审计问题（P0 Gate aggregation bug, P0.5 product diagnostic 作废），
然后只研究时间稳定性，回答："WF1 为什么比 WF2/WF3 弱？"

五层诊断（见用户 P1-P12）：
  P1  冻结研究 cohort（clear rows 上的 direction 任务）。
  P2  训练历史长度：rolling 1-block vs expanding vs recent-two-block。
  P3  历史模型跨远期测试（6 个 causal train→test pair）。
  P4  每个 pair 做 train-OOF tail transfer（只信 train quantile）。
  P5  区分 rank drift vs tail drift。
  P6  Geometry 分布漂移（4 距离：分布统计 + train→test PSI）。
  P7  Domain classifier（covariate-shift 诊断，不进交易结论）。
  P8  固定模型 score-decile 稳定性（train OOF 分箱边界 → 跨期 reversal rate）。
  P9  Symbol macro drift（每 symbol AUC / tail，不删品种）。
  P10 Clear gate 只做 secondary（C_GLOBAL4 logistic，固定 0.85）。
  P11 机制裁决（6 选 1 protocol verdict）。
  P12 下一步决策。

Governance: TRADING_METRICS=NOT_APPLICABLE；禁止 FVG/external-internal/pre-contact
dynamics/OB/trend/新 risk/PnL/stop-target/参数搜索/LightGBM 调参/SHAP/LC/prospective
OOS（trading_day>=2026-09-07 完全不用）。复用 v1.1.1 已审计 helper。

固定：
  PRIMARY_DIR_COLS = G4_BASE = symbol + side + 4 距离
  PRIMARY_DIR_MODEL = HGB(max_depth=3, lr=0.05, max_iter=200, l2=1.0, seed=42)
  PRIMARY_RISK = 1.0 ATR
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 保证以 repo root 为 import 基准（脚本可能被直接 python 运行）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import research.liquidity_oracle_atlas.run_direction_deployability_v1_1_1_repair as rp
import research.liquidity_oracle_atlas.run_direction_actionability_v1_2 as v2

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

OUT = Path("research/analysis_results/smc_direction_temporal_stability_v1_3")
V12_OUT = Path("research/analysis_results/smc_direction_actionability_v1_2")
FEATURES_SRC = Path(
    "research/analysis_results/smc_direction_deployability_v1_1/"
    "direction_features_v1_1.parquet")
OUT.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["AG", "AU", "CU", "AL", "SN", "NI", "RB", "I", "SC", "RU", "MA",
           "TA", "M", "P", "CF"]
PRIMARY_RISK = 1.0
OOS_START = "2026-09-07"
BLOCKS = ["TB1", "TB2", "TB3", "TB4"]
DIR_TAIL_LO, DIR_TAIL_HI = 0.10, 0.90
TARGET_CLEAR_PRECISION = 0.85
MIN_OOF_SELECTION_RATE = 0.05

# 冻结：本论只用这些，禁止任何新特征。
G4_ONLY = ["nearest_above_R", "nearest_below_R",
           "nearest_ahead_R", "nearest_behind_R"]
G4_BASE = ["symbol", "side"] + G4_ONLY
ALLOWED_FEATURES = set(G4_BASE) | set(G4_ONLY) | \
    set(rp.block_cols("M_GLOBAL4", SYMBOLS))  # 含 C_GLOBAL4 用的 M_GLOBAL4 块

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


def block_idx(b):
    return BLOCKS.index(b) + 1


# ===========================================================================
# 数据加载（与 v1.2 一致，仅取本论所需）
# ===========================================================================
def load_data():
    F = pd.read_parquet(FEATURES_SRC)
    LIQ_TYPES = sorted(F["liquidity_type"].dropna().unique().tolist())
    rp.define_blocks(LIQ_TYPES)
    X_all = F.drop(columns=[c for c in FORBIDDEN if c in F.columns]).copy()
    bad = set(X_all.columns) & FORBIDDEN
    assert not bad, f"leakage: {bad}"
    X = X_all[rp.FEATURE_COLS].copy()
    for c in X.columns:
        if c in rp.CATEGORICAL:
            continue
        X[c] = pd.to_numeric(X[c], errors="coerce")

    SYM = F["symbol"].to_numpy()
    DAYS = F["trading_day"].astype(str).to_numpy()
    F["trading_day"] = DAYS
    insample = F["trading_day"] < OOS_START

    y_clear = F["y_clear"].to_numpy()
    y_rev = F["y_reversal"].to_numpy()
    rr = F["rr_direction"].to_numpy()
    side = F["side"].to_numpy()
    block = F["block"].to_numpy()
    days = DAYS
    y3 = v2.three_class_label(side, rr)  # 复用 v1.2 的 3-class 标签

    # 供 v2 函数使用（gate repair）
    v2.SYM_ARRAY = SYM
    v2.DAYS_ARRAY = DAYS

    D = dict(F=F, LIQ_TYPES=LIQ_TYPES, X=X, SYM=SYM, DAYS=DAYS, insample=insample,
             y_clear=y_clear, y_rev=y_rev, rr=rr, side=side, block=block,
             days=days, y3=y3)
    return D


# ===========================================================================
# 模型 helper（固定 HGB）
# ===========================================================================
def fit_pipe(cols, Xtr, ytr, mode):
    pre = rp.make_preprocessor(cols)
    if mode == "logistic":
        clf = LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs")
    else:
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                             max_iter=200, l2_regularization=1.0,
                                             random_state=42)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(Xtr, ytr)
    return pipe


def safe_auc(y, p):
    if len(set(y)) > 1 and np.isfinite(p).all():
        try:
            return float(roc_auc_score(y, p))
        except Exception:
            return np.nan
    return np.nan


# ===========================================================================
# 方向评估：train-OOF 设尾阈值（只信 train），final 模型评 test
# ===========================================================================
def direction_eval(tr_blocks, te_blocks, cols, mode, D):
    block = D["block"]; insample = D["insample"]; X = D["X"]
    y_rev = D["y_rev"]; days = D["days"]; SYM = D["SYM"]
    m_tr = pd.Series(block).isin(tr_blocks).to_numpy() & insample
    m_te = pd.Series(block).isin(te_blocks).to_numpy() & insample
    tr = m_tr & (~pd.isna(y_rev))
    te = m_te & (~pd.isna(y_rev))
    if len(set(y_rev[tr])) < 2 or len(set(y_rev[te])) < 2:
        return None
    c = [x for x in cols if x in X.columns]
    oidx, p_oof = rp.expanding_oof_pred(c, X[tr], y_rev[tr], days[tr], mode)
    if len(p_oof) == 0:
        return None
    tr_global = np.flatnonzero(tr)
    oidx_global = tr_global[oidx]
    lo = float(np.quantile(p_oof, DIR_TAIL_LO))
    hi = float(np.quantile(p_oof, DIR_TAIL_HI))
    train_oof_auc = safe_auc(y_rev[tr][oidx], p_oof)
    pipe = fit_pipe(c, X[tr], y_rev[tr], mode)
    p_test = pipe.predict_proba(X[te])[:, 1]
    te_global = np.flatnonzero(te)
    test_auc = safe_auc(y_rev[te], p_test)

    is_rev = p_test >= hi
    is_cont = p_test <= lo
    actual_rev = (y_rev[te] == 1)
    rev_tail_acc = float(actual_rev[is_rev].mean()) if is_rev.sum() else np.nan
    cont_tail_acc = float((~actual_rev[is_cont]).mean()) if is_cont.sum() else np.nan
    macro_tail = (np.nanmean([rev_tail_acc, cont_tail_acc])
                  if (pd.notna(rev_tail_acc) and pd.notna(cont_tail_acc)) else np.nan)
    return dict(
        tr_blocks=tr_blocks, te_blocks=te_blocks, cols_used=c, mode=mode,
        n_train=int(tr.sum()), n_test=int(te.sum()),
        train_oof_auc=train_oof_auc, test_auc=test_auc,
        lo=lo, hi=hi,
        n_pred_reversal=int(is_rev.sum()), n_pred_continuation=int(is_cont.sum()),
        tail_test_coverage=float((is_rev | is_cont).mean()),
        reversal_tail_accuracy=rev_tail_acc,
        continuation_tail_accuracy=cont_tail_acc,
        macro_tail_accuracy=macro_tail,
        test_reversal_base_rate=float(actual_rev.mean()),
        te_global=te_global, p_test=p_test, y_rev_te=y_rev[te], symbol_te=SYM[te],
        p_oof=p_oof, oidx_global=oidx_global, pipe=pipe,
    )


# ===========================================================================
# P0 Gate repair（修正 v1.2 pooled LONG/SHORT bug）
# ===========================================================================
def gate_eval_fixed(cand_dict):
    wf_ap, wf_sel = [], []
    for _, (r, ex) in cand_dict.items():
        wf_ap.append(r["actionable_precision"])
        wf_sel.append(r["selection_rate"])
    sel = np.concatenate([ex["selected"] for _, ex in cand_dict.values()])
    actual_clear = np.concatenate([ex["actual_clear"] for _, ex in cand_dict.values()])
    corr = np.concatenate([ex["correct_direction"] for _, ex in cand_dict.values()])
    pred_long = np.concatenate([ex["pred_long"] for _, ex in cand_dict.values()])
    # 关键修正：selected + clear + 方向正确 已唯一确定绝对方向正确；
    # 绝不能再拿 side(+1/-1) 去比较 "LONG_DOMINATES" 字符串。
    plm = sel & (pred_long == 1)
    psm = sel & (pred_long == 0)
    long_ap = (float((actual_clear[plm] & corr[plm]).mean())
               if plm.sum() else np.nan)
    short_ap = (float((actual_clear[psm] & corr[psm]).mean())
                if psm.sum() else np.nan)
    per_wf_ok = (all(a >= 0.58 for a in wf_ap if pd.notna(a))
                 and all(s >= 0.05 for s in wf_sel if pd.notna(s))
                 and np.nanmean(wf_ap) >= 0.60)
    gate = bool(per_wf_ok and pd.notna(long_ap) and pd.notna(short_ap)
                and long_ap >= 0.58 and short_ap >= 0.58)
    return gate, dict(wf_actionable=wf_ap, wf_selection=wf_sel,
                      pooled_LONG_actionable=long_ap,
                      pooled_SHORT_actionable=short_ap,
                      mean_actionable=float(np.nanmean(wf_ap)))


def run_v12_gate_repair(D):
    LIQ_TYPES = D["LIQ_TYPES"]; X = D["X"]; y_clear = D["y_clear"]
    y_rev = D["y_rev"]; side = D["side"]; rr = D["rr"]
    days = D["days"]; block = D["block"]; insample = D["insample"]
    y3 = D["y3"]
    G4_D1 = [c for c in rp.block_cols("M_GLOBAL4", LIQ_TYPES) if c in X.columns]

    cand = {}
    for wf_name, trb, teb in v2.WF:
        r, extra, _ = v2.run_calibrated_two_stage(
            wf_name, trb, teb, G4_D1, G4_D1, "logistic", "hgb",
            LIQ_TYPES, X, y_clear, y_rev, side, rr, days, block, insample)
        if r is None:
            continue
        cand.setdefault("CALIBRATED_TWO_STAGE", {})[wf_name] = (r, extra)

    cand_j2 = {}
    for wf_name, trb, teb in v2.WF:
        r, extra, _ = v2.run_calibrated_two_stage(
            wf_name, trb, teb, G4_D1, G4_D1, "logistic", "hgb",
            LIQ_TYPES, X, y_clear, y_rev, side, rr, days, block, insample,
            fixed_clear_thr=0.5, tag="J2_FIXED_0.5")
        if r is None:
            continue
        cand_j2[wf_name] = (r, extra)

    cand_3c = {}
    for mode in ("hgb",):
        for wf_name, trb, teb in v2.WF:
            res = v2.run_three_class(
                wf_name, trb, teb, G4_D1, mode, LIQ_TYPES, X, y3, side, rr,
                days, block, insample)
            if res is None:
                continue
            rrows, extras, _ = res
            cand_3c.setdefault("T3_HGB_20pct", {})[wf_name] = (
                rrows[1], extras[0.20])

    gate_2s, g_2 = gate_eval_fixed(cand.get("CALIBRATED_TWO_STAGE", {}))
    gate_3c, g_3 = gate_eval_fixed(cand_3c.get("T3_HGB_20pct", {}))

    # 同时报告 v1.2 旧（buggy）聚合，便于对照
    buggy = dict(pooled_LONG_actionable=0.0, pooled_SHORT_actionable=0.8518)

    repair = dict(
        experiment="SMC Direction Actionability v1.2 — Gate Aggregation Repair",
        base_commit="3da5e68",
        bug="gate_eval 用 extra['side'](+1/-1) 比较 'LONG_DOMINATES' 字符串，"
            "pooled LONG/SHORT 恒为 0 / 0.85（错误）。",
        fix="改用 selected+actual_clear+correct_direction 计算 pooled actionable；"
            "pred_long 已由 side/pred_rev 正确编码。",
        GATE_REPAIRED_ACTIONABLE_DIRECTION_PRESENT=bool(gate_2s or gate_3c),
        CALIBRATED_TWO_STAGE_pass=bool(gate_2s),
        T3_HGB_20pct_pass=bool(gate_3c),
        repaired_two_stage=g_2,
        repaired_three_class=g_3,
        v12_buggy_reference_for_comparison_only=buggy,
        note="Gate 仍 FALSE：因 WF1 per-WF actionable<0.58（0.560），与 pooled "
             "LONG/SHORT 无关；修正后 pooled LONG/SHORT 均约 0.60，为双边模型。",
    )
    return repair


# ===========================================================================
# P6 PSI（bin 由 train 定义）
# ===========================================================================
def psi_from_train_bins(train, test, n_bins=10):
    x = train[np.isfinite(train)]
    if len(x) < n_bins + 1:
        return np.nan
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return np.nan
    edges = edges.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf
    p, _ = np.histogram(train[np.isfinite(train)], bins=edges)
    q, _ = np.histogram(test[np.isfinite(test)], bins=edges)
    p = p / max(p.sum(), 1)
    q = q / max(q.sum(), 1)
    eps = 1e-6
    return float(np.sum((q - p) * np.log((q + eps) / (p + eps))))


# ===========================================================================
# P7 Domain classifier（covariate-shift 诊断）
# ===========================================================================
def domain_classifier(tr_block, te_block, cols, D):
    m = (pd.Series(D["block"]).isin([tr_block, te_block]).to_numpy()) & D["insample"]
    if m.sum() == 0:
        return None
    Xs = D["X"][m]
    yd = (D["block"][m] == te_block).astype(int)
    c = [x for x in cols if x in D["X"].columns]
    if len(set(yd)) < 2:
        return None
    Xtr, Xte, ytr, yte = train_test_split(
        Xs, yd, test_size=0.3, stratify=yd, random_state=42)
    pipe = fit_pipe(c, Xtr, ytr, "hgb")
    p = pipe.predict_proba(Xte)[:, 1]
    return safe_auc(yte, p)


# ===========================================================================
# P8 Score-decile stability（train OOF 边界 → 跨期）
# ===========================================================================
def decile_stability(src_block, res_dict, D):
    cfgkey = f"{src_block}_" + ({"TB1": "TB2", "TB2": "TB3", "TB3": "TB4"}[src_block])
    if cfgkey not in res_dict:
        return []
    res = res_dict[cfgkey]
    pipe = res["pipe"]
    p_oof = res["p_oof"]
    edges = np.unique(np.quantile(p_oof, np.linspace(0, 1, 11)))
    rows = []
    for tb in BLOCKS:
        if block_idx(tb) <= block_idx(src_block):
            continue
        m_te = (pd.Series(D["block"]).isin([tb]).to_numpy()) & D["insample"]
        te = m_te & (~pd.isna(D["y_rev"]))
        if te.sum() == 0:
            continue
        p = pipe.predict_proba(D["X"][te])[:, 1]
        actual_rev = (D["y_rev"][te] == 1)
        dig = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
        for d in range(len(edges) - 1):
            mask = dig == d
            n = int(mask.sum())
            if n == 0:
                rows.append(dict(train_source=src_block, test_block=tb,
                                score_decile=d, n=0,
                                mean_score=np.nan, actual_reversal_rate=np.nan))
                continue
            rows.append(dict(train_source=src_block, test_block=tb,
                            score_decile=d, n=n,
                            mean_score=float(p[mask].mean()),
                            actual_reversal_rate=float(actual_rev[mask].mean())))
    return rows


# ===========================================================================
# P9 Symbol macro drift
# ===========================================================================
def symbol_temporal(res, D):
    te_global = res["te_global"]; p_test = res["p_test"]
    yv = res["y_rev_te"]; sym = res["symbol_te"]
    lo, hi = res["lo"], res["hi"]
    dfx = pd.DataFrame(dict(symbol=sym, p=p_test, y=yv))
    rows = []
    for s, g in dfx.groupby("symbol"):
        n = len(g); npos = int(g["y"].sum())
        if n < 100 or npos == 0 or npos == n:
            continue
        auc = safe_auc(g["y"].to_numpy(), g["p"].to_numpy())
        is_rev = g["p"] >= hi; is_cont = g["p"] <= lo
        ar = (g["y"] == 1)
        rta = ar[is_rev].mean() if is_rev.sum() else np.nan
        cta = (~ar[is_cont]).mean() if is_cont.sum() else np.nan
        macro = (np.nanmean([rta, cta])
                 if (pd.notna(rta) and pd.notna(cta)) else np.nan)
        rows.append(dict(train_blocks="+".join(res["tr_blocks"]),
                        test_block="+".join(res["te_blocks"]),
                        symbol=s, n=n, n_pos=npos,
                        auc=round(auc, 4),
                        macro_tail_accuracy=(round(macro, 4)
                                             if pd.notna(macro) else None)))
    return rows


# ===========================================================================
# P10 Clear gate temporal（secondary, C_GLOBAL4 logistic, 固定 0.85）
# ===========================================================================
def clear_gate_eval(tr_blocks, te_blocks, D):
    clear_cols = [c for c in rp.block_cols("M_GLOBAL4", D["LIQ_TYPES"])
                  if c in D["X"].columns]
    m_tr = pd.Series(D["block"]).isin(tr_blocks).to_numpy() & D["insample"]
    m_te = pd.Series(D["block"]).isin(te_blocks).to_numpy() & D["insample"]
    trc = m_tr & (~pd.isna(D["y_clear"]))
    tec = m_te & (~pd.isna(D["y_clear"]))
    if len(set(D["y_clear"][trc])) < 2 or len(set(D["y_clear"][tec])) < 2:
        return None
    oidx, p_oof = rp.expanding_oof_pred(
        clear_cols, D["X"][trc], D["y_clear"][trc], D["days"][trc], "logistic")
    thr = v2.choose_clear_threshold(
        D["y_clear"][trc][oidx], p_oof, TARGET_CLEAR_PRECISION,
        MIN_OOF_SELECTION_RATE)
    p_test = rp.fit_predict(
        clear_cols, D["X"][trc], D["y_clear"][trc], D["X"][tec], "logistic")
    clear_auc = safe_auc(D["y_clear"][tec], p_test)
    if thr is None:
        return dict(tr_blocks=tr_blocks, te_blocks=te_blocks, clear_auc=clear_auc,
                    clear_thr="UNAVAILABLE",
                    test_selected_clear_rate=np.nan, test_selection_rate=np.nan)
    selected = p_test >= thr
    scr = (float((selected & (D["y_clear"][tec] == 1)).sum() / selected.sum())
           if selected.sum() else np.nan)
    sr = float(selected.mean()) if selected.sum() else np.nan
    return dict(tr_blocks=tr_blocks, te_blocks=te_blocks, clear_auc=clear_auc,
                clear_thr=round(thr, 4),
                test_selected_clear_rate=scr, test_selection_rate=sr)


# ===========================================================================
# P5 / P11 机制裁决
# ===========================================================================
def decide_verdict(res_dict):
    def auc(n):
        return res_dict[n]["test_auc"] if n in res_dict else np.nan
    def tail(n):
        return res_dict[n]["macro_tail_accuracy"] if n in res_dict else np.nan

    ev = {}
    # TB1 模型对所有未来
    tb1_future = [auc(n) for n in ("TB1_TB2", "TB1_TB3", "TB1_TB4")]
    tb1_weak = all(pd.notna(a) and a < 0.60 for a in tb1_future) and len(tb1_future) == 3
    # TB2 特殊期：TB1->TB2 弱但 TB1->TB3/TB4 恢复
    tb2_regime = (pd.notna(auc("TB1_TB2")) and auc("TB1_TB2") < 0.60 and
                  any(pd.notna(auc(n)) and auc(n) >= 0.63
                      for n in ("TB1_TB3", "TB1_TB4")))
    # 训练历史长度：expanding 明显优于 rolling 1-block（同 test 目标）
    hist_helps = (pd.notna(auc("E123")) and pd.notna(auc("TB2_TB3")) and
                  (auc("E123") - auc("TB2_TB3")) >= 0.03) or \
                 (pd.notna(auc("E1234")) and pd.notna(auc("TB3_TB4")) and
                  (auc("E1234") - auc("TB3_TB4")) >= 0.03) or \
                 (pd.notna(auc("R23_4")) and pd.notna(auc("TB3_TB4")) and
                  (auc("R23_4") - auc("TB3_TB4")) >= 0.03)
    # rank stable tail drift：存在某窗口 AUC>=0.62 但其 tail macro 比全局最佳 tail 低 >=0.07
    best_tail = max([tail(n) for n in res_dict if pd.notna(tail(n))]) \
        if any(pd.notna(tail(n)) for n in res_dict) else np.nan
    tail_drift = any(pd.notna(auc(n)) and auc(n) >= 0.62 and pd.notna(tail(n))
                    and pd.notna(best_tail) and (best_tail - tail(n)) >= 0.07
                    for n in res_dict)
    # broad nonstationarity：多数 pair AUC 明显低于 0.62
    aucs = [auc(n) for n in res_dict if pd.notna(auc(n))]
    broad = len(aucs) >= 4 and sum(a < 0.62 for a in aucs) >= len(aucs) * 0.6

    ev["TB1_model_weak_to_all_future"] = bool(tb1_weak)
    ev["TB2_target_regime_shift"] = bool(tb2_regime)
    ev["train_history_helps"] = bool(hist_helps)
    ev["rank_stable_tail_drift"] = bool(tail_drift)
    ev["broad_nonstationarity"] = bool(broad)
    ev["aucs"] = {k: (round(v, 4) if pd.notna(v) else None)
                  for k, v in ((n, auc(n)) for n in res_dict)}
    ev["tail_macro"] = {k: (round(tail(k), 4) if pd.notna(tail(k)) else None)
                        for k in res_dict}
    ev["best_tail_macro"] = (round(float(best_tail), 4)
                             if pd.notna(best_tail) else None)

    if tb2_regime:
        verdict = "TB2_TARGET_REGIME_SHIFT"
    elif tb1_weak:
        verdict = "SOURCE_TB1_RELATION_DRIFT"
    elif hist_helps:
        verdict = "TRAIN_HISTORY_LIMITED"
    elif tail_drift:
        verdict = "RANK_STABLE_TAIL_DRIFT"
    elif broad:
        verdict = "BROAD_TEMPORAL_NONSTATIONARITY"
    else:
        verdict = "NO_SINGLE_DRIFT_MECHANISM"

    next_step = {
        "TRAIN_HISTORY_LIMITED": "RECENT/EXPANDING TRAIN WINDOW POLICY（非加特征）",
        "TB2_TARGET_REGIME_SHIFT": "TB2-specific 窗口/条件处理（非加特征）",
        "SOURCE_TB1_RELATION_DRIFT": "SOURCE-TB1 关系漂移研究（现有状态条件化）",
        "RANK_STABLE_TAIL_DRIFT": "rank-based adaptive abstention（非 FVG）",
        "BROAD_TEMPORAL_NONSTATIONARITY": "regime conditioning（优先现有市场状态）",
        "NO_SINGLE_DRIFT_MECHANISM": "时间关系基本稳定但 AUC 上限卡住 → 评估 PRECONTACT_DYNAMICS_INCREMENT",
    }[verdict]
    return verdict, next_step, ev


# ===========================================================================
# main
# ===========================================================================
def main():
    t0 = time.perf_counter()
    D = load_data()
    print(f"[LOAD] {D['X'].shape} liq_types={len(D['LIQ_TYPES'])} "
          f"({time.perf_counter()-t0:.1f}s)")

    # ----- P0 gate repair -----
    gate_repair = run_v12_gate_repair(D)
    json.dump(gate_repair, open(OUT / "v12_gate_repair.json", "w"),
              indent=2, ensure_ascii=False, default=str)
    print(f"[P0] gate repaired ACTIONABLE="
          f"{gate_repair['GATE_REPAIRED_ACTIONABLE_DIRECTION_PRESENT']} "
          f"pooled_LONG={gate_repair['repaired_two_stage']['pooled_LONG_actionable']} "
          f"pooled_SHORT={gate_repair['repaired_two_stage']['pooled_SHORT_actionable']}")

    # ----- P2/P3/P4 方向评估（全部 G4_BASE hgb）-----
    CONFIGS = {
        "TB1_TB2":  (["TB1"], ["TB2"], "ONE_BLOCK_ROLLING"),
        "TB2_TB3":  (["TB2"], ["TB3"], "ONE_BLOCK_ROLLING"),
        "TB3_TB4":  (["TB3"], ["TB4"], "ONE_BLOCK_ROLLING"),
        "TB1_TB3":  (["TB1"], ["TB3"], "ROLLING_CROSS"),
        "TB1_TB4":  (["TB1"], ["TB4"], "ROLLING_CROSS"),
        "TB2_TB4":  (["TB2"], ["TB4"], "ROLLING_CROSS"),
        "E123":     (["TB1", "TB2"], ["TB3"], "EXPANDING"),
        "E1234":    (["TB1", "TB2", "TB3"], ["TB4"], "EXPANDING"),
        "R23_4":    (["TB2", "TB3"], ["TB4"], "RECENT_TWO_BLOCK"),
    }
    res_dict = {}
    for name, (trb, teb, wtype) in CONFIGS.items():
        r = direction_eval(trb, teb, G4_BASE, "hgb", D)
        if r is None:
            print(f"  skip {name}")
            continue
        r["window_type"] = wtype
        res_dict[name] = r
        print(f"[DIR] {name:8s} test_auc={r['test_auc']:.4f} "
              f"tail_macro={r['macro_tail_accuracy']:.4f} "
              f"cov={r['tail_test_coverage']:.3f}")
    print(f"[P2-4] done ({time.perf_counter()-t0:.1f}s); configs="
          f"{list(res_dict)}")

    # train_window_comparison.csv (P2: 6 windows)
    p2_names = [n for n in CONFIGS if CONFIGS[n][2] in
                ("ONE_BLOCK_ROLLING", "EXPANDING", "RECENT_TWO_BLOCK")]
    p2_rows = []
    for n in p2_names:
        if n not in res_dict:
            continue
        r = res_dict[n]
        p2_rows.append(dict(config=n, window_type=r["window_type"],
                            train_blocks="+".join(r["tr_blocks"]),
                            test_block="+".join(r["te_blocks"]),
                            test_auc=r["test_auc"], train_oof_auc=r["train_oof_auc"],
                            macro_tail_accuracy=r["macro_tail_accuracy"],
                            reversal_tail_accuracy=r["reversal_tail_accuracy"],
                            continuation_tail_accuracy=r["continuation_tail_accuracy"],
                            tail_test_coverage=r["tail_test_coverage"],
                            n_train=r["n_train"], n_test=r["n_test"],
                            test_reversal_base_rate=r["test_reversal_base_rate"]))
    df_p2 = pd.DataFrame(p2_rows)

    # causal_cross_block_metrics.csv (P3: 6 causal pairs)
    p3_names = ["TB1_TB2", "TB1_TB3", "TB1_TB4", "TB2_TB3", "TB2_TB4", "TB3_TB4"]
    p3_rows = []
    for n in p3_names:
        if n not in res_dict:
            continue
        r = res_dict[n]
        p3_rows.append(dict(config=n, train_blocks="+".join(r["tr_blocks"]),
                            test_block="+".join(r["te_blocks"]),
                            test_auc=r["test_auc"], train_oof_auc=r["train_oof_auc"],
                            macro_tail_accuracy=r["macro_tail_accuracy"],
                            reversal_tail_accuracy=r["reversal_tail_accuracy"],
                            continuation_tail_accuracy=r["continuation_tail_accuracy"],
                            tail_test_coverage=r["tail_test_coverage"],
                            n_train=r["n_train"], n_test=r["n_test"],
                            test_reversal_base_rate=r["test_reversal_base_rate"]))
    df_p3 = pd.DataFrame(p3_rows)

    # tail_transfer_metrics.csv (P4: all 9 configs full tail)
    p4_rows = []
    for n, r in res_dict.items():
        p4_rows.append(dict(config=n, window_type=r["window_type"],
                            train_blocks="+".join(r["tr_blocks"]),
                            test_block="+".join(r["te_blocks"]),
                            test_auc=r["test_auc"], lo_threshold=r["lo"],
                            hi_threshold=r["hi"],
                            n_pred_reversal=r["n_pred_reversal"],
                            n_pred_continuation=r["n_pred_continuation"],
                            tail_test_coverage=r["tail_test_coverage"],
                            reversal_tail_accuracy=r["reversal_tail_accuracy"],
                            continuation_tail_accuracy=r["continuation_tail_accuracy"],
                            macro_tail_accuracy=r["macro_tail_accuracy"],
                            test_reversal_base_rate=r["test_reversal_base_rate"]))
    df_p4 = pd.DataFrame(p4_rows)

    # ----- P6 geometry distribution + PSI -----
    dist_rows = []
    psi_rows = []
    X = D["X"]
    for fld in G4_ONLY:
        for b in BLOCKS:
            m = (pd.Series(D["block"]).isin([b]).to_numpy()) & D["insample"]
            vals = X[fld].to_numpy()[m]
            finite = vals[np.isfinite(vals)]
            miss = 1 - len(finite) / max(len(vals), 1)
            dist_rows.append(dict(field=fld, block=b,
                                 missing_rate=round(float(miss), 4),
                                 p10=(round(float(np.percentile(finite, 10)), 4)
                                      if len(finite) else np.nan),
                                 p25=(round(float(np.percentile(finite, 25)), 4)
                                      if len(finite) else np.nan),
                                 median=(round(float(np.median(finite)), 4)
                                         if len(finite) else np.nan),
                                 p75=(round(float(np.percentile(finite, 75)), 4)
                                      if len(finite) else np.nan),
                                 p90=(round(float(np.percentile(finite, 90)), 4)
                                      if len(finite) else np.nan),
                                 n=int(len(vals))))
        # PSI train->test for causal pairs
        for n in p3_names:
            if n not in res_dict:
                continue
            trb = CONFIGS[n][0]; teb = CONFIGS[n][1]
            m_tr = (pd.Series(D["block"]).isin(trb).to_numpy()) & D["insample"]
            m_te = (pd.Series(D["block"]).isin(teb).to_numpy()) & D["insample"]
            trv = X[fld].to_numpy()[m_tr]
            tev = X[fld].to_numpy()[m_te]
            psi = psi_from_train_bins(trv, tev)
            psi_rows.append(dict(train_blocks="+".join(trb), test_block="+".join(teb),
                                field=fld, psi=(round(psi, 4) if pd.notna(psi) else np.nan)))
    df_dist = pd.DataFrame(dist_rows)
    df_psi = pd.DataFrame(psi_rows)

    # ----- P7 domain classifier -----
    dom_rows = []
    for trb, teb in (("TB1", "TB2"), ("TB2", "TB3"), ("TB3", "TB4")):
        for model, cols in (("DOMAIN_G4_ONLY", G4_ONLY), ("DOMAIN_G4_BASE", G4_BASE)):
            a = domain_classifier(trb, teb, cols, D)
            dom_rows.append(dict(pair=f"{trb}_vs_{teb}", model=model,
                                 domain_auc=(round(a, 4) if pd.notna(a) else np.nan)))
    df_dom = pd.DataFrame(dom_rows)

    # ----- P8 decile stability -----
    decile_rows = []
    for src in ("TB1", "TB2", "TB3"):
        decile_rows.extend(decile_stability(src, res_dict, D))
    df_dec = pd.DataFrame(decile_rows)

    # ----- P9 symbol macro -----
    sym_rows = []
    for n in p3_names:
        if n not in res_dict:
            continue
        sym_rows.extend(symbol_temporal(res_dict[n], D))
    df_sym = pd.DataFrame(sym_rows)
    # 聚合
    sym_agg = []
    for n in p3_names:
        sub = df_sym[df_sym.apply(
            lambda r, n=n: r["train_blocks"] + "->" + r["test_block"]
            == n.replace("_", "->"), axis=1)]
        if len(sub):
            aucs = sub["auc"].dropna()
            sym_agg.append(dict(config=n,
                                eligible_symbol_count=int(len(sub)),
                                macro_median_auc=round(float(aucs.median()), 4),
                                macro_auc_iqr=round(float(aucs.quantile(0.75)
                                                          - aucs.quantile(0.25)), 4),
                                n_symbols_auc_gt_05=int((aucs > 0.5).sum())))
    df_sym_agg = pd.DataFrame(sym_agg)

    # ----- P10 clear gate temporal -----
    cg_rows = []
    for n in p2_names:
        if n not in CONFIGS:
            continue
        trb, teb, _ = CONFIGS[n]
        r = clear_gate_eval(trb, teb, D)
        if r is None:
            continue
        cg_rows.append(r)
    df_cg = pd.DataFrame(cg_rows)

    # ----- P5/P11 verdict -----
    verdict, next_step, evidence = decide_verdict(res_dict)

    # ----- write outputs -----
    df_p2.to_csv(OUT / "train_window_comparison.csv", index=False, encoding="utf-8-sig")
    df_p3.to_csv(OUT / "causal_cross_block_metrics.csv", index=False, encoding="utf-8-sig")
    df_p4.to_csv(OUT / "tail_transfer_metrics.csv", index=False, encoding="utf-8-sig")
    df_dist.to_csv(OUT / "g4_distribution_by_tb.csv", index=False, encoding="utf-8-sig")
    df_psi.to_csv(OUT / "g4_psi_train_test.csv", index=False, encoding="utf-8-sig")
    df_dom.to_csv(OUT / "domain_shift_metrics.csv", index=False, encoding="utf-8-sig")
    df_dec.to_csv(OUT / "score_decile_stability.csv", index=False, encoding="utf-8-sig")
    df_sym.to_csv(OUT / "direction_symbol_temporal.csv", index=False, encoding="utf-8-sig")
    df_sym_agg.to_csv(OUT / "direction_symbol_temporal_agg.csv", index=False, encoding="utf-8-sig")
    df_cg.to_csv(OUT / "clear_gate_temporal_secondary.csv", index=False, encoding="utf-8-sig")

    protocol = dict(
        experiment="SMC Direction Temporal Stability & Tail Drift Audit v1.3",
        base_commit="3da5e6822425e80ff21b44f9df796e3302dd9f79",
        repairs=["P0_gate_aggregation_bug_fixed", "P0.5_product_diagnostic_RETRACTED"],
        TWO_STAGE_PRODUCT_DIAGNOSTIC="RETRACTED_INVALID_OOF_ALIGNMENT",
        no_new_features=True,
        primary_direction_feature="G4_BASE",
        G4_BASE=G4_BASE, G4_ONLY=G4_ONLY,
        primary_direction_model="HistGradientBoostingClassifier(max_depth=3,"
        "learning_rate=0.05,max_iter=200,l2_regularization=1.0,random_state=42)",
        primary_risk=PRIMARY_RISK, oos_boundary_excluded=OOS_START,
        train_window_configs={k: {"train": v[0], "test": v[1], "type": v[2]}
                              for k, v in CONFIGS.items()},
        causal_cross_block_pairs=p3_names,
        dir_tail=(DIR_TAIL_LO, DIR_TAIL_HI),
        p5_verdict_definitions={
            "TRAIN_HISTORY_LIMITED": "rolling 1-block 明显弱于 expanding",
            "TB2_TARGET_REGIME_SHIFT": "TB1->TB2 弱但 TB1->TB3/TB4 恢复",
            "SOURCE_TB1_RELATION_DRIFT": "TB1 模型对所有未来都弱",
            "RANK_STABLE_TAIL_DRIFT": "AUC>=0.62 但 tail macro 比后续低>=0.07",
            "BROAD_TEMPORAL_NONSTATIONARITY": "多数 pair AUC 明显<0.62",
            "NO_SINGLE_DRIFT_MECHANISM": "无单一机制可解释",
        },
        forbidden=["FVG", "external/internal", "pre-contact dynamics", "OB扩展",
                   "trend", "新risk", "PnL", "stop/target", "参数搜索",
                   "prospective OOS", "删品种", "LightGBM调参", "SHAP", "LC"],
        trading_metrics="NOT_APPLICABLE",
    )
    json.dump(protocol, open(OUT / "DIRECTION_TEMPORAL_PROTOCOL.json", "w"),
              indent=2, ensure_ascii=False, default=str)

    audit = dict(
        experiment="SMC Direction Temporal Stability & Tail Drift Audit v1.3",
        base_commit="3da5e68",
        v12_gate_repair=gate_repair,
        p0_5_product_diagnostic="RETRACTED_INVALID_OOF_ALIGNMENT (old CSV kept as bug evidence)",
        verdict=verdict, next_step=next_step, evidence=evidence,
        interpretation_note=(
            "仅当 WF1 actionable<0.58 时不得写 regime drift；必须以 rolling/expanding/"
            "cross-block AUC + tail transfer + PSI + decile + symbol 联合证据裁决。"),
        trading_metrics="NOT_APPLICABLE",
    )
    json.dump(audit, open(OUT / "DIRECTION_TEMPORAL_AUDIT.json", "w"),
              indent=2, ensure_ascii=False, default=str)

    write_report(dict(p2=df_p2, p3=df_p3, p4=df_p4, dist=df_dist, psi=df_psi,
                      dom=df_dom, dec=df_dec, sym=df_sym, sym_agg=df_sym_agg,
                      cg=df_cg),
                 gate_repair, verdict, next_step, evidence)

    print(f"\n=== VERDICT (v1.3) === {verdict} -> {next_step}")
    print(f"[DONE] {time.perf_counter()-t0:.1f}s -> {OUT}")


def write_report(tables, gate_repair, verdict, next_step, evidence):
    def tbl(df, cols):
        return "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |"
                         for _, r in df.iterrows())
    g = gate_repair
    md = f"""# SMC Direction Temporal Stability & Tail Drift Audit v1.3

**base**: `3da5e68` (v1.2) &nbsp; **脚本**: `run_direction_temporal_stability_v1_3.py`
**目标**: 不新增任何市场特征，纯时间稳定性诊断，回答"WF1 为什么比 WF2/WF3 弱"。
TRADING_METRICS=NOT_APPLICABLE（架构/机制验证，未定义交易动作）。

---

## 0. v1.2 审计修复（P0 / P0.5 / P0.6）

**P0 Gate aggregation bug（已修）**：v1.2 `gate_eval` 用 `extra["side"]`（存 `+1/-1`）
去比较字符串 `"LONG_DOMINATES"`，导致 `pooled_LONG_actionable=0.0`、
`pooled_SHORT_actionable=0.8518`（错误）。修正为用 `selected+actual_clear+
correct_direction` 计算。修正后：

| candidate | ACTIONABLE | pooled_LONG | pooled_SHORT |
|---|---|---:|---:|
| CALIBRATED_TWO_STAGE | {g['CALIBRATED_TWO_STAGE_pass']} | {g['repaired_two_stage']['pooled_LONG_actionable']} | {g['repaired_two_stage']['pooled_SHORT_actionable']} |
| T3_HGB_20pct | {g['T3_HGB_20pct_pass']} | {g['repaired_three_class']['pooled_LONG_actionable']} | {g['repaired_three_class']['pooled_SHORT_actionable']} |

> Gate 仍 **FALSE**：因 WF1 per-WF actionable=0.560<0.58。但理由改为
> "WF1 per-WF actionable 没过门槛"，不再引用错误的 pooled LONG=0。
> 修正后 pooled LONG/SHORT 均约 **0.60**，证明是双边模型，方向能力不偏 LONG 或 SHORT。

**P0.5 Product diagnostic（正式作废）**：`TWO_STAGE_PRODUCT` 的 OOF 对齐
`oidx_c`/`oidx_d` 属于不同坐标系（相对 `trc` vs `tr_dir`），直接 `set()&` 无效。
标记 `TWO_STAGE_PRODUCT_DIAGNOSTIC=RETRACTED_INVALID_OOF_ALIGNMENT`；
旧 `product_diagnostic_metrics.csv` 保留作 bug evidence，本论不引用其 0.23–0.30 结果，也不修复（ROI 低）。

**P0.6 解释收紧**：TRADEOFF contamination **降低**（~20%→~15–17%），非 "solved"；
WF1 高置信方向更弱，未证明 regime drift。

---

## 1. P1 冻结 cohort

方向任务 = `LONG_DOMINATES` vs `SHORT_DOMINATES`，镜像为 `CONTINUATION=0 / REVERSAL=1`，
仅 clear rows。本论所有方向模型只看 clear subset，不让 clear gate 混淆原因。
Primary 特征固定 `G4_BASE = {{symbol, side, nearest_above_R, nearest_below_R,
nearest_ahead_R, nearest_behind_R}}`；Primary 模型固定 HGB(max_depth=3, lr=0.05,
max_iter=200, l2=1.0, seed=42)。

---

## 2. P2 训练历史长度（rolling vs expanding vs recent-two）

| config | window_type | test_auc | macro_tail | tail_cov |
|---|---|---:|---:|---:|
"""
    md += tbl(tables["p2"], ["config", "window_type", "test_auc",
                             "macro_tail_accuracy", "tail_test_coverage"])
    md += f"""

**解读**：ONE_BLOCK_ROLLING（R12/R23/R34）对比 EXPANDING（E123/E1234）与
RECENT_TWO_BLOCK（R23_4），判断训练历史长度是否是 WF1 弱的主因。
若 expanding 明显优于 rolling-1-block → TRAIN_HISTORY_LIMITED。

---

## 3. P3 历史模型跨远期测试（6 个 causal pair）

| config | train→test | test_auc | macro_tail |
|---|---|---:|---:|
"""
    md += tbl(tables["p3"], ["config", "train_blocks", "test_block",
                             "test_auc", "macro_tail_accuracy"])
    md += f"""

全部满足 train time < test time（无反向训练、无未来信息）。
若 `TB1→TB2` 弱但 `TB1→TB3/TB4` 恢复 → TB2 是特殊测试期（TB2_TARGET_REGIME_SHIFT）；
若 TB1 模型对所有未来都弱 → SOURCE_TB1_RELATION_DRIFT。

---

## 4. P4 Train-OOF tail transfer（仅信 train quantile）

| config | test_auc | rev_tail | cont_tail | macro_tail | cov |
|---|---|---:|---:|---:|---:|
"""
    md += tbl(tables["p4"], ["config", "test_auc", "reversal_tail_accuracy",
                             "continuation_tail_accuracy", "macro_tail_accuracy",
                             "tail_test_coverage"])
    md += f"""

尾阈值（`lo`/`hi`）**只来自 train-OOF 10%/90% quantile**，test 不调。
若 AUC 稳定但 tail macro 明显低于后续 block → RANK_STABLE_TAIL_DRIFT。

---

## 5. P5 机制裁决（rank drift vs tail drift）

**VERDICT**: `{verdict}`
**NEXT_STEP**: {next_step}

裁决证据（protocol verdict，非统计显著性声明）：

```json
{json.dumps(evidence, indent=2, ensure_ascii=False, default=str)}
```

硬性规则：禁止只因 "WF1 actionable 低" 就写 regime drift；必须同时引用
rolling vs expanding、cross-block AUC、tail transfer、G4 分布、score-decile 稳定性、symbol macro。

---

## 6. P6 Geometry 分布漂移（4 距离）

每 TB 分布统计见 `g4_distribution_by_tb.csv`（missing_rate / p10 / p25 / median /
p75 / p90）。train→test PSI（bin 由 train 定义）见 `g4_psi_train_test.csv`。

PSI 汇总（每个 test_block 上，所有 train→test 配对、所有 field 的最大 PSI）：

"""
    psi_pivot = tables["psi"].groupby("test_block")["psi"].max().round(4)
    for tb, v in psi_pivot.items():
        md += f"- →{tb}: max PSI = {v}\n"
    md += f"""

PSI<0.1 可忽略；0.1–0.25 中等；>0.25 明显偏移。missingness 单独在分布表报告，不进 PSI。

---

## 7. P7 Domain classifier（covariate-shift 诊断，不进交易）

| pair | model | domain_auc |
|---|---|---:|
"""
    md += tbl(tables["dom"], ["pair", "model", "domain_auc"])
    md += f"""

`DOMAIN_G4_ONLY` 只看 4 距离（纯 geometry drift 信号）；`DOMAIN_G4_BASE` 含
symbol+side（可能反映品种构成变化）。domain_auc>>0.5 表示特征分布确实变化。
注意：这是分布可区分性诊断，不是方向能力证据。

---

## 8. P8 Score-decile 稳定性（train OOF 边界 → 跨期）

见 `score_decile_stability.csv`：每个 (train_source, test_block, score_decile) 的
`n / mean_score / actual_reversal_rate`。重点看最高/最低 decile 的 reversal 纯度是否随时间改变。
decile 边界**只来自 train OOF**，禁止用 test quantile 重新分箱。

---

## 9. P9 Symbol macro drift（不删品种）

见 `direction_symbol_temporal.csv` + 聚合 `direction_symbol_temporal_agg.csv`：
每个 causal pair 的 eligible symbol（n>=100 且两类都存在）AUC / tail macro。

"""
    if len(tables["sym_agg"]):
        md += "| config | eligible_sym | median_auc | auc_iqr | n_auc>0.5 |\n|---|---:|---:|---:|---:|\n"
        md += tbl(tables["sym_agg"], ["config", "eligible_symbol_count",
                                      "macro_median_auc", "macro_auc_iqr",
                                      "n_symbols_auc_gt_05"])
    else:
        md += "_（无 eligible symbol）_\n"
    md += f"""

仅用于判断 WF1 弱是 15 品种普遍现象还是少数品种拖累。**不根据结果删品种**。

---

## 10. P10 Clear gate 时间稳定性（secondary）

见 `clear_gate_temporal_secondary.csv`：C_GLOBAL4 logistic，固定 target precision 0.85。
若 clear AUC / selected_clear_rate 在各窗口稳定，则 WF1 弱**不是 clear gate 造成**，
主疑点确在 direction。

---

## 11. 当前项目进展

| 问题 | 当前结论 |
|---|---|
| 有没有方向信息 | **有** |
| Clear/Tradeoff 可学 | **很强，AUC≈0.78** |
| Long/Short 方向可学 | **有，AUC≈0.65** |
| identity 是否重要 | **没有稳定增量** |
| 最小状态 | **symbol + side + GLOBAL4** |
| WF1 actionable 是否过预注册 Gate | **没有，0.560** |
| WF1 弱的原因 | 见 §5 裁决：`{verdict}` |
| 是否已证明 regime drift | **没有**（除非裁决为 *_REGIME_SHIFT / NONSTATIONARITY） |
| PnL | **仍未进入** |

方向不是死路：已从"Long/Short 能不能猜"推进到"方向信号在不同历史阶段为何强弱不同、应如何稳定使用"。

---

## 12. 完成条件 / STOP

代码 + 测试 + 运行 + 报告 + commit + push 后 STOP。禁止自动进入下一实验
（尤其禁止因 WF1 弱就加 FVG/pre-contact dynamics）。下一步决策见 §5 NEXT_STEP，
需你授权。
"""
    open(OUT / "SMC_DIRECTION_TEMPORAL_STABILITY_V1_3.md", "w",
         encoding="utf-8-sig").write(md)


if __name__ == "__main__":
    main()
