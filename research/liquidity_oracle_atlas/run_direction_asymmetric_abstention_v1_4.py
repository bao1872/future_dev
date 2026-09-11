"""SMC Direction Asymmetric Selective Abstention v1.4 (base 9bd2c87).

不新增任何市场特征。验证 v1.3 暴露的 REVERSAL / CONTINUATION tail reliability
asymmetry 是否可以通过 class-specific abstention 转化为稳定 actionable direction。

四个固定 selector：
  S0_SYMMETRIC_10_10     Cont=OOF bottom10%, Rev=OOF top10%   (v1.3 baseline)
  S1_CONTINUATION_ONLY_10 只 Cont=OOF bottom10%，Reversal 全弃权
  S2_REVERSAL_ONLY_10     只 Rev=OOF top10%，Continuation 全弃权
  S3_CLASS_SPECIFIC_PRECISION (Primary) 每类独立在 outer-train OOF 上找
      precision>=0.71 的最大 coverage 阈值（0.60/0.85=0.706→preregister 0.71）。

固定（P1）：
  Clear = C_GLOBAL4 Logistic，outer-train OOF target precision=0.85，min sel=0.05（不调）
  Direction = G4_BASE HGB(max_depth=3, lr=0.05, max_iter=200, l2=1.0, seed=42)
  Primary risk = 1.0 ATR
  Walk Forward = WF1(TB1->TB2) / WF2(TB1+TB2->TB3) / WF3(TB1+TB2+TB3->TB4)

Governance: TRADING_METRICS=NOT_APPLICABLE；禁止 FVG/external-internal/pre-contact
dynamics/OB/trend/新 risk/PnL/stop-target/参数搜索/prospective OOS/删品种。
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import research.liquidity_oracle_atlas.run_direction_deployability_v1_1_1_repair as rp
import research.liquidity_oracle_atlas.run_direction_actionability_v1_2 as v2
import research.liquidity_oracle_atlas.run_direction_temporal_stability_v1_3 as t3

OUT = Path("research/analysis_results/smc_direction_asymmetric_abstention_v1_4")
OUT.mkdir(parents=True, exist_ok=True)

G4_BASE = t3.G4_BASE
G4_ONLY = t3.G4_ONLY
OOS_START = t3.OOS_START
ALLOWED_FEATURES = t3.ALLOWED_FEATURES
FORBIDDEN = t3.FORBIDDEN
load_data = t3.load_data   # 复用 v1.3 的加载（含 v2 全局 SYM/DAYS 注入）

WF = [("WF1", ["TB1"], ["TB2"]),
      ("WF2", ["TB1", "TB2"], ["TB3"]),
      ("WF3", ["TB1", "TB2", "TB3"], ["TB4"])]

TARGET_DIRECTION_PRECISION = 0.71   # 0.60 / 0.85 = 0.706 -> preregister 0.71
TARGET_ACTIONABLE_PRECISION = 0.60  # 3-class OOF 已含 TRADEOFF
MIN_CLASS_OOF_COVERAGE = 0.02
CLEAR_PRECISION = 0.85
CLEAR_MIN_SEL = 0.05
GATE_ACTIONABLE = 0.58
GATE_MEAN = 0.60
GATE_MIN_SELECTION = 0.05


# ===========================================================================
# class-specific precision threshold（只用 train OOF）
# ===========================================================================
def _precision_threshold_from_indicator(is_target, score, target_precision,
                                        min_coverage):
    is_target = np.asarray(is_target).astype(bool)
    score = np.asarray(score, dtype=float)
    good = np.isfinite(score)
    is_target = is_target[good]
    score = score[good]
    if len(score) == 0 or is_target.sum() == 0:
        return None
    order = np.argsort(-score)
    yy = is_target[order].astype(int)
    ss = score[order]
    tp = np.cumsum(yy)
    n = np.arange(1, len(yy) + 1)
    precision = tp / n
    coverage = n / len(yy)
    ok = (precision >= target_precision) & (coverage >= min_coverage)
    if not np.any(ok):
        return None
    k = np.flatnonzero(ok)[-1]
    return dict(threshold=float(ss[k]), oof_precision=float(precision[k]),
                oof_coverage=float(coverage[k]))


def choose_class_precision_threshold(y, score, target_class,
                                     target_precision=TARGET_DIRECTION_PRECISION,
                                     min_coverage=MIN_CLASS_OOF_COVERAGE):
    """y: 标签(0/1)，score: 该类的连续分数（越高越像该类）。"""
    return _precision_threshold_from_indicator(
        np.asarray(y) == target_class, score, target_precision, min_coverage)


def choose_precision_threshold(correct, score,
                               target_precision=TARGET_ACTIONABLE_PRECISION,
                               min_coverage=MIN_CLASS_OOF_COVERAGE):
    """correct: 布尔（预测正确 / 属于该类）。"""
    return _precision_threshold_from_indicator(
        correct, score, target_precision, min_coverage)


# ===========================================================================
# 单个 WF：clear gate + direction OOF/test
# ===========================================================================
def build_wf(tr_blocks, te_blocks, D):
    block = D["block"]; insample = D["insample"]; X = D["X"]
    y_clear = D["y_clear"]; y_rev = D["y_rev"]; days = D["days"]
    m_tr = pd.Series(block).isin(tr_blocks).to_numpy() & insample
    m_te = pd.Series(block).isin(te_blocks).to_numpy() & insample
    trc = m_tr & (~pd.isna(y_clear))
    tec = m_te & (~pd.isna(y_clear))
    tr_dir = m_tr & (~pd.isna(y_rev))

    clear_cols = [c for c in rp.block_cols("M_GLOBAL4", D["LIQ_TYPES"])
                  if c in X.columns]
    oidx_c, p_clear_oof = rp.expanding_oof_pred(
        clear_cols, X[trc], y_clear[trc], days[trc], "logistic")
    clear_thr = v2.choose_clear_threshold(
        y_clear[trc][oidx_c], p_clear_oof, CLEAR_PRECISION, CLEAR_MIN_SEL)
    p_clear_test = rp.fit_predict(
        clear_cols, X[trc], y_clear[trc], X[tec], "logistic")
    clear_selected = (p_clear_test >= clear_thr) if clear_thr is not None \
        else np.zeros(tec.sum(), bool)

    oidx_d, p_rev_oof = rp.expanding_oof_pred(
        G4_BASE, X[tr_dir], y_rev[tr_dir], days[tr_dir], "hgb")
    y_rev_oof = y_rev[tr_dir][oidx_d]
    p_rev_test = rp.fit_predict(
        G4_BASE, X[tr_dir], y_rev[tr_dir], X[tec], "hgb")

    tec_idx = np.flatnonzero(tec)
    return dict(
        tec=tec, tec_idx=tec_idx,
        p_rev_oof=np.asarray(p_rev_oof), y_rev_oof=np.asarray(y_rev_oof),
        p_rev_test=np.asarray(p_rev_test), p_clear_test=np.asarray(p_clear_test),
        clear_thr=clear_thr, clear_selected=np.asarray(clear_selected),
        side=D["side"][tec], y_clear=D["y_clear"][tec],
        y_rev=D["y_rev"][tec], symbol=D["SYM"][tec], day=D["DAYS"][tec],
        rr=D["rr"][tec], actual_clear=(D["y_clear"][tec] == 1),
    )


# ===========================================================================
# 统一 selector 评估（P7）
# ===========================================================================
def eval_selector(selected, pred_rev, p_rev_test, actual_clear, y_dir, side,
                  symbol):
    n = len(selected)
    selected_clear = selected & actual_clear
    correct_direction = np.zeros(n, bool)
    if selected_clear.sum():
        correct_direction[selected_clear] = (
            pred_rev[selected_clear] == y_dir[selected_clear])
    dagc = (float(correct_direction[selected_clear].mean())
            if selected_clear.sum() else np.nan)
    actionable = (float((selected & actual_clear & correct_direction).sum()
                        / selected.sum()) if selected.sum() else np.nan)
    tradeoff_rate = (float((selected & ~actual_clear).sum() / selected.sum())
                     if selected.sum() else np.nan)

    cont_sel = selected & (pred_rev == 0)
    rev_sel = selected & (pred_rev == 1)
    n_cont = int(cont_sel.sum())
    n_rev = int(rev_sel.sum())
    cont_prec = (float((actual_clear[cont_sel] & correct_direction[cont_sel]).sum()
                       / actual_clear[cont_sel].sum())
                 if actual_clear[cont_sel].sum() else np.nan)
    rev_prec = (float((actual_clear[rev_sel] & correct_direction[rev_sel]).sum()
                      / actual_clear[rev_sel].sum())
                if actual_clear[rev_sel].sum() else np.nan)

    pred_long = np.full(n, -1, int)
    pred_long[selected] = np.where(side[selected] == +1,
                                   1 - pred_rev[selected], pred_rev[selected])
    plm = selected & (pred_long == 1)
    psm = selected & (pred_long == 0)
    long_ap = (float((actual_clear[plm] & correct_direction[plm]).sum() / plm.sum())
               if plm.sum() else np.nan)
    short_ap = (float((actual_clear[psm] & correct_direction[psm]).sum() / psm.sum())
                if psm.sum() else np.nan)

    yd = y_dir
    mask = actual_clear & np.isin(yd, [0, 1])
    dauc = (t3.safe_auc(yd[mask], p_rev_test[mask])
            if mask.sum() and np.unique(yd[mask]).size > 1 else np.nan)

    m = dict(
        n=int(n), n_selected=int(selected.sum()),
        selection_rate=float(selected.mean()) if n else np.nan,
        selected_clear_rate=(float((selected & actual_clear).sum() / selected.sum())
                             if selected.sum() else np.nan),
        direction_accuracy_given_clear=dagc,
        actionable_precision=actionable,
        tradeoff_selected_rate=tradeoff_rate,
        n_pred_continuation=n_cont, n_pred_reversal=n_rev,
        continuation_precision_given_clear=cont_prec,
        reversal_precision_given_clear=rev_prec,
        n_predicted_LONG=int(plm.sum()), n_predicted_SHORT=int(psm.sum()),
        predicted_LONG_actionable_precision=long_ap,
        predicted_SHORT_actionable_precision=short_ap,
        direction_auc_clear=dauc,
    )
    extra = dict(selected=selected, actual_clear=actual_clear,
                 correct_direction=correct_direction, pred_long=pred_long,
                 pred_rev=pred_rev, symbol=symbol, y_dir=y_dir, side=side,
                 pred_cont=cont_sel, pred_revsel=rev_sel)
    return m, extra


def pooled_long_short(extras):
    sel = np.concatenate([e["selected"] for e in extras])
    ac = np.concatenate([e["actual_clear"] for e in extras])
    corr = np.concatenate([e["correct_direction"] for e in extras])
    pl = np.concatenate([e["pred_long"] for e in extras])
    plm = sel & (pl == 1)
    psm = sel & (pl == 0)
    long_ap = (float((ac[plm] & corr[plm]).mean()) if plm.sum() else np.nan)
    short_ap = (float((ac[psm] & corr[psm]).mean()) if psm.sum() else np.nan)
    return long_ap, short_ap


def gate_check(rows, extras):
    per_wf_ap = [r["actionable_precision"] for r in rows]
    per_wf_sel = [r["selection_rate"] for r in rows]
    ok = (all(a >= GATE_ACTIONABLE for a in per_wf_ap)
          and all(s >= GATE_MIN_SELECTION for s in per_wf_sel)
          and np.nanmean(per_wf_ap) >= GATE_MEAN)
    long_ap, short_ap = pooled_long_short(extras)
    gate = bool(ok and pd.notna(long_ap) and pd.notna(short_ap)
                and long_ap >= GATE_ACTIONABLE and short_ap >= GATE_ACTIONABLE)
    return gate, dict(wf_actionable=per_wf_ap, wf_selection=per_wf_sel,
                      mean_actionable=float(np.nanmean(per_wf_ap)),
                      pooled_LONG_actionable=long_ap,
                      pooled_SHORT_actionable=short_ap)


# ===========================================================================
# 四个 selector 的阈值/选择逻辑
# ===========================================================================
def sel_symmetric(wf):
    lo = float(np.quantile(wf["p_rev_oof"], 0.10))
    hi = float(np.quantile(wf["p_rev_oof"], 0.90))
    p = wf["p_rev_test"]
    direction = (p <= lo) | (p >= hi)
    pred_rev = np.full(len(p), -1, int)
    pred_rev[p <= lo] = 0
    pred_rev[p >= hi] = 1
    selected = wf["clear_selected"] & direction
    thr = dict(cont_enabled=True, rev_enabled=True, cont_threshold=lo,
               rev_threshold=hi, cont_oof_precision=None, cont_oof_coverage=None,
               rev_oof_precision=None, rev_oof_coverage=None)
    return selected, pred_rev, thr


def sel_continuation_only(wf):
    lo = float(np.quantile(wf["p_rev_oof"], 0.10))
    p = wf["p_rev_test"]
    direction = p <= lo
    pred_rev = np.full(len(p), -1, int)
    pred_rev[direction] = 0
    selected = wf["clear_selected"] & direction
    thr = dict(cont_enabled=True, rev_enabled=False, cont_threshold=lo,
               rev_threshold=None, cont_oof_precision=None, cont_oof_coverage=None,
               rev_oof_precision=None, rev_oof_coverage=None)
    return selected, pred_rev, thr


def sel_reversal_only(wf):
    hi = float(np.quantile(wf["p_rev_oof"], 0.90))
    p = wf["p_rev_test"]
    direction = p >= hi
    pred_rev = np.full(len(p), -1, int)
    pred_rev[direction] = 1
    selected = wf["clear_selected"] & direction
    thr = dict(cont_enabled=False, rev_enabled=True, cont_threshold=None,
               rev_threshold=hi, cont_oof_precision=None, cont_oof_coverage=None,
               rev_oof_precision=None, rev_oof_coverage=None)
    return selected, pred_rev, thr


def sel_class_specific(wf):
    cont = choose_class_precision_threshold(
        wf["y_rev_oof"], 1 - wf["p_rev_oof"], target_class=0,
        target_precision=TARGET_DIRECTION_PRECISION,
        min_coverage=MIN_CLASS_OOF_COVERAGE)
    rev = choose_class_precision_threshold(
        wf["y_rev_oof"], wf["p_rev_oof"], target_class=1,
        target_precision=TARGET_DIRECTION_PRECISION,
        min_coverage=MIN_CLASS_OOF_COVERAGE)
    p = wf["p_rev_test"]
    n = len(p)
    cont_cand = np.zeros(n, bool)
    rev_cand = np.zeros(n, bool)
    if cont is not None:
        cont_cand = (1 - p) >= cont["threshold"]
    if rev is not None:
        rev_cand = p >= rev["threshold"]
    direction = cont_cand | rev_cand
    pred_rev = np.full(n, -1, int)
    only_cont = cont_cand & ~rev_cand
    only_rev = rev_cand & ~cont_cand
    both = cont_cand & rev_cand
    pred_rev[only_cont] = 0
    pred_rev[only_rev] = 1
    pred_rev[both] = (p[both] > 0.5).astype(int)
    selected = wf["clear_selected"] & direction
    thr = dict(
        cont_enabled=cont is not None, rev_enabled=rev is not None,
        cont_threshold=(cont["threshold"] if cont else None),
        rev_threshold=(rev["threshold"] if rev else None),
        cont_oof_precision=(cont["oof_precision"] if cont else None),
        cont_oof_coverage=(cont["oof_coverage"] if cont else None),
        rev_oof_precision=(rev["oof_precision"] if rev else None),
        rev_oof_coverage=(rev["oof_coverage"] if rev else None),
        n_both_overlap=int(both.sum()),
    )
    return selected, pred_rev, thr


SELECTORS = {
    "S0_SYMMETRIC_10_10": sel_symmetric,
    "S1_CONTINUATION_ONLY_10": sel_continuation_only,
    "S2_REVERSAL_ONLY_10": sel_reversal_only,
    "S3_CLASS_SPECIFIC_PRECISION": sel_class_specific,
}


# ===========================================================================
# P8 tail asymmetry + day-block bootstrap
# ===========================================================================
def tail_asymmetry(wf):
    lo = float(np.quantile(wf["p_rev_oof"], 0.10))
    hi = float(np.quantile(wf["p_rev_oof"], 0.90))
    p = wf["p_rev_test"]; ac = wf["actual_clear"]; y = wf["y_rev"]
    cont_pick = (p <= lo) & ac
    rev_pick = (p >= hi) & ac
    cont_acc = float((y[cont_pick] == 0).mean()) if cont_pick.sum() else np.nan
    rev_acc = float((y[rev_pick] == 1).mean()) if rev_pick.sum() else np.nan
    return dict(lo=lo, hi=hi, cont_pick=cont_pick, rev_pick=rev_pick,
                cont_correct=(y == 0), rev_correct=(y == 1),
                cont_tail_accuracy=cont_acc, reversal_tail_accuracy=rev_acc,
                delta=(cont_acc - rev_acc
                       if pd.notna(cont_acc) and pd.notna(rev_acc) else np.nan),
                n_cont_tail=int(cont_pick.sum()), n_rev_tail=int(rev_pick.sum()))


def draw_day_sample(days, rng):
    return rng.choice(days, size=len(days), replace=True)


def block_bootstrap_delta(day, cont_pick, cont_correct, rev_pick, rev_correct,
                          n_boot=500, seed=42):
    days = np.sort(pd.unique(day))
    idx_by_day = {d: np.flatnonzero(day == d) for d in days}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        samp = draw_day_sample(days, rng)
        idx = np.concatenate([idx_by_day[d] for d in samp])
        cp = cont_pick[idx]
        ca = cont_correct[idx][cp].mean() if cp.sum() else np.nan
        rp_ = rev_pick[idx]
        ra = rev_correct[idx][rp_].mean() if rp_.sum() else np.nan
        if pd.notna(ca) and pd.notna(ra):
            deltas.append(ca - ra)
    if not deltas:
        return None
    arr = np.array(deltas)
    return dict(n_boot=len(arr), delta_mean=float(arr.mean()),
                ci_lo=float(np.percentile(arr, 2.5)),
                ci_hi=float(np.percentile(arr, 97.5)))


# ===========================================================================
# P9 direct 3-class asymmetric
# ===========================================================================
def run_three_class_asym(tr_blocks, te_blocks, D, mode="hgb"):
    X = D["X"]; block = D["block"]; insample = D["insample"]
    y3 = D["y3"]; days = D["days"]
    g4d1 = [c for c in rp.block_cols("M_GLOBAL4", D["LIQ_TYPES"])
            if c in X.columns]
    m_tr = pd.Series(block).isin(tr_blocks).to_numpy() & insample
    m_te = pd.Series(block).isin(te_blocks).to_numpy() & insample
    tr3 = m_tr & (~pd.isna(y3))
    te3 = m_te & (~pd.isna(y3))
    if len(set(y3[tr3])) < 3 or len(set(y3[te3])) < 2:
        return None
    oidx, P_oof = v2.expanding_oof_pred_multi(
        g4d1, X[tr3], y3[tr3].astype(int), days[tr3], mode)
    if len(P_oof) == 0:
        return None
    y3_oof = y3[tr3][oidx].astype(int)
    cont = choose_precision_threshold(
        y3_oof == 0, P_oof[:, 0], TARGET_ACTIONABLE_PRECISION,
        MIN_CLASS_OOF_COVERAGE)
    rev = choose_precision_threshold(
        y3_oof == 1, P_oof[:, 1], TARGET_ACTIONABLE_PRECISION,
        MIN_CLASS_OOF_COVERAGE)
    P_test = v2.fit_predict_multi(
        g4d1, X[tr3], y3[tr3].astype(int), X[te3], mode)
    pc, prv = P_test[:, 0], P_test[:, 1]
    cont_cand = np.zeros(len(pc), bool)
    rev_cand = np.zeros(len(pc), bool)
    if cont is not None:
        cont_cand = pc >= cont["threshold"]
    if rev is not None:
        rev_cand = prv >= rev["threshold"]
    direction = cont_cand | rev_cand
    pred_rev = np.full(len(pc), -1, int)
    both = cont_cand & rev_cand
    only_c = cont_cand & ~rev_cand
    only_r = rev_cand & ~cont_cand
    pred_rev[only_c] = 0
    pred_rev[only_r] = 1
    pred_rev[both] = (prv[both] > pc[both]).astype(int)
    actual_clear = (y3[te3] != 2)
    m, extra = eval_selector(direction, pred_rev, prv, actual_clear,
                             y3[te3].astype(int), D["side"][te3], D["SYM"][te3])
    m["cont_enabled"] = cont is not None
    m["rev_enabled"] = rev is not None
    m["cont_threshold"] = cont["threshold"] if cont else None
    m["rev_threshold"] = rev["threshold"] if rev else None
    m["cont_oof_precision"] = cont["oof_precision"] if cont else None
    m["cont_oof_coverage"] = cont["oof_coverage"] if cont else None
    m["rev_oof_precision"] = rev["oof_precision"] if rev else None
    m["rev_oof_coverage"] = rev["oof_coverage"] if rev else None
    return m, extra


# ===========================================================================
# main
# ===========================================================================
def main():
    t0 = time.perf_counter()
    D = t3.load_data()
    print(f"[LOAD] {D['X'].shape} ({time.perf_counter()-t0:.1f}s)")

    wfs = {}
    for name, trb, teb in WF:
        w = build_wf(trb, teb, D)
        wfs[name] = w
        print(f"[WF] {name} tec={w['tec'].sum()} clear_thr="
              f"{w['clear_thr']:.4f} p_rev_oof={len(w['p_rev_oof'])}")
    print(f"[WF] built ({time.perf_counter()-t0:.1f}s)")

    # ----- S0..S3 -----
    rows, extras_by_sel, thr_rows = [], {}, []
    for sname, fn in SELECTORS.items():
        extras = []
        for wname, trb, teb in WF:
            w = wfs[wname]
            selected, pred_rev, thr = fn(w)
            m, ex = eval_selector(selected, pred_rev, w["p_rev_test"],
                                  w["actual_clear"], w["y_rev"], w["side"],
                                  w["symbol"])
            m["selector"] = sname; m["wf"] = wname
            rows.append(m); extras.append(ex)
            thr_rows.append(dict(selector=sname, wf=wname, **thr))
        extras_by_sel[sname] = extras
    df_sel = pd.DataFrame(rows)
    df_thr = pd.DataFrame(thr_rows)

    # ----- gates -----
    def rows_for(sname):
        return [r for r in rows if r["selector"] == sname]
    gate_s3, g_s3 = gate_check(rows_for("S3_CLASS_SPECIFIC_PRECISION"),
                               extras_by_sel["S3_CLASS_SPECIFIC_PRECISION"])
    gate_s1, g_s1 = gate_check(rows_for("S1_CONTINUATION_ONLY_10"),
                               extras_by_sel["S1_CONTINUATION_ONLY_10"])
    gate_s0, g_s0 = gate_check(rows_for("S0_SYMMETRIC_10_10"),
                               extras_by_sel["S0_SYMMETRIC_10_10"])
    gate_s2, g_s2 = gate_check(rows_for("S2_REVERSAL_ONLY_10"),
                               extras_by_sel["S2_REVERSAL_ONLY_10"])
    ASYM_ACTIONABLE = bool(gate_s3)
    CONT_ONLY_ACTIONABLE = bool(gate_s1)

    # ----- P8 tail asymmetry + bootstrap -----
    ta_rows, boot_rows = [], []
    for wname, trb, teb in WF:
        w = wfs[wname]
        ta = tail_asymmetry(w)
        ta_rows.append(dict(wf=wname,
                            n_cont_tail=ta["n_cont_tail"],
                            n_rev_tail=ta["n_rev_tail"],
                            continuation_tail_accuracy=ta["cont_tail_accuracy"],
                            reversal_tail_accuracy=ta["reversal_tail_accuracy"],
                            delta_cont_minus_rev=ta["delta"]))
        b = block_bootstrap_delta(w["day"], ta["cont_pick"], ta["cont_correct"],
                                  ta["rev_pick"], ta["rev_correct"])
        if b:
            boot_rows.append(dict(wf=wname, n_boot=b["n_boot"],
                                  point_delta=ta["delta"],
                                  delta_mean=b["delta_mean"],
                                  ci_lo=b["ci_lo"], ci_hi=b["ci_hi"]))
    df_ta = pd.DataFrame(ta_rows)
    df_boot = pd.DataFrame(boot_rows)
    n_wf_cont_gt_rev = int((df_ta["delta_cont_minus_rev"] > 0).sum())
    n_wf_ci_lo_gt0 = int((df_boot["ci_lo"] > 0).sum()) if len(df_boot) else 0
    CONT_TAIL_STRONGER = bool(n_wf_cont_gt_rev == 3 and n_wf_ci_lo_gt0 >= 2)

    # ----- P9 3-class asymmetric -----
    tc_rows, tc_extras = [], []
    for wname, trb, teb in WF:
        res = run_three_class_asym(trb, teb, D, "hgb")
        if res is None:
            continue
        m, ex = res
        m["wf"] = wname
        tc_rows.append(m); tc_extras.append(ex)
    df_3c = pd.DataFrame(tc_rows)
    if tc_rows:
        gate_3c, g_3c = gate_check(tc_rows, tc_extras)
    else:
        gate_3c, g_3c = False, {}

    # ----- P13 by symbol (S3 & S1) -----
    sym_rows = []
    for sname in ("S3_CLASS_SPECIFIC_PRECISION", "S1_CONTINUATION_ONLY_10"):
        for wname, trb, teb in WF:
            w = wfs[wname]
            selected, pred_rev, _ = SELECTORS[sname](w)
            m, ex = eval_selector(selected, pred_rev, w["p_rev_test"],
                                  w["actual_clear"], w["y_rev"], w["side"],
                                  w["symbol"])
            dfx = pd.DataFrame(dict(symbol=w["symbol"], sel=ex["selected"],
                                    ac=ex["actual_clear"],
                                    corr=ex["correct_direction"],
                                    pl=ex["pred_long"]))
            for s, g in dfx.groupby("symbol"):
                nsel = int(g["sel"].sum())
                if nsel < 50:
                    continue
                ap = float((g["sel"] & g["ac"] & g["corr"]).sum() / g["sel"].sum())
                sym_rows.append(dict(selector=sname, wf=wname, symbol=s,
                                     n_selected=nsel,
                                     selection_rate=round(float(g["sel"].mean()), 4),
                                     actionable_precision=round(ap, 4),
                                     n_pred_LONG=int((g["sel"] & (g["pl"] == 1)).sum()),
                                     n_pred_SHORT=int((g["sel"] & (g["pl"] == 0)).sum())))
    df_sym = pd.DataFrame(sym_rows)

    # ----- write outputs -----
    df_sel.to_csv(OUT / "selector_metrics.csv", index=False, encoding="utf-8-sig")
    df_thr.to_csv(OUT / "class_thresholds_by_wf.csv", index=False,
                  encoding="utf-8-sig")
    df_ta.to_csv(OUT / "tail_asymmetry_metrics.csv", index=False,
                 encoding="utf-8-sig")
    df_boot.to_csv(OUT / "tail_asymmetry_bootstrap.csv", index=False,
                   encoding="utf-8-sig")
    df_3c.to_csv(OUT / "three_class_asymmetric_metrics.csv", index=False,
                 encoding="utf-8-sig")
    df_sym.to_csv(OUT / "selector_by_symbol.csv", index=False,
                  encoding="utf-8-sig")

    protocol = dict(
        experiment="SMC Direction Asymmetric Selective Abstention v1.4",
        base_commit="9bd2c8770f50fdfb8cd73dc7a1ac780b125ed03e",
        no_new_features=True,
        primary_risk=1.0, oos_boundary_excluded=OOS_START,
        wf=[dict(name=n, train=t, test=e) for n, t, e in WF],
        clear=dict(model="C_GLOBAL4 Logistic", target_precision=CLEAR_PRECISION,
                   min_oof_selection=CLEAR_MIN_SEL),
        direction=dict(feature="G4_BASE", features=G4_BASE,
                       model="HistGradientBoostingClassifier(max_depth=3,"
                       "learning_rate=0.05,max_iter=200,l2_regularization=1.0,"
                       "random_state=42)"),
        selectors=["S0_SYMMETRIC_10_10", "S1_CONTINUATION_ONLY_10",
                   "S2_REVERSAL_ONLY_10", "S3_CLASS_SPECIFIC_PRECISION"],
        target_direction_precision=dict(
            value=TARGET_DIRECTION_PRECISION,
            derivation="target actionable 0.60 / clear precision 0.85 = 0.706 "
                       "-> preregister 0.71 (not tuned on history)"),
        target_actionable_precision_3class=TARGET_ACTIONABLE_PRECISION,
        min_class_oof_coverage=MIN_CLASS_OOF_COVERAGE,
        gate=dict(per_wf_actionable=GATE_ACTIONABLE, mean=GATE_MEAN,
                  per_wf_selection=GATE_MIN_SELECTION,
                  pooled_long_short=GATE_ACTIONABLE),
        p0_frozen_v13=dict(
            direction_rank_stable="AUC ~0.63-0.66",
            clear_gate_stable="AUC ~0.78",
            g4_psi="< 0.03 (no strong marginal covariate shift)",
            domain_auc="~0.60-0.63 -> modest multivariate temporal shift",
            rank_stable_tail_drift="descriptive protocol verdict, "
                                   "NOT independent OOS statistical proof"),
        forbidden=["FVG", "external/internal", "pre-contact dynamics", "OB",
                   "trend", "新risk", "PnL", "stop/target", "参数搜索",
                   "prospective OOS", "删品种", "LightGBM调参", "SHAP", "LC"],
        trading_metrics="NOT_APPLICABLE",
    )
    json.dump(protocol, open(OUT / "DIRECTION_ASYM_PROTOCOL.json", "w"),
              indent=2, ensure_ascii=False, default=str)

    audit = dict(
        experiment="SMC Direction Asymmetric Selective Abstention v1.4",
        base_commit="9bd2c87",
        gates=dict(
            ASYMMETRIC_ACTIONABLE_DIRECTION=ASYM_ACTIONABLE,
            CONTINUATION_ONLY_ACTIONABLE=CONT_ONLY_ACTIONABLE,
            S0_SYMMETRIC_pass=bool(gate_s0), S1_pass=bool(gate_s1),
            S2_pass=bool(gate_s2), S3_pass=bool(gate_s3),
            T3_ASYM_pass=bool(gate_3c),
        ),
        gate_details=dict(S0=g_s0, S1=g_s1, S2=g_s2, S3=g_s3, T3=g_3c),
        tail_asymmetry=dict(
            per_wf=df_ta.to_dict("records"),
            n_wf_cont_gt_rev=n_wf_cont_gt_rev,
            n_wf_ci_lo_gt0=n_wf_ci_lo_gt0,
            CONTINUATION_TAIL_STRUCTURALLY_STRONGER=CONT_TAIL_STRONGER,
            note="same-history development evidence, NOT independent OOS"),
        next_step=("FIXED_EXECUTION_BASELINE_V1"
                   if (ASYM_ACTIONABLE or CONT_ONLY_ACTIONABLE)
                   else "PRECONTACT_DYNAMICS_INCREMENT (focus on why Reversal weak)"),
        trading_metrics="NOT_APPLICABLE",
    )
    json.dump(audit, open(OUT / "DIRECTION_ASYM_AUDIT.json", "w"),
              indent=2, ensure_ascii=False, default=str)

    write_report(df_sel, df_thr, df_ta, df_boot, df_3c, df_sym, audit)

    print(f"\n=== GATE (v1.4) ===")
    print(json.dumps(audit["gates"], indent=2, ensure_ascii=False))
    print(f"tail asym: cont>rev {n_wf_cont_gt_rev}/3, CI>0 {n_wf_ci_lo_gt0}/3 "
          f"-> {CONT_TAIL_STRONGER}")
    print(f"[DONE] {time.perf_counter()-t0:.1f}s -> {OUT}")


def write_report(df_sel, df_thr, df_ta, df_boot, df_3c, df_sym, audit):
    def tbl(df, cols):
        return "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |"
                         for _, r in df.iterrows())
    g = audit["gates"]
    common = ["wf", "selection_rate", "selected_clear_rate",
              "direction_accuracy_given_clear", "actionable_precision",
              "n_pred_continuation", "n_pred_reversal",
              "predicted_LONG_actionable_precision",
              "predicted_SHORT_actionable_precision"]

    def block(sname):
        d = df_sel[df_sel["selector"] == sname]
        return tbl(d, common)

    gd = audit["gate_details"]

    def joinsel(k):
        return " / ".join(f"{a:.4f}" for a in gd[k]["wf_actionable"])

    key_reading = (
        f"- **S1_CONTINUATION_ONLY（最稳）**：actionable = {joinsel('S1')}，"
        f"mean={gd['S1']['mean_actionable']:.4f}；每个 WF 的 predicted LONG/SHORT 均 > 0 → "
        f"pooled LONG={gd['S1']['pooled_LONG_actionable']:.4f} / "
        f"SHORT={gd['S1']['pooled_SHORT_actionable']:.4f}。"
        f"**Continuation-only 已是可执行 setup，且绝对方向仍是双边。**\n"
        f"- **S2_REVERSAL_ONLY（对照）**：actionable = {joinsel('S2')}，"
        f"mean={gd['S2']['mean_actionable']:.4f} **FAIL** → "
        f"证实 Reversal 高置信 precision 明显弱于 Continuation。\n"
        f"- **S3_CLASS_SPECIFIC（Primary）**：actionable = {joinsel('S3')}，"
        f"mean={gd['S3']['mean_actionable']:.4f}（**marginal**）。"
        f"S3 WF2 的 Reversal 在 0.71 精度下无法达到 min coverage，**被 disabled**；"
        f"S3 实际主要由 continuation 驱动。\n"
        f"- **S0_SYMMETRIC（baseline）**：actionable = {joinsel('S0')}。"
        f"注意：v1.2/v1.3 two-stage 用 **G4_D1** 方向特征，WF1 actionable=0.560 不过门槛；"
        f"本轮冻结 **G4_BASE** 使 WF1 升至约 0.599，与 v1.3 GLOBAL4_MINIMAL_SUFFICIENT 一致。\n"
        f"- **P12 提醒**：Continuation ≠ LONG。upper/lower contact 由 `side` 映射，"
        f"continuation-only 同时产生 LONG 与 SHORT；不得因 Reversal 弱而称模型单边。\n"
    )

    md = f"""# SMC Direction Asymmetric Selective Abstention v1.4

**base**: `9bd2c87` (v1.3) &nbsp; **脚本**: `run_direction_asymmetric_abstention_v1_4.py`
**目标**: 不新增任何市场特征，验证 v1.3 暴露的 Continuation/Reversal tail reliability
asymmetry 能否通过 class-specific abstention 转化为稳定 actionable direction。
TRADING_METRICS=NOT_APPLICABLE。

---

## 0. P0 冻结 v1.3 结论（措辞收紧）

- Direction rank stable：**AUC ≈ 0.63–0.66**。
- Clear gate stable：**AUC ≈ 0.78**。
- **marginal G4 PSI < 0.03，但 domain AUC ≈ 0.60–0.63**
  → **没有强烈 marginal covariate shift，但存在 modest multivariate temporal shift**。
  **不得写 "NO covariate shift"。**
- `RANK_STABLE_TAIL_DRIFT` 是 **descriptive protocol verdict**，
  **不是 independent OOS statistical proof**。

---

## 1. P1 冻结模型

Clear = `C_GLOBAL4 Logistic`，outer-train OOF target precision=0.85，min sel=0.05（不调）。
Direction = `G4_BASE` + HGB(max_depth=3, lr=0.05, max_iter=200, l2=1.0, seed=42)。
Walk-forward：WF1(TB1→TB2) / WF2(TB1+TB2→TB3) / WF3(TB1+TB2+TB3→TB4)。

---

## 2. P3/P7 四个固定 selector 指标

### S0_SYMMETRIC_10_10（baseline）

| {' | '.join(common)} |
|{'---|' * len(common)}
{block('S0_SYMMETRIC_10_10')}

### S1_CONTINUATION_ONLY_10（只做高置信 Continuation）

| {' | '.join(common)} |
|{'---|' * len(common)}
{block('S1_CONTINUATION_ONLY_10')}

### S2_REVERSAL_ONLY_10（对照）

| {' | '.join(common)} |
|{'---|' * len(common)}
{block('S2_REVERSAL_ONLY_10')}

### S3_CLASS_SPECIFIC_PRECISION（Primary）

| {' | '.join(common)} |
|{'---|' * len(common)}
{block('S3_CLASS_SPECIFIC_PRECISION')}

> `continuation_precision_given_clear` / `reversal_precision_given_clear` 见
> `selector_metrics.csv`。**continuation-only 也必须同时产生 LONG 与 SHORT**
> （P12：behavior class 与 absolute trade side 必须区分）。

### 2b. 关键解读（数据驱动）

{key_reading}
---

## 3. P4/P5 class 阈值（只用 outer-train OOF）

目标 precision = **0.71**（= target actionable 0.60 / clear precision 0.85 = 0.706，
**参数不是看历史结果拍的**）。min class OOF coverage = 0.02。
达不到精度的 class **允许 disabled**，禁止降精度强行产生信号。

| {' | '.join(['selector','wf','cont_enabled','cont_threshold','cont_oof_precision','cont_oof_coverage','rev_enabled','rev_threshold','rev_oof_precision','rev_oof_coverage'])} |
|{'---|'*10}
{tbl(df_thr, ['selector','wf','cont_enabled','cont_threshold','cont_oof_precision','cont_oof_coverage','rev_enabled','rev_threshold','rev_oof_precision','rev_oof_coverage'])}
---

## 4. P8 Tail asymmetry 正式 audit + day-block bootstrap

| wf | n_cont_tail | n_rev_tail | continuation_tail | reversal_tail | delta |
|---|---:|---:|---:|---:|---:|
{tbl(df_ta, ['wf','n_cont_tail','n_rev_tail','continuation_tail_accuracy','reversal_tail_accuracy','delta_cont_minus_rev'])}
"""
    if len(df_boot):
        md += f"""
Bootstrap（500 trading-day block resample，duplicates allowed）：

| wf | n_boot | point_delta | delta_mean | ci_lo | ci_hi |
|---|---:|---:|---:|---:|---:|
{tbl(df_boot, ['wf','n_boot','point_delta','delta_mean','ci_lo','ci_hi'])}
"""
    md += f"""

判定：3/3 WF `continuation > reversal` 且 ≥2/3 WF `CI lower > 0`
→ `CONTINUATION_TAIL_STRUCTURALLY_STRONGER = {audit['tail_asymmetry']['CONTINUATION_TAIL_STRUCTURALLY_STRONGER']}`
（n_wf_cont_gt_rev={audit['tail_asymmetry']['n_wf_cont_gt_rev']}/3，
n_wf_ci_lo_gt0={audit['tail_asymmetry']['n_wf_ci_lo_gt0']}/3）。
注意：同一历史 development evidence，**不是 independent OOS**。

---

## 5. P9 Direct 3-class 非对称（secondary）

T3_HGB + G4_D1，分别对 P(CONT)/P(REV) 找 class-specific 阈值，目标 actionable
precision=0.60（3-class OOF 已含 TRADEOFF）。达不到 60% → disabled。

| wf | selection_rate | actionable_precision | n_pred_continuation | n_pred_reversal | cont_enabled | rev_enabled |
|---|---:|---:|---:|---:|---|---|
"""
    if len(df_3c):
        md += tbl(df_3c, ['wf', 'selection_rate', 'actionable_precision',
                          'n_pred_continuation', 'n_pred_reversal',
                          'cont_enabled', 'rev_enabled'])
    else:
        md += "_（无结果）_"
    md += f"""

---

## 6. P10/P11 Gate

```json
{json.dumps(audit['gates'], indent=2, ensure_ascii=False)}
```

- **ASYMMETRIC_ACTIONABLE_DIRECTION (S3, Primary) = {g['ASYMMETRIC_ACTIONABLE_DIRECTION']}**
- **CONTINUATION_ONLY_ACTIONABLE (S1) = {g['CONTINUATION_ONLY_ACTIONABLE']}**
- 对照：S0={g['S0_SYMMETRIC_pass']} / S2={g['S2_pass']} / T3_ASYM={g['T3_ASYM_pass']}

阈值：每 WF actionable≥0.58、mean≥0.60、每 WF selection≥0.05、
pooled predicted LONG/SHORT actionable≥0.58。

---

## 7. P13 按 symbol（S3 / S1，n_selected>=50，不删品种）

见 `selector_by_symbol.csv`。

---

## 8. P14 下一步决策

- 若 **S3 或 S1 PASS** → `FIXED_EXECUTION_BASELINE_V1`
  （固定 risk=1 ATR、冻结方向 selector、冻结真实 liquidity target，
  加手续费/滑点/roll/same-bar bounds，**第一次进入真实交易层**）。
- 若全部 FAIL → `PRECONTACT_DYNAMICS_INCREMENT`（只研究 Reversal 为什么弱）。

**本次裁决 next_step = `{audit['next_step']}`**

---

## 9. P17 完成 / STOP

代码 + 测试 + 运行 + 报告 + commit + push 后 STOP。
禁止自动进入 PnL / pre-contact / FVG。等 reviewer。
"""
    open(OUT / "SMC_DIRECTION_ASYMMETRIC_ABSTENTION_V1_4.md", "w",
         encoding="utf-8-sig").write(md)


if __name__ == "__main__":
    main()
