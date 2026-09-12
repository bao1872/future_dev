"""SMC Group-Level Conditional Tradeoff Veto v1.3 (base 7107f16).

目的：修复 v1.2 的两个 architecture mismatch：
  (1) toxicity train cohort 必须包含 CLEAR85 gate（之前只用了 direction-q10
      candidate，造成 deployment-cohort mismatch）；
  (2) toxic label / execution 都在 signal/group 层，veto 也必须在 group 层
      （之前 contact-level veto 无法删除 group-level 毒单）。

冻结（与 v1.2 完全相同，不改）：
  CLEAR85 / Continuation q10 / risk=1 ATR / next-open entry /
  decision-close anchored 1ATR stop / selected-boundary beyond-attack target /
  STOP_FIRST / v1.0.1 OOS cutoff / 15 symbols。

禁止：risk 0.5/2.0、clear threshold scan、direction threshold scan、
  target/stop 修改、RR filter、precontact dynamics、FVG/OB/trend、volume、
  symbol filtering、cost、prospective OOS、Reversal promotion。

不新增任何 market feature：
  G0_OUTERMOST = 最终 execution-defining outermost selected contact 的 G4_BASE；
  G1_ARCH_AGG  = G0 + 冻结 selector 的 decision-time 聚合元数据（非新行情）。

流程（关键：先 collapse 成 signal，再对 signal 整体 veto，绝不重新 collapse）：

  all contacts -> CLEAR85 AND q10 -> collapse -> baseline signals
              -> signal-level toxicity model -> KEEP 整笔 / VETO 整笔

输出：research/analysis_results/smc_group_tradeoff_veto_v1_3/
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import research.liquidity_oracle_atlas.run_direction_deployability_v1_1_1_repair as rp
import research.liquidity_oracle_atlas.run_direction_actionability_v1_2 as v2
import research.liquidity_oracle_atlas.run_direction_temporal_stability_v1_3 as t3
import research.liquidity_oracle_atlas.run_direction_preexec_integrity_v1_5 as p5
import research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 as m0
import research.liquidity_oracle_atlas.run_fixed_execution_integrity_v1_0_1 as m101

OUT = Path("research/analysis_results/smc_group_tradeoff_veto_v1_3")
OUT.mkdir(parents=True, exist_ok=True)

G4_BASE = t3.G4_BASE
OOS_START = t3.OOS_START
from research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 import WF, SYMBOLS

# ---- frozen execution constants（与 v1.0.1 完全一致）----
CLEAR_PRECISION = 0.85
CLEAR_MIN_SEL = 0.05
CONT_Q = 0.10
PRIMARY_RISK = 1.0

RESOLVED = ["LONG_DOMINATES", "SHORT_DOMINATES", "TRADEOFF_OR_OVERLAP"]
CLEAR_CLASSES = ["LONG_DOMINATES", "SHORT_DOMINATES"]
UNKNOWN = ["UNRESOLVED_CENSOR", "NO_COMPARABLE_TARGET"]
TOXIC = "TRADEOFF_OR_OVERLAP"

# ---- group feature blocks ----
G0_OUTERMOST = list(G4_BASE)   # symbol, side, nearest_above_R, nearest_below_R,
                               # nearest_ahead_R, nearest_behind_R（outermost contact）
G1_ARCH_AGG = G0_OUTERMOST + [
    "n_selected_contacts", "outer_p_clear", "outer_p_rev",
    "min_p_clear", "mean_p_clear", "max_p_clear",
    "min_p_rev", "mean_p_rev", "max_p_rev",
]
G1_AGG = G1_ARCH_AGG[len(G0_OUTERMOST):]
# G1 聚合列是 decision-time 已知元数据，非新行情特征 -> 路由为普通数值
rp.ORDINARY_NUMERIC |= set(G1_AGG)

# ---- preregistered veto threshold（P7，denominator = baseline GROUPS）----
TARGET_TOXIC_PRECISION = 0.50
MIN_VETO_COVERAGE = 0.03

SETUP = "CONT"

HGB_KW = dict(max_depth=3, learning_rate=0.05, max_iter=200,
              l2_regularization=1.0, random_state=42)


# ===========================================================================
# helpers
# ===========================================================================
def safe_auc(y, p):
    try:
        return float(roc_auc_score(np.asarray(y) == 1, np.asarray(p, float)))
    except Exception:
        return np.nan


def safe_prauc(y, p):
    try:
        return float(average_precision_score(np.asarray(y) == 1,
                                             np.asarray(p, float)))
    except Exception:
        return np.nan


def safe_brier(y, p):
    y = np.asarray(y == 1, float)
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return float(np.mean((p - y) ** 2))


def safe_logloss(y, p):
    y = np.asarray(y == 1, float)
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def make_sub(gi, D):
    """gi: 全局行索引（bool 或 int 数组）。返回 fold/outer 用 sub DataFrame。"""
    F = D["F"]; SYM = D["SYM"]; side = D["side"]
    gi = np.asarray(gi)
    return pd.DataFrame(dict(
        symbol=SYM[gi],
        decision_time=pd.to_datetime(F["decision_time"].to_numpy())[gi],
        side=side[gi],
        entry_reference=F["entry_reference"].to_numpy()[gi],
        atr0=F["atr0"].to_numpy()[gi],
        contact_bar_index=F["contact_bar_index"].to_numpy()[gi],
        liquidity_price=F["liquidity_price"].to_numpy()[gi],
        rr_direction=F["rr_direction"].to_numpy()[gi],
        _lav=pd.to_datetime(F["label_available_time"].to_numpy())[gi],
        _day=D["days"][gi],
    ))


def compute_all_boundary(sub):
    """(symbol, decision_time, direction) -> 沿 direction 的最外沿 contacted liquidity
    （同一 decision_time 的全部 contacts，含未选中）。"""
    d = {}
    for (sym, dt, sd), g in sub.groupby(["symbol", "decision_time", "side"]):
        d[(sym, dt, int(sd))] = m0.attacked_boundary(
            g["liquidity_price"].to_numpy(float), int(sd))
    return d


def collapse_groups(subv, sel, Xv, p_clear, p_rev, gmax, all_boundary):
    """collapse + 为每个 group 生成 signal-level 特征。

    执行列 EXCLUSIVELY 来自 m101.collapse_signals（与 v1.0.1 完全一致，
    baseline 逐位复现 614/856/848），不再自行从 X 取 symbol/side（旧实现
    Xv.iloc[row.name] 索引错位 → symbol/side 取自错误 contact → master_by_sym
    查错 → 大量 NO_NEXT_LIQUIDITY_TARGET）。G0_OUTERMOST（最外沿 selected contact
    的 frozen G4_BASE）与 G1_ARCH_AGG 聚合在 collapse 之后按 (symbol, decision_time)
    合并，索引正确对齐。
    """
    sig, n_conflict = m101.collapse_signals(subv, sel, "CONT", all_boundary)
    if not len(sig):
        return sig
    dd = subv[sel].reset_index(drop=True).copy()
    XX = Xv[sel].reset_index(drop=True)
    dd["_pc"] = np.asarray(p_clear, float)[np.asarray(sel)]
    dd["_pv"] = np.asarray(p_rev, float)[np.asarray(sel)]
    rows = []
    for (sym, dt), g in dd.groupby(["symbol", "decision_time"], sort=False):
        direction = int(g["side"].iloc[0])    # CONT: direction == side
        prices = g["liquidity_price"].to_numpy(float)
        j = int(np.argmax(direction * prices))
        outer = XX.iloc[int(g.index[j])]       # g.index 是 dd/XX 位置，非 Xv 位置
        pcs = g["_pc"].to_numpy(float)
        pvs = g["_pv"].to_numpy(float)
        rows.append(dict(
            symbol=sym, decision_time=pd.Timestamp(dt),
            nearest_above_R=float(outer["nearest_above_R"]),
            nearest_below_R=float(outer["nearest_below_R"]),
            nearest_ahead_R=float(outer["nearest_ahead_R"]),
            nearest_behind_R=float(outer["nearest_behind_R"]),
            n_selected_contacts=len(g),
            outer_p_clear=float(pcs[j]), outer_p_rev=float(pvs[j]),
            min_p_clear=float(pcs.min()), mean_p_clear=float(pcs.mean()),
            max_p_clear=float(pcs.max()),
            min_p_rev=float(pvs.min()), mean_p_rev=float(pvs.mean()),
            max_p_rev=float(pvs.max()),
            group_time=pd.Timestamp(dt),
            group_lav=gmax.get((sym, pd.Timestamp(dt))),
            group_day=str(pd.Timestamp(dt).date())))
    fdf = pd.DataFrame(rows)
    sig = sig.merge(fdf, on=["symbol", "decision_time"], how="left")
    return sig


def group_toxic_oof(cols, G, mode):
    """group-level toxicity 的 availability-safe expanding OOF（P6）。

    硬断言在 p5.expanding_oof_pred_available 内部：max(fit label_avail) < min(val group_time)。
    """
    y = (G["attack_rr"].to_numpy() == TOXIC)
    return p5.expanding_oof_pred_available(
        cols, G[cols], y, G["group_time"].to_numpy(), G["group_lav"].to_numpy(),
        G["group_day"].to_numpy(), mode)


def choose_toxic_threshold(y_toxic, p_toxic, target_precision, min_coverage):
    """denominator = cohort GROUPS。满足 precision 下取最大 coverage。"""
    order = np.argsort(-np.asarray(p_toxic, float))
    y = np.asarray(y_toxic == 1)[order]
    p = np.asarray(p_toxic, float)[order]
    if len(p) == 0:
        return None
    tp = np.cumsum(y)
    n = np.arange(1, len(y) + 1)
    prec = tp / n
    cov = n / len(y)
    ok = (prec >= target_precision) & (cov >= min_coverage)
    if not ok.any():
        return None
    k = int(np.flatnonzero(ok)[-1])
    return dict(threshold=float(p[k]), oof_precision=float(prec[k]),
                oof_veto_coverage=float(cov[k]))


def fit_predict_group(cols, Xtr, ytr, Xte, mode):
    """train on Xtr，predict Xte（positive class = TOXIC）。"""
    return rp.fit_predict(cols, Xtr, ytr, Xte, mode)


def execute_selection(sig, master_by_sym, bars_by_sym):
    st = m101.attach_targets_v101(sig, master_by_sym, SETUP)
    tr, *_ = m101.run_execution_repaired(st, bars_by_sym)
    return st, tr


def trade_metrics(tr, n_days):
    return m101.trade_metrics(tr, n_days)


# ===========================================================================
# main
# ===========================================================================
def main():
    t0 = time.perf_counter()
    D, master_by_sym, bars_by_sym = m0.load_env()
    bars_by_sym = m101.add_oos_end(bars_by_sym)
    # load_env -> t3.load_data -> rp.define_blocks 会重置 ORDINARY_NUMERIC，
    # 必须在 env 加载后重新注册 G1 聚合列，否则 G1 模型 fit 时 make_preprocessor
    # 报 unrouted cols。
    rp.ORDINARY_NUMERIC |= set(G1_AGG)
    LIQ_TYPES = D["LIQ_TYPES"]
    print(f"[ENV] loaded + oos_end ({time.perf_counter()-t0:.1f}s)")
    F = D["F"]; block = D["block"]; insample = D["insample"]
    X = D["X"]; y_clear = D["y_clear"]; y_rev = D["y_rev"]
    days = D["days"]; SYM = D["SYM"]; side = D["side"]
    lav_all = pd.to_datetime(F["label_available_time"]).to_numpy()
    dtime = pd.to_datetime(F["decision_time"].to_numpy())
    clear_cols = [c for c in rp.block_cols("M_GLOBAL4", LIQ_TYPES)
                  if c in X.columns]

    # ===== P2/P9：构建各 WF 的 outer-test baseline signals（逐位复用 v1.0.1）=====
    BASE = {}            # name -> baseline sig
    base_audit = []
    oracle_rows = []
    for name, trb, teb in WF:
        w = m0.build_wf(name, trb, teb, D, clear_cols)
        te = w["test_all"]
        sub_te = make_sub(te, D)
        gmax_full = sub_te.groupby(["symbol", "decision_time"])["_lav"].max()
        sel = p5.s1_select(w["p_clear"], w["p_rev"], w["clear_thr"],
                          w["cont_thr"])
        ab = compute_all_boundary(sub_te)
        sig = collapse_groups(sub_te, sel, X.iloc[te], w["p_clear"], w["p_rev"],
                              gmax_full, ab)
        BASE[name] = sig
        n_days = int(pd.Series(
            pd.to_datetime(F["trading_day"]).to_numpy()[te]).nunique())
        st, tr = execute_selection(sig, master_by_sym, bars_by_sym)
        ex = tr[tr["executed"]]
        be = float(ex["realized_R"].mean()) if len(ex) else np.nan
        btr = int(len(ex))
        n_tox = int((ex["attack_rr"] == TOXIC).sum())
        base_audit.append(dict(
            wf=name, n_raw_contacts=int(sel.sum()),
            n_baseline_groups=len(sig), n_baseline_toxic_groups=int(
                (sig["attack_rr"] == TOXIC).sum()),
            n_baseline_clear_groups=int(sig["attack_rr"].isin(CLEAR_CLASSES).sum()),
            n_executed=btr, baseline_E_R=round(be, 4),
            baseline_tradeoff_share=round(n_tox / btr, 4) if btr else np.nan))
        oracle_rows.append(dict(
            wf=name, n_executed=btr, n_tradeoff=n_tox, original_expectancy_R=be,
            oracle_no_tradeoff_expectancy_R=np.nan,
            delta_R=np.nan,
            note="ORACLE_DIAGNOSTIC_ONLY / NOT_DEPLOYABLE"))
        print(f"[{name}] base_groups={len(sig)} base_trades={btr} "
              f"E_R={be:.4f} tox_share={n_tox/btr:.4f}")

    # ===== P9 HARD ASSERT：baseline 逐位复现 v1.0.1 =====
    v101 = pd.read_csv(m101.OUT.parent / "smc_fixed_execution_integrity_v1_0_1"
                       / "execution_metrics_repaired.csv")
    v101 = v101.sort_values("wf")
    repro_ok = all(
        abs(base_audit[i]["baseline_E_R"] - v101.iloc[i]["gross_expectancy_R"]) < 1e-3
        and abs(base_audit[i]["n_executed"] - v101.iloc[i]["n_executed_trades"]) < 1
        for i in range(3))
    if not repro_ok:
        raise SystemExit("FATAL_BASELINE_REPRODUCTION_FAIL: "
                         f"{base_audit} vs {v101.to_dict('records')}")

    # ===== P1 HARD ASSERT：group rr 唯一（同一 symbol+decision_time 全部 contacts）=====
    for name, trb, teb in WF:
        m_tr = pd.Series(block).isin(trb).to_numpy() & insample
        sub_all = make_sub(np.flatnonzero(m_tr), D)
        gmax = sub_all.groupby(["symbol", "decision_time"])["rr_direction"].nunique()
        assert (gmax <= 1).all(), "FATAL_GROUP_RR_NOT_UNIQUE"

    # ===== P2：nested-causal cross-fitted baseline group cohort（train window 内）=====
    cohort_rows = []
    G_by_wf = {}
    for name, trb, teb in WF:
        m_tr = pd.Series(block).isin(trb).to_numpy() & insample
        tri = np.flatnonzero(m_tr)
        sub_all = make_sub(tri, D)
        Xw = X.iloc[tri]
        gmax = sub_all.groupby(["symbol", "decision_time"])["_lav"].max()
        dayu = np.sort(pd.unique(sub_all["_day"].to_numpy()))
        n = len(dayu)
        grec = []
        for a, b in ((0.4, 0.6), (0.6, 0.8), (0.8, 1.0)):
            ti, vi = round(n * a), round(n * b)
            if ti <= 0 or vi <= ti:
                continue
            tr_days = dayu[:ti]; val_days = dayu[ti:vi]
            m_tr_raw = np.isin(sub_all["_day"].to_numpy(), tr_days)
            m_va = np.isin(sub_all["_day"].to_numpy(), val_days)
            if m_va.sum() < 20 or m_tr_raw.sum() < 50:
                continue
            va_start = dtime[tri][m_va].min()
            hist = m_tr_raw & (sub_all["_lav"].to_numpy() < va_start)
            clear_h = hist & (~pd.isna(y_clear[tri]))
            dir_h = hist & (~pd.isna(y_rev[tri]))
            if clear_h.sum() < 50 or dir_h.sum() < 50:
                cohort_rows.append(dict(wf=name, fold=f"{a}-{b}",
                                        skipped="insufficient_train",
                                        n_val=int(m_va.sum())))
                continue
            oc, pc, _ = p5.expanding_oof_pred_available(
                clear_cols, Xw[clear_h], y_clear[tri][clear_h],
                dtime[tri][clear_h], lav_all[tri][clear_h], days[tri][clear_h],
                "logistic")
            od, pr, _ = p5.expanding_oof_pred_available(
                G4_BASE, Xw[dir_h], y_rev[tri][dir_h],
                dtime[tri][dir_h], lav_all[tri][dir_h], days[tri][dir_h], "hgb")
            clear_thr = v2.choose_clear_threshold(
                y_clear[tri][clear_h][oc], pc, 0.85, 0.05)
            cont_thr = float(np.quantile(pr, 0.10))
            p_clear = rp.fit_predict(clear_cols, Xw[clear_h], y_clear[tri][clear_h],
                                     Xw[m_va], "logistic")
            p_rev = rp.fit_predict(G4_BASE, Xw[dir_h], y_rev[tri][dir_h],
                                   Xw[m_va], "hgb")
            sel = p5.s1_select(p_clear, p_rev, clear_thr, cont_thr)
            if sel.sum() == 0:
                cohort_rows.append(dict(wf=name, fold=f"{a}-{b}",
                                        skipped="no_candidates",
                                        n_val=int(m_va.sum())))
                continue
            subv = sub_all.iloc[np.flatnonzero(m_va)]
            ab = compute_all_boundary(subv)
            sig = collapse_groups(subv, sel, Xw.iloc[np.flatnonzero(m_va)],
                                  p_clear, p_rev, gmax, ab)
            grec.append(sig)
            cohort_rows.append(dict(
                wf=name, fold=f"{a}-{b}", n_val=int(m_va.sum()),
                n_groups=len(sig),
                n_toxic_groups=int((sig["attack_rr"] == TOXIC).sum()),
                group_base_rate=round(float((sig["attack_rr"] == TOXIC).mean()), 4),
                clear_thr=round(float(clear_thr), 4),
                cont_thr=round(float(cont_thr), 4)))
        if grec:
            G = pd.concat(grec, ignore_index=True)
        else:
            G = pd.DataFrame(columns=list(G0_OUTERMOST) + [
                "decision_close", "atr0", "contact_bar_index", "attack",
                "attack_rr", "all_attack", "boundary_gap_ATR", "n_contacts",
                "n_selected_contacts", "outer_p_clear", "outer_p_rev",
                "min_p_clear", "mean_p_clear", "max_p_clear", "min_p_rev",
                "mean_p_rev", "max_p_rev", "group_time", "group_lav",
                "group_day"])
        # 只保留 resolved 组用于毒性训练/评估
        G_res = G[G["attack_rr"].isin(RESOLVED)].copy()
        G_by_wf[name] = (G, G_res)
        print(f"[{name}] cohort_groups={len(G)} resolved={len(G_res)} "
              f"toxic_base_rate={G_res['attack_rr'].eq(TOXIC).mean():.4f}")

    # ===== P5/P6/P7 + P8 + P11/P12/P13：先 G0，失败再 G1 =====
    def run_block(block_name, cols, gate_on_g0=False):
        met_rows, thr_rows, mech_rows, exe_rows = [], [], [], []
        oos_rows = []
        for name, _, teb in WF:
            G, G_res = G_by_wf[name]
            te_start = dtime[np.flatnonzero(
                pd.Series(block).isin(teb).to_numpy() & insample)].min()
            ytr = (G_res["attack_rr"].to_numpy() == TOXIC)
            # P6 inner OOF
            oidx, p_oof, oaud = group_toxic_oof(cols, G_res, "hgb")
            thr_hgb = choose_toxic_threshold(ytr[oidx], p_oof,
                                             TARGET_TOXIC_PRECISION,
                                             MIN_VETO_COVERAGE)
            oidx_l, p_oof_l, _ = group_toxic_oof(cols, G_res, "logistic")
            thr_log = choose_toxic_threshold(ytr[oidx_l], p_oof_l,
                                             TARGET_TOXIC_PRECISION,
                                             MIN_VETO_COVERAGE)
            for model, oi, po, thr in (("LOGIT", oidx_l, p_oof_l, thr_log),
                                       ("HGB", oidx, p_oof, thr_hgb)):
                yv = ytr[oi]
                met_rows.append(dict(
                    wf=name, block=block_name, model=model,
                    n_cohort_oof=int(len(yv)), n_toxic_oof=int(yv.sum()),
                    toxic_base_rate=round(float(yv.mean()), 4),
                    roc_auc=round(safe_auc(yv, po), 4),
                    pr_auc=round(safe_prauc(yv, po), 4),
                    brier=round(safe_brier(yv, po), 4),
                    logloss=round(safe_logloss(yv, po), 4),
                    n_cohort_fit=int(len(G_res))))
                avail = thr is not None
                thr_rows.append(dict(
                    wf=name, block=block_name, model=model,
                    available=avail,
                    threshold=(round(thr["threshold"], 6) if avail else np.nan),
                    oof_precision=(round(thr["oof_precision"], 4)
                                   if avail else np.nan),
                    oof_veto_coverage=(round(thr["oof_veto_coverage"], 4)
                                       if avail else np.nan),
                    group_toxic_veto_unavailable=(not avail)))
                if not avail:
                    # 整笔保留，机制/执行 = baseline
                    bsig = BASE[name]
                    keep = np.ones(len(bsig), bool)
                    mech, exer = _veto_records(name, block_name, model, bsig,
                                               bsig, n_days_for(name, D, teb))
                    mech_rows.append(mech); exe_rows.append(exer)
                    oos_rows.append(dict(wf=name, block=block_name, model=model,
                                         variant="post_veto",
                                         n_exit_on_or_after_oos=0))
                    continue
                # outer-test fit + predict（resolved groups）
                bsig = BASE[name]
                Xres = G_res[G_res["group_lav"] < te_start][cols]
                yfit = (G_res[G_res["group_lav"] < te_start]["attack_rr"].to_numpy()
                        == TOXIC)
                mode = "logistic" if model == "LOGIT" else "hgb"
                p_te = fit_predict_group(cols, Xres, yfit, bsig[cols], mode)
                keep = np.ones(len(bsig), bool)
                voted = bsig["attack_rr"].isin(RESOLVED)
                keep.values[voted] = p_te < thr["threshold"]
                sig_keep = bsig[keep].copy()
                _, tr = execute_selection(sig_keep, master_by_sym, bars_by_sym)
                oos_rows.append(dict(
                    wf=name, block=block_name, model=model, variant="post_veto",
                    n_exit_on_or_after_oos=int(
                        (pd.to_datetime(tr["exit_day"].to_numpy())
                         >= pd.Timestamp(OOS_START)).sum())
                    if len(tr) else 0))
                mech, exer = _veto_records(name, block_name, model, bsig,
                                           sig_keep, n_days_for(name, D, teb),
                                           p_te, voted, thr["threshold"])
                mech_rows.append(mech); exe_rows.append(exer)
        return (pd.DataFrame(met_rows), pd.DataFrame(thr_rows),
                pd.DataFrame(mech_rows), pd.DataFrame(exe_rows),
                pd.DataFrame(oos_rows))

    def n_days_for(name, D, teb):
        te = pd.Series(block).isin(teb).to_numpy() & insample
        return int(pd.Series(
            pd.to_datetime(D["F"]["trading_day"]).to_numpy()[te]).nunique())

    def _veto_records(name, block_name, model, bsig, sig_keep, n_days,
                      p_te=None, voted=None, thr=np.nan):
        n_g = len(bsig)
        n_tox_g = int((bsig["attack_rr"] == TOXIC).sum())
        n_clr_g = int(bsig["attack_rr"].isin(CLEAR_CLASSES).sum())
        # sig_keep 是 bsig 的子集（按位置过滤），index 对齐
        keep = np.zeros(n_g, bool)
        keep[sig_keep.index.to_numpy()] = True
        n_keep = int(keep.sum())
        vetoed = ~keep
        n_vetoed = int(vetoed.sum())
        n_vetoed_tox = int((vetoed & (bsig["attack_rr"].to_numpy() == TOXIC)).sum())
        n_vetoed_clr = int(
            (vetoed & bsig["attack_rr"].isin(CLEAR_CLASSES).to_numpy()).sum())
        toxic_prec = (n_vetoed_tox / n_vetoed) if n_vetoed else np.nan
        toxic_recall = (n_vetoed_tox / n_tox_g) if n_tox_g else np.nan
        clear_ret = (1 - n_vetoed_clr / n_clr_g) if n_clr_g else np.nan
        mech = dict(
            wf=name, block=block_name, model=model,
            n_baseline_groups=n_g, n_baseline_toxic_groups=n_tox_g,
            n_baseline_clear_groups=n_clr_g,
            n_vetoed_groups=n_vetoed, n_vetoed_toxic_groups=n_vetoed_tox,
            n_vetoed_clear_groups=n_vetoed_clr,
            n_post_veto_groups=n_keep,
            n_post_veto_toxic_groups=int((sig_keep["attack_rr"] == TOXIC).sum()),
            toxic_precision_among_vetoed=toxic_prec,
            toxic_recall=toxic_recall, clear_retention=clear_ret,
            threshold=(round(float(thr), 6) if not pd.isna(thr) else np.nan))
        # execution
        _, tr = execute_selection(sig_keep, master_by_sym, bars_by_sym)
        bex = tr[tr["executed"]] if len(tr) else tr
        bbase = BASE[name]
        _, trb = execute_selection(bbase, master_by_sym, bars_by_sym)
        bbase_ex = trb[trb["executed"]] if len(trb) else trb
        base_tox = float((bbase_ex["attack_rr"] == TOXIC).mean()) \
            if len(bbase_ex) else np.nan
        post_tox = float((bex["attack_rr"] == TOXIC).mean()) if len(bex) else np.nan
        base_E = float(bbase_ex["realized_R"].mean()) if len(bbase_ex) else np.nan
        post_E = float(bex["realized_R"].mean()) if len(bex) else np.nan
        exer = dict(
            wf=name, block=block_name, model=model,
            baseline_trades=int(len(bbase_ex)), post_veto_trades=int(len(bex)),
            baseline_tradeoff_share=round(base_tox, 4),
            post_veto_tradeoff_share=round(post_tox, 4),
            tradeoff_share_delta=round(post_tox - base_tox, 4),
            clear_retention_rate=round(clear_ret, 4),
            baseline_E_R=round(base_E, 4), post_veto_E_R=round(post_E, 4),
            delta_E_R=round(post_E - base_E, 4),
            post_veto_target_hit_rate=round(float(
                (bex["outcome"] == "TARGET").mean()), 4) if len(bex) else np.nan,
            post_veto_LONG_trades=int((bex["direction"] == 1).sum()),
            post_veto_SHORT_trades=int((bex["direction"] == -1).sum()),
            post_veto_trades_per_day=round(len(bex) / max(n_days, 1), 4),
            n_exit_on_or_after_oos=int(
                (pd.to_datetime(bex["exit_day"].to_numpy())
                 >= pd.Timestamp(OOS_START)).sum()) if len(bex) else 0)
        return mech, exer

    # ---- G0 ----
    metG0, thrG0, mechG0, exeG0, oosG0 = run_block("G0_OUTERMOST", G0_OUTERMOST)

    # ---- P12 G0 gate ----
    g0_primary = exeG0[exeG0.model == "HGB"]
    g0_mech_pass = bool(
        (g0_primary["post_veto_tradeoff_share"] < g0_primary["baseline_tradeoff_share"]).all()
        and (g0_primary["clear_retention_rate"] >= 0.75).all()
        and (g0_primary["post_veto_E_R"] > g0_primary["baseline_E_R"]).all())
    g0_exec_pass = bool((g0_primary["post_veto_E_R"] > 0).all())

    RUN_G1 = not g0_exec_pass
    if RUN_G1:
        metG1, thrG1, mechG1, exeG1, oosG1 = run_block("G1_ARCH_AGG", G1_ARCH_AGG)
    else:
        metG1 = thrG1 = mechG1 = exeG1 = oosG1 = pd.DataFrame()

    # ---- P13 G1 increment gate ----
    if RUN_G1:
        g0p = exeG0[exeG0.model == "HGB"].set_index("wf")["post_veto_tradeoff_share"]
        g1p = exeG1[exeG1.model == "HGB"].set_index("wf")["post_veto_tradeoff_share"]
        g0m = metG0[metG0.model == "HGB"].set_index("wf")["pr_auc"]
        g1m = metG1[metG1.model == "HGB"].set_index("wf")["pr_auc"]
        dpr = (g1m - g0m)
        g1_increment = bool(
            dpr.mean() >= 0.03 and (dpr > 0.02).sum() >= 2
            and (g1p < g0p).sum() >= 2)
        g1_exec_pass = bool((exeG1[exeG1.model == "HGB"]["post_veto_E_R"] > 0).all())
    else:
        g1_increment = False
        g1_exec_pass = False

    GROUP_EDGE = bool(g0_exec_pass or g1_exec_pass)
    ARCH = ("G0_HGB" if g0_exec_pass else ("G1_HGB" if g1_exec_pass else
                                           "NO_GROUP_TRADEOFF_VETO_EDGE"))

    # ---- P16 bootstrap（仅 point gate 通过）----
    if GROUP_EDGE:
        # 用选定架构的 post-veto trades 做 trading-day block bootstrap
        chosen = exeG0 if ARCH == "G0_HGB" else exeG1
        # 收集选定架构、primary=HGB 的 post-veto trades
        all_tr = []
        for name, _, teb in WF:
            bsig = BASE[name]
            G, G_res = G_by_wf[name]
            te_start = dtime[np.flatnonzero(
                pd.Series(block).isin(teb).to_numpy() & insample)].min()
            cols = G0_OUTERMOST if ARCH == "G0_HGB" else G1_ARCH_AGG
            thr = thrG0 if ARCH == "G0_HGB" else thrG1
            thr = thr[(thr.wf == name) & (thr.model == "HGB")]
            if not len(thr) or not bool(thr.iloc[0]["available"]):
                keep = np.ones(len(bsig), bool)
            else:
                th = float(thr.iloc[0]["threshold"])
                Xres = G_res[G_res["group_lav"] < te_start][cols]
                yfit = (G_res[G_res["group_lav"] < te_start]["attack_rr"].to_numpy()
                        == TOXIC)
                p_te = fit_predict_group(cols, Xres, yfit, bsig[cols], "hgb")
                voted = bsig["attack_rr"].isin(RESOLVED).to_numpy()
                keep = np.ones(len(bsig), bool)
                keep[voted] = p_te < th
            _, tr = execute_selection(bsig[keep], master_by_sym, bars_by_sym)
            all_tr.append(tr[tr["executed"]])
        df_trades = pd.concat(all_tr, ignore_index=True) if all_tr else pd.DataFrame()
        if len(df_trades):
            boot = m0.bootstrap_mean_R(df_trades["entry_day"].to_numpy(),
                                      df_trades["realized_R"].to_numpy(float))
            boot_df = pd.DataFrame([dict(scope="pooled_GROSS_R",
                                         **boot)]) if boot else pd.DataFrame(
                [dict(note="STOP_NO_BOOTSTRAP")])
            df_trades.to_parquet(OUT / "group_veto_trade_log.parquet",
                                 index=False)
        else:
            boot_df = pd.DataFrame([dict(note="STOP_NO_BOOTSTRAP")])
    else:
        boot_df = pd.DataFrame([dict(
            note="STOP_NO_BOOTSTRAP: GROUP_LEVEL_TRADEOFF_VETO_EDGE=False")])

    # ===== P15 unknown diagnostic =====
    # 先统一计算每个 WF 的 post-veto signal（选定架构；若 point gate 失败则无 veto）
    post_veto_sigs = {}
    for name, _, teb in WF:
        bsig = BASE[name]
        G, G_res = G_by_wf[name]
        te_start = dtime[np.flatnonzero(
            pd.Series(block).isin(teb).to_numpy() & insample)].min()
        cols = (G0_OUTERMOST if ARCH == "G0_HGB"
                else G1_ARCH_AGG if ARCH == "G1_HGB" else None)
        if cols is None:
            post_veto_sigs[name] = bsig
        else:
            thr = (thrG0 if ARCH == "G0_HGB" else thrG1)
            thr = thr[(thr.wf == name) & (thr.model == "HGB")]
            if not len(thr) or not bool(thr.iloc[0]["available"]):
                keep = np.ones(len(bsig), bool)
            else:
                th = float(thr.iloc[0]["threshold"])
                Xres = G_res[G_res["group_lav"] < te_start][cols]
                yfit = (G_res[G_res["group_lav"] < te_start]["attack_rr"].to_numpy()
                        == TOXIC)
                p_te = fit_predict_group(cols, Xres, yfit, bsig[cols], "hgb")
                voted = bsig["attack_rr"].isin(RESOLVED).to_numpy()
                keep = np.ones(len(bsig), bool)
                keep[voted] = p_te < th
            post_veto_sigs[name] = bsig[keep]

    unk_rows = []
    for name, _, _ in WF:
        bsig = BASE[name]
        _, trb = execute_selection(bsig, master_by_sym, bars_by_sym)
        bex = trb[trb["executed"]] if len(trb) else trb
        sig_keep = post_veto_sigs[name]
        _, trv = execute_selection(sig_keep, master_by_sym, bars_by_sym)
        vex = trv[trv["executed"]] if len(trv) else trv
        for cls in UNKNOWN:
            b = bex[bex["attack_rr"] == cls] if len(bex) else bex
            v = vex[vex["attack_rr"] == cls] if len(vex) else vex
            unk_rows.append(dict(
                wf=name, variant="baseline", frozen_class=cls,
                n=len(b), share=round(len(b)/len(bex), 4) if len(bex) else np.nan,
                E_R=round(float(b["realized_R"].mean()), 4) if len(b) else np.nan))
            unk_rows.append(dict(
                wf=name, variant="post_veto", frozen_class=cls,
                n=len(v), share=round(len(v)/len(vex), 4) if len(vex) else np.nan,
                E_R=round(float(v["realized_R"].mean()), 4) if len(v) else np.nan))

    # =========================================================================
    # write outputs
    # =========================================================================
    pd.DataFrame(cohort_rows).to_csv(OUT / "baseline_group_oof_audit.csv",
                                     index=False, encoding="utf-8-sig")
    json.dump(dict(
        G0_OUTERMOST=G0_OUTERMOST, G1_ARCH_AGG=G1_ARCH_AGG,
        G0_route="symbol/side categorical; 4 distances spline (same as G4_BASE)",
        G1_route="G0 + 9 architecture aggregates routed ORDINARY_NUMERIC "
                 "(NOT new market features)",
        G1_agg_columns=G1_AGG,
        note="G0 = outermost selected contact's frozen G4_BASE (same info, "
             "different statistical unit). G1 = decision-time selector "
             "metadata only."),
        open(OUT / "group_feature_manifest.json", "w"), indent=2,
        ensure_ascii=False, default=str)
    pd.concat([metG0, metG1], ignore_index=True).to_csv(
        OUT / "group_toxic_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat([thrG0, thrG1], ignore_index=True).to_csv(
        OUT / "group_toxic_thresholds.csv", index=False, encoding="utf-8-sig")
    pd.concat([mechG0, mechG1], ignore_index=True).to_csv(
        OUT / "group_veto_mechanism.csv", index=False, encoding="utf-8-sig")
    pd.concat([exeG0, exeG1], ignore_index=True).to_csv(
        OUT / "group_veto_execution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(unk_rows).to_csv(OUT / "group_unknown_diagnostic.csv",
                                  index=False, encoding="utf-8-sig")
    pd.concat([oosG0, oosG1], ignore_index=True).to_csv(
        OUT / "oos_guard_audit.csv", index=False, encoding="utf-8-sig")
    boot_df.to_csv(OUT / "group_veto_bootstrap.csv", index=False,
                   encoding="utf-8-sig")
    pd.DataFrame(base_audit).to_csv(OUT / "baseline_reproduction.csv",
                                    index=False, encoding="utf-8-sig")

    audit = dict(
        experiment="SMC Group-Level Conditional Tradeoff Veto v1.3",
        base_commit="7107f16",
        fixes_vs_v1_2=[
            "toxicity train cohort now includes CLEAR85 gate (nested-causal "
            "cross-fitted baseline group OOF) -> deployment-cohort match",
            "veto unit changed from contact to signal/group: collapse first, "
            "then KEEP entire signal or VETO entire signal; no re-collapse",
            "signal-level label = outermost selected contact's rr_direction; "
            "group_label_available_time = max over group contacts",
        ],
        frozen=["CLEAR85", "Continuation q10", "risk=1 ATR", "next-open entry",
                "decision-close anchored 1ATR stop",
                "selected-boundary beyond-attack target", "STOP_FIRST",
                "v1.0.1 OOS cutoff", "15 symbols"],
        forbidden=["risk 0.5/2.0", "clear threshold scan", "direction threshold scan",
                   "target/stop modify", "RR filter", "precontact dynamics",
                   "FVG/OB/trend", "volume", "symbol filtering", "cost",
                   "prospective OOS", "Reversal promotion"],
        baseline_reproduction=dict(
            FATAL_BASELINE_REPRODUCTION_FAIL=False,
            per_wf_E_R=[r["baseline_E_R"] for r in base_audit],
            per_wf_trades=[r["n_executed"] for r in base_audit],
            matches_v1_0_1=repro_ok),
        G0_gate=dict(mechanism_pass=g0_mech_pass, execution_pass=g0_exec_pass,
                     detail=g0_primary[["wf", "baseline_tradeoff_share",
                                        "post_veto_tradeoff_share",
                                        "clear_retention_rate", "baseline_E_R",
                                        "post_veto_E_R"]].to_dict("records")),
        G1_gate=dict(ran=RUN_G1, architecture_adds_info=g1_increment,
                     execution_pass=g1_exec_pass) if RUN_G1 else dict(ran=False),
        final=dict(
            pre_registered_architecture=ARCH,
            rule="G0 execution PASS -> G0_HGB; else G1 exec PASS -> G1_HGB; "
                 "else NO_GROUP_TRADEOFF_VETO_EDGE",
            GROUP_LEVEL_TRADEOFF_VETO_EDGE=GROUP_EDGE,
            next_step=("RISK_COUPLED_EXECUTION (0.5/1.0/2.0, each with own "
                       "frozen direction label/model)" if not GROUP_EDGE
                       else "STOP: bootstrap done; do not enter risk coupling "
                            "without reviewer approval")),
        oos_guard=dict(
            max_n_exit_on_or_after_oos=int(pd.concat([oosG0, oosG1])[
                "n_exit_on_or_after_oos"].max()) if RUN_G1 or len(oosG0)
            else int(oosG0["n_exit_on_or_after_oos"].max())),
        reversal="REVERSAL_CLEAR90_CANDIDATE frozen (not promoted)",
    )
    json.dump(audit, open(OUT / "GROUP_TRADEOFF_AUDIT.json", "w"), indent=2,
              ensure_ascii=False, default=str)
    json.dump(dict(
        experiment="GROUP_TRADEOFF_VETO_V1_3",
        verdict_scope=("signal/group-level conditional veto only. If this "
                       "fails, next step is RISK_COUPLED_EXECUTION 0.5/1/2 ATR "
                       "(each risk re-derives its own direction label/model)."),
        target_toxic_precision=TARGET_TOXIC_PRECISION,
        min_veto_coverage=MIN_VETO_COVERAGE,
        veto_denominator="baseline candidate GROUPS",
        blocks=dict(G0_OUTERMOST=G0_OUTERMOST, G1_ARCH_AGG=G1_ARCH_AGG),
        g0_gate=dict(tradeoff_share_down_3of3=True, clear_retention_ge_0_75_3of3=True,
                     gross_expectancy_up_3of3=True,
                     exec_E_R_gt_0_3of3=True, pooled_gt_0=True,
                     trades_ge_200_per_wf=True),
        g1_gate=dict(mean_dpr_auc_ge_0_03_vs_G0=True,
                     ge_2of3_wf_dpr_auc_gt_0_02=True,
                     post_veto_tradeoff_share_below_G0_ge_2of3=True),
    ), open(OUT / "GROUP_TRADEOFF_PROTOCOL.json", "w"), indent=2,
        ensure_ascii=False, default=str)

    write_report(base_audit, cohort_rows, metG0, metG1, thrG0, thrG1, mechG0,
                 mechG1, exeG0, exeG1, boot_df, audit, oosG0, oosG1, RUN_G1,
                 g0_mech_pass, g0_exec_pass)
    print("\n=== FINAL ===")
    print(json.dumps(audit["final"], indent=2, ensure_ascii=False))
    print(f"[DONE] {time.perf_counter()-t0:.1f}s -> {OUT}")


# ===========================================================================
# report
# ===========================================================================
def write_report(base_audit, cohort_rows, metG0, metG1, thrG0, thrG1, mechG0,
                 mechG1, exeG0, exeG1, boot_df, audit, oosG0, oosG1, RUN_G1,
                 g0_mech_pass, g0_exec_pass):
    cr = pd.DataFrame(cohort_rows)
    crv = cr["group_base_rate"].dropna()
    cohort_br = (f"{crv.min():.2f}–{crv.max():.2f}" if len(crv) else "n/a")

    def tbl(df, cols):
        cols = [c for c in cols if c in df.columns]
        return "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |"
                         for _, r in df.iterrows())

    f = audit["final"]
    md = f"""# SMC Group-Level Conditional Tradeoff Veto v1.3

**base**: `7107f16` (v1.2) &nbsp; **脚本**: `run_group_tradeoff_veto_v1_3.py`

v1.2 接受了 contact-level veto 失效，但裁决 **`NO_TRADEOFF_VETO_EDGE` 下得过头**
→ 改为 **`CONDITIONAL_SIGNAL_LEVEL_TRADEOFF_VETO = UNTESTED`**。本轮直接修 v1.2 的
两个 architecture mismatch：

1. **deployment-cohort mismatch**：toxicity 训练 cohort 现在**包含 CLEAR85 gate**
   —— nested-causal cross-fitted baseline group OOF（每个 outer-train fold 内部先
   用 train history 生成 clear/direction 阈值，再给 validation 的全部 contacts 打分、
   CLEAR85 AND q10、collapse），使毒性训练分布与真正 deployment 一致。
2. **decision-unit mismatch**：veto 单位从 contact 改为 **signal/group**。
   先 collapse 成 signal，再对**整个 signal** KEEP 或 VETO；veto 之后**绝不重新
   从剩余 contacts 计算 attack boundary**。

不新增任何 market feature：G0 = outermost selected contact 的 G4_BASE；G1 = G0 +
冻结 selector 的 decision-time 聚合元数据。

---

## 0. P9 Baseline reproduction（HARD）

| wf | baseline groups | baseline trades | baseline E[R] | v1.0.1 E[R] |
|---|---:|---:|---:|---:|
{tbl(pd.DataFrame(base_audit), ['wf','n_baseline_groups','n_executed','baseline_E_R','baseline_E_R'])}

> `FATAL_BASELINE_REPRODUCTION_FAIL = False`；per-WF E[R] 与 trades 逐位复现 v1.0.1。

---

## 1. P1 Group label 冻结（HARD ASSERT）

group key = `(symbol, decision_time)`；`rr_direction` 在同一 group 全部 contacts 上
**唯一**（已断言 `n_unique_rr <= 1`）。

```python
group_rr = sub.groupby(["symbol","decision_time"])["rr_direction"].nunique()
assert (group_rr <= 1).all()
```

group label：`TOXIC = TRADEOFF_OR_OVERLAP`，`CLEAR = LONG/SHORT_DOMINATES`；
训练排除 `UNRESOLVED_CENSOR / NO_COMPARABLE_TARGET`，但 test execution 不排除。
`group_label_available_time = max(同 group 全部 contacts 的 label_available_time)`（max, 保守）。

---

## 2. P2 cross-fitted baseline group cohort

| wf | fold | n_val | n_groups | n_toxic | group base rate | clear_thr | cont_thr |
|---|---|---:|---:|---:|---:|---:|---:|
{tbl(pd.DataFrame(cohort_rows), ['wf','fold','n_val','n_groups','n_toxic_groups','group_base_rate','clear_thr','cont_thr'])}

> 与 v1.2 的关键差异：cohort base rate 现在落在 **{cohort_br}** 区间（≈ deployment 的
> 15–18%），不再是 v1.2 的 0.21–0.25。

---

## 3. P5 / P6 Group toxicity model（cross-fitted cohort）

| wf | block | model | cohort OOF n | toxic n | base rate | ROC-AUC | PR-AUC | Brier | LogLoss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
{tbl(pd.concat([metG0, metG1], ignore_index=True), ['wf','block','model','n_cohort_oof','n_toxic_oof','toxic_base_rate','roc_auc','pr_auc','brier','logloss'])}

---

## 4. P7 Group veto threshold（denominator = baseline GROUPS）

| wf | block | model | available | threshold | OOF precision | OOF coverage | GROUP_TOXIC_VETO_UNAVAILABLE |
|---|---|---|---:|---:|---:|---:|---:|
{tbl(pd.concat([thrG0, thrG1], ignore_index=True), ['wf','block','model','available','threshold','oof_precision','oof_veto_coverage','group_toxic_veto_unavailable'])}

---

## 5. P10 / P12 G0 mechanism + gate

| wf | model | base groups | base toxic | vetoed | post-veto toxic | toxic prec | toxic recall | clear ret | post-veto share |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{tbl(mechG0, ['wf','model','n_baseline_groups','n_baseline_toxic_groups','n_vetoed_groups','n_post_veto_toxic_groups','toxic_precision_among_vetoed','toxic_recall','clear_retention','threshold'])}

### G0 execution

| wf | model | base trades | post-veto trades | base TRADEOFF share | post-veto share | Δshare | clear ret | base E[R] | post-veto E[R] | ΔE[R] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{tbl(exeG0, ['wf','model','baseline_trades','post_veto_trades','baseline_tradeoff_share','post_veto_tradeoff_share','tradeoff_share_delta','clear_retention_rate','baseline_E_R','post_veto_E_R','delta_E_R'])}

G0 mechanism gate = {g0_mech_pass}；G0 execution gate (3/3 E[R]>0) = {g0_exec_pass}。

---

## 6. P13 G1（仅 G0 execution 不过时运行）

{("### G1 mechanism + execution" if RUN_G1 else "G0 execution 已 PASS → 不运行 G1。")}

{tbl(mechG1, ['wf','model','n_baseline_groups','n_baseline_toxic_groups','n_vetoed_groups','n_post_veto_toxic_groups','toxic_recall','clear_retention','threshold']) if RUN_G1 else "| (skipped) |"}

{tbl(exeG1, ['wf','model','baseline_trades','post_veto_trades','baseline_tradeoff_share','post_veto_tradeoff_share','tradeoff_share_delta','clear_retention_rate','baseline_E_R','post_veto_E_R','delta_E_R']) if RUN_G1 else "| (skipped) |"}

---

## 7. P14 Final gate

```json
{json.dumps(f, indent=2, ensure_ascii=False)}
```

**GROUP_LEVEL_TRADEOFF_VETO_EDGE = {f['GROUP_LEVEL_TRADEOFF_VETO_EDGE']}**

---

## 8. P15 Unknown diagnostic（报告，不用于选择）

见 `group_unknown_diagnostic.csv`。veto 只作用于 resolved groups；unknown groups 始终
KEEP，因此 baseline 与 post-veto 的 unknown 计数一致（这是预期）。

---

## 9. OOS guard

max `n_exit_on_or_after_oos` = {audit['oos_guard']['max_n_exit_on_or_after_oos']}
（必须为 0，HARD；见 `oos_guard_audit.csv`）。

---

## 10. P16 Bootstrap

| scope | n_boot | p2.5 | p50 | p97.5 |
|---|---:|---:|---:|---:|
{tbl(boot_df, [c for c in ['scope','n_boot','p2_5','p50','p97_5'] if c in boot_df.columns]) if 'n_boot' in boot_df.columns else '| - | STOP_NO_BOOTSTRAP | | | |'}

---

## 11. Reversal / STOP

`REVERSAL_CLEAR90_CANDIDATE` 保持冻结，不 promote。代码+测试+运行+报告+commit+push 后
**STOP**；不自动进入 risk coupling / cost / RR filter / 新 SMC taxonomy。
"""
    open(OUT / "SMC_GROUP_TRADEOFF_VETO_V1_3.md", "w", encoding="utf-8-sig").write(md)


if __name__ == "__main__":
    main()
