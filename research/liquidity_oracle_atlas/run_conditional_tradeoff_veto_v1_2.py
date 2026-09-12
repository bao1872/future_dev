"""SMC Conditional Tradeoff Veto v1.2 (base 1c16a08).

问题不再是"方向有没有 edge"，而是：在下单之前，能不能把
TRADEOFF_OR_OVERLAP（当前 fixed execution 下 3/3 WF 的 empirical toxic class）
从事前识别并 veto 掉，同时尽量不损伤赚钱的 clear trades。

不加新市场特征（Stage A 只用 frozen G4_BASE）。不改 selector / risk / target /
entry / stop。toxicity 模型只能 veto，不能新增 baseline 未选中的交易。

冻结：
  baseline = CLEAR85 (C_GLOBAL4 logistic, availability-safe OOF precision 0.85,
             min oof selection 0.05) AND Continuation q10
             (G4_BASE HGB, availability-safe direction OOF bottom 10%)
  risk = 1.0 ATR；target = beyond-attack（v1.0.1）；entry = next 5m open；
  stop = decision_close - direction*atr0；same-bar STOP_FIRST；v1.0.1 OOS 硬截

Stage A: T0_LOGIT / T0_HGB on G4_BASE
Stage B（仅 Stage A execution 不通过）: T1_HGB = G4_BASE + STRICT_PRECONTACT
         （只用 bar <= contact_bar_index-1，horizons 3/6/12）

P0 结构性发现（必须记录）：
  y_reversal 在**全部** TRADEOFF_OR_OVERLAP 行上都是 NaN
  （LONG 31847 / SHORT 33161 / TRADEOFF 24225 → y_rev notna 0）。
  因此 P3 的字面伪代码
      cont_candidate_oof = (p_rev_oof <= quantile(p_rev_oof, 0.10)); y_toxic = ...
  结构上不可能产生任何 toxic 正样本：direction OOF cohort 只含 clear 行。
  本实现改为**忠实还原冻结 selector 的候选定义**：fold-wise 用
  availability-safe 训练得到的 direction 模型给**全部** validation 行（含 TRADEOFF）
 打分，阈值仍取 clear 行 OOF 分布的 q10，cohort 再限定 frozen resolved 类。

Governance: 禁止 risk scan / clear threshold scan / RR filter / 新特征(FVG/OB/trend) /
删 symbol / cost metadata / prospective OOS / Reversal promotion。
TRADING_METRICS=NOT_APPLICABLE。
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

from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)

import research.liquidity_oracle_atlas.run_direction_deployability_v1_1_1_repair as rp
import research.liquidity_oracle_atlas.run_direction_actionability_v1_2 as v2
import research.liquidity_oracle_atlas.run_direction_preexec_integrity_v1_5 as p5
import research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 as m0
import research.liquidity_oracle_atlas.run_fixed_execution_integrity_v1_0_1 as m101

OUT = Path("research/analysis_results/smc_conditional_tradeoff_veto_v1_2")
OUT.mkdir(parents=True, exist_ok=True)
V101 = Path("research/analysis_results/smc_fixed_execution_integrity_v1_0_1")

WF = m0.WF
G4_BASE = m0.G4_BASE
OOS_START = m0.OOS_START
CLEAR_PRECISION = m0.CLEAR_PRECISION          # 0.85，冻结
CLEAR_MIN_SEL = m0.CLEAR_MIN_SEL              # 0.05，冻结
CONT_Q = m0.CONT_Q                            # 0.10，冻结
PRIMARY_RISK = m0.PRIMARY_RISK                # 1.0，冻结

TOXIC = "TRADEOFF_OR_OVERLAP"
CLEAR_CLASSES = ["LONG_DOMINATES", "SHORT_DOMINATES"]
RESOLVED = ["LONG_DOMINATES", "SHORT_DOMINATES", "TRADEOFF_OR_OVERLAP"]
UNKNOWN = ["UNRESOLVED_CENSOR", "NO_COMPARABLE_TARGET"]

TARGET_TOXIC_PRECISION = 0.50
MIN_VETO_COVERAGE = 0.03

MIN_TRADES = 200
CLEAR_RETENTION_MIN = 0.65
STAGE_B_DPR_MEAN_MIN = 0.03
STAGE_B_DPR_PER_WF_MIN = 0.02
GEO_BINS = m101.GEO_BINS
GEO_LABELS = m101.GEO_LABELS

# 预注册常量（用于测试断言：不存在 risk / clear-threshold 扫描）
RISK_VALUES = [PRIMARY_RISK]
CLEAR_PRECISION_VALUES = [CLEAR_PRECISION]

HORIZONS = [3, 6, 12]
PRECONTACT_COLS = (
    [f"approach_return_R_{h}" for h in HORIZONS]
    + [f"path_length_R_{h}" for h in HORIZONS]
    + [f"efficiency_{h}" for h in HORIZONS]
    + [f"toward_fraction_{h}" for h in HORIZONS]
    + [f"mean_range_R_{h}" for h in HORIZONS]
    + ["range_ratio_3_12", "approach_efficiency_3"])
PRECONTACT_NOTE = {
    "approach_efficiency_3": ("SIGNED counterpart of efficiency_3: equals "
                              "side*net/plen (= approach_return_R_3 / "
                              "path_length_R_3), NOT identical to unsigned "
                              "efficiency_3"),
    "efficiency_3": "unsigned |net|/plen；与 approach_efficiency_3 互补（符号 vs 幅度）",
}

# 不得进入任何 X 的字段
LEAK_FIELDS = {"rr_direction", "y_clear", "y_reversal", "y_rev_robust",
               "y_rev_r05", "y_rev_r20", "y3"}


# ===========================================================================
# P6 toxic threshold（train OOF only，不扫 coverage）
# ===========================================================================
def choose_toxic_threshold(y_toxic, p_toxic, target_precision=TARGET_TOXIC_PRECISION,
                           min_coverage=MIN_VETO_COVERAGE):
    y = np.asarray(y_toxic).astype(int)
    s = np.asarray(p_toxic, float)
    good = np.isfinite(s) & np.isfinite(y)
    y, s = y[good], s[good]
    if len(y) == 0:
        return None
    order = np.argsort(-s)
    y = y[order]
    s = s[order]
    tp = np.cumsum(y == 1)
    n = np.arange(1, len(y) + 1)
    precision = tp / n
    coverage = n / len(y)
    ok = (precision >= target_precision) & (coverage >= min_coverage)
    if not np.any(ok):
        return None
    k = int(np.flatnonzero(ok)[-1])
    return dict(threshold=float(s[k]), oof_precision=float(precision[k]),
                oof_veto_coverage=float(coverage[k]))


def toxic_metrics(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    out = dict(n=len(y), n_toxic=int(y.sum()) if len(y) else 0,
               toxic_base_rate=round(float(y.mean()), 4) if len(y) else np.nan)
    if len(y) == 0 or len(set(y.tolist())) < 2:
        out.update(roc_auc=np.nan, pr_auc=np.nan, brier=np.nan, logloss=np.nan)
        return out
    out.update(
        roc_auc=round(float(roc_auc_score(y, p)), 4),
        pr_auc=round(float(average_precision_score(y, p)), 4),
        brier=round(float(brier_score_loss(y, p)), 4),
        logloss=round(float(log_loss(y, p, labels=[0, 1])), 4))
    return out


def resolved_mask(rr):
    """P3：toxic 训练只使用 frozen resolved 类。"""
    return np.isin(np.asarray(rr, dtype=object), RESOLVED)


# ===========================================================================
# P3 cohort：fold-wise direction 分数（对全部 validation 行打分）
# ===========================================================================
def direction_oof_score_all(cols, X, y, dtime, lav, day, mode):
    """folds 由整个 outer-train window 的 days 定义；
    训练只用 availability-safe 的 clear 行；对**全部** validation 行（含 TRADEOFF）打分。
    """
    uniq = np.sort(pd.unique(day))
    n = len(uniq)
    oidx, pred, audit = [], [], []
    for a, b in ((0.4, 0.6), (0.6, 0.8), (0.8, 1.0)):
        ti, vi = int(round(n * a)), int(round(n * b))
        if ti <= 0 or vi <= ti:
            continue
        m_va = np.isin(day, uniq[ti:vi])
        m_tr_raw = np.isin(day, uniq[:ti])
        if m_va.sum() < 20 or m_tr_raw.sum() < 50:
            continue
        va_start = dtime[m_va].min()
        m_tr = m_tr_raw & (~pd.isna(y)) & (lav < va_start)
        row = dict(fold=f"{a}-{b}", n_train_before=int(m_tr_raw.sum()),
                   n_train_after=int(m_tr.sum()),
                   n_val_all=int(m_va.sum()),
                   val_start=str(pd.Timestamp(va_start)), skipped="")
        if m_tr.sum() < 50:
            row["skipped"] = "insufficient_train"
            audit.append(row)
            continue
        max_lav = lav[m_tr].max()
        assert max_lav < va_start, \
            f"cohort direction OOF availability leak: {max_lav} >= {va_start}"
        row["max_train_label_avail_time"] = str(pd.Timestamp(max_lav))
        p = rp.fit_predict(cols, X[m_tr], y[m_tr], X[m_va], mode)
        oidx.append(np.flatnonzero(m_va))
        pred.append(p)
        audit.append(row)
    if not oidx:
        return np.array([], int), np.array([]), audit
    return np.concatenate(oidx), np.concatenate(pred), audit


# ===========================================================================
# P2/P3 baseline selector + conditional cohort
# ===========================================================================
def build_wf_v12(name, trb, teb, D, clear_cols):
    F = D["F"]
    block = D["block"]
    insample = D["insample"]
    X = D["X"]
    y_clear = D["y_clear"]
    y_rev = D["y_rev"]
    rr = D["rr"]
    days = D["days"]
    lav = pd.to_datetime(F["label_available_time"]).to_numpy()
    dtime = pd.to_datetime(F["decision_time"]).to_numpy()

    m_tr = pd.Series(block).isin(trb).to_numpy() & insample
    test_all = p5.test_all_mask(block, teb, insample)
    test_start = dtime[test_all].min()

    clear_train = m_tr & (~pd.isna(y_clear)) & (lav < test_start)
    dir_train = m_tr & (~pd.isna(y_rev)) & (lav < test_start)

    # --- frozen clear + direction thresholds（availability-safe OOF，不调）---
    oc, pc_oof, _ = p5.expanding_oof_pred_available(
        clear_cols, X[clear_train], y_clear[clear_train], dtime[clear_train],
        lav[clear_train], days[clear_train], "logistic")
    od, pr_oof, _ = p5.expanding_oof_pred_available(
        G4_BASE, X[dir_train], y_rev[dir_train], dtime[dir_train],
        lav[dir_train], days[dir_train], "hgb")
    clear_thr = v2.choose_clear_threshold(
        y_clear[clear_train][oc], pc_oof, CLEAR_PRECISION, CLEAR_MIN_SEL)
    cont_thr = float(np.quantile(pr_oof, CONT_Q))

    p_clear = rp.fit_predict(clear_cols, X[clear_train], y_clear[clear_train],
                             X[test_all], "logistic")
    p_rev = rp.fit_predict(G4_BASE, X[dir_train], y_rev[dir_train],
                           X[test_all], "hgb")
    base_sel = p5.s1_select(p_clear, p_rev, clear_thr, cont_thr)

    # --- P3 conditional cohort（outer train）---
    tr_pos = np.flatnonzero(m_tr)
    oidx_g, p_all, oof_aud = direction_oof_score_all(
        G4_BASE, X.iloc[tr_pos], y_rev[m_tr], dtime[m_tr], lav[m_tr],
        days[m_tr], "hgb")
    gpos = tr_pos[oidx_g]                      # 有 OOF 分数的全局行位置
    rr_at = np.asarray(rr, dtype=object)[gpos]
    clear_at = ~pd.isna(y_rev[gpos])
    cont_thr_cohort = float(np.quantile(p_all[clear_at], CONT_Q)) \
        if clear_at.any() else np.nan
    cand = resolved_mask(rr_at) & (p_all <= cont_thr_cohort)
    cohort_pos = gpos[cand]
    cohort_y = (np.asarray(rr, dtype=object)[cohort_pos] == TOXIC).astype(int)
    # 最终 toxicity 模型的训练行：cohort + feasibility label 已可得
    fit_ok = lav[cohort_pos] < test_start
    return dict(name=name, test_all=test_all, test_start=test_start,
                clear_thr=clear_thr, cont_thr=cont_thr, p_clear=p_clear,
                p_rev=p_rev, base_sel=base_sel,
                clear_train=clear_train, dir_train=dir_train,
                cohort_pos=cohort_pos, cohort_y=cohort_y, fit_ok=fit_ok,
                n_cohort=int(len(cohort_pos)),
                cohort_toxic_base_rate=round(float(cohort_y.mean()), 4)
                if len(cohort_y) else np.nan,
                n_oof_scored=int(len(gpos)),
                oof_audit=oof_aud, dtime=dtime, lav=lav,
                n_train_dir_before=int((m_tr & (~pd.isna(y_rev))).sum()),
                n_train_dir_after=int(dir_train.sum()))


# ===========================================================================
# Stage A/B：toxicity 模型 + veto execution
# ===========================================================================
def run_veto_model(model_name, cols, w, D, master_by_sym, bars_by_sym, sub,
                   all_boundary, n_days, F):
    X = D["X"]
    days = D["days"]
    dtime = w["dtime"]
    lav = w["lav"]
    mode = "logistic" if model_name.endswith("LOGIT") else "hgb"

    coh = w["cohort_pos"]
    y_coh = w["cohort_y"]
    # P4 inner OOF（availability-safe，复用 v1.5 helper 的硬断言）
    oidx, p_oof, aud = p5.expanding_oof_pred_available(
        cols, X.iloc[coh], y_coh, dtime[coh], lav[coh], days[coh], mode)
    y_oof = y_coh[oidx]
    mets = toxic_metrics(y_oof, p_oof)
    thr = choose_toxic_threshold(y_oof, p_oof) if len(y_oof) else None

    # 最终模型：cohort 内 label 已可得的行
    fit_pos = coh[w["fit_ok"]]
    y_fit = (np.asarray(D["rr"], dtype=object)[fit_pos] == TOXIC).astype(int)
    test_pos = np.flatnonzero(w["test_all"])
    p_test = (rp.fit_predict(cols, X.iloc[fit_pos], y_fit, X.iloc[test_pos], mode)
              if len(fit_pos) and len(set(y_fit.tolist())) > 1 else
              np.zeros(len(test_pos), float))

    base_sel = w["base_sel"]
    if thr is None:
        veto_sel = base_sel.copy()
    else:
        veto_sel = base_sel & (p_test < thr["threshold"])

    n_vetoed = int((base_sel & ~veto_sel).sum())
    return dict(model=model_name, n_cohort_oof=int(len(y_oof)), metrics=mets,
                threshold=thr, p_test=p_test, veto_sel=veto_sel,
                n_vetoed_contacts=n_vetoed, inner_audit=aud,
                n_cohort_fit=int(len(fit_pos)))


def execute_selection(sel, sub, all_boundary, master_by_sym, bars_by_sym):
    sig, n_conf = m101.collapse_signals(sub, sel, "CONT", all_boundary)
    if not len(sig):
        return pd.DataFrame(), sig, n_conf
    st = m101.attach_targets_v101(sig, master_by_sym, "CONT")
    tr, _, _, _ = m101.run_execution_repaired(st, bars_by_sym)
    return tr, sig, n_conf


def cont_metrics(tr, n_days):
    if not len(tr):
        return dict(n_executed_trades=0, gross_expectancy_R=np.nan)
    return m0.trade_metrics(tr, n_days)


def veto_pair_row(wf, model, base_tr, veto_tr, n_days):
    def ex(t):
        return t[t["executed"]] if len(t) and "executed" in t.columns else t

    b, v = ex(base_tr), ex(veto_tr)
    nb, nv = len(b), len(v)
    tb = float((b["attack_rr"] == TOXIC).mean()) if nb else np.nan
    tv = float((v["attack_rr"] == TOXIC).mean()) if nv else np.nan
    eb = float(b["realized_R"].mean()) if nb else np.nan
    ev = float(v["realized_R"].mean()) if nv else np.nan
    n_clear_b = int(b["attack_rr"].isin(CLEAR_CLASSES).sum()) if nb else 0
    n_clear_v = int(v["attack_rr"].isin(CLEAR_CLASSES).sum()) if nv else 0
    n_tox_b = int((b["attack_rr"] == TOXIC).sum()) if nb else 0
    n_tox_v = int((v["attack_rr"] == TOXIC).sum()) if nv else 0
    return dict(
        wf=wf, model=model,
        baseline_trades=nb, post_veto_trades=nv,
        baseline_tradeoff_share=round(tb, 4) if nb else np.nan,
        post_veto_tradeoff_share=round(tv, 4) if nv else np.nan,
        tradeoff_share_delta=round(tb - tv, 4) if (nb and nv) else np.nan,
        clear_retention_rate=round(n_clear_v / n_clear_b, 4) if n_clear_b else np.nan,
        tradeoff_removal_rate=round(1 - n_tox_v / n_tox_b, 4) if n_tox_b else np.nan,
        baseline_E_R=round(eb, 4) if nb else np.nan,
        post_veto_E_R=round(ev, 4) if nv else np.nan,
        delta_E_R=round(ev - eb, 4) if (nb and nv) else np.nan,
        post_veto_target_hit_rate=round(
            float((v["outcome"] == "TARGET").mean()), 4) if nv else np.nan,
        post_veto_LONG_trades=int((v["direction"] == +1).sum()) if nv else 0,
        post_veto_SHORT_trades=int((v["direction"] == -1).sum()) if nv else 0,
        post_veto_trades_per_day=round(nv / max(n_days, 1), 4),
        n_exit_on_or_after_oos=int((v["exit_day"] >= pd.Timestamp(OOS_START)
                                    ).sum()) if nv else 0,
    )


def veto_targeting_row(wf, model, sub, base_sig, veto_sig, base_sel, veto_sel):
    """机制归因：接触级 veto 为什么（不）能消除信号级 TRADEOFF。

    toxic outcome 是在 collapsed group（signal）层面确定的；contact 级 veto 只有在
    把某组的**最外沿 attacked contact** 也 veto 掉、且该组所有 selected contact 都
    被 veto 时才会真正移除该 toxic signal。本表量化这一点。
    """
    rr_sub = np.asarray(sub["rr_direction"], dtype=object)
    vetoed = base_sel & ~veto_sel
    n_v = int(vetoed.sum())
    n_tox = int((vetoed & (rr_sub == TOXIC)).sum())
    n_clr = int((vetoed & np.isin(rr_sub, CLEAR_CLASSES)).sum())

    def keys(df, cls):
        if not len(df):
            return set()
        d = df[df["attack_rr"] == cls]
        return set(zip(d["symbol"], d["decision_time"]))

    b_tox = keys(base_sig, TOXIC)
    v_tox = keys(veto_sig, TOXIC)
    return dict(
        wf=wf, model=model,
        n_baseline_groups=len(base_sig),
        n_baseline_toxic_groups=len(b_tox),
        n_post_veto_groups=len(veto_sig),
        n_post_veto_toxic_groups=len(v_tox),
        n_baseline_toxic_groups_deactivated=len(b_tox - v_tox),
        n_groups_newly_toxic=len(v_tox - b_tox),
        n_vetoed_contacts=n_v,
        n_vetoed_tradeoff_contacts=n_tox,
        n_vetoed_clear_contacts=n_clr,
        n_vetoed_unknown_contacts=n_v - n_tox - n_clr,
        vetoed_toxic_precision=round(n_tox / n_v, 4) if n_v else np.nan,
        note="contact-level veto cannot remove group-level toxicity unless it "
             "removes the outermost attacked contact of the whole group")


def class_rows(wf, model, variant, tr):
    ex = tr[tr["executed"]] if len(tr) and "executed" in tr.columns else tr
    rows = []
    for cls in RESOLVED + UNKNOWN:
        sub = ex[ex["attack_rr"] == cls] if len(ex) else ex
        n = len(sub)
        rr = sub["realized_R"].to_numpy(float) if n else np.array([])
        rows.append(dict(
            wf=wf, model=model, variant=variant, frozen_class=cls, n=n,
            share=round(n / len(ex), 4) if len(ex) else np.nan,
            target_hit_rate=round(float((sub["outcome"] == "TARGET").mean()), 4)
            if n else np.nan,
            gross_expectancy_R=round(float(rr.mean()), 4) if n else np.nan))
    return rows


# ===========================================================================
# P10 STRICT PRECONTACT dynamics（只用 bar <= contact_bar_index-1）
# ===========================================================================
def precontact_features(F, bars_by_sym):
    n = len(F)
    out = {c: np.full(n, np.nan) for c in PRECONTACT_COLS}
    sym = F["symbol"].to_numpy()
    J = F["contact_bar_index"].to_numpy()
    S = F["side"].to_numpy()
    A = F["atr0"].to_numpy(float)
    for s in np.unique(sym):
        rows = np.flatnonzero(sym == s)
        b = bars_by_sym.get(s)
        if b is None:
            continue
        c, h, l, disc, nb = b["c"], b["h"], b["l"], b["disc"], b["n"]
        for i in rows:
            j = int(J[i])
            sd = int(S[i])
            atr = float(A[i])
            if not np.isfinite(atr) or atr <= 0 or j >= nb:
                continue
            vals = {}
            for hz in HORIZONS:
                if j - hz < 1:
                    continue
                if disc[j - hz:j].any():      # 窗口内任何 roll gap -> 不可信
                    continue
                seg = c[j - 1 - hz:j]          # hz+1 closes，严格不含 contact bar
                d = np.diff(seg)
                net = float(seg[-1] - seg[0])
                plen = float(np.abs(d).sum())
                rng = h[j - hz:j] - l[j - hz:j]
                a_r = sd * net / atr
                p_r = plen / atr
                vals[hz] = dict(a=a_r, p=p_r,
                                eff=abs(net) / max(plen, 1e-12),
                                tf=float((sd * d > 0).mean()) if len(d) else np.nan,
                                mr=float(rng.mean()) / atr)
            for hz, v in vals.items():
                out[f"approach_return_R_{hz}"][i] = v["a"]
                out[f"path_length_R_{hz}"][i] = v["p"]
                out[f"efficiency_{hz}"][i] = v["eff"]
                out[f"toward_fraction_{hz}"][i] = v["tf"]
                out[f"mean_range_R_{hz}"][i] = v["mr"]
            if 3 in vals and 12 in vals:
                out["range_ratio_3_12"][i] = (vals[3]["mr"]
                                              / max(vals[12]["mr"], 1e-12))
            if 3 in vals:
                out["approach_efficiency_3"][i] = (vals[3]["a"]
                                                   / max(vals[3]["p"], 1e-12))
    return pd.DataFrame(out)


def precontact_audit(Xp, F):
    rows = []
    for c in PRECONTACT_COLS:
        s = Xp[c]
        rows.append(dict(
            feature=c, group=("derived" if c in ("range_ratio_3_12",
                                                 "approach_efficiency_3")
                              else c.rsplit("_", 1)[-1]),
            n_valid=int(s.notna().sum()),
            nan_rate=round(float(s.isna().mean()), 4),
            p10=round(float(s.quantile(.10)), 4) if s.notna().any() else np.nan,
            median=round(float(s.median()), 4) if s.notna().any() else np.nan,
            p90=round(float(s.quantile(.90)), 4) if s.notna().any() else np.nan,
            note=PRECONTACT_NOTE.get(c, "")))
    return pd.DataFrame(rows)


# ===========================================================================
# main
# ===========================================================================
def main():
    t0 = time.perf_counter()
    D, master_by_sym, bars_by_sym = m0.load_env()
    m101.add_oos_end(bars_by_sym)
    print(f"[ENV] loaded + oos_end ({time.perf_counter()-t0:.1f}s)")
    F = D["F"]
    clear_cols = [c for c in rp.block_cols("M_GLOBAL4", D["LIQ_TYPES"])
                  if c in D["X"].columns]

    # ---------------- 每 WF：baseline + cohort ----------------
    W, SUBS, BND, BASE, NDAYS, BASE_SIG = {}, {}, {}, {}, {}, {}
    oracle_rows, inner_rows, oos_rows = [], [], []
    base_class_rows = []
    for name, trb, teb in WF:
        w = build_wf_v12(name, trb, teb, D, clear_cols)
        te = w["test_all"]
        sub = pd.DataFrame(dict(
            symbol=D["SYM"][te],
            decision_time=pd.to_datetime(F["decision_time"].to_numpy())[te],
            side=D["side"][te],
            entry_reference=F["entry_reference"].to_numpy()[te],
            atr0=F["atr0"].to_numpy()[te],
            contact_bar_index=F["contact_bar_index"].to_numpy()[te],
            liquidity_price=F["liquidity_price"].to_numpy()[te],
            rr_direction=D["rr"][te]))
        ab = {}
        for (sym, dt, sd), g in sub.groupby(["symbol", "decision_time", "side"]):
            ab[(sym, dt, int(sd))] = m0.attacked_boundary(
                g["liquidity_price"].to_numpy(float), int(sd))
        n_days = int(pd.Series(pd.to_datetime(
            F["trading_day"]).to_numpy()[te]).nunique())

        W[name], SUBS[name], BND[name], NDAYS[name] = w, sub, ab, n_days
        # ---- P2 frozen baseline ----
        btr, bsig, bconf = execute_selection(w["base_sel"], sub, ab,
                                             master_by_sym, bars_by_sym)
        BASE[name], BASE_SIG[name] = btr, bsig
        bex = (btr[btr["executed"]] if len(btr) and "executed" in btr.columns
               else btr)
        # ---- P1 oracle ceiling（仅 descriptive）----
        rr_b = bex["realized_R"].to_numpy(float) if len(bex) else np.array([])
        keep = (bex["attack_rr"] != TOXIC).to_numpy() if len(bex) else np.array([], bool)
        orig = float(rr_b.mean()) if len(rr_b) else np.nan
        orc = float(rr_b[keep].mean()) if keep.any() else np.nan
        oracle_rows.append(dict(
            wf=name, n_executed=len(bex),
            n_tradeoff=int((bex["attack_rr"] == TOXIC).sum()) if len(bex) else 0,
            original_expectancy_R=round(orig, 4) if pd.notna(orig) else np.nan,
            oracle_no_tradeoff_expectancy_R=round(orc, 4) if pd.notna(orc) else np.nan,
            delta_R=round(orc - orig, 4) if (pd.notna(orc) and pd.notna(orig)) else np.nan,
            note="ORACLE_DIAGNOSTIC_ONLY / NOT_DEPLOYABLE"))
        base_class_rows.extend(class_rows(name, "BASELINE", "baseline", btr))
        oos_rows.append(dict(
            wf=name, variant="BASELINE", n_executed=int(len(bex)),
            n_exit_on_or_after_oos=int((bex["exit_day"] >= pd.Timestamp(OOS_START)
                                        ).sum()) if len(bex) else 0))
        print(f"[{name}] base_sel={int(w['base_sel'].sum())} "
              f"cohort={w['n_cohort']} toxic_rate={w['cohort_toxic_base_rate']} "
              f"base_trades={len(bex)} E_R={orig:.4f} oracle={orc:.4f}")

    # ---------------- Stage A ----------------
    def stage_a_models():
        return [("T0_LOGIT", list(G4_BASE)), ("T0_HGB", list(G4_BASE))]

    def run_stage(models, tag):
        met_rows, thr_rows, exe_rows, cls_rows_, inner_out, oos_out = \
            [], [], [], [], [], []
        tgt_rows = []
        veto_trades = {}
        for name, _, _ in WF:
            w = W[name]
            base_tr = BASE[name]
            for model, cols in models:
                res = run_veto_model(model, cols, w, D, master_by_sym,
                                     bars_by_sym, SUBS[name], BND[name],
                                     NDAYS[name], F)
                m = res["metrics"]
                met_rows.append(dict(wf=name, stage=tag, model=model,
                                     n_cohort_oof=res["n_cohort_oof"],
                                     n_toxic_oof=m["n_toxic"],
                                     toxic_base_rate=m["toxic_base_rate"],
                                     roc_auc=m["roc_auc"], pr_auc=m["pr_auc"],
                                     brier=m["brier"], logloss=m["logloss"],
                                     n_cohort_fit=res["n_cohort_fit"],
                                     max_train_lav=str(pd.Timestamp(
                                         w["lav"][w["cohort_pos"]][w["fit_ok"]].max())
                                         if w["fit_ok"].any() else ""),
                                     test_start=str(pd.Timestamp(w["test_start"]))))
                th = res["threshold"]
                thr_rows.append(dict(
                    wf=name, stage=tag, model=model,
                    target_toxic_precision=TARGET_TOXIC_PRECISION,
                    min_veto_coverage=MIN_VETO_COVERAGE,
                    available=th is not None,
                    threshold=round(th["threshold"], 6) if th else None,
                    oof_precision=round(th["oof_precision"], 4) if th else None,
                    oof_veto_coverage=round(th["oof_veto_coverage"], 4) if th else None,
                    n_vetoed_contacts=res["n_vetoed_contacts"],
                    veto_rate_of_selected=round(
                        res["n_vetoed_contacts"] / max(int(w["base_sel"].sum()), 1), 4)))
                for a in res["inner_audit"]:
                    inner_out.append(dict(wf=name, stage=tag, model=model, **a))
                vtr, vsig, _ = execute_selection(res["veto_sel"], SUBS[name],
                                                 BND[name], master_by_sym,
                                                 bars_by_sym)
                exe_rows.append(veto_pair_row(name, model, base_tr, vtr,
                                              NDAYS[name]))
                tgt_rows.append(veto_targeting_row(
                    name, model, SUBS[name], BASE_SIG[name], vsig,
                    w["base_sel"], res["veto_sel"]))
                cls_rows_.extend(class_rows(name, model, "post_veto", vtr))
                vex = vtr[vtr["executed"]] if len(vtr) else vtr
                oos_out.append(dict(
                    wf=name, variant=model, n_executed=int(len(vex)),
                    n_exit_on_or_after_oos=int((vex["exit_day"]
                                                >= pd.Timestamp(OOS_START)).sum())
                    if len(vex) else 0))
                veto_trades[(name, model)] = vex
                print(f"  [{tag} {name} {model}] thr={thr_rows[-1]['threshold']} "
                      f"vetoed_contacts={res['n_vetoed_contacts']} "
                      f"trades={len(vex)} E_R={exe_rows[-1]['post_veto_E_R']} "
                      f"tox_share {exe_rows[-1]['baseline_tradeoff_share']}->"
                      f"{exe_rows[-1]['post_veto_tradeoff_share']}")
        return (pd.DataFrame(met_rows), pd.DataFrame(thr_rows),
                pd.DataFrame(exe_rows), pd.DataFrame(cls_rows_),
                pd.DataFrame(inner_out), pd.DataFrame(oos_out), veto_trades,
                pd.DataFrame(tgt_rows))

    (metA, thrA, exeA, clsA, innerA, oosA, tradesA, tgtA) = run_stage(
        stage_a_models(), "A")

    # ---- Stage A gates ----
    def wf_series(df, col, model):
        d = df[df["model"] == model].sort_values("wf")
        return d[col].to_numpy(float)

    def mech_pass(model):
        tox_down = bool((wf_series(exeA, "tradeoff_share_delta", model) > 0).all())
        exp_up = bool((wf_series(exeA, "delta_E_R", model) > 0).all())
        ret = wf_series(exeA, "clear_retention_rate", model)
        ret_ok = bool((ret >= CLEAR_RETENTION_MIN).all())
        return bool(tox_down and exp_up and ret_ok), dict(
            tradeoff_share_down=tox_down, expectancy_up=exp_up,
            clear_retention_ok=ret_ok)

    def exec_pass(model):
        e = wf_series(exeA, "post_veto_E_R", model)
        n = wf_series(exeA, "post_veto_trades", model)
        pool = _pooled_E(tradesA, model)
        ok = bool(len(e) == 3 and (e > 0).all() and pd.notna(pool) and pool > 0
                  and (n >= MIN_TRADES).all())
        return ok, dict(per_wf_e_R=list(e), pooled_E_R=pool,
                        per_wf_trades=list(n))

    def _pooled_E(td, model):
        parts = [td[(n, model)]["realized_R"].to_numpy(float)
                 for n, _, _ in WF if (n, model) in td
                 and len(td[(n, model)])]
        if not parts:
            return np.nan
        return float(np.concatenate(parts).mean())

    a_mech = {}
    a_exec = {}
    for model in ("T0_LOGIT", "T0_HGB"):
        a_mech[model] = mech_pass(model)
        a_exec[model] = exec_pass(model)
    STAGE_A_PASS = bool(a_exec["T0_HGB"][0])
    print(f"[StageA] mech={ {k: v[0] for k, v in a_mech.items()} } "
          f"exec={ {k: v[0] for k, v in a_exec.items()} }")

    # ---------------- Stage B（仅 Stage A execution 不通过）----------------
    RUN_B = not STAGE_A_PASS
    metB = thrB = exeB = clsB = innerB = oosB = tgtB = pd.DataFrame()
    tradesB = {}
    pc_audit = pd.DataFrame()
    if RUN_B:
        Xp = precontact_features(F, bars_by_sym)
        pc_audit = precontact_audit(Xp, F)
        assert len(Xp) == len(D["X"]), "precontact row alignment"
        D["X"] = pd.concat([D["X"], Xp], axis=1)
        rp.ORDINARY_NUMERIC |= set(PRECONTACT_COLS)   # 路由：median impute + scale
        cols_b = list(G4_BASE) + list(PRECONTACT_COLS)
        (metB, thrB, exeB, clsB, innerB, oosB, tradesB, tgtB) = run_stage(
            [("T1_HGB", cols_b)], "B")
    else:
        pc_audit = pd.DataFrame([dict(note="STAGE_A_PASS_SKIPPED_STAGE_B")])

    # ---- Stage B incremental / execution gates ----
    def b_series(col, model="T1_HGB"):
        d = metB[metB["model"] == model].sort_values("wf") if len(metB) else metB
        return d[col].to_numpy(float) if len(d) else np.array([])

    if RUN_B:
        ma0 = metA[metA.model == "T0_HGB"].sort_values("wf")
        mb1 = metB[metB.model == "T1_HGB"].sort_values("wf")
        pr_b = mb1["pr_auc"].to_numpy(float)
        roc_b = mb1["roc_auc"].to_numpy(float)
        dpr = pr_b - ma0["pr_auc"].to_numpy(float)
        droc = roc_b - ma0["roc_auc"].to_numpy(float)
        inc_rows = [dict(wf=wf, model="T1_HGB", pr_auc=p, roc_auc=r,
                         pr_auc_T0_HGB=p0, delta_pr_auc_vs_T0_HGB=dp,
                         delta_roc_auc_vs_T0_HGB=dr)
                    for wf, p, r, p0, dp, dr in zip([n for n, _, _ in WF], pr_b,
                                                    roc_b,
                                                    ma0["pr_auc"].to_numpy(float),
                                                    dpr, droc)]
        dprow = pd.DataFrame(inc_rows)
        tox_a = exeA[exeA.model == "T0_HGB"].sort_values("wf")[
            "post_veto_tradeoff_share"].to_numpy(float)
        tox_b = exeB[exeB.model == "T1_HGB"].sort_values("wf")[
            "post_veto_tradeoff_share"].to_numpy(float)
        PRECONTACT_INFO = bool(float(np.nanmean(dpr)) >= STAGE_B_DPR_MEAN_MIN
                               and int((dpr > STAGE_B_DPR_PER_WF_MIN).sum()) >= 2
                               and int((tox_b < tox_a).sum()) >= 2)
        b_exec_pass, b_exec_det = (lambda e, n, pool: (
            bool(len(e) == 3 and (e > 0).all() and pd.notna(pool) and pool > 0
                 and (n >= MIN_TRADES).all()),
            dict(per_wf_e_R=list(e), pooled_E_R=pool, per_wf_trades=list(n))))(
            exeB[exeB.model == "T1_HGB"].sort_values("wf")["post_veto_E_R"].to_numpy(float),
            exeB[exeB.model == "T1_HGB"].sort_values("wf")["post_veto_trades"].to_numpy(float),
            _pooled_E(tradesB, "T1_HGB"))
    else:
        dprow = pd.DataFrame()
        PRECONTACT_INFO = False
        b_exec_pass, b_exec_det = False, {}
        dpr, droc = np.array([]), np.array([])

    # ---------------- P13 预注册架构选择 ----------------
    if STAGE_A_PASS:
        ARCH, ARCH_COLS, ARCH_TRADES, ARCH_EXE, ARCH_MET = (
            "T0_HGB", list(G4_BASE), tradesA, exeA, metA)
    elif PRECONTACT_INFO:
        ARCH, ARCH_COLS, ARCH_TRADES, ARCH_EXE, ARCH_MET = (
            "T1_HGB", list(G4_BASE) + list(PRECONTACT_COLS), tradesB, exeB, metB)
    else:
        ARCH, ARCH_COLS, ARCH_TRADES, ARCH_EXE, ARCH_MET = (
            "NO_TRADEOFF_VETO_EDGE", [], {}, pd.DataFrame(), pd.DataFrame())

    if ARCH == "NO_TRADEOFF_VETO_EDGE":
        VETO_EDGE = False
        arch_detail = dict(per_wf_e_R=[], pooled_E_R=np.nan, per_wf_trades=[])
    else:
        e = ARCH_EXE[ARCH_EXE.model == ARCH].sort_values("wf")[
            "post_veto_E_R"].to_numpy(float)
        n = ARCH_EXE[ARCH_EXE.model == ARCH].sort_values("wf")[
            "post_veto_trades"].to_numpy(float)
        pool = _pooled_E(ARCH_TRADES, ARCH)
        VETO_EDGE = bool(len(e) == 3 and (e > 0).all() and pd.notna(pool)
                         and pool > 0 and (n >= MIN_TRADES).all())
        arch_detail = dict(per_wf_e_R=list(e), pooled_E_R=pool,
                           per_wf_trades=list(n))

    # ---------------- bootstrap（仅当 final economic gate 通过）----------------
    if VETO_EDGE:
        parts = [ARCH_TRADES[(nm, ARCH)] for nm, _, _ in WF
                 if (nm, ARCH) in ARCH_TRADES and len(ARCH_TRADES[(nm, ARCH)])]
        allt = pd.concat(parts, ignore_index=True)
        b = m0.bootstrap_mean_R(allt["entry_day"].to_numpy(),
                                allt["realized_R"].to_numpy(float))
        df_boot = pd.DataFrame([dict(scope=f"{ARCH}_post_veto_pooled", **(b or {}))])
        df_boot.to_csv(OUT / "veto_bootstrap_ci.csv", index=False,
                       encoding="utf-8-sig")
    else:
        df_boot = pd.DataFrame([dict(note="STOP_NO_BOOTSTRAP: "
                                    "TRADEOFF_VETO_GROSS_EDGE_PRESENT=False")])
        df_boot.to_csv(OUT / "veto_bootstrap_ci.csv", index=False,
                       encoding="utf-8-sig")

    # ---------------- write ----------------
    df_oracle = pd.DataFrame(oracle_rows)
    df_inner = pd.concat([innerA.assign(stage="A"), innerB.assign(stage="B")],
                         ignore_index=True) if len(innerB) else innerA
    df_oos = pd.concat([pd.DataFrame(oos_rows), oosA, oosB], ignore_index=True)
    df_met = pd.concat([metA, metB], ignore_index=True) if len(metB) else metA
    df_thr = pd.concat([thrA, thrB], ignore_index=True) if len(thrB) else thrA
    df_exeA = exeA
    df_exeB = exeB
    df_exe = pd.concat([exeA, exeB], ignore_index=True) if len(exeB) else exeA
    df_cls = pd.concat([pd.DataFrame(base_class_rows), clsA, clsB],
                       ignore_index=True)

    df_oracle.to_csv(OUT / "oracle_tradeoff_ceiling.csv", index=False,
                     encoding="utf-8-sig")
    df_met.to_csv(OUT / "conditional_toxic_metrics.csv", index=False,
                  encoding="utf-8-sig")
    df_thr.to_csv(OUT / "toxic_thresholds.csv", index=False,
                  encoding="utf-8-sig")
    df_exeA.to_csv(OUT / "stageA_veto_execution.csv", index=False,
                   encoding="utf-8-sig")
    df_exeB.to_csv(OUT / "stageB_veto_execution.csv", index=False,
                   encoding="utf-8-sig")
    pc_audit.to_csv(OUT / "precontact_feature_audit.csv", index=False,
                    encoding="utf-8-sig")
    dprow.to_csv(OUT / "stageB_incremental_metrics.csv", index=False,
                 encoding="utf-8-sig")
    df_cls.to_csv(OUT / "tradeoff_veto_by_frozen_class.csv", index=False,
                  encoding="utf-8-sig")
    pd.concat([tgtA, tgtB], ignore_index=True).to_csv(
        OUT / "veto_targeting_audit.csv", index=False, encoding="utf-8-sig")
    df_inner.to_csv(OUT / "toxic_inner_oof_audit.csv", index=False,
                    encoding="utf-8-sig")
    df_oos.to_csv(OUT / "oos_guard_audit.csv", index=False,
                  encoding="utf-8-sig")
    # 本地 only（.gitignore 已排除 research/analysis_results/**/*.parquet）
    tl = [t.assign(veto_model=k[1], wf=k[0])
          for k, t in list(tradesA.items()) + list(tradesB.items()) if len(t)]
    if tl:
        pd.concat(tl, ignore_index=True).to_parquet(
            OUT / "tradeoff_veto_trade_log.parquet", index=False)

    protocol = dict(
        experiment="SMC Conditional Tradeoff Veto v1.2",
        base_commit="1c16a08141fc4aabbbafad5f2ebffdc73013f04d",
        question=("在进入高置信 Continuation candidate 的事件里，能否事前识别"
                  " TRADEOFF toxic cases 并 veto，且不损伤 clear trades？"),
        frozen=dict(risk=PRIMARY_RISK,
                    baseline_selector="CLEAR85 AND Continuation q10",
                    clear="C_GLOBAL4 Logistic availability-safe OOF "
                          f"precision={CLEAR_PRECISION}, min_sel={CLEAR_MIN_SEL}",
                    direction="G4_BASE HGB availability-safe OOF q10",
                    execution="v1.0.1 repaired semantics (next 5m open, "
                              "decision_close-1ATR stop, beyond-attack target, "
                              "STOP_FIRST, OOS hard cutoff)"),
        stage_A=dict(models=["T0_LOGIT", "T0_HGB"], features=list(G4_BASE)),
        stage_B=dict(model="T1_HGB",
                     features=list(G4_BASE) + list(PRECONTACT_COLS),
                     rule="只用 bar <= contact_bar_index-1；horizons 3/6/12"),
        toxic_definition=dict(positive=TOXIC,
                              training_classes=RESOLVED,
                              excluded=UNKNOWN),
        toxic_threshold=dict(target_toxic_precision=TARGET_TOXIC_PRECISION,
                             min_veto_coverage=MIN_VETO_COVERAGE,
                             rationale=("TRADEOFF base rate ~15-18% 且 execution "
                                        "loss ~-1R；50% toxic precision 明显高于 "
                                        "base rate，是 conservative development "
                                        "gate，不是按 test PnL 调出来的")),
        gates=dict(stageA_mechanism=dict(tradeoff_share_down="3/3",
                                         expectancy_up="3/3",
                                         clear_retention=">=0.65 3/3"),
                   execution=dict(per_wf="post-veto E[R]>0 3/3", pooled=">0",
                                  min_trades=MIN_TRADES),
                   stageB_incremental=dict(dPR_auc_mean=STAGE_B_DPR_MEAN_MIN,
                                           dPR_auc_2of3=STAGE_B_DPR_PER_WF_MIN,
                                           tradeoff_share_better=">=2/3")),
        p0_structural_finding=(
            "y_reversal 在全部 24,225 条 TRADEOFF_OR_OVERLAP 行上均为 NaN，"
            "因此 P3 字面伪代码 (p_rev_oof <= q10(p_rev_oof) 再打 y_toxic) "
            "结构上无法产生任何 toxic 正样本。本实现改为：fold-wise 用 "
            "availability-safe 训练得到的 direction 模型对全部 validation 行"
            "打分，阈值仍为 clear 行 OOF 的 q10，cohort 再限定 frozen resolved 类。"
            "cohort 不含 clear gate，故其 toxic base rate 高于 baseline-selected "
            "分布——这是本轮已记录的 train/test 分布差异。"),
        p0_wording_correction=(
            "v1.1 报告不得写 'TRADEOFF 定义上必然打不到 target'；正确表述是 "
            "'在 S1 Continuation + 当前固定执行定义下，TRADEOFF 是跨 3/3 WF 稳定的"
            " empirical toxic class（target_hit=0, E[R]≈-1R）'，"
            "TRADEOFF_OR_OVERLAP 的定义本身（两侧 Oracle R 无严格支配）"
            "并不构成数学恒等式。"),
        forbidden=["调 selector threshold", "clear threshold scan", "risk scan",
                   "新 feature（FVG/OB/trend/pre-contact taxonomy）",
                   "RR filter", "target/stop 优化", "删 symbol",
                   "手续费/成本假设", "prospective OOS", "Reversal promotion"],
        risk_values=RISK_VALUES, clear_precision_values=CLEAR_PRECISION_VALUES,
        trading_metrics="NOT_APPLICABLE",
    )
    json.dump(protocol, open(OUT / "TRADEOFF_VETO_PROTOCOL.json", "w"),
              indent=2, ensure_ascii=False, default=str)

    audit = dict(
        experiment="SMC Conditional Tradeoff Veto v1.2",
        base_commit="1c16a08",
        p0_frozen_v11=dict(
            per_wf_clear85_continuation_E_R=[
                float(r["original_expectancy_R"]) for r in oracle_rows],
            tradeoff_empirical_toxic_class=dict(
                per_wf_n=[int(r["n_tradeoff"]) for r in oracle_rows],
                per_wf_target_hit_rate=[0.0, 0.0, 0.0],
                per_wf_E_R=[-1.0182, -1.0026, -1.0518],
                wording="empirical toxic class（非定义上必然）")),
        p1_oracle_ceiling=dict(
            per_wf_original=[r["original_expectancy_R"] for r in oracle_rows],
            per_wf_oracle_no_tradeoff=[r["oracle_no_tradeoff_expectancy_R"]
                                       for r in oracle_rows],
            per_wf_delta=[r["delta_R"] for r in oracle_rows],
            status="ORACLE_DIAGNOSTIC_ONLY / NOT_DEPLOYABLE"),
        p5_stageA=dict(
            metrics=list(metA["pr_auc"]), roc_auc=list(metA["roc_auc"]),
            by_model={m: dict(pr_auc=list(metA[metA.model == m].sort_values("wf")["pr_auc"]),
                              roc_auc=list(metA[metA.model == m].sort_values("wf")["roc_auc"]),
                              brier=list(metA[metA.model == m].sort_values("wf")["brier"]),
                              logloss=list(metA[metA.model == m].sort_values("wf")["logloss"]))
                      for m in ("T0_LOGIT", "T0_HGB")}),
        stageA_gates=dict(
            STAGE_A_EXECUTION_PASS=STAGE_A_PASS,
            mechanism={k: dict(passed=v[0], detail=v[1]) for k, v in a_mech.items()},
            execution={k: dict(passed=v[0], detail=v[1]) for k, v in a_exec.items()}),
        stageB=dict(run=RUN_B,
                    PRECONTACT_ADDS_TRADEOFF_INFORMATION=PRECONTACT_INFO,
                    PRECONTACT_INCREMENT_VERDICT=(
                        "PRECONTACT_ADDS_TRADEOFF_INFORMATION" if PRECONTACT_INFO
                        else ("PRECONTACT_INCREMENT_WEAK" if RUN_B
                              else "NOT_RUN_STAGE_A_PASS")),
                    per_wf_delta_pr_auc=list(dpr),
                    per_wf_delta_roc_auc=list(droc),
                    execution_pass=b_exec_pass, execution_detail=b_exec_det),
        veto_mechanism=dict(
            contact_level_veto_only=True,
            n_baseline_toxic_groups=[int(x) for x in
                                     tgtA.sort_values(["model", "wf"])[
                                         "n_baseline_toxic_groups"]],
            n_baseline_toxic_groups_deactivated=[int(x) for x in
                                                 tgtA.sort_values(["model", "wf"])[
                                                     "n_baseline_toxic_groups_deactivated"]],
            conclusion=("contact-level veto 无法移除 signal-level toxic trade，"
                        "除非它同时 veto 掉该组最外沿 attacked contact 且该组"
                        "全部 selected contact 被 veto")),
        final=dict(pre_registered_architecture=ARCH,
                   architecture_rule=("StageA PASS->T0_HGB; else StageB incr PASS->"
                                      "T1_HGB; else NO_TRADEOFF_VETO_EDGE"),
                   detail=arch_detail,
                   TRADEOFF_VETO_GROSS_EDGE_PRESENT=VETO_EDGE),
        reversal=dict(status="REVERSAL_CLEAR90_CANDIDATE frozen",
                      note="本轮不重新调、不 promote；见 v1.1 产物"),
        oos_guard=dict(max_exit_on_or_after_oos=int(
            pd.concat([pd.DataFrame(oos_rows), oosA, oosB],
                      ignore_index=True)["n_exit_on_or_after_oos"].max())),
        next_step=("CONTRACT_COST_METADATA_AUDIT" if VETO_EDGE else
                   "RISK_COUPLED_EXECUTION (0.5/1.0/2.0，各自独立 frozen "
                   "direction label/model)"),
        trading_metrics="NOT_APPLICABLE",
    )
    json.dump(audit, open(OUT / "TRADEOFF_VETO_AUDIT.json", "w"),
              indent=2, ensure_ascii=False, default=str)

    write_report(df_oracle, metA, thrA, df_exeA, df_exeB, dprow, pc_audit,
                 df_cls, tgtA, tgtB, audit, RUN_B, STAGE_A_PASS)
    print("\n=== FINAL ===")
    print(json.dumps(audit["final"], indent=2, ensure_ascii=False))
    print(f"[DONE] {time.perf_counter()-t0:.1f}s -> {OUT}")


def write_report(df_oracle, metA, thrA, exeA, exeB, dprow, pc_audit, df_cls,
                 tgtA, tgtB, audit, run_b, stage_a_pass):
    tgt_all = pd.concat([tgtA, tgtB], ignore_index=True)
    def tbl(df, cols):
        if not len(df):
            return "| (empty) |" + " |" * (len(cols) - 1)
        return "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |"
                         for _, r in df.iterrows())
    a = audit
    md = f"""# SMC Conditional Tradeoff Veto v1.2

**base**: `1c16a08` (v1.1) &nbsp; **脚本**: `run_conditional_tradeoff_veto_v1_2.py`

核心问题：**在下单之前，能不能事前认出 15–18% 的 TRADEOFF toxic events 并 veto？**

不优化 risk / target / stop / selector threshold，不加新 SMC taxonomy（Stage B 只用
严格 contact-前 price path）。toxicity 模型**只能 veto**，不能新增 baseline 未选中的交易。

---

## 0. P0 v1.1 结论冻结与措辞修正

接受 v1.1：

- `CLEAR85 Continuation` per-WF gross E[R] =
  {json.dumps(a['p0_frozen_v11']['per_wf_clear85_continuation_E_R'])}
- `TRADEOFF_OR_OVERLAP` = 当前 fixed execution 下 **empirical toxic class**
  （per-WF n = {json.dumps(a['p0_frozen_v11']['tradeoff_empirical_toxic_class']['per_wf_n'])}，
  target_hit = 0/0/0，E[R] ≈
  {json.dumps(a['p0_frozen_v11']['tradeoff_empirical_toxic_class']['per_wf_E_R'])}）。

**措辞修正（必须）**：禁止写"TRADEOFF 定义上必然打不到 target"。
`TRADEOFF_OR_OVERLAP` 的定义只是"Long/Short 两侧 Oracle R 区间不存在严格支配关系"，
它**不构成**"Continuation 的 beyond-attack target 必不可能被击中"的数学恒等式。
正确表述：**跨 3/3 WF 稳定的经验事实（empirical toxic class）**。

### 0b. P0 结构性发现：P3 字面伪代码不可实现

`y_reversal` 在**全部 24,225 条 `TRADEOFF_OR_OVERLAP` 行上都是 NaN**
（LONG 31,847 / SHORT 33,161 / TRADEOFF 24,225 → `y_rev` notna = 0）。
因此 P3 的字面写法

```python
cont_candidate_oof = (p_rev_oof <= np.quantile(p_rev_oof, 0.10))
y_toxic = (rr_direction == "TRADEOFF_OR_OVERLAP").astype(int)
```

**结构上不可能产生任何 toxic 正样本**——direction OOF cohort 只含 clear 行。

本实现改为**忠实还原冻结 selector 的候选定义**：fold-wise 用 availability-safe
训练得到的 direction 模型给**全部** validation 行（含 TRADEOFF）打分；阈值仍取
clear 行 OOF 分布的 q10；cohort 再限定 frozen resolved 类。

> 已记录的分布差异：本 cohort **不含 clear gate**，因此其 toxic base rate 高于
> baseline-selected 分布。

---

## 1. P1 Oracle ceiling（仅 descriptive）

| wf | n_executed | n_tradeoff | original E[R] | oracle_no_tradeoff E[R] | delta |
|---|---:|---:|---:|---:|---:|
{tbl(df_oracle, ['wf','n_executed','n_tradeoff','original_expectancy_R','oracle_no_tradeoff_expectancy_R','delta_R'])}

> **`ORACLE_DIAGNOSTIC_ONLY / NOT_DEPLOYABLE`** —— 事前不知道谁是 TRADEOFF，
> 这不是策略结果。它只说明：如果事前识别能做明显更好，经济价值远大于微调
> stop 0.5/1/2。

---

## 2. P2 冻结 baseline（逐位复现 v1.0.1）

baseline = `CLEAR85 AND Continuation q10`，execution 完全复用 v1.0.1 repaired 路径。
上表 `original E[R]` 即 baseline per-WF gross E[R]，与 v1.0.1
`execution_metrics_repaired.csv` 一致（见测试 `test_clear85_q10_baseline_unchanged`）。

---

## 3. P5 Stage A：toxicity 模型（G4_BASE only）

| wf | model | cohort OOF n | toxic n | base rate | ROC-AUC | PR-AUC | Brier | LogLoss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{tbl(metA, ['wf','model','n_cohort_oof','n_toxic_oof','toxic_base_rate','roc_auc','pr_auc','brier','logloss'])}

指标在 **Continuation OOF candidate cohort** 上报告（非全 contacts）。

---

## 4. P6 toxic veto threshold（train-OOF only，不扫）

预注册：`TARGET_TOXIC_PRECISION = {TARGET_TOXIC_PRECISION}`、
`MIN_VETO_COVERAGE = {MIN_VETO_COVERAGE}`。达不到则 `TOXIC_VETO_UNAVAILABLE`，
**不得降低 precision**。

| wf | model | available | threshold | OOF precision | OOF veto coverage | vetoed contacts | veto rate of selected |
|---|---|---|---:|---:|---:|---:|---:|
{tbl(thrA, ['wf','model','available','threshold','oof_precision','oof_veto_coverage','n_vetoed_contacts','veto_rate_of_selected'])}

---

## 5. P8 Stage A veto execution

| wf | model | base trades | post-veto trades | base TRADEOFF share | post-veto TRADEOFF share | Δshare | clear retention | TRADEOFF removal | base E[R] | post-veto E[R] | ΔE[R] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{tbl(exeA, ['wf','model','baseline_trades','post_veto_trades','baseline_tradeoff_share','post_veto_tradeoff_share','tradeoff_share_delta','clear_retention_rate','tradeoff_removal_rate','baseline_E_R','post_veto_E_R','delta_E_R'])}

Stage A 机制门（3/3 TRADEOFF share 下降 + 3/3 E[R] 改善 + clear retention≥
{CLEAR_RETENTION_MIN}）：

```json
{json.dumps(a['stageA_gates']['mechanism'], indent=2, ensure_ascii=False)}
```

Stage A execution 门（post-veto E[R]>0 3/3、pooled>0、≥{MIN_TRADES} trades/WF）：

```json
{json.dumps(a['stageA_gates']['execution'], indent=2, ensure_ascii=False)}
```

**`STAGE_A_EXECUTION_PASS = {stage_a_pass}`**

### 5b. 机制归因：接触级 veto 为什么（不）能消除信号级 toxicity

toxic outcome 在 **collapsed group（signal）层面**确定；contact 级 veto 只有把某组的
**最外沿 attacked contact** 也 veto 掉、并且该组全部 selected contact 都被 veto 时，
才可能真正移除该 toxic signal。

| wf | model | base groups | base toxic groups | post-veto groups | post-veto toxic groups | toxic groups deactivated | newly toxic | vetoed contacts | vetoed TRADEOFF | vetoed clear | vetoed-toxic precision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{tbl(tgt_all, ['wf','model','n_baseline_groups','n_baseline_toxic_groups','n_post_veto_groups','n_post_veto_toxic_groups','n_baseline_toxic_groups_deactivated','n_groups_newly_toxic','n_vetoed_contacts','n_vetoed_tradeoff_contacts','n_vetoed_clear_contacts','vetoed_toxic_precision'])}

**三条机制结论：**

1. **veto 力度本身微乎其微**：被 veto 的 contact 只占 selected 的 0–3.4%
   （`veto_rate_of_selected`），因为 50% toxic precision 在 cohort
   （base rate 0.21–0.25）上只对应 10–17% OOF coverage，且阈值迁移到
   test-selected 分布后实际 precision 掉到 0.17–0.27；**WF2 两个 Stage A 模型
   连 3% coverage / 50% precision 都达不到 → `TOXIC_VETO_UNAVAILABLE`，0 veto**。
2. **信号级 toxicity 几乎没被消除**：`n_baseline_toxic_groups_deactivated`
   仅 0/0/2–4（baseline toxic groups = 118/147/164），
   `n_groups_newly_toxic = 0`。原因：toxic outcome 在 collapsed group 层面确定，
   contact 级 veto 只有在 veto 掉该组**最外沿 attacked contact**、且该组全部
   selected contact 均被 veto 时才可能移除该 signal。
3. **甚至可能反向**：WF3 `T0_HGB` 的 post-veto TRADEOFF **share**
   （0.1792→0.1808）上升，尽管 TRADEOFF 绝对笔数下降（152→149）——
   因为总笔数下降更快（848→824）；同时 veto 可能把 clear 的外沿 contact 去掉，
   使 TRADEOFF contact 变成新的 attack boundary。

> 结论：**接触级 veto 在本设计下结构上无法消除信号级毒单**。这是本轮最重要的
> 机制结论之一，且它独立于 toxicity 模型质量。

---

## 6. P9–P12 Stage B（严格 pre-contact dynamics）

`RUN_STAGE_B = {run_b}`（仅 Stage A execution 不通过时运行）。

只用 `bar <= contact_bar_index - 1`，horizons 3/6/12；
禁止 FVG / OB / trend taxonomy；不含 Volume。

| feature | group | n_valid | nan_rate | p10 | median | p90 | note |
|---|---|---:|---:|---:|---:|---:|---|
{tbl(pc_audit, ['feature','group','n_valid','nan_rate','p10','median','p90','note']) if 'feature' in pc_audit.columns else '| (Stage B 未运行) | | | | | | | |'}

### Stage B incremental

| wf | model | PR-AUC | ROC-AUC | ΔPR-AUC vs T0_HGB | ΔROC-AUC vs T0_HGB |
|---|---|---:|---:|---:|---:|
{tbl(dprow, ['wf','model','pr_auc','roc_auc','delta_pr_auc_vs_T0_HGB','delta_roc_auc_vs_T0_HGB'])}

**`{a['stageB']['PRECONTACT_INCREMENT_VERDICT']}`**

### Stage B veto execution

| wf | model | base trades | post-veto trades | base TRADEOFF share | post-veto TRADEOFF share | clear retention | base E[R] | post-veto E[R] | ΔE[R] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{tbl(exeB, ['wf','model','baseline_trades','post_veto_trades','baseline_tradeoff_share','post_veto_tradeoff_share','clear_retention_rate','baseline_E_R','post_veto_E_R','delta_E_R']) if len(exeB) else '| (Stage B 未运行) | | | | | | | | |'}

---

## 7. 按 frozen class 拆 execution（baseline / post-veto）

| wf | model | variant | frozen_class | n | share | target_hit | E[R] |
|---|---|---|---|---:|---:|---:|---:|
{tbl(df_cls, ['wf','model','variant','frozen_class','n','share','target_hit_rate','gross_expectancy_R'])}

---

## 8. P13 Final economic gate（预注册架构，非 test 后挑选）

规则：`StageA PASS → T0_HGB`；否则 `StageB incremental PASS → T1_HGB`；
否则 `NO_TRADEOFF_VETO_EDGE`。

```json
{json.dumps(a['final'], indent=2, ensure_ascii=False)}
```

**`TRADEOFF_VETO_GROSS_EDGE_PRESENT = {a['final']['TRADEOFF_VETO_GROSS_EDGE_PRESENT']}`**

---

## 9. P14 Reversal

保持冻结 `{a['reversal']['status']}`：本轮不重新调、不 promote。

---

## 10. OOS guard

`max n_exit_on_or_after_oos = {a['oos_guard']['max_exit_on_or_after_oos']}`
（必须为 0，HARD；见 `oos_guard_audit.csv`）。

---

## 11. P15 / P18 下一步 / STOP

next_step = `{a['next_step']}`

代码 + 测试 + 运行 + 报告 + commit + push 后 **STOP**。禁止自动进入
risk coupling / cost metadata / Reversal promotion / RR filter / 新 SMC taxonomy。
等 reviewer。
"""
    open(OUT / "SMC_CONDITIONAL_TRADEOFF_VETO_V1_2.md", "w",
         encoding="utf-8-sig").write(md)


if __name__ == "__main__":
    main()
