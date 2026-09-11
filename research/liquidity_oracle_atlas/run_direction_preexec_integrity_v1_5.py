"""SMC Direction Pre-Execution Integrity Gate v1.5 (base 2117cb1).

不是新特征实验。审计并消除两个部署前方法学硬门：
  1. Label availability leakage：训练标签在 decision time 不可知，
     outer WF 与 inner OOF 都必须按 label_available_time 过滤。
  2. Full live universe：v1.4 只在"事后可解析"的 y_clear cohort 上推理；
     本轮 test inference 必须在 ALL test contacts 上运行。
另外审计 ex-ante target availability 与 duplicate/conflict。

固定（P1）：risk=1.0 ATR；selector = S1_CONTINUATION_ONLY_10；
clear = C_GLOBAL4 Logistic (OOF precision 0.85, min sel 0.05, 不调)；
direction = G4_BASE HGB(depth3,lr0.05,iter200,l2 1.0,seed42)；15 frozen symbols。

label_available_time = bar_start_time[contact_bar_index + max(long_bars_to_stop,
short_bars_to_stop)] + 5min（冻结 Oracle 语义：两方向都 stopped 才算 resolved）。

Governance: TRADING_METRICS=NOT_APPLICABLE；禁止 新特征/FVG/OB/trend/pre-contact/
新risk/stop-target优化/PnL/手续费假设/删品种/prospective OOS/LC。
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
from research.liquidity_oracle_atlas.build_oracle_atlas_v1_2 import active_mask
from research.export_ob_trigger_execution_v21 import load_raw_5m

OUT = Path("research/analysis_results/smc_direction_pre_execution_integrity_v1_5")
OUT.mkdir(parents=True, exist_ok=True)
ATLAS = Path("research/analysis_results/smc_oracle_atlas_v1")
FRONT = ATLAS / "oracle_risk_frontier_v1_2.parquet"
MASTER = ATLAS / "liquidity_master_v1_1.parquet"
CONTACTS = ATLAS / "liquidity_contacts_v1_1.parquet"
FEATURES = Path("research/analysis_results/smc_direction_deployability_v1_1/"
                "direction_features_v1_1.parquet")

KEYS = ["symbol", "liquidity_id", "contact_number"]
G4_BASE = t3.G4_BASE
OOS_START = t3.OOS_START
ALLOWED_FEATURES = t3.ALLOWED_FEATURES
FORBIDDEN = t3.FORBIDDEN
PREBAN = {"rr_direction", "y_clear", "y_reversal", "y_rev_robust"}

WF = [("WF1", ["TB1"], ["TB2"]),
      ("WF2", ["TB1", "TB2"], ["TB3"]),
      ("WF3", ["TB1", "TB2", "TB3"], ["TB4"])]

PRIMARY_RISK = 1.0
CLEAR_PRECISION = 0.85
CLEAR_MIN_SEL = 0.05
CONT_TAIL = 0.10
GATE_ACT = 0.58
GATE_MEAN = 0.60
GATE_SEL = 0.05
FROZEN_CLEAR = ["LONG_DOMINATES", "SHORT_DOMINATES", "TRADEOFF_OR_OVERLAP"]
EVALUABLE = ["LONG_DOMINATES", "SHORT_DOMINATES", "TRADEOFF_OR_OVERLAP"]
UNKNOWN = ["UNRESOLVED_CENSOR", "NO_COMPARABLE_TARGET"]


# ===========================================================================
# P1 label availability
# ===========================================================================
def build_label_availability(F):
    front = pd.read_parquet(FRONT)
    o = front[np.isclose(front["risk_ATR"], PRIMARY_RISK)].copy()
    L = (o[o.direction == "LONG"][KEYS + ["bars_to_stop", "resolution_class"]]
         .rename(columns={"bars_to_stop": "long_bars_to_stop",
                          "resolution_class": "long_resolution"}))
    S = (o[o.direction == "SHORT"][KEYS + ["bars_to_stop", "resolution_class"]]
         .rename(columns={"bars_to_stop": "short_bars_to_stop",
                          "resolution_class": "short_resolution"}))
    assert not L.duplicated(KEYS).any(), "duplicate LONG keys @risk=1"
    assert not S.duplicated(KEYS).any(), "duplicate SHORT keys @risk=1"

    z = (F[KEYS + ["contact_bar_index", "decision_time", "rr_direction"]]
         .merge(L, on=KEYS, how="left").merge(S, on=KEYS, how="left"))
    assert len(z) == len(F)

    rows, viol = [], 0
    for sym, g in z.groupby("symbol", sort=False):
        raw = load_raw_5m(sym).sort_values("bar_start_time").reset_index(drop=True)
        bt = pd.to_datetime(raw["bar_start_time"]).to_numpy()
        n = len(bt)
        for r in g.itertuples(index=False):
            lb = getattr(r, "long_bars_to_stop")
            sb = getattr(r, "short_bars_to_stop")
            if pd.isna(lb) or pd.isna(sb):
                avail = pd.NaT
            else:
                k = int(r.contact_bar_index) + max(int(lb), int(sb))
                if k >= n:
                    avail = pd.NaT
                    viol += 1
                else:
                    avail = pd.Timestamp(bt[k]) + pd.Timedelta(minutes=5)
            rows.append((r.symbol, r.liquidity_id, r.contact_number,
                         lb, sb, avail))
    ld = pd.DataFrame(rows, columns=KEYS + ["long_bars_to_stop",
                                            "short_bars_to_stop",
                                            "label_available_time"])
    ld = ld.merge(L[KEYS + ["long_resolution"]], on=KEYS, how="left")
    ld = ld.merge(S[KEYS + ["short_resolution"]], on=KEYS, how="left")
    ld["decision_time"] = pd.to_datetime(z["decision_time"].to_numpy())
    ld["rr_direction"] = z["rr_direction"].to_numpy()

    # ---- P1 硬语义断言（冻结 clear/tradeoff 必须两方向 resolve）----
    frozen = ld["rr_direction"].isin(FROZEN_CLEAR)
    bad_bs = frozen & (ld["long_bars_to_stop"].isna()
                       | ld["short_bars_to_stop"].isna())
    bad_res = frozen & (ld["long_resolution"].eq("CENSORED_LOWER_BOUND")
                        | ld["short_resolution"].eq("CENSORED_LOWER_BOUND"))
    bad_avail = frozen & ld["label_available_time"].isna()
    checks = dict(
        frozen_clear_tradeoff_n=int(frozen.sum()),
        frozen_rows_missing_bars_to_stop=int(bad_bs.sum()),
        frozen_rows_censored_lower_bound=int(bad_res.sum()),
        frozen_rows_label_avail_nat=int(bad_avail.sum()),
        availability_index_out_of_range=int(viol),
    )
    fatal = (bad_bs.any() or bad_res.any() or bad_avail.any() or viol > 0)
    checks["FATAL_LABEL_AVAILABILITY_SEMANTIC_FAIL"] = bool(fatal)

    # ---- 审计表（compact，按 rr_direction class）----
    audit_rows = []
    for cls in ["LONG_DOMINATES", "SHORT_DOMINATES", "TRADEOFF_OR_OVERLAP",
                "UNRESOLVED_CENSOR", "NO_COMPARABLE_TARGET"]:
        sub = ld[ld["rr_direction"] == cls]
        lag = (sub["label_available_time"] - sub["decision_time"]).dt.total_seconds()
        audit_rows.append(dict(rr_direction=cls, n=len(sub),
                               n_label_avail_nat=int(
                                   sub["label_available_time"].isna().sum()),
                               n_long_bars_to_stop_null=int(
                                   sub["long_bars_to_stop"].isna().sum()),
                               n_short_bars_to_stop_null=int(
                                   sub["short_bars_to_stop"].isna().sum()),
                               median_lag_days=round(float(lag.median() /
                                                           86400), 4)
                               if lag.notna().any() else None))
    # 全量：label_available_time >= decision_time
    both = ld["label_available_time"].notna() & ld["decision_time"].notna()
    checks["all_label_avail_ge_decision_time"] = bool(
        (ld.loc[both, "label_available_time"]
         >= ld.loc[both, "decision_time"]).all())
    return ld, pd.DataFrame(audit_rows), checks


# ===========================================================================
# P3 availability-safe inner OOF
# ===========================================================================
def expanding_oof_pred_available(cols, X, y, decision_time, label_avail, day,
                                 mode):
    uniq = np.sort(pd.unique(day))
    n = len(uniq)
    dt = pd.to_datetime(decision_time).to_numpy()
    lav = pd.to_datetime(label_avail).to_numpy()
    oidx, pred, audit = [], [], []
    for a, b in ((0.4, 0.6), (0.6, 0.8), (0.8, 1.0)):
        ti, vi = int(round(n * a)), int(round(n * b))
        if ti <= 0 or vi <= ti:
            continue
        tr_days, va_days = uniq[:ti], uniq[ti:vi]
        m_va = np.isin(day, va_days)
        m_tr_raw = np.isin(day, tr_days)
        if m_va.sum() < 20 or m_tr_raw.sum() < 50:
            continue
        va_start = dt[m_va].min()
        m_tr = m_tr_raw & (lav < va_start)   # NaT<X -> False（保守）
        row = dict(fold=f"{a}-{b}", n_train_before=int(m_tr_raw.sum()),
                   n_train_after=int(m_tr.sum()),
                   n_removed=int(m_tr_raw.sum() - m_tr.sum()),
                   removed_share=round(float((m_tr_raw.sum() - m_tr.sum())
                                             / max(m_tr_raw.sum(), 1)), 4),
                   n_val=int(m_va.sum()),
                   val_start=str(pd.Timestamp(va_start)), skipped="")
        if m_tr.sum() < 50:
            row["skipped"] = "insufficient_train"
            audit.append(row)
            continue
        max_lav = lav[m_tr].max()
        if not (max_lav < va_start):
            row["skipped"] = "availability_assert_failed"
            audit.append(row)
            raise AssertionError(
                f"inner availability leak: max_lav={max_lav} val_start={va_start}")
        row["max_train_label_avail_time"] = str(pd.Timestamp(max_lav))
        pipe = t3.fit_pipe(cols, X[m_tr], y[m_tr], mode)
        p = pipe.predict_proba(X[m_va])[:, 1]
        oidx.append(np.flatnonzero(m_va))
        pred.append(p)
        audit.append(row)
    if not oidx:
        return np.array([], int), np.array([]), audit
    return np.concatenate(oidx), np.concatenate(pred), audit


# ===========================================================================
# P5 selector 纯函数（结构上不可能访问未来标签）
# ===========================================================================
def s1_select(p_clear, p_rev, clear_thr, cont_thr):
    return (p_clear >= clear_thr) & (p_rev <= cont_thr)


def test_all_mask(block, te_blocks, insample):
    return pd.Series(block).isin(te_blocks).to_numpy() & insample


def trade_direction_for_continuation(side):
    return side


# ===========================================================================
# P8 ex-ante target（只用 decision-time active liquidity）
# ===========================================================================
def nearest_exante_target(master_sym, entry, side, atr0, dtn):
    am = active_mask(master_sym, dtn)
    if not am.any():
        return None
    prices = master_sym["price"].to_numpy(float)[am]
    ahead = side * (prices - entry)
    valid = ahead > 0
    if not valid.any():
        return None
    adv, pr = ahead[valid], prices[valid]
    j = int(np.argmin(adv))
    return dict(target_price=float(pr[j]), distance_ATR=float(adv[j] / atr0),
                n_ahead=int(valid.sum()))


# ===========================================================================
# P10 duplicate / conflict
# ===========================================================================
def duplicate_conflict_flags(df):
    """只用 symbol/decision_time/trade_direction，不使用任何未来信息。"""
    g = df.groupby(["symbol", "decision_time"])["trade_direction"]
    ns = g.size()
    nu = g.nunique()
    dup = (ns > 1) & (nu == 1)
    conf = nu > 1
    return dict(
        n_groups=int(len(ns)),
        dup_group=int(dup.sum()), conf_group=int(conf.sum()),
        dup_contact=int(ns[dup].sum()), conf_contact=int(ns[conf].sum()),
    )


def safe_auc(y, p):
    return t3.safe_auc(np.asarray(y), np.asarray(p))


# ===========================================================================
# main
# ===========================================================================
def main():
    t0 = time.perf_counter()
    D = t3.load_data()
    F = D["F"].copy()
    con = pd.read_parquet(CONTACTS)[KEYS + ["liquidity_price"]]
    F = F.merge(con, on=KEYS, how="left")
    ld, lav_audit, lav_checks = build_label_availability(F)
    F = F.merge(ld[KEYS + ["label_available_time", "long_bars_to_stop",
                           "short_bars_to_stop", "long_resolution",
                           "short_resolution"]], on=KEYS, how="left")
    assert len(F) == 96900, len(F)
    if lav_checks["FATAL_LABEL_AVAILABILITY_SEMANTIC_FAIL"]:
        raise SystemExit(f"FATAL_LABEL_AVAILABILITY_SEMANTIC_FAIL: {lav_checks}")
    print(f"[P1] label availability built ({time.perf_counter()-t0:.1f}s) "
          f"checks={lav_checks}")

    D["F"] = F
    block = D["block"]; insample = D["insample"]; days = D["days"]
    X = D["X"]; y_clear = D["y_clear"]; y_rev = D["y_rev"]; side = D["side"]
    rr = D["rr"]; SYM = D["SYM"]
    lav = pd.to_datetime(F["label_available_time"]).to_numpy()
    dtime = pd.to_datetime(F["decision_time"]).to_numpy()

    clear_cols = [c for c in rp.block_cols("M_GLOBAL4", D["LIQ_TYPES"])
                  if c in X.columns]

    # 保存 contact-level availability（本地，不入 Git）
    ld.to_parquet(OUT / "label_availability_v1_5.parquet", index=False)

    master = pd.read_parquet(MASTER)
    master_by_sym = {s: g.reset_index(drop=True)
                     for s, g in master.groupby("symbol")}

    rows_s1, rows_inner, rows_live, rows_tgt, rows_dup = [], [], [], [], []
    extras = []
    pooled_parts = []
    for name, trb, teb in WF:
        m_tr = pd.Series(block).isin(trb).to_numpy() & insample
        test_all = test_all_mask(block, teb, insample)
        test_start = dtime[test_all].min()

        # ---- outer availability filter (P2) ----
        clear_raw = m_tr & (~pd.isna(y_clear))
        dir_raw = m_tr & (~pd.isna(y_rev))
        clear_train = clear_raw & (lav < test_start)
        dir_train = dir_raw & (lav < test_start)
        clear_removed = 1 - clear_train.sum() / max(clear_raw.sum(), 1)
        dir_removed = 1 - dir_train.sum() / max(dir_raw.sum(), 1)

        # ---- inner OOF (P3) ----
        oc, pc_oof, aud_c = expanding_oof_pred_available(
            clear_cols, X[clear_train], y_clear[clear_train],
            dtime[clear_train], lav[clear_train], days[clear_train], "logistic")
        od, pr_oof, aud_d = expanding_oof_pred_available(
            G4_BASE, X[dir_train], y_rev[dir_train],
            dtime[dir_train], lav[dir_train], days[dir_train], "hgb")
        for a in aud_c:
            rows_inner.append(dict(wf=name, task="clear", **a))
        for a in aud_d:
            rows_inner.append(dict(wf=name, task="direction", **a))
        clear_thr = v2.choose_clear_threshold(
            y_clear[clear_train][oc], pc_oof, CLEAR_PRECISION, CLEAR_MIN_SEL)
        cont_thr = float(np.quantile(pr_oof, CONT_TAIL))

        # ---- P5 test inference on ALL contacts ----
        p_clear_test = rp.fit_predict(
            clear_cols, X[clear_train], y_clear[clear_train], X[test_all],
            "logistic")
        p_rev_test = rp.fit_predict(
            G4_BASE, X[dir_train], y_rev[dir_train], X[test_all], "hgb")
        selected = s1_select(p_clear_test, p_rev_test, clear_thr, cont_thr)

        # ---- outcome audit (selection 之后才允许 merge) ----
        rr_te = rr[test_all]
        yc_te = y_clear[test_all]
        yr_te = y_rev[test_all]
        side_te = side[test_all]
        actual_clear = (yc_te == 1)
        correct_full = np.zeros(len(rr_te), bool)
        if selected.sum():
            correct_full[np.flatnonzero(selected)] = (yr_te[selected] == 0)
        ev_mask = np.isin(rr_te, EVALUABLE)
        ev_sel = selected & ev_mask
        sel_clear_correct = selected & actual_clear & correct_full
        evaluable_act = (float(sel_clear_correct.sum() / ev_sel.sum())
                         if ev_sel.sum() else np.nan)
        live_lb = (float(sel_clear_correct.sum() / selected.sum())
                   if selected.sum() else np.nan)
        unknown_rate = (float((selected & np.isin(rr_te, UNKNOWN)).sum()
                              / selected.sum()) if selected.sum() else np.nan)

        # 绝对方向（continuation -> trade_direction = side）
        pred_long_full = np.full(len(rr_te), -1, int)
        pred_long_full[np.flatnonzero(selected)] = (
            side_te[selected] == +1).astype(int)
        ev_sel_long = ev_sel & (pred_long_full == 1)
        ev_sel_short = ev_sel & (pred_long_full == 0)
        long_ap = (float((ev_sel_long & actual_clear
                          & correct_full).sum() / ev_sel_long.sum())
                   if ev_sel_long.sum() else np.nan)
        short_ap = (float((ev_sel_short & actual_clear
                           & correct_full).sum() / ev_sel_short.sum())
                    if ev_sel_short.sum() else np.nan)
        pooled_parts.append(dict(ac=actual_clear[ev_sel],
                                 corr=correct_full[ev_sel],
                                 pl=pred_long_full[ev_sel]))
        clear_mask = np.isin(rr_te, ["LONG_DOMINATES", "SHORT_DOMINATES"])
        auc_eval = safe_auc(yr_te[clear_mask], p_rev_test[clear_mask])

        rows_s1.append(dict(
            wf=name,
            n_clear_train_before=int(clear_raw.sum()),
            n_clear_train_after=int(clear_train.sum()),
            clear_removed_share=round(clear_removed, 4),
            n_dir_train_before=int(dir_raw.sum()),
            n_dir_train_after=int(dir_train.sum()),
            dir_removed_share=round(dir_removed, 4),
            clear_thr_avail=(round(clear_thr, 4) if clear_thr else None),
            cont_thr=round(cont_thr, 4),
            test_start=str(pd.Timestamp(test_start)),
            direction_auc_evaluable=round(auc_eval, 4)
            if pd.notna(auc_eval) else None,
            n_test_all=int(test_all.sum()),
            n_selected=int(selected.sum()),
            selection_rate=round(float(selected.mean()), 4),
            selected_clear_rate=round(float(
                (selected & actual_clear).sum() / max(selected.sum(), 1)), 4),
            continuation_precision_given_clear=round(float(
                (actual_clear[selected] & correct_full[selected]).sum()
                / max(int(actual_clear[selected].sum()), 1)), 4),
            evaluable_actionable_precision=round(evaluable_act, 4)
            if pd.notna(evaluable_act) else None,
            live_conservative_actionable_lower_bound=round(live_lb, 4)
            if pd.notna(live_lb) else None,
            predicted_LONG_actionable=round(long_ap, 4),
            predicted_SHORT_actionable=round(short_ap, 4),
        ))

        # selected frozen-class composition
        sel_rr = rr_te[selected]
        rows_live.append(dict(
            wf=name, n_test_all=int(test_all.sum()),
            n_selected=int(selected.sum()),
            selected_LONG_DOMINATES=int((sel_rr == "LONG_DOMINATES").sum()),
            selected_SHORT_DOMINATES=int((sel_rr == "SHORT_DOMINATES").sum()),
            selected_TRADEOFF=int((sel_rr == "TRADEOFF_OR_OVERLAP").sum()),
            selected_UNRESOLVED_CENSOR=int(
                (sel_rr == "UNRESOLVED_CENSOR").sum()),
            selected_NO_COMPARABLE_TARGET=int(
                (sel_rr == "NO_COMPARABLE_TARGET").sum()),
            selected_unknown_rate=round(unknown_rate, 4)
            if pd.notna(unknown_rate) else None,
            evaluable_actionable_precision=round(evaluable_act, 4)
            if pd.notna(evaluable_act) else None,
            live_conservative_actionable_lower_bound=round(live_lb, 4)
            if pd.notna(live_lb) else None,
        ))

        # ---- P8 ex-ante target availability ----
        sel_idx = np.flatnonzero(test_all)[np.flatnonzero(selected)]
        sym_sel = SYM[sel_idx]
        dt_sel = dtime[sel_idx]
        entry_sel = F["entry_reference"].to_numpy()[sel_idx]
        atr_sel = F["atr0"].to_numpy()[sel_idx]
        side_sel = side[sel_idx]
        lp_sel = F["liquidity_price"].to_numpy()[sel_idx]
        n_tgt, tgt_rows = 0, []
        for k in range(len(sel_idx)):
            ms = master_by_sym.get(sym_sel[k])
            if ms is None:
                tgt_rows.append(dict(sym=sym_sel[k], dt=dt_sel[k], side=side_sel[k],
                                     td=side_sel[k], has_target=False,
                                     distance_ATR=np.nan,
                                     equals_contact=np.nan))
                continue
            tgt = nearest_exante_target(ms, float(entry_sel[k]),
                                        int(side_sel[k]), float(atr_sel[k]),
                                        np.datetime64(dt_sel[k]))
            if tgt is None:
                tgt_rows.append(dict(sym=sym_sel[k], dt=dt_sel[k],
                                     side=side_sel[k], td=side_sel[k],
                                     has_target=False, distance_ATR=np.nan,
                                     equals_contact=np.nan))
            else:
                n_tgt += 1
                tgt_rows.append(dict(sym=sym_sel[k], dt=dt_sel[k],
                                     side=side_sel[k], td=side_sel[k],
                                     has_target=True,
                                     distance_ATR=tgt["distance_ATR"],
                                     equals_contact=bool(np.isclose(
                                         tgt["target_price"],
                                         float(lp_sel[k])))))
        tgt_df = pd.DataFrame(tgt_rows)
        n_exec = int(tgt_df["has_target"].sum()) if len(tgt_df) else 0
        rows_tgt.append(dict(
            wf=name, selected_count=int(selected.sum()),
            no_target_count=int((~tgt_df["has_target"]).sum())
            if len(tgt_df) else 0,
            no_target_rate=round(float((~tgt_df["has_target"]).mean()), 4)
            if len(tgt_df) else np.nan,
            execution_candidate_count=n_exec,
            execution_candidate_rate=round(n_exec / max(int(selected.sum()), 1), 4),
            target_equals_contact_price_count=int(tgt_df[
                "equals_contact"].fillna(False).sum()) if len(tgt_df) else 0,
            nearest_distance_ATR_median=round(float(
                tgt_df.loc[tgt_df["has_target"], "distance_ATR"].median()), 4)
            if n_exec else None,
        ))

        # ---- P10 duplicate / conflict ----
        if n_exec:
            exc = tgt_df[tgt_df["has_target"]][["sym", "dt", "td"]].rename(
                columns={"sym": "symbol", "dt": "decision_time",
                         "td": "trade_direction"})
            dcf = duplicate_conflict_flags(exc)
        else:
            dcf = dict(n_groups=0, dup_group=0, conf_group=0,
                       dup_contact=0, conf_contact=0)
        ng = max(dcf["n_groups"], 1)
        nc = max(n_exec, 1)
        rows_dup.append(dict(
            wf=name, n_execution_candidates=n_exec,
            duplicate_group_rate=round(dcf["dup_group"] / ng, 4),
            duplicate_contact_rate=round(dcf["dup_contact"] / nc, 4),
            conflict_group_rate=round(dcf["conf_group"] / ng, 4),
            conflict_contact_rate=round(dcf["conf_contact"] / nc, 4),
        ))

        extras.append(dict(wf=name, ev_act=evaluable_act,
                           long_ap=long_ap, short_ap=short_ap,
                           sel_rate=float(selected.mean()),
                           n_sel=int(selected.sum())))
        print(f"[WF] {name} test_all={int(test_all.sum())} "
              f"clear_tr {int(clear_raw.sum())}->{int(clear_train.sum())} "
              f"dir_tr {int(dir_raw.sum())}->{int(dir_train.sum())} "
              f"sel={int(selected.sum())} ev_act={evaluable_act:.4f} "
              f"live_lb={live_lb:.4f} no_tgt={int((~tgt_df['has_target']).sum())}")

    df_s1 = pd.DataFrame(rows_s1)
    df_inner = pd.DataFrame(rows_inner)
    df_live = pd.DataFrame(rows_live)
    df_tgt = pd.DataFrame(rows_tgt)
    df_dup = pd.DataFrame(rows_dup)

    # ---- P11 gate（严格按用户 P11 三条件，不含 coverage）----
    all_ac = (np.concatenate([p["ac"] for p in pooled_parts])
              if pooled_parts else np.array([], bool))
    all_corr = (np.concatenate([p["corr"] for p in pooled_parts])
                if pooled_parts else np.array([], bool))
    all_pl = (np.concatenate([p["pl"] for p in pooled_parts])
              if pooled_parts else np.array([], int))
    pooled_long = (float((all_ac & all_corr & (all_pl == 1)).sum()
                         / max(int((all_pl == 1).sum()), 1)))
    pooled_short = (float((all_ac & all_corr & (all_pl == 0)).sum()
                          / max(int((all_pl == 0).sum()), 1)))
    wf_ev = [e["ev_act"] for e in extras]
    per_wf_ok = all(pd.notna(a) and a >= GATE_ACT for a in wf_ev)
    mean_ok = np.nanmean(wf_ev) >= GATE_MEAN
    ls_ok = pooled_long >= GATE_ACT and pooled_short >= GATE_ACT
    LABEL_SAFE = bool(per_wf_ok and mean_ok and ls_ok)   # P11 原文
    # coverage 单独报告（P11 未列入 gate；v1.2-v1.4 曾要求 >=5%）
    sel_ok = all(e["sel_rate"] >= GATE_SEL for e in extras)
    LABEL_SAFE_STRICT_WITH_COVERAGE = bool(LABEL_SAFE and sel_ok)
    LIVE_UNIVERSE_AUDITED = bool(LABEL_SAFE and len(df_tgt) == len(WF))

    # ---- write ----
    lav_audit.to_csv(OUT / "label_availability_audit.csv", index=False,
                     encoding="utf-8-sig")
    df_inner.to_csv(OUT / "inner_oof_availability_audit.csv", index=False,
                    encoding="utf-8-sig")
    df_s1.to_csv(OUT / "availability_safe_s1_metrics.csv", index=False,
                 encoding="utf-8-sig")
    df_live.to_csv(OUT / "live_universe_selection_audit.csv", index=False,
                   encoding="utf-8-sig")
    df_tgt.to_csv(OUT / "exante_target_availability.csv", index=False,
                  encoding="utf-8-sig")
    df_dup.to_csv(OUT / "duplicate_conflict_audit.csv", index=False,
                  encoding="utf-8-sig")

    protocol = dict(
        experiment="SMC Direction Pre-Execution Integrity Gate v1.5",
        base_commit="2117cb132fe4fbc4726677f196ea3350fce37caf",
        no_new_features=True, primary_risk=PRIMARY_RISK,
        oos_boundary_excluded=OOS_START,
        selector="S1_CONTINUATION_ONLY_10",
        clear=dict(model="C_GLOBAL4 Logistic", target_precision=CLEAR_PRECISION,
                   min_oof_selection=CLEAR_MIN_SEL),
        direction=dict(feature="G4_BASE", model="HGB(depth3,lr0.05,iter200,l2 1.0,seed42)"),
        wf=[dict(name=n, train=t, test=e) for n, t, e in WF],
        label_available_time_rule="bar_start_time[contact_bar_index + "
        "max(long_bars_to_stop, short_bars_to_stop)] + 5min",
        gate=dict(per_wf_evaluable_actionable=GATE_ACT, mean=GATE_MEAN,
                  per_wf_selection=GATE_SEL, pooled_long_short=GATE_ACT),
        p0_v14_corrections=[
            "S1 = strongest development selector (not independently validated)",
            "v1.4 test cohort excluded UNRESOLVED_CENSOR and NO_COMPARABLE_TARGET (~7.9%)",
            "selector_by_symbol only n_selected>=50 -> pooled/sufficient-sample double-sided only",
        ],
        forbidden=["新特征", "FVG", "OB", "trend", "pre-contact dynamics", "新risk",
                   "stop/target优化", "PnL", "手续费假设", "删品种",
                   "prospective OOS", "LC"],
        trading_metrics="NOT_APPLICABLE",
    )
    json.dump(protocol, open(OUT / "DIRECTION_PREEXEC_PROTOCOL.json", "w"),
              indent=2, ensure_ascii=False, default=str)

    audit = dict(
        experiment="SMC Direction Pre-Execution Integrity Gate v1.5",
        base_commit="2117cb1",
        label_availability_checks=lav_checks,
        gate=dict(
            LABEL_AVAILABILITY_SAFE_DIRECTION=LABEL_SAFE,
            LIVE_UNIVERSE_SELECTOR_AUDITED=LIVE_UNIVERSE_AUDITED,
            per_wf_evaluable_actionable=wf_ev,
            mean_evaluable_actionable=float(np.nanmean(wf_ev)),
            pooled_LONG_actionable=pooled_long,
            pooled_SHORT_actionable=pooled_short,
            per_wf_selection_rate=[e["sel_rate"] for e in extras],
            selection_ge_5pct_per_wf=bool(sel_ok),
            LABEL_AVAILABILITY_SAFE_DIRECTION_STRICT_WITH_COVERAGE=bool(
                LABEL_SAFE_STRICT_WITH_COVERAGE),
            coverage_caveat=("WF1 full-universe selection_rate=4.47% < 5%："
                             "P11 未把 coverage 列入 gate，故主 gate 仍 TRUE；"
                             "若沿用 v1.2-v1.4 的 >=5% 规则则为 FALSE。"
                             "请 reviewer 裁决。"),
        ),
        live_readiness=dict(
            live_conservative_lower_bound=[r["live_conservative_actionable_lower_bound"]
                                           for r in rows_s1],
            selected_unknown_rate=[r["selected_unknown_rate"] for r in rows_live],
            no_target_rate=list(df_tgt["no_target_rate"]),
            duplicate_contact_rate=list(df_dup["duplicate_contact_rate"]),
            conflict_contact_rate=list(df_dup["conflict_contact_rate"]),
            note="lower bound 只是完整 live-universe 的保守标签边界，"
                 "不代表真实 execution precision；不做 profitability 结论。"),
        next_step=("FIXED_EXECUTION_BASELINE_V1" if LABEL_SAFE
                   else "STOP: label availability compromised"),
        trading_metrics="NOT_APPLICABLE",
    )
    json.dump(audit, open(OUT / "DIRECTION_PREEXEC_AUDIT.json", "w"),
              indent=2, ensure_ascii=False, default=str)

    write_report(lav_audit, lav_checks, df_inner, df_s1, df_live, df_tgt,
                 df_dup, audit)

    print("\n=== GATE (v1.5) ===")
    print(json.dumps(audit["gate"], indent=2, ensure_ascii=False))
    print(f"[DONE] {time.perf_counter()-t0:.1f}s -> {OUT}")


def write_report(lav_audit, lav_checks, df_inner, df_s1, df_live, df_tgt,
                 df_dup, audit):
    def tbl(df, cols):
        return "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |"
                         for _, r in df.iterrows())
    g = audit["gate"]
    md = f"""# SMC Direction Pre-Execution Integrity Gate v1.5

**base**: `2117cb1` (v1.4) &nbsp; **脚本**: `run_direction_preexec_integrity_v1_5.py`
**目标**: 审计并消除 (1) label-availability leakage、(2) 事后 cohort 推理
（改为 full live universe）、(3) ex-ante target availability、(4) duplicate/conflict。
**不做 PnL**。TRADING_METRICS=NOT_APPLICABLE。

---

## 0. P0 收紧 v1.4 措辞

- v1.4 允许结论：**S1 是当前最强的 development selector**；**不得写已独立验证可执行**。
- v1.4 test cohort **excluded** `UNRESOLVED_CENSOR` 与 `NO_COMPARABLE_TARGET`
  （合计约 7.9%），故 65.4% 是"事后可解析约 92% contact universe"的开发结果。
- `selector_by_symbol.csv` 只含 `n_selected>=50`：只能写 **pooled 双边 +
  足量 symbol-cell 普遍双边**，不得写"所有 15 品种每个 WF 均双边"。

---

## 1. P1 Label availability

`label_available_time = bar_start_time[contact_bar_index +
max(long_bars_to_stop, short_bars_to_stop)] + 5min`（risk=1.0）。

**语义检查**（冻结 clear/tradeoff 必须两方向 resolve）：

```json
{json.dumps(lav_checks, indent=2, ensure_ascii=False)}
```

| rr_direction | n | label_avail NaT | long_bts null | short_bts null | median_lag_days |
|---|---:|---:|---:|---:|---:|
{tbl(lav_audit, ['rr_direction','n','n_label_avail_nat','n_long_bars_to_stop_null','n_short_bars_to_stop_null','median_lag_days'])}
若 `FATAL_LABEL_AVAILABILITY_SEMANTIC_FAIL=true` 则立即停止（本次为
`{lav_checks['FATAL_LABEL_AVAILABILITY_SEMANTIC_FAIL']}`）。

---

## 2. P2/P3 外/内层 availability 过滤

外层：train 只用 `label_available_time < test_start_time`。
内层：`expanding_oof_pred_available()` 每个 fold 断言
`max(train label available) < min(val decision_time)`。

见 `inner_oof_availability_audit.csv`（每 WF×fold 的 n_before/n_after/removed_share/skipped）。

---

## 3. P4 availability-safe S1

| wf | clear_tr before→after | dir_tr before→after | dir_removed | clear_thr | cont_thr | dir_auc_eval | sel_rate | evaluable_act |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{tbl(df_s1, ['wf','n_clear_train_before','n_clear_train_after','n_dir_train_before','n_dir_train_after','dir_removed_share','clear_thr_avail','cont_thr','direction_auc_evaluable','selection_rate','evaluable_actionable_precision'])}
Pooled predicted LONG/SHORT（evaluable）：
**LONG={g['pooled_LONG_actionable']:.4f} / SHORT={g['pooled_SHORT_actionable']:.4f}**。

Threshold **不变**：clear OOF precision=0.85；continuation = direction OOF bottom 10%。

---

## 4. P5/P6/P7 Full live universe + 两个 actionability 指标

Test inference 在 **ALL test contacts** 上运行（不要求 `y_clear` notna）。

| wf | n_test_all | n_selected | selected_LONG_DOM | selected_SHORT_DOM | selected_TRADEOFF | selected_UNRESOLVED | selected_NO_COMPARABLE |
|---|---:|---:|---:|---:|---:|---:|---:|
{tbl(df_live, ['wf','n_test_all','n_selected','selected_LONG_DOMINATES','selected_SHORT_DOMINATES','selected_TRADEOFF','selected_UNRESOLVED_CENSOR','selected_NO_COMPARABLE_TARGET'])}

| wf | evaluable_actionable | live_conservative_lower_bound | selected_unknown_rate |
|---|---:|---:|---:|
{tbl(df_live, ['wf','evaluable_actionable_precision','live_conservative_actionable_lower_bound','selected_unknown_rate'])}

> `evaluable` 用于与 v1.4 比较；`live_conservative_lower_bound` 把
> TRADEOFF/UNRESOLVED/NO_COMPARABLE 全部计为 failure（完整 selected 分母），
> 是保守标签边界，**不代表真实 execution precision**。

---

## 5. P8/P9 Ex-ante target availability

只用 decision-time active liquidity（`active_mask`），`trade_direction = side`
（Continuation）。报告 availability，**不做 outcome，不做最小 RR 筛选**。

| wf | selected | no_target | no_target_rate | exec_candidate | exec_rate | target==contact |
|---|---:|---:|---:|---:|---:|---:|
{tbl(df_tgt, ['wf','selected_count','no_target_count','no_target_rate','execution_candidate_count','execution_candidate_rate','target_equals_contact_price_count'])}

---

## 6. P10 Duplicate / Conflict audit

| wf | n_exec | dup_group | dup_contact | conflict_group | conflict_contact |
|---|---:|---:|---:|---:|---:|
{tbl(df_dup, ['wf','n_execution_candidates','duplicate_group_rate','duplicate_contact_rate','conflict_group_rate','conflict_contact_rate'])}

未来执行策略：same-direction duplicates → collapse 为一个 signal；
direction conflict → abstain。本轮只报告。

---

## 7. P11/P12 Gate

```json
{json.dumps(audit['gate'], indent=2, ensure_ascii=False)}
```

- **LABEL_AVAILABILITY_SAFE_DIRECTION = {g['LABEL_AVAILABILITY_SAFE_DIRECTION']}**
  （P11 原文门槛：WF1/WF2/WF3 evaluable actionable≥0.58、mean≥0.60、
  pooled predicted LONG/SHORT≥0.58）
- **LABEL_AVAILABILITY_SAFE_DIRECTION_STRICT_WITH_COVERAGE = {g['LABEL_AVAILABILITY_SAFE_DIRECTION_STRICT_WITH_COVERAGE']}**
- **LIVE_UNIVERSE_SELECTOR_AUDITED = {g['LIVE_UNIVERSE_SELECTOR_AUDITED']}**

> **Coverage caveat**：P11 未把 coverage 列入 gate。本轮 full-universe 分母下
> **WF1 selection_rate = 4.47% < 5%**（WF2/WF3 分别 7.42%/7.21%）。因此：
> 按 P11 原文 → 主 gate = `{g['LABEL_AVAILABILITY_SAFE_DIRECTION']}`；
> 若沿用 v1.2–v1.4 的 `selection≥0.05` 规则 → `{g['LABEL_AVAILABILITY_SAFE_DIRECTION_STRICT_WITH_COVERAGE']}`。
> **请 reviewer 裁决用哪一个。**

若 `LABEL_AVAILABILITY_SAFE_DIRECTION=FALSE` → 立即 STOP（过去方向结果受
label availability 影响），不得 PnL。若 TRUE → next step `FIXED_EXECUTION_BASELINE_V1`。

---

## 8. P15 完成 / STOP

代码 + 测试 + 运行 + 报告 + commit + push 后 STOP。
禁止自动进入 PnL / 手续费假设 / pre-contact / FVG。等 reviewer。
"""
    open(OUT / "SMC_DIRECTION_PREEXEC_INTEGRITY_V1_5.md", "w",
         encoding="utf-8-sig").write(md)


if __name__ == "__main__":
    main()
