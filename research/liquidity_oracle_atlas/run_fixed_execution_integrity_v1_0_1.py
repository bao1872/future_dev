"""SMC Fixed Execution Baseline v1.0.1 — Integrity Repair + Geometry Attribution.

base 4c37cb8. 这不是 target/stop 优化实验。

修复：
  P1/P2 持仓路径硬截 prospective OOS（v1.0 只截 discontinuity / raw data end）。
审计：
  P3   OOS path bug 的数值影响（old vs new）。
  P4   selected-contact attack boundary vs ALL-contact attack boundary。
  P5   TARGET_SELECTED_BOUNDARY vs TARGET_ALL_ATTACKED_BOUNDARY（descriptive）。
  P6   重算固定 execution（唯一变化 = OOS 硬截）。
  P7   target_R_exec 固定经济分桶（非 quantile）。
  P8   frozen direction label × execution outcome 归因（仅 post-outcome）。
  P9/P10 entry degradation / same-bar attribution。
  P11  Reversal robustness（保持 SECONDARY_DIAGNOSTIC）。

不改变：selector threshold / stop / risk / target policy / 最低RR筛选 / symbol。
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

import research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 as m0
import research.liquidity_oracle_atlas.run_direction_preexec_integrity_v1_5 as p5

OUT = Path("research/analysis_results/smc_fixed_execution_integrity_v1_0_1")
OUT.mkdir(parents=True, exist_ok=True)

OOS_START = m0.OOS_START
WF = m0.WF
GEO_BINS = [-np.inf, 0.50, 0.75, 1.00, 1.50, np.inf]
GEO_LABELS = ["<0.50", "0.50-0.75", "0.75-1.00", "1.00-1.50", ">=1.50"]
CLEAR = ["LONG_DOMINATES", "SHORT_DOMINATES"]


# ===========================================================================
# P1 OOS cutoff index
# ===========================================================================
def first_oos_bar_index(bars, oos_start=OOS_START):
    day = pd.to_datetime(bars["day"])
    idx = np.flatnonzero(day >= pd.Timestamp(oos_start))
    return int(idx[0]) if len(idx) else int(bars["n"])


def add_oos_end(bars_by_sym, oos_start=OOS_START):
    for s, b in bars_by_sym.items():
        b["oos_end"] = first_oos_bar_index(b, oos_start)
    return bars_by_sym


# ===========================================================================
# P2/P4 collapse（带 all-contact boundary）
# ===========================================================================
def collapse_signals(sub, sel, setup, all_boundary):
    """all_boundary: dict (symbol, decision_time, side) -> attacked_boundary(全部contact)。"""
    d = sub[sel].copy()
    d["direction"] = (d["side"] if setup == "CONT" else -d["side"])
    out, n_conflict = [], 0
    for (sym, dt), g in d.groupby(["symbol", "decision_time"], sort=False):
        if g["direction"].nunique() > 1:
            n_conflict += 1
            continue
        if g["entry_reference"].nunique() != 1 or g["atr0"].nunique() != 1 \
                or g["contact_bar_index"].nunique() != 1:
            raise SystemExit(f"FATAL_EXECUTION_GROUP_MISMATCH {sym} {dt}")
        direction = int(g["direction"].iloc[0])
        prices = g["liquidity_price"].to_numpy(float)
        j = int(np.argmax(direction * prices))
        attack = float(prices[j])
        attack_rr = g["rr_direction"].iloc[j]
        if setup == "CONT":
            all_attack = float(all_boundary.get((sym, dt, direction), attack))
            gap = direction * (all_attack - attack) / float(g["atr0"].iloc[0])
        else:
            all_attack, gap = np.nan, np.nan
        out.append(dict(
            symbol=sym, decision_time=dt, direction=direction,
            side=int(g["side"].iloc[0]),
            decision_close=float(g["entry_reference"].iloc[0]),
            atr0=float(g["atr0"].iloc[0]),
            contact_bar_index=int(g["contact_bar_index"].iloc[0]),
            attack=attack, attack_rr=attack_rr,
            all_attack=all_attack, boundary_gap_ATR=float(gap),
            n_contacts=len(g)))
    return pd.DataFrame(out), n_conflict


# ===========================================================================
# P5 targets（selected boundary + all-attack boundary + OLD）
# ===========================================================================
def attach_targets_v101(sig, master_by_sym, setup):
    rows = []
    for r in sig.itertuples(index=False):
        ms = master_by_sym.get(r.symbol)
        dtn = np.datetime64(pd.Timestamp(r.decision_time))
        pr = ms["price"].to_numpy(float)[m0.active_mask(ms, dtn)]
        if setup == "CONT":
            new_t = m0.continuation_target(pr, r.decision_close, r.direction,
                                           r.attack)
            all_t = m0.continuation_target(pr, r.decision_close, r.direction,
                                           r.all_attack)
            old_t = m0.nearest_ahead(pr, r.decision_close, r.direction)
            no_tgt = "NO_NEXT_LIQUIDITY_TARGET"
        else:
            new_t = m0.nearest_ahead(pr, r.decision_close, r.direction)
            all_t = old_t = None
            no_tgt = "NO_OPPOSING_TARGET"
        dd = lambda t: (abs(t - r.decision_close) / r.atr0
                        if t is not None else np.nan)
        rows.append(dict(
            symbol=r.symbol, decision_time=r.decision_time,
            direction=r.direction, side=r.side,
            decision_close=r.decision_close, atr0=r.atr0,
            contact_bar_index=r.contact_bar_index, attack=r.attack,
            attack_rr=r.attack_rr, all_attack=r.all_attack,
            boundary_gap_ATR=r.boundary_gap_ATR, n_contacts=r.n_contacts,
            target_price=new_t, target_price_all_boundary=all_t,
            old_target_price=old_t,
            target_dist_ATR=dd(new_t),
            all_target_dist_ATR=dd(all_t),
            old_target_dist_ATR=dd(old_t),
            target_equals_attack=bool(new_t is not None
                                      and np.isclose(new_t, r.attack)),
            skip_reason=("" if new_t is not None else no_tgt)))
    return pd.DataFrame(rows)


# ===========================================================================
# P2 repaired execution（OOS 硬截）
# ===========================================================================
def run_execution_repaired(sig, bars_by_sym):
    cand = sig[sig["skip_reason"] == ""].sort_values(["decision_time", "symbol"])
    n_after_target = int((sig["skip_reason"] == "").sum())
    open_until = {}
    trades, n_gap, n_skip_open = [], 0, 0
    for r in cand.itertuples(index=False):
        bars = bars_by_sym[r.symbol]
        n, oos_end = bars["n"], bars["oos_end"]
        ebar = m0.entry_bar_for(bars, r.contact_bar_index)
        if ebar is not None and ebar >= oos_end:
            ebar = None
            skip = "PROSPECTIVE_OOS_ENTRY"
        elif ebar is None:
            skip = "DISCONTINUITY_BEFORE_ENTRY"
        else:
            skip = None
        if skip:
            n_gap += 1
            trades.append(dict(symbol=r.symbol, decision_time=r.decision_time,
                               direction=r.direction, side=r.side,
                               attack_rr=r.attack_rr, executed=False,
                               skip_reason=skip, realized_R=np.nan))
            continue
        assert ebar < oos_end, "PROSPECTIVE_OOS_ENTRY leaked"
        entry_px = float(bars["o"][ebar])
        stop_px = m0.stop_price_for(r.decision_close, r.direction, r.atr0)
        target_px = float(r.target_price)
        risk_px = float(r.direction * (entry_px - stop_px))
        g = m0.entry_gate_reason(r.direction, entry_px, stop_px, target_px)
        if g:
            n_gap += 1
            trades.append(dict(symbol=r.symbol, decision_time=r.decision_time,
                               direction=r.direction, side=r.side,
                               attack_rr=r.attack_rr, executed=False,
                               skip_reason=g, realized_R=np.nan))
            continue
        if r.symbol in open_until and ebar <= open_until[r.symbol]:
            n_skip_open += 1
            trades.append(dict(symbol=r.symbol, decision_time=r.decision_time,
                               direction=r.direction, side=r.side,
                               attack_rr=r.attack_rr, executed=False,
                               skip_reason="SKIPPED_POSITION_ALREADY_OPEN",
                               realized_R=np.nan))
            continue
        end_disc = m0.path_end(bars, ebar)
        end = min(end_disc, oos_end)
        res = m0.execute_path(bars, ebar, end, r.direction, stop_px, target_px,
                              stop_first=True)
        res_opt = m0.execute_path(bars, ebar, end, r.direction, stop_px,
                                  target_px, stop_first=False)
        if res is None:
            if end < n and end == oos_end and oos_end <= end_disc:
                outcome = "OOS_CUTOFF_EXIT"
            elif end < n:
                outcome = "ROLL_EXIT"
            else:
                outcome = "DATA_END_EXIT"
            exit_px, ebar_exit, amb = float(bars["c"][end - 1]), int(end - 1), False
        else:
            outcome, exit_px = res["outcome"], res["exit_px"]
            ebar_exit, amb = res["exit_bar"], res["same_bar_ambiguous"]
        risk_exec = abs(entry_px - stop_px)
        realized_R = r.direction * (exit_px - entry_px) / risk_exec
        target_R = r.direction * (target_px - entry_px) / risk_exec
        ideal_R = r.direction * (exit_px - r.decision_close) / r.atr0
        opt_R = (r.direction * (res_opt["exit_px"] - entry_px) / risk_exec
                 if res_opt is not None else np.nan)
        open_until[r.symbol] = int(ebar_exit)
        trades.append(dict(
            symbol=r.symbol, decision_time=r.decision_time,
            direction=r.direction, side=r.side, attack_rr=r.attack_rr,
            executed=True, skip_reason="",
            entry_time=pd.Timestamp(bars["t"][ebar]),
            exit_time=pd.Timestamp(bars["t"][ebar_exit]),
            entry_day=pd.Timestamp(bars["day"][ebar]),
            exit_day=pd.Timestamp(bars["day"][ebar_exit]),
            entry_bar=int(ebar), exit_bar=int(ebar_exit),
            path_end_bar=int(end),
            entry_px=entry_px, stop_px=stop_px, target_px=target_px,
            risk_px=risk_px,
            target_R_exec=float(target_R),
            outcome=outcome, exit_px=float(exit_px),
            same_bar_ambiguous=bool(amb),
            realized_R=float(realized_R),
            ideal_close_fill_R=float(ideal_R),
            target_first_R=float(opt_R) if res_opt is not None else np.nan,
            oos_bar_read=bool(ebar_exit >= oos_end),
        ))
    return pd.DataFrame(trades), n_after_target, n_gap, n_skip_open


# ===========================================================================
# P3 old-logic OOS audit（v1.0 path_end 不含 OOS）
# ===========================================================================
def old_path_audit(sig, bars_by_sym):
    cand = sig[sig["skip_reason"] == ""].sort_values(["decision_time", "symbol"])
    open_until = {}
    n_exit_oos, n_touch, n_exec = 0, 0, 0
    for r in cand.itertuples(index=False):
        bars = bars_by_sym[r.symbol]
        n, oos_end = bars["n"], bars["oos_end"]
        ebar = m0.entry_bar_for(bars, r.contact_bar_index)
        if ebar is None or ebar >= oos_end:
            continue
        entry_px = float(bars["o"][ebar])
        stop_px = m0.stop_price_for(r.decision_close, r.direction, r.atr0)
        target_px = float(r.target_price)
        if m0.entry_gate_reason(r.direction, entry_px, stop_px, target_px):
            continue
        if r.symbol in open_until and ebar <= open_until[r.symbol]:
            continue
        end_disc = m0.path_end(bars, ebar)   # v1.0 旧逻辑：不截 OOS
        res = m0.execute_path(bars, ebar, end_disc, r.direction, stop_px,
                              target_px, stop_first=True)
        if res is None:
            exit_bar = int(end_disc - 1)
        else:
            exit_bar = int(res["exit_bar"])
        open_until[r.symbol] = exit_bar
        n_exec += 1
        if end_disc > oos_end:
            n_touch += 1
        if exit_bar >= oos_end:
            n_exit_oos += 1
    return n_exec, n_touch, n_exit_oos


# ===========================================================================
# metrics helpers
# ===========================================================================
def geo_buckets(tr, wf, setup):
    ex = tr[tr["executed"]]
    rows = []
    b = pd.cut(ex["target_R_exec"], bins=GEO_BINS, labels=GEO_LABELS,
               right=False)
    for lab in GEO_LABELS:
        m = b == lab
        sub = ex[m]
        n = len(sub)
        if n == 0:
            rows.append(dict(wf=wf, setup=setup, bucket=lab, n=0,
                             share=0.0, target_hit_rate=np.nan,
                             mean_win_R=np.nan, mean_loss_R=np.nan,
                             payoff=np.nan, gross_expectancy_R=np.nan))
            continue
        rr = sub["realized_R"].to_numpy(float)
        w, l = rr[rr > 0], rr[rr < 0]
        mw = float(w.mean()) if len(w) else np.nan
        ml = float(l.mean()) if len(l) else np.nan
        rows.append(dict(wf=wf, setup=setup, bucket=lab, n=n,
                         share=round(n / len(ex), 4),
                         target_hit_rate=round(float(
                             (sub["outcome"] == "TARGET").mean()), 4),
                         mean_win_R=round(mw, 4) if pd.notna(mw) else np.nan,
                         mean_loss_R=round(ml, 4) if pd.notna(ml) else np.nan,
                         payoff=round(mw / abs(ml), 4)
                         if (pd.notna(mw) and pd.notna(ml) and ml != 0) else np.nan,
                         gross_expectancy_R=round(float(rr.mean()), 4)))
    return rows


def direction_attribution(tr, wf, setup):
    ex = tr[tr["executed"]].copy()
    # trade direction(+1=LONG / -1=SHORT) 决定绝对方向；REV 的 direction = -side
    pred = np.where(ex["direction"] == +1, "LONG_DOMINATES", "SHORT_DOMINATES")
    lab = ex["attack_rr"].to_numpy()
    lab = [x if x in CLEAR else None for x in lab]
    ok = np.array([x is not None for x in lab])
    correct = np.array([(lab[i] == pred[i]) if ok[i] else False
                        for i in range(len(lab))])
    hit = (ex["outcome"] == "TARGET").to_numpy()
    rr = ex["realized_R"].to_numpy(float)
    c, w_ = correct & ok, (~correct) & ok
    return dict(
        wf=wf, setup=setup,
        n_executed=int(len(ex)), n_labeled=int(ok.sum()),
        A_direction_correct_target_hit=int((c & hit).sum()),
        B_direction_correct_loss=int((c & ~hit).sum()),
        C_direction_wrong_target_hit=int((w_ & hit).sum()),
        D_direction_wrong_loss=int((w_ & ~hit).sum()),
        P_hit_given_direction_correct=round(float(hit[c].mean()), 4)
        if c.sum() else np.nan,
        P_hit_given_direction_wrong=round(float(hit[w_].mean()), 4)
        if w_.sum() else np.nan,
        E_R_given_direction_correct=round(float(rr[c].mean()), 4)
        if c.sum() else np.nan,
        E_R_given_direction_wrong=round(float(rr[w_].mean()), 4)
        if w_.sum() else np.nan,
    )


def breakeven_hit_rate(row):
    mw, ml = row.get("mean_win_R"), row.get("mean_loss_R")
    if pd.isna(mw) or pd.isna(ml) or (mw + abs(ml)) == 0:
        return np.nan
    return abs(ml) / (mw + abs(ml))


# ===========================================================================
# main
# ===========================================================================
def main():
    t0 = time.perf_counter()
    D, master_by_sym, bars_by_sym = m0.load_env()
    add_oos_end(bars_by_sym)
    print(f"[ENV] loaded + oos_end ({time.perf_counter()-t0:.1f}s) "
          f"oos={OOS_START}")
    F = D["F"]; block = D["block"]; insample = D["insample"]
    clear_cols = [c for c in m0.rp.block_cols("M_GLOBAL4", D["LIQ_TYPES"])
                  if c in D["X"].columns]

    cont_rows, rev_rows, oos_rows = [], [], []
    geo_rows, attr_rows = [], []
    funnel, bound_rows, tsem_rows = [], [], []
    entry_rows, samebar_rows = [], []
    all_cont, all_rev = [], []
    for name, trb, teb in WF:
        w = m0.build_wf(name, trb, teb, D, clear_cols)
        te = w["test_all"]
        sub = pd.DataFrame(dict(
            symbol=D["SYM"][te],
            decision_time=pd.to_datetime(F["decision_time"].to_numpy())[te],
            side=D["side"][te],
            entry_reference=F["entry_reference"].to_numpy()[te],
            atr0=F["atr0"].to_numpy()[te],
            contact_bar_index=F["contact_bar_index"].to_numpy()[te],
            liquidity_price=F["liquidity_price"].to_numpy()[te],
            rr_direction=D["rr"][te],
        ))
        all_boundary = {}
        for (sym, dt, sd), g in sub.groupby(["symbol", "decision_time", "side"]):
            all_boundary[(sym, dt, int(sd))] = m0.attacked_boundary(
                g["liquidity_price"].to_numpy(float), int(sd))

        sel_cont = m0.p5.s1_select(w["p_clear"], w["p_rev"], w["clear_thr"],
                                   w["cont_thr"])
        sel_rev = (w["p_clear"] >= w["clear_thr"]) & (w["p_rev"] >= w["rev_thr"])

        for setup, sel in (("CONT", sel_cont), ("REV", sel_rev)):
            sig, n_conflict = collapse_signals(sub, sel, setup, all_boundary)
            if not len(sig):
                continue
            st = attach_targets_v101(sig, master_by_sym, setup)
            if setup == "CONT":
                g = st["boundary_gap_ATR"].dropna()
                bound_rows.append(dict(
                    wf=name, n_signals=len(st),
                    same_boundary_rate=round(float((g <= 1e-12).mean()), 4),
                    all_boundary_further_rate=round(float((g > 1e-12).mean()), 4),
                    median_gap_ATR=round(float(g.median()), 4),
                    p75_gap_ATR=round(float(g.quantile(.75)), 4),
                    p90_gap_ATR=round(float(g.quantile(.90)), 4)))
                assert (g >= -1e-9).all(), "HARD ASSERTION: boundary_gap_ATR < 0"
                for lab, col in (("TARGET_SELECTED_BOUNDARY", "target_dist_ATR"),
                                 ("TARGET_ALL_ATTACKED_BOUNDARY",
                                  "all_target_dist_ATR"),
                                 ("OLD_ENTRY_NEAREST", "old_target_dist_ATR")):
                    s = st[col]
                    tsem_rows.append(dict(
                        wf=name, definition=lab,
                        no_target_rate=round(float(s.isna().mean()), 4),
                        median_ATR=round(float(s.median()), 4),
                        p25_ATR=round(float(s.quantile(.25)), 4),
                        p75_ATR=round(float(s.quantile(.75)), 4),
                        p90_ATR=round(float(s.quantile(.90)), 4)))

            tr, n_after_tgt, n_gap, n_skip = run_execution_repaired(
                st, bars_by_sym)
            n_old_exec, n_old_touch, n_old_exit = old_path_audit(
                st, bars_by_sym)
            ex = tr[tr["executed"]]
            oos_rows.append(dict(
                wf=name, setup=setup,
                n_old_executed=n_old_exec,
                n_old_trade_paths_touch_oos=n_old_touch,
                n_old_trades_exit_on_or_after_oos=n_old_exit,
                n_new_executed=int(len(ex)),
                n_new_trade_paths_touch_oos=int(sum(
                    1 for _, q in ex.iterrows()
                    if q["path_end_bar"]
                    > bars_by_sym[q["symbol"]]["oos_end"])),
                n_new_trades_exit_on_or_after_oos=int(
                    (ex["exit_day"] >= pd.Timestamp(OOS_START)).sum()),
            ))
            n_days = int(pd.Series(
                pd.to_datetime(F["trading_day"]).to_numpy()[te]).nunique())
            m = m0.trade_metrics(tr, n_days)
            funnel.append(dict(
                wf=name, setup=setup,
                n_raw_selected_contacts=int(sel.sum()),
                n_collapsed_signals=len(sig),
                n_conflict_abstained=n_conflict,
                n_after_target_gate=n_after_tgt,
                n_after_entry_gap_gate=int(len(ex) + n_skip),
                n_executed_trades=int(len(ex)),
                skip_open_position=n_skip))
            row = dict(wf=name, **m)
            geo_rows.extend(geo_buckets(tr, name, setup))
            attr_rows.append(direction_attribution(tr, name, setup))
            entry_rows.append(dict(
                wf=name, setup=setup,
                mean_next_open_R=round(float(ex["realized_R"].mean()), 4),
                mean_ideal_close_R=round(float(ex["ideal_close_fill_R"].mean()), 4),
                delta=round(float(ex["realized_R"].mean()
                                  - ex["ideal_close_fill_R"].mean()), 4)))
            samebar_rows.append(dict(
                wf=name, setup=setup,
                same_bar_ambiguous_rate=float(m["same_bar_ambiguous_rate"]),
                stop_first_expectancy_R=m["gross_expectancy_R"],
                target_first_expectancy_R=m["gross_expectancy_R_target_first"],
                delta=round(float(m["gross_expectancy_R_target_first"]
                                  - m["gross_expectancy_R"]), 4)
                if pd.notna(m["gross_expectancy_R_target_first"]) else np.nan))
            if setup == "CONT":
                cont_rows.append(row); all_cont.append(ex)
            else:
                rev_rows.append(row); all_rev.append(ex)
        print(f"[{name}] cont_expR={cont_rows[-1]['gross_expectancy_R']} "
              f"rev_expR={rev_rows[-1]['gross_expectancy_R']}")

    df_cont = pd.DataFrame(cont_rows)
    df_rev = pd.DataFrame(rev_rows)
    df_oos = pd.DataFrame(oos_rows)
    df_geo = pd.DataFrame(geo_rows)
    df_attr = pd.DataFrame(attr_rows)
    df_ent = pd.DataFrame(entry_rows)
    df_sb = pd.DataFrame(samebar_rows)
    df_bnd = pd.DataFrame(bound_rows)
    df_tsem = pd.DataFrame(tsem_rows)
    df_fun = pd.DataFrame(funnel)
    tr_all = pd.concat(all_cont, ignore_index=True)
    tr_rev = pd.concat(all_rev, ignore_index=True)

    # ---------- P3 HARD ASSERTION ----------
    n_new_oos = int(df_oos["n_new_trades_exit_on_or_after_oos"].sum())
    n_old_oos = int(df_oos["n_old_trades_exit_on_or_after_oos"].sum())
    assert n_new_oos == 0, "HARD ASSERTION: repaired trades still exit in OOS"
    oos_status = ("OOS_PATH_BUG_NO_NUMERICAL_IMPACT" if n_old_oos == 0
                  else "OOS_PATH_BUG_NUMERICAL_IMPACT")

    # ---------- P11 reversal robustness ----------
    rev_rb = []
    for _, r in df_rev.iterrows():
        be = breakeven_hit_rate(r)
        rev_rb.append(dict(
            wf=r["wf"], trades=r["n_executed_trades"],
            actual_hit_rate=r["target_hit_rate"],
            breakeven_hit_rate=round(be, 4) if pd.notna(be) else np.nan,
            hit_minus_breakeven=round(r["target_hit_rate"] - be, 4)
            if pd.notna(be) else np.nan,
            gross_expectancy_R=r["gross_expectancy_R"]))
    df_revrb = pd.DataFrame(rev_rb)
    rev_pooled = round(float(tr_rev["realized_R"].mean()), 4)
    rev_3of3 = bool((df_revrb["hit_minus_breakeven"] > 0).all())

    # ---------- P13 verdicts ----------
    exps = list(df_cont["gross_expectancy_R"])
    pooled = float(tr_all["realized_R"].mean())
    not_3of3_positive = not all(pd.notna(e) and e > 0 for e in exps)
    EXEC_VALID_NEGATIVE = bool(not_3of3_positive)

    entry_small = bool((df_ent[df_ent.setup == "CONT"]["delta"].abs() < 0.02).all())
    sb_not_fixed = bool(not all(pd.notna(x) and x > 0 for x in
                                df_sb[df_sb.setup == "CONT"][
                                    "target_first_expectancy_R"]))
    attr_c = df_attr[df_attr.setup == "CONT"]
    dir_correct_still_loss = bool(
        (attr_c["E_R_given_direction_correct"] <= 0).sum() >= 2)
    geo_c = df_geo[(df_geo.setup == "CONT") & (df_geo.n > 0)]
    hi = geo_c[geo_c.bucket.isin([">=1.50", "1.00-1.50"])]["gross_expectancy_R"]
    lo = geo_c[geo_c.bucket.isin(["<0.50", "0.50-0.75"])]["gross_expectancy_R"]
    geo_structure = bool(len(hi) and len(lo)
                         and float(hi.max()) > float(lo.max()))
    TG_BOTTLENECK = bool(entry_small and sb_not_fixed
                         and dir_correct_still_loss and geo_structure)

    # ---------- write ----------
    df_oos.to_csv(OUT / "oos_path_audit.csv", index=False, encoding="utf-8-sig")
    df_bnd.to_csv(OUT / "attack_boundary_audit.csv", index=False,
                  encoding="utf-8-sig")
    df_tsem.to_csv(OUT / "target_boundary_semantics.csv", index=False,
                   encoding="utf-8-sig")
    df_cont.to_csv(OUT / "execution_metrics_repaired.csv", index=False,
                   encoding="utf-8-sig")
    df_rev.to_csv(OUT / "reversal_metrics_repaired.csv", index=False,
                  encoding="utf-8-sig")
    df_geo.to_csv(OUT / "execution_geometry_buckets.csv", index=False,
                  encoding="utf-8-sig")
    df_attr.to_csv(OUT / "direction_execution_attribution.csv", index=False,
                   encoding="utf-8-sig")
    df_ent.to_csv(OUT / "entry_degradation_audit.csv", index=False,
                  encoding="utf-8-sig")
    df_sb.to_csv(OUT / "same_bar_attribution.csv", index=False,
                 encoding="utf-8-sig")
    df_fun.to_csv(OUT / "execution_funnel_repaired.csv", index=False,
                  encoding="utf-8-sig")
    df_revrb.to_csv(OUT / "reversal_robustness.csv", index=False,
                    encoding="utf-8-sig")
    pd.concat([tr_all.assign(setup="CONT"),
               tr_rev.assign(setup="REV")], ignore_index=True).to_parquet(
        OUT / "execution_trade_log_repaired.parquet", index=False)

    protocol = dict(
        experiment="SMC Fixed Execution Baseline v1.0.1 Integrity Repair "
                   "+ Geometry Attribution",
        base_commit="4c37cb8de3cad517e1d32aa1367310380c8b82bc",
        v1_0_status="PROVISIONAL_PENDING_OOS_PATH_AUDIT",
        oos_start=OOS_START,
        repairs=["P1/P2 execution path hard cutoff at OOS_START",
                 "P3 OOS hard audit (old vs new)"],
        unchanged=["selector threshold", "stop", "risk", "target policy",
                   "no min-RR filter", "no symbol filtering"],
        geometry_bins=dict(bins=[float(x) if np.isfinite(x) else str(x)
                                 for x in GEO_BINS], labels=GEO_LABELS,
                           note="fixed economic bins, NOT test quantiles"),
        forbidden=["调threshold", "调stop", "改risk", "改target policy",
                   "最低RR筛选", "删symbol", "新feature", "FVG/OB/trend",
                   "手续费假设", "prospective OOS"],
        cost="COST_METADATA_DEFERRED_UNTIL_GROSS_EDGE",
    )
    json.dump(protocol, open(OUT / "EXECUTION_INTEGRITY_PROTOCOL.json", "w"),
              indent=2, ensure_ascii=False, default=str)

    audit = dict(
        experiment="SMC Fixed Execution Baseline v1.0.1",
        base_commit="4c37cb8",
        oos=dict(status=oos_status,
                 n_old_trades_exit_on_or_after_oos=n_old_oos,
                 n_new_trades_exit_on_or_after_oos=n_new_oos,
                 new_must_be_zero=True),
        continuation=dict(per_wf_gross_expectancy_R=exps,
                          pooled_gross_expectancy_R=round(pooled, 4)),
        reversal=dict(pooled_gross_expectancy_R=rev_pooled,
                      three_of_three_above_breakeven=rev_3of3,
                      verdict="SECONDARY_DIAGNOSTIC (not promoted)"),
        verdicts=dict(
            EXECUTION_BASELINE_VALID_NEGATIVE=EXEC_VALID_NEGATIVE,
            TARGET_GEOMETRY_PRIMARY_BOTTLENECK=TG_BOTTLENECK,
            verdict=("TARGET_GEOMETRY_PRIMARY_BOTTLENECK" if TG_BOTTLENECK
                     else "NO_SINGLE_EXECUTION_BOTTLENECK"),
            components=dict(entry_degradation_small=entry_small,
                            samebar_optimistic_not_fixed=sb_not_fixed,
                            direction_correct_still_loss=dir_correct_still_loss,
                            geometry_structure_present=geo_structure),
            note="VALID_NEGATIVE 不等同于 DIRECTION_EDGE_FALSE：只证明现有"
                 "方向 edge 未被当前固定 execution mapping 转化为正 expectancy。"),
        next_step=("MINIMUM_EX_ANTE_RR_1_TO_1_GATE" if TG_BOTTLENECK
                   else "RISK_COUPLED_EXECUTION (0.5/1.0/2.0, own frozen "
                        "direction label & model per risk)"),
        cost="COST_METADATA_DEFERRED_UNTIL_GROSS_EDGE",
    )
    json.dump(audit, open(OUT / "EXECUTION_INTEGRITY_AUDIT.json", "w"),
              indent=2, ensure_ascii=False, default=str)

    write_report(df_oos, df_bnd, df_tsem, df_cont, df_rev, df_geo, df_attr,
                 df_ent, df_sb, df_revrb, audit)
    print("\n=== VERDICTS ===")
    print(json.dumps(audit["verdicts"], indent=2, ensure_ascii=False))
    print(f"OOS: {oos_status} old={n_old_oos} new={n_new_oos}")
    print(f"[DONE] {time.perf_counter()-t0:.1f}s -> {OUT}")


def write_report(df_oos, df_bnd, df_tsem, df_cont, df_rev, df_geo, df_attr,
                 df_ent, df_sb, df_revrb, audit):
    def tbl(df, cols):
        return "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |"
                         for _, r in df.iterrows())
    v = audit["verdicts"]
    md = f"""# SMC Fixed Execution Baseline v1.0.1 — Integrity Repair + Geometry Attribution

**base**: `4c37cb8` (v1.0) &nbsp; **脚本**: `run_fixed_execution_integrity_v1_0_1.py`

> v1.0 状态：**`PROVISIONAL_PENDING_OOS_PATH_AUDIT`**。
> v1.0 的持仓路径没有截断 `OOS_START={m0.OOS_START}`；本轮修复 + 审计。

本轮 **不优化** target / stop / risk / threshold / symbol。

---

## 1. P3 OOS path audit

| wf | setup | old_exec | old_paths_touch_oos | old_exit>=OOS | new_exec | new_exit>=OOS |
|---|---|---:|---:|---:|---:|---:|
{tbl(df_oos, ['wf','setup','n_old_executed','n_old_trade_paths_touch_oos','n_old_trades_exit_on_or_after_oos','n_new_executed','n_new_trades_exit_on_or_after_oos'])}

**状态：`{audit['oos']['status']}`**（old={audit['oos']['n_old_trades_exit_on_or_after_oos']}，
new={audit['oos']['n_new_trades_exit_on_or_after_oos']}，new 必须为 0，HARD ASSERTION PASS）。

---

## 2. P6 Repaired execution（唯一变化 = OOS 硬截）

### Continuation（Primary）

| wf | trades | hit | stop | med_tgt_R | avg_win | avg_loss | payoff | **exp_R** | PF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{tbl(df_cont, ['wf','n_executed_trades','target_hit_rate','stop_hit_rate','median_target_R_exec','mean_win_R','mean_loss_R','payoff_ratio','gross_expectancy_R','profit_factor_R'])}

### Reversal（SECONDARY_DIAGNOSTIC）

| wf | trades | hit | med_tgt_R | payoff | exp_R |
|---|---:|---:|---:|---:|---:|
{tbl(df_rev, ['wf','n_executed_trades','target_hit_rate','median_target_R_exec','payoff_ratio','gross_expectancy_R'])}

---

## 3. P4 All-contact attack boundary audit

| wf | signals | same_boundary_rate | all_further_rate | median_gap | p75 | p90 |
|---|---:|---:|---:|---:|---:|---:|
{tbl(df_bnd, ['wf','n_signals','same_boundary_rate','all_boundary_further_rate','median_gap_ATR','p75_gap_ATR','p90_gap_ATR'])}

HARD ASSERTION：`boundary_gap_ATR >= 0`（all-contact boundary 必不比 selected 更近）。

---

## 4. P5 Target boundary semantics（descriptive，不改 Primary）

| wf | definition | no_target | median | p25 | p75 | p90 |
|---|---|---:|---:|---:|---:|---:|
{tbl(df_tsem, ['wf','definition','no_target_rate','median_ATR','p25_ATR','p75_ATR','p90_ATR'])}

---

## 5. P7 Geometry buckets（固定经济分桶，非 quantile）

| wf | setup | bucket | n | share | hit | avg_win | avg_loss | payoff | exp_R |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
{tbl(df_geo, ['wf','setup','bucket','n','share','target_hit_rate','mean_win_R','mean_loss_R','payoff','gross_expectancy_R'])}

**禁止据结果挑最佳 bucket**；只看低 RR 是否系统性拖累。

---

## 6. P8 Direction-label × Execution attribution（仅 post-outcome）

| wf | setup | labeled | A corr+hit | B corr+loss | C wrong+hit | D wrong+loss | P(hit|corr) | P(hit|wrong) | E[R|corr] | E[R|wrong] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{tbl(df_attr, ['wf','setup','n_labeled','A_direction_correct_target_hit','B_direction_correct_loss','C_direction_wrong_target_hit','D_direction_wrong_loss','P_hit_given_direction_correct','P_hit_given_direction_wrong','E_R_given_direction_correct','E_R_given_direction_wrong'])}

若 **B（方向正确但交易亏）** 很大 → 瓶颈在 execution mapping，不在 direction model。

---

## 7. P9 Entry degradation & P10 Same-bar

| wf | setup | next_open_R | ideal_close_R | delta |
|---|---|---:|---:|---:|
{tbl(df_ent, ['wf','setup','mean_next_open_R','mean_ideal_close_R','delta'])}

| wf | setup | ambiguous | stop_first_exp | target_first_exp | delta |
|---|---|---:|---:|---:|---:|
{tbl(df_sb, ['wf','setup','same_bar_ambiguous_rate','stop_first_expectancy_R','target_first_expectancy_R','delta'])}

---

## 8. P11 Reversal robustness（不 Promote）

| wf | trades | actual_hit | breakeven_hit | hit−breakeven | exp_R |
|---|---:|---:|---:|---:|---:|
{tbl(df_revrb, ['wf','trades','actual_hit_rate','breakeven_hit_rate','hit_minus_breakeven','gross_expectancy_R'])}

pooled gross expectancy = **{audit['reversal']['pooled_gross_expectancy_R']}**；
3/3 above breakeven = **{audit['reversal']['three_of_three_above_breakeven']}** →
保持 `SECONDARY_DIAGNOSTIC`。

---

## 9. P12 Cost

`COST_METADATA_DEFERRED_UNTIL_GROSS_EDGE`：Continuation gross point edge ≤0，
补成本的 ROI≈0。本轮无任何手续费/滑点假设。

---

## 10. P13 裁决

```json
{json.dumps(v, indent=2, ensure_ascii=False)}
```

- **EXECUTION_BASELINE_VALID_NEGATIVE = {v['EXECUTION_BASELINE_VALID_NEGATIVE']}**
  → fixed 1ATR + selected-boundary nearest target **没有稳定 gross edge**。
  **这不等同于 `DIRECTION_EDGE_FALSE`。**
- **{v['verdict']}**

---

## 11. P14 下一步 / P17 STOP

next_step = `{audit['next_step']}`

代码 + 测试 + 运行 + 报告 + commit + push 后 STOP。
不自动执行 1:1 RR filter / risk 0.5–2.0 / target 优化 / Reversal promote /
symbol 筛选 / 成本建模。等 reviewer。
"""
    open(OUT / "SMC_FIXED_EXECUTION_INTEGRITY_V1_0_1.md", "w",
         encoding="utf-8-sig").write(md)


if __name__ == "__main__":
    main()
