"""SMC Fixed Execution Baseline v1.0 (base 2006fcb).

第一次真实执行层实验。把已通过 label-availability + full-live-universe 审计的
方向 selector 转成严格事前可执行的固定交易规则，第一次测真实
win rate / payoff / expectancy。

Primary  = S1 Continuation-only (direction = side), risk = 1 ATR
Secondary= S2 Reversal-only    (direction = -side), risk = 1 ATR (diagnostic only)

执行合同（冻结，不优化）：
  collapse duplicate by (symbol, decision_time, setup) 再做 target
  attack_boundary = 同组 contacted liquidity 沿 direction 的最外沿
  CONT target = decision-time active AND ahead_of_entry AND beyond_attack
  REV  target = decision-time active AND ahead_of_entry（离开 attacked level）
  entry = next valid 5m bar open（禁止 contact close 成交）
  stop  = decision_close - direction*atr0（冻结 risk=1 语义，不重新 anchor）
  path  = entry_bar .. 第一个 discontinuity 前（不跨 roll）
  same-bar ambiguous：Primary STOP_FIRST，Secondary TARGET_FIRST
  一品种同时最多一个 position

Governance: 禁止优化 stop/target/threshold、新 feature、FVG/OB/trend/pre-contact、
删品种、新 risk、prospective OOS。REALISTIC_NET_PNL=UNAVAILABLE_COST_METADATA。
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
import research.liquidity_oracle_atlas.run_direction_preexec_integrity_v1_5 as p5
from research.liquidity_oracle_atlas.build_oracle_atlas_v1_2 import active_mask
from research.export_ob_trigger_execution_v21 import load_raw_5m
from research.phase1_tradability.phase1_contract_v1 import discontinuity_flags

OUT = Path("research/analysis_results/smc_fixed_execution_baseline_v1")
OUT.mkdir(parents=True, exist_ok=True)
ATLAS = Path("research/analysis_results/smc_oracle_atlas_v1")
MASTER = ATLAS / "liquidity_master_v1_1.parquet"
CONTACTS = ATLAS / "liquidity_contacts_v1_1.parquet"

KEYS = ["symbol", "liquidity_id", "contact_number"]
G4_BASE = t3.G4_BASE
OOS_START = t3.OOS_START
WF = [("WF1", ["TB1"], ["TB2"]),
      ("WF2", ["TB1", "TB2"], ["TB3"]),
      ("WF3", ["TB1", "TB2", "TB3"], ["TB4"])]
CLEAR_PRECISION = 0.85
CLEAR_MIN_SEL = 0.05
CONT_Q, REV_Q = 0.10, 0.90
PRIMARY_RISK = 1.0
SYMBOLS = ["AG", "AU", "CU", "AL", "SN", "NI", "RB", "I", "SC", "RU", "MA",
           "TA", "M", "P", "CF"]


# ===========================================================================
# P3 / P4 / P5  target 语义
# ===========================================================================
def attacked_boundary(prices, direction):
    """沿 direction 的最外沿 contacted liquidity。"""
    p = np.asarray(prices, float)
    j = int(np.argmax(direction * p))
    return float(p[j])


def nearest_ahead(prices, decision_close, direction):
    """最近的方向前方 active liquidity（OLD 定义 / Reversal 用）。"""
    p = np.asarray(prices, float)
    ahead = direction * (p - decision_close) > 0
    if not ahead.any():
        return None
    vp = p[ahead]
    d = direction * (vp - decision_close)
    return float(vp[int(np.argmin(d))])


def continuation_target(prices, decision_close, direction, attack):
    """P4：必须同时 ahead_of_entry 且 beyond_attack。禁止 target==attack。"""
    p = np.asarray(prices, float)
    ahead = direction * (p - decision_close) > 0
    beyond = direction * (p - attack) > 0
    valid = ahead & beyond
    if not valid.any():
        return None
    vp = p[valid]
    d = direction * (vp - decision_close)
    return float(vp[int(np.argmin(d))])


# ===========================================================================
# P6-P8  entry / stop / path
# ===========================================================================
def entry_bar_for(bars, contact_bar_index):
    """P6：entry = 下一根有效 5m bar；进入该 bar 前若有 discontinuity -> None。"""
    ebar = int(contact_bar_index) + 1
    if ebar >= bars["n"] or bars["disc"][ebar]:
        return None
    return ebar


def path_end(bars, entry_bar):
    """P9：不跨第一个 discontinuity。"""
    di = np.flatnonzero(bars["disc"][entry_bar:])
    return entry_bar + int(di[0]) if len(di) else bars["n"]


def stop_price_for(decision_close, direction, atr0):
    """P7：stop 冻结于 decision_close，不随 next-open 重新 anchor。"""
    return float(decision_close - direction * atr0)


def entry_gate_reason(direction, entry_px, stop_px, target_px):
    if not (direction * (entry_px - stop_px) > 0):
        return "ENTRY_BEYOND_STOP"
    if not (direction * (target_px - entry_px) > 0):
        return "TARGET_PASSED_BEFORE_ENTRY"
    return None


def execute_path(bars, entry_bar, end_bar, direction, stop_px, target_px,
                 stop_first=True):
    O, H, L, C = bars["o"], bars["h"], bars["l"], bars["c"]
    if entry_bar >= end_bar:
        return None
    o, h, l = O[entry_bar:end_bar], H[entry_bar:end_bar], L[entry_bar:end_bar]
    if direction == +1:
        sm, tm = (l <= stop_px), (h >= target_px)
    else:
        sm, tm = (h >= stop_px), (l <= target_px)
    si = int(np.argmax(sm)) if sm.any() else -1
    ti = int(np.argmax(tm)) if tm.any() else -1
    ambiguous = False
    if si >= 0 and ti >= 0:
        if si == ti:
            ambiguous = True
            pick_stop = stop_first
        else:
            pick_stop = si < ti
    elif si >= 0:
        pick_stop = True
    elif ti >= 0:
        pick_stop = False
    else:
        return None
    if pick_stop:
        k = si
        ex = (min(stop_px, o[k]) if direction == +1 else max(stop_px, o[k]))
        return dict(outcome="STOP", exit_px=float(ex), exit_bar=int(entry_bar + k),
                    same_bar_ambiguous=ambiguous)
    k = ti
    ex = (max(target_px, o[k]) if direction == +1 else min(target_px, o[k]))
    return dict(outcome="TARGET", exit_px=float(ex), exit_bar=int(entry_bar + k),
                same_bar_ambiguous=ambiguous)


# ===========================================================================
# environment
# ===========================================================================
def load_env():
    D = t3.load_data()
    F = D["F"].copy()
    con = pd.read_parquet(CONTACTS)[KEYS + ["liquidity_price"]]
    F = F.merge(con, on=KEYS, how="left")
    ld, aud, checks = p5.build_label_availability(F)
    F = F.merge(ld[KEYS + ["label_available_time"]], on=KEYS, how="left")
    D["F"] = F
    if checks["FATAL_LABEL_AVAILABILITY_SEMANTIC_FAIL"]:
        raise SystemExit(f"FATAL_LABEL_AVAILABILITY_SEMANTIC_FAIL: {checks}")

    master = pd.read_parquet(MASTER)
    master_by_sym = {s: g.reset_index(drop=True)
                     for s, g in master.groupby("symbol")}
    bars_by_sym = {}
    for sym in sorted(F["symbol"].unique()):
        raw = load_raw_5m(sym).sort_values("bar_start_time").reset_index(
            drop=True)
        disc = np.asarray(discontinuity_flags(sym), bool)
        bars_by_sym[sym] = dict(
            o=raw["open"].to_numpy(float), h=raw["high"].to_numpy(float),
            l=raw["low"].to_numpy(float), c=raw["close"].to_numpy(float),
            t=pd.to_datetime(raw["bar_start_time"]).to_numpy(),
            day=pd.to_datetime(raw["trading_day"]).to_numpy(),
            disc=disc, n=len(raw))
    return D, master_by_sym, bars_by_sym


# ===========================================================================
# P1 availability-safe selector（复用 v1.5 helper）
# ===========================================================================
def build_wf(name, trb, teb, D, clear_cols):
    F = D["F"]; block = D["block"]; insample = D["insample"]
    X = D["X"]; y_clear = D["y_clear"]; y_rev = D["y_rev"]; days = D["days"]
    lav = pd.to_datetime(F["label_available_time"]).to_numpy()
    dtime = pd.to_datetime(F["decision_time"]).to_numpy()

    m_tr = pd.Series(block).isin(trb).to_numpy() & insample
    test_all = p5.test_all_mask(block, teb, insample)
    test_start = dtime[test_all].min()

    clear_raw = m_tr & (~pd.isna(y_clear))
    dir_raw = m_tr & (~pd.isna(y_rev))
    clear_train = clear_raw & (lav < test_start)
    dir_train = dir_raw & (lav < test_start)

    oc, pc_oof, _ = p5.expanding_oof_pred_available(
        clear_cols, X[clear_train], y_clear[clear_train],
        dtime[clear_train], lav[clear_train], days[clear_train], "logistic")
    od, pr_oof, _ = p5.expanding_oof_pred_available(
        G4_BASE, X[dir_train], y_rev[dir_train],
        dtime[dir_train], lav[dir_train], days[dir_train], "hgb")
    clear_thr = v2.choose_clear_threshold(
        y_clear[clear_train][oc], pc_oof, CLEAR_PRECISION, CLEAR_MIN_SEL)
    cont_thr = float(np.quantile(pr_oof, CONT_Q))
    rev_thr = float(np.quantile(pr_oof, REV_Q))

    p_clear = rp.fit_predict(clear_cols, X[clear_train], y_clear[clear_train],
                             X[test_all], "logistic")
    p_rev = rp.fit_predict(G4_BASE, X[dir_train], y_rev[dir_train],
                           X[test_all], "hgb")
    return dict(name=name, test_all=test_all, p_clear=p_clear, p_rev=p_rev,
                clear_thr=clear_thr, cont_thr=cont_thr, rev_thr=rev_thr,
                n_test_all=int(test_all.sum()))


# ===========================================================================
# P2 collapse
# ===========================================================================
def collapse_contacts(df):
    """df: selected contacts. 返回 collapsed signals（含 conflict 标记）。"""
    out, n_conflict = [], 0
    for (sym, dt), g in df.groupby(["symbol", "decision_time"], sort=False):
        ndir = g["direction"].nunique()
        if ndir > 1:
            n_conflict += 1
            continue
        if g["entry_reference"].nunique() != 1 or g["atr0"].nunique() != 1 \
                or g["contact_bar_index"].nunique() != 1:
            raise SystemExit("FATAL_EXECUTION_GROUP_MISMATCH "
                             f"{sym} {dt}")
        out.append(dict(
            symbol=sym, decision_time=dt,
            direction=int(g["direction"].iloc[0]),
            side=int(g["side"].iloc[0]),
            decision_close=float(g["entry_reference"].iloc[0]),
            atr0=float(g["atr0"].iloc[0]),
            contact_bar_index=int(g["contact_bar_index"].iloc[0]),
            attack=attacked_boundary(g["liquidity_price"].to_numpy(),
                                     int(g["direction"].iloc[0])),
            n_contacts=len(g),
        ))
    return pd.DataFrame(out), n_conflict


# ===========================================================================
# target gate + execution
# ===========================================================================
def attach_targets(sig, master_by_sym, setup):
    rows = []
    for r in sig.itertuples(index=False):
        ms = master_by_sym.get(r.symbol)
        dtn = np.datetime64(pd.Timestamp(r.decision_time))
        am = active_mask(ms, dtn)
        pr = ms["price"].to_numpy(float)[am]
        if setup == "CONT":
            new_t = continuation_target(pr, r.decision_close, r.direction,
                                        r.attack)
            old_t = nearest_ahead(pr, r.decision_close, r.direction)
            no_tgt = "NO_NEXT_LIQUIDITY_TARGET"
        else:
            new_t = nearest_ahead(pr, r.decision_close, r.direction)
            old_t = None
            no_tgt = "NO_OPPOSING_TARGET"
        d_new = (abs(new_t - r.decision_close) / r.atr0 if new_t is not None
                 else np.nan)
        d_old = (abs(old_t - r.decision_close) / r.atr0 if old_t is not None
                 else np.nan)
        rows.append(dict(symbol=r.symbol, decision_time=r.decision_time,
                         direction=r.direction, side=r.side,
                         decision_close=r.decision_close, atr0=r.atr0,
                         contact_bar_index=r.contact_bar_index,
                         attack=r.attack, n_contacts=r.n_contacts,
                         target_price=new_t, old_target_price=old_t,
                         target_dist_ATR=d_new, old_target_dist_ATR=d_old,
                         target_equals_attack=bool(
                             new_t is not None
                             and np.isclose(new_t, r.attack)),
                         skip_reason=("" if new_t is not None else no_tgt)))
    return pd.DataFrame(rows)


def run_execution(sig, bars_by_sym):
    """P6-P11：返回 trades + 各 gate 计数。"""
    n_after_target = int((sig["skip_reason"] == "").sum())
    cand = sig[sig["skip_reason"] == ""].sort_values(["decision_time", "symbol"])
    open_until = {}
    trades, n_gap, n_skip_open = [], 0, 0
    for r in cand.itertuples(index=False):
        bars = bars_by_sym[r.symbol]
        n = bars["n"]
        ebar = entry_bar_for(bars, r.contact_bar_index)
        skip = None
        entry_px = stop_px = target_px = risk_px = np.nan
        if ebar is None:
            skip = "DISCONTINUITY_BEFORE_ENTRY"
        else:
            entry_px = float(bars["o"][ebar])
            stop_px = stop_price_for(r.decision_close, r.direction, r.atr0)
            target_px = float(r.target_price)
            risk_px = float(r.direction * (entry_px - stop_px))
            skip = entry_gate_reason(r.direction, entry_px, stop_px, target_px)
        if skip:
            n_gap += 1
            trades.append(dict(symbol=r.symbol, decision_time=r.decision_time,
                               entry_day=None, direction=r.direction,
                               executed=False, skip_reason=skip,
                               realized_R=np.nan))
            continue
        # P10 one position per symbol
        if r.symbol in open_until and ebar <= open_until[r.symbol]:
            n_skip_open += 1
            trades.append(dict(symbol=r.symbol, decision_time=r.decision_time,
                               entry_day=None, direction=r.direction,
                               executed=False,
                               skip_reason="SKIPPED_POSITION_ALREADY_OPEN",
                               realized_R=np.nan))
            continue
        end = path_end(bars, ebar)
        res = execute_path(bars, ebar, end, r.direction, stop_px, target_px,
                           stop_first=True)
        res_opt = execute_path(bars, ebar, end, r.direction, stop_px,
                               target_px, stop_first=False)
        if res is None:
            if end < n:
                outcome, exit_px, ebar_exit = "ROLL_EXIT", float(
                    bars["c"][end - 1]), int(end - 1)
            else:
                outcome, exit_px, ebar_exit = "DATA_END_EXIT", float(
                    bars["c"][n - 1]), int(n - 1)
            amb = False
        else:
            outcome, exit_px = res["outcome"], res["exit_px"]
            ebar_exit, amb = res["exit_bar"], res["same_bar_ambiguous"]
        risk_exec = abs(entry_px - stop_px)
        realized_R = r.direction * (exit_px - entry_px) / risk_exec
        target_R = r.direction * (target_px - entry_px) / risk_exec
        ideal_R = r.direction * (exit_px - r.decision_close) / r.atr0
        opt_R = np.nan
        if res_opt is not None:
            opt_R = r.direction * (res_opt["exit_px"] - entry_px) / risk_exec
        open_until[r.symbol] = int(ebar_exit)
        trades.append(dict(
            symbol=r.symbol, decision_time=r.decision_time,
            entry_day=pd.Timestamp(bars["day"][ebar]),
            direction=r.direction, executed=True, skip_reason="",
            entry_px=entry_px, stop_px=stop_px, target_px=target_px,
            risk_px=risk_px, risk_px_over_atr0=float(risk_px / r.atr0),
            target_dist_ATR=float(abs(target_px - entry_px) / r.atr0),
            outcome=outcome, exit_px=float(exit_px), exit_bar=ebar_exit,
            same_bar_ambiguous=bool(amb),
            realized_R=float(realized_R), target_R_exec=float(target_R),
            ideal_close_fill_R=float(ideal_R),
            target_first_R=(float(opt_R) if res_opt is not None else np.nan),
        ))
    return pd.DataFrame(trades), n_after_target, n_gap, n_skip_open


def trade_metrics(tr, n_days, n_symbols=15):
    ex = tr[tr["executed"]]
    n = len(ex)
    if n == 0:
        return dict(n_executed_trades=0, gross_expectancy_R=np.nan)
    hit = (ex["outcome"] == "TARGET").mean()
    stop = (ex["outcome"] == "STOP").mean()
    roll = (ex["outcome"] == "ROLL_EXIT").mean()
    amb = ex["same_bar_ambiguous"].mean()
    rr = ex["realized_R"].to_numpy(float)
    wins = rr[rr > 0]
    losses = rr[rr < 0]
    mw = float(wins.mean()) if len(wins) else np.nan
    ml = float(losses.mean()) if len(losses) else np.nan
    payoff = (mw / abs(ml)) if (len(wins) and len(losses) and ml != 0) else np.nan
    gross = float(rr.mean())
    sw = float(wins.sum()) if len(wins) else 0.0
    sl = float(abs(losses.sum())) if len(losses) else 0.0
    pf = (sw / sl) if sl > 0 else np.nan
    return dict(
        n_executed_trades=n,
        trade_frequency_per_day=round(n / max(n_days, 1), 4),
        trades_per_symbol=round(n / n_symbols, 2),
        target_hit_rate=round(float(hit), 4),
        stop_hit_rate=round(float(stop), 4),
        roll_exit_rate=round(float(roll), 4),
        same_bar_ambiguous_rate=round(float(amb), 4),
        median_target_R_exec=round(float(ex["target_R_exec"].median()), 4),
        mean_win_R=round(mw, 4) if pd.notna(mw) else np.nan,
        mean_loss_R=round(ml, 4) if pd.notna(ml) else np.nan,
        payoff_ratio=round(payoff, 4) if pd.notna(payoff) else np.nan,
        gross_expectancy_R=round(gross, 4),
        profit_factor_R=round(pf, 4) if pd.notna(pf) else np.nan,
        gross_expectancy_R_ideal_close_fill=round(
            float(ex["ideal_close_fill_R"].mean()), 4),
        gross_expectancy_R_target_first=round(
            float(ex["target_first_R"].dropna().mean()), 4)
        if ex["target_first_R"].notna().any() else np.nan,
    )


def bootstrap_mean_R(day_arr, r_arr, n_boot=500, seed=42):
    days = np.sort(pd.unique(day_arr))
    idx = {d: np.flatnonzero(day_arr == d) for d in days}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        samp = rng.choice(days, size=len(days), replace=True)
        ii = np.concatenate([idx[d] for d in samp])
        m = r_arr[ii]
        if len(m):
            means.append(float(m.mean()))
    if not means:
        return None
    a = np.array(means)
    return dict(n_boot=len(a), p2_5=round(float(np.percentile(a, 2.5)), 4),
                p50=round(float(np.percentile(a, 50)), 4),
                p97_5=round(float(np.percentile(a, 97.5)), 4))


# ===========================================================================
# main
# ===========================================================================
def main():
    t0 = time.perf_counter()
    D, master_by_sym, bars_by_sym = load_env()
    print(f"[ENV] loaded ({time.perf_counter()-t0:.1f}s)")
    F = D["F"]; block = D["block"]; insample = D["insample"]
    clear_cols = [c for c in rp.block_cols("M_GLOBAL4", D["LIQ_TYPES"])
                  if c in D["X"].columns]

    funnel, cont_rows, rev_rows = [], [], []
    all_trades, all_cont_trades = [], []
    tgt_audit = []
    collapse_rows = []
    for name, trb, teb in WF:
        w = build_wf(name, trb, teb, D, clear_cols)
        te = w["test_all"]
        sub = pd.DataFrame(dict(
            symbol=D["SYM"][te],
            decision_time=pd.to_datetime(F["decision_time"].to_numpy())[te],
            side=D["side"][te],
            entry_reference=F["entry_reference"].to_numpy()[te],
            atr0=F["atr0"].to_numpy()[te],
            contact_bar_index=F["contact_bar_index"].to_numpy()[te],
            liquidity_price=F["liquidity_price"].to_numpy()[te],
        ))
        sel_cont = p5.s1_select(w["p_clear"], w["p_rev"], w["clear_thr"],
                                w["cont_thr"])
        sel_rev = (w["p_clear"] >= w["clear_thr"]) & (w["p_rev"] >= w["rev_thr"])

        for setup, sel, cont_rows_, rev_rows_, store in (
                ("CONT", sel_cont, cont_rows, None, all_cont_trades),
                ("REV", sel_rev, None, rev_rows, [])):
            d = sub[sel].copy()
            d["direction"] = (d["side"] if setup == "CONT" else -d["side"])
            sig, n_conflict = collapse_contacts(d)
            collapse_rows.append(dict(
                wf=name, setup=setup, n_raw_selected_contacts=int(sel.sum()),
                n_collapsed_signals=len(sig),
                n_conflict_groups_abstained=n_conflict,
                collapse_ratio=round(len(sig) / max(int(sel.sum()), 1), 4)))
            if not len(sig):
                continue
            st = attach_targets(sig, master_by_sym, setup)
            if setup == "CONT":
                tgt_audit.append(st)
            tr, n_after_tgt, n_gap, n_skip = run_execution(st, bars_by_sym)
            n_days = int(pd.Series(
                pd.to_datetime(F["trading_day"]).to_numpy()[te]).nunique())
            m = trade_metrics(tr, n_days)
            funnel.append(dict(
                wf=name, setup=setup,
                n_raw_selected_contacts=int(sel.sum()),
                n_collapsed_signals=len(sig),
                n_conflict_abstained=n_conflict,
                n_after_target_gate=n_after_tgt,
                n_after_entry_gap_gate=int(tr["executed"].sum() + n_skip),
                n_executed_trades=int(tr["executed"].sum()),
                skip_open_position=n_skip,
                raw_contact_selection_rate=round(float(sel.mean()), 4),
                collapsed_signal_rate=round(len(sig) / max(int(te.sum()), 1), 4),
                executed_trade_rate=round(
                    int(tr["executed"].sum()) / max(int(te.sum()), 1), 4),
            ))
            row = dict(wf=name, **m)
            if setup == "CONT":
                cont_rows.append(row)
                all_cont_trades.append(tr[tr["executed"]])
            else:
                rev_rows.append(row)
        print(f"[{name}] cont_trades={cont_rows[-1]['n_executed_trades']} "
              f"expR={cont_rows[-1]['gross_expectancy_R']} "
              f"hit={cont_rows[-1]['target_hit_rate']}")

    df_trades = pd.concat(all_cont_trades, ignore_index=True)
    df_cont = pd.DataFrame(cont_rows)
    df_rev = pd.DataFrame(rev_rows)
    df_fun = pd.DataFrame(funnel)
    df_col = pd.DataFrame(collapse_rows)
    df_tgt = pd.concat(tgt_audit, ignore_index=True)

    # ---- P17 HARD ASSERTION：NEW target 绝不等于 attack ----
    eq_rate = float(df_tgt["target_equals_attack"].mean())
    print(f"[P17] target_equals_attack_rate={eq_rate}")
    assert df_tgt["target_equals_attack"].sum() == 0, \
        "HARD ASSERTION FAIL: NEW target equals attack boundary"

    # ---- P15 gate ----
    exps = list(df_cont["gross_expectancy_R"])
    pooled_exp = float(df_trades["realized_R"].mean())
    GATE = bool(all(pd.notna(e) and e > 0 for e in exps) and pooled_exp > 0)

    # ---- P16 bootstrap ----
    if GATE:
        b = bootstrap_mean_R(df_trades["entry_day"].to_numpy(),
                             df_trades["realized_R"].to_numpy(float))
        df_boot = pd.DataFrame([dict(scope="pooled", **b)]) if b else \
            pd.DataFrame([dict(note="STOP_NO_BOOTSTRAP")])
    else:
        df_boot = pd.DataFrame([dict(note="STOP_NO_BOOTSTRAP: "
                                    "GROSS_EXECUTION_EDGE_PRESENT=False")])

    # ---- P14 cost metadata audit ----
    cost = dict(
        scanned=["tick_size", "contract_multiplier", "commission",
                 "exchange_fee", "broker_fee", "slippage"],
        canonical_table_found=False,
        evidence=["research/exports/strategy_s1/s1_config.json: "
                  "'simple gross returns, no fees, no slippage'",
                  "research/analysis_results/phase1_tradability_v1/"
                  "symbol_universe.csv: no tick_size/multiplier/commission"],
        REALISTIC_NET_PNL="UNAVAILABLE_COST_METADATA",
        rule="禁止凭记忆填写手续费/滑点；只报 GROSS R 与 break-even cost",
        break_even_round_trip_cost_R=round(pooled_exp, 4),
        interpretation="每笔交易总往返成本（R）超过该值则 gross edge 消失；"
                       "不得声称 net profitable。",
        break_even_note=("gross expectancy<=0 → break-even cost<=0："
                         "在扣除任何成本之前 edge 已为负，没有任何成本承受空间。"
                         if pooled_exp <= 0 else
                         "gross expectancy>0；成本须低于该值才可能 net positive。"),
    )

    # ---- write ----
    df_fun.to_csv(OUT / "execution_funnel.csv", index=False, encoding="utf-8-sig")
    df_col.to_csv(OUT / "signal_collapse_audit.csv", index=False,
                  encoding="utf-8-sig")
    df_cont.to_csv(OUT / "continuation_execution_metrics.csv", index=False,
                   encoding="utf-8-sig")
    df_rev.to_csv(OUT / "reversal_execution_diagnostic.csv", index=False,
                  encoding="utf-8-sig")
    df_boot.to_csv(OUT / "execution_bootstrap_ci.csv", index=False,
                   encoding="utf-8-sig")
    df_trades.to_parquet(OUT / "execution_trade_log.parquet", index=False)

    # P13 continuation vs reversal
    cmp_rows = []
    for c, r in zip(cont_rows, rev_rows):
        for k in ("n_executed_trades", "target_hit_rate",
                  "median_target_R_exec", "mean_win_R", "mean_loss_R",
                  "payoff_ratio", "gross_expectancy_R", "profit_factor_R"):
            cmp_rows.append(dict(wf=c["wf"], metric=k,
                                 continuation=c.get(k), reversal=r.get(k)))
    df_cmp = pd.DataFrame(cmp_rows)
    df_cmp.to_csv(OUT / "continuation_vs_reversal.csv", index=False,
                  encoding="utf-8-sig")

    # P17 target semantics audit
    trows = []
    for wf, g in zip([w[0] for w in WF], tgt_audit):
        trows.append(dict(
            wf=wf, definition="NEW_BEYOND_ATTACK",
            no_target_rate=round(float(g["target_price"].isna().mean()), 4),
            median_ATR=round(float(g["target_dist_ATR"].median()), 4),
            p25_ATR=round(float(g["target_dist_ATR"].quantile(.25)), 4),
            p75_ATR=round(float(g["target_dist_ATR"].quantile(.75)), 4),
            p90_ATR=round(float(g["target_dist_ATR"].quantile(.90)), 4),
            target_equals_attack_rate=round(
                float(g["target_equals_attack"].mean()), 4)))
        trows.append(dict(
            wf=wf, definition="OLD_ENTRY_NEAREST",
            no_target_rate=round(float(g["old_target_price"].isna().mean()), 4),
            median_ATR=round(float(g["old_target_dist_ATR"].median()), 4),
            p25_ATR=round(float(g["old_target_dist_ATR"].quantile(.25)), 4),
            p75_ATR=round(float(g["old_target_dist_ATR"].quantile(.75)), 4),
            p90_ATR=round(float(g["old_target_dist_ATR"].quantile(.90)), 4),
            target_equals_attack_rate=round(float(np.isclose(
                g["old_target_price"].astype(float),
                g["attack"].astype(float)).mean()), 4)))
    df_ts = pd.DataFrame(trows)
    df_ts.to_csv(OUT / "target_semantics_audit.csv", index=False,
                 encoding="utf-8-sig")

    # P13/secondary by symbol
    sym_rows = []
    for s, g in df_trades.groupby("symbol"):
        rr = g["realized_R"].to_numpy(float)
        w = rr[rr > 0]; l = rr[rr < 0]
        sym_rows.append(dict(
            symbol=s, n_trades=len(g),
            target_hit_rate=round(float((g["outcome"] == "TARGET").mean()), 4),
            mean_win_R=round(float(w.mean()), 4) if len(w) else np.nan,
            mean_loss_R=round(float(l.mean()), 4) if len(l) else np.nan,
            gross_expectancy_R=round(float(rr.mean()), 4)))
    df_sym = pd.DataFrame(sym_rows)
    df_sym.to_csv(OUT / "execution_by_symbol.csv", index=False,
                  encoding="utf-8-sig")

    protocol = dict(
        experiment="SMC Fixed Execution Baseline v1.0",
        base_commit="2006fcb61201873bae44b516a994f1e8e17c8917",
        primary="S1 Continuation-only", secondary="S2 Reversal-only",
        risk=PRIMARY_RISK,
        selector=dict(clear="C_GLOBAL4 Logistic, availability-safe OOF 0.85",
                      direction="G4_BASE HGB, availability-safe OOF",
                      cont_threshold="q10 of availability-safe direction OOF",
                      rev_threshold="q90 of availability-safe direction OOF",
                      test_inference="ALL contacts"),
        collapse="by (symbol, decision_time, setup); conflict -> abstain",
        attack_boundary="outermost contacted liquidity along direction",
        cont_target="active AND ahead_of_entry AND beyond_attack",
        rev_target="active AND ahead_of_entry (opposing side)",
        entry="next valid 5m bar open (not contact close)",
        stop="decision_close - direction*atr0 (frozen, not re-anchored)",
        same_bar="Primary STOP_FIRST; Secondary TARGET_FIRST",
        roll="no crossing discontinuity; exit last trusted close",
        position="max one open position per symbol",
        p0_v15_fixes=["coverage moved out of gate -> coverage_reference",
                      "v1.5 target is availability-only, not frozen exec target"],
        forbidden=["优化stop", "优化target", "调threshold", "新feature", "FVG",
                   "OB", "trend", "pre-contact", "删品种", "新risk",
                   "prospective OOS", "按PnL筛品种"],
        trading_metrics="GROSS R only (REALISTIC_NET_PNL="
                        "UNAVAILABLE_COST_METADATA)",
    )
    json.dump(protocol, open(OUT / "FIXED_EXECUTION_PROTOCOL.json", "w"),
              indent=2, ensure_ascii=False, default=str)
    json.dump(cost, open(OUT / "cost_metadata_audit.json", "w"),
              indent=2, ensure_ascii=False, default=str)

    audit = dict(
        experiment="SMC Fixed Execution Baseline v1.0",
        base_commit="2006fcb",
        gate=dict(GROSS_EXECUTION_EDGE_PRESENT=GATE,
                  per_wf_gross_expectancy_R=exps,
                  pooled_gross_expectancy_R=round(pooled_exp, 4)),
        p17_target_equals_attack_rate=eq_rate,
        cost=cost,
        next_step=("CONTRACT_COST_METADATA_AUDIT then net execution"
                   if GATE else
                   "STOP: no gross edge; do not optimize target/stop/threshold"),
    )
    json.dump(audit, open(OUT / "FIXED_EXECUTION_AUDIT.json", "w"),
              indent=2, ensure_ascii=False, default=str)

    write_report(df_fun, df_col, df_cont, df_rev, df_cmp, df_ts, df_sym,
                 df_boot, audit, cost)
    print("\n=== GATE ===")
    print(json.dumps(audit["gate"], indent=2, ensure_ascii=False))
    print(f"[DONE] {time.perf_counter()-t0:.1f}s -> {OUT}")


def write_report(df_fun, df_col, df_cont, df_rev, df_cmp, df_ts, df_sym,
                 df_boot, audit, cost):
    def tbl(df, cols):
        return "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |"
                         for _, r in df.iterrows())
    g = audit["gate"]
    md = f"""# SMC Fixed Execution Baseline v1.0

> ## 状态：`PROVISIONAL_PENDING_OOS_PATH_AUDIT`
>
> v1.0 的 `path_end()` **只截 discontinuity / raw data end，没有截
> `OOS_START = 2026-09-07`**；v1.0 的 `test_no_prospective_oos()` 只验证
> **signal/contact date**，**没有验证 entry / exit path**。
> 因此持仓路径可能读取 prospective OOS bar。
>
> 修复与审计见 **v1.0.1** (`smc_fixed_execution_integrity_v1_0_1/`)。
> **在此之前本文件所有 PnL 数字均为 provisional，不得作为最终执行基线收口。**

**base**: `2006fcb` (v1.5) &nbsp; **脚本**: `run_fixed_execution_baseline_v1.py`
**第一次真实执行层实验**：把通过 label-availability + full-live-universe 审计的
方向 selector 转为严格事前可执行的固定规则，测真实 win rate / payoff / expectancy。

Primary = **S1 Continuation-only**（direction = side，risk = 1 ATR）
Secondary = **S2 Reversal-only**（diagnostic，不参与 PASS/FAIL）

---

## 0. P0 v1.5 文档语义修正

- `gate.per_wf_selection = 0.05` **移出 Gate** → 改名 `coverage_reference`。
  原因：contact 不是执行计量单位；collapse 后应以 unique signal / trade frequency 报告。
- v1.5 `nearest_exante_target` 只验证 **active target availability**，
  **不是冻结 execution target 语义**（TOUCH_ONLY 下 attacked level 仍 active）。

---

## 1. 执行合同（冻结）

| 环节 | 定义 |
|---|---|
| collapse | `(symbol, decision_time, setup)`；方向冲突 → abstain |
| attack_boundary | 同组 contacted liquidity 沿 direction 的**最外沿** |
| CONT target | decision-time active **AND** ahead_of_entry **AND** beyond_attack |
| REV target | decision-time active **AND** ahead_of_entry（离开 attacked level） |
| entry | **下一根有效 5m bar open**（禁止 contact close 成交） |
| stop | `decision_close - direction*atr0`（冻结 risk=1，**不重新 anchor**） |
| path | entry_bar .. 第一个 discontinuity 前（不跨 roll） |
| same-bar | Primary **STOP_FIRST**，Secondary TARGET_FIRST |
| position | 一品种同时最多一个 position |

---

## 2. Execution funnel

| wf | setup | raw contacts | collapsed | conflict | after target | executed | skip_open |
|---|---|---:|---:|---:|---:|---:|---:|
{tbl(df_fun, ['wf','setup','n_raw_selected_contacts','n_collapsed_signals','n_conflict_abstained','n_after_target_gate','n_executed_trades','skip_open_position'])}

Coverage（P18 新单位，不再用 contact selection≥5% 当 gate）：

| wf | setup | raw_contact_sel | collapsed_signal_rate | executed_trade_rate |
|---|---|---:|---:|---:|
{tbl(df_fun, ['wf','setup','raw_contact_selection_rate','collapsed_signal_rate','executed_trade_rate'])}

---

## 3. P12 Primary Continuation 执行指标

| wf | trades | /day | target_hit | stop_hit | roll | same_bar | med_tgt_R | avg_win_R | avg_loss_R | payoff | **gross_exp_R** | PF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{tbl(df_cont, ['wf','n_executed_trades','trade_frequency_per_day','target_hit_rate','stop_hit_rate','roll_exit_rate','same_bar_ambiguous_rate','median_target_R_exec','mean_win_R','mean_loss_R','payoff_ratio','gross_expectancy_R','profit_factor_R'])}

> `target_hit_rate` 才是真正交易胜率；此前的 65% `actionable` **不是 win rate**。

---

## 4. P13 Continuation vs Reversal

| wf | metric | Continuation | Reversal |
|---|---|---:|---:|
{tbl(df_cmp, ['wf','metric','continuation','reversal'])}

Reversal 仅 `SECONDARY_DIAGNOSTIC`，不得据此加入 Primary。

---

## 5. P17 Target semantics audit（OLD vs NEW）

| wf | definition | no_target_rate | median ATR | p25 | p75 | p90 | target==attack |
|---|---|---:|---:|---:|---:|---:|---:|
{tbl(df_ts, ['wf','definition','no_target_rate','median_ATR','p25_ATR','p75_ATR','p90_ATR','target_equals_attack_rate'])}

**HARD ASSERTION**：`NEW_BEYOND_ATTACK` 的 `target_equals_attack_rate` 必须为 **0**
（实测 `{audit['p17_target_equals_attack_rate']}`）。
OLD 定义下该比例很高，正是 v1.5 median target≈0.32–0.40 ATR 的原因。

---

## 6. P14 Cost metadata

```json
{json.dumps(cost, indent=2, ensure_ascii=False)}
```

只报 **GROSS R**；`break_even_round_trip_cost_R = gross_expectancy_R` 表示
每笔往返成本超过该值则 gross edge 消失。**不得声称 net profitable。**

---

## 7. P15 Gate

```json
{json.dumps(g, indent=2, ensure_ascii=False)}
```

**GROSS_EXECUTION_EDGE_PRESENT = {g['GROSS_EXECUTION_EDGE_PRESENT']}**
（要求 3/3 WF expectancy_R>0 且 pooled>0）。若 FALSE → 停止进一步成本/参数优化，
**不得为过 gate 改 target/stop/threshold**。

---

## 8. P16 Bootstrap（entry trading-day block）

| scope | n_boot | p2.5 | p50 | p97.5 |
|---|---:|---:|---:|---:|
{tbl(df_boot, [c for c in ['scope','n_boot','p2_5','p50','p97_5'] if c in df_boot.columns]) if 'n_boot' in df_boot.columns else '| - | STOP_NO_BOOTSTRAP | | | |'}

---

## 9. 按 symbol（P13 secondary，不筛品种）

| symbol | trades | hit_rate | avg_win_R | avg_loss_R | gross_exp_R |
|---|---:|---:|---:|---:|---:|
{tbl(df_sym, ['symbol','n_trades','target_hit_rate','mean_win_R','mean_loss_R','gross_expectancy_R'])}

---

## 10. P21 完成 / STOP

代码 + 测试 + 运行 + 报告 + commit + push 后 STOP。
禁止据本轮结果优化 target/stop、筛 symbol、改 threshold、把 Reversal 加入主策略。
"""
    open(OUT / "SMC_FIXED_EXECUTION_BASELINE_V1.md", "w",
         encoding="utf-8-sig").write(md)


if __name__ == "__main__":
    main()
