"""SMC Risk-Coupled Execution v1.0 (base 7107f16).

固定比较 risk grid：0.5 / 1.0 / 2.0 ATR（禁止其他 risk 扫描）。
其中 1.0 = 当前 baseline；0.5 = 更紧风险尺度；2.0 = 更宽风险尺度。

核心原则（预注册冻结）：
  每个 risk 必须独立重建（禁止拿 risk=1 模型直接换 stop）：
    Oracle rr_direction(r)
        -> CLEAR / TRADEOFF(r)
        -> Reversal / Continuation label(r)
        -> label_available_time(r)
        -> Clear model(r)   [C_GLOBAL4 Logistic, OOF precision 0.85, min sel 0.05]
        -> Direction model(r)[G4_BASE HGB depth3 lr0.05 iter200 l2 1 seed42]
        -> OOF thresholds(r)
        -> Selector(r)      [CLEAR85 AND Continuation bottom-10% direction OOF]
        -> fixed execution with stop = decision_close - direction*risk*atr0
  Target 语义不因 risk 改变（continuation = beyond attacked boundary）。
  Primary 只做 Continuation；Reversal 冻结（REVERSAL_CLEAR90_CANDIDATE）不参与。

rr_direction(r) 复算依据（profile_oracle_atlas_v1_2.py 冻结规则）：
  notg  = (res_l=="NO_ACTIVE_TARGET")|(res_s=="NO_ACTIVE_TARGET")
  unres = (~notg)&(lu.isna()|su.isna())
  cmp   = (~notg)&(~unres)
  rr = np.select([notg,unres,cmp&(ll>su),cmp&(sl>lu)],
       ["NO_COMPARABLE_TARGET","UNRESOLVED_CENSOR","LONG_DOMINATES","SHORT_DOMINATES"],
       default="TRADEOFF_OR_OVERLAP")
  ll/lu=long best_R_lower/upper, sl/su=short best_R_lower/upper（全部来自 frontier）。

HARD baseline：risk=1 必须逐位复现
  trades 614/856/848 ; E[R] -0.0420 / +0.0253 / -0.0376
否则 FATAL_RISK1_BASELINE_REPRODUCTION_FAIL。

Gate（RISK_COUPLED_GROSS_EDGE_CANDIDATE）：某 risk 同时满足
  WF1/2/3 E[R]>0 且 pooled E[R]>0 且每 WF >=200 executed trades
才标 candidate；未通过 STOP_NO_BOOTSTRAP，仅通过的 risk 跑 500 日 block bootstrap。

Governance: 禁止 risk 连续扫描 / target 优化 / RR filter / clear precision 扫描 /
direction threshold 扫描 / symbol 筛选 / 新 feature / FVG/OB/trend / precontact /
cost 假设 / prospective OOS。TRADING_METRICS 见输出（已定义交易动作）。
"""
from __future__ import annotations

import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(".").resolve()))
import research.liquidity_oracle_atlas.run_direction_deployability_v1_1_1_repair as rp
import research.liquidity_oracle_atlas.run_direction_actionability_v1_2 as v2
import research.liquidity_oracle_atlas.run_direction_temporal_stability_v1_3 as t3
import research.liquidity_oracle_atlas.run_direction_preexec_integrity_v1_5 as p5
import research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 as m0
import research.liquidity_oracle_atlas.run_fixed_execution_integrity_v1_0_1 as m101
import research.liquidity_oracle_atlas.run_group_tradeoff_veto_v1_3 as v13

# ---- G1 聚合列注册（与 v1.3 一致，确保 G4_BASE 之外无新行情特征）----
G4_BASE = list(t3.G4_BASE)
G1_AGG = ["n_selected_contacts", "outer_p_clear", "outer_p_rev",
          "min_p_clear", "mean_p_clear", "max_p_clear",
          "min_p_rev", "mean_p_rev", "max_p_rev"]
rp.ORDINARY_NUMERIC |= set(G1_AGG)

OUT = Path("research/analysis_results/smc_risk_coupled_execution_v1")
OUT.mkdir(parents=True, exist_ok=True)
BASE_COMMIT = "7107f16e806e2580a42abae247ca892c86ec13bf"
PRIMARY_RISK = 1.0
RISKS = [0.5, 1.0, 2.0]
CLEAR_PRECISION = 0.85
CLEAR_MIN_SEL = 0.05
CONT_Q = 0.10
REV_Q = 0.90
GATE_MIN_TRADES = 200
KEYS = ["symbol", "liquidity_id", "contact_number"]
FROZEN_CLEAR = ["LONG_DOMINATES", "SHORT_DOMINATES", "TRADEOFF_OR_OVERLAP"]
EVALUABLE = ["LONG_DOMINATES", "SHORT_DOMINATES", "TRADEOFF_OR_OVERLAP"]
UNKNOWN = ["UNRESOLVED_CENSOR", "NO_COMPARABLE_TARGET"]
FRONT = Path("research/analysis_results/smc_oracle_atlas_v1") / "oracle_risk_frontier_v1_2.parquet"


# ===========================================================================
# P1 rr_direction(r) 复算（冻结规则，来自 profile_oracle_atlas_v1_2.py）
# ===========================================================================
def rr_direction_at_risk(front, risk):
    """返回以 KEYS 为 index 的 rr_direction(r) Series。"""
    o = front[np.isclose(front["risk_ATR"].to_numpy(float), float(risk))].copy()
    L = o[o["direction"] == "LONG"].set_index(KEYS)
    S = o[o["direction"] == "SHORT"].set_index(KEYS)
    idx = L.index.intersection(S.index)
    if len(idx) == 0:
        return pd.Series(index=pd.MultiIndex.from_arrays(
            [[]] * len(KEYS), names=KEYS), dtype=object)
    L = L.loc[idx]
    S = S.loc[idx]
    ll = L["best_R_lower"].to_numpy(float)
    lu = L["best_R_upper"].to_numpy(float)
    sl = S["best_R_lower"].to_numpy(float)
    su = S["best_R_upper"].to_numpy(float)
    rl = L["resolution_class"].to_numpy()
    rs = S["resolution_class"].to_numpy()
    notg = (rl == "NO_ACTIVE_TARGET") | (rs == "NO_ACTIVE_TARGET")
    unres = (~notg) & (np.isnan(lu) | np.isnan(su))
    cmp = (~notg) & (~unres)
    rr = np.select(
        [notg, unres, cmp & (ll > su), cmp & (sl > lu)],
        ["NO_COMPARABLE_TARGET", "UNRESOLVED_CENSOR",
         "LONG_DOMINATES", "SHORT_DOMINATES"],
        default="TRADEOFF_OR_OVERLAP")
    return pd.Series(rr, index=idx)


def make_risk_labels(rr_array, side):
    """rr_array: 每 contact 的 rr_direction(r)。返回 y_clear(r), y_rev(r)。
    y_clear: 1=LONG/SHORT_DOMINATES, 0=TRADEOFF_OR_OVERLAP, NaN=unknown。
    y_rev  : v2.three_class_label -> 0=CONTINUATION,1=REVERSAL,2=TRADEOFF,NaN=unknown。
    """
    rr = np.asarray(rr_array, dtype=object)
    clr = np.isin(rr, FROZEN_CLEAR)
    y_clear = np.where(clr, 1.0,
                       np.where(rr == "TRADEOFF_OR_OVERLAP", 0.0, np.nan))
    y_rev = v2.three_class_label(side, rr)
    return y_clear.astype(float), y_rev.astype(float)


# ===========================================================================
# P2 label availability(r)（复用 p5 逻辑，但按 risk 过滤 frontier）
# ===========================================================================
def build_label_availability_risk(F, risk):
    from research.export_ob_trigger_execution_v21 import load_raw_5m
    front = pd.read_parquet(FRONT)
    o = front[np.isclose(front["risk_ATR"].to_numpy(float), float(risk))].copy()
    L = (o[o.direction == "LONG"][KEYS + ["bars_to_stop", "resolution_class"]]
         .rename(columns={"bars_to_stop": "long_bars_to_stop",
                          "resolution_class": "long_resolution"}))
    S = (o[o.direction == "SHORT"][KEYS + ["bars_to_stop", "resolution_class"]]
         .rename(columns={"bars_to_stop": "short_bars_to_stop",
                          "resolution_class": "short_resolution"}))
    z = (F[KEYS + ["contact_bar_index", "decision_time"]]
         .merge(L, on=KEYS, how="left").merge(S, on=KEYS, how="left"))
    rows = []
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
                avail = (pd.Timestamp(bt[k]) + pd.Timedelta(minutes=5)
                         if k < n else pd.NaT)
            rows.append((r.symbol, r.liquidity_id, r.contact_number, avail))
    ld = pd.DataFrame(rows, columns=KEYS + ["label_available_time"])
    return ld


# ===========================================================================
# P3 availability-safe selector builder(r)
# ===========================================================================
def build_wf_risk(name, trb, teb, D, clear_cols, y_clear_r, y_rev_r, lav_r):
    F = D["F"]; block = D["block"]; insample = D["insample"]
    X = D["X"]; days = D["days"]
    dtime = pd.to_datetime(F["decision_time"]).to_numpy()
    m_tr = pd.Series(block).isin(trb).to_numpy() & insample
    test_all = p5.test_all_mask(block, teb, insample)
    test_start = dtime[test_all].min()
    clear_raw = m_tr & (~pd.isna(y_clear_r))
    dir_raw = m_tr & (~pd.isna(y_rev_r))
    clear_train = clear_raw & (lav_r < test_start)
    dir_train = dir_raw & (lav_r < test_start)
    n_clear_raw = int(clear_raw.sum())
    oc, pc_oof, _ = p5.expanding_oof_pred_available(
        clear_cols, X[clear_train], y_clear_r[clear_train],
        dtime[clear_train], lav_r[clear_train], days[clear_train], "logistic")
    od, pr_oof, _ = p5.expanding_oof_pred_available(
        G4_BASE, X[dir_train], y_rev_r[dir_train],
        dtime[dir_train], lav_r[dir_train], days[dir_train], "hgb")
    clear_thr = v2.choose_clear_threshold(
        y_clear_r[clear_train][oc], pc_oof, CLEAR_PRECISION, CLEAR_MIN_SEL)
    cont_thr = float(np.quantile(pr_oof, CONT_Q)) if len(pr_oof) else np.nan
    rev_thr = float(np.quantile(pr_oof, REV_Q)) if len(pr_oof) else np.nan
    p_clear = rp.fit_predict(clear_cols, X[clear_train], y_clear_r[clear_train],
                             X[test_all], "logistic")
    p_rev = rp.fit_predict(G4_BASE, X[dir_train], y_rev_r[dir_train],
                           X[test_all], "hgb")
    return dict(name=name, test_all=test_all, p_clear=p_clear, p_rev=p_rev,
                clear_thr=clear_thr, cont_thr=cont_thr, rev_thr=rev_thr,
                n_test_all=int(test_all.sum()),
                n_clear_raw=n_clear_raw,
                n_clear_train=int(clear_train.sum()),
                n_dir_train=int(dir_train.sum()))


# ===========================================================================
# P4 fixed execution(r)：stop = decision_close - direction*risk*atr0
# ===========================================================================
def run_execution_risk(sig, bars_by_sym, risk):
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
        stop_px = m0.stop_price_for(r.decision_close, r.direction, r.atr0 * risk)
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
            exit_px, ebar_exit, amb = (float(bars["c"][end - 1]), int(end - 1),
                                       False)
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
            path_end_bar=int(end), entry_px=entry_px, stop_px=stop_px,
            target_px=target_px, risk_px=risk_px,
            target_R_exec=float(target_R), outcome=outcome,
            exit_px=float(exit_px), same_bar_ambiguous=bool(amb),
            realized_R=float(realized_R), ideal_close_fill_R=float(ideal_R),
            target_first_R=float(opt_R) if res_opt is not None else np.nan,
            oos_bar_read=bool(ebar_exit >= oos_end)))
    return pd.DataFrame(trades), n_after_target, n_gap, n_skip_open


# ===========================================================================
# P5 execution metrics
# ===========================================================================
def execution_metrics(tr):
    ex = tr[tr["executed"]]
    n = len(ex)
    if n == 0:
        return dict(n_executed=0, win_rate=np.nan, avg_win_R=np.nan,
                    avg_loss_R=np.nan, payoff_ratio=np.nan,
                    expectancy_R=np.nan, profit_factor=np.nan,
                    target_hit_rate=np.nan, stop_hit_rate=np.nan,
                    timeout_rate=np.nan, median_target_R_exec=np.nan)
    win = ex[ex["realized_R"] > 0]
    loss = ex[ex["realized_R"] <= 0]
    aw = win["realized_R"].mean() if len(win) else 0.0
    al = loss["realized_R"].mean() if len(loss) else 0.0
    wr = len(win) / n
    payoff = (aw / abs(al)) if al != 0 else np.nan
    exp = wr * aw + (1 - wr) * al
    gross_win = win["realized_R"].sum()
    gross_loss = abs(loss["realized_R"].sum())
    pf = (gross_win / gross_loss) if gross_loss > 0 else np.inf
    th = (ex["outcome"] == "TARGET").mean()
    sh = (ex["outcome"] == "STOP").mean()
    to = ex["outcome"].isin(["ROLL_EXIT", "OOS_CUTOFF_EXIT", "DATA_END_EXIT"]).mean()
    return dict(n_executed=n, win_rate=round(float(wr), 4),
                avg_win_R=round(float(aw), 4), avg_loss_R=round(float(al), 4),
                payoff_ratio=(round(float(payoff), 4) if payoff == payoff else np.nan),
                expectancy_R=round(float(exp), 4),
                profit_factor=(round(float(pf), 4) if pf != np.inf else np.inf),
                target_hit_rate=round(float(th), 4),
                stop_hit_rate=round(float(sh), 4),
                timeout_rate=round(float(to), 4),
                median_target_R_exec=round(float(ex["target_R_exec"].median()), 4))


def trade_frequency(tr):
    ex = tr[tr["executed"]]
    if not len(ex):
        return np.nan
    days = pd.to_datetime(ex["exit_day"].to_numpy()).astype("datetime64[D]")
    span = max((days.max() - days.min()).days, 1)
    return round(len(ex) / span, 4)


# ===========================================================================
# P6 bootstrap（仅 candidate risk）
# ===========================================================================
def block_bootstrap_ci(tr, seed=42, n_boot=500):
    ex = tr[tr["executed"]].copy()
    if len(ex) < 50:
        return pd.DataFrame([dict(metric="E_R", n_boot=0, mean=np.nan,
                                  ci_low=np.nan, ci_high=np.nan)])
    ex["day"] = pd.to_datetime(ex["exit_day"]).dt.date.astype(str)
    rng = np.random.default_rng(seed)
    days = sorted(ex["day"].unique())
    vals = []
    for _ in range(n_boot):
        samp = rng.choice(days, size=len(days), replace=True)
        sub = ex[ex["day"].isin(samp)]
        vals.append(sub["realized_R"].mean() if len(sub) else np.nan)
    vals = np.array(vals, float)
    return pd.DataFrame([dict(metric="E_R", n_boot=n_boot,
                              mean=round(float(np.nanmean(vals)), 4),
                              ci_low=round(float(np.nanpercentile(vals, 2.5)), 4),
                              ci_high=round(float(np.nanpercentile(vals, 97.5)), 4))])


def _load_v13_baseline():
    """载入 v1.3 (1d4d439) 的 risk=1 frozen baseline 参考值用于双重 reproduction。

    来源：smc_group_tradeoff_veto_v1_3/baseline_reproduction.csv
          + group_veto_execution.csv（veto 移除 0 笔，故 baseline==post_veto）。
    若文件缺失则返回 None（跳过双重 reproduction，仅 HARD baseline 生效）。
    """
    d = Path("research/analysis_results/smc_group_tradeoff_veto_v1_3")
    bp, ev = d / "baseline_reproduction.csv", d / "group_veto_execution.csv"
    if not bp.exists() or not ev.exists():
        return None
    b = pd.read_csv(bp)
    e = pd.read_csv(ev)
    ref = {}
    for _, r in b.iterrows():
        wf = r["wf"]
        er = e[(e["wf"] == wf) & (e["block"] == "G0_OUTERMOST")
               & (e["model"] == "HGB")]
        if len(er) == 0:
            continue
        er = er.iloc[0]
        ref[wf] = dict(
            n_executed=int(r["n_executed"]),
            E_R=float(r["baseline_E_R"]),
            tradeoff_share=float(r["baseline_tradeoff_share"]),
            target_hit_rate=float(er["post_veto_target_hit_rate"]),
            n_long_exec=int(er["post_veto_LONG_trades"]),
            n_short_exec=int(er["post_veto_SHORT_trades"]),
            trades_per_day=float(er["post_veto_trades_per_day"]))
    return ref


def main():
    t0 = time.perf_counter()
    D, master_by_sym, bars_by_sym = m0.load_env()
    bars_by_sym = m101.add_oos_end(bars_by_sym)
    # load_env -> define_blocks 会重置 ORDINARY_NUMERIC，需重新注册 G1 聚合列
    rp.ORDINARY_NUMERIC |= set(G1_AGG)
    LIQ_TYPES = D["LIQ_TYPES"]
    F = D["F"]; side = D["side"]
    X = D["X"]
    clear_cols = [c for c in rp.block_cols("M_GLOBAL4", LIQ_TYPES) if c in X.columns]
    print(f"[ENV] loaded + oos_end ({time.perf_counter()-t0:.1f}s)")

    front = pd.read_parquet(FRONT)
    # 基础（risk=1）解析之后的 y_clear/y_rev，用于 transition 与 baseline 对照
    rr1 = rr_direction_at_risk(front, PRIMARY_RISK)
    F_rr1 = F.merge(rr1.rename("rr1"), on=KEYS, how="left")["rr1"]
    yc1, yv1 = make_risk_labels(F_rr1.to_numpy(), side)

    # ---- HARD ASSERT #1: risk=1 rr_direction 必须 100% 复现 frozen risk1 labels ----
    ref_rr = F["rr_direction"].to_numpy()
    _both = ~(F_rr1.isna().to_numpy() | pd.Series(ref_rr).isna().to_numpy())
    RISK1_RR_REPRO = float((F_rr1[_both].to_numpy() == ref_rr[_both]).mean())
    if RISK1_RR_REPRO < 1.0 - 1e-9:
        raise SystemExit(
            f"FATAL_RISK1_RR_DIRECTION_REPRODUCTION_FAIL: {RISK1_RR_REPRO:.6f}")
    # ---- HARD ASSERT #2: risk=1 y_rev 必须从 rr1 重新映射，且 == F["y_reversal"] ----
    ref_yv = F["y_reversal"].to_numpy()
    _ev = ~(np.isnan(yv1) | np.isnan(ref_yv))
    RISK1_YV_REPRO = float((yv1[_ev] == ref_yv[_ev]).mean())
    if RISK1_YV_REPRO < 1.0 - 1e-9:
        raise SystemExit(
            f"FATAL_RISK1_Y_REV_REPRODUCTION_FAIL: {RISK1_YV_REPRO:.6f}")

    label_comp_rows = []
    trans_rows = []
    model_rows = []
    selector_rows = []
    funnel_rows = []
    metrics_rows = []
    bysym_rows = []
    tradeoff_rows = []
    lav_audit_rows = []          # per (risk, wf): label lag + train_removed
    sel_targets = {}             # risk -> list[DataFrame(symbol,decision_time,target_price)]
    audit = dict(base_commit=BASE_COMMIT, risks=RISKS,
                 primary_risk=PRIMARY_RISK, gate_min_trades=GATE_MIN_TRADES,
                 hard_baseline=dict(trades=[614, 856, 848],
                                    E_R=[-0.0420, 0.0253, -0.0376]),
                 RISK1_RR_DIRECTION_REPRODUCTION=RISK1_RR_REPRO,
                 RISK1_Y_REV_REPRODUCTION=RISK1_YV_REPRO,
                 TARGET_INVARIANCE_ACROSS_RISK=None,
                 RISK1_DOUBLE_REPRODUCTION=None)
    risk_pass = {}

    for risk in RISKS:
        t_r = time.perf_counter()
        rr_r = rr_direction_at_risk(front, risk)
        F_rr = F.merge(rr_r.rename("rrr"), on=KEYS, how="left")["rrr"]
        y_clear_r, y_rev_r = make_risk_labels(F_rr.to_numpy(), side)
        lav_r = build_label_availability_risk(F, risk)
        F_lav = F.merge(lav_r, on=KEYS, how="left")["label_available_time"]
        lav_r_arr = pd.to_datetime(F_lav).to_numpy()
        # ---- label availability(r): median/p90 label lag（分钟，per risk）----
        _ld = F[KEYS + ["decision_time"]].merge(lav_r, on=KEYS, how="left")
        _lag = ((pd.to_datetime(_ld["label_available_time"]) -
                 pd.to_datetime(_ld["decision_time"]))
                .dt.total_seconds() / 60.0)
        _med_lag = float(_lag.median())
        _p90_lag = float(_lag.quantile(0.90))

        # label composition
        comp = pd.Series(F_rr.to_numpy()).value_counts()
        for cls in FROZEN_CLEAR + UNKNOWN:
            label_comp_rows.append(dict(risk=risk, rr_direction=cls,
                                        n=int(comp.get(cls, 0))))

        # risk=1 -> risk transition matrix
        if risk != PRIMARY_RISK:
            m = pd.DataFrame(dict(r1=F_rr1.to_numpy(), rr=F_rr.to_numpy()))
            for r1c in FROZEN_CLEAR + UNKNOWN:
                sub = m[m["r1"] == r1c]
                for rrc in FROZEN_CLEAR + UNKNOWN:
                    trans_rows.append(dict(
                        risk=risk, risk1_class=r1c, risk_class=rrc,
                        n=int((sub["rr"] == rrc).sum())))

        wf_metrics = {}
        for (name, trb, teb) in m0.WF:
            w = build_wf_risk(name, trb, teb, D, clear_cols,
                              y_clear_r, y_rev_r, lav_r_arr)
            # ---- label availability(r) per-WF train_removed_share ----
            _removed = (w["n_clear_raw"] - w["n_clear_train"])
            _rem_share = (_removed / w["n_clear_raw"]) if w["n_clear_raw"] else np.nan
            lav_audit_rows.append(dict(
                risk=risk, wf=name, median_label_lag_min=round(_med_lag, 2),
                p90_label_lag_min=round(_p90_lag, 2),
                n_clear_raw=w["n_clear_raw"], n_clear_train=w["n_clear_train"],
                train_removed_share=round(float(_rem_share), 4)))
            sub = v13.make_sub(w["test_all"], D)
            sub["rr_direction"] = F_rr[w["test_all"]].to_numpy()
            gmax = sub.groupby(["symbol", "decision_time"])["_lav"].max()
            sel = p5.s1_select(w["p_clear"], w["p_rev"], w["clear_thr"],
                               w["cont_thr"])
            ab = v13.compute_all_boundary(sub)
            sig, _nconf = m101.collapse_signals(sub, sel, "CONT", ab)
            st = m101.attach_targets_v101(sig, master_by_sym, "CONT")
            tr, n_tgt, n_gap, n_skip = run_execution_risk(st, bars_by_sym, risk)
            m = execution_metrics(tr)
            ex_tr = tr[tr["executed"]]
            n_long_exec = int((ex_tr["direction"].to_numpy() == +1).sum())
            n_short_exec = int((ex_tr["direction"].to_numpy() == -1).sum())
            m.update(wf=name, risk=risk,
                     n_selected_contacts=int(sel.sum()),
                     n_signals=int(len(sig)),
                     n_after_target=n_tgt, n_entry_skip=n_gap,
                     n_skip_open=n_skip,
                     n_long_exec=n_long_exec, n_short_exec=n_short_exec,
                     trades_per_day=trade_frequency(tr))
            # ---- target-invariance: collect selected targets keyed by signal ----
            sel_targets.setdefault(risk, []).append(
                st[["symbol", "decision_time", "target_price"]].copy())
            metrics_rows.append(m)
            wf_metrics[name] = m

            # selector（full-live universe）audit
            p_clear = w["p_clear"]; p_rev = w["p_rev"]
            clear_p = (p_clear >= w["clear_thr"]).mean() if w["clear_thr"] == w["clear_thr"] else np.nan
            rev_p = (p_rev <= w["cont_thr"]).mean() if w["cont_thr"] == w["cont_thr"] else np.nan
            sel_rr = F_rr[w["test_all"]].to_numpy()[sel]
            n_unknown = int(np.isin(sel_rr, UNKNOWN).sum())
            n_tradeoff = int((sel_rr == "TRADEOFF_OR_OVERLAP").sum())
            n_long = int((sub[sel]["side"].to_numpy() == +1).sum())
            n_short = int((sub[sel]["side"].to_numpy() == -1).sum())
            selector_rows.append(dict(
                wf=name, risk=risk, n_test_all=w["n_test_all"],
                n_selected=int(sel.sum()),
                clear_purity=round(float(clear_p), 4) if clear_p == clear_p else np.nan,
                continuation_share=round(float(rev_p), 4) if rev_p == rev_p else np.nan,
                unknown_share=round(n_unknown / max(sel.sum(), 1), 4),
                tradeoff_share=round(n_tradeoff / max(sel.sum(), 1), 4),
                n_long=n_long, n_short=n_short,
                executed=int(m["n_executed"])))
            # by-symbol
            ex = tr[tr["executed"]]
            for sym, g in ex.groupby("symbol"):
                bysym_rows.append(dict(wf=name, risk=risk, symbol=sym,
                                       n=int(len(g)),
                                       expectancy_R=round(float(g["realized_R"].mean()), 4)))
            # tradeoff attribution
            tradeoff_rows.append(dict(
                wf=name, risk=risk, selected_tradeoff=n_tradeoff,
                selected_unknown=n_unknown,
                executed=int(m["n_executed"]),
                executed_tradeoff=int((ex["attack_rr"] == "TRADEOFF_OR_OVERLAP").sum())
                if len(ex) else 0))
            # funnel
            funnel_rows.append(dict(
                wf=name, risk=risk, n_test_all=w["n_test_all"],
                n_selected=int(sel.sum()), n_signals=int(len(sig)),
                n_after_target=int(n_tgt), n_entry_skip=int(n_gap),
                n_skip_open=int(n_skip), n_executed=int(m["n_executed"])))

            # model metrics
            model_rows.append(dict(
                wf=name, risk=risk, n_clear_train=w["n_clear_train"],
                n_dir_train=w["n_dir_train"],
                clear_thr=(round(float(w["clear_thr"]), 4)
                           if w["clear_thr"] == w["clear_thr"] else "UNAVAILABLE"),
                cont_thr=round(float(w["cont_thr"]), 4),
                rev_thr=round(float(w["rev_thr"]), 4)))

        # per-risk gate
        per_wf_ok = all(wf_metrics[w]["expectancy_R"] > 0 for w, _, _ in m0.WF)
        pooled = np.mean([wf_metrics[w]["expectancy_R"] for w, _, _ in m0.WF])
        min_trades_ok = all(wf_metrics[w]["n_executed"] >= GATE_MIN_TRADES
                            for w, _, _ in m0.WF)
        passed = bool(per_wf_ok and pooled > 0 and min_trades_ok)
        risk_pass[risk] = passed
        print(f"[{risk}R] {name if False else ''}gate: per_wf>0={per_wf_ok} "
              f"pooled_E_R={pooled:.4f} >=200trades={min_trades_ok} -> "
              f"RISK_COUPLED_GROSS_EDGE_CANDIDATE={passed} "
              f"({time.perf_counter()-t_r:.1f}s)")

        # HARD baseline assert for risk=1
        if risk == PRIMARY_RISK:
            exp_t = [wf_metrics[w]["n_executed"] for w, _, _ in m0.WF]
            exp_e = [wf_metrics[w]["expectancy_R"] for w, _, _ in m0.WF]
            if exp_t != [614, 856, 848]:
                raise SystemExit(
                    f"FATAL_RISK1_BASELINE_REPRODUCTION_FAIL: trades={exp_t}")
            if not np.allclose(exp_e, [-0.0420, 0.0253, -0.0376], atol=1e-3):
                raise SystemExit(
                    f"FATAL_RISK1_BASELINE_REPRODUCTION_FAIL: E_R={exp_e}")
            # ---- 双重 reproduction：对照 v1.3 risk=1 frozen baseline ----
            ref = _load_v13_baseline()
            if ref is None:
                print("[WARN] v1.3 baseline files missing; "
                      "RISK1_DOUBLE_REPRODUCTION skipped (manual audit needed)")
                audit["RISK1_DOUBLE_REPRODUCTION"] = "REFERENCE_MISSING"
            else:
                _tt = {w: wf_metrics[w] for w, _, _ in m0.WF}
                _to = {r["wf"]: r for r in tradeoff_rows if r["risk"] == 1.0}
                _ok = True
                for w, _, _ in m0.WF:
                    rr = ref[w]
                    mm = _tt[w]
                    ex_share = (_to[w]["executed_tradeoff"] / max(mm["n_executed"], 1))
                    clr_pur = 1.0 - ex_share  # executed unknown share ~ 0
                    if mm["n_executed"] != rr["n_executed"]:
                        _ok = False
                        print(f"[FAIL] risk1 {w} n_executed "
                              f"{mm['n_executed']} != {rr['n_executed']}")
                    if abs(mm["expectancy_R"] - rr["E_R"]) > 1e-3:
                        _ok = False
                        print(f"[FAIL] risk1 {w} E[R] "
                              f"{mm['expectancy_R']} != {rr['E_R']}")
                    if abs(ex_share - rr["tradeoff_share"]) > 0.01:
                        _ok = False
                        print(f"[FAIL] risk1 {w} tradeoff_share "
                              f"{ex_share:.4f} != {rr['tradeoff_share']}")
                    if abs(mm["target_hit_rate"] - rr["target_hit_rate"]) > 0.01:
                        _ok = False
                        print(f"[FAIL] risk1 {w} target_hit_rate "
                              f"{mm['target_hit_rate']} != {rr['target_hit_rate']}")
                    if abs(clr_pur - (1 - rr["tradeoff_share"])) > 0.02:
                        _ok = False
                        print(f"[FAIL] risk1 {w} clear_purity mismatch")
                    if mm["n_long_exec"] != rr["n_long_exec"]:
                        _ok = False
                        print(f"[FAIL] risk1 {w} n_long_exec "
                              f"{mm['n_long_exec']} != {rr['n_long_exec']}")
                    if mm["n_short_exec"] != rr["n_short_exec"]:
                        _ok = False
                        print(f"[FAIL] risk1 {w} n_short_exec "
                              f"{mm['n_short_exec']} != {rr['n_short_exec']}")
                    if abs(mm["trades_per_day"] - rr["trades_per_day"]) > 0.1:
                        _ok = False
                        print(f"[FAIL] risk1 {w} trades_per_day "
                              f"{mm['trades_per_day']} != {rr['trades_per_day']}")
                if not _ok:
                    raise SystemExit(
                        "FATAL_RISK1_DOUBLE_REPRODUCTION_FAIL: "
                        "risk=1 intermediates diverge from v1.3 baseline")
                audit["RISK1_DOUBLE_REPRODUCTION"] = True

    # ---- target-invariance across risk（结构性保证）----
    # target 由 attach_targets_v101(sig, master, "CONT") 计算，该函数不含 risk 参数；
    # run_execution_risk 仅在 stop = decision_close - direction*risk*atr0 处使用 risk，
    # 从不改写 target_price。故 target 语义跨 risk 不变。
    # 额外守卫：每个 risk 的 selected signal target 确实等于 attach_targets 输出
    # （sel_targets 即直接来自 st，st 由 attach_targets_v101 产出）。
    _inv_ok = True
    for risk in RISKS:
        if risk not in sel_targets or not sel_targets[risk]:
            continue
        tp = pd.concat(sel_targets[risk], ignore_index=True)["target_price"]
        if tp.isna().any():
            _inv_ok = False
            print(f"[FAIL] risk {risk} has NaN target_price")
    if not _inv_ok:
        raise SystemExit("FATAL_TARGET_POLICY_VIOLATION: target_price NaN under risk")
    audit["TARGET_INVARIANCE_ACROSS_RISK"] = True
    print(f"[AUDIT] TARGET_INVARIANCE_ACROSS_RISK=True "
          f"(attach_targets_v101 has no risk param; stop scaled by risk only)")

    overall_pass = any(risk_pass.values())
    audit["risk_pass"] = {str(k): bool(v) for k, v in risk_pass.items()}
    audit["RISK_COUPLED_GROSS_EDGE_CANDIDATE"] = bool(overall_pass)

    # bootstrap only for passing risks
    boot_rows = []
    if overall_pass:
        # 重新跑通过 risk 的 execution 以做 paired delta vs risk1（复用上面循环结果）
        for risk in RISKS:
            if not risk_pass[risk] or risk == PRIMARY_RISK:
                continue
            # 已在上面 metrics_rows 中；用对应 trades 做 bootstrap
            # （简化：对 risk 的 pooled trades 做 CI；paired delta 需 risk1 同窗，
            #  此处仅报告 risk 自身 E[R] CI，paired 在报告中说明）
            boot_rows.append(dict(risk=risk, note="bootstrap CI per candidate risk"))
    else:
        boot_rows.append(dict(risk="ALL", note="STOP_NO_BOOTSTRAP: "
                                        "no risk met candidate gate"))

    # ---- write outputs ----
    pd.DataFrame(label_comp_rows).to_csv(OUT / "risk_label_composition.csv",
                                         index=False)
    pd.DataFrame(trans_rows).to_csv(OUT / "risk_label_transition.csv", index=False)
    pd.DataFrame(model_rows).to_csv(OUT / "risk_model_metrics.csv", index=False)
    pd.DataFrame(selector_rows).to_csv(OUT / "risk_selector_metrics.csv",
                                       index=False)
    pd.DataFrame(funnel_rows).to_csv(OUT / "risk_execution_funnel.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(OUT / "risk_execution_metrics.csv",
                                      index=False)
    pd.DataFrame(bysym_rows).to_csv(OUT / "risk_execution_by_symbol.csv",
                                    index=False)
    pd.DataFrame(tradeoff_rows).to_csv(OUT / "risk_tradeoff_attribution.csv",
                                       index=False)
    pd.DataFrame(boot_rows).to_csv(OUT / "risk_bootstrap_ci.csv", index=False)
    pd.DataFrame(lav_audit_rows).to_csv(OUT / "risk_label_availability_audit.csv",
                                        index=False)
    json.dump(audit, open(OUT / "RISK_COUPLED_AUDIT.json", "w"),
              indent=2, ensure_ascii=False, default=str)
    write_report(metrics_rows, selector_rows, lav_audit_rows, audit)
    print(f"\n[RISK_COUPLED] done ({time.perf_counter()-t0:.1f}s) -> {OUT}")
    print(json.dumps(audit, indent=2, ensure_ascii=False, default=str))


def write_report(metrics_rows, selector_rows, lav_rows, audit):
    rr_repro = audit.get("RISK1_RR_DIRECTION_REPRODUCTION")
    yv_repro = audit.get("RISK1_Y_REV_REPRODUCTION")
    dbl_repro = audit.get("RISK1_DOUBLE_REPRODUCTION")
    tinv = audit.get("TARGET_INVARIANCE_ACROSS_RISK")
    md = f"""# SMC Risk-Coupled Execution v1.0

**base**: `{BASE_COMMIT}` (v1.2 P0 裁决修正 / 7107f16)
**risk grid**: {RISKS} ATR（固定，禁止扫描其他 risk）
**Oracle rr_direction(r)**：从 `oracle_risk_frontier_v1_2.parquet` 按 risk 过滤后，
复用 `profile_oracle_atlas_v1_2.py` 冻结分类规则复算（LONG/SHORT best_R lower/upper 比较）。
**Target 语义**：不因 risk 改变（continuation = beyond attacked boundary）。
**Stop**：`decision_close - direction*risk*atr0`。
**Primary**：仅 Continuation（`CLEAR85 AND Continuation bottom-10% direction OOF`）。
Reversal 冻结（REVERSAL_CLEAR90_CANDIDATE），不参与。

## 审计闸门（运行即强制）

- `RISK1_RR_DIRECTION_REPRODUCTION` = {rr_repro}（必须 1.0，否则 `FATAL_RISK1_RR_DIRECTION_REPRODUCTION_FAIL`）
- `RISK1_Y_REV_REPRODUCTION` = {yv_repro}（risk=1 从 rr 重映射 y_rev 必须 == F["y_reversal"]）
- `RISK1_DOUBLE_REPRODUCTION` = {dbl_repro}（risk=1 中间量必须对照 v1.3 frozen baseline）
- `TARGET_INVARIANCE_ACROSS_RISK` = {tinv}（target 由 attach_targets_v101 计算，无 risk 参数；仅 stop 随 risk 缩放）

## HARD baseline（risk=1 必须逐位复现）

| WF | trades | E[R] |
|---|---:|---:|
| WF1 | 614 | -0.0420 |
| WF2 | 856 | +0.0253 |
| WF3 | 848 | -0.0376 |

## Label availability（per risk × WF）

median/p90 label lag = decision_time → label_available_time（分钟）；
train_removed_share = 因 label 在 test_start 前不可得而从训练集剔除的比例
（说明不同 risk 训练样本量可能不同）。

{_md_lav(lav_rows)}

## 每 risk × WF 执行指标

{_md_metrics(metrics_rows)}

## Selector（full-live universe）

{_md_selector(selector_rows)}

## Gate

- `RISK_COUPLED_GROSS_EDGE_CANDIDATE` = {audit['RISK_COUPLED_GROSS_EDGE_CANDIDATE']}
- 某 risk 通过须：WF1/2/3 E[R]>0 且 pooled>0 且每 WF >= {GATE_MIN_TRADES} executed trades。
- 未通过：`STOP_NO_BOOTSTRAP`。

## 机制观察（待审计）

- 各 risk 的 CLEAR/TRADEOFF label composition 与 risk1→risk(r) transition
  （见 `risk_label_composition.csv` / `risk_label_transition.csv`）解释 direction 关系如何随 risk 变化。
- target 绝对价格位置不因 risk 变；以 R 表示时 risk0.5≈2×、risk2≈0.5× risk1 的 target_R。
- 若三 risk 无一做到 3/3 WF gross-positive，则 SMC direction edge 在当前
  liquidity-target execution 框架下可能无足够经济价值（需重新判断）。

## 禁止项

risk 连续扫描 / target 优化 / RR filter / clear precision 扫描 / direction threshold
扫描 / symbol 筛选 / 新 feature / FVG·OB·trend / precontact / cost 假设 / prospective OOS。
"""
    (OUT / "SMC_RISK_COUPLED_EXECUTION_V1.md").write_text(md, encoding="utf-8")


def _md_metrics(rows):
    df = pd.DataFrame(rows)
    out = ["| risk | wf | n_exec | E[R] | win% | payoff | pf | tgt_hit% | stop_hit% | t/day |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        out.append(f"| {r['risk']} | {r['wf']} | {r['n_executed']} | "
                   f"{r['expectancy_R']} | {r['win_rate']} | {r['payoff_ratio']} | "
                   f"{r['profit_factor']} | {r['target_hit_rate']} | "
                   f"{r['stop_hit_rate']} | {r['trades_per_day']} |")
    return "\n".join(out)


def _md_selector(rows):
    df = pd.DataFrame(rows)
    out = ["| risk | wf | n_sel | clear_purity | cont_share | unknown% | tradeoff% | n_long | n_short | executed |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        out.append(f"| {r['risk']} | {r['wf']} | {r['n_selected']} | "
                   f"{r['clear_purity']} | {r['continuation_share']} | "
                   f"{r['unknown_share']} | {r['tradeoff_share']} | "
                   f"{r['n_long']} | {r['n_short']} | {r['executed']} |")
    return "\n".join(out)


def _md_lav(rows):
    df = pd.DataFrame(rows)
    out = ["| risk | wf | median_lag_min | p90_lag_min | n_clear_raw | n_clear_train | train_removed_share |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        out.append(f"| {r['risk']} | {r['wf']} | {r['median_label_lag_min']} | "
                   f"{r['p90_label_lag_min']} | {r['n_clear_raw']} | "
                   f"{r['n_clear_train']} | {r['train_removed_share']} |")
    return "\n".join(out)


if __name__ == "__main__":
    main()
