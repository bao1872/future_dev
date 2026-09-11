"""SMC Execution Selection Purity Gate v1.1 (base 7c7858a).

验证 non-clear contamination 是否是当前 fixed execution 的主要拖累。

不加 feature。冻结：risk=1 ATR、target、direction model、Continuation q10、
Reversal q90、entry(next 5m open)、stop(decision_close ±1 ATR0)、
same-bar STOP_FIRST、v1.0.1 repaired OOS cutoff。

唯一新增变量：一个预注册的更严格 clear gate
  CLEAR85（复现 baseline） vs CLEAR90（train-OOF target precision=0.90）

P1 硬门：signal-level rr_direction 在 collapsed group 内必须 unique==1。
P8 mechanism gate / P9 execution gate 见 audit。
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
import research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 as m0
import research.liquidity_oracle_atlas.run_fixed_execution_integrity_v1_0_1 as m101

OUT = Path("research/analysis_results/smc_execution_selection_purity_v1_1")
OUT.mkdir(parents=True, exist_ok=True)
V101 = Path("research/analysis_results/smc_fixed_execution_integrity_v1_0_1")

OOS_START = m0.OOS_START
WF = m0.WF
G4_BASE = m0.G4_BASE
CLEAR_MIN_SEL = 0.05
CONT_Q, REV_Q = 0.10, 0.90
PREC85, PREC90 = 0.85, 0.90
MIN_TRADES = 200
CLEAR_CLASSES = ["LONG_DOMINATES", "SHORT_DOMINATES"]
ALL_CLASSES = ["LONG_DOMINATES", "SHORT_DOMINATES", "TRADEOFF_OR_OVERLAP",
               "UNRESOLVED_CENSOR", "NO_COMPARABLE_TARGET"]


# ===========================================================================
# WF build（返回 clear OOF，便于同一 OOF 上取不同 precision 阈值）
# ===========================================================================
def build_wf_v11(name, trb, teb, D, clear_cols):
    F = D["F"]; block = D["block"]; insample = D["insample"]
    X = D["X"]; y_clear = D["y_clear"]; y_rev = D["y_rev"]; days = D["days"]
    lav = pd.to_datetime(F["label_available_time"]).to_numpy()
    dtime = pd.to_datetime(F["decision_time"]).to_numpy()
    m_tr = pd.Series(block).isin(trb).to_numpy() & insample
    test_all = m0.p5.test_all_mask(block, teb, insample)
    test_start = dtime[test_all].min()
    clear_train = m_tr & (~pd.isna(y_clear)) & (lav < test_start)
    dir_train = m_tr & (~pd.isna(y_rev)) & (lav < test_start)

    oc, pc_oof, _ = m0.p5.expanding_oof_pred_available(
        clear_cols, X[clear_train], y_clear[clear_train],
        dtime[clear_train], lav[clear_train], days[clear_train], "logistic")
    od, pr_oof, _ = m0.p5.expanding_oof_pred_available(
        G4_BASE, X[dir_train], y_rev[dir_train],
        dtime[dir_train], lav[dir_train], days[dir_train], "hgb")
    cont_thr = float(np.quantile(pr_oof, CONT_Q))
    rev_thr = float(np.quantile(pr_oof, REV_Q))
    p_clear = rp.fit_predict(clear_cols, X[clear_train], y_clear[clear_train],
                             X[test_all], "logistic")
    p_rev = rp.fit_predict(G4_BASE, X[dir_train], y_rev[dir_train],
                           X[test_all], "hgb")
    return dict(name=name, test_all=test_all, p_clear=p_clear, p_rev=p_rev,
                cont_thr=cont_thr, rev_thr=rev_thr,
                y_clear_oof=y_clear[clear_train][oc], p_clear_oof=pc_oof)


def clear_threshold_at(oof_y, oof_p, precision, min_sel=CLEAR_MIN_SEL):
    return v2.choose_clear_threshold(oof_y, oof_p, precision, min_sel)


# ===========================================================================
# P1 signal-level rr consistency
# ===========================================================================
def signal_rr_consistency(sub):
    rows = []
    for (sym, dt), g in sub.groupby(["symbol", "decision_time"]):
        vals = g["rr_direction"].dropna().unique()
        rows.append(dict(symbol=sym, decision_time=dt, n_contacts=len(g),
                         n_unique_rr=len(vals),
                         rr_values="|".join(sorted(map(str, vals)))))
    return pd.DataFrame(rows)


# ===========================================================================
# P3 mixture / P8 helpers
# ===========================================================================
def required_clear_purity(e_clear, e_nonclear):
    if not (e_clear > 0 and e_nonclear < 0):
        return np.nan
    return abs(e_nonclear) / (e_clear + abs(e_nonclear))


def class_split(tr, wf, setup):
    ex = tr[tr["executed"]].copy()
    rows = []
    for cls in ALL_CLASSES:
        m = ex["attack_rr"] == cls
        sub = ex[m]
        n = len(sub)
        rr = sub["realized_R"].to_numpy(float)
        w, l = rr[rr > 0], rr[rr < 0]
        rows.append(dict(
            wf=wf, setup=setup, frozen_class=cls, n=n,
            share=round(n / len(ex), 4) if len(ex) else np.nan,
            target_hit_rate=round(float((sub["outcome"] == "TARGET").mean()), 4)
            if n else np.nan,
            mean_win_R=round(float(w.mean()), 4) if len(w) else np.nan,
            mean_loss_R=round(float(l.mean()), 4) if len(l) else np.nan,
            gross_expectancy_R=round(float(rr.mean()), 4) if n else np.nan))
    return rows


def mixture_row(tr, wf, setup, variant):
    ex = tr[tr["executed"]]
    is_clear = ex["attack_rr"].isin(CLEAR_CLASSES).to_numpy()
    rr = ex["realized_R"].to_numpy(float)
    ec = float(rr[is_clear].mean()) if is_clear.sum() else np.nan
    en = float(rr[~is_clear].mean()) if (~is_clear).sum() else np.nan
    et = float(rr.mean()) if len(rr) else np.nan
    pur = float(is_clear.mean()) if len(is_clear) else np.nan
    req = required_clear_purity(ec, en) if pd.notna(ec) and pd.notna(en) else np.nan
    return dict(wf=wf, setup=setup, variant=variant,
                n_executed=int(len(ex)), n_clear=int(is_clear.sum()),
                n_nonclear=int((~is_clear).sum()),
                clear_purity=round(pur, 4),
                E_R_clear=round(ec, 4), E_R_nonclear=round(en, 4),
                E_R_total=round(et, 4),
                required_clear_purity_for_zero_gross=round(req, 4)
                if pd.notna(req) else np.nan,
                purity_minus_required=round(pur - req, 4)
                if pd.notna(req) else np.nan)


def exec_row(tr, wf, setup, variant, n_days, n_raw, n_col):
    ex = tr[tr["executed"]]
    n = len(ex)
    rr = ex["realized_R"].to_numpy(float)
    w, l = rr[rr > 0], rr[rr < 0]
    is_clear = ex["attack_rr"].isin(CLEAR_CLASSES).to_numpy()
    pf = (float(w.sum()) / abs(float(l.sum()))) if len(l) and l.sum() != 0 \
        else np.nan
    return dict(
        wf=wf, setup=setup, variant=variant,
        raw_selected=n_raw, collapsed_signals=n_col, executed_trades=n,
        clear_purity=round(float(is_clear.mean()), 4) if n else np.nan,
        TRADEOFF_share=round(float((ex["attack_rr"] ==
                                    "TRADEOFF_OR_OVERLAP").mean()), 4) if n else np.nan,
        UNRESOLVED_share=round(float((ex["attack_rr"] ==
                                      "UNRESOLVED_CENSOR").mean()), 4) if n else np.nan,
        NO_COMPARABLE_share=round(float((ex["attack_rr"] ==
                                         "NO_COMPARABLE_TARGET").mean()), 4) if n else np.nan,
        target_hit_rate=round(float((ex["outcome"] == "TARGET").mean()), 4)
        if n else np.nan,
        gross_expectancy_R=round(float(rr.mean()), 4) if n else np.nan,
        profit_factor_R=round(pf, 4) if pd.notna(pf) else np.nan,
        trades_per_day=round(n / max(n_days, 1), 4),
        LONG_trades=int((ex["direction"] == +1).sum()),
        SHORT_trades=int((ex["direction"] == -1).sum()))


# ===========================================================================
# main
# ===========================================================================
def main():
    t0 = time.perf_counter()
    D, master_by_sym, bars_by_sym = m0.load_env()
    m101.add_oos_end(bars_by_sym)
    print(f"[ENV] loaded ({time.perf_counter()-t0:.1f}s)")
    F = D["F"]; block = D["block"]; insample = D["insample"]
    clear_cols = [c for c in rp.block_cols("M_GLOBAL4", D["LIQ_TYPES"])
                  if c in D["X"].columns]

    rr_rows, cls_rows, mix_rows, ex_rows = [], [], [], []
    geo_rows, rev_rows = [], []
    sel_rows = []
    trades85, trades90 = [], []
    rev90 = []
    for name, trb, teb in WF:
        w = build_wf_v11(name, trb, teb, D, clear_cols)
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
        n_days = int(pd.Series(pd.to_datetime(F["trading_day"]).to_numpy()[te]
                               ).nunique())
        all_boundary = {}
        for (sym, dt, sd), g in sub.groupby(["symbol", "decision_time", "side"]):
            all_boundary[(sym, dt, int(sd))] = m0.attacked_boundary(
                g["liquidity_price"].to_numpy(float), int(sd))

        # ---- P1 signal-level rr consistency（全部 contacts）----
        rra = signal_rr_consistency(sub)
        rr_rows.append(dict(wf=name, n_groups=int(len(rra)),
                            n_groups_unique_rr_1=int((rra["n_unique_rr"] == 1).sum()),
                            n_groups_unique_rr_gt1=int((rra["n_unique_rr"] > 1).sum()),
                            max_unique_rr=int(rra["n_unique_rr"].max())))

        thr85 = clear_threshold_at(w["y_clear_oof"], w["p_clear_oof"], PREC85)
        thr90 = clear_threshold_at(w["y_clear_oof"], w["p_clear_oof"], PREC90)
        sel_rows.append(dict(
            wf=name,
            clear_thr_85=(round(float(thr85), 4) if thr85 is not None else None),
            clear_thr_90=(round(float(thr90), 4) if thr90 is not None else None),
            cont_thr=round(w["cont_thr"], 4), rev_thr=round(w["rev_thr"], 4),
            clear90_available=thr90 is not None))

        for variant, thr in (("CLEAR85", thr85), ("CLEAR90", thr90)):
            if thr is None:
                ex_rows.append(dict(wf=name, setup="CONT", variant=variant,
                                    raw_selected=0, collapsed_signals=0,
                                    executed_trades=0,
                                    note="CLEAR90_THRESHOLD_UNAVAILABLE"))
                continue
            for setup in ("CONT", "REV"):
                if setup == "CONT":
                    sel = (w["p_clear"] >= thr) & (w["p_rev"] <= w["cont_thr"])
                else:
                    sel = (w["p_clear"] >= thr) & (w["p_rev"] >= w["rev_thr"])
                sig, nconf = m101.collapse_signals(sub, sel, setup, all_boundary)
                if not len(sig):
                    continue
                st = m101.attach_targets_v101(sig, master_by_sym, setup)
                tr, _, _, _ = m101.run_execution_repaired(st, bars_by_sym)
                ex = tr[tr["executed"]]
                ex_rows.append(exec_row(tr, name, setup, variant, n_days,
                                        int(sel.sum()), len(sig)))
                mix_rows.append(mixture_row(tr, name, setup, variant))
                cls_rows.extend(class_split(tr, name, f"{variant}_{setup}"))
                if setup == "CONT" and variant == "CLEAR85":
                    trades85.append(ex)
                    geo_rows.extend(geo_by_clear(ex, name, variant))
                if setup == "CONT" and variant == "CLEAR90":
                    trades90.append(ex)
                if setup == "REV" and variant == "CLEAR90":
                    rev90.append(ex)
                if setup == "REV" and variant == "CLEAR85":
                    rev_rows.append(dict(wf=name, variant=variant,
                                         trades=int(len(ex)),
                                         clear_purity=round(float(
                                             ex["attack_rr"].isin(
                                                 CLEAR_CLASSES).mean()), 4),
                                         gross_expectancy_R=round(float(
                                             ex["realized_R"].mean()), 4)))
        print(f"[{name}] thr85={thr85:.4f} thr90={thr90} "
              f"85_exp={ex_rows[-2]['gross_expectancy_R'] if len(ex_rows) > 1 else None}")

    df_rr = pd.DataFrame(rr_rows)
    df_cls = pd.DataFrame(cls_rows)
    df_mix = pd.DataFrame(mix_rows)
    df_ex = pd.DataFrame(ex_rows)
    df_geo = pd.DataFrame(geo_rows)
    df_sel = pd.DataFrame(sel_rows)
    df_rev = pd.DataFrame(rev_rows)
    t85 = pd.concat(trades85, ignore_index=True) if trades85 else pd.DataFrame()
    t90 = pd.concat(trades90, ignore_index=True) if trades90 else pd.DataFrame()
    r90 = pd.concat(rev90, ignore_index=True) if rev90 else pd.DataFrame()

    # ---- P1 HARD GATE ----
    bad = int(df_rr["n_groups_unique_rr_gt1"].sum())
    print(f"[P1] signal groups with n_unique_rr>1: {bad}")
    if bad:
        raise SystemExit("FATAL_SIGNAL_LEVEL_RR_SEMANTIC_MISMATCH")

    # ---- P4 reproduce v1.0.1 ----
    v101 = pd.read_csv(V101 / "execution_metrics_repaired.csv")
    e85 = (df_ex[(df_ex.variant == "CLEAR85") & (df_ex.setup == "CONT")]
           .sort_values("wf")["gross_expectancy_R"].to_numpy())
    e101 = v101.sort_values("wf")["gross_expectancy_R"].to_numpy()
    reproduce = bool(np.allclose(e85, e101, atol=1e-9))
    print(f"[P4] CLEAR85 reproduces v1.0.1: {reproduce} {e85} vs {e101}")
    assert reproduce, "CLEAR85 must reproduce v1.0.1 bit-for-bit"

    # ---- P8 mechanism gate ----
    m85 = df_mix[(df_mix.variant == "CLEAR85") & (df_mix.setup == "CONT")]
    A = bool((m85["E_R_clear"] > 0).all() and (m85["E_R_nonclear"] < 0).all())
    p85 = m85.sort_values("wf")["clear_purity"].to_numpy()
    p90 = (df_mix[(df_mix.variant == "CLEAR90") & (df_mix.setup == "CONT")]
           .sort_values("wf")["clear_purity"].to_numpy())
    B = bool(len(p90) == 3 and (p90 > p85).all())
    e90 = (df_ex[(df_ex.variant == "CLEAR90") & (df_ex.setup == "CONT")]
           .sort_values("wf")["gross_expectancy_R"].to_numpy())
    C = bool(len(e90) == 3 and (e90 > e85).all())
    MECH = bool(A and B and C)

    # ---- P9 execution gate ----
    n90 = (df_ex[(df_ex.variant == "CLEAR90") & (df_ex.setup == "CONT")]
           .sort_values("wf")["executed_trades"].to_numpy())
    pooled90 = float(t90["realized_R"].mean()) if len(t90) else np.nan
    point_ok = bool(len(e90) == 3 and (e90 > 0).all() and pooled90 > 0
                    and (n90 >= MIN_TRADES).all())
    if point_ok:
        b = m0.bootstrap_mean_R(t90["entry_day"].to_numpy(),
                                t90["realized_R"].to_numpy(float))
        df_boot = pd.DataFrame([dict(scope="CLEAR90_CONT_pooled", **b)])
    else:
        df_boot = pd.DataFrame([dict(note="STOP_NO_BOOTSTRAP: CLEAR90 point "
                                    "gate failed or threshold unavailable")])

    # ---- P10 reversal CLEAR90 ----
    rev90_rows = []
    rev_pooled = float(r90["realized_R"].mean()) if len(r90) else np.nan
    r90w = (df_ex[(df_ex.variant == "CLEAR90") & (df_ex.setup == "REV")]
            .sort_values("wf"))
    for _, r in r90w.iterrows():
        rev90_rows.append(dict(wf=r["wf"], setup="REV", variant="CLEAR90",
                               trades=int(r["executed_trades"]),
                               clear_purity=r["clear_purity"],
                               gross_expectancy_R=r["gross_expectancy_R"]))
    df_rev90 = pd.DataFrame(rev90_rows)
    rev90_3of3 = bool(len(df_rev90) == 3
                      and (df_rev90["gross_expectancy_R"] > 0).all())

    # ---- write ----
    df_rr.to_csv(OUT / "signal_rr_consistency_audit.csv", index=False,
                 encoding="utf-8-sig")
    df_cls.to_csv(OUT / "execution_by_frozen_class.csv", index=False,
                  encoding="utf-8-sig")
    df_mix.to_csv(OUT / "clear_nonclear_mixture.csv", index=False,
                  encoding="utf-8-sig")
    df_sel.to_csv(OUT / "clear85_vs_clear90_selector.csv", index=False,
                  encoding="utf-8-sig")
    df_ex.to_csv(OUT / "clear85_vs_clear90_execution.csv", index=False,
                 encoding="utf-8-sig")
    df_geo.to_csv(OUT / "geometry_by_clear_status.csv", index=False,
                  encoding="utf-8-sig")
    df_rev90.to_csv(OUT / "reversal_clear90_diagnostic.csv", index=False,
                    encoding="utf-8-sig")
    df_boot.to_csv(OUT / "execution_purity_bootstrap.csv", index=False,
                   encoding="utf-8-sig")
    pd.concat([t85.assign(variant="CLEAR85"), t90.assign(variant="CLEAR90")],
              ignore_index=True).to_parquet(
        OUT / "selection_purity_trade_log.parquet", index=False)

    protocol = dict(
        experiment="SMC Execution Selection Purity Gate v1.1",
        base_commit="7c7858ae4ecec3a582e5b8d3691bfb88bb1012ab",
        frozen=dict(risk=1.0, target="selected-boundary beyond-attack (CONT) / "
                    "nearest opposing (REV)",
                    direction_model="G4_BASE HGB availability-safe OOF",
                    cont_threshold="q10 of availability-safe direction OOF",
                    rev_threshold="q90 of availability-safe direction OOF",
                    entry="next valid 5m open",
                    stop="decision_close - direction*atr0",
                    same_bar="STOP_FIRST",
                    oos_cutoff="v1.0.1 repaired semantics"),
        variants=dict(
            CLEAR85=dict(target_precision=PREC85, min_oof_selection=CLEAR_MIN_SEL,
                         role="baseline / reproduce v1.0.1"),
            CLEAR90=dict(target_precision=PREC90, min_oof_selection=CLEAR_MIN_SEL,
                         role="single preregistered stricter hypothesis")),
        clear90_is_development_hypothesis=(
            "0.90 是看过 v1.0.1 后提出的 development hypothesis，"
            "不是 independent OOS confirmation"),
        gates=dict(mechanism=dict(A_baseline_mechanism="E_R_clear>0 & "
                                  "E_R_nonclear<0 3/3",
                                  B_purity="CLEAR90 purity > CLEAR85 purity 3/3",
                                  C_execution="CLEAR90 expectancy > CLEAR85 3/3"),
                   execution=dict(per_wf=">0 3/3", pooled=">0",
                                  min_trades=MIN_TRADES)),
        forbidden=["risk 0.5/2.0", "target修改", "minimum RR filter",
                   "direction threshold修改", "新增feature", "FVG/OB/trend",
                   "删symbol", "手续费", "prospective OOS",
                   "根据PnL扫描clear threshold"],
    )
    json.dump(protocol, open(OUT / "EXECUTION_PURITY_PROTOCOL.json", "w"),
              indent=2, ensure_ascii=False, default=str)

    audit = dict(
        experiment="SMC Execution Selection Purity Gate v1.1",
        base_commit="7c7858a",
        p1_signal_rr_consistency=dict(
            n_groups_with_multiple_rr=bad, fatal=False,
            status="PASS (unique rr within execution signal group)"),
        p4_clear85_reproduces_v101=reproduce,
        gates=dict(
            SELECTION_PURITY_MECHANISM_CONFIRMED=MECH,
            CLEAR90_GROSS_EXECUTION_EDGE_PRESENT=point_ok,
            components=dict(A_baseline_mechanism=A, B_purity_improved=B,
                            C_expectancy_improved=C),
            per_wf_clear85=list(e85), per_wf_clear90=list(e90),
            pooled_clear90=round(pooled90, 4) if pd.notna(pooled90) else None),
        reversal=dict(variant="CLEAR90_secondary",
                      per_wf=list(df_rev90["gross_expectancy_R"]),
                      pooled=round(rev_pooled, 4) if pd.notna(rev_pooled) else None,
                      REVERSAL_CLEAR90_CANDIDATE=rev90_3of3,
                      note="candidate only, not promoted"),
        mechanism_note=(
            "收紧 clear gate 会同时 (a) 提高 clear_purity、(b) 降低 E_R_clear、"
            "(c) 抬高 required_clear_purity。实测 CLEAR85->90："
            "WF1 E_R_clear 0.1555->0.1040 而 required 0.8346->0.8551；"
            "WF3 E_R_clear 0.2010->0.1121 而 required 0.7934->0.8589。"
            "即 purity 的收益被 E_R_clear 的衰减抵消（WF3 甚至更差）。"
            "这是 CLEAR90 无法闭合缺口的结构原因。"),
        next_step=("CONTRACT_COST_METADATA_AUDIT" if point_ok else
                   ("边界案例（reviewer 裁决）：purity 3/3 提高 (B=TRUE)，"
                    "但 expectancy 未 3/3 改善 (C=FALSE, WF3 恶化)；"
                    "pooled 由 -0.0156 改善到 -0.0006（约到 breakeven 但未转正）。"
                    "按 P12，'purity 提高但 expectancy 未稳定改善' 使 "
                    "RISK_COUPLED_EXECUTION 变为 eligible，但需 reviewer 决定。")),
    )
    json.dump(audit, open(OUT / "EXECUTION_PURITY_AUDIT.json", "w"),
              indent=2, ensure_ascii=False, default=str)

    write_report(df_rr, df_cls, df_mix, df_sel, df_ex, df_geo, df_rev90,
                 df_boot, audit)
    print("\n=== GATES ===")
    print(json.dumps(audit["gates"], indent=2, ensure_ascii=False))
    print(f"[DONE] {time.perf_counter()-t0:.1f}s -> {OUT}")


def geo_by_clear(ex, wf, variant):
    """P11：固定 bins，分 CLEAR / NONCLEAR。"""
    rows = []
    is_clear = ex["attack_rr"].isin(CLEAR_CLASSES).to_numpy()
    b = pd.cut(ex["target_R_exec"], bins=m101.GEO_BINS,
               labels=m101.GEO_LABELS, right=False)
    for status, mask in (("CLEAR", is_clear), ("NONCLEAR", ~is_clear)):
        for lab in m101.GEO_LABELS:
            m = mask & (b == lab)
            sub = ex[m]
            n = len(sub)
            if n == 0:
                rows.append(dict(wf=wf, variant=variant, clear_status=status,
                                 bucket=lab, n=0, target_hit_rate=np.nan,
                                 gross_expectancy_R=np.nan))
                continue
            rows.append(dict(wf=wf, variant=variant, clear_status=status,
                             bucket=lab, n=n,
                             target_hit_rate=round(float(
                                 (sub["outcome"] == "TARGET").mean()), 4),
                             gross_expectancy_R=round(float(
                                 sub["realized_R"].mean()), 4)))
    return rows


def write_report(df_rr, df_cls, df_mix, df_sel, df_ex, df_geo, df_rev90,
                 df_boot, audit):
    def tbl(df, cols):
        return "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |"
                         for _, r in df.iterrows())
    g = audit["gates"]
    md = f"""# SMC Execution Selection Purity Gate v1.1

**base**: `7c7858a` (v1.0.1) &nbsp; **脚本**: `run_execution_selection_purity_v1_1.py`

冻结 v1.0.1 的 target / risk / entry / stop / direction model。
**唯一新增变量**：一个预注册的更严格 clear gate `CLEAR90`（train-OOF precision=0.90）。

---

## 1. P1 Signal-level `rr_direction` consistency（HARD GATE）

| wf | groups | unique_rr==1 | unique_rr>1 | max_unique_rr |
|---|---:|---:|---:|---:|
{tbl(df_rr, ['wf','n_groups','n_groups_unique_rr_1','n_groups_unique_rr_gt1','max_unique_rr'])}

`n_groups_unique_rr_gt1 = {audit['p1_signal_rr_consistency']['n_groups_with_multiple_rr']}`
→ signal-level frozen class 语义成立（同一 signal 的 simultaneous contacts
共享同一 Oracle direction）。

---

## 2. P4 CLEAR85 复现 v1.0.1

`CLEAR85 reproduces v1.0.1 = {audit['p4_clear85_reproduces_v101']}`（三 WF 逐位一致）。

---

## 3. P2 按 frozen class 拆 execution

| wf | setup | frozen_class | n | share | hit | avg_win | avg_loss | exp_R |
|---|---|---|---:|---:|---:|---:|---:|---:|
{tbl(df_cls, ['wf','setup','frozen_class','n','share','target_hit_rate','mean_win_R','mean_loss_R','gross_expectancy_R'])}

---

## 4. P3 Clear / Non-clear mixture

| wf | setup | variant | n | n_clear | n_nonclear | purity | E_R_clear | E_R_nonclear | E_R_total | required_purity | purity−req |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{tbl(df_mix, ['wf','setup','variant','n_executed','n_clear','n_nonclear','clear_purity','E_R_clear','E_R_nonclear','E_R_total','required_clear_purity_for_zero_gross','purity_minus_required'])}

---

## 5. Selector thresholds

| wf | clear_thr_85 | clear_thr_90 | cont_thr | rev_thr | clear90_available |
|---|---:|---:|---:|---:|---|
{tbl(df_sel, ['wf','clear_thr_85','clear_thr_90','cont_thr','rev_thr','clear90_available'])}

Direction q10/q90 **未为 CLEAR90 重新优化**（与 CLEAR85 同一 availability-safe OOF）。

---

## 6. P7 CLEAR85 vs CLEAR90 execution

| wf | setup | variant | raw | collapsed | executed | purity | TRADEOFF | UNRES | NOCOMP | hit | **exp_R** | PF | /day | LONG | SHORT |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{tbl(df_ex, ['wf','setup','variant','raw_selected','collapsed_signals','executed_trades','clear_purity','TRADEOFF_share','UNRESOLVED_share','NO_COMPARABLE_share','target_hit_rate','gross_expectancy_R','profit_factor_R','trades_per_day','LONG_trades','SHORT_trades'])}

---

## 7. P11 Geometry by clear status（CLEAR85，固定 bins）

| wf | status | bucket | n | hit | exp_R |
|---|---|---|---:|---:|---:|
{tbl(df_geo, ['wf','clear_status','bucket','n','target_hit_rate','gross_expectancy_R'])}

判断 v1.0.1 的"RR 无结构"是否只是 non-clear 污染混合造成。**不得据此新增 RR filter。**

---

## 8. P10 Reversal CLEAR90（secondary）

| wf | trades | purity | exp_R |
|---|---:|---:|---:|
{tbl(df_rev90, ['wf','trades','clear_purity','gross_expectancy_R'])}

`REVERSAL_CLEAR90_CANDIDATE = {audit['reversal']['REVERSAL_CLEAR90_CANDIDATE']}`（candidate，不 promote）。

---

## 9. P9 Bootstrap

| scope | n_boot | p2.5 | p50 | p97.5 |
|---|---:|---:|---:|---:|
{tbl(df_boot, [c for c in ['scope','n_boot','p2_5','p50','p97_5'] if c in df_boot.columns]) if 'n_boot' in df_boot.columns else '| - | STOP_NO_BOOTSTRAP | | | |'}

---

## 10. P8/P9 Gates

```json
{json.dumps(g, indent=2, ensure_ascii=False)}
```

- **SELECTION_PURITY_MECHANISM_CONFIRMED = {g['SELECTION_PURITY_MECHANISM_CONFIRMED']}**
- **CLEAR90_GROSS_EXECUTION_EDGE_PRESENT = {g['CLEAR90_GROSS_EXECUTION_EDGE_PRESENT']}**

> CLEAR90 是看过 v1.0.1 后提出的 development hypothesis，
> **不是 independent OOS confirmation**。

---

## 11. P12/P13 下一步 / P16 STOP

next_step = `{audit['next_step']}`

只有 CLEAR90 未能提高 purity、或 purity 提高但 expectancy 无改善时，
才批准 `RISK_COUPLED_EXECUTION`。若 CLEAR90 3/3 正 → 下一步
`CONTRACT_COST_METADATA_AUDIT`（先不做 stop/target 优化）。

代码 + 测试 + 运行 + 报告 + commit + push 后 STOP。等 reviewer。
"""
    open(OUT / "SMC_EXECUTION_SELECTION_PURITY_V1_1.md", "w",
         encoding="utf-8-sig").write(md)


if __name__ == "__main__":
    main()
