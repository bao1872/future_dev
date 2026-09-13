"""CAL-1: full-node marginal probability recalibration under RR / edge conditioning.

Baseline: bao1872/future_dev main 3e385b78ba3347997d3b155eba672cc42c86d93e

AUDIT-D showed T2's winner's curse is mainly gain-side: rr_ref * (p_reach - y_reach),
i.e. a small reach-probability error amplified by high RR into a large GraphEV optimism.
CAL-1 asks: can that gain-side bias be recalibrated using ONLY ROOT-time static variables,
with no future state H_i?

Fixed models (no tuning):
  C0_RAW      : the current (p_reach, p_loss_before, p_censor_before)
  C1_GLOBAL   : multinomial recalibration on [log p_reach, log p_loss, log p_censor]
  C2_RR_EDGE  : C1 features + log1p(rr_ref) + edge_index (categorical)

Calibrator is fit ONLY on purged-TB1 full-node truth; TB2 is evaluation only.
Selector index is generated BEFORE outcome merge (no leakage). C0 must reproduce AUDIT-D.
No execution is computed this round.

Scope: calibration experiment only. No G0 / GraphEV / selector / execution / horizon
change. No duration / H_i / SMC / OB-FVG / adverse field / batch / EntryOpen / HSMM /
HMM / RL / GNN / WF2-WF3 / raw bars rescan / parameter grid.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.liquidity_oracle_atlas.run_5m_graph_probability_v1 import (  # noqa: E402
    load_transitions, purge_train, fit_multinomial, G0_NUM, G0_CAT, NEXT, LOSS, CENSOR,
)
from research.liquidity_oracle_atlas.run_5m_graph_target_decision_v1 import (  # noqa: E402
    load_full_node_tb2, predict_reordered, compute_graph_ev, select_targets, FULL_UNIV, OUT,
)

BASELINE_COMMIT = "3e385b78ba3347997d3b155eba672cc42c86d93e"
EPS = 1e-8
CAL_EDGES = (0, 1, 2, 3, 4)
RR_BINS = [0, 1.5, 2.0, 3.0, 4.0, 6.0, np.inf]
RR_LABELS = ["0_1p5", "1p5_2", "2_3", "3_4", "4_6", "6p"]
CV_SCOPE = "CALIBRATOR_ONLY_EXPANDING_CV_ON_FROZEN_G0"

# AUDIT-D C0 reproduction targets
RC = dict(t2_gain_error=0.03788488854977161, t2_loss_error=-0.0023375445935045133,
          t2_ev_error=0.04022243314327612, switched_sed=0.13137104665,
          t2_n=24954, same=14778, switched=10176)


def reliability(pred, actual, n_bins=10):
    d = pd.DataFrame({"p": np.asarray(pred, float), "y": np.asarray(actual, float)})
    try:
        d["bin"] = pd.qcut(d["p"], n_bins, duplicates="drop")
    except Exception:
        d["bin"] = pd.cut(d["p"], n_bins)
    g = (d.groupby("bin", observed=True)
         .agg(n=("y", "size"), pred=("p", "mean"), actual=("y", "mean"))
         .reset_index(drop=True))
    g["gap"] = g["pred"] - g["actual"]
    return g, float((g["n"] / max(len(d), 1) * g["gap"].abs()).sum())


def day_bootstrap(values, days, n_boot=1000, seed=20260913):
    d = pd.DataFrame({"v": np.asarray(values, float), "day": pd.to_datetime(np.asarray(days))})
    daily = d.groupby("day")["v"].mean().to_numpy()
    nd = len(daily)
    rng = np.random.default_rng(seed)
    boot = daily[rng.integers(0, nd, size=(n_boot, nd))].mean(axis=1)
    return dict(mean=float(daily.mean()), ci_lo=float(np.percentile(boot, 2.5)),
                ci_hi=float(np.percentile(boot, 97.5)), n_days=int(nd))


def fit_c1(X_cols, y):
    cols = ["lp_reach", "lp_loss", "lp_censor"]
    m = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=3000)
    m.fit(X_cols[cols], y)
    return m, cols


def fit_c2(X_cols, y):
    num = ["lp_reach", "lp_loss", "lp_censor", "lrr"]
    pre = ColumnTransformer([
        ("num", "passthrough", num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["edge_index"]),
    ])
    pipe = Pipeline([("pre", pre),
                     ("clf", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=3000))])
    pipe.fit(X_cols[num + ["edge_index"]], y)
    return pipe, num + ["edge_index"]


def predict_cal(model, cols, X_cols):
    p = np.asarray(model.predict_proba(X_cols[cols]), float)
    classes = (model.named_steps["clf"].classes_ if hasattr(model, "named_steps")
               else model.classes_)
    assert np.array_equal(classes, np.array([NEXT, LOSS, CENSOR])), "CAL1_CLASS_ORDER_FAIL"
    return p


def select_pos(ev, feat12):
    sel = select_targets(feat12, ev, np.zeros(len(feat12), dtype=float))
    return sel["t0"], sel["t2"], sel["starts"]


def main():
    t_all = time.perf_counter()

    # ---- guards / load ----
    t0 = time.perf_counter()
    tr = load_transitions(FULL_UNIV, blocks=["TB1"], scope_tag="TB12_HARDENED")
    te = load_transitions(FULL_UNIV, blocks=["TB2"], scope_tag="TB12_HARDENED")
    test_start = pd.Timestamp(te["signal_trading_day"].min())
    tr_p, purge_diag = purge_train(tr, test_start)
    assert purge_diag["n_train_signals_after_purge"] == 21135, "CAL1_PURGE_MISMATCH"
    assert te["signal_id"].nunique() == 24954 and len(te) == 55945

    features, outcomes, provenance = load_full_node_tb2()
    assert provenance is not None, "STOP_CAL1_CACHE_MISSING"

    feat12 = (features.loc[features["block"].isin(["TB1", "TB2"])]
              .sort_values(["signal_gid", "edge_index"], kind="stable").reset_index(drop=True))
    assert not feat12.duplicated(["signal_gid", "edge_index"]).any(), "CAL1_KEY_DUP"

    blk = feat12["block"].to_numpy()
    tr_p_keys = tr_p[["symbol", "signal_id"]].drop_duplicates()
    tr_p_keys["_p"] = 1
    j = feat12[["symbol", "signal_id"]].merge(tr_p_keys, on=["symbol", "signal_id"], how="left")
    in_purged = j["_p"].notna().to_numpy()
    tb1_mask = (blk == "TB1") & in_purged
    tb2_mask = (blk == "TB2")

    # P1: purge key EQUALITY (not just cardinality)
    def _keys(df):
        return set(map(tuple, df[["symbol", "signal_id"]].drop_duplicates().to_numpy()))

    full_tb1_trans_keys = _keys(tr)
    purged_trans_keys = _keys(tr_p)
    fullnode_tb1_keys = _keys(feat12.loc[blk == "TB1"])
    kept_keys = _keys(feat12.loc[tb1_mask])
    n_key_missing = len(purged_trans_keys - kept_keys)
    n_key_extra = len(kept_keys - purged_trans_keys)
    assert purged_trans_keys == kept_keys, "STOP_CAL1_TRAIN_UNIVERSE_MAPPING_FAIL"
    assert n_key_missing == 0 and n_key_extra == 0, "STOP_CAL1_TRAIN_UNIVERSE_MAPPING_FAIL"
    removed_keys = fullnode_tb1_keys - kept_keys
    assert removed_keys == (full_tb1_trans_keys - purged_trans_keys), \
        "STOP_CAL1_PURGED_KEY_MISMATCH"
    assert len(removed_keys) == 30, "STOP_CAL1_PURGED_KEY_COUNT"

    cache_universe = dict(
        features_total_rows=int(len(features)),
        block_counts={k: int(v) for k, v in features["block"].value_counts().items()},
        tb1_signals_full=int(feat12.loc[blk == "TB1", "signal_gid"].nunique()),
        tb1_nodes_full=int((blk == "TB1").sum()),
        tb1_signals_purged=int(feat12.loc[tb1_mask, "signal_gid"].nunique()),
        tb1_nodes_purged=int(tb1_mask.sum()),
        tb2_signals=int(feat12.loc[tb2_mask, "signal_gid"].nunique()),
        tb2_nodes=int(tb2_mask.sum()),
        key_unique_signal_gid_edge=int(feat12.duplicated(["signal_gid", "edge_index"]).sum()),
        purge_key_missing=n_key_missing,
        purge_key_extra=n_key_extra,
        purge_removed_keys=len(removed_keys),
    )
    if cache_universe["tb1_signals_purged"] != 21135:
        raise SystemExit(f"STOP_CAL1_TRAIN_UNIVERSE_MAPPING_FAIL {cache_universe}")
    load_seconds = time.perf_counter() - t0

    # ---- G0 fit + predict + C0 marginals ----
    t1 = time.perf_counter()
    g0 = fit_multinomial(tr_p, G0_NUM, G0_CAT)
    P = predict_reordered(g0, feat12, G0_NUM, G0_CAT)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
    ev_c0, p_reach, p_loss, p_censor = compute_graph_ev(
        P[:, 0], P[:, 1], P[:, 2],
        feat12["signal_gid"].to_numpy(np.int64), feat12["rr_ref"].to_numpy(float))
    fit_predict_seconds = time.perf_counter() - t1

    # ---- truth (aligned by key, no reorder) ----
    oc = outcomes.set_index(["signal_gid", "edge_index"])["target_state"]
    ts = oc.reindex(pd.MultiIndex.from_arrays(
        [feat12["signal_gid"].to_numpy(), feat12["edge_index"].to_numpy()])).to_numpy()
    assert not np.isnan(ts).any(), "CAL1_TRUTH_MISSING"
    ts = ts.astype(np.int64)
    y_reach = (ts == NEXT).astype(float)
    y_loss = (ts == LOSS).astype(float)
    y_censor = (ts == CENSOR).astype(float)
    assert bool(((y_reach + y_loss + y_censor) == 1).all())

    rr = feat12["rr_ref"].to_numpy(float)
    edge = feat12["edge_index"].to_numpy(np.int64)
    u_label = rr * y_reach - y_loss

    # ---- selection FIRST (features only, no outcome columns) ----
    t0_pos, t2_c0, starts = select_pos(ev_c0, feat12[["signal_gid", "edge_index"]])

    # ---- calibrators fit on purged TB1 ----
    Xc = pd.DataFrame({
        "lp_reach": np.log(p_reach + EPS), "lp_loss": np.log(p_loss + EPS),
        "lp_censor": np.log(p_censor + EPS), "lrr": np.log1p(rr),
        "edge_index": edge})
    y = ts.copy()
    c1, cols1 = fit_c1(Xc[tb1_mask], y[tb1_mask])
    c2, cols2 = fit_c2(Xc[tb1_mask], y[tb1_mask])
    pc1 = predict_cal(c1, cols1, Xc)
    pc2 = predict_cal(c2, cols2, Xc)
    assert np.allclose(pc1.sum(1), 1) and np.allclose(pc2.sum(1), 1)
    ev_c1 = pc1[:, 0] * rr - pc1[:, 1]
    ev_c2 = pc2[:, 0] * rr - pc2[:, 1]

    _, t2_c1, _ = select_pos(ev_c1, feat12[["signal_gid", "edge_index"]])
    _, t2_c2, _ = select_pos(ev_c2, feat12[["signal_gid", "edge_index"]])

    models = {
        "C0_RAW": dict(pT=p_reach, pL=p_loss, pC=p_censor, ev=ev_c0, t2=t2_c0),
        "C1_GLOBAL": dict(pT=pc1[:, 0], pL=pc1[:, 1], pC=pc1[:, 2], ev=ev_c1, t2=t2_c1),
        "C2_RR_EDGE": dict(pT=pc2[:, 0], pL=pc2[:, 1], pC=pc2[:, 2], ev=ev_c2, t2=t2_c2),
    }

    # TB2 signal-level index (per signal first node)
    sig_block = feat12.iloc[starts]["block"].to_numpy()
    tb2_sig = sig_block == "TB2"
    sig_day = feat12.iloc[starts]["trading_day"].to_numpy()
    t0_tb2 = t0_pos[tb2_sig]

    # ---- C0 reproduction ----
    c0_t2 = t2_c0[tb2_sig]
    c0_sw = c0_t2 != t0_tb2
    c0_gain = (rr * (p_reach - y_reach))[c0_t2].mean()
    c0_loss = (p_loss - y_loss)[c0_t2].mean()
    c0_ev = (ev_c0 - u_label)[c0_t2].mean()
    c0_sed = (ev_c0 - u_label)[c0_t2][c0_sw].mean() - (ev_c0 - u_label)[t0_tb2][c0_sw].mean()
    rep = dict(t2_gain_error=float(c0_gain), t2_loss_error=float(c0_loss),
               t2_ev_error=float(c0_ev), switched_sed=float(c0_sed),
               t2_n=int(len(c0_t2)), same=int((~c0_sw).sum()), switched=int(c0_sw.sum()))
    if not (abs(rep["t2_gain_error"] - RC["t2_gain_error"]) < 1e-8
            and abs(rep["t2_loss_error"] - RC["t2_loss_error"]) < 1e-8
            and abs(rep["t2_ev_error"] - RC["t2_ev_error"]) < 1e-8
            and abs(rep["switched_sed"] - RC["switched_sed"]) < 1e-7
            and rep["same"] == RC["same"] and rep["switched"] == RC["switched"]):
        raise SystemExit(f"STOP_CAL1_BASELINE_REPRODUCTION_FAIL {rep}")

    # ---- TB2 evaluation ----
    tb2_idx = np.where(tb2_mask)[0]
    e0_4 = np.isin(edge, CAL_EDGES)  # P0 FIX: node-level edges 0-4 mask over all nodes
    # explicit reference counts straight from the dataframe (do NOT self-prove with e0_4)
    tb2_edge_counts = {int(e): int(((blk == "TB2") & (edge == e)).sum()) for e in CAL_EDGES}
    tb2_edge_sum_0_4 = int(sum(tb2_edge_counts.values()))
    ref_e04 = feat12.loc[(feat12["block"] == "TB2") & feat12["edge_index"].isin(CAL_EDGES)]
    assert len(ref_e04) == tb2_edge_sum_0_4
    cache_universe["tb2_edge_node_counts"] = tb2_edge_counts
    cache_universe["tb2_edge_node_sum_0_4"] = tb2_edge_sum_0_4
    prob_rows, ev_rows, sel_rows, boot_rows, rr_rows, edge_rows = [], [], [], [], [], []

    for name, M in models.items():
        pT, pL, pC, ev = M["pT"], M["pL"], M["pC"], M["ev"]
        # probability metrics (all TB2 nodes; normalize)
        Pn = np.clip(np.column_stack([pT, pL, pC]), 1e-12, 1.0)
        Pn = Pn / Pn.sum(1, keepdims=True)
        Y = np.column_stack([y_reach, y_loss, y_censor])
        ll = float(-np.mean(np.log((Pn * Y).sum(1)[tb2_mask])))
        br = float(np.mean(((Pn - Y) ** 2).sum(1)[tb2_mask]))
        # reach/loss ECE edges0-4 (TB2)
        m04 = tb2_mask & e0_4
        assert int(m04.sum()) == tb2_edge_sum_0_4, "CAL1_EDGE_MASK_MISMATCH"
        _, ece_reach = reliability(pT[m04], y_reach[m04])
        _, ece_loss = reliability(pL[m04], y_loss[m04])
        # EV calibration edges0-4 (TB2)
        _, ece_ev = reliability(ev[m04], u_label[m04])
        d = pd.DataFrame({"ev": ev[m04], "u": u_label[m04]})
        d["bin"] = pd.qcut(d["ev"], 10, duplicates="drop")
        top = d[d["bin"] == d["bin"].cat.categories[-1]]
        prob_rows.append(dict(model=name, tb2_logloss=ll, tb2_brier=br,
                              reach_ece_edges0_4=ece_reach, loss_ece_edges0_4=ece_loss))
        ev_rows.append(dict(model=name, ev_ece_edges0_4=ece_ev,
                            top_decile_pred_ev=float(top["ev"].mean()),
                            top_decile_actual_u=float(top["u"].mean()),
                            top_decile_gap=float((top["ev"] - top["u"]).mean())))
        # selector (TB2 signals)
        s = M["t2"][tb2_sig]
        sw = s != t0_tb2
        gain_err = (rr * (pT - y_reach))[s].mean()
        loss_err = (pL - y_loss)[s].mean()
        ev_err = (ev - u_label)[s].mean()
        sel_rows.append(dict(model=name, n=int(len(s)), switch_rate=float(sw.mean()),
                             mean_selected_edge=float(edge[s].mean()),
                             pred_reach=float(pT[s].mean()), actual_reach=float(y_reach[s].mean()),
                             reach_error=float((pT - y_reach)[s].mean()),
                             gain_error=float(gain_err), loss_error=float(loss_err),
                             ev_error=float(ev_err)))
        # oracle / winner's curse deltas
        u_t0 = u_label[t0_tb2]
        pred_delta = (ev[s] - ev[t0_tb2])
        oracle_delta = (u_label[s] - u_t0)
        sed = pred_delta - oracle_delta
        for pop, mk in (("ALL", np.ones(len(s), bool)), ("SWITCHED", sw)):
            boot_rows.append(dict(model=name, population=pop, metric="oracle_delta",
                                  **day_bootstrap(oracle_delta[mk], sig_day[tb2_sig][mk])))
            boot_rows.append(dict(model=name, population=pop, metric="selection_error_delta",
                                  **day_bootstrap(sed[mk], sig_day[tb2_sig][mk])))
        # rr bins (edges0-4 TB2)
        for i, lab in enumerate(RR_LABELS):
            mk = m04 & (rr >= RR_BINS[i]) & (rr < RR_BINS[i + 1])
            if mk.sum() == 0:
                continue
            rr_rows.append(dict(model=name, rr_bin=lab, n=int(mk.sum()),
                                pred_reach=float(pT[mk].mean()), actual_reach=float(y_reach[mk].mean()),
                                gain_error=float((rr * (pT - y_reach))[mk].mean()),
                                ev_error=float((ev - u_label)[mk].mean())))
        # selected edge bins (TB2 selected)
        for lab, mk in (("0", edge[s] == 0), ("1", edge[s] == 1), ("2", edge[s] == 2),
                        ("3", edge[s] == 3), ("4", edge[s] == 4), ("5p", edge[s] >= 5)):
            if mk.sum() == 0:
                continue
            edge_rows.append(dict(model=name, selected_edge=lab, n=int(mk.sum()),
                                  pred_reach=float(pT[s][mk].mean()),
                                  actual_reach=float(y_reach[s][mk].mean()),
                                  gain_error=float((rr * (pT - y_reach))[s][mk].mean()),
                                  ev_error=float((ev - u_label)[s][mk].mean())))

    prob_df = pd.DataFrame(prob_rows)
    ev_df = pd.DataFrame(ev_rows)
    sel_df = pd.DataFrame(sel_rows)
    boot_df = pd.DataFrame(boot_rows)
    prob_df.to_csv(OUT / "cal1_tb2_probability_metrics.csv", index=False)
    ev_df.to_csv(OUT / "cal1_tb2_ev_calibration.csv", index=False)
    sel_df.to_csv(OUT / "cal1_selector_summary.csv", index=False)
    boot_df.to_csv(OUT / "cal1_selector_bootstrap.csv", index=False)

    # P0 guard: selector / bootstrap must be UNCHANGED by the diagnostic fix
    EXP_SWITCH = {"C0_RAW": 0.40779033421495553, "C1_GLOBAL": 0.14354412118297669,
                  "C2_RR_EDGE": 0.26244289492666506}
    EXP_OD = {"C0_RAW": -0.0184622879875681, "C1_GLOBAL": -0.006192501704162234,
              "C2_RR_EDGE": -0.0011694482828641735}
    EXP_SED = {"C0_RAW": 0.05203262061944321, "C1_GLOBAL": 0.011433167009210236,
               "C2_RR_EDGE": 0.011247484841146845}
    sw_by_model = dict(zip(sel_df["model"], sel_df["switch_rate"]))
    for nm in models:
        assert abs(float(sw_by_model[nm]) - EXP_SWITCH[nm]) < 1e-12, \
            "STOP_CAL1_SELECTOR_CHANGED_AFTER_DIAGNOSTIC_FIX"
        r_od = boot_df[(boot_df.model == nm) & (boot_df.metric == "oracle_delta")
                       & (boot_df.population == "ALL")].iloc[0]
        r_sd = boot_df[(boot_df.model == nm) & (boot_df.metric == "selection_error_delta")
                       & (boot_df.population == "ALL")].iloc[0]
        assert abs(float(r_od["mean"]) - EXP_OD[nm]) < 1e-12, \
            "STOP_CAL1_SELECTOR_CHANGED_AFTER_DIAGNOSTIC_FIX"
        assert abs(float(r_sd["mean"]) - EXP_SED[nm]) < 1e-12, \
            "STOP_CAL1_SELECTOR_CHANGED_AFTER_DIAGNOSTIC_FIX"
    pd.DataFrame(rr_rows).to_csv(OUT / "cal1_rr_bins.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(OUT / "cal1_edge_bins.csv", index=False)

    # ---- TB1 chronological expanding CV (report only) ----
    tb1_idx = np.where(tb1_mask)[0]
    day = feat12["trading_day"].to_numpy()[tb1_idx]
    udays = np.sort(pd.unique(day))
    eb = np.linspace(0, len(udays), 5).astype(int)
    d2b = {}
    for b in range(4):
        for dd in udays[eb[b]:eb[b + 1]]:
            d2b[dd] = b
    blk_of = np.array([d2b[dd] for dd in day])
    cv_rows = []
    for f in range(1, 4):
        tr_i = tb1_idx[blk_of < f]
        te_i = tb1_idx[blk_of == f]
        if len(tr_i) == 0 or len(te_i) == 0:
            continue
        Xtr, Xte = Xc.iloc[tr_i], Xc.iloc[te_i]
        ytr = y[tr_i]
        for nm in ("C0_RAW", "C1_GLOBAL", "C2_RR_EDGE"):
            if nm == "C0_RAW":
                pT, pL, pC = p_reach[te_i], p_loss[te_i], p_censor[te_i]
            elif nm == "C1_GLOBAL":
                c, cc = fit_c1(Xtr, ytr); p = predict_cal(c, cc, Xte)
                pT, pL, pC = p[:, 0], p[:, 1], p[:, 2]
            else:
                c, cc = fit_c2(Xtr, ytr); p = predict_cal(c, cc, Xte)
                pT, pL, pC = p[:, 0], p[:, 1], p[:, 2]
            Pn = np.clip(np.column_stack([pT, pL, pC]), 1e-12, 1.0)
            Pn = Pn / Pn.sum(1, keepdims=True)
            Yc = np.column_stack([y_reach[te_i], y_loss[te_i], y_censor[te_i]])
            ll = float(-np.mean(np.log((Pn * Yc).sum(1))))
            brr = float(np.mean(((Pn - Yc) ** 2).sum(1)))
            evx = pT * rr[te_i] - pL
            _, ece_ev = reliability(evx, u_label[te_i])
            _, ece_reach = reliability(pT, y_reach[te_i])
            cv_rows.append(dict(model=nm, fold=f, cv_scope=CV_SCOPE, logloss=ll, brier=brr,
                                reach_ece=ece_reach, ev_ece=ece_ev))
    pd.DataFrame(cv_rows).to_csv(OUT / "cal1_cv_metrics.csv", index=False)

    # ---- verdict ----
    def sel_mean(nm, metric, pop):
        r = boot_df[(boot_df.model == nm) & (boot_df.metric == metric) & (boot_df.population == pop)]
        return None if len(r) == 0 else r.iloc[0].to_dict()

    verdicts = {}
    for nm in models:
        v = sel_mean(nm, "oracle_delta", "ALL")
        if v["ci_hi"] < 0:
            t = "PREEXECUTION_INCREMENT_NEGATIVE"
        elif v["ci_lo"] > 0:
            t = "PREEXECUTION_INCREMENT_SUPPORTED"
        else:
            t = "PREEXECUTION_INCREMENT_AMBIGUOUS"
        verdicts[nm] = dict(oracle_delta_ci=v, verdict=t)

    c0_sed = sel_mean("C0_RAW", "selection_error_delta", "ALL")
    c0_w = c0_sed["ci_hi"] - c0_sed["ci_lo"]
    reduces = False
    for nm in models:
        if nm == "C0_RAW":
            continue
        s = sel_mean(nm, "selection_error_delta", "ALL")
        if abs(s["mean"]) < abs(c0_sed["mean"]) and (s["ci_hi"] - s["ci_lo"]) < c0_w:
            reduces = True
    cal_note = "CALIBRATION_REDUCES_SELECTION_BIAS" if reduces \
        else "NO_CLEAR_SELECTION_BIAS_REDUCTION"

    summ = dict(
        audit="CAL1_FULLNODE_MARGINAL_RECALIBRATION_V1",
        baseline_commit=BASELINE_COMMIT,
        purge=purge_diag,
        cache_universe=cache_universe,
        reproduction_c0=rep,
        probability_metrics=prob_df.to_dict("records"),
        ev_calibration=ev_df.to_dict("records"),
        selector_summary=sel_df.to_dict("records"),
        bootstrap=boot_df.to_dict("records"),
        verdicts=verdicts,
        selection_bias_note=cal_note,
        cv_scope=CV_SCOPE,
        cv_scope_note=("Calibrator fit is chronological expanding, BUT the base G0 probabilities "
                       "come from a G0 fit on ALL purged TB1 at once. Therefore this is NOT an "
                       "end-to-end causal CV; it is reporting-only and is not used to tune/select "
                       "C1/C2."),
        timing_seconds=dict(load_seconds=load_seconds, fit_predict_seconds=fit_predict_seconds,
                            total_seconds=time.perf_counter() - t_all),
        interpretation_contract=("Calibration experiment only. Calibrator fit on purged TB1; TB2 "
                                 "evaluation only. Selector index before outcome merge. No execution."),
    )
    with open(OUT / "cal1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2, default=str)
    with open(OUT / "cal1_cache_universe.json", "w", encoding="utf-8") as f:
        json.dump(cache_universe, f, ensure_ascii=False, indent=2, default=str)

    print(json.dumps(summ, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
