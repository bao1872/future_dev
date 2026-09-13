"""AUDIT-D: GraphEV error decomposition + selection / winner's-curse audit.

Baseline: bao1872/future_dev main 044d8254f4ff7f8ff8d17a2ea3950acea6f8a7f4

AUDIT-C found: reach marginal is well calibrated, but graph_ev is severely over-optimistic
in its top decile, and T2 (argmax graph_ev) loses to T0 on oracle label utility. AUDIT-D
decomposes that EV error exactly and audits whether argmax is picking its own estimation
error (winner's curse).

Exact node identity (row-wise):
  pred_gain      = rr_ref * p_reach
  actual_gain    = rr_ref * y_reach
  pred_loss_cost = p_loss_before
  actual_loss    = y_loss
  graph_ev       = pred_gain - pred_loss_cost
  u_label        = actual_gain - actual_loss
  ev_error       = graph_ev - u_label == gain_error - loss_error

Selection identity (per signal):
  pred_delta = graph_ev_T2 - graph_ev_T0
  oracle_delta = u_label_T2 - u_label_T0
  selection_error_delta = error_T2 - error_T0 == pred_delta - oracle_delta

Verdict is only from the trading-day block bootstrap CI of oracle_delta (no arbitrary
threshold). EV-error source is reported numerically only.

Scope: diagnostic only. No model / GraphEV / selector / execution / horizon change. No
H_i, duration model, adverse field, batch/sweep redesign, EntryOpen model, calibration
(isotonic/Platt), WF2/WF3. No cache rebuild, no raw bars rescan.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.liquidity_oracle_atlas.run_5m_graph_probability_v1 import (  # noqa: E402
    load_transitions,
    purge_train,
    fit_multinomial,
    G0_NUM,
    G0_CAT,
    NEXT,
    LOSS,
    CENSOR,
)

from research.liquidity_oracle_atlas.run_5m_graph_target_decision_v1 import (  # noqa: E402
    load_full_node_tb2,
    predict_reordered,
    compute_graph_ev,
    select_targets,
    FULL_UNIV,
    OUT,
)

BASELINE_COMMIT = "044d8254f4ff7f8ff8d17a2ea3950acea6f8a7f4"
CAL_EDGES = (0, 1, 2, 3, 4)
# AUDIT-C reproduction targets
RC_T0_U, RC_T0_G = -0.0185497626816751, -0.03189917255825394
RC_T2_U, RC_T2_G = -0.03572944047112938, +0.004492992672146745
RC_T2_PR, RC_T2_AR = 0.3751538168282413, 0.3693596217039352
RC_ACTUAL_DELTA = -0.023619815950091383


def agg(df):
    return dict(
        n=int(len(df)),
        mean_rr_ref=float(df["rr_ref"].mean()),
        pred_reach=float(df["p_reach"].mean()),
        actual_reach=float(df["y_reach"].mean()),
        reach_error=float((df["p_reach"] - df["y_reach"]).mean()),
        pred_gain=float(df["pred_gain"].mean()),
        actual_gain=float(df["actual_gain"].mean()),
        gain_error=float(df["gain_error"].mean()),
        pred_loss=float(df["pred_loss"].mean()),
        actual_loss=float(df["y_loss"].mean()),
        loss_error=float(df["loss_error"].mean()),
        pred_censor=float(df["p_censor"].mean()),
        actual_censor=float(df["y_censor"].mean()),
        pred_graph_ev=float(df["graph_ev"].mean()),
        actual_u_label=float(df["u_label"].mean()),
        ev_error=float(df["ev_error"].mean()),
    )


def day_bootstrap(values, days, n_boot=1000, seed=20260913):
    d = pd.DataFrame({"v": np.asarray(values, float),
                      "day": pd.to_datetime(np.asarray(days))})
    daily = d.groupby("day")["v"].mean().to_numpy()
    nd = len(daily)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nd, size=(n_boot, nd))
    boot = daily[idx].mean(axis=1)
    return dict(mean=float(daily.mean()), ci_lo=float(np.percentile(boot, 2.5)),
                ci_hi=float(np.percentile(boot, 97.5)), n_days=int(nd))


def main():
    t_all = time.perf_counter()

    # ---- guards + load (frozen) ----
    t0 = time.perf_counter()
    tr = load_transitions(FULL_UNIV, blocks=["TB1"], scope_tag="TB12_HARDENED")
    te = load_transitions(FULL_UNIV, blocks=["TB2"], scope_tag="TB12_HARDENED")
    test_start = pd.Timestamp(te["signal_trading_day"].min())
    tr_p, purge_diag = purge_train(tr, test_start)
    assert purge_diag["n_train_signals_after_purge"] == 21135
    assert te["signal_id"].nunique() == 24954 and len(te) == 55945

    features, outcomes, provenance = load_full_node_tb2()
    assert provenance is not None, "STOP_AUDIT_D_CACHE_MISSING"
    full = (features.loc[features["block"] == "TB2"]
            .sort_values(["signal_gid", "edge_index"], kind="stable")
            .reset_index(drop=True))
    assert len(full) == 812027, "AUDIT_D_FULL_NODES_MISMATCH"
    assert not full.duplicated(["signal_gid", "edge_index"]).any(), "AUDIT_D_KEY_DUP"
    load_seconds = time.perf_counter() - t0

    # ---- G0 fit + predict + propagation ----
    t1 = time.perf_counter()
    g0 = fit_multinomial(tr_p, G0_NUM, G0_CAT)
    P = predict_reordered(g0, full, G0_NUM, G0_CAT)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
    graph_ev, p_reach, p_loss, p_censor = compute_graph_ev(
        P[:, 0], P[:, 1], P[:, 2],
        full["signal_gid"].to_numpy(np.int64), full["rr_ref"].to_numpy(float))
    fit_predict_seconds = time.perf_counter() - t1

    # ---- T0/T2 selection BEFORE outcome merge (leakage guard) ----
    sel = select_targets(full, graph_ev, np.zeros(len(full), dtype=float))
    t0_idx, t2_idx = sel["t0"], sel["t2"]

    # ---- outcome truth ----
    oc = (outcomes[outcomes["signal_gid"].isin(full["signal_gid"])]
          .sort_values(["signal_gid", "edge_index"], kind="stable").reset_index(drop=True))
    full = full.merge(oc[["signal_gid", "edge_index", "target_state"]],
                      on=["signal_gid", "edge_index"], how="left", validate="one_to_one")
    assert full["target_state"].notna().all()
    ts = full["target_state"].to_numpy(np.int64)
    y_reach = (ts == NEXT).astype(float)
    y_loss = (ts == LOSS).astype(float)
    y_censor = (ts == CENSOR).astype(float)
    assert bool(((y_reach + y_loss + y_censor) == 1).all())

    # ---- exact node-level EV error decomposition ----
    rr_ref = full["rr_ref"].to_numpy(float)
    pred_gain = rr_ref * p_reach
    actual_gain = rr_ref * y_reach
    gain_error = pred_gain - actual_gain
    loss_error = p_loss - y_loss
    u_label = actual_gain - y_loss
    ev_error = graph_ev - u_label
    assert np.allclose(ev_error, gain_error - loss_error, atol=1e-12), \
        "STOP_AUDIT_D_EV_IDENTITY_FAIL"
    assert np.allclose(graph_ev, pred_gain - p_loss, atol=1e-12), "AUDIT_D_GEV_IDENTITY_FAIL"

    nodes = pd.DataFrame(dict(
        edge_index=full["edge_index"].to_numpy(np.int64), rr_ref=rr_ref,
        p_reach=p_reach, y_reach=y_reach, p_loss=p_loss, y_loss=y_loss,
        p_censor=p_censor, y_censor=y_censor, graph_ev=graph_ev, u_label=u_label,
        pred_gain=pred_gain, actual_gain=actual_gain, gain_error=gain_error,
        pred_loss=p_loss, loss_error=loss_error, ev_error=ev_error))

    # ---- reproduce AUDIT-C exactly ----
    t0_nodes, t2_nodes = nodes.iloc[t0_idx], nodes.iloc[t2_idx]
    rep = dict(
        t0_u_label=float(t0_nodes["u_label"].mean()), t0_graph_ev=float(t0_nodes["graph_ev"].mean()),
        t2_u_label=float(t2_nodes["u_label"].mean()), t2_graph_ev=float(t2_nodes["graph_ev"].mean()),
        t2_pred_reach=float(t2_nodes["p_reach"].mean()), t2_actual_reach=float(t2_nodes["y_reach"].mean()))
    ok = (abs(rep["t0_u_label"] - RC_T0_U) < 1e-9 and abs(rep["t0_graph_ev"] - RC_T0_G) < 1e-9
          and abs(rep["t2_u_label"] - RC_T2_U) < 1e-9 and abs(rep["t2_graph_ev"] - RC_T2_G) < 1e-9
          and abs(rep["t2_pred_reach"] - RC_T2_PR) < 1e-9
          and abs(rep["t2_actual_reach"] - RC_T2_AR) < 1e-9)
    if not ok:
        raise SystemExit(f"STOP_AUDIT_D_AUDITC_REPRODUCTION_FAIL {rep}")

    # ---- signal-level deltas ----
    g_t0 = graph_ev[t0_idx]; g_t2 = graph_ev[t2_idx]
    u_t0 = u_label[t0_idx]; u_t2 = u_label[t2_idx]
    pred_delta = g_t2 - g_t0
    oracle_delta = u_t2 - u_t0
    err_t0 = g_t0 - u_t0
    err_t2 = g_t2 - u_t2
    sed = err_t2 - err_t0
    assert np.allclose(pred_delta - oracle_delta, sed, atol=1e-12), \
        "STOP_AUDIT_D_SELECTION_DELTA_IDENTITY_FAIL"
    switched = t2_idx != t0_idx
    n_same, n_switch = int((~switched).sum()), int(switched.sum())
    assert n_same == 14778 and n_switch == 10176, \
        f"AUDIT_D_SELECTED_COUNTS_MISMATCH same={n_same} switch={n_switch}"
    sig_day = full.iloc[t0_idx]["trading_day"].to_numpy()
    t2_edge = full["edge_index"].to_numpy(np.int64)[t2_idx]
    cand_count = (full.groupby("signal_gid", sort=False).size()
                  .reindex(full.iloc[t0_idx]["signal_gid"].to_numpy()).to_numpy())

    sel_mask = dict(ALL=np.ones(len(t0_idx), bool), SAME=~switched, SWITCHED=switched)
    switch_rows = []
    for nm, mk in sel_mask.items():
        od = oracle_delta[mk]
        switch_rows.append(dict(
            population=nm, n=int(mk.sum()),
            mean_pred_delta=float(pred_delta[mk].mean()),
            mean_oracle_delta=float(od.mean()),
            mean_selection_error_delta=float(sed[mk].mean()),
            freq_oracle_delta_gt_0=float(np.mean(od > 0)) if mk.sum() else np.nan,
            freq_oracle_delta_lt_0=float(np.mean(od < 0)) if mk.sum() else np.nan,
            freq_oracle_delta_eq_0=float(np.mean(od == 0)) if mk.sum() else np.nan))
    switch = pd.DataFrame(switch_rows)
    switch.to_csv(OUT / "g2_ev_selection_switch.csv", index=False)

    # ---- populations ----
    pop_rows = []
    for nm, dfx in (("ALL_EDGES0_4", nodes[nodes["edge_index"].isin(CAL_EDGES)]),
                    ("T0_SELECTED", t0_nodes), ("T2_SELECTED", t2_nodes),
                    ("SAME_SELECTION", nodes.iloc[t2_idx[~switched]]),
                    ("SWITCHED", nodes.iloc[t2_idx[switched]])):
        pop_rows.append(dict(population=nm, **agg(dfx)))
    pops = pd.DataFrame(pop_rows)
    pops.to_csv(OUT / "g2_ev_error_populations.csv", index=False)

    # ---- deciles (edges 0-4) ----
    sub = nodes[nodes["edge_index"].isin(CAL_EDGES)].copy()
    sub["bin"] = pd.qcut(sub["graph_ev"], 10, duplicates="drop")
    dec_rows = []
    for i, (b, g) in enumerate(sub.groupby("bin", observed=True), start=1):
        dec_rows.append(dict(graph_ev_decile=i, bin_lo=float(b.left), bin_hi=float(b.right),
                             **agg(g)))
    dec = pd.DataFrame(dec_rows)
    dec.to_csv(OUT / "g2_ev_error_deciles.csv", index=False)

    # ---- selected-node EV-error source (T0/T2) ----
    decomp = {}
    for nm, dfx in (("T0", t0_nodes), ("T2", t2_nodes)):
        ge, le, pe = (float(dfx["gain_error"].mean()), float(dfx["loss_error"].mean()),
                      float(dfx["ev_error"].mean()))
        assert abs(pe - (ge - le)) < 1e-9, "AUDIT_D_T0T2_DECOMP_IDENTITY_FAIL"
        decomp[nm] = dict(mean_gain_error=ge, mean_loss_error=le, mean_ev_error=pe,
                          abs_gain_contribution=abs(ge), abs_loss_contribution=abs(le),
                          larger_source=("gain" if abs(ge) >= abs(le) else "loss"))

    # ---- bootstrap (ALL + SWITCHED) ----
    boot_rows = []
    for nm, mk in (("ALL", np.ones(len(t0_idx), bool)), ("SWITCHED", switched)):
        for label, vals in (("oracle_delta", oracle_delta), ("selection_error_delta", sed),
                            ("pred_delta", pred_delta)):
            b = day_bootstrap(vals[mk], sig_day[mk])
            boot_rows.append(dict(population=nm, metric=label, **b))
    boot = pd.DataFrame(boot_rows)

    # ---- execution drag (frozen replay rewards; no raw bars) ----
    rw_path = OUT / "g2_closure_replay_rewards.csv"
    assert rw_path.exists(), "STOP_AUDIT_D_REPLAY_REWARDS_MISSING"
    rw = pd.read_csv(rw_path)
    for c in ("signal_gid", "reward_t0", "reward_t2"):
        assert c in rw.columns, f"STOP_AUDIT_D_REPLAY_COL_MISSING {c}"
    assert not rw["signal_gid"].duplicated().any(), "STOP_AUDIT_D_REPLAY_KEY_DUP"
    sig_ids = full.iloc[t0_idx]["signal_gid"].to_numpy()
    rw = rw.set_index("signal_gid").reindex(sig_ids)
    assert rw["reward_t0"].notna().all() and rw["reward_t2"].notna().all(), \
        "STOP_AUDIT_D_REPLAY_ONE_TO_ONE_MAP_FAIL"
    actual_delta = (rw["reward_t2"] - rw["reward_t0"]).to_numpy(float)
    assert abs(float(actual_delta.mean()) - RC_ACTUAL_DELTA) < 1e-9, \
        "STOP_AUDIT_D_ACTUAL_DELTA_REPRODUCTION_FAIL"
    execution_drag = actual_delta - oracle_delta
    for nm, mk in (("ALL", np.ones(len(t0_idx), bool)), ("SWITCHED", switched)):
        boot_rows.append(dict(population=nm, metric="execution_drag",
                              **day_bootstrap(execution_drag[mk], sig_day[mk])))
        boot_rows.append(dict(population=nm, metric="actual_delta",
                              **day_bootstrap(actual_delta[mk], sig_day[mk])))
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(OUT / "g2_ev_bootstrap.csv", index=False)

    exec_drag = dict(
        ALL=dict(mean_oracle_delta=float(oracle_delta.mean()),
                 mean_actual_delta=float(actual_delta.mean()),
                 mean_execution_drag=float(execution_drag.mean())),
        SWITCHED=dict(mean_oracle_delta=float(oracle_delta[switched].mean()),
                      mean_actual_delta=float(actual_delta[switched].mean()),
                      mean_execution_drag=float(execution_drag[switched].mean())))

    # ---- by selected edge (SWITCHED) ----
    be_rows = []
    for lab, mk in (("1", t2_edge == 1), ("2", t2_edge == 2), ("3", t2_edge == 3),
                    ("4", t2_edge == 4), ("5+", t2_edge >= 5)):
        mk = mk & switched
        if mk.sum() == 0:
            continue
        be_rows.append(dict(selected_edge=lab, n=int(mk.sum()),
                            mean_rr_ref=float(rr_ref[t2_idx][mk].mean()),
                            mean_pred_delta=float(pred_delta[mk].mean()),
                            mean_oracle_delta=float(oracle_delta[mk].mean()),
                            mean_selection_error_delta=float(sed[mk].mean()),
                            mean_execution_drag=float(execution_drag[mk].mean())))
    by_edge = pd.DataFrame(be_rows)
    by_edge.to_csv(OUT / "g2_ev_selection_by_edge.csv", index=False)

    # ---- candidate density quartiles ----
    q = pd.qcut(cand_count, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    dens = pd.DataFrame(dict(q=q.to_numpy(), switched=switched,
                             sed=sed, oracle_delta=oracle_delta))
    dg = dens.groupby("q", observed=True).agg(
        n=("switched", "size"), switch_rate=("switched", "mean"),
        mean_selection_error_delta=("sed", "mean"),
        mean_oracle_delta=("oracle_delta", "mean")).reset_index()
    dg.rename(columns={"q": "candidate_density_quartile"}, inplace=True)
    dg.to_csv(OUT / "g2_ev_selection_by_density.csv", index=False)

    # ---- verdict (oracle_delta bootstrap CI only) ----
    od_all = boot[(boot["population"] == "ALL") & (boot["metric"] == "oracle_delta")].iloc[0]
    if od_all["ci_hi"] < 0:
        verdict = "PRE_EXECUTION_SELECTION_FAILURE_CONFIRMED"
    elif od_all["mean"] < 0:
        verdict = "PRE_EXECUTION_SELECTION_FAILURE_AMBIGUOUS"
    else:
        verdict = "NO_PREEXECUTION_SELECTION_FAILURE"

    summary = {
        "audit": "G2_EV_ERROR_DECOMPOSITION_AND_SELECTION_BIAS_V1",
        "baseline_commit": BASELINE_COMMIT,
        "purge": purge_diag,
        "reproduction": rep,
        "selection_counts": dict(same=n_same, switched=n_switch),
        "populations": pops.to_dict("records"),
        "deciles": dec.to_dict("records"),
        "selected_decomposition": decomp,
        "selection_switch": switch.to_dict("records"),
        "bootstrap": boot.to_dict("records"),
        "execution_drag": exec_drag,
        "by_selected_edge_switched": by_edge.to_dict("records"),
        "by_candidate_density": dg.to_dict("records"),
        "verdict": verdict,
        "timing_seconds": dict(load_seconds=load_seconds,
                               fit_predict_seconds=fit_predict_seconds,
                               total_seconds=time.perf_counter() - t_all),
        "interpretation_contract": (
            "Diagnostic only. Verdict is taken only from the trading-day block bootstrap "
            "CI of oracle_delta (no arbitrary threshold). EV-error source is numeric only."),
    }
    with open(OUT / "g2_ev_error_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
