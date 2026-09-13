"""AUDIT-C: full-node graph probability calibration + probability/utility/execution
failure decomposition.

Baseline: bao1872/future_dev main 32cf36537deed8d5c78f62d788f20df9c1b231e0

Two questions (diagnostic only; no model / selector / execution / horizon change):

  (1) Static chain propagation calibration.
      compute_graph_ev() gives, per candidate node, (p_reach, p_loss_before,
      p_censor_before) summing to 1. The full-node outcome table gives the true
      per-node state (target_state in {NEXT/TARGET, LOSS, CENSOR}). For edges 0..4
      (AUDIT-A showed T2 selects edge 0-4 ~99.9%), compare predicted vs realised.

  (2) Failure-layer decomposition.
      Oracle label utility U_label = rr_ref if TARGET, -1 if LOSS, 0 if CENSOR
      (same utility semantics GraphEV assumes, risk normalised to 1R). Compare
      E[U_label | T0] vs E[U_label | T2]. If T2 already loses on oracle label utility,
      the failure is probability/selection (before EntryOpen). If T2 wins on label
      utility but loses on actual executable R, the failure is utility/execution.

No arbitrary verdict thresholds: the failure-layer call is sign-based only.

Scope: read-only on frozen caches. No H_i feature model, no adverse field, no batch fix,
no G1d, no WF2/WF3, no raw 5m rescan, no cache rebuild, no bar loading.
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

BASELINE_COMMIT = "32cf36537deed8d5c78f62d788f20df9c1b231e0"
CAL_EDGES = (0, 1, 2, 3, 4)


def reliability(pred, actual, n_bins=10):
    """Per-bin reliability table + ECE for a probability vs 0/1 target."""
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    d = pd.DataFrame({"p": pred, "y": actual})
    try:
        d["bin"] = pd.qcut(d["p"], n_bins, duplicates="drop")
    except Exception:
        d["bin"] = pd.cut(d["p"], n_bins)
    g = (d.groupby("bin", observed=True)
         .agg(n=("y", "size"), pred=("p", "mean"), actual=("y", "mean"))
         .reset_index(drop=True))
    g["gap"] = g["pred"] - g["actual"]
    ece = float((g["n"] / max(len(d), 1) * g["gap"].abs()).sum())
    return g, ece


def main():
    t_all = time.perf_counter()

    # ---- guards ----
    t0 = time.perf_counter()
    tr = load_transitions(FULL_UNIV, blocks=["TB1"], scope_tag="TB12_HARDENED")
    te = load_transitions(FULL_UNIV, blocks=["TB2"], scope_tag="TB12_HARDENED")
    assert len(tr) and len(te), "AUDIT_C_TRANSITION_CACHE_MISSING"
    test_start = pd.Timestamp(te["signal_trading_day"].min())
    tr_p, purge_diag = purge_train(tr, test_start)
    assert purge_diag["n_train_signals_before_purge"] == 21165
    assert purge_diag["n_train_signals_after_purge"] == 21135
    assert purge_diag["n_purged_signals"] == 30
    assert te["signal_id"].nunique() == 24954
    assert len(te) == 55945

    features, outcomes, provenance = load_full_node_tb2()
    assert provenance is not None, "STOP_AUDIT_C_CACHE_MISSING"
    load_seconds = time.perf_counter() - t0

    full = (features.loc[features["block"] == "TB2"]
            .sort_values(["signal_gid", "edge_index"], kind="stable")
            .reset_index(drop=True))
    assert not full.duplicated(["signal_gid", "edge_index"]).any(), "AUDIT_C_KEY_DUP"

    # ---- fit exact G0 once, one TB2 predict -> graph propagation ----
    t1 = time.perf_counter()
    g0 = fit_multinomial(tr_p, G0_NUM, G0_CAT)
    P = predict_reordered(g0, full, G0_NUM, G0_CAT)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6), "AUDIT_C_PROBA_SUM_FAIL"
    graph_ev, p_reach, p_loss_before, p_censor_before = compute_graph_ev(
        P[:, 0], P[:, 1], P[:, 2],
        full["signal_gid"].to_numpy(np.int64),
        full["rr_ref"].to_numpy(float),
    )
    full = full.copy()
    full["p_reach"] = p_reach
    full["p_loss_before"] = p_loss_before
    full["p_censor_before"] = p_censor_before
    full["graph_ev"] = graph_ev
    fit_predict_seconds = time.perf_counter() - t1

    # ---- T0/T2 selection MUST happen BEFORE merging outcome truth (leakage guard) ----
    sel = select_targets(full, full["graph_ev"].to_numpy(float),
                         np.zeros(len(full), dtype=float))
    t0_idx, t2_idx = sel["t0"], sel["t2"]

    # ---- ground truth per node: target_state in {NEXT/TARGET, LOSS, CENSOR} ----
    oc = (outcomes[outcomes["signal_gid"].isin(full["signal_gid"])]
          .sort_values(["signal_gid", "edge_index"], kind="stable")
          .reset_index(drop=True))
    assert not oc.duplicated(["signal_gid", "edge_index"]).any(), "AUDIT_C_OUT_KEY_DUP"
    full = full.merge(
        oc[["signal_gid", "edge_index", "target_state", "first_target_index",
            "first_stop_index"]],
        on=["signal_gid", "edge_index"], how="left", validate="one_to_one")
    assert full["target_state"].notna().all(), "AUDIT_C_OUTCOME_JOIN_MISSING"
    ts = full["target_state"].to_numpy(np.int64)
    full["y_reach"] = (ts == NEXT).astype(float)      # NEXT == TARGET/reached
    full["y_loss"] = (ts == LOSS).astype(float)
    full["y_censor"] = (ts == CENSOR).astype(float)
    assert bool(((full["y_reach"] + full["y_loss"] + full["y_censor"]) == 1).all()), \
        "AUDIT_C_STATE_NOT_PARTITION"
    rr_ref = full["rr_ref"].to_numpy(float)
    y_reach = full["y_reach"].to_numpy(float)
    y_loss = full["y_loss"].to_numpy(float)
    y_censor = full["y_censor"].to_numpy(float)
    # oracle label utility with GraphEV's own semantics (risk normalised to 1R)
    u_label = rr_ref * y_reach + (-1.0) * y_loss + 0.0 * y_censor
    full["u_label"] = u_label

    # ---- (1a) per-edge calibration summary ----
    rows = []
    for e in CAL_EDGES:
        g = full[full["edge_index"] == e]
        if len(g) == 0:
            continue
        rows.append(dict(
            edge_index=int(e), n_nodes=int(len(g)),
            pred_reach=float(g["p_reach"].mean()),
            actual_reach=float(g["y_reach"].mean()),
            gap_reach=float(g["p_reach"].mean() - g["y_reach"].mean()),
            pred_loss=float(g["p_loss_before"].mean()),
            actual_loss=float(g["y_loss"].mean()),
            gap_loss=float(g["p_loss_before"].mean() - g["y_loss"].mean()),
            pred_censor=float(g["p_censor_before"].mean()),
            actual_censor=float(g["y_censor"].mean()),
            gap_censor=float(g["p_censor_before"].mean() - g["y_censor"].mean()),
            brier_reach=float(((g["p_reach"] - g["y_reach"]) ** 2).mean()),
            brier_loss=float(((g["p_loss_before"] - g["y_loss"]) ** 2).mean()),
            brier_censor=float(((g["p_censor_before"] - g["y_censor"]) ** 2).mean()),
        ))
    cal_by_edge = pd.DataFrame(rows)
    cal_by_edge.to_csv(OUT / "g2_graph_calibration_by_edge.csv", index=False)

    # ---- (1b) reach reliability deciles (pooled edges 0-4) ----
    sub = full[full["edge_index"].isin(CAL_EDGES)]
    rel, ece_reach = reliability(sub["p_reach"].to_numpy(), sub["y_reach"].to_numpy())
    rel.insert(0, "population", "TB2_full_nodes_edges0_4")
    rel.to_csv(OUT / "g2_graph_calibration_reach_deciles.csv", index=False)

    # ---- (1b') graph_ev (EV) calibration vs realised label utility (edges 0-4) ----
    def continuous_calibration(pred, actual, n_bins=10):
        d = pd.DataFrame({"p": np.asarray(pred, float), "y": np.asarray(actual, float)})
        try:
            d["bin"] = pd.qcut(d["p"], n_bins, duplicates="drop")
        except Exception:
            d["bin"] = pd.cut(d["p"], n_bins)
        g = (d.groupby("bin", observed=True)
             .agg(n=("y", "size"), pred=("p", "mean"), actual=("y", "mean"))
             .reset_index(drop=True))
        g["gap"] = g["pred"] - g["actual"]
        ece = float((g["n"] / max(len(d), 1) * g["gap"].abs()).sum())
        return g, ece

    ev_rel, ev_ece = continuous_calibration(sub["graph_ev"].to_numpy(),
                                            sub["u_label"].to_numpy())
    ev_rel.insert(0, "population", "TB2_full_nodes_edges0_4")
    ev_rel.to_csv(OUT / "g2_graph_calibration_ev_deciles.csv", index=False)

    # selected-node reach calibration (T0 / T2)
    sel_reach = {}
    for nm, idx in (("T0", t0_idx), ("T2", t2_idx)):
        pr = full["p_reach"].to_numpy(float)[idx]
        yr = full["y_reach"].to_numpy(float)[idx]
        sel_reach[nm] = dict(n=int(len(idx)), pred_reach=float(pr.mean()),
                             actual_reach=float(yr.mean()),
                             gap_reach=float(pr.mean() - yr.mean()))

    # ---- (2) failure-layer decomposition (oracle label utility) ----
    prev = pd.read_csv(OUT / "g2_summary.csv")
    prev_ev = dict(zip(prev["selector"], prev["EV_per_signal"]))

    dec = {}
    for nm, idx in (("T0", t0_idx), ("T2", t2_idx)):
        key = "T0_Nearest" if nm == "T0" else "T2_GraphEV"
        dec[nm] = dict(
            n=int(len(idx)),
            label_utility_E=float(u_label[idx].mean()),
            mean_graph_ev_predicted=float(full["graph_ev"].to_numpy(float)[idx].mean()),
            actual_executable_R_E=float(prev_ev[key]),
            mean_rr_ref=float(rr_ref[idx].mean()),
            target_hit_rate=float(y_reach[idx].mean()),
            loss_rate=float(y_loss[idx].mean()),
            censor_rate=float(y_censor[idx].mean()),
            mean_selected_edge_index=float(
                full["edge_index"].to_numpy(np.int64)[idx].mean()),
        )
    label_gap = dec["T2"]["label_utility_E"] - dec["T0"]["label_utility_E"]
    actual_gap = dec["T2"]["actual_executable_R_E"] - dec["T0"]["actual_executable_R_E"]
    if label_gap <= 0:
        failure_layer = "PROBABILITY_OR_SELECTION"
    elif label_gap > 0 and actual_gap <= 0:
        failure_layer = "UTILITY_OR_EXECUTION"
    else:
        failure_layer = "NO_FAILURE_AT_THESE_TARGETS"
    utility_decomp = dict(
        layer_verdict=failure_layer,
        label_utility_gap_T2_minus_T0=float(label_gap),
        actual_R_gap_T2_minus_T0=float(actual_gap),
        per_selector=dec,
        note=("U_label uses GraphEV's utility semantics (RR_ref on TARGET, -1 on LOSS, "
              "0 on CENSOR); actual_executable_R_E is read from frozen g2_summary.csv "
              "(paired, same T0/T2 selections)."),
    )
    pd.DataFrame([dict(selector=k, **v) for k, v in dec.items()]).to_csv(
        OUT / "g2_utility_decomposition.csv", index=False)

    # ---- summary ----
    summary = {
        "audit": "G2_FULLNODE_GRAPH_CALIBRATION_AND_FAILURE_LAYER_V1",
        "baseline_commit": BASELINE_COMMIT,
        "purge": purge_diag,
        "calibration_by_edge": cal_by_edge.to_dict("records"),
        "reach_reliability_ece_edges0_4": ece_reach,
        "graph_ev_reliability_ece_edges0_4": ev_ece,
        "selected_node_reach_calibration": sel_reach,
        "utility_decomposition": utility_decomp,
        "timing_seconds": dict(load_seconds=load_seconds,
                               fit_predict_seconds=fit_predict_seconds,
                               total_seconds=time.perf_counter() - t_all),
        "interpretation_contract": (
            "Diagnostic only. T0/T2 selections are the frozen G2 selections. The failure "
            "layer call is sign-based (no arbitrary threshold)."
        ),
    }
    with open(OUT / "g2_graph_calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
