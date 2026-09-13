"""AUDIT-A: transition risk-set -> full decision graph topology support / OOD audit.

Baseline: bao1872/future_dev main 09905ed60e96fa473068824d0a9643127bab8fb2

Sole question:
  "G0 is trained on the real transition risk-set (short prefix, ~2 edges/signal),
   but T2 selects from the FULL graph (mean ~32 candidate nodes/signal). Does T2
   systematically use topology support OUTSIDE the G0 edge_index categorical training
   support?"

edge_index is a CATEGORICAL feature in G0_CAT and is encoded with
OneHotEncoder(handle_unknown="ignore"), so an unseen edge category is NOT a graceful
ordinal extrapolation — its dummy block is all zeros. This script verifies that sklearn
semantics and then reports the actual support numbers.

Hard scope (this round): descriptive audit only. No model / selector / execution /
horizon change. No H_i, no adverse field, no batch, no sweep, no G1d, no WF2/WF3.
Reuse frozen exact-hash caches. No rebuild.
"""
import json
import sys
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
)

from research.liquidity_oracle_atlas.run_5m_graph_target_decision_v1 import (  # noqa: E402
    load_full_node_tb2,
    predict_reordered,
    compute_graph_ev,
    select_targets,
    FULL_UNIV,
    OUT,
)

BASELINE_COMMIT = "09905ed60e96fa473068824d0a9643127bab8fb2"


def qhigher(x, q):
    x = np.asarray(x, dtype=np.int64)
    if len(x) == 0:
        return np.nan
    return int(np.quantile(x, q, method="higher"))


def verify_onehot_unknown_semantics():
    """Empirically confirm sklearn's OneHotEncoder(handle_unknown='ignore') semantics:
    a category unseen during fit yields an all-zero dummy block (not an error / not a
    nearest-category fallback). Do NOT infer this from docs."""
    from sklearn.preprocessing import OneHotEncoder

    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(np.array([[0], [1], [2]]))
    known = np.asarray(enc.transform(np.array([[1]])).todense())
    unknown = np.asarray(enc.transform(np.array([[9]])).todense())
    ok_known = bool(known.shape == (1, 3) and np.array_equal(known, np.array([[0, 1, 0]])))
    ok_unknown = bool(unknown.shape == (1, 3) and np.all(unknown == 0))
    return ok_known, ok_unknown, known.tolist(), unknown.tolist()


def summarize_edge_population(name, edge, train_edge_counts, train_p95, train_p99,
                              train_max):
    edge = np.asarray(edge, dtype=np.int64)
    if len(edge) == 0:
        return {"population": name, "n": 0}
    # per-edge_index observation count in the train risk-set; unseen category -> 0
    support = (pd.Series(edge).map(train_edge_counts).fillna(0).to_numpy(dtype=np.int64))
    return {
        "population": name,
        "n": int(len(edge)),
        "edge_mean": float(edge.mean()),
        "edge_p50": qhigher(edge, 0.50),
        "edge_p90": qhigher(edge, 0.90),
        "edge_p95": qhigher(edge, 0.95),
        "edge_p99": qhigher(edge, 0.99),
        "edge_max": int(edge.max()),
        # tail usage (not necessarily strict OOD)
        "share_gt_train_p95": float(np.mean(edge > train_p95)),
        "share_gt_train_p99": float(np.mean(edge > train_p99)),
        # strict categorical OOD
        "share_gt_train_max": float(np.mean(edge > train_max)),
        "share_unseen_exact_category": float(np.mean(support == 0)),
        # thin support
        "share_train_count_le_5": float(np.mean(support <= 5)),
        "share_train_count_le_20": float(np.mean(support <= 20)),
        "share_train_count_le_100": float(np.mean(support <= 100)),
        "median_train_count_for_used_edge": float(np.median(support)),
    }


def main():
    ok_known, ok_unknown, known, unknown = verify_onehot_unknown_semantics()
    print(f"[OHE] known=[0,1,2] transform(1) -> {known} ; transform(9) -> {unknown}")
    if not ok_known:
        raise SystemExit("STOP_AUDIT_A_ENCODER_SEMANTICS_UNEXPECTED (known encode wrong)")
    if not ok_unknown:
        raise SystemExit("STOP_AUDIT_A_ENCODER_SEMANTICS_UNEXPECTED (unknown not all-zero)")

    # ---- 1. exact hardened transition risk-set ----
    tr = load_transitions(FULL_UNIV, blocks=["TB1"], scope_tag="TB12_HARDENED")
    te = load_transitions(FULL_UNIV, blocks=["TB2"], scope_tag="TB12_HARDENED")
    if len(tr) == 0 or len(te) == 0:
        raise SystemExit("STOP_AUDIT_A_TRANSITION_CACHE_MISSING")

    test_start = pd.Timestamp(te["signal_trading_day"].min())
    tr_p, purge_diag = purge_train(tr, test_start)

    assert tr_p["signal_id"].nunique() == 21135, "AUDIT_A_TRAIN_SAMPLE_MISMATCH"
    assert te["signal_id"].nunique() == 24954, "AUDIT_A_TEST_SAMPLE_MISMATCH"
    assert len(te) == 55945, "AUDIT_A_TEST_EDGE_MISMATCH"
    print(f"[PURGE] {purge_diag}")

    # ---- 2. exact G2 full-node cache (no rebuild) ----
    features, outcomes, provenance = load_full_node_tb2()
    if provenance is None:
        raise SystemExit("STOP_AUDIT_A_CACHE_MISSING")

    full = (features.loc[features["block"] == "TB2"]
            .sort_values(["signal_gid", "edge_index"], kind="stable")
            .reset_index(drop=True))
    if len(full) == 0:
        raise SystemExit("STOP_AUDIT_A_NO_FULL_NODES")
    assert not full.duplicated(["signal_gid", "edge_index"]).any(), \
        "AUDIT_A_FULL_NODE_KEY_DUP"

    # ---- 3. train edge-index categorical support ----
    edge_counts = (tr_p["edge_index"].astype(int).value_counts().sort_index())
    train_edge = tr_p["edge_index"].to_numpy(np.int64)
    train_p95 = qhigher(train_edge, 0.95)
    train_p99 = qhigher(train_edge, 0.99)
    train_max = int(train_edge.max())

    # ---- 4. fit EXACT G0 once (same tr_p / G0_NUM / G0_CAT) ----
    g0 = fit_multinomial(tr_p, G0_NUM, G0_CAT)

    enc = g0.named_steps["pre"].named_transformers_["cat"]
    edge_categories = np.asarray(enc.categories_[0]).astype(np.int64)
    train_unique = np.sort(tr_p["edge_index"].unique().astype(np.int64))
    assert np.array_equal(np.sort(edge_categories), train_unique), \
        "STOP_AUDIT_A_ENCODER_SUPPORT_MISMATCH"

    # ---- 5. full graph score + T0/T2 selector ----
    p = predict_reordered(g0, full, G0_NUM, G0_CAT)
    graph_ev, p_reach, p_loss_before, p_censor_before = compute_graph_ev(
        p[:, 0], p[:, 1], p[:, 2],
        full["signal_gid"].to_numpy(np.int64),
        full["rr_ref"].to_numpy(float),
    )

    # T1 economics is out of scope this round; zeros satisfy the selector interface.
    dummy_indep = np.zeros(len(full), dtype=float)
    sel = select_targets(full, graph_ev, dummy_indep)

    t0_idx = sel["t0"]
    t2_idx = sel["t2"]
    t0_edge = full["edge_index"].to_numpy(np.int64)[t0_idx]
    t2_edge = full["edge_index"].to_numpy(np.int64)[t2_idx]

    # ---- 6. core populations ----
    rows = [
        summarize_edge_population("TB1_train_riskset_purged", tr_p["edge_index"],
                                  edge_counts, train_p95, train_p99, train_max),
        summarize_edge_population("TB2_transition_riskset", te["edge_index"],
                                  edge_counts, train_p95, train_p99, train_max),
        summarize_edge_population("TB2_full_nodes", full["edge_index"],
                                  edge_counts, train_p95, train_p99, train_max),
        summarize_edge_population("TB2_T0_selected", t0_edge,
                                  edge_counts, train_p95, train_p99, train_max),
        summarize_edge_population("TB2_T2_selected", t2_edge,
                                  edge_counts, train_p95, train_p99, train_max),
    ]
    pop = pd.DataFrame(rows)

    # ---- 7. signal-level candidate graph support ----
    graph_max = (full.groupby("signal_gid", sort=False)["edge_index"]
                 .max().to_numpy(np.int64))
    graph_n_nodes = (full.groupby("signal_gid", sort=False)
                     .size().to_numpy(np.int64))

    signal_support = {
        "n_fullnode_signals": int(full["signal_gid"].nunique()),
        "n_transition_test_signals": int(te["signal_id"].nunique()),
        "n_zero_node_signals_vs_transition":
            int(te["signal_id"].nunique() - full["signal_gid"].nunique()),
        "candidate_nodes_mean": float(graph_n_nodes.mean()),
        "candidate_nodes_p50": qhigher(graph_n_nodes, 0.50),
        "candidate_nodes_p95": qhigher(graph_n_nodes, 0.95),
        "candidate_nodes_p99": qhigher(graph_n_nodes, 0.99),
        "candidate_nodes_max": int(graph_n_nodes.max()),
        "share_signals_graph_extends_gt_train_p95": float(np.mean(graph_max > train_p95)),
        "share_signals_graph_extends_gt_train_p99": float(np.mean(graph_max > train_p99)),
        "share_signals_graph_extends_gt_train_max": float(np.mean(graph_max > train_max)),
    }

    # ---- 8. T2 chosen-edge exact support ----
    t2_support = (pd.Series(t2_edge).map(edge_counts).fillna(0).to_numpy(np.int64))
    selected = pd.DataFrame({
        "signal_gid": full["signal_gid"].to_numpy(np.int64)[t2_idx],
        "selected_edge_index": t2_edge,
        "selected_edge_train_count": t2_support,
        "selected_edge_unseen": t2_support == 0,
        "selected_edge_gt_train_p95": t2_edge > train_p95,
        "selected_edge_gt_train_p99": t2_edge > train_p99,
        "selected_edge_gt_train_max": t2_edge > train_max,
        "selected_graph_ev": graph_ev[t2_idx],
        "selected_p_reach": p_reach[t2_idx],
        "selected_rr_ref": full["rr_ref"].to_numpy(float)[t2_idx],
    })

    # ---- 9. outputs (factual; no threshold tuning) ----
    (edge_counts.rename_axis("edge_index")
     .rename("n_train_edges").reset_index()
     .to_csv(OUT / "g2_topology_support_train_edge_counts.csv", index=False))
    pop.to_csv(OUT / "g2_topology_support_populations.csv", index=False)
    selected.to_csv(OUT / "g2_topology_support_selected.csv", index=False)

    strict_unseen_share = float(np.mean(t2_support == 0))
    verdict = ("STRICT_EDGE_CATEGORY_OOD_PRESENT" if strict_unseen_share > 0
               else "NO_STRICT_EDGE_CATEGORY_OOD")

    report = {
        "audit": "G2_RISKSET_TO_FULLGRAPH_TOPOLOGY_SUPPORT_V1",
        "baseline_commit": BASELINE_COMMIT,
        "purge": purge_diag,
        "ohe_semantics": {
            "known_encode_ok": ok_known,
            "unknown_all_zero_ok": ok_unknown,
            "transform_known_1": known,
            "transform_unknown_9": unknown,
        },
        "train_support": {
            "edge_p95": train_p95,
            "edge_p99": train_p99,
            "edge_max": train_max,
            "encoder_edge_categories": edge_categories.tolist(),
        },
        "signal_support": signal_support,
        "t2_selected": {
            "n": int(len(t2_edge)),
            "edge_mean": float(t2_edge.mean()),
            "edge_p50": qhigher(t2_edge, 0.50),
            "edge_p95": qhigher(t2_edge, 0.95),
            "edge_p99": qhigher(t2_edge, 0.99),
            "edge_max": int(t2_edge.max()),
            "share_unseen_exact_category": strict_unseen_share,
            "share_gt_train_p95": float(np.mean(t2_edge > train_p95)),
            "share_gt_train_p99": float(np.mean(t2_edge > train_p99)),
            "share_gt_train_max": float(np.mean(t2_edge > train_max)),
            "share_train_count_le_20": float(np.mean(t2_support <= 20)),
            "share_train_count_le_100": float(np.mean(t2_support <= 100)),
        },
        "verdict": verdict,
        "interpretation_contract": (
            "STRICT_EDGE_CATEGORY_OOD_PRESENT only shows that T2 uses nodes outside the "
            "G0 edge_index categorical training support. It does NOT by itself prove this "
            "is the sole cause of the G2 economic failure."
        ),
    }

    with open(OUT / "g2_topology_support_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
