"""AUDIT-B: missing duration state / arrival-time residual audit.

Baseline: bao1872/future_dev main 32c561b72a29ace4afa61d872b0f65af0ddfefcf

Sole question:
  At the moment the market ARRIVES at the previous liquidity node (i.e. when edge i>0
  really happens), how many bars have elapsed / remain, and does this time state
  systematically explain the hardened-G0 prediction residual on NEXT / LOSS / CENSOR?

G0 predicts P(next | geometry, edge_index, ROOT static state) and treats node state as
Markov. This audit tests whether "elapsed/remaining bars at arrival" is an omitted state
variable. It is DIAGNOSTIC ONLY: the real future arrival state must NEVER be wired into
the ROOT-time G2 selector (that would be look-ahead leakage).

Scope: read-only on frozen caches. No model / G2 / selector / execution change. No H_i
feature model, no adverse field, no MFE/MAE, no momentum, no batch fix, no G1d, no
WF2/WF3, no raw 5m rescan, no cache rebuild.
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
    FULL_UNIV,
    OUT,
)

BASELINE_COMMIT = "32c561b72a29ace4afa61d872b0f65af0ddfefcf"

BINS = [-1, 4, 9, 14, 19, 24, 29, 34]
BIN_LABELS = ["00_04", "05_09", "10_14", "15_19", "20_24", "25_29", "30_34"]
OUTCOMES = ("next", "loss", "censor")
RESID_COL = {"next": "resid_next", "loss": "resid_loss", "censor": "resid_censor"}


def build_arrival_state(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Construct arrival-time state for each transition edge.

    edge_index k semantics: edge 0 is ROOT->L1, edge k>0 is L_k -> L_{k+1}. So the
    arrival time of edge k>0 MUST come from edge k-1's resolution_bar_index (never from
    the current edge's own resolution).
    """
    df = df.sort_values(["symbol", "signal_id", "edge_index"],
                        kind="stable").reset_index(drop=True)
    g = df.groupby(["symbol", "signal_id"], sort=False)
    df["prev_edge_index"] = g["edge_index"].shift(1)
    df["prev_resolution_bar"] = g["resolution_bar_index"].shift(1)
    df["prev_state"] = g["state_code"].shift(1)

    nz = df["edge_index"] > 0
    assert bool((df.loc[nz, "prev_edge_index"] == df.loc[nz, "edge_index"] - 1).all()), \
        "AUDIT_B_PREV_EDGE_INDEX_INVARIANT_FAIL"
    assert bool((df.loc[nz, "prev_state"] == NEXT).all()), \
        "AUDIT_B_PREV_STATE_NOT_NEXT_FAIL"

    df = df.merge(meta, on=["symbol", "signal_id"], how="left", validate="many_to_one")
    assert df["sig_entry_bar"].notna().all() and df["sig_W"].notna().all(), \
        "AUDIT_B_META_JOIN_MISSING"

    df["is_root"] = df["edge_index"] == 0
    elapsed = np.where(df["is_root"].to_numpy(),
                       0, df["prev_resolution_bar"] - df["sig_entry_bar"] + 1)
    df["elapsed_bars"] = elapsed.astype(np.int64)
    df["remaining_bars"] = (df["sig_W"] - df["elapsed_bars"]).astype(np.int64)

    assert (df["elapsed_bars"] >= 0).all(), "AUDIT_B_ELAPSED_NEGATIVE"
    assert (df["remaining_bars"] >= 0).all(), "AUDIT_B_REMAINING_NEGATIVE"
    assert (df["elapsed_bars"] <= df["sig_W"]).all(), "AUDIT_B_ELAPSED_GT_W"
    assert (df["remaining_bars"] <= df["sig_W"]).all(), "AUDIT_B_REMAINING_GT_W"
    assert ((df["elapsed_bars"] + df["remaining_bars"]) == df["sig_W"]).all(), \
        "AUDIT_B_ELAPSED_REMAINING_SUM_FAIL"

    df["same_bar_from_prev"] = (
        (df["edge_index"] > 0) & (df["resolution_bar_index"] == df["prev_resolution_bar"]))
    df["sig_key"] = df["symbol"].astype(str) + "|" + df["signal_id"].astype(str)
    return df


def synthetic_arrival_state_test():
    """Hand-check: entry_bar=100, W=34; edge res 101,104,104 ->
    elapsed [0,2,5], remaining [34,32,29], same_bar [F,F,T]."""
    tr = pd.DataFrame([
        dict(symbol="S", signal_id=1, edge_index=0, resolution_bar_index=101, state_code=NEXT),
        dict(symbol="S", signal_id=1, edge_index=1, resolution_bar_index=104, state_code=NEXT),
        dict(symbol="S", signal_id=1, edge_index=2, resolution_bar_index=104, state_code=LOSS),
    ])
    meta = pd.DataFrame([dict(symbol="S", signal_id=1, sig_entry_bar=100, sig_W=34)])
    out = build_arrival_state(tr, meta)
    ok = (list(out["elapsed_bars"]) == [0, 2, 5]
          and list(out["remaining_bars"]) == [34, 32, 29]
          and list(out["same_bar_from_prev"]) == [False, False, True])
    if not ok:
        raise SystemExit("STOP_AUDIT_B_SYNTHETIC_ARRIVAL_STATE_FAIL")
    print("[SYNTH] arrival-state OK: elapsed=[0,2,5] remaining=[34,32,29] "
          "same_bar=[F,F,T]")
    return True


def spearman(x, y):
    from scipy.stats import spearmanr
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), int(len(x))
    rho, _ = spearmanr(x, y)
    return float(rho), int(len(x))


def bin_table(df, sample_name):
    d = df.copy()
    d["remaining_bin"] = pd.cut(d["remaining_bars"], bins=BINS, labels=BIN_LABELS)
    rows = []
    for lab in BIN_LABELS:
        g = d[d["remaining_bin"] == lab]
        if len(g) == 0:
            rows.append(dict(sample=sample_name, remaining_bin=lab, n_edges=0,
                             n_signals=0))
            continue
        rows.append(dict(
            sample=sample_name,
            remaining_bin=lab,
            n_edges=int(len(g)),
            n_signals=int(g["sig_key"].nunique()),
            mean_edge_index=float(g["edge_index"].mean()),
            mean_elapsed_bars=float(g["elapsed_bars"].mean()),
            mean_remaining_bars=float(g["remaining_bars"].mean()),
            actual_NEXT=float((g["state_code"] == NEXT).mean()),
            pred_NEXT=float(g["pred_next"].mean()),
            resid_NEXT=float(g["resid_next"].mean()),
            actual_LOSS=float((g["state_code"] == LOSS).mean()),
            pred_LOSS=float(g["pred_loss"].mean()),
            resid_LOSS=float(g["resid_loss"].mean()),
            actual_CENSOR=float((g["state_code"] == CENSOR).mean()),
            pred_CENSOR=float(g["pred_censor"].mean()),
            resid_CENSOR=float(g["resid_censor"].mean()),
        ))
    return rows


def day_paired_bootstrap(df, n_boot=500, seed=20260913):
    """LOW (0-9) vs HIGH (25-34) paired trading-day block bootstrap on daily-mean
    residuals. Bootstrap only the precomputed day-difference ndarray."""
    d = df.copy()
    d["remain_group"] = np.where(d["remaining_bars"] <= 9, "LOW",
                          np.where(d["remaining_bars"] >= 25, "HIGH", "MID"))
    daily = (d[d["remain_group"].isin(["LOW", "HIGH"])]
             .groupby(["signal_trading_day", "remain_group"])
             [["resid_next", "resid_loss", "resid_censor"]].mean().reset_index())
    rng = np.random.default_rng(seed)
    out = {}
    for o in OUTCOMES:
        piv = daily.pivot_table(index="signal_trading_day", columns="remain_group",
                                values=RESID_COL[o])
        if "LOW" not in piv.columns or "HIGH" not in piv.columns:
            out[o] = dict(n_days=0)
            continue
        piv = piv.dropna(subset=["LOW", "HIGH"])
        diff = (piv["LOW"] - piv["HIGH"]).to_numpy(dtype=float)  # LOW - HIGH
        n_days = len(diff)
        obs = float(diff.mean()) if n_days else np.nan
        if n_days:
            idx = rng.integers(0, n_days, size=(n_boot, n_days))
            boot = diff[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            p_gt0 = float(np.mean(boot > 0))
            bmean = float(boot.mean())
        else:
            lo = hi = bmean = p_gt0 = np.nan
        out[o] = dict(observed_day_equal_mean=obs, bootstrap_mean=bmean,
                      ci_lo=float(lo), ci_hi=float(hi), p_diff_gt_0=p_gt0,
                      n_days=int(n_days))
    return out


def within_edge_sanity(primary, edges=(1, 2, 3, 4)):
    out = {}
    for e in edges:
        g = primary[primary["edge_index"] == e]
        row = dict(n=int(len(g)))
        for o in OUTCOMES:
            rho, n = spearman(g["remaining_bars"], g[RESID_COL[o]])
            row[f"rho_{o}"] = rho
        out[int(e)] = row
    return out


def decide_verdict(primary_rho, secondary_rho, boot, within):
    prim_strength = max(abs(primary_rho[o]) for o in OUTCOMES)
    sec_strength = max(abs(secondary_rho[o]) for o in OUTCOMES)
    ci_sig = any((boot[o].get("ci_lo", 0) > 0) or (boot[o].get("ci_hi", 0) < 0)
                 for o in OUTCOMES)
    kk = max(OUTCOMES, key=lambda o: abs(primary_rho[o]))
    sign = np.sign(primary_rho[kk])
    cons = sum(1 for r in within.values()
               if r["n"] >= 100 and np.sign(r[f"rho_{kk}"]) == sign
               and abs(r[f"rho_{kk}"]) > 0.0)
    consist = cons >= 2
    detail = dict(primary_strength=prim_strength, secondary_strength=sec_strength,
                  ci_significant=bool(ci_sig), strongest_outcome=kk,
                  strongest_sign=int(sign), within_edge_consistent_count=int(cons),
                  within_edge_consistent=bool(consist))
    if prim_strength < 0.02 and not ci_sig:
        if sec_strength >= 0.05 and sec_strength >= 2 * max(prim_strength, 1e-9):
            verdict = "LIKELY_BATCH_EFFECT_NOT_TIME_STATE"
        else:
            verdict = "TIME_STATE_NOT_SUPPORTED"
    elif ci_sig and consist and prim_strength >= 0.03:
        verdict = "TIME_STATE_SUPPORTED"
    else:
        verdict = "TIME_STATE_AMBIGUOUS"
    return verdict, detail


def main():
    t_all = time.perf_counter()

    synthetic_arrival_state_test()
    from scipy.stats import spearmanr  # noqa: F401  (import check)

    # ---- guards: hardened samples ----
    t0 = time.perf_counter()
    tr = load_transitions(FULL_UNIV, blocks=["TB1"], scope_tag="TB12_HARDENED")
    te = load_transitions(FULL_UNIV, blocks=["TB2"], scope_tag="TB12_HARDENED")
    assert len(tr) and len(te), "AUDIT_B_TRANSITION_CACHE_MISSING"
    test_start = pd.Timestamp(te["signal_trading_day"].min())
    tr_p, purge_diag = purge_train(tr, test_start)
    assert purge_diag["n_train_signals_before_purge"] == 21165, "AUDIT_B_TB1_BEFORE"
    assert purge_diag["n_train_signals_after_purge"] == 21135, "AUDIT_B_TB1_AFTER"
    assert purge_diag["n_purged_signals"] == 30, "AUDIT_B_TB1_PURGED"
    assert te["signal_id"].nunique() == 24954, "AUDIT_B_TB2_SIGNALS"
    assert len(te) == 55945, "AUDIT_B_TB2_EDGES"

    features, outcomes, provenance = load_full_node_tb2()
    assert provenance is not None, "STOP_AUDIT_B_CACHE_MISSING"
    meta = (features.loc[features["block"] == "TB2"]
            .sort_values(["symbol", "signal_id", "edge_index"], kind="stable")
            .drop_duplicates(["symbol", "signal_id"])
            [["symbol", "signal_id", "entry_bar", "W"]]
            .rename(columns={"entry_bar": "sig_entry_bar", "W": "sig_W"}))
    assert not meta.duplicated(["symbol", "signal_id"]).any(), "AUDIT_B_META_KEY_DUP"
    load_seconds = time.perf_counter() - t0

    # ---- node key uniqueness ----
    assert not te.duplicated(["symbol", "signal_id", "edge_index"]).any(), \
        "AUDIT_B_TRANSITION_KEY_DUP"

    # ---- exact G0 fit + one TB2 predict ----
    t1 = time.perf_counter()
    g0 = fit_multinomial(tr_p, G0_NUM, G0_CAT)
    df = build_arrival_state(te, meta)
    P = predict_reordered(g0, df, G0_NUM, G0_CAT)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6), "AUDIT_B_PROBA_SUM_FAIL"
    df["pred_next"] = P[:, 0]
    df["pred_loss"] = P[:, 1]
    df["pred_censor"] = P[:, 2]
    df["resid_next"] = (df["state_code"] == NEXT).astype(float) - df["pred_next"]
    df["resid_loss"] = (df["state_code"] == LOSS).astype(float) - df["pred_loss"]
    df["resid_censor"] = (df["state_code"] == CENSOR).astype(float) - df["pred_censor"]
    fit_predict_seconds = time.perf_counter() - t1

    # ---- samples ----
    t2 = time.perf_counter()
    nonroot = df[df["edge_index"] > 0].copy()
    primary = nonroot[~nonroot["same_bar_from_prev"]].copy()
    secondary = nonroot.copy()
    root = df[df["edge_index"] == 0].copy()

    sample_counts = dict(n_nonroot=int(len(nonroot)), n_primary=int(len(primary)),
                         n_samebar=int(nonroot["same_bar_from_prev"].sum()),
                         samebar_rate=float(nonroot["same_bar_from_prev"].mean()),
                         n_root=int(len(root)))

    # ---- bins ----
    rows = []
    rows += bin_table(primary, "PRIMARY")
    rows += bin_table(secondary, "SECONDARY")
    rows += bin_table(root, "ROOT")
    bins = pd.DataFrame(rows)
    bins.to_csv(OUT / "g2_time_state_bins.csv", index=False)

    # ---- Spearman (PRIMARY) ----
    prim_rho, sec_rho, elapsed_rho = {}, {}, {}
    for o in OUTCOMES:
        rho, n = spearman(primary["remaining_bars"], primary[RESID_COL[o]])
        prim_rho[o] = rho
        sec_rho[o] = spearman(secondary["remaining_bars"], secondary[RESID_COL[o]])[0]
        elapsed_rho[o] = spearman(primary["elapsed_bars"], primary[RESID_COL[o]])[0]
    prim_n = int(len(primary))

    # ---- bootstrap ----
    boot = day_paired_bootstrap(primary)
    brow = [dict(outcome=o, **boot[o]) for o in OUTCOMES]
    pd.DataFrame(brow).to_csv(OUT / "g2_time_state_bootstrap.csv", index=False)

    # ---- within-edge sanity (edges 1-4, PRIMARY) ----
    within = within_edge_sanity(primary)

    # ---- verdict ----
    verdict, vdetail = decide_verdict(prim_rho, sec_rho, boot, within)

    summary = {
        "audit": "G2_MISSING_DURATION_STATE_V1",
        "baseline_commit": BASELINE_COMMIT,
        "purge": purge_diag,
        "samples": sample_counts,
        "bins": BIN_LABELS,
        "primary_remaining_spearman": {o: dict(rho=prim_rho[o], n=prim_n) for o in OUTCOMES},
        "primary_elapsed_spearman": {o: dict(rho=elapsed_rho[o], n=prim_n) for o in OUTCOMES},
        "secondary_remaining_spearman": {o: dict(rho=sec_rho[o], n=int(len(secondary))) for o in OUTCOMES},
        "bootstrap_low_minus_high": boot,
        "within_edge_primary": within,
        "verdict": verdict,
        "verdict_detail": vdetail,
        "timing_seconds": dict(load_seconds=load_seconds,
                               fit_predict_seconds=fit_predict_seconds,
                               audit_seconds=time.perf_counter() - t2,
                               total_seconds=time.perf_counter() - t_all),
        "interpretation_contract": (
            "Diagnostic only. The real future arrival time state must never enter the "
            "ROOT-time G2 selector (look-ahead leakage). A positive residual means the "
            "model under-predicts that outcome; negative means over-predicts."
        ),
    }
    with open(OUT / "g2_time_state_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
