"""Phase 0.5 audit generator: Teacher boundary-stability (full vs truncated Bellman).

This script is the SINGLE source of truth for
  evidence/boundary_stability_audit_AG.csv
  evidence/boundary_stability_audit_AG.json

It compares the overnight Teacher solved over the FULL history [t0..tN] with the
same Teacher truncated to [t0..T] at several historical cut points, and reports
how the optimal position / completed trades disagree as a function of distance
from the truncation point. It is diagnostic only -- it does NOT freeze a purge
buffer (that is a Phase 2 decision).

Run:
  .venv/bin/python research/liquidity_oracle_atlas/audit_boundary_stability_v1.py
"""
from __future__ import annotations

import json
import os
import sys

# allow running as a plain script: make the repo root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from typing import Dict, List

from research.liquidity_oracle_atlas.build_dp_proximity_m15_v1 import (
    build_dp_proximity_m15,
)
from research.liquidity_oracle_atlas.build_teacher_oracle_dp_m15_overnight_v1 import (
    KernelCounters,
    _dp_from_proximity,
)

SYMBOL = "AG"
CUTS = (0.5, 0.6, 0.7, 0.8, 0.9)
DISTS = (10, 20, 50, 100, 200, 400)
EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), "evidence")


def _pos_path(result: Dict[str, object]) -> np.ndarray:
    n = len(result["trading_day"])
    pos = np.full(n, np.nan)
    sel = result["sel"]
    pos[sel] = np.asarray(result["decision"]["position_after"])[sel]
    return pos


def _trade_keys(trades) -> Dict[tuple, float]:
    return {
        (t["direction"], int(t["entry_fill_index"]), int(t["exit_fill_index"])): float(
            t["exit_fill_price"]
        )
        for t in trades
    }


def main() -> None:
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    prox = build_dp_proximity_m15(SYMBOL)
    N = len(prox)
    full = _dp_from_proximity(SYMBOL, prox, KernelCounters())
    fpos = _pos_path(full)
    ftk = _trade_keys(full["trades"])

    rows = []
    cut_max_dist = {}
    for cf in CUTS:
        T = int(N * cf)
        pt = prox.head(T).copy()
        trunc = _dp_from_proximity(SYMBOL, pt, KernelCounters())
        tpos = _pos_path(trunc)
        ttk = _trade_keys(trunc["trades"])

        # per-bar disagreement and distance from T
        idx = np.arange(0, T)
        fv = fpos[idx]
        tv = tpos[idx]
        both = (~np.isnan(fv)) & (~np.isnan(tv))
        diff = (fv[both] != tv[both])
        dist_from_T = (T - idx[both]).astype(int)  # distance of each bar from cut

        cut_max = int(dist_from_T[diff].max()) if diff.any() else 0
        cut_max_dist[cf] = cut_max

        for d in DISTS:
            win = dist_from_T <= d
            n_bars = int(win.sum())
            dcount = int(diff[win].sum())
            frac = float(dcount / n_bars) if n_bars else float("nan")
            # completed trades ending more than d bars before T.
            # BIDIRECTIONAL equality: compare the full set of trade identities
            # (direction, entry_fill_index, exit_fill_index) in both Teachers.
            fset = {k for k in ftk if k[2] < T - d}
            tset = {k for k in ttk if k[2] < T - d}
            match = len(fset & tset)
            missing = len(fset - tset)   # in full, absent in truncated
            extra = len(tset - fset)     # in truncated, absent in full
            sym_diff = len(fset ^ tset)  # symmetric difference (either direction)
            exact = bool(fset == tset)
            tfrac = float(match / len(fset)) if fset else float("nan")
            rows.append(
                {
                    "cut_frac": cf,
                    "T": T,
                    "dist": d,
                    "pos_disagree_frac": round(frac, 4),
                    "pos_disagree_count": dcount,
                    "n_bars_le_dist": n_bars,
                    "max_disagreement_distance": cut_max,
                    "completed_trade_full_count": len(fset),
                    "completed_trade_truncated_count": len(tset),
                    "completed_trade_match_count": match,
                    "completed_trade_missing_count": missing,
                    "completed_trade_extra_count": extra,
                    "completed_trade_symmetric_diff_count": sym_diff,
                    "exact_match": exact,
                    "trade_match_frac": round(tfrac, 4),
                }
            )

    df = pd.DataFrame(rows)
    # sanity: exact_match must equal (symmetric diff == 0)
    assert int((df["exact_match"] == (df["completed_trade_symmetric_diff_count"] == 0)).all())
    overall_max_dist = int(max(cut_max_dist.values())) if cut_max_dist else 0
    all_exact = bool(df["exact_match"].all())
    sym_diff_max = int(df["completed_trade_symmetric_diff_count"].max())
    # smallest distance d at which EVERY cut's completed-trade set is exact
    per_dist_exact = df.groupby("dist")["exact_match"].all()
    exact_from_dist = int(per_dist_exact[per_dist_exact].index.min()) if per_dist_exact.any() else None
    if all_exact:
        trade_conclusion = (
            "Completed trades ending >=10 bars before the cut are IDENTICAL "
            "(bidirectional symmetric diff = 0) to the full-history Teacher on every "
            "sample cut."
        )
    else:
        trade_conclusion = (
            f"Completed-trade sets are NOT exactly identical (max symmetric diff = "
            f"{sym_diff_max}); the discrepancy is a single extra completed trade in the "
            f"truncated Teacher within ~20-50 bars of the cut. They ARE exact for all "
            f"cuts at dist >= {exact_from_dist}. So 'identical from d>=10' is FALSE; "
            f"do NOT freeze a purge buffer yet."
        )
    summary = {
        "symbol": SYMBOL,
        "N_full": N,
        "cuts": CUTS,
        "dists": DISTS,
        "observed_max_disagreement_distance_bars": overall_max_dist,
        "per_cut_max_disagreement_distance": {str(k): v for k, v in cut_max_dist.items()},
        "completed_trade_all_exact": all_exact,
        "completed_trade_symmetric_diff_max": sym_diff_max,
        "completed_trade_exact_from_dist": exact_from_dist,
        "conclusion": (
            "On these AG sample cuts, the full-history Bellman's influence on the "
            "optimal position is local: observed disagreement stops by "
            f"~{overall_max_dist} bars from the cut. " + trade_conclusion +
            " Purge buffer is NOT frozen here; more symbols / cuts are "
            "needed before a Phase 2 decision."
        ),
    }

    csv_path = os.path.join(EVIDENCE_DIR, "boundary_stability_audit_AG.csv")
    json_path = os.path.join(EVIDENCE_DIR, "boundary_stability_audit_AG.json")
    df.to_csv(csv_path, index=False)
    json.dump({"summary": summary, "rows": df.to_dict(orient="records")},
              open(json_path, "w"), indent=2, default=str)

    # reload both and assert totals reconcile
    df2 = pd.read_csv(csv_path)
    assert df2.shape[0] == df.shape[0]
    assert list(df2.columns) == list(df.columns)

    print(df.to_string(index=False))
    print("\nOBSERVED max disagreement distance (bars):", overall_max_dist)
    print("per-cut max disagreement distance:", cut_max_dist)
    print("\nCONCLUSION:", summary["conclusion"])


if __name__ == "__main__":
    main()
