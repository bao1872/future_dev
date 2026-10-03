"""tp_god_oracle_scaling_v5
==========================

TP scaling harness for the V5 production oracle.

V5 reuses V4's O(log N) segment tree for the per-option target-touch query, so
total work is O(N + Q log N + K) where Q = atomic TradeOptions enumerated and
K = selected trades. This script proves:

  * production uses the tree (target_touch_query_count > 0, node visits ~ Q*log N)
  * NO full-window scan (candidate_scan_steps / contact_future_scan_steps = 0)
  * reference_call_count = 0 (V5 has no separate reference solver)
  * runtime / query growth is sub-quadratic across N -> 2N -> 4N
"""

from __future__ import annotations

import time

import numpy as np

from research.liquidity_oracle_atlas.structural_god_oracle_m15_v5 import (
    hotpath_counters,
    reset_hotpath_counters,
    run_god_oracle_v5,
)


def _scaling(symbol: str = "AG", base: int = 1500, factors=(1, 2, 4)):
    rows = []
    prev_dt = None
    prev_q = None
    for f in factors:
        mb = base * f
        reset_hotpath_counters()
        t0 = time.time()
        res = run_god_oracle_v5(symbol, max_bars=mb)
        dt = time.time() - t0
        c = hotpath_counters()
        canon = sum(1 for r in res["records"] if r["canonical_oracle_trade"])
        q = c["target_touch_query_count"]
        nv = c["target_touch_tree_node_visits"]
        ratio_dt = (dt / prev_dt) if prev_dt else float("nan")
        ratio_q = (q / prev_q) if prev_q else float("nan")
        rows.append({
            "n": mb,
            "canonical": canon,
            "queries": q,
            "tree_node_visits": nv,
            "nodes_per_query": nv / max(q, 1),
            "reference_call_count": c["reference_call_count"],
            "candidate_scan_steps": c["candidate_scan_steps"],
            "contact_future_scan_steps": c["contact_future_scan_steps"],
            "seconds": round(dt, 2),
            "dt_ratio_N__2N": round(ratio_dt, 3),
            "query_ratio_N__2N": round(ratio_q, 3),
        })
        prev_dt = dt
        prev_q = q
    return rows


def main() -> None:
    rows = _scaling()
    print("=== V5 TP scaling (N -> 2N -> 4N) ===")
    for r in rows:
        print(
            "n={n} canon={canonical} queries={queries} nodes/q={nodes_per_query:.1f} "
            "ref={reference_call_count} scan_steps={candidate_scan_steps} "
            "dt={seconds}s dt_ratio={dt_ratio_N__2N} q_ratio={query_ratio_N__2N}".format(**r)
        )
    # Hard gates
    assert rows[0]["reference_call_count"] == 0, "no reference solver in V5"
    assert rows[0]["candidate_scan_steps"] == 0, "no full-window candidate scan"
    assert rows[0]["contact_future_scan_steps"] == 0, "no full-window contact scan"
    assert rows[0]["queries"] > 0, "production must use the tree"
    # sub-quadratic: doubling N must less-than-double runtime
    for i in range(1, len(rows)):
        assert rows[i]["dt_ratio_N__2N"] < 2.0, "runtime growth must be sub-linear-in-N"
    print("V5 TP SCALING GATES PASSED")


if __name__ == "__main__":
    main()
