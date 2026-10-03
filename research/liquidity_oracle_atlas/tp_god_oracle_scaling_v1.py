"""tp_god_oracle_scaling_v1
============================

REMOTE-AUDIT-FIX-01 item F -- TP performance gate harness (evidence is meant to
be RE-RUN remotely, so the numbers reported in the review are reproducible).

It decomposes a full Oracle run into

    one-time SETUP   (proximity build + production environment + V2 events)
    sequential LOOP  (the candidate stream: scan -> best entry -> target touch)

because the total wall-clock is dominated by SETUP, which hides the hot path,
and reports the hot-path COUNTERS alongside N / 2N / 4N scaling.

IMPORTANT: ``max_bars`` bounds the RAW 5m rows; proximity is 15m, so a target of
N decision bars is requested as ``max_bars = 3 * N``. (Verified: max_bars=14800
yields 4934 proximity rows, not 14800.)

Usage
-----
    PYTHONPATH=<repo> python research/liquidity_oracle_atlas/tp_god_oracle_scaling_v1.py

Optional comparison against an extracted baseline module (e.g. the previously
reviewed 5d6f6a1 code copied next to this file):

    PYTHONPATH=<repo>:<dir-with-baseline_v4.py> \
        python .../tp_god_oracle_scaling_v1.py --baseline baseline_v4

This script NEVER writes artifacts.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

# N / 2N / 4N in 15m decision bars. 4N ~= the full AG history (14817 rows).
DEFAULT_NS = (3704, 7408, 14816)
SYMBOL = "AG"


def _setup_cost(max_bars: int):
    from research.liquidity_oracle_atlas.build_dp_proximity_m15_v1 import (
        build_dp_proximity_m15,
    )
    from research.liquidity_oracle_atlas.build_execution_environment_m15_v1 import (
        run_environment_m15,
    )
    from research.liquidity_oracle_atlas.build_structural_dp_labels_m15_v2 import (
        build_structural_events_v2,
    )

    t0 = time.perf_counter()
    build_dp_proximity_m15(SYMBOL, max_bars)
    t1 = time.perf_counter()
    run_environment_m15(SYMBOL, max_bars, capture_provenance=False)
    t2 = time.perf_counter()
    build_structural_events_v2(SYMBOL, max_bars)
    t3 = time.perf_counter()
    return t1 - t0, t2 - t1, t3 - t2, t3 - t0


def _run_cost(module, max_bars: int):
    reset = getattr(module, "reset_hotpath_counters", None)
    if reset is not None:
        reset()
    t0 = time.perf_counter()
    res = module.run_god_oracle_v4(SYMBOL, max_bars=max_bars)
    elapsed = time.perf_counter() - t0
    canon = len([r for r in res["records"] if r.get("canonical_oracle_trade")])
    counters = getattr(module, "hotpath_counters", lambda: {})()
    return elapsed, canon, counters


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default=None,
                    help="module name of an extracted baseline implementation")
    ap.add_argument("--ns", type=int, nargs="*", default=list(DEFAULT_NS))
    args = ap.parse_args(argv)

    import research.liquidity_oracle_atlas.structural_god_oracle_m15_v4 as prod
    modules = [("NEW", prod)]
    if args.baseline:
        import importlib
        modules.append((args.baseline.upper(), importlib.import_module(args.baseline)))

    print("=== one-time SETUP (prox, environment, v2_events, TOTAL_setup) ===")
    setup = {}
    for n in args.ns:
        prox, env, ev, total = _setup_cost(3 * n)
        setup[n] = total
        print(f"N={n:>6}  prox={prox:7.3f}s env={env:7.3f}s events={ev:7.3f}s "
              f"SETUP={total:7.3f}s")

    print("\n=== full run + hot-path counters ===")
    print(f"{'mod':>6} {'N':>6} {'rows':>6} {'TOTAL':>8} {'SETUP':>8} {'LOOP':>8} "
          f"{'trades':>7} {'tt_q':>8} {'tt_nodes':>10} {'nodes/q':>8} {'ref':>5} {'ratio':>6}")
    for name, mod in modules:
        prev_t: Optional[float] = None
        prev_nodes: Optional[int] = None
        for n in args.ns:
            total, canon, c = _run_cost(mod, 3 * n)
            loop = total - setup[n]
            ratio = (total / prev_t) if prev_t else float("nan")
            q = c.get("target_touch_query_count", -1)
            nodes = c.get("target_touch_tree_node_visits", -1)
            ref = c.get("reference_call_count", -1)
            npq = (nodes / q) if q else float("nan")
            nodes_ratio = (nodes / prev_nodes) if prev_nodes else float("nan")
            print(f"{name:>6} {n:>6} {n:>6} {total:8.3f} {setup[n]:8.3f} {loop:8.3f} "
                  f"{canon:>7} {q:>8} {nodes:>10} {npq:>8.2f} {ref:>5} {ratio:6.2f}")
            print(f"            tt_node_scaling N->2N = {nodes_ratio:.2f} "
                  f"(sub-quadratic => <=~2.5; quadratic => ~4)")
            prev_t = total
            prev_nodes = nodes
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
