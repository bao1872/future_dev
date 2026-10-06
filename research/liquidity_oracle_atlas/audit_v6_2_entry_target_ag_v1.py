"""
audit_v6_2_entry_target_ag_v1.py
================================

REAL AG AUDIT for the experimental V6.2 entry-time-target kernel.

Run:

    python research/liquidity_oracle_atlas/audit_v6_2_entry_target_ag_v1.py [SYMBOL]

Reports:
  1. V6.2 canonical trades
  2. LONG / SHORT count
  3. no-visible-target count
  4. target-never-touched count
  5. direction-tie count
  6. entry-boundary rejection count
  7. labels_with_overlapping_exit (count, NOT a constraint)
  8. target_snapshot_mismatch count

Mandatory invariant (causal correctness):
    target_snapshot_mismatch == 0

REMOVED constraint (2026-10-06):
    labels were previously forced to be non-overlapping in calendar time
    (portfolio single-position constraint). That belongs to later backtesting,
    NOT to oracle label generation. Overlap is now REQUIRED behavior and is
    only REPORTED, never enforced.

Four required numbers (this round):
    source opportunities                = sources_evaluated
    canonical labels                    = canonical_trades
    labels with overlapping exit        = labels_with_overlapping_exit
    old V6.1 source locations matched   = intersection of V6.1 / V6.2 source keys

This audit does NOT generate or overwrite the production artifact.
"""
from __future__ import annotations

import sys

import numpy as np

from research.liquidity_oracle_atlas.build_execution_environment_m15_v1 import (
    run_environment_m15,
)
from research.liquidity_oracle_atlas.structural_god_oracle_m15_v6_2_entry_target import (
    LONG,
    SHORT,
    locations_from_geometry_snapshot,
    solve_entry_time_target_oracle,
)
from research.liquidity_oracle_atlas.structural_god_oracle_m15_v6_touch_chain import (
    build_location_frames,
    solve_location_touch_chain,
)

EPS = 1e-6


def main(symbol: str = "AG") -> int:
    print("=" * 72)
    print(f"V6.2 ENTRY-TIME TARGET -- REAL AG AUDIT  symbol={symbol}")
    print("=" * 72)

    env = run_environment_m15(symbol, None, capture_provenance=True)
    ef = env["exec_frame"]
    geoms = env["geom_by_decision"]

    frames = build_location_frames(env["entry_matches"])
    opens = ef["open"].to_numpy(float)
    highs = ef["high"].to_numpy(float)
    lows = ef["low"].to_numpy(float)
    segments = ef["segment"].to_numpy(np.int64)
    tdays = ef["trading_day"].to_numpy(np.int64)
    times = ef["bar_start_time"].to_numpy()
    n = len(ef)

    print(f"bars                : {n}")
    print(f"bars_with_true_touch: {sum(1 for f in frames if f)}")

    # ---------------- V6.1 (old, for diagnostic only) ----------------
    old_trades, old_audit = solve_location_touch_chain(
        frames, opens, segments, tdays, times,
    )
    print("\n--- OLD V6.1 (diagnostic baseline) ---")
    print(f"  canonical_trades: {old_audit['canonical_trades']}")

    # ---------------- V6.2 ----------------
    trades, audit = solve_entry_time_target_oracle(
        frames=frames,
        opens=opens,
        highs=highs,
        lows=lows,
        segments=segments,
        trading_days=tdays,
        geoms=geoms,
        times=times,
    )

    print("\n--- NEW V6.2 audit table ---")
    rows = [
        ("1 canonical_trades", audit["canonical_trades"]),
        ("2 long_count", audit["long_count"]),
        ("2 short_count", audit["short_count"]),
        ("3 no_visible_target_count", audit["no_visible_target_count"]),
        ("4 target_never_touched_count", audit["target_never_touched_count"]),
        ("5 direction_tie_count", audit["direction_tie_count"]),
        ("6 entry_boundary_rejection_count",
         audit["entry_boundary_rejection_count"]),
        ("7 labels_with_overlapping_exit", audit["labels_with_overlapping_exit"]),
        ("8 target_snapshot_mismatch_count",
         audit["target_snapshot_mismatch_count"]),
        ("  ambiguous_candidate_bars", audit["ambiguous_candidate_bars"]),
        ("  sources_evaluated", audit["sources_evaluated"]),
        ("  sources_without_canonical", audit["sources_without_canonical"]),
    ]
    for k, v in rows:
        print(f"  {k:36s}: {v}")

    print("\n--- causal invariant (must hold) ---")
    ok_mm = audit["target_snapshot_mismatch_count"] == 0
    print(f"  target_snapshot_mismatch == 0 : {'PASS' if ok_mm else 'FAIL'}"
          f"  ({audit['target_snapshot_mismatch_count']})")

    print("\n--- overlap report (NOT a constraint) ---")
    print(f"  labels_with_overlapping_exit : {audit['labels_with_overlapping_exit']}")
    print("  (overlap is REQUIRED behavior; portfolio non-overlap belongs to "
          "later backtesting)")

    # ---------------- old-vs-entry-time diagnostic ----------------
    print("\n" + "=" * 72)
    print("OLD V6.1 TARGETS vs ENTRY-TIME GEOMETRY (diagnostic only)")
    print("=" * 72)

    loc_by_bar = [locations_from_geometry_snapshot(g) for g in geoms]

    first_bottom: dict = {}
    first_top: dict = {}
    for k, locs in enumerate(loc_by_bar):
        for L in locs:
            b = round(float(L.bottom), 6)
            t = round(float(L.top), 6)
            first_bottom.setdefault(b, k)
            first_top.setdefault(t, k)

    def visible_at(d, entry_price, direction, target_price):
        """Strict: a visible location whose FIRST-TOUCH EDGE equals the target."""
        if d is None or d < 0 or d >= len(loc_by_bar):
            return False
        for L in loc_by_bar[d]:
            if direction == LONG:
                if (abs(float(L.bottom) - float(target_price)) <= 1e-6
                        and float(L.bottom) > float(entry_price) + EPS):
                    return True
            else:
                if (abs(float(L.top) - float(target_price)) <= 1e-6
                        and float(L.top) < float(entry_price) - EPS):
                    return True
        return False

    def covered_at(d, entry_price, direction, target_price):
        """Loose: ANY visible location covering that price level, right side."""
        if d is None or d < 0 or d >= len(loc_by_bar):
            return False
        tp = float(target_price)
        for L in loc_by_bar[d]:
            if not (float(L.bottom) - 1e-6 <= tp <= float(L.top) + 1e-6):
                continue
            if direction == LONG and float(L.bottom) > float(entry_price) + EPS:
                return True
            if direction == SHORT and float(L.top) < float(entry_price) - EPS:
                return True
        return False

    violating = []
    for seq, t in enumerate(old_trades, start=1):
        d = int(t["best_entry_decision_index"])
        entry_price = float(t["best_entry_price"])
        direction = t["oracle_direction"]
        tp = float(t["target_price"])
        if not visible_at(d, entry_price, direction, tp):
            if direction == LONG:
                fb = first_bottom.get(round(tp, 6))
            else:
                fb = first_top.get(round(tp, 6))
            violating.append({
                "seq": seq,
                "source_bar": int(t["candidate_start_bar"]),
                "direction": direction,
                "entry_decision": d,
                "entry_price": entry_price,
                "target_id": t["target_structure_id"],
                "target_price": tp,
                "visible_at_entry": False,
                "first_visible_bar": fb,
            })

    not_covered = [
        t for t in old_trades
        if not covered_at(
            int(t["best_entry_decision_index"]),
            float(t["best_entry_price"]),
            t["oracle_direction"],
            float(t["target_price"]),
        )
    ]

    print(f"old canonical trades                 : {len(old_trades)}")
    print(f"old_target_not_visible_at_entry_count: {len(violating)}"
          f"   (strict: exact first-touch edge)")
    print(f"old_target_not_covered_at_entry_count: {len(not_covered)}"
          f"   (loose: any visible location covering the level)")
    print(f"share (strict)                       : "
          f"{len(violating) / max(1, len(old_trades)):.4f}")

    print("\n--- first 20 violating old examples ---")
    hdr = (f"{'seq':>4} | {'src':>5} | {'dir':>5} | {'dec':>5} | "
           f"{'entry':>9} | {'target':>9} | {'vis':>3} | {'1st_vis':>7} | target_id")
    print(hdr)
    for v in violating[:20]:
        print(
            f"{v['seq']:>4} | {v['source_bar']:>5} | {v['direction']:>5} | "
            f"{v['entry_decision']:>5} | {v['entry_price']:>9.1f} | "
            f"{v['target_price']:>9.1f} | no  | "
            f"{str(v['first_visible_bar']):>7} | {v['target_id']}"
        )
    if not violating:
        print("  (none)")

    # ---------------- four required numbers ----------------
    print("\n" + "=" * 72)
    print("FOUR REQUIRED NUMBERS (V6.2 after removing portfolio non-overlap)")
    print("=" * 72)

    old_keys = set()
    for t in old_trades:
        old_keys.add((int(t["candidate_start_bar"]),
                      round(float(t["zone_bottom"]), 6),
                      round(float(t["zone_top"]), 6)))
    new_keys = set()
    for t in trades:
        new_keys.add((int(t["candidate_start_bar"]),
                      round(float(t["zone_bottom"]), 6),
                      round(float(t["zone_top"]), 6)))
    matched = len(old_keys & new_keys)

    print(f"  source opportunities              : {audit['sources_evaluated']}")
    print(f"  canonical labels                  : {audit['canonical_trades']}")
    print(f"  labels with overlapping exit      : {audit['labels_with_overlapping_exit']}")
    print(f"  old V6.1 source locations matched : {matched}")
    print(f"      (new unique source keys = {len(new_keys)}; "
          f"old V6.1 unique source keys = {len(old_keys)})")

    return 0 if ok_mm else 1


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "AG"
    raise SystemExit(main(sym))
