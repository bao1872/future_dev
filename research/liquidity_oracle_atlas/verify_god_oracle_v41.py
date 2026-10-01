"""verify_god_oracle_v41
=======================

V4.1 Semantic Fix verification + before/after accounting.

* Full AG run (production + independent reference) -- asserts parity.
* Hard invariants: cross_day/seg/unit fill counts == 0; canonical
  remaining_nan / tp_nan / loss / self_target == 0.
* Before/after tracking of the 82 old missing-remaining events and the
  76 old cross-day trades (loaded from god_oracle_v4_old_dump.json).
* V4 accounting table + gap ATR stats + Target->NextStatic audit.

Read-only report script. Invoke:
    ./.venv/bin/python -m research.liquidity_oracle_atlas.verify_god_oracle_v41
"""

import json
import os

import numpy as np

from research.liquidity_oracle_atlas.structural_god_oracle_m15_v4 import (
    run_god_oracle_v4,
    audit_target_vs_next_static,
    summarize_god_oracle,
    ORACLE_LONG,
    ORACLE_SHORT,
    ORACLE_NOPOS,
    ORACLE_TIE,
    ORACLE_NO_TARGET,
)
from research.liquidity_oracle_atlas.structural_god_oracle_reference_m15_v4 import (
    run_god_oracle_v4_reference,
)

# Known OLD (V4) accounting, captured before the V4.1 fix.
OLD = {
    "oracle_trades": 1207,
    "LONG": 599,
    "SHORT": 608,
    "no_positive": 243,
    "direction_tie": 13,
    "missing_remaining": 82,
    "cross_day_fills": 76,
    "canonical_losses": 0,
    "max_entry_gap_atr": 22.876,
}


def _load_old():
    dst = os.path.join(os.path.dirname(__file__), "god_oracle_v4_old_dump.json")
    with open(dst) as fh:
        return json.load(fh)


def _parity(prod, ref):
    rp = prod["records"]
    rr = ref["records"]
    assert len(rp) == len(rr)
    mismatch = 0
    max_err = 0.0
    for a, b in zip(rp, rr):
        for k in ("oracle_decision", "best_entry_fill_index", "exit_fill_index",
                  "target_price"):
            av, bv = a.get(k), b.get(k)
            if (av is None) != (bv is None):
                mismatch += 1
                break
            if av is None:
                continue
            if isinstance(av, float):
                max_err = max(max_err, abs(av - bv))
                if abs(av - bv) > 1e-9:
                    mismatch += 1
                    break
            elif av != bv:
                mismatch += 1
                break
    return mismatch, max_err


def _gap_stats(canon):
    vals = np.asarray([r["best_entry_gap_atr"] for r in canon
                       if np.isfinite(r["best_entry_gap_atr"])], dtype=float)
    return {
        "median": float(np.median(vals)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
        "max": float(np.max(vals)),
    }


def main():
    prod = run_god_oracle_v4("AG")
    ref = run_god_oracle_v4_reference("AG")
    records = prod["records"]
    meta = prod["meta"]
    summ = summarize_god_oracle(records)
    audit = audit_target_vs_next_static(records)

    mismatch, max_err = _parity(prod, ref)

    # hard invariants
    rem_nan = summ["remaining_target_atr_nan_count"]
    tp_nan = summ["canonical_tp_nan_count"]
    assert meta["cross_day_fill_count"] == 0
    assert meta["cross_segment_fill_count"] == 0
    assert meta["cross_unit_fill_count"] == 0
    assert rem_nan == 0
    assert tp_nan == 0
    assert meta["canonical_loss_count"] == 0
    assert meta["self_target_count"] == 0
    assert mismatch == 0, f"production/reference parity mismatch={mismatch}"

    # before/after tracking
    old = _load_old()
    old_by_id = {e["event_id"]: e for e in old["events"]}
    new_by_id = {r["event_id"]: r for r in records}

    missing_ids = set(old["missing_remaining_ids"])
    cross_ids = set(old["cross_day_ids"])

    track_missing = {"LONG": 0, "SHORT": 0, "NO_POSITIVE": 0,
                     "NO_VALID_TARGET_BRANCH": 0, "TIE": 0}
    track_cross = {"reselected_different_entry": 0, "changed_direction": 0,
                   "became_no_trade": 0}

    for eid in missing_ids:
        nr = new_by_id.get(eid)
        if nr is None:
            continue
        dec = nr["oracle_decision"]
        if dec == ORACLE_LONG:
            track_missing["LONG"] += 1
        elif dec == ORACLE_SHORT:
            track_missing["SHORT"] += 1
        elif dec == ORACLE_NOPOS:
            track_missing["NO_POSITIVE"] += 1
        elif dec == ORACLE_NO_TARGET:
            track_missing["NO_VALID_TARGET_BRANCH"] += 1
        elif dec == ORACLE_TIE:
            track_missing["TIE"] += 1

    for eid in cross_ids:
        oe = old_by_id.get(eid)
        nr = new_by_id.get(eid)
        if oe is None or nr is None:
            continue
        if nr["oracle_decision"] not in (ORACLE_LONG, ORACLE_SHORT):
            track_cross["became_no_trade"] += 1
        else:
            if nr["best_entry_decision_index"] != oe["best_entry_decision_index"]:
                track_cross["reselected_different_entry"] += 1
            if nr["oracle_direction"] != oe["oracle_direction"]:
                track_cross["changed_direction"] += 1

    gap = _gap_stats([r for r in records if r["canonical_oracle_trade"]])

    # ------------------------------------------------------------------ report
    print("=== V4.1 ACCOUNTING (before -> after) ===")
    rows = [
        ("Oracle trades", OLD["oracle_trades"], meta["canonical_oracle_trades"]),
        ("LONG", OLD["LONG"], meta["oracle_long"]),
        ("SHORT", OLD["SHORT"], meta["oracle_short"]),
        ("no positive", OLD["no_positive"], meta["no_positive_opportunity"]),
        ("no valid target branch", "?", meta["no_valid_target_branch"]),
        ("direction tie", OLD["direction_tie"], meta["direction_ties"]),
        ("missing remaining", OLD["missing_remaining"], rem_nan),
        ("cross-day fills", OLD["cross_day_fills"], meta["cross_day_fill_count"]),
        ("cross-segment fills", "?", meta["cross_segment_fill_count"]),
        ("cross-unit fills", "?", meta["cross_unit_fill_count"]),
        ("canonical losses", OLD["canonical_losses"], meta["canonical_loss_count"]),
        ("self target", 0, meta["self_target_count"]),
        ("valid but no entry", 0, meta["valid_but_no_entry"]),
        ("max entry gap ATR", OLD["max_entry_gap_atr"], gap["max"]),
    ]
    for name, b, a in rows:
        print(f"  {name:28s} {str(b):>10s} -> {str(a):>10s}")

    print("\n=== 82 OLD MISSING-TARGET EVENTS (new labels) ===")
    for k, v in track_missing.items():
        print(f"  became {k:24s} {v}")
    print(f"  (total tracked = {sum(track_missing.values())} / {len(missing_ids)})")

    print("\n=== 76 OLD CROSS-DAY TRADES ===")
    for k, v in track_cross.items():
        print(f"  {k:28s} {v}")
    print(f"  (total tracked = {sum(1 for _ in cross_ids)} / {len(cross_ids)})")

    print("\n=== EXECUTION BOUNDARIES ===")
    print(f"  cross_day  = {meta['cross_day_fill_count']}")
    print(f"  cross_seg  = {meta['cross_segment_fill_count']}")
    print(f"  cross_unit = {meta['cross_unit_fill_count']}")

    print("\n=== CANONICAL INVARIANTS ===")
    print(f"  loss            = {meta['canonical_loss_count']}")
    print(f"  remaining_nan   = {rem_nan}")
    print(f"  tp_nan          = {tp_nan}")
    print(f"  self_target     = {meta['self_target_count']}")

    print("\n=== GAP ATR ===")
    for k in ("median", "p90", "p95", "p99", "max"):
        print(f"  {k:8s} = {gap[k]:.4f}")

    print("\n=== REFERENCE / PRODUCTION ===")
    print(f"  shared   = {len(records)}")
    print(f"  mismatch = {mismatch}")
    print(f"  max_err  = {max_err:.2e}")

    print("\n=== T0 / NC / REGRESSION ===")
    print("  (run pytest separately; this script reuses the same modules)")

    print("\n=== TARGET -> NEXT STATIC ===")
    print(f"  touch     = {audit['target_touch_count']}")
    print(f"  match     = {audit['target_equals_next_static_count']}")
    print(f"  mismatch  = {audit['mismatch_count']}")
    print(f"  rate      = {audit['match_rate']:.4f}")

    print("\nALL V4.1 HARD INVARIANTS PASSED; production/reference parity = 0")


if __name__ == "__main__":
    main()
