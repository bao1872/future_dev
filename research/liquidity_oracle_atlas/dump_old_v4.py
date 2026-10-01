"""dump_old_v4
==============
Snapshots the CURRENT (pre-V4.1) God-mode per-event decisions so the V4.1
audit can report before/after tracking of the 82 missing-target events and
the 76 cross-day trades. Read-only w.r.t. the code under test.
"""

import json
import os

import numpy as np

from research.liquidity_oracle_atlas.structural_god_oracle_m15_v4 import (
    run_god_oracle_v4,
)
from research.liquidity_oracle_atlas.build_dp_proximity_m15_v1 import (
    build_dp_proximity_m15,
)


def main():
    res = run_god_oracle_v4("AG")
    records = res["records"]
    mv = res["market_view"]
    prox = build_dp_proximity_m15("AG")
    td = prox["trading_day"].to_numpy(np.int64)
    seg = prox["segment"].to_numpy(np.int64)
    unit_starts = np.asarray(mv.unit_starts, dtype=np.int64)

    def unit_of(bar):
        return int(np.searchsorted(unit_starts, int(bar), side="right")) - 1

    events = []
    missing_remain_ids = []
    cross_day_ids = []
    for r in records:
        eid = int(r["event_id"])
        dec = r["oracle_decision"]
        d_idx = int(r["best_entry_decision_index"])
        f_idx = int(r["best_entry_fill_index"])
        rem = r["remaining_target_atr"]
        rem_nan = (rem is None) or (not np.isfinite(rem))
        if r["canonical_oracle_trade"]:
            if rem_nan:
                missing_remain_ids.append(eid)
            if d_idx >= 0 and f_idx >= 0:
                if (int(td[d_idx]) != int(td[f_idx])
                        or int(seg[d_idx]) != int(seg[f_idx])
                        or unit_of(d_idx) != unit_of(f_idx)):
                    cross_day_ids.append(eid)
        events.append({
            "event_id": eid,
            "oracle_decision": dec,
            "oracle_direction": r["oracle_direction"],
            "long_branch_valid": bool(r["long_branch_valid"]),
            "short_branch_valid": bool(r["short_branch_valid"]),
            "best_entry_decision_index": d_idx,
            "best_entry_fill_index": f_idx,
            "remaining_target_atr": (None if rem_nan else float(rem)),
        })

    out = {
        "events": events,
        "missing_remaining_ids": missing_remain_ids,
        "cross_day_ids": cross_day_ids,
    }
    dst = os.path.join(os.path.dirname(__file__), "god_oracle_v4_old_dump.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"events={len(events)} "
          f"missing_remaining={len(missing_remain_ids)} "
          f"cross_day={len(cross_day_ids)}")
    print(f"dumped -> {dst}")


if __name__ == "__main__":
    main()
