"""Step 1-2：canonical candidate audit + native_direction audit。"""
from __future__ import annotations

import pandas as pd

from research.phase1_native_direction.nd_contract_v1 import (
    RESULTS, load_candidates,
)


def main():
    c = load_candidates()

    rows = []
    for (sym, tf, d), g in c.groupby(
            ["symbol", "source_tf", "native_direction"]):
        rows.append(dict(
            symbol=sym, source_tf=tf, native_direction=int(d),
            candidate_count=len(g),
            first_time=str(g["touch_time"].min()),
            last_time=str(g["touch_time"].max()),
        ))
    aud = pd.DataFrame(rows).sort_values(
        ["symbol", "source_tf", "native_direction"])
    aud.to_csv(RESULTS / "candidate_audit.csv", index=False,
               encoding="utf-8-sig")

    print("=== candidate audit ===")
    print(aud.to_string(index=False))

    print("\n=== 汇总 ===")
    print(f"total candidates      : {len(c)}")
    print(f"LONG  (native=+1)     : {int((c.native_direction == 1).sum())}")
    print(f"SHORT (native=-1)     : {int((c.native_direction == -1).sum())}")
    print(f"LONG/SHORT ratio      : "
          f"{(c.native_direction == 1).sum() / (c.native_direction == -1).sum():.4f}")
    print(f"symbols               : {c.symbol.nunique()}")
    print(f"source_tf             : {sorted(c.source_tf.unique().tolist())}")

    print("\n=== per symbol ===")
    ps = c.pivot_table(index="symbol", columns="native_direction",
                       values="candidate_id", aggfunc="count", fill_value=0)
    ps.columns = [f"dir_{x}" for x in ps.columns]
    ps["total"] = ps.sum(axis=1)
    print(ps.to_string())

    print("\n=== per source_tf ===")
    pt = c.pivot_table(index="source_tf", columns="native_direction",
                       values="candidate_id", aggfunc="count", fill_value=0)
    pt.columns = [f"dir_{x}" for x in pt.columns]
    pt["total"] = pt.sum(axis=1)
    print(pt.to_string())

    print("\n=== symbol x source_tf x direction ===")
    p3 = (c.groupby(["symbol", "source_tf", "native_direction"])
            .size().rename("n").reset_index())
    print(p3.pivot_table(index=["symbol"], columns=["source_tf",
                                                    "native_direction"],
                         values="n", fill_value=0).to_string())

    print("\nCANDIDATE_AUDIT_DONE")


if __name__ == "__main__":
    main()
