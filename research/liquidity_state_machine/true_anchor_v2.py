"""P0.4：在 VALID_AHEAD + CONTINUOUS_CROSS 下重算 TRUE anchor。"""
from __future__ import annotations

import pandas as pd

from research.liquidity_state_machine.liquidity_specificity_placebo_v2 import (
    RESULTS, SymbolData,
)
from research.liquidity_state_machine.match_liquidity_placebo_v2 import (
    resolve_stage2, stage2_pool,
)


def main():
    tr = pd.read_parquet(RESULTS / "true_interactions.parquet")
    va = tr[tr["activation_state"] == "VALID_AHEAD"]
    fresh = va[va["interaction_path"] == "CONTINUOUS_CROSS"]
    pen = fresh[fresh["penetrated"]]

    sbr = int((pen["stage1"] == "SAME_BAR_RECLAIM").sum())
    cb = int((pen["stage1"] == "CLOSE_BEYOND").sum())
    cal = int((pen["stage1"] == "CLOSE_AT_LEVEL").sum())
    print(f"FRESH PENETRATION = {len(pen)}")
    print(f"  SAME_BAR_RECLAIM {sbr} ({sbr/len(pen)*100:.2f}%)")
    print(f"  CLOSE_BEYOND     {cb} ({cb/len(pen)*100:.2f}%)")
    print(f"  CLOSE_AT_LEVEL   {cal} ({cal/len(pen)*100:.2f}%)")

    T = stage2_pool(tr)
    print(f"\nStage-2 TRUE pool = {len(T)}")
    out = []
    for sym, g in T.groupby("symbol"):
        sd = SymbolData(sym)
        for r in g.itertuples(index=False):
            s, t = resolve_stage2(sd, int(r.interaction_i), int(r.side),
                                  float(r.price))
            out.append(dict(level_key=r.level_key, stage2=s,
                            stage2_time=t))
    o = pd.DataFrame(out).merge(
        T[[c for c in ["level_key", "symbol", "liquidity_type",
                       "liquidity_scope", "trading_day", "side",
                       "sweep_vs_1h", "env_align_1h"] if c in T.columns]],
        on="level_key", how="left")
    o.to_parquet(RESULTS / "true_stage2_outcomes.parquet", index=False)

    n = len(o)
    lr = int((o["stage2"] == "LATER_RECLAIM").sum())
    sa = int((o["stage2"] == "STRUCTURAL_ACCEPTANCE").sum())
    print(f"\n=== TRUE_LIQUIDITY_ANCHOR (P0.4) ===")
    print(f"P(LATER_RECLAIM | CLOSE_BEYOND)         = {lr/n:.4f} ({lr}/{n})")
    print(f"P(STRUCTURAL_ACCEPTANCE | CLOSE_BEYOND) = {sa/n:.4f} ({sa}/{n})")
    print("\nby symbol:")
    b = o.groupby("symbol").apply(
        lambda g: pd.Series(dict(
            n=len(g),
            p_later_reclaim=round(
                float((g["stage2"] == "LATER_RECLAIM").mean()), 4))))
    print(b.to_string())
    print(f"  macro median = {b['p_later_reclaim'].median():.4f}  "
          f"positive = {int((b['p_later_reclaim'] > 0.5).sum())}/{len(b)}")
    print("\nby sweep_vs_1h:")
    print(o.groupby("sweep_vs_1h").apply(
        lambda g: round(float((g["stage2"] == "LATER_RECLAIM").mean()), 4)
    ).to_string())
    print("\nby env_align_1h:")
    print(o.groupby("env_align_1h").apply(
        lambda g: round(float((g["stage2"] == "LATER_RECLAIM").mean()), 4)
    ).to_string())
    print("\nTRUE_ANCHOR_DONE")


if __name__ == "__main__":
    main()
