"""FUTURE-R13.6 -- Output Integration Atlas (plan §20).

Descriptive only.  No threshold or region is selected from this atlas.

Uses A0 ROOT TD5.  Dimensions:
    p_win    : 5 quantiles
    p_BE     : 5 quantiles   (p_BE = L / (W + L))
    scale    : W + L          : 3 quantiles

Each cell reports n, weight_sum, actual win rate, mean return, median
return, avg win, avg loss.  A 2D p_win x p_BE table is also produced.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.meta_output_integration_v1 import (
    ARCH_PRIMARY,
    EVIDENCE_DIR,
    root_frame,
    wmean,
)

ATLAS_CSV = os.path.join(EVIDENCE_DIR, "output_integration_atlas_v1.csv")


def _cell_stats(g: pd.DataFrame) -> dict:
    w = g["sample_weight"].to_numpy(float)
    ret = g["episode_return_atr"].to_numpy(float)
    win = g["win"].to_numpy(float)
    win_mask = ret > 0
    loss_mask = ret < 0
    return {
        "n": int(len(g)),
        "weight_sum": round(float(np.sum(w)), 6),
        "win_rate": round(float(wmean(win, w)), 6),
        "mean_return": round(float(wmean(ret, w)), 6),
        "median_return": round(float(np.median(ret)), 6) if len(ret) else np.nan,
        "avg_win": round(float(wmean(ret[win_mask], w[win_mask])), 6)
        if win_mask.any() else np.nan,
        "avg_loss": round(float(wmean(ret[loss_mask], w[loss_mask])), 6)
        if loss_mask.any() else np.nan,
    }


def build_atlas(arch: str = ARCH_PRIMARY) -> pd.DataFrame:
    frame = root_frame(arch).copy()
    p = np.clip(frame["p_win"].to_numpy(float), 0.0, 1.0)
    mw = np.maximum(frame["mu_win"].to_numpy(float), 0.0)
    ml = np.maximum(frame["mu_loss"].to_numpy(float), 0.0)
    denom = mw + ml
    p_be = np.where(denom > 0, ml / denom, np.nan)
    scale = mw + ml
    frame["p_win"] = p
    frame["p_break_even"] = p_be
    frame["scale"] = scale

    frame["p_win_q"] = pd.qcut(frame["p_win"], 5, labels=False,
                               duplicates="drop")
    frame["p_be_q"] = pd.qcut(frame["p_break_even"], 5, labels=False,
                              duplicates="drop")
    frame["scale_q"] = pd.qcut(frame["scale"], 3, labels=False,
                               duplicates="drop")

    rows = []
    for (pw, pb, sc), g in frame.groupby(["p_win_q", "p_be_q", "scale_q"]):
        stat = _cell_stats(g)
        stat.update({"p_win_q": int(pw), "p_be_q": int(pb), "scale_q": int(sc),
                     "dimension": "3D"})
        rows.append(stat)

    for (pw, pb), g in frame.groupby(["p_win_q", "p_be_q"]):
        stat = _cell_stats(g)
        stat.update({"p_win_q": int(pw), "p_be_q": int(pb), "scale_q": -1,
                     "dimension": "2D_pwin_x_pbe"})
        rows.append(stat)

    return pd.DataFrame(rows)


def main() -> None:
    df = build_atlas(ARCH_PRIMARY)
    import research.liquidity_oracle_atlas.run_decomposed_v2_research as R
    R.write_csv_evidence(df, ATLAS_CSV)
    print(f"[R13.6] atlas written: {ATLAS_CSV}  rows={len(df)}")


if __name__ == "__main__":
    main()
