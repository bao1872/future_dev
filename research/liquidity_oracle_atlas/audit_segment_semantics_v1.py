"""Phase 0.5 audit generator: canonical `segment` semantics across 15 symbols.

This script is the SINGLE source of truth for
  evidence/segment_audit_15sym.csv
  evidence/segment_audit_15sym.json

It does NOT train or generate any STRUCT33 artifact. It only reads the canonical
15m execution frame + the canonical discontinuity definition and reports, per
symbol, how many hard segments exist and exactly what canonical event moves the
`segment` boundary (data gap vs price-jump/rollover vs both vs unknown).

Run:
  .venv/bin/python research/liquidity_oracle_atlas/audit_segment_semantics_v1.py
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
from research.phase1_tradability.phase1_contract_v1 import (
    ROLL_GAP_ATR_THRESHOLD,
    build_session_masks,
    compute_atr5,
    discontinuity_flags,
    get_bars,
)

# canonical 15-symbol universe (frozen list from build_robust_trade_oracle_dp_v1)
SYMBOLS = ["AG", "AU", "CU", "AL", "SN", "NI", "RB", "I", "SC", "RU", "MA", "TA", "M", "P", "CF"]
EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), "evidence")


def _classify_segments(symbol: str) -> Dict[str, object]:
    prox = build_dp_proximity_m15(symbol)
    seg = prox["segment"].to_numpy(np.int64)
    t15 = pd.to_datetime(prox["bar_start_time"]).to_numpy()
    n_bars = int(len(seg))
    n_segments = int(seg[-1]) + 1
    segment_changes = int((seg[1:] != seg[:-1]).sum())

    # canonical 5m discontinuity decomposition
    bars = get_bars(symbol)
    t5 = pd.to_datetime(bars["time"]).to_numpy()
    contig, normal = build_session_masks(t5)
    gap_time = np.zeros(len(t5), dtype=bool)
    gap_time[1:] = ~(contig | normal)
    atr = compute_atr5(bars)
    prev_c = np.empty(len(t5))
    prev_c[0] = bars["close"][0]
    prev_c[1:] = bars["close"][:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        gap_atr = np.abs(bars["open"] - prev_c) / atr
    gap_px = np.zeros(len(t5), dtype=bool)
    gap_px[1:] = np.nan_to_num(gap_atr[1:], nan=0.0) > ROLL_GAP_ATR_THRESHOLD
    disc = discontinuity_flags(symbol)

    c_gap_time = c_gap_px = c_both = c_unknown = 0
    for i in range(1, len(seg)):
        if seg[i] == seg[i - 1]:
            continue
        j = min(i + 1, len(seg) - 1)
        t_lo = t15[i]
        t_hi = t15[j] if j > i else t5[-1] + pd.Timedelta(minutes=15)
        mask = (t5 >= t_lo) & (t5 < t_hi)
        dmask = mask & disc
        gt = bool(gap_time[dmask].any())
        gp = bool(gap_px[dmask].any())
        if gt and gp:
            c_both += 1
        elif gt:
            c_gap_time += 1
        elif gp:
            c_gap_px += 1
        else:
            c_unknown += 1

    return {
        "symbol": symbol,
        "n_bars_15m": n_bars,
        "n_segments": n_segments,
        "segment_changes": segment_changes,
        "cause_gap_time": c_gap_time,
        "cause_gap_px": c_gap_px,
        "cause_both": c_both,
        "cause_unknown": c_unknown,
    }


def main() -> None:
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    rows = [_classify_segments(s) for s in SYMBOLS]
    df = pd.DataFrame(rows)

    # mechanical consistency: every segment change must be explained by exactly one
    # canonical cause bucket (gap_time / gap_px / both / unknown).
    explained = (
        df["cause_gap_time"] + df["cause_gap_px"] + df["cause_both"] + df["cause_unknown"]
    )
    assert int((explained == df["segment_changes"]).all()), (
        "segment-change cause buckets do not sum to segment_changes:\n" + str(df)
    )

    overall = {
        "gap_time": int(df["cause_gap_time"].sum()),
        "gap_px": int(df["cause_gap_px"].sum()),
        "both": int(df["cause_both"].sum()),
        "unknown": int(df["cause_unknown"].sum()),
        "segment_changes": int(df["segment_changes"].sum()),
    }
    # overall totals must also reconcile
    assert overall["gap_time"] + overall["gap_px"] + overall["both"] + overall[
        "unknown"
    ] == overall["segment_changes"], overall

    csv_path = os.path.join(EVIDENCE_DIR, "segment_audit_15sym.csv")
    json_path = os.path.join(EVIDENCE_DIR, "segment_audit_15sym.json")
    # write BOTH artifacts from this single run so they cannot diverge
    df.to_csv(csv_path, index=False)
    json.dump(
        {
            "overall": overall,
            "threshold": ROLL_GAP_ATR_THRESHOLD,
            "symbols": SYMBOLS,
            "rows": df.to_dict(orient="records"),
        },
        open(json_path, "w"),
        indent=2,
    )

    # cross-check: reload both and assert totals identical
    df2 = pd.read_csv(csv_path)
    j2 = json.load(open(json_path))
    assert int(df2["segment_changes"].sum()) == j2["overall"]["segment_changes"]
    assert int(df2["cause_gap_time"].sum()) == j2["overall"]["gap_time"]
    assert int(df2["cause_gap_px"].sum()) == j2["overall"]["gap_px"]
    assert int(df2["cause_unknown"].sum()) == j2["overall"]["unknown"]

    print(df.to_string(index=False))
    print("\nOVERALL", json.dumps(overall))
    print("ROLL_GAP_ATR_THRESHOLD =", ROLL_GAP_ATR_THRESHOLD)
    print("wrote (single run):", csv_path, json_path)
    print("CSV/JSON totals reconciled: OK")


if __name__ == "__main__":
    main()
