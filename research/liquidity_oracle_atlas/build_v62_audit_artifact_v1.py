"""
build_v62_audit_artifact_v1.py
==============================

Generate a FROZEN, offline audit artifact for the experimental V6.2
entry-time-target god oracle. The Streamlit V6.2 Audit Viewer reads ONLY
this artifact -- it never re-runs Oracle label math.

Outputs (under artifacts/god_oracle_m15_v62_audit/):
    bars.parquet   : full exec_frame OHLC (for the chart)
    trades.parquet : one row per canonical V6.2 label, flattened, plus
                     entry-time geometry, future geometry, source-handoff
                     annotations and overlap flags
    meta.json      : symbol, counts, and quick-jump special review cases

This is EXPERIMENTAL only. It does not touch the production V6.1 artifact,
the production Viewer, builder semantics or Target math.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.structural_god_oracle_m15_v6_2_entry_target import (
    geometry_zones,
    run_entry_time_target_oracle,
)
from research.liquidity_oracle_atlas.structural_god_oracle_m15_v6_touch_chain import (
    EPS,
    intervals_overlap,
    run_touch_chain_oracle,
)

ART_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "god_oracle_m15_v62_audit"


def _zone_key(z: dict) -> tuple:
    return (
        z.get("tf"), z.get("family"), z.get("side"),
        round(float(z["bottom"]), 6), round(float(z["top"]), 6),
        (round(float(z["level"]), 6) if z.get("level") is not None else None),
    )


def compute_terminal(frames, src_bot, src_top, start):
    for b in range(int(start) + 1, len(frames)):
        if any(not intervals_overlap(src_bot, src_top, x.bottom, x.top, EPS)
               for x in frames[b]):
            return int(b)
    return len(frames)


def build(symbol: str = "AG") -> dict:
    res = run_entry_time_target_oracle(symbol)
    trades = res["trades"]
    frames = res["frames"]
    ef = res["exec_frame"]
    geoms = res["geom_by_decision"]

    # V6.1 trades only to anchor the "old Trade 7 / 8 market segment" jumps.
    v61 = run_touch_chain_oracle(symbol)
    v61_trades = v61["trades"]

    n = len(frames)

    # ---- full bars ----
    bars = ef[["execution_bar_index", "bar_start_time", "open", "high",
               "low", "close"]].copy()
    bars = bars.rename(columns={"execution_bar_index": "bar_index"})

    rows = []
    for i, t in enumerate(trades):
        src_bar = int(t["source_bar"])
        src_bot = float(t["zone_bottom"])
        src_top = float(t["zone_top"])
        d = int(t["best_entry_decision_index"])
        exit_idx = int(t["exit_fill_index"])

        # entry-time visible geometry
        entry_zones = geometry_zones(geoms[d] if 0 <= d < len(geoms) else None)
        entry_keys = {_zone_key(z) for z in entry_zones}

        # future geometry: structures first visible after the entry decision
        future_zones = []
        seen_future = set()
        for d2 in range(d + 1, exit_idx + 1):
            if not (0 <= d2 < len(geoms)):
                continue
            for z in geometry_zones(geoms[d2]):
                k = _zone_key(z)
                if k in entry_keys or k in seen_future:
                    continue
                seen_future.add(k)
                future_zones.append(z)

        # observed_leg_terminal of source A
        terminal = compute_terminal(frames, src_bot, src_top, src_bar)
        terminal_had = False
        terminal_locations = []
        if terminal < n:
            tf = frames[terminal]
            distinct = [x for x in tf
                        if not intervals_overlap(src_bot, src_top, x.bottom, x.top, EPS)]
            same = [x for x in tf
                    if intervals_overlap(src_bot, src_top, x.bottom, x.top, EPS)]
            terminal_had = (len(distinct) == 1 and len(same) >= 1)
            for x in tf:
                role = "distinct" if not intervals_overlap(
                    src_bot, src_top, x.bottom, x.top, EPS) else "retouch"
                terminal_locations.append({
                    "bottom": round(float(x.bottom), 6),
                    "top": round(float(x.top), 6),
                    "role": role,
                })

        # next source (the following trade's source)
        next_src_bar = int(trades[i + 1]["source_bar"]) if i + 1 < len(trades) else None
        next_src_bot = (float(trades[i + 1]["zone_bottom"])
                        if i + 1 < len(trades) else None)
        next_src_top = (float(trades[i + 1]["zone_top"])
                        if i + 1 < len(trades) else None)

        rows.append({
            "trade_id": i,
            "symbol": symbol,
            "source_bar": src_bar,
            "source_structure_id": t.get("source_structure_id"),
            "zone_bottom": src_bot,
            "zone_top": src_top,
            "oracle_direction": t["oracle_direction"],
            "best_entry_decision_index": d,
            "best_entry_fill_index": int(t["best_entry_fill_index"]),
            "best_entry_price": float(t["best_entry_price"]),
            "target_snapshot_index": int(t["target_snapshot_index"]),
            "target_price": float(t["target_price"]),
            "target_location_bottom": float(t["target_location_bottom"]),
            "target_location_top": float(t["target_location_top"]),
            "target_structure_id": t.get("target_structure_id"),
            "exit_fill_index": exit_idx,
            "exit_price": float(t["exit_price"]),
            "exit_reason": t["exit_reason"],
            "utility": float(t["utility"]),
            "observed_leg_terminal": terminal,
            "terminal_had_retouch_plus_distinct": terminal_had,
            "terminal_locations": json.dumps(terminal_locations),
            "entry_geometry": json.dumps(entry_zones),
            "future_geometry": json.dumps(future_zones),
            "next_source_bar": next_src_bar,
            "next_source_bottom": next_src_bot,
            "next_source_top": next_src_top,
        })

    tdf = pd.DataFrame(rows)

    # ---- exit overlap flag (calendar-time overlap of [entry, exit] intervals) ----
    entries = tdf["best_entry_fill_index"].to_numpy(np.int64)
    exits = tdf["exit_fill_index"].to_numpy(np.int64)
    overlaps = np.zeros(len(tdf), dtype=bool)
    for a in range(len(tdf)):
        a_lo, a_hi = entries[a], exits[a]
        for b in range(len(tdf)):
            if b == a:
                continue
            if a_lo <= exits[b] and entries[b] <= a_hi:
                overlaps[a] = True
                break
    tdf["exit_overlaps_another"] = overlaps

    # ---- special review cases (quick-jump) ----
    def ids_in_window(anchor_bar, lo=-40, hi=200):
        if anchor_bar is None:
            return []
        lo_b, hi_b = anchor_bar + lo, anchor_bar + hi
        return [int(r.trade_id) for r in tdf.itertuples()
                if lo_b <= int(r.source_bar) <= hi_b]

    old7 = v61_trades[6]["candidate_start_bar"] if len(v61_trades) > 6 else None
    old8 = v61_trades[7]["candidate_start_bar"] if len(v61_trades) > 7 else None
    repaired = [int(r.trade_id) for r in tdf.itertuples()
                if r.terminal_had_retouch_plus_distinct][:8]
    exit_ov = [int(r.trade_id) for r in tdf.itertuples()
               if r.exit_overlaps_another][:8]

    special_cases = {
        "old_v61_trade7_segment": ids_in_window(old7),
        "old_v61_trade8_segment": ids_in_window(old8),
        "repaired_a_retouch_b_handoff": repaired,
        "exit_overlaps_another": exit_ov,
    }

    meta = {
        "symbol": symbol,
        "math_version": "structural-god-oracle-v6.2-entry-time-target",
        "n_trades": int(len(tdf)),
        "n_bars": int(len(bars)),
        "v61_source_establishments": len(v61["trades"]),
        "v62_source_establishments": int(tdf["source_bar"].nunique()),
        "special_cases": special_cases,
        "notes": (
            "Frozen audit artifact for experimental V6.2. Target causal invariant: "
            "target_snapshot_index == best_entry_decision_index for every trade. "
            "Entry-time geometry is from geom_by_decision[best_entry_decision_index]."
        ),
    }

    ART_DIR.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(ART_DIR / "bars.parquet", index=False)
    tdf.to_parquet(ART_DIR / "trades.parquet", index=False)
    (ART_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    m = build("AG")
    print("artifact written to", ART_DIR)
    print(json.dumps(m, indent=2))
