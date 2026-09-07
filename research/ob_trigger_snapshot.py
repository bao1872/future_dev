#!/usr/bin/env python3
"""5m OB Trigger Multi-TF SMC Snapshot V1.

Research sample = a canonical 5m SMC ``OB_ENTERED`` event, NOT a bar.

For every 5m OB_ENTERED event we freeze, at the trigger instant, the full
SMC environment that was *actually knowable then* (point-in-time, no
lookahead):

    5m  : the SMC state of bar i itself (the bar where OB_ENTERED fired)
    15m : the last 15m bar whose bar_end_time <= trigger_time
    1h  : the last 1h  bar whose bar_end_time <= trigger_time

trigger_time = 5m availability_time[i]  (i = OB enter_index)

We then attach future outcomes (entry at next 5m open, exit at close H
bars later, H in 5m/15m/30m/60m/120m) and write:

    research/exports/ob_trigger_smc_v1/
        trigger_snapshots.csv   # wide: one row per OB_ENTERED event
        trigger_levels.csv      # long: event x timeframe x nearby structure
        manifest.json

All SMC facts are CONSUMED from the canonical output of
``compute_smc_pine`` (via ``compute_smc_momentum_bundle``). Nothing here
redefines indicator semantics. Canonical SMC owns: ``state_timeline``
(swing/internal bias, active OB counts), ``events`` (BOS/CHoCH with
``confirmed_index``), ``order_blocks`` (confirmed/mitigated), ``pivots``
(internal/swing high/low), ``equal_highs_lows`` (EQH/EQL).

Key PIT invariants (STOP conditions, section 19 of the contract):
    1. higher-TF snapshot bar_end_time must be <= trigger_time
    2. no pivot with confirmed_index > snapshot index is ever written
    3. no BOS/CHoCH with confirmed_index > snapshot index is written
    4. no OB with confirmed_index > snapshot index enters the active set
    5. no OB with mitigated_index <= snapshot index is kept active
    6. trigger_snapshots rows == canonical 5m OB_ENTERED event count
"""

from __future__ import annotations

import bisect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.build_pytdx_panel import aggregate_15m  # noqa: E402
from research.indicator_adapter import (  # noqa: E402
    compute_smc_momentum_bundle,
)

SYMBOLS = ["AG", "CU", "RB", "M"]

SRC = ROOT / "research" / "exports" / "v3r_5m"
OUT = ROOT / "research" / "exports" / "ob_trigger_smc_v1"

# 5m / 15m / 30m / 60m / 120m expressed as a count of 5m bars.
HORIZONS_5M = [1, 3, 6, 12, 24]

ONE_HOUR_NS = 60 * 60 * 1_000_000_000

PIVOT_TYPES = {
    "internal_high",
    "internal_low",
    "swing_high",
    "swing_low",
}


# ============================================================
# Cross-timeframe aggregation (thin helpers; no new semantics)
# ============================================================

def aggregate_1h_from_15m(fifteen: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 15m bars into 1h bars by integer hour-epoch bucket.

    Strictly replicates the repository's existing cross-TF authority:
    lower-TF bars are grouped by higher-period epoch bucket and OHLC /
    volume / OI are aggregated. No "must be exactly 4 bars" rule added.
    """

    x = fifteen.sort_values("bar_start_time").copy()

    start_ns = (
        x["bar_start_time"]
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )

    x["_bucket"] = (start_ns // ONE_HOUR_NS) * ONE_HOUR_NS

    out = (
        x.groupby("_bucket", observed=True)
        .agg(
            bar_start_time=("bar_start_time", "min"),
            bar_end_time=("bar_end_time", "max"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            open_oi=("open_oi", "first"),
            close_oi=("close_oi", "last"),
        )
        .reset_index(drop=True)
        .sort_values("bar_start_time")
        .reset_index(drop=True)
    )

    return out


def build_smc_tf(bars: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compute canonical SMC for one timeframe. Consumer only."""

    ind = bars.copy()

    if "volume" not in ind.columns:
        raise ValueError("SMC frame requires volume")

    ind = ind.set_index(pd.DatetimeIndex(ind["bar_start_time"]))

    bundle = compute_smc_momentum_bundle(ind)

    return ind, bundle.smc


# ============================================================
# Point-in-time index helpers
# ============================================================

def asof_completed_index(
    bars: pd.DataFrame,
    trigger_time: pd.Timestamp,
) -> int | None:
    """Index of the last bar whose bar_end_time <= trigger_time."""

    ends = (
        pd.to_datetime(bars["bar_end_time"])
        .to_numpy(dtype="datetime64[ns]")
    )

    target = np.datetime64(trigger_time, "ns")

    idx = int(np.searchsorted(ends, target, side="right") - 1)

    if idx < 0:
        return None

    if not (
        pd.Timestamp(bars.iloc[idx]["bar_end_time"]) <= trigger_time
    ):
        raise AssertionError("higher-TF lookahead")

    return idx


# ============================================================
# Canonical SMC consumers (no redefinition)
# ============================================================

def state_at(smc: dict, i: int) -> dict:
    row = smc["state_timeline"][i]

    if int(row["bar_index"]) != i:
        raise AssertionError("state_timeline alignment")

    return {
        "swing_bias": int(row["swing_bias"]),
        "internal_bias": int(row["internal_bias"]),
        "active_internal_ob_count": int(row["active_internal_ob_count"]),
        "active_swing_ob_count": int(row["active_swing_ob_count"]),
    }


def last_structure_event(
    smc: dict,
    i: int,
    *,
    internal: bool,
) -> dict | None:
    candidates = [
        ev
        for ev in smc.get("events", [])
        if (
            bool(ev["internal"]) == internal
            and int(ev["confirmed_index"]) <= i
        )
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda ev: int(ev["confirmed_index"]),
    )


def active_order_blocks_at(smc: dict, i: int) -> list[dict]:
    result = []

    for ob in smc.get("order_blocks", []):
        confirmed = int(ob["confirmed_index"])

        if confirmed > i:
            continue

        mitigated = ob.get("mitigated_index")

        if mitigated is not None and int(mitigated) <= i:
            continue

        result.append(ob)

    return result


def confirmed_pivots_at(smc: dict, i: int) -> list[dict]:
    return [
        p
        for p in smc.get("pivots", [])
        if p.get("type") in PIVOT_TYPES
        and int(p["confirmed_index"]) <= i
    ]


def confirmed_equal_levels_at(smc: dict, i: int) -> list[dict]:
    return [
        e
        for e in smc.get("equal_highs_lows", [])
        if int(e["confirmed_index"]) <= i
    ]


# ============================================================
# Structure maps for the long table
# ============================================================

def relation_to_price(
    low: float,
    high: float,
    price: float,
) -> tuple[str, float]:
    low = float(low)
    high = float(high)
    price = float(price)

    if low <= price <= high:
        return ("overlap", 0.0)

    if low > price:
        return ("above", (low - price) / price)

    return ("below", (price - high) / price)


def select_nearby_levels(
    rows: list[dict],
    k: int = 5,
) -> list[dict]:
    overlap = [r for r in rows if r["relation"] == "overlap"]

    above = sorted(
        [r for r in rows if r["relation"] == "above"],
        key=lambda r: r["distance_pct"],
    )[:k]

    below = sorted(
        [r for r in rows if r["relation"] == "below"],
        key=lambda r: r["distance_pct"],
    )[:k]

    return overlap + above + below


# ============================================================
# Per-timeframe precomputation (sorted, bisect-friendly)
# ============================================================

def _precomp_tf(smc: dict) -> dict:
    obs_items = []
    for ob in smc.get("order_blocks", []):
        obs_items.append(
            {
                "confirmed": int(ob["confirmed_index"]),
                "mitigated": (
                    int(ob["mitigated_index"])
                    if ob.get("mitigated_index") is not None
                    else None
                ),
                "bias": int(ob["bias"]),
                "internal": bool(ob["internal"]),
                "anchor_index": int(ob["anchor_index"]),
                "bar_low": float(ob["bar_low"]),
                "bar_high": float(ob["bar_high"]),
                "confirmed_index": int(ob["confirmed_index"]),
            }
        )

    piv_items = []
    for p in smc.get("pivots", []):
        if p.get("type") in PIVOT_TYPES:
            piv_items.append(
                {
                    "type": p["type"],
                    "level": float(p["level"]),
                    "anchor_index": int(p["anchor_index"]),
                    "confirmed_index": int(p["confirmed_index"]),
                }
            )

    eq_items = []
    for e in smc.get("equal_highs_lows", []):
        eq_items.append(
            {
                "type": e["type"],
                "level": float(e["level"]),
                "anchor_index": int(e["anchor_index"]),
                "confirmed_index": int(e["confirmed_index"]),
            }
        )

    for lst in (obs_items, piv_items, eq_items):
        lst.sort(key=lambda d: d["confirmed_index"])

    return {
        "obs": obs_items,
        "obs_conf": [d["confirmed"] for d in obs_items],
        "piv": piv_items,
        "piv_conf": [d["confirmed_index"] for d in piv_items],
        "eq": eq_items,
        "eq_conf": [d["confirmed_index"] for d in eq_items],
    }


def _gather_tf_levels(
    smc: dict,
    precomp: dict,
    i: int,
    trigger_close: float,
    violations: list[str],
) -> tuple[dict, list[dict], dict | None, dict | None]:
    """Return (active_ob_counts, level_rows, nearest_above, nearest_below)."""

    obs_conf = precomp["obs_conf"]
    piv_conf = precomp["piv_conf"]
    eq_conf = precomp["eq_conf"]

    k_obs = bisect.bisect_right(obs_conf, i)
    active = [
        o
        for o in precomp["obs"][:k_obs]
        if o["mitigated"] is None or o["mitigated"] > i
    ]

    k_piv = bisect.bisect_right(piv_conf, i)
    piv_slice = precomp["piv"][:k_piv]

    k_eq = bisect.bisect_right(eq_conf, i)
    eq_slice = precomp["eq"][:k_eq]

    # ----- STOP discipline: defense (should be impossible by construction)
    for o in active:
        if o["confirmed_index"] > i:
            violations.append(f"OB confirmed {o['confirmed_index']} > {i}")
        if o["mitigated"] is not None and o["mitigated"] <= i:
            violations.append(f"OB mitigated {o['mitigated']} <= {i}")

    # ----- structural point levels
    struct_rows: list[dict] = []
    for p in piv_slice:
        rel, dist = relation_to_price(p["level"], p["level"], trigger_close)
        struct_rows.append(
            {
                "object_type": p["type"],
                "structure_level": p["level"],
                "bias": 0,
                "source_index": p["anchor_index"],
                "confirmed_index": p["confirmed_index"],
                "age_bars": i - p["confirmed_index"],
                "zone_low": p["level"],
                "zone_high": p["level"],
                "relation": rel,
                "distance_pct": dist,
            }
        )
        if p["confirmed_index"] > i:
            violations.append(f"pivot confirmed {p['confirmed_index']} > {i}")
    for e in eq_slice:
        rel, dist = relation_to_price(e["level"], e["level"], trigger_close)
        struct_rows.append(
            {
                "object_type": e["type"],
                "structure_level": e["level"],
                "bias": 0,
                "source_index": e["anchor_index"],
                "confirmed_index": e["confirmed_index"],
                "age_bars": i - e["confirmed_index"],
                "zone_low": e["level"],
                "zone_high": e["level"],
                "relation": rel,
                "distance_pct": dist,
            }
        )
        if e["confirmed_index"] > i:
            violations.append(f"eq confirmed {e['confirmed_index']} > {i}")

    # ----- active OB level rows (ALL kept, no selection)
    ob_rows: list[dict] = []
    counts = {
        "bull_int": 0,
        "bear_int": 0,
        "bull_swg": 0,
        "bear_swg": 0,
    }
    for o in active:
        rel, dist = relation_to_price(
            o["bar_low"], o["bar_high"], trigger_close
        )
        typ = (
            "active_"
            f"{'bull' if o['bias'] == 1 else 'bear'}_"
            f"{'internal' if o['internal'] else 'swing'}_ob"
        )
        ob_rows.append(
            {
                "object_type": typ,
                "structure_level": (o["bar_low"] + o["bar_high"]) / 2.0,
                "bias": o["bias"],
                "source_index": o["anchor_index"],
                "confirmed_index": o["confirmed_index"],
                "age_bars": i - o["confirmed_index"],
                "zone_low": o["bar_low"],
                "zone_high": o["bar_high"],
                "relation": rel,
                "distance_pct": dist,
            }
        )
        if o["bias"] == 1 and o["internal"]:
            counts["bull_int"] += 1
        elif o["bias"] == -1 and o["internal"]:
            counts["bear_int"] += 1
        elif o["bias"] == 1 and not o["internal"]:
            counts["bull_swg"] += 1
        else:
            counts["bear_swg"] += 1

    selected = select_nearby_levels(struct_rows)
    above = sorted(
        [r for r in selected if r["relation"] == "above"],
        key=lambda r: r["distance_pct"],
    )
    below = sorted(
        [r for r in selected if r["relation"] == "below"],
        key=lambda r: r["distance_pct"],
    )

    level_rows = ob_rows + selected

    return counts, level_rows, (above[0] if above else None), (
        below[0] if below else None
    )


def _struct_fields(
    ev: dict | None,
    i: int,
) -> tuple[str, int, float, float]:
    if ev is None:
        return ("NONE", 0, float("nan"), float("nan"))
    return (
        str(ev["type"]),
        int(ev["bias"]),
        float(ev["level"]),
        float(i - int(ev["confirmed_index"])),
    )


# ============================================================
# Forward outcomes (entry = next 5m open)
# ============================================================

def _outcome_vectors(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    h: int,
) -> dict[str, np.ndarray]:
    n = len(close)
    nan = np.full(n, np.nan)

    if n < h + 1:
        return {
            k: nan.copy()
            for k in (
                "ret",
                "long_mfe",
                "long_mae",
                "short_mfe",
                "short_mae",
            )
        }

    entry = np.log(open_[1 : n - h + 1])
    win_close = np.lib.stride_tricks.sliding_window_view(close, h + 1)
    win_high = np.lib.stride_tricks.sliding_window_view(high, h + 1)
    win_low = np.lib.stride_tricks.sliding_window_view(low, h + 1)

    exit_ = win_close[:, h]  # close at i+h
    ret = nan.copy()
    ret[: n - h] = np.log(exit_) - entry

    path_h = win_high[:, 1:]  # bars i+1 .. i+h
    path_l = win_low[:, 1:]
    up = np.log(path_h) - entry[:, None]
    down = entry[:, None] - np.log(path_l)

    long_mfe = nan.copy()
    long_mfe[: n - h] = np.maximum(up.max(axis=1), 0.0)
    long_mae = nan.copy()
    long_mae[: n - h] = np.maximum(down.max(axis=1), 0.0)
    short_mfe = nan.copy()
    short_mfe[: n - h] = np.maximum(down.max(axis=1), 0.0)
    short_mae = nan.copy()
    short_mae[: n - h] = np.maximum(up.max(axis=1), 0.0)

    return {
        "ret": ret,
        "long_mfe": long_mfe,
        "long_mae": long_mae,
        "short_mfe": short_mfe,
        "short_mae": short_mae,
    }


# ============================================================
# Per-symbol processing
# ============================================================

def _snapshot_tf_columns(
    tf: str,
    i,
    bar_end,
    smc: dict,
    precomp: dict,
    trigger_close: float,
    trigger_time: pd.Timestamp,
    event_id: str,
    symbol: str,
    violations: list[str],
) -> tuple[dict, list[dict]]:
    cols: dict = {}
    prefix = f"{tf}_"

    if i is None or bar_end is None:
        cols[f"{prefix}bar_index"] = None
        cols[f"{prefix}bar_end"] = None
        cols[f"{prefix}lag_minutes"] = None
        cols[f"{prefix}swing_bias"] = None
        cols[f"{prefix}internal_bias"] = None
        cols[f"{prefix}last_swing_structure_type"] = None
        cols[f"{prefix}last_swing_structure_bias"] = None
        cols[f"{prefix}last_swing_structure_level"] = None
        cols[f"{prefix}last_swing_structure_age"] = None
        cols[f"{prefix}last_internal_structure_type"] = None
        cols[f"{prefix}last_internal_structure_bias"] = None
        cols[f"{prefix}last_internal_structure_level"] = None
        cols[f"{prefix}last_internal_structure_age"] = None
        cols[f"{prefix}active_bull_internal_ob_count"] = None
        cols[f"{prefix}active_bear_internal_ob_count"] = None
        cols[f"{prefix}active_bull_swing_ob_count"] = None
        cols[f"{prefix}active_bear_swing_ob_count"] = None
        cols[f"{prefix}nearest_above_type"] = None
        cols[f"{prefix}nearest_above_distance_pct"] = None
        cols[f"{prefix}nearest_below_type"] = None
        cols[f"{prefix}nearest_below_distance_pct"] = None
        return cols, []

    # STOP condition 1: higher-TF bar must have ended by trigger_time.
    if pd.Timestamp(bar_end) > trigger_time:
        violations.append(
            f"{symbol} {tf}: bar_end {bar_end} > trigger {trigger_time}"
        )

    state = state_at(smc, i)
    cols[f"{prefix}bar_index"] = int(i)
    cols[f"{prefix}bar_end"] = str(bar_end)
    cols[f"{prefix}lag_minutes"] = round(
        (trigger_time - pd.Timestamp(bar_end)).total_seconds() / 60.0, 4
    )
    cols[f"{prefix}swing_bias"] = state["swing_bias"]
    cols[f"{prefix}internal_bias"] = state["internal_bias"]

    ls = last_structure_event(smc, i, internal=False)
    li = last_structure_event(smc, i, internal=True)
    lst = _struct_fields(ls, i)
    lit = _struct_fields(li, i)
    cols[f"{prefix}last_swing_structure_type"] = lst[0]
    cols[f"{prefix}last_swing_structure_bias"] = lst[1]
    cols[f"{prefix}last_swing_structure_level"] = lst[2]
    cols[f"{prefix}last_swing_structure_age"] = lst[3]
    cols[f"{prefix}last_internal_structure_type"] = lit[0]
    cols[f"{prefix}last_internal_structure_bias"] = lit[1]
    cols[f"{prefix}last_internal_structure_level"] = lit[2]
    cols[f"{prefix}last_internal_structure_age"] = lit[3]

    counts, level_rows, nearest_above, nearest_below = _gather_tf_levels(
        smc, precomp, i, trigger_close, violations
    )
    cols[f"{prefix}active_bull_internal_ob_count"] = counts["bull_int"]
    cols[f"{prefix}active_bear_internal_ob_count"] = counts["bear_int"]
    cols[f"{prefix}active_bull_swing_ob_count"] = counts["bull_swg"]
    cols[f"{prefix}active_bear_swing_ob_count"] = counts["bear_swg"]

    if nearest_above is not None:
        cols[f"{prefix}nearest_above_type"] = nearest_above["object_type"]
        cols[f"{prefix}nearest_above_distance_pct"] = nearest_above[
            "distance_pct"
        ]
    if nearest_below is not None:
        cols[f"{prefix}nearest_below_type"] = nearest_below["object_type"]
        cols[f"{prefix}nearest_below_distance_pct"] = nearest_below[
            "distance_pct"
        ]

    records = []
    for r in level_rows:
        records.append(
            {
                "event_id": event_id,
                "symbol": symbol,
                "trigger_time": str(trigger_time),
                "timeframe": tf,
                "object_type": r["object_type"],
                "structure_level": r["structure_level"],
                "bias": r["bias"],
                "source_index": r["source_index"],
                "confirmed_index": r["confirmed_index"],
                "age_bars": r["age_bars"],
                "zone_low": r["zone_low"],
                "zone_high": r["zone_high"],
                "relation": r["relation"],
                "distance_pct": r["distance_pct"],
            }
        )

    return cols, records


def process_symbol(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    print("=" * 64)
    print(symbol)
    print("=" * 64)

    five = pd.read_csv(
        SRC / f"{symbol}_5m.csv",
        parse_dates=[
            "bar_start_time",
            "bar_end_time",
            "availability_time",
            "trading_day",
            "tdx_datetime_raw",
        ],
    ).sort_values("bar_start_time").reset_index(drop=True)

    # build_smc_tf needs `volume`; aggregate_15m needs `trade`/`position`.
    five["volume"] = five["trade"]

    n = len(five)
    print(f"  5m bars : {n}")

    five_ind, smc5 = build_smc_tf(five)
    fifteen = aggregate_15m(five)
    fifteen_ind, smc15 = build_smc_tf(fifteen)
    one_hour = aggregate_1h_from_15m(fifteen)
    hour_ind, smc1h = build_smc_tf(one_hour)

    print(
        f"  15m bars: {len(fifteen)}   1h bars: {len(one_hour)}"
    )

    precomp5 = _precomp_tf(smc5)
    precomp15 = _precomp_tf(smc15)
    precomp1h = _precomp_tf(smc1h)

    open_ = five["open"].to_numpy(float)
    high = five["high"].to_numpy(float)
    low = five["low"].to_numpy(float)
    close = five["close"].to_numpy(float)

    outcome_vecs = {
        h: _outcome_vectors(open_, high, low, close, h)
        for h in HORIZONS_5M
    }

    events = [
        ev
        for ev in smc5.get("ob_lifecycle_events", [])
        if ev.get("type") == "OB_ENTERED"
    ]
    n_events = len(events)
    print(f"  OB_ENTERED events: {n_events}")

    violations: list[str] = []

    snap_rows: list[dict] = []
    level_rows: list[dict] = []

    for ev in events:
        i5 = int(ev["enter_index"])
        trigger_time = pd.Timestamp(five.iloc[i5]["availability_time"])

        i15 = asof_completed_index(fifteen, trigger_time)
        i1h = asof_completed_index(one_hour, trigger_time)

        bar_end5 = five.iloc[i5]["bar_end_time"]
        bar_end15 = (
            fifteen.iloc[i15]["bar_end_time"] if i15 is not None else None
        )
        bar_end1h = (
            one_hour.iloc[i1h]["bar_end_time"] if i1h is not None else None
        )

        trigger_close = float(five.iloc[i5]["close"])
        trigger_bar_low = float(five.iloc[i5]["low"])
        trigger_bar_high = float(five.iloc[i5]["high"])

        bias = int(ev["bias"])
        internal = bool(ev["internal"])
        bar_low = float(ev["bar_low"])
        bar_high = float(ev["bar_high"])
        enter_index = int(ev["enter_index"])
        confirmed_index = int(ev["confirmed_index"])
        anchor_index = int(ev["anchor_index"])

        ob_width = bar_high - bar_low
        overlap = max(
            0.0,
            min(trigger_bar_high, bar_high)
            - max(trigger_bar_low, bar_low),
        )
        overlap_fraction = (
            overlap / ob_width if ob_width > 0 else float("nan")
        )
        width_pct = (
            (bar_high - bar_low) / trigger_close * 100.0
            if trigger_close > 0
            else float("nan")
        )

        event_id = (
            f"{symbol}:"
            f"{enter_index}:"
            f"{'I' if internal else 'S'}:"
            f"{bias}:"
            f"{anchor_index}:"
            f"{confirmed_index}"
        )

        row: dict = {
            "event_id": event_id,
            "symbol": symbol,
            "trigger_time": str(trigger_time),
            "trigger_bar_index": i5,
            "trigger_ob_bias": bias,
            "trigger_ob_structure": "internal" if internal else "swing",
            "trigger_ob_age_bars": enter_index - confirmed_index,
            "trigger_ob_low": bar_low,
            "trigger_ob_high": bar_high,
            "trigger_ob_width_pct": width_pct,
            "trigger_ob_overlap_fraction": overlap_fraction,
            "trigger_ob_anchor_index": anchor_index,
            "trigger_ob_anchor_time": str(ev["anchor_time"]),
            "trigger_ob_confirmed_index": confirmed_index,
            "trigger_ob_confirmed_time": str(ev["confirmed_time"]),
            "trigger_ob_enter_index": enter_index,
            "trigger_ob_enter_time": str(ev["enter_time"]),
        }

        c5, r5 = _snapshot_tf_columns(
            "5m", i5, bar_end5, smc5, precomp5,
            trigger_close, trigger_time, event_id, symbol, violations,
        )
        c15, r15 = _snapshot_tf_columns(
            "15m", i15, bar_end15, smc15, precomp15,
            trigger_close, trigger_time, event_id, symbol, violations,
        )
        c1h, r1h = _snapshot_tf_columns(
            "1h", i1h, bar_end1h, smc1h, precomp1h,
            trigger_close, trigger_time, event_id, symbol, violations,
        )
        row.update(c5)
        row.update(c15)
        row.update(c1h)

        # ----- forward outcomes
        for h in HORIZONS_5M:
            vec = outcome_vecs[h]
            ret = vec["ret"][i5]
            lmfe = vec["long_mfe"][i5]
            lmae = vec["long_mae"][i5]
            smfe = vec["short_mfe"][i5]
            smae = vec["short_mae"][i5]

            row[f"fwd_raw_ret_h{h}"] = ret
            row[f"fwd_long_mfe_h{h}"] = lmfe
            row[f"fwd_long_mae_h{h}"] = lmae
            row[f"fwd_short_mfe_h{h}"] = smfe
            row[f"fwd_short_mae_h{h}"] = smae

            if np.isnan(ret):
                dret = float("nan")
                dmfe = float("nan")
                dmae = float("nan")
                contig = 0.0
            else:
                dret = bias * ret
                if bias == 1:
                    dmfe = lmfe
                    dmae = lmae
                    contig = (
                        1.0
                        if (not np.isnan(smae) and smae == 0.0)
                        else 0.0
                    )
                else:
                    dmfe = smfe
                    dmae = smae
                    contig = (
                        1.0
                        if (not np.isnan(lmae) and lmae == 0.0)
                        else 0.0
                    )

            row[f"fwd_dir_ret_h{h}"] = dret
            row[f"fwd_dir_mfe_h{h}"] = dmfe
            row[f"fwd_dir_mae_h{h}"] = dmae
            row[f"fwd_contig_h{h}"] = contig

        snap_rows.append(row)
        level_rows.extend(r5)
        level_rows.extend(r15)
        level_rows.extend(r1h)

    snap_df = pd.DataFrame(snap_rows)
    lvl_df = pd.DataFrame(level_rows)

    usable = {
        h: int(snap_df[f"fwd_raw_ret_h{h}"].notna().sum())
        for h in HORIZONS_5M
    }

    stats = {
        "symbol": symbol,
        "five_minute_bars": int(n),
        "fifteen_minute_bars": int(len(fifteen)),
        "one_hour_bars": int(len(one_hour)),
        "ob_entered_events": int(n_events),
        "snapshot_rows": int(len(snap_df)),
        "level_rows": int(len(lvl_df)),
        "forward_usable_by_horizon": usable,
        "stop_violations": list(violations),
    }

    if violations:
        raise RuntimeError(
            f"{symbol}: STOP conditions violated:\n"
            + "\n".join(violations[:20])
        )

    # STOP condition 6
    if len(snap_df) != n_events:
        raise RuntimeError(
            f"{symbol}: snapshot rows {len(snap_df)} != "
            f"OB_ENTERED events {n_events}"
        )

    print(
        f"  snapshots={len(snap_df)}  levels={len(lvl_df)}  "
        f"usable_h24={usable[24]}/{n_events}"
    )

    return snap_df, lvl_df, stats


# ============================================================
# Main
# ============================================================

def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(
            f"{OUT} exists and is non-empty. Delete only for an "
            "intentional pre-commit rerun."
        )

    OUT.mkdir(parents=True, exist_ok=True)

    snap_frames: list[pd.DataFrame] = []
    level_frames: list[pd.DataFrame] = []
    stats_all: list[dict] = []

    for symbol in SYMBOLS:
        snap_df, lvl_df, stats = process_symbol(symbol)
        snap_frames.append(snap_df)
        level_frames.append(lvl_df)
        stats_all.append(stats)

    snapshots = pd.concat(snap_frames, ignore_index=True)
    levels = pd.concat(level_frames, ignore_index=True)

    snapshots.to_csv(OUT / "trigger_snapshots.csv", index=False)
    levels.to_csv(OUT / "trigger_levels.csv", index=False)

    total_events = sum(s["ob_entered_events"] for s in stats_all)
    total_rows = len(snapshots)
    if total_rows != total_events:
        raise RuntimeError(
            f"TOTAL snapshot rows {total_rows} != "
            f"total OB_ENTERED events {total_events}"
        )

    manifest = {
        "status": "PASS",
        "schema_version": "ob_trigger_smc_v1",
        "discovery_universe": SYMBOLS,
        "sample_definition": (
            "one row per canonical 5m SMC ob_lifecycle_events "
            "type==OB_ENTERED; same-bar bullish internal + bullish swing "
            "OB => two events (keyed by internal/swing), never collapsed"
        ),
        "time_semantics": {
            "trigger_time": "5m availability_time[i] where i = OB enter_index",
            "5m_context": "SMC state of bar i itself",
            "15m_context": (
                "last 15m bar with bar_end_time <= trigger_time "
                "(asof_completed_index)"
            ),
            "1h_context": (
                "last 1h bar with bar_end_time <= trigger_time "
                "(asof_completed_index)"
            ),
            "entry_assumption": (
                "next 5m open after trigger bar; exit = close H 5m bars "
                "later; H in {1,3,6,12,24} (5m/15m/30m/60m/120m)"
            ),
        },
        "aggregation": {
            "15m": "aggregate_15m from build_pytdx_panel (canonical)",
            "1h": (
                "aggregate_1h_from_15m: 15m bars bucketed by integer "
                "hour-epoch, OHLC/volume/OI aggregated; no 'exactly 4 "
                "bars' rule"
            ),
            "smc": (
                "compute_smc_momentum_bundle (canonical SMC only) per TF; "
                "no redefinition of indicator semantics"
            ),
        },
        "outcome_columns": {
            "per_horizon": [
                "fwd_raw_ret_hH",
                "fwd_long_mfe_hH",
                "fwd_long_mae_hH",
                "fwd_short_mfe_hH",
                "fwd_short_mae_hH",
                "fwd_dir_ret_hH (= trigger_ob_bias * raw_ret)",
                "fwd_dir_mfe_hH (= long_mfe if bullish else short_mfe)",
                "fwd_dir_mae_hH (= long_mae if bullish else short_mae)",
                "fwd_contig_hH (=1 if favorable side never breached "
                "entry against bias, else 0)",
            ],
            "horizons_5m_bars": HORIZONS_5M,
        },
        "stop_conditions_checked": [
            "higher-TF snapshot bar_end_time <= trigger_time",
            "no pivot confirmed_index > snapshot index written",
            "no BOS/CHoCH confirmed_index > snapshot index written",
            "no OB confirmed_index > snapshot index in active set",
            "no OB mitigated_index <= snapshot index kept active",
            "trigger_snapshots rows == canonical OB_ENTERED event count",
        ],
        "by_symbol": stats_all,
        "totals": {
            "ob_entered_events": int(total_events),
            "snapshot_rows": int(total_rows),
            "level_rows": int(len(levels)),
        },
    }

    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 64)
    print("OB_TRIGGER_SNAPSHOT_V1_BUILD_PASS")
    print("=" * 64)
    for s in stats_all:
        print(
            f"  {s['symbol']:3s} events={s['ob_entered_events']:5d} "
            f"snap={s['snapshot_rows']:5d} "
            f"levels={s['level_rows']:8d} "
            f"usable_h24={s['forward_usable_by_horizon'][24]}"
        )


if __name__ == "__main__":
    main()
