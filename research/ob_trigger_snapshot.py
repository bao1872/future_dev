#!/usr/bin/env python3
"""5m OB Trigger Multi-TF SMC Snapshot V1 (corrected spatial semantics).

Research sample = a canonical 5m SMC ``OB_ENTERED`` event, NOT a bar.

For every 5m OB_ENTERED event we freeze, at the trigger instant, the full
SMC environment that was *actually knowable then* (point-in-time, no
lookahead), across 5m / 15m / 1h.

This revision fixes the spatial-semantics defects flagged in audit
1727c171:

  * Current structural pivots are the LATEST confirmed value of each
    canonical type (internal_high/low, swing_high/low) -- NOT every
    historical pivot.
  * nearest above/below pressure/support is computed over the FULL
    candidate set (active OBs + current pivots + EQ), so OBs are
    eligible as the nearest level.
  * Active OBs come from a FULL-OB context (swing OB display/calculation
    enabled) so the high-timeframe pressure/support map is complete,
    while the trigger universe stays the canonical default internal
    OB_ENTERED events.
  * 5m/15m/1h bar OHLCV/OI written per snapshot.
  * Trigger-OB same-bar life/death recorded.
  * ``fwd_contig`` removed (wrong semantics); replaced by
    ``fwd_time_contiguous`` (true 5m bar-spacing continuity).

All SMC facts are CONSUMED from the canonical output of
``compute_smc_pine`` (via ``compute_smc_momentum_bundle`` for the default
context and directly for the full-OB context). Indicator semantics are
never redefined.

Active-OB reconstruction replays the canonical active-list semantics
exactly (creation -> insert-front, >=100 cap evicts oldest, mitigation
removes by identity). Verified: 0 mismatches vs ``state_timeline``
active counts over the full history of every timeframe.
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

from panji_indicators import compute_smc_pine  # noqa: E402
from research.build_pytdx_panel import aggregate_15m  # noqa: E402
from research.indicator_adapter import (  # noqa: E402
    compute_smc_momentum_bundle,
)

SYMBOLS = ["AG", "CU", "RB", "M"]

SRC = ROOT / "research" / "exports" / "v3r_5m"
OUT_LOCAL = ROOT / "research" / "exports" / "ob_trigger_smc_v1"
OUT_GIT = ROOT / "research" / "analysis_data" / "ob_trigger_smc_v1"

# 5m / 15m / 30m / 60m / 120m expressed as a count of 5m bars.
HORIZONS_5M = [1, 3, 6, 12, 24]

ONE_HOUR_NS = 60 * 60 * 1_000_000_000

CURRENT_PIVOT_TYPES = (
    "internal_high",
    "internal_low",
    "swing_high",
    "swing_low",
)

EXPECTED_BAR_NS = 5 * 60 * 1_000_000_000


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
        x["bar_start_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    )
    x["_bucket"] = (start_ns // ONE_HOUR_NS) * ONE_HOUR_NS

    return (
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


# ============================================================
# Canonical SMC builders
# ============================================================

def build_smc_tf(bars: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Canonical default SMC (swing OB display/calculation OFF).

    Owns the trigger universe + trend/pivot/BOS-CHoCH continuity exactly
    as we visually inspected it.
    """

    ind = bars.copy()
    if "volume" not in ind.columns:
        raise ValueError("SMC frame requires volume")
    ind = ind.set_index(pd.DatetimeIndex(ind["bar_start_time"]))
    bundle = compute_smc_momentum_bundle(ind)
    return ind, bundle.smc


def build_full_ob_smc_tf(bars: pd.DataFrame) -> dict:
    """Canonical SMC formulas, with swing OB display/calculation ON.

    This does NOT change trigger semantics. It is used only to expose the
    full internal + swing OB spatial context.
    """

    ind = bars.copy()
    ind = ind.set_index(pd.DatetimeIndex(ind["bar_start_time"]))
    times = [ts.isoformat() for ts in ind.index]
    return compute_smc_pine(
        ind["open"].astype(float).tolist(),
        ind["high"].astype(float).tolist(),
        ind["low"].astype(float).tolist(),
        ind["close"].astype(float).tolist(),
        times,
        params={
            "show_internal_order_blocks": True,
            "show_swing_order_blocks": True,
        },
        emit_timeline=True,
    )


# ============================================================
# Point-in-time index helpers
# ============================================================

def asof_completed_index(
    bars: pd.DataFrame,
    trigger_time: pd.Timestamp,
) -> int | None:
    """Index of the last bar whose bar_end_time <= trigger_time."""

    ends = pd.to_datetime(bars["bar_end_time"]).to_numpy(
        dtype="datetime64[ns]"
    )
    target = np.datetime64(trigger_time, "ns")
    idx = int(np.searchsorted(ends, target, side="right") - 1)
    if idx < 0:
        return None
    if not (pd.Timestamp(bars.iloc[idx]["bar_end_time"]) <= trigger_time):
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
    smc: dict, i: int, *, internal: bool
) -> dict | None:
    candidates = [
        ev
        for ev in smc.get("events", [])
        if bool(ev["internal"]) == internal
        and int(ev["confirmed_index"]) <= i
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda ev: int(ev["confirmed_index"]))


def active_order_blocks_replay(
    smc: dict, i: int, internal_flag: bool, cap: int = 100
) -> list[dict]:
    """Reconstruct the canonical active OB list at bar ``i``.

    Faithful to ``panji_indicators``: at creation the OB is inserted at
    the front of the per-(internal/swing) list; when the list reaches
    the 100 cap the oldest is evicted; a crossed (mitigated) OB is
    removed by identity. ``entered`` does NOT remove an OB.
    """

    evs = []
    for e in smc.get("ob_lifecycle_events", []):
        if e["type"] == "OB_CREATED" and bool(e["internal"]) == internal_flag:
            evs.append((int(e["confirmed_index"]), 0, e))
        elif (
            e["type"] == "OB_MITIGATED"
            and bool(e["internal"]) == internal_flag
        ):
            evs.append((int(e["mitigated_index"]), 1, e))
    evs.sort(key=lambda x: (x[0], x[1]))

    target: list[dict] = []
    for _bar, kind, e in evs:
        if _bar > i:
            continue
        if kind == 0:
            ob = {
                "anchor_index": int(e["anchor_index"]),
                "confirmed_index": int(e["confirmed_index"]),
                "bias": int(e["bias"]),
                "internal": bool(e["internal"]),
                "bar_low": float(e["bar_low"]),
                "bar_high": float(e["bar_high"]),
            }
            target.insert(0, ob)
            if len(target) > cap:
                target.pop()
        else:
            key = (
                int(e["anchor_index"]),
                int(e["confirmed_index"]),
                int(e["bias"]),
            )
            for j, t in enumerate(target):
                if (
                    t["anchor_index"],
                    t["confirmed_index"],
                    t["bias"],
                ) == key:
                    target.pop(j)
                    break
    return target


def precompute_active_at(
    smc: dict, needed: set[int], cap: int = 100
) -> dict[int, dict]:
    """Snapshot the faithful active OB list at the requested bar indices.

    Returns ``{bar: {"internal": [...], "swing": [...]}}``.
    """

    evs = []
    for e in smc.get("ob_lifecycle_events", []):
        if e["type"] == "OB_CREATED":
            evs.append((int(e["confirmed_index"]), 0, e))
        elif e["type"] == "OB_MITIGATED":
            evs.append((int(e["mitigated_index"]), 1, e))
    evs.sort(key=lambda x: (x[0], x[1]))

    int_target: list[dict] = []
    swg_target: list[dict] = []
    out: dict[int, dict] = {}

    for bar, kind, e in evs:
        if kind == 0:
            ob = {
                "anchor_index": int(e["anchor_index"]),
                "confirmed_index": int(e["confirmed_index"]),
                "bias": int(e["bias"]),
                "internal": bool(e["internal"]),
                "bar_low": float(e["bar_low"]),
                "bar_high": float(e["bar_high"]),
            }
            if bool(e["internal"]):
                int_target.insert(0, ob)
                if len(int_target) > cap:
                    int_target.pop()
            else:
                swg_target.insert(0, ob)
                if len(swg_target) > cap:
                    swg_target.pop()
        else:
            key = (
                int(e["anchor_index"]),
                int(e["confirmed_index"]),
                int(e["bias"]),
            )
            if bool(e["internal"]):
                for j, t in enumerate(int_target):
                    if (
                        t["anchor_index"],
                        t["confirmed_index"],
                        t["bias"],
                    ) == key:
                        int_target.pop(j)
                        break
            else:
                for j, t in enumerate(swg_target):
                    if (
                        t["anchor_index"],
                        t["confirmed_index"],
                        t["bias"],
                    ) == key:
                        swg_target.pop(j)
                        break
        if bar in needed:
            out[bar] = {
                "internal": list(int_target),
                "swing": list(swg_target),
            }
    return out


# ============================================================
# Pivot / EQ helpers (current structural pivots, not all history)
# ============================================================

def _precomp_piv(smc: dict) -> dict:
    by_type: dict[str, list] = {t: [] for t in CURRENT_PIVOT_TYPES}
    for p in smc.get("pivots", []):
        typ = p.get("type")
        if typ not in by_type:
            continue
        by_type[typ].append(p)
    piv_by_type = {}
    piv_conf_by_type = {}
    for typ, lst in by_type.items():
        lst.sort(key=lambda d: int(d["confirmed_index"]))
        piv_by_type[typ] = lst
        piv_conf_by_type[typ] = [int(d["confirmed_index"]) for d in lst]
    return {"piv_by_type": piv_by_type, "piv_conf_by_type": piv_conf_by_type}


def _precomp_eq(smc: dict) -> dict:
    eq = [e for e in smc.get("equal_highs_lows", [])]
    eq.sort(key=lambda d: int(d["confirmed_index"]))
    return {
        "eq": eq,
        "eq_conf": [int(d["confirmed_index"]) for d in eq],
    }


def current_pivots_at(precomp: dict, i: int) -> dict[str, dict | None]:
    """Latest confirmed pivot of each canonical structural type."""

    out: dict[str, dict | None] = {}
    for typ in CURRENT_PIVOT_TYPES:
        confs = precomp["piv_conf_by_type"][typ]
        pivs = precomp["piv_by_type"][typ]
        k = bisect.bisect_right(confs, i)
        out[typ] = pivs[k - 1] if k > 0 else None
    return out


def confirmed_equal_levels_at(precomp: dict, i: int) -> list[dict]:
    k = bisect.bisect_right(precomp["eq_conf"], i)
    return precomp["eq"][:k]


def current_pivot_fields(
    precomp: dict, i: int, price: float
) -> dict:
    pivots = current_pivots_at(precomp, i)
    out: dict = {}
    for typ, p in pivots.items():
        if p is None:
            for suf in (
                "_level",
                "_age",
                "_relation",
                "_distance_pct",
                "_last_level",
            ):
                out[f"current_{typ}{suf}"] = None
            continue
        level = float(p["level"])
        rel, dist = relation_to_price(level, level, price)
        out[f"current_{typ}_level"] = level
        out[f"current_{typ}_age"] = i - int(p["confirmed_index"])
        out[f"current_{typ}_relation"] = rel
        out[f"current_{typ}_distance_pct"] = dist
        last = p.get("last_level")
        out[f"current_{typ}_last_level"] = (
            float(last) if last is not None else None
        )
    return out


# ============================================================
# Structure maps for the long table
# ============================================================

def relation_to_price(
    low: float, high: float, price: float
) -> tuple[str, float]:
    low = float(low)
    high = float(high)
    price = float(price)
    if low <= price <= high:
        return ("overlap", 0.0)
    if low > price:
        return ("above", (low - price) / price)
    return ("below", (price - high) / price)


def select_nearby_levels(rows: list[dict], k: int = 5) -> list[dict]:
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
# Trigger-OB same-bar life/death
# ============================================================

def find_trigger_mitigation(
    ev: dict, smc: dict
) -> dict | None:
    for x in smc.get("ob_lifecycle_events", []):
        if x.get("type") != "OB_MITIGATED":
            continue
        if (
            int(x["anchor_index"]) == int(ev["anchor_index"])
            and int(x["confirmed_index"]) == int(ev["confirmed_index"])
            and int(x["bias"]) == int(ev["bias"])
            and bool(x["internal"]) == bool(ev["internal"])
        ):
            return x
    return None


# ============================================================
# Forward continuity (true 5m bar-spacing)
# ============================================================

def forward_contiguous(five: pd.DataFrame, i: int, h: int) -> bool:
    if i + h >= len(five):
        return False
    t = (
        pd.to_datetime(five["bar_start_time"])
        .iloc[i : i + h + 1]
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )
    return bool(np.all(np.diff(t) == EXPECTED_BAR_NS))


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

    exit_ = win_close[:, h]
    ret = nan.copy()
    ret[: n - h] = np.log(exit_) - entry

    path_h = win_high[:, 1:]
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
# Per-timeframe column schema
# ============================================================

def _tf_column_keys(prefix: str) -> list[str]:
    keys = [
        f"{prefix}bar_index",
        f"{prefix}bar_end",
        f"{prefix}lag_minutes",
        f"{prefix}swing_bias",
        f"{prefix}internal_bias",
        f"{prefix}last_swing_structure_type",
        f"{prefix}last_swing_structure_bias",
        f"{prefix}last_swing_structure_level",
        f"{prefix}last_swing_structure_age",
        f"{prefix}last_internal_structure_type",
        f"{prefix}last_internal_structure_bias",
        f"{prefix}last_internal_structure_level",
        f"{prefix}last_internal_structure_age",
        f"{prefix}active_bull_internal_ob_count",
        f"{prefix}active_bear_internal_ob_count",
        f"{prefix}active_bull_swing_ob_count",
        f"{prefix}active_bear_swing_ob_count",
        f"{prefix}nearest_above_type",
        f"{prefix}nearest_above_distance_pct",
        f"{prefix}nearest_below_type",
        f"{prefix}nearest_below_distance_pct",
        f"{prefix}open",
        f"{prefix}high",
        f"{prefix}low",
        f"{prefix}close",
        f"{prefix}volume",
        f"{prefix}close_oi",
        f"{prefix}overlap_object_count",
        f"{prefix}overlap_types",
    ]
    for typ in CURRENT_PIVOT_TYPES:
        for suf in (
            "_level",
            "_age",
            "_relation",
            "_distance_pct",
            "_last_level",
        ):
            keys.append(f"{prefix}current_{typ}{suf}")
    return keys


# ============================================================
# Per-timeframe snapshot + level gathering
# ============================================================

def _gather_tf_levels(
    precomp_piv_def: dict,
    precomp_eq_def: dict,
    active_snap: dict,
    i: int,
    trigger_close: float,
    violations: list[str],
) -> tuple[dict, list[dict], dict | None, dict | None, list[dict]]:
    """Return (counts, level_rows, nearest_above, nearest_below, overlap)."""

    current = current_pivots_at(precomp_piv_def, i)
    eq = confirmed_equal_levels_at(precomp_eq_def, i)

    struct_rows: list[dict] = []
    for typ, p in current.items():
        if p is None:
            continue
        level = float(p["level"])
        rel, dist = relation_to_price(level, level, trigger_close)
        struct_rows.append(
            {
                "object_type": f"current_{typ}",
                "structure_class": (
                    "internal" if typ.startswith("internal") else "swing"
                ),
                "bias": 0,
                "object_price_center": level,
                "source_index": int(p["anchor_index"]),
                "confirmed_index": int(p["confirmed_index"]),
                "age_bars": i - int(p["confirmed_index"]),
                "zone_low": level,
                "zone_high": level,
                "relation": rel,
                "distance_pct": dist,
            }
        )
        if int(p["confirmed_index"]) > i:
            violations.append(f"current pivot confirmed {int(p['confirmed_index'])} > {i}")
    for e in eq:
        level = float(e["level"])
        rel, dist = relation_to_price(level, level, trigger_close)
        struct_rows.append(
            {
                "object_type": str(e["type"]),
                "structure_class": "equal",
                "bias": 0,
                "object_price_center": level,
                "source_index": int(e["anchor_index"]),
                "confirmed_index": int(e["confirmed_index"]),
                "age_bars": i - int(e["confirmed_index"]),
                "zone_low": level,
                "zone_high": level,
                "relation": rel,
                "distance_pct": dist,
            }
        )
        if int(e["confirmed_index"]) > i:
            violations.append(f"eq confirmed {int(e['confirmed_index'])} > {i}")

    # active OBs (from FULL-OB context snapshot)
    active = active_snap["internal"] + active_snap["swing"]
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
                "structure_class": "internal" if o["internal"] else "swing",
                "bias": int(o["bias"]),
                "object_price_center": (o["bar_low"] + o["bar_high"]) / 2.0,
                "source_index": int(o["anchor_index"]),
                "confirmed_index": int(o["confirmed_index"]),
                "age_bars": i - int(o["confirmed_index"]),
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

    selected_struct = select_nearby_levels(struct_rows)
    level_rows = ob_rows + selected_struct

    above = sorted(
        [r for r in level_rows if r["relation"] == "above"],
        key=lambda r: r["distance_pct"],
    )
    below = sorted(
        [r for r in level_rows if r["relation"] == "below"],
        key=lambda r: r["distance_pct"],
    )
    overlap = sorted(
        [r for r in level_rows if r["relation"] == "overlap"],
        key=lambda r: (r["object_type"], r["distance_pct"]),
    )

    return (
        counts,
        level_rows,
        (above[0] if above else None),
        (below[0] if below else None),
        overlap,
    )


def _snapshot_tf_columns(
    tf: str,
    i,
    bar_end,
    smc_default: dict,
    smc_obctx: dict,
    precomp_piv_def: dict,
    precomp_eq_def: dict,
    active_snap: dict,
    bars: pd.DataFrame,
    trigger_close: float,
    trigger_time: pd.Timestamp,
    event_id: str,
    symbol: str,
    violations: list[str],
) -> tuple[dict, list[dict], list[dict]]:
    prefix = f"{tf}_"
    cols: dict = {}

    if i is None or bar_end is None:
        for k in _tf_column_keys(prefix):
            cols[k] = None
        return cols, [], []

    # STOP condition 1
    if pd.Timestamp(bar_end) > trigger_time:
        violations.append(
            f"{symbol} {tf}: bar_end {bar_end} > trigger {trigger_time}"
        )

    state = state_at(smc_default, i)
    cols[f"{prefix}bar_index"] = int(i)
    cols[f"{prefix}bar_end"] = str(bar_end)
    cols[f"{prefix}lag_minutes"] = round(
        (trigger_time - pd.Timestamp(bar_end)).total_seconds() / 60.0, 4
    )
    cols[f"{prefix}swing_bias"] = state["swing_bias"]
    cols[f"{prefix}internal_bias"] = state["internal_bias"]

    ls = last_structure_event(smc_default, i, internal=False)
    li = last_structure_event(smc_default, i, internal=True)
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

    counts, level_rows, nearest_above, nearest_below, overlap = (
        _gather_tf_levels(
            precomp_piv_def,
            precomp_eq_def,
            active_snap,
            i,
            trigger_close,
            violations,
        )
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

    # per-bar OHLCV/OI
    bar = bars.iloc[i]
    cols[f"{prefix}open"] = float(bar["open"])
    cols[f"{prefix}high"] = float(bar["high"])
    cols[f"{prefix}low"] = float(bar["low"])
    cols[f"{prefix}close"] = float(bar["close"])
    cols[f"{prefix}volume"] = float(bar["volume"])
    if "close_oi" in bar.index:
        cols[f"{prefix}close_oi"] = float(bar["close_oi"])
    elif "position" in bar.index:
        cols[f"{prefix}close_oi"] = float(bar["position"])
    else:
        cols[f"{prefix}close_oi"] = float("nan")

    # current pivots (prefixed per timeframe)
    for _k, _v in current_pivot_fields(
        precomp_piv_def, i, trigger_close
    ).items():
        cols[f"{prefix}{_k}"] = _v

    # overlap summary
    cols[f"{prefix}overlap_object_count"] = len(overlap)
    cols[f"{prefix}overlap_types"] = "|".join(
        sorted({r["object_type"] for r in overlap})
    )

    level_records = []
    for r in level_rows:
        level_records.append(
            {
                "event_id": event_id,
                "symbol": symbol,
                "trigger_time": str(trigger_time),
                "timeframe": tf,
                "object_type": r["object_type"],
                "structure_class": r["structure_class"],
                "bias": r["bias"],
                "object_price_center": r["object_price_center"],
                "source_index": r["source_index"],
                "confirmed_index": r["confirmed_index"],
                "age_bars": r["age_bars"],
                "zone_low": r["zone_low"],
                "zone_high": r["zone_high"],
                "relation": r["relation"],
                "distance_pct": r["distance_pct"],
            }
        )

    return cols, level_records, overlap


def _struct_fields(ev, i):
    if ev is None:
        return ("NONE", 0, float("nan"), float("nan"))
    return (
        str(ev["type"]),
        int(ev["bias"]),
        float(ev["level"]),
        float(i - int(ev["confirmed_index"])),
    )


# ============================================================
# Per-symbol processing
# ============================================================

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
    five["volume"] = five["trade"]

    n = len(five)
    print(f"  5m bars : {n}")

    # Default canonical SMC (trigger + trend/pivot/BOS-CHoCH)
    five_ind, smc5 = build_smc_tf(five)
    fifteen = aggregate_15m(five)
    fifteen_ind, smc15 = build_smc_tf(fifteen)
    one_hour = aggregate_1h_from_15m(fifteen)
    hour_ind, smc1h = build_smc_tf(one_hour)

    # Full-OB context (swing OB display/calculation ON)
    smc5_obctx = build_full_ob_smc_tf(five)
    smc15_obctx = build_full_ob_smc_tf(fifteen)
    smc1h_obctx = build_full_ob_smc_tf(one_hour)

    print(f"  15m bars: {len(fifteen)}   1h bars: {len(one_hour)}")

    # Phase B invariant: enabling swing OB must NOT change trend biases.
    for tf_name, default, full in (
        ("5m", smc5, smc5_obctx),
        ("15m", smc15, smc15_obctx),
        ("1h", smc1h, smc1h_obctx),
    ):
        if len(default["state_timeline"]) != len(full["state_timeline"]):
            raise RuntimeError(
                f"{symbol} {tf_name}: timeline length mismatch "
                "after enabling swing OB"
            )
        for a, b in zip(default["state_timeline"], full["state_timeline"]):
            if int(a["swing_bias"]) != int(b["swing_bias"]):
                raise RuntimeError(
                    f"{symbol} {tf_name}: swing_bias changed by "
                    "swing-OB enable (STOP)"
                )
            if int(a["internal_bias"]) != int(b["internal_bias"]):
                raise RuntimeError(
                    f"{symbol} {tf_name}: internal_bias changed by "
                    "swing-OB enable (STOP)"
                )
    print("  Phase B invariant: trend biases unchanged by swing-OB enable")

    precomp_piv5 = _precomp_piv(smc5)
    precomp_piv15 = _precomp_piv(smc15)
    precomp_piv1h = _precomp_piv(smc1h)
    precomp_eq5 = _precomp_eq(smc5)
    precomp_eq15 = _precomp_eq(smc15)
    precomp_eq1h = _precomp_eq(smc1h)

    # Trigger universe: canonical default internal OB_ENTERED (unchanged)
    events = [
        ev
        for ev in smc5.get("ob_lifecycle_events", [])
        if ev.get("type") == "OB_ENTERED"
    ]
    n_events = len(events)
    print(f"  OB_ENTERED events: {n_events}")

    # First pass: resolve snapshot indices + collect needed bars
    needed5: set[int] = set()
    needed15: set[int] = set()
    needed1h: set[int] = set()
    resolved = []
    for ev in events:
        i5 = int(ev["enter_index"])
        trigger_time = pd.Timestamp(five.iloc[i5]["availability_time"])
        i15 = asof_completed_index(fifteen, trigger_time)
        i1h = asof_completed_index(one_hour, trigger_time)
        needed5.add(i5)
        if i15 is not None:
            needed15.add(i15)
        if i1h is not None:
            needed1h.add(i1h)
        resolved.append((ev, i5, trigger_time, i15, i1h))

    active5 = precompute_active_at(smc5_obctx, needed5)
    active15 = precompute_active_at(smc15_obctx, needed15)
    active1h = precompute_active_at(smc1h_obctx, needed1h)

    open_ = five["open"].to_numpy(float)
    high = five["high"].to_numpy(float)
    low = five["low"].to_numpy(float)
    close = five["close"].to_numpy(float)
    outcome_vecs = {
        h: _outcome_vectors(open_, high, low, close, h)
        for h in HORIZONS_5M
    }

    violations: list[str] = []
    recon_mismatches = 0
    snap_rows: list[dict] = []
    level_rows: list[dict] = []

    for ev, i5, trigger_time, i15, i1h in resolved:
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

        # Trigger-OB same-bar life/death
        mit = find_trigger_mitigation(ev, smc5)
        mit_index = int(mit["mitigated_index"]) if mit is not None else None
        mit_same_bar = mit_index == i5
        alive_at_close = mit_index is None or mit_index > i5
        if mit_same_bar and alive_at_close:
            raise AssertionError("trigger OB lifecycle contradiction")

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
            "trigger_ob_mitigated_same_bar": bool(mit_same_bar),
            "trigger_ob_alive_at_close": bool(alive_at_close),
            "trigger_ob_mitigated_index": mit_index,
        }

        bar_end5 = five.iloc[i5]["bar_end_time"]
        bar_end15 = (
            fifteen.iloc[i15]["bar_end_time"] if i15 is not None else None
        )
        bar_end1h = (
            one_hour.iloc[i1h]["bar_end_time"] if i1h is not None else None
        )

        c5, r5, _ = _snapshot_tf_columns(
            "5m", i5, bar_end5, smc5, smc5_obctx,
            precomp_piv5, precomp_eq5, active5.get(i5, {"internal": [], "swing": []}),
            five, trigger_close, trigger_time, event_id, symbol, violations,
        )
        c15, r15, _ = _snapshot_tf_columns(
            "15m", i15, bar_end15, smc15, smc15_obctx,
            precomp_piv15, precomp_eq15, active15.get(i15, {"internal": [], "swing": []}),
            fifteen, trigger_close, trigger_time, event_id, symbol, violations,
        )
        c1h, r1h, _ = _snapshot_tf_columns(
            "1h", i1h, bar_end1h, smc1h, smc1h_obctx,
            precomp_piv1h, precomp_eq1h, active1h.get(i1h, {"internal": [], "swing": []}),
            one_hour, trigger_close, trigger_time, event_id, symbol, violations,
        )
        row.update(c5)
        row.update(c15)
        row.update(c1h)

        # Phase M audit: reconstructed active set vs state_timeline counts
        for tf_name, obctx, ii, snap in (
            ("5m", smc5_obctx, i5, active5.get(i5)),
            ("15m", smc15_obctx, i15, active15.get(i15)),
            ("1h", smc1h_obctx, i1h, active1h.get(i1h)),
        ):
            if ii is None or snap is None:
                continue
            st = state_at(obctx, ii)
            if len(snap["internal"]) != int(
                st["active_internal_ob_count"]
            ):
                recon_mismatches += 1
            if len(snap["swing"]) != int(st["active_swing_ob_count"]):
                recon_mismatches += 1

        # forward outcomes
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
            else:
                dret = bias * ret
                if bias == 1:
                    dmfe = lmfe
                    dmae = lmae
                else:
                    dmfe = smfe
                    dmae = smae

            row[f"fwd_dir_ret_h{h}"] = dret
            row[f"fwd_dir_mfe_h{h}"] = dmfe
            row[f"fwd_dir_mae_h{h}"] = dmae
            row[f"fwd_time_contiguous_h{h}"] = forward_contiguous(
                five, i5, h
            )

        snap_rows.append(row)
        level_rows.extend(r5)
        level_rows.extend(r15)
        level_rows.extend(r1h)

    snap_df = pd.DataFrame(snap_rows)
    lvl_df = pd.DataFrame(level_rows)

    # ---- per-symbol evidence (Phase P) ----
    same_bar_mit = int(snap_df["trigger_ob_mitigated_same_bar"].sum())
    swing_nonzero = {
        tf: float(
            (snap_df[f"{tf}_active_bear_swing_ob_count"]
             + snap_df[f"{tf}_active_bull_swing_ob_count"] > 0).mean()
        )
        for tf in ("5m", "15m", "1h")
    }
    pivot_cov = {}
    for tf in ("5m", "15m", "1h"):
        for typ in CURRENT_PIVOT_TYPES:
            col = f"{tf}_current_{typ}_level"
            pivot_cov[f"{tf}_{typ}"] = float(
                snap_df[col].notna().mean()
            )

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
        "same_bar_mitigated": same_bar_mit,
        "same_bar_mitigated_rate": (
            same_bar_mit / n_events if n_events else 0.0
        ),
        "swing_ob_context_nonzero_rate": swing_nonzero,
        "current_pivot_coverage": pivot_cov,
        "forward_usable_by_horizon": usable,
        "recon_mismatches": int(recon_mismatches),
        "stop_violations": list(violations),
    }

    if violations:
        raise RuntimeError(
            f"{symbol}: STOP conditions violated:\n"
            + "\n".join(violations[:20])
        )
    if recon_mismatches != 0:
        raise RuntimeError(
            f"{symbol}: active-OB reconstruction mismatches = "
            f"{recon_mismatches} (STOP)"
        )
    if len(snap_df) != n_events:
        raise RuntimeError(
            f"{symbol}: snapshot rows {len(snap_df)} != "
            f"OB_ENTERED events {n_events}"
        )

    print(
        f"  snapshots={len(snap_df)}  levels={len(lvl_df)}  "
        f"same_bar_mit={same_bar_mit}  "
        f"recon_mismatches={recon_mismatches}"
    )

    return snap_df, lvl_df, stats


# ============================================================
# Main
# ============================================================

def _describe_column(col: str) -> str:
    if col.startswith("trigger_ob_"):
        return "5m trigger OB attribute"
    if col.startswith("trigger_"):
        return "trigger identity / outcome"
    if col in ("event_id", "symbol", "trigger_time", "trigger_bar_index"):
        return "sample identity"
    for tf in ("5m", "15m", "1h"):
        p = f"{tf}_"
        if col.startswith(p):
            rest = col[len(p):]
            if rest in ("bar_index", "bar_end", "lag_minutes"):
                return f"{tf} snapshot bar"
            if rest.endswith("_bias"):
                return f"{tf} SMC trend bias"
            if rest.startswith("last_swing_structure"):
                return f"{tf} last swing BOS/CHoCH"
            if rest.startswith("last_internal_structure"):
                return f"{tf} last internal BOS/CHoCH"
            if "active" in rest and "ob_count" in rest:
                return f"{tf} active OB count"
            if rest.startswith("nearest_above"):
                return f"{tf} nearest above pressure"
            if rest.startswith("nearest_below"):
                return f"{tf} nearest below support"
            if rest in ("open", "high", "low", "close", "volume", "close_oi"):
                return f"{tf} bar OHLCV/OI"
            if rest.startswith("overlap"):
                return f"{tf} overlapping structure count/types"
            if rest.startswith("current_"):
                return f"{tf} current structural pivot"
    if col.startswith("fwd_"):
        return "forward outcome (entry = next 5m open)"
    return "misc"


def main() -> None:
    if OUT_LOCAL.exists() and any(OUT_LOCAL.iterdir()):
        raise RuntimeError(
            f"{OUT_LOCAL} exists and is non-empty. Delete only for an "
            "intentional pre-commit rerun."
        )

    OUT_LOCAL.mkdir(parents=True, exist_ok=True)
    OUT_GIT.mkdir(parents=True, exist_ok=True)

    snap_frames: list[pd.DataFrame] = []
    level_frames: list[pd.DataFrame] = []
    stats_all: list[dict] = []

    for symbol in SYMBOLS:
        snap_df, lvl_df, stats = process_symbol(symbol)
        snap_frames.append(snap_df)
        level_frames.append(lvl_df)
        stats_all.append(stats)

        # Git analysis data: per-symbol wide snapshots
        path = OUT_GIT / f"{symbol}_snapshots.csv"
        snap_df.to_csv(path, index=False)
        size_mb = path.stat().st_size / 1_000_000
        if size_mb > 20:
            raise RuntimeError(
                f"{path} is {size_mb:.1f}MB > 20MB; split before commit"
            )

    snapshots = pd.concat(snap_frames, ignore_index=True)
    levels = pd.concat(level_frames, ignore_index=True)

    snapshots.to_csv(OUT_LOCAL / "trigger_snapshots.csv", index=False)
    levels.to_csv(OUT_LOCAL / "trigger_levels.csv", index=False)

    total_events = sum(s["ob_entered_events"] for s in stats_all)
    total_rows = len(snapshots)
    if total_rows != total_events:
        raise RuntimeError(
            f"TOTAL snapshot rows {total_rows} != "
            f"total OB_ENTERED events {total_events}"
        )

    # ---- schema.json ----
    schema = {
        "snapshot_columns": list(snapshots.columns),
        "level_columns": list(levels.columns),
        "descriptions": {
            c: _describe_column(c) for c in snapshots.columns
        },
        "level_descriptions": {
            c: _describe_column(c) for c in levels.columns
        },
    }
    (OUT_GIT / "schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- summary.json ----
    summary = {
        "schema_version": "ob_trigger_smc_v1",
        "correction": "spatial semantics fixed vs 1727c171",
        "discovery_universe": SYMBOLS,
        "totals": {
            "ob_entered_events": int(total_events),
            "snapshot_rows": int(total_rows),
            "level_rows": int(len(levels)),
        },
        "by_symbol": stats_all,
    }
    (OUT_GIT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- manifest.json (local full data) ----
    manifest = {
        "status": "PASS",
        "schema_version": "ob_trigger_smc_v1",
        "correction_vs_1727c171": [
            "current structural pivots = latest confirmed per type "
            "(not all history)",
            "nearest above/below now includes active OBs",
            "active OBs from FULL-OB context (swing OB enabled) for "
            "spatial completeness; trigger universe unchanged",
            "per-TF bar OHLCV/OI written",
            "trigger-OB same-bar life/death recorded",
            "fwd_contig removed; fwd_time_contiguous added",
            "active-OB reconstruction replays canonical active-list "
            "semantics (0 mismatch vs state_timeline)",
        ],
        "sample_definition": (
            "one row per canonical 5m SMC ob_lifecycle_events "
            "type==OB_ENTERED; same-bar bullish internal + bullish "
            "swing OB => two events (keyed by internal/swing)"
        ),
        "aggregation": {
            "15m": "aggregate_15m from build_pytdx_panel (canonical)",
            "1h": (
                "aggregate_1h_from_15m: 15m bars bucketed by integer "
                "hour-epoch; no 'exactly 4 bars' rule"
            ),
            "smc_default": "compute_smc_momentum_bundle (swing OB off)",
            "smc_full_ob": (
                "compute_smc_pine with show_internal+swing OB on; "
                "trend/pivot semantics unchanged (Phase B invariant)"
            ),
        },
        "by_symbol": stats_all,
        "totals": {
            "ob_entered_events": int(total_events),
            "snapshot_rows": int(total_rows),
            "level_rows": int(len(levels)),
        },
    }
    (OUT_LOCAL / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ---- report committed analysis files ----
    committed = []
    for symbol in SYMBOLS:
        p = OUT_GIT / f"{symbol}_snapshots.csv"
        committed.append(
            {
                "file": str(p),
                "rows": int(len(snap_frames[SYMBOLS.index(symbol)])),
                "columns": int(snap_frames[SYMBOLS.index(symbol)].shape[1]),
                "size_mb": round(p.stat().st_size / 1_000_000, 2),
            }
        )
    for extra in ("schema.json", "summary.json"):
        p = OUT_GIT / extra
        committed.append(
            {"file": str(p), "rows": 0, "columns": 0,
             "size_mb": round(p.stat().st_size / 1_000_000, 2)}
        )

    print("\n" + "=" * 64)
    print("OB_TRIGGER_SNAPSHOT_V1_CORRECTED_BUILD_PASS")
    print("=" * 64)
    for s in stats_all:
        print(
            f"  {s['symbol']:3s} events={s['ob_entered_events']:5d} "
            f"snap={s['snapshot_rows']:5d} "
            f"levels={s['level_rows']:8d} "
            f"same_bar_mit={s['same_bar_mitigated']:4d} "
            f"recon={s['recon_mismatches']}"
        )
    print("\nCommitted analysis files:")
    for c in committed:
        print(
            f"  {c['file']}  rows={c['rows']} "
            f"cols={c['columns']} size={c['size_mb']}MB"
        )


if __name__ == "__main__":
    main()
