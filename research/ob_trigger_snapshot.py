#!/usr/bin/env python3
"""5m OB Trigger Multi-TF SMC Snapshot V2 (coverage + spatial fix).

Research sample = a canonical 5m SMC ``OB_ENTERED`` event, NOT a bar.

For every 5m OB_ENTERED event we freeze, at the trigger instant, the full
SMC environment that was *actually knowable then* (point-in-time, no
lookahead), across 5m / 15m / 1h.

V2 corrections over V1 (4474887):

  * P0: ``precompute_active_at`` now snapshots the faithful canonical
    active-OB list at EVERY requested bar (not only bars with a
    lifecycle event). No ``.get(bar, empty)`` false-green fallback;
    a missing non-None snapshot is a hard error. Phase M audit requires
    100% coverage and 0 count mismatches.
  * Strong trigger invariant: an alive (non-same-bar-mitigated) trigger
    OB MUST be present in the 5m active snapshot at close; a same-bar
    mitigated trigger MUST NOT be.
  * Trigger universe = internal + swing 5m OB_ENTERED from the FULL-OB
    context, after proving canonical internal universe parity (enabling
    swing OB does not change internal OB_ENTERED set).
  * ``distance_pct`` unified to percent (1.0 = 1%); ``trigger_ob_width_pct``
    already percent; ``trigger_ob_overlap_fraction`` stays [0,1].
  * primary nearest above/below/overlap = active OB + current pivots only;
    EQ stays descriptive (long table only), no longer competes for
    nearest pressure/support.
  * current pivots record ``_crossed`` / ``_crossed_age`` (broken by a
    BOS/CHoCH).
  * wide snapshot stores full nearest-zone info; overlap broken into
    active/bull/bear/internal/swing counts.
  * 1h bars carry ``component_15m_count`` (no partial-hour filtering yet).
  * Canonical raw endpoints vs derived spatial bounds (Amendment 1):
    lifecycle replay / identity / trigger universe / mitigation keep the
    RAW ``bar_low``/``bar_high`` exactly as written by canonical
    ``store_order_block``; research spatial geometry (relation_to_price,
    nearest, overlap, center, width, long-table ``zone_low``/``zone_high``)
    uses ``spatial_ob_bounds()`` = min/max(raw) envelope only. The long
    table carries BOTH the raw canonical endpoints and the derived zone.
  * Zone geometry is an EXPLANATORY invariant (Amendment 2), not a naive
    STOP: an inverted canonical OB (raw_low > raw_high) is legal only
    when explained by the canonical high-volatility swap
    (raw endpoints == that bar's high/low); otherwise hard STOP. Counts
    are reported at the UNIQUE-OB level (not per lifecycle event).
  * ``trigger_ob_*`` fields upgraded: ``trigger_ob_canonical_bar_low``/
    ``trigger_ob_canonical_bar_high`` (raw) + ``trigger_ob_zone_low``/
    ``trigger_ob_zone_high`` (derived, width/overlap based on zone) +
    ``trigger_ob_endpoints_swapped``. ``trigger_ob_spatial_overlap_fraction``
    is the derived spatial overlap, NOT the canonical OB_ENTERED test.
  * Output = V2 (local exports + Git analysis data, chunked to <=200 rows
    per CSV so the data is actually readable).

All SMC facts are CONSUMED from the canonical output of
``compute_smc_pine`` (via ``compute_smc_momentum_bundle`` for the default
context and directly for the full-OB context). Indicator semantics are
never redefined.

Active-OB reconstruction replays the canonical active-list semantics
exactly (creation -> cap-first then insert-front, mitigation removes by
identity; entered does NOT remove). Verified 0 mismatches vs
``state_timeline`` active counts with 100% snapshot coverage.
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
OUT_LOCAL = ROOT / "research" / "exports" / "ob_trigger_smc_v2"
OUT_GIT = ROOT / "research" / "analysis_data" / "ob_trigger_smc_v2"

# 5m / 15m / 30m / 60m / 120m expressed as a count of 5m bars.
HORIZONS_5M = [1, 3, 6, 12, 24]

ONE_HOUR_NS = 60 * 60 * 1_000_000_000
EXPECTED_BAR_NS = 5 * 60 * 1_000_000_000

CURRENT_PIVOT_TYPES = (
    "internal_high",
    "internal_low",
    "swing_high",
    "swing_low",
)


# ============================================================
# Cross-timeframe aggregation (thin helpers; no new semantics)
# ============================================================

def aggregate_1h_from_15m(fifteen: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 15m bars into 1h bars by integer hour-epoch bucket.

    Strictly replicates the repository's cross-TF authority: lower-TF
    bars are grouped by higher-period epoch bucket and OHLC / volume /
    OI aggregated. No "must be exactly 4 bars" rule added (the actual
    component count is recorded for later study).
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
            component_15m_count=("close", "size"),
        )
        .reset_index(drop=True)
        .sort_values("bar_start_time")
        .reset_index(drop=True)
    )


# ============================================================
# Canonical SMC builders
# ============================================================

def build_smc_tf(bars: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Canonical default SMC (swing OB display/calculation OFF)."""

    ind = bars.copy()
    if "volume" not in ind.columns:
        raise ValueError("SMC frame requires volume")
    ind = ind.set_index(pd.DatetimeIndex(ind["bar_start_time"]))
    bundle = compute_smc_momentum_bundle(ind)
    return ind, bundle.smc


def build_full_ob_smc_tf(bars: pd.DataFrame) -> dict:
    """Canonical SMC formulas, with swing OB display/calculation ON.

    This does NOT change trigger semantics. It is used only to expose the
    full internal + swing OB spatial context, and to own the trigger
    universe (internal + swing OB_ENTERED) after internal-parity is proven.
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


def _ob_identity(x: dict) -> tuple[int, int, int, bool]:
    return (
        int(x["anchor_index"]),
        int(x["confirmed_index"]),
        int(x["bias"]),
        bool(x["internal"]),
    )


def precompute_active_at(
    smc: dict,
    needed: set[int],
    cap: int = 100,
) -> dict[int, dict[str, list[dict]]]:
    """Faithful canonical active-OB snapshots at EVERY requested bar.

    Canonical semantics:
    - OB_CREATED: if len(target) >= 100: pop oldest; insert newest at front
    - OB_MITIGATED: remove matching OB
    - OB_ENTERED: does NOT change membership

    Critical invariant: every requested bar index receives a snapshot,
    even when no lifecycle event occurs on that exact bar.
    """

    wanted = sorted({int(i) for i in needed})
    if not wanted:
        return {}

    lifecycle: list[tuple[int, int, dict]] = []
    for seq, e in enumerate(smc.get("ob_lifecycle_events", [])):
        typ = e.get("type")
        if typ == "OB_CREATED":
            effective_i = int(e["confirmed_index"])
        elif typ == "OB_MITIGATED":
            effective_i = int(e["mitigated_index"])
        else:
            # OB_ENTERED does not alter active-list membership.
            continue
        lifecycle.append((effective_i, seq, e))
    lifecycle.sort(key=lambda x: (x[0], x[1]))

    active_internal: list[dict] = []
    active_swing: list[dict] = []
    cursor = 0
    out: dict[int, dict[str, list[dict]]] = {}

    for snapshot_i in wanted:
        while cursor < len(lifecycle) and lifecycle[cursor][0] <= snapshot_i:
            _, _, e = lifecycle[cursor]
            target = (
                active_internal if bool(e["internal"]) else active_swing
            )
            if e["type"] == "OB_CREATED":
                if len(target) >= cap:
                    target.pop()
                target.insert(
                    0,
                    {
                        "anchor_index": int(e["anchor_index"]),
                        "confirmed_index": int(e["confirmed_index"]),
                        "bias": int(e["bias"]),
                        "internal": bool(e["internal"]),
                        "bar_low": float(e["bar_low"]),
                        "bar_high": float(e["bar_high"]),
                    },
                )
            else:  # OB_MITIGATED
                key = _ob_identity(e)
                target[:] = [
                    x for x in target if _ob_identity(x) != key
                ]
            cursor += 1

        out[snapshot_i] = {
            "internal": [dict(x) for x in active_internal],
            "swing": [dict(x) for x in active_swing],
        }

    if set(out) != set(wanted):
        raise AssertionError(
            "active snapshot coverage mismatch: "
            f"{len(out)} != {len(wanted)}"
        )
    return out


def active_contains_trigger(active_snap: dict, ev: dict) -> bool:
    key = _ob_identity(ev)
    return any(
        _ob_identity(ob) == key
        for ob in (
            active_snap["internal"] + active_snap["swing"]
        )
    )


def audit_active_reconstruction(
    *,
    symbol: str,
    tf: str,
    smc_full: dict,
    active_map: dict[int, dict],
    needed: set[int],
) -> dict:
    expected = {int(i) for i in needed}
    actual = set(active_map.keys())
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise RuntimeError(
            f"{symbol} {tf}: active snapshot coverage failure "
            f"missing={missing[:20]} extra={extra[:20]}"
        )

    mismatches = []
    for i in sorted(expected):
        snap = active_map[i]
        st = state_at(smc_full, i)
        got_internal = len(snap["internal"])
        got_swing = len(snap["swing"])
        exp_internal = int(st["active_internal_ob_count"])
        exp_swing = int(st["active_swing_ob_count"])
        if got_internal != exp_internal or got_swing != exp_swing:
            mismatches.append(
                {
                    "bar_index": i,
                    "got_internal": got_internal,
                    "expected_internal": exp_internal,
                    "got_swing": got_swing,
                    "expected_swing": exp_swing,
                }
            )
    if mismatches:
        raise RuntimeError(
            f"{symbol} {tf}: active reconstruction mismatch "
            f"{mismatches[:10]}"
        )
    return {
        "requested": len(expected),
        "materialized": len(actual),
        "coverage_rate": 1.0,
        "mismatches": 0,
    }


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
    eq = sorted(
        smc.get("equal_highs_lows", []),
        key=lambda d: int(d["confirmed_index"]),
    )
    return {
        "eq": eq,
        "eq_conf": [int(d["confirmed_index"]) for d in eq],
    }


def current_pivots_at(precomp: dict, i: int) -> dict[str, dict | None]:
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


def pivot_cross_event_at(
    smc: dict,
    pivot: dict,
    pivot_type: str,
    snapshot_i: int,
) -> dict | None:
    """Earliest BOS/CHoCH event that broke this pivot before snapshot."""

    internal = pivot_type.startswith("internal_")
    expected_bias = 1 if pivot_type.endswith("_high") else -1
    candidates = []
    for ev in smc.get("events", []):
        if bool(ev["internal"]) != internal:
            continue
        if int(ev["bias"]) != expected_bias:
            continue
        if int(ev["anchor_index"]) != int(pivot["anchor_index"]):
            continue
        ci = int(ev["confirmed_index"])
        if ci > snapshot_i:
            continue
        if ci < int(pivot["confirmed_index"]):
            continue
        candidates.append(ev)
    if not candidates:
        return None
    return min(candidates, key=lambda e: int(e["confirmed_index"]))


def current_pivot_fields(
    precomp: dict,
    smc: dict,
    i: int,
    price: float,
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
                "_crossed",
                "_crossed_age",
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
        cross = pivot_cross_event_at(smc, p, typ, i)
        out[f"current_{typ}_crossed"] = cross is not None
        out[f"current_{typ}_crossed_age"] = (
            (i - int(cross["confirmed_index"])) if cross is not None else None
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
    if low > high:
        raise ValueError(f"invalid zone: low={low} > high={high}")
    if low <= price <= high:
        return ("overlap", 0.0)
    if low > price:
        return ("above", (low - price) / price * 100.0)
    return ("below", (price - high) / price * 100.0)


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
# Canonical raw endpoints vs derived spatial bounds (Amendment 1/2)
# ============================================================

def spatial_ob_bounds(
    raw_bar_low: float,
    raw_bar_high: float,
) -> tuple[float, float, bool]:
    """Derived spatial envelope ONLY.

    Does NOT redefine canonical OB lifecycle semantics. Canonical
    lifecycle (entered / mitigation) keeps using the raw endpoints
    exactly as written by ``store_order_block``. This helper only
    produces a research-side geometric envelope so we can place the
    OB in price space correctly, even when the canonical
    high-volatility swap inverted the stored endpoints (raw
    bar_low > raw bar_high for a high-vol anchor bar).

    Returns ``(zone_low, zone_high, endpoints_swapped)`` where
    ``zone_low <= zone_high`` always.
    """
    a = float(raw_bar_low)
    b = float(raw_bar_high)
    if not (np.isfinite(a) and np.isfinite(b)):
        raise RuntimeError("non-finite canonical OB endpoint")
    return (min(a, b), max(a, b), a > b)


def validate_canonical_ob_geometry(
    smc: dict,
    bars: pd.DataFrame,
    *,
    symbol: str,
    tf: str,
) -> dict:
    """Explanatory invariant (Amendment 2) replacing the naive STOP.

    An inverted canonical OB (raw_bar_low > raw_bar_high) is legal
    ONLY when it is explained by the canonical high-volatility swap
    in ``_compute_parsed_high_low``: at the OB anchor bar, the raw
    stored endpoints must equal ``(bar.high, bar.low)`` of that bar
    (float tolerance). Any inverted OB NOT explained this way is a
    real defect -> hard STOP.

    Counts are reported at the UNIQUE-OB level (a single OB can emit
    CREATED/ENTERED/MITIGATED lifecycle events that share endpoints,
    so event-level counts would triple-count the same OB).
    """
    unique_obs: dict[tuple, dict] = {}
    lifecycle_events = 0
    swapped_lc = 0
    for e in smc.get("ob_lifecycle_events", []):
        if e.get("type") not in {
            "OB_CREATED",
            "OB_ENTERED",
            "OB_MITIGATED",
        }:
            continue
        lifecycle_events += 1
        raw_low = float(e["bar_low"])
        raw_high = float(e["bar_high"])
        if not (np.isfinite(raw_low) and np.isfinite(raw_high)):
            raise RuntimeError(f"{symbol} {tf}: non-finite OB endpoint")
        if raw_low > raw_high:
            swapped_lc += 1
        key = (
            bool(e["internal"]),
            int(e["bias"]),
            int(e["anchor_index"]),
            int(e["confirmed_index"]),
        )
        unique_obs.setdefault(key, e)

    highs = bars["high"].to_numpy(float)
    lows = bars["low"].to_numpy(float)
    swapped_unique = 0
    unexplained: list[dict] = []
    for key, e in unique_obs.items():
        raw_low = float(e["bar_low"])
        raw_high = float(e["bar_high"])
        if raw_low <= raw_high:
            continue
        swapped_unique += 1
        j = int(e["anchor_index"])
        if j < 0 or j >= len(highs):
            unexplained.append(
                {"key": key, "reason": "anchor_index out of range"}
            )
            continue
        actual_high = float(highs[j])
        actual_low = float(lows[j])
        explained = np.isclose(raw_low, actual_high) and np.isclose(
            raw_high, actual_low
        )
        if not explained:
            unexplained.append(
                {
                    "key": key,
                    "raw_low": raw_low,
                    "raw_high": raw_high,
                    "anchor_high": actual_high,
                    "anchor_low": actual_low,
                }
            )

    if unexplained:
        raise RuntimeError(
            f"{symbol} {tf}: canonical inverted OB not explained by "
            f"parsed high-volatility swap: {unexplained[:10]}"
        )

    return {
        "unique_ob_count": len(unique_obs),
        "lifecycle_event_count": lifecycle_events,
        "swapped_unique_ob_count": swapped_unique,
        "swapped_lifecycle_event_count": swapped_lc,
        "unexplained_swapped_count": 0,
    }


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
        f"{prefix}nearest_above_structure_class",
        f"{prefix}nearest_above_bias",
        f"{prefix}nearest_above_age_bars",
        f"{prefix}nearest_above_zone_low",
        f"{prefix}nearest_above_zone_high",
        f"{prefix}nearest_above_object_price_center",
        f"{prefix}nearest_above_distance_pct",
        f"{prefix}nearest_below_type",
        f"{prefix}nearest_below_structure_class",
        f"{prefix}nearest_below_bias",
        f"{prefix}nearest_below_age_bars",
        f"{prefix}nearest_below_zone_low",
        f"{prefix}nearest_below_zone_high",
        f"{prefix}nearest_below_object_price_center",
        f"{prefix}nearest_below_distance_pct",
        f"{prefix}open",
        f"{prefix}high",
        f"{prefix}low",
        f"{prefix}close",
        f"{prefix}volume",
        f"{prefix}close_oi",
        f"{prefix}overlap_object_count",
        f"{prefix}overlap_active_ob_count",
        f"{prefix}overlap_bull_ob_count",
        f"{prefix}overlap_bear_ob_count",
        f"{prefix}overlap_internal_count",
        f"{prefix}overlap_swing_count",
        f"{prefix}overlap_types",
    ]
    for typ in CURRENT_PIVOT_TYPES:
        for suf in (
            "_level",
            "_age",
            "_relation",
            "_distance_pct",
            "_last_level",
            "_crossed",
            "_crossed_age",
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
    """Return (counts, level_rows, nearest_above, nearest_below, overlap).

    primary = active OB + current pivots (EQ is descriptive only).
    """

    current = current_pivots_at(precomp_piv_def, i)
    eq = confirmed_equal_levels_at(precomp_eq_def, i)

    current_rows: list[dict] = []
    for typ, p in current.items():
        if p is None:
            continue
        level = float(p["level"])
        rel, dist = relation_to_price(level, level, trigger_close)
        current_rows.append(
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
            violations.append(
                f"current pivot confirmed {int(p['confirmed_index'])} > {i}"
            )

    eq_rows: list[dict] = []
    for e in eq:
        level = float(e["level"])
        rel, dist = relation_to_price(level, level, trigger_close)
        eq_rows.append(
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
            violations.append(
                f"eq confirmed {int(e['confirmed_index'])} > {i}"
            )

    active = active_snap["internal"] + active_snap["swing"]
    ob_rows: list[dict] = []
    counts = {
        "bull_int": 0,
        "bear_int": 0,
        "bull_swg": 0,
        "bear_swg": 0,
    }
    for o in active:
        # RAW canonical endpoints are preserved on the active OB
        # (lifecycle identity / replay use them unchanged). Spatial
        # geometry uses the derived envelope only.
        raw_low = float(o["bar_low"])
        raw_high = float(o["bar_high"])
        zlow, zhigh, swapped = spatial_ob_bounds(raw_low, raw_high)
        rel, dist = relation_to_price(zlow, zhigh, trigger_close)
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
                "object_price_center": (zlow + zhigh) / 2.0,
                "source_index": int(o["anchor_index"]),
                "confirmed_index": int(o["confirmed_index"]),
                "age_bars": i - int(o["confirmed_index"]),
                "zone_low": zlow,
                "zone_high": zhigh,
                "canonical_bar_low": raw_low,
                "canonical_bar_high": raw_high,
                "canonical_endpoints_swapped": bool(swapped),
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

    primary_rows = ob_rows + current_rows
    primary_above = sorted(
        [r for r in primary_rows if r["relation"] == "above"],
        key=lambda r: r["distance_pct"],
    )
    primary_below = sorted(
        [r for r in primary_rows if r["relation"] == "below"],
        key=lambda r: r["distance_pct"],
    )
    primary_overlap = [r for r in primary_rows if r["relation"] == "overlap"]

    eq_selected = select_nearby_levels(eq_rows)
    level_rows = primary_rows + eq_selected

    return (
        counts,
        level_rows,
        (primary_above[0] if primary_above else None),
        (primary_below[0] if primary_below else None),
        primary_overlap,
    )


def _write_nearest(
    cols: dict, prefix: str, side: str, obj: dict | None
) -> None:
    base = f"{prefix}nearest_{side}_"
    fields = {
        "type": None,
        "structure_class": None,
        "bias": np.nan,
        "age_bars": np.nan,
        "zone_low": np.nan,
        "zone_high": np.nan,
        "object_price_center": np.nan,
        "distance_pct": np.nan,
    }
    if obj is not None:
        fields.update(
            {
                "type": obj["object_type"],
                "structure_class": obj["structure_class"],
                "bias": int(obj["bias"]),
                "age_bars": int(obj["age_bars"]),
                "zone_low": float(obj["zone_low"]),
                "zone_high": float(obj["zone_high"]),
                "object_price_center": float(obj["object_price_center"]),
                "distance_pct": float(obj["distance_pct"]),
            }
        )
    for k, v in fields.items():
        cols[f"{base}{k}"] = v


def _struct_fields(ev, i):
    if ev is None:
        return ("NONE", 0, float("nan"), float("nan"))
    return (
        str(ev["type"]),
        int(ev["bias"]),
        float(ev["level"]),
        float(i - int(ev["confirmed_index"])),
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
    cols: dict = {k: None for k in _tf_column_keys(prefix)}
    cols["1h_component_15m_count"] = None

    if i is None or bar_end is None:
        return cols, [], []

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

    _write_nearest(cols, prefix, "above", nearest_above)
    _write_nearest(cols, prefix, "below", nearest_below)

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
    if tf == "1h" and "component_15m_count" in bar.index:
        cols["1h_component_15m_count"] = int(bar["component_15m_count"])

    cols.update(
        {
            f"{prefix}{k}": v
            for k, v in current_pivot_fields(
                precomp_piv_def, smc_default, i, trigger_close
            ).items()
        }
    )

    cols[f"{prefix}overlap_object_count"] = len(overlap)
    cols[f"{prefix}overlap_active_ob_count"] = sum(
        r["object_type"].startswith("active_") for r in overlap
    )
    cols[f"{prefix}overlap_bull_ob_count"] = sum(
        (r["object_type"].startswith("active_") and int(r["bias"]) == 1)
        for r in overlap
    )
    cols[f"{prefix}overlap_bear_ob_count"] = sum(
        (r["object_type"].startswith("active_") and int(r["bias"]) == -1)
        for r in overlap
    )
    cols[f"{prefix}overlap_internal_count"] = sum(
        r["structure_class"] == "internal" for r in overlap
    )
    cols[f"{prefix}overlap_swing_count"] = sum(
        r["structure_class"] == "swing" for r in overlap
    )
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
                "canonical_bar_low": r.get("canonical_bar_low"),
                "canonical_bar_high": r.get("canonical_bar_high"),
                "canonical_endpoints_swapped": r.get(
                    "canonical_endpoints_swapped"
                ),
                "relation": r["relation"],
                "distance_pct": r["distance_pct"],
            }
        )

    return cols, level_records, overlap


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

    five_ind, smc5 = build_smc_tf(five)
    fifteen = aggregate_15m(five)
    fifteen_ind, smc15 = build_smc_tf(fifteen)
    one_hour = aggregate_1h_from_15m(fifteen)
    hour_ind, smc1h = build_smc_tf(one_hour)

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

    # Zone geometry: explanatory invariant (Amendment 2). Inverted
    # canonical OB endpoints are legal only when explained by the
    # canonical high-volatility swap; otherwise -> hard STOP. This
    # also reports unique-OB vs lifecycle-event swapped counts.
    geom: dict[str, dict] = {}
    for tf_name, full, bars in (
        ("5m", smc5_obctx, five),
        ("15m", smc15_obctx, fifteen),
        ("1h", smc1h_obctx, one_hour),
    ):
        geom[tf_name] = validate_canonical_ob_geometry(
            full, bars, symbol=symbol, tf=tf_name
        )
    print(
        "  Zone geometry: inverted OB all explained by high-vol swap "
        f"(swapped_uniq 5m={geom['5m']['swapped_unique_ob_count']} "
        f"15m={geom['15m']['swapped_unique_ob_count']} "
        f"1h={geom['1h']['swapped_unique_ob_count']})"
    )

    # Trigger universe = internal + swing 5m OB_ENTERED from FULL context,
    # after proving internal parity with default.
    def _entered_key(e: dict) -> tuple:
        return (
            bool(e["internal"]),
            int(e["bias"]),
            int(e["anchor_index"]),
            int(e["confirmed_index"]),
            int(e["enter_index"]),
            round(float(e["bar_low"]), 12),
            round(float(e["bar_high"]), 12),
        )

    default_internal = [
        _entered_key(e)
        for e in smc5.get("ob_lifecycle_events", [])
        if e.get("type") == "OB_ENTERED" and bool(e["internal"])
    ]
    full_internal = [
        _entered_key(e)
        for e in smc5_obctx.get("ob_lifecycle_events", [])
        if e.get("type") == "OB_ENTERED" and bool(e["internal"])
    ]
    if default_internal != full_internal:
        raise RuntimeError(
            f"{symbol}: enabling swing OB changed canonical internal "
            "OB_ENTERED universe (STOP)"
        )
    print(
        "  Trigger universe: FULL internal == DEFAULT internal (parity ok)"
    )

    events = [
        ev
        for ev in smc5_obctx.get("ob_lifecycle_events", [])
        if ev.get("type") == "OB_ENTERED"
    ]
    n_internal = sum(1 for e in events if bool(e["internal"]))
    n_swing = len(events) - n_internal
    n_events = len(events)
    print(
        f"  OB_ENTERED events: {n_events} "
        f"(internal={n_internal}, swing={n_swing})"
    )

    precomp_piv5 = _precomp_piv(smc5)
    precomp_piv15 = _precomp_piv(smc15)
    precomp_piv1h = _precomp_piv(smc1h)
    precomp_eq5 = _precomp_eq(smc5)
    precomp_eq15 = _precomp_eq(smc15)
    precomp_eq1h = _precomp_eq(smc1h)

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

    # Phase M: 100% coverage + 0 count mismatch, BEFORE generating rows.
    audits = {
        "5m": audit_active_reconstruction(
            symbol=symbol, tf="5m", smc_full=smc5_obctx,
            active_map=active5, needed=needed5,
        ),
        "15m": audit_active_reconstruction(
            symbol=symbol, tf="15m", smc_full=smc15_obctx,
            active_map=active15, needed=needed15,
        ),
        "1h": audit_active_reconstruction(
            symbol=symbol, tf="1h", smc_full=smc1h_obctx,
            active_map=active1h, needed=needed1h,
        ),
    }
    print(
        "  Phase M audit: "
        f"5m cov={audits['5m']['coverage_rate']} "
        f"mm={audits['5m']['mismatches']} | "
        f"15m cov={audits['15m']['coverage_rate']} "
        f"mm={audits['15m']['mismatches']} | "
        f"1h cov={audits['1h']['coverage_rate']} "
        f"mm={audits['1h']['mismatches']}"
    )

    open_ = five["open"].to_numpy(float)
    high = five["high"].to_numpy(float)
    low = five["low"].to_numpy(float)
    close = five["close"].to_numpy(float)
    outcome_vecs = {
        h: _outcome_vectors(open_, high, low, close, h)
        for h in HORIZONS_5M
    }

    violations: list[str] = []
    alive_missing = 0
    same_bar_still_active = 0
    trig_swapped = 0
    trig_normal = 0
    trig_swapped_sbm = 0
    trig_normal_sbm = 0
    snap_rows: list[dict] = []
    level_rows: list[dict] = []

    for ev, i5, trigger_time, i15, i1h in resolved:
        trigger_close = float(five.iloc[i5]["close"])
        trigger_bar_low = float(five.iloc[i5]["low"])
        trigger_bar_high = float(five.iloc[i5]["high"])

        bias = int(ev["bias"])
        internal = bool(ev["internal"])
        raw_bar_low = float(ev["bar_low"])
        raw_bar_high = float(ev["bar_high"])
        enter_index = int(ev["enter_index"])
        confirmed_index = int(ev["confirmed_index"])
        anchor_index = int(ev["anchor_index"])

        # Derived spatial envelope only; canonical raw endpoints kept.
        zlow, zhigh, swapped = spatial_ob_bounds(raw_bar_low, raw_bar_high)
        ob_width = zhigh - zlow
        overlap = max(
            0.0,
            min(trigger_bar_high, zhigh)
            - max(trigger_bar_low, zlow),
        )
        overlap_fraction = (
            overlap / ob_width if ob_width > 0 else float("nan")
        )
        width_pct = (
            (zhigh - zlow) / trigger_close * 100.0
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

        mit = find_trigger_mitigation(ev, smc5_obctx)
        mit_index = int(mit["mitigated_index"]) if mit is not None else None
        mit_same_bar = mit_index == i5
        alive_at_close = mit_index is None or mit_index > i5
        if mit_same_bar and alive_at_close:
            raise AssertionError("trigger OB lifecycle contradiction")

        # Swapped-trigger sanity counters (canonical high-vol subgroup).
        if swapped:
            trig_swapped += 1
            if mit_same_bar:
                trig_swapped_sbm += 1
        else:
            trig_normal += 1
            if mit_same_bar:
                trig_normal_sbm += 1

        active5_snap = active5[i5]
        active15_snap = (
            active15[i15] if i15 is not None else {"internal": [], "swing": []}
        )
        active1h_snap = (
            active1h[i1h] if i1h is not None else {"internal": [], "swing": []}
        )

        # Strong trigger invariant.
        trigger_active_at_close = active_contains_trigger(active5_snap, ev)
        if mit_same_bar:
            if trigger_active_at_close:
                same_bar_still_active += 1
                violations.append(
                    f"{event_id}: same-bar mitigated trigger "
                    "still active at close"
                )
        else:
            if not trigger_active_at_close:
                alive_missing += 1
                violations.append(
                    f"{event_id}: alive trigger missing from active 5m "
                    "snapshot"
                )

        row: dict = {
            "event_id": event_id,
            "symbol": symbol,
            "trigger_time": str(trigger_time),
            "trigger_bar_index": i5,
            "trigger_ob_bias": bias,
            "trigger_ob_structure": "internal" if internal else "swing",
            "trigger_ob_age_bars": enter_index - confirmed_index,
            "trigger_ob_canonical_bar_low": raw_bar_low,
            "trigger_ob_canonical_bar_high": raw_bar_high,
            "trigger_ob_zone_low": zlow,
            "trigger_ob_zone_high": zhigh,
            "trigger_ob_endpoints_swapped": bool(swapped),
            "trigger_ob_width_pct": width_pct,
            "trigger_ob_spatial_overlap_fraction": overlap_fraction,
            "trigger_ob_anchor_index": anchor_index,
            "trigger_ob_anchor_time": str(ev["anchor_time"]),
            "trigger_ob_confirmed_index": confirmed_index,
            "trigger_ob_confirmed_time": str(ev["confirmed_time"]),
            "trigger_ob_enter_index": enter_index,
            "trigger_ob_enter_time": str(ev["enter_time"]),
            "trigger_ob_mitigated_same_bar": bool(mit_same_bar),
            "trigger_ob_alive_at_close": bool(alive_at_close),
            "trigger_ob_present_in_active_snapshot": bool(
                trigger_active_at_close
            ),
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
            precomp_piv5, precomp_eq5, active5_snap,
            five, trigger_close, trigger_time, event_id, symbol, violations,
        )
        c15, r15, _ = _snapshot_tf_columns(
            "15m", i15, bar_end15, smc15, smc15_obctx,
            precomp_piv15, precomp_eq15, active15_snap,
            fifteen, trigger_close, trigger_time, event_id, symbol, violations,
        )
        c1h, r1h, _ = _snapshot_tf_columns(
            "1h", i1h, bar_end1h, smc1h, smc1h_obctx,
            precomp_piv1h, precomp_eq1h, active1h_snap,
            one_hour, trigger_close, trigger_time, event_id, symbol, violations,
        )
        row.update(c5)
        row.update(c15)
        row.update(c1h)

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

    # 1h component count distribution
    comp_dist = (
        one_hour["component_15m_count"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    comp_dist = {int(k): int(v) for k, v in comp_dist.items()}

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
            pivot_cov[f"{tf}_{typ}"] = float(snap_df[col].notna().mean())

    usable = {
        h: int(snap_df[f"fwd_raw_ret_h{h}"].notna().sum())
        for h in HORIZONS_5M
    }

    same_bar_mit = int(snap_df["trigger_ob_mitigated_same_bar"].sum())

    stats = {
        "symbol": symbol,
        "five_minute_bars": int(n),
        "fifteen_minute_bars": int(len(fifteen)),
        "one_hour_bars": int(len(one_hour)),
        "ob_entered_events": int(n_events),
        "trigger_internal_count": int(n_internal),
        "trigger_swing_count": int(n_swing),
        "snapshot_rows": int(len(snap_df)),
        "level_rows": int(len(lvl_df)),
        "same_bar_mitigated": same_bar_mit,
        "same_bar_mitigated_rate": (
            same_bar_mit / n_events if n_events else 0.0
        ),
        "alive_trigger_missing_from_active": int(alive_missing),
        "same_bar_still_active": int(same_bar_still_active),
        "swing_ob_context_nonzero_rate": swing_nonzero,
        "current_pivot_coverage": pivot_cov,
        "forward_usable_by_horizon": usable,
        "phase_m_audit": audits,
        "pit_violations": list(violations),
        "one_hour_component_15m_count_distribution": comp_dist,
        "canonical_ob_geometry": geom,
        "trigger_swapped_count": int(trig_swapped),
        "trigger_swapped_rate": (
            trig_swapped / n_events if n_events else 0.0
        ),
        "trigger_swapped_same_bar_mitigated_count": int(trig_swapped_sbm),
        "trigger_swapped_same_bar_mitigated_rate": (
            trig_swapped_sbm / trig_swapped if trig_swapped else 0.0
        ),
        "trigger_normal_same_bar_mitigated_rate": (
            trig_normal_sbm / trig_normal if trig_normal else 0.0
        ),
    }

    if violations:
        raise RuntimeError(
            f"{symbol}: STOP conditions violated:\n"
            + "\n".join(violations[:40])
        )
    if len(snap_df) != n_events:
        raise RuntimeError(
            f"{symbol}: snapshot rows {len(snap_df)} != "
            f"OB_ENTERED events {n_events}"
        )

    print(
        f"  snapshots={len(snap_df)}  levels={len(lvl_df)}  "
        f"same_bar_mit={same_bar_mit}  "
        f"alive_missing={alive_missing}  "
        f"same_bar_still_active={same_bar_still_active}"
    )

    return snap_df, lvl_df, stats


# ============================================================
# Chunked analysis-data writer
# ============================================================

def write_analysis_chunks(
    df: pd.DataFrame,
    symbol: str,
    root: Path,
    chunk_rows: int = 200,
) -> list[dict]:
    symbol_dir = root / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for start in range(0, len(df), chunk_rows):
        end = min(start + chunk_rows, len(df))
        chunk = df.iloc[start:end].copy()
        path = symbol_dir / (
            f"{symbol}_{start:04d}_{end - 1:04d}.csv"
        )
        chunk.to_csv(path, index=False)
        size = path.stat().st_size
        if size >= 900_000:
            raise RuntimeError(
                f"{path} = {size} bytes; analysis chunk too large"
            )
        records.append(
            {
                "file": str(path.relative_to(ROOT)),
                "row_start": start,
                "row_end": end - 1,
                "rows": len(chunk),
                "bytes": size,
            }
        )
    return records


# ============================================================
# Main
# ============================================================

def _describe_column(col: str) -> str:
    if col.startswith("trigger_ob_"):
        return "5m trigger OB attribute"
    if col in ("event_id", "symbol", "trigger_time", "trigger_bar_index"):
        return "sample identity"
    if col.startswith("fwd_"):
        return "forward outcome (entry = next 5m open)"
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
                return f"{tf} nearest above pressure (OB+current pivot)"
            if rest.startswith("nearest_below"):
                return f"{tf} nearest below support (OB+current pivot)"
            if rest in ("open", "high", "low", "close", "volume", "close_oi"):
                return f"{tf} bar OHLCV/OI"
            if rest.startswith("overlap"):
                return f"{tf} overlapping structure (OB+current pivot)"
            if rest.startswith("current_"):
                return f"{tf} current structural pivot"
            if rest == "component_15m_count":
                return "1h bar: number of constituent 15m bars"
    return "misc"


def main() -> None:
    if OUT_LOCAL.exists() and any(OUT_LOCAL.iterdir()):
        raise RuntimeError(
            f"{OUT_LOCAL} exists and is non-empty. Delete only for an "
            "intentional pre-commit rerun."
        )
    if OUT_GIT.exists() and any(OUT_GIT.iterdir()):
        raise RuntimeError(
            f"{OUT_GIT} exists and is non-empty. Delete only for an "
            "intentional pre-commit rerun."
        )

    OUT_LOCAL.mkdir(parents=True, exist_ok=True)
    OUT_GIT.mkdir(parents=True, exist_ok=True)

    snap_frames: list[pd.DataFrame] = []
    level_frames: list[pd.DataFrame] = []
    stats_all: list[dict] = []
    chunk_manifests: list[dict] = []

    for symbol in SYMBOLS:
        snap_df, lvl_df, stats = process_symbol(symbol)
        snap_frames.append(snap_df)
        level_frames.append(lvl_df)
        stats_all.append(stats)

        # Git analysis data: chunked per-symbol wide snapshots.
        chunks = write_analysis_chunks(snap_df, symbol, OUT_GIT, 200)
        chunk_manifests.append({"symbol": symbol, "chunks": chunks})

        # Local full exports (gitignored).
        (OUT_LOCAL / f"{symbol}_snapshots.csv").write_text(
            snap_df.to_csv(index=False), encoding="utf-8"
        )

    snapshots = pd.concat(snap_frames, ignore_index=True)
    levels = pd.concat(level_frames, ignore_index=True)

    (OUT_LOCAL / "trigger_levels.csv").write_text(
        levels.to_csv(index=False), encoding="utf-8"
    )

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
        "distance_pct_definition": "1.0 == 1% (percent)",
        "overlap_fraction_definition": (
            "trigger_ob_spatial_overlap_fraction in [0,1]; DERIVED "
            "spatial overlap using min/max envelope, NOT the canonical "
            "OB_ENTERED test"
        ),
        "raw_vs_spatial_statement": (
            "canonical_bar_low/bar_high preserve Source Owner lifecycle "
            "semantics (store_order_block raw endpoints); "
            "zone_low/zone_high = min/max(...) are derived "
            "spatial-analysis bounds only"
        ),
    }
    (OUT_GIT / "schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- summary.json ----
    total_alive_missing = sum(
        s["alive_trigger_missing_from_active"] for s in stats_all
    )
    total_same_bar_active = sum(
        s["same_bar_still_active"] for s in stats_all
    )
    evidence = {
        "default_internal_trigger_eq_full_internal_trigger": all(
            s["trigger_internal_count"] > 0 for s in stats_all
        ),
        "trigger_universe_internal_count": {
            s["symbol"]: s["trigger_internal_count"] for s in stats_all
        },
        "trigger_universe_swing_count": {
            s["symbol"]: s["trigger_swing_count"] for s in stats_all
        },
        "active_snapshot_requested": {
            s["symbol"]: {
                tf: s["phase_m_audit"][tf]["requested"]
                for tf in ("5m", "15m", "1h")
            }
            for s in stats_all
        },
        "active_snapshot_materialized": {
            s["symbol"]: {
                tf: s["phase_m_audit"][tf]["materialized"]
                for tf in ("5m", "15m", "1h")
            }
            for s in stats_all
        },
        "active_snapshot_coverage": {
            s["symbol"]: {
                tf: s["phase_m_audit"][tf]["coverage_rate"]
                for tf in ("5m", "15m", "1h")
            }
            for s in stats_all
        },
        "active_reconstruction_mismatch": {
            s["symbol"]: {
                tf: s["phase_m_audit"][tf]["mismatches"]
                for tf in ("5m", "15m", "1h")
            }
            for s in stats_all
        },
        "alive_trigger_missing_from_active_snapshot": total_alive_missing,
        "same_bar_mitigated_trigger_still_active": total_same_bar_active,
        "pit_violations": sum(len(s["pit_violations"]) for s in stats_all),
        "canonical_swapped_unique_ob_by_symbol_tf": {
            s["symbol"]: {
                tf: s["canonical_ob_geometry"][tf]["swapped_unique_ob_count"]
                for tf in ("5m", "15m", "1h")
            }
            for s in stats_all
        },
        "canonical_swapped_lifecycle_event_by_symbol_tf": {
            s["symbol"]: {
                tf: s["canonical_ob_geometry"][tf][
                    "swapped_lifecycle_event_count"
                ]
                for tf in ("5m", "15m", "1h")
            }
            for s in stats_all
        },
        "unexplained_swapped_ob": 0,
        "one_hour_component_15m_count_distribution": {
            s["symbol"]: s["one_hour_component_15m_count_distribution"]
            for s in stats_all
        },
        "trigger_swapped_count": {
            s["symbol"]: s["trigger_swapped_count"] for s in stats_all
        },
        "trigger_swapped_rate": {
            s["symbol"]: s["trigger_swapped_rate"] for s in stats_all
        },
        "trigger_swapped_same_bar_mitigated_rate": {
            s["symbol"]: s["trigger_swapped_same_bar_mitigated_rate"]
            for s in stats_all
        },
        "trigger_normal_same_bar_mitigated_rate": {
            s["symbol"]: s["trigger_normal_same_bar_mitigated_rate"]
            for s in stats_all
        },
        "distance_pct_definition": "1.0 == 1%",
        "raw_vs_spatial_statement": (
            "canonical_bar_low/bar_high preserve Source Owner lifecycle "
            "semantics; zone_low/zone_high=min/max(...) are derived "
            "spatial-analysis bounds only"
        ),
        "chunk_gt_900kb": 0,
    }

    summary = {
        "schema_version": "ob_trigger_smc_v2",
        "correction": "V2: active-OB coverage + spatial semantics fix",
        "discovery_universe": SYMBOLS,
        "totals": {
            "ob_entered_events": int(total_events),
            "snapshot_rows": int(total_rows),
            "level_rows": int(len(levels)),
        },
        "by_symbol": stats_all,
        "chunk_manifests": chunk_manifests,
        "evidence": evidence,
    }
    (OUT_GIT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ---- manifest.json (local full data) ----
    manifest = {
        "status": "PASS",
        "schema_version": "ob_trigger_smc_v2",
        "correction_vs_4474887": [
            "P0: precompute_active_at snapshots EVERY requested bar "
            "(not only lifecycle-event bars); no .get(empty) fallback; "
            "Phase M audit requires 100% coverage + 0 mismatch",
            "strong trigger invariant: alive trigger OB must be present "
            "in 5m active snapshot at close; same-bar mitigated must not",
            "trigger universe = internal + swing 5m OB_ENTERED from "
            "FULL-OB context (internal parity proven)",
            "distance_pct unified to percent (1.0 = 1%); "
            "trigger_ob_spatial_overlap_fraction stays [0,1] (derived)",
            "primary nearest/overlap = active OB + current pivots; "
            "EQ descriptive only (long table), no longer competes",
            "current pivots record _crossed / _crossed_age",
            "wide snapshot stores full nearest-zone info; overlap split "
            "into active/bull/bear/internal/swing counts",
            "1h bars carry component_15m_count (no partial-hour filter)",
            "Amendment 1: canonical RAW bar_low/bar_high preserved on "
            "lifecycle replay/identity/trigger; research spatial geometry "
            "uses spatial_ob_bounds()=min/max envelope only; long table "
            "keeps both raw + derived zone",
            "Amendment 2: zone geometry is an EXPLANATORY invariant; "
            "inverted OB legal only if explained by canonical high-vol "
            "swap (raw endpoints == anchor bar high/low), else hard STOP; "
            "counts at UNIQUE-OB level",
            "trigger_ob_* upgraded: canonical raw + spatial zone + "
            "endpoints_swapped; width/overlap based on derived zone",
            "output V2; Git data chunked to <=200 rows per CSV",
        ],
        "sample_definition": (
            "one row per canonical 5m SMC ob_lifecycle_events "
            "type==OB_ENTERED (internal OR swing); keyed by "
            "internal/swing"
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
        "evidence": evidence,
    }
    (OUT_LOCAL / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 64)
    print("OB_TRIGGER_SNAPSHOT_V2_BUILD_PASS")
    print("=" * 64)
    for s in stats_all:
        print(
            f"  {s['symbol']:3s} events={s['ob_entered_events']:5d} "
            f"(I={s['trigger_internal_count']} S={s['trigger_swing_count']}) "
            f"snap={s['snapshot_rows']:5d} "
            f"levels={s['level_rows']:8d} "
            f"same_bar_mit={s['same_bar_mitigated']:4d} "
            f"alive_missing={s['alive_trigger_missing_from_active']} "
            f"same_bar_active={s['same_bar_still_active']}"
        )
    print("\nEvidence:")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
