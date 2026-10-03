from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


EPS = 1e-6

LONG = "LONG"
SHORT = "SHORT"
TARGET_TOUCH = "TARGET_TOUCH"


# ============================================================
# DATA
# ============================================================

@dataclass(frozen=True)
class ZoneTouch:
    structure_id: str
    family: str
    timeframe: str
    side: Optional[str]

    bottom: float
    top: float

    level: Optional[float] = None


@dataclass
class LocationTouch:
    """
    One PRICE LOCATION touched on one bar.

    Several overlapping SR / liquidity zones on the same bar
    are one LocationTouch.

    Two disjoint zones on the same bar remain two LocationTouch objects.
    """
    bar: int

    bottom: float
    top: float

    members: List[ZoneTouch]


# ============================================================
# STRUCTURE ID — metadata only, never semantics
# ============================================================

def structure_id_from_match(m: Dict[str, Any]) -> str:

    tf = str(m["tf"])
    fam = str(m["family"])

    bottom = float(m["bottom"])
    top = float(m["top"])

    if fam == "SR":
        return f"SR|{tf}|{top:.6f}|{bottom:.6f}"

    side = str(m.get("side"))
    level = float(m["level"])

    return (
        f"LIQ|{tf}|{side}|{level:.6f}|"
        f"{top:.6f}|{bottom:.6f}"
    )


# ============================================================
# BASIC LOCATION MATH
# ============================================================

def intervals_overlap(
    a_bottom: float,
    a_top: float,
    b_bottom: float,
    b_top: float,
    eps: float = EPS,
) -> bool:

    return not (
        b_top < a_bottom - eps
        or
        b_bottom > a_top + eps
    )


def location_overlap(
    a: LocationTouch,
    b: LocationTouch,
    eps: float = EPS,
) -> bool:

    return intervals_overlap(
        a.bottom,
        a.top,
        b.bottom,
        b.top,
        eps,
    )


# ============================================================
# STEP 1
# One bar -> true touched PRICE LOCATIONS
# ============================================================

def locations_from_bar_matches(
    bar: int,
    matches: List[Dict[str, Any]],
    eps: float = EPS,
) -> List[LocationTouch]:
    """
    Convert true-touch structures on ONE bar into spatial locations.

    Overlapping zones are merged into one contiguous location.
    Disjoint zones stay separate.

    IMPORTANT:
    This grouping is PRICE based, not structure_id based.
    """

    zones: List[ZoneTouch] = []

    for m in matches:

        z = ZoneTouch(
            structure_id=structure_id_from_match(m),
            family=str(m["family"]),
            timeframe=str(m["tf"]),
            side=m.get("side"),
            bottom=float(m["bottom"]),
            top=float(m["top"]),
            level=(
                float(m["level"])
                if m.get("level") is not None
                else None
            ),
        )

        zones.append(z)

    if not zones:
        return []

    zones.sort(
        key=lambda z: (z.bottom, z.top)
    )

    out: List[LocationTouch] = []

    current_members = [zones[0]]
    current_bottom = zones[0].bottom
    current_top = zones[0].top

    for z in zones[1:]:

        if intervals_overlap(
            current_bottom,
            current_top,
            z.bottom,
            z.top,
            eps,
        ):
            # Same contiguous price location ON THIS BAR.
            current_members.append(z)
            current_bottom = min(current_bottom, z.bottom)
            current_top = max(current_top, z.top)

        else:
            out.append(
                LocationTouch(
                    bar=bar,
                    bottom=float(current_bottom),
                    top=float(current_top),
                    members=list(current_members),
                )
            )

            current_members = [z]
            current_bottom = z.bottom
            current_top = z.top

    out.append(
        LocationTouch(
            bar=bar,
            bottom=float(current_bottom),
            top=float(current_top),
            members=list(current_members),
        )
    )

    return out


def build_location_frames(
    entry_matches: List[List[Dict[str, Any]]],
) -> List[List[LocationTouch]]:

    return [
        locations_from_bar_matches(bar, matches)
        for bar, matches in enumerate(entry_matches)
    ]


# ============================================================
# STEP 2
# Classify FIRST new location relative to frozen A
# ============================================================

def classify_new_locations(
    source: LocationTouch,
    locations: List[LocationTouch],
    eps: float = EPS,
):
    """
    `locations` are all location-distinct touches on the FIRST
    bar that leaves frozen source A.

    With 15m OHLC we know all were touched during this bar,
    but we do NOT know their intrabar ordering.

    Therefore exactly one distinct location is required for
    a canonical next-touch label.
    """

    if len(locations) == 0:
        raise AssertionError(
            "classify_new_locations requires at least one distinct location"
        )

    if len(locations) > 1:
        return "AMBIGUOUS", None, None

    target = locations[0]

    if target.bottom > source.top + eps:
        return (
            LONG,
            target,
            float(target.bottom),
        )

    if target.top < source.bottom - eps:
        return (
            SHORT,
            target,
            float(target.top),
        )

    raise AssertionError(
        "distinct target unexpectedly overlaps frozen source"
    )


# ============================================================
# STEP 3
# Execution legality
# ============================================================

def same_execution_unit(
    d: int,
    segments: np.ndarray,
    trading_days: np.ndarray,
) -> bool:

    if d + 1 >= len(segments):
        return False

    return (
        int(segments[d]) == int(segments[d + 1])
        and
        int(trading_days[d]) == int(trading_days[d + 1])
    )


def best_entry(
    contact_bars: List[int],
    opens: np.ndarray,
    segments: np.ndarray,
    trading_days: np.ndarray,
    target_bar: int,
    target_price: float,
    direction: str,
    eps: float = EPS,
) -> Optional[Tuple[int, int, float]]:
    """
    Optimize ONLY Entry.

    LONG  -> minimum next open
    SHORT -> maximum next open

    Decision must be an A-touch bar.
    Fill = open[d+1].
    """

    best = None

    for d in sorted(set(contact_bars)):

        if d >= target_bar:
            continue

        if d + 1 >= len(opens):
            continue

        if not same_execution_unit(
            d,
            segments,
            trading_days,
        ):
            continue

        fill = float(opens[d + 1])

        if direction == LONG:

            if fill >= target_price - eps:
                continue

            candidate = (
                fill,
                d,
                d + 1,
            )

            if (
                best is None
                or candidate[0] < best[0] - eps
                or (
                    abs(candidate[0] - best[0]) <= eps
                    and candidate[1] < best[1]
                )
            ):
                best = candidate

        else:

            if fill <= target_price + eps:
                continue

            candidate = (
                fill,
                d,
                d + 1,
            )

            if (
                best is None
                or candidate[0] > best[0] + eps
                or (
                    abs(candidate[0] - best[0]) <= eps
                    and candidate[1] < best[1]
                )
            ):
                best = candidate

    if best is None:
        return None

    fill_price, decision, fill_bar = best

    return (
        int(decision),
        int(fill_bar),
        float(fill_price),
    )


# ============================================================
# Metadata representative only
# ============================================================

def representative_structure(
    location: LocationTouch,
    direction: str,
) -> str:
    """
    structure_id is only metadata for Viewer.

    LONG source:
        highest member top

    SHORT source:
        lowest member bottom
    """

    if direction == LONG:

        z = max(
            location.members,
            key=lambda x: x.top,
        )

    else:

        z = min(
            location.members,
            key=lambda x: x.bottom,
        )

    return z.structure_id


def representative_target(
    location: LocationTouch,
    direction: str,
) -> str:

    if direction == LONG:

        z = min(
            location.members,
            key=lambda x: x.bottom,
        )

    else:

        z = max(
            location.members,
            key=lambda x: x.top,
        )

    return z.structure_id


# ============================================================
# STEP 4
# Entire touch chain
# ============================================================

def solve_location_touch_chain(
    frames: List[List[LocationTouch]],
    opens: np.ndarray,
    segments: np.ndarray,
    trading_days: np.ndarray,
    times: Optional[np.ndarray] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    FIRST-PRINCIPLES ORACLE.

    Frozen source A never expands.

    Repeated touches that overlap frozen A:
        same source opportunity
        -> add legal contact bar only

    First later non-overlapping location:
        B = fixed target
        direction fixed by B vs A
        exit fixed at B first-touch bar

    After A->B:
        B becomes the new frozen A.

    NO structure-id ownership.
    NO episode owner.
    NO global DP.
    NO greedy target search.
    """

    n = len(frames)

    trades: List[Dict[str, Any]] = []

    audit = {
        "bars_with_true_touch": 0,
        "location_touches": 0,
        "same_location_retouch_bars": 0,
        "target_transitions": 0,
        "ambiguous_target_bars": 0,
        "no_legal_entry": 0,
        "canonical_trades": 0,
    }

    for frame in frames:
        if frame:
            audit["bars_with_true_touch"] += 1
            audit["location_touches"] += len(frame)

    def tstamp(i: int):
        if times is None:
            return None
        if i < 0 or i >= len(times):
            return None
        return times[i]

    # --------------------------------------------------------
    # Initialize source:
    # only a single-location bar can establish an unambiguous
    # starting A at the left boundary of the dataset.
    # --------------------------------------------------------

    source: Optional[LocationTouch] = None
    source_contacts: List[int] = []

    bar = 0

    while bar < n:

        frame = frames[bar]

        # ----------------------------------------------------
        # No active source yet.
        # ----------------------------------------------------

        if source is None:

            if len(frame) == 1:

                source = frame[0]

                # IMPORTANT:
                # source zone is now FROZEN.
                source_contacts = [bar]

            elif len(frame) > 1:

                # Dataset-boundary ambiguity only.
                # Do not manufacture intrabar ordering.
                audit["ambiguous_target_bars"] += 1

            bar += 1
            continue

        # ----------------------------------------------------
        # Active frozen source A.
        # Compare EVERY later touch to frozen A only.
        # Never expand A.
        # ----------------------------------------------------

        same_locations = [
            x for x in frame
            if location_overlap(source, x)
        ]

        distinct_locations = [
            x for x in frame
            if not location_overlap(source, x)
        ]

        # No new location yet.
        if not distinct_locations:

            if same_locations:

                audit["same_location_retouch_bars"] += 1
                source_contacts.append(bar)

            bar += 1
            continue

        # ----------------------------------------------------
        # FIRST distinct location bar = fixed target event.
        # ----------------------------------------------------

        audit["target_transitions"] += 1

        direction, target, target_price = (
            classify_new_locations(
                source,
                distinct_locations,
            )
        )

        if direction == "AMBIGUOUS":

            audit["ambiguous_target_bars"] += 1

            # We do not know which location came first inside
            # this 15m bar. End this chain segment.
            source = None
            source_contacts = []

            bar += 1
            continue

        assert target is not None
        assert target_price is not None

        entry = best_entry(
            source_contacts,
            opens,
            segments,
            trading_days,
            target_bar=bar,
            target_price=float(target_price),
            direction=direction,
        )

        if entry is None:

            audit["no_legal_entry"] += 1

        else:

            d, f, entry_price = entry

            utility = (
                float(target_price) - entry_price
                if direction == LONG
                else
                entry_price - float(target_price)
            )

            if utility > EPS:

                source_sid = representative_structure(
                    source,
                    direction,
                )

                target_sid = representative_target(
                    target,
                    direction,
                )

                trades.append({
                    "structure_id": source_sid,

                    "candidate_start_bar": int(source.bar),
                    "candidate_start_time": tstamp(source.bar),

                    # LOCATION zone, not one arbitrary SR zone.
                    "zone_bottom": float(source.bottom),
                    "zone_top": float(source.top),

                    "oracle_direction": direction,

                    "best_entry_decision_index": int(d),
                    "best_entry_fill_index": int(f),
                    "best_entry_fill_time": tstamp(f),
                    "best_entry_price": float(entry_price),

                    "target_structure_id": target_sid,
                    "target_price": float(target_price),

                    "exit_fill_index": int(bar),
                    "exit_fill_time": tstamp(bar),
                    "exit_price": float(target_price),

                    "exit_reason": TARGET_TOUCH,

                    "utility": float(utility),

                    "event_id": None,
                    "canonical_oracle_trade": True,
                })

                audit["canonical_trades"] += 1

        # ----------------------------------------------------
        # B becomes next A.
        # Exactly one transition A -> B has been consumed.
        # ----------------------------------------------------

        source = target
        source_contacts = [bar]

        bar += 1

    return trades, audit


# ============================================================
# Production owner
# ============================================================

from research.liquidity_oracle_atlas.build_execution_environment_m15_v1 import (
    run_environment_m15,
)


def run_touch_chain_oracle(symbol: str, max_bars=None):

    env = run_environment_m15(
        symbol,
        max_bars,
        capture_provenance=True,
    )

    ef = env["exec_frame"]
    entry_matches = env["entry_matches"]

    frames = build_location_frames(
        entry_matches
    )

    trades, audit = solve_location_touch_chain(
        frames=frames,
        opens=ef["open"].to_numpy(float),
        segments=ef["segment"].to_numpy(np.int64),
        trading_days=ef["trading_day"].to_numpy(np.int64),
        times=ef["bar_start_time"].to_numpy(),
    )

    return {
        "symbol": symbol,
        "math_version": "structural-god-oracle-v6.1-location-chain",
        "trades": trades,
        "audit": audit,
        "frames": frames,
        "exec_frame": ef,
        "geom_by_decision": env["geom_by_decision"],
    }


# ============================================================
# Interface-compatibility stubs (artifact builder import only)
# ============================================================
# NOTE: the V6.1 location-chain core is driven entirely by
# `solve_location_touch_chain` / `run_touch_chain_oracle`. The two names
# below exist ONLY so the (untouched this round) artifact builder's import
# surface stays resolvable. They are NOT part of the production call path
# and deliberately carry no per-group screenshot logic.

MATH_VERSION = "structural-god-oracle-v6.1-location-chain"


def print_screenshot_audit(res, around_price=None, band=None):
    """Interface-compatibility stub for the artifact builder import.

    The retired per-group screenshot audit is intentionally not re-added;
    the location-chain model has no per-group source.
    """
    audit = res.get("audit", {})
    print("[screenshot-audit] location-chain audit:")
    for k in (
        "bars_with_true_touch", "location_touches", "same_location_retouch_bars",
        "target_transitions", "ambiguous_target_bars", "no_legal_entry",
        "canonical_trades",
    ):
        print(f"  {k}: {audit.get(k)}")
