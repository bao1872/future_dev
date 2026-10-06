"""
structural_god_oracle_m15_v6_2_entry_target.py
==============================================

EXPERIMENTAL V6.2 label kernel -- ENTRY-TIME TARGET.

Why this exists
---------------
V6.1 discovers the FUTURE first distinct touched location B, uses B to decide
direction + target, and only afterwards optimizes Entry inside the A->B leg.
That is a causal violation: the target was not knowable at the moment of entry.

V6.2 fixes the root: the target is frozen from the geometry actually visible at
the ENTRY DECISION:

    target_snapshot_index == best_entry_decision_index

The future first distinct touched location is retained ONLY as the terminal
boundary of the ENTRY OPPORTUNITY LEG (conceptually `observed_leg_terminal`).
It is NOT the target and it does NOT decide direction.

Frozen pipeline
---------------
    true touch -> price Location -> frozen source A
    -> entry-opportunity interval [A_first_touch, observed_leg_terminal)
    -> for every legal decision d in that interval:
           entry_price = open[d+1]
           target      = first visible Location in geom_by_decision[d]
                         along the planned direction
           target frozen
           only THEN may future bars be inspected for the first target touch
    -> both LONG and SHORT branches are evaluated, each with its OWN
       entry-time frozen target
    -> God oracle keeps the greatest-utility completed branch

NOT changed here: SR math, Liquidity math, run_environment_m15 semantics,
execution boundary contract, Viewer, builder, production artifact.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from research.liquidity_oracle_atlas.structural_god_oracle_m15_v6_touch_chain import (
    EPS,
    LONG,
    SHORT,
    TARGET_TOUCH,
    ZoneTouch,
    LocationTouch,
    build_location_frames,
    intervals_overlap,
    location_overlap,
    representative_structure,
    representative_target,
    same_execution_unit,
    structure_id_from_match,
)

MATH_VERSION = "structural-god-oracle-v6.2-entry-time-target"


# ============================================================
# 1. Point-in-time geometry -> visible PRICE LOCATIONS
# ============================================================

def geometry_zones(geom: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten one geometry snapshot into zone dicts.

    geom[tf] = (channels, liq_up, liq_down, atr)
      channels : list of (top, bottom, strength)
      liq_*    : list of dicts with top / bottom / level / broken

    Broken liquidity zones are dropped (they are no longer tradeable structure).
    """
    out: List[Dict[str, Any]] = []
    if not geom:
        return out

    for tf, g in geom.items():
        if g is None:
            continue

        channels, liq_up, liq_down = g[0], g[1], g[2]

        for item in channels:
            top, bottom, strength = item[0], item[1], item[2]
            out.append({
                "tf": tf,
                "family": "SR",
                "side": None,
                "top": float(top),
                "bottom": float(bottom),
                "level": None,
                "strength": float(strength),
            })

        for side, levels in (("BUY", liq_up), ("SELL", liq_down)):
            for z in levels:
                if bool(z.get("broken")):
                    continue
                out.append({
                    "tf": tf,
                    "family": "LIQ",
                    "side": side,
                    "top": float(z["top"]),
                    "bottom": float(z["bottom"]),
                    "level": float(z["level"]),
                    "strength": None,
                })

    return out


def locations_from_geometry_snapshot(
    geom: Optional[Dict[str, Any]],
    eps: float = EPS,
) -> List[LocationTouch]:
    """Visible PRICE LOCATIONS at one decision time.

    Overlapping SR / Liquidity zones are ONE price location, not several
    targets. structure_id stays metadata only.
    """
    zones = geometry_zones(geom)
    if not zones:
        return []

    zt: List[ZoneTouch] = []
    for z in zones:
        b = min(z["bottom"], z["top"])
        t = max(z["bottom"], z["top"])
        zt.append(ZoneTouch(
            structure_id=structure_id_from_match(z),
            family=str(z["family"]),
            timeframe=str(z["tf"]),
            side=z["side"],
            bottom=float(b),
            top=float(t),
            level=(float(z["level"]) if z["level"] is not None else None),
        ))

    zt.sort(key=lambda x: (x.bottom, x.top))

    out: List[LocationTouch] = []
    members = [zt[0]]
    cb, ct = zt[0].bottom, zt[0].top

    for z in zt[1:]:
        if intervals_overlap(cb, ct, z.bottom, z.top, eps):
            members.append(z)
            cb = min(cb, z.bottom)
            ct = max(ct, z.top)
        else:
            out.append(LocationTouch(
                bar=-1,
                bottom=float(cb),
                top=float(ct),
                members=list(members),
            ))
            members = [z]
            cb, ct = z.bottom, z.top

    out.append(LocationTouch(
        bar=-1,
        bottom=float(cb),
        top=float(ct),
        members=list(members),
    ))

    return out


# ============================================================
# 2. Entry-time target selection
# ============================================================

def choose_entry_time_target(
    locations: List[LocationTouch],
    entry_price: float,
    direction: str,
    eps: float = EPS,
) -> Optional[Tuple[LocationTouch, float]]:
    """Nearest visible location along the planned direction.

    LONG  : location.bottom > entry_price ; target price = location.bottom
            (first-touch LOWER edge, entered from below)
    SHORT : location.top    < entry_price ; target price = location.top
            (first-touch UPPER edge, entered from above)

    Among eligible, the FIRST price the move would reach is chosen:
    LONG -> smallest bottom, SHORT -> largest top.
    """
    if direction == LONG:
        elig = [L for L in locations if L.bottom > float(entry_price) + eps]
        if not elig:
            return None
        t = min(elig, key=lambda L: L.bottom)
        return t, float(t.bottom)

    if direction == SHORT:
        elig = [L for L in locations if L.top < float(entry_price) - eps]
        if not elig:
            return None
        t = max(elig, key=lambda L: L.top)
        return t, float(t.top)

    raise AssertionError(f"unknown direction {direction!r}")


# ============================================================
# 3. Future evaluation (only AFTER the target is frozen)
# ============================================================

def first_target_touch(
    highs: np.ndarray,
    lows: np.ndarray,
    start_bar: int,
    target_price: float,
    direction: str,
    eps: float = EPS,
) -> Optional[int]:
    """First bar k >= start_bar where the frozen target is touched.

    LONG  : high[k] >= target_price
    SHORT : low[k]  <= target_price

    Same-fill-bar touch (k == start_bar) is legal: the fill happens at that
    bar's open.
    """
    if direction == LONG:
        for k in range(int(start_bar), len(highs)):
            if float(highs[k]) >= float(target_price) - eps:
                return int(k)
        return None

    if direction == SHORT:
        for k in range(int(start_bar), len(lows)):
            if float(lows[k]) <= float(target_price) + eps:
                return int(k)
        return None

    raise AssertionError(f"unknown direction {direction!r}")


# ============================================================
# 4. One (decision, direction) branch
# ============================================================

def evaluate_entry_branch(
    d: int,
    entry_price: float,
    direction: str,
    source: LocationTouch,
    geoms: List[Optional[Dict[str, Any]]],
    highs: np.ndarray,
    lows: np.ndarray,
    eps: float = EPS,
) -> Dict[str, Any]:
    """Evaluate one hypothetical (decision d, direction) branch.

    Target comes ONLY from geom_by_decision[d]. No later snapshot participates.
    """
    geom = geoms[d] if 0 <= d < len(geoms) else None
    locations = locations_from_geometry_snapshot(geom, eps)

    # A visible structure overlapping frozen source A cannot be its own target.
    locations = [
        L for L in locations
        if not intervals_overlap(
            L.bottom, L.top, source.bottom, source.top, eps,
        )
    ]

    chosen = choose_entry_time_target(locations, entry_price, direction, eps)
    if chosen is None:
        return {"ok": False, "reason": "no_visible_target"}

    target_location, target_price = chosen

    f = d + 1
    k = first_target_touch(highs, lows, f, target_price, direction, eps)
    if k is None:
        return {"ok": False, "reason": "target_never_touched"}

    utility = (
        float(target_price) - float(entry_price)
        if direction == LONG
        else float(entry_price) - float(target_price)
    )
    if utility <= eps:
        return {"ok": False, "reason": "non_positive_utility"}

    return {
        "ok": True,
        "reason": None,
        "direction": direction,
        "target_location": target_location,
        "target_price": float(target_price),
        "exit_bar": int(k),
        "utility": float(utility),
    }


# ============================================================
# 5. Sequential solver
# ============================================================

def solve_entry_time_target_oracle(
    frames: List[List[LocationTouch]],
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    segments: np.ndarray,
    trading_days: np.ndarray,
    geoms: List[Optional[Dict[str, Any]]],
    times: Optional[np.ndarray] = None,
    eps: float = EPS,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """V6.2 entry-time-target oracle.

    Labels from different sources MAY overlap in calendar time. This is
    REQUIRED, not an error: a label answers "what is the God-mode label for
    THIS candidate opportunity?", not "how would a single-position account
    sequence these trades?". Portfolio / one-position-at-a-time constraints
    belong to a later backtesting layer, never to oracle label generation.

    The structural location chain A -> B -> C owns the source sequence:
    after A's label is generated, the source advances to observed_leg_terminal
    B (the first bar with a location distinct from frozen A). It does NOT wait
    for A's trade to exit.
    """
    n = len(frames)
    opens = np.asarray(opens, dtype=float)
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    segments = np.asarray(segments, dtype=np.int64)
    trading_days = np.asarray(trading_days, dtype=np.int64)

    trades: List[Dict[str, Any]] = []

    audit = {
        "canonical_trades": 0,
        "long_count": 0,
        "short_count": 0,
        "no_visible_target_count": 0,
        "target_never_touched_count": 0,
        "direction_tie_count": 0,
        "entry_boundary_rejection_count": 0,
        "ambiguous_candidate_bars": 0,
        "sources_evaluated": 0,
        "sources_without_canonical": 0,
        "labels_with_overlapping_exit": 0,
        "target_snapshot_mismatch_count": 0,
    }

    def tstamp(i: int):
        if times is None:
            return None
        if i is None or i < 0 or i >= len(times):
            return None
        return times[i]

    i = 0
    prev_source = None
    while i < n:
        frame = frames[i]

        if not frame:
            i += 1
            continue

        # Candidate A is either an unambiguous single price location, OR a
        # multi-location bar that V6.1 resolves as (A retouch + exactly one
        # distinct B): the distinct B becomes the next source. Overlapping
        # retouches of the frozen previous source do NOT make the bar ambiguous.
        if len(frame) == 1:
            source = frame[0]
        else:
            if prev_source is None:
                # No frozen A to disambiguate against: genuinely ambiguous start.
                audit["ambiguous_candidate_bars"] += 1
                i += 1
                continue
            distinct = [x for x in frame if not location_overlap(prev_source, x)]
            if len(distinct) == 1:
                source = distinct[0]
            elif len(distinct) == 0:
                # Pure retouch of the frozen source: no new source here, keep
                # scanning without resetting the chain.
                i += 1
                continue
            else:
                audit["ambiguous_candidate_bars"] += 1
                i += 1
                continue

        audit["sources_evaluated"] += 1
        prev_source = source

        # ---- observed_leg_terminal: first later bar with a location that is
        #      NOT frozen A. This only CLOSES the entry-opportunity interval.
        terminal = None
        for b in range(i + 1, n):
            fb = frames[b]
            if any(not location_overlap(source, x) for x in fb):
                terminal = b
                break
        if terminal is None:
            terminal = n  # interval runs to the end of the data

        best = None  # (utility, d, branch)

        for d in range(i, terminal):
            f = d + 1
            if f >= n:
                break

            if not same_execution_unit(d, segments, trading_days):
                audit["entry_boundary_rejection_count"] += 1
                continue

            entry_price = float(opens[f])

            results = []
            for direction in (LONG, SHORT):
                br = evaluate_entry_branch(
                    d, entry_price, direction, source,
                    geoms, highs, lows, eps,
                )
                if br["ok"]:
                    results.append(br)
                elif br["reason"] == "no_visible_target":
                    audit["no_visible_target_count"] += 1
                elif br["reason"] == "target_never_touched":
                    audit["target_never_touched_count"] += 1

            if not results:
                continue

            if len(results) == 2 and abs(results[0]["utility"] - results[1]["utility"]) <= eps:
                # Same decision, both directions equally good: do NOT guess.
                audit["direction_tie_count"] += 1
                continue

            for br in results:
                cand = (br["utility"], int(d), br)
                if (
                    best is None
                    or cand[0] > best[0] + eps
                    or (abs(cand[0] - best[0]) <= eps and cand[1] < best[1])
                ):
                    best = cand

        if best is None:
            audit["sources_without_canonical"] += 1
            # advance beyond this failed source / leg
            i = terminal if terminal < n else n
            continue

        utility, d, br = best
        f = d + 1
        entry_price = float(opens[f])
        target_location = br["target_location"]
        target_price = float(br["target_price"])
        k = int(br["exit_bar"])
        direction = br["direction"]

        trades.append({
            "source_bar": int(i),
            "candidate_start_bar": int(i),
            "candidate_start_time": tstamp(i),
            "source_structure_id": representative_structure(source, direction),
            "zone_bottom": float(source.bottom),
            "zone_top": float(source.top),

            "oracle_direction": direction,

            "best_entry_decision_index": int(d),
            "best_entry_fill_index": int(f),
            "best_entry_fill_time": tstamp(f),
            "best_entry_price": float(entry_price),

            # CRITICAL INVARIANT: target frozen at the entry decision.
            "target_snapshot_index": int(d),
            "target_price": target_price,
            "target_structure_id": representative_target(
                target_location, direction,
            ),
            "target_location_bottom": float(target_location.bottom),
            "target_location_top": float(target_location.top),

            "exit_fill_index": k,
            "exit_fill_time": tstamp(k),
            "exit_price": float(target_price),
            "exit_reason": TARGET_TOUCH,

            "utility": float(utility),
        })

        audit["canonical_trades"] += 1
        if direction == LONG:
            audit["long_count"] += 1
        else:
            audit["short_count"] += 1

        # Advance the STRUCTURAL CHAIN to observed_leg_terminal B -- the first
        # bar with a location distinct from frozen source A. This does NOT wait
        # for A's trade to exit. Labels from different sources may therefore
        # overlap in calendar time (required, not an error).
        i = terminal if terminal < n else n

    # ---- overlap REPORTING (NOT a constraint) ----
    # Labels are permitted to overlap in calendar time. We only COUNT how many
    # labels' [entry, exit] intervals intersect another label's interval, so
    # the reviewer can see how prevalent overlap is. Overlap is correct
    # behavior; it is NOT a violation.
    m = len(trades)
    overlap_count = 0
    for a in range(m):
        ta = trades[a]
        a_lo = ta["best_entry_fill_index"]
        a_hi = ta["exit_fill_index"]
        for b in range(m):
            if b == a:
                continue
            tb = trades[b]
            if a_lo <= tb["exit_fill_index"] and tb["best_entry_fill_index"] <= a_hi:
                overlap_count += 1
                break
    audit["labels_with_overlapping_exit"] = overlap_count

    # ---- causal invariant (unchanged) ----
    for t in trades:
        if t["target_snapshot_index"] != t["best_entry_decision_index"]:
            audit["target_snapshot_mismatch_count"] += 1

    return trades, audit


def run_entry_time_target_oracle(symbol: str, max_bars=None):
    """Convenience runner: canonical environment -> V6.2 labels."""
    from research.liquidity_oracle_atlas.build_execution_environment_m15_v1 import (
        run_environment_m15,
    )

    env = run_environment_m15(symbol, max_bars, capture_provenance=True)

    ef = env["exec_frame"]
    entry_matches = env["entry_matches"]

    frames = build_location_frames(entry_matches)

    trades, audit = solve_entry_time_target_oracle(
        frames=frames,
        opens=ef["open"].to_numpy(float),
        highs=ef["high"].to_numpy(float),
        lows=ef["low"].to_numpy(float),
        segments=ef["segment"].to_numpy(np.int64),
        trading_days=ef["trading_day"].to_numpy(np.int64),
        geoms=env["geom_by_decision"],
        times=ef["bar_start_time"].to_numpy(),
    )

    return {
        "symbol": symbol,
        "math_version": MATH_VERSION,
        "trades": trades,
        "audit": audit,
        "frames": frames,
        "exec_frame": ef,
        "geom_by_decision": env["geom_by_decision"],
    }
