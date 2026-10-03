"""structural_god_oracle_m15_v6_touch_chain
============================================

FUT-M15-STRUCTURAL-GOD-ORACLE-V6-TOUCH-CHAIN

Touch-Chain God Oracle -- a NEW, frozen model that REPLACES the V4 greedy
candidate / V5 global-profit-DP trade-sequence logic.

FUNDAMENTAL OBJECT (frozen)
---------------------------
The fundamental object is NOT a greedy candidate and NOT a global-profit DP.

    one structural touch episode
        ->
    the next distinct structural touch episode

Every adjacent pair defines exactly one trade label. If the market touches

    A -> B -> C -> D

then labels are

    A->B
    B->C
    C->D

Therefore the expected number of labels should be approximately

    number_of_distinct_touch_episodes - 1

with only a small reduction from ambiguity / execution-boundary / no-entry
cases. The label count thus stays close to the number of structure touches
the user actually sees on the chart, instead of collapsing to a handful.

WHY THIS IS CORRECT (and the earlier models were not)
-----------------------------------------------------
The earlier models asked "from this candidate, which future Target yields
more profit?" and froze a single Target shared across many future contact
bars, choosing direction by eventual profit. That produced absurd labels
(e.g. a 10-day SHORT to a level far below).

This model asks the *inverse* question:

    "From this Entry, which structure does the market touch NEXT?"

The next ACTUAL structural touch B already defines the direction (B above A
=> LONG, B below A => SHORT) and the Exit (first touch of B). Only the Entry
is optimized (God-mode best fill within A's own contact bars).

CANONICAL TRUE-TOUCH OWNER (reused, not rewritten)
--------------------------------------------------
Structural touches are detected with the canonical R4 15m true-touch owner
``touch_from_prev_geometry_r4`` (in build_execution_environment_m15_v1.py),
which judges a bar against the geometry known at the PREVIOUS 15m close using
``bar_hits_zone`` (range intersects zone, delta = 0). This is TRUE touch, not
0.5-ATR proximity. ``entry_touch_from_prev_geometry`` is the equivalent 5m
owner; both share the identical bar_hits_zone contract. We reuse the m15 owner
because this oracle lives on the 15m execution axis (same axis as V4 / Viewer /
artifact).

NO GLOBAL SCHEDULER
-------------------
No weighted-interval DP, no greedy cursor. The chain itself sequences the
trades: Exit of trade i is the touch that begins the next structural
transition. Trades are naturally non-overlapping.

This module is PRODUCTION-READY research code. V4 / V5 are left intact for
audit history. This module does NOT regenerate the artifact and does NOT
modify the Viewer (those are later integration steps, out of scope here).

Authoring note: implement V6 as a NEW module. Keep V4/V5 intact.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from research.liquidity_oracle_atlas.build_execution_environment_m15_v1 import (
    run_environment_m15,
)

MATH_VERSION = "structural-god-oracle-v6.0-touch-chain"
TASK_ID = "FUT-M15-STRUCTURAL-GOD-ORACLE-V6-TOUCH-CHAIN"
EPS = 1e-6  # zone-ordering slack for direction classification

ORACLE_LONG = "LONG"
ORACLE_SHORT = "SHORT"
R_TARGET = "TARGET_TOUCH"


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
class Touch(NamedTuple):
    """One true-touch record: a bar touched one structure."""

    bar: int
    structure_id: str
    family: str          # "SR" | "LIQ"
    tf: str
    side: Optional[str]  # LIQ: "BUY" | "SELL"; SR: None
    zone_bottom: float
    zone_top: float
    level: Optional[float]
    strength: Optional[float]


class Episode(NamedTuple):
    """A maximal consecutive run of bars touching one structure id."""

    structure_id: str
    touched_bars: List[int]
    start_bar: int
    end_bar: int
    zone_bottom: float
    zone_top: float
    family: str
    tf: str
    side: Optional[str]


class TouchGroup(NamedTuple):
    """All structure episodes whose episode STARTS at the same bar."""

    start_bar: int
    episodes: List[Episode]
    structures: Dict[str, Episode]  # structure_id -> Episode


# --------------------------------------------------------------------------- #
# Step 1 -- TRUE TOUCH records (reuse canonical owner via entry_matches)
# --------------------------------------------------------------------------- #
def extract_touch_records(entry_matches: List[List[Dict[str, Any]]]) -> List[List[Touch]]:
    """Convert canonical ``entry_matches`` provenance into per-bar Touch lists.

    ``entry_matches[bar]`` is the list of hit zones returned by the canonical
    true-touch owner (touch_from_prev_geometry_r4) for the bar vs geometry known
    at the PREVIOUS bar close. Each record carries tf/family/side/top/bottom/
    level/strength. We build a STABLE structure_id so that consecutive touches
    of the same structure collapse into one episode.

    SR identity  : (tf, top, bottom)   -- matches V4 _parse_sr (strength ignored)
    LIQ identity : (tf, side, level, top, bottom)
    """
    touch_by_bar: List[List[Touch]] = []
    for bar, matches in enumerate(entry_matches):
        recs: List[Touch] = []
        for m in matches:
            tf = m["tf"]
            fam = m["family"]
            side = m.get("side")
            top = float(m["top"])
            bottom = float(m["bottom"])
            level = m.get("level")
            strength = m.get("strength")
            if fam == "SR":
                sid = f"SR|{tf}|{top:.6f}|{bottom:.6f}"
            else:
                lv = float(level) if level is not None else float("nan")
                sid = f"LIQ|{tf}|{side}|{lv:.6f}|{top:.6f}|{bottom:.6f}"
            recs.append(Touch(bar, sid, fam, tf, side, bottom, top, level, strength))
        touch_by_bar.append(recs)
    return touch_by_bar


# --------------------------------------------------------------------------- #
# Step 2/3 -- touch episodes + touch groups
# --------------------------------------------------------------------------- #
def build_episodes(touch_by_bar: List[List[Touch]]) -> List[Episode]:
    """Collapse per-bar touches into maximal consecutive episodes per structure.

    Consecutive touches of the SAME structural identity are ONE episode. A gap
    (the structure is no longer touched) ends the episode; a later separated
    revisit starts a NEW episode.
    """
    by_struct: Dict[str, List[int]] = defaultdict(list)
    meta: Dict[str, Tuple[float, float, str, str, Optional[str]]] = {}
    for bar, recs in enumerate(touch_by_bar):
        for rec in recs:
            by_struct[rec.structure_id].append(bar)
            if rec.structure_id not in meta:
                meta[rec.structure_id] = (
                    rec.zone_bottom, rec.zone_top, rec.family, rec.tf, rec.side,
                )
    episodes: List[Episode] = []
    for sid, bars in by_struct.items():
        zb, zt, fam, tf, side = meta[sid]
        ubars = sorted(set(bars))
        run = [ubars[0]]
        for b in ubars[1:]:
            if b == run[-1] + 1:
                run.append(b)
            else:
                episodes.append(Episode(sid, list(run), run[0], run[-1], zb, zt, fam, tf, side))
                run = [b]
        episodes.append(Episode(sid, list(run), run[0], run[-1], zb, zt, fam, tf, side))
    return episodes


def build_touch_groups(episodes: List[Episode]) -> List[TouchGroup]:
    """Group episodes by their start_bar into simultaneous touch groups.

    If multiple DIFFERENT structures first touch on the SAME bar, they form one
    simultaneous touch group. There is no knowable intrabar ordering, so we do
    NOT manufacture one.
    """
    by_start: Dict[int, List[Episode]] = defaultdict(list)
    for ep in episodes:
        by_start[ep.start_bar].append(ep)
    groups: List[TouchGroup] = []
    for start in sorted(by_start):
        eps_here = by_start[start]
        structures = {ep.structure_id: ep for ep in eps_here}
        groups.append(TouchGroup(start_bar=start, episodes=eps_here, structures=structures))
    return groups


# --------------------------------------------------------------------------- #
# Step 4/5/6 -- classify one adjacent A->B transition
# --------------------------------------------------------------------------- #
def _classify_transition(
    Azone: Dict[str, Tuple[float, float]],
    Bzone: Dict[str, Tuple[float, float]],
    distinct_b_ids: Any,
    eps: float = EPS,
) -> Tuple[str, Optional[str], Optional[str], Optional[float]]:
    """Classify the transition A -> B where B is the FIRST later touch group
    that contains at least one LOCATION-DISTINCT structure zone
    (``collect_source_cluster_and_next_target`` guarantees this).

    Only the location-distinct B zones (``distinct_b_ids``) drive direction and
    target selection; same-location zones (overlapping A) have NO effect.

    The Exit is the NEAR edge of the chosen target:

      LONG  -- price travels UP from A, so the LOWEST zone_bottom is hit first.
      SHORT -- price travels DOWN from A, so the HIGHEST zone_top is hit first.

    Returns (status, direction, target_id, exit_price).

    status:
      AMBIGUOUS_SAME_BAR  -- the distinct targets include one ABOVE and one
                             BELOW A -> no canonical direction
      OVERLAP             -- (defensive) no clean upper/lower distinct target
      CANONICAL           -- a single unambiguous direction; target chosen
    """
    new_ids = list(distinct_b_ids)
    if not new_ids:
        raise AssertionError(
            "find_next_distinct_group must not return a same-only B"
        )

    upper: List[Tuple[str, float]] = []   # (bid, b_bottom) -- LONG near edge
    lower: List[Tuple[str, float]] = []   # (bid, b_top)    -- SHORT near edge
    for bid in new_ids:
        bb, bt = Bzone[bid]
        is_upper = False
        is_lower = False
        for _aid, (ab, at) in Azone.items():
            if bb > at + eps:
                is_upper = True
            if bt < ab - eps:
                is_lower = True
        if is_upper:
            upper.append((bid, float(bb)))
        if is_lower:
            lower.append((bid, float(bt)))

    if upper and lower:
        return ("AMBIGUOUS_SAME_BAR", None, None, None)
    if upper:
        # price travelling upward reaches the LOWEST bottom first
        target_id, exit_price = min(upper, key=lambda x: x[1])
        return ("CANONICAL", ORACLE_LONG, target_id, float(exit_price))
    if lower:
        # price travelling downward reaches the HIGHEST top first
        target_id, exit_price = max(lower, key=lambda x: x[1])
        return ("CANONICAL", ORACLE_SHORT, target_id, float(exit_price))
    return ("OVERLAP", None, None, None)


def _choose_anchor(
    Azone: Dict[str, Tuple[float, float]],
    direction: str,
    target_id: str,
    Bzone: Dict[str, Tuple[float, float]],
    eps: float = EPS,
) -> Optional[str]:
    """Pick the A structure_id to record as the trade's anchor structure.

    The anchor must be on the correct side of the CHOSEN target, then we pick
    the A structure CLOSEST to the target:

      LONG  : among A strictly below target -> highest zone_top (nearest below)
      SHORT : among A strictly above target -> lowest zone_bottom (nearest above)

    Entry search then uses ONLY this anchor's touched_bars (never a union of all
    A structures).
    """
    bb, bt = Bzone[target_id]
    valid: List[Tuple[str, float]] = []
    if direction == ORACLE_LONG:
        for aid, (ab, at) in Azone.items():
            if bb > at + eps:
                valid.append((aid, float(at)))
        if not valid:
            return None
        return max(valid, key=lambda x: x[1])[0]
    else:
        for aid, (ab, at) in Azone.items():
            if bt < ab - eps:
                valid.append((aid, float(ab)))
        if not valid:
            return None
        return min(valid, key=lambda x: x[1])[0]


# --------------------------------------------------------------------------- #
# Location-distinct ownership helpers
# --------------------------------------------------------------------------- #
def zones_overlap(
    z1: Tuple[float, float],
    z2: Tuple[float, float],
    eps: float = EPS,
) -> bool:
    a_bottom, a_top = z1
    b_bottom, b_top = z2
    # strictly separated (disjoint) => not overlapping
    return not (b_top < a_bottom - eps or b_bottom > a_top + eps)


def split_same_vs_distinct_location(
    Azone: Dict[str, Tuple[float, float]],
    Bzone: Dict[str, Tuple[float, float]],
    eps: float = EPS,
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
    """Split B's structures into same-location vs location-distinct relative to A.

    A B-structure is SAME LOCATION if its zone overlaps ANY structure zone in
    the current A group. Only a B-zone overlapping NONE of A's zones is
    location-distinct. A different structure_id / timeframe / SR object whose
    zone still overlaps A is the SAME location and must be skipped -- never
    rejected as OVERLAP.
    """
    same: Dict[str, Tuple[float, float]] = {}
    distinct: Dict[str, Tuple[float, float]] = {}
    for bid, bz in Bzone.items():
        is_same_location = any(
            zones_overlap(az, bz, eps) for az in Azone.values()
        )
        if is_same_location:
            same[bid] = bz
        else:
            distinct[bid] = bz
    return same, distinct


# --------------------------------------------------------------------------- #
# Step 7 -- ONE source location -> ONE next target (source-cluster collector)
# --------------------------------------------------------------------------- #
def collect_source_cluster_and_next_target(
    groups: List[TouchGroup],
    start_idx: int,
    eps: float = EPS,
) -> Tuple[List[int], Optional[int], Optional[TouchGroup], set]:
    """Collect ONE source opportunity starting at ``groups[start_idx]``.

    - The source LOCATION is defined by the first group's zones.
    - Every later group whose structures are ALL in the SAME overlapping price
      location is ABSORBED into the source cluster (repeated touches of the
      same location are ONE source opportunity, regardless of structure_id /
      timeframe).
    - The cluster stops at the FIRST later group containing a location-distinct
      structure -- that group is the single next target B.

    Returns ``(source_group_indices, target_group_index, target_group,
    distinct_target_ids)``. ``target_group_index``/``target_group`` are None
    (and ``distinct_target_ids`` empty) when no later distinct location exists.

    Example: A@10, A@20, A@30, B@40  ->
        source_indices = [0, 1, 2], target = B@40.
    These source groups represent ONE trade opportunity.
    """
    A0 = groups[start_idx]
    source_indices = [start_idx]

    # The source location grows only by structures overlapping the current
    # source location cluster.
    source_zones: Dict[str, Tuple[float, float]] = {
        sid: (ep.zone_bottom, ep.zone_top) for sid, ep in A0.structures.items()
    }

    j = start_idx + 1
    while j < len(groups):
        G = groups[j]
        Gzone = {
            sid: (ep.zone_bottom, ep.zone_top) for sid, ep in G.structures.items()
        }
        _same, distinct = split_same_vs_distinct_location(source_zones, Gzone, eps)
        if distinct:
            return source_indices, j, G, set(distinct.keys())
        # Entire group is the same source location: absorb it.
        source_indices.append(j)
        source_zones.update(Gzone)
        j += 1

    return source_indices, None, None, set()


# --------------------------------------------------------------------------- #
# Step 9 -- best Entry within A's contact bars (God-mode)
# --------------------------------------------------------------------------- #
def _same_unit(d: int, segments: np.ndarray, tds: np.ndarray) -> bool:
    """decision close[d] -> fill open[d+1] must stay in the same execution unit."""
    if d + 1 >= len(segments):
        return False
    return bool(segments[d + 1] == segments[d]) and bool(tds[d + 1] == tds[d])


def best_entry(
    contact_bars: List[int],
    opens: np.ndarray,
    segments: np.ndarray,
    tds: np.ndarray,
    exit_price: float,
    direction: str,
    n: int,
    exit_bar: int,
    eps: float = EPS,
) -> Optional[Tuple[int, float]]:
    """God-mode best Entry among A's legal contact bars.

    LONG  : minimize open[d+1] (best fill below target), subject to
            open[d+1] < exit_price and same execution unit.
    SHORT : maximize open[d+1], subject to open[d+1] > exit_price and same unit.
    Tie (equal fill) -> earliest decision bar.

    Bars >= exit_bar are excluded (entry must precede the fixed Exit).
    Returns (decision_bar, entry_price) or None if no legal entry.
    """
    best: Optional[Tuple[int, float]] = None
    for d in contact_bars:
        if d >= exit_bar:
            continue
        if d + 1 >= n:
            continue
        if not _same_unit(d, segments, tds):
            continue
        o = float(opens[d + 1])
        if direction == ORACLE_LONG and not (o < exit_price):
            continue
        if direction == ORACLE_SHORT and not (o > exit_price):
            continue
        if best is None:
            best = (d, o)
        else:
            if direction == ORACLE_LONG:
                if o < best[1] - eps or (abs(o - best[1]) <= eps and d < best[0]):
                    best = (d, o)
            else:
                if o > best[1] + eps or (abs(o - best[1]) <= eps and d < best[0]):
                    best = (d, o)
    return best


# --------------------------------------------------------------------------- #
# Step 4-10 -- solve the whole chain
# --------------------------------------------------------------------------- #
def solve_touch_chain(
    groups: List[TouchGroup],
    opens: np.ndarray,
    segments: np.ndarray,
    tds: np.ndarray,
    n: int,
    total_true_touch_records: int,
    total_touch_episodes: int,
    total_touch_groups: int,
    eps: float = EPS,
    times: Optional[np.ndarray] = None,
    overlap_examples: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Walk the touch-group chain and emit the canonical trade stream.

    ONE source price location -> ONE next distinct target -> ONE trade.

    Repeated same-location touches (overlapping zones, even different
    structure_id / timeframe) are ABSORBED into a single source cluster
    (``collect_source_cluster_and_next_target``); they only enlarge the legal
    Entry search set for the ONE A->B trade. The chain advances to B, so
    A->B->C produces exactly two trades. Every source cluster resolves into
    exactly one reconciliation bucket (hard-asserted at the end).
    """
    trades: List[Dict[str, Any]] = []
    audit = {
        "total_true_touch_records": int(total_true_touch_records),
        "total_touch_episodes": int(total_touch_episodes),
        "total_touch_groups": int(total_touch_groups),
        "source_clusters": 0,
        "same_location_groups_absorbed": 0,
        "target_transitions": 0,
        "ambiguous_same_bar_target_groups": 0,
        "overlapping_zone_transitions": 0,
        "no_legal_entry_transitions": 0,
        "no_later_distinct_target_transitions": 0,
        "canonical_trades": 0,
    }

    def _t(idx: int) -> Any:
        return None if times is None or idx >= len(times) else times[idx]

    m = len(groups)
    k = 0
    while k < m:
        source_indices, j, B, distinct_b_ids = (
            collect_source_cluster_and_next_target(groups, k, eps)
        )
        audit["source_clusters"] += 1
        audit["same_location_groups_absorbed"] += len(source_indices) - 1
        if B is None:
            # The final cluster reaches the end of the chain with no target.
            audit["no_later_distinct_target_transitions"] += 1
            break
        audit["target_transitions"] += 1

        Azone = {
            sid: (ep.zone_bottom, ep.zone_top)
            for idx in source_indices
            for sid, ep in groups[idx].structures.items()
        }
        Bzone = {
            sid: (ep.zone_bottom, ep.zone_top) for sid, ep in B.structures.items()
        }

        status, direction, target_id, exit_price = _classify_transition(
            Azone, Bzone, distinct_b_ids, eps
        )
        if status == "AMBIGUOUS_SAME_BAR":
            audit["ambiguous_same_bar_target_groups"] += 1
            k = j
            continue
        if status == "OVERLAP":
            audit["overlapping_zone_transitions"] += 1
            if overlap_examples is not None and len(overlap_examples) < 20:
                overlap_examples.append({
                    "A_start_bar": int(groups[source_indices[0]].start_bar),
                    "A_structures": sorted(Azone.keys()),
                    "B_start_bar": int(B.start_bar),
                    "B_structures": sorted(B.structures.keys()),
                })
            k = j
            continue

        # CANONICAL: pick the anchor A FIRST, then search Entry ONLY inside the
        # anchor's own PRICE LOCATION across the whole source cluster (the
        # cluster IS one location; absorbed groups enlarge the contact set).
        exit_bar = B.start_bar
        anchor = _choose_anchor(Azone, direction, target_id, Bzone, eps)
        if anchor is None:
            audit["no_legal_entry_transitions"] += 1
            k = j
            continue
        anchor_zone = Azone[anchor]
        contact = sorted({
            bar
            for idx in source_indices
            for sid, ep in groups[idx].structures.items()
            if zones_overlap(Azone[sid], anchor_zone, eps)
            for bar in ep.touched_bars
        })

        entry = best_entry(
            contact, opens, segments, tds, float(exit_price), direction, n, exit_bar, eps
        )
        if entry is None:
            audit["no_legal_entry_transitions"] += 1
            k = j
            continue

        d_star, entry_price = entry
        a_zb, a_zt = anchor_zone

        utility = (
            (float(exit_price) - entry_price)
            if direction == ORACLE_LONG
            else (entry_price - float(exit_price))
        )

        trades.append({
            "structure_id": anchor,
            "oracle_direction": direction,
            "candidate_start_bar": int(groups[source_indices[0]].start_bar),
            "candidate_start_time": _t(groups[source_indices[0]].start_bar),
            "zone_bottom": float(a_zb),
            "zone_top": float(a_zt),
            "best_entry_decision_index": int(d_star),
            "best_entry_fill_index": int(d_star + 1),
            "best_entry_fill_time": _t(d_star + 1),
            "best_entry_price": float(entry_price),
            "target_structure_id": target_id,
            "target_price": float(exit_price),
            "exit_fill_index": int(exit_bar),
            "exit_fill_time": _t(exit_bar),
            "exit_price": float(exit_price),
            "exit_reason": R_TARGET,
            "utility": float(utility),
            "event_id": None,
            "canonical_oracle_trade": True,
            # provenance (diagnostic; not part of the artifact trade schema)
            "absorbed_source_group_start_bars": [
                int(groups[idx].start_bar) for idx in source_indices
            ],
            "entry_contact_bars": [int(b) for b in contact],
        })
        audit["canonical_trades"] += 1
        k = j

    # ---- HARD RECONCILIATION (no unexplained disappearance) ----
    # Every source cluster resolves into EXACTLY ONE named bucket:
    #   source_clusters
    #     = target_transitions + no_later_distinct_target_transitions
    #     = canonical + ambiguous + overlap + no_entry + no_later_distinct
    #   target_transitions
    #     = canonical + ambiguous + overlap + no_entry
    bucket_sum = (
        audit["canonical_trades"]
        + audit["ambiguous_same_bar_target_groups"]
        + audit["overlapping_zone_transitions"]
        + audit["no_legal_entry_transitions"]
        + audit["no_later_distinct_target_transitions"]
    )
    if audit["source_clusters"] != bucket_sum:
        raise AssertionError(
            f"HARD_FAIL_TOUCH_CHAIN_RECONCILIATION: "
            f"source_clusters={audit['source_clusters']} != buckets={bucket_sum}"
        )
    if audit["target_transitions"] != (
        bucket_sum - audit["no_later_distinct_target_transitions"]
    ):
        raise AssertionError(
            f"HARD_FAIL_TOUCH_CHAIN_RECONCILIATION: "
            f"target_transitions={audit['target_transitions']} != "
            f"buckets-with-target={bucket_sum - audit['no_later_distinct_target_transitions']}"
        )
    if audit["total_touch_groups"] != (
        audit["source_clusters"] + audit["same_location_groups_absorbed"]
    ):
        raise AssertionError(
            f"HARD_FAIL_TOUCH_CHAIN_GROUP_ACCOUNTING: "
            f"total_touch_groups={audit['total_touch_groups']} != "
            f"source_clusters={audit['source_clusters']} + "
            f"absorbed={audit['same_location_groups_absorbed']}"
        )

    # ---- HARD DUPLICATE AUDIT: one source location -> one trade -> one exit --
    # Targets strictly advance along the chain, so the same target group / exit
    # can never serve two canonical trades.
    seen: Dict[Tuple[str, int], int] = {}
    for t in trades:
        key = (str(t["target_structure_id"]), int(t["exit_fill_index"]))
        if key in seen:
            raise AssertionError(
                f"HARD_FAIL_DUPLICATE_TARGET_EXIT: "
                f"target={key[0]} exit_fill_index={key[1]} emitted twice "
                f"(same source location was not consumed)"
            )
        seen[key] = 1
    return trades, audit


# --------------------------------------------------------------------------- #
# Top-level driver
# --------------------------------------------------------------------------- #
def run_touch_chain_oracle(
    symbol: str,
    max_bars: Optional[int] = None,
    eps: float = EPS,
) -> Dict[str, Any]:
    """Run the V6 touch-chain oracle for one symbol.

    Reuses the canonical R4 15m environment (true-touch provenance enabled).
    Returns trades + audit counts + supporting structures for the screenshot
    audit.
    """
    env = run_environment_m15(symbol, max_bars, capture_provenance=True)
    entry_matches = env["entry_matches"]
    ef = env["exec_frame"]
    n = len(entry_matches)

    opens = np.asarray(ef["open"].to_numpy(float))
    closes = np.asarray(ef["close"].to_numpy(float))
    times = np.asarray(ef["bar_start_time"].to_numpy())
    segments = np.asarray(ef["segment"].to_numpy(np.int64))
    tds = np.asarray(ef["trading_day"].to_numpy(np.int64))

    touch_by_bar = extract_touch_records(entry_matches)
    episodes = build_episodes(touch_by_bar)
    groups = build_touch_groups(episodes)

    total_true_touch_records = sum(len(x) for x in touch_by_bar)
    trades, audit = solve_touch_chain(
        groups, opens, segments, tds, n,
        total_true_touch_records, len(episodes), len(groups), eps, times,
    )
    return {
        "symbol": symbol,
        "math_version": MATH_VERSION,
        "trades": trades,
        "audit": audit,
        "groups": groups,
        "episodes": episodes,
        "n": n,
        "opens": opens,
        "times": times,
    }


# --------------------------------------------------------------------------- #
# Screenshot-2 audit (manual review aid)
# --------------------------------------------------------------------------- #
def print_screenshot_audit(
    result: Dict[str, Any],
    around_price: float,
    band: float = 30.0,
) -> None:
    """Print the touch chain around a price level (e.g. the old Event-8 support).

    Shows the current A near the level, the next distinct B above it (=> LONG),
    the legal A entry bars / next-open prices, the chosen best Entry, and the
    fixed B Exit. Demonstrates that the old long-duration SHORT-to-7655 label
    cannot appear for this A->B transition.
    """
    groups: List[TouchGroup] = result["groups"]
    trades = result["trades"]
    opens = result["opens"]
    times = result["times"]

    # find the trade whose A zone is closest to around_price
    target_trade = None
    best_dist = float("inf")
    for t in trades:
        mid = (t["zone_bottom"] + t["zone_top"]) / 2.0
        dist = abs(mid - around_price)
        if dist < best_dist:
            best_dist = dist
            target_trade = t
    if target_trade is None:
        print("[screenshot] no trade found")
        return

    print("=" * 72)
    print(f"SCREENSHOT-2 AUDIT  (around price {around_price:.1f}, band {band:.1f})")
    print("=" * 72)
    a = target_trade
    print(f"current A (anchor structure): {a['structure_id']}")
    print(f"  A zone        : [{a['zone_bottom']:.1f}, {a['zone_top']:.1f}]")
    print(
        f"  source cluster: {a['absorbed_source_group_start_bars']} "
        f"(start bars; {len(a['absorbed_source_group_start_bars']) - 1} absorbed)"
    )
    print(f"  A start bar   : {a['candidate_start_bar']}")
    print(f"  DIRECTION     : {a['oracle_direction']}")
    print(f"  target B      : {a['target_structure_id']}")
    print(f"  target/exit   : {a['target_price']:.1f}  (exit_bar={a['exit_fill_index']})")
    print(f"  exit_reason   : {a['exit_reason']}")
    print("-" * 72)
    print("legal A entry bars (anchor price location across the WHOLE source cluster):")
    for d in a["entry_contact_bars"]:
        if d >= a["exit_fill_index"]:
            break
        print(
            f"  decision_bar={d:5d}  next_open={opens[d + 1]:.1f}  "
            f"same_unit_ok  -> fill@{d + 1}"
        )
    print("-" * 72)
    print(
        f"CHOSEN best Entry: decision_bar={a['best_entry_decision_index']} "
        f"fill={a['best_entry_fill_index']} price={a['best_entry_price']:.1f}"
    )
    print(
        f"FIXED Exit       : bar={a['exit_fill_index']} "
        f"price={a['exit_price']:.1f}"
    )
    print("=" * 72)
    # explicit negative assertion for the old wrong label
    if a["oracle_direction"] == ORACLE_LONG:
        print(
            "OK: A->B is LONG (next distinct touch is ABOVE A). "
            "No long-duration SHORT-to-7655 label exists for this transition."
        )
    else:
        print(
            "NOTE: this A->B transition is SHORT; verify it is a genuine "
            "next-distinct-touch-below, not the old 10-day SHORT artifact."
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    symbol = "AG"
    res = run_touch_chain_oracle(symbol)
    audit = res["audit"]
    print(f"[v6] symbol={symbol} n={res['n']}")
    for k in (
        "total_true_touch_records",
        "total_touch_episodes",
        "total_touch_groups",
        "source_clusters",
        "same_location_groups_absorbed",
        "target_transitions",
        "ambiguous_same_bar_target_groups",
        "overlapping_zone_transitions",
        "no_legal_entry_transitions",
        "no_later_distinct_target_transitions",
        "canonical_trades",
    ):
        print(f"  {k:36s}: {audit[k]}")

    # ---- duplicate audit: group canonical trades by (target, exit) ----
    dup: Dict[Tuple[str, int], int] = {}
    for t in res["trades"]:
        key = (str(t["target_structure_id"]), int(t["exit_fill_index"]))
        dup[key] = dup.get(key, 0) + 1
    duplicates = {k: v for k, v in dup.items() if v > 1}
    print(f"[v6] duplicate (target, exit_fill_index) groups: {len(duplicates)}")
    for key, cnt in sorted(duplicates.items())[:20]:
        print(f"    target={key[0]} exit={key[1]} count={cnt}")

    print_screenshot_audit(res, around_price=7690.0, band=30.0)


if __name__ == "__main__":
    main()
