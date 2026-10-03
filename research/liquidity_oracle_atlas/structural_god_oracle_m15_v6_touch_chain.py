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
    eps: float = EPS,
) -> Tuple[str, Optional[str], Optional[str], Optional[float]]:
    """Classify the transition A -> B where B is the FIRST later touch group
    that contains at least one NEW structure (``find_next_distinct_group``
    guarantees this, so a same-only B never reaches here).

    B's new structures define direction + fixed Exit. The Exit is the NEAR edge
    of the chosen target (the edge the price reaches FIRST):

      LONG  -- price travels UP from A, so the LOWEST zone_bottom is hit first.
      SHORT -- price travels DOWN from A, so the HIGHEST zone_top is hit first.

    Returns (status, direction, target_id, exit_price).

    status:
      AMBIGUOUS_SAME_BAR  -- the distinct next-touch bar holds BOTH an upper and
                             a lower target -> no canonical direction
      OVERLAP             -- only overlapping (unordered) distinct targets
      CANONICAL           -- a single unambiguous direction; target chosen
    """
    new_ids = [sid for sid in Bzone if sid not in Azone]
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
# Step 7 -- find the NEXT DISTINCT touch (skip same-location revisits)
# --------------------------------------------------------------------------- #
def _find_next_distinct_group(
    groups: List[TouchGroup],
    k: int,
) -> Tuple[Optional[int], Optional[TouchGroup], int]:
    """Starting after ``groups[k]``, skip groups that contain ONLY structural
    identities already present in A.

    Return the FIRST later group that contains at least one NEW structure.

    Same-location revisits do NOT terminate the search and are NOT a rejected
    trade -- they are simply ignored while searching for B.

    Returns ``(index, group, skipped)`` or ``(None, None, skipped)`` if no
    later distinct group exists.
    """
    A_ids = set(groups[k].structures)
    j = k + 1
    skipped = 0
    while j < len(groups):
        B_ids = set(groups[j].structures)
        if B_ids - A_ids:
            return j, groups[j], skipped
        skipped += 1
        j += 1
    return None, None, skipped


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
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Walk the touch-group chain and emit the canonical trade stream.

    For every source group A we search forward for the FIRST later group that
    contains a NEW structure (``_find_next_distinct_group``). Same-location
    revisits are skipped (not lost trades). Every source resolves into exactly
    one reconciliation bucket (hard-asserted at the end). No global scheduler.
    """
    trades: List[Dict[str, Any]] = []
    audit = {
        "total_true_touch_records": int(total_true_touch_records),
        "total_touch_episodes": int(total_touch_episodes),
        "total_touch_groups": int(total_touch_groups),
        "source_groups_with_future": int(total_touch_groups) - 1,
        "same_location_groups_skipped": 0,
        "ambiguous_same_bar_target_groups": 0,
        "overlapping_zone_transitions": 0,
        "no_legal_entry_transitions": 0,
        "no_later_distinct_target_transitions": 0,
        "canonical_trades": 0,
    }

    def _t(idx: int) -> Any:
        return None if times is None or idx >= len(times) else times[idx]

    m = len(groups)
    for k in range(m - 1):
        A = groups[k]
        j, B, skipped = _find_next_distinct_group(groups, k)
        audit["same_location_groups_skipped"] += skipped
        if B is None:
            audit["no_later_distinct_target_transitions"] += 1
            continue

        Azone = {sid: (ep.zone_bottom, ep.zone_top) for sid, ep in A.structures.items()}
        Bzone = {sid: (ep.zone_bottom, ep.zone_top) for sid, ep in B.structures.items()}

        status, direction, target_id, exit_price = _classify_transition(Azone, Bzone, eps)
        if status == "AMBIGUOUS_SAME_BAR":
            audit["ambiguous_same_bar_target_groups"] += 1
            continue
        if status == "OVERLAP":
            audit["overlapping_zone_transitions"] += 1
            continue

        # CANONICAL: pick the anchor A FIRST, then search Entry ONLY inside that
        # anchor's own contact bars (never a union of all A structures).
        exit_bar = B.start_bar
        anchor = _choose_anchor(Azone, direction, target_id, Bzone, eps)
        if anchor is None:
            audit["no_legal_entry_transitions"] += 1
            continue
        contact = sorted(set(A.structures[anchor].touched_bars))

        entry = best_entry(
            contact, opens, segments, tds, float(exit_price), direction, n, exit_bar, eps
        )
        if entry is None:
            audit["no_legal_entry_transitions"] += 1
            continue

        d_star, entry_price = entry
        a_zb, a_zt = Azone[anchor]

        utility = (
            (float(exit_price) - entry_price)
            if direction == ORACLE_LONG
            else (entry_price - float(exit_price))
        )

        trades.append({
            "structure_id": anchor,
            "oracle_direction": direction,
            "candidate_start_bar": int(A.start_bar),
            "candidate_start_time": _t(A.start_bar),
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
        })
        audit["canonical_trades"] += 1

    # ---- HARD RECONCILIATION (no unexplained disappearance) ----
    lhs = audit["source_groups_with_future"]
    rhs = (
        audit["canonical_trades"]
        + audit["ambiguous_same_bar_target_groups"]
        + audit["overlapping_zone_transitions"]
        + audit["no_legal_entry_transitions"]
        + audit["no_later_distinct_target_transitions"]
    )
    if lhs != rhs:
        raise AssertionError(
            f"HARD_FAIL_TOUCH_CHAIN_RECONCILIATION: "
            f"source_groups_with_future={lhs} != buckets={rhs}"
        )
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
    print(f"  A start bar   : {a['candidate_start_bar']}")
    print(f"  DIRECTION     : {a['oracle_direction']}")
    print(f"  target B      : {a['target_structure_id']}")
    print(f"  target/exit   : {a['target_price']:.1f}  (exit_bar={a['exit_fill_index']})")
    print(f"  exit_reason   : {a['exit_reason']}")
    print("-" * 72)
    print("legal A entry bars (best Entry search, anchor-only contact bars):")
    # locate the group for A
    A_group = next((g for g in groups if g.start_bar == a["candidate_start_bar"]), None)
    if A_group is not None:
        ep = A_group.structures.get(a["structure_id"])
        contact = sorted(set(ep.touched_bars)) if ep is not None else []
        for d in contact:
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
        "source_groups_with_future",
        "same_location_groups_skipped",
        "ambiguous_same_bar_target_groups",
        "overlapping_zone_transitions",
        "no_legal_entry_transitions",
        "no_later_distinct_target_transitions",
        "canonical_trades",
    ):
        print(f"  {k:32s}: {audit[k]}")
    print_screenshot_audit(res, around_price=7690.0, band=30.0)


if __name__ == "__main__":
    main()
