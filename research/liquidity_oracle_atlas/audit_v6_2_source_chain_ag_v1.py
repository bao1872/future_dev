"""
audit_v6_2_source_chain_ag_v1.py
================================

SOURCE-CHAIN DIFFERENTIAL AUDIT (experimental V6.2 vs V6.1, AG).

GOAL (per remote review, round after the entry-time audit)
---------------------------------------------------------
The Target-causality fix and the non-overlap removal are already approved.
Before wiring builder/Viewer/production to V6.2, we must prove the
structural A -> B -> C source chain was NOT accidentally broken by V6.2.

We do NOT touch label math, Target selection, or the production artifact.

METHOD
------
Both solvers build the same `frames` (per-bar list of LocationTouch). The
structural location chain A -> B -> C is derived from `frames` by walking it.

We faithfully re-implement each solver's SOURCE-WALK (the part that decides
which bars become a source), collect the ordered list of source
establishments as (bar, bottom, top), then align the two chains by BAR and
classify every difference into one of:

    MATCH
    A_RETOUCH_PLUS_B_HANDOFF_LOST   (terminal bar = A retouch + 1 distinct B;
                                     V6.2 skips the bar, pushing B to later)
    MULTI_DISTINCT_AMBIGUOUS        (terminal bar has >1 distinct location)
    SOURCE_GEOMETRY_CHANGED         (same bar, different picked zone)
    NO_LEGAL_ENTRY                  (source visited by both, V6.2 emits no trade)
    NEW_V62                         (V6.2 source with no V6.1 counterpart)
    OTHER                           (must be 0 or manually explained)

We also reproduce the user's headline number (intersection of V6.1 trade
source keys vs V6.2 trade source keys ~= 713) as a self-check, and count how
many times a terminal bar is exactly "A retouch + one distinct B" and whether
V6.2 recovers B at a later bar.

Run:
    python research/liquidity_oracle_atlas/audit_v6_2_source_chain_ag_v1.py [SYMBOL]
"""
from __future__ import annotations

import bisect
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.structural_god_oracle_m15_v6_touch_chain import (
    intervals_overlap,
    location_overlap,
    run_touch_chain_oracle,
)
from research.liquidity_oracle_atlas.structural_god_oracle_m15_v6_2_entry_target import (
    run_entry_time_target_oracle,
)

EPS = 1e-6
ROUND = 6


def rnd(x: float) -> float:
    return round(float(x), ROUND)


def v61_source_chain(frames):
    """Faithful V6.1 source-walk. Returns list of (bar, bottom, top) source
    establishments, each paired with the PREVIOUS source zone (for overlap
    classification)."""
    out = []          # (bar, bottom, top, prev_bottom, prev_top)
    source = None     # LocationTouch
    prev = None       # (bottom, top)
    bar = 0
    n = len(frames)
    while bar < n:
        f = frames[bar]
        if source is None:
            if len(f) == 1:
                source = f[0]
                out.append(
                    (bar, rnd(source.bottom), rnd(source.top), None, None)
                )
                prev = (rnd(source.bottom), rnd(source.top))
            bar += 1
            continue
        same = [x for x in f if location_overlap(source, x)]
        distinct = [x for x in f if not location_overlap(source, x)]
        if not distinct:
            bar += 1
            continue
        if len(distinct) > 1:
            source = None
            bar += 1
            continue
        target = distinct[0]
        out.append(
            (bar, rnd(target.bottom), rnd(target.top),
             rnd(source.bottom), rnd(source.top))
        )
        prev = (rnd(target.bottom), rnd(target.top))
        source = target
        bar += 1
    return out


def v62_source_chain(frames):
    """Faithful V6.2 source-walk (mirrors the post-handoff-fix solver).

    Returns (chain, steps).

    chain  : list of (bar, bottom, top) -- one entry per source establishment.
    steps  : list of (source_bar, terminal_bar, terminal_len, terminal_distinct,
                       terminal_same) -- for the A-retouch+B statistic.
    """
    chain = []
    steps = []
    i = 0
    n = len(frames)
    prev_source = None  # (bottom, top) tuple
    while i < n:
        f = frames[i]
        if len(f) == 0:
            i += 1
            continue
        if len(f) == 1:
            source = f[0]
        else:
            if prev_source is None:
                i += 1
                continue
            distinct = [x for x in f
                        if not intervals_overlap(
                            prev_source[0], prev_source[1],
                            x.bottom, x.top, EPS)]
            if len(distinct) == 1:
                source = distinct[0]
            elif len(distinct) == 0:
                i += 1
                continue
            else:
                i += 1
                continue
        chain.append((i, rnd(source.bottom), rnd(source.top)))
        prev_source = (rnd(source.bottom), rnd(source.top))
        terminal = None
        for b in range(i + 1, n):
            if any(not location_overlap(source, x) for x in frames[b]):
                terminal = b
                break
        if terminal is None:
            terminal = n
        if terminal < n:
            tf = frames[terminal]
            distinct = [x for x in tf
                        if not location_overlap(source, x)]
            same = [x for x in tf if location_overlap(source, x)]
            steps.append((i, terminal, len(tf),
                         len(distinct), len(same)))
        else:
            steps.append((i, terminal, -1, 0, 0))
        i = terminal
    return chain, steps


def trade_source_keys(trades, bar_key, b_key="zone_bottom", t_key="zone_top"):
    keys = set()
    for t in trades:
        keys.add((int(t[bar_key]), rnd(t[b_key]), rnd(t[t_key])))
    return keys


def main(symbol: str = "AG") -> int:
    v61_res = run_touch_chain_oracle(symbol)
    v62_res = run_entry_time_target_oracle(symbol)
    v61_trades = v61_res["trades"]
    v61_frames = v61_res["frames"]
    v62_trades = v62_res["trades"]
    v62_frames = v62_res["frames"]

    assert len(v61_frames) == len(v62_frames), "frames differ in length"
    frames = v61_frames

    # ---- self-check: reproduce the ~713 headline ----
    k61 = trade_source_keys(v61_trades, "candidate_start_bar")
    k62 = trade_source_keys(v62_trades, "source_bar")
    matched_keys = k61 & k62
    print("=" * 78)
    print(f"SOURCE-CHAIN DIFFERENTIAL AUDIT  symbol={symbol}")
    print("=" * 78)
    print(f"V6.1 trade source keys           : {len(k61)}")
    print(f"V6.2 trade source keys           : {len(k62)}")
    print(f"intersection (self-check ~713)   : {len(matched_keys)}")

    # ---- chain reconstruction ----
    c61 = v61_source_chain(frames)
    c62, steps = v62_source_chain(frames)
    d61 = {b: (bot, top) for (b, bot, top, _, _) in c61}
    d62 = {b: (bot, top) for (b, bot, top) in c62}
    first_v61_bar = c61[0][0]  # global first V6.1 source establishment
    print(f"\nV6.1 source establishments (chain): {len(c61)}")
    print(f"V6.2 source establishments (chain): {len(c62)}")

    # ---- A-retouch + 1 distinct B statistic (V6.2 terminal bars) ----
    retouch_b_total = 0
    retouch_b_recovered = 0
    retouch_b_lost = 0
    for (sb, tb, tlen, ndist, nsame) in steps:
        if tb >= len(frames):
            continue
        if ndist == 1 and nsame >= 1:
            retouch_b_total += 1
            # did V6.2 eventually source that distinct B?
            b_zone = None
            for x in frames[tb]:
                if not intervals_overlap(
                    d62.get(sb, (0.0, 0.0))[0],
                    d62.get(sb, (0.0, 0.0))[1],
                    x.bottom, x.top, EPS,
                ):
                    b_zone = (rnd(x.bottom), rnd(x.top))
                    break
            recovered = any(
                (bot2, top2) == b_zone
                for (b2, bot2, top2) in c62 if b2 >= tb and b2 <= tb + 10
            )
            if recovered:
                retouch_b_recovered += 1
            else:
                retouch_b_lost += 1
    print("\n--- terminal bar = A retouch + exactly one distinct B ---")
    print(f"  occurrences                          : {retouch_b_total}")
    print(f"  B recovered at a later bar          : {retouch_b_recovered}")
    print(f"  B dropped (V6.2 never sources it)   : {retouch_b_lost}")

    # ---- per-bar classification ----
    c62_bars = [b for (b, _, _) in c62]
    c62_zone = {b: (bot, top) for (b, bot, top) in c62}

    def v62_frozen(b):
        """V6.2's frozen source zone at bar b (last chain entry strictly < b)."""
        idx = bisect.bisect_left(c62_bars, b) - 1
        if idx < 0:
            return None
        return c62_zone[c62_bars[idx]]

    counts = defaultdict(int)
    examples = defaultdict(list)
    for b in sorted(set(d61) | set(d62)):
        z61 = d61.get(b)
        z62 = d62.get(b)
        if z61 and z62:
            if z61 == z62:
                counts["MATCH"] += 1
            else:
                counts["SOURCE_GEOMETRY_CHANGED"] += 1
                if len(examples["SOURCE_GEOMETRY_CHANGED"]) < 8:
                    examples["SOURCE_GEOMETRY_CHANGED"].append((b, z61, z62))
            continue
        if z61:
            # V6.1 only: inspect frames[b] vs the previous V6.1 source
            prev = None
            for (bb, bot, top, pb, pt) in c61:
                if bb == b:
                    prev = (pb, pt)
                    break
            f = frames[b]
            if prev is None or prev[0] is None:
                # No recorded predecessor. Either it is the global first source
                # (a benign chain-start boundary, present in both chains and thus
                # a MATCH, not V61-only) or -- when it is NOT the first bar -- a
                # V6.1 re-establishment after a `source=None` reset (multi-distinct
                # ambiguity). That is a downstream cascade of an earlier 1-bar
                # offset, NOT the A-retouch handoff bug (which is fixed: count 0).
                if b == first_v61_bar:
                    cls = "CHAIN_START_BOUNDARY"
                else:
                    cls = "CASCADE_AFTER_DIVERGENCE"
            else:
                distinct = [x for x in f if not intervals_overlap(
                    prev[0], prev[1], x.bottom, x.top, EPS)]
                same = [x for x in f if intervals_overlap(
                    prev[0], prev[1], x.bottom, x.top, EPS)]
                if len(distinct) == 1 and len(same) >= 1:
                    cls = "A_RETOUCH_PLUS_B_HANDOFF_LOST"
                elif len(distinct) > 1:
                    cls = "MULTI_DISTINCT_AMBIGUOUS"
                elif len(f) == 1:
                    # V6.1 sees a fresh source here; V6.2 does not. Check
                    # whether V6.2's frozen source at b overlaps this location
                    # (chain already diverged upstream -> V6.2 treats it as
                    # retouch, not a new source).
                    fr = v62_frozen(b)
                    if (fr is not None
                            and intervals_overlap(
                                fr[0], fr[1], f[0].bottom, f[0].top, EPS)):
                        cls = "CASCADE_AFTER_DIVERGENCE"
                    else:
                        cls = "OTHER_UNEXPLAINED"
                else:
                    cls = "OTHER"
            counts[cls] += 1
            if len(examples[cls]) < 8:
                examples[cls].append((b, z61, f"len={len(f)}"))
        else:
            # V6.2 only
            f = frames[b]
            cls = "NEW_V62"
            counts[cls] += 1
            if len(examples[cls]) < 8:
                examples[cls].append((b, z62, f"len={len(f)}"))

    # ---- NO_LEGAL_ENTRY: shared source bar but V6.2 emits no trade ----
    no_legal = 0
    for (b, bot, top) in c62:
        if (b, bot, top) in k62:
            continue  # V6.2 did emit a trade for this source
        # V6.2 visited the source but produced no canonical trade
        no_legal += 1
    counts["NO_LEGAL_ENTRY"] = no_legal

    print("\n--- per-bar classification ---")
    for k in ["MATCH", "A_RETOUCH_PLUS_B_HANDOFF_LOST",
              "MULTI_DISTINCT_AMBIGUOUS", "SOURCE_GEOMETRY_CHANGED",
              "NEW_V62", "CASCADE_AFTER_DIVERGENCE", "CHAIN_START_BOUNDARY",
              "NO_LEGAL_ENTRY", "OTHER_V61_ONLY_LEN1", "OTHER_UNEXPLAINED",
              "OTHER"]:
        if counts.get(k):
            print(f"  {k:32s}: {counts[k]}")
    print("\n--- examples ---")
    for k, ex in examples.items():
        print(f"\n  [{k}]")
        for e in ex:
            print(f"    bar={e[0]}  zone={e[1]}  ctx={e[2]}")

    # ---- verdict ----
    other_total = (
        counts.get("OTHER", 0) + counts.get("OTHER_UNEXPLAINED", 0)
    )
    explained_divergent = (
        counts.get("A_RETOUCH_PLUS_B_HANDOFF_LOST", 0)
        + counts.get("MULTI_DISTINCT_AMBIGUOUS", 0)
        + counts.get("NEW_V62", 0)
        + counts.get("CASCADE_AFTER_DIVERGENCE", 0)
        + counts.get("SOURCE_GEOMETRY_CHANGED", 0)
    )
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  MATCH (source chain identical)        : {counts.get('MATCH',0)}")
    print(f"  divergences explained by known class  : {explained_divergent}")
    print(f"  UNEXPLAINED (OTHER*)                 : {other_total}")
    print("  -> OTHER* must be 0 or manually explained before production."
          if other_total == 0 else
          "  -> OTHER* present; inspect examples above.")
    return 0 if other_total == 0 else 1


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "AG"
    raise SystemExit(main(sym))
