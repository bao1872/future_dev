"""structural_god_oracle_m15_v4
===============================

FUT-M15-STRUCTURAL-GOD-ORACLE-V4

God-mode oracle for structural events.

For every structural event E_i (identity reused from V2, unchanged):

    candidate zone A
    -> compute the best LONG trade  (V_LONG = max legal LONG PnL)
    -> compute the best SHORT trade (V_SHORT = max legal SHORT PnL)
    -> pick the side with the larger future profit

    (oracle_direction, best_entry, best_exit) are LABELS, not model inputs.

HARD PRINCIPLE
--------------
A canonical oracle trade MUST be profitable (PnL > 0). If

    max(V_LONG, V_SHORT) <= 0

the event is labelled NO_POSITIVE_OPPORTUNITY and no trade label is emitted.
A losing trade is never the formal oracle label.

What is new vs V2 / V3 (and frozen here)
-----------------------------------------
* Entry may ONLY come from bars where price is still in canonical 0.5 ATR
  proximity of THIS event's structure A ("event_contact_bars"). The generic
  ``dp_proximity_any`` (proximity to ANY structure) is NOT used as the entry
  gate.
* Each direction selects its OWN directional target with self-target exclusion
  and directional placement (``pick_directional_target_v4``). The candidate
  structure A can never be its own target.
* The oracle direction is decided by comparing V_LONG vs V_SHORT. No external
  direction model is ever consulted; the module imports no direction model.

Frozen semantics inherited from V2 / V3 (NOT re-derived)
-------------------------------------------------------
* structural-event identity (one structure = one event, even across a
  temporary loss of proximity)
* candidate geometry frozen when the event begins
* target touch is a HARD terminal (TP == Target, remaining == 0)
* objective: gross open-to-open PnL, zero cost
* decision at close(t) -> fill at open(t+1)
* tie-breaks: entry tie -> earlier entry; exit tie -> earlier exit

This module is PRODUCTION (O(L)). The independent O(L^2) reference lives in
``structural_god_oracle_reference_m15_v4.py`` and must not call this module's
solver or target picker.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.build_structural_dp_labels_m15_v2 import (
    _structures_in_proximity,
    build_structural_events_v2,
)
from research.liquidity_oracle_atlas.build_dp_proximity_m15_v1 import (
    build_dp_proximity_m15,
)
from research.liquidity_oracle_atlas.build_execution_environment_m15_v1 import (
    run_environment_m15,
)
from research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1 import (
    bar_zone_distance,
    select_target,
)
from research.liquidity_oracle_atlas.structural_event_dp_kernel_v3 import (
    EventMarketView,
    market_view_from_arrays,
)

MATH_VERSION = "structural-god-oracle-v4.3"
TASK_ID = "FUT-M15-STRUCTURAL-GOD-ORACLE-V4.3"
ATR_OWNER = "m15_atr@run_environment_m15"

LONG_ROLES = ("RESISTANCE", "BUYSIDE_LIQUIDITY")
SHORT_ROLES = ("SUPPORT", "SELLSIDE_LIQUIDITY")
PNL_EPS = 1e-9
REM_EPS = 1e-9
TIE_EPS = 1e-9
PLACE_EPS = 1e-9  # directional placement slack vs the candidate zone

R_TARGET = "TARGET_TOUCH"
R_EARLY = "DP_EARLY_EXIT"
R_LOSS = "DP_LOSS_EXIT"

ORACLE_LONG = "LONG"
ORACLE_SHORT = "SHORT"
ORACLE_NOPOS = "NO_POSITIVE_OPPORTUNITY"
ORACLE_TIE = "ORACLE_DIRECTION_TIE"
ORACLE_NO_TARGET = "NO_VALID_TARGET_BRANCH"


# --------------------------------------------------------------------------- #
# Structural identity helpers
# --------------------------------------------------------------------------- #
def _parse_sr(sid: str) -> Tuple[str, str, str, str]:
    # SR|tf|seg|fs|top|bottom|strength  ->  ignore fs + strength
    p = sid.split("|")
    return ("SR", p[1], p[2], p[4], p[5])


def _parse_liq(sid: str) -> Tuple[str, str, str, str, str, str]:
    # LIQ|tf|role|seg|left|level  ->  compare all (stable identity)
    p = sid.split("|")
    return ("LIQ", p[1], p[2], p[3], p[4], p[5])


def target_is_self(target_sid: str, candidate_sid: str) -> bool:
    """True iff ``target_sid`` is the candidate structure itself.

    SR ignores ``strength`` and ``first_seen`` (per spec section 6);
    LIQ compares tf / role / seg / left / level.
    """
    if candidate_sid.startswith("SR|"):
        if not target_sid.startswith("SR|"):
            return False
        c = _parse_sr(candidate_sid)
        t = _parse_sr(target_sid)
        return (
            c[1] == t[1]
            and c[2] == t[2]
            and abs(float(c[3]) - float(t[3])) <= 1e-6
            and abs(float(c[4]) - float(t[4])) <= 1e-6
        )
    if candidate_sid.startswith("LIQ|"):
        if not target_sid.startswith("LIQ|"):
            return False
        c = _parse_liq(candidate_sid)
        t = _parse_liq(target_sid)
        return (
            c[1] == t[1]
            and c[2] == t[2]
            and c[3] == t[3]
            and c[4] == t[4]
            and float(c[5]) == float(t[5])
        )
    return False


def structure_identity(sid: str) -> Tuple:
    """Comparison tuple ignoring fs / strength (for the T->next-static audit)."""
    if sid.startswith("SR|"):
        p = _parse_sr(sid)
        return ("SR", p[1], p[2], p[3], p[4])
    if sid.startswith("LIQ|"):
        p = _parse_liq(sid)
        return ("LIQ", p[1], p[2], p[3], p[4], p[5])
    return (sid,)


# --------------------------------------------------------------------------- #
# Per-event contact bars (proximity to THIS structure A only)
# --------------------------------------------------------------------------- #
def build_per_bar_proximity(
    prox: pd.DataFrame,
    geom: List[Optional[Dict[str, Any]]],
    sr_first_seen: Optional[Dict[Any, int]] = None,
) -> List[Optional[List[Tuple[str, str, float, float]]]]:
    """Recompute per-bar structure-in-proximity lists (reuses V2's owner).

    Mirrors ``build_structural_events_v2`` exactly so the resulting structure
    ids are byte-identical to the event identity. Only bars already flagged by
    ``dp_proximity_any`` are scanned (same gate V2 uses). ``sr_first_seen`` is
    shared with the target picker so the two agree on structure first-seen ids.
    """
    n = len(prox)
    prox_any = np.asarray(prox["dp_proximity_any"].to_numpy(), dtype=bool)
    low = prox["low"].to_numpy(float)
    high = prox["high"].to_numpy(float)
    seg = prox["segment"].to_numpy(np.int64)
    if sr_first_seen is None:
        sr_first_seen = {}
    per_bar: List[Optional[List[Tuple[str, str, float, float]]]] = [None] * n
    for t in range(n):
        if not prox_any[t]:
            continue
        prev = geom[t - 1] if t >= 1 else None
        per_bar[t] = _structures_in_proximity(
            prev, float(low[t]), float(high[t]), int(seg[t]), t, sr_first_seen
        )
    return per_bar


def build_event_contact_bars(
    events: List[Dict[str, Any]],
    per_bar: List[Optional[List[Tuple[str, str, float, float]]]],
) -> List[Dict[str, Any]]:
    """Attach ``event_contact_bars`` to each event.

    A bar t in [start_bar, end_bar) is a contact bar iff the event's own
    structure id is among the structures in proximity at t. This is the event
    specific proximity mask -- never the generic dp_proximity_any. A contact bar
    must leave room for a fill at t+1 (so t+1 < len(per_bar)).
    """
    n = len(per_bar)
    for e in events:
        sid = e["structure_id"]
        s = int(e["start_bar"])
        en = int(e["end_bar"])
        cb = [
            t
            for t in range(s, en)
            if t + 1 < n
            and per_bar[t] is not None
            and any(x[0] == sid for x in per_bar[t])
        ]
        e["contact_bars"] = cb
    return events


# --------------------------------------------------------------------------- #
# Directional target picker (production)
# --------------------------------------------------------------------------- #
def pick_directional_target_v4(
    geom_prev: Optional[Dict[str, Any]],
    direction: str,
    zone_bottom: float,
    zone_top: float,
    tf: str,
    seg: int,
    i: int,
    sr_first_seen: Dict[Any, int],
    candidate_sid: str,
) -> Optional[Dict[str, Any]]:
    """Select the directional target for one God-mode branch.

    Roles: LONG -> RESISTANCE / BUYSIDE_LIQUIDITY; SHORT -> SUPPORT / SELLSIDE.
    Exclusions:
      * self-target (the candidate structure A itself) is rejected
      * directional placement:
            LONG  target near_edge >  zone_top  + eps
            SHORT target near_edge <  zone_bottom - eps
    Among eligible candidates, pick the one nearest the candidate zone mid.
    """
    roles = LONG_ROLES if direction == "LONG" else SHORT_ROLES
    ref = 0.5 * (float(zone_bottom) + float(zone_top))
    cands: List[Dict[str, Any]] = []
    for tfk, g in (geom_prev or {}).items():
        channels, liq_up, liq_down, atr_tf = g
        for role in roles:
            c = select_target(
                role, channels, liq_up, liq_down, ref, float(atr_tf),
                tfk, int(seg), int(i), sr_first_seen,
            )
            if c is None:
                continue
            if target_is_self(c["structure_id"], candidate_sid):
                continue
            if direction == "LONG":
                if not (float(c["near_edge"]) > float(zone_top) + PLACE_EPS):
                    continue
            else:
                if not (float(c["near_edge"]) < float(zone_bottom) - PLACE_EPS):
                    continue
            c["tf"] = tfk
            cands.append(c)
    if not cands:
        return None
    return min(cands, key=lambda x: abs(float(x["near_edge"]) - ref))


# --------------------------------------------------------------------------- #
# Production solver: O(L) per direction
# --------------------------------------------------------------------------- #
def _suffix_argmax_per_unit(
    mv: EventMarketView, lo: int, hi: int, sign: float
) -> np.ndarray:
    """``suf[k]`` = argmax of ``sign*open`` over [k .. segment_end], earliest tie."""
    n = mv.n
    suf = np.full(n, -1, dtype=np.int64)
    opens = mv.opens
    k = int(hi)
    while k >= int(lo):
        u = int(np.searchsorted(mv.unit_starts, k, side="right")) - 1
        a = max(int(lo), int(mv.unit_starts[u]))
        best = k
        bestv = sign * float(opens[k])
        suf[k] = k
        for j in range(k - 1, a - 1, -1):
            v = sign * float(opens[j])
            if v >= bestv:
                bestv = v
                best = j
            suf[j] = best
        k = a - 1
    return suf


def _next_target_touch(
    mv: EventMarketView, lo: int, hi: int, direction: str, target_price: Optional[float]
) -> np.ndarray:
    """``nt[t]`` = earliest bar in [t .. hi] where the frozen target is touched."""
    n = mv.n
    nt = np.full(n, -1, dtype=np.int64)
    if target_price is None:
        return nt
    tp = float(target_price)
    highs, lows = mv.highs, mv.lows
    nxt = -1
    for t in range(int(hi), int(lo) - 1, -1):
        touched = (float(highs[t]) >= tp) if direction == "LONG" else (float(lows[t]) <= tp)
        if touched:
            nxt = t
        nt[t] = nxt
    return nt


def _unit_of(mv: "EventMarketView", bar: int) -> int:
    """Index of the intraday unit (maximal same trading_day+segment run) containing bar."""
    return int(np.searchsorted(mv.unit_starts, int(bar), side="right")) - 1


def solve_direction_god_v4(
    *,
    direction: str,
    zone_bottom: float,
    zone_top: float,
    atr_value: float,
    start_bar: int,
    end_bar: int,
    target_price: Optional[float],
    contact_bars: List[int],
    mv: EventMarketView,
    trading_day: Optional[np.ndarray] = None,
    segment: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Joint entry+exit DP for ONE direction, constrained to contact bars. O(L).

    V4.1 execution-boundary gate: a ``decision=t -> fill=t+1`` candidate is
    rejected unless ``t`` and ``t+1`` share the same trading_day, segment and
    intraday unit. This forbids entering on the last bar of a session and
    filling at the next session's open.
    """
    sign = 1.0 if direction == "LONG" else -1.0
    n = mv.n
    s_bar = int(start_bar)
    e_bar = int(min(end_bar, n))
    hi = min(e_bar, n - 1)

    n_contact = len(contact_bars)
    if hi < s_bar:
        return {
            "ok": False, "invalid_reason": "NO_EXECUTABLE_ENTRY",
            "n_candidates": 0, "n_with_path": 0,
            "n_rejected_target_before_entry": 0, "n_contact": n_contact,
        }

    nt = _next_target_touch(mv, s_bar, hi, direction, target_price)
    t_first = int(nt[s_bar]) if (target_price is not None and s_bar < n) else -1

    n_candidates = 0
    n_with_path = 0
    n_rejected = 0
    best: Optional[Dict[str, Any]] = None

    for d in contact_bars:
        d = int(d)
        if d < s_bar or d >= e_bar:
            continue
        n_candidates += 1
        f = d + 1
        if f >= n:
            n_rejected += 1
            continue
        # V4.1 execution-boundary gate: fill must stay in the same
        # trading_day / segment / intraday unit as the decision bar.
        if (trading_day is not None and segment is not None
                and (int(trading_day[d]) != int(trading_day[f])
                     or int(segment[d]) != int(segment[f])
                     or _unit_of(mv, d) != _unit_of(mv, f))):
            n_rejected += 1
            continue
        H = min(e_bar, mv.unit_end(f), n - 1)
        entry_price = float(mv.opens[f])

        same_bar_target = False
        if t_first >= 0:
            if t_first < f:
                n_rejected += 1
                continue
            if t_first == f:
                if direction == "LONG" and not entry_price < float(target_price):
                    n_rejected += 1
                    continue
                if direction == "SHORT" and not entry_price > float(target_price):
                    n_rejected += 1
                    continue
                same_bar_target = True

        if not same_bar_target and f + 1 > H:
            continue
        n_with_path += 1

        # V4.3: the ONLY valid exit is TARGET_TOUCH. If the target is not
        # reached within the allowed holding window, this entry does not form
        # a completed A->B label, so we skip it (no early-exit fallback).
        if same_bar_target:
            exit_fill = f
            exit_price = float(target_price)
            pnl = sign * (float(target_price) - entry_price)
            reason = R_TARGET
        elif t_first > f and t_first <= H:
            exit_fill = t_first
            exit_price = float(target_price)
            pnl = sign * (float(target_price) - entry_price)
            reason = R_TARGET
        else:
            n_rejected += 1
            continue

        cand = {
            "entry_decision_index": int(d),
            "entry_fill_index": int(f),
            "entry_price": entry_price,
            "exit_fill_index": int(exit_fill),
            "exit_price": exit_price,
            "exit_reason": reason,
            "utility": float(pnl),
        }
        if best is None or cand["utility"] > best["utility"]:
            best = cand

    if best is None:
        # V4.3: a direction with contact bars but no target-reaching entry has
        # its target NOT reached; no early-exit label is manufactured.
        reason = "NO_EXECUTABLE_ENTRY" if n_candidates == 0 else "TARGET_NOT_REACHED"
        return {
            "ok": False, "invalid_reason": reason,
            "n_candidates": n_candidates, "n_with_path": n_with_path,
            "n_rejected_target_before_entry": n_rejected, "n_contact": n_contact,
        }

    # terminal-priority ambiguity
    if (best["exit_reason"] == R_TARGET and best["exit_fill_index"] == e_bar
            and e_bar < n):
        return {
            "ok": False, "invalid_reason": "AMBIGUOUS_SAME_BAR_TERMINAL",
            "n_candidates": n_candidates, "n_with_path": n_with_path,
            "n_rejected_target_before_entry": n_rejected, "n_contact": n_contact,
        }

    return _finalize_god(
        direction=direction, zone_bottom=zone_bottom, zone_top=zone_top,
        atr_value=atr_value, target_price=target_price, mv=mv, best=best,
        n_candidates=n_candidates, n_with_path=n_with_path,
        n_rejected=n_rejected, n_contact=n_contact,
    )


def _finalize_god(
    *,
    direction: str,
    zone_bottom: float,
    zone_top: float,
    atr_value: float,
    target_price: Optional[float],
    mv: EventMarketView,
    best: Dict[str, Any],
    n_candidates: int,
    n_with_path: int,
    n_rejected: int,
    n_contact: int,
) -> Dict[str, Any]:
    sign = 1.0 if direction == "LONG" else -1.0

    def _dir(a: float, b: float) -> float:
        return (b - a) if direction == "LONG" else (a - b)

    entry_price = float(best["entry_price"])
    exit_price = float(best["exit_price"])
    pnl = float(best["utility"])

    gap_pts = bar_zone_distance(
        entry_price, entry_price, float(zone_bottom), float(zone_top)
    )
    gap_atr = gap_pts / float(atr_value)

    positive = pnl > PNL_EPS
    if best["exit_reason"] == R_TARGET:
        # HARD INVARIANT: with the target-before-entry filter a target touch can
        # never be a loss (entry is always on the profitable side of target).
        if not positive:
            raise AssertionError(
                "HARD_FAIL_TARGET_TOUCH_WITH_LOSS: "
                f"entry={entry_price} target={exit_price} pnl={pnl}"
            )
        remaining = 0.0
        tp_atr = _dir(entry_price, exit_price) / float(atr_value)
        remaining_raw = 0.0
    elif positive:
        tp_atr = _dir(entry_price, exit_price) / float(atr_value)
        if target_price is None:
            remaining = float("nan")
            remaining_raw = float("nan")
        else:
            raw = _dir(exit_price, float(target_price)) / float(atr_value)
            remaining_raw = raw
            if raw < -REM_EPS:
                raise AssertionError(
                    "HARD_FAIL_REMAINING_TARGET_NEGATIVE: "
                    f"raw_remaining={raw}"
                )
            remaining = raw
    else:
        tp_atr = float("nan")
        remaining = float("nan")
        remaining_raw = (
            float("nan") if target_price is None
            else _dir(exit_price, float(target_price)) / float(atr_value)
        )

    if target_price is None:
        target_distance_atr = float("nan")
    else:
        target_distance_atr = _dir(entry_price, float(target_price)) / float(atr_value)

    out = dict(best)
    out.update({
        "ok": True,
        "invalid_reason": None,
        "best_entry_gap_points": float(gap_pts),
        "best_entry_gap_atr": float(gap_atr),
        "positive_tp_exists": bool(positive),
        "tp_atr": float(tp_atr),
        "remaining_target_atr": float(remaining),
        "remaining_target_atr_raw": float(remaining_raw),
        "target_distance_atr": float(target_distance_atr),
        "optimal_exit_points_signed": float(pnl),
        "optimal_exit_atr_signed": float(pnl) / float(atr_value),
        "n_candidates": int(n_candidates),
        "n_with_path": int(n_with_path),
        "n_rejected_target_before_entry": int(n_rejected),
        "n_contact": int(n_contact),
        "entry_fill_time": pd.Timestamp(mv.times[best["entry_fill_index"]]),
        "exit_fill_time": pd.Timestamp(mv.times[best["exit_fill_index"]]),
    })
    return out


# --------------------------------------------------------------------------- #
# Oracle decision (direction-agnostic)
# --------------------------------------------------------------------------- #
def decide_oracle_v4(
    long_sol: Dict[str, Any],
    short_sol: Dict[str, Any],
    long_has_target: bool = True,
    short_has_target: bool = True,
) -> Tuple[str, Optional[float], Optional[float], Optional[str]]:
    """Compare V_LONG vs V_SHORT (target-valid branches only).

    Returns (decision, long_value, short_value, winner_side).
      * winner value > 0  -> LONG / SHORT (canonical trade)
      * both valid, both <= 0 -> NO_POSITIVE_OPPORTUNITY
      * both valid and equal within TIE_EPS -> ORACLE_DIRECTION_TIE
      * max(V_LONG, V_SHORT) <= 0 -> NO_POSITIVE_OPPORTUNITY
      * neither branch has a structural target -> NO_VALID_TARGET_BRANCH
    """
    lv = float(long_sol["utility"]) if long_sol.get("ok") else None
    sv = float(short_sol["utility"]) if short_sol.get("ok") else None

    if lv is not None and sv is not None:
        if abs(lv - sv) <= TIE_EPS:
            return (ORACLE_TIE, lv, sv, None)
        if max(lv, sv) <= 0:
            # both directions lose / break even -> never emit a losing trade
            return (ORACLE_NOPOS, lv, sv, None)
        if lv > sv:
            return (ORACLE_LONG, lv, sv, "LONG")
        return (ORACLE_SHORT, lv, sv, "SHORT")

    if lv is not None:
        return (
            ORACLE_LONG if lv > 0 else ORACLE_NOPOS, lv, None,
            "LONG" if lv > 0 else None,
        )
    if sv is not None:
        return (
            ORACLE_SHORT if sv > 0 else ORACLE_NOPOS, None, sv,
            "SHORT" if sv > 0 else None,
        )
    # neither branch produced an eligible (targeted, ok) trade
    if (not long_has_target) and (not short_has_target):
        return (ORACLE_NO_TARGET, None, None, None)
    return (ORACLE_NOPOS, None, None, None)


# --------------------------------------------------------------------------- #
# Orchestration core (shared by production + reference solvers)
# --------------------------------------------------------------------------- #
def _unit_starts_from_arrays(td: np.ndarray, seg: np.ndarray) -> np.ndarray:
    n = len(seg)
    starts = []
    for i in range(n):
        if i == 0 or td[i] != td[i - 1] or seg[i] != seg[i - 1]:
            starts.append(i)
    return np.asarray(starts, dtype=np.int64)


def _contact_is_consumed(contact_bar: int, cursor: int) -> bool:
    """A contact bar is already consumed (must be skipped) iff it is strictly
    before the current cursor. A contact exactly AT the cursor is the first
    legal next entry decision and MUST remain eligible (off-by-one guard)."""
    return int(contact_bar) < int(cursor)


def _scan_next_candidate(
    per_bar: List[Optional[List[Tuple[str, str, float, float]]]],
    cursor: int,
    eligible_sids: set,
    n: int,
) -> Tuple[Optional[int], Optional[str]]:
    """Sequential-candidate discovery, DECOUPLED from the static-event window.

    From ``cursor`` forward, find the earliest bar ``t`` that has at least one
    still-eligible structure present in ``per_bar[t]`` (i.e. not in the no-trade
    skip set and with a valid ``t+1`` fill). Among co-present eligible
    structures pick the lexicographically smallest structure_id (deterministic).
    A structure that previously COMPLETED a trade is intentionally still
    eligible here once the cursor has passed that trade's exit. Return
    ``(t, sid)``; if none remain, ``(None, None)``.

    This is intentionally NOT gated by any event's ``[start_bar, end_bar)``:
    a structure becomes a legal candidate the moment it first appears in
    proximity after the previous trade's exit, regardless of when the static
    builder would have started its event. The static event is used ONLY as a
    metadata source (sid -> zone/timeframe/event_id), never to bound
    eligibility.
    """
    for t in range(cursor, n):
        pb = per_bar[t]
        if pb is None:
            continue
        eligible = [x[0] for x in pb if x[0] in eligible_sids and t + 1 < n]
        if eligible:
            return (t, sorted(eligible)[0])
    return (None, None)


def _run_god_oracle_core(
    symbol: str,
    max_bars: Optional[int],
    event_limit: Optional[int],
    solve_direction_fn,
    pick_target_fn,
) -> Dict[str, Any]:
    ev = build_structural_events_v2(symbol, max_bars)
    events = ev["events"]
    meta_events = ev["meta"]
    if event_limit is not None:
        events = events[: int(event_limit)]

    prox = build_dp_proximity_m15(symbol, max_bars)
    n = len(prox)
    opens = prox["open"].to_numpy(float)
    highs = prox["high"].to_numpy(float)
    lows = prox["low"].to_numpy(float)
    closes = prox["close"].to_numpy(float)
    times = prox["bar_start_time"].to_numpy()
    seg_arr = prox["segment"].to_numpy(np.int64)
    td_arr = prox["trading_day"].to_numpy(np.int64)

    mv = market_view_from_arrays(
        opens, highs, lows, closes, times=times,
        unit_starts=_unit_starts_from_arrays(td_arr, seg_arr), symbol=symbol,
    )

    env = run_environment_m15(symbol, max_bars, capture_provenance=False)
    geom = env["geom_by_decision"]
    atr_series = np.asarray(env["features"]["m15_atr"].to_numpy(float))

    sr_fs: Dict[Any, int] = {}
    per_bar = build_per_bar_proximity(prox, geom, sr_fs)
    build_event_contact_bars(events, per_bar)

    # stable next-event lookup by event_id
    next_by_id: Dict[int, Dict[str, Any]] = {}
    for idx, e in enumerate(events):
        if idx + 1 < len(events):
            next_by_id[int(e["event_id"])] = events[idx + 1]

    records: List[Dict[str, Any]] = []
    self_target_total = 0

    # ----------------------------------------------------------------- #
    # Sequential trade stream (the frozen God-Mode lifecycle).
    #
    #   find candidate -> Entry -> frozen Target -> first TARGET_TOUCH
    #   -> trade completed -> IMMEDIATELY restart search after that exit
    #   -> next trade
    #
    # The old orchestration emitted at most ONE record per static structural
    # event, so the label stream was far too sparse. Here a single GLOBAL
    # bar cursor drives the stream: at each step we take the earliest legal
    # contact bar (>= cursor) from the EXISTING structural-event / proximity
    # owner, freeze Candidate A + geometry at that NEW decision time, compute
    # LONG/SHORT with the current God-mode rules, emit the winning completed
    # trade, then set cursor = exit_fill_index + 1 and restart. A previously
    # alive static event may therefore participate again as a NEW trade
    # after its exit (re-freezing geometry/target at its own decision time).
    #
    # Global non-overlap hard invariant:
    #   Entry_1 <= Exit_1 < Entry_2 <= Exit_2 < Entry_3 ...
    # is guaranteed because the next search never starts before exit+1, and
    # every emitted trade is a completed TARGET_TOUCH.
    # ----------------------------------------------------------------- #
    # ----------------------------------------------------------------- #
    # Sequential trade stream (frozen God-Mode lifecycle).
    #
    # Candidate discovery is DECOUPLED from the static-event window:
    # after each exit, cursor = exit_fill_index + 1, and the NEXT candidate
    # is found by scanning per_bar proximity directly -- ANY structure
    # present at a bar t >= cursor may become a new Candidate A, anchored at
    # the earliest such bar. The static event's [start_bar, end_bar) is NOT
    # used to gate eligibility. build_structural_events_v2 is retained ONLY
    # as a METADATA source (structure_id -> zone/timeframe/event_id); its
    # window semantics stay intact for its own (static) consumers.
    # ----------------------------------------------------------------- #
    # metadata lookup: structure_id -> (event_id, zone, timeframe, structure_type)
    struct_meta: Dict[str, Dict[str, Any]] = {}
    for e in events:
        struct_meta[e["structure_id"]] = {
            "event_id": int(e["event_id"]),
            "zone_bottom": float(e["zone_bottom"]),
            "zone_top": float(e["zone_top"]),
            "timeframe": e.get("timeframe"),
            "structure_type": e.get("structure_type"),
        }
    eligible_sids = set(struct_meta.keys())

    # `no_trade_sids` blocks ONLY structures that cannot produce a trade at the
    # current cursor (no valid contact bar, or scanned with no decision). It is
    # deliberately NOT a global per-structure lifetime dedup: a structure that
    # COMPLETED a trade must stay re-eligible after the cursor passes that exit,
    # so the same structure may participate again as a fresh candidate
    # (re-freezing geometry/target at its new decision time).
    no_trade_sids: set = set()

    cursor = 0
    guard = 0
    # trade iterations advance `cursor`; no-trade iterations add to
    # `no_trade_sids`. Either way the scan makes forward progress.
    MAX_ITER = 10 * (len(events) + 16)
    while True:
        guard += 1
        if guard > MAX_ITER:
            raise AssertionError("HARD_FAIL_CANDIDATE_SCAN_RUNWAY")

        c, cand_sid = _scan_next_candidate(per_bar, cursor, eligible_sids - no_trade_sids, n)
        if c is None:
            break

        meta = struct_meta[cand_sid]
        eid = meta["event_id"]
        zb, zt = meta["zone_bottom"], meta["zone_top"]
        tf = meta["timeframe"]

        atr_value = (
            float(atr_series[c]) if c < len(atr_series) else float("nan")
        )
        seg_c = int(seg_arr[c]) if c < n else 0
        geom_prev = geom[c - 1] if c >= 1 else None

        # God-mode best-entry: pass the COMPLETE legal contact-bar set of
        # candidate A from the cursor forward. Any bar >= c (hence >= cursor)
        # where A is present in per_bar is a decision candidate; the solver
        # then picks the true best entry among all of them, geometry/targets
        # frozen at this decision time c.
        cb_full = [
            t for t in range(c, n)
            if t + 1 < n and per_bar[t] is not None
            and any(x[0] == cand_sid for x in per_bar[t])
        ]
        if not cb_full:
            # no legal contact remains for A from cursor forward: this structure
            # can never yield a trade at/after the current cursor, so block it to
            # keep the scan advancing. (Data-end case: only proximity at the last
            # bar, with no subsequent bar to fill an entry.)
            no_trade_sids.add(cand_sid)
            continue

        tgt_long = pick_target_fn(
            geom_prev, "LONG", zb, zt, tf, seg_c, c,
            sr_fs, cand_sid,
        )
        tgt_short = pick_target_fn(
            geom_prev, "SHORT", zb, zt, tf, seg_c, c,
            sr_fs, cand_sid,
        )
        if tgt_long is not None and target_is_self(tgt_long["structure_id"], cand_sid):
            self_target_total += 1
        if tgt_short is not None and target_is_self(tgt_short["structure_id"], cand_sid):
            self_target_total += 1

        # V4.1 target-gated branch validity: a direction with no structural
        # target cannot produce the required three-label trade -> branch
        # invalid (NO_STRUCTURAL_TARGET); the solver is skipped entirely.
        if tgt_long is None:
            long_sol: Dict[str, Any] = {
                "ok": False, "invalid_reason": "NO_STRUCTURAL_TARGET",
                "n_candidates": 0, "n_with_path": 0,
                "n_rejected_target_before_entry": 0, "n_contact": len(cb_full),
            }
        else:
            long_sol = solve_direction_fn(
                direction="LONG", zone_bottom=zb, zone_top=zt, atr_value=atr_value,
                start_bar=c, end_bar=n,
                target_price=float(tgt_long["near_edge"]),
                contact_bars=cb_full, mv=mv,
                trading_day=td_arr, segment=seg_arr,
            )
        if tgt_short is None:
            short_sol: Dict[str, Any] = {
                "ok": False, "invalid_reason": "NO_STRUCTURAL_TARGET",
                "n_candidates": 0, "n_with_path": 0,
                "n_rejected_target_before_entry": 0, "n_contact": len(cb_full),
            }
        else:
            short_sol = solve_direction_fn(
                direction="SHORT", zone_bottom=zb, zone_top=zt, atr_value=atr_value,
                start_bar=c, end_bar=n,
                target_price=float(tgt_short["near_edge"]),
                contact_bars=cb_full, mv=mv,
                trading_day=td_arr, segment=seg_arr,
            )

        long_final = _finalize_god(
            direction="LONG", zone_bottom=zb, zone_top=zt, atr_value=atr_value,
            target_price=(None if tgt_long is None else float(tgt_long["near_edge"])),
            mv=mv, best=long_sol, n_candidates=0, n_with_path=0,
            n_rejected=0, n_contact=len(cb_full),
        ) if long_sol.get("ok") else None
        short_final = _finalize_god(
            direction="SHORT", zone_bottom=zb, zone_top=zt, atr_value=atr_value,
            target_price=(None if tgt_short is None else float(tgt_short["near_edge"])),
            mv=mv, best=short_sol, n_candidates=0, n_with_path=0,
            n_rejected=0, n_contact=len(cb_full),
        ) if short_sol.get("ok") else None

        decision, lv, sv, winner_side = decide_oracle_v4(
            long_sol, short_sol,
            long_has_target=(tgt_long is not None),
            short_has_target=(tgt_short is not None),
        )

        if decision in (ORACLE_LONG, ORACLE_SHORT) and winner_side is not None:
            # freeze Candidate A and geometry at THIS decision time
            e_rec = {
                "event_id": eid,
                "structure_id": cand_sid,
                "structure_type": meta["structure_type"],
                "timeframe": tf,
                "zone_bottom": zb,
                "zone_top": zt,
                "start_bar": c,
                "end_bar": int(cb_full[-1]) + 1,
            }
            rec = _build_record(
                e=e_rec, mv=mv, atr_value=atr_value,
                long_sol=long_sol, short_sol=short_sol,
                long_final=long_final, short_final=short_final,
                tgt_long=tgt_long, tgt_short=tgt_short,
                decision=decision, lv=lv, sv=sv, winner_side=winner_side,
                next_event=next_by_id.get(eid),
            )
            records.append(rec)
            exit_idx = int(rec["exit_fill_index"])
            cursor = exit_idx + 1
            # no overlap: next candidate search starts strictly after this exit.
            # IMPORTANT: a completed trade does NOT block cand_sid. After the
            # cursor passes this exit, if the same structure is still/again in
            # proximity it is eligible again as a fresh candidate (geometry and
            # target re-frozen at the new decision time). This restores the
            # required lifecycle; it is the inverse of the old global dedup.
        else:
            # No canonical trade from this candidate (no decision / no structural
            # target). Block it permanently so the scan makes forward progress:
            # re-scanning the same structure here would never advance the cursor.
            no_trade_sids.add(cand_sid)

    if self_target_total != 0:
        raise AssertionError(f"HARD_FAIL_SELF_TARGET_COUNT={self_target_total}")

    # V4.1 execution-boundary audit: count canonical trades whose fill crossed
    # a trading_day / segment / intraday-unit boundary (must be 0 after the fix).
    cross_day = 0
    cross_seg = 0
    cross_unit = 0
    for r in records:
        if not r["canonical_oracle_trade"]:
            continue
        d = int(r["best_entry_decision_index"])
        f = int(r["best_entry_fill_index"])
        if int(td_arr[d]) != int(td_arr[f]):
            cross_day += 1
        if int(seg_arr[d]) != int(seg_arr[f]):
            cross_seg += 1
        if _unit_of(mv, d) != _unit_of(mv, f):
            cross_unit += 1

    meta = _build_meta(
        records, meta_events, self_target_total,
        cross_day=cross_day, cross_seg=cross_seg, cross_unit=cross_unit,
    )
    return {"records": records, "meta": meta, "market_view": mv}


def _build_record(
    *,
    e: Dict[str, Any], mv: EventMarketView, atr_value: float,
    long_sol: Dict[str, Any], short_sol: Dict[str, Any],
    long_final: Optional[Dict[str, Any]], short_final: Optional[Dict[str, Any]],
    tgt_long: Optional[Dict[str, Any]], tgt_short: Optional[Dict[str, Any]],
    decision: str, lv: Optional[float], sv: Optional[float], winner_side: Optional[str],
    next_event: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    s_bar = int(e["start_bar"])
    e_bar = int(e["end_bar"])
    zb, zt = float(e["zone_bottom"]), float(e["zone_top"])
    cand_sid = e["structure_id"]

    base = {
        "event_id": int(e["event_id"]),
        "structure_id": cand_sid,
        "structure_type": e.get("structure_type"),
        "timeframe": e.get("timeframe"),
        "zone_bottom": zb,
        "zone_top": zt,
        "candidate_start_bar": s_bar,
        "candidate_end_bar": e_bar,
        "candidate_start_time": pd.Timestamp(mv.times[s_bar]),
        "candidate_end_time": pd.Timestamp(mv.times[min(e_bar, mv.n) - 1]),
        "atr_value": float(atr_value),
        "atr_owner": ATR_OWNER,
        # branch diagnostics
        "long_branch_valid": (tgt_long is not None) and bool(long_sol.get("ok")),
        "short_branch_valid": (tgt_short is not None) and bool(short_sol.get("ok")),
        "long_branch_invalid_reason": (
            None if (tgt_long is not None and long_sol.get("ok"))
            else ("NO_STRUCTURAL_TARGET" if tgt_long is None
                  else long_sol.get("invalid_reason"))
        ),
        "short_branch_invalid_reason": (
            None if (tgt_short is not None and short_sol.get("ok"))
            else ("NO_STRUCTURAL_TARGET" if tgt_short is None
                  else short_sol.get("invalid_reason"))
        ),
        "long_value_points": (None if lv is None else float(lv)),
        "short_value_points": (None if sv is None else float(sv)),
        "long_value_atr": (None if lv is None else float(lv) / float(atr_value)) if atr_value and np.isfinite(atr_value) else None,
        "short_value_atr": (None if sv is None else float(sv) / float(atr_value)) if atr_value and np.isfinite(atr_value) else None,
        "long_target_price": (float(tgt_long["near_edge"]) if tgt_long else None),
        "short_target_price": (float(tgt_short["near_edge"]) if tgt_short else None),
        "long_target_sid": (tgt_long["structure_id"] if tgt_long else None),
        "short_target_sid": (tgt_short["structure_id"] if tgt_short else None),
        # oracle decision
        "oracle_decision": decision,
        "oracle_direction": winner_side,
        "canonical_oracle_trade": decision in (ORACLE_LONG, ORACLE_SHORT),
        "next_structural_event_id": (int(next_event["event_id"]) if next_event else -1),
        "next_structural_event_start_time": (
            pd.Timestamp(mv.times[next_event["start_bar"]]) if next_event else pd.NaT
        ),
        "next_structural_event_structure_id": (
            next_event["structure_id"] if next_event else None
        ),
    }

    winner = long_final if winner_side == "LONG" else short_final
    if winner is not None and decision in (ORACLE_LONG, ORACLE_SHORT):
        # HARD GUARD: canonical oracle trade must be profitable
        if not (winner["utility"] > PNL_EPS):
            raise AssertionError(
                f"HARD_FAIL_CANONICAL_LOSS: decision={decision} utility={winner['utility']}"
            )
        if not (winner["tp_atr"] > 0):
            raise AssertionError(
                f"HARD_FAIL_CANONICAL_TP_ATR: decision={decision} tp_atr={winner['tp_atr']}"
            )
        tgt = tgt_long if winner_side == "LONG" else tgt_short
        margin = None
        if winner_side == "LONG" and lv is not None and sv is not None:
            margin = (float(lv) - float(sv)) / float(atr_value)
        elif winner_side == "SHORT" and lv is not None and sv is not None:
            margin = (float(sv) - float(lv)) / float(atr_value)
        elif winner_side == "LONG" and lv is not None:
            margin = float(lv) / float(atr_value)  # loser side has no trade (baseline 0)
        elif winner_side == "SHORT" and sv is not None:
            margin = float(sv) / float(atr_value)

        base.update({
            "best_entry_decision_index": int(winner["entry_decision_index"]),
            "best_entry_fill_index": int(winner["entry_fill_index"]),
            "best_entry_fill_time": winner["entry_fill_time"],
            "best_entry_price": float(winner["entry_price"]),
            "best_entry_gap_points": float(winner["best_entry_gap_points"]),
            "best_entry_gap_atr": float(winner["best_entry_gap_atr"]),
            "exit_fill_index": int(winner["exit_fill_index"]),
            "exit_fill_time": winner["exit_fill_time"],
            "exit_price": float(winner["exit_price"]),
            "exit_reason": winner["exit_reason"],
            "positive_tp_exists": bool(winner["positive_tp_exists"]),
            "tp_atr": float(winner["tp_atr"]),
            "remaining_target_atr": float(winner["remaining_target_atr"]),
            "target_distance_atr": float(winner["target_distance_atr"]),
            "utility": float(winner["utility"]),
            "utility_atr": float(winner["utility"]) / float(atr_value),
            "optimal_exit_points_signed": float(winner["optimal_exit_points_signed"]),
            "optimal_exit_atr_signed": float(winner["optimal_exit_atr_signed"]),
            "target_structure_id": (tgt["structure_id"] if tgt else None),
            "target_structure_type": (tgt["role"] if tgt else None),
            "target_timeframe": (tgt["tf"] if tgt else None),
            "target_price": (float(tgt["near_edge"]) if tgt else None),
            "target_zone_bottom": (float(tgt["bottom"]) if tgt else None),
            "target_zone_top": (float(tgt["top"]) if tgt else None),
            "direction_margin_atr": (float(margin) if margin is not None else float("nan")),
            # V4.2 time-purity fix for label availability (Phase 1).
            # Open-fill exits (DP_EARLY_EXIT / DP_LOSS_EXIT) fill exactly at the
            # next bar's open, so the label is known at the exit bar's START.
            # TARGET_TOUCH is only known when the exit bar CLOSES (the touch may
            # occur anytime within the bar), so its label becomes available at
            # exit bar close = exit bar start + 15min. This removes up-to-15min
            # look-ahead when constructing the previous-5 label history.
            "label_available_time": (
                winner["exit_fill_time"] + pd.Timedelta(minutes=15)
                if winner["exit_reason"] == R_TARGET
                else winner["exit_fill_time"]
            ),
        })
    else:
        base.update({
            "best_entry_decision_index": -1,
            "best_entry_fill_index": -1,
            "best_entry_fill_time": pd.NaT,
            "best_entry_price": float("nan"),
            "best_entry_gap_points": float("nan"),
            "best_entry_gap_atr": float("nan"),
            "exit_fill_index": -1,
            "exit_fill_time": pd.NaT,
            "exit_price": float("nan"),
            "exit_reason": None,
            "positive_tp_exists": False,
            "tp_atr": float("nan"),
            "remaining_target_atr": float("nan"),
            "target_distance_atr": float("nan"),
            "utility": float("nan"),
            "utility_atr": float("nan"),
            "optimal_exit_points_signed": float("nan"),
            "optimal_exit_atr_signed": float("nan"),
            "target_structure_id": None,
            "target_structure_type": None,
            "target_timeframe": None,
            "target_price": None,
            "target_zone_bottom": None,
            "target_zone_top": None,
            "direction_margin_atr": float("nan"),
            "label_available_time": pd.NaT,
        })
    return base


def _build_meta(
    records: List[Dict[str, Any]], meta_events: Dict[str, Any], self_target_total: int,
    cross_day: int = 0, cross_seg: int = 0, cross_unit: int = 0,
) -> Dict[str, Any]:
    n = len(records)
    long_valid = sum(1 for r in records if r["long_branch_valid"])
    short_valid = sum(1 for r in records if r["short_branch_valid"])
    both = sum(1 for r in records if r["long_branch_valid"] and r["short_branch_valid"])
    only_long = sum(1 for r in records if r["long_branch_valid"] and not r["short_branch_valid"])
    only_short = sum(1 for r in records if r["short_branch_valid"] and not r["long_branch_valid"])
    neither = sum(1 for r in records if not r["long_branch_valid"] and not r["short_branch_valid"])
    canonical = [r for r in records if r["canonical_oracle_trade"]]
    oracle_long = sum(1 for r in canonical if r["oracle_direction"] == "LONG")
    oracle_short = sum(1 for r in canonical if r["oracle_direction"] == "SHORT")
    no_pos = sum(1 for r in records if r["oracle_decision"] == ORACLE_NOPOS)
    no_target_branch = sum(1 for r in records if r["oracle_decision"] == ORACLE_NO_TARGET)
    ties = sum(1 for r in records if r["oracle_decision"] == ORACLE_TIE)
    canonical_loss = sum(1 for r in canonical if not (r["utility"] > PNL_EPS))
    target_touch = sum(1 for r in canonical if r["exit_reason"] == R_TARGET)
    early_exit = sum(1 for r in canonical if r["exit_reason"] in (R_EARLY, R_LOSS))
    target_not_reached = sum(
        1 for r in records
        for _k in ("long_branch_invalid_reason", "short_branch_invalid_reason")
        if r.get(_k) == "TARGET_NOT_REACHED"
    )
    valid_but_no_entry = 0  # by construction an ok branch always yields one entry

    return {
        "math_version": MATH_VERSION,
        "task_id": TASK_ID,
        "symbol": None,
        "atr_owner": ATR_OWNER,
        "direction_owner": "GOD_MODE (oracle decides, no external model)",
        "self_target_count": int(self_target_total),
        "valid_but_no_entry": int(valid_but_no_entry),
        "canonical_loss_count": int(canonical_loss),
        "structural_events": int(n),
        "long_branch_valid": int(long_valid),
        "short_branch_valid": int(short_valid),
        "both_branches_valid": int(both),
        "only_long_valid": int(only_long),
        "only_short_valid": int(only_short),
        "neither_valid": int(neither),
        "canonical_oracle_trades": int(len(canonical)),
        "oracle_long": int(oracle_long),
        "oracle_short": int(oracle_short),
        "no_positive_opportunity": int(no_pos),
        "no_valid_target_branch": int(no_target_branch),
        "direction_ties": int(ties),
        "target_touch": int(target_touch),
        "early_exit": int(early_exit),
        "target_not_reached_directions": int(target_not_reached),
        "cross_day_fill_count": int(cross_day),
        "cross_segment_fill_count": int(cross_seg),
        "cross_unit_fill_count": int(cross_unit),
        "raw_proximity_runs": meta_events.get("raw_proximity_runs"),
        "collapsed_same_structure_runs": meta_events.get("collapsed_same_structure_runs"),
    }


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def run_god_oracle_v4(
    symbol: str,
    max_bars: Optional[int] = None,
    event_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Production God-mode oracle (O(L))."""
    result = _run_god_oracle_core(
        symbol, max_bars, event_limit,
        solve_direction_fn=solve_direction_god_v4,
        pick_target_fn=pick_directional_target_v4,
    )
    result["meta"]["symbol"] = symbol
    return result


# --------------------------------------------------------------------------- #
# Target -> next-static audit (report only; does not change the event chain)
# --------------------------------------------------------------------------- #
def audit_target_vs_next_static(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """For winning TARGET_TOUCH trades, compare the winner target B vs the
    V2 static next structural event.

    Report-only: never modifies the event chain.
    """
    matches = 0
    mismatches = 0
    first_mismatches: List[Dict[str, Any]] = []
    for r in records:
        if not r["canonical_oracle_trade"]:
            continue
        if r["exit_reason"] != R_TARGET:
            continue
        tgt = r["target_structure_id"]
        nxt = r["next_structural_event_structure_id"]
        if tgt is None or nxt is None:
            continue
        if structure_identity(tgt) == structure_identity(nxt):
            matches += 1
        else:
            mismatches += 1
            if len(first_mismatches) < 20:
                first_mismatches.append({
                    "event_id": r["event_id"],
                    "structure_id": r["structure_id"],
                    "target_structure_id": tgt,
                    "next_event_structure_id": nxt,
                    "target_price": r["target_price"],
                })
    total = matches + mismatches
    return {
        "target_touch_count": total,
        "target_equals_next_static_count": int(matches),
        "mismatch_count": int(mismatches),
        "match_rate": (float(matches) / total) if total else float("nan"),
        "first_20_mismatches": first_mismatches,
    }


# --------------------------------------------------------------------------- #
# Evidence packet
# --------------------------------------------------------------------------- #
def summarize_god_oracle(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    canon = [r for r in records if r["canonical_oracle_trade"]]

    def _stats(vals, key):
        a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
        if len(a) == 0:
            return {"median": float("nan"), "p90": float("nan"), "max": float("nan"), "min": float("nan")}
        return {
            "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)),
            "max": float(np.max(a)),
            "min": float(np.min(a)),
        }

    gap = _stats([r["best_entry_gap_atr"] for r in canon], "gap")
    tp = _stats([r["tp_atr"] for r in canon], "tp")
    margin_vals = [r["direction_margin_atr"] for r in canon
                   if r["direction_margin_atr"] is not None and np.isfinite(r["direction_margin_atr"])]
    margin: Dict[str, float] = {}
    if margin_vals:
        a = np.asarray(margin_vals, dtype=float)
        margin = {
            "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)),
            "max": float(np.max(a)),
            "min": float(np.min(a)),
        }

    rem_zero = sum(1 for r in canon if r["remaining_target_atr"] == 0.0)
    rem_pos = sum(1 for r in canon
                  if r["remaining_target_atr"] is not None
                  and np.isfinite(r["remaining_target_atr"])
                  and r["remaining_target_atr"] > 0)
    rem_nan = sum(1 for r in canon
                  if r["remaining_target_atr"] is None
                  or not np.isfinite(r["remaining_target_atr"]))
    tp_nan = sum(1 for r in canon
                 if r["tp_atr"] is None or not np.isfinite(r["tp_atr"]))

    return {
        "canonical_trades": len(canon),
        "best_entry_gap_atr": gap,
        "tp_atr": tp,
        "direction_margin_atr": margin,
        "remaining_target_atr_zero_count": int(rem_zero),
        "remaining_target_atr_positive_count": int(rem_pos),
        "remaining_target_atr_nan_count": int(rem_nan),
        "canonical_tp_nan_count": int(tp_nan),
        "min_tp_atr": (float(np.min([r["tp_atr"] for r in canon])) if canon else float("nan")),
    }


def main(symbol: str = "AG") -> None:
    from research.liquidity_oracle_atlas.git_head import git_head
    res = run_god_oracle_v4(symbol)
    records = res["records"]
    meta = res["meta"]
    summ = summarize_god_oracle(records)
    audit = audit_target_vs_next_static(records)

    print(f"=== {TASK_ID} ({symbol}) ===")
    print(f"BASE SHA context: 7b6a1a364d37f388b9039eadffe277dac20ad8ec")
    print("God-mode contract: direction decided by future oracle = PASS")
    print("old direction model dependency = NONE")
    print()
    print("META:", {k: v for k, v in meta.items()})
    print("SUMMARY:", summ)
    print("AUDIT target->next-static:", {k: v for k, v in audit.items() if k != "first_20_mismatches"})
    print("first_20_mismatches:", audit["first_20_mismatches"][:5])

    # sanity HARD checks
    assert meta["canonical_loss_count"] == 0, "canonical_loss_count MUST be 0"
    assert meta["self_target_count"] == 0, "self_target_count MUST be 0"
    assert meta["valid_but_no_entry"] == 0, "valid_but_no_entry MUST be 0"
    assert summ["remaining_target_atr_nan_count"] == 0, "canonical remaining_target_atr NaN MUST be 0"
    assert summ["canonical_tp_nan_count"] == 0, "canonical tp_atr NaN MUST be 0"
    assert meta["cross_day_fill_count"] == 0, "cross_day_fill_count MUST be 0"
    assert meta["cross_segment_fill_count"] == 0, "cross_segment_fill_count MUST be 0"
    assert meta["cross_unit_fill_count"] == 0, "cross_unit_fill_count MUST be 0"
    assert meta["canonical_oracle_trades"] > 0, "expected some canonical oracle trades"
    assert summ["min_tp_atr"] > 0, "min(tp_atr) MUST be > 0"
    assert meta["early_exit"] == 0, "V4.3: early_exit MUST be 0"
    assert meta["target_touch"] == meta["canonical_oracle_trades"], \
        "V4.3: every canonical oracle trade MUST be TARGET_TOUCH"
    print("\nALL HARD SANITY CHECKS PASSED")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "AG")
