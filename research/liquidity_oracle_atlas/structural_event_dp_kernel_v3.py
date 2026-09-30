"""structural_event_dp_kernel_v3
===============================

FUT-M15-STRUCTURAL-DP-V3-KERNEL-PLUGGABLE

A **direction-pluggable** structural-event DP kernel.

The kernel answers exactly ONE question:

    given a structural event + a KNOWN direction
        -> what is the best ENTRY and the best EXIT?

It does NOT decide the direction, and it is deliberately blind to where a
direction came from: this module never imports a direction model, never reads
E9 / A9 / the legacy oracle, and never infers LONG from "support" or SHORT from
"resistance". Direction arrives only through :class:`DirectionInput`.

Frozen semantics inherited from V2 (NOT re-derived here)
--------------------------------------------------------
  * structural event identity (same structure = one event, even across a
    temporary loss of proximity)  -> build_structural_dp_labels_m15_v2
  * candidate geometry frozen when the event starts
  * target = nearest eligible SR / Liquidity target ahead of the direction
  * target touch is a HARD terminal (TP == Target, remaining == 0)

What V3 changes vs V2
---------------------
  * V2 borrowed its entry from the legacy hindsight oracle; V3 computes the
    entry itself, jointly with the exit, inside the structural event.
  * A structurally/execution-valid event ALWAYS yields exactly one best entry
    (`argmax_j Utility(j)`), even if every candidate loses money. There is no
    NO_ENTRY, no armed quota, no one-entry-per-proximity-episode quota and no
    competition for an entry budget with other events.

Objective (frozen, unchanged)
-----------------------------
    gross open-to-open PnL, zero cost
    decision at close(t) -> fill at open(t+1)

Terminal priority inside one bar (T0-O)
---------------------------------------
    target touch -> EXIT -> label completes -> next structural event may begin
If the earliest target touch falls on the very bar where the NEXT structural
event starts and OHLC cannot order the two, the kernel refuses to guess and
returns `AMBIGUOUS_SAME_BAR_TERMINAL`.

Tie-breaks (frozen, applied identically in Reference and Production)
--------------------------------------------------------------------
    entry tie -> earlier entry decision index
    exit  tie -> earlier exit bar
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.build_structural_dp_labels_m15_v2 import (
    _pick_target,
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
)

MATH_VERSION = "structural-event-dp-v3-pluggable"
TASK_ID = "FUT-M15-STRUCTURAL-DP-V3-KERNEL-PLUGGABLE"
ATR_OWNER = "m15_atr@run_environment_m15"

VALID_DIRECTIONS = ("LONG", "SHORT")
BAR_MINUTES = 15
PNL_EPS = 1e-9
REM_EPS = 1e-9

# exit reasons
R_TARGET = "TARGET_TOUCH"
R_EARLY = "DP_EARLY_EXIT"
R_LOSS = "DP_LOSS_EXIT"

# invalid reasons ("cannot be constructed", never "not worth trading")
INVALID_REASONS = (
    "NO_DIRECTION_INPUT",
    "DIRECTION_NOT_AVAILABLE_YET",
    "INVALID_DIRECTION_VALUE",
    "BAD_GEOMETRY",
    "BAD_ATR",
    "NO_TARGET",
    "NO_EXECUTABLE_ENTRY",
    "INSUFFICIENT_PATH",
    "AMBIGUOUS_SAME_BAR_TERMINAL",
)


# --------------------------------------------------------------------------- #
# Direction contract
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DirectionInput:
    """External direction decision handed to the kernel.

    The kernel reads ``direction`` only. ``available_time`` is the moment the
    direction became legally usable and MUST satisfy
    ``available_time <= event_decision_time`` (the close of the event's first
    bar). ``source_id`` records provenance; the kernel never interprets it.
    """

    event_id: int
    direction: str
    direction_score: Optional[float] = None
    prediction_time: Optional[Any] = None
    available_time: Optional[Any] = None
    source_id: str = ""

    def validate(self) -> Optional[str]:
        if self.direction not in VALID_DIRECTIONS:
            return "INVALID_DIRECTION_VALUE"
        return None


def alternating_direction_fixture(event_id: int, **_: Any) -> DirectionInput:
    """Diagnostic-ONLY fixture: LONG / SHORT / LONG / SHORT ...

    Explicitly NOT a direction model. Allowed only for synthetic T0 tests and
    for the non-canonical real-data diagnostic subset.
    """
    return DirectionInput(
        event_id=int(event_id),
        direction="LONG" if int(event_id) % 2 == 0 else "SHORT",
        direction_score=None,
        prediction_time=None,
        available_time=None,
        source_id="ALTERNATING_FIXTURE_NONCANONICAL",
    )


# --------------------------------------------------------------------------- #
# Market view
# --------------------------------------------------------------------------- #
@dataclass
class EventMarketView:
    """Read-only 15m bar arrays + intraday-unit boundaries."""

    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    times: np.ndarray
    unit_starts: np.ndarray
    n: int
    symbol: str = ""

    def unit_end(self, bar: int) -> int:
        """Last bar index of the intraday unit containing ``bar`` (O(log U))."""
        i = int(np.searchsorted(self.unit_starts, int(bar), side="right")) - 1
        if i + 1 < len(self.unit_starts):
            return int(self.unit_starts[i + 1]) - 1
        return int(self.n) - 1


def market_view_from_arrays(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    times: Optional[Sequence[Any]] = None,
    unit_starts: Optional[Sequence[int]] = None,
    symbol: str = "",
) -> EventMarketView:
    """Build a market view (used by T0 synthetic tests and real-data runs)."""
    o = np.asarray(opens, dtype=float)
    h = np.asarray(highs, dtype=float)
    l = np.asarray(lows, dtype=float)
    c = np.asarray(closes, dtype=float)
    n = len(o)
    if times is None:
        t = (np.arange(n).astype("timedelta64[m]")
             + np.datetime64("1970-01-01T00:00", "m")).astype("datetime64[ns]")
    else:
        t = np.asarray(pd.to_datetime(pd.Series(list(times))).to_numpy(), dtype="datetime64[ns]")
    if unit_starts is None:
        us = np.zeros(1, dtype=np.int64)  # one single unit covering everything
    else:
        us = np.asarray(sorted(int(x) for x in unit_starts), dtype=np.int64)
    return EventMarketView(opens=o, highs=h, lows=l, closes=c, times=t,
                           unit_starts=us, n=n, symbol=symbol)


# --------------------------------------------------------------------------- #
# Production kernel: O(L) per event
# --------------------------------------------------------------------------- #
def _suffix_argmax_per_unit(
    mv: EventMarketView, lo: int, hi: int, sign: float
) -> np.ndarray:
    """``suf[k]`` = argmax of ``sign*open`` over [k .. segment_end], earliest tie.

    Segments are maximal runs inside [lo, hi] belonging to the same intraday
    unit, so an exit can never be selected outside the entry's own session /
    day unit. Single backward pass -> O(hi - lo + 1).
    """
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
            # backward scan: on an equal value the CURRENT (earlier) index must
            # win, otherwise the suffix would drift to the LATER bar and break
            # the frozen "exit tie -> earlier exit" rule.
            if v >= bestv:
                bestv = v
                best = j
            suf[j] = best
        k = a - 1
    return suf


def _next_target_touch(
    mv: EventMarketView, lo: int, hi: int, direction: str, target_price: Optional[float]
) -> np.ndarray:
    """``nt[t]`` = earliest bar in [t .. hi] where the frozen target is touched.

    -1 when the target is never touched within the window. O(hi - lo + 1).
    """
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


def solve_event_production_v3(
    *,
    direction: str,
    zone_bottom: float,
    zone_top: float,
    atr_value: float,
    start_bar: int,
    end_bar: int,
    target_price: Optional[float],
    mv: EventMarketView,
) -> Dict[str, Any]:
    """Joint entry+exit DP for ONE structural event. O(L).

    ``end_bar`` is EXCLUSIVE (start bar of the next structural event, or n).
    Legal exit range is capped so that ``exit <= end_bar`` (the next event's
    first bar), i.e. ``Exit_i <= Start(E_{i+1})``.
    """
    sign = 1.0 if direction == "LONG" else -1.0
    n = mv.n
    s_bar = int(start_bar)
    e_bar = int(min(end_bar, n)) if end_bar is not None else n

    hi = min(e_bar, n - 1)
    if hi < s_bar:
        return {"ok": False, "invalid_reason": "NO_EXECUTABLE_ENTRY"}

    # precompute O(L)
    suf = _suffix_argmax_per_unit(mv, s_bar + 1, hi, sign)
    nt = _next_target_touch(mv, s_bar + 1, hi, direction, target_price)

    n_candidates = 0
    n_with_path = 0
    best: Optional[Dict[str, Any]] = None

    d_hi = min(e_bar - 2, n - 2)
    for d in range(s_bar, d_hi + 1):
        n_candidates += 1
        f = d + 1
        H = min(e_bar, mv.unit_end(f), n - 1)
        if f + 1 > H:
            continue                      # no room for any legal exit
        n_with_path += 1
        entry_price = float(mv.opens[f])
        pnl = None
        exit_fill = None
        exit_price = None
        reason = None

        tt = int(nt[f]) if f < n else -1
        if target_price is not None and tt >= 0 and tt <= H:
            exit_fill = tt
            exit_price = float(target_price)
            pnl = sign * (float(target_price) - entry_price)
            reason = R_TARGET
        else:
            k = int(suf[f + 1]) if (f + 1) < n else -1
            if k < 0 or k > H:
                continue
            exit_price = float(mv.opens[k])
            pnl = sign * (exit_price - entry_price)
            exit_fill = k
            reason = R_EARLY if pnl > PNL_EPS else R_LOSS

        if best is None or pnl > best["utility"]:   # strict -> earlier entry wins ties
            best = {
                "entry_decision_index": int(d),
                "entry_fill_index": int(f),
                "entry_price": entry_price,
                "exit_fill_index": int(exit_fill),
                "exit_price": float(exit_price),
                "exit_reason": reason,
                "utility": float(pnl),
            }

    if best is None:
        reason = "NO_EXECUTABLE_ENTRY" if n_candidates == 0 else "INSUFFICIENT_PATH"
        return {"ok": False, "invalid_reason": reason,
                "n_candidates": n_candidates, "n_with_path": n_with_path}

    # terminal-priority ambiguity: target first touched exactly on the bar where
    # the NEXT structural event starts -> OHLC cannot order them -> do not guess.
    if (best["exit_reason"] == R_TARGET and best["exit_fill_index"] == e_bar
            and e_bar < n):
        return {"ok": False, "invalid_reason": "AMBIGUOUS_SAME_BAR_TERMINAL",
                "n_candidates": n_candidates, "n_with_path": n_with_path}

    return _finalize_solution(
        direction=direction, zone_bottom=zone_bottom, zone_top=zone_top,
        atr_value=atr_value, target_price=target_price, mv=mv, best=best,
        n_candidates=n_candidates, n_with_path=n_with_path,
    )


def _finalize_solution(
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
) -> Dict[str, Any]:
    sign = 1.0 if direction == "LONG" else -1.0

    def _directional(a: float, b: float) -> float:
        return (b - a) if direction == "LONG" else (a - b)

    entry_price = float(best["entry_price"])
    exit_price = float(best["exit_price"])
    pnl = float(best["utility"])

    gap_pts = bar_zone_distance(entry_price, entry_price,
                                float(zone_bottom), float(zone_top))
    gap_atr = gap_pts / float(atr_value)

    positive = pnl > PNL_EPS
    if best["exit_reason"] == R_TARGET:
        remaining = 0.0
        remaining_raw = 0.0
        # §12 literal: a non-positive exit reports tp_atr = NA, even when the
        # terminal was a target touch (the frozen target can sit BEHIND the
        # entry because the target is frozen at event start, not at entry).
        tp_atr = (_directional(entry_price, exit_price) / float(atr_value)
                  if positive else float("nan"))
    elif positive:
        tp_atr = _directional(entry_price, exit_price) / float(atr_value)
        if target_price is None:
            remaining = float("nan")
            remaining_raw = float("nan")
        else:
            raw = _directional(exit_price, float(target_price)) / float(atr_value)
            remaining_raw = raw
            if raw < -REM_EPS:
                raise AssertionError(
                    "HARD_FAIL_REMAINING_TARGET_NEGATIVE: "
                    f"raw_remaining={raw} (must never be clamped)"
                )
            remaining = raw
    else:
        tp_atr = float("nan")
        remaining = float("nan")
        remaining_raw = (
            float("nan") if target_price is None
            else _directional(exit_price, float(target_price)) / float(atr_value)
        )

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
        "optimal_exit_points_signed": float(pnl),
        "optimal_exit_atr_signed": float(pnl) / float(atr_value),
        "n_candidates": int(n_candidates),
        "n_with_path": int(n_with_path),
        "entry_fill_time": pd.Timestamp(mv.times[best["entry_fill_index"]]),
        "exit_fill_time": pd.Timestamp(mv.times[best["exit_fill_index"]]),
    })
    return out


# --------------------------------------------------------------------------- #
# Event-level orchestration (validity + record)
# --------------------------------------------------------------------------- #
def evaluate_event_v3(
    event: Dict[str, Any],
    mv: EventMarketView,
    direction_input: Optional[DirectionInput],
    atr_value: float,
    target: Optional[Dict[str, Any]],
    *,
    solver: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Validate one structural event and (if valid) solve its entry/exit.

    Validity = "can this label be constructed at all?" — never "is it worth
    trading?".
    """
    solve = solver or solve_event_production_v3
    s_bar = int(event["start_bar"])
    e_bar = int(event.get("end_bar", mv.n))
    ev_id = int(event["event_id"])

    base = {
        "event_id": ev_id,
        "structure_id": event.get("structure_id"),
        "structure_type": event.get("structure_type"),
        "timeframe": event.get("timeframe"),
        "zone_bottom": float(event["zone_bottom"]),
        "zone_top": float(event["zone_top"]),
        "event_start_bar": s_bar,
        "event_end_bar": e_bar,
        "event_start_time": pd.Timestamp(mv.times[s_bar]),
        "event_end_time": pd.Timestamp(mv.times[min(e_bar, mv.n - 1)]),
    }

    def _invalid(reason: str, **extra) -> Dict[str, Any]:
        rec = dict(base)
        rec.update({
            "event_valid": False,
            "invalid_reason": reason,
            "entry_present": False,
            "direction": None,
            "target_price": None,
            "best_entry_fill_index": -1,
            "exit_fill_index": -1,
        })
        rec.update(extra)
        return rec

    # 1) geometry
    zb, zt = float(event["zone_bottom"]), float(event["zone_top"])
    if not (np.isfinite(zb) and np.isfinite(zt)) or zb > zt:
        return _invalid("BAD_GEOMETRY")

    # 2) direction
    if direction_input is None:
        return _invalid("NO_DIRECTION_INPUT")
    dv = direction_input.validate()
    if dv:
        return _invalid(dv, direction=str(direction_input.direction))
    event_decision_time = pd.Timestamp(mv.times[s_bar]) + pd.Timedelta(minutes=BAR_MINUTES)
    if direction_input.available_time is not None:
        if pd.Timestamp(direction_input.available_time) > event_decision_time:
            return _invalid(
                "DIRECTION_NOT_AVAILABLE_YET",
                available_time=pd.Timestamp(direction_input.available_time),
                event_decision_time=event_decision_time,
            )

    # 3) ATR
    if not (np.isfinite(atr_value) and float(atr_value) > 0):
        return _invalid("BAD_ATR", atr_value=float(atr_value))

    # 4) target
    target_price = None if target is None else float(target["near_edge"])
    if target is None:
        return _invalid("NO_TARGET")

    sol = solve(
        direction=direction_input.direction,
        zone_bottom=zb, zone_top=zt, atr_value=float(atr_value),
        start_bar=s_bar, end_bar=e_bar, target_price=target_price, mv=mv,
    )
    if not sol.get("ok", False):
        return _invalid(sol.get("invalid_reason", "NO_EXECUTABLE_ENTRY"),
                        n_candidates=sol.get("n_candidates", 0))

    rec = dict(base)
    rec.update({
        "event_valid": True,
        "invalid_reason": None,
        "entry_present": True,
        "direction": direction_input.direction,
        "direction_score": direction_input.direction_score,
        "direction_source_id": direction_input.source_id,
        "direction_available_time": (
            None if direction_input.available_time is None
            else pd.Timestamp(direction_input.available_time)
        ),
        "target_structure_id": target.get("structure_id"),
        "target_structure_type": target.get("role"),
        "target_timeframe": target.get("tf"),
        "target_price": target_price,
        "atr_value": float(atr_value),
        "best_entry_decision_index": int(sol["entry_decision_index"]),
        "best_entry_fill_index": int(sol["entry_fill_index"]),
        "best_entry_time": sol["entry_fill_time"],
        "best_entry_price": float(sol["entry_price"]),
        "best_entry_gap_atr": float(sol["best_entry_gap_atr"]),
        "best_entry_gap_points": float(sol["best_entry_gap_points"]),
        "exit_fill_index": int(sol["exit_fill_index"]),
        "exit_time": sol["exit_fill_time"],
        "exit_price": float(sol["exit_price"]),
        "exit_reason": sol["exit_reason"],
        "positive_tp_exists": bool(sol["positive_tp_exists"]),
        "tp_atr": float(sol["tp_atr"]),
        "remaining_target_atr": float(sol["remaining_target_atr"]),
        "remaining_target_atr_raw": float(sol["remaining_target_atr_raw"]),
        "optimal_exit_points_signed": float(sol["optimal_exit_points_signed"]),
        "optimal_exit_atr_signed": float(sol["optimal_exit_atr_signed"]),
        "utility": float(sol["utility"]),
        "n_entry_candidates": int(sol["n_candidates"]),
        "n_candidates_with_path": int(sol["n_with_path"]),
        "label_available_time": sol["exit_fill_time"],
    })
    return rec


def event_has_executable_path(mv: EventMarketView, start_bar: int, end_bar: int) -> bool:
    """Direction-free check: does this event admit at least one entry with room
    for a legal exit? Used for structural-only accounting (before direction)."""
    n = mv.n
    e_bar = int(min(end_bar, n))
    for d in range(int(start_bar), min(e_bar - 2, n - 2) + 1):
        f = d + 1
        H = min(e_bar, mv.unit_end(f), n - 1)
        if f + 1 <= H:
            return True
    return False


def _unit_starts_from_arrays(td: np.ndarray, seg: np.ndarray) -> np.ndarray:
    n = len(seg)
    starts = []
    for i in range(n):
        if i == 0 or td[i] != td[i - 1] or seg[i] != seg[i - 1]:
            starts.append(i)
    return np.asarray(starts, dtype=np.int64)


def build_market_view(symbol: str, max_bars: Optional[int] = None) -> EventMarketView:
    prox = build_dp_proximity_m15(symbol, max_bars)
    td = pd.to_datetime(prox["trading_day"]).to_numpy()
    seg = prox["segment"].to_numpy(np.int64)
    return market_view_from_arrays(
        opens=prox["open"].to_numpy(float),
        highs=prox["high"].to_numpy(float),
        lows=prox["low"].to_numpy(float),
        closes=prox["close"].to_numpy(float),
        times=prox["bar_start_time"].to_numpy(),
        unit_starts=_unit_starts_from_arrays(td, seg),
        symbol=symbol,
    )


def run_structural_event_dp_v3(
    symbol: str,
    direction_provider: Callable[[Dict[str, Any], pd.Timestamp], Optional[DirectionInput]],
    max_bars: Optional[int] = None,
    event_limit: Optional[int] = None,
    solver: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run V3 over the real structural-event sequence of ``symbol``.

    ``direction_provider(event, event_decision_time) -> DirectionInput | None``
    is the ONLY way direction enters the kernel.
    """
    ev = build_structural_events_v2(symbol, max_bars)
    events: List[Dict[str, Any]] = ev["events"]
    meta_events = ev["meta"]
    if event_limit is not None:
        events = events[: int(event_limit)]

    mv = build_market_view(symbol, max_bars)
    env = run_environment_m15(symbol, max_bars, capture_provenance=False)
    atr_series = np.asarray(env["features"]["m15_atr"].to_numpy(float))
    geom = env["geom_by_decision"]
    prox_seg = None
    if len(mv.times):
        prox_seg = build_dp_proximity_m15(symbol, max_bars)["segment"].to_numpy(np.int64)

    sr_first_seen: Dict[Any, int] = {}
    records: List[Dict[str, Any]] = []
    for e in events:
        s_bar = int(e["start_bar"])
        atr_value = float(atr_series[s_bar]) if s_bar < len(atr_series) else float("nan")
        event_decision_time = pd.Timestamp(mv.times[s_bar]) + pd.Timedelta(minutes=BAR_MINUTES)
        d_in = direction_provider(e, event_decision_time)

        target = None
        if d_in is not None and d_in.validate() is None:
            ref = float(mv.closes[s_bar - 1]) if s_bar >= 1 else float(mv.opens[s_bar])
            geom_prev = geom[s_bar - 1] if s_bar >= 1 else None
            target = _pick_target(
                geom_prev, ref,
                int(prox_seg[s_bar]) if prox_seg is not None else 0,
                s_bar, sr_first_seen, d_in.direction,
            )
        records.append(evaluate_event_v3(e, mv, d_in, atr_value, target, solver=solver))

    valid = [r for r in records if r["event_valid"]]
    meta = {
        "math_version": MATH_VERSION,
        "task_id": TASK_ID,
        "symbol": symbol,
        "atr_owner": ATR_OWNER,
        "direction_owner": "PLUGGABLE (kernel never decides direction)",
        "tie_break_entry": "earlier entry decision index",
        "tie_break_exit": "earlier exit bar",
        "structural_events": int(len(records)),
        "valid_events": int(len(valid)),
        "invalid_events": int(len(records) - len(valid)),
        "entered_valid_events": int(len(valid)),   # valid => exactly one entry
        "valid_but_no_entry": 0,
        "invalid_reason_counts": {
            k: int(sum(1 for r in records if r.get("invalid_reason") == k))
            for k in INVALID_REASONS
        },
        "positive_tp": int(sum(1 for r in valid if r["positive_tp_exists"])),
        "no_positive_tp": int(sum(1 for r in valid if not r["positive_tp_exists"])),
        "target_reached": int(sum(1 for r in valid if r["exit_reason"] == R_TARGET)),
        "target_reached_with_loss": int(
            sum(1 for r in valid
                if r["exit_reason"] == R_TARGET and not r["positive_tp_exists"])
        ),
        "early_exit": int(sum(1 for r in valid if r["exit_reason"] == R_EARLY)),
        "raw_proximity_runs": meta_events.get("raw_proximity_runs"),
        "collapsed_same_structure_runs": meta_events.get("collapsed_same_structure_runs"),
    }
    return {"records": records, "meta": meta, "market_view": mv}
