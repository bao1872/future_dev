"""structural_god_oracle_reference_m15_v4
=========================================

INDEPENDENT brute-force reference for FUT-M15-STRUCTURAL-GOD-ORACLE-V4.

Deliberately shares NO solver / target-picker code with
``structural_god_oracle_m15_v4``. It re-implements the directional target
selection and the entry/exit solver as plain nested loops so a bug in the
production O(L) backwards DP cannot hide behind a shared helper.

Complexity: O(L^2) per direction (every legal contact entry x every legal exit).
Slow is fine -- it only runs on synthetic fixtures and small real slices for the
parity check.

The orchestration glue (event iteration, decision, record building) is reused
from the production module via ``_run_god_oracle_core``; only the SOLVER and the
TARGET PICKER are independently re-implemented here, exactly as the spec
requires.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1 import (
    bar_zone_distance,
    select_target,
)
from research.liquidity_oracle_atlas.structural_god_oracle_m15_v4 import (
    LONG_ROLES,
    PNL_EPS,
    REM_EPS,
    PLACE_EPS,
    R_TARGET,
    R_EARLY,
    R_LOSS,
    _run_god_oracle_core,
)

PNL_EPS_R = PNL_EPS
REM_EPS_R = REM_EPS
PLACE_EPS_R = PLACE_EPS


# --------------------------------------------------------------------------- #
# Structural identity helpers (re-implemented, independent)
# --------------------------------------------------------------------------- #
def _parse_sr(sid: str) -> Tuple[str, str, str, str]:
    p = sid.split("|")
    return ("SR", p[1], p[2], p[4], p[5])


def _parse_liq(sid: str) -> Tuple[str, str, str, str, str, str]:
    p = sid.split("|")
    return ("LIQ", p[1], p[2], p[3], p[4], p[5])


def target_is_self_reference(target_sid: str, candidate_sid: str) -> bool:
    if candidate_sid.startswith("SR|"):
        if not target_sid.startswith("SR|"):
            return False
        c = _parse_sr(candidate_sid)
        t = _parse_sr(target_sid)
        return (
            c[1] == t[1] and c[2] == t[2]
            and abs(float(c[3]) - float(t[3])) <= 1e-6
            and abs(float(c[4]) - float(t[4])) <= 1e-6
        )
    if candidate_sid.startswith("LIQ|"):
        if not target_sid.startswith("LIQ|"):
            return False
        c = _parse_liq(candidate_sid)
        t = _parse_liq(target_sid)
        return (
            c[1] == t[1] and c[2] == t[2] and c[3] == t[3]
            and c[4] == t[4] and float(c[5]) == float(t[5])
        )
    return False


def pick_directional_target_v4_ref(
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
    """Independent re-implementation of the directional target picker.

    Same contract as the production ``pick_directional_target_v4``: LONG ->
    RESISTANCE / BUYSIDE_LIQUIDITY, SHORT -> SUPPORT / SELLSIDE_LIQUIDITY, with
    self-target exclusion and directional placement. Must NOT call the production
    picker.
    """
    roles = LONG_ROLES if direction == "LONG" else ("SUPPORT", "SELLSIDE_LIQUIDITY")
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
            if target_is_self_reference(c["structure_id"], candidate_sid):
                continue
            if direction == "LONG":
                if not (float(c["near_edge"]) > float(zone_top) + PLACE_EPS_R):
                    continue
            else:
                if not (float(c["near_edge"]) < float(zone_bottom) - PLACE_EPS_R):
                    continue
            c["tf"] = tfk
            cands.append(c)
    if not cands:
        return None
    return min(cands, key=lambda x: abs(float(x["near_edge"]) - ref))


# --------------------------------------------------------------------------- #
# Independent unit-end lookup + brute-force solver
# --------------------------------------------------------------------------- #
def _ref_unit_end(unit_starts: np.ndarray, bar: int, n: int) -> int:
    last = 0
    for u in unit_starts:
        u = int(u)
        if u <= bar:
            last = u
        else:
            return u - 1
    return int(n) - 1


def _ref_unit_of(unit_starts: np.ndarray, bar: int) -> int:
    """Index of the intraday unit containing bar (independent of production)."""
    return int(np.searchsorted(unit_starts, int(bar), side="right")) - 1


def solve_direction_god_reference_v4(
    *,
    direction: str,
    zone_bottom: float,
    zone_top: float,
    atr_value: float,
    start_bar: int,
    end_bar: int,
    target_price: Optional[float],
    contact_bars: List[int],
    mv: Any,
    trading_day: Optional[np.ndarray] = None,
    segment: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Brute force: enumerate every legal (entry, exit) pair, take the best. O(L^2).

    V4.1 execution-boundary gate: a ``decision=t -> fill=t+1`` candidate is
    rejected unless ``t`` and ``t+1`` share the same trading_day, segment and
    intraday unit. Independently re-implemented (no call into production).
    """
    sign = 1.0 if direction == "LONG" else -1.0
    opens = np.asarray(mv.opens, dtype=float)
    highs = np.asarray(mv.highs, dtype=float)
    lows = np.asarray(mv.lows, dtype=float)
    n = int(mv.n)
    unit_starts = np.asarray(mv.unit_starts, dtype=np.int64)

    s_bar = int(start_bar)
    e_bar = int(min(end_bar, n))
    hi = min(e_bar, n - 1)
    n_contact = len(contact_bars)

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
        # V4.1 execution-boundary gate (independent re-implementation)
        if (trading_day is not None and segment is not None
                and (int(trading_day[d]) != int(trading_day[f])
                     or int(segment[d]) != int(segment[f])
                     or _ref_unit_of(unit_starts, d) != _ref_unit_of(unit_starts, f))):
            n_rejected += 1
            continue
        H = min(e_bar, _ref_unit_end(unit_starts, f, n), n - 1)
        entry_price = float(opens[f])

        # independent first-target-touch scan measured from the EVENT START
        first_touch = None
        if target_price is not None:
            tp = float(target_price)
            for t in range(s_bar, hi + 1):
                if direction == "LONG" and float(highs[t]) >= tp:
                    first_touch = t
                    break
                if direction == "SHORT" and float(lows[t]) <= tp:
                    first_touch = t
                    break

        same_bar_target = False
        if first_touch is not None:
            if first_touch < f:
                n_rejected += 1
                continue
            if first_touch == f:
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

        if same_bar_target:
            exit_fill = f
            exit_price = float(target_price)
            pnl = sign * (float(target_price) - entry_price)
            reason = R_TARGET
        elif first_touch is not None and first_touch > f and first_touch <= H:
            exit_fill = int(first_touch)
            exit_price = float(target_price)
            pnl = sign * (float(target_price) - entry_price)
            reason = R_TARGET
        else:
            # V4.3: no early exit. A direction whose structural target is never
            # touched inside its intraday unit does not open -> reject.
            continue

        cand = {
            "entry_decision_index": int(d),
            "entry_fill_index": int(f),
            "entry_price": entry_price,
            "exit_fill_index": exit_fill,
            "exit_price": exit_price,
            "exit_reason": reason,
            "utility": float(pnl),
        }
        if best is None or cand["utility"] > best["utility"]:
            best = cand

    if best is None:
        reason = "NO_EXECUTABLE_ENTRY" if n_candidates == 0 else "TARGET_NOT_REACHED"
        return {
            "ok": False, "invalid_reason": reason,
            "n_candidates": n_candidates, "n_with_path": n_with_path,
            "n_rejected_target_before_entry": n_rejected, "n_contact": n_contact,
        }

    if best["exit_reason"] == R_TARGET and best["exit_fill_index"] == e_bar and e_bar < n:
        return {
            "ok": False, "invalid_reason": "AMBIGUOUS_SAME_BAR_TERMINAL",
            "n_candidates": n_candidates, "n_with_path": n_with_path,
            "n_rejected_target_before_entry": n_rejected, "n_contact": n_contact,
        }

    # metrics (duplicated on purpose; independent of the production finalize)
    zb, zt = float(zone_bottom), float(zone_top)
    ep = float(best["entry_price"])
    xp = float(best["exit_price"])
    pnl = float(best["utility"])

    gap_pts = bar_zone_distance(ep, ep, zb, zt)
    gap_atr = gap_pts / float(atr_value)

    def _dir(a: float, b: float) -> float:
        return (b - a) if direction == "LONG" else (a - b)

    positive = pnl > PNL_EPS_R
    if best["exit_reason"] == R_TARGET:
        if not positive:
            raise AssertionError(
                f"HARD_FAIL_TARGET_TOUCH_WITH_LOSS: entry={ep} target={xp} pnl={pnl}"
            )
        tp_atr = _dir(ep, xp) / float(atr_value)
        remaining = 0.0
        remaining_raw = 0.0
    elif positive:
        tp_atr = _dir(ep, xp) / float(atr_value)
        if target_price is None:
            remaining = float("nan")
            remaining_raw = float("nan")
        else:
            raw = _dir(xp, float(target_price)) / float(atr_value)
            remaining_raw = raw
            if raw < -REM_EPS_R:
                raise AssertionError("HARD_FAIL_REMAINING_TARGET_NEGATIVE")
            remaining = raw
    else:
        tp_atr = float("nan")
        remaining = float("nan")
        remaining_raw = (
            float("nan") if target_price is None
            else _dir(xp, float(target_price)) / float(atr_value)
        )

    target_distance_atr = (
        _dir(ep, float(target_price)) / float(atr_value)
        if target_price is not None else float("nan")
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
# Public entry point (reuses the production orchestration core with the
# independent solver + target picker)
# --------------------------------------------------------------------------- #
def run_god_oracle_v4_reference(
    symbol: str,
    max_bars: Optional[int] = None,
    event_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Independent O(L^2) reference run for parity against production."""
    result = _run_god_oracle_core(
        symbol, max_bars, event_limit,
        solve_direction_fn=solve_direction_god_reference_v4,
        pick_target_fn=pick_directional_target_v4_ref,
    )
    result["meta"]["symbol"] = symbol
    result["meta"]["solver"] = "reference_O(L^2)"
    return result
