"""structural_event_dp_reference_v3
==================================

INDEPENDENT brute-force reference kernel for FUT-M15-STRUCTURAL-DP-V3.

Deliberately shares NO code with the production kernel: it re-implements the
unit-end lookup, the target-touch scan and the exit argmax as plain nested
loops, so a bug in the production backwards DP / suffix-argmax cannot hide
behind a shared helper.

Complexity: O(L^2) per event (every legal entry x every legal exit).
Slow is fine — it only runs on synthetic fixtures and small real slices.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

PNL_EPS = 1e-9
R_TARGET = "TARGET_TOUCH"
R_EARLY = "DP_EARLY_EXIT"
R_LOSS = "DP_LOSS_EXIT"


def _ref_unit_end(unit_starts: np.ndarray, bar: int, n: int) -> int:
    """Independent unit-end lookup: plain forward scan (no searchsorted)."""
    last = 0
    for u in unit_starts:
        u = int(u)
        if u <= bar:
            last = u
        else:
            return u - 1
    return int(n) - 1


def solve_event_reference_v3(
    *,
    direction: str,
    zone_bottom: float,
    zone_top: float,
    atr_value: float,
    start_bar: int,
    end_bar: int,
    target_price: Optional[float],
    mv: Any,
) -> Dict[str, Any]:
    """Brute force: enumerate every legal (entry, exit) pair, take the best."""
    sign = 1.0 if direction == "LONG" else -1.0
    opens = np.asarray(mv.opens, dtype=float)
    highs = np.asarray(mv.highs, dtype=float)
    lows = np.asarray(mv.lows, dtype=float)
    n = int(mv.n)
    unit_starts = np.asarray(mv.unit_starts, dtype=np.int64)

    s_bar = int(start_bar)
    e_bar = int(min(end_bar, n))
    hi = min(e_bar, n - 1)

    n_candidates = 0
    n_with_path = 0
    best: Optional[Dict[str, Any]] = None

    for d in range(s_bar, min(e_bar - 2, n - 2) + 1):
        n_candidates += 1
        f = d + 1
        H = min(e_bar, _ref_unit_end(unit_starts, f, n), n - 1)
        entry_price = float(opens[f])

        # --- first target touch measured from the EVENT START ---------------- #
        # Brute force: independent forward scan from start_bar (never reuses the
        # production suffix/next-touch arrays).
        first_touch = None
        if target_price is not None:
            tp = float(target_price)
            for t in range(s_bar, hi + 1):     # from the EVENT START horizon
                if direction == "LONG":
                    if float(highs[t]) >= tp:
                        first_touch = t
                        break
                else:
                    if float(lows[t]) <= tp:
                        first_touch = t
                        break

        # --- entry legality w.r.t. the target ------------------------------- #
        same_bar_target = False
        if first_touch is not None:
            if first_touch < f:
                continue                       # target already touched -> over
            if first_touch == f:
                if direction == "LONG" and not entry_price < float(target_price):
                    continue                   # fill already beyond the target
                if direction == "SHORT" and not entry_price > float(target_price):
                    continue
                same_bar_target = True

        if not same_bar_target and f + 1 > H:
            continue
        n_with_path += 1

        touch_bar = first_touch
        if same_bar_target:
            touch_bar = f
        elif first_touch is not None and first_touch > H:
            touch_bar = None

        if touch_bar is not None:
            exit_fill = int(touch_bar)
            exit_price = float(target_price)
            pnl = sign * (float(target_price) - entry_price)
            reason = R_TARGET
        else:
            best_k = None
            best_p = None
            for k in range(f + 1, H + 1):
                p = sign * (float(opens[k]) - entry_price)
                if best_p is None or p > best_p:   # strict -> earlier exit wins ties
                    best_p = p
                    best_k = k
            if best_k is None:
                continue
            exit_fill = int(best_k)
            exit_price = float(opens[best_k])
            pnl = float(best_p)
            reason = R_EARLY if pnl > PNL_EPS else R_LOSS

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
            best = cand   # strict -> earlier entry wins ties

    if best is None:
        reason = "NO_EXECUTABLE_ENTRY" if n_candidates == 0 else "INSUFFICIENT_PATH"
        return {"ok": False, "invalid_reason": reason,
                "n_candidates": n_candidates, "n_with_path": n_with_path}

    if best["exit_reason"] == R_TARGET and best["exit_fill_index"] == e_bar and e_bar < n:
        return {"ok": False, "invalid_reason": "AMBIGUOUS_SAME_BAR_TERMINAL",
                "n_candidates": n_candidates, "n_with_path": n_with_path}

    # ---- metrics (duplicated on purpose; no import from the kernel) -------- #
    zb, zt = float(zone_bottom), float(zone_top)
    ep = float(best["entry_price"])
    xp = float(best["exit_price"])
    pnl = float(best["utility"])

    if direction == "LONG":
        gap = 0.0 if (ep >= zb and ep <= zt) else (zb - ep if ep < zb else ep - zt)
    else:
        gap = 0.0 if (ep >= zb and ep <= zt) else (zb - ep if ep < zb else ep - zt)
    gap_atr = gap / float(atr_value)

    def _dir(a: float, b: float) -> float:
        return (b - a) if direction == "LONG" else (a - b)

    positive = pnl > PNL_EPS
    if best["exit_reason"] == R_TARGET:
        # HARD INVARIANT (V3.1): with the target-before-entry filter, a target
        # touch can never be a loss. Duplicated guard in the reference too.
        if not positive:
            raise AssertionError(
                "HARD_FAIL_TARGET_TOUCH_WITH_LOSS: "
                f"entry={ep} target={xp} pnl={pnl}"
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
            remaining = raw
    else:
        tp_atr = float("nan")
        remaining = float("nan")
        remaining_raw = (
            float("nan") if target_price is None
            else _dir(xp, float(target_price)) / float(atr_value)
        )

    out = dict(best)
    out.update({
        "ok": True,
        "invalid_reason": None,
        "best_entry_gap_points": float(gap),
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
