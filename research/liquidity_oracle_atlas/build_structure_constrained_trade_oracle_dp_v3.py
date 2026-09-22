"""
build_structure_constrained_trade_oracle_dp_v3
===============================================

Intraday Structure-Constrained Trade Oracle DP V3 — R3 candidate gate.

Task ID : FUTURE-R3-CANONICAL-M5-TOUCH-NEXTBAR-GATE-V1
Base SHA: 22216903edbea98ce93714d2c9612f36c7f189eb

V3 frozen change vs V2
-----------------------
The ONLY change is the exogenous new-entry gate:

  * V2 entry gate = current 5m range lies within ``ENTRY_PROX_ATR`` (0.50) ATR
    of ANY pre-existing SR/Liquidity zone (distance-based proximity).
  * V3 entry gate = the canonical R3 candidate artifact ``candidate_any``:
    ``same_unit[t] AND (touch_bits[t-1] has 5m SR or 5m LIQ)`` — a TRUE-TOUCH
    5m eligibility, with the full 4TF trigger context preserved as
    ``candidate_trigger_bits``.

The Bellman mathematics, unit decomposition, 6-state kernel, frequency-control
entry-right, and (trading_day, segment) units are UNCHANGED from V2. V3 simply
feeds ``candidate_any`` where V2 fed ``proximity_any``. The inner R2 parameter
is still named ``proximity_any``; the V3 outer surface MUST NOT re-expose the
old name.

Streamlit / DP / Model all read the SAME persisted candidate artifact, so the
candidate universe seen on the chart, allowed by the DP, and trained on by the
future Model are identical (one SHA, one file).

Canonical owners reused (READ ONLY; no math copied):
  * research.liquidity_oracle_atlas.build_candidate_gate_r3_v1.load_candidate_gate
  * research.liquidity_oracle_atlas.build_structure_constrained_trade_oracle_dp_v2
        (solve_day_dp_v2, _solve_unit_v2, _walk_unit_path, build_intraday_units,
         _unit_terminal_reason, is_new_entry — kernel + unit walk are reused,
         NOT reimplemented)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from research.export_ob_trigger_execution_v21 import load_raw_5m
from research.liquidity_oracle_atlas.build_candidate_gate_r3_v1 import (
    load_candidate_gate_verified,
    validate_gate_against_raw,
)
from research.liquidity_oracle_atlas.build_structure_constrained_trade_oracle_dp_v2 import (
    _solve_unit_v2,
    _walk_unit_path,
    build_intraday_units,
)


MATH_VERSION = "intraday_dp_oracle_r3_m5_touch_nextbar_candidate"
TASK_ID = "FUTURE-R3-CANONICAL-M5-TOUCH-NEXTBAR-GATE-V1"


# --------------------------------------------------------------------------- #
# Core solver (thin wrapper: candidate_any is the ONLY exogenous gate)           #
# --------------------------------------------------------------------------- #
def solve_day_dp_v3(
    opens: np.ndarray,
    candidate_any: np.ndarray,
    cost_points: np.ndarray,
    start: int,
    end: int,
) -> Dict[str, np.ndarray]:
    """Solve one unit with the R3 candidate gate.

    Identical Bellman to V2; ``candidate_any`` replaces ``proximity_any`` as the
    new-entry eligibility signal. Returns the full per-decision Q[6,3] table plus
    best action / edge / ambiguity for every one of the 6 states.
    """
    # Kernel is the shared R2 implementation; its parameter name is proximity_any
    # but semantically it is "may open here" — which is exactly candidate_any.
    from research.liquidity_oracle_atlas.build_structure_constrained_trade_oracle_dp_v2 import (
        solve_day_dp_v2,
    )

    return solve_day_dp_v2(
        opens=opens,
        proximity_any=candidate_any,
        cost_points=cost_points,
        start=start,
        end=end,
    )


# --------------------------------------------------------------------------- #
# Lightweight R3 DP inputs: raw 5m open + verified candidate artifact only       #
# --------------------------------------------------------------------------- #
def load_r3_dp_inputs(symbol: str) -> Dict[str, Any]:
    """Lightweight R3 DP inputs: raw 5m ``open`` + verified candidate artifact.

    Explicitly performs ZERO 4TF resample and ZERO candidate recomputation:
    the candidate universe is read verbatim from the single canonical artifact
    (fail-closed verified) and then hard-aligned against the current raw 5m
    frame so a stale artifact can never silently shift the DP's candidate gate.
    """
    raw = load_raw_5m(symbol).sort_values("bar_start_time").reset_index(drop=True)
    gate = load_candidate_gate_verified(symbol)
    validate_gate_against_raw(gate, raw)
    return {
        "opens": raw["open"].to_numpy(float),
        "candidate_any": gate["candidate_any"].to_numpy(bool),
        "segment": gate["segment"].to_numpy(np.int64),
        "trading_day": pd.to_datetime(gate["trading_day"]).to_numpy(),
    }


# --------------------------------------------------------------------------- #
# Production runner: load candidate artifact, run the shared kernel per unit     #
# --------------------------------------------------------------------------- #
def run_symbol_dp_v3(
    symbol: str,
    cost_points: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run the R3 DP over a symbol using the canonical candidate artifact.

    New-entry eligibility at each bar is ``candidate_any`` from the persisted
    gate artifact (never recomputed here, never from distance-based proximity).
    Returns aggregate audit counters; full action tables are NOT persisted in
    checkpoint A (that is R3 action-artifact work, checkpoint B).
    """
    inputs = load_r3_dp_inputs(symbol)
    opens = inputs["opens"]
    n = len(opens)
    cand = inputs["candidate_any"][:n]
    if cost_points is None:
        cost = np.zeros(n, dtype=float)
    else:
        cost = np.asarray(cost_points, dtype=float)[:n]

    seg_arr = inputs["segment"][:n]
    td_arr = inputs["trading_day"][:n]
    starts, ends = build_intraday_units(td_arr, seg_arr)

    illegal = 0
    new_entries = 0
    units_run = 0
    for s, e in zip((int(x) for x in starts), (int(x) for x in ends)):
        if e - s < 2:
            continue
        core, _ = _solve_unit_v2(opens, cand, cost, s, e)
        path = _walk_unit_path(core, cand, s, e)
        units_run += 1
        for d in path:
            t = int(d["t"])
            if d["ne"]:
                new_entries += 1
                # a new entry may ONLY occur at a candidate bar; otherwise it is
                # an illegal new-entry (gate breach) and must be zero.
                if not bool(cand[t]):
                    illegal += 1

    return {
        "symbol": symbol,
        "math_version": MATH_VERSION,
        "units_run": units_run,
        "new_entries": new_entries,
        "illegal_new_entries": illegal,
        "candidate_bars": int(cand.sum()),
    }
