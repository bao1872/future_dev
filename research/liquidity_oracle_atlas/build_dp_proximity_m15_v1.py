"""build_dp_proximity_m15_v1
=============================

DP-internal proximity owner for the 15m Oracle.

This module is the SINGLE OWNER of the DP's own proximity concept. It does
NOT use, import, or reference the R4 Candidate Trading Zones
(``candidate_gate_r4``, ``candidate_any``, ``candidate_episode_id``,
``touch_proof``, ``merged_episode_id``, ``quota_reset_after``, ``gap<=3``).
Those are an entirely separate subsystem that the DP must not depend on.

The DP proximity is the ORIGINAL R2 definition, mechanically ported from the
5m clock to the 15m clock (frozen 5m R2 at SHA cc7891723...):

    P_t^{15m} = 1{ distance(15m bar_t range, ANY pre-existing 15m/1h/4h SR/LIQ)
                     <= 0.50 * ATR_{15m,tf} }

Structures are the geometry KNOWN AT THE PREVIOUS 15m close (causal). The
geometry is supplied by the canonical R4 execution-environment owner
(``build_execution_environment_m15_v1.run_environment_m15``), which is the
single auditable source of 15m/1h/4h SR/LIQ geometry. Reusing it is NOT
reusing the Candidate gate: the gate consumes true-touch bits; the DP consumes
distance-based proximity bits from the same geometry. This is exactly the
frozen R2 semantics, just on the 15m execution clock.

Outputs (DP-internal only; never called "Candidate"):
    dp_proximity_bits        uint16  8-bit (15m/1h/4h x {SR, LIQ})
    dp_proximity_any         bool
    dp_proximity_episode_id  int64   continuous P=1 runs (resets at unit bndry)

A proximity episode is a maximal contiguous run of P=1. There is NO gap<=3
merging: any P=0 ends the episode. This reproduces the frozen V2 behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.build_execution_environment_m15_v1 import (
    run_environment_m15,
)
from research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1 import (
    ENTRY_PROX_ATR,
    proximity_bits_from_prev_geometry,
)


def build_dp_proximity_m15(
    symbol: str, max_bars: int | None = None
) -> pd.DataFrame:
    """Compute the DP-internal 15m proximity for one symbol.

    Returns a DataFrame aligned 1:1 with the 15m execution frame rows:
        symbol, execution_bar_index, bar_start_time, trading_day, segment,
        open, high, low, close,
        dp_proximity_bits, dp_proximity_any, dp_proximity_episode_id
    """
    env = run_environment_m15(symbol, max_bars, capture_provenance=False)
    exec_frame = env["exec_frame"]
    geom = env["geom_by_decision"]
    n = len(exec_frame)

    H = exec_frame["high"].to_numpy(float)
    L = exec_frame["low"].to_numpy(float)

    bits = np.zeros(n, dtype=np.uint16)
    any_f = np.zeros(n, dtype=bool)
    for i in range(n):
        prev = geom[i - 1] if i >= 1 else None
        if prev is None:
            continue
        b = proximity_bits_from_prev_geometry(
            float(L[i]), float(H[i]), prev, alpha=ENTRY_PROX_ATR
        )
        bits[i] = np.uint16(b)
        any_f[i] = bool(b != 0)

    # unit boundaries (trading-day + segment) -> episode id resets at each unit
    td_arr = pd.to_datetime(exec_frame["trading_day"]).to_numpy()
    seg_arr = exec_frame["segment"].to_numpy(np.int64)
    starts, _ = _build_intraday_units(td_arr, seg_arr)
    ep_id = compute_proximity_episode_id(any_f, starts, n)

    out = pd.DataFrame(
        {
            "symbol": symbol,
            "execution_bar_index": np.arange(n, dtype=np.int64),
            "bar_start_time": pd.to_datetime(exec_frame["bar_start_time"]),
            "trading_day": pd.to_datetime(exec_frame["trading_day"]),
            "segment": seg_arr,
            "open": exec_frame["open"].to_numpy(float),
            "high": H,
            "low": L,
            "close": exec_frame["close"].to_numpy(float),
            "dp_proximity_bits": bits.astype(np.int64),
            "dp_proximity_any": any_f,
            "dp_proximity_episode_id": ep_id,
        }
    )
    return out


def _build_intraday_units(
    trading_day: np.ndarray, segment: np.ndarray
) -> "tuple[np.ndarray, np.ndarray]":
    """Contiguous (trading_day, segment) blocks -> (starts, ends) inclusive."""
    trading_day = np.asarray(trading_day)
    segment = np.asarray(segment)

    boundary = np.empty(len(segment), dtype=bool)
    boundary[0] = True
    if len(segment) > 1:
        boundary[1:] = (trading_day[1:] != trading_day[:-1]) | (
            segment[1:] != segment[:-1]
        )

    starts = np.flatnonzero(boundary)
    ends = np.r_[starts[1:] - 1, len(segment) - 1]
    return starts, ends


def compute_proximity_episode_id(
    proximity_any: np.ndarray, starts: np.ndarray, n: int
) -> np.ndarray:
    """Continuous P=1 runs -> 1-based episode id (resets at each unit start).

    Exogenous P_t runs; the episode id restarts at every unit boundary so the
    DP's per-episode entry quota is scoped to one (trading_day, segment) unit.
    """
    P = np.asarray(proximity_any, dtype=bool).copy()
    out = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return out
    unit_start = np.zeros(n, dtype=bool)
    unit_start[np.asarray(starts)] = True
    prev_p = np.r_[False, P[:-1]]
    episode_start = P & (~prev_p | unit_start)
    eid = np.cumsum(episode_start).astype(np.int64)
    out[P] = eid[P]
    return out
