"""
build_candidate_gate_r3_v1
============================

Canonical R3 Candidate Gate — ONE canonical owner, generated ONCE per symbol,
consumed by Streamlit / DP / future Model (no duplicate candidate logic anywhere).

Frozen semantics (contract FUTURE-R3-CANONICAL-M5-TOUCH-NEXTBAR-GATE-V1):

  * True-touch itself has ONE canonical owner: the existing
    ``entry_bits_from_prev_geometry`` / mask-only structure stream in
    ``experiment_structure_interaction_entry_v1``. This module does NOT implement
    SR / Liquidity touch math. It only reuses that owner to produce
    ``touch_bits[t]`` = the complete 4TF x {SR, LIQ} true-touch mask of bar t
    (current bar range vs structures ALREADY KNOWN at the previous 5m close).

  * ``touch_bits[t]`` keeps the COMPLETE 4TF x {SR, LIQ} context (8 bits).

  * Candidate eligibility is decided ONLY by the 5m bit:
        Candidate[t+1] = same_unit[t,t+1] AND (touch_bits[t] has 5m SR or 5m LIQ)
    where same_unit = (segment[t] == segment[t+1]) AND
                      (trading_day[t] == trading_day[t+1]).

  * Higher-TF touch can NEVER independently create a Candidate, but ALL
    simultaneous higher-TF touch bits are preserved as
    ``candidate_trigger_bits`` (the causal confluence / context for the bar).

  * The current candidate-bar 4TF environment is the responsibility of the
    canonical ``FormingEnvironmentBuilder`` (NOT this module).

Every 5m bar gets exactly one row. The artifact is the single source of truth:
Streamlit, DP and the future Model all read the SAME persisted parquet (SHA
recorded in summary.json), so they can never drift into "one graph / one DP /
one model" three different candidate universes.

Canonical owners reused (READ ONLY; no math copied):
  * research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1
        MASK_BIT, KernelCounters, build_base_frame, _stream_from_base
  * (environment side) build_forming_environment_v1.FormingEnvironmentBuilder
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1 import (
    KernelCounters,
    MASK_BIT,
    TF_ORDER,
    build_base_frame,
    _stream_from_base,
)


# --------------------------------------------------------------------------- #
# Frozen constants                                                             #
# --------------------------------------------------------------------------- #
CANDIDATE_MATH_VERSION = "r3_m5_touch_nextbar_gate_v1"
TOUCH_OWNER = "entry_bits_from_prev_geometry"
CANDIDATE_RULE = "candidate[t] = same_unit[t] and has_m5_touch(touch_bits[t-1])"

ARTIFACT_DIR = Path("artifacts/candidate_gate_r3_m5_touch_nextbar_v1")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE = ARTIFACT_DIR / "summary.json"

# 5m eligibility mask: 5m SR bit OR 5m LIQ bit (hard-coded bit index removed;
# derived from the frozen MASK_BIT so it cannot drift).
M5_GATE_MASK = np.uint16(
    (1 << MASK_BIT[("m5", "SR")]) | (1 << MASK_BIT[("m5", "LIQ")])
)


# --------------------------------------------------------------------------- #
# Core gate (the ONLY candidate derivation)                                     #
# --------------------------------------------------------------------------- #
def derive_nextbar_candidate_gate(
    touch_bits: np.ndarray,
    segment: np.ndarray,
    trading_day: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Derive next-bar candidate membership + retained trigger context.

    Args:
        touch_bits:   uint16 array, complete 4TF x {SR, LIQ} true-touch mask
                      per 5m bar (bar t range vs pre-existing structure).
        segment:      int array, exchange-session / segment id per bar.
        trading_day:  datetime64 array, trading-day label per bar.

    Returns a dict with:
        candidate_any           : bool, Candidate[t+1]
        candidate_trigger_bits  : uint16, full 4TF context of the trigger bar
                                  (touch_bits[t-1]); 0 when not a candidate
        prev_touch_bits         : uint16, touch_bits[t-1]
        same_unit               : bool, no cross-session / cross-unit inheritance
    """
    touch_bits = np.asarray(touch_bits, dtype=np.uint16)
    segment = np.asarray(segment)
    trading_day = np.asarray(trading_day)

    n = len(touch_bits)

    # Previous bar's COMPLETE 4TF touch information.
    prev_touch_bits = np.zeros(n, dtype=np.uint16)
    if n > 1:
        prev_touch_bits[1:] = touch_bits[:-1]

    # No cross-session / cross-unit inheritance.
    same_unit = np.zeros(n, dtype=bool)
    if n > 1:
        same_unit[1:] = (
            (segment[1:] == segment[:-1])
            & (trading_day[1:] == trading_day[:-1])
        )

    # ONLY 5m SR/LIQ grants candidate eligibility.
    m5_trigger = (prev_touch_bits & M5_GATE_MASK) != 0
    candidate_any = same_unit & m5_trigger

    # Full 4TF trigger context is retained.
    candidate_trigger_bits = np.where(
        candidate_any, prev_touch_bits, 0
    ).astype(np.uint16)

    return {
        "candidate_any": candidate_any,
        "candidate_trigger_bits": candidate_trigger_bits,
        "prev_touch_bits": prev_touch_bits,
        "same_unit": same_unit,
    }


def compute_candidate_episode_id(
    candidate_any: np.ndarray,
    same_unit: np.ndarray,
) -> np.ndarray:
    """Single canonical Candidate episode id (computed ONCE, shared everywhere).

    A new episode starts at a candidate bar that is either the first candidate
    in the series or follows a non-candidate bar or a unit boundary.
    Non-candidate bars get -1.
    """
    c = np.asarray(candidate_any, dtype=bool)
    n = len(c)
    prev_c = np.r_[False, c[:-1]] if n > 1 else np.zeros(0, dtype=bool)
    episode_start = c & (~prev_c | ~np.asarray(same_unit, dtype=bool))
    eid = np.cumsum(episode_start, dtype=np.int64)
    out = np.full(n, -1, dtype=np.int64)
    out[c] = eid[c]
    return out


def touch_bit(bits: np.ndarray, tf: str, family: str) -> np.ndarray:
    """Unified bit decoder — the ONLY place TF x {SR, LIQ} bits are parsed.

    Callers (Streamlit, DP, Model) MUST use this; no per-consumer decoding.
    """
    shift = MASK_BIT[(tf, family)]
    return ((np.asarray(bits, dtype=np.uint16) >> shift) & 1).astype(bool)


# --------------------------------------------------------------------------- #
# Per-symbol artifact generation                                                #
# --------------------------------------------------------------------------- #
def build_candidate_gate_for_symbol(symbol: str) -> Tuple[pd.DataFrame, str]:
    """Generate the canonical candidate artifact for one symbol.

    Returns (df, sha256_hex). Persists <symbol>.parquet under ARTIFACT_DIR.
    The 4TF true-touch mask is produced by the existing canonical mask-only
    stream (NOT recomputed here).
    """
    counters = KernelCounters()
    info = build_base_frame(symbol, counters)
    base = info["base"]
    n = len(base)

    stream = _stream_from_base(
        info["base"], info["form"], info["seg_completed"], counters,
        None, symbol,
        False,   # capture_geom
        True,    # capture_entry_mask  -> canonical TRUE-TOUCH mask
        False,   # emit_events
        True,    # mask_only
        False,   # capture_proximity
    )
    # IMPORTANT: this is the existing canonical TRUE-TOUCH mask, not the
    # distance-based proximity gate used by the R2 DP.
    touch_bits = np.asarray(stream["entry_mask"], dtype=np.uint16)

    gate = derive_nextbar_candidate_gate(
        touch_bits,
        base["segment"].to_numpy(),
        pd.to_datetime(base["trading_day"]).to_numpy(),
    )
    candidate_episode_id = compute_candidate_episode_id(
        gate["candidate_any"], gate["same_unit"]
    )

    times = pd.to_datetime(base["time"]).to_numpy()
    decision_time = pd.to_datetime(times) + pd.Timedelta(minutes=5)
    cand = gate["candidate_any"]
    trigger_bar_index = np.where(cand, np.arange(n) - 1, -1).astype(np.int64)
    trigger_decision_time = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    if n > 1:
        prev_dt = pd.to_datetime(times) + pd.Timedelta(minutes=5)
        trigger_decision_time[cand] = prev_dt[np.arange(n)[cand] - 1]

    df = pd.DataFrame(
        {
            "symbol": symbol,
            "bar_index": np.arange(n, dtype=np.int64),
            "bar_start_time": pd.to_datetime(base["time"]),
            "decision_time": decision_time,
            "segment": base["segment"].to_numpy().astype(np.int64),
            "trading_day": pd.to_datetime(base["trading_day"]),
            "touch_bits": touch_bits.astype(np.int64),
            "prev_touch_bits": gate["prev_touch_bits"].astype(np.int64),
            "candidate_any": cand.astype(bool),
            "candidate_trigger_bits": gate["candidate_trigger_bits"].astype(np.int64),
            "candidate_episode_id": candidate_episode_id,
            "trigger_bar_index": trigger_bar_index,
            "trigger_decision_time": trigger_decision_time,
        }
    )

    out_path = ARTIFACT_DIR / f"{symbol}.parquet"
    df.to_parquet(out_path, index=False)
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    return df, sha


def build_all_candidate_gates(
    symbols: List[str], prefix_bars: Optional[int] = None
) -> Dict[str, Any]:
    """Generate artifacts for all symbols + write summary.json with SHA256.

    ``prefix_bars`` (debug only) limits each symbol to its first N bars.
    """
    summary: Dict[str, Any] = {
        "candidate_math_version": CANDIDATE_MATH_VERSION,
        "touch_owner": TOUCH_OWNER,
        "candidate_rule": CANDIDATE_RULE,
        "symbol_artifacts": {},
    }
    for sym in symbols:
        df, sha = build_candidate_gate_for_symbol(sym)
        if prefix_bars is not None:
            df = df.head(prefix_bars)
        cand_rows = int(df["candidate_any"].sum())
        touch_rows = int((df["touch_bits"] != 0).sum())
        summary["symbol_artifacts"][sym] = {
            "sha256": sha,
            "rows": int(len(df)),
            "touch_rows": touch_rows,
            "candidate_rows": cand_rows,
        }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, default=str))
    return summary


# --------------------------------------------------------------------------- #
# Loader (shared by Viewer / DP / Model)                                        #
# --------------------------------------------------------------------------- #
def gate_path(symbol: str) -> Path:
    return ARTIFACT_DIR / f"{symbol}.parquet"


def load_candidate_gate(symbol: str) -> pd.DataFrame:
    """Load the persisted canonical candidate artifact for one symbol.

    This is the single source of truth consumed by Streamlit, the DP and the
    future Model. Raises FileNotFoundError if the artifact was not generated.
    """
    p = gate_path(symbol)
    if not p.exists():
        raise FileNotFoundError(
            f"candidate gate artifact missing for {symbol}: run "
            f"build_all_candidate_gates first ({p})"
        )
    return pd.read_parquet(p)


def load_candidate_gate_summary() -> Dict[str, Any]:
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(f"candidate gate summary missing: {SUMMARY_FILE}")
    return json.loads(SUMMARY_FILE.read_text())


ALL_SYMBOLS = [
    "AG", "AL", "AU", "CF", "CU", "I", "M", "MA",
    "NI", "P", "RB", "RU", "SC", "SN", "TA",
]
