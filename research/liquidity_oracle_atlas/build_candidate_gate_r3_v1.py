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

from research.export_ob_trigger_execution_v21 import load_raw_5m
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


def touch_bit(bits, tf: str, family: str) -> np.ndarray:
    """Unified bit decoder — the ONLY place TF x {SR, LIQ} bits are parsed.

    Callers (Streamlit, DP, Model) MUST use this; no per-consumer decoding.

    Accepts a scalar or array ``bits`` and ALWAYS returns an ndarray (at least
    1-d), so ``touch_bit(np.uint16(x), tf, fam)[0]`` is valid for a scalar.
    """
    shift = MASK_BIT[(tf, family)]
    arr = np.atleast_1d(np.asarray(bits, dtype=np.uint16))
    return ((arr >> shift) & 1).astype(bool)


# --------------------------------------------------------------------------- #
# Per-symbol artifact generation                                                #
# --------------------------------------------------------------------------- #
def _build_proof_df(symbol, entry_matches, base, times):
    """Flatten per-bar trigger matches into a compact long-format proof table.

    Only ACTUAL matches are stored (one row per (trigger_bar, hit_zone)); the full
    historical geometry is NOT persisted. ``trigger_bar_index`` == the bar at which
    the touch occurred (= ``candidate_any`` is True at ``trigger_bar + 1``).
    """
    n = len(base)
    rows = []
    for i in range(n):
        for m in entry_matches[i]:
            rows.append({
                "symbol": symbol,
                "trigger_bar_index": int(i),
                "trigger_decision_time": pd.Timestamp(times[i]),
                "tf": m["tf"],
                "family": m["family"],
                "side": m.get("side"),
                "slot": int(m["slot"]),
                "top": float(m["top"]),
                "bottom": float(m["bottom"]),
                "level": (
                    float(m["level"]) if m.get("level") is not None else np.nan
                ),
                "strength": (
                    float(m["strength"]) if m.get("strength") is not None else np.nan
                ),
                "intersects": bool(m.get("intersects", True)),
            })
    return pd.DataFrame(rows)


def build_candidate_gate_for_symbol(symbol: str) -> Tuple[pd.DataFrame, str, str, int]:
    """Generate the canonical candidate artifact + touch-proof sidecar for one symbol.

    Returns (df, candidate_sha256, proof_sha256, proof_rows). Persists
    <symbol>.parquet (candidate gate) and <symbol>_touch_proof.parquet (FIX2
    trigger proof) under ARTIFACT_DIR. The 4TF true-touch mask AND the per-trigger
    hit-zone provenance are produced by the SAME canonical mask-only stream pass
    (no second divergent touch computation). Candidate math is unchanged.
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
        capture_provenance=True,  # FIX2: record hit zones in the SAME pass
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
            "candidate_math_version": CANDIDATE_MATH_VERSION,
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

    # ---- FIX2: compact touch-proof sidecar (actual matches only) ------------ #
    proof_df = _build_proof_df(symbol, stream["entry_matches"], base, times)
    proof_path = ARTIFACT_DIR / f"{symbol}_touch_proof.parquet"
    proof_df.to_parquet(proof_path, index=False)
    proof_sha = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    return df, sha, proof_sha, int(len(proof_df))


def build_all_candidate_gates(symbols: List[str]) -> Dict[str, Any]:
    """Generate artifacts for all symbols + write summary.json with SHA256.

    The summary/spec SHA is computed from the SAME parquet that is persisted, so
    summary.json and the on-disk artifact can never drift (prefix debug mode
    removed: the canonical generator has no fuzzy truncation path).
    """
    summary: Dict[str, Any] = {
        "candidate_math_version": CANDIDATE_MATH_VERSION,
        "touch_owner": TOUCH_OWNER,
        "candidate_rule": CANDIDATE_RULE,
        "symbol_artifacts": {},
    }
    for sym in symbols:
        df, sha, proof_sha, proof_rows = build_candidate_gate_for_symbol(sym)
        cand_rows = int(df["candidate_any"].sum())
        touch_rows = int((df["touch_bits"] != 0).sum())
        summary["symbol_artifacts"][sym] = {
            "sha256": sha,
            "proof_sha256": proof_sha,
            "rows": int(len(df)),
            "touch_rows": touch_rows,
            "candidate_rows": cand_rows,
            "proof_rows": proof_rows,
        }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, default=str))
    return summary


# --------------------------------------------------------------------------- #
# Loader (shared by Viewer / DP / Model)                                        #
# --------------------------------------------------------------------------- #
def gate_path(symbol: str) -> Path:
    return ARTIFACT_DIR / f"{symbol}.parquet"


def proof_path(symbol: str) -> Path:
    return ARTIFACT_DIR / f"{symbol}_touch_proof.parquet"


def load_candidate_gate(symbol: str) -> pd.DataFrame:
    """Load the persisted canonical candidate artifact for one symbol.

    This is the single source of truth consumed by Streamlit, the DP and the
    future Model. Raises FileNotFoundError if the artifact was not generated.

    NOTE: this is the low-level read. Every CONSUMER (Viewer / DP / Model) MUST
    call :func:`load_candidate_gate_verified` instead, which fails closed on any
    SHA / manifest / row-count / math-version drift.
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


def load_candidate_gate_verified(symbol: str) -> pd.DataFrame:
    """The ONLY sanctioned consumer entry point for the candidate artifact.

    Fails closed (raises RuntimeError) on any of:
      * artifact file missing            -> STOP_R3_GATE_MISSING
      * symbol absent from manifest      -> STOP_R3_GATE_MANIFEST_MISSING
      * on-disk SHA != manifest SHA      -> STOP_R3_GATE_SHA_MISMATCH
      * row count != manifest rows       -> STOP_R3_GATE_ROWCOUNT_MISMATCH
      * artifact missing math-version     -> STOP_R3_GATE_VERSION_MISSING
      * artifact math-version drifted     -> STOP_R3_GATE_VERSION_MISMATCH
      * summary math-version drifted      -> STOP_R3_GATE_SUMMARY_VERSION_MISMATCH

    This is what guarantees Viewer / DP / Model consume the SAME verified fact:
    a stale or hand-edited parquet can never silently load.
    """
    p = gate_path(symbol)
    if not p.exists():
        raise RuntimeError(f"STOP_R3_GATE_MISSING:{symbol}")

    summary = load_candidate_gate_summary()
    spec = summary.get("symbol_artifacts", {}).get(symbol)
    if spec is None:
        raise RuntimeError(f"STOP_R3_GATE_MANIFEST_MISSING:{symbol}")

    actual_sha = hashlib.sha256(p.read_bytes()).hexdigest()
    expected_sha = spec["sha256"]
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"STOP_R3_GATE_SHA_MISMATCH:{symbol}:{expected_sha}:{actual_sha}"
        )

    df = pd.read_parquet(p)

    if len(df) != int(spec["rows"]):
        raise RuntimeError(
            f"STOP_R3_GATE_ROWCOUNT_MISMATCH:{symbol}:"
            f"{int(spec['rows'])}:{len(df)}"
        )

    if "candidate_math_version" not in df.columns:
        raise RuntimeError(f"STOP_R3_GATE_VERSION_MISSING:{symbol}")
    mv = df["candidate_math_version"].iloc[0]
    if mv != CANDIDATE_MATH_VERSION:
        raise RuntimeError(
            f"STOP_R3_GATE_VERSION_MISMATCH:{symbol}:{mv}:{CANDIDATE_MATH_VERSION}"
        )
    if summary.get("candidate_math_version") != CANDIDATE_MATH_VERSION:
        raise RuntimeError(f"STOP_R3_GATE_SUMMARY_VERSION_MISMATCH:{symbol}")

    return df


def load_candidate_proof_verified(symbol: str) -> pd.DataFrame:
    """The ONLY sanctioned reader for the R3 touch-proof sidecar.

    The proof artifact records, for every trigger (touch) bar, the EXACT SR /
    Liquidity zones that the canonical ``entry_touch_from_prev_geometry`` flagged
    (FIX2). It is the single source of "which structure was actually hit", shared
    verbatim by the Viewer / DP / Model — the Viewer must never re-run
    ``bar_hits_zone`` itself.

    Fails closed (raises RuntimeError) on:
      * artifact missing               -> STOP_R3_PROOF_MISSING
      * symbol absent from manifest    -> STOP_R3_PROOF_MANIFEST_MISSING
      * on-disk SHA != manifest SHA     -> STOP_R3_PROOF_SHA_MISMATCH
    """
    p = proof_path(symbol)
    if not p.exists():
        raise RuntimeError(f"STOP_R3_PROOF_MISSING:{symbol}")
    summary = load_candidate_gate_summary()
    spec = summary.get("symbol_artifacts", {}).get(symbol)
    if spec is None:
        raise RuntimeError(f"STOP_R3_PROOF_MANIFEST_MISSING:{symbol}")
    expected = spec.get("proof_sha256")
    if expected is None:
        raise RuntimeError(f"STOP_R3_PROOF_MANIFEST_MISSING:{symbol}")
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"STOP_R3_PROOF_SHA_MISMATCH:{symbol}:{expected}:{actual}"
        )
    return pd.read_parquet(p)


def gate_alignment_mismatch_count(gate: pd.DataFrame, raw: pd.DataFrame) -> int:
    """Count bars where the artifact disagrees with the raw 5m frame.

    Returns the number of mismatched bars (0 == perfectly aligned). Used by the
    formal verifier to populate ``alignment_mismatch`` evidence without raising.
    Compares row count, ``bar_index == arange(N)``, exact ``bar_start_time`` and
    exact ``trading_day`` (the artifact is the single source of ``segment``).
    """
    n = len(raw)
    if len(gate) != n:
        return int(abs(len(gate) - n)) + n  # length mismatch is fatal; over-count
    if not np.array_equal(
        gate["bar_index"].to_numpy(np.int64), np.arange(n, dtype=np.int64)
    ):
        return n
    gt = pd.to_datetime(gate["bar_start_time"]).to_numpy()
    rt = pd.to_datetime(raw["bar_start_time"]).to_numpy()
    if len(gt) != len(rt) or not np.array_equal(gt, rt):
        return n
    gd = pd.to_datetime(gate["trading_day"]).to_numpy()
    rd = pd.to_datetime(raw["trading_day"]).to_numpy()
    if len(gd) != len(rd) or not np.array_equal(gd, rd):
        return n
    return 0


def validate_gate_against_raw(gate: pd.DataFrame, raw: pd.DataFrame) -> None:
    """Hard gate: the candidate artifact MUST align exactly with raw 5m.

    Fail-closed (raises RuntimeError) on any misalignment so the DP / Model can
    never silently run against a stale or shifted candidate universe:

      * length mismatch              -> STOP_R3_GATE_LENGTH_MISMATCH
      * bar_index != arange(N)       -> STOP_R3_GATE_BAR_INDEX_MISMATCH
      * bar_start_time misaligned     -> STOP_R3_GATE_TIME_ALIGNMENT
      * trading_day misaligned        -> STOP_R3_GATE_DAY_ALIGNMENT

    Shared by the DP runner and the future Model dataset builder.
    """
    n = len(raw)
    if len(gate) != n:
        raise RuntimeError(
            f"STOP_R3_GATE_LENGTH_MISMATCH:{len(gate)}:{n}"
        )
    if not np.array_equal(
        gate["bar_index"].to_numpy(np.int64), np.arange(n, dtype=np.int64)
    ):
        raise RuntimeError("STOP_R3_GATE_BAR_INDEX_MISMATCH")
    gt = pd.to_datetime(gate["bar_start_time"]).to_numpy()
    rt = pd.to_datetime(raw["bar_start_time"]).to_numpy()
    if len(gt) != len(rt) or not np.array_equal(gt, rt):
        raise RuntimeError("STOP_R3_GATE_TIME_ALIGNMENT")
    gd = pd.to_datetime(gate["trading_day"]).to_numpy()
    rd = pd.to_datetime(raw["trading_day"]).to_numpy()
    if len(gd) != len(rd) or not np.array_equal(gd, rd):
        raise RuntimeError("STOP_R3_GATE_DAY_ALIGNMENT")


ALL_SYMBOLS = [
    "AG", "AL", "AU", "CF", "CU", "I", "M", "MA",
    "NI", "P", "RB", "RU", "SC", "SN", "TA",
]
