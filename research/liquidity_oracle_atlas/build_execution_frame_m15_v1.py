"""build_execution_frame_m15_v1
===============================

Canonical R4 Execution Frame owner (15m decision axis).

Single responsibility: convert the raw 5m market-data source into the
deterministic, causal, COMPLETED 15m execution bars. Everything downstream
(Candidate, DP, Environment, Model features) consumes THIS frame and is
forbidden from re-resampling 5m or re-deriving a 15m frame.

Raw 5m is retained ONLY as the plumbing input. The output frame has no 5m
decision / state / feature columns. This is the single, auditable entry point
that guarantees "the whole research system has no 5m decision axis".

Artifacts are persisted under
artifacts/candidate_gate_r4_m15_touch_nextbar_v1/ and SHA-verified on load
(fail-closed), so the ExecutionFrame consumed by Candidate / DP / Model is
byte-identical to what was generated.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from research.export_ob_trigger_execution_v21 import load_raw_5m
from research.phase1_tradability.phase1_contract_v1 import discontinuity_flags
from research.liquidity_oracle_atlas.experiment_structural_reversion_pgm_v1 import (
    raw_frame_from_owner,
    resample_causal,
)

ARTIFACT_DIR = Path("artifacts/candidate_gate_r4_m15_touch_nextbar_v1")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
FRAME_MATH_VERSION = "r4_m15_execution_frame_v1"
SUMMARY_FILE = ARTIFACT_DIR / "summary.json"


# --------------------------------------------------------------------------- #
# Raw 5m -> base frame (plumbing only)                                          #
# --------------------------------------------------------------------------- #
def _build_raw_base(symbol: str, max_bars: Optional[int] = None) -> pd.DataFrame:
    raw = load_raw_5m(symbol).sort_values("bar_start_time").reset_index(drop=True)
    disc = np.asarray(discontinuity_flags(symbol), dtype=bool)
    if len(raw) != len(disc):
        raise SystemExit("STOP_RAW_DISC_LENGTH_MISMATCH")
    if max_bars is not None:
        n = min(int(max_bars), len(raw))
        raw = raw.iloc[:n].reset_index(drop=True)
        disc = disc[:n]
    bars = dict(
        n=len(raw),
        t=pd.to_datetime(raw["bar_start_time"]).to_numpy(),
        day=pd.to_datetime(raw["trading_day"]).to_numpy(),
        disc=disc,
        o=raw["open"].to_numpy(float),
        h=raw["high"].to_numpy(float),
        l=raw["low"].to_numpy(float),
        c=raw["close"].to_numpy(float),
    )
    return raw_frame_from_owner(bars)


def build_execution_frame_m15(
    symbol: str, max_bars: Optional[int] = None
) -> pd.DataFrame:
    """Deterministically build COMPLETED 15m execution bars from raw 5m.

    Returns a DataFrame with columns:
        symbol, execution_bar_index, bar_start_time, decision_time,
        trading_day, segment, open, high, low, close, n_base

    where decision_time = bar_start_time + 15min (the 15m bar END is the causal
    decision instant per AGENTS.md TDX bar contract: bar label = interval end,
    availability_time = bar end). ``n_base`` is the number of 5m bars aggregated
    into each 15m bar.

    Differential invariant (verified by test_execution_frame_m15_v1):
        O = first 5m open,  H = max(5m H),  L = min(5m L),  C = last 5m close.
    """
    base = _build_raw_base(symbol, max_bars)
    m15 = resample_causal(base, 15).copy()
    n = len(m15)
    m15 = m15.reset_index(drop=True)
    times = pd.to_datetime(m15["time"])
    out = pd.DataFrame(
        {
            "symbol": symbol,
            "execution_bar_index": np.arange(n, dtype=np.int64),
            "bar_start_time": times,
            "decision_time": times + pd.Timedelta(minutes=15),
            "trading_day": pd.to_datetime(m15["trading_day"]),
            "segment": m15["segment"].to_numpy(np.int64),
            "open": m15["open"].to_numpy(float),
            "high": m15["high"].to_numpy(float),
            "low": m15["low"].to_numpy(float),
            "close": m15["close"].to_numpy(float),
            "n_base": m15["n_base"].to_numpy(np.int64),
        }
    )
    return out


# --------------------------------------------------------------------------- #
# Persistence + fail-closed loaders                                            #
# --------------------------------------------------------------------------- #
def frame_path(symbol: str) -> Path:
    return ARTIFACT_DIR / f"{symbol}_exec_frame.parquet"


def save_execution_frame_m15(symbol: str, max_bars: Optional[int] = None) -> Dict[str, Any]:
    """Build + persist the canonical 15m execution frame; return its SHA metadata."""
    df = build_execution_frame_m15(symbol, max_bars)
    p = frame_path(symbol)
    df.to_parquet(p, index=False)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    return {
        "symbol": symbol,
        "rows": int(len(df)),
        "sha256": sha,
        "path": str(p),
    }


def load_summary() -> Dict[str, Any]:
    if not SUMMARY_FILE.exists():
        return {}
    return json.loads(SUMMARY_FILE.read_text())


def load_execution_frame_m15_verified(symbol: str) -> pd.DataFrame:
    """The ONLY sanctioned reader for the R4 15m execution frame.

    Fails closed on:
      * frame missing                  -> STOP_R4_FRAME_MISSING
      * symbol absent from manifest    -> STOP_R4_FRAME_MANIFEST_MISSING
      * on-disk SHA != manifest SHA     -> STOP_R4_FRAME_SHA_MISMATCH
    """
    p = frame_path(symbol)
    if not p.exists():
        raise RuntimeError(f"STOP_R4_FRAME_MISSING:{symbol}")
    summary = load_summary()
    spec = summary.get("execution_frames", {}).get(symbol)
    if spec is None:
        raise RuntimeError(f"STOP_R4_FRAME_MANIFEST_MISSING:{symbol}")
    expected = spec.get("sha256")
    if expected is None:
        raise RuntimeError(f"STOP_R4_FRAME_MANIFEST_MISSING:{symbol}")
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"STOP_R4_FRAME_SHA_MISMATCH:{symbol}:{expected}:{actual}"
        )
    return pd.read_parquet(p)
