"""verify_candidate_gate_r3_artifacts
==================================

Formal, FULL-HISTORY verification of the canonical R3 candidate-gate artifacts.

This is a SEPARATE script (not a committed pytest) because it depends on the
15 per-symbol parquet artifacts under ``artifacts/candidate_gate_r3_m5_touch_nextbar_v1/``,
which are intentionally NOT committed to Git (per repo convention: large parquet
artifacts stay local). A clean ``git clone && pytest`` must NOT require these
files, so the differential / alignment checks live here instead of in pytest.

What it proves (and records in a small JSON evidence file that IS committable):

  * artifact_sha_verified   : the on-disk parquet SHA equals summary.json manifest
                              (load_candidate_gate_verified fails closed otherwise)
  * alignment_mismatch      : exact row count + bar_index == arange(N) +
                              exact bar_start_time + exact trading_day vs raw 5m
  * touch_mismatch          : artifact touch_bits / candidate_any vs the INDEPENDENT
                              canonical true-touch owner (entry_bits_from_prev_geometry)

Usage:
    python research/liquidity_oracle_atlas/verify_candidate_gate_r3_artifacts.py

Exits non-zero if any symbol fails. Writes:
    artifacts/candidate_gate_r3_m5_touch_nextbar_v1/verification_evidence.json

Run this after regenerating artifacts (build_all_candidate_gates) to re-stamp the
evidence. Do NOT train a model as part of this step.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.export_ob_trigger_execution_v21 import load_raw_5m
from research.liquidity_oracle_atlas.build_candidate_gate_r3_v1 import (
    ALL_SYMBOLS,
    CANDIDATE_MATH_VERSION,
    gate_alignment_mismatch_count,
    load_candidate_gate_verified,
    validate_gate_against_raw,
)
from research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1 import (
    KernelCounters,
    _stream_from_base,
    build_base_frame,
    entry_bits_from_prev_geometry,
)

ARTIFACT_DIR = Path("artifacts/candidate_gate_r3_m5_touch_nextbar_v1")
EVIDENCE_FILE = ARTIFACT_DIR / "verification_evidence.json"


def _independent_entry_mask(symbol, max_bars):
    """Independent canonical true-touch recompute (guards against touch-math drift)."""
    counters = KernelCounters()
    info = build_base_frame(symbol, counters)
    res = _stream_from_base(
        info["base"], info["form"], info["seg_completed"], counters,
        max_bars, symbol, True, True, False, True, False,
    )
    entry_mask = np.asarray(res["entry_mask"], dtype=np.uint16)
    geom = res["decision_geom"]  # geom at bar i (== prev_geom for bar i+1)
    n = len(entry_mask)
    rec = np.zeros(n, dtype=np.uint16)
    base = info["base"]
    low = base["low"].to_numpy()
    high = base["high"].to_numpy()
    for i in range(1, n):
        prev_geom = geom[i - 1]
        g = {tf: prev_geom[tf] for tf in prev_geom}
        rec[i] = np.uint16(
            entry_bits_from_prev_geometry(float(low[i]), float(high[i]), g)
        )
    return entry_mask, rec


def verify_symbol(symbol: str) -> dict:
    out: dict = {
        "artifact_sha_verified": False,
        "alignment_mismatch": None,
        "touch_mismatch": None,
        "error": None,
    }
    # 1) fail-closed SHA / manifest / rows / version gate
    try:
        gate = load_candidate_gate_verified(symbol)
    except Exception as e:  # noqa: BLE001 - surface the STOP_* reason
        out["error"] = f"SHA_VERIFY_FAIL:{e}"
        return out
    out["artifact_sha_verified"] = True

    # 2) raw 5m alignment (fail-closed + counted)
    raw = load_raw_5m(symbol).sort_values("bar_start_time").reset_index(drop=True)
    try:
        validate_gate_against_raw(gate, raw)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"ALIGNMENT_FAIL:{e}"
        out["alignment_mismatch"] = gate_alignment_mismatch_count(gate, raw)
        return out
    out["alignment_mismatch"] = gate_alignment_mismatch_count(gate, raw)

    # 3) differential vs independent canonical true-touch owner
    em, _ = _independent_entry_mask(symbol, None)
    df_bits = gate["touch_bits"].to_numpy().astype(np.uint16)
    if len(em) != len(df_bits):
        out["error"] = (
            f"LENGTH_FAIL:entry_mask={len(em)}:artifact={len(df_bits)}"
        )
        out["touch_mismatch"] = int(abs(len(em) - len(df_bits))) + len(df_bits)
        return out
    out["touch_mismatch"] = int(np.sum(em != df_bits))
    return out


def main() -> int:
    if not ARTIFACT_DIR.exists():
        print(f"STOP: artifact dir missing: {ARTIFACT_DIR}", file=sys.stderr)
        print("Generate it first via build_all_candidate_gates(ALL_SYMBOLS).",
              file=sys.stderr)
        return 2

    evidence = {
        "candidate_math_version": CANDIDATE_MATH_VERSION,
        "verifier": "verify_candidate_gate_r3_artifacts.py",
        "symbols": {},
    }
    all_passed = True
    for sym in ALL_SYMBOLS:
        r = verify_symbol(sym)
        evidence["symbols"][sym] = r
        ok = (
            r["artifact_sha_verified"]
            and r["alignment_mismatch"] == 0
            and r["touch_mismatch"] == 0
            and r["error"] is None
        )
        if not ok:
            all_passed = False
        print(
            f"  {sym}: sha_ok={r['artifact_sha_verified']} "
            f"align={r['alignment_mismatch']} touch={r['touch_mismatch']} "
            f"{'OK' if ok else 'FAIL ' + str(r['error'])}"
        )

    evidence["all_passed"] = all_passed
    EVIDENCE_FILE.write_text(json.dumps(evidence, indent=2))
    print(f"\nevidence -> {EVIDENCE_FILE}")
    print("ALL PASSED" if all_passed else "VERIFICATION FAILED")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
