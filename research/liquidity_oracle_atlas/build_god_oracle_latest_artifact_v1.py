"""
build_god_oracle_latest_artifact_v1.py
======================================

OFFLINE artifact generator for the God-Mode Oracle viewer.

Run the CURRENT working-tree Oracle ONCE for AG, then materialize a single
"latest" artifact set that the Streamlit viewer reads — no Oracle / environment
math is ever executed from the viewer.

Artifacts (one clearly named "latest" location):

    artifacts/god_oracle_m15_latest/
        trades.parquet
        bars.parquet
        structures.parquet
        manifest.json

manifest.json records:
    source_git_sha
    generated_at
    symbol
    canonical_trade_count
    math_version
    oracle_meta            (full res["meta"] for sidebar diagnostics)

Trade fields persisted (all the viewer needs):
    trade_seq, event_id, structure_id, candidate_decision_index, candidate_time,
    zone_bottom, zone_top, oracle_direction, best_entry_fill_index,
    best_entry_fill_time, best_entry_price, best_entry_gap_atr,
    best_entry_decision_index, target_structure_id, target_structure_type,
    target_price, exit_fill_index, exit_fill_time, exit_price, tp_atr,
    exit_reason, utility

Geometry per bar (production SR zones + liquidity levels, EXACTLY the Oracle's
decision-time owner) is persisted as JSON in structures.parquet so the viewer
does NOT recompute anything.

HARD VALIDATION (must hold or the script refuses to write and exits non-zero):

    Generic invariants (every build):
        canonical count > 0
        every trade exits on TARGET_TOUCH
        every trade utility > 0
        Exit_i < Entry_{i+1}   (sequential one-position stream, no overlap)
        source_git_sha == current HEAD

    Option A / FIX-02 semantic sentinel:
        The diagnosed 349/350/7700/7773 case (SR|m15|0|346|7705.0|7694.0|75.0)
        is LOCAL-regression-only -- it must be ABSENT from the full stream,
        because the earlier PREEMPT structure (SR|m15|0|346|7691.0|7681.0|72.0)
        now legitimately completes first and advances the cursor past it.
        The builder therefore validates the PREEMPT sentinel instead:
            oracle_direction            = SHORT
            candidate_start_bar         = 347
            best_entry_decision_index   = 348
            best_entry_fill_index       = 349
            best_entry_price            = 7705
            exit_fill_index             = 570
            target_price                = 7655
            exit_price                  = 7655
            utility                     = 50
            exit_reason                 = TARGET_TOUCH
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.structural_god_oracle_m15_v4 import (
    run_god_oracle_v4,
)
from research.liquidity_oracle_atlas.build_execution_environment_m15_v1 import (
    run_environment_m15,
)

SYMBOL = "AG"
MATH_VERSION = "structural_god_oracle_m15_v4"
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../future_dev
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "god_oracle_m15_latest"

# Option A / FIX-02: the diagnosed 349/350/7700/7773 case is LOCAL-regression
# only (see tests). It must NOT appear in the full stream, so the builder uses
# the PREEMPT sentinel that replaced it as the hard checkpoint.
DIAG_SID = "SR|m15|0|346|7705.0|7694.0|75.0"
PREEMPT_SID = "SR|m15|0|346|7691.0|7681.0|72.0"
PREEMPT_EXPECT = {
    "oracle_direction": "SHORT",
    "candidate_start_bar": 347,
    "best_entry_decision_index": 348,
    "best_entry_fill_index": 349,
    "best_entry_price": 7705.0,
    "exit_fill_index": 570,
    "target_price": 7655.0,
    "exit_price": 7655.0,
    "utility": 50.0,
    "exit_reason": "TARGET_TOUCH",
}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _assert_source_sha(source_sha: str) -> None:
    if source_sha == "unknown":
        raise SystemExit("HARD STOP: could not resolve current git HEAD sha")
    try:
        cur = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception as exc:  # pragma: no cover - environment failure
        raise SystemExit(f"HARD STOP: git rev-parse HEAD failed: {exc}")
    if source_sha != cur:
        raise SystemExit(
            f"HARD STOP: source_git_sha {source_sha} != current HEAD {cur}"
        )


def _hard_validate(canon: list, source_sha: str) -> dict:
    # ---- generic invariants (every build) ----
    if not canon:
        raise SystemExit("HARD STOP: zero canonical trades produced")
    if not all(r["exit_reason"] == "TARGET_TOUCH" for r in canon):
        raise SystemExit("HARD STOP: a canonical trade is not TARGET_TOUCH")
    if not all(float(r["utility"]) > 0 for r in canon):
        raise SystemExit("HARD STOP: a canonical trade has non-positive utility")
    for i in range(len(canon) - 1):
        if not (int(canon[i]["exit_fill_index"])
                < int(canon[i + 1]["best_entry_decision_index"])):
            raise SystemExit(
                f"HARD STOP: trades overlap at i={i} "
                f"(exit {canon[i]['exit_fill_index']} >= next entry "
                f"{canon[i + 1]['best_entry_decision_index']})"
            )
    _assert_source_sha(source_sha)

    # ---- Option A / FIX-02 semantic sentinel ----
    # The diagnosed 349/350/7700/7773 case is LOCAL-regression-only; it must
    # NOT appear in the full stream (an earlier PREEMPT trade consumes the
    # cursor). Assert its absence.
    if any(r["structure_id"] == DIAG_SID for r in canon):
        raise SystemExit(
            f"HARD STOP: diagnosed structure {DIAG_SID} unexpectedly present "
            f"in full stream (Option A requires it to be absent)"
        )

    # PREEMPT is the full-stream sentinel that replaced DIAG.
    t2 = next((r for r in canon if r["structure_id"] == PREEMPT_SID), None)
    if t2 is None:
        raise SystemExit(
            f"HARD STOP: PREEMPT sentinel {PREEMPT_SID} not found in canonical trades"
        )
    for k, exp in PREEMPT_EXPECT.items():
        got = t2[k]
        if isinstance(exp, float):
            if abs(float(got) - exp) > 1e-6:
                raise SystemExit(f"HARD STOP: {PREEMPT_SID} {k} = {got}, expected {exp}")
        else:
            if str(got) != str(exp):
                raise SystemExit(
                    f"HARD STOP: {PREEMPT_SID} {k} = {got!r}, expected {exp!r}"
                )
    return t2


def main() -> None:
    print(f"[gen] running production oracle for {SYMBOL} ...")
    res = run_god_oracle_v4(SYMBOL)
    records = res["records"]
    meta = res["meta"]
    mv = res["market_view"]
    canon = [r for r in records if r.get("canonical_oracle_trade")]
    canon.sort(key=lambda r: int(r["candidate_start_bar"]))
    backend_canonical = len(canon)
    print(f"[gen] backend canonical count = {backend_canonical}")

    head = _git_sha()
    # HARD VALIDATION before touching any artifact
    t2 = _hard_validate(canon, head)
    print(
        f"[gen] hard-validation PASS: {PREEMPT_SID} -> "
        f"dir={t2['oracle_direction']} cand_start={t2['candidate_start_bar']} "
        f"decision={t2['best_entry_decision_index']} "
        f"fill={t2['best_entry_fill_index']} "
        f"entry={t2['best_entry_price']:.1f} "
        f"exit={t2['exit_fill_index']} "
        f"target={t2['target_price']:.1f} "
        f"profit={t2['utility']:.1f}"
    )

    # ---- trades.parquet ----
    trade_rows = []
    for seq, r in enumerate(canon, start=1):
        trade_rows.append({
            "trade_seq": seq,
            "event_id": int(r["event_id"]),
            "structure_id": str(r["structure_id"]),
            "candidate_decision_index": int(r["candidate_start_bar"]),
            "candidate_time": pd.Timestamp(r["candidate_start_time"]),
            "zone_bottom": float(r["zone_bottom"]),
            "zone_top": float(r["zone_top"]),
            "oracle_direction": str(r["oracle_direction"]),
            "best_entry_decision_index": int(r["best_entry_decision_index"]),
            "best_entry_fill_index": int(r["best_entry_fill_index"]),
            "best_entry_fill_time": pd.Timestamp(r["best_entry_fill_time"]),
            "best_entry_price": float(r["best_entry_price"]),
            "best_entry_gap_atr": float(r["best_entry_gap_atr"]),
            "target_structure_id": (r.get("target_structure_id") or ""),
            "target_structure_type": (r.get("target_structure_type") or ""),
            "target_price": float(r["target_price"]),
            "exit_fill_index": int(r["exit_fill_index"]),
            "exit_fill_time": pd.Timestamp(r["exit_fill_time"]),
            "exit_price": float(r["exit_price"]),
            "tp_atr": float(r["tp_atr"]),
            "exit_reason": str(r["exit_reason"]),
            "utility": float(r["utility"]),
        })
    trades_df = pd.DataFrame(trade_rows)

    # ---- bars.parquet (full OHLC + bar_start_time) ----
    n = int(mv.n)
    opens = np.asarray(mv.opens, dtype=float)
    highs = np.asarray(mv.highs, dtype=float)
    lows = np.asarray(mv.lows, dtype=float)
    closes = np.asarray(mv.closes, dtype=float)
    times = np.asarray(mv.times)
    bars_df = pd.DataFrame({
        "bar_index": np.arange(n, dtype=np.int64),
        "bar_start_time": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
    })

    # ---- structures.parquet (production geometry per bar, JSON) ----
    print(f"[gen] loading production geometry (geom_by_decision) ...")
    env = run_environment_m15(SYMBOL, None, capture_provenance=False)
    geom = env["geom_by_decision"]
    struct_rows = []
    for b in range(n):
        g = geom[b] if b < len(geom) else None
        struct_rows.append({
            "bar_index": int(b),
            "geom": json.dumps(g if g else {}, default=str),
        })
    struct_df = pd.DataFrame(struct_rows)

    # ---- manifest.json ----
    manifest = {
        "source_git_sha": _git_sha(),
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "symbol": SYMBOL,
        "canonical_trade_count": backend_canonical,
        "math_version": MATH_VERSION,
        "oracle_meta": meta,
        "sentinel_structure_id": PREEMPT_SID,
        "sentinel_expect": PREEMPT_EXPECT,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    trades_df.to_parquet(ARTIFACT_DIR / "trades.parquet", index=False)
    bars_df.to_parquet(ARTIFACT_DIR / "bars.parquet", index=False)
    struct_df.to_parquet(ARTIFACT_DIR / "structures.parquet", index=False)
    with open(ARTIFACT_DIR / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    # artifact SHA (sha256 of manifest, deterministic fingerprint)
    sha = hashlib.sha256(
        (ARTIFACT_DIR / "manifest.json").read_bytes()
    ).hexdigest()
    print(f"[gen] wrote artifacts to {ARTIFACT_DIR}")
    print(f"[gen] artifact SHA (manifest) = {sha}")
    print(f"[gen] trades={len(trades_df)} bars={len(bars_df)} structures={len(struct_df)}")


if __name__ == "__main__":
    main()
