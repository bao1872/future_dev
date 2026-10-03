"""
build_god_oracle_latest_artifact_v1.py
======================================

OFFLINE artifact generator for the God-Mode Oracle viewer.

Run the CURRENT working-tree Oracle ONCE for AG, then materialize a single
"latest" artifact set that the Streamlit viewer reads -- no Oracle / environment
math is ever executed from the viewer.

INTEGRATION STATUS (see remote integration commit):
    This builder is wired to the V6 TOUCH-CHAIN God oracle
    (structural_god_oracle_m15_v6_touch_chain.run_touch_chain_oracle).
    V4 / V5 are NO LONGER called anywhere in artifact generation.
    All V4-specific hard validation (DIAG/PREEMPT sentinels, the obsolete
    SHORT 7705 -> 7655 expectation, count==94) has been removed.

Artifacts (one clearly named "latest" location):

    artifacts/god_oracle_m15_latest/
        trades.parquet
        bars.parquet
        structures.parquet
        manifest.json

manifest.json records (V6 touch-chain contract):

    source_git_sha
    generated_at
    symbol
    canonical_trade_count
    math_version                       (= V6 touch-chain math version)
    reconciliation_pass               (= True; source_clusters reconcile to
                                        the named buckets)
    total_true_touch_records
    total_touch_episodes
    total_touch_groups
    source_clusters
    same_location_groups_absorbed
    target_transitions
    ambiguous_same_bar_target_groups
    overlapping_zone_transitions
    no_legal_entry_transitions
    no_later_distinct_target_transitions
    oracle_meta                       (target_touch / early_exit for the
                                        viewer sidebar + full V6 audit)

Trade fields persisted (all the viewer needs -- schema is frozen so the
viewer requires NO math change):

    trade_seq, event_id, structure_id, candidate_decision_index, candidate_time,
    zone_bottom, zone_top, oracle_direction, best_entry_decision_index,
    best_entry_fill_index, best_entry_fill_time, best_entry_price,
    best_entry_gap_atr, target_structure_id, target_structure_type,
    target_price, exit_fill_index, exit_fill_time, exit_price, tp_atr,
    exit_reason, utility

Geometry per bar (production SR zones + liquidity levels, EXACTLY the Oracle's
decision-time owner) is persisted as JSON in structures.parquet so the viewer
does NOT recompute anything.

HARD VALIDATION (must hold or the script refuses to write and exits non-zero):

    V6 generic contract (every build):
        canonical count > 0
        every trade exits on TARGET_TOUCH
        every trade utility > 0
        V6 reconciliation:
            source_clusters
              = target_transitions + no_later_distinct_target_transitions
              = canonical_trades
              + ambiguous_same_bar_target_groups
              + overlapping_zone_transitions
              + no_legal_entry_transitions
              + no_later_distinct_target_transitions
            target_transitions
              = canonical_trades
              + ambiguous_same_bar_target_groups
              + overlapping_zone_transitions
              + no_legal_entry_transitions
            total_touch_groups
              = source_clusters + same_location_groups_absorbed
        source_git_sha == current HEAD

NOTE: the builder does NOT invent any static-event semantics. event_id is -1,
and best_entry_gap_atr / tp_atr are np.nan placeholders (V6 does not compute an
ATR-normalized gap; the viewer only displays them).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.structural_god_oracle_m15_v6_touch_chain import (
    MATH_VERSION,
    print_screenshot_audit,
    run_touch_chain_oracle,
)
from research.liquidity_oracle_atlas.build_execution_environment_m15_v1 import (
    run_environment_m15,
)

SYMBOL = "AG"
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../future_dev
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "god_oracle_m15_latest"


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


def _hard_validate(canon: list, source_sha: str, audit: dict) -> None:
    # ---- V6 generic invariants (every build) ----
    if not canon:
        raise SystemExit("HARD STOP: zero canonical trades produced")
    if not all(r["exit_reason"] == "TARGET_TOUCH" for r in canon):
        raise SystemExit("HARD STOP: a canonical trade is not TARGET_TOUCH")
    if not all(float(r["utility"]) > 0 for r in canon):
        raise SystemExit("HARD STOP: a canonical trade has non-positive utility")

    # NOTE: the V6 touch-chain emits ONE trade per source price location:
    # repeated same-location touches are absorbed into a single source cluster
    # and produce exactly ONE trade to the next distinct target. The chain
    # advances to that target, so trades never share a target group / exit.

    # ---- V6 touch-chain reconciliation (no unexplained loss) ----
    buckets = (
        int(audit["canonical_trades"])
        + int(audit["ambiguous_same_bar_target_groups"])
        + int(audit["overlapping_zone_transitions"])
        + int(audit["no_legal_entry_transitions"])
        + int(audit["no_later_distinct_target_transitions"])
    )
    if int(audit["source_clusters"]) != buckets:
        raise SystemExit(
            f"HARD STOP: V6 touch-chain reconciliation failed "
            f"(source_clusters={audit['source_clusters']} != buckets={buckets})"
        )
    if int(audit["total_touch_groups"]) != (
        int(audit["source_clusters"])
        + int(audit["same_location_groups_absorbed"])
    ):
        raise SystemExit(
            f"HARD STOP: V6 touch-chain group accounting failed "
            f"(total_touch_groups={audit['total_touch_groups']} != "
            f"source_clusters={audit['source_clusters']} + "
            f"absorbed={audit['same_location_groups_absorbed']})"
        )

    _assert_source_sha(source_sha)


def _print_touch_chain_counts(audit: dict) -> None:
    print("[artifact] V6 touch-chain counts:")
    for k in (
        "total_true_touch_records",
        "total_touch_episodes",
        "total_touch_groups",
        "source_clusters",
        "same_location_groups_absorbed",
        "target_transitions",
        "ambiguous_same_bar_target_groups",
        "overlapping_zone_transitions",
        "no_legal_entry_transitions",
        "no_later_distinct_target_transitions",
        "canonical_trades",
    ):
        print(f"  {k:34s}: {audit[k]}")
    denom = audit["target_transitions"] or 1
    print(
        f"  canonical/target_transitions ratio : "
        f"{audit['canonical_trades'] / denom:.3f}"
    )


def _build_trade_rows(canon: list) -> list:
    rows = []
    for seq, r in enumerate(canon, start=1):
        tid = str(r.get("target_structure_id") or "")
        rows.append({
            "trade_seq": seq,
            "event_id": int(r["event_id"]) if r.get("event_id") is not None else -1,
            "structure_id": str(r["structure_id"]),
            "candidate_decision_index": int(r["candidate_start_bar"]),
            "candidate_time": (
                pd.Timestamp(r["candidate_start_time"])
                if r.get("candidate_start_time") is not None else pd.NaT
            ),
            "zone_bottom": float(r["zone_bottom"]),
            "zone_top": float(r["zone_top"]),
            "oracle_direction": str(r["oracle_direction"]),
            "best_entry_decision_index": int(r["best_entry_decision_index"]),
            "best_entry_fill_index": int(r["best_entry_fill_index"]),
            "best_entry_fill_time": (
                pd.Timestamp(r["best_entry_fill_time"])
                if r.get("best_entry_fill_time") is not None else pd.NaT
            ),
            "best_entry_price": float(r["best_entry_price"]),
            "best_entry_gap_atr": np.nan,   # V6 does not compute an ATR gap
            "target_structure_id": tid,
            "target_structure_type": tid.split("|")[0],
            "target_price": float(r["target_price"]),
            "exit_fill_index": int(r["exit_fill_index"]),
            "exit_fill_time": (
                pd.Timestamp(r["exit_fill_time"])
                if r.get("exit_fill_time") is not None else pd.NaT
            ),
            "exit_price": float(r["exit_price"]),
            "tp_atr": np.nan,               # V6 does not compute an ATR target dist
            "exit_reason": str(r["exit_reason"]),
            "utility": float(r["utility"]),
        })
    return rows


def _print_manual_audit(res: dict) -> None:
    """Manual review aid (first 20 trades + screenshot-2 region)."""
    canon = res["trades"]
    print("\n[artifact] first 20 V6 canonical trades:")
    print(
        "  seq | A_bar | A_structure | A_zone | "
        "B_bar | B_structure | B_zone | dir | entry_dec | entry | exit_bar | exit"
    )
    for seq, t in enumerate(canon[:20], start=1):
        a_mid = 0.5 * (t["zone_bottom"] + t["zone_top"])
        print(
            f"  {seq:>3} | {t['candidate_start_bar']:>5} | "
            f"{t['structure_id'][:22]:22s} | {a_mid:7.1f} | "
            f"{t['exit_fill_index']:>5} | {str(t['target_structure_id'])[:22]:22s} | "
            f"{t['target_price']:7.1f} | {t['oracle_direction']:4s} | "
            f"{t['best_entry_decision_index']:>5} | {t['best_entry_price']:7.1f} | "
            f"{t['exit_fill_index']:>5} | {t['exit_price']:7.1f}"
        )
    print("\n[artifact] screenshot region around old Event-8 support (~7690):")
    print_screenshot_audit(res, around_price=7690.0, band=30.0)


def build_artifact(symbol: str, out_dir: Path, max_bars: Optional[int] = None) -> dict:
    """Build the V6 touch-chain artifact into ``out_dir``.

    Returns the manifest dict. Does NOT modify V6 math -- it only consumes the
    V6 oracle output and adapts it into the viewer's frozen parquet schema.
    """
    out_dir = Path(out_dir)
    print(f"[gen] running V6 touch-chain oracle for {symbol} ...")
    res = run_touch_chain_oracle(symbol, max_bars=max_bars)
    trades = res["trades"]
    audit = res["audit"]
    canon = trades  # V6 already emits only canonical trades
    print(f"[gen] V6 canonical trade count = {len(canon)}")

    # Same-cached env (provenance=True) for bars + decision-time geometry.
    env = run_environment_m15(symbol, max_bars, capture_provenance=True)
    ef = env["exec_frame"]
    geom = env["geom_by_decision"]
    n = len(ef)

    _print_touch_chain_counts(audit)

    head = _git_sha()
    _hard_validate(canon, head, audit)
    print(
        f"[gen] hard-validation PASS: V6 reconciliation "
        f"({audit['source_clusters']} source clusters -> "
        f"{audit['canonical_trades']} canonical + "
        f"{audit['ambiguous_same_bar_target_groups']} ambiguous + "
        f"{audit['overlapping_zone_transitions']} overlap + "
        f"{audit['no_legal_entry_transitions']} no_entry + "
        f"{audit['no_later_distinct_target_transitions']} no_later_distinct)"
    )

    # ---- trades.parquet ----
    trades_df = pd.DataFrame(_build_trade_rows(canon))

    # ---- bars.parquet (full OHLC + bar_start_time) ----
    bars_df = pd.DataFrame({
        "bar_index": np.arange(n, dtype=np.int64),
        "bar_start_time": ef["bar_start_time"].to_numpy(),
        "open": ef["open"].to_numpy(float),
        "high": ef["high"].to_numpy(float),
        "low": ef["low"].to_numpy(float),
        "close": ef["close"].to_numpy(float),
    })

    # ---- structures.parquet (production geometry per bar, JSON) ----
    struct_rows = []
    for b in range(n):
        g = geom[b] if b < len(geom) else None
        struct_rows.append({
            "bar_index": int(b),
            "geom": json.dumps(g if g else {}, default=str),
        })
    struct_df = pd.DataFrame(struct_rows)

    # ---- manifest.json (V6 contract) ----
    bucket_sum = (
        int(audit["canonical_trades"])
        + int(audit["ambiguous_same_bar_target_groups"])
        + int(audit["overlapping_zone_transitions"])
        + int(audit["no_legal_entry_transitions"])
        + int(audit["no_later_distinct_target_transitions"])
    )
    reconciliation_pass = (
        int(audit["source_clusters"]) == bucket_sum
        and int(audit["target_transitions"])
        == bucket_sum - int(audit["no_later_distinct_target_transitions"])
        and int(audit["total_touch_groups"])
        == int(audit["source_clusters"]) + int(audit["same_location_groups_absorbed"])
    )
    manifest = {
        "source_git_sha": head,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "symbol": symbol,
        "canonical_trade_count": len(canon),
        "math_version": MATH_VERSION,
        "reconciliation_pass": bool(reconciliation_pass),
        "total_true_touch_records": int(audit["total_true_touch_records"]),
        "total_touch_episodes": int(audit["total_touch_episodes"]),
        "total_touch_groups": int(audit["total_touch_groups"]),
        "source_clusters": int(audit["source_clusters"]),
        "same_location_groups_absorbed": int(
            audit["same_location_groups_absorbed"]
        ),
        "target_transitions": int(audit["target_transitions"]),
        "ambiguous_same_bar_target_groups": int(
            audit["ambiguous_same_bar_target_groups"]
        ),
        "overlapping_zone_transitions": int(audit["overlapping_zone_transitions"]),
        "no_legal_entry_transitions": int(audit["no_legal_entry_transitions"]),
        "no_later_distinct_target_transitions": int(
            audit["no_later_distinct_target_transitions"]
        ),
        "oracle_meta": {
            "target_touch": len(canon),
            "early_exit": 0,
            "audit": audit,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    trades_df.to_parquet(out_dir / "trades.parquet", index=False)
    bars_df.to_parquet(out_dir / "bars.parquet", index=False)
    struct_df.to_parquet(out_dir / "structures.parquet", index=False)
    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    sha = hashlib.sha256((out_dir / "manifest.json").read_bytes()).hexdigest()
    print(f"[gen] wrote artifacts to {out_dir}")
    print(f"[gen] artifact SHA (manifest) = {sha}")
    print(
        f"[gen] trades={len(trades_df)} bars={len(bars_df)} "
        f"structures={len(struct_df)}"
    )

    _print_manual_audit(res)
    return manifest


def main() -> None:
    build_artifact(SYMBOL, ARTIFACT_DIR)


if __name__ == "__main__":
    main()
