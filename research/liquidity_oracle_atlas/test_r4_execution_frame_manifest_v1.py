"""Execution-frame canonical manifest (FIX2, Blocking 1).

Proves:
- load_execution_frame_m15_verified(symbol) succeeds for all 15 frozen symbols
  (the manifest that the verified loader requires is populated and consistent).
- The manifest SHA matches the on-disk parquet (fail-closed guarantee).
"""

import hashlib

import pytest

from research.liquidity_oracle_atlas.build_execution_frame_m15_v1 import (
    frame_path,
    load_execution_frame_m15_verified,
    load_summary,
)
from research.liquidity_oracle_atlas.upstream_materialization_v1 import SYMBOLS


def test_verified_loader_succeeds_15_15():
    for s in SYMBOLS:
        df = load_execution_frame_m15_verified(s)  # must not raise
        assert len(df) > 0


def test_summary_manifest_present_and_consistent():
    summary = load_summary()
    assert "execution_frames" in summary
    frames = summary["execution_frames"]
    for s in SYMBOLS:
        spec = frames.get(s)
        assert spec is not None, f"missing manifest entry for {s}"
        p = frame_path(s)
        assert p.exists()
        assert spec["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest(), (
            f"manifest SHA mismatch for {s}"
        )
        assert spec["rows"] == len(__import__("pandas").read_parquet(p))
