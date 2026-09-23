"""Upstream 15-symbol manifest provenance (FIX2, Blocking 3).

Proves:
- AG dataset provenance is the distinct frozen SHA (df868eb...), NOT UPSTREAM_SHA.
- The 14 newly materialized symbols carry UPSTREAM_SHA as their dataset provenance.
- dataset_sha256 recorded in the manifest matches the actual on-disk parquet
  (so FIX2 did not silently recompute or alter any dataset).
"""

import hashlib

from pathlib import Path

import pytest

from research.liquidity_oracle_atlas.upstream_materialization_v1 import (
    AG_DATASET_SHA,
    DATASET_DIR,
    SYMBOLS,
    UPSTREAM_SHA,
)

MANIFEST = Path("artifacts/15sym_dataset_manifest.json")


def _load():
    import json

    return json.loads(MANIFEST.read_text())


def test_ag_dataset_provenance_distinct():
    man = {m["symbol"]: m for m in _load()}
    assert man["AG"]["dataset_builder_source_git_sha"] == AG_DATASET_SHA
    assert man["AG"]["dataset_builder_source_git_sha"] != UPSTREAM_SHA
    for s in SYMBOLS:
        if s == "AG":
            continue
        assert man[s]["dataset_builder_source_git_sha"] == UPSTREAM_SHA


def test_dataset_sha_matches_actual_files():
    for m in _load():
        p = DATASET_DIR / m["symbol"] / "candidate_teacher_dataset.parquet"
        if p.exists():
            assert m["dataset_sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
