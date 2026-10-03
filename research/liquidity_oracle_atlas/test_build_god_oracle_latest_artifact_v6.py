"""test_build_god_oracle_latest_artifact_v6
============================================

V6 touch-chain -> latest artifact -> Viewer integration tests.

These verify that the artifact builder is wired to the V6 touch-chain oracle
(and no longer to V4/V5), that the generated artifact honours the V6 contract
(reconciliation, schema, count), and that the (display-only) Viewer can load
the artifact and render trades without error.

The artifact is materialized into a TEMP directory only -- the canonical
`artifacts/god_oracle_m15_latest/` is intentionally NOT regenerated here (it is
produced separately, only after remote sign-off). This keeps the test hermetic
and avoids publishing a new artifact before review.
"""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from research.liquidity_oracle_atlas import build_god_oracle_latest_artifact_v1 as builder
from research.liquidity_oracle_atlas.structural_god_oracle_m15_v6_touch_chain import (
    MATH_VERSION,
)

REPO = Path(__file__).resolve().parents[2]
BUILDER_SRC = (
    REPO / "research" / "liquidity_oracle_atlas" / "build_god_oracle_latest_artifact_v1.py"
).read_text()

# Columns the Viewer reads off trades.parquet (schema must stay frozen so the
# Viewer needs no math change).
VIEWER_TRADE_COLUMNS = {
    "trade_seq", "event_id", "structure_id", "candidate_decision_index",
    "candidate_time", "zone_bottom", "zone_top", "oracle_direction",
    "best_entry_decision_index", "best_entry_fill_index", "best_entry_fill_time",
    "best_entry_price", "best_entry_gap_atr", "target_structure_id",
    "target_structure_type", "target_price", "exit_fill_index", "exit_fill_time",
    "exit_price", "tp_atr", "exit_reason", "utility",
}


def _load_viewer():
    path = REPO / "pages" / "6_Indicator_Viewer.py"
    spec = importlib.util.spec_from_file_location("indicator_viewer_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# 1. Builder is wired to V6, not V4/V5
# --------------------------------------------------------------------------- #
def test_builder_uses_v6_not_v4_v5():
    # static source check
    assert "run_touch_chain_oracle" in BUILDER_SRC
    assert "structural_god_oracle_m15_v4" not in BUILDER_SRC
    assert "structural_god_oracle_m15_v5" not in BUILDER_SRC
    assert "run_god_oracle_v4" not in BUILDER_SRC

    # runtime check: the imported label owner is the V6 module
    assert builder.run_touch_chain_oracle.__module__.endswith(
        "structural_god_oracle_m15_v6_touch_chain"
    )
    assert "structural_god_oracle_m15_v4" not in builder.__dict__
    assert "structural_god_oracle_m15_v5" not in builder.__dict__


# --------------------------------------------------------------------------- #
# 2. Generated artifact satisfies the V6.1 contract + Viewer can load/render
# --------------------------------------------------------------------------- #
def test_v6_artifact_contract_and_viewer_render(tmp_path):
    manifest = builder.build_artifact("AG", tmp_path, max_bars=1500)

    # math_version == V6.1 location-chain
    assert manifest["math_version"] == MATH_VERSION

    # the ONLY audit keys persisted are the V6.1 seven
    audit = manifest["oracle_meta"]["audit"]
    assert set(audit.keys()) == {
        "bars_with_true_touch",
        "location_touches",
        "same_location_retouch_bars",
        "target_transitions",
        "ambiguous_target_bars",
        "no_legal_entry",
        "canonical_trades",
    }

    # manifest count == len(trades.parquet) == > 0
    trades = pd.read_parquet(tmp_path / "trades.parquet")
    assert manifest["canonical_trade_count"] == len(trades)
    assert manifest["canonical_trade_count"] > 0

    # every emitted trade is a genuine V6.1 TARGET_TOUCH label
    assert (trades["exit_reason"] == "TARGET_TOUCH").all()
    assert (trades["utility"] > 0).all()

    # first 5 trade rows: one Entry, one Exit, TARGET_TOUCH, utility > 0
    for i in range(min(5, len(trades))):
        row = trades.iloc[i]
        assert int(row["best_entry_fill_index"]) >= 0
        assert int(row["exit_fill_index"]) >= 0
        assert row["exit_reason"] == "TARGET_TOUCH"
        assert float(row["utility"]) > 0

    # frozen Viewer schema preserved
    assert VIEWER_TRADE_COLUMNS.issubset(set(trades.columns))

    # Viewer load count == artifact count (Viewer not modified)
    viewer = _load_viewer()
    saved = viewer._ARTIFACT_DIR
    viewer._ARTIFACT_DIR = tmp_path
    try:
        data = viewer.load_oracle("AG")
    finally:
        viewer._ARTIFACT_DIR = saved
    assert len(data["canon"]) == manifest["canonical_trade_count"]

    # first 5 trades render without exception
    for i in range(min(5, len(data["canon"]))):
        fig = viewer._build_fig(data, data["canon"][i])
        assert fig is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
