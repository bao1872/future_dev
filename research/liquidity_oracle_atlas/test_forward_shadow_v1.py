"""§47 regression tests — selection gate (24)(25) and forward (26)-(30)."""

import json
import os

import pytest

from research.liquidity_oracle_atlas import (
    forward_shadow_v1 as FS,
    run_decomposed_v2_research as R,
)


# --------------------------------------------------------------------------- #
# (24) selection reads TRAIN OOF only ; (25) VAL unavailable until frozen      #
# --------------------------------------------------------------------------- #
def test_selection_json_contains_train_only_contract():
    sel = R.load_selection()
    for key in ("selected_architecture", "selected_feature_schema",
                "win_schema_sha256", "payoff_schema_sha256",
                "lightgbm_params", "oof_fold_results", "one_se_decision"):
        assert key in sel, key
    assert sel["stage"] == "TRAIN_ONLY_SELECTION"
    # Five outer folds for every candidate.
    for cand, folds in sel["oof_fold_results"].items():
        assert len(folds) == 5, cand
    assert sel["governance"]["val_outcomes_read"] is False
    assert sel["governance"]["old_test_label_reads"] == 0
    assert sel["governance"]["old_test_policy_reads"] == 0


def test_selection_status_is_frozen_value():
    sel = R.load_selection()
    assert sel["status"] in ("NO_V2_MODEL_IMPROVEMENT", "V2_CANDIDATE_SELECTED")
    assert sel["selected_architecture"] in sel["eligible_pool"]


def test_val_blocked_when_selection_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "SELECTION_JSON", str(tmp_path / "nope.json"))
    with pytest.raises(R.StopV2ValUnlocked):
        R.read_val_labels()


# --------------------------------------------------------------------------- #
# (26) forward start strictly after freeze commit                              #
# --------------------------------------------------------------------------- #
def test_forward_start_rule_is_after_freeze():
    lock = FS.build_forward_lock(
        code_sha="HEAD", feature_schema_sha="sha", model_sha={},
        training_artifact_sha={}, policies=["F0", "F1"],
        forward_start_rule=("first common trading day strictly after the "
                            "final pre-forward freeze commit"),
        include_v2=False)
    assert "strictly after" in lock["forward_start_rule"]
    assert not lock["include_v2"]


# --------------------------------------------------------------------------- #
# (27) outcome reader blocked before 80 common days                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reader", [
    FS.read_forward_pnl, FS.read_forward_win_rate, FS.read_forward_ev,
    FS.read_forward_by_symbol, FS.read_forward_deciles,
    FS.read_forward_regime,
])
def test_outcome_readers_locked_before_80_days(reader):
    with pytest.raises(FS.StopV2ForwardLocked) as exc:
        reader(79, all_symbols=True)
    assert "STOP_V2_FORWARD_LOCKED" in str(exc.value)


def test_reader_still_locked_at_79_even_with_all_symbols():
    with pytest.raises(FS.StopV2ForwardLocked):
        FS.read_forward_pnl(79, all_symbols=True)


def test_incomplete_symbols_block_even_after_80_days():
    with pytest.raises(FS.StopV2ForwardIncomplete):
        FS.read_forward_pnl(80, all_symbols=False)


def test_lock_opens_at_80_days_with_all_symbols():
    assert FS.assert_outcome_lock_open(80, all_symbols=True) is True
    assert FS.read_forward_pnl(80, all_symbols=True) == "pnl"


# --------------------------------------------------------------------------- #
# (28) model / artifact hashes checked on every ingestion                      #
# --------------------------------------------------------------------------- #
def test_monitor_checks_hashes_and_symbols(tmp_path):
    art = tmp_path / "a.bin"
    art.write_bytes(b"artifact-bytes")
    model = tmp_path / "m.txt"
    model.write_bytes(b"model-bytes")

    mon = FS.monitor_ingestion(
        symbols_present=[f"S{i}" for i in range(15)],
        feature_rows_by_symbol={f"S{i}": 10 for i in range(15)},
        artifact_paths={"state": str(art)},
        model_paths={"td5_win": str(model)},
        days_accumulated=10)

    assert mon["all_symbols_present"] is True
    assert mon["feature_computation_complete"] is True
    assert mon["hashes_valid"] is True
    # Hash is content-addressed, so a changed artifact is detected.
    art.write_bytes(b"tampered")
    mon2 = FS.monitor_ingestion(
        symbols_present=[f"S{i}" for i in range(15)],
        feature_rows_by_symbol={f"S{i}": 10 for i in range(15)},
        artifact_paths={"state": str(art)},
        model_paths={"td5_win": str(model)},
        days_accumulated=11)
    assert mon2["artifact_sha256"]["state"] != mon["artifact_sha256"]["state"]
    assert mon2["hashes_valid"] is True   # present, but the SHA changed


def test_monitor_detects_missing_symbols():
    mon = FS.monitor_ingestion(
        symbols_present=["A", "B"], feature_rows_by_symbol={"A": 1, "B": 1})
    assert mon["all_symbols_present"] is False
    assert mon["n_symbols_present"] == 2


# --------------------------------------------------------------------------- #
# (29) no interim PnL artifact emitted                                         #
# --------------------------------------------------------------------------- #
def test_no_interim_outcome_artifact_exists():
    forbidden = ("pnl", "win_rate", "decile")
    base = os.path.dirname(FS.FORWARD_LOCK_JSON)
    for name in os.listdir(base) if os.path.isdir(base) else []:
        low = name.lower()
        assert not any(f in low for f in forbidden) or "v1_" in low, name


def test_monitor_never_reports_outcomes():
    mon = FS.monitor_ingestion(
        symbols_present=[f"S{i}" for i in range(15)],
        feature_rows_by_symbol={f"S{i}": 5 for i in range(15)},
        days_accumulated=80)
    for key in ("pnl", "win_rate", "ev", "deciles", "regime", "sharpe"):
        assert key not in mon
    assert mon["lock_open"] is False


# --------------------------------------------------------------------------- #
# (30) all 15 symbols required                                                 #
# --------------------------------------------------------------------------- #
def test_all_15_symbols_required():
    assert FS.ALL_SYMBOLS_REQUIRED == 15
    assert FS.FORWARD_REQUIRED_DAYS == 80
    assert FS.FORWARD_REQUIRED_BLOCKS == 16
    assert FS.FORWARD_BOOTSTRAP_B == 5000


def test_forward_lock_requires_renewal_excluded():
    lock = FS.build_forward_lock(
        code_sha="x", feature_schema_sha="y", model_sha={},
        training_artifact_sha={}, policies=["F0", "F1"],
        forward_start_rule="after freeze", include_v2=False)
    assert lock["renewal_included"] is False
    assert "TRADABLE_ALPHA_NOT_ESTABLISHED" in lock["costs"]["gross_only_note"]
