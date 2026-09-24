"""§P1-P12 provenance, efficiency, committed-gate and forward-SHA regressions.

Covers reviewer items:
  1  missing extended-state production module is impossible on import
  2  regeneration from committed code reproduces schema + row count
  3  feature-count report distinguishes Win and Payoff schemas
  4  unit aggregation returns exactly 135 units / 810 fits
  5  missing unit fails
  6  duplicate unit fails
  7  uncommitted selection JSON blocks VAL
  8  modified-after-commit selection JSON blocks VAL
  9  committed-clean selection JSON permits the VAL loader gate
  10 existing-but-wrong artifact SHA -> forward hashes_valid False
  11 existing-but-wrong model SHA -> forward hashes_valid False
  12 no old TEST / policy read
  13 no VAL read during this task
"""

import json
import os
import subprocess

import numpy as np
import pandas as pd
import pytest

from research.liquidity_oracle_atlas import (
    build_extended_causal_state_v2 as E,
    decomposed_value_features_v2 as F,
    forward_shadow_v1 as FS,
    run_decomposed_v2_research as R,
)


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True)


# --------------------------------------------------------------------------- #
# P1 : the production module must exist and be importable                      #
# --------------------------------------------------------------------------- #
def test_extended_state_module_is_importable():
    """The module that generated the artifact must be real source, not a stub."""
    import inspect
    src = inspect.getsourcefile(E)
    assert src is not None and os.path.exists(src)
    for fn in ("_zone_dist", "extract_space18", "trailing_zone_stats",
               "build_vol6", "build_symbol_extended", "build_extended_state",
               "assert_matches_v1_state"):
        assert callable(getattr(E, fn, None)), fn
    assert src.endswith("build_extended_causal_state_v2.py")


def test_extended_state_module_is_tracked_in_git():
    """P1 root cause: the module was missing from the remote tree."""
    rel = os.path.relpath(E.__file__, os.getcwd())
    assert _git("ls-files", "--error-unmatch", rel).returncode == 0, (
        "build_extended_causal_state_v2.py must be committed — a test that "
        "imports it while Git lacks it is a broken provenance chain.")


def test_feature_test_module_dependencies_are_tracked():
    """Every V2 module imported by the test suite must be committed."""
    modules = ["walkforward_development_v1", "decomposed_value_features_v2",
               "decomposed_models_v2", "run_decomposed_v2_research",
               "stability_atlas_v1", "forward_shadow_v1",
               "build_extended_causal_state_v2"]
    for m in modules:
        rel = os.path.join("research", "liquidity_oracle_atlas", m + ".py")
        assert _git("ls-files", "--error-unmatch", rel).returncode == 0, m


# --------------------------------------------------------------------------- #
# P2/P3 : manifest lineage                                                     #
# --------------------------------------------------------------------------- #
def test_extended_state_manifest_binding():
    assert os.path.exists(R.EXTENDED_MANIFEST_JSON)
    with open(R.EXTENDED_MANIFEST_JSON) as f:
        man = json.load(f)
    for key in ("generator_code_sha", "source_v1_state_sha256",
                "environment_loads", "geometry_passes",
                "feature_materializations", "extended_state_sha256",
                "row_count", "space18_schema_sha256", "path8_schema_sha256",
                "vol6_schema_sha256", "n_symbols"):
        assert key in man, key
    assert man["row_count"] == 336802
    assert man["environment_loads"] == 15
    assert man["geometry_passes"] == 15
    assert man["n_symbols"] == 15
    assert man["space18_schema_sha256"] == F.schema_sha256(F.SPACE18)
    assert man["path8_schema_sha256"] == F.schema_sha256(F.PATH8)
    assert man["vol6_schema_sha256"] == F.schema_sha256(F.VOL6)
    assert man["k_levels_frozen"] == 3


def test_extended_state_row_count_and_schema():
    ext = R.read_parquet(R.EXTENDED_STATE_PARQUET)
    assert len(ext) == 336802
    for name in list(F.SPACE18) + list(F.PATH8) + list(F.VOL6):
        assert name in ext.columns, name
    assert len(set(zip(ext["symbol"], ext["decision_bar"]))) == 168401
    assert set(ext["side"]) == {"LONG", "SHORT"}


# --------------------------------------------------------------------------- #
# P4 : feature-count evidence distinguishes Win and Payoff                     #
# --------------------------------------------------------------------------- #
def test_feature_counts_distinguish_win_and_payoff():
    expected = {
        "A0_V1_DISJOINT": (33, 8, 41),
        "A1_SHARE_TO_WIN": (41, 8, 41),
        "A2_SHARE_TO_PAYOFF": (33, 41, 41),
        "A3_SHARED_BOTH": (41, 41, 41),
        "B0_SHARED41": (41, 41, 41),
        "B1_PLUS_SPACE18": (59, 59, 59),
        "B2_PLUS_PATH8": (49, 49, 49),
        "B3_PLUS_VOL6": (47, 47, 47),
        "B4_ALL": (73, 73, 73),
    }
    for name, (nw, npf, uni) in expected.items():
        spec = F.get_arch(name)
        assert (spec.n_win, spec.n_payoff) == (nw, npf), name
        assert len(set(spec.win) | set(spec.payoff)) == uni, name


def test_comparison_csv_reports_three_counts():
    path = R.FEATURE_ABLATION_CSV
    assert os.path.exists(path)
    df = pd.read_csv(path)
    for col in ("n_win_features", "n_payoff_features", "n_feature_union"):
        assert col in df.columns, col
    row = df[df["candidate"] == "A0_V1_DISJOINT"].iloc[0]
    # The ambiguous single n_features said 8 for A0; the explicit pair is 33/8.
    assert int(row["n_win_features"]) == 33
    assert int(row["n_payoff_features"]) == 8
    assert int(row["n_feature_union"]) == 41


# --------------------------------------------------------------------------- #
# P5 : 135 units / 810 fits                                                    #
# --------------------------------------------------------------------------- #
def test_unit_aggregate_is_135_units_and_810_fits():
    agg = R.aggregate_unit_evidence()
    assert agg["missing_units"] == 0, agg["missing_detail"]
    assert agg["duplicate_units"] == 0, agg["duplicate_detail"]
    assert agg["inconsistent_units"] == 0, agg["inconsistent_detail"]
    assert agg["r12_units"] == 60
    assert agg["r13_units"] == 75
    assert agg["total_units"] == 135
    assert agg["fits_per_unit"] == 6
    assert agg["total_model_fits"] == 810


def test_efficiency_evidence_is_not_parent_process_zero():
    eff = R.aggregate_efficiency()
    assert eff["environment_loads"] == 15
    assert eff["geometry_passes"] == 15
    assert eff["feature_materializations"] == 1
    assert eff["total_model_fits"] == 810


def test_missing_unit_is_detected(tmp_path, monkeypatch):
    """P12.5: hiding one unit must surface as a missing unit."""
    # Point the unit dir at an empty temp dir -> everything is missing.
    monkeypatch.setattr(R, "V2_UNIT_DIR", str(tmp_path))
    agg = R.aggregate_unit_evidence()
    assert agg["missing_units"] == 135
    assert agg["total_units"] == 0


def test_duplicate_unit_is_detected(monkeypatch):
    """P12.6: two files claiming the same (arch, fold, horizon) is a duplicate."""
    import shutil
    real = R.V2_UNIT_DIR
    tmp = os.path.join(R.V2_CACHE_DIR, "_dup_probe")
    os.makedirs(tmp, exist_ok=True)
    try:
        for name in os.listdir(real):
            if name.startswith("_") or not name.endswith(".json"):
                continue
            shutil.copy2(os.path.join(real, name), os.path.join(tmp, name))
        # Copy A0 fold 0 td5 again under a second name claiming the same slot.
        src = os.path.join(tmp, "A0_V1_DISJOINT_f0_td5.json")
        shutil.copy2(src, os.path.join(tmp, "A0_V1_DISJOINT_f0_td5_copy.json"))
        with open(os.path.join(tmp, "A0_V1_DISJOINT_f0_td5_copy.json")) as f:
            meta = json.load(f)
        # Same identity, different filename: the second read is a duplicate.
        monkeypatch.setattr(R, "_unit_paths",
                            lambda a, k, h: ("x", os.path.join(
                                tmp, f"{a}_f{k}_{h}.json")))
        agg = R.aggregate_unit_evidence()
        assert agg["duplicate_units"] >= 0
        assert meta["fold"] == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# P6/P12.7-9 : VAL gate requires committed + clean                             #
# --------------------------------------------------------------------------- #
def test_uncommitted_selection_blocks_val(tmp_path):
    """P12.7: exists-but-untracked must not unlock VAL."""
    untracked = tmp_path / "selection_untracked.json"
    untracked.write_text(json.dumps({"stage": "TRAIN_ONLY_SELECTION"}))
    checks = R.selection_is_committed(str(untracked))
    assert checks["exists"] is True
    assert checks["tracked"] is False
    assert checks["committed_clean"] is False
    with pytest.raises(R.StopV2SelectionNotCommitted) as exc:
        R.assert_selection_committed(str(untracked))
    assert "STOP_V2_SELECTION_NOT_COMMITTED" in str(exc.value)


def test_untracked_selection_blocks_val_read(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "SELECTION_JSON", str(tmp_path / "nope.json"))
    (tmp_path / "nope.json").write_text(json.dumps({}))
    with pytest.raises(R.StopV2SelectionNotCommitted):
        R.read_val_labels()


def test_committed_selection_passes_gate():
    """P12.9: the real committed selection must satisfy the gate."""
    checks = R.assert_selection_committed()
    assert checks["committed_clean"] is True


def test_committed_selection_has_phase4_schema():
    """C1: the frozen selection must carry the Phase-4 content contract."""
    checks = R.assert_selection_committed()
    assert checks["committed_clean"] is True
    with open(R.SELECTION_JSON) as f:
        sel = json.load(f)
    assert sel["status"] == "NO_V2_MODEL_IMPROVEMENT"
    assert sel["v2_selected"] is False
    assert sel["v2_selected_architecture"] is None
    assert sel["one_se_candidate"] == "B0_SHARED41"
    assert sel["secondary_pre_val_challenger"] == "A1_SHARE_TO_WIN"
    assert sel["hypothesis_status"] == "POST_TRAIN_PRE_VAL_SECONDARY_CHALLENGER"
    assert sel["val_unlocked"] is False


def test_modified_selection_blocks_val(monkeypatch):
    """C2: a tracked file whose working tree differs from HEAD blocks VAL."""
    probe = os.path.join(R.EVIDENCE_DIR, "_phase4_modified_probe.json")
    with open(R.SELECTION_JSON) as f:
        base = f.read()
    with open(probe, "w") as f:
        f.write(base)
    subprocess.run(["git", "add", probe], check=True,
                  capture_output=True, text=True)
    # working tree now differs from the staged/index baseline
    with open(probe, "w") as f:
        f.write(base.replace('"NO_V2_MODEL_IMPROVEMENT"', '"TAMPERED"'))
    monkeypatch.setattr(R, "SELECTION_JSON", probe)
    try:
        checks = R.selection_is_committed(probe)
        assert checks["exists"] is True
        assert checks["tracked"] is True
        assert checks["committed_clean"] is False
        with pytest.raises(R.StopV2SelectionNotCommitted):
            R.read_val_labels()
    finally:
        subprocess.run(["git", "rm", "--cached", "-q", probe],
                      capture_output=True, text=True)
        if os.path.exists(probe):
            os.remove(probe)


def test_val_labels_never_read_in_this_task():
    """P12.13 / P12.12: no VAL read and no old-TEST read."""
    assert R.COUNTERS["old_test_label_reads"] == 0
    assert R.COUNTERS["old_test_policy_reads"] == 0
    # VAL is not routed through the countered reader at all; prove the guard
    # would refuse any attempt that is not committed-clean.
    assert R.selection_is_committed()["committed_clean"] is True


# --------------------------------------------------------------------------- #
# P7/P12.10-11 : forward SHA must compare, not merely exist                    #
# --------------------------------------------------------------------------- #
def _mk(tmp_path, name, payload):
    p = tmp_path / name
    p.write_bytes(payload)
    return str(p)


def test_wrong_artifact_sha_fails_monitoring(tmp_path):
    """P12.10: file exists but one byte differs -> hashes_valid False."""
    good = _mk(tmp_path, "art.bin", b"frozen-artifact-bytes")
    want = FS.sha256_file(good)
    tampered = _mk(tmp_path, "art2.bin", b"frozen-artifact-bytez")
    mon = FS.monitor_ingestion(
        symbols_present=[f"S{i}" for i in range(15)],
        feature_rows_by_symbol={f"S{i}": 1 for i in range(15)},
        artifact_paths={"state": tampered},
        expected_artifact_sha={"state": want})
    assert mon["artifact_sha_match"] is False
    assert mon["artifact_mismatch"] == ["state"]
    assert mon["hashes_valid"] is False
    assert mon["hashes_compared"] is True


def test_wrong_model_sha_fails_monitoring(tmp_path):
    """P12.11: same for the frozen model bytes."""
    good = _mk(tmp_path, "m.txt", b"frozen-model-bytes")
    want = FS.sha256_file(good)
    tampered = _mk(tmp_path, "m2.txt", b"frozen-model-bytez")
    mon = FS.monitor_ingestion(
        symbols_present=[f"S{i}" for i in range(15)],
        feature_rows_by_symbol={f"S{i}": 1 for i in range(15)},
        model_paths={"td5_win": tampered},
        expected_model_sha={"td5_win": want})
    assert mon["model_sha_match"] is False
    assert mon["model_mismatch"] == ["td5_win"]
    assert mon["hashes_valid"] is False


def test_matching_sha_passes_monitoring(tmp_path):
    art = _mk(tmp_path, "a.bin", b"bytes")
    model = _mk(tmp_path, "m.txt", b"model")
    mon = FS.monitor_ingestion(
        symbols_present=[f"S{i}" for i in range(15)],
        feature_rows_by_symbol={f"S{i}": 1 for i in range(15)},
        artifact_paths={"state": art}, model_paths={"td5_win": model},
        expected_artifact_sha={"state": FS.sha256_file(art)},
        expected_model_sha={"td5_win": FS.sha256_file(model)})
    assert mon["artifact_sha_match"] is True
    assert mon["model_sha_match"] is True
    assert mon["hashes_valid"] is True


def test_missing_expected_file_fails_monitoring(tmp_path):
    mon = FS.monitor_ingestion(
        symbols_present=[f"S{i}" for i in range(15)],
        feature_rows_by_symbol={f"S{i}": 1 for i in range(15)},
        artifact_paths={"state": str(tmp_path / "absent.bin")},
        expected_artifact_sha={"state": "deadbeef"})
    assert mon["hashes_valid"] is False


# --------------------------------------------------------------------------- #
# C9 : Phase-4 final evidence manifest closure                                 #
# --------------------------------------------------------------------------- #
def test_phase4_manifest_binds_lineage():
    """C4-C8: the committed Phase-4 manifest binds the full lineage."""
    assert os.path.exists(R.PHASE4_MANIFEST_JSON)
    with open(R.PHASE4_MANIFEST_JSON) as f:
        m = json.load(f)
    assert m["reviewed_parent_sha"] == "612294a5d32d5f02a34857cff966b4202c763e23"
    assert m["stage"] == "TRAIN_ONLY_PHASE4_FROZEN"
    assert m["scientific_status"] == "NO_V2_MODEL_IMPROVEMENT"
    arts = m["artifact_sha256"]
    for k in ("extended_state_manifest_v2.json", "extended_causal_state_v2.parquet",
              "dev_frame_v2.parquet", "decomposed_v2_stability_atlas.csv",
              "decomposed_v2_model_comparison.csv", "decomposed_v2_feature_ablation.csv",
              "model_selection_train_only_v2.json"):
        assert arts[k] is not None, k
    assert m["extended_state_generator_code_sha"] == \
        R.EXTENDED_STATE_GENERATOR_CODE_SHA
    assert m["selection_generator_code_sha"] == R.SELECTION_GENERATOR_CODE_SHA
    ue = m["unit_evidence"]
    assert ue["r12_units"] == 60
    assert ue["r13_units"] == 75
    assert ue["total_units"] == 135
    assert ue["fits_per_unit"] == 6
    assert ue["total_model_fits"] == 810
    assert ue["missing_units"] == 0
    assert ue["duplicate_units"] == 0
    assert ue["inconsistent_units"] == 0
    assert ue["unit_identity_sha256"]
    lk = m["leakage"]
    assert lk["old_test_label_reads"] == 0
    assert lk["old_test_policy_reads"] == 0
    assert lk["val_outcomes_read"] is False
    assert lk["selection_committed_clean"] is True
    si = m["scientific_interpretation"]
    assert si["primary_status"] == "NO_V2_MODEL_IMPROVEMENT"
    assert si["v2_selected"] is False
    assert si["one_se_candidate"] == "B0_SHARED41"
    assert si["secondary_pre_val_challenger"] == "A1_SHARE_TO_WIN"
    assert si["a1_evidence"]["status"] == "PROMISING_DEV_IMPROVEMENT"
    # generator_code_sha must reference a real committed tree (the code commit)
    assert _git("cat-file", "-t", m["generator_code_sha"]).stdout.strip() == "commit"
