"""FUTURE-R9AB-R10-M15-DECOMPOSED-WIN-PAYOFF-V1 — PRE-TEST evidence generator.

Supersedes the older unified "Opportunity Value Model" framing. Emits PRE-TEST
evidence ONLY:

  - R8 provenance       : reuses the existing frozen R8 upstream artifacts and
                          derives WIN33 / PAY8 into artifacts/decomposed_value_v1/.
  - R9A Win-Probability : fit 3 horizon classifiers on TRAIN, VAL diagnostics
                          (10 prob deciles). TRAIN/VAL only. No TEST read.
  - R9B Payoff-Ratio    : fit 3 horizons x 2 conditional heads on TRAIN, VAL
                          diagnostics (10 RR deciles). TRAIN/VAL only.
  - R9C Composer        : NO model fit. TRAIN priors + VAL 5x5 grid + 10 EV
                          deciles (deterministic arithmetic only).
  - R10 synthetic ledger: deterministic synthetic-axis PRE-TEST runner (no real
                          TEST, no models, no labels).

It NEVER reads TEST labels and NEVER executes a real TEST prediction / simulation.
The emitted JSON is the lineage artifact consumed by
sequential_decomposed_policy_v1.run_formal_opportunity_value_test (which expects
research/liquidity_oracle_atlas/evidence/decomposed_value_renewal_v1_pretest_summary.json).
"""

import json
import os
import subprocess
import time

from research.liquidity_oracle_atlas.entry_path_atlas_v1 import SYMBOLS
from research.liquidity_oracle_atlas.build_decomposed_value_dataset_v1 import (
    materialize_decomposed,
    load_manifest,
)
from research.liquidity_oracle_atlas import win_probability_model_v1 as R9A
from research.liquidity_oracle_atlas import payoff_ratio_model_v1 as R9B
from research.liquidity_oracle_atlas import decomposed_value_composer_v1 as R9C
from research.liquidity_oracle_atlas import sequential_decomposed_policy_v1 as R10

EVIDENCE_DIR = os.path.join("research", "liquidity_oracle_atlas", "evidence")
OUT_JSON = os.path.join(EVIDENCE_DIR, "decomposed_value_renewal_v1_pretest_summary.json")


def _git_head_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(__file__), stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def main(verbose: bool = True):
    t0 = time.time()

    # 1. Build decomposed WIN33 / PAY8 (reuse frozen R8 upstream; no R8 rebuild).
    if verbose:
        print("[1/6] materialize decomposed WIN33/PAY8 (run_r8=False) ...",
              flush=True)
    dec_manifest = materialize_decomposed(
        symbols=SYMBOLS, run_r8=False, verbose=False)
    r8_manifest = load_manifest()  # artifacts/decomposed_value_v1/r8_manifest_v1.json

    # 2. R9A: Win-Probability (TRAIN/VAL only).
    if verbose:
        print("[2/6] R9A win-probability fit + VAL diagnostics ...", flush=True)
    R9A.reset_counters()
    win_frames = R9A.load_fit_frames()
    win_bundles = R9A.fit_all(win_frames, verbose=False)
    win_diag = R9A.val_diagnostics(win_bundles, win_frames)
    win_model_manifest = R9A.freeze_models(
        win_bundles, win_diag, frames=win_frames)

    # 3. R9B: Payoff-Ratio (TRAIN/VAL only).
    if verbose:
        print("[3/6] R9B payoff-ratio fit + VAL diagnostics ...", flush=True)
    R9B.reset_counters()
    pay_frames = R9B.load_fit_frames()
    pay_bundles = R9B.fit_all(pay_frames, verbose=False)
    pay_diag = R9B.val_diagnostics(pay_bundles, pay_frames)
    pay_model_manifest = R9B.freeze_models(
        pay_bundles, pay_diag, frames=pay_frames)

    # 4. R9C: Composer (no model) — TRAIN priors + VAL combined diagnostics.
    if verbose:
        print("[4/6] R9C composer priors + VAL combined diagnostics ...",
              flush=True)
    priors = R9C.compute_train_priors()
    combined = R9C.val_combined_diagnostics(win_bundles, pay_bundles, priors)
    r9c_model_fits = int(R9C.COUNTERS["model_fits"])

    # 5. R10: synthetic-ledger PRE-TEST (deterministic, no real TEST).
    if verbose:
        print("[5/6] R10 synthetic-ledger PRE-TEST ...", flush=True)
    pre_res, _trades, _dec = R10.run_pretest_synthetic_evidence(verbose=False)

    # 6. Assemble + write.
    if verbose:
        print("[6/6] assemble + write evidence JSON ...", flush=True)
    evidence = {
        "task_ids": [
            "FUTURE-R8-M15-STRUCTURAL-RENEWAL-DATASET-V1",
            "FUTURE-R9A-M15-WIN-PROBABILITY-MODEL-V1",
            "FUTURE-R9B-M15-PAYOFF-RATIO-MODEL-V1",
            "FUTURE-R9C-M15-MATHEMATICAL-COMPOSER-V1",
            "FUTURE-R10-M15-SEQUENTIAL-DECOMPOSED-POLICY-V1",
        ],
        "program": "FUTURE-R9AB-R10-M15-DECOMPOSED-WIN-PAYOFF-V1",
        "stage": "PRE_TEST",
        "generator_code_sha": _git_head_sha(),
        "base_sha": dec_manifest.get("base_sha"),
        "scientific_status":
            "LOCKED_DEVELOPMENT_TEST_NOT_PRISTINE_CONFIRMATION",
        # Embedded verbatim so the formal runner's lineage gate can verify each
        # DEC_ARTIFACT_DIR artifact SHA without re-reading the manifest file.
        "r8_manifest": r8_manifest,
        "r9a": {
            "model_manifest": win_model_manifest,
            "val_diagnostics": win_diag,
            "fit_counters": dict(R9A.COUNTERS),
        },
        "r9b": {
            "model_manifest": pay_model_manifest,
            "val_diagnostics": pay_diag,
            "fit_counters": dict(R9B.COUNTERS),
        },
        "r9c": {
            "train_priors": priors,
            "val_combined_diagnostics": combined,
            "model_fits": r9c_model_fits,
        },
        "r10_pretest_synthetic": pre_res,
        "governance": {
            "test_labels_read": False,
            "test_prediction_executed": False,
            "formal_test_executed": False,
            "e9_root_axis_materialized": False,
            "R9A_counters": dict(R9A.COUNTERS),
            "R9B_counters": dict(R9B.COUNTERS),
            "R9C_counters": dict(R9C.COUNTERS),
            "R10_counters": dict(R10.COUNTERS),
        },
        "runtime_sec": time.time() - t0,
    }

    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(evidence, f, indent=2, default=str)
    if verbose:
        print(f"WROTE {OUT_JSON} in {evidence['runtime_sec']:.1f}s", flush=True)
        print("  R9A classifier_fits =", R9A.COUNTERS["classifier_fits"])
        print("  R9B payoff_regressor_fits =", R9B.COUNTERS["payoff_regressor_fits"])
        print("  R9C model_fits =", r9c_model_fits)
        print("  R10 performance_gate_pass =", pre_res.get("performance_gate_pass"))
    return evidence


if __name__ == "__main__":
    main()
