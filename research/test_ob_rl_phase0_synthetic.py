#!/usr/bin/env python3

"""Synthetic Gate-A test for Phase RL-0 (representation layer).

Covers: CSV -> Parquet exact roundtrip, key/column-order preservation,
the NO_OB vs NaN contract, Quantile UNKNOWN preservation, the Model
View whitelist (no META / no reward / no 4h / all TFs), and that the
audit report is generated without any strategy ranking.

NO real Gate-B CSV is read. Run with:

    python research/test_ob_rl_phase0_synthetic.py
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import build_ob_rl_parquet_v0 as PQ  # noqa: E402
from research import audit_ob_rl_dataset_v0 as AUD  # noqa: E402
from research import ob_rl_model_view_v0_spec as MV  # noqa: E402
from research.ob_rl_dataset_v0_spec import (  # noqa: E402
    VALIDATED_TFS,
    QUARANTINED_TFS,
)

RNG = np.random.default_rng(20260908)
TMP = Path(tempfile.mkdtemp(prefix="obrl_phase0_"))

RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    if not ok:
        print(f"FAIL  {name} :: {detail}", flush=True)


# ============================================================
# Synthetic warehouse
# ============================================================
N_CAND = 40
ACTIONS = (
    "SKIP",
    "FOLLOW_1.5R",
    "FOLLOW_2.0R",
    "FOLLOW_2.5R",
    "FADE_1.5R",
    "FADE_2.0R",
    "FADE_2.5R",
)

cids = [f"AG_C{i:05d}" for i in range(N_CAND)]

state = pd.DataFrame(
    {
        "candidate_id": cids,
        "candidate_group_id": [c + "_g" for c in cids],
        "symbol": ["AG"] * N_CAND,
        "source_tf": ["5m", "15m", "1h", "5m"] * (N_CAND // 4),
        "source_ob_structure": ["internal", "swing"] * (N_CAND // 2),
        "source_ob_bias": [1.0, -1.0] * (N_CAND // 2),
        "source_ob_width_atr5": RNG.uniform(0.2, 3.0, N_CAND),
        "touch_behavior": RNG.choice(
            ["NO_BREACH", "BREACH_RECLAIM", "CLOSE_BEYOND"],
            N_CAND,
        ),
        "touch_ordinal": RNG.integers(1, 6, N_CAND),
        "touch_time": pd.date_range(
            "2024-01-02 09:15", periods=N_CAND, freq="5min"
        ),
        "touch_5m_bar_index": np.arange(N_CAND) * 3,
        "trading_day": ["2024-01-02"] * N_CAND,
        "decision_weight": 1.0 / RNG.integers(1, 4, N_CAND),
        "quant_state": ["UNKNOWN"] * 10
        + list(
            RNG.choice(["LOW", "MID", "HIGH"], N_CAND - 10)
        ),
        "quant_width_percentile_train": [np.nan] * 10
        + list(RNG.uniform(0, 1, N_CAND - 10)),
        "quant_width": [np.nan] * 10
        + list(RNG.uniform(0, 0.05, N_CAND - 10)),
        "group_candidate_count": RNG.integers(1, 4, N_CAND),
    }
)

# Action-invariant state (must NOT change across 1.5R / 2.0R / 2.5R).
# Target-dependent columns (target_fit_*, stop_structure_*) are
# generated per action block instead.
base: dict[str, np.ndarray] = {}
no_ob_state = np.arange(N_CAND) % 7 == 0
no_struct_state = np.arange(N_CAND) % 11 == 0

for tf in VALIDATED_TFS:
    for c in MV.dsa_features(tf):
        base[c] = RNG.uniform(-2.0, 2.0, N_CAND)
    # stop_structure_* = backward structure / stop_atr, and V0 stop is
    # fixed at 1 ATR -> invariant across 1.5R / 2.0R / 2.5R.
    for k in ("internal", "swing"):
        v = RNG.uniform(0.1, 3.0, N_CAND)
        v[no_struct_state] = np.nan
        base[f"stop_structure_{k}_{tf}"] = v
    v = RNG.uniform(0.1, 3.0, N_CAND)
    v[no_ob_state] = np.nan
    base[f"stop_structure_ob_{tf}"] = v
    for c in (
        f"forward_active_ob_bias_rel_{tf}",
        f"backward_active_ob_bias_{tf}",
    ):
        b = RNG.choice([-1.0, 1.0], N_CAND)
        base[c] = np.where(no_ob_state, np.nan, b)
    for c in (
        f"forward_active_ob_structure_class_{tf}",
        f"backward_active_ob_structure_class_{tf}",
    ):
        sc = RNG.choice(["internal", "swing"], N_CAND)
        base[c] = np.where(no_ob_state, None, sc)
    base[f"momentum_direction_{tf}"] = RNG.choice(
        ["expanding", "contracting", "flat"], N_CAND
    )
    base[f"momentum_value_rel_{tf}"] = RNG.normal(0, 1, N_CAND)
    base[f"momentum_delta_rel_{tf}"] = RNG.normal(0, 1, N_CAND)
    base[f"internal_bias_rel_{tf}"] = RNG.choice(
        [-1.0, 0.0, 1.0], N_CAND
    )
    base[f"swing_bias_rel_{tf}"] = RNG.choice(
        [-1.0, 0.0, 1.0], N_CAND
    )

frames = []
for action in ACTIONS:
    x = state.copy()
    x["action"] = action
    if action == "SKIP":
        x["trade_mode"] = "SKIP"
        x["trade_direction"] = 0.0
        x["stop_atr"] = 0.0
        x["target_R"] = 0.0
    else:
        mode, rr = action.split("_")
        x["trade_mode"] = mode
        x["trade_direction"] = x["source_ob_bias"] * (
            1.0 if mode == "FOLLOW" else -1.0
        )
        x["stop_atr"] = 1.0
        x["target_R"] = float(rr[:-1])
    x["target_atr"] = x["stop_atr"] * x["target_R"]

    skip_mask = x["trade_mode"].to_numpy() == "SKIP"
    # every 7th row has NO active OB in front/behind
    no_ob = np.arange(len(x)) % 7 == 0
    # every 11th row has no internal/swing structure recorded
    no_struct = np.arange(len(x)) % 11 == 0

    for tf in VALIDATED_TFS:
        for c in MV.smc_features(tf):
            if c.startswith("target_fit_"):
                # target-dependent: differs across 1.5R / 2.0R / 2.5R
                v = RNG.uniform(0.1, 3.0, len(x))
                v[no_ob if "_ob_" in c else no_struct] = np.nan
                x[c] = np.where(skip_mask, np.nan, v)
            else:
                # action-invariant (V0 stop fixed at 1 ATR)
                x[c] = np.where(skip_mask, np.nan, base[c])
        for c in MV.dsa_features(tf):
            x[c] = base[c]
        for c in MV.momentum_features(tf):
            x[c] = base[c]
        # warehouse-only columns that must NOT reach the model view
        x[f"dsa_raw_regime_strength_{tf}"] = 0.0
        x[f"internal_high_level_{tf}"] = 7100.0
        x[f"forward_internal_atr_{tf}"] = 1.0

    for h in (6, 12, 24):
        r = RNG.normal(0, 1, len(x))
        r[RNG.random(len(x)) < 0.2] = np.nan
        x[f"gross_R_h{h}"] = (
            0.0 if action == "SKIP" else r
        )
        x[f"exit_code_h{h}"] = (
            -1
            if action == "SKIP"
            else RNG.choice([1, 5, 6, 7], len(x))
        )
    x["primary_reward_R"] = x["gross_R_h12"]
    x["reward_version"] = "GROSS_R_V0"
    frames.append(x)

action = pd.concat(frames, ignore_index=True)

OUT = TMP / "out"
OUT.mkdir(parents=True, exist_ok=True)
state.to_csv(OUT / "ob_rl_state_v0.csv", index=False)
action.to_csv(OUT / "ob_rl_action_v0.csv", index=False)
(OUT / "dataset_manifest.json").write_text(
    json.dumps({"dataset_version": "ob_rl_dataset_v0"}),
    encoding="utf-8",
)

# ============================================================
# 1) Gate-B hash lock
# ============================================================
# NOTE: the audit module imports EXPECTED_GATE_B_FILES by value, so it
# must be pointed at the synthetic lock explicitly.
PQ.EXPECTED_GATE_B_FILES = {
    "ob_rl_state_v0.csv": PQ.sha256_file(
        OUT / "ob_rl_state_v0.csv"
    ),
    "ob_rl_action_v0.csv": PQ.sha256_file(
        OUT / "ob_rl_action_v0.csv"
    ),
    "dataset_manifest.json": PQ.sha256_file(
        OUT / "dataset_manifest.json"
    ),
}
AUD.EXPECTED_GATE_B_FILES = PQ.EXPECTED_GATE_B_FILES
try:
    PQ.verify_gate_b_files(OUT)
    check("gate-B hash lock passes on matching files", True)
except RuntimeError as e:
    check("gate-B hash lock passes on matching files", False, e)

(OUT / "ob_rl_state_v0.csv").write_bytes(b"tampered")
try:
    PQ.verify_gate_b_files(OUT)
    check("gate-B hash lock blocks tampered file", False, "no raise")
except RuntimeError as e:
    check(
        "gate-B hash lock blocks tampered file",
        "hash mismatch" in str(e),
        str(e)[:90],
    )
state.to_csv(OUT / "ob_rl_state_v0.csv", index=False)

# ============================================================
# 2) CSV -> Parquet -> readback exact parity
# ============================================================
PQ.OUT_ROOT = OUT
PQ.require_pyarrow()
# synthetic scale
PQ.EXPECTED_STATE_ROWS = N_CAND
PQ.EXPECTED_ACTION_ROWS = N_CAND * len(ACTIONS)

# --- staged write: nothing final exists before verification ---
probe = PQ.stage_parquet(
    OUT / "ob_rl_state_v0.csv",
    OUT / "probe_state.parquet",
    key_cols=("candidate_id",),
    expected_rows=N_CAND,
)
check(
    "staged write creates tmp only (no final artifact)",
    probe["tmp"].exists() and not probe["final"].exists(),
)
check(
    "staged verification is exact",
    probe["result"]["content_exact"],
)
probe["tmp"].unlink(missing_ok=True)

# --- forced mismatch: cardinality ---
try:
    PQ.stage_parquet(
        OUT / "ob_rl_action_v0.csv",
        OUT / "probe_fail.parquet",
        key_cols=("candidate_id", "action"),
        expected_rows=N_CAND * len(ACTIONS) + 1,
    )
    check("staged failure raises", False, "no raise")
except RuntimeError:
    check("staged failure raises", True)
check(
    "failed stage leaves no final and no tmp",
    not (OUT / "probe_fail.parquet").exists()
    and not (
        OUT / ".probe_fail.parquet.tmp"
    ).exists(),
)

# --- forced mismatch: content ---
df_state = pd.read_csv(
    OUT / "ob_rl_state_v0.csv", low_memory=False
)
bad = df_state.drop(columns=[df_state.columns[-1]])
bad_path = OUT / "probe_content.parquet"
bad.to_parquet(
    bad_path, engine="pyarrow", compression="zstd", index=False
)
try:
    PQ.verify_parquet(
        df_state,
        bad_path,
        key_cols=("candidate_id",),
        expected_rows=N_CAND,
    )
    check("content mismatch detected", False, "no raise")
except Exception:
    check("content mismatch detected", True)
bad_path.unlink(missing_ok=True)

# --- real staged conversion: both verified, then promoted ---
receipt = PQ.convert_all(OUT)
receipt_path = PQ.write_receipt(OUT, receipt)
check(
    "both parquet promoted after joint verification",
    (OUT / "ob_rl_state_v0.parquet").exists()
    and (OUT / "ob_rl_action_v0.parquet").exists(),
)
check(
    "no staging leftovers after promotion",
    not (OUT / ".ob_rl_state_v0.parquet.tmp").exists()
    and not (OUT / ".ob_rl_action_v0.parquet.tmp").exists(),
)
check(
    "parquet roundtrip state exact",
    receipt["state"]["content_exact"],
)
check(
    "parquet roundtrip action exact",
    receipt["action"]["content_exact"],
)

# --- receipt content ---
check(
    "receipt records engine/compression",
    receipt["engine"] == "pyarrow"
    and receipt["compression"] == "zstd",
)
check(
    "receipt records parity flags for both tables",
    all(
        receipt[t][k] is True
        for t in ("state", "action")
        for k in ("schema_match", "key_match", "content_exact")
    ),
)
check(
    "receipt records csv + parquet sha256",
    all(
        len(receipt[t]["csv_sha256"]) == 64
        and len(receipt[t]["parquet_sha256"]) == 64
        for t in ("state", "action")
    ),
)
check(
    "receipt records provenance SHAs",
    receipt["gate_b_dataset_builder_sha"]
    == "c475239e872246fcee64ca439f7d3d8e550a0c4e"
    and len(receipt["representation_code_sha"]) == 40,
)
check("receipt file exists", receipt_path.exists())

# --- pre-existing final artifact must block conversion ---
try:
    PQ.convert_all(OUT)
    check("pre-existing final blocks conversion", False, "no raise")
except RuntimeError as e:
    check(
        "pre-existing final blocks conversion",
        "already exists before conversion" in str(e),
        str(e)[:90],
    )

# --- partial promotion failure rolls back EVERYTHING ---
t1 = OUT / "rb_t1.parquet"
t1.write_bytes(b"x")
f1 = OUT / "rb_f1.parquet"
t2 = OUT / "rb_t2.parquet"
t2.write_bytes(b"x")
f2 = OUT / "rb_f2_dir"
f2.mkdir(exist_ok=True)
try:
    PQ.promote_all(
        [{"tmp": t1, "final": f1}, {"tmp": t2, "final": f2}]
    )
    check("partial promote failure raises", False, "no raise")
except Exception:
    check("partial promote failure raises", True)
check(
    "rollback removes already-promoted final",
    not f1.exists(),
)
check(
    "rollback removes both tmp files",
    not t1.exists() and not t2.exists(),
)
shutil.rmtree(f2, ignore_errors=True)

# ============================================================
# 2b) Receipt must be VERIFIED, not trusted
# ============================================================
verified = AUD.verify_receipt(OUT, receipt)
check(
    "verify_receipt passes on intact files",
    verified["state"]["content_exact"] is True
    and verified["action"]["content_exact"] is True,
)

orig_bytes = (OUT / "ob_rl_state_v0.parquet").read_bytes()
(OUT / "ob_rl_state_v0.parquet").write_bytes(orig_bytes + b"x")
try:
    AUD.verify_receipt(OUT, receipt)
    check("tampered parquet stops audit", False, "no raise")
except RuntimeError as e:
    check(
        "tampered parquet stops audit",
        "does not match receipt" in str(e),
        str(e)[:90],
    )
(OUT / "ob_rl_state_v0.parquet").write_bytes(orig_bytes)

bad_flag = copy.deepcopy(receipt)
bad_flag["state"]["content_exact"] = False
try:
    AUD.verify_receipt(OUT, bad_flag)
    check("false parity flag stops audit", False, "no raise")
except RuntimeError as e:
    check(
        "false parity flag stops audit",
        "parity flag" in str(e),
        str(e)[:90],
    )

bad_rows = copy.deepcopy(receipt)
bad_rows["action"]["rows"] = receipt["action"]["rows"] + 1
try:
    AUD.verify_receipt(OUT, bad_rows)
    check("receipt row drift stops audit", False, "no raise")
except RuntimeError as e:
    check(
        "receipt row drift stops audit",
        "receipt row drift" in str(e),
        str(e)[:90],
    )

bad_ver = copy.deepcopy(receipt)
bad_ver["representation_version"] = "SOMETHING_ELSE"
try:
    AUD.verify_receipt(OUT, bad_ver)
    check("representation version drift stops", False, "no raise")
except RuntimeError as e:
    check(
        "representation version drift stops",
        "version drift" in str(e),
        str(e)[:90],
    )

# 3) column order preserved
pq_state = pd.read_parquet(OUT / "ob_rl_state_v0.parquet")
check(
    "state column order preserved",
    list(pq_state.columns) == list(state.columns),
)
pq_action = pd.read_parquet(OUT / "ob_rl_action_v0.parquet")
check(
    "action column order preserved",
    list(pq_action.columns) == list(action.columns),
)

# 4) keys preserved
check(
    "state key preserved",
    bool(pq_state["candidate_id"].is_unique)
    and pq_state["candidate_id"].astype(str).tolist()
    == state["candidate_id"].astype(str).tolist(),
)
check(
    "action key preserved",
    not pq_action.duplicated(["candidate_id", "action"]).any(),
)

# ============================================================
# 5-7) Missing-value contract
# ============================================================
non_skip = pq_action[pq_action["action"] != "SKIP"]
# NaN was stamped per action BLOCK, so positions repeat every N_CAND.
block_pos = np.arange(len(non_skip)) % N_CAND
nan_rows = block_pos % 7 == 0
fit = non_skip["target_fit_ob_5m"].to_numpy(float)
check(
    "numeric OB distance stays NaN (not 0)",
    bool(np.isnan(fit[nan_rows]).all())
    and not bool((fit == 0).any()),
    fit[nan_rows][:5],
)

sc = non_skip["forward_active_ob_structure_class_5m"]
check(
    "categorical NO_OB contract available",
    MV.CATEGORICAL_MISSING_V0[
        "forward_active_ob_structure_class"
    ]
    == "NO_OB",
)
filled = sc.where(sc.notna(), "NO_OB")
check(
    "NO_OB fill does not touch numeric NaN",
    bool(np.isnan(fit[nan_rows]).all())
    and (filled.to_numpy()[nan_rows] == "NO_OB").all(),
)

qs = state["quant_state"]
check(
    "Quantile UNKNOWN preserved",
    int((qs == "UNKNOWN").sum()) == 10
    and bool(
        state.loc[
            qs == "UNKNOWN", "quant_width_percentile_train"
        ]
        .isna()
        .all()
    ),
)

# ============================================================
# 8-13) Model View whitelist
# ============================================================
try:
    mv = MV.audit_model_features(pq_action)
    check("model feature audit passes", True)
except RuntimeError as e:
    check("model feature audit passes", False, repr(e))
    mv = {}

check(
    "model feature count == 62",
    len(MV.MODEL_FEATURES_V0) == 62,
    len(MV.MODEL_FEATURES_V0),
)
check(
    "no META in model features",
    not (set(MV.MODEL_FEATURES_V0) & set(MV.MODEL_META_V0)),
)
check(
    "no WEIGHT in model features",
    not (set(MV.MODEL_FEATURES_V0) & set(MV.MODEL_WEIGHT_V0)),
)
check(
    "no reward in model features",
    not [
        c
        for c in MV.MODEL_FEATURES_V0
        if c.startswith("gross_R")
        or c.startswith("exit_code")
        or "reward" in c
    ],
)
check(
    "no 4h in model features",
    not [
        c
        for c in MV.MODEL_FEATURES_V0
        if any(tf in c for tf in QUARANTINED_TFS)
    ],
)
check(
    "source_ob_width_atr5 present",
    "source_ob_width_atr5" in MV.MODEL_FEATURES_V0,
)
for tf in VALIDATED_TFS:
    for need in MV.smc_features(tf):
        check(f"SMC {need}", need in MV.MODEL_FEATURES_V0)
    for need in MV.dsa_features(tf):
        check(f"DSA {need}", need in MV.MODEL_FEATURES_V0)
    for need in MV.momentum_features(tf):
        check(f"MOM {need}", need in MV.MODEL_FEATURES_V0)
for need in MV.ACTION_FEATURES_V0:
    check(f"ACTION {need}", need in MV.MODEL_FEATURES_V0)
check(
    "warehouse-only columns excluded",
    "dsa_raw_regime_strength_5m" not in MV.MODEL_FEATURES_V0
    and "internal_high_level_5m" not in MV.MODEL_FEATURES_V0
    and "forward_internal_atr_5m" not in MV.MODEL_FEATURES_V0,
)

# audit-only binning
check(
    "rr_fit_bin thresholds",
    MV.rr_fit_bin(np.nan) == "NO_LEVEL"
    and MV.rr_fit_bin(0.5) == "BEYOND_STRUCTURE"
    and MV.rr_fit_bin(1.0) == "FITS_BEFORE_STRUCTURE"
    and MV.rr_fit_bin(2.5) == "FITS_BEFORE_STRUCTURE",
)
check(
    "quantile audit bin uses PIT percentile directly",
    MV.quantile_audit_bin(0.10) == "Q1"
    and MV.quantile_audit_bin(0.95) == "Q5"
    and MV.quantile_audit_bin(np.nan) == "UNKNOWN",
)
for col in (
    "dsa_vwap_dev_rel_5m",
    "momentum_value_rel_15m",
    "internal_bias_rel_1h",
    # V0 stop is fixed at 1 ATR -> invariant across RR
    "stop_structure_ob_1h",
    "stop_structure_internal_5m",
):
    c = MV.collapse_for_descriptive_rank(
        pq_action, (col,)
    )
    check(
        f"collapse action-invariant PASS: {col}",
        not c.duplicated(
            ["candidate_id", "trade_mode"]
        ).any()
        and len(c) == N_CAND * 2,
        len(c),
    )

for col in ("target_fit_swing_5m", "target_R"):
    try:
        MV.collapse_for_descriptive_rank(
            pq_action, (col,)
        )
        check(
            f"collapse target-dependent STOP: {col}",
            False,
            "no raise",
        )
    except RuntimeError as e:
        check(
            f"collapse target-dependent STOP: {col}",
            "varies across target actions" in str(e),
            str(e)[:90],
        )

try:
    MV.collapse_for_descriptive_rank(
        pq_action, ("does_not_exist",)
    )
    check("collapse missing column STOP", False, "no raise")
except RuntimeError as e:
    check(
        "collapse missing column STOP",
        "rank columns missing" in str(e),
        str(e)[:90],
    )

# ============================================================
# 14) Audit report generation, no strategy ranking
# ============================================================
AUD.OUT_ROOT = OUT
try:
    AUD.main()
    check("audit report generated", True)
except Exception as e:
    check("audit report generated", False, repr(e))
    import traceback

    traceback.print_exc()

for f in ("dataset_audit.json", "dataset_columns.json", "README.md"):
    check(f"audit output {f}", (OUT / f).exists())

if (OUT / "dataset_audit.json").exists():
    blob = json.loads(
        (OUT / "dataset_audit.json").read_text()
    )
    check(
        "audit has required sections",
        all(
            k in blob
            for k in (
                "dataset",
                "storage",
                "coverage",
                "reward_integrity",
                "model_view",
            )
        ),
        list(blob.keys()),
    )
    txt = json.dumps(blob).lower()
    check(
        "no ranking language in audit",
        not [
            s
            for s in AUD.FORBIDDEN_OUTPUT_SUBSTRINGS
            if s in txt
        ],
    )
    check(
        "action rows == 7N in audit",
        blob["dataset"]["action_rows"] == N_CAND * len(ACTIONS),
    )
    check(
        "skip semantics reported",
        blob["reward_integrity"]["skip"]["gross_R_all_zero"]
        and blob["reward_integrity"]["skip"][
            "exit_code_all_minus_one"
        ],
    )

try:
    AUD.guard_no_ranking({"x": "best_action"})
    check("ranking guard fires", False, "no raise")
except RuntimeError:
    check("ranking guard fires", True)

# ============================================================
# 15) Finalized manifest + provenance evidence chain
# ============================================================
fm = json.loads((OUT / "dataset_manifest.json").read_text())
check("manifest rl0_finalized", fm.get("rl0_finalized") is True)
check(
    "gate_b original manifest hash preserved",
    fm.get("gate_b_manifest_sha256_original")
    == PQ.EXPECTED_GATE_B_FILES["dataset_manifest.json"],
)
check(
    "source_data_baseline_sha recorded",
    fm.get("source_data_baseline_sha")
    == "0b0caad7ddba837f4d837c1fbdd990abe28f6167",
)
check(
    "gate_b_dataset_builder_sha recorded",
    fm.get("gate_b_dataset_builder_sha")
    == "c475239e872246fcee64ca439f7d3d8e550a0c4e",
)
check(
    "representation_code_sha comes from the receipt",
    fm.get("representation_code_sha")
    == receipt["representation_code_sha"],
)
check(
    "audit_code_sha recorded and equals representation SHA",
    len(str(fm.get("audit_code_sha", ""))) == 40
    and fm.get("audit_code_sha")
    == fm.get("representation_code_sha"),
)

drifted = copy.deepcopy(receipt)
drifted["representation_code_sha"] = "0" * 40
try:
    AUD.write_final_manifest(
        OUT,
        receipt=drifted,
        verified_receipt=verified,
        extra={},
    )
    check("representation/audit HEAD drift stops", False, "no raise")
except RuntimeError as e:
    check(
        "representation/audit HEAD drift stops",
        "HEAD drift" in str(e),
        str(e)[:90],
    )
check(
    "model_view_version recorded",
    fm.get("model_view_version") == MV.MODEL_VIEW_VERSION,
)
check(
    "parquet parity persisted in manifest",
    fm["parquet"]["state"]["content_exact"] is True
    and fm["parquet"]["action"]["content_exact"] is True
    and fm["parquet"]["state"]["schema_match"] is True
    and fm["parquet"]["action"]["key_match"] is True,
)

# audit must not run without a receipt
backup = (OUT / PQ.RECEIPT_NAME).read_text()
(OUT / PQ.RECEIPT_NAME).unlink()
try:
    AUD.main()
    check("audit stops without receipt", False, "no raise")
except RuntimeError as e:
    check(
        "audit stops without receipt",
        "receipt missing" in str(e),
        str(e)[:90],
    )
(OUT / PQ.RECEIPT_NAME).write_text(backup)

# re-running audit recognises the already-finalized manifest
try:
    AUD.main()
    check("audit idempotent after finalization", True)
except RuntimeError as e:
    check("audit idempotent after finalization", False, repr(e))

# a manifest that is neither Gate-B nor finalized must STOP
(OUT / "dataset_manifest.json").write_text(
    json.dumps({"dataset_version": "tampered"}), encoding="utf-8"
)
try:
    AUD.verify_original_manifest(OUT)
    check("foreign manifest STOP", False, "no raise")
except RuntimeError as e:
    check(
        "foreign manifest STOP",
        "neither the audited Gate-B file" in str(e),
        str(e)[:90],
    )

# ============================================================
print("\n========= PHASE RL-0 SYNTHETIC RESULT =========")
fails = [r for r in RESULTS if not r[1]]
for name, _, det in fails:
    print(f"FAIL  {name} :: {det}")
print(
    f"checks={len(RESULTS)} pass={len(RESULTS) - len(fails)} "
    f"fail={len(fails)}"
)
print("===============================================")
shutil.rmtree(TMP, ignore_errors=True)
raise SystemExit(1 if fails else 0)
