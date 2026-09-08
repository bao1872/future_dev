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
            if c.endswith("structure_class"):
                x[c] = np.where(
                    no_ob,
                    None,
                    RNG.choice(["internal", "swing"], len(x)),
                )
            elif "_ob_" in c:
                # numeric OB geometry: NaN whenever there is no OB.
                # Must NEVER be 0 (NO_OB != distance 0).
                v = RNG.uniform(0.1, 3.0, len(x))
                v[no_ob] = np.nan
                x[c] = np.where(skip_mask, np.nan, v)
            elif c.endswith("_bias") or c.endswith("_bias_rel"):
                x[c] = RNG.choice([-1.0, 1.0], len(x))
            else:
                v = RNG.uniform(0.1, 3.0, len(x))
                v[no_struct] = np.nan
                x[c] = np.where(skip_mask, np.nan, v)
        for c in MV.dsa_features(tf):
            x[c] = RNG.uniform(-2, 2, len(x))
        x[f"momentum_direction_{tf}"] = RNG.choice(
            ["expanding", "contracting", "flat"], len(x)
        )
        x[f"momentum_value_rel_{tf}"] = RNG.normal(0, 1, len(x))
        x[f"momentum_delta_rel_{tf}"] = RNG.normal(0, 1, len(x))
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

state_csv = PQ.write_parquet(
    OUT / "ob_rl_state_v0.csv", OUT / "ob_rl_state_v0.parquet"
)
st_res = PQ.verify_parquet(
    state_csv,
    OUT / "ob_rl_state_v0.parquet",
    key_cols=("candidate_id",),
    expected_rows=N_CAND,
)
check("parquet roundtrip state exact", st_res["content_exact"])

action_csv = PQ.write_parquet(
    OUT / "ob_rl_action_v0.csv", OUT / "ob_rl_action_v0.parquet"
)
ac_res = PQ.verify_parquet(
    action_csv,
    OUT / "ob_rl_action_v0.parquet",
    key_cols=("candidate_id", "action"),
    expected_rows=N_CAND * len(ACTIONS),
)
check("parquet roundtrip action exact", ac_res["content_exact"])

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
collapsed = MV.collapse_for_descriptive_rank(pq_action)
check(
    "descriptive rank collapses to candidate x trade_mode",
    not collapsed.duplicated(
        ["candidate_id", "trade_mode"]
    ).any()
    and len(collapsed) == N_CAND * 2,
    len(collapsed),
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
