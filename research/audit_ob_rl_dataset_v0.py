#!/usr/bin/env python3

"""Dataset Audit for OB RL Dataset V0.

Produces (git-trackable, small):

    dataset_audit.json      dataset / storage / coverage / reward
    dataset_columns.json    column roles for all 232 action columns
    README.md               human-readable description

This module answers "is the dataset sound". It does NOT search for an
edge, rank actions, pick a best RR, or recommend a rule; a hard guard
blocks those words from ever appearing in the outputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.build_ob_rl_dataset_v0 import OUT_ROOT  # noqa: E402
from research.build_ob_rl_parquet_v0 import (  # noqa: E402
    EXPECTED_GATE_B_FILES,
    sha256_file,
)
from research.ob_rl_dataset_v0_spec import (  # noqa: E402
    VALIDATED_TFS,
    DIAGNOSTIC_HORIZONS,
)
from research.ob_rl_model_view_v0_spec import (  # noqa: E402
    MODEL_FEATURES_V0,
    MODEL_META_V0,
    MODEL_WEIGHT_V0,
    CATEGORICAL_FEATURES_V0,
    CONTINUOUS_FEATURES_V0,
    CATEGORICAL_MISSING_V0,
    WAREHOUSE_ONLY_FAMILIES,
    MODEL_VIEW_ROLE_MAP,
    MODEL_VIEW_VERSION,
    audit_model_features,
)

EXIT_NAMES = {
    0: "INSUFFICIENT",
    1: "NONCONTIG",
    2: "GAP_STOP",
    3: "GAP_TARGET",
    4: "BOTH",
    5: "STOP",
    6: "TARGET",
    7: "TIMEOUT",
    -1: "SKIP",
}

FORBIDDEN_OUTPUT_SUBSTRINGS = (
    "top_feature",
    "best_action",
    "best_rr",
    "positive_edge",
    "recommended_rule",
    "recommendation",
)


def _read(name: str, columns: list[str] | None = None):
    pq = OUT_ROOT / f"{name}.parquet"
    if pq.exists():
        return pd.read_parquet(pq, columns=columns)
    return pd.read_csv(
        OUT_ROOT / f"{name}.csv",
        low_memory=False,
        usecols=columns,
    )


def _full_columns(name: str) -> list[str]:
    """Full warehouse column list without loading the frame."""
    pq = OUT_ROOT / f"{name}.parquet"
    if pq.exists():
        try:
            import pyarrow.parquet as pqmod

            return list(
                pqmod.ParquetFile(pq).schema_arrow.names
            )
        except Exception:
            return list(pd.read_parquet(pq).columns)
    return list(
        pd.read_csv(OUT_ROOT / f"{name}.csv", nrows=0).columns
    )


def guard_no_ranking(payload: dict) -> None:
    blob = json.dumps(payload).lower()
    hits = [
        s for s in FORBIDDEN_OUTPUT_SUBSTRINGS if s in blob
    ]
    if hits:
        raise RuntimeError(
            f"audit output contains ranking language: {hits}"
        )


def nan_rate(s: pd.Series) -> float:
    return float(s.isna().mean())


def main() -> None:
    state = _read("ob_rl_state_v0")

    action_cols = list(
        dict.fromkeys(
            list(MODEL_FEATURES_V0)
            + list(MODEL_META_V0)
            + list(MODEL_WEIGHT_V0)
            + [
                "action",
                "stop_atr",
                "target_atr",
                "gross_R_h6",
                "gross_R_h12",
                "gross_R_h24",
                "exit_code_h6",
                "exit_code_h12",
                "exit_code_h24",
                "primary_reward_R",
                "reward_version",
            ]
        )
    )
    action = _read("ob_rl_action_v0", columns=action_cols)

    mv = audit_model_features(action)

    non_skip = action[action["action"].astype(str) != "SKIP"]
    skip = action[action["action"].astype(str) == "SKIP"]

    # ---------------- dataset ----------------
    dataset = {
        "state_rows": int(len(state)),
        "state_columns": int(state.shape[1]),
        "action_rows": int(len(action)),
        "action_columns_universe": int(
            len(_full_columns("ob_rl_action_v0"))
        ),
        "state_key": {
            "candidate_id_unique": int(
                state["candidate_id"].nunique()
            ),
            "candidate_id_duplicates": int(
                state["candidate_id"].duplicated().sum()
            ),
        },
        "action_key": {
            "(candidate_id, action)_duplicates": int(
                action.duplicated(["candidate_id", "action"]).sum()
            )
        },
        "symbol_counts": {
            str(k): int(v)
            for k, v in state["symbol"].value_counts().items()
        },
        "source_tf_counts": {
            str(k): int(v)
            for k, v in state["source_tf"].value_counts().items()
        },
        "action_counts": {
            str(k): int(v)
            for k, v in action["action"].value_counts().items()
        },
    }

    # ---------------- storage ----------------
    storage = {
        "gate_b_expected": {},
        "gate_b_actual": {},
        "parquet": {},
    }
    for name in EXPECTED_GATE_B_FILES:
        p = OUT_ROOT / name
        storage["gate_b_expected"][name] = EXPECTED_GATE_B_FILES[
            name
        ]
        storage["gate_b_actual"][name] = (
            sha256_file(p) if p.exists() else None
        )
    for name in ("ob_rl_state_v0", "ob_rl_action_v0"):
        pq = OUT_ROOT / f"{name}.parquet"
        if pq.exists():
            storage["parquet"][name] = {
                "bytes": int(pq.stat().st_size),
                "sha256": sha256_file(pq),
            }

    # ---------------- coverage ----------------
    coverage = {"smc": {}, "active_ob": {}, "dsa": {},
                "momentum": {}, "quantile": {}}
    for tf in VALIDATED_TFS:
        coverage["smc"][tf] = {
            f"target_fit_{k}_nan_rate": round(
                nan_rate(non_skip[f"target_fit_{k}_{tf}"]), 6
            )
            for k in ("internal", "swing", "ob")
        }
        coverage["smc"][tf].update(
            {
                f"stop_structure_{k}_nan_rate": round(
                    nan_rate(
                        non_skip[f"stop_structure_{k}_{tf}"]
                    ),
                    6,
                )
                for k in ("internal", "swing", "ob")
            }
        )
        coverage["active_ob"][tf] = {
            "forward_structure_class_nan_rate": round(
                nan_rate(
                    non_skip[
                        f"forward_active_ob_structure_class_{tf}"
                    ]
                ),
                6,
            ),
            "backward_structure_class_nan_rate": round(
                nan_rate(
                    non_skip[
                        f"backward_active_ob_structure_class_{tf}"
                    ]
                ),
                6,
            ),
            "categorical_missing_token": CATEGORICAL_MISSING_V0[
                "forward_active_ob_structure_class"
            ],
        }
        coverage["dsa"][tf] = {
            "dsa_alignment_nan_rate": round(
                nan_rate(non_skip[f"dsa_alignment_{tf}"]), 6
            ),
            "dsa_vwap_dev_rel_nan_rate": round(
                nan_rate(non_skip[f"dsa_vwap_dev_rel_{tf}"]), 6
            ),
        }
        coverage["momentum"][tf] = {
            "momentum_direction_nan_rate": round(
                nan_rate(
                    non_skip[f"momentum_direction_{tf}"]
                ),
                6,
            ),
            "momentum_value_rel_nan_rate": round(
                nan_rate(
                    non_skip[f"momentum_value_rel_{tf}"]
                ),
                6,
            ),
            "momentum_delta_rel_nan_rate": round(
                nan_rate(
                    non_skip[f"momentum_delta_rel_{tf}"]
                ),
                6,
            ),
        }
    qs = state["quant_state"].value_counts()
    n = len(state)
    known = int(n - int(qs.get("UNKNOWN", 0)))
    coverage["quantile"] = {
        "counts": {str(k): int(v) for k, v in qs.items()},
        "known": known,
        "unknown": int(qs.get("UNKNOWN", 0)),
        "known_coverage_pct": round(known / n * 100.0, 2),
        "note": "UNKNOWN preserved; not imputed, not re-ranked",
    }

    # ---------------- reward integrity ----------------
    reward = {"coverage": {}, "exit_codes": {}, "skip": {}}
    for h in DIAGNOSTIC_HORIZONS:
        v = non_skip[f"gross_R_h{h}"].to_numpy(float)
        finite = int(np.isfinite(v).sum())
        reward["coverage"][f"H{h}"] = {
            "non_skip_rows": int(len(v)),
            "finite": finite,
            "finite_pct": round(finite / len(v) * 100.0, 2),
        }
        vc = non_skip[f"exit_code_h{h}"].value_counts()
        reward["exit_codes"][f"H{h}"] = {
            EXIT_NAMES.get(int(k), str(int(k))): int(val)
            for k, val in sorted(vc.items())
        }
    reward["skip"] = {
        "rows": int(len(skip)),
        "gross_R_all_zero": bool(
            (skip["gross_R_h12"].to_numpy(float) == 0.0).all()
        ),
        "exit_code_all_minus_one": bool(
            (skip["exit_code_h12"].to_numpy(int) == -1).all()
        ),
    }

    # ---------------- model view ----------------
    model_view = {
        "model_view_version": MODEL_VIEW_VERSION,
        "model_feature_count": len(MODEL_FEATURES_V0),
        "categorical": list(CATEGORICAL_FEATURES_V0),
        "continuous": list(CONTINUOUS_FEATURES_V0),
        "meta": list(MODEL_META_V0),
        "weight": list(MODEL_WEIGHT_V0),
        "action": ["action", "trade_mode", "trade_direction",
                   "target_R", "stop_atr", "target_atr"],
        "reward": [
            "gross_R_h6",
            "gross_R_h12",
            "gross_R_h24",
            "exit_code_h6",
            "exit_code_h12",
            "exit_code_h24",
            "primary_reward_R",
            "reward_version",
        ],
        "categorical_missing_contract": CATEGORICAL_MISSING_V0,
        "warehouse_only_families": list(
            WAREHOUSE_ONLY_FAMILIES
        ),
        "audit": mv,
    }

    audit = {
        "dataset": dataset,
        "storage": storage,
        "coverage": coverage,
        "reward_integrity": reward,
        "model_view": model_view,
    }
    guard_no_ranking(audit)

    (OUT_ROOT / "dataset_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ---------------- columns ----------------
    all_cols = _full_columns("ob_rl_action_v0")
    columns_doc = {
        "role_map": MODEL_VIEW_ROLE_MAP,
        "columns": [
            {
                "name": c,
                "role": MODEL_VIEW_ROLE_MAP.get(
                    c,
                    "WAREHOUSE_ONLY"
                    if c not in MODEL_FEATURES_V0
                    else "MODEL_FEATURE",
                ),
                "in_model_view": c in MODEL_FEATURES_V0,
            }
            for c in all_cols
        ],
    }
    (OUT_ROOT / "dataset_columns.json").write_text(
        json.dumps(columns_doc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ---------------- README ----------------
    readme = f"""# OB RL Dataset V0

Offline `State x Action x Reward` dataset for OB touch research.

- State rows: {dataset['state_rows']:,} (one per OB touch)
- Action rows: {dataset['action_rows']:,} (candidates x 7 actions)
- Reward version: GROSS_R_V0 (no cost model yet)
- Primary horizon: H12 (H6 / H24 diagnostics)

## Storage

Parquet (pyarrow / zstd) is the local authoritative format. CSV is a
one-off build artifact. Neither the CSV nor the Parquet is committed;
the dataset is reproducible from the frozen V3 source data.

## Model View V0

{model_view['model_feature_count']} whitelisted fields:
Event (6) + SMC (36) + DSA (6) + Momentum (9) + Quantile (2) +
Action (3).

Continuous values stay continuous; binning is an audit-layer concern
only.

META and WEIGHT columns are never model features.

## Missing-value contract

- Categorical active-OB fields may be filled with `NO_OB`.
- Numeric OB distance / fit stays NaN. `NO_OB` is NOT `distance = 0`.
- Quantile `UNKNOWN` is preserved, never imputed or re-ranked.

## Not in V0

{chr(10).join('- ' + f for f in WAREHOUSE_ONLY_FAMILIES)}

Excluded from V0 does not mean rejected; it means deferred for later
ablation.
"""
    (OUT_ROOT / "README.md").write_text(
        readme, encoding="utf-8"
    )

    print("DATASET_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
