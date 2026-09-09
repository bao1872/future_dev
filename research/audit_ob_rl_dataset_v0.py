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
from research import build_ob_rl_parquet_v0 as PQ  # noqa: E402
from research.build_ob_rl_parquet_v0 import (  # noqa: E402
    sha256_file,
)
from research.build_ob_rl_dataset_v0 import (  # noqa: E402
    resolve_git_head,
)
from research.ob_rl_dataset_v0_spec import (  # noqa: E402
    VALIDATED_TFS,
    DIAGNOSTIC_HORIZONS,
    SOURCE_DATA_BASELINE_SHA,
    GATE_B_DATASET_BUILDER_SHA,
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


def verify_manifest_link(
    root: Path,
    receipt: dict,
) -> dict:
    """Bind the Parquet receipt to the corrected dataset manifest.

    Before finalization:
        current manifest SHA must equal the receipt source SHA.

    After finalization:
        the manifest has been enriched by this audit, so the original
        pre-finalization source SHA is preserved explicitly in
        parquet_source_manifest_sha256.
    """

    source = (
        PQ.verify_rebuilt_dataset(
            root
        )
    )

    manifest = source[
        "manifest"
    ]

    p = (
        root
        / "dataset_manifest.json"
    )

    current_sha = sha256_file(
        p
    )

    already_finalized = (
        manifest.get(
            "rl0_finalized"
        )
        is True
    )

    if already_finalized:

        source_sha = manifest.get(
            "parquet_source_manifest_sha256"
        )

        if (
            not isinstance(
                source_sha,
                str,
            )
            or len(source_sha) != 64
        ):
            raise RuntimeError(
                "finalized manifest missing "
                "parquet source manifest SHA"
            )

    else:

        source_sha = current_sha

    receipt_sha = receipt.get(
        "dataset_manifest_sha256"
    )

    if receipt_sha != source_sha:
        raise RuntimeError(
            "Parquet receipt/source manifest "
            "SHA mismatch: "
            f"receipt={receipt_sha} "
            f"source={source_sha}"
        )

    return {
        "manifest":
            manifest,

        "current_manifest_sha256":
            current_sha,

        "source_manifest_sha256":
            source_sha,

        "already_finalized":
            already_finalized,
    }


def read_receipt(root: Path) -> dict:
    p = root / PQ.RECEIPT_NAME
    if not p.exists():
        raise RuntimeError(
            "parquet conversion receipt missing: exact parity "
            "cannot be claimed"
        )
    return json.loads(p.read_text(encoding="utf-8"))


def dsa_contract_columns(
    *,
    action_table: bool,
) -> list[str]:

    cols = []

    for tf in VALIDATED_TFS:

        cols.extend(
            [
                f"dsa_direction_{tf}",
                f"dsa_vwap_dev_pct_{tf}",
            ]
        )

        if action_table:
            cols.append(
                f"dsa_vwap_dev_rel_{tf}"
            )

    return cols


def verify_receipt(
    root: Path,
    receipt: dict,
    manifest_state: dict | None = None,
) -> dict:
    """Re-verify receipt provenance and final Parquet artifacts."""

    if manifest_state is None:
        manifest_state = (
            verify_manifest_link(
                root,
                receipt,
            )
        )

    source_manifest = (
        manifest_state[
            "manifest"
        ]
    )

    if (
        receipt.get(
            "representation_version"
        )
        != PQ.REPRESENTATION_VERSION
    ):
        raise RuntimeError(
            "representation version drift"
        )

    if (
        receipt.get(
            "dsa_contract_version"
        )
        != PQ.DSA_CONTRACT_VERSION
    ):
        raise RuntimeError(
            "DSA contract version drift"
        )

    if receipt.get("engine") != "pyarrow":
        raise RuntimeError(
            "unexpected parquet engine"
        )

    if (
        receipt.get(
            "compression"
        )
        != "zstd"
    ):
        raise RuntimeError(
            "unexpected parquet compression"
        )

    if (
        receipt.get(
            "source_data_baseline_sha"
        )
        != SOURCE_DATA_BASELINE_SHA
    ):
        raise RuntimeError(
            "source baseline drift"
        )

    if (
        receipt.get(
            "historical_gate_b_dataset_builder_sha"
        )
        != GATE_B_DATASET_BUILDER_SHA
    ):
        raise RuntimeError(
            "historical Gate-B lineage drift"
        )

    if (
        receipt.get(
            "dataset_builder_code_sha"
        )
        != PQ.EXPECTED_DATASET_BUILDER_CODE_SHA
    ):
        raise RuntimeError(
            "corrected dataset builder "
            "provenance drift"
        )

    if (
        receipt.get(
            "dataset_manifest_sha256"
        )
        != manifest_state[
            "source_manifest_sha256"
        ]
    ):
        raise RuntimeError(
            "receipt/source manifest link drift"
        )

    artifact_sha = source_manifest.get(
        "artifact_sha256",
        {},
    )

    specs = (
        (
            "state",
            "ob_rl_state_v0",
            PQ.EXPECTED_STATE_ROWS,
            False,
        ),

        (
            "action",
            "ob_rl_action_v0",
            PQ.EXPECTED_ACTION_ROWS,
            True,
        ),
    )

    result = {}

    for (
        tag,
        name,
        expected_rows,
        action_table,
    ) in specs:

        r = receipt.get(
            tag
        )

        if not isinstance(
            r,
            dict,
        ):
            raise RuntimeError(
                f"missing receipt section {tag}"
            )

        for flag in (
            "schema_match",
            "key_match",
            "content_exact",
        ):
            if r.get(flag) is not True:
                raise RuntimeError(
                    f"{tag} parity flag "
                    f"{flag} != true"
                )

        csv_path = (
            root
            / f"{name}.csv"
        )

        pq_path = (
            root
            / f"{name}.parquet"
        )

        if not csv_path.exists():
            raise RuntimeError(
                f"missing CSV {csv_path}"
            )

        if not pq_path.exists():
            raise RuntimeError(
                f"missing parquet {pq_path}"
            )

        expected_csv_sha = (
            artifact_sha.get(
                f"{name}.csv"
            )
        )

        actual_csv_sha = (
            sha256_file(
                csv_path
            )
        )

        if (
            not expected_csv_sha
            or r.get(
                "csv_sha256"
            )
            != expected_csv_sha
            or actual_csv_sha
            != expected_csv_sha
        ):
            raise RuntimeError(
                f"{tag} corrected CSV "
                "provenance drift"
            )

        actual_pq_sha = (
            sha256_file(
                pq_path
            )
        )

        if (
            r.get(
                "parquet_sha256"
            )
            != actual_pq_sha
        ):
            raise RuntimeError(
                f"{tag} parquet hash "
                "does not match receipt"
            )

        if (
            r.get("rows")
            != expected_rows
        ):
            raise RuntimeError(
                f"{tag} receipt row drift"
            )

        receipt_csv_dsa = r.get(
            "csv_dsa_confirmation_contract"
        )

        receipt_pq_dsa = r.get(
            "dsa_confirmation_contract"
        )

        if (
            receipt_csv_dsa
            != receipt_pq_dsa
        ):
            raise RuntimeError(
                f"{tag} CSV/Parquet "
                "DSA receipt drift"
            )

        pq_dsa = pd.read_parquet(
            pq_path,
            columns=dsa_contract_columns(
                action_table=action_table,
            ),
        )

        actual_dsa = (
            PQ.verify_dsa_confirmation_contract(
                pq_dsa,
                action_table=action_table,
            )
        )

        if actual_dsa != receipt_pq_dsa:
            raise RuntimeError(
                f"{tag} DSA confirmation "
                "contract drift"
            )

        result[tag] = {
            "rows":
                int(r["rows"]),

            "columns":
                int(r["columns"]),

            "csv_sha256":
                actual_csv_sha,

            "parquet_sha256":
                actual_pq_sha,

            "schema_match":
                True,

            "key_match":
                True,

            "content_exact":
                True,

            "dsa_confirmation_contract":
                actual_dsa,
        }

    return result


def write_final_manifest(
    root: Path,
    *,
    receipt: dict,
    verified_receipt: dict,
    extra: dict,
) -> Path:

    rep_sha = receipt.get(
        "representation_code_sha"
    )

    audit_sha = resolve_git_head()

    if rep_sha != audit_sha:
        raise RuntimeError(
            "representation/audit HEAD drift: "
            f"parquet built by {rep_sha}, "
            f"audit by {audit_sha}"
        )

    p = (
        root
        / "dataset_manifest.json"
    )

    data = json.loads(
        p.read_text(
            encoding="utf-8"
        )
    )

    # Keep all corrected dataset authority written by the builder.
    # Add representation/finalization provenance only.
    data.update(
        {
            "parquet_source_manifest_sha256":
                receipt[
                    "dataset_manifest_sha256"
                ],

            "representation_version":
                receipt[
                    "representation_version"
                ],

            "dsa_contract_version":
                receipt[
                    "dsa_contract_version"
                ],

            "dataset_builder_code_sha":
                receipt[
                    "dataset_builder_code_sha"
                ],

            "representation_code_sha":
                rep_sha,

            "audit_code_sha":
                audit_sha,

            "model_view_version":
                MODEL_VIEW_VERSION,

            "parquet": {
                "state":
                    verified_receipt[
                        "state"
                    ],

                "action":
                    verified_receipt[
                        "action"
                    ],
            },

            "rl0_finalized":
                True,
        }
    )

    data.update(
        extra
    )

    tmp = (
        root
        / ".dataset_manifest.json.tmp"
    )

    tmp.unlink(
        missing_ok=True
    )

    tmp.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    tmp.replace(
        p
    )

    return p


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
    receipt = read_receipt(
        OUT_ROOT
    )

    manifest_state = (
        verify_manifest_link(
            OUT_ROOT,
            receipt,
        )
    )

    verified_receipt = (
        verify_receipt(
            OUT_ROOT,
            receipt,
            manifest_state,
        )
    )

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
    source_manifest = (
        manifest_state[
            "manifest"
        ]
    )

    storage = {
        "source_dataset": {
            "builder_code_sha":
                source_manifest[
                    "builder_code_sha"
                ],

            "artifact_sha256":
                source_manifest[
                    "artifact_sha256"
                ],

            "source_manifest_sha256":
                manifest_state[
                    "source_manifest_sha256"
                ],

            "current_manifest_sha256":
                manifest_state[
                    "current_manifest_sha256"
                ],

            "already_finalized":
                manifest_state[
                    "already_finalized"
                ],
        },

        "parquet":
            {},
    }

    for name in (
        "ob_rl_state_v0",
        "ob_rl_action_v0",
    ):

        pq_path = (
            OUT_ROOT
            / f"{name}.parquet"
        )

        if pq_path.exists():

            storage[
                "parquet"
            ][name] = {
                "bytes":
                    int(
                        pq_path.stat().st_size
                    ),

                "sha256":
                    sha256_file(
                        pq_path
                    ),
            }

    storage[
        "parity"
    ] = {
        "representation_version":
            receipt.get(
                "representation_version"
            ),

        "dsa_contract_version":
            receipt.get(
                "dsa_contract_version"
            ),

        "engine":
            receipt.get(
                "engine"
            ),

        "compression":
            receipt.get(
                "compression"
            ),

        "dataset_builder_code_sha":
            receipt.get(
                "dataset_builder_code_sha"
            ),

        "representation_code_sha":
            receipt.get(
                "representation_code_sha"
            ),

        **{
            tag: {
                k:
                    verified_receipt[
                        tag
                    ].get(k)

                for k in (
                    "schema_match",
                    "key_match",
                    "content_exact",
                    "csv_sha256",
                    "parquet_sha256",
                    "dsa_confirmation_contract",
                )
            }

            for tag in (
                "state",
                "action",
            )
        },
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

    # Only now -- audit PASSed -- is the committed manifest finalized.
    write_final_manifest(
        OUT_ROOT,
        receipt=receipt,
        verified_receipt=verified_receipt,
        extra={
            "model_view_feature_count": len(MODEL_FEATURES_V0),
        },
    )

    print("DATASET_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
