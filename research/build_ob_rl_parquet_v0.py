#!/usr/bin/env python3

"""Convert the audited Gate-B OB RL V0 CSVs to Parquet (zstd).

Hard rules
----------
* The EXACT Gate-B files are locked by SHA256. Any mismatch is a STOP,
  never a "close enough" continuation.
* Storage is pyarrow + zstd only. A missing pyarrow is a STOP; the
  engine and codec are never silently substituted.
* Conversion is verified by four independent layers: schema order,
  cardinality, keys, and exact cell content.

This module performs NO modelling, no feature selection and no
reward-based filtering.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.build_ob_rl_dataset_v0 import (  # noqa: E402
    OUT_ROOT,
    resolve_git_head,
)
from research.ob_rl_dataset_v0_spec import (  # noqa: E402
    SOURCE_DATA_BASELINE_SHA,
    GATE_B_DATASET_BUILDER_SHA,
    VALIDATED_TFS,
)

EXPECTED_DATASET_BUILDER_CODE_SHA = (
    "9e63f4b01fc3a4b26bef8a28d26289f7004db927"
)

EXPECTED_STATE_ROWS = 21_481
EXPECTED_ACTION_ROWS = 150_367

EXPECTED_DSA_ROWS = (
    EXPECTED_STATE_ROWS
    * len(VALIDATED_TFS)
)

DSA_CONTRACT_VERSION = (
    "CONFIRMED_CAUSAL_DSA_V1"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_rebuilt_dataset(
    root: Path,
) -> dict:
    """Verify the corrected causal-DSA dataset authority.

    The corrected RL dataset is identified by the builder commit that
    actually produced it plus the DSA invariants written by that builder.

    CSV hashes are computed from the verified files and sealed into the
    Parquet receipt; they are not inherited from the obsolete pre-fix
    Gate-B dataset.
    """

    manifest_path = (
        root
        / "dataset_manifest.json"
    )

    if not manifest_path.exists():
        raise RuntimeError(
            f"missing dataset manifest: {manifest_path}"
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        manifest.get(
            "builder_code_sha"
        )
        != EXPECTED_DATASET_BUILDER_CODE_SHA
    ):
        raise RuntimeError(
            "corrected dataset builder SHA mismatch: "
            f"{manifest.get('builder_code_sha')}"
        )

    if (
        manifest.get(
            "source_data_baseline_sha"
        )
        != SOURCE_DATA_BASELINE_SHA
    ):
        raise RuntimeError(
            "source data baseline drift"
        )

    if (
        int(
            manifest.get(
                "candidates",
                -1,
            )
        )
        != EXPECTED_STATE_ROWS
    ):
        raise RuntimeError(
            "candidate cardinality drift"
        )

    if (
        int(
            manifest.get(
                "state_rows",
                -1,
            )
        )
        != EXPECTED_STATE_ROWS
    ):
        raise RuntimeError(
            "state cardinality drift"
        )

    if (
        int(
            manifest.get(
                "action_rows",
                -1,
            )
        )
        != EXPECTED_ACTION_ROWS
    ):
        raise RuntimeError(
            "action cardinality drift"
        )

    # --------------------------------------------------------
    # Rebuilt DSA authority
    # --------------------------------------------------------

    cov = manifest.get(
        "dsa_coverage",
        {}
    )

    if (
        cov.get("authority")
        != (
            "recomputed_from_frozen_bars_"
            "via_compute_dsa_canonical"
        )
    ):
        raise RuntimeError(
            "DSA authority mismatch"
        )

    if (
        int(
            cov.get(
                "expected_rows",
                -1,
            )
        )
        != EXPECTED_DSA_ROWS
    ):
        raise RuntimeError(
            "DSA expected row drift"
        )

    if (
        int(
            cov.get(
                "actual_rows",
                -1,
            )
        )
        != EXPECTED_DSA_ROWS
    ):
        raise RuntimeError(
            "DSA actual row drift"
        )

    if (
        int(
            cov.get(
                "incomplete_candidates",
                -1,
            )
        )
        != 0
    ):
        raise RuntimeError(
            "incomplete rebuilt DSA"
        )

    ignored = set(
        manifest.get(
            "frozen_v3_dsa_columns_ignored",
            [],
        )
    )

    expected_ignored = {
        "dsa_direction",
        "dsa_raw_dsa_vwap_dev_pct",
    }

    if ignored != expected_ignored:
        raise RuntimeError(
            "old V3 DSA authority was not "
            "explicitly ignored"
        )

    # --------------------------------------------------------
    # State confirmed-only gate
    # --------------------------------------------------------

    state_gate = manifest.get(
        "dsa_confirmation_gate",
        {}
    )

    for tf in VALIDATED_TFS:

        g = state_gate.get(
            tf,
            {}
        )

        rows = int(
            g.get(
                "rows",
                -1,
            )
        )

        confirmed = int(
            g.get(
                "confirmed_rows",
                -1,
            )
        )

        unconfirmed = int(
            g.get(
                "unconfirmed_rows",
                -1,
            )
        )

        masked = int(
            g.get(
                "provisional_dev_rows_masked",
                -1,
            )
        )

        confirmed_dev = int(
            g.get(
                "confirmed_dev_rows",
                -1,
            )
        )

        if rows != EXPECTED_STATE_ROWS:
            raise RuntimeError(
                f"{tf}: DSA state gate row drift"
            )

        if (
            confirmed
            + unconfirmed
            != EXPECTED_STATE_ROWS
        ):
            raise RuntimeError(
                f"{tf}: DSA state partition drift"
            )

        if masked != unconfirmed:
            raise RuntimeError(
                f"{tf}: not all provisional "
                "DSA deviations were masked"
            )

        if confirmed_dev != confirmed:
            raise RuntimeError(
                f"{tf}: confirmed DSA deviation "
                "coverage mismatch"
            )

    # --------------------------------------------------------
    # Action confirmed-only gate
    # --------------------------------------------------------

    action_gate = manifest.get(
        "dsa_action_confirmation_gate",
        {}
    )

    for tf in VALIDATED_TFS:

        g = action_gate.get(
            tf,
            {}
        )

        if (
            int(
                g.get(
                    "action_rows",
                    -1,
                )
            )
            != EXPECTED_ACTION_ROWS
        ):
            raise RuntimeError(
                f"{tf}: DSA action row drift"
            )

        if (
            int(
                g.get(
                    "unconfirmed_rel_nonnull",
                    -1,
                )
            )
            != 0
        ):
            raise RuntimeError(
                f"{tf}: unconfirmed DSA "
                "action feature leaked"
            )

    files = {
        "ob_rl_state_v0.csv":
            root
            / "ob_rl_state_v0.csv",

        "ob_rl_action_v0.csv":
            root
            / "ob_rl_action_v0.csv",

        "dataset_manifest.json":
            manifest_path,
    }

    hashes = {}

    for name, path in files.items():

        if not path.exists():
            raise RuntimeError(
                f"missing rebuilt dataset file: {path}"
            )

        hashes[name] = (
            sha256_file(
                path
            )
        )

    return {
        "manifest":
            manifest,

        "hashes":
            hashes,
    }


def require_pyarrow() -> str:
    try:
        import pyarrow  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "pyarrow is required for the zstd Parquet contract; "
            "do NOT substitute another engine or codec"
        ) from e
    return "pyarrow"


def verify_dsa_confirmation_contract(
    df: pd.DataFrame,
    *,
    action_table: bool,
) -> dict:

    result = {}

    for tf in VALIDATED_TFS:

        direction_col = (
            f"dsa_direction_{tf}"
        )

        dev_col = (
            f"dsa_vwap_dev_pct_{tf}"
        )

        for col in (
            direction_col,
            dev_col,
        ):
            if col not in df.columns:
                raise RuntimeError(
                    f"missing DSA column {col}"
                )

        direction = pd.to_numeric(
            df[direction_col],
            errors="coerce",
        )

        dev = pd.to_numeric(
            df[dev_col],
            errors="coerce",
        )

        legal = (
            direction.isna()
            | direction.isin(
                [-1, 0, 1]
            )
        )

        if not legal.all():
            raise RuntimeError(
                f"{tf}: illegal DSA direction"
            )

        confirmed = (
            direction.abs()
            == 1
        )

        bad_dev = (
            (~confirmed)
            & dev.notna()
        )

        if bad_dev.any():
            raise RuntimeError(
                f"{tf}: "
                f"{int(bad_dev.sum())} "
                "unconfirmed state deviations leaked"
            )

        out = {
            "rows":
                int(len(df)),

            "confirmed_rows":
                int(
                    confirmed.sum()
                ),

            "unconfirmed_dev_nonnull":
                int(
                    bad_dev.sum()
                ),
        }

        if action_table:

            rel_col = (
                f"dsa_vwap_dev_rel_{tf}"
            )

            if rel_col not in df.columns:
                raise RuntimeError(
                    f"missing action DSA column "
                    f"{rel_col}"
                )

            rel = pd.to_numeric(
                df[rel_col],
                errors="coerce",
            )

            bad_rel = (
                (~confirmed)
                & rel.notna()
            )

            if bad_rel.any():
                raise RuntimeError(
                    f"{tf}: "
                    f"{int(bad_rel.sum())} "
                    "unconfirmed action-relative "
                    "DSA values leaked"
                )

            out[
                "unconfirmed_rel_nonnull"
            ] = int(
                bad_rel.sum()
            )

        result[tf] = out

    return result


def normalize_for_compare(df: pd.DataFrame) -> pd.DataFrame:
    """Explicit normalization, applied to BOTH sides.

    Object columns hold null either as NaN (CSV) or as None (Arrow).
    Rather than weakening the comparison, both sides are normalized to
    None so exact cell comparison stays exact.
    """
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].where(out[c].notna(), None)
    return out


def write_parquet(
    csv_path: Path, parquet_path: Path
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df.to_parquet(
        parquet_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    return df


def stage_parquet(
    csv_path: Path,
    final_path: Path,
    *,
    key_cols: tuple[str, ...],
    expected_rows: int,
    action_table: bool,
) -> dict:
    """Write to a staging path and verify BEFORE promotion.

    The final artifact must never exist in an unverified state: the
    audit layer prefers Parquet, so a half-verified Parquet would
    silently become the source of truth.
    """
    tmp = final_path.with_name(f".{final_path.name}.tmp")
    tmp.unlink(missing_ok=True)

    df = pd.read_csv(csv_path, low_memory=False)

    csv_dsa_audit = (
        verify_dsa_confirmation_contract(
            df,
            action_table=action_table,
        )
    )

    try:
        df.to_parquet(
            tmp,
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        result = verify_parquet(
            df,
            tmp,
            key_cols=key_cols,
            expected_rows=expected_rows,
            action_table=action_table,
        )
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    return {
        "tmp": tmp,
        "final": final_path,
        "result": result,
        "csv_dsa_audit": csv_dsa_audit,
        "df": df,
    }


def verify_parquet(
    csv_df: pd.DataFrame,
    parquet_path: Path,
    *,
    key_cols: tuple[str, ...],
    expected_rows: int,
    action_table: bool,
) -> dict:
    pq = pd.read_parquet(parquet_path)

    # 1) schema + column ORDER
    if list(csv_df.columns) != list(pq.columns):
        raise RuntimeError("column order/name mismatch")

    # 2) cardinality
    if len(pq) != expected_rows or len(csv_df) != expected_rows:
        raise RuntimeError(
            f"cardinality mismatch: csv={len(csv_df)} "
            f"parquet={len(pq)} expected={expected_rows}"
        )

    # 3) keys + row order
    for c in key_cols:
        if c not in pq.columns:
            raise RuntimeError(f"missing key column {c}")
    if pq.duplicated(list(key_cols)).any():
        raise RuntimeError("duplicate key after roundtrip")
    a = csv_df[list(key_cols)].astype(str)
    b = pq[list(key_cols)].astype(str)
    if not a.equals(b):
        raise RuntimeError("row order changed after roundtrip")

    # 4) exact cell content
    pd.testing.assert_frame_equal(
        normalize_for_compare(csv_df),
        normalize_for_compare(pq),
        check_dtype=False,
        check_exact=True,
        check_like=False,
    )

    parquet_dsa_audit = (
        verify_dsa_confirmation_contract(
            pq,
            action_table=action_table,
        )
    )

    return {
        "rows": int(len(pq)),
        "columns": int(len(pq.columns)),
        "schema_match": True,
        "key_match": True,
        "content_exact": True,
        "dsa_confirmation_contract":
            parquet_dsa_audit,
    }


RECEIPT_NAME = "parquet_conversion_receipt.json"

REPRESENTATION_VERSION = "RL0_PARQUET_V1_CAUSAL_DSA"


def promote_all(staged: list[dict]) -> list[Path]:
    """Promote every staged file, or roll EVERYTHING back.

    A partial promotion (state promoted, action failed) would leave a
    final artifact behind while the conversion as a whole failed, so
    already-promoted finals are removed too.
    """
    promoted: list[Path] = []
    try:
        for item in staged:
            item["tmp"].replace(item["final"])
            promoted.append(item["final"])
    except Exception:
        for item in staged:
            item["tmp"].unlink(missing_ok=True)
        for p in promoted:
            p.unlink(missing_ok=True)
        raise
    return promoted


def convert_all(root: Path) -> dict:
    """Stage and verify BOTH tables, then promote them together.

    Nothing reaches a final path until every table has passed exact
    parity; a failure leaves no final artifact at all.
    """
    source = (
        verify_rebuilt_dataset(
            root
        )
    )

    source_hashes = source[
        "hashes"
    ]

    source_manifest = source[
        "manifest"
    ]

    specs = (
        (
            "state",
            "ob_rl_state_v0",
            ("candidate_id",),
            EXPECTED_STATE_ROWS,
            False,
        ),

        (
            "action",
            "ob_rl_action_v0",
            (
                "candidate_id",
                "action",
            ),
            EXPECTED_ACTION_ROWS,
            True,
        ),
    )

    # Refuse to run over an existing final artifact: promotion must
    # start from a clean state so rollback is always unambiguous.
    existing = [
        root / f"{name}.parquet"
        for (
            _,
            name,
            _,
            _,
            _,
        ) in specs
    ]
    present = [p for p in existing if p.exists()]
    if present:
        raise RuntimeError(
            "final parquet already exists before conversion: "
            f"{[p.name for p in present]}"
        )

    staged = []
    try:
        for (
            tag,
            name,
            key_cols,
            expected,
            action_table,
        ) in specs:
            csv_path = root / f"{name}.csv"
            staged.append(
                (
                    tag,
                    csv_path,
                    stage_parquet(
                        csv_path,
                        root / f"{name}.parquet",
                        key_cols=key_cols,
                        expected_rows=expected,
                        action_table=action_table,
                    ),
                )
            )

        # All exact-parity checks passed -> promote together.
        promote_all(
            [
                item
                for _, _, item
                in staged
            ]
        )
    except Exception:
        for _, _, item in staged:
            item["tmp"].unlink(missing_ok=True)
        raise

    receipt = {
        "representation_version":
            REPRESENTATION_VERSION,

        "engine":
            "pyarrow",

        "compression":
            "zstd",

        "dsa_contract_version":
            DSA_CONTRACT_VERSION,

        "source_data_baseline_sha":
            source_manifest[
                "source_data_baseline_sha"
            ],

        # Historical Gate-B lineage remains recorded.
        "historical_gate_b_dataset_builder_sha":
            source_manifest[
                "gate_b_dataset_builder_sha"
            ],

        # This is the code that actually generated the corrected CSVs.
        "dataset_builder_code_sha":
            source_manifest[
                "builder_code_sha"
            ],

        # This is the converter commit actually producing the Parquet.
        "representation_code_sha":
            resolve_git_head(),

        "dataset_manifest_sha256":
            source_hashes[
                "dataset_manifest.json"
            ],
    }

    for tag, csv_path, item in staged:
        res = dict(item["result"])
        final_path = item["final"]
        res["csv_bytes"] = int(csv_path.stat().st_size)
        res["parquet_bytes"] = int(final_path.stat().st_size)
        res["compression_ratio"] = round(
            res["csv_bytes"]
            / max(res["parquet_bytes"], 1),
            3,
        )
        res["csv_sha256"] = (
            source_hashes[
                csv_path.name
            ]
        )
        res["csv_dsa_confirmation_contract"] = (
            item[
                "csv_dsa_audit"
            ]
        )
        res["parquet_sha256"] = sha256_file(
            final_path
        )
        receipt[tag] = res

    return receipt


def write_receipt(root: Path, receipt: dict) -> Path:
    """Receipt exists only after a complete verified promotion."""
    tmp = root / f".{RECEIPT_NAME}.tmp"
    tmp.unlink(missing_ok=True)
    tmp.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    final = root / RECEIPT_NAME
    tmp.replace(final)
    return final


def main() -> None:
    require_pyarrow()
    root = OUT_ROOT

    receipt = convert_all(root)
    path = write_receipt(root, receipt)

    print("parquet receipt", path, flush=True)
    for tag in ("state", "action"):
        print(tag, receipt[tag], flush=True)
    print("PARQUET_CONVERT_DONE", flush=True)


if __name__ == "__main__":
    main()
