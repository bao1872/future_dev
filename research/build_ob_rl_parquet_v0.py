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
    GATE_B_DATASET_BUILDER_SHA,
)

# Gate-B outputs already audited and reported. Frozen here so the
# converter cannot silently pick up a different/rebuilt CSV.
EXPECTED_GATE_B_FILES = {
    "ob_rl_state_v0.csv": (
        "4cb1c775eebc595e8c9f72787e30a482"
        "070584a7fd8c0aaafa8953974313ab35"
    ),
    "ob_rl_action_v0.csv": (
        "df8ed3c84db582c88ba49ae90a164c9d"
        "630ff274a1823b8c559b4ca12ca99422"
    ),
    "dataset_manifest.json": (
        "fbd139ce644ff48628ac66e6964c9b55"
        "e5a98c884305b0cb93078acc1b9de967"
    ),
}

EXPECTED_STATE_ROWS = 21_481
EXPECTED_ACTION_ROWS = 150_367


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_gate_b_files(root: Path) -> dict:
    got = {}
    for name, expected in EXPECTED_GATE_B_FILES.items():
        p = root / name
        if not p.exists():
            raise RuntimeError(f"missing Gate-B file: {p}")
        digest = sha256_file(p)
        got[name] = digest
        if digest != expected:
            raise RuntimeError(
                f"Gate-B file hash mismatch for {name}: "
                f"expected {expected} got {digest}"
            )
    return got


def require_pyarrow() -> str:
    try:
        import pyarrow  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "pyarrow is required for the zstd Parquet contract; "
            "do NOT substitute another engine or codec"
        ) from e
    return "pyarrow"


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
) -> dict:
    """Write to a staging path and verify BEFORE promotion.

    The final artifact must never exist in an unverified state: the
    audit layer prefers Parquet, so a half-verified Parquet would
    silently become the source of truth.
    """
    tmp = final_path.with_name(f".{final_path.name}.tmp")
    tmp.unlink(missing_ok=True)

    df = pd.read_csv(csv_path, low_memory=False)

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
        )
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    return {
        "tmp": tmp,
        "final": final_path,
        "result": result,
        "df": df,
    }


def verify_parquet(
    csv_df: pd.DataFrame,
    parquet_path: Path,
    *,
    key_cols: tuple[str, ...],
    expected_rows: int,
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

    return {
        "rows": int(len(pq)),
        "columns": int(len(pq.columns)),
        "schema_match": True,
        "key_match": True,
        "content_exact": True,
    }


RECEIPT_NAME = "parquet_conversion_receipt.json"

REPRESENTATION_VERSION = "RL0_PARQUET_V0"


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
    gate_b_hashes = verify_gate_b_files(root)

    specs = (
        (
            "state",
            "ob_rl_state_v0",
            ("candidate_id",),
            EXPECTED_STATE_ROWS,
        ),
        (
            "action",
            "ob_rl_action_v0",
            ("candidate_id", "action"),
            EXPECTED_ACTION_ROWS,
        ),
    )

    # Refuse to run over an existing final artifact: promotion must
    # start from a clean state so rollback is always unambiguous.
    existing = [
        root / f"{name}.parquet" for _, name, _, _ in specs
    ]
    present = [p for p in existing if p.exists()]
    if present:
        raise RuntimeError(
            "final parquet already exists before conversion: "
            f"{[p.name for p in present]}"
        )

    staged = []
    try:
        for tag, name, key_cols, expected in specs:
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
                    ),
                )
            )

        # All exact-parity checks passed -> promote together.
        promote_all([item for _, _, item in staged])
    except Exception:
        for _, _, item in staged:
            item["tmp"].unlink(missing_ok=True)
        raise

    receipt = {
        "representation_version": REPRESENTATION_VERSION,
        "engine": "pyarrow",
        "compression": "zstd",
        "gate_b_dataset_builder_sha": (
            GATE_B_DATASET_BUILDER_SHA
        ),
        "representation_code_sha": resolve_git_head(),
    }

    for tag, csv_path, item in staged:
        res = dict(item["result"])
        final_path = item["final"]
        res["csv_bytes"] = int(csv_path.stat().st_size)
        res["parquet_bytes"] = int(final_path.stat().st_size)
        res["compression_ratio"] = round(
            res["csv_bytes"] / max(res["parquet_bytes"], 1), 3
        )
        res["csv_sha256"] = gate_b_hashes[
            f"{csv_path.name}"
        ]
        res["parquet_sha256"] = sha256_file(final_path)
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
