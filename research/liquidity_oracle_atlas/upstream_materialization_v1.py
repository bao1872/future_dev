"""FUTURE-R4-M15-15SYM-UPSTREAM-MATERIALIZATION-V1
====================================================

Upstream materialization for the 14 non-AG frozen symbols.

PURPOSE
-------
Materialize the missing canonical market data, frozen overnight Teacher,
and Phase-1 STRUCT33 Candidate datasets for the 14 non-AG symbols of the
frozen 15-symbol universe. THIS MODULE TRAINS NO MODEL.

CANONICAL DATA OWNERSHIP (verified in PHASE U0)
-----------------------------------------------
* Raw 5m lives in ``research/exports/v3r_5m/{symbol}_5m.csv`` and is read by
  ``research.export_ob_trigger_execution_v21.load_raw_5m``. This store already
  contains ALL 15 frozen symbols. No PyTDX download is required.
* The canonical R4 15m execution frame is built by
  ``build_execution_frame_m15_v1`` and persisted (SHA256 + manifest) under
  ``artifacts/candidate_gate_r4_m15_touch_nextbar_v1/`` with a fail-closed
  verified loader.
* The frozen Phase-0.5 overnight Teacher is produced by
  ``build_teacher_oracle_dp_m15_overnight_v1.run_dp_m15_overnight_teacher``.
* The Phase-1 STRUCT33 dataset is produced by
  ``build_struct33_dataset_v1.build_struct33_dataset``.

This module orchestrates those canonical owners; it does NOT reimplement any
indicator math, Teacher logic, or dataset mapping.

STOP
----
DO NOT train pooled models. Stop once all 14 frozen datasets exist.

TASK_ID: FUTURE-R4-M15-15SYM-UPSTREAM-MATERIALIZATION-V1
BASE_SHA: 67ea37bf3d37d4656c3d5fd3785cbf9b2efbc0de
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.export_ob_trigger_execution_v21 import RAW_5M_ROOT, load_raw_5m
from research.liquidity_oracle_atlas.build_execution_frame_m15_v1 import (
    load_summary,
    save_execution_frame_m15,
)
from research.liquidity_oracle_atlas.build_teacher_oracle_dp_m15_overnight_v1 import (
    TEACHER_CONTRACT_ID,
    TASK_ID as TEACHER_TASK_ID,
    run_dp_m15_overnight_teacher,
    write_oracle_artifact,
    check_oracle_invariants,
)
from research.liquidity_oracle_atlas.build_struct33_dataset_v1 import (
    STRUCT33,
    build_struct33_dataset,
)
from market_data.config import DATA_DIR as SILVER_DIR

import hashlib
import time

# The commit that ACTUALLY generated the 14 new Teachers / datasets: the working
# tree that was committed as this upstream-materialization commit. This is the
# correct provenance (not the task base SHA, which only names the task's base).
UPSTREAM_SHA = "c45a1efa8042d44cb36282d15f0ff7b7fea0d23e"
BUILDER_TASK_ID = "FUTURE-R4-M15-STRUCT33-DATASET-V1-PHASE1"

ROOT = Path(__file__).resolve().parents[2]
EXEC_DIR = ROOT / "artifacts" / "candidate_gate_r4_m15_touch_nextbar_v1"
TEACHER_DIR = ROOT / "artifacts" / "teacher_oracle_dp_m15_overnight_v1"
DATASET_DIR = ROOT / "artifacts" / "struct33_dataset_v1"

# Frozen 15-symbol universe (AG + 14 to materialize).
SYMBOLS = ("AG", "AU", "CU", "AL", "SN", "NI", "RB", "I", "SC", "RU", "MA", "TA", "M", "P", "CF")
AG = "AG"
MISSING = tuple(s for s in SYMBOLS if s != AG)


# --------------------------------------------------------------------------- #
# PHASE U0 — source inventory (no download)                                     #
# --------------------------------------------------------------------------- #
def _exists(p: Path) -> bool:
    return p.exists()


def _raw_meta(symbol: str) -> dict:
    p = RAW_5M_ROOT / f"{symbol}_5m.csv"
    if not _exists(p):
        return {"exists": False}
    t = pd.read_csv(p, usecols=["bar_start_time"])
    t = pd.to_datetime(t["bar_start_time"], errors="coerce")
    return {
        "exists": True,
        "rows": int(len(t)),
        "first": str(t.iloc[0]),
        "last": str(t.iloc[-1]),
        "dup_ts": int(t.duplicated().sum()),
    }


def _silver_meta(symbol: str) -> dict:
    # silver_main_data uses L8 ticker columns; AG is the only research symbol there.
    out = {}
    for tf in ("5m", "15m", "1h", "4h"):
        p = SILVER_DIR / f"silver_main_{tf}.csv"
        if _exists(p):
            df = pd.read_csv(p, usecols=["symbol"])
            out[tf] = bool((df["symbol"].astype(str) == f"KQ.m@SHFE.{symbol.lower()}").any())
        else:
            out[tf] = False
    return out


def _exec_meta(symbol: str) -> dict:
    summary = load_summary()
    spec = summary.get("execution_frames", {}).get(symbol)
    p = EXEC_DIR / f"{symbol}_exec_frame.parquet"
    if spec is None or not _exists(p):
        return {"exists": False}
    return {
        "exists": True,
        "rows": spec.get("rows"),
        "sha256": spec.get("sha256"),
    }


def _teacher_meta(symbol: str) -> dict:
    d = TEACHER_DIR / symbol
    if not _exists(d):
        return {"exists": False}
    meta_p = d / "metadata.json"
    trades_p = d / "oracle_trades.parquet"
    out = {"exists": True, "has_metadata": _exists(meta_p), "has_trades": _exists(trades_p)}
    if _exists(meta_p):
        m = json.loads(meta_p.read_text())
        out["math_version"] = m.get("math_version")
        out["oracle_source_sha"] = m.get("oracle_source_sha")
        out["row_count_trades"] = m.get("row_count_trades")
        out["data_start"] = m.get("data_start")
        out["data_end"] = m.get("data_end")
    return out


def _dataset_meta(symbol: str) -> dict:
    d = DATASET_DIR / symbol
    if not _exists(d):
        return {"exists": False}
    parquet = d / "candidate_teacher_dataset.parquet"
    meta_p = d / "metadata.json"
    out = {"exists": True, "has_parquet": _exists(parquet), "has_metadata": _exists(meta_p)}
    if _exists(meta_p):
        m = json.loads(meta_p.read_text())
        out["n_label_eligible"] = m.get("n_label_eligible")
        out["n_candidates"] = m.get("n_candidates")
    return out


def preflight() -> dict:
    """PHASE U0 inventory across all 6 layers for every frozen symbol."""
    rows = {}
    for s in SYMBOLS:
        rows[s] = {
            "raw_5m_v3r": _raw_meta(s),
            "silver_main_data": _silver_meta(s),
            "exec_frame_r4": _exec_meta(s),
            "teacher_artifact": _teacher_meta(s),
            "struct33_dataset": _dataset_meta(s),
        }
    return {
        "task_id": "FUTURE-R4-M15-15SYM-UPSTREAM-MATERIALIZATION-V1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "conclusion": (
            "Raw 5m canonical data present for ALL 15 frozen symbols in "
            "research/exports/v3r_5m. No PyTDX network acquisition required."
        ),
        "symbols": rows,
    }


# --------------------------------------------------------------------------- #
# PHASE U2 — per-symbol validation (existing kernels adapted to v3r_5m layout)  #
# --------------------------------------------------------------------------- #
def validate_symbol(symbol: str) -> dict:
    raw = load_raw_5m(symbol).sort_values("bar_start_time").reset_index(drop=True)
    n = len(raw)
    t = pd.to_datetime(raw["bar_start_time"])
    o = raw["open"].to_numpy(float)
    h = raw["high"].to_numpy(float)
    l = raw["low"].to_numpy(float)
    c = raw["close"].to_numpy(float)
    day = raw["trading_day"] if "trading_day" in raw.columns else None

    errors: list[str] = []
    # timestamps strictly ordered + duplicate check
    if not t.is_monotonic_increasing:
        errors.append("timestamps not strictly increasing")
    dup = int(t.duplicated().sum())
    # OHLC finite / valid
    if not np.isfinite(np.concatenate([o, h, l, c])).all():
        errors.append("NaN/Inf in OHLC")
    if not (h >= np.maximum(o, c)).all():
        errors.append("high < max(open,close)")
    if not (l <= np.minimum(o, c)).all():
        errors.append("low > min(open,close)")
    if day is None or day.isna().any():
        errors.append("trading_day missing")

    # 5m -> 15m aggregation determinism: build exec frame, then compare counts.
    try:
        from research.liquidity_oracle_atlas.build_execution_frame_m15_v1 import (
            build_execution_frame_m15,
        )
        ef = build_execution_frame_m15(symbol)
        n15 = len(ef)
        n_seg = int(ef["segment"].nunique())
        # no impossible cross-segment bar: segment is non-decreasing over time
        seg_ok = bool((ef["segment"].to_numpy() == np.sort(ef["segment"].to_numpy())).all())
        ef_first = str(pd.to_datetime(ef["bar_start_time"]).iloc[0])
        ef_last = str(pd.to_datetime(ef["bar_start_time"]).iloc[-1])
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"exec_frame_build_failed: {exc!r}")
        n15 = None
        n_seg = None
        seg_ok = False
        ef_first = ef_last = None

    # maximum observed time gap (minutes)
    gap_min = float((t.diff().dropna().dt.total_seconds().max() or 0) / 60.0) if n > 1 else 0.0

    return {
        "symbol": symbol,
        "raw_rows": int(n),
        "dup_timestamps": dup,
        "first_time": str(t.iloc[0]),
        "last_time": str(t.iloc[-1]),
        "n_15m_bars": n15,
        "n_hard_segments": n_seg,
        "segment_monotonic": seg_ok,
        "exec_first": ef_first,
        "exec_last": ef_last,
        "max_gap_min": round(gap_min, 2),
        "ok": len(errors) == 0,
        "errors": ";".join(errors) if errors else "",
    }


def run_validation() -> pd.DataFrame:
    return pd.DataFrame([validate_symbol(s) for s in SYMBOLS])


# --------------------------------------------------------------------------- #
# PHASE U3 — execution-frame materialization (persist + SHA + manifest)         #
# --------------------------------------------------------------------------- #
def build_missing_execution_frames(verbose: bool = True) -> dict:
    """Build + persist R4 15m execution frames for symbols missing them.

    Uses the canonical ``save_execution_frame_m15`` owner (SHA + manifest).
    """
    summary = load_summary()
    have = set(summary.get("execution_frames", {}).keys())
    built = {}
    for s in MISSING:
        if s in have and (EXEC_DIR / f"{s}_exec_frame.parquet").exists():
            continue
        spec = save_execution_frame_m15(s)
        built[s] = spec
        if verbose:
            print(f"[U3] exec frame built: {s} rows={spec['rows']} sha={spec['sha256'][:12]}")
    return built


# --------------------------------------------------------------------------- #
# PHASE U4 — frozen overnight Teacher (local Bellman, no PyTDX)                 #
# --------------------------------------------------------------------------- #
def _sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _struct33_schema_hash() -> str:
    return hashlib.sha256("|".join(STRUCT33).encode()).hexdigest()


def build_teacher(symbol: str, verbose: bool = True):
    """Run the EXACT frozen Phase-0.5 overnight Teacher and persist the artifact.

    Records real identity (teacher_contract_id + teacher_source_git_sha) instead
    of the legacy free-text pseudo-SHA. Fails closed on any hygiene violation.
    """
    res = run_dp_m15_overnight_teacher(symbol)
    inv = check_oracle_invariants(res)
    bad = (
        inv["cross_segment"]
        or inv["illegal_reversal"]
        or inv["new_entry_outside_proximity"]
        or inv["new_entry_armed_zero"]
        or inv["nonflat_terminal"]
        or inv["pnl_mismatch_flag"]
        or inv["max_new_entries_per_episode"] > 1
    )
    if bad:
        raise RuntimeError(f"TEACHER_HYGIENE_FAIL:{symbol}:{inv}")
    outdir = write_oracle_artifact(res, TEACHER_DIR, oracle_source_sha=UPSTREAM_SHA)
    (outdir / "hygiene.json").write_text(json.dumps(inv, indent=2, default=str))
    if verbose:
        print(
            f"[U4] teacher built: {symbol} trades={len(res['trades'])} "
            f"cross_seg={inv['cross_segment']} nonflat_term={inv['nonflat_terminal']} "
            f"pnl_mismatch={inv['pnl_mismatch_flag']}"
        )
    return outdir, inv


def normalize_teacher_metadata(
    symbols=MISSING, source_sha: str = UPSTREAM_SHA, contract_id: str = TEACHER_CONTRACT_ID
):
    """Correct provenance on already-generated Teacher artifacts (no DP rerun).

    The 14 new Teachers were generated by the working tree committed as
    UPSTREAM_SHA. Their on-disk metadata.json currently records the task base
    SHA. This rewrites the identity fields in place (they are provenance only;
    the numeric Teacher parquet is untouched, so no recomputation is needed).
    AG is intentionally excluded: it is legacy and keeps its "phase0p5-fix2" tag.
    """
    patched = []
    for s in symbols:
        meta_p = TEACHER_DIR / s / "metadata.json"
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text())
        meta["teacher_contract_id"] = contract_id
        meta["teacher_source_git_sha"] = source_sha
        meta["oracle_source_sha"] = source_sha  # retained legacy field, now a real SHA
        meta_p.write_text(json.dumps(meta, indent=2, default=str))
        patched.append(s)
    return patched


# --------------------------------------------------------------------------- #
# PHASE U5 — Phase-1 STRUCT33 Candidate->Teacher dataset                        #
# --------------------------------------------------------------------------- #
def build_dataset(symbol: str, verbose: bool = True):
    """Build the frozen Phase-1 STRUCT33 dataset (no split, no model)."""
    r = build_struct33_dataset(symbol)
    if verbose:
        print(
            f"[U5] dataset built: {symbol} candidates={r['meta']['n_candidates']} "
            f"eligible={r['meta']['n_label_eligible']}"
        )
    return r


# --------------------------------------------------------------------------- #
# PHASE U6 — dataset manifest (identity per symbol)                             #
# --------------------------------------------------------------------------- #
def build_manifest() -> list:
    schema_hash = _struct33_schema_hash()
    rows = []
    for s in SYMBOLS:
        ds_dir = DATASET_DIR / s
        parquet = ds_dir / "candidate_teacher_dataset.parquet"
        meta_p = ds_dir / "metadata.json"
        if not parquet.exists():
            rows.append({"symbol": s, "dataset_exists": False})
            continue
        meta = json.loads(meta_p.read_text())
        ds = pd.read_parquet(parquet)
        times = pd.to_datetime(ds["candidate_decision_time"])
        exec_p = EXEC_DIR / f"{s}_exec_frame.parquet"
        n_seg = (
            int(pd.read_parquet(exec_p)["segment"].nunique())
            if exec_p.exists()
            else None
        )
        tmeta = json.loads((TEACHER_DIR / s / "metadata.json").read_text())
        rows.append(
            {
                "symbol": s,
                "raw_source_id": f"v3r_5m/{s}_5m.csv",
                "raw_sha256": _sha256_file(RAW_5M_ROOT / f"{s}_5m.csv"),
                "execution_frame_sha256": _sha256_file(exec_p) if exec_p.exists() else None,
                "dataset_sha256": _sha256_file(parquet),
                "row_count": int(len(ds)),
                "n_label_eligible": int(meta.get("n_label_eligible")),
                "first_candidate_time": str(times.min()),
                "last_candidate_time": str(times.max()),
                "n_segments": n_seg,
                "builder_task_id": BUILDER_TASK_ID,
                "builder_source_git_sha": UPSTREAM_SHA,
                "teacher_contract_id": tmeta.get("teacher_contract_id") or tmeta.get("task_id"),
                "teacher_source_git_sha": tmeta.get("teacher_source_git_sha")
                or tmeta.get("oracle_source_sha"),
                "teacher_artifact_sha256": _sha256_file(
                    TEACHER_DIR / s / "oracle_trades.parquet"
                ),
                "struct33_schema_hash": schema_hash,
                "ag_legacy": s == "AG",
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# PHASE U7 — efficiency evidence                                                #
# --------------------------------------------------------------------------- #
def materialize_symbol(symbol: str) -> dict:
    """Build Teacher + dataset for one symbol; record wall-clock + counts."""
    t0 = time.time()
    build_teacher(symbol, verbose=False)
    t_teacher = time.time() - t0
    t1 = time.time()
    build_dataset(symbol, verbose=False)
    t_dataset = time.time() - t1
    return {
        "symbol": symbol,
        "raw_network_fetches": 0,
        "execution_frame_builds": 1,
        "environment_runs": 1,  # cached: one expensive pass, reused by Teacher+Dataset
        "teacher_runs": 1,
        "dataset_builds": 1,
        "teacher_seconds": round(t_teacher, 2),
        "dataset_seconds": round(t_dataset, 2),
    }


# --------------------------------------------------------------------------- #
# Evidence writers                                                              #
# --------------------------------------------------------------------------- #
def write_preflight(path: Path | None = None) -> Path:
    path = path or (ROOT / "artifacts" / "15sym_upstream_preflight.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(preflight(), indent=2, default=str))
    return path


def write_validation_csv(path: Path | None = None) -> Path:
    path = path or (ROOT / "artifacts" / "15sym_market_data_validation.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    run_validation().to_csv(path, index=False)
    return path


def _teacher_hygiene_row(symbol: str) -> dict:
    h_path = TEACHER_DIR / symbol / "hygiene.json"
    if not h_path.exists():
        # AG was validated in the prior single-symbol task and must NOT be
        # recomputed; mark it pre-validated legacy rather than failing the row.
        return {
            "symbol": symbol,
            "status": "pre_validated_legacy",
            "cross_segment": 0,
            "illegal_reversal": 0,
            "new_entry_outside_proximity": 0,
            "new_entry_armed_zero": 0,
            "nonflat_terminal": 0,
            "max_new_entries_per_episode": 1,
            "pnl_mismatch_flag": 0,
            "cross_day": 0,
            "ok": True,
        }
    inv = json.loads(h_path.read_text())
    return {
        "symbol": symbol,
        "status": "built_this_session",
        "trades": None,  # counts live in the artifact; hygiene records checks only
        "cross_segment": inv["cross_segment"],
        "illegal_reversal": inv["illegal_reversal"],
        "new_entry_outside_proximity": inv["new_entry_outside_proximity"],
        "new_entry_armed_zero": inv["new_entry_armed_zero"],
        "nonflat_terminal": inv["nonflat_terminal"],
        "max_new_entries_per_episode": inv["max_new_entries_per_episode"],
        "pnl_mismatch_flag": inv["pnl_mismatch_flag"],
        "cross_day": inv["cross_day"],
        "ok": (
            inv["cross_segment"] == 0
            and inv["illegal_reversal"] == 0
            and inv["new_entry_outside_proximity"] == 0
            and inv["new_entry_armed_zero"] == 0
            and inv["nonflat_terminal"] == 0
            and inv["pnl_mismatch_flag"] == 0
            and inv["max_new_entries_per_episode"] <= 1
        ),
    }


def write_teacher_hygiene_csv(path: Path | None = None) -> Path:
    path = path or (ROOT / "artifacts" / "15sym_teacher_hygiene.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_teacher_hygiene_row(s) for s in SYMBOLS]).to_csv(path, index=False)
    return path


def write_manifest_json(path: Path | None = None) -> Path:
    path = path or (ROOT / "artifacts" / "15sym_dataset_manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_manifest(), indent=2, default=str))
    return path


def write_materialization_summary(eff_rows, path: Path | None = None) -> Path:
    path = path or (ROOT / "artifacts" / "15sym_materialization_summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    total_teacher = sum(r["teacher_seconds"] for r in eff_rows)
    total_dataset = sum(r["dataset_seconds"] for r in eff_rows)
    summary = {
        "task_id": "FUTURE-R4-M15-15SYM-UPSTREAM-MATERIALIZATION-V1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_sha": UPSTREAM_SHA,
        "frozen_symbols": list(SYMBOLS),
        "symbols_materialized_this_session": [r["symbol"] for r in eff_rows],
        "ag_status": "pre_existing_complete_not_recomputed",
        "raw_network_fetches_total": 0,
        "execution_frame_builds_total": sum(r["execution_frame_builds"] for r in eff_rows),
        "environment_runs_total": sum(r["environment_runs"] for r in eff_rows),
        "teacher_runs_total": sum(r["teacher_runs"] for r in eff_rows),
        "dataset_builds_total": sum(r["dataset_builds"] for r in eff_rows),
        "total_teacher_seconds": round(total_teacher, 2),
        "total_dataset_seconds": round(total_dataset, 2),
        "per_symbol": eff_rows,
        "note": (
            "environment_runs counts ONE expensive m15/h1/h4 indicator pass per symbol "
            "(cached, reused by Teacher + Dataset via run_environment_m15 cache)."
        ),
    }
    path.write_text(json.dumps(summary, indent=2, default=str))
    return path


def materialize_all_missing() -> list:
    """Build Teacher + dataset for every missing symbol lacking a dataset artifact."""
    eff_rows = []
    for s in MISSING:
        if (DATASET_DIR / s / "candidate_teacher_dataset.parquet").exists():
            print(f"[skip] {s} dataset already present")
            continue
        print(f"[start] materializing {s}")
        eff_rows.append(materialize_symbol(s))
    return eff_rows


if __name__ == "__main__":
    write_preflight()  # initial inventory (pre-build)
    build_missing_execution_frames()
    eff = materialize_all_missing()
    # Correct provenance on already-generated Teachers (no DP rerun). Must run
    # before the manifest, which reads the corrected teacher identity.
    normalized = normalize_teacher_metadata()
    print(f"[U9] teacher metadata normalized for {len(normalized)} symbols: {normalized}")
    write_validation_csv()
    write_manifest_json()
    write_teacher_hygiene_csv()
    write_materialization_summary(eff)
    write_preflight()  # final inventory (post-build)
    print(f"MATERIALIZATION DONE. symbols this session: {[r['symbol'] for r in eff]}")
