"""FUTURE-R11-R14 V2 — single research entry point + guarded IO.

TRADING_METRICS: NOT_APPLICABLE
reason: No trading action has been defined.
Phases 1-4 produce feature/atlas/model evidence only; no policy is simulated,
so no win-rate / payoff-ratio / expectancy claim may be made here.

Why this module exists
----------------------
The old TEST has already been opened by V1. Plan §44 therefore forbids V2 from
ever touching it. Relying on developer discipline is not acceptable, so EVERY
V2 read goes through `read_frame()` / `read_parquet()`, which pattern-matches
the path against a hardcoded forbidden list, bumps a tamper counter and raises
`STOP_V2_OLD_TEST_LEAKAGE`.

Plan §27 additionally requires that no VAL outcome is loaded until
`model_selection_train_only_v2.json` exists AND is committed. `read_val_*`
therefore calls `assert_selection_frozen()` first.

Isolation rule (plan §2): no V1 module is modified. V1 is imported as read-only
helpers only.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Directories                                                                  #
# --------------------------------------------------------------------------- #
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(MODULE_DIR))

V2_ARTIFACT_DIR = os.path.join("artifacts", "decomposed_value_v2")
V2_CACHE_DIR = os.path.join(V2_ARTIFACT_DIR, "cache")
V2_UNIT_DIR = os.path.join(V2_CACHE_DIR, "unit")
V2_OOF_DIR = os.path.join(V2_ARTIFACT_DIR, "oof")
EXTENDED_STATE_PARQUET = os.path.join(
    V2_ARTIFACT_DIR, "extended_causal_state_v2.parquet")

EVIDENCE_DIR = os.path.join("research", "liquidity_oracle_atlas", "evidence")
SELECTION_JSON = os.path.join(
    EVIDENCE_DIR, "model_selection_train_only_v2.json")

STABILITY_ATLAS_CSV = os.path.join(EVIDENCE_DIR, "decomposed_v2_stability_atlas.csv")
MODEL_COMPARISON_CSV = os.path.join(EVIDENCE_DIR, "decomposed_v2_model_comparison.csv")
FEATURE_ABLATION_CSV = os.path.join(EVIDENCE_DIR, "decomposed_v2_feature_ablation.csv")

# Frozen V1 inputs that V2 IS allowed to read.
V1_ARTIFACT_DIR = os.path.join("artifacts", "decomposed_value_v1")
ALLOWED_V1_STATE = os.path.join(V1_ARTIFACT_DIR, "state_v1.parquet")
ALLOWED_V1_LABELS_TRAIN = os.path.join(V1_ARTIFACT_DIR, "labels_train_v1.parquet")
ALLOWED_V1_LABELS_VAL = os.path.join(V1_ARTIFACT_DIR, "labels_val_v1.parquet")


# --------------------------------------------------------------------------- #
# Leakage guard                                                                #
# --------------------------------------------------------------------------- #
# Anything whose name matches a forbidden token may not be read by V2. The list
# covers old TEST labels AND every product of the closed V1 Formal run, so that
# V1 outcomes cannot leak into model/feature/symbol/regime selection (§44-§46).
# Each entry is (token, kind). `kind` selects which plan-§3 counter is bumped:
#   "label"  -> old TEST label data
#   "policy" -> products of the closed V1 Formal run (outcomes / evidence)
FORBIDDEN_TOKENS: tuple[tuple[str, str], ...] = (
    ("labels_test_v1", "label"),                    # old TEST labels never again
    ("decomposed_value_test_diagnostics", "policy"),  # V1 TEST diagnostics
    ("decomposed_value_daily_returns", "policy"),     # V1 Formal daily PnL
    ("decomposed_value_trade_ledger", "policy"),      # V1 Formal trades
    ("decomposed_value_decision_ledger", "policy"),   # V1 Formal decisions
    ("decomposed_value_policy_summary", "policy"),    # V1 Formal policy outcomes
    ("decomposed_value_per_symbol", "policy"),        # V1 Formal per-symbol
    ("decomposed_value_summary.json", "policy"),      # V1 Formal verdicts
    ("decomposed_value_manifest.json", "policy"),     # V1 Formal manifest
    ("decomposed_predictions_test", "policy"),        # V1 TEST predictions
)


class StopV2Leakage(RuntimeError):
    pass


class StopV2ValUnlocked(RuntimeError):
    pass


COUNTERS: dict[str, int] = {
    # plan §3 hard leakage counters
    "old_test_label_reads": 0,
    "old_test_policy_reads": 0,
    # plan §49 efficiency counters
    "environment_loads": 0,
    "geometry_passes": 0,
    "feature_materializations": 0,
    "model_fit_count": 0,
}


def reset_counters() -> None:
    for k in list(COUNTERS):
        COUNTERS[k] = 0


def bump(name: str, n: int = 1) -> None:
    COUNTERS[name] = COUNTERS.get(name, 0) + int(n)


def _classify(path: str) -> Optional[str]:
    """Return 'label' / 'policy' if the path is forbidden, else None."""
    norm = path.replace("\\", "/")
    for tok, kind in FORBIDDEN_TOKENS:
        if tok in norm:
            return kind
    return None


def guard_path(path: str) -> str:
    """Raise STOP_V2_OLD_TEST_LEAKAGE and record the attempt, or return path."""
    kind = _classify(path)
    if kind is not None:
        key = ("old_test_label_reads" if kind == "label"
               else "old_test_policy_reads")
        bump(key)
        raise StopV2Leakage(
            f"STOP_V2_OLD_TEST_LEAKAGE kind={kind} path={path!r} "
            f"{key}={COUNTERS[key]}")
    return path


def read_frame(path: str, **kwargs) -> pd.DataFrame:
    """Guarded CSV reader."""
    guard_path(path)
    return pd.read_csv(path, **kwargs)


def read_parquet(path: str, columns: Optional[list] = None) -> pd.DataFrame:
    """Guarded parquet reader."""
    guard_path(path)
    return pd.read_parquet(path, columns=columns)


def write_parquet(df: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# VAL unlock gate (plan §27)                                                   #
# --------------------------------------------------------------------------- #
def selection_exists() -> bool:
    return os.path.exists(SELECTION_JSON)


def load_selection() -> dict:
    if not selection_exists():
        raise StopV2ValUnlocked(
            "STOP_V2_VAL_UNLOCKED missing "
            f"{SELECTION_JSON} (freeze selection before touching VAL)")
    with open(SELECTION_JSON) as f:
        return json.load(f)


def assert_selection_frozen() -> dict:
    """Every VAL-outcome read must pass this gate first."""
    return load_selection()


def read_val_labels(columns: Optional[list] = None) -> pd.DataFrame:
    """VAL may only be loaded after the TRAIN-only selection is frozen."""
    assert_selection_frozen()
    return read_parquet(ALLOWED_V1_LABELS_VAL, columns=columns)


def read_train_labels(columns: Optional[list] = None) -> pd.DataFrame:
    return read_parquet(ALLOWED_V1_LABELS_TRAIN, columns=columns)


def read_state(columns: Optional[list] = None) -> pd.DataFrame:
    return read_parquet(ALLOWED_V1_STATE, columns=columns)


# --------------------------------------------------------------------------- #
# Evidence writers                                                             #
# --------------------------------------------------------------------------- #
def write_csv_evidence(df: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_json_evidence(obj: Any, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def efficiency_snapshot() -> dict:
    return dict(COUNTERS)


def assert_clean_efficiency() -> dict:
    """Plan §52: leakage counters must be zero whenever we report."""
    snap = efficiency_snapshot()
    if snap["old_test_label_reads"] != 0 or snap["old_test_policy_reads"] != 0:
        raise StopV2Leakage(
            "STOP_V2_OLD_TEST_LEAKAGE final "
            f"label={snap['old_test_label_reads']} "
            f"policy={snap['old_test_policy_reads']}")
    return snap


def main() -> dict:
    """Phase-1 smoke: prove the guard and the frozen plan are wired."""
    from research.liquidity_oracle_atlas.walkforward_development_v1 import (
        build_fold_plan, oof_row_count_check, iter_fold_splits)

    reset_counters()
    t0 = time.time()
    out: dict[str, Any] = {"base_head_precondition": "ea02a21"}

    labels = read_train_labels(columns=[
        "symbol", "decision_bar", "side", "horizon", "decision_time",
        "label_available_time", "sample_weight", "episode_return_atr"])
    plan = build_fold_plan(labels)
    out["plan"] = plan.describe()
    out["n_label_rows"] = int(len(labels))

    splits = list(iter_fold_splits(labels, plan))
    out["folds"] = [{
        "fold": s.fold, "n_fit_core": s.n_fit_core, "n_es": s.n_es,
        "n_outer": s.n_outer,
        "purity": s.purity_stats,
    } for s in splits]
    out["oof_coverage"] = oof_row_count_check(splits, labels, plan)
    out["efficiency"] = assert_clean_efficiency()
    out["runtime_sec"] = time.time() - t0
    return out


if __name__ == "__main__":
    import pprint
    pprint.pprint(main())
