"""FUTURE-R11-R14 V2 — R14 forward shadow (plan §37-§43).

TRADING_METRICS: NOT_APPLICABLE
reason: No trading action has been defined.
No forward performance exists yet: the forward period has not begun and this
module is deliberately incapable of emitting any before the lock opens.

Allowed during accumulation (§41)
    data arrived?
    all 15 symbols present?
    feature computation completed?
    artifact hashes valid?
    model SHA unchanged?

Hard-blocked until 80 COMMON trading days have accumulated (§40)
    PnL, win rate, EV performance, symbol performance, deciles,
    regime performance.

Policies (§37)
    F0  P0     frozen Direction baseline
    F1  PR_V1  exact model SHA + policy from the closed V1 Formal run
    F2  PC_V2  only if a selected V2 candidate beats A0 on TRAIN OOF EV MSE.
               Current selection status is NO_V2_MODEL_IMPROVEMENT, so F2 is
               NOT created by this experiment.

Renewal is excluded from the primary forward confirmation (§38).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Optional

from research.liquidity_oracle_atlas import run_decomposed_v2_research as R

FORWARD_REQUIRED_DAYS = 80
FORWARD_BLOCK_DAYS = 5
FORWARD_REQUIRED_BLOCKS = FORWARD_REQUIRED_DAYS // FORWARD_BLOCK_DAYS  # 16
FORWARD_BOOTSTRAP_B = 5000
FORWARD_BOOTSTRAP_SEED = 20260925

FORWARD_LOCK_JSON = os.path.join(R.EVIDENCE_DIR, "forward_lock_v1.json")
SHADOW_STATE_JSON = os.path.join(
    R.V2_ARTIFACT_DIR, "forward_shadow_state_v1.json")

ALL_SYMBOLS_REQUIRED = 15


class StopV2ForwardLocked(RuntimeError):
    pass


class StopV2ForwardIncomplete(RuntimeError):
    pass


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Monitoring (allowed)                                                         #
# --------------------------------------------------------------------------- #
def monitor_ingestion(*, symbols_present, feature_rows_by_symbol,
                      artifact_paths: Optional[dict] = None,
                      model_paths: Optional[dict] = None,
                      days_accumulated: int = 0) -> dict:
    """§41 allowed checks ONLY. Never computes an outcome."""
    n_sym = len(set(symbols_present))
    features_complete = bool(feature_rows_by_symbol) and all(
        int(v) > 0 for v in feature_rows_by_symbol.values())
    hashes = {k: (sha256_file(v) if os.path.exists(v) else None)
              for k, v in (artifact_paths or {}).items()}
    model_hashes = {k: (sha256_file(v) if os.path.exists(v) else None)
                    for k, v in (model_paths or {}).items()}
    return {
        "data_arrived": bool(n_sym) and features_complete,
        "n_symbols_present": n_sym,
        "all_symbols_present": n_sym >= ALL_SYMBOLS_REQUIRED,
        "feature_computation_complete": features_complete,
        "artifact_sha256": hashes,
        "model_sha256": model_hashes,
        "hashes_valid": all(v is not None for v in hashes.values())
        and all(v is not None for v in model_hashes.values()),
        "days_accumulated": int(days_accumulated),
        "days_required": FORWARD_REQUIRED_DAYS,
        "lock_open": False,   # see assert_outcome_lock_open
    }


def assert_outcome_lock_open(days_accumulated: int, *, all_symbols: bool,
                             require: int = FORWARD_REQUIRED_DAYS) -> bool:
    """§40: no outcome inspection before `require` common trading days."""
    if int(days_accumulated) < require:
        raise StopV2ForwardLocked(
            f"STOP_V2_FORWARD_LOCKED days={int(days_accumulated)} "
            f"required={require}")
    if not all_symbols:
        raise StopV2ForwardIncomplete(
            "STOP_V2_FORWARD_SYMBOLS_INCOMPLETE "
            f"required={ALL_SYMBOLS_REQUIRED}")
    return True


# --------------------------------------------------------------------------- #
# Outcome readers (blocked)                                                    #
# --------------------------------------------------------------------------- #
def _guard(what: str, days_accumulated: int, all_symbols: bool):
    assert_outcome_lock_open(days_accumulated, all_symbols=all_symbols)
    # Reached only after the lock opens; the caller supplies the numbers.
    return what


def read_forward_pnl(days_accumulated: int, all_symbols: bool):
    return _guard("pnl", days_accumulated, all_symbols)


def read_forward_win_rate(days_accumulated: int, all_symbols: bool):
    return _guard("win_rate", days_accumulated, all_symbols)


def read_forward_ev(days_accumulated: int, all_symbols: bool):
    return _guard("ev", days_accumulated, all_symbols)


def read_forward_by_symbol(days_accumulated: int, all_symbols: bool):
    return _guard("symbol_performance", days_accumulated, all_symbols)


def read_forward_deciles(days_accumulated: int, all_symbols: bool):
    return _guard("deciles", days_accumulated, all_symbols)


def read_forward_regime(days_accumulated: int, all_symbols: bool):
    return _guard("regime_performance", days_accumulated, all_symbols)


# --------------------------------------------------------------------------- #
# Pre-forward lock (Phase 7; content depends on whether V2 beats A0)           #
# --------------------------------------------------------------------------- #
def build_forward_lock(*, code_sha: str, feature_schema_sha: str,
                       model_sha: dict, training_artifact_sha: dict,
                       policies: list, forward_start_rule: str,
                       include_v2: bool) -> dict:
    """Assemble the preregistration payload. Writes nothing by itself."""
    return {
        "task": "FUTURE-R14-FRESH-FORWARD-CONFIRMATION",
        "code_sha": code_sha,
        "feature_schema_sha256": feature_schema_sha,
        "model_sha256": model_sha,
        "training_artifact_sha256": training_artifact_sha,
        "policies": list(policies),
        "include_v2": bool(include_v2),
        "forward_start_rule": forward_start_rule,
        "required_common_trading_days": FORWARD_REQUIRED_DAYS,
        "required_complete_blocks": FORWARD_REQUIRED_BLOCKS,
        "bootstrap": {"b": FORWARD_BOOTSTRAP_B, "seed": FORWARD_BOOTSTRAP_SEED,
                      "block_days": FORWARD_BLOCK_DAYS},
        "costs": {"gross_only_note":
                  "TRADABLE_ALPHA_NOT_ESTABLISHED until per-symbol "
                  "commission / exchange fee / slippage / tick value exist"},
        "renewal_included": False,
    }


def write_forward_lock(lock: dict, path: str = FORWARD_LOCK_JSON) -> str:
    return R.write_json_evidence(lock, path)
