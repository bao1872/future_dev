"""decomposed_value_composer_isolation_audit_v1.py — R13.5B Composer Isolation Audit.

DIAGNOSTIC ONLY. No model fit, no parameter tuning, no feature selection,
no DEV VAL, no old TEST, no change to the frozen Phase-4 result.

R13.5 already proved the exact per-row identity

    EV_hat - Y = probability_error + magnitude_error

and showed that, at the candidate level, Win sorting (+0.22 ATR top-vs-bottom)
collapses once Payoff is composed into EV (EV top-vs-bottom ~ +0.06 ATR, CI
includes 0). R13.5B isolates *which step* of the Composer destroys the Win
signal's return-ranking ability.

Key identity (Payoff model provides two distinct operations on the Win signal):

    p_BE           = mu_L / (mu_W + mu_L)
    EV_C           = (mu_W + mu_L) * (p - p_BE)
                    = p*mu_W - (1-p)*mu_L          (== reconstructed EV)

We compare five scores, each a scalar per row, by their return-ranking ability
(top-20% minus bottom-20% actual return, weighted, with complete 5-trading-day
block bootstrap CI):

    p                : pure Win signal
    EV_W             : Win signal + FIXED TRAIN payoff prior magnitude
                       (W0, L0 constant)  -> tests "does per-row Payoff
                        magnitude matter, or just a constant scale?"
    p - p_BE         : Win signal shifted by Payoff break-even threshold
    EV_C             : full Composer  (== EV)
    EV_R             : Payoff magnitudes only, FIXED TRAIN win-rate prior p0
                       -> tests "do Payoff magnitudes alone rank return?"

Decision logic (per reviewer R13.5B):
  * p good & EV_W good but p-p_BE fails  -> break-even threshold is the culprit
  * p-p_BE good but x(mu_W+mu_L) fails   -> magnitude scaling is the culprit
  * EV_R good & p good but EV_C fails    -> genuine joint composition failure

Reviewed base: d7c228e5b2c0bedb0c4890a75335d97c4937ed1a
Parent R13.5:    3b493511a2584d8194080d208141f2c98fd80b8c
Phase-4 remains frozen: NO_V2_MODEL_IMPROVEMENT
Direction layer (E9) deliberately NOT audited: no frozen causal TRAIN-OOF E9
root axis exists for this population.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas import run_decomposed_v2_research as R
from research.liquidity_oracle_atlas import decomposed_value_closure_audit_v1 as C

# --------------------------------------------------------------------------- #
# Paths / constants                                                            #
# --------------------------------------------------------------------------- #
HORIZON = C.HORIZON
N_FOLDS = C.N_FOLDS
BOOTSTRAP_B = C.BOOTSTRAP_B
BOOTSTRAP_SEED = C.BOOTSTRAP_SEED
BOOTSTRAP_BLOCK = C.BOOTSTRAP_BLOCK

ARCH_PRIMARY = C.ARCH_PRIMARY
ARCH_SECONDARY = C.ARCH_SECONDARY
POP_ALL = C.POP_ALL
POP_ROOT = C.POP_ROOT

EVIDENCE_DIR = R.EVIDENCE_DIR
LABELS_TRAIN = R.ALLOWED_V1_LABELS_TRAIN
STATE_V1 = R.ALLOWED_V1_STATE
PHASE4_MANIFEST_JSON = R.PHASE4_MANIFEST_JSON

REVIEWED_BASE_SHA = "d7c228e5b2c0bedb0c4890a75335d97c4937ed1a"
PARENT_R135_SHA = "3b493511a2584d8194080d208141f2c98fd80b8c"
PHASE4_STATUS = "NO_V2_MODEL_IMPROVEMENT"

SUMMARY_JSON = os.path.join(EVIDENCE_DIR, "decomposed_value_composer_isolation_audit_v1_summary.json")
DETAIL_CSV = os.path.join(EVIDENCE_DIR, "decomposed_value_composer_isolation_audit_v1_detail.csv")
MANIFEST_JSON = os.path.join(EVIDENCE_DIR, "decomposed_value_composer_isolation_audit_v1_manifest.json")
LEDGER_PARQUET = os.path.join(os.path.dirname(R.V2_OOF_DIR), "composer_isolation_audit_v1.parquet")

SCORE_NAMES = ["p", "EV_W", "p_minus_pBE", "EV_C", "EV_R"]


# --------------------------------------------------------------------------- #
# Small helpers                                                                #
# --------------------------------------------------------------------------- #
def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wmean(x, w):
    return C.wmean(x, w)


# --------------------------------------------------------------------------- #
# Score construction                                                           #
# --------------------------------------------------------------------------- #
def add_isolation_scores(full_closure: pd.DataFrame, W0: float, L0: float,
                         p0: float) -> pd.DataFrame:
    """Add the five Composer-isolation scores to a per-row closure frame.

    `full_closure` already carries episode_return_atr, p_win, mu_win, mu_loss,
    sample_weight, ev_recomputed (from C.add_closure_columns). W0/L0/p0 are the
    FIXED TRAIN payoff priors (constant across rows).
    """
    out = full_closure.copy()
    y = out["episode_return_atr"].to_numpy(float)
    w = out["sample_weight"].to_numpy(float)
    p = np.clip(out["p_win"].to_numpy(float), 0.0, 1.0)
    W = np.maximum(out["mu_win"].to_numpy(float), 0.0)
    L = np.maximum(out["mu_loss"].to_numpy(float), 0.0)
    denom = W + L
    pbe = np.where(denom > 0, L / denom, np.nan)

    ev_c = (W + L) * (p - pbe)
    np.testing.assert_allclose(
        ev_c, out["ev_recomputed"].to_numpy(float), rtol=1e-10, atol=1e-10)

    out["score_p"] = p
    out["score_EV_W"] = p * W0 - (1.0 - p) * L0
    out["score_p_minus_pBE"] = p - pbe
    out["score_EV_C"] = ev_c
    out["score_EV_R"] = p0 * W - (1.0 - p0) * L
    return out


def _train_priors(oof_closure: pd.DataFrame):
    """Weighted TRAIN payoff/win-rate priors from the full OOF (all rows)."""
    w = oof_closure["sample_weight"].to_numpy(float)
    W = np.maximum(oof_closure["mu_win"].to_numpy(float), 0.0)
    L = np.maximum(oof_closure["mu_loss"].to_numpy(float), 0.0)
    p = np.clip(oof_closure["p_win"].to_numpy(float), 0.0, 1.0)
    return wmean(W, w), wmean(L, w), wmean(p, w)


# --------------------------------------------------------------------------- #
# Complete-block bootstrap contrast                                            #
# --------------------------------------------------------------------------- #
def _complete_block_indices(trading_days: np.ndarray, block: int):
    days = np.sort(pd.unique(trading_days))
    n_complete = len(days) // block
    days = days[: n_complete * block]
    pos = {d: np.where(trading_days == d)[0] for d in days}
    blocks = [days[i * block:(i + 1) * block] for i in range(n_complete)]
    block_idx = [np.concatenate([pos[d] for d in b]) for b in blocks]
    return block_idx


def _top_bottom_contrast(s, y, w):
    m = np.isfinite(s) & np.isfinite(y) & (w > 0)
    s, y, w = s[m], y[m], w[m]
    if len(s) < 20:
        return np.nan
    hi, lo = np.quantile(s, 0.8), np.quantile(s, 0.2)
    top, bot = s >= hi, s <= lo
    if not (top.any() and bot.any()):
        return np.nan
    return wmean(y[top], w[top]) - wmean(y[bot], w[bot])


def composer_isolation_point(df: pd.DataFrame) -> Dict[str, float]:
    y = df["episode_return_atr"].to_numpy(float)
    w = df["sample_weight"].to_numpy(float)
    out = {}
    for name in SCORE_NAMES:
        out[name] = float(_top_bottom_contrast(
            df["score_" + name].to_numpy(float), y, w))
    return out


def composer_isolation_bootstrap(df: pd.DataFrame, n_boot: int = BOOTSTRAP_B,
                                 seed: int = BOOTSTRAP_SEED,
                                 block: int = BOOTSTRAP_BLOCK
                                 ) -> Dict[str, Dict[str, float]]:
    y = df["episode_return_atr"].to_numpy(float)
    w = df["sample_weight"].to_numpy(float)
    td = df["trading_day"].to_numpy()
    block_idx = _complete_block_indices(td, block)
    n_blocks = len(block_idx)
    rng = np.random.default_rng(seed)

    scores = {name: df["score_" + name].to_numpy(float) for name in SCORE_NAMES}

    out: Dict[str, Dict[str, float]] = {}
    for name, arr in scores.items():
        boots = np.empty(n_boot)
        for b in range(n_boot):
            chosen = rng.integers(0, n_blocks, size=n_blocks)
            sel = np.concatenate([block_idx[c] for c in chosen])
            boots[b] = _top_bottom_contrast(arr[sel], y[sel], w[sel])
        boots = boots[np.isfinite(boots)]
        if boots.size:
            lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
            excludes_zero = bool((lo > 0) or (hi < 0))
        else:
            lo = hi = np.nan
            excludes_zero = False
        out[name] = {
            "point": float(composer_isolation_point(df)[name]),
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "ci_excludes_zero": excludes_zero,
            "n_complete_blocks": int(n_blocks),
        }
    return out


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def build_audit() -> Dict[str, Any]:
    a0 = C.load_td5_oof(ARCH_PRIMARY)
    a1 = C.load_td5_oof(ARCH_SECONDARY)

    key = ["symbol", "decision_bar", "side", "horizon"]
    if set(map(tuple, a0[key].to_numpy())) != set(map(tuple, a1[key].to_numpy())):
        raise RuntimeError("STOP_R13_5B_ARCH_KEY_UNIVERSE_MISMATCH")

    C.verify_oof_labels(a0)
    C.verify_oof_labels(a1)

    audit: Dict[str, Any] = {"populations": {}, "scores": {}}
    ledger_parts = []
    detail_rows = []

    for arch, oof in [(ARCH_PRIMARY, a0), (ARCH_SECONDARY, a1)]:
        full = C.add_closure_columns(oof)
        full = C.join_state(full)
        W0, L0, p0 = _train_priors(full)

        for population in [POP_ALL, POP_ROOT]:
            df = full.copy()
            if population == POP_ROOT:
                df = df[df["candidate_at_decision"] == True].copy()  # noqa: E712
            df = add_isolation_scores(df, W0, L0, p0)

            point = composer_isolation_point(df)
            if population == POP_ROOT:
                boot = composer_isolation_bootstrap(df)
            else:
                boot = {k: {"point": v, "ci_lo": np.nan, "ci_hi": np.nan,
                            "ci_excludes_zero": None, "n_complete_blocks": None}
                        for k, v in point.items()}

            for name in SCORE_NAMES:
                detail_rows.append({
                    "architecture": arch,
                    "population": population,
                    "score": name,
                    "point_contrast": point[name],
                    "ci_lo": boot[name]["ci_lo"],
                    "ci_hi": boot[name]["ci_hi"],
                    "ci_excludes_zero": boot[name]["ci_excludes_zero"],
                    "n_complete_blocks": boot[name]["n_complete_blocks"],
                })

            audit["populations"].setdefault(arch, {})[population] = {
                "n_rows": int(len(df)),
                "W0": float(W0), "L0": float(L0), "p0": float(p0),
                "point": point,
                "bootstrap": boot if population == POP_ROOT else None,
            }

            # ledger rows
            led = df[[
                "symbol", "decision_bar", "decision_time", "trading_day",
                "side", "fold", "candidate_at_decision",
                "episode_return_atr", "sample_weight",
                "p_win", "mu_win", "mu_loss", "ev_recomputed",
                "score_p", "score_EV_W", "score_p_minus_pBE",
                "score_EV_C", "score_EV_R",
            ]].copy()
            led["architecture"] = arch
            led["population"] = population
            led["actual_return"] = led["episode_return_atr"]
            ledger_parts.append(led)

    audit["score_names"] = SCORE_NAMES
    return audit, ledger_parts, pd.DataFrame(detail_rows)


def write_artifacts(audit, ledger_parts, detail_df) -> Dict[str, str]:
    ledger = pd.concat(ledger_parts, ignore_index=True)
    R.write_parquet(ledger, LEDGER_PARQUET)
    detail_df.to_csv(DETAIL_CSV, index=False)
    return {
        "ledger": sha256_of(LEDGER_PARQUET),
        "detail": sha256_of(DETAIL_CSV),
    }


def build_manifest(audit, artifact_shas, generator_code_sha: str) -> dict:
    a0_shards = [sha256_of(R._unit_paths(ARCH_PRIMARY, f, HORIZON)[0])
                 for f in range(N_FOLDS)]
    a1_shards = [sha256_of(R._unit_paths(ARCH_SECONDARY, f, HORIZON)[0])
                 for f in range(N_FOLDS)]
    return {
        "task_id": "FUTURE-R13.5B-TD5-COMPOSER-ISOLATION-AUDIT-V1",
        "reviewed_base_sha": REVIEWED_BASE_SHA,
        "parent_r135_sha": PARENT_R135_SHA,
        "phase4_status": PHASE4_STATUS,
        "phase4_manifest_sha": sha256_of(PHASE4_MANIFEST_JSON),
        "generator_code_sha": generator_code_sha,
        "bootstrap": {
            "method": "trading_day_block_bootstrap_complete",
            "block_trading_days": BOOTSTRAP_BLOCK,
            "n_boot": BOOTSTRAP_B,
            "seed": BOOTSTRAP_SEED,
            "n_complete_blocks_ROOT_A0":
                audit["populations"][ARCH_PRIMARY][POP_ROOT]["bootstrap"]
                ["p"]["n_complete_blocks"],
        },
        "input_artifact_shas": {
            "a0_td5_oof_shards": a0_shards,
            "a1_td5_oof_shards": a1_shards,
            "labels_train_v1": sha256_of(LABELS_TRAIN),
            "state_v1": sha256_of(STATE_V1),
            "phase4_manifest": sha256_of(PHASE4_MANIFEST_JSON),
        },
        "output_artifact_shas": artifact_shas,
        "summary_sha": sha256_of(SUMMARY_JSON) if os.path.exists(SUMMARY_JSON) else None,
        "score_names": SCORE_NAMES,
        "direction_layer": {
            "production_root_direction": "E9",
            "audited_in_r13_5b": False,
            "reason": "NO_FROZEN_CAUSAL_TRAIN_OOF_E9_ROOT_AXIS",
        },
        "governance": {
            "model_fits": 0,
            "hyperparameter_searches": 0,
            "feature_changes": 0,
            "direction_model_fits": 0,
            "dev_val_reads": 0,
            "old_test_label_reads": 0,
            "old_test_policy_reads": 0,
            "E9_train_oof_axis_reads": 0,
        },
    }


def main() -> None:
    audit, ledger_parts, detail_df = build_audit()
    artifact_shas = write_artifacts(audit, ledger_parts, detail_df)
    R.write_json_evidence(audit, SUMMARY_JSON)

    generator_code_sha = sha256_of(os.path.abspath(__file__))
    manifest = build_manifest(audit, artifact_shas, generator_code_sha)
    R.write_json_evidence(manifest, MANIFEST_JSON)

    print("R13.5B composer isolation audit complete.")
    b = audit["populations"][ARCH_PRIMARY][POP_ROOT]["bootstrap"]
    for name in SCORE_NAMES:
        r = b[name]
        print("  A0/ROOT %-12s point=%.4f CI[%.4f, %.4f] excl0=%s" % (
            name, r["point"], r["ci_lo"], r["ci_hi"], r["ci_excludes_zero"]))


if __name__ == "__main__":
    main()
