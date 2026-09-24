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

import hashlib
import json
import os
import time
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.build_decomposed_value_dataset_v1 import (
    HORIZONS,
)
from research.liquidity_oracle_atlas import decomposed_models_v2 as M

# --------------------------------------------------------------------------- #
# Directories                                                                  #
# --------------------------------------------------------------------------- #
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(MODULE_DIR))

V2_ARTIFACT_DIR = os.path.join("artifacts", "decomposed_value_v2")
V2_CACHE_DIR = os.path.join(V2_ARTIFACT_DIR, "cache")
V2_UNIT_DIR = os.path.join(V2_CACHE_DIR, "unit")
V2_OOF_DIR = os.path.join(V2_ARTIFACT_DIR, "oof")
EXTENDED_MANIFEST_JSON = os.path.join(
    V2_ARTIFACT_DIR, "extended_state_manifest_v2.json")
EXTENDED_STATE_PARQUET = os.path.join(
    V2_ARTIFACT_DIR, "extended_causal_state_v2.parquet")

EVIDENCE_DIR = os.path.join("research", "liquidity_oracle_atlas", "evidence")
SELECTION_JSON = os.path.join(
    EVIDENCE_DIR, "model_selection_train_only_v2.json")
PHASE4_MANIFEST_JSON = os.path.join(
    EVIDENCE_DIR, "decomposed_v2_phase4_manifest.json")

# Historical generator identities — recorded in the Phase-4 evidence manifest
# but NEVER rewritten here. These are the code commits that produced the
# extended-state artifact and the train-only selection respectively.
EXTENDED_STATE_GENERATOR_CODE_SHA = (
    "2eb2f7243c8bc1e1d1dd24101ebd8f9dac90f302")
SELECTION_GENERATOR_CODE_SHA = (
    "9a221cdc1f45b031f3f0e13f0860c6b6715391f6")

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


class StopV2SelectionNotCommitted(RuntimeError):
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


def selection_is_committed(path: Optional[str] = None) -> dict:
    """P6: the selection must be TRACKED and byte-identical to HEAD.

    Merely existing on disk is not enough — an uncommitted or locally modified
    selection JSON must not be able to unlock DEV VAL.
    """
    import subprocess

    path = path or SELECTION_JSON

    def git(*args):
        return subprocess.run(["git", *args], cwd=os.getcwd(),
                              capture_output=True, text=True)

    rel = os.path.relpath(path, os.getcwd())
    checks = {
        "exists": os.path.exists(path),
        "tracked": git("ls-files", "--error-unmatch", rel).returncode == 0,
        "no_unstaged_diff": git("diff", "--quiet", "HEAD", "--", rel).returncode == 0,
        "no_staged_diff": git("diff", "--cached", "--quiet", "HEAD", "--",
                              rel).returncode == 0,
    }
    checks["committed_clean"] = all(checks.values())
    return checks


def assert_selection_committed(path: Optional[str] = None) -> dict:
    checks = selection_is_committed(path)
    if not checks["committed_clean"]:
        raise StopV2SelectionNotCommitted(
            "STOP_V2_SELECTION_NOT_COMMITTED " + json.dumps(checks))
    return checks


def assert_selection_frozen() -> dict:
    """Every VAL-outcome read must pass BOTH gates: frozen AND committed."""
    assert_selection_committed()
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
    """Parent-process counters only.

    Heavy fitting happens in isolated subprocesses (plan §51), so these are
    NOT the fit count. Use `aggregate_efficiency()` for reported evidence.
    """
    return dict(COUNTERS)


FITS_PER_UNIT = 6          # 3 ES fits + 3 refits
R12_ARCH_NAMES = ("A0", "A1", "A2", "A3")
R13_ARCH_NAMES = ("B0", "B1", "B2", "B3", "B4")


def aggregate_unit_evidence(plan=None, horizons=HORIZONS) -> dict:
    """P5: mechanically aggregate the real evidence from unit metadata.

    Validates that every (architecture x fold x horizon) unit exists EXACTLY
    once and that its recorded content matches its identity.
    """
    from research.liquidity_oracle_atlas import (
        decomposed_value_features_v2 as F,
    )
    if plan is None:
        plan = build_plan_from_devframe()

    seen, missing, duplicate, inconsistent = set(), [], [], []
    per_arch = {}
    for group, names in (("R12", R12_ARCH_NAMES), ("R13", R13_ARCH_NAMES)):
        for short in names:
            arch = F.get_arch(short)
            n = 0
            for k in range(plan.n_outer):
                for h in horizons:
                    _shard, meta_p = _unit_paths(arch.name, k, h)
                    key = (arch.name, int(k), h)
                    if not os.path.exists(meta_p):
                        missing.append(list(key))
                        continue
                    with open(meta_p) as f:
                        meta = json.load(f)
                    if (int(meta.get("fold", -1)) != int(k)
                            or meta.get("horizon") != h
                            or meta.get("arch") != arch.name):
                        inconsistent.append(list(key))
                    if key in seen:
                        duplicate.append(list(key))
                    seen.add(key)
                    n += 1
            per_arch[arch.name] = n

    r12_units = sum(per_arch.get(F.get_arch(s).name, 0) for s in R12_ARCH_NAMES)
    r13_units = sum(per_arch.get(F.get_arch(s).name, 0) for s in R13_ARCH_NAMES)
    total = r12_units + r13_units

    return {
        "r12_units": r12_units,
        "r13_units": r13_units,
        "total_units": total,
        "fits_per_unit": FITS_PER_UNIT,
        "total_model_fits": total * FITS_PER_UNIT,
        "missing_units": len(missing),
        "duplicate_units": len(duplicate),
        "inconsistent_units": len(inconsistent),
        "missing_detail": missing[:20],
        "duplicate_detail": duplicate[:20],
        "inconsistent_detail": inconsistent[:20],
        "units_per_architecture": per_arch,
        "expected_r12_units": len(R12_ARCH_NAMES) * plan.n_outer * len(horizons),
        "expected_r13_units": len(R13_ARCH_NAMES) * plan.n_outer * len(horizons),
    }


def aggregate_efficiency(plan=None) -> dict:
    """P5: reported efficiency = extended-state manifest + unit aggregate."""
    units = aggregate_unit_evidence(plan=plan)
    env = {"environment_loads": None, "geometry_passes": None,
           "feature_materializations": None}
    if os.path.exists(EXTENDED_MANIFEST_JSON):
        with open(EXTENDED_MANIFEST_JSON) as f:
            man = json.load(f)
        env = {
            "environment_loads": man.get("environment_loads"),
            "geometry_passes": man.get("geometry_passes"),
            "feature_materializations": man.get("feature_materializations"),
        }
    return {**env, **units}


def assert_clean_efficiency() -> dict:
    """Plan §52: leakage counters must be zero whenever we report."""
    snap = efficiency_snapshot()
    if snap["old_test_label_reads"] != 0 or snap["old_test_policy_reads"] != 0:
        raise StopV2Leakage(
            "STOP_V2_OLD_TEST_LEAKAGE final "
            f"label={snap['old_test_label_reads']} "
            f"policy={snap['old_test_policy_reads']}")
    return snap


def _git_head_sha() -> str:
    """Current repository HEAD sha (read-only, no network)."""
    import subprocess
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=os.getcwd(),
        capture_output=True, text=True).stdout.strip()


# --------------------------------------------------------------------------- #
# Phase-4 final evidence closure (C4-C8)                                       #
# --------------------------------------------------------------------------- #
def _sha256_path(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _unit_identity_records() -> list:
    """C6: ordered canonical unit-metadata identities for a stable SHA.

    No fit rerun — reads the already-committed unit JSON metadata only.
    """
    from research.liquidity_oracle_atlas import (
        decomposed_value_features_v2 as F)
    plan = build_plan_from_devframe()
    records = []
    for short in R12_ARCH_NAMES + R13_ARCH_NAMES:
        arch = F.get_arch(short)
        for k in range(plan.n_outer):
            for h in HORIZONS:
                _, meta_p = _unit_paths(arch.name, k, h)
                if not os.path.exists(meta_p):
                    continue
                with open(meta_p) as f:
                    meta = json.load(f)
                bi = meta.get("best_iteration", {}) or {}
                records.append({
                    "arch": meta.get("arch"),
                    "fold": int(meta.get("fold")),
                    "horizon": meta.get("horizon"),
                    "best_iteration_win": bi.get("win"),
                    "best_iteration_win_mag": bi.get("win_mag"),
                    "best_iteration_loss_mag": bi.get("loss_mag"),
                    "ev_mse": meta.get("ev_mse"),
                })
    records.sort(key=lambda r: (r["arch"], r["fold"], r["horizon"]))
    return records


def _unit_identity_sha(records: list) -> str:
    canon = json.dumps(records, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode()).hexdigest()


def build_phase4_manifest(write: bool = True) -> dict:
    """C4-C8: bind the full Phase-4 lineage into one committed evidence file.

    Pure evidence assembly — reads committed artifacts only, performs NO model
    fit and changes NO scientific result. Leakage counters must be zero.
    """
    assert_clean_efficiency()
    from research.liquidity_oracle_atlas import (
        decomposed_value_features_v2 as F)

    r12 = pd.read_csv(MODEL_COMPARISON_CSV)
    a0 = next(r for r in r12.to_dict(orient="records")
              if r["candidate"] == "A0_V1_DISJOINT")
    a1 = next(r for r in r12.to_dict(orient="records")
              if r["candidate"] == "A1_SHARE_TO_WIN")

    agg = aggregate_unit_evidence()
    sel = json.load(open(SELECTION_JSON))
    sel_checks = selection_is_committed()
    unit_records = _unit_identity_records()

    manifest = {
        "task": "FUTURE-R11-R14-V2-PHASE4-EVIDENCE-CLOSURE",
        "stage": "TRAIN_ONLY_PHASE4_FROZEN",
        "scientific_status": "NO_V2_MODEL_IMPROVEMENT",
        "reviewed_parent_sha": "612294a5d32d5f02a34857cff966b4202c763e23",
        "generator_code_sha": _git_head_sha(),

        "artifact_sha256": {
            "extended_state_manifest_v2.json":
                _sha256_path(EXTENDED_MANIFEST_JSON),
            "extended_causal_state_v2.parquet":
                _sha256_path(EXTENDED_STATE_PARQUET),
            "dev_frame_v2.parquet": _sha256_path(DEV_FRAME_PARQUET),
            "decomposed_v2_stability_atlas.csv":
                _sha256_path(STABILITY_ATLAS_CSV),
            "decomposed_v2_model_comparison.csv":
                _sha256_path(MODEL_COMPARISON_CSV),
            "decomposed_v2_feature_ablation.csv":
                _sha256_path(FEATURE_ABLATION_CSV),
            "model_selection_train_only_v2.json":
                _sha256_path(SELECTION_JSON),
        },
        "extended_state_generator_code_sha":
            EXTENDED_STATE_GENERATOR_CODE_SHA,
        "selection_generator_code_sha": SELECTION_GENERATOR_CODE_SHA,

        "unit_evidence": {
            "r12_units": agg["r12_units"],
            "r13_units": agg["r13_units"],
            "total_units": agg["total_units"],
            "fits_per_unit": FITS_PER_UNIT,
            "total_model_fits": agg["total_model_fits"],
            "missing_units": agg["missing_units"],
            "duplicate_units": agg["duplicate_units"],
            "inconsistent_units": agg["inconsistent_units"],
            "units_per_architecture": agg["units_per_architecture"],
            "unit_identity_sha256": _unit_identity_sha(unit_records),
        },

        "leakage": {
            "old_test_label_reads": int(COUNTERS["old_test_label_reads"]),
            "old_test_policy_reads": int(COUNTERS["old_test_policy_reads"]),
            "val_outcomes_read": False,
            "selection_committed_clean": bool(sel_checks["committed_clean"]),
        },

        "scientific_interpretation": {
            "primary_status": "NO_V2_MODEL_IMPROVEMENT",
            "v2_selected": False,
            "one_se_candidate": sel["one_se_candidate"],
            "secondary_pre_val_challenger": "A1_SHARE_TO_WIN",
            "a1_evidence": {
                "a0_mean_ev_mse": round(float(a0["mean_ev_mse"]), 10),
                "a1_mean_ev_mse": round(float(a1["mean_ev_mse"]), 10),
                "paired_point": round(float(a1["paired_point"]), 10),
                "paired_ci_95_low": round(float(a1["paired_ci_low"]), 10),
                "paired_ci_95_high": round(float(a1["paired_ci_high"]), 10),
                "status": str(a1.get("paired_status", "")),
            },
            "note": ("A1 retained as secondary pre-VAL challenger; NOT "
                     "promoted to SUPPORTED (paired 95% CI crosses zero)."),
        },
    }
    if write:
        write_json_evidence(manifest, PHASE4_MANIFEST_JSON)
    return manifest


DEV_FRAME_PARQUET = os.path.join(V2_ARTIFACT_DIR, "dev_frame_v2.parquet")
V1_WIN_FEATURES = os.path.join(V1_ARTIFACT_DIR, "win_features_v1.parquet")
V1_PAYOFF_FEATURES = os.path.join(V1_ARTIFACT_DIR, "payoff_features_v1.parquet")

JOIN_KEYS = ("symbol", "decision_bar", "side")


def build_development_frame(extra: Optional[pd.DataFrame] = None,
                            save: bool = True) -> pd.DataFrame:
    """TRAIN labels joined with the feature contract, on (symbol, bar, side).

    V1 WIN33 / PAY8 come from the frozen V1 feature parquets (so A0 IS the V1
    matrix by construction). `extra` carries V2 columns keyed the same way.
    """
    labels = read_train_labels()
    win = read_parquet(V1_WIN_FEATURES)
    pay = read_parquet(V1_PAYOFF_FEATURES)

    # Labels already carry some geometry columns (e.g. log_structural_rr). The
    # frozen V1 FEATURE parquet is the contract's source of truth, so the label
    # duplicates are dropped instead of being suffix-split.
    overlap = ((set(win.columns) | set(pay.columns))
               - set(JOIN_KEYS)) & set(labels.columns)
    if overlap:
        labels = labels.drop(columns=sorted(overlap))

    df = labels.merge(win, on=list(JOIN_KEYS), how="left",
                      validate="many_to_one")
    df = df.merge(pay, on=list(JOIN_KEYS), how="left",
                  validate="many_to_one")
    if extra is not None:
        df = df.merge(extra, on=list(JOIN_KEYS), how="left",
                      validate="many_to_one")

    # Join completeness = every label row FOUND a feature row. Individual
    # feature values may legitimately be NaN (e.g. liquidity distance when no
    # liquidity zone is active); LightGBM consumes those natively and V1 did
    # too. An unmatched key, by contrast, leaves the whole row NaN.
    from research.liquidity_oracle_atlas.decomposed_value_features_v2 import (
        SHARED41,
    )
    block = df[list(SHARED41)]
    unmatched = block.isna().all(axis=1)
    if bool(unmatched.any()) or len(df) != len(labels):
        raise StopV2Leakage(
            "STOP_V2_FEATURE_JOIN_INCOMPLETE "
            f"n_unmatched={int(unmatched.sum())} "
            f"rows={len(df)} labels={len(labels)}")
    bump("feature_materializations")
    if save:
        write_parquet(df, DEV_FRAME_PARQUET)
    return df


def load_development_frame(columns: Optional[list] = None) -> pd.DataFrame:
    return read_parquet(DEV_FRAME_PARQUET, columns=columns)


def build_plan_from_devframe() -> Any:
    """The outer calendar is built ONCE on the full frame (all horizons)."""
    from research.liquidity_oracle_atlas.walkforward_development_v1 import (
        build_fold_plan,
    )
    frame = load_development_frame(columns=[
        "symbol", "decision_bar", "side", "horizon", "decision_time",
        "label_available_time", "sample_weight", "episode_return_atr"])
    return build_fold_plan(frame)


def _unit_paths(arch_name: str, fold: int, horizon: str):
    tag = f"{arch_name}_f{fold}_{horizon}"
    return (os.path.join(V2_OOF_DIR, f"{tag}.parquet"),
            os.path.join(V2_UNIT_DIR, f"{tag}.json"))


def run_architecture(arch_name: str, *, horizons=HORIZONS, force: bool = False,
                     isolated: bool = True, plan=None, dev_path=None):
    """Run one architecture across all outer folds and horizons.

    Plan §51: each (arch, fold, horizon) unit runs in its OWN subprocess by
    default, so no two LightGBM boosters ever share a process. Completed units
    are cached on disk, so an interrupted run resumes instead of refitting.
    """
    import subprocess as _sp
    import sys
    from research.liquidity_oracle_atlas import decomposed_value_features_v2 as F
    from research.liquidity_oracle_atlas import decomposed_models_v2 as M

    arch = F.get_arch(arch_name)
    if arch is None:
        raise StopV2Leakage(f"STOP_V2_UNKNOWN_ARCH {arch_name}")
    if plan is None:
        plan = build_plan_from_devframe()
    dev_path = dev_path or DEV_FRAME_PARQUET

    os.makedirs(V2_OOF_DIR, exist_ok=True)
    os.makedirs(V2_UNIT_DIR, exist_ok=True)

    shards, metas = [], []
    n_launched = 0
    for k in range(plan.n_outer):
        for h in horizons:
            shard_p, meta_p = _unit_paths(arch.name, k, h)
            if (not force) and os.path.exists(shard_p) and os.path.exists(meta_p):
                with open(meta_p) as f:
                    metas.append(json.load(f))
                shards.append(shard_p)
                continue
            args_path = os.path.join(V2_UNIT_DIR, f"_args_{arch.name}_{k}_{h}.json")
            with open(args_path, "w") as f:
                json.dump({
                    "arch": arch.name, "fold": int(k),
                    "outer_bounds": list(plan.outer[k]),
                    "es_frac": float(plan.es_frac), "horizon": h,
                    "dev_path": dev_path, "out_path": shard_p,
                    "out_meta": meta_p,
                }, f)
            if isolated:
                env = dict(os.environ)
                env["OMP_NUM_THREADS"] = "1"
                env["PYTHONPATH"] = os.getcwd()
                r = _sp.run([sys.executable, "-m",
                             "research.liquidity_oracle_atlas.decomposed_models_v2",
                             args_path], env=env, capture_output=True, text=True)
                if r.returncode != 0:
                    raise StopV2FitError(
                        f"STOP_V2_UNIT_FAILED arch={arch.name} fold={k} "
                        f"horizon={h} rc={r.returncode}\n{r.stderr[-2000:]}")
            else:
                M.worker_main(args_path)
            with open(meta_p) as f:
                metas.append(json.load(f))
            shards.append(shard_p)
            n_launched += 1

    oof = pd.concat([read_parquet(p) for p in shards], ignore_index=True)
    return {
        "arch": arch.name,
        "win_cols": tuple(arch.win),
        "payoff_cols": tuple(arch.payoff),
        "n_win": arch.n_win,
        "n_payoff": arch.n_payoff,
        # one-SE simplicity metric: the SHARED feature count.
        "n_features": arch.n_payoff,
        "shared_state": arch.shared_state,
        "oof": oof,
        "metas": metas,
        "n_units_launched": n_launched,
        "n_units_total": len(shards),
    }


class StopV2FitError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Comparison / selection inputs (plan §13, §24, §26)                           #
# --------------------------------------------------------------------------- #
def per_fold_ev_mse(oof: pd.DataFrame) -> list:
    """One weighted EV MSE per outer fold (all horizons pooled in that fold)."""
    out = []
    for k in sorted(oof["fold"].unique()):
        g = oof[oof["fold"] == k]
        out.append(M.weighted_ev_mse(
            g["episode_return_atr"], g["ev_c"], g["sample_weight"]))
    return out


def ev_decile_spread(oof: pd.DataFrame, n_bins: int = 10):
    """Mean actual return in the top EV decile minus the bottom EV decile."""
    q = pd.qcut(oof["ev_c"].rank(method="first"), n_bins, labels=False)
    means = []
    for d in range(n_bins):
        m = (q.to_numpy() == d)
        means.append(M.weighted_mean(
            oof["episode_return_atr"].to_numpy(float)[m],
            oof["sample_weight"].to_numpy(float)[m]))
    return float(means[-1] - means[0]), means


def overall_metrics(oof: pd.DataFrame) -> dict:
    y = oof["episode_return_atr"].to_numpy(float)
    w = oof["sample_weight"].to_numpy(float)
    win = (y > 0).astype(float)
    avg_win = M.weighted_mean(y[y > 0], w[y > 0])
    avg_loss = M.weighted_mean(np.abs(y[~(y > 0)]), w[~(y > 0)])
    spread, _ = ev_decile_spread(oof)
    return {
        "n_rows": int(len(oof)),
        "weighted_ev_mse": M.weighted_ev_mse(y, oof["ev_c"].to_numpy(float), w),
        "weighted_ev_mae": M.weighted_ev_mae(y, oof["ev_c"].to_numpy(float), w),
        "win_brier": M.weighted_brier(win, oof["p_win"].to_numpy(float), w),
        "win_logloss": M.weighted_logloss(win, oof["p_win"].to_numpy(float), w),
        "actual_win_rate": M.weighted_mean(win, w),
        "win_magnitude_mae": M.weighted_mae(
            y[y > 0], oof["mu_win"].to_numpy(float)[y > 0], w[y > 0]),
        "loss_magnitude_mae": M.weighted_mae(
            np.abs(y[~(y > 0)]),
            oof["mu_loss"].to_numpy(float)[~(y > 0)], w[~(y > 0)]),
        "actual_rr": (avg_win / avg_loss
                      if np.isfinite(avg_loss) and avg_loss > 0 else float("nan")),
        "ev_decile_spread": spread,
    }


def build_model_comparison(arch_names: Sequence[str], *,
                           baseline: str = "A0",
                           horizons=HORIZONS, write: bool = True,
                           out_csv: str = MODEL_COMPARISON_CSV):
    """R12/R13 comparison table: per-fold EV MSE, mean/SE, paired bootstrap."""
    oofs, summaries, shared, counts = {}, {}, {}, {}
    for name in arch_names:
        r = run_architecture(name, horizons=horizons)
        oofs[name] = r["oof"]
        shared[name] = bool(r["shared_state"])
        # P4: `n_features` is the one-SE simplicity metric (the SHARED count for
        # every eligible candidate). It is ambiguous for the R12 ablations, so
        # the three explicit counts are reported alongside it.
        counts[name] = {
            "n_win_features": int(r["n_win"]),
            "n_payoff_features": int(r["n_payoff"]),
            "n_feature_union": int(len(set(r["win_cols"]) | set(r["payoff_cols"]))),
        }
        summaries[name] = M.fold_summary(
            candidate=r["arch"], n_features=r["n_features"],
            fold_losses=per_fold_ev_mse(r["oof"]))

    base_oof = oofs[baseline]
    base_days = M.daily_loss_series(base_oof)

    rows = []
    for name in arch_names:
        row = dict(summaries[name])
        row.update(counts[name])
        row.update(overall_metrics(oofs[name]))
        row["shared_state"] = shared[name]
        if name != baseline:
            cand_days = M.daily_loss_series(oofs[name])
            aligned = cand_days.align(base_days, join="inner")
            boot = M.paired_block_bootstrap(
                aligned[0].to_numpy(float), aligned[1].to_numpy(float))
            row.update({f"paired_{k}": v for k, v in boot.items()})
        rows.append(row)

    out = pd.DataFrame(rows)
    if write:
        write_csv_evidence(out, out_csv)
    return out


def build_train_only_selection(baseline: str = "A0",
                                 comparison_csv: str = FEATURE_ABLATION_CSV,
                                 require_shared_state: bool = True,
                                 write: bool = True) -> dict:
    """Phase 4 deliverable: one-SE selection over TRAIN-only R13 results.

    Plan §27: this JSON is the ONLY artifact that unlocks DEV VAL. It must be
    committed before any VAL outcome is read. NO VAL data is touched here.
    """
    # Leakage counters must be zero in the reporting process (plan §52).
    assert_clean_efficiency()
    from research.liquidity_oracle_atlas import decomposed_value_features_v2 as F

    df = pd.read_csv(comparison_csv)
    candidates = []
    for r in df.to_dict(orient="records"):
        candidates.append({
            "candidate": str(r["candidate"]),
            "n_features": int(r["n_features"]),
            "mean_ev_mse": float(r["mean_ev_mse"]),
            "se_ev_mse": float(r["se_ev_mse"]),
            "shared_state": bool(r.get("shared_state", False)),
        })
    # The Composer requires a coherent shared-state candidate (plan §25).
    eligible = [c for c in candidates
                if (not require_shared_state or c["shared_state"])]
    sel = M.one_se_select(eligible)
    selected = next(c for c in candidates if c["candidate"] == sel["candidate"])
    sel_arch = F.get_arch(selected["candidate"])

    # A1_SHARE_TO_WIN is the only preregistered architecture that improves EV
    # direction on TRAIN OOF (R12), but its paired CI crosses zero, so it is a
    # secondary challenger only — not a V2 winner, and it must not unlock VAL
    # on its own. Numbers come from the committed R12 comparison table.
    r12 = pd.read_csv(MODEL_COMPARISON_CSV)
    a1 = next((r for r in r12.to_dict(orient="records")
               if r["candidate"] == "A1_SHARE_TO_WIN"), None)
    secondary = None
    if a1 is not None:
        secondary = {
            "candidate": "A1_SHARE_TO_WIN",
            "mean_ev_mse": float(a1["mean_ev_mse"]),
            "se_ev_mse": float(a1["se_ev_mse"]),
            "paired_point": float(a1["paired_point"]),
            "paired_ci_low": float(a1["paired_ci_low"]),
            "paired_ci_high": float(a1["paired_ci_high"]),
            "paired_status": str(a1.get("paired_status", "")),
            "ci_crosses_zero": bool(
                float(a1["paired_ci_low"]) < 0 < float(a1["paired_ci_high"])),
            "source": "R12 model_comparison.csv (TRAIN OOF)",
        }

    out = {
        "task": "FUTURE-R13-PAYOFF-GEOMETRY-V2-TRAIN-ONLY-SELECTION",
        "status": "NO_V2_MODEL_IMPROVEMENT",
        "basis": "TRAIN-only OOF EV MSE (no VAL outcome read)",
        "val_unlocked": False,
        "v2_selected": False,
        "v2_selected_architecture": None,
        "one_se_candidate": sel["candidate"],
        "one_se_candidate_aliases": sorted({sel["candidate"], "A3_SHARED_BOTH"}),
        "one_se_candidate_shared_state": selected["shared_state"],
        "one_se_candidate_n_features": selected["n_features"],
        "one_se_candidate_mean_ev_mse": selected["mean_ev_mse"],
        "one_se_candidate_se_ev_mse": selected["se_ev_mse"],
        "one_se_candidate_win_schema_sha256": (
            sel_arch.win_schema_sha256 if sel_arch else None),
        "one_se_candidate_payoff_schema_sha256": (
            sel_arch.payoff_schema_sha256 if sel_arch else None),
        "one_se_threshold": sel["one_se_threshold"],
        "best_candidate": sel["best_candidate"],
        "n_eligible": sel["n_eligible"],
        "secondary_pre_val_challenger": "A1_SHARE_TO_WIN",
        "secondary_pre_val_challenger_evidence": secondary,
        "hypothesis_status": "POST_TRAIN_PRE_VAL_SECONDARY_CHALLENGER",
        "leakage": {
            "old_test_label_reads": int(COUNTERS["old_test_label_reads"]),
            "old_test_policy_reads": int(COUNTERS["old_test_policy_reads"]),
        },
        "baseline": baseline,
        "require_shared_state": require_shared_state,
        "candidates": candidates,
        "verdict": (
            "No V2 extended-state family (SPACE18/PATH8/VOL6) improved TRAIN "
            "EV MSE within one SE of the V1 shared baseline; the one-SE "
            "selector therefore keeps the simplest shared-state candidate "
            "(B0_SHARED41 == A3_SHARED_BOTH, SHARED41). Status is "
            "NO_V2_MODEL_IMPROVEMENT; v2_selected=False. A1_SHARE_TO_WIN is "
            "retained as the only secondary pre-VAL challenger (TRAIN OOF "
            "direction improvement but CI crosses zero)."),
        "generator_code_sha": _git_head_sha(),
    }
    if write:
        write_json_evidence(out, SELECTION_JSON)
        print(f"  wrote {SELECTION_JSON}", flush=True)
    return out


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
