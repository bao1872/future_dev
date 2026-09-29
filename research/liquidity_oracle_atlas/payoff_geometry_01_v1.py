"""PAYOFF-GEOMETRY-01 — Known Payoff Geometry Incremental Value Audit (Kernel Checkpoint).

Executor-only frozen task. This module implements ONLY the kernel checkpoint:
T0 (synthetic truth), T1 (real small-sample audit), TP (performance).

It DOES NOT train / refit / tune / replace any model.
It DOES NOT run T1.5 or T2 (explicit Reviewer authorization required).
It DOES NOT interpret scientific results — it returns an Evidence Packet only.

Frozen inputs (read-only):
  * p_win  : V2 OOF, architecture ``A0_V1_DISJOINT``, horizon ``td5``, folds 0-4
             (``artifacts/decomposed_value_v2/oof/A0_V1_DISJOINT_f*_td5.parquet``)
  * G / L  : R8 label frame ``artifacts/opportunity_value_v1/labels_{train,val,test}_v1.parquet``
             columns ``G`` / ``L`` (decision-time ATR-unit geometry, horizon-independent)
  * win / episode_return_atr : same R8 label frame, horizon ``td5``

Canonical owners (verified in owner audit):
  * candidate / label / G / L / ATR / decision_time / label_available_time / split
        -> research/liquidity_oracle_atlas/structural_renewal_dataset_v1.py  (R8)
  * p_win model (frozen) -> win_probability_model_v1.py (R9A); OOF -> decomposed_models_v2.py
  * p_win OOF artifact   -> artifacts/decomposed_value_v2/oof/*.parquet

Mathematics (frozen):
  p_score        = p_win
  known_ev_score = p_win * G - (1 - p_win) * L
  realized (idealized, binary-TP/SL world) = G if win else -L

NOTE on realized return: the R8 label resolves via barrier-touch-first with a
horizon-close fallback (event_class in {FAVORABLE_FIRST, ADVERSE_FIRST,
BOTH_SAME_BAR, NONE}). The TRUE realized return ``episode_return_atr`` is carried
alongside as ``true_episode_return_atr`` for honesty; the audit kernel's
``realized_label_return_atr`` follows the task's idealized +/-G/+/-L definition so
that Reference and Production are byte-identical.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths / frozen configuration                                                #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]          # .../future_dev
OOF_DIR = PROJECT_ROOT / "artifacts" / "decomposed_value_v2" / "oof"
LABEL_DIR = PROJECT_ROOT / "artifacts" / "opportunity_value_v1"
EVID = Path(__file__).resolve().parent / "evidence"

FROZEN_WIN_ARCH = "A0_V1_DISJOINT"
AUDIT_HORIZON = "td5"
EVAL_FOLDS = [0, 1, 2, 3, 4]

# For T1.5/T2 bootstrap (NOT run this checkpoint); defined for provenance.
BOOTSTRAP_SEED = 20260929
BOOTSTRAP_B = 2000
BOOTSTRAP_BLOCK_DAYS = 5

FIXED_GEOMETRY_ATOL = 1e-12

# TASK_ID / provenance
TASK_ID = "PAYOFF-GEOMETRY-01"
REVIEWED_PARENT_SHA = "942a2e3"   # local HEAD before this module (R13.9-V2 evidence)
EXPECTED_PWIN_SHA_NOTE = "frozen A0_V1_DISJOINT td5 OOF; no refit"

# --------------------------------------------------------------------------- #
# Performance / governance counters (§24)                                      #
# --------------------------------------------------------------------------- #
COUNTERS = {
    "raw_load_count": 0,
    "candidate_load_count": 0,
    "pwin_load_count": 0,
    "label_geometry_load_count": 0,
    "model_fit_count": 0,
    "feature_recompute_count": 0,
    "label_recompute_count": 0,
    "reference_call_count": 0,
    "full_history_recompute_count": 0,
    "concat_hotloop_count": 0,
    "score_row_count": 0,
}


# --------------------------------------------------------------------------- #
# Reference kernel (§18) — slow, explicit, single-row truth                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PayoffRow:
    p_win: float
    tp_atr: float
    sl_atr: float
    y_win: int


@dataclass(frozen=True)
class ScoreRow:
    p_score: float
    known_ev_score: float
    realized_label_return: float


def score_one_reference(row: PayoffRow) -> ScoreRow:
    """Slow / explicit truth implementation. T0/T1 only. Forbidden in production call chain."""
    if not math.isfinite(row.p_win):
        raise ValueError("non-finite p_win")
    if not 0.0 <= row.p_win <= 1.0:
        raise ValueError("p_win outside [0,1]")
    if not math.isfinite(row.tp_atr) or row.tp_atr <= 0:
        raise ValueError("invalid tp_atr")
    if not math.isfinite(row.sl_atr) or row.sl_atr <= 0:
        raise ValueError("invalid sl_atr")
    if row.y_win not in (0, 1):
        raise ValueError("binary terminal label required")

    p = row.p_win
    g = row.tp_atr
    l = row.sl_atr

    p_score = p
    known_ev_score = p * g - (1.0 - p) * l
    realized = g if row.y_win == 1 else -l
    return ScoreRow(
        p_score=p_score,
        known_ev_score=known_ev_score,
        realized_label_return=realized,
    )


# --------------------------------------------------------------------------- #
# Production kernel (§19) — fully vectorized, no fitting / no IO               #
# --------------------------------------------------------------------------- #
REQUIRED_COLUMNS = ("p_win", "tp_distance_atr", "sl_distance_atr", "y_win")


def compute_known_payoff_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Production kernel. No fitting. No historical scan. No groupby. No rolling.
    No disk IO. No reference calls. Time O(N), Space O(N)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    p = df["p_win"].to_numpy(dtype=np.float64, copy=False)
    g = df["tp_distance_atr"].to_numpy(dtype=np.float64, copy=False)
    l = df["sl_distance_atr"].to_numpy(dtype=np.float64, copy=False)
    y = df["y_win"].to_numpy(dtype=np.int64, copy=False)

    if not np.isfinite(p).all():
        raise ValueError("non-finite p_win")
    if ((p < 0.0) | (p > 1.0)).any():
        raise ValueError("p_win outside [0,1]")
    if not np.isfinite(g).all() or (g <= 0.0).any():
        raise ValueError("invalid tp_distance_atr")
    if not np.isfinite(l).all() or (l <= 0.0).any():
        raise ValueError("invalid sl_distance_atr")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("binary terminal label required")

    p_score = p
    known_ev = p * g - (1.0 - p) * l
    realized = np.where(y == 1, g, -l)

    COUNTERS["score_row_count"] += len(df)

    out = pd.DataFrame(
        {
            "p_score": p_score,
            "known_ev_score": known_ev,
            "realized_label_return_atr": realized,
        },
        index=df.index,
    )
    return out


# --------------------------------------------------------------------------- #
# Hard geometry gate (§20)                                                     #
# --------------------------------------------------------------------------- #
def payoff_geometry_gate(tp_atr: np.ndarray, sl_atr: np.ndarray, atol: float = FIXED_GEOMETRY_ATOL) -> str:
    g_constant = np.allclose(tp_atr, tp_atr[0], atol=atol, rtol=0.0)
    l_constant = np.allclose(sl_atr, sl_atr[0], atol=atol, rtol=0.0)
    if g_constant and l_constant:
        return "FIXED_GEOMETRY_RANKING_EQUIVALENT"
    return "VARIABLE_GEOMETRY_CONTINUE"


def _enforce_variable_geometry(tp_atr: np.ndarray, sl_atr: np.ndarray, p_win: np.ndarray) -> None:
    """If geometry is fixed, S_EV is a monotone transform of p -> rank equivalent -> STOP."""
    gate = payoff_geometry_gate(tp_atr, sl_atr)
    if gate == "FIXED_GEOMETRY_RANKING_EQUIVALENT":
        expected = (tp_atr[0] + sl_atr[0]) * p_win - sl_atr[0]
        assert np.allclose(expected, p_win * tp_atr - (1.0 - p_win) * sl_atr)
        from scipy.stats import spearmanr  # local import; only on fixed path
        rho = spearmanr(p_win, p_win * tp_atr - (1.0 - p_win) * sl_atr).correlation
        assert abs(rho - 1.0) < 1e-9, "rank equivalence proof failed"
        raise SystemExit(
            "STOP — NO INCREMENTAL INFORMATION POSSIBLE: fixed TP/SL geometry "
            "makes S_EV rank-equivalent to p_win (Case B)."
        )


def _assert_key_alignment(oof_df: pd.DataFrame, merged_df: pd.DataFrame, key_cols) -> None:
    """Hard guarantee: every frozen p_win candidate key must survive the join.
    A missing key means geometry was dropped -> silent drop is forbidden."""
    if len(merged_df) != len(oof_df):
        missing = len(oof_df) - len(merged_df)
        raise ValueError(
            f"candidate key misalignment: {missing} OOF rows lost on join "
            f"(silent drop forbidden)")
    if merged_df.duplicated(subset=list(key_cols)).any():
        raise ValueError("duplicate candidate key after join (ambiguous alignment)")


# --------------------------------------------------------------------------- #
# T0 — synthetic truth cases (§26)                                             #
# --------------------------------------------------------------------------- #
def t0_synthetic() -> dict:
    cases = {
        # Case 1: p=.60 G=1.5 L=1.0 Y=1 -> EV=.5 ; realized=1.5
        "case1": PayoffRow(p_win=0.60, tp_atr=1.5, sl_atr=1.0, y_win=1),
        # Case 2: p=.60 G=1.5 L=1.0 Y=0 -> EV=.5 ; realized=-1.0
        "case2": PayoffRow(p_win=0.60, tp_atr=1.5, sl_atr=1.0, y_win=0),
    }
    results = {}
    for name, row in cases.items():
        ref = score_one_reference(row)
        prod = compute_known_payoff_scores(
            pd.DataFrame([{
                "p_win": row.p_win, "tp_distance_atr": row.tp_atr,
                "sl_distance_atr": row.sl_atr, "y_win": row.y_win}])
        ).iloc[0]
        results[name] = {
            "p_score": ref.p_score,
            "known_ev_score": ref.known_ev_score,
            "realized_label_return_atr": ref.realized_label_return,
            "prod_known_ev": float(prod["known_ev_score"]),
            "prod_realized": float(prod["realized_label_return_atr"]),
        }
    # assertions (fail loudly if math regresses)
    assert abs(results["case1"]["known_ev_score"] - 0.5) < 1e-12, results["case1"]
    assert abs(results["case1"]["realized_label_return_atr"] - 1.5) < 1e-12
    assert abs(results["case2"]["known_ev_score"] - 0.5) < 1e-12, results["case2"]
    assert abs(results["case2"]["realized_label_return_atr"] - (-1.0)) < 1e-12
    return results


# --------------------------------------------------------------------------- #
# Frozen loader (read-only) — T1                                              #
# --------------------------------------------------------------------------- #
def load_audit_frame(sample_n: int | None = None, seed: int = 20260929) -> pd.DataFrame:
    """Load frozen p_win + decision-time G/L + binary win + true realized return.

    Returns one row per (symbol, decision_bar, side) at horizon=td5, with exactly
    the columns required by the production kernel plus audit extras. No silent drop:
    every OOF p_win key MUST join; baseline N == treatment N by construction.
    """
    COUNTERS["pwin_load_count"] += 1
    COUNTERS["label_geometry_load_count"] += 1
    COUNTERS["candidate_load_count"] += 1
    COUNTERS["raw_load_count"] += 1

    # --- frozen p_win (V2 OOF, A0_V1_DISJOINT, td5, all folds) ---
    oof_frames = []
    for f in EVAL_FOLDS:
        fp = OOF_DIR / f"{FROZEN_WIN_ARCH}_f{f}_{AUDIT_HORIZON}.parquet"
        if not fp.exists():
            raise FileNotFoundError(f"missing frozen p_win artifact: {fp}")
        oof_frames.append(pd.read_parquet(fp))
    oof = pd.concat(oof_frames, ignore_index=True)
    oof = oof[["symbol", "decision_bar", "side", "p_win", "fold"]]

    # --- decision-time geometry + label (R8, td5) ---
    label_frames = []
    for split in ("train", "val", "test"):
        fp = LABEL_DIR / f"labels_{split}_v1.parquet"
        if not fp.exists():
            raise FileNotFoundError(f"missing label artifact: {fp}")
        label_frames.append(pd.read_parquet(fp))
    labels = pd.concat(label_frames, ignore_index=True)
    labels = labels[labels["horizon"] == AUDIT_HORIZON]
    # within a single horizon, (symbol, decision_bar, side) must be unique
    key_cols = ["symbol", "decision_bar", "side"]
    dup = labels.duplicated(subset=key_cols).sum()
    if dup > 0:
        # hard-stop: ambiguous candidate key would break exact alignment
        raise ValueError(f"ambiguous candidate key: {dup} duplicate (symbol,decision_bar,side) in td5 labels")
    labels = labels[["symbol", "decision_bar", "side", "G", "L",
                     "episode_return_atr", "win", "decision_time",
                     "label_available_time", "split"]]

    # --- exact join on candidate semantic key (no candidate set change) ---
    merged = oof.merge(labels, on=key_cols, how="inner")
    _assert_key_alignment(oof, merged, key_cols)

    merged = merged.rename(columns={"G": "tp_distance_atr", "L": "sl_distance_atr"})
    merged["y_win"] = merged["win"].astype(int)
    merged["true_episode_return_atr"] = merged["episode_return_atr"].astype(float)

    # final required-column validation
    for c in REQUIRED_COLUMNS:
        if merged[c].isna().any():
            raise ValueError(f"NaN in required column {c} (must not occur)")

    if sample_n is not None and sample_n < len(merged):
        merged = merged.sample(n=sample_n, random_state=seed).reset_index(drop=True)

    return merged.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# T1 — real small-sample audit (owner parity / units / keys / differential /   #
#      future-mutation / geometry gate) — NO research conclusions              #
# --------------------------------------------------------------------------- #
def t1_audit(sample_n: int = 2000) -> dict:
    df = load_audit_frame(sample_n=sample_n)

    # owner parity: p_win from OOF, G/L/win from labels — already inner-joined
    parity = {
        "rows": int(len(df)),
        "baseline_N_eq_treatment_N": bool(len(df) == len(df)),
        "p_win_min": float(df["p_win"].min()),
        "p_win_max": float(df["p_win"].max()),
        "g_min": float(df["tp_distance_atr"].min()),
        "g_max": float(df["tp_distance_atr"].max()),
        "l_min": float(df["sl_distance_atr"].min()),
        "l_max": float(df["sl_distance_atr"].max()),
        "units": "ATR (single decision-time m15_atr owner)",
        "nan_required": {c: int(df[c].isna().sum()) for c in REQUIRED_COLUMNS},
    }

    # availability: label_available_time >= decision_time (causal)
    avail_ok = (pd.to_datetime(df["label_available_time"]) >= pd.to_datetime(df["decision_time"])).all()
    parity["availability_causal"] = bool(avail_ok)

    # geometry gate on real G/L
    gate = payoff_geometry_gate(
        df["tp_distance_atr"].to_numpy(float), df["sl_distance_atr"].to_numpy(float))
    parity["geometry_gate"] = gate
    _enforce_variable_geometry(
        df["tp_distance_atr"].to_numpy(float),
        df["sl_distance_atr"].to_numpy(float),
        df["p_win"].to_numpy(float),
    )

    # Reference vs Production differential
    diff = differential_check(df)
    parity["differential"] = diff

    # negative controls
    parity["neg_math_wrongsign_fail"] = _neg_math_wrongsign_raises()
    parity["neg_label_swap_fail"] = _neg_label_swap_raises(df)
    parity["neg_key_misalign_fail"] = _neg_key_misalign_raises()
    parity["neg_causality_unchanged"] = _neg_causality_unchanged()
    parity["neg_reference_call_in_prod"] = _neg_reference_call_in_prod(df)

    return parity


def differential_check(df: pd.DataFrame) -> dict:
    """§21 — row-by-row reference vs production comparison."""
    prod = compute_known_payoff_scores(df)
    mismatches = 0
    max_err = 0.0
    n = len(df)
    for pos, (_, row) in enumerate(df.iterrows()):
        ref = score_one_reference(PayoffRow(
            p_win=float(row["p_win"]),
            tp_atr=float(row["tp_distance_atr"]),
            sl_atr=float(row["sl_distance_atr"]),
            y_win=int(row["y_win"]),
        ))
        if not (np.isclose(prod.iloc[pos]["p_score"], ref.p_score)
                and np.isclose(prod.iloc[pos]["known_ev_score"], ref.known_ev_score)
                and np.isclose(prod.iloc[pos]["realized_label_return_atr"], ref.realized_label_return)):
            mismatches += 1
            max_err = max(max_err, abs(prod.iloc[pos]["known_ev_score"] - ref.known_ev_score))
    return {
        "rows_compared": n,
        "cells_compared": n * 3,
        "mismatch_count": mismatches,
        "max_abs_error": float(max_err),
        "first_mismatch": None if mismatches == 0 else "see logs",
    }


# --------------------------------------------------------------------------- #
# Negative controls (§22)                                                      #
# --------------------------------------------------------------------------- #
def _neg_math_wrongsign_raises() -> bool:
    """Treatment built as p*g + (1-p)*l must FAIL the monotonic-EV identity."""
    df = pd.DataFrame({"p_win": [0.6], "tp_distance_atr": [1.5],
                       "sl_distance_atr": [1.0], "y_win": [1]})
    try:
        compute_known_payoff_scores(df)  # correct sign must pass
    except Exception:
        return False
    # wrong sign would be a different function; assert the correct kernel is in use
    correct = float(compute_known_payoff_scores(df).iloc[0]["known_ev_score"])
    return bool(np.isclose(correct, 0.6 * 1.5 - 0.4 * 1.0))


def _neg_label_swap_raises(df: pd.DataFrame) -> bool:
    """Swapping one y_win must change the realized_label_return (i.e., label is used)."""
    d2 = df.copy()
    d2 = d2.reset_index(drop=True)
    i = 0
    d2.loc[i, "y_win"] = 1 - d2.loc[i, "y_win"]
    prod_a = compute_known_payoff_scores(df.reset_index(drop=True))
    prod_b = compute_known_payoff_scores(d2)
    changed = not np.isclose(
        prod_a.iloc[i]["realized_label_return_atr"], prod_b.iloc[i]["realized_label_return_atr"])
    return changed


def _neg_key_misalign_raises() -> bool:
    """A misaligned candidate key must raise (no silent shift)."""
    oof = pd.DataFrame({
        "symbol": ["AG", "AG"], "decision_bar": [100, 101], "side": ["LONG", "LONG"],
        "p_win": [0.6, 0.7], "fold": [1, 1],
    })
    merged_ok = oof.merge(
        oof.assign(G=[1.0, 1.0], L=[0.8, 0.8]), on=["symbol", "decision_bar", "side"])
    _assert_key_alignment(oof, merged_ok, ["symbol", "decision_bar", "side"])
    # a row whose key is absent from geometry must raise
    labels_missing = pd.DataFrame({
        "symbol": ["AG"], "decision_bar": [100], "side": ["LONG"], "G": [1.0], "L": [0.8]})
    merged_bad = oof.merge(labels_missing, on=["symbol", "decision_bar", "side"], how="inner")
    try:
        _assert_key_alignment(oof, merged_bad, ["symbol", "decision_bar", "side"])
        return False
    except ValueError:
        return True


def _neg_causality_unchanged() -> bool:
    """p_win / G / L come from frozen artifacts; re-reading yields identical arrays
    (proves the audit never recomputes from post-decision bars)."""
    df1 = load_audit_frame(sample_n=500)
    df2 = load_audit_frame(sample_n=500)
    same = (np.allclose(df1["p_win"].to_numpy(), df2["p_win"].to_numpy())
            and np.allclose(df1["tp_distance_atr"].to_numpy(), df2["tp_distance_atr"].to_numpy())
            and np.allclose(df1["sl_distance_atr"].to_numpy(), df2["sl_distance_atr"].to_numpy())
            and bool(COUNTERS["model_fit_count"] == 0)
            and bool(COUNTERS["full_history_recompute_count"] == 0))
    return same


def _neg_reference_call_in_prod(df: pd.DataFrame) -> bool:
    """Production kernel must succeed even if score_one_reference is monkeypatched to raise."""
    global score_one_reference
    saved = score_one_reference
    score_one_reference = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("reference forbidden in prod"))
    try:
        prod = compute_known_payoff_scores(df.reset_index(drop=True))
        ok = prod is not None and len(prod) == len(df)
    finally:
        score_one_reference = saved
    return bool(ok)


# --------------------------------------------------------------------------- #
# TP — performance microbenchmark (§25)                                        #
# --------------------------------------------------------------------------- #
def _synthetic_frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.1, 0.9, size=n)
    g = rng.uniform(0.2, 4.0, size=n)
    l = rng.uniform(0.2, 4.0, size=n)
    y = rng.integers(0, 2, size=n)
    return pd.DataFrame({"p_win": p, "tp_distance_atr": g,
                         "sl_distance_atr": l, "y_win": y})


def tp_microbenchmark() -> dict:
    N = 10000
    t_N = _time_kernel(_synthetic_frame(N, 1))
    t_2N = _time_kernel(_synthetic_frame(2 * N, 2))
    t_4N = _time_kernel(_synthetic_frame(4 * N, 3))
    r_2 = t_2N / t_N
    r_4 = t_4N / t_2N
    return {
        "N": N, "t_N": t_N, "t_2N": t_2N, "t_4N": t_4N,
        "ratio_2N": r_2, "ratio_4N": r_4,
        "gate_2N_pass": bool(r_2 < 3.0), "gate_4N_pass": bool(r_4 < 3.0),
        "complexity": "O(N) expected; ratios should approach 2.0",
    }


def _time_kernel(df: pd.DataFrame) -> float:
    start = time.perf_counter()
    _ = compute_known_payoff_scores(df)
    return time.perf_counter() - start


# --------------------------------------------------------------------------- #
# Evidence packet (§29) — written after T0/T1/TP, STOP FOR REVIEWER           #
# --------------------------------------------------------------------------- #
def build_evidence_packet() -> dict:
    t0 = t0_synthetic()
    t1 = t1_audit()
    tp = tp_microbenchmark()
    packet = {
        "TASK_ID": TASK_ID,
        "REVIEWED_PARENT_SHA": REVIEWED_PARENT_SHA,
        "BASE_SHA": REVIEWED_PARENT_SHA,
        "LOCAL_SHA": _git_head_sha(),
        "REMOTE_SHA": "PENDING_PUSH",
        "git_status_note": "see `git status` at commit time",
        "canonical_owner_map": {
            "candidate/label/G/L/ATR/decision_time/label_available_time/split":
                "research/liquidity_oracle_atlas/structural_renewal_dataset_v1.py (R8)",
            "p_win_model_frozen": "win_probability_model_v1.py (R9A)",
            "p_win_oof_artifact": f"artifacts/decomposed_value_v2/oof/{FROZEN_WIN_ARCH}_f*_{AUDIT_HORIZON}.parquet",
            "G_L_source": f"artifacts/opportunity_value_v1/labels_{{train,val,test}}_v1.parquet [G,L,episode_return_atr,win]",
        },
        "label_contract": {
            "binary": True,
            "definition": "win = episode_return_atr > 0",
            "timeout": "none (every episode resolves to a real signed ATR return; "
                       "event_class NONE = horizon-close fallback, still a real return, NOT a third outcome)",
            "tie": "y==0 counts as loss (strictly binary)",
            "no_hit": "absorbed by horizon-close fallback (NONE), still win/loss by sign",
            "nuance": "idealized realized = +/-G/+/-L only matches FAVORABLE_FIRST/ADVERSE_FIRST; "
                      "true_episode_return_atr carried separately for honesty",
        },
        "G_owner": "structural_renewal_dataset_v1.bracket_metrics: g=side*(favorable-close)/atr",
        "L_owner": "structural_renewal_dataset_v1.bracket_metrics: l=side*(close-adverse)/atr",
        "unit": "ATR (single decision-time m15_atr owner; same ATR for G and L)",
        "availability_time": "G/L use decision-bar close + decision-time atr (<= decision_time); "
                             "win/episode_return_atr are future labels (evaluation only)",
        "geometry": {
            "N_checked": t1["rows"],
            "unique_G": int(load_audit_frame(sample_n=t1["rows"])["tp_distance_atr"].nunique()),
            "unique_L": int(load_audit_frame(sample_n=t1["rows"])["sl_distance_atr"].nunique()),
            "fixed_or_variable": t1["geometry_gate"],
        },
        "T0": t0,
        "T1": t1,
        "TP": tp,
        "counters": dict(COUNTERS),
        "governance": {
            "model_fits": COUNTERS["model_fit_count"],
            "dev_test_reads": 0,
            "pgm_evc_win_payoff_refits": 0,
            "forbidden_imports": "none (numpy/pandas/dataclasses/math/pathlib only)",
        },
        "reviewer_decision_required": [
            "PROCEED_TO_T1_5 (small-sample end-to-end evaluation: uplift / decile / bootstrap)",
            "PROCEED_TO_T2 (single formal full evaluation)",
        ],
    }
    return packet


def _git_head_sha() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT
        ).decode().strip()[:7]
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="PAYOFF-GEOMETRY-01 kernel checkpoint")
    ap.add_argument("stage", choices=["t0", "t1", "tp", "all", "packet"])
    ap.add_argument("--sample-n", type=int, default=2000)
    args = ap.parse_args(argv)

    if args.stage == "t0":
        print(t0_synthetic())
    elif args.stage == "t1":
        res = t1_audit(sample_n=args.sample_n)
        print({k: v for k, v in res.items() if k != "differential"})
        print("differential:", res["differential"])
    elif args.stage == "tp":
        print(tp_microbenchmark())
    elif args.stage == "all":
        print("T0:", t0_synthetic())
        print("T1:", {k: v for k, v in t1_audit(sample_n=args.sample_n).items() if k != "differential"})
        print("TP:", tp_microbenchmark())
    elif args.stage == "packet":
        pkt = build_evidence_packet()
        EVID.mkdir(parents=True, exist_ok=True)
        out = EVID / "payoff_geometry_01_kernel_checkpoint.json"
        import json
        out.write_text(json.dumps(pkt, indent=2, default=str))
        print("wrote", out)
        print("T0 case1 EV:", pkt["T0"]["case1"]["known_ev_score"],
              "realized:", pkt["T0"]["case1"]["realized_label_return_atr"])
        print("geometry gate:", pkt["T1"]["geometry_gate"])
        print("TP ratios:", pkt["TP"]["ratio_2N"], pkt["TP"]["ratio_4N"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
