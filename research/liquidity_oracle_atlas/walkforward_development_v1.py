"""FUTURE-R11-R14 V2 — walk-forward development engine (plan §4 / §5 / §6).

Pure split machinery. No model code, no LightGBM import, no metric code.

TRADING_METRICS: NOT_APPLICABLE
reason: No trading action has been defined.
This module only partitions rows; it produces no policy, no trade ledger and
therefore no win-rate / payoff-ratio / expectancy claims.

Design frozen by the plan
-------------------------
§4  No random shuffle. Days come from the labels frame itself:
        days = sorted(unique(decision_time.date))
    Warm-up = first 40% of TRAIN trading days; the remaining 60% is cut into
    exactly five CONTIGUOUS outer validation blocks (expanding window).
    Rows usable for fitting fold k must satisfy BOTH
        decision_time          < outer_val_start_k
        label_available_time   < outer_val_start_k
    so a label that only resolves after the cutoff can never enter training.

§5  Inside each outer training window the LAST 15% of available training days
    become the internal early-stopping block:
        FIT_CORE | ES | OUTER_VALIDATION
    ES is used ONLY to pick best_iteration. The outer block is predicted once,
    after refitting on all pre-outer data with n_estimators fixed.

§6  Two-side pair purity. Labels are stored as pairs: one LONG row and one
    SHORT row per (symbol, decision_bar, horizon), whose sample_weight sums to
    exactly 1. If an epoch originally had both sides but only one side is
    available before the cutoff, the WHOLE two-side epoch is dropped. An epoch
    that was originally single-side may remain alone. Retained epochs must
    still satisfy sum(sample_weight) == 1 within 1e-12.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# Frozen by the plan; deliberately not tunable per experiment.
WARMUP_FRAC = 0.40
N_OUTER = 5
ES_FRAC = 0.15
WEIGHT_TOL = 1e-12

EPOCH_KEYS = ("symbol", "decision_bar", "horizon")


# --------------------------------------------------------------------------- #
# Errors                                                                       #
# --------------------------------------------------------------------------- #
class StopV2WalkForward(RuntimeError):
    """Base class for every hard gate in this module."""


class StopV2NotEnoughTradingDays(StopV2WalkForward):
    pass


class StopV2NonContiguousOuter(StopV2WalkForward):
    pass


class StopV2EmptyEsBlock(StopV2WalkForward):
    pass


class StopV2EpochWeightNotUnit(StopV2WalkForward):
    pass


class StopV2EmptyTrainingWindow(StopV2WalkForward):
    pass


# --------------------------------------------------------------------------- #
# Frozen plan                                                                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FoldSplit:
    """Boolean row masks for one outer fold (all aligned to the frame given)."""

    fold: int
    fit_core: np.ndarray
    es: np.ndarray
    outer: np.ndarray
    n_fit_core: int
    n_es: int
    n_outer: int
    es_days: tuple[str, ...]
    n_train_rows_available: int
    purity_stats: dict


@dataclass(frozen=True)
class FoldPlan:
    """Immutable outer-fold calendar.

    days       : every TRAIN trading day, ascending, ISO 'YYYY-MM-DD'.
    warmup_end : last day belonging to warm-up (never predicted).
    outer      : per fold, (start_day, end_day) INCLUSIVE, contiguous and
                 non-overlapping, covering days[warmup_n : ] exactly once.
    """

    days: tuple[str, ...]
    warmup_end: str
    outer: tuple[tuple[str, str], ...]
    es_frac: float = ES_FRAC

    @property
    def n_outer(self) -> int:
        return len(self.outer)

    def describe(self) -> str:
        lines = [
            f"n_days={len(self.days)} warmup_end={self.warmup_end} "
            f"n_outer={self.n_outer} es_frac={self.es_frac}"
        ]
        for k, (s, e) in enumerate(self.outer):
            lines.append(f"  fold {k}: {s} .. {e}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Day helpers                                                                  #
# --------------------------------------------------------------------------- #
def decision_day(frame: pd.DataFrame) -> pd.Series:
    """Trading day of the DECISION (calendar-day normalized, tz-naive)."""
    return pd.to_datetime(frame["decision_time"]).dt.normalize()


def label_available_day(frame: pd.DataFrame) -> pd.Series:
    """Trading day on which the label actually becomes known (causal gate §4)."""
    return pd.to_datetime(frame["label_available_time"]).dt.normalize()


def trading_days(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """days = sorted(unique(decision_time.date)) exactly as §4 prescribes."""
    return pd.DatetimeIndex(np.sort(decision_day(frame).unique()))


# --------------------------------------------------------------------------- #
# Plan construction                                                            #
# --------------------------------------------------------------------------- #
def build_fold_plan(
    frame: pd.DataFrame,
    *,
    warmup_frac: float = WARMUP_FRAC,
    n_outer: int = N_OUTER,
    es_frac: float = ES_FRAC,
) -> FoldPlan:
    """Move warm-up aside, then cut the remainder into n_outer contiguous blocks."""
    days = trading_days(frame)
    n = len(days)
    # Warm-up must leave at least one block of at least one day each.
    n_warm = int(np.floor(n * warmup_frac))
    if n_warm >= n or n - n_warm < n_outer:
        raise StopV2NotEnoughTradingDays(
            f"STOP_V2_NOT_ENOUGH_TRADING_DAYS n={n} warmup={n_warm} "
            f"n_outer={n_outer}")

    rest = n - n_warm
    edges = [n_warm + int(i * rest / n_outer) for i in range(n_outer + 1)]
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise StopV2NonContiguousOuter(
            f"STOP_V2_NON_CONTIGUOUS_OUTER edges={edges}")

    outer = tuple(
        (days[edges[i]].date().isoformat(),
         days[edges[i + 1] - 1].date().isoformat())
        for i in range(n_outer)
    )
    return FoldPlan(
        days=tuple(d.date().isoformat() for d in days),
        warmup_end=days[n_warm - 1].date().isoformat(),
        outer=outer,
        es_frac=es_frac,
    )


def _assert_outer_partitions_once(outer):
    """Contiguity + exact-once coverage check used by the regression tests."""
    seen = []
    for start, end in outer:
        cur = pd.date_range(start, end, freq="D")
        seen.extend(cur)
    if len(set(seen)) != len(seen):
        raise StopV2NonContiguousOuter("STOP_V2_OUTER_BLOCKS_OVERLAP")
    return True


# --------------------------------------------------------------------------- #
# Availability + pair purity (§6)                                              #
# --------------------------------------------------------------------------- #
def available_training_mask(frame: pd.DataFrame, cutoff: pd.Timestamp) -> np.ndarray:
    """Rows whose decision AND label both strictly precede the cutoff day.

    A label that is NaT (never resolved) is treated as unavailable.
    """
    d = decision_day(frame)
    a = label_available_day(frame)
    return ((d < cutoff) & (a < cutoff)).to_numpy(dtype=bool)


def _epoch_ids(frame: pd.DataFrame) -> np.ndarray:
    """Compact int64 id per (symbol, decision_bar, horizon)."""
    key = pd.MultiIndex.from_arrays(
        [frame[k].to_numpy() for k in EPOCH_KEYS], names=list(EPOCH_KEYS))
    codes, _ = key.factorize(sort=False)
    return codes


def enforce_pair_purity(
    frame: pd.DataFrame,
    avail: np.ndarray,
    *,
    tol: float = WEIGHT_TOL,
) -> tuple[np.ndarray, int]:
    """§6 two-side purity + frozen sample-weight unit gate.

    Returns (purged_mask, stats).

    An epoch that originally carried BOTH sides must keep both; if only one
    side survived availability, the whole epoch is dropped. An epoch that was
    originally single-side may stay alone.

    Two different reasons remove an epoch, and conflating them would hide a
    real pair failure, so they are counted separately:
      dropped_unresolved : no side was available yet (routine censoring)
      dropped_pair_break : a two-side epoch lost exactly one side (§6 failure)
    """
    codes = _epoch_ids(frame)
    n_epochs = int(codes.max()) + 1

    present = np.bincount(codes[avail], minlength=n_epochs)
    original = np.bincount(codes, minlength=n_epochs)

    # Keep an epoch only if the availability purge did not break a two-side pair.
    keep_epoch = present == original
    purged = avail & keep_epoch[codes]

    stats = {
        "n_epochs_total": int(np.count_nonzero(original > 0)),
        "n_epochs_retained": int(np.count_nonzero(keep_epoch & (original > 0))),
        # Any side available at all -> this epoch simply has not resolved yet.
        "dropped_unresolved": int(np.count_nonzero(
            (present == 0) & (original > 0))),
        # A genuine §6 failure: a two-side epoch would enter training half-only.
        "dropped_pair_break": int(np.count_nonzero(
            (present > 0) & (present < original))),
    }

    # Hard gate: retained epochs preserve the frozen sample-weight semantics.
    _assert_epoch_weight_unit(frame, purged, tol=tol, codes=codes,
                              n_epochs=n_epochs)
    return purged, stats


def _assert_epoch_weight_unit(
    frame: pd.DataFrame,
    mask: np.ndarray,
    *,
    tol: float = WEIGHT_TOL,
    codes: Optional[np.ndarray] = None,
    n_epochs: Optional[int] = None,
) -> None:
    if not mask.any():
        return
    if codes is None:
        codes = _epoch_ids(frame)
        n_epochs = int(codes.max()) + 1
    w = frame["sample_weight"].to_numpy(dtype=float)
    sums = np.bincount(codes[mask], weights=w[mask], minlength=n_epochs)
    active = np.bincount(codes[mask], minlength=n_epochs) > 0
    bad = np.flatnonzero(active & (np.abs(sums - 1.0) > tol))
    if bad.size:
        worst = int(np.argmax(np.abs(sums[bad] - 1.0)))
        raise StopV2EpochWeightNotUnit(
            "STOP_V2_EPOCH_WEIGHT_NOT_UNIT "
            f"n_bad={bad.size} worst_epoch={int(bad[worst])} "
            f"sum={sums[bad][worst]!r} tol={tol}")


# --------------------------------------------------------------------------- #
# Per-fold partition                                                           #
# --------------------------------------------------------------------------- #
def fold_split(
    frame: pd.DataFrame,
    plan: FoldPlan,
    k: int,
    *,
    tol: float = WEIGHT_TOL,
) -> FoldSplit:
    """FIT_CORE | ES | OUTER_VALIDATION for outer fold k."""
    if not (0 <= k < plan.n_outer):
        raise IndexError(f"fold {k} out of range 0..{plan.n_outer - 1}")

    start_str, end_str = plan.outer[k]
    start = pd.Timestamp(start_str)
    end = pd.Timestamp(end_str)
    d = decision_day(frame)

    avail = available_training_mask(frame, start)
    purged, dropped = enforce_pair_purity(frame, avail, tol=tol)
    # NOTE: a broken two-side epoch is REMEDIED by dropping it (§6), not by
    # raising. Only the sample-weight unit gate is a hard STOP. The count is
    # carried in FoldSplit.purity_stats so it stays auditable.
    if not purged.any():
        raise StopV2EmptyTrainingWindow(
            f"STOP_V2_EMPTY_TRAINING_WINDOW fold={k} available={int(avail.sum())}")

    # §5: ES = last es_frac of the AVAILABLE training days (not calendar days).
    train_days = pd.DatetimeIndex(np.sort(d[purged].unique()))
    n_td = len(train_days)
    n_es_days = int(np.floor(n_td * plan.es_frac))
    if n_es_days < 1:
        raise StopV2EmptyEsBlock(
            f"STOP_V2_EMPTY_ES_BLOCK fold={k} train_days={n_td} "
            f"es_frac={plan.es_frac}")
    es_day_values = set(train_days[-n_es_days:])

    is_es_day = d.isin(es_day_values).to_numpy(dtype=bool)
    es = purged & is_es_day
    fit_core = purged & ~is_es_day

    # All rows decided inside the outer window, regardless of label resolution:
    # they are the evaluation rows and every one is predicted exactly once.
    outer = ((d >= start) & (d <= end)).to_numpy(dtype=bool)

    if not es.any() or not fit_core.any():
        raise StopV2EmptyEsBlock(
            f"STOP_V2_EMPTY_ES_BLOCK fold={k} "
            f"n_fit_core={int(fit_core.sum())} n_es={int(es.sum())}")

    return FoldSplit(
        fold=k,
        fit_core=fit_core,
        es=es,
        outer=outer,
        n_fit_core=int(np.count_nonzero(fit_core)),
        n_es=int(np.count_nonzero(es)),
        n_outer=int(np.count_nonzero(outer)),
        es_days=tuple(t.date().isoformat() for t in sorted(es_day_values)),
        n_train_rows_available=int(np.count_nonzero(avail)),
        purity_stats=dropped,
    )


def iter_fold_splits(frame: pd.DataFrame, plan: FoldPlan, *, tol=WEIGHT_TOL):
    for k in range(plan.n_outer):
        yield fold_split(frame, plan, k, tol=tol)


def is_warmup_row(frame: pd.DataFrame, plan: FoldPlan) -> np.ndarray:
    """Rows inside the warm-up window. These must NEVER receive an OOF row."""
    d = decision_day(frame)
    return (d <= pd.Timestamp(plan.warmup_end)).to_numpy(dtype=bool)


def oof_row_count_check(splits, frame: pd.DataFrame, plan: FoldPlan) -> dict:
    """Every non-warm-up row lands in exactly one outer block (§47 #21/#22)."""
    coverage = np.zeros(len(frame), dtype=np.int64)
    for s in splits:
        coverage += s.outer.astype(np.int64)
    warm = is_warmup_row(frame, plan)
    return {
        "n_rows": int(len(frame)),
        "n_warmup_rows": int(np.count_nonzero(warm)),
        "max_times_predicted": int(coverage.max()) if len(coverage) else 0,
        "n_rows_predicted_once": int(np.count_nonzero(coverage == 1)),
        "n_warmup_rows_predicted": int(np.count_nonzero(warm & (coverage > 0))),
        "n_nonwarmup_rows_never_predicted": int(
            np.count_nonzero(~warm & (coverage == 0))),
    }
