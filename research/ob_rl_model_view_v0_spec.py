#!/usr/bin/env python3

"""Model View V0 -- the human-readable model input for OB RL V0.

The research warehouse keeps all 232 action columns. Model View V0 is
a WHITELIST selected from it, so the first model is never handed a
232-column feature soup.

Design rules
------------
* Continuous values stay CONTINUOUS. Binning belongs to the human
  audit layer (``AUDIT_BUCKETS_V0``), never to the model input.
* ``META`` (time / identity) and ``WEIGHT`` are never model features.
* No reward / exit / outcome field may appear in the model features.
* A missing active OB is ``NO_OB`` for CATEGORICAL fields only.
  Numeric OB distance / fit stays NaN: ``NO_OB`` must never be
  conflated with ``OB distance = 0``.
* Not being in V0 does NOT mean "useless". It means "deferred"; every
  excluded family remains available for later ablation.
"""

from __future__ import annotations

import pandas as pd

from research.ob_rl_dataset_v0_spec import (
    VALIDATED_TFS,
    QUARANTINED_TFS,
)

MODEL_VIEW_VERSION = "ob_rl_model_view_v0"

COLUMN_ROLES = (
    "META",
    "WEIGHT",
    "MODEL_FEATURE",
    "ACTION",
    "REWARD",
)

# ------------------------------------------------------------
# Never model features
# ------------------------------------------------------------

MODEL_META_V0 = (
    "candidate_id",
    "candidate_group_id",
    "symbol",
    "touch_time",
    "touch_5m_bar_index",
    "trading_day",
)

MODEL_WEIGHT_V0 = ("decision_weight",)


# ------------------------------------------------------------
# Event
# ------------------------------------------------------------

EVENT_FEATURES_V0 = (
    "source_tf",
    "source_ob_structure",
    "source_ob_bias",
    # OB zone WIDTH (ATR-normalised) stays: the OB region itself is
    # the object under study. Absolute zone prices do not.
    "source_ob_width_atr5",
    "touch_behavior",
    "touch_ordinal",
)


# ------------------------------------------------------------
# SMC -- the core module
# ------------------------------------------------------------

def smc_features(tf: str) -> tuple[str, ...]:
    """12 fields per timeframe.

    target_fit / stop_structure already encode the full spatial
    information (fit x target_atr = forward distance), so the raw
    forward/backward ATR distances are deliberately NOT duplicated.
    """
    return (
        f"internal_bias_rel_{tf}",
        f"swing_bias_rel_{tf}",
        f"target_fit_internal_{tf}",
        f"target_fit_swing_{tf}",
        f"target_fit_ob_{tf}",
        f"stop_structure_internal_{tf}",
        f"stop_structure_swing_{tf}",
        f"stop_structure_ob_{tf}",
        f"forward_active_ob_bias_rel_{tf}",
        f"forward_active_ob_structure_class_{tf}",
        f"backward_active_ob_bias_{tf}",
        f"backward_active_ob_structure_class_{tf}",
    )


def dsa_features(tf: str) -> tuple[str, ...]:
    return (
        f"dsa_alignment_{tf}",
        f"dsa_vwap_dev_rel_{tf}",
    )


def momentum_features(tf: str) -> tuple[str, ...]:
    return (
        f"momentum_direction_{tf}",
        f"momentum_value_rel_{tf}",
        f"momentum_delta_rel_{tf}",
    )


QUANTILE_FEATURES_V0 = (
    "quant_state",
    "quant_width_percentile_train",
)

# Action is decomposed, not left as an opaque "FOLLOW_2.0R" string.
ACTION_FEATURES_V0 = (
    "trade_mode",
    "trade_direction",
    "target_R",
)

MODEL_FEATURES_V0 = (
    *EVENT_FEATURES_V0,
    *(c for tf in VALIDATED_TFS for c in smc_features(tf)),
    *(c for tf in VALIDATED_TFS for c in dsa_features(tf)),
    *(c for tf in VALIDATED_TFS for c in momentum_features(tf)),
    *QUANTILE_FEATURES_V0,
    *ACTION_FEATURES_V0,
)

# ------------------------------------------------------------
# Categorical vs continuous
# ------------------------------------------------------------

CATEGORICAL_FEATURES_V0 = (
    "source_tf",
    "source_ob_structure",
    "touch_behavior",
    "quant_state",
    "trade_mode",
    "trade_direction",
    *(
        f"momentum_direction_{tf}" for tf in VALIDATED_TFS
    ),
    *(
        f"forward_active_ob_structure_class_{tf}"
        for tf in VALIDATED_TFS
    ),
    *(
        f"backward_active_ob_structure_class_{tf}"
        for tf in VALIDATED_TFS
    ),
)

CONTINUOUS_FEATURES_V0 = tuple(
    c
    for c in MODEL_FEATURES_V0
    if c not in CATEGORICAL_FEATURES_V0
)

# ------------------------------------------------------------
# Missing-value contract
#
# NO_OB is a CATEGORICAL state. Numeric OB geometry stays NaN.
# ------------------------------------------------------------

CATEGORICAL_MISSING_V0 = {
    "forward_active_ob_structure_class": "NO_OB",
    "backward_active_ob_structure_class": "NO_OB",
}

NUMERIC_MISSING_STAYS_NAN_V0 = (
    "target_fit_ob",
    "stop_structure_ob",
    "forward_active_ob_bias_rel",
    "backward_active_ob_bias",
    "target_fit_internal",
    "target_fit_swing",
    "stop_structure_internal",
    "stop_structure_swing",
)

# ------------------------------------------------------------
# Deliberately excluded from V0 (still in the warehouse)
# ------------------------------------------------------------

WAREHOUSE_ONLY_FAMILIES = (
    "absolute pivot prices (internal/swing high/low level)",
    "absolute OB zone prices (zone_low / zone_high)",
    "all 70 dsa_raw_* fields",
    "last BOS / CHoCH structure type, bias and age",
    "group confluence flags",
    "quantile q10 / q50 / q90 / crossed / top30",
    "raw OHLC and volume / OI",
    "raw forward / backward ATR distances (recoverable from fit)",
    "future outcome diagnostics",
)

WAREHOUSE_ONLY_PREFIXES = (
    "dsa_raw_",
    "current_",
    "last_",
    "source_ob_zone_",
    "group_",
    "quant_q",
    "quant_crossed",
    "quant_top30",
    "quant_fold",
    "overlap_",
    "nearest_",
    "above_ob_",
    "below_ob_",
    "ob_above_atr_",
    "ob_below_atr_",
    "internal_high_",
    "internal_low_",
    "swing_high_",
    "swing_low_",
    "forward_internal_atr_",
    "backward_internal_atr_",
    "forward_swing_atr_",
    "backward_swing_atr_",
    "forward_active_ob_atr_",
    "backward_active_ob_atr_",
    "forward_active_ob_zone_",
    "backward_active_ob_zone_",
)


# ------------------------------------------------------------
# Audit-only binning (NEVER applied to the model input)
# ------------------------------------------------------------

def rr_fit_bin(x) -> str:
    """Structural RR fit bucket.

    fit == 1 is not a tuned threshold: it is the exact point where
    the target sits AT the structure. fit < 1 means the target is
    already beyond it.
    """
    if pd.isna(x):
        return "NO_LEVEL"
    if x < 1.0:
        return "BEYOND_STRUCTURE"
    return "FITS_BEFORE_STRUCTURE"


QUANTILE_AUDIT_BINS = (
    (0.00, 0.20, "Q1"),
    (0.20, 0.40, "Q2"),
    (0.40, 0.60, "Q3"),
    (0.60, 0.80, "Q4"),
    (0.80, 1.01, "Q5"),
)


def quantile_audit_bin(x) -> str:
    """Direct bins on the OOS percentile. Never re-ranked."""
    if pd.isna(x):
        return "UNKNOWN"
    v = float(x)
    for lo, hi, name in QUANTILE_AUDIT_BINS:
        if lo <= v < hi:
            return name
    return "UNKNOWN"


def collapse_for_descriptive_rank(
    df: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    """Collapse candidate x action -> candidate x trade_mode for
    descriptive (outcome-independent) layering.

    Each (candidate, trade_mode) appears three times (1.5R / 2.0R /
    2.5R). Ranking the expanded frame would triple-count every state.

    Collapsing by ``drop_duplicates`` alone is unsafe: target-dependent
    columns (target_fit_*, stop_structure_* differ by construction)
    would silently keep whichever RR happened to sort first. Columns
    must therefore be declared explicitly AND proven invariant across
    the target actions.
    """
    ids = ["candidate_id", "trade_mode"]

    x = df[df["trade_mode"].astype(str).ne("SKIP")].copy()

    missing = set(columns) - set(x.columns)
    if missing:
        raise RuntimeError(
            f"rank columns missing: {sorted(missing)}"
        )

    g = x.groupby(ids, observed=True, dropna=False)

    varying = []
    for c in columns:
        n = g[c].nunique(dropna=False)
        if (n > 1).any():
            varying.append(c)
    if varying:
        raise RuntimeError(
            "descriptive-rank column varies across target "
            f"actions: {varying}"
        )

    return (
        x[ids + list(columns)]
        .drop_duplicates(ids)
        .reset_index(drop=True)
    )


# ------------------------------------------------------------
# Audits
# ------------------------------------------------------------

MODEL_VIEW_ROLE_MAP = {
    **{c: "META" for c in MODEL_META_V0},
    **{c: "WEIGHT" for c in MODEL_WEIGHT_V0},
    **{c: "MODEL_FEATURE" for c in MODEL_FEATURES_V0},
    **{c: "ACTION" for c in ("action", "stop_atr", "target_atr")},
    **{
        c: "REWARD"
        for c in (
            "gross_R_h6",
            "gross_R_h12",
            "gross_R_h24",
            "exit_code_h6",
            "exit_code_h12",
            "exit_code_h24",
            "primary_reward_R",
            "reward_version",
        )
    },
}


def audit_model_features(action_df: pd.DataFrame) -> dict:
    if len(set(MODEL_FEATURES_V0)) != len(MODEL_FEATURES_V0):
        raise RuntimeError("duplicate entries in MODEL_FEATURES_V0")

    missing = set(MODEL_FEATURES_V0) - set(action_df.columns)
    if missing:
        raise RuntimeError(
            f"model features missing: {sorted(missing)}"
        )

    forbidden = set(MODEL_FEATURES_V0) & set(MODEL_META_V0)
    if forbidden:
        raise RuntimeError(
            f"metadata in model features: {sorted(forbidden)}"
        )

    weight_leak = set(MODEL_FEATURES_V0) & set(MODEL_WEIGHT_V0)
    if weight_leak:
        raise RuntimeError(
            f"weight column in model features: {sorted(weight_leak)}"
        )

    future = [
        c
        for c in MODEL_FEATURES_V0
        if c.startswith("gross_R")
        or c.startswith("exit_code")
        or "reward" in c
    ]
    if future:
        raise RuntimeError(f"reward leakage: {future}")

    quarantined = [
        c
        for c in MODEL_FEATURES_V0
        if any(f"_{tf}" in c or tf in c for tf in QUARANTINED_TFS)
    ]
    if quarantined:
        raise RuntimeError(
            f"quarantined TF in model features: {quarantined}"
        )

    excluded_hit = [
        c
        for c in MODEL_FEATURES_V0
        if c.startswith(WAREHOUSE_ONLY_PREFIXES)
    ]
    if excluded_hit:
        raise RuntimeError(
            f"warehouse-only column in model features: "
            f"{excluded_hit}"
        )

    assert_tf_coverage()

    return {
        "model_view_version": MODEL_VIEW_VERSION,
        "model_feature_count": len(MODEL_FEATURES_V0),
        "categorical_count": len(CATEGORICAL_FEATURES_V0),
        "continuous_count": len(CONTINUOUS_FEATURES_V0),
        "meta_leakage": 0,
        "reward_leakage": 0,
        "weight_leakage": 0,
        "quarantined_tf_features": 0,
        "warehouse_only_violations": 0,
    }


def assert_tf_coverage() -> None:
    """Guard against a future refactor silently dropping a TF."""
    for tf in VALIDATED_TFS:
        for c in (
            f"dsa_alignment_{tf}",
            f"dsa_vwap_dev_rel_{tf}",
            f"momentum_direction_{tf}",
            f"momentum_value_rel_{tf}",
            f"momentum_delta_rel_{tf}",
            f"target_fit_internal_{tf}",
            f"target_fit_swing_{tf}",
            f"target_fit_ob_{tf}",
            f"stop_structure_internal_{tf}",
            f"stop_structure_swing_{tf}",
            f"stop_structure_ob_{tf}",
            f"internal_bias_rel_{tf}",
            f"swing_bias_rel_{tf}",
            f"forward_active_ob_bias_rel_{tf}",
            f"forward_active_ob_structure_class_{tf}",
            f"backward_active_ob_bias_{tf}",
            f"backward_active_ob_structure_class_{tf}",
        ):
            if c not in MODEL_FEATURES_V0:
                raise RuntimeError(
                    f"TF coverage broken, missing {c}"
                )
