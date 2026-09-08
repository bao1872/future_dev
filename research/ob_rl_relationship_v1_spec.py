#!/usr/bin/env python3

"""RL-1A specification: Baseline + SMC x Risk/Reward discovery.

Scope of this round is deliberately narrow:

    A. What does the FOLLOW / FADE x 1.5R / 2R / 2.5R payoff surface
       look like BEFORE any environment is considered?
    B. Does an SMC state change the payoff distribution of the SAME
       action?
    C. Does SMC mainly decide DIRECTION, whether to TRADE AT ALL, or
       which RR to use?

DSA, Momentum, Quantile scans, interactions and any ML are explicitly
out of scope for RL-1A.

Outcome authority
-----------------
The action warehouse carries gross_R_h{6,12,24} and exit_code_h{...}
only. There is NO MFE / MAE in V0, so this round analyses reward
distribution plus target / stop / timeout structure and never pretends
otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.analyze_ob_candidate_v3_phase1 import (
    EXIT_GAP_STOP,
    EXIT_GAP_TARGET,
    EXIT_BOTH,
    EXIT_STOP,
    EXIT_TARGET,
    EXIT_TIMEOUT,
)
from research.ob_rl_dataset_v0_spec import (
    VALIDATED_TFS,
    QUARANTINED_TFS,
)

RL1_VERSION = "ob_rl_relationship_v1"

# ------------------------------------------------------------
# Horizons
# ------------------------------------------------------------

PRIMARY_HORIZON = 12

BASELINE_HORIZONS = (6, 12, 24)

# SMC discovery runs on H12 ONLY: scanning H6/H12/H24 against dozens
# of SMC states would triple the search space.
SMC_DISCOVERY_HORIZONS = (12,)

# ------------------------------------------------------------
# Conditioning views
# ------------------------------------------------------------

BASELINE_VIEWS = {
    # primary
    "by_source": ("source_tf", "source_ob_structure"),
    "by_symbol_source": (
        "symbol",
        "source_tf",
        "source_ob_structure",
    ),
    # secondary overview only
    "pooled": (),
}

SMC_VIEWS = BASELINE_VIEWS

PRIMARY_VIEW = "by_source"

# ------------------------------------------------------------
# State definitions
# ------------------------------------------------------------

BIAS_STATES = ("ALIGNED", "OPPOSED", "NEUTRAL", "UNKNOWN")
BIAS_CONTRAST = ("ALIGNED", "OPPOSED")

FIT_STATES = (
    "TARGET_BEYOND_STRUCTURE",
    "TARGET_AT_OR_BEFORE_STRUCTURE",
)
FIT_CONTRAST = (
    "TARGET_AT_OR_BEFORE_STRUCTURE",
    "TARGET_BEYOND_STRUCTURE",
)

STOP_STRUCTURE_STATES = (
    "STRUCTURE_INSIDE_STOP",
    "STRUCTURE_AT_OR_BEYOND_STOP",
)
STOP_STRUCTURE_CONTRAST = (
    "STRUCTURE_INSIDE_STOP",
    "STRUCTURE_AT_OR_BEYOND_STOP",
)

MISSING_LEVEL = "NO_LEVEL"
MISSING_OB = "NO_OB"

# Fixed BEFORE looking at reward; descriptive only.
FIT_BINS = (
    (-np.inf, 0.50, "<0.5"),
    (0.50, 1.00, "0.5-1.0"),
    (1.00, 1.50, "1.0-1.5"),
    (1.50, 2.00, "1.5-2.0"),
    (2.00, np.inf, ">=2.0"),
)

# ------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------

BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 20260908
BOOTSTRAP_CI = (0.025, 0.975)

MIN_VALID_ROWS_FOR_CI = 100
MIN_TRADING_DAYS_FOR_CI = 20

CI_OK = "OK"
CI_LOW_SUPPORT = "LOW_SUPPORT"

# ------------------------------------------------------------
# Guard: this round reports distributions, not recommendations
# ------------------------------------------------------------

FORBIDDEN_OUTPUT_TOKENS = (
    "best_action",
    "best_rr",
    "top_setup",
    "positive_edge",
    "recommended_rule",
    "recommendation",
    "buy",
    "trade_this",
)

# Families explicitly OUT of scope for RL-1A.
OUT_OF_SCOPE_FAMILIES = (
    "DSA",
    "MOMENTUM",
    "QUANTILE",
    "SMC_X_DSA",
    "SMC_X_MOMENTUM",
    "DSA_X_MOMENTUM",
    "ML",
    "FEATURE_IMPORTANCE",
    "RULE_MINING",
)

# ------------------------------------------------------------
# SMC registry
# ------------------------------------------------------------

def smc_registry() -> tuple[tuple, ...]:
    """(family, feature, tf, missing_label, contrast, state_kind)."""
    rows = []
    for tf in VALIDATED_TFS:
        for kind in ("internal", "swing"):
            rows.append(
                (
                    "SMC_BIAS",
                    f"{kind}_bias_rel_{tf}",
                    tf,
                    "UNKNOWN",
                    BIAS_CONTRAST,
                    "bias",
                )
            )
        for kind, missing in (
            ("internal", MISSING_LEVEL),
            ("swing", MISSING_LEVEL),
            ("ob", MISSING_OB),
        ):
            rows.append(
                (
                    "SMC_TARGET_FIT",
                    f"target_fit_{kind}_{tf}",
                    tf,
                    missing,
                    FIT_CONTRAST,
                    "target_fit",
                )
            )
            rows.append(
                (
                    "SMC_STOP_STRUCTURE",
                    f"stop_structure_{kind}_{tf}",
                    tf,
                    missing,
                    STOP_STRUCTURE_CONTRAST,
                    "stop_structure",
                )
            )
        rows.append(
            (
                "SMC_FORWARD_OB",
                f"forward_ob_state_{tf}",
                tf,
                MISSING_OB,
                None,
                "forward_ob",
            )
        )
    return tuple(rows)


def smc_fit_bin_registry() -> tuple[tuple, ...]:
    rows = []
    for tf in VALIDATED_TFS:
        for kind, missing in (
            ("internal", MISSING_LEVEL),
            ("swing", MISSING_LEVEL),
            ("ob", MISSING_OB),
        ):
            feature = f"target_fit_{kind}_{tf}"
            rows.append(
                (
                    "SMC_TARGET_FIT_BIN",
                    feature,
                    tf,
                    missing,
                )
            )
    return tuple(rows)


# ------------------------------------------------------------
# State mappers
# ------------------------------------------------------------

def bias_state(x) -> str:
    if pd.isna(x):
        return "UNKNOWN"
    v = float(x)
    if v > 0:
        return "ALIGNED"
    if v < 0:
        return "OPPOSED"
    return "NEUTRAL"


def target_fit_state(x, *, missing_label: str) -> str:
    if pd.isna(x):
        return missing_label
    v = float(x)
    if v < 1.0:
        return "TARGET_BEYOND_STRUCTURE"
    return "TARGET_AT_OR_BEFORE_STRUCTURE"


def stop_structure_state(x, *, missing_label: str) -> str:
    """stop_structure = backward structure / stop ATR (stop = 1 ATR).

    < 1  : structure sits between entry and stop  -> INSIDE STOP
    >= 1 : stop is hit before the structure       -> AT/BEYOND STOP
    """
    if pd.isna(x):
        return missing_label
    v = float(x)
    if v < 1.0:
        return "STRUCTURE_INSIDE_STOP"
    return "STRUCTURE_AT_OR_BEYOND_STOP"


def forward_ob_state(
    structure_class, bias_rel
) -> str:
    if pd.isna(structure_class):
        return MISSING_OB
    sc = str(structure_class).upper()
    if pd.isna(bias_rel):
        rel = "UNKNOWN"
    else:
        v = float(bias_rel)
        rel = (
            "ALIGNED"
            if v > 0
            else "OPPOSED"
            if v < 0
            else "NEUTRAL"
        )
    return f"{sc}_{rel}"


def fit_bin_label(x) -> str:
    if pd.isna(x):
        return "NO_LEVEL"
    v = float(x)
    for lo, hi, label in FIT_BINS:
        if lo <= v < hi:
            return label
    return "NO_LEVEL"


def derived_state_col(
    feature: str,
    state_kind: str,
) -> str:
    return f"__rl1_{state_kind}__{feature}"


def fit_bin_state_col(feature: str) -> str:
    return f"__rl1_target_fit_bin__{feature}"


# ------------------------------------------------------------
# Output schema (long format, appendable by later rounds)
# ------------------------------------------------------------

CELL_COLUMNS = (
    "analysis_family",
    "feature",
    "env_tf",
    "state_definition",
    "view",
    "horizon",
    "symbol",
    "source_tf",
    "source_ob_structure",
    "trade_mode",
    "target_R",
    "state",
    "rows_total",
    "valid_n",
    "valid_pct",
    "weight_valid",
    "trading_days",
    "mean_R",
    "p10",
    "p25",
    "median",
    "p75",
    "p90",
    "target_rate",
    "stop_rate",
    "timeout_rate",
    "both_rate",
)

CONTRAST_COLUMNS = (
    "analysis_family",
    "feature",
    "env_tf",
    "state_definition",
    "view",
    "horizon",
    "symbol",
    "source_tf",
    "source_ob_structure",
    "trade_mode",
    "target_R",
    "contrast",
    "state_high",
    "state_low",
    "n_high",
    "n_low",
    "delta_mean_R",
    "ci_low",
    "ci_high",
    "bootstrap_valid_reps",
    "ci_status",
)

PAIRED_COLUMNS = (
    "pair_kind",
    "view",
    "horizon",
    "symbol",
    "source_tf",
    "source_ob_structure",
    "trade_mode",
    "target_R",
    "rr_low",
    "rr_high",
    "n_pairs",
    "weight_total",
    "trading_days",
    "delta_mean_R",
    "delta_p10",
    "delta_p25",
    "delta_median",
    "delta_p75",
    "delta_p90",
    "ci_low",
    "ci_high",
    "bootstrap_valid_reps",
    "ci_status",
)


def assert_no_quarantined_tf(tf: str) -> None:
    if tf in QUARANTINED_TFS:
        raise RuntimeError(
            f"timeframe {tf!r} is quarantined"
        )
    if tf not in VALIDATED_TFS:
        raise RuntimeError(f"timeframe {tf!r} is not validated")