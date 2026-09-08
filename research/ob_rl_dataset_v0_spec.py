#!/usr/bin/env python3

"""OB RL Dataset V0 -- frozen specification.

Offline ``State x Action x Reward`` dataset built on top of the frozen
V3 candidate universe. It is NOT a live RL environment: it is a
static, auditable table from which per-state payoff distributions can
be computed directly.

Design boundaries
-----------------
* V0 keeps only the trading structure we have actually re-confirmed:
  OB event + SMC multi-TF + DSA multi-TF + Momentum multi-TF +
  Quantile opportunity + Action/RR + Reward. It does NOT try to be a
  complete environment warehouse (that is the Atlas' job).
* The STATE table is ABSOLUTE market state: above / below, high / low,
  bull / bear. It must NOT contain forward / backward / aligned /
  opposed, because those depend on the action.
* Action-relative encodings live only in the ACTION table.
* Reward is GROSS. No commission, slippage or tick model exists yet,
  so no "net R" may be produced.
* Reward is computed by REUSING the already-validated Phase-1
  first-hit simulator. No second simulator may be written.

Authority
---------
Levels vocabulary is FROZEN from the real committed V3 data (all four
symbols, 3933 level chunks): exactly 10 distinct
(object_type, structure_class) pairs. Active OB membership is an
exact-membership test against that frozen set, never a fuzzy
``str.contains("active")``.
"""

from __future__ import annotations

DATASET_VERSION = "ob_rl_dataset_v0"

BASELINE_SHA = "0b0caad7ddba837f4d837c1fbdd990abe28f6167"

# ------------------------------------------------------------
# Timeframes
# ------------------------------------------------------------

VALIDATED_TFS = ("5m", "15m", "1h")

QUARANTINED_TFS = ("4h",)

# ------------------------------------------------------------
# Action space (frozen for V0)
# ------------------------------------------------------------

STOP_ATR = 1.0

TARGET_R = (1.5, 2.0, 2.5)

TRADE_MODES = ("FOLLOW", "FADE")

ACTIONS = (
    "SKIP",
    "FOLLOW_1.5R",
    "FOLLOW_2.0R",
    "FOLLOW_2.5R",
    "FADE_1.5R",
    "FADE_2.0R",
    "FADE_2.5R",
)

PRIMARY_HORIZON = 12

DIAGNOSTIC_HORIZONS = (6, 12, 24)

SAME_BAR_POLICY = "conservative"

REWARD_VERSION = "GROSS_R_V0"

# SKIP is not produced by the Phase-1 simulator, so it carries its
# own explicit code. It is deliberately OUTSIDE the canonical
# EXIT_* range (0..7) to avoid ever being mistaken for a real exit.
SKIP_EXIT_CODE = -1

# ------------------------------------------------------------
# OB event fields (carried from V3 candidates)
# ------------------------------------------------------------

EVENT_FIELDS = (
    "candidate_id",
    "candidate_group_id",
    "symbol",
    "source_tf",
    "source_ob_internal",
    "source_ob_structure",
    "source_ob_bias",
    "source_ob_zone_low",
    "source_ob_zone_high",
    "source_ob_width_atr5",
    "touch_ordinal",
    "is_first_touch",
    "touch_intrabar_far_edge_breach",
    "touch_close_beyond_far_edge",
    "touch_reclaimed_by_close",
    "group_candidate_count",
    "group_has_5m",
    "group_has_15m",
    "group_has_1h",
)

TOUCH_BEHAVIOR_VALUES = (
    "NO_BREACH",
    "BREACH_RECLAIM",
    "CLOSE_BEYOND",
)

# ------------------------------------------------------------
# SMC V0
# ------------------------------------------------------------

SMC_BIAS_FIELDS = (
    "swing_bias",
    "internal_bias",
)

SMC_EVENT_FIELDS = (
    "last_swing_structure_type",
    "last_swing_structure_bias",
    "last_swing_structure_age",
    "last_internal_structure_type",
    "last_internal_structure_bias",
    "last_internal_structure_age",
)

SMC_PIVOT_FIELDS = (
    "current_internal_high_level",
    "current_internal_high_distance_pct",
    "current_internal_low_level",
    "current_internal_low_distance_pct",
    "current_swing_high_level",
    "current_swing_high_distance_pct",
    "current_swing_low_level",
    "current_swing_low_distance_pct",
)

# Distance-bearing SMC pivot fields -> exec-ATR names.
SMC_PIVOT_DISTANCE_MAP = {
    "current_internal_high_distance_pct": "internal_high_atr",
    "current_internal_low_distance_pct": "internal_low_atr",
    "current_swing_high_distance_pct": "swing_high_atr",
    "current_swing_low_distance_pct": "swing_low_atr",
}

# ------------------------------------------------------------
# Frozen levels vocabulary (OBSERVED on real V3 data)
# ------------------------------------------------------------

FROZEN_LEVEL_VOCABULARY = (
    ("active_bull_internal_ob", "internal"),
    ("active_bear_internal_ob", "internal"),
    ("active_bull_swing_ob", "swing"),
    ("active_bear_swing_ob", "swing"),
    ("current_internal_high", "internal"),
    ("current_internal_low", "internal"),
    ("current_swing_high", "swing"),
    ("current_swing_low", "swing"),
    ("EQH", "equal"),
    ("EQL", "equal"),
)

# Exact membership. A fuzzy substring test is NOT the authority.
ACTIVE_OB_TYPES = frozenset(
    {
        "active_bull_internal_ob",
        "active_bear_internal_ob",
        "active_bull_swing_ob",
        "active_bear_swing_ob",
    }
)

LEVELS_REQUIRED_COLUMNS = (
    "event_id",
    "symbol",
    "timeframe",
    "object_type",
    "structure_class",
    "bias",
    "zone_low",
    "zone_high",
    "relation",
    "distance_pct",
)

LEVEL_RELATIONS = ("above", "below", "overlap")

# ------------------------------------------------------------
# DSA V0 (two core states only)
# ------------------------------------------------------------

DSA_FIELDS = (
    "dsa_direction",
    "dsa_raw_dsa_vwap_dev_pct",
)

DSA_RENAMES = {
    "dsa_direction": "dsa_direction",
    "dsa_raw_dsa_vwap_dev_pct": "dsa_vwap_dev_pct",
}

# ------------------------------------------------------------
# Momentum V0 (canonical, never reimplemented)
# ------------------------------------------------------------

MOMENTUM_JOIN_FIELD = "bar_index"

MOMENTUM_FIELDS = (
    "momentum_direction",
    "sqzmom_val",
    "sqzmom_delta",
)

MOMENTUM_CANONICAL_STRING_FIELDS = ("momentum_direction",)

MOMENTUM_DIRECTION_VALUES = (
    "expanding",
    "contracting",
    "flat",
)

# ------------------------------------------------------------
# Quantile V0
# ------------------------------------------------------------

QUANTILE_FIELDS = (
    "quant_width",
    "quant_width_percentile_train",
    "quant_top30_train",
    "quant_crossed",
)

QUANT_STATE_VALUES = ("LOW", "MID", "HIGH", "UNKNOWN")

QUANT_LOW_MAX = 0.30
QUANT_HIGH_MIN = 0.70

# ------------------------------------------------------------
# Causality contract
# ------------------------------------------------------------

FORBIDDEN_STATE_PREFIXES = (
    "entry_",
    "future_",
    "reward",
    "gross_R",
    "exit_code",
)

# Columns that exist on the raw candidates table but must never be
# promoted into the state feature list.
FORBIDDEN_STATE_COLUMNS = frozenset(
    {
        "entry_5m_bar_index",
        "entry_time",
        "entry_next_5m_open",
        "entry_delay_minutes",
        "touch_bar_start_time",
        "touch_time",
        "source_confirmed_available_time",
        "quant_state_decision_time",
        "quant_state_age_minutes",
        "quant_fold",
        "quant_q10",
        "quant_q50",
        "quant_q90",
    }
)


def assert_validated_tf(tf: str) -> None:
    if tf in QUARANTINED_TFS:
        raise RuntimeError(
            f"timeframe {tf!r} is quarantined: no validated 4h "
            "bar authority exists (V3 4h buckets are wall-clock "
            "epoch buckets holding only 1-3 x 1h components)"
        )
    if tf not in VALIDATED_TFS:
        raise RuntimeError(f"timeframe {tf!r} is not validated")


def is_quarantined(tf: str) -> bool:
    return tf in QUARANTINED_TFS


def parse_action(action: str) -> tuple[str, float]:
    """Parse 'FOLLOW_2.0R' -> ('FOLLOW', 2.0). 'SKIP' -> skip."""
    if action == "SKIP":
        return "SKIP", 0.0
    mode, _, rr = action.partition("_")
    if mode not in TRADE_MODES:
        raise RuntimeError(f"unknown trade mode in {action!r}")
    if not rr.endswith("R"):
        raise RuntimeError(f"malformed action {action!r}")
    target_r = float(rr[:-1])
    if target_r not in TARGET_R:
        raise RuntimeError(
            f"action {action!r} target {target_r} not in {TARGET_R}"
        )
    return mode, target_r


def assert_action_space(actions) -> None:
    if tuple(actions) != ACTIONS:
        raise RuntimeError(
            "action space drift: "
            f"{tuple(actions)} != {ACTIONS}"
        )


def assert_level_vocabulary(levels) -> dict:
    """Exact vocabulary test against the frozen observed set.

    An unmapped (object_type, structure_class) pair is a STOP, never a
    silent 'other' bucket.
    """
    pairs = (
        levels[["object_type", "structure_class"]]
        .fillna("")
        .astype(str)
        .drop_duplicates()
    )
    observed = {
        (ot, sc) for ot, sc in pairs.itertuples(index=False)
    }
    frozen = set(FROZEN_LEVEL_VOCABULARY)

    unknown = observed - frozen
    if unknown:
        raise RuntimeError(
            "levels vocabulary drift: unknown pairs "
            f"{sorted(unknown)}"
        )

    return {
        "observed_pairs": len(observed),
        "frozen_pairs": len(frozen),
        "unknown_pairs": 0,
        "active_ob_types": sorted(
            {ot for ot, _ in observed} & ACTIVE_OB_TYPES
        ),
    }


def assert_action_cardinality(
    n_candidates: int, action_df
) -> None:
    expected = n_candidates * len(ACTIONS)
    if len(action_df) != expected:
        raise RuntimeError(
            "action cardinality mismatch: "
            f"{len(action_df)} != {expected}"
        )
    if action_df.duplicated(["candidate_id", "action"]).any():
        raise RuntimeError("duplicate candidate-action row")
