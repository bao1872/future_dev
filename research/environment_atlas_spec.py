#!/usr/bin/env python3

"""Environment Atlas V1 - fixed ontology and experiment specification.

This module is SPEC ONLY. It holds no data and performs no I/O beyond
constant definitions. It exists so that the environment definition is
frozen BEFORE any payoff distribution is inspected.

Design rule
-----------
The OB touch is the CANDIDATE GENERATOR. Everything else describes the
environment in which that candidate occurred. Environment and outcome
are built and stored separately so future information can never leak
into a feature table.

Environment families
--------------------
    STRUCTURE   SMC bias / BOS-CHoCH / current pivots
    LEVELS      pressure + support identity, distance, density
    DSA         running regime (multi-TF joint state, full raw fields)
    MOMENTUM    SQZMOM / volatility phase / expansion state
    VOLATILITY  ATR (execution and TF-local)
    QUANTILE    opportunity distribution state (OOS, PIT)

Event families (NOT environment, NOT a ranking)
-----------------------------------------------
    OB_TOUCH        source timeframe / internal-swing / ordinal
    CONFLUENCE      multi-TF co-occurrence on the same 5m decision bar
    TOUCH_BEHAVIOR  NO_BREACH / BREACH_RECLAIM / CLOSE_BEYOND
"""

from __future__ import annotations

ATLAS_VERSION = "ob_environment_atlas_v1"

ATLAS_BASELINE_SHA = "7541962fb36fc9586b8cecfe0d80eb9fcfa79ef0"

# ------------------------------------------------------------
# Ontology
# ------------------------------------------------------------

ENVIRONMENT_FAMILIES = (
    "STRUCTURE",
    "LEVELS",
    "DSA",
    "MOMENTUM",
    "VOLATILITY",
    "QUANTILE",
)

EVENT_FAMILY = (
    "OB_TOUCH",
    "CONFLUENCE",
    "TOUCH_BEHAVIOR",
)

VALIDATED_TFS = ("5m", "15m", "1h")

QUARANTINED_TFS = ("4h",)

# ------------------------------------------------------------
# Outcome horizons (5m bars)
# ------------------------------------------------------------

HORIZONS = (
    3,   # 15 minutes
    6,   # 30 minutes
    12,  # 60 minutes
    24,  # 120 minutes
)

HORIZON_LABELS = {
    3: "15m",
    6: "30m",
    12: "60m",
    24: "120m",
}

# ------------------------------------------------------------
# Level map distance bins (execution ATR)
# ------------------------------------------------------------

DIST_BINS = (0.5, 1.0, 2.0, 3.0)


def density_field(side: str, bin_atr: float) -> str:
    """Level-map column for 'objects within N ATR' on one side."""
    return f"{side}_count_within_{bin_atr:g}atr"

# ------------------------------------------------------------
# Cell sample-size gates.
#
# Gates are evaluated PER HORIZON on the horizon's VALID sample, never
# on the cell total (H24 strict-continuity coverage is ~19%).
# ------------------------------------------------------------

MIN_RAW_N = 50
MIN_WEIGHTED_N = 30
MIN_TRADING_DAYS = 20

ROBUST_RAW_N = 100
ROBUST_TRADING_DAYS = 40

STATUS_INSUFFICIENT = "INSUFFICIENT"
STATUS_EXPLORATORY = "EXPLORATORY"
STATUS_ROBUST = "ROBUST"

# ------------------------------------------------------------
# Descriptive environment layering.
#
# Continuous environment variables are layered by their OWN
# distribution (quintiles), never by a hand-picked threshold. This is
# a DESCRIPTIVE split made before any payoff is inspected; it is not a
# strategy parameter and must never be tuned against an outcome.
# ------------------------------------------------------------

DESCRIPTIVE_Q_BINS = (
    float("-inf"),
    0.20,
    0.40,
    0.60,
    0.80,
    float("inf"),
)

DESCRIPTIVE_Q_LABELS = ("Q1", "Q2", "Q3", "Q4", "Q5")

# Ranked WITHIN (symbol, source_tf, trade_mode) so that a quintile
# always means "high for this generator and this trade mode".
DESCRIPTIVE_Q_GROUP = ("symbol", "source_tf", "trade_mode")

# Projection kinds:
#   categorical  carried verbatim (string / bool state)
#   raw_q        numeric -> descriptive quintile
#   signed       numeric x trade_direction -> descriptive quintile
#   signed_cat   numeric x trade_direction -> kept as-is (relative dir)
PROJECTION_KINDS = (
    "categorical",
    "raw_q",
    "signed",
    "signed_cat",
)

# ------------------------------------------------------------
# Canonical Momentum (SQZMOM).
#
# Source owner: panji_indicators.compute_sqzmom_lb +
# build_momentum_history, reached via
# research.indicator_adapter.compute_smc_momentum_bundle.
#
# momentum_direction / momentum_change / volatility_phase are CANONICAL
# STRINGS. They are carried verbatim and are NEVER cast to float and
# never multiplied by a trade direction.
# ------------------------------------------------------------

MOMENTUM_STATE_FIELDS = (
    "volatility_phase",
    "momentum_direction",
    "momentum_change",
    "sqzmom_delta",
    "sqzmom_val",
    "squeeze_period_volume_mean",
    "release_volume_ratio",
)

MOMENTUM_CANONICAL_STRING_FIELDS = (
    "volatility_phase",
    "momentum_direction",
    "momentum_change",
)

MOMENTUM_DIRECTION_VALUES = (
    "expanding",
    "contracting",
    "flat",
)

MOMENTUM_CHANGE_VALUES = (
    "enhancing",
    "weakening",
    "flat",
)

# Explicit numeric derivation used for directional combination:
#   sqzmom_val > 0 -> +1 (expanding)
#   sqzmom_val < 0 -> -1 (contracting)
#   sqzmom_val = 0 ->  0 (flat)
MOMENTUM_SIGN_FIELD = "momentum_sqzmom_sign"

# attach_momentum() carries canonical states with a "momentum_" prefix,
# so the canonical string columns on candidate_env_tf are these names.
# They are strings and must never be float-cast.
MOMENTUM_CARRIED_STRING_FIELDS = (
    "momentum_volatility_phase",
    "momentum_momentum_direction",
    "momentum_momentum_change",
)

MOMENTUM_JOIN_FIELD = "bar_index"

# ------------------------------------------------------------
# DSA raw environment - EXACT frozen schema.
#
# This is the full dsa_raw_* column set committed by V3 (read from
# research/analysis_data/ob_candidate_universe_v3/schema.json).
# It is NOT a hand-picked subset. The builder asserts exact parity at
# runtime so the environment mother table cannot silently drop fields.
# ------------------------------------------------------------

EXPECTED_DSA_RAW_FIELDS = (
    "dsa_raw_avg_amount_20d",
    "dsa_raw_change_pct",
    "dsa_raw_cross_down_count",
    "dsa_raw_cross_up_count",
    "dsa_raw_current_segment_amount_mean",
    "dsa_raw_current_segment_amount_sum",
    "dsa_raw_current_segment_volume_mean",
    "dsa_raw_current_segment_volume_sum",
    "dsa_raw_current_vs_prev_amount_mean_ratio",
    "dsa_raw_current_vs_prev_amount_ratio",
    "dsa_raw_current_vs_prev_volume_mean_ratio",
    "dsa_raw_current_vs_prev_volume_ratio",
    "dsa_raw_dsa_dir_bars",
    "dsa_raw_dsa_vwap",
    "dsa_raw_dsa_vwap_dev_pct",
    "dsa_raw_last_cross_down_date",
    "dsa_raw_last_cross_down_price",
    "dsa_raw_last_cross_up_date",
    "dsa_raw_last_cross_up_price",
    "dsa_raw_offset_mean",
    "dsa_raw_offset_percentile",
    "dsa_raw_offset_rate",
    "dsa_raw_offset_std",
    "dsa_raw_offset_variance_rate",
    "dsa_raw_prev_segment_amount_mean",
    "dsa_raw_prev_segment_amount_sum",
    "dsa_raw_prev_segment_bars",
    "dsa_raw_prev_segment_change_pct",
    "dsa_raw_prev_segment_direction",
    "dsa_raw_prev_segment_end_bar_index",
    "dsa_raw_prev_segment_end_price",
    "dsa_raw_prev_segment_end_time",
    "dsa_raw_prev_segment_id",
    "dsa_raw_prev_segment_slope",
    "dsa_raw_prev_segment_start_bar_index",
    "dsa_raw_prev_segment_start_price",
    "dsa_raw_prev_segment_start_time",
    "dsa_raw_prev_segment_volume_mean",
    "dsa_raw_prev_segment_volume_sum",
    "dsa_raw_regime_strength",
    "dsa_raw_regime_value",
    "dsa_raw_rope_cross_down_count",
    "dsa_raw_rope_cross_down_date",
    "dsa_raw_rope_cross_down_price",
    "dsa_raw_rope_cross_up_count",
    "dsa_raw_rope_cross_up_date",
    "dsa_raw_rope_cross_up_price",
    "dsa_raw_rope_dir0_pct",
    "dsa_raw_rope_dir1_pct",
    "dsa_raw_rope_dir_neg1_pct",
    "dsa_raw_segment_bars",
    "dsa_raw_segment_change_pct",
    "dsa_raw_segment_direction",
    "dsa_raw_segment_end_bar_index",
    "dsa_raw_segment_end_price",
    "dsa_raw_segment_end_time",
    "dsa_raw_segment_id",
    "dsa_raw_segment_slope",
    "dsa_raw_segment_start_bar_index",
    "dsa_raw_segment_start_price",
    "dsa_raw_segment_start_time",
    "dsa_raw_touch_rope",
    "dsa_raw_touch_vwap",
    "dsa_raw_trend_transition",
    "dsa_raw_vol_zscore",
    "dsa_raw_vwap_ret_10",
    "dsa_raw_vwap_ret_20",
    "dsa_raw_vwap_ret_5",
    "dsa_raw_vwap_ret_avg",
    "dsa_raw_vwap_ret_total",
)

# Carried into candidate_env_tf = direction + ALL raw fields.
DSA_ENV_FIELDS = ("dsa_direction",) + EXPECTED_DSA_RAW_FIELDS

# ------------------------------------------------------------
# Structural environment fields (per timeframe)
# ------------------------------------------------------------

STRUCTURE_FIELDS = (
    "internal_bias",
    "swing_bias",
    "last_internal_structure_type",
    "last_internal_structure_bias",
    "last_internal_structure_level",
    "last_internal_structure_age",
    "last_swing_structure_type",
    "last_swing_structure_bias",
    "last_swing_structure_level",
    "last_swing_structure_age",
)

CURRENT_PIVOT_FIELDS = (
    "current_internal_high_level",
    "current_internal_high_age",
    "current_internal_high_relation",
    "current_internal_high_distance_pct",
    "current_internal_high_crossed",
    "current_internal_high_crossed_age",
    "current_internal_low_level",
    "current_internal_low_age",
    "current_internal_low_relation",
    "current_internal_low_distance_pct",
    "current_internal_low_crossed",
    "current_internal_low_crossed_age",
    "current_swing_high_level",
    "current_swing_high_age",
    "current_swing_high_relation",
    "current_swing_high_distance_pct",
    "current_swing_high_crossed",
    "current_swing_high_crossed_age",
    "current_swing_low_level",
    "current_swing_low_age",
    "current_swing_low_relation",
    "current_swing_low_distance_pct",
    "current_swing_low_crossed",
    "current_swing_low_crossed_age",
)

OVERLAP_FIELDS = (
    "overlap_object_count",
    "overlap_active_ob_count",
    "overlap_bull_ob_count",
    "overlap_bear_ob_count",
    "overlap_internal_count",
    "overlap_swing_count",
    "overlap_types",
)

NEAREST_FIELDS = (
    "nearest_above_type",
    "nearest_above_structure_class",
    "nearest_above_bias",
    "nearest_above_age_bars",
    "nearest_above_zone_low",
    "nearest_above_zone_high",
    "nearest_above_object_price_center",
    "nearest_above_distance_pct",
    "nearest_below_type",
    "nearest_below_structure_class",
    "nearest_below_bias",
    "nearest_below_age_bars",
    "nearest_below_zone_low",
    "nearest_below_zone_high",
    "nearest_below_object_price_center",
    "nearest_below_distance_pct",
)

# ------------------------------------------------------------
# Levels schema (as committed by V3).
#
# The primary key is `event_id`, NOT `candidate_id`; the mapping is
# PROVEN at runtime. `relation` and `distance_pct` are the committed
# touch-time zone-edge geometry and are used as-is: distances are NOT
# recomputed from object center and NOT measured from the next open.
# ------------------------------------------------------------

LEVELS_COLUMNS = (
    "event_id",
    "symbol",
    "trigger_time",
    "timeframe",
    "object_type",
    "structure_class",
    "bias",
    "object_price_center",
    "source_index",
    "confirmed_index",
    "age_bars",
    "zone_low",
    "zone_high",
    "canonical_bar_low",
    "canonical_bar_high",
    "canonical_endpoints_swapped",
    "relation",
    "distance_pct",
)

LEVELS_RELATIONS = ("above", "below", "overlap")

# Type identity for pressure/support. There is deliberately NO silent
# "other" bucket: an unmapped object type must STOP the build.
LEVEL_TYPE_BUCKETS = (
    "active_ob",
    "internal_pivot",
    "swing_pivot",
    "equal_level",
)

# ------------------------------------------------------------
# Pre-registered ENVIRONMENT PROJECTIONS.
#
# The mother table carries the complete raw environment. The Atlas
# studies a PRE-REGISTERED projection of it, fixed BEFORE any payoff
# is inspected. This is deliberately NOT a cartesian product of all
# 70 dsa_raw_* fields, and deliberately NOT reduced to
# "direction + sign + nearest room".
#
# Each projection is (facet_suffix, env_column_suffix, kind).
# ------------------------------------------------------------

ACTIVE_OB_COUNT_FIELDS = (
    "active_bull_internal_ob_count",
    "active_bear_internal_ob_count",
    "active_bull_swing_ob_count",
    "active_bear_swing_ob_count",
)

# --- DSA running-state projections ---------------------------

DSA_PIVOT_FIELDS = (
    "dsa_raw_regime_strength",
    "dsa_raw_dsa_dir_bars",
    "dsa_raw_trend_transition",
    "dsa_raw_offset_percentile",
    "dsa_raw_dsa_vwap_dev_pct",
    "dsa_raw_segment_direction",
    "dsa_raw_segment_slope",
    "dsa_raw_segment_bars",
    "dsa_raw_touch_rope",
    "dsa_raw_touch_vwap",
    "dsa_raw_rope_dir1_pct",
    "dsa_raw_rope_dir0_pct",
    "dsa_raw_rope_dir_neg1_pct",
)

DSA_PROJECTIONS = (
    ("regime_strength", "dsa_raw_regime_strength", "signed"),
    ("regime_age", "dsa_raw_dsa_dir_bars", "raw_q"),
    ("transition", "dsa_raw_trend_transition", "categorical"),
    ("offset_percentile", "dsa_raw_offset_percentile", "raw_q"),
    ("vwap_deviation", "dsa_raw_dsa_vwap_dev_pct", "signed"),
    (
        "segment_direction",
        "dsa_raw_segment_direction",
        "signed_cat",
    ),
    ("segment_slope", "dsa_raw_segment_slope", "signed"),
    ("segment_bars", "dsa_raw_segment_bars", "raw_q"),
    ("touch_rope", "dsa_raw_touch_rope", "categorical"),
    ("touch_vwap", "dsa_raw_touch_vwap", "categorical"),
    ("rope_dir1_pct", "dsa_raw_rope_dir1_pct", "raw_q"),
    ("rope_dir0_pct", "dsa_raw_rope_dir0_pct", "raw_q"),
    ("rope_dir_neg1_pct", "dsa_raw_rope_dir_neg1_pct", "raw_q"),
)

# --- Momentum projections -------------------------------------
#
# Canonical STRINGS (volatility_phase / momentum_direction /
# momentum_change) are studied as-is and are never multiplied.
# ------------------------------------------------------------

MOMENTUM_PIVOT_FIELDS = (
    "momentum_sqzmom_delta",
    "momentum_release_volume_ratio",
    "momentum_squeeze_period_volume_mean",
)

MOMENTUM_TF_PROJECTIONS = (
    ("sqzmom_delta", "momentum_sqzmom_delta", "signed"),
    (
        "release_volume_ratio",
        "momentum_release_volume_ratio",
        "raw_q",
    ),
    (
        "squeeze_volume_mean",
        "momentum_squeeze_period_volume_mean",
        "raw_q",
    ),
)

# --- Structure projections ------------------------------------

STRUCTURE_PIVOT_FIELDS = (
    "last_internal_structure_type",
    "last_internal_structure_bias",
    "last_internal_structure_age",
    "last_swing_structure_type",
    "last_swing_structure_bias",
    "last_swing_structure_age",
)

STRUCTURE_PROJECTIONS = (
    (
        "internal_structure_type",
        "last_internal_structure_type",
        "categorical",
    ),
    (
        "internal_structure_bias",
        "last_internal_structure_bias",
        "signed_cat",
    ),
    (
        "internal_structure_age",
        "last_internal_structure_age",
        "raw_q",
    ),
    (
        "swing_structure_type",
        "last_swing_structure_type",
        "categorical",
    ),
    (
        "swing_structure_bias",
        "last_swing_structure_bias",
        "signed_cat",
    ),
    (
        "swing_structure_age",
        "last_swing_structure_age",
        "raw_q",
    ),
)

# --- Volatility projections -----------------------------------
#
# atr_pct = atr14 / close, derived in the builder so that ATR is
# comparable across symbols and price levels.
# ------------------------------------------------------------

VOLATILITY_ENV_FIELD = "atr_pct"

VOLATILITY_PROJECTIONS = (
    ("atr_pct", VOLATILITY_ENV_FIELD, "raw_q"),
)

# --- Quantile projections -------------------------------------

QUANTILE_FACET_FIELDS = (
    "quant_bin",
    "quant_width_percentile_train",
    "quant_top30_train",
    "quant_crossed",
)

# Full environment columns pivoted candidate x timeframe.
ENV_PIVOT_FIELDS = (
    (
        "dsa_direction",
        "internal_bias",
        "swing_bias",
        "momentum_sqzmom_sign",
        "momentum_momentum_direction",
        "momentum_momentum_change",
        "momentum_volatility_phase",
        VOLATILITY_ENV_FIELD,
    )
    + DSA_PIVOT_FIELDS
    + MOMENTUM_PIVOT_FIELDS
    + STRUCTURE_PIVOT_FIELDS
    + ACTIVE_OB_COUNT_FIELDS
)

# ------------------------------------------------------------
# Pre-registered Level-2 interactions.
#
# A full cartesian product over all families is FORBIDDEN.
# LEVELS pairs are expanded per timeframe (5m / 15m / 1h) so TF
# identity is never collapsed into a single min().
# ------------------------------------------------------------

LEVEL2_INTERACTIONS = (
    ("DSA", "MOMENTUM"),
    ("DSA", "LEVELS"),
    ("MOMENTUM", "LEVELS"),
    ("SMC", "DSA"),
    ("SMC", "MOMENTUM"),
    ("TOUCH", "DSA"),
    ("TOUCH", "MOMENTUM"),
    ("TOUCH", "LEVELS"),
    ("QUANTILE", "MOMENTUM"),
    ("QUANTILE", "LEVELS"),
)

# Pairs whose LEVELS side is expanded per timeframe.
LEVEL2_TF_EXPANDED = frozenset(
    {("DSA", "LEVELS"), ("MOMENTUM", "LEVELS"),
     ("TOUCH", "LEVELS"), ("QUANTILE", "LEVELS")}
)

# ------------------------------------------------------------
# Level-2 state axes -- EXPLICIT, never guessed.
#
# NO column name may be derived by string concatenation such as
# f"{a.lower()}_joint_rel": the four LEVELS pairs use four different
# real column names, and a guessed name produced silent no-op
# facets. Every registered pair MUST appear in exactly one map.
# ------------------------------------------------------------

LEVEL2_STATE_AXIS = {
    ("DSA", "MOMENTUM"): (
        "dsa_joint_rel",
        "momentum_rel_joint",
    ),
    ("SMC", "DSA"): (
        "smc_internal_joint_rel",
        "dsa_joint_rel",
    ),
    ("SMC", "MOMENTUM"): (
        "smc_internal_joint_rel",
        "momentum_rel_joint",
    ),
    ("TOUCH", "DSA"): (
        "touch_behavior",
        "dsa_joint_rel",
    ),
    ("TOUCH", "MOMENTUM"): (
        "touch_behavior",
        "momentum_rel_joint",
    ),
    ("QUANTILE", "MOMENTUM"): (
        "quant_bin",
        "momentum_rel_joint",
    ),
}

# LEVELS side of a TF-expanded pair uses the pair's OWN state column.
LEVEL2_LEVEL_STATE_AXIS = {
    ("DSA", "LEVELS"): "dsa_joint_rel",
    ("MOMENTUM", "LEVELS"): "momentum_rel_joint",
    ("TOUCH", "LEVELS"): "touch_behavior",
    ("QUANTILE", "LEVELS"): "quant_bin",
}


def assert_level2_registry() -> None:
    """The registry IS the test surface. Registry and runner cannot
    silently drift apart, because the runner has no fallback."""
    declared = set(LEVEL2_INTERACTIONS)
    plain = set(LEVEL2_STATE_AXIS)
    expanded = set(LEVEL2_LEVEL_STATE_AXIS)

    if LEVEL2_TF_EXPANDED != expanded:
        raise RuntimeError(
            "LEVEL2_TF_EXPANDED != LEVEL2_LEVEL_STATE_AXIS keys: "
            f"{sorted(LEVEL2_TF_EXPANDED ^ expanded)}"
        )
    if expanded - declared:
        raise RuntimeError(
            "expanded pairs not registered: "
            f"{sorted(expanded - declared)}"
        )
    if plain & expanded:
        raise RuntimeError(
            "pair registered in both axis maps: "
            f"{sorted(plain & expanded)}"
        )
    uncovered = declared - plain - expanded
    if uncovered:
        raise RuntimeError(
            "registered Level-2 pair has no state axis: "
            f"{sorted(uncovered)}"
        )
    orphan = (plain | expanded) - declared
    if orphan:
        raise RuntimeError(
            "state axis defined for unregistered pair: "
            f"{sorted(orphan)}"
        )

# ------------------------------------------------------------
# Source owners
# ------------------------------------------------------------

SOURCE_OWNERS = {
    "SMC": {
        "repo": "bao1872/market_dev",
        "sha": "8686b803c53c3a423badb80491fbd21f06879fbb",
        "consumed_via": (
            "future_dev/research/ob_trigger_snapshot.py"
            "::build_full_ob_smc_tf / build_smc_tf"
        ),
        "canonical": "panji_indicators.compute_smc_pine",
        "note": "frozen; never redefined",
    },
    "LEVELS": {
        "repo": "future_dev",
        "sha": ATLAS_BASELINE_SHA,
        "consumed_via": (
            "future_dev/research/ob_trigger_snapshot.py"
            "::_snapshot_tf_columns / _gather_tf_levels"
        ),
        "note": (
            "level_records emitted per (candidate, context_tf); "
            "primary key event_id; relation/distance_pct are the "
            "committed touch-close zone-edge geometry"
        ),
    },
    "DSA": {
        "repo": "bao1872/market_dev",
        "sha": "8686b803c53c3a423badb80491fbd21f06879fbb",
        "paths": (
            "backend/app/strategy/selectors/dsa_selector.py",
            "backend/app/strategy_assets/algorithms/features/"
            "dynamic_swing_anchored_vwap.py",
        ),
        "consumed_via": (
            "future_dev/research/dsa_adapter.py"
            "::compute_dsa_canonical"
        ),
        "note": (
            "panji_indicators.py is the frozen 1:1 extraction of the "
            "market_dev DSA; math is consumed, never redefined"
        ),
    },
    "MOMENTUM": {
        "repo": "bao1872/market_dev",
        "sha": "8686b803c53c3a423badb80491fbd21f06879fbb",
        "consumed_via": (
            "future_dev/research/indicator_adapter.py"
            "::compute_smc_momentum_bundle"
        ),
        "canonical": (
            "panji_indicators.compute_sqzmom_lb + "
            "panji_indicators.build_momentum_history"
        ),
        "note": (
            "SQZMOM is NOT reimplemented anywhere in the atlas; "
            "momentum_history['daily_state'] is the source-owner "
            "return structure; direction/change/phase are strings"
        ),
    },
    "QUANTILE": {
        "repo": "future_dev",
        "sha": ATLAS_BASELINE_SHA,
        "consumed_via": (
            "future_dev/research/quantile_opportunity_contiguous.py"
        ),
        "reused": (
            "research/fit_quantile_v2_models.py::make_model",
            "research/run_quantile_rebaseline.py"
            "::FEATURE_SETS, make_folds",
            "research/build_pytdx_panel.py"
            "::aggregate_15m, build_features, build_targets",
        ),
        "note": (
            "contiguous-only OOS; percentile from current-fold train "
            "width distribution; PIT attach window 15 minutes"
        ),
    },
    "ATR": {
        "repo": "future_dev",
        "sha": ATLAS_BASELINE_SHA,
        "consumed_via": (
            "future_dev/research/export_ob_trigger_execution_v21.py"
            "::pine_atr / audit_atr_recurrence"
        ),
        "note": "Pine/Wilder RMA ATR14",
    },
    "AGGREGATION": {
        "repo": "future_dev",
        "sha": ATLAS_BASELINE_SHA,
        "consumed_via": (
            "research/build_pytdx_panel.py::aggregate_15m",
            "research/ob_trigger_snapshot.py"
            "::aggregate_1h_from_15m",
        ),
        "note": "5m -> 15m -> 1h only; no validated 4h authority",
    },
}

# ------------------------------------------------------------
# 4h authority finding (investigated, NOT invented)
# ------------------------------------------------------------

FOUR_HOUR_AUTHORITY = {
    "status": "NOT_FOUND",
    "searched": (
        "future_dev/research/*.py "
        "(resample / four_hour / FOUR_HOUR)",
        "future_dev/panji_indicators.py",
        "market_dev/backend/**/*.py "
        '("4h" / four_hour / FOUR_HOUR / 240-minute)',
    ),
    "findings": (
        "The only 4h bar construction in future_dev is "
        "build_ob_candidate_universe_v3.aggregate_4h_from_1h, which "
        "buckets by wall-clock epoch (start_ns // 4h). Chinese futures "
        "sessions contain breaks, so every resulting bucket holds only "
        "1-3 x 1h components (observed AG {3: 1199, 2: 407}; "
        "frac_complete_240 = 0.0).",
        "The string '4h' in prepare_quantile_v2_data.py is a FORWARD "
        "HORIZON LABEL (48 x 5m bars), not a 4h bar series.",
        "market_dev/backend returned 0 matches for any 4h aggregation "
        "or session-semantics authority.",
    ),
    "consequence": (
        "4h remains HARD-QUARANTINED. No 4h environment field, no 4h "
        "outcome, no 4h candidate. This atlas is a "
        "5m / 15m / 1h validated-environment atlas."
    ),
}


def assert_validated_tf(tf: str) -> None:
    if tf in QUARANTINED_TFS:
        raise RuntimeError(
            f"timeframe {tf!r} is quarantined: "
            f"{FOUR_HOUR_AUTHORITY['consequence']}"
        )
    if tf not in VALIDATED_TFS:
        raise RuntimeError(f"timeframe {tf!r} is not validated")


def is_quarantined(tf: str) -> bool:
    return tf in QUARANTINED_TFS
