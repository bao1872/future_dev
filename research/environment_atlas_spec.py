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

# Only these timeframes have a validated aggregation authority.
VALIDATED_TFS = ("5m", "15m", "1h")

# 4h is hard-quarantined. See FOUR_HOUR_AUTHORITY below.
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

# ------------------------------------------------------------
# Cell sample-size gates
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
# Canonical Momentum (SQZMOM) state fields.
#
# These are exactly the keys emitted by the canonical
# build_momentum_history()["daily_state"] in panji_indicators.py.
# They are consumed, never recomputed.
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

MOMENTUM_JOIN_FIELD = "bar_index"

# ------------------------------------------------------------
# DSA raw environment fields carried into the atlas.
#
# IMPORTANT: DSA is used here as a RUNNING-REGIME environment, not as
# three additive direction variables. The joint multi-TF state matters,
# so the full raw environment is retained per timeframe.
# ------------------------------------------------------------

DSA_ENV_FIELDS = (
    "dsa_direction",
    "dsa_raw_regime_value",
    "dsa_raw_regime_strength",
    "dsa_raw_dsa_dir_bars",
    "dsa_raw_trend_transition",
    "dsa_raw_offset_rate",
    "dsa_raw_offset_mean",
    "dsa_raw_offset_std",
    "dsa_raw_offset_percentile",
    "dsa_raw_offset_variance_rate",
    "dsa_raw_vwap_ret_avg",
    "dsa_raw_vwap_ret_total",
    "dsa_raw_vwap_ret_5",
    "dsa_raw_vwap_ret_10",
    "dsa_raw_vwap_ret_20",
    "dsa_raw_dsa_vwap",
    "dsa_raw_dsa_vwap_dev_pct",
    "dsa_raw_change_pct",
    "dsa_raw_vol_zscore",
    "dsa_raw_segment_id",
    "dsa_raw_segment_direction",
    "dsa_raw_segment_bars",
    "dsa_raw_segment_change_pct",
    "dsa_raw_segment_slope",
    "dsa_raw_prev_segment_id",
    "dsa_raw_prev_segment_direction",
    "dsa_raw_prev_segment_bars",
    "dsa_raw_prev_segment_slope",
    "dsa_raw_current_vs_prev_volume_mean_ratio",
    "dsa_raw_current_vs_prev_amount_mean_ratio",
    "dsa_raw_rope_dir1_pct",
    "dsa_raw_rope_dir0_pct",
    "dsa_raw_rope_dir_neg1_pct",
    "dsa_raw_touch_rope",
    "dsa_raw_touch_vwap",
    "dsa_raw_cross_up_count",
    "dsa_raw_cross_down_count",
    "dsa_raw_rope_cross_up_count",
    "dsa_raw_rope_cross_down_count",
)

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
# NOTE the primary key is `event_id`, NOT `candidate_id`.
# The mapping must be PROVEN at runtime (see
# build_ob_environment_atlas_v1.resolve_level_join_key).
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

# Object-type buckets used for pressure/support density.
# Actual enumerations are VERIFIED against real levels data; this is
# only the classification vocabulary, never assumed silently.
LEVEL_TYPE_BUCKETS = (
    "active_ob",
    "internal_pivot",
    "swing_pivot",
    "equal_level",
    "other",
)

# ------------------------------------------------------------
# Pre-registered Level-2 interactions.
#
# A full cartesian product over all families is FORBIDDEN.
# Only these pairs are analysed.
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
            "primary key event_id"
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
            "return structure"
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
