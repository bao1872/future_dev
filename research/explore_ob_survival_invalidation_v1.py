#!/usr/bin/env python3

"""
OB Survival / Invalidation V1

Question
--------
Conditional on the source OB NOT already closing beyond its far edge
on the touch bar, do simple causal state factors change the probability
of a FUTURE 5m close invalidating that OB?

Definition
----------
Execution-grid invalidation:

bull OB:
    future 5m close < source_ob_zone_low

bear OB:
    future 5m close > source_ob_zone_high

Future starts at the next 5m bar after the touch bar.

Fixed:
- 21,481 frozen OB-touch candidates
- exclude touch-bar CLOSE_BEYOND
- expected eligible universe = 20,013
- horizons 6 / 12 / 24
- H12 primary
- whole-trading-day 70/30 discovery/validation
- original decision_weight preserved
- no ML
- no DSA
- no Momentum
- no feature combinations
- no tuning / bootstrap / CI
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from research.audit_ob_candidate_v3_posthoc import (  # noqa: E402
    load_full_or_chunks,
)

from research.analyze_ob_candidate_v3_phase1 import (  # noqa: E402
    build_path_arrays,
    weighted_mean,
    weighted_quantile,
)

from research.ob_rl_dataset_v0_spec import (  # noqa: E402
    VALIDATED_TFS,
)


# ============================================================
# Fixed experiment contract
# ============================================================

DATA_ROOT = (
    ROOT
    / "research"
    / "analysis_results"
    / "ob_rl_dataset_v0"
)

STATE_PARQUET = (
    DATA_ROOT
    / "ob_rl_state_v0.parquet"
)

OUT = (
    ROOT
    / "research"
    / "analysis_results"
    / "ob_survival_invalidation_v1"
)

EXPECTED_CANDIDATES = 21_481

EXPECTED_ALREADY_INVALIDATED = 1_468

EXPECTED_ELIGIBLE = (
    EXPECTED_CANDIDATES
    - EXPECTED_ALREADY_INVALIDATED
)

HORIZONS = (
    6,
    12,
    24,
)

PRIMARY_HORIZON = 12

DISCOVERY_FRACTION = 0.70


# ============================================================
# Pre-registered survival thresholds
# ============================================================

MIN_DISCOVERY_N = 200

MIN_VALIDATION_N = 100

MIN_SYMBOL_N = 30

MIN_DISCOVERY_DELTA_PP = 5.0

MIN_VALIDATION_DELTA_PP = 3.0

MIN_SYMBOLS_SAME_SIGN = 3


FACTOR_BUCKETS = {
    "source_tf": (
        "5m",
        "15m",
        "1h",
    ),

    "source_ob_structure": (
        "internal",
        "swing",
    ),

    "touch_bin": (
        "1",
        "2",
        "3",
        "4+",
    ),

    "ob_width_bucket": (
        "NARROW_LE_0.5",
        "MEDIUM_0.5_1.0",
        "WIDE_GT_1.0",
    ),

    "quant_state": (
        "LOW",
        "MID",
        "HIGH",
    ),

    "smc_internal_align_count": (
        0,
        1,
        2,
        3,
    ),

    "smc_swing_align_count": (
        0,
        1,
        2,
        3,
    ),
}


# ============================================================
# Utilities
# ============================================================

def git_head() -> str:

    p = subprocess.run(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    sha = p.stdout.strip()

    if len(sha) != 40:
        raise RuntimeError(
            f"invalid HEAD: {sha!r}"
        )

    return sha


def parse_bool(
    s: pd.Series,
) -> pd.Series:

    if pd.api.types.is_bool_dtype(
        s
    ):
        return s.astype(bool)

    z = (
        s.astype(str)
        .str.strip()
        .str.lower()
    )

    bad = ~z.isin(
        [
            "true",
            "false",
        ]
    )

    if bad.any():
        raise RuntimeError(
            "invalid boolean values: "
            f"{sorted(z[bad].unique())}"
        )

    return z.eq(
        "true"
    )


def weighted_rate(
    x,
    w,
) -> float:

    return weighted_mean(
        np.asarray(
            x,
            dtype=float,
        ),
        np.asarray(
            w,
            dtype=float,
        ),
    )


def finite_delta_pp(
    a,
    b,
) -> float:

    if (
        pd.isna(a)
        or pd.isna(b)
    ):
        return np.nan

    return float(
        100.0
        * (
            float(a)
            - float(b)
        )
    )


# ============================================================
# Load corrected state + frozen OB geometry
# ============================================================

def load_input() -> pd.DataFrame:

    state_cols = [
        "candidate_id",
        "symbol",
        "trading_day",
        "source_tf",
        "source_ob_structure",
        "source_ob_bias",
        "source_ob_width_atr5",
        "touch_ordinal",
        "touch_behavior",
        "quant_state",
        "decision_weight",
    ]

    for tf in VALIDATED_TFS:
        state_cols.extend(
            [
                f"internal_bias_{tf}",
                f"swing_bias_{tf}",
            ]
        )

    if not STATE_PARQUET.exists():
        raise RuntimeError(
            "missing corrected state parquet: "
            f"{STATE_PARQUET}"
        )

    state = pd.read_parquet(
        STATE_PARQUET,
        columns=state_cols,
    )

    if len(state) != EXPECTED_CANDIDATES:
        raise RuntimeError(
            "state cardinality drift: "
            f"{len(state)}"
        )

    if not state[
        "candidate_id"
    ].is_unique:
        raise RuntimeError(
            "state candidate_id not unique"
        )

    candidates = (
        load_full_or_chunks(
            "candidates"
        )
    )

    candidate_cols = [
        "candidate_id",
        "symbol",
        "source_ob_bias",
        "source_ob_zone_low",
        "source_ob_zone_high",
        "entry_next_5m_open",
        "5m_atr14",
        "touch_close_beyond_far_edge",
    ]

    missing = (
        set(candidate_cols)
        - set(candidates.columns)
    )

    if missing:
        raise RuntimeError(
            "frozen candidates missing: "
            f"{sorted(missing)}"
        )

    geom = (
        candidates[
            candidate_cols
        ]
        .copy()
    )

    if len(geom) != EXPECTED_CANDIDATES:
        raise RuntimeError(
            "candidate geometry "
            "cardinality drift"
        )

    if not geom[
        "candidate_id"
    ].is_unique:
        raise RuntimeError(
            "candidate geometry "
            "candidate_id not unique"
        )

    geom[
        "touch_close_beyond_far_edge"
    ] = parse_bool(
        geom[
            "touch_close_beyond_far_edge"
        ]
    )

    geom = geom.rename(
        columns={
            "symbol":
                "geom_symbol",

            "source_ob_bias":
                "geom_source_ob_bias",
        }
    )

    state[
        "_row_order"
    ] = np.arange(
        len(state)
    )

    x = state.merge(
        geom,
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )

    x = (
        x.sort_values(
            "_row_order"
        )
        .reset_index(
            drop=True
        )
    )

    if x[
        "geom_symbol"
    ].isna().any():
        raise RuntimeError(
            "candidate geometry "
            "merge incomplete"
        )

    if not np.array_equal(
        x[
            "symbol"
        ].astype(str).to_numpy(),
        x[
            "geom_symbol"
        ].astype(str).to_numpy(),
    ):
        raise RuntimeError(
            "symbol mismatch "
            "state vs frozen candidate"
        )

    state_bias = pd.to_numeric(
        x[
            "source_ob_bias"
        ],
        errors="raise",
    ).to_numpy(float)

    geom_bias = pd.to_numeric(
        x[
            "geom_source_ob_bias"
        ],
        errors="raise",
    ).to_numpy(float)

    if not np.array_equal(
        state_bias,
        geom_bias,
    ):
        raise RuntimeError(
            "source_ob_bias mismatch"
        )

    if not np.isin(
        state_bias,
        [-1.0, 1.0],
    ).all():
        raise RuntimeError(
            "source_ob_bias must be ±1"
        )

    state_close = (
        x[
            "touch_behavior"
        ]
        .astype(str)
        .eq(
            "CLOSE_BEYOND"
        )
        .to_numpy()
    )

    frozen_close = (
        x[
            "touch_close_beyond_far_edge"
        ]
        .astype(bool)
        .to_numpy()
    )

    if not np.array_equal(
        state_close,
        frozen_close,
    ):
        raise RuntimeError(
            "touch CLOSE_BEYOND "
            "semantic mismatch"
        )

    already_invalidated = int(
        frozen_close.sum()
    )

    if (
        already_invalidated
        != EXPECTED_ALREADY_INVALIDATED
    ):
        raise RuntimeError(
            "touch-bar invalidated "
            "count drift: "
            f"{already_invalidated}"
        )

    x[
        "eligible"
    ] = ~frozen_close

    eligible_n = int(
        x[
            "eligible"
        ].sum()
    )

    if eligible_n != EXPECTED_ELIGIBLE:
        raise RuntimeError(
            "eligible universe drift: "
            f"{eligible_n} "
            f"!= {EXPECTED_ELIGIBLE}"
        )

    weight = pd.to_numeric(
        x[
            "decision_weight"
        ],
        errors="coerce",
    )

    if (
        weight.isna().any()
        or (weight <= 0).any()
    ):
        raise RuntimeError(
            "invalid decision_weight"
        )

    entry = pd.to_numeric(
        x[
            "entry_next_5m_open"
        ],
        errors="coerce",
    ).to_numpy(float)

    atr = pd.to_numeric(
        x[
            "5m_atr14"
        ],
        errors="coerce",
    ).to_numpy(float)

    zone_low = pd.to_numeric(
        x[
            "source_ob_zone_low"
        ],
        errors="coerce",
    ).to_numpy(float)

    zone_high = pd.to_numeric(
        x[
            "source_ob_zone_high"
        ],
        errors="coerce",
    ).to_numpy(float)

    far_edge = np.where(
        state_bias > 0,
        zone_low,
        zone_high,
    )

    far_edge_atr = np.full(
        len(x),
        np.nan,
    )

    ok = (
        np.isfinite(entry)
        & np.isfinite(atr)
        & (atr > 0)
        & np.isfinite(far_edge)
    )

    far_edge_atr[
        ok
    ] = (
        far_edge[ok]
        - entry[ok]
    ) / atr[ok]

    x[
        "far_edge_atr"
    ] = far_edge_atr

    return x


# ============================================================
# Derived factors
# ============================================================

def add_factors(
    x: pd.DataFrame,
) -> pd.DataFrame:

    x = x.copy()

    x[
        "source_tf"
    ] = (
        x[
            "source_tf"
        ]
        .astype(str)
    )

    x[
        "source_ob_structure"
    ] = (
        x[
            "source_ob_structure"
        ]
        .astype(str)
    )

    x[
        "quant_state"
    ] = (
        x[
            "quant_state"
        ]
        .astype(str)
    )

    ordinal = pd.to_numeric(
        x[
            "touch_ordinal"
        ],
        errors="coerce",
    )

    x[
        "touch_bin"
    ] = np.select(
        [
            ordinal.eq(1),
            ordinal.eq(2),
            ordinal.eq(3),
            ordinal.ge(4),
        ],
        [
            "1",
            "2",
            "3",
            "4+",
        ],
        default="UNKNOWN",
    )

    width = pd.to_numeric(
        x[
            "source_ob_width_atr5"
        ],
        errors="coerce",
    )

    if (
        width.dropna()
        < 0
    ).any():
        raise RuntimeError(
            "negative source_ob_width_atr5"
        )

    x[
        "ob_width_bucket"
    ] = np.select(
        [
            width.le(0.5),
            (
                width.gt(0.5)
                & width.le(1.0)
            ),
            width.gt(1.0),
        ],
        [
            "NARROW_LE_0.5",
            "MEDIUM_0.5_1.0",
            "WIDE_GT_1.0",
        ],
        default="UNKNOWN",
    )

    source_bias = (
        pd.to_numeric(
            x[
                "source_ob_bias"
            ],
            errors="raise",
        )
        .to_numpy(float)
    )

    for kind in (
        "internal",
        "swing",
    ):

        rel = []

        for tf in VALIDATED_TFS:

            col = (
                f"{kind}_bias_{tf}"
            )

            raw = pd.to_numeric(
                x[col],
                errors="coerce",
            )

            if raw.isna().any():
                raise RuntimeError(
                    f"{col}: NaN "
                    "not allowed"
                )

            observed = set(
                raw.unique()
            )

            if not observed.issubset(
                {-1.0, 0.0, 1.0}
            ):
                raise RuntimeError(
                    f"{col}: illegal values "
                    f"{sorted(observed)}"
                )

            rel.append(
                raw.to_numpy(float)
                * source_bias
            )

        matrix = np.column_stack(
            rel
        )

        x[
            f"smc_{kind}_align_count"
        ] = (
            matrix
            == 1
        ).sum(
            axis=1
        ).astype(int)

    return x


# ============================================================
# Whole-trading-day split
# ============================================================

def add_split(
    x: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Timestamp,
]:

    x = x.copy()

    x[
        "_day"
    ] = pd.to_datetime(
        x[
            "trading_day"
        ],
        errors="raise",
    )

    days = np.array(
        sorted(
            x[
                "_day"
            ].unique()
        )
    )

    if len(days) < 10:
        raise RuntimeError(
            "too few trading days"
        )

    cut = int(
        np.floor(
            len(days)
            * DISCOVERY_FRACTION
        )
    )

    cut = min(
        max(cut, 1),
        len(days) - 1,
    )

    split_day = pd.Timestamp(
        days[cut]
    )

    x[
        "split"
    ] = np.where(
        x[
            "_day"
        ]
        < split_day,
        "DISCOVERY",
        "VALIDATION",
    )

    discovery_days = set(
        x.loc[
            x[
                "split"
            ]
            == "DISCOVERY",
            "_day",
        ]
    )

    validation_days = set(
        x.loc[
            x[
                "split"
            ]
            == "VALIDATION",
            "_day",
        ]
    )

    if (
        discovery_days
        & validation_days
    ):
        raise RuntimeError(
            "same trading day "
            "crossed split"
        )

    return (
        x,
        split_day,
    )


# ============================================================
# Future close-invalidation outcomes
# ============================================================

def build_outcomes(
    x: pd.DataFrame,
) -> dict:

    path = load_full_or_chunks(
        "path"
    )

    arr = build_path_arrays(
        x[
            [
                "candidate_id",
            ]
        ],
        path,
    )

    close = arr[
        "close_atr"
    ]

    contig = arr[
        "contig"
    ]

    if close.shape[0] != len(x):
        raise RuntimeError(
            "path candidate "
            "cardinality drift"
        )

    if close.shape[1] < max(
        HORIZONS
    ):
        raise RuntimeError(
            "path horizon too short"
        )

    bias = pd.to_numeric(
        x[
            "source_ob_bias"
        ],
        errors="raise",
    ).to_numpy(float)

    edge = pd.to_numeric(
        x[
            "far_edge_atr"
        ],
        errors="coerce",
    ).to_numpy(float)

    out = {}

    for h in HORIZONS:

        cl = close[
            :,
            :h,
        ]

        ct = contig[
            :,
            :h,
        ]

        analyzable = (
            np.isfinite(
                edge
            )
            & np.isfinite(
                cl
            ).all(
                axis=1
            )
            & ct.all(
                axis=1
            )
        )

        bull_cross = (
            cl
            < edge[:, None]
        )

        bear_cross = (
            cl
            > edge[:, None]
        )

        crossed = np.where(
            bias[:, None] > 0,
            bull_cross,
            bear_cross,
        )

        invalidated = (
            analyzable
            & crossed.any(
                axis=1
            )
        )

        first_step = np.full(
            len(x),
            np.nan,
        )

        if invalidated.any():

            idx = np.flatnonzero(
                invalidated
            )

            first_step[
                idx
            ] = (
                np.argmax(
                    crossed[idx],
                    axis=1,
                )
                + 1
            )

        out[h] = {
            "analyzable":
                analyzable,

            "invalidated":
                invalidated,

            "first_step":
                first_step,
        }

    return out


# ============================================================
# One subset -> invalidation distribution
# ============================================================

def summarize(
    mask: np.ndarray,
    *,
    horizon: int,
    outcomes: dict,
    weights: np.ndarray,
) -> dict:

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    o = outcomes[
        horizon
    ]

    analyzed = (
        mask
        & o[
            "analyzable"
        ]
    )

    n = int(
        analyzed.sum()
    )

    if n == 0:

        return {
            "candidate_n":
                int(mask.sum()),

            "analyzable_n":
                0,

            "analyzable_weight":
                0.0,

            "invalidated_n":
                0,

            "invalidation_rate":
                np.nan,

            "survival_rate":
                np.nan,

            "mean_first_invalidation_step":
                np.nan,

            "median_first_invalidation_step":
                np.nan,
        }

    w = weights[
        analyzed
    ]

    invalidated = o[
        "invalidated"
    ][
        analyzed
    ]

    rate = weighted_rate(
        invalidated,
        w,
    )

    invalidated_global = (
        analyzed
        & o[
            "invalidated"
        ]
    )

    first = o[
        "first_step"
    ][
        invalidated_global
    ]

    w_first = weights[
        invalidated_global
    ]

    if len(first):

        mean_step = weighted_mean(
            first,
            w_first,
        )

        median_step = weighted_quantile(
            first,
            0.50,
            w_first,
        )

    else:

        mean_step = np.nan
        median_step = np.nan

    return {
        "candidate_n":
            int(mask.sum()),

        "analyzable_n":
            n,

        "analyzable_weight":
            float(
                w.sum()
            ),

        "invalidated_n":
            int(
                invalidated.sum()
            ),

        "invalidation_rate":
            rate,

        "survival_rate":
            (
                1.0 - rate
                if np.isfinite(rate)
                else np.nan
            ),

        "mean_first_invalidation_step":
            mean_step,

        "median_first_invalidation_step":
            median_step,
    }


# ============================================================
# Relationship tables
# ============================================================

def build_tables(
    x: pd.DataFrame,
    outcomes: dict,
):

    weights = pd.to_numeric(
        x[
            "decision_weight"
        ],
        errors="raise",
    ).to_numpy(float)

    eligible = (
        x[
            "eligible"
        ]
        .astype(bool)
        .to_numpy()
    )

    baseline_rows = []

    relationship_rows = []

    symbol_rows = []

    symbols = sorted(
        x[
            "symbol"
        ]
        .astype(str)
        .unique()
    )

    for h in HORIZONS:

        for split in (
            "DISCOVERY",
            "VALIDATION",
        ):

            split_mask = (
                x[
                    "split"
                ]
                .eq(split)
                .to_numpy()
            )

            st = summarize(
                eligible
                & split_mask,
                horizon=h,
                outcomes=outcomes,
                weights=weights,
            )

            baseline_rows.append(
                {
                    "horizon":
                        h,

                    "split":
                        split,

                    **st,
                }
            )

    for (
        factor,
        buckets,
    ) in FACTOR_BUCKETS.items():

        known = (
            x[
                factor
            ]
            .isin(
                buckets
            )
            .to_numpy()
        )

        for bucket in buckets:

            bucket_eq = (
                x[
                    factor
                ]
                .eq(
                    bucket
                )
                .to_numpy()
            )

            for h in HORIZONS:

                for split in (
                    "DISCOVERY",
                    "VALIDATION",
                ):

                    split_mask = (
                        x[
                            "split"
                        ]
                        .eq(split)
                        .to_numpy()
                    )

                    pool = (
                        eligible
                        & split_mask
                        & known
                    )

                    bucket_mask = (
                        pool
                        & bucket_eq
                    )

                    rest_mask = (
                        pool
                        & ~bucket_eq
                    )

                    b = summarize(
                        bucket_mask,
                        horizon=h,
                        outcomes=outcomes,
                        weights=weights,
                    )

                    r = summarize(
                        rest_mask,
                        horizon=h,
                        outcomes=outcomes,
                        weights=weights,
                    )

                    relationship_rows.append(
                        {
                            "factor":
                                factor,

                            "bucket":
                                str(bucket),

                            "horizon":
                                h,

                            "split":
                                split,

                            **{
                                f"bucket_{k}":
                                    v
                                for k, v
                                in b.items()
                            },

                            **{
                                f"rest_{k}":
                                    v
                                for k, v
                                in r.items()
                            },

                            "delta_invalidation_pp":
                                finite_delta_pp(
                                    b[
                                        "invalidation_rate"
                                    ],
                                    r[
                                        "invalidation_rate"
                                    ],
                                ),

                            "delta_mean_first_step":
                                (
                                    float(
                                        b[
                                            "mean_first_invalidation_step"
                                        ]
                                        - r[
                                            "mean_first_invalidation_step"
                                        ]
                                    )
                                    if (
                                        np.isfinite(
                                            b[
                                                "mean_first_invalidation_step"
                                            ]
                                        )
                                        and np.isfinite(
                                            r[
                                                "mean_first_invalidation_step"
                                            ]
                                        )
                                    )
                                    else np.nan
                                ),
                        }
                    )

                # Validation cross-symbol.
                for symbol in symbols:

                    symbol_mask = (
                        x[
                            "symbol"
                        ]
                        .astype(str)
                        .eq(symbol)
                        .to_numpy()
                    )

                    validation_mask = (
                        x[
                            "split"
                        ]
                        .eq(
                            "VALIDATION"
                        )
                        .to_numpy()
                    )

                    pool = (
                        eligible
                        & validation_mask
                        & symbol_mask
                        & known
                    )

                    bucket_mask = (
                        pool
                        & bucket_eq
                    )

                    rest_mask = (
                        pool
                        & ~bucket_eq
                    )

                    b = summarize(
                        bucket_mask,
                        horizon=h,
                        outcomes=outcomes,
                        weights=weights,
                    )

                    r = summarize(
                        rest_mask,
                        horizon=h,
                        outcomes=outcomes,
                        weights=weights,
                    )

                    symbol_rows.append(
                        {
                            "factor":
                                factor,

                            "bucket":
                                str(bucket),

                            "horizon":
                                h,

                            "symbol":
                                symbol,

                            "bucket_analyzable_n":
                                b[
                                    "analyzable_n"
                                ],

                            "rest_analyzable_n":
                                r[
                                    "analyzable_n"
                                ],

                            "bucket_invalidation_rate":
                                b[
                                    "invalidation_rate"
                                ],

                            "rest_invalidation_rate":
                                r[
                                    "invalidation_rate"
                                ],

                            "delta_invalidation_pp":
                                finite_delta_pp(
                                    b[
                                        "invalidation_rate"
                                    ],
                                    r[
                                        "invalidation_rate"
                                    ],
                                ),
                        }
                    )

    return (
        pd.DataFrame(
            baseline_rows
        ),
        pd.DataFrame(
            relationship_rows
        ),
        pd.DataFrame(
            symbol_rows
        ),
    )


# ============================================================
# Pre-registered H12 survival screen
# ============================================================

def build_screen(
    relationships: pd.DataFrame,
    by_symbol: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        factor,
        buckets,
    ) in FACTOR_BUCKETS.items():

        for bucket in buckets:

            bucket_label = str(
                bucket
            )

            d = relationships[
                (
                    relationships[
                        "factor"
                    ]
                    == factor
                )
                & (
                    relationships[
                        "bucket"
                    ]
                    == bucket_label
                )
                & (
                    relationships[
                        "horizon"
                    ]
                    == PRIMARY_HORIZON
                )
                & (
                    relationships[
                        "split"
                    ]
                    == "DISCOVERY"
                )
            ]

            v = relationships[
                (
                    relationships[
                        "factor"
                    ]
                    == factor
                )
                & (
                    relationships[
                        "bucket"
                    ]
                    == bucket_label
                )
                & (
                    relationships[
                        "horizon"
                    ]
                    == PRIMARY_HORIZON
                )
                & (
                    relationships[
                        "split"
                    ]
                    == "VALIDATION"
                )
            ]

            if (
                len(d) != 1
                or len(v) != 1
            ):
                raise RuntimeError(
                    "relationship cardinality "
                    "drift"
                )

            d = d.iloc[0]

            v = v.iloc[0]

            d_delta = float(
                d[
                    "delta_invalidation_pp"
                ]
            )

            v_delta = float(
                v[
                    "delta_invalidation_pp"
                ]
            )

            support_ok = (
                d[
                    "bucket_analyzable_n"
                ]
                >= MIN_DISCOVERY_N

                and d[
                    "rest_analyzable_n"
                ]
                >= MIN_DISCOVERY_N

                and v[
                    "bucket_analyzable_n"
                ]
                >= MIN_VALIDATION_N

                and v[
                    "rest_analyzable_n"
                ]
                >= MIN_VALIDATION_N
            )

            discovery_ok = (
                np.isfinite(
                    d_delta
                )
                and abs(
                    d_delta
                )
                >= MIN_DISCOVERY_DELTA_PP
            )

            validation_ok = (
                np.isfinite(
                    v_delta
                )
                and abs(
                    v_delta
                )
                >= MIN_VALIDATION_DELTA_PP
                and (
                    d_delta
                    * v_delta
                    > 0
                )
            )

            ss = by_symbol[
                (
                    by_symbol[
                        "factor"
                    ]
                    == factor
                )
                & (
                    by_symbol[
                        "bucket"
                    ]
                    == bucket_label
                )
                & (
                    by_symbol[
                        "horizon"
                    ]
                    == PRIMARY_HORIZON
                )
            ].copy()

            ss = ss[
                (
                    ss[
                        "bucket_analyzable_n"
                    ]
                    >= MIN_SYMBOL_N
                )
                & (
                    ss[
                        "rest_analyzable_n"
                    ]
                    >= MIN_SYMBOL_N
                )
                & np.isfinite(
                    ss[
                        "delta_invalidation_pp"
                    ]
                )
            ]

            symbol_available = int(
                len(ss)
            )

            symbol_same_sign = 0

            if (
                np.isfinite(
                    v_delta
                )
                and v_delta != 0
            ):

                symbol_same_sign = int(
                    (
                        ss[
                            "delta_invalidation_pp"
                        ]
                        * v_delta
                        > 0
                    ).sum()
                )

            cross_symbol_ok = (
                symbol_available
                >= MIN_SYMBOLS_SAME_SIGN
                and symbol_same_sign
                >= MIN_SYMBOLS_SAME_SIGN
            )

            if not support_ok:

                status = (
                    "FAIL_SUPPORT"
                )

            elif not discovery_ok:

                status = (
                    "FAIL_DISCOVERY_EFFECT"
                )

            elif not validation_ok:

                status = (
                    "FAIL_VALIDATION"
                )

            elif not cross_symbol_ok:

                status = (
                    "FAIL_CROSS_SYMBOL"
                )

            elif v_delta > 0:

                status = (
                    "SURVIVES_HIGHER_INVALIDATION"
                )

            else:

                status = (
                    "SURVIVES_LOWER_INVALIDATION"
                )

            rows.append(
                {
                    "factor":
                        factor,

                    "bucket":
                        bucket_label,

                    "status":
                        status,

                    "discovery_bucket_rate":
                        d[
                            "bucket_invalidation_rate"
                        ],

                    "discovery_rest_rate":
                        d[
                            "rest_invalidation_rate"
                        ],

                    "discovery_delta_pp":
                        d_delta,

                    "validation_bucket_rate":
                        v[
                            "bucket_invalidation_rate"
                        ],

                    "validation_rest_rate":
                        v[
                            "rest_invalidation_rate"
                        ],

                    "validation_delta_pp":
                        v_delta,

                    "discovery_bucket_n":
                        int(
                            d[
                                "bucket_analyzable_n"
                            ]
                        ),

                    "validation_bucket_n":
                        int(
                            v[
                                "bucket_analyzable_n"
                            ]
                        ),

                    "symbol_available":
                        symbol_available,

                    "symbol_same_sign":
                        symbol_same_sign,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    x = load_input()

    x = add_factors(
        x
    )

    x, split_day = add_split(
        x
    )

    outcomes = build_outcomes(
        x
    )

    (
        baseline,
        relationships,
        by_symbol,
    ) = build_tables(
        x,
        outcomes,
    )

    screen = build_screen(
        relationships,
        by_symbol,
    )

    baseline.to_csv(
        OUT
        / "baseline.csv",
        index=False,
    )

    relationships.to_csv(
        OUT
        / "factor_relationships.csv",
        index=False,
    )

    by_symbol.to_csv(
        OUT
        / "validation_by_symbol.csv",
        index=False,
    )

    screen.to_csv(
        OUT
        / "relationship_screen.csv",
        index=False,
    )

    survivors = screen[
        screen[
            "status"
        ]
        .astype(str)
        .str.startswith(
            "SURVIVES_"
        )
    ].copy()

    audit = {
        "experiment":
            "ob_survival_invalidation_v1",

        "git_head":
            git_head(),

        "candidate_count":
            int(
                len(x)
            ),

        "already_invalidated_touch_bar":
            int(
                (
                    ~x[
                        "eligible"
                    ]
                ).sum()
            ),

        "eligible_candidates":
            int(
                x[
                    "eligible"
                ].sum()
            ),

        "invalidation_definition":
            (
                "future_5m_close_beyond_"
                "source_ob_far_edge"
            ),

        "future_starts":
            "next_5m_bar_after_touch",

        "split_method":
            "whole_trading_day_70_30",

        "split_day":
            split_day.isoformat(),

        "horizons":
            list(
                HORIZONS
            ),

        "primary_horizon":
            PRIMARY_HORIZON,

        "factors": {
            k:
                list(v)
            for k, v
            in FACTOR_BUCKETS.items()
        },

        "thresholds": {
            "min_discovery_n":
                MIN_DISCOVERY_N,

            "min_validation_n":
                MIN_VALIDATION_N,

            "min_symbol_n":
                MIN_SYMBOL_N,

            "min_discovery_delta_pp":
                MIN_DISCOVERY_DELTA_PP,

            "min_validation_delta_pp":
                MIN_VALIDATION_DELTA_PP,

            "min_symbols_same_sign":
                MIN_SYMBOLS_SAME_SIGN,
        },

        "baseline_rows":
            int(
                len(baseline)
            ),

        "relationship_rows":
            int(
                len(relationships)
            ),

        "symbol_rows":
            int(
                len(by_symbol)
            ),

        "screen_rows":
            int(
                len(screen)
            ),

        "survivor_count":
            int(
                len(survivors)
            ),
    }

    (
        OUT
        / "audit.json"
    ).write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()

    print(
        "OB_SURVIVAL_INVALIDATION_V1_DONE"
    )

    print(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()

    print(
        "STATUS_COUNTS"
    )

    print(
        screen[
            "status"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print(
        "SURVIVORS"
    )

    if len(
        survivors
    ):

        print(
            survivors[
                [
                    "factor",
                    "bucket",
                    "status",
                    "discovery_delta_pp",
                    "validation_delta_pp",
                    "validation_bucket_n",
                    "symbol_available",
                    "symbol_same_sign",
                ]
            ]
            .sort_values(
                "validation_delta_pp",
                ascending=False,
            )
            .to_string(
                index=False
            )
        )

    else:

        print(
            "NO_STABLE_INVALIDATION_RELATIONSHIPS"
        )


if __name__ == "__main__":
    main()
