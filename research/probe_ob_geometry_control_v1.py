#!/usr/bin/env python3

"""
OB Geometry Control V1

Question
--------
Do the previously-surviving OB invalidation relationships remain
after controlling for the ACTUAL touch-close distance to the
source-OB far edge?

Geometry:
    bull OB:
        (touch_close - zone_low) / ATR5

    bear OB:
        (zone_high - touch_close) / ATR5

Only the seven H12 surface relationships that already survived
OB Survival / Invalidation V1 are tested.

No new feature search.
No ML.
No classifier.
No tuning.
No bootstrap.
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
    sys.path.insert(
        0,
        str(ROOT),
    )


import research.explore_ob_survival_invalidation_v1 as B  # noqa: E402

from research.audit_ob_candidate_v3_posthoc import (  # noqa: E402
    load_full_or_chunks,
)

from research.export_ob_trigger_execution_v21 import (  # noqa: E402
    load_raw_5m,
)

from research.analyze_ob_candidate_v3_phase1 import (  # noqa: E402
    weighted_mean,
    weighted_quantile,
)


# ============================================================
# Fixed experiment contract
# ============================================================

OUT = (
    ROOT
    / "research"
    / "analysis_results"
    / "ob_geometry_control_v1"
)

EXPECTED_CANDIDATES = 21_481
EXPECTED_ELIGIBLE = 20_013

HORIZONS = (
    6,
    12,
    24,
)

PRIMARY_HORIZON = 12

GEOMETRY_STRATA = (
    "G1_NEAREST",
    "G2",
    "G3",
    "G4_FARTHEST",
)


# ============================================================
# ONLY previously-surviving relationships
# ============================================================

TARGETS = (
    {
        "factor": "source_tf",
        "bucket": "5m",
        "expected_sign": 1,
    },
    {
        "factor": "source_tf",
        "bucket": "15m",
        "expected_sign": -1,
    },
    {
        "factor": "source_tf",
        "bucket": "1h",
        "expected_sign": -1,
    },
    {
        "factor": "touch_bin",
        "bucket": "1",
        "expected_sign": 1,
    },
    {
        "factor": "touch_bin",
        "bucket": "4+",
        "expected_sign": -1,
    },
    {
        "factor": "ob_width_bucket",
        "bucket": "MEDIUM_0.5_1.0",
        "expected_sign": 1,
    },
    {
        "factor": "ob_width_bucket",
        "bucket": "WIDE_GT_1.0",
        "expected_sign": -1,
    },
)


FACTOR_BUCKETS = {
    "source_tf": (
        "5m",
        "15m",
        "1h",
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
}


# ============================================================
# Pre-registered thresholds
# ============================================================

MIN_DISCOVERY_N = 200
MIN_VALIDATION_N = 100

MIN_DISCOVERY_STRATUM_N = 50
MIN_VALIDATION_STRATUM_N = 30

MIN_ADJUSTED_STRATA = 3
MIN_ADJUSTED_COVERAGE = 0.70

MIN_DISCOVERY_ADJ_DELTA_PP = 5.0
MIN_VALIDATION_ADJ_DELTA_PP = 3.0

MIN_SYMBOL_N = 30
MIN_SYMBOL_STRATUM_N = 8
MIN_SYMBOL_STRATA = 2
MIN_SYMBOL_COVERAGE = 0.50

MIN_SYMBOLS_SAME_SIGN = 3


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


def same_expected_sign(
    value: float,
    expected_sign: int,
) -> bool:

    return (
        np.isfinite(value)
        and value
        * expected_sign
        > 0
    )


# ============================================================
# Derived surface factors
# ============================================================

def add_surface_factors(
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

    return x


# ============================================================
# Actual touch-close geometry
# ============================================================

def attach_touch_geometry(
    x: pd.DataFrame,
) -> pd.DataFrame:

    candidates = load_full_or_chunks(
        "candidates"
    )

    needed = [
        "candidate_id",
        "symbol",
        "touch_5m_bar_index",
        "touch_bar_start_time",
    ]

    missing = (
        set(needed)
        - set(candidates.columns)
    )

    if missing:
        raise RuntimeError(
            "frozen candidates missing "
            f"{sorted(missing)}"
        )

    c = (
        candidates[
            needed
        ]
        .copy()
    )

    if len(c) != EXPECTED_CANDIDATES:
        raise RuntimeError(
            "candidate cardinality drift"
        )

    if not c[
        "candidate_id"
    ].is_unique:
        raise RuntimeError(
            "candidate_id not unique"
        )

    c = c.rename(
        columns={
            "symbol":
                "touch_geom_symbol",
        }
    )

    x = x.merge(
        c,
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )

    if x[
        "touch_5m_bar_index"
    ].isna().any():
        raise RuntimeError(
            "touch geometry merge incomplete"
        )

    if not np.array_equal(
        x[
            "symbol"
        ]
        .astype(str)
        .to_numpy(),
        x[
            "touch_geom_symbol"
        ]
        .astype(str)
        .to_numpy(),
    ):
        raise RuntimeError(
            "geometry symbol mismatch"
        )

    touch_close = np.full(
        len(x),
        np.nan,
    )

    for symbol in sorted(
        x[
            "symbol"
        ]
        .astype(str)
        .unique()
    ):

        five = load_raw_5m(
            symbol
        )

        rows = np.flatnonzero(
            x[
                "symbol"
            ]
            .astype(str)
            .eq(symbol)
            .to_numpy()
        )

        idx = pd.to_numeric(
            x.iloc[
                rows
            ][
                "touch_5m_bar_index"
            ],
            errors="raise",
        ).astype(int).to_numpy()

        if (
            (idx < 0).any()
            or (
                idx
                >= len(five)
            ).any()
        ):
            raise RuntimeError(
                f"{symbol}: touch index "
                "out of raw 5m range"
            )

        got_time = pd.to_datetime(
            five.iloc[
                idx
            ][
                "bar_start_time"
            ]
        ).to_numpy(
            dtype="datetime64[ns]"
        )

        expected_time = pd.to_datetime(
            x.iloc[
                rows
            ][
                "touch_bar_start_time"
            ]
        ).to_numpy(
            dtype="datetime64[ns]"
        )

        if not np.array_equal(
            got_time,
            expected_time,
        ):
            raise RuntimeError(
                f"{symbol}: raw 5m "
                "touch-time mismatch"
            )

        close = pd.to_numeric(
            five.iloc[
                idx
            ][
                "close"
            ],
            errors="coerce",
        ).to_numpy(float)

        if not np.isfinite(
            close
        ).all():
            raise RuntimeError(
                f"{symbol}: nonfinite "
                "touch close"
            )

        touch_close[
            rows
        ] = close

    x[
        "touch_close"
    ] = touch_close

    bias = pd.to_numeric(
        x[
            "source_ob_bias"
        ],
        errors="raise",
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

    atr = pd.to_numeric(
        x[
            "5m_atr14"
        ],
        errors="coerce",
    ).to_numpy(float)

    if not np.isin(
        bias,
        [-1.0, 1.0],
    ).all():
        raise RuntimeError(
            "source_ob_bias must be ±1"
        )

    far_edge = np.where(
        bias > 0,
        zone_low,
        zone_high,
    )

    distance_price = np.where(
        bias > 0,
        touch_close
        - far_edge,
        far_edge
        - touch_close,
    )

    distance_atr = np.full(
        len(x),
        np.nan,
    )

    ok = (
        np.isfinite(
            distance_price
        )
        & np.isfinite(
            atr
        )
        & (
            atr > 0
        )
    )

    distance_atr[
        ok
    ] = (
        distance_price[
            ok
        ]
        / atr[
            ok
        ]
    )

    eligible = (
        x[
            "eligible"
        ]
        .astype(bool)
        .to_numpy()
    )

    bad_eligible = (
        eligible
        & np.isfinite(
            distance_atr
        )
        & (
            distance_atr
            < -1e-10
        )
    )

    if bad_eligible.any():
        raise RuntimeError(
            "eligible OB has negative "
            "far-edge distance"
        )

    tiny_negative = (
        np.isfinite(
            distance_atr
        )
        & (
            distance_atr
            < 0
        )
        & (
            distance_atr
            >= -1e-10
        )
    )

    distance_atr[
        tiny_negative
    ] = 0.0

    x[
        "far_edge_distance_atr5"
    ] = distance_atr

    eligible_finite = (
        eligible
        & np.isfinite(
            distance_atr
        )
    )

    if int(
        eligible_finite.sum()
    ) != EXPECTED_ELIGIBLE:

        raise RuntimeError(
            "eligible geometry coverage "
            f"{int(eligible_finite.sum())} "
            f"!= {EXPECTED_ELIGIBLE}"
        )

    return x


# ============================================================
# Geometry strata: discovery-only unsupervised quartiles
# ============================================================

def add_geometry_strata(
    x: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    list[float],
]:

    x = x.copy()

    discovery = (
        x[
            "split"
        ]
        .eq(
            "DISCOVERY"
        )
        .to_numpy()
    )

    eligible = (
        x[
            "eligible"
        ]
        .astype(bool)
        .to_numpy()
    )

    d = pd.to_numeric(
        x[
            "far_edge_distance_atr5"
        ],
        errors="coerce",
    ).to_numpy(float)

    base = (
        discovery
        & eligible
        & np.isfinite(d)
    )

    values = d[
        base
    ]

    if len(values) < 1000:
        raise RuntimeError(
            "too few discovery geometry "
            "observations"
        )

    q = np.quantile(
        values,
        [
            0.25,
            0.50,
            0.75,
        ],
    ).astype(float)

    if not (
        q[0]
        < q[1]
        < q[2]
    ):
        raise RuntimeError(
            "geometry quartile "
            "cutpoints not unique"
        )

    code = np.digitize(
        d,
        bins=q,
        right=True,
    )

    labels = np.full(
        len(x),
        "UNKNOWN",
        dtype=object,
    )

    finite = np.isfinite(
        d
    )

    for i, label in enumerate(
        GEOMETRY_STRATA
    ):

        labels[
            finite
            & (
                code == i
            )
        ] = label

    x[
        "geometry_stratum"
    ] = labels

    return (
        x,
        q.tolist(),
    )


# ============================================================
# Raw subset summary wrapper
# ============================================================

def subset_summary(
    mask: np.ndarray,
    *,
    horizon: int,
    outcomes: dict,
    weights: np.ndarray,
) -> dict:

    return B.summarize(
        mask,
        horizon=horizon,
        outcomes=outcomes,
        weights=weights,
    )


# ============================================================
# Geometry-standardized effect
# ============================================================

def adjusted_effect(
    x: pd.DataFrame,
    *,
    factor: str,
    bucket: str,
    horizon: int,
    split: str,
    outcomes: dict,
    weights: np.ndarray,
    symbol: str | None = None,
    min_stratum_n: int,
    min_strata: int,
    min_coverage: float,
) -> tuple[
    dict,
    list[dict],
]:

    factor_known = (
        x[
            factor
        ]
        .isin(
            FACTOR_BUCKETS[
                factor
            ]
        )
        .to_numpy()
    )

    bucket_eq = (
        x[
            factor
        ]
        .astype(str)
        .eq(
            str(bucket)
        )
        .to_numpy()
    )

    pool = (
        x[
            "eligible"
        ]
        .astype(bool)
        .to_numpy()
        & x[
            "split"
        ]
        .eq(
            split
        )
        .to_numpy()
        & factor_known
    )

    if symbol is not None:

        pool &= (
            x[
                "symbol"
            ]
            .astype(str)
            .eq(
                symbol
            )
            .to_numpy()
        )

    bucket_mask = (
        pool
        & bucket_eq
    )

    rest_mask = (
        pool
        & ~bucket_eq
    )

    raw_bucket = subset_summary(
        bucket_mask,
        horizon=horizon,
        outcomes=outcomes,
        weights=weights,
    )

    raw_rest = subset_summary(
        rest_mask,
        horizon=horizon,
        outcomes=outcomes,
        weights=weights,
    )

    raw_delta = finite_delta_pp(
        raw_bucket[
            "invalidation_rate"
        ],
        raw_rest[
            "invalidation_rate"
        ],
    )

    analyzable = outcomes[
        horizon
    ][
        "analyzable"
    ]

    pool_analyzable = (
        pool
        & analyzable
    )

    total_pool_weight = float(
        weights[
            pool_analyzable
        ].sum()
    )

    strata_rows = []

    included_rows = []

    for stratum in GEOMETRY_STRATA:

        s = (
            x[
                "geometry_stratum"
            ]
            .eq(
                stratum
            )
            .to_numpy()
        )

        bmask = (
            bucket_mask
            & s
        )

        rmask = (
            rest_mask
            & s
        )

        b = subset_summary(
            bmask,
            horizon=horizon,
            outcomes=outcomes,
            weights=weights,
        )

        r = subset_summary(
            rmask,
            horizon=horizon,
            outcomes=outcomes,
            weights=weights,
        )

        common_pool = (
            pool
            & s
            & analyzable
        )

        common_weight = float(
            weights[
                common_pool
            ].sum()
        )

        delta = finite_delta_pp(
            b[
                "invalidation_rate"
            ],
            r[
                "invalidation_rate"
            ],
        )

        usable = (
            b[
                "analyzable_n"
            ]
            >= min_stratum_n

            and r[
                "analyzable_n"
            ]
            >= min_stratum_n

            and np.isfinite(
                delta
            )
        )

        row = {
            "factor":
                factor,

            "bucket":
                str(bucket),

            "horizon":
                horizon,

            "split":
                split,

            "symbol":
                (
                    symbol
                    if symbol is not None
                    else "POOLED"
                ),

            "geometry_stratum":
                stratum,

            "usable":
                bool(
                    usable
                ),

            "bucket_n":
                int(
                    b[
                        "analyzable_n"
                    ]
                ),

            "rest_n":
                int(
                    r[
                        "analyzable_n"
                    ]
                ),

            "bucket_rate":
                b[
                    "invalidation_rate"
                ],

            "rest_rate":
                r[
                    "invalidation_rate"
                ],

            "delta_pp":
                delta,

            "common_weight":
                common_weight,
        }

        strata_rows.append(
            row
        )

        if usable:
            included_rows.append(
                row
            )

    included_weight = float(
        sum(
            row[
                "common_weight"
            ]
            for row
            in included_rows
        )
    )

    coverage = (
        included_weight
        / total_pool_weight
        if total_pool_weight > 0
        else 0.0
    )

    adjusted_bucket_rate = np.nan
    adjusted_rest_rate = np.nan
    adjusted_delta = np.nan

    if included_weight > 0:

        adjusted_bucket_rate = float(
            sum(
                (
                    row[
                        "common_weight"
                    ]
                    / included_weight
                )
                * float(
                    row[
                        "bucket_rate"
                    ]
                )
                for row
                in included_rows
            )
        )

        adjusted_rest_rate = float(
            sum(
                (
                    row[
                        "common_weight"
                    ]
                    / included_weight
                )
                * float(
                    row[
                        "rest_rate"
                    ]
                )
                for row
                in included_rows
            )
        )

        adjusted_delta = (
            100.0
            * (
                adjusted_bucket_rate
                - adjusted_rest_rate
            )
        )

    adjustment_usable = (
        len(
            included_rows
        )
        >= min_strata
        and coverage
        >= min_coverage
    )

    summary = {
        "factor":
            factor,

        "bucket":
            str(bucket),

        "horizon":
            horizon,

        "split":
            split,

        "symbol":
            (
                symbol
                if symbol is not None
                else "POOLED"
            ),

        "bucket_analyzable_n":
            int(
                raw_bucket[
                    "analyzable_n"
                ]
            ),

        "rest_analyzable_n":
            int(
                raw_rest[
                    "analyzable_n"
                ]
            ),

        "raw_bucket_rate":
            raw_bucket[
                "invalidation_rate"
            ],

        "raw_rest_rate":
            raw_rest[
                "invalidation_rate"
            ],

        "raw_delta_pp":
            raw_delta,

        "adjusted_bucket_rate":
            adjusted_bucket_rate,

        "adjusted_rest_rate":
            adjusted_rest_rate,

        "adjusted_delta_pp":
            adjusted_delta,

        "included_strata":
            int(
                len(
                    included_rows
                )
            ),

        "geometry_coverage":
            float(
                coverage
            ),

        "adjustment_usable":
            bool(
                adjustment_usable
            ),
    }

    return (
        summary,
        strata_rows,
    )


# ============================================================
# Geometry profile itself
# ============================================================

def build_geometry_profile(
    x: pd.DataFrame,
    outcomes: dict,
    weights: np.ndarray,
) -> pd.DataFrame:

    rows = []

    eligible = (
        x[
            "eligible"
        ]
        .astype(bool)
        .to_numpy()
    )

    distance = pd.to_numeric(
        x[
            "far_edge_distance_atr5"
        ],
        errors="coerce",
    ).to_numpy(float)

    for horizon in HORIZONS:

        for split in (
            "DISCOVERY",
            "VALIDATION",
        ):

            split_mask = (
                x[
                    "split"
                ]
                .eq(
                    split
                )
                .to_numpy()
            )

            for stratum in GEOMETRY_STRATA:

                s = (
                    x[
                        "geometry_stratum"
                    ]
                    .eq(
                        stratum
                    )
                    .to_numpy()
                )

                mask = (
                    eligible
                    & split_mask
                    & s
                )

                st = subset_summary(
                    mask,
                    horizon=horizon,
                    outcomes=outcomes,
                    weights=weights,
                )

                analyzed = (
                    mask
                    & outcomes[
                        horizon
                    ][
                        "analyzable"
                    ]
                )

                if analyzed.any():

                    mean_distance = weighted_mean(
                        distance[
                            analyzed
                        ],
                        weights[
                            analyzed
                        ],
                    )

                    median_distance = weighted_quantile(
                        distance[
                            analyzed
                        ],
                        0.50,
                        weights[
                            analyzed
                        ],
                    )

                else:

                    mean_distance = np.nan
                    median_distance = np.nan

                rows.append(
                    {
                        "horizon":
                            horizon,

                        "split":
                            split,

                        "geometry_stratum":
                            stratum,

                        "analyzable_n":
                            st[
                                "analyzable_n"
                            ],

                        "mean_distance_atr5":
                            mean_distance,

                        "median_distance_atr5":
                            median_distance,

                        "invalidation_rate":
                            st[
                                "invalidation_rate"
                            ],
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Full adjusted tables
# ============================================================

def build_adjusted_tables(
    x: pd.DataFrame,
    outcomes: dict,
):

    weights = pd.to_numeric(
        x[
            "decision_weight"
        ],
        errors="raise",
    ).to_numpy(float)

    pooled_rows = []
    strata_rows = []
    symbol_rows = []

    symbols = sorted(
        x[
            "symbol"
        ]
        .astype(str)
        .unique()
    )

    for target in TARGETS:

        factor = target[
            "factor"
        ]

        bucket = target[
            "bucket"
        ]

        for horizon in HORIZONS:

            for split in (
                "DISCOVERY",
                "VALIDATION",
            ):

                if split == "DISCOVERY":

                    min_stratum_n = (
                        MIN_DISCOVERY_STRATUM_N
                    )

                else:

                    min_stratum_n = (
                        MIN_VALIDATION_STRATUM_N
                    )

                summary, details = adjusted_effect(
                    x,
                    factor=factor,
                    bucket=bucket,
                    horizon=horizon,
                    split=split,
                    outcomes=outcomes,
                    weights=weights,
                    min_stratum_n=min_stratum_n,
                    min_strata=MIN_ADJUSTED_STRATA,
                    min_coverage=MIN_ADJUSTED_COVERAGE,
                )

                summary[
                    "expected_sign"
                ] = target[
                    "expected_sign"
                ]

                pooled_rows.append(
                    summary
                )

                strata_rows.extend(
                    details
                )

            # Cross-symbol Validation only.
            for symbol in symbols:

                summary, _ = adjusted_effect(
                    x,
                    factor=factor,
                    bucket=bucket,
                    horizon=horizon,
                    split="VALIDATION",
                    outcomes=outcomes,
                    weights=weights,
                    symbol=symbol,
                    min_stratum_n=MIN_SYMBOL_STRATUM_N,
                    min_strata=MIN_SYMBOL_STRATA,
                    min_coverage=MIN_SYMBOL_COVERAGE,
                )

                summary[
                    "expected_sign"
                ] = target[
                    "expected_sign"
                ]

                symbol_rows.append(
                    summary
                )

    return (
        pd.DataFrame(
            pooled_rows
        ),
        pd.DataFrame(
            strata_rows
        ),
        pd.DataFrame(
            symbol_rows
        ),
    )


# ============================================================
# H12 pre-registered screen
# ============================================================

def build_screen(
    pooled: pd.DataFrame,
    by_symbol: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for target in TARGETS:

        factor = target[
            "factor"
        ]

        bucket = target[
            "bucket"
        ]

        expected_sign = int(
            target[
                "expected_sign"
            ]
        )

        d = pooled[
            (
                pooled[
                    "factor"
                ]
                == factor
            )
            & (
                pooled[
                    "bucket"
                ]
                == bucket
            )
            & (
                pooled[
                    "horizon"
                ]
                == PRIMARY_HORIZON
            )
            & (
                pooled[
                    "split"
                ]
                == "DISCOVERY"
            )
        ]

        v = pooled[
            (
                pooled[
                    "factor"
                ]
                == factor
            )
            & (
                pooled[
                    "bucket"
                ]
                == bucket
            )
            & (
                pooled[
                    "horizon"
                ]
                == PRIMARY_HORIZON
            )
            & (
                pooled[
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
                "pooled screen "
                "cardinality drift"
            )

        d = d.iloc[0]
        v = v.iloc[0]

        d_adj = float(
            d[
                "adjusted_delta_pp"
            ]
        )

        v_adj = float(
            v[
                "adjusted_delta_pp"
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

            and bool(
                d[
                    "adjustment_usable"
                ]
            )

            and bool(
                v[
                    "adjustment_usable"
                ]
            )
        )

        discovery_residual_ok = (
            same_expected_sign(
                d_adj,
                expected_sign,
            )
            and abs(
                d_adj
            )
            >= MIN_DISCOVERY_ADJ_DELTA_PP
        )

        validation_residual_ok = (
            same_expected_sign(
                v_adj,
                expected_sign,
            )
            and abs(
                v_adj
            )
            >= MIN_VALIDATION_ADJ_DELTA_PP
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
                == bucket
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
            & (
                ss[
                    "adjustment_usable"
                ]
                .astype(bool)
            )
            & np.isfinite(
                ss[
                    "adjusted_delta_pp"
                ]
            )
        ]

        symbol_available = int(
            len(ss)
        )

        symbol_same_sign = int(
            (
                ss[
                    "adjusted_delta_pp"
                ]
                * expected_sign
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

        elif not discovery_residual_ok:

            status = (
                "COLLAPSES_AFTER_GEOMETRY_CONTROL"
            )

        elif not validation_residual_ok:

            status = (
                "FAIL_VALIDATION"
            )

        elif not cross_symbol_ok:

            status = (
                "FAIL_CROSS_SYMBOL"
            )

        elif expected_sign > 0:

            status = (
                "RESIDUAL_SURVIVES_HIGHER_INVALIDATION"
            )

        else:

            status = (
                "RESIDUAL_SURVIVES_LOWER_INVALIDATION"
            )

        raw_d = float(
            d[
                "raw_delta_pp"
            ]
        )

        raw_v = float(
            v[
                "raw_delta_pp"
            ]
        )

        rows.append(
            {
                "factor":
                    factor,

                "bucket":
                    bucket,

                "expected_sign":
                    expected_sign,

                "status":
                    status,

                "discovery_raw_delta_pp":
                    raw_d,

                "discovery_adjusted_delta_pp":
                    d_adj,

                "discovery_geometry_coverage":
                    d[
                        "geometry_coverage"
                    ],

                "validation_raw_delta_pp":
                    raw_v,

                "validation_adjusted_delta_pp":
                    v_adj,

                "validation_geometry_coverage":
                    v[
                        "geometry_coverage"
                    ],

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
# Geometry sanity summary
# ============================================================

def geometry_h12_summary(
    profile: pd.DataFrame,
) -> dict:

    result = {}

    for split in (
        "DISCOVERY",
        "VALIDATION",
    ):

        z = profile[
            (
                profile[
                    "horizon"
                ]
                == PRIMARY_HORIZON
            )
            & (
                profile[
                    "split"
                ]
                == split
            )
        ].set_index(
            "geometry_stratum"
        ).reindex(
            GEOMETRY_STRATA
        )

        rates = pd.to_numeric(
            z[
                "invalidation_rate"
            ],
            errors="coerce",
        ).to_numpy(float)

        finite = np.isfinite(
            rates
        ).all()

        monotonic = bool(
            finite
            and np.all(
                np.diff(
                    rates
                )
                <= 0
            )
        )

        near_minus_far = (
            float(
                100.0
                * (
                    rates[0]
                    - rates[-1]
                )
            )
            if finite
            else np.nan
        )

        result[
            split.lower()
        ] = {
            "rates":
                {
                    stratum:
                        (
                            float(rate)
                            if np.isfinite(rate)
                            else None
                        )
                    for stratum, rate
                    in zip(
                        GEOMETRY_STRATA,
                        rates,
                    )
                },

            "monotonic_decreasing":
                monotonic,

            "nearest_minus_farthest_pp":
                near_minus_far,
        }

    return result


# ============================================================
# Main
# ============================================================

def main() -> None:

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    x = B.load_input()

    if len(x) != EXPECTED_CANDIDATES:
        raise RuntimeError(
            "candidate count drift"
        )

    if int(
        x[
            "eligible"
        ].sum()
    ) != EXPECTED_ELIGIBLE:
        raise RuntimeError(
            "eligible count drift"
        )

    x = add_surface_factors(
        x
    )

    x, split_day = B.add_split(
        x
    )

    x = attach_touch_geometry(
        x
    )

    x, cutpoints = add_geometry_strata(
        x
    )

    outcomes = B.build_outcomes(
        x
    )

    weights = pd.to_numeric(
        x[
            "decision_weight"
        ],
        errors="raise",
    ).to_numpy(float)

    geometry_profile = build_geometry_profile(
        x,
        outcomes,
        weights,
    )

    (
        pooled,
        strata,
        by_symbol,
    ) = build_adjusted_tables(
        x,
        outcomes,
    )

    screen = build_screen(
        pooled,
        by_symbol,
    )

    geometry_summary = (
        geometry_h12_summary(
            geometry_profile
        )
    )

    geometry_profile.to_csv(
        OUT
        / "geometry_profile.csv",
        index=False,
    )

    pooled.to_csv(
        OUT
        / "adjusted_relationships.csv",
        index=False,
    )

    strata.to_csv(
        OUT
        / "stratum_details.csv",
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

    audit = {
        "experiment":
            "ob_geometry_control_v1",

        "git_head":
            git_head(),

        "candidate_count":
            int(
                len(x)
            ),

        "eligible_candidates":
            int(
                x[
                    "eligible"
                ].sum()
            ),

        "geometry_definition":
            (
                "directional touch-close "
                "distance to source-OB "
                "far edge / touch ATR5"
            ),

        "geometry_cutpoint_source":
            (
                "eligible discovery "
                "candidates only; "
                "outcome-blind quartiles"
            ),

        "geometry_cutpoints":
            cutpoints,

        "geometry_strata":
            list(
                GEOMETRY_STRATA
            ),

        "targets":
            list(
                TARGETS
            ),

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

        "thresholds": {
            "min_discovery_n":
                MIN_DISCOVERY_N,

            "min_validation_n":
                MIN_VALIDATION_N,

            "min_discovery_stratum_n":
                MIN_DISCOVERY_STRATUM_N,

            "min_validation_stratum_n":
                MIN_VALIDATION_STRATUM_N,

            "min_adjusted_strata":
                MIN_ADJUSTED_STRATA,

            "min_adjusted_coverage":
                MIN_ADJUSTED_COVERAGE,

            "min_discovery_adj_delta_pp":
                MIN_DISCOVERY_ADJ_DELTA_PP,

            "min_validation_adj_delta_p":
                MIN_VALIDATION_ADJ_DELTA_PP,

            "min_symbol_n":
                MIN_SYMBOL_N,

            "min_symbol_stratum_n":
                MIN_SYMBOL_STRATUM_N,

            "min_symbol_strata":
                MIN_SYMBOL_STRATA,

            "min_symbol_coverage":
                MIN_SYMBOL_COVERAGE,

            "min_symbols_same_sign":
                MIN_SYMBOLS_SAME_SIGN,
        },

        "geometry_h12":
            geometry_summary,

        "geometry_profile_rows":
            int(
                len(
                    geometry_profile
                )
            ),

        "adjusted_relationship_rows":
            int(
                len(
                    pooled
                )
            ),

        "stratum_rows":
            int(
                len(
                    strata
                )
            ),

        "symbol_rows":
            int(
                len(
                    by_symbol
                )
            ),

        "screen_rows":
            int(
                len(
                    screen
                )
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
        "OB_GEOMETRY_CONTROL_V1_DONE"
    )

    print()
    print(
        "GEOMETRY_CUTPOINTS"
    )
    print(
        cutpoints
    )

    print()
    print(
        "H12_GEOMETRY"
    )

    print(
        json.dumps(
            geometry_summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print(
        "SCREEN"
    )

    print(
        screen[
            [
                "factor",
                "bucket",
                "status",
                "discovery_raw_delta_pp",
                "discovery_adjusted_delta_pp",
                "validation_raw_delta_pp",
                "validation_adjusted_delta_pp",
                "validation_geometry_coverage",
                "symbol_available",
                "symbol_same_sign",
            ]
        ]
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
