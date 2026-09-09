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
    EXIT_INSUFFICIENT,
    EXIT_NONCONTIG,
    EXIT_GAP_STOP,
    EXIT_GAP_TARGET,
    EXIT_BOTH,
    EXIT_STOP,
    EXIT_TARGET,
    EXIT_TIMEOUT,
)

from research.ob_rl_dataset_v0_spec import (  # noqa: E402
    ACTIONS,
    parse_action,
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


ACTION_PARQUET = (
    B.DATA_ROOT
    / "ob_rl_action_v0.parquet"
)

TRADE_ACTIONS = tuple(
    a
    for a in ACTIONS
    if a != "SKIP"
)

if len(TRADE_ACTIONS) != 6:
    raise RuntimeError(
        "expected exactly six trade actions"
    )

MIN_ACTION_SELECTION_TRADES = 200


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
# Trading-performance layer
# ============================================================

def weighted_rate(
    mask: np.ndarray,
    w: np.ndarray,
) -> float:

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    w = np.asarray(
        w,
        dtype=float,
    )

    ok = (
        np.isfinite(w)
        & (w > 0)
    )

    if not ok.any():
        return np.nan

    return float(
        np.sum(
            w[
                ok & mask
            ]
        )
        / np.sum(
            w[ok]
        )
    )


def longest_true_run(
    x: np.ndarray,
) -> int:

    best = 0
    cur = 0

    for v in np.asarray(
        x,
        dtype=bool,
    ):

        if v:
            cur += 1
            best = max(
                best,
                cur,
            )
        else:
            cur = 0

    return int(
        best
    )


def load_trade_rewards() -> pd.DataFrame:

    if not ACTION_PARQUET.exists():

        raise RuntimeError(
            "missing corrected action parquet: "
            f"{ACTION_PARQUET}"
        )

    cols = [
        "candidate_id",
        "symbol",
        "touch_time",
        "trading_day",
        "action",
        "decision_weight",
        "gross_R_h12",
        "exit_code_h12",
    ]

    a = pd.read_parquet(
        ACTION_PARQUET,
        columns=cols,
    )

    expected = (
        EXPECTED_CANDIDATES
        * len(ACTIONS)
    )

    if len(a) != expected:

        raise RuntimeError(
            "action parquet cardinality drift: "
            f"{len(a)} != {expected}"
        )

    if a.duplicated(
        [
            "candidate_id",
            "action",
        ]
    ).any():

        raise RuntimeError(
            "duplicate candidate/action"
        )

    got_actions = set(
        a[
            "action"
        ]
        .astype(str)
        .unique()
    )

    if got_actions != set(
        ACTIONS
    ):

        raise RuntimeError(
            "action space drift: "
            f"{sorted(got_actions)}"
        )

    a = a[
        a[
            "action"
        ].isin(
            TRADE_ACTIONS
        )
    ].copy()

    a[
        "touch_time"
    ] = pd.to_datetime(
        a[
            "touch_time"
        ],
        errors="raise",
    )

    a[
        "trading_day"
    ] = pd.to_datetime(
        a[
            "trading_day"
        ],
        errors="raise",
    )

    return a


def daily_curve(
    g: pd.DataFrame,
    all_days: list[pd.Timestamp],
) -> pd.DataFrame:

    rows = []

    for day in all_days:

        z = g[
            g[
                "trading_day"
            ]
            == day
        ]

        if len(z):

            r = pd.to_numeric(
                z[
                    "gross_R_h12"
                ],
                errors="coerce",
            ).to_numpy(float)

            w = pd.to_numeric(
                z[
                    "decision_weight"
                ],
                errors="coerce",
            ).to_numpy(float)

            code = pd.to_numeric(
                z[
                    "exit_code_h12"
                ],
                errors="coerce",
            ).to_numpy(float)

            ok = (
                np.isfinite(r)
                & np.isfinite(w)
                & (w > 0)
                & (
                    code
                    != EXIT_INSUFFICIENT
                )
                & (
                    code
                    != EXIT_NONCONTIG
                )
            )

            if ok.any():

                day_r = weighted_mean(
                    r[ok],
                    w[ok],
                )

                trade_n = int(
                    ok.sum()
                )

            else:

                day_r = 0.0
                trade_n = 0

        else:

            day_r = 0.0
            trade_n = 0

        rows.append(
            {
                "trading_day":
                    day,

                "daily_R":
                    float(
                        day_r
                    ),

                "trade_n":
                    trade_n,
            }
        )

    out = pd.DataFrame(
        rows
    )

    out[
        "cum_R"
    ] = out[
        "daily_R"
    ].cumsum()

    running_peak = np.maximum.accumulate(
        np.maximum(
            out[
                "cum_R"
            ].to_numpy(float),
            0.0,
        )
    )

    out[
        "drawdown_R"
    ] = (
        out[
            "cum_R"
        ].to_numpy(float)
        - running_peak
    )

    return out


def trade_metrics(
    g: pd.DataFrame,
    all_days: list[pd.Timestamp],
) -> tuple[
    dict,
    pd.DataFrame,
]:

    if g.empty:

        return (
            {
                "trades": 0,
                "effective_weight": 0.0,
                "win_rate": np.nan,
                "loss_rate": np.nan,
                "zero_rate": np.nan,
                "avg_win_R": np.nan,
                "avg_loss_R": np.nan,
                "realized_payoff_ratio": np.nan,
                "realized_breakeven_win_rate": np.nan,
                "expectancy_R": np.nan,
                "median_R": np.nan,
                "profit_factor": np.nan,
                "gross_signal_total_R": np.nan,
                "target_hit_rate": np.nan,
                "stop_hit_rate": np.nan,
                "timeout_rate": np.nan,
                "both_hit_rate": np.nan,
                "daily_sharpe_252": np.nan,
                "max_drawdown_R": np.nan,
                "max_losing_streak": 0,
                "active_days": 0,
                "trading_days": len(
                    all_days
                ),
                "active_day_rate": 0.0,
                "trades_per_active_day": np.nan,
            },
            daily_curve(
                g,
                all_days,
            ),
        )

    x = g.copy()

    r = pd.to_numeric(
        x[
            "gross_R_h12"
        ],
        errors="coerce",
    ).to_numpy(float)

    code = pd.to_numeric(
        x[
            "exit_code_h12"
        ],
        errors="coerce",
    ).to_numpy(float)

    w = pd.to_numeric(
        x[
            "decision_weight"
        ],
        errors="coerce",
    ).to_numpy(float)

    analyzed = (
        np.isfinite(r)
        & np.isfinite(code)
        & np.isfinite(w)
        & (w > 0)
        & (
            code
            != EXIT_INSUFFICIENT
        )
        & (
            code
            != EXIT_NONCONTIG
        )
    )

    z = x.loc[
        analyzed
    ].copy()

    rr = r[
        analyzed
    ]

    cc = code[
        analyzed
    ].astype(int)

    ww = w[
        analyzed
    ]

    n = int(
        len(rr)
    )

    curve = daily_curve(
        z,
        all_days,
    )

    if n == 0:

        return (
            {
                "trades": 0,
                "effective_weight": 0.0,
                "win_rate": np.nan,
                "loss_rate": np.nan,
                "zero_rate": np.nan,
                "avg_win_R": np.nan,
                "avg_loss_R": np.nan,
                "realized_payoff_ratio": np.nan,
                "realized_breakeven_win_rate": np.nan,
                "expectancy_R": np.nan,
                "median_R": np.nan,
                "profit_factor": np.nan,
                "gross_signal_total_R": np.nan,
                "target_hit_rate": np.nan,
                "stop_hit_rate": np.nan,
                "timeout_rate": np.nan,
                "both_hit_rate": np.nan,
                "daily_sharpe_252": np.nan,
                "max_drawdown_R": np.nan,
                "max_losing_streak": 0,
                "active_days": 0,
                "trading_days": len(
                    all_days
                ),
                "active_day_rate": 0.0,
                "trades_per_active_day": np.nan,
            },
            curve,
        )

    wins = (
        rr > 0
    )

    losses = (
        rr < 0
    )

    zeros = (
        rr == 0
    )

    avg_win = (
        weighted_mean(
            rr[
                wins
            ],
            ww[
                wins
            ],
        )
        if wins.any()
        else np.nan
    )

    avg_loss = (
        weighted_mean(
            rr[
                losses
            ],
            ww[
                losses
            ],
        )
        if losses.any()
        else np.nan
    )

    payoff = (
        float(
            avg_win
            / abs(
                avg_loss
            )
        )
        if (
            np.isfinite(
                avg_win
            )
            and np.isfinite(
                avg_loss
            )
            and avg_loss < 0
        )
        else np.nan
    )

    realized_be = (
        float(
            abs(
                avg_loss
            )
            / (
                avg_win
                + abs(
                    avg_loss
                )
            )
        )
        if (
            np.isfinite(
                avg_win
            )
            and np.isfinite(
                avg_loss
            )
            and avg_win > 0
            and avg_loss < 0
        )
        else np.nan
    )

    gross_profit = float(
        np.sum(
            ww[
                wins
            ]
            * rr[
                wins
            ]
        )
    )

    gross_loss = float(
        np.sum(
            ww[
                losses
            ]
            * rr[
                losses
            ]
        )
    )

    profit_factor = (
        float(
            gross_profit
            / abs(
                gross_loss
            )
        )
        if gross_loss < 0
        else np.inf
    )

    daily_r = curve[
        "daily_R"
    ].to_numpy(float)

    if (
        len(daily_r) >= 2
        and np.std(
            daily_r,
            ddof=1,
        )
        > 0
    ):

        sharpe = float(
            np.mean(
                daily_r
            )
            / np.std(
                daily_r,
                ddof=1,
            )
            * np.sqrt(
                252.0
            )
        )

    else:

        sharpe = np.nan

    max_dd = float(
        -min(
            0.0,
            float(
                curve[
                    "drawdown_R"
                ].min()
            ),
        )
    )

    order = np.argsort(
        pd.to_datetime(
            z[
                "touch_time"
            ]
        ).to_numpy(
            dtype="datetime64[ns]"
        ),
        kind="stable",
    )

    losing_streak = longest_true_run(
        rr[
            order
        ]
        < 0
    )

    active_days = int(
        (
            curve[
                "trade_n"
            ]
            > 0
        ).sum()
    )

    trading_days = int(
        len(
            curve
        )
    )

    return (
        {
            "trades":
                n,

            "effective_weight":
                float(
                    np.sum(
                        ww
                    )
                ),

            "win_rate":
                weighted_rate(
                    wins,
                    ww,
                ),

            "loss_rate":
                weighted_rate(
                    losses,
                    ww,
                ),

            "zero_rate":
                weighted_rate(
                    zeros,
                    ww,
                ),

            "avg_win_R":
                avg_win,

            "avg_loss_R":
                avg_loss,

            "realized_payoff_ratio":
                payoff,

            "realized_breakeven_win_rate":
                realized_be,

            "expectancy_R":
                weighted_mean(
                    rr,
                    ww,
                ),

            "median_R":
                weighted_quantile(
                    rr,
                    0.50,
                    ww,
                ),

            "profit_factor":
                profit_factor,

            "gross_signal_total_R":
                float(
                    np.sum(
                        ww
                        * rr
                    )
                ),

            "target_hit_rate":
                weighted_rate(
                    (
                        cc
                        == EXIT_TARGET
                    )
                    | (
                        cc
                        == EXIT_GAP_TARGET
                    ),
                    ww,
                ),

            "stop_hit_rate":
                weighted_rate(
                    (
                        cc
                        == EXIT_STOP
                    )
                    | (
                        cc
                        == EXIT_GAP_STOP
                    ),
                    ww,
                ),

            "timeout_rate":
                weighted_rate(
                    cc
                    == EXIT_TIMEOUT,
                    ww,
                ),

            "both_hit_rate":
                weighted_rate(
                    cc
                    == EXIT_BOTH,
                    ww,
                ),

            "daily_sharpe_252":
                sharpe,

            "max_drawdown_R":
                max_dd,

            "max_losing_streak":
                losing_streak,

            "active_days":
                active_days,

            "trading_days":
                trading_days,

            "active_day_rate":
                (
                    active_days
                    / trading_days
                    if trading_days
                    else np.nan
                ),

            "trades_per_active_day":
                (
                    n
                    / active_days
                    if active_days
                    else np.nan
                ),
        },
        curve,
    )


def build_trading_evaluation(
    x: pd.DataFrame,
    screen: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    rewards = load_trade_rewards()

    candidate_meta = x[
        [
            "candidate_id",
            "symbol",
            "trading_day",
            "split",
            "eligible",
            "source_tf",
            "touch_bin",
            "ob_width_bucket",
        ]
    ].copy()

    candidate_meta[
        "trading_day"
    ] = pd.to_datetime(
        candidate_meta[
            "trading_day"
        ],
        errors="raise",
    )

    rewards = rewards.drop(
        columns=[
            "symbol",
            "trading_day",
        ],
        errors="ignore",
    )

    z = rewards.merge(
        candidate_meta,
        on="candidate_id",
        how="left",
        validate="many_to_one",
    )

    if z[
        "split"
    ].isna().any():

        raise RuntimeError(
            "trade reward/meta merge incomplete"
        )

    all_action_rows = []

    selected_rows = []

    curve_rows = []

    symbol_rows = []

    screen_key = (
        screen.set_index(
            [
                "factor",
                "bucket",
            ]
        )[
            "status"
        ]
        .to_dict()
    )

    split_days = {}

    for split in (
        "DISCOVERY",
        "VALIDATION",
    ):

        split_days[
            split
        ] = [
            pd.Timestamp(
                d
            )
            for d in sorted(
                candidate_meta.loc[
                    candidate_meta[
                        "split"
                    ]
                    == split,
                    "trading_day",
                ].unique()
            )
        ]

    symbols = sorted(
        candidate_meta[
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

        known_values = set(
            FACTOR_BUCKETS[
                factor
            ]
        )

        base = z[
            z[
                "eligible"
            ].astype(bool)
            & z[
                factor
            ].isin(
                known_values
            )
        ].copy()

        bucket_base = base[
            base[
                factor
            ]
            .astype(str)
            .eq(
                str(
                    bucket
                )
            )
        ].copy()

        action_discovery = []

        for action in TRADE_ACTIONS:

            mode, target_r = parse_action(
                action
            )

            nominal_be = (
                1.0
                / (
                    1.0
                    + target_r
                )
            )

            for split in (
                "DISCOVERY",
                "VALIDATION",
            ):

                g = bucket_base[
                    (
                        bucket_base[
                            "split"
                        ]
                        == split
                    )
                    & (
                        bucket_base[
                            "action"
                        ]
                        == action
                    )
                ]

                metrics, _ = trade_metrics(
                    g,
                    split_days[
                        split
                    ],
                )

                row = {
                    "factor":
                        factor,

                    "bucket":
                        bucket,

                    "split":
                        split,

                    "action":
                        action,

                    "trade_mode":
                        mode,

                    "target_R":
                        target_r,

                    "nominal_breakeven_win_rate":
                        nominal_be,

                    **metrics,
                }

                all_action_rows.append(
                    row
                )

                if split == "DISCOVERY":

                    action_discovery.append(
                        row
                    )

        candidates = [
            row
            for row in action_discovery
            if (
                row[
                    "trades"
                ]
                >= MIN_ACTION_SELECTION_TRADES

                and np.isfinite(
                    row[
                        "expectancy_R"
                    ]
                )
            )
        ]

        if not candidates:

            selected_rows.append(
                {
                    "factor":
                        factor,

                    "bucket":
                        bucket,

                    "mechanism_status":
                        screen_key[
                            (
                                factor,
                                bucket,
                            )
                        ],

                    "selection_status":
                        "NO_DISCOVERY_ACTION_SUPPORT",
                }
            )

            continue

        chosen = sorted(
            candidates,
            key=lambda r: (
                -float(
                    r[
                        "expectancy_R"
                    ]
                ),
                str(
                    r[
                        "action"
                    ]
                ),
            ),
        )[0]

        action = chosen[
            "action"
        ]

        mode, target_r = parse_action(
            action
        )

        nominal_be = (
            1.0
            / (
                1.0
                + target_r
            )
        )

        packed = {
            "factor":
                factor,

            "bucket":
                bucket,

            "mechanism_status":
                screen_key[
                    (
                        factor,
                        bucket,
                    )
                ],

            "selection_status":
                "DISCOVERY_ACTION_SELECTED",

            "selected_action":
                action,

            "trade_mode":
                mode,

            "target_R":
                target_r,

            "nominal_breakeven_win_rate":
                nominal_be,
        }

        for split in (
            "DISCOVERY",
            "VALIDATION",
        ):

            g = bucket_base[
                (
                    bucket_base[
                        "split"
                    ]
                    == split
                )
                & (
                    bucket_base[
                        "action"
                    ]
                    == action
                )
            ]

            metrics, curve = trade_metrics(
                g,
                split_days[
                    split
                ],
            )

            for key, value in metrics.items():

                packed[
                    f"{split.lower()}_{key}"
                ] = value

            curve = curve.copy()

            curve[
                "factor"
            ] = factor

            curve[
                "bucket"
            ] = bucket

            curve[
                "mechanism_status"
            ] = screen_key[
                (
                    factor,
                    bucket,
                )
            ]

            curve[
                "selected_action"
            ] = action

            curve[
                "split"
            ] = split

            curve_rows.append(
                curve
            )

        validation_rest = base[
            (
                base[
                    "split"
                ]
                == "VALIDATION"
            )
            & (
                base[
                    "action"
                ]
                == action
            )
            & (
                ~base[
                    factor
                ]
                .astype(str)
                .eq(
                    str(
                        bucket
                    )
                )
            )
        ]

        rest_metrics, _ = trade_metrics(
            validation_rest,
            split_days[
                "VALIDATION"
            ],
        )

        for key, value in rest_metrics.items():

            packed[
                f"validation_rest_{key}"
            ] = value

        packed[
            "validation_expectancy_delta_vs_rest_R"
        ] = (
            packed[
                "validation_expectancy_R"
            ]
            - packed[
                "validation_rest_expectancy_R"
            ]
            if (
                np.isfinite(
                    packed[
                        "validation_expectancy_R"
                    ]
                )
                and np.isfinite(
                    packed[
                        "validation_rest_expectancy_R"
                    ]
                )
            )
            else np.nan
        )

        selected_rows.append(
            packed
        )

        for symbol in symbols:

            g = bucket_base[
                (
                    bucket_base[
                        "split"
                    ]
                    == "VALIDATION"
                )
                & (
                    bucket_base[
                        "action"
                    ]
                    == action
                )
                & (
                    bucket_base[
                        "symbol"
                    ]
                    .astype(str)
                    .eq(
                        symbol
                    )
                )
            ]

            metrics, _ = trade_metrics(
                g,
                split_days[
                    "VALIDATION"
                ],
            )

            symbol_rows.append(
                {
                    "factor":
                        factor,

                    "bucket":
                        bucket,

                    "mechanism_status":
                        screen_key[
                            (
                                factor,
                                bucket,
                            )
                        ],

                    "selected_action":
                        action,

                    "symbol":
                        symbol,

                    **metrics,
                }
            )

    all_actions = pd.DataFrame(
        all_action_rows
    )

    selected = pd.DataFrame(
        selected_rows
    )

    curves = (
        pd.concat(
            curve_rows,
            ignore_index=True,
        )
        if curve_rows
        else pd.DataFrame()
    )

    by_symbol = pd.DataFrame(
        symbol_rows
    )

    return (
        all_actions,
        selected,
        curves,
        by_symbol,
    )


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

    (
        all_action_metrics,
        selected_policy_metrics,
        selected_policy_curve,
        selected_policy_by_symbol,
    ) = build_trading_evaluation(
        x,
        screen,
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

    all_action_metrics.to_csv(
        OUT
        / "all_action_trading_metrics.csv",
        index=False,
    )

    selected_policy_metrics.to_csv(
        OUT
        / "selected_policy_metrics.csv",
        index=False,
    )

    selected_policy_curve.to_csv(
        OUT
        / "selected_policy_equity_curve.csv",
        index=False,
    )

    selected_policy_by_symbol.to_csv(
        OUT
        / "selected_policy_by_symbol.csv",
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

        "all_action_metric_rows":
            int(
                len(
                    all_action_metrics
                )
            ),

        "selected_policy_rows":
            int(
                len(
                    selected_policy_metrics
                )
            ),

        "selected_policy_curve_rows":
            int(
                len(
                    selected_policy_curve
                )
            ),

        "selected_policy_symbol_rows":
            int(
                len(
                    selected_policy_by_symbol
                )
            ),

        "trading_metrics": {
            "reward":
                "GROSS_R_H12_ONLY",

            "costs":
                "NO_COMMISSION_NO_SLIPPAGE",

            "action_selection":
                (
                    "max Discovery expectancy_R "
                    "among six frozen trade actions; "
                    "same action frozen for Validation"
                ),

            "trade_actions":
                list(
                    TRADE_ACTIONS
                ),

            "min_discovery_action_trades":
                MIN_ACTION_SELECTION_TRADES,

            "equity_curve":
                (
                    "normalized daily weighted-mean R; "
                    "zero on split trading days "
                    "with no analyzed trade"
                ),

            "sharpe":
                (
                    "mean(daily_R)/std(daily_R)"
                    "*sqrt(252), including zero-trade days"
                ),

            "strategy_claim":
                (
                    "no qualitative strategy verdict "
                    "without Validation win rate, "
                    "realized payoff ratio, "
                    "expectancy_R and sample size"
                ),
        },
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

    print()
    print(
        "SELECTED_POLICY_VALIDATION_METRICS"
    )

    cols = [
        "factor",
        "bucket",
        "mechanism_status",
        "selected_action",
        "validation_trades",
        "validation_win_rate",
        "validation_avg_win_R",
        "validation_avg_loss_R",
        "validation_realized_payoff_ratio",
        "validation_expectancy_R",
        "validation_profit_factor",
        "validation_daily_sharpe_252",
        "validation_max_drawdown_R",
        "validation_max_losing_streak",
        "validation_active_day_rate",
        "validation_expectancy_delta_vs_rest_R",
    ]

    print(
        selected_policy_metrics[
            [
                c
                for c in cols
                if c
                in selected_policy_metrics.columns
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
