#!/usr/bin/env python3

"""
OB Path Relationship V1

Question
--------
At an OB touch, do a small number of interpretable causal conditions
change the subsequent H12 path/payoff distribution in a repeatable way?

This is descriptive relationship discovery, NOT a strategy model.

Fixed:
- corrected causal RL state
- frozen V3 future path
- H12 only
- FOLLOW / FADE
- stop = 1 ATR
- target = 2R
- conservative same-bar policy
- original decision_weight unchanged
- whole-trading-day 70/30 discovery/validation split
- no ML
- no tuning
- no bootstrap
- no feature combinations
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


from research.audit_ob_candidate_v3_posthoc import (  # noqa: E402
    load_full_or_chunks,
)

from research.analyze_ob_candidate_v3_phase1 import (  # noqa: E402
    build_path_arrays,
    rotated,
    simulate,
    apply_policy,
    weighted_mean,
    weighted_quantile,
    agg_rr,
)

from research.ob_rl_dataset_v0_spec import (  # noqa: E402
    VALIDATED_TFS,
    STOP_ATR,
    PRIMARY_HORIZON,
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
    / "ob_path_relationship_v1"
)

EXPECTED_CANDIDATES = 21_481

TARGET_R = 2.0

HORIZON = PRIMARY_HORIZON

POLICY = "conservative"

DISCOVERY_FRACTION = 0.70


# ============================================================
# Minimal survival thresholds
# ============================================================

MIN_DISCOVERY_N = 200

MIN_VALIDATION_N = 100

MIN_SYMBOL_N = 30

MIN_DISCOVERY_DELTA_R = 0.05

MIN_VALIDATION_DELTA_R = 0.02

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

    "touch_behavior": (
        "NO_BREACH",
        "BREACH_RECLAIM",
        "CLOSE_BEYOND",
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


DIRECTIONS = (
    "FOLLOW",
    "FADE",
)


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


def finite_delta(
    a,
    b,
) -> float:

    if (
        pd.isna(a)
        or pd.isna(b)
    ):
        return np.nan

    return float(
        a - b
    )


def prefix_dict(
    d: dict,
    prefix: str,
) -> dict:

    return {
        f"{prefix}_{k}":
            v
        for k, v
        in d.items()
    }


# ============================================================
# State load / hard checks
# ============================================================

def load_state() -> pd.DataFrame:

    needed = [
        "candidate_id",
        "symbol",
        "trading_day",
        "source_tf",
        "source_ob_structure",
        "source_ob_bias",
        "touch_ordinal",
        "touch_behavior",
        "quant_state",
        "decision_weight",
    ]

    for tf in VALIDATED_TFS:

        needed.extend(
            [
                f"internal_bias_{tf}",
                f"swing_bias_{tf}",
            ]
        )

    if not STATE_PARQUET.exists():
        raise RuntimeError(
            f"missing corrected state parquet: "
            f"{STATE_PARQUET}"
        )

    state = pd.read_parquet(
        STATE_PARQUET,
        columns=needed,
    )

    if len(state) != EXPECTED_CANDIDATES:
        raise RuntimeError(
            "candidate cardinality drift: "
            f"{len(state)} != "
            f"{EXPECTED_CANDIDATES}"
        )

    if not state[
        "candidate_id"
    ].is_unique:
        raise RuntimeError(
            "candidate_id is not unique"
        )

    w = pd.to_numeric(
        state[
            "decision_weight"
        ],
        errors="coerce",
    )

    if (
        w.isna().any()
        or (w <= 0).any()
    ):
        raise RuntimeError(
            "invalid decision_weight"
        )

    bias = pd.to_numeric(
        state[
            "source_ob_bias"
        ],
        errors="coerce",
    )

    if not bias.isin(
        [-1, 1]
    ).all():
        raise RuntimeError(
            "source_ob_bias must be ±1"
        )

    return state


# ============================================================
# Derived INTERPRETABLE factors only
# ============================================================

def add_factors(
    state: pd.DataFrame,
) -> pd.DataFrame:

    x = state.copy()

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
        "touch_behavior"
    ] = (
        x[
            "touch_behavior"
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

            raw = (
                pd.to_numeric(
                    x[
                        f"{kind}_bias_{tf}"
                    ],
                    errors="coerce",
                )
                .to_numpy(float)
            )

            rel.append(
                raw
                * source_bias
            )

        rel_matrix = np.column_stack(
            rel
        )

        x[
            f"smc_{kind}_align_count"
        ] = (
            rel_matrix
            == 1
        ).sum(
            axis=1
        ).astype(int)

    return x


# ============================================================
# Whole-trading-day split
# ============================================================

def add_split(
    state: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Timestamp]:

    x = state.copy()

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
            ]
            .unique()
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
        max(
            cut,
            1,
        ),
        len(days) - 1,
    )

    split_day = pd.Timestamp(
        days[
            cut
        ]
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
            "same trading day leaked "
            "across split"
        )

    return (
        x,
        split_day,
    )


# ============================================================
# Future path outcomes
# ============================================================

def build_outcomes(
    state: pd.DataFrame,
) -> dict:

    path = load_full_or_chunks(
        "path"
    )

    arr = build_path_arrays(
        state[
            [
                "candidate_id",
            ]
        ],
        path,
    )

    contig = arr[
        "contig"
    ]

    source_bias = (
        pd.to_numeric(
            state[
                "source_ob_bias"
            ],
            errors="raise",
        )
        .to_numpy(float)
    )

    signs = {
        "FOLLOW":
            source_bias,

        "FADE":
            -source_bias,
    }

    out = {}

    for (
        direction,
        sign,
    ) in signs.items():

        rot = rotated(
            arr,
            sign,
        )

        hi = (
            rot[
                "hi"
            ][
                :,
                :HORIZON,
            ]
        )

        lo = (
            rot[
                "lo"
            ][
                :,
                :HORIZON,
            ]
        )

        ct = (
            contig[
                :,
                :HORIZON,
            ]
        )

        excursion_ok = (
            np.isfinite(
                hi
            ).all(
                axis=1
            )
            & np.isfinite(
                lo
            ).all(
                axis=1
            )
            & ct.all(
                axis=1
            )
        )

        mfe = np.full(
            len(state),
            np.nan,
        )

        mae = np.full(
            len(state),
            np.nan,
        )

        if excursion_ok.any():

            mfe[
                excursion_ok
            ] = np.maximum(
                np.max(
                    hi[
                        excursion_ok
                    ],
                    axis=1,
                ),
                0.0,
            )

            mae[
                excursion_ok
            ] = np.maximum(
                -np.min(
                    lo[
                        excursion_ok
                    ],
                    axis=1,
                ),
                0.0,
            )

        base, code = simulate(
            rot,
            contig,
            STOP_ATR,
            TARGET_R,
            HORIZON,
            True,
        )

        reward = apply_policy(
            base,
            code,
            TARGET_R,
            POLICY,
        )

        out[
            direction
        ] = {
            "mfe":
                mfe,

            "mae":
                mae,

            "excursion_ok":
                excursion_ok,

            "reward":
                reward,

            "code":
                code,
        }

    return out


# ============================================================
# One mask -> path/payoff distribution
# ============================================================

def summarize(
    mask: np.ndarray,
    *,
    direction: str,
    outcomes: dict,
    weights: np.ndarray,
) -> dict:

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    idx = np.flatnonzero(
        mask
    )

    o = outcomes[
        direction
    ]

    rr = agg_rr(
        o[
            "reward"
        ],
        o[
            "code"
        ],
        weights,
        idx,
    )

    exc_mask = (
        mask
        & o[
            "excursion_ok"
        ]
    )

    n_exc = int(
        exc_mask.sum()
    )

    if n_exc:

        w_exc = weights[
            exc_mask
        ]

        mfe = o[
            "mfe"
        ][
            exc_mask
        ]

        mae = o[
            "mae"
        ][
            exc_mask
        ]

        mean_mfe = weighted_mean(
            mfe,
            w_exc,
        )

        mean_mae = weighted_mean(
            mae,
            w_exc,
        )

        median_mfe = weighted_quantile(
            mfe,
            0.50,
            w_exc,
        )

        median_mae = weighted_quantile(
            mae,
            0.50,
            w_exc,
        )

    else:

        mean_mfe = np.nan
        mean_mae = np.nan
        median_mfe = np.nan
        median_mae = np.nan

    return {
        "candidate_n":
            int(
                mask.sum()
            ),

        "n_excursion":
            n_exc,

        "mean_mfe_atr":
            mean_mfe,

        "median_mfe_atr":
            median_mfe,

        "mean_mae_atr":
            mean_mae,

        "median_mae_atr":
            median_mae,

        "mean_net_excursion_atr":
            (
                mean_mfe
                - mean_mae
                if (
                    np.isfinite(
                        mean_mfe
                    )
                    and np.isfinite(
                        mean_mae
                    )
                )
                else np.nan
            ),

        "n_rr":
            (
                int(
                    rr[
                        "n"
                    ]
                )
                if rr
                else 0
            ),

        "n_weighted_rr":
            (
                float(
                    rr[
                        "n_weighted"
                    ]
                )
                if rr
                else 0.0
            ),

        "mean_R_2R":
            (
                float(
                    rr[
                        "mean_R"
                    ]
                )
                if rr
                else np.nan
            ),

        "median_R_2R":
            (
                float(
                    rr[
                        "median_R"
                    ]
                )
                if rr
                else np.nan
            ),

        "target_hit_rate":
            (
                float(
                    rr[
                        "target_hit_rate"
                    ]
                )
                if rr
                else np.nan
            ),

        "stop_hit_rate":
            (
                float(
                    rr[
                        "stop_hit_rate"
                    ]
                )
                if rr
                else np.nan
            ),

        "timeout_rate":
            (
                float(
                    rr[
                        "timeout_rate"
                    ]
                )
                if rr
                else np.nan
            ),

        "both_hit_rate":
            (
                float(
                    rr[
                        "both_hit_rate"
                    ]
                )
                if rr
                else np.nan
            ),
    }


# ============================================================
# Relationship tables
# ============================================================

def build_relationship_tables(
    state: pd.DataFrame,
    outcomes: dict,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    weights = (
        pd.to_numeric(
            state[
                "decision_weight"
            ],
            errors="raise",
        )
        .to_numpy(float)
    )

    baseline_rows = []

    relationship_rows = []

    symbol_rows = []

    for split in (
        "DISCOVERY",
        "VALIDATION",
    ):

        split_mask = (
            state[
                "split"
            ]
            .eq(
                split
            )
            .to_numpy()
        )

        for direction in DIRECTIONS:

            st = summarize(
                split_mask,
                direction=direction,
                outcomes=outcomes,
                weights=weights,
            )

            baseline_rows.append(
                {
                    "split":
                        split,

                    "direction":
                        direction,

                    **st,
                }
            )

    for (
        factor,
        buckets,
    ) in FACTOR_BUCKETS.items():

        known = (
            state[
                factor
            ]
            .isin(
                buckets
            )
            .to_numpy()
        )

        for bucket in buckets:

            bucket_eq = (
                state[
                    factor
                ]
                .eq(
                    bucket
                )
                .to_numpy()
            )

            for split in (
                "DISCOVERY",
                "VALIDATION",
            ):

                split_mask = (
                    state[
                        "split"
                    ]
                    .eq(
                        split
                    )
                    .to_numpy()
                )

                eligible = (
                    split_mask
                    & known
                )

                bucket_mask = (
                    eligible
                    & bucket_eq
                )

                rest_mask = (
                    eligible
                    & ~bucket_eq
                )

                for direction in DIRECTIONS:

                    bucket_st = summarize(
                        bucket_mask,
                        direction=direction,
                        outcomes=outcomes,
                        weights=weights,
                    )

                    rest_st = summarize(
                        rest_mask,
                        direction=direction,
                        outcomes=outcomes,
                        weights=weights,
                    )

                    relationship_rows.append(
                        {
                            "factor":
                                factor,

                            "bucket":
                                str(
                                    bucket
                                ),

                            "split":
                                split,

                            "direction":
                                direction,

                            **prefix_dict(
                                bucket_st,
                                "bucket",
                            ),

                            **prefix_dict(
                                rest_st,
                                "rest",
                            ),

                            "delta_mean_R_2R":
                                finite_delta(
                                    bucket_st[
                                        "mean_R_2R"
                                    ],
                                    rest_st[
                                        "mean_R_2R"
                                    ],
                                ),

                            "delta_target_hit_pp":
                                (
                                    100.0
                                    * finite_delta(
                                        bucket_st[
                                            "target_hit_rate"
                                        ],
                                        rest_st[
                                            "target_hit_rate"
                                        ],
                                    )
                                ),

                            "delta_mean_mfe_atr":
                                finite_delta(
                                    bucket_st[
                                        "mean_mfe_atr"
                                    ],
                                    rest_st[
                                        "mean_mfe_atr"
                                    ],
                                ),

                            "delta_mean_mae_atr":
                                finite_delta(
                                    bucket_st[
                                        "mean_mae_atr"
                                    ],
                                    rest_st[
                                        "mean_mae_atr"
                                    ],
                                ),
                        }
                    )

            # Cross-symbol VALIDATION only.
            for direction in DIRECTIONS:

                for symbol in sorted(
                    state[
                        "symbol"
                    ]
                    .astype(str)
                    .unique()
                ):

                    symbol_mask = (
                        state[
                            "symbol"
                        ]
                        .astype(str)
                        .eq(
                            symbol
                        )
                        .to_numpy()
                    )

                    validation_mask = (
                        state[
                            "split"
                        ]
                        .eq(
                            "VALIDATION"
                        )
                        .to_numpy()
                    )

                    eligible = (
                        validation_mask
                        & symbol_mask
                        & known
                    )

                    bucket_mask = (
                        eligible
                        & bucket_eq
                    )

                    rest_mask = (
                        eligible
                        & ~bucket_eq
                    )

                    bucket_st = summarize(
                        bucket_mask,
                        direction=direction,
                        outcomes=outcomes,
                        weights=weights,
                    )

                    rest_st = summarize(
                        rest_mask,
                        direction=direction,
                        outcomes=outcomes,
                        weights=weights,
                    )

                    symbol_rows.append(
                        {
                            "factor":
                                factor,

                            "bucket":
                                str(
                                    bucket
                                ),

                            "direction":
                                direction,

                            "symbol":
                                symbol,

                            "bucket_n_rr":
                                bucket_st[
                                    "n_rr"
                                ],

                            "rest_n_rr":
                                rest_st[
                                    "n_rr"
                                ],

                            "bucket_mean_R_2R":
                                bucket_st[
                                    "mean_R_2R"
                                ],

                            "rest_mean_R_2R":
                                rest_st[
                                    "mean_R_2R"
                                ],

                            "delta_mean_R_2R":
                                finite_delta(
                                    bucket_st[
                                        "mean_R_2R"
                                    ],
                                    rest_st[
                                        "mean_R_2R"
                                    ],
                                ),

                            "delta_target_hit_pp":
                                (
                                    100.0
                                    * finite_delta(
                                        bucket_st[
                                            "target_hit_rate"
                                        ],
                                        rest_st[
                                            "target_hit_rate"
                                        ],
                                    )
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
# Pre-registered survival screen
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

            for direction in DIRECTIONS:

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
                            "direction"
                        ]
                        == direction
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
                            "direction"
                        ]
                        == direction
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
                        "relationship row "
                        "cardinality drift"
                    )

                d = d.iloc[0]
                v = v.iloc[0]

                d_delta = float(
                    d[
                        "delta_mean_R_2R"
                    ]
                )

                v_delta = float(
                    v[
                        "delta_mean_R_2R"
                    ]
                )

                support_ok = (
                    d[
                        "bucket_n_rr"
                    ]
                    >= MIN_DISCOVERY_N

                    and d[
                        "rest_n_rr"
                    ]
                    >= MIN_DISCOVERY_N

                    and v[
                        "bucket_n_rr"
                    ]
                    >= MIN_VALIDATION_N

                    and v[
                        "rest_n_rr"
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
                    >= MIN_DISCOVERY_DELTA_R
                )

                validation_ok = (
                    np.isfinite(
                        v_delta
                    )
                    and abs(
                        v_delta
                    )
                    >= MIN_VALIDATION_DELTA_R

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
                            "direction"
                        ]
                        == direction
                    )
                ].copy()

                ss = ss[
                    (
                        ss[
                            "bucket_n_rr"
                        ]
                        >= MIN_SYMBOL_N
                    )
                    & (
                        ss[
                            "rest_n_rr"
                        ]
                        >= MIN_SYMBOL_N
                    )
                    & np.isfinite(
                        ss[
                            "delta_mean_R_2R"
                        ]
                    )
                ]

                symbol_available = int(
                    len(
                        ss
                    )
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
                                "delta_mean_R_2R"
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
                        "SURVIVES_POSITIVE"
                    )

                else:

                    status = (
                        "SURVIVES_NEGATIVE"
                    )

                rows.append(
                    {
                        "factor":
                            factor,

                        "bucket":
                            bucket_label,

                        "direction":
                            direction,

                        "status":
                            status,

                        "discovery_delta_R":
                            d_delta,

                        "validation_delta_R":
                            v_delta,

                        "discovery_delta_target_hit_pp":
                            d[
                                "delta_target_hit_pp"
                            ],

                        "validation_delta_target_hit_pp":
                            v[
                                "delta_target_hit_pp"
                            ],

                        "discovery_bucket_n":
                            int(
                                d[
                                    "bucket_n_rr"
                                ]
                            ),

                        "validation_bucket_n":
                            int(
                                v[
                                    "bucket_n_rr"
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

    state = load_state()

    state = add_factors(
        state
    )

    state, split_day = add_split(
        state
    )

    outcomes = build_outcomes(
        state
    )

    (
        baseline,
        relationships,
        by_symbol,
    ) = build_relationship_tables(
        state,
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

    stable = screen[
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
            "ob_path_relationship_v1",

        "git_head":
            git_head(),

        "candidate_count":
            int(
                len(
                    state
                )
            ),

        "split_method":
            "whole_trading_day_70_30",

        "split_day":
            split_day.isoformat(),

        "horizon":
            HORIZON,

        "stop_atr":
            STOP_ATR,

        "target_R":
            TARGET_R,

        "same_bar_policy":
            POLICY,

        "directions":
            list(
                DIRECTIONS
            ),

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

            "min_discovery_delta_R":
                MIN_DISCOVERY_DELTA_R,

            "min_validation_delta_R":
                MIN_VALIDATION_DELTA_R,

            "min_symbols_same_sign":
                MIN_SYMBOLS_SAME_SIGN,
        },

        "relationship_rows":
            int(
                len(
                    relationships
                )
            ),

        "screen_rows":
            int(
                len(
                    screen
                )
            ),

        "stable_relationship_count":
            int(
                len(
                    stable
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
        "OB_PATH_RELATIONSHIP_V1_DONE"
    )

    print(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()

    if len(
        stable
    ):

        print(
            "STABLE_RELATIONSHIPS"
        )

        print(
            stable[
                [
                    "factor",
                    "bucket",
                    "direction",
                    "status",
                    "discovery_delta_R",
                    "validation_delta_R",
                    "symbol_available",
                    "symbol_same_sign",
                ]
            ]
            .to_string(
                index=False
            )
        )

    else:

        print(
            "NO_STABLE_RELATIONSHIPS"
        )


if __name__ == "__main__":
    main()
