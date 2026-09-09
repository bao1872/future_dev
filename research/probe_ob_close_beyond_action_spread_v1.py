#!/usr/bin/env python3

"""
CLOSE_BEYOND paired action-spread probe.

Question
--------
For the SAME OB-touch candidate, after CLOSE_BEYOND:

    FADE reward - FOLLOW reward

is the action preference positive and repeatable?

No model.
No tuning.
No simulator rerun.
No new features.

Uses the already-built causal RL action parquet directly.
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


DATA_ROOT = (
    ROOT
    / "research"
    / "analysis_results"
    / "ob_rl_dataset_v0"
)

ACTION_PARQUET = (
    DATA_ROOT
    / "ob_rl_action_v0.parquet"
)

OUT = (
    ROOT
    / "research"
    / "analysis_results"
    / "ob_close_beyond_action_spread_v1"
)


EXPECTED_CANDIDATES = 21_481

DISCOVERY_FRACTION = 0.70

FOLLOW_ACTION = "FOLLOW_2.0R"

FADE_ACTION = "FADE_2.0R"

HORIZONS = (
    6,
    12,
    24,
)

PRIMARY_HORIZON = 12

MIN_POOLED_N = 100

MIN_SYMBOL_N = 30


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


def weighted_mean(
    x,
    w,
) -> float:

    x = np.asarray(
        x,
        dtype=float,
    )

    w = np.asarray(
        w,
        dtype=float,
    )

    ok = (
        np.isfinite(x)
        & np.isfinite(w)
        & (w > 0)
    )

    if not ok.any():
        return np.nan

    return float(
        np.sum(
            x[ok]
            * w[ok]
        )
        / np.sum(
            w[ok]
        )
    )


def weighted_rate(
    mask,
    w,
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
                ok
                & mask
            ]
        )
        / np.sum(
            w[ok]
        )
    )


# ============================================================
# Load paired action data
# ============================================================

def load_pairs() -> pd.DataFrame:

    if not ACTION_PARQUET.exists():

        raise RuntimeError(
            "missing corrected action parquet: "
            f"{ACTION_PARQUET}"
        )

    reward_cols = [
        f"gross_R_h{h}"
        for h in HORIZONS
    ]

    cols = [
        "candidate_id",
        "symbol",
        "trading_day",
        "source_tf",
        "touch_behavior",
        "decision_weight",
        "action",
        *reward_cols,
    ]

    x = pd.read_parquet(
        ACTION_PARQUET,
        columns=cols,
    )

    x = x[
        x[
            "action"
        ].isin(
            [
                FOLLOW_ACTION,
                FADE_ACTION,
            ]
        )
    ].copy()

    if len(x) != (
        EXPECTED_CANDIDATES
        * 2
    ):
        raise RuntimeError(
            "paired action cardinality drift: "
            f"{len(x)}"
        )

    counts = (
        x.groupby(
            "candidate_id",
            observed=True,
        )
        .size()
    )

    if not (
        counts == 2
    ).all():

        raise RuntimeError(
            "each candidate must have "
            "exactly FOLLOW + FADE"
        )

    for col in (
        "symbol",
        "trading_day",
        "source_tf",
        "touch_behavior",
        "decision_weight",
    ):

        varying = (
            x.groupby(
                "candidate_id",
                observed=True,
            )[col]
            .nunique(
                dropna=False
            )
        )

        if (
            varying > 1
        ).any():

            raise RuntimeError(
                f"{col} varies "
                "within candidate"
            )

    meta = (
        x[
            [
                "candidate_id",
                "symbol",
                "trading_day",
                "source_tf",
                "touch_behavior",
                "decision_weight",
            ]
        ]
        .drop_duplicates(
            "candidate_id"
        )
        .set_index(
            "candidate_id"
        )
    )

    if len(meta) != EXPECTED_CANDIDATES:

        raise RuntimeError(
            "candidate meta drift"
        )

    out = meta.copy()

    for h in HORIZONS:

        reward_col = (
            f"gross_R_h{h}"
        )

        p = x.pivot(
            index="candidate_id",
            columns="action",
            values=reward_col,
        )

        for action in (
            FOLLOW_ACTION,
            FADE_ACTION,
        ):

            if action not in p.columns:

                raise RuntimeError(
                    f"missing action "
                    f"{action} at H{h}"
                )

        out[
            f"follow_R_h{h}"
        ] = (
            p[
                FOLLOW_ACTION
            ]
        )

        out[
            f"fade_R_h{h}"
        ] = (
            p[
                FADE_ACTION
            ]
        )

        out[
            f"delta_R_h{h}"
        ] = (
            out[
                f"fade_R_h{h}"
            ]
            - out[
                f"follow_R_h{h}"
            ]
        )

    return (
        out
        .reset_index()
    )


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

    d = set(
        x.loc[
            x[
                "split"
            ]
            == "DISCOVERY",
            "_day",
        ]
    )

    v = set(
        x.loc[
            x[
                "split"
            ]
            == "VALIDATION",
            "_day",
        ]
    )

    if d & v:

        raise RuntimeError(
            "trading-day leakage"
        )

    return (
        x,
        split_day,
    )


# ============================================================
# Paired summary
# ============================================================

def summarize(
    g: pd.DataFrame,
    horizon: int,
) -> dict:

    follow = (
        pd.to_numeric(
            g[
                f"follow_R_h{horizon}"
            ],
            errors="coerce",
        )
        .to_numpy(float)
    )

    fade = (
        pd.to_numeric(
            g[
                f"fade_R_h{horizon}"
            ],
            errors="coerce",
        )
        .to_numpy(float)
    )

    w = (
        pd.to_numeric(
            g[
                "decision_weight"
            ],
            errors="coerce",
        )
        .to_numpy(float)
    )

    ok = (
        np.isfinite(
            follow
        )
        & np.isfinite(
            fade
        )
        & np.isfinite(
            w
        )
        & (w > 0)
    )

    if not ok.any():

        return {
            "candidate_n":
                int(
                    len(g)
                ),

            "paired_n":
                0,

            "paired_weight":
                0.0,

            "mean_follow_R":
                np.nan,

            "mean_fade_R":
                np.nan,

            "mean_delta_R":
                np.nan,

            "fade_better_rate":
                np.nan,

            "follow_better_rate":
                np.nan,

            "tie_rate":
                np.nan,
        }

    follow = follow[ok]

    fade = fade[ok]

    w = w[ok]

    delta = (
        fade
        - follow
    )

    return {
        "candidate_n":
            int(
                len(g)
            ),

        "paired_n":
            int(
                ok.sum()
            ),

        "paired_weight":
            float(
                w.sum()
            ),

        "mean_follow_R":
            weighted_mean(
                follow,
                w,
            ),

        "mean_fade_R":
            weighted_mean(
                fade,
                w,
            ),

        "mean_delta_R":
            weighted_mean(
                delta,
                w,
            ),

        "fade_better_rate":
            weighted_rate(
                delta > 0,
                w,
            ),

        "follow_better_rate":
            weighted_rate(
                delta < 0,
                w,
            ),

        "tie_rate":
            weighted_rate(
                np.isclose(
                    delta,
                    0.0,
                ),
                w,
            ),
    }


# ============================================================
# Tables
# ============================================================

def build_tables(
    x: pd.DataFrame,
):

    pooled_rows = []

    symbol_rows = []

    tf_rows = []

    for h in HORIZONS:

        for split in (
            "DISCOVERY",
            "VALIDATION",
        ):

            z = x[
                x[
                    "split"
                ]
                == split
            ]

            for (
                group,
                mask,
            ) in (
                (
                    "CLOSE_BEYOND",
                    z[
                        "touch_behavior"
                    ]
                    .astype(str)
                    .eq(
                        "CLOSE_BEYOND"
                    ),
                ),
                (
                    "NON_CLOSE_BEYOND",
                    ~z[
                        "touch_behavior"
                    ]
                    .astype(str)
                    .eq(
                        "CLOSE_BEYOND"
                    ),
                ),
            ):

                st = summarize(
                    z[
                        mask
                    ],
                    h,
                )

                pooled_rows.append(
                    {
                        "horizon":
                            h,

                        "split":
                            split,

                        "group":
                            group,

                        **st,
                    }
                )

        # Cross-sectional diagnostics:
        # VALIDATION + CLOSE_BEYOND only.
        v = x[
            (
                x[
                    "split"
                ]
                == "VALIDATION"
            )
            & (
                x[
                    "touch_behavior"
                ]
                .astype(str)
                .eq(
                    "CLOSE_BEYOND"
                )
            )
        ]

        for (
            symbol,
            g,
        ) in v.groupby(
            "symbol",
            observed=True,
        ):

            symbol_rows.append(
                {
                    "horizon":
                        h,

                    "symbol":
                        str(
                            symbol
                        ),

                    **summarize(
                        g,
                        h,
                    ),
                }
            )

        for (
            source_tf,
            g,
        ) in v.groupby(
            "source_tf",
            observed=True,
        ):

            tf_rows.append(
                {
                    "horizon":
                        h,

                    "source_tf":
                        str(
                            source_tf
                        ),

                    **summarize(
                        g,
                        h,
                    ),
                }
            )

    return (
        pd.DataFrame(
            pooled_rows
        ),
        pd.DataFrame(
            symbol_rows
        ),
        pd.DataFrame(
            tf_rows
        ),
    )


# ============================================================
# Primary H12 decision
# ============================================================

def build_screen(
    pooled: pd.DataFrame,
    by_symbol: pd.DataFrame,
) -> dict:

    def row(
        split: str,
        group: str,
    ):

        r = pooled[
            (
                pooled[
                    "horizon"
                ]
                == PRIMARY_HORIZON
            )
            & (
                pooled[
                    "split"
                ]
                == split
            )
            & (
                pooled[
                    "group"
                ]
                == group
            )
        ]

        if len(r) != 1:

            raise RuntimeError(
                "primary pooled row "
                "cardinality drift"
            )

        return r.iloc[0]

    d_cb = row(
        "DISCOVERY",
        "CLOSE_BEYOND",
    )

    d_rest = row(
        "DISCOVERY",
        "NON_CLOSE_BEYOND",
    )

    v_cb = row(
        "VALIDATION",
        "CLOSE_BEYOND",
    )

    v_rest = row(
        "VALIDATION",
        "NON_CLOSE_BEYOND",
    )

    discovery_incremental = (
        float(
            d_cb[
                "mean_delta_R"
            ]
        )
        - float(
            d_rest[
                "mean_delta_R"
            ]
        )
    )

    validation_incremental = (
        float(
            v_cb[
                "mean_delta_R"
            ]
        )
        - float(
            v_rest[
                "mean_delta_R"
            ]
        )
    )

    ss = by_symbol[
        by_symbol[
            "horizon"
        ]
        == PRIMARY_HORIZON
    ].copy()

    ss = ss[
        (
            ss[
                "paired_n"
            ]
            >= MIN_SYMBOL_N
        )
        & np.isfinite(
            ss[
                "mean_delta_R"
            ]
        )
    ]

    symbols_available = int(
        len(ss)
    )

    symbols_fade_preferred = int(
        (
            ss[
                "mean_delta_R"
            ]
            > 0
        ).sum()
    )

    support_ok = (
        d_cb[
            "paired_n"
        ]
        >= MIN_POOLED_N

        and v_cb[
            "paired_n"
        ]
        >= MIN_POOLED_N
    )

    pooled_preference_ok = (
        d_cb[
            "mean_delta_R"
        ]
        > 0

        and v_cb[
            "mean_delta_R"
        ]
        > 0
    )

    incremental_ok = (
        discovery_incremental
        > 0

        and validation_incremental
        > 0
    )

    cross_symbol_ok = (
        symbols_available
        >= 3

        and symbols_fade_preferred
        >= 3
    )

    if (
        support_ok
        and pooled_preference_ok
        and incremental_ok
        and cross_symbol_ok
    ):

        status = (
            "ACTION_PREFERENCE_SURVIVES"
        )

    else:

        status = (
            "ACTION_PREFERENCE_FAIL"
        )

    return {
        "status":
            status,

        "discovery_close_beyond_delta_R":
            float(
                d_cb[
                    "mean_delta_R"
                ]
            ),

        "validation_close_beyond_delta_R":
            float(
            v_cb[
                "mean_delta_R"
            ]
        ),

        "discovery_rest_delta_R":
            float(
                d_rest[
                    "mean_delta_R"
                ]
            ),

        "validation_rest_delta_R":
            float(
                v_rest[
                    "mean_delta_R"
                ]
            ),

        "discovery_incremental_delta_R":
            discovery_incremental,

        "validation_incremental_delta_R":
            validation_incremental,

        "discovery_paired_n":
            int(
                d_cb[
                    "paired_n"
                ]
            ),

        "validation_paired_n":
            int(
                v_cb[
                    "paired_n"
                ]
            ),

        "symbols_available":
            symbols_available,

        "symbols_fade_preferred":
            symbols_fade_preferred,
    }


# ============================================================
# Main
# ============================================================

def main() -> None:

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs = load_pairs()

    pairs, split_day = add_split(
        pairs
    )

    (
        pooled,
        by_symbol,
        by_source_tf,
    ) = build_tables(
        pairs
    )

    screen = build_screen(
        pooled,
        by_symbol,
    )

    pooled.to_csv(
        OUT
        / "pooled.csv",
        index=False,
    )

    by_symbol.to_csv(
        OUT
        / "validation_by_symbol.csv",
        index=False,
    )

    by_source_tf.to_csv(
        OUT
        / "validation_by_source_tf.csv",
        index=False,
    )

    (
        OUT
        / "screen.json"
    ).write_text(
        json.dumps(
            screen,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    audit = {
        "experiment":
            "ob_close_beyond_action_spread_v1",

        "git_head":
            git_head(),

        "candidate_count":
            int(
                len(
                    pairs
                )
            ),

        "split_method":
            "whole_trading_day_70_30",

        "split_day":
            split_day.isoformat(),

        "actions": {
            "follow":
                FOLLOW_ACTION,

            "fade":
                FADE_ACTION,
        },

        "horizons":
            list(
                HORIZONS
            ),

        "primary_horizon":
            PRIMARY_HORIZON,

        "screen":
            screen,
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
        "OB_CLOSE_BEYOND_ACTION_SPREAD_DONE"
    )

    print(
        json.dumps(
            screen,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()

    h12_symbols = (
        by_symbol[
            by_symbol[
                "horizon"
            ]
            == PRIMARY_HORIZON
        ]
        [
            [
                "symbol",
                "paired_n",
                "mean_follow_R",
                "mean_fade_R",
                "mean_delta_R",
                "fade_better_rate",
            ]
        ]
    )

    print(
        "H12_VALIDATION_BY_SYMBOL"
    )

    print(
        h12_symbols
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
