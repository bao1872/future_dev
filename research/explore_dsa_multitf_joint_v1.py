#!/usr/bin/env python3

"""
Multi-timeframe DSA joint-state exploration V1.

Question
--------
At one OB-touch decision event, how does the JOINT 5m + 15m + 1h
DSA-VWAP environment change FOLLOW vs FADE payoff?

This is exploratory only:
- H12 only
- no optimization
- no ML
- no bootstrap
- no threshold tuning
- no strategy search

The environment state is defined before reward is examined.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    ROOT
    / "research"
    / "analysis_results"
    / "ob_rl_dataset_v0"
)

PARQUET = (
    DATASET_ROOT
    / "ob_rl_action_v0.parquet"
)

OUT = (
    ROOT
    / "research"
    / "analysis_results"
    / "dsa_multitf_joint_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


TFS = (
    "5m",
    "15m",
    "1h",
)

RRS = (
    1.5,
    2.0,
    2.5,
)

WEIGHT = "decision_weight"
REWARD = "gross_R_h12"

MIN_REPORT_N = 30


# ============================================================
# Weighted mean
# ============================================================

def wmean(
    values,
    weights,
):
    v = np.asarray(
        values,
        dtype=float,
    )

    w = np.asarray(
        weights,
        dtype=float,
    )

    ok = (
        np.isfinite(v)
        & np.isfinite(w)
        & (w > 0)
    )

    if not ok.any():
        return np.nan

    return float(
        np.sum(
            v[ok] * w[ok]
        )
        / np.sum(w[ok])
    )


# ============================================================
# DSA side state
#
# IMPORTANT:
#
# dsa_vwap_dev_rel already means:
#
# raw VWAP deviation
# × FOLLOW trade direction
#
# Therefore:
#
# > 0 = price is on the VWAP side supporting FOLLOW
# < 0 = price is on the VWAP side opposing FOLLOW
#
# We derive state from FOLLOW rows only.
# ============================================================

def side_state(v):
    if pd.isna(v):
        return "?"

    x = float(v)

    if x > 0:
        return "+"

    if x < 0:
        return "-"

    return "0"


# ============================================================
# Load
# ============================================================

required = [
    "candidate_id",
    "symbol",
    "source_tf",
    "source_ob_structure",
    "trade_mode",
    "target_R",
    WEIGHT,
    REWARD,
]

for tf in TFS:
    required.append(
        f"dsa_vwap_dev_rel_{tf}"
    )

df = pd.read_parquet(
    PARQUET,
    columns=required,
)

df = df[
    df["trade_mode"].isin(
        ["FOLLOW", "FADE"]
    )
].copy()


# ============================================================
# Hard integrity checks
# ============================================================

if df.duplicated(
    [
        "candidate_id",
        "trade_mode",
        "target_R",
    ]
).any():
    raise RuntimeError(
        "duplicate candidate × trade_mode × RR"
    )


counts = (
    df.groupby("candidate_id")
    .size()
)

if not (
    counts == 6
).all():
    raise RuntimeError(
        "each candidate must have "
        "FOLLOW/FADE × 3 RR = 6 rows"
    )


# ============================================================
# Verify state invariance across RR
# ============================================================

for tf in TFS:

    feature = (
        f"dsa_vwap_dev_rel_{tf}"
    )

    variation = (
        df.groupby(
            [
                "candidate_id",
                "trade_mode",
            ]
        )[feature]
        .nunique(
            dropna=False
        )
    )

    bad = int(
        (variation > 1).sum()
    )

    if bad:
        raise RuntimeError(
            f"{tf}: state varies across RR: "
            f"{bad}"
        )


# ============================================================
# FOLLOW / FADE weight equality
# ============================================================

weight_check = (
    df.pivot_table(
        index=[
            "candidate_id",
            "target_R",
        ],
        columns="trade_mode",
        values=WEIGHT,
        aggfunc="first",
    )
)

weight_diff = np.abs(
    weight_check["FOLLOW"]
    - weight_check["FADE"]
)

if (
    weight_diff.fillna(0)
    > 1e-15
).any():
    raise RuntimeError(
        "FOLLOW/FADE decision_weight mismatch"
    )


# ============================================================
# Construct one decision-state row per candidate
#
# Use FOLLOW representation because "+" explicitly means
# environment supports FOLLOW direction.
# ============================================================

follow_state = (
    df[
        df["trade_mode"]
        == "FOLLOW"
    ][
        [
            "candidate_id",
            "symbol",
            "source_tf",
            "source_ob_structure",
            WEIGHT,
            *[
                f"dsa_vwap_dev_rel_{tf}"
                for tf in TFS
            ],
        ]
    ]
    .drop_duplicates(
        "candidate_id"
    )
    .copy()
)


for tf in TFS:

    feature = (
        f"dsa_vwap_dev_rel_{tf}"
    )

    follow_state[
        f"side_{tf}"
    ] = (
        follow_state[
            feature
        ]
        .map(side_state)
    )


follow_state[
    "joint_state"
] = (
    follow_state[
        "side_5m"
    ]
    + follow_state[
        "side_15m"
    ]
    + follow_state[
        "side_1h"
    ]
)


follow_state[
    "aligned_count"
] = (
    (
        follow_state[
            "side_5m"
        ]
        == "+"
    ).astype(int)
    +
    (
        follow_state[
            "side_15m"
        ]
        == "+"
    ).astype(int)
    +
    (
        follow_state[
            "side_1h"
        ]
        == "+"
    ).astype(int)
)


follow_state[
    "valid_three_tf_state"
] = (
    follow_state[
        [
            "side_5m",
            "side_15m",
            "side_1h",
        ]
    ]
    .isin(
        ["+", "-"]
    )
    .all(axis=1)
)


# ============================================================
# Build same-candidate FOLLOW vs FADE payoff
# ============================================================

follow = (
    df[
        df["trade_mode"]
        == "FOLLOW"
    ][
        [
            "candidate_id",
            "target_R",
            REWARD,
        ]
    ]
    .rename(
        columns={
            REWARD:
                "follow_R"
        }
    )
)


fade = (
    df[
        df["trade_mode"]
        == "FADE"
    ][
        [
            "candidate_id",
            "target_R",
            REWARD,
        ]
    ]
    .rename(
        columns={
            REWARD:
                "fade_R"
        }
    )
)


pair = (
    follow.merge(
        fade,
        on=[
            "candidate_id",
            "target_R",
        ],
        validate="one_to_one",
    )
    .merge(
        follow_state,
        on="candidate_id",
        validate="many_to_one",
    )
)


pair[
    "delta_R"
] = (
    pd.to_numeric(
        pair["follow_R"],
        errors="coerce",
    )
    -
    pd.to_numeric(
        pair["fade_R"],
        errors="coerce",
    )
)


# ============================================================
# Untuned majority selector
#
# 2 or 3 aligned TFs -> FOLLOW
# 0 or 1 aligned TFs -> FADE
#
# Fixed before reading payoff.
# ============================================================

pair[
    "majority_action"
] = np.where(
    pair[
        "aligned_count"
    ]
    >= 2,
    "FOLLOW",
    "FADE",
)


pair[
    "majority_selected_R"
] = np.where(
    pair[
        "majority_action"
    ]
    == "FOLLOW",
    pair[
        "follow_R"
    ],
    pair[
        "fade_R"
    ],
)


# ============================================================
# Summary helpers
# ============================================================

def summarize_groups(
    frame,
    group_cols,
    *,
    view,
):

    rows = []

    grouped = frame.groupby(
        group_cols,
        dropna=False,
        observed=True,
    )

    for keys, g in grouped:

        if not isinstance(
            keys,
            tuple,
        ):
            keys = (
                keys,
            )

        row = {
            "view":
                view,
        }

        for col, value in zip(
            group_cols,
            keys,
        ):
            row[col] = value

        valid = g[
            np.isfinite(
                pd.to_numeric(
                    g["follow_R"],
                    errors="coerce",
                )
            )
            &
            np.isfinite(
                pd.to_numeric(
                    g["fade_R"],
                    errors="coerce",
                )
            )
        ].copy()

        row[
            "candidate_rows"
        ] = int(
            len(g)
        )

        row[
            "valid_n"
        ] = int(
            len(valid)
        )

        if len(valid):

            row[
                "mean_follow_R"
            ] = wmean(
                valid[
                    "follow_R"
                ],
                valid[
                    WEIGHT
                ],
            )

            row[
                "mean_fade_R"
            ] = wmean(
                valid[
                    "fade_R"
                ],
                valid[
                    WEIGHT
                ],
            )

            row[
                "mean_delta_R"
            ] = wmean(
                valid[
                    "delta_R"
                ],
                valid[
                    WEIGHT
                ],
            )

            row[
                "majority_selected_R"
            ] = wmean(
                valid[
                    "majority_selected_R"
                ],
                valid[
                    WEIGHT
                ],
            )

        else:

            row[
                "mean_follow_R"
            ] = np.nan

            row[
                "mean_fade_R"
            ] = np.nan

            row[
                "mean_delta_R"
            ] = np.nan

            row[
                "majority_selected_R"
            ] = np.nan

        rows.append(row)

    return pd.DataFrame(
        rows
    )


# ============================================================
# Only exact +/- three-TF states
# ============================================================

valid_pair = pair[
    pair[
        "valid_three_tf_state"
    ]
].copy()


# ============================================================
# 1. Exact 8-state summary
# ============================================================

exact_frames = []


# pooled
exact_frames.append(
    summarize_groups(
        valid_pair,
        [
            "target_R",
            "joint_state",
            "aligned_count",
        ],
        view="pooled",
    )
)


# source TF
exact_frames.append(
    summarize_groups(
        valid_pair,
        [
            "source_tf",
            "target_R",
            "joint_state",
            "aligned_count",
        ],
        view="by_source_tf",
    )
)


# source TF + OB class
exact_frames.append(
    summarize_groups(
        valid_pair,
        [
            "source_tf",
            "source_ob_structure",
            "target_R",
            "joint_state",
            "aligned_count",
        ],
        view=(
            "by_source_tf_structure"
        ),
    )
)


exact = pd.concat(
    exact_frames,
    ignore_index=True,
)


exact.to_csv(
    OUT
    / "joint_exact_states.csv",
    index=False,
)


# ============================================================
# 2. Aligned-count summary 0/1/2/3
# ============================================================

vote_frames = []


vote_frames.append(
    summarize_groups(
        valid_pair,
        [
            "target_R",
            "aligned_count",
        ],
        view="pooled",
    )
)


vote_frames.append(
    summarize_groups(
        valid_pair,
        [
            "source_tf",
            "target_R",
            "aligned_count",
        ],
        view="by_source_tf",
    )
)


vote_frames.append(
    summarize_groups(
        valid_pair,
        [
            "source_tf",
            "source_ob_structure",
            "target_R",
            "aligned_count",
        ],
        view=(
            "by_source_tf_structure"
        ),
    )
)


vote = pd.concat(
    vote_frames,
    ignore_index=True,
)


vote.to_csv(
    OUT
    / "joint_aligned_count.csv",
    index=False,
)


# ============================================================
# 3. Majority selector summary
# ============================================================

selector_rows = []


for (
    source_tf,
    rr,
), g in valid_pair.groupby(
    [
        "source_tf",
        "target_R",
    ]
):

    valid = g[
        np.isfinite(
            g[
                "majority_selected_R"
            ]
        )
    ]

    selector_rows.append(
        {
            "view":
                "by_source_tf",
            "source_tf":
                source_tf,
            "target_R":
                rr,
            "valid_n":
                len(valid),

            "always_follow_R":
                wmean(
                    valid[
                        "follow_R"
                    ],
                    valid[
                        WEIGHT
                    ],
                ),

            "always_fade_R":
                wmean(
                    valid[
                        "fade_R"
                    ],
                    valid[
                        WEIGHT
                    ],
                ),

            "majority_selector_R":
                wmean(
                    valid[
                        "majority_selected_R"
                    ],
                    valid[
                        WEIGHT
                    ],
                ),
        }
    )


for rr, g in valid_pair.groupby(
    "target_R"
):

    valid = g[
        np.isfinite(
            g[
                "majority_selected_R"
            ]
        )
    ]

    selector_rows.append(
        {
            "view":
                "pooled",
            "source_tf":
                "ALL",
            "target_R":
                rr,
            "valid_n":
                len(valid),

            "always_follow_R":
                wmean(
                    valid[
                        "follow_R"
                    ],
                    valid[
                        WEIGHT
                    ],
                ),

            "always_fade_R":
                wmean(
                    valid[
                        "fade_R"
                    ],
                    valid[
                        WEIGHT
                    ],
                ),

            "majority_selector_R":
                wmean(
                    valid[
                        "majority_selected_R"
                    ],
                    valid[
                        WEIGHT
                    ],
                ),
        }
    )


selector = pd.DataFrame(
    selector_rows
)


selector.to_csv(
    OUT
    / "majority_selector.csv",
    index=False,
)


# ============================================================
# 4. Symbol consistency
#
# Keep exact state and aligned count.
# ============================================================

symbol_rows = []


for (
    symbol,
    source_tf,
    rr,
    state,
), g in valid_pair.groupby(
    [
        "symbol",
        "source_tf",
        "target_R",
        "joint_state",
    ]
):

    valid = g[
        np.isfinite(
            g[
                "delta_R"
            ]
        )
    ]

    if len(valid) < MIN_REPORT_N:
        continue

    symbol_rows.append(
        {
            "symbol":
                symbol,
            "source_tf":
                source_tf,
            "target_R":
                rr,
            "joint_state":
                state,
            "aligned_count":
                int(
                    valid[
                        "aligned_count"
                    ].iloc[0]
                ),
            "valid_n":
                len(valid),
            "mean_delta_R":
                wmean(
                    valid[
                        "delta_R"
                    ],
                    valid[
                        WEIGHT
                    ],
                ),
        }
    )


symbol = pd.DataFrame(
    symbol_rows
)


symbol.to_csv(
    OUT
    / "symbol_consistency.csv",
    index=False,
)


# ============================================================
# Coverage / audit
# ============================================================

audit = {
    "experiment":
        "dsa_multitf_joint_v1",

    "horizon":
        12,

    "timeframes":
        list(TFS),

    "total_candidates":
        int(
            follow_state[
                "candidate_id"
            ].nunique()
        ),

    "valid_three_tf_candidates":
        int(
            follow_state[
                "valid_three_tf_state"
            ].sum()
        ),

    "invalid_or_zero_candidates":
        int(
            (
                ~follow_state[
                    "valid_three_tf_state"
                ]
            ).sum()
        ),

    "joint_state_counts":
        {
            str(k):
                int(v)
            for k, v in (
                follow_state[
                    follow_state[
                        "valid_three_tf_state"
                    ]
                ][
                    "joint_state"
                ]
                .value_counts()
                .sort_index()
                .items()
            )
        },

    "aligned_count_counts":
        {
            str(k):
                int(v)
            for k, v in (
                follow_state[
                    follow_state[
                        "valid_three_tf_state"
                    ]
                ][
                    "aligned_count"
                ]
                .value_counts()
                .sort_index()
                .items()
            )
        },
}


with (
    OUT
    / "audit.json"
).open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        audit,
        f,
        indent=2,
        ensure_ascii=False,
    )


print(
    "DSA_MULTITF_JOINT_DONE"
)

print(
    json.dumps(
        audit,
        indent=2,
        ensure_ascii=False,
    )
)
