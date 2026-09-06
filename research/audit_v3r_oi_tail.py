#!/usr/bin/env python3
"""Targeted OI tail audit for Direction V3R.

The earlier audit only recorded row_id and the absolute relative
change, which is not enough to say anything about roll artefacts. A
small number is not the same thing as a clean number.

This audit records, for the largest absolute relative OI changes
that actually enter the model:

    decision timestamp
    trading_day
    position before
    position after
    relative change
    whether the bar sits at a session boundary

and additionally resolves specific row ids that showed up in the
extremes of several instruments at once, because independent
intraday OI changes should not concentrate on identical bar
positions across contracts.

Nothing is removed and no threshold is applied. This is
inspection only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    __file__
).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )

from research.run_direction_v3r import (  # noqa: E402
    INSTRUMENTS,
    MASTER_FEATURES,
    PRIMARY_HORIZON,
    SRC_5M,
    add_features,
    add_targets,
    continuity_prefix,
)

from research.run_direction_v0 import (  # noqa: E402
    build_execution_grid,
    quantile_state,
)

from research.run_direction_v3r import (  # noqa: E402
    build_15m,
)

from research.run_quantile_rebaseline import (  # noqa: E402
    FEATURE_SETS,
)


OUT = (
    ROOT
    / "research"
    / "exports"
    / "v3r_oi_audit"
)

TOP_N = 20

QUANT_FEATURE_SET = "F1_VOL"

FIVE_NS = (
    5 * 60 * 1_000_000_000
)


def build_master(
    five: pd.DataFrame,
    state: pd.DataFrame,
) -> pd.DataFrame:
    """Rebuild exactly the master sample the model used."""

    cs = (
        continuity_prefix(
            five
        )
    )

    grid = build_execution_grid(
        five,
        state,
    )

    grid[
        "width"
    ] = (
        grid[
            "q90"
        ]
        - grid[
            "q10"
        ]
    )

    grid = add_targets(
        grid,
        five,
        cs,
    )

    grid = add_features(
        grid,
        five,
        cs,
    )

    work = grid.dropna(
        subset=(
            [
                f"fut_ret_{PRIMARY_HORIZON}"
            ]
            + MASTER_FEATURES
            + [
                "width"
            ]
        )
    ).reset_index(
        drop=True
    )

    return work


def main() -> None:

    if OUT.exists() and any(
        OUT.iterdir()
    ):
        raise RuntimeError(
            f"{OUT} exists and is "
            "non-empty. Delete only for an "
            "intentional pre-commit rerun."
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    quant_cols = (
        FEATURE_SETS[
            QUANT_FEATURE_SET
        ]
    )

    top_rows = []
    summary_rows = []
    cluster_rows = []

    for (
        instrument
    ) in INSTRUMENTS:

        print(
            f"{instrument} ...",
            flush=True,
        )

        five = pd.read_csv(
            SRC_5M
            / f"{instrument}_5m.csv",
            parse_dates=[
                "bar_start_time",
                "bar_end_time",
                "availability_time",
                "trading_day",
                "tdx_datetime_raw",
            ],
        )

        state = quantile_state(
            build_15m(
                five
            ),
            quant_cols,
        )

        work = build_master(
            five,
            state,
        )

        f = five.sort_values(
            "bar_start_time"
        ).reset_index(
            drop=True
        )

        pos = f[
            "position"
        ].to_numpy(
            dtype=float
        )

        start_ns = (
            f[
                "bar_start_time"
            ]
            .to_numpy(
                dtype=(
                    "datetime64[ns]"
                )
            )
            .astype(
                np.int64
            )
        )

        # Gap flag: bar j follows a gap when the previous spacing
        # is not exactly one 5m step.
        gap_before = np.concatenate(
            [
                [False],
                np.diff(
                    start_ns
                )
                != FIVE_NS,
            ]
        )

        row_id = (
            work[
                "row_id"
            ].to_numpy(
                dtype=int
            )
        )

        # rel_dOI_5m uses position[k] and position[k-1],
        # where k = row_id - 1.
        k = (
            row_id
            - 1
        )

        oi_now = pos[
            k
        ]

        oi_prev = pos[
            k
            - 1
        ]

        rel = (
            work[
                "rel_dOI_5m"
            ].to_numpy(
                dtype=float
            )
        )

        decision_time = (
            f[
                "bar_start_time"
            ]
            .to_numpy()[
                row_id
            ]
        )

        trading_day = (
            f[
                "trading_day"
            ]
            .to_numpy()[
                k
            ]
        )

        # Flag for the OI DIFFERENCE WINDOW ONLY.
        #
        # rel_dOI_5m compares position[k-1] to position[k]. This is
        # true when bar k is not adjacent to bar k-1, i.e. when the
        # difference spans a session gap and is therefore not a
        # 5-minute change at all.
        #
        # It says nothing about whether the DECISION bar itself sits
        # at a session open. Those are different questions. A
        # decision at 21:00 can legitimately use a feature bar from
        # the tail of the previous session, which is not leakage,
        # only stale last-available state.
        oi_crosses_gap = (
            gap_before[
                k
            ]
        )

        frame = pd.DataFrame(
            {
                "row_id": row_id,
                "decision_time": (
                    decision_time
                ),
                "trading_day": (
                    trading_day
                ),
                "position_prev": (
                    oi_prev
                ),
                "position_now": (
                    oi_now
                ),
                "abs_position_change": (
                    oi_now
                    - oi_prev
                ),
                "rel_dOI_5m": rel,
                "abs_rel_dOI_5m": np.abs(
                    rel
                ),
                "oi_window_crosses_gap": (
                    oi_crosses_gap
                ),
            }
        )

        summary_rows.append(
            {
                "instrument": (
                    instrument
                ),
                "n": int(
                    len(
                        frame
                    )
                ),
                "abs_max": float(
                    frame[
                        "abs_rel_dOI_5m"
                    ].max()
                ),
                "abs_p99": float(
                    frame[
                        "abs_rel_dOI_5m"
                    ].quantile(
                        0.99
                    )
                ),
                "n_above_10pct": int(
                    (
                        frame[
                            "abs_rel_dOI_5m"
                        ]
                        > 0.10
                    ).sum()
                ),
                "n_above_5pct": int(
                    (
                        frame[
                            "abs_rel_dOI_5m"
                        ]
                        > 0.05
                    ).sum()
                ),
                "top20_oi_window_crosses_gap_share": (
                    float(
                        frame
                        .nlargest(
                            TOP_N,
                            "abs_rel_dOI_5m",
                        )[
                            "oi_window_crosses_gap"
                        ].mean()
                    )
                ),
                "all_oi_window_crosses_gap_share": (
                    float(
                        frame[
                            "oi_window_crosses_gap"
                        ].mean()
                    )
                ),
            }
        )

        top = frame.nlargest(
            TOP_N,
            "abs_rel_dOI_5m",
        ).copy()

        top.insert(
            0,
            "instrument",
            instrument,
        )

        top.insert(
            2,
            "rank",
            range(
                1,
                len(
                    top
                )
                + 1,
            ),
        )

        top_rows.append(
            top
        )

    tops = pd.concat(
        top_rows,
        ignore_index=(
            True
        ),
    )

    tops.to_csv(
        OUT
        / "oi_top20_by_instrument.csv",
        index=False,
    )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUT
        / "oi_tail_summary.csv",
        index=False,
    )

    # ---- resolve the shared row ids ----

    shared = (
        tops.groupby(
            "row_id"
        )
        .agg(
            n_instruments=(
                "instrument",
                "nunique",
            ),
            instruments=(
                "instrument",
                lambda s: (
                    ",".join(
                        sorted(
                            s
                        )
                    )
                ),
            ),
        )
        .reset_index()
    )

    shared = (
        shared[
            shared[
                "n_instruments"
            ]
            > 1
        ]
        .sort_values(
            "n_instruments",
            ascending=(
                False
            ),
        )
    )

    for (
        _,
        row,
    ) in shared.iterrows():

        rid = int(
            row[
                "row_id"
            ]
        )

        for (
            instrument
        ) in str(
            row[
                "instruments"
            ]
        ).split(
            ","
        ):

            five = pd.read_csv(
                SRC_5M
                / f"{instrument}_5m.csv",
                parse_dates=[
                    "bar_start_time",
                    "bar_end_time",
                    "trading_day",
                ],
            )

            f = five.sort_values(
                "bar_start_time"
            ).reset_index(
                drop=True
            )

            if rid >= len(
                f
            ):
                continue

            k = (
                rid
                - 1
            )

            if k < 1:
                continue

            cluster_rows.append(
                {
                    "row_id": rid,
                    "n_instruments": int(
                        row[
                            "n_instruments"
                        ]
                    ),
                    "instruments": (
                        row[
                            "instruments"
                        ]
                    ),
                    "instrument": (
                        instrument
                    ),
                    "bar_start_time": str(
                        f[
                            "bar_start_time"
                        ].iloc[
                            k
                        ]
                    ),
                    "bar_end_time": str(
                        f[
                            "bar_end_time"
                        ].iloc[
                            k
                        ]
                    ),
                    "trading_day": str(
                        f[
                            "trading_day"
                        ].iloc[
                            k
                        ]
                    )[
                        :10
                    ],
                    "position_prev": float(
                        f[
                            "position"
                        ].iloc[
                            k
                            - 1
                        ]
                    ),
                    "position_now": float(
                        f[
                            "position"
                        ].iloc[
                            k
                        ]
                    ),
                    "rel_change": float(
                        (
                            f[
                                "position"
                            ].iloc[
                                k
                            ]
                            - f[
                                "position"
                            ].iloc[
                                k
                                - 1
                            ]
                        )
                        / abs(
                            f[
                                "position"
                            ].iloc[
                                k
                                - 1
                            ]
                        )
                    ),
                }
            )

    if cluster_rows:

        pd.DataFrame(
            cluster_rows
        ).to_csv(
            OUT
            / "oi_shared_rowid_resolution.csv",
            index=False,
        )

    (
        OUT
        / "audit_config.json"
    ).write_text(
        json.dumps(
            {
                "scope": (
                    "largest absolute relative OI "
                    "changes entering the master "
                    "sample"
                ),
                "top_n_per_instrument": (
                    TOP_N
                ),
                "fields": [
                    "decision_time",
                    "trading_day",
                    "position_prev",
                    "position_now",
                    "rel_dOI_5m",
                    "oi_window_crosses_gap",
                ],
                "flag_semantics": (
                    "oi_window_crosses_gap refers to "
                    "the OI difference window "
                    "(position[k-1] -> position[k]) "
                    "only, NOT to the decision bar. "
                    "A zero value means no OI window "
                    "in the master sample spans a "
                    "session gap. It does not mean "
                    "the master sample contains no "
                    "session-open decision bar."
                ),
                "nothing_removed": True,
                "no_threshold_applied": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== OI TAIL SUMMARY ==="
    )

    with pd.option_context(
        "display.width",
        250,
    ):
        print(
            summary.to_string(
                index=False
            )
        )

    print(
        "\n=== SHARED row_id RESOLUTION ==="
    )

    if cluster_rows:

        print(
            pd.DataFrame(
                cluster_rows
            ).to_string(
                index=False
            )
        )

    else:

        print(
            "(none)"
        )

    print(
        "\nV3R_OI_AUDIT_DONE"
    )


if __name__ == "__main__":
    main()
