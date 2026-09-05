#!/usr/bin/env python3
"""V3 fold-level stability check (B1 / B2 / B3 vs B0).

Same spirit and same size as the V2 check: recompute the
deterministic walk-forward through the locked code path and report
EVERY FOLD separately, so an instrument-level improvement cannot
hide the fact that it came from one or two windows.

Scope:

    H5 + ALL + GBR
    B0 = price + RV  (the V2 S1 baseline)
    B1 = B0 + volume
    B2 = B0 + OI
    B3 = B0 + volume + OI + 2 interactions

Each fold's metrics use that fold's own TRAIN UP frequency as the
baseline. No bootstrap, no HAC, no new modelling.
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

from research.run_direction_v0 import (  # noqa: E402
    build_execution_grid,
    quantile_state,
)

from research.run_direction_v1 import (  # noqa: E402
    add_targets,
    continuity_prefix,
)

from research.run_direction_v2 import (  # noqa: E402
    PANEL_15M,
    PANEL_5M,
    QUANT_FEATURE_SET,
    add_state_features,
    make_learner,
)

from research.run_direction_v3 import (  # noqa: E402
    COMMON_FEATURES,
    FEATURE_BLOCKS,
    INSTRUMENTS,
    add_participation_features,
)

from research.run_quantile_rebaseline import (  # noqa: E402
    FEATURE_SETS,
    make_folds,
)

from research.check_v2_fold_stability import (  # noqa: E402
    HORIZON,
    LEARNER,
    fold_metrics,
    per_fold_run,
)


OUT = (
    ROOT
    / "research"
    / "exports"
    / "direction_v3_stability"
)

BASE_BLOCK = "B0"

TEST_BLOCKS = (
    "B1",
    "B2",
    "B3",
)


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

    fold_rows = []

    for (
        instrument
    ) in INSTRUMENTS:

        print(
            f"{instrument} ...",
            flush=True,
        )

        panel = pd.read_csv(
            PANEL_15M
            / f"{instrument}_panel.csv",
            parse_dates=[
                "meta_decision_time"
            ],
        )

        five = pd.read_csv(
            PANEL_5M
            / f"{instrument}_5m.csv",
            parse_dates=[
                "bar_start_time"
            ],
        )

        state = quantile_state(
            panel,
            quant_cols,
        )

        cs = continuity_prefix(
            five
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

        grid = add_state_features(
            grid,
            five,
            cs,
        )

        grid = add_participation_features(
            grid,
            five,
            cs,
        )

        ret_col = (
            f"fut_ret_{HORIZON}"
        )

        work = grid.dropna(
            subset=(
                [
                    ret_col
                ]
                + COMMON_FEATURES
                + [
                    "width"
                ]
            )
        ).reset_index(
            drop=True
        )

        work[
            "y"
        ] = (
            (
                work[
                    ret_col
                ].to_numpy()
                > 0
            ).astype(
                int
            )
        )

        base = per_fold_run(
            work,
            feature_names=(
                FEATURE_BLOCKS[
                    BASE_BLOCK
                ]
            ),
        )

        for block in (
            TEST_BLOCKS
        ):

            test = per_fold_run(
                work,
                feature_names=(
                    FEATURE_BLOCKS[
                        block
                    ]
                ),
            )

            merged = base.merge(
                test,
                on="fold",
                suffixes=(
                    f"_{BASE_BLOCK}",
                    f"_{block}",
                ),
            )

            for (
                metric
            ) in (
                "auc",
                "brier_skill",
                "logloss_skill",
                "return_spread",
            ):

                merged[
                    f"delta_{metric}"
                ] = (
                    merged[
                        f"{metric}_{block}"
                    ]
                    - merged[
                        f"{metric}_{BASE_BLOCK}"
                    ]
                )

            merged.insert(
                0,
                "instrument",
                instrument,
            )

            merged.insert(
                1,
                "block",
                block,
            )

            fold_rows.append(
                merged
            )

    folds = pd.concat(
        fold_rows,
        ignore_index=(
            True
        ),
    )

    folds.to_csv(
        OUT
        / "fold_level_vs_b0.csv",
        index=False,
    )

    # ---- pooled per block ----

    pool_rows = []

    for (
        block,
        g,
    ) in folds.groupby(
        "block",
        observed=(
            True
        ),
    ):

        for (
            metric
        ) in (
            "auc",
            "brier_skill",
            "logloss_skill",
            "return_spread",
        ):

            d = (
                g[
                    f"delta_{metric}"
                ].dropna()
            )

            pool_rows.append(
                {
                    "block": block,
                    "metric": metric,
                    "instrument_fold_pairs": int(
                        len(
                            d
                        )
                    ),
                    "positive_pairs": int(
                        (
                            d > 0
                        ).sum()
                    ),
                    "positive_share": float(
                        (
                            d > 0
                        ).mean()
                    ),
                    "median_delta": float(
                        d.median()
                    ),
                }
            )

    pooled = pd.DataFrame(
        pool_rows
    )

    pooled.to_csv(
        OUT
        / "fold_stability_pooled.csv",
        index=False,
    )

    # ---- per instrument ----

    per_inst_rows = []

    for keys, g in folds.groupby(
        [
            "block",
            "instrument",
        ],
        observed=(
            True
        ),
    ):

        (
            block,
            instrument,
        ) = keys

        row = {
            "block": block,
            "instrument": (
                instrument
            ),
            "n_folds": int(
                len(
                    g
                )
            ),
        }

        for (
            metric
        ) in (
            "auc",
            "brier_skill",
            "return_spread",
        ):

            row[
                f"folds_{metric}_better"
            ] = int(
                (
                    g[
                        f"delta_{metric}"
                    ]
                    > 0
                ).sum()
            )

        per_inst_rows.append(
            row
        )

    per_inst = pd.DataFrame(
        per_inst_rows
    )

    per_inst.to_csv(
        OUT
        / "fold_stability_by_instrument.csv",
        index=False,
    )

    config = {
        "scope": (
            f"{HORIZON} + ALL + "
            f"{LEARNER}, "
            f"{', '.join(TEST_BLOCKS)} vs "
            f"{BASE_BLOCK}, per fold"
        ),
        "baseline": (
            "each fold's own TRAIN UP "
            "frequency"
        ),
        "no_bootstrap": True,
        "no_hac": True,
        "recomputed_through_locked_path": (
            True
        ),
    }

    (
        OUT
        / "stability_config.json"
    ).write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== POOLED instrument x fold ==="
    )

    print(
        pooled.to_string(
            index=False
        )
    )

    print(
        "\n=== PER INSTRUMENT (folds better "
        "than B0, out of 7) ==="
    )

    with pd.option_context(
        "display.width",
        260,
    ):
        print(
            per_inst.to_string(
                index=False
            )
        )

    print(
        "\nV3_FOLD_STABILITY_DONE"
    )


if __name__ == "__main__":
    main()
