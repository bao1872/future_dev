#!/usr/bin/env python3
"""V2 fold-level stability check (S1 vs S0).

Scope deliberately tiny:

    H5 + ALL + GBR, S1 vs S0, per fold.

V2 pooled the OOS of all folds into one metric. This asks the only
question that matters before building on that result:

    is the S1 gain produced by the whole sample, or by one or two
    lucky time windows?

No bootstrap, no HAC, no new modelling. This recomputes the
deterministic walk-forward through the same locked code path and
reports every fold separately.

Per fold:

    AUC, Brier skill, logloss skill, return spread
    (baseline = that fold's own TRAIN UP frequency)

Two aggregation levels:

    within each instrument: how many folds have S1 > S0
    across all instrument x fold pairs: positive share
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import (
    StandardScaler,
)


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
    COMMON_FEATURES,
    FEATURE_BLOCKS,
    HORIZON_BARS,
    INSTRUMENTS,
    PANEL_15M,
    PANEL_5M,
    QUANT_FEATURE_SET,
    SPREAD_QUANTILE,
    PROB_CLIP,
    add_state_features,
    make_learner,
)

from research.run_quantile_rebaseline import (  # noqa: E402
    FEATURE_SETS,
    make_folds,
)


OUT = (
    ROOT
    / "research"
    / "exports"
    / "direction_v2_stability"
)

HORIZON = "H5"

LEARNER = "gbr"

BLOCK_A = "S0"

BLOCK_B = "S1"


def fold_metrics(
    y_true: np.ndarray,
    p: np.ndarray,
    p_train: float,
    fut_ret: np.ndarray,
) -> dict:

    p_safe = np.clip(
        p,
        PROB_CLIP,
        1.0
        - PROB_CLIP,
    )

    brier_model = float(
        np.mean(
            (
                p_safe
                - y_true
            )
            ** 2
        )
    )

    brier_base = float(
        np.mean(
            (
                p_train
                - y_true
            )
            ** 2
        )
    )

    ll_model = float(
        log_loss(
            y_true,
            p_safe,
            labels=[
                0,
                1
            ],
        )
    )

    ll_base = float(
        log_loss(
            y_true,
            np.full(
                len(
                    y_true
                ),
                p_train,
            ),
            labels=[
                0,
                1
            ],
        )
    )

    hi = np.quantile(
        p_safe,
        1.0
        - SPREAD_QUANTILE,
    )

    lo = np.quantile(
        p_safe,
        SPREAD_QUANTILE,
    )

    top = (
        p_safe
        >= hi
    )

    bottom = (
        p_safe
        <= lo
    )

    spread = (
        float(
            np.mean(
                fut_ret[
                    top
                ]
            )
            - np.mean(
                fut_ret[
                    bottom
                ]
            )
        )
        if (
            top.any()
            and bottom.any()
        )
        else np.nan
    )

    return {
        "auc": float(
            roc_auc_score(
                y_true,
                p_safe
            )
        ),
        "brier_skill": (
            1.0
            - brier_model
            / brier_base
            if brier_base
            > 0
            else np.nan
        ),
        "logloss_skill": (
            1.0
            - ll_model
            / ll_base
            if ll_base
            > 0
            else np.nan
        ),
        "return_spread": (
            spread
        ),
    }


def per_fold_run(
    work: pd.DataFrame,
    *,
    feature_names: list[str],
) -> pd.DataFrame:
    """Walk-forward, metrics computed SEPARATELY per fold."""

    n_bars = (
        HORIZON_BARS[
            HORIZON
        ]
    )

    ret_col = (
        f"fut_ret_{HORIZON}"
    )

    n = len(
        work
    )

    folds = make_folds(
        n,
        horizon=n_bars,
    )

    rows = []

    for fold in folds:

        tr = np.arange(
            fold[
                "train_start"
            ],
            fold[
                "train_end_exclusive"
            ],
        )

        te = np.arange(
            fold[
                "test_start"
            ],
            fold[
                "test_end_exclusive"
            ],
        )

        if (
            te[0]
            - fold[
                "train_end_exclusive"
            ]
        ) < n_bars:
            raise RuntimeError(
                "purge invariant "
                "failed"
            )

        train = (
            work.iloc[
                tr
            ]
        )

        test = (
            work.iloc[
                te
            ]
        )

        if (
            len(tr)
            < 200
            or len(te)
            < 50
            or train[
                "y"
            ].nunique()
            < 2
            or test[
                "y"
            ].nunique()
            < 2
        ):
            continue

        scaler = (
            StandardScaler()
        )

        scaler.fit(
            train[
                feature_names
            ].to_numpy(
                dtype=float
            )
        )

        model = make_learner(
            LEARNER
        )

        model.fit(
            scaler.transform(
                train[
                    feature_names
                ].to_numpy(
                    dtype=float
                )
            ),
            train[
                "y"
            ].to_numpy(),
        )

        p = model.predict_proba(
            scaler.transform(
                test[
                    feature_names
                ].to_numpy(
                    dtype=float
                )
            )
        )[
            :,
            1
        ]

        p_train = float(
            train[
                "y"
            ].mean()
        )

        m = fold_metrics(
            test[
                "y"
            ].to_numpy(
                dtype=int
            ),
            p,
            p_train,
            test[
                ret_col
            ].to_numpy(
                dtype=float
            ),
        )

        rows.append(
            {
                "fold": int(
                    fold[
                        "fold"
                    ]
                ),
                "train_rows": int(
                    len(
                        tr
                    )
                ),
                "test_rows": int(
                    len(
                        te
                    )
                ),
                "train_up_rate": (
                    p_train
                ),
                **m,
            }
        )

    return pd.DataFrame(
        rows
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

        a = per_fold_run(
            work,
            feature_names=(
                FEATURE_BLOCKS[
                    BLOCK_A
                ]
            ),
        )

        b = per_fold_run(
            work,
            feature_names=(
                FEATURE_BLOCKS[
                    BLOCK_B
                ]
            ),
        )

        merged = a.merge(
            b,
            on="fold",
            suffixes=(
                "_S0",
                "_S1",
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
                    f"{metric}_S1"
                ]
                - merged[
                    f"{metric}_S0"
                ]
            )

        merged.insert(
            0,
            "instrument",
            instrument,
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
        / "fold_level_s1_vs_s0.csv",
        index=False,
    )

    # ---- aggregation ----

    per_inst_rows = []

    for (
        instrument,
        g,
    ) in folds.groupby(
        "instrument",
        observed=(
            True
        ),
    ):

        row = {
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
            "logloss_skill",
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

            row[
                f"share_{metric}_better"
            ] = float(
                (
                    g[
                        f"delta_{metric}"
                    ]
                    > 0
                ).mean()
            )

            row[
                f"median_delta_{metric}"
            ] = float(
                g[
                    f"delta_{metric}"
                ].median()
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

    pool_rows = []

    for (
        metric
    ) in (
        "auc",
        "brier_skill",
        "logloss_skill",
        "return_spread",
    ):

        d = (
            folds[
                f"delta_{metric}"
            ]
            .dropna()
        )

        pool_rows.append(
            {
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
                "mean_delta": float(
                    d.mean()
                ),
                "p25_delta": float(
                    d.quantile(
                        0.25
                    )
                ),
                "p75_delta": float(
                    d.quantile(
                        0.75
                    )
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

    config = {
        "scope": (
            f"{HORIZON} + ALL + "
            f"{LEARNER}, {BLOCK_B} vs "
            f"{BLOCK_A}, per fold"
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
        "\n=== PER INSTRUMENT (folds where "
        "S1 > S0) ==="
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
        "\n=== POOLED instrument x fold ==="
    )

    print(
        pooled.to_string(
            index=False
        )
    )

    print(
        "\nV2_FOLD_STABILITY_DONE"
    )


if __name__ == "__main__":
    main()
