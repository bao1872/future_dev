#!/usr/bin/env python3
"""PyTDX Quantile Rebaseline (Experiment 1).

Single question:

    Under the new PyTDX data base, can the quantile behaviour that
    was observed on the old dataset be reproduced?

Deliberately NOT repeated here:
    pooled training, leave-one-instrument-out, block bootstrap,
    HAC / Newey-West, CAViaR, HAR, hyperparameter tuning,
    transaction-flow features, SMC / DSA / Momentum.

Fixed design:

    horizons     H4 primary, H8 control
    quantiles    Q10, Q50, Q90
    features     F1, F1_VOL
    models       QuantileRegressor, fixed shallow GBR quantile
    benchmark    unconditional train-set quantile
    validation   purged expanding walk-forward

Reported per instrument:

    1. Q10 pinball skill
    2. Q50 pinball skill
    3. Q90 pinball skill
    4. Q90-Q10 interval skill
    5. predicted width -> future path range Spearman

No model parameter is tuned on the new data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_pinball_loss,
)


ROOT = Path(
    __file__
).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )

from research.fit_quantile_v2_models import (  # noqa: E402
    make_model,
    interval_score,
    safe_spearman,
)


SRC = (
    ROOT
    / "research"
    / "exports"
    / "pytdx_panel"
)

OUT = (
    ROOT
    / "research"
    / "exports"
    / "pytdx_rebaseline"
)

INSTRUMENTS = (
    "AG",
    "CU",
    "AL",
    "SN",
    "I",
    "SC",
    "M",
    "CF",
)

PRIMARY_HORIZON = 4

HORIZONS = (
    PRIMARY_HORIZON,
    8,
)

QUANTILES = (
    0.10,
    0.50,
    0.90,
)

FEATURE_SETS = {
    "F1": [
        "feat_15m_ret_1",
        "feat_15m_ret_4",
        "feat_15m_ret_8",
        "feat_15m_ret_16",
        "feat_15m_location_32",
        "feat_time_bars_since_segment_start",
        "feat_time_after_long_gap",
    ],
}

FEATURE_SETS[
    "F1_VOL"
] = FEATURE_SETS[
    "F1"
] + [
    "feat_5m_1h_rv",
    "feat_5m_rv_rate_ratio_1h_4h",
]

MODELS = (
    "linear_qr",
    "gbr_quantile",
)

# Walk-forward geometry, fixed fractions of the sample.
TRAIN_FRACTION = 0.30

TEST_FRACTION = 0.10

MIN_TEST_ROWS = 250


def make_folds(
    n_rows: int,
    *,
    horizon: int,
) -> list[dict]:
    """Purged expanding walk-forward folds.

    Same semantics as the previous experiment: train always starts
    at row 0, the test window starts `horizon` rows after the train
    window ends, and the window slides forward. Geometry is scaled
    to the sample size because the new panel is several times
    longer than the old one.
    """

    min_train = int(
        n_rows
        * TRAIN_FRACTION
    )

    test_rows = max(
        int(
            n_rows
            * TEST_FRACTION
        ),
        MIN_TEST_ROWS,
    )

    folds = []

    train_end = min_train

    fold_id = 0

    while True:

        test_start = (
            train_end
            + horizon
        )

        test_end = min(
            test_start
            + test_rows,
            n_rows,
        )

        if (
            test_start
            >= n_rows
        ):
            break

        if (
            test_end
            - test_start
        ) < MIN_TEST_ROWS:
            break

        folds.append(
            {
                "fold": fold_id,
                "train_start": 0,
                "train_end_exclusive": (
                    train_end
                ),
                "test_start": (
                    test_start
                ),
                "test_end_exclusive": (
                    test_end
                ),
                "purge_rows": (
                    horizon
                ),
            }
        )

        fold_id += 1

        train_end += (
            test_rows
        )

    if len(folds) < 3:
        raise RuntimeError(
            f"only {len(folds)} folds "
            f"for n={n_rows}"
        )

    return folds


def run_combo(
    frame: pd.DataFrame,
    *,
    horizon: int,
    feature_cols: list[str],
    model_name: str,
) -> dict:

    target = (
        f"target_raw_return_h"
        f"{horizon}"
    )

    y_all = (
        frame[
            target
        ]
        .to_numpy(
            dtype=float
        )
    )

    X_all = frame[
        feature_cols
    ].apply(
        pd.to_numeric,
        errors=(
            "coerce"
        ),
    )

    n = len(
        frame
    )

    folds = make_folds(
        n,
        horizon=horizon,
    )

    pred = {
        q: np.full(
            n,
            np.nan,
        )
        for q in QUANTILES
    }

    baseline = {
        q: np.full(
            n,
            np.nan,
        )
        for q in QUANTILES
    }

    fold_id = np.full(
        n,
        -1,
        dtype=int,
    )

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
        ) < horizon:
            raise RuntimeError(
                "purge invariant "
                "failed"
            )

        X_tr = (
            X_all.iloc[
                tr
            ]
        )

        X_te = (
            X_all.iloc[
                te
            ]
        )

        y_tr = (
            y_all[
                tr
            ]
        )

        for q in QUANTILES:

            model = (
                make_model(
                    model_name,
                    q,
                )
            )

            model.fit(
                X_tr,
                y_tr,
            )

            pred[
                q
            ][
                te
            ] = model.predict(
                X_te
            )

            baseline[
                q
            ][
                te
            ] = float(
                np.quantile(
                    y_tr,
                    q,
                )
            )

        fold_id[
            te
        ] = fold[
            "fold"
        ]

    oos = fold_id >= 0

    y = y_all[
        oos
    ]

    path_range = (
        frame[
            "path_range"
        ]
        .to_numpy(
            dtype=float
        )[
            oos
        ]
    )

    result: dict = {
        "oos_rows": int(
            oos.sum()
        ),
        "folds": len(
            folds
        ),
    }

    for q in QUANTILES:

        p = pred[
            q
        ][
            oos
        ]

        b = baseline[
            q
        ][
            oos
        ]

        model_loss = float(
            mean_pinball_loss(
                y,
                p,
                alpha=q,
            )
        )

        base_loss = float(
            mean_pinball_loss(
                y,
                b,
                alpha=q,
            )
        )

        result[
            f"q{int(q * 100):02d}_pinball_skill"
        ] = (
            1.0
            -
            model_loss
            /
            base_loss
            if base_loss > 0
            else np.nan
        )

        result[
            f"q{int(q * 100):02d}_calibration"
        ] = float(
            (
                y <= p
            ).mean()
        )

    # Interval skill on the ordered subset.
    lo = pred[
        0.10
    ][
        oos
    ]

    hi = pred[
        0.90
    ][
        oos
    ]

    blo = baseline[
        0.10
    ][
        oos
    ]

    bhi = baseline[
        0.90
    ][
        oos
    ]

    ordered = lo <= hi

    result[
        "crossing_rate"
    ] = float(
        (
            ~ordered
        ).mean()
    )

    if ordered.any():

        yy = y[
            ordered
        ]

        model_score = float(
            np.mean(
                interval_score(
                    yy,
                    lo[
                        ordered
                    ],
                    hi[
                        ordered
                    ],
                    alpha=0.20,
                )
            )
        )

        base_score = float(
            np.mean(
                interval_score(
                    yy,
                    blo[
                        ordered
                    ],
                    bhi[
                        ordered
                    ],
                    alpha=0.20,
                )
            )
        )

        result[
            "interval_80_coverage"
        ] = float(
            (
                (
                    yy
                    >= lo[
                        ordered
                    ]
                )
                &
                (
                    yy
                    <= hi[
                        ordered
                    ]
                )
            ).mean()
        )

        result[
            "interval_skill"
        ] = (
            1.0
            -
            model_score
            /
            base_score
            if base_score > 0
            else np.nan
        )

    else:

        result[
            "interval_80_coverage"
        ] = np.nan

        result[
            "interval_skill"
        ] = np.nan

    # Observation-level width vs future path range.
    width = (
        hi - lo
    )

    keep = (
        np.isfinite(
            width
        )
        & np.isfinite(
            path_range
        )
        & (
            width >= 0
        )
    )

    result[
        "width_path_spearman"
    ] = (
        safe_spearman(
            width[
                keep
            ],
            path_range[
                keep
            ],
        )
    )

    result[
        "width_abs_return_spearman"
    ] = (
        safe_spearman(
            width[
                keep
            ],
            np.abs(
                y[
                    keep
                ]
            ),
        )
    )

    result[
        "width_path_n"
    ] = int(
        keep.sum()
    )

    return result


def positive_share(
    values,
) -> float:

    s = pd.to_numeric(
        pd.Series(
            values
        ),
        errors=(
            "coerce"
        ),
    ).dropna()

    if len(s) == 0:
        return float(
            "nan"
        )

    return float(
        (
            s > 0
        ).mean()
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

    rows = []

    for (
        instrument
    ) in INSTRUMENTS:

        print(
            "=" * 64
        )

        print(
            f"{instrument}"
        )

        print(
            "=" * 64
        )

        panel = pd.read_csv(
            SRC
            / f"{instrument}_panel.csv",
            parse_dates=[
                "meta_base_bar_time",
                "meta_decision_time",
            ],
        )

        for horizon in HORIZONS:

            target = (
                f"target_raw_return_h"
                f"{horizon}"
            )

            long_col = (
                f"target_long_mfe_h"
                f"{horizon}"
            )

            short_col = (
                f"target_short_mfe_h"
                f"{horizon}"
            )

            valid = (
                panel[
                    target
                ].notna()
                & panel[
                    long_col
                ].notna()
                & panel[
                    short_col
                ].notna()
            )

            frame = (
                panel.loc[
                    valid
                ]
                .reset_index(
                    drop=True
                )
                .copy()
            )

            frame[
                "path_range"
            ] = (
                frame[
                    long_col
                ]
                + frame[
                    short_col
                ]
            )

            for (
                fs_name,
                cols,
            ) in (
                FEATURE_SETS.items()
            ):

                missing = [
                    c
                    for c in cols
                    if c
                    not in (
                        frame.columns
                    )
                ]

                if missing:
                    raise RuntimeError(
                        f"{instrument}: "
                        f"missing "
                        f"{missing}"
                    )

                for (
                    model_name
                ) in MODELS:

                    res = (
                        run_combo(
                            frame,
                            horizon=(
                                horizon
                            ),
                            feature_cols=(
                                cols
                            ),
                            model_name=(
                                model_name
                            ),
                        )
                    )

                    row = {
                        "instrument": (
                            instrument
                        ),
                        "horizon": (
                            horizon
                        ),
                        "feature_set": (
                            fs_name
                        ),
                        "model": (
                            model_name
                        ),
                        **res,
                    }

                    rows.append(
                        row
                    )

                    print(
                        f"  H{horizon} "
                        f"{fs_name:7s} "
                        f"{model_name:12s} "
                        f"oos={res['oos_rows']:6d} "
                        f"folds={res['folds']} "
                        f"Q10={res['q10_pinball_skill']:+.4f} "
                        f"Q50={res['q50_pinball_skill']:+.4f} "
                        f"Q90={res['q90_pinball_skill']:+.4f} "
                        f"INT={res['interval_skill']:+.4f} "
                        f"W={res['width_path_spearman']:.3f}"
                    )

    results = pd.DataFrame(
        rows
    )

    results.to_csv(
        OUT
        / "rebaseline_by_instrument.csv",
        index=False,
    )

    # ========================================================
    # Cross-instrument summary
    # ========================================================

    summary_rows = []

    metrics = [
        "q10_pinball_skill",
        "q50_pinball_skill",
        "q90_pinball_skill",
        "interval_skill",
        "width_path_spearman",
        "width_abs_return_spearman",
    ]

    for (
        horizon,
        fs_name,
        model_name,
    ), g in results.groupby(
        [
            "horizon",
            "feature_set",
            "model",
        ],
        observed=True,
    ):

        row = {
            "horizon": int(
                horizon
            ),
            "feature_set": (
                fs_name
            ),
            "model": (
                model_name
            ),
            "instrument_count": int(
                g[
                    "instrument"
                ].nunique()
            ),
        }

        for metric in metrics:

            values = pd.to_numeric(
                g[
                    metric
                ],
                errors=(
                    "coerce"
                ),
            ).dropna()

            row[
                f"median_{metric}"
            ] = float(
                values.median()
            )

            row[
                f"positive_share_{metric}"
            ] = positive_share(
                values
            )

            row[
                f"min_{metric}"
            ] = float(
                values.min()
            )

            row[
                f"max_{metric}"
            ] = float(
                values.max()
            )

        summary_rows.append(
            row
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUT
        / "rebaseline_cross_summary.csv",
        index=False,
    )

    config = {
        "purpose": (
            "Reproduce the quantile behaviour "
            "on the PyTDX data base."
        ),
        "source": (
            "PyTDX L8 5m bars, locally "
            "aggregated to 15m"
        ),
        "horizons": list(
            HORIZONS
        ),
        "primary_horizon": (
            PRIMARY_HORIZON
        ),
        "quantiles": list(
            QUANTILES
        ),
        "feature_sets": (
            FEATURE_SETS
        ),
        "models": list(
            MODELS
        ),
        "train_fraction": (
            TRAIN_FRACTION
        ),
        "test_fraction": (
            TEST_FRACTION
        ),
        "benchmark": (
            "unconditional train-set "
            "quantile"
        ),
        "no_hyperparameter_tuning": (
            True
        ),
        "no_pooled_model": True,
        "no_leave_one_instrument_out": (
            True
        ),
        "no_bootstrap": True,
        "no_hac": True,
        "no_caviar": True,
        "no_har": True,
        "no_transaction_data": True,
        "no_smc": True,
        "no_dsa": True,
        "no_momentum": True,
    }

    (
        OUT
        / "experiment_config.json"
    ).write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    numeric = (
        results.select_dtypes(
            include=[
                np.number
            ]
        )
    )

    if np.isinf(
        numeric.to_numpy(
            dtype=float
        )
    ).any():
        raise RuntimeError(
            "output contains +/-inf"
        )

    validation = {
        "status": "PASS",
        "instrument_count": (
            len(
                INSTRUMENTS
            )
        ),
        "row_count": int(
            len(
                results
            )
        ),
        "expected_row_count": (
            len(INSTRUMENTS)
            * len(HORIZONS)
            * len(FEATURE_SETS)
            * len(MODELS)
        ),
        "no_hyperparameter_tuning": (
            True
        ),
        "no_pooled_model": True,
        "no_transaction_data": True,
    }

    if (
        validation[
            "row_count"
        ]
        != validation[
            "expected_row_count"
        ]
    ):
        raise RuntimeError(
            "row count mismatch"
        )

    (
        OUT
        / "validation.json"
    ).write_text(
        json.dumps(
            validation,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Compact report
    # ========================================================

    lines = [
        "# PyTDX Quantile Rebaseline",
        "",
        "H4 primary, H8 control.",
        "No tuning, no pooled model, no LOO.",
        "",
        "## Per instrument (H4)",
        "",
        "| instrument | model | feature | "
        "Q10 | Q50 | Q90 | interval | "
        "width/path |",
        "|---|---|---|---:|---:|---:|---:"
        "|---:|",
    ]

    h4 = results[
        results[
            "horizon"
        ]
        == PRIMARY_HORIZON
    ]

    for (
        _,
        r,
    ) in h4.sort_values(
        [
            "instrument",
            "model",
            "feature_set",
        ]
    ).iterrows():

        lines.append(
            f"| {r['instrument']} "
            f"| {r['model']} "
            f"| {r['feature_set']} "
            f"| {r['q10_pinball_skill']:+.5f} "
            f"| {r['q50_pinball_skill']:+.5f} "
            f"| {r['q90_pinball_skill']:+.5f} "
            f"| {r['interval_skill']:+.5f} "
            f"| {r['width_path_spearman']:.4f} "
            f"|"
        )

    lines += [
        "",
        "## Cross-instrument summary "
        "(H4)",
        "",
        "| model | feature | median Q10 | "
        "pos | median Q90 | pos | "
        "median interval | pos | "
        "median width/path |",
        "|---|---|---:|---:|---:|---:"
        "|---:|---:|---:|",
    ]

    s4 = summary[
        summary[
            "horizon"
        ]
        == PRIMARY_HORIZON
    ]

    for (
        _,
        r,
    ) in s4.sort_values(
        [
            "model",
            "feature_set",
        ]
    ).iterrows():

        lines.append(
            f"| {r['model']} "
            f"| {r['feature_set']} "
            f"| "
            f"{r['median_q10_pinball_skill']:+.5f} "
            f"| "
            f"{r['positive_share_q10_pinball_skill']:.2f} "
            f"| "
            f"{r['median_q90_pinball_skill']:+.5f} "
            f"| "
            f"{r['positive_share_q90_pinball_skill']:.2f} "
            f"| "
            f"{r['median_interval_skill']:+.5f} "
            f"| "
            f"{r['positive_share_interval_skill']:.2f} "
            f"| "
            f"{r['median_width_path_spearman']:.4f} "
            f"|"
        )

    (
        OUT
        / "run_summary.md"
    ).write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 64
    )

    print(
        "CROSS SUMMARY (H4)"
    )

    print(
        "=" * 64
    )

    cols = [
        "model",
        "feature_set",
        "median_q10_pinball_skill",
        "positive_share_q10_pinball_skill",
        "median_q90_pinball_skill",
        "positive_share_q90_pinball_skill",
        "median_interval_skill",
        "positive_share_interval_skill",
        "median_width_path_spearman",
        "positive_share_width_path_spearman",
    ]

    with pd.option_context(
        "display.width",
        250,
    ):
        print(
            s4[cols].to_string(
                index=False
            )
        )

    print(
        "\nQUANTILE_REBASELINE_PASS"
    )


if __name__ == "__main__":
    main()
