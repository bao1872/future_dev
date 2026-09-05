#!/usr/bin/env python3
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import mean_pinball_loss

from fit_quantile_v2_models import (
    make_model,
    make_folds,
    interval_score,
    safe_spearman,
    q50_decile_table,
    width_decile_table,
)


ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    ROOT
    / "research"
    / "exports"
    / "quantile_v2_robustness_data"
)

INPUT_MANIFEST = (
    INPUT_DIR
    / "manifest.json"
)

OUT = (
    ROOT
    / "research"
    / "exports"
    / "quantile_v2_robustness_model"
)


# ============================================================
# Locked instrument design
# ============================================================

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

DISCOVERY_INSTRUMENT = "AG"

HOLDOUT_7 = (
    "CU",
    "AL",
    "SN",
    "I",
    "SC",
    "M",
    "CF",
)

METAL_HOLDOUT = (
    "CU",
    "AL",
    "SN",
    "I",
)

DRIVER_DIVERSE_HOLDOUT = (
    "CU",
    "I",
    "SC",
    "M",
    "CF",
)

GROUPS = {
    "ALL_8": INSTRUMENTS,
    "HOLDOUT_7": HOLDOUT_7,
    "METAL_HOLDOUT": METAL_HOLDOUT,
    "DRIVER_DIVERSE_HOLDOUT": (
        DRIVER_DIVERSE_HOLDOUT
    ),
}


# ============================================================
# Same Silver model design
# ============================================================

HORIZONS = (
    4,
    8,
    16,
)

QUANTILES = (
    0.10,
    0.50,
    0.90,
)

MODELS = (
    "linear_qr",
    "gbr_quantile",
)


F1_DIRECTION = [
    "feat_15m_ret_1",
    "feat_15m_ret_4",
    "feat_15m_ret_8",
    "feat_15m_ret_16",
    "feat_15m_location_32",
    "feat_time_bars_since_segment_start",
    "feat_time_after_long_gap",
]

F2_ACTIVITY = (
    F1_DIRECTION
    + [
        "feat_15m_volume_ratio_32",
        "feat_15m_oi_log_change_4",
    ]
)

F3_VOL = (
    F2_ACTIVITY
    + [
        "feat_5m_1h_rv",
        "feat_5m_rv_rate_ratio_1h_4h",
    ]
)

FEATURE_SETS = {
    "F1_DIRECTION": F1_DIRECTION,
    "F2_ACTIVITY": F2_ACTIVITY,
    "F3_VOL": F3_VOL,
}


# ============================================================
# Output helpers
# ============================================================

def prepare_output_dir() -> None:

    if OUT.exists():

        existing = list(
            OUT.iterdir()
        )

        if existing:

            raise RuntimeError(
                f"{OUT} exists and is non-empty. "
                "Delete only for an intentional "
                "pre-commit rerun."
            )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )


def safe_pearson(
    x,
    y,
) -> float:

    a = pd.Series(
        x,
        dtype=float,
    )

    b = pd.Series(
        y,
        dtype=float,
    )

    valid = (
        a.notna()
        &
        b.notna()
    )

    if valid.sum() < 3:
        return float("nan")

    a = a[valid]
    b = b[valid]

    if (
        a.nunique() < 2
        or b.nunique() < 2
    ):
        return float("nan")

    return float(
        a.corr(
            b,
            method="pearson",
        )
    )


# ============================================================
# Load and validate 8 panels
# ============================================================

def load_panels():

    manifest = json.loads(
        INPUT_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected_features = list(
        manifest[
            "feature_schema"
        ]
    )

    if expected_features != F3_VOL:

        raise RuntimeError(
            "Robustness feature schema "
            "does not equal locked F3_VOL"
        )

    panels = {}

    expected_columns = None

    for instrument in INSTRUMENTS:

        path = (
            INPUT_DIR
            / f"{instrument}_panel.csv"
        )

        if not path.is_file():

            raise RuntimeError(
                f"Missing panel: {path}"
            )

        df = pd.read_csv(
            path,
            parse_dates=[
                "meta_base_bar_time",
                "meta_decision_time",
            ],
            low_memory=False,
        )

        if len(df) != 3300:

            raise RuntimeError(
                f"{instrument}: "
                f"rows={len(df)} !=3300"
            )

        missing = [
            col
            for col in F3_VOL
            if col not in df.columns
        ]

        if missing:

            raise RuntimeError(
                f"{instrument}: "
                f"missing={missing}"
            )

        feature_cols = [
            c
            for c in df.columns
            if c.startswith(
                "feat_"
            )
        ]

        if any(
            token in c.lower()
            for c in feature_cols
            for token in (
                "smc",
                "momentum",
                "oracle",
                "sqz",
                "semivar",
                "jump",
            )
        ):

            raise RuntimeError(
                f"{instrument}: "
                "forbidden feature found"
            )

        if expected_columns is None:

            expected_columns = list(
                df.columns
            )

        elif (
            list(df.columns)
            != expected_columns
        ):

            raise RuntimeError(
                f"{instrument}: "
                "schema/order differs"
            )

        panels[
            instrument
        ] = df

    return (
        panels,
        manifest,
    )


# ============================================================
# Cross-instrument dependence
#
# This is descriptive only.
# It tells us how far the markets are from IID.
# ============================================================

def build_dependence_table(
    panels,
):

    rows = []

    for (
        a_name,
        b_name,
    ) in combinations(
        INSTRUMENTS,
        2,
    ):

        a = panels[
            a_name
        ][
            [
                "meta_decision_time",
                "feat_15m_ret_1",
                "feat_5m_1h_rv",
            ]
        ].rename(
            columns={
                "feat_15m_ret_1": (
                    "ret_a"
                ),

                "feat_5m_1h_rv": (
                    "rv_a"
                ),
            }
        )

        b = panels[
            b_name
        ][
            [
                "meta_decision_time",
                "feat_15m_ret_1",
                "feat_5m_1h_rv",
            ]
        ].rename(
            columns={
                "feat_15m_ret_1": (
                    "ret_b"
                ),

                "feat_5m_1h_rv": (
                    "rv_b"
                ),
            }
        )

        joined = a.merge(
            b,
            on="meta_decision_time",
            how="inner",
        )

        rows.append(
            {
                "instrument_a": (
                    a_name
                ),

                "instrument_b": (
                    b_name
                ),

                "overlap_rows": int(
                    len(joined)
                ),

                "ret1_pearson": (
                    safe_pearson(
                        joined[
                            "ret_a"
                        ],
                        joined[
                            "ret_b"
                        ],
                    )
                ),

                "ret1_spearman": (
                    safe_spearman(
                        joined[
                            "ret_a"
                        ],
                        joined[
                            "ret_b"
                        ],
                    )
                ),

                "rv1h_pearson": (
                    safe_pearson(
                        joined[
                            "rv_a"
                        ],
                        joined[
                            "rv_b"
                        ],
                    )
                ),

                "rv1h_spearman": (
                    safe_spearman(
                        joined[
                            "rv_a"
                        ],
                        joined[
                            "rv_b"
                        ],
                    )
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if len(result) != 28:

        raise RuntimeError(
            "Expected 28 instrument pairs"
        )

    return result


# ============================================================
# Fold distribution metrics
# ============================================================

def fold_distribution_stats(
    y,
    q10,
    q50,
    q90,
    b10,
    b90,
):

    y = np.asarray(
        y,
        dtype=float,
    )

    q10 = np.asarray(
        q10,
        dtype=float,
    )

    q50 = np.asarray(
        q50,
        dtype=float,
    )

    q90 = np.asarray(
        q90,
        dtype=float,
    )

    b10 = np.asarray(
        b10,
        dtype=float,
    )

    b90 = np.asarray(
        b90,
        dtype=float,
    )

    crossing = (
        (q10 > q50)
        |
        (q50 > q90)
    )

    ordered = (
        ~crossing
    )

    if ordered.any():

        coverage = float(
            (
                (
                    y[
                        ordered
                    ]
                    >=
                    q10[
                        ordered
                    ]
                )
                &
                (
                    y[
                        ordered
                    ]
                    <=
                    q90[
                        ordered
                    ]
                )
            ).mean()
        )

        model_score = float(
            np.mean(
                interval_score(
                    y[
                        ordered
                    ],
                    q10[
                        ordered
                    ],
                    q90[
                        ordered
                    ],
                    alpha=0.20,
                )
            )
        )

        base_score = float(
            np.mean(
                interval_score(
                    y[
                        ordered
                    ],
                    b10[
                        ordered
                    ],
                    b90[
                        ordered
                    ],
                    alpha=0.20,
                )
            )
        )

        score_skill = (
            1.0
            -
            model_score
            /
            base_score
            if base_score > 0
            else np.nan
        )

    else:

        coverage = np.nan
        model_score = np.nan
        base_score = np.nan
        score_skill = np.nan

    return {
        "crossing_rate": float(
            crossing.mean()
        ),

        "ordered_share": float(
            ordered.mean()
        ),

        "interval_80_coverage_ordered": (
            coverage
        ),

        "interval_score": (
            model_score
        ),

        "baseline_interval_score": (
            base_score
        ),

        "interval_score_skill": (
            score_skill
        ),

        "q50_spearman": (
            safe_spearman(
                q50,
                y,
            )
        ),

        "q50_sign_accuracy": float(
            (
                (q50 > 0)
                ==
                (y > 0)
            ).mean()
        ),
    }


# ============================================================
# One instrument / horizon / feature set / model
# ============================================================

def evaluate_combo(
    *,
    instrument,
    frame,
    horizon,
    feature_set_name,
    feature_cols,
    model_name,
):

    target_col = (
        f"target_raw_return_h"
        f"{horizon}"
    )

    X_all = (
        frame[
            feature_cols
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    y_all = pd.to_numeric(
        frame[
            target_col
        ],
        errors="coerce",
    )

    folds = make_folds(
        len(frame),
        horizon=horizon,
    )

    pred = {
        q: np.full(
            len(frame),
            np.nan,
            dtype=float,
        )
        for q
        in QUANTILES
    }

    baseline = {
        q: np.full(
            len(frame),
            np.nan,
            dtype=float,
        )
        for q
        in QUANTILES
    }

    fold_ids = np.full(
        len(frame),
        -1,
        dtype=int,
    )

    fold_quantile_rows = []
    fold_distribution_rows = []
    linear_effect_rows = []

    for fold in folds:

        train_idx = np.arange(
            fold[
                "train_start"
            ],
            fold[
                "train_end_exclusive"
            ],
        )

        test_idx = np.arange(
            fold[
                "test_start"
            ],
            fold[
                "test_end_exclusive"
            ],
        )

        if (
            test_idx[0]
            -
            fold[
                "train_end_exclusive"
            ]
        ) < horizon:

            raise RuntimeError(
                "Purge invariant failed"
            )

        X_train = (
            X_all.iloc[
                train_idx
            ]
        )

        X_test = (
            X_all.iloc[
                test_idx
            ]
        )

        y_train = (
            y_all.iloc[
                train_idx
            ]
        )

        y_test = (
            y_all.iloc[
                test_idx
            ]
        )

        fold_pred = {}
        fold_base = {}

        for q in QUANTILES:

            model = make_model(
                model_name,
                q,
            )

            model.fit(
                X_train,
                y_train,
            )

            p = np.asarray(
                model.predict(
                    X_test
                ),
                dtype=float,
            )

            b = float(
                y_train.quantile(
                    q
                )
            )

            pred[
                q
            ][
                test_idx
            ] = p

            baseline[
                q
            ][
                test_idx
            ] = b

            fold_pred[
                q
            ] = p

            fold_base[
                q
            ] = np.full(
                len(test_idx),
                b,
                dtype=float,
            )

            model_loss = float(
                mean_pinball_loss(
                    y_test,
                    p,
                    alpha=q,
                )
            )

            base_loss = float(
                mean_pinball_loss(
                    y_test,
                    fold_base[
                        q
                    ],
                    alpha=q,
                )
            )

            skill = (
                1.0
                -
                model_loss
                /
                base_loss
                if base_loss > 0
                else np.nan
            )

            calibration = float(
                (
                    y_test.to_numpy()
                    <= p
                ).mean()
            )

            fold_quantile_rows.append(
                {
                    "instrument": (
                        instrument
                    ),

                    "horizon": (
                        horizon
                    ),

                    "feature_set": (
                        feature_set_name
                    ),

                    "model": (
                        model_name
                    ),

                    "quantile": q,

                    "fold": (
                        fold[
                            "fold"
                        ]
                    ),

                    "train_rows": int(
                        len(train_idx)
                    ),

                    "test_rows": int(
                        len(test_idx)
                    ),

                    "pinball_loss": (
                        model_loss
                    ),

                    "baseline_pinball_loss": (
                        base_loss
                    ),

                    "pinball_skill": (
                        skill
                    ),

                    "calibration": (
                        calibration
                    ),

                    "calibration_error": (
                        calibration
                        - q
                    ),
                }
            )

            # --------------------------------------------
            # Feature-sign robustness:
            # only standardized linear F3 coefficients.
            # --------------------------------------------

            if (
                model_name
                == "linear_qr"
                and
                feature_set_name
                == "F3_VOL"
            ):

                coef = np.asarray(
                    model.named_steps[
                        "model"
                    ].coef_,
                    dtype=float,
                )

                if len(
                    coef
                ) != len(
                    feature_cols
                ):

                    raise RuntimeError(
                        "Coefficient length mismatch"
                    )

                for feature, value in zip(
                    feature_cols,
                    coef,
                ):

                    linear_effect_rows.append(
                        {
                            "instrument": (
                                instrument
                            ),

                            "horizon": (
                                horizon
                            ),

                            "quantile": q,

                            "fold": (
                                fold[
                                    "fold"
                                ]
                            ),

                            "feature": (
                                feature
                            ),

                            "coefficient": float(
                                value
                            ),
                        }
                    )

        dist = fold_distribution_stats(
            y_test.to_numpy(
                dtype=float
            ),
            fold_pred[
                0.10
            ],
            fold_pred[
                0.50
            ],
            fold_pred[
                0.90
            ],
            fold_base[
                0.10
            ],
            fold_base[
                0.90
            ],
        )

        fold_distribution_rows.append(
            {
                "instrument": (
                    instrument
                ),

                "horizon": (
                    horizon
                ),

                "feature_set": (
                    feature_set_name
                ),

                "model": (
                    model_name
                ),

                "fold": (
                    fold[
                        "fold"
                    ]
                ),

                **dist,
            }
        )

        fold_ids[
            test_idx
        ] = (
            fold[
                "fold"
            ]
        )

    # ========================================================
    # Pooled OOS
    # ========================================================

    oos = (
        fold_ids
        >= 0
    )

    if int(
        oos.sum()
    ) != 2250:

        raise RuntimeError(
            f"{instrument} "
            f"H={horizon} "
            f"{feature_set_name} "
            f"{model_name}: "
            f"OOS rows={oos.sum()}"
        )

    pred_frame = pd.DataFrame(
        {
            "decision_time": (
                frame.loc[
                    oos,
                    "meta_decision_time",
                ].to_numpy()
            ),

            "horizon": (
                horizon
            ),

            "feature_set": (
                feature_set_name
            ),

            "model": (
                model_name
            ),

            "fold": (
                fold_ids[
                    oos
                ]
            ),

            "realized_return": (
                y_all[
                    oos
                ].to_numpy()
            ),

            "realized_path_range": (
                frame.loc[
                    oos,
                    "realized_path_range",
                ].to_numpy()
            ),

            "pred_q10": (
                pred[
                    0.10
                ][
                    oos
                ]
            ),

            "pred_q50": (
                pred[
                    0.50
                ][
                    oos
                ]
            ),

            "pred_q90": (
                pred[
                    0.90
                ][
                    oos
                ]
            ),

            "baseline_q10": (
                baseline[
                    0.10
                ][
                    oos
                ]
            ),

            "baseline_q50": (
                baseline[
                    0.50
                ][
                    oos
                ]
            ),

            "baseline_q90": (
                baseline[
                    0.90
                ][
                    oos
                ]
            ),
        }
    )

    # ========================================================
    # Pooled quantile metrics
    # ========================================================

    pooled_quantile_rows = []

    for q, pred_col, base_col in (
        (
            0.10,
            "pred_q10",
            "baseline_q10",
        ),
        (
            0.50,
            "pred_q50",
            "baseline_q50",
        ),
        (
            0.90,
            "pred_q90",
            "baseline_q90",
        ),
    ):

        y = pred_frame[
            "realized_return"
        ]

        p = pred_frame[
            pred_col
        ]

        b = pred_frame[
            base_col
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

        skill = (
            1.0
            -
            model_loss
            /
            base_loss
            if base_loss > 0
            else np.nan
        )

        fold_q = pd.DataFrame(
            fold_quantile_rows
        )

        fold_q = fold_q[
            fold_q[
                "quantile"
            ]
            == q
        ]

        pooled_quantile_rows.append(
            {
                "instrument": (
                    instrument
                ),

                "horizon": (
                    horizon
                ),

                "feature_set": (
                    feature_set_name
                ),

                "model": (
                    model_name
                ),

                "quantile": q,

                "oos_rows": int(
                    len(y)
                ),

                "pinball_loss": (
                    model_loss
                ),

                "baseline_pinball_loss": (
                    base_loss
                ),

                "pinball_skill": (
                    skill
                ),

                "calibration": float(
                    (
                        y.to_numpy()
                        <=
                        p.to_numpy()
                    ).mean()
                ),

                "calibration_error": float(
                    (
                        y.to_numpy()
                        <=
                        p.to_numpy()
                    ).mean()
                    - q
                ),

                "positive_skill_fold_share": float(
                    (
                        fold_q[
                            "pinball_skill"
                        ]
                        > 0
                    ).mean()
                ),

                "median_fold_skill": float(
                    fold_q[
                        "pinball_skill"
                    ].median()
                ),
            }
        )

    # ========================================================
    # Pooled distribution metrics
    # ========================================================

    dist = fold_distribution_stats(
        pred_frame[
            "realized_return"
        ],
        pred_frame[
            "pred_q10"
        ],
        pred_frame[
            "pred_q50"
        ],
        pred_frame[
            "pred_q90"
        ],
        pred_frame[
            "baseline_q10"
        ],
        pred_frame[
            "baseline_q90"
        ],
    )

    q50_deciles = (
        q50_decile_table(
            pred_frame,
            horizon=horizon,
            feature_set=(
                feature_set_name
            ),
            model=model_name,
        )
    )

    if not q50_deciles.empty:

        q50_deciles.insert(
            0,
            "instrument",
            instrument,
        )

        q50_deciles = (
            q50_deciles.sort_values(
                "decile"
            )
        )

        q50_mono = safe_spearman(
            q50_deciles[
                "decile"
            ],
            q50_deciles[
                "realized_median"
            ],
        )

        q50_top_bottom = float(
            q50_deciles.iloc[
                -1
            ][
                "realized_median"
            ]
            -
            q50_deciles.iloc[
                0
            ][
                "realized_median"
            ]
        )

    else:

        q50_mono = np.nan
        q50_top_bottom = np.nan

    width_deciles = (
        width_decile_table(
            pred_frame,
            horizon=horizon,
            feature_set=(
                feature_set_name
            ),
            model=model_name,
        )
    )

    if not width_deciles.empty:

        width_deciles.insert(
            0,
            "instrument",
            instrument,
        )

        width_deciles = (
            width_deciles.sort_values(
                "decile"
            )
        )

        width_abs_mono = (
            safe_spearman(
                width_deciles[
                    "decile"
                ],
                width_deciles[
                    "median_abs_return"
                ],
            )
        )

        width_path_mono = (
            safe_spearman(
                width_deciles[
                    "decile"
                ],
                width_deciles[
                    "median_path_range"
                ],
            )
        )

        width_path_top_bottom = float(
            width_deciles.iloc[
                -1
            ][
                "median_path_range"
            ]
            -
            width_deciles.iloc[
                0
            ][
                "median_path_range"
            ]
        )

    else:

        width_abs_mono = np.nan
        width_path_mono = np.nan
        width_path_top_bottom = np.nan

    fold_dist = pd.DataFrame(
        fold_distribution_rows
    )

    distribution_row = {
        "instrument": (
            instrument
        ),

        "horizon": (
            horizon
        ),

        "feature_set": (
            feature_set_name
        ),

        "model": (
            model_name
        ),

        "oos_rows": 2250,

        **dist,

        "positive_interval_skill_fold_share": float(
            (
                fold_dist[
                    "interval_score_skill"
                ]
                > 0
            ).mean()
        ),

        "median_fold_interval_skill": float(
            fold_dist[
                "interval_score_skill"
            ].median()
        ),

        "q50_decile_monotonicity": (
            q50_mono
        ),

        "q50_top_minus_bottom_median": (
            q50_top_bottom
        ),

        "width_abs_return_monotonicity": (
            width_abs_mono
        ),

        "width_path_range_monotonicity": (
            width_path_mono
        ),

        "width_top_minus_bottom_path_range": (
            width_path_top_bottom
        ),
    }

    return {
        "fold_quantile": pd.DataFrame(
            fold_quantile_rows
        ),

        "fold_distribution": pd.DataFrame(
            fold_distribution_rows
        ),

        "quantile_metrics": pd.DataFrame(
            pooled_quantile_rows
        ),

        "distribution_metrics": pd.DataFrame(
            [
                distribution_row
            ]
        ),

        "q50_deciles": q50_deciles,

        "width_deciles": width_deciles,

        "linear_effects": pd.DataFrame(
            linear_effect_rows
        ),
    }


# ============================================================
# Merge Q metrics into per-instrument model summary
# ============================================================

def build_model_summary(
    quantile_metrics,
    distribution_metrics,
):

    q = (
        quantile_metrics
        .pivot_table(
            index=[
                "instrument",
                "horizon",
                "feature_set",
                "model",
            ],
            columns="quantile",
            values="pinball_skill",
        )
        .reset_index()
        .rename(
            columns={
                0.10: (
                    "q10_pinball_skill"
                ),

                0.50: (
                    "q50_pinball_skill"
                ),

                0.90: (
                    "q90_pinball_skill"
                ),
            }
        )
    )

    out = (
        distribution_metrics
        .merge(
            q,
            on=[
                "instrument",
                "horizon",
                "feature_set",
                "model",
            ],
            how="left",
        )
    )

    out[
        "mean_pinball_skill"
    ] = (
        out[
            [
                "q10_pinball_skill",
                "q50_pinball_skill",
                "q90_pinball_skill",
            ]
        ]
        .mean(
            axis=1
        )
    )

    out[
        "tail_pinball_skill"
    ] = (
        out[
            [
                "q10_pinball_skill",
                "q90_pinball_skill",
            ]
        ]
        .mean(
            axis=1
        )
    )

    out[
        "abs_interval_coverage_error"
    ] = (
        out[
            "interval_80_coverage_ordered"
        ]
        - 0.80
    ).abs()

    return out


# ============================================================
# Nested feature-set delta by instrument
# ============================================================

def build_feature_deltas(
    model_summary,
):

    rows = []

    transitions = (
        (
            "F1_DIRECTION",
            "F2_ACTIVITY",
        ),
        (
            "F2_ACTIVITY",
            "F3_VOL",
        ),
    )

    metrics = (
        "mean_pinball_skill",
        "tail_pinball_skill",
        "q10_pinball_skill",
        "q50_pinball_skill",
        "q90_pinball_skill",
        "interval_score_skill",
        "q50_decile_monotonicity",
        "width_path_range_monotonicity",
    )

    for instrument in INSTRUMENTS:

        for h in HORIZONS:

            for model in MODELS:

                x = (
                    model_summary[
                        (
                            model_summary[
                                "instrument"
                            ]
                            == instrument
                        )
                        &
                        (
                            model_summary[
                                "horizon"
                            ]
                            == h
                        )
                        &
                        (
                            model_summary[
                                "model"
                            ]
                            == model
                        )
                    ]
                    .set_index(
                        "feature_set"
                    )
                )

                for (
                    old,
                    new,
                ) in transitions:

                    row = {
                        "instrument": (
                            instrument
                        ),

                        "horizon": h,

                        "model": model,

                        "from_feature_set": (
                            old
                        ),

                        "to_feature_set": (
                            new
                        ),
                    }

                    for metric in metrics:

                        row[
                            f"delta_{metric}"
                        ] = float(
                            x.loc[
                                new,
                                metric,
                            ]
                            -
                            x.loc[
                                old,
                                metric,
                            ]
                        )

                    rows.append(
                        row
                    )

    result = pd.DataFrame(
        rows
    )

    if len(result) != 96:

        raise RuntimeError(
            f"Feature delta rows "
            f"{len(result)} !=96"
        )

    return result


# ============================================================
# Cross-instrument model aggregation
# ============================================================

def build_cross_instrument_summary(
    model_summary,
):

    rows = []

    for (
        group_name,
        members,
    ) in GROUPS.items():

        x = model_summary[
            model_summary[
                "instrument"
            ].isin(
                members
            )
        ]

        for (
            h,
            feature_set,
            model,
        ), g in x.groupby(
            [
                "horizon",
                "feature_set",
                "model",
            ],
            observed=True,
        ):

            def positive_share(
                col,
            ):

                s = pd.to_numeric(
                    g[col],
                    errors="coerce",
                ).dropna()

                return float(
                    (
                        s > 0
                    ).mean()
                )

            rows.append(
                {
                    "group": (
                        group_name
                    ),

                    "horizon": int(
                        h
                    ),

                    "feature_set": (
                        feature_set
                    ),

                    "model": (
                        model
                    ),

                    "instrument_count": int(
                        g[
                            "instrument"
                        ].nunique()
                    ),

                    "median_q10_skill": float(
                        g[
                            "q10_pinball_skill"
                        ].median()
                    ),

                    "positive_q10_share": (
                        positive_share(
                            "q10_pinball_skill"
                        )
                    ),

                    "median_q50_skill": float(
                        g[
                            "q50_pinball_skill"
                        ].median()
                    ),

                    "positive_q50_share": (
                        positive_share(
                            "q50_pinball_skill"
                        )
                    ),

                    "median_q90_skill": float(
                        g[
                            "q90_pinball_skill"
                        ].median()
                    ),

                    "positive_q90_share": (
                        positive_share(
                            "q90_pinball_skill"
                        )
                    ),

                    "median_tail_skill": float(
                        g[
                            "tail_pinball_skill"
                        ].median()
                    ),

                    "positive_tail_share": (
                        positive_share(
                            "tail_pinball_skill"
                        )
                    ),

                    "median_mean_skill": float(
                        g[
                            "mean_pinball_skill"
                        ].median()
                    ),

                    "positive_mean_share": (
                        positive_share(
                            "mean_pinball_skill"
                        )
                    ),

                    "median_interval_skill": float(
                        g[
                            "interval_score_skill"
                        ].median()
                    ),

                    "positive_interval_share": (
                        positive_share(
                            "interval_score_skill"
                        )
                    ),

                    "median_abs_coverage_error": float(
                        g[
                            "abs_interval_coverage_error"
                        ].median()
                    ),

                    "median_q50_monotonicity": float(
                        g[
                            "q50_decile_monotonicity"
                        ].median()
                    ),

                    "positive_q50_mono_share": (
                        positive_share(
                            "q50_decile_monotonicity"
                        )
                    ),

                    "median_width_path_monotonicity": float(
                        g[
                            "width_path_range_monotonicity"
                        ].median()
                    ),

                    "positive_width_path_mono_share": (
                        positive_share(
                            "width_path_range_monotonicity"
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Cross-instrument feature delta
# ============================================================

def build_cross_delta_summary(
    deltas,
):

    rows = []

    metric_names = [
        col
        for col in deltas.columns
        if col.startswith(
            "delta_"
        )
    ]

    for (
        group_name,
        members,
    ) in GROUPS.items():

        x = deltas[
            deltas[
                "instrument"
            ].isin(
                members
            )
        ]

        for keys, g in x.groupby(
            [
                "horizon",
                "model",
                "from_feature_set",
                "to_feature_set",
            ],
            observed=True,
        ):

            (
                h,
                model,
                old,
                new,
            ) = keys

            row = {
                "group": (
                    group_name
                ),

                "horizon": int(
                    h
                ),

                "model": (
                    model
                ),

                "from_feature_set": (
                    old
                ),

                "to_feature_set": (
                    new
                ),

                "instrument_count": int(
                    g[
                        "instrument"
                    ].nunique()
                ),
            }

            for metric in metric_names:

                values = pd.to_numeric(
                    g[
                        metric
                    ],
                    errors="coerce",
                ).dropna()

                row[
                    f"median_{metric}"
                ] = float(
                    values.median()
                )

                row[
                    f"positive_share_{metric}"
                ] = float(
                    (
                        values > 0
                    ).mean()
                )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Linear feature sign robustness
# ============================================================

def build_linear_effect_robustness(
    effects,
):

    instrument_rows = []

    for (
        instrument,
        h,
        q,
        feature,
    ), g in effects.groupby(
        [
            "instrument",
            "horizon",
            "quantile",
            "feature",
        ],
        observed=True,
    ):

        values = pd.to_numeric(
            g[
                "coefficient"
            ],
            errors="coerce",
        ).dropna()

        positive_share = float(
            (
                values > 0
            ).mean()
        )

        negative_share = float(
            (
                values < 0
            ).mean()
        )

        instrument_rows.append(
            {
                "instrument": (
                    instrument
                ),

                "horizon": int(
                    h
                ),

                "quantile": float(
                    q
                ),

                "feature": (
                    feature
                ),

                "mean_coefficient": float(
                    values.mean()
                ),

                "positive_fold_share": (
                    positive_share
                ),

                "fold_sign_consistency": max(
                    positive_share,
                    negative_share,
                ),
            }
        )

    instrument_df = pd.DataFrame(
        instrument_rows
    )

    cross_rows = []

    for (
        group_name,
        members,
    ) in GROUPS.items():

        x = instrument_df[
            instrument_df[
                "instrument"
            ].isin(
                members
            )
        ]

        for (
            h,
            q,
            feature,
        ), g in x.groupby(
            [
                "horizon",
                "quantile",
                "feature",
            ],
            observed=True,
        ):

            positive_share = float(
                (
                    g[
                        "mean_coefficient"
                    ]
                    > 0
                ).mean()
            )

            negative_share = float(
                (
                    g[
                        "mean_coefficient"
                    ]
                    < 0
                ).mean()
            )

            cross_rows.append(
                {
                    "group": (
                        group_name
                    ),

                    "horizon": int(
                        h
                    ),

                    "quantile": float(
                        q
                    ),

                    "feature": (
                        feature
                    ),

                    "instrument_count": int(
                        g[
                            "instrument"
                        ].nunique()
                    ),

                    "positive_instrument_share": (
                        positive_share
                    ),

                    "instrument_sign_consistency": max(
                        positive_share,
                        negative_share,
                    ),

                    "median_within_instrument_fold_sign_consistency": float(
                        g[
                            "fold_sign_consistency"
                        ].median()
                    ),
                }
            )

    return (
        instrument_df,
        pd.DataFrame(
            cross_rows
        ),
    )


# ============================================================
# Pre-specified Silver-derived hypotheses
# applied ONLY to HOLDOUT_7
# ============================================================

def build_primary_hypotheses(
    model_summary,
    quantile_metrics,
    deltas,
):

    detail_rows = []

    def add_detail(
        hypothesis,
        values,
    ):

        for (
            instrument,
            value,
        ) in values:

            detail_rows.append(
                {
                    "hypothesis": (
                        hypothesis
                    ),

                    "instrument": (
                        instrument
                    ),

                    "value": float(
                        value
                    ),

                    "positive": bool(
                        value > 0
                    ),
                }
            )

    # H1: H4 GBR F1 Q10
    q = quantile_metrics[
        (
            quantile_metrics[
                "instrument"
            ].isin(
                HOLDOUT_7
            )
        )
        &
        (
            quantile_metrics[
                "horizon"
            ] == 4
        )
        &
        (
            quantile_metrics[
                "feature_set"
            ] == "F1_DIRECTION"
        )
        &
        (
            quantile_metrics[
                "model"
            ] == "gbr_quantile"
        )
        &
        (
            quantile_metrics[
                "quantile"
            ] == 0.10
        )
    ]

    add_detail(
        "H1_H4_GBR_F1_Q10_SKILL",
        zip(
            q[
                "instrument"
            ],
            q[
                "pinball_skill"
            ],
        ),
    )

    # H2: H4 GBR F1 Q90
    q = quantile_metrics[
        (
            quantile_metrics[
                "instrument"
            ].isin(
                HOLDOUT_7
            )
        )
        &
        (
            quantile_metrics[
                "horizon"
            ] == 4
        )
        &
        (
            quantile_metrics[
                "feature_set"
            ] == "F1_DIRECTION"
        )
        &
        (
            quantile_metrics[
                "model"
            ] == "gbr_quantile"
        )
        &
        (
            quantile_metrics[
                "quantile"
            ] == 0.90
        )
    ]

    add_detail(
        "H2_H4_GBR_F1_Q90_SKILL",
        zip(
            q[
                "instrument"
            ],
            q[
                "pinball_skill"
            ],
        ),
    )

    base = model_summary[
        (
            model_summary[
                "instrument"
            ].isin(
                HOLDOUT_7
            )
        )
        &
        (
            model_summary[
                "horizon"
            ] == 4
        )
        &
        (
            model_summary[
                "feature_set"
            ] == "F1_DIRECTION"
        )
        &
        (
            model_summary[
                "model"
            ] == "gbr_quantile"
        )
    ]

    add_detail(
        "H3_H4_GBR_F1_INTERVAL_SKILL",
        zip(
            base[
                "instrument"
            ],
            base[
                "interval_score_skill"
            ],
        ),
    )

    add_detail(
        "H4_H4_GBR_F1_WIDTH_PATH_MONO",
        zip(
            base[
                "instrument"
            ],
            base[
                "width_path_range_monotonicity"
            ],
        ),
    )

    q = quantile_metrics[
        (
            quantile_metrics[
                "instrument"
            ].isin(
                HOLDOUT_7
            )
        )
        &
        (
            quantile_metrics[
                "horizon"
            ] == 8
        )
        &
        (
            quantile_metrics[
                "feature_set"
            ] == "F1_DIRECTION"
        )
        &
        (
            quantile_metrics[
                "model"
            ] == "gbr_quantile"
        )
        &
        (
            quantile_metrics[
                "quantile"
            ] == 0.10
        )
    ]

    add_detail(
        "H5_H8_GBR_F1_Q10_SKILL",
        zip(
            q[
                "instrument"
            ],
            q[
                "pinball_skill"
            ],
        ),
    )

    d = deltas[
        (
            deltas[
                "instrument"
            ].isin(
                HOLDOUT_7
            )
        )
        &
        (
            deltas[
                "horizon"
            ] == 4
        )
        &
        (
            deltas[
                "model"
            ] == "gbr_quantile"
        )
        &
        (
            deltas[
                "from_feature_set"
            ] == "F2_ACTIVITY"
        )
        &
        (
            deltas[
                "to_feature_set"
            ] == "F3_VOL"
        )
    ]

    add_detail(
        "H6_H4_GBR_F2_TO_F3_TAIL_DELTA",
        zip(
            d[
                "instrument"
            ],
            d[
                "delta_tail_pinball_skill"
            ],
        ),
    )

    detail = pd.DataFrame(
        detail_rows
    )

    summary_rows = []

    for hypothesis, g in (
        detail.groupby(
            "hypothesis",
            observed=True,
        )
    ):

        summary_rows.append(
            {
                "hypothesis": (
                    hypothesis
                ),

                "instrument_count": int(
                    len(g)
                ),

                "median_value": float(
                    g[
                        "value"
                    ].median()
                ),

                "mean_value": float(
                    g[
                        "value"
                    ].mean()
                ),

                "positive_instrument_share": float(
                    g[
                        "positive"
                    ].mean()
                ),

                "min_value": float(
                    g[
                        "value"
                    ].min()
                ),

                "max_value": float(
                    g[
                        "value"
                    ].max()
                ),
            }
        )

    return (
        detail,
        pd.DataFrame(
            summary_rows
        ),
    )


# ============================================================
# Main
# ============================================================

def main():

    prepare_output_dir()

    panels, manifest = (
        load_panels()
    )

    dependence = (
        build_dependence_table(
            panels
        )
    )

    fold_quantile_all = []
    fold_distribution_all = []
    quantile_all = []
    distribution_all = []
    q50_deciles_all = []
    width_deciles_all = []
    linear_effects_all = []

    for instrument in INSTRUMENTS:

        source = panels[
            instrument
        ]

        print(
            "=" * 72
        )

        print(
            f"INSTRUMENT {instrument}"
        )

        print(
            "=" * 72
        )

        for h in HORIZONS:

            target_col = (
                f"target_raw_return_h"
                f"{h}"
            )

            long_mfe_col = (
                f"target_long_mfe_h"
                f"{h}"
            )

            short_mfe_col = (
                f"target_short_mfe_h"
                f"{h}"
            )

            valid = (
                source[
                    target_col
                ].notna()
                &
                source[
                    long_mfe_col
                ].notna()
                &
                source[
                    short_mfe_col
                ].notna()
            )

            frame = (
                source.loc[
                    valid
                ]
                .reset_index(
                    drop=True
                )
                .copy()
            )

            frame[
                "realized_path_range"
            ] = (
                frame[
                    long_mfe_col
                ]
                +
                frame[
                    short_mfe_col
                ]
            )

            for (
                feature_set_name,
                feature_cols,
            ) in (
                FEATURE_SETS.items()
            ):

                for model_name in MODELS:

                    result = (
                        evaluate_combo(
                            instrument=(
                                instrument
                            ),

                            frame=frame,

                            horizon=h,

                            feature_set_name=(
                                feature_set_name
                            ),

                            feature_cols=(
                                feature_cols
                            ),

                            model_name=(
                                model_name
                            ),
                        )
                    )

                    fold_quantile_all.append(
                        result[
                            "fold_quantile"
                        ]
                    )

                    fold_distribution_all.append(
                        result[
                            "fold_distribution"
                        ]
                    )

                    quantile_all.append(
                        result[
                            "quantile_metrics"
                        ]
                    )

                    distribution_all.append(
                        result[
                            "distribution_metrics"
                        ]
                    )

                    if not result[
                        "q50_deciles"
                    ].empty:

                        q50_deciles_all.append(
                            result[
                                "q50_deciles"
                            ]
                        )

                    if not result[
                        "width_deciles"
                    ].empty:

                        width_deciles_all.append(
                            result[
                                "width_deciles"
                            ]
                        )

                    if not result[
                        "linear_effects"
                    ].empty:

                        linear_effects_all.append(
                            result[
                                "linear_effects"
                            ]
                        )

    fold_quantile = pd.concat(
        fold_quantile_all,
        ignore_index=True,
    )

    fold_distribution = pd.concat(
        fold_distribution_all,
        ignore_index=True,
    )

    quantile_metrics = pd.concat(
        quantile_all,
        ignore_index=True,
    )

    distribution_metrics = pd.concat(
        distribution_all,
        ignore_index=True,
    )

    q50_deciles = pd.concat(
        q50_deciles_all,
        ignore_index=True,
    )

    width_deciles = pd.concat(
        width_deciles_all,
        ignore_index=True,
    )

    linear_effects = pd.concat(
        linear_effects_all,
        ignore_index=True,
    )

    model_summary = (
        build_model_summary(
            quantile_metrics,
            distribution_metrics,
        )
    )

    deltas = (
        build_feature_deltas(
            model_summary
        )
    )

    cross_summary = (
        build_cross_instrument_summary(
            model_summary
        )
    )

    cross_deltas = (
        build_cross_delta_summary(
            deltas
        )
    )

    (
        linear_effect_by_instrument,
        linear_effect_robustness,
    ) = build_linear_effect_robustness(
        linear_effects
    )

    (
        primary_detail,
        primary_summary,
    ) = build_primary_hypotheses(
        model_summary,
        quantile_metrics,
        deltas,
    )

    # ========================================================
    # Hard validation
    # ========================================================

    if len(
        fold_quantile
    ) != 2160:

        raise RuntimeError(
            f"fold quantile rows "
            f"{len(fold_quantile)} "
            "!=2160"
        )

    if len(
        quantile_metrics
    ) != 432:

        raise RuntimeError(
            f"quantile metrics rows "
            f"{len(quantile_metrics)} "
            "!=432"
        )

    if len(
        model_summary
    ) != 144:

        raise RuntimeError(
            f"model summary rows "
            f"{len(model_summary)} "
            "!=144"
        )

    if len(
        fold_distribution
    ) != 720:

        raise RuntimeError(
            f"fold distribution rows "
            f"{len(fold_distribution)} "
            "!=720"
        )

    if len(
        primary_detail
    ) != 42:

        raise RuntimeError(
            f"primary hypothesis detail "
            f"rows={len(primary_detail)} "
            "!=42"
        )

    if len(
        primary_summary
    ) != 6:

        raise RuntimeError(
            "primary hypothesis summary "
            f"rows={len(primary_summary)} "
            "!=6"
        )

    if (
        model_summary[
            [
                "instrument",
                "horizon",
                "feature_set",
                "model",
            ]
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate instrument "
            "model summary"
        )

    numeric_outputs = [
        dependence,
        fold_quantile,
        fold_distribution,
        quantile_metrics,
        distribution_metrics,
        model_summary,
        deltas,
        cross_summary,
        cross_deltas,
        q50_deciles,
        width_deciles,
        linear_effects,
        linear_effect_by_instrument,
        linear_effect_robustness,
        primary_detail,
        primary_summary,
    ]

    for df in numeric_outputs:

        numeric = (
            df.select_dtypes(
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
                "Output contains +/-inf"
            )

    # ========================================================
    # Save
    # ========================================================

    outputs = {
        "instrument_dependence.csv": (
            dependence
        ),

        "fold_quantile_metrics.csv": (
            fold_quantile
        ),

        "fold_distribution_metrics.csv": (
            fold_distribution
        ),

        "instrument_quantile_metrics.csv": (
            quantile_metrics
        ),

        "instrument_distribution_metrics.csv": (
            distribution_metrics
        ),

        "instrument_model_summary.csv": (
            model_summary
        ),

        "feature_set_deltas_by_instrument.csv": (
            deltas
        ),

        "cross_instrument_model_summary.csv": (
            cross_summary
        ),

        "cross_instrument_feature_deltas.csv": (
            cross_deltas
        ),

        "q50_deciles.csv": (
            q50_deciles
        ),

        "width_deciles.csv": (
            width_deciles
        ),

        "linear_f3_effects_by_fold.csv": (
            linear_effects
        ),

        "linear_f3_effects_by_instrument.csv": (
            linear_effect_by_instrument
        ),

        "linear_f3_effect_robustness.csv": (
            linear_effect_robustness
        ),

        "primary_hypothesis_detail.csv": (
            primary_detail
        ),

        "primary_hypothesis_summary.csv": (
            primary_summary
        ),
    }

    for name, df in outputs.items():

        df.to_csv(
            OUT / name,
            index=False,
        )

    config = {
        "purpose": (
            "Out-of-instrument "
            "Quantile V2 robustness "
            "replication."
        ),

        "discovery_instrument": (
            DISCOVERY_INSTRUMENT
        ),

        "holdout_instruments": list(
            HOLDOUT_7
        ),

        "groups": {
            k: list(v)
            for k, v
            in GROUPS.items()
        },

        "horizons": list(
            HORIZONS
        ),

        "quantiles": list(
            QUANTILES
        ),

        "models": list(
            MODELS
        ),

        "feature_sets": (
            FEATURE_SETS
        ),

        "primary_hypotheses": [
            "H1_H4_GBR_F1_Q10_SKILL",
            "H2_H4_GBR_F1_Q90_SKILL",
            "H3_H4_GBR_F1_INTERVAL_SKILL",
            "H4_H4_GBR_F1_WIDTH_PATH_MONO",
            "H5_H8_GBR_F1_Q10_SKILL",
            "H6_H4_GBR_F2_TO_F3_TAIL_DELTA",
        ],

        "model_parameters_changed_from_silver": (
            False
        ),

        "feature_parameters_changed_from_silver": (
            False
        ),

        "no_hyperparameter_search": (
            True
        ),

        "no_backtest": (
            True
        ),

        "no_pooled_model": (
            True
        ),

        "no_leave_one_instrument_out": (
            True
        ),

        "no_smc": True,
        "no_momentum": True,
        "no_oracle": True,
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

    file_sizes = {}

    for path in OUT.iterdir():

        if not path.is_file():
            continue

        size_mb = (
            path.stat().st_size
            / 1024
            / 1024
        )

        file_sizes[
            path.name
        ] = round(
            size_mb,
            4,
        )

        if size_mb > 50:

            raise RuntimeError(
                f"{path.name} >50MB"
            )

    validation = {
        "status": "PASS",

        "instrument_count": 8,

        "holdout_instrument_count": 7,

        "fold_count_per_horizon": 5,

        "feature_set_count": 3,

        "model_count": 2,

        "quantile_count": 3,

        "fold_quantile_rows": int(
            len(fold_quantile)
        ),

        "fold_distribution_rows": int(
            len(fold_distribution)
        ),

        "instrument_quantile_rows": int(
            len(quantile_metrics)
        ),

        "instrument_model_rows": int(
            len(model_summary)
        ),

        "primary_hypothesis_rows": int(
            len(primary_summary)
        ),

        "dependence_pair_rows": int(
            len(dependence)
        ),

        "no_model_parameter_changes": (
            True
        ),

        "no_feature_changes": True,

        "no_hyperparameter_search": (
            True
        ),

        "no_backtest": True,

        "no_pooled_model": True,

        "file_sizes_mb": (
            file_sizes
        ),
    }

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
    # Compact factual report
    # ========================================================

    lines = [
        "# Quantile V2 Cross-Instrument Robustness",
        "",
        "AG = discovery instrument.",
        "CU/AL/SN/I/SC/M/CF = holdout instruments.",
        "",
        "No tuning. No backtest. No pooled model.",
        "",
        "## Primary hypotheses on HOLDOUT_7",
        "",
        "| hypothesis | median | positive share | min | max |",
        "|---|---:|---:|---:|---:|",
    ]

    for _, row in (
        primary_summary
        .sort_values(
            "hypothesis"
        )
        .iterrows()
    ):

        lines.append(
            f"| {row['hypothesis']} "
            f"| {row['median_value']:.5f} "
            f"| {row['positive_instrument_share']:.3f} "
            f"| {row['min_value']:.5f} "
            f"| {row['max_value']:.5f} |"
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
        json.dumps(
            validation,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "QUANTILE_V2_ROBUSTNESS_MODEL_PASS"
    )


if __name__ == "__main__":
    main()
