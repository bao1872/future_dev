#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "research"
    / "exports"
    / "quantile_v2_data"
    / "quantile_panel_15m.csv"
)

SOURCE_VALIDATION = (
    ROOT
    / "research"
    / "exports"
    / "quantile_v2_data"
    / "validation_summary.json"
)

OUT = (
    ROOT
    / "research"
    / "exports"
    / "quantile_v2_model"
)


# ============================================================
# Locked experiment design
# ============================================================

HORIZONS = (4, 8, 16)

QUANTILES = (0.10, 0.50, 0.90)

MIN_TRAIN_ROWS = 1000

TEST_ROWS = 450

STEP_ROWS = 450

RANDOM_STATE = 42


# ============================================================
# Nested feature sets
# ============================================================

F1_DIRECTION = [
    "feat_15m_ret_1",
    "feat_15m_ret_4",
    "feat_15m_ret_8",
    "feat_15m_ret_16",
    "feat_15m_location_32",
    "feat_time_bars_since_segment_start",
    "feat_time_after_long_gap",
]

F2_ACTIVITY = F1_DIRECTION + [
    "feat_15m_volume_ratio_32",
    "feat_15m_oi_log_change_4",
]

F3_VOL = F2_ACTIVITY + [
    "feat_5m_1h_rv",
    "feat_5m_rv_rate_ratio_1h_4h",
]

F4_TAIL = F3_VOL + [
    "feat_5m_4h_neg_semivar_share",
    "feat_5m_1h_jump_share",
]

FEATURE_SETS = {
    "F1_DIRECTION": F1_DIRECTION,
    "F2_ACTIVITY": F2_ACTIVITY,
    "F3_VOL": F3_VOL,
    "F4_TAIL": F4_TAIL,
}


FORBIDDEN_FEATURE_TOKENS = (
    "smc",
    "momentum",
    "sqz",
    "oracle",
    "target",
)


# ============================================================
# Helpers
# ============================================================

def prepare_output_dir() -> None:

    if OUT.exists():

        existing = list(
            OUT.iterdir()
        )

        if existing:

            raise RuntimeError(
                f"{OUT} already exists and "
                "is non-empty. "
                "Delete only for an intentional "
                "pre-commit rerun."
            )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )


def safe_spearman(
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

    if int(
        valid.sum()
    ) < 3:

        return float("nan")

    a = (
        a[valid]
        .rank(
            method="average"
        )
    )

    b = (
        b[valid]
        .rank(
            method="average"
        )
    )

    if (
        a.nunique() < 2
        or b.nunique() < 2
    ):

        return float("nan")

    return float(
        np.corrcoef(
            a.to_numpy(float),
            b.to_numpy(float),
        )[0, 1]
    )


def interval_score(
    y,
    lower,
    upper,
    alpha=0.20,
) -> np.ndarray:

    y = np.asarray(
        y,
        dtype=float,
    )

    lower = np.asarray(
        lower,
        dtype=float,
    )

    upper = np.asarray(
        upper,
        dtype=float,
    )

    width = (
        upper
        - lower
    )

    penalty_low = (
        (2.0 / alpha)
        * (lower - y)
        * (y < lower)
    )

    penalty_high = (
        (2.0 / alpha)
        * (y - upper)
        * (y > upper)
    )

    return (
        width
        + penalty_low
        + penalty_high
    )


# ============================================================
# Purged expanding OOS folds
# ============================================================

def make_folds(
    n_rows: int,
    *,
    horizon: int,
) -> list[dict]:

    folds = []

    train_end = (
        MIN_TRAIN_ROWS
    )

    fold_id = 0

    while True:

        test_start = (
            train_end
            + horizon
        )

        test_end = min(
            test_start
            + TEST_ROWS,
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
        ) < 250:
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
            STEP_ROWS
        )

    if len(
        folds
    ) != 5:

        raise RuntimeError(
            f"h={horizon}: "
            f"expected 5 folds, "
            f"got {len(folds)}"
        )

    return folds


# ============================================================
# Models
# ============================================================

def make_model(
    model_name: str,
    q: float,
):

    if model_name == "linear_qr":

        # Pure linear quantile regression.
        #
        # No hyperparameter search.
        #
        # alpha=0 is deliberate:
        # this is our statistical benchmark,
        # not a tuned sparse model.
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),

                (
                    "scaler",
                    StandardScaler(),
                ),

                (
                    "model",
                    QuantileRegressor(
                        quantile=q,
                        alpha=0.0,
                        solver="highs",
                    ),
                ),
            ]
        )

    if model_name == "gbr_quantile":

        # Fixed shallow nonlinear challenger.
        #
        # No GridSearch.
        # No Optuna.
        # No tuning on current result.
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),

                (
                    "model",
                    GradientBoostingRegressor(
                        loss="quantile",
                        alpha=q,
                        n_estimators=150,
                        learning_rate=0.03,
                        max_depth=2,
                        min_samples_leaf=30,
                        subsample=0.80,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    raise ValueError(
        f"Unknown model: "
        f"{model_name}"
    )


MODELS = (
    "linear_qr",
    "gbr_quantile",
)


# ============================================================
# Model-effect extraction
# ============================================================

def extract_feature_effects(
    fitted,
    *,
    feature_names,
    model_name,
    horizon,
    feature_set,
    quantile,
    fold,
) -> pd.DataFrame:

    estimator = (
        fitted.named_steps[
            "model"
        ]
    )

    if (
        model_name
        == "linear_qr"
    ):

        values = np.asarray(
            estimator.coef_,
            dtype=float,
        )

        effect_kind = (
            "standardized_coefficient"
        )

    elif (
        model_name
        == "gbr_quantile"
    ):

        values = np.asarray(
            estimator.feature_importances_,
            dtype=float,
        )

        effect_kind = (
            "tree_importance"
        )

    else:

        return pd.DataFrame()

    if len(values) != len(
        feature_names
    ):

        raise RuntimeError(
            "Feature-effect length "
            "mismatch"
        )

    return pd.DataFrame(
        {
            "horizon": horizon,

            "feature_set": (
                feature_set
            ),

            "model": model_name,

            "quantile": quantile,

            "fold": fold,

            "feature": (
                feature_names
            ),

            "effect_kind": (
                effect_kind
            ),

            "value": values,

            "abs_value": np.abs(
                values
            ),
        }
    )


# ============================================================
# Decile diagnostics
# ============================================================

def q50_decile_table(
    frame: pd.DataFrame,
    *,
    horizon: int,
    feature_set: str,
    model: str,
) -> pd.DataFrame:

    x = frame.dropna(
        subset=[
            "pred_q50",
            "realized_return",
        ]
    ).copy()

    if (
        len(x) < 500
        or x[
            "pred_q50"
        ].nunique() < 10
    ):

        return pd.DataFrame()

    rank = (
        x["pred_q50"]
        .rank(
            method="first"
        )
    )

    x[
        "decile"
    ] = (
        pd.qcut(
            rank,
            q=10,
            labels=False,
        )
        + 1
    )

    rows = []

    for decile, g in (
        x.groupby(
            "decile",
            observed=True,
        )
    ):

        rows.append(
            {
                "horizon": (
                    horizon
                ),

                "feature_set": (
                    feature_set
                ),

                "model": model,

                "decile": int(
                    decile
                ),

                "n": int(
                    len(g)
                ),

                "mean_prediction": float(
                    g[
                        "pred_q50"
                    ].mean()
                ),

                "realized_mean": float(
                    g[
                        "realized_return"
                    ].mean()
                ),

                "realized_median": float(
                    g[
                        "realized_return"
                    ].median()
                ),

                "positive_rate": float(
                    (
                        g[
                            "realized_return"
                        ]
                        > 0
                    ).mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def width_decile_table(
    frame: pd.DataFrame,
    *,
    horizon: int,
    feature_set: str,
    model: str,
) -> pd.DataFrame:

    x = frame.copy()

    x[
        "pred_width"
    ] = (
        x[
            "pred_q90"
        ]
        -
        x[
            "pred_q10"
        ]
    )

    # Do not hide quantile crossing.
    # Width discrimination is only measured
    # where lower <= upper.
    x = x[
        x[
            "pred_width"
        ]
        >= 0
    ].copy()

    x = x.dropna(
        subset=[
            "pred_width",
            "realized_return",
            "realized_path_range",
        ]
    )

    if (
        len(x) < 500
        or x[
            "pred_width"
        ].nunique() < 10
    ):

        return pd.DataFrame()

    rank = (
        x[
            "pred_width"
        ]
        .rank(
            method="first"
        )
    )

    x[
        "decile"
    ] = (
        pd.qcut(
            rank,
            q=10,
            labels=False,
        )
        + 1
    )

    rows = []

    for decile, g in (
        x.groupby(
            "decile",
            observed=True,
        )
    ):

        rows.append(
            {
                "horizon": horizon,

                "feature_set": (
                    feature_set
                ),

                "model": model,

                "decile": int(
                    decile
                ),

                "n": int(
                    len(g)
                ),

                "mean_pred_width": float(
                    g[
                        "pred_width"
                    ].mean()
                ),

                "median_abs_return": float(
                    g[
                        "realized_return"
                    ]
                    .abs()
                    .median()
                ),

                "mean_abs_return": float(
                    g[
                        "realized_return"
                    ]
                    .abs()
                    .mean()
                ),

                "median_path_range": float(
                    g[
                        "realized_path_range"
                    ].median()
                ),

                "mean_path_range": float(
                    g[
                        "realized_path_range"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Aggregate feature effects
# ============================================================

def summarize_feature_effects(
    effects: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    group_cols = [
        "horizon",
        "feature_set",
        "model",
        "quantile",
        "feature",
        "effect_kind",
    ]

    for keys, g in (
        effects.groupby(
            group_cols,
            observed=True,
        )
    ):

        row = dict(
            zip(
                group_cols,
                keys,
            )
        )

        values = pd.to_numeric(
            g[
                "value"
            ],
            errors="coerce",
        ).dropna()

        row[
            "folds"
        ] = int(
            len(values)
        )

        row[
            "mean_value"
        ] = float(
            values.mean()
        )

        row[
            "median_value"
        ] = float(
            values.median()
        )

        row[
            "mean_abs_value"
        ] = float(
            values.abs().mean()
        )

        if (
            row[
                "effect_kind"
            ]
            ==
            "standardized_coefficient"
        ):

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

            row[
                "positive_share"
            ] = positive_share

            row[
                "sign_consistency"
            ] = max(
                positive_share,
                negative_share,
            )

        else:

            row[
                "positive_share"
            ] = np.nan

            row[
                "sign_consistency"
            ] = np.nan

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Main experiment
# ============================================================

def main() -> None:

    prepare_output_dir()

    if not INPUT.is_file():

        raise RuntimeError(
            f"Missing panel: "
            f"{INPUT}"
        )

    panel = pd.read_csv(
        INPUT,
        parse_dates=[
            "meta_decision_time",
        ],
        low_memory=False,
    )

    if len(
        panel
    ) != 3300:

        raise RuntimeError(
            f"Expected 3300 rows, "
            f"got {len(panel)}"
        )

    # --------------------------------------------------------
    # Locked feature validation
    # --------------------------------------------------------

    all_features = sorted(
        set(
            col
            for features
            in FEATURE_SETS.values()
            for col in features
        )
    )

    missing = [
        col
        for col in all_features
        if col not in panel.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing features: "
            f"{missing}"
        )

    bad = [
        col
        for col in all_features
        if any(
            token in col.lower()
            for token
            in FORBIDDEN_FEATURE_TOKENS
        )
    ]

    if bad:

        raise RuntimeError(
            f"Forbidden features: "
            f"{bad}"
        )

    if (
        "feat_5m_1h_ret_sum"
        in all_features
        or
        "feat_5m_4h_ret_sum"
        in all_features
    ):

        raise RuntimeError(
            "Duplicate return representation "
            "must not enter model"
        )

    # --------------------------------------------------------
    # Containers
    # --------------------------------------------------------

    fold_quantile_rows = []

    fold_distribution_rows = []

    all_predictions = []

    all_effects = []

    q50_deciles_all = []

    width_deciles_all = []

    folds_manifest = {}

    # ========================================================
    # Horizon
    # ========================================================

    for h in HORIZONS:

        target_col = (
            f"target_raw_return_h{h}"
        )

        long_mfe_col = (
            f"target_long_mfe_h{h}"
        )

        short_mfe_col = (
            f"target_short_mfe_h{h}"
        )

        valid = (
            panel[
                target_col
            ].notna()
            &
            panel[
                long_mfe_col
            ].notna()
            &
            panel[
                short_mfe_col
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

        folds = make_folds(
            len(frame),
            horizon=h,
        )

        folds_manifest[
            str(h)
        ] = folds

        # ====================================================
        # Nested feature set
        # ====================================================

        for (
            feature_set_name,
            feature_cols,
        ) in (
            FEATURE_SETS.items()
        ):

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

            # Every chosen feature must have
            # actual information.
            for col in feature_cols:

                s = X_all[
                    col
                ]

                if (
                    s.notna().sum()
                    < 1000
                ):

                    raise RuntimeError(
                        f"{col}: "
                        "too few non-NA"
                    )

                if (
                    s.nunique(
                        dropna=True
                    )
                    < 2
                ):

                    raise RuntimeError(
                        f"{col}: "
                        "constant feature"
                    )

            # =================================================
            # Model
            # =================================================

            for model_name in MODELS:

                # One prediction array per quantile.
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

                # =============================================
                # Fold
                # =============================================

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
                    ) < h:

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

                    if (
                        y_train.isna().any()
                        or
                        y_test.isna().any()
                    ):

                        raise RuntimeError(
                            "Unexpected target NaN "
                            "inside valid frame"
                        )

                    fold_prediction = {}

                    fold_baseline = {}

                    # -----------------------------------------
                    # Quantile fits
                    # -----------------------------------------

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

                        fold_prediction[
                            q
                        ] = p

                        fold_baseline[
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
                                fold_baseline[
                                    q
                                ],
                                alpha=q,
                            )
                        )

                        skill = (
                            1.0
                            -
                            model_loss
                            / base_loss
                            if base_loss > 0
                            else np.nan
                        )

                        calibration = float(
                            (
                                y_test.to_numpy()
                                <= p
                            ).mean()
                        )

                        base_calibration = float(
                            (
                                y_test.to_numpy()
                                <= fold_baseline[
                                    q
                                ]
                            ).mean()
                        )

                        fold_quantile_rows.append(
                            {
                                "horizon": h,

                                "approx_minutes": (
                                    h * 15
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
                                    len(
                                        train_idx
                                    )
                                ),

                                "test_rows": int(
                                    len(
                                        test_idx
                                    )
                                ),

                                "purge_rows": h,

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

                                "baseline_calibration": (
                                    base_calibration
                                ),
                            }
                        )

                        effects = (
                            extract_feature_effects(
                                model,
                                feature_names=(
                                    feature_cols
                                ),
                                model_name=(
                                    model_name
                                ),
                                horizon=h,
                                feature_set=(
                                    feature_set_name
                                ),
                                quantile=q,
                                fold=(
                                    fold[
                                        "fold"
                                    ]
                                ),
                            )
                        )

                        all_effects.append(
                            effects
                        )

                    # -----------------------------------------
                    # Fold distribution diagnostics
                    # -----------------------------------------

                    q10 = (
                        fold_prediction[
                            0.10
                        ]
                    )

                    q50 = (
                        fold_prediction[
                            0.50
                        ]
                    )

                    q90 = (
                        fold_prediction[
                            0.90
                        ]
                    )

                    b10 = (
                        fold_baseline[
                            0.10
                        ]
                    )

                    b50 = (
                        fold_baseline[
                            0.50
                        ]
                    )

                    b90 = (
                        fold_baseline[
                            0.90
                        ]
                    )

                    y_np = (
                        y_test.to_numpy(
                            dtype=float
                        )
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
                                    y_np[
                                        ordered
                                    ]
                                    >=
                                    q10[
                                        ordered
                                    ]
                                )
                                &
                                (
                                    y_np[
                                        ordered
                                    ]
                                    <=
                                    q90[
                                        ordered
                                    ]
                                )
                            ).mean()
                        )

                        model_interval_score = float(
                            np.mean(
                                interval_score(
                                    y_np[
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

                        baseline_interval_score = float(
                            np.mean(
                                interval_score(
                                    y_np[
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

                        interval_skill = (
                            1.0
                            -
                            model_interval_score
                            /
                            baseline_interval_score
                            if (
                                baseline_interval_score
                                > 0
                            )
                            else np.nan
                        )

                    else:

                        coverage = np.nan

                        model_interval_score = (
                            np.nan
                        )

                        baseline_interval_score = (
                            np.nan
                        )

                        interval_skill = (
                            np.nan
                        )

                    baseline_coverage = float(
                        (
                            (y_np >= b10)
                            &
                            (y_np <= b90)
                        ).mean()
                    )

                    fold_distribution_rows.append(
                        {
                            "horizon": h,

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

                            "crossing_rate": float(
                                crossing.mean()
                            ),

                            "ordered_share": float(
                                ordered.mean()
                            ),

                            "interval_80_coverage_ordered": (
                                coverage
                            ),

                            "baseline_interval_80_coverage": (
                                baseline_coverage
                            ),

                            "interval_score": (
                                model_interval_score
                            ),

                            "baseline_interval_score": (
                                baseline_interval_score
                            ),

                            "interval_score_skill": (
                                interval_skill
                            ),

                            "q50_sign_accuracy": float(
                                (
                                    (q50 > 0)
                                    ==
                                    (y_np > 0)
                                ).mean()
                            ),

                            "q50_spearman": (
                                safe_spearman(
                                    q50,
                                    y_np,
                                )
                            ),
                        }
                    )

                    fold_ids[
                        test_idx
                    ] = (
                        fold[
                            "fold"
                        ]
                    )

                # =============================================
                # Pooled OOS
                # =============================================

                oos = (
                    fold_ids >= 0
                )

                if int(
                    oos.sum()
                ) != 2250:

                    raise RuntimeError(
                        f"h={h} "
                        f"{feature_set_name} "
                        f"{model_name}: "
                        f"OOS rows "
                        f"{oos.sum()} "
                        f"!=2250"
                    )

                pred_frame = pd.DataFrame(
                    {
                        "decision_time": (
                            frame.loc[
                                oos,
                                "meta_decision_time",
                            ].to_numpy()
                        ),

                        "horizon": h,

                        "feature_set": (
                            feature_set_name
                        ),

                        "model": model_name,

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

                all_predictions.append(
                    pred_frame
                )

                q50_deciles = (
                    q50_decile_table(
                        pred_frame,
                        horizon=h,
                        feature_set=(
                            feature_set_name
                        ),
                        model=model_name,
                    )
                )

                if not q50_deciles.empty:

                    q50_deciles_all.append(
                        q50_deciles
                    )

                width_deciles = (
                    width_decile_table(
                        pred_frame,
                        horizon=h,
                        feature_set=(
                            feature_set_name
                        ),
                        model=model_name,
                    )
                )

                if not width_deciles.empty:

                    width_deciles_all.append(
                        width_deciles
                    )

    # ========================================================
    # Assemble raw outputs
    # ========================================================

    fold_quantile = pd.DataFrame(
        fold_quantile_rows
    )

    fold_distribution = pd.DataFrame(
        fold_distribution_rows
    )

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    effects = pd.concat(
        all_effects,
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

    # ========================================================
    # Aggregate quantile metrics
    # ========================================================

    quantile_rows = []

    for (
        h,
        feature_set,
        model,
        q,
    ), g in (
        fold_quantile.groupby(
            [
                "horizon",
                "feature_set",
                "model",
                "quantile",
            ],
            observed=True,
        )
    ):

        pred_subset = predictions[
            (
                predictions[
                    "horizon"
                ] == h
            )
            &
            (
                predictions[
                    "feature_set"
                ] == feature_set
            )
            &
            (
                predictions[
                    "model"
                ] == model
            )
        ]

        pred_col = {
            0.10: "pred_q10",
            0.50: "pred_q50",
            0.90: "pred_q90",
        }[
            float(q)
        ]

        base_col = {
            0.10: "baseline_q10",
            0.50: "baseline_q50",
            0.90: "baseline_q90",
        }[
            float(q)
        ]

        y = pred_subset[
            "realized_return"
        ]

        p = pred_subset[
            pred_col
        ]

        b = pred_subset[
            base_col
        ]

        model_loss = float(
            mean_pinball_loss(
                y,
                p,
                alpha=q,
            )
        )

        baseline_loss = float(
            mean_pinball_loss(
                y,
                b,
                alpha=q,
            )
        )

        pooled_skill = (
            1.0
            -
            model_loss
            /
            baseline_loss
            if baseline_loss > 0
            else np.nan
        )

        calibration = float(
            (
                y.to_numpy()
                <=
                p.to_numpy()
            ).mean()
        )

        quantile_rows.append(
            {
                "horizon": int(
                    h
                ),

                "approx_minutes": (
                    int(h)
                    * 15
                ),

                "feature_set": (
                    feature_set
                ),

                "model": model,

                "quantile": float(
                    q
                ),

                "oos_rows": int(
                    len(y)
                ),

                "pinball_loss": (
                    model_loss
                ),

                "baseline_pinball_loss": (
                    baseline_loss
                ),

                "pinball_skill": (
                    pooled_skill
                ),

                "calibration": (
                    calibration
                ),

                "calibration_error": (
                    calibration
                    - q
                ),

                "positive_skill_fold_share": float(
                    (
                        g[
                            "pinball_skill"
                        ]
                        > 0
                    ).mean()
                ),

                "median_fold_skill": float(
                    g[
                        "pinball_skill"
                    ].median()
                ),
            }
        )

    quantile_metrics = pd.DataFrame(
        quantile_rows
    )

    # ========================================================
    # Aggregate distribution metrics
    # ========================================================

    distribution_rows = []

    for (
        h,
        feature_set,
        model,
    ), g in (
        predictions.groupby(
            [
                "horizon",
                "feature_set",
                "model",
            ],
            observed=True,
        )
    ):

        q10 = (
            g[
                "pred_q10"
            ].to_numpy(float)
        )

        q50 = (
            g[
                "pred_q50"
            ].to_numpy(float)
        )

        q90 = (
            g[
                "pred_q90"
            ].to_numpy(float)
        )

        b10 = (
            g[
                "baseline_q10"
            ].to_numpy(float)
        )

        b90 = (
            g[
                "baseline_q90"
            ].to_numpy(float)
        )

        y = (
            g[
                "realized_return"
            ].to_numpy(float)
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

            interval_model = float(
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
                    )
                )
            )

            interval_base = float(
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
                    )
                )
            )

            interval_skill = (
                1.0
                -
                interval_model
                /
                interval_base
                if interval_base > 0
                else np.nan
            )

        else:

            coverage = np.nan

            interval_model = np.nan

            interval_base = np.nan

            interval_skill = np.nan

        qdec = q50_deciles[
            (
                q50_deciles[
                    "horizon"
                ] == h
            )
            &
            (
                q50_deciles[
                    "feature_set"
                ] == feature_set
            )
            &
            (
                q50_deciles[
                    "model"
                ] == model
            )
        ].sort_values(
            "decile"
        )

        if len(
            qdec
        ) == 10:

            q50_mono = (
                safe_spearman(
                    qdec[
                        "decile"
                    ],
                    qdec[
                        "realized_median"
                    ],
                )
            )

            q50_top_bottom = float(
                qdec.iloc[
                    -1
                ][
                    "realized_median"
                ]
                -
                qdec.iloc[
                    0
                ][
                    "realized_median"
                ]
            )

        else:

            q50_mono = np.nan

            q50_top_bottom = np.nan

        wdec = width_deciles[
            (
                width_deciles[
                    "horizon"
                ] == h
            )
            &
            (
                width_deciles[
                    "feature_set"
                ] == feature_set
            )
            &
            (
                width_deciles[
                    "model"
                ] == model
            )
        ].sort_values(
            "decile"
        )

        if len(
            wdec
        ) == 10:

            width_abs_mono = (
                safe_spearman(
                    wdec[
                        "decile"
                    ],
                    wdec[
                        "median_abs_return"
                    ],
                )
            )

            width_path_mono = (
                safe_spearman(
                    wdec[
                        "decile"
                    ],
                    wdec[
                        "median_path_range"
                    ],
                )
            )

            path_top_bottom = float(
                wdec.iloc[
                    -1
                ][
                    "median_path_range"
                ]
                -
                wdec.iloc[
                    0
                ][
                    "median_path_range"
                ]
            )

        else:

            width_abs_mono = np.nan

            width_path_mono = np.nan

            path_top_bottom = np.nan

        fold_g = (
            fold_distribution[
                (
                    fold_distribution[
                        "horizon"
                    ] == h
                )
                &
                (
                    fold_distribution[
                        "feature_set"
                    ] == feature_set
                )
                &
                (
                    fold_distribution[
                        "model"
                    ] == model
                )
            ]
        )

        distribution_rows.append(
            {
                "horizon": int(
                    h
                ),

                "approx_minutes": (
                    int(h)
                    * 15
                ),

                "feature_set": (
                    feature_set
                ),

                "model": model,

                "oos_rows": int(
                    len(g)
                ),

                "crossing_rate": float(
                    crossing.mean()
                ),

                "ordered_share": float(
                    ordered.mean()
                ),

                "interval_80_coverage_ordered": (
                    coverage
                ),

                "interval_coverage_error": (
                    coverage
                    - 0.80
                    if np.isfinite(
                        coverage
                    )
                    else np.nan
                ),

                "interval_score": (
                    interval_model
                ),

                "baseline_interval_score": (
                    interval_base
                ),

                "interval_score_skill": (
                    interval_skill
                ),

                "positive_interval_skill_fold_share": float(
                    (
                        fold_g[
                            "interval_score_skill"
                        ]
                        > 0
                    ).mean()
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

                "q50_decile_monotonicity": (
                    q50_mono
                ),

                "q50_top_minus_bottom_realized_median": (
                    q50_top_bottom
                ),

                "width_abs_return_monotonicity": (
                    width_abs_mono
                ),

                "width_path_range_monotonicity": (
                    width_path_mono
                ),

                "width_top_minus_bottom_path_range": (
                    path_top_bottom
                ),
            }
        )

    distribution_metrics = pd.DataFrame(
        distribution_rows
    )

    # ========================================================
    # Model summary
    # ========================================================

    quantile_pivot = (
        quantile_metrics
        .pivot_table(
            index=[
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

    model_summary = (
        distribution_metrics.merge(
            quantile_pivot,
            on=[
                "horizon",
                "feature_set",
                "model",
            ],
            how="left",
        )
    )

    model_summary[
        "mean_pinball_skill"
    ] = (
        model_summary[
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

    model_summary[
        "tail_pinball_skill"
    ] = (
        model_summary[
            [
                "q10_pinball_skill",
                "q90_pinball_skill",
            ]
        ]
        .mean(
            axis=1
        )
    )

    # ========================================================
    # Nested feature-set deltas
    # ========================================================

    ordered_sets = [
        "F1_DIRECTION",
        "F2_ACTIVITY",
        "F3_VOL",
        "F4_TAIL",
    ]

    delta_rows = []

    metric_cols = [
        "mean_pinball_skill",
        "tail_pinball_skill",
        "q10_pinball_skill",
        "q50_pinball_skill",
        "q90_pinball_skill",
        "interval_score_skill",
        "q50_decile_monotonicity",
        "width_path_range_monotonicity",
    ]

    for h in HORIZONS:

        for model in MODELS:

            subset = (
                model_summary[
                    (
                        model_summary[
                            "horizon"
                        ] == h
                    )
                    &
                    (
                        model_summary[
                            "model"
                        ] == model
                    )
                ]
                .set_index(
                    "feature_set"
                )
            )

            for i in range(
                1,
                len(
                    ordered_sets
                )
            ):

                previous = (
                    ordered_sets[
                        i - 1
                    ]
                )

                current = (
                    ordered_sets[
                        i
                    ]
                )

                if (
                    previous
                    not in subset.index
                    or current
                    not in subset.index
                ):
                    continue

                row = {
                    "horizon": h,

                    "model": model,

                    "from_feature_set": (
                        previous
                    ),

                    "to_feature_set": (
                        current
                    ),
                }

                for metric in metric_cols:

                    row[
                        f"delta_{metric}"
                    ] = float(
                        subset.loc[
                            current,
                            metric,
                        ]
                        -
                        subset.loc[
                            previous,
                            metric,
                        ]
                    )

                delta_rows.append(
                    row
                )

    feature_set_deltas = pd.DataFrame(
        delta_rows
    )

    # ========================================================
    # Effect summary
    # ========================================================

    effect_summary = (
        summarize_feature_effects(
            effects
        )
    )

    # ========================================================
    # Validation
    # ========================================================

    if len(
        quantile_metrics
    ) != 72:

        raise RuntimeError(
            "quantile_metrics rows "
            f"{len(quantile_metrics)} "
            "!=72"
        )

    if len(
        model_summary
    ) != 24:

        raise RuntimeError(
            "model_summary rows "
            f"{len(model_summary)} "
            "!=24"
        )

    if len(
        fold_quantile
    ) != 360:

        raise RuntimeError(
            "fold_quantile rows "
            f"{len(fold_quantile)} "
            "!=360"
        )

    if len(
        fold_distribution
    ) != 120:

        raise RuntimeError(
            "fold_distribution rows "
            f"{len(fold_distribution)} "
            "!=120"
        )

    if len(
        predictions
    ) != (
        3
        * 4
        * 2
        * 2250
    ):

        raise RuntimeError(
            "Unexpected OOS prediction "
            f"rows={len(predictions)}"
        )

    duplicate_key = [
        "decision_time",
        "horizon",
        "feature_set",
        "model",
    ]

    if predictions.duplicated(
        duplicate_key
    ).any():

        raise RuntimeError(
            "Duplicate OOS prediction"
        )

    numeric_outputs = [
        fold_quantile,
        fold_distribution,
        quantile_metrics,
        distribution_metrics,
        model_summary,
        feature_set_deltas,
        effects,
        effect_summary,
        q50_deciles,
        width_deciles,
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
    # Persist
    # ========================================================

    fold_quantile.to_csv(
        OUT
        / "fold_quantile_metrics.csv",
        index=False,
    )

    fold_distribution.to_csv(
        OUT
        / "fold_distribution_metrics.csv",
        index=False,
    )

    quantile_metrics.to_csv(
        OUT
        / "quantile_metrics.csv",
        index=False,
    )

    distribution_metrics.to_csv(
        OUT
        / "distribution_metrics.csv",
        index=False,
    )

    model_summary.to_csv(
        OUT
        / "model_summary.csv",
        index=False,
    )

    feature_set_deltas.to_csv(
        OUT
        / "feature_set_deltas.csv",
        index=False,
    )

    predictions.to_csv(
        OUT
        / "oos_predictions.csv",
        index=False,
    )

    effects.to_csv(
        OUT
        / "feature_effects_by_fold.csv",
        index=False,
    )

    effect_summary.to_csv(
        OUT
        / "feature_effect_summary.csv",
        index=False,
    )

    q50_deciles.to_csv(
        OUT
        / "q50_deciles.csv",
        index=False,
    )

    width_deciles.to_csv(
        OUT
        / "width_deciles.csv",
        index=False,
    )

    # ========================================================
    # Config + validation
    # ========================================================

    config = {
        "purpose": (
            "Classical short-horizon "
            "conditional quantile forecasting."
        ),

        "source_panel": str(
            INPUT.relative_to(
                ROOT
            )
        ),

        "source_rows": int(
            len(panel)
        ),

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

        "validation": {
            "type": (
                "purged expanding OOS"
            ),

            "min_train_rows": (
                MIN_TRAIN_ROWS
            ),

            "test_rows": (
                TEST_ROWS
            ),

            "step_rows": (
                STEP_ROWS
            ),

            "purge": (
                "horizon rows"
            ),
        },

        "benchmark": (
            "empirical training-sample "
            "quantile per fold"
        ),

        "no_hyperparameter_search": True,

        "no_backtest": True,

        "no_smc": True,

        "no_momentum": True,

        "no_oracle_feature": True,

        "raw_return_is_primary_target": True,
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

        if path.is_file():

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
                    f"{path.name} "
                    ">50MB"
                )

    validation = {
        "status": "PASS",

        "source_rows": int(
            len(panel)
        ),

        "feature_set_count": int(
            len(
                FEATURE_SETS
            )
        ),

        "max_feature_count": int(
            max(
                len(x)
                for x
                in FEATURE_SETS.values()
            )
        ),

        "models": list(
            MODELS
        ),

        "quantiles": list(
            QUANTILES
        ),

        "fold_count_by_horizon": {
            str(h): len(
                folds_manifest[
                    str(h)
                ]
            )
            for h
            in HORIZONS
        },

        "quantile_metric_rows": int(
            len(
                quantile_metrics
            )
        ),

        "model_summary_rows": int(
            len(
                model_summary
            )
        ),

        "oos_prediction_rows": int(
            len(
                predictions
            )
        ),

        "no_random_split": True,

        "purged_oos": True,

        "no_hyperparameter_search": True,

        "no_backtest": True,

        "no_smc": True,

        "no_momentum": True,

        "no_oracle_feature": True,

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
    # Compact factual summary
    # ========================================================

    lines = [
        "# Quantile V2 Model Research",
        "",
        "No trading strategy.",
        "No backtest.",
        "No hyperparameter search.",
        "",
        (
            "| H | model | features | "
            "Q10 skill | Q50 skill | "
            "Q90 skill | coverage | "
            "crossing | Q50 mono | "
            "width/path mono |"
        ),
        (
            "|---:|---|---|---:|---:|---:|"
            "---:|---:|---:|---:|"
        ),
    ]

    for _, row in (
        model_summary.sort_values(
            [
                "horizon",
                "mean_pinball_skill",
            ],
            ascending=[
                True,
                False,
            ],
        ).iterrows()
    ):

        lines.append(
            f"| {int(row['horizon'])} "
            f"| {row['model']} "
            f"| {row['feature_set']} "
            f"| {row['q10_pinball_skill']:.4f} "
            f"| {row['q50_pinball_skill']:.4f} "
            f"| {row['q90_pinball_skill']:.4f} "
            f"| {row['interval_80_coverage_ordered']:.4f} "
            f"| {row['crossing_rate']:.4f} "
            f"| {row['q50_decile_monotonicity']:.4f} "
            f"| {row['width_path_range_monotonicity']:.4f} |"
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
        "QUANTILE_V2_MODEL_PASS"
    )


if __name__ == "__main__":
    main()
