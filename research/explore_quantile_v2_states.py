#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "research"
    / "exports"
    / "quantile_v2_data"
    / "quantile_panel_15m.csv"
)

SOURCE_MANIFEST = (
    ROOT
    / "research"
    / "exports"
    / "quantile_v2_data"
    / "manifest.json"
)

OUT = (
    ROOT
    / "research"
    / "exports"
    / "quantile_v2_explore"
)


# ============================================================
# Research design
# ============================================================

HORIZONS = (
    4,   # 4 observed 15m bars
    8,   # 8 observed 15m bars
    16,  # 16 observed 15m bars
)

MIN_TRAIN_ROWS = 1000

TEST_ROWS = 450
STEP_ROWS = 450

MARGINAL_BINS = 5
PAIR_BINS = 3


# ============================================================
# PRE-SPECIFIED feature set
#
# Do NOT add features after seeing results.
# ============================================================

CORE_FEATURES = [

    # --------------------------------------------------------
    # 15m direction memory
    # --------------------------------------------------------

    "feat_15m_ret_1",
    "feat_15m_ret_4",
    "feat_15m_ret_8",
    "feat_15m_ret_16",

    # --------------------------------------------------------
    # Price location / activity
    # --------------------------------------------------------

    "feat_15m_location_32",
    "feat_15m_volume_ratio_32",
    "feat_15m_oi_log_change_4",

    # --------------------------------------------------------
    # Generic session state
    # --------------------------------------------------------

    "feat_time_after_long_gap",
    "feat_time_bars_since_segment_start",

    # --------------------------------------------------------
    # 5m aggregated direction
    # --------------------------------------------------------

    "feat_5m_1h_ret_sum",
    "feat_5m_4h_ret_sum",

    # --------------------------------------------------------
    # Multi-scale realized variance
    # --------------------------------------------------------

    "feat_5m_1h_rv",
    "feat_5m_4h_rv",
    "feat_5m_8h_rv",

    "feat_5m_rv_rate_ratio_1h_4h",
    "feat_5m_rv_rate_ratio_4h_8h",

    # --------------------------------------------------------
    # Good / Bad volatility
    # --------------------------------------------------------

    "feat_5m_1h_neg_semivar_share",
    "feat_5m_4h_neg_semivar_share",

    # --------------------------------------------------------
    # Jump / gap
    # --------------------------------------------------------

    "feat_5m_1h_jump_share",
    "feat_5m_4h_jump_share",
    "feat_5m_4h_gap_sq",

    # --------------------------------------------------------
    # Intrawindow shape
    # --------------------------------------------------------

    "feat_5m_1h_positive_return_share",
    "feat_5m_1h_max_abs_return",
]


BINARY_FEATURES = {
    "feat_time_after_long_gap",
}


# ============================================================
# PRE-SPECIFIED pair hypotheses
#
# These are not selected from results.
# ============================================================

PAIR_SPECS = [

    (
        "momentum_1h_x_badvol_1h",
        "feat_5m_1h_ret_sum",
        "feat_5m_1h_neg_semivar_share",
    ),

    (
        "momentum_1h_x_volaccel",
        "feat_5m_1h_ret_sum",
        "feat_5m_rv_rate_ratio_1h_4h",
    ),

    (
        "jump_1h_x_location_32",
        "feat_5m_1h_jump_share",
        "feat_15m_location_32",
    ),

    (
        "badvol_1h_x_rv_1h",
        "feat_5m_1h_neg_semivar_share",
        "feat_5m_1h_rv",
    ),

    (
        "oi_change_x_volume",
        "feat_15m_oi_log_change_4",
        "feat_15m_volume_ratio_32",
    ),
]


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
                "rerun before commit."
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
        & b.notna()
    )

    if int(
        valid.sum()
    ) < 3:

        return float("nan")

    a = a[
        valid
    ].rank(
        method="average"
    )

    b = b[
        valid
    ].rank(
        method="average"
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


def target_stats(
    raw: pd.Series,
    norm: pd.Series,
) -> dict:

    x = pd.to_numeric(
        raw,
        errors="coerce",
    )

    z = pd.to_numeric(
        norm,
        errors="coerce",
    )

    valid = (
        x.notna()
        & z.notna()
    )

    x = x[
        valid
    ]

    z = z[
        valid
    ]

    if len(x) == 0:

        return {
            "n": 0,

            "q10": np.nan,
            "q50": np.nan,
            "q90": np.nan,

            "width_90_10": np.nan,
            "asym_90_plus_10": np.nan,

            "mean": np.nan,

            "positive_rate": np.nan,

            "norm_q10": np.nan,
            "norm_q50": np.nan,
            "norm_q90": np.nan,
        }

    q10 = float(
        x.quantile(
            0.10
        )
    )

    q50 = float(
        x.quantile(
            0.50
        )
    )

    q90 = float(
        x.quantile(
            0.90
        )
    )

    return {
        "n": int(
            len(x)
        ),

        "q10": q10,
        "q50": q50,
        "q90": q90,

        # Future distribution width.
        "width_90_10": (
            q90 - q10
        ),

        # Positive => upper tail dominates.
        # Negative => lower tail dominates.
        "asym_90_plus_10": (
            q90 + q10
        ),

        "mean": float(
            x.mean()
        ),

        "positive_rate": float(
            (
                x > 0
            ).mean()
        ),

        "norm_q10": float(
            z.quantile(
                0.10
            )
        ),

        "norm_q50": float(
            z.quantile(
                0.50
            )
        ),

        "norm_q90": float(
            z.quantile(
                0.90
            )
        ),
    }


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

        # Train labels use future H bars.
        # Therefore keep H rows between
        # train and OOS test.
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
    ) < 5:

        raise RuntimeError(
            f"h={horizon}: "
            f"only {len(folds)} folds"
        )

    return folds


# ============================================================
# State binning
#
# Thresholds MUST be learned on past TRAIN X only.
# ============================================================

def fit_bins(
    train: pd.Series,
    *,
    bins: int,
    binary: bool = False,
):

    s = pd.to_numeric(
        train,
        errors="coerce",
    ).dropna()

    if len(s) < 200:

        return None

    if binary:

        values = sorted(
            float(v)
            for v in (
                s.unique()
            )
        )

        if len(
            values
        ) < 2:

            return None

        return {
            "kind": "categorical",
            "values": values,
        }

    try:

        _groups, edges = (
            pd.qcut(
                s,
                q=bins,
                retbins=True,
                duplicates="drop",
            )
        )

    except ValueError:

        return None

    edges = np.asarray(
        edges,
        dtype=float,
    )

    if len(
        edges
    ) < 3:

        return None

    # Future values may exceed
    # historical train extrema.
    edges[0] = -np.inf
    edges[-1] = np.inf

    return {
        "kind": "edges",
        "edges": edges,
    }


def apply_bins(
    values: pd.Series,
    spec,
) -> pd.Series:

    s = pd.to_numeric(
        values,
        errors="coerce",
    )

    if spec[
        "kind"
    ] == "categorical":

        mapping = {
            v: i + 1
            for i, v
            in enumerate(
                spec[
                    "values"
                ]
            )
        }

        return (
            s.map(
                mapping
            )
            .astype(
                "Float64"
            )
        )

    binned = pd.cut(
        s,
        bins=spec[
            "edges"
        ],
        labels=False,
        include_lowest=True,
    )

    return (
        binned.astype(
            "Float64"
        )
        + 1
    )


# ============================================================
# Descriptive feature distribution
# ============================================================

def describe_feature(
    s: pd.Series,
    name: str,
) -> dict:

    x = pd.to_numeric(
        s,
        errors="coerce",
    ).dropna()

    return {
        "feature": name,

        "n": int(
            len(x)
        ),

        "mean": float(
            x.mean()
        ),

        "std": float(
            x.std(
                ddof=0
            )
        ),

        "p01": float(
            x.quantile(
                0.01
            )
        ),

        "p10": float(
            x.quantile(
                0.10
            )
        ),

        "p25": float(
            x.quantile(
                0.25
            )
        ),

        "p50": float(
            x.quantile(
                0.50
            )
        ),

        "p75": float(
            x.quantile(
                0.75
            )
        ),

        "p90": float(
            x.quantile(
                0.90
            )
        ),

        "p99": float(
            x.quantile(
                0.99
            )
        ),

        "unique": int(
            x.nunique()
        ),
    }


# ============================================================
# Aggregate exact pooled OOS assignments
# ============================================================

def aggregate_assignments(
    assignments: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:

    rows = []

    for keys, g in (
        assignments.groupby(
            group_cols,
            observed=True,
        )
    ):

        if not isinstance(
            keys,
            tuple,
        ):
            keys = (
                keys,
            )

        row = dict(
            zip(
                group_cols,
                keys,
            )
        )

        row.update(
            target_stats(
                g[
                    "target_raw"
                ],
                g[
                    "target_norm"
                ],
            )
        )

        row[
            "folds"
        ] = int(
            g[
                "fold"
            ].nunique()
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Feature-screen aggregation
# ============================================================

def summarize_feature_screen(
    contrasts: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        horizon,
        feature,
    ), g in (
        contrasts.groupby(
            [
                "horizon",
                "feature",
            ],
            observed=True,
        )
    ):

        def sign_consistency(
            col: str,
        ) -> float:

            x = pd.to_numeric(
                g[col],
                errors="coerce",
            ).dropna()

            if len(x) == 0:

                return float(
                    "nan"
                )

            positive = float(
                (
                    x > 0
                ).mean()
            )

            negative = float(
                (
                    x < 0
                ).mean()
            )

            return max(
                positive,
                negative,
            )

        rows.append(
            {
                "horizon": int(
                    horizon
                ),

                "approx_minutes": (
                    int(horizon)
                    * 15
                ),

                "feature": feature,

                "folds": int(
                    g[
                        "fold"
                    ].nunique()
                ),

                # ----------------------------------------
                # Direction center
                # ----------------------------------------

                "median_q50_high_minus_low": float(
                    g[
                        "q50_high_minus_low"
                    ].median()
                ),

                "mean_q50_high_minus_low": float(
                    g[
                        "q50_high_minus_low"
                    ].mean()
                ),

                "q50_spread_sign_consistency": (
                    sign_consistency(
                        "q50_high_minus_low"
                    )
                ),

                # ----------------------------------------
                # Future distribution width
                # ----------------------------------------

                "median_width_high_minus_low": float(
                    g[
                        "width_high_minus_low"
                    ].median()
                ),

                "width_spread_sign_consistency": (
                    sign_consistency(
                        "width_high_minus_low"
                    )
                ),

                # ----------------------------------------
                # Tail asymmetry
                # ----------------------------------------

                "median_asym_high_minus_low": float(
                    g[
                        "asym_high_minus_low"
                    ].median()
                ),

                "asym_spread_sign_consistency": (
                    sign_consistency(
                        "asym_high_minus_low"
                    )
                ),

                "median_positive_rate_high_minus_low": float(
                    g[
                        "positive_rate_high_minus_low"
                    ].median()
                ),

                # ----------------------------------------
                # Does conditional distribution
                # move monotonically through bins?
                # ----------------------------------------

                "median_q50_bin_monotonicity": float(
                    g[
                        "q50_bin_monotonicity"
                    ].median()
                ),

                "median_width_bin_monotonicity": float(
                    g[
                        "width_bin_monotonicity"
                    ].median()
                ),

                "median_asym_bin_monotonicity": float(
                    g[
                        "asym_bin_monotonicity"
                    ].median()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    prepare_output_dir()

    if not INPUT.is_file():

        raise RuntimeError(
            f"Missing input panel: "
            f"{INPUT}"
        )

    panel = pd.read_csv(
        INPUT,
        parse_dates=[
            "meta_base_bar_time",
            "meta_decision_time",
            "meta_feature_last_5m_time",
        ],
        low_memory=False,
    )

    manifest = json.loads(
        SOURCE_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    expected_rows = int(
        manifest[
            "outputs"
        ][
            "quantile_panel_15m.csv"
        ][
            "rows"
        ]
    )

    if len(
        panel
    ) != expected_rows:

        raise RuntimeError(
            f"Panel rows "
            f"{len(panel)} "
            f"!= manifest "
            f"{expected_rows}"
        )

    # Lock current research dataset.
    if len(
        panel
    ) != 3300:

        raise RuntimeError(
            f"Unexpected panel size: "
            f"{len(panel)}"
        )

    missing = [
        col
        for col in CORE_FEATURES
        if col not in (
            panel.columns
        )
    ]

    if missing:

        raise RuntimeError(
            "Missing core features: "
            f"{missing}"
        )

    bad_features = [

        col
        for col in CORE_FEATURES

        if any(
            token
            in col.lower()

            for token
            in FORBIDDEN_FEATURE_TOKENS
        )
    ]

    if bad_features:

        raise RuntimeError(
            "Forbidden features "
            f"selected: "
            f"{bad_features}"
        )

    for h in HORIZONS:

        for prefix in (
            "target_raw_return_h",
            "target_norm_return_h",
        ):

            col = (
                f"{prefix}{h}"
            )

            if col not in (
                panel.columns
            ):

                raise RuntimeError(
                    f"Missing target: "
                    f"{col}"
                )

    # ========================================================
    # A. X-only distributions
    # ========================================================

    feature_distribution = pd.DataFrame(
        [
            describe_feature(
                panel[col],
                col,
            )
            for col in CORE_FEATURES
        ]
    )

    feature_distribution.to_csv(
        OUT
        / "core_feature_distribution.csv",
        index=False,
    )

    # ========================================================
    # B. X-only redundancy
    # ========================================================

    corr = (
        panel[
            CORE_FEATURES
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .corr(
            method="spearman"
        )
    )

    redundancy_rows = []

    for i, feature_a in enumerate(
        CORE_FEATURES
    ):

        for feature_b in (
            CORE_FEATURES[
                i + 1:
            ]
        ):

            rho = float(
                corr.loc[
                    feature_a,
                    feature_b,
                ]
            )

            redundancy_rows.append(
                {
                    "feature_a": (
                        feature_a
                    ),

                    "feature_b": (
                        feature_b
                    ),

                    "spearman_rho": (
                        rho
                    ),

                    "abs_spearman_rho": (
                        abs(rho)
                    ),

                    "high_redundancy": (
                        abs(rho)
                        >= 0.75
                    ),
                }
            )

    redundancy = pd.DataFrame(
        redundancy_rows
    ).sort_values(
        "abs_spearman_rho",
        ascending=False,
    )

    redundancy.to_csv(
        OUT
        / "feature_redundancy.csv",
        index=False,
    )

    # ========================================================
    # Experiment containers
    # ========================================================

    target_distribution_rows = []

    marginal_fold_bin_rows = []

    marginal_contrast_rows = []

    marginal_assignments = []

    pair_fold_cell_rows = []

    pair_assignments = []

    folds_by_horizon = {}

    # ========================================================
    # C. Horizon loop
    # ========================================================

    for h in HORIZONS:

        raw_col = (
            f"target_raw_return_h{h}"
        )

        norm_col = (
            f"target_norm_return_h{h}"
        )

        valid = (
            panel[
                raw_col
            ].notna()
            &
            panel[
                norm_col
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

        folds = make_folds(
            len(frame),
            horizon=h,
        )

        folds_by_horizon[
            str(h)
        ] = folds

        # ----------------------------------------------------
        # Unconditional distribution
        # ----------------------------------------------------

        target_distribution_rows.append(
            {
                "horizon": h,

                "approx_minutes": (
                    h * 15
                ),

                "scope": (
                    "full_valid_sample"
                ),

                **target_stats(
                    frame[
                        raw_col
                    ],
                    frame[
                        norm_col
                    ],
                ),
            }
        )

        oos_mask = np.zeros(
            len(frame),
            dtype=bool,
        )

        for fold in folds:

            oos_mask[
                fold[
                    "test_start"
                ]:
                fold[
                    "test_end_exclusive"
                ]
            ] = True

        target_distribution_rows.append(
            {
                "horizon": h,

                "approx_minutes": (
                    h * 15
                ),

                "scope": (
                    "pooled_purged_oos"
                ),

                **target_stats(
                    frame.loc[
                        oos_mask,
                        raw_col,
                    ],
                    frame.loc[
                        oos_mask,
                        norm_col,
                    ],
                ),
            }
        )

        # ====================================================
        # D. Single-feature state screening
        # ====================================================

        for feature in CORE_FEATURES:

            for fold in folds:

                train = (
                    frame.iloc[
                        fold[
                            "train_start"
                        ]:
                        fold[
                            "train_end_exclusive"
                        ]
                    ]
                )

                test = (
                    frame.iloc[
                        fold[
                            "test_start"
                        ]:
                        fold[
                            "test_end_exclusive"
                        ]
                    ]
                    .copy()
                )

                # IMPORTANT:
                # Feature-state thresholds are learned
                # only using past training X.
                spec = fit_bins(
                    train[
                        feature
                    ],

                    bins=(
                        MARGINAL_BINS
                    ),

                    binary=(
                        feature
                        in BINARY_FEATURES
                    ),
                )

                if spec is None:

                    continue

                test[
                    "state_bin"
                ] = apply_bins(
                    test[
                        feature
                    ],
                    spec,
                )

                test = test[
                    test[
                        "state_bin"
                    ].notna()
                ].copy()

                if test.empty:

                    continue

                baseline = target_stats(
                    test[
                        raw_col
                    ],
                    test[
                        norm_col
                    ],
                )

                bin_stats = []

                for (
                    state_bin,
                    group,
                ) in test.groupby(
                    "state_bin",
                    observed=True,
                ):

                    stats = target_stats(
                        group[
                            raw_col
                        ],
                        group[
                            norm_col
                        ],
                    )

                    row = {
                        "horizon": h,

                        "approx_minutes": (
                            h * 15
                        ),

                        "feature": feature,

                        "fold": (
                            fold[
                                "fold"
                            ]
                        ),

                        "train_rows": int(
                            len(train)
                        ),

                        "test_rows": int(
                            len(test)
                        ),

                        "purge_rows": (
                            fold[
                                "purge_rows"
                            ]
                        ),

                        "state_bin": int(
                            state_bin
                        ),

                        "feature_min": float(
                            pd.to_numeric(
                                group[
                                    feature
                                ],
                                errors="coerce",
                            ).min()
                        ),

                        "feature_max": float(
                            pd.to_numeric(
                                group[
                                    feature
                                ],
                                errors="coerce",
                            ).max()
                        ),

                        **stats,

                        "baseline_q10": (
                            baseline[
                                "q10"
                            ]
                        ),

                        "baseline_q50": (
                            baseline[
                                "q50"
                            ]
                        ),

                        "baseline_q90": (
                            baseline[
                                "q90"
                            ]
                        ),
                    }

                    marginal_fold_bin_rows.append(
                        row
                    )

                    bin_stats.append(
                        row
                    )

                # ------------------------------------------------
                # High state vs low state
                # ------------------------------------------------

                if len(
                    bin_stats
                ) >= 2:

                    bdf = pd.DataFrame(
                        bin_stats
                    ).sort_values(
                        "state_bin"
                    )

                    low = (
                        bdf.iloc[
                            0
                        ]
                    )

                    high = (
                        bdf.iloc[
                            -1
                        ]
                    )

                    marginal_contrast_rows.append(
                        {
                            "horizon": h,

                            "feature": (
                                feature
                            ),

                            "fold": (
                                fold[
                                    "fold"
                                ]
                            ),

                            "low_bin": int(
                                low[
                                    "state_bin"
                                ]
                            ),

                            "high_bin": int(
                                high[
                                    "state_bin"
                                ]
                            ),

                            "low_n": int(
                                low[
                                    "n"
                                ]
                            ),

                            "high_n": int(
                                high[
                                    "n"
                                ]
                            ),

                            # --------------------------------
                            # Direction
                            # --------------------------------

                            "q50_high_minus_low": float(
                                high[
                                    "q50"
                                ]
                                -
                                low[
                                    "q50"
                                ]
                            ),

                            # --------------------------------
                            # Opportunity / dispersion
                            # --------------------------------

                            "width_high_minus_low": float(
                                high[
                                    "width_90_10"
                                ]
                                -
                                low[
                                    "width_90_10"
                                ]
                            ),

                            # --------------------------------
                            # Tail asymmetry
                            # --------------------------------

                            "asym_high_minus_low": float(
                                high[
                                    "asym_90_plus_10"
                                ]
                                -
                                low[
                                    "asym_90_plus_10"
                                ]
                            ),

                            "positive_rate_high_minus_low": float(
                                high[
                                    "positive_rate"
                                ]
                                -
                                low[
                                    "positive_rate"
                                ]
                            ),

                            # --------------------------------
                            # Monotonicity across state bins
                            # --------------------------------

                            "q50_bin_monotonicity": (
                                safe_spearman(
                                    bdf[
                                        "state_bin"
                                    ],
                                    bdf[
                                        "q50"
                                    ],
                                )
                            ),

                            "width_bin_monotonicity": (
                                safe_spearman(
                                    bdf[
                                        "state_bin"
                                    ],
                                    bdf[
                                        "width_90_10"
                                    ],
                                )
                            ),

                            "asym_bin_monotonicity": (
                                safe_spearman(
                                    bdf[
                                        "state_bin"
                                    ],
                                    bdf[
                                        "asym_90_plus_10"
                                    ],
                                )
                            ),
                        }
                    )

                marginal_assignments.append(
                    pd.DataFrame(
                        {
                            "horizon": h,

                            "feature": (
                                feature
                            ),

                            "fold": (
                                fold[
                                    "fold"
                                ]
                            ),

                            "state_bin": (
                                test[
                                    "state_bin"
                                ]
                                .astype(int)
                            ),

                            "decision_time": (
                                test[
                                    "meta_decision_time"
                                ].to_numpy()
                            ),

                            "target_raw": (
                                test[
                                    raw_col
                                ].to_numpy()
                            ),

                            "target_norm": (
                                test[
                                    norm_col
                                ].to_numpy()
                            ),
                        }
                    )
                )

        # ====================================================
        # E. Pre-specified 2D regime experiments
        # ====================================================

        for (
            pair_name,
            feature_x,
            feature_y,
        ) in PAIR_SPECS:

            for fold in folds:

                train = (
                    frame.iloc[
                        fold[
                            "train_start"
                        ]:
                        fold[
                            "train_end_exclusive"
                        ]
                    ]
                )

                test = (
                    frame.iloc[
                        fold[
                            "test_start"
                        ]:
                        fold[
                            "test_end_exclusive"
                        ]
                    ]
                    .copy()
                )

                spec_x = fit_bins(
                    train[
                        feature_x
                    ],
                    bins=PAIR_BINS,
                )

                spec_y = fit_bins(
                    train[
                        feature_y
                    ],
                    bins=PAIR_BINS,
                )

                if (
                    spec_x is None
                    or spec_y is None
                ):
                    continue

                test[
                    "x_bin"
                ] = apply_bins(
                    test[
                        feature_x
                    ],
                    spec_x,
                )

                test[
                    "y_bin"
                ] = apply_bins(
                    test[
                        feature_y
                    ],
                    spec_y,
                )

                test = test[
                    test[
                        "x_bin"
                    ].notna()
                    &
                    test[
                        "y_bin"
                    ].notna()
                ].copy()

                for (
                    x_bin,
                    y_bin,
                ), group in test.groupby(
                    [
                        "x_bin",
                        "y_bin",
                    ],
                    observed=True,
                ):

                    pair_fold_cell_rows.append(
                        {
                            "horizon": h,

                            "approx_minutes": (
                                h * 15
                            ),

                            "pair": (
                                pair_name
                            ),

                            "feature_x": (
                                feature_x
                            ),

                            "feature_y": (
                                feature_y
                            ),

                            "fold": (
                                fold[
                                    "fold"
                                ]
                            ),

                            "x_bin": int(
                                x_bin
                            ),

                            "y_bin": int(
                                y_bin
                            ),

                            **target_stats(
                                group[
                                    raw_col
                                ],
                                group[
                                    norm_col
                                ],
                            ),
                        }
                    )

                pair_assignments.append(
                    pd.DataFrame(
                        {
                            "horizon": h,

                            "pair": (
                                pair_name
                            ),

                            "feature_x": (
                                feature_x
                            ),

                            "feature_y": (
                                feature_y
                            ),

                            "fold": (
                                fold[
                                    "fold"
                                ]
                            ),

                            "x_bin": (
                                test[
                                    "x_bin"
                                ]
                                .astype(int)
                            ),

                            "y_bin": (
                                test[
                                    "y_bin"
                                ]
                                .astype(int)
                            ),

                            "decision_time": (
                                test[
                                    "meta_decision_time"
                                ].to_numpy()
                            ),

                            "target_raw": (
                                test[
                                    raw_col
                                ].to_numpy()
                            ),

                            "target_norm": (
                                test[
                                    norm_col
                                ].to_numpy()
                            ),
                        }
                    )
                )

    # ========================================================
    # F. Final aggregate tables
    # ========================================================

    target_distribution = pd.DataFrame(
        target_distribution_rows
    )

    marginal_fold_bins = pd.DataFrame(
        marginal_fold_bin_rows
    )

    marginal_contrasts = pd.DataFrame(
        marginal_contrast_rows
    )

    marginal_assignment_df = pd.concat(
        marginal_assignments,
        ignore_index=True,
    )

    pair_fold_cells = pd.DataFrame(
        pair_fold_cell_rows
    )

    pair_assignment_df = pd.concat(
        pair_assignments,
        ignore_index=True,
    )

    marginal_pooled = (
        aggregate_assignments(
            marginal_assignment_df,
            [
                "horizon",
                "feature",
                "state_bin",
            ],
        )
    )

    pair_summary = (
        aggregate_assignments(
            pair_assignment_df,
            [
                "horizon",
                "pair",
                "feature_x",
                "feature_y",
                "x_bin",
                "y_bin",
            ],
        )
    )

    feature_screen = (
        summarize_feature_screen(
            marginal_contrasts
        )
    )

    # ========================================================
    # G. Leakage / duplicate validation
    # ========================================================

    if (
        marginal_assignment_df
        .duplicated(
            [
                "horizon",
                "feature",
                "decision_time",
            ]
        )
        .any()
    ):

        raise RuntimeError(
            "Duplicate marginal "
            "OOS assignment"
        )

    if (
        pair_assignment_df
        .duplicated(
            [
                "horizon",
                "pair",
                "decision_time",
            ]
        )
        .any()
    ):

        raise RuntimeError(
            "Duplicate pair "
            "OOS assignment"
        )

    for (
        h_string,
        folds,
    ) in (
        folds_by_horizon
        .items()
    ):

        h = int(
            h_string
        )

        for fold in folds:

            if (
                fold[
                    "test_start"
                ]
                -
                fold[
                    "train_end_exclusive"
                ]
            ) < h:

                raise RuntimeError(
                    f"h={h}: "
                    "purge failed"
                )

    # ========================================================
    # H. Write outputs
    # ========================================================

    target_distribution.to_csv(
        OUT
        / "target_distribution.csv",
        index=False,
    )

    marginal_fold_bins.to_csv(
        OUT
        / "marginal_fold_bins.csv",
        index=False,
    )

    marginal_contrasts.to_csv(
        OUT
        / "marginal_fold_contrasts.csv",
        index=False,
    )

    marginal_pooled.to_csv(
        OUT
        / "marginal_pooled_bins.csv",
        index=False,
    )

    feature_screen.to_csv(
        OUT
        / "feature_screen.csv",
        index=False,
    )

    pair_fold_cells.to_csv(
        OUT
        / "pair_fold_cells.csv",
        index=False,
    )

    pair_summary.to_csv(
        OUT
        / "pair_summary.csv",
        index=False,
    )

    # ========================================================
    # I. Compact config
    # ========================================================

    config = {
        "purpose": (
            "Purged expanding-OOS "
            "descriptive state screening "
            "before Quantile V2 model fitting."
        ),

        "source_panel": str(
            INPUT.relative_to(
                ROOT
            )
        ),

        "panel_rows": int(
            len(panel)
        ),

        "horizons": list(
            HORIZONS
        ),

        "quantiles_reported": [
            0.10,
            0.50,
            0.90,
        ],

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
            "horizon-sized gap "
            "between training "
            "and OOS test"
        ),

        "marginal_bins": (
            MARGINAL_BINS
        ),

        "pair_bins": (
            PAIR_BINS
        ),

        "core_features": (
            CORE_FEATURES
        ),

        "pair_specs": [
            {
                "name": name,
                "x": x,
                "y": y,
            }
            for (
                name,
                x,
                y,
            )
            in PAIR_SPECS
        ],

        "uses_model": False,

        "uses_backtest": False,

        "uses_smc": False,

        "uses_momentum": False,

        "uses_oracle_as_feature": (
            False
        ),

        "statistical_inference_claim": (
            False
        ),

        "note": (
            "State thresholds are fitted "
            "using past training X only. "
            "Conditional target quantiles "
            "are measured on later "
            "purged OOS blocks."
        ),
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

    # ========================================================
    # J. Simple factual Markdown
    # ========================================================

    lines = [
        "# Quantile V2 State Exploration",
        "",
        "No predictive model.",
        "No backtest.",
        "No p-value claim.",
        "",
        "All state thresholds were learned "
        "from past training X only and "
        "applied to later purged OOS blocks.",
        "",
    ]

    for h in HORIZONS:

        subset = feature_screen[
            feature_screen[
                "horizon"
            ]
            == h
        ].copy()

        subset[
            "screen_rank"
        ] = (
            subset[
                "median_q50_high_minus_low"
            ].abs()
            *
            subset[
                "q50_spread_sign_consistency"
            ].fillna(
                0.0
            )
        )

        subset = (
            subset
            .sort_values(
                "screen_rank",
                ascending=False,
            )
            .head(
                8
            )
        )

        lines.extend(
            [
                (
                    f"## H={h} "
                    f"({h * 15}m)"
                ),
                "",
                (
                    "| feature | "
                    "median Q50 high-low | "
                    "sign consistency | "
                    "Q50 monotonicity | "
                    "width high-low |"
                ),
                (
                    "|---|---:|---:|---:|---:|"
                ),
            ]
        )

        for _, row in (
            subset.iterrows()
        ):

            lines.append(
                f"| {row['feature']} "
                f"| "
                f"{row['median_q50_high_minus_low']:.6f} "
                f"| "
                f"{row['q50_spread_sign_consistency']:.3f} "
                f"| "
                f"{row['median_q50_bin_monotonicity']:.3f} "
                f"| "
                f"{row['median_width_high_minus_low']:.6f} "
                f"|"
            )

        lines.append(
            ""
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

    # ========================================================
    # K. Final validation
    # ========================================================

    expected_files = [
        "experiment_config.json",

        "target_distribution.csv",

        "core_feature_distribution.csv",

        "feature_redundancy.csv",

        "marginal_fold_bins.csv",

        "marginal_fold_contrasts.csv",

        "marginal_pooled_bins.csv",

        "feature_screen.csv",

        "pair_fold_cells.csv",

        "pair_summary.csv",

        "run_summary.md",
    ]

    file_sizes_mb = {}

    for name in expected_files:

        path = (
            OUT
            / name
        )

        if not path.is_file():

            raise RuntimeError(
                f"Missing output: "
                f"{name}"
            )

        size_mb = (
            path.stat().st_size
            / 1024
            / 1024
        )

        file_sizes_mb[
            name
        ] = round(
            size_mb,
            4,
        )

        if size_mb > 50:

            raise RuntimeError(
                f"{name} > 50MB"
            )

    numeric_outputs = [
        target_distribution,

        feature_distribution,

        redundancy,

        marginal_fold_bins,

        marginal_contrasts,

        marginal_pooled,

        feature_screen,

        pair_fold_cells,

        pair_summary,
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
                "Output contains "
                "+/- infinity"
            )

    validation = {
        "status": "PASS",

        "source_rows": int(
            len(panel)
        ),

        "core_feature_count": int(
            len(
                CORE_FEATURES
            )
        ),

        "pair_spec_count": int(
            len(
                PAIR_SPECS
            )
        ),

        "fold_count_by_horizon": {
            str(h): len(
                folds_by_horizon[
                    str(h)
                ]
            )
            for h in HORIZONS
        },

        "marginal_fold_bin_rows": int(
            len(
                marginal_fold_bins
            )
        ),

        "feature_screen_rows": int(
            len(
                feature_screen
            )
        ),

        "pair_summary_rows": int(
            len(
                pair_summary
            )
        ),

        "no_future_feature_bins": True,

        "purged_oos": True,

        "no_model": True,

        "no_backtest": True,

        "no_smc": True,

        "no_momentum": True,

        "no_oracle_feature": True,

        "file_sizes_mb": (
            file_sizes_mb
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

    print(
        json.dumps(
            validation,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "QUANTILE_V2_STATE_EXPLORE_PASS"
    )


if __name__ == "__main__":
    main()
