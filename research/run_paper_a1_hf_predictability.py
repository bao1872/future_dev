#!/usr/bin/env python3

from __future__ import (
    annotations,
)

import json
import sys
from pathlib import (
    Path,
)

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LassoCV,
)
from sklearn.model_selection import (
    TimeSeriesSplit,
)
from sklearn.pipeline import (
    Pipeline,
)
from sklearn.preprocessing import (
    StandardScaler,
)


ROOT = Path(
    __file__
).resolve().parents[1]

# Make the `research` package importable when run as
# `python research/run_paper_a1_hf_predictability.py` from ROOT.
if str(
    ROOT
) not in sys.path:
    sys.path.insert(
        0,
        str(
            ROOT
        ),
    )

from research.run_direction_v3r import (
    INSTRUMENTS,
    SRC_5M,
)


OUT = (
    ROOT
    / "research"
    / "exports"
    / "paper_a1_hf_predictability"
)

# Paper-inspired exponentially increasing local horizons.
# 5m data -> 5, 10, 20, 40, 80 minutes.
WINDOWS = (
    1,
    2,
    4,
    8,
    16,
)

PRIMARY_HORIZON = 3      # 15m
SECONDARY_HORIZON = 1    # 5m
HORIZONS = (
    1,
    3,
)

INITIAL_TRAIN_MONTHS = 6
TEST_MONTHS = 1

MIN_TRAIN_ROWS = 4000
MIN_TEST_ROWS = 500

FIVE_MIN = pd.Timedelta(
    minutes=5,
)

RANDOM_STATE = 42


# ----------------------------------------------------------------------
# Strict session continuity
# ----------------------------------------------------------------------

def add_session_structure(
    df: pd.DataFrame,
) -> pd.DataFrame:

    x = (
        df
        .sort_values(
            "bar_start_time",
        )
        .drop_duplicates(
            "bar_start_time",
        )
        .reset_index(
            drop=True,
        )
        .copy()
    )

    delta = x[
        "bar_start_time"
    ].diff()

    new_session = (
        delta.isna()
        | (
            delta
            != FIVE_MIN
        )
    )

    x["session_id"] = (
        new_session.cumsum()
    )

    x["session_pos"] = (
        x.groupby(
            "session_id"
        ).cumcount()
    )

    return x


# ----------------------------------------------------------------------
# Paper-driven reduced feature set (Price + Trade only)
# ----------------------------------------------------------------------

def build_features_one_session(
    g: pd.DataFrame,
) -> pd.DataFrame:

    g = g.copy()

    close = pd.to_numeric(
        g["close"],
        errors="coerce",
    )

    volume = pd.to_numeric(
        g["trade"],
        errors="coerce",
    )

    logp = np.log(
        close
    )

    r1 = logp.diff()

    out = pd.DataFrame(
        index=g.index,
    )

    for k in WINDOWS:

        # ----------------------------------
        # 1. Past return
        # ----------------------------------

        out[
            f"past_return_{k}"
        ] = (
            logp
            - logp.shift(
                k
            )
        )

        # ----------------------------------
        # 2. Total trading activity
        # ----------------------------------

        vol_sum = (
            volume
            .rolling(
                k,
                min_periods=k,
            )
            .sum()
        )

        out[
            f"volume_sum_{k}"
        ] = vol_sum

        # ----------------------------------
        # 3. Largest single-bar activity
        # ----------------------------------

        out[
            f"volume_max_{k}"
        ] = (
            volume
            .rolling(
                k,
                min_periods=k,
            )
            .max()
        )

        # ----------------------------------
        # 4. Simplified price impact proxy
        #
        # Not paper's exact Lambda because
        # trade-direction data are unavailable.
        # ----------------------------------

        out[
            f"price_impact_proxy_{k}"
        ] = (
            out[
                f"past_return_{k}"
            ]
            / vol_sum.replace(
                0,
                np.nan,
            )
        )

        # ----------------------------------
        # 5. Short-run return autocovariance
        # ----------------------------------

        cross = (
            r1
            * r1.shift(
                1
            )
        )

        out[
            f"return_autocov_{k}"
        ] = (
            cross
            .rolling(
                k,
                min_periods=k,
            )
            .mean()
        )

        # ----------------------------------
        # 6. Absolute variation / activity
        #
        # Reduced-information proxy for
        # price movement intensity.
        # ----------------------------------

        out[
            f"abs_return_sum_{k}"
        ] = (
            r1.abs()
            .rolling(
                k,
                min_periods=k,
            )
            .sum()
        )

    return out


# ----------------------------------------------------------------------
# Targets (locked execution semantics)
# ----------------------------------------------------------------------

def build_targets_one_session(
    g: pd.DataFrame,
) -> pd.DataFrame:

    out = pd.DataFrame(
        index=g.index,
    )

    open_ = pd.to_numeric(
        g["open"],
        errors="coerce",
    )

    close = pd.to_numeric(
        g["close"],
        errors="coerce",
    )

    next_open = open_.shift(
        -1
    )

    out["target_h1"] = (
        close.shift(
            -1
        )
        / next_open
        - 1.0
    )

    out["target_h3"] = (
        close.shift(
            -3
        )
        / next_open
        - 1.0
    )

    return out


# ----------------------------------------------------------------------
# Full panel per instrument
# ----------------------------------------------------------------------

def build_panel(
    instrument: str,
) -> pd.DataFrame:

    raw = pd.read_csv(
        SRC_5M
        / f"{instrument}_5m.csv",
        parse_dates=[
            "bar_start_time",
            "bar_end_time",
            "availability_time",
            "trading_day",
        ],
    )

    x = add_session_structure(
        raw
    )

    feature_parts = []
    target_parts = []

    for _, g in x.groupby(
        "session_id",
        sort=False,
    ):

        feature_parts.append(
            build_features_one_session(
                g
            )
        )

        target_parts.append(
            build_targets_one_session(
                g
            )
        )

    features = pd.concat(
        feature_parts
    ).sort_index()

    targets = pd.concat(
        target_parts
    ).sort_index()

    panel = pd.concat(
        [
            x[
                [
                    "bar_start_time",
                    "session_id",
                    "session_pos",
                ]
            ],
            features,
            targets,
        ],
        axis=1,
    )

    return panel


# ----------------------------------------------------------------------
# Calendar expanding folds
# ----------------------------------------------------------------------

def make_folds(
    times: pd.Series,
) -> list[dict]:

    start = (
        pd.Timestamp(
            times.min()
        ).normalize()
    )

    end = pd.Timestamp(
        times.max()
    )

    train_end = (
        start
        + pd.DateOffset(
            months=INITIAL_TRAIN_MONTHS
        )
    )

    folds = []
    fold_id = 0

    while True:

        test_start = train_end

        test_end = (
            test_start
            + pd.DateOffset(
                months=TEST_MONTHS
            )
        )

        if test_start >= end:
            break

        folds.append(
            {
                "fold": fold_id,
                "train_start": start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": min(
                    test_end,
                    end
                    + pd.Timedelta(
                        seconds=1
                    ),
                ),
            }
        )

        fold_id += 1

        train_end = test_end

    return folds


# ----------------------------------------------------------------------
# Two models (fixed specifications)
# ----------------------------------------------------------------------

def make_lasso():

    return Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "model",
                LassoCV(
                    cv=TimeSeriesSplit(
                        n_splits=5,
                    ),
                    n_alphas=50,
                    max_iter=20000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_rf():

    return RandomForestRegressor(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=50,
        max_features=0.5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


# ----------------------------------------------------------------------
# Spearman
# ----------------------------------------------------------------------

def safe_spearman(
    a,
    b,
):

    x = pd.DataFrame(
        {
            "a": a,
            "b": b,
        }
    ).dropna()

    if len(
        x
    ) < 20:
        return np.nan

    return float(
        x["a"].corr(
            x["b"],
            method="spearman",
        )
    )


# ----------------------------------------------------------------------
# Single-fold core
# ----------------------------------------------------------------------

def run_one_fold(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    fold: dict,
):

    train_mask = (
        (
            frame["bar_start_time"]
            >= fold["train_start"]
        )
        & (
            frame["bar_start_time"]
            < fold["train_end"]
        )
    )

    test_mask = (
        (
            frame["bar_start_time"]
            >= fold["test_start"]
        )
        & (
            frame["bar_start_time"]
            < fold["test_end"]
        )
    )

    use_cols = (
        feature_cols
        + [target_col]
    )

    train = (
        frame.loc[
            train_mask,
            use_cols,
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    test = (
        frame.loc[
            test_mask,
            [
                "bar_start_time",
                "session_pos",
            ]
            + use_cols,
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    if (
        len(
            train
        )
        < MIN_TRAIN_ROWS
        or len(
            test
        )
        < MIN_TEST_ROWS
    ):
        return None

    X_train = train[
        feature_cols
    ]

    X_test = test[
        feature_cols
    ]

    # Fit in basis points for numerical stability.
    y_train_bps = (
        train[target_col]
        .to_numpy(
            dtype=float
        )
        * 10000.0
    )

    y_test = (
        test[target_col]
        .to_numpy(
            dtype=float
        )
    )

    train_mean = float(
        train[target_col].mean()
    )

    results = {}

    models = {
        "LASSO": make_lasso(),
        "RF": make_rf(),
    }

    for model_name, model in (
        models.items()
    ):

        model.fit(
            X_train,
            y_train_bps,
        )

        pred_bps = model.predict(
            X_test
        )

        pred = (
            pred_bps
            / 10000.0
        )

        baseline = np.full(
            len(
                y_test
            ),
            train_mean,
        )

        mse_model = float(
            np.mean(
                (
                    y_test
                    - pred
                )
                ** 2
            )
        )

        mse_base = float(
            np.mean(
                (
                    y_test
                    - baseline
                )
                ** 2
            )
        )

        oos_r2 = (
            1.0
            - mse_model
            / mse_base
            if mse_base
            > 0
            else np.nan
        )

        ic = safe_spearman(
            pred,
            y_test,
        )

        # Simple economic diagnostic:
        # position = sign forecast.
        position = np.sign(
            pred
        )

        gross_return = (
            position
            * y_test
        )

        results[
            model_name
        ] = {
            "model": model,
            "pred": pred,
            "y": y_test,
            "times": (
                test[
                    "bar_start_time"
                ]
                .to_numpy()
            ),
            "session_pos": (
                test[
                    "session_pos"
                ]
                .to_numpy()
            ),
            "oos_r2": float(
                oos_r2
            ),
            "ic": float(
                ic
            ),
            "gross_mean": float(
                np.mean(
                    gross_return
                )
            ),
            "gross_win_rate": float(
                np.mean(
                    gross_return
                    > 0
                )
            ),
        }

    return results


# ----------------------------------------------------------------------
# H15 overlap handling
# ----------------------------------------------------------------------

def trading_metrics(
    pred: np.ndarray,
    y: np.ndarray,
    session_pos: np.ndarray,
    horizon: int,
):

    position = np.sign(
        pred
    )

    gross = (
        position
        * y
    )

    rows = []

    if horizon == 1:
        offsets = (
            0,
        )
    else:
        offsets = (
            0,
            1,
            2,
        )

    for offset in offsets:

        if horizon == 1:

            mask = np.ones(
                len(
                    gross
                ),
                dtype=bool,
            )

        else:

            mask = (
                session_pos
                % 3
                == offset
            )

        r = gross[
            mask
        ]

        if len(
            r
        ) == 0:
            continue

        mean_r = float(
            np.mean(
                r
            )
        )

        gross_bps = (
            mean_r
            * 10000.0
        )

        rows.append(
            {
                "offset": offset,
                "n_trades": int(
                    len(
                        r
                    )
                ),
                "mean_gross_return": (
                    mean_r
                ),
                "mean_gross_bps": (
                    gross_bps
                ),
                "median_gross_return": float(
                    np.median(
                        r
                    )
                ),
                "win_rate": float(
                    np.mean(
                        r
                        > 0
                    )
                ),

                # Complete round trip:
                # max total execution cost
                # before expected return reaches zero.
                "break_even_roundtrip_cost_bps": (
                    gross_bps
                ),

                # If cost is quoted per side:
                "break_even_oneway_cost_bps": (
                    gross_bps
                    / 2.0
                ),
            }
        )

    return rows


# ----------------------------------------------------------------------
# Summary: RF vs LASSO
# ----------------------------------------------------------------------

def build_summary(
    folds_df,
    trades_df,
):

    summary = (
        folds_df
        .groupby(
            [
                "instrument",
                "horizon",
                "model",
            ],
            observed=True,
        )
        .agg(
            folds=(
                "fold",
                "nunique",
            ),
            median_oos_r2=(
                "oos_r2",
                "median",
            ),
            mean_oos_r2=(
                "oos_r2",
                "mean",
            ),
            median_ic=(
                "spearman_ic",
                "median",
            ),
            positive_ic_fold_share=(
                "spearman_ic",
                lambda x: float(
                    np.mean(
                        np.asarray(
                            x
                        )
                        > 0
                    )
                ),
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        OUT
        / "a1_model_summary.csv",
        index=False,
    )

    # Same fold paired RF - LASSO.
    p = folds_df.pivot(
        index=[
            "instrument",
            "horizon",
            "fold",
        ],
        columns="model",
        values=[
            "oos_r2",
            "spearman_ic",
        ],
    )

    paired_rows = []

    for (
        instrument,
        horizon,
    ), g in p.groupby(
        level=[
            0,
            1,
        ]
    ):

        if (
            (
                "oos_r2",
                "RF",
            )
            not in g.columns
            or (
                "oos_r2",
                "LASSO",
            )
            not in g.columns
        ):
            continue

        dr2 = (
            g[
                (
                    "oos_r2",
                    "RF",
                )
            ]
            - g[
                (
                    "oos_r2",
                    "LASSO",
                )
            ]
        )

        dic = (
            g[
                (
                    "spearman_ic",
                    "RF",
                )
            ]
            - g[
                (
                    "spearman_ic",
                    "LASSO",
                )
            ]
        )

        paired_rows.append(
            {
                "instrument": instrument,
                "horizon": horizon,
                "folds": len(
                    g
                ),
                "median_rf_minus_lasso_r2": float(
                    dr2.median()
                ),
                "rf_better_r2_fold_share": float(
                    np.mean(
                        dr2
                        > 0
                    )
                ),
                "median_rf_minus_lasso_ic": float(
                    dic.median()
                ),
                "rf_better_ic_fold_share": float(
                    np.mean(
                        dic
                        > 0
                    )
                ),
            }
        )

    pd.DataFrame(
        paired_rows
    ).to_csv(
        OUT
        / "a1_rf_vs_lasso.csv",
        index=False,
    )


def write_config(
    feature_cols,
):

    config = {
        "experiment": (
            "A1 Reduced-Information "
            "HF Predictability Transfer"
        ),

        "paper": (
            "Ait-Sahalia, Fan, Xue, Zhu "
            "(Management Science, 2025), "
            "How and When Are High-Frequency "
            "Stock Returns Predictable?"
        ),

        "replication_status": (
            "NOT an exact replication; "
            "historical trades/quotes/LOB "
            "are unavailable"
        ),

        "question": [
            (
                "Do exponentially spaced "
                "local price/trading-activity "
                "states predict single-instrument "
                "H5/H15 futures returns?"
            ),
            (
                "Does nonlinear Random Forest "
                "outperform sparse LASSO?"
            ),
        ],

        "windows_5m_bars": list(
            WINDOWS
        ),

        "windows_minutes": [
            5
            * x
            for x in WINDOWS
        ],

        "primary_horizon": "15m",
        "secondary_horizon": "5m",

        "execution": (
            "signal after current 5m close; "
            "entry next 5m open"
        ),

        "models": [
            "LASSO",
            "RandomForest",
        ],

        "feature_columns": (
            feature_cols
        ),

        "important_limitations": [
            (
                "No historical trades/quotes/"
                "order-book data."
            ),
            (
                "price_impact_proxy is NOT "
                "paper Lambda."
            ),
            (
                "No LOB imbalance."
            ),
            (
                "No transaction imbalance."
            ),
            (
                "No quoted/effective spread."
            ),
        ],

        "forbidden": [
            "hyperparameter search",
            "new feature families",
            "Quantile",
            "macro",
            "OI",
            "LightGBM",
            "HMM",
            "SMC",
            "DSA",
            "stop-loss",
            "take-profit",
        ],
    }

    (
        OUT
        / "a1_config.json"
    ).write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def print_audit_summary(
    folds_df: pd.DataFrame,
    trades_df: pd.DataFrame,
):
    """Terminal-only summary required by the A1 spec (section 15).

    No new models, no parameter changes, no interpretation story.
    """

    print(
        "\n=== A1 H15 PRIMARY PER INSTRUMENT ==="
    )

    for inst in INSTRUMENTS:

        fdf = folds_df[
            (
                folds_df.instrument
                == inst
            )
            & (
                folds_df.horizon
                == 3
            )
        ]

        if fdf.empty:
            continue

        lasso = fdf[
            fdf.model
            == "LASSO"
        ]

        rf = fdf[
            fdf.model
            == "RF"
        ]

        piv = fdf.pivot(
            index="fold",
            columns="model",
            values="spearman_ic",
        )

        if (
            "RF" in piv.columns
            and "LASSO" in piv.columns
        ):

            d_ic = (
                piv["RF"]
                - piv["LASSO"]
            )

            ic_delta = float(
                d_ic.median()
            )

            rf_better_ic_share = float(
                (
                    d_ic
                    > 0
                ).mean()
            )

        else:

            ic_delta = float(
                "nan"
            )

            rf_better_ic_share = float(
                "nan"
            )

        tdf = trades_df[
            (
                trades_df.instrument
                == inst
            )
            & (
                trades_df.horizon
                == 3
            )
            & (
                trades_df.model
                == "RF"
            )
        ]

        off = {}

        for o in (
            0,
            1,
            2,
        ):

            row = tdf[
                tdf.offset
                == o
            ]

            off[o] = (
                float(
                    row[
                        "mean_gross_bps"
                    ].mean()
                )
                if len(
                    row
                )
                else float(
                    "nan"
                )
            )

        pos_share = (
            float(
                (
                    tdf[
                        tdf.offset
                        == 0
                    ][
                        "mean_gross_return"
                    ]
                    > 0
                ).mean()
            )
            if len(
                tdf
            )
            else float(
                "nan"
            )
        )

        be = (
            float(
                tdf[
                    tdf.offset
                    == 0
                ][
                    "break_even_oneway_cost_bps"
                ].mean()
            )
            if len(
                tdf
            )
            else float(
                "nan"
            )
        )

        print(
            f"{inst:4s} | LASSO_IC {lasso.spearman_ic.median():+.4f} "
            f"RF_IC {rf.spearman_ic.median():+.4f} | "
            f"LASSO_R2 {lasso.oos_r2.median():+.4f} "
            f"RF_R2 {rf.oos_r2.median():+.4f} | "
            f"dIC {ic_delta:+.4f} RF>LASSO {rf_better_ic_share:.2f} | "
            f"H15_bps o0 {off[0]:+.3f} o1 {off[1]:+.3f} o2 {off[2]:+.3f} | "
            f"posShare {pos_share:.2f} BE1way {be:+.3f}"
        )

    print(
        "\n=== A1 ALL-16 AGGREGATE ==="
    )

    rows = []

    for inst in INSTRUMENTS:

        fdf = folds_df[
            (
                folds_df.instrument
                == inst
            )
            & (
                folds_df.horizon
                == 3
            )
        ]

        if fdf.empty:
            continue

        lasso = fdf[
            fdf.model
            == "LASSO"
        ]

        rf = fdf[
            fdf.model
            == "RF"
        ]

        piv = fdf.pivot(
            index="fold",
            columns="model",
            values=[
                "spearman_ic",
                "oos_r2",
            ],
        )

        d_ic = (
            piv[
                (
                    "spearman_ic",
                    "RF",
                )
            ]
            - piv[
                (
                    "spearman_ic",
                    "LASSO",
                )
            ]
        )

        d_r2 = (
            piv[
                (
                    "oos_r2",
                    "RF",
                )
            ]
            - piv[
                (
                    "oos_r2",
                    "LASSO",
                )
            ]
        )

        tdf = trades_df[
            (
                trades_df.instrument
                == inst
            )
            & (
                trades_df.horizon
                == 3
            )
            & (
                trades_df.model
                == "RF"
            )
        ]

        rows.append(
            {
                "inst": inst,
                "rf_ic_pos": bool(
                    rf.spearman_ic.median()
                    > 0
                ),
                "lasso_ic_pos": bool(
                    lasso.spearman_ic.median()
                    > 0
                ),
                "rf_gt_lasso_ic": bool(
                    rf.spearman_ic.median()
                    > lasso.spearman_ic.median()
                ),
                "rf_gt_lasso_r2": bool(
                    rf.oos_r2.median()
                    > lasso.oos_r2.median()
                ),
                "o0": float(
                    tdf[
                        tdf.offset
                        == 0
                    ][
                        "mean_gross_bps"
                    ].mean()
                ),
                "o1": float(
                    tdf[
                        tdf.offset
                        == 1
                    ][
                        "mean_gross_bps"
                    ].mean()
                ),
                "o2": float(
                    tdf[
                        tdf.offset
                        == 2
                    ][
                        "mean_gross_bps"
                    ].mean()
                ),
            }
        )

    df = pd.DataFrame(
        rows
    )

    print(
        f"RF median IC > 0     : {int(df.rf_ic_pos.sum())}/16"
    )

    print(
        f"LASSO median IC > 0  : {int(df.lasso_ic_pos.sum())}/16"
    )

    print(
        f"RF > LASSO IC        : {int(df.rf_gt_lasso_ic.sum())}/16"
    )

    print(
        f"RF > LASSO R2        : {int(df.rf_gt_lasso_r2.sum())}/16"
    )

    for o in (
        "o0",
        "o1",
        "o2",
    ):
        print(
            f"RF H15 gross {o} + : {int((df[o] > 0).sum())}/16"
        )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    if (
        OUT.exists()
        and any(
            OUT.iterdir()
        )
    ):
        raise RuntimeError(
            f"{OUT} exists and is non-empty"
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_rows = []
    trade_rows = []
    prediction_rows = []
    importance_rows = []

    for instrument in INSTRUMENTS:

        print(
            f"\n=== {instrument} ===",
            flush=True,
        )

        panel = build_panel(
            instrument
        )

        feature_cols = [
            c
            for c in panel.columns
            if c.startswith(
                (
                    "past_return_",
                    "volume_sum_",
                    "volume_max_",
                    "price_impact_proxy_",
                    "return_autocov_",
                    "abs_return_sum_",
                )
            )
        ]

        folds = make_folds(
            panel[
                "bar_start_time"
            ]
        )

        for horizon in HORIZONS:

            target_col = (
                f"target_h{horizon}"
            )

            for fold in folds:

                result = run_one_fold(
                    panel,
                    feature_cols,
                    target_col,
                    fold,
                )

                if result is None:
                    continue

                for (
                    model_name,
                    res,
                ) in result.items():

                    fold_rows.append(
                        {
                            "instrument": (
                                instrument
                            ),
                            "horizon": (
                                horizon
                            ),
                            "fold": (
                                fold["fold"]
                            ),
                            "model": (
                                model_name
                            ),
                            "oos_r2": (
                                res[
                                    "oos_r2"
                                ]
                            ),
                            "spearman_ic": (
                                res[
                                    "ic"
                                ]
                            ),
                            "gross_mean_all": (
                                res[
                                    "gross_mean"
                                ]
                            ),
                            "gross_win_rate_all": (
                                res[
                                    "gross_win_rate"
                                ]
                            ),
                        }
                    )

                    for row in trading_metrics(
                        res["pred"],
                        res["y"],
                        res["session_pos"],
                        horizon,
                    ):

                        row.update(
                            {
                                "instrument": instrument,
                                "horizon": horizon,
                                "fold": fold["fold"],
                                "model": model_name,
                            }
                        )

                        trade_rows.append(
                            row
                        )

                    for (
                        ts,
                        p,
                        y,
                        sp,
                    ) in zip(
                        res["times"],
                        res["pred"],
                        res["y"],
                        res["session_pos"],
                    ):

                        prediction_rows.append(
                            {
                                "instrument": instrument,
                                "horizon": horizon,
                                "fold": fold["fold"],
                                "model": model_name,
                                "bar_start_time": ts,
                                "session_pos": int(
                                    sp
                                ),
                                "prediction": float(
                                    p
                                ),
                                "realized_return": float(
                                    y
                                ),
                            }
                        )

                    # ------------------------
                    # Feature importance
                    # ------------------------

                    model = res[
                        "model"
                    ]

                    if model_name == "LASSO":

                        coef = (
                            model.named_steps[
                                "model"
                            ]
                            .coef_
                        )

                        for (
                            f,
                            value,
                        ) in zip(
                            feature_cols,
                            coef,
                        ):

                            importance_rows.append(
                                {
                                    "instrument": instrument,
                                    "horizon": horizon,
                                    "fold": fold["fold"],
                                    "model": model_name,
                                    "feature": f,
                                    "importance": float(
                                        abs(
                                            value
                                        )
                                    ),
                                    "selected": bool(
                                        value
                                        != 0
                                    ),
                                }
                            )

                    elif model_name == "RF":

                        imp = (
                            model
                            .feature_importances_
                        )

                        for (
                            f,
                            value,
                        ) in zip(
                            feature_cols,
                            imp,
                        ):

                            importance_rows.append(
                                {
                                    "instrument": instrument,
                                    "horizon": horizon,
                                    "fold": fold["fold"],
                                    "model": model_name,
                                    "feature": f,
                                    "importance": float(
                                        value
                                    ),
                                    "selected": True,
                                }
                            )

    folds_df = pd.DataFrame(
        fold_rows
    )

    trades_df = pd.DataFrame(
        trade_rows
    )

    preds_df = pd.DataFrame(
        prediction_rows
    )

    importance_df = pd.DataFrame(
        importance_rows
    )

    folds_df.to_csv(
        OUT
        / "a1_fold_metrics.csv",
        index=False,
    )

    trades_df.to_csv(
        OUT
        / "a1_trading_metrics.csv",
        index=False,
    )

    preds_df.to_csv(
        OUT
        / "a1_predictions.csv",
        index=False,
    )

    importance_df.to_csv(
        OUT
        / "a1_feature_importance.csv",
        index=False,
    )

    build_summary(
        folds_df,
        trades_df,
    )

    write_config(
        feature_cols,
    )

    print_audit_summary(
        folds_df,
        trades_df,
    )

    print(
        "\nA1_DONE"
    )


if __name__ == "__main__":
    main()
