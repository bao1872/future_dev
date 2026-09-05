#!/usr/bin/env python3
"""Quantile V3 validation suite: is the H4 result real?

This script does NOT search for a better model. It attacks the existing
Silver / cross-instrument H4 finding with the tests that could kill it:

    A1  non-overlap sub-sampling   (overlapping labels)
    A2  Newey-West / HAC           (serial correlation)
    A3  moving-block bootstrap     (non-parametric CI)
    A4  roll-window exclusion      (continuous-contract contamination)
    B1  RV-only quantile regression(volatility-scaling benchmark)
    B2  CAViaR-SAV                 (classical dynamic quantile benchmark)
    B3  dynamic quantile test      (residual tail structure)
    C1  F1 vs F1+VOL               (independent RV increment)
    C2  observation-level width    (not decile monotonicity)
    D1  HAR future-RV              (opportunity model)

Only H4. No hyperparameter search, no new learners, no backtest,
no SMC / Momentum / Oracle.

Performance contract
--------------------
All heavy numeric work is vectorized with NumPy / pandas:
cumulative sums, searchsorted, boolean masks, broadcasting,
sliding_window_view and lfilter. Python loops exist only over the small
experiment dimensions (instrument x scenario x model x quantile x fold).
No per-bar Python loop anywhere.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import chi2

from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LinearRegression,
    QuantileRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )

from research.fit_quantile_v2_models import (  # noqa: E402
    make_model,
    make_folds,
    interval_score,
    safe_spearman,
)


# ============================================================
# Paths
# ============================================================

PANEL_DIR = (
    ROOT
    / "research"
    / "exports"
    / "quantile_v2_robustness_data"
)

ROLL_DIR = (
    ROOT
    / "research"
    / "exports"
    / "quantile_v3_roll_audit"
)

RAW_INDEX = (
    ROOT
    / "research"
    / "robustness_data"
    / "raw"
    / "download_index.json"
)

OUT = (
    ROOT
    / "research"
    / "exports"
    / "quantile_v3_validation"
)


# ============================================================
# Locked experiment design
# ============================================================

H = 4

BOOTSTRAPS = 1000

BLOCK_LENGTH = 16

RNG_SEED = 20260905

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

F1 = [
    "feat_15m_ret_1",
    "feat_15m_ret_4",
    "feat_15m_ret_8",
    "feat_15m_ret_16",
    "feat_15m_location_32",
    "feat_time_bars_since_segment_start",
    "feat_time_after_long_gap",
]

F1_VOL = F1 + [
    "feat_5m_1h_rv",
    "feat_5m_rv_rate_ratio_1h_4h",
]

STATIC_MODELS = {
    "GBR_F1": F1,
    "GBR_F1_VOL": F1_VOL,
    "RV_ONLY_QR": None,
}

CAVIAR_MODEL = "CAVIAR_SAV"

QUANTILES = (
    0.10,
    0.50,
    0.90,
)

CAVIAR_QUANTILES = (
    0.10,
    0.90,
)

SCENARIOS = (
    "FULL",
    "NO_ROLL_3D",
)

ROLL_WINDOW_DAYS = 3

# Roll-window exclusion removes 6%-24% of rows by design.
# The floor is set so the sensitivity run stays estimable
# while still failing loudly on catastrophic filtering.
MIN_FILTERED_TRAIN_ROWS = 600

MIN_FILTERED_TEST_ROWS = 150


# ============================================================
# Vectorized statistics
# ============================================================

def pinball_array(
    y,
    q,
    tau,
):

    y = np.asarray(
        y,
        dtype=float,
    )

    q = np.asarray(
        q,
        dtype=float,
    )

    e = y - q

    return np.where(
        e >= 0,
        tau * e,
        (tau - 1.0) * e,
    )


def newey_west_mean_test(
    values,
    *,
    lag=3,
):
    """
    H4 overlapping forecast:
    default HAC lag = H-1 = 3.
    """

    x = np.asarray(
        values,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    n = len(x)

    if n < 30:
        return (
            np.nan,
            np.nan,
        )

    mean = float(
        x.mean()
    )

    demeaned = (
        x - mean
    )

    gamma0 = (
        demeaned
        @ demeaned
        / n
    )

    long_var = gamma0

    for k in range(
        1,
        lag + 1,
    ):

        weight = (
            1.0
            -
            k
            /
            (lag + 1.0)
        )

        gamma = (
            demeaned[k:]
            @
            demeaned[:-k]
            / n
        )

        long_var += (
            2.0
            * weight
            * gamma
        )

    se = np.sqrt(
        max(
            long_var,
            0.0,
        )
        / n
    )

    if se <= 0:
        return (
            np.nan,
            np.nan,
        )

    t_stat = (
        mean / se
    )

    return (
        mean,
        float(t_stat),
    )


def circular_block_indices(
    n,
    *,
    b,
    block_length,
    rng,
):
    """
    Vectorized moving/circular block bootstrap.
    """

    blocks = int(
        np.ceil(
            n / block_length
        )
    )

    starts = rng.integers(
        0,
        n,
        size=(
            b,
            blocks,
        ),
    )

    offsets = np.arange(
        block_length
    )

    idx = (
        starts[
            :,
            :,
            None,
        ]
        +
        offsets[
            None,
            None,
            :,
        ]
    ) % n

    return (
        idx.reshape(
            b,
            -1,
        )[
            :,
            :n,
        ]
    )


def bootstrap_mean_ci(
    x,
    *,
    rng,
):

    x = np.asarray(
        x,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    idx = (
        circular_block_indices(
            len(x),
            b=BOOTSTRAPS,
            block_length=(
                BLOCK_LENGTH
            ),
            rng=rng,
        )
    )

    boot_mean = (
        x[idx]
        .mean(
            axis=1
        )
    )

    ci_low, ci_high = (
        np.quantile(
            boot_mean,
            [
                0.025,
                0.975,
            ],
        )
    )

    observed = float(
        x.mean()
    )

    centered = (
        x - observed
    )

    boot_null = (
        centered[idx]
        .mean(
            axis=1
        )
    )

    p_one_sided = (
        1.0
        +
        np.sum(
            boot_null
            >= observed
        )
    ) / (
        BOOTSTRAPS
        + 1.0
    )

    return {
        "mean": observed,
        "bootstrap_ci_low": float(
            ci_low
        ),
        "bootstrap_ci_high": float(
            ci_high
        ),
        "bootstrap_p_positive": float(
            p_one_sided
        ),
    }


def bootstrap_spearman_ci(
    x,
    y,
    *,
    rng,
):
    """
    Spearman = Pearson correlation of ranks.
    Rank once, bootstrap vectorized.
    """

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

    a = (
        a[valid]
        .rank(
            method="average"
        )
        .to_numpy(
            dtype=float
        )
    )

    b = (
        b[valid]
        .rank(
            method="average"
        )
        .to_numpy(
            dtype=float
        )
    )

    a = (
        a - a.mean()
    )

    b = (
        b - b.mean()
    )

    observed = float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )

    idx = (
        circular_block_indices(
            len(a),
            b=BOOTSTRAPS,
            block_length=(
                BLOCK_LENGTH
            ),
            rng=rng,
        )
    )

    aa = a[idx]
    bb = b[idx]

    aa = (
        aa
        -
        aa.mean(
            axis=1,
            keepdims=True,
        )
    )

    bb = (
        bb
        -
        bb.mean(
            axis=1,
            keepdims=True,
        )
    )

    numerator = (
        aa * bb
    ).sum(
        axis=1
    )

    denominator = np.sqrt(
        (
            aa * aa
        ).sum(
            axis=1
        )
        *
        (
            bb * bb
        ).sum(
            axis=1
        )
    )

    boot = (
        numerator
        /
        denominator
    )

    lo, hi = np.nanquantile(
        boot,
        [
            0.025,
            0.975,
        ],
    )

    return {
        "spearman": observed,
        "bootstrap_ci_low": float(
            lo
        ),
        "bootstrap_ci_high": float(
            hi
        ),
    }


def dq_test(
    y,
    q_pred,
    *,
    tau,
    lags=2,
):

    y = np.asarray(
        y,
        dtype=float,
    )

    q_pred = np.asarray(
        q_pred,
        dtype=float,
    )

    hit = (
        (
            y <= q_pred
        ).astype(float)
        - tau
    )

    if len(hit) <= (
        lags + 10
    ):
        return (
            np.nan,
            np.nan,
        )

    windows = (
        np.lib
        .stride_tricks
        .sliding_window_view(
            hit,
            lags + 1,
        )
    )

    current = (
        windows[
            :,
            -1
        ]
    )

    lag_matrix = (
        windows[
            :,
            :-1
        ][
            :,
            ::-1
        ]
    )

    q_current = (
        q_pred[
            lags:
        ]
    )

    X = np.column_stack(
        [
            np.ones(
                len(current)
            ),
            lag_matrix,
            q_current,
        ]
    )

    beta = (
        np.linalg.pinv(
            X.T @ X
        )
        @
        X.T
        @
        current
    )

    fitted = (
        X @ beta
    )

    stat = float(
        fitted @ fitted
        /
        (
            tau
            *
            (1.0 - tau)
        )
    )

    df = int(
        np.linalg.matrix_rank(
            X
        )
    )

    p_value = float(
        chi2.sf(
            stat,
            df,
        )
    )

    return (
        stat,
        p_value,
    )


# ============================================================
# RV-only quantile benchmark
# ============================================================

def make_rv_only_qr(
    tau,
):

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
                    quantile=tau,
                    alpha=0.0,
                    solver="highs",
                ),
            ),
        ]
    )


def rv_only_X(
    frame,
):

    return pd.DataFrame(
        {
            "sqrt_rv_1h": np.sqrt(
                np.maximum(
                    frame[
                        "feat_5m_1h_rv"
                    ].to_numpy(
                        dtype=float
                    ),
                    0.0,
                )
            )
        },
        index=frame.index,
    )


# ============================================================
# CAViaR-SAV via lfilter (C-level recursion)
# ============================================================

def caviar_path(
    beta,
    shock,
    q_init,
):
    """
    q_t = b0 + b1*q_{t-1} + b2*shock_t

    scipy.signal.lfilter performs
    the recursion in compiled code.
    """

    b0, b1, b2 = beta

    shock = np.asarray(
        shock,
        dtype=float,
    )

    innovation = (
        b0
        +
        b2
        * shock
    )

    path, _state = lfilter(
        [1.0],
        [
            1.0,
            -b1,
        ],
        innovation,
        zi=[
            b1 * q_init
        ],
    )

    return path


def fit_caviar_sav(
    y_train,
    shock_train,
    *,
    tau,
):

    y_train = np.asarray(
        y_train,
        dtype=float,
    )

    shock_train = np.asarray(
        shock_train,
        dtype=float,
    )

    q_init = float(
        np.quantile(
            y_train,
            tau,
        )
    )

    scale = max(
        float(
            np.std(
                y_train
            )
        ),
        1e-6,
    )

    sign = (
        -1.0
        if tau < 0.5
        else 1.0
    )

    starts = [
        np.array(
            [
                (
                    1.0 - 0.80
                )
                * q_init,
                0.80,
                0.0,
            ]
        ),

        np.array(
            [
                (
                    1.0 - 0.50
                )
                * q_init,
                0.50,
                sign * 0.10,
            ]
        ),

        np.array(
            [
                (
                    1.0 - 0.95
                )
                * q_init,
                0.95,
                sign * 0.25,
            ]
        ),
    ]

    bounds = [
        (
            -10.0 * scale,
            10.0 * scale,
        ),
        (
            0.0,
            0.995,
        ),
        (
            -5.0,
            5.0,
        ),
    ]

    def objective(
        beta,
    ):

        q = caviar_path(
            beta,
            shock_train,
            q_init,
        )

        return float(
            pinball_array(
                y_train,
                q,
                tau,
            ).mean()
        )

    best = None

    for start in starts:

        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 500,
            },
        )

        if (
            result.success
            and
            (
                best is None
                or
                result.fun
                <
                best.fun
            )
        ):
            best = result

    if best is None:
        raise RuntimeError(
            f"CAViaR failed tau={tau}"
        )

    train_path = caviar_path(
        best.x,
        shock_train,
        q_init,
    )

    return (
        best.x,
        float(
            train_path[-1]
        ),
    )


def forecast_caviar(
    beta,
    q_previous,
    future_shock,
):

    b0, b1, b2 = beta

    innovation = (
        b0
        +
        b2
        * np.asarray(
            future_shock,
            dtype=float,
        )
    )

    path, _ = lfilter(
        [1.0],
        [
            1.0,
            -b1,
        ],
        innovation,
        zi=[
            b1 * q_previous
        ],
    )

    return path


# ============================================================
# Roll mask (fully vectorized)
# ============================================================

def build_roll_mask(
    decision_time,
    roll_dates,
    *,
    days=3,
):

    dates = (
        pd.to_datetime(
            decision_time
        )
        .values
        .astype(
            "datetime64[D]"
        )
        .astype(
            np.int64
        )
    )

    rolls = (
        pd.to_datetime(
            roll_dates
        )
        .values
        .astype(
            "datetime64[D]"
        )
        .astype(
            np.int64
        )
    )

    if len(rolls) == 0:
        return np.zeros(
            len(dates),
            dtype=bool,
        )

    # rows ~3300, roll events only a few:
    # matrix is tiny.
    distance = np.abs(
        dates[
            :,
            None
        ]
        -
        rolls[
            None,
            :,
        ]
    )

    return (
        distance.min(
            axis=1
        )
        <= days
    )


# ============================================================
# Vectorized realized-variance arrays
# ============================================================

def build_rv_arrays(
    five,
    decision_time,
):

    five = (
        five
        .sort_values(
            "datetime_ns"
        )
        .reset_index(
            drop=True
        )
    )

    close = five[
        "close"
    ].to_numpy(
        dtype=float
    )

    times = five[
        "datetime_ns"
    ].to_numpy(
        dtype=np.int64
    )

    ret = np.full(
        len(close),
        np.nan,
    )

    ret[1:] = np.log(
        close[1:]
        /
        close[:-1]
    )

    r2 = np.nan_to_num(
        ret * ret,
        nan=0.0,
    )

    cs = np.concatenate(
        [
            [0.0],
            np.cumsum(
                r2
            ),
        ]
    )

    # `datetime_ns` is UTC epoch ns while `datetime` is
    # naive Asia/Shanghai. The offset is constant (no DST),
    # and is derived from the file itself rather than hardcoded
    # so the decision-time search stays in the same convention.
    local_ns = (
        pd.to_datetime(
            five["datetime"]
        )
        .values
        .astype(
            "datetime64[ns]"
        )
        .astype(
            np.int64
        )
    )

    offset_ns = int(
        np.median(
            times - local_ns
        )
    )

    if not (
        np.all(
            times - local_ns
            == offset_ns
        )
    ):
        raise RuntimeError(
            "Non-constant timezone offset in raw 5m file"
        )

    decision_ns = (
        pd.to_datetime(
            decision_time
        )
        .values
        .astype(
            "datetime64[ns]"
        )
        .astype(
            np.int64
        )
        + offset_ns
    )

    # First 5m bar beginning at decision.
    pos = np.searchsorted(
        times,
        decision_ns,
        side="left",
    )

    def trailing(
        n,
    ):

        out = np.full(
            len(pos),
            np.nan,
        )

        valid = (
            pos >= n
        )

        p = pos[
            valid
        ]

        out[
            valid
        ] = (
            cs[p]
            -
            cs[
                p - n
            ]
        )

        return out

    def future(
        n,
    ):

        out = np.full(
            len(pos),
            np.nan,
        )

        valid = (
            pos + n
            <= len(
                r2
            )
        )

        p = pos[
            valid
        ]

        out[
            valid
        ] = (
            cs[
                p + n
            ]
            -
            cs[p]
        )

        return out

    return {
        "rv_1h": trailing(
            12
        ),
        "rv_4h": trailing(
            48
        ),
        "rv_8h": trailing(
            96
        ),
        "future_rv_1h": future(
            12
        ),
    }


# ============================================================
# Static model OOS
# ============================================================

def fit_static_oos(
    frame,
    *,
    model_name,
    tau,
    feature_cols=None,
    allowed_mask=None,
):

    y = (
        frame[
            "target_raw_return_h4"
        ]
        .to_numpy(
            dtype=float
        )
    )

    if allowed_mask is None:
        allowed_mask = np.ones(
            len(frame),
            dtype=bool,
        )

    if model_name == "RV_ONLY_QR":

        X = rv_only_X(
            frame
        )

    else:

        X = frame[
            feature_cols
        ]

    pred = np.full(
        len(frame),
        np.nan,
    )

    baseline = np.full(
        len(frame),
        np.nan,
    )

    fold_id = np.full(
        len(frame),
        -1,
        dtype=int,
    )

    folds = make_folds(
        len(frame),
        horizon=4,
    )

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

        train_idx = train_idx[
            allowed_mask[
                train_idx
            ]
        ]

        test_idx = test_idx[
            allowed_mask[
                test_idx
            ]
        ]

        if len(
            train_idx
        ) < MIN_FILTERED_TRAIN_ROWS:
            raise RuntimeError(
                "Too few filtered train rows: "
                f"{len(train_idx)}"
            )

        if len(
            test_idx
        ) < MIN_FILTERED_TEST_ROWS:
            raise RuntimeError(
                "Too few filtered test rows: "
                f"{len(test_idx)}"
            )

        if (
            model_name
            == "RV_ONLY_QR"
        ):

            model = make_rv_only_qr(
                tau
            )

        else:

            model = make_model(
                "gbr_quantile",
                tau,
            )

        model.fit(
            X.iloc[
                train_idx
            ],
            y[
                train_idx
            ],
        )

        pred[
            test_idx
        ] = model.predict(
            X.iloc[
                test_idx
            ]
        )

        b = float(
            np.quantile(
                y[
                    train_idx
                ],
                tau,
            )
        )

        baseline[
            test_idx
        ] = b

        fold_id[
            test_idx
        ] = (
            fold["fold"]
        )

    valid = (
        fold_id >= 0
    )

    return pd.DataFrame(
        {
            "row_id": np.arange(
                len(frame)
            )[
                valid
            ],
            "fold": fold_id[
                valid
            ],
            "y": y[
                valid
            ],
            "pred": pred[
                valid
            ],
            "baseline": (
                baseline[
                    valid
                ]
            ),
        }
    )


def fit_caviar_oos(
    frame,
    *,
    tau,
    allowed_mask=None,
):

    y = (
        frame[
            "target_raw_return_h4"
        ]
        .to_numpy(
            dtype=float
        )
    )

    shock = np.abs(
        frame[
            "feat_15m_ret_1"
        ]
        .to_numpy(
            dtype=float
        )
    )

    if allowed_mask is None:
        allowed_mask = np.ones(
            len(frame),
            dtype=bool,
        )

    pred = np.full(
        len(frame),
        np.nan,
    )

    baseline = np.full(
        len(frame),
        np.nan,
    )

    fold_id = np.full(
        len(frame),
        -1,
        dtype=int,
    )

    folds = make_folds(
        len(frame),
        horizon=4,
    )

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

        train_idx = train_idx[
            allowed_mask[
                train_idx
            ]
        ]

        test_idx = test_idx[
            allowed_mask[
                test_idx
            ]
        ]

        if len(
            train_idx
        ) < MIN_FILTERED_TRAIN_ROWS:
            raise RuntimeError(
                "Too few filtered train rows"
            )

        if len(
            test_idx
        ) < MIN_FILTERED_TEST_ROWS:
            raise RuntimeError(
                "Too few filtered test rows"
            )

        beta, q_last = (
            fit_caviar_sav(
                y[
                    train_idx
                ],
                shock[
                    train_idx
                ],
                tau=tau,
            )
        )

        pred[
            test_idx
        ] = forecast_caviar(
            beta,
            q_last,
            shock[
                test_idx
            ],
        )

        baseline[
            test_idx
        ] = float(
            np.quantile(
                y[
                    train_idx
                ],
                tau,
            )
        )

        fold_id[
            test_idx
        ] = (
            fold["fold"]
        )

    valid = (
        fold_id >= 0
    )

    return pd.DataFrame(
        {
            "row_id": np.arange(
                len(frame)
            )[
                valid
            ],
            "fold": fold_id[
                valid
            ],
            "y": y[
                valid
            ],
            "pred": pred[
                valid
            ],
            "baseline": (
                baseline[
                    valid
                ]
            ),
        }
    )


# ============================================================
# Quantile evaluation
# ============================================================

def evaluate_quantile_prediction(
    pred,
    *,
    tau,
    rng,
):

    y = pred[
        "y"
    ].to_numpy(
        float
    )

    model_q = pred[
        "pred"
    ].to_numpy(
        float
    )

    baseline_q = pred[
        "baseline"
    ].to_numpy(
        float
    )

    model_loss = pinball_array(
        y,
        model_q,
        tau,
    )

    baseline_loss = pinball_array(
        y,
        baseline_q,
        tau,
    )

    loss_diff = (
        baseline_loss
        -
        model_loss
    )

    overall_skill = (
        1.0
        -
        model_loss.mean()
        /
        baseline_loss.mean()
    )

    hac_mean, hac_t = (
        newey_west_mean_test(
            loss_diff,
            lag=3,
        )
    )

    boot = bootstrap_mean_ci(
        loss_diff,
        rng=rng,
    )

    offset_rows = []

    row_id = pred[
        "row_id"
    ].to_numpy(
        int
    )

    for offset in range(4):

        mask = (
            row_id % 4
            ==
            offset
        )

        ml = model_loss[
            mask
        ]

        bl = baseline_loss[
            mask
        ]

        skill = (
            1.0
            -
            ml.mean()
            /
            bl.mean()
        )

        dq_stat, dq_p = dq_test(
            y[
                mask
            ],
            model_q[
                mask
            ],
            tau=tau,
            lags=2,
        )

        offset_rows.append(
            {
                "offset": offset,
                "n": int(
                    mask.sum()
                ),
                "pinball_skill": float(
                    skill
                ),
                "calibration": float(
                    (
                        y[
                            mask
                        ]
                        <=
                        model_q[
                            mask
                        ]
                    ).mean()
                ),
                "dq_stat": (
                    dq_stat
                ),
                "dq_p_value": (
                    dq_p
                ),
            }
        )

    return (
        {
            "oos_rows": int(
                len(y)
            ),
            "pinball_skill": float(
                overall_skill
            ),
            "hac_loss_diff_mean": (
                hac_mean
            ),
            "hac_t_stat": (
                hac_t
            ),
            **boot,
        },
        pd.DataFrame(
            offset_rows
        ),
    )


def interval_stats(
    y,
    q10,
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
        q10 > q90
    )

    ordered = (
        ~crossing
    )

    if not ordered.any():

        return {
            "crossing_rate": float(
                crossing.mean()
            ),
            "interval_80_coverage": (
                np.nan
            ),
            "interval_score": (
                np.nan
            ),
            "baseline_interval_score": (
                np.nan
            ),
            "interval_score_skill": (
                np.nan
            ),
        }

    yy = y[
        ordered
    ]

    l = q10[
        ordered
    ]

    u = q90[
        ordered
    ]

    bl = b10[
        ordered
    ]

    bu = b90[
        ordered
    ]

    model_score = float(
        np.mean(
            interval_score(
                yy,
                l,
                u,
                alpha=0.20,
            )
        )
    )

    base_score = float(
        np.mean(
            interval_score(
                yy,
                bl,
                bu,
                alpha=0.20,
            )
        )
    )

    return {
        "crossing_rate": float(
            crossing.mean()
        ),
        "interval_80_coverage": float(
            (
                (yy >= l)
                &
                (yy <= u)
            ).mean()
        ),
        "interval_score": (
            model_score
        ),
        "baseline_interval_score": (
            base_score
        ),
        "interval_score_skill": (
            1.0
            -
            model_score
            /
            base_score
            if base_score > 0
            else np.nan
        ),
    }


# ============================================================
# Observation-level width test
# ============================================================

def evaluate_width(
    wide,
    *,
    rng,
):

    x = wide.dropna(
        subset=[
            "pred_q10",
            "pred_q90",
            "path_range",
            "y",
        ]
    ).copy()

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

    x = x[
        x[
            "pred_width"
        ] >= 0
    ]

    path_result = (
        bootstrap_spearman_ci(
            x[
                "pred_width"
            ],
            x[
                "path_range"
            ],
            rng=rng,
        )
    )

    abs_result = (
        bootstrap_spearman_ci(
            x[
                "pred_width"
            ],
            x[
                "y"
            ].abs(),
            rng=rng,
        )
    )

    return {
        "n": int(
            len(x)
        ),
        "width_path_spearman": (
            path_result[
                "spearman"
            ]
        ),
        "width_path_ci_low": (
            path_result[
                "bootstrap_ci_low"
            ]
        ),
        "width_path_ci_high": (
            path_result[
                "bootstrap_ci_high"
            ]
        ),
        "width_abs_return_spearman": (
            abs_result[
                "spearman"
            ]
        ),
        "width_abs_ci_low": (
            abs_result[
                "bootstrap_ci_low"
            ]
        ),
        "width_abs_ci_high": (
            abs_result[
                "bootstrap_ci_high"
            ]
        ),
    }


# ============================================================
# HAR opportunity model
# ============================================================

def run_har_oos(
    frame,
):

    eps = 1e-12

    valid = (
        frame[
            [
                "rv_1h",
                "rv_4h",
                "rv_8h",
                "future_rv_1h",
            ]
        ]
        .notna()
        .all(
            axis=1
        )
    )

    x = frame.loc[
        valid,
        [
            "rv_1h",
            "rv_4h",
            "rv_8h",
        ],
    ].to_numpy(
        float
    )

    y = np.log(
        frame.loc[
            valid,
            "future_rv_1h",
        ].to_numpy(
            float
        )
        +
        eps
    )

    X = np.log(
        x + eps
    )

    current_log_rv = (
        X[:, 0]
    )

    path_range = (
        frame.loc[
            valid,
            "path_range",
        ].to_numpy(
            float
        )
    )

    pred = np.full(
        len(y),
        np.nan,
    )

    persistence = np.full(
        len(y),
        np.nan,
    )

    folds = make_folds(
        len(y),
        horizon=4,
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

        model = LinearRegression()

        model.fit(
            X[tr],
            y[tr],
        )

        pred[
            te
        ] = model.predict(
            X[te]
        )

        persistence[
            te
        ] = (
            current_log_rv[
                te
            ]
        )

    oos = np.isfinite(
        pred
    )

    mse_model = np.mean(
        (
            y[oos]
            -
            pred[oos]
        ) ** 2
    )

    mse_persist = np.mean(
        (
            y[oos]
            -
            persistence[oos]
        ) ** 2
    )

    return {
        "oos_rows": int(
            oos.sum()
        ),
        "mse_skill_vs_persistence": float(
            1.0
            -
            mse_model
            /
            mse_persist
        ),
        "future_rv_spearman": (
            safe_spearman(
                pred[oos],
                y[oos],
            )
        ),
        "path_range_spearman": (
            safe_spearman(
                pred[oos],
                path_range[
                    oos
                ],
            )
        ),
    }


# ============================================================
# Loading
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


def load_h4_frame(
    instrument,
):

    panel = pd.read_csv(
        PANEL_DIR
        / f"{instrument}_panel.csv",
        parse_dates=[
            "meta_base_bar_time",
            "meta_decision_time",
        ],
        low_memory=False,
    )

    valid = (
        panel[
            "target_raw_return_h4"
        ].notna()
        &
        panel[
            "target_long_mfe_h4"
        ].notna()
        &
        panel[
            "target_short_mfe_h4"
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
            "target_long_mfe_h4"
        ]
        +
        frame[
            "target_short_mfe_h4"
        ]
    )

    return (
        panel,
        frame,
    )


def attach_rv(
    panel,
    frame,
    instrument,
):

    index = json.loads(
        RAW_INDEX.read_text(
            encoding="utf-8"
        )
    )

    five = pd.read_csv(
        ROOT
        / index[
            instrument
        ][
            "five_minute"
        ][
            "path"
        ],
        parse_dates=[
            "datetime"
        ],
        low_memory=False,
    )

    arrays = build_rv_arrays(
        five,
        frame[
            "meta_decision_time"
        ],
    )

    reference = (
        frame[
            "feat_5m_1h_rv"
        ]
        .to_numpy(
            dtype=float
        )
    )

    if not np.allclose(
        arrays["rv_1h"],
        reference,
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    ):
        raise RuntimeError(
            f"{instrument}: vectorized rv_1h "
            "does not match panel "
            "feat_5m_1h_rv"
        )

    for key, value in (
        arrays.items()
    ):

        frame[
            key
        ] = value

    return frame


def load_roll_dates(
    instrument,
):

    events = pd.read_csv(
        ROLL_DIR
        / "roll_events.csv",
        parse_dates=[
            "date"
        ],
    )

    events = events[
        events[
            "instrument"
        ]
        == instrument
    ]

    return (
        events[
            "date"
        ]
        .dropna()
        .sort_values()
        .to_numpy()
    )


# ============================================================
# Main
# ============================================================

def main():

    prepare_output_dir()

    rng = np.random.default_rng(
        RNG_SEED
    )

    benchmark_rows = []
    nonoverlap_rows = []
    hac_rows = []
    dq_rows = []
    width_rows = []
    har_rows = []
    roll_exposure_rows = []
    prediction_parts = []

    min_train_seen = None
    min_test_seen = None

    rv_reference_check = {}

    for instrument in INSTRUMENTS:

        print(
            "=" * 72
        )

        print(
            f"INSTRUMENT {instrument}"
        )

        print(
            "=" * 72
        )

        panel, frame = (
            load_h4_frame(
                instrument
            )
        )

        frame = attach_rv(
            panel,
            frame,
            instrument,
        )

        rv_reference_check[
            instrument
        ] = True

        roll_dates = (
            load_roll_dates(
                instrument
            )
        )

        roll_mask = build_roll_mask(
            frame[
                "meta_decision_time"
            ],
            roll_dates,
            days=ROLL_WINDOW_DAYS,
        )

        # ====================================================
        # D1: HAR opportunity model
        # ====================================================

        har = run_har_oos(
            frame
        )

        har_rows.append(
            {
                "instrument": (
                    instrument
                ),
                **har,
            }
        )

        # ====================================================
        # Scenarios
        # ====================================================

        scenario_preds = {}

        for scenario in SCENARIOS:

            if scenario == "FULL":

                allowed = np.ones(
                    len(frame),
                    dtype=bool,
                )

            else:

                allowed = (
                    ~roll_mask
                )

            for (
                model_name,
                feature_cols,
            ) in (
                STATIC_MODELS.items()
            ):

                for tau in QUANTILES:

                    pred = (
                        fit_static_oos(
                            frame,
                            model_name=(
                                model_name
                            ),
                            tau=tau,
                            feature_cols=(
                                feature_cols
                            ),
                            allowed_mask=(
                                allowed
                            ),
                        )
                    )

                    key = (
                        scenario,
                        model_name,
                    )

                    scenario_preds.setdefault(
                        key,
                        {},
                    )[
                        tau
                    ] = pred

            if scenario == "FULL":

                for tau in (
                    CAVIAR_QUANTILES
                ):

                    pred = (
                        fit_caviar_oos(
                            frame,
                            tau=tau,
                            allowed_mask=(
                                allowed
                            ),
                        )
                    )

                    key = (
                        scenario,
                        CAVIAR_MODEL,
                    )

                    scenario_preds.setdefault(
                        key,
                        {},
                    )[
                        tau
                    ] = pred

        # ====================================================
        # Roll exposure summary
        # ====================================================

        full_key = (
            "FULL",
            "GBR_F1",
        )

        oos_row_ids = (
            scenario_preds[
                full_key
            ][
                0.10
            ][
                "row_id"
            ]
            .to_numpy(
                int
            )
        )

        roll_exposure_rows.append(
            {
                "instrument": (
                    instrument
                ),
                "panel_rows": int(
                    len(panel)
                ),
                "h4_valid_rows": int(
                    len(frame)
                ),
                "roll_event_count": int(
                    len(
                        roll_dates
                    )
                ),
                "roll_dates": (
                    "|".join(
                        pd.to_datetime(
                            roll_dates
                        ).strftime(
                            "%Y-%m-%d"
                        )
                    )
                ),
                "excluded_rows": int(
                    roll_mask.sum()
                ),
                "excluded_row_share": float(
                    roll_mask.mean()
                ),
                "oos_rows_full": int(
                    len(
                        oos_row_ids
                    )
                ),
                "excluded_oos_rows": int(
                    roll_mask[
                        oos_row_ids
                    ].sum()
                ),
                "excluded_oos_share": float(
                    roll_mask[
                        oos_row_ids
                    ].mean()
                ),
            }
        )

        # ====================================================
        # Metrics
        # ====================================================

        for (
            scenario,
            model_name,
        ), by_tau in sorted(
            scenario_preds.items()
        ):

            # ------------------------------------------------
            # Wide frame for interval / width diagnostics
            # ------------------------------------------------

            wide = None

            if (
                0.10 in by_tau
                and
                0.90 in by_tau
            ):

                left = by_tau[
                    0.10
                ]

                right = by_tau[
                    0.90
                ]

                wide = left.merge(
                    right[
                        [
                            "row_id",
                            "pred",
                            "baseline",
                        ]
                    ].rename(
                        columns={
                            "pred": (
                                "pred_q90"
                            ),
                            "baseline": (
                                "baseline_q90"
                            ),
                        }
                    ),
                    on="row_id",
                    how="inner",
                ).rename(
                    columns={
                        "pred": (
                            "pred_q10"
                        ),
                        "baseline": (
                            "baseline_q10"
                        ),
                    }
                )

                wide[
                    "path_range"
                ] = (
                    frame.loc[
                        wide[
                            "row_id"
                        ]
                        .to_numpy(
                            int
                        ),
                        "path_range",
                    ]
                    .to_numpy(
                        float
                    )
                )

                istats = (
                    interval_stats(
                        wide[
                            "y"
                        ],
                        wide[
                            "pred_q10"
                        ],
                        wide[
                            "pred_q90"
                        ],
                        wide[
                            "baseline_q10"
                        ],
                        wide[
                            "baseline_q90"
                        ],
                    )
                )

                width_row = (
                    evaluate_width(
                        wide,
                        rng=rng,
                    )
                )

                width_rows.append(
                    {
                        "instrument": (
                            instrument
                        ),
                        "scenario": (
                            scenario
                        ),
                        "model": (
                            model_name
                        ),
                        **width_row,
                    }
                )

            for tau, pred in sorted(
                by_tau.items()
            ):

                (
                    metrics,
                    offsets,
                ) = (
                    evaluate_quantile_prediction(
                        pred,
                        tau=tau,
                        rng=rng,
                    )
                )

                calibration = float(
                    (
                        pred[
                            "y"
                        ].to_numpy()
                        <=
                        pred[
                            "pred"
                        ].to_numpy()
                    ).mean()
                )

                row = {
                    "instrument": (
                        instrument
                    ),
                    "scenario": (
                        scenario
                    ),
                    "model": (
                        model_name
                    ),
                    "quantile": tau,
                    "oos_rows": (
                        metrics[
                            "oos_rows"
                        ]
                    ),
                    "pinball_skill": (
                        metrics[
                            "pinball_skill"
                        ]
                    ),
                    "calibration": (
                        calibration
                    ),
                    "calibration_error": (
                        calibration
                        - tau
                    ),
                }

                if wide is not None:

                    row.update(
                        istats
                    )

                benchmark_rows.append(
                    row
                )

                hac_rows.append(
                    {
                        "instrument": (
                            instrument
                        ),
                        "scenario": (
                            scenario
                        ),
                        "model": (
                            model_name
                        ),
                        "quantile": tau,
                        **metrics,
                    }
                )

                positive_share = float(
                    (
                        offsets[
                            "pinball_skill"
                        ]
                        > 0
                    ).mean()
                )

                median_skill = float(
                    offsets[
                        "pinball_skill"
                    ].median()
                )

                for (
                    _,
                    offset_row,
                ) in (
                    offsets.iterrows()
                ):

                    nonoverlap_rows.append(
                        {
                            "instrument": (
                                instrument
                            ),
                            "scenario": (
                                scenario
                            ),
                            "model": (
                                model_name
                            ),
                            "quantile": tau,
                            **offset_row.to_dict(),
                            "positive_offset_share": (
                                positive_share
                            ),
                            "median_offset_skill": (
                                median_skill
                            ),
                        }
                    )

                    dq_rows.append(
                        {
                            "instrument": (
                                instrument
                            ),
                            "scenario": (
                                scenario
                            ),
                            "model": (
                                model_name
                            ),
                            "quantile": tau,
                            "offset": int(
                                offset_row[
                                    "offset"
                                ]
                            ),
                            "n": int(
                                offset_row[
                                    "n"
                                ]
                            ),
                            "dq_stat": (
                                offset_row[
                                    "dq_stat"
                                ]
                            ),
                            "dq_p_value": (
                                offset_row[
                                    "dq_p_value"
                                ]
                            ),
                        }
                    )

                prediction_parts.append(
                    pred.assign(
                        instrument=(
                            instrument
                        ),
                        scenario=(
                            scenario
                        ),
                        model=(
                            model_name
                        ),
                        quantile=tau,
                    )
                )

    # ========================================================
    # Assemble tables
    # ========================================================

    benchmark = pd.DataFrame(
        benchmark_rows
    )

    nonoverlap = pd.DataFrame(
        nonoverlap_rows
    )

    hac = pd.DataFrame(
        hac_rows
    )

    dq = pd.DataFrame(
        dq_rows
    )

    width = pd.DataFrame(
        width_rows
    )

    har = pd.DataFrame(
        har_rows
    )

    roll_exposure = pd.DataFrame(
        roll_exposure_rows
    )

    predictions = pd.concat(
        prediction_parts,
        ignore_index=True,
    )[
        [
            "instrument",
            "scenario",
            "model",
            "quantile",
            "row_id",
            "fold",
            "y",
            "pred",
            "baseline",
        ]
    ]

    # ========================================================
    # Roll sensitivity: FULL vs NO_ROLL_3D
    # ========================================================

    roll_delta = (
        benchmark
        .pivot_table(
            index=[
                "instrument",
                "model",
                "quantile",
            ],
            columns="scenario",
            values=[
                "pinball_skill",
                "oos_rows",
            ],
        )
        .reset_index()
    )

    roll_delta.columns = [
        (
            f"{a}_{b}"
            if b != ""
            else a
        )
        for a, b in (
            roll_delta.columns
        )
    ]

    roll_delta[
        "skill_delta_no_roll_minus_full"
    ] = (
        roll_delta[
            "pinball_skill_NO_ROLL_3D"
        ]
        -
        roll_delta[
            "pinball_skill_FULL"
        ]
    )

    # ========================================================
    # Hard validation
    # ========================================================

    expected_benchmark = (
        len(INSTRUMENTS)
        * (
            len(STATIC_MODELS)
            * len(QUANTILES)
            * len(SCENARIOS)
            + len(
                CAVIAR_QUANTILES
            )
        )
    )

    if len(
        benchmark
    ) != expected_benchmark:

        raise RuntimeError(
            f"benchmark rows "
            f"{len(benchmark)} "
            f"!={expected_benchmark}"
        )

    if len(
        nonoverlap
    ) != expected_benchmark * 4:
        raise RuntimeError(
            "nonoverlap row count mismatch"
        )

    if len(
        hac
    ) != expected_benchmark:
        raise RuntimeError(
            "hac row count mismatch"
        )

    if len(
        dq
    ) != expected_benchmark * 4:
        raise RuntimeError(
            "dq row count mismatch"
        )

    expected_width = (
        len(INSTRUMENTS)
        * (
            len(STATIC_MODELS)
            * len(SCENARIOS)
            + 1
        )
    )

    if len(
        width
    ) != expected_width:
        raise RuntimeError(
            f"width rows "
            f"{len(width)} "
            f"!={expected_width}"
        )

    if len(
        har
    ) != len(
        INSTRUMENTS
    ):
        raise RuntimeError(
            "har row count mismatch"
        )

    if len(
        roll_exposure
    ) != len(
        INSTRUMENTS
    ):
        raise RuntimeError(
            "roll exposure row mismatch"
        )

    if not all(
        rv_reference_check.values()
    ):
        raise RuntimeError(
            "RV reference check incomplete"
        )

    if (
        len(
            rv_reference_check
        )
        != len(INSTRUMENTS)
    ):
        raise RuntimeError(
            "RV reference check incomplete"
        )

    numeric_outputs = [
        benchmark,
        nonoverlap,
        hac,
        dq,
        width,
        har,
        roll_exposure,
        roll_delta,
        predictions,
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
        "benchmark_quantile_metrics.csv": (
            benchmark
        ),
        "nonoverlap_metrics.csv": (
            nonoverlap
        ),
        "hac_bootstrap_metrics.csv": (
            hac
        ),
        "dq_metrics.csv": dq,
        "width_observation_metrics.csv": (
            width
        ),
        "roll_exclusion_metrics.csv": (
            roll_delta
        ),
        "roll_exposure_summary.csv": (
            roll_exposure
        ),
        "har_metrics.csv": har,
        "h4_oos_predictions.csv": (
            predictions
        ),
    }

    for name, df in (
        outputs.items()
    ):

        df.to_csv(
            OUT / name,
            index=False,
        )

    config = {
        "purpose": (
            "Validate H4 quantile forecasts against "
            "classical benchmarks and robustness tests."
        ),

        "horizon": H,

        "instruments": list(
            INSTRUMENTS
        ),

        "scenarios": list(
            SCENARIOS
        ),

        "static_models": {
            name: (
                None
                if cols is None
                else list(cols)
            )
            for name, cols
            in STATIC_MODELS.items()
        },

        "caviar_model": CAVIAR_MODEL,

        "quantiles": list(
            QUANTILES
        ),

        "caviar_quantiles": list(
            CAVIAR_QUANTILES
        ),

        "bootstraps": BOOTSTRAPS,

        "block_length": BLOCK_LENGTH,

        "rng_seed": RNG_SEED,

        "hac_lag": 3,

        "dq_lags": 2,

        "roll_window_days": (
            ROLL_WINDOW_DAYS
        ),

        "min_filtered_train_rows": (
            MIN_FILTERED_TRAIN_ROWS
        ),

        "min_filtered_test_rows": (
            MIN_FILTERED_TEST_ROWS
        ),

        "caviar_shock": (
            "abs(feat_15m_ret_1)"
        ),

        "no_hyperparameter_search": (
            True
        ),

        "no_new_learners": True,

        "no_backtest": True,

        "no_smc": True,
        "no_momentum": True,
        "no_oracle": True,

        "vectorized": True,

        "no_per_bar_python_loop": (
            True
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

        if size_mb > 100:

            raise RuntimeError(
                f"{path.name} >100MB"
            )

    validation = {
        "status": "PASS",

        "instrument_count": (
            len(INSTRUMENTS)
        ),

        "scenario_count": (
            len(SCENARIOS)
        ),

        "static_model_count": (
            len(STATIC_MODELS)
        ),

        "quantile_count": (
            len(QUANTILES)
        ),

        "benchmark_rows": int(
            len(benchmark)
        ),

        "nonoverlap_rows": int(
            len(nonoverlap)
        ),

        "hac_rows": int(
            len(hac)
        ),

        "dq_rows": int(
            len(dq)
        ),

        "width_rows": int(
            len(width)
        ),

        "har_rows": int(
            len(har)
        ),

        "roll_exposure_rows": int(
            len(roll_exposure)
        ),

        "roll_delta_rows": int(
            len(roll_delta)
        ),

        "prediction_rows": int(
            len(predictions)
        ),

        "rv_reference_allclose": (
            rv_reference_check
        ),

        "min_filtered_train_rows": (
            MIN_FILTERED_TRAIN_ROWS
        ),

        "min_filtered_test_rows": (
            MIN_FILTERED_TEST_ROWS
        ),

        "excluded_row_share_by_instrument": (
            {
                row[
                    "instrument"
                ]: round(
                    row[
                        "excluded_row_share"
                    ],
                    4,
                )
                for _, row
                in (
                    roll_exposure.iterrows()
                )
            }
        ),

        "no_hyperparameter_search": (
            True
        ),

        "no_new_learners": True,

        "no_backtest": True,

        "no_smc": True,
        "no_momentum": True,
        "no_oracle": True,

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
        "# Quantile V3 Validation Suite (H4)",
        "",
        "No hyperparameter search. No new learners.",
        "No backtest. No SMC / Momentum / Oracle.",
        "",
        "## Benchmark pinball skill (FULL)",
        "",
        "| instrument | model | Q10 | Q50 "
        "| Q90 | interval |",
        "|---|---|---:|---:|---:|---:|",
    ]

    full = benchmark[
        benchmark[
            "scenario"
        ]
        == "FULL"
    ]

    for (
        instrument,
        model_name,
    ), g in full.groupby(
        [
            "instrument",
            "model",
        ],
        observed=True,
    ):

        g = g.set_index(
            "quantile"
        )

        def val(
            q,
            col,
        ):

            if (
                q
                not in g.index
            ):
                return float(
                    "nan"
                )

            value = g.loc[
                q,
                col,
            ]

            if isinstance(
                value,
                pd.Series,
            ):
                value = (
                    value.iloc[
                        0
                    ]
                )

            return float(
                value
            )

        interval = (
            g[
                "interval_score_skill"
            ]
            .dropna()
            .iloc[0]
            if (
                "interval_score_skill"
                in g.columns
                and
                g[
                    "interval_score_skill"
                ]
                .notna()
                .any()
            )
            else float("nan")
        )

        lines.append(
            f"| {instrument} "
            f"| {model_name} "
            f"| {val(0.10, 'pinball_skill'):.5f} "
            f"| {val(0.50, 'pinball_skill'):.5f} "
            f"| {val(0.90, 'pinball_skill'):.5f} "
            f"| {interval:.5f} |"
        )

    lines += [
        "",
        "## Width vs future path range "
        "(observation level, FULL)",
        "",
        "| instrument | model | n | spearman "
        "| ci low | ci high |",
        "|---|---|---:|---:|---:|---:|",
    ]

    w = width[
        width[
            "scenario"
        ]
        == "FULL"
    ]

    for (
        _,
        row,
    ) in w.sort_values(
        [
            "instrument",
            "model",
        ]
    ).iterrows():

        lines.append(
            f"| {row['instrument']} "
            f"| {row['model']} "
            f"| {int(row['n'])} "
            f"| {row['width_path_spearman']:.4f} "
            f"| {row['width_path_ci_low']:.4f} "
            f"| {row['width_path_ci_high']:.4f} |"
        )

    lines += [
        "",
        "## HAR opportunity model",
        "",
        "| instrument | oos rows | mse skill "
        "| future-rv spearman | path-range "
        "spearman |",
        "|---|---:|---:|---:|---:|",
    ]

    for (
        _,
        row,
    ) in har.iterrows():

        lines.append(
            f"| {row['instrument']} "
            f"| {int(row['oos_rows'])} "
            f"| {row['mse_skill_vs_persistence']:.5f} "
            f"| {row['future_rv_spearman']:.4f} "
            f"| {row['path_range_spearman']:.4f} |"
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
        "QUANTILE_V3_VALIDATION_PASS"
    )


if __name__ == "__main__":
    main()
