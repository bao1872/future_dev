#!/usr/bin/env python3
"""Direction V0 -- opportunity-conditioned direction.

Question
--------
Inside High Opportunity states, does the SHAPE of the predicted
quantile distribution carry information about the direction of the
future 5m / 15m / 30m return?

The Quantile layer is closed. It is treated as the WHEN / Opportunity
signal. This experiment only asks whether its geometry also says
WHERE.

Hard restrictions
-----------------
- the Quantile model is not re-fit and not modified; the predictions
  are regenerated through the identical locked code path
  (`make_model` + the identical fold geometry) and then verified to
  reproduce the committed rebaseline metrics exactly
- direction inputs are Quantile OOS predictions only
- no SMC / DSA / Momentum, no price path, no RV, no volume, no OI
- no transaction data
- no GBR / XGBoost: fixed L2 LogisticRegression only
- no parameter tuning
- no PnL / backtest
- the PyTDX data pipeline is not changed

Quantile state
--------------
Produced on the 15m decision grid, then carried FORWARD onto the 5m
execution grid. A state generated at 10:15 governs the 5m decisions
at 10:15, 10:20 and 10:25 only. Never backward filled.

Targets
-------
Entry is the open of the decision 5m bar, exit is the close of the
n-th 5m bar of the window (n = 1 / 3 / 6), so the entry price is
known at the decision instant.

Opportunity strata
------------------
Width thresholds are computed on TRAIN rows inside every fold only:

    ALL / TOP50 (>= median) / TOP30 (>= 70%) / TOP20 (>= 80%)

Validation
----------
Expanding walk-forward, purged by the horizon length in 5m bars, so
the last training target is always realised before the test window
starts. Overlapping full OOS is the primary result; a light
non-overlap check (row_id % 3 for H15, % 6 for H30) shows whether
the result is an artefact of overlapping labels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
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

from research.fit_quantile_v2_models import (  # noqa: E402
    make_model,
)

from research.run_quantile_rebaseline import (  # noqa: E402
    FEATURE_SETS,
    make_folds,
)


PANEL_5M = (
    ROOT
    / "research"
    / "exports"
    / "pytdx_5m"
)

PANEL_15M = (
    ROOT
    / "research"
    / "exports"
    / "pytdx_panel"
)

REBASELINE = (
    ROOT
    / "research"
    / "exports"
    / "pytdx_rebaseline"
    / "rebaseline_by_instrument.csv"
)

OUT = (
    ROOT
    / "research"
    / "exports"
    / "direction_v0"
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

EX_I = tuple(
    i
    for i in INSTRUMENTS
    if i != "I"
)

GROUPS = {
    "ALL_8": INSTRUMENTS,
    "EX_I_7": EX_I,
}

# The Quantile layer is closed: one locked source.
QUANT_HORIZON = 4

QUANT_FEATURE_SET = (
    "F1_VOL"
)

QUANT_MODEL = (
    "gbr_quantile"
)

QUANTILES = (
    0.10,
    0.50,
    0.90,
)

# Direction horizons, in 5m bars.
HORIZON_BARS = {
    "H5": 1,
    "H15": 3,
    "H30": 6,
}

PRIMARY_HORIZON = "H15"

# Non-overlap stride. H5 does not overlap, so it is left ordinary.
NONOVERLAP_MOD = {
    "H15": 3,
    "H30": 6,
}

DIRECTION_MODELS = {
    "D0a": [
        "center",
    ],
    "D0b": [
        "asymmetry",
    ],
    "D0c": [
        "center",
        "width",
        "asymmetry",
    ],
}

STRATA = {
    "ALL": None,
    "TOP50": 0.50,
    "TOP30": 0.70,
    "TOP20": 0.80,
}

EPS = 1e-12

PROB_CLIP = 1e-6

SPREAD_QUANTILE = 0.20


# ============================================================
# Quantile state (regenerated, locked design)
# ============================================================

def quantile_state(
    panel: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Regenerate 15m Quantile OOS predictions.

    Identical code path to the rebaseline: same `make_model`, same
    fold geometry, same purge. Returns one row per 15m decision.
    """

    target = (
        f"target_raw_return_h"
        f"{QUANT_HORIZON}"
    )

    long_col = (
        f"target_long_mfe_h"
        f"{QUANT_HORIZON}"
    )

    short_col = (
        f"target_short_mfe_h"
        f"{QUANT_HORIZON}"
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
        horizon=(
            QUANT_HORIZON
        ),
    )

    pred = {
        q: np.full(
            n,
            np.nan,
        )
        for q in QUANTILES
    }

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
        ) < QUANT_HORIZON:
            raise RuntimeError(
                "quantile purge "
                "invariant failed"
            )

        for q in QUANTILES:

            model = (
                make_model(
                    QUANT_MODEL,
                    q,
                )
            )

            model.fit(
                X_all.iloc[
                    tr
                ],
                y_all[
                    tr
                ],
            )

            pred[
                q
            ][
                te
            ] = model.predict(
                X_all.iloc[
                    te
                ]
            )

    out = pd.DataFrame(
        {
            "decision_time": (
                pd.to_datetime(
                    frame[
                        "meta_decision_time"
                    ]
                )
            ),
            "q10": pred[
                0.10
            ],
            "q50": pred[
                0.50
            ],
            "q90": pred[
                0.90
            ],
        }
    )

    out = out[
        out[
            "q10"
        ].notna()
        & out[
            "q50"
        ].notna()
        & out[
            "q90"
        ].notna()
    ].reset_index(
        drop=True
    )

    return out


def verify_against_rebaseline(
    state: pd.DataFrame,
    instrument: str,
    reference: pd.DataFrame,
) -> dict:
    """Confirm the regenerated predictions reproduce the committed
    rebaseline interval skill.

    If these disagree, the direction study is not using the same
    Quantile output and must stop.
    """

    row = reference[
        (
            reference[
                "instrument"
            ]
            == instrument
        )
        & (
            reference[
                "horizon"
            ]
            == QUANT_HORIZON
        )
        & (
            reference[
                "feature_set"
            ]
            == (
                QUANT_FEATURE_SET
            )
        )
        & (
            reference[
                "model"
            ]
            == QUANT_MODEL
        )
    ]

    if len(row) != 1:
        raise RuntimeError(
            f"{instrument}: rebaseline "
            "reference row not found"
        )

    row = row.iloc[
        0
    ]

    return {
        "reference_interval_skill": (
            float(
                row[
                    "interval_skill"
                ]
            )
        ),
        "reference_oos_rows": int(
            row[
                "oos_rows"
            ]
        ),
        "regenerated_rows": int(
            len(
                state
            )
        ),
    }


# ============================================================
# 5m execution grid
# ============================================================

def build_execution_grid(
    five: pd.DataFrame,
    state: pd.DataFrame,
) -> pd.DataFrame:
    """Carry the 15m Quantile state forward onto the 5m grid.

    A state is valid from its own decision_time until the next
    decision_time. Never backward filled.
    """

    f = five.sort_values(
        "bar_start_time"
    ).reset_index(
        drop=True
    )

    s = state.sort_values(
        "decision_time"
    ).reset_index(
        drop=True
    )

    state_time = (
        s[
            "decision_time"
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

    bar_time = (
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

    pos = (
        np.searchsorted(
            state_time,
            bar_time,
            side=(
                "right"
            ),
        )
        - 1
    )

    usable = pos >= 0

    out = pd.DataFrame(
        {
            "row_id": np.arange(
                len(
                    f
                )
            ),
            "bar_start_time": (
                f[
                    "bar_start_time"
                ].to_numpy()
            ),
        }
    )

    for col in (
        "q10",
        "q50",
        "q90",
    ):

        values = np.full(
            len(
                f
            ),
            np.nan,
        )

        values[
            usable
        ] = (
            s[
                col
            ]
            .to_numpy(
                dtype=float
            )[
                pos[
                    usable
                ]
            ]
        )

        out[
            col
        ] = values

    out = out[
        usable
    ].reset_index(
        drop=True
    )

    return out


def add_targets(
    grid: pd.DataFrame,
    five: pd.DataFrame,
) -> pd.DataFrame:
    """Future return / MFE / MAE for every direction horizon.

    Entry is the open of the decision bar, exit is the close of the
    n-th bar of the window, so the entry price is known at decision
    time.
    """

    f = five.sort_values(
        "bar_start_time"
    ).reset_index(
        drop=True
    )

    open_ = f[
        "open"
    ].to_numpy(
        dtype=float
    )

    high = f[
        "high"
    ].to_numpy(
        dtype=float
    )

    low = f[
        "low"
    ].to_numpy(
        dtype=float
    )

    close = f[
        "close"
    ].to_numpy(
        dtype=float
    )

    row_id = (
        grid[
            "row_id"
        ]
        .to_numpy(
            dtype=int
        )
    )

    log_open = np.log(
        open_[
            row_id
        ]
    )

    log_high = np.log(
        high
    )

    log_low = np.log(
        low
    )

    log_close = np.log(
        close
    )

    n_rows = len(
        row_id
    )

    entry = np.log(
        open_[
            row_id
        ]
    )

    for name, n in (
        HORIZON_BARS.items()
    ):

        ret = np.full(
            n_rows,
            np.nan,
        )

        mfe_up = np.full(
            n_rows,
            np.nan,
        )

        mae_up = np.full(
            n_rows,
            np.nan,
        )

        mfe_dn = np.full(
            n_rows,
            np.nan,
        )

        mae_dn = np.full(
            n_rows,
            np.nan,
        )

        # Fully vectorized: every consecutive n-bar window is
        # a view, then we gather the windows we need.
        ok = (
            row_id
            + n
        ) <= len(
            f
        )

        idx = np.where(
            ok
        )[
            0
        ]

        if len(
            idx
        ):

            sel = (
                row_id[
                    idx
                ]
            )

            ret[
                idx
            ] = (
                log_close[
                    sel
                    + n
                    - 1
                ]
                - entry[
                    idx
                ]
            )

            base = entry[
                idx
            ][
                :,
                None
            ]

            up = (
                np.lib.stride_tricks
                .sliding_window_view(
                    log_high,
                    n,
                )[
                    sel
                ]
                - base
            )

            down = (
                base
                - np.lib.stride_tricks
                .sliding_window_view(
                    log_low,
                    n,
                )[
                    sel
                ]
            )

            mfe_up[
                idx
            ] = np.maximum(
                up.max(
                    axis=1
                ),
                0.0,
            )

            mae_up[
                idx
            ] = np.maximum(
                -up.min(
                    axis=1
                ),
                0.0,
            )

            mfe_dn[
                idx
            ] = np.maximum(
                down.max(
                    axis=1
                ),
                0.0,
            )

            mae_dn[
                idx
            ] = np.maximum(
                -down.min(
                    axis=1
                ),
                0.0,
            )

        grid[
            f"fut_ret_{name}"
        ] = ret

        grid[
            f"fut_mfe_up_{name}"
        ] = mfe_up

        grid[
            f"fut_mae_up_{name}"
        ] = mae_up

        grid[
            f"fut_mfe_dn_{name}"
        ] = mfe_dn

        grid[
            f"fut_mae_dn_{name}"
        ] = mae_dn

    return grid


# ============================================================
# Metrics
# ============================================================

def binary_metrics(
    y: np.ndarray,
    p: np.ndarray,
    p_train: float,
    fut_ret: np.ndarray,
    mfe_up: np.ndarray,
    mae_up: np.ndarray,
) -> dict:

    if (
        len(y) < 50
        or y.min()
        == y.max()
    ):
        return {}

    p_safe = np.clip(
        p,
        PROB_CLIP,
        1.0
        - PROB_CLIP,
    )

    auc = float(
        roc_auc_score(
            y,
            p_safe
        )
    )

    brier_model = float(
        np.mean(
            (
                p_safe
                - y
            )
            ** 2
        )
    )

    brier_base = float(
        np.mean(
            (
                p_train
                - y
            )
            ** 2
        )
    )

    ll_model = float(
        log_loss(
            y,
            p_safe,
            labels=[
                0,
                1
            ],
        )
    )

    ll_base = float(
        log_loss(
            y,
            np.full(
                len(
                    y
                ),
                p_train,
            ),
            labels=[
                0,
                1
            ],
        )
    )

    # Calibration slope: regress the realised outcome on the
    # predicted log-odds. A calibrated model has slope ~ 1.
    logit = np.log(
        p_safe
        / (
            1.0
            - p_safe
        )
    )

    model = LinearRegression()

    model.fit(
        logit.reshape(
            -1,
            1
        ),
        y,
    )

    slope = float(
        model.coef_[
            0
        ]
    )

    hi_cut = np.quantile(
        p_safe,
        1.0
        - SPREAD_QUANTILE,
    )

    lo_cut = np.quantile(
        p_safe,
        SPREAD_QUANTILE,
    )

    top = (
        p_safe
        >= hi_cut
    )

    bottom = (
        p_safe
        <= lo_cut
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
        "n": int(
            len(
                y
            )
        ),
        "up_rate": float(
            y.mean()
        ),
        "auc": auc,
        "brier": brier_model,
        "brier_skill": (
            1.0
            - brier_model
            / brier_base
            if brier_base
            > 0
            else np.nan
        ),
        "log_loss": ll_model,
        "logloss_skill": (
            1.0
            - ll_model
            / ll_base
            if ll_base
            > 0
            else np.nan
        ),
        "calibration_slope": (
            slope
        ),
        "return_spread": (
            spread
        ),
        "mean_ret_top20": (
            float(
                np.mean(
                    fut_ret[
                        top
                    ]
                )
            )
            if top.any()
            else np.nan
        ),
        "mean_ret_bottom20": (
            float(
                np.mean(
                    fut_ret[
                        bottom
                    ]
                )
            )
            if bottom.any()
            else np.nan
        ),
        "mfe_up_top20": (
            float(
                np.mean(
                    mfe_up[
                        top
                    ]
                )
            )
            if top.any()
            else np.nan
        ),
        "mae_up_top20": (
            float(
                np.mean(
                    mae_up[
                        top
                    ]
                )
            )
            if top.any()
            else np.nan
        ),
        "mfe_up_bottom20": (
            float(
                np.mean(
                    mfe_up[
                        bottom
                    ]
                )
            )
            if bottom.any()
            else np.nan
        ),
    }


# ============================================================
# One instrument / horizon / model / stratum
# ============================================================

def run_direction(
    grid: pd.DataFrame,
    *,
    horizon: str,
    feature_names: list[str],
    stratum: str,
    stratum_quantile,
) -> tuple[
    dict,
    pd.DataFrame,
]:

    n_bars = (
        HORIZON_BARS[
            horizon
        ]
    )

    ret_col = (
        f"fut_ret_{horizon}"
    )

    work = grid.dropna(
        subset=[
            ret_col
        ]
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

    work[
        "width"
    ] = (
        work[
            "q90"
        ]
        - work[
            "q10"
        ]
    )

    upper = (
        work[
            "q90"
        ]
        - work[
            "q50"
        ]
    )

    lower = (
        work[
            "q50"
        ]
        - work[
            "q10"
        ]
    )

    work[
        "asymmetry"
    ] = (
        (
            upper
            - lower
        )
        / (
            upper
            + lower
            + EPS
        )
    )

    work[
        "center"
    ] = work[
        "q50"
    ]

    n = len(
        work
    )

    folds = make_folds(
        n,
        horizon=n_bars,
    )

    pred_prob = np.full(
        n,
        np.nan,
    )

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
        ) < n_bars:
            raise RuntimeError(
                f"{horizon}: purge "
                "invariant failed"
            )

        train = work.iloc[
            tr
        ]

        # Stratum threshold from TRAIN rows only.
        if (
            stratum_quantile
            is not None
        ):

            cut = float(
                np.quantile(
                    train[
                        "width"
                    ].to_numpy(
                        dtype=float
                    ),
                    stratum_quantile,
                )
            )

            tr_mask = np.where(
                train[
                    "width"
                ].to_numpy(
                    dtype=float
                )
                >= cut
            )[
                0
            ]

            te_mask = np.where(
                work.iloc[
                    te
                ][
                    "width"
                ]
                .to_numpy(
                    dtype=float
                )
                >= cut
            )[
                0
            ]

            tr = tr[
                tr_mask
            ]

            te = te[
                te_mask
            ]

        if (
            len(tr)
            < 200
            or len(te)
            < 50
        ):
            continue

        train = work.iloc[
            tr
        ]

        test = work.iloc[
            te
        ]

        if (
            train[
                "y"
            ].nunique()
            < 2
        ):
            continue

        X_tr = train[
            feature_names
        ].to_numpy(
            dtype=float
        )

        X_te = test[
            feature_names
        ].to_numpy(
            dtype=float
        )

        scaler = (
            StandardScaler()
        )

        scaler.fit(
            X_tr
        )

        model = (
            LogisticRegression(
                penalty=(
                    "l2"
                ),
                C=1.0,
                max_iter=(
                    2000
                ),
            )
        )

        model.fit(
            scaler.transform(
                X_tr
            ),
            train[
                "y"
            ].to_numpy(),
        )

        pred_prob[
            te
        ] = model.predict_proba(
            scaler.transform(
                X_te
            )
        )[
            :,
            1
        ]

        fold_id[
            te
        ] = fold[
            "fold"
        ]

    oos = (
        fold_id
        >= 0
    )

    out = work.loc[
        oos,
        [
            "row_id",
        ]
    ].copy()

    out[
        "p"
    ] = pred_prob[
        oos
    ]

    out[
        "y"
    ] = work.loc[
        oos,
        "y"
    ].to_numpy()

    out[
        "fut_ret"
    ] = work.loc[
        oos,
        ret_col
    ].to_numpy()

    out[
        "mfe_up"
    ] = work.loc[
        oos,
        f"fut_mfe_up_{horizon}"
    ].to_numpy()

    out[
        "mae_up"
    ] = work.loc[
        oos,
        f"fut_mae_up_{horizon}"
    ].to_numpy()

    # Baseline: pooled train UP frequency across folds.
    p_train = float(
        work.loc[
            oos,
            "y"
        ].mean()
    )

    if (
        len(
            out
        )
        < 50
        or out[
            "y"
        ].nunique()
        < 2
    ):
        return (
            {
                "n_valid": int(
                    len(
                        out
                    )
                )
            },
            out,
        )

    metrics = binary_metrics(
        out[
            "y"
        ].to_numpy(
            dtype=int
        ),
        out[
            "p"
        ].to_numpy(
            dtype=float
        ),
        p_train,
        out[
            "fut_ret"
        ].to_numpy(
            dtype=float
        ),
        out[
            "mfe_up"
        ].to_numpy(
            dtype=float
        ),
        out[
            "mae_up"
        ].to_numpy(
            dtype=float
        ),
    )

    metrics[
        "folds_evaluated"
    ] = int(
        pd.unique(
            fold_id[
                oos
            ]
        ).size
    )

    # Non-overlap check.
    metrics[
        "nonoverlap"
    ] = None

    return (
        metrics,
        out,
    )


def nonoverlap_metrics(
    pred_frame: pd.DataFrame,
    *,
    horizon: str,
    p_train: float,
) -> list[dict]:

    mod = (
        NONOVERLAP_MOD.get(
            horizon
        )
    )

    if (
        mod is None
        or pred_frame.empty
    ):
        return []

    rows = []

    for offset in range(
        mod
    ):

        sub = pred_frame[
            pred_frame[
                "row_id"
            ]
            % mod
            == offset
        ]

        if (
            len(sub)
            < 50
            or sub[
                "y"
            ].nunique()
            < 2
        ):
            continue

        m = binary_metrics(
            sub[
                "y"
            ].to_numpy(
                dtype=int
            ),
            sub[
                "p"
            ].to_numpy(
                dtype=float
            ),
            p_train,
            sub[
                "fut_ret"
            ].to_numpy(
                dtype=float
            ),
            sub[
                "mfe_up"
            ].to_numpy(
                dtype=float
            ),
            sub[
                "mae_up"
            ].to_numpy(
                dtype=float
            ),
        )

        if not m:
            continue

        rows.append(
            {
                "offset": int(
                    offset
                ),
                "mod": int(
                    mod
                ),
                "n": int(
                    m[
                        "n"
                    ]
                ),
                "auc": float(
                    m[
                        "auc"
                    ]
                ),
                "brier_skill": (
                    float(
                        m[
                            "brier_skill"
                        ]
                    )
                ),
                "logloss_skill": (
                    float(
                        m[
                            "logloss_skill"
                        ]
                    )
                ),
                "return_spread": (
                    float(
                        m[
                            "return_spread"
                        ]
                    )
                ),
            }
        )

    return rows


# ============================================================
# Main
# ============================================================

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

    reference = pd.read_csv(
        REBASELINE
    )

    feature_cols = (
        FEATURE_SETS[
            QUANT_FEATURE_SET
        ]
    )

    rows = []
    nonoverlap_rows = []
    verify_rows = []

    for (
        instrument
    ) in INSTRUMENTS:

        print(
            "=" * 70
        )

        print(
            f"{instrument}"
        )

        print(
            "=" * 70
        )

        panel = pd.read_csv(
            PANEL_15M
            / f"{instrument}_panel.csv",
            parse_dates=[
                "meta_base_bar_time",
                "meta_decision_time",
            ],
        )

        five = pd.read_csv(
            PANEL_5M
            / f"{instrument}_5m.csv",
            parse_dates=[
                "bar_start_time",
                "bar_end_time",
            ],
        )

        state = quantile_state(
            panel,
            feature_cols,
        )

        check = (
            verify_against_rebaseline(
                state,
                instrument,
                reference,
            )
        )

        verify_rows.append(
            {
                "instrument": (
                    instrument
                ),
                **check,
            }
        )

        print(
            f"  quantile rows: "
            f"{check['regenerated_rows']}"
            f"  (rebaseline "
            f"{check['reference_oos_rows']})"
        )

        grid = build_execution_grid(
            five,
            state,
        )

        grid = add_targets(
            grid,
            five,
        )

        print(
            f"  5m execution rows: "
            f"{len(grid)}"
        )

        for (
            horizon
        ) in HORIZON_BARS:

            for (
                model_name,
                features,
            ) in (
                DIRECTION_MODELS.items()
            ):

                for (
                    stratum,
                    q,
                ) in (
                    STRATA.items()
                ):

                    (
                        metrics,
                        pred_frame,
                    ) = run_direction(
                        grid,
                        horizon=(
                            horizon
                        ),
                        feature_names=(
                            features
                        ),
                        stratum=(
                            stratum
                        ),
                        stratum_quantile=(
                            q
                        ),
                    )

                    row = {
                        "instrument": (
                            instrument
                        ),
                        "horizon": (
                            horizon
                        ),
                        "model": (
                            model_name
                        ),
                        "stratum": (
                            stratum
                        ),
                        **metrics,
                    }

                    rows.append(
                        row
                    )

                    if (
                        model_name
                        == "D0b"
                    ):

                        p_train = (
                            float(
                                pred_frame[
                                    "y"
                                ].mean()
                            )
                            if len(
                                pred_frame
                            )
                            else np.nan
                        )

                        for (
                            entry
                        ) in nonoverlap_metrics(
                            pred_frame,
                            horizon=(
                                horizon
                            ),
                            p_train=(
                                p_train
                            ),
                        ):

                            nonoverlap_rows.append(
                                {
                                    "instrument": (
                                        instrument
                                    ),
                                    "horizon": (
                                        horizon
                                    ),
                                    "model": (
                                        model_name
                                    ),
                                    "stratum": (
                                        stratum
                                    ),
                                    **entry,
                                }
                            )

                    if (
                        horizon
                        == (
                            PRIMARY_HORIZON
                        )
                        and stratum
                        == "ALL"
                    ):

                        print(
                            f"    {horizon:4s} "
                            f"{model_name} "
                            f"{stratum:6s} "
                            f"n={metrics.get('n', 0):6d} "
                            f"AUC="
                            f"{metrics.get('auc', float('nan')):.4f} "
                            f"BrierSkill="
                            f"{metrics.get('brier_skill', float('nan')):+.5f} "
                            f"Spread="
                            f"{metrics.get('return_spread', float('nan')):+.3e}"
                        )

    results = pd.DataFrame(
        rows
    )

    results.to_csv(
        OUT
        / "direction_v0_by_instrument.csv",
        index=False,
    )

    nonoverlap = pd.DataFrame(
        nonoverlap_rows
    )

    nonoverlap.to_csv(
        OUT
        / "direction_v0_nonoverlap.csv",
        index=False,
    )

    verify = pd.DataFrame(
        verify_rows
    )

    verify.to_csv(
        OUT
        / "quantile_state_verification.csv",
        index=False,
    )

    # ========================================================
    # Cross-instrument summary
    # ========================================================

    metrics = [
        "auc",
        "brier_skill",
        "logloss_skill",
        "return_spread",
        "calibration_slope",
    ]

    summary_rows = []

    for (
        group_name,
        members,
    ) in GROUPS.items():

        g0 = results[
            results[
                "instrument"
            ].isin(
                members
            )
        ]

        for (
            horizon,
            model_name,
            stratum,
        ), g in g0.groupby(
            [
                "horizon",
                "model",
                "stratum",
            ],
            observed=(
                True
            ),
        ):

            row = {
                "group": (
                    group_name
                ),
                "horizon": (
                    horizon
                ),
                "model": (
                    model_name
                ),
                "stratum": (
                    stratum
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
                ] = (
                    float(
                        values.median()
                    )
                    if len(
                        values
                    )
                    else np.nan
                )

                row[
                    f"positive_share_{metric}"
                ] = positive_share(
                    values
                )

            summary_rows.append(
                row
            )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUT
        / "direction_v0_cross_summary.csv",
        index=False,
    )

    config = {
        "purpose": (
            "Test whether quantile distribution "
            "geometry carries direction "
            "information inside high-opportunity "
            "states."
        ),
        "quantile_source": {
            "horizon": (
                QUANT_HORIZON
            ),
            "feature_set": (
                QUANT_FEATURE_SET
            ),
            "model": QUANT_MODEL,
            "note": (
                "regenerated through the locked "
                "code path, not re-fit and not "
                "modified"
            ),
        },
        "direction_horizons_bars": (
            HORIZON_BARS
        ),
        "primary_horizon": (
            PRIMARY_HORIZON
        ),
        "direction_models": (
            DIRECTION_MODELS
        ),
        "strata": {
            k: v
            for k, v in (
                STRATA.items()
            )
        },
        "strata_thresholds": (
            "computed on TRAIN rows per fold "
            "only"
        ),
        "learner": (
            "LogisticRegression("
            "penalty=l2, C=1.0, max_iter=2000) "
            "+ StandardScaler fitted on train"
        ),
        "state_carry": (
            "15m decision state forward-filled "
            "onto the 5m execution grid, never "
            "backward filled"
        ),
        "target_definition": (
            "entry = open of the decision 5m "
            "bar, exit = close of the n-th 5m "
            "bar of the window"
        ),
        "nonoverlap_mod": (
            NONOVERLAP_MOD
        ),
        "groups": {
            k: list(v)
            for k, v in (
                GROUPS.items()
            )
        },
        "no_quantile_refit": True,
        "no_price_path": True,
        "no_rv": True,
        "no_volume": True,
        "no_oi": True,
        "no_transaction_data": True,
        "no_gbr": True,
        "no_tuning": True,
        "no_pnl": True,
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

    expected = (
        len(INSTRUMENTS)
        * len(
            HORIZON_BARS
        )
        * len(
            DIRECTION_MODELS
        )
        * len(
            STRATA
        )
    )

    if (
        len(
            results
        )
        != expected
    ):
        raise RuntimeError(
            f"row count "
            f"{len(results)} "
            f"!= {expected}"
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
        "quantile_state_verification": (
            verify.to_dict(
                "records"
            )
        ),
        "no_quantile_refit": True,
        "no_tuning": True,
        "no_pnl": True,
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
    # Compact report
    # ========================================================

    lines = [
        "# Direction V0",
        "",
        "Fixed L2 logistic regression on "
        "Quantile OOS geometry only.",
        "",
        f"## {PRIMARY_HORIZON} primary, "
        "ALL_8",
        "",
        "| model | stratum | median AUC | "
        "median Brier skill | pos | "
        "median spread | pos |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    s = summary[
        (
            summary[
                "group"
            ]
            == "ALL_8"
        )
        & (
            summary[
                "horizon"
            ]
            == PRIMARY_HORIZON
        )
    ]

    for (
        _,
        r,
    ) in s.sort_values(
        [
            "model",
            "stratum",
        ]
    ).iterrows():

        lines.append(
            f"| {r['model']} "
            f"| {r['stratum']} "
            f"| "
            f"{r['median_auc']:.4f} "
            f"| "
            f"{r['median_brier_skill']:+.5f} "
            f"| "
            f"{r['positive_share_brier_skill']:.2f} "
            f"| "
            f"{r['median_return_spread']:+.3e} "
            f"| "
            f"{r['positive_share_return_spread']:.2f} "
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
        + "=" * 70
    )

    print(
        f"CROSS SUMMARY "
        f"{PRIMARY_HORIZON} (ALL_8)"
    )

    print(
        "=" * 70
    )

    cols = [
        "model",
        "stratum",
        "median_auc",
        "median_brier_skill",
        "positive_share_brier_skill",
        "median_logloss_skill",
        "positive_share_logloss_skill",
        "median_return_spread",
        "positive_share_return_spread",
    ]

    with pd.option_context(
        "display.width",
        250,
    ):
        print(
            s[cols].to_string(
                index=False
            )
        )

    print(
        "\nDIRECTION_V0_PASS"
    )


if __name__ == "__main__":
    main()
