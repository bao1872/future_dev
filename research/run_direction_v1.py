#!/usr/bin/env python3
"""Direction V1 -- price path continuation vs reversal.

V0 showed that Quantile distribution GEOMETRY carries no usable
direction information. V1 therefore keeps Quantile as the WHEN /
opportunity gate only, and asks whether the RECENT PRICE PATH says
WHERE.

    Quantile Opportunity  -> WHEN
    recent 5-30m path     -> WHERE (continuation or reversal)

Continuity correction (carried in from the V0 audit)
----------------------------------------------------
The V0 targets used array-position windows, which silently cross
lunch breaks and overnight gaps:

    H5  0.0% of rows crossed a gap
    H15 9.4%
    H30 23.5%

Here every target window AND every path-feature window must be
continuous in CALENDAR time (adjacent bars exactly 5 minutes apart).
A window that crosses a session gap is invalid, because "the next
15 minutes" does not exist across a closed market.

Causality
---------
At decision time t the last completed 5m bar is the one covering
[t-5m, t). All path features use bars up to and including that one.
The target starts at t, using the open of the bar beginning at t,
which is known at t. Features and target never share a bar.

Restrictions
------------
- the Quantile layer is not re-fit; predictions are regenerated
  through the identical locked code path and verified against the
  committed rebaseline
- no RV, no semivariance, no volume, no OI, no transaction data
- no SMC / DSA / Momentum
- no tuning, no PnL, no backtest
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingClassifier,
)
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
    RANDOM_STATE,
    make_model,
)

from research.run_quantile_rebaseline import (  # noqa: E402
    FEATURE_SETS,
    make_folds,
)

from research.run_direction_v0 import (  # noqa: E402
    build_execution_grid,
    quantile_state,
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
    / "direction_v1"
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

QUANT_HORIZON = 4

QUANT_FEATURE_SET = (
    "F1_VOL"
)

QUANT_MODEL = (
    "gbr_quantile"
)

HORIZON_BARS = {
    "H5": 1,
    "H15": 3,
    "H30": 6,
}

PRIMARY_HORIZON = "H15"

PRIMARY_STRATUM = "TOP30"

NONOVERLAP_MOD = {
    "H15": 3,
    "H30": 6,
}

STRATA = {
    "ALL": None,
    "TOP50": 0.50,
    "TOP30": 0.70,
    "TOP20": 0.80,
}

FEATURE_BLOCKS = {
    "P0": [
        "r5",
    ],
    "P1": [
        "r5",
        "r10",
        "r15",
        "r30",
    ],
    "P2": [
        "r5",
        "r10",
        "r15",
        "r30",
        "sign_balance_15m",
        "sign_balance_30m",
        "efficiency_15m",
        "efficiency_30m",
    ],
    "P3": [
        "r5",
        "r10",
        "r15",
        "r30",
        "sign_balance_15m",
        "sign_balance_30m",
        "efficiency_15m",
        "efficiency_30m",
        "body_to_range",
        "close_location",
    ],
}

LEARNERS = (
    "logistic",
    "gbr",
)

FIVE_NS = 5 * 60 * 1_000_000_000

EPS = 1e-12

PROB_CLIP = 1e-6

SPREAD_QUANTILE = 0.20

# Path-state mechanism bins.
STATE_BINS = 5


# ============================================================
# Shallow GBR classifier
# ============================================================

def make_gbr_classifier():
    """Classifier mirroring the locked shallow quantile GBR.

    Same architecture and same hyper-parameters as the quantile
    challenger, only the loss changes because the target is binary:

        n_estimators=150, learning_rate=0.03,
        max_depth=2, min_samples_leaf=30,
        subsample=0.80, random_state=RANDOM_STATE

    No tuning. If the GBR beats logistic clearly that is evidence
    of a non-linear continuation/reversal relation, not a licence
    to search depth or learning rate.
    """

    return (
        GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=30,
            subsample=0.80,
            random_state=(
                RANDOM_STATE
            ),
        )
    )


# ============================================================
# Continuity helper
# ============================================================

def continuity_prefix(
    five: pd.DataFrame,
) -> np.ndarray:
    """Cumulative count of calendar discontinuities.

    cs[k] = number of gaps among bars [0, k-1], so the number of
    gaps inside a window [j, j+n-1] is cs[j+n-1] - cs[j], and the
    window is continuous exactly when that is zero.
    """

    start_ns = (
        five[
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

    diff = np.diff(
        start_ns
    )

    # adjacent[j] is True when bar j and bar j+1 are 5 minutes
    # apart in calendar time.
    adjacent = np.concatenate(
        [
            diff
            == FIVE_NS,
            [False],
        ]
    )

    bad = (
        ~adjacent
    ).astype(
        np.int64
    )

    return np.concatenate(
        [
            [0],
            np.cumsum(
                bad
            ),
        ]
    )


def window_is_continuous(
    cs: np.ndarray,
    start: np.ndarray,
    n: int,
    n_bars: int,
) -> np.ndarray:
    """Boolean mask: [start, start+n-1] has no session gap."""

    ok = np.zeros(
        len(
            start
        ),
        dtype=bool,
    )

    fits = (
        start
        + n
    ) <= n_bars

    idx = np.where(
        fits
    )[
        0
    ]

    if len(
        idx
    ):

        j = start[
            idx
        ]

        gaps = (
            cs[
                j + n - 1
            ]
            - cs[j]
        )

        ok[
            idx
        ] = (
            gaps == 0
        )

    return ok


# ============================================================
# Continuous targets
# ============================================================

def add_targets(
    grid: pd.DataFrame,
    five: pd.DataFrame,
    cs: np.ndarray,
) -> pd.DataFrame:

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

    n5 = len(
        f
    )

    n_rows = len(
        row_id
    )

    entry = np.log(
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

    for (
        name,
        n,
    ) in HORIZON_BARS.items():

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

        cont = (
            window_is_continuous(
                cs,
                row_id,
                n,
                n5,
            )
        )

        idx = np.where(
            cont
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
                np.lib
                .stride_tricks
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
                - np.lib
                .stride_tricks
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

        grid[
            f"cont_{name}"
        ] = cont

    return grid


# ============================================================
# Path features
# ============================================================

def add_path_features(
    grid: pd.DataFrame,
    five: pd.DataFrame,
    cs: np.ndarray,
) -> pd.DataFrame:

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

    n5 = len(
        f
    )

    log_close = np.log(
        close
    )

    # One-bar log returns; ret[i] = log(close[i]/close[i-1]).
    ret = np.full(
        n5,
        np.nan,
    )

    ret[
        1:
    ] = (
        log_close[
            1:
        ]
        - log_close[
            :-1
        ]
    )

    row_id = (
        grid[
            "row_id"
        ]
        .to_numpy(
            dtype=int
        )
    )

    # Last CLOSED bar at decision time t is row_id - 1.
    k = (
        row_id
        - 1
    )

    n_rows = len(
        row_id
    )

    usable = k >= 1

    data: dict[
        str, np.ndarray
    ] = {}

    # ---- multi-horizon returns, continuous only ----

    for (
        label,
        n,
    ) in (
        (
            "r5",
            1,
        ),
        (
            "r10",
            2,
        ),
        (
            "r15",
            3,
        ),
        (
            "r30",
            6,
        ),
    ):

        out = np.full(
            n_rows,
            np.nan,
        )

        # window of n bars ending at k, i.e. [k-n+1, k]
        start = (
            k
            - n
            + 1
        )

        cont = (
            window_is_continuous(
                cs,
                start,
                n,
                n5,
            )
            & usable
        )

        idx = np.where(
            cont
        )[
            0
        ]

        if len(
            idx
        ):

            out[
                idx
            ] = (
                log_close[
                    k[
                        idx
                    ]
                ]
                - log_close[
                    start[
                        idx
                    ]
                    - 1
                ]
            )

        data[
            label
        ] = out

    # ---- sign balance and efficiency ----

    for (
        label,
        n,
    ) in (
        (
            "sign_balance_15m",
            3,
        ),
        (
            "sign_balance_30m",
            6,
        ),
    ):

        out = np.full(
            n_rows,
            np.nan,
        )

        for i in range(
            n_rows
        ):

            if not (
                usable[
                    i
                ]
            ):
                continue

            a = (
                k[
                    i
                ]
                - n
                + 1
            )

            if a < 1:
                continue

            if (
                cs[
                    k[
                        i
                    ]
                    + 1
                ]
                - cs[
                    a
                ]
            ) != 0:
                continue

            window = (
                ret[
                    a : k[
                        i
                    ]
                    + 1
                ]
            )

            if not np.all(
                np.isfinite(
                    window
                )
            ):
                continue

            out[
                i
            ] = np.mean(
                np.sign(
                    window
                )
            )

        data[
            label
        ] = out

    for (
        label,
        n,
    ) in (
        (
            "efficiency_15m",
            3,
        ),
        (
            "efficiency_30m",
            6,
        ),
    ):

        out = np.full(
            n_rows,
            np.nan,
        )

        for i in range(
            n_rows
        ):

            if not (
                usable[
                    i
                ]
            ):
                continue

            a = (
                k[
                    i
                ]
                - n
                + 1
            )

            if a < 1:
                continue

            if (
                cs[
                    k[
                        i
                    ]
                    + 1
                ]
                - cs[
                    a
                ]
            ) != 0:
                continue

            window = (
                ret[
                    a : k[
                        i
                    ]
                    + 1
                ]
            )

            if not np.all(
                np.isfinite(
                    window
                )
            ):
                continue

            denom = (
                np.sum(
                    np.abs(
                        window
                    )
                )
                + EPS
            )

            out[
                i
            ] = (
                abs(
                    np.sum(
                        window
                    )
                )
                / denom
            )

        data[
            label
        ] = out

    # ---- current closed-bar geometry, dimensionless ----

    body = np.full(
        n_rows,
        np.nan,
    )

    loc = np.full(
        n_rows,
        np.nan,
    )

    idx = np.where(
        usable
    )[
        0
    ]

    if len(
        idx
    ):

        kk = k[
            idx
        ]

        rng = (
            high[
                kk
            ]
            - low[
                kk
            ]
        )

        safe = (
            rng > 0
        )

        jj = idx[
            safe
        ]

        kk2 = kk[
            safe
        ]

        body[
            jj
        ] = (
            (
                close[
                    kk2
                ]
                - open_[
                    kk2
                ]
            )
            / rng[
                safe
            ]
        )

        loc[
            jj
        ] = (
            (
                close[
                    kk2
                ]
                - low[
                    kk2
                ]
            )
            / rng[
                safe
            ]
        )

    data[
        "body_to_range"
    ] = body

    data[
        "close_location"
    ] = loc

    for key, values in (
        data.items()
    ):

        grid[
            key
        ] = values

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

    logit = np.log(
        p_safe
        / (
            1.0
            - p_safe
        )
    )

    cal = LinearRegression()

    cal.fit(
        logit.reshape(
            -1,
            1
        ),
        y,
    )

    slope = float(
        cal.coef_[
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
        "calibration_slope": (
            slope
        ),
        "return_spread": (
            spread
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
    }


def spearman(
    a,
    b,
) -> float:

    x = pd.Series(
        a,
        dtype=float,
    )

    y = pd.Series(
        b,
        dtype=float,
    )

    ok = (
        x.notna()
        & y.notna()
    )

    if (
        ok.sum()
        < 30
    ):
        return float(
            "nan"
        )

    return float(
        x[ok]
        .rank()
        .corr(
            y[
                ok
            ].rank()
        )
    )


# ============================================================
# Model run
# ============================================================

def run_model(
    grid: pd.DataFrame,
    *,
    horizon: str,
    feature_names: list[str],
    stratum: str,
    stratum_quantile,
    learner: str,
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
        subset=(
            [
                ret_col
            ]
            + feature_names
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

    n = len(
        work
    )

    if n < 500:
        return (
            {
                "n_valid": 0
            },
            pd.DataFrame(),
        )

    folds = make_folds(
        n,
        horizon=n_bars,
    )

    prob = np.full(
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

        if (
            stratum_quantile
            is not None
        ):

            cut = float(
                np.quantile(
                    work.iloc[
                        tr
                    ][
                        "width"
                    ].to_numpy(
                        dtype=float
                    ),
                    stratum_quantile,
                )
            )

            tr = tr[
                work.iloc[
                    tr
                ][
                    "width"
                ].to_numpy(
                    dtype=float
                )
                >= cut
            ]

            te = te[
                work.iloc[
                    te
                ][
                    "width"
                ].to_numpy(
                    dtype=float
                )
                >= cut
            ]

        if (
            len(tr)
            < 200
            or len(te)
            < 50
        ):
            continue

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

        if (
            learner
            == "logistic"
        ):

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

        else:

            model = (
                make_gbr_classifier()
            )

        model.fit(
            scaler.transform(
                X_tr
            ),
            train[
                "y"
            ].to_numpy(),
        )

        prob[
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

    out = pd.DataFrame(
        {
            "row_id": work.loc[
                oos,
                "row_id"
            ].to_numpy(),
            "p": prob[
                oos
            ],
            "y": work.loc[
                oos,
                "y"
            ].to_numpy(),
            "fut_ret": work.loc[
                oos,
                ret_col
            ].to_numpy(),
            "mfe_up": work.loc[
                oos,
                f"fut_mfe_up_{horizon}"
            ].to_numpy(),
            "r15_path": (
                work.loc[
                    oos,
                    "r15"
                ].to_numpy()
                if "r15"
                in work.columns
                else np.nan
            ),
        }
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

    p_train = float(
        out[
            "y"
        ].mean()
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

    return (
        metrics,
        out,
    )


# ============================================================
# Path-state mechanism table (no model)
# ============================================================

def mechanism_table(
    grid: pd.DataFrame,
    *,
    horizon: str,
) -> pd.DataFrame:
    """Future return by past-path state, thresholds from TRAIN.

    OOS rows are binned by the past r15 using quantiles computed on
    that fold's training rows only.
    """

    ret_col = (
        f"fut_ret_{horizon}"
    )

    work = grid.dropna(
        subset=[
            ret_col,
            "r15",
            "width",
        ]
    ).reset_index(
        drop=True
    )

    n_bars = (
        HORIZON_BARS[
            horizon
        ]
    )

    n = len(
        work
    )

    if n < 500:
        return pd.DataFrame()

    folds = make_folds(
        n,
        horizon=n_bars,
    )

    records = []

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

        train = (
            work.iloc[
                tr
            ]
        )

        # Opportunity threshold from TRAIN only.
        cut = float(
            np.quantile(
                train[
                    "width"
                ].to_numpy(
                    dtype=float
                ),
                STRATA[
                    PRIMARY_STRATUM
                ],
            )
        )

        edges = np.quantile(
            train[
                "r15"
            ].to_numpy(
                dtype=float
            ),
            np.linspace(
                0.0,
                1.0,
                STATE_BINS
                + 1,
            ),
        )

        edges[
            0
        ] = -np.inf

        edges[
            -1
        ] = np.inf

        test = (
            work.iloc[
                te
            ]
        )

        bin_id = (
            np.digitize(
                test[
                    "r15"
                ].to_numpy(
                    dtype=float
                ),
                edges[
                    1:-1
                ],
            )
        )

        is_top = (
            test[
                "width"
            ].to_numpy(
                dtype=float
            )
            >= cut
        )

        frame = pd.DataFrame(
            {
                "bin": bin_id,
                "fut": test[
                    ret_col
                ].to_numpy(
                    dtype=float
                ),
                "top": is_top,
            }
        )

        records.append(
            frame
        )

    if not records:
        return pd.DataFrame()

    all_rows = pd.concat(
        records,
        ignore_index=True,
    )

    out = []

    for (
        label,
        subset,
    ) in (
        (
            "ALL",
            all_rows,
        ),
        (
            PRIMARY_STRATUM,
            all_rows[
                all_rows[
                    "top"
                ]
            ],
        ),
    ):

        if len(
            subset
        ) < 50:
            continue

        for b, g in (
            subset.groupby(
                "bin"
            )
        ):

            out.append(
                {
                    "stratum": label,
                    "path_state_bin": int(
                        b
                    ),
                    "n": int(
                        len(
                            g
                        )
                    ),
                    "mean_future_return": (
                        float(
                            g[
                                "fut"
                            ].mean()
                        )
                    ),
                    "up_probability": (
                        float(
                            (
                                g[
                                    "fut"
                                ]
                                > 0
                            ).mean()
                        )
                    ),
                }
            )

    return pd.DataFrame(
        out
    )


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

    quant_cols = (
        FEATURE_SETS[
            QUANT_FEATURE_SET
        ]
    )

    rows = []
    nonoverlap_rows = []
    mech_rows = []
    verify_rows = []
    coverage_rows = []
    spearman_rows = []

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

        ref = reference[
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
                == (
                    QUANT_HORIZON
                )
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

        if len(ref) != 1:
            raise RuntimeError(
                f"{instrument}: "
                "rebaseline row "
                "not found"
            )

        ref = ref.iloc[
            0
        ]

        verify_rows.append(
            {
                "instrument": (
                    instrument
                ),
                "reference_interval_skill": (
                    float(
                        ref[
                            "interval_skill"
                        ]
                    )
                ),
                "reference_oos_rows": int(
                    ref[
                        "oos_rows"
                    ]
                ),
                "regenerated_rows": int(
                    len(
                        state
                    )
                ),
            }
        )

        cs = continuity_prefix(
            five
        )

        grid = build_execution_grid(
            five,
            state,
        )

        # Opportunity variable. Quantile contributes nothing
        # else in V1.
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

        grid = add_path_features(
            grid,
            five,
            cs,
        )

        cov = {
            "instrument": (
                instrument
            ),
            "grid_rows": int(
                len(
                    grid
                )
            ),
        }

        for name in HORIZON_BARS:

            cov[
                f"{name}_valid"
            ] = int(
                grid[
                    f"fut_ret_{name}"
                ]
                .notna()
                .sum()
            )

            cov[
                f"{name}_pct"
            ] = round(
                float(
                    grid[
                        f"fut_ret_{name}"
                    ]
                    .notna()
                    .mean()
                    * 100
                ),
                3,
            )

        for f in (
            "r5",
            "r15",
            "r30",
            "efficiency_30m",
        ):

            cov[
                f"feat_{f}_pct"
            ] = round(
                float(
                    grid[
                        f
                    ]
                    .notna()
                    .mean()
                    * 100
                ),
                3,
            )

        coverage_rows.append(
            cov
        )

        print(
            f"  grid={cov['grid_rows']} "
            f"H5={cov['H5_valid']} "
            f"H15={cov['H15_valid']} "
            f"H30={cov['H30_valid']}"
        )

        # ---- path -> future relation (no model) ----

        for (
            horizon
        ) in HORIZON_BARS:

            ret_col = (
                f"fut_ret_{horizon}"
            )

            sub = grid.dropna(
                subset=[
                    ret_col,
                    "r15",
                    "width",
                ]
            )

            cut30 = float(
                np.quantile(
                    sub[
                        "width"
                    ].to_numpy(
                        dtype=float
                    ),
                    STRATA[
                        PRIMARY_STRATUM
                    ],
                )
            )

            top = sub[
                sub[
                    "width"
                ].to_numpy(
                    dtype=float
                )
                >= cut30
            ]

            spearman_rows.append(
                {
                    "instrument": (
                        instrument
                    ),
                    "horizon": (
                        horizon
                    ),
                    "all_path_future_spearman": (
                        spearman(
                            sub[
                                "r15"
                            ],
                            sub[
                                ret_col
                            ],
                        )
                    ),
                    "top30_path_future_spearman": (
                        spearman(
                            top[
                                "r15"
                            ],
                            top[
                                ret_col
                            ],
                        )
                    ),
                    "n_all": int(
                        len(
                            sub
                        )
                    ),
                    "n_top30": int(
                        len(
                            top
                        )
                    ),
                }
            )

        mech = mechanism_table(
            grid,
            horizon=(
                PRIMARY_HORIZON
            ),
        )

        if len(
            mech
        ):

            mech.insert(
                0,
                "instrument",
                instrument,
            )

            mech_rows.append(
                mech
            )

        # ---- predictive models ----

        for (
            horizon
        ) in HORIZON_BARS:

            for (
                block,
                features,
            ) in (
                FEATURE_BLOCKS.items()
            ):

                for (
                    stratum,
                    q,
                ) in (
                    STRATA.items()
                ):

                    for (
                        learner
                    ) in LEARNERS:

                        (
                            metrics,
                            pred_frame,
                        ) = run_model(
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
                            learner=(
                                learner
                            ),
                        )

                        rows.append(
                            {
                                "instrument": (
                                    instrument
                                ),
                                "horizon": (
                                    horizon
                                ),
                                "features": (
                                    block
                                ),
                                "stratum": (
                                    stratum
                                ),
                                "learner": (
                                    learner
                                ),
                                **metrics,
                            }
                        )

                        # Non-overlap robustness on the primary
                        # feature block and stratum.
                        if (
                            block
                            == "P2"
                            and stratum
                            == (
                                PRIMARY_STRATUM
                            )
                        ):

                            mod = (
                                NONOVERLAP_MOD.get(
                                    horizon
                                )
                            )

                            if (
                                mod
                                is not None
                                and len(
                                    pred_frame
                                )
                            ):

                                p_train = (
                                    float(
                                        pred_frame[
                                            "y"
                                        ].mean()
                                    )
                                )

                                for (
                                    off
                                ) in range(
                                    mod
                                ):

                                    s = pred_frame[
                                        pred_frame[
                                            "row_id"
                                        ]
                                        % mod
                                        == off
                                    ]

                                    if (
                                        len(
                                            s
                                        )
                                        < 50
                                        or s[
                                            "y"
                                        ].nunique()
                                        < 2
                                    ):
                                        continue

                                    m = binary_metrics(
                                        s[
                                            "y"
                                        ].to_numpy(
                                            dtype=(
                                                int
                                            )
                                        ),
                                        s[
                                            "p"
                                        ].to_numpy(
                                            dtype=(
                                                float
                                            )
                                        ),
                                        p_train,
                                        s[
                                            "fut_ret"
                                        ].to_numpy(
                                            dtype=(
                                                float
                                            )
                                        ),
                                        s[
                                            "mfe_up"
                                        ].to_numpy(
                                            dtype=(
                                                float
                                            )
                                        ),
                                    )

                                    if (
                                        not m
                                    ):
                                        continue

                                    nonoverlap_rows.append(
                                        {
                                            "instrument": (
                                                instrument
                                            ),
                                            "horizon": (
                                                horizon
                                            ),
                                            "features": (
                                                block
                                            ),
                                            "stratum": (
                                                stratum
                                            ),
                                            "learner": (
                                                learner
                                            ),
                                            "offset": int(
                                                off
                                            ),
                                            "mod": int(
                                                mod
                                            ),
                                            "n": int(
                                                m[
                                                    "n"
                                                ]
                                            ),
                                            "auc": (
                                                float(
                                                    m[
                                                        "auc"
                                                    ]
                                                )
                                            ),
                                            "brier_skill": (
                                                float(
                                                    m[
                                                        "brier_skill"
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

            print(
                f"    {horizon} done"
            )

    results = pd.DataFrame(
        rows
    )

    results.to_csv(
        OUT
        / "direction_v1_by_instrument.csv",
        index=False,
    )

    pd.DataFrame(
        nonoverlap_rows
    ).to_csv(
        OUT
        / "direction_v1_nonoverlap.csv",
        index=False,
    )

    pd.DataFrame(
        coverage_rows
    ).to_csv(
        OUT
        / "continuity_coverage.csv",
        index=False,
    )

    pd.DataFrame(
        verify_rows
    ).to_csv(
        OUT
        / "quantile_state_verification.csv",
        index=False,
    )

    pd.DataFrame(
        spearman_rows
    ).to_csv(
        OUT
        / "path_future_spearman.csv",
        index=False,
    )

    if mech_rows:
        pd.concat(
            mech_rows,
            ignore_index=(
                True
            ),
        ).to_csv(
            OUT
            / "path_state_mechanism.csv",
            index=False,
        )

    # ========================================================
    # Cross summary
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

        for keys, g in g0.groupby(
            [
                "horizon",
                "features",
                "stratum",
                "learner",
            ],
            observed=(
                True
            ),
        ):

            (
                horizon,
                block,
                stratum,
                learner,
            ) = keys

            row = {
                "group": (
                    group_name
                ),
                "horizon": (
                    horizon
                ),
                "features": (
                    block
                ),
                "stratum": (
                    stratum
                ),
                "learner": (
                    learner
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
        / "direction_v1_cross_summary.csv",
        index=False,
    )

    config = {
        "purpose": (
            "Test whether recent price path predicts "
            "future direction, conditioned on quantile "
            "opportunity state."
        ),
        "primary_hypothesis": (
            f"{PRIMARY_HORIZON} + "
            f"{PRIMARY_STRATUM}"
        ),
        "quantile_source": {
            "horizon": (
                QUANT_HORIZON
            ),
            "feature_set": (
                QUANT_FEATURE_SET
            ),
            "model": QUANT_MODEL,
            "role": (
                "opportunity gate only (width)"
            ),
            "refit": False,
        },
        "continuity_correction": (
            "target and path windows must be "
            "continuous in calendar time; V0 used "
            "array position and crossed session "
            "gaps on 9.4% of H15 rows and 23.5% "
            "of H30 rows"
        ),
        "causality": (
            "at decision time t the last closed "
            "bar is [t-5m, t); all path features "
            "use bars up to that one; the target "
            "starts at t using the open of the bar "
            "beginning at t"
        ),
        "feature_blocks": (
            FEATURE_BLOCKS
        ),
        "learners": {
            "logistic": (
                "LogisticRegression("
                "penalty=l2, C=1.0, max_iter=2000)"
            ),
            "gbr": (
                "GradientBoostingClassifier with "
                "the locked shallow GBR "
                "hyper-parameters: n_estimators="
                "150, learning_rate=0.03, "
                "max_depth=2, min_samples_leaf=30, "
                "subsample=0.80"
            ),
        },
        "horizons_bars": (
            HORIZON_BARS
        ),
        "strata": STRATA,
        "strata_thresholds": (
            "computed on TRAIN rows per fold only"
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
        "no_rv": True,
        "no_semivariance": True,
        "no_volume": True,
        "no_oi": True,
        "no_transaction_data": True,
        "no_smc": True,
        "no_dsa": True,
        "no_momentum": True,
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
            FEATURE_BLOCKS
        )
        * len(STRATA)
        * len(LEARNERS)
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
            verify_rows
        ),
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
    # Report
    # ========================================================

    lines = [
        "# Direction V1",
        "",
        f"Primary hypothesis: "
        f"{PRIMARY_HORIZON} + "
        f"{PRIMARY_STRATUM}",
        "",
        "## Cross summary (ALL_8, H15)",
        "",
        "| features | stratum | learner | "
        "median AUC | median Brier | pos | "
        "median spread |",
        "|---|---|---|---:|---:|---:|---:|",
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
            "features",
            "stratum",
            "learner",
        ]
    ).iterrows():

        lines.append(
            f"| {r['features']} "
            f"| {r['stratum']} "
            f"| {r['learner']} "
            f"| {r['median_auc']:.4f} "
            f"| "
            f"{r['median_brier_skill']:+.5f} "
            f"| "
            f"{r['positive_share_brier_skill']:.2f} "
            f"| "
            f"{r['median_return_spread']:+.3e} "
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
        "features",
        "stratum",
        "learner",
        "median_auc",
        "median_brier_skill",
        "positive_share_brier_skill",
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
        "\nDIRECTION_V1_PASS"
    )


if __name__ == "__main__":
    main()
