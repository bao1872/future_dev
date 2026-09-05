#!/usr/bin/env python3
"""Direction V2 -- state-dependent direction.

V0: quantile distribution geometry carries no direction information.
V1: recent price path carries only a weak signal, concentrated at
    H5 rather than at the pre-registered H15 + TOP30.

V1 therefore reframes the question. The literature does not claim
that intraday momentum is a fixed effect; it claims the effect is
STATE DEPENDENT (volatility, volume, liquidity). V2 tests exactly
that with the data already in hand:

    the same past 5-15m move sometimes continues and sometimes
    reverses -- can the current REALIZED state tell them apart?

Architecture change
-------------------
High opportunity did NOT raise direction AUC in V0 or V1, so
Quantile stops being a "predictability gate". It becomes the payoff
gate:

    Direction model -> WHICH WAY      (trained on ALL observations)
    Quantile width  -> IS IT WORTH TRADING

The primary question is therefore not "is AUC higher in TOP30", but
"does the same direction score translate into a larger future return
spread when opportunity is high".

Pre-registered confirmatory primary
-----------------------------------
    H5, ALL observations, S3 vs S0

Judged on median AUC / Brier / logloss improvement and positive
instrument share. No arbitrary AUC > 0.55 threshold.

Feature blocks (all continuous-window, all causal)
--------------------------------------------------
    S0  r5, r10, r15, r30
    S1  + RV15, RV30, RV60, rv_acceleration
    S2  + RS_up_30, RS_down_30, semivar_balance_30
    S3  + r5 x rv_acceleration, r15 x rv_acceleration,
          r15 x semivar_balance

Horizons
--------
    H5  primary   (1 x 5m bar)
    H15 secondary (3 x 5m bars)
    H30 dropped: V1 showed it decaying and it adds nothing to a
    5m-15m execution cycle.

Fair comparison
---------------
S1-S3 need RV60, which requires 12 continuous bars and is available
on fewer rows than r30. Evaluating each block on its own available
rows would make S0 and S3 differ by SAMPLE as well as by features.
All blocks are therefore evaluated on the single common row set
where every S3 feature and the target are available.
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
)

from research.run_quantile_rebaseline import (  # noqa: E402
    FEATURE_SETS,
    make_folds,
)

from research.run_direction_v0 import (  # noqa: E402
    build_execution_grid,
    quantile_state,
)

from research.run_direction_v1 import (  # noqa: E402
    add_targets,
    continuity_prefix,
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
    / "direction_v2"
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
}

PRIMARY_HORIZON = "H5"

NONOVERLAP_MOD = {
    "H15": 3,
}

STRATA = {
    "ALL": None,
    "TOP50": 0.50,
    "TOP30": 0.70,
    "TOP20": 0.80,
}

LEARNERS = (
    "logistic",
    "gbr",
)

FIVE_NS = 5 * 60 * 1_000_000_000

EPS = 1e-12

PROB_CLIP = 1e-6

SPREAD_QUANTILE = 0.20

RV_STATE_BINS = 3


# ============================================================
# Feature construction
# ============================================================

def add_state_features(
    grid: pd.DataFrame,
    five: pd.DataFrame,
    cs: np.ndarray,
) -> pd.DataFrame:
    """Price path + realized state features.

    All windows are continuous in calendar time and end at the last
    CLOSED bar k = row_id - 1, so nothing from the decision bar or
    the future enters a feature.
    """

    f = five.sort_values(
        "bar_start_time"
    ).reset_index(
        drop=True
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

    # ---- price path returns ----

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

        start = (
            k
            - n
            + 1
        )

        cont = (
            window_ok(
                cs,
                start,
                n,
                n5,
            )
            & usable
            & (
                start >= 1
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

    # ---- realized variance ----
    #
    # Cumulative-sum formulation: the sum of `arr` over the
    # continuous window [start, k] is cs[k+1] - cs[start].

    sq = np.nan_to_num(
        ret * ret,
        nan=0.0,
    )

    for (
        label,
        n,
        arr,
    ) in (
        (
            "RV15",
            3,
            sq,
        ),
        (
            "RV30",
            6,
            sq,
        ),
        (
            "RV60",
            12,
            sq,
        ),
    ):

        data[
            label
        ] = window_sum(
            arr,
            start=(
                k
                - n
                + 1
            ),
            end=k,
            n=n,
            n_bars=n5,
            cs=cs,
            usable=usable,
            n_rows=n_rows,
        )

    # ---- realized semivariance over 30m ----

    up_sq = np.where(
        ret > 0,
        sq,
        0.0,
    )

    down_sq = np.where(
        ret < 0,
        sq,
        0.0,
    )

    for (
        label,
        arr,
    ) in (
        (
            "RS_up_30",
            up_sq,
        ),
        (
            "RS_down_30",
            down_sq,
        ),
    ):

        data[
            label
        ] = window_sum(
            arr,
            start=(
                k
                - 6
                + 1
            ),
            end=k,
            n=6,
            n_bars=n5,
            cs=cs,
            usable=usable,
            n_rows=n_rows,
        )

    rs_up = data[
        "RS_up_30"
    ]

    rs_dn = data[
        "RS_down_30"
    ]

    data[
        "semivar_balance_30"
    ] = (
        (
            rs_up
            - rs_dn
        )
        / (
            rs_up
            + rs_dn
            + EPS
        )
    )

    # Short-window variance rate over the long-window rate.
    data[
        "rv_acceleration"
    ] = (
        (
            data[
                "RV15"
            ]
            / 3.0
        )
        / (
            data[
                "RV60"
            ]
            / 12.0
            + EPS
        )
    )

    # ---- interactions ----

    data[
        "r5_x_rv_acc"
    ] = (
        data[
            "r5"
        ]
        * data[
            "rv_acceleration"
        ]
    )

    data[
        "r15_x_rv_acc"
    ] = (
        data[
            "r15"
        ]
        * data[
            "rv_acceleration"
        ]
    )

    data[
        "r15_x_semivar"
    ] = (
        data[
            "r15"
        ]
        * data[
            "semivar_balance_30"
        ]
    )

    for key, values in (
        data.items()
    ):

        grid[
            key
        ] = values

    return grid


def window_sum(
    arr: np.ndarray,
    *,
    start: np.ndarray,
    end: np.ndarray,
    n: int,
    n_bars: int,
    cs: np.ndarray,
    usable: np.ndarray,
    n_rows: int,
) -> np.ndarray:
    """Sum `arr` over the continuous window [start, end].

    Vectorized with a cumulative sum. The window is used only when
    it fits, holds exactly n bars, has no session gap, and starts
    at a bar that has a valid predecessor return (start >= 1).
    """

    out = np.full(
        n_rows,
        np.nan,
    )

    contiguous = (
        window_ok(
            cs,
            start,
            n,
            n_bars,
        )
        & usable
        & (
            start >= 1
        )
    )

    idx = np.where(
        contiguous
    )[
        0
    ]

    if len(
        idx
    ):

        cs_arr = np.concatenate(
            [
                [0.0],
                np.cumsum(
                    arr
                ),
            ]
        )

        a = start[
            idx
        ]

        b = end[
            idx
        ]

        out[
            idx
        ] = (
            cs_arr[
                b + 1
            ]
            - cs_arr[a]
        )

    return out


def window_ok(
    cs: np.ndarray,
    start: np.ndarray,
    n: int,
    n_bars: int,
) -> np.ndarray:
    """[start, start+n-1] has no session gap."""

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

        ok[
            idx
        ] = (
            cs[
                j + n - 1
            ]
            - cs[j]
        ) == 0

    return ok


FEATURE_BLOCKS = {
    "S0": [
        "r5",
        "r10",
        "r15",
        "r30",
    ],
    "S1": [
        "r5",
        "r10",
        "r15",
        "r30",
        "RV15",
        "RV30",
        "RV60",
        "rv_acceleration",
    ],
    "S2": [
        "r5",
        "r10",
        "r15",
        "r30",
        "RV15",
        "RV30",
        "RV60",
        "rv_acceleration",
        "RS_up_30",
        "RS_down_30",
        "semivar_balance_30",
    ],
    "S3": [
        "r5",
        "r10",
        "r15",
        "r30",
        "RV15",
        "RV30",
        "RV60",
        "rv_acceleration",
        "RS_up_30",
        "RS_down_30",
        "semivar_balance_30",
        "r5_x_rv_acc",
        "r15_x_rv_acc",
        "r15_x_semivar",
    ],
}

# Every block is evaluated on this feature set's available rows,
# so the blocks differ by features only, never by sample.
COMMON_FEATURES = FEATURE_BLOCKS[
    "S3"
]


# ============================================================
# Learner
# ============================================================

def make_learner(
    name: str,
):

    if name == "logistic":

        return (
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
# Metrics
# ============================================================

def binary_metrics(
    y: np.ndarray,
    p: np.ndarray,
    p_train: float,
    fut_ret: np.ndarray,
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
# One run: train on ALL, evaluate by opportunity stratum
# ============================================================

def run_model(
    work: pd.DataFrame,
    *,
    horizon: str,
    feature_names: list[str],
    learner: str,
) -> pd.DataFrame:
    """Walk-forward OOS on ALL rows.

    Returns an OOS frame carrying the fold-specific train width
    thresholds, so strata can be applied afterwards without
    retraining.
    """

    n_bars = (
        HORIZON_BARS[
            horizon
        ]
    )

    ret_col = (
        f"fut_ret_{horizon}"
    )

    n = len(
        work
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

    cuts: dict[
        int, dict[
            str, float
        ]
    ] = {}

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
        ):
            continue

        # Opportunity thresholds from TRAIN rows of this fold.
        train_width = (
            train[
                "width"
            ].to_numpy(
                dtype=float
            )
        )

        cuts[
            fold[
                "fold"
            ]
        ] = {
            name: (
                float(
                    np.quantile(
                        train_width,
                        q,
                    )
                )
                if q
                is not None
                else -np.inf
            )
            for name, q in (
                STRATA.items()
            )
        }

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

        model = make_learner(
            learner
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
            "fold": fold_id[
                oos
            ],
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
            "width": work.loc[
                oos,
                "width"
            ].to_numpy(),
            "r15": work.loc[
                oos,
                "r15"
            ].to_numpy(),
            "RV30": work.loc[
                oos,
                "RV30"
            ].to_numpy(),
        }
    )

    out[
        "width_cut"
    ] = out[
        "fold"
    ].map(
        lambda f: cuts.get(
            f, {}
        )
    )

    return out


def metrics_by_stratum(
    pred_frame: pd.DataFrame,
) -> list[dict]:

    rows = []

    for (
        stratum
    ) in STRATA:

        if (
            stratum
            == "ALL"
        ):

            sub = (
                pred_frame
            )

        else:

            sub = (
                pred_frame[
                    pred_frame.apply(
                        lambda r: (
                            r[
                                "width"
                            ]
                            >= r[
                                "width_cut"
                            ][
                                stratum
                            ]
                        ),
                        axis=1,
                    )
                ]
            )

        if (
            len(sub)
            < 50
            or sub[
                "y"
            ].nunique()
            < 2
        ):
            continue

        p_train = float(
            sub[
                "y"
            ].mean()
        )

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
        )

        if not m:
            continue

        rows.append(
            {
                "stratum": (
                    stratum
                ),
                **m,
            }
        )

    return rows


def rv_state_mechanism(
    pred_frame: pd.DataFrame,
    work: pd.DataFrame,
) -> list[dict]:
    """Spearman(r15, future return) inside RV terciles.

    RV thresholds are computed on TRAIN rows of each fold, then
    applied to that fold's OOS rows.
    """

    n_bars = (
        HORIZON_BARS[
            PRIMARY_HORIZON
        ]
    )

    ret_col = (
        f"fut_ret_{PRIMARY_HORIZON}"
    )

    n = len(
        work
    )

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
        ):
            continue

        edges = np.quantile(
            train[
                "RV30"
            ].to_numpy(
                dtype=float
            ),
            [
                1.0
                / RV_STATE_BINS,
                2.0
                / RV_STATE_BINS,
            ],
        )

        records.append(
            pd.DataFrame(
                {
                    "rv_state": np.digitize(
                        test[
                            "RV30"
                        ].to_numpy(
                            dtype=float
                        ),
                        edges,
                    ),
                    "r15": test[
                        "r15"
                    ].to_numpy(
                        dtype=float
                    ),
                    "fut": test[
                        ret_col
                    ].to_numpy(
                        dtype=float
                    ),
                }
            )
        )

    if not records:
        return []

    all_rows = pd.concat(
        records,
        ignore_index=(
            True
        ),
    )

    out = []

    for state, g in (
        all_rows.groupby(
            "rv_state"
        )
    ):

        out.append(
            {
                "rv_state": int(
                    state
                ),
                "rv_state_label": (
                    [
                        "low",
                        "mid",
                        "high",
                    ][
                        state
                    ]
                    if state
                    < 3
                    else "?"
                ),
                "n": int(
                    len(
                        g
                    )
                ),
                "path_future_spearman": (
                    spearman(
                        g[
                            "r15"
                        ],
                        g[
                            "fut"
                        ],
                    )
                ),
            }
        )

    return out


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

        for horizon in HORIZON_BARS:

            ret_col = (
                f"fut_ret_{horizon}"
            )

            # Common rows: every S3 feature and the target.
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

            cov[
                f"{horizon}_common_rows"
            ] = int(
                len(
                    work
                )
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

            if (
                len(
                    work
                )
                < 500
            ):
                continue

            for (
                block,
                features,
            ) in (
                FEATURE_BLOCKS.items()
            ):

                for (
                    learner
                ) in LEARNERS:

                    pred_frame = (
                        run_model(
                            work,
                            horizon=(
                                horizon
                            ),
                            feature_names=(
                                features
                            ),
                            learner=(
                                learner
                            ),
                        )
                    )

                    for (
                        entry
                    ) in metrics_by_stratum(
                        pred_frame
                    ):

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
                                "learner": (
                                    learner
                                ),
                                **entry,
                            }
                        )

                    # Non-overlap robustness, primary block only.
                    if (
                        block
                        == "S3"
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
                                        "learner": (
                                            learner
                                        ),
                                        "offset": int(
                                            off
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
                                    }
                                )

            # Mechanism check on the primary horizon.
            if (
                horizon
                == PRIMARY_HORIZON
            ):

                for (
                    entry
                ) in rv_state_mechanism(
                    pred_frame,
                    work,
                ):

                    mech_rows.append(
                        {
                            "instrument": (
                                instrument
                            ),
                            **entry,
                        }
                    )

        coverage_rows.append(
            cov
        )

        print(
            f"  H5 common={cov.get('H5_common_rows')}"
            f"  H15 common="
            f"{cov.get('H15_common_rows')}"
        )

    results = pd.DataFrame(
        rows
    )

    results.to_csv(
        OUT
        / "direction_v2_by_instrument.csv",
        index=False,
    )

    pd.DataFrame(
        nonoverlap_rows
    ).to_csv(
        OUT
        / "direction_v2_nonoverlap.csv",
        index=False,
    )

    pd.DataFrame(
        coverage_rows
    ).to_csv(
        OUT
        / "v2_coverage.csv",
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
        mech_rows
    ).to_csv(
        OUT
        / "rv_state_mechanism.csv",
        index=False,
    )

    # ========================================================
    # S3 vs S0 confirmatory comparison
    # ========================================================

    delta_rows = []

    for (
        instrument
    ) in INSTRUMENTS:

        for (
            horizon
        ) in HORIZON_BARS:

            for (
                learner
            ) in LEARNERS:

                base = (
                    results[
                        (
                            results[
                                "instrument"
                            ]
                            == instrument
                        )
                        & (
                            results[
                                "horizon"
                            ]
                            == horizon
                        )
                        & (
                            results[
                                "learner"
                            ]
                            == learner
                        )
                        & (
                            results[
                                "stratum"
                            ]
                            == "ALL"
                        )
                    ].set_index(
                        "features"
                    )
                )

                if (
                    "S0"
                    not in base.index
                    or "S3"
                    not in base.index
                ):
                    continue

                delta_rows.append(
                    {
                        "instrument": (
                            instrument
                        ),
                        "horizon": (
                            horizon
                        ),
                        "learner": (
                            learner
                        ),
                        "auc_S0": float(
                            base.loc[
                                "S0",
                                "auc",
                            ]
                        ),
                        "auc_S3": float(
                            base.loc[
                                "S3",
                                "auc",
                            ]
                        ),
                        "delta_auc": float(
                            base.loc[
                                "S3",
                                "auc",
                            ]
                            - base.loc[
                                "S0",
                                "auc",
                            ]
                        ),
                        "delta_brier_skill": float(
                            base.loc[
                                "S3",
                                "brier_skill",
                            ]
                            - base.loc[
                                "S0",
                                "brier_skill",
                            ]
                        ),
                        "delta_logloss_skill": float(
                            base.loc[
                                "S3",
                                "logloss_skill",
                            ]
                            - base.loc[
                                "S0",
                                "logloss_skill",
                            ]
                        ),
                    }
                )

    deltas = pd.DataFrame(
        delta_rows
    )

    deltas.to_csv(
        OUT
        / "s3_vs_s0_delta.csv",
        index=False,
    )

    # ========================================================
    # Cross-instrument summary
    # ========================================================

    summary_rows = []

    metrics = [
        "auc",
        "brier_skill",
        "logloss_skill",
        "return_spread",
    ]

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
        / "direction_v2_cross_summary.csv",
        index=False,
    )

    config = {
        "purpose": (
            "Test whether realized volatility state "
            "conditions the continuation / reversal "
            "relation."
        ),
        "pre_registered_primary": (
            f"{PRIMARY_HORIZON}, ALL "
            "observations, S3 vs S0"
        ),
        "primary_criterion": (
            "median AUC / Brier / logloss "
            "improvement and positive instrument "
            "share; no arbitrary AUC threshold"
        ),
        "architecture": (
            "Direction model trained on ALL rows "
            "= WHICH WAY; Quantile width = IS IT "
            "WORTH TRADING (payoff gate, not "
            "predictability gate)"
        ),
        "horizons_bars": (
            HORIZON_BARS
        ),
        "dropped_horizon": (
            "H30: V1 showed decay and it adds "
            "nothing to a 5m-15m cycle"
        ),
        "feature_blocks": (
            FEATURE_BLOCKS
        ),
        "fair_comparison": (
            "all blocks evaluated on the common "
            "row set where every S3 feature and "
            "the target exist, so blocks differ "
            "by features only"
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
                "width only, for post-hoc "
                "opportunity strata"
            ),
            "refit": False,
        },
        "continuity": (
            "every target and feature window must "
            "be continuous in calendar time"
        ),
        "causality": (
            "features end at the last closed bar "
            "k = row_id - 1; the target starts at "
            "t using the open of the decision bar"
        ),
        "learners": (
            "fixed L2 logistic; fixed shallow GBR "
            "(150 trees, lr 0.03, depth 2, "
            "min_samples_leaf 30, subsample 0.8)"
        ),
        "strata": STRATA,
        "nonoverlap_mod": (
            NONOVERLAP_MOD
        ),
        "no_volume": True,
        "no_oi": True,
        "no_transaction_data": True,
        "no_smc": True,
        "no_dsa": True,
        "no_momentum": True,
        "no_cross_market": True,
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
        "delta_row_count": int(
            len(
                deltas
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
        "# Direction V2",
        "",
        f"Pre-registered primary: "
        f"{PRIMARY_HORIZON}, ALL, S3 vs S0",
        "",
        "## Cross summary (ALL_8, H5)",
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
        "\nDIRECTION_V2_PASS"
    )


if __name__ == "__main__":
    main()
