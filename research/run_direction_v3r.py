#!/usr/bin/env python3
"""Direction V3R -- 16-instrument replication.

Not a new model. This applies the corrections found in the V2/V3
code audit to a single unified experiment, then asks whether the
one result that survived audit -- price + RV beating price only --
replicates from 8 instruments to 16.

Corrections carried in
----------------------
1. Brier / logloss baseline. V2/V3 used the OOS test labels as the
   naive benchmark, which leaks. Here every OOS row carries its
   own fold's TRAIN UP rate as p_baseline, and skill is measured
   against that. AUC, predictions and return spread were unaffected
   by the old bug; Brier and logloss were not.

2. Calibration slope is removed. The old value came from an OLS fit
   of y on logit(p), which is not the standard logistic calibration
   regression, so those numbers were never interpretable.

3. Volume features are rebuilt. The old vol_surprise_30m/60m
   windows overlapped the 12-bar baseline by 5 and 11 bars, so they
   measured small differences between nearly identical windows
   rather than a participation surge. Only two clean features
   remain, both non-overlapping.

4. Session continuity. A return over n bars needs n+1 consecutive
   price points, because ret[j] uses close[j-1]. The old check only
   required the n bars inside the window, which let a window that
   began at a session open include a gap in its first return.

5. One master sample. Every block is evaluated on the identical
   rows, with the identical folds, so C0 -> C1 -> C2 -> C3 -> C4 is
   a clean nested comparison. V2 and V3 previously used different
   common samples and were not strictly comparable.

Blocks
------
    C0  r5, r10, r15, r30
    C1  C0 + RV15, RV30, RV60, rv_acceleration
    C2  C1 + vol_surprise_5m, vol_acceleration
    C3  C1 + rel_dOI_5m, rel_dOI_15m, rel_dOI_30m
    C4  C1 + clean volume + relative OI  (no manual interactions)

Pre-registered
--------------
    Test 1  C1 vs C0   H5 + ALL + GBR + 16 instruments
                       CONFIRMATORY
    Test 2  C2 vs C1   clean volume          secondary
    Test 3  C3 vs C1   relative OI           exploratory until
                                             the L8 OI tail audit
    Test 4  C4 vs C1   full participation    secondary

Return spread is computed INSIDE each fold (top 20% / bottom 20% of
that fold's predictions) and then aggregated across folds, so
probability-level drift between folds cannot manufacture it.

Groups reported: ALL_16, OLD_8, NEW_8. The new 8 never took part in
any earlier stage, so agreement between OLD_8 and NEW_8 is the
first genuine out-of-universe replication available here.
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

from research.run_direction_v0 import (  # noqa: E402
    build_execution_grid,
    quantile_state,
)

from research.build_pytdx_panel import (  # noqa: E402
    aggregate_15m,
    build_features,
    build_targets,
)

from research.run_quantile_rebaseline import (  # noqa: E402
    FEATURE_SETS,
    make_folds,
)


SRC_5M = (
    ROOT
    / "research"
    / "exports"
    / "v3r_5m"
)

OUT = (
    ROOT
    / "research"
    / "exports"
    / "direction_v3r"
)

INSTRUMENTS = (
    "AG",
    "AU",
    "CU",
    "AL",
    "SN",
    "NI",
    "RB",
    "I",
    "SC",
    "RU",
    "MA",
    "TA",
    "M",
    "P",
    "CF",
    "LC",
)

OLD_8 = (
    "AG",
    "CU",
    "AL",
    "SN",
    "I",
    "SC",
    "M",
    "CF",
)

NEW_8 = tuple(
    i
    for i in INSTRUMENTS
    if i not in OLD_8
)

GROUPS = {
    "ALL_16": INSTRUMENTS,
    "OLD_8": OLD_8,
    "NEW_8": NEW_8,
}

# Quantile layer stays frozen. Used only for the opportunity width.
QUANT_HORIZON = 4
QUANT_FEATURE_SET = "F1_VOL"
QUANT_MODEL = "gbr_quantile"

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
    "gbr",
    "logistic",
)

PRIMARY_LEARNER = "gbr"

FIVE_NS = 5 * 60 * 1_000_000_000

EPS = 1e-12

PROB_CLIP = 1e-6

SPREAD_QUANTILE = 0.20

VOL_BASE_BARS = 12


FEATURE_BLOCKS = {
    "C0": [
        "r5",
        "r10",
        "r15",
        "r30",
    ],
    "C1": [
        "r5",
        "r10",
        "r15",
        "r30",
        "RV15",
        "RV30",
        "RV60",
        "rv_acceleration",
    ],
    "C2": [
        "r5",
        "r10",
        "r15",
        "r30",
        "RV15",
        "RV30",
        "RV60",
        "rv_acceleration",
        "vol_surprise_5m",
        "vol_acceleration",
    ],
    "C3": [
        "r5",
        "r10",
        "r15",
        "r30",
        "RV15",
        "RV30",
        "RV60",
        "rv_acceleration",
        "rel_dOI_5m",
        "rel_dOI_15m",
        "rel_dOI_30m",
    ],
    "C4": [
        "r5",
        "r10",
        "r15",
        "r30",
        "RV15",
        "RV30",
        "RV60",
        "rv_acceleration",
        "vol_surprise_5m",
        "vol_acceleration",
        "rel_dOI_5m",
        "rel_dOI_15m",
        "rel_dOI_30m",
    ],
}

MASTER_FEATURES = FEATURE_BLOCKS[
    "C4"
]


# ============================================================
# Continuity
# ============================================================

def continuity_prefix(
    five: pd.DataFrame,
) -> np.ndarray:
    """cs[k] = number of gaps among bars [0, k-1]."""

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

    adjacent = np.concatenate(
        [
            np.diff(
                start_ns
            )
            == FIVE_NS,
            [False],
        ]
    )

    return np.concatenate(
        [
            [0],
            np.cumsum(
                (
                    ~adjacent
                ).astype(
                    np.int64
                )
            ),
        ]
    )


def span_ok(
    cs: np.ndarray,
    start: np.ndarray,
    n: int,
    n_bars: int,
) -> np.ndarray:
    """[start, start+n-1] within bounds and gap-free."""

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


def diff_window_ok(
    cs: np.ndarray,
    start: np.ndarray,
    n: int,
    n_bars: int,
) -> np.ndarray:
    """Gap-free with the PRECEDING bar as well.

    A return / RV window over n bars uses close[j-1] for every bar
    in it, so it needs n+1 consecutive price points. This is the
    continuity bug fix: the window [start-1, k] must be continuous,
    not just [start, k].
    """

    prev = (
        start
        - 1
    )

    return span_ok(
        cs,
        prev,
        n + 1,
        n_bars,
    ) & (
        prev >= 0
    )


def window_sum(
    arr: np.ndarray,
    *,
    start: np.ndarray,
    end: np.ndarray,
    n_rows: int,
    mask: np.ndarray,
) -> np.ndarray:
    """Sum arr over [start, end] where mask allows."""

    out = np.full(
        n_rows,
        np.nan,
    )

    idx = np.where(
        mask
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

        out[
            idx
        ] = (
            cs_arr[
                end[
                    idx
                ]
                + 1
            ]
            - cs_arr[
                start[
                    idx
                ]
            ]
        )

    return out


# ============================================================
# Features
# ============================================================

def add_features(
    grid: pd.DataFrame,
    five: pd.DataFrame,
    cs: np.ndarray,
) -> pd.DataFrame:

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

    vol = f[
        "trade"
    ].to_numpy(
        dtype=float
    )

    oi = f[
        "position"
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

    sq = np.nan_to_num(
        ret * ret,
        nan=0.0,
    )

    row_id = (
        grid[
            "row_id"
        ].to_numpy(
            dtype=int
        )
    )

    # Last closed bar at decision time t.
    k = (
        row_id
        - 1
    )

    n_rows = len(
        row_id
    )

    usable = k >= 1

    data: dict = {}

    # ---- price path returns (n+1 continuity) ----

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

        start = (
            k
            - n
            + 1
        )

        mask = (
            diff_window_ok(
                cs,
                start,
                n,
                n5,
            )
            & usable
        )

        out = np.full(
            n_rows,
            np.nan,
        )

        idx = np.where(
            mask
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

    # ---- realized variance (n+1 continuity) ----

    for (
        label,
        n,
    ) in (
        (
            "RV15",
            3,
        ),
        (
            "RV30",
            6,
        ),
        (
            "RV60",
            12,
        ),
    ):

        start = (
            k
            - n
            + 1
        )

        mask = (
            diff_window_ok(
                cs,
                start,
                n,
                n5,
            )
            & usable
        )

        data[
            label
        ] = window_sum(
            sq,
            start=start,
            end=k,
            n_rows=n_rows,
            mask=mask,
        )

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

    # ---- clean volume ----
    #
    # vol_surprise_5m : current bar volume against the median of
    #                   the previous 12 continuous bars. Zero
    #                   overlap with the baseline.
    #
    # vol_acceleration: mean of the last 3 bars against the mean of
    #                   the 9 bars before them. Disjoint windows.

    base_start = (
        k
        - VOL_BASE_BARS
    )

    base_mask = (
        span_ok(
            cs,
            base_start,
            VOL_BASE_BARS,
            n5,
        )
        & (
            base_start
            >= 0
        )
        & usable
    )

    baseline = np.full(
        n_rows,
        np.nan,
    )

    idx = np.where(
        base_mask
    )[
        0
    ]

    if len(
        idx
    ):

        windows = (
            np.lib
            .stride_tricks
            .sliding_window_view(
                vol,
                VOL_BASE_BARS,
            )[
                base_start[
                    idx
                ]
            ]
        )

        baseline[
            idx
        ] = np.median(
            windows,
            axis=1,
        )

    cur_mask = (
        span_ok(
            cs,
            base_start,
            VOL_BASE_BARS
            + 1,
            n5,
        )
        & (
            base_start
            >= 0
        )
        & usable
    )

    idx = np.where(
        cur_mask
    )[
        0
    ]

    data[
        "vol_surprise_5m"
    ] = np.where(
        cur_mask,
        vol[
            k
        ]
        / (
            baseline
            + EPS
        ),
        np.nan,
    )

    recent_mask = (
        span_ok(
            cs,
            k
            - 2,
            3,
            n5,
        )
        & usable
    )

    prior_mask = (
        span_ok(
            cs,
            k
            - 11,
            9,
            n5,
        )
        & usable
    )

    both = (
        recent_mask
        & prior_mask
    )

    recent = (
        window_sum(
            vol,
            start=(
                k
                - 2
            ),
            end=k,
            n_rows=n_rows,
            mask=recent_mask,
        )
        / 3.0
    )

    prior = (
        window_sum(
            vol,
            start=(
                k
                - 11
            ),
            end=(
                k
                - 3
            ),
            n_rows=n_rows,
            mask=prior_mask,
        )
        / 9.0
    )

    data[
        "vol_acceleration"
    ] = np.where(
        both,
        recent
        / (
            prior
            + EPS
        ),
        np.nan,
    )

    # ---- relative OI ----

    for (
        label,
        n,
    ) in (
        (
            "rel_dOI_5m",
            1,
        ),
        (
            "rel_dOI_15m",
            3,
        ),
        (
            "rel_dOI_30m",
            6,
        ),
    ):

        start = (
            k
            - n
            + 1
        )

        mask = (
            span_ok(
                cs,
                start,
                n,
                n5,
            )
            & (
                start >= 1
            )
            & usable
        )

        out = np.full(
            n_rows,
            np.nan,
        )

        idx = np.where(
            mask
        )[
            0
        ]

        if len(
            idx
        ):

            prev = oi[
                start[
                    idx
                ]
                - 1
            ]

            out[
                idx
            ] = (
                (
                    oi[
                        k[
                            idx
                        ]
                    ]
                    - prev
                )
                / (
                    np.abs(
                        prev
                    )
                    + EPS
                )
            )

        data[
            label
        ] = out

    for key, values in (
        data.items()
    ):

        grid[
            key
        ] = values

    return grid


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

    close = f[
        "close"
    ].to_numpy(
        dtype=float
    )

    row_id = (
        grid[
            "row_id"
        ].to_numpy(
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

        # A target uses only bars inside its own window.
        mask = (
            span_ok(
                cs,
                row_id,
                n,
                n5,
            )
        )

        idx = np.where(
            mask
        )[
            0
        ]

        if len(
            idx
        ):

            ret[
                idx
            ] = (
                log_close[
                    row_id[
                        idx
                    ]
                    + n
                    - 1
                ]
                - entry[
                    idx
                ]
            )

        grid[
            f"fut_ret_{name}"
        ] = ret

    return grid


# ============================================================
# Model
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


def fold_metrics(
    y: np.ndarray,
    p: np.ndarray,
    p_base: np.ndarray,
    fut_ret: np.ndarray,
) -> dict:
    """Metrics for one fold.

    p_base is per row and comes from that fold's TRAIN UP rate, so
    the naive benchmark never sees a test label. Return spread is
    computed inside this fold only.
    """

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

    b_safe = np.clip(
        p_base,
        PROB_CLIP,
        1.0
        - PROB_CLIP,
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
                b_safe
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
            b_safe,
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
        "n": int(
            len(
                y
            )
        ),
        "auc": float(
            roc_auc_score(
                y,
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


def run_model(
    work: pd.DataFrame,
    *,
    horizon: str,
    feature_names: list[str],
    learner: str,
) -> pd.DataFrame:
    """Walk-forward on ALL rows.

    Returns one row per OOS observation carrying p_model and the
    fold's TRAIN baseline probability, plus the fold's train width
    thresholds so opportunity strata are applied afterwards.
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

    base = np.full(
        n,
        np.nan,
    )

    fold_id = np.full(
        n,
        -1,
        dtype=int,
    )

    cuts: dict = {}

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

        p_train = float(
            train[
                "y"
            ].mean()
        )

        cuts[
            fold[
                "fold"
            ]
        ] = {
            name: (
                float(
                    np.quantile(
                        train[
                            "width"
                        ].to_numpy(
                            dtype=float
                        ),
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
            learner
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

        prob[
            te
        ] = model.predict_proba(
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

        # Baseline from TRAIN only.
        base[
            te
        ] = p_train

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
            "p_model": prob[
                oos
            ],
            "p_baseline": base[
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


def aggregate(
    pred: pd.DataFrame,
    *,
    instrument: str,
    horizon: str,
    block: str,
    learner: str,
) -> tuple[
    list[dict],
    list[dict],
]:
    """Pooled metrics per stratum, plus per-fold metrics."""

    pooled_rows = []
    fold_rows = []

    for (
        stratum
    ) in STRATA:

        if (
            stratum
            == "ALL"
        ):

            sub = pred

        else:

            sub = pred[
                pred.apply(
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

        if (
            len(sub)
            < 50
            or sub[
                "y"
            ].nunique()
            < 2
        ):
            continue

        m = fold_metrics(
            sub[
                "y"
            ].to_numpy(
                dtype=int
            ),
            sub[
                "p_model"
            ].to_numpy(
                dtype=float
            ),
            sub[
                "p_baseline"
            ].to_numpy(
                dtype=float
            ),
            sub[
                "fut_ret"
            ].to_numpy(
                dtype=float
            ),
        )

        if not m:
            continue

        pooled_rows.append(
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
                "stratum": (
                    stratum
                ),
                **m,
            }
        )

        # Per fold, same stratum.
        for (
            fid,
            g,
        ) in sub.groupby(
            "fold",
            observed=(
                True
            ),
        ):

            if (
                len(
                    g
                )
                < 50
                or g[
                    "y"
                ].nunique()
                < 2
            ):
                continue

            fm = fold_metrics(
                g[
                    "y"
                ].to_numpy(
                    dtype=int
                ),
                g[
                    "p_model"
                ].to_numpy(
                    dtype=float
                ),
                g[
                    "p_baseline"
                ].to_numpy(
                    dtype=float
                ),
                g[
                    "fut_ret"
                ].to_numpy(
                    dtype=float
                ),
            )

            if not fm:
                continue

            fold_rows.append(
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
                    "stratum": (
                        stratum
                    ),
                    "fold": int(
                        fid
                    ),
                    **fm,
                }
            )

    return (
        pooled_rows,
        fold_rows,
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

    quant_cols = (
        FEATURE_SETS[
            QUANT_FEATURE_SET
        ]
    )

    pooled_rows = []
    fold_rows = []
    coverage_rows = []
    oi_audit_rows = []

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
            "=" * 64,
            flush=True,
        )

        five = pd.read_csv(
            SRC_5M
            / f"{instrument}_5m.csv",
            parse_dates=[
                "bar_start_time",
                "bar_end_time",
                "availability_time",
                "trading_day",
                "tdx_datetime_raw",
            ],
        )

        # ---- 15m panel for the frozen quantile layer ----

        fifteen = (
            build_15m(
                five
            )
        )

        state = quantile_state(
            fifteen,
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

        grid = add_features(
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

        for (
            horizon
        ) in HORIZON_BARS:

            ret_col = (
                f"fut_ret_{horizon}"
            )

            work = grid.dropna(
                subset=(
                    [
                        ret_col
                    ]
                    + MASTER_FEATURES
                    + [
                        "width"
                    ]
                )
            ).reset_index(
                drop=True
            )

            cov[
                f"{horizon}_master_rows"
            ] = int(
                len(
                    work
                )
            )

            if (
                len(
                    work
                )
                < 500
            ):
                continue

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
                horizon
                == PRIMARY_HORIZON
            ):

                oi_audit_rows.append(
                    audit_oi_tail(
                        work,
                        instrument,
                    )
                )

            for (
                block,
                feats,
            ) in (
                FEATURE_BLOCKS.items()
            ):

                for (
                    learner
                ) in LEARNERS:

                    pred = run_model(
                        work,
                        horizon=(
                            horizon
                        ),
                        feature_names=(
                            feats
                        ),
                        learner=(
                            learner
                        ),
                    )

                    (
                        p_rows,
                        f_rows,
                    ) = aggregate(
                        pred,
                        instrument=(
                            instrument
                        ),
                        horizon=(
                            horizon
                        ),
                        block=block,
                        learner=(
                            learner
                        ),
                    )

                    pooled_rows.extend(
                        p_rows
                    )

                    fold_rows.extend(
                        f_rows
                    )

        coverage_rows.append(
            cov
        )

        print(
            f"  H5={cov.get('H5_master_rows')}"
            f"  H15="
            f"{cov.get('H15_master_rows')}",
            flush=True,
        )

    pooled = pd.DataFrame(
        pooled_rows
    )

    folds = pd.DataFrame(
        fold_rows
    )

    pooled.to_csv(
        OUT
        / "v3r_pooled_by_instrument.csv",
        index=False,
    )

    folds.to_csv(
        OUT
        / "v3r_fold_level.csv",
        index=False,
    )

    pd.DataFrame(
        coverage_rows
    ).to_csv(
        OUT
        / "v3r_coverage.csv",
        index=False,
    )

    pd.DataFrame(
        oi_audit_rows
    ).to_csv(
        OUT
        / "oi_tail_audit.csv",
        index=False,
    )

    # ---- nested deltas on the master sample ----

    delta_rows = []

    pairs = (
        (
            "C1",
            "C0",
        ),
        (
            "C2",
            "C1",
        ),
        (
            "C3",
            "C1",
        ),
        (
            "C4",
            "C1",
        ),
    )

    for (
        top,
        base,
    ) in pairs:

        for (
            horizon
        ) in HORIZON_BARS:

            for (
                learner
            ) in LEARNERS:

                a = (
                    folds[
                        (
                            folds[
                                "horizon"
                            ]
                            == (
                                horizon
                            )
                        )
                        & (
                            folds[
                                "learner"
                            ]
                            == (
                                learner
                            )
                        )
                        & (
                            folds[
                                "stratum"
                            ]
                            == (
                                "ALL"
                            )
                        )
                        & (
                            folds[
                                "features"
                            ]
                            == (
                                base
                            )
                        )
                    ].set_index(
                        [
                            "instrument",
                            "fold",
                        ]
                    )
                )

                b = (
                    folds[
                        (
                            folds[
                                "horizon"
                            ]
                            == (
                                horizon
                            )
                        )
                        & (
                            folds[
                                "learner"
                            ]
                            == (
                                learner
                            )
                        )
                        & (
                            folds[
                                "stratum"
                            ]
                            == (
                                "ALL"
                            )
                        )
                        & (
                            folds[
                                "features"
                            ]
                            == (
                                top
                            )
                        )
                    ].set_index(
                        [
                            "instrument",
                            "fold",
                        ]
                    )
                )

                joined = a.join(
                    b,
                    how="inner",
                    lsuffix=(
                        f"_{base}"
                    ),
                    rsuffix=(
                        f"_{top}"
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

                    d = (
                        joined[
                            f"{metric}_{top}"
                        ]
                        - joined[
                            f"{metric}_{base}"
                        ]
                    ).dropna()

                    if (
                        len(
                            d
                        )
                        == 0
                    ):
                        continue

                    delta_rows.append(
                        {
                            "comparison": (
                                f"{top}_vs_{base}"
                            ),
                            "horizon": (
                                horizon
                            ),
                            "learner": (
                                learner
                            ),
                            "metric": (
                                metric
                            ),
                            "pairs": int(
                                len(
                                    d
                                )
                            ),
                            "positive": int(
                                (
                                    d
                                    > 0
                                ).sum()
                            ),
                            "positive_share": float(
                                (
                                    d
                                    > 0
                                ).mean()
                            ),
                            "median_delta": float(
                                d.median()
                            ),
                        }
                    )

    deltas = pd.DataFrame(
        delta_rows
    )

    deltas.to_csv(
        OUT
        / "v3r_nested_deltas.csv",
        index=False,
    )

    # ---- payoff gate: per-fold spread TOP20 - ALL ----

    gate_rows = []

    for (
        horizon
    ) in HORIZON_BARS:

        for (
            learner
        ) in LEARNERS:

            for (
                block
            ) in (
                "C1",
                "C4",
            ):

                a = (
                    folds[
                        (
                            folds[
                                "horizon"
                            ]
                            == (
                                horizon
                            )
                        )
                        & (
                            folds[
                                "learner"
                            ]
                            == (
                                learner
                            )
                        )
                        & (
                            folds[
                                "features"
                            ]
                            == (
                                block
                            )
                        )
                        & (
                            folds[
                                "stratum"
                            ]
                            == (
                                "ALL"
                            )
                        )
                    ].set_index(
                        [
                            "instrument",
                            "fold",
                        ]
                    )[
                        "return_spread"
                    ]
                )

                b = (
                    folds[
                        (
                            folds[
                                "horizon"
                            ]
                            == (
                                horizon
                            )
                        )
                        & (
                            folds[
                                "learner"
                            ]
                            == (
                                learner
                            )
                        )
                        & (
                            folds[
                                "features"
                            ]
                            == (
                                block
                            )
                        )
                        & (
                            folds[
                                "stratum"
                            ]
                            == (
                                "TOP20"
                            )
                        )
                    ].set_index(
                        [
                            "instrument",
                            "fold",
                        ]
                    )[
                        "return_spread"
                    ]
                )

                joined = pd.concat(
                    [
                        a.rename(
                            "all_spread"
                        ),
                        b.rename(
                            "top20_spread"
                        ),
                    ],
                    axis=1,
                ).dropna()

                if (
                    len(
                        joined
                    )
                    == 0
                ):
                    continue

                d = (
                    joined[
                        "top20_spread"
                    ]
                    - joined[
                        "all_spread"
                    ]
                )

                gate_rows.append(
                    {
                        "horizon": (
                            horizon
                        ),
                        "learner": (
                            learner
                        ),
                        "features": (
                            block
                        ),
                        "pairs": int(
                            len(
                                d
                            )
                        ),
                        "positive": int(
                            (
                                d
                                > 0
                            ).sum()
                        ),
                        "positive_share": float(
                            (
                                d
                                > 0
                            ).mean()
                        ),
                        "median_delta": float(
                            d.median()
                        ),
                    }
                )

    gate = pd.DataFrame(
        gate_rows
    )

    gate.to_csv(
        OUT
        / "v3r_payoff_gate.csv",
        index=False,
    )

    # ---- group summaries ----

    summary_rows = []

    for (
        group_name,
        members,
    ) in GROUPS.items():

        g0 = pooled[
            pooled[
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

            for (
                metric
            ) in (
                "auc",
                "brier_skill",
                "logloss_skill",
                "return_spread",
            ):

                v = pd.to_numeric(
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
                        v.median()
                    )
                    if len(
                        v
                    )
                    else np.nan
                )

                row[
                    f"positive_share_{metric}"
                ] = positive_share(
                    v
                )

            summary_rows.append(
                row
            )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUT
        / "v3r_group_summary.csv",
        index=False,
    )

    config = {
        "purpose": (
            "Replicate the audited price + RV result "
            "on 16 instruments with the V2/V3 defects "
            "fixed."
        ),
        "corrections": [
            "Brier/logloss baseline is the fold TRAIN "
            "UP rate, never OOS labels",
            "calibration slope removed (was an OLS "
            "fit, not logistic calibration)",
            "volume features rebuilt non-overlapping",
            "continuity requires n+1 price points for "
            "an n-bar return or RV window",
            "one master sample and one fold set for "
            "every block",
        ],
        "pre_registered_primary": (
            f"Test 1: C1 vs C0, {PRIMARY_HORIZON} "
            f"+ ALL + {PRIMARY_LEARNER} + 16 "
            f"instruments"
        ),
        "blocks": FEATURE_BLOCKS,
        "horizons_bars": (
            HORIZON_BARS
        ),
        "learners": list(
            LEARNERS
        ),
        "groups": {
            k: list(v)
            for k, v in (
                GROUPS.items()
            )
        },
        "quantile": {
            "horizon": (
                QUANT_HORIZON
            ),
            "feature_set": (
                QUANT_FEATURE_SET
            ),
            "model": (
                QUANT_MODEL
            ),
            "refit": False,
            "role": (
                "width only, frozen"
            ),
        },
        "spread_is_fold_local": True,
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

    if np.isinf(
        pooled.select_dtypes(
            include=[
                np.number
            ]
        )
        .to_numpy(
            dtype=float
        )
    ).any():
        raise RuntimeError(
            "output contains +/-inf"
        )

    (
        OUT
        / "validation.json"
    ).write_text(
        json.dumps(
            {
                "status": "PASS",
                "instrument_count": (
                    len(
                        INSTRUMENTS
                    )
                ),
                "pooled_rows": int(
                    len(
                        pooled
                    )
                ),
                "fold_rows": int(
                    len(
                        folds
                    )
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 64
    )

    print(
        "NESTED DELTAS (fold level, ALL stratum)"
    )

    print(
        "=" * 64
    )

    show = deltas[
        (
            deltas[
                "horizon"
            ]
            == PRIMARY_HORIZON
        )
        & (
            deltas[
                "learner"
            ]
            == PRIMARY_LEARNER
        )
    ]

    print(
        show.to_string(
            index=False
        )
    )

    print(
        "\nPAYOFF GATE (per-fold TOP20 - ALL spread)"
    )

    print(
        gate[
            gate[
                "horizon"
            ]
            == PRIMARY_HORIZON
        ].to_string(
            index=False
        )
        if len(
            gate
        )
        else "(empty)"
    )

    print(
        "\nDIRECTION_V3R_PASS"
    )


def build_15m(
    five: pd.DataFrame,
) -> pd.DataFrame:
    """15m decision panel for the frozen quantile layer.

    Reuses the aggregation, feature and target definitions already
    validated in build_pytdx_panel, so the frozen Quantile model
    sees exactly the schema it was fitted on.
    """

    bars = aggregate_15m(
        five
    )

    features = build_features(
        bars,
        five,
    )

    panel = pd.concat(
        [
            bars[
                [
                    "bar_start_time",
                    "bar_end_time",
                ]
            ].rename(
                columns={
                    "bar_start_time": (
                        "meta_base_bar_time"
                    ),
                    "bar_end_time": (
                        "meta_decision_time"
                    ),
                }
            ),
            bars[
                [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_oi",
                ]
            ],
            features,
        ],
        axis=1,
    )

    for h in (
        QUANT_HORIZON,
        8,
    ):

        for (
            name,
            values,
        ) in build_targets(
            bars,
            h,
        ).items():

            panel[
                name
            ] = values

    return panel


def audit_oi_tail(
    work: pd.DataFrame,
    instrument: str,
) -> dict:
    """Report the relative-OI tail actually entering the model.

    L8 `position` jumps when the continuous series rolls, so
    before trusting any OI result we look at whether roll-sized
    jumps really survive into the master sample. Nothing is
    removed or thresholded here.
    """

    col = (
        "rel_dOI_5m"
    )

    v = pd.to_numeric(
        work[
            col
        ],
        errors=(
            "coerce"
        ),
    ).dropna()

    a = (
        v.abs()
    )

    rec = {
        "instrument": (
            instrument
        ),
        "n": int(
            len(
                v
            )
        ),
        "abs_p50": float(
            a.quantile(
                0.50
            )
        ),
        "abs_p90": float(
            a.quantile(
                0.90
            )
        ),
        "abs_p99": float(
            a.quantile(
                0.99
            )
        ),
        "abs_p999": float(
            a.quantile(
                0.999
            )
        ),
        "abs_max": float(
            a.max()
        ),
    }

    biggest = (
        a.sort_values(
            ascending=(
                False
            )
        ).head(
            5
        )
    )

    stamps = []

    for (
        idx,
        val,
    ) in biggest.items():

        row = work.loc[
            idx
        ]

        ts = row.get(
            "row_id"
        )

        stamps.append(
            {
                "abs_value": float(
                    val
                ),
                "row_id": int(
                    ts
                ),
            }
        )

    rec[
        "largest_abs"
    ] = (
        stamps
    )

    return rec


if __name__ == "__main__":
    main()
