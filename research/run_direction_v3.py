#!/usr/bin/env python3
"""Direction V3 -- participation state (volume / OI).

V2 established the new baseline:

    B0 = price path + realized volatility state

and showed that semivariance and hand-built interactions added
nothing, so S2 / S3 are NOT carried forward. V3 asks one question:

    does PARTICIPATION information add anything beyond price + RV?

Blocks
------
    B0  r5, r10, r15, r30, RV15, RV30, RV60, rv_acceleration
    B1  B0 + volume surprise (5m / 30m / 60m) + volume acceleration
    B2  B0 + dOI (5m / 15m / 30m) + relative dOI
    B3  B0 + volume + OI + r5 x volume_surprise + r5 x relative_dOI

All participation features are RELATIVE. Absolute volume never
enters the model, because the research question is not "is high
volume bullish" but "does the future relation of a price move
change when that move is accompanied by unusual participation".

Volume baseline is a rolling MEDIAN over the previous 12 continuous
bars, which is robust and instrument-free.

Open interest uses the PyTDX `position` field, which was
cross-validated against TqSdk on real contracts. The transaction
fields `zengcang`, `nature` and `direction` are UNTRUSTED and are
not used anywhere.

Horizons and learner
--------------------
    H5 primary, H15 secondary (H30 stays dropped)
    GBR primary, logistic secondary

V2 showed logistic can improve ranking but its probability
calibration degrades once state features are added, while GBR
improves both, so GBR carries the confirmatory test.

Pre-registered confirmatory comparison: H5 + ALL + GBR, B3 vs B0.
A result of B3 ~= B0 should be read as "participation adds nothing"
and should stop the local bar-state feature search.
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

from research.run_direction_v0 import (  # noqa: E402
    build_execution_grid,
    quantile_state,
)

from research.run_direction_v1 import (  # noqa: E402
    add_targets,
    continuity_prefix,
)

from research.run_direction_v2 import (  # noqa: E402
    HORIZON_BARS,
    INSTRUMENTS,
    NONOVERLAP_MOD,
    PANEL_15M,
    PANEL_5M,
    PRIMARY_HORIZON,
    PROB_CLIP,
    QUANT_FEATURE_SET,
    QUANT_HORIZON,
    QUANT_MODEL,
    REBASELINE,
    SPREAD_QUANTILE,
    STRATA,
    add_state_features,
    binary_metrics,
    make_learner,
    positive_share,
    rv_state_mechanism,
    spearman,
    window_ok,
    window_sum,
)

from research.run_quantile_rebaseline import (  # noqa: E402
    FEATURE_SETS,
    make_folds,
)


OUT = (
    ROOT
    / "research"
    / "exports"
    / "direction_v3"
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

LEARNERS = (
    "gbr",
    "logistic",
)

PRIMARY_LEARNER = "gbr"

EPS = 1e-12

# Trailing window used as the "normal participation" baseline.
VOL_BASE_BARS = 12


# ============================================================
# Feature blocks
# ============================================================

FEATURE_BLOCKS = {
    "B0": [
        "r5",
        "r10",
        "r15",
        "r30",
        "RV15",
        "RV30",
        "RV60",
        "rv_acceleration",
    ],
    "B1": [
        "r5",
        "r10",
        "r15",
        "r30",
        "RV15",
        "RV30",
        "RV60",
        "rv_acceleration",
        "vol_surprise_5m",
        "vol_surprise_30m",
        "vol_surprise_60m",
        "vol_acceleration",
    ],
    "B2": [
        "r5",
        "r10",
        "r15",
        "r30",
        "RV15",
        "RV30",
        "RV60",
        "rv_acceleration",
        "dOI_5m",
        "dOI_15m",
        "dOI_30m",
        "relative_dOI",
    ],
    "B3": [
        "r5",
        "r10",
        "r15",
        "r30",
        "RV15",
        "RV30",
        "RV60",
        "rv_acceleration",
        "vol_surprise_5m",
        "vol_surprise_30m",
        "vol_surprise_60m",
        "vol_acceleration",
        "dOI_5m",
        "dOI_15m",
        "dOI_30m",
        "relative_dOI",
        "r5_x_vol_surprise",
        "r5_x_relative_dOI",
    ],
}

COMMON_FEATURES = FEATURE_BLOCKS[
    "B3"
]


# ============================================================
# Participation features
# ============================================================

def add_participation_features(
    grid: pd.DataFrame,
    five: pd.DataFrame,
    cs: np.ndarray,
) -> pd.DataFrame:

    f = five.sort_values(
        "bar_start_time"
    ).reset_index(
        drop=True
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

    row_id = (
        grid[
            "row_id"
        ]
        .to_numpy(
            dtype=int
        )
    )

    # Last CLOSED bar at decision time t.
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

    # ---- volume baseline: rolling median over the previous
    #      12 continuous bars. Robust and instrument-free.
    #
    # baseline window = [k-12, k-1]

    base_start = (
        k
        - VOL_BASE_BARS
    )

    baseline = np.full(
        n_rows,
        np.nan,
    )

    ok_base = (
        window_ok(
            cs,
            base_start,
            VOL_BASE_BARS,
            n5,
        )
        & usable
        & (
            base_start
            >= 0
        )
    )

    idx = np.where(
        ok_base
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

    # ---- volume surprise: mean volume over the window,
    #      divided by the baseline level.

    for (
        label,
        n,
    ) in (
        (
            "vol_surprise_5m",
            1,
        ),
        (
            "vol_surprise_30m",
            6,
        ),
        (
            "vol_surprise_60m",
            12,
        ),
    ):

        mean_vol = (
            window_sum(
                vol,
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
            / n
        )

        data[
            label
        ] = (
            mean_vol
            / (
                baseline
                + EPS
            )
        )

    # recent 15m rate over the prior 45m rate
    recent = (
        window_sum(
            vol,
            start=(
                k
                - 2
            ),
            end=k,
            n=3,
            n_bars=n5,
            cs=cs,
            usable=usable,
            n_rows=n_rows,
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
            n=9,
            n_bars=n5,
            cs=cs,
            usable=usable,
            n_rows=n_rows,
        )
        / 9.0
    )

    data[
        "vol_acceleration"
    ] = (
        recent
        / (
            prior
            + EPS
        )
    )

    # ---- open interest changes ----

    for (
        label,
        n,
    ) in (
        (
            "dOI_5m",
            1,
        ),
        (
            "dOI_15m",
            3,
        ),
        (
            "dOI_30m",
            6,
        ),
    ):

        start = (
            k
            - n
            + 1
        )

        ok = (
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

        out = np.full(
            n_rows,
            np.nan,
        )

        idx = np.where(
            ok
        )[
            0
        ]

        if len(
            idx
        ):

            out[
                idx
            ] = (
                oi[
                    k[
                        idx
                    ]
                ]
                - oi[
                    start[
                        idx
                    ]
                    - 1
                ]
            )

        data[
            label
        ] = out

    oi_prev = np.full(
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

        oi_prev[
            idx
        ] = oi[
            k[
                idx
            ]
            - 1
        ]

    data[
        "relative_dOI"
    ] = (
        data[
            "dOI_5m"
        ]
        / (
            np.abs(
                oi_prev
            )
            + EPS
        )
    )

    # ---- two economically motivated interactions ----

    data[
        "r5_x_vol_surprise"
    ] = (
        grid[
            "r5"
        ].to_numpy(
            dtype=float
        )
        * data[
            "vol_surprise_5m"
        ]
    )

    data[
        "r5_x_relative_dOI"
    ] = (
        grid[
            "r5"
        ].to_numpy(
            dtype=float
        )
        * data[
            "relative_dOI"
        ]
    )

    for key, values in (
        data.items()
    ):

        grid[
            key
        ] = values

    return grid


# ============================================================
# Model run (train on ALL, strata applied afterwards)
# ============================================================

def run_model(
    work: pd.DataFrame,
    *,
    horizon: str,
    feature_names: list[str],
    learner: str,
) -> pd.DataFrame:

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
        int, dict
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


# ============================================================
# Main
# ============================================================

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
    verify_rows = []
    coverage_rows = []
    mech_rows = []

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

        grid = add_participation_features(
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

                    if (
                        block
                        == "B3"
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
            f"  H5 common="
            f"{cov.get('H5_common_rows')}"
            f"  H15 common="
            f"{cov.get('H15_common_rows')}"
        )

    results = pd.DataFrame(
        rows
    )

    results.to_csv(
        OUT
        / "direction_v3_by_instrument.csv",
        index=False,
    )

    pd.DataFrame(
        nonoverlap_rows
    ).to_csv(
        OUT
        / "direction_v3_nonoverlap.csv",
        index=False,
    )

    pd.DataFrame(
        coverage_rows
    ).to_csv(
        OUT
        / "v3_coverage.csv",
        index=False,
    )

    pd.DataFrame(
        verify_rows
    ).to_csv(
        OUT
        / "quantile_state_verification.csv",
        index=False,
    )

    if mech_rows:
        pd.DataFrame(
            mech_rows
        ).to_csv(
            OUT
            / "rv_state_mechanism.csv",
            index=False,
        )

    # ---- B3 vs B0 confirmatory delta ----

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
                    "B0"
                    not in base.index
                    or "B3"
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
                        "auc_B0": float(
                            base.loc[
                                "B0",
                                "auc",
                            ]
                        ),
                        "auc_B3": float(
                            base.loc[
                                "B3",
                                "auc",
                            ]
                        ),
                        "delta_auc": float(
                            base.loc[
                                "B3",
                                "auc",
                            ]
                            - base.loc[
                                "B0",
                                "auc",
                            ]
                        ),
                        "delta_brier_skill": float(
                            base.loc[
                                "B3",
                                "brier_skill",
                            ]
                            - base.loc[
                                "B0",
                                "brier_skill",
                            ]
                        ),
                        "delta_logloss_skill": float(
                            base.loc[
                                "B3",
                                "logloss_skill",
                            ]
                            - base.loc[
                                "B0",
                                "logloss_skill",
                            ]
                        ),
                        "delta_return_spread": float(
                            base.loc[
                                "B3",
                                "return_spread",
                            ]
                            - base.loc[
                                "B0",
                                "return_spread",
                            ]
                        ),
                    }
                )

    deltas = pd.DataFrame(
        delta_rows
    )

    deltas.to_csv(
        OUT
        / "b3_vs_b0_delta.csv",
        index=False,
    )

    # ---- cross summary ----

    summary_rows = []

    metrics = [
        "auc",
        "brier_skill",
        "logloss_skill",
        "return_spread",
        "calibration_slope",
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
        / "direction_v3_cross_summary.csv",
        index=False,
    )

    config = {
        "purpose": (
            "Test whether participation state "
            "(volume / OI) adds beyond price "
            "path + realized volatility."
        ),
        "pre_registered_primary": (
            f"{PRIMARY_HORIZON} + ALL + "
            f"{PRIMARY_LEARNER}, B3 vs B0"
        ),
        "baseline": (
            "B0 = price + RV, i.e. V2's S1. "
            "V2's S2 / S3 are not carried "
            "forward because they added "
            "nothing."
        ),
        "feature_blocks": (
            FEATURE_BLOCKS
        ),
        "volume_relative_only": (
            True
        ),
        "volume_baseline": (
            "rolling median over the previous "
            "12 continuous bars"
        ),
        "oi_source": (
            "PyTDX position, cross-validated "
            "against TqSdk on real contracts"
        ),
        "forbidden_fields_unused": [
            "zengcang",
            "nature",
            "nature_name",
            "direction",
            "amount",
        ],
        "horizons_bars": (
            HORIZON_BARS
        ),
        "learners": LEARNERS,
        "primary_learner": (
            PRIMARY_LEARNER
        ),
        "fair_comparison": (
            "every block evaluated on the "
            "common row set where all B3 "
            "features and the target exist"
        ),
        "continuity": (
            "every window continuous in "
            "calendar time"
        ),
        "causality": (
            "features end at the last closed "
            "bar k = row_id - 1"
        ),
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
        "\nDIRECTION_V3_PASS"
    )


if __name__ == "__main__":
    main()
