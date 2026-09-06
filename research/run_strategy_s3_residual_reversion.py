#!/usr/bin/env python3
"""S3-V0 - Latent Factor Residual Reversion.

Locked design (no tuning, no extra features). Core mathematical
logic below is reproduced verbatim from the agreed S3-V0 spec.

Goal (one sentence):
    Test whether short-horizon RELATIVE deviations, after removing
    a common factor via PCA, show a stable 5m/15m mean reversion.

No macro, no Quantile, no Volume/OI, no ML model selection, no
parameter tuning. Three strategies share the EXACT same event set
so they are comparable:

    A  RAW_CONTRARIAN     - trade raw 30m cross-section dispersion
    B  RESID_CONTRARIAN   - trade PCA-residual 30m dispersion
    C  RESID_MOMENTUM     - control (opposite of B)

Answers:
    B > A  -> residualization adds value
    B > C  -> residual dispersion is mean-reversion, not momentum

This script implements exactly the locked spec. The only additions
beyond the verbatim core are the five result tables required by the
audit (primary H15 summary, paired comparison, cluster breadth,
instrument attribution, PCA/clustering audit) -- read-only output
wiring, no change to the math.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA


ROOT = Path(
    __file__
).resolve().parents[1]

SRC = (
    ROOT
    / "research"
    / "exports"
    / "v3r_5m"
)

OUT = (
    ROOT
    / "research"
    / "exports"
    / "strategy_s3_residual"
)

INSTRUMENTS = [
    "AG", "AU", "CU", "AL", "SN", "NI", "RB", "I",
    "SC", "RU", "MA", "TA", "M", "P", "CF", "LC",
]

N_CLUSTERS = 4

FORMATION_BARS = 6        # 30m
H5_BARS = 1
H15_BARS = 3

TRAIN_MONTHS = 6
TEST_MONTHS = 2

MIN_CORR_OVERLAP = 500
MIN_PCA_ROWS = 1000

TOP_FRACTION = 0.25

FIVE_MIN = pd.Timedelta(
    minutes=5
)

FIVE_NS = int(
    FIVE_MIN.value
)

STRATEGIES = (
    "RAW_CONTRARIAN",
    "RESID_CONTRARIAN",
    "RESID_MOMENTUM",
)


# ----------------------------------------------------------------------
# 1. Single-instrument causal data
# ----------------------------------------------------------------------

def prepare_instrument(
    path: Path,
) -> pd.DataFrame:

    x = pd.read_csv(
        path,
        parse_dates=[
            "bar_start_time",
            "bar_end_time",
        ],
    )

    x = (
        x.sort_values(
            "bar_start_time"
        )
        .drop_duplicates(
            "bar_start_time"
        )
        .reset_index(
            drop=True
        )
    )

    t = x[
        "bar_start_time"
    ].to_numpy(
        dtype="datetime64[ns]"
    )

    close = x[
        "close"
    ].to_numpy(
        dtype=float
    )

    open_ = x[
        "open"
    ].to_numpy(
        dtype=float
    )

    n = len(
        x
    )

    # ---------------------------
    # Current 5m return
    # ---------------------------

    r5 = np.full(
        n,
        np.nan,
    )

    continuous_prev = np.zeros(
        n,
        dtype=bool,
    )

    if n > 1:
        dt = (
            np.diff(
                t
            )
            .astype(
                "timedelta64[ns]"
            )
            .astype(
                np.int64
            )
        )
        continuous_prev[
            1:
        ] = (
            dt
            == FIVE_NS
        )

        idx = np.where(
            continuous_prev
        )[0]

        r5[
            idx
        ] = (
            close[
                idx
            ]
            / close[
                idx
                - 1
            ]
            - 1.0
        )

    # ---------------------------
    # Future return:
    # signal known after current bar closes
    # entry = next bar OPEN
    # ---------------------------

    fut_h5 = np.full(
        n,
        np.nan,
    )

    fut_h15 = np.full(
        n,
        np.nan,
    )

    for i in range(
        n
    ):

        # H5:
        # current i
        # entry next bar i+1 open
        # exit i+1 close
        if i + 1 < n:
            seg = t[
                i : i + 2
            ]

            if (
                len(
                    seg
                )
                == 2
                and np.all(
                    (
                        np.diff(
                            seg
                        )
                        .astype(
                            "timedelta64[ns]"
                        )
                        .astype(
                            np.int64
                        )
                        == FIVE_NS
                    )
                )
            ):
                entry = open_[
                    i + 1
                ]

                if np.isfinite(
                    entry
                ) and entry > 0:
                    fut_h5[
                        i
                    ] = (
                        close[
                            i + 1
                        ]
                        / entry
                        - 1.0
                    )

        # H15:
        # entry i+1 open
        # exit i+3 close
        if i + 3 < n:
            seg = t[
                i : i + 4
            ]

            if np.all(
                (
                    np.diff(
                        seg
                    )
                    .astype(
                        "timedelta64[ns]"
                    )
                    .astype(
                        np.int64
                    )
                    == FIVE_NS
                )
            ):
                entry = open_[
                    i + 1
                ]

                if np.isfinite(
                    entry
                ) and entry > 0:
                    fut_h15[
                        i
                    ] = (
                        close[
                            i + 3
                        ]
                        / entry
                        - 1.0
                    )

    return pd.DataFrame(
        {
            "bar_start_time": x[
                "bar_start_time"
            ],
            "r5": r5,
            "fut_h5": fut_h5,
            "fut_h15": fut_h15,
        }
    )


# ----------------------------------------------------------------------
# 2. Wide panel
# ----------------------------------------------------------------------

def build_wide():

    ret = {}
    h5 = {}
    h15 = {}

    for inst in INSTRUMENTS:
        d = prepare_instrument(
            SRC
            / f"{inst}_5m.csv"
        ).set_index(
            "bar_start_time"
        )

        ret[
            inst
        ] = d[
            "r5"
        ]
        h5[
            inst
        ] = d[
            "fut_h5"
        ]
        h15[
            inst
        ] = d[
            "fut_h15"
        ]

    ret_wide = pd.DataFrame(
        ret
    ).sort_index()

    h5_wide = pd.DataFrame(
        h5
    ).reindex(
        ret_wide.index
    )

    h15_wide = pd.DataFrame(
        h15
    ).reindex(
        ret_wide.index
    )

    return (
        ret_wide,
        h5_wide,
        h15_wide,
    )


# ----------------------------------------------------------------------
# 3. Calendar walk-forward folds (fixed start, expanding train)
# ----------------------------------------------------------------------

def make_calendar_folds(
    index: pd.DatetimeIndex,
) -> list[dict]:

    start = (
        pd.Timestamp(
            index.min()
        )
        .normalize()
    )

    end = pd.Timestamp(
        index.max()
    )

    train_end = start + pd.DateOffset(
        months=TRAIN_MONTHS
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

        if test_end > end:
            break

        folds.append(
            {
                "fold": fold_id,
                "train_start": start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )

        fold_id += 1
        train_end += pd.DateOffset(
            months=TEST_MONTHS
        )

    return folds


# ----------------------------------------------------------------------
# 4. TRAIN-only correlation clustering
# ----------------------------------------------------------------------

def train_corr_distance(
    train: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    corr = train.corr(
        min_periods=MIN_CORR_OVERLAP
    )

    overlap = pd.DataFrame(
        index=INSTRUMENTS,
        columns=INSTRUMENTS,
        dtype=float,
    )

    for a in INSTRUMENTS:
        for b in INSTRUMENTS:
            overlap.loc[
                a,
                b,
            ] = int(
                train[
                    [
                        a,
                        b,
                    ]
                ]
                .dropna()
                .shape[0]
            )

    corr = corr.reindex(
        index=INSTRUMENTS,
        columns=INSTRUMENTS,
    )

    corr = corr.fillna(
        0.0
    )

    np.fill_diagonal(
        corr.values,
        1.0,
    )

    corr = corr.clip(
        -1.0,
        1.0,
    )

    distance = np.sqrt(
        0.5
        * (
            1.0
            - corr
        )
    )

    np.fill_diagonal(
        distance.values,
        0.0,
    )

    return (
        distance,
        overlap,
    )


def fit_clusters(
    distance: pd.DataFrame,
) -> dict[
    int,
    list[str],
]:

    try:
        model = AgglomerativeClustering(
            n_clusters=N_CLUSTERS,
            metric="precomputed",
            linkage="average",
        )
    except TypeError:
        # older sklearn compatibility
        model = AgglomerativeClustering(
            n_clusters=N_CLUSTERS,
            affinity="precomputed",
            linkage="average",
        )

    labels = model.fit_predict(
        distance.to_numpy()
    )

    clusters: dict[
        int,
        list[str],
    ] = {}

    for inst, label in zip(
        distance.index,
        labels,
    ):
        clusters.setdefault(
            int(
                label
            ),
            [],
        ).append(
            inst
        )

    return clusters


# ----------------------------------------------------------------------
# 5. PCA residual core (TRAIN fit / TEST transform via shared scale)
# ----------------------------------------------------------------------

def rolling_sum_contiguous(
    df: pd.DataFrame,
    bars: int,
) -> pd.DataFrame:

    arr = df.to_numpy(
        dtype=float
    )

    idx = df.index.view(
        "int64"
    )

    out = np.full(
        arr.shape,
        np.nan,
        dtype=float,
    )

    for i in range(
        bars - 1,
        len(
            df
        ),
    ):

        times = idx[
            i
            - bars
            + 1 : i
            + 1
        ]

        if not np.all(
            np.diff(
                times
            )
            == FIVE_NS
        ):
            continue

        window = arr[
            i
            - bars
            + 1 : i
            + 1
        ]

        if not np.isfinite(
            window
        ).all():
            continue

        out[
            i
        ] = window.sum(
            axis=0
        )

    return pd.DataFrame(
        out,
        index=df.index,
        columns=df.columns,
    )


def fit_transform_cluster(
    ret_wide: pd.DataFrame,
    members: list[str],
    train_start,
    train_end,
):

    sync = (
        ret_wide[
            members
        ]
        .dropna()
        .copy()
    )

    train_mask = (
        (
            sync.index
            >= train_start
        )
        & (
            sync.index
            < train_end
        )
    )

    train = sync.loc[
        train_mask
    ]

    if len(
        train
    ) < MIN_PCA_ROWS:
        return None

    mu = train.mean()
    sigma = train.std(
        ddof=1
    )

    if (
        (
            ~np.isfinite(
                sigma
            )
        ).any()
        or (
            sigma
            <= 0
        ).any()
    ):
        return None

    z = (
        sync
        - mu
    ) / sigma

    pca = PCA(
        n_components=1
    )

    pca.fit(
        z.loc[
            train_mask
        ]
    )

    component = pca.components_[
        0
    ]

    score = (
        z.to_numpy()
        @ component
    )

    fitted = np.outer(
        score,
        component,
    )

    residual = (
        z.to_numpy()
        - fitted
    )

    residual = pd.DataFrame(
        residual,
        index=z.index,
        columns=z.columns,
    )

    raw30 = rolling_sum_contiguous(
        z,
        FORMATION_BARS,
    )

    resid30 = rolling_sum_contiguous(
        residual,
        FORMATION_BARS,
    )

    raw_scale = (
        raw30.loc[
            train_mask
        ]
        .std(
            ddof=1
        )
    )

    resid_scale = (
        resid30.loc[
            train_mask
        ]
        .std(
            ddof=1
        )
    )

    raw_signal = (
        raw30
        / raw_scale
    )

    resid_signal = (
        resid30
        / resid_scale
    )

    return {
        "members": members,
        "raw_signal": raw_signal,
        "resid_signal": resid_signal,
        "residual_1bar": residual,
        "pc1_component": component,
        "explained_variance": float(
            pca.explained_variance_ratio_[
                0
            ]
        ),
    }


# ----------------------------------------------------------------------
# 6. Strategy construction
# ----------------------------------------------------------------------

def long_short_return(
    signal: pd.Series,
    future: pd.Series,
    *,
    contrarian: bool,
) -> tuple[
    float,
    list[str],
    list[str],
] | None:

    x = pd.concat(
        [
            signal.rename(
                "signal"
            ),
            future.rename(
                "future"
            ),
        ],
        axis=1,
    ).dropna()

    n = len(
        x
    )

    if n < 2:
        return None

    n_select = max(
        1,
        int(
            np.floor(
                n
                * TOP_FRACTION
            )
        ),
    )

    x = x.sort_values(
        "signal"
    )

    low = x.head(
        n_select
    )

    high = x.tail(
        n_select
    )

    if contrarian:
        longs = low
        shorts = high
    else:
        longs = high
        shorts = low

    r = (
        0.5
        * longs[
            "future"
        ].mean()
        - 0.5
        * shorts[
            "future"
        ].mean()
    )

    return (
        float(
            r
        ),
        list(
            longs.index
        ),
        list(
            shorts.index
        ),
    )


# ----------------------------------------------------------------------
# 7. Per-fold cluster x timestamp events
# ----------------------------------------------------------------------

def run_fold(
    fold,
    ret_wide,
    h5_wide,
    h15_wide,
):

    train_mask = (
        (
            ret_wide.index
            >= fold[
                "train_start"
            ]
        )
        & (
            ret_wide.index
            < fold[
                "train_end"
            ]
        )
    )

    test_mask = (
        (
            ret_wide.index
            >= fold[
                "test_start"
            ]
        )
        & (
            ret_wide.index
            < fold[
                "test_end"
            ]
        )
    )

    train_ret = ret_wide.loc[
        train_mask
    ]

    distance, overlap = train_corr_distance(
        train_ret
    )

    clusters = fit_clusters(
        distance
    )

    event_rows = []
    cluster_rows = []

    for cluster_id, members in clusters.items():

        if len(
            members
        ) < 2:
            cluster_rows.append(
                {
                    "fold": fold[
                        "fold"
                    ],
                    "cluster": cluster_id,
                    "members": ",".join(
                        members
                    ),
                    "excluded": True,
                    "reason": "singleton",
                }
            )
            continue

        fit = fit_transform_cluster(
            ret_wide,
            members,
            fold[
                "train_start"
            ],
            fold[
                "train_end"
            ],
        )

        if fit is None:
            cluster_rows.append(
                {
                    "fold": fold[
                        "fold"
                    ],
                    "cluster": cluster_id,
                    "members": ",".join(
                        members
                    ),
                    "excluded": True,
                    "reason": "insufficient_pca_rows",
                }
            )
            continue

        cluster_rows.append(
            {
                "fold": fold[
                    "fold"
                ],
                "cluster": cluster_id,
                "members": ",".join(
                    members
                ),
                "excluded": False,
                "pc1_explained_variance": (
                    fit[
                        "explained_variance"
                    ]
                ),
            }
        )

        raw_signal = fit[
            "raw_signal"
        ]

        resid_signal = fit[
            "resid_signal"
        ]

        idx = (
            raw_signal.index.intersection(
                resid_signal.index
            )
        )

        idx = idx[
            (
                idx
                >= fold[
                    "test_start"
                ]
            )
            & (
                idx
                < fold[
                    "test_end"
                ]
            )
        ]

        for ts in idx:

            raw_s = raw_signal.loc[
                ts
            ]

            res_s = resid_signal.loc[
                ts
            ]

            if (
                not np.isfinite(
                    raw_s.to_numpy()
                ).all()
                or not np.isfinite(
                    res_s.to_numpy()
                ).all()
            ):
                continue

            minute_number = (
                ts.hour
                * 60
                + ts.minute
            )

            offset = (
                (
                    minute_number
                    // 5
                )
                % 3
            )

            for horizon, future_wide in (
                (
                    "H5",
                    h5_wide,
                ),
                (
                    "H15",
                    h15_wide,
                ),
            ):

                if ts not in future_wide.index:
                    continue

                future = future_wide.loc[
                    ts,
                    members,
                ]

                if not np.isfinite(
                    future.to_numpy(
                        dtype=float
                    )
                ).all():
                    continue

                a = long_short_return(
                    raw_s,
                    future,
                    contrarian=True,
                )

                b = long_short_return(
                    res_s,
                    future,
                    contrarian=True,
                )

                c = long_short_return(
                    res_s,
                    future,
                    contrarian=False,
                )

                if (
                    a is None
                    or b is None
                    or c is None
                ):
                    continue

                event_rows.append(
                    {
                        "fold": fold[
                            "fold"
                        ],
                        "cluster": cluster_id,
                        "timestamp": ts,
                        "offset": offset,
                        "horizon": horizon,
                        "raw_contrarian": a[
                            0
                        ],
                        "resid_contrarian": b[
                            0
                        ],
                        "resid_momentum": c[
                            0
                        ],
                        "raw_long": ",".join(
                            a[
                                1
                            ]
                        ),
                        "raw_short": ",".join(
                            a[
                                2
                            ]
                        ),
                        "resid_long": ",".join(
                            b[
                                1
                            ]
                        ),
                        "resid_short": ",".join(
                            b[
                                2
                            ]
                        ),
                    }
                )

    return (
        pd.DataFrame(
            event_rows
        ),
        pd.DataFrame(
            cluster_rows
        ),
        distance,
        overlap,
    )


# ----------------------------------------------------------------------
# 8. Portfolio aggregation (active clusters equal-weight)
# ----------------------------------------------------------------------

def aggregate_portfolio(
    events: pd.DataFrame,
) -> pd.DataFrame:

    group_cols = [
        "fold",
        "timestamp",
        "offset",
        "horizon",
    ]

    out = (
        events.groupby(
            group_cols,
            observed=True,
        )[
            [
                "raw_contrarian",
                "resid_contrarian",
                "resid_momentum",
            ]
        ]
        .mean()
        .reset_index()
    )

    return out


# ----------------------------------------------------------------------
# 9. Per-fold strategy summary
# ----------------------------------------------------------------------

def summarise_strategy(
    portfolio: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        horizon,
        offset,
        fold,
    ), g in portfolio.groupby(
        [
            "horizon",
            "offset",
            "fold",
        ],
        observed=True,
    ):

        for strategy in STRATEGIES:

            col = {
                "RAW_CONTRARIAN": "raw_contrarian",
                "RESID_CONTRARIAN": "resid_contrarian",
                "RESID_MOMENTUM": "resid_momentum",
            }[
                strategy
            ]

            r = (
                g[
                    col
                ]
                .dropna()
                .to_numpy(
                    dtype=float
                )
            )

            if len(
                r
            ) == 0:
                continue

            mean_r = float(
                np.mean(
                    r
                )
            )

            break_even_cost_bps = (
                mean_r
                * 10000.0
                / 2.0
            )

            rows.append(
                {
                    "horizon": horizon,
                    "offset": int(
                        offset
                    ),
                    "fold": int(
                        fold
                    ),
                    "strategy": strategy,
                    "n": len(
                        r
                    ),
                    "mean_return": mean_r,
                    "median_return": float(
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
                    "break_even_cost_bps": break_even_cost_bps,
                }
            )

    return pd.DataFrame(
        rows
    )


# ----------------------------------------------------------------------
# 10. Paired deltas (same timestamps by construction)
# ----------------------------------------------------------------------

def paired_deltas(
    portfolio: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        horizon,
        offset,
        fold,
    ), g in portfolio.groupby(
        [
            "horizon",
            "offset",
            "fold",
        ],
        observed=True,
    ):

        d_resid_vs_raw = (
            g[
                "resid_contrarian"
            ]
            - g[
                "raw_contrarian"
            ]
        )

        d_contra_vs_mom = (
            g[
                "resid_contrarian"
            ]
            - g[
                "resid_momentum"
            ]
        )

        rows.append(
            {
                "horizon": horizon,
                "offset": int(
                    offset
                ),
                "fold": int(
                    fold
                ),
                "n": len(
                    g
                ),
                "resid_vs_raw_mean": float(
                    d_resid_vs_raw.mean()
                ),
                "resid_vs_raw_positive_share": float(
                    (
                        d_resid_vs_raw
                        > 0
                    ).mean()
                ),
                "contra_vs_momentum_mean": float(
                    d_contra_vs_mom.mean()
                ),
                "contra_vs_momentum_positive_share": float(
                    (
                        d_contra_vs_mom
                        > 0
                    ).mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ----------------------------------------------------------------------
# 11. Mechanism diagnostic (NOT a filter)
# ----------------------------------------------------------------------

def variance_ratio(
    s: pd.Series,
    q: int = 3,
) -> float:

    x = (
        s.dropna()
        .to_numpy(
            dtype=float
        )
    )

    if len(
        x
    ) < 100:
        return np.nan

    v1 = np.var(
        x,
        ddof=1,
    )

    if v1 <= 0:
        return np.nan

    idx = s.dropna().index
    vals = (
        s.dropna()
        .to_numpy()
    )

    sums = []

    for i in range(
        q - 1,
        len(
            vals
        ),
    ):

        tt = idx[
            i
            - q
            + 1 : i
            + 1
        ]

        if not np.all(
            np.diff(
                tt.view(
                    "int64"
                )
            )
            == FIVE_NS
        ):
            continue

        sums.append(
            vals[
                i
                - q
                + 1 : i
                + 1
            ].sum()
        )

    if len(
        sums
    ) < 50:
        return np.nan

    vq = np.var(
        sums,
        ddof=1,
    )

    return float(
        vq
        / (
            q
            * v1
        )
    )


# ======================================================================
# AUDIT / ATTRIBUTION TABLES (read-only output wiring, no math change)
# ======================================================================

def build_primary_h15_summary(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Table 1: Primary H15, by strategy x offset, aggregated over
    folds."""

    sub = summary[
        summary.horizon
        == "H15"
    ]

    rows = []

    for (
        strategy,
        offset,
    ), g in sub.groupby(
        [
            "strategy",
            "offset",
        ],
        observed=True,
    ):

        mean_r = g[
            "mean_return"
        ].to_numpy(
            dtype=float
        )

        rows.append(
            {
                "strategy": strategy,
                "offset": int(
                    offset
                ),
                "folds": int(
                    len(
                        g
                    )
                ),
                "positive_folds": int(
                    (
                        mean_r
                        > 0
                    ).sum()
                ),
                "median_fold_mean": float(
                    np.median(
                        mean_r
                    )
                ),
                "mean_fold_mean": float(
                    np.mean(
                        mean_r
                    )
                ),
                "median_break_even_cost_bps": float(
                    np.median(
                        g[
                            "break_even_cost_bps"
                        ].to_numpy(
                            dtype=float
                        )
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_paired_comparison(
    deltas: pd.DataFrame,
) -> pd.DataFrame:
    """Table 2: paired deltas aggregated over folds, by comparison
    x horizon x offset."""

    rows = []

    for (
        horizon,
        offset,
    ), g in deltas.groupby(
        [
            "horizon",
            "offset",
        ],
        observed=True,
    ):

        rv = g[
            "resid_vs_raw_mean"
        ].to_numpy(
            dtype=float
        )

        cm = g[
            "contra_vs_momentum_mean"
        ].to_numpy(
            dtype=float
        )

        rows.append(
            {
                "comparison": "RESID_vs_RAW",
                "horizon": horizon,
                "offset": int(
                    offset
                ),
                "folds": int(
                    len(
                        g
                    )
                ),
                "median_paired_delta": float(
                    np.median(
                        rv
                    )
                ),
                "positive_fold_share": float(
                    (
                        rv
                        > 0
                    ).mean()
                ),
            }
        )

        rows.append(
            {
                "comparison": "RESID_vs_MOM",
                "horizon": horizon,
                "offset": int(
                    offset
                ),
                "folds": int(
                    len(
                        g
                    )
                ),
                "median_paired_delta": float(
                    np.median(
                        cm
                    )
                ),
                "positive_fold_share": float(
                    (
                        cm
                        > 0
                    ).mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_cluster_breadth(
    events: pd.DataFrame,
    clusters: pd.DataFrame,
) -> pd.DataFrame:
    """Table 3: per (fold, cluster) RESID_CONTRARIAN H15 mean.
    Answers: does one cluster carry everything?"""

    sub = events[
        events.horizon
        == "H15"
    ]

    grp = sub.groupby(
        [
            "fold",
            "cluster",
        ],
        observed=True,
    )

    rows = []

    for (
        fold,
        cluster,
    ), g in grp:

        rows.append(
            {
                "fold": int(
                    fold
                ),
                "cluster": int(
                    cluster
                ),
                "n_events": int(
                    len(
                        g
                    )
                ),
                "mean_resid_contrarian_h15": float(
                    g[
                        "resid_contrarian"
                    ].mean()
                ),
                "median_resid_contrarian_h15": float(
                    g[
                        "resid_contrarian"
                    ].median()
                ),
                "win_rate": float(
                    (
                        g[
                            "resid_contrarian"
                        ]
                        > 0
                    ).mean()
                ),
            }
        )

    out = pd.DataFrame(
        rows
    )

    members = clusters[
        [
            "fold",
            "cluster",
            "members",
            "excluded",
            "pc1_explained_variance",
        ]
    ]

    out = out.merge(
        members,
        on=[
            "fold",
            "cluster",
        ],
        how="left",
    )

    # Aggregate breadth metric: share of (fold,cluster) cells that
    # are positive.
    n_cells = len(
        out
    )

    pos_cells = int(
        (
            out[
                "mean_resid_contrarian_h15"
            ]
            > 0
        ).sum()
    )

    out.attrs[
        "positive_cluster_cells"
    ] = (
        f"{pos_cells}/{n_cells}"
    )

    return out


def instrument_attribution(
    events: pd.DataFrame,
    h5_wide: pd.DataFrame,
    h15_wide: pd.DataFrame,
) -> pd.DataFrame:
    """Table 4: per-instrument long/short selection frequency and
    net P&L contribution to RESID_CONTRARIAN (gross, 0.5 long /
    0.5 short per event). Prevents a single name carrying the
    strategy."""

    contrib = {}

    for _, row in events.iterrows():

        horizon = row[
            "horizon"
        ]

        fw = (
            h15_wide
            if horizon
            == "H15"
            else h5_wide
        )

        ts = row[
            "timestamp"
        ]

        if ts not in fw.index:
            continue

        fut = fw.loc[
            ts
        ]

        longs = str(
            row[
                "resid_long"
            ]
        ).split(
            ","
        )

        shorts = str(
            row[
                "resid_short"
            ]
        ).split(
            ","
        )

        for inst in longs:

            if (
                inst
                in fut.index
                and np.isfinite(
                    fut[
                        inst
                    ]
                )
            ):
                d = contrib.setdefault(
                    inst,
                    {
                        "n_long": 0,
                        "n_short": 0,
                        "contrib": 0.0,
                    },
                )
                d[
                    "n_long"
                ] += 1
                d[
                    "contrib"
                ] += (
                    0.5
                    * fut[
                        inst
                    ]
                )

        for inst in shorts:

            if (
                inst
                in fut.index
                and np.isfinite(
                    fut[
                        inst
                    ]
                )
            ):
                d = contrib.setdefault(
                    inst,
                    {
                        "n_long": 0,
                        "n_short": 0,
                        "contrib": 0.0,
                    },
                )
                d[
                    "n_short"
                ] += 1
                d[
                    "contrib"
                ] -= (
                    0.5
                    * fut[
                        inst
                    ]
                )

    if not contrib:
        return pd.DataFrame(
            columns=[
                "instrument",
                "n_long",
                "n_short",
                "contrib",
            ]
        )

    df = pd.DataFrame(
        contrib
    ).T.reset_index().rename(
        columns={
            "index": "instrument",
        }
    )

    df = df.sort_values(
        "contrib",
        ascending=False,
    ).reset_index(
        drop=True
    )

    return df


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    if OUT.exists() and any(
        OUT.iterdir()
    ):
        raise RuntimeError(
            f"{OUT} already non-empty"
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ret_wide, h5_wide, h15_wide = build_wide()

    folds = make_calendar_folds(
        ret_wide.index
    )

    all_events = []
    all_clusters = []

    for fold in folds:

        print(
            "FOLD",
            fold,
            flush=True,
        )

        (
            events,
            cluster_info,
            distance,
            overlap,
        ) = run_fold(
            fold,
            ret_wide,
            h5_wide,
            h15_wide,
        )

        all_events.append(
            events
        )

        all_clusters.append(
            cluster_info
        )

        distance.to_csv(
            OUT
            / f"fold_{fold['fold']}_distance.csv"
        )

        overlap.to_csv(
            OUT
            / f"fold_{fold['fold']}_overlap.csv"
        )

    events = pd.concat(
        all_events,
        ignore_index=True,
    )

    clusters = pd.concat(
        all_clusters,
        ignore_index=True,
    )

    portfolio = aggregate_portfolio(
        events
    )

    summary = summarise_strategy(
        portfolio
    )

    deltas = paired_deltas(
        portfolio
    )

    # ---- base outputs ----
    events.to_csv(
        OUT / "s3_cluster_events.csv",
        index=False,
    )

    portfolio.to_csv(
        OUT / "s3_portfolio.csv",
        index=False,
    )

    clusters.to_csv(
        OUT / "s3_clusters.csv",
        index=False,
    )

    summary.to_csv(
        OUT / "s3_fold_summary.csv",
        index=False,
    )

    deltas.to_csv(
        OUT / "s3_paired_deltas.csv",
        index=False,
    )

    # ---- audit / attribution tables (required, read-only) ----
    primary_h15 = build_primary_h15_summary(
        summary
    )

    paired = build_paired_comparison(
        deltas
    )

    breadth = build_cluster_breadth(
        events,
        clusters,
    )

    attrib = instrument_attribution(
        events,
        h5_wide,
        h15_wide,
    )

    primary_h15.to_csv(
        OUT / "s3_primary_h15_summary.csv",
        index=False,
    )

    paired.to_csv(
        OUT / "s3_paired_comparison.csv",
        index=False,
    )

    breadth.to_csv(
        OUT / "s3_cluster_breadth.csv",
        index=False,
    )

    attrib.to_csv(
        OUT / "s3_instrument_attribution.csv",
        index=False,
    )

    # Table 5 = PCA / clustering audit = s3_clusters.csv (per-fold
    # membership + PC1 explained variance) plus per-fold distance /
    # overlap csvs already written above.

    config = {
        "name": (
            "S3-V0 Latent Factor Residual Reversion"
        ),
        "universe": INSTRUMENTS,
        "n_clusters": N_CLUSTERS,
        "formation_bars": FORMATION_BARS,
        "primary_horizon": "H15",
        "secondary_horizon": "H5",
        "train_months": TRAIN_MONTHS,
        "test_months": TEST_MONTHS,
        "strategies": list(
            STRATEGIES
        ),
        "rules": [
            "clustering TRAIN only",
            "PCA TRAIN only",
            "normalization TRAIN only",
            "signal after current 5m close",
            "entry next 5m open",
            "strict 5m continuity",
            "same event sample for all 3 strategies",
            "no Quantile",
            "no macro",
            "no volume",
            "no OI",
            "no ML",
            "no parameter tuning",
        ],
        "cluster_breadth_positive_cells": breadth.attrs.get(
            "positive_cluster_cells",
            "",
        ),
    }

    (
        OUT
        / "s3_config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ---- console snapshot ----
    print(
        "\n=== S3 PRIMARY H15 SUMMARY ==="
    )
    print(
        primary_h15.to_string(
            index=False
        )
    )

    print(
        "\n=== S3 PAIRED COMPARISON ==="
    )
    print(
        paired.to_string(
            index=False
        )
    )

    print(
        "\n=== S3 INSTRUMENT ATTRIBUTION (top/bottom) ==="
    )
    print(
        attrib.head(
            5
        ).to_string(
            index=False
        )
    )
    print(
        "..."
    )
    print(
        attrib.tail(
            5
        ).to_string(
            index=False
        )
    )

    print(
        "\n=== S3 CLUSTER BREADTH ==="
    )
    print(
        f"positive (fold,cluster) cells: "
        f"{breadth.attrs.get('positive_cluster_cells','')}"
    )
    print(
        breadth.to_string(
            index=False
        )
    )

    print(
        "\nS3_V0_DONE"
    )


if __name__ == "__main__":
    main()
