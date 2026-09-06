#!/usr/bin/env python3
"""Strategy Exploration S1 - Opportunity-Conditioned Confirmed Breakout.

The question is not "can we predict direction". Several rounds of
direction work said no, and the replication across eight unseen
instruments said the Price + RV result was not universal.

The question is whether we need to predict it at all.

Quantile can tell us that a bar is likely to move a lot. It cannot
tell us which way. So instead of guessing who moves first, this
waits for the market to answer and then asks the only thing that
matters for a strategy:

    after a confirmed breakout, does price continue (FOLLOW)
    or reverse (FADE)?

Design
------

    bars    [b-6 ... b-1]  [b]        [b+1 ... b+n]
            prior 30m      close      entry at open[b+1]
            high / low     confirms   exit at close[b+n]

The breakout is confirmed by the CLOSE of bar b against the range
of the six preceding closed bars. Entry is the OPEN of bar b+1.
Nothing is executed inside the triggering bar, because 5m OHLC does
not say whether high or low was touched first and assuming
otherwise manufactures intrabar fills.

Holding is a fixed number of bars, from HORIZON_BARS. There is no
stop, no target and no trailing exit. This is a gross mechanism
test, not a PnL backtest.

Opportunity conditioning uses the frozen strict-OOS Quantile width,
exactly as carried on the 5m grid. The TOP30 cut is taken inside
each fold on TRAIN events only, so the test half never sees its own
threshold.

Continuity
----------

The whole span [b-6, b+3] must be gap-free. That is the trigger
window, the confirmation bar, and the longest holding period. H5 and
H15 therefore run on one identical event set, and no event straddles
a session boundary or a data gap.

Restrictions honoured here
--------------------------

    no breakout lookback search      fixed at 6 bars
    no breakout threshold            any close outside the range
    no RV / volume / OI features     none are used
    no ML                            means and medians only
    no PnL, no fees, no slippage     gross returns
    no stop / target                 fixed time exit
    same master events               FOLLOW and FADE, ALL and TOP30
                                     all read the identical rows
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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

from research.run_direction_v3r import (  # noqa: E402
    INSTRUMENTS,
    QUANT_FEATURE_SET,
    SRC_5M,
    build_15m,
    continuity_prefix,
    span_ok,
)

from research.run_quantile_rebaseline import (  # noqa: E402
    FEATURE_SETS,
    make_folds,
)


OUT = (
    ROOT
    / "research"
    / "exports"
    / "strategy_s1"
)

# H5 = 1 five-minute bar, H15 = 3. Same convention as the
# direction experiments.
HORIZON_BARS = {
    "H5": 1,
    "H15": 3,
}

PRIMARY_H = "H5"

TRIGGER_BARS = 6

# [b-6, b+3]: trigger window, confirmation bar, longest hold.
BASE_SPAN = (
    TRIGGER_BARS
    + 1
    + max(
        HORIZON_BARS.values()
    )
)

OPPORTUNITY_TOP_FRACTION = 0.30

STRATEGIES = (
    "FOLLOW",
    "FADE",
)

OPPORTUNITY_SETS = (
    "ALL",
    "TOP30",
)


def forward_rolling_max(
    values: np.ndarray,
    n: int,
) -> np.ndarray:
    """max(values[i .. i+n-1]), NaN near the end."""

    if n == 1:
        return values.astype(
            float
        ).copy()

    s = pd.Series(
        values.astype(
            float
        )
    )

    return (
        s.iloc[::-1]
        .rolling(
            n,
            min_periods=(
                n
            ),
        )
        .max()
        .iloc[::-1]
        .to_numpy()
    )


def forward_rolling_min(
    values: np.ndarray,
    n: int,
) -> np.ndarray:

    if n == 1:
        return values.astype(
            float
        ).copy()

    s = pd.Series(
        values.astype(
            float
        )
    )

    return (
        s.iloc[::-1]
        .rolling(
            n,
            min_periods=(
                n
            ),
        )
        .min()
        .iloc[::-1]
        .to_numpy()
    )


def build_events(
    five: pd.DataFrame,
    width_by_row: np.ndarray,
) -> pd.DataFrame:
    """One row per confirmed breakout."""

    f = (
        five
        .sort_values(
            "bar_start_time"
        )
        .reset_index(
            drop=True
        )
    )

    n5 = len(
        f
    )

    cs = continuity_prefix(
        f
    )

    o = (
        f[
            "open"
        ].to_numpy(
            dtype=float
        )
    )

    hi = (
        f[
            "high"
        ].to_numpy(
            dtype=float
        )
    )

    lo = (
        f[
            "low"
        ].to_numpy(
            dtype=float
        )
    )

    cl = (
        f[
            "close"
        ].to_numpy(
            dtype=float
        )
    )

    # Prior range = the six closed bars before the
    # confirmation bar.
    prior_high = (
        pd.Series(
            hi
        )
        .rolling(
            TRIGGER_BARS,
            min_periods=(
                TRIGGER_BARS
            ),
        )
        .max()
        .shift(
            1
        )
        .to_numpy()
    )

    prior_low = (
        pd.Series(
            lo
        )
        .rolling(
            TRIGGER_BARS,
            min_periods=(
                TRIGGER_BARS
            ),
        )
        .min()
        .shift(
            1
        )
        .to_numpy()
    )

    b = np.arange(
        n5
    )

    up = cl > prior_high

    down = cl < prior_low

    fired = np.where(
        up,
        1,
        np.where(
            down,
            -1,
            0,
        ),
    )

    # Whole span gap-free and in bounds.
    contiguous = span_ok(
        cs,
        b
        - TRIGGER_BARS,
        BASE_SPAN,
        n5,
    ) & (
        (
            b
            - TRIGGER_BARS
        )
        >= 0
    )

    has_width = ~np.isnan(
        width_by_row
    )

    keep = (
        (
            fired
            != 0
        )
        & contiguous
        & has_width
    )

    d = (
        b
        + 1
    )[
        keep
    ]

    direction = fired[
        keep
    ]

    width_ev = (
        width_by_row[
            keep
        ]
    )

    entry = o[
        d
    ]

    rows = {
        "row_id": d,
        "breakout_bar": b[
            keep
        ],
        "decision_time": (
            f[
                "bar_start_time"
            ]
            .to_numpy()[
                d
            ]
        ),
        "direction": (
            direction
        ),
        "prior_high": (
            prior_high[
                keep
            ]
        ),
        "prior_low": (
            prior_low[
                keep
            ]
        ),
        "confirm_close": (
            cl[
                b[
                    keep
                ]
            ]
        ),
        "entry_price": (
            entry
        ),
        "width": (
            width_ev
        ),
    }

    for (
        h,
        n,
    ) in HORIZON_BARS.items():

        exit_close = cl[
            d
            + n
            - 1
        ]

        raw = (
            exit_close
            / entry
            - 1.0
        )

        pmax = (
            forward_rolling_max(
                hi,
                n,
            )[
                d
            ]
            / entry
            - 1.0
        )

        pmin = (
            forward_rolling_min(
                lo,
                n,
            )[
                d
            ]
            / entry
            - 1.0
        )

        # Position: FOLLOW takes the breakout direction,
        # FADE takes the opposite.
        pos_follow = (
            direction.astype(
                float
            )
        )

        rows[
            f"ret_{h}"
        ] = raw

        rows[
            f"follow_{h}"
        ] = (
            pos_follow
            * raw
        )

        rows[
            f"fade_{h}"
        ] = (
            -pos_follow
            * raw
        )

        # Signed excursions in trade space.
        long_side = (
            pos_follow
            > 0
        )

        mfe_follow = np.where(
            long_side,
            pmax,
            -pmin,
        )

        mae_follow = np.where(
            long_side,
            pmin,
            -pmax,
        )

        rows[
            f"mfe_follow_{h}"
        ] = mfe_follow

        rows[
            f"mae_follow_{h}"
        ] = mae_follow

        rows[
            f"mfe_fade_{h}"
        ] = (
            -mae_follow
        )

        rows[
            f"mae_fade_{h}"
        ] = (
            -mfe_follow
        )

    return pd.DataFrame(
        rows
    ).reset_index(
        drop=True
    )


def summarise(
    g: pd.DataFrame,
    h: str,
    strategy: str,
) -> dict:

    col = (
        f"{strategy.lower()}_{h}"
    )

    mfe_col = (
        f"mfe_{strategy.lower()}_{h}"
    )

    mae_col = (
        f"mae_{strategy.lower()}_{h}"
    )

    v = g[
        col
    ].to_numpy(
        dtype=float
    )

    if len(
        v
    ) == 0:
        return {}

    return {
        "n_events": int(
            len(
                v
            )
        ),
        "mean_signed_return": (
            float(
                np.mean(
                    v
                )
            )
        ),
        "median_signed_return": (
            float(
                np.median(
                    v
                )
            )
        ),
        "win_rate": float(
            np.mean(
                v > 0
            )
        ),
        "mean_mfe": float(
            np.mean(
                g[
                    mfe_col
                ].to_numpy(
                    dtype=float
                )
            )
        ),
        "mean_mae": float(
            np.mean(
                g[
                    mae_col
                ].to_numpy(
                    dtype=float
                )
            )
        ),
    }


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

    quant_cols = FEATURE_SETS[
        QUANT_FEATURE_SET
    ]

    event_frames = []
    instrument_rows = []
    fold_rows = []
    pooled_rows = []
    notes = []

    for (
        instrument
    ) in INSTRUMENTS:

        print(
            f"{instrument} ...",
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
            ],
        )

        state = quantile_state(
            build_15m(
                five
            ),
            quant_cols,
        )

        grid = build_execution_grid(
            five,
            state,
        )

        width_by_row = np.full(
            len(
                five
            ),
            np.nan,
        )

        width_by_row[
            grid[
                "row_id"
            ].to_numpy(
                dtype=int
            )
        ] = (
            grid[
                "q90"
            ].to_numpy(
                dtype=float
            )
            - grid[
                "q10"
            ].to_numpy(
                dtype=float
            )
        )

        ev = build_events(
            five,
            width_by_row,
        )

        if ev.empty:
            notes.append(
                f"{instrument}: no events"
            )
            continue

        ev.insert(
            0,
            "instrument",
            instrument,
        )

        # Pooled over every event of this instrument, with no
        # fold restriction. An instrument too small to support
        # fold analysis still has to appear somewhere, otherwise
        # the stated universe quietly shrinks.
        pooled = {
            "instrument": (
                instrument
            ),
            "n_events": int(
                len(
                    ev
                )
            ),
            "up_events": int(
                (
                    ev.direction
                    > 0
                ).sum()
            ),
            "down_events": int(
                (
                    ev.direction
                    < 0
                ).sum()
            ),
        }

        for h in HORIZON_BARS:

            for (
                strategy
            ) in (
                STRATEGIES
            ):

                m = summarise(
                    ev,
                    h,
                    strategy,
                )

                for (
                    k,
                    v,
                ) in m.items():

                    if (
                        k
                        == "n_events"
                    ):
                        continue

                    pooled[
                        f"{strategy.lower()}"
                        f"_{h}_{k}"
                    ] = v

        pooled_rows.append(
            pooled
        )

        # ---- folds over events ----

        for (
            h,
            n,
        ) in HORIZON_BARS.items():

            purge = (
                TRIGGER_BARS
                + 1
                + n
            )

            try:
                folds = (
                    make_folds(
                        len(
                            ev
                        ),
                        horizon=(
                            purge
                        ),
                    )
                )
            except RuntimeError as exc:

                notes.append(
                    f"{instrument} {h}: "
                    f"{exc}"
                )
                continue

            fold_id = np.full(
                len(
                    ev
                ),
                -1,
                dtype=int,
            )

            is_top = np.zeros(
                len(
                    ev
                ),
                dtype=bool,
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

                fold_id[
                    te
                ] = fold[
                    "fold"
                ]

                # Threshold from TRAIN events only.
                cut = float(
                    np.quantile(
                        ev[
                            "width"
                        ]
                        .to_numpy(
                            dtype=float
                        )[
                            tr
                        ],
                        1.0
                        - OPPORTUNITY_TOP_FRACTION,
                    )
                )

                is_top[
                    te
                ] = (
                    ev[
                        "width"
                    ]
                    .to_numpy(
                        dtype=float
                    )[
                        te
                    ]
                    >= cut
                )

            mask = fold_id >= 0

            if not mask.any():
                notes.append(
                    f"{instrument} {h}: "
                    "no test events"
                )
                continue

            ev_h = ev.loc[
                mask
            ].copy()

            ev_h[
                "fold"
            ] = fold_id[
                mask
            ]

            ev_h[
                "is_top30"
            ] = is_top[
                mask
            ]

            ev_h[
                "horizon"
            ] = h

            event_frames.append(
                ev_h
            )

            for (
                strategy
            ) in STRATEGIES:

                for (
                    oset
                ) in (
                    OPPORTUNITY_SETS
                ):

                    g = (
                        ev_h[
                            ev_h[
                                "is_top30"
                            ]
                        ]
                        if oset
                        == "TOP30"
                        else ev_h
                    )

                    m = summarise(
                        g,
                        h,
                        strategy,
                    )

                    if not m:
                        continue

                    m.update(
                        {
                            "instrument": (
                                instrument
                            ),
                            "horizon": (
                                h
                            ),
                            "strategy": (
                                strategy
                            ),
                            "opportunity": (
                                oset
                            ),
                        }
                    )

                    instrument_rows.append(
                        m
                    )

            # ---- per fold ----

            for (
                fid
            ) in sorted(
                set(
                    fold_id[
                        mask
                    ].tolist()
                )
            ):

                gf = ev_h[
                    ev_h[
                        "fold"
                    ]
                    == fid
                ]

                for (
                    strategy
                ) in (
                    STRATEGIES
                ):

                    for (
                        oset
                    ) in (
                        OPPORTUNITY_SETS
                    ):

                        gg = (
                            gf[
                                gf[
                                    "is_top30"
                                ]
                            ]
                            if oset
                            == "TOP30"
                            else gf
                        )

                        m = (
                            summarise(
                                gg,
                                h,
                                strategy,
                            )
                        )

                        if not m:
                            continue

                        m.update(
                            {
                                "instrument": (
                                    instrument
                                ),
                                "horizon": (
                                    h
                                ),
                                "strategy": (
                                    strategy
                                ),
                                "opportunity": (
                                    oset
                                ),
                                "fold": int(
                                    fid
                                ),
                            }
                        )

                        fold_rows.append(
                            m
                        )

    events = pd.concat(
        event_frames,
        ignore_index=(
            True
        ),
    )

    events.to_csv(
        OUT
        / "s1_events.csv",
        index=False,
    )

    by_inst = pd.DataFrame(
        instrument_rows
    )

    by_fold = pd.DataFrame(
        fold_rows
    )

    by_inst.to_csv(
        OUT
        / "s1_by_instrument.csv",
        index=False,
    )

    by_fold.to_csv(
        OUT
        / "s1_by_fold.csv",
        index=False,
    )

    # Pooled over every event of each instrument, no fold
    # restriction, so the stated 16-instrument universe is
    # fully covered (LC has too few events for fold analysis
    # and would otherwise vanish from every table).
    pooled = pd.DataFrame(
        pooled_rows
    )

    pooled.to_csv(
        OUT
        / "s1_pooled_by_instrument.csv",
        index=False,
    )

    # ---- core comparisons ----

    comp = []

    for h in HORIZON_BARS:

        sub_i = by_inst[
            by_inst[
                "horizon"
            ]
            == h
        ]

        sub_f = by_fold[
            by_fold[
                "horizon"
            ]
            == h
        ]

        def pivot(
            df: pd.DataFrame,
        ):
            return {
                (
                    r.strategy,
                    r.opportunity,
                ): r
                for r in df.itertuples()
            }

        # 1. FOLLOW vs zero, and 2. FOLLOW vs FADE.
        for (
            oset
        ) in (
            OPPORTUNITY_SETS
        ):

            fi = sub_i[
                (
                    sub_i.strategy
                    == "FOLLOW"
                )
                & (
                    sub_i.opportunity
                    == oset
                )
            ]

            ff = sub_f[
                (
                    sub_f.strategy
                    == "FOLLOW"
                )
                & (
                    sub_f.opportunity
                    == oset
                )
            ]

            if len(
                fi
            ):

                comp.append(
                    {
                        "horizon": h,
                        "comparison": (
                            f"FOLLOW {oset} "
                            "vs zero "
                            "(instrument)"
                        ),
                        "unit": (
                            "instrument"
                        ),
                        "n_units": int(
                            len(
                                fi
                            )
                        ),
                        "n_positive": int(
                            (
                                fi.mean_signed_return
                                > 0
                            ).sum()
                        ),
                        "positive_share": float(
                            (
                                fi.mean_signed_return
                                > 0
                            ).mean()
                        ),
                        "median_of_unit_means": float(
                            fi.mean_signed_return.median()
                        ),
                    }
                )

            if len(
                ff
            ):

                comp.append(
                    {
                        "horizon": h,
                        "comparison": (
                            f"FOLLOW {oset} "
                            "vs zero "
                            "(fold)"
                        ),
                        "unit": "fold",
                        "n_units": int(
                            len(
                                ff
                            )
                        ),
                        "n_positive": int(
                            (
                                ff.mean_signed_return
                                > 0
                            ).sum()
                        ),
                        "positive_share": float(
                            (
                                ff.mean_signed_return
                                > 0
                            ).mean()
                        ),
                        "median_of_unit_means": float(
                            ff.mean_signed_return.median()
                        ),
                    }
                )

            # FOLLOW vs FADE on the same units.
            ai = sub_i[
                (
                    sub_i.strategy
                    == "FADE"
                )
                & (
                    sub_i.opportunity
                    == oset
                )
            ]

            if len(
                fi
            ) and len(
                ai
            ):

                m = fi[
                    [
                        "instrument",
                        "mean_signed_return",
                    ]
                ].merge(
                    ai[
                        [
                            "instrument",
                            "mean_signed_return",
                        ]
                    ],
                    on=(
                        "instrument"
                    ),
                    suffixes=(
                        "_follow",
                        "_fade",
                    ),
                )

                d = (
                    m.mean_signed_return_follow
                    - m.mean_signed_return_fade
                )

                comp.append(
                    {
                        "horizon": h,
                        "comparison": (
                            f"FOLLOW vs FADE "
                            f"{oset} "
                            "(instrument)"
                        ),
                        "unit": (
                            "instrument"
                        ),
                        "n_units": int(
                            len(
                                d
                            )
                        ),
                        "n_positive": int(
                            (
                                d > 0
                            ).sum()
                        ),
                        "positive_share": float(
                            (
                                d > 0
                            ).mean()
                        ),
                        "median_of_unit_means": float(
                            d.median()
                        ),
                    }
                )

        # 3. TOP30 FOLLOW vs ALL FOLLOW.
        for unit in (
            "instrument",
            "fold",
        ):

            df = (
                sub_i
                if unit
                == "instrument"
                else sub_f
            )

            a = df[
                (
                    df.strategy
                    == "FOLLOW"
                )
                & (
                    df.opportunity
                    == "ALL"
                )
            ]

            t = df[
                (
                    df.strategy
                    == "FOLLOW"
                )
                & (
                    df.opportunity
                    == "TOP30"
                )
            ]

            if (
                not len(
                    a
                )
            ) or (
                not len(
                    t
                )
            ):
                continue

            key = (
                "instrument"
                if unit
                == "instrument"
                else [
                    "instrument",
                    "fold",
                ]
            )

            m = a[
                list(
                    np.atleast_1d(
                        key
                    )
                )
                + [
                    "mean_signed_return"
                ]
            ].merge(
                t[
                    list(
                        np.atleast_1d(
                            key
                        )
                    )
                    + [
                        "mean_signed_return"
                    ]
                ],
                on=key,
                suffixes=(
                    "_all",
                    "_top30",
                ),
            )

            d = (
                m.mean_signed_return_top30
                - m.mean_signed_return_all
            )

            comp.append(
                {
                    "horizon": h,
                    "comparison": (
                        "TOP30 FOLLOW vs "
                        "ALL FOLLOW"
                    ),
                    "unit": unit,
                    "n_units": int(
                        len(
                            d
                        )
                    ),
                    "n_positive": int(
                        (
                            d > 0
                        ).sum()
                    ),
                    "positive_share": float(
                        (
                            d > 0
                        ).mean()
                    ),
                    "median_of_unit_means": float(
                        d.median()
                    ),
                }
            )

    comparisons = pd.DataFrame(
        comp
    )

    comparisons.to_csv(
        OUT
        / "s1_core_comparisons.csv",
        index=False,
    )

    # ---- Quantile opportunity amplification ----
    # The TOP30 vs ALL gap answers one specific question: when
    # Quantile says a big move is coming, does the post-breakout
    # drift grow and keep its sign? If TOP30 keeps the per-
    # instrument sign of ALL and grows in magnitude, the
    # opportunity signal is amplifying the instrument's own
    # continuation/reversal tendency rather than adding a new
    # direction.
    amp_rows = []

    for h in HORIZON_BARS:

        fi = by_inst[
            (
                by_inst.horizon
                == h
            )
            & (
                by_inst.strategy
                == "FOLLOW"
            )
        ]

        for instrument in sorted(
            fi.instrument.unique()
        ):

            a = fi[
                (
                    fi.instrument
                    == instrument
                )
                & (
                    fi.opportunity
                    == "ALL"
                )
            ]

            t = fi[
                (
                    fi.instrument
                    == instrument
                )
                & (
                    fi.opportunity
                    == "TOP30"
                )
            ]

            if (
                not len(
                    a
                )
            ) or (
                not len(
                    t
                )
            ):
                continue

            a_val = float(
                a.mean_signed_return.iloc[
                    0
                ]
            )

            t_val = float(
                t.mean_signed_return.iloc[
                    0
                ]
            )

            same_sign = (
                np.sign(
                    a_val
                )
                == np.sign(
                    t_val
                )
            ) and a_val != 0

            amp_rows.append(
                {
                    "horizon": h,
                    "instrument": (
                        instrument
                    ),
                    "follow_all": a_val,
                    "follow_top30": t_val,
                    "same_sign": bool(
                        same_sign
                    ),
                    "abs_top30_over_abs_all": float(
                        abs(
                            t_val
                        )
                        / (
                            abs(
                                a_val
                            )
                            + 1e-12
                        )
                    ),
                }
            )

    amp = pd.DataFrame(
        amp_rows
    )

    amp.to_csv(
        OUT
        / "s1_amplification.csv",
        index=False,
    )

    (
        OUT
        / "s1_config.json"
    ).write_text(
        json.dumps(
            {
                "trigger_bars": (
                    TRIGGER_BARS
                ),
                "confirmation": (
                    "close of the bar after the "
                    "trigger window"
                ),
                "entry": (
                    "open of the next bar, never "
                    "inside the triggering bar"
                ),
                "horizon_bars": (
                    HORIZON_BARS
                ),
                "primary_horizon": (
                    PRIMARY_H
                ),
                "exit": (
                    "fixed time, no stop, no "
                    "target"
                ),
                "contiguous_span_bars": (
                    BASE_SPAN
                ),
                "opportunity": {
                    "source": (
                        "frozen strict-OOS "
                        "quantile width "
                        "(q90 - q10)"
                    ),
                    "sets": list(
                        OPPORTUNITY_SETS
                    ),
                    "top_fraction": (
                        OPPORTUNITY_TOP_FRACTION
                    ),
                    "threshold": (
                        "computed inside each "
                        "fold on TRAIN events "
                        "only"
                    ),
                },
                "strategies": list(
                    STRATEGIES
                ),
                "returns": (
                    "simple gross returns, no "
                    "fees, no slippage"
                ),
                "notes": notes,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== S1 CORE COMPARISONS ==="
    )

    with pd.option_context(
        "display.width",
        250,
    ):
        print(
            comparisons.to_string(
                index=False
            )
        )

    print(
        "\n=== S1 AMPLIFICATION "
        "(Quantile TOP30 vs ALL, FOLLOW) ==="
    )

    for h in HORIZON_BARS:

        a = amp[
            amp.horizon
            == h
        ]

        print(
            f"\n  {h}: "
            f"n={len(a)} "
            f"same_sign={int(a.same_sign.sum())}/"
            f"{len(a)} "
            f"median(|TOP30|/|ALL|)="
            f"{a.abs_top30_over_abs_all.median():.2f}"
        )

    print(
        "\n=== S1 POOLED (all 16 instruments, "
        "no fold restriction) ==="
    )

    with pd.option_context(
        "display.width",
        250,
    ):

        cols = [
            c
            for c in pooled.columns
            if c
            in (
                "instrument",
                "n_events",
                "up_events",
                "down_events",
                "follow_H5_mean_signed_return",
                "follow_H15_mean_signed_return",
                "fade_H5_mean_signed_return",
                "fade_H15_mean_signed_return",
            )
        ]

        print(
            pooled[
                cols
            ].to_string(
                index=False
            )
        )

    if notes:
        print(
            "\n=== NOTES ==="
        )
        for n_ in notes:
            print(
                "  ",
                n_,
            )

    print(
        "\nS1_BREAKOUT_DONE"
    )


if __name__ == "__main__":
    main()
