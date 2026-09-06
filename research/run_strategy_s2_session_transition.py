#!/usr/bin/env python3
"""Strategy Exploration S2 - Session Transition (V1, price only).

S1 asked whether waiting for a confirmed breakout rescues a
direction edge. It did not (H5 44% / H15 52% of folds positive,
no universal continuation). The Quantile layer that was meant to
supply "opportunity" is now QUARANTINED: its H4/H8 targets were
built as array-index windows i+1..i+H with no calendar check, so a
large fraction of them silently span lunch / overnight / weekend
gaps. Its edge is neither proven clean nor discarded; it is simply
not used here.

S2 asks a different, market-structure question that needs no
Quantile at all:

    After a clear price pressure in the LAST 15/30 minutes of a
    session, does the NEXT session's open 5/15/30 minutes
    CONTINUE or REVERSE that pressure?

Chinese commodity futures are not a 24h continuous market. There
are explicit breaks: morning->lunch->afternoon, day close ->
night open, night close -> next day, and weekends/holidays. A
session tail that is heavily sold or bought may either exhaust
itself (reversal at the next open) or persist (continuation).

V1 is deliberately minimal. No model, no ML, no parameter search.
The previous-session state is binned into quintiles of its own
return (strong down .. strong up), and we look at the raw future
mean / median / win-rate of the next-session open-to-5/15/30m
return. This is a descriptive mechanism test, not an OOS-validated
trading rule.

Sessions are detected purely from 15-minute calendar gaps in the
5m bars: a run of bars spaced exactly 15 minutes apart is one
session; any larger gap is a transition (lunch / overnight /
long gap). The previous-session state uses only bars inside the
previous session; the target uses only bars inside the next
session. Nothing crosses the gap.

OI is intentionally excluded from V1 (the source `position` column
is the L8 continuous series with known roll jumps). V2 can add a
cleaned delta-OI and a volume surprise on top of the raw columns
stored here.
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

if str(
    ROOT
) not in sys.path:
    sys.path.insert(
        0,
        str(
            ROOT
        ),
    )

from research.run_direction_v3r import (  # noqa: E402
    INSTRUMENTS,
    SRC_5M,
)


OUT = (
    ROOT
    / "research"
    / "exports"
    / "session_transition_s2"
)

# Target horizons measured from the next session open, in minutes.
TARGET_MIN = [
    5,
    15,
    30,
]

STATE_VARS = [
    "prev_ret_15m",
    "prev_ret_30m",
]

# A within-session 5m bar is exactly 15 minutes after the previous
# one. Anything larger is a session break.
WITHIN_SESSION_SEC = 15 * 60


def detect_sessions(
    t: np.ndarray,
) -> np.ndarray:
    """Return a per-row session id from 15-minute calendar gaps."""

    n = len(
        t
    )

    if n == 0:
        return np.zeros(
            0,
            dtype=int,
        )

    gap_sec = (
        np.diff(
            t
        )
        .astype(
            "timedelta64[s]"
        )
        .astype(
            "int64"
        )
    )

    # Indices where a new session begins.
    starts = (
        np.where(
            gap_sec
            > WITHIN_SESSION_SEC
        )[0]
        + 1
    )

    sess_id = np.zeros(
        n,
        dtype=int,
    )

    for s in starts:
        sess_id[
            s:
        ] += 1

    return sess_id


def gap_type(
    gap_min: float,
) -> str:

    if gap_min <= 240:
        return "lunch"

    if gap_min <= 1440:
        return "overnight"

    return "long_gap"


def build_events(
    instrument: str,
    five: pd.DataFrame,
) -> pd.DataFrame:

    t = five[
        "bar_start_time"
    ].to_numpy(
        dtype="datetime64[s]"
    )

    o = five[
        "open"
    ].to_numpy(
        dtype=float
    )

    c = five[
        "close"
    ].to_numpy(
        dtype=float
    )

    v = five[
        "trade"
    ].to_numpy(
        dtype=float
    )

    n = len(
        five
    )

    sess_id = detect_sessions(
        t
    )

    rows = []

    # Group bar indices by session id.
    sess_bounds = np.concatenate(
        [
            [
                0,
            ],
            np.where(
                np.diff(
                    sess_id
                )
                != 0
            )[0]
            + 1,
            [
                n,
            ],
        ]
    )

    for k in range(
        len(
            sess_bounds
        )
        - 1
    ):

        prev = np.arange(
            sess_bounds[
                k
            ],
            sess_bounds[
                k
                + 1
            ],
        )

        if k + 1 >= len(
            sess_bounds
        ) - 1:
            break

        nxt = np.arange(
            sess_bounds[
                k
                + 1
            ],
            sess_bounds[
                k
                + 2
            ],
        )

        # Need at least 6 bars (30 min) for both the 30m state and
        # the 30m target.
        if (
            len(
                prev
            )
            < 6
            or len(
                nxt
            )
            < 6
        ):
            continue

        prev_ret_15m = (
            c[
                prev[
                    -1
                ]
            ]
            / c[
                prev[
                    -3
                ]
            ]
            - 1.0
        )

        prev_ret_30m = (
            c[
                prev[
                    -1
                ]
            ]
            / c[
                prev[
                    -6
                ]
            ]
            - 1.0
        )

        open0 = o[
            nxt[
                0
            ]
        ]

        if not np.isfinite(
            open0
        ) or open0 <= 0:
            continue

        tgt = {
            5: c[
                nxt[
                    0
                ]
            ]
            / open0
            - 1.0,
            15: c[
                nxt[
                    2
                ]
            ]
            / open0
            - 1.0,
            30: c[
                nxt[
                    5
                ]
            ]
            / open0
            - 1.0,
        }

        gap_min = float(
            (
                t[
                    nxt[
                        0
                    ]
                ]
                - t[
                    prev[
                        -1
                    ]
                ]
            )
            / np.timedelta64(
                1,
                "m",
            )
        )

        rows.append(
            {
                "instrument": instrument,
                "prev_session_end": pd.Timestamp(
                    t[
                        prev[
                            -1
                        ]
                    ]
                ),
                "next_session_start": pd.Timestamp(
                    t[
                        nxt[
                            0
                        ]
                    ]
                ),
                "gap_min": gap_min,
                "gap_type": gap_type(
                    gap_min
                ),
                "prev_bars": int(
                    len(
                        prev
                    )
                ),
                "next_bars": int(
                    len(
                        nxt
                    )
                ),
                "prev_ret_15m": float(
                    prev_ret_15m
                ),
                "prev_ret_30m": float(
                    prev_ret_30m
                ),
                "prev_vol": float(
                    v[
                        prev
                    ].sum()
                ),
                "next_open_to_5m": float(
                    tgt[
                        5
                    ]
                ),
                "next_open_to_15m": float(
                    tgt[
                        15
                    ]
                ),
                "next_open_to_30m": float(
                    tgt[
                        30
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def assign_quintiles(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Per-instrument quintile bins of each STATE_VAR.

    Full-sample quintiles are used for this descriptive V1; they
    rank the previous-session return against the instrument's own
    distribution. Bin 1 = strong down, bin 5 = strong up.
    """

    out = events.copy()

    for inst in events[
        "instrument"
    ].unique():

        mask = (
            out.instrument
            == inst
        )

        for sv in STATE_VARS:

            vals = out.loc[
                mask,
                sv,
            ]

            q = vals.quantile(
                [
                    0.2,
                    0.4,
                    0.6,
                    0.8,
                ]
            ).to_numpy()

            if not np.all(
                np.isfinite(
                    q
                )
            ):
                out.loc[
                    mask,
                    f"{sv}_bin",
                ] = np.nan
                continue

            bin_idx = np.digitize(
                vals.to_numpy(),
                q,
            ) + 1  # 1..5

            out.loc[
                mask,
                f"{sv}_bin",
            ] = bin_idx.astype(
                float
            )

    return out


def summarise(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    target_col = {
        5: "next_open_to_5m",
        15: "next_open_to_15m",
        30: "next_open_to_30m",
    }

    for sv in STATE_VARS:

        bin_col = f"{sv}_bin"

        for h in TARGET_MIN:

            tc = target_col[
                h
            ]

            for (
                b,
                g,
            ) in df.groupby(
                bin_col
            ):

                if not np.isfinite(
                    b
                ):
                    continue

                y = g[
                    tc
                ].to_numpy(
                    dtype=float
                )

                y = y[
                    np.isfinite(
                        y
                    )
                ]

                if len(
                    y
                ) == 0:
                    continue

                rows.append(
                    {
                        "state_var": sv,
                        "bin": int(
                            b
                        ),
                        "horizon_min": h,
                        "n": int(
                            len(
                                y
                            )
                        ),
                        "mean": float(
                            y.mean()
                        ),
                        "median": float(
                            np.median(
                                y
                            )
                        ),
                        "win_rate": float(
                            (
                                y
                                > 0
                            ).mean()
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


def main() -> None:

    if OUT.exists() and any(
        OUT.iterdir()
    ):
        raise RuntimeError(
            f"{OUT} exists and is non-empty. "
            "Delete only for an intentional rerun."
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_events = []

    for instrument in INSTRUMENTS:

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

        ev = build_events(
            instrument,
            five,
        )

        if len(
            ev
        ):
            all_events.append(
                ev
            )

    events = pd.concat(
        all_events,
        ignore_index=True,
    )

    events = assign_quintiles(
        events
    )

    events.to_csv(
        OUT
        / "s2_events.csv",
        index=False,
    )

    by_state = summarise(
        events
    )

    # summarise() ignores instrument, so this is already pooled
    # across all 16 instruments.
    by_state.to_csv(
        OUT
        / "s2_by_state.csv",
        index=False,
    )

    # Per-instrument split for heterogeneity inspection.
    inst_rows = []

    for inst in events[
        "instrument"
    ].unique():

        sub = summarise(
            events[
                events.instrument
                == inst
            ]
        )

        sub[
            "instrument"
        ] = inst

        inst_rows.append(
            sub
        )

    by_inst = pd.concat(
        inst_rows,
        ignore_index=True,
    )

    by_inst.to_csv(
        OUT
        / "s2_by_instrument_state.csv",
        index=False,
    )

    # Split by gap type (lunch / overnight / long_gap).
    gap_rows = []

    for gt in [
        "lunch",
        "overnight",
        "long_gap",
    ]:

        sub = summarise(
            events[
                events.gap_type
                == gt
            ]
        )

        sub[
            "gap_type"
        ] = gt

        gap_rows.append(
            sub
        )

    by_gap = pd.concat(
        gap_rows,
        ignore_index=True,
    )

    by_gap.to_csv(
        OUT
        / "s2_by_gaptype.csv",
        index=False,
    )

    (
        OUT
        / "s2_config.json"
    ).write_text(
        json.dumps(
            {
                "name": "S2 Session Transition V1 (price only)",
                "depends_on_quantile": False,
                "quantile_status": "QUARANTINED",
                "session_detection": (
                    "15-minute calendar gaps in 5m bars; "
                    ">15min gap = transition"
                ),
                "prev_state": STATE_VARS,
                "targets": {
                    "5m": "next_open_to_5m",
                    "15m": "next_open_to_15m",
                    "30m": "next_open_to_30m",
                },
                "binning": (
                    "per-instrument full-sample quintiles of "
                    "previous-session return; bin1=strong down, "
                    "bin5=strong up (descriptive, not OOS)"
                ),
                "excluded_from_v1": [
                    "OI (L8 position has roll jumps)",
                    "volume surprise (V2)",
                    "models / ML",
                    "fees / slippage",
                ],
                "gap_types": {
                    "lunch": "<=240 min",
                    "overnight": "240-1440 min",
                    "long_gap": ">1440 min (weekend/holiday)",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---- console summary ----
    print(
        "\n=== S2 EVENTS ==="
    )
    print(
        f"total transitions: {len(events)}"
    )

    print(
        "\n=== S2 POOLED: mean next-session return by state bin ==="
    )

    with pd.option_context(
        "display.width",
        200,
    ):

        for sv in STATE_VARS:

            print(
                f"\n  {sv}:"
            )

            sub = by_state[
                by_state.state_var
                == sv
            ]

            pivot = sub.pivot(
                index="bin",
                columns="horizon_min",
                values="mean",
            )

            print(
                pivot.to_string()
            )

            cnt = sub.pivot(
                index="bin",
                columns="horizon_min",
                values="n",
            )

            print(
                "  (n):"
            )

            print(
                cnt.to_string()
            )

    print(
        "\n=== S2 BY GAP TYPE: mean next_open_to_15m by bin ==="
    )

    with pd.option_context(
        "display.width",
        200,
    ):

        for gt in [
            "lunch",
            "overnight",
            "long_gap",
        ]:

            sub = by_gap[
                (
                    by_gap.gap_type
                    == gt
                )
                & (
                    by_gap.state_var
                    == "prev_ret_15m"
                )
                & (
                    by_gap.horizon_min
                    == 15
                )
            ].sort_values(
                "bin"
            )

            print(
                f"\n  {gt} (n total="
                f"{int(sub.n.sum())}):"
            )

            print(
                sub[
                    [
                        "bin",
                        "n",
                        "mean",
                        "median",
                        "win_rate",
                    ]
                ].to_string(
                    index=False
                )
            )

    print(
        "\nS2_SESSION_TRANSITION_DONE"
    )


if __name__ == "__main__":
    main()
