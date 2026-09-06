#!/usr/bin/env python3
"""Closure check for the V3R data window.

V3R was run on a frozen window of 2025-01-02 09:00 to 2026-09-04
15:00, but the downloader never enforced the end. Every instrument
only stopped there because the old drop_incomplete_tail() deleted
the whole trading day containing any future bar. That is a side
effect, not a design, and it moves with the download clock.

The endpoint is now explicit. This check answers one question:

    with the corrected downloader (fixed drop_incomplete_tail +
    frozen CALENDAR_END), is the data inside the frozen window
    byte-identical to what V3R actually consumed?

If yes, V3R needs no re-run, because nothing it saw has changed.
If any instrument differs before the endpoint, that is a STOP
condition and the delta has to be reported.

Bars after the endpoint are expected to exist in a fresh download
and are expected to be excluded. They are recorded separately and
must NOT be counted as "new V3R rows".
"""

from __future__ import annotations

import hashlib
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

from market_data.pytdx_source import (  # noqa: E402
    connect,
    download_5m_l8,
    drop_incomplete_tail,
)

from research.download_v3r_5m import (  # noqa: E402
    CALENDAR_END,
    CALENDAR_START,
    INSTRUMENTS,
    KEEP_COLUMNS,
    MAX_PAGES,
)


COMMITTED = (
    ROOT
    / "research"
    / "exports"
    / "v3r_5m"
)

OUT = (
    ROOT
    / "research"
    / "exports"
    / "v3r_window_closure"
)


def content_hash(
    df: pd.DataFrame,
) -> str:
    """Hash of the frame contents.

    Two things matter here.

    Timestamps are compared as ISO strings. Calling .tobytes() on
    an object-dtype array returns pointer bytes rather than the
    string content, so it produces different hashes for frames
    that hold identical values.

    Prices are hashed at float32 precision, because that is what
    the vendor serves. Reading a CSV back gives the nearest
    float64 to a decimal string, which can sit one float32 ULP
    away from the value obtained by widening the original float32.
    Those are the same bar, not a data change.
    """

    d = df.sort_values(
        "bar_start_time"
    ).reset_index(
        drop=True
    )

    h = hashlib.sha256()

    for col in KEEP_COLUMNS:

        v = d[
            col
        ]

        if (
            pd.api.types
            .is_datetime64_any_dtype(
                v
            )
        ):

            text = (
                "\x1f".join(
                    v.dt.strftime(
                        "%Y-%m-%d "
                        "%H:%M:%S"
                    )
                    .fillna(
                        ""
                    )
                    .tolist()
                )
            )

        elif (
            pd.api.types
            .is_integer_dtype(
                v
            )
        ):

            text = (
                "\x1f".join(
                    v.astype(
                        "int64"
                    )
                    .astype(
                        str
                    )
                    .tolist()
                )
            )

        else:

            text = (
                "\x1f".join(
                    map(
                        repr,
                        v.astype(
                            np.float32
                        ).tolist(),
                    )
                )
            )

        h.update(
            text.encode(
                "utf-8"
            )
        )

        h.update(
            b"\x1e"
        )

    return h.hexdigest()


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

    start = pd.Timestamp(
        CALENDAR_START
    )

    end = pd.Timestamp(
        CALENDAR_END
    )

    now = pd.Timestamp.now()

    api = connect()

    rows = []

    try:

        for (
            instrument
        ) in INSTRUMENTS:

            print(
                f"{instrument} ...",
                flush=True,
            )

            fresh = (
                download_5m_l8(
                    instrument,
                    api=api,
                    max_pages=(
                        MAX_PAGES
                    ),
                    not_before=(
                        start
                    ),
                )
            )

            if fresh.empty:
                raise RuntimeError(
                    f"{instrument}: no bars"
                )

            fresh = (
                drop_incomplete_tail(
                    fresh,
                    now=now,
                )
            )

            fresh = fresh[
                fresh[
                    "bar_start_time"
                ]
                >= start
            ].copy()

            after_start = int(
                len(
                    fresh
                )
            )

            fresh = fresh[
                fresh[
                    "availability_time"
                ]
                <= end
            ].copy()

            fresh = fresh[
                KEEP_COLUMNS
            ].copy()

            comm = pd.read_csv(
                COMMITTED
                / f"{instrument}_5m.csv",
                parse_dates=[
                    "bar_start_time",
                    "bar_end_time",
                    "availability_time",
                    "trading_day",
                    "tdx_datetime_raw",
                ],
            )

            comm = comm[
                KEEP_COLUMNS
            ].copy()

            comm = comm.sort_values(
                "bar_start_time"
            ).reset_index(
                drop=True
            )

            fresh = (
                fresh
                .sort_values(
                    "bar_start_time"
                )
                .reset_index(
                    drop=True
                )
            )

            hash_equal = (
                content_hash(
                    fresh
                )
                == content_hash(
                    comm
                )
            )

            # Column-level check, only meaningful when the
            # row counts agree.
            col_equal: dict[
                str, object
            ] = {}

            if len(
                fresh
            ) == len(
                comm
            ):

                for col in (
                    "open",
                    "high",
                    "low",
                    "close",
                    "trade",
                    "position",
                ):

                    # Prices come from the vendor as float32.
                    # Compare at that precision, otherwise a CSV
                    # round-trip shows up as a one-ULP "change"
                    # that is not there.
                    av = (
                        fresh[
                            col
                        ]
                        .to_numpy(
                            dtype=float
                        )
                        .astype(
                            np.float32
                        )
                    )

                    bv = (
                        comm[
                            col
                        ]
                        .to_numpy(
                            dtype=float
                        )
                        .astype(
                            np.float32
                        )
                    )

                    col_equal[
                        col
                    ] = bool(
                        np.array_equal(
                            av,
                            bv,
                            equal_nan=(
                                True
                            ),
                        )
                    )

            else:

                col_equal = {
                    col: (
                        "n/a"
                    )
                    for col in (
                        "open",
                        "high",
                        "low",
                        "close",
                        "trade",
                        "position",
                    )
                }

            rows.append(
                {
                    "instrument": (
                        instrument
                    ),
                    "committed_rows": int(
                        len(
                            comm
                        )
                    ),
                    "fresh_rows_in_window": int(
                        len(
                            fresh
                        )
                    ),
                    "rows_equal": (
                        len(
                            comm
                        )
                        == len(
                            fresh
                        )
                    ),
                    "committed_first": str(
                        comm[
                            "bar_start_time"
                        ].min()
                    ),
                    "fresh_first": str(
                        fresh[
                            "bar_start_time"
                        ].min()
                    ),
                    "committed_last": str(
                        comm[
                            "bar_end_time"
                        ].max()
                    ),
                    "fresh_last": str(
                        fresh[
                            "bar_end_time"
                        ].max()
                    ),
                    "first_equal": str(
                        comm[
                            "bar_start_time"
                        ].min()
                    )
                    == str(
                        fresh[
                            "bar_start_time"
                        ].min()
                    ),
                    "last_equal": str(
                        comm[
                            "bar_end_time"
                        ].max()
                    )
                    == str(
                        fresh[
                            "bar_end_time"
                        ].max()
                    ),
                    "hash_equal": (
                        hash_equal
                    ),
                    **{
                        f"{c}_equal": (
                            col_equal[
                                c
                            ]
                        )
                        for c in (
                            "open",
                            "high",
                            "low",
                            "close",
                            "trade",
                            "position",
                        )
                    },
                    "rows_after_endpoint_excluded": (
                        after_start
                        - int(
                            len(
                                fresh
                            )
                        )
                    ),
                }
            )

    finally:

        api.close()

    res = pd.DataFrame(
        rows
    )

    res.to_csv(
        OUT
        / "window_closure_comparison.csv",
        index=False,
    )

    identical = (
        res[
            "hash_equal"
        ]
        == True  # noqa: E712
    )

    verdict = (
        "IDENTICAL"
        if bool(
            identical.all()
        )
        else "DIFFERENCE_FOUND"
    )

    summary = {
        "calendar_start": (
            CALENDAR_START
        ),
        "calendar_end": (
            CALENDAR_END
        ),
        "instruments": int(
            len(
                res
            )
        ),
        "identical_instruments": int(
            identical.sum()
        ),
        "verdict": verdict,
        "note": (
            "Bars after the endpoint exist in a "
            "fresh download and are excluded "
            "here. They are outside the frozen "
            "V3R window and are not new V3R "
            "rows."
        ),
    }

    (
        OUT
        / "window_closure_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=(
                False
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== WINDOW CLOSURE COMPARISON ==="
    )

    with pd.option_context(
        "display.width",
        250,
    ):
        print(
            res.to_string(
                index=False
            )
        )

    print(
        f"\nverdict = {verdict}"
    )

    print(
        "V3R_WINDOW_CLOSURE_DONE"
    )


if __name__ == "__main__":
    main()
