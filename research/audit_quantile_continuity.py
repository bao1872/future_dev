#!/usr/bin/env python3
"""Quantile Continuity Audit (Q-Audit).

Single, bounded question, decided by the auditor before any new
strategy work:

    The frozen Quantile layer predicts, for each 15m decision bar,
    the dispersion (q90 - q10) of the next ~60 minute return. That
    dispersion is the "opportunity" score. But the target is built
    by build_targets() as an ARRAY-INDEX window i+1..i+H over the
    15m panel, with no check that those H bars are contiguous in
    calendar time. A window that crosses lunch / an overnight gap /
    a weekend is therefore silently treated as "the next H tradeable
    15m bars", not "the next ~60 minutes".

    If the Quantile opportunity edge lives mostly in the CROSS_GAP
    rows, what we found may actually be a session-transition effect
    (supporting S2), not an intraday volatility signal.

This script does NOT change the model or the features. It:

  1. rebuilds the exact frozen 15m panel via build_15m();
  2. flags every decision row whose H4 (and H8) target window is
     strictly 15-minute-contiguous vs crosses a gap;
  3. reports cross-gap contamination per instrument;
  4. recomputes the existing core Quantile diagnostics
     (calibration, 80% interval coverage, width -> |return|
     rank correlation, TOP30/BOTTOM30 mean |return|) on the FULL,
     CONTIGUOUS-only and CROSS_GAP-only OOS rows.

The Quantile predictions themselves come from the unchanged
quantile_state(); only the row stratification is new.
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

from research.run_direction_v0 import (  # noqa: E402
    quantile_state,
)

from research.run_direction_v3r import (  # noqa: E402
    INSTRUMENTS,
    QUANT_FEATURE_SET,
    QUANT_HORIZON,
    SRC_5M,
    build_15m,
)

from research.run_quantile_rebaseline import (  # noqa: E402
    FEATURE_SETS,
)


OUT = (
    ROOT
    / "research"
    / "exports"
    / "quantile_continuity_audit"
)

CONTIG_DELTA_SEC = 15 * 60


def continuity_flags(
    panel: pd.DataFrame,
    horizon: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Per decision row, is the target window i+1..i+H strictly
    contiguous (every adjacent 15m bar exactly 15 min apart)?

    Returns (contig_bool, max_gap_minutes) aligned to panel index.
    """

    t = panel[
        "meta_base_bar_time"
    ].to_numpy(
        dtype="datetime64[s]"
    )

    n = len(
        panel
    )

    contig = np.zeros(
        n,
        dtype=bool,
    )

    maxgap = np.full(
        n,
        np.nan,
    )

    for i in range(
        n - horizon
    ):

        seg = t[
            i + 1 : i + 1 + horizon
        ]

        deltas = (
            np.diff(
                seg
            )
            .astype(
                "timedelta64[s]"
            )
            .astype(
                "int64"
            )
        )

        contig[
            i
        ] = bool(
            np.all(
                deltas
                == CONTIG_DELTA_SEC
            )
        )

        if len(
            deltas
        ):
            maxgap[
                i
            ] = float(
                deltas.max()
            ) / 60.0

    return contig, maxgap


def spearman(
    x: np.ndarray,
    y: np.ndarray,
) -> float:

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    m = (
        np.isfinite(
            x
        )
        & np.isfinite(
            y
        )
    )

    x, y = x[
        m
    ], y[
        m
    ]

    if len(
        x
    ) < 3:
        return np.nan

    rx = (
        pd.Series(
            x
        )
        .rank()
        .to_numpy()
    )

    ry = (
        pd.Series(
            y
        )
        .rank()
        .to_numpy()
    )

    rx = rx - rx.mean()

    ry = ry - ry.mean()

    denom = np.sqrt(
        (
            rx**2
        ).sum()
        * (
            ry**2
        ).sum()
    )

    return (
        float(
            (
                rx
                * ry
            ).sum()
            / denom
        )
        if denom > 0
        else np.nan
    )


def diag(
    sub: pd.DataFrame,
) -> dict | None:

    y = sub[
        "y"
    ].to_numpy(
        dtype=float
    )

    lo = sub[
        "q10"
    ].to_numpy(
        dtype=float
    )

    hi = sub[
        "q90"
    ].to_numpy(
        dtype=float
    )

    m = (
        np.isfinite(
            y
        )
        & np.isfinite(
            lo
        )
        & np.isfinite(
            hi
        )
    )

    y, lo, hi = (
        y[
            m
        ],
        lo[
            m
        ],
        hi[
            m
        ],
    )

    if (
        len(
            y
        )
        < 10
    ):
        return None

    width = hi - lo

    ay = np.abs(
        y
    )

    order = np.argsort(
        width
    )

    k = max(
        1,
        int(
            0.3
            * len(
                width
            )
        ),
    )

    top = float(
        ay[
            order[
                -k:
            ]
        ].mean()
    )

    bot = float(
        ay[
            order[
                :k
            ]
        ].mean()
    )

    return {
        "n": int(
            len(
                y
            )
        ),
        "cal_q10": float(
            (
                y
                <= lo
            ).mean()
        ),
        "cal_q90": float(
            (
                y
                <= hi
            ).mean()
        ),
        "interval_80_cov": float(
            (
                (
                    y
                    >= lo
                )
                & (
                    y
                    <= hi
                )
            ).mean()
        ),
        "width_spearman_absret": spearman(
            width,
            ay,
        ),
        "top30_mean_absret": top,
        "bottom30_mean_absret": bot,
        "top30_over_bottom30": (
            top
            / bot
            if bot > 0
            else np.nan
        ),
    }


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

    contig_rows = []
    diag_rows = []

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

        panel = build_15m(
            five
        )

        contig_h4, maxgap_h4 = continuity_flags(
            panel,
            4,
        )

        contig_h8, maxgap_h8 = continuity_flags(
            panel,
            8,
        )

        panel[
            "contig_h4"
        ] = contig_h4

        panel[
            "maxgap_h4_min"
        ] = maxgap_h4

        panel[
            "contig_h8"
        ] = contig_h8

        panel[
            "maxgap_h8_min"
        ] = maxgap_h8

        valid4 = panel[
            "target_raw_return_h4"
        ].notna().to_numpy()

        valid8 = panel[
            "target_raw_return_h8"
        ].notna().to_numpy()

        cg4 = (
            ~contig_h4
        ) & valid4

        cg8 = (
            ~contig_h8
        ) & valid8

        contig_rows.append(
            {
                "instrument": instrument,
                "h4_total": int(
                    valid4.sum()
                ),
                "h4_cross_gap": int(
                    cg4.sum()
                ),
                "h4_pct_cross_gap": float(
                    cg4.sum()
                    / max(
                        1,
                        valid4.sum(),
                    )
                ),
                "h4_median_gap_min": (
                    float(
                        np.nanmedian(
                            panel[
                                "maxgap_h4_min"
                            ].to_numpy()[
                                cg4
                            ]
                        )
                    )
                    if cg4.any()
                    else np.nan
                ),
                "h8_total": int(
                    valid8.sum()
                ),
                "h8_cross_gap": int(
                    cg8.sum()
                ),
                "h8_pct_cross_gap": float(
                    cg8.sum()
                    / max(
                        1,
                        valid8.sum(),
                    )
                ),
                "h8_median_gap_min": (
                    float(
                        np.nanmedian(
                            panel[
                                "maxgap_h8_min"
                            ].to_numpy()[
                                cg8
                            ]
                        )
                    )
                    if cg8.any()
                    else np.nan
                ),
            }
        )

        # Frozen Quantile predictions, unchanged model/features.
        state = quantile_state(
            panel,
            FEATURE_SETS[
                QUANT_FEATURE_SET
            ],
        )

        merged = state.merge(
            panel[
                [
                    "meta_decision_time",
                    "contig_h4",
                    "contig_h8",
                    "target_raw_return_h4",
                ]
            ].rename(
                columns={
                    "target_raw_return_h4": "y",
                }
            ),
            left_on="decision_time",
            right_on="meta_decision_time",
            how="left",
        )

        contig = merged[
            "contig_h4"
        ].fillna(
            False
        ).to_numpy()

        for split_name, mask in [
            (
                "FULL",
                np.ones(
                    len(
                        merged
                    ),
                    dtype=bool,
                ),
            ),
            (
                "CONTIG",
                contig,
            ),
            (
                "CROSS_GAP",
                ~contig,
            ),
        ]:

            d = diag(
                merged[
                    mask
                ]
            )

            if d is None:
                continue

            d.update(
                {
                    "instrument": instrument,
                    "split": split_name,
                    "horizon": (
                        f"H{QUANT_HORIZON}"
                    ),
                }
            )

            diag_rows.append(
                d
            )

    contig_df = pd.DataFrame(
        contig_rows
    )

    diag_df = pd.DataFrame(
        diag_rows
    )

    contig_df.to_csv(
        OUT
        / "qaudit_continuity.csv",
        index=False,
    )

    diag_df.to_csv(
        OUT
        / "qaudit_diagnostics.csv",
        index=False,
    )

    summary = {
        "question": (
            "Does the frozen Quantile opportunity edge live in "
            "intraday-contiguous targets or in cross-gap "
            "(session/overnight) targets?"
        ),
        "model": "unchanged frozen Quantile (build_15m + "
                 "quantile_state, F1_VOL, H4)",
        "h4_total_rows": int(
            contig_df.h4_total.sum()
        ),
        "h4_cross_gap_rows": int(
            contig_df.h4_cross_gap.sum()
        ),
        "h4_pct_cross_gap": float(
            contig_df.h4_cross_gap.sum()
            / max(
                1,
                contig_df.h4_total.sum(),
            )
        ),
        "h8_total_rows": int(
            contig_df.h8_total.sum()
        ),
        "h8_cross_gap_rows": int(
            contig_df.h8_cross_gap.sum()
        ),
        "h8_pct_cross_gap": float(
            contig_df.h8_cross_gap.sum()
            / max(
                1,
                contig_df.h8_total.sum(),
            )
        ),
    }

    (
        OUT
        / "qaudit_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== Q-AUDIT: H4/H8 TARGET CONTINUITY ==="
    )

    with pd.option_context(
        "display.width",
        220,
    ):
        print(
            contig_df.to_string(
                index=False
            )
        )

    print(
        "\n=== Q-AUDIT: CORE DIAGNOSTICS (FULL / CONTIG / "
        "CROSS_GAP) ==="
    )

    with pd.option_context(
        "display.width",
        260,
    ):
        print(
            diag_df.to_string(
                index=False
            )
        )

    print(
        "\nQ_AUDIT_DONE"
    )


if __name__ == "__main__":
    main()
