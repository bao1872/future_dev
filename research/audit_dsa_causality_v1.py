#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from research.audit_ob_candidate_v3_posthoc import (
    build_source_bars,
)

from research.dsa_adapter import (
    compute_dsa_canonical,
)

from research.ob_rl_dataset_v0_spec import (
    VALIDATED_TFS,
)


SYMBOLS = (
    "AG",
    "CU",
    "RB",
    "M",
)

OUT = (
    ROOT
    / "research"
    / "analysis_results"
    / "dsa_causality_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


DISCRETE_FIELDS = (
    "dsa_direction",
    "dsa_raw_regime_value",
    "dsa_raw_dsa_dir_bars",
    "dsa_raw_segment_direction",
)

FLOAT_FIELDS = (
    "dsa_raw_dsa_vwap",
    "dsa_raw_dsa_vwap_dev_pct",
)

ATOL = 1e-10


def compare_discrete(
    a,
    b,
) -> np.ndarray:

    aa = pd.to_numeric(
        a,
        errors="coerce",
    ).to_numpy(float)

    bb = pd.to_numeric(
        b,
        errors="coerce",
    ).to_numpy(float)

    return np.isclose(
        aa,
        bb,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


def compare_float(
    a,
    b,
) -> np.ndarray:

    aa = pd.to_numeric(
        a,
        errors="coerce",
    ).to_numpy(float)

    bb = pd.to_numeric(
        b,
        errors="coerce",
    ).to_numpy(float)

    return np.isclose(
        aa,
        bb,
        rtol=0.0,
        atol=ATOL,
        equal_nan=True,
    )


def audit_symbol_tf(
    symbol: str,
    tf: str,
    bars: pd.DataFrame,
) -> tuple[dict, list[dict]]:

    full = compute_dsa_canonical(
        bars
    )

    required = (
        *DISCRETE_FIELDS,
        *FLOAT_FIELDS,
    )

    missing = [
        c
        for c in required
        if c not in full.columns
    ]

    if missing:
        raise RuntimeError(
            f"{symbol}/{tf}: "
            f"missing DSA fields {missing}"
        )

    raw_dir = (
        pd.to_numeric(
            full[
                "dsa_raw_segment_direction"
            ],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(int)
    )

    flip_indices = (
        np.flatnonzero(
            raw_dir[1:]
            != raw_dir[:-1]
        )
        + 1
    )

    failures = []

    compared_rows = 0

    max_vwap_error = 0.0
    max_dev_error = 0.0

    # --------------------------------------------------------
    # Strong historical proof at every flip:
    #
    # Everything BEFORE flip T in the final corrected history
    # must equal a computation that never saw T.
    # --------------------------------------------------------

    for loc in flip_indices:

        loc = int(loc)

        if loc < 60:
            continue

        prefix_before = (
            compute_dsa_canonical(
                bars.iloc[:loc].copy()
            )
        )

        n = len(
            prefix_before
        )

        if n != loc:
            raise RuntimeError(
                f"{symbol}/{tf}: "
                "prefix length mismatch "
                f"{n} != {loc}"
            )

        reference = (
            full.iloc[:loc]
            .reset_index(drop=True)
        )

        prefix_before = (
            prefix_before
            .reset_index(drop=True)
        )

        compared_rows += n

        for field in DISCRETE_FIELDS:

            same = compare_discrete(
                reference[field],
                prefix_before[field],
            )

            bad = np.flatnonzero(
                ~same
            )

            for j in bad[:20]:
                failures.append(
                    {
                        "symbol":
                            symbol,

                        "tf":
                            tf,

                        "check":
                            "PRE_FLIP_HISTORY",

                        "flip_bar_index":
                            loc,

                        "bar_index":
                            int(j),

                        "field":
                            field,

                        "full":
                            reference[
                                field
                            ].iloc[j],

                        "prefix":
                            prefix_before[
                                field
                            ].iloc[j],
                    }
                )

        for field in FLOAT_FIELDS:

            a = pd.to_numeric(
                reference[field],
                errors="coerce",
            ).to_numpy(float)

            b = pd.to_numeric(
                prefix_before[field],
                errors="coerce",
            ).to_numpy(float)

            same = np.isclose(
                a,
                b,
                rtol=0.0,
                atol=ATOL,
                equal_nan=True,
            )

            finite = (
                np.isfinite(a)
                & np.isfinite(b)
            )

            if finite.any():

                err = float(
                    np.max(
                        np.abs(
                            a[finite]
                            - b[finite]
                        )
                    )
                )

                if (
                    field
                    == "dsa_raw_dsa_vwap"
                ):
                    max_vwap_error = max(
                        max_vwap_error,
                        err,
                    )

                if (
                    field
                    == "dsa_raw_dsa_vwap_dev_pct"
                ):
                    max_dev_error = max(
                        max_dev_error,
                        err,
                    )

            bad = np.flatnonzero(
                ~same
            )

            for j in bad[:20]:

                failures.append(
                    {
                        "symbol":
                            symbol,

                        "tf":
                            tf,

                        "check":
                            "PRE_FLIP_HISTORY",

                        "flip_bar_index":
                            loc,

                        "bar_index":
                            int(j),

                        "field":
                            field,

                        "full":
                            (
                                None
                                if not np.isfinite(
                                    a[j]
                                )
                                else float(a[j])
                            ),

                        "prefix":
                            (
                                None
                                if not np.isfinite(
                                    b[j]
                                )
                                else float(b[j])
                            ),
                    }
                )

        # ----------------------------------------------------
        # The flip bar itself is allowed to know the flip.
        # Its full-history value must equal a prefix that ends
        # exactly at that bar.
        # ----------------------------------------------------

        prefix_at_flip = (
            compute_dsa_canonical(
                bars.iloc[
                    : loc + 1
                ].copy()
            )
        )

        p = (
            prefix_at_flip.iloc[-1]
        )

        f = (
            full.iloc[loc]
        )

        for field in DISCRETE_FIELDS:

            av = pd.to_numeric(
                pd.Series(
                    [f[field]]
                ),
                errors="coerce",
            ).iloc[0]

            bv = pd.to_numeric(
                pd.Series(
                    [p[field]]
                ),
                errors="coerce",
            ).iloc[0]

            same = (
                (
                    pd.isna(av)
                    and pd.isna(bv)
                )
                or av == bv
            )

            if not same:

                failures.append(
                    {
                        "symbol":
                            symbol,

                        "tf":
                            tf,

                        "check":
                            "AT_FLIP",

                        "flip_bar_index":
                            loc,

                        "bar_index":
                            loc,

                        "field":
                            field,

                        "full":
                            av,

                        "prefix":
                            bv,
                    }
                )

        for field in FLOAT_FIELDS:

            av = float(
                f[field]
            )

            bv = float(
                p[field]
            )

            same = np.isclose(
                av,
                bv,
                rtol=0.0,
                atol=ATOL,
                equal_nan=True,
            )

            if not same:

                failures.append(
                    {
                        "symbol":
                            symbol,

                        "tf":
                            tf,

                        "check":
                            "AT_FLIP",

                        "flip_bar_index":
                            loc,

                        "bar_index":
                            loc,

                        "field":
                            field,

                        "full":
                            (
                                None
                                if not np.isfinite(av)
                                else av
                            ),

                        "prefix":
                            (
                                None
                                if not np.isfinite(bv)
                                else bv
                            ),
                    }
                )

    # --------------------------------------------------------
    # Confirmation transitions:
    # 0 -> +/-1 must also be exactly reproducible from a prefix.
    # --------------------------------------------------------

    regime = pd.to_numeric(
        full[
            "dsa_direction"
        ],
        errors="coerce",
    ).fillna(0).to_numpy(int)

    confirmed_now = (
        np.abs(regime)
        == 1
    )

    confirmed_prev = np.r_[
        False,
        confirmed_now[:-1],
    ]

    confirm_indices = np.flatnonzero(
        confirmed_now
        & ~confirmed_prev
    )

    for loc in confirm_indices:

        loc = int(loc)

        if loc < 60:
            continue

        prefix = (
            compute_dsa_canonical(
                bars.iloc[
                    : loc + 1
                ].copy()
            )
        )

        p = prefix.iloc[-1]
        f = full.iloc[loc]

        for field in (
            *DISCRETE_FIELDS,
            *FLOAT_FIELDS,
        ):

            av = f[field]
            bv = p[field]

            if field in DISCRETE_FIELDS:

                same = (
                    (
                        pd.isna(av)
                        and pd.isna(bv)
                    )
                    or float(av)
                    == float(bv)
                )

            else:

                same = np.isclose(
                    float(av),
                    float(bv),
                    rtol=0.0,
                    atol=ATOL,
                    equal_nan=True,
                )

            if not same:

                failures.append(
                    {
                        "symbol":
                            symbol,

                        "tf":
                            tf,

                        "check":
                            "CONFIRMATION_BAR",

                        "flip_bar_index":
                            None,

                        "bar_index":
                            loc,

                        "field":
                            field,

                        "full":
                            av,

                        "prefix":
                            bv,
                    }
                )

    summary = {
        "symbol":
            symbol,

        "tf":
            tf,

        "bars":
            int(len(bars)),

        "flip_count":
            int(
                len(
                    flip_indices
                )
            ),

        "confirmation_count":
            int(
                len(
                    confirm_indices
                )
            ),

        "historical_rows_compared":
            int(
                compared_rows
            ),

        "failure_count":
            int(
                len(
                    failures
                )
            ),

        "max_vwap_abs_error":
            float(
                max_vwap_error
            ),

        "max_dev_pct_abs_error":
            float(
                max_dev_error
            ),
    }

    return (
        summary,
        failures,
    )


def main():

    summaries = []
    failures = []

    for symbol in SYMBOLS:

        bars_by_tf = (
            build_source_bars(
                symbol
            )
        )

        for tf in VALIDATED_TFS:

            print(
                "AUDIT",
                symbol,
                tf,
                flush=True,
            )

            summary, bad = (
                audit_symbol_tf(
                    symbol,
                    tf,
                    bars_by_tf[tf],
                )
            )

            summaries.append(
                summary
            )

            failures.extend(
                bad
            )

    pd.DataFrame(
        summaries
    ).to_csv(
        OUT
        / "summary.csv",
        index=False,
    )

    pd.DataFrame(
        failures
    ).to_csv(
        OUT
        / "failures.csv",
        index=False,
    )

    total_failures = sum(
        x[
            "failure_count"
        ]
        for x in summaries
    )

    audit = {
        "audit":
            "dsa_causality_v1",

        "timeframes":
            list(
                VALIDATED_TFS
            ),

        "symbols":
            list(
                SYMBOLS
            ),

        "atol":
            ATOL,

        "total_failure_count":
            int(
                total_failures
            ),

        "pass":
            bool(
                total_failures
                == 0
            ),
    }

    (
        OUT
        / "audit.json"
    ).write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        )
    )

    if total_failures:
        raise RuntimeError(
            "DSA_CAUSALITY_AUDIT_FAILED"
        )

    print(
        "DSA_CAUSALITY_AUDIT_PASS"
    )


if __name__ == "__main__":
    main()
