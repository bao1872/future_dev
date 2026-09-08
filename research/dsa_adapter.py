"""Canonical DSA adapter.

Source repo:
    bao1872/market_dev

Source SHA (current HEAD at time of authoring):
    8686b803c53c3a423badb80491fbd21f06879fbb

Source paths (canonical DSA SSOT in market_dev):
    backend/app/strategy/selectors/dsa_selector.py
        - compute_dsa_bundle / compute_dsa_history (SSOT, calculation layer)
        - _remove_dsa_lookahead
        - MIN_DIR_BARS
    backend/app/strategy_assets/algorithms/features/dynamic_swing_anchored_vwap.py
        - DSAConfig
        - dynamic_swing_anchored_vwap (Pine v6 Zeiierman kernel, row-exact)
    backend/app/strategy_assets/algorithms/features/atr_rope_event_factor_lab_v4.py
        - ATRRopeConfig
        - compute_atr_rope

The canonical DSA mathematical semantics are NOT redefined here. They are
consumed 1:1 from ``future_dev/panji_indicators.py`` (imported below), which
is the frozen, declared-canonical (AGENTS.md) 1:1 extraction of the
market_dev DSA sources cited above. The ``dynamic_swing_anchored_vwap`` kernel
in ``panji_indicators`` is byte-identical to the current market_dev kernel
(verified by source inspection at market_dev SHA 8686b80).

This module only:
  * prepares the input frame to the canonical contract (OHLCV + amount),
  * calls the canonical ``compute_dsa_history``,
  * projects the result into the stable research interface
    (``bar_index`` + ``dsa_direction`` + ``dsa_raw_*``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from panji_indicators import (
    ATRRopeConfig,
    DSAConfig,
    MIN_DIR_BARS,
    compute_dsa_history,
)


def compute_dsa_canonical(bars: pd.DataFrame) -> pd.DataFrame:
    """Return one row per input bar.

    The mathematical implementation MUST come from validated
    market_dev canonical DSA (consumed 1:1 via ``panji_indicators``).

    Required normalized interface:
        bar_index      : int, 0..n-1 (positional, matches caller's frame)
        dsa_direction  : +1 bullish / 0 flat / -1 bearish
                         (mapped directly from canonical ``regime_value``)

    Every additional canonical DSA state field is exported with prefix
    ``dsa_raw_`` (e.g. ``dsa_raw_regime_value``, ``dsa_raw_dsa_vwap``,
    ``dsa_raw_dsa_dir_bars``, ``dsa_raw_trend_transition``, ...).

    Notes:
        * Canonical DSA requires open/high/low/close/volume/amount. In this
          dataset ``amount`` is an unusable decode artifact, so it is fed as
          NaN; the amount-derived canonical fields therefore become NaN
          rather than fabricated values.
        * ``bar_start_time`` is used as the frame index so canonical segment
          timestamp fields are meaningful.
    """
    if bars is None or len(bars) == 0:
        raise ValueError("compute_dsa_canonical: empty bars")

    df = bars.copy()
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(
                f"compute_dsa_canonical: missing required column {col!r}"
            )
    if "amount" not in df.columns:
        df["amount"] = np.nan

    if "bar_start_time" in df.columns:
        df = df.set_index(
            pd.DatetimeIndex(pd.to_datetime(df["bar_start_time"]))
        )
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    config = {
        "dsa_config": DSAConfig(),
        "rope_config": ATRRopeConfig(regime_lookback=55),
        "min_dir_bars": MIN_DIR_BARS,
    }

    history = compute_dsa_history(df, config)
    if history is None or len(history) == 0:
        raise RuntimeError("compute_dsa_canonical: canonical returned empty")

    out = history.reset_index(drop=True)
    out.insert(0, "bar_index", np.arange(len(out), dtype=int))

    # dsa_direction: canonical flat-aware direction state (regime_value).
    out["dsa_direction"] = out["regime_value"].fillna(0).astype(int)

    rename = {
        c: f"dsa_raw_{c}"
        for c in out.columns
        if c not in ("bar_index", "dsa_direction")
    }
    out = out.rename(columns=rename)
    return out
