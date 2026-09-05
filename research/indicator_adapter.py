from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from panji_indicators import (
    DSAConfig,
    _remove_dsa_lookahead,
    build_momentum_history,
    compute_smc_pine,
    compute_sqzmom_lb,
    dynamic_swing_anchored_vwap,
)


@dataclass
class CanonicalBundle:
    dsa_vwap: pd.Series
    dsa_dir: pd.Series
    dsa_pivots: list[dict]
    dsa_segments: list[dict]
    smc: dict
    momentum: dict
    momentum_history: dict


def _canonical_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "datetime" not in out.columns:
            raise ValueError("indicator frame requires DatetimeIndex or datetime column")
        out.index = pd.DatetimeIndex(pd.to_datetime(out["datetime"], errors="raise"))
    return out.sort_index()


def compute_canonical_bundle(df: pd.DataFrame) -> CanonicalBundle:
    """Consume Panji canonical calculations without redefining indicator semantics.

    Full history must be passed here. Crop only after this function returns.
    """
    bars = _canonical_frame(df)

    dsa_cfg = DSAConfig()
    dsa_vwap, dsa_dir, pivots, segments = dynamic_swing_anchored_vwap(bars, dsa_cfg)
    dsa_vwap, dsa_dir = _remove_dsa_lookahead(
        bars, dsa_vwap, dsa_dir, dsa_cfg
    )

    times = [ts.isoformat() for ts in bars.index]
    smc = compute_smc_pine(
        bars["open"].astype(float).tolist(),
        bars["high"].astype(float).tolist(),
        bars["low"].astype(float).tolist(),
        bars["close"].astype(float).tolist(),
        times,
        params=None,
        emit_timeline=True,
    )

    momentum = compute_sqzmom_lb(
        bars["open"].to_numpy(float),
        bars["high"].to_numpy(float),
        bars["low"].to_numpy(float),
        bars["close"].to_numpy(float),
    )
    momentum_history = build_momentum_history(
        momentum,
        volume_series=bars["volume"].to_numpy(float),
        times=times,
    )

    return CanonicalBundle(
        dsa_vwap=dsa_vwap,
        dsa_dir=dsa_dir,
        dsa_pivots=pivots,
        dsa_segments=segments,
        smc=smc,
        momentum=momentum,
        momentum_history=momentum_history,
    )
