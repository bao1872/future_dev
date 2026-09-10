"""Native-direction target-vs-stop first passage 标签状态机。

不设任何固定 horizon：让 2.5R target 与 1R stop 自然分出胜负。
只沿 OB 原生方向判定，禁止 up_success or down_success 这类旧逻辑。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

BULLISH = 1
BEARISH = -1

TARGET_R = 2.5
STOP_R = 1.0


@dataclass
class NativeLabelResult:
    label: Optional[int]
    status: str
    resolution_bar_index: Optional[int]
    bars_to_resolution: Optional[int]
    resolution_price: Optional[float]
    resolution_type: Optional[str]


def label_native_direction_event(
    *,
    start_idx: int,
    reference_price: float,
    r_ref: float,
    native_direction: int,
    open_,
    high,
    low,
    discontinuity_before_bar,
) -> NativeLabelResult:
    assert native_direction in (BULLISH, BEARISH)
    assert r_ref > 0

    d = native_direction

    target = reference_price + d * TARGET_R * r_ref
    stop = reference_price - d * STOP_R * r_ref

    for i in range(start_idx, len(open_)):

        # No label may cross a known continuous-contract discontinuity.
        if discontinuity_before_bar[i]:
            return NativeLabelResult(
                label=None,
                status="ROLL_CENSORED",
                resolution_bar_index=i,
                bars_to_resolution=i - start_idx,
                resolution_price=None,
                resolution_type=None,
            )

        o = float(open_[i])
        h = float(high[i])
        l = float(low[i])

        # 1. Process open first.
        #    A gap may already have crossed target or stop.
        directional_open_R = d * (o - reference_price) / r_ref

        if directional_open_R >= TARGET_R:
            return NativeLabelResult(
                1,
                "RESOLVED",
                i,
                i - start_idx + 1,
                o,
                "TARGET",
            )

        if directional_open_R <= -STOP_R:
            return NativeLabelResult(
                0,
                "RESOLVED",
                i,
                i - start_idx + 1,
                o,
                "STOP",
            )

        # 2. Intrabar target / stop.
        if d == BULLISH:
            target_hit = h >= target
            stop_hit = l <= stop
        else:
            target_hit = l <= target
            stop_hit = h >= stop

        # With only 5m OHLC, ordering is unknowable.
        if target_hit and stop_hit:
            return NativeLabelResult(
                None,
                "AMBIGUOUS_INTRABAR",
                i,
                i - start_idx + 1,
                None,
                None,
            )

        if target_hit:
            return NativeLabelResult(
                1,
                "RESOLVED",
                i,
                i - start_idx + 1,
                target,
                "TARGET",
            )

        if stop_hit:
            return NativeLabelResult(
                0,
                "RESOLVED",
                i,
                i - start_idx + 1,
                stop,
                "STOP",
            )

    return NativeLabelResult(
        label=None,
        status="END_OF_DATA_CENSORED",
        resolution_bar_index=None,
        bars_to_resolution=None,
        resolution_price=None,
        resolution_type=None,
    )
