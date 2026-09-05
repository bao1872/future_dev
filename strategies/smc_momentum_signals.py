from __future__ import annotations

import numpy as np
import pandas as pd

from research.indicator_adapter import compute_smc_momentum_bundle

NAME = "smc_momentum_4h_1h_v0"

DESCRIPTION = (
    "4H SMC swing bias filters direction; "
    "1H BOS/CHoCH triggers; "
    "1H SQZMOM confirms momentum direction. "
    "Signal candidates only, no position state or backtest."
)

BUY_MOMENTUM_COLORS = {"lime", "maroon"}
SELL_MOMENTUM_COLORS = {"red", "green"}


def _bar_available_times(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Availability of bar j = observed start time of bar j+1.

    This avoids assuming nominal duration across futures session breaks.
    The final bar has no observed next-bar transition and is intentionally
    not treated as available for causal alignment.
    """
    if len(index) < 2:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(index[1:])


def _latest_available_higher_index(
    higher_index: pd.DatetimeIndex,
    decision_time: pd.Timestamp,
) -> int | None:
    """Latest higher-TF bar known by decision_time.

    availability[j] = higher_index[j + 1]
    """
    available = _bar_available_times(higher_index)

    if len(available) == 0:
        return None

    k = int(
        available.searchsorted(
            pd.Timestamp(decision_time),
            side="right",
        )
        - 1
    )

    if k < 0:
        return None

    # available[k] belongs to higher bar k
    return k


def generate_signals(
    higher_4h: pd.DataFrame,
    lower_1h: pd.DataFrame,
) -> pd.DataFrame:
    """Generate causal candidate signals.

    No positions, no exits, no PnL.
    """

    higher = higher_4h.sort_index()
    lower = lower_1h.sort_index()

    higher_bundle = compute_smc_momentum_bundle(higher)
    lower_bundle = compute_smc_momentum_bundle(lower)

    higher_smc = higher_bundle.smc
    lower_smc = lower_bundle.smc
    lower_mom = lower_bundle.momentum

    timeline = higher_smc.get("state_timeline")

    if not isinstance(timeline, list):
        raise ValueError("canonical SMC state_timeline missing")

    if len(timeline) != len(higher):
        raise ValueError(
            f"4H state_timeline length {len(timeline)} != bars {len(higher)}"
        )

    bcolors = list(lower_mom.get("bcolor", []))
    vals = list(lower_mom.get("val", []))

    if len(bcolors) != len(lower):
        raise ValueError(
            f"1H momentum bcolor length {len(bcolors)} != bars {len(lower)}"
        )

    if len(vals) != len(lower):
        raise ValueError(
            f"1H momentum val length {len(vals)} != bars {len(lower)}"
        )

    rows: list[dict] = []

    for ev in lower_smc.get("events", []):
        ci = ev.get("confirmed_index")

        if ci is None:
            continue

        ci = int(ci)

        if ci < 0 or ci >= len(lower):
            continue

        # Event becomes knowable only after this 1H bar closes.
        # We use the next observed 1H bar start as the decision time.
        # Do not use the final bar because no next transition is observed.
        if ci + 1 >= len(lower):
            continue

        signal_bar_time = lower.index[ci]
        decision_time = lower.index[ci + 1]

        hi = _latest_available_higher_index(higher.index, decision_time)

        if hi is None:
            continue

        state = timeline[hi]

        if not isinstance(state, dict):
            raise ValueError(f"invalid state_timeline row at {hi}")

        if "swing_bias" not in state:
            raise ValueError("state_timeline row missing swing_bias")

        higher_bias = int(state["swing_bias"] or 0)

        event_type = str(ev.get("type"))
        bullish = bool(ev.get("bullish"))
        internal = bool(ev.get("internal"))

        mom_color = str(bcolors[ci])

        raw_val = vals[ci]
        mom_val = (
            None
            if raw_val is None or not np.isfinite(float(raw_val))
            else float(raw_val)
        )

        side = None

        if (
            higher_bias == 1
            and bullish
            and event_type in {"BOS", "CHoCH"}
            and mom_color in BUY_MOMENTUM_COLORS
        ):
            side = "BUY"

        elif (
            higher_bias == -1
            and not bullish
            and event_type in {"BOS", "CHoCH"}
            and mom_color in SELL_MOMENTUM_COLORS
        ):
            side = "SELL"

        if side is None:
            continue

        structure_level = "Internal" if internal else "Swing"

        higher_bar_time = higher.index[hi]

        reason = [
            ("4H swing trend bullish" if higher_bias == 1 else "4H swing trend bearish"),
            (
                f"1H {structure_level.lower()} "
                f"{'bullish' if bullish else 'bearish'} "
                f"{event_type}"
            ),
            f"1H momentum {mom_color}",
        ]

        rows.append(
            {
                "bar_index": ci,
                "signal_bar_time": signal_bar_time,
                "decision_time": decision_time,
                "side": side,
                "event_type": event_type,
                "structure_level": structure_level,
                "event_level": float(ev["level"]),
                "higher_tf": "4h",
                "higher_bar_index": hi,
                "higher_bar_time": higher_bar_time,
                "higher_swing_bias": higher_bias,
                "lower_tf": "1h",
                "lower_event_bullish": bullish,
                "momentum_val": mom_val,
                "momentum_bcolor": mom_color,
                "reason": reason,
            }
        )

    columns = [
        "bar_index",
        "signal_bar_time",
        "decision_time",
        "side",
        "event_type",
        "structure_level",
        "event_level",
        "higher_tf",
        "higher_bar_index",
        "higher_bar_time",
        "higher_swing_bias",
        "lower_tf",
        "lower_event_bullish",
        "momentum_val",
        "momentum_bcolor",
        "reason",
    ]

    return pd.DataFrame(rows, columns=columns)
