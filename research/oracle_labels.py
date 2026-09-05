from __future__ import annotations

import numpy as np
import pandas as pd


# tie-breaking deliberately prefers Flat
POSITIONS = np.array([0, 1, -1], dtype=int)

FLAT_IDX = 0


def optimal_position_path(
    close,
    *,
    trade_penalty: float,
) -> dict:
    """Hindsight global optimum by dynamic programming.

    This is an Oracle label generator.
    It intentionally uses the full future price path.

    position[t] is the target position at close[t],
    held over interval t -> t+1.

    Objective is cumulative log-return capture
    minus turnover penalty.
    """

    prices = np.asarray(close, dtype=float)

    if prices.ndim != 1:
        raise ValueError("close must be one-dimensional")

    if len(prices) < 2:
        raise ValueError("at least 2 prices are required")

    if not np.all(np.isfinite(prices)):
        raise ValueError("close contains non-finite values")

    if np.any(prices <= 0):
        raise ValueError("close prices must be positive")

    penalty = float(trade_penalty)

    if penalty < 0:
        raise ValueError("trade_penalty must be >= 0")

    n = len(prices)
    m = len(POSITIONS)

    log_prices = np.log(prices)

    score = np.full((n, m), -np.inf, dtype=float)

    prev = np.full((n, m), -1, dtype=int)

    # At close[0], Oracle may choose a position.
    # Start state before bar 0 is flat.
    for j, pos in enumerate(POSITIONS):
        score[0, j] = -penalty * abs(int(pos))
        prev[0, j] = FLAT_IDX

    for t in range(1, n):
        r = log_prices[t] - log_prices[t - 1]

        for j, new_pos in enumerate(POSITIONS):

            best_score = -np.inf
            best_prev = -1

            for i, old_pos in enumerate(POSITIONS):

                candidate = (
                    score[t - 1, i]
                    + int(old_pos) * r
                    - penalty * abs(int(new_pos) - int(old_pos))
                )

                # Strict greater keeps deterministic
                # tie-breaking according to POSITIONS:
                # Flat -> Long -> Short.
                if candidate > best_score:
                    best_score = candidate
                    best_prev = i

            score[t, j] = best_score
            prev[t, j] = best_prev

    # Force end-of-dataset flat:
    state = FLAT_IDX

    path = np.zeros(n, dtype=int)

    for t in range(n - 1, -1, -1):
        path[t] = POSITIONS[state]

        if t == 0:
            break

        state = prev[t, state]

        if state < 0:
            raise RuntimeError("oracle backpointer broken")

    previous = np.concatenate([np.array([0], dtype=int), path[:-1]])

    delta = path - previous

    action = np.full(n, "HOLD", dtype=object)

    action[delta > 0] = "BUY"
    action[delta < 0] = "SELL"

    interval_returns = np.diff(log_prices)

    gross_log_capture = float(np.sum(path[:-1] * interval_returns))

    turnover_units = int(np.sum(np.abs(delta)))

    objective = gross_log_capture - penalty * turnover_units

    action_count = int(np.sum(action != "HOLD"))

    return {
        "position": path,
        "action": action,
        "delta": delta,
        "objective_log_return": float(objective),
        "gross_log_capture": gross_log_capture,
        "turnover_units": turnover_units,
        "action_count": action_count,
        "trade_penalty": penalty,
    }


def oracle_labels(
    bars: pd.DataFrame,
    *,
    trade_penalty: float,
) -> tuple[pd.DataFrame, dict]:
    """Return per-bar Oracle training labels."""

    if "close" not in bars.columns:
        raise ValueError("bars requires close column")

    result = optimal_position_path(
        bars["close"].to_numpy(float),
        trade_penalty=trade_penalty,
    )

    labels = pd.DataFrame(
        {
            "bar_index": np.arange(len(bars), dtype=int),
            "time": bars.index,
            "close": bars["close"].to_numpy(float),
            "oracle_position": result["position"],
            "oracle_action": result["action"],
            "oracle_delta": result["delta"],
        },
        index=bars.index,
    )

    meta = {
        k: v
        for k, v in result.items()
        if k not in {"position", "action", "delta"}
    }

    return labels, meta
