import numpy as np
import pandas as pd
import pytest

from research.liquidity_oracle_atlas.trade_oracle_dp_m15_return_audit_v1 import (
    build_trade_return_frame,
)


def _make_inputs():
    # 3 trades: FROM_FLAT-LONG, FROM_FLAT-SHORT, REVERSAL-LONG (from SHORT)
    trades = pd.DataFrame(
        {
            "trade_id": ["t1", "t2", "t3"],
            "symbol": ["AG"] * 3,
            "direction": ["LONG", "SHORT", "LONG"],
            "entry_decision_index": [10, 11, 12],
            "entry_fill_time": pd.to_datetime(
                ["2024-01-02 09:30", "2024-01-02 10:00", "2024-01-02 10:30"]
            ),
            "exit_fill_time": pd.to_datetime(
                ["2024-01-02 10:00", "2024-01-02 10:30", "2024-01-02 11:00"]
            ),
            "holding_bars": [2, 2, 2],
            "gross_points": [30.0, -20.0, 15.0],
            "cost_points": [0.0, 0.0, 0.0],
            "net_points": [30.0, -20.0, 15.0],
            "MFE": [40.0, 10.0, 20.0],
            "MAE": [-5.0, -25.0, -3.0],
            "entry_proximity_episode_id": [3, 3, 7],
        }
    )
    actions = pd.DataFrame(
        {
            "decision_bar_index": [10, 11, 12],
            "position_before": [0, 0, -1],  # t3 reversal from SHORT
        }
    )
    features = pd.DataFrame(
        {
            "execution_bar_index": [10, 11, 12],
            "m15_atr": [100.0, 200.0, 50.0],
        }
    )
    return trades, actions, features


def test_atr_normalization():
    trades, actions, features = _make_inputs()
    out = build_trade_return_frame(trades, actions, features)
    # gross_atr = gross / atr
    assert out.loc[0, "gross_atr"] == pytest.approx(30.0 / 100.0)
    assert out.loc[1, "gross_atr"] == pytest.approx(-20.0 / 200.0)
    assert out.loc[2, "gross_atr"] == pytest.approx(15.0 / 50.0)
    # net == gross under zero cost
    assert out.loc[0, "net_atr"] == pytest.approx(out.loc[0, "gross_atr"])
    # MFE/MAE normalized
    assert out.loc[0, "MFE_atr"] == pytest.approx(40.0 / 100.0)
    assert out.loc[1, "MAE_atr"] == pytest.approx(-25.0 / 200.0)
    # dp_proximity_episode_id renamed through
    assert list(out["dp_proximity_episode_id"]) == [3, 3, 7]


def test_entry_type_classification():
    trades, actions, features = _make_inputs()
    out = build_trade_return_frame(trades, actions, features)
    assert out.loc[0, "entry_type"] == "FROM_FLAT"
    assert out.loc[1, "entry_type"] == "FROM_FLAT"
    assert out.loc[2, "entry_type"] == "REVERSAL"


def test_atr_alignment_stop():
    trades, actions, features = _make_inputs()
    # missing entry_decision_index -> NaN atr -> STOP
    bad = trades.copy()
    bad["entry_decision_index"] = [10, 11, 999]  # 999 has no atr row
    with pytest.raises(RuntimeError, match="STOP_TRADE_RETURN_ATR_ALIGNMENT"):
        build_trade_return_frame(bad, actions, features)


def test_bad_atr_stop():
    trades, actions, features = _make_inputs()
    bad_feat = features.copy()
    bad_feat.loc[0, "m15_atr"] = 0.0
    with pytest.raises(RuntimeError, match="STOP_TRADE_RETURN_BAD_ATR"):
        build_trade_return_frame(trades, actions, bad_feat)
