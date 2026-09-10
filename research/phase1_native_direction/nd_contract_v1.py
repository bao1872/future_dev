"""Phase 1 (NATIVE DIRECTION) — 共享常量、数据加载与因果工具。

与旧 research/phase1_tradability 完全独立。旧结论（58.88% base rate、
direction-agnostic UP-or-DOWN 标签、12bar primary、103 维冻结模型、
Phase 1 discovery PASS）在本实验中一律视为 INVALID，不得引用。

本实验只回答一句话：

    一个带原生方向的 canonical OB 触发后，按 OB 自己的方向交易，
    这个事件本身是否值得做？

native_direction 来源（从 Source Owner 代码确认，不靠列名猜测）：
    build_ob_candidate_universe_v3.py:1187  -> "bull": (source_ob_bias == 1)
    build_ob_candidate_universe_v3.py:386   -> bias==1 时 far edge = zone_low
                                               （止损在下方 => LONG）
    ob_trigger_snapshot.py:436              -> 突破 pivot_high 的事件 bias=+1
    => native_direction = source_ob_bias，+1 = Bullish/LONG，-1 = Bearish/SHORT
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.export_ob_trigger_execution_v21 import load_raw_5m
from research.phase1_tradability.phase1_contract_v1 import (
    ROLL_GAP_ATR_THRESHOLD, compute_atr5, discontinuity_flags,
)

BULLISH = 1
BEARISH = -1

TARGET_R = 2.5
STOP_R = 1.0
ATR_WINDOW = 5

RESULTS = Path("research/analysis_results/phase1_native_direction_v1")
RESULTS.mkdir(parents=True, exist_ok=True)

STATE16 = Path(
    "research/analysis_results/ob_rl_dataset_v0_16/ob_rl_state_v0.parquet")

# 15 个满足完整 canonical 数据合同的品种（LC 因 quantile train <600 不可构建）
SYMBOLS = ("AG", "AL", "AU", "CF", "CU", "I", "M", "MA", "NI", "P",
           "RB", "RU", "SC", "SN", "TA")
DEV4 = ("AG", "CU", "M", "RB")

FIVE_MIN = np.timedelta64(5, "m")

_BAR_CACHE: dict = {}


def get_bars(sym: str) -> dict:
    if sym in _BAR_CACHE:
        return _BAR_CACHE[sym]
    raw = load_raw_5m(sym).sort_values("bar_start_time").reset_index(drop=True)
    bars = dict(
        open=raw["open"].to_numpy(float),
        high=raw["high"].to_numpy(float),
        low=raw["low"].to_numpy(float),
        close=raw["close"].to_numpy(float),
        time=pd.to_datetime(raw["bar_start_time"]).to_numpy(),
        n=len(raw),
    )
    _BAR_CACHE[sym] = bars
    return bars


def disc_flags(sym: str) -> np.ndarray:
    return discontinuity_flags(sym, threshold=ROLL_GAP_ATR_THRESHOLD)


def load_candidates() -> pd.DataFrame:
    """canonical entered 候选（V3 仅由 OB_ENTERED 事件产生）。

    不套用任何旧实验的过滤；只做本实验必需的字段投影与硬断言。
    """
    s = pd.read_parquet(STATE16)
    s["candidate_id"] = s["candidate_id"].astype(str)
    s["candidate_group_id"] = s["candidate_group_id"].astype(str)
    assert s["candidate_id"].is_unique, "candidate_id 不唯一"

    c = s[[
        "candidate_id", "candidate_group_id", "symbol", "source_tf",
        "trading_day", "touch_5m_bar_index", "touch_time",
        "source_ob_bias",
    ]].copy()
    c = c.rename(columns={"source_ob_bias": "native_direction"})
    c["native_direction"] = c["native_direction"].astype(int)

    assert c["candidate_id"].is_unique
    assert set(c["native_direction"].dropna().unique()).issubset({-1, 1}), \
        "native_direction 只能取 {-1, +1}"
    return c.reset_index(drop=True)
