"""Phase 1 — OB Event Tradability：共享常量、数据加载与因果工具。

继承 M2–M8 已确认的结论（见 RESULTS.md）：
  1. 正常午休/夜盘/周末/节假日不是缺K线，未来路径用 next valid bars 跳过。
  2. 真正缺失K线与连续合约异常跳变必须单独处理。
  4. candidate_id 必须显式关联，禁止 positional concat。
  6. 特征只能来自决策时刻已知信息。
  7. 原 62 维大量字段是 action-relative，Phase 1 禁用。
  12. 连续主力 rollover 必须防止污染标签。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.export_ob_trigger_execution_v21 import load_raw_5m
from research.rl_62d_simulator_v1 import build_session_masks
from research.train_rl_62d_v1 import ROLL_GAP_ATR_THRESHOLD

# ---------------- 冻结的 Phase 1 参考尺 ----------------
TARGET_R = 2.5
STOP_R = 1.0
ATR_WINDOW = 5
ATR_MIN_PERIODS = 2

RESULTS = Path("research/analysis_results/phase1_tradability_v1")
RESULTS.mkdir(parents=True, exist_ok=True)

STATE_PARQUET = Path(
    "research/analysis_results/ob_rl_dataset_v0/ob_rl_state_v0.parquet")
EVENT_INDEX = Path("research/analysis_results/rl_62d_v1/event_index_v2.csv")

SYMBOLS = ("AG", "CU", "M", "RB")

_BAR_CACHE: dict = {}


def get_bars(sym: str) -> dict:
    """按品种加载 5m OHLC 并缓存（NumPy 数组，禁止在事件循环里重建）。"""
    if sym in _BAR_CACHE:
        return _BAR_CACHE[sym]
    raw = load_raw_5m(sym).sort_values("bar_start_time").reset_index(drop=True)
    o = raw["open"].to_numpy(float)
    h = raw["high"].to_numpy(float)
    l = raw["low"].to_numpy(float)
    c = raw["close"].to_numpy(float)
    t = pd.to_datetime(raw["bar_start_time"]).to_numpy()
    bars = dict(open=o, high=h, low=l, close=c, time=t, n=len(raw))
    _BAR_CACHE[sym] = bars
    return bars


def compute_atr5(bars: dict) -> np.ndarray:
    """ATR5：真实波幅 5 根滚动均值，只包含当前及之前的数据（因果）。"""
    h, l, c = bars["high"], bars["low"], bars["close"]
    n = len(c)
    prev_c = np.empty(n)
    prev_c[0] = c[0]
    prev_c[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - ATR_WINDOW + 1)
        win = tr[lo:i + 1]
        if len(win) >= ATR_MIN_PERIODS:
            atr[i] = float(np.nanmean(win))
    return atr


def discontinuity_flags(sym: str) -> np.ndarray:
    """discontinuity_before_bar[i] = True 表示进入第 i 根之前存在不可信边界。

    判定（复用既有阈值与语义）：
      - 非 5 分钟连续 且 不属于反复出现的正常交易时段边界；
      - 或 相邻两根之间的价格跳空 |open[i] - close[i-1]| / ATR5 > ROLL_GAP_ATR_THRESHOLD。
    """
    bars = get_bars(sym)
    t = bars["time"]
    contig, normal = build_session_masks(t)
    gap_time = np.zeros(len(t), dtype=bool)
    gap_time[1:] = ~(contig | normal)          # 第 i 根与 i-1 之间异常

    atr = compute_atr5(bars)
    gap_px = np.zeros(len(t), dtype=bool)
    prev_c = np.empty(len(t))
    prev_c[0] = bars["close"][0]
    prev_c[1:] = bars["close"][:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        gap_atr = np.abs(bars["open"] - prev_c) / atr
    gap_px[1:] = np.nan_to_num(gap_atr[1:], nan=0.0) > ROLL_GAP_ATR_THRESHOLD

    return gap_time | gap_px


# ---------------- Phase 1 特征分组（按列名语义预注册）----------------
GROUP_RULES = [
    ("OB属性", ("source_ob_", "quant_", "touch_behavior", "touch_ordinal",
                "is_first_touch", "touch_intrabar_far_edge_breach",
                "touch_close_beyond_far_edge", "touch_reclaimed_by_close",
                "group_")),
    ("趋势", ("swing_bias_", "internal_bias_", "_structure_bias_")),
    ("结构", ("_structure_type_", "_structure_age_", "structure_class_",
              "above_ob_structure_class_", "below_ob_structure_class_")),
    ("动量", ("momentum_direction_", "sqzmom_val_", "sqzmom_delta_")),
    ("DSA", ("dsa_direction_", "dsa_raw_dsa_vwap_dev_pct_",
             "dsa_vwap_dev_pct_")),
    ("风险几何", ("_atr_", "ob_above_atr_", "ob_below_atr_", "_relation_",
                  "source_ob_width_atr5")),
    ("多周期背景", ("_15m", "_1h")),
]

# 硬性排除：动作相关 / 未来 / 标识
HARD_EXCLUDE = {
    "candidate_id", "candidate_group_id", "trading_day", "touch_time",
    "touch_5m_bar_index", "decision_weight",
    "trade_direction", "trade_mode", "target_R",
    # 分组变量：由 Baseline1 单独使用，不作为事件状态特征进入全模型
    "symbol", "source_tf",
}
EXCLUDE_PATTERNS = ("target_fit_", "_zone_low", "_zone_high", "_level_")
# *_rel_* 一律视为动作相对（除非白名单证明安全）
REL_EXCLUDE = "_rel_"


def group_of(col: str) -> str:
    for g, pats in GROUP_RULES:
        for p in pats:
            if p in col:
                return g
    return "其他事件状态"


def is_excluded(col: str) -> tuple[bool, str]:
    if col in HARD_EXCLUDE:
        return True, "标识/动作字段"
    for p in EXCLUDE_PATTERNS:
        if p in col:
            return True, f"匹配排除模式 {p}"
    if REL_EXCLUDE in col:
        return True, "动作相对字段(_rel_)"
    if col.startswith("stop_structure_"):
        return True, "stop_structure 依赖具体 action/direction"
    return False, ""
