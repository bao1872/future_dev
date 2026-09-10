"""OB Event Value Test — 共享常量、加载、session 坐标、事前状态、未来结果。

NO MODEL / NO ML / NO feature selection / NO parameter tuning。

因果边界（本实验最重要的纪律）：
    所有匹配变量与归一化尺 R0 只能使用 <= t0-1 的信息。
    t0 是触发 bar，OB_ENTERED 已在 t0 发生，因此 t0 的
    open/high/low/close/volume 一律不得进入匹配。
    R0 = ATR5[t0-1]（事件前最后一根完整 bar 时已知的 ATR）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.export_ob_trigger_execution_v21 import load_raw_5m
from research.phase1_tradability.phase1_contract_v1 import (
    ROLL_GAP_ATR_THRESHOLD, compute_atr5, discontinuity_flags,
)
from research.rl_62d_simulator_v1 import minute_of_day

RESULTS = Path("research/analysis_results/ob_event_value_v1")
RESULTS.mkdir(parents=True, exist_ok=True)

V3_DIR = Path("research/exports/ob_candidate_universe_v3")
STATE16 = Path(
    "research/analysis_results/ob_rl_dataset_v0_16/ob_rl_state_v0.parquet")

SYMBOLS = ("AG", "AL", "AU", "CF", "CU", "I", "M", "MA", "NI", "P",
           "RB", "RU", "SC", "SN", "TA")

HORIZONS = (1, 3, 6, 12, 24)
ANCHOR_H = 12
K_CONTROLS = 3
DAY_WINDOW = 120
MAX_H = max(HORIZONS)

MATCH_COLS = [
    "atr_rel_pre",
    "pre_ret_3_R",
    "pre_ret_12_R",
    "pre_rv_12_R",
    "pre_range_12_R",
    "volume_z_20_pre",
]

_BARS: dict = {}


def get_bars(sym: str) -> dict:
    if sym in _BARS:
        return _BARS[sym]
    raw = load_raw_5m(sym).sort_values("bar_start_time").reset_index(drop=True)
    b = dict(
        open=raw["open"].to_numpy(float),
        high=raw["high"].to_numpy(float),
        low=raw["low"].to_numpy(float),
        close=raw["close"].to_numpy(float),
        volume=raw["volume"].to_numpy(float)
        if "volume" in raw.columns else raw["trade"].to_numpy(float),
        time=pd.to_datetime(raw["bar_start_time"]).to_numpy(),
        trading_day=raw["trading_day"].astype(str).to_numpy()
        if "trading_day" in raw.columns else None,
        n=len(raw),
    )
    b["atr5"] = compute_atr5(b)
    b["disc"] = discontinuity_flags(sym, threshold=ROLL_GAP_ATR_THRESHOLD)
    b.update(session_coords(b["time"]))
    _BARS[sym] = b
    return b


def session_coords(t: np.ndarray) -> dict:
    """session 坐标（复用既有 minute_of_day 与 5 分钟断口分段）。

    session_type            = 该段首根的 minute_of_day（区分日盘/夜盘/午盘）
    minute_from_session_open= 距段首的分钟数
    time_bucket_30m         = minute_from_session_open // 30
    """
    ts = pd.to_datetime(pd.Series(t))
    brk = ts.diff() > pd.Timedelta(minutes=5)
    seg = brk.cumsum().to_numpy()          # session_segment id
    mod = minute_of_day(t)                 # 当日分钟数

    seg_start_mod = np.empty(len(t), dtype=np.int64)
    seg_start_idx = np.empty(len(t), dtype=np.int64)
    # 每段首根索引
    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    cur = 0
    for i in range(len(t)):
        while cur + 1 < len(starts) and starts[cur + 1] <= i:
            cur += 1
        seg_start_idx[i] = starts[cur]
        seg_start_mod[i] = mod[starts[cur]]

    dt = ts.to_numpy()
    mfs = (dt - dt[seg_start_idx]) / np.timedelta64(1, "m")
    return dict(
        session_segment=seg,
        session_type=seg_start_mod,
        minute_from_session_open=mfs.astype(np.int64),
        time_bucket_30m=(mfs.astype(np.int64) // 30),
    )


def pre_event_state(bars: dict) -> dict:
    """全部匹配变量，严格截止 t0-1（向量化）。"""
    c = pd.Series(bars["close"])
    h = pd.Series(bars["high"])
    lo = pd.Series(bars["low"])
    v = pd.Series(bars["volume"])
    atr = pd.Series(bars["atr5"])

    r0 = atr.shift(1)                       # R0 = ATR5[t0-1]
    dc = c.diff()

    out = {}
    out["atr_rel_pre"] = (r0 / c.shift(1)).to_numpy(float)
    out["pre_ret_3_R"] = ((c.shift(1) - c.shift(4)) / r0).to_numpy(float)
    out["pre_ret_12_R"] = ((c.shift(1) - c.shift(13)) / r0).to_numpy(float)
    out["pre_rv_12_R"] = (np.sqrt(
        (dc ** 2).rolling(12).sum()) / r0).to_numpy(float)
    out["pre_range_12_R"] = (
        (h.rolling(12).max() - lo.rolling(12).min()) / r0).to_numpy(float)

    vm = v.rolling(20).mean()
    vs = v.rolling(20).std(ddof=0).clip(lower=1e-12)
    out["volume_z_20_pre"] = ((v.shift(1) - vm.shift(0)) / vs).to_numpy(float)
    out["R0"] = r0.to_numpy(float)
    return out


def forward_outcomes(bars: dict, t0: int, r0: float,
                     horizon: int) -> dict | None:
    """outcome 只使用 t0+1 .. t0+horizon（向量化版本见 build_outcome_cols）。"""
    if not np.isfinite(r0) or r0 <= 0:
        return None
    p0 = float(bars["close"][t0])
    a, b = t0 + 1, t0 + 1 + horizon
    if b > bars["n"]:
        return None
    hh = bars["high"][a:b]
    ll = bars["low"][a:b]
    cc = bars["close"][a:b]
    if len(hh) != horizon:
        return None
    max_high = float(hh.max())
    min_low = float(ll.min())
    up = (max_high - p0) / r0
    dn = (p0 - min_low) / r0
    mde = max(up, dn)
    pc = np.concatenate([[p0], cc])
    rv = float(np.sqrt(np.sum((np.diff(pc) / r0) ** 2)))
    return dict(
        up_exc_R=up, down_exc_R=dn,
        max_directional_excursion_R=mde,
        forward_range_R=(max_high - min_low) / r0,
        abs_close_move_R=abs(float(cc[-1]) - p0) / r0,
        realized_vol_R=rv,
        hit_abs_1p5R=int(mde >= 1.5),
        hit_abs_2p5R=int(mde >= 2.5),
    )


OUTCOME_KEYS = ("up_exc_R", "down_exc_R",
                "max_directional_excursion_R", "forward_range_R",
                "abs_close_move_R", "realized_vol_R",
                "hit_abs_1p5R", "hit_abs_2p5R")


def build_outcome_cols(bars: dict) -> dict:
    """对每个 horizon 预计算全部 bar 的 outcome（向量化）。"""
    c = bars["close"].astype(float)
    h = bars["high"].astype(float)
    lo = bars["low"].astype(float)
    r0 = np.r_[np.nan, bars["atr5"][:-1]]        # R0 = ATR5[t0-1]
    n = bars["n"]
    res = {hz: {} for hz in HORIZONS}

    with np.errstate(divide="ignore", invalid="ignore"):
        for hz in HORIZONS:
            up = np.full(n, np.nan)
            dn = np.full(n, np.nan)
            rng = np.full(n, np.nan)
            amv = np.full(n, np.nan)
            rv = np.full(n, np.nan)
            for i in range(n):
                a, b = i + 1, i + 1 + hz
                if b > n or not np.isfinite(r0[i]) or r0[i] <= 0:
                    continue
                hh = h[a:b]
                ll = lo[a:b]
                cc = c[a:b]
                p0 = c[i]
                up[i] = (hh.max() - p0) / r0[i]
                dn[i] = (p0 - ll.min()) / r0[i]
                rng[i] = (hh.max() - ll.min()) / r0[i]
                amv[i] = abs(cc[-1] - p0) / r0[i]
                rv[i] = np.sqrt(np.sum(
                    (np.diff(np.r_[p0, cc]) / r0[i]) ** 2))
            mde = np.fmax(up, dn)
            res[hz] = dict(
                up_exc_R=up, down_exc_R=dn,
                max_directional_excursion_R=mde,
                forward_range_R=rng, abs_close_move_R=amv,
                realized_vol_R=rv,
                hit_abs_1p5R=(mde >= 1.5).astype(float),
                hit_abs_2p5R=(mde >= 2.5).astype(float),
            )
    return res


def future_clean(bars: dict, t0: int, horizon: int) -> bool:
    """未来 horizon 根内不得跨越不可信边界。"""
    a, b = t0 + 1, t0 + 1 + horizon
    if b > bars["n"]:
        return False
    return not bool(bars["disc"][a:b].any())
