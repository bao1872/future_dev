#!/usr/bin/env python3

"""62维 RL V1 -- 修正后的单笔交易模拟器（V1）。

为什么需要新模拟器
------------------
原权威模拟器（analyze_ob_candidate_v3_phase1.py::simulate）要求
「入场后 12 根 5 分钟 K 线在墙上时钟上连续 60 分钟」。审计证明，
被它删除的 9948 个事件里有 9943 个（99.95%）只是碰到了
正常午休 / 日盘与夜盘之间 / 夜盘结束 / 周末 / 节假日，
真实缺失 K 线为 0。这等价于按「离休市还有多远」筛样本。

本模块把语义改为：
    入场之后未来 12 根真实有效的 5 分钟交易 K 线
正常交易时段间隔直接跳过；只有交易时段内部的缺失才拒绝。

相对权威模拟器的语义差异（仅此三项，其余保持不变）
------------------------------------------------
1. 观察窗口：12 根有效交易 K 线（可跨休市），
   不再是墙上时钟连续的 60 分钟。
2. 有利跳空穿过止盈：按原止盈价计算（R = +target_R），
   不再按开盘价多赚。原模拟器 gap_target 时按开盘价成交。
3. 不利跳空穿过止损：按实际开盘价退出（允许亏损超过 1R）。
   这条与原模拟器一致，未修改。

保持不变的语义
--------------
* 止损固定 1.0 ATR，盈利目标 1.5 / 2.0 / 2.5 R。
* 同一根 K 线同时触发止损与止盈 -> conservative，按止损处理。
* 12 根内未触发 -> 按第 12 根收盘价平仓（timeout）。
* 收益为 GROSS R，无手续费、滑点。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 观察窗口长度：入场后的有效交易 K 线根数。
# 禁止改成 6。
HORIZON_BARS = 12

# 止损固定为 1.0 ATR（与权威数据集 STOP_ATR 一致）。
STOP_ATR = 1.0

TARGET_R = (1.5, 2.0, 2.5)

# 某个断点时刻组合至少出现这么多次，才认定为真实交易时段边界。
STRUCT_MIN_COUNT = 5

# 退出原因代码
EXIT_TIMEOUT = 0
EXIT_GAP_STOP = 1
EXIT_GAP_TARGET = 2
EXIT_STOP = 3
EXIT_TARGET = 4
EXIT_BOTH = 5
EXIT_NOT_EVALUABLE = -1

EXIT_CODE_NAMES = {
    EXIT_TIMEOUT: "TIMEOUT",
    EXIT_GAP_STOP: "GAP_STOP",
    EXIT_GAP_TARGET: "GAP_TARGET",
    EXIT_STOP: "STOP",
    EXIT_TARGET: "TARGET",
    EXIT_BOTH: "BOTH_CONSERVATIVE_STOP",
    EXIT_NOT_EVALUABLE: "NOT_EVALUABLE",
}

# 窗口构建状态
WIN_OK = 0
WIN_INSUFFICIENT = 1
WIN_DATA_ANOMALY = 2


def minute_of_day(times) -> np.ndarray:
    idx = pd.DatetimeIndex(times)
    return idx.hour.to_numpy() * 60 + idx.minute.to_numpy()


def build_session_masks(times, structural=None):
    """返回 (contiguous, normal_gap) 两个长度为 n-1 的布尔数组。

    contiguous[k]  第 k 根与第 k+1 根相差正好 5 分钟。
    normal_gap[k]  第 k 根与第 k+1 根之间的间隔属于该品种
                   反复出现的真实交易时段边界（允许跳过）。
    两者都为 False 时，属于交易时段内部的真实缺失，必须拒绝。

    structural 可直接注入已知的时段边界集合 {(起时刻, 止时刻)}，
    便于用合成数据做定向测试。
    """

    t = pd.to_datetime(pd.Series(times)).to_numpy()
    diff = (t[1:] - t[:-1]) / np.timedelta64(1, "m")
    contiguous = np.isclose(diff, 5.0)

    br = np.where(~contiguous)[0]
    normal_gap = np.zeros(len(contiguous), dtype=bool)

    if len(br):
        f_mod = minute_of_day(t[br])
        to_mod = minute_of_day(t[br + 1])

        if structural is None:
            pairs = pd.DataFrame({"fs": f_mod, "ts": to_mod})
            cnt = pairs.groupby(["fs", "ts"]).size()
            structural = set(
                cnt[cnt >= STRUCT_MIN_COUNT].index
            )

        for i, k in enumerate(br):
            if (f_mod[i], to_mod[i]) in structural:
                normal_gap[k] = True

    return contiguous, normal_gap


def build_valid_windows(
    entry_idx: np.ndarray,
    n_bars: int,
    contiguous: np.ndarray,
    normal_gap: np.ndarray,
    horizon: int = HORIZON_BARS,
):
    """为每笔交易收集入场后的 horizon 根有效交易 K 线。

    返回：
        win   int[n, horizon]，每行的 K 线绝对下标
        stat  int[n]，WIN_OK / WIN_INSUFFICIENT / WIN_DATA_ANOMALY
    """

    n = len(entry_idx)
    win = np.full((n, horizon), -1, dtype=np.int64)
    stat = np.zeros(n, dtype=np.int64)

    for i in range(n):
        cur = int(entry_idx[i])
        if cur < 0 or cur >= n_bars:
            stat[i] = WIN_INSUFFICIENT
            continue
        win[i, 0] = cur
        ok = True
        for j in range(1, horizon):
            nxt = cur + 1
            if nxt >= n_bars:
                stat[i] = WIN_INSUFFICIENT
                ok = False
                break
            if contiguous[cur] or normal_gap[cur]:
                win[i, j] = nxt
                cur = nxt
            else:
                # 交易时段内部缺失 K 线：不得静默跨过。
                stat[i] = WIN_DATA_ANOMALY
                ok = False
                break
        if not ok:
            continue

    return win, stat


def _rot(price: np.ndarray, entry: np.ndarray, atr: np.ndarray,
         direction: np.ndarray) -> np.ndarray:
    """把价格旋转到「交易有利为正」的坐标，单位为 ATR。"""

    return direction[:, None] * (
        price - entry[:, None]
    ) / atr[:, None]


def simulate_actions(
    open_p: np.ndarray,
    high_p: np.ndarray,
    low_p: np.ndarray,
    close_p: np.ndarray,
    entry: np.ndarray,
    atr: np.ndarray,
    direction: np.ndarray,
    target_r: float,
    stop_atr: float = STOP_ATR,
):
    """对一个 (方向, 目标) 组合计算 R 收益。

    入参数组形状均为 [n, horizon]（已按有效 K 线取好），
    entry / atr / direction 形状为 [n]。

    返回：
        R            [n] 实际收益（R 倍数）
        exit_code    [n] 退出原因
        exit_pos     [n] 在第几根有效 K 线上退出（0 起）
    """

    n = open_p.shape[0]

    o = _rot(open_p, entry, atr, direction)
    c = _rot(close_p, entry, atr, direction)

    # 有利极值 / 不利极值（方向已在 _rot 内处理）
    fv = np.maximum(
        _rot(high_p, entry, atr, direction),
        _rot(low_p, entry, atr, direction),
    )
    av = np.minimum(
        _rot(high_p, entry, atr, direction),
        _rot(low_p, entry, atr, direction),
    )

    target_atr = stop_atr * target_r

    gap_stop = o <= -stop_atr
    gap_target = o >= target_atr
    hit_stop = av <= -stop_atr
    hit_target = fv >= target_atr

    # 入场那根 K 线的开盘价就是成交价，不存在跳空。
    gap_stop[:, 0] = False
    gap_target[:, 0] = False

    event = gap_stop | gap_target | hit_stop | hit_target
    any_ev = event.any(axis=1)
    first = np.argmax(event, axis=1)

    idx = first[:, None]

    def take(a):
        return np.take_along_axis(a, idx, axis=1).ravel()

    gs = take(gap_stop)
    gt = take(gap_target) & ~gs
    hs = take(hit_stop)
    ht = take(hit_target)

    is_both = hs & ht & ~gs & ~gt
    is_stop = hs & ~ht & ~gs & ~gt
    is_target = ht & ~hs & ~gs & ~gt
    is_timeout = ~any_ev

    r = np.full(n, np.nan)
    r = np.where(is_timeout, c[:, -1] / stop_atr, r)
    r = np.where(gs, take(o) / stop_atr, r)
    r = np.where(gt, target_r, r)
    r = np.where(is_both, -1.0, r)
    r = np.where(is_stop, -1.0, r)
    r = np.where(is_target, target_r, r)

    code = np.full(n, EXIT_TIMEOUT, dtype=int)
    code = np.where(is_target, EXIT_TARGET, code)
    code = np.where(is_stop, EXIT_STOP, code)
    code = np.where(is_both, EXIT_BOTH, code)
    code = np.where(gt, EXIT_GAP_TARGET, code)
    code = np.where(gs, EXIT_GAP_STOP, code)

    horizon = open_p.shape[1]
    exit_pos = np.where(
        is_timeout, horizon - 1, first
    )

    return r, code, exit_pos
