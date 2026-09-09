#!/usr/bin/env python3

"""修正后模拟器的定向测试：跨休市与跳空执行规则。

覆盖：
 1. 午休后正常继续
 2. 日盘到夜盘正常继续
 3. 夜盘到次日日盘正常继续
 4. 周五到周一正常继续
 5. 不利跳空穿过止损
 6. 有利跳空穿过止盈
 7. 开盘未越界但盘中触发止损
 8. 开盘未越界但盘中触发止盈
 9. 同一根K线同时触发止损与止盈
10. 真实交易时段内部缺失K线仍然拒绝
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.rl_62d_simulator_v1 import (
    EXIT_BOTH,
    EXIT_GAP_STOP,
    EXIT_GAP_TARGET,
    EXIT_STOP,
    EXIT_TARGET,
    EXIT_TIMEOUT,
    WIN_DATA_ANOMALY,
    WIN_INSUFFICIENT,
    WIN_OK,
    build_session_masks,
    build_valid_windows,
    simulate_actions,
)

# 已知真实交易时段边界（时刻 -> 时刻，单位：当日分钟）
STRUCTURAL = {
    (10 * 60 + 10, 10 * 60 + 30),   # 上午小节休息
    (11 * 60 + 25, 13 * 60 + 30),   # 午休
    (14 * 60 + 55, 21 * 60 + 0),    # 日盘 -> 夜盘
    (2 * 60 + 25, 9 * 60 + 0),      # 夜盘结束 -> 次日日盘
    (14 * 60 + 55, 9 * 60 + 0),     # 周五收盘 -> 周一开盘
}


def _run(
    times,
    o,
    h,
    l,
    c,
    entry_i: int,
    direction: int,
    target_r: float,
    atr: float = 1.0,
):
    ts = pd.to_datetime(pd.Series(times))
    contig, normal = build_session_masks(
        ts, structural=STRUCTURAL
    )
    win, stat = build_valid_windows(
        np.array([entry_i]),
        len(ts),
        contig,
        normal,
    )
    if stat[0] != WIN_OK:
        return None, int(stat[0])

    idx = win[0]
    op = np.asarray(o)[idx][None, :]
    hp = np.asarray(h)[idx][None, :]
    lp = np.asarray(l)[idx][None, :]
    cp = np.asarray(c)[idx][None, :]

    entry = np.array([float(np.asarray(o)[entry_i])])
    r, code, pos = simulate_actions(
        op,
        hp,
        lp,
        cp,
        entry,
        np.array([atr]),
        np.array([float(direction)]),
        target_r,
    )
    return (
        float(r[0]),
        int(code[0]),
        int(pos[0]),
        int(idx[-1]),
    ), int(stat[0])


def _flat(n: int, price: float = 100.0):
    o = np.full(n, price)
    return o, o.copy(), o.copy(), o.copy()


def _rng(day: str, start_hm: str, n: int, step: int = 5):
    base = pd.Timestamp(f"{day} {start_hm}")
    return [
        base + pd.Timedelta(minutes=step * i) for i in range(n)
    ]


# ------------------------------------------------------------


def test_lunch_break_continues():
    # 10:00 10:05 10:10 |午休| 10:30 ...
    times = _rng("2026-03-02", "10:00", 3) + _rng(
        "2026-03-02", "10:30", 10
    )
    o, h, l, c = _flat(len(times))
    (r, code, pos, last), st = _run(
        times, o, h, l, c, 1, 1, 2.0
    )
    assert st == WIN_OK
    assert code == EXIT_TIMEOUT
    assert abs(r) < 1e-12


def test_day_to_night_continues():
    # 14:45 14:50 14:55 |日盘->夜盘| 21:00 ...
    times = _rng("2026-03-02", "14:45", 3) + _rng(
        "2026-03-02", "21:00", 10
    )
    o, h, l, c = _flat(len(times))
    (r, code, pos, last), st = _run(
        times, o, h, l, c, 1, 1, 2.0
    )
    assert st == WIN_OK
    assert code == EXIT_TIMEOUT


def test_night_to_next_day_continues():
    # 02:15 02:20 02:25 |夜盘结束| 09:00 ...
    times = _rng("2026-03-02", "02:15", 3) + _rng(
        "2026-03-02", "09:00", 10
    )
    o, h, l, c = _flat(len(times))
    (r, code, pos, last), st = _run(
        times, o, h, l, c, 1, 1, 2.0
    )
    assert st == WIN_OK
    assert code == EXIT_TIMEOUT


def test_friday_to_monday_continues():
    # 周五 14:45 14:50 14:55 |周末| 周一 09:00 ...
    times = _rng("2026-03-06", "14:45", 3) + _rng(
        "2026-03-09", "09:00", 10
    )
    o, h, l, c = _flat(len(times))
    (r, code, pos, last), st = _run(
        times, o, h, l, c, 1, 1, 2.0
    )
    assert st == WIN_OK
    assert code == EXIT_TIMEOUT


def test_adverse_gap_through_stop():
    # 多头：成交 100，止损 99；第二根直接开在 97 -> 亏损 3R
    times = _rng("2026-03-02", "10:00", 12)
    o = np.full(12, 100.0)
    o[1] = 97.0
    h = np.full(12, 100.0)
    h[1] = 97.5
    l = np.full(12, 100.0)
    l[1] = 96.5
    c = np.full(12, 100.0)
    c[1] = 97.0
    (r, code, pos, last), st = _run(
        times, o, h, l, c, 0, 1, 2.0
    )
    assert st == WIN_OK
    assert code == EXIT_GAP_STOP
    assert abs(r - (-3.0)) < 1e-9, f"不利跳空应按开盘价，得到 {r}"
    assert r < -1.0, "真实亏损必须允许超过 1R"


def test_favorable_gap_through_target():
    # 多头：成交 100，2R 目标 102；第二根开在 105 -> 仍按 102 计算
    times = _rng("2026-03-02", "10:00", 12)
    o = np.full(12, 100.0)
    o[1] = 105.0
    h = np.full(12, 100.0)
    h[1] = 105.5
    l = np.full(12, 100.0)
    l[1] = 104.5
    c = np.full(12, 100.0)
    c[1] = 105.0
    (r, code, pos, last), st = _run(
        times, o, h, l, c, 0, 1, 2.0
    )
    assert st == WIN_OK
    assert code == EXIT_GAP_TARGET
    assert abs(r - 2.0) < 1e-9, f"有利跳空应封顶在目标，得到 {r}"


def test_intrabar_stop_without_gap():
    times = _rng("2026-03-02", "10:00", 12)
    o = np.full(12, 100.0)
    h = np.full(12, 100.5)
    l = np.full(12, 100.0)
    l[3] = 98.5
    c = np.full(12, 100.0)
    (r, code, pos, last), st = _run(
        times, o, h, l, c, 0, 1, 2.0
    )
    assert st == WIN_OK
    assert code == EXIT_STOP
    assert abs(r - (-1.0)) < 1e-9


def test_intrabar_target_without_gap():
    times = _rng("2026-03-02", "10:00", 12)
    o = np.full(12, 100.0)
    h = np.full(12, 100.0)
    h[2] = 102.5
    l = np.full(12, 99.5)
    c = np.full(12, 100.0)
    (r, code, pos, last), st = _run(
        times, o, h, l, c, 0, 1, 2.0
    )
    assert st == WIN_OK
    assert code == EXIT_TARGET
    assert abs(r - 2.0) < 1e-9


def test_same_bar_stop_and_target():
    times = _rng("2026-03-02", "10:00", 12)
    o = np.full(12, 100.0)
    h = np.full(12, 100.0)
    h[4] = 103.0
    l = np.full(12, 100.0)
    l[4] = 98.0
    c = np.full(12, 100.0)
    (r, code, pos, last), st = _run(
        times, o, h, l, c, 0, 1, 2.0
    )
    assert st == WIN_OK
    assert code == EXIT_BOTH
    assert abs(r - (-1.0)) < 1e-9, "同K线双向触发必须按止损"


def test_in_session_missing_bar_rejected():
    # 11:00 之后缺 11:05，且 11:00->11:10 不是已知时段边界
    # 10:50 10:55 11:00 |缺 11:05| 11:10 ...
    times = _rng("2026-03-02", "10:50", 3) + _rng(
        "2026-03-02", "11:10", 10
    )
    o, h, l, c = _flat(len(times))
    out, st = _run(times, o, h, l, c, 1, 1, 2.0)
    assert st == WIN_DATA_ANOMALY
    assert out is None


def test_insufficient_at_end_of_data():
    times = _rng("2026-03-02", "10:00", 5)
    o, h, l, c = _flat(len(times))
    out, st = _run(times, o, h, l, c, 0, 1, 2.0)
    assert st == WIN_INSUFFICIENT


def test_short_direction_symmetric():
    # 空头：成交 100，止损 101；第二根开在 103 -> 亏损 3R
    times = _rng("2026-03-02", "10:00", 12)
    o = np.full(12, 100.0)
    o[1] = 103.0
    h = np.full(12, 100.0)
    h[1] = 103.5
    l = np.full(12, 100.0)
    l[1] = 102.5
    c = np.full(12, 100.0)
    c[1] = 103.0
    (r, code, pos, last), st = _run(
        times, o, h, l, c, 0, -1, 2.0
    )
    assert st == WIN_OK
    assert code == EXIT_GAP_STOP
    assert abs(r - (-3.0)) < 1e-9


def _run_all():
    fns = [
        test_lunch_break_continues,
        test_day_to_night_continues,
        test_night_to_next_day_continues,
        test_friday_to_monday_continues,
        test_adverse_gap_through_stop,
        test_favorable_gap_through_target,
        test_intrabar_stop_without_gap,
        test_intrabar_target_without_gap,
        test_same_bar_stop_and_target,
        test_in_session_missing_bar_rejected,
        test_insufficient_at_end_of_data,
        test_short_direction_symmetric,
    ]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("RL_62D_SIMULATOR_V1_TESTS_PASS")


if __name__ == "__main__":
    _run_all()
