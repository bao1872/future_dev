#!/usr/bin/env python3

"""62维 RL V1 -- 连续性删除原因审计。

第一阶段发现：合格候选 20013 个，其中 9948 个因为「主视野 h12 前向
12 根 5 分钟 K 线不连续」被权威模拟器判定为不可评估，最终只剩 10065。

本脚本逐项回答：**这 9948 个事件到底为什么被删除。**

判定方法（不使用墙上时钟时间差猜测）
------------------------------------
1. 先用每个品种「全部历史 5 分钟 K 线」提取所有相邻断点；
2. 按 (断点前时刻, 断点后时刻) 聚合。反复出现（>= STRUCT_MIN_COUNT）
   的断点组合 = 该品种真实交易时段边界（午休、日盘→夜盘、夜盘结束等）；
3. 被删除事件的断点若命中真实时段边界 -> 正常市场休市；
4. 断点没有命中任何已知时段边界 -> 真实数据缺失或异常；
5. 节假日/周末用跨自然日天数区分（同一时刻组合既可能是夜盘结束，
   也可能是周末后的开盘）。

产出：
    research/analysis_results/rl_62d_v1/continuity_exclusion_reasons.csv
    research/analysis_results/rl_62d_v1/continuity_by_symbol.csv
    research/analysis_results/rl_62d_v1/continuity_by_hour.csv
    research/analysis_results/rl_62d_v1/continuity_examples.csv
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from research.build_rl_62d_audit_v1 import (
    compute_analyzable_mask,
    load_state_action,
)
from research.rl_62d_core_v1 import PRIMARY_HORIZON

RESULTS = Path("research/analysis_results/rl_62d_v1")

SEED = 20240901

# 某个断点时刻组合至少出现这么多次，才认为它是真实交易时段边界。
STRUCT_MIN_COUNT = 5

# 删除原因（固定词表）
R_LUNCH = "正常午间休市"
R_DAY_END = "日盘正常结束"
R_DAY_NIGHT = "日盘与夜盘之间"
R_NIGHT_END = "夜盘正常结束"
R_HOLIDAY = "节假日或周末"
R_TRUE_MISSING = "真实缺失K线"
R_END_OF_DATA = "数据末尾不足12根"
R_OTHER = "其他"

# 是否属于正常市场休市 / 是否属于真实数据缺失
REASON_META = {
    R_LUNCH: (True, False),
    R_DAY_END: (True, False),
    R_DAY_NIGHT: (True, False),
    R_NIGHT_END: (True, False),
    R_HOLIDAY: (True, False),
    R_TRUE_MISSING: (False, True),
    R_END_OF_DATA: (False, False),
    R_OTHER: (False, False),
}


# ------------------------------------------------------------
# 交易时段边界建模
# ------------------------------------------------------------


def minute_of_day(ts) -> np.ndarray:
    idx = pd.DatetimeIndex(ts)
    return idx.hour.to_numpy() * 60 + idx.minute.to_numpy()


def build_structural_transitions(times: pd.Series):
    """从全部历史 K 线提取真实交易时段边界。

    返回 { (断点前时刻分钟, 断点后时刻分钟): 出现次数 }，
    只保留出现次数 >= STRUCT_MIN_COUNT 的组合。
    """

    t = times.to_numpy()
    d = (t[1:] - t[:-1]) / np.timedelta64(1, "m")
    br = np.where(d != 5.0)[0] + 1

    f = pd.to_datetime(t[br - 1])
    to = pd.to_datetime(t[br])

    df = pd.DataFrame(
        {
            "fs": minute_of_day(f),
            "ts": minute_of_day(to),
        }
    )
    cnt = df.groupby(["fs", "ts"]).size()

    keep = cnt[cnt >= STRUCT_MIN_COUNT]
    return {k: int(v) for k, v in keep.items()}


def label_transition(
    fs_min: int,
    ts_min: int,
    cal_gap_days: int,
) -> str:
    """给一个已确认的时段边界断点打标签。"""

    if cal_gap_days >= 2:
        return R_HOLIDAY

    f_hour = fs_min // 60
    t_hour = ts_min // 60

    from_is_night = bool(f_hour >= 21 or f_hour < 6)
    to_is_night = bool(t_hour >= 21 or t_hour < 6)

    if from_is_night and not to_is_night:
        return R_NIGHT_END
    if (not from_is_night) and to_is_night:
        return R_DAY_NIGHT
    if (not from_is_night) and (not to_is_night):
        if cal_gap_days >= 1:
            return R_DAY_END
        return R_LUNCH
    return R_OTHER


def classify_break(
    t1: pd.Timestamp,
    t2: pd.Timestamp,
    structural: dict,
):
    """把一次 K 线断点分类到固定原因词表。"""

    gap_minutes = (t2 - t1).total_seconds() / 60.0
    key = (t1.hour * 60 + t1.minute, t2.hour * 60 + t2.minute)
    cal_gap_days = (t2.normalize() - t1.normalize()).days

    if key in structural:
        return (
            label_transition(key[0], key[1], cal_gap_days),
            gap_minutes,
        )

    # 断点时刻组合不属于任何已知交易时段边界。
    if cal_gap_days >= 2:
        return R_HOLIDAY, gap_minutes
    return R_TRUE_MISSING, gap_minutes


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    state, action = load_state_action()

    eligible = ~state["touch_close_beyond_far_edge"].astype(bool)
    analyzable = (
        state["candidate_id"]
        .map(compute_analyzable_mask(action))
        .fillna(False)
        .astype(bool)
    )

    excluded = state.loc[eligible & ~analyzable].copy()
    n_excluded_expected = len(excluded)

    rows = []
    boundaries = {}

    for sym, grp in excluded.groupby("symbol"):
        from research.export_ob_trigger_execution_v21 import (
            load_raw_5m,
        )

        raw = (
            load_raw_5m(sym)
            .sort_values("bar_start_time")
            .reset_index(drop=True)
        )
        times = pd.to_datetime(raw["bar_start_time"])
        t_arr = times.to_numpy()
        n_bars = len(times)

        structural = build_structural_transitions(times)
        boundaries[sym] = {
            f"{k[0] // 60:02d}:{k[0] % 60:02d}->"
            f"{k[1] // 60:02d}:{k[1] % 60:02d}": v
            for k, v in sorted(
                structural.items(), key=lambda kv: -kv[1]
            )
        }

        for cid, i, tt in zip(
            grp["candidate_id"].to_numpy(),
            grp["touch_5m_bar_index"].to_numpy(int),
            grp["touch_time"].to_numpy(),
        ):
            if i + 1 >= n_bars or i + PRIMARY_HORIZON >= n_bars:
                rows.append(
                    {
                        "candidate_id": cid,
                        "symbol": sym,
                        "touch_time": tt,
                        "break_step": None,
                        "break_from": None,
                        "break_to": None,
                        "gap_minutes": np.nan,
                        "reason": R_END_OF_DATA,
                    }
                )
                continue

            reason = None
            step = None
            gap = np.nan
            for k in range(1, PRIMARY_HORIZON + 1):
                d = (
                    pd.Timestamp(t_arr[i + k])
                    - pd.Timestamp(t_arr[i + k - 1])
                ).total_seconds() / 60.0
                if abs(d - 5.0) < 1e-9:
                    continue
                step = k
                reason, gap = classify_break(
                    pd.Timestamp(t_arr[i + k - 1]),
                    pd.Timestamp(t_arr[i + k]),
                    structural,
                )
                break

            if reason is None:
                reason = R_OTHER

            rows.append(
                {
                    "candidate_id": cid,
                    "symbol": sym,
                    "touch_time": tt,
                    "break_step": step,
                    "break_from": (
                        str(pd.Timestamp(t_arr[i + step - 1]))
                        if step
                        else None
                    ),
                    "break_to": (
                        str(pd.Timestamp(t_arr[i + step]))
                        if step
                        else None
                    ),
                    "gap_minutes": (
                        float(gap) if np.isfinite(gap) else np.nan
                    ),
                    "reason": reason,
                }
            )

        print(f"{sym}: 时段边界 {boundaries[sym]}")

    detail = pd.DataFrame(rows).reset_index(drop=True)

    if len(detail) != n_excluded_expected:
        raise RuntimeError(
            f"删除事件分类数量不符: {len(detail)} != "
            f"{n_excluded_expected}"
        )

    # ---- 原因汇总 ----
    vc = detail["reason"].value_counts()
    total = len(detail)
    reasons = []
    for reason in (
        R_LUNCH,
        R_DAY_END,
        R_DAY_NIGHT,
        R_NIGHT_END,
        R_HOLIDAY,
        R_TRUE_MISSING,
        R_END_OF_DATA,
        R_OTHER,
    ):
        n = int(vc.get(reason, 0))
        normal, missing = REASON_META[reason]
        reasons.append(
            {
                "原因": reason,
                "事件数": n,
                "占全部删除事件比例": (
                    round(n / total, 6) if total else 0.0
                ),
                "是否属于正常市场休市": normal,
                "是否属于真实数据缺失": missing,
            }
        )
    reasons_df = pd.DataFrame(reasons)
    reasons_df.to_csv(
        RESULTS / "continuity_exclusion_reasons.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ---- 按品种 ----
    elig_by_sym = state.loc[eligible].groupby("symbol").size()
    by_sym = []
    for sym in sorted(excluded["symbol"].unique()):
        d = detail[detail["symbol"] == sym]
        n_el = int(elig_by_sym.get(sym, 0))
        n_ex = len(d)
        n_normal = int(
            d["reason"].map(lambda x: REASON_META[x][0]).sum()
        )
        n_missing = int(
            d["reason"].map(lambda x: REASON_META[x][1]).sum()
        )
        by_sym.append(
            {
                "品种": sym,
                "合格事件数": n_el,
                "被删除事件数": n_ex,
                "删除比例": (
                    round(n_ex / n_el, 6) if n_el else 0.0
                ),
                "正常休市导致": n_normal,
                "真实缺失导致": n_missing,
                "其他原因": n_ex - n_normal - n_missing,
            }
        )
    by_sym_df = pd.DataFrame(by_sym)
    by_sym_df.to_csv(
        RESULTS / "continuity_by_symbol.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ---- 按触碰小时 ----
    elig_all = state.loc[eligible].copy()
    elig_all["hour"] = pd.to_datetime(
        elig_all["touch_time"]
    ).dt.hour
    detail["hour"] = pd.to_datetime(detail["touch_time"]).dt.hour

    by_hour = []
    for h in sorted(elig_all["hour"].unique()):
        n_el = int((elig_all["hour"] == h).sum())
        n_ex = int((detail["hour"] == h).sum())
        sub = detail[detail["hour"] == h]
        main_reason = (
            sub["reason"].value_counts().idxmax()
            if len(sub)
            else ""
        )
        by_hour.append(
            {
                "触碰小时": int(h),
                "合格事件数": n_el,
                "被删除事件数": n_ex,
                "删除比例": (
                    round(n_ex / n_el, 6) if n_el else 0.0
                ),
                "主要原因": main_reason,
            }
        )
    by_hour_df = pd.DataFrame(by_hour)
    by_hour_df.to_csv(
        RESULTS / "continuity_by_hour.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ---- 典型案例 ----
    rng = np.random.default_rng(SEED)
    samples = []
    for reason in sorted(detail["reason"].unique()):
        sub = detail[detail["reason"] == reason]
        k = min(20, len(sub))
        take = sub.iloc[
            rng.choice(len(sub), size=k, replace=False)
        ].copy()
        take["sample_reason"] = reason
        samples.append(take)
    examples = pd.concat(samples, ignore_index=True)
    examples = examples[
        [
            "sample_reason",
            "candidate_id",
            "symbol",
            "touch_time",
            "break_step",
            "break_from",
            "break_to",
            "gap_minutes",
            "reason",
        ]
    ]
    examples.to_csv(
        RESULTS / "continuity_examples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    detail.to_csv(
        RESULTS / "continuity_exclusion_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )

    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        sha = "unknown"

    n_normal = int(
        detail["reason"].map(lambda x: REASON_META[x][0]).sum()
    )
    (RESULTS / "continuity_audit_meta.json").write_text(
        json.dumps(
            {
                "git_head": sha,
                "seed": SEED,
                "struct_min_count": STRUCT_MIN_COUNT,
                "excluded_events": int(total),
                "normal_market_closure_events": n_normal,
                "normal_market_closure_rate": (
                    round(n_normal / total, 6) if total else 0.0
                ),
                "session_boundaries": boundaries,
                "judgement": (
                    "不使用墙上时钟时间差；以品种全部历史K线中反复"
                    "出现的断点时刻组合作为真实交易时段边界"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("RL_62D_CONTINUITY_AUDIT_DONE")
    print("excluded_events", total)
    print(reasons_df.to_string(index=False))


if __name__ == "__main__":
    main()
