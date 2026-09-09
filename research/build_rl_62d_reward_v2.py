#!/usr/bin/env python3

"""62维 RL V1 -- 修正后奖励矩阵 / 样本恢复 / 时间隔离 / 类别编码。

本阶段仍然禁止训练模型。

产出：
    reward_matrix_audit_v2.csv
    event_distribution_by_touch_hour_v2.csv
    sample_recovery_v2.csv
    time_split_audit_v2.json
    feature_encoder_v1.json
    reward_matrix_v2.npy
    event_index_v2.csv
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from research.export_ob_trigger_execution_v21 import load_raw_5m
from research.ob_rl_model_view_v0_spec import (
    CATEGORICAL_FEATURES_V0,
    MODEL_FEATURES_V0,
)
from research.rl_62d_core_v1 import (
    ACTION_NAMES,
    DATASET_SKIP_ACTION,
    N_ACTIONS,
    NO_TRADE_INDEX,
    PRIMARY_HORIZON,
)
from research.rl_62d_simulator_v1 import (
    EXIT_CODE_NAMES,
    EXIT_GAP_STOP,
    EXIT_GAP_TARGET,
    STOP_ATR,
    TARGET_R,
    WIN_DATA_ANOMALY,
    WIN_INSUFFICIENT,
    WIN_OK,
    build_session_masks,
    build_valid_windows,
    simulate_actions,
)

RESULTS = Path("research/analysis_results/rl_62d_v1")

STATE_PARQUET = Path(
    "research/analysis_results/ob_rl_dataset_v0/"
    "ob_rl_state_v0.parquet"
)
ACTION_PARQUET = Path(
    "research/analysis_results/ob_rl_dataset_v0/"
    "ob_rl_action_v0.parquet"
)

# 类别字段：按权威 CATEGORICAL_FEATURES_V0，
# 但 trade_direction 是数值（±1），按数值处理。
CATEGORICAL_COLUMNS = tuple(
    c
    for c in CATEGORICAL_FEATURES_V0
    if c != "trade_direction"
)
CONTINUOUS_COLUMNS = tuple(
    c for c in MODEL_FEATURES_V0 if c not in CATEGORICAL_COLUMNS
)

MISSING_TOKEN = "__MISSING__"

# 第一部分：奖励矩阵
# ------------------------------------------------------------


def derive_atr5(state: pd.DataFrame) -> np.ndarray:
    """从已冻结特征精确反解执行 ATR。

    source_ob_width_atr5 = (zone_high - zone_low) / atr5
    因此 atr5 = (zone_high - zone_low) / source_ob_width_atr5。
    这样可保证与 62 维里所有 ATR 归一化字段同一单位。
    """

    hi = pd.to_numeric(
        state["source_ob_zone_high"], errors="coerce"
    ).to_numpy(float)
    lo = pd.to_numeric(
        state["source_ob_zone_low"], errors="coerce"
    ).to_numpy(float)
    w = pd.to_numeric(
        state["source_ob_width_atr5"], errors="coerce"
    ).to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        atr5 = (hi - lo) / w
    atr5[~np.isfinite(atr5)] = np.nan
    atr5[atr5 <= 0] = np.nan
    return atr5


def build_rewards(universe: pd.DataFrame):
    """为每个事件计算 6 个真实交易动作的历史收益。"""

    n = len(universe)
    rewards = np.full((n, N_ACTIONS), np.nan)
    rewards[:, NO_TRADE_INDEX] = 0.0

    codes = np.full((n, N_ACTIONS), -99, dtype=int)
    obs_end_ns = np.full(n, -1, dtype=np.int64)
    cross_break = np.zeros((n, N_ACTIONS), dtype=bool)

    status = np.full(n, -1, dtype=int)

    bias = pd.to_numeric(
        universe["source_ob_bias"], errors="coerce"
    ).to_numpy(float)
    atr5 = derive_atr5(universe)

    for sym, grp in universe.groupby("symbol"):
        raw = (
            load_raw_5m(sym)
            .sort_values("bar_start_time")
            .reset_index(drop=True)
        )
        times = pd.to_datetime(raw["bar_start_time"])
        n_bars = len(raw)

        op = raw["open"].to_numpy(float)
        hp = raw["high"].to_numpy(float)
        lp = raw["low"].to_numpy(float)
        cp = raw["close"].to_numpy(float)

        contig, normal = build_session_masks(times)

        entry_idx = (
            grp["touch_5m_bar_index"].to_numpy(int) + 1
        )

        win, stat = build_valid_windows(
            entry_idx, n_bars, contig, normal
        )

        gpos = universe.index.get_indexer(grp.index)
        status[gpos] = stat

        ok = stat == WIN_OK
        if not ok.any():
            continue

        sub_pos = gpos[ok]
        w = win[ok]

        o12 = op[w]
        h12 = hp[w]
        l12 = lp[w]
        c12 = cp[w]

        entry = op[w[:, 0]]
        atr = atr5[sub_pos]
        b = bias[sub_pos]

        bar_end = (
            times.to_numpy() + np.timedelta64(5, "m")
        )

        # 窗口内是否跨越过正常休市
        step_gap = np.zeros(w.shape, dtype=bool)
        for j in range(w.shape[1] - 1):
            step_gap[:, j] = ~contig[w[:, j]]

        col = 1
        for mode in ("FOLLOW", "FADE"):
            direction = b if mode == "FOLLOW" else -b
            for target_r in TARGET_R:
                r, code, pos = simulate_actions(
                    o12,
                    h12,
                    l12,
                    c12,
                    entry,
                    atr,
                    direction,
                    target_r,
                    stop_atr=STOP_ATR,
                )
                rewards[sub_pos, col] = r
                codes[sub_pos, col] = code

                last_bar = w[
                    np.arange(len(sub_pos)), pos
                ]
                end_ns = (
                    bar_end[last_bar]
                    .astype("datetime64[ns]")
                    .astype(np.int64)
                )
                obs_end_ns[sub_pos] = np.maximum(
                    obs_end_ns[sub_pos], end_ns
                )

                used = np.zeros(step_gap.shape, dtype=bool)
                for j in range(w.shape[1]):
                    used[:, j] = j <= pos
                cross_break[sub_pos, col] = (
                    step_gap & used
                ).any(axis=1)

                col += 1

    finite = np.isfinite(rewards).all(axis=1)
    evaluable = (status == WIN_OK) & finite

    obs_end = obs_end_ns.astype("datetime64[ns]")
    obs_end[obs_end_ns < 0] = np.datetime64("NaT")

    return (
        rewards,
        codes,
        obs_end,
        cross_break,
        status,
        evaluable,
    )


# ------------------------------------------------------------
# 第二部分：时间切分（按收益观察窗口隔离）
# ------------------------------------------------------------


def build_time_split_v2(universe, obs_end):
    """60/20/20 初始按交易日切分，再按收益观察窗口剔除边界事件。"""

    days = sorted(
        pd.to_datetime(universe["trading_day"]).unique()
    )
    n_days = len(days)
    b1 = int(n_days * 0.60)
    b2 = int(n_days * 0.80)

    td = pd.to_datetime(universe["trading_day"])
    dset = [set(days[:b1]), set(days[b1:b2]), set(days[b2:])]
    mask = [td.isin(dset[k]).to_numpy() for k in range(3)]

    dec = pd.to_datetime(universe["touch_time"]).to_numpy()

    dropped = np.zeros(len(universe), dtype=bool)

    # 训练段的收益观察结束时间必须早于选择段第一笔决策
    for src, dst in ((0, 1), (1, 2)):
        if mask[dst].sum() == 0:
            continue
        dst_start = dec[mask[dst]].min()
        bad = mask[src] & (obs_end >= dst_start)
        dropped |= bad
        mask[src] = mask[src] & ~bad

    no_overlap = True
    for src, dst in ((0, 1), (1, 2)):
        if mask[src].sum() and mask[dst].sum():
            if obs_end[mask[src]].max() >= dec[mask[dst]].min():
                no_overlap = False

    dec_dt = pd.to_datetime(universe["touch_time"])
    span = (
        obs_end - dec_dt.to_numpy()
    ) / np.timedelta64(1, "m")

    cross_day = (
        pd.to_datetime(obs_end).normalize()
        != dec_dt.dt.normalize().to_numpy()
    )
    dec_s = pd.to_datetime(dec_dt)
    obs_s = pd.to_datetime(obs_end)
    cross_weekend = (
        (
            obs_s.dt.normalize() - dec_s.dt.normalize()
        ).dt.days
        >= 2
    )

    def block(m):
        sub = universe.loc[m]
        return {
            "start_date": (
                str(dec_s[m].min().date()) if m.any() else None
            ),
            "end_date": (
                str(dec_s[m].max().date()) if m.any() else None
            ),
            "trading_days": int(
                pd.to_datetime(
                    universe.loc[m, "trading_day"]
                ).nunique()
            )
            if m.any()
            else 0,
            "events": int(m.sum()),
            "by_symbol": {
                str(k): int(v)
                for k, v in sub["symbol"]
                .value_counts()
                .items()
            },
            "max_reward_observation_span_minutes": (
                float(np.nanmax(span[m])) if m.any() else None
            ),
        }

    return (
        {
            "split_basis": (
                "trading_day_60_20_20_then_"
                "reward_observation_window_embargo"
            ),
            "embargo_rule": (
                "前一段任何事件的 reward_observation_end_time "
                "必须早于后一段第一笔事件的 decision_time；"
                "不满足则剔除前一段边界事件"
            ),
            "train": block(mask[0]),
            "selection": block(mask[1]),
            "test": block(mask[2]),
            "boundary_events_dropped": int(dropped.sum()),
            "max_reward_observation_span_minutes": float(
                np.nanmax(span)
            ),
            "cross_calendar_day_events": int(
                np.nansum(cross_day)
            ),
            "cross_weekend_events": int(
                np.nansum(cross_weekend)
            ),
            "no_reward_window_overlap": bool(no_overlap),
            "universe_events": int(len(universe)),
            "covered_events": int(
                mask[0].sum() + mask[1].sum() + mask[2].sum()
            ),
        },
        mask,
        dropped,
    )


# ------------------------------------------------------------
# 第三部分：类别编码（只在训练段拟合）
# ------------------------------------------------------------


def build_encoder(universe, action, train_mask):
    usable_ids = set(
        universe.loc[train_mask, "candidate_id"].to_numpy()
    )
    sub = action[
        action["candidate_id"].isin(usable_ids)
        & (action["action"] != DATASET_SKIP_ACTION)
    ]

    vocab = {}
    for c in CATEGORICAL_COLUMNS:
        vals = (
            sub[c]
            .astype("object")
            .where(sub[c].notna(), MISSING_TOKEN)
            .astype(str)
            .unique()
            .tolist()
        )
        vocab[c] = sorted(vals)

    encoded = sum(len(v) for v in vocab.values()) + len(
        CONTINUOUS_COLUMNS
    )

    return {
        "encoder_version": "rl_62d_feature_encoder_v1",
        "raw_feature_count": 62,
        "encoded_feature_count": int(encoded),
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "continuous_columns": list(CONTINUOUS_COLUMNS),
        "category_vocabularies": vocab,
        "missing_token": MISSING_TOKEN,
        "unknown_policy": (
            "选择段/测试段出现训练段未见的类别时，"
            "全部置入 __UNKNOWN__ 独热位（每列额外 1 位）"
        ),
        "fitted_on": "TRAIN",
        "note": (
            "raw_feature_count 恒为 62，是原始信息维度；"
            "encoded_feature_count 是类别展开后的网络输入维度，"
            "不是新的因子数量"
        ),
    }


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------


def _stats(r):
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return {}
    win = r[r > 0]
    loss = r[r < 0]
    gp = float(win.sum())
    gl = float(-loss.sum())
    return {
        "mean_R": round(float(r.mean()), 6),
        "std_R": round(float(r.std()), 6),
        "win_rate": round(float((r > 0).mean()), 6),
        "avg_win_R": round(float(win.mean()), 6) if len(win) else 0.0,
        "avg_loss_R": round(float(loss.mean()), 6) if len(loss) else 0.0,
        "profit_factor": (
            round(gp / gl, 6) if gl > 0 else float("inf")
        ),
        "max_loss_R": round(float(r.min()), 6),
        "max_win_R": round(float(r.max()), 6),
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    state = pd.read_parquet(STATE_PARQUET)
    action = pd.read_parquet(ACTION_PARQUET)

    eligible = ~state["touch_close_beyond_far_edge"].astype(bool)
    universe = state.loc[eligible].copy()
    universe = universe.sort_values(
        ["trading_day", "symbol", "touch_5m_bar_index"],
        kind="mergesort",
    ).reset_index(drop=True)

    (
        rewards,
        codes,
        obs_end,
        cross_break,
        status,
        evaluable,
    ) = build_rewards(universe)

    # ---- 样本恢复 ----
    rec = []
    n_el = len(universe)
    n_ev = int(evaluable.sum())
    rec.append(
        {
            "项目": "合格事件数(eligible_candidates)",
            "数量": n_el,
        }
    )
    rec.append({"项目": "可评价事件数", "数量": n_ev})
    rec.append(
        {"项目": "不可评价事件数", "数量": n_el - n_ev}
    )
    rec.append(
        {
            "项目": "其中 数据末尾不足12根有效K线",
            "数量": int((status == WIN_INSUFFICIENT).sum()),
        }
    )
    rec.append(
        {
            "项目": "其中 交易时段内部缺失K线(拒绝)",
            "数量": int((status == WIN_DATA_ANOMALY).sum()),
        }
    )
    rec.append(
        {
            "项目": "其中 收益非有限值",
            "数量": int(
                ((status == WIN_OK) & ~evaluable).sum()
            ),
        }
    )
    for sym, n in (
        universe.loc[evaluable]["symbol"].value_counts().items()
    ):
        rec.append(
            {"项目": f"可评价事件数_{sym}", "数量": int(n)}
        )
    pd.DataFrame(rec).to_csv(
        RESULTS / "sample_recovery_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ---- 奖励矩阵审计 ----
    rows = [
        {"check": "eligible_candidates", "value": n_el},
        {"check": "evaluable_events", "value": n_ev},
        {
            "check": "not_evaluable_events",
            "value": n_el - n_ev,
        },
        {
            "check": "no_trade_all_zero",
            "value": bool(
                np.all(rewards[:, NO_TRADE_INDEX] == 0.0)
            ),
        },
        {
            "check": "all_finite_on_evaluable",
            "value": bool(
                np.isfinite(rewards[evaluable]).all()
            ),
        },
        {
            "check": "action_columns",
            "value": int(rewards.shape[1]),
        },
        {"check": "stop_atr", "value": float(STOP_ATR)},
        {
            "check": "target_R",
            "value": ",".join(str(t) for t in TARGET_R),
        },
        {
            "check": "horizon_valid_bars",
            "value": 12,
        },
        {
            "check": "same_bar_policy",
            "value": "conservative(同K线双向触发按止损)",
        },
    ]
    for i, name in enumerate(ACTION_NAMES):
        s = _stats(rewards[evaluable, i])
        rows.append(
            {
                "check": f"action_{name}",
                "value": (
                    "mean_R={mean_R},std_R={std_R},"
                    "win_rate={win_rate},avg_win_R={avg_win_R},"
                    "avg_loss_R={avg_loss_R},"
                    "profit_factor={profit_factor},"
                    "max_loss_R={max_loss_R},"
                    "max_win_R={max_win_R}".format(**s)
                )
                if s
                else "NO_DATA",
            }
        )

    ev = evaluable
    rows.append(
        {
            "check": "跨休市交易数量(事件x动作)",
            "value": int(cross_break[ev, 1:].sum()),
        }
    )
    rows.append(
        {
            "check": "跨休市交易比例",
            "value": round(
                float(cross_break[ev, 1:].mean()), 6
            ),
        }
    )
    rows.append(
        {
            "check": "发生不利跳空止损数量",
            "value": int((codes[ev, 1:] == EXIT_GAP_STOP).sum()),
        }
    )
    rows.append(
        {
            "check": "发生有利跳空止盈数量",
            "value": int(
                (codes[ev, 1:] == EXIT_GAP_TARGET).sum()
            ),
        }
    )
    for k, v in sorted(EXIT_CODE_NAMES.items()):
        rows.append(
            {
                "check": f"exit_code_{v}",
                "value": int((codes[ev, 1:] == k).sum()),
            }
        )
    pd.DataFrame(rows).to_csv(
        RESULTS / "reward_matrix_audit_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ---- 时段分布 ----
    universe["hour"] = pd.to_datetime(
        universe["touch_time"]
    ).dt.hour
    dist = []
    for scope, sub_mask in [("ALL", np.ones(len(universe), bool))] + [
        (s, (universe["symbol"] == s).to_numpy())
        for s in sorted(universe["symbol"].unique())
    ]:
        idx = np.where(sub_mask)[0]
        for h in sorted(universe.loc[idx, "hour"].unique()):
            hi = idx[universe.loc[idx, "hour"].to_numpy() == h]
            n_ok = int(evaluable[hi].sum())
            dist.append(
                {
                    "scope": scope,
                    "触碰小时": int(h),
                    "合格事件数": len(hi),
                    "可评价事件数": n_ok,
                    "删除数": len(hi) - n_ok,
                    "删除比例": round(
                        (len(hi) - n_ok) / len(hi), 6
                    ),
                }
            )
    pd.DataFrame(dist).to_csv(
        RESULTS / "event_distribution_by_touch_hour_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ---- 时间切分 ----
    ev_universe = universe.loc[evaluable].reset_index(drop=True)

    split, masks, dropped = build_time_split_v2(
        universe, obs_end
    )
    (RESULTS / "time_split_audit_v2.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ---- 编码器（只在训练段拟合）----
    train_mask = np.zeros(len(universe), dtype=bool)
    for k in range(3):
        if k == 0:
            train_mask = masks[0].copy()
    train_mask &= evaluable
    enc = build_encoder(universe, action, train_mask)
    (RESULTS / "feature_encoder_v1.json").write_text(
        json.dumps(enc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ---- 保存矩阵 ----
    np.save(
        RESULTS / "reward_matrix_v2.npy",
        rewards[evaluable],
    )
    ev_universe.assign(
        reward_observation_end_time=pd.to_datetime(
            obs_end[evaluable]
        )
    )[
        [
            "candidate_id",
            "symbol",
            "trading_day",
            "touch_time",
            "reward_observation_end_time",
        ]
    ].to_csv(
        RESULTS / "event_index_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("RL_62D_REWARD_V2_DONE")
    print("eligible", n_el, "evaluable", n_ev)
    print(
        "insufficient",
        int((status == WIN_INSUFFICIENT).sum()),
        "data_anomaly",
        int((status == WIN_DATA_ANOMALY).sum()),
    )
    print(
        "cross_break_rate",
        round(float(cross_break[ev, 1:].mean()), 6),
    )
    print(
        "gap_stop",
        int((codes[ev, 1:] == EXIT_GAP_STOP).sum()),
        "gap_target",
        int((codes[ev, 1:] == EXIT_GAP_TARGET).sum()),
    )
    print(
        "encoded_feature_count",
        enc["encoded_feature_count"],
        "raw",
        enc["raw_feature_count"],
    )
    print("no_overlap", split["no_reward_window_overlap"])


if __name__ == "__main__":
    main()
