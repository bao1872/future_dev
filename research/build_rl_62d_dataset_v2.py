#!/usr/bin/env python3

"""62维 RL V1 -- V2 数据层：奖励矩阵重建 + 语义预检 + 时间隔离 + 编码器。

产出目录 research/analysis_results/rl_62d_v1/
    reward_matrix_v2.npy
    event_index_v2.csv
    reward_matrix_audit_v2.csv
    sample_recovery_v2.csv
    event_distribution_by_touch_hour_v2.csv
    time_split_audit_v2.json
    feature_encoder_v1.json
    semantic_checks_v2.json
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
)
from research.rl_62d_simulator_v1 import (
    EXIT_CODE_NAMES,
    EXIT_GAP_STOP,
    EXIT_GAP_TARGET,
    HORIZON_BARS,
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

CATEGORICAL_COLUMNS = tuple(
    c for c in CATEGORICAL_FEATURES_V0 if c != "trade_direction"
)
CONTINUOUS_COLUMNS = tuple(
    c for c in MODEL_FEATURES_V0 if c not in CATEGORICAL_COLUMNS
)

MISSING_TOKEN = "__MISSING__"
UNKNOWN_TOKEN = "__UNKNOWN__"


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def derive_atr5(state: pd.DataFrame) -> np.ndarray:
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
        a = (hi - lo) / w
    a[~np.isfinite(a)] = np.nan
    a[a <= 0] = np.nan
    return a


def load_universe():
    state = pd.read_parquet(STATE_PARQUET)
    action = pd.read_parquet(ACTION_PARQUET)
    eligible = ~state["touch_close_beyond_far_edge"].astype(bool)
    universe = (
        state.loc[eligible]
        .sort_values(
            ["trading_day", "symbol", "touch_5m_bar_index"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    return state, action, universe


def build_rewards(universe: pd.DataFrame):
    n = len(universe)
    rewards = np.full((n, N_ACTIONS), np.nan)
    rewards[:, NO_TRADE_INDEX] = 0.0
    codes = np.full((n, N_ACTIONS), -99, dtype=int)
    obs_end_ns = np.full(n, -1, dtype=np.int64)
    cross_break = np.zeros((n, N_ACTIONS), dtype=bool)
    status = np.full(n, -1, dtype=int)
    max_gap_atr = np.zeros((n, N_ACTIONS), dtype=float)

    bias = pd.to_numeric(
        universe["source_ob_bias"], errors="coerce"
    ).to_numpy(float)
    atr5 = derive_atr5(universe)

    raw_cache = {}

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
        raw_cache[sym] = (times, op, hp, lp, cp, contig, n_bars)

        entry_idx = grp["touch_5m_bar_index"].to_numpy(int) + 1
        win, stat = build_valid_windows(
            entry_idx, n_bars, contig, normal
        )
        gpos = universe.index.get_indexer(grp.index)
        status[gpos] = stat

        ok = stat == WIN_OK
        if not ok.any():
            continue

        sub = gpos[ok]
        w = win[ok]
        o12, h12, l12, c12 = op[w], hp[w], lp[w], cp[w]
        entry = op[w[:, 0]]
        atr = atr5[sub]
        b = bias[sub]
        bar_end = times.to_numpy() + np.timedelta64(5, "m")

        step_gap = np.zeros(w.shape, dtype=bool)
        for j in range(w.shape[1] - 1):
            step_gap[:, j] = ~contig[w[:, j]]

        # 跨休市跳空幅度（以 ATR 计），用于合约换月排查
        gap_atr = np.zeros(w.shape, dtype=float)
        for j in range(1, w.shape[1]):
            g = np.abs(op[w[:, j]] - cp[w[:, j - 1]]) / atr
            gap_atr[:, j] = g

        col = 1
        for mode in ("FOLLOW", "FADE"):
            direction = b if mode == "FOLLOW" else -b
            for target_r in TARGET_R:
                r, code, pos = simulate_actions(
                    o12, h12, l12, c12, entry, atr,
                    direction, target_r, stop_atr=STOP_ATR,
                )
                rewards[sub, col] = r
                codes[sub, col] = code
                last_bar = w[np.arange(len(sub)), pos]
                end_ns = (
                    bar_end[last_bar]
                    .astype("datetime64[ns]")
                    .astype(np.int64)
                )
                obs_end_ns[sub] = np.maximum(
                    obs_end_ns[sub], end_ns
                )
                used = np.zeros(step_gap.shape, dtype=bool)
                for j in range(w.shape[1]):
                    used[:, j] = j <= pos
                cross_break[sub, col] = (step_gap & used).any(axis=1)
                max_gap_atr[sub, col] = np.where(
                    used.any(axis=1),
                    (gap_atr * used).max(axis=1),
                    0.0,
                )
                col += 1

    finite = np.isfinite(rewards).all(axis=1)
    evaluable = (status == WIN_OK) & finite
    obs_end = obs_end_ns.astype("datetime64[ns]")
    obs_end[obs_end_ns < 0] = np.datetime64("NaT")

    return (
        rewards, codes, obs_end, cross_break, max_gap_atr,
        status, evaluable, raw_cache,
    )


def validate_against_authoritative(
    universe, action, raw_cache, atr5
):
    """用旧语义（墙上时钟连续）复算，与权威 primary_reward_R 对照。

    这同时验证：入场价 = 触碰下一根 K 线开盘价、ATR 单位、
    方向约定、止损/止盈几何，以及决策与入场的时序关系。
    """

    piv = action[action["action"] != DATASET_SKIP_ACTION].pivot(
        index="candidate_id",
        columns="action",
        values="primary_reward_R",
    )

    rows = []
    tot = 0
    same = 0
    diff_gt = 0

    for sym, grp in universe.groupby("symbol"):
        times, op, hp, lp, cp, contig, n_bars = raw_cache[sym]
        entry_idx = grp["touch_5m_bar_index"].to_numpy(int) + 1
        a = atr5[universe.index.get_indexer(grp.index)]
        b = pd.to_numeric(
            grp["source_ob_bias"], errors="coerce"
        ).to_numpy(float)

        # 旧语义：入场起 12 根必须墙上时钟连续
        win = np.full((len(grp), HORIZON_BARS), -1, dtype=np.int64)
        okmask = np.zeros(len(grp), dtype=bool)
        for i, e in enumerate(entry_idx):
            good = True
            for j in range(HORIZON_BARS):
                k = e + j
                if k >= n_bars:
                    good = False
                    break
                if j > 0 and not contig[k - 1]:
                    good = False
                    break
                win[i, j] = k
            okmask[i] = good

        if not okmask.any():
            continue
        w = win[okmask]
        ids = grp["candidate_id"].to_numpy()[okmask]
        sub_b = b[okmask]
        sub_a = a[okmask]

        o12, h12, l12, c12 = op[w], hp[w], lp[w], cp[w]
        entry = op[w[:, 0]]

        col = 1
        for mode in ("FOLLOW", "FADE"):
            direction = sub_b if mode == "FOLLOW" else -sub_b
            for target_r in TARGET_R:
                r, code, _ = simulate_actions(
                    o12, h12, l12, c12, entry, sub_a,
                    direction, target_r, stop_atr=STOP_ATR,
                )
                name = f"{mode}_{target_r}R"
                old = piv.reindex(ids)[name].to_numpy(float)
                m = np.isfinite(old) & np.isfinite(r)
                d = np.abs(old[m] - r[m])
                tot += int(m.sum())
                same += int((d < 1e-6).sum())
                diff_gt += int(
                    ((d >= 1e-6) & (code[m] == EXIT_GAP_TARGET)).sum()
                )
                col += 1
        rows.append(sym)

    return {
        "compared_pairs": int(tot),
        "exact_match": int(same),
        "exact_match_rate": (
            round(same / tot, 6) if tot else None
        ),
        "differences_all_on_favorable_gap": bool(
            tot - same == diff_gt
        ),
        "difference_count": int(tot - same),
        "difference_on_gap_target_count": int(diff_gt),
        "symbols": rows,
    }


def semantic_checks(universe, action, obs_end, max_gap_atr,
                    evaluable):
    """四项高风险基础口径检查。"""

    # 1) 夜盘交易日归属
    tt = pd.to_datetime(universe["touch_time"])
    td = pd.to_datetime(universe["trading_day"])
    delta = (td - tt.dt.normalize()).dt.days.to_numpy()
    hour = tt.dt.hour.to_numpy()
    night_mask = hour >= 21
    day_mask = hour < 21

    # 2) 合约换月：跨休市跳空幅度分布
    g = max_gap_atr[evaluable, 1:]
    g = g[np.isfinite(g)]

    out = {
        "touch_vs_decision": {
            "decision_time_definition": (
                "触碰 K 线收盘后决策；入场价为下一根有效 K 线开盘价"
            ),
            "entry_bar_offset_from_touch": 1,
            "evidence": (
                "validate_against_authoritative 用旧连续语义复算，"
                "与权威 primary_reward_R 逐笔对照"
            ),
        },
        "trading_day_semantics": {
            "night_session_hour_ge_21_trading_day_"
            "minus_calendar_day_mode": (
                int(pd.Series(delta[night_mask]).mode().iloc[0])
                if night_mask.any()
                else None
            ),
            "day_session_hour_lt_21_trading_day_"
            "minus_calendar_day_mode": (
                int(pd.Series(delta[day_mask]).mode().iloc[0])
                if day_mask.any()
                else None
            ),
            "night_events": int(night_mask.sum()),
            "day_events": int(day_mask.sum()),
            "conclusion": (
                "21:00 及之后的夜盘归属下一自然日的交易日，"
                "符合交易所交易日语义"
                if night_mask.any()
                and pd.Series(delta[night_mask]).mode().iloc[0] == 1
                else "需人工复核"
            ),
        },
        "rollover_risk": {
            "continuous_series": True,
            "data_source": "PyTDX *L8 主力连续",
            "gap_atr_mean": round(float(np.mean(g)), 6) if len(g) else None,
            "gap_atr_p50": round(float(np.percentile(g, 50)), 6) if len(g) else None,
            "gap_atr_p95": round(float(np.percentile(g, 95)), 6) if len(g) else None,
            "gap_atr_p99": round(float(np.percentile(g, 99)), 6) if len(g) else None,
            "gap_atr_max": round(float(np.max(g)), 6) if len(g) else None,
            "events_gap_gt_3atr": int((g > 3.0).sum()) if len(g) else 0,
            "events_gap_gt_5atr": int((g > 5.0).sum()) if len(g) else 0,
            "events_gap_gt_10atr": int((g > 10.0).sum()) if len(g) else 0,
            "events_gap_gt_20atr": int((g > 20.0).sum()) if len(g) else 0,
            "events_gap_gt_30atr": int((g > 30.0).sum()) if len(g) else 0,
            "evaluable_action_pairs": int(len(g)),
        },
        "price_executability": {
            "entry_price": "下一根有效 K 线开盘价",
            "stop_price": "entry - direction x 1.0 ATR",
            "target_price": "entry + direction x target_R x 1.0 ATR",
            "timeout_settlement": f"第 {HORIZON_BARS} 根有效 K 线收盘价",
            "gap_through_stop": "按实际开盘价退出（允许亏损超过 1R）",
            "gap_through_target": "按原止盈价结算（不额外赚取跳空）",
            "same_bar_both": "conservative，按止损处理",
            "uses_future_confirmed_price": False,
        },
    }
    return out


def build_time_split_v2(universe, obs_end, evaluable):
    days = sorted(
        pd.to_datetime(universe["trading_day"]).unique()
    )
    n = len(days)
    b1, b2 = int(n * 0.60), int(n * 0.80)
    td = pd.to_datetime(universe["trading_day"])
    sets = [set(days[:b1]), set(days[b1:b2]), set(days[b2:])]
    mask = [td.isin(sets[k]).to_numpy() for k in range(3)]
    for k in range(3):
        mask[k] &= evaluable

    dec = pd.to_datetime(universe["touch_time"]).to_numpy()
    dropped = np.zeros(len(universe), dtype=bool)
    for src, dst in ((0, 1), (1, 2)):
        if not mask[dst].any():
            continue
        dst_start = dec[mask[dst]].min()
        bad = mask[src] & (obs_end >= dst_start)
        dropped |= bad
        mask[src] = mask[src] & ~bad

    no_overlap = True
    for src, dst in ((0, 1), (1, 2)):
        if mask[src].any() and mask[dst].any():
            if obs_end[mask[src]].max() >= dec[mask[dst]].min():
                no_overlap = False

    dec_s = pd.to_datetime(universe["touch_time"])
    obs_s = pd.Series(pd.to_datetime(obs_end))
    ev = evaluable & np.isfinite(
        (obs_s - dec_s).dt.total_seconds().to_numpy()
    )
    span = (
        (obs_s - dec_s).dt.total_seconds().to_numpy() / 60.0
    )
    cross_day = (
        obs_s.dt.normalize() != dec_s.dt.normalize()
    ).to_numpy()
    cross_weekend = (
        (obs_s.dt.normalize() - dec_s.dt.normalize()).dt.days >= 2
    ).to_numpy()

    def block(m):
        sub = universe.loc[m]
        return {
            "start_date": (
                str(dec_s[m].min().date()) if m.any() else None
            ),
            "end_date": (
                str(dec_s[m].max().date()) if m.any() else None
            ),
            "events": int(m.sum()),
            "by_symbol": {
                str(k): int(v)
                for k, v in sub["symbol"].value_counts().items()
            },
            "max_reward_observation_span_minutes": (
                round(float(np.nanmax(span[m])), 3)
                if m.any()
                else None
            ),
        }

    return (
        {
            "split_basis": "trading_day_60_20_20_then_observation_window_embargo",
            "embargo_rule": (
                "前一段任何事件的 reward_observation_end_time "
                "必须早于后一段第一笔事件的 decision_time"
            ),
            "train": block(mask[0]),
            "selection": block(mask[1]),
            "test": block(mask[2]),
            "boundary_events_dropped": int(dropped.sum()),
            "max_reward_observation_span_minutes": round(
                float(np.nanmax(span[ev])), 3
            ),
            "cross_calendar_day_events": int(
                (cross_day & evaluable).sum()
            ),
            "cross_weekend_events": int(
                (cross_weekend & evaluable).sum()
            ),
            "no_reward_window_overlap": bool(no_overlap),
        },
        mask,
    )


def build_action_feature_rows(universe, action):
    """[事件数 x 6, 62] 的原始特征行（未编码）。"""

    order = universe["candidate_id"].to_numpy()
    sub = action[
        action["candidate_id"].isin(order)
        & (action["action"] != DATASET_SKIP_ACTION)
    ]
    ev = pd.DataFrame(
        {"candidate_id": order, "_row": np.arange(len(order))}
    )
    sub = (
        ev.merge(sub, on="candidate_id", how="left")
        .sort_values(["_row", "action"], kind="mergesort")
        .reset_index(drop=True)
    )
    assert np.array_equal(
        sub["candidate_id"].to_numpy(), np.repeat(order, 6)
    )
    return sub[list(MODEL_FEATURES_V0)].reset_index(drop=True)


def build_encoder(rows, train_row_mask):
    tr = rows.loc[train_row_mask]

    vocab = {}
    for c in CATEGORICAL_COLUMNS:
        vals = (
            tr[c]
            .astype("object")
            .where(tr[c].notna(), MISSING_TOKEN)
            .astype(str)
            .unique()
            .tolist()
        )
        vocab[c] = sorted(vals)

    cont = list(CONTINUOUS_COLUMNS)
    fill, mean, std = {}, {}, {}
    miss_cols = []
    for c in cont:
        v = pd.to_numeric(tr[c], errors="coerce").to_numpy(float)
        miss = ~np.isfinite(v)
        med = float(np.nanmedian(v)) if (~miss).any() else 0.0
        fill[c] = med if np.isfinite(med) else 0.0
        filled = np.where(miss, fill[c], v)
        m = float(filled.mean())
        s = float(filled.std())
        mean[c] = m
        std[c] = s if s > 0 else 1.0
        if miss.any():
            miss_cols.append(c)

    encoded = (
        sum(len(v) + 1 for v in vocab.values())
        + len(cont)
        + len(miss_cols)
    )

    return {
        "encoder_version": "rl_62d_feature_encoder_v1",
        "raw_feature_count": 62,
        "encoded_feature_count": int(encoded),
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "continuous_columns": cont,
        "missing_indicator_columns": miss_cols,
        "category_vocabularies": vocab,
        "continuous_fill_values": fill,
        "continuous_mean": mean,
        "continuous_std": std,
        "missing_token": MISSING_TOKEN,
        "unknown_token": UNKNOWN_TOKEN,
        "fitted_on": "TRAIN",
        "note": (
            "raw_feature_count 恒为 62，是原始信息维度；"
            "encoded_feature_count 是类别独热 + 缺失标记展开后的"
            "网络输入维度，不是新的因子数量"
        ),
    }


def transform(rows, enc):
    blocks = []
    for c in enc["categorical_columns"]:
        v = (
            rows[c]
            .astype("object")
            .where(rows[c].notna(), MISSING_TOKEN)
            .astype(str)
            .to_numpy()
        )
        vocab = enc["category_vocabularies"][c]
        idx = {s: i for i, s in enumerate(vocab)}
        arr = np.zeros((len(v), len(vocab) + 1), dtype=np.float32)
        for i, s in enumerate(v):
            arr[i, idx.get(s, len(vocab))] = 1.0
        blocks.append(arr)

    cont = enc["continuous_columns"]
    num = np.zeros((len(rows), len(cont)), dtype=np.float32)
    miss = np.zeros(
        (len(rows), len(cont)), dtype=np.float32
    )
    for j, c in enumerate(cont):
        v = pd.to_numeric(rows[c], errors="coerce").to_numpy(float)
        m = ~np.isfinite(v)
        miss[:, j] = m.astype(np.float32)
        v = np.where(m, enc["continuous_fill_values"][c], v)
        num[:, j] = (
            (v - enc["continuous_mean"][c])
            / enc["continuous_std"][c]
        )
    blocks.append(num)

    keep = [
        cont.index(c) for c in enc["missing_indicator_columns"]
    ]
    if keep:
        blocks.append(miss[:, keep])

    return np.hstack(blocks).astype(np.float32)


def _stats(r):
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return "NO_DATA"
    win, loss = r[r > 0], r[r < 0]
    gp, gl = float(win.sum()), float(-loss.sum())
    return (
        f"mean_R={r.mean():.6f},std_R={r.std():.6f},"
        f"win_rate={(r > 0).mean():.6f},"
        f"avg_win_R={win.mean() if len(win) else 0:.6f},"
        f"avg_loss_R={loss.mean() if len(loss) else 0:.6f},"
        f"profit_factor={(gp / gl if gl > 0 else float('inf')):.6f},"
        f"max_loss_R={r.min():.6f},max_win_R={r.max():.6f}"
    )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    state, action, universe = load_universe()
    atr5 = derive_atr5(universe)

    (
        rewards, codes, obs_end, cross_break, max_gap_atr,
        status, evaluable, raw_cache,
    ) = build_rewards(universe)

    val = validate_against_authoritative(
        universe, action, raw_cache, atr5
    )
    sem = semantic_checks(
        universe, action, obs_end, max_gap_atr, evaluable
    )
    sem["authoritative_replication"] = val
    sem["git_head"] = git_head()

    split, masks = build_time_split_v2(
        universe, obs_end, evaluable
    )

    uni_ev = universe.loc[evaluable].reset_index(drop=True)
    rows = build_action_feature_rows(uni_ev, action)
    train_event = masks[0][evaluable]
    train_row = np.repeat(train_event, 6)
    enc = build_encoder(rows, train_row)

    X = transform(rows, enc)
    np.save(RESULTS / "features_v2.npy", X)

    # 输出
    ev = evaluable
    n_el, n_ev = len(universe), int(ev.sum())
    rec = [
        {"项目": "合格事件数", "数量": n_el},
        {"项目": "可评价事件数", "数量": n_ev},
        {"项目": "不可评价事件数", "数量": n_el - n_ev},
        {
            "项目": "其中 数据末尾不足12根有效K线",
            "数量": int((status == WIN_INSUFFICIENT).sum()),
        },
        {
            "项目": "其中 交易时段内部缺失K线(拒绝)",
            "数量": int((status == WIN_DATA_ANOMALY).sum()),
        },
        {
            "项目": "其中 收益非有限值",
            "数量": int(((status == WIN_OK) & ~ev).sum()),
        },
    ]
    for s, n in universe.loc[ev, "symbol"].value_counts().items():
        rec.append({"项目": f"可评价事件数_{s}", "数量": int(n)})
    pd.DataFrame(rec).to_csv(
        RESULTS / "sample_recovery_v2.csv", index=False,
        encoding="utf-8-sig",
    )

    audit = [
        {"check": "eligible_candidates", "value": n_el},
        {"check": "evaluable_events", "value": n_ev},
        {"check": "not_evaluable_events", "value": n_el - n_ev},
        {
            "check": "no_trade_all_zero",
            "value": bool(np.all(rewards[:, NO_TRADE_INDEX] == 0.0)),
        },
        {
            "check": "all_finite_on_evaluable",
            "value": bool(np.isfinite(rewards[ev]).all()),
        },
        {"check": "action_columns", "value": int(rewards.shape[1])},
        {"check": "stop_atr", "value": float(STOP_ATR)},
        {"check": "horizon_valid_bars", "value": HORIZON_BARS},
        {
            "check": "跨休市交易数量(事件x动作)",
            "value": int(cross_break[ev, 1:].sum()),
        },
        {
            "check": "跨休市交易比例",
            "value": round(float(cross_break[ev, 1:].mean()), 6),
        },
        {
            "check": "发生不利跳空止损数量",
            "value": int((codes[ev, 1:] == EXIT_GAP_STOP).sum()),
        },
        {
            "check": "发生有利跳空止盈数量",
            "value": int((codes[ev, 1:] == EXIT_GAP_TARGET).sum()),
        },
    ]
    for i, name in enumerate(ACTION_NAMES):
        audit.append(
            {"check": f"action_{name}", "value": _stats(rewards[ev, i])}
        )
    for k, v in sorted(EXIT_CODE_NAMES.items()):
        audit.append(
            {
                "check": f"exit_code_{v}",
                "value": int((codes[ev, 1:] == k).sum()),
            }
        )
    pd.DataFrame(audit).to_csv(
        RESULTS / "reward_matrix_audit_v2.csv", index=False,
        encoding="utf-8-sig",
    )

    universe["hour"] = pd.to_datetime(
        universe["touch_time"]
    ).dt.hour
    dist = []
    scopes = [("ALL", np.ones(len(universe), bool))] + [
        (s, (universe["symbol"] == s).to_numpy())
        for s in sorted(universe["symbol"].unique())
    ]
    for scope, m in scopes:
        idx = np.where(m)[0]
        for h in sorted(universe.loc[idx, "hour"].unique()):
            hi = idx[universe.loc[idx, "hour"].to_numpy() == h]
            n_ok = int(ev[hi].sum())
            dist.append(
                {
                    "scope": scope,
                    "触碰小时": int(h),
                    "合格事件数": len(hi),
                    "可评价事件数": n_ok,
                    "删除数": len(hi) - n_ok,
                    "删除比例": round((len(hi) - n_ok) / len(hi), 6),
                }
            )
    pd.DataFrame(dist).to_csv(
        RESULTS / "event_distribution_by_touch_hour_v2.csv",
        index=False, encoding="utf-8-sig",
    )

    (RESULTS / "time_split_audit_v2.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULTS / "feature_encoder_v1.json").write_text(
        json.dumps(enc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULTS / "semantic_checks_v2.json").write_text(
        json.dumps(sem, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    np.save(RESULTS / "reward_matrix_v2.npy", rewards[ev])
    np.save(
        RESULTS / "max_gap_atr_v2.npy", max_gap_atr[ev]
    )

    idx = universe.loc[ev].reset_index(drop=True).copy()
    idx["reward_observation_end_time"] = pd.to_datetime(
        obs_end[ev]
    )
    idx["split"] = "DROPPED"
    for k, nm in enumerate(("TRAIN", "SELECTION", "TEST")):
        pos = np.where(masks[k])[0]
        sel = idx.index[
            idx["candidate_id"].isin(
                universe.loc[pos, "candidate_id"]
            )
        ]
        idx.loc[sel, "split"] = nm
    idx[
        [
            "candidate_id", "symbol", "trading_day", "touch_time",
            "reward_observation_end_time", "split",
        ]
    ].to_csv(
        RESULTS / "event_index_v2.csv", index=False,
        encoding="utf-8-sig",
    )

    print("RL_62D_DATASET_V2_DONE")
    print("eligible", n_el, "evaluable", n_ev)
    print(
        "insufficient", int((status == WIN_INSUFFICIENT).sum()),
        "data_anomaly", int((status == WIN_DATA_ANOMALY).sum()),
    )
    print("auth_replication", val)
    print("rollover", sem["rollover_risk"])
    print("encoded_dim", enc["encoded_feature_count"])
    print("split", {k: split[k]["events"] for k in ("train", "selection", "test")})
    print("no_overlap", split["no_reward_window_overlap"])


if __name__ == "__main__":
    main()
