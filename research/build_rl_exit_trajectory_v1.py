"""M4-PRE: 有限期限动态退出强化学习的轨迹数据（只建数据，不训练）。

问题定义
--------
一次交易 = 一个 episode：

    episode = 事件 + 一个初始交易动作

入场后最多 12 根有效 5 分钟 K 线（含入场那根），
在每根 K 线「开盘时」决策 HOLD / EXIT，第 12 根强制结束。

严格控制（与权威模拟器 rl_62d_simulator_v1 完全一致）
----------------------------------------------------
* entry_idx = touch_5m_bar_index + 1，成交价 = 该根开盘价。
* 窗口 win[:, 0] = 入场那根，共 12 根（因此入场后还有 11 根）。
* R 坐标：direction * (price - entry) / atr，止损 -1.0(=1R)，止盈 +target_R。
* 优先级：开盘跳空止损 > 开盘跳空止盈 > 同时触发(保守止损) > 止损 > 止盈 > timeout。
* 有利跳空穿止盈按原止盈价；不利跳空按实际开盘价。

因果约束
--------
* 决策时点 = 第 j 根开盘。动态市场状态只取「上一根已完成 K 线」(j-1)。
* 状态严禁使用当前 K 线的 high / low / close。
* EXIT 价值严格 = 当前开盘价换算的 R。

输出
----
research/analysis_results/rl_exit_v1/
    rl_exit_trajectory_v1.parquet
    rl_exit_audit.json

用法：python -m research.build_rl_exit_trajectory_v1
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import research.m2_nondeep_temporal_v1 as m2
from research.export_ob_trigger_execution_v21 import load_raw_5m
from research.ob_rl_model_view_v0_spec import MODEL_FEATURES_V0
from research.rl_62d_core_v1 import ACTION_NAMES
from research.rl_62d_simulator_v1 import (
    EXIT_CODE_NAMES,
    EXIT_GAP_STOP,
    EXIT_GAP_TARGET,
    HORIZON_BARS,
    STOP_ATR,
    TARGET_R,
    WIN_OK,
    build_session_masks,
    build_valid_windows,
)
from research.train_rl_62d_v1 import ROLL_GAP_ATR_THRESHOLD

OUT_DIR = Path("research/analysis_results/rl_exit_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRAJ = OUT_DIR / "rl_exit_trajectory_v1.parquet"

PANEL_PATH = Path(
    "research/analysis_results/m2/temporal_market_panel.parquet"
)
STATE_PARQUET = Path(
    "research/analysis_results/ob_rl_dataset_v0/ob_rl_state_v0.parquet"
)
IDX_CSV = Path("research/analysis_results/rl_62d_v1/event_index_v2.csv")
GAPS_NPY = Path("research/analysis_results/rl_62d_v1/max_gap_atr_v2.npy")
REWARD_NPY = Path("research/analysis_results/rl_62d_v1/reward_matrix_v2.npy")

# 动态市场状态：逐根更新（决策开盘前最近一根已完成 K 线）
DYN_RAW = [
    "sqzmom_val", "sqzmom_delta", "dsa_raw_dsa_vwap_dev_pct",
    "vol20", "vol_part",
]
DYN_TEMP = []
for _c in DYN_RAW:
    DYN_TEMP += [f"{_c}__delta_1", f"{_c}__slope_6"]
DYN_STATE = [
    "momentum_direction_code",
    "momentum_direction__changed_last_3",
    "momentum_direction__bars_since_change",
    "dsa_direction",
    "dsa_direction__changed_last_3",
    "dsa_direction__bars_since_change",
]
DYN_COLS = DYN_RAW + DYN_TEMP + DYN_STATE

POS_COLS = [
    "current_unrealized_R", "distance_to_stop_R", "distance_to_target_R",
    "bars_held", "bars_remaining", "trade_direction", "target_R",
    "max_favorable_excursion_R_so_far", "max_adverse_excursion_R_so_far",
    "previous_step_price_change_R", "minutes_since_previous_valid_bar",
    "crossed_session_break",
]


def load_universe_keep():
    """返回与 M2 完全一致的 kept 事件集合。"""
    idx = pd.read_csv(IDX_CSV)
    idx["candidate_id"] = idx["candidate_id"].astype(str)
    gaps = np.load(GAPS_NPY)
    keep = (
        np.nan_to_num(gaps[:, 1:], nan=0.0).max(axis=1)
        <= ROLL_GAP_ATR_THRESHOLD
    )
    order = idx["candidate_id"].to_numpy()
    return order, keep


def main():
    # -------------------------------------------------------------- #
    # 1. 事件集合（与 M2 一致）
    # -------------------------------------------------------------- #
    D = m2.load_base()
    n_kept = D["n_kept"]
    keep_ids = D["keep_ids"]
    ev_sym = D["ev_sym"]
    ev_day = D["ev_day"]
    ev_bar = D["ev_bar"]
    snap3 = D["snap_rows"].reshape(n_kept, 6, len(MODEL_FEATURES_V0))
    cat_set = set(D["cat_cols"])
    print(f"[M4] kept events={n_kept}, cat_features={len(cat_set)}", flush=True)

    # ATR / bias（与 build_rl_62d_reward_v2.derive_atr5 同一公式）
    state = pd.read_parquet(STATE_PARQUET)
    state["candidate_id"] = state["candidate_id"].astype(str)
    state = state.set_index("candidate_id").loc[keep_ids]
    hi = pd.to_numeric(state["source_ob_zone_high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(state["source_ob_zone_low"], errors="coerce").to_numpy(float)
    w = pd.to_numeric(state["source_ob_width_atr5"], errors="coerce").to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        atr = (hi - lo) / w
    atr[~np.isfinite(atr)] = np.nan
    atr[atr <= 0] = np.nan
    bias = pd.to_numeric(state["source_ob_bias"], errors="coerce").to_numpy(float)
    print(f"[M4] atr valid={int(np.isfinite(atr).sum())}/{n_kept}, "
          f"bias valid={int(np.isfinite(bias).sum())}/{n_kept}", flush=True)

    # 面板（每品种加 momentum_direction_code，与 M2 口径一致）
    panel_all = pd.read_parquet(PANEL_PATH)
    panels = {}
    for s, g in panel_all.groupby("symbol"):
        gg = g.set_index("bar_index").copy()
        gg["momentum_direction_code"] = pd.factorize(gg["momentum_direction"])[0]
        panels[s] = gg

    # -------------------------------------------------------------- #
    # 2. 逐品种构建轨迹
    # -------------------------------------------------------------- #
    frames = []
    # all-HOLD 终值（用于与权威奖励矩阵一致性校验）
    hold_R = np.full((n_kept, 6), np.nan)
    epi_stats = []

    for sym in sorted(pd.unique(ev_sym)):
        pos = np.where(ev_sym == sym)[0]
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
        entry_idx = ev_bar[pos] + 1
        win, stat = build_valid_windows(entry_idx, n_bars, contig, normal)

        ok = stat == WIN_OK
        if not ok.any():
            continue
        sub_pos = pos[ok]
        w = win[ok]
        n = len(sub_pos)
        print(f"[M4] {sym}: events={len(pos)} win_ok={n} "
              f"dropped={int((~ok).sum())}", flush=True)

        o12 = op[w]
        h12 = hp[w]
        l12 = lp[w]
        c12 = cp[w]
        entry = op[w[:, 0]]
        atr_s = atr[sub_pos]
        bias_s = bias[sub_pos]

        # 上一根已完成 K 线（j=0 时为入场前一根）
        prev_idx = np.empty_like(w)
        prev_idx[:, 0] = w[:, 0] - 1
        prev_idx[:, 1:] = w[:, :-1]

        t_dec = times.to_numpy()[w]                     # 决策K线开始时间
        t_prev = times.to_numpy()[np.clip(prev_idx, 0, None)]
        minutes = (t_dec - t_prev) / np.timedelta64(1, "m")
        valid_prev = prev_idx >= 0
        crossed = np.where(
            valid_prev,
            ~contig[np.clip(prev_idx, 0, n_bars - 2)],
            False,
        )

        # 动态市场状态（逐根，仅取上一根已完成 K 线）
        pan = panels[sym]
        dyn_vals = {}
        for c in DYN_COLS:
            dyn_vals[c] = pan[c].to_numpy(float)[prev_idx]

        for ai in range(6):
            mode = "FOLLOW" if ai < 3 else "FADE"
            target_r = TARGET_R[ai % 3]
            dirv = bias_s if mode == "FOLLOW" else -bias_s

            d = dirv[:, None]
            o = d * (o12 - entry[:, None]) / atr_s[:, None]
            ch = d * (h12 - entry[:, None]) / atr_s[:, None]
            cl = d * (l12 - entry[:, None]) / atr_s[:, None]
            cc = d * (c12 - entry[:, None]) / atr_s[:, None]
            fv = np.maximum(ch, cl)      # 有利极值
            av = np.minimum(ch, cl)      # 不利极值

            gap_stop = o <= -STOP_ATR
            gap_target = o >= target_r
            hit_stop = av <= -STOP_ATR
            hit_target = fv >= target_r
            gap_stop[:, 0] = False
            gap_target[:, 0] = False

            event = gap_stop | gap_target | hit_stop | hit_target
            any_ev = event.any(axis=1)
            first = np.argmax(event, axis=1)
            idx1 = first[:, None]

            def take(a):
                return np.take_along_axis(a, idx1, axis=1).ravel()

            gs = take(gap_stop)
            gt = take(gap_target) & ~gs
            hs = take(hit_stop)
            ht = take(hit_target)
            is_both = hs & ht & ~gs & ~gt
            is_stop = hs & ~ht & ~gs & ~gt
            is_target = ht & ~hs & ~gs & ~gt
            is_timeout = ~any_ev

            r = np.full(n, np.nan)
            r = np.where(is_timeout, cc[:, -1] / STOP_ATR, r)
            r = np.where(gs, take(o) / STOP_ATR, r)
            r = np.where(gt, target_r, r)
            r = np.where(is_both, -1.0, r)
            r = np.where(is_stop, -1.0, r)
            r = np.where(is_target, target_r, r)

            code = np.full(n, 0, dtype=int)
            code = np.where(is_target, 4, code)
            code = np.where(is_stop, 3, code)
            code = np.where(is_both, 5, code)
            code = np.where(gt, EXIT_GAP_TARGET, code)
            code = np.where(gs, EXIT_GAP_STOP, code)

            exit_pos = np.where(is_timeout, HORIZON_BARS - 1, first)
            hold_R[sub_pos, ai] = r

            # ---------- 持仓状态 ----------
            cumax = np.maximum.accumulate(fv, axis=1)
            cumin = np.minimum.accumulate(av, axis=1)
            mfe = np.empty_like(o)
            mae = np.empty_like(o)
            mfe[:, 0] = o[:, 0]
            mae[:, 0] = o[:, 0]
            mfe[:, 1:] = np.maximum(cumax[:, :-1], o[:, 1:])
            mae[:, 1:] = np.minimum(cumin[:, :-1], o[:, 1:])
            prev_chg = np.empty_like(o)
            prev_chg[:, 0] = 0.0
            prev_chg[:, 1:] = o[:, 1:] - o[:, :-1]

            steps = np.arange(HORIZON_BARS)
            ep = np.repeat(np.arange(n), HORIZON_BARS)
            st = np.tile(steps, n)
            mask = st <= exit_pos[ep]

            bg = snap3[sub_pos, ai, :]      # (n, 62) 入场背景，整 episode 不变

            rows = dict(
                candidate_id=keep_ids[sub_pos][ep][mask],
                symbol=sym,
                trading_day=ev_day[sub_pos][ep][mask],
                initial_action=ACTION_NAMES[1 + ai],
                trade_direction=dirv[ep][mask].astype("int8"),
                target_R=np.float64(target_r),
                step=st[mask].astype("int8"),
                # step 0 = 入场状态，仅初始化；agent 强制 HOLD，不允许 EXIT
                is_decision_point=(st >= 1)[mask],
                action_space=np.where(
                    (st >= 1)[mask], "HOLD_EXIT", "HOLD_ONLY"
                ),
                decision_time=t_dec.ravel()[mask],
                decision_bar_index=w.ravel()[mask].astype("int32"),
                dyn_bar_index=prev_idx.ravel()[mask].astype("int32"),
                current_open=o12.ravel()[mask],
                entry_price=entry[ep][mask],
                atr5=atr_s[ep][mask],
                exit_value_R=o.ravel()[mask],
                terminal=(st == exit_pos[ep])[mask],
                terminal_reason=np.where(
                    (st == exit_pos[ep]),
                    np.array([EXIT_CODE_NAMES[c] for c in code])[ep],
                    "",
                )[mask],
                all_hold_terminal_R=r[ep][mask],
            )
            for k, cname in enumerate(MODEL_FEATURES_V0):
                col = bg[ep, k][mask]
                if cname in cat_set:
                    # 真离散字段（如 source_tf）保留为类别，不强行转数值
                    rows[f"bg_{cname}"] = pd.Categorical(col.astype(str))
                else:
                    rows[f"bg_{cname}"] = col.astype("float32")
            for c in DYN_COLS:
                rows[f"dyn_{c}"] = dyn_vals[c].ravel()[mask].astype("float32")

            rows["current_unrealized_R"] = o.ravel()[mask]
            rows["distance_to_stop_R"] = o.ravel()[mask] + STOP_ATR
            rows["distance_to_target_R"] = target_r - o.ravel()[mask]
            rows["bars_held"] = st[mask].astype("int8")
            rows["bars_remaining"] = (HORIZON_BARS - 1 - st[mask]).astype("int8")
            rows["max_favorable_excursion_R_so_far"] = mfe.ravel()[mask]
            rows["max_adverse_excursion_R_so_far"] = mae.ravel()[mask]
            rows["previous_step_price_change_R"] = prev_chg.ravel()[mask]
            rows["minutes_since_previous_valid_bar"] = minutes.ravel()[mask]
            rows["crossed_session_break"] = crossed.ravel()[mask]

            frames.append(pd.DataFrame(rows))
            epi_stats.append(dict(
                symbol=sym, action=ACTION_NAMES[1 + ai], episodes=int(n),
                gap_stop=int(gs.sum()), gap_target=int(gt.sum()),
                both=int(is_both.sum()), stop=int(is_stop.sum()),
                target=int(is_target.sum()), timeout=int(is_timeout.sum()),
            ))
            print(f"[M4]   {sym} {ACTION_NAMES[1+ai]}: episodes={n} "
                  f"rows={int(mask.sum())}", flush=True)

    traj = pd.concat(frames, ignore_index=True)
    traj.to_parquet(TRAJ, index=False)
    print(f"\n[M4] trajectory rows={len(traj)} cols={traj.shape[1]}", flush=True)

    # -------------------------------------------------------------- #
    # 3. 因果 / 一致性检查
    # -------------------------------------------------------------- #
    order, keep = load_universe_keep()
    rewards = np.load(REWARD_NPY)
    ref = rewards[keep][:, 1:]          # (n_kept, 6)
    diff = np.abs(hold_R - ref)
    max_diff = float(np.nanmax(diff))
    n_bad = int((diff > 1e-9).sum())
    n_cmp = int(np.isfinite(diff).sum())

    checks = dict(
        episode_count=int(traj.groupby(["candidate_id", "initial_action"]).ngroups),
        row_count=int(len(traj)),
        max_step=int(traj["step"].max()),
        steps_distribution={
            int(k): int(v) for k, v in traj["step"].value_counts().sort_index().items()
        },
        terminal_reason_distribution={
            k: int(v)
            for k, v in traj.loc[traj["terminal"], "terminal_reason"]
            .value_counts().items()
        },
        forced_gap_stop_count=int(
            sum(e["gap_stop"] for e in epi_stats)
        ),
        forced_gap_target_count=int(
            sum(e["gap_target"] for e in epi_stats)
        ),
        both_conservative_stop_count=int(
            sum(e["both"] for e in epi_stats)
        ),
        stop_count=int(sum(e["stop"] for e in epi_stats)),
        target_count=int(sum(e["target"] for e in epi_stats)),
        timeout_count=int(sum(e["timeout"] for e in epi_stats)),
        # 因果检查
        all_hold_vs_authoritative_max_abs_diff=max_diff,
        all_hold_mismatches=n_bad,
        all_hold_compared=n_cmp,
        episode_single_trading_day=bool(
            traj.groupby("candidate_id")["trading_day"].nunique().max() == 1
        ),
        dyn_bar_before_decision_bar=bool(
            (traj["dyn_bar_index"] < traj["decision_bar_index"]).all()
        ),
        exit_value_matches_current_open=bool(
            np.nanmax(
                np.abs(
                    traj["exit_value_R"]
                    - traj["trade_direction"]
                    * (traj["current_open"] - traj["entry_price"])
                    / traj["atr5"]
                )
            ) < 1e-9
        ),
        terminal_only_at_last_step_of_episode=bool(
            traj.groupby(["candidate_id", "initial_action"])["terminal"].sum().max() == 1
        ),
        step0_is_hold_only=bool(
            (traj.loc[traj["step"] == 0, "action_space"] == "HOLD_ONLY").all()
            and (not traj.loc[traj["step"] == 0, "is_decision_point"].any())
        ),
        max_decision_points_per_episode=int(
            traj[traj["is_decision_point"]]
            .groupby(["candidate_id", "initial_action"])["step"].nunique().max()
        ),
        horizon_bars=HORIZON_BARS,
        stop_atr=STOP_ATR,
        target_R=list(TARGET_R),
        per_symbol_action=epi_stats,
    )
    (OUT_DIR / "rl_exit_audit.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== M4-PRE 轨迹统计 ===")
    for k in (
        "episode_count", "row_count", "max_step",
        "forced_gap_stop_count", "forced_gap_target_count",
        "both_conservative_stop_count", "stop_count", "target_count",
        "timeout_count",
    ):
        print(f"  {k}: {checks[k]}")
    print(f"  steps_distribution: {checks['steps_distribution']}")
    print(f"  terminal_reason_distribution: {checks['terminal_reason_distribution']}")
    print("\n=== 因果检查 ===")
    for k in (
        "all_hold_vs_authoritative_max_abs_diff", "all_hold_mismatches",
        "all_hold_compared", "episode_single_trading_day",
        "dyn_bar_before_decision_bar", "exit_value_matches_current_open",
        "terminal_only_at_last_step_of_episode",
    ):
        print(f"  {k}: {checks[k]}")
    print("\nM4_PRE_DONE")


if __name__ == "__main__":
    main()
