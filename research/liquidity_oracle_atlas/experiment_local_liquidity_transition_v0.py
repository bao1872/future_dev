"""LOCAL-0 — Local Liquidity Transition Baseline v0

===========================================================================
研究问题
===========================================================================
在每个 5m bar **收盘时**，冻结当时最近的 active upper liquidity U 与最近的
active lower liquidity D，问：

    U 与 D 谁先被 penetration？

===========================================================================
数学对象（本轮定稿）
===========================================================================
两个 competing liquidity transitions + right censoring：

    市场事件   : UP , DOWN
    观察状态   : RIGHT_CENSOR   （连续段/数据末尾截断，不是"市场选择"）
    不判定     : AMBIGUOUS      （同一 5m bar 上下同时穿透，不猜 intrabar 顺序）

CENSOR 不再作为第三个预测类别。方向模型只在 resolved (UP/DOWN) 上训练。

===========================================================================
LOCAL_LIFECYCLE_V2 — bar-index lifecycle（独立于 frozen，不修改 frozen）
===========================================================================
每个 identity 只有三个整数：

    activation_bar   = available_bar_index
    penetration_bar  = corrected strict first penetration bar（-1 = 无）
    expiry_bar       = 所在连续段的第一个 discontinuity bar（exclusive；无则 n）

    penetration 搜索：从 activation_bar + 1 开始，到 expiry_bar 截断
        side=+1 : 第一个 j 使 high[j] >  level
        side=-1 : 第一个 j 使 low[j]  <  level
        精确相等只是 touch，不消费（frozen v1.1 的 TOUCH re-arm 会漏记）

active at decision bar i：

    activation_bar <= i  AND  i < expiry_bar
    AND (penetration_bar < 0  OR  i < penetration_bar)

注意：
  * activation **不推迟**。level 在 activation_bar 收盘时已 causal available，
    当时就可以作为 boundary；future penetration 从下一根新 bar 开始观察。
  * expiry 必须进入 active mask（否则未穿透的 level 会跨 discontinuity 存活）。

===========================================================================
历史 P0（均已修）
===========================================================================
P0-1 (f2bb046) master 的 fp 是 side-relative，混合不同 side 无定义。
      -> 改按 (price, side) 分组，upper 只取 side=+1，lower 只取 side=-1。
P0-2 (f2bb046) frozen lifecycle 的 TOUCH re-arm 漏记 strict break。
      -> 重建 strict crossing；实测 7,193/80,289 = 8.96% 旧 fp 偏晚。
P0-3 (585e72e) expiry 没有进入 active mask。-> 本轮修。
P0-4 (585e72e) av_obs 把 activation 推迟了一整根 bar。-> 本轮删除。

===========================================================================
复用（只读，不修改任何 frozen 代码）
===========================================================================
    load_raw_bars / load_master / active_mask_at  <- run_5m_graph_probability_v1
    prepare_master_price_groups / active_prices_chunk
                                                  <- run_5m_graph_target_decision_v1
    （后者只用于 parity 测试，不再是 active 主路径）

===========================================================================
输出（只提交小型结果；大型 parquet/cache 不入库）
===========================================================================
    local0_activation_contract_audit.csv
    local0_lifecycle_correction_audit.csv
    local0_lifecycle_correction_examples.csv
    local0_expiry_audit.csv
    local0_side_position_audit.csv
    local0_dataset_audit.json
    local0_label_distribution.csv
    local0_by_symbol.csv
    local0_distance_bins.csv
    local0_model_metrics.csv
    local0_day_bootstrap.csv
    local0_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# --- frozen base helpers (READ-ONLY reuse) -------------------------------
from research.liquidity_oracle_atlas.run_5m_graph_probability_v1 import (  # noqa: E402
    load_raw_bars,
    load_master,
    active_mask_at,
)
from research.liquidity_oracle_atlas.run_5m_graph_target_decision_v1 import (  # noqa: E402
    prepare_master_price_groups,
    active_prices_chunk,
)

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen constants (NEVER tuned by results)
# ---------------------------------------------------------------------------
MASTER_PATH = (
    REPO / "research/analysis_results/smc_oracle_atlas_v1/liquidity_master_v1_1.parquet"
)
OUT = REPO / "research/analysis_results/local_liquidity_transition_v0"
CACHE = OUT / "cache"

FULL_UNIV = ["AG", "AL", "AU", "CF", "CU", "I", "M", "MA", "NI", "P",
             "RB", "RU", "SC", "SN", "TA"]

BAR_MINUTES = 5
BAR_NS = np.int64(BAR_MINUTES * 60 * 1_000_000_000)
CHUNK_BARS = 256          # memory-safe fixed chunk; NOT tuned by results
IDENTITY_CHUNK = 256
N_BLOCKS = 4
TRAIN_BLOCK = "TB1"
TEST_BLOCK = "TB2"

# 市场事件 / 观察状态
UP, DOWN, RIGHT_CENSOR, AMBIGUOUS = 0, 1, 2, 3
RESOLVED_CODES = (UP, DOWN)
LABEL_NAMES = np.array(["UP", "DOWN", "RIGHT_CENSOR", "AMBIGUOUS"], dtype=object)

SEED = 20260913
N_BOOT = 1000
N_ECE_BINS = 10

DIST_BIN_EDGES = (-np.inf, -1.0, -0.5, 0.0, 0.5, 1.0, np.inf)
DIST_BIN_LABELS = [
    "(-inf,-1]", "(-1,-0.5]", "(-0.5,0]", "(0,0.5]", "(0.5,1]", "(1,inf)",
]

NUM_FEATURES = ["up_distance_R", "down_distance_R", "width_R", "log_distance_ratio"]
CAT_FEATURES = ["symbol"]

# leakage guard: 这些字段绝不允许进入模型特征
FORBIDDEN_FEATURES = {
    "first_penetration_time", "resolution_time", "resolution_ns",
    "resolution_bars", "label", "label_name", "upper_group", "lower_group",
    "up_n_active", "down_n_active", "up_n_finite", "down_n_finite",
    "up_fp_time", "dn_fp_time", "mixed_fp_pattern", "trading_day",
    "decision_bar_index", "resolution_bar_index", "right_censored",
    "upper_activation_bar", "upper_expiry_bar", "lower_activation_bar",
    "lower_expiry_bar", "upper_pen_bar", "lower_pen_bar",
    "future_high", "future_low",
}

I64MAX = np.iinfo(np.int64).max
I64MIN = np.iinfo(np.int64).min
INAT = np.datetime64("NaT").astype("datetime64[ns]").view("int64")
NO_BAR = np.int64(-1)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
def to_ns_int(x) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype.kind == "M":
        return a.astype("datetime64[ns]").view("int64")
    return pd.to_datetime(a).to_numpy().astype("datetime64[ns]").view("int64")


def to_dt64_ns(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.int64).view("datetime64[ns]")


# ===========================================================================
# LOCAL_LIFECYCLE_V2 — bar-index corrected lifecycle
# ===========================================================================
def bar_expiry_index(bars: dict) -> np.ndarray:
    """bar_expiry[i] = 第一个 discontinuity bar index > i；无则 n。

    即：在 bar i 做决策时，观察窗口的 exclusive 上界。
    """
    n = int(bars["n"])
    disc_idx = np.flatnonzero(np.asarray(bars["disc"], dtype=bool))
    if len(disc_idx) == 0:
        return np.full(n, n, dtype=np.int64)
    k = np.searchsorted(disc_idx, np.arange(n, dtype=np.int64), side="right")
    has = k < len(disc_idx)
    out = np.full(n, n, dtype=np.int64)
    out[has] = disc_idx[np.minimum(k[has], len(disc_idx) - 1)]
    return out


def build_corrected_lifecycle(master_sym: pd.DataFrame, bars: dict,
                              chunk: int = IDENTITY_CHUNK) -> dict:
    """重建每个 identity 的 bar-index lifecycle。

    activation_bar  = available_bar_index
    expiry_bar      = bar_expiry[activation_bar]  （所在连续段上界，exclusive）
    penetration_bar = 第一个 j in [activation_bar+1, expiry_bar) 使
                      high[j] > level (side=+1) / low[j] < level (side=-1)
                      无则 -1
    """
    n = int(bars["n"])
    h = np.asarray(bars["h"], dtype=np.float64)
    low = np.asarray(bars["l"], dtype=np.float64)
    bexp = bar_expiry_index(bars)
    cols = np.arange(n, dtype=np.int64)

    price = master_sym["price"].to_numpy(np.float64)
    side = master_sym["side"].to_numpy(np.int64)
    avb = master_sym["available_bar_index"].to_numpy(np.float64)

    m_total = len(price)
    valid = np.isfinite(avb) & (avb >= 0) & (avb < n)

    activation = np.full(m_total, I64MAX, dtype=np.int64)   # invalid -> never active
    activation[valid] = avb[valid].astype(np.int64)
    expiry = np.full(m_total, 0, dtype=np.int64)            # invalid -> never active
    expiry[valid] = bexp[activation[valid]]
    search_start = np.full(m_total, n, dtype=np.int64)
    search_start[valid] = np.minimum(activation[valid] + 1, n)

    penetration = np.full(m_total, NO_BAR, dtype=np.int64)
    for s_val in (1, -1):
        idx = np.flatnonzero((side == s_val) & valid)
        for lo in range(0, len(idx), chunk):
            sel = idx[lo:lo + chunk]
            lv = price[sel][:, None]
            a_i = search_start[sel][:, None]
            e_i = expiry[sel][:, None]
            if s_val == 1:
                cross = h[None, :] > lv
            else:
                cross = low[None, :] < lv
            hit = cross & (cols[None, :] >= a_i) & (cols[None, :] < e_i)
            any_hit = hit.any(axis=1)
            first = np.argmax(hit, axis=1).astype(np.int64)
            penetration[sel] = np.where(any_hit, first, NO_BAR)

    return dict(activation_bar=activation, penetration_bar=penetration,
                expiry_bar=expiry, search_start=search_start)


def lifecycle_correction_audit(sym: str, master_sym: pd.DataFrame, bars: dict,
                               lc: dict) -> dict:
    """旧 frozen fp vs corrected fp 的 parity audit（按 symbol 一行）。"""
    old = to_ns_int(master_sym["first_penetration_time"])
    n = int(bars["n"])
    pen = lc["penetration_bar"]
    end_ns = to_ns_int(bars["t"]) + BAR_NS
    new = np.full(len(pen), INAT, dtype=np.int64)
    ok = pen >= 0
    new[ok] = end_ns[np.minimum(pen[ok], n - 1)]
    old_nat = old == INAT
    new_nat = new == INAT
    both_fin = (~old_nat) & (~new_nat)
    return dict(
        symbol=sym,
        n_identities=int(len(old)),
        both_nat=int((old_nat & new_nat).sum()),
        old_eq_new=int((both_fin & (old == new)).sum()),
        old_later_than_new=int((both_fin & (old > new)).sum()),
        old_earlier_than_new=int((both_fin & (old < new)).sum()),
        old_nat_new_finite=int((old_nat & (~new_nat)).sum()),
        old_finite_new_nat=int(((~old_nat) & new_nat).sum()),
    )


def activation_contract_audit(sym: str, master_sym: pd.DataFrame,
                              bars: dict, lc: dict) -> dict:
    """available_time vs activation_bar 收盘时间 的对照 audit。

    权威 activation = available_bar_index（不推迟）。
    """
    n = int(bars["n"])
    end_ns = to_ns_int(bars["t"]) + BAR_NS
    av_ns = to_ns_int(master_sym["available_time"])
    act = lc["activation_bar"]
    valid = act < I64MAX
    act_end = np.full(len(act), INAT, dtype=np.int64)
    act_end[valid] = end_ns[np.minimum(act[valid], n - 1)]
    before = int(((av_ns < act_end) & valid).sum())
    equal = int(((av_ns == act_end) & valid).sum())
    after = int(((av_ns > act_end) & valid).sum())
    diff = np.full(len(act), I64MAX, dtype=np.int64)
    diff[valid] = (av_ns[valid] - act_end[valid]) // BAR_NS
    return dict(
        symbol=sym,
        n_identities=int(len(act)),
        n_no_bar_after_available=int((~valid).sum()),
        available_time_before_activation_end=before,
        available_time_equal_activation_end=equal,
        available_time_after_activation_end=after,
        diff_bars_min=int(diff[valid].min()) if valid.any() else 0,
        diff_bars_median=float(np.median(diff[valid])) if valid.any() else 0.0,
        diff_bars_max=int(diff[valid].max()) if valid.any() else 0,
    )


def wrong_side_audit(sym: str, master_sym: pd.DataFrame, bars: dict, lc: dict):
    """active identity 出现在 close 错误一侧的审计。

    active side=+1 应该 level >= close；active side=-1 应该 level <= close。
    activation bar 本身允许出现错误侧（level 刚可知，尚无观察 bar）。
    activation bar **之后** 必须为 0，否则 STOP。
    """
    n = int(bars["n"])
    close = np.asarray(bars["c"], dtype=np.float64)
    price = master_sym["price"].to_numpy(np.float64)
    side = master_sym["side"].to_numpy(np.int64)
    act = lc["activation_bar"]
    pen = lc["penetration_bar"]
    exp = lc["expiry_bar"]

    total = 0
    after_activation = 0
    examples = []
    for i in range(len(price)):
        if act[i] >= I64MAX:
            continue
        i0 = int(act[i])
        end = (int(pen[i]) - 1) if pen[i] >= 0 else (int(exp[i]) - 1)
        if end < i0:
            continue
        seg = close[i0:end + 1]
        bad = seg > price[i] if side[i] > 0 else seg < price[i]
        cnt = int(bad.sum())
        if cnt == 0:
            continue
        total += cnt
        n_after = int(bad[1:].sum())
        after_activation += n_after
        if n_after and len(examples) < 50:
            k = 1 + int(np.flatnonzero(bad[1:])[0])
            examples.append(dict(
                symbol=sym,
                liquidity_id=str(master_sym["liquidity_id"].to_numpy(object)[i]),
                side=int(side[i]), price=float(price[i]),
                activation_bar=i0, penetration_bar=int(pen[i]),
                expiry_bar=int(exp[i]), violation_bar=i0 + k,
                close=float(seg[k]),
            ))
    return total, after_activation, examples


# ===========================================================================
# (price, side) level groups
# ===========================================================================
def build_level_groups(master_sym: pd.DataFrame, lc: dict,
                       bars: dict = None) -> dict:
    """(price, side) 分组 + bar-index lifecycle。

    bars 给定时额外填充 datetime 字段（available_time / first_penetration_time），
    只用于与 frozen active_prices_chunk 的 parity 测试；active 主路径是 bar-index。
    """
    price = master_sym["price"].to_numpy(np.float64)
    side = master_sym["side"].to_numpy(np.int64)
    order = np.lexsort((side, price))
    price_s = price[order]
    side_s = side[order]
    new_grp = np.r_[True,
                    (price_s[1:] != price_s[:-1]) | (side_s[1:] != side_s[:-1])]
    starts = np.flatnonzero(new_grp)
    n_id = len(order)
    lengths = np.diff(np.r_[starts, n_id]).astype(np.int64)

    act = np.asarray(lc["activation_bar"], np.int64)[order]
    pen = np.asarray(lc["penetration_bar"], np.int64)[order]
    exp = np.asarray(lc["expiry_bar"], np.int64)[order]

    # datetime 字段：仅用于与 frozen active_prices_chunk 的 parity 测试
    av_ns = to_ns_int(master_sym["available_time"])[order]
    fp_ns = np.full(n_id, INAT, dtype=np.int64)
    if bars is not None:
        end_ns = to_ns_int(bars["t"]) + BAR_NS
        ok = pen >= 0
        fp_ns[ok] = end_ns[np.minimum(pen[ok], int(bars["n"]) - 1)]
    return dict(
        unique_price=price_s[starts],
        unique_side=side_s[starts],
        group_starts=starts,
        group_lengths=lengths,
        id_group=np.repeat(np.arange(len(starts), dtype=np.int64), lengths),
        activation_bar=act,
        penetration_bar=pen,
        expiry_bar=exp,
        identity_order=np.asarray(order, dtype=np.int64),
        available_time=av_ns.view("datetime64[ns]"),
        first_penetration_time=fp_ns.view("datetime64[ns]"),
        liquidity_id_sorted=master_sym["liquidity_id"].to_numpy(object)[order],
        identity_price=price_s,
        identity_side=side_s,
        identity_ltype=master_sym["liquidity_type"].to_numpy(object)[order],
    )


# ---------------------------------------------------------------------------
# Active kernel (bar-index, vectorized)
# ---------------------------------------------------------------------------
def active_level_groups_chunk(bar_idx: np.ndarray, info: dict) -> np.ndarray:
    """(chunk, n_group) bool：group 在 decision bar 上是否 active。

    active = activation_bar <= i < expiry_bar
             AND (penetration_bar < 0 OR i < penetration_bar)
    """
    i = np.asarray(bar_idx, dtype=np.int64)[:, None]
    active_id = (
        (info["activation_bar"][None, :] <= i)
        & (i < info["expiry_bar"][None, :])
        & (
            (info["penetration_bar"][None, :] < 0)
            | (i < info["penetration_bar"][None, :])
        )
    )
    return np.logical_or.reduceat(active_id, info["group_starts"], axis=1)


def nearest_active_pair_chunk_v2(bar_idx: np.ndarray, close: np.ndarray,
                                 info: dict) -> dict:
    """upper 只取 side=+1 且 price > close；lower 只取 side=-1 且 price < close。"""
    price = info["unique_price"]
    gside = info["unique_side"]
    active = active_level_groups_chunk(bar_idx, info)
    delta = price[None, :] - close[:, None]

    is_up = gside[None, :] > 0
    is_dn = gside[None, :] < 0

    work = np.where(active & is_up & (delta > 0.0), delta, np.inf)
    up_idx = np.argmin(work, axis=1)
    up_dist = work[np.arange(len(close)), up_idx]
    has_up = np.isfinite(up_dist)

    work = np.where(active & is_dn & (delta < 0.0), -delta, np.inf)
    down_idx = np.argmin(work, axis=1)
    down_dist = work[np.arange(len(close)), down_idx]
    has_down = np.isfinite(down_dist)

    upper = np.full(len(close), np.nan, dtype=np.float64)
    lower = np.full(len(close), np.nan, dtype=np.float64)
    upper[has_up] = price[up_idx[has_up]]
    lower[has_down] = price[down_idx[has_down]]
    return {
        "upper_price": upper,
        "lower_price": lower,
        "upper_group": np.where(has_up, up_idx, -1).astype(np.int64),
        "lower_group": np.where(has_down, down_idx, -1).astype(np.int64),
        "has_upper": has_up,
        "has_lower": has_down,
    }


def resolve_group_event(bar_idx: np.ndarray, group_idx: np.ndarray,
                        info: dict) -> dict:
    """selected (price, side) group 内 active identities 的 next event。

    contract:
      * finite penetration_bar 必须一致
      * active identities 的 expiry_bar 必须一致
      * 不允许 "一个 finite / 一个 NaT" 的 mixed pattern
    """
    gid = info["id_group"]
    ab = info["activation_bar"]
    eb = info["expiry_bar"]
    pb = info["penetration_bar"]
    i = np.asarray(bar_idx, dtype=np.int64)[:, None]

    sel = gid[None, :] == group_idx[:, None]
    act = (sel & (ab[None, :] <= i) & (i < eb[None, :])
           & ((pb[None, :] < 0) | (i < pb[None, :])))
    fin = act & (pb[None, :] >= 0)

    n_active = act.sum(axis=1).astype(np.int64)
    n_finite = fin.sum(axis=1).astype(np.int64)

    pmin = np.min(np.where(fin, pb[None, :], I64MAX), axis=1)
    pmax = np.max(np.where(fin, pb[None, :], I64MIN), axis=1)
    emin = np.min(np.where(act, eb[None, :], I64MAX), axis=1)
    emax = np.max(np.where(act, eb[None, :], I64MIN), axis=1)

    has_finite = n_finite > 0
    conflict_pen = has_finite & (pmin != pmax)
    conflict_exp = (emin != emax)
    conflict_mixed = has_finite & (n_finite < n_active)
    conflict = conflict_pen | conflict_exp | conflict_mixed
    return dict(
        pen=np.where(has_finite, pmin, NO_BAR),
        expiry=np.where(n_active > 0, emin, NO_BAR),
        n_active=n_active, n_finite=n_finite,
        has_finite=has_finite, mixed=conflict_mixed,
        conflict=conflict,
        conflict_pen=conflict_pen, conflict_exp=conflict_exp,
    )


# ---------------------------------------------------------------------------
# Outcome
# ---------------------------------------------------------------------------
def classify_pair_event(up_pen, dn_pen) -> np.ndarray:
    """UP=0 / DOWN=1 / RIGHT_CENSOR=2 / AMBIGUOUS=3（输入为 bar index，-1 = 无）。"""
    up = np.atleast_1d(np.asarray(up_pen, dtype=np.int64))
    dn = np.atleast_1d(np.asarray(dn_pen, dtype=np.int64))
    up_missing = up < 0
    dn_missing = dn < 0
    out = np.full(up.shape, RIGHT_CENSOR, dtype=np.int64)
    out[dn_missing & ~up_missing] = UP
    out[up_missing & ~dn_missing] = DOWN
    both = ~up_missing & ~dn_missing
    out[both & (up < dn)] = UP
    out[both & (dn < up)] = DOWN
    out[both & (up == dn)] = AMBIGUOUS
    return out


# ---------------------------------------------------------------------------
# Time blocks
# ---------------------------------------------------------------------------
def build_blocks(bars_by_symbol: dict):
    day_arrays = [
        np.asarray(b["td"]).astype("datetime64[D]") for b in bars_by_symbol.values()
    ]
    all_days = np.unique(np.concatenate(day_arrays))
    chunks = np.array_split(all_days, N_BLOCKS)
    day_block_code = np.repeat(
        np.arange(N_BLOCKS, dtype=np.int64), [len(c) for c in chunks]
    )
    boundaries = [dict(block=f"TB{i + 1}", first_day=str(c[0]),
                       last_day=str(c[-1]), n_days=int(len(c)))
                  for i, c in enumerate(chunks)]
    return all_days, day_block_code, boundaries


def block_codes_for(bars: dict, all_days: np.ndarray,
                    day_block_code: np.ndarray) -> np.ndarray:
    td = np.asarray(bars["td"]).astype("datetime64[D]")
    return day_block_code[np.searchsorted(all_days, td)]


# ---------------------------------------------------------------------------
# Conflict detail
# ---------------------------------------------------------------------------
def collect_group_rows(sym, info: dict, bar_idx: np.ndarray,
                       group_idx: np.ndarray, cap: int = 400):
    ab, eb, pb = info["activation_bar"], info["expiry_bar"], info["penetration_bar"]
    gid, lid = info["id_group"], info["liquidity_id_sorted"]
    px, sd = info["identity_price"], info["identity_side"]
    rows = []
    for b, g in zip(bar_idx, group_idx):
        for j in np.flatnonzero(gid == int(g)):
            active = bool((ab[j] <= b) and (b < eb[j])
                          and ((pb[j] < 0) or (b < pb[j])))
            rows.append(dict(
                symbol=sym, decision_bar_index=int(b), price=float(px[j]),
                side=int(sd[j]), liquidity_id=str(lid[j]),
                activation_bar=int(ab[j]) if ab[j] < I64MAX else -1,
                expiry_bar=int(eb[j]), penetration_bar=int(pb[j]),
                is_active=active,
            ))
        if len(rows) >= cap:
            break
    return rows


# ---------------------------------------------------------------------------
# Per-symbol sample builder
# ---------------------------------------------------------------------------
def build_symbol_samples(sym: str, master_sym: pd.DataFrame, bars: dict,
                         all_days: np.ndarray, day_block_code: np.ndarray,
                         lc: dict, stop_on_conflict: bool = True):
    info = build_level_groups(master_sym, lc, bars)
    n = int(bars["n"])
    t_ns = to_ns_int(bars["t"])
    end_ns = t_ns + BAR_NS
    bexp = bar_expiry_index(bars)
    close = np.asarray(bars["c"], dtype=np.float64)
    atr = np.asarray(bars["atr"], dtype=np.float64)
    codes = block_codes_for(bars, all_days, day_block_code)
    bar_idx = np.arange(n, dtype=np.int64)

    up_px = np.full(n, np.nan, dtype=np.float64)
    dn_px = np.full(n, np.nan, dtype=np.float64)
    up_g = np.full(n, -1, dtype=np.int64)
    dn_g = np.full(n, -1, dtype=np.int64)
    up_pen = np.full(n, NO_BAR, dtype=np.int64)
    dn_pen = np.full(n, NO_BAR, dtype=np.int64)
    up_exp = np.full(n, NO_BAR, dtype=np.int64)
    dn_exp = np.full(n, NO_BAR, dtype=np.int64)
    up_na = np.zeros(n, dtype=np.int64)
    dn_na = np.zeros(n, dtype=np.int64)
    up_nf = np.zeros(n, dtype=np.int64)
    dn_nf = np.zeros(n, dtype=np.int64)
    conflict_mask = np.zeros(n, dtype=bool)

    for lo in range(0, n, CHUNK_BARS):
        hi = min(lo + CHUNK_BARS, n)
        sl = slice(lo, hi)
        bic = bar_idx[sl]
        pair = nearest_active_pair_chunk_v2(bic, close[sl], info)
        up_px[sl] = pair["upper_price"]
        dn_px[sl] = pair["lower_price"]
        up_g[sl] = pair["upper_group"]
        dn_g[sl] = pair["lower_group"]

        iu = np.flatnonzero(pair["has_upper"])
        if len(iu):
            r = resolve_group_event(bic[iu], pair["upper_group"][iu], info)
            up_pen[lo + iu] = r["pen"]
            up_exp[lo + iu] = r["expiry"]
            up_na[lo + iu] = r["n_active"]
            up_nf[lo + iu] = r["n_finite"]
            conflict_mask[lo + iu[r["conflict"]]] = True
        idn = np.flatnonzero(pair["has_lower"])
        if len(idn):
            r = resolve_group_event(bic[idn], pair["lower_group"][idn], info)
            dn_pen[lo + idn] = r["pen"]
            dn_exp[lo + idn] = r["expiry"]
            dn_na[lo + idn] = r["n_active"]
            dn_nf[lo + idn] = r["n_finite"]
            conflict_mask[lo + idn[r["conflict"]]] = True

        if conflict_mask[sl].any():
            if not stop_on_conflict:
                continue
            bad = np.flatnonzero(conflict_mask[sl])
            rows = []
            for b in bad:
                for gsel in (up_g[lo + b], dn_g[lo + b]):
                    if gsel < 0:
                        continue
                    rows += collect_group_rows(
                        sym, info, np.array([bic[b]]), np.array([gsel]))
                if len(rows) >= 400:
                    break
            pd.DataFrame(rows).to_csv(OUT / "local0_group_event_conflict.csv",
                                      index=False)
            raise SystemExit(
                "STOP_LOCAL0_GROUP_EVENT_CONFLICT: same (price, side) active "
                f"identities disagree (symbol={sym}, "
                f"n_conflict_bars={int(conflict_mask.sum())}); "
                f"details -> {OUT / 'local0_group_event_conflict.csv'}"
            )

    has_up = up_g >= 0
    has_dn = dn_g >= 0
    both = has_up & has_dn
    atr_ok = np.isfinite(atr) & (atr > 0.0) & np.isfinite(close)
    primary = both & atr_ok

    if primary.any():
        assert bool((up_px[primary] > close[primary]).all()), "UPPER_NOT_ABOVE_CLOSE"
        assert bool((dn_px[primary] < close[primary]).all()), "LOWER_NOT_BELOW_CLOSE"
        assert bool(((up_pen[primary] > bar_idx[primary])
                     | (up_pen[primary] == NO_BAR)).all()), "UPPER_PEN_NOT_FUTURE"
        assert bool(((dn_pen[primary] > bar_idx[primary])
                     | (dn_pen[primary] == NO_BAR)).all()), "LOWER_PEN_NOT_FUTURE"
        # L1: selected identity 必须还没 expiry
        assert bool((bar_idx[primary] < up_exp[primary]).all()), "UPPER_EXPIRED"
        assert bool((bar_idx[primary] < dn_exp[primary]).all()), "LOWER_EXPIRED"

    label = np.full(n, -1, dtype=np.int64)
    res_bar = np.full(n, NO_BAR, dtype=np.int64)
    if both.any():
        lab = classify_pair_event(up_pen[both], dn_pen[both])
        label[both] = lab
        is_cens = lab == RIGHT_CENSOR
        res_bar[both] = np.where(
            is_cens, bexp[both] - 1,
            np.where(lab == DOWN, dn_pen[both], up_pen[both]))
        res_bar[both] = np.maximum(res_bar[both], bar_idx[both])

    idx = np.flatnonzero(primary)
    up_dist = np.full(n, np.nan, dtype=np.float64)
    dn_dist = np.full(n, np.nan, dtype=np.float64)
    up_dist[idx] = (up_px[idx] - close[idx]) / atr[idx]
    dn_dist[idx] = (close[idx] - dn_px[idx]) / atr[idx]

    df = pd.DataFrame(dict(
        symbol=sym,
        decision_bar_index=bar_idx[idx],
        decision_time=to_dt64_ns(end_ns[idx]),
        trading_day=np.asarray(bars["td"]).astype("datetime64[D]")[idx],
        block=np.array([f"TB{c + 1}" for c in codes[idx]], dtype=object),
        close=close[idx],
        atr=atr[idx],
        upper_price=up_px[idx],
        lower_price=dn_px[idx],
        up_distance_R=up_dist[idx],
        down_distance_R=dn_dist[idx],
        width_R=up_dist[idx] + dn_dist[idx],
        log_distance_ratio=np.log((up_dist[idx] + 1e-8) / (dn_dist[idx] + 1e-8)),
        label=label[idx],
        resolution_bar_index=res_bar[idx],
        resolution_time=to_dt64_ns(end_ns[np.clip(res_bar[idx], 0, n - 1)]),
        resolution_bars=(res_bar[idx] - bar_idx[idx]).astype(np.float64),
        right_censored=(label[idx] == RIGHT_CENSOR).astype(int),
        up_n_active=up_na[idx], down_n_active=dn_na[idx],
        up_n_finite=up_nf[idx], down_n_finite=dn_nf[idx],
        upper_group=up_g[idx], lower_group=dn_g[idx],
        upper_activation_bar=info["activation_bar"][np.clip(up_g[idx], 0, None)],
        upper_expiry_bar=up_exp[idx],
        lower_activation_bar=info["activation_bar"][np.clip(dn_g[idx], 0, None)],
        lower_expiry_bar=dn_exp[idx],
        upper_pen_bar=up_pen[idx], lower_pen_bar=dn_pen[idx],
    ))
    df["mixed_fp_pattern"] = (
        ((df["up_n_finite"] > 0) & (df["up_n_finite"] < df["up_n_active"]))
        | ((df["down_n_finite"] > 0) & (df["down_n_finite"] < df["down_n_active"]))
    ).astype(int)

    audit = dict(
        symbol=sym,
        n_bars=int(n),
        n_conflict_bars=int(conflict_mask.sum()),
        n_atr_or_close_invalid=int((~atr_ok).sum()),
        n_has_upper=int(has_up.sum()),
        n_has_lower=int(has_dn.sum()),
        n_missing_upper=int((~has_up).sum()),
        n_missing_lower=int((~has_dn).sum()),
        n_missing_both=int((~has_up & ~has_dn).sum()),
        n_pair_both=int(both.sum()),
        n_primary=int(primary.sum()),
        n_mixed_fp_pattern=int(df["mixed_fp_pattern"].sum()),
        n_multi_identity_selected=int(
            ((df["up_n_active"] > 1) | (df["down_n_active"] > 1)).sum()),
    )
    return df, audit


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def binary_logloss(P: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.maximum(P[np.arange(len(y)), y], 1e-12))))


def binary_brier(P: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((P[:, UP] - (y == UP).astype(float)) ** 2))


def ece_binary(p: np.ndarray, y: np.ndarray, n_bins: int = N_ECE_BINS) -> float:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(p) == 0:
        return float("nan")
    b = np.clip(np.floor(p * n_bins).astype(int), 0, n_bins - 1)
    tot, n = 0.0, float(len(p))
    for k in range(n_bins):
        m = b == k
        if m.any():
            tot += (m.sum() / n) * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(tot)


def fit_binary(train: pd.DataFrame) -> Pipeline:
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=3000)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(train[NUM_FEATURES + CAT_FEATURES], train["label"].to_numpy())
    classes = np.asarray(pipe.named_steps["clf"].classes_, dtype=int)
    assert np.array_equal(np.sort(classes), np.array([UP, DOWN])), \
        f"CLASSES_NOT_BINARY_UP_DOWN: {classes}"
    return pipe


def proba_binary(pipe, df: pd.DataFrame) -> np.ndarray:
    P = pipe.predict_proba(df[pipe.feature_names_in_])
    classes = np.asarray(pipe.named_steps["clf"].classes_, dtype=int)
    out = np.zeros((len(df), 2), dtype=np.float64)
    out[:, classes] = P
    return out


def evaluate_binary(P: np.ndarray, y: np.ndarray, name: str) -> dict:
    return dict(
        model=name,
        n=int(len(y)),
        logloss=binary_logloss(P, y),
        brier=binary_brier(P, y),
        ece_up=ece_binary(P[:, UP], (y == UP).astype(float)),
        ece_down=ece_binary(P[:, DOWN], (y == DOWN).astype(float)),
        up_pct=float(np.mean(y == UP)),
        down_pct=float(np.mean(y == DOWN)),
    )


def day_paired_bootstrap(days: np.ndarray, ll_b0: np.ndarray,
                         ll_b1: np.ndarray, seed: int = SEED,
                         n_boot: int = N_BOOT) -> dict:
    g0 = pd.Series(ll_b0).groupby(days).mean()
    g1 = pd.Series(ll_b1).groupby(days).mean()
    delta_day = (g1 - g0).to_numpy()
    rng = np.random.default_rng(seed)
    n = len(delta_day)
    reps = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        reps[i] = float(delta_day[rng.integers(0, n, n)].mean())
    lo, hi = float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))
    if hi < 0.0:
        verdict = "LOCAL_GEOMETRY_INCREMENT_SUPPORTED"
    elif lo > 0.0:
        verdict = "LOCAL_GEOMETRY_INCREMENT_NEGATIVE"
    else:
        verdict = "LOCAL_GEOMETRY_INCREMENT_AMBIGUOUS"
    return dict(n_days=int(n), n_boot=int(n_boot), seed=int(seed),
                mean_delta_logloss=float(delta_day.mean()),
                ci_lo=lo, ci_hi=hi, verdict=verdict)


# ---------------------------------------------------------------------------
# Hard leakage guards (independent recompute from the tabular lifecycle)
# ---------------------------------------------------------------------------
def run_leakage_guards(samples: pd.DataFrame, master_lc: pd.DataFrame,
                       tb2_start_ns: int, train: pd.DataFrame,
                       test: pd.DataFrame, n_guard_rows: int = 3000):
    msgs, ok = [], True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        msgs.append(f"[{'PASS' if cond else 'FAIL'}] {name}"
                    + (f" :: {detail}" if detail else ""))

    check("G8a train block is TB1 only",
          set(train["block"].unique()) <= {TRAIN_BLOCK},
          str(sorted(train["block"].unique())))
    check("G8b test block is TB2 only",
          set(test["block"].unique()) <= {TEST_BLOCK},
          str(sorted(test["block"].unique())))
    check("G8c TB3/TB4 present but unused",
          int(samples["block"].isin(["TB3", "TB4"]).sum()) >= 0,
          f"n_tb34={int(samples['block'].isin(['TB3', 'TB4']).sum())}")

    tr_res = to_ns_int(train["resolution_time"])
    check("G7 kept-train resolution_time < TB2 start",
          bool((tr_res < tb2_start_ns).all()),
          f"max_train_res={pd.Timestamp(int(max(tr_res)), unit='ns')} "
          f"tb2_start={pd.Timestamp(tb2_start_ns, unit='ns')}")

    used = set(NUM_FEATURES) | set(CAT_FEATURES)
    check("G6 no forbidden feature", not (used & FORBIDDEN_FEATURES),
          str(sorted(used & FORBIDDEN_FEATURES)))
    check("G5 features are the declared decision-time set",
          used == {"up_distance_R", "down_distance_R", "width_R",
                   "log_distance_ratio", "symbol"}, str(sorted(used)))

    s = samples.reset_index(drop=True)
    sub = (s.iloc[np.linspace(0, len(s) - 1, n_guard_rows).astype(int)]
           if len(s) > n_guard_rows else s)

    g1 = g2 = g3 = g4 = g_exp = g_pen = g_mixed = True
    bad = ""
    for sym, gg in sub.groupby("symbol"):
        ms = master_lc[master_lc["symbol"] == sym]
        px = ms["price"].to_numpy(np.float64)
        sd = ms["side"].to_numpy(np.int64)
        ab = ms["activation_bar"].to_numpy(np.int64)
        eb = ms["expiry_bar"].to_numpy(np.int64)
        pb = ms["penetration_bar"].to_numpy(np.int64)
        for _, r in gg.iterrows():
            i = int(r["decision_bar_index"])
            close = float(r["close"])
            am = (ab <= i) & (i < eb) & ((pb < 0) | (i < pb))
            act_px = px[am]
            up_c = act_px[(act_px > close) & (sd[am] > 0)]
            dn_c = act_px[(act_px < close) & (sd[am] < 0)]
            if len(up_c) == 0 or len(dn_c) == 0:
                g3, bad = False, f"{sym} no side-consistent pair at bar {i}"
                break
            if not np.isclose(float(up_c.min()), float(r["upper_price"])):
                g3, bad = False, f"{sym} bar {i} upper {up_c.min()} != {r['upper_price']}"
                break
            if not np.isclose(float(dn_c.max()), float(r["lower_price"])):
                g4, bad = False, f"{sym} bar {i} lower {dn_c.max()} != {r['lower_price']}"
                break
            for lvl, want, k in ((float(r["upper_price"]), 1, "upper"),
                                 (float(r["lower_price"]), -1, "lower")):
                m = am & (px == lvl) & (sd == want)
                if not m.any():
                    g1, bad = False, f"{sym} bar {i} no active identity at {lvl}"
                    break
                if not bool((ab[m] <= i).all()):
                    g1, bad = False, f"{sym} bar {i} activation > decision bar"
                    break
                if not bool((i < eb[m]).all()):
                    g_exp, bad = False, f"{sym} bar {i} expired identity still active"
                    break
                if not bool(((pb[m] < 0) | (i < pb[m])).all()):
                    g2, bad = False, f"{sym} bar {i} consumed identity still active"
                    break
                pens = pb[m][pb[m] >= 0]
                if len(pens) > 1 and len(np.unique(pens)) > 1:
                    g_pen, bad = False, f"{sym} bar {i} pen conflict at {lvl}"
                    break
                if 0 < len(pens) < int(m.sum()):
                    g_mixed, bad = False, f"{sym} bar {i} mixed pattern at {lvl}"
                    break
            if not (g1 and g2 and g3 and g4 and g_exp and g_pen and g_mixed):
                break
        if not (g1 and g2 and g3 and g4 and g_exp and g_pen and g_mixed):
            break

    check("G1 activation_bar <= decision_bar (independent recompute)", g1, bad)
    check("G2 penetration_bar > decision_bar or none (independent)", g2, bad)
    check("G3 upper = nearest active side=+1 above close", g3, bad)
    check("G4 lower = nearest active side=-1 below close", g4, bad)
    check("L2 no identity active at/after expiry_bar", g_exp, bad)
    check("F same-(price,side) penetration consistent", g_pen, bad)
    check("F no mixed finite/NaT inside same group", g_mixed, bad)

    for m in msgs:
        print("  " + m)
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--mode", default="run", choices=["run", "bench"])
    ap.add_argument("--force-rebuild", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    # 本脚本自己的上一轮产物：只有真正跑到模型阶段才会重新生成，
    # 先删除，避免 STOP 时留下过期结果被误读。
    for stale in ("local0_model_metrics.csv", "local0_day_bootstrap.csv",
                  "local0_summary.json", "local0_distance_bins.csv",
                  "local0_group_event_conflict.csv",
                  "local0_wrong_side_violations.csv"):
        p = OUT / stale
        if p.exists():
            p.unlink()
    t_total = time.perf_counter()
    timing = {}

    symbols = args.symbols or list(FULL_UNIV)
    print(f"[LOCAL-0] symbols={symbols}")

    t0 = time.perf_counter()
    master = load_master()
    ms_by_sym = {s: master[master["symbol"] == s].copy() for s in symbols}
    bars_by_sym = {s: load_raw_bars(s) for s in symbols}
    timing["load_seconds"] = round(time.perf_counter() - t0, 2)

    all_days, day_block_code, boundaries = build_blocks(bars_by_sym)
    print("[BLOCKS] " + "; ".join(
        f"{b['block']}=[{b['first_day']}..{b['last_day']}]({b['n_days']}d)"
        for b in boundaries))

    # ---------------- corrected lifecycle ----------------
    t0 = time.perf_counter()
    lc_by_sym = {s: build_corrected_lifecycle(ms_by_sym[s], bars_by_sym[s])
                 for s in symbols}
    n_identities_total = int(sum(len(ms_by_sym[s]) for s in symbols))
    timing["lifecycle_build_seconds"] = round(time.perf_counter() - t0, 2)
    timing["identities_per_second"] = float(
        n_identities_total / max(timing["lifecycle_build_seconds"], 1e-9))
    print(f"[LIFECYCLE] identities={n_identities_total} "
          f"seconds={timing['lifecycle_build_seconds']} "
          f"ids/sec={timing['identities_per_second']:.0f}")
    if args.mode == "bench":
        print(json.dumps(timing, indent=2))
        return

    # ---------------- audit A: lifecycle parity (unchanged logic) ----------
    aud = pd.DataFrame([lifecycle_correction_audit(s, ms_by_sym[s],
                                                   bars_by_sym[s], lc_by_sym[s])
                        for s in symbols])
    tot = {k: (int(aud[k].sum()) if k != "symbol" else "TOTAL")
           for k in aud.columns}
    aud = pd.concat([aud, pd.DataFrame([tot])], ignore_index=True)
    aud.to_csv(OUT / "local0_lifecycle_correction_audit.csv", index=False)
    print("[LIFECYCLE AUDIT]")
    print(aud.to_string(index=False))

    # ---------------- audit B: activation contract -------------------------
    act_aud = pd.DataFrame([activation_contract_audit(s, ms_by_sym[s],
                                                      bars_by_sym[s],
                                                      lc_by_sym[s])
                            for s in symbols])
    tot = dict(
        symbol="TOTAL",
        n_identities=int(act_aud["n_identities"].sum()),
        n_no_bar_after_available=int(act_aud["n_no_bar_after_available"].sum()),
        available_time_before_activation_end=int(
            act_aud["available_time_before_activation_end"].sum()),
        available_time_equal_activation_end=int(
            act_aud["available_time_equal_activation_end"].sum()),
        available_time_after_activation_end=int(
            act_aud["available_time_after_activation_end"].sum()),
        diff_bars_min=int(act_aud["diff_bars_min"].min()),
        diff_bars_median=float(act_aud["diff_bars_median"].median()),
        diff_bars_max=int(act_aud["diff_bars_max"].max()),
    )
    act_aud = pd.concat([act_aud, pd.DataFrame([tot])], ignore_index=True)
    act_aud.to_csv(OUT / "local0_activation_contract_audit.csv", index=False)
    print("[ACTIVATION CONTRACT]")
    print(act_aud.to_string(index=False))

    # ---------------- audit C: expiry + wrong-side -------------------------
    t0 = time.perf_counter()
    exp_rows, ws_rows, ws_examples = [], [], []
    ws_total = ws_after = 0
    for s in symbols:
        bars = bars_by_sym[s]
        lc = lc_by_sym[s]
        ms = ms_by_sym[s]
        disc_idx = np.flatnonzero(np.asarray(bars["disc"], dtype=bool))
        exp = lc["expiry_bar"]
        valid = lc["activation_bar"] < I64MAX
        exp_rows.append(dict(
            symbol=s,
            n_bars=int(bars["n"]),
            n_discontinuities=int(len(disc_idx)),
            discontinuity_bars=";".join(str(int(x)) for x in disc_idx[:10]),
            n_identities=int(len(exp)),
            n_identities_expiring_before_end_of_data=int(((exp < bars["n"]) & valid).sum()),
            n_identities_roll_censored=int(
                ((lc["penetration_bar"] < 0) & (exp < bars["n"]) & valid).sum()),
        ))
        tt, aa, ex = wrong_side_audit(s, ms, bars, lc)
        ws_total += tt
        ws_after += aa
        ws_examples += ex
        ws_rows.append(dict(symbol=s, wrong_side_all_active=tt,
                            wrong_side_after_activation_bar=aa,
                            selected_pair_wrong_side=0))
    pd.DataFrame(exp_rows).to_csv(OUT / "local0_expiry_audit.csv", index=False)
    pd.DataFrame(ws_rows).to_csv(OUT / "local0_side_position_audit.csv",
                                 index=False)
    print("[EXPIRY]")
    print(pd.DataFrame(exp_rows).to_string(index=False))
    print(f"[WRONG-SIDE] all_active={ws_total} "
          f"after_activation_bar={ws_after} ({time.perf_counter()-t0:.1f}s)")
    if ws_after > 0:
        pd.DataFrame(ws_examples).to_csv(
            OUT / "local0_wrong_side_violations.csv", index=False)
        raise SystemExit(
            "STOP_LOCAL0_WRONG_SIDE_NOT_CONSUMED: "
            f"{ws_after} wrong-side observations AFTER the activation bar; "
            f"examples -> {OUT / 'local0_wrong_side_violations.csv'}")

    # ---------------- local pair + label ----------------
    t0 = time.perf_counter()
    frames, audits = [], []
    for s in symbols:
        cp = CACHE / f"local0_samples_{s}.parquet"
        t1 = time.perf_counter()
        df, au = build_symbol_samples(s, ms_by_sym[s], bars_by_sym[s],
                                      all_days, day_block_code, lc_by_sym[s])
        audits.append(au)
        df.to_parquet(cp, index=False)
        print(f"[BUILD] {s} bars={au['n_bars']} primary={au['n_primary']} "
              f"conflicts={au['n_conflict_bars']} "
              f"({time.perf_counter()-t1:.1f}s)")
        frames.append(df)
    samples = pd.concat(frames, ignore_index=True)
    timing["pair_build_seconds"] = round(time.perf_counter() - t0, 2)

    # ---------------- label / dataset audit ----------------
    t0 = time.perf_counter()
    samples["label_name"] = samples["label"].map(
        {UP: "UP", DOWN: "DOWN", RIGHT_CENSOR: "RIGHT_CENSOR",
         AMBIGUOUS: "AMBIGUOUS"})

    audit = dict(
        n_bars_total=int(sum(int(bars_by_sym[s]["n"]) for s in symbols)),
        n_primary_total=int(len(samples)),
        n_pair_both_total=int(sum(a["n_pair_both"] for a in audits)),
        n_missing_upper=int(sum(a["n_missing_upper"] for a in audits)),
        n_missing_lower=int(sum(a["n_missing_lower"] for a in audits)),
        n_missing_both=int(sum(a["n_missing_both"] for a in audits)),
        n_atr_or_close_invalid=int(sum(a["n_atr_or_close_invalid"] for a in audits)),
        n_mixed_fp_pattern=int(samples["mixed_fp_pattern"].sum()),
        n_multi_identity_selected=int(
            ((samples["up_n_active"] > 1) | (samples["down_n_active"] > 1)).sum()),
        per_symbol=audits,
    )
    vc = samples["label_name"].value_counts()
    for k in ["UP", "DOWN", "RIGHT_CENSOR", "AMBIGUOUS"]:
        audit[f"n_{k}"] = int(vc.get(k, 0))
    audit["ambiguous_rate"] = float(vc.get("AMBIGUOUS", 0) / max(len(samples), 1))
    audit["right_censor_rate"] = float(
        vc.get("RIGHT_CENSOR", 0) / max(len(samples), 1))
    rb = samples["resolution_bars"].to_numpy(float)
    for q, nm in [(0.5, "p50"), (0.9, "p90"), (0.95, "p95"), (0.99, "p99")]:
        audit[f"resolution_bars_{nm}"] = float(np.nanpercentile(rb, q * 100))
    audit["resolution_bars_mean"] = float(np.nanmean(rb))
    cens = samples["right_censored"].to_numpy(bool)
    if cens.any():
        audit["right_censor_resolution_bars_mean"] = float(np.nanmean(rb[cens]))
    for lbl in ["UP", "DOWN", "RIGHT_CENSOR", "AMBIGUOUS"]:
        m = (samples["label_name"] == lbl).to_numpy()
        if m.any():
            audit[f"resolution_bars_mean_{lbl}"] = float(np.nanmean(rb[m]))
    for col in ["up_distance_R", "down_distance_R", "width_R"]:
        v = samples[col].to_numpy(float)
        audit[f"{col}_mean"] = float(np.nanmean(v))
        for q, nm in [(0.5, "p50"), (0.9, "p90"), (0.99, "p99")]:
            audit[f"{col}_{nm}"] = float(np.nanpercentile(v, q * 100))

    dist = (samples.groupby(["block", "label_name"]).size()
            .unstack(fill_value=0).reset_index())
    dist["total"] = dist.drop(columns=["block"]).sum(axis=1)
    dist.to_csv(OUT / "local0_label_distribution.csv", index=False)

    bys = (samples.groupby(["symbol", "label_name"]).size()
           .unstack(fill_value=0))
    for k in ["UP", "DOWN", "RIGHT_CENSOR", "AMBIGUOUS"]:
        if k not in bys:
            bys[k] = 0
    bys["n"] = bys[["UP", "DOWN", "RIGHT_CENSOR", "AMBIGUOUS"]].sum(axis=1)
    for k in ["UP", "DOWN", "RIGHT_CENSOR", "AMBIGUOUS"]:
        bys[f"{k}_pct"] = bys[k] / bys["n"]
    bys = bys.reset_index()[
        ["symbol", "n", "UP", "DOWN", "RIGHT_CENSOR", "AMBIGUOUS",
         "UP_pct", "DOWN_pct", "RIGHT_CENSOR_pct", "AMBIGUOUS_pct"]]
    bys.to_csv(OUT / "local0_by_symbol.csv", index=False)
    timing["label_seconds"] = round(time.perf_counter() - t0, 2)

    # ---------------- train / test split (resolved only) ----------------
    tb2_mask = samples["block"] == TEST_BLOCK
    tb2_start_ns = int(np.min(to_ns_int(samples.loc[tb2_mask, "decision_time"])))

    train_all = samples[(samples["block"] == TRAIN_BLOCK)
                        & samples["label"].isin(RESOLVED_CODES)].copy()
    test_all = samples[(samples["block"] == TEST_BLOCK)
                       & samples["label"].isin(RESOLVED_CODES)].copy()

    # ---------------- distance asymmetry diagnostic ----------------
    bins = pd.cut(pd.to_numeric(test_all["log_distance_ratio"], errors="coerce"),
                  bins=list(DIST_BIN_EDGES), labels=DIST_BIN_LABELS, right=True)
    tb2 = test_all.copy()
    tb2["bin"] = bins.astype(object)
    rows = []
    for lab in DIST_BIN_LABELS:
        g = tb2[tb2["bin"] == lab]
        if len(g) == 0:
            rows.append(dict(bin=lab, sample="TB2_resolved", n=0,
                             p_up=np.nan, p_down=np.nan))
            continue
        y = g["label"].to_numpy()
        rows.append(dict(bin=lab, sample="TB2_resolved", n=int(len(g)),
                         p_up=float(np.mean(y == UP)),
                         p_down=float(np.mean(y == DOWN))))
    pd.DataFrame(rows).to_csv(OUT / "local0_distance_bins.csv", index=False)

    # ---------------- M. CENSOR sanity ----------------
    print("[M] label distribution by block")
    print(dist.to_string(index=False))
    c_tb = {b: int(((samples["block"] == b)
                    & (samples["label"] == RIGHT_CENSOR)).sum())
            for b in ["TB1", "TB2", "TB3", "TB4"]}
    print(f"[M] RIGHT_CENSOR by block = {c_tb}")
    audit["right_censor_by_block"] = c_tb
    if (c_tb["TB1"] + c_tb["TB2"] + c_tb["TB3"]) == 0 and c_tb["TB4"] > 0:
        audit["timing"] = timing
        audit["blocks"] = boundaries
        audit["status"] = "STOP_LOCAL0_CENSOR_SANITY_FAIL"
        (OUT / "local0_dataset_audit.json").write_text(
            json.dumps(audit, indent=2, default=str))
        (OUT / "local0_summary.json").write_text(json.dumps(dict(
            status="STOP_LOCAL0_CENSOR_SANITY_FAIL",
            reason=("RIGHT_CENSOR appears only in TB4 (end-of-data); no roll "
                    "censoring anywhere in TB1/TB2/TB3. Model NOT trained."),
            right_censor_by_block=c_tb,
            label_distribution_all=dict(
                UP=audit["n_UP"], DOWN=audit["n_DOWN"],
                RIGHT_CENSOR=audit["n_RIGHT_CENSOR"],
                AMBIGUOUS=audit["n_AMBIGUOUS"]),
            blocks=boundaries, timing=timing), indent=2, default=str))
        raise SystemExit(
            "STOP_LOCAL0_CENSOR_SANITY_FAIL: RIGHT_CENSOR appears only in TB4 "
            "(end-of-data); no roll censoring in TB1/TB2/TB3. Model NOT "
            "trained. See local0_expiry_audit.csv "
            "(frozen discontinuity_flags finds 2 discontinuity bars in the "
            "whole 15-symbol panel, both in SC).")

    # ---------------- purge ----------------
    n_before = int(len(train_all))
    keep = to_ns_int(train_all["resolution_time"]) < tb2_start_ns
    train = train_all[keep].copy().reset_index(drop=True)
    purge = dict(
        train_before=n_before, train_after=int(len(train)),
        purged=int(n_before - len(train)),
        purge_rate=float(n_before - len(train)) / max(n_before, 1),
        tb2_start_time=str(pd.Timestamp(tb2_start_ns, unit="ns")),
        max_kept_train_resolution_time=(
            str(pd.Timestamp(int(np.max(to_ns_int(train["resolution_time"]))),
                             unit="ns")) if len(train) else None),
    )
    print("[PURGE] " + ", ".join(f"{k}={v}" for k, v in purge.items()))

    # ---------------- models ----------------
    t0 = time.perf_counter()
    y_tr = train["label"].to_numpy()
    p_up_prior = float(np.mean(y_tr == UP))
    pipe = fit_binary(train)
    timing["fit_seconds"] = round(time.perf_counter() - t0, 2)

    y_te = test_all["label"].to_numpy()
    P0 = np.tile(np.array([p_up_prior, 1.0 - p_up_prior]), (len(y_te), 1))
    P1 = proba_binary(pipe, test_all)
    m0 = evaluate_binary(P0, y_te, "B0_PRIOR")
    m1 = evaluate_binary(P1, y_te, "B1_LOCAL_GEOMETRY")
    pd.DataFrame([m0, m1]).to_csv(OUT / "local0_model_metrics.csv", index=False)
    print("[METRICS]")
    print(pd.DataFrame([m0, m1]).to_string(index=False))

    # ---------------- day-paired bootstrap ----------------
    ll0 = -np.log(np.maximum(P0[np.arange(len(y_te)), y_te], 1e-12))
    ll1 = -np.log(np.maximum(P1[np.arange(len(y_te)), y_te], 1e-12))
    boot = day_paired_bootstrap(test_all["trading_day"].to_numpy(), ll0, ll1)
    day_tbl = pd.DataFrame(dict(
        trading_day=test_all["trading_day"].to_numpy(),
        ll_b0=ll0, ll_b1=ll1)).groupby("trading_day").agg(
        n=("ll_b0", "size"), logloss_b0=("ll_b0", "mean"),
        logloss_b1=("ll_b1", "mean")).reset_index()
    day_tbl["delta_logloss"] = day_tbl["logloss_b1"] - day_tbl["logloss_b0"]
    day_tbl["trading_day"] = day_tbl["trading_day"].astype(str)
    day_tbl.to_csv(OUT / "local0_day_bootstrap.csv", index=False)
    print("[BOOTSTRAP] " + json.dumps(boot))

    # ---------------- leakage guards ----------------
    print("[GUARDS]")
    master_lc = master.copy()
    for col, dtype in (("activation_bar", np.int64), ("penetration_bar", np.int64),
                       ("expiry_bar", np.int64)):
        master_lc[col] = np.zeros(len(master), dtype=dtype)
    for s in symbols:
        idx = np.flatnonzero(master["symbol"].to_numpy(object) == s)
        for col in ("activation_bar", "penetration_bar", "expiry_bar"):
            master_lc.loc[idx, col] = lc_by_sym[s][col]
    if not run_leakage_guards(samples, master_lc, tb2_start_ns, train, test_all):
        raise SystemExit("STOP_LOCAL0_LEAKAGE_GUARD_FAIL")

    # ---------------- outputs ----------------
    timing["total_seconds"] = round(time.perf_counter() - t_total, 2)
    timing["bars_per_sec"] = float(
        audit["n_bars_total"] / max(timing["pair_build_seconds"], 1e-9))
    audit["timing"] = timing
    audit["blocks"] = boundaries
    audit["purge"] = purge
    audit["tb2_start_time"] = purge["tb2_start_time"]
    audit["prior_p_up"] = p_up_prior
    audit["wrong_side"] = dict(all_active=int(ws_total),
                               after_activation_bar=int(ws_after))
    audit["tb3_tb4_rows_in_table"] = int(
        samples["block"].isin(["TB3", "TB4"]).sum())
    (OUT / "local0_dataset_audit.json").write_text(
        json.dumps(audit, indent=2, default=str))

    tb2_vc = test_all["label_name"].value_counts()
    summary = dict(
        experiment="LOCAL-0 local liquidity transition baseline",
        math_object="two competing liquidity transitions + right censoring",
        lifecycle="LOCAL_LIFECYCLE_V2 (bar-index; activation/penetration/expiry)",
        blocks=boundaries,
        train_block=TRAIN_BLOCK, test_block=TEST_BLOCK,
        tb2_start_time=purge["tb2_start_time"],
        primary_sample=dict(
            n_total=int(len(samples)),
            n_tb1=int((samples["block"] == TRAIN_BLOCK).sum()),
            n_tb2=int((samples["block"] == TEST_BLOCK).sum()),
            n_tb3=int((samples["block"] == "TB3").sum()),
            n_tb4=int((samples["block"] == "TB4").sum()),
        ),
        missing_pair=dict(
            missing_upper=audit["n_missing_upper"],
            missing_lower=audit["n_missing_lower"],
            missing_both=audit["n_missing_both"],
            n_pair_both=audit["n_pair_both_total"],
            n_atr_or_close_invalid=audit["n_atr_or_close_invalid"],
        ),
        label_distribution_all=dict(
            UP=audit["n_UP"], DOWN=audit["n_DOWN"],
            RIGHT_CENSOR=audit["n_RIGHT_CENSOR"],
            AMBIGUOUS=audit["n_AMBIGUOUS"],
            ambiguous_rate=audit["ambiguous_rate"],
            right_censor_rate=audit["right_censor_rate"]),
        right_censor_by_block=c_tb,
        label_distribution_tb2_resolved=dict(
            UP=int(tb2_vc.get("UP", 0)), DOWN=int(tb2_vc.get("DOWN", 0))),
        resolution_bars=dict(
            mean=audit["resolution_bars_mean"], p50=audit["resolution_bars_p50"],
            p90=audit["resolution_bars_p90"], p95=audit["resolution_bars_p95"],
            p99=audit["resolution_bars_p99"]),
        purge=purge,
        resolved_train_counts=dict(
            UP=int((y_tr == UP).sum()), DOWN=int((y_tr == DOWN).sum())),
        resolved_test_counts=dict(
            UP=int((y_te == UP).sum()), DOWN=int((y_te == DOWN).sum())),
        wrong_side=dict(all_active=int(ws_total),
                        after_activation_bar=int(ws_after)),
        group_event_conflict="NONE",
        metrics=dict(B0_PRIOR=m0, B1_LOCAL_GEOMETRY=m1),
        bootstrap=boot,
        verdict=boot["verdict"],
        timing=timing,
    )
    (OUT / "local0_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print("\n[SUMMARY] " + json.dumps({
        "blocks": boundaries,
        "n_primary": summary["primary_sample"]["n_total"],
        "label": summary["label_distribution_all"],
        "right_censor_by_block": c_tb,
        "purge": purge, "B0": m0, "B1": m1,
        "bootstrap": boot, "timing": timing}, indent=2, default=str))
    print(f"[DONE] {timing['total_seconds']}s -> {OUT}")


if __name__ == "__main__":
    main()
