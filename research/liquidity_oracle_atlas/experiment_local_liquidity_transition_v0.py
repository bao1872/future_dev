"""LOCAL-0 — Local Liquidity Transition Baseline v0

===========================================================================
研究问题（本轮唯一问题）
===========================================================================
在每个 5m bar **收盘时**，冻结当时最近的 active upper liquidity U 与最近的
active lower liquidity D，问：

    U 与 D 谁先被 penetration？

本轮**不是** SMC 增量实验。禁止加入 BOS / CHoCH / Sweep / OB / FVG /
event sequence / HMM / HSMM / PGM / RL / target selection / execution / PnL。
只建立局部环境 + 最简单 geometry baseline。

===========================================================================
因果合同（继承 frozen lifecycle v1.1，不修改）
===========================================================================
    decision_time           = bar_end = bar_start + 5min
    available_time          = 因果可得时间
    first_penetration_time  = penetration bar 的 END time
    active at t             = available_time <= t
                              AND (first_penetration_time is NaT
                                   OR first_penetration_time > t)

===========================================================================
复用（只读，不修改任何 frozen 旧实验代码）
===========================================================================
    load_raw_bars                  <- run_5m_graph_probability_v1
    load_master / active_mask_at   <- run_5m_graph_probability_v1
    prepare_master_price_groups    <- run_5m_graph_target_decision_v1
    active_prices_chunk            <- run_5m_graph_target_decision_v1

===========================================================================
状态：STOPPED at §7 — STOP_LOCAL0_SAME_PRICE_FP_CONFLICT
===========================================================================
按 §4 字面定义（按 price 分组，不考虑 master 的 `side`）构建 local pair 时，
§7 same-price identity contract 在全 15 品种上失败：

    7,181 / 485,379 primary bars (1.48%) 的 selected price group 内，
    同时存在 >=2 个 active identity 且给出 **不同** 的 finite
    first_penetration_time。

根因（已核实，见 local0_same_price_conflict.csv）：
    master 的 `first_penetration_time` 是 **side-relative** 的：
        side=+1 (HIGH / resistance) -> 价格**向上**突破该 level 的时间
        side=-1 (LOW  / support)    -> 价格**向下**跌破该 level 的时间
    同一个 price 上可以同时存在 side=+1 与 side=-1 的 identity
    （全样本 6,587 / 22,489 个 price cell = 29.3% 同时含两侧）,
    两者的 fp 指向**方向相反的两个不同物理事件**，因此
    “price p 的下一次穿透时间”在不固定 side 时 **无定义**。

    AG 实例（decision_time=2025-03-22T02:25, price=8143, close=8207,
    selected as LOWER）：
        CONFIRMED_SWING_LOW        side=-1  fp=2025-04-07T09:05  (跌破 8143)
        PREV_CONTIG_SESSION_HIGH   side=+1  fp=2025-04-11T23:05  (上破 8143)
    两者在 decision_time 都是 active，finite fp 不一致 -> 冲突。

按 §7 规定：不自行选 min/max 掩盖，不重新解释定义，STOP 并上报。

===========================================================================
输出（只提交小型结果；大型 parquet/cache 不入库）
===========================================================================
    local0_same_price_conflict.csv          <- STOP artifact（冲突细节）
    local0_same_price_conflict_census.csv   <- 全 15 品种冲突规模普查
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
N_BLOCKS = 4
TRAIN_BLOCK = "TB1"
TEST_BLOCK = "TB2"

UP, DOWN, CENSOR, AMBIGUOUS = 0, 1, 2, 3
PRIMARY_CODES = (UP, DOWN, CENSOR)
LABEL_NAMES = np.array(["UP", "DOWN", "CENSOR", "AMBIGUOUS"], dtype=object)

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
    "future_high", "future_low",
}

I64MAX = np.iinfo(np.int64).max
I64MIN = np.iinfo(np.int64).min
INAT = np.datetime64("NaT").astype("datetime64[ns]").view("int64")


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
def to_ns_int(x) -> np.ndarray:
    """任意 datetime64 / Timestamp 数组 -> int64 nanoseconds（NaT -> INAT）。"""
    a = np.asarray(x)
    if a.dtype.kind == "M":
        return a.astype("datetime64[ns]").view("int64")
    return pd.to_datetime(a).to_numpy().astype("datetime64[ns]").view("int64")


def to_dt64_ns(x: np.ndarray) -> np.ndarray:
    """int64 ns -> datetime64[ns]（INAT -> NaT）。"""
    return np.asarray(x, dtype=np.int64).view("datetime64[ns]")


# ---------------------------------------------------------------------------
# Master price-group prep (wraps the frozen helper; unit-normalized to ns)
# ---------------------------------------------------------------------------
def build_price_group_info(master_sym: pd.DataFrame) -> dict:
    base = prepare_master_price_groups(master_sym)
    order = base["identity_order"]
    n_id = len(order)
    starts = base["group_starts"]
    lengths = np.diff(np.r_[starts, n_id]).astype(np.int64)

    av = base["available_time"].astype("datetime64[ns]")
    fp = base["first_penetration_time"].astype("datetime64[ns]")

    info = dict(base)
    info["available_time"] = av
    info["first_penetration_time"] = fp
    info["group_lengths"] = lengths
    info["id_group"] = np.repeat(np.arange(len(starts), dtype=np.int64), lengths)
    info["av_ns"] = av.view("int64")
    info["fp_ns"] = fp.view("int64")
    info["fp_isnat"] = np.isnat(fp)
    info["master_index"] = np.asarray(order, dtype=np.int64)
    info["liquidity_id_sorted"] = master_sym["liquidity_id"].to_numpy(object)[order]
    info["identity_price"] = master_sym["price"].to_numpy(np.float64)[order]
    info["identity_side"] = master_sym["side"].to_numpy(np.int64)[order]
    info["identity_ltype"] = master_sym["liquidity_type"].to_numpy(object)[order]
    return info


# ---------------------------------------------------------------------------
# Core: nearest active upper / lower (chunked, vectorized)
# ---------------------------------------------------------------------------
def nearest_active_pair_chunk(decision_time: np.ndarray, close: np.ndarray,
                              master_info: dict) -> dict:
    """对一个 chunk 的 5m bar close，找 decision_time 当时最近 active upper/lower。

    所有信息必须 decision-time causal。返回 group index（unique_price 的下标）。
    """
    price = master_info["unique_price"]

    active = active_prices_chunk(decision_time, master_info)

    delta = price[None, :] - close[:, None]

    # nearest upper (strictly above close)
    work = np.where(active & (delta > 0.0), delta, np.inf)
    up_idx = np.argmin(work, axis=1)
    up_dist = work[np.arange(len(close)), up_idx]
    has_up = np.isfinite(up_dist)

    # nearest lower (strictly below close)
    work = np.where(active & (delta < 0.0), -delta, np.inf)
    down_idx = np.argmin(work, axis=1)
    down_dist = work[np.arange(len(close)), down_idx]
    has_down = np.isfinite(down_dist)

    upper = np.full(len(close), np.nan, dtype=np.float64)
    lower = np.full(len(close), np.nan, dtype=np.float64)

    upper[has_up] = price[up_idx[has_up]]
    lower[has_down] = price[down_idx[has_down]]
    up_idx = np.where(has_up, up_idx, -1).astype(np.int64)
    down_idx = np.where(has_down, down_idx, -1).astype(np.int64)

    return {
        "upper_price": upper,
        "lower_price": lower,
        "upper_group": up_idx,
        "lower_group": down_idx,
        "has_upper": has_up,
        "has_lower": has_down,
    }


def resolve_group_fp(dt_ns: np.ndarray, group_idx: np.ndarray,
                     master_info: dict) -> dict:
    """给定 (decision_time, selected price group)，返回该 group 的 next penetration。

    只看当时 **active** 的 identity：
        active = available_time <= dt AND (fp is NaT OR fp > dt)

    若 group 内 active 且 finite 的 fp 出现两个不同值 -> conflict=True
    （禁止自行选 min/max 掩盖）。

    Returns dict with:
        fp          int64 ns 的 next penetration（无 finite -> INAT）
        n_active    group 内 active identity 数
        n_finite    group 内 active 且 fp finite 的 identity 数
        has_finite
        mixed       active 中同时存在 NaT 与 finite（允许，但需计数）
        conflict    同一 price 的 active identity 给出不一致 finite fp
    """
    gid = master_info["id_group"]
    av = master_info["av_ns"]
    fp = master_info["fp_ns"]
    isnat = master_info["fp_isnat"]

    sel = gid[None, :] == group_idx[:, None]
    act = sel & (av[None, :] <= dt_ns[:, None]) & (
        isnat[None, :] | (fp[None, :] > dt_ns[:, None])
    )
    fin = act & (~isnat)[None, :]

    n_active = act.sum(axis=1).astype(np.int64)
    n_finite = fin.sum(axis=1).astype(np.int64)

    fpmin = np.min(np.where(fin, fp[None, :], I64MAX), axis=1)
    fpmax = np.max(np.where(fin, fp[None, :], I64MIN), axis=1)

    has_finite = n_finite > 0
    out_fp = np.where(has_finite, fpmin, INAT)
    conflict = has_finite & (fpmin != fpmax)
    mixed = has_finite & (n_finite < n_active)
    return dict(fp=out_fp, n_active=n_active, n_finite=n_finite,
                has_finite=has_finite, mixed=mixed, conflict=conflict)


# ---------------------------------------------------------------------------
# Outcome classification (no future scan; uses frozen first_penetration_time)
# ---------------------------------------------------------------------------
def classify_pair_time(t_up, t_down) -> np.ndarray:
    """UP=0 / DOWN=1 / CENSOR=2 / AMBIGUOUS=3。"""
    t_up = np.atleast_1d(np.asarray(t_up, dtype="datetime64[ns]"))
    t_down = np.atleast_1d(np.asarray(t_down, dtype="datetime64[ns]"))

    up_missing = np.isnat(t_up)
    down_missing = np.isnat(t_down)

    out = np.full(t_up.shape, CENSOR, dtype=np.int64)
    out[down_missing & ~up_missing] = UP
    out[up_missing & ~down_missing] = DOWN

    both = ~up_missing & ~down_missing
    out[both & (t_up < t_down)] = UP
    out[both & (t_down < t_up)] = DOWN
    out[both & (t_up == t_down)] = AMBIGUOUS
    return out


# ---------------------------------------------------------------------------
# Time blocks (global, over FULL_UNIV trading days)
# ---------------------------------------------------------------------------
def build_blocks(bars_by_symbol: dict):
    """全部品种 raw 5m 的 global unique trading_day 排序后四等分。"""
    day_arrays = [
        np.asarray(b["td"]).astype("datetime64[D]") for b in bars_by_symbol.values()
    ]
    all_days = np.unique(np.concatenate(day_arrays))
    chunks = np.array_split(all_days, N_BLOCKS)
    day_block_code = np.repeat(
        np.arange(N_BLOCKS, dtype=np.int64), [len(c) for c in chunks]
    )
    boundaries = []
    for i, ch in enumerate(chunks):
        boundaries.append(dict(
            block=f"TB{i + 1}",
            first_day=str(ch[0]),
            last_day=str(ch[-1]),
            n_days=int(len(ch)),
        ))
    return all_days, day_block_code, boundaries


def block_codes_for(bars: dict, all_days: np.ndarray,
                    day_block_code: np.ndarray) -> np.ndarray:
    td = np.asarray(bars["td"]).astype("datetime64[D]")
    return day_block_code[np.searchsorted(all_days, td)]


# ---------------------------------------------------------------------------
# Contiguous-segment end time (discontinuity / end-of-data)
# ---------------------------------------------------------------------------
def segment_end_ns(bars: dict) -> np.ndarray:
    """每根 bar 所属连续段的最后一根 bar 的 END time（int64 ns）。"""
    n = int(bars["n"])
    t_ns = to_ns_int(bars["t"])
    disc = np.asarray(bars["disc"], dtype=bool)
    seg_id = np.cumsum(disc.astype(np.int64))
    _, first_idx = np.unique(seg_id, return_index=True)
    last_idx = np.r_[first_idx[1:], n] - 1
    lengths = np.diff(np.r_[first_idx, n])
    per_bar_last = np.repeat(last_idx, lengths)
    return t_ns[per_bar_last] + BAR_NS


# ---------------------------------------------------------------------------
# Conflict detail (STOP artifact)
# ---------------------------------------------------------------------------
def collect_conflict_rows(sym, master_info: dict, dt_ns: np.ndarray,
                          group_idx: np.ndarray, cap: int = 500):
    av = master_info["av_ns"]
    fp = master_info["fp_ns"]
    isnat = master_info["fp_isnat"]
    gid = master_info["id_group"]
    lid = master_info["liquidity_id_sorted"]
    px = master_info["identity_price"]
    side = master_info["identity_side"]
    ltype = master_info["identity_ltype"]

    rows = []
    for dtv, g in zip(dt_ns, group_idx):
        ids = np.flatnonzero(gid == int(g))
        for i in ids:
            active = bool(av[i] <= dtv) and bool(isnat[i] or fp[i] > dtv)
            rows.append(dict(
                symbol=sym,
                decision_time=str(to_dt64_ns(np.int64(dtv))),
                price=float(px[i]),
                side=int(side[i]),
                liquidity_type=str(ltype[i]),
                liquidity_id=str(lid[i]),
                available_time=str(to_dt64_ns(np.int64(av[i]))),
                first_penetration_time=(
                    "" if isnat[i] else str(to_dt64_ns(np.int64(fp[i])))
                ),
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
                         conflict_cap: int = 500,
                         stop_on_conflict: bool = True):
    info = build_price_group_info(master_sym)
    n = int(bars["n"])
    t_ns = to_ns_int(bars["t"])
    dt_ns = t_ns + BAR_NS
    close = np.asarray(bars["c"], dtype=np.float64)
    atr = np.asarray(bars["atr"], dtype=np.float64)
    seg_end = segment_end_ns(bars)
    codes = block_codes_for(bars, all_days, day_block_code)

    up_px = np.full(n, np.nan, dtype=np.float64)
    dn_px = np.full(n, np.nan, dtype=np.float64)
    up_g = np.full(n, -1, dtype=np.int64)
    dn_g = np.full(n, -1, dtype=np.int64)
    up_fp = np.full(n, INAT, dtype=np.int64)
    dn_fp = np.full(n, INAT, dtype=np.int64)
    up_na = np.zeros(n, dtype=np.int64)
    dn_na = np.zeros(n, dtype=np.int64)
    up_nf = np.zeros(n, dtype=np.int64)
    dn_nf = np.zeros(n, dtype=np.int64)
    conflict_mask = np.zeros(n, dtype=bool)

    for lo in range(0, n, CHUNK_BARS):
        hi = min(lo + CHUNK_BARS, n)
        sl = slice(lo, hi)
        dtc = to_dt64_ns(dt_ns[sl])
        pair = nearest_active_pair_chunk(dtc, close[sl], info)
        up_px[sl] = pair["upper_price"]
        dn_px[sl] = pair["lower_price"]
        up_g[sl] = pair["upper_group"]
        dn_g[sl] = pair["lower_group"]

        # ---- resolve next penetration time for the selected groups ----
        iu = np.flatnonzero(pair["has_upper"])
        if len(iu):
            res = resolve_group_fp(dt_ns[sl][iu], pair["upper_group"][iu], info)
            up_fp[lo + iu] = res["fp"]
            up_na[lo + iu] = res["n_active"]
            up_nf[lo + iu] = res["n_finite"]
            conflict_mask[lo + iu[res["conflict"]]] = True
        idn = np.flatnonzero(pair["has_lower"])
        if len(idn):
            res = resolve_group_fp(dt_ns[sl][idn], pair["lower_group"][idn], info)
            dn_fp[lo + idn] = res["fp"]
            dn_na[lo + idn] = res["n_active"]
            dn_nf[lo + idn] = res["n_finite"]
            conflict_mask[lo + idn[res["conflict"]]] = True

        if conflict_mask[sl].any():
            n_bad = int(conflict_mask[sl].sum())
            if not stop_on_conflict:
                continue
            bad = np.flatnonzero(conflict_mask[sl])
            rows = []
            for b in bad:
                # report BOTH selected sides: the conflict may sit on either one
                for side_name, gsel in (("upper", up_g[lo + b]),
                                        ("lower", dn_g[lo + b])):
                    if gsel < 0:
                        continue
                    extra = collect_conflict_rows(
                        sym, info, np.array([dt_ns[lo + b]]), np.array([gsel]),
                        cap=conflict_cap,
                    )
                    for r in extra:
                        r["selected_side"] = side_name
                    rows += extra
                if len(rows) >= conflict_cap:
                    break
            pd.DataFrame(rows).to_csv(OUT / "local0_same_price_conflict.csv",
                                      index=False)
            raise SystemExit(
                "STOP_LOCAL0_SAME_PRICE_FP_CONFLICT: same-price active "
                f"identities disagree on next first_penetration_time "
                f"(symbol={sym}, n_conflict_bars={int(conflict_mask.sum())}); "
                f"details -> {OUT / 'local0_same_price_conflict.csv'}"
            )

    has_up = up_g >= 0
    has_dn = dn_g >= 0
    both = has_up & has_dn
    atr_ok = np.isfinite(atr) & (atr > 0.0) & np.isfinite(close)
    primary = both & atr_ok

    # ---- hard causal guards (structural) ----
    if primary.any():
        assert bool((up_px[primary] > close[primary]).all()), "UPPER_NOT_ABOVE_CLOSE"
        assert bool((dn_px[primary] < close[primary]).all()), "LOWER_NOT_BELOW_CLOSE"
        assert bool(
            ((up_fp[primary] > dt_ns[primary]) | (up_fp[primary] == INAT)).all()
        ), "UPPER_FP_NOT_FUTURE"
        assert bool(
            ((dn_fp[primary] > dt_ns[primary]) | (dn_fp[primary] == INAT)).all()
        ), "LOWER_FP_NOT_FUTURE"

    label = np.full(n, -1, dtype=np.int64)
    res_ns = np.full(n, INAT, dtype=np.int64)
    if both.any():
        lab = classify_pair_time(to_dt64_ns(up_fp[both]), to_dt64_ns(dn_fp[both]))
        label[both] = lab
        is_censor = lab == CENSOR
        is_down = lab == DOWN
        res_ns[both] = np.where(
            is_censor, seg_end[both], np.where(is_down, dn_fp[both], up_fp[both])
        )

    idx = np.flatnonzero(primary)
    up_dist = np.full(n, np.nan, dtype=np.float64)
    dn_dist = np.full(n, np.nan, dtype=np.float64)
    up_dist[idx] = (up_px[idx] - close[idx]) / atr[idx]
    dn_dist[idx] = (close[idx] - dn_px[idx]) / atr[idx]

    df = pd.DataFrame(dict(
        symbol=sym,
        decision_time=to_dt64_ns(dt_ns[idx]),
        trading_day=np.asarray(bars["td"]).astype("datetime64[D]")[idx],
        block=np.array([f"TB{c + 1}" for c in codes[idx]], dtype=object),
        close=close[idx],
        atr=atr[idx],
        upper_price=up_px[idx],
        lower_price=dn_px[idx],
        up_distance_R=up_dist[idx],
        down_distance_R=dn_dist[idx],
        width_R=up_dist[idx] + dn_dist[idx],
        log_distance_ratio=np.log(
            (up_dist[idx] + 1e-8) / (dn_dist[idx] + 1e-8)
        ),
        label=label[idx],
        resolution_time=to_dt64_ns(res_ns[idx]),
        resolution_bars=(res_ns[idx] - dt_ns[idx]).astype(np.float64)
        / float(BAR_NS),
        up_n_active=up_na[idx],
        down_n_active=dn_na[idx],
        up_n_finite=up_nf[idx],
        down_n_finite=dn_nf[idx],
        # OUTCOME-only audit fields. NEVER model features (see FORBIDDEN_FEATURES).
        up_fp_time=to_dt64_ns(up_fp[idx]),
        dn_fp_time=to_dt64_ns(dn_fp[idx]),
    ))
    df["mixed_fp_pattern"] = (
        ((df["up_n_finite"] > 0) & (df["up_n_finite"] < df["up_n_active"]))
        | ((df["down_n_finite"] > 0) & (df["down_n_finite"] < df["down_n_active"]))
    ).astype(int)

    audit = dict(
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
            ((df["up_n_active"] > 1) | (df["down_n_active"] > 1)).sum()
        ),
    )
    return df, audit


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def multiclass_logloss(P: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.maximum(P[np.arange(len(y)), y], 1e-12))))


def multiclass_brier(P: np.ndarray, y: np.ndarray, n_classes: int = 3) -> float:
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def ece_binary(p: np.ndarray, y: np.ndarray, n_bins: int = N_ECE_BINS) -> float:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(p) == 0:
        return float("nan")
    b = np.clip(np.floor(p * n_bins).astype(int), 0, n_bins - 1)
    tot = 0.0
    n = float(len(p))
    for k in range(n_bins):
        m = b == k
        if not m.any():
            continue
        tot += (m.sum() / n) * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(tot)


def fit_b1(train: pd.DataFrame) -> Pipeline:
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=3000)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(train[NUM_FEATURES + CAT_FEATURES], train["label"].to_numpy())
    classes = pipe.named_steps["clf"].classes_
    assert set(np.asarray(classes).tolist()) <= set(PRIMARY_CODES), \
        f"CLASSES_NOT_SUBSET_OF_UP_DOWN_CENSOR: {classes}"
    return pipe


def proba_matrix(pipe, df: pd.DataFrame) -> np.ndarray:
    """Return a FULL 3-column matrix in fixed [UP, DOWN, CENSOR] order.

    If a class is absent from the training sample (e.g. CENSOR after the
    mandated whole-outcome purge), its column stays exactly 0.0 — the model
    simply has no mass on it. Nothing is imputed or re-weighted.
    """
    P = pipe.predict_proba(df[pipe.feature_names_in_])
    classes = np.asarray(pipe.named_steps["clf"].classes_, dtype=int)
    out = np.zeros((len(df), 3), dtype=np.float64)
    out[:, classes] = P
    return out


def evaluate(P: np.ndarray, y: np.ndarray, name: str) -> dict:
    return dict(
        model=name,
        n=int(len(y)),
        logloss=multiclass_logloss(P, y),
        brier=multiclass_brier(P, y),
        ece_up=ece_binary(P[:, UP], (y == UP).astype(float)),
        ece_down=ece_binary(P[:, DOWN], (y == DOWN).astype(float)),
        ece_censor=ece_binary(P[:, CENSOR], (y == CENSOR).astype(float)),
        up_pct=float(np.mean(y == UP)),
        down_pct=float(np.mean(y == DOWN)),
        censor_pct=float(np.mean(y == CENSOR)),
    )


def day_paired_bootstrap(test: pd.DataFrame, ll_b0: np.ndarray,
                         ll_b1: np.ndarray, seed: int = SEED,
                         n_boot: int = N_BOOT) -> dict:
    d = test["trading_day"].to_numpy()
    days = np.unique(d)
    g0 = pd.Series(ll_b0).groupby(d).mean()
    g1 = pd.Series(ll_b1).groupby(d).mean()
    delta_day = (g1 - g0).to_numpy()
    rng = np.random.default_rng(seed)
    n = len(delta_day)
    reps = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        pick = rng.integers(0, n, n)
        reps[i] = float(delta_day[pick].mean())
    lo = float(np.percentile(reps, 2.5))
    hi = float(np.percentile(reps, 97.5))
    if hi < 0.0:
        verdict = "LOCAL_GEOMETRY_INCREMENT_SUPPORTED"
    elif lo > 0.0:
        verdict = "LOCAL_GEOMETRY_INCREMENT_NEGATIVE"
    else:
        verdict = "LOCAL_GEOMETRY_INCREMENT_AMBIGUOUS"
    return dict(
        n_days=int(n),
        n_boot=int(n_boot),
        seed=int(seed),
        mean_delta_logloss=float(delta_day.mean()),
        ci_lo=lo,
        ci_hi=hi,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Hard leakage guards (independent code path where possible)
# ---------------------------------------------------------------------------
def run_leakage_guards(samples: pd.DataFrame, master: pd.DataFrame,
                       tb2_start_ns: int, train: pd.DataFrame,
                       test: pd.DataFrame, n_guard_rows: int = 3000):
    """17 条 hard leakage guard 的可执行部分。

    1/2/3/4 通过 **独立代码路径** 复核：直接用 frozen `active_mask_at` +
    pandas 原始 master 表重算 active 集合，而不是复用 builder 的中间数组。
    """
    msgs = []
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        msgs.append(f"[{'PASS' if cond else 'FAIL'}] {name}"
                    + (f" :: {detail}" if detail else ""))

    # ---- G8: TB3/TB4 never enter fit / metric / bootstrap ----
    check("G8a train block is TB1 only", set(train["block"].unique()) <= {TRAIN_BLOCK},
          str(sorted(train["block"].unique())))
    check("G8b test block is TB2 only", set(test["block"].unique()) <= {TEST_BLOCK},
          str(sorted(test["block"].unique())))
    tb34 = int(samples["block"].isin(["TB3", "TB4"]).sum())
    check("G8c TB3/TB4 rows exist in table but unused", tb34 >= 0,
          f"n_tb34_in_table={tb34}")

    # ---- G7: purge ----
    tr_res = to_ns_int(train["resolution_time"])
    check("G7 kept-train resolution_time < TB2 start",
          bool((tr_res < tb2_start_ns).all()),
          f"max_train_res={pd.Timestamp(max(tr_res) if len(tr_res) else 0, unit='ns')} "
          f"tb2_start={pd.Timestamp(tb2_start_ns, unit='ns')}")

    # ---- G6: no future fields in model features ----
    used = set(NUM_FEATURES) | set(CAT_FEATURES)
    check("G6 no forbidden feature", not (used & FORBIDDEN_FEATURES),
          str(sorted(used & FORBIDDEN_FEATURES)))
    check("G5 features are decision-time-only (declared set)",
          used == {"up_distance_R", "down_distance_R", "width_R",
                   "log_distance_ratio", "symbol"}, str(sorted(used)))

    # ---- G1..G4: independent recompute on a deterministic subsample ----
    s = samples.reset_index(drop=True)
    if len(s) > n_guard_rows:
        pick = np.linspace(0, len(s) - 1, n_guard_rows).astype(int)
        sub = s.iloc[pick]
    else:
        sub = s

    g1 = g2 = g3 = g4 = True
    g_fp = g_conflict = True
    bad_detail = ""
    for sym, g in sub.groupby("symbol"):
        ms = master[master["symbol"] == sym]
        av = pd.to_datetime(ms["available_time"]).to_numpy()
        fp = pd.to_datetime(ms["first_penetration_time"]).to_numpy()
        px = ms["price"].to_numpy(np.float64)
        for _, r in g.iterrows():
            dt = np.datetime64(pd.Timestamp(r["decision_time"]).to_pydatetime(),
                               "ns")
            close = float(r["close"])
            am = active_mask_at(av, fp, dt)
            act_px = px[am]
            up_c = act_px[act_px > close]
            dn_c = act_px[act_px < close]
            if len(up_c) == 0 or len(dn_c) == 0:
                g3 = False
                bad_detail = f"{sym} no active pair at {dt}"
                break
            if not np.isclose(float(up_c.min()), float(r["upper_price"])):
                g3 = False
                bad_detail = f"{sym} {dt} upper {up_c.min()} != {r['upper_price']}"
                break
            if not np.isclose(float(dn_c.max()), float(r["lower_price"])):
                g4 = False
                bad_detail = f"{sym} {dt} lower {dn_c.max()} != {r['lower_price']}"
                break
            for side_price, is_up in ((float(r["upper_price"]), True),
                                      (float(r["lower_price"]), False)):
                m = am & (px == side_price)
                if not m.any():
                    g1 = False
                    bad_detail = f"{sym} {dt} no active identity at {side_price}"
                    break
                if not bool((av[m] <= dt).all()):
                    g1 = False
                    bad_detail = f"{sym} {dt} available_time > decision_time"
                    break
                if not bool((np.isnat(fp[m]) | (fp[m] > dt)).all()):
                    g2 = False
                    bad_detail = f"{sym} {dt} fp <= decision_time (consumed)"
                    break
                fps = fp[m]
                fps = fps[~np.isnat(fps)]
                if len(fps) > 1 and len(np.unique(fps)) > 1:
                    g_conflict = False
                    bad_detail = f"{sym} {dt} same-price fp conflict at {side_price}"
                    break
                if len(fps) == 1:
                    expect = (r["up_fp_time"] if is_up else r["dn_fp_time"])
                    if pd.isna(expect) or np.datetime64(
                            pd.Timestamp(expect).to_pydatetime(), "ns") != fps[0]:
                        g_fp = False
                        bad_detail = (f"{sym} {dt} fp mismatch {fps[0]} vs {expect}")
                        break
            if not (g1 and g2 and g3 and g4 and g_fp and g_conflict):
                break
        if not (g1 and g2 and g3 and g4 and g_fp and g_conflict):
            break

    check("G1 available_time <= decision_time (independent recompute)", g1, bad_detail)
    check("G2 fp NaT or > decision_time (independent recompute)", g2, bad_detail)
    check("G3 upper is nearest active price > close", g3, bad_detail)
    check("G4 lower is nearest active price < close", g4, bad_detail)
    check("G1b resolved fp equals active identity fp", g_fp, bad_detail)
    check("G7b same-price active fp consistent", g_conflict, bad_detail)

    for m in msgs:
        print("  " + m)
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--mode", default="run", choices=["run", "census"])
    ap.add_argument("--reuse-cache", action="store_true")
    ap.add_argument("--force-rebuild", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    t_total = time.perf_counter()
    timing = {}

    symbols = args.symbols or list(FULL_UNIV)
    print(f"[LOCAL-0] symbols={symbols}")

    # ---------------- load ----------------
    t0 = time.perf_counter()
    master = load_master()
    ms_by_sym = {s: master[master["symbol"] == s].copy() for s in symbols}
    bars_by_sym = {}
    for s in symbols:
        bars_by_sym[s] = load_raw_bars(s)
    timing["load_seconds"] = round(time.perf_counter() - t0, 2)

    all_days, day_block_code, boundaries = build_blocks(bars_by_sym)
    print("[BLOCKS] " + "; ".join(
        f"{b['block']}=[{b['first_day']}..{b['last_day']}]({b['n_days']}d)"
        for b in boundaries))

    # ---------------- pair build + label ----------------
    t0 = time.perf_counter()
    frames = []
    audits = []
    info_by_sym = {}
    for s in symbols:
        cp = CACHE / f"local0_samples_{s}.parquet"
        if args.reuse_cache and cp.exists() and not args.force_rebuild:
            df = pd.read_parquet(cp)
            print(f"[CACHE] reuse {cp.name} rows={len(df)}")
        else:
            t1 = time.perf_counter()
            info = build_price_group_info(ms_by_sym[s])
            info_by_sym[s] = info
            df, au = build_symbol_samples(
                s, ms_by_sym[s], bars_by_sym[s], all_days, day_block_code,
                stop_on_conflict=(args.mode == "run"))
            au["symbol"] = s
            audits.append(au)
            df.to_parquet(cp, index=False)
            print(f"[BUILD] {s} bars={au['n_bars']} primary={au['n_primary']} "
                  f"conflicts={au['n_conflict_bars']} "
                  f"({time.perf_counter()-t1:.1f}s)")
        frames.append(df)
    samples = pd.concat(frames, ignore_index=True)
    timing["pair_build_seconds"] = round(time.perf_counter() - t0, 2)

    # ---------------- conflict census (diagnostic; does NOT stop) ----------
    if args.mode == "census":
        rows = []
        for a in audits:
            m = ms_by_sym[a["symbol"]]
            cell = m.groupby("price")["side"].nunique()
            rows.append(dict(
                symbol=a["symbol"],
                n_identities=int(len(m)),
                n_price_cells=int(len(cell)),
                n_price_cells_both_sides=int((cell > 1).sum()),
                n_bars=a["n_bars"],
                n_primary=a["n_primary"],
                n_conflict_bars=a["n_conflict_bars"],
                conflict_rate=float(a["n_conflict_bars"]) / max(a["n_primary"], 1),
            ))
        cen = pd.DataFrame(rows)
        cen.to_csv(OUT / "local0_same_price_conflict_census.csv", index=False)
        print(cen.to_string(index=False))
        print(f"[CENSUS] total_conflict_bars={int(cen['n_conflict_bars'].sum())} "
              f"of primary={int(cen['n_primary'].sum())}")
        print("STOP LOCAL0_SAME_PRICE_FP_CONFLICT (see "
              "local0_same_price_conflict_census.csv)")
        return

    # ---------------- label / audit ----------------
    t0 = time.perf_counter()
    samples["label_name"] = samples["label"].map(
        {UP: "UP", DOWN: "DOWN", CENSOR: "CENSOR", AMBIGUOUS: "AMBIGUOUS"})

    audit = {}
    if audits:
        audit["per_symbol_raw"] = audits
    audit["n_bars_total"] = int(
        sum(int(bars_by_sym[s]["n"]) for s in symbols))
    audit["n_primary_total"] = int(len(samples))
    audit["n_pair_both_total"] = int(
        sum(a["n_pair_both"] for a in audits)) if audits else None
    audit["n_missing_upper"] = int(
        sum(a["n_missing_upper"] for a in audits)) if audits else None
    audit["n_missing_lower"] = int(
        sum(a["n_missing_lower"] for a in audits)) if audits else None
    audit["n_missing_both"] = int(
        sum(a["n_missing_both"] for a in audits)) if audits else None
    audit["n_atr_or_close_invalid"] = int(
        sum(a["n_atr_or_close_invalid"] for a in audits)) if audits else None
    audit["n_mixed_fp_pattern"] = int(samples["mixed_fp_pattern"].sum())
    audit["n_multi_identity_selected"] = int(
        ((samples["up_n_active"] > 1) | (samples["down_n_active"] > 1)).sum())

    vc = samples["label_name"].value_counts()
    for k in ["UP", "DOWN", "CENSOR", "AMBIGUOUS"]:
        audit[f"n_{k}"] = int(vc.get(k, 0))
    audit["ambiguous_rate"] = float(vc.get("AMBIGUOUS", 0) / max(len(samples), 1))

    rb = samples["resolution_bars"].to_numpy(float)
    for q, name in [(0.5, "p50"), (0.9, "p90"), (0.95, "p95"), (0.99, "p99")]:
        audit[f"resolution_bars_{name}"] = float(np.nanpercentile(rb, q * 100))
    audit["resolution_bars_mean"] = float(np.nanmean(rb))
    for lbl in ["UP", "DOWN", "CENSOR", "AMBIGUOUS"]:
        m = (samples["label_name"] == lbl).to_numpy()
        if m.any():
            audit[f"resolution_bars_mean_{lbl}"] = float(np.nanmean(rb[m]))
    for col in ["up_distance_R", "down_distance_R", "width_R"]:
        v = samples[col].to_numpy(float)
        audit[f"{col}_mean"] = float(np.nanmean(v))
        for q, name in [(0.5, "p50"), (0.9, "p90"), (0.99, "p99")]:
            audit[f"{col}_{name}"] = float(np.nanpercentile(v, q * 100))

    # ---- label distribution csv ----
    dist = (samples.groupby(["block", "label_name"]).size()
            .unstack(fill_value=0).reset_index())
    dist["total"] = dist.drop(columns=["block"]).sum(axis=1)
    dist.to_csv(OUT / "local0_label_distribution.csv", index=False)

    # ---- by symbol ----
    bys = (samples.groupby(["symbol", "label_name"]).size()
           .unstack(fill_value=0))
    for k in ["UP", "DOWN", "CENSOR", "AMBIGUOUS"]:
        if k not in bys:
            bys[k] = 0
    bys["n"] = bys[["UP", "DOWN", "CENSOR", "AMBIGUOUS"]].sum(axis=1)
    for k in ["UP", "DOWN", "CENSOR", "AMBIGUOUS"]:
        bys[f"{k}_pct"] = bys[k] / bys["n"]
    bys = bys.reset_index()[
        ["symbol", "n", "UP", "DOWN", "CENSOR", "AMBIGUOUS",
         "UP_pct", "DOWN_pct", "CENSOR_pct", "AMBIGUOUS_pct"]]
    bys.to_csv(OUT / "local0_by_symbol.csv", index=False)
    timing["label_seconds"] = round(time.perf_counter() - t0, 2)

    # ---------------- train / test / purge ----------------
    tb2_mask = samples["block"] == TEST_BLOCK
    tb2_start_ns = int(np.min(to_ns_int(samples.loc[tb2_mask, "decision_time"])))

    train_all = samples[samples["block"] == TRAIN_BLOCK].copy()
    test_all = samples[samples["block"] == TEST_BLOCK].copy()

    train_all = train_all[train_all["label"].isin(PRIMARY_CODES)]
    test_all = test_all[test_all["label"].isin(PRIMARY_CODES)]

    n_before = int(len(train_all))
    keep = to_ns_int(train_all["resolution_time"]) < tb2_start_ns
    train = train_all[keep].copy().reset_index(drop=True)
    n_after = int(len(train))
    purge = dict(
        train_before=n_before,
        train_after=n_after,
        purged=int(n_before - n_after),
        purge_rate=float(n_before - n_after) / max(n_before, 1),
        tb2_start_time=str(pd.Timestamp(tb2_start_ns, unit="ns")),
        max_kept_train_resolution_time=(
            str(pd.Timestamp(int(np.max(to_ns_int(train["resolution_time"]))),
                             unit="ns")) if n_after else None),
    )
    print("[PURGE] " + ", ".join(f"{k}={v}" for k, v in purge.items()))

    # ---------------- distance asymmetry diagnostic (TB2 primary) ----------
    bins = pd.cut(
        pd.to_numeric(test_all["log_distance_ratio"], errors="coerce"),
        bins=list(DIST_BIN_EDGES), labels=DIST_BIN_LABELS, right=True,
    )
    tb2 = test_all.copy()
    tb2["bin"] = bins.astype(object)
    rows = []
    for lab in DIST_BIN_LABELS:
        g = tb2[tb2["bin"] == lab]
        if len(g) == 0:
            rows.append(dict(bin=lab, sample="TB2_primary", n=0,
                             p_up=np.nan, p_down=np.nan, p_censor=np.nan))
            continue
        y = g["label"].to_numpy()
        rows.append(dict(bin=lab, sample="TB2_primary", n=int(len(g)),
                         p_up=float(np.mean(y == UP)),
                         p_down=float(np.mean(y == DOWN)),
                         p_censor=float(np.mean(y == CENSOR))))
    pd.DataFrame(rows).to_csv(OUT / "local0_distance_bins.csv", index=False)

    # ---------------- models ----------------
    t0 = time.perf_counter()
    y_tr = train["label"].to_numpy()
    train_label_counts = {LABEL_NAMES[c]: int((y_tr == c).sum()) for c in PRIMARY_CODES}
    train_class_absent = [LABEL_NAMES[c] for c in PRIMARY_CODES
                          if int((y_tr == c).sum()) == 0]
    print(f"[TRAIN] label_counts={train_label_counts} "
          f"absent_after_purge={train_class_absent}")
    prior = np.array([np.mean(y_tr == c) for c in PRIMARY_CODES], dtype=float)
    pipe = fit_b1(train)
    timing["fit_seconds"] = round(time.perf_counter() - t0, 2)

    y_te = test_all["label"].to_numpy()
    P0 = np.tile(prior, (len(y_te), 1))
    P1 = proba_matrix(pipe, test_all)

    m0 = evaluate(P0, y_te, "B0_PRIOR")
    m1 = evaluate(P1, y_te, "B1_LOCAL_GEOMETRY")
    pd.DataFrame([m0, m1]).to_csv(OUT / "local0_model_metrics.csv", index=False)
    print("[METRICS]")
    print(pd.DataFrame([m0, m1]).to_string(index=False))

    # ---------------- day-paired bootstrap ----------------
    ll0 = -np.log(np.maximum(P0[np.arange(len(y_te)), y_te], 1e-12))
    ll1 = -np.log(np.maximum(P1[np.arange(len(y_te)), y_te], 1e-12))
    boot = day_paired_bootstrap(test_all.reset_index(drop=True), ll0, ll1)

    day_tbl = pd.DataFrame(dict(
        trading_day=test_all["trading_day"].to_numpy(),
        ll_b0=ll0, ll_b1=ll1,
    )).groupby("trading_day").agg(
        n=("ll_b0", "size"), logloss_b0=("ll_b0", "mean"),
        logloss_b1=("ll_b1", "mean"),
    ).reset_index()
    day_tbl["delta_logloss"] = day_tbl["logloss_b1"] - day_tbl["logloss_b0"]
    day_tbl["trading_day"] = day_tbl["trading_day"].astype(str)
    day_tbl.to_csv(OUT / "local0_day_bootstrap.csv", index=False)
    print("[BOOTSTRAP] " + json.dumps(boot))

    # ---------------- leakage guards ----------------
    print("[GUARDS]")
    guard_ok = run_leakage_guards(samples, master, tb2_start_ns, train, test_all)
    if not guard_ok:
        raise SystemExit("STOP_LOCAL0_LEAKAGE_GUARD_FAIL")

    # ---------------- outputs ----------------
    timing["total_seconds"] = round(time.perf_counter() - t_total, 2)
    timing["bars_per_sec"] = float(
        audit["n_bars_total"] / max(timing["pair_build_seconds"], 1e-9))
    audit["timing"] = timing
    audit["blocks"] = boundaries
    audit["purge"] = purge
    audit["tb2_start_time"] = purge["tb2_start_time"]
    audit["train_prior"] = dict(
        zip(["UP", "DOWN", "CENSOR"], [float(x) for x in prior]))
    audit["train_label_counts_after_purge"] = train_label_counts
    audit["train_class_absent_after_purge"] = train_class_absent
    audit["tb3_tb4_rows_in_table"] = int(
        samples["block"].isin(["TB3", "TB4"]).sum())
    (OUT / "local0_dataset_audit.json").write_text(
        json.dumps(audit, indent=2, default=str))

    tb2_vc = test_all["label_name"].value_counts()
    summary = dict(
        experiment="LOCAL-0 local liquidity transition baseline",
        question=("At each 5m bar close, which of the nearest active upper / "
                  "lower liquidity is penetrated first?"),
        blocks=boundaries,
        train_block=TRAIN_BLOCK,
        test_block=TEST_BLOCK,
        tb2_start_time=purge["tb2_start_time"],
        primary_sample=dict(
            n_total=int(len(samples)),
            n_tb1_primary=int((samples["block"] == TRAIN_BLOCK).sum()),
            n_tb2_primary=int((samples["block"] == TEST_BLOCK).sum()),
            n_tb3_primary=int((samples["block"] == "TB3").sum()),
            n_tb4_primary=int((samples["block"] == "TB4").sum()),
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
            CENSOR=audit["n_CENSOR"], AMBIGUOUS=audit["n_AMBIGUOUS"],
            ambiguous_rate=audit["ambiguous_rate"],
        ),
        label_distribution_tb2=dict(
            UP=int(tb2_vc.get("UP", 0)), DOWN=int(tb2_vc.get("DOWN", 0)),
            CENSOR=int(tb2_vc.get("CENSOR", 0)),
            AMBIGUOUS=int(tb2_vc.get("AMBIGUOUS", 0)),
        ),
        resolution_bars=dict(
            mean=audit["resolution_bars_mean"], p50=audit["resolution_bars_p50"],
            p90=audit["resolution_bars_p90"], p95=audit["resolution_bars_p95"],
            p99=audit["resolution_bars_p99"],
        ),
        purge=purge,
        train_label_counts_after_purge=train_label_counts,
        train_class_absent_after_purge=train_class_absent,
        metrics=dict(B0_PRIOR=m0, B1_LOCAL_GEOMETRY=m1),
        bootstrap=boot,
        verdict=boot["verdict"],
        same_price=dict(
            n_multi_identity_selected=audit["n_multi_identity_selected"],
            n_mixed_fp_pattern=audit["n_mixed_fp_pattern"],
            conflict="NONE",
        ),
        timing=timing,
    )
    (OUT / "local0_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    print("\n[SUMMARY] " + json.dumps({
        "blocks": [b["block"] for b in boundaries],
        "n_primary": summary["primary_sample"]["n_total"],
        "label": summary["label_distribution_all"],
        "purge": purge,
        "B0": m0, "B1": m1,
        "bootstrap": boot,
        "timing": timing,
    }, indent=2, default=str))
    print(f"[DONE] {timing['total_seconds']}s -> {OUT}")


if __name__ == "__main__":
    main()
