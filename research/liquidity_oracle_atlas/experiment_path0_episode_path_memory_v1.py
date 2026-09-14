"""PATH-0 — Simple episode path memory (PRICE PATH MEMORY only)

===========================================================================
唯一研究问题
===========================================================================
在控制

    current episode start geometry
    previous episode endpoint event（event_mask）
    previous episode duration

之后，上一 **完整** structural episode 的简单连续路径 morphology
是否对下一 episode 的 structural endpoint（4 个 bit）提供
稳定 TB1→TB2 OOS 增量信息？

模型层级（冻结）：

    M0  current geometry
    M1  M0 + prev endpoint one-hot
    M2  M1 + prev duration            <- primary baseline
    M3  M2 + prev path morphology     <- primary test = M3 - M2

===========================================================================
冻结边界
===========================================================================
* 只读 gitignored `episode0_episodes.parquet`；不重跑 / 不修改 EPISODE-0。
  重算 episode identity hash 并硬校验 == REVIEWER_FROZEN_EPISODE_HASH。
* 训练 target：start/end 都在 TB1；测试 target：start/end 都在 TB2。
  TB1→TB2 crossing target 排除。prev episode 允许是 crossing（causal online
  history：prev 在 target start 前已完整结束）。TB3/TB4 完全不触碰。
* 禁止：Path Signature / SMC / BOS/CHoCH/Sweep / liquidity type-scope /
  volume / symbol feature / HMM-PGM / RL / PnL / feature search / tuning。
* 单一 sample unit：真正 contiguous 的 (prev episode, target episode)，
  要求 prev.end_bar == target.start_bar 且两侧 gap == 0。

===========================================================================
输出
===========================================================================
    path0_summary.json
    path0_sample_audit.json
    path0_model_metrics.csv
    path0_bootstrap.csv
    path0_per_bit.csv
    path0_by_symbol.csv
（大型 path0_samples.parquet 存 gitignored cache）
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.liquidity_oracle_atlas.run_5m_graph_probability_v1 import (  # noqa: E402
    load_raw_bars,
)
from research.liquidity_oracle_atlas.experiment_local_liquidity_transition_v0 import (  # noqa: E402
    FULL_UNIV, TRAIN_BLOCK, TEST_BLOCK, OUT, CACHE, build_blocks, block_codes_for,
)

# reviewer 在 930b595c1aa2580d96c4c12444b70c2f0030fb73 上正式冻结的 episode identity
REVIEWER_FROZEN_EPISODE_HASH = (
    "cf840a7191e0265e31442f7b8d8d5ae4173751b0cba15729f95d5c146ab9355c"
)

BIT_NAME = [(0, "UP_PEN", 1), (1, "DOWN_PEN", 2),
            (2, "NEW_UPPER", 4), (3, "NEW_LOWER", 8)]
BIT_NAMES = [n for _, n, _ in BIT_NAME]
BIT_MASKS = np.array([m for _, _, m in BIT_NAME], np.int64)

CUR_NUM = ["cur_up_distance_R", "cur_down_distance_R", "cur_width_R",
           "cur_log_distance_ratio"]
PREV_CAT = ["prev_event_mask"]
M2_EXTRA_NUM = ["prev_duration_bars"]
PATH_NUM = [
    "prev_net_move_R",
    "prev_total_variation_R",
    "prev_signed_efficiency",
    "prev_range_R",
    "prev_max_up_excursion_R",
    "prev_max_down_excursion_R",
    "prev_direction_change_rate",
    "prev_first_half_net_R",
    "prev_second_half_net_R",
    "prev_close_location_in_range",
    "prev_min_upper_high_gap_frac",
    "prev_min_lower_low_gap_frac",
    "prev_upper_touch_rate",
    "prev_lower_touch_rate",
]
assert len(PATH_NUM) == 14

MODELS = {
    "M0_CURRENT_GEOMETRY": (CUR_NUM, []),
    "M1_GEOMETRY_PREV_EVENT": (CUR_NUM, PREV_CAT),
    "M2_GEOMETRY_PREV_EVENT_DURATION": (CUR_NUM + M2_EXTRA_NUM, PREV_CAT),
    "M3_GEOMETRY_PREV_EVENT_DURATION_PATH": (
        CUR_NUM + M2_EXTRA_NUM + PATH_NUM, PREV_CAT),
}
M0, M1, M2, M3 = MODELS.keys()
COMPARISONS = [
    (M3, M2, "PRIMARY_path_morphology_given_event_duration"),
    (M1, M0, "SECONDARY_prev_endpoint_given_geometry"),
    (M2, M1, "SECONDARY_duration_given_endpoint"),
    (M3, M0, "SECONDARY_total_prev_history"),
]

BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 20260914
N_PROB_BINS = 10


# ===========================================================================
# frozen episode identity
# ===========================================================================
def episode_identity_hash(ep: pd.DataFrame) -> str:
    key = (ep["symbol"].astype(str) + "|" + ep["start_bar"].astype(str) + "|"
           + ep["end_bar"].astype(str) + "|" + ep["event_mask"].astype(str))
    h = hashlib.sha256()
    h.update("\n".join(sorted(key.tolist())).encode())
    return h.hexdigest()


# ===========================================================================
# path morphology（只用 prev.start_bar ... prev.end_bar）
# ===========================================================================
def path_morphology(h, l, c, A, U, D, s: int, e: int) -> dict:
    d = int(e - s)
    hi = h[s + 1:e + 1]
    lo = l[s + 1:e + 1]
    delta = c[s + 1:e + 1] - c[s:e]
    net = float(c[e] - c[s])
    tv = float(np.abs(delta).sum())

    path_high = float(max(c[s], hi.max()))
    path_low = float(min(c[s], lo.min()))
    rng = path_high - path_low

    nz = delta[delta != 0.0]
    if len(nz) < 2:
        dcr = 0.0
    else:
        sg = np.sign(nz)
        dcr = float((sg[1:] != sg[:-1]).sum()) / float(len(nz) - 1)

    k = (d + 1) // 2                     # ceil(d/2)
    mid = s + k
    first = float(c[mid] - c[s])
    second = float(c[e] - c[mid])

    W = float(U - D)
    if W > 0.0:
        gap_up = float(np.min((U - hi) / W))
        gap_dn = float(np.min((lo - D) / W))
    else:
        gap_up = np.nan
        gap_dn = np.nan

    return dict(
        prev_net_move_R=net / A,
        prev_total_variation_R=tv / A,
        prev_signed_efficiency=net / max(tv, 1e-12),
        prev_range_R=rng / A,
        prev_max_up_excursion_R=(float(hi.max()) - float(c[s])) / A,
        prev_max_down_excursion_R=(float(c[s]) - float(lo.min())) / A,
        prev_direction_change_rate=dcr,
        prev_first_half_net_R=first / A,
        prev_second_half_net_R=second / A,
        prev_close_location_in_range=(
            0.5 if rng <= 0.0 else float((c[e] - path_low) / rng)),
        prev_min_upper_high_gap_frac=gap_up,
        prev_min_lower_low_gap_frac=gap_dn,
        prev_upper_touch_rate=float((hi == U).sum()) / d,
        prev_lower_touch_rate=float((lo == D).sum()) / d,
    )


# ===========================================================================
# metrics
# ===========================================================================
def binary_logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def binary_brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def ece_binary(y: np.ndarray, p: np.ndarray, n_bins: int = N_PROB_BINS) -> float:
    """固定 [0,.1),...,[.9,1.0] bins（不 qcut）。"""
    b = np.clip((p * n_bins).astype(np.int64), 0, n_bins - 1)
    e = 0.0
    for i in range(n_bins):
        m = b == i
        if m.any():
            e += float(m.mean()) * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(e)


# ===========================================================================
# model
# ===========================================================================
def make_pipeline(num_cols, cat_cols) -> Pipeline:
    parts = [("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ]), num_cols)]
    if cat_cols:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    pre = ColumnTransformer(parts)
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                   max_iter=3000, class_weight=None)),
    ])


# ===========================================================================
# main
# ===========================================================================
def main():
    t_total = time.perf_counter()
    timing = {}
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    # ---------------------------------------------------------------- cache
    t0 = time.perf_counter()
    ep_path = CACHE / "episode0_episodes.parquet"
    ep = pd.read_parquet(ep_path)
    ep_hash = episode_identity_hash(ep)
    print(f"[EPISODE HASH] {ep_hash}")
    if ep_hash != REVIEWER_FROZEN_EPISODE_HASH:
        raise SystemExit(
            "STOP_PATH0_EPISODE_HASH_MISMATCH: "
            f"{ep_hash} != {REVIEWER_FROZEN_EPISODE_HASH}")

    bars_by_sym = {s: load_raw_bars(s) for s in FULL_UNIV}
    all_days, day_block_code, boundaries = build_blocks(bars_by_sym)
    tb2_start_ns = int(pd.Timestamp(
        str(boundaries[1]["first_day"])).value)
    timing["cache_load_seconds"] = round(time.perf_counter() - t0, 2)

    n_episodes_total = int(len(ep))
    assert n_episodes_total == 25273, n_episodes_total

    # ------------------------------------------------------ contiguous pairs
    t0 = time.perf_counter()
    ep = ep.sort_values(["symbol", "start_bar"]).reset_index(drop=True)
    rows = []
    for sym, g in ep.groupby("symbol", sort=False):
        g = g.reset_index(drop=True)
        sb = g["start_bar"].to_numpy(np.int64)
        ebi = g["end_bar"].to_numpy(np.int64)
        gaf = g["gap_bars_after_episode"].to_numpy(np.int64)
        gbf = g["gap_bars_before_episode"].to_numpy(np.int64)
        ok = ((ebi[:-1] == sb[1:]) & (gaf[:-1] == 0) & (gbf[1:] == 0))
        pi = np.flatnonzero(ok)
        if not len(pi):
            continue
        ti = pi + 1
        bars = bars_by_sym[sym]
        h = np.asarray(bars["h"], float)
        l = np.asarray(bars["l"], float)
        c = np.asarray(bars["c"], float)
        atr = np.asarray(bars["atr"], float)
        for a, b in zip(pi, ti):
            p = g.loc[a]
            tt = g.loc[b]
            rows.append(dict(
                symbol=sym,
                prev_start_bar=int(p["start_bar"]),
                prev_end_bar=int(p["end_bar"]),
                prev_start_block=str(p["start_block"]),
                prev_end_block=str(p["end_block"]),
                prev_event_mask=int(p["event_mask"]),
                prev_duration_bars=int(p["duration_bars"]),
                prev_start_time=p["start_time"],
                prev_end_time=p["end_time"],
                target_start_bar=int(tt["start_bar"]),
                target_end_bar=int(tt["end_bar"]),
                target_start_block=str(tt["start_block"]),
                target_end_block=str(tt["end_block"]),
                target_event_mask=int(tt["event_mask"]),
                target_start_time=tt["start_time"],
                target_end_time=tt["end_time"],
                target_prev_crosses=bool(p["crosses_tb1_tb2"]),
                tmp_atr_prev=float(atr[int(p["start_bar"])]),
                tmp_atr_cur=float(atr[int(tt["start_bar"])]),
                tmp_cur_up=float(tt["start_upper_price"]),
                tmp_cur_dn=float(tt["start_lower_price"]),
                tmp_cur_close=float(tt["start_close"]),
                tmp_prev_up=float(p["start_upper_price"]),
                tmp_prev_dn=float(p["start_lower_price"]),
            ))
    sm = pd.DataFrame(rows)
    timing["sample_pair_seconds"] = round(time.perf_counter() - t0, 2)
    n_contiguous_pairs = int(len(sm))

    # ------------------------------------------------------------ exclusion
    n_target_censor = int((sm["target_event_mask"] == 0).sum())
    raw_train_blk = (sm["target_start_block"] == TRAIN_BLOCK) & \
        (sm["target_end_block"] == TRAIN_BLOCK)
    raw_test_blk = (sm["target_start_block"] == TEST_BLOCK) & \
        (sm["target_end_block"] == TEST_BLOCK)
    n_train_candidates = int(raw_train_blk.sum())
    n_test_candidates = int(raw_test_blk.sum())
    train_blk, test_blk = raw_train_blk, raw_test_blk
    n_cross_block = int((~(train_blk | test_blk)).sum())
    sm = sm[sm["target_event_mask"] != 0].copy()
    train_blk = (sm["target_start_block"] == TRAIN_BLOCK) & \
        (sm["target_end_block"] == TRAIN_BLOCK)
    test_blk = (sm["target_start_block"] == TEST_BLOCK) & \
        (sm["target_end_block"] == TEST_BLOCK)
    sm = sm[train_blk | test_blk].copy()
    n_after_censor_and_block = int(len(sm))

    ok_prev_atr = (sm["tmp_atr_prev"] > 0.0) & np.isfinite(sm["tmp_atr_prev"])
    ok_cur_atr = (sm["tmp_atr_cur"] > 0.0) & np.isfinite(sm["tmp_atr_cur"])
    n_bad_prev_atr = int((~ok_prev_atr).sum())
    n_bad_cur_atr = int((~ok_cur_atr).sum())
    sm = sm[ok_prev_atr & ok_cur_atr].copy()
    sm = sm.reset_index(drop=True)

    # ------------------------------------------------------------- features
    t0 = time.perf_counter()
    cur_up = (sm["tmp_cur_up"] - sm["tmp_cur_close"]) / sm["tmp_atr_cur"]
    cur_dn = (sm["tmp_cur_close"] - sm["tmp_cur_dn"]) / sm["tmp_atr_cur"]
    sm["cur_up_distance_R"] = cur_up
    sm["cur_down_distance_R"] = cur_dn
    sm["cur_width_R"] = cur_up + cur_dn
    sm["cur_log_distance_ratio"] = np.log(
        (cur_up + 1e-8) / (cur_dn + 1e-8))

    n_path_increments = 0
    path_rows = []
    max_src = []
    for sym, g in sm.groupby("symbol", sort=False):
        bars = bars_by_sym[sym]
        h = np.asarray(bars["h"], float)
        l = np.asarray(bars["l"], float)
        c = np.asarray(bars["c"], float)
        atr = np.asarray(bars["atr"], float)
        for r in g.itertuples():
            pf = path_morphology(h, l, c, atr[r.prev_start_bar],
                                 r.tmp_prev_up, r.tmp_prev_dn,
                                 r.prev_start_bar, r.prev_end_bar)
            pf["_row"] = r.Index
            path_rows.append(pf)
            n_path_increments += int(r.prev_end_bar - r.prev_start_bar)
            max_src.append(int(r.prev_end_bar) - int(r.target_start_bar))
    pdf = pd.DataFrame(path_rows).set_index("_row")
    sm = sm.join(pdf)
    max_src = np.asarray(max_src, np.int64)
    timing["path_feature_seconds"] = round(time.perf_counter() - t0, 2)

    # ------------------------------------------------------- causal guards
    if int((sm["prev_end_bar"] != sm["target_start_bar"]).sum()):
        raise SystemExit("STOP_PATH0_PAIR_NOT_CONTIGUOUS")
    if int((sm["prev_end_time"] != sm["target_start_time"]).sum()):
        raise SystemExit("STOP_PATH0_PAIR_TIME_GAP")
    if int((sm["prev_event_mask"] == 0).sum()):
        raise SystemExit("STOP_PATH0_PREV_CENSOR_CONTIGUOUS")
    if int(max_src.max()) != 0:
        raise SystemExit(
            f"STOP_PATH0_PATH_FEATURE_FUTURE: max_src={int(max_src.max())}")

    # --------------------------------------------------------- train / test
    tr = sm[(sm["target_start_block"] == TRAIN_BLOCK)
            & (sm["target_end_block"] == TRAIN_BLOCK)].reset_index(drop=True)
    te = sm[(sm["target_start_block"] == TEST_BLOCK)
            & (sm["target_end_block"] == TEST_BLOCK)].reset_index(drop=True)
    if not len(tr) or not len(te):
        raise SystemExit("STOP_PATH0_EMPTY_SPLIT")

    tr_end_ns = pd.DatetimeIndex(tr["target_end_time"]).asi8
    if not bool((tr_end_ns < tb2_start_ns).all()):
        raise SystemExit("STOP_PATH0_TRAIN_OUTCOME_LEAK")
    n_test_prev_crosses = int(te["target_prev_crosses"].sum())

    y_tr = np.stack([((tr["target_event_mask"].to_numpy() & m) != 0)
                     .astype(np.int64) for m in BIT_MASKS], axis=1)
    y_te = np.stack([((te["target_event_mask"].to_numpy() & m) != 0)
                     .astype(np.int64) for m in BIT_MASKS], axis=1)
    for k, nm in enumerate(BIT_NAMES):
        if len(np.unique(y_tr[:, k])) < 2:
            raise SystemExit(f"STOP_PATH0_TRAIN_BIT_ABSENT: {nm}")

    td_of = {s: np.asarray(bars_by_sym[s]["td"]).astype("datetime64[D]")
             for s in FULL_UNIV}
    te_day = np.array([td_of[s][b] for s, b in
                       zip(te["symbol"], te["target_start_bar"])])

    # ---------------------------------------------------------------- fit
    t0 = time.perf_counter()
    P_tr = np.zeros((len(tr), 4))
    P_te = {name: np.zeros((len(te), 4)) for name in MODELS}
    coef_info = {}
    for name, (num_cols, cat_cols) in MODELS.items():
        cols = num_cols + cat_cols
        for k, nm in enumerate(BIT_NAMES):
            pipe = make_pipeline(num_cols, cat_cols)
            pipe.fit(tr[cols], y_tr[:, k])
            P_te[name][:, k] = pipe.predict_proba(te[cols])[:, 1]
        coef_info[name] = len(cols)
    timing["fit_seconds"] = round(time.perf_counter() - t0, 2)

    prior = y_tr.mean(axis=0)
    P_prior = np.tile(prior, (len(te), 1))
    for k in range(4):
        P_tr[:, k] = prior[k]

    # ------------------------------------------------------------ metrics
    t0 = time.perf_counter()
    metric_rows = []
    for name in list(MODELS) + ["B_PRIOR"]:
        P = P_prior if name == "B_PRIOR" else P_te[name]
        ll = [binary_logloss(y_te[:, k], P[:, k]) for k in range(4)]
        br = [binary_brier(y_te[:, k], P[:, k]) for k in range(4)]
        ec = [ece_binary(y_te[:, k], P[:, k]) for k in range(4)]
        for k, nm in enumerate(BIT_NAMES):
            metric_rows.append(dict(
                model=name, bit=nm, n=len(te),
                prevalence=float(y_te[:, k].mean()),
                logloss=ll[k], brier=br[k], ece=ec[k],
                train_prevalence=float(prior[k])))
        metric_rows.append(dict(
            model=name, bit="MEAN_BIT", n=len(te),
            prevalence=float(y_te.mean()),
            logloss=float(np.mean(ll)), brier=float(np.mean(br)),
            ece=float(np.mean(ec)), train_prevalence=float(prior.mean())))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "path0_model_metrics.csv", index=False)
    mtab = metrics[metrics["bit"] == "MEAN_BIT"].set_index("model")
    timing["metric_seconds"] = round(time.perf_counter() - t0, 2)

    # per-sample mean-bit BCE（不作为独立样本扩大 n）
    def sample_loss(P):
        out = np.zeros(len(y_te))
        for k in range(4):
            q = np.clip(P[:, k], 1e-15, 1.0 - 1e-15)
            out += -(y_te[:, k] * np.log(q)
                     + (1.0 - y_te[:, k]) * np.log1p(-q))
        return out / 4.0

    losses = {name: sample_loss(P_te[name]) for name in MODELS}
    uniq_days = np.unique(te_day)
    day_pos = np.searchsorted(uniq_days, te_day)
    nd = len(uniq_days)
    day_cnt = np.bincount(day_pos, minlength=nd)

    def day_stats(loss):
        s = np.bincount(day_pos, weights=loss, minlength=nd)
        keep = day_cnt > 0
        return s[keep] / day_cnt[keep], keep

    def boot_ci(dv, nk):
        boot = np.empty(BOOTSTRAP_REPS)
        for b in range(BOOTSTRAP_REPS):
            boot[b] = dv[rng.integers(0, nk, nk)].mean()
        return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    boot_rows = []
    for hi_m, lo_m, label in COMPARISONS:
        da, keep = day_stats(losses[hi_m] - losses[lo_m])
        nk = int(keep.sum())
        lo, hi = boot_ci(da, nk)
        if hi < 0:
            verdict = "CI_below_zero"
        elif lo > 0:
            verdict = "CI_above_zero"
        else:
            verdict = "CI_contains_zero"
        boot_rows.append(dict(
            comparison=f"{hi_m} - {lo_m}", role=label,
            mean_bit_logloss_A=float(mtab.loc[hi_m, "logloss"]),
            mean_bit_logloss_B=float(mtab.loc[lo_m, "logloss"]),
            delta_sample_weighted=float((losses[hi_m] - losses[lo_m]).mean()),
            delta_daily_mean=float(da.mean()), n_days=nk,
            ci_lo=lo, ci_hi=hi, verdict=verdict))
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(OUT / "path0_bootstrap.csv", index=False)
    layer_audit = {r["comparison"]: dict(
        role=r["role"], delta_daily_mean=r["delta_daily_mean"],
        ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], verdict=r["verdict"])
        for r in boot_rows}

    # ------------------------------------------- per bit（secondary 理解用）
    per_bit = []
    keep_days = day_cnt > 0
    nk = int(keep_days.sum())
    for k, nm in enumerate(BIT_NAMES):
        def bce(P):
            q = np.clip(P[:, k], 1e-15, 1.0 - 1e-15)
            return -(y_te[:, k] * np.log(q)
                     + (1.0 - y_te[:, k]) * np.log1p(-q))
        dd = bce(P_te[M3]) - bce(P_te[M2])
        dv = (np.bincount(day_pos, weights=dd, minlength=nd)[keep_days]
              / day_cnt[keep_days])
        lo, hi = boot_ci(dv, nk)
        per_bit.append(dict(
            bit=nm, n=len(te), prevalence=float(y_te[:, k].mean()),
            logloss_M2=binary_logloss(y_te[:, k], P_te[M2][:, k]),
            logloss_M3=binary_logloss(y_te[:, k], P_te[M3][:, k]),
            delta_M3_minus_M2=float(dd.mean()),
            delta_daily_mean=float(dv.mean()),
            ci_lo=lo, ci_hi=hi))
    per_bit_df = pd.DataFrame(per_bit)
    per_bit_df.to_csv(OUT / "path0_per_bit.csv", index=False)

    # ----------------------------------------------------------- by symbol
    sym_rows = []
    for s, g in te.groupby("symbol"):
        idx = g.index.to_numpy()
        sym_rows.append(dict(
            symbol=s, n=int(len(g)),
            mean_bit_logloss_M2=float(losses[M2][idx].mean()),
            mean_bit_logloss_M3=float(losses[M3][idx].mean()),
            delta=float((losses[M3][idx] - losses[M2][idx]).mean())))
    sym_df = pd.DataFrame(sym_rows).sort_values("delta").reset_index(drop=True)
    sym_df.to_csv(OUT / "path0_by_symbol.csv", index=False)

    timing["bootstrap_seconds"] = round(time.perf_counter() - t0, 2)

    # -------------------------------------------------------------- hashes
    key = (sm["symbol"].astype(str) + "|" + sm["prev_start_bar"].astype(str)
           + "|" + sm["prev_end_bar"].astype(str) + "|"
           + sm["target_start_bar"].astype(str) + "|"
           + sm["target_end_bar"].astype(str) + "|"
           + sm["target_event_mask"].astype(str))
    hh = hashlib.sha256()
    hh.update("\n".join(sorted(key.tolist())).encode())
    sample_key_sha256 = hh.hexdigest()

    sm.to_parquet(CACHE / "path0_samples.parquet", index=False)

    # ------------------------------------------------------------- outputs
    audit = dict(
        n_episodes_total=n_episodes_total,
        n_contiguous_episode_pairs=n_contiguous_pairs,
        train_candidates=n_train_candidates,
        test_candidates=n_test_candidates,
        n_after_censor_and_block_exclusion=n_after_censor_and_block,
        excluded_target_censor=n_target_censor,
        excluded_cross_block_target=n_cross_block,
        excluded_invalid_current_atr=n_bad_cur_atr,
        excluded_invalid_prev_atr=n_bad_prev_atr,
        n_train_final=int(len(tr)),
        n_test_final=int(len(te)),
        n_test_prev_crosses_tb1_tb2=n_test_prev_crosses,
        n_unique_sample_keys=int(key.nunique()),
        n_unique_episode_pairs=int(len(sm)),
        exclusion_priority=["cross_block_target", "target_censor",
                            "invalid_atr"],
    )
    (OUT / "path0_sample_audit.json").write_text(
        json.dumps(audit, indent=2, default=str))

    primary = boot_df.iloc[0]
    if primary["ci_hi"] < 0:
        primary_verdict = "PATH_MORPHOLOGY_INCREMENT_SUPPORTED"
    elif primary["ci_lo"] > 0:
        primary_verdict = "PATH_MORPHOLOGY_INCREMENT_NEGATIVE"
    else:
        primary_verdict = "PATH_MORPHOLOGY_INCREMENT_AMBIGUOUS"

    summary = dict(
        experiment="PATH-0 simple episode path memory",
        base="930b595c1aa2580d96c4c12444b70c2f0030fb73",
        frozen_episode_hash=REVIEWER_FROZEN_EPISODE_HASH,
        recomputed_episode_hash=ep_hash,
        primary_question=(
            "does previous completed episode simple price-path morphology "
            "add stable TB1->TB2 OOS information about the next episode "
            "structural endpoint bits, given current geometry, previous "
            "endpoint event and previous duration?"),
        model_layers={
            M0: ["current geometry"],
            M1: ["+ prev_event_mask one-hot"],
            M2: ["+ prev_duration_bars"],
            M3: ["+ 14 prev path morphology features"],
        },
        feature_names={k: dict(numeric=v[0], categorical=v[1])
                       for k, v in MODELS.items()},
        n_features=coef_info,
        labels="4 independent bits (UP_PEN, DOWN_PEN, NEW_UPPER, "
               "NEW_LOWER); same-bar multi-event kept, no bullish/bearish "
               "mapping, no multiclass collapse",
        blocks=dict(boundaries=boundaries, tb2_start=str(boundaries[1]["first_day"]),
                    tb3_tb4_used_for_feature_label_fit_metric_bootstrap=False),
        sample_audit=audit,
        metrics=metrics_to_dict(mtab),
        bootstrap=boot_rows,
        primary_comparison=dict(
            comparison=f"{M3} - {M2}",
            delta_daily_mean=float(primary["delta_daily_mean"]),
            delta_sample_weighted=float(primary["delta_sample_weighted"]),
            ci_lo=float(primary["ci_lo"]), ci_hi=float(primary["ci_hi"]),
            n_days=int(primary["n_days"]),
            bootstrap_reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED,
            verdict=primary_verdict),
        layer_audit=layer_audit,
        per_bit_M3_minus_M2=per_bit,
        by_symbol=dict(
            n_symbol=len(sym_df),
            n_delta_negative=int((sym_df["delta"] < 0).sum()),
            n_delta_positive=int((sym_df["delta"] > 0).sum())),
        path0_sample_key_sha256=sample_key_sha256,
        n_path_increments_scanned=int(n_path_increments),
        timing=timing,
        calibration_note=(
            "ECE reported only. Even if logloss improves, the model output "
            "must NOT be called a calibrated belief; calibration is a "
            "separate future experiment."),
        interpretation_limits=(
            "PATH-0 only tests whether simple price-path morphology of the "
            "previous completed episode adds OOS information. It does NOT "
            "establish SMC validity, path-signature value, Semi-Markov "
            "structure, latent psychological state, tradability or RL."),
    )
    summary["PATH0_VERDICT"] = primary_verdict
    timing["total_seconds"] = round(time.perf_counter() - t_total, 2)
    summary["timing"] = timing
    (OUT / "path0_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    print(f"[AUDIT] {audit}")
    print(f"[METRICS mean-bit]\n{metrics_table(mtab).to_string()}")
    print(f"[BOOTSTRAP]\n{boot_df.to_string(index=False)}")
    print(f"[PER-BIT M3-M2]\n{per_bit_df.to_string(index=False)}")
    print(f"[BY-SYMBOL] neg={int((sym_df['delta'] < 0).sum())} "
          f"pos={int((sym_df['delta'] > 0).sum())}")
    print(f"[KEY] {sample_key_sha256}")
    print(f"[PATH SCAN] n_path_increments_scanned={int(n_path_increments)}")
    print(f"[TIMING] {timing}")
    print(f"[VERDICT] {primary_verdict}")
    print(f"[DONE] -> {OUT}")


def metrics_to_dict(mtab: pd.DataFrame) -> dict:
    cols = ["n", "prevalence", "logloss", "brier", "ece", "train_prevalence"]
    return {k: {c: float(v[c]) for c in cols}
            for k, v in mtab.to_dict(orient="index").items()}


def metrics_table(mtab: pd.DataFrame) -> pd.DataFrame:
    return mtab.reset_index()[
        ["model", "logloss", "brier", "ece", "prevalence"]]


if __name__ == "__main__":
    main()
